"""Fixed-point MLP inference for learned controllers that must become RTL (07 P1-22, with the 07 P1-21 toolkit).

07 P1-22 asks for "a fixed-point MLP inference helper in fxp": "A policy that cannot be quantized is not an FPGA
controller." This module is that helper. It builds on the scalar kernels of :mod:`mrm.control.fxp` (``fx_cast``,
``fx_mul``, ``fx_add``, ``fx_round_shift``, ``fx_overflow``; nothing there is changed) and adds: the quantizer that
chooses every format, the packed int64 form that a compiled controller can carry in its parameter row, a numba
evaluation that allocates nothing and makes no array views (so a compiled controller's ``step``, which the engine's
runtime-free tick loop calls, may use it: docs/ENGINE.md section 7), a rigorous float-vs-fixed error bound, and
resource counts and ``$readmemh`` images for an FPGA. :meth:`mrm.sim.gym_env.MRMEnv.compiled_policy` runs a quantized
policy on the engine as a CompiledController.

Network (PyTorch layout, ``W`` is (out, in))::

    y = a_L(W_L a_{L-1}( ... a_1(W_1 x + b_1) ... ) + b_L),   a_l in {linear, relu, hardtanh}

hardtanh is clip(z, -1, 1). All three activations are 1-Lipschitz, which the error bound uses.

Fixed-point semantics of one layer (raw int64 values; f_x, f_w, f_b, f_out fraction bits; f_acc = f_w + f_x)::

    acc = cast(b, f_b -> f_acc) + sum_i W[o, i] x[i]     exact products; an acc_bits-wide accumulator fitted by the
                                                         overflow mode after every addition (the fx_matvec rule)
    z   = round(acc, f_acc -> f_out)                     one rounding (rounding mode), at accumulator width
    y   = fit(act(z), out_bits)                          relu: max(z, 0); hardtanh: clip(z, -2^f_out, 2^f_out);
                                                         then one fit into the signed output format

The activation acts on the rounded wide value before the output fit (as an RTL datapath would: compare at the
accumulator width, then narrow). This differs from ``fx.fx_matvec`` (fit, then relu) only under WRAP overflow of the
output, where fitting first can turn a large negative value positive. hardtanh never depends on the overflow mode:
with at least one integer bit the clipped value always fits; a Q0.n output clips its upper end at 1 - 2^-n.

:func:`quantize_mlp` chooses every format: weights at ``w_bits`` with the fewest integer bits that hold max|W|;
biases at the accumulator's fraction bits (f_b = f_acc, so the bias enters the accumulator without rounding);
hidden activations at ``act_bits`` with the integer bits of the float network's interval over the input box
(times ``margin``); hardtanh outputs with one integer bit (+-1 exact); and the accumulator width from the exact
worst case over EVERY raw input of the input format (Python integers: sum_i |W_raw| max|x_raw| + |b_raw|). An
accumulator that wide can never overflow, whatever the input, so the evaluation may sum in plain int64 (the
``proven`` fast path, equal bit for bit to the per-addition fitted path, which runs when the accumulator was set
narrower by hand).

:func:`error_bound` bounds |y_fixed(x_q) - y_float(x)| for every x in a box, with x_q the input quantized to the
input format (or x itself when it is exactly representable, ``input_exact=True``: the gym environment's
observations are). Per layer, with X_i = max |x_i| of the float layer input over the box (interval arithmetic),
e_i the bound on its input error and W_q, b_q the quantized parameters::

    |z_fixed - z_float|_o <= sum_i |W_q - W|_oi X_i + sum_i |W_q|_oi e_i + |b_q - b|_o + r_out
    e_out = that (activations are 1-Lipschitz; + 2^-f_out for a Q0.n hardtanh at +1)

with r_out = 2^-(f_out + 1) for the round-half modes and 2^-f_out for floor / toward-zero. The bound holds when no
output saturates; the result reports ``saturation_free`` from the same intervals (derived here: a standard
interval / Lipschitz propagation, Ch17 asks for float-vs-fixed equivalence "within bounds" without a formula).

Packed form (int64 vector; offsets relative to its first element)::

    header (H_SIZE)   [magic, n_layers, n_in, n_out, in_bits, in_frac, rounding, overflow, max_width, proven,
                       size, version]
    per layer (L_SIZE) [n_out, n_in, w_bits, w_frac, b_frac, acc_bits, out_bits, out_frac, act, w_off, b_off, 0]
    data              weights row-major, then biases, layer after layer

Everything runs under NUMBA_DISABLE_JIT=1 with the same results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numba as nb
import numpy as np

from mrm.control.fxp import (MAX_BITS, Overflow, QFormat, Rounding, fx_add, fx_cast, fx_mul, fx_overflow,
                             fx_round_shift, quantize)

__all__ = [
    "ACT_LINEAR", "ACT_RELU", "ACT_HARDTANH", "ACTIVATIONS", "FxpLayer", "FxpMlp", "MlpErrorBound",
    "quantize_mlp", "mlp_forward_float", "error_bound", "fxp_mlp_eval", "fxp_mlp_forward_batch",
    "fxp_mlp_check", "packed_n_in", "packed_n_out", "packed_max_width", "packed_out_frac", "packed_in_frac",
    "H_SIZE", "L_SIZE", "MAGIC", "VERSION", "memh_lines",
]

ACT_LINEAR = 0
ACT_RELU = 1
ACT_HARDTANH = 2
ACTIVATIONS = {"linear": ACT_LINEAR, "identity": ACT_LINEAR, "relu": ACT_RELU, "hardtanh": ACT_HARDTANH}
_ACT_NAMES = {ACT_LINEAR: "linear", ACT_RELU: "relu", ACT_HARDTANH: "hardtanh"}

MAGIC = 0x4D4C50          # 'MLP'
VERSION = 1
# header
(H_MAGIC, H_N_LAYERS, H_N_IN, H_N_OUT, H_IN_BITS, H_IN_FRAC, H_RND, H_OVF, H_MAX_WIDTH, H_PROVEN, H_SIZE_,
 H_VER) = range(12)
H_SIZE = 12
# layer record
(L_N_OUT, L_N_IN, L_W_BITS, L_W_FRAC, L_B_FRAC, L_ACC_BITS, L_OUT_BITS, L_OUT_FRAC, L_ACT, L_W_OFF, L_B_OFF,
 L_RES) = range(12)
L_SIZE = 12

_SAT = int(Overflow.SATURATE)
_CHECKED = int(Overflow.CHECKED)
_HALF_MODES = (int(Rounding.HALF_UP), int(Rounding.HALF_EVEN), int(Rounding.HALF_AWAY))


# =============================================================================================== numba kernels
@nb.njit(cache=True)
def _activate(acc, f_acc, out_bits, out_frac, act, rnd, ovf):
    """Round the accumulator to f_out, apply the activation at that width, fit into the output format."""
    z = fx_round_shift(acc, f_acc - out_frac, rnd)
    if act == ACT_HARDTANH:
        one = np.int64(1) << out_frac
        if z > one:
            z = one
        elif z < -one:
            z = -one
        return fx_overflow(z, out_bits, True, _SAT)      # only a Q0.n output can clip here (+1 -> 1 - lsb)
    if act == ACT_RELU and z < 0:
        z = np.int64(0)
    return fx_overflow(z, out_bits, True, ovf)


@nb.njit(cache=True)
def _dense(p, w0, b0, n_out, n_in, w_frac, b_frac, x, x0, x_frac, acc_bits, out_bits, out_frac, act, rnd, ovf,
           proven, y, y0):
    """One layer from the packed vector p (weights at p[w0:], biases at p[b0:]) on x[x0:x0+n_in] into y[y0:].
    Index arithmetic only (no views, no allocation)."""
    f_acc = w_frac + x_frac
    for o in range(n_out):
        acc = fx_cast(p[b0 + o], b_frac, acc_bits, f_acc, True, rnd, ovf)
        if proven != 0:
            # |every partial sum| < 2^(acc_bits - 1) for every raw input (quantize_mlp proof): plain int64 is exact
            s = np.int64(0)
            for i in range(n_in):
                s += p[w0 + o * n_in + i] * x[x0 + i]
            acc = acc + s
        else:
            for i in range(n_in):
                prod = fx_mul(p[w0 + o * n_in + i], w_frac, x[x0 + i], x_frac, acc_bits, f_acc, True, rnd, ovf)
                acc = fx_add(acc, prod, acc_bits, True, ovf)
        y[y0 + o] = _activate(acc, f_acc, out_bits, out_frac, act, rnd, ovf)


@nb.njit(cache=True)
def fxp_mlp_eval(p, base, buf, x_off, y_off, s_off):
    """Evaluate the packed network p[base:] (module docstring) on buf[x_off : x_off + n_in] (raw at the input
    format) and write its n_out raw outputs (at the last layer's output format) to buf[y_off:]. Scratch:
    buf[s_off : s_off + 2 * max_width], which must not overlap the input or the output. Allocates nothing and makes
    no views, so a compiled controller may call it from the engine's tick loop. Returns n_out."""
    n_layers = p[base + H_N_LAYERS]
    rnd = p[base + H_RND]
    ovf = p[base + H_OVF]
    proven = p[base + H_PROVEN]
    mw = p[base + H_MAX_WIDTH]
    src = x_off
    x_frac = p[base + H_IN_FRAC]
    for l in range(n_layers):
        r = base + H_SIZE + l * L_SIZE
        dst = s_off + (l % 2) * mw
        _dense(p, base + p[r + L_W_OFF], base + p[r + L_B_OFF], p[r + L_N_OUT], p[r + L_N_IN], p[r + L_W_FRAC],
               p[r + L_B_FRAC], buf, src, x_frac, p[r + L_ACC_BITS], p[r + L_OUT_BITS], p[r + L_OUT_FRAC],
               p[r + L_ACT], rnd, ovf, proven, buf, dst)
        src = dst
        x_frac = p[r + L_OUT_FRAC]
    n_out = p[base + H_N_OUT]
    for o in range(n_out):
        buf[y_off + o] = buf[src + o]
    return n_out


@nb.njit(cache=True)
def fxp_mlp_forward_batch(p, x, y):
    """Rows of x (B, n_in) int64 raw inputs -> rows of y (B, n_out) int64 raw outputs."""
    n_in = p[H_N_IN]
    n_out = p[H_N_OUT]
    mw = p[H_MAX_WIDTH]
    buf = np.zeros(n_in + n_out + 2 * mw, np.int64)
    for b in range(x.shape[0]):
        for i in range(n_in):
            buf[i] = x[b, i]
        fxp_mlp_eval(p, 0, buf, 0, n_in, n_in + n_out)
        for o in range(n_out):
            y[b, o] = buf[n_in + o]


@nb.njit(cache=True)
def fxp_mlp_check(p, base):
    """Structural check of a packed network; 0 if valid, else an error code > 0: 1 too short, 2 magic or version,
    3 size, 4 layer count, 5 layer widths, 6 data offsets, 7 accumulator or output width, 8 output width of the
    network, 9 rounding or overflow mode, 10 activation, 11 fraction bits (every shift of :func:`fxp_mlp_eval` in
    0..62 and the output fraction bits <= the accumulator's), 12 input format, 13 a weight times the largest raw
    input of its layer reaches 2^62 (fx_mul's limit). A network that passes evaluates without an exception in the
    WRAP and SATURATE modes (the compiled MLP policy checks it at init, so no exception can leave the engine's tick
    loop); every network of :func:`quantize_mlp` passes."""
    if p.shape[0] - base < H_SIZE:
        return 1
    if p[base + H_MAGIC] != MAGIC or p[base + H_VER] != VERSION:
        return 2
    size = p[base + H_SIZE_]
    if size < H_SIZE or base + size > p.shape[0]:
        return 3
    n_layers = p[base + H_N_LAYERS]
    if n_layers < 1 or H_SIZE + n_layers * L_SIZE > size:
        return 4
    if p[base + H_RND] < 0 or p[base + H_RND] > 4 or p[base + H_OVF] < 0 or p[base + H_OVF] > 2:
        return 9
    x_frac = p[base + H_IN_FRAC]
    x_bits = p[base + H_IN_BITS]
    if x_bits < 2 or x_bits > MAX_BITS or x_frac < 0:
        return 12
    width = p[base + H_N_IN]
    for l in range(n_layers):
        r = base + H_SIZE + l * L_SIZE
        if p[r + L_N_IN] != width or p[r + L_N_OUT] < 1 or p[r + L_N_OUT] > p[base + H_MAX_WIDTH]:
            return 5
        if (p[r + L_W_OFF] < H_SIZE or p[r + L_B_OFF] < H_SIZE
                or p[r + L_W_OFF] + p[r + L_N_OUT] * p[r + L_N_IN] > size or p[r + L_B_OFF] + p[r + L_N_OUT] > size):
            return 6
        if (p[r + L_ACC_BITS] < 2 or p[r + L_ACC_BITS] > MAX_BITS or p[r + L_OUT_BITS] < 2
                or p[r + L_OUT_BITS] > MAX_BITS):
            return 7
        if p[r + L_ACT] < ACT_LINEAR or p[r + L_ACT] > ACT_HARDTANH:
            return 10
        f_acc = p[r + L_W_FRAC] + x_frac
        if (p[r + L_W_FRAC] < 0 or p[r + L_B_FRAC] < 0 or p[r + L_OUT_FRAC] < 0 or f_acc > MAX_BITS
                or p[r + L_OUT_FRAC] > f_acc or abs(p[r + L_B_FRAC] - f_acc) > MAX_BITS):
            return 11
        x_max = np.int64(1) << (x_bits - 1)              # largest |raw input| of this layer
        w_lim = ((np.int64(1) << MAX_BITS) - 1) // x_max
        w0 = base + p[r + L_W_OFF]
        for j in range(p[r + L_N_OUT] * p[r + L_N_IN]):
            if abs(p[w0 + j]) > w_lim:
                return 13
        width = p[r + L_N_OUT]
        x_frac = p[r + L_OUT_FRAC]
        x_bits = p[r + L_OUT_BITS]
    if width != p[base + H_N_OUT]:
        return 8
    return 0


def packed_n_in(p, base: int = 0) -> int:
    return int(p[base + H_N_IN])


def packed_n_out(p, base: int = 0) -> int:
    return int(p[base + H_N_OUT])


def packed_max_width(p, base: int = 0) -> int:
    return int(p[base + H_MAX_WIDTH])


def packed_in_frac(p, base: int = 0) -> int:
    return int(p[base + H_IN_FRAC])


def packed_out_frac(p, base: int = 0) -> int:
    """Fraction bits of the network output (the last layer's output format)."""
    n_layers = int(p[base + H_N_LAYERS])
    return int(p[base + H_SIZE + (n_layers - 1) * L_SIZE + L_OUT_FRAC])


# =============================================================================================== Python side
def _int_bits_for(maxabs: float, total_bits: int) -> int:
    """Fewest integer bits m (>= 0) of a signed total_bits format Qm.(total-1-m) whose range holds +-maxabs."""
    m = 0
    while m < total_bits - 1:
        f = total_bits - 1 - m
        if maxabs <= 2.0 ** m - 2.0 ** -f:
            return m
        m += 1
    return total_bits - 1


def _mode(value, enum_cls) -> int:
    if isinstance(value, str):
        return int(enum_cls[value.upper()])
    return int(enum_cls(value))


def _act_code(a) -> int:
    if isinstance(a, str):
        if a.lower() not in ACTIVATIONS:
            raise ValueError(f"activation {a!r} not in {sorted(ACTIVATIONS)}")
        return ACTIVATIONS[a.lower()]
    a = int(a)
    if a not in _ACT_NAMES:
        raise ValueError(f"activation code {a} not in {sorted(_ACT_NAMES)}")
    return a


def mlp_forward_float(weights: Sequence[np.ndarray], biases: Sequence[np.ndarray], activations: Sequence,
                      x) -> np.ndarray:
    """The float64 network (module docstring) on x (n_in,) or (B, n_in)."""
    h = np.asarray(x, dtype=np.float64)
    for w, b, a in zip(weights, biases, activations):
        h = h @ np.asarray(w, np.float64).T + np.asarray(b, np.float64)
        code = _act_code(a)
        if code == ACT_RELU:
            h = np.maximum(h, 0.0)
        elif code == ACT_HARDTANH:
            h = np.clip(h, -1.0, 1.0)
    return h


def _float_intervals(weights, biases, activations, lo, hi):
    """Interval propagation of the float network: per layer (in_lo, in_hi, pre_lo, pre_hi, out_lo, out_hi)."""
    out = []
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    for w, b, a in zip(weights, biases, activations):
        w = np.asarray(w, np.float64)
        wp, wn = np.maximum(w, 0.0), np.minimum(w, 0.0)
        z_lo = wp @ lo + wn @ hi + np.asarray(b, np.float64)
        z_hi = wp @ hi + wn @ lo + np.asarray(b, np.float64)
        code = _act_code(a)
        if code == ACT_RELU:
            o_lo, o_hi = np.maximum(z_lo, 0.0), np.maximum(z_hi, 0.0)
        elif code == ACT_HARDTANH:
            o_lo, o_hi = np.clip(z_lo, -1.0, 1.0), np.clip(z_hi, -1.0, 1.0)
        else:
            o_lo, o_hi = z_lo, z_hi
        out.append((lo, hi, z_lo, z_hi, o_lo, o_hi))
        lo, hi = o_lo, o_hi
    return out


@dataclass(frozen=True)
class FxpLayer:
    """One quantized dense layer. ``w_raw`` (n_out, n_in) at ``w_fmt``; ``b_raw`` (n_out,) at ``b_frac`` fraction
    bits in an ``acc_bits``-wide register; outputs in ``out_fmt`` after ``activation``."""
    w_raw: np.ndarray
    b_raw: np.ndarray
    w_fmt: QFormat
    b_frac: int
    acc_bits: int
    out_fmt: QFormat
    activation: int
    acc_bits_needed: int = 0           # exact worst case over every raw input (quantize_mlp); 0 = unknown

    @property
    def n_out(self) -> int:
        return int(self.w_raw.shape[0])

    @property
    def n_in(self) -> int:
        return int(self.w_raw.shape[1])

    @property
    def w(self) -> np.ndarray:
        """Quantized weights as real values."""
        return self.w_raw.astype(np.float64) * 2.0 ** -self.w_fmt.frac_bits

    @property
    def b(self) -> np.ndarray:
        """Quantized biases as real values."""
        return self.b_raw.astype(np.float64) * 2.0 ** -self.b_frac


@dataclass(frozen=True)
class FxpMlp:
    """A quantized MLP (module docstring). Build with :func:`quantize_mlp`; evaluate with :meth:`forward` (raw) or
    :meth:`forward_real` (real in, real out); :meth:`pack` gives the int64 vector of :func:`fxp_mlp_eval`."""
    in_fmt: QFormat
    layers: tuple
    rounding: int = int(Rounding.HALF_EVEN)
    overflow: int = int(Overflow.SATURATE)
    proven: bool = True
    _packed: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def n_in(self) -> int:
        return self.layers[0].n_in

    @property
    def n_out(self) -> int:
        return self.layers[-1].n_out

    @property
    def out_fmt(self) -> QFormat:
        return self.layers[-1].out_fmt

    @property
    def max_width(self) -> int:
        return max([self.n_in] + [l.n_out for l in self.layers])

    def pack(self) -> np.ndarray:
        """The packed int64 vector (module docstring), cached."""
        if self._packed is not None:
            return self._packed
        n_l = len(self.layers)
        size = H_SIZE + n_l * L_SIZE + sum(l.n_out * l.n_in + l.n_out for l in self.layers)
        p = np.zeros(size, np.int64)
        p[H_MAGIC], p[H_N_LAYERS], p[H_N_IN], p[H_N_OUT] = MAGIC, n_l, self.n_in, self.n_out
        p[H_IN_BITS], p[H_IN_FRAC] = self.in_fmt.bits, self.in_fmt.frac_bits
        p[H_RND], p[H_OVF], p[H_MAX_WIDTH] = self.rounding, self.overflow, self.max_width
        p[H_PROVEN], p[H_SIZE_], p[H_VER] = int(self.proven), size, VERSION
        o = H_SIZE + n_l * L_SIZE
        for i, l in enumerate(self.layers):
            r = H_SIZE + i * L_SIZE
            p[r + L_N_OUT], p[r + L_N_IN] = l.n_out, l.n_in
            p[r + L_W_BITS], p[r + L_W_FRAC], p[r + L_B_FRAC] = l.w_fmt.bits, l.w_fmt.frac_bits, l.b_frac
            p[r + L_ACC_BITS], p[r + L_OUT_BITS], p[r + L_OUT_FRAC] = l.acc_bits, l.out_fmt.bits, l.out_fmt.frac_bits
            p[r + L_ACT] = l.activation
            p[r + L_W_OFF] = o
            p[o:o + l.n_out * l.n_in] = l.w_raw.ravel()
            o += l.n_out * l.n_in
            p[r + L_B_OFF] = o
            p[o:o + l.n_out] = l.b_raw
            o += l.n_out
        err = int(fxp_mlp_check(p, 0))
        if err:
            raise ValueError(f"packed network failed its structural check (code {err})")
        object.__setattr__(self, "_packed", p)
        return p

    def forward(self, x_raw) -> np.ndarray:
        """Raw inputs (n_in,) or (B, n_in) at the input format -> raw outputs at :attr:`out_fmt` (int64)."""
        x = np.asarray(x_raw, dtype=np.int64)
        single = x.ndim == 1
        x2 = np.ascontiguousarray(x.reshape(-1, self.n_in))
        lo, hi = self.in_fmt.raw_min, self.in_fmt.raw_max
        if x2.size and (x2.min() < lo or x2.max() > hi):
            raise ValueError(f"raw inputs outside the input format {self.in_fmt} [{lo}, {hi}]")
        y = np.zeros((x2.shape[0], self.n_out), np.int64)
        fxp_mlp_forward_batch(self.pack(), x2, y)
        return y[0] if single else y

    def quantize_input(self, x) -> np.ndarray:
        """Real inputs -> raw inputs (the network's rounding mode, saturating at the input format)."""
        return np.asarray(quantize(np.asarray(x, np.float64), self.in_fmt, Rounding(self.rounding).name.lower(),
                                   "saturate"), dtype=np.int64)

    def output_value(self, y_raw) -> np.ndarray:
        """Raw outputs -> real values (exact)."""
        return np.asarray(y_raw, np.float64) * self.out_fmt.lsb

    def forward_real(self, x) -> np.ndarray:
        """Real inputs -> real outputs of the fixed-point network (input quantized first)."""
        return self.output_value(self.forward(self.quantize_input(x)))

    def describe(self) -> dict:
        """Formats per layer, for reports and for the RTL twin."""
        return {"in_fmt": str(self.in_fmt), "rounding": Rounding(self.rounding).name.lower(),
                "overflow": Overflow(self.overflow).name.lower(), "proven_no_acc_overflow": bool(self.proven),
                "layers": [{"n_in": l.n_in, "n_out": l.n_out, "w_fmt": str(l.w_fmt), "b_frac": l.b_frac,
                            "acc_bits": l.acc_bits, "acc_bits_needed": l.acc_bits_needed, "out_fmt": str(l.out_fmt),
                            "activation": _ACT_NAMES[l.activation]} for l in self.layers]}

    def resources(self) -> dict:
        """FPGA-facing counts: multiply-accumulates per inference, multiplier operand widths, parameter memory bits
        (weights at their width, biases at the accumulator width), widest accumulator."""
        layers = []
        x_bits = self.in_fmt.bits
        for l in self.layers:
            layers.append({"macs": l.n_out * l.n_in, "mult_bits": (l.w_fmt.bits, x_bits),
                           "weight_bits": l.n_out * l.n_in * l.w_fmt.bits, "bias_bits": l.n_out * l.acc_bits,
                           "acc_bits": l.acc_bits})
            x_bits = l.out_fmt.bits
        return {"macs": sum(d["macs"] for d in layers),
                "parameter_bits": sum(d["weight_bits"] + d["bias_bits"] for d in layers),
                "max_acc_bits": max(d["acc_bits"] for d in layers), "layers": layers}

    def memh(self) -> dict[str, list[str]]:
        """``$readmemh`` images: two's complement hex, one value per line, weights row-major (out, in)."""
        out = {}
        for i, l in enumerate(self.layers):
            out[f"layer{i}_w"] = memh_lines(l.w_raw.ravel(), l.w_fmt.bits)
            out[f"layer{i}_b"] = memh_lines(l.b_raw, l.acc_bits)
        return out


def memh_lines(raw, bits: int) -> list[str]:
    """Two's complement hex lines (ceil(bits / 4) digits) of raw integers that fit a signed ``bits`` format."""
    nd = -(-int(bits) // 4)
    mask = (1 << int(bits)) - 1
    lo, hi = -(1 << (int(bits) - 1)), (1 << (int(bits) - 1)) - 1
    lines = []
    for v in np.asarray(raw, np.int64).ravel():
        v = int(v)
        if not lo <= v <= hi:
            raise ValueError(f"{v} does not fit a signed {bits}-bit format")
        lines.append(f"{v & mask:0{nd}x}")
    return lines


def quantize_mlp(weights: Sequence[np.ndarray], biases: Sequence[np.ndarray], activations: Sequence,
                 *, in_fmt: QFormat | str = "Q0.15", w_bits: int = 16, act_bits: int = 16,
                 out_fmt: QFormat | str | None = None, acc_bits: int | Sequence[int] | None = None,
                 rounding: str = "half_even", overflow: str = "saturate", x_lo=None, x_hi=None,
                 margin: float = 1.25) -> FxpMlp:
    """Quantize a float MLP (module docstring for every format rule).

    weights: (n_out, n_in) arrays, one per layer; biases: (n_out,) arrays; activations: 'linear' | 'relu' |
    'hardtanh' per layer. in_fmt: input format (default Q0.15, which holds the gym environment's normalized
    observations of converters up to 16 bits exactly). w_bits / act_bits: weight and hidden-activation widths.
    out_fmt: format of the last layer (default: act_bits wide, Q1.(act_bits - 2) for hardtanh). acc_bits: None (the
    exact worst case of each layer, which proves that no accumulator overflow is possible for any input), one width,
    or one per layer (a narrower accumulator is fitted by ``overflow`` after every addition; the fast path is then
    off). x_lo / x_hi: the input box for the hidden-format ranges (default the input format's range). margin: head
    room on the hidden ranges (covers the quantization error; >= 1).
    """
    ws = [np.asarray(w, np.float64) for w in weights]
    bs = [np.asarray(b, np.float64).reshape(-1) for b in biases]
    acts = [_act_code(a) for a in activations]
    if not (len(ws) == len(bs) == len(acts)) or not ws:
        raise ValueError("weights, biases and activations must have the same (non-zero) length")
    for i, (w, b) in enumerate(zip(ws, bs)):
        if w.ndim != 2 or b.shape != (w.shape[0],):
            raise ValueError(f"layer {i}: weights must be (n_out, n_in) and biases (n_out,)")
        if i and w.shape[1] != ws[i - 1].shape[0]:
            raise ValueError(f"layer {i}: n_in {w.shape[1]} != previous n_out {ws[i - 1].shape[0]}")
        if not (np.all(np.isfinite(w)) and np.all(np.isfinite(b))):
            raise ValueError(f"layer {i}: non-finite parameters")
    fin = QFormat.parse(in_fmt) if isinstance(in_fmt, str) else in_fmt
    if not fin.signed:
        raise ValueError("the input format must be signed")
    rnd = _mode(rounding, Rounding)
    ovf = _mode(overflow, Overflow)
    if not 2 <= w_bits <= 32 or not 2 <= act_bits <= 32:
        raise ValueError("w_bits and act_bits must be in 2..32")
    if not margin >= 1.0:
        raise ValueError("margin must be >= 1 (head room on the hidden ranges)")
    n_in = ws[0].shape[1]
    lo = np.full(n_in, fin.min_value) if x_lo is None else np.broadcast_to(np.asarray(x_lo, float), (n_in,))
    hi = np.full(n_in, fin.max_value) if x_hi is None else np.broadcast_to(np.asarray(x_hi, float), (n_in,))
    if np.any(lo > hi) or np.any(lo < fin.min_value) or np.any(hi > fin.max_value):
        raise ValueError(f"input box must lie inside the input format {fin} [{fin.min_value}, {fin.max_value}]")
    ivals = _float_intervals(ws, bs, acts, lo, hi)
    n_l = len(ws)
    if acc_bits is None:
        acc_list = [None] * n_l
    elif np.ndim(acc_bits) == 0:
        acc_list = [int(acc_bits)] * n_l
    else:
        acc_list = [int(a) for a in acc_bits]
        if len(acc_list) != n_l:
            raise ValueError("acc_bits: one width per layer")
    layers = []
    x_frac = fin.frac_bits
    x_raw_max = 1 << (fin.bits - 1)                 # largest |raw input| of the format: -2^(bits-1)
    proven = True
    for i, (w, b, a) in enumerate(zip(ws, bs, acts)):
        m_w = _int_bits_for(float(np.max(np.abs(w))) if w.size else 0.0, w_bits)
        wf = QFormat(m_w, w_bits - 1 - m_w)
        w_raw = np.asarray(quantize(w, wf, rounding, "checked"), np.int64).reshape(w.shape)
        if i == n_l - 1 and out_fmt is not None:
            of = QFormat.parse(out_fmt) if isinstance(out_fmt, str) else out_fmt
            if not of.signed:
                raise ValueError("the output format must be signed")
        elif a == ACT_HARDTANH:
            of = QFormat(1, act_bits - 2)
        else:
            _, _, z_lo, z_hi, o_lo, o_hi = ivals[i]
            mx = float(np.max(np.maximum(np.abs(o_lo), np.abs(o_hi)))) * float(margin) if o_lo.size else 0.0
            m_o = _int_bits_for(mx, act_bits)
            of = QFormat(m_o, act_bits - 1 - m_o)
        f_acc = wf.frac_bits + x_frac
        if of.frac_bits > f_acc:
            raise ValueError(f"layer {i}: output fraction bits {of.frac_bits} exceed the accumulator's {f_acc}; "
                             "use wider weights (w_bits)")
        if f_acc > 60:
            raise ValueError(f"layer {i}: accumulator fraction bits {f_acc} > 60; use narrower formats")
        b_frac = f_acc                                  # the bias enters the accumulator without rounding
        b_raw = np.asarray(quantize(b, QFormat(61 - f_acc, f_acc), rounding, "checked"), np.int64).reshape(b.shape)
        # exact worst case of every partial sum over every raw input of the format (Python integers)
        worst = 0
        for o in range(w_raw.shape[0]):
            s = sum(abs(int(v)) for v in w_raw[o]) * x_raw_max + abs(int(b_raw[o]))
            worst = max(worst, s)
        need = worst.bit_length() + 1
        if need > MAX_BITS:
            raise ValueError(f"layer {i}: the worst-case accumulator needs {need} bits (> {MAX_BITS}); use narrower "
                             "formats")
        acc = need if acc_list[i] is None else int(acc_list[i])
        if not 2 <= acc <= MAX_BITS:
            raise ValueError(f"layer {i}: acc_bits {acc} outside 2..{MAX_BITS}")
        if acc < need:
            proven = False
        if b_raw.size and int(np.max(np.abs(b_raw))) > (1 << (acc - 1)) - 1:
            raise ValueError(f"layer {i}: a bias does not fit the {acc}-bit accumulator")
        layers.append(FxpLayer(w_raw=w_raw, b_raw=b_raw, w_fmt=wf, b_frac=b_frac, acc_bits=acc, out_fmt=of,
                               activation=a, acc_bits_needed=need))
        x_frac = of.frac_bits
        x_raw_max = 1 << (of.bits - 1)
    return FxpMlp(in_fmt=fin, layers=tuple(layers), rounding=rnd, overflow=ovf, proven=proven)


@dataclass(frozen=True)
class MlpErrorBound:
    """Result of :func:`error_bound`: ``bound`` (n_out,) the largest |y_fixed - y_float| over the box (valid when
    ``saturation_free``), per-layer ``layers`` [{e_in_max, e_out_max, pre_lo, pre_hi, saturation_free}]."""
    bound: np.ndarray
    saturation_free: bool
    layers: tuple


def error_bound(mlp: FxpMlp, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray], activations: Sequence,
                x_lo=None, x_hi=None, *, input_exact: bool = False) -> MlpErrorBound:
    """Bound |y_fixed(x_q) - y_float(x)| over the input box [x_lo, x_hi] (module docstring for the derivation).

    ``weights`` / ``biases`` / ``activations``: the float network that was quantized. ``input_exact``: x is exactly
    representable in the input format (x_q = x; e.g. the gym environment's observations), else the input
    quantization error (half an LSB for the round-half modes, one LSB otherwise) is included.
    """
    ws = [np.asarray(w, np.float64) for w in weights]
    bs = [np.asarray(b, np.float64).reshape(-1) for b in biases]
    acts = [_act_code(a) for a in activations]
    if len(ws) != len(mlp.layers):
        raise ValueError("the float network and the quantized one have different depths")
    fin = mlp.in_fmt
    n_in = mlp.n_in
    lo = np.full(n_in, fin.min_value) if x_lo is None else np.broadcast_to(np.asarray(x_lo, float), (n_in,))
    hi = np.full(n_in, fin.max_value) if x_hi is None else np.broadcast_to(np.asarray(x_hi, float), (n_in,))
    if np.any(lo < fin.min_value) or np.any(hi > fin.max_value):
        raise ValueError("the input box must lie inside the input format (no input saturation)")
    half = mlp.rounding in _HALF_MODES

    def r_of(frac: int) -> float:
        return 2.0 ** -frac * (0.5 if half else 1.0)
    e = np.zeros(n_in) if input_exact else np.full(n_in, r_of(fin.frac_bits))
    ivals = _float_intervals(ws, bs, acts, lo, hi)
    out_layers = []
    sat_free = True
    for i, (layer, w, b, a) in enumerate(zip(mlp.layers, ws, bs, acts)):
        if layer.w_raw.shape != w.shape or layer.activation != a:
            raise ValueError(f"layer {i}: shape or activation differs from the quantized network")
        in_lo, in_hi, z_lo, z_hi, _, _ = ivals[i]
        xmax = np.maximum(np.abs(in_lo), np.abs(in_hi))
        wq, bq = layer.w, layer.b
        e_pre = np.abs(wq - w) @ xmax + np.abs(wq) @ e + np.abs(bq - b)
        r = r_of(layer.out_fmt.frac_bits)
        e_out = e_pre + r
        of = layer.out_fmt
        if layer.activation == ACT_HARDTANH:
            if of.int_bits == 0:
                e_out = e_out + of.lsb                       # +1 is clipped to 1 - lsb
            ok = of.min_value <= -1.0
        else:
            # the fixed pre-activation stays within e_pre + r of the float one; the output fits when that interval
            # (after relu) lies inside the output format
            p_lo = z_lo - e_pre - r
            p_hi = z_hi + e_pre + r
            if layer.activation == ACT_RELU:
                p_lo = np.maximum(p_lo, 0.0)
                p_hi = np.maximum(p_hi, 0.0)
            ok = bool(np.all(p_lo >= of.min_value) and np.all(p_hi <= of.max_value))
        # accumulator: no overflow when the proof covers every input, else check the box (real units)
        if layer.acc_bits < layer.acc_bits_needed:
            x_frac = fin.frac_bits if i == 0 else mlp.layers[i - 1].out_fmt.frac_bits
            acc_max = 2.0 ** (layer.acc_bits - 1 - layer.w_fmt.frac_bits - x_frac)     # real value of 2^(acc-1)
            reach = np.abs(wq) @ (xmax + e) + np.abs(bq)
            ok = ok and bool(np.all(reach < acc_max))
        sat_free = sat_free and bool(ok)
        out_layers.append({"e_in_max": float(np.max(e)) if e.size else 0.0, "e_out_max": float(np.max(e_out)),
                           "pre_lo": z_lo.copy(), "pre_hi": z_hi.copy(), "saturation_free": bool(ok)})
        e = e_out
    return MlpErrorBound(bound=np.asarray(e, float), saturation_free=sat_free, layers=tuple(out_layers))
