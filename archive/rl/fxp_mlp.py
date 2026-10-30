# quantize small neural controllers and run their fixed-point inference with declared RTL semantics.
# a multilayer perceptron uses y=activation(W*x+b) at each layer, with W shaped (outputs, inputs).
# activations are linear, ReLU, or hardtanh=clip(x, -1, 1).
# each is 1-Lipschitz.
# weights, biases, inputs, outputs, and accumulators have explicit integer formats.
# quantize_mlp chooses formats and returns FxpMlp.
# forward uses raw integers.
# forward_real quantizes real inputs, runs integer inference, and converts the result.
# pack produces a flat int64 vector.
# fxp_mlp_eval evaluates packed data with caller-owned scratch, without allocations or array views.
# MRMEnv.compiled_policy uses it inside the engine's compiled controller path.
# error_bound propagates float-versus-fixed error over an input box and reports whether its no-saturation assumptions hold.
# resources and memh export FPGA counts and memory images.
# round the wide accumulator first, apply activation, then fit the signed output format.
# this order matters under wrap overflow.
# see _activate.
# proven=True permits a faster int64 sum only when every possible raw input keeps every accumulator within its declared width.
# FxpMlp verifies that proof from actual weights and biases.
# otherwise fit each addition explicitly.
# all kernels give the same results with NUMBA_DISABLE_JIT=1.
# packed form details are in FxpMlp.pack.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numba as nb
import numpy as np

from mrm.control.fxp import (MAX_BITS, Overflow, QFormat, Rounding, fx_add, fx_cast, fx_mul, fx_overflow,
                             fx_round_shift, quantize)

__all__ = [
    "ACT_LINEAR", "ACT_RELU", "ACT_HARDTANH", "ACTIVATIONS", "FxpLayer", "FxpMlp", "MlpErrorBound",
    "quantize_mlp", "acc_bits_needed", "mlp_forward_float", "error_bound", "fxp_mlp_eval", "fxp_mlp_forward_batch",
    "fxp_mlp_check", "packed_n_in", "packed_n_out", "packed_max_width", "packed_out_frac", "packed_in_frac",
    "H_SIZE", "L_SIZE", "MAGIC", "VERSION", "memh_lines",
]

# activation codes (the packed form stores the code)
ACT_LINEAR = 0
ACT_RELU = 1
ACT_HARDTANH = 2
ACTIVATIONS = {"linear": ACT_LINEAR, "identity": ACT_LINEAR, "relu": ACT_RELU, "hardtanh": ACT_HARDTANH}
_ACT_NAMES = {ACT_LINEAR: "linear", ACT_RELU: "relu", ACT_HARDTANH: "hardtanh"}

MAGIC = 0x4D4C50          # 'MLP' - the first word of a packed network
VERSION = 1               # version of the packed form
# indices of the packed header (module comments, "Packed form").
# H_TOTAL_SIZE holds the length of the whole packed vector - H_SIZE is the length of the header itself.
(H_MAGIC, H_N_LAYERS, H_N_IN, H_N_OUT, H_IN_BITS, H_IN_FRAC, H_RND, H_OVF, H_MAX_WIDTH, H_PROVEN, H_TOTAL_SIZE,
 H_VER) = range(12)
H_SIZE = 12
# indices of one layer record (L_RESERVED is unused, 0)
(L_N_OUT, L_N_IN, L_W_BITS, L_W_FRAC, L_B_FRAC, L_ACC_BITS, L_OUT_BITS, L_OUT_FRAC, L_ACT, L_W_OFF, L_B_OFF,
 L_RESERVED) = range(12)
L_SIZE = 12

_SATURATE = int(Overflow.SATURATE)
_CHECKED = int(Overflow.CHECKED)
_HALF_MODES = (int(Rounding.HALF_UP), int(Rounding.HALF_EVEN), int(Rounding.HALF_AWAY))   # round to nearest


# numba kernels
# round the wide accumulator, apply activation, then fit the output format.
# keep activation before narrowing.
# unlike fx_matvec's fit-then-ReLU order, this preserves the sign of a large negative value under wrap overflow.
# hardtanh clips to raw +/-2^out_frac before saturation.
# with an integer bit, +/-1 fit exactly.
# Q0.n clips +1 to 1-LSB regardless of overflow mode.
@nb.njit(cache=True)
def _activate(acc, f_acc, out_bits, out_frac, act, rnd, ovf):
    z = fx_round_shift(acc, f_acc - out_frac, rnd)
    if act == ACT_HARDTANH:
        one = np.int64(1) << out_frac
        if z > one:
            z = one
        elif z < -one:
            z = -one
        return fx_overflow(z, out_bits, True, _SATURATE)      # only a Q0.n output can clip here (+1 -> 1 - lsb)
    if act == ACT_RELU and z < 0:
        z = np.int64(0)
    return fx_overflow(z, out_bits, True, ovf)


# one layer from the packed vector p (weights at p[w0:], biases at p[b0:]) on x[x0:x0+n_in] into y[y0:].
# index arithmetic only (no views, no allocation).
@nb.njit(cache=True)
def _dense(p, w0, b0, n_out, n_in, w_frac, b_frac, x, x0, x_frac, acc_bits, out_bits, out_frac, act, rnd, ovf,
           proven, y, y0):
    f_acc = w_frac + x_frac
    for o in range(n_out):
        acc = fx_cast(p[b0 + o], b_frac, acc_bits, f_acc, True, rnd, ovf)
        if proven != 0:
            # |every partial sum| < 2^(acc_bits - 1) for every raw input (quantize_mlp proof) - plain int64 is exact
            s = np.int64(0)
            for i in range(n_in):
                s += p[w0 + o * n_in + i] * x[x0 + i]
            acc = acc + s
        else:
            for i in range(n_in):
                prod = fx_mul(p[w0 + o * n_in + i], w_frac, x[x0 + i], x_frac, acc_bits, f_acc, True, rnd, ovf)
                acc = fx_add(acc, prod, acc_bits, True, ovf)
        y[y0 + o] = _activate(acc, f_acc, out_bits, out_frac, act, rnd, ovf)


# evaluate packed p[base:] using raw input/output regions in buf.
# return output width.
# input starts at x_off in the declared input format.
# output starts at y_off in the last layer's format.
# scratch starts at s_off and contains 2*max_width entries.
# it must not overlap input or output.
# alternate scratch halves between layers, then copy the final result to output.
# allocate nothing and create no views, permitting calls inside the engine's compiled tick loop.
@nb.njit(cache=True)
def fxp_mlp_eval(p, base, buf, x_off, y_off, s_off):
    n_layers = p[base + H_N_LAYERS]
    rnd = p[base + H_RND]
    ovf = p[base + H_OVF]
    proven = p[base + H_PROVEN]
    max_width = p[base + H_MAX_WIDTH]
    source_offset = x_off
    x_frac = p[base + H_IN_FRAC]
    for layer_index in range(n_layers):
        record_offset = base + H_SIZE + layer_index * L_SIZE                   # this layer's record
        destination_offset = s_off + (layer_index % 2) * max_width                # the layers alternate between the two scratch halves
        _dense(p, base + p[record_offset + L_W_OFF], base + p[record_offset + L_B_OFF], p[record_offset + L_N_OUT],
            p[record_offset + L_N_IN], p[record_offset + L_W_FRAC],
               p[record_offset + L_B_FRAC], buf, source_offset, x_frac, p[record_offset + L_ACC_BITS],
                   p[record_offset + L_OUT_BITS], p[record_offset + L_OUT_FRAC],
               p[record_offset + L_ACT], rnd, ovf, proven, buf, destination_offset)
        source_offset = destination_offset
        x_frac = p[record_offset + L_OUT_FRAC]
    n_out = p[base + H_N_OUT]
    for o in range(n_out):
        buf[y_off + o] = buf[source_offset + o]
    return n_out


# Rows of x (B, n_in) int64 raw inputs -> rows of y (B, n_out) int64 raw outputs.
@nb.njit(cache=True)
def fxp_mlp_forward_batch(p, x, y):
    n_in = p[H_N_IN]
    n_out = p[H_N_OUT]
    max_width = p[H_MAX_WIDTH]
    buf = np.zeros(n_in + n_out + 2 * max_width, np.int64)      # [input, output, scratch]
    for b in range(x.shape[0]):
        for i in range(n_in):
            buf[i] = x[b, i]
        fxp_mlp_eval(p, 0, buf, 0, n_in, n_in + n_out)
        for o in range(n_out):
            y[b, o] = buf[n_in + o]


# check a packed network before compiled inference.
# p is the packed int64 array, and base is its starting offset.
# return 0 when valid, otherwise return the error code below.
# 1 - array too short
# 2 - invalid magic or version
# 3 - invalid total size
# 4 - invalid layer count
# 5 - incompatible layer widths
# 6 - invalid data offsets
# 7 - invalid accumulator or output width
# 8 - invalid network output width
# 9 - invalid rounding or overflow mode
# 10 - invalid activation
# 11 - shift outside 0..62 or output fraction wider than the accumulator
# 12 - invalid input format
# 13 - a weight times the largest raw input reaches the fx_mul limit of 2^62
# 14 - proven flag set for an accumulator narrower than its worst-case partial sum
# acc_bits_needed computes that worst case from packed weights and biases.
# a valid network evaluates in WRAP and SATURATE modes without exceptions.
# the compiled policy checks the network at init, before the tick loop.
# quantize_mlp produces networks that pass this check.
@nb.njit(cache=True)
def fxp_mlp_check(p, base):
    if p.shape[0] - base < H_SIZE:
        return 1
    if p[base + H_MAGIC] != MAGIC or p[base + H_VER] != VERSION:
        return 2
    size = p[base + H_TOTAL_SIZE]
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
    for layer_index in range(n_layers):
        record_offset = base + H_SIZE + layer_index * L_SIZE
        if p[record_offset + L_N_IN] != width or p[record_offset + L_N_OUT] < 1 or p[record_offset + L_N_OUT] > p[base + H_MAX_WIDTH]:
            return 5
        if (p[record_offset + L_W_OFF] < H_SIZE or p[record_offset + L_B_OFF] < H_SIZE
                or p[record_offset + L_W_OFF] + p[record_offset + L_N_OUT] * p[record_offset + L_N_IN] > size
                or p[record_offset + L_B_OFF] + p[record_offset + L_N_OUT] > size):
            return 6
        if (p[record_offset + L_ACC_BITS] < 2 or p[record_offset + L_ACC_BITS] > MAX_BITS or p[record_offset + L_OUT_BITS] < 2
                or p[record_offset + L_OUT_BITS] > MAX_BITS):
            return 7
        if p[record_offset + L_ACT] < ACT_LINEAR or p[record_offset + L_ACT] > ACT_HARDTANH:
            return 10
        f_acc = p[record_offset + L_W_FRAC] + x_frac
        if (p[record_offset + L_W_FRAC] < 0 or p[record_offset + L_B_FRAC] < 0 or p[record_offset + L_OUT_FRAC] < 0 or f_acc > MAX_BITS
                or p[record_offset + L_OUT_FRAC] > f_acc or abs(p[record_offset + L_B_FRAC] - f_acc) > MAX_BITS):
            return 11
        x_max = np.int64(1) << (x_bits - 1)              # largest |raw input| of this layer
        weight_limit = ((np.int64(1) << MAX_BITS) - 1) // x_max
        w0 = base + p[record_offset + L_W_OFF]
        for j in range(p[record_offset + L_N_OUT] * p[record_offset + L_N_IN]):
            if abs(p[w0 + j]) > weight_limit:
                return 13
        if p[base + H_PROVEN] != 0:
            # the proof of the int64 fast path - every partial sum fits the accumulator for every raw input
            raw_limit = np.int64(1) << MAX_BITS
            b0 = base + p[record_offset + L_B_OFF]
            bias_shift = f_acc - p[record_offset + L_B_FRAC]          # aligns the bias to the accumulator's fraction bits
            n_in = p[record_offset + L_N_IN]
            for o in range(p[record_offset + L_N_OUT]):
                bias_abs = abs(p[b0 + o])
                # worst |partial sum| of output o, capped at lim = 2^62
                if bias_shift >= 0:
                    worst = (raw_limit if (bias_shift >= MAX_BITS or bias_abs > (raw_limit >> bias_shift))
                             else bias_abs << bias_shift)
                else:
                    worst = (bias_abs >> (-bias_shift)) + 1
                for i in range(n_in):
                    t = abs(p[w0 + o * n_in + i]) * x_max         # < 2^62 (code 13)
                    worst = raw_limit if worst >= raw_limit - t else worst + t
                needed_bits = 1                                          # bit_length(worst) + 1 (signed)
                v = worst
                while v > 0:
                    needed_bits += 1
                    v >>= 1
                if needed_bits > p[record_offset + L_ACC_BITS]:
                    return 14
        width = p[record_offset + L_N_OUT]
        x_frac = p[record_offset + L_OUT_FRAC]
        x_bits = p[record_offset + L_OUT_BITS]
    if width != p[base + H_N_OUT]:
        return 8
    return 0


# Number of network inputs of the packed network p[base:].
def packed_n_in(p, base: int = 0) -> int:
    return int(p[base + H_N_IN])


# Number of network outputs.
def packed_n_out(p, base: int = 0) -> int:
    return int(p[base + H_N_OUT])


# Widest layer (inputs included) - half the scratch that `fxp_mlp_eval` needs.
def packed_max_width(p, base: int = 0) -> int:
    return int(p[base + H_MAX_WIDTH])


# Fraction bits of the network input format.
def packed_in_frac(p, base: int = 0) -> int:
    return int(p[base + H_IN_FRAC])


# Fraction bits of the network output (the last layer's output format).
def packed_out_frac(p, base: int = 0) -> int:
    n_layers = int(p[base + H_N_LAYERS])
    return int(p[base + H_SIZE + (n_layers - 1) * L_SIZE + L_OUT_FRAC])


# Python side - formats, the quantized network, the error bound
# Fewest integer bits m (>= 0) of a signed total_bits format Qm.(total-1-m) whose range holds +-maxabs.
def _int_bits_for(maxabs: float, total_bits: int) -> int:
    m = 0
    while m < total_bits - 1:
        f = total_bits - 1 - m
        if maxabs <= 2.0 ** m - 2.0 ** -f:
            return m
        m += 1
    return total_bits - 1


# the integer value of a Rounding or Overflow mode given by name or by value.
def _mode(value, enum_cls) -> int:
    if isinstance(value, str):
        return int(enum_cls[value.upper()])
    return int(enum_cls(value))


# the activation code of a name ('relu') or a code - ValueError for an unknown one.
def _act_code(a) -> int:
    if isinstance(a, str):
        if a.lower() not in ACTIVATIONS:
            raise ValueError(f"activation {a!r} not in {sorted(ACTIVATIONS)}")
        return ACTIVATIONS[a.lower()]
    a = int(a)
    if a not in _ACT_NAMES:
        raise ValueError(f"activation code {a} not in {sorted(_ACT_NAMES)}")
    return a


# the float64 network (module comments) on x (n_in,) or (B, n_in).
def mlp_forward_float(weights: Sequence[np.ndarray], biases: Sequence[np.ndarray], activations: Sequence,
                      x) -> np.ndarray:
    h = np.asarray(x, dtype=np.float64)
    for w, b, a in zip(weights, biases, activations):
        h = h @ np.asarray(w, np.float64).T + np.asarray(b, np.float64)
        code = _act_code(a)
        if code == ACT_RELU:
            h = np.maximum(h, 0.0)
        elif code == ACT_HARDTANH:
            h = np.clip(h, -1.0, 1.0)
    return h


# Interval propagation of the float network - per layer (in_lo, in_hi, pre_lo, pre_hi, out_lo, out_hi).
def _float_intervals(weights, biases, activations, lo, hi):
    out = []
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    for w, b, a in zip(weights, biases, activations):
        w = np.asarray(w, np.float64)
        positive_weights, negative_weights = np.maximum(w, 0.0), np.minimum(w, 0.0)
        z_lo = positive_weights @ lo + negative_weights @ hi + np.asarray(b, np.float64)
        z_hi = positive_weights @ hi + negative_weights @ lo + np.asarray(b, np.float64)
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


# one quantized dense layer.
# `w_raw` (n_out, n_in) at `w_fmt` - `b_raw` (n_out,) at `b_frac` fraction bits in an `acc_bits`-wide register - outputs in `out_fmt` after `activation`.
@dataclass(frozen=True)
class FxpLayer:
    w_raw: np.ndarray
    b_raw: np.ndarray
    w_fmt: QFormat
    b_frac: int
    acc_bits: int
    out_fmt: QFormat
    activation: int
    acc_bits_needed: int = 0           # exact worst case over every raw input (quantize_mlp) - 0 = unknown

    # Number of layer outputs.
    @property
    def n_out(self) -> int:
        return int(self.w_raw.shape[0])

    # Number of layer inputs.
    @property
    def n_in(self) -> int:
        return int(self.w_raw.shape[1])

    # Quantized weights as real values.
    @property
    def w(self) -> np.ndarray:
        return self.w_raw.astype(np.float64) * 2.0 ** -self.w_fmt.frac_bits

    # Quantized biases as real values.
    @property
    def b(self) -> np.ndarray:
        return self.b_raw.astype(np.float64) * 2.0 ** -self.b_frac


# compute the signed accumulator width that prevents overflow for every raw input.
# use Python integers for the exact worst case of each partial sum.
# the width is bit_length(max_o(sum_i(|W_oi|*max|x_raw|)+|b_o|) at f_acc)+1.
# if the bias has more fraction bits than the accumulator, include one rounding LSB.
def _layer_acc_bits_needed(w_raw, b_raw, b_frac: int, f_acc: int, x_raw_max: int) -> int:
    bias_shift = int(f_acc) - int(b_frac)
    worst = 0
    for o in range(w_raw.shape[0]):
        bias_abs = abs(int(b_raw[o]))
        bias = bias_abs << bias_shift if bias_shift >= 0 else (bias_abs >> -bias_shift) + 1
        worst = max(worst, sum(abs(int(v)) for v in w_raw[o]) * int(x_raw_max) + bias)
    return worst.bit_length() + 1


# Per layer, the accumulator width that proves no overflow for any raw input of `in_fmt` (each layer's input range from the previous layer's output format - the rule of `quantize_mlp`).
def acc_bits_needed(in_fmt: QFormat, layers: Sequence) -> list[int]:
    out = []
    x_frac, x_raw_max = in_fmt.frac_bits, 1 << (in_fmt.bits - 1)
    for layer in layers:
        out.append(_layer_acc_bits_needed(layer.w_raw, layer.b_raw, layer.b_frac, layer.w_fmt.frac_bits + x_frac,
            x_raw_max))
        x_frac, x_raw_max = layer.out_fmt.frac_bits, 1 << (layer.out_fmt.bits - 1)
    return out


# hold a quantized network and verify whether plain int64 accumulation is safe.
# build with quantize_mlp.
# forward reads raw input.
# forward_real reads real input.
# pack returns the compiled representation.
# proven=None derives safety from acc_bits_needed.
# recompute from actual parameters, not FxpLayer.acc_bits_needed metadata.
# proven=True raises if a declared accumulator is too narrow.
# proven=False always emulates overflow
# after each addition.
# otherwise the verified fast path is bit-identical to that declared arithmetic.
@dataclass(frozen=True)
class FxpMlp:
    in_fmt: QFormat
    layers: tuple
    rounding: int = int(Rounding.HALF_EVEN)
    overflow: int = int(Overflow.SATURATE)
    proven: bool | None = None
    _packed: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        needed_bits = acc_bits_needed(self.in_fmt, self.layers)
        narrow_layers = [(i, l.acc_bits, n) for i, (l, n) in enumerate(zip(self.layers, needed_bits)) if l.acc_bits < n]
        if self.proven is None:
            proven = not narrow_layers
        elif self.proven and narrow_layers:
            i, a, n = narrow_layers[0]
            raise ValueError(f"FxpMlp(proven=True): layer {i}'s {a}-bit accumulator is narrower than its worst case "
                             f"({n} bits over every raw input), so the no-overflow proof does not hold; pass "
                             f"proven=None to derive it (the declared accumulator is then emulated)")
        else:
            proven = bool(self.proven)
        object.__setattr__(self, "proven", proven)

    # Number of network inputs.
    @property
    def n_in(self) -> int:
        return self.layers[0].n_in

    # Number of network outputs.
    @property
    def n_out(self) -> int:
        return self.layers[-1].n_out

    # format of the network output (the last layer's output format).
    @property
    def out_fmt(self) -> QFormat:
        return self.layers[-1].out_fmt

    # Widest layer, inputs included.
    @property
    def max_width(self) -> int:
        return max([self.n_in] + [l.n_out for l in self.layers])

    # return and cache the network's flat int64 representation.
    # offsets are relative to its start.
    # header — magic, layer count, input/output widths, input bits/fraction, rounding, overflow, maximum width, proof flag, total size, version.
    # H_SIZE is the header length.
    # each L_SIZE layer record stores output/input widths, weight bits/fraction, bias fraction, accumulator bits, output bits/fraction, activation, weight offset, bias offset, and reserved zero.
    # data follows all records, with row-major weights then biases for each layer.
    # run fxp_mlp_check before caching to verify the packed structure and proof flag.
    def pack(self) -> np.ndarray:
        if self._packed is not None:
            return self._packed
        n_layers = len(self.layers)
        size = H_SIZE + n_layers * L_SIZE + sum(layer.n_out * layer.n_in + layer.n_out for layer in self.layers)
        packed = np.zeros(size, np.int64)
        packed[H_MAGIC], packed[H_N_LAYERS], packed[H_N_IN], packed[H_N_OUT] = MAGIC, n_layers, self.n_in, self.n_out
        packed[H_IN_BITS], packed[H_IN_FRAC] = self.in_fmt.bits, self.in_fmt.frac_bits
        packed[H_RND], packed[H_OVF], packed[H_MAX_WIDTH] = self.rounding, self.overflow, self.max_width
        packed[H_PROVEN], packed[H_TOTAL_SIZE], packed[H_VER] = int(self.proven), size, VERSION
        data_offset = H_SIZE + n_layers * L_SIZE
        for i, layer in enumerate(self.layers):
            record_offset = H_SIZE + i * L_SIZE
            packed[record_offset + L_N_OUT], packed[record_offset + L_N_IN] = layer.n_out, layer.n_in
            (packed[record_offset + L_W_BITS], packed[record_offset + L_W_FRAC],
             packed[record_offset + L_B_FRAC]) = layer.w_fmt.bits, layer.w_fmt.frac_bits, layer.b_frac
            (packed[record_offset + L_ACC_BITS], packed[record_offset + L_OUT_BITS],
             packed[record_offset + L_OUT_FRAC]) = layer.acc_bits, layer.out_fmt.bits, layer.out_fmt.frac_bits
            packed[record_offset + L_ACT] = layer.activation
            packed[record_offset + L_W_OFF] = data_offset
            packed[data_offset:data_offset + layer.n_out * layer.n_in] = layer.w_raw.ravel()
            data_offset += layer.n_out * layer.n_in
            packed[record_offset + L_B_OFF] = data_offset
            packed[data_offset:data_offset + layer.n_out] = layer.b_raw
            data_offset += layer.n_out
        check_error = int(fxp_mlp_check(packed, 0))
        if check_error:
            raise ValueError(f"packed network failed its structural check (code {check_error})")
        object.__setattr__(self, "_packed", packed)
        return packed

    # Raw inputs (n_in,) or (B, n_in) at the input format -> raw outputs at `out_fmt` (int64).
    def forward(self, x_raw) -> np.ndarray:
        x = np.asarray(x_raw, dtype=np.int64)
        single = x.ndim == 1
        x2 = np.ascontiguousarray(x.reshape(-1, self.n_in))
        lo, hi = self.in_fmt.raw_min, self.in_fmt.raw_max
        if x2.size and (x2.min() < lo or x2.max() > hi):
            raise ValueError(f"raw inputs outside the input format {self.in_fmt} [{lo}, {hi}]")
        y = np.zeros((x2.shape[0], self.n_out), np.int64)
        fxp_mlp_forward_batch(self.pack(), x2, y)
        return y[0] if single else y

    # Real inputs -> raw inputs (the network's rounding mode, saturating at the input format).
    def quantize_input(self, x) -> np.ndarray:
        return np.asarray(quantize(np.asarray(x, np.float64), self.in_fmt, Rounding(self.rounding).name.lower(),
                                   "saturate"), dtype=np.int64)

    # Raw outputs -> real values (exact).
    def output_value(self, y_raw) -> np.ndarray:
        return np.asarray(y_raw, np.float64) * self.out_fmt.lsb

    # Real inputs -> real outputs of the fixed-point network (input quantized first).
    def forward_real(self, x) -> np.ndarray:
        return self.output_value(self.forward(self.quantize_input(x)))

    # Formats per layer, for reports and for the RTL twin.
    def describe(self) -> dict:
        return {"in_fmt": str(self.in_fmt), "rounding": Rounding(self.rounding).name.lower(),
                "overflow": Overflow(self.overflow).name.lower(), "proven_no_acc_overflow": bool(self.proven),
                "layers": [{"n_in": l.n_in, "n_out": l.n_out, "w_fmt": str(l.w_fmt), "b_frac": l.b_frac,
                            "acc_bits": l.acc_bits, "acc_bits_needed": l.acc_bits_needed, "out_fmt": str(l.out_fmt),
                            "activation": _ACT_NAMES[l.activation]} for l in self.layers]}

    # FPGA-facing counts - multiply-accumulates per inference, multiplier operand widths, parameter memory bits (weights at their width, biases at the accumulator width), widest accumulator.
    def resources(self) -> dict:
        layers = []
        x_bits = self.in_fmt.bits
        for layer in self.layers:
            layers.append({"macs": layer.n_out * layer.n_in, "mult_bits": (layer.w_fmt.bits, x_bits),
                           "weight_bits": layer.n_out * layer.n_in * layer.w_fmt.bits,
                               "bias_bits": layer.n_out * layer.acc_bits,
                           "acc_bits": layer.acc_bits})
            x_bits = layer.out_fmt.bits
        return {"macs": sum(d["macs"] for d in layers),
                "parameter_bits": sum(d["weight_bits"] + d["bias_bits"] for d in layers),
                "max_acc_bits": max(d["acc_bits"] for d in layers), "layers": layers}

    # `$readmemh` images - two's complement hex, one value per line, weights row-major (out, in).
    def memh(self) -> dict[str, list[str]]:
        out = {}
        for i, layer in enumerate(self.layers):
            out[f"layer{i}_w"] = memh_lines(layer.w_raw.ravel(), layer.w_fmt.bits)
            out[f"layer{i}_b"] = memh_lines(layer.b_raw, layer.acc_bits)
        return out


# Two's complement hex lines (ceil(bits / 4) digits) of raw integers that fit a signed `bits` format.
def memh_lines(raw, bits: int) -> list[str]:
    nd = -(-int(bits) // 4)                       # hex digits per value, ceil(bits / 4)
    mask = (1 << int(bits)) - 1
    lo, hi = -(1 << (int(bits) - 1)), (1 << (int(bits) - 1)) - 1
    lines = []
    for v in np.asarray(raw, np.int64).ravel():
        v = int(v)
        if not lo <= v <= hi:
            raise ValueError(f"{v} does not fit a signed {bits}-bit format")
        lines.append(f"{v & mask:0{nd}x}")
    return lines


# quantize a float network and choose safe formats for each layer.
# weights are (n_out, n_in), biases (n_out,), and activations are linear, relu, or hardtanh.
# in_fmt defaults to Q0.15, representing normalized gym converter observations up to 16 bits exactly.
# w_bits and act_bits set weight and hidden-output widths.
# choose the fewest integer bits holding max|W| for each weight format.
# set bias fractional bits equal to accumulator fractional bits, so bias alignment requires no rounding.
# propagate float intervals from x_lo/x_hi, default input-format range.
# apply margin>=1 for headroom.
# hardtanh defaults to one integer bit so +/-1 are exact.
# out_fmt can override the final format.
# margin is a quantization design hyperparameter, default 1.25.
# bound each partial sum over every raw value of its input format using Python integers —
# sum(abs(W_raw)*max(abs(x_raw))) + abs(b_raw).
# acc_bits=None uses that bound.
# a supplied width, or per-layer width list, can be narrower.
# then emulate overflow after each addition and disable the proven fast path.
# reject unsupported formats, shifts, or biases before compiled evaluation.
def quantize_mlp(weights: Sequence[np.ndarray], biases: Sequence[np.ndarray], activations: Sequence,
                 *, in_fmt: QFormat | str = "Q0.15", w_bits: int = 16, act_bits: int = 16,
                 out_fmt: QFormat | str | None = None, acc_bits: int | Sequence[int] | None = None,
                 rounding: str = "half_even", overflow: str = "saturate", x_lo=None, x_hi=None,
                 margin: float = 1.25) -> FxpMlp:
    float_weights = [np.asarray(w, np.float64) for w in weights]
    float_biases = [np.asarray(b, np.float64).reshape(-1) for b in biases]
    activation_codes = [_act_code(a) for a in activations]
    if not (len(float_weights) == len(float_biases) == len(activation_codes)) or not float_weights:
        raise ValueError("weights, biases and activations must have the same (non-zero) length")
    for i, (w, b) in enumerate(zip(float_weights, float_biases)):
        if w.ndim != 2 or b.shape != (w.shape[0],):
            raise ValueError(f"layer {i}: weights must be (n_out, n_in) and biases (n_out,)")
        if i and w.shape[1] != float_weights[i - 1].shape[0]:
            raise ValueError(f"layer {i}: n_in {w.shape[1]} != previous n_out {float_weights[i - 1].shape[0]}")
        if not (np.all(np.isfinite(w)) and np.all(np.isfinite(b))):
            raise ValueError(f"layer {i}: non-finite parameters")
    input_format = QFormat.parse(in_fmt) if isinstance(in_fmt, str) else in_fmt
    if not input_format.signed:
        raise ValueError("the input format must be signed")
    rnd = _mode(rounding, Rounding)
    ovf = _mode(overflow, Overflow)
    if not 2 <= w_bits <= 32 or not 2 <= act_bits <= 32:
        raise ValueError("w_bits and act_bits must be in 2..32")
    if not margin >= 1.0:
        raise ValueError("margin must be >= 1 (head room on the hidden ranges)")
    n_in = float_weights[0].shape[1]
    lo = np.full(n_in, input_format.min_value) if x_lo is None else np.broadcast_to(np.asarray(x_lo, float), (n_in,))
    hi = np.full(n_in, input_format.max_value) if x_hi is None else np.broadcast_to(np.asarray(x_hi, float), (n_in,))
    if np.any(lo > hi) or np.any(lo < input_format.min_value) or np.any(hi > input_format.max_value):
        raise ValueError(f"input box must lie inside the input format {input_format} [{input_format.min_value}, {input_format.max_value}]")
    float_intervals = _float_intervals(float_weights, float_biases, activation_codes, lo, hi)
    n_layers = len(float_weights)
    if acc_bits is None:
        accumulator_widths = [None] * n_layers
    elif np.ndim(acc_bits) == 0:
        accumulator_widths = [int(acc_bits)] * n_layers
    else:
        accumulator_widths = [int(a) for a in acc_bits]
        if len(accumulator_widths) != n_layers:
            raise ValueError("acc_bits: one width per layer")
    layers = []
    x_frac = input_format.frac_bits
    x_raw_max = 1 << (input_format.bits - 1)                 # largest |raw input| of the format - -2^(bits-1)
    proven = True
    for i, (w, b, a) in enumerate(zip(float_weights, float_biases, activation_codes)):
        weight_integer_bits = _int_bits_for(float(np.max(np.abs(w))) if w.size else 0.0, w_bits)
        weight_format = QFormat(weight_integer_bits, w_bits - 1 - weight_integer_bits)
        w_raw = np.asarray(quantize(w, weight_format, rounding, "checked"), np.int64).reshape(w.shape)
        if i == n_layers - 1 and out_fmt is not None:
            output_format = QFormat.parse(out_fmt) if isinstance(out_fmt, str) else out_fmt
            if not output_format.signed:
                raise ValueError("the output format must be signed")
        elif a == ACT_HARDTANH:
            output_format = QFormat(1, act_bits - 2)
        else:
            _, _, z_lo, z_hi, o_lo, o_hi = float_intervals[i]
            max_output_magnitude = float(np.max(np.maximum(np.abs(o_lo),
                np.abs(o_hi)))) * float(margin) if o_lo.size else 0.0
            output_integer_bits = _int_bits_for(max_output_magnitude, act_bits)
            output_format = QFormat(output_integer_bits, act_bits - 1 - output_integer_bits)
        f_acc = weight_format.frac_bits + x_frac
        if output_format.frac_bits > f_acc:
            raise ValueError(f"layer {i}: output fraction bits {output_format.frac_bits} exceed the accumulator's {f_acc}; "
                             "use wider weights (w_bits)")
        if f_acc > 60:
            raise ValueError(f"layer {i}: accumulator fraction bits {f_acc} > 60; use narrower formats")
        b_frac = f_acc                                  # the bias enters the accumulator without rounding
        b_raw = np.asarray(quantize(b, QFormat(61 - f_acc, f_acc), rounding, "checked"), np.int64).reshape(b.shape)
        # bound the accumulator for every raw input in this format.
        needed_bits = _layer_acc_bits_needed(w_raw, b_raw, b_frac, f_acc, x_raw_max)
        if needed_bits > MAX_BITS:
            raise ValueError(f"layer {i}: the worst-case accumulator needs {needed_bits} bits (> {MAX_BITS}); use narrower "
                             "formats")
        accumulator_bits = needed_bits if accumulator_widths[i] is None else int(accumulator_widths[i])
        if not 2 <= accumulator_bits <= MAX_BITS:
            raise ValueError(f"layer {i}: acc_bits {accumulator_bits} outside 2..{MAX_BITS}")
        if accumulator_bits < needed_bits:
            proven = False
        if b_raw.size and int(np.max(np.abs(b_raw))) > (1 << (accumulator_bits - 1)) - 1:
            raise ValueError(f"layer {i}: a bias does not fit the {accumulator_bits}-bit accumulator")
        layers.append(FxpLayer(w_raw=w_raw, b_raw=b_raw, w_fmt=weight_format, b_frac=b_frac,
            acc_bits=accumulator_bits, out_fmt=output_format,
                               activation=a, acc_bits_needed=needed_bits))
        x_frac = output_format.frac_bits
        x_raw_max = 1 << (output_format.bits - 1)
    return FxpMlp(in_fmt=input_format, layers=tuple(layers), rounding=rnd, overflow=ovf, proven=proven)


# result of `error_bound` - `bound` (n_out,) the largest |y_fixed - y_float| over the box (valid when `saturation_free`), per-layer `layers` [{e_in_max, e_out_max, pre_lo, pre_hi, saturation_free}].
@dataclass(frozen=True)
class MlpErrorBound:
    bound: np.ndarray
    saturation_free: bool
    layers: tuple


# bound |y_fixed(x_quantized)-y_float(x)| for every input in [x_lo, x_hi].
# weights, biases, and activations describe the original float network.
# input_exact=True means inputs
# are representable already, as with gym observations.
# otherwise include input quantization error.
# nearest rounding contributes half an LSB.
# floor/toward-zero contribute one LSB.
# with X bounding float layer inputs, e bounding input error, and quantized Wq/bq, e_pre = abs(Wq-W)*X + abs(Wq)*e + abs(bq-b)
# e_out = e_pre + output_rounding_error.
# activations are 1-Lipschitz.
# add one output LSB for Q0.n hardtanh clipping at +1.
# use float intervals to check that output formats do not saturate.
# for a manually narrowed accumulator, also check its reachable range.
# return MlpErrorBound with per-output bounds, per-layer details, and saturation_free.
# the error bound requires those no-overflow/saturation conditions.
# this interval/Lipschitz derivation supplies the quantitative check requested by the design notes - textbook chapter 17 provides the fixed-point precision motivation.
def error_bound(mlp: FxpMlp, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray], activations: Sequence,
                x_lo=None, x_hi=None, *, input_exact: bool = False) -> MlpErrorBound:
    float_weights = [np.asarray(w, np.float64) for w in weights]
    float_biases = [np.asarray(b, np.float64).reshape(-1) for b in biases]
    activation_codes = [_act_code(a) for a in activations]
    if len(float_weights) != len(mlp.layers):
        raise ValueError("the float network and the quantized one have different depths")
    input_format = mlp.in_fmt
    n_in = mlp.n_in
    lo = np.full(n_in, input_format.min_value) if x_lo is None else np.broadcast_to(np.asarray(x_lo, float), (n_in,))
    hi = np.full(n_in, input_format.max_value) if x_hi is None else np.broadcast_to(np.asarray(x_hi, float), (n_in,))
    if np.any(lo < input_format.min_value) or np.any(hi > input_format.max_value):
        raise ValueError("the input box must lie inside the input format (no input saturation)")
    half = mlp.rounding in _HALF_MODES

    def r_of(frac: int) -> float:
        return 2.0 ** -frac * (0.5 if half else 1.0)
    e = np.zeros(n_in) if input_exact else np.full(n_in, r_of(input_format.frac_bits))
    float_intervals = _float_intervals(float_weights, float_biases, activation_codes, lo, hi)
    out_layers = []
    saturation_free = True
    for i, (layer, w, b, a) in enumerate(zip(mlp.layers, float_weights, float_biases, activation_codes)):
        if layer.w_raw.shape != w.shape or layer.activation != a:
            raise ValueError(f"layer {i}: shape or activation differs from the quantized network")
        in_lo, in_hi, z_lo, z_hi, _, _ = float_intervals[i]
        input_magnitude = np.maximum(np.abs(in_lo), np.abs(in_hi))
        quantized_weights, quantized_biases = layer.w, layer.b
        e_pre = np.abs(quantized_weights - w) @ input_magnitude + np.abs(quantized_weights) @ e + np.abs(quantized_biases - b)
        r = r_of(layer.out_fmt.frac_bits)
        e_out = e_pre + r
        output_format = layer.out_fmt
        if layer.activation == ACT_HARDTANH:
            if output_format.int_bits == 0:
                e_out = e_out + output_format.lsb                       # +1 is clipped to 1 - lsb
            ok = output_format.min_value <= -1.0
        else:
            # the fixed pre-activation stays within e_pre + r of the float one - the output fits when that interval (after relu) lies inside the output format
            p_lo = z_lo - e_pre - r
            p_hi = z_hi + e_pre + r
            if layer.activation == ACT_RELU:
                p_lo = np.maximum(p_lo, 0.0)
                p_hi = np.maximum(p_hi, 0.0)
            ok = bool(np.all(p_lo >= output_format.min_value) and np.all(p_hi <= output_format.max_value))
        # accumulator - no overflow when the proof covers every input, else check the box (real units)
        if layer.acc_bits < layer.acc_bits_needed:
            x_frac = input_format.frac_bits if i == 0 else mlp.layers[i - 1].out_fmt.frac_bits
            acc_max = 2.0 ** (layer.acc_bits - 1 - layer.w_fmt.frac_bits - x_frac)     # real value of 2^(acc-1)
            reach = np.abs(quantized_weights) @ (input_magnitude + e) + np.abs(quantized_biases)
            ok = ok and bool(np.all(reach < acc_max))
        saturation_free = saturation_free and bool(ok)
        out_layers.append({"e_in_max": float(np.max(e)) if e.size else 0.0, "e_out_max": float(np.max(e_out)),
                           "pre_lo": z_lo.copy(), "pre_hi": z_hi.copy(), "saturation_free": bool(ok)})
        e = e_out
    return MlpErrorBound(bound=np.asarray(e, float), saturation_free=saturation_free, layers=tuple(out_layers))
