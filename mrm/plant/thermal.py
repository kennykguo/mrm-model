"""Thermal plant runtime: ROM load and integrity check, MIMO assembly and scaling, exact ZOH/FOH stepping, interim model.

Contract: docs/ARCHITECTURE.md section 5.3 (``thermal_rom/1``) and, in docs/research/07_design_critique.md ("07"):
P1-2 (source x observer table over named sites, interim model), P0-1 (sub-us waveguide self-heating poles), P0-10
(exact first-order hold for piecewise-linear inputs), P1-3 (reading side: sidecar error checks), P2-5 (k(T) residue
scaling), D15 (die-level fast/slow time scales), D22 (interim numbers). Numbers: docs/research/04 sections 2.2-2.5.

Physics and discretization
--------------------------
Every source -> observer pair is a Foster sum (04 section 2.5, [CO22-TCPMT]):

    Z(s) = sum_k r_k / (1 + s tau_k)          r_k in K/W, tau_k in s
    step response to dP (W):  dT(t) = dP * sum_k r_k (1 - exp(-t / tau_k))

The state is one first-order lag per (observer, source, pole) with a nonzero residue ("lag" m):

    tau_m dx_m/dt = -x_m + r_m u_src(m)(t)          x_m in K, u in W
    T_obs(o) = T_base + sum_{m in lags of o} x_m    (outputs are rises above T_base; T_base is added by the caller)

Exact discretization over a step h for an input that is linear in time (u0 at the start, u1 at the end); with
y = h/tau and a = exp(-y) (07 P0-10, closed form per Foster pole):

    x_m <- a x_m + r_m (b0 u0 + b1 u1),   b1 = 1 - (1 - a)/y,   b0 = (1 - a)/y - a,   b0 + b1 = 1 - a

A held (zero-order-hold) input is the special case u0 = u1 = u, which gives the exact ZOH x <- a x + (1 - a) r u.
The heater DAC, the junction and driver heat are held inputs; scenario aggressor waveforms are piecewise linear.
Small y uses a 5-term series (no cancellation); any h (multi-tick steps, P0-10 ``substep_policy: auto``) is exact.

Batched layout (07 P0-4): residues and time constants are (B, M) float64 (one row per uncertainty world); the lag
structure (lag_src, lag_obs, ...) is shared by all worlds. The kernels are numba @njit with arrays and scalars only
(no module globals other than constants), and run under NUMBA_DISABLE_JIT=1. ``*_b`` kernels work on one world
(1-D rows, for use inside a prange over B); the un-suffixed kernels loop over b.

Hook used by mrm.params.realize (exact signature; also in docs/handoff/plant_thermal.md)
-----------------------------------------------------------------------------------------
    assemble(rom: ThermalRom | InterimRom,
             layout: LayoutSites | Mapping[str, Mapping] | None,
             n_rings: int,
             scales: ThermalScales | Mapping[str, Any] | None = None,
             *, sources: Sequence[str] | None = None, observers: Sequence[str] | None = None,
             temp_sensors: Sequence[Mapping] | None = None, ring_pitch_um: float | None = None,
             prune_abs_k_per_w: float = 0.0) -> ThermalSystem

Every parameter of InterimRom and every field of ThermalScales may be a scalar or a (B,) array (one value per
uncertainty world); ``InterimRom.z_heater_ring_k_per_w`` may also be (B, N) (per-ring variation). B is the common
broadcast length. A typical per-substep use (fixed step h):

    sys = assemble(InterimRom(z_heater_ring_k_per_w=z0), LayoutSites.line(8, 30.0), 8, ThermalScales(...))
    a, c0, c1 = sys.coefficients(h)                     # (B, M) each, once per distinct h
    x = sys.zero_state()                                # (B, M)
    thermal_advance(x, a, c0, c1, sys.lag_src, sys.obs_ptr, u0, u1, src_scale, t_base_k, t_obs)
        # u0, u1 (B|1, S) source powers in W at the start/end of the step; t_obs (B, O) absolute temperatures

With optical self-heating on, use mrm.plant.selfheat.implicit_step instead of thermal_advance (07 P0-1).
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numba import njit

__all__ = [
    # constants
    "SCHEMA_ROM", "SOURCE_KINDS", "OBSERVER_KINDS", "SCALING_HYPOTHESES",
    "SRC_HEATER", "SRC_WG", "SRC_JUNCTION", "SRC_DRIVER", "SRC_AGGRESSOR",
    "OBS_RING", "OBS_HEATER", "OBS_PD", "OBS_TEMP",
    "LAG_EXOGENOUS", "LAG_OPT_SELF", "LAG_OPT_CROSS_RING", "LAG_OPT_OTHER",
    "PAIR_SELF", "PAIR_CROSS", "PAIR_SITE", "LAYER_FAST", "LAYER_SLOW",
    "XT_TABLE_D_UM", "XT_TABLE_TAU_S", "DEFAULT_TAU_SPLIT_S",
    # data classes
    "RomIntegrityError", "ThermalRom", "InterimRom", "LayoutSites", "ThermalScales", "ThermalSystem",
    # artefact I/O
    "load_rom", "save_rom", "rom_path", "sha256_file", "rom_validation_numbers",
    # assembly
    "assemble", "interim_three_pole", "crosstalk_tau_s", "site_fraction",
    # coefficients and kernels
    "foh_scalar", "foh_coefficients", "zoh_coefficients",
    "thermal_step_b", "thermal_step", "thermal_step_h_b", "thermal_outputs_b", "thermal_outputs",
    "thermal_advance_b", "thermal_advance",
    "observer_split_b", "simulate",
    # analysis helpers
    "frequency_response", "step_response", "ramp_response", "dc_gain", "state_space",
    "reciprocity_error", "cross_optical_lag_gain", "assert_cross_lag_gain", "selfheat_dc_k_per_w",
    "k_temp_residue_scale",
]

SCHEMA_ROM = "thermal_rom/1"
SOURCE_KINDS = ("heater", "waveguide_absorption", "junction", "driver", "aggressor")
OBSERVER_KINDS = ("ring_mode_weighted", "heater_resistor", "monitor_pd", "temp_sensor")
SRC_HEATER, SRC_WG, SRC_JUNCTION, SRC_DRIVER, SRC_AGGRESSOR = 0, 1, 2, 3, 4
OBS_RING, OBS_HEATER, OBS_PD, OBS_TEMP = 0, 1, 2, 3
# Lag kinds: which input drives the lag in mrm.plant.selfheat.implicit_step (thermal_step ignores the kind).
LAG_EXOGENOUS = 0        # heater, junction, driver, aggressor: u0/u1 given by the caller
LAG_OPT_SELF = 1         # waveguide absorption of ring k -> ring k mode temperature: solved implicitly (P0-1)
LAG_OPT_CROSS_RING = 2   # waveguide absorption of ring j -> ring k != j mode temperature: lags one substep (P0-1)
LAG_OPT_OTHER = 3        # waveguide absorption -> any non-ring observer: uses the newly solved P_abs
PAIR_SELF, PAIR_CROSS, PAIR_SITE = 0, 1, 2
LAYER_FAST, LAYER_SLOW = 0, 1
# Order as written in 07 P1-2; an int-coded per-world array uses these indices.
SCALING_HYPOTHESES = ("resistance", "capacitance", "independent")

# Lags with tau below this belong to the fast layer (ring, oxide: 1-124 us in 04 section 2.3 L1), others to the
# slow layer (die, package: 0.1-9 ms, L2). assumed: the boundary between L1 and L2 of 04 section 2.3.
DEFAULT_TAU_SPLIT_S = 3.0e-4

# Sidecar acceptance thresholds: ARCHITECTURE 5.3 and 07 P1-3 (printed in the contract).
ROM_STEP_ERR_MAX = 0.02          # max |dT - dT_FEM| / dT_ss, 10 ns .. 10 ms
ROM_FREQ_DB_MAX = 0.3            # |H| error, DC .. 1 MHz
ROM_FREQ_DEG_MAX = 3.0           # phase error, DC .. 1 MHz
ROM_SUPERPOSITION_MAX = 0.10     # 8-ring bank superposition error on the crosstalk residues
ROM_TAU_FAST_MAX_S = 1.0e-6      # P0-1: the ROM must carry a pole below 1 us (waveguide-core self-heating)
ROM_TAU_SLOW_MIN_S = 1.0e-3      # P1-2: the shared tau set must include the millisecond crosstalk poles

# Crosstalk time constants of neighbours 1..7 at 100 um pitch, the first neighbour 60 um from the heat source
# (simulated, printed: [CO22-TCPMT] Table II via docs/research/04 section 2.4 X2, UCUT; no-UCUT within 1.3 %).
# The distances 160..660 um assume the 100 um pitch continues (inferred).
XT_TABLE_D_UM = np.array([60.0, 160.0, 260.0, 360.0, 460.0, 560.0, 660.0])
XT_TABLE_TAU_S = np.array([4.14, 5.67, 6.57, 7.21, 7.82, 8.48, 8.99]) * 1e-3

_FOH_SERIES_Y = 1.0e-3   # below this y = h/tau the FOH coefficients use their Taylor series (numerics only)


class RomIntegrityError(ValueError):
    """A thermal ROM artefact failed its schema or sha256 check against the sidecar."""


# =============================================================================================================
# Exact hold coefficients
# =============================================================================================================
@njit(cache=True)
def foh_scalar(tau, h):
    """Exact hold coefficients of one Foster pole over a step h (s): returns (a, b0, b1).

    tau dx/dt = -x + r u(t), u linear from u0 to u1 over [0, h]  =>  x(h) = a x(0) + r (b0 u0 + b1 u1), with
    y = h/tau, a = exp(-y), b1 = 1 - (1 - a)/y, b0 = (1 - a)/y - a (07 P0-10). ZOH is u0 = u1: b0 + b1 = 1 - a.
    For y < 1e-3 the 5-term Taylor series is used: b1 = sum_{n>=1} (-1)^(n+1) y^n/(n+1)!,
    b0 = sum_{n>=1} (-1)^(n+1) n y^n/(n+1)!.
    """
    if h <= 0.0:
        return 1.0, 0.0, 0.0
    y = h / tau
    a = math.exp(-y)
    if y < 1.0e-3:
        y2 = y * y
        y3 = y2 * y
        y4 = y3 * y
        y5 = y4 * y
        b1 = y / 2.0 - y2 / 6.0 + y3 / 24.0 - y4 / 120.0 + y5 / 720.0
        b0 = y / 2.0 - y2 / 3.0 + y3 / 8.0 - y4 / 30.0 + y5 / 144.0
        return a, b0, b1
    q = -math.expm1(-y) / y
    return a, q - a, 1.0 - q


def foh_coefficients(taus_s, res_k_per_w, h_s):
    """Vectorized exact FOH coefficients for arrays of poles: returns (a, c0, c1) with c = b * r.

    Units: taus_s s, res_k_per_w K/W, h_s s; a dimensionless, c0, c1 K/W. Equation: see foh_scalar (07 P0-10).
    Update: x <- a x + c0 u0 + c1 u1 (u in W at the start and the end of the step).
    """
    tau = np.asarray(taus_s, dtype=np.float64)
    r = np.asarray(res_k_per_w, dtype=np.float64)
    h = float(h_s)
    if h < 0.0:
        raise ValueError(f"step h_s must be >= 0, got {h_s}")
    if np.any(~(tau > 0.0)):
        raise ValueError("every tau must be > 0")
    if h == 0.0:
        one = np.ones(np.broadcast(tau, r).shape)
        return one, np.zeros_like(one), np.zeros_like(one)
    y = h / tau
    a = np.exp(-y)
    q = -np.expm1(-y) / y
    b1 = 1.0 - q
    b0 = q - a
    small = y < _FOH_SERIES_Y
    if np.any(small):
        ys = y[small]
        b1[small] = ys / 2.0 - ys**2 / 6.0 + ys**3 / 24.0 - ys**4 / 120.0 + ys**5 / 720.0
        b0[small] = ys / 2.0 - ys**2 / 3.0 + ys**3 / 8.0 - ys**4 / 30.0 + ys**5 / 144.0
    a, b0, b1 = np.broadcast_arrays(a, b0, b1)
    return a.copy(), b0 * r, b1 * r


def zoh_coefficients(taus_s, res_k_per_w, h_s):
    """Exact ZOH coefficients (a, c) with c = (1 - a) r: x <- a x + c u for an input held over the step (07 P0-10)."""
    a, c0, c1 = foh_coefficients(taus_s, res_k_per_w, h_s)
    return a, c0 + c1


# =============================================================================================================
# Per-step kernels
# =============================================================================================================
@njit(cache=True)
def thermal_step_b(x, a, c0, c1, lag_src, u0, u1, src_scale):
    """Advance the lag states of one world by one step (in place).

    x (M,) K; a, c0, c1 (M,) from foh_coefficients (c in K/W); lag_src (M,) int64; u0, u1 (S,) W at the start and
    the end of the step (u1 = u0 for held inputs); src_scale (S,) dimensionless multiplier per source (1.0 normally;
    the quasi-static k(T) scaling of 07 P2-5 or a fault). Equation: x_m <- a_m x_m + s_src (c0_m u0_src + c1_m u1_src).
    """
    for m in range(x.shape[0]):
        s = lag_src[m]
        x[m] = a[m] * x[m] + src_scale[s] * (c0[m] * u0[s] + c1[m] * u1[s])


@njit(cache=True)
def thermal_step_h_b(x, taus, res, lag_src, u0, u1, src_scale, h):
    """Like thermal_step_b for a variable step h (s): the exact FOH coefficients are computed inline per lag.

    For adaptive multi-tick steps (07 P0-10 ``substep_policy: auto``). taus (M,) s, res (M,) K/W.
    """
    for m in range(x.shape[0]):
        s = lag_src[m]
        a, b0, b1 = foh_scalar(taus[m], h)
        x[m] = a * x[m] + src_scale[s] * res[m] * (b0 * u0[s] + b1 * u1[s])


@njit(cache=True)
def thermal_outputs_b(x, obs_ptr, t_base, out):
    """Observer temperatures of one world: out[o] = t_base + sum of the lag states of o (K). obs_ptr (O+1,)."""
    for o in range(out.shape[0]):
        acc = t_base
        for m in range(obs_ptr[o], obs_ptr[o + 1]):
            acc += x[m]
        out[o] = acc


@njit(cache=True)
def _row(b, n):
    return b if n > 1 else 0


@njit(cache=True)
def _check_batch(x, a, c0, c1, lag_src, u0, u1, src_scale):
    """Shape guard of the batched kernels (numba does not bounds-check): every per-world input has 1 or B rows,
    a/c0/c1 have the same rows and M columns, lag_src has M entries. Raises ValueError."""
    B = x.shape[0]
    M = x.shape[1]
    na = a.shape[0]
    if c0.shape[0] != na or c1.shape[0] != na or (na != 1 and na != B):
        raise ValueError("thermal kernel: a, c0, c1 must all have 1 or B rows")
    if a.shape[1] != M or c0.shape[1] != M or c1.shape[1] != M or lag_src.shape[0] != M:
        raise ValueError("thermal kernel: a, c0, c1, lag_src must have M (= x.shape[1]) lags")
    for n in (u0.shape[0], u1.shape[0], src_scale.shape[0]):
        if n != 1 and n != B:
            raise ValueError("thermal kernel: u0, u1, src_scale must have 1 or B rows")


@njit(cache=True)
def thermal_step(x, a, c0, c1, lag_src, u0, u1, src_scale):
    """Batched thermal_step_b. x (B, M); a, c0, c1 (B|1, M); u0, u1 (B|1, S) W; src_scale (B|1, S)."""
    _check_batch(x, a, c0, c1, lag_src, u0, u1, src_scale)
    for b in range(x.shape[0]):
        ra = _row(b, a.shape[0])
        ru = _row(b, u0.shape[0])
        thermal_step_b(x[b], a[ra], c0[ra], c1[ra], lag_src, u0[ru], u1[_row(b, u1.shape[0])],
                       src_scale[_row(b, src_scale.shape[0])])


@njit(cache=True)
def thermal_outputs(x, obs_ptr, t_base, out):
    """Batched thermal_outputs_b. x (B, M); t_base (B|1,) K; out (B, O) K."""
    if out.shape[0] != x.shape[0] or (t_base.shape[0] != 1 and t_base.shape[0] != x.shape[0]):
        raise ValueError("thermal_outputs: out must have B rows and t_base 1 or B entries")
    if out.shape[1] + 1 != obs_ptr.shape[0]:
        raise ValueError("thermal_outputs: out must have O = len(obs_ptr) - 1 columns")
    for b in range(x.shape[0]):
        thermal_outputs_b(x[b], obs_ptr, t_base[_row(b, t_base.shape[0])], out[b])


@njit(cache=True, _nrt=False)
def thermal_advance_b(x, a, c0, c1, lag_src, obs_ptr, u0, u1, src_scale, t_base, out):
    """thermal_step_b and thermal_outputs_b fused in one pass over the lags (one world): x advanced in place, out (O,)
    the end-of-step observer temperatures t_base + sum x (K). Same equation as thermal_step_b."""
    for o in range(out.shape[0]):
        acc = t_base
        for m in range(obs_ptr[o], obs_ptr[o + 1]):
            s = lag_src[m]
            xm = a[m] * x[m] + src_scale[s] * (c0[m] * u0[s] + c1[m] * u1[s])
            x[m] = xm
            acc += xm
        out[o] = acc


@njit(cache=True)
def thermal_advance(x, a, c0, c1, lag_src, obs_ptr, u0, u1, src_scale, t_base, out):
    """Batched thermal_advance_b: x (B, M); a, c0, c1 (B|1, M); u0, u1, src_scale (B|1, S); t_base (B|1,);
    out (B, O)."""
    _check_batch(x, a, c0, c1, lag_src, u0, u1, src_scale)
    if out.shape[0] != x.shape[0] or (t_base.shape[0] != 1 and t_base.shape[0] != x.shape[0]):
        raise ValueError("thermal_advance: out must have B rows and t_base 1 or B entries")
    if out.shape[1] + 1 != obs_ptr.shape[0]:
        raise ValueError("thermal_advance: out must have O = len(obs_ptr) - 1 columns")
    for b in range(x.shape[0]):
        ra = _row(b, a.shape[0])
        thermal_advance_b(x[b], a[ra], c0[ra], c1[ra], lag_src, obs_ptr, u0[_row(b, u0.shape[0])],
                          u1[_row(b, u1.shape[0])], src_scale[_row(b, src_scale.shape[0])],
                          t_base[_row(b, t_base.shape[0])], out[b])


@njit(cache=True)
def observer_split_b(x, a, c0, c1, lag_src, obs_ptr, obs, self_src, u0, u1, src_scale):
    """Split one observer's end-of-step temperature rise into a known part and an implicit gain (one world).

    Returns (pre, g): T_obs(end) - T_base = pre + g * u_self, where u_self is the (held) power of source self_src over
    the step, pre includes every other lag driven by u0/u1, and g = sum over the lags of (obs, self_src) of
    s (c0 + c1) in K/W. Used for the heater TCR loop (07 P1-2: R = R0 [1 + alpha (T_heater - T0)], with T_heater the
    heater_resistor observer): electrical.heater_power_row(t_heater_pre = T_base + pre, g_self = g). Values passed in
    u0/u1 for sources not yet known in this step (other rings' heaters, optical absorption) must be the ones the caller
    later uses in the state update, or the coupling is lagged by one step.
    """
    pre = 0.0
    g = 0.0
    for m in range(obs_ptr[obs], obs_ptr[obs + 1]):
        s = lag_src[m]
        pre += a[m] * x[m]
        if s == self_src:
            g += src_scale[s] * (c0[m] + c1[m])
        else:
            pre += src_scale[s] * (c0[m] * u0[s] + c1[m] * u1[s])
    return pre, g


# =============================================================================================================
# Artefact I/O (ARCHITECTURE 5.3, 07 P2-11)
# =============================================================================================================
def sha256_file(path) -> str:
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar_path(path: Path) -> Path:
    """``<stem>.sidecar.json`` (the convention of mrm/extraction/common.py); ``<name>.sidecar.json`` and
    ``<stem>.json`` (the name mrm/extraction/thermal_fem.artefact_paths writes) are also read."""
    return path.with_name(path.stem + ".sidecar.json")


def _find_sidecar(path: Path) -> Path | None:
    # same candidates as mrm.params.config.sidecar_candidates (minus the .meta.json names)
    for cand in (_sidecar_path(path), path.with_name(path.name + ".sidecar.json"), path.with_suffix(".json")):
        if cand.exists():
            return cand
    return None


def rom_path(family: str, film_nm: int | float, root=None) -> Path:
    """Default artefact path ``mrm/data/extracted/thermal_rom_<family>_<film>.npz`` (ARCHITECTURE 5.3)."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "data" / "extracted"
    return base / f"thermal_rom_{family}_{int(round(float(film_nm)))}.npz"


@dataclass(frozen=True, eq=False)
class ThermalRom:
    """A loaded ``thermal_rom/1`` artefact (ARCHITECTURE 5.3): one ring's self table plus distance-binned cross terms.

    taus_s (K,) s shared pole set; sources (S,) and observers (O,) kind names (subsets of SOURCE_KINDS,
    OBSERVER_KINDS); self_res (S, O, K) K/W, source of ring i -> observer of ring i at the default site offsets;
    cross_r_um (D,) centre-to-centre distance grid; cross_res (S, O, D, K) K/W (radially symmetric, valid d > 2R);
    site_offsets_um {monitor_pd: (x, y), temp_sensor: (x, y), driver: (x, y)} the offsets used for self_res;
    meta: the sidecar (schema, sha256, family, film_nm, inputs, tool versions, mesh, validation errors, created_utc).
    """
    taus_s: np.ndarray
    sources: tuple
    observers: tuple
    self_res: np.ndarray
    cross_r_um: np.ndarray
    cross_res: np.ndarray
    site_offsets_um: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        k = np.asarray(self.taus_s).shape[0]
        s, o = len(self.sources), len(self.observers)
        if np.asarray(self.self_res).shape != (s, o, k):
            raise RomIntegrityError(f"self_res shape {np.asarray(self.self_res).shape} != (S, O, K) = {(s, o, k)}")
        d = np.asarray(self.cross_r_um).shape[0]
        if np.asarray(self.cross_res).shape != (s, o, d, k):
            raise RomIntegrityError(f"cross_res shape {np.asarray(self.cross_res).shape} != {(s, o, d, k)}")
        bad = ([n for n in self.sources if n not in SOURCE_KINDS]
               + [n for n in self.observers if n not in OBSERVER_KINDS])
        if bad:
            raise RomIntegrityError(f"unknown source/observer kinds {bad}")
        tau = np.asarray(self.taus_s, dtype=float)
        if np.any(~(tau > 0)) or np.any(np.diff(tau) <= 0):
            raise RomIntegrityError("taus_s must be positive and strictly increasing")
        if d and np.any(np.diff(np.asarray(self.cross_r_um, dtype=float)) <= 0):
            raise RomIntegrityError("cross_r_um must be strictly increasing")
        for nm, arr in (("self_res", self.self_res), ("cross_res", self.cross_res)):
            if not np.all(np.isfinite(arr)):
                raise RomIntegrityError(f"{nm} contains non-finite values")


def save_rom(path, rom: ThermalRom, sidecar: Mapping[str, Any] | None = None) -> Path:
    """Write a ``thermal_rom/1`` artefact and its sidecar ``<stem>.sidecar.json`` with the sha256 of the .npz
    (ARCHITECTURE 5.3; same sidecar naming as mrm/extraction/common.write_artefact).

    The .npz holds taus_s, sources, observers, self_res, cross_r_um, cross_res, site_offset_names, site_offsets_um.
    The sidecar gets schema and sha256 added to ``sidecar`` (family, film_nm, inputs, tool_versions, mesh,
    validation{step_max_rel_err, freq_max_db_err, freq_max_deg_err, superposition_rel_err}, created_utc).
    """
    path = Path(path)
    names = sorted(rom.site_offsets_um)
    offs = np.array([rom.site_offsets_um[n] for n in names], dtype=np.float64).reshape(len(names), 2)
    with open(path, "wb") as f:
        np.savez(f, taus_s=np.asarray(rom.taus_s, np.float64), sources=np.array(rom.sources, dtype="U32"),
                 observers=np.array(rom.observers, dtype="U32"), self_res=np.asarray(rom.self_res, np.float64),
                 cross_r_um=np.asarray(rom.cross_r_um, np.float64), cross_res=np.asarray(rom.cross_res, np.float64),
                 site_offset_names=np.array(names, dtype="U32"), site_offsets_um=offs)
    side = dict(sidecar or {})
    side["schema"] = SCHEMA_ROM
    side["sha256"] = sha256_file(path)
    side["data_file"] = path.name
    with open(_sidecar_path(path), "w") as f:
        json.dump(side, f, indent=2, sort_keys=True, default=str)
    return path


def load_rom(path, *, verify: bool = True, warn: bool = True) -> ThermalRom:
    """Load a ``thermal_rom/1`` .npz and verify it against its sidecar ``<stem>.sidecar.json`` (or ``<stem>.json``)
    (ARCHITECTURE 5.3, 07 P2-11).

    verify=True: the sidecar must exist, carry schema 'thermal_rom/1' and the sha256 of the .npz bytes; otherwise
    RomIntegrityError. warn=True: RuntimeWarning when the sidecar's validation errors exceed the 07 P1-3 targets
    (step < 2 % of steady state, |dH| < 0.3 dB and < 3 deg to 1 MHz, superposition < 10 %) or are missing, or when the
    pole set lacks a sub-us pole (P0-1) or a millisecond pole (P1-2).
    Both writers are read: save_rom (``site_offset_names``, flat validation keys) and
    mrm/extraction/thermal_fem.write_artefact (``site_names``, ``<stem>.json``, nested validation report, offsets also
    in ``inputs.config.site_offsets_um``); see rom_validation_numbers.
    """
    path = Path(path)
    side_path = _find_sidecar(path)
    meta: dict = {}
    if side_path is not None:
        meta = json.loads(side_path.read_text())
    elif verify:
        raise RomIntegrityError(f"missing sidecar {_sidecar_path(path)}")
    if verify:
        if meta.get("schema") != SCHEMA_ROM:
            raise RomIntegrityError(f"{side_path}: schema {meta.get('schema')!r} != {SCHEMA_ROM!r}")
        digest = sha256_file(path)
        if meta.get("sha256") != digest:
            raise RomIntegrityError(f"{path}: sha256 {digest} does not match the sidecar ({meta.get('sha256')})")
    with np.load(path, allow_pickle=False) as z:
        taus = np.asarray(z["taus_s"], np.float64)
        sources = tuple(str(s) for s in z["sources"])
        observers = tuple(str(s) for s in z["observers"])
        self_res = np.asarray(z["self_res"], np.float64)
        cross_r = np.asarray(z["cross_r_um"], np.float64)
        cross_res = np.asarray(z["cross_res"], np.float64)
        offsets = {}
        # names key: 'site_offset_names' (save_rom) or 'site_names' (mrm/extraction/thermal_fem.write_artefact)
        name_key = next((k for k in ("site_offset_names", "site_names") if k in z.files), None)
        if name_key is not None and "site_offsets_um" in z.files:
            for n, xy in zip(z[name_key], np.asarray(z["site_offsets_um"], np.float64).reshape(-1, 2)):
                offsets[str(n)] = (float(xy[0]), float(xy[1]))
    # sidecar fallbacks: a top-level site_offsets_um dict, or the extraction config (inputs.config.site_offsets_um)
    cfg_offs = ((meta.get("inputs") or {}).get("config") or {}).get("site_offsets_um") or {}
    for src in (meta.get("site_offsets_um") or {}, cfg_offs):
        offsets.update({k: tuple(map(float, v)) for k, v in src.items() if k not in offsets})
    meta = dict(meta)
    meta["path"] = str(path)
    rom = ThermalRom(taus, sources, observers, self_res, cross_r, cross_res, offsets, meta)
    if warn:
        _check_rom_quality(rom)
    return rom


def rom_validation_numbers(meta: Mapping[str, Any]) -> dict:
    """The four 07 P1-3 acceptance numbers of a ROM sidecar: {step_max_rel_err, freq_max_db_err, freq_max_deg_err,
    superposition_rel_err} (None when absent).

    Read from flat keys ``validation.<key>`` (save_rom), or from the nested report that
    mrm/extraction/thermal_fem.verify writes (``validation.{self, nearest_neighbour}.{max_time_err_frac_of_dc, max_db,
    max_deg}``; 07 P1-3 asks for the self and nearest-neighbour terms, so the worse of the two is used). The
    superposition error is also looked up in ``stages.<stage>.superposition_rel_err`` (thermal_fem.update_sidecar).
    """
    v = dict(meta.get("validation") or {})
    out = {k: v.get(k) for k in ("step_max_rel_err", "freq_max_db_err", "freq_max_deg_err", "superposition_rel_err")}
    nested = {"step_max_rel_err": "max_time_err_frac_of_dc", "freq_max_db_err": "max_db",
              "freq_max_deg_err": "max_deg"}
    for key, sub in nested.items():
        if out[key] is None:
            vals = [float(v[g][sub]) for g in ("self", "nearest_neighbour")
                    if isinstance(v.get(g), Mapping) and v[g].get(sub) is not None]
            out[key] = max(vals) if vals else None
    if out["superposition_rel_err"] is None:
        for st in (meta.get("stages") or {}).values():
            if isinstance(st, Mapping) and st.get("superposition_rel_err") is not None:
                out["superposition_rel_err"] = float(st["superposition_rel_err"])
    return out


def _check_rom_quality(rom: ThermalRom) -> None:
    v = rom_validation_numbers(rom.meta)
    checks = (("step_max_rel_err", ROM_STEP_ERR_MAX), ("freq_max_db_err", ROM_FREQ_DB_MAX),
              ("freq_max_deg_err", ROM_FREQ_DEG_MAX), ("superposition_rel_err", ROM_SUPERPOSITION_MAX))
    for key, lim in checks:
        val = v.get(key)
        if val is None:
            warnings.warn(f"thermal ROM {rom.meta.get('path')}: sidecar lacks validation.{key} (07 P1-3)",
                          RuntimeWarning, stacklevel=3)
        elif not (abs(float(val)) <= lim):
            warnings.warn(f"thermal ROM {rom.meta.get('path')}: validation.{key} = {val} exceeds {lim} (07 P1-3)",
                          RuntimeWarning, stacklevel=3)
    if rom.taus_s[0] > ROM_TAU_FAST_MAX_S:
        warnings.warn(f"thermal ROM has no pole below {ROM_TAU_FAST_MAX_S} s; 07 P0-1 needs the waveguide-core "
                      "self-heating poles", RuntimeWarning, stacklevel=3)
    if rom.taus_s[-1] < ROM_TAU_SLOW_MIN_S:
        warnings.warn(f"thermal ROM has no pole above {ROM_TAU_SLOW_MIN_S} s; 07 P1-2 needs the ms crosstalk poles",
                      RuntimeWarning, stacklevel=3)


# =============================================================================================================
# Layout, scales and the interim model specification
# =============================================================================================================
def _xy(a, n=None, name="xy"):
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == 2:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be (n, 2) in um, got shape {arr.shape}")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} must have {n} rows, got {arr.shape[0]}")
    return arr


@dataclass(frozen=True, eq=False)
class LayoutSites:
    """Positions (um) of the thermal sites of one bank (07 P1-2 ``layout.sites``).

    ring_xy_um (N, 2): ring centres. pd_xy_um (N, 2): monitor PD sites (None, or a NaN row for one ring: ring + the
    model's default offset). driver_xy_um (N, 2): driver sites (None: no driver heat source; a NaN row: no driver for
    that ring). temp_sensor_xy_um (T, 2) with temp_sensor_ring (T,) the attached ring (-1: free site) and names.
    aggressor_xy_um (A, 2) with names.
    """
    ring_xy_um: np.ndarray
    pd_xy_um: np.ndarray | None = None
    driver_xy_um: np.ndarray | None = None
    temp_sensor_xy_um: np.ndarray | None = None
    temp_sensor_ring: np.ndarray | None = None
    temp_sensor_names: tuple = ()
    aggressor_xy_um: np.ndarray | None = None
    aggressor_names: tuple = ()

    def __post_init__(self):
        n = _xy(self.ring_xy_um, name="ring_xy_um").shape[0]
        object.__setattr__(self, "ring_xy_um", _xy(self.ring_xy_um, name="ring_xy_um"))
        for nm in ("pd_xy_um", "driver_xy_um"):
            v = getattr(self, nm)
            if v is not None:
                object.__setattr__(self, nm, _xy(v, n, nm))
        if self.temp_sensor_xy_um is not None:
            ts = _xy(self.temp_sensor_xy_um, name="temp_sensor_xy_um")
            object.__setattr__(self, "temp_sensor_xy_um", ts)
            ring = (np.full(ts.shape[0], -1, np.int64) if self.temp_sensor_ring is None
                    else np.asarray(self.temp_sensor_ring, np.int64).reshape(-1))
            if ring.shape[0] != ts.shape[0] or np.any(ring >= n) or np.any(ring < -1):
                raise ValueError("temp_sensor_ring must have one entry per sensor, each -1 or a ring index")
            object.__setattr__(self, "temp_sensor_ring", ring)
            names = tuple(self.temp_sensor_names) or tuple(f"ts{i}" for i in range(ts.shape[0]))
            object.__setattr__(self, "temp_sensor_names", names)
        if self.aggressor_xy_um is not None:
            ag = _xy(self.aggressor_xy_um, name="aggressor_xy_um")
            object.__setattr__(self, "aggressor_xy_um", ag)
            names = tuple(self.aggressor_names) or tuple(f"aggressor{i}" for i in range(ag.shape[0]))
            object.__setattr__(self, "aggressor_names", names)

    @property
    def n_rings(self) -> int:
        return self.ring_xy_um.shape[0]

    @staticmethod
    def line(n_rings: int, pitch_um: float, *, axis: int = 0, aggressor_xy_um=None, temp_sensor_xy_um=None,
             temp_sensor_ring=None, temp_sensor_names=(), aggressor_names=(), pd_xy_um=None,
             driver_xy_um=None) -> "LayoutSites":
        """A straight bank: ring i at i * pitch_um along x (axis 0) or y (axis 1)."""
        xy = np.zeros((int(n_rings), 2))
        xy[:, axis] = np.arange(int(n_rings)) * float(pitch_um)
        return LayoutSites(xy, pd_xy_um, driver_xy_um, temp_sensor_xy_um, temp_sensor_ring, tuple(temp_sensor_names),
                           aggressor_xy_um, tuple(aggressor_names))

    @staticmethod
    def from_sites(sites: Mapping[str, Mapping] | None, n_rings: int, ring_pitch_um: float | None = None,
                   temp_sensors: Sequence[Mapping] | None = None) -> "LayoutSites":
        """Parse ``layout.sites`` {name: {x_um, y_um, kind, [ring]}} and the ``temp_sensors`` table (07 P1-2, P1-10).

        kind ring: name 'ring<i>' or field ring=i (missing rings: ring<i> on a line at ring_pitch_um, schema doc).
        kind monitor_pd / driver: field ring=i or a trailing index in the name; a driver site with no ring index is
        shared by every ring (one driver block). kind aggressor, and kind heater without an explicit ``ring`` field
        (a trailing digit in the name does not attach it; the schema's site fields have no ``ring``): aggressor heat
        sources. A heater site with ``ring`` is that ring's own heater, which the model places at the ring centre.
        kind temp_sensor: a free sensor site named after the site, unless a ``temp_sensors`` entry references it
        through ``site`` (then that entry, with its own name, is the one sensor at that site; no duplicate).
        temp_sensors entries {name, site | attach}: attach 'ring_<i>' puts the sensor at ring i (offset added by the
        model), 'bank' at the ring centroid, 'die' 1000 um from the centroid (assumed placement for a die-level
        sensor).
        """
        sites = dict(sites or {})
        referenced = {str(spec["site"]) for spec in (temp_sensors or ()) if spec.get("site")}
        ring_xy = np.full((n_rings, 2), np.nan)
        pd = np.full((n_rings, 2), np.nan)
        drv = np.full((n_rings, 2), np.nan)
        shared_drv = None
        ag_xy, ag_names, ts_xy, ts_ring, ts_names = [], [], [], [], []

        def idx_of(name, spec):
            if "ring" in spec and spec["ring"] is not None:
                return int(spec["ring"])
            digits = ""
            for ch in reversed(str(name)):
                if ch.isdigit():
                    digits = ch + digits
                else:
                    break
            return int(digits) if digits else None

        for name, spec in sites.items():
            kind = spec.get("kind", "other")
            p = (float(spec["x_um"]), float(spec["y_um"]))
            i = idx_of(name, spec)
            if kind == "ring":
                if i is None or not 0 <= i < n_rings:
                    raise ValueError(f"ring site {name!r} needs a ring index 0..{n_rings - 1}")
                ring_xy[i] = p
            elif kind == "monitor_pd":
                if i is None or not 0 <= i < n_rings:
                    raise ValueError(f"monitor_pd site {name!r} needs a ring index")
                pd[i] = p
            elif kind == "driver":
                if i is None:
                    shared_drv = p
                elif 0 <= i < n_rings:
                    drv[i] = p
            elif kind == "aggressor" or (kind == "heater" and (i is None or "ring" not in spec)):
                ag_xy.append(p)
                ag_names.append(str(name))
            elif kind == "temp_sensor" and str(name) not in referenced:
                ts_xy.append(p)
                ts_ring.append(-1)
                ts_names.append(str(name))
        missing = np.isnan(ring_xy[:, 0])
        if np.any(missing):
            if ring_pitch_um is None and n_rings > 1:
                raise ValueError(f"layout has no site for rings {np.flatnonzero(missing).tolist()} and no "
                                 "layout.ring_pitch_um")
            pitch = 0.0 if ring_pitch_um is None else float(ring_pitch_um)
            ring_xy[missing, 0] = np.flatnonzero(missing) * pitch
            ring_xy[missing, 1] = 0.0
        if shared_drv is not None:
            drv[np.isnan(drv[:, 0])] = shared_drv
        centroid = ring_xy.mean(axis=0)
        for k, spec in enumerate(temp_sensors or ()):
            name = str(spec.get("name", f"ts{k}"))
            if spec.get("site"):
                s = sites[spec["site"]]
                ts_xy.append((float(s["x_um"]), float(s["y_um"])))
                ts_ring.append(-1)
            else:
                att = str(spec.get("attach", "die"))
                if att.startswith("ring_"):
                    i = int(att.split("_", 1)[1])
                    ts_xy.append(tuple(ring_xy[i]))
                    ts_ring.append(i)
                elif att == "bank":
                    ts_xy.append(tuple(centroid))
                    ts_ring.append(-1)
                else:   # die: assumed 1 mm from the bank centroid
                    ts_xy.append((centroid[0], centroid[1] + 1000.0))
                    ts_ring.append(-1)
            ts_names.append(name)
        return LayoutSites(          # NaN rows: PD at the model's default offset, no driver for that ring
            ring_xy,
            None if np.all(np.isnan(pd[:, 0])) else pd,
            None if np.all(np.isnan(drv[:, 0])) else drv,
            np.array(ts_xy, float).reshape(-1, 2) if ts_xy else None,
            np.array(ts_ring, np.int64) if ts_xy else None, tuple(ts_names),
            np.array(ag_xy, float).reshape(-1, 2) if ag_xy else None, tuple(ag_names))


@dataclass(frozen=True, eq=False)
class ThermalScales:
    """Per-world thermal uncertainty factors (07 P1-2 ``uncertainty.thermal``, D15, section 4.8). Scalars or (B,).

    dc_scale: a global factor, or a mapping {"*": f, "<source>-><observer>": f, "<source>->*": f, "*-><observer>": f}
        (kind names); the factors that match a lag multiply. rom_dc_scale = efficiency/(Z_FEM * dlambda/dT) goes here.
    time_scale_fast, time_scale_slow: tau multipliers of the fast (tau < tau_split_s, or the interim self poles) and
        slow (die, package, crosstalk) layers, one draw per world (D15).
    scaling_hypothesis: 'resistance' (tau_k scales with its residue, tau = RC, 07 P1-2), 'capacitance' or
        'independent' (residue and tau scale independently); a string, a (B,) sequence of strings, or int codes in
        SCALING_HYPOTHESES order. Lag m: r' = sigma_m x_m r, tau' = s_layer tau * (sigma_m if resistance else 1), with
        sigma_m the product of the matching dc_scale factors (the conductance error of the stack, e.g.
        rom_dc_scale) and x_m = crosstalk_scale on ring-to-ring pairs (1 elsewhere).
    crosstalk_scale: multiplies every ring-to-ring cross residue (not aggressor or free-sensor paths). It is a
        magnitude uncertainty (layout, metal shunts, EIC stacking; 07 4.8 sweeps 1-10 %), not a conductance error of
        one RC, so it never scales tau: a larger crosstalk from a shorter path or a metal shunt comes with a shorter
        tau, not a longer one (04 section 2.4 d^2 law). The crosstalk tau has its own law and time_scale_slow.
    package_tail_tau_s, package_tail_share_frac: a die/package tail pole carrying a share of each self pair's DC
        (07 4.8: 1 to 10 ms, 2 to 10 %). None keeps the model's own tail (interim) or none (ROM). The other tailable
        self poles are rescaled so that each pair's DC is unchanged. A tail given here is absolute: its tau is not
        multiplied by time_scale_slow or by the 'resistance' factor (4.8 lists it as its own interval; the model's
        own tail, InterimRom.tail_tau_s, is a family value and does get time_scale_slow).
    transport_delay_s: heat transport delay per world (07 P1-2, 4.8: 0 to 2 us), stored as
        ThermalSystem.transport_delay_s (B,). It is not applied by the thermal kernels: 07 P0-6 gives every delay one
        owner, and the heater actuator FIFO adds it as whole ticks of heater latency (docs/handoff/plant_thermal.md).
    """
    dc_scale: Any = 1.0
    time_scale_fast: Any = 1.0
    time_scale_slow: Any = 1.0
    scaling_hypothesis: Any = "resistance"
    crosstalk_scale: Any = 1.0
    package_tail_tau_s: Any = None
    package_tail_share_frac: Any = None
    tau_split_s: float = DEFAULT_TAU_SPLIT_S
    transport_delay_s: Any = 0.0


@dataclass(frozen=True, eq=False)
class InterimRom:
    """Preset numbers of the interim thermal model of 07 P1-2 (used until the FEM ROM of the heater family exists).

    Field names follow mrm/params/schema.py ``ring.thermal.*`` (SI). Scalars or (B,) arrays; z_heater_ring_k_per_w may
    be (B, N). Self path (heater -> ring), with Z0 = z_heater_ring_k_per_w and S_H the normalized shape:

        Z_RH(s) = Z0 [f_F/(1 + s tau_F) + (1 - f_F - f_T)/(1 + s tau_S) + f_T/(1 + s tau_T)] = Z0 S_H(s)
        Z_RR(s) = z Z0 [f_sh/(1 + s tau_sh) + (1 - f_sh) S_H(s)]          optical (waveguide) self-heating (P0-1)
        Z_HH(s) = Z0 [S_H(s) + (h - 1)/(1 + s tau_hn)]                      heater resistor observing its own heater
        Z_{RH} = Z_{HR}: heater -> ring equals waveguide absorption -> heater resistor (reciprocity, 07 P1-2 test)

    f_sh defaults to (z - 1)/z, the largest share for which every pole's 2x2 residue matrix over {heater, ring} stays
    positive semidefinite (a passive RC network). Junction heat uses the waveguide residues (P1-8: dissipated inside
    the mode region). Ring-to-ring crosstalk C = dlambda_victim/dlambda_heated: xt_nearest_frac to the nearest
    neighbour(s), xt_next_frac (optional) to the second rank, xt_floor_frac to all others, each a single pole
    tau_xt(d) (crosstalk_tau_s) or, with crosstalk_dynamics='self', the self shape S_H (the textbook corner). Sites
    (monitor PD, driver, temperature sensor, aggressor) use site_fraction(d) with the same tau_xt(d).

    Defaults and provenance:
      z_heater_ring_k_per_w 6.1e3 K/W: derived, R_th under H_solver (07 4.6); realize() passes efficiency/(dlambda/dT).
      z_ratio_opt_heater 1.5: assumed, middle of the 07 4.8 interval 1..2 ([PHY], [R06 H2]).
      tau_fast_s 3.2e-6, tau_slow_s 13.0e-6: inferred, [CO22-TCPMT] Table I two-pole fit (04 T2: 3.18/13.01 us).
      share_fast_frac 0.4085 = 0.43 x (1 - 0.05): inferred 43 % share of the two-pole fit, scaled for the tail.
      tail_tau_s 4e-3, tail_share_frac 0.05: assumed inside 07's 1-10 ms and 2-10 % ([CO22-TCPMT] circuit-level 4 ms).
      selfheat_tau_s 0.5e-6: literature, de Cea 2019 tau_th ~0.5 us (03 section 6.3); band 0.27-1.6 us (03 9.1).
      heater_self_ratio 1.5 (R_th,H/Z0), heater_node_tau_s 1e-6: assumed (06 E1 bounds R_th,H >= R_th only).
      xt_nearest_frac 0.03, xt_floor_frac 0.015: inferred (07 P1-2; 04 section 2.5 from X1-X5).
      xt_tau_ref_s 4.1e-3 at xt_dist_ref_um 60, xt_tau_min_s 1e-4: 07 P1-2 (04 section 2.4, [CO22-TCPMT] Eq. 4, X2).
      xt_tau_far_s None: use the printed X2 table beyond the near field; xt_far_um 100 (07 P1-2).
      site_ref_um 60: site_fraction = xt_nearest at the [CO22-TCPMT] first-neighbour distance (assumed law).
      pd_offset_um (0, 25), temp_sensor_offset_um (40, 0): assumed, the same site offsets as the FEM ROM
        (mrm/extraction/thermal_fem.ThermalConfig), so both thermal models place the sites alike.
      driver_offset_um None: no driver heat source unless the layout gives driver sites (G26 drives from a separate
        EIC beside the PIC, 07 P1-3); (0, -30) is the FEM ROM's monolithic-driver offset.
    """
    z_heater_ring_k_per_w: Any = 6.1e3
    z_ratio_opt_heater: Any = 1.5
    tau_fast_s: Any = 3.2e-6
    share_fast_frac: Any = 0.4085
    tau_slow_s: Any = 13.0e-6
    tail_tau_s: Any = 4.0e-3
    tail_share_frac: Any = 0.05
    selfheat_tau_s: Any = 0.5e-6
    selfheat_share_frac: Any = None
    heater_self_ratio: Any = 1.5
    heater_node_tau_s: Any = 1.0e-6
    xt_nearest_frac: Any = 0.03
    xt_floor_frac: Any = 0.015
    xt_next_frac: Any = None
    xt_tau_ref_s: Any = 4.1e-3
    xt_dist_ref_um: float = 60.0
    xt_tau_min_s: Any = 1.0e-4
    xt_tau_far_s: Any = None
    xt_far_um: float = 100.0
    xt_tau_law: str = "capped"
    crosstalk_dynamics: str = "imec"
    site_ref_um: float = 60.0
    site_max_frac: float = 1.0
    pd_offset_um: tuple = (0.0, 25.0)
    temp_sensor_offset_um: tuple = (40.0, 0.0)
    driver_offset_um: tuple | None = None
    name: str = "interim_three_pole"

    @staticmethod
    def textbook_corner(**kw) -> "InterimRom":
        """The pessimistic textbook corner of 07 P1-2/D22: 10 us + 300 us (25 %) self path, no tail, crosstalk 10 %
        nearest / 3 % next / 0 beyond with the self-path dynamics (textbook REF tauThUs, tauSlowUs, crosstalk*)."""
        base = dict(tau_fast_s=10e-6, share_fast_frac=0.75, tau_slow_s=300e-6, tail_share_frac=0.0,
                    xt_nearest_frac=0.10, xt_next_frac=0.03, xt_floor_frac=0.0, crosstalk_dynamics="self",
                    name="interim_textbook_corner")
        base.update(kw)
        return InterimRom(**base)


def crosstalk_tau_s(d_um, tau_ref_s=4.1e-3, dist_ref_um=60.0, tau_min_s=1.0e-4, tau_far_s=None, far_um=100.0,
                    law: str = "capped"):
    """First-order crosstalk time constant at centre-to-centre distance d (um) (07 P1-2; 04 section 2.4).

    Near field (derived d^2 diffusion law, [CO22-TCPMT] Eq. 4): tau_near = max(tau_min, tau_ref (d/d_ref)^2).
    law='literal' (07 P1-2 as written): tau_near for d <= far_um, tau_far beyond (tau_far: a number, or None = the
        printed X2 table interpolated in d). Note: at d = 100 um the d^2 law gives 11.4 ms, above every X2 value
        (4.1-9.0 ms), so the literal law is not monotone in d.
    law='capped' (default): tau = min(tau_near, tau_far(d)) with tau_far(d) the X2 table (or the given number) and
        the result floored at tau_min. Monotone and continuous; equal to the d^2 law inside about 60 um.
    Arrays broadcast; returns s.
    """
    d = np.asarray(d_um, dtype=np.float64)
    near = np.maximum(np.asarray(tau_min_s, float), np.asarray(tau_ref_s, float) * (d / float(dist_ref_um)) ** 2)
    if tau_far_s is None:
        far = np.interp(d, XT_TABLE_D_UM, XT_TABLE_TAU_S)
    else:
        far = np.broadcast_to(np.asarray(tau_far_s, float), np.broadcast(d, np.asarray(tau_far_s, float)).shape)
    if law == "literal":
        return np.where(d <= float(far_um), near, far)
    if law != "capped":
        raise ValueError(f"crosstalk tau law must be 'capped' or 'literal', got {law!r}")
    return np.maximum(np.minimum(near, far), np.asarray(tau_min_s, float))


def site_fraction(d_um, nearest_frac=0.03, floor_frac=0.015, ref_um=60.0, max_frac=1.0):
    """Interim static coupling of a site path, as a fraction of Z_heater->ring(0) (assumed law, 04 X1-X5 shape):
    c(d) = clip(nearest_frac * ref_um / d, floor_frac, max_frac). At ref_um (the [CO22-TCPMT] first-neighbour
    distance) it equals the nearest-neighbour crosstalk; it never falls below the measured long-range floor."""
    d = np.maximum(np.asarray(d_um, dtype=np.float64), 1e-6)
    c = np.asarray(nearest_frac, float) * float(ref_um) / d
    return np.clip(c, np.asarray(floor_frac, float), float(max_frac))


# =============================================================================================================
# The assembled system
# =============================================================================================================
@dataclass(frozen=True, eq=False)
class ThermalSystem:
    """The assembled MIMO thermal plant of one bank for B uncertainty worlds (07 P1-2, P0-4).

    Arrays (numba-ready): taus_s, res_k_per_w (B, M) s and K/W; lag_src, lag_obs, lag_kind, lag_pair, lag_layer,
    lag_tail (M,) int64, lags sorted by observer with obs_ptr (O+1,); src_kind, src_ring (S,) and obs_kind, obs_ring
    (O,) int64 (ring -1 for free sites); per-ring index maps ring_obs, heater_obs, pd_obs, heater_src, wg_src,
    junction_src, driver_src (N,) (-1 when absent); temp_obs (T,), aggressor_src (A,); opt_lags (M_opt,) the lags of
    kind LAG_OPT_SELF or LAG_OPT_OTHER (the second pass of selfheat.implicit_step). Names: src_names, obs_names.
    transport_delay_s (B,) s: ThermalScales.transport_delay_s, for the heater actuator FIFO (not applied here).
    """
    taus_s: np.ndarray
    res_k_per_w: np.ndarray
    lag_src: np.ndarray
    lag_obs: np.ndarray
    lag_kind: np.ndarray
    lag_pair: np.ndarray
    lag_layer: np.ndarray
    lag_tail: np.ndarray
    obs_ptr: np.ndarray
    src_kind: np.ndarray
    src_ring: np.ndarray
    src_names: tuple
    obs_kind: np.ndarray
    obs_ring: np.ndarray
    obs_names: tuple
    ring_obs: np.ndarray
    heater_obs: np.ndarray
    pd_obs: np.ndarray
    temp_obs: np.ndarray
    heater_src: np.ndarray
    wg_src: np.ndarray
    junction_src: np.ndarray
    driver_src: np.ndarray
    aggressor_src: np.ndarray
    opt_lags: np.ndarray
    n_rings: int
    meta: dict = field(default_factory=dict)
    transport_delay_s: np.ndarray | None = None

    @property
    def batch(self) -> int:
        return self.taus_s.shape[0]

    @property
    def n_lags(self) -> int:
        return self.taus_s.shape[1]

    @property
    def n_src(self) -> int:
        return self.src_kind.shape[0]

    @property
    def n_obs(self) -> int:
        return self.obs_kind.shape[0]

    def coefficients(self, h_s: float):
        """Exact FOH coefficients (a, c0, c1), each (B, M), for a step h_s (s). See foh_coefficients."""
        return foh_coefficients(self.taus_s, self.res_k_per_w, h_s)

    def zero_state(self, batch: int | None = None) -> np.ndarray:
        """All-zero lag states (B, M) (every observer at T_base)."""
        return np.zeros((self.batch if batch is None else int(batch), self.n_lags))

    def steady_state(self, u_w) -> np.ndarray:
        """Lag states (B, M) in steady state for constant source powers u_w (B|1, S) W: x_m = r_m u_src(m)."""
        u = np.atleast_2d(np.asarray(u_w, dtype=np.float64))
        return self.res_k_per_w * u[:, self.lag_src]

    def outputs(self, x, t_base_k=0.0) -> np.ndarray:
        """Observer temperatures (B, O) = t_base_k + sum of lag states (Python form of thermal_outputs)."""
        x = np.atleast_2d(x)
        out = np.zeros((x.shape[0], self.n_obs))
        thermal_outputs(x, self.obs_ptr, np.atleast_1d(np.asarray(t_base_k, np.float64)), out)
        return out

    def source_index(self, kind: str, ring: int | None = None, name: str | None = None) -> int:
        """Index of a source by kind and ring (or by name for aggressors)."""
        return _find(self.src_names, self.src_kind, self.src_ring, SOURCE_KINDS, kind, ring, name)

    def observer_index(self, kind: str, ring: int | None = None, name: str | None = None) -> int:
        """Index of an observer by kind and ring (or by name for temperature sensors)."""
        return _find(self.obs_names, self.obs_kind, self.obs_ring, OBSERVER_KINDS, kind, ring, name)

    def kernel_arrays(self):
        """(lag_src, lag_obs, lag_kind, opt_lags, obs_ptr, src_ring, ring_obs): the index arrays of
        selfheat.implicit_step, in its argument order."""
        return (self.lag_src, self.lag_obs, self.lag_kind, self.opt_lags, self.obs_ptr, self.src_ring,
                self.ring_obs)


def _find(names, kinds, rings, kind_list, kind, ring, name):
    if name is not None:
        if name in names:
            return names.index(name)
        raise KeyError(f"no element named {name!r}")
    k = kind_list.index(kind)
    hit = np.flatnonzero((kinds == k) & ((rings == ring) if ring is not None else True))
    if hit.size != 1:
        raise KeyError(f"{kind} ring={ring}: {hit.size} matches")
    return int(hit[0])


# =============================================================================================================
# Assembly
# =============================================================================================================
def _as_b(v, name):
    """A scalar, (B,) or (B, 1) parameter as a 1-D array of length 1 or B."""
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 0:
        return a.reshape(1)
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0]
    if a.ndim != 1:
        raise ValueError(f"{name}: expected a scalar or (B,), got shape {a.shape}")
    return a


def _batch_of(values: dict) -> int:
    b = 1
    for name, a in values.items():
        n = a.shape[0]
        if n != 1:
            if b not in (1, n):
                raise ValueError(f"inconsistent batch sizes: {name} has {n}, another has {b}")
            b = n
    return b


def _hyp_codes(h, B):
    if isinstance(h, str):
        h = [h]
    arr = np.asarray(h)
    if arr.dtype.kind in "iu":
        codes = arr.astype(np.int64).reshape(-1)
    else:
        try:
            codes = np.array([SCALING_HYPOTHESES.index(str(x)) for x in arr.reshape(-1)], np.int64)
        except ValueError as e:
            raise ValueError(f"scaling_hypothesis must be one of {SCALING_HYPOTHESES}, got {h!r}") from e
    if np.any((codes < 0) | (codes >= len(SCALING_HYPOTHESES))):
        raise ValueError(f"scaling_hypothesis codes out of range: {codes}")
    if codes.shape[0] not in (1, B):
        raise ValueError(f"scaling_hypothesis has {codes.shape[0]} entries, batch is {B}")
    return np.broadcast_to(codes, (B,)).copy()


class _Lags:
    """Collects lags (one first-order term of one source -> observer pair) before sorting and scaling."""

    def __init__(self, B):
        self.B = B
        self.obs, self.src, self.tau, self.res = [], [], [], []
        self.layer, self.pair, self.tailable, self.tail, self.fixed_tau = [], [], [], [], []

    def add(self, obs, src, tau, res, layer, pair, tailable=False, tail=False, fixed_tau=False):
        self.obs.append(obs)
        self.src.append(src)
        self.tau.append(np.broadcast_to(np.asarray(tau, float), (self.B,)).copy())
        self.res.append(np.broadcast_to(np.asarray(res, float), (self.B,)).copy())
        self.layer.append(layer)
        self.pair.append(pair)
        self.tailable.append(bool(tailable))
        self.tail.append(bool(tail))
        self.fixed_tau.append(bool(fixed_tau))    # absolute tau: no time_scale, no 'resistance' factor


class _Elements:
    """Source and observer instances of the bank with their positions (um) and ring attachment."""

    def __init__(self):
        self.src_kind, self.src_ring, self.src_names, self.src_xy = [], [], [], []
        self.obs_kind, self.obs_ring, self.obs_names, self.obs_xy = [], [], [], []

    def add_src(self, kind, ring, name, xy):
        self.src_kind.append(kind)
        self.src_ring.append(ring)
        self.src_names.append(name)
        self.src_xy.append(np.asarray(xy, float))
        return len(self.src_kind) - 1

    def add_obs(self, kind, ring, name, xy):
        self.obs_kind.append(kind)
        self.obs_ring.append(ring)
        self.obs_names.append(name)
        self.obs_xy.append(np.asarray(xy, float))
        return len(self.obs_kind) - 1


def _build_elements(layout: LayoutSites, n, src_kinds, obs_kinds, pd_offset, ts_offset, driver_offset=None):
    el = _Elements()
    idx = {k: np.full(n, -1, np.int64) for k in ("heater_src", "wg_src", "junction_src", "driver_src",
                                                    "ring_obs", "heater_obs", "pd_obs")}
    ring_xy = layout.ring_xy_um
    for i in range(n):
        if "heater" in src_kinds:
            idx["heater_src"][i] = el.add_src(SRC_HEATER, i, f"heater{i}", ring_xy[i])
        if "waveguide_absorption" in src_kinds:
            idx["wg_src"][i] = el.add_src(SRC_WG, i, f"waveguide_absorption{i}", ring_xy[i])
        if "junction" in src_kinds:
            idx["junction_src"][i] = el.add_src(SRC_JUNCTION, i, f"junction{i}", ring_xy[i])
        if "driver" in src_kinds and (layout.driver_xy_um is not None or driver_offset is not None):
            dxy = (layout.driver_xy_um[i] if layout.driver_xy_um is not None
                   else ring_xy[i] + np.asarray(driver_offset, float))
            if np.all(np.isfinite(dxy)):          # a NaN row: this ring has no driver site
                idx["driver_src"][i] = el.add_src(SRC_DRIVER, i, f"driver{i}", dxy)
    agg = []
    if "aggressor" in src_kinds and layout.aggressor_xy_um is not None:
        for a, (nm, xy) in enumerate(zip(layout.aggressor_names, layout.aggressor_xy_um)):
            agg.append(el.add_src(SRC_AGGRESSOR, -1, nm, xy))
    for i in range(n):
        if "ring_mode_weighted" in obs_kinds:
            idx["ring_obs"][i] = el.add_obs(OBS_RING, i, f"ring_mode_weighted{i}", ring_xy[i])
        if "heater_resistor" in obs_kinds:
            idx["heater_obs"][i] = el.add_obs(OBS_HEATER, i, f"heater_resistor{i}", ring_xy[i])
        if "monitor_pd" in obs_kinds:
            pxy = layout.pd_xy_um[i] if layout.pd_xy_um is not None else np.full(2, np.nan)
            if not np.all(np.isfinite(pxy)):      # no PD site for this ring: the model's default offset
                pxy = ring_xy[i] + np.asarray(pd_offset, float)
            idx["pd_obs"][i] = el.add_obs(OBS_PD, i, f"monitor_pd{i}", pxy)
    ts = []
    if "temp_sensor" in obs_kinds and layout.temp_sensor_xy_um is not None:
        for t, nm in enumerate(layout.temp_sensor_names):
            r = int(layout.temp_sensor_ring[t])
            # an attached sensor sits at its ring plus the model's default offset (the offset of the ROM self table)
            xy = ring_xy[r] + np.asarray(ts_offset, float) if r >= 0 else layout.temp_sensor_xy_um[t]
            ts.append(el.add_obs(OBS_TEMP, r, nm, xy))
    idx["temp_obs"] = np.array(ts, np.int64)
    idx["aggressor_src"] = np.array(agg, np.int64)
    return el, idx


def _ranks(ring_xy):
    """Symmetric neighbour rank between rings: 1 = nearest distance of either ring, 2 = next distinct distance."""
    n = ring_xy.shape[0]
    d = np.linalg.norm(ring_xy[:, None, :] - ring_xy[None, :, :], axis=-1)
    rank = np.full((n, n), 99, np.int64)
    for i in range(n):
        others = np.delete(np.arange(n), i)
        if others.size == 0:
            continue
        uniq = []
        for v in np.sort(d[i, others]):
            if not uniq or v > uniq[-1] * (1 + 1e-6) + 1e-9:
                uniq.append(v)
        for j in others:
            for r, v in enumerate(uniq[:2]):
                if abs(d[i, j] - v) <= v * 1e-6 + 1e-9:
                    rank[i, j] = r + 1
    rank = np.minimum(rank, rank.T)
    np.fill_diagonal(rank, 0)
    return rank, d


def _interim_lags(spec: InterimRom, el: _Elements, layout: LayoutSites, n: int, B: int, p: dict, lg: _Lags):
    """Interim residues and poles (07 P1-2) for every source/observer pair. p: parameter arrays (B, N)."""
    z0 = p["z0"]                                  # (B, N) K/W
    z0_mean = z0.mean(axis=1)
    zr = p["z_ratio"]
    f_f, f_t = p["share_fast"], p["tail_share"]
    f_s = 1.0 - f_f - f_t
    tau_f, tau_s, tau_t = p["tau_fast"], p["tau_slow"], p["tail_tau"]
    f_sh = p["f_sh"]
    hr = p["heater_self_ratio"]
    rank, dring = _ranks(layout.ring_xy_um)

    def xt_tau(d):
        return crosstalk_tau_s(d, p["xt_tau_ref"][:, 0], spec.xt_dist_ref_um, p["xt_tau_min"][:, 0],
                               None if spec.xt_tau_far_s is None else p["xt_tau_far"][:, 0], spec.xt_far_um,
                               spec.xt_tau_law)

    def site_c(d):
        return site_fraction(d, p["xt_nearest"][:, 0], p["xt_floor"][:, 0], spec.site_ref_um, spec.site_max_frac)

    def add_shape(o, s, scale, i, pair, with_tail=True):
        """Z = scale * S_H(s) for ring i's self-path shape."""
        lg.add(o, s, tau_f[:, i], scale * f_f[:, i], LAYER_FAST, pair, tailable=True)
        lg.add(o, s, tau_s[:, i], scale * f_s[:, i], LAYER_FAST, pair, tailable=True)
        if with_tail:
            lg.add(o, s, tau_t[:, i], scale * f_t[:, i], LAYER_SLOW, pair, tailable=True, tail=True)

    ringlike = (SRC_WG, SRC_JUNCTION)
    n_src = len(el.src_kind)
    n_obs = len(el.obs_kind)
    for s in range(n_src):
        sk, sr, sxy = el.src_kind[s], el.src_ring[s], el.src_xy[s]
        for o in range(n_obs):
            ok, orr, oxy = el.obs_kind[o], el.obs_ring[o], el.obs_xy[o]
            local_src = sk in (SRC_HEATER, SRC_WG, SRC_JUNCTION)
            local_obs = ok in (OBS_RING, OBS_HEATER)
            if sr >= 0 and orr >= 0:
                pair = PAIR_SELF if sr == orr else PAIR_CROSS
            else:
                pair = PAIR_SITE
            if pair == PAIR_SELF and local_src and local_obs:
                i = sr
                if sk == SRC_HEATER and ok == OBS_RING:
                    add_shape(o, s, z0[:, i], i, pair)
                elif sk == SRC_HEATER and ok == OBS_HEATER:
                    add_shape(o, s, z0[:, i], i, pair)
                    lg.add(o, s, p["heater_node_tau"][:, i], (hr[:, i] - 1.0) * z0[:, i], LAYER_FAST, pair)
                elif sk in ringlike and ok == OBS_RING:
                    lg.add(o, s, p["selfheat_tau"][:, i], zr[:, i] * f_sh[:, i] * z0[:, i], LAYER_FAST, pair)
                    add_shape(o, s, zr[:, i] * (1.0 - f_sh[:, i]) * z0[:, i], i, pair)
                else:   # ring-like source -> heater resistor: reciprocal of heater -> ring
                    add_shape(o, s, z0[:, i], i, pair)
                continue
            if pair == PAIR_CROSS and local_src and local_obs:
                j, i = sr, orr                      # source ring j -> observer ring i
                rk = rank[i, j]
                c = (p["xt_nearest"][:, 0] if rk == 1 else
                     (p["xt_next"][:, 0] if (rk == 2 and spec.xt_next_frac is not None) else p["xt_floor"][:, 0]))
                if sk == SRC_HEATER and ok == OBS_RING:
                    zf = z0[:, j]
                elif sk in ringlike and ok == OBS_HEATER:
                    zf = z0[:, i]
                else:                                # heater<->heater or ring-like<->ring: symmetric
                    zf = np.sqrt(z0[:, i] * z0[:, j])
                if spec.crosstalk_dynamics == "self":
                    add_shape(o, s, c * zf, j, pair)
                else:
                    lg.add(o, s, xt_tau(dring[i, j]), c * zf, LAYER_SLOW, pair)
                continue
            # site law: monitor PDs, drivers, temperature sensors, aggressors
            d = float(np.linalg.norm(sxy - oxy))
            if sr >= 0:
                zf = z0[:, sr]
            elif orr >= 0:
                zf = z0[:, orr]
            else:
                zf = z0_mean
            lg.add(o, s, xt_tau(d), site_c(d) * zf, LAYER_SLOW, pair)


def _rom_lags(rom: ThermalRom, el: _Elements, layout: LayoutSites, B: int, tau_split: float, lg: _Lags, meta: dict,
              tol_um: float = 1.0):   # assumed: 1 um, well below the site sizes and offsets (tens of um)
    """FEM ROM residues: self_res for same-ring pairs, distance-interpolated cross_res otherwise (ARCHITECTURE 5.3).

    self_res was solved with the site elements (driver source, monitor_pd and temp_sensor observers) at the ROM's
    default offsets (site_offsets_um). A same-ring pair whose site element sits more than tol_um away from that offset
    (for example one shared driver block far from the rings) uses cross_res at its actual source-to-observer distance
    instead, with a warning."""
    taus = np.asarray(rom.taus_s, float)
    site_src = {SRC_DRIVER: "driver"}
    site_obs = {OBS_PD: "monitor_pd", OBS_TEMP: "temp_sensor"}
    moved = set()

    def off_ok(name, xy, ring):
        if name not in rom.site_offsets_um:
            return True                    # no default known: keep the self table (warned at assembly)
        dev = np.linalg.norm(xy - layout.ring_xy_um[ring] - np.asarray(rom.site_offsets_um[name], float))
        if dev > tol_um:
            moved.add(name)
            return False
        return True
    layer = np.where(taus < tau_split, LAYER_FAST, LAYER_SLOW)
    smap = {SOURCE_KINDS.index(k): i for i, k in enumerate(rom.sources)}
    omap = {OBSERVER_KINDS.index(k): i for i, k in enumerate(rom.observers)}
    skipped, out_of_grid = set(), []
    rgrid = np.asarray(rom.cross_r_um, float)

    def interp(si, oi, d):
        if rgrid.size == 0:
            return np.zeros_like(taus)
        if d < rgrid[0] or d > rgrid[-1]:
            out_of_grid.append(d)
        return np.array([np.interp(d, rgrid, rom.cross_res[si, oi, :, k]) for k in range(taus.size)])

    for s in range(len(el.src_kind)):
        sk, sr, sxy = el.src_kind[s], el.src_ring[s], el.src_xy[s]
        if sk not in smap:
            skipped.add(SOURCE_KINDS[sk])
            continue
        si = smap[sk]
        for o in range(len(el.obs_kind)):
            ok, orr, oxy = el.obs_kind[o], el.obs_ring[o], el.obs_xy[o]
            if ok not in omap:
                skipped.add(OBSERVER_KINDS[ok])
                continue
            oi = omap[ok]
            same = sr >= 0 and orr >= 0 and sr == orr
            if same and (sk not in site_src or off_ok(site_src[sk], sxy, sr)) and (
                    ok not in site_obs or off_ok(site_obs[ok], oxy, orr)):
                r, pair = np.asarray(rom.self_res[si, oi], float), PAIR_SELF
            elif same:
                r, pair = interp(si, oi, float(np.linalg.norm(sxy - oxy))), PAIR_SITE
            else:
                # source centre (ring centre for ring sources, site centre for drivers/aggressors) to observer centre
                # (ring centre for ring/heater observers, site centre for PDs/sensors): the extraction's convention
                r = interp(si, oi, float(np.linalg.norm(sxy - oxy)))
                pair = PAIR_CROSS if (sr >= 0 and orr >= 0) else PAIR_SITE
            for k in range(taus.size):
                if r[k] != 0.0:
                    lg.add(o, s, taus[k], r[k], int(layer[k]), pair, tailable=(pair == PAIR_SELF))
    if skipped:
        meta["rom_skipped_kinds"] = sorted(skipped)
    if moved:
        meta["rom_sites_off_default"] = sorted(moved)
        warnings.warn(f"layout {sorted(moved)} sites are more than {tol_um} um from the ROM self-table offsets "
                      f"{ {k: rom.site_offsets_um[k] for k in sorted(moved)} } um: their same-ring pairs use the "
                      "distance table cross_res at the actual site distance", RuntimeWarning, stacklevel=3)
    if out_of_grid:
        warnings.warn(f"thermal ROM cross_r_um grid {rgrid[0]}..{rgrid[-1]} um does not cover distances "
                      f"{sorted(set(round(v, 2) for v in out_of_grid))[:6]} um; clamped to the grid ends "
                      "(radially symmetric table, valid for d > 2R)", RuntimeWarning, stacklevel=3)


def _apply_tail(lg: _Lags, tail_tau, tail_share):
    """Replace or add the package tail on every tailable pair, preserving each pair's DC (07 P1-2, 4.8).

    Tailable pairs: the self pairs (and, with crosstalk_dynamics='self', the ring-to-ring pairs of the textbook
    corner, which use the self shape). The tail tau is absolute (fixed_tau: not rescaled later)."""
    pairs: dict = {}
    for m in range(len(lg.obs)):
        if lg.tailable[m]:
            pairs.setdefault((lg.obs[m], lg.src[m]), []).append(m)
    for (o, s), ms in pairs.items():
        pair_type = lg.pair[ms[0]]
        dc = np.sum([lg.res[m] for m in ms], axis=0)
        body = [m for m in ms if not lg.tail[m]]
        tails = [m for m in ms if lg.tail[m]]
        body_sum = np.sum([lg.res[m] for m in body], axis=0) if body else np.zeros(lg.B)
        with np.errstate(divide="ignore", invalid="ignore"):
            fac = np.where(body_sum != 0.0, dc * (1.0 - tail_share) / body_sum, 0.0)
        for m in body:
            lg.res[m] = lg.res[m] * fac
        if tails:
            m = tails[0]
            lg.res[m] = tail_share * dc
            lg.tau[m] = np.broadcast_to(tail_tau, (lg.B,)).copy()
            lg.fixed_tau[m] = True
            for extra in tails[1:]:
                lg.res[extra] = np.zeros(lg.B)
        else:
            lg.add(o, s, tail_tau, tail_share * dc, LAYER_SLOW, pair_type, tailable=True, tail=True, fixed_tau=True)


def _dc_factor(dc_scale, sk_name, ok_name, B):
    if not isinstance(dc_scale, Mapping):
        return np.broadcast_to(_as_b(dc_scale, "dc_scale"), (B,))
    f = np.ones(B)
    for key, v in dc_scale.items():
        if key == "*":
            f = f * _as_b(v, "dc_scale[*]")
            continue
        if "->" not in key:
            raise ValueError(f"dc_scale key {key!r} must be '*' or '<source>-><observer>'")
        a, b = (t.strip() for t in key.split("->", 1))
        for t, lst in ((a, SOURCE_KINDS), (b, OBSERVER_KINDS)):
            if t != "*" and t not in lst:
                raise ValueError(f"dc_scale key {key!r}: unknown kind {t!r}")
        if a in ("*", sk_name) and b in ("*", ok_name):
            f = f * _as_b(v, f"dc_scale[{key}]")
    return np.broadcast_to(f, (B,))


def assemble(rom, layout, n_rings: int, scales=None, *, sources: Sequence[str] | None = None,
             observers: Sequence[str] | None = None, temp_sensors: Sequence[Mapping] | None = None,
             ring_pitch_um: float | None = None, prune_abs_k_per_w: float = 0.0) -> ThermalSystem:
    """Assemble the MIMO thermal system of a bank of n_rings for B worlds (07 P1-2, P0-4 step 3).

    rom: a ThermalRom (FEM artefact) or an InterimRom (07 P1-2 interim numbers).
    layout: LayoutSites, a ``layout.sites`` mapping (parsed with LayoutSites.from_sites, using ring_pitch_um and
        temp_sensors), or None for a single ring at the origin.
    scales: ThermalScales or a mapping of its fields; None = all factors 1 and the 'resistance' hypothesis.
    sources / observers: kind names to include (default: all; drivers need driver sites, or InterimRom.driver_offset_um,
        or for a ThermalRom an explicit "driver" in ``sources`` (then the ROM's default offset); aggressors and
        temperature sensors need their sites).
    prune_abs_k_per_w: drop lags whose |residue| is at most this in every world (0: drop only exact zeros). Cost
        scales with the lag count M (about 4-10 ns per lag per substep): a FEM ROM with K poles has up to
        K x (sources) x (observers) lags, so restrict ``observers``/``sources`` to what the run reads.

    Superposition: every source of every ring couples to every observer; same-ring pairs use the self table (d = 0),
    ring-to-ring pairs the cross residues at the centre distance, free sites the distance law. Scaling (per world):
    dc_scale per (source, observer) group, crosstalk_scale on ring-to-ring residues, time_scale_fast/slow per layer,
    tau also times the dc_scale factor under 'resistance' (never the crosstalk scale), and a package tail from
    ThermalScales (DC preserving, absolute tau). B is the common length of every (B,) input, including a
    per-world scaling_hypothesis list. Returns a ThermalSystem with taus_s and res_k_per_w of shape (B, M).
    """
    n = int(n_rings)
    if layout is None:
        if n != 1:
            raise ValueError("layout is required for more than one ring")
        layout = LayoutSites(np.zeros((1, 2)))
    elif not isinstance(layout, LayoutSites):
        layout = LayoutSites.from_sites(layout, n, ring_pitch_um, temp_sensors)
    if layout.n_rings != n:
        raise ValueError(f"layout has {layout.n_rings} rings, n_rings is {n}")
    if scales is None:
        scales = ThermalScales()
    elif isinstance(scales, Mapping):
        scales = ThermalScales(**scales)
    src_kinds = tuple(SOURCE_KINDS if sources is None else sources)
    obs_kinds = tuple(OBSERVER_KINDS if observers is None else observers)
    for k in src_kinds:
        if k not in SOURCE_KINDS:
            raise ValueError(f"unknown source kind {k!r}")
    for k in obs_kinds:
        if k not in OBSERVER_KINDS:
            raise ValueError(f"unknown observer kind {k!r}")

    sc = {"time_scale_fast": _as_b(scales.time_scale_fast, "time_scale_fast"),
          "time_scale_slow": _as_b(scales.time_scale_slow, "time_scale_slow"),
          "crosstalk_scale": _as_b(scales.crosstalk_scale, "crosstalk_scale")}
    if scales.package_tail_share_frac is not None:
        sc["pkg_tail_share"] = _as_b(scales.package_tail_share_frac, "package_tail_share_frac")
        # default tau when only the share is drawn: assumed 4 ms, [CO22-TCPMT] circuit-level start-up (04 T2)
        sc["pkg_tail_tau"] = _as_b(4.0e-3 if scales.package_tail_tau_s is None else scales.package_tail_tau_s,
                               "package_tail_tau_s")
    if isinstance(scales.dc_scale, Mapping):
        for k, v in scales.dc_scale.items():
            sc[f"dc[{k}]"] = _as_b(v, f"dc_scale[{k}]")
    else:
        sc["dc"] = _as_b(scales.dc_scale, "dc_scale")
    sc["transport_delay_s"] = _as_b(scales.transport_delay_s, "transport_delay_s")
    if np.any(~(sc["transport_delay_s"] >= 0.0)):
        raise ValueError("transport_delay_s must be >= 0")
    hyp_raw = scales.scaling_hypothesis
    sc["scaling_hypothesis"] = np.zeros(1 if isinstance(hyp_raw, str) else np.asarray(hyp_raw).reshape(-1).shape[0])
    meta: dict = {"n_rings": n}

    if isinstance(rom, InterimRom):
        p = _interim_params(rom, n)
        B = _batch_of({**sc, **{k: v[:, 0] if v.shape[1] == 1 else v for k, v in p.items()}})
        p = {k: np.broadcast_to(v, (B, n)) for k, v in p.items()}
        el, idx = _build_elements(layout, n, src_kinds, obs_kinds, rom.pd_offset_um, rom.temp_sensor_offset_um,
                                  rom.driver_offset_um)
        lg = _Lags(B)
        _interim_lags(rom, el, layout, n, B, p, lg)
        meta.update(model="interim", name=rom.name, crosstalk_dynamics=rom.crosstalk_dynamics,
                    xt_tau_law=rom.xt_tau_law)
        bad = (p["z_ratio"] * (1.0 - p["f_sh"]) < 1.0 - 1e-12)
        if np.any(bad):
            warnings.warn("interim thermal model: z_ratio*(1 - selfheat_share) < 1, so the heater/ring residue matrix "
                          "of the heater poles is not positive semidefinite (not a passive RC network); "
                          "selfheat_share_frac = (z - 1)/z restores it", RuntimeWarning, stacklevel=2)
    elif isinstance(rom, ThermalRom):
        B = _batch_of(sc)
        for nm in ("monitor_pd", "temp_sensor"):
            if nm in obs_kinds and nm in rom.observers and nm not in rom.site_offsets_um:
                warnings.warn(f"thermal ROM has no site offset for {nm}; its sites are placed at the ring centre",
                              RuntimeWarning, stacklevel=2)
        # drivers, as in the interim model: only where the layout places driver sites (G26 drives from a separate
        # EIC beside the PIC, 07 P1-3), or at the ROM's default offset when the caller lists "driver" explicitly
        drv_off = (rom.site_offsets_um.get("driver") if ("driver" in rom.sources and sources is not None
                                                         and "driver" in sources) else None)
        el, idx = _build_elements(layout, n, src_kinds, obs_kinds,
                                  rom.site_offsets_um.get("monitor_pd", (0.0, 0.0)),
                                  rom.site_offsets_um.get("temp_sensor", (0.0, 0.0)), drv_off)
        lg = _Lags(B)
        _rom_lags(rom, el, layout, B, float(scales.tau_split_s), lg, meta)
        meta.update(model="rom", rom_path=rom.meta.get("path"), rom_sha256=rom.meta.get("sha256"),
                    family=rom.meta.get("family"), film_nm=rom.meta.get("film_nm"))
    else:
        raise TypeError(f"rom must be a ThermalRom or an InterimRom, got {type(rom).__name__}")

    if "pkg_tail_share" in sc:
        ts = np.broadcast_to(sc["pkg_tail_share"], (B,))
        if np.any((ts < 0) | (ts >= 1)):
            raise ValueError("package_tail_share_frac must be in [0, 1)")
        _apply_tail(lg, np.broadcast_to(sc["pkg_tail_tau"], (B,)), ts)

    M = len(lg.obs)
    if M == 0:
        raise ValueError("the assembled thermal system has no lags (no sources or observers)")
    tau = np.stack(lg.tau, axis=1)
    res = np.stack(lg.res, axis=1)
    obs = np.array(lg.obs, np.int64)
    src = np.array(lg.src, np.int64)
    layer = np.array(lg.layer, np.int64)
    pair = np.array(lg.pair, np.int64)
    tail = np.array(lg.tail, np.int64)
    fixed = np.array(lg.fixed_tau, bool)
    src_kind = np.array(el.src_kind, np.int64)
    src_ring = np.array(el.src_ring, np.int64)
    obs_kind = np.array(el.obs_kind, np.int64)
    obs_ring = np.array(el.obs_ring, np.int64)

    # ---- per-world scaling
    hyp = _hyp_codes(scales.scaling_hypothesis, B)
    sigma = np.ones((B, M))
    for m in range(M):
        sigma[:, m] = _dc_factor(scales.dc_scale, SOURCE_KINDS[src_kind[src[m]]], OBSERVER_KINDS[obs_kind[obs[m]]], B)
    xt = np.ones((B, M))
    xt[:, pair == PAIR_CROSS] = np.broadcast_to(sc["crosstalk_scale"], (B,))[:, None]
    s_t = np.where(layer[None, :] == LAYER_FAST, np.broadcast_to(sc["time_scale_fast"], (B,))[:, None],
                   np.broadcast_to(sc["time_scale_slow"], (B,))[:, None])
    resistance = (hyp == SCALING_HYPOTHESES.index("resistance"))[:, None]
    # tau follows the conductance factor sigma under 'resistance' (tau = RC); crosstalk_scale is a magnitude
    # uncertainty and never moves tau; a package tail given in ThermalScales keeps its absolute tau
    tau = np.where(fixed[None, :], tau, tau * s_t * np.where(resistance, sigma, 1.0))
    res = res * sigma * xt
    if np.any(~(tau > 0)) or not np.all(np.isfinite(tau)):
        raise ValueError("assembled time constants must be finite and > 0 (check time scales and dc_scale)")

    # ---- prune, sort by observer, then source, then tau
    keep = np.any(np.abs(res) > float(prune_abs_k_per_w), axis=0)
    if prune_abs_k_per_w == 0.0:
        keep = np.any(res != 0.0, axis=0)
    order = np.lexsort((tau[0], src, obs))
    order = order[keep[order]]
    tau, res, obs, src, layer, pair, tail = (tau[:, order], res[:, order], obs[order], src[order], layer[order],
                                             pair[order], tail[order])
    O = obs_kind.shape[0]
    obs_ptr = np.zeros(O + 1, np.int64)
    np.add.at(obs_ptr, obs + 1, 1)
    obs_ptr = np.cumsum(obs_ptr)

    kind = np.zeros(obs.shape[0], np.int64)
    opt = src_kind[src] == SRC_WG
    ring_o = obs_kind[obs] == OBS_RING
    same = src_ring[src] == obs_ring[obs]
    kind[opt & ring_o & same] = LAG_OPT_SELF
    kind[opt & ring_o & ~same] = LAG_OPT_CROSS_RING
    kind[opt & ~ring_o] = LAG_OPT_OTHER

    meta.update(scaling_hypothesis=[SCALING_HYPOTHESES[c] for c in np.unique(hyp)],
                tail_from_scales="pkg_tail_share" in sc, layout=layout)
    return ThermalSystem(
        np.ascontiguousarray(tau), np.ascontiguousarray(res), src, obs, kind, pair, layer, tail, obs_ptr,
        src_kind, src_ring, tuple(el.src_names), obs_kind, obs_ring, tuple(el.obs_names),
        idx["ring_obs"], idx["heater_obs"], idx["pd_obs"], idx["temp_obs"], idx["heater_src"], idx["wg_src"],
        idx["junction_src"], idx["driver_src"], idx["aggressor_src"],
        np.flatnonzero((kind == LAG_OPT_SELF) | (kind == LAG_OPT_OTHER)).astype(np.int64), n, meta,
        np.broadcast_to(sc["transport_delay_s"], (B,)).copy())


def _interim_params(spec: InterimRom, n: int) -> dict:
    """InterimRom fields as (B|1, N|1) arrays, with defaults resolved and bounds checked."""
    def bn(v, name):
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 0:
            return a.reshape(1, 1)
        if a.ndim == 1:
            return a[:, None]
        if a.ndim == 2 and a.shape[1] in (1, n):
            return a
        raise ValueError(f"InterimRom.{name}: expected scalar, (B,) or (B, N), got shape {a.shape}")

    p = {"z0": bn(spec.z_heater_ring_k_per_w, "z_heater_ring_k_per_w"),
         "z_ratio": bn(spec.z_ratio_opt_heater, "z_ratio_opt_heater"),
         "tau_fast": bn(spec.tau_fast_s, "tau_fast_s"), "share_fast": bn(spec.share_fast_frac, "share_fast_frac"),
         "tau_slow": bn(spec.tau_slow_s, "tau_slow_s"), "tail_tau": bn(spec.tail_tau_s, "tail_tau_s"),
         "tail_share": bn(spec.tail_share_frac, "tail_share_frac"),
         "selfheat_tau": bn(spec.selfheat_tau_s, "selfheat_tau_s"),
         "heater_self_ratio": bn(spec.heater_self_ratio, "heater_self_ratio"),
         "heater_node_tau": bn(spec.heater_node_tau_s, "heater_node_tau_s"),
         "xt_nearest": bn(spec.xt_nearest_frac, "xt_nearest_frac"), "xt_floor": bn(spec.xt_floor_frac, "xt_floor_frac"),
         "xt_next": bn(0.0 if spec.xt_next_frac is None else spec.xt_next_frac, "xt_next_frac"),
         "xt_tau_ref": bn(spec.xt_tau_ref_s, "xt_tau_ref_s"), "xt_tau_min": bn(spec.xt_tau_min_s, "xt_tau_min_s"),
         "xt_tau_far": bn(0.0 if spec.xt_tau_far_s is None else spec.xt_tau_far_s, "xt_tau_far_s")}
    z = p["z_ratio"]
    p["f_sh"] = (bn(spec.selfheat_share_frac, "selfheat_share_frac") if spec.selfheat_share_frac is not None
                 else (z - 1.0) / z)
    if np.any(p["z0"] <= 0):
        raise ValueError("z_heater_ring_k_per_w must be > 0")
    if np.any(z < 1.0):
        raise ValueError("z_ratio_opt_heater must be >= 1 (07 P0-1 item 5: Z_opt/Z_heater >= 1 by reciprocity)")
    if np.any(p["heater_self_ratio"] < 1.0):
        raise ValueError("heater_self_ratio must be >= 1 (R_th,H >= Z_heater->ring, 06 E1)")
    if np.any((p["f_sh"] < 0) | (p["f_sh"] > 1)):
        raise ValueError("selfheat_share_frac must be in [0, 1]")
    if np.any(p["share_fast"] < 0) or np.any(p["tail_share"] < 0) or np.any(p["share_fast"] + p["tail_share"] > 1):
        raise ValueError("share_fast_frac and tail_share_frac must be >= 0 with a sum <= 1")
    for k in ("tau_fast", "tau_slow", "tail_tau", "selfheat_tau", "heater_node_tau", "xt_tau_ref", "xt_tau_min"):
        if np.any(p[k] <= 0):
            raise ValueError(f"interim time constant {k} must be > 0")
    if spec.crosstalk_dynamics not in ("imec", "self"):
        raise ValueError("crosstalk_dynamics must be 'imec' or 'self'")
    return p


def interim_three_pole(layout=None, n_rings: int = 1, scales=None, *, sources=None, observers=None,
                       temp_sensors=None, ring_pitch_um=None, **preset) -> ThermalSystem:
    """Build the interim thermal ThermalSystem of 07 P1-2 from preset numbers: assemble(InterimRom(**preset), ...).

    preset: InterimRom fields (schema names ``ring.thermal.*`` in SI), e.g. z_heater_ring_k_per_w, tau_fast_s,
    share_fast_frac, tau_slow_s, tail_tau_s, tail_share_frac, selfheat_tau_s, selfheat_share_frac,
    z_ratio_opt_heater, xt_nearest_frac, xt_floor_frac, xt_tau_ref_s, ... See InterimRom for the equations.
    """
    return assemble(InterimRom(**preset), layout, n_rings, scales, sources=sources, observers=observers,
                    temp_sensors=temp_sensors, ring_pitch_um=ring_pitch_um)


# =============================================================================================================
# Python-level drivers and analysis helpers
# =============================================================================================================
def simulate(system: ThermalSystem, u_samples_w, h_s: float, *, hold: str = "zoh", x0=None, t_base_k=0.0,
             src_scale=None) -> np.ndarray:
    """Open-loop run with the exact kernels: returns observer temperatures (n+1, B, O) at t = 0, h, ..., n h.

    u_samples_w (n+1, B|1, S) W: input samples at the step boundaries. hold='zoh': step k holds u[k] over
    [k h, (k+1) h); hold='foh': u is linear between samples (exact for piecewise-linear inputs, 07 P0-10).
    """
    u = np.asarray(u_samples_w, dtype=np.float64)
    if u.ndim == 2:
        u = u[:, None, :]
    if hold not in ("zoh", "foh"):
        raise ValueError("hold must be 'zoh' or 'foh'")
    if u.ndim != 3 or u.shape[2] != system.n_src:
        raise ValueError(f"u_samples_w must be (n+1, B|1, S={system.n_src}), got {u.shape}")
    B = max(system.batch, u.shape[1])
    if system.batch not in (1, B) or u.shape[1] not in (1, B):
        raise ValueError(f"batch mismatch: system has {system.batch} worlds, u has {u.shape[1]}")
    a, c0, c1 = system.coefficients(h_s)
    x = np.zeros((B, system.n_lags)) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    tb = np.atleast_1d(np.asarray(t_base_k, np.float64))
    ss = np.ones((1, system.n_src)) if src_scale is None else np.atleast_2d(np.asarray(src_scale, np.float64))
    out = np.zeros((u.shape[0], B, system.n_obs))
    thermal_outputs(x, system.obs_ptr, tb, out[0])
    for k in range(u.shape[0] - 1):
        u1 = u[k] if hold == "zoh" else u[k + 1]
        thermal_step(x, a, c0, c1, system.lag_src, np.ascontiguousarray(u[k]), np.ascontiguousarray(u1), ss)
        thermal_outputs(x, system.obs_ptr, tb, out[k + 1])
    return out


def _select(system: ThermalSystem, obs, src):
    o_sel = np.arange(system.n_obs) if obs is None else np.atleast_1d(np.asarray(obs, np.int64))
    s_sel = np.arange(system.n_src) if src is None else np.atleast_1d(np.asarray(src, np.int64))
    o_map = np.full(system.n_obs, -1, np.int64)
    o_map[o_sel] = np.arange(o_sel.size)
    s_map = np.full(system.n_src, -1, np.int64)
    s_map[s_sel] = np.arange(s_sel.size)
    lio, lis = o_map[system.lag_obs], s_map[system.lag_src]
    m = np.flatnonzero((lio >= 0) & (lis >= 0))
    return o_sel, s_sel, m, lio[m], lis[m]


def _accumulate(system, obs, src, kernel):
    o_sel, s_sel, m, lio, lis = _select(system, obs, src)
    contrib = kernel(system.taus_s[:, m], system.res_k_per_w[:, m])           # (B, Msel, F)
    out = np.zeros((system.batch, o_sel.size, s_sel.size, contrib.shape[-1]), dtype=contrib.dtype)
    np.add.at(out, (slice(None), lio, lis, slice(None)), contrib)
    return out


def frequency_response(system: ThermalSystem, f_hz, obs=None, src=None) -> np.ndarray:
    """Complex transfer impedances Z(j 2 pi f) = sum_k r_k/(1 + j 2 pi f tau_k), shape (B, O', S', F), K/W.

    obs, src: index arrays selecting observers/sources (default: all). f_hz in Hz (cycles per second).
    """
    w = 2.0 * np.pi * np.atleast_1d(np.asarray(f_hz, np.float64))
    return _accumulate(system, obs, src,
                       lambda tau, r: r[..., None] / (1.0 + 1j * w[None, None, :] * tau[..., None]))


def step_response(system: ThermalSystem, t_s, obs=None, src=None) -> np.ndarray:
    """Unit-power step responses sum_k r_k (1 - exp(-t/tau_k)) in K/W, shape (B, O', S', T); t_s in s."""
    t = np.atleast_1d(np.asarray(t_s, np.float64))
    return _accumulate(system, obs, src, lambda tau, r: r[..., None] * -np.expm1(-t[None, None, :] / tau[..., None]))


def ramp_response(system: ThermalSystem, t_s, obs=None, src=None) -> np.ndarray:
    """Responses to a unit ramp u = t (W/s): sum_k r_k [t - tau_k (1 - exp(-t/tau_k))], (B, O', S', T) in K s/W."""
    t = np.atleast_1d(np.asarray(t_s, np.float64))
    return _accumulate(system, obs, src, lambda tau, r: r[..., None] * (
        t[None, None, :] + tau[..., None] * np.expm1(-t[None, None, :] / tau[..., None])))


def dc_gain(system: ThermalSystem, obs=None, src=None) -> np.ndarray:
    """Static transfer resistances Z(0) = sum_k r_k, shape (B, O', S') in K/W."""
    return _accumulate(system, obs, src, lambda tau, r: r[..., None])[..., 0]


def state_space(system: ThermalSystem, b: int = 0, obs=None, src=None):
    """Continuous state space of world b for the selected observers/sources (for linearize and validation).

    Returns (A, Bm, C, D): A = diag(-1/tau_m) (1/s), Bm[m, s] = r_m/tau_m (K/(W s)), C[o, m] = 1, D = 0, with one
    state per selected lag m (tau_m dx_m/dt = -x_m + r_m u_s). Outputs are rises above T_base in K.
    """
    o_sel, s_sel, m, lio, lis = _select(system, obs, src)
    tau = system.taus_s[b, m]
    r = system.res_k_per_w[b, m]
    A = np.diag(-1.0 / tau)
    Bm = np.zeros((m.size, s_sel.size))
    Bm[np.arange(m.size), lis] = r / tau
    C = np.zeros((o_sel.size, m.size))
    C[lio, np.arange(m.size)] = 1.0
    return A, Bm, C, np.zeros((o_sel.size, s_sel.size))


def reciprocity_error(system: ThermalSystem, f_hz=(0.0, 1e3, 1e4, 1e5, 1e6), b=None) -> float:
    """Largest relative reciprocity violation over like pairs (07 P1-2 test, 07 4.2), on |Z(j 2 pi f)|.

    Pairs checked for every ring i, j (including i = j where it is not trivial): heater_j -> heater_resistor_i vs
    heater_i -> heater_resistor_j; waveguide_j -> ring_i vs waveguide_i -> ring_j; heater_j -> ring_i vs
    waveguide_i -> heater_resistor_j. Returns max |Z_ab - Z_ba| / max|Z| over the pairs that exist.
    """
    H = frequency_response(system, f_hz)
    if b is not None:
        H = H[b:b + 1]
    n = system.n_rings
    num, scale = 0.0, 0.0
    combos = ((system.heater_src, system.heater_obs, system.heater_src, system.heater_obs),
              (system.wg_src, system.ring_obs, system.wg_src, system.ring_obs),
              (system.heater_src, system.ring_obs, system.wg_src, system.heater_obs))
    for s1, o1, s2, o2 in combos:
        for i in range(n):
            for j in range(n):
                if min(s1[j], o1[i], s2[i], o2[j]) < 0:
                    continue
                z1 = H[:, o1[i], s1[j]]
                z2 = H[:, o2[j], s2[i]]
                num = max(num, float(np.max(np.abs(z1 - z2))))
                scale = max(scale, float(np.max(np.abs(z1))), float(np.max(np.abs(z2))))
    return num / scale if scale > 0 else 0.0


def selfheat_dc_k_per_w(system: ThermalSystem) -> np.ndarray:
    """DC optical self-heating resistance Z_opt->ring,i(0) of each ring, (B, N) K/W (sum of LAG_OPT_SELF residues)."""
    out = np.zeros((system.batch, system.n_rings))
    m = np.flatnonzero(system.lag_kind == LAG_OPT_SELF)
    np.add.at(out, (slice(None), system.src_ring[system.lag_src[m]]), system.res_k_per_w[:, m])
    return out


def cross_optical_lag_gain(system: ThermalSystem, h_s: float, dpabs_dt_max_w_per_k) -> np.ndarray:
    """Per-substep gain of the lagged cross-ring optical heating (07 P0-1 item 2), (B, N).

    g_i = sum over LAG_OPT_CROSS_RING lags m into ring i of (1 - a_m) r_m * max|dP_abs,j/dT_j| (j = source ring),
    where dpabs_dt_max_w_per_k (B|1, N) W/K bounds |dP_abs/dT| of each ring (for one Lorentzian line:
    (3 sqrt(3)/8) P_abs,peak * (dlambda/dT)/HWHM, see selfheat.lorentzian_bounds). realize() asserts g < 0.05.
    """
    d = np.atleast_2d(np.asarray(dpabs_dt_max_w_per_k, np.float64))
    a, c0, c1 = system.coefficients(h_s)
    m = np.flatnonzero(system.lag_kind == LAG_OPT_CROSS_RING)
    out = np.zeros((max(system.batch, d.shape[0]), system.n_rings))
    if m.size == 0:
        return out
    j = system.src_ring[system.lag_src[m]]
    i = system.obs_ring[system.lag_obs[m]]
    contrib = np.abs(c0[:, m] + c1[:, m]) * np.broadcast_to(d, (out.shape[0], d.shape[1]))[:, j]
    np.add.at(out, (slice(None), i), contrib)
    return out


def assert_cross_lag_gain(system: ThermalSystem, h_s: float, dpabs_dt_max_w_per_k, limit: float = 0.05) -> float:
    """Raise ValueError if the lagged cross-ring optical gain reaches ``limit`` (07 P0-1: below 0.05). Returns max."""
    g = cross_optical_lag_gain(system, h_s, dpabs_dt_max_w_per_k)
    gmax = float(np.max(g)) if g.size else 0.0
    if gmax >= limit:
        b, i = np.unravel_index(int(np.argmax(g)), g.shape)
        raise ValueError(f"lagged cross-ring optical heating gain {gmax:.3g} >= {limit} at world {b}, ring {i} for a "
                         f"{h_s:g} s substep (07 P0-1); reduce the substep or solve the bank jointly")
    return gmax


def k_temp_residue_scale(t_k, *, film_nm: float = 160.0, weights=(0.15, 0.80, 0.05), t_ref_k: float = 300.0):
    """Quasi-static k(T) scaling of the thermal residues, R(T)/R(T_ref) (07 P2-5; 02 section 5).

    R = R_ref [w_film k_film(T_ref)/k_film(T) + w_ox k_ox(T_ref)/k_ox(T) + w_sub k_Si(T_ref)/k_Si(T)], with the
    in-plane film fit (Matthiessen, 02 section 5.2), oxide 1.38 (T/300)^0.33 and bulk Si 148 (T/300)^-1.3
    (Palankovski, 02 section 5.1, 5.3), taken from mrm.materials. weights = shares of the DC resistance in the film,
    the oxide and the substrate: assumed (no FEM sensitivity exists yet; the BOX path usually dominates).
    Apply as ``src_scale`` of the step kernels (per source, evaluated at a slowly varying temperature).
    """
    from mrm import materials as mat   # Python-level only (never inside a kernel)
    t = np.asarray(t_k, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (3,) or np.any(w < 0) or abs(w.sum() - 1.0) > 1e-9:
        raise ValueError("weights must be three non-negative shares (film, oxide, substrate) summing to 1")
    kf = np.asarray(mat.k_si_film_w_per_m_k(t, film_nm)) / float(mat.k_si_film_w_per_m_k(t_ref_k, film_nm))
    ko = np.asarray(mat.k_sio2_w_per_m_k(t)) / float(mat.k_sio2_w_per_m_k(t_ref_k))
    ks = np.asarray(mat.k_si_bulk_w_per_m_k(t)) / float(mat.k_si_bulk_w_per_m_k(t_ref_k))
    return w[0] / kf + w[1] / ko + w[2] / ks
