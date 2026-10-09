"""Parametric disturbance generators evaluated inside numba, the scenario data model and the engine contract.

Contract: docs/ARCHITECTURE.md section 2 (``scenarios/``) and, in docs/research/07_design_critique.md ("07"):
P0-9 (generators evaluated inside the kernel: piecewise-linear segments, square, sawtooth and burst envelopes with
phase, random walks from the counter RNG; ``DisturbanceFrame`` arrays only for recorded profiles), P0-10
(``time_compression`` for slow exogenous signals only, the load-time check and the mandatory convergence
experiment), P1-4 (four disturbance paths that are never merged), P0-11 (traffic events), P1-12 (line enable,
laser failover, fibre-loss step), P1-8 (transition density, swing and driver enable per tick), P1-10 (fault masks),
D6 (aggressor = power into the ROM). The engine-facing structure is documented in docs/handoff/scenarios.md.

Signals
-------
A scenario produces a fixed set of named signals (:data:`SIGNALS`, ids append-only). Each signal has a level that
sets its width per batch element: ``die`` (1), ``ring`` (N), ``line`` (L) or ``aggressor`` (A, one element per
aggressor site). The four disturbance paths of 07 P1-4 are separate signals and are never summed into one another:

1. ``thermal.t_base_k``: package/TEC base temperature (absolute, K). Seconds to hours; time-compressible.
2. ``thermal.aggressor_w``: on-die aggressor POWER (W) at each aggressor site, fed into the ROM aggressor column.
3. ``thermal.xpu_rise_k``: XPU/ASIC power-event temperature rise at the PIC (K, die-wide common mode).
4. ``laser.dfreq_hz``: laser line frequency offset (Hz), optical only, no thermal path (laser-sweep emulation,
   drift, failover).

Plus ``laser.power_delta_db``, ``laser.line_enable``, ``optical.input_loss_delta_db`` (re-mate / fibre loss), the
data-path signals of P0-11 and P1-8, the bias drive of P0-7 and the fault masks of P1-10.

Signal value = default (or 0 where an ``absolute`` component defines the level) + the sum of its components, then
clipped to the signal's [lo, hi]. Two absolute components on one element are a load error.

Generators (component kinds)
----------------------------
Every component maps time to a contribution ``offset + amp * map(raw(x))`` while it is active, where
``x = t_axis - t0 + phase`` and ``t_axis = t`` (or ``t * time_compression`` for a component flagged ``slow``).
Outside its window [t0, t1) it contributes 0, or holds its value at the window edge (``hold_before``,
``hold_after``). An optional burst envelope (``burst_on_s``, ``burst_off_s``, ``burst_phase_s``) gates it.

======================  ===========================================================================================
kind                    raw(x)
======================  ===========================================================================================
const                   1
pwl                     linear interpolation in a (t_i, v_i) table, ends held; periodic optional
trapezoid               0 -> 1 over ``rise``, 1 for ``high``, 1 -> 0 over ``fall``, 0 for the rest of ``period``
                        (square = rise = fall = 0; symmetric triangle = high = 0, rise = fall = P/2;
                        one-sided sawtooth = rise = P - fall, small fall)
rect_sine               \\|sin(pi x / P)\\| (the corrected slide-10 aggressor shape, 01 verification log item 5)
sine                    sin(2 pi x / P) (ripple)
smooth_step             (1 - cos(pi x / D)) / 2 on [0, D], 0 before, 1 after (XPU power-event edge)
random_walk             Brownian path with drift, linear between knots every ``knot_s``: W(0) = 0,
                        W_{j+1} = W_j + drift*knot + sigma*sqrt(knot)*z_j, z_j = N(0, 1) from the counter RNG
random_levels           a level held for each interval j = floor(x / interval): uniform, normal or a choice from
                        a table, drawn from the counter RNG (Sun et al. random ones density, connector re-mates)
sampled                 uniform-grid table (zoh or linear; periodic optional): calibrated aggressor waveforms and
                        recorded DisturbanceFrame profiles
======================  ===========================================================================================

Maps: ``identity`` and ``bernoulli_td`` (2 r (1 - r): the transition density of i.i.d. bits with ones density r).

Random draws use the counter RNG of mrm/core/rng.py: key = stream_key(seed_b, stream_id(stream)), counter
(tick = knot or interval index j, ring = element index, slot = the component's slot). seed_b = base + b
(rng.batch_seeds), so element b of a batch equals a B = 1 run with seed base + b. A draw never depends on the
evaluation order, the step pattern or the thread count; the per-element caches of the random walk and the random
levels only save work (07 P1-13).

Time semantics for the engine
-----------------------------
Every time field is converted to plant ticks when the scenario is compiled for a dt (a time within 1e-6 tick of a
whole tick is snapped onto it), and the kernels compute in ticks, so window, burst and waveform edges given in
seconds fall on whole ticks exactly. Signals are evaluated at tick instants (right-continuous). The defined input
over the plant interval [k, k+1) is the linear interpolation between the values at ticks k and k+1 (first-order
hold). A jump between two ticks therefore becomes a one-tick ramp. Fixed-tick and multi-tick (07 P0-10
``substep_policy: auto``) runs see the same input when the engine never steps past :func:`scen_next_break_b`, which
returns a tick up to which the tick-sampled signal is exactly linear (curved components -- smooth kinds and ramps
under the bernoulli_td map -- within the signal's ``foh_tol``; a signal whose sum may reach a clip limit, flagged
``SI_CLIP`` at compile, breaks every tick).

Cost: the helpers are inlined (numba ``inline='always'``); a non-inlined call with many array arguments costs tens
of ns. Signals without components are written once by :func:`scen_init_b` and skipped by :func:`scen_eval_b`.

Numba rules: the kernels take only arrays and scalars and read no module global except integer constants; they run
under NUMBA_DISABLE_JIT=1.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import warnings
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numba import njit

from mrm.core import rng as _rng

__all__ = [
    # errors
    "ScenarioError", "CompressionError",
    # signal table
    "SignalSpec", "SIGNALS", "SIGNAL_BY_NAME", "signal_id", "LEVELS",
    "SIG_T_BASE_K", "SIG_XPU_RISE_K", "SIG_AGGRESSOR_W", "SIG_LASER_DFREQ_HZ", "SIG_LASER_POWER_DELTA_DB",
    "SIG_LASER_LINE_ENABLE", "SIG_INPUT_LOSS_DELTA_DB", "SIG_ONES_DENSITY", "SIG_TRANSITION_DENSITY",
    "SIG_SWING_SCALE", "SIG_DRIVER_ENABLE", "SIG_BIAS_OFFSET_V", "SIG_BIAS_V", "SIG_FAULT_ADC_DROP",
    "SIG_FAULT_ADC_STUCK", "SIG_FAULT_ADC_SATURATE", "SIG_FAULT_DAC_GLITCH_CODES", "SIG_FAULT_CONTROLLER_RESET",
    # layout constants
    "KINDS", "K_CONST", "K_PWL", "K_TRAPEZOID", "K_RECT_SINE", "K_SINE", "K_SMOOTH_STEP", "K_RANDOM_WALK",
    "K_RANDOM_LEVELS", "K_SAMPLED", "MAPS", "DISTS",
    "CI_KIND", "CI_SIGNAL", "CI_E_LO", "CI_E_HI", "CI_FLAGS", "CI_MAP", "CI_TAB_OFF", "CI_TAB_LEN", "CI_SLOT",
    "CI_CACHE_OFF", "CI_DIST", "NCI",
    "CF_T0", "CF_T1", "CF_AMP", "CF_OFFSET", "CF_PERIOD", "CF_PHASE", "CF_P0", "CF_P1", "CF_P2", "CF_P3", "CF_P4",
    "CF_BURST_ON", "CF_BURST_OFF", "CF_BURST_PHASE", "CF_HSMOOTH", "NCF",
    "SI_OFF", "SI_WIDTH", "SI_C0", "SI_NC", "SI_CLIP", "NSI", "SF_LO", "SF_HI", "NSF",
    "FL_SLOW", "FL_ABSOLUTE", "FL_COMMON", "FL_HOLD_BEFORE", "FL_HOLD_AFTER", "FL_PERIODIC", "FL_LINEAR",
    "FL_PER_ELEMENT", "NO_BREAK",
    # kernels
    "scen_init_b", "scen_eval_b", "scen_eval_signal_b", "scen_next_break_b", "scen_eval", "scen_render",
    # numpy reference shapes (also used by calibrate and the library)
    "np_trapezoid", "np_rect_sine", "np_smooth_step",
    # data model
    "Component", "AggressorSpec", "ResetSpec", "Scenario", "ScenarioArrays", "DisturbanceFrame",
    "const", "step", "pulse", "pwl", "trapezoid", "square", "triangle", "sawtooth", "rect_sine", "sine",
    "smooth_step", "random_walk", "random_levels", "sampled", "with_burst", "stack",
    # time compression (07 P0-10)
    "FAST_RATE_REF_K_PER_S", "COMPRESSION_RATE_FRACTION", "component_max_rate", "check_time_compression",
    "CONVERGENCE_METRICS", "DEFAULT_CONVERGENCE_TOLERANCES", "ConvergenceCertificate", "ConvergenceReport",
    "convergence_plan", "assess_convergence", "require_compression_validated",
]


class ScenarioError(ValueError):
    """An invalid scenario definition (unknown signal, conflicting absolute components, pending calibration...)."""


class CompressionError(ScenarioError):
    """time_compression violates the 07 P0-10 load-time rule, or a compressed result lacks its convergence check."""


# =====================================================================================================================
# Signal table (append-only ids)
# =====================================================================================================================

LEVELS = ("die", "ring", "line", "aggressor")

#: Lightmatter's printed resonance sensitivity, used ONLY to express laser-frequency rates in the K/s units of the
#: 07 P0-10 compression check (an equivalence for a rate rule, not a plant parameter): ~15 GHz/degC, printed as an
#: aside in LM26-P section 1 (01 section 2.5, verified).
_LM_HZ_PER_K = 15.0e9


@dataclass(frozen=True)
class SignalSpec:
    """One scenario signal (the engine contract; docs/handoff/scenarios.md).

    id: fixed integer id (append-only); name: dotted name; level: die | ring | line | aggressor (width 1, N, L, A);
    unit: SI unit of the values; default: value when no component sets the element (provenance in ``default_note``);
    lo, hi: clip limits applied after summation; slow_eligible: may carry ``slow`` (time-compressible) components
    (07 P0-10: slow exogenous signals only); rate_to_k_per_s: factor that converts |d value/dt| into the K/s
    equivalent used by the compression check (None: no equivalence, the check only warns); foh_tol: largest
    deviation from linear allowed between two scen_next_break_b breaks for smooth kinds (signal units, design
    choice); path: which 07 P1-4 / P0-11 / P1-12 path it is; consumer: who reads it.
    """
    id: int
    name: str
    level: str
    unit: str
    default: float
    default_note: str = ""
    lo: float = -math.inf
    hi: float = math.inf
    slow_eligible: bool = False
    rate_to_k_per_s: float | None = None
    foh_tol: float = 1e-3
    path: str = ""
    consumer: str = ""
    doc: str = ""


SIG_T_BASE_K = 0
SIG_XPU_RISE_K = 1
SIG_AGGRESSOR_W = 2
SIG_LASER_DFREQ_HZ = 3
SIG_LASER_POWER_DELTA_DB = 4
SIG_LASER_LINE_ENABLE = 5
SIG_INPUT_LOSS_DELTA_DB = 6
SIG_ONES_DENSITY = 7
SIG_TRANSITION_DENSITY = 8
SIG_SWING_SCALE = 9
SIG_DRIVER_ENABLE = 10
SIG_BIAS_OFFSET_V = 11
SIG_BIAS_V = 12
SIG_FAULT_ADC_DROP = 13
SIG_FAULT_ADC_STUCK = 14
SIG_FAULT_ADC_SATURATE = 15
SIG_FAULT_DAC_GLITCH_CODES = 16
SIG_FAULT_CONTROLLER_RESET = 17

SIGNALS: tuple[SignalSpec, ...] = (
    SignalSpec(SIG_T_BASE_K, "thermal.t_base_k", "die", "K", 298.15,
               default_note="assumed 25 degC (lower end of the G26 aggressor and soak labels, LM26-P); presets set "
                            "scenario.ambient_c",
               lo=0.0, slow_eligible=True, rate_to_k_per_s=1.0, foh_tol=1e-3,
               path="07 P1-4 path 1: package / TEC base temperature (s to h)",
               consumer="plant thermal: T_base of the ROM step (absolute K)",
               doc="Absolute base temperature of the die below the ROM (TEC set point, soak profile, ambient)."),
    SignalSpec(SIG_XPU_RISE_K, "thermal.xpu_rise_k", "die", "K", 0.0, default_note="no XPU event",
               rate_to_k_per_s=1.0, foh_tol=1e-3,
               path="07 P1-4 path 3: XPU / ASIC power events (10-120 degC/s within ~30 ms at the PIC)",
               consumer="plant thermal: added to T_base as a die-wide common-mode rise",
               doc="Temperature rise at the PIC caused by a processor power-state change (not the TEC, not the "
                   "on-die aggressor)."),
    SignalSpec(SIG_AGGRESSOR_W, "thermal.aggressor_w", "aggressor", "W", 0.0, default_note="aggressor off",
               lo=0.0, foh_tol=1e-6,
               path="07 P1-4 path 2: on-die aggressor heater power into the ROM aggressor column (D6)",
               consumer="plant thermal: source power of each aggressor site (piecewise linear -> exact FOH)",
               doc="Power of each on-die aggressor heater, W, >= 0 (a heater can only heat: PAT-1 [0050])."),
    SignalSpec(SIG_LASER_DFREQ_HZ, "laser.dfreq_hz", "line", "Hz", 0.0, default_note="static line error only "
               "(laser.line_offset_ghz in the config)", slow_eligible=True, rate_to_k_per_s=1.0 / _LM_HZ_PER_K,
               foh_tol=1.0e6,
               path="07 P1-4 path 4: laser wavelength, optical only (sweep emulation, drift, failover)",
               consumer="plant optics: added to each line frequency (positive = higher frequency)",
               doc="Dynamic line frequency offset in Hz (positive = shorter wavelength). A heating-equivalent "
                   "sweep is positive: the line moves blue as the resonance would move red."),
    SignalSpec(SIG_LASER_POWER_DELTA_DB, "laser.power_delta_db", "line", "dB", 0.0, default_note="no change",
               slow_eligible=True, foh_tol=1e-4,
               path="07 P1-12 / textbook A17: laser power drift and failover power step",
               consumer="plant optics: line power x 10**(value/10)",
               doc="Change of each line's power at the laser output, dB."),
    SignalSpec(SIG_LASER_LINE_ENABLE, "laser.line_enable", "line", "frac", 1.0, default_note="all lines on",
               lo=0.0, hi=1.0, foh_tol=1e-4,
               path="07 P1-12: laser_line_enable (T, B, L)",
               consumer="plant optics: line power factor in [0, 1] (0 = off; a ramp is a turn-on)",
               doc="Line enable as a power factor in [0, 1]."),
    SignalSpec(SIG_INPUT_LOSS_DELTA_DB, "optical.input_loss_delta_db", "die", "dB", 0.0,
               default_note="nominal optical_path.chip_input_loss_db", slow_eligible=True, foh_tol=1e-4,
               path="07 P1-12 scenario A6: fibre-loss step (connector re-mate)",
               consumer="plant optics: every line power x 10**(-value/10)",
               doc="Change of the laser-to-chip loss, dB (positive = more loss)."),
    SignalSpec(SIG_ONES_DENSITY, "data.ones_density", "ring", "frac", 0.5,
               default_note="assumed balanced (scrambled) traffic", lo=0.0, hi=1.0, foh_tol=1e-4,
               path="07 P0-11: ones density of the transmitted data", consumer="plant datapath / optics (mark "
               "density of the two-level average)", doc="Fraction of ones in the transmitted bits."),
    SignalSpec(SIG_TRANSITION_DENSITY, "data.transition_density", "ring", "frac", 0.5,
               default_note="derived: random NRZ with ones density 0.5 has 2 p (1 - p) = 0.5 transitions per bit; "
                            "presets set datapath.transition_density", lo=0.0, hi=1.0, foh_tol=1e-4,
               path="07 P1-8: transition_density (T, B, N)", consumer="plant electrical: CV^2 junction heat and "
               "driver energy per transition", doc="Transitions per bit."),
    SignalSpec(SIG_SWING_SCALE, "data.swing_scale", "ring", "frac", 1.0, default_note="nominal swing",
               lo=0.0, foh_tol=1e-4,
               path="07 P1-8: swing_vpp (T, B, N) as a factor on driver.swing_vpp (squelch = 0)",
               consumer="plant electrical: junction swing = nominal swing x value",
               doc="Factor on the driver swing; 0 is squelch (modulation off, average kept)."),
    SignalSpec(SIG_DRIVER_ENABLE, "data.driver_enable", "ring", "frac", 1.0, default_note="driver on",
               lo=0.0, hi=1.0, foh_tol=1e-4, path="07 P1-8: driver_enable (T, B, N)",
               consumer="plant electrical: driver static power and data on/off (>= 0.5 = on)",
               doc="Driver enable."),
    SignalSpec(SIG_BIAS_OFFSET_V, "tx.bias_offset_v", "ring", "V", 0.0, default_note="no offset (config "
               "scenario.bias_offset_v sets a constant)", foh_tol=1e-4,
               path="07 P0-7: additive bias offset (ripple, droop), always allowed",
               consumer="plant electrical: BiasBank.step(v_offset=...)", doc="Additive reverse-bias offset, V."),
    SignalSpec(SIG_BIAS_V, "tx.bias_v", "ring", "V", math.nan, default_note="NaN = not driven by the scenario "
               "(config scenario.bias_v sets it when the scenario owns tx.bias)", foh_tol=1e-4,
               path="07 P0-7: absolute bias when tx.bias is bound to the scenario (scenario A7 relOMA plane)",
               consumer="plant electrical: absolute reverse bias, V (only when Scenario.drives has tx.bias)",
               doc="Absolute reverse bias set by the scenario."),
    SignalSpec(SIG_FAULT_ADC_DROP, "fault.adc_drop", "ring", "flag", 0.0, lo=0.0, hi=1.0,
               path="07 P1-10 fault mask adc_drop_mask", consumer="plant sensing (>= 0.5: sample dropped, "
               "monitor_valid false)"),
    SignalSpec(SIG_FAULT_ADC_STUCK, "fault.adc_stuck", "ring", "flag", 0.0, lo=0.0, hi=1.0,
               path="07 P1-10 fault mask adc_stuck", consumer="plant sensing (>= 0.5: code frozen)"),
    SignalSpec(SIG_FAULT_ADC_SATURATE, "fault.adc_saturate", "ring", "flag", 0.0, lo=0.0, hi=1.0,
               path="07 P1-10 fault mask adc_saturate", consumer="plant sensing (>= 0.5: code at full scale)"),
    SignalSpec(SIG_FAULT_DAC_GLITCH_CODES, "fault.dac_glitch_codes", "ring", "code", 0.0,
               path="07 P1-10 fault dac_glitch (proposal: additive heater code after the DAC FIFO, per tick)",
               consumer="plant electrical (proposal; docs/handoff/plant_electrical.md asked for the interface)"),
    SignalSpec(SIG_FAULT_CONTROLLER_RESET, "fault.controller_reset", "die", "flag", 0.0, lo=0.0, hi=1.0,
               path="07 P1-10 fault controller_reset_at", consumer="runner (>= 0.5: reset the controller state "
               "at the next edge)"),
)
SIGNAL_BY_NAME: dict[str, SignalSpec] = {s.name: s for s in SIGNALS}
assert all(s.id == i for i, s in enumerate(SIGNALS))


def signal_id(name: str) -> int:
    """Integer id of a signal name (ScenarioError with a did-you-mean for an unknown name)."""
    try:
        return SIGNAL_BY_NAME[name].id
    except KeyError:
        import difflib
        near = difflib.get_close_matches(name, list(SIGNAL_BY_NAME), n=3)
        raise ScenarioError(f"unknown scenario signal {name!r}" + (f"; did you mean {near}?" if near else
                            f"; known: {sorted(SIGNAL_BY_NAME)}")) from None


# =====================================================================================================================
# Array layout constants (true constants, read by the kernels)
# =====================================================================================================================

K_CONST = 0
K_PWL = 1
K_TRAPEZOID = 2
K_RECT_SINE = 3
K_SINE = 4
K_SMOOTH_STEP = 5
K_RANDOM_WALK = 6
K_RANDOM_LEVELS = 7
K_SAMPLED = 8
KINDS = ("const", "pwl", "trapezoid", "rect_sine", "sine", "smooth_step", "random_walk", "random_levels", "sampled")

MAP_IDENTITY = 0
MAP_BERNOULLI_TD = 1
MAPS = ("identity", "bernoulli_td")

DIST_UNIFORM = 0
DIST_NORMAL = 1
DIST_CHOICE = 2
DISTS = ("uniform", "normal", "choice")

# comp_i64 (C, NCI)
CI_KIND = 0
CI_SIGNAL = 1
CI_E_LO = 2          # first element of the signal covered (inclusive)
CI_E_HI = 3          # end element (exclusive)
CI_FLAGS = 4
CI_MAP = 5
CI_TAB_OFF = 6       # offset of the component's table in tab_f64 rows
CI_TAB_LEN = 7       # pwl: points; sampled: samples per element; random_levels choice: number of choices
CI_SLOT = 8          # counter-RNG slot
CI_CACHE_OFF = 9     # random_walk cache offset (3 floats per cached element)
CI_DIST = 10
NCI = 11

# comp_f64 (Bs, C, NCF)
CF_T0 = 0            # window start on the component's time axis, s
CF_T1 = 1            # window end (inf = never)
CF_AMP = 2
CF_OFFSET = 3
CF_PERIOD = 4        # s (0 = not periodic)
CF_PHASE = 5         # s, x = t_axis - t0 + phase
CF_P0 = 6            # kind-specific (see Component)
CF_P1 = 7
CF_P2 = 8
CF_P3 = 9
CF_P4 = 10
CF_BURST_ON = 11     # s (0 = no burst envelope)
CF_BURST_OFF = 12
CF_BURST_PHASE = 13
CF_HSMOOTH = 14      # longest step (axis s) over which a smooth kind stays within the signal's foh_tol of linear
NCF = 15

# sig_i64 (S, NSI), sig_f64 (S, NSF)
SI_OFF = 0
SI_WIDTH = 1
SI_C0 = 2
SI_NC = 3
SI_CLIP = 4          # 1 when the components may push the sum past a finite clip limit (next break per tick)
NSI = 5
SF_LO = 0
SF_HI = 1
NSF = 2

FL_SLOW = 1
FL_ABSOLUTE = 2
FL_COMMON = 4
FL_HOLD_BEFORE = 8
FL_HOLD_AFTER = 16
FL_PERIODIC = 32
FL_LINEAR = 64
FL_PER_ELEMENT = 128

NO_BREAK = 2**62     # scen_next_break_b: no break ahead (inside the int64 tick range of mrm.core.timing)


# =====================================================================================================================
# Kernels (time in TICKS: every time field was converted to plant ticks at compile, edges snapped to the tick grid)
# =====================================================================================================================

@njit(cache=True, inline="always")
def _pmod(x, p):
    """x mod p in [0, p) for p > 0 as x - p floor(x / p), corrected into the interval. Exact for whole-tick x and p
    below 2**53 (q = floor(x/p) is exact there, and so is p q); otherwise the error is about one ulp of x. It is
    about 15x faster than np.fmod inside numba."""
    r = x - p * math.floor(x / p)
    if r < 0.0:
        r += p
    if r >= p:
        r -= p
    return r


@njit(cache=True, inline="always")
def _pwl_seg(tab, toff, n, x):
    """Index i with t_i <= x < t_{i+1} in the sorted times tab[toff:toff+n]; -1 before t_0, n-1 at or after t_{n-1}."""
    if x < tab[toff]:
        return -1
    if x >= tab[toff + n - 1]:
        return n - 1
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if tab[toff + mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True, inline="always")
def _rw_value(x, knot, sd, mu, key, erng, slot, cache, coff):
    """Random walk W(x), linear between knots every ``knot`` ticks: W_0 = 0, W_{j+1} = W_j + mu + sd z_j with
    z_j = N(0, 1) at counter (tick = j, ring = erng, slot). cache[coff:coff+3] = (j, W_j, W_{j+1}) only saves work:
    the result is the same for any evaluation order."""
    if not (x > 0.0):
        return 0.0
    u = x / knot
    jf = math.floor(u)
    f = u - jf
    j = np.int64(jf)
    cj = cache[coff]
    if cj >= 0.0 and np.int64(cj) <= j:
        k0 = np.int64(cj)
        v0 = cache[coff + 1]
        v1 = cache[coff + 2]
    else:
        k0 = np.int64(0)
        v0 = 0.0
        v1 = mu + sd * _rng.normal(key, erng, np.int64(0), slot)
    while k0 < j:
        k0 += 1
        v0 = v1
        v1 = v0 + mu + sd * _rng.normal(key, erng, k0, slot)
    cache[coff] = float(k0)
    cache[coff + 1] = v0
    cache[coff + 2] = v1
    return v0 + f * (v1 - v0)


@njit(cache=True, inline="always")
def _rl_value(x, interval, dist, a, b, lo, hi, key, erng, slot, tab, toff, n, cache, coff):
    """Random level of interval j = floor(x / interval) (j = 0 for x < 0 or an infinite interval). cache[coff:coff+2]
    = (j, level) saves the Philox draw while the interval does not change (the level is a pure function of j)."""
    j = np.int64(0)
    if interval > 0.0 and interval < math.inf and x > 0.0:
        j = np.int64(math.floor(x / interval))
    if cache[coff] == float(j):
        return cache[coff + 1]
    if dist == DIST_NORMAL:
        v = a + b * _rng.normal(key, erng, j, slot)
    elif dist == DIST_CHOICE:
        u = _rng.uniform(key, erng, j, slot)
        i = np.int64(math.floor(u * n))
        if i > n - 1:
            i = n - 1
        v = tab[toff + i]
    else:
        u = _rng.uniform(key, erng, j, slot)
        v = a + (b - a) * u
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    cache[coff] = float(j)
    cache[coff + 1] = v
    return v


@njit(cache=True, inline="always")
def _raw(kind, x, period, p0, p1, p2, p3, p4, flags, dist, tab, toff, tlen, e_rel, key, erng, slot, cache, coff):
    """Raw generator value at local time x (ticks; see the kind table in the module docstring)."""
    if kind == K_CONST:
        return 1.0
    if kind == K_TRAPEZOID:
        xm = _pmod(x, period)
        if xm < p0:
            return xm / p0
        xm -= p0
        if xm < p1:
            return 1.0
        xm -= p1
        if xm < p2:
            return 1.0 - xm / p2
        return 0.0
    if kind == K_RECT_SINE:
        return abs(math.sin(math.pi * (_pmod(x, period) / period)))
    if kind == K_SINE:
        return math.sin(2.0 * math.pi * (_pmod(x, period) / period))
    if kind == K_SMOOTH_STEP:
        if x <= 0.0:
            return 0.0
        if x >= p0:
            return 1.0
        return 0.5 * (1.0 - math.cos(math.pi * x / p0))
    if kind == K_PWL:
        n = tlen
        vbase = toff + n + (e_rel * n if (flags & FL_PER_ELEMENT) != 0 else 0)
        xx = _pmod(x, period) if (flags & FL_PERIODIC) != 0 else x
        i = _pwl_seg(tab, toff, n, xx)
        if i < 0:
            return tab[vbase]
        if i >= n - 1:
            return tab[vbase + n - 1]
        ta = tab[toff + i]
        tb = tab[toff + i + 1]
        fr = (xx - ta) / (tb - ta)
        return tab[vbase + i] + fr * (tab[vbase + i + 1] - tab[vbase + i])
    if kind == K_SAMPLED:
        n = tlen
        base = toff + (e_rel * n if (flags & FL_PER_ELEMENT) != 0 else 0)
        u = x / p0
        if (flags & FL_PERIODIC) != 0:
            u = _pmod(u, float(n))
            i0 = np.int64(math.floor(u))
            if i0 > n - 1:
                i0 = n - 1
            fr = u - i0
            i1 = i0 + 1
            if i1 >= n:
                i1 = 0
        else:
            if u <= 0.0:
                return tab[base]
            if u >= n - 1:
                return tab[base + n - 1]
            i0 = np.int64(math.floor(u))
            fr = u - i0
            i1 = i0 + 1
        if (flags & FL_LINEAR) != 0:
            return tab[base + i0] + fr * (tab[base + i1] - tab[base + i0])
        return tab[base + i0]
    if kind == K_RANDOM_WALK:
        return _rw_value(x, p0, p1, p2, key, erng, slot, cache, coff)
    if kind == K_RANDOM_LEVELS:
        return _rl_value(x, p0, dist, p1, p2, p3, p4, key, erng, slot, tab, toff, tlen, cache, coff)
    return 0.0


@njit(cache=True, inline="always")
def _comp_value(c, t, compression, e, comp_i64, comp_f64_b, tab_b, keys_b, cache_b):
    """Contribution of component c to element e at tick t (0 outside its window unless held)."""
    fl = comp_i64[c, CI_FLAGS]
    ta = t * compression if (fl & FL_SLOW) != 0 else t
    t0 = comp_f64_b[c, CF_T0]
    t1 = comp_f64_b[c, CF_T1]
    if ta < t0:
        if (fl & FL_HOLD_BEFORE) == 0:
            return 0.0
        ta = t0
    elif ta >= t1:
        if (fl & FL_HOLD_AFTER) == 0:
            return 0.0
        ta = t1 - (1e-9 + 1e-15 * abs(t1))                         # left limit at the window end
    on = comp_f64_b[c, CF_BURST_ON]
    if on > 0.0:
        ym = _pmod(ta - t0 - comp_f64_b[c, CF_BURST_PHASE], on + comp_f64_b[c, CF_BURST_OFF])
        if ym >= on:
            return 0.0
    x = ta - t0 + comp_f64_b[c, CF_PHASE]
    kind = comp_i64[c, CI_KIND]
    e_lo = comp_i64[c, CI_E_LO]
    common = (fl & FL_COMMON) != 0
    erng = np.int64(0) if common else np.int64(e)
    coff = comp_i64[c, CI_CACHE_OFF]
    if kind == K_RANDOM_WALK and not common:
        coff += 3 * (e - e_lo)
    elif kind == K_RANDOM_LEVELS and not common:
        coff += 2 * (e - e_lo)
    r = _raw(kind, x, comp_f64_b[c, CF_PERIOD], comp_f64_b[c, CF_P0], comp_f64_b[c, CF_P1], comp_f64_b[c, CF_P2],
             comp_f64_b[c, CF_P3], comp_f64_b[c, CF_P4], fl, comp_i64[c, CI_DIST], tab_b, comp_i64[c, CI_TAB_OFF],
             comp_i64[c, CI_TAB_LEN], e - e_lo, keys_b[c], erng, comp_i64[c, CI_SLOT], cache_b, coff)
    if comp_i64[c, CI_MAP] == MAP_BERNOULLI_TD:
        r = 2.0 * r * (1.0 - r)
    return comp_f64_b[c, CF_OFFSET] + comp_f64_b[c, CF_AMP] * r


@njit(cache=True, inline="always")
def _per_element(kind, fl):
    if (fl & FL_PER_ELEMENT) != 0:
        return True
    return (kind == K_RANDOM_WALK or kind == K_RANDOM_LEVELS) and (fl & FL_COMMON) == 0


@njit(cache=True, inline="always")
def scen_eval_signal_b(sig, tick, compression, sig_i64, sig_f64, elem_default, comp_i64, comp_f64_b, tab_b, keys_b,
                       cache_b, out):
    """Evaluate one signal of one batch element at plant tick ``tick``; writes out[off:off+width] only.

    Arguments: sig (signal id), tick (int64 >= 0), compression (scenario.time_compression), then the ScenarioArrays
    fields for row b: sig_i64 (S, NSI), sig_f64 (S, NSF), elem_default (V,), comp_i64 (C, NCI), comp_f64_b =
    comp_f64[bb] (C, NCF), tab_b = tab_f64[bb] (NT,), keys_b = keys[b] (C,) uint64, cache_b = cache[b] (NK,) float64
    (mutated: random-walk cache), out (V,) float64; bb = b if comp_f64.shape[0] > 1 else 0. Every time field is in
    ticks of the dt the arrays were compiled for (ScenarioArrays.dt_s).
    """
    t = float(tick)
    off = sig_i64[sig, SI_OFF]
    w = sig_i64[sig, SI_WIDTH]
    for e in range(w):
        out[off + e] = elem_default[off + e]
    c0 = sig_i64[sig, SI_C0]
    for c in range(c0, c0 + sig_i64[sig, SI_NC]):
        e_lo = comp_i64[c, CI_E_LO]
        e_hi = comp_i64[c, CI_E_HI]
        if _per_element(comp_i64[c, CI_KIND], comp_i64[c, CI_FLAGS]):
            for e in range(e_lo, e_hi):
                out[off + e] += _comp_value(c, t, compression, e, comp_i64, comp_f64_b, tab_b, keys_b, cache_b)
        else:
            v = _comp_value(c, t, compression, e_lo, comp_i64, comp_f64_b, tab_b, keys_b, cache_b)
            for e in range(e_lo, e_hi):
                out[off + e] += v
    lo = sig_f64[sig, SF_LO]
    hi = sig_f64[sig, SF_HI]
    for e in range(w):
        v = out[off + e]
        if v < lo:
            out[off + e] = lo
        elif v > hi:
            out[off + e] = hi


@njit(cache=True, inline="always")
def scen_init_b(elem_default, out):
    """out[:] = elem_default: the value of every signal without components. Call once per batch row at reset."""
    for i in range(out.shape[0]):
        out[i] = elem_default[i]


@njit(cache=True)
def scen_eval_b(tick, compression, sig_i64, sig_f64, elem_default, comp_i64, comp_f64_b, tab_b, keys_b, cache_b,
                out):
    """Evaluate, at plant tick ``tick``, every signal of one batch element that HAS components; signals without
    components keep what :func:`scen_init_b` wrote (their defaults). Arguments as :func:`scen_eval_signal_b`
    without ``sig``. Call inside the engine's prange over b."""
    for s in range(sig_i64.shape[0]):
        if sig_i64[s, SI_NC] > 0:
            scen_eval_signal_b(s, tick, compression, sig_i64, sig_f64, elem_default, comp_i64, comp_f64_b, tab_b,
                               keys_b, cache_b, out)


@njit(cache=True)
def scen_eval(tick, compression, sig_i64, sig_f64, elem_default, comp_i64, comp_f64, tab_f64, keys, cache, out):
    """Batched evaluation (serial loop over b): out (B, V). comp_f64 (Bs, C, NCF), tab_f64 (Bs, NT), Bs in {1, B};
    keys (B, C) uint64; cache (B, NK)."""
    bs = comp_f64.shape[0]
    for b in range(out.shape[0]):
        bb = b if bs > 1 else 0
        scen_init_b(elem_default, out[b])
        scen_eval_b(tick, compression, sig_i64, sig_f64, elem_default, comp_i64, comp_f64[bb], tab_f64[bb], keys[b],
                    cache[b], out[b])


@njit(cache=True)
def scen_render(ticks, compression, sig_i64, sig_f64, elem_default, comp_i64, comp_f64, tab_f64, keys, cache, out):
    """Evaluate every signal at each tick of ``ticks`` (T,): out (T, B, V)."""
    for i in range(ticks.shape[0]):
        scen_eval(ticks[i], compression, sig_i64, sig_f64, elem_default, comp_i64, comp_f64, tab_f64, keys, cache,
                  out[i])


@njit(cache=True, inline="always")
def _next_edge_x(kind, x, period, p0, p1, p2, flags, tab, toff, tlen, hsmooth):
    """Distance in local time x (ticks) to the next point where the raw generator stops being linear (inf: never)."""
    inf = math.inf
    if kind == K_CONST:
        return inf
    if kind == K_TRAPEZOID:
        xm = _pmod(x, period)
        best = period - xm
        for pos in (p0, p0 + p1, p0 + p1 + p2):
            if pos > xm and pos - xm < best:
                best = pos - xm
        return best
    if kind == K_RECT_SINE:
        return min(hsmooth, period - _pmod(x, period))
    if kind == K_SINE:
        return hsmooth
    if kind == K_SMOOTH_STEP:
        if x < 0.0:
            return -x
        if x < p0:
            return min(hsmooth, p0 - x)
        return inf
    if kind == K_PWL:
        n = tlen
        if (flags & FL_PERIODIC) != 0:
            xm = _pmod(x, period)
            i = _pwl_seg(tab, toff, n, xm)
            if i + 1 <= n - 1 and tab[toff + i + 1] > xm:
                return min(tab[toff + i + 1] - xm, period - xm)
            return period - xm
        i = _pwl_seg(tab, toff, n, x)
        if i >= n - 1:
            return inf
        return tab[toff + i + 1] - x
    if kind == K_RANDOM_WALK or kind == K_RANDOM_LEVELS or kind == K_SAMPLED:
        d = p0
        if not (d > 0.0 and d < inf):
            return inf
        periodic_sampled = kind == K_SAMPLED and (flags & FL_PERIODIC) != 0
        if kind == K_SAMPLED and not periodic_sampled and x >= (tlen - 1) * d:
            return inf
        if x < 0.0 and not periodic_sampled:
            # constant before x = 0 (walk at 0, first sample held) or first level up to x = d
            return d - x if kind == K_RANDOM_LEVELS else -x
        # knots at every multiple of d (a periodic table also has them at negative x, e.g. a negative phase)
        return (math.floor(x / d) + 1.0) * d - x
    return inf


@njit(cache=True, inline="always")
def _comp_next_edge(c, t, compression, comp_i64, comp_f64_b, tab_b):
    """Plant tick (float) of component c's next linearity break after tick t (inf: none)."""
    fl = comp_i64[c, CI_FLAGS]
    cf = compression if (fl & FL_SLOW) != 0 else 1.0
    ta = t * cf
    t0 = comp_f64_b[c, CF_T0]
    t1 = comp_f64_b[c, CF_T1]
    if ta < t0:
        return t0 / cf
    if ta >= t1:
        return math.inf
    best = t1
    on = comp_f64_b[c, CF_BURST_ON]
    if on > 0.0:
        per = on + comp_f64_b[c, CF_BURST_OFF]
        ym = _pmod(ta - t0 - comp_f64_b[c, CF_BURST_PHASE], per)
        nxt = on - ym if ym < on else per - ym
        best = min(best, ta + nxt)
    x = ta - t0 + comp_f64_b[c, CF_PHASE]
    dx = _next_edge_x(comp_i64[c, CI_KIND], x, comp_f64_b[c, CF_PERIOD], comp_f64_b[c, CF_P0], comp_f64_b[c, CF_P1],
                      comp_f64_b[c, CF_P2], fl, tab_b, comp_i64[c, CI_TAB_OFF], comp_i64[c, CI_TAB_LEN],
                      comp_f64_b[c, CF_HSMOOTH])
    hs = comp_f64_b[c, CF_HSMOOTH]          # curved components (smooth kinds, mapped ramps) break every hs ticks
    if hs < dx:
        dx = hs
    best = min(best, ta + dx)
    return best / cf


@njit(cache=True)
def scen_next_break_b(sig, tick, compression, sig_i64, comp_i64, comp_f64_b, tab_b):
    """Largest tick kb > tick such that signal ``sig`` sampled at ticks tick..kb lies on one straight line (exact
    FOH over [tick, kb]; 07 P0-10 ``substep_policy: auto``). An edge at tick m (or between m-1 and m) gives
    kb = m - 1; an edge within 1e-6 tick (plus 1e-14 relative) above a tick counts as on it, against float rounding.
    Returns tick + 1 when a break is that close and NO_BREAK (2**62) when the signal stays linear for ever. Curved
    components (rect_sine, sine, smooth_step, and ramps under the bernoulli_td map) break every CF_HSMOOTH ticks
    (summed deviation <= the signal's foh_tol). A signal
    whose components may push the sum past a clip limit (sig_i64[sig, SI_CLIP] = 1, decided at compile) can have a
    kink where it meets the limit, which this function cannot locate without the values: it returns tick + 1 (exact,
    per-tick) for such a signal.
    """
    if sig_i64[sig, SI_CLIP] != 0:
        return tick + 1
    t = float(tick)
    te = math.inf
    c0 = sig_i64[sig, SI_C0]
    for c in range(c0, c0 + sig_i64[sig, SI_NC]):
        e = _comp_next_edge(c, t, compression, comp_i64, comp_f64_b, tab_b)
        if e < te:
            te = e
    if not (te < 4.0e18):
        return np.int64(NO_BREAK)
    kb = np.int64(math.ceil(te - (1e-6 + 1e-14 * te))) - 1
    if kb < tick + 1:
        kb = tick + 1
    return kb


# =====================================================================================================================
# NumPy shapes (reference forms used by calibrate.py, the library and the tests)
# =====================================================================================================================

def np_trapezoid(x, period, rise, high, fall):
    """NumPy version of the trapezoid raw value (0 -> 1 -> 0) at local times x (array)."""
    xm = np.mod(np.asarray(x, dtype=np.float64), period)
    out = np.zeros_like(xm)
    if rise > 0:
        m = xm < rise
        out[m] = xm[m] / rise
    m = (xm >= rise) & (xm < rise + high)
    out[m] = 1.0
    if fall > 0:
        m = (xm >= rise + high) & (xm < rise + high + fall)
        out[m] = 1.0 - (xm[m] - rise - high) / fall
    return out


def np_rect_sine(x, period):
    """NumPy |sin(pi x / P)|."""
    return np.abs(np.sin(np.pi * np.mod(np.asarray(x, dtype=np.float64), period) / period))


def np_smooth_step(x, duration):
    """NumPy raised-cosine step (0 before, 1 after ``duration``)."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0, 0.0, np.where(x >= duration, 1.0, 0.5 * (1.0 - np.cos(np.pi * np.clip(x, 0, duration)
                                                                                    / duration))))


# =====================================================================================================================
# Data model
# =====================================================================================================================

def _crc20(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFF        # slot < 2**20 (mrm.core.rng.SLOT_MAX)


def _as_world(v, name: str) -> np.ndarray:
    a = np.atleast_1d(np.asarray(v, dtype=np.float64))
    if a.ndim != 1:
        raise ScenarioError(f"{name} must be a scalar or a (B,) array of per-world values, got shape {a.shape}")
    return a


@dataclass(frozen=True, eq=False)
class Component:
    """One generator instance on one signal. Use the builder functions (:func:`const`, :func:`pwl`, ...).

    signal: signal name; kind: one of KINDS; elements: element indices of the signal (None = all); site: aggressor
    site name (for ``thermal.aggressor_w`` only, instead of elements); amp, offset: scalar or (B,) per world;
    t0_s, t1_s: active window on the component's time axis; period_s, phase_s: see the module docstring;
    params: (p0, ..., p4) kind-specific -- trapezoid (rise, high, fall), smooth_step (duration), random_walk
    (knot_s, sigma per sqrt(s), drift per s), random_levels (interval_s, a, b, lo, hi) with dist uniform [a, b],
    normal (mean a, sd b) or choice (table), sampled (interval_s); table_t: pwl times (n,); table_v: pwl/sampled
    values (n,) | (Bs, n) | (Bs, E, n) per element, or the choices of random_levels; absolute: defines the level
    (the signal default is dropped on the elements it covers); slow: on the time-compressible axis (07 P0-10);
    common: random kinds draw once for all elements; map: identity | bernoulli_td; stream: counter-RNG stream name
    (``scenario_<name>``); link: components with the same link (and stream) share draws; linear (sampled), periodic
    (pwl, sampled); provenance, source, note: where its numbers come from; label: unique name inside the scenario.
    """
    signal: str
    kind: str
    elements: tuple[int, ...] | None = None
    site: str | None = None
    amp: Any = 1.0
    offset: Any = 0.0
    t0_s: float = 0.0
    t1_s: float = math.inf
    period_s: float = 0.0
    phase_s: float = 0.0
    params: tuple[float, ...] = ()
    table_t: Any = None
    table_v: Any = None
    burst_on_s: float = 0.0
    burst_off_s: float = 0.0
    burst_phase_s: float = 0.0
    hold_before: bool = False
    hold_after: bool = False
    absolute: bool = False
    slow: bool = False
    common: bool = True
    map: str = "identity"
    dist: str = "uniform"
    stream: str = "scenario_default"
    link: str = ""
    linear: bool = True
    periodic: bool = False
    provenance: str = "assumed"
    source: str = ""
    note: str = ""
    label: str = ""

    def __post_init__(self):
        spec = SIGNAL_BY_NAME.get(self.signal)
        if spec is None:
            signal_id(self.signal)                    # raises with a suggestion
        if self.kind not in KINDS:
            raise ScenarioError(f"unknown generator kind {self.kind!r}; known: {KINDS}")
        if self.map not in MAPS:
            raise ScenarioError(f"unknown map {self.map!r}; known: {MAPS}")
        if self.dist not in DISTS:
            raise ScenarioError(f"unknown distribution {self.dist!r}; known: {DISTS}")
        if spec.level == "aggressor":
            if self.site is None:
                raise ScenarioError(f"{self.signal} components need site=<aggressor site name>")
        elif self.site is not None:
            raise ScenarioError(f"site= is only for thermal.aggressor_w, not {self.signal}")
        if self.slow and not spec.slow_eligible:
            raise CompressionError(
                f"{self.label or self.kind} on {self.signal}: only slow exogenous signals may be time-compressed "
                f"(07 P0-10); eligible: {[s.name for s in SIGNALS if s.slow_eligible]}")
        if not self.stream.startswith("scenario_"):
            raise ScenarioError(f"stream {self.stream!r} must be named scenario_<name> (mrm.core.rng)")
        if not (self.t1_s > self.t0_s):
            raise ScenarioError(f"{self.label or self.kind}: window [t0, t1) = [{self.t0_s}, {self.t1_s}) is empty")
        p = tuple(float(v) for v in self.params) + (0.0,) * (5 - len(self.params))
        object.__setattr__(self, "params", p[:5])
        periodic_kinds = ("trapezoid", "rect_sine", "sine")
        if (self.kind in periodic_kinds or self.periodic) and not (self.period_s > 0 or self.kind == "sampled"):
            raise ScenarioError(f"{self.label or self.kind}: period_s must be > 0")
        if self.kind == "trapezoid":
            r, h, f = p[0], p[1], p[2]
            if min(r, h, f) < 0 or r + h + f > self.period_s * (1 + 1e-12):
                raise ScenarioError(f"trapezoid rise/high/fall {r, h, f} must be >= 0 and fit in period "
                                    f"{self.period_s}")
        if self.kind == "smooth_step" and not p[0] > 0:
            raise ScenarioError("smooth_step needs a duration > 0")
        if self.kind in ("random_walk", "sampled") and not p[0] > 0:
            raise ScenarioError(f"{self.kind} needs an interval > 0")
        if self.kind == "random_levels" and not p[0] > 0:
            raise ScenarioError("random_levels needs an interval > 0 (math.inf for a single level)")
        if self.kind in ("pwl", "sampled") or (self.kind == "random_levels" and self.dist == "choice"):
            if self.table_v is None:
                raise ScenarioError(f"{self.label or self.kind} needs table_v")
        if self.kind == "pwl":
            tt = np.asarray(self.table_t, dtype=np.float64)
            if tt.ndim != 1 or tt.size < 1 or np.any(np.diff(tt) < 0) or not np.all(np.isfinite(tt)):
                raise ScenarioError("pwl table_t must be a non-decreasing finite 1-D array")
        if self.burst_on_s < 0 or self.burst_off_s < 0:
            raise ScenarioError("burst_on_s and burst_off_s must be >= 0")
        if self.map == "bernoulli_td" and self.kind == "random_walk":
            raise ScenarioError("map bernoulli_td needs a bounded raw value in [0, 1]; a random walk is unbounded")
        _as_world(self.amp, "amp")
        _as_world(self.offset, "offset")

    # ---- helpers used by compile / checks
    @property
    def spec(self) -> SignalSpec:
        return SIGNAL_BY_NAME[self.signal]

    @property
    def is_random(self) -> bool:
        return self.kind in ("random_walk", "random_levels")

    @property
    def slot(self) -> int:
        return _crc20(self.link or f"{self.signal}|{self.label}|{self.kind}")

    def world_count(self) -> int:
        """Number of per-world rows (1 if every field is shared by all worlds)."""
        n = {len(_as_world(self.amp, "amp")), len(_as_world(self.offset, "offset"))}
        if self.table_v is not None and self.kind in ("pwl", "sampled"):
            tv = np.asarray(self.table_v, dtype=np.float64)
            if tv.ndim >= 2:
                n.add(tv.shape[0])
        n.discard(1)
        if len(n) > 1:
            raise ScenarioError(f"{self.label or self.kind}: per-world arrays disagree in length {sorted(n)}")
        return n.pop() if n else 1

    def describe(self) -> dict:
        """JSON-friendly description (arrays as sha256 digests): used by fingerprints and manifests."""
        def enc(v):
            if v is None:
                return None
            a = np.asarray(v, dtype=np.float64)
            if a.ndim == 0:
                return float(a)
            return {"shape": list(a.shape), "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        for k in ("amp", "offset", "table_t", "table_v"):
            d[k] = enc(d[k])
        d["params"] = list(self.params)
        d["elements"] = None if self.elements is None else list(self.elements)
        return d


@dataclass(frozen=True, eq=False)
class AggressorSpec:
    """An on-die aggressor (07 P1-4, D6): a site in ``layout.sites``, the victim ring at which the target is stated,
    the target (a calibrate.AggressorTarget, or None for a direct power waveform) and its power components.

    ``power`` holds the Components on ``thermal.aggressor_w`` (site set). It is empty while a target awaits
    calibration (Scenario.compile then raises). ``record`` holds the calibration record (residuals, assumed thermal
    response, provenance ``calibrated``) or the direct-power provenance.
    """
    site: str
    victim_ring: int = 0
    target: Any = None
    power: tuple[Component, ...] = ()
    record: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def pending(self) -> bool:
        return self.target is not None and not self.power


@dataclass(frozen=True)
class ResetSpec:
    """How the run starts (textbook scenario B5 and 07 P1-1): ``cold`` (heaters at zero, no stored calibration:
    factory_cal withheld) or ``warm`` (factory calibration table passed at reset). The engine / runner reads it."""
    start: str = "warm"
    heater_init: str = "factory"        # factory | zero
    factory_cal: bool = True
    note: str = ""

    def __post_init__(self):
        if self.start not in ("cold", "warm"):
            raise ScenarioError(f"reset start must be cold or warm, got {self.start!r}")
        if self.heater_init not in ("factory", "zero"):
            raise ScenarioError(f"heater_init must be factory or zero, got {self.heater_init!r}")


#: 07 P0-10 compression rule: compressed slow ramps <= this fraction of the fast-transient rate.
COMPRESSION_RATE_FRACTION = 0.01
#: Fast-transient reference rate for that rule, K/s: peak |dT/dt| of the slide-10 aggressor, pi*88.5 K/0.1307 s =
#: 2127 K/s (derived from the read-off fit, 01 verification log item 5; CAL-13a).
FAST_RATE_REF_K_PER_S = math.pi * 88.5 / 0.1307


@dataclass(frozen=True, eq=False)
class Scenario:
    """A named disturbance scenario: components on signals, aggressors, defaults, time compression, reset.

    Build with the library (mrm.scenarios.library) or the registry (mrm.scenarios.registry.create/from_config);
    combine with ``a + b`` or :func:`stack`; compile with :meth:`compile` for the kernel, render with
    :meth:`render` for a DisturbanceFrame. Times are on the real (scenario) time axis; components flagged ``slow``
    run ``time_compression`` times faster in simulated time (07 P0-10).
    """
    name: str
    components: tuple[Component, ...] = ()
    aggressors: tuple[AggressorSpec, ...] = ()
    time_compression: float = 1.0
    defaults: Mapping[str, Any] = field(default_factory=dict)
    reset: ResetSpec | None = None
    duration_s: float | None = None
    slow_cycle_s: float | None = None
    fast_rate_ref_k_per_s: float = FAST_RATE_REF_K_PER_S
    description: str = ""
    provenance: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self):
        c = float(self.time_compression)
        if not (math.isfinite(c) and c >= 1.0):
            raise CompressionError(f"time_compression must be a finite number >= 1, got {self.time_compression!r}")
        for k in self.defaults:
            signal_id(k)
        labels = [x.label for x in self.components if x.label]
        dup = {l for l in labels if labels.count(l) > 1}
        if dup:
            raise ScenarioError(f"scenario {self.name!r}: duplicate component labels {sorted(dup)}")
        sites = [a.site for a in self.aggressors]
        if len(set(sites)) != len(sites):
            raise ScenarioError(f"scenario {self.name!r}: aggressor site listed twice: {sites}")

    # ---- composition
    def __add__(self, other: "Scenario") -> "Scenario":
        return stack([self, other])

    def replace(self, **kw) -> "Scenario":
        return dataclasses.replace(self, **kw)

    def with_compression(self, factor: float) -> "Scenario":
        """The same scenario with ``time_compression = factor`` (checked at compile)."""
        return dataclasses.replace(self, time_compression=float(factor))

    def with_defaults(self, **defaults) -> "Scenario":
        """Override signal defaults: keys are signal names with '.' replaced by '__' (or pass a dict to replace)."""
        d = dict(self.defaults)
        d.update({k.replace("__", "."): v for k, v in defaults.items()})
        return dataclasses.replace(self, defaults=d)

    def with_aggressor_power(self, site: str, t_knots_s, power_w, *, period_s: float | None,
                             record: Mapping[str, Any] | None = None, linear: bool = True) -> "Scenario":
        """Attach a power waveform to aggressor ``site``: uniform knots t_knots_s (n,) starting at 0, power_w (n,)
        or (Bs, n) per world (W, >= 0). Periodic when period_s is given (knot spacing = period/n). Used by
        calibrate.calibrate_scenario; ``record`` (residuals, assumed response) is stored on the AggressorSpec."""
        tk = np.asarray(t_knots_s, dtype=np.float64)
        pw = np.asarray(power_w, dtype=np.float64)
        if pw.ndim == 1:
            pw = pw[None, :]
        if tk.ndim != 1 or pw.shape[-1] != tk.size or tk.size < 2:
            raise ScenarioError("t_knots_s must be (n,) and power_w (n,) or (Bs, n), n >= 2")
        if np.any(pw < -1e-15):
            raise ScenarioError("aggressor power must be >= 0 (a heater can only heat, PAT-1 [0050])")
        d = np.diff(tk)
        if abs(tk[0]) > 1e-15 or np.max(np.abs(d - d[0])) > 1e-9 * max(d[0], 1e-300):
            raise ScenarioError("t_knots_s must be uniform and start at 0")
        if period_s is not None and abs(tk.size * d[0] - period_s) > 1e-9 * period_s:
            raise ScenarioError(f"periodic knots: n * spacing = {tk.size * d[0]} != period {period_s}")
        rec = dict(record or {})
        comp = Component("thermal.aggressor_w", "sampled", site=site, params=(float(d[0]),),
                         table_v=np.clip(pw, 0.0, None), periodic=period_s is not None, linear=linear,
                         period_s=float(period_s or 0.0), hold_after=period_s is None,
                         provenance=rec.get("provenance", "calibrated"), source=rec.get("source", ""),
                         note=rec.get("note", "aggressor power waveform"), label=f"aggressor_power:{site}",
                         stream="scenario_aggressor")
        aggs = []
        found = False
        for a in self.aggressors:
            if a.site == site:
                aggs.append(dataclasses.replace(a, power=(comp,), record=rec))
                found = True
            else:
                aggs.append(a)
        if not found:
            aggs.append(AggressorSpec(site=site, power=(comp,), record=rec))
        return dataclasses.replace(self, aggressors=tuple(aggs))

    # ---- queries
    @property
    def all_components(self) -> tuple[Component, ...]:
        out = list(self.components)
        for a in self.aggressors:
            out.extend(a.power)
        return tuple(out)

    @property
    def aggressor_sites(self) -> tuple[str, ...]:
        """Aggressor site names in order of first use."""
        seen: list[str] = []
        for a in self.aggressors:
            if a.site not in seen:
                seen.append(a.site)
        for c in self.components:
            if c.site is not None and c.site not in seen:
                seen.append(c.site)
        return tuple(seen)

    @property
    def pending_aggressors(self) -> tuple[str, ...]:
        return tuple(a.site for a in self.aggressors if a.pending)

    def signals_used(self) -> set[str]:
        return {c.signal for c in self.all_components} | set(self.defaults)

    @property
    def drives(self) -> tuple[str, ...]:
        """Actuator channels the scenario sets ABSOLUTELY (for mrm.core.channels.resolve_bindings
        ``scenario_drives``): ``tx.bias`` when tx.bias_v is set."""
        used = self.signals_used()
        if "tx.bias_v" in used:
            d = self.defaults.get("tx.bias_v")
            if any(c.signal == "tx.bias_v" for c in self.all_components) or (
                    d is not None and np.all(np.isfinite(np.asarray(d, dtype=np.float64)))):
                return ("tx.bias",)
        return ()

    @property
    def offsets(self) -> tuple[str, ...]:
        """Actuator channels the scenario adds an offset to (``scenario_offsets``): ``tx.bias`` for
        tx.bias_offset_v."""
        return ("tx.bias",) if "tx.bias_offset_v" in self.signals_used() else ()

    def fingerprint(self, *, include_compression: bool = True) -> str:
        """sha256 (hex, 16 chars) of everything that defines the scenario's signals."""
        d = {
            "name": self.name,
            "components": [c.describe() for c in self.components],
            "aggressors": [{"site": a.site, "victim_ring": a.victim_ring,
                            "target": None if a.target is None else a.target.describe(),
                            "power": [c.describe() for c in a.power]} for a in self.aggressors],
            "defaults": {k: np.asarray(v, dtype=np.float64).tolist() for k, v in sorted(self.defaults.items())},
            "reset": None if self.reset is None else dataclasses.asdict(self.reset),
        }
        if include_compression:
            d["time_compression"] = float(self.time_compression)
        text = json.dumps(d, sort_keys=True, default=str, allow_nan=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def summary(self) -> str:
        """Human-readable listing of the components and their provenance."""
        lines = [f"scenario {self.name} (time_compression {self.time_compression:g})"]
        if self.description:
            lines.append(f"  {self.description}")
        for c in self.all_components:
            where = f" site={c.site}" if c.site else (f" elements={list(c.elements)}" if c.elements else "")
            lines.append(f"  - {c.label or c.kind}: {c.kind} on {c.signal}{where}"
                         f"{' [slow]' if c.slow else ''}{' [absolute]' if c.absolute else ''}; "
                         f"{c.provenance} {c.source}; {c.note}")
        for a in self.aggressors:
            state = "PENDING CALIBRATION" if a.pending else a.record.get("provenance", "direct")
            lines.append(f"  aggressor {a.site} -> victim ring {a.victim_ring}: {state}")
        if self.reset is not None:
            lines.append(f"  reset: {self.reset.start} (heater_init {self.reset.heater_init}, factory_cal "
                         f"{self.reset.factory_cal})")
        return "\n".join(lines)

    # ---- compile
    def compile(self, *, dt_s: float, n_rings: int, n_lines: int, batch: int = 1, seed: int = 0,
                aggressor_sites: Sequence[str] | None = None, check_compression: bool = True) -> "ScenarioArrays":
        """Build the kernel arrays (see ScenarioArrays and docs/handoff/scenarios.md).

        dt_s: plant tick (s); n_rings N, n_lines L; batch B; seed: scenario seed base (experiment.seeds.scenario),
        element b uses seed + b; aggressor_sites: the site order of the engine's ROM aggressor columns (default:
        this scenario's sites). Raises ScenarioError for a pending aggressor calibration, an unknown site, an element
        out of range or conflicting absolute components, and CompressionError for a time_compression that breaks the
        07 P0-10 rule (``check_compression``).
        """
        return _compile(self, dt_s=float(dt_s), n_rings=int(n_rings), n_lines=int(n_lines), batch=int(batch),
                        seed=int(seed), aggressor_sites=aggressor_sites, check_compression=check_compression)

    def render(self, *, dt_s: float, n_ticks: int | None = None, ticks=None, stride_ticks: int = 1, tick0: int = 0,
               n_rings: int = 1, n_lines: int = 1, batch: int = 1, seed: int = 0,
               aggressor_sites: Sequence[str] | None = None) -> "DisturbanceFrame":
        """Evaluate every signal on a tick grid (for plots, the NumPy reference kernel and tests)."""
        arr = self.compile(dt_s=dt_s, n_rings=n_rings, n_lines=n_lines, batch=batch, seed=seed,
                           aggressor_sites=aggressor_sites)
        return arr.render(n_ticks=n_ticks, ticks=ticks, stride_ticks=stride_ticks, tick0=tick0)

    @classmethod
    def from_frame(cls, frame: "DisturbanceFrame", name: str = "recorded", **kw) -> "Scenario":
        """A scenario that replays a DisturbanceFrame (07 P0-9: recorded profiles)."""
        return cls(name=name, components=frame.to_components(), **kw)


def stack(scenarios: Iterable[Scenario], name: str | None = None) -> Scenario:
    """Combine scenarios: components add per signal, aggressors are merged by site (a site may appear once),
    defaults must agree, the time compression must agree (or be 1 in all but one), and at most one reset spec."""
    scs = [s for s in scenarios]
    if not scs:
        return Scenario(name=name or "static")
    comps: list[Component] = []
    aggs: list[AggressorSpec] = []
    defaults: dict[str, Any] = {}
    reset = None
    comp_c = 1.0
    prov: list = []
    dur = None
    cyc = None
    fast = math.inf
    for s in scs:
        for c in s.components:
            if c.label and any(c.label == o.label for o in comps):
                c = dataclasses.replace(c, label=f"{s.name}.{c.label}")
            comps.append(c)
        for a in s.aggressors:
            if any(a.site == o.site for o in aggs):
                raise ScenarioError(f"aggressor site {a.site!r} is defined by two stacked scenarios")
            aggs.append(a)
        for k, v in s.defaults.items():
            if k in defaults and not np.array_equal(np.asarray(defaults[k]), np.asarray(v), equal_nan=True):
                raise ScenarioError(f"stacked scenarios disagree on the default of {k}: {defaults[k]} vs {v}")
            defaults[k] = v
        if s.reset is not None:
            if reset is not None and reset != s.reset:
                raise ScenarioError("two stacked scenarios define different reset specs")
            reset = s.reset
        if s.time_compression != 1.0:
            if comp_c != 1.0 and comp_c != s.time_compression:
                raise CompressionError("stacked scenarios disagree on time_compression")
            comp_c = s.time_compression
        prov.extend(s.provenance)
        if s.duration_s is not None:
            dur = s.duration_s if dur is None else max(dur, s.duration_s)
        if s.slow_cycle_s is not None:
            cyc = s.slow_cycle_s if cyc is None else max(cyc, s.slow_cycle_s)
        fast = min(fast, s.fast_rate_ref_k_per_s)        # the strictest reference wins
    return Scenario(name=name or "+".join(s.name for s in scs), components=tuple(comps), aggressors=tuple(aggs),
                    time_compression=comp_c, defaults=defaults, reset=reset, duration_s=dur, slow_cycle_s=cyc,
                    fast_rate_ref_k_per_s=fast, description="; ".join(s.description for s in scs if s.description),
                    provenance=tuple(prov))


# =====================================================================================================================
# Builders
# =====================================================================================================================

def _common_kw(kw: dict) -> dict:
    allowed = {f.name for f in dataclasses.fields(Component)} - {"signal", "kind", "params", "table_t", "table_v",
                                                                 "amp", "offset", "period_s"}
    bad = set(kw) - allowed
    if bad:
        raise ScenarioError(f"unknown component options {sorted(bad)}; allowed: {sorted(allowed)}")
    return kw


def const(signal: str, value, **kw) -> Component:
    """Constant ``value`` (scalar or (B,) per world) while active."""
    return Component(signal, "const", amp=value, **_common_kw(kw))


def step(signal: str, t_s: float, delta, **kw) -> Component:
    """A step of ``delta`` at time t_s, held for ever."""
    return Component(signal, "const", amp=delta, t0_s=float(t_s), **_common_kw(kw))


def pulse(signal: str, t0_s: float, t1_s: float, value, **kw) -> Component:
    """``value`` on [t0_s, t1_s), 0 elsewhere."""
    return Component(signal, "const", amp=value, t0_s=float(t0_s), t1_s=float(t1_s), **_common_kw(kw))


def pwl(signal: str, t_s, values, *, amp=1.0, offset=0.0, periodic: bool = False, period_s: float | None = None,
        **kw) -> Component:
    """Piecewise-linear table: times t_s (n,) relative to t0_s, values (n,) | (Bs, n) | (Bs, E, n). Ends are held;
    with ``periodic`` the table repeats with period_s (default t_s[-1] - t_s[0]; t_s must start at 0)."""
    t = np.asarray(t_s, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if periodic:
        period_s = float(period_s if period_s is not None else t[-1] - t[0])
        if abs(t[0]) > 0:
            raise ScenarioError("a periodic pwl table must start at t = 0")
        if not period_s > 0 or t[-1] > period_s * (1 + 1e-12):
            raise ScenarioError(f"a periodic pwl table must end within its period ({t[-1]} > {period_s})")
    per_el = v.ndim == 3
    return Component(signal, "pwl", table_t=t, table_v=v, amp=amp, offset=offset, periodic=periodic,
                     period_s=float(period_s or 0.0), **_common_kw(dict(kw, **({"common": False} if per_el else {}))))


def trapezoid(signal: str, period_s: float, rise_s: float, high_s: float, fall_s: float, *, amp=1.0, offset=0.0,
              **kw) -> Component:
    """Periodic trapezoid between ``offset`` and ``offset + amp``: rise, high, fall, then low for the rest."""
    return Component(signal, "trapezoid", period_s=float(period_s), params=(rise_s, high_s, fall_s), amp=amp,
                     offset=offset, **_common_kw(kw))


def square(signal: str, period_s: float, duty: float, *, amp=1.0, offset=0.0, **kw) -> Component:
    """Square wave: ``offset + amp`` for duty*period, then ``offset``."""
    if not 0.0 <= duty <= 1.0:
        raise ScenarioError("duty must be in [0, 1]")
    return trapezoid(signal, period_s, 0.0, duty * period_s, 0.0, amp=amp, offset=offset, **kw)


def triangle(signal: str, period_s: float, *, amp=1.0, offset=0.0, rise_frac: float = 0.5, **kw) -> Component:
    """Triangle: up over rise_frac*period, down over the rest (0.5 = symmetric)."""
    if not 0.0 < rise_frac < 1.0:
        raise ScenarioError("rise_frac must be in (0, 1)")
    return trapezoid(signal, period_s, rise_frac * period_s, 0.0, (1.0 - rise_frac) * period_s, amp=amp,
                     offset=offset, **kw)


def sawtooth(signal: str, period_s: float, *, fall_s: float = 0.0, amp=1.0, offset=0.0, **kw) -> Component:
    """One-sided sawtooth: up over period - fall_s, down over fall_s (0 = an instantaneous return)."""
    return trapezoid(signal, period_s, period_s - fall_s, 0.0, fall_s, amp=amp, offset=offset, **kw)


def rect_sine(signal: str, period_s: float, *, amp=1.0, offset=0.0, **kw) -> Component:
    """``offset + amp * |sin(pi x / period)|`` (the slide-10 aggressor shape, 01 verification log item 5)."""
    return Component(signal, "rect_sine", period_s=float(period_s), amp=amp, offset=offset, **_common_kw(kw))


def sine(signal: str, period_s: float, *, amp=1.0, offset=0.0, **kw) -> Component:
    """``offset + amp * sin(2 pi x / period)``."""
    return Component(signal, "sine", period_s=float(period_s), amp=amp, offset=offset, **_common_kw(kw))


def smooth_step(signal: str, t0_s: float, duration_s: float, *, amp=1.0, **kw) -> Component:
    """Raised-cosine step of size ``amp`` starting at t0_s, complete after duration_s (peak rate pi*amp/(2*D))."""
    return Component(signal, "smooth_step", t0_s=float(t0_s), params=(duration_s,), amp=amp, **_common_kw(kw))


def random_walk(signal: str, *, knot_s: float, sigma_per_rt_s: float, drift_per_s: float = 0.0, amp=1.0,
                **kw) -> Component:
    """Brownian path with drift from the counter RNG, linear between knots every knot_s (07 P0-9)."""
    return Component(signal, "random_walk", params=(knot_s, sigma_per_rt_s, drift_per_s), amp=amp,
                     **_common_kw(kw))


def random_levels(signal: str, *, interval_s: float, dist: str = "uniform", a: float = 0.0, b: float = 1.0,
                  lo: float = -math.inf, hi: float = math.inf, choices=None, amp=1.0, offset=0.0,
                  **kw) -> Component:
    """A level per interval from the counter RNG: uniform [a, b], normal (mean a, sd b) or a choice among
    ``choices``; clipped to [lo, hi]. interval_s = math.inf draws one level for the whole run."""
    tv = None if choices is None else np.asarray(choices, dtype=np.float64)
    if dist == "choice" and (tv is None or tv.ndim != 1 or tv.size == 0):
        raise ScenarioError("dist='choice' needs a non-empty 1-D choices list")
    return Component(signal, "random_levels", params=(interval_s, a, b, lo, hi), dist=dist, table_v=tv, amp=amp,
                     offset=offset, **_common_kw(kw))


def sampled(signal: str, interval_s: float, values, *, periodic: bool = False, linear: bool = True, amp=1.0,
            offset=0.0, **kw) -> Component:
    """Uniform-grid table: values (n,) | (Bs, n) | (Bs, E, n) at x = i*interval_s (zoh or linear; periodic with
    period n*interval_s)."""
    v = np.asarray(values, dtype=np.float64)
    n = v.shape[-1]
    return Component(signal, "sampled", params=(interval_s,), table_v=v, periodic=periodic, linear=linear,
                     period_s=float(n * interval_s if periodic else 0.0), amp=amp, offset=offset,
                     **_common_kw(dict(kw, **({"common": False} if v.ndim == 3 else {}))))


def with_burst(comp: Component, on_s: float, off_s: float, phase_s: float = 0.0) -> Component:
    """Gate a component with a burst envelope: on for on_s, off for off_s, repeating (phase relative to t0)."""
    if on_s <= 0 or off_s < 0:
        raise ScenarioError("burst needs on_s > 0 and off_s >= 0")
    return dataclasses.replace(comp, burst_on_s=float(on_s), burst_off_s=float(off_s), burst_phase_s=float(phase_s))


# =====================================================================================================================
# Compile
# =====================================================================================================================

@dataclass(eq=False)
class ScenarioArrays:
    """The compiled scenario: every array the numba kernels take (docs/handoff/scenarios.md).

    sig_i64 (S, NSI): per signal [offset into the output vector, width, first component, component count];
    sig_f64 (S, NSF): [clip lo, clip hi]; elem_default (V,): default value of every output element (0 where an
    absolute component covers it); comp_i64 (C, NCI); comp_f64 (Bs, C, NCF) with Bs in {1, B}; tab_f64 (Bs, NT);
    keys (B, C) uint64 counter-RNG keys; cache0 (B, NK) initial random-walk cache (copy it per run:
    :meth:`new_cache`); compression: scenario.time_compression; layout {signal: (offset, width)}.
    """
    sig_i64: np.ndarray
    sig_f64: np.ndarray
    elem_default: np.ndarray
    comp_i64: np.ndarray
    comp_f64: np.ndarray
    tab_f64: np.ndarray
    keys: np.ndarray
    cache0: np.ndarray
    compression: float
    dt_s: float
    batch: int
    n_rings: int
    n_lines: int
    aggressor_sites: tuple[str, ...]
    layout: dict[str, tuple[int, int]]
    n_out: int
    drives: tuple[str, ...]
    offsets: tuple[str, ...]
    reset: ResetSpec | None
    fingerprint: str
    component_labels: tuple[str, ...]
    compression_report: Mapping[str, Any] = field(default_factory=dict)

    def new_cache(self) -> np.ndarray:
        """A fresh random-walk / random-level cache (B, NK) for one run (results never depend on it)."""
        return self.cache0.copy()

    def new_out(self) -> np.ndarray:
        """(B, V) output rows pre-filled with elem_default (what scen_init_b writes)."""
        return np.broadcast_to(self.elem_default, (self.batch, self.n_out)).copy()

    @property
    def active_signals(self) -> tuple[str, ...]:
        """Signals with components (the ones scen_eval_b evaluates; the others are constant defaults)."""
        return tuple(s.name for s in SIGNALS if self.sig_i64[s.id, SI_NC] > 0)

    @property
    def clipped_signals(self) -> tuple[str, ...]:
        """Signals whose components may reach past a clip limit (SI_CLIP): scen_next_break_b steps them per tick."""
        return tuple(s.name for s in SIGNALS if self.sig_i64[s.id, SI_CLIP] != 0)

    def kernel_args(self, cache: np.ndarray | None = None) -> tuple:
        """(compression, sig_i64, sig_f64, elem_default, comp_i64, comp_f64, tab_f64, keys, cache): the arguments of
        :func:`scen_eval` after ``tick``. Time fields are in ticks of ``dt_s``: the engine must run at that dt."""
        return (self.compression, self.sig_i64, self.sig_f64, self.elem_default, self.comp_i64, self.comp_f64,
                self.tab_f64, self.keys, self.new_cache() if cache is None else cache)

    def row(self, b: int) -> tuple[np.ndarray, np.ndarray]:
        """(comp_f64_b, tab_b) of batch element b (shared row when Bs = 1)."""
        bb = b if self.comp_f64.shape[0] > 1 else 0
        return self.comp_f64[bb], self.tab_f64[bb]

    def evaluate(self, tick: int, cache: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Every signal at one tick as {name: (B, width)} (Python convenience)."""
        out = np.zeros((self.batch, self.n_out))
        cache = self.new_cache() if cache is None else cache
        scen_eval(np.int64(tick), self.compression, self.sig_i64, self.sig_f64, self.elem_default, self.comp_i64,
                  self.comp_f64, self.tab_f64, self.keys, cache, out)
        return {name: out[:, o:o + w].copy() for name, (o, w) in self.layout.items()}

    def next_break(self, signal: str, tick: int, b: int = 0) -> int:
        """scen_next_break_b for one signal and batch element."""
        cf, tb = self.row(b)
        return int(scen_next_break_b(np.int64(signal_id(signal)), np.int64(tick), self.compression, self.sig_i64,
                                     self.comp_i64, cf, tb))

    def render(self, *, n_ticks: int | None = None, ticks=None, stride_ticks: int = 1, tick0: int = 0,
               signals: Sequence[str] | None = None) -> "DisturbanceFrame":
        """Evaluate on ticks tick0 + i*stride (i < n_ticks), or on explicit ``ticks``; returns a DisturbanceFrame."""
        if ticks is None:
            if n_ticks is None:
                raise ScenarioError("render needs n_ticks or ticks")
            ticks = int(tick0) + int(stride_ticks) * np.arange(int(n_ticks), dtype=np.int64)
            stride = int(stride_ticks)
        else:
            ticks = np.asarray(ticks, dtype=np.int64)
            d = np.diff(ticks)
            stride = int(d[0]) if d.size and np.all(d == d[0]) and d[0] > 0 else 0
        if np.any(ticks < 0):
            raise ScenarioError("ticks must be >= 0")
        out = np.zeros((ticks.size, self.batch, self.n_out))
        scen_render(ticks, self.compression, self.sig_i64, self.sig_f64, self.elem_default, self.comp_i64,
                    self.comp_f64, self.tab_f64, self.keys, self.new_cache(), out)
        names = list(self.layout) if signals is None else list(signals)
        sig = {n: out[:, :, self.layout[n][0]:self.layout[n][0] + self.layout[n][1]].copy() for n in names}
        return DisturbanceFrame(dt_s=self.dt_s, tick0=int(ticks[0]) if ticks.size else 0, stride_ticks=stride,
                                signals=sig, ticks=ticks.copy(), interp="linear", source=f"render of "
                                f"{self.fingerprint}", provenance="derived", aggressor_sites=self.aggressor_sites)


def _element_runs(idx: Sequence[int]) -> list[tuple[int, int, list[int]]]:
    """Split element indices into contiguous runs: [(lo, hi, positions_in_idx)]."""
    order = sorted(range(len(idx)), key=lambda i: idx[i])
    runs: list[tuple[int, int, list[int]]] = []
    for pos in order:
        e = idx[pos]
        if runs and runs[-1][1] == e:
            lo, hi, ps = runs[-1]
            runs[-1] = (lo, hi + 1, ps + [pos])
        elif runs and runs[-1][1] > e:
            raise ScenarioError(f"element {e} listed twice")
        else:
            runs.append((e, e + 1, [pos]))
    return runs


#: Snap tolerance: a time within this many ticks (plus 1e-12 relative) of a whole tick is put on it, so that
#: user-given edges (0.1 s at dt = 1 us is 99999.99999999999 ticks in float) fall on the intended tick (design choice).
_SNAP_TICKS = 1e-6


def _ticks(v_s: float, dt_s: float) -> float:
    """Seconds -> plant ticks (float), snapped to the nearest whole tick when within _SNAP_TICKS."""
    v = float(v_s)
    if not math.isfinite(v):
        return v
    q = v / dt_s
    r = round(q)
    return float(r) if abs(q - r) <= _SNAP_TICKS + 1e-12 * abs(q) else q


def _ticks_arr(v_s: np.ndarray, dt_s: float) -> np.ndarray:
    q = np.asarray(v_s, dtype=np.float64) / dt_s
    r = np.round(q)
    return np.where(np.abs(q - r) <= _SNAP_TICKS + 1e-12 * np.abs(q), r, q)


def _rows_elems(tv: np.ndarray) -> np.ndarray:
    """Table values as (rows, E, n): (n,) -> (1, 1, n); (Bs, n) -> (Bs, 1, n); (Bs, E, n) unchanged."""
    if tv.ndim == 1:
        return tv[None, None, :]
    if tv.ndim == 2:
        return tv[:, None, :]
    if tv.ndim == 3:
        return tv
    raise ScenarioError(f"table values must be (n,), (Bs, n) or (Bs, E, n), got shape {tv.shape}")


def _check_e(c: Component, tv: np.ndarray, idx: list[int]) -> None:
    if tv.shape[1] != len(idx):
        raise ScenarioError(f"{c.label or c.kind}: per-element table has {tv.shape[1]} rows but the component "
                            f"covers {len(idx)} elements")


def _smooth_curvature(c: Component, amp_max: float) -> float:
    """Upper bound of |d2 value/dx2| for the smooth kinds (signal units per axis s^2).

    raw bounds (axis s): rect_sine |r'| <= pi/P, |r''| <= (pi/P)^2, r in [0, 1]; sine 2 pi/P, (2 pi/P)^2, [-1, 1];
    smooth_step pi/(2D), (pi/D)^2/2, [0, 1]. With the bernoulli_td map m(r) = 2 r (1 - r):
    m'' = 2 r'' (1 - 2 r) - 4 r'^2, so |m''| <= 2 |r''| max|1 - 2 r| + 4 r'^2."""
    if c.kind == "rect_sine":
        c1, c2, m12 = math.pi / c.period_s, (math.pi / c.period_s) ** 2, 1.0
    elif c.kind == "sine":
        c1, c2, m12 = 2.0 * math.pi / c.period_s, (2.0 * math.pi / c.period_s) ** 2, 3.0
    elif c.kind == "smooth_step":
        c1, c2, m12 = math.pi / (2.0 * c.params[0]), 0.5 * (math.pi / c.params[0]) ** 2, 1.0
    elif c.map == "bernoulli_td" and c.kind in ("trapezoid", "pwl") or (c.kind == "sampled" and c.linear
                                                                         and c.map == "bernoulli_td"):
        # linear segments (r'' = 0) become parabolas under the map: |m''| = 4 r'^2 with r' the steepest segment
        c1 = component_max_rate(dataclasses.replace(c, amp=1.0, map="identity"))
        return amp_max * 4.0 * c1 * c1
    else:
        return 0.0
    if c.map == "bernoulli_td":
        return amp_max * (2.0 * c2 * m12 + 4.0 * c1 * c1)
    return amp_max * c2


def _raw_range(c: Component) -> tuple[float, float]:
    """Range of the raw generator value after the map (before amp and offset) while the component is active."""
    k = c.kind
    p = c.params
    if k == "const":
        lo, hi = 1.0, 1.0
    elif k in ("trapezoid", "rect_sine", "smooth_step"):
        lo, hi = 0.0, 1.0
    elif k == "sine":
        lo, hi = -1.0, 1.0
    elif k in ("pwl", "sampled"):
        v = np.asarray(c.table_v, dtype=np.float64)
        lo, hi = (float(np.nanmin(v)), float(np.nanmax(v))) if v.size and np.any(~np.isnan(v)) else (0.0, 0.0)
    elif k == "random_walk":
        lo, hi = (0.0, 0.0) if (p[1] == 0.0 and p[2] == 0.0) else (-math.inf, math.inf)
    elif k == "random_levels":
        if c.dist == "uniform":
            lo, hi = min(p[1], p[2]), max(p[1], p[2])
        elif c.dist == "normal":
            lo, hi = (p[1], p[1]) if p[2] == 0.0 else (-math.inf, math.inf)
        else:
            v = np.asarray(c.table_v, dtype=np.float64)
            lo, hi = float(np.min(v)), float(np.max(v))
        lo, hi = max(lo, p[3]), min(hi, p[4])
    else:
        lo, hi = -math.inf, math.inf
    if c.map == "bernoulli_td":                              # m(r) = 2 r (1 - r): concave, peak 0.5 at r = 0.5
        if math.isfinite(lo) and math.isfinite(hi):
            lo, hi = min(2.0 * lo * (1.0 - lo), 2.0 * hi * (1.0 - hi)), \
                (0.5 if lo <= 0.5 <= hi else max(2.0 * lo * (1.0 - lo), 2.0 * hi * (1.0 - hi)))
        else:
            lo, hi = -math.inf, 0.5
    return lo, hi


def _component_value_range(c: Component, amp: np.ndarray, off: np.ndarray) -> tuple[float, float]:
    """[min, max] of a component's contribution over all worlds, 0 included (inactive outside its window or in a
    burst gap). A superset of the reachable values; used only to decide SI_CLIP."""
    rlo, rhi = _raw_range(c)
    vals = [0.0]
    for a, o in zip(np.atleast_1d(amp), np.atleast_1d(off)):
        for r in (rlo, rhi):
            if a == 0.0:
                vals.append(float(o))
            elif math.isfinite(r):
                vals.append(float(o + a * r))
            else:
                vals.append(math.copysign(math.inf, a * r))
    return min(vals), max(vals)


def _compile(sc: Scenario, *, dt_s: float, n_rings: int, n_lines: int, batch: int, seed: int,
             aggressor_sites: Sequence[str] | None, check_compression: bool) -> ScenarioArrays:
    if not (dt_s > 0 and math.isfinite(dt_s)):
        raise ScenarioError(f"dt_s must be a positive finite number, got {dt_s}")
    if n_rings < 1 or n_lines < 1 or batch < 1:
        raise ScenarioError("n_rings, n_lines and batch must be >= 1")
    pend = sc.pending_aggressors
    if pend:
        raise ScenarioError(f"scenario {sc.name!r}: aggressor(s) {list(pend)} have a target but no power waveform; "
                            f"run mrm.scenarios.calibrate.calibrate_scenario with the plant's thermal response "
                            f"first (07 P1-4: the power is calibrated to the target)")
    sites = tuple(aggressor_sites) if aggressor_sites is not None else sc.aggressor_sites
    for s in sc.aggressor_sites:
        if s not in sites:
            raise ScenarioError(f"aggressor site {s!r} is not among the engine's aggressor sites {list(sites)}")
    widths = {"die": 1, "ring": n_rings, "line": n_lines, "aggressor": len(sites)}

    # signal layout (fixed order of SIGNALS)
    layout: dict[str, tuple[int, int]] = {}
    off = 0
    for s in SIGNALS:
        w = widths[s.level]
        layout[s.name] = (off, w)
        off += w
    n_out = off

    # defaults
    elem_default = np.zeros(n_out)
    for s in SIGNALS:
        o, w = layout[s.name]
        v = sc.defaults.get(s.name, s.default)
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 0:
            elem_default[o:o + w] = float(a)
        elif a.shape == (w,):
            elem_default[o:o + w] = a
        else:
            raise ScenarioError(f"default of {s.name} must be a scalar or ({w},), got shape {a.shape}")

    # component entries split into contiguous element runs
    entries: list[tuple[Component, int, int, list[int]]] = []
    idx_of: dict[int, list[int]] = {}
    for c in sc.all_components:
        spec = c.spec
        w = widths[spec.level]
        if c.site is not None:
            idx = [sites.index(c.site)]
        elif c.elements is None:
            idx = list(range(w))
        else:
            idx = [int(e) for e in c.elements]
        if not idx:
            continue
        idx_of[id(c)] = idx
        bad = [e for e in idx if not 0 <= e < w]
        if bad:
            raise ScenarioError(f"{c.label or c.kind} on {c.signal}: elements {bad} out of range 0..{w - 1}")
        for lo, hi, pos in _element_runs(idx):
            entries.append((c, lo, hi, pos))
    entries.sort(key=lambda t: SIGNAL_BY_NAME[t[0].signal].id)

    # absolute components replace the default on their elements; two on one element is an error
    abs_owner: dict[tuple[str, int], Component] = {}
    for c, lo, hi, _ in entries:
        if not c.absolute:
            continue
        o, _w = layout[c.signal]
        for e in range(lo, hi):
            k = (c.signal, e)
            if k in abs_owner and abs_owner[k] is not c:
                raise ScenarioError(f"{c.signal}[{e}] is set absolutely by both {abs_owner[k].label or 'a component'!r}"
                                    f" and {c.label or c.kind!r}; one absolute definition per element")
            abs_owner[k] = c
            elem_default[o + e] = 0.0

    # an additive component on an element whose default is NaN (tx.bias_v: not driven) would give NaN silently
    for c, lo, hi, _ in entries:
        if c.absolute:
            continue
        o, _w = layout[c.signal]
        nan_el = [e for e in range(lo, hi) if math.isnan(elem_default[o + e])]
        if nan_el:
            raise ScenarioError(f"{c.label or c.kind} adds to {c.signal}{nan_el}, whose default is NaN (not driven); "
                                f"make it absolute or set the signal default (e.g. scenario.bias_v)")

    # duplicate random draws on one signal (same stream and slot, different components) are almost always a bug
    seen_slots: dict[tuple[str, str, int], Component] = {}
    for c, _lo, _hi, _ in entries:
        if c.is_random:
            k = (c.signal, c.stream, c.slot)
            if k in seen_slots and seen_slots[k] is not c:
                raise ScenarioError(f"components {seen_slots[k].label or seen_slots[k].kind!r} and "
                                    f"{c.label or c.kind!r} on {c.signal} share stream {c.stream} and slot {c.slot} "
                                    f"(identical draws): give them different labels or links")
            seen_slots[k] = c

    # per-world rows
    wc = [c.world_count() for c, *_ in entries]
    for n in wc:
        if n not in (1, batch):
            raise ScenarioError(f"per-world arrays have {n} rows but the batch is {batch}")
    bs = batch if any(n == batch for n in wc) and batch > 1 else 1

    C = len(entries)
    comp_i64 = np.zeros((C, NCI), dtype=np.int64)
    comp_f64 = np.zeros((bs, C, NCF), dtype=np.float64)
    tables: list[np.ndarray] = []           # each (bs, len)
    tab_off = 0
    cache_off = 0
    sig_i64 = np.zeros((len(SIGNALS), NSI), dtype=np.int64)
    sig_f64 = np.zeros((len(SIGNALS), NSF), dtype=np.float64)
    for s in SIGNALS:
        o, w = layout[s.name]
        sig_i64[s.id, SI_OFF] = o
        sig_i64[s.id, SI_WIDTH] = w
        sig_f64[s.id, SF_LO] = s.lo
        sig_f64[s.id, SF_HI] = s.hi

    def tile(a: np.ndarray) -> np.ndarray:            # (n_rows, ...) -> (bs, ...)
        return np.broadcast_to(a, (bs,) + a.shape[1:]) if a.shape[0] == 1 else a

    # curved components on one signal share its foh_tol (their deviations from linear add up)
    n_smooth: dict[str, int] = {}
    for c, *_r in entries:
        if _smooth_curvature(c, 1.0) > 0.0:
            n_smooth[c.signal] = n_smooth.get(c.signal, 0) + 1
    # reachable value range per output element (for SI_CLIP): default (0 under an absolute component) + components
    v_lo = np.nan_to_num(elem_default.copy(), nan=0.0)
    v_hi = v_lo.copy()

    for ci, (c, lo, hi, pos) in enumerate(entries):
        spec = c.spec
        sid = spec.id
        if sig_i64[sid, SI_NC] == 0:
            sig_i64[sid, SI_C0] = ci
        sig_i64[sid, SI_NC] += 1
        flags = 0
        flags |= FL_SLOW if c.slow else 0
        flags |= FL_ABSOLUTE if c.absolute else 0
        flags |= FL_COMMON if c.common else 0
        flags |= FL_HOLD_BEFORE if c.hold_before else 0
        flags |= FL_HOLD_AFTER if c.hold_after else 0
        flags |= FL_PERIODIC if c.periodic else 0
        flags |= FL_LINEAR if c.linear else 0
        comp_i64[ci, CI_KIND] = KINDS.index(c.kind)
        comp_i64[ci, CI_SIGNAL] = sid
        comp_i64[ci, CI_E_LO] = lo
        comp_i64[ci, CI_E_HI] = hi
        comp_i64[ci, CI_MAP] = MAPS.index(c.map)
        comp_i64[ci, CI_DIST] = DISTS.index(c.dist)
        comp_i64[ci, CI_SLOT] = c.slot
        # tables
        tab = None
        if c.kind == "pwl":
            tt = np.asarray(c.table_t, dtype=np.float64)
            n = tt.size
            tv = _rows_elems(np.asarray(c.table_v, dtype=np.float64))                  # (rows, E, n)
            if tv.shape[-1] != n:
                raise ScenarioError(f"{c.label or 'pwl'}: table_v has {tv.shape[-1]} points, table_t {n}")
            if tv.shape[1] > 1:
                _check_e(c, tv, idx_of[id(c)])
                tv = tv[:, pos, :]
                flags |= FL_PER_ELEMENT
            tv = tile(tv)
            tab = np.concatenate([np.broadcast_to(_ticks_arr(tt, dt_s), (bs, n)), tv.reshape(bs, -1)], axis=1)
            comp_i64[ci, CI_TAB_LEN] = n
        elif c.kind == "sampled":
            tv = _rows_elems(np.asarray(c.table_v, dtype=np.float64))
            n = tv.shape[-1]
            if n < 1:
                raise ScenarioError(f"{c.label or 'sampled'}: empty table")
            if tv.shape[1] > 1:
                _check_e(c, tv, idx_of[id(c)])
                tv = tv[:, pos, :]
                flags |= FL_PER_ELEMENT
            tab = tile(tv).reshape(bs, -1)
            comp_i64[ci, CI_TAB_LEN] = n
        elif c.kind == "random_levels" and c.dist == "choice":
            tv = np.asarray(c.table_v, dtype=np.float64).ravel()
            tab = np.broadcast_to(tv, (bs, tv.size))
            comp_i64[ci, CI_TAB_LEN] = tv.size
        if tab is not None:
            comp_i64[ci, CI_TAB_OFF] = tab_off
            tables.append(np.ascontiguousarray(tab))
            tab_off += tab.shape[1]
        if c.kind == "random_walk":
            comp_i64[ci, CI_CACHE_OFF] = cache_off
            cache_off += 3 * (1 if c.common else hi - lo)
        elif c.kind == "random_levels":
            comp_i64[ci, CI_CACHE_OFF] = cache_off
            cache_off += 2 * (1 if c.common else hi - lo)
        comp_i64[ci, CI_FLAGS] = flags
        # floats
        amp = tile(_as_world(c.amp, "amp")[:, None])[:, 0]
        offv = tile(_as_world(c.offset, "offset")[:, None])[:, 0]
        tk = lambda v: _ticks(v, dt_s)                       # noqa: E731  seconds -> ticks, snapped to the grid
        comp_f64[:, ci, CF_T0] = tk(c.t0_s)
        comp_f64[:, ci, CF_T1] = tk(c.t1_s)
        comp_f64[:, ci, CF_AMP] = amp
        comp_f64[:, ci, CF_OFFSET] = offv
        comp_f64[:, ci, CF_PERIOD] = tk(c.period_s)
        comp_f64[:, ci, CF_PHASE] = tk(c.phase_s)
        pp = list(c.params)
        if c.kind == "trapezoid":
            pp[0], pp[1], pp[2] = tk(pp[0]), tk(pp[1]), tk(pp[2])
            excess = pp[0] + pp[1] + pp[2] - tk(c.period_s)
            if excess > 0:                                   # snapping pushed the sum past the period
                pp[2] = max(0.0, pp[2] - excess)
        elif c.kind == "smooth_step":
            pp[0] = tk(pp[0])
        elif c.kind == "random_walk":
            knot = tk(pp[0])
            pp = [knot, pp[1] * math.sqrt(knot * dt_s), pp[2] * knot * dt_s, 0.0, 0.0]   # sd and mean per knot
        elif c.kind in ("random_levels", "sampled"):
            pp[0] = tk(pp[0])
        for k in range(5):
            comp_f64[:, ci, CF_P0 + k] = pp[k]
        comp_f64[:, ci, CF_BURST_ON] = tk(c.burst_on_s)
        comp_f64[:, ci, CF_BURST_OFF] = tk(c.burst_off_s)
        comp_f64[:, ci, CF_BURST_PHASE] = tk(c.burst_phase_s)
        kappa = _smooth_curvature(c, float(np.max(np.abs(amp))))
        tol = spec.foh_tol / max(1, n_smooth.get(c.signal, 1))
        comp_f64[:, ci, CF_HSMOOTH] = math.sqrt(8.0 * tol / kappa) / dt_s if kappa > 0 else math.inf
        clo, chi = _component_value_range(c, np.asarray(amp), np.asarray(offv))
        o_sig = layout[c.signal][0]
        v_lo[o_sig + lo:o_sig + hi] += clo
        v_hi[o_sig + lo:o_sig + hi] += chi
        # aliasing guard (design choice): a fast periodic kind needs >= 4 ticks per period
        if c.kind in ("sine", "rect_sine", "trapezoid") and not c.slow and c.period_s < 4 * dt_s:
            warnings.warn(f"{c.label or c.kind} on {c.signal}: period {c.period_s} s is below 4 plant ticks "
                          f"({dt_s} s); it aliases on the tick grid", RuntimeWarning, stacklevel=3)

    for s in SIGNALS:
        o, w = layout[s.name]
        if w == 0 or sig_i64[s.id, SI_NC] == 0:
            continue
        with np.errstate(invalid="ignore"):
            span = max(1.0, float(np.nanmax(np.abs(np.concatenate([v_lo[o:o + w], v_hi[o:o + w]])))))
        eps = 1e-12 * span if math.isfinite(span) else 0.0
        if np.any(v_lo[o:o + w] < s.lo - eps) or np.any(v_hi[o:o + w] > s.hi + eps):
            sig_i64[s.id, SI_CLIP] = 1

    tab_f64 = np.concatenate(tables, axis=1) if tables else np.zeros((bs, 0))
    seeds = _rng.batch_seeds(seed, batch)
    keys = np.zeros((batch, C), dtype=np.uint64)
    for ci, (c, *_r) in enumerate(entries):
        sid = _rng.stream_id(c.stream)
        for b in range(batch):
            keys[b, ci] = _rng.stream_key(np.int64(seeds[b]), np.int64(sid))
    cache0 = np.zeros((batch, cache_off), dtype=np.float64)
    for ci, (c, lo, hi, _) in enumerate(entries):
        if c.is_random:
            o = comp_i64[ci, CI_CACHE_OFF]
            w = 3 if c.kind == "random_walk" else 2
            for e in range(1 if c.common else hi - lo):
                cache0[:, o + w * e] = -1.0                  # no cached knot / interval

    rep: dict[str, Any] = {}
    if check_compression:
        rep = check_time_compression(sc)
    return ScenarioArrays(
        sig_i64=sig_i64, sig_f64=sig_f64, elem_default=elem_default, comp_i64=comp_i64,
        comp_f64=np.ascontiguousarray(comp_f64), tab_f64=np.ascontiguousarray(tab_f64), keys=keys, cache0=cache0,
        compression=float(sc.time_compression), dt_s=dt_s, batch=batch, n_rings=n_rings, n_lines=n_lines,
        aggressor_sites=sites, layout=layout, n_out=n_out, drives=sc.drives, offsets=sc.offsets, reset=sc.reset,
        fingerprint=sc.fingerprint(), component_labels=tuple(c.label or c.kind for c, *_ in entries),
        compression_report=rep)


# =====================================================================================================================
# DisturbanceFrame (07 P0-9: recorded profiles)
# =====================================================================================================================

@dataclass(eq=False)
class DisturbanceFrame:
    """Signals on a tick grid: ``signals[name]`` is (T, Bs, K) float64 at ticks ``tick0 + i*stride_ticks`` (or the
    explicit ``ticks``), Bs in {1, B}, K the signal width. ``interp``: how to replay between samples (linear | zoh).
    ``slow``: signals recorded on the real time axis (time-compressible). Frame values are TOTAL signal values and
    replace the signal default on replay, except for the signals listed in ``additive`` (recorded deltas). Use
    :meth:`to_components` or Scenario.from_frame to replay it through the kernel (07 P0-9: frames only for recorded
    profiles). Zero-width and all-NaN (undriven) signals are not replayed. ``aggressor_sites`` names the columns of
    ``thermal.aggressor_w`` (needed to replay aggressor power; render fills it)."""
    dt_s: float
    tick0: int
    stride_ticks: int
    signals: dict[str, np.ndarray]
    ticks: np.ndarray | None = None
    interp: str = "linear"
    slow: frozenset = frozenset()
    additive: frozenset = frozenset()
    provenance: str = "read-off"
    source: str = ""
    note: str = ""
    aggressor_sites: tuple[str, ...] = ()

    def __post_init__(self):
        if self.interp not in ("linear", "zoh"):
            raise ScenarioError("interp must be linear or zoh")
        self.signals = dict(self.signals)                  # never mutate the caller's mapping
        self.aggressor_sites = tuple(self.aggressor_sites)
        T = None
        for k, v in self.signals.items():
            signal_id(k)
            a = np.asarray(v, dtype=np.float64)
            if a.ndim == 1:
                a = a[:, None, None]
            elif a.ndim == 2:
                a = a[:, None, :]
            if a.ndim != 3:
                raise ScenarioError(f"frame signal {k}: expected (T, Bs, K), got {a.shape}")
            if T is not None and a.shape[0] != T:
                raise ScenarioError("frame signals must share T")
            T = a.shape[0]
            self.signals[k] = a

    @property
    def n_samples(self) -> int:
        return next(iter(self.signals.values())).shape[0] if self.signals else 0

    def times_s(self) -> np.ndarray:
        """Sample times in s (tick * dt)."""
        t = self.ticks if self.ticks is not None else self.tick0 + self.stride_ticks * np.arange(self.n_samples)
        return np.asarray(t, dtype=np.float64) * self.dt_s

    def to_components(self) -> tuple[Component, ...]:
        """One ``sampled`` component per signal (per element and per world)."""
        if self.stride_ticks <= 0:
            raise ScenarioError("only a uniform tick grid (stride_ticks > 0) can be replayed")
        out = []
        for name, a in self.signals.items():
            spec = SIGNAL_BY_NAME[name]
            if a.shape[2] == 0 or np.all(np.isnan(a)):
                continue
            vals = np.transpose(a, (1, 2, 0))                     # (Bs, K, T)
            kw = dict(params=(self.stride_ticks * self.dt_s,), t0_s=self.tick0 * self.dt_s,
                      linear=self.interp == "linear", hold_before=True, hold_after=True,
                      absolute=name not in self.additive, slow=name in self.slow, provenance=self.provenance,
                      source=self.source, note=self.note or "recorded profile", stream="scenario_frame")
            if spec.level == "aggressor":
                if len(self.aggressor_sites) != a.shape[2]:
                    raise ScenarioError(f"frame has {a.shape[2]} aggressor columns but aggressor_sites "
                                        f"{list(self.aggressor_sites)}; name the sites to replay aggressor power")
                for j, site in enumerate(self.aggressor_sites):
                    out.append(Component(name, "sampled", site=site, table_v=vals[:, j, :],
                                         label=f"frame:{name}:{site}", **kw))
                continue
            out.append(Component(name, "sampled", table_v=vals, common=vals.shape[1] == 1, label=f"frame:{name}",
                                 **kw))
        return tuple(out)


# =====================================================================================================================
# Time compression (07 P0-10)
# =====================================================================================================================

def component_max_rate(c: Component) -> float:
    """Upper bound of |d contribution/dt| on the component's own time axis, in signal units per s. Jumps (steps,
    square edges, random levels) are not ramps and count as 0; a random walk counts drift + 4 sigma/sqrt(knot)."""
    amp = float(np.max(np.abs(_as_world(c.amp, "amp"))))
    p = c.params
    if c.kind == "pwl":
        t = np.asarray(c.table_t, dtype=np.float64)
        v = np.asarray(c.table_v, dtype=np.float64)
        dt = np.diff(t)
        dv = np.abs(np.diff(v, axis=-1))
        ok = dt > 0
        return amp * float(np.max(dv[..., ok] / dt[ok])) if np.any(ok) else 0.0
    if c.kind == "trapezoid":
        r = [1.0 / x for x in (p[0], p[2]) if x > 0]
        return amp * max(r) if r else 0.0
    if c.kind == "rect_sine":
        return amp * math.pi / c.period_s
    if c.kind == "sine":
        return amp * 2.0 * math.pi / c.period_s
    if c.kind == "smooth_step":
        return amp * math.pi / (2.0 * p[0])
    if c.kind == "random_walk":
        return amp * (abs(p[2]) + 4.0 * p[1] / math.sqrt(p[0]))
    if c.kind == "sampled" and c.linear:
        v = np.asarray(c.table_v, dtype=np.float64)
        if v.shape[-1] < 2:
            return 0.0
        d = np.abs(np.diff(v, axis=-1))
        if c.periodic:
            d = np.concatenate([d, np.abs(v[..., :1] - v[..., -1:])], axis=-1)
        return amp * float(np.max(d)) / p[0]
    return 0.0


def check_time_compression(sc: Scenario, *, fraction: float = COMPRESSION_RATE_FRACTION) -> dict:
    """The 07 P0-10 load-time rule: every compressed (``slow``) component, at time_compression c, ramps at no more
    than ``fraction`` (1 %) of the fast-transient rate (Scenario.fast_rate_ref_k_per_s, default the slide-10 peak
    2127 K/s). Rates are compared in K/s equivalents (signal ``rate_to_k_per_s``; laser frequency via
    Lightmatter's 15 GHz/degC). Slow components on signals without an equivalence (dB) only warn. Returns a report
    {component: {rate_k_per_s, limit_k_per_s, max_compression}}; raises CompressionError on a violation."""
    c = float(sc.time_compression)
    limit = fraction * float(sc.fast_rate_ref_k_per_s)
    rep: dict[str, Any] = {"time_compression": c, "limit_k_per_s": limit, "components": {}}
    errors = []
    for comp in sc.all_components:
        if not comp.slow:
            continue
        rate = component_max_rate(comp)
        f = comp.spec.rate_to_k_per_s
        name = comp.label or comp.kind
        if f is None:
            rep["components"][name] = {"rate_per_s": rate * c, "unit": comp.spec.unit, "checked": False}
            if c > 1.0 and rate > 0.0:
                warnings.warn(f"{name} on {comp.signal} is time-compressed but its unit ({comp.spec.unit}) has no "
                              f"K/s equivalent; the 07 P0-10 rate rule cannot check it", RuntimeWarning,
                              stacklevel=3)
            continue
        rk = rate * f
        cmax = limit / rk if rk > 0 else math.inf
        rep["components"][name] = {"rate_k_per_s": rk * c, "limit_k_per_s": limit, "max_compression": cmax,
                                   "checked": True}
        if rk * c > limit * (1 + 1e-12):
            errors.append(f"{name} on {comp.signal}: {rk:.4g} K/s-equivalent x compression {c:g} = {rk * c:.4g} "
                          f"K/s exceeds {fraction:.0%} of the fast-transient rate {sc.fast_rate_ref_k_per_s:.4g} "
                          f"K/s (largest allowed compression {cmax:.4g})")
    if errors:
        raise CompressionError("07 P0-10 time_compression rule: " + "; ".join(errors))
    return rep


#: Metrics the mandatory convergence experiment compares (07 P0-10).
CONVERGENCE_METRICS = ("max_abs_detuning_pm", "wrong_line_events", "fsr_hop_events", "heater_range_used_codes",
                       "integrated_errors")
#: Agreement rule per metric against the uncompressed run: (absolute tolerance, relative tolerance); a metric
#: passes when |x_c - x_1| <= max(abs, rel * |x_1|). Event counts must match exactly. These tolerances are design
#: choices (assumed), not physics: 0.2 pm is the 07 P0-10 max_detuning_step_pm; 16 codes is 2**-12 of a 16-bit DAC.
DEFAULT_CONVERGENCE_TOLERANCES: dict[str, tuple[float, float]] = {
    "max_abs_detuning_pm": (0.2, 0.05),
    "wrong_line_events": (0.0, 0.0),
    "fsr_hop_events": (0.0, 0.0),
    "heater_range_used_codes": (16.0, 0.01),
    "integrated_errors": (1.0, 0.10),
}


@dataclass(frozen=True)
class ConvergenceCertificate:
    """Evidence that a compressed run of this scenario on this plant agrees with the uncompressed run (07 P0-10).
    scenario_fingerprint excludes the compression; plant_fingerprint is the caller's config hash."""
    scenario_fingerprint: str
    plant_fingerprint: str
    factors: tuple[float, ...]
    max_validated_factor: float
    metrics: Mapping[str, Mapping[str, float]]
    tolerances: Mapping[str, tuple[float, float]]

    def to_dict(self) -> dict:
        return {"scenario_fingerprint": self.scenario_fingerprint, "plant_fingerprint": self.plant_fingerprint,
                "factors": list(self.factors), "max_validated_factor": self.max_validated_factor,
                "metrics": {str(k): dict(v) for k, v in self.metrics.items()},
                "tolerances": {k: list(v) for k, v in self.tolerances.items()}}

    @classmethod
    def from_dict(cls, d: Mapping) -> "ConvergenceCertificate":
        return cls(scenario_fingerprint=d["scenario_fingerprint"], plant_fingerprint=d["plant_fingerprint"],
                   factors=tuple(float(x) for x in d["factors"]), max_validated_factor=float(d["max_validated_factor"]),
                   metrics=d["metrics"], tolerances={k: tuple(v) for k, v in d["tolerances"].items()})


@dataclass(frozen=True)
class ConvergenceReport:
    """Outcome of :func:`assess_convergence`: passed, per-factor per-metric details, and the certificate (or None)."""
    passed: bool
    details: Mapping[float, Mapping[str, Mapping[str, float]]]
    failures: tuple[str, ...]
    certificate: ConvergenceCertificate | None


def convergence_plan(sc: Scenario, factors: Sequence[float] = (1.0, 10.0, 100.0), *, cycles: float = 1.0,
                     cycle_s: float | None = None) -> dict[float, tuple[Scenario, float]]:
    """The mandatory 07 P0-10 experiment: {factor: (scenario at that compression, simulated duration in s)} for
    ``cycles`` slow cycles (default one soak cycle: Scenario.slow_cycle_s). Run each on the same plant and seeds,
    collect CONVERGENCE_METRICS, and pass them to :func:`assess_convergence`."""
    cyc = cycle_s if cycle_s is not None else sc.slow_cycle_s
    if cyc is None:
        raise ScenarioError("convergence_plan needs cycle_s (the scenario declares no slow_cycle_s)")
    fs = sorted({float(f) for f in factors} | {1.0})
    out = {}
    for f in fs:
        s = sc.with_compression(f)
        check_time_compression(s)
        out[f] = (s, cycles * cyc / f)
    return out


def assess_convergence(sc: Scenario, results: Mapping[float, Mapping[str, float]], *,
                       tolerances: Mapping[str, tuple[float, float]] | None = None,
                       plant_fingerprint: str = "") -> ConvergenceReport:
    """Compare the metrics of each compressed run with the factor-1 run (07 P0-10: max detuning, wrong-line and
    FSR-hop events, heater range used and integrated errors must agree). ``results``: {factor: {metric: value}}.
    Returns a report; its certificate validates every factor up to the largest one for which all smaller tested
    factors also pass."""
    tol = dict(DEFAULT_CONVERGENCE_TOLERANCES)
    tol.update(tolerances or {})
    res = {float(k): v for k, v in results.items()}
    if 1.0 not in res:
        raise ScenarioError("assess_convergence needs the factor-1 (uncompressed) results")
    ref = res[1.0]
    missing = [m for m in CONVERGENCE_METRICS if m not in ref]
    if missing:
        raise ScenarioError(f"factor-1 results lack the metrics {missing} (07 P0-10)")
    details: dict[float, dict[str, dict[str, float]]] = {}
    failures: list[str] = []
    passed_upto = 1.0
    ok_chain = True
    for f in sorted(res):
        if f == 1.0:
            continue
        d = {}
        ok = True
        for m in CONVERGENCE_METRICS:
            if m not in res[f]:
                failures.append(f"factor {f:g}: metric {m} missing")
                ok = False
                continue
            a, r = tol[m]
            x1 = float(ref[m])
            xc = float(res[f][m])
            allowed = max(a, r * abs(x1))
            good = abs(xc - x1) <= allowed
            d[m] = {"reference": x1, "compressed": xc, "allowed": allowed, "pass": float(good)}
            if not good:
                ok = False
                failures.append(f"factor {f:g}: {m} = {xc:.6g} vs {x1:.6g} (allowed +-{allowed:.3g})")
        details[f] = d
        if ok and ok_chain:
            passed_upto = f
        else:
            ok_chain = False
    passed = not failures
    cert = None
    if passed_upto > 1.0:
        cert = ConvergenceCertificate(scenario_fingerprint=sc.fingerprint(include_compression=False),
                                      plant_fingerprint=plant_fingerprint, factors=tuple(sorted(res)),
                                      max_validated_factor=passed_upto,
                                      metrics={str(k): dict(v) for k, v in res.items()},
                                      tolerances=tol)
    return ConvergenceReport(passed=passed, details=details, failures=tuple(failures), certificate=cert)


def require_compression_validated(sc: Scenario, certificate: ConvergenceCertificate | None, *,
                                  plant_fingerprint: str = "") -> None:
    """Raise CompressionError unless the scenario is uncompressed or the certificate covers its compression on
    this plant (07 P0-10: "a convergence test is mandatory before any compressed result is reported")."""
    c = float(sc.time_compression)
    if c == 1.0:
        return
    if certificate is None:
        raise CompressionError(f"scenario {sc.name!r} runs at time_compression {c:g} without a convergence "
                               f"certificate; run convergence_plan / assess_convergence first (07 P0-10)")
    if certificate.scenario_fingerprint != sc.fingerprint(include_compression=False):
        raise CompressionError("the convergence certificate belongs to a different scenario")
    if certificate.plant_fingerprint != plant_fingerprint:
        raise CompressionError("the convergence certificate belongs to a different plant configuration")
    if c > certificate.max_validated_factor * (1 + 1e-12):
        raise CompressionError(f"time_compression {c:g} exceeds the validated factor "
                               f"{certificate.max_validated_factor:g}")
