"""Named scenario library: Lightmatter's published experiments and the disturbances the sources name.

Every builder returns a :class:`mrm.scenarios.generators.Scenario`. Numbers come from docs/research/01 (with its
verification-log corrections), 04 and the textbook REF (src/physics/index.ts of the lightmatter-capstone textbook);
each carries its provenance word and source. The textbook's scenario ids (Ch17 groups A and B, Ch18) are given in
brackets. Builders take keyword arguments whose names follow the unit-suffix grammar (``_s``, ``_c``, ``_k``,
``_db``, ``_hz``, ``_nm``, ...), so ``scenario.params`` in a config maps onto them one to one.

The four disturbance paths of 07 P1-4 stay separate: base temperature (soak, ambient, cold start), on-die aggressor
POWER (A1, A2, direct waveforms), XPU power events (B3), laser-only wavelength (sweep emulation A3, drift, failover).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mrm.scenarios import generators as g
from mrm.scenarios.calibrate import LAMBDA_REF_M, LM_DLAMBDA_DT_M_PER_K, AggressorTarget

__all__ = [
    "static", "a1_slide10_aggressor", "a2_808_aggressor", "soak", "xpu_power_step", "laser_drift",
    "laser_sweep_emulation", "connector_remate", "fibre_loss_step", "traffic_start_stop", "squelch", "idle",
    "training_pattern",
    "ones_density_steps", "bias_ripple", "bias_step", "laser_failover", "line_enable", "cold_start", "warm_start",
    "reloma_plane", "aggressor_direct", "aggressor_from_config",
    # published numbers (with provenance comments)
    "A1_T_MAX_C", "A1_DELTA_T_K", "A1_PERIOD_S", "A1_T_BASE_C", "A2_RATE_K_PER_S", "A2_T_LOW_C", "A2_T_HIGH_C",
    "A2_PERIOD_S", "SOAK_T_LOW_C", "SOAK_T_HIGH_C", "SOAK_CYCLE_S", "SOAK_N_CYCLES", "XPU_RATE_K_PER_S_RANGE",
    "XPU_T_PEAK_RATE_S", "LASER_DRIFT_NM_PER_24H", "LASER_POWER_DRIFT_PCT_PER_24H", "SWEEP_RATE_NM_PER_S",
    "SWEEP_SPAN_NM", "SWEEP_PERIOD_S", "REMATE_N", "REMATE_MEAN_DB", "REMATE_MEAN_PLUS_3SIGMA_DB",
    "SUN_STEP_S", "SQUELCH_OMA_DBM", "JUNCTION_RANGE_C", "BIAS_RANGE_V",
]

_C0 = 299_792_458.0       # speed of light, m/s (exact)
_T0 = 273.15              # 0 degC in K (exact)
_DAY_S = 86400.0          # 24 h in s (exact)

# ---- A1: the slide-10 on-die aggressor (LM26-S slide 10, "On Chip Thermal Aggressor") ------------------------------
A1_T_MAX_C = 113.6        # read-off + derived fit: T = T_max - dT |sin(pi t / P)|, 0.34 degC rms (01 verif. log 5)
A1_DELTA_T_K = 88.5       # read-off + derived fit (01 verification log item 5; CAL-13a)
A1_PERIOD_S = 0.1307      # read-off + derived fit (01 verification log item 5; peaks every 130.5 ms raw)
A1_T_BASE_C = 25.0        # printed lower end of the G26 aggressor runs ("between 25 degC and >105 degC", LM26-P
                          # section 3.2); the fit minimum is 25.1 degC (CAL-13a derived) -- assumed to be the base
# ---- A2: the paper's 808 degC/s run (LM26-P section 3.2, Fig. 3(b)) -----------------------------------------------
A2_RATE_K_PER_S = 808.0   # printed: "a temperature ramp rate of 808 degC/s within the MRM"
A2_T_LOW_C = 25.0         # printed: "ramped up and down between 25oC and >105oC"
A2_T_HIGH_C = 105.0       # printed (">105"; caption "105 degC")
A2_PERIOD_S = 0.100       # printed: "with a period of 100ms"
# ---- soak (LM26-S slide 10 left; 01 section 2.5 and D5, verified) -------------------------------------------------
SOAK_T_LOW_C = 33.0       # read-off (troughs ~33 degC; labels say 25 or 35)
SOAK_T_HIGH_C = 108.0     # read-off (peaks ~107.5-108 degC; labels say 105)
SOAK_CYCLE_S = 20.3 * 60  # read-off: period ~20.3 min
SOAK_N_CYCLES = 90        # read-off: ~90 cycles (+-3), 90 x 20.3 min = 30.45 h (01 verification log item 24)
# ---- XPU power events (textbook REF.xpuPowerStepCPerS [MRVL]; 04 D7, secondary) ------------------------------------
XPU_RATE_K_PER_S_RANGE = (10.0, 120.0)  # printed (secondary: SemiAnalysis report of Marvell ECTC 2026)
XPU_T_PEAK_RATE_S = 0.030               # printed "within ~30 ms" (same source)
# ---- laser (Guide VLSP; YOLE via the textbook REF; CAL-12c, CAL-12d) -----------------------------------------------
LASER_DRIFT_NM_PER_24H = 0.02           # printed limit after burn-in (YOLE via textbook, not re-checked by 01)
LASER_POWER_DRIFT_PCT_PER_24H = 2.5     # printed limit after burn-in (YOLE via textbook)
# ---- laser-sweep emulation (SIB Fig. 1; 01 section 2.5) ------------------------------------------------------------
SWEEP_RATE_NM_PER_S = 50.0              # printed
SWEEP_SPAN_NM = 4.0                     # printed
SWEEP_PERIOD_S = 0.44                   # read-off (trapezoid period of the TX/RX heater traces); ramps 4/50 = 0.08 s
# ---- connector re-mates (textbook Ch17 A6 / Ch18 A9) ---------------------------------------------------------------
REMATE_N = 10                           # printed (ten insertions, LM26-S via textbook Ch17 A6)
REMATE_MEAN_DB = 0.43                   # literature (facet coupling, mean, per facet [ECTC] via textbook Ch18)
REMATE_MEAN_PLUS_3SIGMA_DB = 1.48       # literature (mean + 3 sigma per facet [ECTC] via textbook Ch18)
# ---- data path ---------------------------------------------------------------------------------------------------
SUN_STEP_S = 0.200                      # printed: ones density changed every 200 ms (Sun et al. VLSI 2015 Fig. 5)
SQUELCH_OMA_DBM = -15.0                 # printed normative: modulation <= -15 dBm OMA, average kept (OCI via OCP)
# ---- operating ranges ----------------------------------------------------------------------------------------------
JUNCTION_RANGE_C = (0.0, 105.0)         # printed: closed-loop junction range (LM26-S slide 9; HC slide 16)
BIAS_RANGE_V = (1.0, 3.75)              # printed G26 bias axis (07 P0-7; LM26-S slide 9)


def _k(t_c: float) -> float:
    return float(t_c) + _T0


def _dnu_hz(dlambda_m: float, lambda_m: float) -> float:
    """Frequency change of a wavelength change: dnu = -c dlambda / lambda^2 (first order)."""
    return -_C0 * dlambda_m / (lambda_m * lambda_m)


def _elements(x) -> tuple[int, ...] | None:
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return (int(x),)
    return tuple(int(i) for i in x)


def _state_pwl(signal: str, edges_s: Sequence[float], states: Sequence[float], *, label: str, elements=None,
               stream: str, provenance: str, source: str, note: str) -> g.Component:
    """Absolute piecewise-constant state: states[0] before edges_s[0], states[i+1] from edges_s[i] on (jumps are
    duplicate pwl times, right-continuous)."""
    if len(states) != len(edges_s) + 1:
        raise g.ScenarioError("state machine needs len(states) == len(edges) + 1")
    e = [float(x) for x in edges_s]
    if any(b < a for a, b in zip(e, e[1:])) or (e and e[0] < 0):
        raise g.ScenarioError("state edges must be >= 0 and non-decreasing")
    t = [0.0]
    v = [float(states[0])]
    for i, te in enumerate(e):
        t += [te, te]
        v += [float(states[i]), float(states[i + 1])]
    if len(t) == 1:
        t, v = [0.0, 1.0], [v[0], v[0]]
    return g.pwl(signal, t, v, absolute=True, elements=elements, label=label, stream=stream, provenance=provenance,
                 source=source, note=note)


# =====================================================================================================================
# Base
# =====================================================================================================================

def static(*, ambient_c: float | None = None) -> g.Scenario:
    """No disturbance: constant base temperature (``ambient_c``, default the signal default 25 degC, assumed)."""
    d = {} if ambient_c is None else {"thermal.t_base_k": _k(ambient_c)}
    return g.Scenario(name="static", defaults=d, description="constant ambient, no disturbance")


# =====================================================================================================================
# On-die aggressors (07 P1-4 path 2; D6)
# =====================================================================================================================

def a1_slide10_aggressor(*, site: str = "aggressor", victim_ring: int = 0, referencing: str = "resonance",
                         waveform: str = "rectified_sine", t_max_c: float = A1_T_MAX_C,
                         delta_t_k: float = A1_DELTA_T_K, period_s: float = A1_PERIOD_S,
                         t_base_c: float = A1_T_BASE_C, phase_s: float = 0.0, thermometer_fraction: float = 1.0,
                         dlambda_dt_ref_pm_per_k: float | None = None, set_base: bool = True) -> g.Scenario:
    """[A1] The G26 slide-10 on-die aggressor, the design case (07 P1-4 A1, D6; CAL-13a).

    Printed trajectory (LM26-S slide 10, read-off, corrected in 01 verification log item 5):
    T(t) = 113.6 - 88.5 |sin(pi t / 130.7 ms)| degC, a cusp at T_max at t = 0 and a rounded minimum at 25.1 degC.
    Peak |dT/dt| = pi dT / P = 2127 degC/s, mean 2 dT / P = 1354 degC/s (derived). 07 D6 and P1-4 still describe it
    as parabolic segments with a linear +-2200 degC/s sawtooth; that reading is wrong (a parabola fits 5x worse and
    would need a 2780 degC/s peak), so the rectified sine is the default.

    The disturbance enters as aggressor POWER at ``site`` (a layout site of kind aggressor) through the ROM (07 D6).
    The target is stated at ``victim_ring``: ``referencing='resonance'`` (default) converts the printed degC into a
    resonance trajectory with Lightmatter's ~15 GHz/degC (85.86 pm/K at 1310 nm, printed aside LM26-P section 1);
    ``'temperature'`` keeps degC at the ring (or at a thermometer seeing ``thermometer_fraction`` of it, the
    H86_sensor hypothesis of 07 D2). ``waveform='symmetric_triangle'`` is the 07 4.8 alternative shape with the same
    swing and period (1354 degC/s edges). The power is solved by mrm.scenarios.calibrate.calibrate_scenario; until
    then the scenario cannot compile. ``set_base`` makes ``t_base_c`` (25 degC, the printed lower end; assumed to be
    the aggressor-off temperature) the default base temperature; a stacked scenario that drives the base (soak, cold
    start) replaces it.
    """
    if waveform not in ("rectified_sine", "symmetric_triangle"):
        raise g.ScenarioError(f"A1 waveform must be rectified_sine or symmetric_triangle, got {waveform!r}")
    s_ref = LM_DLAMBDA_DT_M_PER_K if dlambda_dt_ref_pm_per_k is None else dlambda_dt_ref_pm_per_k * 1e-12
    common = dict(units=referencing, period_s=float(period_s), t_base_k=_k(t_base_c), delta_k=float(delta_t_k),
                  s_ref_m_per_k=s_ref, thermometer_fraction=thermometer_fraction, name="A1 slide-10 aggressor",
                  provenance="read-off", source="LM26-S",
                  note="slide 10 temperature panel; rectified-sine fit 0.34 degC rms (01 verification log item 5)")
    if waveform == "rectified_sine":
        tgt = AggressorTarget(form="rectified_sine", t_max_k=_k(t_max_c), phase_s=float(phase_s), **common)
    else:
        # same swing and period; the maximum at t = 0 like the rectified sine (phase P/2 of a rise-first triangle)
        tgt = AggressorTarget(form="trapezoid", t_low_k=_k(t_max_c - delta_t_k), rise_s=period_s / 2, high_s=0.0,
                              fall_s=period_s / 2, phase_s=float(phase_s) + period_s / 2, **common)
    # the base is a signal DEFAULT, not an absolute component, so A1 stacks with a soak, a cold start or any other
    # scenario that drives the base temperature (they replace the default); scenario.ambient_c also overrides it
    defaults = {"thermal.t_base_k": _k(t_base_c)} if set_base else {}
    return g.Scenario(
        name="a1_slide10_aggressor", defaults=defaults,
        aggressors=(g.AggressorSpec(site=site, victim_ring=int(victim_ring), target=tgt,
                                    note="power calibrated to the target (07 P1-4)"),),
        description=f"A1 slide-10 aggressor ({waveform}, {referencing} referencing): {t_max_c} - {delta_t_k} "
                    f"|sin(pi t/{period_s})| degC at ring {victim_ring}",
        provenance=({"item": "A1 trajectory", "value": [t_max_c, delta_t_k, period_s], "provenance": "read-off",
                     "source": "LM26-S", "note": "01 verification log item 5 (rectified sine, corrected)"},
                    {"item": "15 GHz/degC", "value": s_ref, "provenance": "printed", "source": "LM26-P",
                     "note": "aside in section 1; used only to express printed degC as resonance"},
                    {"item": "A1 base temperature", "value": t_base_c, "provenance": "assumed", "source": "LM26-P",
                     "note": "package at the printed 25 degC lower end during the run (signal default)"}))


def a2_808_aggressor(*, site: str = "aggressor", victim_ring: int = 0, referencing: str = "resonance",
                     reading: str = "symmetric_triangle", rate_k_per_s: float = A2_RATE_K_PER_S,
                     t_low_c: float = A2_T_LOW_C, t_high_c: float | None = None, period_s: float = A2_PERIOD_S,
                     t_base_c: float | None = None, thermometer_fraction: float = 1.0,
                     dlambda_dt_ref_pm_per_k: float | None = None, set_base: bool = True) -> g.Scenario:
    """[A2] The paper's 808 degC/s on-chip aggressor (LM26-P section 3.2, printed; CAL-13b; 07 P1-4 A2).

    Printed: "a temperature ramp rate of 808 degC/s within the MRM", "ramped up and down between 25 degC and >105 degC
    with a period of 100 ms", one MRM, local resistive heater. The waveform is not published (Fig. 3(b) is
    under-sampled, EXC-3), so ``reading`` selects a hypothesis (01 D7 with its hedge; 07 P1-4):

    - ``symmetric_triangle`` (default): 808 is dT per period, dT = 808 x 0.1 = 80.8 K (25 -> 105.8 degC), up and
      down within 100 ms, edges 2 x 80.8 / 0.1 = 1616 degC/s ("at least 1600", 07 P1-4);
    - ``rectified_sine``: the same dT and period with the slide-10 shape (mean 1616, peak 2538 degC/s, 01 D7);
    - ``one_sided_sawtooth``: 808 degC/s is the true slope; 25 -> 105 degC (80 K, printed) takes 99.0 ms, leaving
      1.0 ms for the return (07 P1-4 checked). A heater cannot cool that fast, so its calibration reports a large
      residual on the return edge; that is the honest outcome of this reading.
    - ``ramp_per_period``: 808 degC/s is the true slope of BOTH ramps and the printed "period" is one ramp: up over
      100 ms, down over 100 ms (a 200 ms cycle, 80.8 K). This is the reading the 04 D1 verification note favours on
      arithmetic (inferred); unlike the one-sided sawtooth a heater can follow it.

    It may be the same heater as A1 (07 P1-4). The power is calibrated like A1.
    """
    if reading not in ("symmetric_triangle", "rectified_sine", "one_sided_sawtooth", "ramp_per_period"):
        raise g.ScenarioError(f"unknown A2 reading {reading!r}")
    s_ref = LM_DLAMBDA_DT_M_PER_K if dlambda_dt_ref_pm_per_k is None else dlambda_dt_ref_pm_per_k * 1e-12
    tb = t_low_c if t_base_c is None else t_base_c
    if reading == "one_sided_sawtooth":
        hi = A2_T_HIGH_C if t_high_c is None else t_high_c
        dT = hi - t_low_c
        rise = dT / rate_k_per_s
        if rise >= period_s:
            raise g.ScenarioError(f"one-sided sawtooth: a {dT} K ramp at {rate_k_per_s} K/s does not fit in "
                                  f"{period_s} s")
        shape = dict(form="trapezoid", t_low_k=_k(t_low_c), rise_s=rise, high_s=0.0, fall_s=period_s - rise)
    elif reading == "ramp_per_period":
        dT = rate_k_per_s * period_s if t_high_c is None else t_high_c - t_low_c
        ramp = dT / rate_k_per_s                                  # = period_s for the printed numbers
        shape = dict(form="trapezoid", t_low_k=_k(t_low_c), rise_s=ramp, high_s=0.0, fall_s=ramp)
        period_s = 2.0 * ramp                                     # the cycle is two ramps
    else:
        dT = rate_k_per_s * period_s if t_high_c is None else t_high_c - t_low_c
        if reading == "symmetric_triangle":
            shape = dict(form="trapezoid", t_low_k=_k(t_low_c), rise_s=period_s / 2, high_s=0.0,
                         fall_s=period_s / 2)
        else:
            shape = dict(form="rectified_sine", t_max_k=_k(t_low_c + dT))
    tgt = AggressorTarget(units=referencing, period_s=float(period_s), t_base_k=_k(tb), delta_k=float(dT),
                          s_ref_m_per_k=s_ref, thermometer_fraction=thermometer_fraction,
                          name=f"A2 808 degC/s ({reading})", provenance="printed", source="LM26-P",
                          note="section 3.2; waveform not published, reading is a hypothesis (01 D7)", **shape)
    defaults = {"thermal.t_base_k": _k(tb)} if set_base else {}     # a default: stacks with soak / cold start
    return g.Scenario(
        name="a2_808_aggressor", defaults=defaults,
        aggressors=(g.AggressorSpec(site=site, victim_ring=int(victim_ring), target=tgt),),
        description=f"A2 808 degC/s aggressor, reading {reading}, dT {dT:.4g} K, period {period_s} s",
        provenance=({"item": "A2", "value": [rate_k_per_s, t_low_c, period_s], "provenance": "printed",
                     "source": "LM26-P", "note": f"reading {reading} (hypothesis)"},))


def aggressor_direct(*, site: str = "aggressor", waveform: str = "square", power_w: float = 0.1,
                     period_s: float = 0.1, duty: float = 0.5, fall_s: float = 0.0, phase_s: float = 0.0,
                     burst_on_s: float = 0.0, burst_off_s: float = 0.0, t0_s: float = 0.0,
                     provenance: str = "assumed", victim_ring: int = 0) -> g.Scenario:
    """An aggressor driven by a stated POWER waveform, without a target (07 P1-4 ``scenario.aggressor``: site,
    waveform square | sawtooth | burst | triangle | rectified_sine | const, period_s, duty, burst_on_s, burst_off_s,
    power_w). Uncalibrated: the power is a user choice (provenance ``assumed`` or ``override``). ``parabolic`` is
    rejected: the slide-10 shape it stood for is a rectified sine and a temperature target, not a power waveform
    (01 verification log item 5); use a1_slide10_aggressor."""
    kw = dict(site=site, phase_s=phase_s, t0_s=t0_s, label=f"aggressor_direct:{site}", stream="scenario_aggressor",
              provenance=provenance, source="", note=f"direct {waveform} power waveform")
    if waveform == "parabolic":
        raise g.ScenarioError("waveform 'parabolic' was the uncorrected slide-10 reading; the shape is a rectified "
                              "sine of the TEMPERATURE (01 verification log item 5): use a1_slide10_aggressor "
                              "with a target, or waveform='rectified_sine' for a power waveform")
    if waveform == "square":
        c = g.square("thermal.aggressor_w", period_s, duty, amp=power_w, **kw)
    elif waveform == "burst":
        if burst_on_s <= 0:
            raise g.ScenarioError("burst needs burst_on_s > 0")
        c = g.const("thermal.aggressor_w", power_w, **kw)
    elif waveform == "sawtooth":
        c = g.sawtooth("thermal.aggressor_w", period_s, fall_s=fall_s, amp=power_w, **kw)
    elif waveform == "triangle":
        c = g.triangle("thermal.aggressor_w", period_s, amp=power_w, **kw)
    elif waveform == "rectified_sine":
        c = g.rect_sine("thermal.aggressor_w", period_s, amp=power_w, **kw)
    elif waveform == "const":
        c = g.const("thermal.aggressor_w", power_w, **kw)
    else:
        raise g.ScenarioError(f"unknown aggressor waveform {waveform!r}")
    if burst_on_s > 0:
        c = g.with_burst(c, burst_on_s, burst_off_s)
    rec = {"provenance": provenance, "source": "", "note": f"direct power waveform {waveform}"}
    return g.Scenario(name="aggressor_direct",
                      aggressors=(g.AggressorSpec(site=site, victim_ring=int(victim_ring), power=(c,),
                                                  record=rec),),
                      description=f"direct aggressor power {waveform} {power_w} W at {site}")


def aggressor_from_config(aggressor: Mapping[str, Any] | None, target: Mapping[str, Any] | None) -> g.Scenario:
    """Build the aggressor of ``scenario.aggressor`` / ``scenario.target`` (07 P1-4). With a target, the power is
    calibrated later; ``target.trajectory`` (or ``target.name``) selects ``a1`` (default) or ``a2`` and its options
    (e.g. {units: resonance, trajectory: a2, reading: one_sided_sawtooth}). Without a target the aggressor map is a
    direct power waveform (aggressor_direct)."""
    agg = dict(aggressor or {})
    tgt = dict(target or {})
    site = agg.pop("site", "aggressor")
    if not tgt:
        try:
            return aggressor_direct(site=site, **agg)
        except TypeError as e:
            raise g.ScenarioError(f"scenario.aggressor: {e}") from None
    traj = str(tgt.pop("trajectory", tgt.pop("name", "a1"))).lower()
    units = tgt.pop("units", "resonance")
    victim = int(agg.pop("victim_ring", tgt.pop("victim_ring", 0)))
    if agg:
        raise g.ScenarioError(f"scenario.aggressor keys {sorted(agg)} conflict with scenario.target: a target "
                              f"calibrates the power, so give the waveform under scenario.target")
    if traj in ("a1", "slide10", "a1_slide10_aggressor"):
        return a1_slide10_aggressor(site=site, victim_ring=victim, referencing=units, **tgt)
    if traj in ("a2", "808", "a2_808_aggressor"):
        return a2_808_aggressor(site=site, victim_ring=victim, referencing=units, **tgt)
    raise g.ScenarioError(f"unknown target trajectory {traj!r}; known: a1, a2")


# =====================================================================================================================
# Base temperature (07 P1-4 path 1) and XPU events (path 3)
# =====================================================================================================================

def soak(*, t_low_c: float = SOAK_T_LOW_C, t_high_c: float = SOAK_T_HIGH_C, cycle_s: float = SOAK_CYCLE_S,
         n_cycles: int = SOAK_N_CYCLES, start: str = "low", time_compression: float = 1.0,
         compressible: bool = True) -> g.Scenario:
    """[A4] The 30.4 h endurance soak, one experiment (07 D12, P0-10; 01 D5 verified).

    Package/TEC base temperature cycling 33 <-> 108 degC (read-off; the labels say 35 -> 105 or 25 -> 105 -> 25),
    about 90 cycles of 20.3 min, a symmetric triangle: ramp 75 K / 609 s = 0.123 degC/s (01 says ~0.124), 30.45 h.
    It is slow (07 P0-10: about 16,000 times slower than the fast edges), so it is time-compressible: at
    ``time_compression`` c the base temperature runs c times faster in simulated time. The 07 P0-10 load-time
    rule limits c to about 170 (1 % of the 2127 K/s fast-transient rate); a convergence experiment
    (generators.convergence_plan, one cycle at 1, 10, 100) is mandatory before a compressed result is reported.
    After the last cycle the base temperature holds its final value.
    """
    if start not in ("low", "high"):
        raise g.ScenarioError("start must be low or high")
    phase = 0.0 if start == "low" else cycle_s / 2
    total = n_cycles * cycle_s
    c = g.triangle("thermal.t_base_k", cycle_s, amp=t_high_c - t_low_c, offset=_k(t_low_c), absolute=True,
                   slow=compressible, phase_s=phase, t1_s=total, hold_after=True, label="soak",
                   stream="scenario_soak", provenance="read-off", source="LM26-S",
                   note="slide 10 left: 33-108 degC, ~90 cycles of ~20.3 min, ~0.124 degC/s, 30.4 h")
    return g.Scenario(name="soak", components=(c,), time_compression=time_compression, duration_s=total,
                      slow_cycle_s=cycle_s,
                      description=f"soak {t_low_c}-{t_high_c} degC, {n_cycles} x {cycle_s / 60:.1f} min "
                                  f"({(t_high_c - t_low_c) / (cycle_s / 2):.3f} degC/s)",
                      provenance=({"item": "soak", "value": [t_low_c, t_high_c, cycle_s, n_cycles],
                                   "provenance": "read-off", "source": "LM26-S", "note": "01 section 2.5, D5"},))


def xpu_power_step(*, rate_k_per_s: float = XPU_RATE_K_PER_S_RANGE[1], t_peak_rate_s: float = XPU_T_PEAK_RATE_S,
                   t0_s: float = 0.01, delta_k: float | None = None, t_down_s: float | None = None) -> g.Scenario:
    """[B3] An XPU / ASIC power-state change seen at the PIC (07 P1-4 path 3): 10 to 120 degC/s within about
    30 ms (textbook REF.xpuPowerStepCPerS [MRVL], secondary; 04 D7). Ayar Labs prints a similar ~100 degC/s peak
    about 100 ms after a 0 -> 500 W step at an optical engine 3 mm away (04 D5, vendor).

    Shape (assumed): a raised-cosine rise of the die-wide temperature, duration D = 2 t_peak, so the peak rate
    pi dT / (2 D) is reached at t_peak after the step. Without ``delta_k`` the size follows from the rate:
    dT = 2 D rate / pi (4.6 K at 120 degC/s and 30 ms), consistent with Ayar's "brief peak over a few degC" (04 D5).
    The slow package tail that follows (seconds) is not part of this event; add it on ``thermal.t_base_k`` if
    needed. ``t_down_s`` adds the return to the old power state.
    The signal is ``thermal.xpu_rise_k``, a die-wide common-mode rise added to the base temperature by the plant.
    """
    D = 2.0 * t_peak_rate_s
    dT = 2.0 * D * rate_k_per_s / math.pi if delta_k is None else float(delta_k)
    comps = [g.smooth_step("thermal.xpu_rise_k", t0_s, D, amp=dT, label="xpu_up", stream="scenario_xpu",
                           provenance="printed", source="textbook-REF",
                           note=f"[MRVL] {rate_k_per_s} degC/s within ~{t_peak_rate_s * 1e3:.0f} ms (secondary); "
                                f"raised-cosine shape assumed")]
    if t_down_s is not None:
        if t_down_s < t0_s + D:
            raise g.ScenarioError("t_down_s must come after the rise has finished")
        comps.append(g.smooth_step("thermal.xpu_rise_k", t_down_s, D, amp=-dT, label="xpu_down",
                                   stream="scenario_xpu", provenance="assumed", note="return to the old state"))
    return g.Scenario(name="xpu_power_step", components=tuple(comps),
                      description=f"XPU power step: {dT:.3g} K at peak {math.pi * dT / (2 * D):.4g} degC/s")


# =====================================================================================================================
# Laser path (07 P1-4 path 4: optical only)
# =====================================================================================================================

def laser_drift(*, mode: str = "linear", drift_nm_per_24h: float = LASER_DRIFT_NM_PER_24H,
                power_drift_pct_per_24h: float = LASER_POWER_DRIFT_PCT_PER_24H, lines=None, common: bool = False,
                wavelength_sign: float = 1.0, power_sign: float = -1.0, knot_s: float = 60.0,
                lambda_nm: float = LAMBDA_REF_M * 1e9, compressible: bool = True,
                time_compression: float = 1.0) -> g.Scenario:
    """Slow laser wavelength and power drift (Guide VLSP limits after burn-in: +-0.02 nm and +-2.5 % per 24 h,
    printed, YOLE via the textbook; CAL-12c/d; textbook A17).

    ``mode='linear'`` (default): the worst case that stays inside the limit, a straight drift reaching the limit in
    24 h, wavelength red (``wavelength_sign`` +1, i.e. the line frequency falls by c * 0.02 nm / lambda^2 = 3.49 GHz)
    and power down (``power_sign`` -1). ``mode='random_walk'``: an independent Brownian path per line (``common``
    False) with sigma_24h = limit / 3.48, so that P(max over 24 h of |drift| > limit) ~ 1e-3 by the reflection
    principle (4 (1 - Phi(3.48)) = 1e-3; derived). Both are slow signals (time-compressible, 07 P0-10).
    The drift values are scenario parameters (CAL-12 identifies only the limits).

    mrm.plant.laser has its own drift model (laser.freq_drift_hz_per_s, laser.freq_random_walk_hz_per_rt_s, ...;
    off by default). Use one or the other: this scenario's drift enters through laser.dfreq_hz and
    laser.power_delta_db and ADDS to the plant's, so enabling both applies the drift twice. The two random-walk
    calibrations also differ (assumed in both): here P(max over 24 h > limit) ~ 1e-3; in mrm.plant.laser the 24 h
    rms is limit/2.
    """
    if mode not in ("linear", "random_walk"):
        raise g.ScenarioError("mode must be linear or random_walk")
    lam = lambda_nm * 1e-9
    dnu_24h = _dnu_hz(wavelength_sign * drift_nm_per_24h * 1e-9, lam)                         # Hz per 24 h
    ddb_24h = 10.0 * math.log10(1.0 + power_sign * power_drift_pct_per_24h / 100.0)          # dB per 24 h
    els = _elements(lines)
    kw = dict(elements=els, slow=compressible, stream="scenario_laser_drift", provenance="printed",
              source="YOLE")
    if mode == "linear":
        comps = (g.random_walk("laser.dfreq_hz", knot_s=knot_s, sigma_per_rt_s=0.0, drift_per_s=dnu_24h / _DAY_S,
                               label="laser_freq_drift", note="linear drift at the 24 h limit", **kw),
                 g.random_walk("laser.power_delta_db", knot_s=knot_s, sigma_per_rt_s=0.0,
                               drift_per_s=ddb_24h / _DAY_S, label="laser_power_drift",
                               note="linear drift at the 24 h limit", **kw))
    else:
        z = 3.48          # derived: 4 (1 - Phi(z)) = 1e-3
        comps = (g.random_walk("laser.dfreq_hz", knot_s=knot_s, sigma_per_rt_s=abs(dnu_24h) / z / math.sqrt(_DAY_S),
                               common=common, label="laser_freq_drift", note="random walk, 1e-3 to exceed the "
                               "24 h limit", **kw),
                 g.random_walk("laser.power_delta_db", knot_s=knot_s,
                               sigma_per_rt_s=abs(ddb_24h) / z / math.sqrt(_DAY_S), common=common,
                               label="laser_power_drift", note="random walk", **kw))
    return g.Scenario(name="laser_drift", components=comps, time_compression=time_compression,
                      description=f"laser drift ({mode}): {drift_nm_per_24h} nm and "
                                  f"{power_drift_pct_per_24h} % per 24 h")


def laser_sweep_emulation(*, rate_nm_per_s: float = SWEEP_RATE_NM_PER_S, span_nm: float = SWEEP_SPAN_NM,
                          period_s: float = SWEEP_PERIOD_S, lines=None, direction: str = "heating",
                          t0_s: float = 0.0, lambda_nm: float = LAMBDA_REF_M * 1e9) -> g.Scenario:
    """[A3] Lightmatter's laser-sweep emulation of an 800 degC/s aggressor (SIB, Sep 2025; 01 section 2.5 and D6).

    A tunable laser swept at 50 nm/s over 4 nm (printed); the TX and RX heaters follow a trapezoid of period about
    0.44 s (read-off), so the ramps take 4/50 = 0.08 s and the two plateaus (0.44 - 0.16)/2 = 0.14 s each
    (inferred). Optical only: the die is never heated (07 P1-4 path 4, ``laser.dfreq_hz``). ``direction='heating'``
    moves the lines blue first (frequency up), which detunes the ring as heating would. 50 nm/s equals 800 degC/s
    at the bench's 62.5 pm/K, or 582 degC/s at 85.86 pm/K (01 verification log item 24). All lines move together
    (a die-wide emulation) unless ``lines`` is given.
    """
    if direction not in ("heating", "cooling"):
        raise g.ScenarioError("direction must be heating or cooling")
    ramp = span_nm / rate_nm_per_s
    plateau = (period_s - 2 * ramp) / 2
    if plateau < 0:
        raise g.ScenarioError(f"two {ramp} s ramps do not fit in a {period_s} s period")
    amp = abs(_dnu_hz(span_nm * 1e-9, lambda_nm * 1e-9)) * (1.0 if direction == "heating" else -1.0)
    c = g.trapezoid("laser.dfreq_hz", period_s, ramp, plateau, ramp, amp=amp, elements=_elements(lines), t0_s=t0_s,
                    label="laser_sweep", stream="scenario_laser_sweep", provenance="printed", source="SIB",
                    note="50 nm/s over 4 nm (printed); 0.44 s period (read-off); 0.14 s plateaus (inferred)")
    return g.Scenario(name="laser_sweep_emulation", components=(c,),
                      description=f"laser sweep {rate_nm_per_s} nm/s over {span_nm} nm, period {period_s} s")


def laser_failover(*, line: int = 0, t_fail_s: float = 0.01, gap_s: float = 0.0, dfreq_step_ghz: float = 10.0,
                   dpower_db: float = -1.0) -> g.Scenario:
    """[B2] Laser failover: a spare laser auto-tuned to the dead line's wavelength is swapped in, a simultaneous
    step in wavelength and power (YOLE, HOTI26 via the textbook Ch16-Ch18; 07 P1-12).

    The step sizes and the switch-over gap are not published (assumed): ``dfreq_step_ghz`` 10 GHz (inside the
    +-15 GHz Guide accuracy population, CAL-12b), ``dpower_db`` -1 dB, ``gap_s`` 0 (instant; with gap_s > 0 the line
    is dark for that long first).
    """
    t_on = t_fail_s + gap_s
    comps = [g.step("laser.dfreq_hz", t_on, dfreq_step_ghz * 1e9, elements=(int(line),), label="failover_freq",
                    stream="scenario_failover", provenance="assumed", source="YOLE",
                    note="spare laser residual tuning error (assumed size)"),
             g.step("laser.power_delta_db", t_on, dpower_db, elements=(int(line),), label="failover_power",
                    stream="scenario_failover", provenance="assumed", source="YOLE",
                    note="spare laser power difference (assumed size)")]
    if gap_s > 0:
        comps.append(g.pulse("laser.line_enable", t_fail_s, t_on, -1.0, elements=(int(line),),
                             label="failover_dark", stream="scenario_failover", provenance="assumed",
                             note="line dark during the switch-over"))
    return g.Scenario(name="laser_failover", components=tuple(comps),
                      description=f"failover of line {line} at {t_fail_s} s: {dfreq_step_ghz} GHz, {dpower_db} dB")


def line_enable(*, events: Sequence[Sequence[Any]] = ((0.01, 0, False),)) -> g.Scenario:
    """Laser line enable schedule (07 P1-12 ``laser_line_enable``): events (t_s, line(s), on) applied in time order;
    every line starts on. A line that is off has zero power (line_enable is a power factor in [0, 1])."""
    per_line: dict[int, list[tuple[float, float]]] = {}
    for ev in events:
        t, lines, on = float(ev[0]), ev[1], bool(ev[2])
        for ln in _elements(lines) or ():
            per_line.setdefault(ln, []).append((t, 1.0 if on else 0.0))
    comps = []
    for ln, evs in sorted(per_line.items()):
        evs.sort()
        comps.append(_state_pwl("laser.line_enable", [e[0] for e in evs], [1.0] + [e[1] for e in evs],
                                elements=(ln,), label=f"line_enable_{ln}", stream="scenario_line_enable",
                                provenance="assumed", source="", note="line enable schedule (07 P1-12)"))
    return g.Scenario(name="line_enable", components=tuple(comps), description=f"line enable events {list(events)}")


# =====================================================================================================================
# Optical path: re-mates and fibre loss (07 P1-12 scenario A6)
# =====================================================================================================================

def connector_remate(*, n_remates: int = REMATE_N, interval_s: float = 1.0, t0_s: float = 0.5,
                     mean_db: float = REMATE_MEAN_DB, sigma_db: float | None = None, dark_s: float = 0.0) \
        -> g.Scenario:
    """[A6 / Ch18 A9] Ten connector re-mates (printed count, LM26-S via textbook Ch17 A6): each insertion redraws
    the fibre-to-chip coupling loss, a power step that is not a detuning (07 P1-12).

    Loss per insertion ~ N(mean, sigma) per facet, mean 0.43 dB and mean + 3 sigma 1.48 dB (literature, [ECTC] via
    textbook Ch18), so sigma = 0.35 dB (derived); the signal is the change against the nominal loss (the nominal
    chip_input_loss_db is taken to include the mean), clipped so the loss never drops below zero. Re-mate j happens
    at t0_s + j*interval_s (the interval is assumed; the published test re-measured after each insertion). With
    ``dark_s`` > 0 every line is dark for that long at each insertion (assumed). After the last re-mate the level
    holds. Draws come from the counter RNG (stream scenario_remate), so each world b has its own sequence.
    """
    sd = (REMATE_MEAN_PLUS_3SIGMA_DB - REMATE_MEAN_DB) / 3.0 if sigma_db is None else float(sigma_db)
    t1 = t0_s + n_remates * interval_s
    comps = [g.random_levels("optical.input_loss_delta_db", interval_s=interval_s, dist="normal", a=0.0, b=sd,
                             lo=-mean_db, t0_s=t0_s, t1_s=t1, hold_after=True, label="remate_loss",
                             stream="scenario_remate", provenance="literature", source="textbook-REF",
                             note="[ECTC] facet coupling 0.43 dB mean, 1.48 dB mean+3 sigma (textbook Ch18 A9)")]
    if dark_s > 0:
        if dark_s >= interval_s:
            raise g.ScenarioError("dark_s must be shorter than interval_s")
        comps.append(g.square("laser.line_enable", interval_s, dark_s / interval_s, amp=-1.0, t0_s=t0_s, t1_s=t1,
                              label="remate_dark", stream="scenario_remate", provenance="assumed",
                              note="connector open during each insertion"))
    return g.Scenario(name="connector_remate", components=tuple(comps),
                      description=f"{n_remates} re-mates every {interval_s} s from {t0_s} s, sigma {sd:.3g} dB")


def fibre_loss_step(*, t_s: float = 0.01, delta_db: float = REMATE_MEAN_PLUS_3SIGMA_DB - REMATE_MEAN_DB) \
        -> g.Scenario:
    """A deterministic fibre-loss step (07 P1-12 scenario A6), default +1.05 dB = the 3-sigma change of one
    re-mate (derived from [ECTC] via textbook Ch18)."""
    c = g.step("optical.input_loss_delta_db", t_s, delta_db, label="fibre_loss_step", stream="scenario_fibre",
               provenance="literature", source="textbook-REF", note="3-sigma re-mate change (derived)")
    return g.Scenario(name="fibre_loss_step", components=(c,), description=f"fibre loss +{delta_db} dB at {t_s} s")


# =====================================================================================================================
# Data path (07 P0-11, P1-8)
# =====================================================================================================================

def traffic_start_stop(*, t_start_s: float = 0.01, t_stop_s: float | None = None, rings=None,
                       ones_density: float = 0.5) -> g.Scenario:
    """Traffic start (and optional stop) (07 P0-11, P1-8; 04 section 3.3: the events that matter for
    pattern-dependent self-heating). Off: driver disabled, no swing, no transitions. On: driver enabled, nominal
    swing, i.i.d. data with ``ones_density`` p and transition density 2 p (1 - p) (derived; 0.5 for scrambled
    traffic). The P1-8 junction CV^2 and driver heat switch with it."""
    p = float(ones_density)
    td = 2.0 * p * (1.0 - p)
    edges = [t_start_s] + ([t_stop_s] if t_stop_s is not None else [])
    def st(on, off):
        return [off, on] + ([off] if t_stop_s is not None else [])
    els = _elements(rings)
    kw = dict(elements=els, stream="scenario_traffic", provenance="assumed", source="",
              note="traffic on/off state (07 P0-11)")
    comps = (_state_pwl("data.driver_enable", edges, st(1.0, 0.0), label="traffic_driver", **kw),
             _state_pwl("data.swing_scale", edges, st(1.0, 0.0), label="traffic_swing", **kw),
             _state_pwl("data.transition_density", edges, st(td, 0.0), label="traffic_td", **kw),
             _state_pwl("data.ones_density", edges, st(p, 0.5), label="traffic_ones", **kw))
    return g.Scenario(name="traffic_start_stop", components=comps,
                      description=f"traffic on at {t_start_s} s" + (f", off at {t_stop_s} s" if t_stop_s else ""))


def squelch(*, t_on_s: float = 0.01, t_off_s: float | None = None, residual_swing: float = 0.0, rings=None) \
        -> g.Scenario:
    """[B1] Squelch: modulation amplitude cut to <= -15 dBm OMA while the average optical power is kept (OCI MSA via
    OCP, printed normative; 04 D20). Modelled as the swing factor dropping to ``residual_swing`` (0 = the strict
    limit, assumed) between t_on_s and t_off_s; the driver stays enabled and the junction sits at the mean level of
    the driver's level convention, so the average is kept (plant electrical). Data-derived error signals (bit
    statistics, eye monitors) lose their input here (textbook Ch18 A10). The swing change is additive, so squelch
    can be stacked with traffic_start_stop."""
    # additive (residual - 1) on the nominal factor 1, so squelch stacks with traffic_start_stop: while traffic is
    # off (swing 0) the sum is clipped back to 0 by the signal's lower limit
    c = g.pulse("data.swing_scale", t_on_s, math.inf if t_off_s is None else t_off_s, float(residual_swing) - 1.0,
                elements=_elements(rings), label="squelch", stream="scenario_squelch", provenance="printed",
                source="OCP", note="OCI squelch: <= -15 dBm OMA with average power kept")
    return g.Scenario(name="squelch", components=(c,), description=f"squelch from {t_on_s} s")


def idle(*, t0_s: float = 0.01, t1_s: float | None = None, level: float = 0.0, rings=None) -> g.Scenario:
    """Idle: the transmitter holds a constant level (``level`` 0 or 1) between t0_s and t1_s: no transitions, ones
    density = level, driver on (04 section 3.3; 07 P0-11). A constant level is what a lock that assumes balanced data
    misreads most."""
    if level not in (0.0, 1.0, 0, 1):
        raise g.ScenarioError("idle level must be 0 or 1")
    edges = [t0_s] + ([t1_s] if t1_s is not None else [])
    def st(v_idle, v_norm):
        return [v_norm, v_idle] + ([v_norm] if t1_s is not None else [])
    els = _elements(rings)
    kw = dict(elements=els, stream="scenario_idle", provenance="assumed", source="", note="idle constant level")
    comps = (_state_pwl("data.ones_density", edges, st(float(level), 0.5), label="idle_ones", **kw),
             _state_pwl("data.transition_density", edges, st(0.0, 0.5), label="idle_td", **kw))
    return g.Scenario(name="idle", components=comps, description=f"idle at level {level} from {t0_s} s")


def training_pattern(*, bits: str = "10", t0_s: float = 0.01, duration_s: float | None = None,
                     rings=None) -> g.Scenario:
    """A repeating training or framing word between t0_s and t0_s + duration_s (07 P0-11 "training patterns"; 04
    section 3.3: training patterns and unscrambled framing are among the events that move pattern-dependent
    self-heating, unlike steady scrambled traffic). For a word b_0..b_{n-1} repeated without gaps (derived, exact):
    ones density = (number of ones) / n and transition density = #{i : b_i != b_{(i+1) mod n}} / n. Default '10'
    (a clock pattern: 0.5 ones, 1 transition per bit, the largest CV^2 heat at balanced density); e.g. '1100'
    gives (0.5, 0.5) and '1111100000' (0.5, 0.2). Outside the window the data are scrambled traffic (0.5, 0.5;
    assumed, as in ``idle``). The word itself is the user's choice (assumed)."""
    word = str(bits).strip()
    if not word or any(ch not in "01" for ch in word):
        raise g.ScenarioError(f"bits must be a non-empty string of 0 and 1, got {bits!r}")
    n = len(word)
    p = word.count("1") / n
    td = sum(word[i] != word[(i + 1) % n] for i in range(n)) / n
    edges = [t0_s] + ([t0_s + duration_s] if duration_s is not None else [])

    def st(v_pat, v_norm):
        return [v_norm, v_pat] + ([v_norm] if duration_s is not None else [])
    kw = dict(elements=_elements(rings), stream="scenario_training", provenance="assumed", source="",
              note=f"training word {word!r}: ones {p:.4g}, transitions {td:.4g} per bit (derived)")
    comps = (_state_pwl("data.ones_density", edges, st(p, 0.5), label="training_ones", **kw),
             _state_pwl("data.transition_density", edges, st(td, 0.5), label="training_td", **kw))
    return g.Scenario(name="training_pattern", components=comps,
                      description=f"training word {word!r} (ones {p:.3g}, transitions {td:.3g}) from {t0_s} s")


def ones_density_steps(*, levels: Sequence[float] = (0.0, 0.5, 1.0), step_s: float = SUN_STEP_S,
                       mode: str = "cycle", t0_s: float = 0.0, rings=None, common: bool = True) -> g.Scenario:
    """Ones-density steps 0 -> 0.5 -> 1 every 200 ms (Sun et al., VLSI 2015 Fig. 5: "random ones-density changed
    every 200 ms, including all-0 and all-1", printed; 04 D21; 07 P0-11). ``mode='cycle'`` repeats ``levels`` in
    order; ``mode='random'`` draws a level from ``levels`` each step (counter RNG; ``common`` = same data statistics
    on every ring). The transition density follows as 2 p (1 - p) (i.i.d. bits, derived): 0 for all-0 and all-1."""
    lv = [float(x) for x in levels]
    if not lv or any(not 0.0 <= x <= 1.0 for x in lv):
        raise g.ScenarioError("levels must be in [0, 1]")
    els = _elements(rings)
    kw = dict(elements=els, absolute=True, t0_s=t0_s, hold_before=True, stream="scenario_ones_density",
              provenance="printed", source="SUN15", note="ones density steps every 200 ms (Sun et al. 2015)")
    if mode == "cycle":
        n = len(lv)
        t = []
        p = []
        for i, x in enumerate(lv):
            t += [i * step_s, (i + 1) * step_s]
            p += [x, x]
        td = [2 * x * (1 - x) for x in p]
        comps = (g.pwl("data.ones_density", t, p, periodic=True, period_s=n * step_s, label="ones_cycle", **kw),
                 g.pwl("data.transition_density", t, td, periodic=True, period_s=n * step_s, label="td_cycle", **kw))
    elif mode == "random":
        comps = (g.random_levels("data.ones_density", interval_s=step_s, dist="choice", choices=lv, common=common,
                                 link="ones_density", label="ones_random", **kw),
                 g.random_levels("data.transition_density", interval_s=step_s, dist="choice", choices=lv,
                                 common=common, link="ones_density", map="bernoulli_td", label="td_random", **kw))
    else:
        raise g.ScenarioError("mode must be cycle or random")
    return g.Scenario(name="ones_density_steps", components=comps,
                      description=f"ones density {mode} over {lv} every {step_s} s")


# =====================================================================================================================
# Bias (07 P0-7: offsets always allowed; absolute only when the scenario owns tx.bias)
# =====================================================================================================================

def bias_ripple(*, amplitude_v: float = 0.005, freq_hz: float = 100e3, rings=None, phase_s: float = 0.0,
                t0_s: float = 0.0) -> g.Scenario:
    """Supply ripple on the modulator bias: a sine on ``tx.bias_offset_v`` (07 P0-7, additive). No public number:
    5 mV at 100 kHz is assumed (a switching-regulator residue; at about 50 pm/V it moves the resonance by about
    0.25 pm). 100 kHz keeps 10 samples per period at the 1 us tick."""
    c = g.sine("tx.bias_offset_v", 1.0 / freq_hz, amp=amplitude_v, elements=_elements(rings), phase_s=phase_s,
               t0_s=t0_s, label="bias_ripple", stream="scenario_bias", provenance="assumed", source="",
               note="supply ripple (assumed size and frequency)")
    return g.Scenario(name="bias_ripple", components=(c,), description=f"bias ripple {amplitude_v} V at {freq_hz} Hz")


def bias_step(*, t_s: float = 0.01, delta_v: float = -0.05, rings=None) -> g.Scenario:
    """A bias droop or step on ``tx.bias_offset_v`` (07 P0-7, additive; size assumed)."""
    c = g.step("tx.bias_offset_v", t_s, delta_v, elements=_elements(rings), label="bias_step",
               stream="scenario_bias", provenance="assumed", note="bias droop (assumed)")
    return g.Scenario(name="bias_step", components=(c,), description=f"bias step {delta_v} V at {t_s} s")


def reloma_plane(*, biases_v: Sequence[float] = (1.0, 1.5, 2.0, 2.25, 2.5, 3.0, 3.5, 3.75),
                 t_low_c: float = JUNCTION_RANGE_C[0], t_high_c: float = JUNCTION_RANGE_C[1],
                 sweep_s: float = 10.0) -> g.Scenario:
    """[A7] The relOMA plane: base temperature swept 0 -> 105 degC (printed closed-loop range, LM26-S slide 9) at each
    bias of the printed 1.0-3.75 V axis, with the loop closed. The scenario OWNS tx.bias here
    (controller_interface.bind.tx.bias: scenario, 07 P0-7). Each bias holds for one monotonic sweep of ``sweep_s``
    (assumed), up for even and down for odd bias indices, so the temperature is continuous (a 105 K jump between
    biases would unlock the loop). Quasi-static maps are better done by experiment.kind quasi_static."""
    nb = len(biases_v)
    # a triangle (up for bias 0, down for bias 1, ...), so the base temperature never jumps by 105 K between biases
    tb = g.triangle("thermal.t_base_k", 2.0 * sweep_s, amp=t_high_c - t_low_c, offset=_k(t_low_c),
                    absolute=True, t1_s=nb * sweep_s, hold_after=True, label="reloma_temperature",
                    stream="scenario_reloma", provenance="printed", source="LM26-S",
                    note="0-105 degC closed-loop junction range")
    edges = [i * sweep_s for i in range(1, nb)]
    bias = _state_pwl("tx.bias_v", edges, list(biases_v), label="reloma_bias", stream="scenario_reloma",
                      provenance="printed", source="LM26-S", note="G26 bias axis 1.0-3.75 V")
    return g.Scenario(name="reloma_plane", components=(tb, bias), duration_s=nb * sweep_s,
                      description=f"relOMA plane over {list(biases_v)} V x {t_low_c}-{t_high_c} degC")


# =====================================================================================================================
# Start-up (textbook B5; 07 P1-1)
# =====================================================================================================================

def cold_start(*, ambient_c_range: Sequence[float] = JUNCTION_RANGE_C, laser_turn_on_s: float = 0.0,
               lines=None) -> g.Scenario:
    """[B5] Cold start in an unknown thermal environment: heaters at zero, no stored calibration (factory_cal
    withheld, 07 P1-1), ambient anywhere in 0-105 degC (printed closed-loop range, LM26-S slide 9; textbook B5),
    drawn uniformly per world from the counter RNG. The laser static error (+-20 GHz) and the per-ring offsets come
    from the config (realize). ``laser_turn_on_s`` > 0 ramps the lines on from zero (assumed)."""
    lo, hi = float(ambient_c_range[0]), float(ambient_c_range[1])
    comps = [g.random_levels("thermal.t_base_k", interval_s=math.inf, dist="uniform", a=_k(lo), b=_k(hi),
                             absolute=True, label="cold_ambient", stream="scenario_cold_start",
                             provenance="printed", source="LM26-S", note="ambient uniform over 0-105 degC")]
    if laser_turn_on_s > 0:
        comps.append(g.pwl("laser.line_enable", [0.0, laser_turn_on_s], [0.0, 1.0], absolute=True,
                           elements=_elements(lines), label="laser_turn_on", stream="scenario_cold_start",
                           provenance="assumed", note="laser turn-on ramp"))
    return g.Scenario(name="cold_start", components=tuple(comps),
                      reset=g.ResetSpec(start="cold", heater_init="zero", factory_cal=False,
                                        note="textbook B5: heaters at zero, no stored calibration"),
                      description=f"cold start, ambient uniform {lo}-{hi} degC")


def warm_start(*, ambient_c: float = 25.0) -> g.Scenario:
    """Warm start: the factory calibration table is passed at reset (07 P1-1) and the heaters start from it; ambient
    ``ambient_c`` (25 degC assumed: the factory-calibration temperature is not published)."""
    # a default (not an absolute component): a stacked soak or ambient profile replaces it
    return g.Scenario(name="warm_start", defaults={"thermal.t_base_k": _k(ambient_c)},
                      reset=g.ResetSpec(start="warm", heater_init="factory", factory_cal=True),
                      description=f"warm start at {ambient_c} degC",
                      provenance=({"item": "warm-start ambient", "value": ambient_c, "provenance": "assumed",
                                   "source": "", "note": "factory-calibration temperature not published"},))
