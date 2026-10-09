"""Drop-port set-point PI lock, normalized as a fraction of the measured drop peak (ARCHITECTURE section 6 item 1;
04 C1 + C3; 07 P0-1). Python reference and compiled (numba, flat ABI, 07 P0-9) versions with identical arithmetic.

This module also holds the helpers shared by the other controllers of the same owner (dither_lockin, bias_dither,
extremum_seeking, feedforward, midranging): the knowledge-gated nominal model (:class:`NominalModel`, 07 P1-1, D13),
the monitor block mean and the driver-law gain schedule.

Which side, and why (07 P0-1)
-----------------------------
Detuning delta = lambda_res - lambda_laser (pm). Heating moves the resonance red (delta up). The drop monitor y and
the absorbed power P_abs share the (NRZ-broadened) Lorentzian shape of half width w, and self-heating adds
G P_abs(delta) to the heater-only detuning u: delta = u + G P_abs(delta). The branch with the resonance RED of the
laser (delta > 0, the rising edge of the drop peak in laser wavelength: the GF/G26 lock side, 07 P0-1, 01) is
open-loop stable, d delta/du = 1/(1 + G |P_abs'|). The blue side is unstable where G P_abs' > 1; at g26 nominal
(bistability ratio ~6) the fold sits only ~w/(3 r) ~ 9 pm blue of the peak. So the lock point is on the red side at
y_set = f M_pk (delta_set = w sqrt(1/f - 1) for a pure Lorentzian), where dy/dcode < 0 and the heater-to-detuning
gain is divided by the self-heating relief (1 + G |P_abs'|), 3-7x in g26, which divides every other disturbance too.

Algorithm, per ring (04 C1 MAP/RETURN; KRA24 "hot-to-cold sweep, reset hot, sweep to the rising edge, then PI"; C3)
-----------------------------------------------------------------------------------------------------------------
Hand-off contract: at reset the ring sits on the hot (stable, red) branch of its assigned line, after an acquisition
(PAT-3 ramp-to-maximum, the WDM supervisor) or a warm start. The heater code is ``init_codes`` or, by default, the
latched register ``tx.heater.applied``.

1. MAP (``start='map'``): lower the heater and record (code, y). The resonance walks blue along the stable branch
   and y rises to the peak. Steps are ``map_step_codes`` on the flank, where the self-heating relief keeps dy/dcode
   nearly constant, and ``fine_step_codes`` once the slope over the last ``slope_window`` steps falls below
   ``near_ratio`` of the flank slope (the top). The MAP stops as soon as y falls ``peak_drop_frac`` below its maximum
   (just past the peak, still on the stable branch: in g26 the fold is only ~9 pm further, and falling off it moves
   the resonance ~1.5 nm blue, where rings 4-7 meet line n-4 with their next order). M_pk = the largest block mean;
   the slope s = dy/dcode at y = f M_pk is a least-squares fit over ``fit_half_width`` points on each side of the
   f-crossing: the real lock-point gain, self-heating relief, TCR and heater law included (no model needed). Then
   LOCK (the PI raises the heater to the set-point). An up-ramp alone cannot measure the peak: under bistability it
   captures at ~0.5 M_pk (07 P0-1). If the fold is crossed anyway (y drops by ``fall_frac``), RETURN. If the
   monitor clips at the ADC full scale (``adc_overrange``; g26 worlds with monitor gain_scale ~2) the peak cannot
   be measured: M_pk = the full scale and the set-point lies further red than f of the true peak. When the monitor
   is clipped from the hand-off on (no unclipped MAP record; the g26_hot_selfheat corner, +8 dBm at the ring, puts
   the drop peak at ~2.5x the full scale), DECLIP ramps red by ``ramp_step_codes``, recording, until the monitor
   reads below f x full scale, fits the slope over the last 2 ``fit_half_width`` + 1 unclipped points and locks
   there (it used to FAULT on every ring of that corner).
2. RETURN: raise the heater by ``ramp_step_codes`` per step. Once the code is above the guard (u_peak +
   ``guard_frac`` (u_start - u_peak), so a capture by another line or the next order of the FSR is ignored: in g26
   rings 4-7 have their next order 3.4 FWHM from line n-4, BND-1), track the largest y; when it exceeded
   ``recapture_level`` M_pk and y has fallen back by ``recapture_drop`` of it, the ring is red of the peak: after the
   capture jump of a bistable ring, or after a smooth passage over the peak without bistability. Always the same
   approach direction (hysteresis, 04 C1).
3. LOCK: discrete PI in velocity form on the heater-code error e = (y - f M_pk) / |s| * sigma(u):

       du[k] = k_c ((e[k] - e[k-1]) + (T / T_i) e[k]) + v[k],   v[k] = v[k-1] + k_2 e[k]   (type 2 when k_2 > 0)
       u[k]  = clip(u[k-1] + du[k], code_min, code_max)

   Anti-windup (04 C3): the PI part is a velocity form (it cannot wind up at a clamp); the type-2 velocity v is an
   integrator of its own, so when the clamp is active and v pushes into it, v is reset to 0 (clamping / conditional
   integration). Without that, v kept integrating while the heater sat at a rail and held it there after the
   disturbance had reversed (measured on a toy ring: ~0.9 ms longer at full scale with k_2 = 0.002).

   T = controller period; T_i = PI zero, placed on the dominant thermal pole (default 13 us, the slow interim pole of
   07 P1-2 / R04; nominal ``ring.thermal.tau_slow_s`` with the ``thermal_dynamics`` token); k_c = 2 pi f_c T_i gives
   the open loop k_c/(T_i s) e^{-s theta} with crossover f_c = ``crossover_hz``. sigma(u) = dP/dcode(u_ref) /
   dP/dcode(u) is the driver-law gain schedule (07 P1-11; electrical handoff note 3: dP/dcode varies 1.9x over the G26
   range in voltage drive): u_ref/u for a quadratic law (voltage or current drive), 1 for power drive, clipped to
   [sigma_min, sigma_max]; u_ref is the code where s was measured.
4. Fall-off (pushed past the peak to the fold): y drops by more than ``fall_frac`` M_pk within ``jump_steps`` steps
   to below ``fall_level`` M_pk (the cold branch; a PI transient from the peak to the set-point stays above it). The PI
   would run away (one-sided actuator, 04 C3), so RECAPTURE ramps up by ``recapture_step_codes`` per step as in RETURN
   (guard: the code at the fall + the MAP guard, or ``guard_codes`` without a MAP), then the PI resumes (bumpless).

A step without a new monitor sample, with ``monitor_valid`` false or a ring inactive under round-robin holds
everything. ``start='lock'`` takes the factory table (``peak_codes``, ``slope_per_code``) instead of the map (warm
start, 07 P1-1); ``start='capture'`` first ramps up by ``ramp_step_codes`` until a resonance has been seen (the
largest monitor level since the start exceeds capture_ratio x the smallest + capture_min_codes) and the monitor has
fallen back by ``recapture_drop`` of that largest level (the ring is then red of the peak: after the capture jump of a
bistable ring, typically at ~0.5 M_pk, or after a smooth passage over the peak without bistability, just past it),
then maps. After a smooth passage the MAP record starts above the set-point, so the slope comes from the Lorentzian
scaling in :func:`_fit_slope` (approximate). It gives up at full scale (FAULT, back to the start code). An earlier
version waited for a jump within ``jump_steps`` steps, which a ring without bistability never makes: it ramped to full
scale and parked there.

Known failure modes (04 C3, measured in tests/verification/test_ctrl_a_pi.py): a laser power step moves the lock
point (the peak is not re-measured; at f = 0.7 a -1 dB step moves delta from 0.65 w to 0.36 w); a fold crossing costs
a recapture; dark current and neighbour-line light shift y_set by their share of M_pk; a clipped monitor (above the
ADC full scale) under-reads M_pk.

Units: codes (heater DAC, monitor ADC) and seconds; pm only in telemetry.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numba as nb
import numpy as np

from mrm.control import api
from mrm.control.api import BaseController, CompiledController, ControllerSpec, Observation
from mrm.control.registry import register
from mrm.core.channels import ActId, ObsId

__all__ = [
    "KNOWLEDGE_KEYS", "NominalModel", "monitor_block_mean", "block_inputs", "law_code", "LAW_NONE", "LAW_QUADRATIC",
    "LAW_LINEAR", "sigma_schedule", "per_ring", "design_pi_gains", "make_params", "PiLock", "PiLockCompiled", "pi_init", "pi_step",
    "ST_HOLD", "ST_MAP", "ST_RETURN", "ST_LOCK", "ST_RECAPTURE", "ST_FAULT", "ST_CAPTURE", "ST_DECLIP", "STATE_NAMES",
]

# ---- knowledge-gated nominal model (07 P1-1, D13) -----------------------------------------------------------------

#: knowledge token -> config keys a controller may read from ``spec.nominal`` when the token is listed in
#: ``controller_interface.knowledge``. 07 P1-1 names line_grid, ring_to_line, heater_r_nominal and
#: dlambda_dt_nominal; the others are proposed here (docs/handoff/controllers_a.md).
KNOWLEDGE_KEYS: dict[str, tuple[str, ...]] = {
    "heater_law": ("ring.heater.driver_mode", "ring.heater.full_scale_v", "ring.heater.full_scale_a",
                   "ring.heater.full_scale_w", "ring.heater.r_cold_ohm", "ring.heater.dac_bits"),
    "heater_r_nominal": ("ring.heater.r_cold_ohm",),
    "heater_efficiency": ("ring.heater.p_fsr_mw", "ring.optics.fsr_thz"),
    "dlambda_dt_nominal": ("ring.thermal.dlambda_dt_pm_per_k",),
    "linewidth": ("ring.optics.q_loaded",),
    "thermal_dynamics": ("ring.thermal.tau_fast_s", "ring.thermal.tau_slow_s", "ring.thermal.share_fast_frac"),
    "bias_efficiency": ("ring.junction.eff_at_0v_pm_per_v", "ring.junction.v_bi_v",
                        "ring.junction.depletion_exponent", "ring.junction.fill_fraction"),
}

_C0 = 299792458.0
_LAMBDA_DEFAULT_M = 1310e-9      # O-band grid centre (printed G26), used only without the line_grid token


def _nominal_value(nominal, key: str, setting: bool = False):
    """One value of ``spec.nominal`` (a Config with ``si``/``get``, or a mapping of dotted keys in config units).
    Numbers come back in SI, settings as given; None when absent."""
    if nominal is None:
        return None
    if not setting and callable(getattr(nominal, "si", None)):
        try:
            v = nominal.si(key, None)
        except Exception:                      # a setting, or a key this config does not carry
            v = None
        if v is not None:
            return float(v)
    get = getattr(nominal, "get", None)
    if not callable(get):
        return None
    try:
        v = get(key, None)
    except Exception:
        return None
    if isinstance(v, Mapping) and "value" in v:
        v = v["value"]
    if v is None or setting or isinstance(v, str):
        return v
    try:
        from mrm.params.units import to_si
        return float(to_si(key, v))
    except Exception:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class NominalModel:
    """What a controller may know of the plant model: read from ``spec.nominal`` ONLY for the tokens listed in
    ``spec.knowledge`` (07 P1-1; the spec is built from the unsampled design, never from Truth). A field is None when
    its token is withheld (g26 lists only line_grid and ring_to_line, so every field below is None there).

    law ('voltage' | 'current' | 'power' | 'dac_pwm'), v_fs_v, i_fs_a, p_fs_w, r_cold_ohm, dac_bits: the nominal
    driver law, P = (V_fs c/(2^b - 1))^2/R, (I_fs c/(2^b - 1))^2 R or P_fs c/(2^b - 1) (cold R, no TCR);
    pm_per_mw = FSR/P_FSR (07 D16) at the grid centre; dlambda_dt_pm_per_k; fwhm_pm = lambda/Q_L; tau_slow_s,
    tau_fast_s, share_fast (interim thermal poles, 07 P1-2); bias_eff_pm_per_v (= fill x eff_at_0v), bias_v_bi_v,
    bias_exponent (07 P1-7); lambda_m (line_grid mean, else 1310 nm).
    """
    tokens: tuple = ()
    law: str | None = None
    v_fs_v: float | None = None
    i_fs_a: float | None = None
    p_fs_w: float | None = None
    r_cold_ohm: float | None = None
    dac_bits: int | None = None
    pm_per_mw: float | None = None
    dlambda_dt_pm_per_k: float | None = None
    fwhm_pm: float | None = None
    tau_slow_s: float | None = None
    tau_fast_s: float | None = None
    share_fast: float | None = None
    bias_eff_pm_per_v: float | None = None
    bias_v_bi_v: float | None = None
    bias_exponent: float | None = None
    lambda_m: float = _LAMBDA_DEFAULT_M

    @classmethod
    def from_spec(cls, spec: ControllerSpec) -> "NominalModel":
        know = set(spec.knowledge or ())
        nom = spec.nominal
        lam = _C0 / float(np.mean(spec.line_grid_hz)) if spec.line_grid_hz else _LAMBDA_DEFAULT_M
        kw: dict = {"tokens": tuple(sorted(know & set(KNOWLEDGE_KEYS))), "lambda_m": lam}

        def num(key):
            return _nominal_value(nom, key)
        if "heater_law" in know:
            law = _nominal_value(nom, "ring.heater.driver_mode", setting=True)
            kw.update(law=None if law is None else str(law), v_fs_v=num("ring.heater.full_scale_v"),
                      i_fs_a=num("ring.heater.full_scale_a"), p_fs_w=num("ring.heater.full_scale_w"),
                      r_cold_ohm=num("ring.heater.r_cold_ohm"))
            bits = num("ring.heater.dac_bits")
            kw["dac_bits"] = spec.heater_dac.bits if bits is None else int(bits)
        if "heater_r_nominal" in know and kw.get("r_cold_ohm") is None:
            kw["r_cold_ohm"] = num("ring.heater.r_cold_ohm")
        if "heater_efficiency" in know:
            p_fsr_w = num("ring.heater.p_fsr_mw")
            fsr_hz = num("ring.optics.fsr_thz")
            if p_fsr_w and fsr_hz:
                kw["pm_per_mw"] = (lam * lam * fsr_hz / _C0 * 1e12) / (p_fsr_w * 1e3)
        if "dlambda_dt_nominal" in know:
            s = num("ring.thermal.dlambda_dt_pm_per_k")
            kw["dlambda_dt_pm_per_k"] = None if s is None else s * 1e12
        if "linewidth" in know:
            q = num("ring.optics.q_loaded")
            kw["fwhm_pm"] = None if not q else lam / q * 1e12
        if "thermal_dynamics" in know:
            kw.update(tau_slow_s=num("ring.thermal.tau_slow_s"), tau_fast_s=num("ring.thermal.tau_fast_s"),
                      share_fast=num("ring.thermal.share_fast_frac"))
        if "bias_efficiency" in know:
            e = num("ring.junction.eff_at_0v_pm_per_v")
            f = num("ring.junction.fill_fraction")
            kw.update(bias_eff_pm_per_v=None if e is None else e * 1e12 * (1.0 if f is None else f),
                      bias_v_bi_v=num("ring.junction.v_bi_v"), bias_exponent=num("ring.junction.depletion_exponent"))
        return cls(**kw)

    def dp_dcode_w(self, code) -> np.ndarray | None:
        """Nominal dP/dcode (W per code) at ``code``; None without the ``heater_law`` token."""
        if self.law is None or self.dac_bits is None:
            return None
        full = float((1 << int(self.dac_bits)) - 1)
        c = np.asarray(code, dtype=float)
        if self.law == "voltage" and self.v_fs_v and self.r_cold_ohm:
            return 2.0 * self.v_fs_v ** 2 * c / (full * full * self.r_cold_ohm)
        if self.law == "current" and self.i_fs_a and self.r_cold_ohm:
            return 2.0 * self.i_fs_a ** 2 * c * self.r_cold_ohm / (full * full)
        if self.law in ("power", "dac_pwm") and self.p_fs_w:
            return np.full_like(c, self.p_fs_w / full)
        return None

    def pm_per_code(self, code) -> np.ndarray | None:
        """Open-loop heater tuning (pm per code; no self-heating relief, no TCR); None unless both ``heater_law`` and
        ``heater_efficiency`` are known."""
        d = self.dp_dcode_w(code)
        if d is None or self.pm_per_mw is None:
            return None
        return d * 1e3 * self.pm_per_mw


LAW_NONE, LAW_QUADRATIC, LAW_LINEAR = 0, 1, 2


def law_code(schedule: str, nominal: NominalModel) -> int:
    """Gain-schedule law: 'auto' takes the nominal driver law when the token is given (voltage/current -> quadratic,
    power/dac_pwm -> linear) and otherwise assumes a quadratic law (voltage or current drive: the designer's
    assumption, 07 P1-11 lists both as hypotheses); or 'quadratic' | 'linear' | 'none'."""
    if schedule == "auto":
        return LAW_LINEAR if nominal.law in ("power", "dac_pwm") else LAW_QUADRATIC
    try:
        return {"quadratic": LAW_QUADRATIC, "linear": LAW_LINEAR, "none": LAW_NONE}[schedule]
    except KeyError:
        raise ValueError(f"schedule must be auto | quadratic | linear | none, got {schedule!r}") from None


@nb.njit(cache=True)
def sigma_schedule(law, code, code_ref, sig_min, sig_max):
    """dP/dcode(code_ref)/dP/dcode(code): code_ref/code for the quadratic law (LAW_QUADRATIC), else 1; clipped."""
    if law != 1:
        return 1.0
    c = code if code > 1.0 else 1.0
    s = code_ref / c
    if s < sig_min:
        s = sig_min
    if s > sig_max:
        s = sig_max
    return s


def monitor_block_mean(obs: Observation, y_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean monitor code of the samples released since the last step, (B, N) float64, and the has-new-data mask.
    Rings without a sample keep ``y_prev``. Integer codes sum exactly in float64, so this equals a sequential loop
    bit for bit (the compiled controllers' arithmetic)."""
    samples = np.asarray(obs.sensors["tx.monitor.samples"], dtype=np.float64)
    count = np.asarray(obs.sensors["tx.monitor.count"]).astype(np.int64)
    mask = np.arange(samples.shape[-1])[None, None, :] < count[..., None]
    s = np.where(mask, samples, 0.0).sum(axis=-1)
    has = count > 0
    return np.where(has, s / np.maximum(count, 1), y_prev), has


def block_inputs(obs: Observation, y_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(y, ok): block mean (held without new data) and the update mask = new data & monitor_valid & active ring."""
    y, has = monitor_block_mean(obs, y_prev)
    ok = has.copy()
    if "monitor_valid" in obs.sensors:
        ok &= np.asarray(obs.sensors["monitor_valid"], dtype=bool)
    if obs.active is not None:
        ok &= np.asarray(obs.active, dtype=bool)[None, :]
    return y, ok


def per_ring(value, shape, name: str, dtype=float):
    """Broadcast a scalar, a per-ring list (N,) or a (B, N) array to ``shape`` = (B, N); None stays None."""
    if value is None:
        return None
    a = np.asarray(value, dtype=dtype)
    if a.ndim == 1 and a.size == shape[1]:
        a = a[None, :]
    try:
        return np.broadcast_to(a, shape).astype(dtype).copy()
    except ValueError:
        raise ValueError(f"{name}: expected a scalar, N = {shape[1]} values or a (B, N) = {shape} array, got shape "
                         f"{a.shape}") from None


def make_params(pcls, params, overrides, name: str):
    """A ``pcls`` dataclass from a mapping (or an instance) plus keyword overrides; unknown keys raise TypeError
    (BaseController's rule, for the compiled controllers that do not derive from it)."""
    import dataclasses
    if params is not None and isinstance(params, pcls):
        return dataclasses.replace(params, **overrides) if overrides else params
    values = dict(params or {})
    values.update(overrides)
    names = {f.name for f in dataclasses.fields(pcls)}
    unknown = set(values) - names
    if unknown:
        raise TypeError(f"{name}: unknown controller.params {sorted(unknown)}; allowed: {sorted(names)}")
    return pcls(**values)


def design_pi_gains(spec: ControllerSpec, nominal: NominalModel, crossover_hz: float,
                    ti_s: float | None) -> tuple[float, float]:
    """(k_c, T/T_i) of the velocity-form PI: zero on the dominant thermal pole T_i (``ti_s``, else the nominal
    tau_slow with the thermal_dynamics token, else 13 us), crossover f_c: k_c = 2 pi f_c T_i."""
    ti = float(ti_s) if ti_s is not None else (float(nominal.tau_slow_s) if nominal.tau_slow_s else 13e-6)
    if ti <= 0 or crossover_hz <= 0:
        raise ValueError("ti_s and crossover_hz must be > 0")
    return 2.0 * math.pi * float(crossover_hz) * ti, spec.period_s / ti


ST_HOLD, ST_MAP, ST_RETURN, ST_LOCK, ST_RECAPTURE, ST_FAULT, ST_CAPTURE, ST_DECLIP = 0, 1, 2, 3, 4, 5, 6, 7
STATE_NAMES = {ST_HOLD: "hold", ST_MAP: "map", ST_RETURN: "return", ST_LOCK: "lock", ST_RECAPTURE: "recapture",
               ST_FAULT: "fault", ST_CAPTURE: "capture", ST_DECLIP: "declip"}
_START = {"map": ST_MAP, "lock": ST_LOCK, "capture": ST_CAPTURE}


# ---- the per-ring algorithm in plain Python (the reference; pi_step is its compiled port) --------------------------
# Per-ring float state (same order as the compiled layout): u, y, e_prev, v, peak, slope, u_ref, y_max, u_max, y_lo
# (CAPTURE: smallest monitor level), u_start, guard, u_guard, rise_max (flank slope reference), step_prev (its running
# sum); int state: st, hist_n, lock_cnt, map_n, n_fall, fresh, fine_seen (near the top).

class _PiRing:
    """State of one ring. Plain Python floats; the compiled port (:func:`pi_step`) repeats every operation in the same
    order on the same values, so both return the same codes (tests/verification/test_ctrl_a_pi.py)."""

    __slots__ = ("st", "u", "y", "e_prev", "v", "peak", "slope", "u_ref", "y_max", "u_max", "y_lo", "u_start",
                 "guard", "u_guard", "rise_max", "step_prev", "hist", "hist_n", "lock_cnt", "map_n", "n_fall", "fresh",
                 "fine_seen", "buf_u", "buf_y")

    def __init__(self, J: int, H: int):
        self.st = ST_HOLD
        self.u = self.y = self.e_prev = self.v = self.peak = self.u_ref = 0.0
        self.y_max = self.u_max = self.y_lo = self.u_start = self.guard = self.u_guard = self.rise_max = 0.0
        self.step_prev = 0.0
        self.slope = math.nan
        self.hist = [0.0] * J
        self.hist_n = self.lock_cnt = self.map_n = self.n_fall = self.fine_seen = 0
        self.fresh = 1
        self.buf_u = np.zeros(H)
        self.buf_y = np.zeros(H)


def _pi_error(r: _PiRing, f: float, law: int, smin: float, smax: float) -> float:
    s = abs(r.slope)
    if not (s > 0.0) or math.isinf(s):
        s = 1.0
    return (r.y - f * r.peak) / s * sigma_schedule(law, r.u, r.u_ref, smin, smax)


def _fit_slope(buf_u, buf_y, j_end: int, f: float, peak: float, K: int) -> tuple[float, float]:
    """(dy/du at the set-point, u there) from the MAP record [0, j_end): least squares over the points around the last
    sample at or below f*peak before the maximum. When the record starts above the set-point (hand-off closer to the
    peak than f), the slope fitted at the start is scaled to the set-point with the Lorentzian law |dy/dx| =
    2 sqrt(M/y - 1) y^2 / M (relief ignored: documented approximation). Falls back to the chord from the first point
    to the maximum."""
    if j_end < 2:
        return math.nan, buf_u[0]
    j_max = 0
    for j in range(1, j_end):
        if buf_y[j] > buf_y[j_max]:
            j_max = j
    target = f * peak
    j_c = -1
    for j in range(j_max, -1, -1):
        if buf_y[j] <= target:
            j_c = j
            break
    below = j_c >= 0
    if not below:
        j_c = 0
    lo = j_c - K if j_c - K > 0 else 0
    hi = j_c + K + 1 if j_c + K + 1 < j_max + 1 else j_max + 1
    n = hi - lo
    slope = math.nan
    if n >= 2:
        su = 0.0
        sy = 0.0
        for j in range(lo, hi):
            su += buf_u[j]
            sy += buf_y[j]
        mu = su / n
        my = sy / n
        num = 0.0
        den = 0.0
        for j in range(lo, hi):
            du = buf_u[j] - mu
            num += du * (buf_y[j] - my)
            den += du * du
        if den > 0.0:
            slope = num / den
    if not (slope < 0.0):
        if j_max > 0 and buf_u[j_max] != buf_u[0]:
            slope = (buf_y[j_max] - buf_y[0]) / (buf_u[j_max] - buf_u[0])
        if not (slope < 0.0):
            slope = math.nan
    if not below and slope < 0.0 and peak > 0.0:
        y0 = buf_y[j_c]
        if 0.0 < y0 < peak:
            g0 = 2.0 * math.sqrt(peak / y0 - 1.0) * y0 * y0
            gf = 2.0 * math.sqrt(1.0 / f - 1.0) * target * target
            slope = slope * gf / g0
    return slope, buf_u[j_c]


def _fit_tail(buf_u, buf_y, n: int, K: int, y_clip: float) -> tuple[float, float]:
    """(dy/du, u at the last point) from the last 2 K + 1 points of a DECLIP record [0, n), clipped points (y >=
    y_clip) excluded; NaN slope with fewer than 2 points."""
    lo = n - 2 * K - 1 if n - 2 * K - 1 > 0 else 0
    m = 0
    su = 0.0
    sy = 0.0
    for j in range(lo, n):
        if buf_y[j] < y_clip:
            su += buf_u[j]
            sy += buf_y[j]
            m += 1
    if m < 2:
        return math.nan, buf_u[n - 1] if n > 0 else 0.0
    mu = su / m
    my = sy / m
    num = 0.0
    den = 0.0
    for j in range(lo, n):
        if buf_y[j] < y_clip:
            du = buf_u[j] - mu
            num += du * (buf_y[j] - my)
            den += du * du
    slope = num / den if den > 0.0 else math.nan
    return slope, buf_u[n - 1]


def _pi_ring_step(r: _PiRing, y_blk: float, ok: bool, first: bool, u0: float, start: int, cfg: tuple,
                  clip: bool = False) -> float:
    """One controller step of one ring (module docstring); returns the heater code (rint of u)."""
    (f, k_c, t_ratio, k2, smin, smax, map_step, ramp_step, recap_step, fall_frac, _reserved, cap_ratio, cap_min,
     lock_tol, fine_step, near_ratio, drop_frac, guard_frac, guard_def, fall_level, rec_level, rec_drop, law, K,
     settle_n, W, n_ref, delay, lo, hi, adc_fs, span) = cfg
    if first:
        r.u = u0
        r.u_ref = u0
        r.u_start = u0
        r.st = start if delay <= 0 else ST_HOLD
        r.fresh = 1
        r.guard = guard_def
        r.lock_cnt = 0
    if not ok:
        return float(np.rint(r.u))
    y = y_blk
    r.y = y
    J = len(r.hist)
    hmax = y
    hmin = y
    if r.hist_n > 0:
        hmax = r.hist[0]
        hmin = r.hist[0]
        for j in range(1, r.hist_n):
            if r.hist[j] > hmax:
                hmax = r.hist[j]
            if r.hist[j] < hmin:
                hmin = r.hist[j]
    st = r.st
    if st == ST_HOLD:                    # warm start: let the thermal state settle before mapping
        r.lock_cnt += 1
        if r.lock_cnt >= delay:
            r.st = start
            r.lock_cnt = 0
            r.u_start = r.u
    elif st == ST_CAPTURE:
        # ramp up until a resonance was seen and the monitor fell back by rec_drop of its maximum: red of the peak
        # (after a capture jump or a smooth passage over the peak), then MAP
        if y < r.y_lo or r.y_lo <= 0.0:
            r.y_lo = y
        if y > r.y_max:
            r.y_max = y
        if r.y_max > cap_ratio * r.y_lo + cap_min and y < (1.0 - rec_drop) * r.y_max:
            r.st = ST_MAP
            r.map_n = 0
            r.y_max = 0.0
            r.u_start = r.u
            r.fine_seen = 0
            r.rise_max = 0.0
            r.step_prev = 0.0
        elif r.u >= hi:
            r.st = ST_FAULT                  # no resonance in the heater range: back to the start code
            r.u = r.u_start
        else:
            r.u = r.u + ramp_step if r.u + ramp_step < hi else hi
    elif st == ST_MAP:
        H = r.buf_u.shape[0]
        if r.map_n < H:
            r.buf_u[r.map_n] = r.u
            r.buf_y[r.map_n] = y
            r.map_n += 1
        if y > r.y_max:
            r.y_max = y
            r.u_max = r.u
        if clip:
            # the monitor clipped at the ADC full scale: the peak cannot be measured; take the full scale as M_pk
            # (the set-point then lies further red than f of the true peak; telemetry peak_codes shows it)
            r.peak = adc_fs
            r.slope, r.u_ref = _fit_slope(r.buf_u, r.buf_y, r.map_n, f, r.peak, K)
            d = r.u_start - r.u
            r.guard = guard_frac * d if d > 0.0 else guard_def
            if r.slope < 0.0:
                r.st = ST_LOCK
            else:
                # clipped from the hand-off on (no record below the full scale to fit): DECLIP
                r.st = ST_DECLIP
                r.map_n = 0
            r.fresh = 1
            r.lock_cnt = 0
        elif r.y_max > 0.0 and (r.y_max - y) > fall_frac * r.y_max:
            # fell off: the fold was crossed before the peak was detected
            r.peak = r.y_max
            r.slope, r.u_ref = _fit_slope(r.buf_u, r.buf_y, r.map_n - 1, f, r.peak, K)
            d = r.u_start - r.u_max
            r.guard = guard_frac * d if d > 0.0 else 0.0
            r.u_guard = r.u_max + r.guard
            r.y_max = 0.0
            r.st = ST_RETURN if r.slope < 0.0 else ST_FAULT
        elif r.fine_seen == 1 and (r.y_max - y) > drop_frac * r.y_max:
            # the peak has been passed while still on the stable branch: hand over to the PI
            r.peak = r.y_max
            r.slope, r.u_ref = _fit_slope(r.buf_u, r.buf_y, r.map_n, f, r.peak, K)
            d = r.u_start - r.u_max
            r.guard = guard_frac * d if d > 0.0 else 0.0
            r.st = ST_LOCK if r.slope < 0.0 else ST_FAULT
            r.fresh = 1
            r.lock_cnt = 0
        elif r.map_n >= H or r.u <= lo:
            r.st = ST_FAULT
        else:
            # near the top: the local slope dy/du collapses only there (the self-heating relief keeps it nearly
            # constant on the flank). Slope over the last W steps against its mean over the first REF steps after
            # the settle steps; below near_ratio of it: fine steps and the peak test armed.
            j = r.map_n - 1
            j0 = settle_n + W
            if r.fine_seen == 0 and j >= j0 and j < H:
                du = r.buf_u[j - W] - r.buf_u[j]
                sl = (r.buf_y[j] - r.buf_y[j - W]) / du if du > 0.0 else 0.0
                if j < j0 + n_ref:
                    r.step_prev += sl
                    if j == j0 + n_ref - 1:
                        r.rise_max = r.step_prev / n_ref
                elif sl < near_ratio * r.rise_max:
                    r.fine_seen = 1
            step = fine_step if r.fine_seen == 1 else map_step
            r.u = r.u - step if r.u - step > lo else lo
    elif st == ST_DECLIP:
        # the monitor was clipped at the ADC full scale from the hand-off on: ramp red by ramp_step, recording, until
        # the monitor reads below the set-point level f x full scale, then fit the slope over the last points and lock
        H = r.buf_u.shape[0]
        if r.map_n < H:
            r.buf_u[r.map_n] = r.u
            r.buf_y[r.map_n] = y
            r.map_n += 1
        if (not clip) and y < f * r.peak:
            r.slope, r.u_ref = _fit_tail(r.buf_u, r.buf_y, r.map_n, K, 0.999 * adc_fs)
            r.st = ST_LOCK if r.slope < 0.0 else ST_FAULT
            r.fresh = 1
            r.lock_cnt = 0
        elif r.u >= hi or r.map_n >= H:
            r.st = ST_FAULT
        else:
            r.u = r.u + ramp_step if r.u + ramp_step < hi else hi
    elif st == ST_RETURN or st == ST_RECAPTURE:
        # ramp up past the guard until the monitor has peaked and turned down: after a capture jump (bistable) or a
        # smooth passage over the peak (no bistability) the ring is then on the red, stable side
        step_codes = ramp_step if st == ST_RETURN else recap_step
        if r.u >= r.u_guard and y > r.y_max:
            r.y_max = y
        if r.y_max > rec_level * r.peak and y < (1.0 - rec_drop) * r.y_max:
            r.st = ST_LOCK
            r.fresh = 1
            r.lock_cnt = 0
        elif r.u >= hi or r.u >= r.u_guard + span:
            # give up (the line is gone or out of reach): back to the code at the fall, never park at full scale
            # (in a bank a full-scale heater loads the shared supply, 07 P1-12)
            r.st = ST_FAULT
            r.u = r.u_guard - r.guard
        else:
            r.u = r.u + step_codes if r.u + step_codes < hi else hi
    elif st == ST_LOCK:
        if r.hist_n >= 1 and (hmax - y) > fall_frac * r.peak and y < fall_level * r.peak:
            r.st = ST_RECAPTURE
            r.n_fall += 1
            r.lock_cnt = 0
            r.u_guard = r.u + r.guard
            r.y_max = 0.0
        else:
            e = _pi_error(r, f, law, smin, smax)
            if r.fresh == 1:
                r.e_prev = e
                r.v = 0.0
                r.fresh = 0
            else:
                r.v = r.v + k2 * e
                du = k_c * ((e - r.e_prev) + t_ratio * e) + r.v
                un = r.u + du
                if un < lo:
                    r.u = lo
                    if r.v < 0.0:                 # anti-windup of the type-2 velocity (module docstring)
                        r.v = 0.0
                elif un > hi:
                    r.u = hi
                    if r.v > 0.0:
                        r.v = 0.0
                else:
                    r.u = un
                r.e_prev = e
            if r.peak > 0.0 and abs(y / r.peak - f) < lock_tol:
                r.lock_cnt += 1
            else:
                r.lock_cnt = 0
    # jump-detection history: the last J block means, newest first
    for j in range(J - 1, 0, -1):
        r.hist[j] = r.hist[j - 1]
    r.hist[0] = y
    if r.hist_n < J:
        r.hist_n += 1
    return float(np.rint(r.u))


def _check_pi_params(p, name: str) -> None:
    if p.start not in _START:
        raise ValueError(f"{name}: start must be one of {sorted(_START)}, got {p.start!r}")
    if not 0.05 <= p.fraction <= 0.98:
        raise ValueError(f"{name}: fraction {p.fraction} outside [0.05, 0.98]")
    if p.jump_steps < 1 or p.map_buffer < 8 or p.fit_half_width < 1:
        raise ValueError(f"{name}: jump_steps >= 1, fit_half_width >= 1 and map_buffer >= 8 required")
    if p.schedule not in ("auto", "quadratic", "linear", "none"):
        raise ValueError(f"{name}: schedule must be auto | quadratic | linear | none, got {p.schedule!r}")
    if min(p.map_step_codes, p.fine_step_codes, p.ramp_step_codes, p.recapture_step_codes) <= 0:
        raise ValueError(f"{name}: sweep steps must be > 0 codes")


# ---- Python controller --------------------------------------------------------------------------------------------

@register("pi_lock")
class PiLock(BaseController):
    """Drop-port fraction-of-peak PI lock on the stable (red) side (module docstring); writes ``tx.heater``.

    Telemetry (B, N): state, locked, detuning_est_pm (Lorentzian inversion with the ``linewidth`` token, else NaN),
    integrator (heater code u), integrator_saturated, peak_codes, slope_per_code, error_norm (y/M_pk - f),
    fall_count.
    """
    name = "pi_lock"

    @dataclass
    class Params:
        fraction: float = 0.7              # set-point as a fraction of the measured peak, red (stable) side
        start: str = "map"                 # map | lock (factory table) | capture (ramp up to a capture, then map)
        init_codes: object = None          # hand-off heater code(s): scalar, (N,) or (B, N); None = tx.heater.applied
        peak_codes: object = None          # factory peak monitor code(s) (start='lock')
        slope_per_code: object = None      # factory dy/dcode at the set-point, codes/code < 0 (start='lock')
        crossover_hz: float = 2500.0       # design choice: loop delays ~12 us and a 13 us slow pole (see docstring)
        ti_s: float | None = None          # PI zero; None = nominal tau_slow (thermal_dynamics token) or 13 us
        k2: float = 0.0                    # type-2 gain per step (0 = type 1; 04 constraint 3); reset at a clamp
        schedule: str = "auto"             # driver-law gain schedule: auto | quadratic | linear | none
        sigma_min: float = 0.5
        sigma_max: float = 4.0
        map_step_codes: float = 8.0        # MAP down-step on the flank of the stable branch
        fine_step_codes: float = 1.0       # smallest MAP step near the top (the fold is ~9 pm past the peak in g26)
        near_ratio: float = 0.5            # MAP slope below near_ratio * the flank slope: near the top (fine, armed)
        map_settle_steps: int = 6          # MAP steps before slopes are used (the first responses lag the steps)
        slope_window: int = 3              # MAP slope over the last slope_window steps
        ref_steps: int = 8                 # MAP steps averaged into the flank slope reference
        peak_drop_frac: float = 2e-4       # peak passed: y < (1 - peak_drop_frac) * max (~2.4 pm past a 170 pm HWHM)
        ramp_step_codes: float = 16.0      # RETURN / CAPTURE up-ramp
        recapture_step_codes: float = 32.0
        recapture_span_codes: float = 8000.0  # RETURN/RECAPTURE gives up this far above the guard (-> FAULT)
        fall_frac: float = 0.35            # fall-off: drop > fall_frac * peak within jump_steps steps ...
        fall_level: float = 0.4            # ... to below fall_level * peak (a PI transient from the peak stays above)
        recapture_level: float = 0.3       # RETURN/RECAPTURE ends once the monitor peaked above level * peak ...
        recapture_drop: float = 0.02       # ... and fell back by this fraction (the ring is then red of the peak)
        guard_frac: float = 0.5            # accept a capture only above u_peak (u_fall in LOCK) + guard_frac * (MAP
        #                                    start - peak code): rejects captures by another line or order (g26: the
        #                                    next order of rings 4-7 is 3.4 FWHM from line n-4, BND-1)
        guard_codes: float = 500.0         # the guard when no MAP was run (start='lock')
        capture_ratio: float = 2.0         # start='capture': a resonance was seen once max(y) > ratio * min(y) +
        #                                    capture_min_codes (since the start); then wait for the fall back
        capture_min_codes: float = 500.0
        jump_steps: int = 3
        lock_tol: float = 0.02             # |y/M_pk - f| < lock_tol for lock_steps steps -> telemetry 'locked'
        lock_steps: int = 20
        fit_half_width: int = 4            # points on each side of the f-crossing in the slope fit
        map_buffer: int = 4096             # longest MAP, steps
        start_delay_s: float = 0.0         # hold the hand-off code this long before starting (thermal settling)
        latency_ticks: int = 0             # digital pipeline depth (07 P0-6)

    def __init__(self, params=None, **overrides):
        super().__init__(params, **overrides)
        _check_pi_params(self.params, self.name)
        self.latency_ticks = int(self.params.latency_ticks)

    def _cfg(self, spec: ControllerSpec) -> tuple:
        p = self.params
        self.nominal = NominalModel.from_spec(spec)
        self.law = law_code(p.schedule, self.nominal)
        self.k_c, self.t_ratio = design_pi_gains(spec, self.nominal, p.crossover_hz, p.ti_s)
        self.code_lo, self.code_hi = float(spec.heater_dac.code_min), float(spec.heater_dac.code_max)
        return (float(p.fraction), self.k_c, self.t_ratio, float(p.k2), float(p.sigma_min), float(p.sigma_max),
                float(p.map_step_codes), float(p.ramp_step_codes), float(p.recapture_step_codes), float(p.fall_frac),
                0.0, float(p.capture_ratio), float(p.capture_min_codes), float(p.lock_tol),
                float(p.fine_step_codes), float(p.near_ratio), float(p.peak_drop_frac), float(p.guard_frac),
                float(p.guard_codes), float(p.fall_level), float(p.recapture_level), float(p.recapture_drop),
                int(self.law), int(p.fit_half_width), int(p.map_settle_steps),
                int(p.slope_window), int(p.ref_steps), int(round(p.start_delay_s / spec.period_s)), self.code_lo,
                self.code_hi, float(spec.adc.full_scale_codes), float(p.recapture_span_codes))

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        p = self.params
        sh = (spec.n_batch, spec.n_rings)
        self.cfg = self._cfg(spec)
        init = per_ring(p.init_codes, sh, "init_codes")
        self.init_codes = None if init is None else np.rint(init)      # whole codes, as the compiled port
        peak = per_ring(p.peak_codes, sh, "peak_codes")
        slope = per_ring(p.slope_per_code, sh, "slope_per_code")
        if p.start == "lock" and (peak is None or slope is None):
            raise ValueError(f"{self.name}: start='lock' needs peak_codes and slope_per_code (the factory table)")
        self.rings = [[_PiRing(int(p.jump_steps), int(p.map_buffer)) for _ in range(sh[1])] for _ in range(sh[0])]
        for b in range(sh[0]):
            for n in range(sh[1]):
                if peak is not None:
                    self.rings[b][n].peak = float(peak[b, n])
                if slope is not None:
                    self.rings[b][n].slope = float(slope[b, n])
        self.first = True
        self.y = np.zeros(sh)

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        spec = self._require_spec()
        y, ok = block_inputs(obs, self.y)
        self.y = y
        u0 = None
        if self.first:
            applied = (np.asarray(obs.sensors["tx.heater.applied"], dtype=float) if "tx.heater.applied" in obs.sensors
                       else np.full((spec.n_batch, spec.n_rings), float(spec.channels["tx.heater"].init)))
            u0 = applied if self.init_codes is None else self.init_codes
        out = np.zeros((spec.n_batch, spec.n_rings), dtype=np.int64)
        start = _START[self.params.start]
        over = (np.asarray(obs.sensors["adc_overrange"], dtype=bool) if "adc_overrange" in obs.sensors
                else np.zeros((spec.n_batch, spec.n_rings), bool))
        for b in range(spec.n_batch):
            for n in range(spec.n_rings):
                out[b, n] = int(_pi_ring_step(self.rings[b][n], float(y[b, n]), bool(ok[b, n]), self.first,
                                              0.0 if u0 is None else float(u0[b, n]), start, self.cfg,
                                              bool(over[b, n])))
        self.first = False
        return {"tx.heater": out}

    def ring_values(self, attr: str) -> np.ndarray:
        """(B, N) float array of one per-ring state attribute (u, y, peak, slope, st, ...)."""
        return np.array([[float(getattr(r, attr)) for r in row] for row in self.rings])

    def telemetry(self) -> dict[str, np.ndarray]:
        return _pi_telemetry(self.params, self.nominal, self.code_lo, self.code_hi, self.ring_values("st"),
                             self.ring_values("u"), self.ring_values("y"), self.ring_values("peak"),
                             self.ring_values("slope"), self.ring_values("lock_cnt"), self.ring_values("n_fall"))


def _pi_telemetry(p, nominal, lo, hi, st, u, y, peak, slope, lock_cnt, n_fall) -> dict[str, np.ndarray]:
    pk = np.where(peak > 0, peak, np.nan)
    yn = y / pk
    if nominal.fwhm_pm:
        det = 0.5 * nominal.fwhm_pm * np.sqrt(np.maximum(1.0 / np.maximum(yn, 1e-9) - 1.0, 0.0))
    else:
        det = np.full(u.shape, np.nan)
    lock = st == ST_LOCK
    return {"state": st, "locked": lock & (lock_cnt >= p.lock_steps), "detuning_est_pm": det, "integrator": u,
            "integrator_saturated": lock & ((u <= lo) | (u >= hi)), "peak_codes": peak, "slope_per_code": slope,
            "error_norm": yn - p.fraction, "fall_count": n_fall}


# ---- compiled port (07 P0-9 flat ABI) -----------------------------------------------------------------------------
# p_i64: [0 N, 1 M, 2 off_samples, 3 off_count, 4 off_valid, 5 off_applied, 6 off_heater, 7 start, 8 law, 9 J, 10 H,
#         11 K, 12 has_init, 13 lock_steps, 14 map_settle_steps, 15 slope_window, 16 ref_steps, 17 delay,
#         18.. init codes (N)]
# p_f64: [0 f, 1 k_c, 2 T/T_i, 3 k2, 4 smin, 5 smax, 6 map_step, 7 ramp_step, 8 recap_step, 9 fall_frac,
#         10 reserved, 11 capture_ratio, 12 capture_min, 13 lock_tol, 14 fine_step, 15 near_ratio, 16 drop_frac,
#         17 guard_frac, 18 guard_codes, 19 fall_level, 20 recapture_level, 21 recapture_drop, 22 code_lo,
#         23 code_hi, 24 recapture_span, 25.. peak (N), 25+N.. slope (N)]
# s_i64: [0 first] + per ring n at 1 + 7 n: [st, hist_n, lock_cnt, map_n, n_fall, fresh, fine_seen]
# s_f64: per ring n at n R (R = 15 + J + 2 H): [u, y, e_prev, v, peak, slope, u_ref, y_max, u_max, y_lo, u_start,
#        guard, u_guard, rise_max, step_prev, hist (J), buf_u (H), buf_y (H)]
_PI_NI = 18
_PI_NUF = 22          # user float parameters before the per-ring tables (then span)
_PI_NF = 25
_PI_SI = 7
_PI_SF = 15
_SAMPLES, _COUNT = int(ObsId.TX_MONITOR_SAMPLES), int(ObsId.TX_MONITOR_COUNT)
_VALID, _APPLIED, _HEATER = int(ObsId.MONITOR_VALID), int(ObsId.TX_HEATER_APPLIED), int(ActId.TX_HEATER)
_OVER = int(ObsId.ADC_OVERRANGE)


@nb.njit(cache=True)
def pi_init(spec_i64, spec_f64):
    """Compiled init (07 P0-9): parameters from the spec arrays and PiLockCompiled's user parameters."""
    n = spec_i64[api.SPEC_I_N_RINGS]
    p = np.zeros(_PI_NI + n + 2, dtype=np.int64)
    p[0] = n
    p[1] = spec_i64[api.SPEC_I_M]
    p[_PI_NI + n] = api.obs_offset(spec_i64, _OVER)
    p[_PI_NI + n + 1] = (1 << spec_i64[api.SPEC_I_ADC_BITS]) - 1
    p[2] = api.obs_offset(spec_i64, _SAMPLES)
    p[3] = api.obs_offset(spec_i64, _COUNT)
    p[4] = api.obs_offset(spec_i64, _VALID)
    p[5] = api.obs_offset(spec_i64, _APPLIED)
    p[6] = api.act_offset(spec_i64, _HEATER)
    for k in range(11):
        p[7 + k] = api.user_i64(spec_i64, k)          # start law J H K has_init lock_steps settle W ref delay
    for i in range(n):
        p[_PI_NI + i] = api.user_i64(spec_i64, 11 + i)
    pf = np.zeros(_PI_NF + 2 * n)
    for k in range(_PI_NUF):
        pf[k] = api.user_f64(spec_f64, k)
    pf[22] = spec_i64[api.SPEC_I_HEATER_MIN]
    pf[23] = spec_i64[api.SPEC_I_HEATER_MAX]
    pf[24] = api.user_f64(spec_f64, _PI_NUF)
    for i in range(2 * n):
        pf[_PI_NF + i] = api.user_f64(spec_f64, _PI_NUF + 1 + i)
    J = p[9]
    H = p[10]
    s_i = np.zeros(1 + _PI_SI * n, dtype=np.int64)
    s_i[0] = 1
    R = _PI_SF + J + 2 * H
    s_f = np.zeros(R * n)
    for i in range(n):
        s_f[i * R + 4] = pf[_PI_NF + i]            # peak (factory table or 0)
        s_f[i * R + 5] = pf[_PI_NF + n + i]        # slope (factory table or nan)
        s_i[1 + _PI_SI * i + 5] = 1                # fresh
    return p, pf, s_i, s_f


@nb.njit(cache=True)
def _pi_err_c(y, u, u_ref, peak, slope, f, law, smin, smax):
    s = abs(slope)
    if not (s > 0.0) or math.isinf(s):
        s = 1.0
    return (y - f * peak) / s * sigma_schedule(law, u, u_ref, smin, smax)


@nb.njit(cache=True)
def _fit_slope_c(sf, bu, by, j_end, f, peak, K):
    """Compiled twin of _fit_slope (same operations, same order); bu, by: offsets of the record in sf."""
    if j_end < 2:
        return math.nan, sf[bu]
    j_max = 0
    for j in range(1, j_end):
        if sf[by + j] > sf[by + j_max]:
            j_max = j
    target = f * peak
    j_c = -1
    for j in range(j_max, -1, -1):
        if sf[by + j] <= target:
            j_c = j
            break
    below = j_c >= 0
    if not below:
        j_c = 0
    lo = j_c - K if j_c - K > 0 else 0
    hi = j_c + K + 1 if j_c + K + 1 < j_max + 1 else j_max + 1
    n = hi - lo
    slope = math.nan
    if n >= 2:
        su = 0.0
        sy = 0.0
        for j in range(lo, hi):
            su += sf[bu + j]
            sy += sf[by + j]
        mu = su / n
        my = sy / n
        num = 0.0
        den = 0.0
        for j in range(lo, hi):
            du = sf[bu + j] - mu
            num += du * (sf[by + j] - my)
            den += du * du
        if den > 0.0:
            slope = num / den
    if not (slope < 0.0):
        if j_max > 0 and sf[bu + j_max] != sf[bu]:
            slope = (sf[by + j_max] - sf[by]) / (sf[bu + j_max] - sf[bu])
        if not (slope < 0.0):
            slope = math.nan
    if not below and slope < 0.0 and peak > 0.0:
        y0 = sf[by + j_c]
        if 0.0 < y0 < peak:
            g0 = 2.0 * math.sqrt(peak / y0 - 1.0) * y0 * y0
            gf = 2.0 * math.sqrt(1.0 / f - 1.0) * target * target
            slope = slope * gf / g0
    return slope, sf[bu + j_c]


@nb.njit(cache=True)
def _fit_tail_c(sf, bu, by, n, K, y_clip):
    """Compiled twin of _fit_tail (same operations, same order)."""
    lo = n - 2 * K - 1 if n - 2 * K - 1 > 0 else 0
    m = 0
    su = 0.0
    sy = 0.0
    for j in range(lo, n):
        if sf[by + j] < y_clip:
            su += sf[bu + j]
            sy += sf[by + j]
            m += 1
    if m < 2:
        return math.nan, sf[bu + n - 1] if n > 0 else 0.0
    mu = su / m
    my = sy / m
    num = 0.0
    den = 0.0
    for j in range(lo, n):
        if sf[by + j] < y_clip:
            du = sf[bu + j] - mu
            num += du * (sf[by + j] - my)
            den += du * du
    slope = num / den if den > 0.0 else math.nan
    return slope, sf[bu + n - 1]


@nb.njit(cache=True)
def pi_step(p, pf, s, sf, obs, act):
    """Compiled step (07 P0-9 ABI): the per-ring algorithm of :func:`_pi_ring_step`, operation for operation."""
    n_r = p[0]
    m_max = p[1]
    start = p[7]
    law = p[8]
    J = p[9]
    H = p[10]
    K = p[11]
    f = pf[0]
    k_c = pf[1]
    t_ratio = pf[2]
    k2 = pf[3]
    smin = pf[4]
    smax = pf[5]
    map_step = pf[6]
    ramp_step = pf[7]
    recap_step = pf[8]
    fall_frac = pf[9]
    cap_ratio = pf[11]
    cap_min = pf[12]
    lock_tol = pf[13]
    fine_step = pf[14]
    near_ratio = pf[15]
    drop_frac = pf[16]
    guard_frac = pf[17]
    guard_def = pf[18]
    fall_level = pf[19]
    rec_level = pf[20]
    rec_drop = pf[21]
    lo = pf[22]
    hi = pf[23]
    R = _PI_SF + J + 2 * H
    first = s[0] == 1
    active_ring = obs[1]
    for i in range(n_r):
        si = 1 + _PI_SI * i
        b = i * R
        if first:
            if p[12] == 1:
                u0 = float(p[_PI_NI + i])
            elif p[5] >= 0:
                u0 = float(obs[p[5] + i])
            else:
                u0 = 0.0
            sf[b] = u0
            sf[b + 6] = u0
            sf[b + 10] = u0
            s[si] = start if p[17] <= 0 else ST_HOLD
            s[si + 5] = 1
            sf[b + 11] = guard_def
            s[si + 2] = 0
        cnt = obs[p[3] + i]
        ok = cnt > 0
        if p[4] >= 0 and obs[p[4] + i] == 0:
            ok = False
        if active_ring >= 0 and active_ring != i:
            ok = False
        if ok:
            acc = 0.0
            for m in range(cnt):
                acc += float(obs[p[2] + i * m_max + m])
            y = acc / cnt
            sf[b + 1] = y
            hb = b + _PI_SF
            hn = s[si + 1]
            hmax = y
            hmin = y
            if hn > 0:
                hmax = sf[hb]
                hmin = sf[hb]
                for j in range(1, hn):
                    if sf[hb + j] > hmax:
                        hmax = sf[hb + j]
                    if sf[hb + j] < hmin:
                        hmin = sf[hb + j]
            st = s[si]
            u = sf[b]
            clip = p[_PI_NI + n_r] >= 0 and obs[p[_PI_NI + n_r] + i] != 0
            if st == ST_HOLD:
                s[si + 2] += 1
                if s[si + 2] >= p[17]:
                    s[si] = start
                    s[si + 2] = 0
                    sf[b + 10] = u
            elif st == ST_CAPTURE:
                if y < sf[b + 9] or sf[b + 9] <= 0.0:
                    sf[b + 9] = y
                if y > sf[b + 7]:
                    sf[b + 7] = y
                if sf[b + 7] > cap_ratio * sf[b + 9] + cap_min and y < (1.0 - rec_drop) * sf[b + 7]:
                    s[si] = ST_MAP
                    s[si + 3] = 0
                    sf[b + 7] = 0.0
                    sf[b + 10] = u
                    s[si + 6] = 0
                    sf[b + 13] = 0.0
                    sf[b + 14] = 0.0
                elif u >= hi:
                    s[si] = ST_FAULT
                    sf[b] = sf[b + 10]
                else:
                    sf[b] = u + ramp_step if u + ramp_step < hi else hi
            elif st == ST_MAP:
                bu = b + _PI_SF + J
                by = bu + H
                mn = s[si + 3]
                if mn < H:
                    sf[bu + mn] = u
                    sf[by + mn] = y
                    mn += 1
                    s[si + 3] = mn
                if y > sf[b + 7]:
                    sf[b + 7] = y
                    sf[b + 8] = u
                ymax = sf[b + 7]
                if clip:
                    pk = float(p[_PI_NI + n_r + 1])
                    sf[b + 4] = pk
                    sl, ur = _fit_slope_c(sf, bu, by, mn, f, pk, K)
                    sf[b + 5] = sl
                    sf[b + 6] = ur
                    d = sf[b + 10] - u
                    sf[b + 11] = guard_frac * d if d > 0.0 else guard_def
                    if sl < 0.0:
                        s[si] = ST_LOCK
                    else:
                        s[si] = ST_DECLIP
                        s[si + 3] = 0
                    s[si + 5] = 1
                    s[si + 2] = 0
                elif ymax > 0.0 and (ymax - y) > fall_frac * ymax:
                    sf[b + 4] = ymax
                    sl, ur = _fit_slope_c(sf, bu, by, mn - 1, f, ymax, K)
                    sf[b + 5] = sl
                    sf[b + 6] = ur
                    d = sf[b + 10] - sf[b + 8]
                    sf[b + 11] = guard_frac * d if d > 0.0 else 0.0
                    sf[b + 12] = sf[b + 8] + sf[b + 11]
                    sf[b + 7] = 0.0
                    s[si] = ST_RETURN if sl < 0.0 else ST_FAULT
                elif s[si + 6] == 1 and (ymax - y) > drop_frac * ymax:
                    sf[b + 4] = ymax
                    sl, ur = _fit_slope_c(sf, bu, by, mn, f, ymax, K)
                    sf[b + 5] = sl
                    sf[b + 6] = ur
                    d = sf[b + 10] - sf[b + 8]
                    sf[b + 11] = guard_frac * d if d > 0.0 else 0.0
                    s[si] = ST_LOCK if sl < 0.0 else ST_FAULT
                    s[si + 5] = 1
                    s[si + 2] = 0
                elif mn >= H or u <= lo:
                    s[si] = ST_FAULT
                else:
                    W = p[15]
                    n_ref = p[16]
                    j = mn - 1
                    j0 = p[14] + W
                    if s[si + 6] == 0 and j >= j0 and j < H:
                        du = sf[bu + j - W] - sf[bu + j]
                        sl = (sf[by + j] - sf[by + j - W]) / du if du > 0.0 else 0.0
                        if j < j0 + n_ref:
                            sf[b + 14] += sl
                            if j == j0 + n_ref - 1:
                                sf[b + 13] = sf[b + 14] / n_ref
                        elif sl < near_ratio * sf[b + 13]:
                            s[si + 6] = 1
                    step = fine_step if s[si + 6] == 1 else map_step
                    sf[b] = u - step if u - step > lo else lo
            elif st == ST_DECLIP:
                bu = b + _PI_SF + J
                by = bu + H
                mn = s[si + 3]
                if mn < H:
                    sf[bu + mn] = u
                    sf[by + mn] = y
                    mn += 1
                    s[si + 3] = mn
                if (not clip) and y < f * sf[b + 4]:
                    sl, ur = _fit_tail_c(sf, bu, by, mn, K, 0.999 * float(p[_PI_NI + n_r + 1]))
                    sf[b + 5] = sl
                    sf[b + 6] = ur
                    s[si] = ST_LOCK if sl < 0.0 else ST_FAULT
                    s[si + 5] = 1
                    s[si + 2] = 0
                elif u >= hi or mn >= H:
                    s[si] = ST_FAULT
                else:
                    sf[b] = u + ramp_step if u + ramp_step < hi else hi
            elif st == ST_RETURN or st == ST_RECAPTURE:
                step_codes = ramp_step if st == ST_RETURN else recap_step
                if u >= sf[b + 12] and y > sf[b + 7]:
                    sf[b + 7] = y
                if sf[b + 7] > rec_level * sf[b + 4] and y < (1.0 - rec_drop) * sf[b + 7]:
                    s[si] = ST_LOCK
                    s[si + 5] = 1
                    s[si + 2] = 0
                elif u >= hi or u >= sf[b + 12] + pf[24]:
                    s[si] = ST_FAULT
                    sf[b] = sf[b + 12] - sf[b + 11]
                else:
                    sf[b] = u + step_codes if u + step_codes < hi else hi
            elif st == ST_LOCK:
                peak = sf[b + 4]
                if hn >= 1 and (hmax - y) > fall_frac * peak and y < fall_level * peak:
                    s[si] = ST_RECAPTURE
                    s[si + 4] += 1
                    s[si + 2] = 0
                    sf[b + 12] = u + sf[b + 11]
                    sf[b + 7] = 0.0
                else:
                    e = _pi_err_c(y, u, sf[b + 6], peak, sf[b + 5], f, law, smin, smax)
                    if s[si + 5] == 1:
                        sf[b + 2] = e
                        sf[b + 3] = 0.0
                        s[si + 5] = 0
                    else:
                        v = sf[b + 3] + k2 * e
                        sf[b + 3] = v
                        du = k_c * ((e - sf[b + 2]) + t_ratio * e) + v
                        un = u + du
                        if un < lo:
                            sf[b] = lo
                            if v < 0.0:
                                sf[b + 3] = 0.0
                        elif un > hi:
                            sf[b] = hi
                            if v > 0.0:
                                sf[b + 3] = 0.0
                        else:
                            sf[b] = un
                        sf[b + 2] = e
                    if peak > 0.0 and abs(y / peak - f) < lock_tol:
                        s[si + 2] += 1
                    else:
                        s[si + 2] = 0
            for j in range(J - 1, 0, -1):
                sf[hb + j] = sf[hb + j - 1]
            sf[hb] = y
            if hn < J:
                s[si + 1] = hn + 1
        act[p[6] + i] = np.int64(np.rint(sf[b]))
    s[0] = 0


def _ring_list(v, n: int, name: str, default: float, owner: str) -> np.ndarray:
    if v is None:
        return np.full(n, default, dtype=float)
    a = np.asarray(v, dtype=float).ravel()
    if a.size == 1:
        return np.full(n, float(a[0]))
    if a.size != n:
        raise ValueError(f"{owner}: {name} needs a scalar or N = {n} values (shared by every world in the "
                         f"compiled ABI)")
    return a


class PiLockCompiled(CompiledController):
    """Compiled :class:`PiLock` (07 P0-9 flat ABI; the kernel calls :func:`pi_step` per batch element).

    Same parameters as :class:`PiLock`. Per-ring tables (init_codes, peak_codes, slope_per_code) are scalars or (N,)
    lists shared by every batch element (the ABI tiles the state over B). Telemetry reads the state arrays.
    """
    name = "pi_lock_c"

    def __init__(self, params=None, **overrides):
        self.params = PiLock._make_params(params, overrides)
        _check_pi_params(self.params, "pi_lock_c")
        super().__init__("pi_lock_c", pi_init, pi_step, latency_ticks=int(self.params.latency_ticks))

    @classmethod
    def from_params(cls, params):
        return cls(params)

    def reset(self, spec: ControllerSpec) -> None:
        p = self.params
        n = spec.n_rings
        cfg = PiLock._cfg(self, spec)               # sets nominal, law, k_c, t_ratio, code_lo, code_hi
        if p.start == "lock" and (p.peak_codes is None or p.slope_per_code is None):
            raise ValueError(f"{self.name}: start='lock' needs peak_codes and slope_per_code (the factory table)")
        init = _ring_list(p.init_codes, n, "init_codes", 0.0, self.name)
        self.params_i64 = np.concatenate([[_START[p.start], self.law, int(p.jump_steps), int(p.map_buffer),
                                           int(p.fit_half_width), 0 if p.init_codes is None else 1,
                                           int(p.lock_steps), int(p.map_settle_steps), int(p.slope_window),
                                           int(p.ref_steps), int(round(p.start_delay_s / spec.period_s))],
                                          np.rint(init).astype(np.int64)]).astype(np.int64)
        self.params_f64 = np.concatenate([np.asarray(cfg[:_PI_NUF], dtype=float), [float(p.recapture_span_codes)],
                                          _ring_list(p.peak_codes, n, "peak_codes", 0.0, self.name),
                                          _ring_list(p.slope_per_code, n, "slope_per_code", math.nan, self.name)])
        super().reset(spec)

    def ring_values(self, k: int, ints: bool = False) -> np.ndarray:
        """(B, N) values of per-ring state slot k (float layout, or the int layout with ints=True)."""
        p_i, _, s_i, s_f = self.state
        n = self.spec.n_rings
        if ints:
            return np.stack([s_i[:, 1 + _PI_SI * i + k] for i in range(n)], axis=1).astype(float)
        R = _PI_SF + int(p_i[0, 9]) + 2 * int(p_i[0, 10])
        return np.stack([s_f[:, i * R + k] for i in range(n)], axis=1)

    def telemetry(self) -> dict[str, np.ndarray]:
        return _pi_telemetry(self.params, self.nominal, self.code_lo, self.code_hi, self.ring_values(0, True),
                             self.ring_values(0), self.ring_values(1), self.ring_values(4), self.ring_values(5),
                             self.ring_values(2, True), self.ring_values(4, True))


register("pi_lock_c", PiLockCompiled)
