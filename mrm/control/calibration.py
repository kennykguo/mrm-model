"""Controller-side calibration: the knowledge gate on ``spec.nominal``, the nominal heater model, the simulated
factory calibration (07 P1-1 ``factory_cal(plant, procedure)``) and the cold-start path.

Everything here sees only what a controller or a factory test station may see: the
:class:`mrm.control.api.ControllerSpec`, its knowledge-filtered nominal design, and sensor channels (ADC codes,
sample ticks, temperature-sensor codes, applied codes). Nothing here imports the plant, ``mrm.params.realize`` or
Truth (07 P1-1 ``test_control_does_not_import_truth``).

1. Knowledge gate (07 P1-1)
---------------------------
``spec.nominal`` is the unsampled design (the Config). A controller may read a nominal value only when its token is
in ``controller_interface.knowledge``. :data:`KNOWLEDGE_TOKENS` lists the tokens this module defines and the config
keys each one releases; :data:`FIELD_TOKENS` maps each field group to every token that releases it (these tokens and
the finer ones of the controllers-A helpers, so either vocabulary works). :func:`nominal_design` returns a
:class:`NominalDesign` whose fields are ``None`` when no token releases them, so a controller falls back to
measurement (the acquisition sweep measures the peak, the background and the slope) or to its own documented
defaults. The 07 P1-1 example name ``heater_r_nominal`` releases the cold heater resistance.

2. Nominal heater model
-----------------------
Drive coordinate w in [0, 1]: w = (code / code_max)^q with q = 2 for a voltage or current DAC (P = V^2/R, I^2 R)
and q = 1 for a power-linear drive (power, dac_pwm). w is proportional to the ideal (cold-resistance) heater power,
so a ring's resonance moves almost linearly in w; sweeps step uniformly in w.
Heater power with the TCR droop of the heater resistor (07 P1-11, textbook HEATER_OHM_COLD):
    voltage: P (1 + y P) = V^2 / R0          ->  P = (sqrt(1 + 4 y V^2 / R0) - 1) / (2 y)
    current: P = I^2 R0 (1 + y P)            ->  P = I^2 R0 / (1 - y I^2 R0)
    power / dac_pwm: P = w P_fs
with y = TCR * R_th,heater (1/W) and R_th,heater = heater_self_ratio * Z, Z = FSR_lambda / (s0 P_FSR) (K/W, the
first-order R2 inversion; 07 P0-3, D16). Supply current: voltage I = V / R(T), current I = I_set, power
I = sqrt(P / R(T)) (rms, as mrm.plant.electrical.supply_limit_row).
Tuning: dlambda/dP = FSR_lambda / P_FSR (m/W, 07 D16 anchor). Linewidth: FWHM = lambda / Q_L.

3. Factory calibration (07 P1-1)
--------------------------------
:func:`factory_cal` runs a simulated factory heater sweep on a test station (:class:`CalBench`): for each group of
rings (sized by the die supply limit), one sweep with every laser line on, labelled by the pattern fit of section 4
(one full FSR), and ceil(log2 L) sweeps with a binary code of lines enabled (a test station can switch its lines),
the model-free cross-check of those labels. The result is a :class:`FactoryCalTable` of measured, noisy numbers per
ring: the drive coordinate of every (line, order) resonance seen, the landing level and the floor, and the die
temperature at the calibration. :meth:`FactoryCalTable.predict_w` gives the warm-start drive for (ring, line) at
another die temperature (nominal dlambda/dT when known).

4. Cold start (scenario B5)
---------------------------
Without the table a controller labels the lines itself from one sweep over a full FSR: :func:`label_lines_by_pattern`
matches the observed resonance positions (one per line per FSR) to the pattern the nominal line grid predicts, by
scanning the unknown FSR and the cyclic shift (no ring model needed; the g26 grid of 8 lines at 400 GHz with an FSR
between 1.6 and 2.0 THz leaves exactly one empty gap per FSR, between lines 4 and 3, which fixes the labels).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "KNOWLEDGE_TOKENS", "FIELD_TOKENS", "NominalDesign", "nominal_design", "has_knowledge",
    "drive_exponent", "w_of_code", "code_of_w", "heater_power_w", "heater_current_a",
    "CalBench", "FactoryProcedure", "FactoryCalTable", "factory_cal",
    "label_lines_by_pattern", "label_bank_by_pattern", "predicted_pattern", "fsr_prior_hz", "drop_first",
    "with_first", "law_exponents",
]

C0 = 299_792_458.0

#: knowledge token -> (description, config keys it releases)
KNOWLEDGE_TOKENS: dict[str, tuple[str, tuple[str, ...]]] = {
    "line_grid": ("nominal laser grid (spec.line_grid_hz)", ("channel_plan.center_thz", "channel_plan.spacing_ghz",
                                                            "channel_plan.lines")),
    "ring_to_line": ("nominal ring-to-line assignment (spec.ring_to_line)", ("channel_plan.ring_to_line",)),
    "heater_nominal": ("heater driver law and full scale, cold resistance, TCR, P_FSR, heater self ratio; the die "
                       "heater supply limits", (
                           "ring.heater.driver_mode", "ring.heater.full_scale_v", "ring.heater.full_scale_a",
                           "ring.heater.full_scale_w", "ring.heater.r_cold_ohm", "ring.heater.tcr_per_k",
                           "ring.heater.p_fsr_mw", "ring.thermal.heater_self_ratio",
                           "array.heater_total_power_max_w", "array.heater_supply_current_max_a")),
    "optics_nominal": ("loaded Q (linewidth), FSR, design resonance offset and its temperature",
                       ("ring.optics.q_loaded", "ring.optics.fsr_thz", "ring.optics.resonance_offset_nm",
                        "ring.optics.resonance_offset_ref_temp_c")),
    "dlambda_dt_nominal": ("nominal thermo-optic tuning dlambda/dT and its curvature",
                           ("ring.thermal.dlambda_dt_pm_per_k", "ring.thermal.dlambda_dt_beta_per_k")),
    "monitor_nominal": ("monitor responsivity and ADC full scale", ("ring.monitor.responsivity_a_per_w",
                                                                    "ring.monitor.adc_full_scale_ua")),
}

_LAW_Q = {"voltage": 2, "current": 2, "power": 1, "dac_pwm": 1}


def has_knowledge(spec, token: str) -> bool:
    """True if ``token`` is in ``spec.knowledge`` (07 P1-1)."""
    return token in set(getattr(spec, "knowledge", ()) or ())


@dataclass(frozen=True)
class NominalDesign:
    """Nominal design numbers a controller may use (SI). A field is None when its knowledge token is missing.

    heater_law (str), drive_q (2 or 1), v_fs_v, i_fs_a, p_fs_w (full scales of the law in use), r_cold_ohm, tcr_per_k,
    p_fsr_w, heater_self_ratio, y_per_w (heater-resistor droop TCR * R_th,H, 1/W), supply_power_max_w,
    supply_current_max_a [heater_nominal]; q_loaded, fsr_hz, fsr_range_hz (the FSR leaf's declared range, which
    spans every FSR hypothesis of the uncertainty set; None if the leaf has none), offset_m, offset_ref_temp_k
    [optics_nominal];
    s0_m_per_k, beta_per_k [dlambda_dt_nominal]; responsivity_a_per_w, adc_full_scale_a [monitor_nominal];
    lambda0_m: centre line wavelength [line_grid]; code_max (from the DAC facts, always known).
    """
    code_max: int
    heater_law: str | None = None
    drive_q: int | None = None
    v_fs_v: float | None = None
    i_fs_a: float | None = None
    p_fs_w: float | None = None
    r_cold_ohm: float | None = None
    tcr_per_k: float | None = None
    p_fsr_w: float | None = None
    heater_self_ratio: float | None = None
    y_per_w: float | None = None
    supply_power_max_w: float | None = None
    supply_current_max_a: float | None = None
    q_loaded: float | None = None
    fsr_hz: float | None = None
    fsr_range_hz: tuple | None = None
    offset_m: float | None = None
    offset_ref_temp_k: float | None = None
    s0_m_per_k: float | None = None
    beta_per_k: float | None = None
    responsivity_a_per_w: float | None = None
    adc_full_scale_a: float | None = None
    lambda0_m: float | None = None

    # ---- derived (None when an input is missing)
    @property
    def fsr_m(self) -> float | None:
        if self.fsr_hz is None or self.lambda0_m is None:
            return None
        return self.fsr_hz * self.lambda0_m ** 2 / C0

    @property
    def fwhm_m(self) -> float | None:
        if self.q_loaded is None or self.lambda0_m is None:
            return None
        return self.lambda0_m / self.q_loaded

    @property
    def tuning_m_per_w(self) -> float | None:
        """dlambda/dP_heater = FSR / P_FSR (07 D16)."""
        if self.p_fsr_w is None or self.fsr_m is None:
            return None
        return self.fsr_m / self.p_fsr_w

    @property
    def p_drive_fs_w(self) -> float | None:
        """Ideal (cold-resistance) heater power at w = 1: V_fs^2/R0, I_fs^2 R0 or P_fs."""
        if self.heater_law is None:
            return None
        if self.heater_law == "voltage" and self.v_fs_v is not None and self.r_cold_ohm:
            return self.v_fs_v ** 2 / self.r_cold_ohm
        if self.heater_law == "current" and self.i_fs_a is not None and self.r_cold_ohm:
            return self.i_fs_a ** 2 * self.r_cold_ohm
        if self.heater_law in ("power", "dac_pwm") and self.p_fs_w is not None:
            return self.p_fs_w
        return None

    @property
    def tuning_m_per_w_drive(self) -> float | None:
        """Resonance shift per unit of the drive coordinate w near w = 0 (m), ideal law: tuning x P_drive_fs."""
        a, b = self.tuning_m_per_w, self.p_drive_fs_w
        return None if a is None or b is None else a * b

    @property
    def fwhm_w(self) -> float | None:
        """Linewidth in drive-coordinate units (first order, no droop)."""
        a, b = self.fwhm_m, self.tuning_m_per_w_drive
        return None if a is None or b is None else a / b


def drive_exponent(law: str | None, default: int = 2) -> int:
    """q of w = (code/code_max)^q: 2 for voltage/current drives, 1 for power/dac_pwm (module docstring)."""
    return _LAW_Q.get(law, default) if law is not None else default


def w_of_code(code, code_max: int, q: int):
    """Drive coordinate w = (code/code_max)^q (array or scalar)."""
    x = np.clip(np.asarray(code, dtype=np.float64) / float(code_max), 0.0, 1.0)
    return x ** q


def code_of_w(w, code_max: int, q: int, code_min: int = 0):
    """Nearest heater code for drive coordinate w (int64), clipped to [code_min, code_max]."""
    x = np.clip(np.asarray(w, dtype=np.float64), 0.0, 1.0) ** (1.0 / q)
    return np.clip(np.rint(x * code_max), code_min, code_max).astype(np.int64)


def heater_power_w(code, d: NominalDesign):
    """Nominal heater power (W) at ``code`` with the TCR droop (module docstring, section 2). Needs heater_nominal."""
    if d.heater_law is None or d.p_drive_fs_w is None:
        raise ValueError("heater_power_w needs the heater_nominal knowledge token")
    x = np.clip(np.asarray(code, dtype=np.float64) / d.code_max, 0.0, 1.0)
    y = d.y_per_w or 0.0
    if d.heater_law == "voltage":
        p0 = x ** 2 * d.p_drive_fs_w
        return p0 if y <= 0 else (np.sqrt(1.0 + 4.0 * y * p0) - 1.0) / (2.0 * y)
    if d.heater_law == "current":
        p0 = x ** 2 * d.p_drive_fs_w
        return p0 / np.maximum(1.0 - y * p0, 1e-3)
    return x * d.p_drive_fs_w


def heater_current_a(code, d: NominalDesign):
    """Nominal supply current (A) of one heater at ``code`` (rms for a power-linear drive)."""
    if not d.r_cold_ohm:
        raise ValueError("heater_current_a needs the heater_nominal knowledge token")
    p = heater_power_w(code, d)
    r = d.r_cold_ohm * (1.0 + (d.y_per_w or 0.0) * p)       # R(T_h) = R0 (1 + TCR R_th,H P) = R0 (1 + y P)
    x = np.clip(np.asarray(code, dtype=np.float64) / d.code_max, 0.0, 1.0)
    if d.heater_law == "current" and d.i_fs_a is not None:
        return x * d.i_fs_a
    return np.sqrt(np.maximum(p, 0.0) / r)


def _val(cfg, key, numeric=True):
    """Value of ``key`` in SI (numeric) or as stored. ``cfg`` is an mrm.params Config (``si``) or a plain mapping in
    config units (``{key: value}`` or ``{key: {value: ...}}``), converted here by the key's unit suffix
    (mrm.params.units.to_si, the single conversion of ARCHITECTURE section 1)."""
    if cfg is None:
        return None
    try:
        if numeric and callable(getattr(cfg, "si", None)):
            v = cfg.si(key, None)
        else:
            v = cfg.get(key, None)
            v = v.get("value") if isinstance(v, Mapping) and "value" in v else v
            if numeric and v is not None:
                from mrm.params.units import to_si
                v = to_si(key, float(v))
    except (KeyError, TypeError, ValueError):
        return None
    if v is None:
        return None
    if numeric:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return v


def _range_si(cfg, key):
    """The leaf's declared ``range`` (07 P1-14) of ``key`` in SI, or None: a Config leaf (``cfg.leaves[key].range``)
    or a plain mapping entry ``{key: {value, range}}``."""
    if cfg is None:
        return None
    rng = None
    try:
        leaves = getattr(cfg, "leaves", None)
        if isinstance(leaves, Mapping) and key in leaves:
            rng = getattr(leaves[key], "range", None)
        else:
            v = cfg.get(key, None)
            rng = v.get("range") if isinstance(v, Mapping) else None
        if rng is None or len(rng) != 2:
            return None
        from mrm.params.units import to_si
        lo, hi = (float(to_si(key, float(x))) for x in rng)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return (min(lo, hi), max(lo, hi)) if np.isfinite(lo) and np.isfinite(hi) else None


#: field group -> the knowledge tokens that release it: this module's tokens, and the finer tokens of the
#: controllers-A helpers (mrm.control.baselines.pi_lock.KNOWLEDGE_KEYS: heater_law, heater_r_nominal,
#: heater_efficiency, linewidth, dlambda_dt_nominal), so either vocabulary works until one is chosen
#: (docs/handoff/controllers_b.md section 2).
FIELD_TOKENS: dict[str, tuple[str, ...]] = {
    "heater_law": ("heater_nominal", "heater_law"),
    "r_cold": ("heater_nominal", "heater_law", "heater_r_nominal"),
    "heater_tcr": ("heater_nominal",),
    "p_fsr": ("heater_nominal", "heater_efficiency"),
    "supply": ("heater_nominal",),
    "q_loaded": ("optics_nominal", "linewidth"),
    "fsr": ("optics_nominal", "heater_efficiency"),
    "offset": ("optics_nominal",),
    "dlambda_dt": ("dlambda_dt_nominal",),
    "monitor": ("monitor_nominal",),
}


def _knows(spec, group: str) -> bool:
    know = set(getattr(spec, "knowledge", ()) or ())
    return any(t in know for t in FIELD_TOKENS[group])


def nominal_design(spec) -> NominalDesign:
    """The knowledge-filtered nominal design of ``spec`` (07 P1-1; module docstring section 1).

    Reads ``spec.nominal`` (a Config with ``si``/``get``, or a plain mapping in config units) only for the field
    groups whose tokens (:data:`FIELD_TOKENS`) are in ``spec.knowledge``. The heater code range of the DAC facts is
    always known."""
    cfg = getattr(spec, "nominal", None)
    code_max = int(spec.heater_dac.code_max)
    kw: dict[str, Any] = {"code_max": code_max}
    grid = getattr(spec, "line_grid_hz", None)
    if grid:
        kw["lambda0_m"] = C0 / float(np.mean(grid))
    if cfg is None:
        return NominalDesign(**kw)
    if _knows(spec, "heater_law"):
        law = _val(cfg, "ring.heater.driver_mode", numeric=False)
        kw.update(heater_law=law, drive_q=drive_exponent(law),
                  v_fs_v=_val(cfg, "ring.heater.full_scale_v"), i_fs_a=_val(cfg, "ring.heater.full_scale_a"),
                  p_fs_w=_val(cfg, "ring.heater.full_scale_w"))
    if _knows(spec, "r_cold"):
        kw.update(r_cold_ohm=_val(cfg, "ring.heater.r_cold_ohm"))
    if _knows(spec, "heater_tcr"):
        kw.update(tcr_per_k=_val(cfg, "ring.heater.tcr_per_k"),
                  heater_self_ratio=_val(cfg, "ring.thermal.heater_self_ratio"))
    if _knows(spec, "p_fsr"):
        kw.update(p_fsr_w=_val(cfg, "ring.heater.p_fsr_mw"))
    if _knows(spec, "supply"):
        kw.update(supply_power_max_w=_val(cfg, "array.heater_total_power_max_w"),
                  supply_current_max_a=_val(cfg, "array.heater_supply_current_max_a"))
    if _knows(spec, "q_loaded"):
        kw.update(q_loaded=_val(cfg, "ring.optics.q_loaded"))
    if _knows(spec, "fsr"):
        kw.update(fsr_hz=_val(cfg, "ring.optics.fsr_thz"), fsr_range_hz=_range_si(cfg, "ring.optics.fsr_thz"))
    if _knows(spec, "offset"):
        kw.update(offset_m=_val(cfg, "ring.optics.resonance_offset_nm"),
                  offset_ref_temp_k=_val(cfg, "ring.optics.resonance_offset_ref_temp_c"))
    if _knows(spec, "dlambda_dt"):
        kw.update(s0_m_per_k=_val(cfg, "ring.thermal.dlambda_dt_pm_per_k"),
                  beta_per_k=_val(cfg, "ring.thermal.dlambda_dt_beta_per_k"))
    if _knows(spec, "monitor"):
        kw.update(responsivity_a_per_w=_val(cfg, "ring.monitor.responsivity_a_per_w"),
                  adc_full_scale_a=_val(cfg, "ring.monitor.adc_full_scale_ua"))
    d = NominalDesign(**kw)
    # heater-resistor droop y = TCR * heater_self_ratio * Z, Z = FSR / (s0 P_FSR) (first-order R2 inversion)
    if (d.tcr_per_k is not None and d.heater_self_ratio is not None and d.p_fsr_w and d.fsr_m is not None
            and d.s0_m_per_k):
        z = d.fsr_m / (d.s0_m_per_k * d.p_fsr_w)
        d = NominalDesign(**{**d.__dict__, "y_per_w": d.tcr_per_k * d.heater_self_ratio * z})
    return d


# =====================================================================================================================
# Line labels from the resonance pattern of one sweep (cold start, scenario B5)
# =====================================================================================================================

def predicted_pattern(line_freq_hz, fsr_hz: float, first_line: int, k: int):
    """Positions (in FSR units, from 0) of the first ``k`` line crossings of a ring heated from the crossing of
    ``first_line``, with their line indices and relative orders.

    Heating lowers the resonance frequency; the resonance meets line m (order offset j) when it has moved by
    frac((f_first - f_m) / FSR) + j FSRs. Returns (v (k,), lines (k,), orders (k,))."""
    f = np.asarray(line_freq_hz, dtype=np.float64)
    n_l = f.size
    v = np.mod((f[first_line] - f) / float(fsr_hz), 1.0)
    reps = k // n_l + 2
    allv = (v[None, :] + np.arange(reps)[:, None]).ravel()
    lines = np.tile(np.arange(n_l), reps)
    orders = np.repeat(np.arange(reps), n_l)
    idx = np.argsort(allv, kind="stable")[:k]
    return allv[idx], lines[idx], orders[idx]


def _gmap(w, cs, gammas):
    """g(w) = w^gamma / (1 + c w^gamma) on the grid (gamma, c): (G*C, K) rows, and the (gamma, c) of each row."""
    rows, par = [], []
    for gm in gammas:
        wg = np.power(np.maximum(w, 0.0), gm)
        rows.append(wg[None, :] / (1.0 + cs[:, None] * wg[None, :]))
        par += [(float(gm), float(c)) for c in cs]
    return np.concatenate(rows, axis=0), par


def label_lines_by_pattern(w_events, line_freq_hz, *, fsr_range_hz=(1.3e12, 2.1e12), n_fsr: int = 321,
                           curvature_range=(-0.25, 0.75), n_curv: int = 21, exponents=(1.0,),
                           require_interior: bool = True, amb_min: float = 2.0) -> dict:
    """Label the resonances of a heater sweep with their laser lines from the pattern alone (module docstring 4).

    w_events: drive coordinates of the resonances passed in one up-sweep (any order; sorted here), at least L + 1
    of them for an unambiguous answer (one full FSR shows the grid's empty gap). Model: the resonance position is
    a + b * g_c(w) with g_c(w) = w / (1 + c w) (one curvature parameter absorbs the heater-resistor droop and the
    thermo-optic curvature of the drive-to-shift map; the prior c in [-0.25, 0.75] spans the droop y P_fs ~ 0.56 and
    a thermo-optic beta of a few 1e-3 /K, and excludes the strongly bent maps that alias window-A FSRs onto a
    window-B pattern); for every FSR on a grid over ``fsr_range_hz`` (:func:`fsr_prior_hz` gives the nominal
    window when the optics knowledge allows), every c and every first line, (a, b) are a least-squares fit of the
    observed g_c(w_j) to the predicted crossing positions. ``exponents``: drive-law exponents gamma tried as
    g(w) = w^gamma / (1 + c w^gamma) (the drive coordinate w = (code/code_max)^q assumes the heater law; a
    power-linear driver read with q = 2 gives P ~ w^(1/2), a voltage driver read with q = 1 gives P ~ w^2:
    :func:`law_exponents`).

    Returns dict(lines (K,), orders (K,) relative order of each crossing (0 = the first FSR), fsr_hz, curvature,
    exponent, offset, scale (drive per FSR in g units), rms (fit residual in FSR units), ambiguity (best residual of
    a different labelling / best residual), edge (the best FSR or curvature lies on the border of its search grid:
    the device is outside the prior, the labels are a guess), ok (rms < 0.03 and ambiguity > ``amb_min`` and K > L,
    and not edge when ``require_interior``)). amb_min = 2: the self-heating jump offset of each resonance depends on
    its line's power, which distorts the pattern by ~0.01 in w; with 1.5, about 1 % of synthetic maps so distorted
    (window A, B and both drive laws) came out confidently wrong, with 2.0 none (and ~80 % of the right labels
    kept); a confidently wrong label puts a ring on a foreign line while it reports a verified lock.
    """
    w = np.sort(np.asarray(w_events, dtype=np.float64).ravel())
    f = np.asarray(line_freq_hz, dtype=np.float64)
    k, n_l = w.size, f.size
    out = {"lines": np.full(k, -1), "orders": np.zeros(k, int), "fsr_hz": np.nan, "curvature": np.nan,
           "exponent": np.nan, "offset": np.nan, "scale": np.nan, "rms": np.inf, "ambiguity": 0.0, "edge": False,
           "ok": False}
    if k < 3:
        return out
    fs = np.linspace(fsr_range_hz[0], fsr_range_hz[1], n_fsr)
    cs = np.linspace(curvature_range[0], curvature_range[1], n_curv)
    x, par = _gmap(w, cs, exponents)                                     # (G*C, K)
    best = (np.inf, None)
    best_other = {}
    for fsr in fs:
        for m in range(n_l):
            v, ln, od = predicted_pattern(f, fsr, m, k)
            vm = v.mean()
            dv = v - vm
            svv = float(dv @ dv)
            if svv <= 0:
                continue
            xm = x.mean(axis=1, keepdims=True)
            slope = ((x - xm) @ dv) / svv                                  # (C,)
            inter = xm[:, 0] - slope * vm
            resid = x - (inter[:, None] + slope[:, None] * v[None, :])
            rms = np.sqrt(np.mean(resid ** 2, axis=1)) / np.where(slope > 0, slope, np.nan)
            rms = np.where(slope > 0, rms, np.inf)
            j = int(np.nanargmin(rms))
            r = float(rms[j])
            key = tuple(ln.tolist())
            if r < best_other.get(key, np.inf):
                best_other[key] = r
            if r < best[0]:
                best = (r, (fsr, par[j], m, ln, od, inter[j], slope[j]))
    if best[1] is None:
        return out
    r0, (fsr, (gm, c), m, ln, od, a, b) = best
    others = [v for kk, v in best_other.items() if kk != tuple(ln.tolist())]
    amb = (min(others) / max(r0, 1e-9)) if others else np.inf
    edge = _on_edge(fsr, c, fs, cs)
    out.update(lines=ln.astype(int), orders=od.astype(int), fsr_hz=float(fsr), curvature=float(c),
               exponent=float(gm), offset=float(a), scale=float(b), rms=float(r0), ambiguity=float(amb), edge=edge,
               ok=bool(r0 < 0.03 and amb > amb_min and k > n_l and not (require_interior and edge)))
    return out


def _on_edge(fsr, c, fs, cs) -> bool:
    """True when the best FSR or curvature sits on the first or last point of its search grid (a grid of one point
    has no edge)."""
    e = False
    if fs.size > 1:
        e |= bool(np.isclose(fsr, fs[0], rtol=0, atol=0.25 * (fs[1] - fs[0]))
                  or np.isclose(fsr, fs[-1], rtol=0, atol=0.25 * (fs[1] - fs[0])))
    if cs.size > 1:
        e |= bool(np.isclose(c, cs[0], rtol=0, atol=0.25 * (cs[1] - cs[0]))
                  or np.isclose(c, cs[-1], rtol=0, atol=0.25 * (cs[1] - cs[0])))
    return e


def law_exponents(design: "NominalDesign | None", q: int) -> tuple:
    """Drive-law exponents for the pattern fit (:func:`label_lines_by_pattern` ``exponents``): the heater law is a
    hypothesis of the uncertainty set (07 P1-11: voltage, current, power, dac_pwm), so a power-linear driver read in
    the q = 2 coordinate shows P ~ w^(1/2) and a voltage/current driver read with q = 1 shows P ~ w^2. Both are
    tried: (1, 1/2) for q = 2, (1, 2) for q = 1."""
    return (1.0, 0.5) if int(q) == 2 else (1.0, 2.0)


def drop_first(events: Sequence, n_lines: int) -> list[np.ndarray]:
    """The resonances to fit: every map without its lowest resonance when it has at least L + 2 (the first one is
    often a landing from a state captured at the sweep start, with a different self-heating offset)."""
    out = []
    for e in events:
        w = np.sort(np.asarray(e, dtype=np.float64).ravel())
        out.append(w[1:] if w.size >= n_lines + 2 else w)
    return out


def with_first(fit: dict, w_all, line_freq_hz) -> dict:
    """Extend a fit made without the lowest resonance (:func:`drop_first`) to every resonance of ``w_all``: the
    dropped one is the crossing just before the first fitted one in the cyclic line order."""
    w = np.sort(np.asarray(w_all, dtype=np.float64).ravel())
    k = len(fit["lines"])
    if k == w.size or k == 0 or not np.isfinite(fit.get("fsr_hz", np.nan)):
        return fit
    f = np.asarray(line_freq_hz, dtype=np.float64)
    _, cyc, _ = predicted_pattern(f, fit["fsr_hz"], int(fit["lines"][0]), f.size)
    out = dict(fit)
    out["lines"] = np.concatenate([[int(cyc[-1])], np.asarray(fit["lines"], int)])
    out["orders"] = np.concatenate([[0], np.asarray(fit["orders"], int) + 1])
    return out


def fsr_prior_hz(design: NominalDesign | None, rel: float = 0.2, default=(1.3e12, 2.1e12),
                 margin: float = 0.03) -> tuple:
    """FSR search window of the pattern fit. With the optics_nominal knowledge: the FSR leaf's declared range (it
    spans every FSR hypothesis of the uncertainty set, g26: 1.4-1.9 THz) widened by ``margin``, joined with the
    nominal FSR +- ``rel``; else ``default`` (every FSR hypothesis of g26: 1.47-1.88 THz with margin).

    The window must contain every FSR the device may have: a fit restricted to the nominal +- 15 % (1.53-2.07 THz)
    labels a window-A world (1.47 THz) with a wrong pattern AND reports it as confident (seen in closed loop on a
    sampled g26 world, tests/verification/test_ctrl_b_calibration.py::test_pattern_labels_window_a)."""
    if design is not None and design.fsr_hz:
        lo, hi = design.fsr_hz * (1.0 - rel), design.fsr_hz * (1.0 + rel)
        rng = getattr(design, "fsr_range_hz", None)
        if rng is not None:
            lo, hi = min(lo, rng[0] * (1.0 - margin)), max(hi, rng[1] * (1.0 + margin))
        return (lo, hi)
    return default


def label_bank_by_pattern(events: Sequence, line_freq_hz, *, fsr_range_hz=(1.3e12, 2.1e12), n_fsr: int = 321,
                          curvature_range=(-0.25, 0.75), n_curv: int = 21, exponents=(1.0,),
                          require_interior: bool = True, amb_min: float = 2.0) -> list[dict]:
    """Joint version of :func:`label_lines_by_pattern` for the rings of one bank: the FSR, the drive-law exponent and
    the curvature c are shared (one design, one heater family and driver), each ring has its own offset, scale and
    first line. The shared (FSR, exponent, c) minimizes the summed per-ring residuals; each ring then gets the labels
    of its best first line there and an ambiguity against its second-best labelling there. Rings with fewer than 3
    resonances are skipped (their dict has ok False); a shared fit on the border of the FSR or curvature grid
    (``edge``) makes every ring not ok when ``require_interior``. Returns one dict per ring, as
    label_lines_by_pattern."""
    f = np.asarray(line_freq_hz, dtype=np.float64)
    n_l = f.size
    ws = [np.sort(np.asarray(e, dtype=np.float64).ravel()) for e in events]
    fs = np.linspace(fsr_range_hz[0], fsr_range_hz[1], n_fsr)
    cs = np.linspace(curvature_range[0], curvature_range[1], n_curv)
    use = [i for i, w in enumerate(ws) if w.size >= 3]
    n_gc = len(tuple(exponents)) * n_curv
    tot = np.zeros((n_fsr, n_gc))
    per = {}
    par = None
    for i in use:
        w = ws[i]
        k = w.size
        x, par = _gmap(w, cs, exponents)                               # (G*C, K)
        xm = x.mean(axis=1, keepdims=True)
        r_all = np.full((n_fsr, n_gc, n_l), np.inf)
        for a, fsr in enumerate(fs):
            for m in range(n_l):
                v, _, _ = predicted_pattern(f, fsr, m, k)
                dv = v - v.mean()
                svv = float(dv @ dv)
                if svv <= 0:
                    continue
                slope = ((x - xm) @ dv) / svv
                inter = xm[:, 0] - slope * v.mean()
                resid = x - (inter[:, None] + slope[:, None] * v[None, :])
                rms = np.where(slope > 0, np.sqrt(np.mean(resid ** 2, axis=1)) / np.where(slope > 0, slope, 1.0),
                               np.inf)
                r_all[a, :, m] = rms
        tot += np.minimum(r_all.min(axis=2), 1.0) ** 2
        per[i] = r_all
    out = []
    if not use:
        return [label_lines_by_pattern(w, f, fsr_range_hz=fsr_range_hz, n_fsr=n_fsr,
                                       curvature_range=curvature_range, n_curv=n_curv, exponents=exponents,
                                       require_interior=require_interior, amb_min=amb_min)
                if w.size else {"ok": False, "lines": np.zeros(0, int)} for w in ws]
    a, j = np.unravel_index(int(np.argmin(tot)), tot.shape)
    fsr, (gm, c) = fs[a], par[j]
    edge = _on_edge(fsr, c, fs, cs)
    for i, w in enumerate(ws):
        if i not in per:
            out.append({"lines": np.full(w.size, -1), "orders": np.zeros(w.size, int), "fsr_hz": float(fsr),
                        "curvature": float(c), "exponent": float(gm), "offset": np.nan, "scale": np.nan,
                        "rms": np.inf, "ambiguity": 0.0, "edge": edge, "ok": False})
            continue
        r = per[i][a, j]
        o = np.argsort(r)
        m, r0 = int(o[0]), float(r[o[0]])
        v, ln, od = predicted_pattern(f, fsr, m, w.size)
        wg = np.power(np.maximum(w, 0.0), gm)
        x = wg / (1.0 + c * wg)
        dv = v - v.mean()
        b = float(((x - x.mean()) @ dv) / (dv @ dv))
        amb = float(r[o[1]] / max(r0, 1e-9)) if n_l > 1 else np.inf
        out.append({"lines": ln.astype(int), "orders": od.astype(int), "fsr_hz": float(fsr), "curvature": float(c),
                    "exponent": float(gm), "offset": float(x.mean() - b * v.mean()), "scale": b, "rms": r0,
                    "ambiguity": amb, "edge": edge,
                    "ok": bool(r0 < 0.03 and amb > amb_min and w.size > n_l and not (require_interior and edge))})
    return out


# =====================================================================================================================
# Factory calibration (07 P1-1)
# =====================================================================================================================

@runtime_checkable
class CalBench(Protocol):
    """What :func:`factory_cal` needs from a test station: the controller spec, and the sensor/actuator access of
    mrm.plant.plant.Plant (``advance(inputs, n_ticks)``, ``sample_sensors() -> (sensors, new)`` at the last
    controller edge). The station's scenario must contain the procedure's laser line schedule
    (:meth:`FactoryProcedure.scenario`)."""
    spec: Any

    def advance(self, inputs: Mapping[str, Any] | None = None, n_ticks: int = 1) -> None: ...

    def sample_sensors(self) -> tuple[dict, dict]: ...


@dataclass(frozen=True)
class FactoryProcedure:
    """A simulated factory heater sweep (07 P1-1).

    ambient_c: station temperature (the scenario base temperature); settle_s: wait after the lasers are on;
    group_size: rings swept together (1, default: one ring at a time in bus order, so each ring's all-lines sweep
    starts from the history a warm-start acquisition sees; None: as many as the nominal supply limit allows at w_max,
    faster, but rings upstream in a group dim the lines of the rings behind them while they sweep);
    w_max: top of each sweep in drive units (one FSR plus a resonance is ~0.62-0.70 at g26 nominal; 0.85);
    dw: drive step per controller period (None: the acquisition default); n_events: resonances to record per sweep
    (None: L + 1); cool_s: heaters off
    between sweeps (the fast thermal poles settle; the ms tail is part of the measurement error);
    line_coding: one all-lines sweep plus ceil(log2 L) sweeps with the lines whose index has bit k set: a model-free
    check of the line labels, which come from the pattern fit of the all-lines sweep (factory_cal docstring);
    table_noise_codes: rms of an extra rounding/storage noise on the stored heater codes (the table is a
    measurement, 07 P1-1 "noisy").
    """
    ambient_c: float = 25.0
    settle_s: float = 300e-6
    group_size: int | None = 1
    w_max: float = 0.85
    dw: float | None = None
    n_events: int | None = None
    cool_s: float = 300e-6
    line_coding: bool = True
    table_noise_codes: float = 0.0
    seed: int = 0

    def line_sets(self, n_lines: int) -> list[np.ndarray]:
        """Enabled-line masks (L,) of the sweeps of one group: all lines, then one per bit of the line index."""
        sets = [np.ones(n_lines, bool)]
        if self.line_coding and n_lines > 1:
            nb = int(math.ceil(math.log2(n_lines)))
            for k in range(nb):
                sets.append(((np.arange(n_lines) >> k) & 1).astype(bool))
        return sets

    def plan(self, spec, design: NominalDesign | None = None) -> dict:
        """Timing of the procedure in controller periods: groups, sweep length, schedule of line sets."""
        d = nominal_design(spec) if design is None else design
        n_r, n_l = spec.n_rings, spec.n_lines
        dw = self.dw if self.dw is not None else ((d.fwhm_w / 3.5) if d.fwhm_w else 0.004)
        steps = int(math.ceil(self.w_max / dw)) + 3
        gs = self.group_size
        if gs is None:
            gs = 1
            if d.supply_power_max_w is not None and d.heater_law is not None:
                code = code_of_w(self.w_max, d.code_max, d.drive_q or 2)
                p1, i1 = float(heater_power_w(code, d)), float(heater_current_a(code, d))
                gs = int(max(1, min(n_r, math.floor(0.9 * min(d.supply_power_max_w / max(p1, 1e-12),
                                                              d.supply_current_max_a / max(i1, 1e-12))))))
        order = list(spec.bus_order)
        groups = [order[i::int(math.ceil(n_r / gs))] for i in range(int(math.ceil(n_r / gs)))]
        per = spec.period_ticks
        settle = int(math.ceil(self.settle_s / spec.tick_s / per))
        cool = int(math.ceil(self.cool_s / spec.tick_s / per))
        sets = self.line_sets(n_l)
        sched = []                          # (start period, end period, group index, line set index)
        t = settle
        # every group's all-lines sweep first (in bus order), then the line-coding sweeps: a ring's all-lines sweep
        # must start from the state of the laser turn-on, which line toggling would change
        for si in range(len(sets)):
            for gi in range(len(groups)):
                sched.append((t, t + steps, gi, si))
                t += steps + cool
        return {"dw": dw, "steps": steps, "groups": groups, "sets": sets, "schedule": sched, "settle": settle,
                "cool": cool, "total_periods": t, "period_ticks": per}

    def scenario(self, spec, design: NominalDesign | None = None):
        """The station scenario: ambient, and the laser line enables of :meth:`plan` (mrm.scenarios)."""
        from mrm.scenarios import generators as g     # scenario builders only; no plant import
        pl = self.plan(spec, design)
        dt = spec.tick_s * pl["period_ticks"]
        n_l = spec.n_lines
        t_k, vals = [0.0], [np.ones(n_l)]
        for (a, b, gi, si) in pl["schedule"]:
            m = pl["sets"][si].astype(float)
            t_k += [a * dt - 0.5 * dt, a * dt - 0.25 * dt]
            vals += [vals[-1], m]
            t_k += [b * dt, b * dt + 0.25 * dt]
            vals += [m, np.ones(n_l)]
        tv = np.array(t_k)
        v = np.array(vals).T                                  # (L, n)
        comps = (g.pwl("laser.line_enable", tv, v[None, :, :], absolute=True, label="factory_lines",
                       provenance="assumed", note="factory line coding (07 P1-1)"),)
        return g.Scenario(name="factory_cal", components=comps,
                          defaults={"thermal.t_base_k": self.ambient_c + 273.15},
                          description="simulated factory heater sweep with line coding")


@dataclass
class FactoryCalTable:
    """Per-ring factory table (measured, noisy; 07 P1-1). Arrays have shape (B, N, E) with E event slots (NaN or -1
    where empty): ``code`` (heater code at each resonance of the all-lines sweep: the up-sweep maximum),
    ``w`` (drive coordinate), ``line`` (line index from the line coding, -1 if undecoded), ``order`` (relative
    order from the pattern fit, 0 = first FSR seen), ``height``/``floor`` (monitor codes). Per (B, N): ``ok``
    (line coding and pattern fit agree), ``fsr_hz`` (fitted), ``scale`` (drive per FSR). ``t_cal_c`` (B,): die
    temperature read by the first temperature sensor during the sweeps (NaN without one); ``q``: drive exponent;
    ``code_max``; ``ambient_c``: station setting."""
    code: np.ndarray
    w: np.ndarray
    line: np.ndarray
    order: np.ndarray
    height: np.ndarray
    floor: np.ndarray
    ok: np.ndarray
    fsr_hz: np.ndarray
    scale: np.ndarray
    t_cal_c: np.ndarray
    q: int
    code_max: int
    ambient_c: float
    notes: list = field(default_factory=list)

    def events(self, b: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(w (K,), line (K,)) of the recorded resonances of ring n, sorted by drive."""
        sel = np.isfinite(self.w[b, n])
        w, ln = self.w[b, n][sel], self.line[b, n][sel]
        o = np.argsort(w)
        return w[o], ln[o]

    def predict_w(self, b: int, n: int, line: int, *, t_die_c: float | None = None,
                  design: NominalDesign | None = None, w_min: float = 0.0) -> float:
        """Drive of the lowest-order resonance of ``line`` for ring n at or above ``w_min`` (NaN if none), shifted for
        the die temperature: dw = -(t_die - t_cal) s0 / (dlambda/dw), with s0 and the tuning from the nominal design
        (dlambda_dt_nominal, heater_nominal, optics_nominal); no shift without them."""
        w, ln = self.events(b, n)
        shift = 0.0
        if (t_die_c is not None and design is not None and np.isfinite(self.t_cal_c[b]) and design.s0_m_per_k
                and design.tuning_m_per_w_drive):
            shift = -(t_die_c - float(self.t_cal_c[b])) * design.s0_m_per_k / design.tuning_m_per_w_drive
        cand = w[(ln == line)] + shift
        cand = cand[cand >= w_min]
        return float(cand.min()) if cand.size else float("nan")


def _temp_c(sensors, spec) -> float:
    if not spec.temp_sensors:
        return float("nan")
    ts = spec.temp_sensors[0]
    code = np.asarray(sensors[ts.channel], dtype=float)
    return code[:, 0] * ts.lsb_k + ts.offset_c


def _match_shifted(w_ref: np.ndarray, w_sub: np.ndarray, tol: float, max_shift: float) -> np.ndarray:
    """Which reference resonances have a partner in a line-coding sweep, after a common drive shift |d| <=
    max_shift (the ms thermal tail of the previous sweep warms the ring, so a later sweep finds every resonance at
    a slightly lower drive). The shift matches the most events; ties go to the smaller |d|. max_shift must stay
    below half the smallest resonance spacing, or a shift by one resonance aliases the labels."""
    if w_sub.size == 0 or w_ref.size == 0:
        return np.zeros(w_ref.size, bool)
    best = (-1, np.inf, 0.0)
    for d in np.linspace(-max_shift, max_shift, 41):
        dist = np.abs(w_ref[:, None] - (w_sub[None, :] + d)).min(axis=0)       # each sub event to its nearest ref
        ok = dist <= tol
        key = (int(ok.sum()), abs(d) + float(dist[ok].sum()) * 1e-3)
        if key[0] > best[0] or (key[0] == best[0] and key[1] < best[1]):
            best = (key[0], key[1], d)
    d = best[2]
    return np.array([np.any(np.abs(w_sub + d - x) <= tol) for x in w_ref])


def factory_cal(plant, procedure: FactoryProcedure | None = None) -> FactoryCalTable:
    """Run the simulated factory heater sweep (07 P1-1) on a test station and return the per-ring table.

    ``plant`` is a :class:`CalBench` whose scenario holds ``procedure.scenario(spec)``, or a callable
    ``bench_factory(scenario) -> CalBench`` (then the scenario is built here). Only sensor channels are read
    (block-mean monitor codes, the die temperature sensor, the applied heater codes); the station's Truth is never
    used. Each group of rings (one ring by default, in bus order) sweeps up from w = 0 in steps of ``dw`` per
    controller period and records every passed resonance (a rise of >= detect above the floor followed by a fall,
    the acquisition FSM's search rule, mrm.control.baselines.ramp_acquire); a swept ring is then parked between
    lines (the floor after its first resonance, approached from below: a ring left at code 0 can sit on a foreign
    line by self-heating and dim it for the rings behind it on the bus); rings not yet swept hold code 0. The
    all-lines sweeps give the positions and, through the joint pattern fit of the bank
    (:func:`label_bank_by_pattern` without each map's first resonance, :func:`drop_first`), the line labels and
    relative orders; the line-coding sweeps give each resonance's line index without a model (bit k present iff a
    resonance of that sweep lies within 0.3 of the smallest spacing after a common shift) and are reported against
    the labels (``notes``). Returns a :class:`FactoryCalTable` for all B worlds.
    """
    from mrm.control.baselines.ramp_acquire import RampFSM, RampParams, ST_HOLD, monitor_mean
    from mrm.control.api import Observation
    proc = FactoryProcedure() if procedure is None else procedure
    if callable(plant) and not isinstance(plant, CalBench):
        spec_probe = getattr(plant, "spec", None)
        if spec_probe is None:
            raise TypeError("factory_cal: a bench factory must carry the ControllerSpec as .spec")
        bench = plant(proc.scenario(spec_probe))
    else:
        bench = plant
    spec = bench.spec
    d = nominal_design(spec)
    pl = proc.plan(spec, d)
    B, N, L = spec.n_batch, spec.n_rings, spec.n_lines
    n_ev = proc.n_events if proc.n_events is not None else L + 1
    fsm = RampFSM(spec, RampParams(dw_up=pl["dw"], start_delay_s=0.0), d)
    fsm.state[:] = ST_HOLD                      # every ring holds code 0 until its sweep
    E = 2 * L + 4
    ev_w = {}                                   # (group, set) -> (B, N, E) drive of the events
    park = np.zeros((B, N))                     # off-line park of each swept ring (floor after its resonance 0)
    per = pl["period_ticks"]
    codes = np.zeros((B, N), np.int64)
    t_cal = []
    y_prev = None
    sched = {a: (a, b, gi, si) for (a, b, gi, si) in pl["schedule"]}
    active = None
    k_period = 0
    while k_period < pl["total_periods"]:
        bench.advance({"tx.heater": codes}, per)
        k_period += 1
        sensors, new = bench.sample_sensors()
        obs = Observation(int(k_period * per), spec.tick_s, {c: sensors[c] for c in spec.sensors},
                          {c: new[c] for c in spec.sensors})
        y, valid, clip = monitor_mean(obs, y_prev)
        y_prev = y
        tc = _temp_c(sensors, spec)
        if np.all(np.isfinite(tc)) and np.any(tc != spec.temp_sensors[0].offset_c):
            t_cal.append(tc)
        if k_period in sched:
            a, b_end, gi, si = sched[k_period]
            mask = np.zeros((B, N), bool)
            mask[:, pl["groups"][gi]] = True
            fsm.start_map(mask, w_from=0.0, w_hi=proc.w_max, n_events=n_ev)
            active = (gi, si, b_end, mask)
        codes = fsm.step(y, valid, clip)
        if active is not None:
            gi, si, b_end, mask = active
            done = np.all((fsm.state[mask] == ST_HOLD)) or k_period >= b_end
            if done:
                ev_w[(gi, si)] = (fsm.ev_w.copy(), fsm.ev_y.copy(), fsm.ev_floor.copy())
                # park the swept rings between lines (approached from below, as the bank map of the WDM
                # supervisor): a ring left at code 0 can sit on a foreign line by self-heating and dim it for the
                # rings behind it on the bus
                if si == 0:
                    for b in range(B):
                        for n in pl["groups"][gi]:
                            fw = fsm.ev_wfloor[b, n]
                            park[b, n] = float(fw[1]) if fsm.ev_n[b, n] >= 2 and np.isfinite(fw[1]) else 0.0
                fsm.park_from_below(mask, park)
                active = None
    # ---- decode
    w0 = np.full((B, N, E), np.nan)
    h0 = np.full((B, N, E), np.nan)
    f0 = np.full((B, N, E), np.nan)
    line = np.full((B, N, E), -1, np.int64)
    order = np.zeros((B, N, E), np.int64)
    ok = np.zeros((B, N), bool)
    fsr = np.full((B, N), np.nan)
    scale = np.full((B, N), np.nan)
    rng = np.random.default_rng(proc.seed)
    grid = getattr(spec, "line_grid_hz", None)
    notes = []
    # joint pattern fit of every ring's all-lines sweep, per world (one FSR and curvature for the bank)
    bank = {}
    if grid is not None:
        for b in range(B):
            evs = []
            for n in range(N):
                gi = next(i for i, g in enumerate(pl["groups"]) if n in g)
                rec0 = ev_w.get((gi, 0))
                e = rec0[0][b, n] if rec0 is not None else np.zeros(0)
                evs.append(np.sort(e[np.isfinite(e)]))
            fits = label_bank_by_pattern(drop_first(evs, L), grid, fsr_range_hz=fsr_prior_hz(d),
                                         exponents=law_exponents(d, fsm.q))
            bank[b] = [with_first(fits[n], evs[n], grid) for n in range(N)]
    for gi, rings in enumerate(pl["groups"]):
        rec = ev_w.get((gi, 0))
        if rec is None:
            continue
        base, base_y, base_f = rec
        for b in range(B):
            for n in rings:
                sel = np.isfinite(base[b, n])
                o = np.argsort(base[b, n][sel])
                w_all = base[b, n][sel][o]
                k = w_all.size
                if k == 0:
                    continue
                h0[b, n, :k] = base_y[b, n][sel][o]
                f0[b, n, :k] = base_f[b, n][sel][o]
                if proc.table_noise_codes > 0:
                    cd = code_of_w(w_all, d.code_max, fsm.q) + rng.normal(0.0, proc.table_noise_codes, k)
                    w_all = np.sort(w_of_code(np.clip(cd, 0, d.code_max), d.code_max, fsm.q))
                w0[b, n, :k] = w_all
                sp = np.diff(w_all)
                tol = 0.3 * (sp.min() if sp.size else 0.05)
                lab = np.zeros(k, np.int64)
                for si in range(1, len(pl["sets"])):
                    e = ev_w.get((gi, si))
                    if e is None:
                        continue
                    we = e[0][b, n][np.isfinite(e[0][b, n])]
                    hit = _match_shifted(w_all, we, tol, 0.45 * (sp.min() if sp.size else 0.05))
                    lab |= hit.astype(np.int64) << (si - 1)
                lab = np.where(lab < L, lab, -1)
                line[b, n, :k] = lab if proc.line_coding else -1
                if grid is not None:
                    fit = bank[b][n]
                    order[b, n, :k] = fit["orders"]
                    fsr[b, n] = fit["fsr_hz"]
                    scale[b, n] = fit["scale"]
                    # the line coding is the model-free check of the pattern labels. It is fragile: the ms thermal
                    # tail of the previous sweep shifts a later sweep's resonances to lower drive (by up to about
                    # half a spacing at w ~ 0.5) and can leave the ring captured at w = 0 (a lost resonance decodes
                    # as line 0). A confident pattern fit over a full FSR (the empty gap) is therefore the label
                    # source; a disagreement is reported in the notes. Without a confident fit the coding labels
                    # are kept and the ring is not ok.
                    frac = float(np.mean(fit["lines"] == lab)) if proc.line_coding else 1.0
                    ok[b, n] = bool(fit["ok"])
                    if fit["ok"] or not proc.line_coding:
                        line[b, n, :k] = fit["lines"]
                    if proc.line_coding and frac < 1.0:
                        notes.append(f"world {b} ring {n}: line coding {lab.tolist()} vs pattern "
                                     f"{fit['lines'].tolist()} (agreement {frac:.2f}, fit rms {fit['rms']:.3g}, "
                                     f"ambiguity {fit['ambiguity']:.3g}) -> {'pattern' if ok[b, n] else 'unresolved'}")
                else:
                    ok[b, n] = proc.line_coding
    code = np.where(np.isfinite(w0), code_of_w(np.nan_to_num(w0), d.code_max, fsm.q), -1)
    tcal = np.nanmean(np.array(t_cal), axis=0) if t_cal else np.full(B, np.nan)
    return FactoryCalTable(code=code, w=w0, line=line, order=order, height=h0, floor=f0, ok=ok, fsr_hz=fsr,
                           scale=scale, t_cal_c=np.asarray(tcal, float).reshape(B), q=fsm.q, code_max=d.code_max,
                           ambient_c=proc.ambient_c, notes=notes)
