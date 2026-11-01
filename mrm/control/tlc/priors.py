"""TLC priors from the nominal G26 design (spec 2.3-3 and 2.3-4; Phase 3).

The TLC core models its plant with the ECE496 ``PlantParams`` (Lorentzian drop lineshape with an NRZ shift, a
two-pole heater, optical self-heating, a single-pole thermal crosstalk). On G26 those numbers must come from what
the controller may know: the nominal design released by the ``model_nominal`` token (``spec.nominal``), the same
tier as the model-based baselines (spec 6, Q7). Nothing here sees a sampled world or Truth.

How each prior is obtained (all from ``mrm.control.models`` on the nominal design):

- kappa (GHz/K): the resonance tuning slope at the OMA-optimal static point, converted with lambda^2 / c.
- mod_shift (GHz): half the resonance difference between the two NRZ drive levels at the operating point
  (``nrz_resonance_shift_m``; G26 nominal: about 5.7 GHz, 33 pm).
- hwhm (GHz) and the drop peak: a least-squares fit of the source's ``drop_avg`` lineshape (with that
  mod_shift) to the nominal static drop power over detuning (``oma_curve``), within 1.5 half widths of the peak
  (the lock region; the G26 tails sit above a Lorentzian, so a wider span biases the width). The absorbed-power
  peak comes from the same shape.
- laser power: the nominal input power at each ring (mW); abs_frac = absorbed peak / input power.
- PD scale: the source's DC level is p = laser_power * depth / pd_full_scale_mw with depth the source's dip
  depth; pd_full_scale_mw is set so that p equals the nominal drop-peak monitor level (codes per watt, 65535
  codes full scale), averaged over rings (the spread is in ``derivation``; B4's DC-scale state absorbs it).
- heater: full-scale power from the static point (voltage DAC: P = P_fs u); the ring's own heater residues split
  by pole into the source's fast and slow paths. The ~5 % package tail (ms) has no slot in the source model; it
  is added to the slow path and B10's efficiency estimate absorbs the difference.
- self-heating: the optical-source residues, split the same way.
- crosstalk: DC fractions of the nearest rings and a residue-weighted single time constant.
- unheated offset: the detuning the static point would have with no heater and no absorbed power.

Controller parameters for G26: n_rings from the spec, the 8-ring tone plan (tone_plan.py), the 250 kS/s monitor
(no decimation; FIR cutoff 60 kHz keeps every tone and 2nd harmonic within 0.1 % of unity gain, as the source's
150 kHz at 1 MS/s does), and, only when B6 finds no setpoint, the R13 ER floor dropped (``_r13_er_floor``);
everything else at the source defaults.
"""
from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from mrm.control import models as M
from mrm.control.api import ControllerSpec
from mrm.control.tlc.core.b6_setpoints import B6Setpoints
from mrm.control.tlc.core.b10_calibration import B10Calibration
from mrm.control.tlc.core.optics import drop_avg, er_db, inflection
from mrm.control.tlc.core.params import ControllerParams, PlantParams
from mrm.control.tlc.tone_plan import EIGHT_RING_BINS, search, to_hz
from mrm.control.tlc.units import pm_per_ghz

PD_RATE_HZ_FALLBACK = 250e3
FIR_CUTOFF_HZ = 60e3          # 250 kS/s: gain within 0.1 % of 1 at every G26 tone and 2nd harmonic (<= 27 kHz)
FAST_TAU_SPLIT_S = 8e-6       # residues with tau below this go to the fast path (G26 poles: 3.2 us, 13 us, ms)
FIT_SPAN_HWHM = 1.5           # lineshape fit over |delta| <= 1.5 half widths: the lock region (rms misfit 0.5 %)


@dataclass(frozen=True)
class TlcPriors:
    """The core's parameters for one ControllerSpec, and the derivation numbers for the report."""
    pp: PlantParams
    cp: ControllerParams
    derivation: dict


def g26_priors(spec: ControllerSpec) -> TlcPriors:
    """PlantParams / ControllerParams for the TLC core from ``spec.nominal`` (needs ``model_nominal``)."""
    M.check_knowledge(spec.knowledge, who="tlc")
    cfg = spec.nominal.model_config()
    pp_nom = M.nominal_plant_params(cfg)
    n = int(spec.n_rings)
    ppg = _ring_pm_per_ghz(spec)
    d_opt = M.oma_optimal_detuning(pp_nom, 0, cfg=cfg)[0]
    sp = M.static_point(pp_nom, 0, detuning_m=d_opt, cfg=cfg)

    kappa = np.asarray(sp.s_m_per_k) * 1e12 / ppg                       # GHz/K per ring
    cond = M.conditions_from_config(pp_nom, 0, cfg=cfg)
    sh = M.nrz_resonance_shift_m(pp_nom, 0, sp.t_ring_k, cond)
    mod_ghz = np.abs(sh[1] - sh[0]) / 2 * 1e12 / ppg
    shape = _fit_lineshape(pp_nom, cfg, ppg, mod_ghz)
    heater = _paths(pp_nom, "heater_src")
    selfh = _paths(pp_nom, "wg_src")
    xt_frac, tau_x = _crosstalk(pp_nom)
    u_op = (np.asarray(sp.heater_code) / spec.heater_dac.code_max) ** 2
    p_fs_w = float(np.mean(np.asarray(sp.heater_power_w) / u_op))
    depth = float(np.mean(PlantParams().depth))
    codes_per_w = float(np.mean(sp.codes_per_w))
    full_scale = float(spec.adc.full_scale_codes)
    d_op_ghz = np.asarray(d_opt) * 1e12 / ppg
    unheated = d_op_ghz - kappa * (heater["r_total"] * np.asarray(sp.heater_power_w)
                                   + selfh["r_total"] * np.asarray(sp.p_abs_w))
    laser_mw = np.asarray(sp.p_in_w) * 1e3
    peak_level = shape["drop_peak_w"] * codes_per_w / full_scale            # nominal drop peak, fraction of FS
    pd_fs_ring = laser_mw * depth / peak_level
    pp = dataclasses.replace(
        PlantParams(),
        kappa=float(np.mean(kappa)),
        channel_spacing=float(np.mean(np.diff(spec.line_grid_hz))) / 1e9 if spec.line_grid_hz else 400.0,
        fsr=float(spec.nominal.si("ring.optics.fsr_thz")) / 1e9,
        p_heater_max=p_fs_w,
        r_hf=heater["r_fast"], tau_hf=heater["tau_fast"], r_hs=heater["r_slow"], tau_hs=heater["tau_slow"],
        coupling=tuple(xt_frac), tau_x=tau_x,
        r_shf=selfh["r_fast"], tau_shf=selfh["tau_fast"], r_shs=selfh["r_slow"], tau_shs=selfh["tau_slow"],
        abs_frac=float(np.mean(shape["abs_peak_w"] / (laser_mw * 1e-3))),
        hwhm=tuple(shape["hwhm_ghz"]), depth=(depth,) * n, mod_shift=float(np.mean(mod_ghz)),
        laser_power=tuple(laser_mw),
        pd_rate=float(spec.adc.rate_hz) or PD_RATE_HZ_FALLBACK, pd_bits=int(spec.adc.bits),
        pd_full_scale_mw=float(np.mean(pd_fs_ring)),
        pd_noise_fs=float(np.mean(sp.y_sigma_codes)) / full_scale,
        unheated_offset=tuple(unheated), heater_eff_scale=(1.0,) * n,
    )
    cp = dataclasses.replace(ControllerParams(), n_rings=n, tone_freqs=_tones(n), decim=1,
                             fir_cutoff=FIR_CUTOFF_HZ)
    cp, r13 = _r13_er_floor(pp, cp)
    derivation = {"oma_optimal_detuning_pm": (np.asarray(d_opt) * 1e12).tolist(),
                  "kappa_ghz_per_k": kappa.tolist(), "heater_u_op": u_op.tolist(),
                  "lineshape_fit_rms_rel": shape["fit_rms_rel"].tolist(),
                  "mod_shift_ghz": mod_ghz.tolist(), "heater_paths": heater,
                  "selfheat_paths": selfh, "crosstalk_frac": list(xt_frac), "tau_x_s": tau_x,
                  "drop_peak_level_fs": peak_level.tolist(),
                  "pd_full_scale_spread_rel": float(np.ptp(pd_fs_ring) / np.mean(pd_fs_ring)),
                  "r13_er_floor": r13}
    return TlcPriors(pp=pp, cp=cp, derivation=derivation)


def _r13_er_floor(pp: PlantParams, cp: ControllerParams) -> tuple[ControllerParams, dict]:
    """B6 places each setpoint where ER >= er_min_db (R13, 3.5 dB) over +/- 2 GHz. With the G26 lineshape and NRZ
    swing no shift inside W-OP reaches it, and the source B6 then has no setpoint at all (it raises). R13 is an
    ECE496 link requirement, not plant knowledge, so on G26 the floor is dropped (0 dB) and B6 keeps its OMA
    balancing; the report carries the ER the G26 lineshape gives at the inflection point."""
    infl = [inflection(h, pp.mod_shift) for h in pp.hwhm]
    er_at = [float(er_db(x, h, d, pp.mod_shift)) for x, h, d in zip(infl, pp.hwhm, pp.depth)]
    info = {"source_db": cp.er_min_db, "er_at_inflection_db": er_at, "applied_db": cp.er_min_db}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # B6's all-NaN OMA map before it raises
            B6Setpoints(B10Calibration(pp, cp), cp)
    except ValueError:
        cp = dataclasses.replace(cp, er_min_db=0.0)
        info["applied_db"] = 0.0
    return cp, info


# -- pieces ---------------------------------------------------------------------------------------------------------
def _ring_pm_per_ghz(spec: ControllerSpec) -> np.ndarray:
    if spec.line_grid_hz is None:
        raise ValueError("tlc priors need the line_grid knowledge token")
    grid = np.asarray(spec.line_grid_hz, dtype=float)
    r2l = np.asarray(spec.ring_to_line if spec.ring_to_line is not None else np.arange(spec.n_rings))
    return pm_per_ghz(grid[r2l])


def _tones(n: int) -> tuple[float, ...]:
    return to_hz(EIGHT_RING_BINS) if n == len(EIGHT_RING_BINS) else to_hz(search(n))


def _fit_lineshape(pp_nom, cfg, ppg: np.ndarray, mod_ghz: np.ndarray) -> dict:
    """Fit peak * drop_avg(delta - offset; hwhm, mod_shift) to the nominal static drop power, per ring, with the
    physical mod_shift; the absorbed-power peak from the same shape."""
    d_m, _, _, p_drop, p_abs = M.oma_curve(pp_nom, 0, cfg=cfg)
    n = p_drop.shape[1]
    out = {k: np.zeros(n) for k in ("hwhm_ghz", "drop_peak_w", "abs_peak_w", "fit_rms_rel")}
    for i in range(n):
        d, y, mod = d_m * 1e12 / ppg[i], p_drop[:, i], float(mod_ghz[i])
        above = d[y >= y.max() / 2]
        hw0 = max(0.5 * (above.max() - above.min()), 1.0)
        keep = np.abs(d - d[np.argmax(y)]) <= FIT_SPAN_HWHM * hw0
        res = least_squares(lambda x: x[1] * drop_avg(d[keep] - x[2], x[0], mod) - y[keep],
                            x0=[hw0, y.max(), 0.0], bounds=([1.0, 0.0, -50.0], [500.0, 1.0, 50.0]))
        hw, peak, off = res.x
        g = drop_avg(d[keep] - off, hw, mod)
        out["hwhm_ghz"][i], out["drop_peak_w"][i] = hw, peak
        out["abs_peak_w"][i] = float(np.dot(g, p_abs[keep, i]) / np.dot(g, g))
        out["fit_rms_rel"][i] = float(np.sqrt(np.mean(res.fun ** 2)) / y.max())
    return out


def _lags(pp_nom, obs: int, src: int):
    sysm = pp_nom.thermal.system
    m = [k for k in range(sysm.n_lags) if int(sysm.lag_obs[k]) == obs and int(sysm.lag_src[k]) == src]
    return np.asarray(sysm.taus_s[0, m]), np.asarray(sysm.res_k_per_w[0, m])


def _paths(pp_nom, source: str, ring: int = 0) -> dict:
    """Ring ``ring``'s own path from ``source`` (heater_src | wg_src), split into the source's fast and slow paths."""
    sysm = pp_nom.thermal.system
    tau, res = _lags(pp_nom, int(sysm.ring_obs[ring]), int(getattr(sysm, source)[ring]))
    fast = tau < FAST_TAU_SPLIT_S
    r_f, r_s = float(res[fast].sum()), float(res[~fast].sum())
    t_f = float(np.dot(res[fast], tau[fast]) / r_f) if r_f else 1e-6
    slow_core = ~fast & (tau < 1e-3)                    # the slow ring pole; the ms package tail joins its gain only
    t_s = float(np.dot(res[slow_core], tau[slow_core]) / res[slow_core].sum()) if slow_core.any() else 1e-5
    return {"r_fast": r_f, "tau_fast": t_f, "r_slow": r_s, "tau_slow": t_s, "r_total": r_f + r_s,
            "tail_share": float(res[tau >= 1e-3].sum() / (r_f + r_s))}


def _crosstalk(pp_nom, ring: int = 0, max_distance: int = 3) -> tuple[tuple[float, ...], float]:
    """DC fraction of a heater's own-ring rise seen 1..max_distance rings away, and one residue-weighted tau."""
    sysm = pp_nom.thermal.system
    obs = int(sysm.ring_obs[ring])
    _, own = _lags(pp_nom, obs, int(sysm.heater_src[ring]))
    fracs, taus, weights = [], [], []
    for d in range(1, max_distance + 1):
        if ring + d >= int(sysm.n_rings):
            break
        tau, res = _lags(pp_nom, obs, int(sysm.heater_src[ring + d]))
        fracs.append(float(res.sum() / own.sum()))
        taus.append(float(np.dot(res, tau) / res.sum()) if res.sum() else 0.0)
        weights.append(float(res.sum()))
    tau_x = float(np.dot(weights, taus) / sum(weights)) if sum(weights) else 1e-3
    return tuple(fracs), tau_x
