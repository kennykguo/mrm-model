"""derive(cfg): the nominal B = 1 view of a config, plus the diagnostics of 07 P0-1, P0-3, P1-6 and P1-9.

``derive`` is documentation, not the plant path (07 P0-4): it calls ``realize(cfg, 1, uncertainty='nominal',
variation='off')`` and computes, per ring, from the realized SI values:

- ``implied_index_weight`` (07 P0-3): the dn_eff/dn_Si that the chosen s0 needs,
  Gamma_Si = (s0*n_g/lambda - n_eff*alpha_L - Gamma_ox*dn_ox/dT) / (dn_Si/dT), with dn/dT of Si and SiO2 at
  (lambda, T0) from :mod:`mrm.materials`; ``index_weight_ceiling`` = n_g/n_g,Si (all electric energy in Si,
  06 section 2.3); ``index_weight_ratio`` > 1 means the dlambda/dT hypothesis exceeds the materials ceiling.
- ``self_heating_gain_pm_per_mw`` (07 P0-1 item 5) = s(T_lock)*Z_opt->ring(0), which equals
  efficiency*Z_opt/Z_heater under the R2 form lock_slope (the default is fsr_integral since the integration).
- ``bistability_ratio`` (07 P0-1 item 5) = G_sh*P_abs,peak/((4*sqrt(3)/9)*FWHM_lambda), with
  P_abs,peak = eta*4*kappa_t*kappa_i/kappa^2*P_in at the ring (03 section 6.4). Above 1 the ring is bistable.
- ``t_ring_lock_c`` and ``t_amb_order_hop_c`` (07 P0-3, BND-2): the hop ambient solves
  int_{T_ref}^{T_hop} s dT = offset - G_sh*P_abs,peak (peak lock, reference bias, no lock offset).
- ``next_order_distance_ghz`` and ``next_order_distance_fwhm`` (07 P1-6, BND-1): distance from any other order of
  a locked ring to any other line.
- the ``identifiable`` block of 07 P1-9: ``monitor_codes_per_uw_at_drop``, ``heater_pm_per_code`` on a code grid,
  ``fwhm_pm``, ``disturbance_pm_per_k`` (s at lock) and ``selfheat_pm_per_mw_in`` (pm per mW at the ring input).

Every diagnostic records its inputs; its provenance is the weakest of theirs (07 P1-14).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mrm.params import relations as rel
from mrm.params.realize import PlantParams, realize
from mrm.units import C0

if TYPE_CHECKING:  # pragma: no cover
    from mrm.params.config import Config

__all__ = ["Derived", "derive", "effective_provenance", "BISTABILITY_FACTOR", "DIAGNOSTIC_UNITS"]

#: onset of Lorentzian thermal bistability: G*P_abs,max >= (4 sqrt(3)/9) FWHM (03 section 6.4; textbook Ch10 0.77)
BISTABILITY_FACTOR = 4.0 * math.sqrt(3.0) / 9.0
#: display unit of each diagnostic (values are stored in these units)
DIAGNOSTIC_UNITS = {
    "implied_index_weight": "", "index_weight_ceiling": "", "index_weight_ratio": "",
    "self_heating_gain_pm_per_mw": "pm/mW", "bistability_ratio": "", "p_abs_peak_mw": "mW",
    "p_in_ring_dbm": "dBm", "t_ring_lock_c": "degC", "t_amb_order_hop_c": "degC", "next_order_distance_ghz": "GHz",
    "next_order_distance_fwhm": "FWHM", "monitor_codes_per_uw_at_drop": "codes/uW", "heater_pm_per_code": "pm/code",
    "heater_code_grid": "code", "fwhm_pm": "pm", "disturbance_pm_per_k": "pm/K", "selfheat_pm_per_mw_in": "pm/mW",
    "heater_efficiency_nm_per_mw": "nm/mW", "z_heater_ring_k_per_mw": "K/mW", "fsr_nm": "nm", "fsr_thz": "THz",
    "radius_um": "um", "rom_dc_scale": "",
}


@dataclass
class Derived:
    """Nominal view: ``params`` (B = 1 PlantParams), ``diagnostics`` {name: (N,) array or scalar, display units},
    ``inputs`` {name: [config paths]} and ``provenance`` {name: weakest input provenance}; ``warnings``."""
    params: PlantParams
    diagnostics: dict[str, Any]
    inputs: dict[str, list[str]]
    provenance: dict[str, str]
    warnings: list[str] = field(default_factory=list)


def effective_provenance(path: str, cfg: "Config", derivations: dict, _seen: set | None = None) -> str:
    """Provenance of a config leaf, or the weakest input provenance of a derived quantity (recursively)."""
    from mrm.params.config import weakest
    stack = _seen if _seen is not None else frozenset()     # the recursion path (guards cycles only)
    if path in stack:
        return "assumed"
    stack = stack | {path}
    leaf = cfg.leaves.get(path)
    if leaf is not None and not leaf.is_free:
        return leaf.provenance
    d = derivations.get(path)
    if d is None:
        return "assumed" if leaf is None else leaf.provenance
    provs = [effective_provenance(i, cfg, derivations, stack) for i in d.inputs
             if i in cfg.leaves or i in derivations]
    return weakest(provs) if provs else "assumed"


def _code_power_slope_w(params: PlantParams, codes: np.ndarray) -> np.ndarray:
    """dP/dcode (W per code) at the given codes for the nominal driver law, at the cold resistance (no TCR).

    The ideal DAC transfer is code/(2^bits - 1) of full scale (mrm.plant.electrical.dac_frac with no bit errors)."""
    h = params.heater
    full = float((1 << h.dac_bits) - 1)
    r = h.r_cold_ohm[0]                     # (N,)
    mode = int(h.driver_mode[0, 0])
    c = codes[:, None]
    if mode == 0:     # voltage: P = (V_fs c/(2^bits - 1))^2/R
        return 2.0 * h.full_scale_v[0, 0] ** 2 * c / (full * full * r[None, :])
    if mode == 1:     # current: P = (I_fs c/(2^bits - 1))^2 R
        return 2.0 * h.full_scale_a[0, 0] ** 2 * c * r[None, :] / (full * full)
    if mode == 2:     # power: P = P_fs c/(2^bits - 1)
        return np.full((codes.size, r.size), h.full_scale_w[0, 0] / full)
    return np.full((codes.size, r.size), np.nan)   # dac_pwm: law owned by mrm.plant.electrical


def derive(cfg: "Config", *, thermal_hook="auto") -> Derived:
    """Nominal B = 1 view with diagnostics (see the module docstring). Raises like :func:`realize`."""
    p = realize(cfg, 1, 0, uncertainty="nominal", variation="off", thermal_hook=thermal_hook)
    o, t, v = p.optics, p.thermal, p.values
    lam = o.lambda_line_m[0]
    n = o.n_g.shape[1]
    diag: dict[str, Any] = {}
    inputs: dict[str, list[str]] = {}
    warns: list[str] = []

    # ---- implied index weight (07 P0-3)
    from mrm import materials as mat
    t0 = float(t.dlambda_dt_t0_k[0, 0])
    dn_si = np.asarray(mat.dn_dt_si_per_k(lam, t0), float)
    ng_si = np.asarray(mat.n_g_si(lam, t0), float)
    dn_ox = float(mat.dn_dt_sio2_per_k(t0))
    g_ox = np.broadcast_to(np.asarray(v.get("ring.optics.index_weight_oxide", 0.0), float), (1, n))[0]
    s0 = t.dlambda_dt_s0_m_per_k[0]
    gamma_si = (s0 * o.n_g[0] / lam - o.n_eff[0] * o.alpha_l_per_k[0, 0] - g_ox * dn_ox) / dn_si
    ceiling = o.n_g[0] / ng_si
    diag["implied_index_weight"] = gamma_si
    diag["index_weight_ceiling"] = ceiling
    diag["index_weight_ratio"] = gamma_si / ceiling
    inputs["implied_index_weight"] = [rel.P_S0, rel.P_NG, rel.P_NEFF, rel.P_ALPHA_L, rel.P_T0,
                                      "ring.optics.index_weight_oxide"]
    inputs["index_weight_ceiling"] = [rel.P_NG]
    inputs["index_weight_ratio"] = inputs["implied_index_weight"]
    if "ring.optics.index_weight_oxide" not in v:
        warns.append("implied_index_weight: ring.optics.index_weight_oxide is absent; the oxide term (about 1 %) "
                     "is neglected")
    if np.any(diag["index_weight_ratio"] > 1.0):
        warns.append(f"dlambda/dT = {s0.max() * 1e12:.1f} pm/K needs an index weight above the materials ceiling "
                     f"(ratio {diag['index_weight_ratio'].max():.3f}; 07 D2, 02 section 3)")

    # ---- self-heating gain and bistability (07 P0-1)
    w = 2 * math.pi * C0 / o.lambda_res_cold_m[0]
    k_i, k_t, k_d = w / o.q_intrinsic[0], w / o.q_through[0], w / o.q_drop[0]
    k = k_i + k_t + k_d
    lay, las = p.layout, p.laser
    pos = np.argsort(lay.bus_order)          # bus position of each ring
    p_line = las.power_w[0][lay.ring_to_line]
    p_ring = p_line * las.input_transmission[0, 0] * np.exp(-las.bus_loss_per_m[0, 0] * lay.ring_spacing_on_bus_m * pos)
    ext = o.heat_fraction[0] * 4.0 * k_t * k_i / (k * k)
    p_abs = ext * p_ring
    g_sh = t.self_heating_gain_m_per_w[0]
    fwhm_m = o.lambda_res_cold_m[0] / o.q_loaded[0]
    diag["p_in_ring_dbm"] = 10.0 * np.log10(p_ring / 1e-3)
    diag["p_abs_peak_mw"] = p_abs * 1e3
    diag["self_heating_gain_pm_per_mw"] = g_sh * 1e12 * 1e-3
    diag["bistability_ratio"] = g_sh * p_abs / (BISTABILITY_FACTOR * fwhm_m)
    ins_sh = [rel.P_EFF, "ring.thermal.z_ratio_opt_heater", rel.P_TLOCK, rel.P_S0, rel.P_BETA]
    ins_pabs = [rel.P_QL, rel.P_QI, rel.P_QT, rel.P_QD, rel.P_ETA, "laser.power_per_line_dbm",
                "optical_path.chip_input_loss_db", "optical_path.bus_loss_db_per_mm"]
    inputs["self_heating_gain_pm_per_mw"] = ins_sh
    inputs["p_abs_peak_mw"] = ins_pabs
    inputs["p_in_ring_dbm"] = ins_pabs[-3:]
    inputs["bistability_ratio"] = ins_sh + ins_pabs
    if np.any(diag["bistability_ratio"] > 1.0) and p.flags.self_heating:
        warns.append(f"bistability_ratio up to {diag['bistability_ratio'].max():.2f} > 1: the ring is thermally "
                     f"bistable at this power (07 P0-1); acquisition must approach from the correct side")

    # ---- lock and hop temperatures (07 P0-3, BND-2)
    beta = t.dlambda_dt_beta_per_k[0, 0]
    t_ref = o.t_ref_k[0, 0]
    diag["t_ring_lock_c"] = t.t_ring_lock_k[0] - 273.15
    inputs["t_ring_lock_c"] = [rel.P_TLOCK]
    shift = v["derived.optics.lock_offset_m"][0] - g_sh * p_abs      # the heater-reachable order (R2)
    try:
        t_hop = rel.solve_t_for_shift(t_ref, shift, t0, s0, beta)
    except ValueError:
        t_hop = np.full(n, np.nan)
    diag["t_amb_order_hop_c"] = t_hop - 273.15
    inputs["t_amb_order_hop_c"] = [rel.P_OFFSET, rel.P_OFFSET_T] + ins_sh + ins_pabs

    # ---- next order distance (07 P1-6, BND-1)
    f_lines = las.line_freq_grid_hz
    fsr = o.fsr_hz[0]
    dist = np.full(n, np.inf)
    span = float(f_lines.max() - f_lines.min()) if f_lines.size > 1 else 0.0
    for i in range(n):
        a = int(lay.ring_to_line[i])
        kmax = int(np.ceil(span / fsr[i])) + 1
        others = np.delete(f_lines, a)
        if others.size == 0:
            continue
        for kk in range(-kmax, kmax + 1):
            if kk == 0:
                continue
            dist[i] = min(dist[i], float(np.min(np.abs(f_lines[a] + kk * fsr[i] - others))))
    diag["next_order_distance_ghz"] = dist / 1e9
    diag["next_order_distance_fwhm"] = dist / o.fwhm_hz[0]
    inputs["next_order_distance_ghz"] = ["derived.optics.fsr_hz", "channel_plan.spacing_ghz", "channel_plan.lines"]
    inputs["next_order_distance_fwhm"] = inputs["next_order_distance_ghz"] + [rel.P_QL]
    if np.any(diag["next_order_distance_fwhm"] < 2.0):
        warns.append(f"next order within {diag['next_order_distance_fwhm'].min():.2f} FWHM of another line "
                     f"(BND-1 wants >= 2 FWHM)")

    # ---- identifiable block (07 P1-9)
    m = p.monitor
    codes_per_a = (2.0 ** m.adc_bits) / m.adc_full_scale_a[0, 0]
    diag["monitor_codes_per_uw_at_drop"] = m.responsivity_a_per_w[0] * m.gain_scale[0, 0] * codes_per_a * 1e-6
    inputs["monitor_codes_per_uw_at_drop"] = ["ring.monitor.responsivity_a_per_w", "ring.monitor.gain_scale",
                                              "ring.monitor.adc_bits", "ring.monitor.adc_full_scale_ua"]
    h = p.heater
    grid = np.unique(np.linspace(max(h.code_min, 1), h.code_max, 5).round().astype(np.int64))
    eff = t.heater_efficiency_m_per_w[0] if p.flags.heater_anchor == 0 else t.s_lock_m_per_k[0] * t.z_heater_ring_k_per_w[0]
    diag["heater_code_grid"] = grid
    diag["heater_pm_per_code"] = _code_power_slope_w(p, grid.astype(float)) * eff[None, :] * 1e12
    inputs["heater_pm_per_code"] = [rel.P_EFF, "ring.heater.r_cold_ohm", "ring.heater.dac_bits",
                                    "ring.heater.full_scale_v", "ring.heater.full_scale_a", "ring.heater.full_scale_w"]
    diag["fwhm_pm"] = fwhm_m * 1e12
    inputs["fwhm_pm"] = [rel.P_QL, rel.P_OFFSET]
    diag["disturbance_pm_per_k"] = t.s_lock_m_per_k[0] * 1e12
    inputs["disturbance_pm_per_k"] = [rel.P_S0, rel.P_BETA, rel.P_TLOCK]
    diag["selfheat_pm_per_mw_in"] = g_sh * ext * 1e12 * 1e-3
    inputs["selfheat_pm_per_mw_in"] = ins_sh + ins_pabs[:5]

    # ---- convenient anchors in display units
    diag["heater_efficiency_nm_per_mw"] = t.heater_efficiency_m_per_w[0] * 1e9 * 1e-3
    diag["z_heater_ring_k_per_mw"] = t.z_heater_ring_k_per_w[0] * 1e-3
    diag["fsr_thz"] = fsr / 1e12
    diag["fsr_nm"] = v["derived.optics.fsr_m"][0] * 1e9
    diag["radius_um"] = o.radius_m[0] * 1e6
    for kname, path in (("heater_efficiency_nm_per_mw", rel.P_EFF), ("z_heater_ring_k_per_mw", rel.P_Z),
                        ("fsr_thz", "derived.optics.fsr_hz"), ("fsr_nm", "derived.optics.fsr_m"),
                        ("radius_um", rel.P_RADIUS)):
        inputs[kname] = [path]
    if "rom_dc_scale" in p.diagnostics:
        diag["rom_dc_scale"] = np.asarray(p.diagnostics["rom_dc_scale"])
        inputs["rom_dc_scale"] = [rel.P_Z, "ring.thermal.rom_file"]

    from mrm.params.config import weakest
    prov = {}
    for name, ins in inputs.items():
        provs = [effective_provenance(i, cfg, p.derivations) for i in ins if i in cfg.leaves or i in p.derivations]
        prov[name] = weakest(provs) if provs else "assumed"
    return Derived(params=p, diagnostics=diag, inputs=inputs, provenance=prov, warnings=warns)
