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
- ``next_order_distance_ghz`` and ``next_order_distance_fwhm`` (07 P1-6, BND-1; :func:`next_order_distances`):
  per ring, the smallest distance from any other resonance order of the ring, locked with its own order on its own
  line, to any other TX line, in GHz and in units of that order's FWHM. Computed with the plant's phase model
  (mrm.plant.phase: the cold phase curve with n_g and dn_g/dlambda, translated rigidly by the lock shift), so the
  comb of the heated ring and the dispersion are included; the FWHM is the TCMT linewidth at the order's wavelength
  (coupler dispersion included), lines at the nominal world's realized frequencies. This is the BND-1 replica's
  definition (mrm.validation.replicas.adjacent_order_penalty) without its restriction to orders inside the TX span
  +- 2 FWHM: the same value whenever the replica's is <= 2 FWHM. (Until the physics_optics review this diagnostic
  put every order at f_line + k FSR_cold and divided by the cold FWHM: 3.37 FWHM for g26 where the plant has 2.79.)
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

__all__ = ["Derived", "derive", "effective_provenance", "next_order_distances", "BISTABILITY_FACTOR",
           "DIAGNOSTIC_UNITS"]

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


def _cold_ring_optics(phs, b: int, n: int, lam: np.ndarray) -> dict:
    """Phase-model optics inputs of ring n of world b in its reference state (T_ref, reference bias, heater off) at
    the wavelengths ``lam`` (K,): phi, t_rt, K^2 of both couplers and a (mrm.plant.phase.ring_optics_row)."""
    from mrm.plant import phase as ph
    rp = np.ascontiguousarray(phs.ring[b, n][None, :])
    r = phs.table_row(b)
    lam = np.ascontiguousarray(lam, dtype=float)
    K = lam.size
    t = np.array([rp[0, ph.RP_T_REF]])
    v = np.array([rp[0, ph.RP_V_REF]])
    o = {k: np.zeros((1, K)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
    two = [np.zeros((1, 2)) for _ in range(3)]
    ph.ring_optics_row(rp, lam, t, t.copy(), v, v.copy(), phs.grid, phs.dneff[r], phs.alpha_fca[r],
                       phs.capacitance[r], phs.r_series[r], o["phi0"], o["phi1"], o["a0"], o["a1"], o["e0"], o["e1"],
                       o["k2t"], o["k2d"], o["trt"], *two, np.zeros(1, np.int8))
    return {k: o[k][0] for k in ("phi0", "trt", "k2t", "k2d", "a0")}


def next_order_distances(p: PlantParams, b: int = 0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Per ring of world b: the smallest distance (Hz, and in units of the order's FWHM) from any other resonance
    order of the ring to any other TX line, with the ring locked with its own order on its own line (07 P1-6, BND-1).

    The orders come from the plant's phase model (mrm.plant.phase equation 2): the order-(m + k) resonance x_k of the
    cold phase curve (Newton on phi(x) = 2 pi (m + k), d phi/dx = -2 pi c t_rt / x^2), moved by the lock shift
    lambda_line - lambda_res that puts order m on the ring's line (the rigid translation of the phase model). Its FWHM
    is the TCMT linewidth (kappa_i + kappa_e + kappa_d) / 2 pi (mrm.plant.optics.exact_to_tcmt) at the order's
    wavelength in the reference state, with K^2 at that wavelength (coupler dispersion, 07 P1-5). Lines: the realized
    frequencies of world b, direction 'tx' only. Orders k = +-1 ... +-(ceil(span / FSR) + 1). This is the definition
    of mrm.validation.replicas.adjacent_order_penalty, which also drops the orders outside the TX span +- 2 FWHM.
    Returns (dist_hz (N,), dist_fwhm (N,), rows: one dict per ring and order: ring, order_offset, lam_nm, f_thz,
    fwhm_ghz, distance_ghz, distance_fwhm); inf for a ring without another TX line."""
    from mrm.plant import optics as op
    from mrm.plant import phase as ph
    phs = ph.PhaseParams.from_plant_params(p)
    f_lines = np.asarray(p.laser.line_freq_hz, float)[b]
    L = f_lines.size
    dirs = tuple(p.laser.direction) if len(p.laser.direction) == L else ("tx",) * L
    tx = np.array([d == "tx" for d in dirs])
    r2l = np.asarray(p.layout.ring_to_line, dtype=np.int64)
    N = int(p.n_rings)
    fsr = np.broadcast_to(np.asarray(p.optics.fsr_hz, float)[b if np.ndim(p.optics.fsr_hz) > 1 else 0], (N,))
    span = float(np.ptp(f_lines[tx])) if np.any(tx) else 0.0
    d_hz, d_fwhm = np.full(N, np.inf), np.full(N, np.inf)
    rows: list[dict] = []
    for n in range(N):
        a = int(r2l[n])
        others = f_lines[tx & (np.arange(L) != a)]
        if others.size == 0:
            continue
        rp = phs.ring[b, n]
        lam_res = float(rp[ph.RP_LAM_RES])
        m_own = round(float(rp[ph.RP_N_EFF] * rp[ph.RP_LEN]) / lam_res)       # n_eff L = m lambda_res (R3)
        shift = C0 / f_lines[a] - lam_res                                       # the lock: order m on its line
        kmax = int(math.ceil(span / fsr[n])) + 1
        ks = np.array([k for k in range(-kmax, kmax + 1) if k != 0], dtype=float)
        target = 2.0 * math.pi * (m_own + ks)
        x = lam_res - ks * lam_res ** 2 * fsr[n] / C0                           # higher order = bluer
        for _ in range(50):
            c = _cold_ring_optics(phs, b, n, x)
            step = (c["phi0"] - target) * x * x / (2.0 * math.pi * C0 * c["trt"])
            x = x + step
            if np.max(np.abs(step)) < 1e-19:
                break
        else:
            raise RuntimeError(f"next_order_distances: the order solve of ring {n} did not converge")
        lam_o = x + shift
        c = _cold_ring_optics(phs, b, n, lam_o)
        ki, ke, kd = op.exact_to_tcmt(c["k2t"], c["k2d"], c["a0"], c["trt"])
        fwhm = (np.asarray(ki) + np.asarray(ke) + np.asarray(kd)) / (2.0 * math.pi)
        f_o = C0 / lam_o
        dist = np.min(np.abs(f_o[:, None] - others[None, :]), axis=1)
        d_hz[n] = float(dist.min())
        d_fwhm[n] = float(np.min(dist / fwhm))
        for j, k in enumerate(ks):
            rows.append({"ring": n, "order_offset": int(k), "lam_nm": float(lam_o[j] * 1e9),
                         "f_thz": float(f_o[j] / 1e12), "fwhm_ghz": float(fwhm[j] / 1e9),
                         "distance_ghz": float(dist[j] / 1e9), "distance_fwhm": float(dist[j] / fwhm[j])})
    return d_hz, d_fwhm, rows


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

    # ---- next order distance (07 P1-6, BND-1): the plant's phase model, not an ideal comb
    fsr = o.fsr_hz[0]
    dist_hz, dist_fwhm, _ = next_order_distances(p)
    diag["next_order_distance_ghz"] = dist_hz / 1e9
    diag["next_order_distance_fwhm"] = dist_fwhm
    inputs["next_order_distance_ghz"] = ["derived.optics.fsr_hz", rel.P_NG, "ring.optics.dn_g_dlambda_per_nm",
                                         rel.P_OFFSET, "channel_plan.spacing_ghz", "channel_plan.lines"]
    inputs["next_order_distance_fwhm"] = inputs["next_order_distance_ghz"] + [
        rel.P_QL, rel.P_XD, "ring.optics.dln_kappa2_through_dlambda_per_nm",
        "ring.optics.dln_kappa2_drop_dlambda_per_nm"]
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
