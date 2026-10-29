# compute a nominal parameter view and physical diagnostics for inspection.
# `derive` calls `realize` for one world with nominal uncertainty and device variation off.
# the plant uses `realize` directly.
# these extra calculations support parameter tables, explanations, and checks of physical limits.
# diagnostics cover the material limit on resonance sensitivity, self-heating and bistability, lock and order-hop temperatures, nearby resonance orders, and combined gains that measurements can identify.
# diagnostic values use `DIAGNOSTIC_UNITS`.
# plant parameters remain in SI.
# each diagnostic records its input paths.
# its provenance is the weakest known input provenance.
# formulas and assumptions are beside the calculation.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mrm.params import relations
from mrm.params.realize import PlantParams, realize
from mrm.units import C0

if TYPE_CHECKING:  # pragma: no cover
    from mrm.params.config import Config

__all__ = ["Derived", "derive", "effective_provenance", "next_order_distances", "BISTABILITY_FACTOR",
           "DIAGNOSTIC_UNITS"]

# Lorentzian bistability threshold — G*P_abs,max / FWHM = 4*sqrt(3)/9 this dimensionless factor is about 0.770.
BISTABILITY_FACTOR = 4.0 * math.sqrt(3.0) / 9.0
# display unit of each diagnostic (values are stored in these units)
DIAGNOSTIC_UNITS = {
    "implied_index_weight": "", "index_weight_ceiling": "", "index_weight_ratio": "",
    "self_heating_gain_pm_per_mw": "pm/mW", "bistability_ratio": "", "p_abs_peak_mw": "mW",
    "p_in_ring_dbm": "dBm", "t_ring_lock_c": "degC", "t_amb_order_hop_c": "degC", "next_order_distance_ghz": "GHz",
    "next_order_distance_fwhm": "FWHM", "monitor_codes_per_uw_at_drop": "codes/uW", "heater_pm_per_code": "pm/code",
    "heater_code_grid": "code", "fwhm_pm": "pm", "disturbance_pm_per_k": "pm/K", "selfheat_pm_per_mw_in": "pm/mW",
    "heater_efficiency_nm_per_mw": "nm/mW", "z_heater_ring_k_per_mw": "K/mW", "fsr_nm": "nm", "fsr_thz": "THz",
    "radius_um": "um", "rom_dc_scale": "",
}


# result record and provenance

# nominal plant parameters plus diagnostics, provenance, and warnings.
# `params` has one world.
# diagnostic arrays usually have one value per ring, or shape `(codes, N)` for heater slopes.
# `inputs` lists the paths used by each diagnostic.
# values use the display units in `DIAGNOSTIC_UNITS`.
@dataclass
class Derived:
    params: PlantParams
    diagnostics: dict[str, Any]
    inputs: dict[str, list[str]]
    provenance: dict[str, str]
    warnings: list[str] = field(default_factory=list)


# follow a quantity's inputs and return their weakest provenance.
# use a given leaf's own provenance directly.
# trace free or derived values recursively.
# missing records or a dependency cycle fall back to assumed.
def effective_provenance(path: str, cfg: "Config", derivations: dict, _seen: set | None = None) -> str:
    from mrm.params.config import weakest
    stack = _seen if _seen is not None else frozenset()     # the recursion path (guards cycles only)
    if path in stack:
        return "assumed"
    stack = stack | {path}
    leaf = cfg.leaves.get(path)
    if leaf is not None and not leaf.is_free:
        return leaf.provenance
    derivation = derivations.get(path)
    if derivation is None:
        return "assumed" if leaf is None else leaf.provenance
    input_provenance = [effective_provenance(i, cfg, derivations, stack) for i in derivation.inputs
             if i in cfg.leaves or i in derivations]
    return weakest(input_provenance) if input_provenance else "assumed"


# derive

# realize the nominal device and compute its per-ring physical diagnostics.
# use the same relations and phase model as the plant.
# `thermal_hook` is passed to `realize`.
# return display-unit values, their input paths and provenance, and warnings when the selected anchors exceed diagnostic limits.
def derive(cfg: "Config", *, thermal_hook="auto") -> Derived:
    params = realize(cfg, 1, 0, uncertainty="nominal", variation="off", thermal_hook=thermal_hook)
    # the optics and thermal parameter blocks of the B = 1 PlantParams (index [0] is the one world)
    optics_params, thermal_params, values = params.optics, params.thermal, params.values
    line_wavelength_m = optics_params.lambda_line_m[0]
    n_rings = optics_params.n_g.shape[1]
    diagnostics: dict[str, Any] = {}
    inputs: dict[str, list[str]] = {}
    warnings: list[str] = []

    # can the stated resonance slope fit the material limit?
    # subtract length expansion and the oxide contribution to infer dn_eff/dn_Si.
    # compare with n_g/n_g,Si, the limit with all electric energy in silicon.
    from mrm import materials
    t0 = float(thermal_params.dlambda_dt_t0_k[0, 0])
    silicon_thermo_optic = np.asarray(materials.dn_dt_si_per_k(line_wavelength_m, t0), float)
    silicon_group_index = np.asarray(materials.n_g_si(line_wavelength_m, t0), float)
    oxide_thermo_optic = float(materials.dn_dt_sio2_per_k(t0))
    oxide_index_weight = np.broadcast_to(np.asarray(values.get("ring.optics.index_weight_oxide", 0.0), float), (1,
        n_rings))[0]
    s0 = thermal_params.dlambda_dt_s0_m_per_k[0]
    gamma_si = (s0 * optics_params.n_g[0] / line_wavelength_m - optics_params.n_eff[0] * optics_params.alpha_l_per_k[0, 0]
                - oxide_index_weight * oxide_thermo_optic) / silicon_thermo_optic
    ceiling = optics_params.n_g[0] / silicon_group_index
    diagnostics["implied_index_weight"] = gamma_si
    diagnostics["index_weight_ceiling"] = ceiling
    diagnostics["index_weight_ratio"] = gamma_si / ceiling
    inputs["implied_index_weight"] = [relations.P_S0, relations.P_NG, relations.P_NEFF, relations.P_ALPHA_L,
                                      relations.P_T0, "ring.optics.index_weight_oxide"]
    inputs["index_weight_ceiling"] = [relations.P_NG]
    inputs["index_weight_ratio"] = inputs["implied_index_weight"]
    if "ring.optics.index_weight_oxide" not in values:
        warnings.append("implied_index_weight: ring.optics.index_weight_oxide is absent; the oxide term (about 1 %) "
                     "is neglected")
    if np.any(diagnostics["index_weight_ratio"] > 1.0):
        warnings.append(f"dlambda/dT = {s0.max() * 1e12:.1f} pm/K needs an index weight above the materials ceiling "
                     f"(ratio {diagnostics['index_weight_ratio'].max():.3f}; docs/research/02_materials.md, section 3)")

    # is optical heating strong enough to permit bistability?
    # compare the peak heat-induced resonance shift with 0.770 times the linewidth.
    # a ratio above one permits multiple equilibria for some detunings.
    # it does not identify the branch occupied by a time-dependent simulation.
    omega = 2 * math.pi * C0 / optics_params.lambda_res_cold_m[0]
    # energy decay rates in 1/s — intrinsic loss, through coupling, and drop coupling
    k_i, k_t, k_d = (omega / optics_params.q_intrinsic[0], omega / optics_params.q_through[0],
                     omega / optics_params.q_drop[0])
    total_decay_rate = k_i + k_t + k_d
    layout, laser = params.layout, params.laser
    bus_position = np.argsort(layout.bus_order)          # bus position of each ring
    line_power_w = laser.power_w[0][layout.ring_to_line]
    ring_input_w = line_power_w * laser.input_transmission[0, 0] * np.exp(-laser.bus_loss_per_m[0, 0]
                                                              * layout.ring_spacing_on_bus_m * bus_position)
    abs_fraction = optics_params.heat_fraction[0] * 4.0 * k_t * k_i / (total_decay_rate * total_decay_rate)   # heat per input power, on resonance (-)
    peak_absorbed_w = abs_fraction * ring_input_w
    selfheat_gain_m_per_w = thermal_params.self_heating_gain_m_per_w[0]
    fwhm_m = optics_params.lambda_res_cold_m[0] / optics_params.q_loaded[0]
    diagnostics["p_in_ring_dbm"] = 10.0 * np.log10(ring_input_w / 1e-3)
    diagnostics["p_abs_peak_mw"] = peak_absorbed_w * 1e3
    diagnostics["self_heating_gain_pm_per_mw"] = selfheat_gain_m_per_w * 1e12 * 1e-3
    diagnostics["bistability_ratio"] = selfheat_gain_m_per_w * peak_absorbed_w / (BISTABILITY_FACTOR * fwhm_m)
    selfheat_inputs = [relations.P_EFF, "ring.thermal.z_ratio_opt_heater", relations.P_TLOCK, relations.P_S0,
        relations.P_BETA]
    absorption_inputs = [relations.P_QL, relations.P_QI, relations.P_QT, relations.P_QD, relations.P_ETA,
                "laser.power_per_line_dbm", "optical_path.chip_input_loss_db", "optical_path.bus_loss_db_per_mm"]
    inputs["self_heating_gain_pm_per_mw"] = selfheat_inputs
    inputs["p_abs_peak_mw"] = absorption_inputs
    inputs["p_in_ring_dbm"] = absorption_inputs[-3:]
    inputs["bistability_ratio"] = selfheat_inputs + absorption_inputs
    if np.any(diagnostics["bistability_ratio"] > 1.0) and params.flags.self_heating:
        warnings.append(f"bistability_ratio up to {diagnostics['bistability_ratio'].max():.2f} > 1: the ring is thermally "
                     f"bistable at this power; acquisition must approach from the correct side")

    # how much temperature margin remains before the heater-reachable order changes?
    # integrate resonance sensitivity to estimate the ambient threshold at peak absorption and reference bias.
    # BND-2 uses this order-hop margin.
    beta = thermal_params.dlambda_dt_beta_per_k[0, 0]
    t_ref = optics_params.t_ref_k[0, 0]
    diagnostics["t_ring_lock_c"] = thermal_params.t_ring_lock_k[0] - 273.15
    inputs["t_ring_lock_c"] = [relations.P_TLOCK]
    shift = values["derived.optics.lock_offset_m"][0] - selfheat_gain_m_per_w * peak_absorbed_w      # the heater-reachable order (R2)
    try:
        t_hop = relations.solve_t_for_shift(t_ref, shift, t0, s0, beta)
    except ValueError:
        t_hop = np.full(n_rings, np.nan)
    diagnostics["t_amb_order_hop_c"] = t_hop - 273.15
    inputs["t_amb_order_hop_c"] = [relations.P_OFFSET, relations.P_OFFSET_T] + selfheat_inputs + absorption_inputs

    # can another resonance order approach a different laser line?
    # use the plant phase model so heating and dispersion move the order locations.
    fsr = optics_params.fsr_hz[0]
    dist_hz, dist_fwhm, _ = next_order_distances(params)
    diagnostics["next_order_distance_ghz"] = dist_hz / 1e9
    diagnostics["next_order_distance_fwhm"] = dist_fwhm
    inputs["next_order_distance_ghz"] = ["derived.optics.fsr_hz", relations.P_NG, "ring.optics.dn_g_dlambda_per_nm",
                                         relations.P_OFFSET, "channel_plan.spacing_ghz", "channel_plan.lines"]
    inputs["next_order_distance_fwhm"] = inputs["next_order_distance_ghz"] + [
        relations.P_QL, relations.P_XD, "ring.optics.dln_kappa2_through_dlambda_per_nm",
        "ring.optics.dln_kappa2_drop_dlambda_per_nm"]
    if np.any(diagnostics["next_order_distance_fwhm"] < 2.0):
        warnings.append(f"next order within {diagnostics['next_order_distance_fwhm'].min():.2f} FWHM of another line "
                     f"(BND-1 wants >= 2 FWHM)")

    # which combined gains can measurements determine?
    # report ADC codes per optical power, resonance shift per heater code, linewidth, temperature sensitivity, and input-power sensitivity.
    monitor = params.monitor
    # the full-scale current reads code 2^bits - 1, so one LSB is full scale / (2^bits - 1), as in the plant (mrm.plant.sensing.MonitorHw.build)
    codes_per_a = ((1 << monitor.adc_bits) - 1) / monitor.adc_full_scale_a[0, 0]
    diagnostics["monitor_codes_per_uw_at_drop"] = (monitor.responsivity_a_per_w[0] * monitor.gain_scale[0,
        0] * codes_per_a
                                            * 1e-6)
    inputs["monitor_codes_per_uw_at_drop"] = ["ring.monitor.responsivity_a_per_w", "ring.monitor.gain_scale",
                                              "ring.monitor.adc_bits", "ring.monitor.adc_full_scale_ua"]
    heater = params.heater
    code_grid = np.unique(np.linspace(max(heater.code_min, 1), heater.code_max, 5).round().astype(np.int64))
    # use the local tuning slope at lock.
    # the fsr_integral efficiency averages sensitivity over a full FSR sweep, so it can differ when beta is nonzero.
    tuning_m_per_w = (thermal_params.heater_efficiency_m_per_w[0] if params.flags.heater_anchor == 0
                      else thermal_params.s_lock_m_per_k[0] * thermal_params.z_heater_ring_k_per_w[0])
    diagnostics["heater_code_grid"] = code_grid
    diagnostics["heater_pm_per_code"] = _code_power_slope_w(params, code_grid.astype(float)) * tuning_m_per_w[None,
        :] * 1e12
    inputs["heater_pm_per_code"] = [relations.P_EFF, "ring.heater.r_cold_ohm", "ring.heater.dac_bits",
                                    "ring.heater.full_scale_v", "ring.heater.full_scale_a", "ring.heater.full_scale_w"]
    diagnostics["fwhm_pm"] = fwhm_m * 1e12
    inputs["fwhm_pm"] = [relations.P_QL, relations.P_OFFSET]
    diagnostics["disturbance_pm_per_k"] = thermal_params.s_lock_m_per_k[0] * 1e12
    inputs["disturbance_pm_per_k"] = [relations.P_S0, relations.P_BETA, relations.P_TLOCK]
    diagnostics["selfheat_pm_per_mw_in"] = selfheat_gain_m_per_w * abs_fraction * 1e12 * 1e-3
    inputs["selfheat_pm_per_mw_in"] = selfheat_inputs + absorption_inputs[:5]

    # convenient anchors in display units
    diagnostics["heater_efficiency_nm_per_mw"] = thermal_params.heater_efficiency_m_per_w[0] * 1e9 * 1e-3
    diagnostics["z_heater_ring_k_per_mw"] = thermal_params.z_heater_ring_k_per_w[0] * 1e-3
    diagnostics["fsr_thz"] = fsr / 1e12
    diagnostics["fsr_nm"] = values["derived.optics.fsr_m"][0] * 1e9
    diagnostics["radius_um"] = optics_params.radius_m[0] * 1e6
    for kname, path in (("heater_efficiency_nm_per_mw", relations.P_EFF), ("z_heater_ring_k_per_mw", relations.P_Z),
                        ("fsr_thz", "derived.optics.fsr_hz"), ("fsr_nm", "derived.optics.fsr_m"),
                        ("radius_um", relations.P_RADIUS)):
        inputs[kname] = [path]
    if "rom_dc_scale" in params.diagnostics:
        diagnostics["rom_dc_scale"] = np.asarray(params.diagnostics["rom_dc_scale"])
        inputs["rom_dc_scale"] = [relations.P_Z, "ring.thermal.rom_file"]

    from mrm.params.config import weakest
    provenance = {}
    for name, diagnostic_inputs in inputs.items():
        input_provenance = [effective_provenance(i, cfg,
            params.derivations) for i in diagnostic_inputs if i in cfg.leaves or i in params.derivations]
        provenance[name] = weakest(input_provenance) if input_provenance else "assumed"
    return Derived(params=params, diagnostics=diagnostics, inputs=inputs, provenance=provenance, warnings=warnings)


# next-order distance (BND-1)

# find how close each locked ring's other resonances come to other TX lines.
# for world b, put the ring's own order on its assigned line.
# solve other cold resonances with Newton steps on the plant's phase model, then translate them by the lock shift.
# include group-index dispersion and coupling dispersion.
# use the coupled-mode linewidth at each order's wavelength.
# search both directions through `ceil(TX_span / FSR) + 1` neighboring orders.
# return per-ring minimum distances in Hz and linewidths, plus one detail row per ring and order.
# a ring with no other TX line has infinite distance.
# this supports the adjacent-order bound BND-1, which asks for at least two linewidths of separation.
# the validation replica uses the same definition but discards orders beyond the TX span plus two linewidths.
def next_order_distances(p: PlantParams, b: int = 0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    from mrm.plant import optics
    from mrm.plant import phase
    phase_params = phase.PhaseParams.from_plant_params(p)
    f_lines = np.asarray(p.laser.line_freq_hz, float)[b]
    n_lines = f_lines.size
    directions = tuple(p.laser.direction) if len(p.laser.direction) == n_lines else ("tx",) * n_lines
    tx_mask = np.array([d == "tx" for d in directions])
    ring_to_line = np.asarray(p.layout.ring_to_line, dtype=np.int64)
    n_rings = int(p.n_rings)
    fsr = np.broadcast_to(np.asarray(p.optics.fsr_hz, float)[b if np.ndim(p.optics.fsr_hz) > 1 else 0], (n_rings,))
    span = float(np.ptp(f_lines[tx_mask])) if np.any(tx_mask) else 0.0
    minimum_hz, minimum_linewidths = np.full(n_rings, np.inf), np.full(n_rings, np.inf)
    rows: list[dict] = []
    for n in range(n_rings):
        own_line = int(ring_to_line[n])
        other_tx_hz = f_lines[tx_mask & (np.arange(n_lines) != own_line)]
        if other_tx_hz.size == 0:
            continue
        ring_row = phase_params.ring[b, n]
        cold_resonance_m = float(ring_row[phase.RP_LAM_RES])
        own_order = round(float(ring_row[phase.RP_N_EFF] * ring_row[phase.RP_LEN]) / cold_resonance_m)   # n_eff L = m lambda_res
        shift = C0 / f_lines[own_line] - cold_resonance_m                                # the lock - order m on its line
        max_order_offset = int(math.ceil(span / fsr[n])) + 1
        order_offsets = np.array([k for k in range(-max_order_offset, max_order_offset + 1) if k != 0], dtype=float)
        target_phase_rad = 2.0 * math.pi * (own_order + order_offsets)
        cold_wavelength_m = cold_resonance_m - order_offsets * cold_resonance_m ** 2 * fsr[n] / C0                           # higher order = bluer
        for _ in range(50):
            cold = _cold_ring_optics(phase_params, b, n, cold_wavelength_m)
            step = (cold["phi0"] - target_phase_rad) * cold_wavelength_m * cold_wavelength_m / (2.0 * math.pi * C0 * cold["trt"])
            cold_wavelength_m = cold_wavelength_m + step
            if np.max(np.abs(step)) < 1e-19:
                break
        else:
            raise RuntimeError(f"next_order_distances: the order solve of ring {n} did not converge")
        order_wavelength_m = cold_wavelength_m + shift
        cold = _cold_ring_optics(phase_params, b, n, order_wavelength_m)
        ki, ke, kd = optics.exact_to_tcmt(cold["k2t"], cold["k2d"], cold["a0"], cold["trt"])
        fwhm = (np.asarray(ki) + np.asarray(ke) + np.asarray(kd)) / (2.0 * math.pi)
        order_frequency_hz = C0 / order_wavelength_m
        distance_hz = np.min(np.abs(order_frequency_hz[:, None] - other_tx_hz[None, :]), axis=1)
        minimum_hz[n] = float(distance_hz.min())
        minimum_linewidths[n] = float(np.min(distance_hz / fwhm))
        for j, k in enumerate(order_offsets):
            rows.append({"ring": n, "order_offset": int(k), "lam_nm": float(order_wavelength_m[j] * 1e9),
                         "f_thz": float(order_frequency_hz[j] / 1e12), "fwhm_ghz": float(fwhm[j] / 1e9),
                         "distance_ghz": float(distance_hz[j] / 1e9), "distance_fwhm": float(distance_hz[j] / fwhm[j])})
    return minimum_hz, minimum_linewidths, rows


# private helpers

# return ideal heater power sensitivity in W/code, shaped `(codes, N)`.
# use the nominal driver law and cold resistance without temperature dependence.
# the ideal DAC fraction is code divided by its maximum code.
# voltage and current drive give quadratic power.
# direct power drive is linear.
# PWM returns NaN because its switching law belongs to the electrical plant.
def _code_power_slope_w(params: PlantParams, codes: np.ndarray) -> np.ndarray:
    heater = params.heater
    maximum_code = float((1 << heater.dac_bits) - 1)
    resistance_ohm = heater.r_cold_ohm[0]                     # (N,)
    mode = int(heater.driver_mode[0, 0])
    code_column = codes[:, None]
    if mode == 0:     # voltage drive — P = (V_fs c/(2^bits - 1))^2/R
        return 2.0 * heater.full_scale_v[0, 0] ** 2 * code_column / (maximum_code * maximum_code * resistance_ohm[None, :])
    if mode == 1:     # current drive — P = (I_fs c/(2^bits - 1))^2 R
        return 2.0 * heater.full_scale_a[0, 0] ** 2 * code_column * resistance_ohm[None,
            :] / (maximum_code * maximum_code)
    if mode == 2:     # power drive — P = P_fs c/(2^bits - 1)
        return np.full((codes.size, resistance_ohm.size), heater.full_scale_w[0, 0] / maximum_code)
    return np.full((codes.size, resistance_ohm.size), np.nan)   # DAC PWM law is handled by mrm.plant.electrical


# evaluate one ring at its reference temperature and bias with the heater off.
# `lam` is a wavelength vector in m.
# return phase in rad, round-trip time in s, the two power-coupling fractions, and round-trip amplitude.
# allocate the extra outputs required by the shared phase kernel, then retain only these values.
def _cold_ring_optics(phase_params, b: int, n: int, lam: np.ndarray) -> dict:
    from mrm.plant import phase
    ring_row = np.ascontiguousarray(phase_params.ring[b, n][None, :])
    table_row = phase_params.table_row(b)
    lam = np.ascontiguousarray(lam, dtype=float)
    n_lam = lam.size
    reference_temp_k = np.array([ring_row[0, phase.RP_T_REF]])
    reference_bias_v = np.array([ring_row[0, phase.RP_V_REF]])
    out = {k: np.zeros((1, n_lam)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
    unused_outputs = [np.zeros((1, 2)) for _ in range(3)]
    phase.ring_optics_row(ring_row, lam, reference_temp_k, reference_temp_k.copy(), reference_bias_v,
        reference_bias_v.copy(), phase_params.grid, phase_params.dneff[table_row],
                          phase_params.alpha_fca[table_row], phase_params.capacitance[table_row],
                          phase_params.r_series[table_row], out["phi0"], out["phi1"], out["a0"], out["a1"],
                          out["e0"], out["e1"], out["k2t"], out["k2d"], out["trt"], *unused_outputs,
                          np.zeros(1, np.int8))
    return {k: out[k][0] for k in ("phi0", "trt", "k2t", "k2d", "a0")}
