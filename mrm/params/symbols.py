# name physical quantities in the variable map using LaTeX symbols.
# `SYMBOLS` maps config and derived paths to symbols.
# `PLANTPARAMS_SYMBOLS` maps parameter-group fields.
# runtime symbols live in `mrm/sim/state_spec.py`.
# equation text stays beside its calculation as `Derivation.formula_tex`.
# these labels do not affect the simulation.
# notation follows the lightmatter-capstone textbook, chapters 0 to 5.
# through and drop couplers use subscripts t and d instead of the textbook's 1 and 2.
# loaded Q uses subscript L.
# intrinsic Q uses int.
# H identifies the heater.
# round-trip attenuation a is a field amplitude.
# alpha is power attenuation.
# keep the detuning sign explicit.
# the textbook uses laser wavelength minus resonance wavelength.
# the model's runtime detuning has the opposite sign.
# the cold blue offset still means laser wavelength minus cold resonance.
# the phase-consistent effective index is distinct from the configured index.
# codes, flags, indices, and tables usually keep their field name.
# unrelated quantities that intentionally share a symbol must be listed in `INTENTIONAL_SHARED`.
# values connected by a map edge can share a symbol.
# `check_tex` checks the supported syntax for symbols and formulas.
# `to_unicode` makes a compact plain-text canvas label from that syntax.

from __future__ import annotations

import re

__all__ = ["SYMBOLS", "PLANTPARAMS_SYMBOLS", "INTENTIONAL_SHARED", "symbol", "check_tex", "to_unicode",
           "KNOWN_COMMANDS"]

# config path (preset leaf, setting or `derived.*` quantity) -> LaTeX symbol
SYMBOLS: dict[str, str] = {
    # array, channel plan, laser, path
    "array.n_rings": r"N",
    "array.heater_supply_current_max_a": r"I_{\mathrm{sup,max}}",
    "array.heater_total_power_max_w": r"P_{\mathrm{sup,max}}",
    "channel_plan.center_thz": r"f_{\mathrm{center}}",
    "channel_plan.spacing_ghz": r"\Delta f_{\mathrm{ch}}",
    "laser.linewidth_mhz": r"\Delta f_{\mathrm{laser}}",
    "laser.power_per_line_dbm": r"P_{L,0}",
    "laser.rin_db_per_hz": r"\mathrm{RIN}",
    "optical_path.bus_loss_db_per_mm": r"\alpha_{\mathrm{bus}}",
    # data path and driver
    "datapath.bitrate_gbps": r"R_b",
    "driver.energy_per_transition_fj": r"E_{\mathrm{tr}}",
    "driver.output_resistance_ohm": r"R_{\mathrm{out}}",
    "driver.static_power_mw": r"P_{\mathrm{drv,static}}",
    "driver.swing_pp_v": r"V_{\mathrm{pp}}",
    # geometry
    "ring.geometry.film_nm": r"h_{\mathrm{film}}",
    "ring.geometry.slab_nm": r"h_{\mathrm{slab}}",
    "ring.geometry.width_nm": r"w_{\mathrm{wg}}",
    "ring.geometry.radius_um": r"R",
    # optics
    "ring.optics.n_eff": r"n_{\mathrm{eff}}",
    "ring.optics.n_g": r"n_g",
    "ring.optics.fsr_thz": r"\Delta f_{\mathrm{FSR}}",
    "ring.optics.fsr_nm": r"\Delta\lambda_{\mathrm{FSR}}",
    "ring.optics.order": r"m",
    "ring.optics.q_loaded": r"Q_L",
    "ring.optics.q_intrinsic": r"Q_{\mathrm{int}}",
    "ring.optics.q_through": r"Q_t",
    "ring.optics.q_drop": r"Q_d",
    "ring.optics.kappa_d_over_kappa": r"x_d",
    "ring.optics.heat_fraction": r"\eta_{\mathrm{heat}}",
    "ring.optics.resonance_offset_nm": r"\delta_0",
    "ring.optics.resonance_offset_ref_bias_v": r"V_{\mathrm{ref}}",
    "ring.optics.resonance_offset_ref_temp_c": r"T_{\mathrm{ref}}",
    "ring.optics.alpha_l_per_k": r"\alpha_L",
    "ring.optics.alpha_l_slope_per_k2": r"\mathrm{d}\alpha_L/\mathrm{d}T",
    "ring.optics.dn_eff_dt_per_k": r"\partial n_{\mathrm{eff}}/\partial T",
    "ring.optics.dn_g_dlambda_per_nm": r"\mathrm{d}n_g/\mathrm{d}\lambda",
    "ring.optics.dln_kappa2_through_dlambda_per_nm": r"\mathrm{d}\ln\kappa_t^2/\mathrm{d}\lambda",
    "ring.optics.dln_kappa2_drop_dlambda_per_nm": r"\mathrm{d}\ln\kappa_d^2/\mathrm{d}\lambda",
    "ring.optics.dln_alpha_dt_heater_per_k": r"\partial\ln\alpha/\partial T_H",
    "ring.optics.dln_alpha_dt_ring_per_k": r"\partial\ln\alpha/\partial T_{\mathrm{ring}}",
    "ring.optics.backscatter_splitting_ghz": r"\Delta f_{\mathrm{split}}",
    "ring.optics.index_weight_oxide": r"1 - \Gamma",
    "ring.optics.loss_budget.alpha_fca_db_per_cm": r"\alpha_{\mathrm{FCA}}",
    "ring.optics.loss_budget.alpha_other_abs_db_per_cm": r"\alpha_{\mathrm{other}}",
    "ring.optics.loss_budget.alpha_scatter_db_per_cm": r"\alpha_{\mathrm{scat}}",
    "ring.optics.loss_budget.alpha_bend_radiation_db_per_cm": r"\alpha_{\mathrm{rad}}",
    # heater
    "ring.heater.p_fsr_mw": r"P_{\mathrm{FSR}}",
    "ring.heater.p_fsr_ref_temp_c": r"T_{\mathrm{meas}}",
    "ring.heater.p_fsr_ref_wavelength_nm": r"\lambda_{\mathrm{meas}}",
    "ring.heater.efficiency_nm_per_mw": r"\eta_{\mathrm{tune}}",
    "ring.heater.r_cold_ohm": r"R_{H,0}",
    "ring.heater.r_ref_temp_c": r"T_{H,0}",
    "ring.heater.tcr_per_k": r"\alpha_{\mathrm{TCR}}",
    "ring.heater.full_scale_a": r"I_{H,\mathrm{FS}}",
    "ring.heater.full_scale_v": r"V_{H,\mathrm{FS}}",
    "ring.heater.full_scale_w": r"P_{H,\mathrm{FS}}",
    "ring.heater.driver_compliance_v": r"V_{\mathrm{comp}}",
    # junction and bias DAC
    "ring.junction.capacitance_at_0v_ff": r"C_j(0\,\mathrm{V})",
    "ring.junction.depletion_exponent": r"m_j",
    "ring.junction.eff_at_0v_pm_per_v": r"\left.\mathrm{d}\lambda_r/\mathrm{d}V\right\vert_{0\,\mathrm{V}}",
    "ring.junction.series_resistance_ohm": r"R_s",
    "ring.junction.v_bi_v": r"V_{\mathrm{bi}}",
    "ring.bias_dac.v_max_v": r"V_{\mathrm{bias,max}}",
    "ring.bias_dac.v_min_v": r"V_{\mathrm{bias,min}}",
    # monitor photodiode and ADC
    "ring.monitor.responsivity_a_per_w": r"\mathcal{R}",
    "ring.monitor.dark_i0_na": r"I_{\mathrm{dark},0}",
    "ring.monitor.dark_t0_c": r"T_{\mathrm{dark},0}",
    "ring.monitor.dark_activation_ev": r"E_a",
    "ring.monitor.input_noise_pa_per_rt_hz": r"i_n",
    "ring.monitor.flicker_corner_hz": r"f_{\mathrm{corner}}",
    "ring.monitor.integration_window_s": r"T_{\mathrm{int}}",
    "ring.monitor.adc_full_scale_ua": r"I_{\mathrm{ADC,FS}}",
    "ring.monitor.adc_rate_hz": r"f_{\mathrm{ADC}}",
    # thermal
    "ring.thermal.dlambda_dt_pm_per_k": r"s_0",
    "ring.thermal.dlambda_dt_beta_per_k": r"\beta_T",
    "ring.thermal.dlambda_dt_ref_ring_temp_c": r"T_0",
    "ring.thermal.z_heater_ring_k_per_mw": r"R_{\mathrm{th}}",
    "ring.thermal.z_ratio_opt_heater": r"R_{\mathrm{th,opt}}/R_{\mathrm{th}}",
    "ring.thermal.heater_self_ratio": r"R_{\mathrm{th},H}/R_{\mathrm{th}}",
    "ring.thermal.t_ring_lock_c": r"T_{\mathrm{lock}}",
    "ring.thermal.tau_fast_s": r"\tau_{\mathrm{fast}}",
    "ring.thermal.tau_slow_s": r"\tau_{\mathrm{slow}}",
    "ring.thermal.tail_tau_s": r"\tau_{\mathrm{tail}}",
    "ring.thermal.selfheat_tau_s": r"\tau_{\mathrm{self}}",
    "ring.thermal.heater_node_tau_s": r"\tau_H",
    "ring.thermal.xt_tau_ref_s": r"\tau_{\mathrm{xt,ref}}",
    "ring.thermal.xt_tau_min_s": r"\tau_{\mathrm{xt,min}}",
    "ring.thermal.xt_dist_ref_um": r"d_{\mathrm{xt,ref}}",
    "ring.thermal.xt_nearest_frac": r"k_{i,i\pm 1}",
    "ring.thermal.xt_next_frac": r"k_{i,i\pm 2}",
    "ring.thermal.xt_floor_frac": r"k_{\mathrm{floor}}",
    # run settings
    "scenario.ambient_c": r"T_{\mathrm{amb}}",
    "scenario.bias_v": r"V_{\mathrm{bias}}",
    "sim.dt_plant_s": r"\Delta t",
    "controller_interface.update_rate_hz": r"f_{\mathrm{ctrl}}",
    # derived constants (relations R1-R5, realize)
    "derived.optics.length_m": r"L",
    "derived.optics.fsr_hz": r"\Delta f_{\mathrm{FSR}}",
    "derived.optics.fsr_m": r"\Delta\lambda_{\mathrm{FSR}}",
    "derived.optics.round_trip_time_s": r"\tau_{\mathrm{rt}}",
    "derived.optics.n_eff_phase": r"n_{\mathrm{eff,phase}}",
    "derived.optics.n_eff_correction": r"n_{\mathrm{eff,phase}} - n_{\mathrm{eff}}",
    "derived.optics.lambda_line_m": r"\lambda_L",
    "derived.optics.lambda_res_cold_m": r"\lambda_{r0}",
    "derived.optics.lock_offset_m": r"\delta_0 \bmod \Delta\lambda_{\mathrm{FSR}}",
    "derived.optics.alpha_i_per_m": r"\alpha_{\mathrm{int}}",
    "derived.optics.alpha_heat_per_m": r"\alpha_{\mathrm{heat}}",
    "derived.optics.alpha_nonheat_per_m": r"\alpha_{\mathrm{nonheat}}",
    "derived.optics.kappa2_through": r"\kappa_t^2",
    "derived.optics.kappa2_drop": r"\kappa_d^2",
    "derived.optics.round_trip_amplitude": r"a",
    "derived.optics.alpha_l_slope_eff_per_k2": r"\mathrm{d}\alpha_L/\mathrm{d}T",
    "derived.optics.d2n_eff_dt2_per_k2": r"\partial^2 n_{\mathrm{eff}}/\partial T^2",
    "derived.thermal.beta_eff_per_k": r"\beta_T",
    "derived.thermal.s_lock_m_per_k": r"s(T_{\mathrm{lock}})",
    "derived.thermal.dt_fsr_k": r"\Delta T_{\mathrm{FSR}}",
    "derived.heater.p_fsr_span_m": r"\Delta\lambda_{\mathrm{FSR,meas}}",
    "derived.geometry.dln_kappa2_through": r"\Delta\ln\kappa_t^2",
    "derived.geometry.dln_kappa2_drop": r"\Delta\ln\kappa_d^2",
}

# PlantParams field ("<group>.<field>", or the size "n_rings") -> LaTeX symbol (mrm.params.realize)
PLANTPARAMS_SYMBOLS: dict[str, str] = {
    "n_rings": r"N",
    # optics
    "optics.alpha_fca_per_m": r"\alpha_{\mathrm{FCA}}",
    "optics.alpha_heat_per_m": r"\alpha_{\mathrm{heat}}",
    "optics.alpha_i_per_m": r"\alpha_{\mathrm{int}}",
    "optics.alpha_l_per_k": r"\alpha_L",
    "optics.alpha_l_slope_per_k2": r"\mathrm{d}\alpha_L/\mathrm{d}T",
    "optics.backscatter_splitting_hz": r"\Delta f_{\mathrm{split}}",
    "optics.bias_ref_v": r"V_{\mathrm{ref}}",
    "optics.d2n_eff_dt2_per_k2": r"\partial^2 n_{\mathrm{eff}}/\partial T^2",
    "optics.dln_alpha_dt_heater_per_k": r"\partial\ln\alpha/\partial T_H",
    "optics.dln_alpha_dt_ring_per_k": r"\partial\ln\alpha/\partial T_{\mathrm{ring}}",
    "optics.dln_kappa2_drop_dlambda_per_m": r"\mathrm{d}\ln\kappa_d^2/\mathrm{d}\lambda",
    "optics.dln_kappa2_through_dlambda_per_m": r"\mathrm{d}\ln\kappa_t^2/\mathrm{d}\lambda",
    "optics.dn_eff_dt_per_k": r"\partial n_{\mathrm{eff}}/\partial T",
    "optics.dn_g_dlambda_per_m": r"\mathrm{d}n_g/\mathrm{d}\lambda",
    "optics.fsr_hz": r"\Delta f_{\mathrm{FSR}}",
    "optics.fwhm_hz": r"\Delta f_{\mathrm{FWHM}}",
    "optics.heat_fraction": r"\eta_{\mathrm{heat}}",
    "optics.kappa2_drop": r"\kappa_d^2",
    "optics.kappa2_through": r"\kappa_t^2",
    "optics.lambda_line_m": r"\lambda_L",
    "optics.lambda_res_cold_m": r"\lambda_{r0}",
    "optics.length_m": r"L",
    "optics.n_eff": r"n_{\mathrm{eff,phase}}",
    "optics.n_eff_configured": r"n_{\mathrm{eff}}",
    "optics.n_g": r"n_g",
    "optics.order": r"m",
    "optics.q_drop": r"Q_d",
    "optics.q_intrinsic": r"Q_{\mathrm{int}}",
    "optics.q_loaded": r"Q_L",
    "optics.q_through": r"Q_t",
    "optics.radius_m": r"R",
    "optics.resonance_offset_m": r"\delta_0",
    "optics.round_trip_amplitude": r"a",
    "optics.round_trip_time_s": r"\tau_{\mathrm{rt}}",
    "optics.t_ref_k": r"T_{\mathrm{ref}}",
    # thermal
    "thermal.dlambda_dt_beta_per_k": r"\beta_T",
    "thermal.dlambda_dt_s0_m_per_k": r"s_0",
    "thermal.dlambda_dt_t0_k": r"T_0",
    "thermal.heater_efficiency_m_per_w": r"\eta_{\mathrm{tune}}",
    "thermal.s_lock_m_per_k": r"s(T_{\mathrm{lock}})",
    "thermal.self_heating_gain_m_per_w": r"s(T_{\mathrm{lock}})\,R_{\mathrm{th,opt}}",
    "thermal.t_ring_lock_k": r"T_{\mathrm{lock}}",
    "thermal.z_heater_ring_k_per_w": r"R_{\mathrm{th}}",
    "thermal.z_opt_ring_k_per_w": r"R_{\mathrm{th,opt}}",
    "thermal.z_ratio_opt_heater": r"R_{\mathrm{th,opt}}/R_{\mathrm{th}}",
    # junction
    "junction.capacitance_at_0v_f": r"C_j(0\,\mathrm{V})",
    "junction.depletion_a_m": r"A_j",
    "junction.depletion_exponent": r"m_j",
    "junction.eff_at_0v_m_per_v": r"\left.\mathrm{d}\lambda_r/\mathrm{d}V\right\vert_{0\,\mathrm{V}}",
    "junction.series_resistance_ohm": r"R_s",
    "junction.v_bi_v": r"V_{\mathrm{bi}}",
    # heater
    "heater.compliance_v": r"V_{\mathrm{comp}}",
    "heater.full_scale_a": r"I_{H,\mathrm{FS}}",
    "heater.full_scale_v": r"V_{H,\mathrm{FS}}",
    "heater.full_scale_w": r"P_{H,\mathrm{FS}}",
    "heater.p_fsr_w": r"P_{\mathrm{FSR}}",
    "heater.r_cold_ohm": r"R_{H,0}",
    "heater.r_ref_temp_k": r"T_{H,0}",
    "heater.tcr_per_k": r"\alpha_{\mathrm{TCR}}",
    "heater.supply_current_max_a": r"I_{\mathrm{sup,max}}",
    "heater.supply_power_max_w": r"P_{\mathrm{sup,max}}",
    # bias DAC
    "bias_dac.v_max_v": r"V_{\mathrm{bias,max}}",
    "bias_dac.v_min_v": r"V_{\mathrm{bias,min}}",
    # monitor
    "monitor.adc_full_scale_a": r"I_{\mathrm{ADC,FS}}",
    "monitor.dark_activation_j": r"E_a",
    "monitor.dark_i0_a": r"I_{\mathrm{dark},0}",
    "monitor.dark_t0_k": r"T_{\mathrm{dark},0}",
    "monitor.flicker_corner_hz": r"f_{\mathrm{corner}}",
    "monitor.input_noise_a_per_rt_hz": r"i_n",
    "monitor.responsivity_a_per_w": r"\mathcal{R}",
    # data path and driver
    "datapath.bitrate_hz": r"R_b",
    "datapath.driver_energy_per_transition_j": r"E_{\mathrm{tr}}",
    "datapath.driver_output_resistance_ohm": r"R_{\mathrm{out}}",
    "datapath.driver_static_power_w": r"P_{\mathrm{drv,static}}",
    "datapath.rx_sensitivity_oma_w": r"\mathrm{OMA}_{\mathrm{sens}}",
    "datapath.swing_pp_v": r"V_{\mathrm{pp}}",
    # laser
    "laser.bus_loss_per_m": r"\alpha_{\mathrm{bus}}",
    "laser.line_freq_hz": r"f_{L,0}",
    "laser.linewidth_hz": r"\Delta f_{\mathrm{laser}}",
    "laser.power_w": r"P_{L,0}",
    "laser.rin_per_hz": r"\mathrm{RIN}",
    # timing
    "timing.dt_s": r"\Delta t",
}

# symbols that two quantities share on purpose although no edge of the map links them -> why that is not misleading
INTENTIONAL_SHARED: dict[str, str] = {
    r"m": "the longitudinal order: the preset/realize order and the order of the resonance nearest the line at run "
          "time (Truth order) are the same quantity",
    r"\lambda_L": "the line wavelength: nominal grid wavelength of each ring's line (preset, PlantParams) and the "
                  "actual line wavelength of each tick (S.line_float)",
    r"V_{\mathrm{bias}}": "the junction bias: the absolute bias of the scenario setting and the tick-average bias at "
                          "the junction (S.ring_float, Truth)",
    r"R_s": "the junction series resistance: the preset value (and its PlantParams field) and the run-time value at "
            "the two NRZ levels (S.ring_float)",
    r"R_{\mathrm{th,opt}}": "the DC thermal resistance from the absorbed light to the ring: the PlantParams value and "
                            "the kernel's value, which Plant._build takes from the thermal system",
}

# LaTeX check

# every LaTeX command a symbol or formula may use (all are KaTeX commands)
KNOWN_COMMANDS = frozenset("""
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi rho sigma tau upsilon
phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi Omega
mathrm mathcal mathbf text operatorname overline bar hat tilde dot
frac tfrac dfrac sqrt left right bigl bigr Bigl Bigr partial nabla int sum prod lim exp ln log max min sin cos tan
bmod pmod cdot times pm mp to approx equiv le ge leq geq neq propto infty ll gg quad qquad vert lvert rvert langle
rangle lfloor rfloor lceil rceil ldots cdots prime circ
""".split())
# commands that need a braced argument right after them -> how many braced arguments
_BRACED_ARG_COUNT = {"mathrm": 1, "mathcal": 1, "mathbf": 1, "text": 1, "operatorname": 1, "overline": 1, "bar": 1,
                     "hat": 1, "tilde": 1, "dot": 1, "frac": 2, "tfrac": 2, "dfrac": 2, "sqrt": 1}
# characters that may follow a backslash (spacing commands and escaped braces, underscore, percent)
_ESCAPED_CHARS = set(",;!: {}_%")


# return syntax problems found in a LaTeX label or formula.
# check braces, paired delimiters, allowed commands, required arguments, and script targets.
# reject table separators and math delimiters so callers can embed the result in Markdown safely.
# this checks a supported syntax subset without running the KaTeX renderer.
def check_tex(tex: str) -> list[str]:
    problems = []
    if not tex or not tex.strip():
        return ["empty"]
    for ch in "|$&#":
        if ch in tex:
            problems.append(f"character {ch!r} (use \\vert for a bar)")
    brace_depth, delimiter_depth, cursor, length = 0, 0, 0, len(tex)
    while cursor < length:
        character = tex[cursor]
        if character == "\\":
            match = re.match(r"[A-Za-z]+", tex[cursor + 1:])
            if match:
                command = match.group(0)
                cursor += 1 + len(command)
                if command not in KNOWN_COMMANDS:
                    problems.append(f"unknown command \\{command}")
                elif command == "left":
                    delimiter_depth += 1
                elif command == "right":
                    delimiter_depth -= 1
                if command in ("left", "right"):
                    if cursor >= length or not (tex[cursor] in "().[]" or tex[cursor:].startswith(("\\vert", "\\{",
                        "\\}", "\\langle",
                                                                                "\\rangle"))):
                        problems.append(f"\\{command} without a delimiter")
                if command in _BRACED_ARG_COUNT:
                    argument_cursor = cursor
                    for _k in range(_BRACED_ARG_COUNT[command]):
                        while argument_cursor < length and tex[argument_cursor] == " ":
                            argument_cursor += 1
                        if argument_cursor >= length or tex[argument_cursor] != "{":
                            problems.append(f"\\{command} needs a braced argument")
                            break
                        argument_depth, argument_cursor = 1, argument_cursor + 1
                        while argument_cursor < length and argument_depth:
                            argument_depth += {"{": 1, "}": -1}.get(tex[argument_cursor],
                                0) if tex[argument_cursor - 1] != "\\" else 0
                            argument_cursor += 1
                continue
            if cursor + 1 < length and tex[cursor + 1] in _ESCAPED_CHARS:
                cursor += 2
                continue
            problems.append(f"bad escape at {cursor}")
            cursor += 1
            continue
        if character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth -= 1
            if brace_depth < 0:
                problems.append("unbalanced '}'")
                brace_depth = 0
        elif character in "^_":
            argument_cursor = cursor + 1
            while argument_cursor < length and tex[argument_cursor] == " ":
                argument_cursor += 1
            if argument_cursor >= length or tex[argument_cursor] in "}^_":
                problems.append(f"{character} without a target")
        cursor += 1
    if brace_depth:
        problems.append("unbalanced '{'")
    if delimiter_depth:
        problems.append("unbalanced \\left/\\right")
    return problems


# Unicode label

# Greek commands mapped to Unicode glyphs
_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ", "varepsilon": "ε", "zeta": "ζ",
          "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
          "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "ϕ", "varphi": "φ",
          "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
          "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω"}
# other commands -> their plain-text glyph or word
_COMMAND_GLYPHS = {"partial": "∂", "nabla": "∇", "int": "∫", "sum": "Σ", "prod": "Π", "cdot": "·", "times": "×",
                   "pm": "±", "mp": "∓", "to": "→", "approx": "≈", "equiv": "≡", "le": "≤", "ge": "≥", "leq": "≤",
                   "geq": "≥", "neq": "≠", "propto": "∝", "infty": "∞", "vert": "|", "lvert": "|", "rvert": "|",
                   "langle": "⟨", "rangle": "⟩", "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉", "ldots": "…",
                   "cdots": "⋯", "prime": "′", "circ": "∘", "bmod": " mod ", "pmod": " mod ", "quad": "  ",
                   "qquad": "   ", "exp": "exp", "ln": "ln", "log": "log", "max": "max", "min": "min", "sin": "sin",
                   "cos": "cos", "tan": "tan", "lim": "lim"}
# characters that have a Unicode superscript form (digits, signs, brackets and n), for ^2, ^-1, ^n
_SUPERSCRIPTS = dict(zip("0123456789+-−=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁿ"))
# characters that have a Unicode subscript form (digits, signs and brackets), for _0 ...
# _9
_SUBSCRIPTS = dict(zip("0123456789+-−=()", "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎"))


# read a braced group, command, or character at s[i].
# skip leading spaces.
# return the argument without outer braces and the next index.
# use brace depth to retain nested groups.
# skip escaped characters.
def _tex_argument(s: str, i: int) -> tuple[str, int]:
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        brace_depth, end = 1, i + 1
        while end < len(s) and brace_depth:
            if s[end] == "\\":
                end += 2
                continue
            brace_depth += {"{": 1, "}": -1}.get(s[end], 0)
            end += 1
        return s[i + 1:end - 1], end
    if s[i] == "\\":
        match = re.match(r"\\([A-Za-z]+|.)", s[i:])
        return match.group(0), i + len(match.group(0))
    return s[i], i + 1


# convert a LaTeX symbol into a compact plain-text label for a canvas.
# use Greek glyphs, available Unicode scripts, and slash fractions.
# retain unavailable scripts after an underscore or caret.
# ignore math grouping and spacing, then normalize whitespace.
# this is a label formatter, not a renderer.
def to_unicode(tex: str) -> str:
    out, cursor, source = [], 0, tex
    while cursor < len(source):
        character = source[cursor]
        if character == "\\":
            match = re.match(r"\\([A-Za-z]+|.)", source[cursor:])
            command = match.group(1)
            cursor += len(match.group(0))
            if command in _GREEK:
                out.append(_GREEK[command])
            elif command in ("mathrm", "text", "operatorname", "mathbf"):
                argument, cursor = _tex_argument(source, cursor)
                out.append(to_unicode(argument))
            elif command == "mathcal":
                argument, cursor = _tex_argument(source, cursor)
                out.append({"R": "ℛ", "F": "ℱ", "L": "ℒ"}.get(argument, argument))
            elif command in ("overline", "bar"):
                argument, cursor = _tex_argument(source, cursor)
                out.append("⟨" + to_unicode(argument) + "⟩")
            elif command in ("hat", "tilde", "dot"):
                argument, cursor = _tex_argument(source, cursor)
                out.append(to_unicode(argument) + {"hat": "̂", "tilde": "̃", "dot": "̇"}[command])
            elif command in ("frac", "tfrac", "dfrac"):
                numerator, cursor = _tex_argument(source, cursor)
                denominator, cursor = _tex_argument(source, cursor)
                numerator, denominator = to_unicode(numerator), to_unicode(denominator)
                wrap = lambda x: f"({x})" if re.search(r"[ +\-−]", x) else x  # noqa: E731
                out.append(f"{wrap(numerator)}/{wrap(denominator)}")
            elif command == "sqrt":
                argument, cursor = _tex_argument(source, cursor)
                argument = to_unicode(argument)
                out.append("√" + (argument if len(argument) == 1 else f"({argument})"))
            elif command in ("left", "right", "bigl", "bigr", "Bigl", "Bigr"):
                if cursor < len(source) and source[cursor] == ".":
                    cursor += 1
            elif command in _COMMAND_GLYPHS:
                out.append(_COMMAND_GLYPHS[command])
            elif command in (",", ";", ":", " "):
                out.append(" " if command != "," else " ")
            elif command == "!":
                pass
            else:
                out.append(command)
        elif character in "^_":
            argument, cursor = _tex_argument(source, cursor + 1)
            text = to_unicode(argument)
            table = _SUPERSCRIPTS if character == "^" else _SUBSCRIPTS
            if text and all(ch in table for ch in text):
                out.append("".join(table[ch] for ch in text))
            else:
                out.append(("^" if character == "^" else "_") + text)
        elif character in "{} ":
            cursor += 1                                   # TeX spacing comes from commands
        elif character in "+-=":
            previous = "".join(out).rstrip()
            operator = {"-": "−"}.get(character, character)
            out.append(f" {operator} " if previous and previous[-1] not in "(,=±−+_^/" else operator)
            cursor += 1
        else:
            out.append(character)
            cursor += 1
    return re.sub(r"\s+", " ", "".join(out)).strip()


# the LaTeX symbol of a config path or `derived.*` quantity ("" when it has none).
def symbol(path: str) -> str:
    return SYMBOLS.get(path, "")
