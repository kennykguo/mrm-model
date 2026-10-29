# physics readability pass

completed on 29 october 2026.

reviewed 151 functions and methods, including nested helpers.

| module | functions | status |
|---|---|---|
| optics.py | 56 | typed, documented, and checked |
| phase.py | 15 | typed, documented, and checked |
| thermal.py | 80 | typed, documented, and checked |

the code uses lowercase purpose comments, one complete statement per line, and same-line physical type contracts.
each contract identifies the argument or return position, equation symbol, units, meaning, dtype, and shape or broadcast rule.
function and workspace names describe their actual role. stored configuration keys, record fields, and numeric layouts are unchanged.
previous function names remain aliases. phase column constants have descriptive names and retain their previous aliases.

the normalized executable AST matches the starting source in all three modules.
the comparison permits names and annotations to change while retaining calculations, literals, defaults, loop order, and decorators.
reader annotation markers are checked separately and preserved.

## validation

- 272 physics checks passed, with one expected failure.
- one legacy phase test still requests retired TCMT. the same failure was reproduced using the starting optics source.
- seven engine, compiled/reference parity, and self-heating integration checks passed.
- 17 representative checks also passed with JIT disabled.
- interface checks cover all 151 functions, including nested helpers and variable keyword arguments.
- detailed thermal assumptions and artifact conventions are retained in [thermal implementation notes](THERMAL_IMPLEMENTATION_NOTES.md).

## function inventory

| module | function | parameters |
|---|---|---|
| optics | [energy_decay_rates_from_ring](/Users/kenny/mrm-model/mrm/plant/optics.py:566) | 4 |
| optics | [tcmt_backscatter_rate](/Users/kenny/mrm-model/mrm/plant/optics.py:592) | 2 |
| optics | [exact_to_tcmt](/Users/kenny/mrm-model/mrm/plant/optics.py:605) | 5 |
| optics | [tcmt_to_exact](/Users/kenny/mrm-model/mrm/plant/optics.py:647) | 4 |
| optics | [backscatter_hz_to_round_trip_coupling](/Users/kenny/mrm-model/mrm/plant/optics.py:671) | 2 |
| optics | [phase_sensitivity_to_resonance_shift](/Users/kenny/mrm-model/mrm/plant/optics.py:685) | 2 |
| optics | [average_binary_data_levels](/Users/kenny/mrm-model/mrm/plant/optics.py:699) | 3 |
| optics | [_wrap_order_index](/Users/kenny/mrm-model/mrm/plant/optics.py:714) | 2 |
| optics | [interpolate_scatterer_reflection_and_phase_derivative](/Users/kenny/mrm-model/mrm/plant/optics.py:735) | 2 |
| optics | [_squared_magnitude](/Users/kenny/mrm-model/mrm/plant/optics.py:788) | 1 |
| optics | [_squared_magnitude_derivative](/Users/kenny/mrm-model/mrm/plant/optics.py:800) | 2 |
| optics | [calculate_quarter_ring_energy_per_power_and_loss_fraction](/Users/kenny/mrm-model/mrm/plant/optics.py:813) | 2 |
| optics | [solve_exact_ring_fields_and_energy](/Users/kenny/mrm-model/mrm/plant/optics.py:848) | 12 |
| optics | [solve_exact_ring_fields_and_phase_derivatives](/Users/kenny/mrm-model/mrm/plant/optics.py:1023) | 12 |
| optics | [_complex_reciprocal](/Users/kenny/mrm-model/mrm/plant/optics.py:1280) | 1 |
| optics | [calculate_propagation_and_scatterer_coefficients](/Users/kenny/mrm-model/mrm/plant/optics.py:1302) | 3 |
| optics | [make_workspace](/Users/kenny/mrm-model/mrm/plant/optics.py:1380) | 1 |
| optics | [make_bus_outputs](/Users/kenny/mrm-model/mrm/plant/optics.py:1395) | 2 |
| optics | [_validate_ring_workspace_buffers](/Users/kenny/mrm-model/mrm/plant/optics.py:1426) | 5 |
| optics | [solve_ring_fields_at_all_laser_wavelengths](/Users/kenny/mrm-model/mrm/plant/optics.py:1469) | 5 |
| optics | [solve_ring_fields_and_loss_phase_derivative_at_all_laser_wavelengths](/Users/kenny/mrm-model/mrm/plant/optics.py:1640) | 5 |
| optics | [evaluate_ring_response_per_input_watt](/Users/kenny/mrm-model/mrm/plant/optics.py:1996) | 13 |
| optics | [evaluate_ring_optical_outputs](/Users/kenny/mrm-model/mrm/plant/optics.py:2068) | 24 |
| optics | [evaluate_ring_absorbed_power_and_resonance_wavelength_derivative](/Users/kenny/mrm-model/mrm/plant/optics.py:2295) | 20 |
| optics | [bound_ring_absorbed_power_and_phase_curvature](/Users/kenny/mrm-model/mrm/plant/optics.py:2430) | 10 |
| optics | [bound_ring_absorbed_power_phase_derivative](/Users/kenny/mrm-model/mrm/plant/optics.py:2500) | 10 |
| optics | [write_bus_output_and_returned_reflection_powers](/Users/kenny/mrm-model/mrm/plant/optics.py:2579) | 6 |
| optics | [write_bus_output_powers](/Users/kenny/mrm-model/mrm/plant/optics.py:2596) | 6 |
| optics | [write_returned_reflection_power_and_forward_error_bound](/Users/kenny/mrm-model/mrm/plant/optics.py:2647) | 5 |
| optics | [propagate_bus_optical_power_into_output_buffers](/Users/kenny/mrm-model/mrm/plant/optics.py:2713) | 27 |
| optics | [unpack_self_heating_optical_inputs](/Users/kenny/mrm-model/mrm/plant/optics.py:2812) | 2 |
| optics | [unpack_self_heating_optical_outputs](/Users/kenny/mrm-model/mrm/plant/optics.py:2871) | 2 |
| optics | [unpack_self_heating_optical_state](/Users/kenny/mrm-model/mrm/plant/optics.py:2909) | 2 |
| optics | [_set_trial_temperature_phase](/Users/kenny/mrm-model/mrm/plant/optics.py:2993) | 7 |
| optics | [absorbed_power_at_trial_temperature](/Users/kenny/mrm-model/mrm/plant/optics.py:3044) | 4 |
| optics | [commit_optics_at_solved_temperature](/Users/kenny/mrm-model/mrm/plant/optics.py:3099) | 4 |
| optics | [bound_absorbed_power_temperature_response](/Users/kenny/mrm-model/mrm/plant/optics.py:3172) | 6 |
| optics | [bound_absorbed_power_temperature_response_from_ring_parameters](/Users/kenny/mrm-model/mrm/plant/optics.py:3192) | 6 |
| optics | [pass_through_power_to_next_ring](/Users/kenny/mrm-model/mrm/plant/optics.py:3280) | 4 |
| optics | [cache_matches_last_optical_evaluation](/Users/kenny/mrm-model/mrm/plant/optics.py:3346) | 4 |
| optics | [cache_clear_last_optical_evaluation](/Users/kenny/mrm-model/mrm/plant/optics.py:3367) | 2 |
| optics | [pack_self_heating_optical_state](/Users/kenny/mrm-model/mrm/plant/optics.py:3394) | 20 |
| optics | [evaluate_ring_response_for_input_samples](/Users/kenny/mrm-model/mrm/plant/optics.py:3560) | 14 |
| optics | [RingResponse.p_refl](/Users/kenny/mrm-model/mrm/plant/optics.py:3637) | 1 |
| optics | [RingResponse.p_thru](/Users/kenny/mrm-model/mrm/plant/optics.py:3646) | 1 |
| optics | [RingResponse.p_drop](/Users/kenny/mrm-model/mrm/plant/optics.py:3655) | 1 |
| optics | [RingResponse.p_add_out](/Users/kenny/mrm-model/mrm/plant/optics.py:3664) | 1 |
| optics | [RingResponse.p_in_total](/Users/kenny/mrm-model/mrm/plant/optics.py:3672) | 5 |
| optics | [_validate_backscatter_array](/Users/kenny/mrm-model/mrm/plant/optics.py:3691) | 1 |
| optics | [_evaluate_ring_response](/Users/kenny/mrm-model/mrm/plant/optics.py:3708) | 13 |
| optics | [evaluate_exact_ring_response](/Users/kenny/mrm-model/mrm/plant/optics.py:3821) | 11 |
| optics | [ring_scattering_matrix](/Users/kenny/mrm-model/mrm/plant/optics.py:3846) | 9 |
| optics | [_broadcast_finite_array](/Users/kenny/mrm-model/mrm/plant/optics.py:3929) | 4 |
| optics | [_validate_and_broadcast_bus_inputs](/Users/kenny/mrm-model/mrm/plant/optics.py:3949) | 13 |
| optics | [bus_propagate](/Users/kenny/mrm-model/mrm/plant/optics.py:4104) | 16 |
| optics | [absorbed_power_and_resonance_slope](/Users/kenny/mrm-model/mrm/plant/optics.py:4227) | 13 |
| phase | [PhaseParams.batch](/Users/kenny/mrm-model/mrm/plant/phase.py:122) | 1 |
| phase | [PhaseParams.table_row](/Users/kenny/mrm-model/mrm/plant/phase.py:130) | 2 |
| phase | [PhaseParams.from_plant_params](/Users/kenny/mrm-model/mrm/plant/phase.py:142) | 3 |
| phase | [PhaseParams.from_plant_params.per_ring](/Users/kenny/mrm-model/mrm/plant/phase.py:157) | 1 |
| phase | [calculate_bus_segment_power_transmission](/Users/kenny/mrm-model/mrm/plant/phase.py:213) | 1 |
| phase | [calculate_thermal_resonance_shift_m](/Users/kenny/mrm-model/mrm/plant/phase.py:240) | 5 |
| phase | [rebase_thermal_tuning_coefficients](/Users/kenny/mrm-model/mrm/plant/phase.py:258) | 4 |
| phase | [evaluate_ring_resonance_shifts](/Users/kenny/mrm-model/mrm/plant/phase.py:287) | 13 |
| phase | [evaluate_ring_resonance_shifts_with_workspace](/Users/kenny/mrm-model/mrm/plant/phase.py:311) | 14 |
| phase | [calculate_intrinsic_loss_temperature_factor](/Users/kenny/mrm-model/mrm/plant/phase.py:342) | 4 |
| phase | [write_ring_optical_inputs_for_data_level](/Users/kenny/mrm-model/mrm/plant/phase.py:388) | 11 |
| phase | [evaluate_binary_ring_optical_inputs](/Users/kenny/mrm-model/mrm/plant/phase.py:452) | 22 |
| phase | [evaluate_binary_ring_optical_inputs_with_workspace](/Users/kenny/mrm-model/mrm/plant/phase.py:491) | 23 |
| phase | [evaluate_coupler_power_fractions](/Users/kenny/mrm-model/mrm/plant/phase.py:550) | 4 |
| phase | [evaluate_ring_optical_inputs](/Users/kenny/mrm-model/mrm/plant/phase.py:582) | 24 |
| thermal | [calculate_linear_hold_lag_weights](/Users/kenny/mrm-model/mrm/plant/thermal.py:133) | 2 |
| thermal | [calculate_linear_hold_coefficients](/Users/kenny/mrm-model/mrm/plant/thermal.py:185) | 3 |
| thermal | [calculate_constant_hold_coefficients](/Users/kenny/mrm-model/mrm/plant/thermal.py:242) | 3 |
| thermal | [advance_thermal_state_for_world](/Users/kenny/mrm-model/mrm/plant/thermal.py:263) | 8 |
| thermal | [advance_thermal_state_for_world_with_step_size](/Users/kenny/mrm-model/mrm/plant/thermal.py:283) | 8 |
| thermal | [write_observer_temperatures_for_world](/Users/kenny/mrm-model/mrm/plant/thermal.py:306) | 4 |
| thermal | [shared_or_world_row_index](/Users/kenny/mrm-model/mrm/plant/thermal.py:323) | 2 |
| thermal | [validate_thermal_step_array_shapes](/Users/kenny/mrm-model/mrm/plant/thermal.py:336) | 8 |
| thermal | [advance_thermal_states](/Users/kenny/mrm-model/mrm/plant/thermal.py:362) | 8 |
| thermal | [write_observer_temperatures](/Users/kenny/mrm-model/mrm/plant/thermal.py:384) | 4 |
| thermal | [advance_thermal_state_and_temperatures_for_world](/Users/kenny/mrm-model/mrm/plant/thermal.py:403) | 11 |
| thermal | [advance_thermal_states_and_temperatures](/Users/kenny/mrm-model/mrm/plant/thermal.py:431) | 11 |
| thermal | [separate_known_heat_and_source_gain_for_observer](/Users/kenny/mrm-model/mrm/plant/thermal.py:464) | 11 |
| thermal | [sha256_file](/Users/kenny/mrm-model/mrm/plant/thermal.py:497) | 1 |
| thermal | [build_rom_sidecar_path](/Users/kenny/mrm-model/mrm/plant/thermal.py:511) | 1 |
| thermal | [find_rom_sidecar_path](/Users/kenny/mrm-model/mrm/plant/thermal.py:520) | 1 |
| thermal | [build_thermal_rom_path](/Users/kenny/mrm-model/mrm/plant/thermal.py:533) | 3 |
| thermal | [ThermalRom.__post_init__](/Users/kenny/mrm-model/mrm/plant/thermal.py:565) | 1 |
| thermal | [save_rom](/Users/kenny/mrm-model/mrm/plant/thermal.py:594) | 3 |
| thermal | [load_rom](/Users/kenny/mrm-model/mrm/plant/thermal.py:622) | 3 |
| thermal | [read_rom_validation_metrics](/Users/kenny/mrm-model/mrm/plant/thermal.py:672) | 1 |
| thermal | [warn_about_rom_quality](/Users/kenny/mrm-model/mrm/plant/thermal.py:694) | 1 |
| thermal | [validate_site_coordinates_um](/Users/kenny/mrm-model/mrm/plant/thermal.py:720) | 3 |
| thermal | [LayoutSites.__post_init__](/Users/kenny/mrm-model/mrm/plant/thermal.py:760) | 1 |
| thermal | [LayoutSites.n_rings](/Users/kenny/mrm-model/mrm/plant/thermal.py:789) | 1 |
| thermal | [LayoutSites.line](/Users/kenny/mrm-model/mrm/plant/thermal.py:798) | 10 |
| thermal | [LayoutSites.from_sites](/Users/kenny/mrm-model/mrm/plant/thermal.py:827) | 4 |
| thermal | [LayoutSites.from_sites.site_ring_index](/Users/kenny/mrm-model/mrm/plant/thermal.py:844) | 2 |
| thermal | [InterimRom.textbook_corner](/Users/kenny/mrm-model/mrm/plant/thermal.py:1067) | 1 |
| thermal | [calculate_crosstalk_time_constant_s](/Users/kenny/mrm-model/mrm/plant/thermal.py:1085) | 7 |
| thermal | [calculate_site_coupling_fraction](/Users/kenny/mrm-model/mrm/plant/thermal.py:1113) | 5 |
| thermal | [ThermalSystem.batch](/Users/kenny/mrm-model/mrm/plant/thermal.py:1170) | 1 |
| thermal | [ThermalSystem.n_lags](/Users/kenny/mrm-model/mrm/plant/thermal.py:1179) | 1 |
| thermal | [ThermalSystem.n_src](/Users/kenny/mrm-model/mrm/plant/thermal.py:1188) | 1 |
| thermal | [ThermalSystem.n_obs](/Users/kenny/mrm-model/mrm/plant/thermal.py:1197) | 1 |
| thermal | [ThermalSystem.coefficients](/Users/kenny/mrm-model/mrm/plant/thermal.py:1206) | 2 |
| thermal | [ThermalSystem.zero_state](/Users/kenny/mrm-model/mrm/plant/thermal.py:1217) | 2 |
| thermal | [ThermalSystem.steady_state](/Users/kenny/mrm-model/mrm/plant/thermal.py:1226) | 2 |
| thermal | [ThermalSystem.outputs](/Users/kenny/mrm-model/mrm/plant/thermal.py:1236) | 3 |
| thermal | [ThermalSystem.source_index](/Users/kenny/mrm-model/mrm/plant/thermal.py:1249) | 4 |
| thermal | [ThermalSystem.observer_index](/Users/kenny/mrm-model/mrm/plant/thermal.py:1260) | 4 |
| thermal | [ThermalSystem.kernel_arrays](/Users/kenny/mrm-model/mrm/plant/thermal.py:1271) | 1 |
| thermal | [find_thermal_element_index](/Users/kenny/mrm-model/mrm/plant/thermal.py:1295) | 7 |
| thermal | [normalize_world_parameter_vector](/Users/kenny/mrm-model/mrm/plant/thermal.py:1319) | 2 |
| thermal | [infer_world_count](/Users/kenny/mrm-model/mrm/plant/thermal.py:1337) | 1 |
| thermal | [encode_thermal_scaling_hypotheses](/Users/kenny/mrm-model/mrm/plant/thermal.py:1353) | 2 |
| thermal | [_Lags.__init__](/Users/kenny/mrm-model/mrm/plant/thermal.py:1382) | 2 |
| thermal | [_Lags.add](/Users/kenny/mrm-model/mrm/plant/thermal.py:1393) | 10 |
| thermal | [_Elements.__init__](/Users/kenny/mrm-model/mrm/plant/thermal.py:1422) | 1 |
| thermal | [_Elements.add_src](/Users/kenny/mrm-model/mrm/plant/thermal.py:1431) | 5 |
| thermal | [_Elements.add_obs](/Users/kenny/mrm-model/mrm/plant/thermal.py:1447) | 5 |
| thermal | [build_thermal_sources_and_observers](/Users/kenny/mrm-model/mrm/plant/thermal.py:1467) | 7 |
| thermal | [calculate_neighbor_ranks_and_distances](/Users/kenny/mrm-model/mrm/plant/thermal.py:1525) | 1 |
| thermal | [append_interim_heat_path_lags](/Users/kenny/mrm-model/mrm/plant/thermal.py:1557) | 7 |
| thermal | [append_interim_heat_path_lags.calculate_path_time_constant_s](/Users/kenny/mrm-model/mrm/plant/thermal.py:1580) | 3 |
| thermal | [append_interim_heat_path_lags.calculate_path_coupling_fraction](/Users/kenny/mrm-model/mrm/plant/thermal.py:1594) | 1 |
| thermal | [append_interim_heat_path_lags.append_heater_response_shape](/Users/kenny/mrm-model/mrm/plant/thermal.py:1602) | 6 |
| thermal | [append_extracted_heat_path_lags](/Users/kenny/mrm-model/mrm/plant/thermal.py:1686) | 8 |
| thermal | [append_extracted_heat_path_lags.site_matches_rom_offset](/Users/kenny/mrm-model/mrm/plant/thermal.py:1704) | 3 |
| thermal | [append_extracted_heat_path_lags.interpolate_cross_residues](/Users/kenny/mrm-model/mrm/plant/thermal.py:1725) | 3 |
| thermal | [replace_package_tail_preserving_dc_gain](/Users/kenny/mrm-model/mrm/plant/thermal.py:1780) | 3 |
| thermal | [calculate_pair_dc_scale](/Users/kenny/mrm-model/mrm/plant/thermal.py:1813) | 4 |
| thermal | [calculate_shared_pair_time_scales](/Users/kenny/mrm-model/mrm/plant/thermal.py:1845) | 9 |
| thermal | [assemble](/Users/kenny/mrm-model/mrm/plant/thermal.py:1912) | 9 |
| thermal | [validate_and_broadcast_interim_parameters](/Users/kenny/mrm-model/mrm/plant/thermal.py:2090) | 2 |
| thermal | [validate_and_broadcast_interim_parameters.broadcast_interim_ring_parameter](/Users/kenny/mrm-model/mrm/plant/thermal.py:2097) | 2 |
| thermal | [assemble_interim_thermal_system](/Users/kenny/mrm-model/mrm/plant/thermal.py:2154) | 8 |
| thermal | [simulate](/Users/kenny/mrm-model/mrm/plant/thermal.py:2176) | 7 |
| thermal | [select_thermal_response_lags](/Users/kenny/mrm-model/mrm/plant/thermal.py:2213) | 3 |
| thermal | [sum_selected_thermal_responses](/Users/kenny/mrm-model/mrm/plant/thermal.py:2242) | 4 |
| thermal | [frequency_response](/Users/kenny/mrm-model/mrm/plant/thermal.py:2262) | 4 |
| thermal | [step_response](/Users/kenny/mrm-model/mrm/plant/thermal.py:2277) | 4 |
| thermal | [ramp_response](/Users/kenny/mrm-model/mrm/plant/thermal.py:2293) | 4 |
| thermal | [dc_gain](/Users/kenny/mrm-model/mrm/plant/thermal.py:2307) | 3 |
| thermal | [state_space](/Users/kenny/mrm-model/mrm/plant/thermal.py:2321) | 4 |
| thermal | [reciprocity_error](/Users/kenny/mrm-model/mrm/plant/thermal.py:2351) | 3 |
| thermal | [selfheat_dc_k_per_w](/Users/kenny/mrm-model/mrm/plant/thermal.py:2379) | 1 |
| thermal | [cross_optical_lag_gain](/Users/kenny/mrm-model/mrm/plant/thermal.py:2393) | 3 |
| thermal | [assert_cross_lag_gain](/Users/kenny/mrm-model/mrm/plant/thermal.py:2416) | 4 |
| thermal | [calculate_conductivity_residue_scale](/Users/kenny/mrm-model/mrm/plant/thermal.py:2437) | 4 |

the source snapshots, rename manifests, baseline failure log, and calculation comparisons are in the folder named by /tmp/mrm-physics-readability-workdir.
