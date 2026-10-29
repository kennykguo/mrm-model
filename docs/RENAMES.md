# Renamed names (readability pass, October 2026)

The readability pass of October 2026 renamed cryptic names in the code so that a reader can guess them from
the physics or the role: `_op.ring_lines_nb` became `optics.ring_lines_nb`, `args_f` became `packed_f`,
and the kernel block `S.rf` became `S.ring_float`. No result changed: the engine is bit-identical, the golden
traces match, and every name that a user types or that is written to disk (YAML keys, CLI commands, trace and
metric names, golden keys) kept its spelling. This table helps you read older notes, run logs and diffs.

Names local to one function are not listed. Module aliases (`_op` for `mrm.plant.optics`) are listed once,
with every file that used them. Kinds: function, class, constant, kernel-array (a KParams or KState field of
the compiled engine, mrm/sim/state_spec.py), module-alias, attribute, parameter, fixture (a pytest fixture),
and variable-map id prefix (the value ids of docs/variables and the depends_on entries of
mrm/sim/state_spec.py). A few rows combine two kinds, for example function / fixture.

## mrm.plant

| old | new | kind | file |
|---|---|---|---|
| `_coupler_cache_entry` | `cache_coupler_field_factors` | function - previous name remains an alias | archive/optics/retired_cached_approximations.py |
| `_cached_round_trip_loss_factors` | `cache_round_trip_loss_factors` | function - previous name remains an alias | archive/optics/retired_cached_approximations.py |
| `_airy_cache_entry` | `cache_airy_response_factors` | function - previous name remains an alias | archive/optics/retired_cached_approximations.py |
| `refresh_selfheating_optical_caches` | `cache_refresh_selfheating_optics` | function - previous name remains an alias | archive/optics/retired_cached_approximations.py |
| `_airy_cache_refresh` | `cache_refresh_ring_optical_factors` | function - previous name remains an alias | archive/optics/retired_cached_approximations.py |
| `_responses_at_last_temperature` | `cache_read_responses_at_last_temperature` | function - previous name remains an alias | archive/optics/retired_cached_approximations.py |
| `write_cached_optical_outputs` | `cache_write_optical_outputs` | function - previous name remains an alias | archive/optics/retired_cached_approximations.py |
| `_write_all_cached_optical_outputs` | `cache_write_all_optical_outputs` | function - previous name remains an alias | archive/optics/retired_cached_approximations.py |
| `last_optical_evaluation_matches` | `cache_matches_last_optical_evaluation` | function - previous name remains an alias | mrm/plant/optics.py, archive/optics/retired_cached_approximations.py |
| `clear_last_optical_evaluation` | `cache_clear_last_optical_evaluation` | function - previous name remains an alias | mrm/plant/optics.py, archive/optics/retired_cached_approximations.py |
| `eng_lines_changed` | `cache_update_laser_wavelengths` | function - previous name remains an alias | mrm/sim/kernel.py |
| `ring_lines_nb` | `evaluate_ring_optical_outputs` | function - previous name remains an alias | mrm/plant/optics.py, mrm/sim/kernel.py |
| `el` | `electrical` | module-alias | mrm/plant/plant.py, mrm/sim/kernel_ref.py, mrm/control/models.py, mrm/validation/replicas.py, tests/verification/test_ctrl_b_wdm.py, test_ctrl_c_harness.py |
| `K` | `kernel` | module-alias | mrm/plant/plant.py, mrm/sim/kernel_ref.py, mrm/sim/runner.py, mrm/sim/gym_env.py, webviz/generate.py, tools/perf_bench.py, tools/perf_tick_prof.py, tests/perf/test_perf.py |
| `op` | `optics` | module-alias | mrm/plant/plant.py, mrm/sim/kernel_ref.py, mrm/control/models.py, mrm/validation/replicas.py, tests/verification/test_ctrl_b_plant.py, test_ctrl_b_bitstat.py |
| `ph` | `phase` | module-alias | mrm/plant/plant.py, mrm/sim/kernel_ref.py, mrm/control/models.py, mrm/validation/replicas.py |
| `sg` | `(direct imports Scenario, SI_NC, scen_eval)` | module-alias | mrm/plant/plant.py |
| `sn` | `sensing` | module-alias | mrm/plant/plant.py |
| `th` | `thermal` | module-alias | mrm/plant/plant.py, mrm/sim/kernel_ref.py, mrm/validation/replicas.py, tests/closure/test_thermal_crosstalk.py, tests/closure/test_thermal_notes.py |
| `_rng` | `rng` | module-alias | mrm/plant/sensing.py, mrm/plant/laser.py, mrm/plant/datapath.py, mrm/plant/plant.py, mrm/sim/kernel.py, mrm/sim/kernel_ref.py, mrm/params/realize.py, mrm/scenarios/generators.py, mrm/core/timing.py |
| `mat` | `materials` | module-alias | mrm/plant/thermal.py |
| `_ones` | `_ones_in_bits` | function | mrm/plant/datapath.py |
| `_pack32` | `_pack_bits32` | function | mrm/plant/datapath.py |
| `_SLOT_PRBS_PHASE` | `SLOT_PRBS_PHASE` | constant | mrm/plant/datapath.py |
| `_TABLES` | `_PRBS_TABLES` | constant | mrm/plant/datapath.py |
| `_bit_err` | `_bit_err_param` | function | mrm/plant/electrical.py |
| `_f1` | `_param_b` | function | mrm/plant/electrical.py |
| `_f2` | `_param_bn` | function | mrm/plant/electrical.py |
| `_ix` | `_bidx` | function | mrm/plant/electrical.py |
| `_p_from_v2` | `_power_from_v2` | function | mrm/plant/electrical.py |
| `_rate_ticks` | `_update_period_ticks` | function | mrm/plant/electrical.py |
| `_ticks` | `_whole_ticks` | function | mrm/plant/electrical.py |
| `BiasBank._zero` | `BiasBank._no_offset` | attribute | mrm/plant/electrical.py |
| `BiasBank._zz` | `BiasBank._no_noise` | attribute | mrm/plant/electrical.py |
| `HeaterBank._i` | `HeaterBank._i_a` | attribute | mrm/plant/electrical.py |
| `HeaterBank._p` | `HeaterBank._p_w` | attribute | mrm/plant/electrical.py |
| `HeaterBank._r` | `HeaterBank._r_ohm` | attribute | mrm/plant/electrical.py |
| `HeaterBank._v` | `HeaterBank._v_v` | attribute | mrm/plant/electrical.py |
| `_fill_w` | `_fill_brownian_nodes` | function | mrm/plant/laser.py |
| `_SLOT_POW` | `_SLOT_POWER` | constant | mrm/plant/laser.py |
| `_uniform_lm` | `_device_uniforms` | function | mrm/plant/laser.py |
| `_afac` | `cache_round_trip_loss_factors` | function | mrm/plant/optics.py |
| `_fast_check` | `cache_refresh_ring_optical_factors` | function | mrm/plant/optics.py |
| `_fast_flag` | `_airy_path_flag` | function | mrm/plant/optics.py |
| `_fc_entry` | `cache_airy_response_factors` | function | mrm/plant/optics.py |
| `_fh_entry` | `_far_hold_entry` | function | mrm/plant/optics.py |
| `_fk_entry` | `cache_coupler_field_factors` | function | mrm/plant/optics.py |
| `_imod` | `_pos_mod` | function | mrm/plant/optics.py |
| `_le_values` | `cache_read_responses_at_last_temperature` | function | mrm/plant/optics.py |
| `_nl` | `_finite_array` | function | mrm/plant/optics.py |
| `_stages_d` | `_exact_stages_slope` | function | mrm/plant/optics.py |
| `_stages_v` | `_exact_stages` | function | mrm/plant/optics.py |
| `_U` | `_as_unsigned_array_index` | constant | mrm/plant/optics.py, mrm/plant/electrical.py, mrm/sim/kernel.py |
| `args_f` | `packed_f` | parameter | mrm/plant/optics.py, mrm/plant/selfheat.py |
| `args_i` | `packed_i` | parameter | mrm/plant/optics.py, mrm/plant/selfheat.py |
| `nb` | `line_rows` | parameter | mrm/plant/optics.py |
| `rg` | `ring_rows` | parameter | mrm/plant/optics.py |
| `fal` | `fca_loss` | parameter | mrm/plant/phase.py |
| `tfac` | `loss_tfac` | parameter | mrm/plant/phase.py |
| `_b` | `_kernel_arr_b` | function | mrm/plant/plant.py |
| `_bn` | `_kernel_arr_bn` | function | mrm/plant/plant.py |
| `_c` | `_kernel_arr` | function | mrm/plant/plant.py |
| `_edge_jit` | `_edge_jitter_ticks` | function | mrm/plant/plant.py |
| `_full_n` | `_per_ring_index` | function | mrm/plant/plant.py |
| `Plant._args_f0` | `Plant._packed_f0` | attribute | mrm/plant/plant.py |
| `_CODE_MASK` | `SAMPLE_CODE_MASK` | constant | mrm/plant/sensing.py |
| `_FLAG_SHIFT` | `SAMPLE_FLAG_SHIFT` | constant | mrm/plant/sensing.py |
| `_keys` | `stream_keys` | function | mrm/plant/sensing.py |
| `_SF_ACC_N` | `_SF_ACC_TICKS` | constant | mrm/plant/sensing.py |
| `_SF_ACC_VS` | `_SF_ACC_VAR_SHOT` | constant | mrm/plant/sensing.py |
| `_SF_ACC_VT` | `_SF_ACC_VAR_TIA` | constant | mrm/plant/sensing.py |
| `_SF_DEC` | `_SF_DECAY` | constant | mrm/plant/sensing.py |
| `_SF_YDET` | `_SF_Y_SIGNAL` | constant | mrm/plant/sensing.py |
| `_SF_YNS` | `_SF_Y_SHOT` | constant | mrm/plant/sensing.py |
| `_SF_YNT` | `_SF_Y_TIA` | constant | mrm/plant/sensing.py |
| `_TP_RES` | `_TP_RESOLUTION` | constant | mrm/plant/sensing.py |
| `_TP_SELF` | `_TP_SELF_HEAT` | constant | mrm/plant/sensing.py |

## mrm.sim

| old | new | kind | file |
|---|---|---|---|
| `mf` | `manifest` | module-alias | mrm/sim/experiment.py |
| `_api` | `api` | module-alias | mrm/sim/gym_env.py, mrm/control/fxp.py |
| `M (metrics alias)` | `direct imports AF_*/AI_* from mrm.sim.metrics` | module-alias | mrm/sim/gym_env.py |
| `_cfg` | `numba_config` | module-alias | mrm/sim/kernel.py |
| `_el` | `electrical` | module-alias | mrm/sim/kernel.py |
| `_op` | `optics` | module-alias | mrm/sim/kernel.py |
| `_ph` | `phase` | module-alias | mrm/sim/kernel.py |
| `_sg` | `generators` | module-alias | mrm/sim/kernel.py |
| `_sh` | `(removed: unused)` | module-alias | mrm/sim/kernel.py |
| `_spec` | `state_spec` | module-alias | mrm/sim/kernel.py |
| `_th` | `thermal` | module-alias | mrm/sim/kernel.py |
| `shm` | `selfheat` | module-alias | mrm/sim/kernel_ref.py |
| `_dt` | `datetime` | module-alias | mrm/sim/manifest.py, mrm/control/cli.py, mrm/waveform/lut.py, mrm/validation/replicas.py, mrm/validation/report.py, mrm/calibration/pipeline.py, mrm/calibration/preset.py, mrm/extraction/thermal_fem.py, mrm/extraction/common.py, mrm/extraction/junction.py |
| `el / op / ph / th` | `electrical / optics / phase / thermal` | module-alias | mrm/sim/runner.py |
| `_HDR` | `_HEADER` | constant | mrm/sim/bridge.py |
| `_HELLO` | `_HELLO_FIXED` | constant | mrm/sim/bridge.py |
| `_EXT` | `_EXTENSION_BY_NAME` | constant | mrm/sim/experiment.py |
| `_header` | `_report_header` | function | mrm/sim/experiment.py |
| `_KINDS` | `_KIND_RUNNERS` | constant | mrm/sim/experiment.py |
| `_OPS` | `_CRITERION_OPS` | constant | mrm/sim/experiment.py |
| `_ov_str` | `_override_str` | function | mrm/sim/experiment.py |
| `_qs_one` | `_locked_steady_state_case` | function | mrm/sim/experiment.py |
| `_REDUCE` | `_RING_REDUCTIONS` | constant | mrm/sim/experiment.py |
| `_unc_var` | `_uncertainty_variation` | function | mrm/sim/experiment.py |
| `0x6A6974 literal` | `_JITTER_RNG_TAG` | constant | mrm/sim/gym_env.py |
| `0x6D726D literal` | `_SCENARIO_RNG_TAG` | constant | mrm/sim/gym_env.py |
| `_ACT_BI_ID` | `_BIAS_ACT_ID` | constant | mrm/sim/gym_env.py |
| `_ACT_HE_ID` | `_HEATER_ACT_ID` | constant | mrm/sim/gym_env.py |
| `_BIA_ID` | `_BIAS_APPLIED_ID` | constant | mrm/sim/gym_env.py |
| `_CP_BI_C0` | `_USER_BIAS_C0` | constant | mrm/sim/gym_env.py |
| `_CP_BI_S` | `_USER_BIAS_SHIFT` | constant | mrm/sim/gym_env.py |
| `_CP_DMAX_B` | `_USER_MAX_DELTA_BIAS` | constant | mrm/sim/gym_env.py |
| `_CP_DMAX_H` | `_USER_MAX_DELTA_HEATER` | constant | mrm/sim/gym_env.py |
| `_CP_FIN` | `_USER_IN_FRAC_BITS` | constant | mrm/sim/gym_env.py |
| `_CP_HAS_BI` | `_USER_HAS_BIAS` | constant | mrm/sim/gym_env.py |
| `_CP_HE_C0` | `_USER_HEATER_C0` | constant | mrm/sim/gym_env.py |
| `_CP_HE_S` | `_USER_HEATER_SHIFT` | constant | mrm/sim/gym_env.py |
| `_CP_HEADER` | `_USER_HEADER_SIZE` | constant | mrm/sim/gym_env.py |
| `_CP_K` | `_USER_HISTORY_BLOCKS` | constant | mrm/sim/gym_env.py |
| `_CP_MON_C0` | `_USER_MONITOR_C0` | constant | mrm/sim/gym_env.py |
| `_CP_MON_S` | `_USER_MONITOR_SHIFT` | constant | mrm/sim/gym_env.py |
| `_CP_NT` | `_USER_N_TEMP` | constant | mrm/sim/gym_env.py |
| `_HEA_ID` | `_HEATER_APPLIED_ID` | constant | mrm/sim/gym_env.py |
| `_norm_seeds` | `_normalize_seeds` | function | mrm/sim/gym_env.py |
| `_P_A_BI` | `_P_ACT_OFF_BIAS` | constant | mrm/sim/gym_env.py |
| `_P_A_HE` | `_P_ACT_OFF_HEATER` | constant | mrm/sim/gym_env.py |
| `_P_BI_C0` | `_P_BIAS_C0` | constant | mrm/sim/gym_env.py |
| `_P_BI_S` | `_P_BIAS_SHIFT` | constant | mrm/sim/gym_env.py |
| `_P_D` | `_P_OBS_SIZE` | constant | mrm/sim/gym_env.py |
| `_P_DMAX_B` | `_P_MAX_DELTA_BIAS` | constant | mrm/sim/gym_env.py |
| `_P_DMAX_H` | `_P_MAX_DELTA_HEATER` | constant | mrm/sim/gym_env.py |
| `_P_FIN` | `_P_IN_FRAC_BITS` | constant | mrm/sim/gym_env.py |
| `_P_FOUT` | `_P_OUT_FRAC_BITS` | constant | mrm/sim/gym_env.py |
| `_P_HAS_BI` | `_P_HAS_BIAS` | constant | mrm/sim/gym_env.py |
| `_P_HE_C0` | `_P_HEATER_C0` | constant | mrm/sim/gym_env.py |
| `_P_HE_S` | `_P_HEATER_SHIFT` | constant | mrm/sim/gym_env.py |
| `_P_K` | `_P_HISTORY_BLOCKS` | constant | mrm/sim/gym_env.py |
| `_P_M` | `_P_SAMPLES_PER_BLOCK` | constant | mrm/sim/gym_env.py |
| `_P_MLP` | `_P_MLP_BASE` | constant | mrm/sim/gym_env.py |
| `_P_MON_C0` | `_P_MONITOR_C0` | constant | mrm/sim/gym_env.py |
| `_P_MON_S` | `_P_MONITOR_SHIFT` | constant | mrm/sim/gym_env.py |
| `_P_MW` | `_P_MLP_MAX_WIDTH` | constant | mrm/sim/gym_env.py |
| `_P_N` | `_P_N_RINGS` | constant | mrm/sim/gym_env.py |
| `_P_NOUT` | `_P_N_OUT` | constant | mrm/sim/gym_env.py |
| `_P_NT` | `_P_N_TEMP` | constant | mrm/sim/gym_env.py |
| `_P_O_BIA` | `_P_OBS_OFF_BIAS_APPLIED` | constant | mrm/sim/gym_env.py |
| `_P_O_CNT` | `_P_OBS_OFF_COUNT` | constant | mrm/sim/gym_env.py |
| `_P_O_HEA` | `_P_OBS_OFF_HEATER_APPLIED` | constant | mrm/sim/gym_env.py |
| `_P_O_LAT` | `_P_OBS_OFF_LATEST` | constant | mrm/sim/gym_env.py |
| `_P_O_SAMP` | `_P_OBS_OFF_SAMPLES` | constant | mrm/sim/gym_env.py |
| `_P_TEMP` | `_P_TEMP_BASE` | constant | mrm/sim/gym_env.py |
| `act_map kind in (0, 1)` | `dst in (DST_HEATER, DST_BIAS) via AM_DST/AM_OFFSET/AM_SIZE` | constant | mrm/sim/gym_env.py |
| `MRMEnv._acc_snapshot` | `MRMEnv._accumulator_snapshot` | function | mrm/sim/gym_env.py |
| `_auto_h` | `_auto_substep` | function | mrm/sim/kernel.py |
| `_auto_h_a` | `_auto_substep_a` | function | mrm/sim/kernel.py |
| `_cum_q` | `_charge_at` | function | mrm/sim/kernel.py |
| `_decl` | `_declared_line_a` | function | mrm/sim/kernel.py |
| `_eo` | `_ext_offset` | function | mrm/sim/kernel.py |
| `_heater_split_fill` | `_heater_split_table` | function | mrm/sim/kernel.py |
| `_row` | `_world_row` | function | mrm/sim/kernel.py |
| `_rr_apply_a` | `_round_robin_apply_a` | function | mrm/sim/kernel.py |
| `_rr_save_a` | `_round_robin_save_a` | function | mrm/sim/kernel.py |
| `_scen_eval` | `_scenario_eval` | function | mrm/sim/kernel.py |
| `_sh_blocks` | `_input_row_views` | function | mrm/sim/kernel.py |
| `_sh_out_blocks` | `_output_row_views` | function | mrm/sim/kernel.py |
| `_SIG_AGG` | `_SIG_AGGRESSOR` | constant | mrm/sim/kernel.py |
| `_SIG_BIAS` | `_SIG_BIAS_V` | constant | mrm/sim/kernel.py |
| `_SIG_BOFF` | `_SIG_BIAS_OFFSET` | constant | mrm/sim/kernel.py |
| `_SIG_CRESET` | `_SIG_CONTROLLER_RESET` | constant | mrm/sim/kernel.py |
| `_SIG_DEN` | `_SIG_DRIVER_ENABLE` | constant | mrm/sim/kernel.py |
| `_SIG_DENS` | `_SIG_ONES_DENSITY` | constant | mrm/sim/kernel.py |
| `_SIG_DF` | `_SIG_LASER_DFREQ` | constant | mrm/sim/kernel.py |
| `_SIG_DROP` | `_SIG_FAULT_ADC_DROP` | constant | mrm/sim/kernel.py |
| `_SIG_EN` | `_SIG_LINE_ENABLE` | constant | mrm/sim/kernel.py |
| `_SIG_GLITCH` | `_SIG_FAULT_DAC_GLITCH` | constant | mrm/sim/kernel.py |
| `_SIG_LOSS` | `_SIG_INPUT_LOSS_DB` | constant | mrm/sim/kernel.py |
| `_sig_off` | `_signal_offset` | function | mrm/sim/kernel.py |
| `_SIG_PDB` | `_SIG_LASER_POWER_DB` | constant | mrm/sim/kernel.py |
| `_SIG_SAT` | `_SIG_FAULT_ADC_SATURATE` | constant | mrm/sim/kernel.py |
| `_SIG_STUCK` | `_SIG_FAULT_ADC_STUCK` | constant | mrm/sim/kernel.py |
| `_SIG_SWING` | `_SIG_SWING_SCALE` | constant | mrm/sim/kernel.py |
| `_SIG_TD` | `_SIG_TRANSITION_DENSITY` | constant | mrm/sim/kernel.py |
| `_SIG_XPU` | `_SIG_XPU_RISE` | constant | mrm/sim/kernel.py |
| `_th_coef` | `_pack_lag_coefs` | function | mrm/sim/kernel.py |
| `_th_dead_init` | `_find_dead_sources` | function | mrm/sim/kernel.py |
| `_th_gains` | `_thermal_gains` | function | mrm/sim/kernel.py |
| `_th_live` | `_live_lags` | function | mrm/sim/kernel.py |
| `_th_struct` | `_lag_lists` | function | mrm/sim/kernel.py |
| `_th_tables` | `_lag_pair_tables` | function | mrm/sim/kernel.py |
| `_th_wake` | `_wake_dead_sources` | function | mrm/sim/kernel.py |
| `_uu_fill` | `_pack_source_inputs` | function | mrm/sim/kernel.py |
| `args_f / args_i` | `packed_f / packed_i` | parameter | mrm/sim/kernel.py |
| `EXT_I` | `N_EXT` | constant | mrm/sim/kernel.py |
| `rg / nb_` | `ring_rows / line_rows` | parameter | mrm/sim/kernel.py |
| `_cold_x` | `_cold_coordinate` | function | mrm/sim/kernel_ref.py |
| `ReferenceEngine.args_f / args_i` | `ReferenceEngine.packed_f / packed_i` | attribute | mrm/sim/kernel_ref.py |
| `mpf[8 + i]` | `mpf[MP_ABOVE0 + i]` | constant | mrm/sim/metrics.py |
| `rec_i[r, 0..4]` | `rec_i[r, _REC_FIELD / _REC_DECIMATION / _REC_REDUCER / _REC_FIRST_SLOT / _REC_SAMPLES]` | constant | mrm/sim/metrics.py |
| `red == 0 / 1 / 2` | `red == _RED_MEAN / _RED_MIN / _RED_MAX` | constant | mrm/sim/metrics.py |
| `("phi0", ..., "trt") literal x3` | `_OPTICS_OUT_KEYS` | constant | mrm/sim/runner.py |
| `_data_state` | `_data_state_arrays` | function | mrm/sim/runner.py |
| `_heater_drive / _code_for_power (hp)` | `(heater_params)` | parameter | mrm/sim/runner.py |
| `_hold_in_data_state(ds, vb, grid, dn, al, cc, rs, t_amb, t_ring, t_heat, det, ph_lock, bus, r2l, sysm, Gb, pj_src, pd_src)` | `(data_arrays, v_bias, v_grid, dneff_tab, alpha_tab, cap_tab, rs_tab, t_amb_k, t_ring_k, t_heater_k, det_m, phi_lock, bus_order, ring_to_line, thermal_sys, gain_b, junction_src, driver_src)` | parameter | mrm/sim/runner.py |
| `KParams.sh_args_i` | `KParams.selfheat_int` | kernel-array | mrm/sim/state_spec.py |
| `KState.act_tmp` | `KState.act_popped` | kernel-array | mrm/sim/state_spec.py |
| `KState.c_pf` | `KState.ctrl_p_f64` | kernel-array | mrm/sim/state_spec.py |
| `KState.c_pi` | `KState.ctrl_p_i64` | kernel-array | mrm/sim/state_spec.py |
| `KState.c_sf` | `KState.ctrl_s_f64` | kernel-array | mrm/sim/state_spec.py |
| `KState.c_sf0` | `KState.ctrl_s_f64_init` | kernel-array | mrm/sim/state_spec.py |
| `KState.c_si` | `KState.ctrl_s_i64` | kernel-array | mrm/sim/state_spec.py |
| `KState.c_si0` | `KState.ctrl_s_i64_init` | kernel-array | mrm/sim/state_spec.py |
| `KState.jx` | `KState.junction` | kernel-array | mrm/sim/state_spec.py |
| `KState.lf` | `KState.line_float` | kernel-array | mrm/sim/state_spec.py |
| `KState.lt` | `KState.truth_line` | kernel-array | mrm/sim/state_spec.py |
| `KState.or_a` | `KState.oracle_amp` | kernel-array | mrm/sim/state_spec.py |
| `KState.or_cw` | `KState.oracle_cw` | kernel-array | mrm/sim/state_spec.py |
| `KState.or_fw` | `KState.oracle_fw` | kernel-array | mrm/sim/state_spec.py |
| `KState.or_k2d` | `KState.oracle_k2d` | kernel-array | mrm/sim/state_spec.py |
| `KState.or_k2t` | `KState.oracle_k2t` | kernel-array | mrm/sim/state_spec.py |
| `KState.or_phi` | `KState.oracle_phi` | kernel-array | mrm/sim/state_spec.py |
| `KState.or_psi` | `KState.oracle_psi` | kernel-array | mrm/sim/state_spec.py |
| `KState.or_res0` | `KState.oracle_res0` | kernel-array | mrm/sim/state_spec.py |
| `KState.or_res1` | `KState.oracle_res1` | kernel-array | mrm/sim/state_spec.py |
| `KState.or_trt` | `KState.oracle_trt` | kernel-array | mrm/sim/state_spec.py |
| `KState.r8` | `KState.ring_int8` | kernel-array | mrm/sim/state_spec.py |
| `KState.rb` | `KState.ring_bool` | kernel-array | mrm/sim/state_spec.py |
| `KState.rf` | `KState.ring_float` | kernel-array | mrm/sim/state_spec.py |
| `KState.ri` | `KState.ring_int` | kernel-array | mrm/sim/state_spec.py |
| `KState.sh_args_f` | `KState.selfheat_float` | kernel-array | mrm/sim/state_spec.py |
| `KState.tv` | `KState.truth` | kernel-array | mrm/sim/state_spec.py |
| `KState.wf` | `KState.world_float` | kernel-array | mrm/sim/state_spec.py |
| `KState.wi` | `KState.world_int` | kernel-array | mrm/sim/state_spec.py |
| `lf.<row>` | `line_float.<row>` | variable-map id prefix (depends_on, docs/variables ids) | mrm/sim/state_spec.py |
| `r8.<row>` | `ring_int8.<row>` | variable-map id prefix (depends_on, docs/variables ids) | mrm/sim/state_spec.py |
| `rb.<row>` | `ring_bool.<row>` | variable-map id prefix (depends_on, docs/variables ids) | mrm/sim/state_spec.py |
| `rf.<row>` | `ring_float.<row>` | variable-map id prefix (depends_on, docs/variables ids) | mrm/sim/state_spec.py |
| `ri.<row>` | `ring_int.<row>` | variable-map id prefix (depends_on, docs/variables ids) | mrm/sim/state_spec.py |
| `sh.<row>` | `selfheat.<row>` | variable-map id prefix (depends_on, docs/variables ids) | mrm/sim/state_spec.py |
| `wf.<row>` | `world_float.<row>` | variable-map id prefix (depends_on, docs/variables ids) | mrm/sim/state_spec.py |
| `wi.<row>` | `world_int.<row>` | variable-map id prefix (depends_on, docs/variables ids) | mrm/sim/state_spec.py |
| `_ARRAYS` | `_ARRAY_NAMES` | constant | mrm/sim/vectors.py |
| `_result` | `_replay_result` | function | mrm/sim/vectors.py |
| `_vname` | `_verilog_name` | function | mrm/sim/vectors.py |

## mrm.core

| old | new | kind | file |
|---|---|---|---|
| `_attr` | `_hardware_value` | function | mrm/core/channels.py |
| `_build_row` | `_channel_from_row` | function | mrm/core/channels.py |
| `_idtype` | `_signed_dtype` | function | mrm/core/channels.py |
| `_udtype` | `_unsigned_dtype` | function | mrm/core/channels.py |
| `AGGS` | `AGG_RULES` | constant | mrm/core/channels.py |
| `_I_TWO_62` | `_TWO_62_INT` | constant | mrm/core/rng.py |
| `_ISSUED` | `_ISSUED_IDS` | constant | mrm/core/rng.py |
| `_M32` | `_MASK32` | constant | mrm/core/rng.py |
| `_round` | `_philox_round` | function | mrm/core/rng.py |
| `_SH32` | `_SHIFT32` | constant | mrm/core/rng.py |
| `_u53` | `_uniform53` | function | mrm/core/rng.py |
| `fifo_push/fifo_peek/fifo_pop_ready/fifo_pop_all_ready(rel, out_rel)` | `(release_ticks, out_release_ticks)` | function | mrm/core/timing.py |
| `first_response_tick(k, ctrl_latency, dac_latency, adc_period, adc_phase, adc_latency, ctrl_period, ctrl_phase)` | `first_response_tick(edge_tick, ctrl_latency_ticks, dac_latency_ticks, adc_period_ticks, adc_phase_ticks, adc_latency_ticks, ctrl_period_ticks, ctrl_phase_ticks)` | function | mrm/core/timing.py |

## mrm.params

| old | new | kind | file |
|---|---|---|---|
| `sch` | `schema` | module-alias | mrm/params/config.py, mrm/params/realize.py, mrm/params/table.py, tests/closure/test_presets.py, tests/closure/test_sensing_notes.py, tests/verification/test_params_schema.py, test_params_units.py, mrm/sim/experiment.py |
| `rel` | `relations` | module-alias | mrm/params/realize.py, mrm/params/derive.py |
| `_ALT_FIELDS` | `_ALTERNATIVE_FIELDS` | constant | mrm/params/config.py |
| `_ART_CACHE` | `_ARTEFACT_CACHE` | constant | mrm/params/config.py |
| `_entry_prov` | `_check_entry_provenance` | function | mrm/params/config.py |
| `_enum_word` | `_bool_as_enum_word` | function | mrm/params/config.py |
| `_Flat` | `_FlatTree` | class | mrm/params/config.py |
| `_Flat.unc_origin` | `_FlatTree.uncertainty_origin` | attribute | mrm/params/config.py |
| `_Flat.var_origin` | `_FlatTree.variation_origin` | attribute | mrm/params/config.py |
| `_GEOM_DIM_FIELDS` | `_GEOMETRY_DIM_FIELDS` | constant | mrm/params/config.py |
| `_GEOM_DIMS` | `_GEOMETRY_DIMS` | constant | mrm/params/config.py |
| `_GEOM_FIELDS` | `_GEOMETRY_FIELDS` | constant | mrm/params/config.py |
| `_hyp_choice_map` | `_hypothesis_choice_map` | function | mrm/params/config.py |
| `_Node` | `_FlatNode` | class | mrm/params/config.py |
| `_PROV_RE` | `_PROVENANCE_RE` | constant | mrm/params/config.py |
| `_sizes` | `_level_sizes` | function | mrm/params/config.py |
| `_UNC_COMMON` | `_UNCERTAINTY_COMMON_FIELDS` | constant | mrm/params/config.py |
| `_UNC_FIELDS` | `_UNCERTAINTY_FIELDS` | constant | mrm/params/config.py |
| `_VAR_FIELDS` | `_VARIATION_FIELDS` | constant | mrm/params/config.py |
| `_corr_chol` | `_correlation_cholesky` | function | mrm/params/realize.py |
| `_draw_u` | `_draw_uniform` | function | mrm/params/realize.py |
| `_GEOM_SLOT` | `_GEOMETRY_DIM_ORDER` | constant | mrm/params/realize.py |
| `_hyp_choices` | `_hypothesis_choices` | function | mrm/params/realize.py, mrm/validation/replicas.py, tests/closure/test_presets.py |
| `_JAC_DIMS` | `_JACOBIAN_INPUT_DIMS` | constant | mrm/params/realize.py |
| `_JAC_EVAL_NAME` | `_JACOBIAN_EVAL_NAME` | constant | mrm/params/realize.py |
| `_jac_log_derivative` | `_jacobian_log_derivative` | function | mrm/params/realize.py |
| `_junction / _heater / _monitor (g, bn, b1)` | `(optional, per_ring, per_die)` | parameter | mrm/params/realize.py |
| `_LAW_FULL_SCALE` | `_DRIVER_LAW_FULL_SCALE` | constant | mrm/params/realize.py |
| `_beta` | `_thermo_optic_beta` | function | mrm/params/relations.py |
| `_LAM_LINE` | `_LAMBDA_LINE` | constant | mrm/params/relations.py |
| `_LAM_RES` | `_LAMBDA_RES_COLD` | constant | mrm/params/relations.py |
| `_LB` | `_LOSS_BUDGET` | constant | mrm/params/relations.py |
| `_omega` | `_cold_resonance_omega` | function | mrm/params/relations.py |
| `_overdetermined` | `_resolve_overdetermined` | function | mrm/params/relations.py |
| `_r1` | `_solve_r1_thermo_optic` | function | mrm/params/relations.py |
| `_r1_resid` | `_residual_r1_thermo_optic` | function | mrm/params/relations.py |
| `_r2_eff` | `_solve_r2_heater_efficiency` | function | mrm/params/relations.py |
| `_r2_resid` | `_residual_r2_heater` | function | mrm/params/relations.py |
| `_r2_z` | `_solve_r2_heater_z` | function | mrm/params/relations.py |
| `_r3` | `_solve_r3_geometry` | function | mrm/params/relations.py |
| `_r3_resid` | `_residual_r3_geometry` | function | mrm/params/relations.py |
| `_r4_budget_components` | `_solve_r4_loss_budget` | function | mrm/params/relations.py |
| `_r4_budget_eta` | `_solve_r4_heat_fraction` | function | mrm/params/relations.py |
| `_r4_q` | `_solve_r4_q` | function | mrm/params/relations.py |
| `_r4_q_anchors` | `_solve_r4_q_from_anchors` | function | mrm/params/relations.py |
| `_r4_q_geometry` | `_solve_r4_q_sampled_geometry` | function | mrm/params/relations.py |
| `_r4_resid` | `_residual_r4_q` | function | mrm/params/relations.py |
| `_r5` | `_solve_r5_latency` | function | mrm/params/relations.py |
| `_r5_resid` | `_residual_r5_latency` | function | mrm/params/relations.py |
| `_rel` | `_max_rel_diff` | function | mrm/params/relations.py |
| `_solve_q` | `_solve_one_free_q` | function | mrm/params/relations.py |
| `_ANALYTIC` | `_JUNCTION_ANALYTIC` | constant | mrm/params/schema.py |
| `_cond_holds` | `_condition_holds` | function | mrm/params/schema.py |
| `_LUT` | `_JUNCTION_LUT` | constant | mrm/params/schema.py |
| `_ORDER2` | `_THERMO_OPTIC_ORDER2` | constant | mrm/params/schema.py |
| `Cond` | `Condition` | class | mrm/params/schema.py |
| `_ARGS` | `_BRACED_ARG_COUNT` | constant | mrm/params/symbols.py |
| `_ESCAPES` | `_ESCAPED_CHARS` | constant | mrm/params/symbols.py |
| `_group` | `_tex_argument` | function | mrm/params/symbols.py |
| `_SUB` | `_SUBSCRIPTS` | constant | mrm/params/symbols.py |
| `_SUP` | `_SUPERSCRIPTS` | constant | mrm/params/symbols.py |
| `_WORDS` | `_COMMAND_GLYPHS` | constant | mrm/params/symbols.py |
| `_common` | `_add_common_args` | function | mrm/params/table.py |
| `_fmt_num` | `_format_number` | function | mrm/params/table.py |
| `_hook` | `_thermal_hook_arg` | function | mrm/params/table.py |
| `_load` | `_load_config` | function | mrm/params/table.py |
| `_md` | `_md_cell` | function | mrm/params/table.py, mrm/validation/report.py, tools/gen_variable_map.py |
| `_A` | `_DIM_A` | constant | mrm/params/units.py |
| `_F` | `_DIM_F` | constant | mrm/params/units.py |
| `_HZ` | `_DIM_HZ` | constant | mrm/params/units.py |
| `_J` | `_DIM_J` | constant | mrm/params/units.py |
| `_K` | `_DIM_K` | constant | mrm/params/units.py |
| `_L` | `_DIM_M` | constant | mrm/params/units.py |
| `_NONE` | `_DIM_NONE` | constant | mrm/params/units.py |
| `_OHM` | `_DIM_OHM` | constant | mrm/params/units.py |
| `_T` | `_DIM_S` | constant | mrm/params/units.py |
| `_V` | `_DIM_V` | constant | mrm/params/units.py |
| `_W` | `_DIM_W` | constant | mrm/params/units.py |

## mrm.control

| old | new | kind | file |
|---|---|---|---|
| `_rep` | `replace` | module-alias | mrm/control/baselines/kalman_lqg.py |
| `M` | `linear_models` | module-alias | mrm/control/baselines/kalman_lqg.py, mrm/control/baselines/mpc.py, mrm/validation/oracle.py |
| `g` | `generators` | module-alias | mrm/control/calibration.py, mrm/scenarios/library.py, mrm/scenarios/registry.py, tests/verification/test_ctrl_a_feedforward.py, test_ctrl_a_harness.py (local), test_ctrl_b_plant.py, test_ctrl_b_engine.py (local), test_ctrl_b_ramp.py, test_ctrl_b_robust.py, test_ctrl_b_wdm_slow.py (local) |
| `_ActId` | `ActId` | module-alias | mrm/control/fxp.py |
| `_ObsId` | `ObsId` | module-alias | mrm/control/fxp.py |
| `_canon_nominal` | `_canonical_nominal` | function | mrm/control/api.py |
| `_BParams` | `_BiasDitherParams` | class | mrm/control/baselines/bias_dither.py |
| `_NI` | `_IP_INIT_CODES` | kernel-array | mrm/control/baselines/bias_dither.py |
| `_NUF` | `_N_USER_F` | kernel-array | mrm/control/baselines/bias_dither.py |
| `_SF` | `_FS_HIST` | kernel-array | mrm/control/baselines/bias_dither.py |
| `_SI` | `_IS_PER_RING` | kernel-array | mrm/control/baselines/bias_dither.py |
| `_check` | `_check_dither_params` | function | mrm/control/baselines/dither_lockin.py |
| `_D_NI` | `_IP_INIT_CODES` | kernel-array | mrm/control/baselines/dither_lockin.py |
| `_D_NUF` | `_N_USER_F` | kernel-array | mrm/control/baselines/dither_lockin.py |
| `_d_ring_step` | `_dither_ring_step` | function | mrm/control/baselines/dither_lockin.py |
| `_D_SF` | `_FS_HIST` | kernel-array | mrm/control/baselines/dither_lockin.py |
| `_D_SI` | `_IS_PER_RING` | kernel-array | mrm/control/baselines/dither_lockin.py |
| `_d_telemetry` | `_dither_telemetry` | function | mrm/control/baselines/dither_lockin.py |
| `_DParams` | `_DitherParams` | class | mrm/control/baselines/dither_lockin.py |
| `_DRing` | `_DitherRing` | class | mrm/control/baselines/dither_lockin.py |
| `_W_FACTOR` | `INFLECTION_WIDTH` | constant | mrm/control/baselines/dither_lockin.py |
| `_Common` | `_CommonParams` | class | mrm/control/baselines/extremum_seeking.py |
| `_ESBase` | `_EsBase` | class | mrm/control/baselines/extremum_seeking.py |
| `_PORing` | `_PoRing` | class | mrm/control/baselines/extremum_seeking.py |
| `_Ring` | `_EsRing` | class | mrm/control/baselines/extremum_seeking.py |
| `_HDR` | `_LQG_HEADER_LEN` | kernel-array | mrm/control/baselines/kalman_lqg.py |
| `p_i64/p_f64 header and s_f64 slot numbers` | `_IP_*/_H_*/_OP_*/_S_* named slots` | kernel-array | mrm/control/baselines/kalman_lqg.py |
| `_MHDR` | `_MPC_HEADER_LEN` | kernel-array | mrm/control/baselines/mpc.py |
| `p_f64 header slot numbers` | `_H_* named slots` | kernel-array | mrm/control/baselines/mpc.py |
| `_C0` | `_SPEED_OF_LIGHT_M_PER_S` | constant | mrm/control/baselines/pi_lock.py |
| `_pi_err_c` | `_pi_error_c` | function | mrm/control/baselines/pi_lock.py |
| `_PI_NF` | `_FP_TABLES` | kernel-array | mrm/control/baselines/pi_lock.py |
| `_PI_NI` | `_IP_INIT_CODES` | kernel-array | mrm/control/baselines/pi_lock.py |
| `_PI_NUF` | `_N_USER_F` | kernel-array | mrm/control/baselines/pi_lock.py |
| `_PI_SF` | `_FS_HIST` | kernel-array | mrm/control/baselines/pi_lock.py |
| `_PI_SI` | `_IS_PER_RING` | kernel-array | mrm/control/baselines/pi_lock.py |
| `p_i64/p_f64/s_i64/s_f64 slot numbers` | `_IP_*/_FP_*/_IS_*/_FS_* named slots` | kernel-array | mrm/control/baselines/pi_lock.py, mrm/control/baselines/dither_lockin.py, mrm/control/baselines/bias_dither.py |
| `_BUF` | `_DOWN_BUF_LEN` | constant | mrm/control/baselines/ramp_acquire.py |
| `_gmap` | `_drive_shift_map` | function | mrm/control/calibration.py |
| `_knows` | `_knows_group` | function | mrm/control/calibration.py |
| `_LAW_Q` | `_DRIVE_EXPONENT_OF_LAW` | constant | mrm/control/calibration.py |
| `_temp_c` | `_die_temp_c` | function | mrm/control/calibration.py |
| `_val` | `_config_value` | function | mrm/control/calibration.py |
| `_ctrl_spec_i64` | `_spec_i64_with_params` | function | mrm/control/cli.py |
| `_RUNS` | `_GOLDEN_RUNS` | constant | mrm/control/cli.py |
| `_select` | `_select_cases` | function | mrm/control/cli.py |
| `_E_BITS` | `_ERROR_BITS` | constant | mrm/control/fxp.py |
| `_PI_ACC` | `_P_ACC_BITS` | constant | mrm/control/fxp.py |
| `_PI_ACT` | `_P_HEATER_OFF` | constant | mrm/control/fxp.py |
| `_PI_APPLIED` | `_P_APPLIED_OFF` | constant | mrm/control/fxp.py |
| `_PI_FRAC` | `_P_FRAC` | constant | mrm/control/fxp.py |
| `_PI_HI` | `_P_CODE_MAX` | constant | mrm/control/fxp.py |
| `_PI_KI` | `_P_KI` | constant | mrm/control/fxp.py |
| `_PI_KP` | `_P_KP` | constant | mrm/control/fxp.py |
| `_PI_LATEST` | `_P_LATEST_OFF` | constant | mrm/control/fxp.py |
| `_PI_LO` | `_P_CODE_MIN` | constant | mrm/control/fxp.py |
| `_PI_N` | `_P_N_RINGS` | constant | mrm/control/fxp.py |
| `_PI_NEW` | `_P_NEW_OFF` | constant | mrm/control/fxp.py |
| `_PI_NSET` | `_P_N_SETPOINTS` | constant | mrm/control/fxp.py |
| `_PI_RND` | `_P_ROUNDING` | constant | mrm/control/fxp.py |
| `_PI_SP` | `_P_SETPOINTS` | constant | mrm/control/fxp.py |
| `_PI_VALID` | `_P_VALID_OFF` | constant | mrm/control/fxp.py |
| `_QRE` | `_QFORMAT_RE` | constant | mrm/control/fxp.py |
| `_scalar_or` | `_scalar_or_array` | function | mrm/control/fxp.py |
| `_SI_CAP` | `_S_CAPTURED` | constant | mrm/control/fxp.py |
| `_SI_INT` | `_S_INTEGRATOR` | constant | mrm/control/fxp.py |
| `_SI_U` | `_S_OUTPUT` | constant | mrm/control/fxp.py |
| `_SI_U0` | `_S_U0` | constant | mrm/control/fxp.py |
| `_SI_YSET` | `_S_Y_SET` | constant | mrm/control/fxp.py |
| `_SAT` | `_SATURATE` | constant | mrm/control/fxp_mlp.py |
| `H_SIZE_` | `H_TOTAL_SIZE` | constant | mrm/control/fxp_mlp.py |
| `L_RES` | `L_RESERVED` | constant | mrm/control/fxp_mlp.py |
| `_plain` | `_plain_data` | function | mrm/control/knowledge.py |
| `_cfg_key` | `_config_fingerprint` | function | mrm/control/models.py |
| `_ctrl_c_phase` | `_control_phase_params` | attribute | mrm/control/models.py |
| `_DCODE_FD` | `_FD_CODE_STEP` | constant | mrm/control/models.py |
| `_design` | `_model_config` | function | mrm/control/models.py |
| `_DLAM_FD_M` | `_FD_DETUNING_STEP_M` | constant | mrm/control/models.py |
| `_DT_FD_K` | `_FD_TEMP_STEP_K` | constant | mrm/control/models.py |
| `_phase` | `_phase_params` | function | mrm/control/models.py |
| `_PP_CACHE` | `_PLANT_PARAMS_CACHE` | constant | mrm/control/models.py |
| `_thermal_dc` | `_thermal_dc_gain` | function | mrm/control/models.py |
| `_timing` | `_loop_timing` | function | mrm/control/models.py |

## mrm.scenarios

| old | new | kind | file |
|---|---|---|---|
| `_g` | `generators` | module-alias | mrm/scenarios/calibrate.py |
| `lib` | `library` | module-alias | mrm/scenarios/registry.py, tests/validation/test_controllers_on_engine.py, tests/verification/test_aggressor_layout.py, test_bridge.py, test_bridge_c.py, test_controllers_review.py (local), test_ctrl_a_es.py (local), test_ctrl_a_feedforward.py, test_ctrl_a_harness.py (local), test_ctrl_a_scenarios.py, test_ctrl_b_plant.py, test_ctrl_b_bitstat.py, test_ctrl_b_bitstat_density.py, test_ctrl_b_engine.py, test_ctrl_b_ramp_aggressor.py, test_ctrl_b_robust.py, test_ctrl_b_wdm_slow.py, mrm/sim/experiment.py |
| `_diff_operator` | `_difference_operator` | function | mrm/scenarios/calibrate.py |
| `_estimate_settle` | `_estimate_settle_time` | function | mrm/scenarios/calibrate.py |
| `_foster_hat` | `_foster_hat_response` | function | mrm/scenarios/calibrate.py |
| `_gl_nodes` | `_gauss_legendre_01` | function | mrm/scenarios/calibrate.py |
| `_as_world` | `_world_array` | function | mrm/scenarios/generators.py |
| `_check_e` | `_check_element_rows` | function | mrm/scenarios/generators.py |
| `_common_kw` | `_check_options` | function | mrm/scenarios/generators.py |
| `_comp_next_edge` | `_component_next_break` | function | mrm/scenarios/generators.py |
| `_comp_value` | `_component_value` | function | mrm/scenarios/generators.py |
| `_compile` | `_compile_scenario` | function | mrm/scenarios/generators.py |
| `_crc20` | `_rng_slot` | function | mrm/scenarios/generators.py |
| `_next_edge_x` | `_raw_next_break` | function | mrm/scenarios/generators.py |
| `_per_element` | `_is_per_element` | function | mrm/scenarios/generators.py |
| `_pmod` | `_mod_positive` | function | mrm/scenarios/generators.py |
| `_pwl_seg` | `_pwl_segment` | function | mrm/scenarios/generators.py |
| `_raw` | `_raw_value` | function | mrm/scenarios/generators.py |
| `_rl_value` | `_random_level_value` | function | mrm/scenarios/generators.py |
| `_rows_elems` | `_as_rows_elements` | function | mrm/scenarios/generators.py |
| `_rw_value` | `_random_walk_value` | function | mrm/scenarios/generators.py |
| `_ticks` | `_to_ticks` | function | mrm/scenarios/generators.py |
| `_ticks_arr` | `_to_ticks_array` | function | mrm/scenarios/generators.py |
| `_C0` | `C0 (from mrm.units)` | constant | mrm/scenarios/library.py, mrm/scenarios/calibrate.py |
| `_k` | `_c_to_k` | function | mrm/scenarios/library.py |
| `_T0` | `T0_K (from mrm.units)` | constant | mrm/scenarios/library.py |
| `_getter` | `_config_getter` | function | mrm/scenarios/registry.py |

## mrm.waveform

| old | new | kind | file |
|---|---|---|---|
| `_ERF_10_90` | `_RISE_10_90_PER_SIGMA` | constant | mrm/waveform/driver.py |
| `_interp1` | `_interp_held` | function | mrm/waveform/driver.py |
| `_interp2` | `_interp_held_pair` | function | mrm/waveform/driver.py |
| `_phi` | `_std_normal_cdf` | function | mrm/waveform/driver.py |
| `_rhs` | `_network_rhs` | function | mrm/waveform/driver.py |
| `_AX6` | `_TAB6_AXES` | constant | mrm/waveform/lut.py |
| `_locate` | `_locate_cell` | function | mrm/waveform/lut.py |
| `_lor` | `_lorentzian` | function | mrm/waveform/lut.py |
| `_node_nb` | `_normalized_node_nb` | function | mrm/waveform/lut.py |
| `_ST_NAMES` | `_STATUS_NAMES` | constant | mrm/waveform/lut.py |
| `p2_6_prbs_check` | `prbs_mean_correction_check` | function | mrm/waveform/lut.py |

## mrm.validation

| old | new | kind | file |
|---|---|---|---|
| `tio` | `targets_io` | module-alias | mrm/validation/replicas.py, mrm/validation/report.py, tests/calibration/test_fit_machinery.py, tests/closure/test_targets_schema.py, tests/verification/test_validation_isolation.py |
| `ex` | `experiment` | module-alias | mrm/validation/report.py, webviz/generate.py |
| `rp` | `replicas` | module-alias | mrm/validation/report.py |
| `sens` | `sensitivity` | module-alias | mrm/validation/report.py |
| `_run` | `_report_run` | function | mrm/validation/cli.py |
| `_setup` | `_report_setup` | function | mrm/validation/cli.py |
| `Oracle._last` | `Oracle._last_codes` | attribute | mrm/validation/oracle.py |
| `Oracle._moved` | `Oracle._conditions_moved` | function | mrm/validation/oracle.py |
| `Oracle.conds` | `Oracle.design_conditions` | attribute | mrm/validation/oracle.py |
| `Oracle.cR` | `Oracle.ring_k_per_state` | attribute | mrm/validation/oracle.py |
| `Oracle.d_prev` | `Oracle._d_prev_pm` | attribute | mrm/validation/oracle.py |
| `Oracle.integ` | `Oracle.integral_pm` | attribute | mrm/validation/oracle.py |
| `Oracle.ki` | `Oracle.ki_per_step` | attribute | mrm/validation/oracle.py |
| `Oracle.lagmaps` | `Oracle.lag_maps` | attribute | mrm/validation/oracle.py |
| `Oracle.tol` | `Oracle.retarget_tol` | attribute | mrm/validation/oracle.py |
| `Oracle.u_line` | `Oracle.delay_line` | attribute | mrm/validation/oracle.py |
| `_args` | `_replica_args` | function | mrm/validation/replicas.py |
| `_artefact` | `_load_artefact` | function | mrm/validation/replicas.py |
| `_cfg_path` | `_config_file` | function | mrm/validation/replicas.py |
| `_db` | `_to_db` | function | mrm/validation/replicas.py |
| `_DRIVER_LAWS` | `(removed: unused)` | constant | mrm/validation/replicas.py |
| `_EO_X` | `_EO_DETUNING_X` | constant | mrm/validation/replicas.py |
| `_FS_KEY` | `_FULL_SCALE_KEY` | constant | mrm/validation/replicas.py |
| `_full` | `_writeable_full` | function | mrm/validation/replicas.py |
| `_is_pp` | `_is_plant_params` | function | mrm/validation/replicas.py |
| `_pp_meta` | `_plant_meta` | function | mrm/validation/replicas.py |
| `_sweep` | `_laser_sweep` | function | mrm/validation/replicas.py |
| `_trajectory_c` | `_victim_trajectory_c` | function | mrm/validation/replicas.py |
| `_unit` | `_target_unit` | function | mrm/validation/replicas.py |
| `_VAL8_KEYS` | `_VAL8_TARGET_IDS` | constant | mrm/validation/replicas.py |
| `_values_scalar` | `_plant_value` | function | mrm/validation/replicas.py |
| `StaticRing._drv` | `StaticRing._driver_heat_params` | attribute | mrm/validation/replicas.py |
| `StaticRing._mon` | `StaticRing._monitor_hw` | attribute | mrm/validation/replicas.py |
| `StaticRing._table` | `StaticRing._dac_transfer` | function | mrm/validation/replicas.py |
| `StaticRing._u_table` | `StaticRing._dac_table` | attribute | mrm/validation/replicas.py |
| `StaticRing.alpha` | `StaticRing.tcr_per_k` | attribute | mrm/validation/replicas.py |
| `StaticRing.bit_err` | `StaticRing.bit_err_lsb` | attribute | mrm/validation/replicas.py |
| `StaticRing.bow` | `StaticRing.inl_bow_lsb` | attribute | mrm/validation/replicas.py |
| `StaticRing.r0` | `StaticRing.r_cold_ohm` | attribute | mrm/validation/replicas.py |
| `StaticRing.t_r0` | `StaticRing.tcr_ref_k` | attribute | mrm/validation/replicas.py |
| `StaticRing.tab` | `StaticRing.junction_tables` | attribute | mrm/validation/replicas.py |
| `_band` | `_corner_band` | function | mrm/validation/report.py |
| `_closure_model` | `_closure_model_column` | function | mrm/validation/report.py |
| `_contamination` | `_section_contamination` | function | mrm/validation/report.py |
| `_dec` | `_json_restore` | function | mrm/validation/report.py |
| `_enc` | `_json_safe` | function | mrm/validation/report.py |
| `_fmt` | `_fmt_value` | function | mrm/validation/report.py |
| `_git` | `_git_stamp` | function | mrm/validation/report.py |
| `_log_default` | `_log_to_stderr` | function | mrm/validation/report.py |
| `_measured` | `_measured_text` | function | mrm/validation/report.py |
| `_meta` | `_part_meta` | function | mrm/validation/report.py |
| `_mpl` | `_pyplot` | function | mrm/validation/report.py |
| `_now` | `_utc_now` | function | mrm/validation/report.py |
| `_num` | `_fmt_num` | function | mrm/validation/report.py |
| `_part_line` | `_part_table_row` | function | mrm/validation/report.py |
| `_record` | `_prediction_record` | function | mrm/validation/report.py |
| `_rel` | `_docs_relpath` | function | mrm/validation/report.py |
| `_save` | `_save_svg` | function | mrm/validation/report.py |
| `_scales` | `_residual_scales` | function | mrm/validation/report.py |
| `_sens_experiment` | `_sensitivity_experiment_yaml` | function | mrm/validation/report.py |
| `_transform` | `_apply_transform` | function | mrm/validation/report.py |
| `1.959963984540054 (literal)` | `_Z_95` | constant | mrm/validation/sensitivity.py |
| `_indices` | `_saltelli_jansen` | function | mrm/validation/sensitivity.py |
| `_trunc_z` | `_truncation_z` | function | mrm/validation/sensitivity.py |
| `warnings_ignored` | `_warnings_ignored` | function | mrm/validation/sensitivity.py |
| `_target_from` | `_target_from_entry` | function | mrm/validation/targets_io.py |
| `_z` | `_z_score` | function | mrm/validation/targets_io.py |
| `_z_elementwise` | `_max_z_score` | function | mrm/validation/targets_io.py |

## mrm.calibration

| old | new | kind | file |
|---|---|---|---|
| `R` | `replicas` | module-alias | mrm/calibration/groups.py, mrm/calibration/report.py |
| `G` | `fit_groups` | module-alias | mrm/calibration/pipeline.py, mrm/calibration/preset.py |
| `P` | `preset` | module-alias | mrm/calibration/report.py |
| `(literal 0.02 in fit)` | `_DISTINCT_MINIMUM_DU` | constant | mrm/calibration/core.py |
| `(literal 1e3 in target_residuals)` | `_NONFINITE_RESIDUAL` | constant | mrm/calibration/core.py |
| `(literal 20000 in Problem.predictions)` | `_CACHE_MAX_ENTRIES` | constant | mrm/calibration/core.py |
| `(literal 0.9225 / 1.084 in coupling_er_problems)` | `_X0_OF_0V_NOTCH` | constant | mrm/calibration/groups.py |
| `(literal 1.075 in modulation_depth_scan and pipeline._closing_rows)` | `CAL14_BIAS_V` | constant | mrm/calibration/groups.py |
| `BRANCH_START` | `BRANCH_START_ALPHA_OTHER_DB_PER_CM` | constant | mrm/calibration/groups.py |
| `Context._files` | `Context._widened_children` | attribute | mrm/calibration/groups.py |
| `Context._scratch` | `Context._scratch_path` | function | mrm/calibration/groups.py |
| `Context._tmp` | `Context._scratch_dir` | attribute | mrm/calibration/groups.py |
| `DB_PER_CM` | `DB_PER_CM_TO_PER_M` | constant | mrm/calibration/groups.py |
| `P_C0` | `LEAF_CAPACITANCE_0V` | constant | mrm/calibration/groups.py |
| `P_COUPLING` | `LEAF_ALPHA_OTHER` | constant | mrm/calibration/groups.py |
| `P_CT_HEATER` | `LEAF_DLN_ALPHA_DT_HEATER` | constant | mrm/calibration/groups.py |
| `P_CT_RING` | `LEAF_DLN_ALPHA_DT_RING` | constant | mrm/calibration/groups.py |
| `P_DISP` | `LEAF_COUPLER_DISPERSION` | constant | mrm/calibration/groups.py |
| `P_EFF` | `LEAF_JUNCTION_EFF` | constant | mrm/calibration/groups.py |
| `P_FILL` | `LEAF_FILL_FRACTION` | constant | mrm/calibration/groups.py |
| `P_GAIN` | `LEAF_MONITOR_GAIN` | constant | mrm/calibration/groups.py |
| `P_LINE` | `LEAF_LINE_OFFSET` | constant | mrm/calibration/groups.py |
| `P_OFFSET` | `LEAF_RESONANCE_OFFSET` | constant | mrm/calibration/groups.py |
| `P_PFSR` | `LEAF_P_FSR` | constant | mrm/calibration/groups.py |
| `P_Q` | `LEAF_Q_LOADED` | constant | mrm/calibration/groups.py |
| `P_R0` | `LEAF_R_COLD` | constant | mrm/calibration/groups.py |
| `P_RATIO` | `LEAF_HEATER_SELF_RATIO` | constant | mrm/calibration/groups.py |
| `P_RS` | `LEAF_SERIES_RESISTANCE` | constant | mrm/calibration/groups.py |
| `P_SWING` | `LEAF_DRIVE_SWING` | constant | mrm/calibration/groups.py |
| `P_VFS` | `LEAF_HEATER_FULL_SCALE` | constant | mrm/calibration/groups.py |
| `_closing_rows` | `_cal14_closing_drive` | function | mrm/calibration/pipeline.py |
| `_crossing` | `_interpolate_crossing` | function | mrm/calibration/pipeline.py |
| `_decision` | `_leaf_decision` | function | mrm/calibration/pipeline.py |
| `_fit_all` | `_fit_hypotheses` | function | mrm/calibration/pipeline.py |
| `_fit_targets` | `_target_summaries` | function | mrm/calibration/pipeline.py |
| `_fmt` | `_fmt_sig` | function | mrm/calibration/pipeline.py |
| `_gate` | `_tolerance_gate` | function | mrm/calibration/pipeline.py |
| `_group_coupling` | `_group_coupling_er` | function | mrm/calibration/pipeline.py |
| `_hull` | `_interval_hull` | function | mrm/calibration/pipeline.py |
| `_identifiable` | `_identifiability` | function | mrm/calibration/pipeline.py |
| `_leaf_info` | `_leaf_value_and_provenance` | function | mrm/calibration/pipeline.py |
| `_rejected` | `_rejected_hypotheses` | function | mrm/calibration/pipeline.py |
| `_residuals` | `_residual_summary` | function | mrm/calibration/pipeline.py |
| `coupling_worlds(byname, fr, profs=)` | `coupling_worlds(problem_by_hyp, nominal_fit, profiles=)` | function | mrm/calibration/pipeline.py |
| `_get` | `_get_path` | function | mrm/calibration/preset.py |
| `_header` | `_preset_header` | function | mrm/calibration/preset.py |
| `_ident_block` | `_identifiable_block` | function | mrm/calibration/preset.py |
| `_round` | `_round_sig` | function | mrm/calibration/preset.py |
| `_set` | `_set_path` | function | mrm/calibration/preset.py |
| `_VARIATION_PSEUDO` | `_VARIATION_PSEUDO_PATHS` | constant | mrm/calibration/preset.py |
| `_one` | `_replica_record` | function | mrm/calibration/report.py |
| `_short` | `_fmt_residual` | function | mrm/calibration/report.py |

## mrm.extraction

| old | new | kind | file |
|---|---|---|---|
| `MO` | `modes` | module-alias | mrm/extraction/coupling.py |
| `M` | `materials` | module-alias | mrm/extraction/junction.py, tests/verification/test_materials.py |
| `C` | `common` | module-alias | mrm/extraction/modes.py, mrm/extraction/coupling.py |
| `mt` | `materials` | module-alias | mrm/extraction/thermal_fem.py |
| `_pml_s` | `_pml_stretch` | function | mrm/extraction/common.py |
| `CouplerModel.st` | `CouplerModel.settings` | attribute | mrm/extraction/coupling.py |
| `MEEP2D` | `MEEP2D_SCRIPT` | constant | mrm/extraction/coupling.py |
| `_cmd` | `_extraction_command` | function | mrm/extraction/junction.py |
| `_f` | `_sig` | function | mrm/extraction/junction.py |
| `_fl` | `_sigs` | function | mrm/extraction/junction.py |
| `_pick` | `_at` | function | mrm/extraction/junction.py |
| `_summ` | `_run_summary` | function | mrm/extraction/junction.py |
| `build_reports(..., mp, jac)` | `build_reports(..., mode_props, jac)` | function | mrm/extraction/junction.py |
| `dv (parameter of carriers_per_m, depletion_width_nm, plasma_fields, overlap, direct_resolve_dneff)` | `sweep` | function | mrm/extraction/junction.py |
| `ModeResult._bdg` | `ModeResult._weight_basis` | attribute | mrm/extraction/junction.py |
| `ModeResult._wdg` | `ModeResult._weight_dofs` | attribute | mrm/extraction/junction.py |
| `modes_consistency(art, ma, msd)` | `modes_consistency(art, modes_art, modes_side)` | function | mrm/extraction/junction.py |
| `weight_field(mp, xs, d)` | `weight_field(mode_props, xs, d)` | function | mrm/extraction/junction.py |
| `WeightField._f` | `WeightField._interpolator` | attribute | mrm/extraction/junction.py |
| `decea_compare(ex, R, n_b, ng_b, iw, fr)` | `decea_compare(ex, R, n_b, ng_b, weights, fractions)` | function | mrm/extraction/modes.py |
| `Extractor._probs` | `Extractor._problems` | attribute | mrm/extraction/modes.py |
| `Extractor.lam0` | `Extractor.lam0_um` | attribute | mrm/extraction/modes.py |
| `Extractor.st` | `Extractor.settings` | attribute | mrm/extraction/modes.py |
| `K0C` | `T0_K` | constant | mrm/extraction/modes.py, mrm/extraction/coupling.py |
| `_Basis` | `_GrowingBasis` | class | mrm/extraction/mor.py |
| `_design` | `_foster_design` | function | mrm/extraction/mor.py |
| `_bin_points(r, z, wo, ws, h, info)` | `_bin_points(r, z, w_obs, w_src, bin_um, info)` | function | mrm/extraction/thermal_fem.py |
| `FSR_PFSR` | `HEATER_EFFICIENCY_NM_PER_MW` | constant | mrm/extraction/thermal_fem.py |
| `imec_run(materials=)` | `imec_run(material_set=)` | function | mrm/extraction/thermal_fem.py |
| `key_metrics(A, Bm, ...)` | `key_metrics(ring_model, site_model, ...)` | function | mrm/extraction/thermal_fem.py |
| `mw_c (parameter of build_ring_model, build_site_model, convergence_study, observer_points, ring_points_3d, run_3d)` | `mw_coarse` | function | mrm/extraction/thermal_fem.py |
| `NN_FLOOR` | `NONCOLLOCATED_BAND_FLOOR` | constant | mrm/extraction/thermal_fem.py |
| `ThermalConfig.t_f` | `ThermalConfig.film_thick_um` | attribute | mrm/extraction/thermal_fem.py |
| `ThermalConfig.t_s` | `ThermalConfig.slab_thick_um` | attribute | mrm/extraction/thermal_fem.py |
| `ThermalConfig.w` | `ThermalConfig.rib_width_um` | attribute | mrm/extraction/thermal_fem.py |

## mrm (top level)

| old | new | kind | file |
|---|---|---|---|
| `_LOADED` | `_PROVIDERS_LOADED` | constant | mrm/cli.py |
| `_arr` | `_as_array` | function | mrm/materials.py |
| `_c` | `_coeffs` | function | mrm/materials.py |
| `_corner` | `_corner_value` | function | mrm/materials.py |
| `_frey_parts` | `_frey2006_parts` | function | mrm/materials.py |
| `_lam_um` | `_wavelength_um` | function | mrm/materials.py |
| `_li_parts` | `_li1980_parts` | function | mrm/materials.py |
| `_out` | `_float_or_array` | function | mrm/materials.py |
| `_src` | `_source_and_provenance` | function | mrm/materials.py |

## tools

| old | new | kind | file |
|---|---|---|---|
| `_op / _ph / _el / K` | `optics / phase / electrical / kernel` | module-alias | tools/perf_profile.py |
| `g / lib` | `generators / library` | module-alias | tools/perf_scenarios.py, tools/perf_scenarios_adv.py |
| `_caches` | `_cache_files` | function | tools/clear_numba_cache.py |
| `DEFAULT` | `KERNEL_MODULES` | constant | tools/clear_numba_cache.py |
| `_fmt` | `_value_text` | function | tools/gen_variable_map.py |
| `_split_by` | `_split_updated_by` | function | tools/gen_variable_map.py |
| `_tex` | `_md_math` | function | tools/gen_variable_map.py |
| `LINE` | `BENCH_LINE` | constant | tools/perf_ab.py |
| `run` | `run_bench` | function | tools/perf_ab.py |
| `_plant` | `bench_plant` | function | tools/perf_bench.py |
| `_cfg` | `load_g26` | function | tools/perf_scenarios.py |
| `_rec` | `record_specs` | function | tools/perf_scenarios.py |
| `KW` | `PROVENANCE_KW` | constant | tools/perf_scenarios_adv.py |

## webviz

| old | new | kind | file |
|---|---|---|---|
| `ph / th / ex` | `phase / thermal / experiment` | module-alias | webviz/check_data.py |
| `el / th / gen` | `electrical / thermal / generators` | module-alias | webviz/derived.py |
| `creg` | `(removed: unused import)` | module-alias | webviz/generate.py |
| `dv` | `derived` | module-alias | webviz/generate.py |
| `el / op / ph / sn / th` | `electrical / optics / phase / sensing / thermal` | module-alias | webviz/generate.py |
| `gen` | `generators` | module-alias | webviz/generate.py |
| `msm` | `metrics` | module-alias | webviz/generate.py |
| `sreg` | `scenario_registry` | module-alias | webviz/generate.py |
| `tfem` | `thermal_fem` | module-alias | webviz/generate.py |
| `_arr` | `_none_to_nan` | function | webviz/check_data.py |
| `_orig / _wrap` | `_runner_run / _run_and_keep` | function | webviz/check_data.py |
| `D` | `DATA_DIR` | constant | webviz/check_data.py |
| `(literal 6 and 3 in group_matrix)` | `_XT_AGGRESSOR / _XT_OTHER_OFFSET` | constant | webviz/derived.py |
| `_f` | `_sig_str` | function | webviz/derived.py |
| `_pack` | `_pack_stats` | function | webviz/derived.py |
| `_quant` | `_quantity_entry` | function | webviz/derived.py |
| `_Series` | `_SampleSeries` | class | webviz/derived.py |
| `_Series.r` | `_SampleSeries.record` | function | webviz/derived.py |
| `_stats` | `_statistics` | function | webviz/derived.py |
| `_t_frac` | `_step_response_time` | function | webviz/derived.py |
| `STATS_REC` | `STATS_RECORDS` | constant | webviz/derived.py |
| `_agg_cal` | `_aggressor_calibration_json` | function | webviz/generate.py |
| `_bist_scenario` | `_bistability_scenario` | function | webviz/generate.py |
| `_fin` | `_finite_num` | function | webviz/generate.py |
| `_frames` | `_device_frame_reducer` | function | webviz/generate.py |
| `_none` | `_nan_to_null` | function | webviz/generate.py, webviz/derived.py |
| `_prov` | `_provenance` | function | webviz/generate.py |
| `_row_n` | `_per_ring_row` | function | webviz/generate.py |
| `_TapPlant._cap` | `_TapPlant._capture` | function | webviz/generate.py |
| `_TapPlant._dev_row` | `_TapPlant._device_row` | function | webviz/generate.py |
| `db` | `to_db` | function | webviz/generate.py |
| `DEV_LF` | `DEVICE_LINE_FLOAT_ROWS` | constant | webviz/generate.py |
| `DEV_RF` | `DEVICE_RING_FLOAT_ROWS` | constant | webviz/generate.py |
| `DEV_RI` | `DEVICE_RING_INT_ROWS` | constant | webviz/generate.py |
| `HALF_NM` | `SPECTRUM_HALF_NM` | constant | webviz/generate.py |
| `N_SPEC` | `SPECTRUM_POINTS` | constant | webviz/generate.py |
| `rnd` | `round_sig` | function | webviz/generate.py, webviz/derived.py |
| `RUNNING` | `RUNNING_METRICS` | constant | webviz/generate.py |
| `sc_of` | `scenario_of` | function | webviz/generate.py |
| `Spectra.optics` | `Spectra.phase_rows` | function | webviz/generate.py |
| `acc` | `framesDue` | attribute | webviz/site/app.js |
| `us` | `formatMicroseconds (exported by ui.js)` | function | webviz/site/js/constants.js, webviz/site/js/panels.js |
| `Rg / D / MD / C` | `ringsOf / devicesOf / settingsOf / toCelsius` | function | webviz/site/js/panels.js, webviz/site/js/parts.js |

## tests

| old | new | kind | file |
|---|---|---|---|
| `G / R` | `groups / replicas` | module-alias | tests/calibration/test_calibrated_preset.py |
| `gold` | `golden_cli` | module-alias | tests/golden/test_golden.py |
| `R / tio / el` | `replicas / targets_io / electrical` | module-alias | tests/validation/test_replicas_static.py |
| `tf` | `thermal_fem` | module-alias | tests/validation/test_thermal_replication.py |
| `op / ph` | `optics / phase` | module-alias | tests/verification/test_bus_loss.py |
| `ch` | `channels` | module-alias | tests/verification/test_core_channels.py |
| `tm` | `timing` | module-alias | tests/verification/test_core_timing.py |
| `H` | `harness` | module-alias | tests/verification/test_ctrl_a_*.py (import test_ctrl_a_harness) |
| `bd / dl / pl` | `bias_dither / dither_lockin / pi_lock` | module-alias | tests/verification/test_ctrl_a_bias_dither.py, test_ctrl_a_dither.py, test_ctrl_a_nojit.py, test_ctrl_a_pi.py |
| `el / op / ph / shm / th` | `electrical / optics / phase / selfheat / thermal` | module-alias | tests/verification/test_ctrl_a_harness.py |
| `scen_registry` | `scenario_registry` | module-alias | tests/verification/test_ctrl_a_harness.py, mrm/control/cli.py, mrm/sim/{bridge,experiment,gym_env, runner}.py, tools/perf_bench.py, tests/perf/test_perf.py |
| `cal / el` | `calibration / electrical` | module-alias | tests/verification/test_ctrl_b_calibration.py |
| `ph (unused import)` | `(removed)` | module-alias | tests/verification/test_ctrl_b_plant.py |
| `ra` | `ramp_acquire` | module-alias | tests/verification/test_ctrl_b_ramp.py |
| `K / MP` | `kalman_lqg / mpc` | module-alias | tests/verification/test_ctrl_c_compiled.py, test_ctrl_c_engine.py, test_ctrl_c_kalman_lqg.py, test_ctrl_c_mpc.py |
| `M` | `models` | module-alias | tests/verification/test_ctrl_c_harness.py, test_ctrl_c_engine.py, test_ctrl_c_models.py, test_ctrl_c_oracle.py |
| `M` | `control_models` | module-alias | tests/verification/test_ctrl_c_kalman_lqg.py, test_ctrl_c_mpc.py (a fixture is named models there) |
| `O` | `oracle` | module-alias | tests/verification/test_ctrl_c_oracle.py |
| `op / ph / K / scen_registry` | `optics / phase / kernel / scenario_registry` | module-alias | tests/verification/test_engine_adapter.py |
| `op / ph / th / sg / scen_registry / K` | `optics / phase / thermal / generators / scenario_registry / kernel` | module-alias | tests/verification/test_engine_physics.py |
| `_rng / se / lib / scen_registry / K` | `rng / sensing / library / scenario_registry / kernel` | module-alias | tests/verification/test_engine_sensing.py |
| `lib / K` | `library / kernel` | module-alias | tests/verification/test_engine_start_review.py |
| `K / K_DT_MAX (import alias)` | `kernel / DEFAULT_SH_DT_MAX_K` | module-alias | tests/verification/test_engine_warm_start.py |
| `lz` | `laser` | module-alias | tests/verification/test_laser.py |
| `o` | `optics` | module-alias | tests/verification/test_optics.py |
| `el / op / ph / shm / th / scen_registry` | `electrical / optics / phase / selfheat / thermal / scenario_registry` | module-alias | tests/verification/test_plant_wiring.py |
| `g / sreg` | `generators / scenario_registry` | module-alias | tests/verification/test_runner_fault_reset.py |
| `sh / th` | `selfheat / thermal` | module-alias | tests/verification/test_selfheat.py |
| `dp / el / op / ph / drv_mod / L / tt` | `datapath / electrical / optics / phase / driver / lut / tcmt_time` | module-alias | tests/verification/test_waveform.py |
| `T` | `TARGETS` | constant | tests/calibration/test_fit_machinery.py, tests/calibration/test_calibrated_preset.py |
| `_nums` | `_numbers_in` | function | tests/closure/test_params_notes.py, tests/closure/test_thermal_notes.py |
| `lv` | `leaf` | function | tests/closure/test_relations.py |
| `T` | `TIMES_S` | constant | tests/closure/test_thermal_crosstalk.py, tests/closure/test_thermal_notes.py |
| `_cfg` | `_load_g26` | function | tests/validation/test_controllers_on_engine.py |
| `_fmt` | `_cell` | function | tests/validation/test_controllers_on_engine.py |
| `_tr` | `_trace` | function | tests/validation/test_controllers_on_engine.py |
| `_lags` | `_lag_terms` | function | tests/verification/test_aggressor_layout.py |
| `_fields` | `_name_dtype_count` | function | tests/verification/test_bridge.py |
| `_synthetic` | `_synthetic_obs` | function | tests/verification/test_bridge_c.py |
| `ic_init / ic_step / ic_reference` | `integrator_init / integrator_step / integrator_reference` | function | tests/verification/test_control_api.py |
| `EX` | `EXAMPLES_DIR` | constant | tests/verification/test_ctrl_a_examples.py |
| `_CACHE` | `_PLACED_CACHE` | constant | tests/verification/test_ctrl_a_harness.py |
| `OWN` | `BASELINE_FILES` | constant | tests/verification/test_ctrl_a_pi.py |
| `ALL` | `ALL_KNOWLEDGE` | constant | tests/verification/test_ctrl_b_calibration.py |
| `CtrlBPlant.thr` | `CtrlBPlant.bitstat_threshold` | attribute | tests/verification/test_ctrl_b_plant.py |
| `_exact` | `_run_declared` | function | tests/verification/test_ctrl_b_robust.py |
| `BIND` | `BIND_ASSIGNMENT` | constant | tests/verification/test_ctrl_b_robust.py |
| `Declare` | `DeclareAssignment` | class | tests/verification/test_ctrl_b_robust.py |
| `_synthetic` | `_supervisor_with_maps` | function | tests/verification/test_ctrl_b_wdm.py |
| `TIGHT` | `TIGHT_SUPPLY` | constant | tests/verification/test_ctrl_b_wdm_slow.py |
| `_SUB` | `_NO_JIT_SCRIPT` | constant | tests/verification/test_ctrl_c_compiled.py |
| `_hw` | `_datapath_hw` | function | tests/verification/test_datapath.py |
| `_heater` | `_heater_params` | function | tests/verification/test_electrical.py |
| `_jt` | `_analytic_junction_table` | function | tests/verification/test_electrical.py |
| `LAM / NG / L_RING / A_FIT` | `LAMBDA_M / N_G / RING_LENGTH_M / SHIFT_AMP_FIT_M` | constant | tests/verification/test_electrical.py |
| `RTH_H / Y_PER_W` | `R_TH_HEATER_K_PER_W / SELF_HEAT_Y_PER_W` | constant | tests/verification/test_electrical.py |
| `(selfheat_views indices 4, 5, 8, 9, 12, 13, 14, 17, 20)` | `SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_REFERENCE_ROUND_TRIP_PHASE, SELF_HEATING_VIEW_INDEX_LEVEL_ONE_REFERENCE_ROUND_TRIP_PHASE, SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION, SELF_HEATING_VIEW_INDEX_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION, SELF_HEATING_VIEW_INDEX_COUPLER_PHASE, SELF_HEATING_VIEW_INDEX_THROUGH_COUPLER_POWER_COUPLING, SELF_HEATING_VIEW_INDEX_DROP_COUPLER_POWER_COUPLING, SELF_HEATING_VIEW_INDEX_REFERENCE_TEMPERATURE, SELF_HEATING_VIEW_INDEX_ROUND_TRIP_BACKSCATTER_COEFFICIENTS` | constant | tests/verification/test_engine_adapter.py |
| `_reference / _outputs / _old_pabs / _old_outputs` | `_run_reference_engine / _copy_outputs / _closed_form_pabs / _closed_form_outputs` | function | tests/verification/test_engine_adapter.py |
| `REC / _same` | `RECORDS / _assert_same_run` | constant / function | tests/verification/test_engine_blocks.py |
| `(selfheat_views indices 4, 5, 8-11, 13-19, 22)` | `SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_REFERENCE_ROUND_TRIP_PHASE, SELF_HEATING_VIEW_INDEX_LEVEL_ONE_REFERENCE_ROUND_TRIP_PHASE, SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION, SELF_HEATING_VIEW_INDEX_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION, SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_ABSORBED_SHARE_OF_INTRINSIC_LOSS, SELF_HEATING_VIEW_INDEX_LEVEL_ONE_ABSORBED_SHARE_OF_INTRINSIC_LOSS, SELF_HEATING_VIEW_INDEX_THROUGH_COUPLER_POWER_COUPLING, SELF_HEATING_VIEW_INDEX_DROP_COUPLER_POWER_COUPLING, SELF_HEATING_VIEW_INDEX_ROUND_TRIP_GROUP_DELAY, SELF_HEATING_VIEW_INDEX_PHASE_RESONANCE_WAVELENGTH_DERIVATIVE, SELF_HEATING_VIEW_INDEX_REFERENCE_TEMPERATURE, SELF_HEATING_VIEW_INDEX_REFERENCE_THERMAL_TUNING_SLOPE, SELF_HEATING_VIEW_INDEX_RELATIVE_THERMAL_TUNING_CURVATURE, SELF_HEATING_VIEW_INDEX_RING_LINE_OUTPUTS` | constant | tests/verification/test_engine_physics.py |
| `_cfg / _with_defaults / _rows / _sweep / _onset / _folds / _self_lags / _rows_slope / _equilibria` | `_g26_config / _scenario_plus / _selfheat_views / _heater_staircase / _last_index_before_jump / _fold_points / _own_heating_lags / _optics_rows_slope / _static_equilibria` | function | tests/verification/test_engine_physics.py |
| `ONE / TWO` | `ONE_RING / TWO_RINGS` | constant | tests/verification/test_engine_physics.py |
| `lock1` | `one_ring_lock` | fixture | tests/verification/test_engine_physics.py |
| `_load / _edges / _hw` | `_load_config / _bitstat_at_edges / _monitor_hw` | function | tests/verification/test_engine_sensing.py |
| `_det` | `_detuning_pm` | function | tests/verification/test_engine_start_review.py |
| `_KEPT / _ROWS` | `_SETTLE_STATE_ARRAYS / _SETTLE_RING_ROWS` | constant | tests/verification/test_engine_start_review.py |
| `_cfg / _first_diff` | `_g26_config / _first_differing_tick` | function | tests/verification/test_engine_timing.py |
| `li_init / li_fast / li_step` | `lockin_init / lockin_fast / lockin_step` | function | tests/verification/test_engine_timing.py |
| `_metrics` | `_fake_metrics` | function | tests/verification/test_experiment.py |
| `BASE / OLD_KIND` | `BASE_YAML / OLD_KIND_YAML` | constant | tests/verification/test_experiment.py |
| `RUN / MINI_SOAK` | `RUN_YAML / MINI_SOAK_YAML` | constant | tests/verification/test_experiment_runs.py |
| `_conv_results` | `_convergence_results` | function | tests/verification/test_experiments_review.py |
| `MINI_SOAK / QS_CORNER` | `MINI_SOAK_YAML / SURVEY_CORNER_YAML` | constant | tests/verification/test_experiments_review.py |
| `_load` | `_load_artefact` | function | tests/verification/test_extraction_junction.py, test_extraction_modes.py |
| `art / art_default` | `artefact / default_artefact` | fixture | tests/verification/test_extraction_junction.py |
| `Q_L_OK` | `_q_loaded_ok` | function | tests/verification/test_extraction_modes.py |
| `SH` | `OUT_SHIFT_BITS` | constant | tests/verification/test_fxp.py |
| `BIAS_OV` | `BIAS_BY_CONTROLLER` | constant | tests/verification/test_gym_env.py |
| `env2` | `two_world_env` | fixture | tests/verification/test_gym_env.py |
| `_round / _fit / _net / _policy` | `_ref_round / _ref_fit / _random_network / _random_policy` | function | tests/verification/test_gym_env_fxp.py |
| `K / _print` | `kernel / _print_measurement` | module-alias / function | tests/verification/test_gym_env_perf.py |
| `(selfheat_views indices 22, 23, 24)` | `SELF_HEATING_VIEW_INDEX_RING_LINE_OUTPUTS, SELF_HEATING_VIEW_INDEX_RING_OUTPUTS, SELF_HEATING_VIEW_INDEX_LASER_LINE_OUTPUTS` | constant | tests/verification/test_kernel_parity.py, tests/verification/test_optics.py, test_plant_wiring.py |
| `_compare / _fast_count / _refcounts` | `_assert_parity / _count_samples_fast / _array_refcounts` | function | tests/verification/test_kernel_parity.py |
| `_SUBPROCESS` | `_NO_JIT_SCRIPT` | constant | tests/verification/test_kernel_parity.py |
| `integ_init / integ_step` | `integrator_init / integrator_step` | function | tests/verification/test_kernel_parity.py |
| `_bank` | `_laser_bank` | function | tests/verification/test_laser.py |
| `mf / _fake_run` | `manifest / _write_fake_run` | module-alias / function | tests/verification/test_manifest.py |
| `close / _DISPATCH` | `_close / _REFERENCE_FUNCTIONS` | function / constant | tests/verification/test_materials.py |
| `_state / _feed / _drive / _compare` | `_metrics_state / _feed_traces / _drive_scalar_trace / _assert_metrics_equal` | function | tests/verification/test_metrics.py |
| `KEYS` | `COMPARED_METRICS` | constant | tests/verification/test_metrics.py |
| `_SNIPPET` | `_NO_JIT_SCRIPT` | constant | tests/verification/test_optics.py |
| `airy / generic_loop` | `_airy_powers / _dense_loop_solve` | function | tests/verification/test_optics.py |
| `mini / write / leafv` | `_mini_dict / _write_yaml / _leaf` | function | tests/verification/test_params_config.py |
| `d` | `derived` | fixture | tests/verification/test_params_derive.py |
| `mini` | `_mini_dict` | function | tests/verification/test_params_realize.py, test_params_schema.py |
| `_load` | `_load_quiet` | function | tests/verification/test_params_review.py |
| `_run / pparams (fixture)` | `_optics_inputs / phase_params` | function / fixture | tests/verification/test_phase.py |
| `ReferencePlant.args_f / args_i` | `ReferencePlant.packed_f / packed_i` | attribute | tests/verification/test_plant_wiring.py |
| `setup` | `g26_setup` | fixture | tests/verification/test_plant_wiring.py |
| `_sig / _PARITY` | `_signal / _NO_JIT_PARITY_SCRIPT` | function / constant | tests/verification/test_scenarios.py |
| `dbm / one_ring / lorentz / run / y_of / p_heater_for / settle_on / jump_index / single_pole_system / sign_changes` | `_dbm_to_w / _one_ring_system / _lorentzian_bank / _run_open_loop / _detuning_hwhm / _heater_power_for / _settle_on / _jump_index / _single_pole_system / _sign_changes` | function | tests/verification/test_selfheat.py |
| `HW / ZR` | `HWHM / Z_RATIO` | constant | tests/verification/test_selfheat.py |
| `se / _hw / _run` | `sensing / _monitor_hw / _run_monitor` | module-alias / function | tests/verification/test_sensing.py |
| `lib / scenario` | `library / _aggressor_scenario` | module-alias / function | tests/verification/test_vectors.py, test_vectors_equivalence.py |
| `_ivp_reference` | `_tcmt_rhs` | function | tests/verification/test_waveform.py |
| `lut` | `nrz_lut` | fixture | tests/verification/test_waveform.py |
| `_VIEW_OUT_RL, _VIEW_OUT_R, _VIEW_OUT_L, _VIEW_BUS_T2, _VIEW_BUS_T2_OUT` | `optics.SELF_HEATING_VIEW_INDEX_*` | constant | tests/verification/test_{ctrl_a_harness, bus_loss}.py |

## The reading guide

docs/guide/mrm_guide.tex was synced with the code after this pass (23 October 2026).
Every code name that it shows is the new name. The line numbers of its section 'Core lines of code to read,
in order' are for the tree after the readability pass; they drift with later edits. Find a symbol
with grep, for example `grep -n 'def _airy_line' mrm/plant/optics.py`.

## physical quantities - 25 october 2026

physical intermediates now distinguish propagation, attenuation exponents, field coefficients, power fractions, and rates.
derivative names identify both quantities.
see [physical quantity names](QUANTITY_NAMES.md) for examples and the compatibility rules.
the function-by-function mapping is in the work folder named by `/tmp/mrm-quantity-names-workdir`.

## explicit optics storage indices - 26 october 2026

optics storage constants now identify the output array, workspace, header, cache, or view and the stored quantity.
for example, the reflected-power response row is `RING_RESPONSE_ROW_REFLECTED_POWER_FRACTION`.
the absorbed-power output row is `RING_LINE_OUTPUT_ROW_ABSORBED_POWER`.
all storage declarations are grouped at the top of optics.py, with the original positions and layout.
the complete mapping is in the work folder named by `/tmp/mrm-optics-row-names-workdir`.

## physics readability - 29 october 2026

previous names remain available as aliases. the stored row positions and physical calculations are unchanged.

| old | new | module |
|---|---|---|
| `_interpolated_scatterer_amplitude` | `interpolate_scatterer_reflection_and_phase_derivative` | optics.py |
| `_quarter_ring_energy_and_loss` | `calculate_quarter_ring_energy_per_power_and_loss_fraction` | optics.py |
| `_exact_line` | `solve_exact_ring_fields_and_energy` | optics.py |
| `_exact_line_d` | `solve_exact_ring_fields_and_phase_derivatives` | optics.py |
| `_phase_parts` | `calculate_propagation_and_scatterer_coefficients` | optics.py |
| `_exact_stages` | `solve_ring_fields_at_all_laser_wavelengths` | optics.py |
| `_exact_stages_slope` | `solve_ring_fields_and_loss_phase_derivative_at_all_laser_wavelengths` | optics.py |
| `ring_rows_nb` | `evaluate_ring_response_per_input_watt` | optics.py |
| `dpabs_dlambda_res_nb` | `evaluate_ring_absorbed_power_and_resonance_wavelength_derivative` | optics.py |
| `pabs_bounds_nb` | `bound_ring_absorbed_power_and_phase_curvature` | optics.py |
| `pabs_slope_bound_nb` | `bound_ring_absorbed_power_phase_derivative` | optics.py |
| `bus_finalize_nb` | `write_bus_output_and_returned_reflection_powers` | optics.py |
| `_bus_pout_nb` | `write_bus_output_powers` | optics.py |
| `_bus_refl_nb` | `write_returned_reflection_power_and_forward_error_bound` | optics.py |
| `bus_propagate_nb` | `propagate_bus_optical_power_into_output_buffers` | optics.py |
| `_selfheating_input_views` | `unpack_self_heating_optical_inputs` | optics.py |
| `_selfheating_output_views` | `unpack_self_heating_optical_outputs` | optics.py |
| `selfheat_views` | `unpack_self_heating_optical_state` | optics.py |
| `selfheat_bounds` | `bound_absorbed_power_temperature_response` | optics.py |
| `selfheat_bounds_general` | `bound_absorbed_power_temperature_response_from_ring_parameters` | optics.py |
| `pack_selfheat_args` | `pack_self_heating_optical_state` | optics.py |
| `_evaluate_ring_input_array` | `evaluate_ring_response_for_input_samples` | optics.py |
| `ring_exact` | `evaluate_exact_ring_response` | optics.py |
| `bus_t2_from_plant_params` | `calculate_bus_segment_power_transmission` | phase.py |
| `thermal_shift_m` | `calculate_thermal_resonance_shift_m` | phase.py |
| `local_thermo_optic` | `rebase_thermal_tuning_coefficients` | phase.py |
| `resonance_shift_row` | `evaluate_ring_resonance_shifts` | phase.py |
| `resonance_shift_ws` | `evaluate_ring_resonance_shifts_with_workspace` | phase.py |
| `_loss_temperature_factor` | `calculate_intrinsic_loss_temperature_factor` | phase.py |
| `_fill_level` | `write_ring_optical_inputs_for_data_level` | phase.py |
| `ring_optics_levels_row` | `evaluate_binary_ring_optical_inputs` | phase.py |
| `ring_optics_levels_ws` | `evaluate_binary_ring_optical_inputs_with_workspace` | phase.py |
| `coupling_row` | `evaluate_coupler_power_fractions` | phase.py |
| `ring_optics_row` | `evaluate_ring_optical_inputs` | phase.py |
| `foh_scalar` | `calculate_linear_hold_lag_weights` | thermal.py |
| `foh_coefficients` | `calculate_linear_hold_coefficients` | thermal.py |
| `zoh_coefficients` | `calculate_constant_hold_coefficients` | thermal.py |
| `thermal_step_b` | `advance_thermal_state_for_world` | thermal.py |
| `thermal_step_h_b` | `advance_thermal_state_for_world_with_step_size` | thermal.py |
| `thermal_outputs_b` | `write_observer_temperatures_for_world` | thermal.py |
| `_row` | `shared_or_world_row_index` | thermal.py |
| `_check_batch` | `validate_thermal_step_array_shapes` | thermal.py |
| `thermal_step` | `advance_thermal_states` | thermal.py |
| `thermal_outputs` | `write_observer_temperatures` | thermal.py |
| `thermal_advance_b` | `advance_thermal_state_and_temperatures_for_world` | thermal.py |
| `thermal_advance` | `advance_thermal_states_and_temperatures` | thermal.py |
| `observer_split_b` | `separate_known_heat_and_source_gain_for_observer` | thermal.py |
| `_sidecar_path` | `build_rom_sidecar_path` | thermal.py |
| `_find_sidecar` | `find_rom_sidecar_path` | thermal.py |
| `rom_path` | `build_thermal_rom_path` | thermal.py |
| `rom_validation_numbers` | `read_rom_validation_metrics` | thermal.py |
| `_check_rom_quality` | `warn_about_rom_quality` | thermal.py |
| `_xy` | `validate_site_coordinates_um` | thermal.py |
| `crosstalk_tau_s` | `calculate_crosstalk_time_constant_s` | thermal.py |
| `site_fraction` | `calculate_site_coupling_fraction` | thermal.py |
| `_find` | `find_thermal_element_index` | thermal.py |
| `_as_b` | `normalize_world_parameter_vector` | thermal.py |
| `_batch_of` | `infer_world_count` | thermal.py |
| `_hyp_codes` | `encode_thermal_scaling_hypotheses` | thermal.py |
| `_build_elements` | `build_thermal_sources_and_observers` | thermal.py |
| `_ranks` | `calculate_neighbor_ranks_and_distances` | thermal.py |
| `_interim_lags` | `append_interim_heat_path_lags` | thermal.py |
| `_rom_lags` | `append_extracted_heat_path_lags` | thermal.py |
| `_apply_tail` | `replace_package_tail_preserving_dc_gain` | thermal.py |
| `_dc_factor` | `calculate_pair_dc_scale` | thermal.py |
| `_whole_pair_time_scales` | `calculate_shared_pair_time_scales` | thermal.py |
| `_interim_params` | `validate_and_broadcast_interim_parameters` | thermal.py |
| `interim_three_pole` | `assemble_interim_thermal_system` | thermal.py |
| `_select` | `select_thermal_response_lags` | thermal.py |
| `_accumulate` | `sum_selected_thermal_responses` | thermal.py |
| `k_temp_residue_scale` | `calculate_conductivity_residue_scale` | thermal.py |
| `RP_LAM_RES` | `RING_PARAMETER_COLUMN_COLD_RESONANCE_WAVELENGTH` | phase.py |
| `RP_N_EFF` | `RING_PARAMETER_COLUMN_EFFECTIVE_INDEX` | phase.py |
| `RP_N_G` | `RING_PARAMETER_COLUMN_GROUP_INDEX` | phase.py |
| `RP_DNG` | `RING_PARAMETER_COLUMN_GROUP_INDEX_WAVELENGTH_DERIVATIVE` | phase.py |
| `RP_LEN` | `RING_PARAMETER_COLUMN_ROUND_TRIP_LENGTH` | phase.py |
| `RP_T_REF` | `RING_PARAMETER_COLUMN_REFERENCE_TEMPERATURE` | phase.py |
| `RP_S0` | `RING_PARAMETER_COLUMN_THERMAL_TUNING_SLOPE` | phase.py |
| `RP_BETA` | `RING_PARAMETER_COLUMN_RELATIVE_THERMAL_TUNING_CURVATURE` | phase.py |
| `RP_T0` | `RING_PARAMETER_COLUMN_TUNING_EXPANSION_TEMPERATURE` | phase.py |
| `RP_K2T` | `RING_PARAMETER_COLUMN_THROUGH_POWER_COUPLING` | phase.py |
| `RP_K2D` | `RING_PARAMETER_COLUMN_DROP_POWER_COUPLING` | phase.py |
| `RP_DLNK2T` | `RING_PARAMETER_COLUMN_LOG_THROUGH_COUPLING_WAVELENGTH_DERIVATIVE` | phase.py |
| `RP_DLNK2D` | `RING_PARAMETER_COLUMN_LOG_DROP_COUPLING_WAVELENGTH_DERIVATIVE` | phase.py |
| `RP_ALPHA_I` | `RING_PARAMETER_COLUMN_INTRINSIC_POWER_LOSS_PER_LENGTH` | phase.py |
| `RP_ALPHA_NONHEAT` | `RING_PARAMETER_COLUMN_NONABSORPTIVE_POWER_LOSS_PER_LENGTH` | phase.py |
| `RP_CT_RING` | `RING_PARAMETER_COLUMN_LOG_LOSS_RING_TEMPERATURE_DERIVATIVE` | phase.py |
| `RP_CT_HEATER` | `RING_PARAMETER_COLUMN_LOG_LOSS_HEATER_TEMPERATURE_DERIVATIVE` | phase.py |
| `RP_FILL` | `RING_PARAMETER_COLUMN_JUNCTION_FILL_FRACTION` | phase.py |
| `RP_V_REF` | `RING_PARAMETER_COLUMN_REFERENCE_REVERSE_BIAS` | phase.py |
| `RP_FDN_REF` | `RING_PARAMETER_COLUMN_REFERENCE_JUNCTION_INDEX_CHANGE` | phase.py |
| `RP_FAL_REF` | `RING_PARAMETER_COLUMN_REFERENCE_JUNCTION_POWER_LOSS_PER_LENGTH` | phase.py |
| `N_RP` | `RING_PARAMETER_COLUMN_COUNT` | phase.py |
