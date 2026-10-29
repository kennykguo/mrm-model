# ENGINE BRIEF: what the kernel / plant / runner owner needs

> **Superseded by docs/ENGINE.md** (engine and controllers stage integration, 2026-10-09): how the engine works and
> how to use it. This brief is kept for its reference tables (sections 2, 4 and 5: leaf builders, leaf kernel
> signatures, PlantParams consumers). Known gaps 1 (plant/io.py) and 10 (auto substeps) are closed, gap 7 has the
> waveform LUT (docs/handoff/waveform.md section 3, not wired); section 3 step 6's slope `2 pi c t_rt / lam^2` is
> wrong after a heater shift: use `2 pi c t_rt / (lam - shift)^2` (docs/handoff/engine.md section 3 item 6).

For the agent that writes `mrm/sim/kernel.py`, `mrm/sim/kernel_ref.py`, `mrm/plant/plant.py`, `mrm/sim/runner.py`
(and later metrics, experiment, manifest, gym_env, bridge). Written by the foundation-stage integrator on 2026-10-09.
Contract: docs/ARCHITECTURE.md (v2) and the normative sections of docs/research/07_design_critique.md ("07"):
P0-6 (timing), P0-7 (channels), P0-8 (online metrics), P0-9 (compiled kernel, budget), P0-10 (multi-tick, quasi-static,
compression), P1-17 (experiments), P1-18 (NumPy reference), P1-22 (gym). Contract changes made so far are in
docs/CHANGELOG.md; integrator decisions and open requests in docs/handoff/integrator.md.

**Start here:** `tests/verification/test_plant_wiring.py::ReferencePlant` is a working Python plant tick for B
worlds, built only from `PlantParams` and the leaf modules' public adapters, in the event order below. It passes
against the real g26 preset (sensor channels = ControllerSpec channels, energy conservation, heater -> ring by Z,
monitor codes, dark current vs PD temperature, scenario bias). Port it into the compiled kernel; keep it (or
`kernel_ref.py`) as the NumPy reference of 07 P1-18.

Everything below is SI. Shapes: B worlds, N rings, L lines, S temperature sensors, M lags of the thermal system,
O observers, Sx thermal sources. "row" kernels act on one world (use them inside `prange` over B); the batched
forms loop over b.

---------------------------------------------------------------------------------------------------------------------

## 1. Getting the inputs

```python
from mrm import params
from mrm.params.realize import PlantParams            # never `import mrm.params.realize as m` (the attribute is the function)
from mrm.control.api import ControllerSpec
from mrm.scenarios import registry as scen_registry, calibrate_scenario, foster_from_system

cfg = params.load("g26", alts={...}, overrides=[...])           # Config (warns while artefacts fall back)
pp = params.realize(cfg, batch=B, seeds={"device": 0, "uncertainty": 0}, uncertainty="sample"|"lhs"|"nominal",
                    variation="sample"|"off")                   # PlantParams, frozen, SI; pp.select(b) -> B=1 view
sc = scen_registry.from_config(cfg)                              # or scen_registry.create("a1", ...)
spec = ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg,  # cfg has fingerprint()/to_dict()
                                  scenario_drives=sc.drives, scenario_offsets=sc.offsets)
```

- `pp.thermal.system` is the assembled `mrm.plant.thermal.ThermalSystem` (realize calls `thermal.assemble`; check
  it is not None). `pp.thermal_inputs` keeps every assembly input.
- Seeds (07 P1-13, mrm/core/rng.py): element b uses seed `base + b` (`rng.batch_seeds(base, B)`); key
  `rng.stream_key(seed_b, stream_id)`. Device draws: `pp.seeds["device"]`. Noise banks: `experiment.seeds.noise`.
  Scenario: `experiment.seeds.scenario`. Never pool Monte Carlo sets whose base seeds are closer than B.
- Aggressor calibration (07 P1-4, before compiling the scenario) when `sc.pending_aggressors` is not empty:
  ```python
  model = {a.site: {"s0_m_per_k": pp.thermal.dlambda_dt_s0_m_per_k[:, a.victim_ring],
                    "beta_per_k": pp.thermal.dlambda_dt_beta_per_k[:, 0], "t_ref_k": pp.thermal.dlambda_dt_t0_k[:, 0],
                    "ring_t0_k": pp.thermal.t_ring_lock_k[:, a.victim_ring]}   # recommended for controller studies
           for a in sc.aggressors}
  resp = {a.site: foster_from_system(pp.thermal.system, victim_ring=a.victim_ring, site=a.site) for a in sc.aggressors}
  sc, cal = calibrate_scenario(sc, resp, model=model)              # cal[site].record() -> run manifest
  ```
  The aggressor site name must be a layout site of kind `aggressor` (g26: `aggressor`, the library default).
- Compile the scenario at the plant dt with the ROM aggressor columns in order:
  `arr = sc.compile(dt_s=pp.timing.dt_s, n_rings=N, n_lines=L, batch=B, seed=scen_seed,
  aggressor_sites=tuple(sys.src_names[i] for i in sys.aggressor_src))`.

## 2. Builders (Python, once per run; every one takes PlantParams)

| object | builder | notes |
|---|---|---|
| phase params | `mrm.plant.phase.PhaseParams.from_plant_params(pp)` -> `.ring` (B,N,21), `.grid` (G,), `.dneff/.alpha_fca/.capacitance/.r_series` (Bt,G), `.table_row(b)` | junction table: LUT (Bt=1) or analytic per world (Bt=B) via `electrical.junction_table_from_plant_params` |
| heater DAC path | `electrical.HeaterDriverParams.from_plant_params(pp, transport_delay=True)`; state `electrical.HeaterBank(params, initial_code)` | adds the thermal transport delay (ThermalSystem.transport_delay_s) as whole ticks of DAC latency; DAC bit errors from `dac_bit_err_z`, INL bow from `inl_bow_u` |
| bias DAC path | `electrical.BiasDacParams.from_plant_params(pp)`; `electrical.BiasBank(params, initial_code)` | ideal transfer (no bias INL/DNL keys); noise redrawn every `update_period_ticks` |
| monitor PD/AFE/ADC | `sensing.MonitorHw.from_plant_params(pp)`; `sensing.MonitorBank(hw, noise_seeds, pop_period_ticks=controller period)` | baseline, mux order, stuck flag, flicker floor from MonitorParams; DNL by `pp.seeds["device"]` |
| reference tap (optional) | `hw.reference_tap(responsivity_a_per_w=..., dark_i0_a=..., adc_full_scale_a=...)` from `pp.monitor.ref_tap_*` (nan = drop PD's) | no controller channel exists yet (integrator.md section 3); `MonitorHw.from_plant_params` refuses `ref_tap_present` (2026-10-16) |
| temperature sensors | `sensing.TempSensorsHw.from_plant_params(pp)`; `sensing.TempSensorBank(hw, noise_seeds, pop_period_ticks)` | order = `pp.temp_sensors.names` = `sys.temp_obs` order |
| heater readback | `sensing.ReadbackHw.from_plant_params(pp)` (raises when `readback_present` is False); `ReadbackBank` | g26: readback absent |
| data path | `datapath.DatapathHw.from_plant_params(pp)`; `datapath.DatapathBank(hw, noise_seeds, pop_period_ticks)` | BER/sampler hardware from DatapathParams (nan -> module defaults) |
| laser lines | `laser.LaserHw.from_plant_params(pp)`; `laser.LaserBank(hw, scenario_seeds, noise_seeds, time_compression)` | f0 = `pp.laser.line_freq_hz` (grid + static error); drift params stay 0 (the scenario owns drift) |
| thermal | `sys = pp.thermal.system`; `a, c0, c1 = sys.coefficients(h)`; `sys.kernel_arrays()`; `sys.zero_state(B)` | interim 3-pole model in g26; FEM ROM when `fidelity.thermal_model: rom` |
| self-heating rows | `optics.pack_selfheat_args(...)` per world, `np.stack` rows -> `args_f (B,nf)`, `args_i (B,ni)` | section 4.2 |
| scenario | `arr = sc.compile(...)`; `arr.new_cache()`, `arr.new_out()`, `arr.kernel_args(cache)`, `arr.layout[name] -> (offset, width)` | section 4.8 |
| controller | `spec` above; `ActionValidator(spec)`; `CompiledController` / Python controllers | section 4.9 |

## 3. Event order (07 P0-6; mrm/core/timing.py docstring)

Tick k is the instant at the END of interval [k-1, k). Per plant interval [k, k+1), for every world:

1. **Scenario** values at k and k+1 (`scen_eval_b`, or only at `scen_next_break_b` ticks and linear in between: FOH
   is exact by the scenario contract). Signals: section 4.8.
2. **Laser** `laser_tick_row` -> line frequency (B,L) and power at the module output. Chip input power
   `p_chip = p * pp.laser.input_transmission * 10**(-optical.input_loss_delta_db/10)`; `lam = c / f`.
   (`laser.line_enable` multiplies the power inside the bank; `p_scale = 10**(laser.power_delta_db/10)`;
   `f_offset_hz = laser.dfreq_hz`.)
3. **Ones density** `pattern_tick_row` -> rho (N,) (scenario `data.ones_density`, `data.driver_enable` >= 0.5).
4. **Bias DAC** `bias_bank_step_row` (command = the latched controller command or, when the scenario owns `tx.bias`,
   `BiasDacParams.code_for_voltage(tx.bias_v)`; `v_offset = tx.bias_offset_v`; noise z from `rng.fill_normal` /
   `rng.normal` with stream `dac_bias`, slot 0, tick k) -> v_avg (N,).
5. **NRZ levels** `nrz_levels_row(v_avg, swing_pp_v * data.swing_scale, level_convention, -0.5, 4.0, v_lo, v_hi,
   flags)`. Level 1 (weighted by rho) = `v_hi` (integrator assumption). Squelch: swing 0.
6. **Optics inputs** `phase.ring_optics_row` at the PREVIOUS ring temperature (t_ring, t_heater) and (v_lo, v_hi):
   phi0/phi1, a0/a1, eta0/eta1, k2t/k2d at the line, t_rt. Write them into the world's self-heating row
   (`optics.selfheat_views`): `p_in = p_chip`, `p1 = rho`, `phi0_ref/phi1_ref`, `a0/a1`, `eta0/eta1`, `k2_th/k2_dr`,
   `t_rt`, `dphi_dlamres = 2 pi c t_rt / lam^2`, `t_ref = t_ring_prev`, `(s0, beta) = phase.local_thermo_optic(
   t_ring_prev, s0, beta, T0)` (the adapter's quadratic law is then R1 exactly). Then `optics.selfheat_bounds(...)` ->
   p_max, d2p_max, dpdt_max. `psi` stays 0 (the coupler phase is inside the measured cold resonance).
7. **Thermal inputs** u0/u1 (Sx,): junction and driver heat (`junction_swing_capacitance_row` on (v_lo, v_hi), then
   `data_activity_heat_row`), held (u0 = u1); aggressor power at k and k+1 (FOH); heater entries = the PREVIOUS
   tick's heater power (needed by the TCR split for the other heaters); waveguide-absorption entries = previous
   absorbed power (the implicit step ignores them).
8. **Heater** for each ring: `pre, g = thermal.observer_split_b(x, a, c0, c1, lag_src, obs_ptr, sys.heater_obs[n],
   sys.heater_src[n], u0, u1, src_scale)`; then `heater_bank_step_row(..., t_heater_pre = T_base_end + pre,
   g_self = g, code_add = dac_glitch codes or None)` -> heater power out_p (N,) W; write it into u0 = u1 at
   `sys.heater_src`.
9. **Implicit thermal + self-heating step** (07 P0-1) `selfheat.implicit_step_b(x, a, c0, c1, *sys.kernel_arrays(),
   bus_order, u0, u1, src_scale, T_base_end, p_prev, t_prev, optics.selfheat_pabs, optics.selfheat_commit, args_f[b],
   args_i[b], p_max, d2p_max, max_iter=10, tol_k=1e-9, n_outer=1, g_work, t_obs, p_new, t_ring, n_iter)` -> n_fail.
   T_base_end = scenario `thermal.t_base_k + thermal.xpu_rise_k` at k+1 (records keep both). Then
   `p_prev = p_new`, `t_prev = t_ring`, `t_heater = t_obs[sys.heater_obs]`. The optics walk ran inside (commit_fn):
   read `out_rl (RING_LINE_OUTPUT_ROW_COUNT,N,L)`, `out_r (RING_OUTPUT_ROW_COUNT,N)`, `out_l (LASER_LINE_OUTPUT_ROW_COUNT,L)` from `selfheat_views`.
   Self-heating off (`pp.flags.self_heating == 0`): `bus_propagate_nb` at the ring temperatures, then
   `thermal_advance_b(x, a, c0, c1, lag_src, obs_ptr, u0, u1, src_scale, t_base, t_obs)` with u[wg] = 0.
10. **Sensors of the interval**: `monitor_tick_row(k, p_pd = out_r[RING_OUTPUT_ROW_TOTAL_CLOCKWISE_DROP_POWER], p_bg = sum_l out_rl[RING_LINE_OUTPUT_ROW_INPUT_POWER, n, l],
    t_pd = t_obs[sys.pd_obs], t_adc = hw.adc_temperature(t_pd, T_base_end), pga_sel, fault, ...)` (t_adc:
    ring.monitor.adc_ref_site, the package base T_base_end by default, t_pd for monitor_pd; kernel row RF_T_ADC);
    `errors_tick_row(k, oma = |out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ONE_BUS_OUTPUT_POWER, n, line(n)] - out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ZERO_BUS_OUTPUT_POWER, n, line(n)]|, ...)` with
    `line = pp.layout.ring_to_line`; `temp_tick_row(k, t_obs[sys.temp_obs], ...)`; `readback_tick_row(k, heater v,
    i, ...)`.

At a controller edge k (after the interval [k-1, k)): (1) state at k; (2) release every sensor bank
(`monitor_release_row`, `temp_release_row`, `readback_release_row`, `errors_release_row`), run the bit-statistics
window (`MonitorBank.level_codes` -> `bitstat_window_row` on RING_OUTPUT_ROW_LEVEL_ONE_TOTAL_CLOCKWISE_DROP_POWER/RING_OUTPUT_ROW_LEVEL_ZERO_TOTAL_CLOCKWISE_DROP_POWER, threshold = actuator
`datapath.bitstat_threshold`, density = the scenario's `data.ones_density`, not the tick's realized n1/nb), run
the `fast_kernel` once per released monitor sample (07 P0-7); (3) controller step; (4) clip `act_i64` to each
channel's [min, max] (never cast to uint16 first), latch at k + latency_ticks; (5) the DAC FIFOs and slew/settle
live in the electrical rows. `tx.heater.applied` / `tx.bias.applied` are the latched registers (`reg`).

Reset (07 P0-1 item 6, `pp.timing.reset_protocol`): 0 laser_ramp (ramp the line powers over
`pp.timing.reset_ramp_ticks` from the cold state, t_prev = -inf), 1 branch_low (t_prev = -inf), 2 branch_high
(t_prev = +inf). `Scenario.reset` (ResetSpec) says cold or warm start and whether the factory calibration is given.

## 4. Kernel signatures and where each argument comes from

### 4.1 `mrm.plant.phase` (new; integrator)

```
ring_optics_row(rp (N,21), lam_l (L,), t_ring_k (N,), t_heater_k (N,), v0 (N,), v1 (N,),
                grid (G,), dn_tab (G,), al_tab (G,), c_tab (G,), rs_tab (G,),
                out_phi0, out_phi1, out_a0, out_a1, out_eta0, out_eta1, out_k2t, out_k2d, out_trt (N,L),
                out_shift (N,2), out_cj (N,2), out_rs (N,2), out_flags (N,) int8)
resonance_shift_row(rp, t_ring_k, v, grid, dn_tab, al_tab, c_tab, rs_tab, out_shift, out_fal, out_cj, out_rs, out_flags)
thermal_shift_m(t_k, t_ref_k, s0, beta, t0) -> m          local_thermo_optic(t_prev_k, s0, beta, t0) -> (s, beta')
```
`rp = PhaseParams.ring[b]`; table row `r = PhaseParams.table_row(b)`. Zero `out_flags` before the call (OR-ed).
Columns `RP_*` (module constants) come from: optics.lambda_res_cold_m, n_eff, n_g, dn_g_dlambda_per_m, length_m,
t_ref_k, kappa2_through/drop, dln_kappa2_*_dlambda_per_m, alpha_i_per_m, alpha_i - alpha_heat, dln_alpha_dt_ring/
heater_per_k, bias_ref_v; thermal.dlambda_dt_s0_m_per_k, dlambda_dt_beta_per_k, dlambda_dt_t0_k; junction.
fill_fraction; and the table at the reference bias. Equations in the module docstring (rigid translation of the
cold phase curve by the R1 + junction resonance shift; phi = 2 pi m exactly when the resonance reaches the line).

### 4.2 `mrm.plant.optics`

```
pack_selfheat_args(p_in_w (L,), phi_ref (N,L), k2_th, k2_dr, a_rt, t_rt_s, eta (N,L), lam_m (L,), t_ref_k (N,),
                   s0_m_per_k (N,), beta_per_k=0.0, bus_order=None, bus_t2=None, bus_t2_out=1.0, psi=None,
                   rb=None (N,M_b) complex, nrz={"p1": (N,), "phi1", "a1", "eta1"}, method="exact"|"tcmt",
                   m_orders=3, commit_slope=False) -> (args_f, args_i)          # Python, once per world
selfheat_views(args_f, args_i) -> (p_in, bus_t2, bus_t2_out, p1, phi0_ref, phi1_ref, phi0, phi1, a0, a1, eta0, eta1,
                   psi, k2_th, k2_dr, t_rt, dphi_dlamres, t_ref, s0, beta, rb, p_cur, out_rl, out_r, out_l, cw, fw,
                   bus_order, bus_pos)                                           # [njit] 29 views
selfheat_pabs(t, k, args_f, args_i) -> (P_abs, dP/dT)      selfheat_commit(t, k, args_f, args_i)      [njit]
selfheat_bounds(args_f, args_i, dt_max_k, p_max (N,), d2p_max (N,), dpdt_max (N,))                       [njit]
bus_propagate_nb(p_in_w, bus_order, bus_t2, bus_t2_out, p1, phi0, phi1, a0, a1, eta0, eta1, psi, k2_th, k2_dr, t_rt,
                 rb, dphi_dlamres, two_level, use_tcmt, m_orders, with_slope, out_rl, out_r, out_l, work, cw, fw) [njit]
make_bus_outputs(N, L) -> (out_rl, out_r, out_l)    make_workspace(L) -> (work, cw, fw)
```
Inputs: `rb = pp.optics.backscatter_rb[b]` (zeros unless `fidelity.backscatter: on`); `method = "tcmt" if
pp.flags.optics_model else "exact"`, `m_orders = pp.optics.tcmt_orders`; `bus_order = pp.layout.bus_order`;
`bus_t2[k]` = power transmission of the bus segment before ring k = `exp(-pp.laser.bus_loss_per_m *
pp.layout.ring_spacing_on_bus_m)` (1 for the first ring in bus order; g26: 0.073 dB/mm over 50 um, 0.99916 per
segment), `mrm.plant.phase.bus_t2_from_plant_params(pp)` (B, N); `bus_t2_out` = 1 (the output segment is not
modelled). Correction (2026-10-16, physics_optics review): this line said the spacing is absent in g26; it is 50 um,
and until then Plant._build passed no `bus_t2`, so the engine ran every segment at 1 (docs/CHANGELOG.md).
Outputs: `RING_OUTPUT_ROW_TOTAL_CLOCKWISE_DROP_POWER` (monitor input, rho-weighted, every line and order), `RING_OUTPUT_ROW_TOTAL_ABSORBED_POWER` (self-heating source),
`RING_OUTPUT_ROW_LEVEL_ONE_TOTAL_CLOCKWISE_DROP_POWER/0` (per level, for the bit statistics), `RING_LINE_OUTPUT_ROW_LEVEL_ONE_BUS_OUTPUT_POWER/0` (per level at the bus output, OMA), `RING_LINE_OUTPUT_ROW_INPUT_POWER`
(bus power at each ring), `LASER_LINE_OUTPUT_ROW_BUS_OUTPUT_POWER`, `LASER_LINE_OUTPUT_ROW_FORWARD_POWER_RELATIVE_ERROR_BOUND` (single-pass error bound; large only when several rings sit on one
line). Each prange worker needs its own `cw, fw, work` and args row. With `selfheat_bounds` use `dt_max_k` >= the
largest ring temperature change of a step (the wiring test uses 50 K). The engine uses 50 K for every step
(`mrm.plant.plant.DEFAULT_SH_DT_MAX_K`, `KParams.dt_max_k`; nothing scales it with the substep length, and an auto
substep moves a ring by mK). Measured exceptions (physics_thermal review, 2026-10-16): `Plant.settle`'s first DC
iterations (~100 K from branch_low, ~240 K from branch_high; `settle()` reports `max_step_k`) and the first tick after
a reset to a bracket end (~77 K). Without backscatter the 1.25 curvature margin covers |T - T_ref| up to ~136 K at
g26's beta; the g26 settled states equal those of a covering bound (`test_engine_warm_start.py`).

### 4.3 `mrm.plant.thermal`

```
sys.coefficients(h_s) -> (a, c0, c1) (B,M)        thermal.foh_scalar(tau, h) [njit]   (adaptive steps in-kernel)
sys.kernel_arrays() -> (lag_src, lag_obs, lag_kind, opt_lags, obs_ptr, src_ring, ring_obs)
observer_split_b(x, a, c0, c1, lag_src, obs_ptr, obs, self_src, u0, u1, src_scale) -> (pre, g)         [njit]
thermal_advance_b(x, a, c0, c1, lag_src, obs_ptr, u0, u1, src_scale, t_base, out_t_obs)                [njit]
thermal_step_h_b(x, taus, res, lag_src, u0, u1, src_scale, h)   thermal_outputs_b(x, obs_ptr, t_base, out)  [njit]
cross_optical_lag_gain(sys, h_s, dpabs_dt_max (B|1,N)) -> (B,)   assert_cross_lag_gain(sys, h_s, dpdt_max)
```
Index arrays on `sys`: `heater_src, wg_src, junction_src, driver_src, aggressor_src` (sources, Sx = `sys.n_src`),
`ring_obs, heater_obs, pd_obs, temp_obs` (observers). `src_scale` (1|B, Sx) = ones (07 P2-5 k(T) scaling would go
here: `thermal.k_temp_residue_scale`, not wired). Multi-tick steps (07 P0-10): the FOH step is exact for any h;
assert `cross_optical_lag_gain < 0.05` at the largest h, else use `n_outer > 1`.

### 4.4 `mrm.plant.selfheat`

```
implicit_step_b(x, a, c0, c1, lag_src, lag_obs, lag_kind, opt_lags, obs_ptr, src_ring, ring_obs, bus_order,
                u0, u1, src_scale, t_base, p_prev, t_prev, pabs_fn, commit_fn, args_f, args_i, p_max, d2p_max,
                max_iter, tol_k, n_outer, g_work, t_obs, p_new, t_ring, n_iter) -> n_fail                     [njit]
implicit_step(...)  (batched, same arguments with (B, ...) arrays)       SelfHeatWork(sys, batch, t_init_k).step(...)
STATUS_CONVERGED = 0, STATUS_MAX_ITER = 1 (fold crossing), STATUS_BOUND = 2 (p_max does not bound P_abs: caller error)
```
Kernels taking function arguments are not numba-cached: about 2 s of compile per (pabs_fn, commit_fn) pair and
process. Count `n_fail` into the run's diagnostics.

### 4.5 `mrm.plant.electrical`

```
heater_bank_step_row(tick, cmd, cmd_valid, reg, pending, has_pending, last_update, fifo, y_slew, y_out,
    law, bits, pwm_bits, pwm_ripple, dt_clk, bit_err, bow, code_lo, code_hi, latency_ticks, min_update_ticks,
    slew_u_per_s, settle_tau_s, dt, r0, alpha, t0, v_fs, i_fs, p_fs, v_comp, p_max, i_max, t_heater_pre, g_self,
    code_dac, u_target, ripple, u_avg, out_p, out_v, out_i, out_r, out_flags, code_add=None) -> supply droop  [njit]
bias_bank_step_row(tick, cmd, cmd_valid, reg, pending, has_pending, last_update, fifo, y_slew, y_out, noise_hold,
    bits, bit_err, bow, v_min, v_max, latency_ticks, min_update_ticks, noise_period_ticks, noise_rms_codes, z,
    slew_v_per_s, settle_tau_s, dt, v_offset, code_dac, v_target, out_v_avg, out_v_end)                      [njit]
nrz_levels_row(v_bias, swing_vpp, mode, v_floor, v_ceil, out_lo, out_hi, out_flags)
junction_eval_row(v, grid, dneff_col, alpha_col, c_col, rs_col, fill, length_m, out_dneff, out_alpha, out_c_f,
    out_rs_ohm, out_flags)
junction_swing_capacitance_row(v_lo, v_hi, grid, c_col, fill, length_m, out_c_eff)
data_activity_heat_row(c_eff, swing_vpp, bitrate_hz, transition_density, enable, r_s, r_drv, p_static_w, e_tr_j,
    add_load_share, out_pj, out_pd)
actuator_quiescent_row(reg, has_pending, fifo, y_slew, y_out, target, tol) -> bool      (07 P0-10 tick skipping)
settle_advance_row(target, y_slew, y_out, tau, dt, n_ticks)
```
Parameters: every field of `HeaterDriverParams` / `BiasDacParams` (built in section 2), row b; state arrays of
`HeaterBank` / `BiasBank` (`reg, pending, has_pending, last_update, fifo, y_slew, y_out[, noise_hold]`). Data heat:
`bitrate_hz = pp.datapath.bitrate_hz[b,0]`, `transition_density` = scenario `data.transition_density`, `enable =
data.driver_enable >= 0.5`, `r_s` = junction R_s from `ring_optics_row` (`out_rs`), `r_drv, p_static_w, e_tr_j =
pp.datapath.driver_output_resistance_ohm / driver_static_power_w / driver_energy_per_transition_j`, `add_load_share
= 0`. `out_pj` -> `sys.junction_src`, `out_pd` -> `sys.driver_src`. dac_glitch: `code_add = where(fault &
sensing.FAULT_DAC_GLITCH, rint(fault.dac_glitch_codes), 0)`. Tick skipping (P0-10): only while
`actuator_quiescent_row` is True and up to the next command tick, then `settle_advance_row`; never across a bias
noise redraw or with PWM ripple on.

### 4.6 `mrm.plant.sensing`

```
monitor_tick_row(k, p_pd_w, p_bg_w, t_pd_k, t_adc_k, pga_sel, fault, pr, pw, lat, phase, pga_gains, fl_coef,
    afe_type, window_ticks, period_ticks, code_max, shot_on, stuck_flagged, slot_base, key_adc, key_shot, key_dnl,
    sf, si, fx, fc, fv, fr, fm, out_sampled)
monitor_release_row(now, lat, fv, fr, fm, out_samples, out_ticks, out_count, latest, out_valid, out_over, out_new)
monitor_level_codes_row(p1_w, p0_w, p_bg_w, t_pd_k, t_adc_k, pga_sel, pr, pw, pga_gains, shot_on, tau_s,
    out_mu1, out_mu0, out_sig1, out_sig0, s_tia_a2_per_hz=-1.0, rin_per_hz=0.0)
temp_tick_row(k, t_site_k, tp, period, lat, code_max, fl_coef, key, st, fx, fc, primed, fv, fr, fm, out_sampled)
temp_release_row(now, fv, fr, fm, latest, out_new)
readback_tick_row(k, v_v, i_a, lsb_v, lsb_i, noise_codes, period, lat, code_max, average, key, st, fv, fr, fm,
    out_sampled)          readback_release_row(now, fv, fr, fm, latest_v, latest_i, out_new)
fault_mask_row(adc_drop, adc_stuck, adc_saturate, dac_glitch_codes, out (N,) int8)     FAULT_* constants
```
Parameters from the `*Hw` objects (`hw.pr, pw, latency_ticks, phase, pga_gains, fl_coef, afe_type, window_ticks,
period_ticks, code_max, shot_on, stuck_flagged, slot_base`), state from the banks (`sf, si, fx, fc, fv, fr, fm`,
keys `key_adc, key_shot, key_dnl`). `MonitorBank.tick` / `.release` / `.channels` show the exact call and the
channel dicts. Noise is drawn at the sample (dt-invariant); `reset(seed)` re-keys noise, never device draws.
FIFO depth: pass `pop_period_ticks` = the actual release period, or samples are lost (`MonitorBank.fifo_lost`).

### 4.7 `mrm.plant.datapath`

```
pattern_tick_row(k, mode, bpt_i, bpt_f, density, traffic_on, words, pre, period, key, bitpos, out_n1, out_nb, out_rho)
errors_tick_row(k, oma_tx_w, nbits, link_t, oma_sens_w, q_ref, ber_floor, period, lat, key, lam, fv, fr, fm, out_new)
errors_release_row(now, fv, fr, fm, latest, out_valid)
bitstat_window_row(edge, n_win, mode, density, mu1, mu0, sig1, sig0, thr, words, pre, period, bitpos, key,
    out_above, out_n1, out_a1, out_a0)
```
`DatapathBank.pattern / errors / bitstat_from_monitor / release` show the calls; `release` gives `tx.ones_count`,
`tx.bits` (sums), `datapath.bitstat_above`, `rx.error_count` (held) and `rx.error_count_valid`.

### 4.8 `mrm.scenarios`

```
scen_init_b(elem_default, out)
scen_eval_b(tick, compression, sig_i64, sig_f64, elem_default, comp_i64, comp_f64_b, tab_b, keys_b, cache_b, out)
scen_eval_signal_b(sig, tick, compression, sig_i64, sig_f64, elem_default, comp_i64, comp_f64_b, tab_b, keys_b,
    cache_b, out)
scen_next_break_b(sig, tick, compression, sig_i64, comp_i64, comp_f64_b, tab_b) -> int64 (NO_BREAK = 2**62)
```
`bb = b if arr.comp_f64.shape[0] > 1 else 0`; `arr.layout[name] = (offset, width)`. Signals (ids append-only):
`thermal.t_base_k` (1), `thermal.xpu_rise_k` (1, add to T_base for the ROM, record separately),
`thermal.aggressor_w` (A, FOH into `sys.aggressor_src`), `laser.dfreq_hz` (L), `laser.power_delta_db` (L),
`laser.line_enable` (L, factor 0..1), `optical.input_loss_delta_db` (1), `data.ones_density` (N),
`data.transition_density` (N), `data.swing_scale` (N), `data.driver_enable` (N, on >= 0.5), `tx.bias_offset_v` (N),
`tx.bias_v` (N, NaN unless the scenario drives tx.bias), `fault.adc_drop/adc_stuck/adc_saturate` (N, >= 0.5),
`fault.dac_glitch_codes` (N), `fault.controller_reset` (1). Time compression: only `slow` components of t_base and
the laser/optical signals; reporting code must call `require_compression_validated` (07 P0-10).

### 4.9 Controller side (`mrm.control.api`, `mrm.core.channels`, `mrm.core.timing`)

- `spec.channels` (name -> ChannelSpec), `spec.layout` (AbiLayout: `pack_obs(tick, sensors, new)`, `unpack_act(act,
  cast=False)`), `spec.period_ticks`, `phase_ticks`, `samples_per_step` (M), `tick_jitter_ticks`; edges at
  `phase + j*period + timing.edge_jitter(rng.stream_key(seed_b, rng.Stream.SCHEDULE), j, J)`.
- Python path: `Observation(tick, dt_s, sensors, new, active)`; `ActionValidator(spec).validate(actions)`;
  `validate_telemetry`; CompositeController base period `ctrl.base_period_ticks(dt)`.
- Compiled path: `spec_arrays(spec, params_i64, params_f64, latency_ticks)`; `call_step_batch(step_fn, p_i64, p_f64,
  s_i64, s_f64, obs_i64, act_i64)`; pre-fill `act_i64` from the applied registers at reset; `obs_offset /
  act_offset / user_i64 / user_f64`. fast_kernel: `fobs = [tick, N, new[N], code[N], sample_tick[N]]`, `fact =
  [heater_offset[N], bias_offset[N]]` (`fast_obs_size`, `fast_act_size`, `fo_*`, `fa_*`), once per released
  monitor sample before `step`; offsets added to the latched command before the DAC FIFO and clipped.
- Timing helpers: `timing.first_response_tick(k, ctrl_latency, dac_latency, adc_period, adc_phase, adc_latency,
  ctrl_period, ctrl_phase)` (for `test_loop_delay_impulse`), `fifo_capacity(latency, push_period, pop_period)`.
- The sensors the plant must release (g26): `tx.monitor.{samples, sample_ticks, count, latest}`, `monitor_valid`,
  `adc_overrange`, `temp.<name>`, `tx.ones_count`, `tx.bits`, `datapath.bitstat_above`, `rx.error_count`,
  `rx.error_count_valid`, `tx.heater.applied`, `tx.bias.applied` (+ `heater.readback_v/_i` when present); dtypes
  and shapes exactly as `spec.channels` (checked by the wiring test).

## 5. PlantParams field -> consumer (where each group goes)

| group | consumer |
|---|---|
| `optics.*`, `thermal.dlambda_dt_*`, `junction.fill_fraction` | `PhaseParams.from_plant_params` (phase, K^2, loss, eta) |
| `optics.backscatter_rb`, `optics.tcmt_orders`, `flags.optics_model`, `flags.backscatter` | `pack_selfheat_args` / `bus_propagate_nb` |
| `thermal.system` | thermal kernels, selfheat, `foster_from_system` (aggressor calibration), transport delay -> heater FIFO |
| `thermal.t_ring_lock_k`, `s_lock_m_per_k`, `self_heating_gain_m_per_w` | diagnostics, quasi-static mode (07 P0-10), aggressor `ring_t0_k` |
| `junction.*` (+ optics.alpha_fca_per_m, bias_ref_v) | `junction_table_from_plant_params` |
| `heater.*`, `timing.heater_latency_ticks`, `flags.heater_pwm_ripple` | `HeaterDriverParams.from_plant_params` |
| `bias_dac.*`, `timing.bias_latency_ticks` | `BiasDacParams.from_plant_params` |
| `monitor.*`, `timing.{adc_period, integration_window, monitor_latency}_ticks`, `flags.shot_noise`, `seeds.device` | `MonitorHw.from_plant_params` |
| `temp_sensors.*` | `TempSensorsHw.from_plant_params` (sites -> `sys.temp_obs`) |
| `heater.readback_*`, `timing.readback_*` | `ReadbackHw.from_plant_params` |
| `datapath.*`, `timing.error_count_latency_ticks`, `flags.pattern_noise` | `DatapathHw.from_plant_params`; `swing_pp_v`, `level_convention`, driver fields -> steps 5 and 7 |
| `laser.*` | `LaserHw.from_plant_params`; `input_transmission`, `bus_loss_per_m` -> step 2 / `bus_t2`; `rin_per_hz` -> `bitstat_from_monitor` |
| `layout.bus_order`, `ring_to_line`, `ring_spacing_on_bus_m` | optics walk, OMA line, `bus_t2` |
| `timing.dt_s`, `controller_period_ticks`, `reset_*`, `substep_policy`, `max_detuning_step_m`, `min_samples_per_dither_period` | runner / kernel loop |
| `flags.as_array()` | int32 flag array of the kernel (07 P0-9) |
| `world`, `world_choices`, `config_hash`, `diagnostics` | run manifest and reports |

## 6. Compute budget (07 P0-9): measured parts, B = 1, N = L = 8, shared M2 under load (pessimistic 1.5-2x)

| part | cost per 1 us tick |
|---|---|
| optics bus pass, exact, backscatter off | 5.7 us (CW) / 8.2 us (NRZ) ; with backscatter 9-15 / 15-27 us |
| one Newton evaluation of one ring (8 lines) | 0.66 us (CW) / 1.1 us (NRZ); with backscatter 2.4 / 4.1 us |
| implicit thermal step, interim, all observers (770 lags) | 4-6 us ; ring + heater observers 3.3-4 us ; rings only 1.3-2 us |
| same with the FEM ROM (K = 14 poles) | 26-84 us (far over budget until lags are merged per (source, tau)) |
| monitor | 0.8-1.5 us (5.3 us with 18 flicker poles) |
| laser | 0.34 us (0.8 us with RIN) |
| ones density, binomial | 2 us (PRBS or off: < 0.5 us) |
| scenario, full evaluation of 8 components | 1.4 us (evaluate only at `scen_next_break_b` ticks: negligible) |

The 10 us/tick target (>= 0.1 simulated s per wall s) is already exceeded by the optics alone with NRZ. Levers: warm
Newton starts (the walk restarts from the previous root, usually one evaluation away), backscatter off for long
soaks, CW (p1 at 0/1) where the data is not under study, restricting observers/sources (`assemble(observers=...)`),
`prange` over B (1.6-1.7 us per element-step measured by [CTRL] at B >= 64), the P0-10 multi-tick / quasi-static
modes, and pattern noise off. Report measured numbers in `tests/perf/`.

## 7. Known gaps (the engine owns or must work around)

1. No `mrm/plant/io.py`: the SensorBank side lives in `sensing.py` / `datapath.py`; the ActuatorBank side (controller
   pipeline FIFO of `Controller.latency_ticks`, latch timing) is the engine's.
2. Optional channels not declared in `mrm.core.channels`: reference tap (`tx.monitor_ref.*`), `datapath.bitstat_ones`.
3. External reflection (07 P2-2 `scenario.external_reflection`) needs a backward bus pass: not built (optics handoff
   section 5 has the design).
4. 07 P2-5 temperature-dependent conductivity is not wired (`src_scale` stays 1; `k_temp_residue_scale` exists
   with assumed weights; no schema switch). The aggressor calibration assumes nominal residues.
5. Junction temperature axis (07 P1-7: dV_bi/dT) is not applied (add `mrm.materials.dvbi_dt_v_per_k * (T - T_ref)`
   to v_lo/v_hi before `ring_optics_row` if wanted).
6. ROM mode loses per-ring heater-efficiency spread (`dc_scale` per world only); g26 runs the interim model.
7. `data_modulation: lut` (07 P2-6 waveform LUT) and the `mrm/waveform/` package do not exist yet; use `mean`.
8. Laser drift: only through the scenario (`laser_drift`); `LaserHw` drift parameters stay 0.
9. Temperature-sensor `self_heating_k`, the heater readback mean-vs-rms question (dac_pwm) and the bias DAC
   INL/DNL are minor and documented in the module handoffs.
10. The fixed 1 us step with `substep_policy: auto` (07 P0-10) needs: thermal FOH steps of any h (exact), the
    actuator quiescence check, scenario breaks, and the lag-gain assertion at the largest h; none of the leaf modules
    blocks it.

## 8. Physics to expect (so a correct engine is not "fixed" away)

- **Wrong-line capture at power-up in g26 nominal.** The cold resonance of ring n sits 0.73 nm (~2 FWHM) BLUE of
  line n+3 (7.6 nm offset vs a 3 x 2.29 nm grid step; FSR 10.3 nm). At +4 dBm per line a cold ring can heat itself
  onto that line when the lasers come on; in the wiring test two sampled rings did, and their monitors clipped at
  the 40 uA full scale. The laser ramp and the WDM supervisor (07 P1-12) must handle it.
- **Data-activity heat** (07 P1-8): junction CV^2 (~0.14 mW, through Z_opt 9.9 K/mW) plus assumed driver heat
  (~4.5 mW per ring) lift each ring ~2 K; squelch and traffic on/off steps move rings by this much.
- Heater TCR: dP/dcode varies 1.9x (voltage drive) and 4.6x (current drive) over the G26 code range (electrical
  handoff note 3), so loop gain is strongly code-dependent.
- A1 aggressor under resonance referencing heats the victim ~101 K (s0 70 pm/K, beta 1.4e-3), reaching ~126 degC
  from a 25 degC base: above the 0-105 degC junction range (scenarios handoff section 8).

## 9. 07 tests that need the engine (not yet written)

`test_loop_delay_impulse` and latency-vs-rate through the plant (P0-6); the fast_kernel against a 1 MHz Python step
with bit-identical I/Q (P0-7); `test_spec_independent_of_seed_and_uncertainty`, `test_no_truth_leak` (P1-1); metrics
within 1 % between dt = 1 us and 0.25 us, self-heating on and off; numba vs the NumPy reference (P1-18); batch and
thread invariance of the whole plant; the P0-10 compression convergence experiment; `tests/perf` targets (P0-9).
