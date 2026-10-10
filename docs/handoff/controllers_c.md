# Handoff: controllers C (`controllers_c`): linearized model, Kalman/LQG, MPC, oracle

Owner files: `mrm/control/models.py`, `mrm/control/baselines/kalman_lqg.py`, `mrm/control/baselines/mpc.py`,
`mrm/validation/oracle.py`, `tests/verification/test_ctrl_c_{harness,models,kalman_lqg,mpc,compiled,oracle,engine}.py`.
Contract: ARCHITECTURE section 6 items 8 and 11; 07 P1-1 (knowledge, Truth isolation, linearize, oracle), P0-1 (the
self-heating feedback in the model, the open-loop-stable OMA point), P0-6 (delays), P0-9 (compiled flat ABI); 04 C15.
Section 8 lists what the adversarial review changed; the numbers below are the reviewed ones.

## 1. Public API

```python
# mrm.control.models
KNOWLEDGE_TOKEN = "model_nominal"
check_knowledge(knowledge, required=("model_nominal",), who="controller") -> None      # raises KnowledgeError
linearize(nominal_cfg, operating_point=None, Ts=None, *, latency_ticks=0, measurement="mean", knowledge=None)
    -> LinearModel          # A, B, C, D, delay_ticks = linearize(...)
linearize_bank(nominal_cfg, operating_point=None, Ts=None, *, latency_ticks=0, measurement="mean", knowledge=None,
               pp=None, b=0, cond=None) -> list[LinearModel]       # every ring; pp/b/cond for verification and the oracle
OperatingPoint(ring=0, detuning_m=None, detuning_frac_of_peak=None, t_base_k=None, bias_v=None, ones_density=0.5,
               transition_density=None, laser_power_scale=1.0, pga_select=0)
LinearModel: A B C D (ZOH at Ts) delay_ticks | Ac Bc Cc Dc Ts dt_s delay_exact_ticks tau_u_ticks tau_m_ticks ring op
    x_names u_names=("heater_code", "ring_temperature_disturbance_k") y_names=("monitor_code", "detuning_m")
    .u0 .y0 .detuning0_m .selfheat_factor .delay_s .dc_gain() .frequency_response(f, with_delay=True)
    .step_response(t, input_index=0, with_delay=True) .n_delay(Ts=None)
    .delayed(Ts=None, outputs=(0, 1), n_delay=None) -> (Phi, Gu, Gd, C, Dd, nd)    # nd = max(q + 1, n_delay)
StaticPoint: ... heater_current_a (A; rms for dac_pwm), supply_scale (plant's array-supply droop factor, 1 = none)
nominal_plant_params(cfg)            # realize(cfg, 1, uncertainty="nominal", variation="off"), cached
Conditions(lam_m, p_chip_w, t_base_k, v_lo, v_hi, rho, transition_density, aggressor_w=None, data_enable=None,
           t_extra_obs_k=None)
conditions_from_config(pp, b=0, *, cfg=None, op=None) -> Conditions
static_point(pp, b=0, detuning_m=None, *, cond=None, cfg=None, op=None, pga_select=0, with_slopes=True) -> StaticPoint
oma_curve(pp, b=0, detunings_m=None, *, cond=None, cfg=None, op=None) -> (d, oma, loop_gain, p_drop, p_abs)
oma_optimal_detuning(pp, b=0, *, cond=None, cfg=None, op=None, stable_margin=0.0, refine=True)
    -> (delta_opt, oma_opt, oma_unrestricted, delta_unrestricted)      # (N,) each
resonance_detuning_m(pp, b, t_ring_k, cond, line_of_ring=None) -> (N,)
bank_optics(pp, b, t_ring_k, t_heater_k, cond, lam_m=None) -> (BusResult, r_s, c_eff)
measurement_age_ticks(period, adc_period, adc_latency, window, *, adc_phase_ticks=0, ctrl_phase_ticks=0,
                      mode="mean"|"latest", afe_tau_ticks=0.0) -> float
discretize_zoh(Ac, Bc, Ts) ; delay_discretize(Ac, Bc, Ts, delay_s) -> (Phi, G0, G1, q, r)
ring_block_lags(pp, b, n) -> (lag_index, residue, source)

# mrm.control.baselines.kalman_lqg   (registered: "kalman_lqg", "kalman_lqg_c")
KalmanLQG(params=None, **kw)         # BaseController; Params below; latency_ticks = params.latency_ticks (1)
    Params: target="oma_opt"|"detuning"|"frac_of_peak", target_detuning_pm=None, target_frac=0.75, latency_ticks=1,
            measurement="mean"|"latest", q_pm=1.0, r_codes=40.0, q_d_pm=0.3, q_rate_pm_per_ms=2.0, q_x_pm=0.01,
            sigma_extra_codes=2.0, ramp=True, lock_pm=5.0, start="warm"|"cold", probe_peak=False, ramp_step=40.0,
            ramp_margin_codes=None, probe_step=10.0, probe_fine_step=2.0, relock_step=40.0,
            relock_guard_codes=None, fold_frac=0.6
    .build_design(spec) -> LqgDesign (raises on schedule != simultaneous, or a ring without a regulator)
    .hysteresis_codes(design) ; telemetry(): state, locked, detuning_est_pm, integrator (d estimate, pm),
    integrator_saturated, innovation_codes, peak_codes
design_lqg(models, *, q_pm=1.0, r_codes=40.0, q_d_pm=0.3, q_rate_pm_per_ms=2.0, q_x_pm=0.01, sigma_extra_codes=2.0,
           samples_per_step=2.5, ramp=True, n_delay=None, kalman=True) -> LqgDesign
    LqgDesign.regulated (N,) bool: False where the regulator/LQR has no solution (no heater authority); Kf = 0 there
closed_loop_spectral_radius(design, ring, plant: LinearModel, gain=1.0) -> float
CompiledLqg(params=None, **kw)       # CompiledController (flat ABI), lock only (start="warm", probe_peak=False)
lqg_init(spec_i64, spec_f64) / lqg_step(p_i64, p_f64, s_i64, s_f64, obs_i64, act_i64)     # numba
ST_INIT, ST_RAMP, ST_PROBE, ST_RELOCK, ST_LOCK, ST_FAIL = 0..5

# mrm.control.baselines.mpc          (registered: "mpc", "mpc_c")
LinearMPC(params=None, **kw)         # derives from KalmanLQG (estimator, acquisition); Params add horizon_steps=20,
    control_moves=4, r_du_codes=40.0, slew_codes_per_step=500.0, code_margin=0, qp_max_iter=100, qp_tol=1e-3,
    qp_rho_scale=0.1, qp_alpha=1.6; telemetry adds qp_iterations, qp_residual, slew_active, bound_active
    .build_mpc(spec) -> MpcDesign    # slew = floor(min(slew_codes_per_step, hw slew * Ts)), at least 1 code
design_mpc(lqg, *, Np=20, Nu=4, q_pm=1.0, r_codes=40.0, slew_codes=500.0, rho_scale=0.1, sigma=1e-9, alpha=1.6)
admm_box_qp(P, q, A, lo, hi, Minv, rho, sigma=1e-9, alpha=1.6, x0=None, z0=None, y0=None, max_iter=200, tol=1e-6)
    -> (x, z, y, iterations, primal_res, dual_res)                   # vectorized, per-problem stopping
mpc_equivalent_gain(md, ring) ; mpc_closed_loop_spectral_radius(md, ring, plant, gain=1.0)
CompiledMpc(params=None, **kw)       # lock only; mpc_init / mpc_step numba (same ADMM)

# mrm.validation.oracle              (NOT registered; truth_fed = True; the registry refuses it)
TruthState(tick, x (B, M) thermal lag states, t_ring_k (B, N), conditions [Conditions per world],
           heater_applied=None)
oracle_target(pp, conditions | None, *, stable_margin=0.0, cfg=None) -> OracleTarget
    OracleTarget: detuning_m, oma_w, oma_unrestricted_w, detuning_unrestricted_m, p_in_w, reloma_db, loop_gain,
                  heater_code (B, N), reachable (B, N) bool, static [StaticPoint]
reachable(pp, static_point) -> (N,) bool    # heater power > 0, code inside the range with dP/dcode > 0, supply OK
oma_penalty_db(oma_w, oracle_oma_w) ; reloma_db(oma_w, p_in_w)
Oracle(pp=None, cfg=None, *, stable_margin=0.0, q_pm=0.5, r_codes=400.0, ki_per_step=0.3, retarget=True,
       retarget_power_frac=2e-3, retarget_lambda_m=0.2e-12, retarget_v=2e-3, retarget_min_ticks=200)
    .reset(spec, truth_pp=None, truth=None, cfg=None) ; .step(obs, truth: TruthState) -> {"tx.heater": (B, N) int64}
    .reach (B, N) bool ; .det_target ; .oma_target ; .n_retargets
```

## 2. What the model is (details in the module docstrings)

- Built ONLY from `realize(spec.nominal, 1, uncertainty="nominal", variation="off")` and the plant's own static laws
  (phase/R1, exact bus optics NRZ-averaged, data-activity heat, thermal DC gains, the driver law with TCR, the
  monitor's ideal code). Operating point: every ring at the OMA-optimal open-loop-stable detuning of the nominal
  model (g26 with the extracted coupling_rib160, which appeared at 10:08 during the review: +104.9 pm, resonance red
  of the laser, loop gain -5.45, relOMA -6.36 dB, drop 29.8 uW, monitor code 46 400 of 65 535 = f* 0.77 of the
  nominal drop peak, heater 13.3 mW at code 34 256; the bank draws 128 mA of the 150 mA supply-current limit. With
  the fallback coupling it was +107.4 pm, loop gain -5.3, code 47 300).
- Per ring: the ring's own thermal block (grouped by (source, tau): 8 states in g26 interim), the TCR feedback
  dP_h = k_u du + k_T dT_H, the optical self-heating feedback b_w a_R c_R (07 P0-1: divides the heater-power-to-
  detuning gain by 1 + G|P'| = 6.3 in g26), the monitor gain. Disturbance input = ring-temperature disturbance (K),
  which enters every ring-temperature path like the thermal states (a rigid-shift input was tried first: the drop
  power's T_R and line-wavelength sensitivities differ by ~0.4 % through the coupler dispersion, which a nm-scale
  heater/aggressor offset turned into 25-40 pm of lock error).
- Delay (07 P0-6): tau_u = controller latency + heater DAC latency (+ nominal transport delay); tau_m = mean age of the
  samples a step consumes, PER RING (an ADC mux samples the rings that share a converter at staggered phases; g26:
  6.5 ticks for the block mean, 3.5 for the newest sample); all on the input, exact fractional discretization in
  `LinearModel.delayed()`. g26 with latency 1: tau = 8.5 ticks. Rings with different delay lines are padded to one
  state size in `design_lqg` (padding is exactly harmless: same closed loop, tested).
- The OMA curve and the oracle target solve the heater-resistor temperature of each point when the loss depends on
  it (`dln_alpha_dt_heater_per_k`, the `heater_t_loss` ER hypothesis); before the review they used T_H = T_R, which
  overstated the oracle OMA by 0.4-0.5 % (0.02 dB) in such worlds.
- Verified against the ReferencePlant (nominal world, warm start): a 60-code step 2.871 pm (plant) vs 2.862 pm (model),
  |error| <= 0.11 pm from 3 us on; dither at 2/10/25 kHz: detuning and monitor-code amplitude within 2 % and phase
  within 3.1 deg (test_ctrl_c_models.py). The plant lags the model by a constant ~0.35 us (0.3/1.3/3.1 deg at
  2/10/25 kHz): its implicit self-heating step at dt = 1 us over the ~0.5 us closed self-heating pole; negligible
  against the 8.5 us loop delay.

## 3. Results (ReferencePlant, g26 nominal world, B = 1, N = 8; Truth metrics in the test harness only)

Lock error = Truth detuning minus the oracle point (total) or minus the monitor-consistent reference (tracking: the
oracle point moved by the PD dark-current offset, see 5.1). Excursions: 5 pm threshold, 1 pm hysteresis (g26
metrics), after the settle window. OMA penalty: mean 10 log10(OMA_oracle / OMA), oracle at the start conditions.
All numbers are after the review's harness fix (section 8 item 1: the plant now runs at the config's scenario bias,
1.05 V; before, every closed-loop number carried a 0.6 pm offset from a 1.0 V plant) and with coupling_rib160
present, except the noise/quantization rows and the engine runs (fallback coupling; differences of this size do not
change them).

| scenario | controller | total rms / max (pm) | tracking rms / max (pm) | excursions | time to lock | OMA penalty (dB) |
|---|---|---|---|---|---|---|
| steady lock, 2 ms | kalman_lqg (= _c) | 0.10 / 0.15 (mean -0.10) | same | 0 | 0 | 0.0009 |
| | mpc (= _c) | 0.11 / 0.15 | same | 0 | 0 | 0.0009 |
| | oracle | 0.025 / 0.12 | | 0 | 0 | < 0.001 |
| A2 808 degC/s window (3 ms at 20-23 ms, 23 pm/ms open loop) | kalman_lqg (= _c) | 11.7 / 39.2 | 0.51 / 1.46 | 0 | 0 | 0.27 |
| | mpc (= _c) | (not measured) | 0.42 / 1.13 | 0 | 0 | |
| | oracle | 0.16 / 0.51 | | 0 | 0 | |
| A1 slide-10 window (3 ms, 43-40 ms before the hot cusp) | kalman_lqg (= _c) | 0.72 / 2.75 | 0.33 / 0.90 | 0 | 0 | 0.0014 |
| laser power +0.5 dB step | kalman_lqg | lock point moves +21.5-22.5 pm | settles within 0.5 ms to < 0.13 pm | 0 | | ~0.2 dB after the step (builder: 0.24 dB at 25 pm; not re-measured) |
| | oracle | target unchanged; within 0.26 pm 0.2 ms after the step | | 0 | | ~0 |
| probe-peak acquisition (warm) | kalman_lqg | last 1 ms: 0.05 / 0.12 | | 0 | 2.0 ms (probe included) | |
| cold start, ring 0, only line 0 lit (slow) | kalman_lqg | -2.9 pm offset (-0.5 pm with the fallback coupling) | | | hand-over 10.5 ms | |
| monitor 10-bit ADC (pat2) | lqg / mpc | 0.15 / 0.30 | | 0 | 0 | 0.0007 |
| monitor noise 6.4 pA/rtHz + 100 kHz flicker corner (range top) | lqg / mpc | 0.11 / 0.25 (jitter 0.04) | | 0 | 0 | 0.0006 |
| both | lqg / mpc | 0.15 / 0.40 | | 0 | 0 | 0.0007 |

"(= _c)": the compiled versions give identical commands (LQG code for code; MPC within 1 code in < 1 % of steps) and
identical Truth metrics. Compiled cost (numba, B = 1, N = 8, loaded M2): LQG 1.6 us per controller step (0.2 us per
ring), MPC 3.5-6 us (ADMM converges in 1 iteration when warm). Python path: 0.2-0.4 ms (LQG) and 0.4-0.9 ms (MPC)
per step. MPC with a binding slew limit (2 codes per step < the ~5 the A2 ramp needs): every move within the limit,
slew_active reported, lag grows to 4.8 / 16.7 pm rms / max instead of instability. MPC with a binding upper code
limit (code_margin; +0.5 dB step): every command inside the limit, bound_active on the ring at the limit, no
wind-up. Noise and quantization inside the declared ranges do not matter: the g26 monitor slope is ~200 codes/pm.

### 3.1 Where they fail (documented; tests in test_ctrl_c_kalman_lqg.py, slow)

- **Laser power drop** (`test_laser_power_drop_past_the_set_point`): the lock is a monitor set-point lock at
  f* = 0.77 of the nominal peak. -1.0 dB: still locked but 68-72 pm toward the peak. -1.5 dB (beyond
  -10 log10 f* = 1.1 dB): the loop cools every ring through the peak, all 8 rings fold to the blue branch (5.7-6.4 nm),
  heaters run to code 0. `locked` telemetry drops to 0 (no false lock claim); the state stays LOCK: there is no
  loss-of-lock detection or re-acquisition in LOCK (04 C15's "innovation too large -> re-acquire" fallback is not
  implemented; a re-acquisition with the stale peak would cycle, it needs a re-probe).
- **A1 toward the hot cusp** (`test_a1_toward_the_cusp_dark_current_runaway`; a 63 ms ad-hoc run from 43 ms before
  the cusp, fallback coupling): the calibrated A1 aggressor heats ring 0's monitor PD (interim site law) to 350 degC
  at -22 ms and 510 degC at -9 ms while the ring stays near 135 degC. The dark current (14 000 codes at -22 ms,
  35 000 at -14 ms, 62 000 at -9 ms) eats the ~47 000-code net set-point; tracking error 6.5 pm at -21 ms, 29 pm at -18 ms, then
  runaway: ring 0's heater reached 65 535 at -10.7 ms, its next order (FSR 10.3 nm) swept across lines 3, 2, 1 and
  starved rings 3, 2, 1, which folded (-2.3 to -4.5 nm) and whose heaters went to 0. A bank-wide failure started by
  one ring's monitor. Plant question first (request 4); controller lesson: a set-point lock needs a dark/baseline
  reference and a heater authority window. Since 2026-10-10 this layout is the stress scenario
  `a1_slide10_aggressor_pd_adjacent`, which the test runs (request 4).
- **Model mismatch** (`test_model_mismatch_worlds_report`, 6 LHS worlds, seeds 11): only 2 of 6 worlds can hold
  their OMA points at all (worlds 0, 1: bank above the 200 mW supply power; worlds 3, 4: within the power but above
  the 150 mA supply current, so the supply droops every heater by 8-18 % and the warm start folds during the hold,
  before any controller acts). Of the 2 reachable worlds the nominal set-point holds all 8 rings of one (offsets
  4-74 pm from the oracle point, 0.02-1.1 dB OMA penalty; one ring's monitor saturates at its point, 1.55 x the
  nominal code > 65 535, so it never gets a valid sample, stays in INIT and holds its warm code) and 7 of 8 of the
  other (ring 7's nominal set-point is 1.07 x its true peak: the loop cools it through the peak and it folds).
  `test_model_mismatch_every_reachable_world_locks` stays xfail. The builder's report ("loses lock in 2 reachable
  worlds") counted the two supply-current worlds as reachable: those were harness artefacts, not controller failures.
- Linear robustness (unchanged): the LQG and the unconstrained MPC designed on the nominal model are stable (spectral
  radius < 1) on all 12 LHS draws (of 16) whose ring-0 point is in range, and with the heater gain scaled 0.3-6x.
  The MPC is not exactly type 2 (move blocking): 0.005 pm steady error on a 20 pm/ms ramp; the LQG is.

## 4. Requests (handoffs)

1. **integrator / baselines package owner**: `mrm/control/baselines/__init__.py` is empty, so `registry.resolve`
   does not see these baselines until something imports them. Please add
   `from mrm.control.baselines import kalman_lqg, mpc  # noqa: F401` (registers kalman_lqg, kalman_lqg_c, mpc, mpc_c).
   Until then `registry.create("mrm.control.baselines.kalman_lqg:KalmanLQG")` works.
2. **params (content / machinery)**: 07 D13's knowledge filter and `controller_knowledge` preset do not exist; these
   controllers require the token `model_nominal` in `controller_interface.knowledge` (g26 lists only `line_grid`,
   `ring_to_line`) and then use `spec.nominal` as the nominal design. Please (a) add the token to the knowledge enum
   / documentation, (b) make sure `ControllerSpec.nominal` is the controller-knowledge view: `mrm/sim/runner.py`
   passes `nominal=cfg`, so a run that selects a corner (`alts=`) hands the corner to the model (the model would
   know the corner).
3. **engine (mrm/sim/runner.py, plant.py)**: (a) the oracle needs the thermal lag states and per-world Conditions:
   `runner._run_python` calls `ctrl.step(obs, truth=plant.truth())` but `Truth` has neither `x` nor the Conditions
   (NRZ levels, ones and transition density, base temperature); please build a `mrm.validation.oracle.TruthState`
   or add those to Truth; `Oracle(pp, cfg).reset(spec)` is supported. (b) A real warm start (thermal state at the
   lock steady state, as `tests/verification/test_ctrl_c_harness.py::warm_start` does with `models.static_point`)
   would let the lock-only compiled controllers run on the engine; `heater_code` + laser ramp starts cold.
4. **plant_thermal / extraction / scenarios**: the calibrated A1/A2 aggressor heats ring 0's monitor PD through the
   interim site law to ~270 degC in the A2/A1 test windows and 350-510 degC in the last 22 ms before the A1 hot cusp,
   while the victim ring stays near 135 degC. The PD dark current then reaches 5 000-62 000 codes: 25-35 pm of
   set-point error in the windows and a runaway that takes down four rings near the cusp (3.1). Physical, or an
   artefact of the placeholder site path (plant_thermal handoff 6.8)? The FEM ROM should settle it.
   **Resolved 2026-10-10** (docs/handoff/integrator.md section 7): it came from the layout (site 5 um from PD0), not
   physics. The g26 default site is now (0, -30 um), 55 um from PD0 (ring distances unchanged), so the A1/A2 windows
   of test_ctrl_c_kalman_lqg / mpc / oracle see little PD heating; the old layout is the stress scenario
   `a1_slide10_aggressor_pd_adjacent` (`test_a1_toward_the_cusp_dark_current_runaway`).
5. **WDM supervisor / acquisition owners**: in the g26 bank a cold heater ramp crosses three foreign lines, and a ring
   that falls off its fold (1.2 nm drop at +4 dBm) or runs hot (3.1, A1) brings its next order onto other lines
   (FSR 10.3 nm vs a 9.16 nm four-line span). These acquisitions are single-ring; no line arbitration, no heater
   authority window, no loss-of-lock supervision.
6. **metrics owner**: (a) OMA penalty with a laser power change needs the oracle OMA at the same instant (it scales
   with P_in); the OMA-optimal DETUNING does not depend on P_in (the ring is linear in P_in at a fixed detuning), so
   `oracle_target` is cheap to refresh when only the power moves. (b) Use `OracleTarget.reachable`: in g26's
   uncertainty set many worlds cannot hold their OMA point (3 of 4 and 4 of 6 in two LHS draws: supply power or
   supply CURRENT limit, codes past the DAC range or in compliance); an OMA penalty there measures the world.

## 5. Physics findings (please read before using the numbers)

1. Dark current under the aggressor (4 above): the tracking error is scored against the monitor-consistent
   reference; the total error against the oracle point (39 pm max under A2) is mostly this offset. At the g26
   default site since 2026-10-10 (55 um from the PD) the total error is 0.75 pm max under A2 and 0.45 pm under A1
   (kalman_lqg windows; mpc A2 tracking 0.19 pm, oracle A2 0.47 pm).
2. The OMA-optimal stable point in g26 nominal is the red (self-locked) side at +105 pm (+107 pm with the fallback
   coupling); the blue-side OMA peak is only 92-94 % of it and open-loop unstable (loop gain about +5).
3. A monitor set-point lock moves with laser power (+0.5 dB -> +22 pm; -1.0 dB -> -70 pm; below -1.1 dB it folds)
   and with the monitor gain: across the uncertainty set the nominal set-point lands anywhere from 0.5 to 1.5 of a
   world's true peak; fraction-of-peak normalization (the probe) is the remedy, but see section 6.
4. Feasibility: the array supply has a CURRENT limit (150 mA) as well as a power limit (200 mW); both decide whether
   a world can hold its OMA point (`StaticPoint.supply_scale`, which matches the plant's droop factor to 0.2 %).
   In 4 of 16 LHS worlds ring 0's OMA point needs negative heater power (the ring already sits red of it).
5. A saturated monitor (gain 2x worlds: the code at the OMA point exceeds 65 535) never gives a valid sample: the
   controller stays in INIT and holds its code. A PGA-select policy (monitor.pga_select) is not implemented.
6. Thermal lag during a probe: the ring keeps moving ~10 pm after the last step near the peak; the probe therefore
   takes the top in 2-code steps once the slope falls below half its maximum (the fold sits only h^2/(2 G P_pk)
   ~10 pm past the peak at +4 dBm).

## 6. Deviations and open issues

- Not in the (c) list of ARCHITECTURE section 6, but compiled versions are provided anyway (lock only: the
  acquisition FSM runs on the Python path only).
- `linearize` returns a `LinearModel` that unpacks as the contract's 5-tuple; the second input is a ring-temperature
  disturbance (K), and C has two rows (monitor code, detuning).
- `mrm/control/models.py` imports `mrm.params.realize` and the plant leaf libraries (inside functions) to realize the
  NOMINAL design (the task asks for exactly that). A future 07 P1-1 `test_control_does_not_import_truth` scan of all
  of `mrm/control` would flag these imports; it should allow `models.py` to use realize/plant leaf libraries on the
  nominal config and forbid `mrm.plant.plant` (Truth), `mrm.sim` and `mrm.validation` (models.py imports none).
- Not implemented from 04 C15: the "innovation too large -> re-acquire" fallback (3.1). Only
  `controller_interface.schedule = simultaneous` is supported (round_robin is refused at reset with a clear error:
  the design assumes every ring's command acts each period).
- Open: the probe-peak acquisition and the nominal set-point fail in some reachable uncertainty worlds (set-point
  above the true peak, saturated monitor): `test_model_mismatch_every_reachable_world_locks` is `xfail(strict=False)`.
  Fixes to try: a PGA autorange, a re-probe on loss of lock, a measured-slope (dither) gain correction.
- Open: the oracle's dynamic hold is a Truth-fed LQR + regulator feed-forward + integral of the true error on the true
  linearization; it re-solves its target when the line power/wavelength/junction levels move (rate-limited). Its
  holding error is 0.1-0.4 pm under A1/A2; it has not been run on the engine (request 3a). Unreachable targets are
  held at the nearest code, not regulated.
- The cold-start test uses only line 0 (single ring). Through the engine (`mrm.sim.runner.run`, Python slow path,
  laser-ramp reset, cold heaters) KalmanLQG and LinearMPC both acquired and locked ring 0 (builder: 104.3 pm against
  the 103.35 pm oracle point of that single-line world); see section 7 for the review's re-run.

## 7. Tests

`tests/verification/test_ctrl_c_*.py`: 51 tests (44 fast, 7 slow).
- Fast (`-m "not slow"`): 44 passed (3 min 02 s at load ~20; 1 min 33 s together with the neighbours at lower load;
  slowest single test 21 s), also after coupling_rib160 appeared.
- Slow (re-run with coupling_rib160 present): cold start (passed, -2.9 pm), mismatch report (passed), the
  every-world claim (xfail, known), the power-drop and A1-cusp failure reports (passed): 4 passed, 1 xfailed in 58 s.
  Engine runs re-run by the review (fallback coupling): both passed (kalman_lqg 104.31 pm and mpc
  104.28 pm against the 103.35 pm target; 203 s and 395 s wall including the kernel compile; peak RSS 0.95 and
  1.19 GB).
- Neighbours: test_control_api.py and test_plant_wiring.py passed (113 passed together with the ctrl_c fast tests);
  tests/validation/test_replicas_static.py's cross-check of `models.static_point` passes.
- NUMBA_DISABLE_JIT=1: the compiled step functions give the same commands (subprocess test); the model tests pass
  (api, delays, mux, self-heating: 4 passed in 46 s).
- Truth isolation: an AST scan of the three control files (`test_no_truth_imports`) and a design that depends only
  on the spec's nominal config.

## 8. Adversarial review (what changed, and why)

1. **Harness ran the plant at the wrong bias (bug, all closed-loop numbers).** `quiet_scenario()` and library
   scenarios (`registry.create('a2')`) carry no config defaults, so tx.bias_v was undriven and the bias DAC sat at
   code 0 = v_min 1.0 V instead of g26's scenario.bias_v 1.05 V; the warm start also read the conditions at the
   first tick, before the bias DAC latency. The plant then sat 0.2 pm and ~110 monitor codes away from the
   controller's nominal model, which every set-point lock showed as a -0.6 pm offset. Fixed: `with_config_defaults`
   (as `scenarios.registry.from_config` and the engine apply them) and a warm start after the bias DAC latency; new
   test `test_plant_runs_at_the_controllers_nominal_conditions`. Steady-lock error -0.62 -> -0.10 pm.
2. **Harness feasibility ignored the supply CURRENT limit (bug, mismatch conclusions).** Two LHS worlds within the
   200 mW power limit but over the 150 mA current limit were counted as reachable; their supply droop folded the
   rings during the open-loop hold and the report blamed the controller. Fixed: `StaticPoint.supply_scale` (the
   plant's droop factor, matched to 0.2 %), `oracle.reachable` (used by `Rig.feasible_worlds`); new test
   `test_supply_limits_decide_reachability`.
3. **Oracle crashed on sampled worlds (bug).** A target past the code range or in the driver's compliance has
   dP/dcode = 0: the regulator equations were singular (LinAlgError) and, behind them, the filter and LQR Riccati
   equations failed; the oracle also assumed every world had world 0's state sizes. Fixed: per-world designs, lag
   maps and delay lines; `design_lqg(kalman=False)` for the full-state oracle; `LqgDesign.regulated` (a ring
   without heater authority gets zero gains instead of failing the bank); the oracle holds unreachable targets at the
   nearest code and reports `reach` / `OracleTarget.reachable`; a heater settle-filter state is filled when present.
   New tests `test_oracle_holds_sampled_worlds`, `test_design_flags_a_ring_without_heater_authority`.
4. **Per-ring measurement delay under an ADC mux (bug for channels_per_adc > 1).** tau_m used phase 0 for every
   ring; it now uses the plant's mux phase rule per ring, and `design_lqg` pads delay lines to one size. New test
   `test_adc_mux_gives_each_ring_its_own_measurement_delay` (also checks that padding is exactly harmless).
5. **Heater slope at the code-range end (bug).** dP/dcode used a centred difference that read the clipped DAC past
   code_max and halved the slope there; it is one-sided at the range ends now.
6. **OMA curve heater temperature (bug, heater_t_loss worlds).** See section 2; oracle OMA was 0.02 dB high.
7. **MPC fractional slew limit (bug, compiled path).** A hardware slew (codes/s x Ts) is fractional; the compiled
   step clipped the rounded command to u_last +- slew and stored a fractional command in its filter state (the
   applied code was its truncation), and the Python path could step past the limit after rounding. The limit is now
   floored to whole codes (>= 1). New test `test_fractional_slew_limit_is_whole_codes`.
8. **round_robin schedule**: silently wrong before (the predictor ran on commands never applied); refused now.
9. New coverage: MPC code limit binding (`test_mpc_code_limit_binds`); the documented failure modes of 3.1;
   noise/quantization experiments (table). Docstrings corrected (MPC: trust-constr, not exactly type 2; LQG limits).
