# Handoff: plant electrical (`mrm/plant/electrical.py`)

Status: done and adversarially reviewed. `tests/verification/test_electrical.py` has 56 tests. They run in about 18 s
with a cold numba cache, about 5 s warm, and 1.5 s under `NUMBA_DISABLE_JIT=1`. They need no solver (one test uses
`scipy.integrate.solve_ivp`, one imports `mrm.plant.thermal`).

Contract items covered: 07 P1-11 (driver laws voltage | current | power | dac_pwm, TCR tied to family as an input),
P0-6 (register latch -> DAC latency FIFO -> slew -> settle; `applied_codes` is the latched register), P1-10 (DAC
INL/DNL, max update rate, compliance, analog V/I/R values for readback), P1-12 (array supply limit: clip and flag),
P2-5 (electrothermal loop gain), P0-7 (bias DAC), P1-7 runtime side (junction LUT, analytic fallback, derived
report-only efficiency and dQi/dV), P1-8 (data-activity heat). Not covered: the P1-10 `dac_glitch` fault (no owner
named; see "Open" below).

## Changes made by the review (API-relevant)

- `actuator_quiescent_row(reg, has_pending, fifo, y_slew, y_out, target, tol)`: **new argument `has_pending`**. The
  old check returned True while a command waited in the latch (DAC max update rate), so P0-10 tick skipping could drop
  that write.
- `HeaterDriverParams.build` and `heater_steady_state` reject a driver law whose own full scale is missing (inf):
  `0 * inf` gave NaN power. They also reject r_cold <= 0, slew <= 0, settle < 0 and compliance <= 0.
- Latencies and rates are converted with `mrm.core.timing.seconds_to_ticks` / `rate_to_period_ticks` (one tick rule;
  negative or non-whole values raise `TimingError`, a `ValueError`, naming the key). The old local rule accepted a
  negative latency, which made the FIFO read stale slots.
- Current-drive runaway (k I^2 >= 1) with no compliance limit returned NaN. It now returns the explicit P = I^2 r_o
  with `FLAG_RUNAWAY`, so a runaway grows tick by tick and stays finite.
- `bias_bank_step_row` indexed the noise draw `z[n]` without broadcasting; a (B, 1) or (1, 1) `z` read past the array
  in numba. It now broadcasts.
- `load_junction_table`: a key `r_series_ohm` was read as ohm*m. By the 07 P1-15 suffix grammar it is ohm, so it is
  now accepted only with `junction_length_m` (converted r' = R*L); otherwise write `r_series_ohm_m`. The table
  rejects a `dneff` that is not relative to 0 V (an absolute n_eff), negative alpha, non-positive C, and warns when
  the grid does not span -0.5 .. 4.0 V. A NaN voltage now returns NaN with `FLAG_GRID_CLIP` instead of indexing past
  the grid.
- New helper `junction_dqi_dv_frac_per_v(table, v, fill, alpha_i_total_per_m)` (07 P1-7 derived, report-only).
- `HeaterDriverParams.build` warns when `pwm_ripple` is combined with `settle_tau_s > 0`: the ripple term is added
  after the settle filter and is then overestimated.

## How to call it

- Every per-tick kernel is `@njit(cache=True)` and comes in two forms. `*_row` works on one world: arrays (N,), and
  parameter rows of shape (N,) or (1,). Use it inside `prange` over B in `sim/kernel.py` (tested: a `prange` wrapper
  over `heater_bank_step_row` is bit-identical to the batched kernel). The batched form loops over b for (B, N)
  arrays, with parameters (B|1, N|1).
- Python slow path: `HeaterDriverParams.build(B, N, dt_s, law=..., ...)` with `HeaterBank(params, initial_code)`, and
  `BiasDacParams.build(...)` with `BiasBank(...)`. Call `.step(tick, cmd, cmd_valid, ...)` once per tick, in order.
- Tick semantics follow `mrm/core/timing.py`: `step(tick=k, cmd)` runs after the controller edge at instant k and
  produces the actuator output for plant interval [k, k+1). `cmd` must be the command whose latch tick
  (edge + `Controller.latency_ticks`) is k. The controller pipeline FIFO belongs to timing/runner, not to this module.
  The DAC latency FIFO is here: a plain delay line, out(k) = reg(k - dac_latency_ticks) (tested at dt = 1 and 0.1 us:
  the delay in seconds is the same).

## Requests and notes, by consumer

1. **params / realize (schema owner)**
   - Key mapping for `HeaterDriverParams.build`:
     `ring.heater.driver_mode` -> `law` (per world, (B,)); `dac_bits` -> `bits`; `pwm_bits`; `pwm_period_s`;
     `fidelity.heater_pwm_ripple` (on) -> `pwm_ripple=True`; `full_scale_v/_a/_w` ->
     `v_full_scale_v / i_full_scale_a / p_full_scale_w`; `r_cold_ohm`; `tcr_per_k` (from
     `mrm.materials.heater_tcr_per_k(family)`); `r_ref_temp_c` -> `tcr_ref_k` (in K; only T - T0 matters, but use
     the same scale as the ROM's heater observer); `code_min/code_max`; `latency_s`; `slew_codes_per_s`;
     `settle_tau_s`; `dac_max_update_hz`; `driver_compliance_v` -> `compliance_v`;
     `array.heater_total_power_max_w` -> `supply_power_max_w`; `array.heater_supply_current_max_a` ->
     `supply_current_max_a`.
   - The keys requested earlier (`ring.heater.pwm_bits`, `fidelity.heater_pwm_ripple`, `ring.junction.loss_ratio_q`,
     `driver.level_convention`) are now in `schema.py`. Thank you. With ripple on, dt must be a whole number of PWM
     clocks (2.56 us / 256 = 10 ns); 1 us and 0.1 us both are.
   - The schema doc of `ring.junction.depletion_exponent` still says "1/3..1/2". The code accepts [1/3, 2/3] (see
     physics note 1); the schema bounds (0.2, 0.7) already allow it. Please align the doc string.
   - INL/DNL: sample `bit_err_lsb` with `dac_bit_errors_from_spec(dac_dnl_lsb, n_bits, z)`, where z is
     standard-normal of shape (B, N, n_bits) from stream `variation`. Then `dac_dnl_lsb` is the 1-sigma DNL at
     mid-scale. Set `inl_bow_lsb` to a per-device draw scaled by `dac_inl_lsb`, for example uniform in ±1; the bow
     peaks at mid-scale and gives |INL| = |bow|. For `dac_pwm`, the first `bits - pwm_bits` entries are the coarse
     8-bit DAC's errors: sample them with n_bits = 8, not 16.
   - Bias DAC `BiasDacParams.build`: `ring.bias_dac.{bits, v_min_v, v_max_v, update_rate_hz, latency_s,
     slew_v_per_s, settle_tau_s, noise_rms_codes}` map 1:1. The noise is redrawn once per `update_rate_hz` period
     and held between draws, so its spectrum does not depend on dt. With `update_rate_hz` absent (inf) it is redrawn
     every tick: its rms is dt-invariant but its spectral density is not.
   - Junction, analytic mode: `analytic_junction_table_from_ring(eff_at_0v_m_per_v, v_bi_v, m=depletion_exponent,
     fill_fraction, ring_length_m, lambda_m, n_g, capacitance_at_0v_f, series_resistance_ohm,
     alpha_fca_ring_per_m, loss_ratio_q | (dqi_dv_frac_per_v, alpha_i_total_per_m))`. It builds the
     full-circumference table, so that runtime fill scaling returns the ring-level values exactly (tested).
     `loss_ratio_q` can also be computed from `mrm.materials.plasma_dispersion_derivatives` as q = λ·Δα/(4π·Δn).

2. **thermal (`mrm/plant/thermal.py`) and the kernel**
   - The implicit TCR accessor exists now: `thermal.observer_split_b(...)` returns (pre, g) for the
     heater_resistor observer. Pass `t_heater_pre = T_base + pre` and `g_self = g` to the heater step. A test runs
     this loop with `thermal.foh_coefficients` / `thermal_step_b` and checks that the reported R equals
     R0·(1 + α·(T_end − T0)) to 1e-12 every tick and that the loop converges to `heater_steady_state`.
   - Explicit coupling (`g_self = 0`, `t_heater_pre = T_H(k)`) is still allowed. With a pole much faster than dt it
     rings with period 2 and ratio L ≈ -0.19 per tick at 25 mW (tested). Prefer the implicit form.
   - Heater power `out_p` (W) is the ZOH input of source `heater`: the driver law at the interval-average drive.
   - `data_activity_heat_row` returns P_j (source `junction`) and P_d (source `driver`, at `driver.site`).
   - P0-10 multi-tick stepping: skip ticks only while `actuator_quiescent_row(reg, has_pending, fifo, y_slew,
     y_out, target, tol)` is True, and only up to the next tick with a command. Then advance the settle filter with
     `settle_advance_row(target, y_slew, y_out, tau, dt, n_ticks)` (tested equal to tick-by-tick stepping). The check
     does not see the dac_pwm ripple or the bias noise: use ripple off in quasi-static mode, and do not skip across
     a bias-noise redraw.

3. **sensing (`mrm/plant/sensing.py`)**: heater readback should sample `HeaterTickOut.v_v / i_a` (true analog values;
   rms for `dac_pwm`) with its own ADC, rate and latency. `r_ohm` is the end-of-tick resistance. Not quantized
   here. If the readback ADC averages the PWM waveform it sees mean values, not rms; sensing decides.

4. **optics / plant**: bias -> `BiasBank.step(...).v_avg` (V, reverse positive). Then `nrz_levels_row(v_bias,
   swing_vpp, mode, -0.5, 4.0, lo, hi, flags)` with mode from `driver.level_convention`, then `junction_eval_row`
   at `lo` and at `hi`. It returns ring-level `fill·Δn_eff(V)`, ring-average `α_FCA(V)` (1/m), C_j (F) and R_s (Ω).
   `junction_swing_capacitance_row(lo, hi, ...)` gives C_eff for the P1-8 heat; pass `swing_vpp = hi - lo` (after
   clamping) to `data_activity_heat_row`, the interval C_eff was integrated over. Optional temperature axis (07
   P1-7): add dV_bi/dT·(T_j − T_ref) (`mrm.materials.dvbi_dt_v_per_k`) to v and to both levels before the lookup.

5. **scenarios**: per tick, provide `transition_density`, `swing_vpp` and `driver_enable` (07 P1-8), plus the
   additive `bias_offset_v` (`BiasBank.step(v_offset=...)`). The bias-noise draws `z` (B, N) come from the counter
   RNG, stream category `dac_<channel>` (`Stream.DAC` in `mrm/core/rng.py`), for example `dac_bias`.

6. **extraction/junction.py**: `load_junction_table` reads `schema: junction/1` with the keys `v_reverse_v`,
   `dneff` (relative to 0 V; checked), `alpha_fca_per_m` (absolute, >= 0), `capacitance_f_per_m` (> 0) and
   `r_series_ohm_m` (Ω·m for a full-circumference junction, R_s = r'/(f·L); scalar or per grid point). The keys may
   be at the top level or under `"table"`. A key `r_series_ohm` (Ω) needs `junction_length_m`. The grid should span
   -0.5 .. 4.0 V. The sha256 check against the sidecar is realize()'s job.

## Physics notes and disagreements with the contract (please review)

1. **Depletion exponent (07 P1-7, 06 M2, schema `depletion_exponent` "1/3..1/2").** The charge-control derivation
   (03 §5.3, whose own table writes the m = 1/3 row as A in pm·V^-2/3) gives Δλ ∝ ΔQ_j ∝ (V + V_bi)^(1 − m_C), where
   m_C is the C-V grading coefficient (1/2 abrupt, 1/3 linearly graded). So the exponent of Δλ is 1/2 to 2/3, not
   1/3 to 1/2. 1/3 is the depletion *width* exponent of a graded junction; it would apply only if Δn_eff tracked W
   rather than the removed charge. m = 1/2 is right under both readings and is the default. The code accepts m in
   [1/3, 2/3]. The analytic C(V) uses m_cap = 1 − m, so that dλ/dV ∝ C(V) (m = 1/3 then means a hyperabrupt
   C-V exponent of 2/3).
2. **07 P2-5 "+29 %".** The exact static gain increase for current drive is 1/(1 − L) = 1 + α·R_th,H·P. At
   y = 0.00904 /mW and 25.5 mW that is **+23 %**. The 29 % is the approximation 1/(1 − 0.23).
3. **07 P1-11 "the TCR adds about ±10 %" understates the self-heating TCR.** With the textbook's y (fitted to the G25
   droop), dP/dcode from code 19,500 to 49,300 varies **1.87×** in voltage drive (not 2.53×) and **4.55×** in current
   drive. Only the ambient part is about ±10 %: +80 K of die temperature lowers the power at a fixed voltage code by
   5.7 % (06 E1 says about 8 %, which ignores the self-heating). Tested. This matters for gain scheduling.
4. **NRZ level convention.** 07 P1-7 writes "bias − Vpp/2", the centered convention. With the printed G26 axis
   (≥ 1.0 V) and Vpp ≤ 1.3 V, that never reaches forward bias (the lowest level is 0.35 V). A single-ended
   rail-to-rail CMOS driver on one terminal, with the DC bias on the other, gives V_bias − Vpp and V_bias, which reach
   −0.3 V at 1.0 V. Both are implemented. Which terminal the G26 "bias" axis refers to is not published.
5. **dac_pwm.** PAT-2 does not say whether the 8-bit DAC is a voltage or a current DAC; the code assumes voltage, so
   P = <V²>/R(T). The top 256 codes (N = 255) saturate at full scale. The slew limit acts on the period-average power
   fraction u, so a slew in codes/s maps to u only approximately. The ripple bypasses the settle filter.
6. **P1-8 capacitance.** For a large swing the exact per-transition dissipation, averaged over up and down, is
   ½·Vpp·ΔQ, so the code uses C_eff = ΔQ/Vpp over [V_lo, V_hi]. Verified against an ODE of a nonlinear C_j(V)
   charged through R (1e-4). Up and down transitions differ individually for a nonlinear C; their mean does not.
   `add_load_share=1` gives the R_drv share to the driver; leave it 0 if `E_tr` already includes the load.
7. **Supply limit (P1-12).** One droop factor s per world: I and V scale by s, P by s². The TCR is not re-solved
   within that tick. For dac_pwm the current sum uses rms currents (an upper bound on the mean supply current).
8. **Tick-average power** is the law at the tick-average drive, not the average of the instantaneous power. It is
   exact without a settle filter and slew limit, and second order during a transient within one tick.
9. **Textbook reproduction.** With y = 0.00904 /mW and R0 = 45.41 Ω (backed out of 55.9 Ω at 1.195 V), the model
   gives a Δλ/V² droop of 21.73 % from 0.3 to 1.4 V, against 21.74 % in the G25 read-off. A one-parameter fit of the
   whole G25 V->λ curve gives 0.348 nm/mW with a 32 pm rms residual; without TCR the residual is 338 pm. y is an
   effective parameter: α and R_th,H are not separately identifiable (06 E1).

## Open

- The P1-10 fault `dac_glitch` (a scenario-bound channel mask) has no implementation here. A code-transition glitch
  (major-carry glitch energy) would be a per-tick additive term on u after the FIFO; tell me the scenario interface
  and I will add it, or the faults owner can apply it to `cmd`.
- Heater DAC output noise is not modelled (no key exists).
