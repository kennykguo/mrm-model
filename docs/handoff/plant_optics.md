# Handoff: plant_optics (`mrm/plant/optics.py`)

Owner of `mrm/plant/optics.py` and `tests/verification/test_optics.py`. This note lists what other owners need to
connect to the optics, and the places where the contract (ARCHITECTURE v2 / 07) looks wrong or optimistic. Nothing
here edits another owner's file. Revised after the adversarial review (section 6 lists what the review changed).

## 1. Interfaces to call (all in `mrm.plant.optics`)

Compiled path (numba, no allocation; preallocate with `make_bus_outputs(N, L)` and `make_workspace(L)`):

```
bus_propagate_nb(p_in_w(L), bus_order(N) int64, bus_t2(N), bus_t2_out, p1(N),
                 phi0(N,L), phi1(N,L), a0(N,L), a1(N,L), eta0(N,L), eta1(N,L), psi(N,L),
                 k2_th(N,L), k2_dr(N,L), t_rt(N,L), rb(N,M_b) complex, dphi_dlamres(N,L),
                 two_level, use_tcmt, m_orders, with_slope,
                 out_rl(RING_LINE_OUTPUT_ROW_COUNT,N,L), out_r(RING_OUTPUT_ROW_COUNT,N), out_l(LASER_LINE_OUTPUT_ROW_COUNT,L), work(L), cw(COMPLEX_WORKSPACE_ROW_COUNT,L) complex, fw(REAL_WORKSPACE_ROW_COUNT,L))
ring_lines_nb(k, pin(L), p1_k, phi0, phi1, a0, a1, eta0, eta1, psi, k2_th, k2_dr, t_rt, rb, dphi_dlamres,
              two_level, use_tcmt, m_orders, with_slope, out_rl, out_r, p_thru(L), cw, fw)
dpabs_dlambda_res_nb(k, pin(L), p1_k, phi0, phi1, a0, a1, eta0, eta1, psi, k2_th, k2_dr, t_rt, rb,
                     dphi_dlamres, two_level, use_tcmt, m_orders, cw, fw) -> (P_abs W, dP_abs/dlambda_res W/m)
pabs_bounds_nb(k, pin(L), p1_k, a0, a1, eta0, eta1, k2_th, k2_dr, two_level) -> (p_max W, d2p_max W/rad^2)
pabs_slope_bound_nb(k, pin(L), p1_k, a0, a1, eta0, eta1, k2_th, k2_dr, two_level) -> dp_max W/rad     (new)
bus_finalize_nb(p_in_w, bus_order, bus_t2, bus_t2_out, out_rl, out_l)
```

Ready-made self-heating adapter (new; the `pabs_fn` / `commit_fn` ABI that `docs/handoff/plant_thermal.md` section 3
asks the optics owner for):

```
args_f, args_i = pack_selfheat_args(p_in_w, phi_ref, k2_th, k2_dr, a_rt, t_rt_s, eta, lam_m, t_ref_k, s0_m_per_k,
                                    beta_per_k=0.0, bus_order=None, bus_t2=None, bus_t2_out=1.0, psi=None, rb=None,
                                    nrz=None, method="exact", m_orders=3, commit_slope=False)   # one world, 1-D rows
selfheat_pabs(t, k, args_f, args_i) -> (P_abs W, dP_abs/dT W/K)      # pabs_fn
selfheat_commit(t, k, args_f, args_i)                                # commit_fn (bus walk + bus_finalize_nb)
selfheat_bounds(args_f, args_i, dt_max_k, p_max(N), d2p_max(N), dpdt_max(N))   # W, W/K^2, W/K
selfheat_views(args_f, args_i) -> 29 views: inputs to update per substep, out_rl/out_r/out_l to read
```

- Phase model inside the adapter: a rigid shift of each ring's comb, `lambda_res,k(T) - lambda_res,k(T_ref) =
  s0 [dT + beta dT^2/2]` (07 P0-3 R1, the same law as `selfheat.pabs_lorentzian`), applied through
  `dphi_dlambda_res` per line. `phi_ref` are the phases at `T_ref,k` for the current bias, laser wavelength and
  junction state.
- K^2, `a_rt` and `eta` are held at what the plant writes for the substep. Evaluate `a_rt` and `eta` at the previous
  ring temperature: their ~1e-3/K dependence then lags one substep (loop gain far below 0.05), and dP/dT stays exact
  for the solver's no-skip guarantee.
- The walk restarts by itself at the first ring in bus order, so Gauss-Seidel sweeps and later substeps need no
  reset. `selfheat_bounds` uses the walk buffer as scratch: call it between substeps.
- Batch: `np.stack` the rows of each world (same N, L, M_b give the same layout; the row length is even so the
  complex views stay 16-byte aligned). Verified in njit with `args_f[b]` rows.
- Cost: about 0.4-0.6 us of view overhead per evaluation on top of `dpabs_dlambda_res_nb`.
- Verified with `selfheat.solve_ring_temperature` in bus order (tests `test_selfheat_adapter_*`): roots satisfy
  T = T_other + g P_abs(T) against an independent `bus_propagate`, and the bounds hold over a 40 K scan. A cold start
  7 K from the root needs 13-16 certified-Newton evaluations; a warm restart needs none beyond the residual check.

Output fields: `RING_LINE_OUTPUT_*` per (ring, line) (p_in, thru, CW drop, CCW add-port output, refl, loss, absorbed, energies,
per-level thru/drop, per-level bus output pout1/pout0, dabs_dphi), `RING_OUTPUT_*` per ring (drop_sum = the monitor's optical
input over every line and order, abs_sum = the self-heating source, per-level drop sums P1/P0, dabs_dlam), `LASER_LINE_OUTPUT_*`
per line (bus output, refl returned to input, single-pass bound).

Inputs the plant must prepare each substep (07 P1-5, P1-6, P1-7, P0-2):
- `phi0/phi1`: ABSOLUTE round-trip phase at each LINE for junction levels 0/1,
  `2 pi [n_eff(lambda_l, T_ring) L(T) + fill L dn_eff(V)] / lambda_l`, without the coupler phase. The per-order
  backscatter uses the absolute order `floor(phi/2 pi)`; a phase re-anchored by a multiple of 2 pi that is not a
  multiple of 2 pi M_b gives the orders other beta values (07 P2-9 re-anchoring must respect this).
  Note (2026-10-16, physics_optics review): mrm.plant.phase prepares phi by translating the cold phase curve
  rigidly by the resonance shift (equation 2 there), not with n_eff(lambda, T); every order then moves by the same
  shift. The error against the modes artefact's n_eff(lambda, T) table is 0.43 % of the shift per FSR at the other
  orders (+-33 pm at the adjacent orders at the 7.6 nm g26 lock); the locked order is exact. Bound and consequences:
  the mrm.plant.phase docstring.
- `bus_t2` (N,): power transmission of the bus segment before each ring; the plant passes
  `mrm.plant.phase.bus_t2_from_plant_params(pp)` (1 for the first ring in bus order, exp(-bus_loss x spacing)
  after it; g26 0.99916 per segment). Until 2026-10-16 the plant passed none (every segment at 1).
- `psi`: the coupler ring-side phase (sum of both couplers). Extraction should define `coupler_phase_rad` as the
  ring-side self-coupling phase relative to the uncoupled bent guide of the same length, so it adds to phi.
- `k2_th/k2_dr`: per-pass power coupling at each LINE wavelength (never at the resonance, never per heater code).
  `OpticsParams` stores `kappa2_*` at `lambda_res_cold` plus `dln_kappa2_*_dlambda_per_m`; evaluate
  `K2(lambda_l) = K2_0 exp(dln * (lambda_l - lambda_res_cold))` per line.
- `a0/a1`: round-trip amplitude `exp(-alpha_i(V, T) L / 2)` per level; `eta0/eta1` = heat fraction per level.
- `t_rt`: `n_g(lambda_l) L / c` (only energies and TCMT rates depend on it).
- `rb`: `OpticsParams.backscatter_rb[b]` (N, M_b) as realize produces it (section 3).

Python path (analysis, calibration, tests): `ring_exact`, `ring_tcmt`, `ring_smatrix`, `bus_propagate` (returns
`BusResult`), `dpabs_dlambda_res`, `exact_to_tcmt` (the one TCMT mapping), `tcmt_to_exact`, `beta_to_rb`,
`dphi_dlambda_res`, `nrz_average`.

Parallel batches: every kernel writes its workspace (`cw`, `fw`, `work`) and outputs. With `prange` over B, give
each batch element its own (for example `cw` of shape (B, COMPLEX_WORKSPACE_ROW_COUNT, L), passing `cw[b]`).

## 2. For the self-heating, thermal and kernel owners

- Use the adapter of section 1, or build your own from `dpabs_dlambda_res_nb` (Newton evaluation), `ring_lines_nb`
  (commit; its through output times `bus_t2[next]` is the next ring's input) and `bus_finalize_nb` after the last
  ring. Test `test_kernel_walk_reproduces_bus_propagate` checks that this walk equals `bus_propagate_nb` bit for bit.
- `thermal.cross_optical_lag_gain` needs `dpabs_dt_max` (B|1, N) W/K: `selfheat_bounds` returns it (`dpdt_max`),
  from the new exact Airy slope bound `pabs_slope_bound_nb` (max |dP/dphi| in closed form; it reduces to
  (3 sqrt 3/8) peak/HWHM in the Lorentzian limit).
- Bound validity: `pabs_bounds_nb`, `pabs_slope_bound_nb` and therefore `selfheat_bounds` are verified for
  `|r_b| <= MAX_ROUND_TRIP_BACKSCATTER_FOR_VERIFIED_POWER_BOUNDS = 1` (beta up to FSR/2 pi, about 290 GHz at 1.8 THz; 07 P2-2 expects 1 to 10 GHz). Beyond
  it the curvature bound can fail (8 % above d2p_max at |r_b| = 1.6). realize's Gaussian beta with the schema's
  100 GHz splitting ceiling stays far inside.
- **Compute budget (07 P0-9): a serious concern.** Measured on the shared M2 (load average 10-18 during the runs,
  so the numbers spread):
  - one full bus pass (N = 8, L = 8): 9-15 us exact with backscatter, 15-27 us NRZ two-level, 23-43 us NRZ with
    slopes, 5.7 / 8.2 us without backscatter (Airy fast path, automatic when a ring's `rb` row is all zero), and
    12-20 us TCMT M = 3;
  - one Newton evaluation of one ring (8 lines): 0.66 / 1.1 us on the Airy path (CW / NRZ), 2.4 / 4.1 us with
    backscatter (CW / NRZ), and 3.5 us TCMT (NRZ).

  A substep with N rings costs about N x (evaluations per ring + 1 commit) single-ring evaluations, for example
  8 x 2 x 1.1 us = 18 us on the Airy path with NRZ against the 10 us budget for the whole plant. Suggestions:
  - warm-start Newton from the previous root (usually one evaluation away from a fold);
  - keep `fidelity.backscatter: off` (the schema default) for long soaks, and on for BiDi and reflection studies;
  - use prange over B;
  - consider evaluating NRZ two-level only when `p1` is not 0 or 1.

## 3. For the params/realize owner (`mrm/params/realize.py`)

Done on your side (checked): `OpticsParams.backscatter_rb` (B, N, M_b) complex with `rb_n = 2 pi beta_n / FSR`,
beta_n complex Gaussian with E|beta_n|^2 = (splitting/2)^2, uniform phase, its own stream;
`ring.optics.backscatter_orders` (default 5) and `ring.optics.tcmt_orders` (default 3). These match the optics
convention (beta = g/2 pi, splitting = 2|beta|). Two small requests:
- `ring.optics.backscatter_orders` and `tcmt_orders` should be bounded `>= 1` in the schema. The kernels now treat
  M_b = 0 as no backscatter (exact: Airy path; TCMT: g = 0), and the Python wrappers reject it.
- With `fidelity.optics_model: tcmt`, `tcmt_orders` should be odd. Even M is now centred on the line, but odd M is
  symmetric about every resonance.

## 4. For the data-path owner (`mrm/plant/datapath.py`)

- P1/P0 at the monitor of ring k: `out_r[RING_OUTPUT_ROW_LEVEL_ONE_TOTAL_CLOCKWISE_DROP_POWER, k]`, `out_r[RING_OUTPUT_ROW_LEVEL_ZERO_TOTAL_CLOCKWISE_DROP_POWER, k]` (every line, other rings
  averaged over their own data).
- P1/P0 at the receiver for line l modulated by ring k: `out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ONE_BUS_OUTPUT_POWER, k, l]`, `out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ZERO_BUS_OUTPUT_POWER, k, l]`.
- These are exact for independent data on different rings (the single-pass cascade factorizes; test
  `test_nrz_average_factorizes_over_independent_rings`). `LASER_LINE_OUTPUT_ROW_BUS_INPUT_RETURNED_REFLECTION_POWER` is now exact for independent data too (it uses
  E[T^2] for the double crossing).

## 5. Contract issues (please review)

1. **07 section 4.2: "TCMT within 1 % of the exact ring for Q > 1000 within +-0.3 FSR".** The deviation scales
   exactly as c/F^2. It comes from the sinh(x)/x prefactors and the cosh(x) background of the exact pole sum, not
   from the order truncation, so more orders do not help (they make the through port slightly worse). At FSR
   1.8 THz and Q_L = 1000 (F = 7.9) it is 1.5 % (M = 1), 2.0 % (M = 3) and 3.0 % (M = 7). 1 % needs F of about 10
   (M = 1), 11-14 (M = 3) or 14-17 (M = 7), depending on the coupling. The test asserts 1 % for Q_L >= 2000
   (M = 1, 3) and >= 2500 (M = 7) on the through power, the drop power relative to its peak and the loss power
   relative to its peak, plus the 1/F^2 law from F = 8 to 110.
2. **07 P2-4: "the TCMT Lorentzian tail under-estimates the adjacent-400 GHz tail by about 20 to 30 %".** For the 03
   reference ring the single-order TCMT gives 15.2 % at FSR 1.8 THz (the default window B), 18.9 % at 1.6 THz,
   20.7 % at 1.52 THz and 24.0 % at 1.4 THz (window A). The 3-order complex sum over-estimates by 3-5 %; 7 orders
   are within 1 %.
3. **03 section 3.2 vs 07 P1-5.** 03 recommends evaluating kappa at the resonance; 07 P1-5 (normative) says at each
   line. The optics follows 07 (K2 per ring and line). The two differ by about 0.3 % within a linewidth but by
   about 15 % for a line one FSR away at 1.5 %/nm, so the difference matters for adjacent-order leakage.
4. **Backscatter "agrees with TCMT to O(1/F^2)" is a modelling choice, not physics.** A real scatterer at an
   arbitrary position deviates from TCMT at O(1/F): one scatterer at L/4 gives 8.2 % of the reflected power at
   Q_L = 3500, against 0.11 % for the pair. The module uses a pair of identical scatterers at L/4 and 3L/4 with a
   per-order reflectance flat at every resonance, which cancels the O(1/F) terms exactly. The complex port
   amplitudes agree with TCMT to O(1/F^2) at every order (new test), which also locks the phase convention of the
   mapping. The effect is far inside the beta uncertainty (1 to 10 GHz, random per order).
5. **External reflection (07 P2-2 `scenario.external_reflection`): not built (open).** A design note for whoever
   builds it:
   - The mean effect needs a backward pass. By mirror symmetry, light entering a ring at its through port sees the
     forward response of the same ring with `rb -> conj(rb)`, with ports 0/1 and 2/3 and CW/CCW swapped (verified to
     rounding for exact and TCMT; `test_backward_input_equals_mirrored_ring`). So the forward kernels can be reused.
   - The backward light heats the ring through the CCW mode. It reaches the monitor only through backscatter.
   - The phase-averaged backward pass is not enough. With backscatter, the dominant term is the interference
     between the forward and backward excitations of each ring. Its relative size is about
     2 (2|g|/kappa) sqrt(P_back/P_fwd), for example 6 % for 2|g|/kappa = 0.3 at -20 dB, against 1 % for the mean.
     It drifts with the reflector phase. Modelling it needs a coherent relative phase per (ring, line), which is the
     P2-10 S-matrix netlist (deferred), or a dedicated coherent backward pass with a lumped drifting phase per ring.
   - Either way, backward light makes ring k depend on rings after it, so the self-heating walk must lag that part
     by one substep.
6. **Single pass.** The bound returned per line (`LASER_LINE_OUTPUT_ROW_FORWARD_POWER_RELATIVE_ERROR_BOUND`) is `1/(1 - eps)^2 - 1` with
   `eps = sum_{i<j} |r_i||r_j|` over all pairs (corrected; see section 6). It is the exact two-ring worst case,
   tight to leading order for N in-phase reflectors, and below 1e-3 in normal operation. With NRZ it uses the RMS
   reflection of each ring and bounds the data-averaged error to leading order.
7. Couplers are lossless point couplers; coupler excess loss must be folded into `a_rt`. The multi-order TCMT sum
   is not exactly energy conserving (shared-port leakage); the exact ring is (1e-14).

## 6. What the adversarial review changed

- Fixed: the TCMT order window was not centred for even M. M = 2 used orders n_c and n_c + 1 on both sides, giving
  -5.3 dB against +2.0 dB at -+0.4 FSR. It now takes the M nearest orders.
- Fixed: `LASER_LINE_OUTPUT_ROW_BUS_INPUT_RETURNED_REFLECTION_POWER` under NRZ used E[T]^2 for the double crossing of upstream rings; it now uses E[T^2], from the
  identity E[(P T)^2] = THRU0^2 + (THRU - THRU0)(THRU1 + THRU0).
- Fixed: the single-pass bound. The earlier max-over-cuts formula is not a bound for three or more rings: rings
  detuned +4, +1.6 and -4 half-widths with r_b = 0.15 give a 14.8 % coherent error against a reported 8.5 %. The
  all-pairs eps bounds it (16.5 %); checked by an independent coherent-cascade test, and over 3000 random lossless
  chains, where it is tight to 1e-4.
- Fixed: `pabs_bounds_nb` now caps each line's peak at loss fraction 1, and documents its validity domain
  (`MAX_ROUND_TRIP_BACKSCATTER_FOR_VERIFIED_POWER_BOUNDS`). Added `pabs_slope_bound_nb` and the self-heating adapter.
- Fixed: the module docstring said that where the backscatter sits changes the port powers at O(r_b^2/F). It is a
  relative O(1/F) of the reflected power and O(F r_b^2) of through and drop.
- Added input validation (finite inputs; eta, p1, a1, eta1, bus_t2 and bus_t2_out in range; non-negative line power;
  method names).
- Added tests:
  - Strengthened the 1 % test (drop and loss relative to their peaks; M = 7).
  - Centred-window test; complex-amplitude agreement test; explicit Lorentzian-limit test.
  - Kernel bus-order walk equals `bus_propagate_nb`; dark lines and open segments give zeros, not NaN.
  - Three-ring coherent bound; NRZ `refl_in` against the joint-state expectation.
  - Mirror identity; bounds over |r_b| <= 1, including the out-of-domain failure.
  - Self-heating adapter tests; the JIT/no-JIT comparison extended to the new paths.
