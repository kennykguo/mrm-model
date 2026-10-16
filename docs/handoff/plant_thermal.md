# Handoff: plant_thermal (`mrm/plant/thermal.py`, `mrm/plant/selfheat.py`)

Owner files: `mrm/plant/thermal.py`, `mrm/plant/selfheat.py`, `tests/verification/test_thermal.py`,
`tests/verification/test_selfheat.py`. Contract: ARCHITECTURE 5.3; 07 P1-2, P0-1, P0-10, P1-3 (reading side), P2-5.
Tests: 74 pass (48 in test_thermal.py, 26 in test_selfheat.py; section 7 lists the review additions), in about 15 to
20 s with numba and about 23 s with `NUMBA_DISABLE_JIT=1` (machine load average about 10).

## 1. The hook for `mrm.params.realize` (exact signature)

```python
from mrm.plant.thermal import assemble, InterimRom, ThermalScales, LayoutSites, load_rom, rom_path

assemble(rom: ThermalRom | InterimRom,
         layout: LayoutSites | Mapping[str, Mapping] | None,
         n_rings: int,
         scales: ThermalScales | Mapping[str, Any] | None = None,
         *, sources: Sequence[str] | None = None, observers: Sequence[str] | None = None,
         temp_sensors: Sequence[Mapping] | None = None, ring_pitch_um: float | None = None,
         prune_abs_k_per_w: float = 0.0) -> ThermalSystem
```

- `rom`: `load_rom(rom_path(family, film_nm))` when `fidelity.thermal_model: rom` (it checks sha256 against the
  sidecar), else `InterimRom(**values)`.
- `layout`: either a `LayoutSites` or the raw `layout.sites` mapping `{name: {x_um, y_um, kind, [ring]}}` (parsed by
  `LayoutSites.from_sites(sites, n_rings, ring_pitch_um, temp_sensors)`; missing ring sites go on a line at
  `ring_pitch_um`; `temp_sensors` is the `temp_sensors` table with `site` or `attach: ring_<i> | bank | die`).
- B: every `InterimRom` field and every `ThermalScales` field can be a scalar or a (B,) array; `z_heater_ring_k_per_w`
  can also be (B, N). B is the common broadcast length (1 or B, else ValueError).
- `sources` / `observers`: kind subsets. They are the main cost lever (section 5).

### Schema key -> `InterimRom` field (SI after `mrm.params` conversion)

| schema `ring.thermal.*` | InterimRom field | note |
|---|---|---|
| `z_heater_ring_k_per_mw` (derived, level ring) | `z_heater_ring_k_per_w` | K/W after conversion; (B,) or (B, N) |
| `z_ratio_opt_heater` | `z_ratio_opt_heater` | must be >= 1 (raises otherwise) |
| `tau_fast_s`, `share_fast_frac`, `tau_slow_s` | same | shares are absolute: slow = 1 - fast - tail (schema doc) |
| `tail_tau_s`, `tail_share_frac` | same | |
| `selfheat_tau_s` | same | sub-us pole of the waveguide self-heating path (07 P0-1) |
| `selfheat_share_frac` | same | **recommend passing None** (= (z - 1)/z, see 6.5); a given value that breaks passivity warns |
| `xt_nearest_frac`, `xt_floor_frac` | same | |
| `xt_tau_ref_s`, `xt_dist_ref_um`, `xt_tau_min_s` | same | |
| `xt_tau_far_s` | same | None = the printed X2 table (4.14-8.99 ms); a number = one far value |
| `xt_far_um` | same | used only by `xt_tau_law="literal"` (see 6.1) |

**Requested new keys** (all optional; the defaults are documented in `InterimRom` with provenance):
`heater_self_ratio` (R_th,H / Z_heater->ring(0), default 1.5 assumed, >= 1), `heater_node_tau_s` (1 us assumed),
`xt_next_frac` (textbook corner only), `xt_tau_law` (`capped` | `literal`), `crosstalk_dynamics` (`imec` | `self`, the
textbook corner), `site_ref_um` (60), `pd_offset_um` (0, 25), `temp_sensor_offset_um` (40, 0), `driver_offset_um`
(None). `InterimRom.textbook_corner(**kw)` builds the 07 P1-2 pessimistic corner. Note on `heater_self_ratio`: the
calibrated quantity is y = TCR x R_th,H (CAL-4); for a given TCR hypothesis, R_th,H = y/TCR, so the ratio is
y/(TCR Z0) per world (textbook y = 0.00904 /mW: 1.48 at TCR 1e-3 under H_solver).

### `uncertainty.thermal.*` -> `ThermalScales`

`ThermalScales(dc_scale=..., time_scale_fast=..., time_scale_slow=..., scaling_hypothesis=..., crosstalk_scale=...,
package_tail_tau_s=..., package_tail_share_frac=..., tau_split_s=3e-4, transport_delay_s=0.0)`.
- `dc_scale`: scalar/(B,) global, or `{"*": f, "heater->ring_mode_weighted": f, "heater->*": f, ...}` (factors that
  match a lag multiply). **Put `rom_dc_scale` under `"*"`**: it measures a conductance error of the whole network;
  applying it to one group under `scaling_hypothesis="resistance"` gives the heater and absorption paths of the same
  ring different time constants.
- `scaling_hypothesis`: str, (B,) strings, or int codes in the order `SCALING_HYPOTHESES = ("resistance",
  "capacitance", "independent")` (the order written in 07 P1-2). Default `"resistance"`.
- `package_tail_*`: replaces the interim tail (or adds one to every ROM self pair), keeping each pair's DC; the other
  tailable self poles keep their ratio. A tail given here is **absolute**: its tau is not multiplied by
  `time_scale_slow` or by the 'resistance' factor (07 4.8 lists it as its own 1-10 ms interval). The interim model's
  own `tail_tau_s` is a family value and does get `time_scale_slow`.
- `crosstalk_scale`: scales the ring-to-ring residues only. It **never moves tau**, also under 'resistance': it is a
  magnitude uncertainty (layout, metal shunts, EIC stacking), and a larger crosstalk from a shorter path or a metal
  shunt comes with a shorter tau (04 2.4), not a longer one. Under 'resistance' tau follows the `dc_scale` factor only.
- `transport_delay_s` (new, 07 P1-2 and 4.8: 0 to 2 us): accepted per world and stored as
  `ThermalSystem.transport_delay_s` (B,). The thermal kernels do not apply it (07 P0-6, one owner per delay): the
  heater actuator FIFO adds it as whole ticks.
- A per-world `scaling_hypothesis` list now sets B like any other (B,) field.

### Assertions `realize()` should run (07 P0-1 item 2, P0-4 step 5)

```python
from mrm.plant.thermal import assert_cross_lag_gain
from mrm.plant.selfheat import lorentzian_bounds
p_max, dpdt_max, d2p_max = lorentzian_bounds(p_peak_w (B,N,L), hwhm_m (B,N,L), s0_m_per_k (B,N), beta_per_k)
assert_cross_lag_gain(system, h_s, dpdt_max)      # raises ValueError if the lagged gain >= 0.05
```
Interim defaults, +9 dBm, Q 3800, 8 rings: 1.9e-3 at 20 um pitch, 9e-4 at 30 um, 3.3e-4 at 60 um for h = 1 us. The
gain grows about linearly with h (0.19 at 100 us and 20 um pitch). For the long steps of 07 P0-10 (`substep_policy:
auto`) assert at the largest h, and if it fails run the implicit step with `n_outer > 1` (Gauss-Seidel bank sweeps;
the error after n sweeps is about gain^n), which makes the cross-ring coupling implicit.

### `PlantParams` representation

07 P0-4 lists thermal residues as (B, N_dst, N_src, K). The runtime representation is sparse instead: one first-order
lag per (observer, source, pole) with `taus_s`, `res_k_per_w` (B, M) and shared index arrays. A shared pole set cannot
hold the interim crosstalk (one tau per distance) or the 'resistance' scaling (tau follows each pair's residue
factor). Dense views exist: `dc_gain(sys)` (B, O, S), `frequency_response`, `step_response`, `state_space`. Store the
`ThermalSystem` object (or its arrays) in `PlantParams`.

## 2. For the plant / kernel owner (`plant.py`, `sim/kernel.py`)

Per substep h (all kernels `@njit`; `*_b` = one world for use inside `prange` over B):

1. Coefficients once per distinct h: `a, c0, c1 = sys.coefficients(h)` ((B, M) each). For adaptive steps inside the
   kernel use `thermal_step_h_b` (FOH coefficients computed inline by `foh_scalar`).
2. Inputs `u0`, `u1` (S,) in W at the start and end of the step: heater, junction and driver are held (u1 = u0);
   aggressor waveforms that are piecewise linear pass their start and end values (exact FOH, 07 P0-10).
   Waveguide-absorption entries are ignored by the implicit step.
3. Heater TCR (electrical owner): `pre, g = observer_split_b(x, a, c0, c1, lag_src, obs_ptr, heater_obs[i],
   heater_src[i], u0, u1, src_scale)` -> `electrical.heater_power_row(t_heater_pre = T_base + pre, g_self = g)`. For
   sources not yet known in this step (other heaters, optical absorption), pass the previous values: those couplings
   then lag one step (they are 1.5 to 3 % with ms poles, or the weak absorption -> heater-resistor path).
4. Self-heating on: `selfheat.implicit_step_b(x, a, c0, c1, *sys.kernel_arrays(), bus_order, u0, u1, src_scale,
   t_base_end_k, p_prev, t_prev, pabs_fn, commit_fn, args_f, args_i, p_max, d2p_max, max_iter, tol_k, n_outer,
   g_work, t_obs, p_new, t_ring, n_iter) -> n_fail` (positional; use max_iter 10, tol_k 1e-9, n_outer 1 at 1 us).
   Then `p_prev[:] = p_new; t_prev[:] = t_ring`. `implicit_step` is the batched form; `SelfHeatWork` wraps both.
   Self-heating off: `thermal_advance_b(x, a, c0, c1, lag_src, obs_ptr, u0, u1, src_scale, t_base, t_obs)` with the
   absorbed powers (if any) in u.
5. `t_obs` (O,) are absolute temperatures (K) of every observer at the end of the step; `t_obs[ring_obs[k]]` equals
   `t_ring[k]`.

- `pabs_fn(t_k, k, args_f, args_i) -> (P_abs W, dP_abs/dT W/K)` and `commit_fn(t_k, k, args_f, args_i) -> None` are
  numba functions passed as arguments (`commit_fn` runs after ring k is solved, in bus order, e.g. to propagate the bus
  field; `commit_none` is a no-op). `p_max` (N,) must bound P_abs; `d2p_max` (N,) must bound |d2P_abs/dT2| (W/K^2):
  with it the solver never skips a root (exact branch continuation). Kernels taking function arguments are not cached
  by numba (dispatcher types miss the cache), so expect about 1.5-2 s of compile per (pabs_fn, commit_fn) pair.
- "Precomputed arguments" path: `pack_lorentzian_args` + `pabs_lorentzian` (per ring a resonance law with s0, beta;
  per line a Lorentzian peak). The optics owner can fill these rows each substep from its exact ring.
- Reset protocol (07 P0-1 item 6): `t_prev = -inf` selects the cold branch, `+inf` the hot branch; `laser_ramp` = start
  at the cold temperature and ramp the line powers.
- Non-converged solves (`n_fail > 0`) happen while a ring crosses between branches (a fold, status 1) or when
  `p_max` does not bound P_abs (status 2, a caller error: the solve reached the bracket end without a sign change;
  before the review this was reported as converged with |F| up to 13 K). The state then receives the power that puts
  the ring at the returned point, before its root: the ring lags, never overshoots.
- Multi-tick steps (07 P0-10 `auto`): the ROM step is exact for any h; the implicit solve is the h -> large
  quasi-static root with branch continuation (verified: the hysteresis edges come out within 0.003 HWHM).
- Base temperature `T_base` is additive (no filter). A TEC/package filter, if wanted, belongs to the scenario.
- `observers=("ring_mode_weighted", "heater_resistor")` drops the PD and sensor observers when the dark current
  temperature dependence and the temperature sensors are not used.

## 3. For the optics owner (`mrm/plant/optics.py`)

Provide `pabs_fn` with the signature above (absolute ring temperature in, total absorbed power summed over lines,
directions and orders out, with its T derivative), plus bounds `p_max` (e.g. eta x total line power at the ring) and
`d2p_max` (per line about 2 p_peak s^2/HWHM^2, exact for a Lorentzian; `lorentzian_bounds` computes it) and `dpdt_max`
for the cross-lag gain assertion. If the ring field depends on upstream rings, update `args_f` in `commit_fn`.
An inexact dP/dT (finite differences) or a low d2p_max does not stall the solve (it falls back to secant/bisection
inside the sign bracket, tested), but then the no-skip guarantee near a fold is lost.

## 4. For the extraction owner (`mrm/extraction/thermal_fem.py`)

`load_rom` reads your `write_artefact` output natively (checked in the review against a real quick-run artefact):
`site_names` + `site_offsets_um` in the .npz (also `inputs.config.site_offsets_um` in the sidecar), the sidecar
`<stem>.json`, and the nested `verify()` report (`validation.{self, nearest_neighbour}.{max_db, max_deg,
max_time_err_frac_of_dc}`, the worse of the two is checked against 07 P1-3). Before the review the site offsets were
dropped (PDs and attached sensors were placed at the ring centre) and every load warned that the validation keys were
missing. Still needed from you: **`superposition_rel_err`** (the 8-ring bank, 07 P1-3), read from
`validation.superposition_rel_err` or `stages.<stage>.superposition_rel_err`.
Observation on the quick-run artefact (test_quick_extraction_reruns, K = 14): the waveguide_absorption -> ring
residues have a near-degenerate pole cluster (0.90, 1.03, 1.19 us) with residues of +1.1e4, -1.7e4, +6.4e3 K/W that
cancel to about 1e3 K/W, and the self validation is 3.7 dB / 21.9 deg (target 0.3 dB / 3 deg). If the production
run looks the same, the fit is ill-conditioned there. Fewer poles would also cut the runtime cost (section 5).
`save_rom(path, rom, sidecar)` writes the same content with these names:
- `.npz`: `taus_s` (K,) strictly increasing; `sources`, `observers` (unicode arrays of kind names);
  `self_res` (S, O, K); `cross_r_um` (D,) increasing; `cross_res` (S, O, D, K) (K/W); `site_offset_names` (n,) +
  `site_offsets_um` (n, 2). A `site_offsets_um` dict in the sidecar is also read.
- sidecar `<stem>.sidecar.json` (your `write_artefact` convention; `<stem>.json` is also accepted): `schema:
  "thermal_rom/1"`, `sha256` of the .npz bytes, plus `family`, `film_nm`, `inputs`, `tool_versions`, `mesh`,
  `created_utc`, and **`validation: {step_max_rel_err, freq_max_db_err, freq_max_deg_err, superposition_rel_err}`**
  (these exact keys; missing or out-of-tolerance values raise a RuntimeWarning at load, 07 P1-3).
- Distance convention adopted from your docstring: source centre (ring centre for ring sources, site centre for driver
  and aggressor) to observer centre (ring centre for ring/heater observers, site centre for PD/sensor). Same-ring pairs
  use `self_res`; a same-ring PD, sensor or driver site more than 1 um away from `site_offsets_um` (for example one
  shared driver block) uses `cross_res` at its actual distance instead, with a warning.
- Drivers: as in the interim model, a driver source exists only where the layout has driver sites, or at your
  default offset when the caller lists "driver" in `sources` (G26 drives from a separate EIC, 07 P1-3).
- Poles: at least one below 1 us (07 P0-1) and one above 1 ms (07 P1-2), else a warning. Lags with tau < 300 us are
  the fast layer (`time_scale_fast`), the rest slow; `ThermalScales.tau_split_s` changes the split.
  **Correction (2026-10-16, physics_thermal review):** that per-pole rule holds only for a pair whose residues have
  one sign (the driving-point pairs, and every interim lag). A non-collocated FEM pair (heater -> ring mode, every
  cross pair) has mixed-sign residues several times its DC on both sides of 300 us; with unequal factors they no
  longer cancel, and the step response overshot and dipped (metal_above_160 heater -> ring at x2 / x0.5: peak
  1.21 x DC, drop 0.22 x DC, |H| 1.27 x DC near 600 Hz; silicide_undercut_200 1.08 x DC). Such a pair (mixed signs,
  poles in both layers) is now time-scaled as a whole by `s_fast^w s_slow^(1 - w)`, w its step-response fraction
  at `tau_split_s` (`thermal._whole_pair_time_scales`; `ThermalSystem.meta['time_scale_whole_pairs']` counts the
  pairs), so its shape is the fit's. Equal factors give the former result bit for bit; the interim model is
  unchanged. `lag_layer` still labels each pole by its tau. Test:
  `test_thermal.py::test_rom_layer_time_scales_keep_every_pair_monotone` (six artefacts, three factor pairs).
- If you store k(T) scaling ratios (P2-5), tell me the key; `k_temp_residue_scale` currently uses assumed weights.

## 5. Cost (B = 1, Apple M2, one 1 us substep, 8 Lorentzian rings, n_outer = 1)

Measured while other agents' jobs ran (load average 3 to 10); repeat runs vary by about 40 %.

| observers / sources | lags M | implicit step |
|---|---|---|
| all (ring, heater, PD, 2 sensors; heater, wg, junction, aggressor) | 770 | 4.1 us |
| ring + heater resistor; all sources | 520 | 3.3 us |
| ring only; heater, wg, aggressor | 176 | 1.25 us |

About 4.5 ns per lag. The P0-9 budget (>= 0.1 simulated s per wall s at 1 us, about 10 us per step for the whole plant)
is tight with every observer on. Not done: merging lags that share (observer, tau) across sources.

**Review measurement with a real FEM ROM (quick-run artefact, K = 14 poles, 8 rings, one aggressor; compiled
`implicit_run_b`, machine load average about 10 on 8 cores, so about 2x slower than unloaded):**

| model / observers / sources | lags M | implicit step |
|---|---|---|
| ROM, all | 11 550 | 84 us |
| ROM, ring + heater resistor; heater, wg, aggressor | 3 808 | 34 us |
| same, `prune_abs_k_per_w=1.0` | 2 820 | 26 us |
| ROM, ring only; heater, wg, prune 1 K/W | 1 306 | 10 us |
| interim, all (same machine state) | 745 | 8.6 us |

With a K-pole shared ROM, M grows as K x sources x observers, so the FEM ROM is 4 to 15 times the interim cost and
far over the P0-9 budget unless observers/sources are restricted. Proposed follow-up (an ABI change, so not done in
the review): because every lag of one (source, tau) has the same dynamics, x_m = r_m z_(src, tau) exactly from a zero
or steady start, so the state can be one unit lag z per distinct (source, tau over all worlds) and the outputs a CSR
residue matrix times z (about S x K states and M multiply-adds of r_m z, with z in L1 cache).

## 6. Contract issues (said here instead of silently deviating)

1. **07 P1-2 crosstalk tau law is not monotone.** `max(0.1 ms, 4.1 ms (d/60 um)^2)` up to 100 um gives 11.4 ms at
   100 um, above every printed far value (4.1-9.0 ms, X2). Default law `capped`: tau = min(d^2 law, far(d)), floored at
   0.1 ms, with far(d) the printed X2 table (or `xt_tau_far_s`); equal to the d^2 law below about 60 um. The literal
   07 law is available as `xt_tau_law="literal"`. The X2 distances 160..660 um are inferred (100 um pitch continued).
   **Correction (2026-10-16, physics_thermal review):** the table and the 60 um anchor are imec's heat-source-to-
   victim distances (04 2.5: 60 um for the first neighbour at a 100 um pitch), but the assembly passed centre
   distances (imec's own geometry then gave 4.75 ms instead of the printed 4.14 ms). `_interim_lags` now passes
   d_cc - r_src - r_obs (`InterimRom.ring_radius_um`, default 7.5 um, the FEM ROM's ring; realize passes each ring's
   r_ref), with r = 0 for point sites; `site_fraction` keeps the centre distance (an assumed law). g26 nominal:
   nearest neighbour 2.85 -> 1.52 ms, own PD 0.71 -> 0.38 ms, aggressor -> ring 0 1.03 -> 0.62 ms.
2. **'capacitance' and 'independent' coincide numerically.** A DC scale can only come from a resistance change; under
   'resistance' each tau follows its residue factor (07 as written), under the other two the residue and tau scale
   independently. The label is kept for reports; the params owner may want to drop one or define it differently.
   Default is 'resistance' (07 P1-2: DC-only scaling is inconsistent).
3. **Self-path shares with a tail.** 07: "3.2 us (43 %) + 13.0 us (57 %), plus a 1-10 ms tail carrying 2-10 %".
   The schema reads slow = 1 - fast - tail. The `InterimRom` default keeps the measured 43:57 ratio
   (share_fast 0.4085, slow 0.5415, tail 0.05); a preset with share_fast 0.43 and tail 0.05 gives 45:55.
4. **07 P0-1 test (d) formula is first order.** The simulated shift equals the exact root of the static cubic to
   1e-9 HWHM. The linearized formula is within 4.4 % for +10 % at y = 1 (the test), but 7.3 % off at the steepest
   point y = 1/sqrt 3 (15 % at y = 0.3); its error scales with the step (0.8 % for +1 %). The test checks y = 1 at
   +10 % and the convergence for smaller steps.
5. **`selfheat_share_frac` and `z_ratio` are not independent.** For a passive network the per-pole residue matrix over
   {heater, ring} must be positive semidefinite, which needs share <= (z - 1)/z. The default construction
   Z_RR = Z_RH + (z - 1) Z0 / (1 + s tau_sh) uses exactly (z - 1)/z (1/3 at z = 1.5). Consequence: z = 1 (in-ring heater,
   identical distributions) gives no sub-us pole with the imec-shaped heater path, against P0-1 item 3. Use z >= 1.2
   with the interim model, or accept the passivity warning.
6. **04 V5 "63 % point 7.3 us"** is 7.35 us for 0.427/0.573 at 3.18/13.01 us (rounding only).
7. **Transport delay (07 P1-2, 4.8: 0-2 us) is not applied here.** Since the review, `ThermalScales.transport_delay_s`
   carries the per-world value to `ThermalSystem.transport_delay_s` (B,). Suggested owner: the heater actuator FIFO in
   `electrical.py` (extra whole ticks of heater latency). It then also delays the heater-resistor observer by <= 2 us,
   which is negligible for the TCR loop (poles 3.2 and 13 us). A per-lag delay with an input history in the kernels is
   the exact alternative if needed.
8. **Site paths in the interim model are assumed.** 07 gives only ring-to-ring crosstalk by neighbour rank. PDs,
   drivers, temperature sensors and aggressors use c(d) = clip(0.03 x 60 um/d, 0.015, 1) of Z0 with tau_xt(d). The
   aggressor's DC is calibrated against its target anyway (07 P1-4); the sensor and PD values are placeholders until
   the FEM ROM exists. **Since 2026-10-16** the floor of that law is its own field, `site_floor_frac` (default:
   xt_floor_frac, so nothing changed by default): imec's 1.5 % floor is a package term measured to 300 um with
   natural convection and applies at any distance on the die (a die sensor 1 mm away sees 1.5 % of Z0 per heater,
   33x the FEM ROM's TEC-held package, 0.09 %). The g26 choice `uncertainty.thermal.stack: side_by_side_tec_held` sets it,
   the crosstalk fractions and xt_tau_ref_s to the FEM values (physics_thermal review).
9. **Interim crosstalk with per-ring Z0 variation.** To keep reciprocity exact, heater_j -> ring_i uses Z0_j,
   waveguide_i -> heater_resistor_j uses Z0_j, and like-like pairs use sqrt(Z0_i Z0_j).
10. **The lagged cross-ring optical coupling (07 P0-1) is only valid for short steps.** Its per-step gain grows with
    h, so the multi-tick steps of 07 P0-10 can break the 0.05 bound at a small pitch. `n_outer` sweeps fix it; the
    contract does not mention this interaction.

## 7. Adversarial review (2026-10-09): changes made and why

Found and fixed (tests added for each in `tests/verification/test_thermal.py` / `test_selfheat.py`):
1. `load_rom` dropped the site offsets of the extraction's artefacts (`site_names` key, offsets in
   `inputs.config`), so PDs and attached sensors were placed at the ring centre; and it never read the nested
   validation report (always "missing" warnings). Fixed; `rom_validation_numbers(meta)` is public.
2. `solve_ring_temperature` returned status 0 (converged) with |F| up to 13 K when `p_max` did not bound P_abs.
   Now status 2, counted as a failure by `implicit_step_b`, and the state stays consistent with the returned point.
   A negative g (mixed-sign short-step fit) now uses the mirrored bracket instead of skipping the solve.
3. `pack_lorentzian_args` took B = N for (N, L) per-line arrays (np.atleast_3d), giving N rows of args; the
   batched `implicit_step` then read past the row count for B_world > N (no bounds check in numba). Fixed.
4. The batched kernels (`thermal_step`, `thermal_advance`, `thermal_outputs`, `implicit_step`) and `simulate` now
   refuse row counts other than 1 or B (a mismatch was a silent out-of-bounds read under numba).
5. `ThermalScales(scaling_hypothesis=[...])` with otherwise scalar scales raised "batch is 1"; the list now sets B.
6. Under 'resistance', `crosstalk_scale` also multiplied the crosstalk tau (a 1-10 % sweep moved tau_xt 1.4-14 ms,
   against the 04 d^2 law and against realize's own fallback bound). Now only `dc_scale` moves tau.
7. A package tail from `ThermalScales` was multiplied by `time_scale_slow` and the resistance factor (a drawn 10 ms
   tail could become 20 ms, outside 07 4.8). Now absolute.
8. `transport_delay_s` (07 4.8) had no field, so `realize()` would raise on `uncertainty.thermal.transport_delay_s`.
   Now accepted and exposed (not applied; FIFO owner).
9. ROM path added a driver source per ring at the ROM offset whenever the ROM had a driver source (inconsistent with
   the interim model and with G26's separate EIC); and a shared driver block far away still used the 30 um self
   table (about 10x too strong). Fixed as described in section 4.
10. `LayoutSites.from_sites`: a temp_sensor site referenced by a `temp_sensors` entry was added twice. Fixed.
11. `from_sites` put the PD of a ring without a PD site at the ring centre (and a missing per-ring driver at the ring
    centre). Missing rows are now NaN: the PD goes to the model's default offset, and that ring has no driver.
12. `implicit_step_b` now checks that `bus_order` is a permutation of the rings and that every ring has its ring
    observer, before it touches the state (a short or repeated bus_order used stale p_new for the optical lags).
13. Missing 07 P1-2 test added: "locking ring j steps ring i by the cross_opt residue" (dynamic, against the linear
    ROM with the one-substep lag, and at steady state). Also: the implicit step with no light equals the exact FOH
    kernel for piecewise-linear inputs.

Contract points that remain open: transport delay application (FIFO owner); ROM cost against P0-9 (section 5);
P0-4's dense (B, N_dst, N_src, K) residue array (sparse lags instead, section 1).