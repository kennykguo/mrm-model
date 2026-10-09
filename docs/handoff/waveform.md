# Handoff: waveform mode (`mrm/waveform/`)

Owner of `mrm/waveform/{__init__,tcmt_time,driver,eye,lut}.py`, the artefact
`mrm/data/extracted/waveform_lut_nrz.npz` + sidecar `waveform_lut_nrz.json`, `tests/verification/test_waveform.py`
(42 tests) and `tests/validation/test_eo_bandwidth.py` (5 tests). Both files pass in about 20 s, and also under
`NUMBA_DISABLE_JIT=1` (about 25 s; the interpolation-accuracy test is skipped there). A subprocess test checks that the
kernels give the same numbers with and without numba (rtol 1e-10).

Contract items: 07 P2-6 (waveform mode and the normalized LUT), D9 (relOMA = OMA/P_in), 4.3 CAL-10 and 4.4 VAL-3
(the `eo_bandwidth_at_bias` replica physics). Physics: docs/research/03 sections 2.2-2.5, 5.4-5.6, 9.2-9.3.

## 1. What it does

1. `tcmt_time`: the CW/CCW TCMT of one ring with backscatter, driven by the INSTANTANEOUS junction voltage through
   tables Delta(V), kappa_i(V), eta(V). `RingTCMT.from_plant_params` builds the tables with the plant's own phase model
   (`mrm.plant.phase.ring_optics_row`: R1 thermal shift, junction dneff(V) from the LUT or the analytic table, loss
   alpha(V, T), K^2 at the line, logarithmic TCMT mapping of `mrm.plant.optics`). It can solve the ring temperature
   that puts the laser at a requested normalized detuning `x_bias = 2 Delta/kappa` at the bias. Integrator: an
   exponential step that is exact for constant rates. It uses a Simpson exponent and a first-order forcing correction.
   Against DOP853 the observed error is fourth order: 1.5e-7 of max |a| at h = 0.25 ps. The small-signal response
   is the exact linearization, including backscatter and loss modulation. For g = 0 it equals the Sacher & Poon /
   Karimelahi closed form to 1e-10 (poles -kappa/2 +- i Delta, zero at -(kappa_i + kappa_d)). It reproduces every row
   of the 03 section 5.6 bandwidth and peaking table to its printed digits (including the q rows).
2. `driver`: patterns are PRBS7/9/15/23/31 (the same sequence as `mrm.plant.datapath.prbs_bits`) and cyclic Markov
   patterns with a given mark density and transition density. The NRZ source has Gaussian edges (an assumed form) and
   drives the junction through R_drv. The junction network is one node (C_pad = 0) or two nodes, with C_j(V) and
   R_s(V) from the junction table. It is integrated by RK4 with substeps. The energy dissipated in R_s per
   transition equals `electrical.data_activity_heat_row` (07 P1-8) to 2e-3, which is a cross-module check.
3. `eye`: `simulate` is the end-to-end run. `eye_metrics` gives levels P1/P0, OMA, ER, inner eye height and width,
   and relOMA. `eo_response` and `eo_bandwidth` give the 3 dB point (|H/H0|^2 = 1/2) and the peaking.
4. `lut`: the normalized-coordinate LUT of 07 P2-6, its numba lookup for the engine, the builder, and the validation
   helpers.

Definitions:
- **OMA** is the largest |P1(phi) - P0(phi)| over the sampling phase phi. P1 and P0 are the means over the ones and
  the zeros of bit k sampled at k UI + phi, for phi in [0, 2 UI). The maximum is refined by a parabola vertex, so
  the OMA converges quadratically in samples per bit.
- **Eye height** is taken at its own best phase. It is a min/max over bits, so it converges only to first order.
- **Mean** is the time average over the period of the cyclic pattern.
- **Level convention**: bit 1 is the higher reverse voltage (integrator decision 6).

## 2. Public API (exact signatures)

```python
# mrm.waveform.tcmt_time
tcmt_steady_nb(delta, kappa_i, kappa_e, kappa_d, g, s_in) -> (a1, a2)                                    [njit]
tcmt_run_nb(h, s_in, kappa_e, kappa_d, g, delta_half, kappa_i_half, a1_0, a2_0, out_a1, out_a2)           [njit]
tables_at_nb(v, grid, delta_tab, kappa_i_tab, out_delta, out_kappa_i)                                     [njit]
tcmt_run_tables_nb(h, s_in, kappa_e, kappa_d, g, v_half, grid, delta_tab, kappa_i_tab, out_a1, out_a2,
                   work_delta, work_kappa_i)                                                              [njit]
port_powers(a1, a2, s_in, kappa_i, kappa_e, kappa_d, eta=1.0) -> dict(thru, drop, refl, add, loss, abs, u_cw, u_ccw)
optical_response_rates(kappa_i, kappa_e, kappa_d, delta, f_hz, *, ddelta=1.0, dkappa_i=0.0, g=0j, s_in=1.0,
                       port="through") -> complex array
h_opt_closed_form(s, kappa_i, kappa_e, kappa_d, delta) -> complex array
class RingTCMT(kappa_e, kappa_d, g, v_grid, delta_tab, kappa_i_tab, eta_tab, omega_l=nan, meta={}):
    from_rates(kappa_i, kappa_e, kappa_d, delta_bias, *, v_bias=0.0, ddelta_dv=0.0, dkappa_i_dv=0.0, g=0j, eta=1.0,
               v_span=(-0.5, 4.0), n_grid=451)
    from_plant_params(pp, b=0, ring=0, *, v_bias, x_bias=None, t_ring_k=None, t_heater_k=None, line=None,
                      lambda_line_m=None, backscatter=None, v_grid=None, order=None)
        # x_bias XOR t_ring_k. Lock order: `order` (absolute), else the cold-resonance order (with x_bias; the
        # convention of mrm.control.models; warns when that needs a comb shift outside -0.5..1.5 FSR) or the order
        # nearest the line (with t_ring_k). Order m + 1 sits one FSR BLUE of order m.
    delta(v) kappa_i(v) eta(v) kappa(v) x(v) fwhm_hz(v) tau_p_s(v) r_e(v) ddelta_dv(v) dkappa_i_dv(v)
    steady(v, p_in_w=1.0) -> dict;  run(v_half, h_s, p_in_w=1.0, t0_s=0.0) -> TcmtTrace
    optical_response(v_bias, f_hz, *, port="through", p_in_w=1.0, loss_modulation=True)   # W per V of junction V

# mrm.waveform.driver
prbs(order, n_bits=None) -> uint8;  markov_bits(n_bits, mark_density, transition_density, seed=0, n_tries=64)
   # transition_density in (0, 2 min(p1, 1 - p1)]
MAX_HALF_STEPS = 25_000_000   # DriverJunction.run (hence simulate) refuses larger grids (~100 B per half step)
pattern_stats(bits) -> (mark density, cyclic transition density);  rise_to_sigma(t_r);  sigma_to_rise(sigma)
nrz_source_nb(t, bits, t_bit, sigma, v0, v1) -> V                                                       [njit]
junction_rc_run_nb(t0, h_half, bits, t_bit, sigma, v0, v1, r_drv, c_pad, grid, c_tab, rs_tab, n_sub,
                   out_vj, out_vsrc)                                                                    [njit]
class DriverJunction(v_grid, c_j_tab, r_s_tab, r_drv_ohm, c_pad_f=0.0, rise_time_s=0.0, meta={}):
    from_plant_params(pp, b=0, ring=0, *, rise_time_s, c_pad_f=0.0, r_drv_ohm=None, r_s_scale=1.0)
    c_j(v) r_s(v) sigma_s h_electrical(v_bias, f_hz, include_driver=True) rc_pole_hz(v_bias) tau_min_s()
    levels(v_bias, swing_vpp, level_mode=0) -> (v_lo, v_hi, flags)        # electrical.nrz_levels_row
    run(bits, t_bit_s, samples_per_bit, v_lo, v_hi, *, n_warm_bits=None, n_sub=None) -> JunctionRun
        # JunctionRun(t_s, v_src, v_j, bits, t_bit_s, samples_per_bit, n_warm_bits, levels=(v_lo, v_hi));
        # simulate(..., junction=run) refuses a run of another pattern, bit rate or levels
    step_rise_time_s(v_lo, v_hi, t_bit_s=400e-12, samples_per_bit=4000, quantity="charge") -> {rise, fall, mean}

# mrm.waveform.eye
simulate(ring, drv, bits, bitrate_hz, *, v_bias, swing_vpp, level_mode=0, samples_per_bit=64, p_in_w=1e-3,
         n_warm_bits=None, junction=None) -> WaveformResult   # .mean(port), .mean_correction(port), .eye(port),
                                                              # .static[port]{p1,p0,mean}, .levels, .x_levels
eye_metrics(p, bits, samples_per_bit, t_bit_s=1.0) -> EyeMetrics(p1, p0, oma, er_db, eye_height, eye_width_s,
                                                                  eye_width_ui, phase_s, mean, polarity, rel_oma)
eo_response(ring, drv, v_bias, f_hz, *, port="through", p_in_w=1e-3, include_driver=False, loss_modulation=True)
eo_bandwidth(ring, drv, v_bias, *, port="through", include_driver=False, loss_modulation=True, f_max_hz=None,
             n=4001) -> {f3db_hz, peaking_db, f_hz, h_rel, x_bias, fwhm_hz, rc_pole_hz}
bandwidth_3db(f_hz, h) -> (f3db, peaking_db)

# mrm.waveform.lut
load_lut(name_or_path="nrz", *, verify=True) -> WaveformLUT      # sha256 vs sidecar, config hash
WaveformLUT.eval(x0, x1, p1, td, tr_ui, tau_ui, r_e=0.5, *, strict=True) -> dict   # raises LutDomainError
WaveformLUT.kernel_args() -> (ax_xc, ax_w, ax_p1, ax_td, ax_tr, ax_tau, ax_re, tab6, tab7)
WaveformLUT.domain, .config, .config_hash, .sha256, .validation
lut_eval_nb(ax_xc, ax_w, ax_p1, ax_td, ax_tr, ax_tau, ax_re, tab6, tab7, x0, x1, p1, td, tr_ui, tau_ui, r_e,
            out) -> status                                                                        [njit]
lut_eval_strict_nb(...same...) -> 0, raises ValueError outside the domain                          [njit]
lut_query_nb(phi0, phi1, k2_th, k2_dr, a0, a1, t_rt_s, bitrate_hz, rise_time_s, out5)              [njit]
   # out5 = (x0, x1, t_r/UI, tau_p/UI, r_e)
OUT_C_MEAN=0 OUT_C_OMA_THRU=1 OUT_D_OMA_THRU=2 OUT_C_OMA_DROP=3 OUT_D_OMA_DROP=4 OUT_D_MEAN_U=5 (N_OUT=6)
ST_XC=1 ST_W=2 ST_P1=4 ST_TD=8 ST_TR=16 ST_TAU=32 ST_RE=64 ST_NONFINITE=128
normalized_node(x0, x1, bits, tau_ui, tr_ui, re_grid=(0.5,), *, samples_per_bit=None, n_warm=None) -> dict
normalized_correction(x0, x1, bits, tau_ui, tr_ui, **kw) -> c_mean
build_lut(axes=None, *, n_bits=1024, n_warm=32, base_samples=32, seed=2026, progress=False) -> WaveformLUT
p2_6_prbs_check(tau_ui=0.15, *, order=9, xc=None, w=None, tr=None, limit=0.03) -> dict
interpolation_check(lut, n_random=200, seed=7, n_bits=1024, practical=False) -> dict
```

Rebuild the artefact with `NUMBA_NUM_THREADS=4 .venv/bin/python tools/memguard.py --limit-gb 3 -- .venv/bin/python
-m mrm.waveform.lut --name nrz`. It takes 213 s on 4 threads with the machine at load 15 and peaks at 0.22 GB.
`--quick` (smoke-test axes) writes `waveform_lut_quick` and refuses the name `nrz`.

## 3. For the engine owner: `fidelity.data_modulation: lut` (ENGINE_BRIEF gap 7)

Once per run (Python):
- `lut = load_lut("nrz")` (a schema key is requested in section 6). Record `lut.config_hash` and `lut.sha256` in the
  run manifest (07 P2-6: "record the config hash").
- Per world and ring, get the 10-90 % rise time of the junction CHARGE edge (the resonance shift follows the
  removed charge):

  ```python
  DriverJunction.from_plant_params(pp, b, n, rise_time_s=<driver edge>).step_rise_time_s(v_lo, v_hi)["mean"]
  ```

  Use the nominal NRZ levels. The driver edge has no schema key yet (section 6). In g26 with an 8 ps driver edge,
  the charge edge is 8.9 ps (R_drv 50 ohm, R_s 100 ohm, C_j 10 fF at 1 V).

Per substep, for each ring n and its own line l = `pp.layout.ring_to_line[n]` (other lines sit many linewidths away
and stay on the static mean):

```python
lut_query_nb(phi0[n,l], phi1[n,l], k2t[n,l], k2d[n,l], a0[n,l], a1[n,l], trt[n,l], bitrate_hz, t_r[n], q5)
st = lut_eval_nb(*lut.kernel_args(), q5[0], q5[1], rho[n], td[n], q5[2], q5[3], q5[4], out)
```

The inputs come from `ring_optics_row`, at the previous ring temperature like the optics inputs. `rho` is the ones
density and `td` is the scenario's `data.transition_density`.

Apply the outputs to the static NRZ-averaged numbers of that (ring, line):
- RL_DROP, RL_LOSS, RL_ABS and RL_UCW are multiplied by `out[OUT_C_MEAN]`, and so are their contributions to
  R_DROP_SUM and R_ABS_SUM.
- RL_THRU decreases by `(c_mean - 1) * (RL_DROP + RL_LOSS)`. This is energy conservation (exact in the TCMT).
- For the bus-output OMA, use the additive form `|POUT1 - POUT0| + out[OUT_D_OMA_THRU] * RL_PIN` where the static
  OMA is small (x_c near 0), and `x out[OUT_C_OMA_THRU]` elsewhere.
- The drop-level OMA (bit statistics) has the matching pair OUT_C_OMA_DROP and OUT_D_OMA_DROP. The latter is in
  units of the drop peak 4 kappa_e kappa_d P_in / kappa^2.

The self-heating P_abs inside the implicit solve should see the same factor. Either the optics adapter multiplies
the line's level-averaged absorbed power by c_mean, or the engine holds c_mean from the previous substep. The factor
moves slowly with T, so the lag is harmless. Doing this inside the walk would need an optics change (section 6).

A nonzero `st` means the query is outside the stored domain. `lut_eval_strict_nb` raises; `lut_eval_nb` clamps and
reports. The engine must pick a policy and count it in the diagnostics. Domain: |x_c| <= 6 HWHM, swing w <= 1 FWHM,
p1 in [0.3, 0.7], td in [0, 0.6] with td <= 2 min(p1, 1 - p1), t_r <= 0.7 UI, tau_p/UI in [0.07, 0.25],
kappa_e/kappa in [0.3, 0.7]. One lookup costs about 0.7 us (measured at load 15-19), so 8 rings cost about 6 us per
tick. Re-evaluate only when the inputs move (for example x by more than 0.01), or reuse the lag above.

## 4. Results

**07 P2-6 test** ("the correction to the mean drop power is below 3 % for PRBS at tau_p ~ 3 ps and a 20 ps UI").
I computed it directly with PRBS9 at tau_p/UI = 0.15, over x_c 0-6 and edges 0-0.7 UI. The claim holds only for
small swings:

| swing w (FWHM) | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| max \|c - 1\| | 0.1 % | 0.4 % | 1.0 % | 1.7 % | 2.6 % | 3.7 % | 6.4 % | 9.6 % | 13 % | 22 % | 30 % |

- At w = 0.3, the value is 2.4 % with ideal edges and 3.7 % with 0.7 UI edges.
- The maximum is at x_c = 0, where the resonance sweeps across the laser on every edge.
- The G25-like drive is 0.13-0.2 FWHM. In g26 at 1.3 Vpp, w = 0.19 and c = 0.9976.
- The 170-230 pm that 07 D9 says the printed relOMA needs (0.5-0.7 FWHM) is NOT in the < 3 % regime. There the LUT
  matters.
- Full grid: sidecar `validation.p2_6_prbs9_tau_0p15`.

**LUT accuracy.** The artefact is 3.53 MB of float32 compressed data with 201,960 nodes per r_e. These numbers include
pattern sampling, from random queries against direct runs (sidecar `validation`):

| region | c_mean error | through-OMA ratio error |
|---|---|---|
| practical domain (x_c 0.3-3, w <= 0.6, PRBS-like data, t_r <= 0.5 UI, tau_p/UI 0.1-0.2) | max 0.39 %, p95 0.1 % | max 1.6 %, p95 1.3 % |
| full domain | max 1.3 %, p95 0.2 % | p95 4.7 %, max 22 % |

- In the full domain the additive through-OMA error is 1.1 % (p95) and 8 % (max) of the largest static OMA.
- The worst full-domain cases are near x_c -> 0, at the edges of (p1, td) and at small tau.

Against full runs on g26 (Gaussian driver edge, nonlinear C_j(V), loss modulation, PRBS9, 50 Gb/s), the LUT applied
to the static average reproduces the time-domain mean drop power within 0.03-0.57 % and the through OMA within
0.2-2.3 %. The 2.3 % case is the red-side point with 12 ps edges, which is the cost of the canonical single-pole edge.

**g26 examples** (ring 0, nominal, 1 V bias, centered levels):
- x = 1/sqrt(3), 1.3 Vpp, 8 ps edges: relOMA -6.1 dB, ER 4.8 dB, eye width 0.98 UI. The relOMA agrees with 07 D9's
  -5.9 to -8.1 dB for a G25-like drive.
- x = 0.8, 2.6 Vpp: relOMA -3.05 dB, mean drop correction 0.9795.

**CAL-10 / VAL-3** (`tests/validation/test_eo_bandwidth.py`; g25_test_vehicle nominal, modes_rib160 present; R_drv
50 ohm in series; driver edge excluded):

| hypothesis | preset (R_s 100 ohm) | R_s fitted to CAL-10 | VAL-3a (35.4 +- 3) | VAL-3b (1.09 +- 0.06) |
|---|---|---|---|---|
| x = 1/sqrt(3), re-centred (primary) | 45.5 GHz (CAL-10 fail) | x3.74 = 374 ohm, RC pole 36.4 GHz | 34.65 pass | 1.121 pass |
| x = 0.813, re-centred | 59.4 GHz | x5.49 = 549 ohm (above the 390 ohm prior) | 36.44 pass | 1.179 fail |
| x = 1/sqrt(3), laser fixed at the -1 V wavelength | | | 45.78 fail | 1.481 fail |
| x = 0.813, laser fixed | | | 47.50 fail | 1.537 fail |

- I chose the primary hypotheses (max slope, re-centred laser) from 03 section 5.6 as the usual measurement setup.
  I did not tune them to VAL-3, but I saw all four rows before writing the asserts.
- VAL-3b excludes the fixed-laser reading: the bias moves the resonance by about 8.8 GHz/V.

**Performance** (shared M2 at load 15-19):

| operation | time |
|---|---|
| `simulate`, PRBS9, 64 samples/bit | about 0.17 s (5 us per sample; the RK4 electrical part with erfc edges dominates) |
| `simulate`, PRBS15, 32 samples/bit | about 8 s |
| `RingTCMT.from_plant_params` | 0.36 s |
| `eo_bandwidth` | about 1-5 ms |
| normalized LUT node (1024 bits) | 2.7 ms |
| `lut_eval_nb` | about 0.7 us (0.36 us measured in the review, load 15) |

## 5. Deviations from the contract and doubts

1. **P2-6's "< 3 %" holds only for swings up to 0.25 FWHM** (section 4). The test asserts the honest value.
2. **LUT keys**: P2-6 lists (detuning/HWHM, kappa_e/kappa, swing, mark and transition density, rise time). A
   normalized table also needs **tau_p/UI**, so I added it. kappa_e/kappa needs no axis for the means (exact
   algebra, lut.py docstring). It is an axis only for the through-port OMA.
3. **No eye heights in the LUT.** They are kinked in the phase and in the parameters. Use waveform mode for them.
4. **LUT model**: single-mode TCMT, no backscatter, no loss modulation (q), canonical single-pole edge. The residual
   against the full chain is quantified above. The waveform runs themselves include all of these (single order only;
   adjacent orders are at least one FSR away).
5. **Driver**: the Gaussian-edge NRZ source is an assumed form. There is no schema key for its rise time or for a
   pad capacitance (C_pad defaults to 0).
6. **EO bandwidth**: the replica here keeps the preset R_drv (50 ohm) in series with R_s. The stand-in
   `_eo_response` in `mrm/validation/replicas.py` uses `r_source_ohm = 0` by default. The CAL-10 fit and the VAL-3
   prediction must use the same assumption. Only R_total x C is identified.
7. The waveform run holds the ring temperature fixed. Thermal time scales are microseconds or longer, and runs are
   nanoseconds.
8. **numba cache**: `lut._node_nb` calls `eye._best_phase` and `tcmt_time._mode_step`. numba's on-disk cache does not
   notice edits in another file, so after editing those, delete `mrm/waveform/__pycache__`. The LUT artefact must be
   rebuilt if any of these kernels change. Its config records the integrator and OMA definitions.
   `test_normalized_model_is_the_physical_tcmt` compares the cached `_node_nb` with a fresh `tcmt_run_nb`, so a stale
   cache after a semantic change shows up there.
9. **tau_p/UI domain**: [0.07, 0.25] covers Q_L 2020-4200 at 50-56 Gb/s (the presets run 50 and 56 Gb/s); at 25 Gb/s
   only Q_L >= 4030 is inside (the builder's comment said "25-56 Gb/s"; corrected in lut.py).
10. **Lock order of `RingTCMT.from_plant_params(x_bias=...)`** (found in the review): the default is the order of the
   cold resonance, the convention of `mrm.control.models.resonance_detuning_m` (the oracle and controllers use it).
   In sampled worlds (`uncertainty=sample, variation=sample`, seeds=1) lambda_res spreads over 1298-1327 nm, so that
   order can need a 2-FSR heating or a cooling: ring temperatures of 562 K and 157 K came out of the solve. It now
   warns when the shift is outside -0.5..1.5 FSR and takes `order=` to pick another lock order. The EO bandwidth moved
   by only 0.8-2.6 % between the two orders in those worlds (kappa barely depends on T), but a population replica should
   choose the order explicitly. The same question applies to the controllers' convention (params/controllers).

## 5a. Adversarial review (second owner)

Re-derived and checked: the TCMT conventions and steady state against the plant (signs of Delta, g, the CW/CCW
eigenbasis, port formulas), the exponential step and its forcing correction, the linearization H = K F(iw) +
conj(K F(-iw)), the two-node network and its transfer function, the Gaussian-edge source window, the normalized
model (b = a kappa/(2 sqrt(kappa_e) s_in), the through-port algebra 1 + 4 r_e Im b + 4 r_e^2 u, the mirror symmetry),
the single-pole edge recursion, the LUT strides and corner weights, the analytic statics and the eval-side
mirror/relabel mapping, lut_query_nb against optics.tcmt_rates, the sidecar sha256 and the params sidecar
convention. dDelta/dV of the g26 ring is +8.76 GHz/V (red shift with reverse bias, laser blue: x grows with the
bit-1 level), as the 50 pm/V of 03 section 5.6 requires.

Fixed: the lock-order handling above; `t_ring_k` and `x_bias` together are refused (x_bias was silently ignored but
recorded in meta); a guarded first guess when the R1 quadratic has no real root; `eo_bandwidth` returns nan instead
of noise when the DC response vanishes (x = 0 without loss modulation; before, h_rel was rounding noise divided by
rounding noise); `simulate(junction=...)` checks that the run has the same pattern, bit rate and levels
(`JunctionRun.levels` added); `DriverJunction.run` refuses grids above `MAX_HALF_STEPS` (a full PRBS23 period would
have allocated tens of GB on the shared machine); `markov_bits` refuses td = 0 (a cyclic pattern without
transitions is constant); `python -m mrm.waveform.lut --quick` no longer overwrites the engine artefact; the
integrator docstring no longer claims the forcing correction is exact for a linear lambda(t).

Tests added (tests/verification/test_waveform.py):
- `test_simulate_end_to_end_against_dop853`: the whole chain against ONE joint electrical + optical ODE (two-node
  network, C_j(V), R_s(V), sqrt shift law, loss modulation, backscatter): V_j to 4e-8 V, P_thru to 6e-7 P_in, drop to
  2e-6. It is the only test that sees the half-step alignment of V_j and the optics.
- `test_cal10_inversion_of_03` (2 cases): the 03 section 5.6 inversion of 30.9 GHz to an RC pole of 34.8 / 25.1 GHz.
- `test_loss_modulation_on_resonance_and_vanishing_dc_gain`: on resonance, loss modulation alone gives
  f_3dB = FWHM/2 exactly; index modulation alone gives nan.
- `test_ring_from_plant_params_lock_order`, `test_simulate_input_guards`.
- `test_lut_query_mapping_mirror_and_relabel`: at a node all four equivalent engine queries (levels swapped, laser
  on the other side) return the stored run, and a direct run of the mirrored dynamics agrees.
- `test_lut_build_resolution_is_converged`: the build's samples per bit against 4x more at the hardest corners
  (c_mean to 2e-5; OMA to 3.7e-3-5.2e-3 at x_c = 0, w = 1, tau_p/UI = 0.07).

## 6. Requests to other owners

- **params_machinery / schema**:
  - `driver.rise_time_ps` (10-90 % of the driver output edge; assumed; for example 8, range 4-15).
  - `driver.pad_capacitance_ff` (default 0).
  - An artefact key for the LUT, for example `fidelity.data_modulation_lut: extracted/waveform_lut_nrz.npz`, so
    realize verifies the sha256 (the `<stem>.json` sidecar convention is already accepted).
  - Matching PlantParams fields (`datapath.driver_rise_time_s`, `datapath.pad_capacitance_f`).
- **calibration**: CAL-10 identifies R_total x C at 1 V through `mrm.waveform.eo_bandwidth`. Here, under the primary
  hypothesis, R_s = 374 ohm at C_j(0) = 15 fF, against the preset's assumed 100 ohm. That preset gives 45.5 GHz, far
  above 30.9 +- 2. Fit with the same series R_drv convention that the VAL-3 replica uses.
- **validation replicas**: replace the `_eo_response` stand-in of `eo_bandwidth_at_bias` with
  `RingTCMT.from_plant_params(pp, v_bias=bias, x_bias=x)` + `DriverJunction.from_plant_params(pp, rise_time_s=0)` +
  `eo_bandwidth`. It adds R_drv, loss modulation, backscatter and the plant's phase model, and reproduces the 03 table.
- **plant_optics** (optional): a per-(ring, line) mean-correction factor input to `bus_propagate_nb` and the
  self-heating adapter, so that `data_modulation: lut` enters the implicit solve without a lag.
- **integrator**: ENGINE_BRIEF gap 7 can now point here. The schema keys above are still needed.
- **params_content / controllers** (from the review): in sampled worlds the cold resonance spreads over about
  +-15 nm (1298-1327 nm for g26 ring 0, seeds=1), so "lock the cold-resonance order" (mrm.control.models, the oracle,
  and this module's default) can ask for a 2-FSR heating or for cooling far below T_ref. Please confirm the intended
  lambda_res spread and define one lock-order rule (for example the order reachable with the least heating from the
  ambient) for the oracle, the controllers' models and `RingTCMT.from_plant_params(order=...)`.
