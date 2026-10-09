# Architecture and contracts (v2)

This file is the contract every module is written against. v1 was reviewed by four independent critics; their merged
review is `docs/research/07_design_critique.md` ("07" below). **Sections of 07 cited here are normative**: implement
them as written there unless this file says otherwise. If code and this file disagree, fix one of them in the same
change and note it in `docs/CHANGELOG.md`.

Research behind every number: `docs/research/01..06` (01 Lightmatter public corpus and GF 45SPCLO platform,
02 materials (+ `02_materials.json`), 03 compact-model equations, 04 thermal data and control literature, 05 tools,
06 textbook audit). The capstone brief is `docs/research/00_capstone_brief.md`.

## 0. Scope of v1 (what is built now)

Normative from 07: **all of P0-1 … P0-11; P1-1 … P1-20; P1-22; P1-21 except the Verilator co-simulation (fxp
toolkit, golden vectors and the binary bridge are in scope; the HDL co-sim is a documented stretch); P2-1, P2-2,
P2-4, P2-5, P2-6, P2-7, P2-11, P2-12.** Optional if time allows: P2-3, P2-8, P2-9. Deferred with a stub and a
reason: P2-10 (generic S-matrix netlist; add it with the second photonic element).

Decisions D1-D22 of 07 section 0 are adopted as written.

## 1. Units, naming, provenance

- SI inside code (m, s, W, K, Hz, A, V, rad, J). Hz means cycles per second everywhere; angular rates are named
  `*_rad_per_s`.
- Config keys follow the suffix grammar of 07 P1-15: `<name>[_<qualifier>]_<unit>`, unit tokens from one table in
  `mrm/params/units.py` (`nm um mm m pm s ms us ns ps hz khz mhz ghz thz w mw uw nw a ma ua na v mv ohm f ff k c db dbm
  pct frac per rt ev`), joined with `per` and `rt`. `_c` is affine (Celsius), `_db`/`_dbm` are logarithmic. Unitless
  keys have no suffix. `mrm.params` converts to SI exactly once.
- Provenance words (closed set, ranked weakest-last): `printed > read-off > calibrated > inferred > solver > literature
  > assumed`, plus `derived` (inherits the weakest input), `override`, `private`. Generation tags (closed set): `G25 G26
  EVK100 L20 M1000 Guide OCP OCI GF PAT none`. Every preset leaf is a mapping
  `{value, provenance, generation, source, range?, alternatives?, note?, textbook?, disagrees?}`; `source` is a key in
  `mrm/params/sources.yaml`. Bare numbers are allowed only in overrides (auto-tagged `override`). See 07 P1-14.

## 2. Package layout and ownership

```
mrm/
  units.py                 physical constants, conversions (dB, dBm, wavelength<->frequency), unit-suffix parser
  materials.py             Si/SiO2 n(lambda,T) (Li 1980 shape on Frey/Franta level), dn/dT, alpha_L, k(T), rho*c,
                           plasma dispersion (Nedeljkovic 2011 @1.31um; Soref-Bennett switch), TCR by heater family,
                           Ge PD dark current law. Coefficients from docs/research/02_materials.json.
  params/
    units.py               suffix grammar and conversion table
    schema.py              every key: type, enum, unit, required/forbidden under each fidelity choice (07 P1-15)
    sources.yaml           source keys (mirror of the textbook's tags + literature + GF + patents)
    config.py              load(preset|path, overrides, alts) -> Config; extends:, alternatives, provenance checks
    relations.py           R1..R5 relation groups of 07 P0-3; closure / OverdeterminedError
    realize.py             realize(cfg, batch, seeds) -> PlantParams (07 P0-4): uncertainty worlds (B,), variation
                           (B,N)/(B,L), geometry Jacobian sampling (07 P1-16), relation solve, thermal MIMO assembly,
                           latency->ticks, timing assertions, self-heating lag bound
    derive.py              nominal B=1 view for docs; diagnostics (implied index weight, bistability ratio, ...)
    table.py               PARAMETERS.md generator, `mrm params show|table|explain|diff|validate`
    presets/*.yaml         g26 (default), g25_test_vehicle, corner presets of 07 4.8, teaching_* (self-heating off)
    presets/private/       git-ignored, provenance `private`
  core/
    channels.py            ChannelSpec, channel registry, flat int64 ABI layout generator (07 P0-7, D4)
    timing.py              integer-tick clock, latency FIFOs, event order at a controller edge (07 P0-6)
    rng.py                 counter-based RNG (splitmix64/Philox) usable in numba, stream-id enum (07 P1-13)
  extraction/              offline solvers -> mrm/data/extracted/ (never imported by runtime code)
    modes.py coupling.py thermal_fem.py junction.py fdtd_check.py (optional) run_all.py
  data/extracted/          committed artefacts + JSON sidecars (schema id, sha256, inputs, tool versions, errors)
  plant/
    optics.py              exact add-drop ring with CW/CCW backscatter + TCMT (complex sum over orders) (P2-2, P2-4)
    thermal.py             ROM load/assemble/scale; exact ZOH and FOH per Foster pole; interim 3-pole model (P1-2)
    selfheat.py            implicit per-ring self-heating solve with branch continuation (P0-1)
    electrical.py          heater driver laws (voltage|current|power|dac_pwm), DAC INL/DNL, slew, settle, TCR,
                           compliance; bias DAC; junction runtime (LUT); data-activity heat (P1-7, P1-8, P1-11)
    sensing.py             PD (responsivity, Arrhenius dark current, shot noise), AFE (rc | integrate_dump, exact
                           noise discretization), ADC (bits, offset, gain, INL/DNL, PGA, saturation), temp sensors,
                           heater readback, faults (P1-9, P1-10, P1-13)
    datapath.py            ones count, bit-statistics sampler, error counter (P0-11)
    laser.py               line table, static error, drift (random walk + linear), RIN, power drift, line enable
    plant.py               Plant: Python API over the kernel; Truth
  scenarios/               parametric generators evaluable inside numba + named library + registry (P1-4, P0-10)
  control/
    api.py                 ControllerSpec, Observation, Controller protocol, BaseController, PerRing/PerDie
                           adapters, telemetry, CompositeController (P1-1, P1-19, P1-20, P2-7)
    registry.py            name -> class
    models.py              linearize(nominal_cfg, op, Ts) -> (A,B,C,D,delay) for model-based controllers
    calibration.py         simulated factory calibration (warm/cold start)
    fxp.py lfsr.py         fixed-point toolkit (P1-21)
    baselines/             one file per controller family (section 6)
  sim/
    kernel.py              @njit run_block: plant substeps + sensor sampling + compiled controller + metric
                           accumulators (P0-8, P0-9); kernel_ref.py: the pure-NumPy reference (P1-18)
    runner.py              run(cfg|experiment) -> RunResult; Python-controller slow path; record specs
    metrics.py             streaming accumulators and summarize()
    experiment.py          run | sweep | mc | compare | robust | acceptance | quasi_static (P0-10, P1-17)
    manifest.py            runs/<UTC>_<hash8>/ resolved yaml, manifest.json, metrics.json, trace.h5
    gym_env.py             vectorized gymnasium env (P1-22)
    bridge.py              binary lockstep bridge for external controllers (P1-21)
    vectors.py             golden vector export (.npz/.hex)
  waveform/                short time-domain TCMT + driver + junction RC runs; normalized-coordinate LUT (P2-6)
  validation/
    targets/{calibration,validation,bounds,excluded}.yaml  (07 P0-5, section 4; frozen with a sha256)
    replicas.py oracle.py report.py
  calibration/             fits that may read ONLY targets/calibration.yaml
  cli.py                   `mrm ...` entry point
experiments/               experiment YAMLs (acceptance/, examples/)
examples/                  controller templates + tutorial notebook
tests/{verification,closure,calibration,validation,golden,perf}/
docs/ ARCHITECTURE.md PARAMETERS.md (generated) VALIDATION.md (generated) EXPLAINED.md CHANGELOG.md research/
tools/memguard.py          all solver runs go through it (16 GB machine)
```

## 3. Time scales

| Physics | Time scale | Treatment |
|---|---|---|
| Photon lifetime | ~3 ps | adiabatic: exact steady-state ring per substep (control mode); TCMT ODE in `waveform/` |
| NRZ bits (50 Gb/s) | 20 ps | mark-density average per substep; `data_modulation=lut` correction (P2-6) |
| Junction RC | ~4 ps | waveform mode only |
| Waveguide-core self-heating | 0.05-1 us | ROM poles, implicit solve (P0-1) |
| AFE / integrating ADC | 1-10 us | sensing state |
| Heater/ring thermal | 1-100 us | ROM poles, exact ZOH/FOH |
| Crosstalk, substrate, package | 0.1-10 ms | ROM slow poles |
| Ambient, TEC, laser drift | s-h | scenario signals; `time_compression` (P0-10) |

Plant tick `sim.dt_plant_s` (default 1 us; 0.1 us allowed). Time is an int64 tick count (P0-6).

## 4. Config tree

Top-level sections: `meta fidelity sim channel_plan laser optical_path ring{geometry optics junction heater thermal
monitor bias_dac} driver temp_sensors layout array uncertainty variation controller_interface controller scenario
metrics experiment datapath`. The authoritative key list is `mrm/params/schema.py`; `docs/PARAMETERS.md` is generated
from it and from the presets. Keys renamed or added by 07 (P0-1 … P1-16) are adopted; v1 keys that 07 deletes are
deleted (e.g. `ring.optics.absorbed_fraction`, `d2n_eff_dt2_per_k2`, `controller_interface.*_latency_samples`,
`array.pitch_um` (now `layout.sites`), `temp_sensor` (now `temp_sensors`)).

## 5. Artefact formats

All artefacts live in `mrm/data/extracted/` with a JSON sidecar `{schema, sha256 of the data file, inputs (geometry,
materials model, film), tool versions, mesh/resolution, validation numbers, created_utc}`. `realize()` verifies the
sha256.

### 5.1 Modes: `modes_<xsec>.json` (schema `modes/1`)

One file per cross-section hypothesis (`rib160`, `rib200`, `strip400x200`, ...), each solved straight and bent at
`r_ref_um`:

```
xsec: {width_nm, film_nm, slab_nm, sidewall_deg, clad}, r_ref_um, lambda0_nm, t_ref_c
n_eff_poly_lambda: [c0, c1, c2]     n_eff_bent(lambda) = c0 + c1*(lam-lam0) + c2*(lam-lam0)^2, lam in nm
n_g, dn_g_dlambda_per_nm
dneff_dt_per_k: {t_c: [...], value: [...]}        thermo-optic of the bent mode vs absolute temperature
alpha_l_per_k: {t_c: [...], value: [...]}         clamped expansion used
dlambda_dt_pm_per_k: {t_c: [...], value: [...]}   = (lam/n_g)*(dneff/dT + n_eff*alpha_L)  (H_solver)
index_weights: {si: , sio2: , ...}                dn_eff/dn_i;  check: sum n_g,i dn_eff/dn_i = n_g
power_fraction: {si: , ...}, mode_area_um2, alpha_rad_db_per_cm (bend radiation at r_ref)
jacobian: {d: [width_nm, film_nm, slab_nm], of: {lambda_res_nm_per_nm: [...], n_g_per_nm: [...], ...}}
crosscheck: {tidy3d_n_eff, femwell_n_eff, delta, same_r_ref: true}
```

### 5.2 Coupling: `coupling_<xsec>.json` (schema `coupling/1`)

Per coupler (`through`, `drop`) at its design gap: `kappa2_per_pass` at lambda0, `dln_kappa2_dlambda_per_nm`,
`dln_kappa2_dgap_per_nm`, `coupler_phase_rad`, and a table `kappa2(lambda_nm, gap_nm)`. Method: curved-gap supermode/CMT
integral along the bend (07 P1-5).

### 5.3 Thermal ROM: `thermal_rom_<family>_<film>.npz` (schema `thermal_rom/1`)

```
taus_s             (K,)        shared pole set (must span 10 ns .. 10 ms)
sources            [heater, waveguide_absorption, junction, driver, aggressor]
observers          [ring_mode_weighted, heater_resistor, monitor_pd, temp_sensor]
self_res           (S, O, K)   K/W, source of ring i -> observer of ring i (observer sites at their default offsets)
cross_r_um         (D,)        centre-to-centre distance grid (radially symmetric approximation, valid d > 2R)
cross_res          (S, O, D, K) K/W
site_offsets_um    {monitor_pd: [x, y], temp_sensor: [x, y], driver: [x, y]}  defaults used for self_res
```

Step response of any pair: `sum_k r_k (1 - exp(-t/tau_k))`. Mixed-sign residues allowed for non-collocated pairs. The
sidecar records the family, stack, mesh, the time-domain error (< 2 % of steady state, 10 ns .. 10 ms) and the
frequency-domain error (< 0.3 dB, < 3 deg, DC .. 1 MHz) against the full FEM, and the superposition error of the
8-ring bank (07 P1-3). Families: `inring_noundercut` (default), `silicide_undercut`, `metal_above`. Films: 160, 200.
Until the FEM artefacts exist, `thermal.py` builds the interim 3-pole model of 07 P1-2 from preset values.

### 5.4 Junction: `junction_<xsec>.json` (schema `junction/1`)

`v_reverse_v` grid from -0.5 to 4.0 V (forward negative), for a junction occupying the full circumference:
`dneff` (relative to 0 V), `alpha_fca_per_m` (absolute), `capacitance_f_per_m`, `r_series_ohm_m`, doping profile,
method (`devsim` or `analytic`), and the devsim-vs-analytic depletion-width check. Runtime scales by
`ring.junction.fill_fraction` (calibrated, CAL-5).

## 6. Controllers shipped in v1 (`mrm/control/baselines/`)

Each with a Python implementation; the ones marked (c) also as a compiled numba controller with the flat ABI:
1. drop-port set-point PI, fraction-of-peak normalized (c);
2. heater-dither lock-in (1f/2f) peak lock (c);
3. bias-dither lock-in (uses `tx.bias`), the fast-kernel form (P0-7 `fast_kernel`);
4. ramp-and-maximize acquisition (Lightmatter patent PAT-3) + lock hand-off;
5. extremum seeking (perturb and observe);
6. bit-statistics lock (Sun et al.) on `datapath.bitstat_above`;
7. temperature-sensor feed-forward + PI;
8. Kalman filter + LQG/MPC using `mrm.control.models.linearize` (model-based, honest nominal knowledge only);
9. dual-actuator mid-ranging (bias fast, heater slow);
10. WDM bank supervisor: sequential acquisition, wrong-line detection, barrel-shift remap (PAT-1), supply-limit
    sequencing;
11. the oracle (in `mrm/validation/oracle.py`, not registered).

## 7. Accuracy rules (replaces v1 section 6)

1. A number without a provenance word is a bug (enforced by the loader).
2. Measured Lightmatter data beats solver output, solver output beats literature, literature beats assumption. When a
   printed value is not a measurement, or violates a physical bound computed from materials data, the bound-respecting
   value is the default and the printed value is a required robustness alternative; docs/VALIDATION.md states the gap.
3. Dependent quantities are computed only by the relation solvers, after sampling (07 P0-3, P0-4).
4. Test classes and what blocks CI: 07 section 4.1.
5. Calibration reads only `targets/calibration.yaml`; held-out targets are frozen by sha256 (07 P0-5).
6. A controller claim is reported with the nominal result, the worst named corner and quantiles over the uncertainty
   set, and the first line of every controller report says performance is conditional on the assumed thermal dynamics.
7. Memory: 16 GB machine. Solver runs go through `tools/memguard.py` (`--heavy` for > 1 GB); Monte Carlo records
   metrics only.
