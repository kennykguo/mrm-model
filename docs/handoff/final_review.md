# Handoff: final review (2026-10-16/17)

An adversarial review of the whole model in eight dimensions, an independent verifier for every finding, then one
fixer per dimension and this final check. Scope set by Kenny: finish the listed fixes and stop; no new features or
settings; every g26 default stays as it is; g26 stays the default preset and g26_calibrated stays optional; no
calibration refit, no solver extraction and no validation-report regeneration. What is still open is in
docs/KNOWN_ISSUES.md. Commits of the fixes: 735f742 (optics, thermal, sensing, engine, controllers), a62f429,
ad69b43, 83d5e5c and f3532af (params, in parts); the calibration stale marking, the experiments/docs fixes and this
final check are not committed.

## 1. Coverage

| dimension | findings confirmed | what the reviewer checked (summary) |
|---|---|---|
| physics_optics | 3 | exact add-drop ring and TCMT against closed forms (1e-15), energy balance, Q / FSR / FWHM by hand, sign conventions, junction and FCA, units, backscatter, phase-model algebra, self-heating bounds, engine wiring, waveform, laser comb |
| physics_thermal | 8 | FOH/ZOH coefficients against expm, kernel thermal pass, fold points and the analytic cubic, certified Newton step, no spurious oscillation, electrothermal droop, DAC and slew, interim passivity and reciprocity, crosstalk tau law, ROM loading |
| engine | 7 (+1 refuted) | 07 P0-6 event order, latency, auto-policy guards, light-commit invariant, kernel against selfheat and kernel_ref, thread and batch invariance, RNG streams, settle, supply limit, reachable order, oracle criterion, metrics, data heat, NaN handling, the Python path |
| physics_sensing | 7 | Arrhenius dark current, unit conversions, shot and AFE noise (dt invariance), flicker synthesis, ADC transfer with INL/DNL, faults, temperature sensor, heater readback, bit statistics, BER model, junction LUT conventions, kernel wiring |
| controllers | 11 | registry refuses the oracle, observation channels, held-command clipping, latency, bridge handshake, fxp rounding and overflow, FxpPi and fxp_mlp accumulator proofs, gym codec, nominal models, ENGINE.md section 5 against the scorecard |
| params | 13 (+2 refuted) | all 17 presets load with valid provenance; relations close to 1e-12; unit conversions and source numbers by hand; geometry variation against the Berkeley spreads; uncertainty/variation separation; the 07 4.8 set; provenance propagation; calibrated presets; derive diagnostics |
| validation | 9 | frozen split intact; all 44 target predictions recomputed bit for bit on both presets; fit record against the report; tornado worlds; A1 on g26_calibrated; contamination; Sobol and identifiability statistics; profile intervals; evaluate; calibration.md numbers |
| experiments_docs | 11 | experiment, manifest, CLI and golden tests; the nine quick acceptance runs and four reruns bit for bit; experiment hashes; compression gate; criteria semantics; scenario numbers; self reference; rerun mechanics; CLI against the docs; golden rules; ranking statistics |

## 2. Fixed (one line each; the tests are named in docs/CHANGELOG.md, 2026-10-16 entries)

physics_optics
- -2: the engine never applied the bus segment loss (bus_t2); Plant, quasi_static and mrm.control.models now use one law (ring 7 gets 0.59 % less light).
- -3: the next-order distance (BND-1 diagnostic) comes from the plant's phase model and TCMT linewidths, not an ideal comb (g26 2.79-3.01 FWHM, was 3.33-3.37).

physics_thermal
- -1: ROM die-level time scales scale a mixed-sign pole pair as a whole (no 21 % heater -> ring overshoot).
- -2: the FEM TEC-held crosstalk package is a weight-0 stack choice and the corner g26_tec_package; the open conflict is a strict xfail closure test.
- -3: the crosstalk tau law takes the heat-source-to-victim distance (centre distance less the radii): nearest neighbour 2.85 -> 1.52 ms.
- -4: heater_node_tau_s provenance is "assumed"; its note says it is only the pole of the excess (h-1) Z0.
- -5: new key ring.thermal.site_floor_frac, the far floor of the site law, separate from the ring floor.
- -6: the selfheat_tau_s note quotes the ROM step fractions (27 % at 100 ns, 54 % at 1 us).
- -7: the DEFAULT_SH_DT_MAX_K comment is true (one fixed 50 K bound); settle reports its largest step.
- -8: ARCHITECTURE 5.3 states the thermal-model rule (interim by default, ROM selectable).

engine
- -1: settle holds every scenario signal at its reset-tick value.
- -2: the reachable order hops until no ring needs negative heater power (g26 seeds 1000: 10 -> 0 rings without a code).
- -3: warm-start codes are solved in the data state of the reset tick (traffic_start_stop settles within 1 pm).
- -4: kernel blocks end on controller edges, so chunk_ticks changes no auto-policy result (without tick jitter).
- -5: each world stops its own settle, so a settled world does not depend on its batch.
- -7: ReferenceEngine handles fault.controller_reset; new parity cases drive every scenario input.
- -8: ENGINE.md section 10 and mrm/cli.py comments no longer list built parts as not built.

physics_sensing
- -1: the bit-statistics window takes the scenario ones density, so a PRBS window is exact at every edge.
- -2: the monitor ADC reference sits at the package base by default (ring.monitor.adc_ref_site).
- -3: a reference-tap config is refused instead of silently dropped.
- -4: background_fraction is documented as a fraction of the total bus power; g26 range [0, 3e-4].
- -5: a per-world monitor baseline stays per world (was put onto rings when B == N).
- -6: round_robin with a shared integrating ADC needs C x W <= P; the rings on one converter share its DNL and INL.
- -7: the g26 junction_model comment names the existing LUT and why analytic is kept.

controllers
- -1: ControllerSpec reads controller_interface.expose.
- -2: spec.nominal is a knowledge-filtered view (NominalView); the fingerprint no longer changes with the seeds.
- -3: the round_robin schedule is applied (kernel, kernel_ref, runner, env), with parity tests.
- -4: on_out_of_range=error works on the compiled path (ActionRangeError).
- -5: the scorecard's steady lock needs a stationary point (drift <= 0.1 pm/ms); es_sine goes 8/8 -> 0/8.
- -6: the Python path rebinds a fast-kernel controller after a reset (bit-identical to the compiled path).
- -7, -8, -11: feedforward_pi, oracle and RewardSpec docstrings state the measured behaviour.
- -9: FxpMlp derives its accumulator proof; a false proven=True raises.
- -10: the monitor ampere scale needs monitor or model knowledge.

params
- -1, -11: the sampled geometry moves the couplers through the coupling Jacobian; one local gap sigma; the stand-in Q_L and kappa_d/kappa draws are gone.
- -2: the FSR follows n_g at the mask radius (sd 11.7 GHz; was exactly 1.8 THz).
- -3: the film-200 world is the rib200 artefact with its own Q anchor (eta 0.833).
- -4: Z_heater->ring is taken at the P_FSR measurement line, the same on every line (5.704 K/mW).
- -5: g26 beta is the secant of the solver s(T) to the lock (1.2377e-3); the old tangent is an alternative.
- -7: the line wavelengths carry the channel-plan provenance.
- -8: depletion_exponent range [0.334, 0.666] builds the analytic junction table.
- -9: the loader checks world values and alternatives against the leaf range.
- -10: the bend radiation is exponential in the geometry (no negative loss).
- -12: heater_self_ratio 1.132 keeps y = 0.00904 /mW at the new Z; the notes quote the realized model.
- -13: r_cold_ohm range [37, 58]; the calibration's R0 prior reads the declared range.
- -14: the calibrated coupling_regime is a label of er_mechanism (the realized cold 0 V regime).
- Calibration not refitted (as asked): STALE header in both calibrated presets (comment only, hashes unchanged), a `stale` key in latest_fit.json, a note in calibration.md, the dispersion test xfail.

experiments_docs
- -1: new lock-independent metric order_changes; A4-hop and the soak golden ask order_changes == 0.
- -2: a convergence certificate records the code fingerprint and is refused for other code.
- -3: rerun takes a reused certificate from the run's own directory.
- -4: max_abs_detuning_pm convergence tolerance is per ring, capped at 0.5 pm.
- -5: corner contents are in the experiment hash; a corner file is copied into inputs/.
- -6: already fixed by engine-8 (checked).
- -7 to -11: doc corrections (start option, manifest layout, CLI usage, experiments.md loader text, the controller README's A1 numbers).

validation: not fixed (skipped as asked); all nine findings are in docs/KNOWN_ISSUES.md section 2.

## 3. Refuted by the verifier (one line each)

- engine-6 (the auto-policy oracle guard is off by one tick): the code reading is right, but the claimed effect (the
  oracle sees the state at oracle_next + h instead of oracle_next + 1) does not happen.
- params-6 (a path-keyed d lambda/dT variation merges an epistemic world variable into device variation): 07 P1-16
  asks for per-ring d lambda/dT variation, and the epistemic world draw stays separate; nothing is merged.
- params-15 (the thermal uncertainty set never reaches the FEM ROM's crosstalk dynamics): the reviewer's numbers are
  right, but the conclusion drawn from them is not. The related FEM crosstalk gap was confirmed separately as
  physics_thermal-2 (section 2).

## 4. Final check (this task)

TEST_STATUS_PLACEHOLDER
