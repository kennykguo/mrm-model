# Handoff: experiments, run manifests and the CLI (`mrm/sim/experiment.py`, `manifest.py`, `cli.py`)

Task key: experiments. Implements 07 P1-17 (experiment files, run directories, `mrm.run`), the parts of P0-10 that
were not in the engine (`experiment.kind: quasi_static`, time compression with its load-time rule and the mandatory
convergence experiment, the soak as one experiment, `--dry-run`), P2-12 on the run side (`mrm run | sweep | mc |
compare | robust | acceptance | rerun`) and the "judging a controller" rules of 07 section 4.8. 07 =
docs/research/07_design_critique.md.

## 1. Files

New (owned):

| file | what |
|---|---|
| `mrm/sim/experiment.py` | the experiment layer: loading (with this module's extension schema), the seven kinds, the self reference, the compression gate, criteria, world summaries, Kendall-tau ranking stability, the dry-run estimate, writing run directories, `rerun` |
| `mrm/sim/manifest.py` | run directories `runs/<UTC>_<hash8>/`, git state and dirty-diff hash, environment, artefact hashes, JSON and HDF5 writers/readers, bit-for-bit `compare_runs` |
| `mrm/sim/cli.py` | `register_cli`: `mrm run`, `sweep`, `mc`, `compare`, `robust`, `acceptance`, `rerun` (mrm/cli.py already listed the provider) |
| `experiments/acceptance/*.yaml` | 10 acceptance experiments (section 7) |
| `experiments/examples/*.yaml` | 11 examples: steady_pi_lock, a1_pi_lock, mc_pi_lock, sweep_pi_gains, sweep_hypotheses, compare_locks, robust_pi_lock, quasi_static_g26, ones_density_steps, soak_window_hot, soak_reuse_certificate |
| `tests/verification/test_experiment.py` | 37 tests: loading, extension schema, criteria, summaries, ranking, planning, estimate, input copies, monitor windows, every shipped file loads (~13 s) |
| `tests/verification/test_experiment_runs.py` | 16 tests: every kind on the engine, rerun bit for bit (also of a compressed run, and from a run directory whose original parent file changed), scoring start (scored metrics equal the scored ticks of a per-tick trace; oracle reference and peak slew across the split), self reference, compression gate and certificate reuse, MC chunking, NUMBA_DISABLE_JIT=1, git state captured at the start (~30 s) |
| `tests/verification/test_manifest.py` | 10 tests: git, environment, artefacts (ROM included), JSON, HDF5 bits, compare_runs, run-dir names (~1 s) |
| `tests/verification/test_cli_run.py` | 8 tests: registration, `--dry-run` (with `--threads`), run + rerun, forced kinds, acceptance exit codes, errors, usage errors (~4 s) |
| this file | |

Changed (owned):

- `mrm/__init__.py`: `mrm.run(config, **kwargs)` (lazy import; `import mrm` stays light).
- `mrm/sim/runner.py` (extended, every existing signature and result unchanged):
  - new option `run(..., score_from_s=...)`: the metric accumulators restart at that time (`MetricsState.reset`
    at the split tick with the latched heater registers); plant, controller and records are untouched (under the
    fixed substep policy; under `auto` only for a split on a controller edge, docs/ENGINE.md section 2, correction
    2026-10-16). On the Python controller path a split that falls on a controller edge moves one tick later (the
    Python loop must not stop on an edge between two segments). `manifest_inputs["score_from_tick" |
    "score_from_s"]`. Two pieces of plant history survive the restart (review fixes): the lock reference of
    `lock_reference: oracle` (the reset would set it to 0, the line, until the next oracle evaluation up to 1 ms
    later) and the previous controller
    latch (tick, register) of the peak-slew accumulator (the reset dated it at the split, overstating
    `peak_slew_codes_per_s` up to 10x depending on where the split fell in the controller period). Tested: the
    scored metrics equal the same metrics over the scored ticks of a per-tick record.
  - `manifest_inputs["warm_start"]["no_code"]`: (B, N) rings that had no quasi-static lock code and started at
    code_min (new private `_warm_start_codes`; `warm_start_codes` keeps its signature and output).
  - `_run_python(..., bind=True)` (private): the second segment of a split run does not re-bind a compiled
    controller.
  - Bit-identity checked (section 8): all 6 regression and 11 adversarial scenarios of tools/perf_scenarios*.py
    have 0 differing signals against runs/perf_r2/final3 and final3_adv; test_kernel_parity, engine timing,
    metrics, physics and warm-start tests pass.

No corner preset was added: all 12 corners 07 4.8 names exist (`mrm.sim.experiment.NAMED_CORNERS`), and a new
corner would fail `tests/closure/test_presets.py::test_every_corner_has_a_pin_check` until its pin check is written
(closure tests are not mine). Other uncertainty pins (latency x2, low Q 2600, dark current 30 nA, ...) can be used as
corners without presets: `experiment.corners: [{name: slow_loop, alts: {uncertainty.timing.latency_scale: high}}]`.

## 2. Public API

```python
mrm.run(config, **kwargs)                                  # -> mrm.sim.experiment.run(config, **kwargs)
mrm.sim.experiment.load(source, overrides=None, alts=None) -> Experiment
mrm.sim.experiment.from_config(cfg, options: Mapping | None = None) -> Experiment
mrm.sim.experiment.run(experiment, *, overrides=None, alts=None, options: Mapping | None = None,
                       kind: str | None = None, runs_dir: str | Path | None = None, write: bool = True,
                       argv: Sequence[str] | None = None, log: Callable[[str], None] | None = None,
                       rerun_of: str | None = None) -> ExperimentResult
mrm.sim.experiment.rerun(run_dir, *, runs_dir=None, write=True, compare=True, log=None,
                         allow_changed_config=False) -> tuple[ExperimentResult, dict | None]
mrm.sim.experiment.estimate(experiment, *, overrides=None, alts=None, kind: str | None = None,
                            options: Mapping | None = None) -> dict
mrm.sim.experiment.world_summary(m: Mapping) -> dict[str, np.ndarray]          # (B,) per metric
mrm.sim.experiment.rank_stability(values: Mapping[str, np.ndarray], nominal: Mapping[str, float]) -> dict
mrm.sim.experiment.evaluate_criteria(criteria: Sequence[Criterion], cases: Mapping[str, Mapping]) -> list[dict]
mrm.sim.experiment.parse_records(spec, *, n_ticks: int, period_ticks: int) -> list[RecordSpec]
mrm.sim.experiment.default_records(n_ticks: int, period_ticks: int) -> list[RecordSpec]
mrm.sim.experiment.plant_fingerprint(cfg, start: str, worlds: Mapping | None = None) -> str
mrm.sim.experiment.load_certificate(path) -> mrm.scenarios.ConvergenceCertificate
# constants: KINDS, STARTS, NAMED_CORNERS, RANK_METRICS, RANKABLE, Q1_CAVEAT, PERF_REFERENCE, EXTENSION_KEYS
# classes: Experiment (cfg, options, source; .config(overrides, alts), .option(name), .hash, .kind, .seeds, ...),
#          ExperimentResult (kind, name, experiment_hash, cases, summary, report, passed, status, run_dir, wall_s,
#          manifest, extra; .metrics), CaseResult, Case, Criterion, OptionKey, ExperimentError

mrm.sim.manifest.git_state(root=None) -> dict          # commit, branch, dirty, diff_sha256, status
mrm.sim.manifest.environment() -> dict                 # python, numpy, scipy, numba, llvmlite, h5py, pyyaml, platform
mrm.sim.manifest.artefacts_used(cfg, pp=None, *, data_root=None) -> list[dict]
mrm.sim.manifest.file_sha256(path) -> str
mrm.sim.manifest.new_run_dir(runs_dir, hash8, *, now=None) -> Path
mrm.sim.manifest.runs_root(runs_dir=None) -> Path      # runs_dir, else $MRM_RUNS_DIR, else ./runs
mrm.sim.manifest.to_jsonable(x) -> Any                 # NaN -> null, +-inf -> "inf"/"-inf"
mrm.sim.manifest.write_json(path, obj) -> None
mrm.sim.manifest.write_trace_h5(path, groups, *, attrs=None) -> None
mrm.sim.manifest.read_trace_h5(path) -> dict
mrm.sim.manifest.read_run(run_dir) -> dict             # manifest, metrics, trace, resolved_yaml, dir
mrm.sim.manifest.compare_runs(a, b, *, ignore=()) -> dict   # identical, n_compared, differing, only_in_a/b, max_abs_diff
mrm.sim.cli.register_cli(register) -> None
mrm.sim.cli.format_estimate(est: dict) -> str
mrm.sim.runner.run(..., score_from_s=...)              # new runner option (section 1)
```

CLI: `mrm run EXP [--set PATH=VALUE ...] [--alt PATH=NAME ...] [--runs-dir DIR] [--threads N] [--no-write]
[--quiet] [--json] [--dry-run]`; `mrm sweep|mc|compare|robust EXP ...` force the kind; `mrm acceptance [EXP ...]
[--all] ...` runs each file's own kind and prints a pass/fail table; `mrm rerun RUN_DIR [--runs-dir DIR]
[--no-compare] [--allow-changed-config]`. Exit 0 = done and every criterion passed; 1 = a criterion failed, a
compressed run was refused, a rerun was not bit-identical, or an error (message, no traceback); 2 = usage.

## 3. Experiment files

An mrm.params config (`meta.kind: experiment`, `extends:` a preset; bare physical numbers are overrides) with the
07 P1-17 sections. Core keys are mrm.params schema keys (strict): `controller.{name, params}`, `scenario.{name |
stack, params, time_compression, ambient_c, ...}`, `metrics.*`, `experiment.{kind, duration_s, batch, n_mc,
seeds.{device, uncertainty, noise, scenario}, uncertainty, variation, grid, controllers}`, `sim.*`.

Extension keys (`mrm.sim.experiment.EXTENSION_KEYS`, this module's own strict schema; unknown names, wrong types and
enum values raise `ExperimentError` with a did-you-mean over both key sets):

| key | type | default | meaning |
|---|---|---|---|
| `experiment.start` | cold \| cold_settled \| warm | ResetSpec of the scenario, else cold | cold = heaters at code_min, `sim.reset_protocol` (g26 laser_ramp), cold lags (power-up); cold_settled = code_min, thermal state settled, branch_low; warm = quasi-static oracle lock codes, settled, branch_high |
| `experiment.record` | list | run: `default_records` (detuning mean + envelope, relOMA, oracle relOMA, heater and monitor codes, monitor current, ring T, line_on; <= 10 000 windows); others: none | `'field@dec:reducer'` (dec ticks or `auto`; reducer mean, min, max, last, envelope) or maps |
| `experiment.corners` | list | compare/robust: the `NAMED_CORNERS` (07 4.8's 12; g26_tec_package since 2026-10-16); else none | corner preset names or `{name, alts}` |
| `experiment.n_threads` | int | os.cpu_count() | kernel threads |
| `experiment.self_reference` | map | off | `{duration_s: 0.02, window_s: 0.005}` (positive, window_s <= duration_s): lock error about each ring's own steady operating point (section 5) |
| `experiment.score_from_s` | float | 0 | metrics accumulate from this time (runner `score_from_s`) |
| `experiment.acceptance` | list | none | criteria `{id, metric, op, value, reduce, rings, cases, source, note}` (section 5) |
| `experiment.convergence_certificate` | str | none | a run directory whose 07 P0-10 certificate covers this compressed scenario and plant |
| `experiment.compression_factors` | list | [1, 10, 100] | factors of the convergence experiment |
| `experiment.histogram` | bool | false | detuning histogram (~2.1 MB per world) |
| `experiment.oracle` | bool | true | in-kernel oracle |
| `experiment.p_in_w` | float | plant's | quasi_static: power per line at the chip (BND-4 is stated at 1 mW into the MRM) |
| `experiment.max_worlds_per_plant` | int | 512 | MC chunk size (~1.55 MB per world) |
| `experiment.rank_metrics` | list | `RANK_METRICS` | compare ranking metrics (names of `RANKABLE`: lower is better) |

How they reach the code: `load()` takes them out of the file (and out of every non-preset parent in the `extends:`
chain, child wins) before `mrm.params.load` validates the rest, from a stripped temp copy whose parents are absolute
paths (`Config.files` shows the original paths). So `mrm.params.load('exp.yaml')` of a file that uses extension keys
fails today; `mrm.run('exp.yaml')` / `mrm run exp.yaml` work, and `mrm.run(mrm.params.load('plain.yaml'))` works for
a file without them. When the integrator adds the keys to mrm/params/schema.py (request 10.1), `load()` stops
stripping them and reads them back from the Config (`_options_from_cfg`; tested with a monkeypatched schema). `--set
experiment.<ext>=...` overrides work in both states. `--set experiment.variation=off` is mapped back to the enum word
(YAML 1.1 reads off as False; mrm.params' own `--set` has the same trap).

Grid keys (`experiment.grid`, kind sweep): a config path (overrides, config units; `controller.params.<p>` and
`scenario.params.<p>` included), `alt:<path>` (a leaf alternative or an `uncertainty.<entry>` pin), `corner`,
`controller` (an entry). Kind quasi_static uses `grid.ambient_c` and `grid.bias_v`. Controller entries (compare,
sweep): a name or `{name, params, overrides, alts, start, label}` (bias controllers need their bindings as entry
overrides).

## 4. Kinds

| kind | cases | output |
|---|---|---|
| run | one runner.run of `batch` worlds (experiment.uncertainty / variation); a compressed scenario first passes the compression gate (section 6) | world summary, default records |
| sweep | one run per Cartesian grid point | per-point world summaries, every point's metrics in trace.h5 |
| mc | `n_mc` worlds (sample or lhs), metrics only; sample mode in chunks of `max_worlds_per_plant` with seed bases advanced (identical worlds, tested); lhs must fit one Plant | quantiles over worlds |
| compare | per controller: nominal (B = 1, nominal, variation off), every corner (B = 1, pins, variation off), `n_mc` (default 256) worlds (uncertainty default lhs, experiment.variation), metrics only | nominal result, worst named corner (overall and per metric), quantiles (p5..p95, max, nan), Kendall-tau stability per rank metric |
| robust | the same for the one `controller` | + criteria over the whole set |
| acceptance | main case (nominal / variation off unless set; `batch` worlds) + corners and worlds when given | criteria; RESULT PASS/FAIL |
| quasi_static | runner.quasi_static over `grid.ambient_c` x `grid.bias_v` (defaults 0..105 by 5, 1.0..3.75 by 0.25), plus one survey per corner | code-range coverage, points without a code, FSR-hop risk, wrong-line risk, heater-code range, BND-4 variation, oracle relOMA map |

Compressed scenarios are allowed in run and acceptance (main case only); elsewhere `ExperimentError` (a certificate
is per plant and controller).

## 5. Judging a controller (07 4.8)

- Report: the first line of every report with a controller is `Q1_CAVEAT` ("Performance is conditional on the
  assumed thermal dynamics ... until Q1 ... is answered"). Then nominal, worst named corner, quantiles, ranking.
- Primary metrics per world (`world_summary`): `oma_penalty_db_worst` / `_mean` (OMA penalty against the oracle at
  the same instant, worst lane / bank mean), `excursion_count` (sum), `excursion_max_s`, `time_unlocked_s`,
  `wrong_line_rings` (rings on a foreign line at the end) and `wrong_line_events`, `time_to_lock_s` (last ring; inf
  when a ring ends unlocked), `unlocked_rings`; reported: `mean_reloma_db` (absolute relOMA = OMA/P_in, 07 D9),
  `oracle_reloma_db`, `max_abs_lock_error_pm`, `fsr_hop_events`, `ber_mean_worst`, heater rails and range,
  `warm_start_no_code_rings`; monitor-error metrics when a self reference ran.
- Ranking: `RANK_METRICS` = (oma_penalty_db_worst, excursion_count, excursion_max_s, wrong_line_rings,
  time_to_lock_s), lower is better; NaN and inf rank worst. For each metric, Kendall's tau-b between every world's
  ranking (the LHS worlds and the corners) and the nominal ranking (scipy.stats.kendalltau); reported: mean, median,
  min, fraction of worlds with the same order, fraction defined (tau is undefined when a ranking is all ties, e.g.
  two controllers with 0 excursions), and the order of the per-controller medians.
- Lock reference. A side-of-fringe lock (pi_lock: 0.7 of the peak) sits 0.3-0.5 FWHM from its line and ~20 pm from
  the oracle optimum by design, so neither `lock_reference: line` nor `oracle` judges its loop dynamics (with
  `oracle` it never "locks", so it scores 0 excursions: a ranking trap). `experiment.self_reference` runs a steady
  pre-run of the same worlds, controller and start under the static scenario with the main scenario's signal
  defaults (ambient, bias, transition density) and takes each ring's median detuning over the last `window_s` as
  `metrics.lock_target_pm` (lock_reference target) of the main run, plus the median monitor current for
  `monitor_error_pct_min/_max` (signed % error of the drop monitor against it, BND-3a) and `monitor_dev_ua_max`
  (BND-3b). d_ref and the monitor reference are medians over the window means that lie inside the last `window_s`
  of the pre-run (about 10 windows; `CaseResult.reference["n_windows"]`). The monitor extremes come from a min / max
  record whose windows start at the scoring start and end at the end of the run (`_monitor_decimation`: the largest
  common divisor of both tick counts that keeps <= 10 000 windows), so every scored tick is judged and none before
  it. Refused for scenarios that drive the base temperature with a component (soak, cold_start, reloma_plane). The
  OMA penalty then says what the operating point costs.
- Start-up. pi_lock maps its peak through the resonance in the first ~2.8 ms (BER ~0.5 for ~0.3 ms, detuning down
  to 0 pm and back): `experiment.score_from_s` (0.005 in the shipped files) leaves that out of every metric.
- Criteria (`evaluate_criteria`): `reduce` over rings (max, min, sum, mean, all, any, none), then `op value` in
  every world of every selected case; NaN never holds; `band` / `band_either_sign` need `<metric>_min/_max`;
  metric names are keys of the metrics dict, dotted paths (`world.excursion_count`, `time_above_s.5pm`) or world
  summary names. A claim holds only if it holds in every world (07 4.8).

## 6. Run directories, rerun, dry run, time compression

- `runs/<UTC>_<hash8>/` (hash8 = first 8 hex of `Experiment.hash` = sha256 of the config hash and the resolved
  extension options; `_2`, `_3` on collisions): `experiment.resolved.yaml` (every leaf with provenance, generation,
  source, alternatives, artefact; the resolved options; `_source`: files, overrides, alts, options given),
  `manifest.json` (schema mrm.run/1: run id, kind, status, passed, experiment and config hashes and every case's
  config hash, created_utc, wall_s, rerun_of, git {commit, branch, dirty, diff_sha256 of `git diff HEAD --binary`
  plus every untracked file, status; captured when the run starts}, environment {python, numpy, scipy, numba, llvmlite, h5py, pyyaml, platform,
  machine, cpu_count, NUMBA_DISABLE_JIT}, argv (the parsed command line), seeds, source {path, overrides, alts,
  chain of files with sha256, inputs}, options, artefacts (sha256 now, recorded sha256, sidecar sha256, used_by,
  selected), cases {runner manifest_inputs incl. aggressor calibration, warm start, diagnostics, warnings, wall
  time}, convergence), `metrics.json` (summary + every case's metrics; NaN -> null), `trace.h5` (exact arrays:
  /metrics, /trace, /truth_final, /self_reference, /quasi_static/<case>/<array> with or without corners),
  `inputs/` (the experiment file and every non-preset file of its `extends:` chain as `_extends_<i>_<name>`, the
  copies' `extends:` naming each other: the directory reruns, and can be sent elsewhere, without the original files;
  snapshotted when the experiment is loaded, so an edit during a long run is not recorded), `report.txt`,
  `convergence.json` (compressed runs). The manifest's artefacts are the union over the experiment's config and every
  case config (corners and sweep points select their own alternatives).
- `mrm rerun runs/<id>` reloads `inputs/<file>.yaml` with the recorded overrides, alternatives and options, refuses
  when the experiment hash differs (presets, artefacts or the loader changed; `--allow-changed-config`), warns when
  the git commit, dirty diff, numpy / scipy / numba / llvmlite / python differ, runs into a new directory
  (`rerun_of`) and compares every array of trace.h5 bit for bit (NaN payload and the sign of zero included;
  run ids and wall times excluded); `rerun_comparison.json`. Tested for run (sampled worlds, B = 2), compare (2
  controllers, corners, LHS worlds), a compressed run with its convergence cases and an open-loop config run, and
  from the CLI; also on recorded acceptance runs (section 8).
- `--dry-run` (`estimate`): per case (its own config and controller: a sweep over the controller, its rate or the
  policy is costed per point; `--threads` is honoured) B, ticks, self-reference ticks, controller path, edges, start,
  record MB, RAM, wall time; totals. Basis (stated in the output): docs/handoff/perf.md 8.5: B = 1 fixed 9.3 us per tick; B >= 8 on
  8 threads 2.1 us per world-tick (wall per tick = max(B x 2.1, ceil(B / threads) x 9.3) us); auto policy x0.53
  (100 kHz controller; x0.70 measured here when the scenario has slow components, e.g. a soak) to x0.32 (10 kHz),
  x0.9 for B > 1; Python controllers +300 us per edge + 2 us per world;
  setup 1 s per case; quasi-static 0.012 s per point and world (measured here); RAM 0.24 GB + 1.55 MB per world
  (+2.1 MB with the histogram; measured here: B = 1 0.24, 64 0.33, 256 0.63, 512 1.03 GB peak RSS) + records; a cold
  numba cache adds 110 s and 2.3 GB. Measured against the estimate (compiled pi_lock_c, B = 1, warm cache, load 2-3):
  A1 3.6 s est / 2.1 s actual, A2 4.3 / 2.6, A3 6.6 / 5.0, A6 7.2 / 5.5, B1 2.9 / 0.9, B3 3.3 / 1.3 (conservative
  by 1-2 s of setup); B5 (wdm_supervisor, Python, B = 8) 4.1 s est (at 200 us per edge, before the update to 300)
  / 17 s actual: Python controllers vary ~10x (wdm_supervisor ~2 ms per edge at B = 8).
- Time compression (07 P0-10). `scenario.time_compression` is checked when the config's scenario is built
  (mrm.scenarios: compressed slow ramps <= 1 % of the 2127 K/s fast transient; the soak allows <= 173). A compressed
  run (kind run or acceptance) needs a certificate: `experiment.convergence_certificate` (a run directory), else the
  convergence experiment runs first: one slow cycle (`Scenario.slow_cycle_s`) at every `compression_factors` value on
  the same plant, controller, start, worlds and seeds, with the same scoring start; per world,
  `mrm.scenarios.assess_convergence` against factor 1 on CONVERGENCE_METRICS (max detuning, wrong-line and FSR-hop
  events, heater range used, integrated errors) with DEFAULT_CONVERGENCE_TOLERANCES; validated factor = the smallest
  over worlds. Then `require_compression_validated` (scenario fingerprint without compression, `plant_fingerprint` =
  resolved config without the `experiment` section and `scenario.time_compression`, plus the start and the worlds:
  batch, uncertainty, variation, seeds, scoring start); refused -> status `refused`, convergence results still
  written. Tested on a mini soak (factors 1, 2; reuse from another experiment of another duration; refusal for
  another plant and for other worlds) and on the full A4 soak (section 7).

## 7. Acceptance table

**Where the table comes from.** The textbook's acceptance table itself is not transcribed anywhere in docs/research
(searched 00_capstone_brief.md, 00_prior_notes_checklist.md, 06_textbook_audit.md, and every other file there and in
docs/handoff for "acceptance"); nothing outside the repository was read. What the repository has: the textbook's
scenario ids (Ch17 groups A and B, as mapped by `mrm.scenarios.library` / `registry` aliases a1-a4, a6, a7, b1-b3,
b5), 07 D14 ("the textbook's acceptance-table row [polarization scramble] should be corrected": RX-path monitors,
excluded as a TX stimulus, so there is no A5 file), 06 R2 (stabilized relOMA as an acceptance target), and the G26
printed results and frozen bounds that give pass criteria (LM26-P BER < 1e-9 over the soak and during the aggressor;
BND-2, BND-3a/b, BND-4 in mrm/validation/targets/bounds.yaml; the 5 pm lock budget of 04 section 4, an assumed
criterion). The rows below are therefore reconstructed from those ids; check them against the textbook's table.

Controller: `pi_lock_c` (drop-monitor set-point PI, the most likely G26 scheme, 06 C1), warm start, scored after its
peak map, lock judged about its own operating point; B5 uses `wdm_supervisor` from a cold start. Swap with `--set
controller.name=...`. g26 nominal world, seeds 0, runs in `runs/acceptance_20261010/` (git-ignored).

| file | scenario | criteria (source) | result |
|---|---|---|---|
| `a1_slide10_aggressor.yaml` | A1: slide-10 aggressor, rectified sine 113.6 - 88.5 \|sin\| degC at ring 0, one period from the minimum (0.1357 s) | lock <= 5 pm about the operating point (04 sec. 4, assumed); no wrong line (P1-12); no FSR hop (BND-2); mean BER <= 1e-9 (LM26-P); drop-monitor signed error inside [-1.9, 3.0] % +- 0.3, either sign (BND-3a) | **FAIL** (lock, BER, monitor): ring 0 leaves its point by 348.7 pm near the hot cusp (its heater at code 0: heater authority, as in the scorecard), mean BER 1.3e-8, monitor error -69.2 / +2.9 %; rings 1-7 hold; no wrong line or hop; OMA penalty worst lane 0.32 dB; relOMA -6.18 dB (oracle -6.04) |
| `a2_808_aggressor.yaml` | A2: 808 degC/s, symmetric triangle 25 -> 105.8 degC, 100 ms period, two periods (0.205 s) | as A1, monitor within +-0.5 uA (BND-3b) | **FAIL** (lock, monitor): ring 0 103.2 pm off and unlocked at the end of the hot ramp, monitor deviation 10.2 uA; BER 2.9e-85; no wrong line or hop; OMA penalty worst 1.16 dB |
| `a3_laser_sweep.yaml` | A3: laser-sweep emulation, 50 nm/s over 4 nm, 0.44 s trapezoid, from 10 ms (0.455 s) | lock, line, hop, BER | **FAIL** (lock): the heater lock lags the 50 nm/s ramps by up to 13.9 pm (8 excursions, longest 0.25 s); BER 6e-230; no wrong line or hop |
| `a4_soak.yaml` | A4: soak 33 <-> 108 degC, 90 x 20.3 min, compression 100 (1096.2 s simulated), auto policy; convergence experiment at 1, 10, 100 first | no BER window over 1e-9 (LM26-P Fig. 3(a)); no wrong-line event; no FSR hop, judged as max \|detuning\| < 5000 pm < FSR/2 (review change, below); heater never at a rail (BND-2, CAL-9); OMA penalty <= 1 dB (BND-4, VAL-4) | **PASS** (`runs/acceptance_20261010/20261010T115721Z_d1511028`, 4.8 h wall): convergence first, one 20.3 min cycle at c = 1 / 10 / 100 (7905 / 773 / 95 s wall): max detuning 126.4963 / 126.4925 / 126.4958 pm (allowed +-6.32 pm), heater range 26 029 / 26 029 / 26 022 codes (allowed +-260), 0 wrong-line and FSR-hop events, 0 integrated errors at every factor: validated up to 100. Main soak, 1096.2 s simulated in 8547 s: no 2 s BER window over 1e-9 (mean BER 4e-229), no wrong-line event, no hop, heater never at a rail (range 26 023 codes), worst-lane OMA penalty 0.12 dB, relOMA -6.16 dB (oracle -6.04) |
| `a6_connector_remate.yaml` | A6: ten re-mates, N(0.43, 0.35) dB per facet, every 50 ms from 10 ms (0.52 s) | lock, line, hop, BER | **FAIL** (lock): each power step moves the non-normalized set-point lock by up to 20.7 pm (24 excursions; every ring off its first point at the end); BER 1.5e-203; no wrong line or hop |
| `a7_reloma_plane.yaml` | A7: relOMA plane, quasi-static over 0..105 degC (12 points) x 1.0..3.75 V (12 points), + the 12 named corners reported | coverage = 1, no FSR-hop point (BND-2), BND-4 variation < 1 dB at every bias, judged on the nominal world | **PASS**: coverage 1, no hop point, BND-4 5e-9 dB (the lock temperature, not the ambient, sets the optics; only g26_heater_loss moves it: 0.064 dB; the FSR corners 0.06 dB). Corners without full coverage (they need an FSR hop somewhere on the plane): g26_hot_selfheat 0.67, g26_h86 0.83, g26_fsr_window_a 0.83, g26_textbook_thermal 0.83, g26_fsr_edge 0.92. Oracle relOMA (bank mean) -5.99 dB at 1.0 V to -9.23 dB at 3.75 V at every ambient: the 07 D9 / CAL-14 gap to the printed -2.1 to -3.0 dB (G25-like drive), reported, not judged |
| `b1_squelch.yaml` | B1: squelch (swing 0, average kept) 10-30 ms (0.06 s) | lock, line, hop (no BER while squelched) | **FAIL** (lock): the data heat and the bit-averaged monitor level change: 19.2 pm, 16 excursions (each <= 86 us); no wrong line or hop |
| `b2_laser_failover.yaml` | B2: line 0 +10 GHz and -1 dB at 10 ms (0.05 s) | every ring locked about its operating point at the end; line; hop; BER | **FAIL** (relock): ring 0 re-locks on its line but 51.6 pm from its former operating point (the -1 dB step moves a set-point lock); BER 6e-131; no wrong line or hop |
| `b3_xpu_power_step.yaml` | B3: XPU step, 120 degC/s within 30 ms (4.6 K die-wide) from 10 ms (0.1 s) | lock, line, hop, BER | **PASS**: max error 0.12 pm |
| `b5_cold_start.yaml` | B5: cold start, ambient uniform 0..105 degC per world (8 worlds: 12.1 to 83.2 degC drawn), heaters at 0, no calibration, power-up; `wdm_supervisor` | every ring within 300 pm (~0.9 FWHM) of its declared line at the end; none on a foreign line | **PASS**: all 64 rings on their lines after 25-27 ms (one world, 12.1 degC, barrel-shifted by 2 lines: a legal cyclic assignment, PAT-1); OMA penalty worst lane 1.6-2.4 dB (the supervisor's side-of-fringe point); FSR-hop counter 0-22 (acquisition sweeps across orders; not judged) |

**Review (2026-10-10, adversarial reviewer).** (1) A4's hop criterion was vacuous: the engine counts an FSR hop only
when a ring RE-LOCKS on another order, and with the line reference at 5 pm pi_lock_c never "locks" in A4 (it sits
~125 pm red of its line; `ever_locked` is False for every ring in every A4 case, convergence included), so
`fsr_hop_events` was 0 by construction. A4-hop now asks max |detuning| < 5000 pm (FSR/2 = 5094-5220 pm in g26: the
order nearest the declared line never changed). Evaluated on the recorded A4 metrics (no rerun of the 4.8 h soak):
126.5 pm, PASS; every other A4 criterion unchanged and PASS. The edit changes A4's experiment hash (383b0daa), not
its plant fingerprint, so the recorded certificate still covers it (checked). The same counter is part of the 07
P0-10 convergence comparison and was 0 = 0 there for the same reason (max detuning, also compared, would catch a
hop). (2) The nine quick acceptance experiments were re-run after the review fixes into
`runs/acceptance_review_20261010/` (same pass / fail as the table; values moved by <= 0.001 pm and <= 1e-4 % from
the exact self-reference window and monitor windows; reruns of those directories are bit-identical; the builder's
recorded A7 rerun is still bit-identical). The A1-A3, A6, B1-B3 hop criteria are meaningful only for rings that
re-lock; their max detuning (reported) is the lock-independent check.

A5 (polarization scramble) has no file: 07 D14 / excluded.yaml EXC-1, the 7.7 dB swing is on RX-path monitors, not a
TX stimulus. Re-run with `mrm acceptance --all --runs-dir runs/...` (A4 took 4.8 h: `--dry-run` first; to skip its
convergence step, pass `--set experiment.convergence_certificate=runs/acceptance_20261010/20261010T115721Z_d1511028`
as long as plant, controller and worlds are unchanged).

## 8. Tests and checks run

All through `tools/memguard.py` (default 3 GB cap), warm numba cache, no njit source changed (so no cache clear
was needed; the runner change is plain Python).

- New files: `tests/verification/test_experiment.py` 27 passed (~7 s), `test_experiment_runs.py` 12 passed (~25 s;
  includes a NUMBA_DISABLE_JIT=1 subprocess run compared bit for bit with the compiled run, and the rerun of a
  compressed run with its convergence cases), `test_manifest.py` 10 passed (~1 s), `test_cli_run.py` 7 passed
  (~4 s): 56 passed together in 36 s. `test_params_cli.py` (9) passes beside them.
- `mrm rerun` of three recorded acceptance runs after every later edit: A1 (compiled controller, self reference;
  89 arrays), B5 (Python wdm_supervisor, B = 8; 79 arrays), A7 (quasi-static with 12 corners; 289 arrays): all
  bit-for-bit identical. Certificate reuse: `experiments/examples/soak_reuse_certificate.yaml` with the A4 run
  directory is accepted (validated up to 100); a run of other worlds or another plant is refused.
- Engine bit-identity after the runner change: `tools/perf_scenarios.py runs/experiments_regression/ref` and
  `tools/perf_scenarios_adv.py runs/experiments_regression/adv`, compared with `tools/perf_compare.py` against
  `runs/perf_r2/final3` and `final3_adv`: 0 differing signals in all 6 + 11 scenarios. `test_kernel_parity.py`,
  `test_engine_timing.py`, `test_metrics.py`, `test_engine_physics.py`, `test_engine_warm_start.py`: 62 passed,
  1 xfailed.
- Fast suite (`-m "not slow"`, while the soak ran): 1608 passed, 74 skipped, 3 xfailed, 4 failed in 236 s. The 4
  failures are not mine: VAL-8a/b/c (`tests/validation/test_thermal_replication.py`, the known thermal-artefact
  failures) and `tests/golden/test_golden.py::test_every_golden_file_is_in_the_changelog` (golden files another
  agent is adding in parallel). `test_params_cli.py::test_format_doc_key_list_is_current` passes: this work adds no
  schema key.
- Acceptance experiments: section 7 (runs in `runs/acceptance_20261010/`).
- Full-size compare (`experiments/examples/compare_locks.yaml`: pi_lock_c, pi_lock_c with type 2, kalman_lqg_c,
  mpc_c under A1; nominal + 12 corners + 256 LHS worlds each, variation off; `--threads 6`):
  `runs/experiments_examples/20261010T115446Z_4b9c9b13`, 716 s wall (estimate 548 s at 8 threads; the soak ran
  beside it), peak RSS 1.20 GB. Results in section 9 item 7.
- `experiments/examples/robust_pi_lock.yaml` (12 corners + 128 LHS worlds under A1, `--threads 4`): 100 s wall
  (estimate 92 s), FAIL over the set (worst lock error 8627 pm, up to 8 rings on foreign lines; the warm-start
  problem of section 9 item 7), `runs/experiments_examples/20261010T122808Z_9750d84a`.
- `experiments/examples/soak_window_hot.yaml` (60 s of the soak at compression 1 from the 108 degC turnaround, warm
  start, auto policy): 407 s wall (estimate 295 s at the static auto factor), no excursion, no wrong line, worst-lane
  OMA penalty 0.85 dB, BER 3e-151, `runs/experiments_examples/20261010T120938Z_1c448ae3`.
- Every other example runs (exit 0, scratch run directories): steady_pi_lock 1.3 s, a1_pi_lock 2.3 s, mc_pi_lock
  (64 worlds) 7.4 s, sweep_hypotheses 7.5 s, quasi_static_g26 1.4 s, ones_density_steps 7.2 s, sweep_pi_gains 16 s.
  sweep_pi_gains (`runs/experiments_examples/20261010T170031Z_f350a251`): set-point fraction 0.6 / 0.7 / 0.8 gives a
  worst-lane OMA penalty of 0.79 / 0.32 / 0.24 dB under A1; the crossover (1 / 2.5 / 5 kHz) changes little; every
  point leaves ring 0's point by 326-371 pm at the hot cusp (heater authority, not loop gain). This run found a bug,
  fixed and tested: a sweep over `controller.params.<p>` (or `controller.name`) did not reach the controller (every
  point ran the base parameters); each case now rebuilds the controller from its own Config
  (`test_sweep_points_and_criteria` asserts the 0.6 / 0.8 lock points differ).
- Measured engine speed under the soak ramp (auto policy, 100 kHz pi_lock_c, B = 1, load 1.5-3): 6.5 us per tick
  at c = 1 and 10, 7.8 us at c = 100 (A4); the static auto figure of perf.md (4.9 us) is 25-35 % optimistic for a
  ramping base temperature. `estimate()` now uses 6.5 us (x0.70) for B = 1 when the scenario has slow components:
  the A4 estimate is 4.4 h for the 4.8 h it took.
- The A4 run started from the tree of 07:57 local; its manifest's git state was captured at write time (12:55),
  before the fix that captures it at the start of a run (applied afterwards, tested). The edits in between touched
  documentation, dry-run constants, the CLI's error handling and the examples, not results.

## 9. Deviations, doubts and findings

1. **Extension keys live in this module's schema, not in mrm/params/schema.py** (the task allowed either; changing
   schema.py is the integrator's). Consequence: `mrm.params.load('exp.yaml')` fails on a file that uses them, so
   07 P1-17's `mrm.run(mrm.params.load('exp.yaml'))` works only for files without them; `mrm.run('exp.yaml')` and
   `mrm run` always work. `load()` strips them into a temp copy (parents made absolute). Request 10.1 removes the
   difference; the code already reads schema-known keys back from the Config.
2. **The acceptance table is reconstructed** (section 7): the textbook's table is not in docs/research. Rows follow
   the textbook scenario ids the scenario library carries; criteria come from the printed G26 results, the frozen
   bounds and the 04 lock budget (an assumption). The acceptance controller is pi_lock_c (assumed G26-like).
3. **New lock reference 'self' (experiment.self_reference)**: not in 07. It costs a second, short run per case and is
   refused for scenarios that drive the base temperature. The pre-run uses the static scenario with the main
   scenario's signal defaults; a scenario whose t = 0 state differs from its defaults in another way would get a
   reference from a slightly different operating point. It overrides `metrics.lock_reference`. d_ref is the median
   of ~10 window means over the last `window_s` of the pre-run.
4. **New runner option `score_from_s`** (runner.py is mine to extend): exact (records, plant, controller untouched;
   tested bit for bit on the compiled and Python paths at the fixed policy; under `auto` only a split on a controller
   edge keeps the trajectory, ENGINE.md section 2, correction 2026-10-16). first_lock_s / time_to_lock_s become relative to the
   scoring start. On the Python path a split on a controller edge moves one tick.
5. **Convergence (07 P0-10)**: implemented with mrm.scenarios' plan, metrics and tolerances (design choices of the
   scenarios owner: 0.2 pm or 5 % on max detuning, exact event counts, 16 codes or 1 % on heater range, 1 error or
   10 % on integrated errors). Doubts: (a) integrated errors are compared raw, but a run at compression c spends c
   times fewer simulated seconds per slow cycle, so a steady error rate gives c times fewer errors; with BER at lock
   ~1e-200 both are ~0 and this does not bite, but a soak with a non-zero steady BER would fail the check for that
   reason alone; the scenarios owner should decide whether to compare rates. (b) Per world, the validated factor is
   the smallest over worlds. (c) The certificate's fingerprint covers plant, controller, scenario (without
   compression), start and the worlds (batch, uncertainty, variation, seeds, scoring start), not the duration.
6. **ResetSpec.factory_cal is not acted on**: `start: warm` = the integrator's settled warm start at the quasi-static
   codes, not a simulated factory table (mrm.control.calibration.factory_cal needs a CalBench; `runner.run(...,
   factory_table=)` exists but no experiment key builds a table yet).
7. **Findings from the full compare and probes (please read before trusting LHS quantiles).**
   - g26 with `uncertainty.thermal.stack = stacked_eic` alone (every other entry nominal): the settled warm start at
     the quasi-static codes sits 800 to 1100 pm BLUE of the lines (side_by_side: +97 to +121 pm red, as designed);
     7 of 8 rings then end on foreign lines under pi_lock_c in a static run (sweep probe; also the eic_probe:
     codes 41482-42582 vs 34280-35090). The quasi-static solve (static thermal gains) and the kernel's settled
     steady state disagree by ~12-16 K of ring temperature in that world (diagnosis in request 10.4). With FSR
     window A alone the lock holds but the worst-lane OMA penalty is 4.3 dB (static).
   - 32 LHS worlds, static 30 ms, pi_lock_c, variation off: 18 of 32 worlds have a ring off its point by > 5 pm or on
     a foreign line: all 8 stacked_eic worlds, 9 of the 10 FSR-window-A worlds, 3 worlds with neither. The same LHS
     design with device variation on: 13 of 32 bad, but only 2 of the 8 stacked_eic worlds (device variation changes
     the outcome in those worlds; not investigated); 0.9 rings per world (median 1) have no quasi-static lock code and
     start at code_min (the warm-start limitation of ENGINE.md section 3); 20 of the 30 wrong-line rings had a code.
   - Full compare under A1 (256 LHS worlds, variation off): median world 2 (pi_lock_c) / 5-6 (kalman_lqg_c, mpc_c)
     rings on foreign lines and 3 / 6 rings unlocked at the end; worst-lane OMA penalty median 6.1 / 27 dB. On the
     nominal world the model-based controllers rank first on OMA penalty (0.21 vs 0.32 dB) and the PI locks first
     on excursions; across worlds the OMA-penalty ranking is unstable (Kendall tau mean -0.13 against the nominal
     order, same order in 6 % of worlds), excursion count 0.29, longest excursion 0.33, time to lock 0.69 (defined
     in 35 % of worlds), wrong-line rings undefined (ties at 0 on the nominal world). Worst named corner for all
     four: g26_hot_selfheat (7-8 rings on foreign lines). These numbers mix controller robustness with the
     warm-start problem above; the engine owner should resolve that first (request 10.4) before any controller
     claim is made over the set.
8. **A7 / BND-4**: the oracle relOMA is independent of ambient in the nominal world (5e-9 dB) because the lock fixes
   the ring temperature; only a heater-temperature loss (g26_heater_loss, 0.064 dB) or the FSR corners (0.06 dB)
   move it. The absolute relOMA (-6.0 dB at 1 V to -9.2 dB at 3.75 V) is 4-6 dB below the printed -2.1 to -3.0 dB:
   the known 07 D9 gap (G25-like drive; CAL-14 not fitted yet).
9. **BER criteria** use the mean BER over the run (expected errors / bits) when the run is shorter than the 2 s
   G26 BER window; A4 uses the 2 s windows.
10. **B5's FSR-hop counter is not judged**: acquisition sweeps each ring across orders of its line (0-22 counted).
11. **Dry-run accuracy**: conservative by 1-2 s per case for short compiled runs; Python controllers vary ~10x per
    edge (wdm_supervisor ~2 ms at B = 8), which the estimate cannot know.
12. **`mrm acceptance --all` includes the soak** (~4.8 h with its convergence experiment); use `--dry-run` first.
13. No corner preset added (section 1).

## 10. Requests

1. **Integrator / params (schema)**: add the extension keys to `mrm/params/schema.py` so `mrm.params.load` accepts
   complete experiment files, then regenerate the PARAMETERS_FORMAT.md key appendix (`mrm params keys`). Proposed:
   ```python
   S("experiment.start", "enum", "start state (mrm.sim.experiment)", enum=("cold", "cold_settled", "warm")),
   S("experiment.n_threads", "int", "kernel threads"),
   S("experiment.score_from_s", "float", "metrics accumulate from this time"),
   S("experiment.convergence_certificate", "str", "run directory with a 07 P0-10 certificate"),
   S("experiment.compression_factors", "list_float", "convergence factors", default=(1.0, 10.0, 100.0)),
   S("experiment.histogram", "bool", "detuning histogram", default=False),
   S("experiment.oracle", "bool", "in-kernel oracle", default=True),
   S("experiment.p_in_w", "float", "quasi_static power per line at the chip"),
   S("experiment.max_worlds_per_plant", "int", "Monte Carlo chunk", default=512),
   S("experiment.rank_metrics", "list_str", "compare rank metrics"),
   # open subtrees validated by mrm.sim.experiment (add to OPEN_PREFIXES, type map):
   #   experiment.record, experiment.corners, experiment.self_reference, experiment.acceptance
   ```
   Defaults must match `EXTENSION_KEYS` (the experiment hash uses the resolved values). After that, `load()`
   leaves them in the file (tested with a monkeypatched schema).
2. **Integrator (pyproject)**: h5py into `[project] dependencies` (07 P1-17: "h5py becomes a core dependency"); it is
   installed in .venv.
3. **Integrator (docs)**: CHANGELOG entry (new modules and CLI commands; runner option `score_from_s`; runner
   `manifest_inputs["warm_start"]["no_code"]`; acceptance and example files; the extension-key arrangement of
   item 1); docs/ENGINE.md section 10 ("Not built yet: experiments ...") and section 2 (the new runner option);
   ARCHITECTURE section 2 (`sim/cli.py`).
4. **Engine owner (warm start)**: (a) in the stacked_eic world the settled state at the quasi-static codes is
   800-1100 pm blue of the lines (section 9 item 7). Diagnosis (nominal world, 25 degC, 1.0 V): at the same codes
   quasi_static expects 19.4-20.5 mW of heater power and the engine delivers 18.0-19.0 mW (7 % less; side_by_side:
   equal to 0.001 mW), so the ring ends 12-16 K colder than the quasi-static 133.8 degC, on the COLD self-heating
   branch (P_abs 0.14-0.79 mW against 1.52 mW at the lock; heater resistor 128-131 degC). Probably the static
   heater-resistor temperature (TCR) or the heater gains quasi_static uses for the stacked EIC; the probe is
   `params.load("g26", alts={"uncertainty.thermal.stack": "stacked_eic"})` + `runner.run(cfg, None, "static",
   duration_s=2e-3, uncertainty="nominal", variation="off", heater_code="quasi_static", settle=True)`.
   (b) `warm_start_codes`: when the quasi-static heater power is negative (fsr_hop_risk) or the code is above
   code_max, try the neighbouring order instead of starting the ring at code_min (0.9 rings per sampled world with
   variation). Until both are fixed, LHS quantiles of lock-only controllers mostly measure the warm start.
5. **Controllers owners**: acceptance results of pi_lock_c (section 7): heater authority at the A1/A2 hot end (ring
   0, as in the scorecard), a 14 pm lag under the 50 nm/s laser sweep, set-point moves of 19-52 pm under power
   steps (re-mates, failover) and squelch (a normalized set-point would remove the power-step part); the model-based
   controllers' collapse across LHS worlds under A1 (section 9 item 7) once the warm start is fixed.
6. **Scenarios owner**: the integrated-errors comparison under compression (section 9 item 5a); `Scenario.reset`
   (ResetSpec) is honoured for the start state only, `factory_cal` is not.
7. **Validation / textbook owner**: check the reconstructed acceptance rows and criteria (section 7) against the
   textbook's actual table; the BND-3a/b replicas could reuse the monitor-error metrics of the self reference.
8. **Params**: `mrm params ... --set experiment.variation=off` reads off as False (YAML 1.1); mrm.sim maps it back
   for enum keys (`_enum_word`), mrm.params' own overrides do not.
9. **Content (corners)**: more 4.8 corners as presets (loop latency x2, low Q 2600, dark current 30 nA, current /
   power / dac_pwm driver law, film 200 nm, metal_above, stacked_eic) need pin checks in
   tests/closure/test_presets.py; experiments can already use them as `{name, alts}` corners.
10. **Engine owner (metrics)**: `MetricsState.reset` is a full reset. For the scoring start the runner now keeps
    two pieces of plant history across it by hand (the 'oracle' lock reference and the previous latch tick and
    register of the slew accumulator, section 1). A `MetricsState.restart(k0, reg)` in mrm/sim/metrics.py that keeps
    them would be the cleaner home; the runner's workaround is tested against a per-tick trace.
11. **Scenarios owner (convergence)**: `fsr_hop_events` in CONVERGENCE_METRICS counts re-locks on another order, so
    it depends on the lock definition: with `lock_reference: line` at 5 pm a side-of-fringe lock never locks and the
    counter is 0 at every factor (A4: 0 = 0 by construction). A lock-independent order count (the order nearest the
    declared line changing, i.e. |detuning| crossing FSR/2) would make that comparison meaningful.
12. **Validation / textbook owner (criteria)**: B2-relock and A6-lock measure the operating-point shift against the
    pre-run reference (5 pm), not re-acquisition: a side-of-fringe set-point lock legitimately moves its point with a
    power step (A6 re-mates, B2 -1 dB), so they fail on a set-point property rather than on losing the line. Decide
    whether the table means "back on its line and stable" (wrong_line + excursion metrics) or "same operating point"
    (current criteria).

## 11. Adversarial review (2026-10-10)

Reviewed against 07 P0-10, P1-17, P2-12 and 4.8. Fixed in place (each with a test unless stated):

| # | severity | file | finding | fix |
|---|---|---|---|---|
| 1 | major | runner.py | `score_from_s` with `lock_reference: oracle`: `MetricsState.reset` set the reference to 0 (the line) until the next oracle evaluation (<= 1000 ticks), so the scored lock error was the distance to the line (pi_lock_c: ~125 pm, larger than the unsplit run's 109 pm) and spurious excursions were counted | keep the oracle's current reference across the split |
| 2 | major | runner.py | `score_from_s`: the previous-latch tick of the slew accumulator was reset to the split, so `peak_slew_codes_per_s` of the scored window depended on where the split fell in the controller period (pi_lock_c under A1: 6e5 to 6e6 codes/s; true 6e5) | keep the previous latch (tick, register) across the split |
| 3 | major | a4_soak.yaml | A4-hop (`fsr_hop_events == 0`) could not fail: the counter needs re-locks and no ring ever locks with the line reference at 5 pm | judged as max \|detuning\| < 5000 pm (< FSR/2); re-evaluated on the recorded metrics: PASS (section 7) |
| 4 | major | experiment.py | `inputs/` held only the leaf file; non-preset parents were referenced by absolute path (a run directory was not self-contained or exchangeable), and the copy was taken at write time (an edit during a 4.8 h run would be recorded as what ran) | copy the whole non-preset chain into `inputs/` at load time |
| 5 | minor | experiment.py | self reference took the windows with END time >= t_end - window_s: one window more than the last window_s | windows that START inside the last window_s; error if none |
| 6 | minor | experiment.py | monitor min / max windows of `sf_ticks` left the last `n mod sf` ticks unjudged (A1: last 0.7 ms) | windows of gcd(n, sf) ticks (`_monitor_decimation`) |
| 7 | minor | experiment.py | `self_reference` values not validated (a 2 ms pre-run with the default 5 ms window silently used the whole pre-run, start-up included) | positive numbers, window_s <= duration_s |
| 8 | minor | experiment.py | `reduce: none` ignored `rings` | rings applied first |
| 9 | minor | experiment.py | an unknown criterion metric on a non-engine metrics dict (quasi-static) raised KeyError (crash) instead of failing the criterion | caught |
| 10 | minor | experiment.py | a bad sweep value, corner or controller entry failed only after the earlier cases ran (possibly hours) | every case's config, scenario and controller built before the first run |
| 11 | minor | experiment.py | `rank_metrics` not validated (a typo crashed after the runs; relOMA, higher is better, would rank inverted) | must be names of `RANKABLE` |
| 12 | minor | experiment.py | open-loop acceptance reports started with the Q1 controller caveat | caveat only when a case has a controller |
| 13 | minor | experiment.py | quasi_static trace.h5 layout differed with and without corners | always /quasi_static/<case>/<array> |
| 14 | minor | experiment.py | manifest artefacts only from the main config | union over every case config |
| 15 | minor | experiment.py, cli.py | `--dry-run` ignored `--threads` and costed every sweep point with the base controller | `estimate(..., options=)`; per-case config and controller |
| 16 | minor | cli.py | a malformed `--alt` exited 1 via SystemExit (documented: 2); `mrm rerun` leaked tracebacks for KeyError / TypeError | exit 2; caught |
| 17 | minor | experiment.py | `world_summary` monitor reductions could warn on all-NaN rows | guarded (no test) |

Checked and found correct: the extension-key strip / read-back, rerun's hash and environment checks, the
compression gate and certificate fingerprint (no file paths in it: the A4 certificate covers
soak_reuse_certificate.yaml and a rerun directory), MC chunk seeding, Kendall tau-b handling of NaN / inf / ties,
criteria NaN semantics, h5 bit round trip, the Python-path split shift.

Not changed (noted): `Experiment.hash` includes `n_threads` (results do not depend on it, so `--threads` changes
hash8; changing the hash now would invalidate every recorded run's rerun check). The convergence check compares
integrated errors raw (request 6). Wrong-line state and the FSR-hop lock order restart at the scoring start by design
(the start-up is not scored).

Tests and checks after the fixes (all through tools/memguard.py, 3 GB cap, warm numba cache, no njit source changed):
test_experiment.py 37, test_experiment_runs.py 16, test_manifest.py 10, test_cli_run.py 8, test_params_cli.py 9:
all pass. tools/perf_scenarios.py and perf_scenarios_adv.py against runs/perf_r2/final3 and final3_adv: 0 of 6 + 11
scenarios differ (runs/experiments_review/). test_kernel_parity, engine timing, metrics, physics, warm start: 62
passed, 1 xfailed.
Fast suite (`-m "not slow"`): 1683 passed, 74 skipped, 3 xfailed, 4 failed in 270 s, peak RSS 1.0 GB; the 4 failures
are the pre-existing ones outside these files (VAL-8a/b/c thermal replication; tests/golden changelog check of
another agent's golden files).

## 12. Integration (2026-10-10, integrator; docs/handoff/integrator.md section 8)

### 12.1 Requests of section 10

- 10.1 **done**: the extension keys are mrm/params/schema.py keys (exactly the proposed S(...) list; the four open
  subtrees in OPEN_PREFIXES); `load()` no longer strips them (`_is_extension` is False for all), `_options_from_cfg`
  reads them back. Two changes on this side: the experiment hash is `config_hash_without_options(cfg)` + options, so
  every recorded experiment keeps its hash (checked for all ten acceptance files) and recorded run directories still
  rerun; a dotted override of a map option (`--set experiment.self_reference.window_s=...`) now merges into the
  file's map instead of being ignored (it is stored flattened by mrm.params). Extension overrides are still checked by
  this module's `_check_option` (same messages). `mrm.params.load(exp)` and `mrm.run(mrm.params.load(exp))` work for
  every shipped file. PARAMETERS_FORMAT.md appendix regenerated.
- 10.2 **done** (pyproject.toml). 10.3 **done** (CHANGELOG; ENGINE.md sections 2, 3, 6, 10 and the new section 11;
  ARCHITECTURE section 2).
- 10.4a **diagnosed and fixed in quasi_static**: the stacked_eic disagreement is the array heater supply limit (07
  P1-12), which the plant applies and quasi_static did not check (integrator.md 8.2 item 2). quasi_static reports
  `supply_scale` / `supply_limited`; a supply-limited point counts no ring as reachable. 10.4b **done as an opt-in**:
  `heater_code="quasi_static_reachable"`, used by `start: warm` (`_start_kwargs`); the runner default is unchanged
  (the regression scenario mc8_pi has 10 rings without a code).
- 10.8 **done** in mrm.params (`parse_overrides` maps a YAML boolean to the enum word); `_enum_word` here is kept.
- 10.10 **done**: `MetricsState.restart(k0, reg)`; the runner calls it.
- Open: 10.5, 10.6, 10.7, 10.9, 10.11, 10.12 (owners named in section 10).

New fields reach the reports: world summaries `warm_start_order_hopped_rings` and `warm_start_supply_limited`
(and the metrics `warm_start_order_hopped` (B, N), `warm_start_supply_limited` (B,)); the quasi_static kind reports
supply-limited points with the smallest droop factor (`supply_limited_points`, `supply_scale_min`; trace.h5
`/quasi_static/<case>/supply_scale|supply_limited`).

### 12.2 Every acceptance and example experiment, end to end (integrated tree, `mrm acceptance` / `mrm run`)

Machine: the shared 8-core M2, two soak runs in the background (load 3-13). Acceptance runs in
`runs/acceptance_integrator_20261010/`, examples in `runs/examples_integrator_20261010/` (git-ignored).

| file | result | values (worst over rings) | wall |
|---|---|---|---|
| a1_slide10_aggressor | FAIL (A1-lock, A1-ber, A1-monitor) | lock 348.736 pm (ring 0 at the hot cusp, heater at code 0), mean BER 1.29e-8, monitor error -69.17 / +2.86 %; no wrong line, no hop | 2.4 s |
| a2_808_aggressor | FAIL (A2-lock, A2-monitor) | lock 103.184 pm, monitor deviation 10.194 uA; BER 2.94e-85 | 2.6 s |
| a3_laser_sweep | FAIL (A3-lock) | lock 13.932 pm (lag under 50 nm/s); BER 5.98e-230 | 5.2 s |
| a6_connector_remate | FAIL (A6-lock) | lock 20.716 pm (set-point moves with each power step); BER 1.48e-203 | 5.8 s |
| a7_reloma_plane | PASS | coverage 1, no hop point, no supply-limited point, BND-4 5.39e-9 dB; corners: g26_low_eta coverage 1 -> 0.917 and g26_fsr_edge 0.917 -> 0.833 (new: supply-limited at 0 degC, droop 0.985 / 0.989) | 28.5 s |
| b1_squelch | FAIL (B1-lock) | lock 19.183 pm | 1.1 s |
| b2_laser_failover | FAIL (B2-relock) | ring 0 not locked about its former operating point at the end; BER 5.57e-131 | 1.3 s |
| b3_xpu_power_step | PASS | lock 0.122 pm; BER 1.85e-229 | 1.5 s |
| b5_cold_start | PASS | all 64 rings on their lines (wdm_supervisor), worst-lane OMA penalty median 2.2 dB | 21.5 s |
| a4_soak | PASS (section 12.3; with the A4 certificate) | BER windows 0, max \|detuning\| 126.499 pm, no rail, OMA penalty 0.12 dB; bit-identical to the builder's A4 | 12 590 s |

The nine quick runs equal the review runs of `runs/acceptance_review_20261010/` in every shared trace.h5 array (89 /
289 / 79 arrays compared, 0 differing) apart from A7's two corner coverages above; the new arrays are the warm-start
flags. `mrm rerun` of the new A1, A7 and B5 directories: bit-for-bit identical.

| example | result | wall |
|---|---|---|
| steady_pi_lock | worst-lane OMA penalty 0.117 dB, lock 0.062 pm, no excursion | 2.6 s |
| a1_pi_lock | as A1: OMA penalty 0.316 dB, 4 excursions, ring 0 348.7 pm | 4.3 s |
| mc_pi_lock (64 sampled worlds, static) | median OMA penalty 1.63 dB, p75 15.0, p95 31.9 dB; wrong-line rings p75 1, p95 5; warm start: 9 rings without a code in the 64 worlds (98 with the heater-up order: 89 moved one order bluer; per world median 1, max 6), 1 supply-limited world; 26 worlds have a ring > 5 pm from its own operating point, 10 of them without any warm-start flag (worlds 1, 3, 11, 13, 30, 32, 33, 39, 43, 48). Rerun 2026-10-16 after the engine review fixes (iterated order hop, per-world settle stop; runs/engine_fix/exp): median 1.63 dB, p75 15.8, p95 31.0 dB; 1 ring without a code (world 21 ring 4), 101 hopped rings, 2 supply-limited worlds (36, 63); 27 worlds off by > 5 pm, the same 10 unflagged | 20 s |
| sweep_pi_gains | fraction 0.6 / 0.7 / 0.8: OMA penalty 0.79 / 0.32 / 0.24 dB; crossover changes little; ring 0 326-371 pm at the cusp in every point (as before) | 32 s |
| sweep_hypotheses | default: 0.316 dB; FSR window A: 2 wrong-line rings, 12.7 dB; H86: 2 wrong-line rings, 4.0 dB; H86 + window A: 5 wrong-line rings, 53.8 dB | 13 s |
| quasi_static_g26 | coverage 1, no supply-limited point, BND-4 5.11e-9 dB; oracle relOMA -5.99 (1 V) to -9.23 dB (3.75 V) | 2.7 s |
| ones_density_steps | OMA penalty 0.236 dB, max lock error 48.9 pm (lock reference: line), BER 1.2e-190 | 9.5 s |
| robust_pi_lock (A1, 12 corners + 128 LHS worlds) | FAIL over the set (as before): lock up to 8627 pm, up to 8 rings on foreign lines; worst corner g26_hot_selfheat (8 rings on foreign lines, 19.8 dB); 40 of 128 worlds start supply-limited | 114 s (6 threads) |
| compare_locks (4 controllers, A1, 12 corners + 256 LHS worlds) | as before: median wrong-line rings 2 (PI locks) and 5-6 (kalman_lqg_c, mpc_c), OMA-penalty rank stability tau mean -0.13; **82 of 256 worlds start supply-limited**; without them (174 worlds) median wrong-line rings 1 (PI) and 4.5-5 (model based), median worst-lane OMA penalty 3.6 / 17-18 dB | 928 s (6 threads, loaded) |
| soak_window_hot (60 s at c = 1 from the 108 degC turnaround) | no excursion, no wrong line, OMA penalty 0.85 dB, BER 3.3e-151 (same as before) | 587 s |
| soak_reuse_certificate | PASS, identical to A4 (section 12.3) | 12 609 s |

Reading: with variation off no LHS world needs the reachable order (0 order-hopped rings), so compare / robust
quantiles did not move; about a third of those worlds are supply-limited from the start (the assumed 150 mA limit;
29 of 32 stacked_eic and 27 of 43 FSR-window-A worlds at 25 degC), which explains most of the "all stacked_eic worlds
fail" and much of the window-A failures of section 9 item 7. The remaining failures (A1 heater authority, the
model-based controllers' collapse across worlds, the 10 unflagged sampled worlds of mc_pi_lock) are controller or
plant behaviour, not the warm start.

### 12.3 A4 soak and certificate reuse (integrated tree)

Both with `experiment.convergence_certificate=runs/acceptance_20261010/20261010T115721Z_d1511028` (the certificate's
plant fingerprint 8e979786... is unchanged by the integration, so the gate accepted it and no convergence run was
needed), run side by side on the loaded machine:

| run | result | values | wall |
|---|---|---|---|
| `mrm acceptance experiments/acceptance/a4_soak.yaml` (`runs/acceptance_integrator_20261010/20261010T175753Z_1ddc6a1f`) | **PASS** | no 2 s BER window over 1e-9 (mean BER 3.88e-229), no wrong-line event, max \|detuning\| 126.499 pm (< 5000), heater never at a rail, worst-lane OMA penalty 0.12 dB; relOMA -6.156 dB (oracle -6.043) | 12 590 s (11.5 us per tick: load 4-12; 8547 s in the builder's run) |
| `mrm run experiments/examples/soak_reuse_certificate.yaml` (`runs/examples_integrator_20261010/20261010T175754Z_740b2ef1`) | PASS (same criteria, inherited) | identical | 12 609 s |

Bit for bit: the new A4 main case equals the builder's recorded A4 main case (48 metric arrays and 18 trace arrays
compared, 0 differing; only the two new warm-start flag arrays are added), and the certificate-reuse run equals the
A4 run (50 + 18 arrays). So the integrated tree reproduces the 4.8 h soak exactly, and the `run` and `acceptance`
kinds give the same plant run. The compressed-soak golden (hardware.md section 9) equals the A4 convergence case at
c = 100 in all 47 shared metric arrays.
