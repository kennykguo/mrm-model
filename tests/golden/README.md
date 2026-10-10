# Golden traces and golden vectors

Owner: hardware task (07 P1-21, section 4.9). These tests block CI (07 4.1).

## Golden traces: `<preset>__<scenario>.npz`

Metrics only (`mrm.sim.metrics.summarize` of each run, flattened as `<run>/<metric>`), no traces. The cases, their
run conditions and the comparison rule are in `mrm/control/cli.py` (`GOLDEN_CASES`, module docstring):

| file | scenario | runs |
|---|---|---|
| `g26__static.npz` | static lock, 25 degC, 20 ms | open_loop, pi_lock_c, fxp_pi |
| `g26__a1_slide10_aggressor.npz` | the fast aggressor (A1 slide-10), one period from the cool minimum | open_loop, pi_lock_c, fxp_pi |
| `g26__soak_compressed.npz` | the compressed soak: one 20.3 min cycle 33 -> 108 -> 33 degC at time compression 100 (12.18 s), auto substeps, scored from 5 ms; the c = 100 case of the A4 convergence experiment, which validated compression 100 (07 P0-10) | pi_lock_c only (no convergence experiment covers the others); slow (95 s on an idle machine, ~3.5 min under load) |

Every run: g26 nominal world, seeds 0, B = 1, heaters at the quasi-static lock codes with the thermal state
settled; fixed 1 us ticks except the soak (auto). The static and A1 cases build their scenario by registry name (no
config defaults: the bias is not driven and sits at the bias DAC's initial 1.0 V); the soak takes the scenario of
its config (`scenario: config`, as an experiment file does: bias 1.05 V).

    mrm golden                       # rerun every case and compare (exit 1 on a difference)
    mrm golden --list
    mrm golden --update --case g26__static --reason "why the plant changed"

A file changes only through `mrm golden --update --reason ...`, which prints one CHANGELOG line per changed file;
`test_golden.py::test_every_golden_file_is_in_the_changelog` fails until docs/CHANGELOG.md contains each file's
data hash. Never edit a file by hand: `load_golden` checks the hash.

## Golden vectors: `vectors/fxp_pi__g26__aggressor_square.npz`

The observation and action streams (flat int64 ABI) of mrm.control.fxp.FxpPi on g26 nominal under a direct square
aggressor (0.05 W at the g26 aggressor site, period 2 ms, phase 1 ms), 4 ms = 400 controller edges, settled warm
start. Only the `.npz` is committed; the `$readmemh` files are generated from it:

    mrm vectors hex tests/golden/vectors/fxp_pi__g26__aggressor_square.npz --out /some/dir/fxp_pi
    mrm vectors replay tests/golden/vectors/fxp_pi__g26__aggressor_square.npz      # FxpPi, compiled ABI

Regenerate only with a CHANGELOG line that names the new file's sha256 (first 12 hex digits of
`vectors.load_vectors(path).sha256()`; `test_golden.py::test_every_golden_file_is_in_the_changelog` checks it).
test_vectors.py checks that today's plant still produces the committed file:

    mrm vectors record --preset g26 --scenario aggressor_direct \
        --scenario-params '{"waveform": "square", "power_w": 0.05, "period_s": 0.002, "duty": 0.5, "phase_s": 0.001}' \
        --controller mrm.control.fxp:FxpPi --duration-s 4e-3 --out <dir>/fxp_pi__g26__aggressor_square
    # then copy <dir>/fxp_pi__g26__aggressor_square.npz here

Users: tests/verification/test_vectors.py (replays, plant check), test_bridge_c.py (the C twin), tests/hdl/ (the
Verilog twin under Icarus).
