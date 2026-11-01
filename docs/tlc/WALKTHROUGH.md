# How the TLC runs inside mrm (Phase 1 walkthrough)

A short tour of one closed-loop run, following the data from the plant to the heater and back.

## 1. Who calls whom
- **mrm's runner** steps the plant every **1 µs** (a *tick*). Every **10 ticks** (10 µs, G26's 100 kHz controller
  rate) it stops at a *controller edge* and calls `TlcController.step(obs)`.
- `obs` holds only what the hardware would show: monitor ADC codes (`tx.monitor.samples`, with
  `tx.monitor.count` saying how many are new), the applied heater codes, validity flags. Never the true detuning.
- `step` must return `{"tx.heater": codes}`: one 16-bit DAC code per ring and world, `(B, N)` int.

## 2. Why the adapter is multirate
The ECE496 core was designed to run every **100 µs**: it eats a block of PD samples and plans 10 heater samples
(100 kS/s) that carry the dither tones. mrm calls us every 10 µs, so the adapter:

```
edge:   0    1    2   ...   9   10   11  ...
        │    │    │         │    │    │
push    ●────●────●── ... ──●────●────●     2-3 monitor samples per edge → SampleBuffer
core    ▲ run                    ▲ run      every 10th edge: drain buffer → core.step(block) → plan[10]
emit    p0   p1   p2  ...  p9   p0'  p1'    one planned sample per edge → code
```

The samples delivered at an edge were taken in the 10 µs before it, so the block the core sees at edge 10 is
edges 1-10: 25 samples at G26's 250 kS/s (the source had 100 at 1 MS/s; Phase 3 rescales B2 for that).

## 3. Units: the core never sees a code
| Quantity | Core | mrm | Law (`mrm/control/tlc/units.py`) |
|---|---|---|---|
| heater | power fraction u ∈ [0, 1] | 16-bit voltage DAC code | power ∝ V² ∝ code², so `code = round(65535·√u)` |
| PD | normalised level | ADC code (0-65535) | `level = code / 65535` (Phase 3 maps it to the source's scale) |
| detuning | GHz | pm | `pm = GHz · λ²/c` (5.72 pm/GHz at 1310 nm), same sign: heating raises both |

## 4. What the stub proves
The stub core holds the warm-start command (and can add the source's 4 dither tones). With nothing disturbing
the static world, holding the heater holds the lock: the first report shows the stub at 0.002 pm rms (10 µs-mean trace), against 0.013 pm
for `pi_lock_c`, whose integrator walks a little in the noise. That is a statement about the static scenario, not about the TLC.
Once the plumbing is proven, every later phase swaps a real part in behind the same interface.

## 5. Where results come from
- `tools/tlc_run.py` runs an experiment and saves the controller's telemetry log (`tlc_telemetry.npz`), because
  mrm's runner records only plant Truth.
- `tools/tlc_report.py` reads the run directory: mrm's metrics, the true detuning trace (Truth is allowed here:
  this is scoring, not control) and the telemetry, and writes `docs/tlc/results.md`.
- The boundary test (`tests/tlc/test_boundaries.py`) fails if anything under `mrm/control/tlc/` imports the plant
  or the simulator, so the controller cannot cheat.
