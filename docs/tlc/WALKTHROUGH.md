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

## 6. The real core (Phases 2-3)
- **Vendored, not rewritten.** `mrm/control/tlc/core/` is the ECE496 code with relative imports and one
  structural change: the ring count is `ControllerParams.n_rings` instead of a module constant. A test runs the
  vendored and the original core side by side on the original plant and requires every logged signal to be
  bit-identical; another reproduces the published R1/R3/R4 numbers.
- **Priors: how the core "knows" G26.** The core models its plant with ECE496 `PlantParams` (Lorentzian lineshape,
  two heater poles, self-heating, crosstalk). `priors.py` fills them from the nominal G26 design that the
  `model_nominal` token releases: the tuning slope at the lock temperature (74 pm/K → 12.9 GHz/K), a Lorentzian
  fitted to the nominal drop curve (HWHM ≈ 32 GHz, about 1.6x ECE496's), the NRZ swing from the junction model
  (±33 pm), the heater and self-heating residues of the thermal network split into a fast and a slow path, and the
  crosstalk fractions. The controller never sees a sampled world.
- **Eight tones.** The source plan has four. `tone_plan.py` encodes its collision rule (tone and 2nd harmonic
  clear of every other tone, harmonic, intermodulation product and the 10 kHz ripple, on the 250 Hz lock-in grid);
  the search finds the source's four-tone plan exactly, and an eight-tone plan up to 13.5 kHz.

## 7. What the first measurements say (Phases 3-4)
- **The slope channel works, the ratio channel barely does.** The lock-in's first harmonic z1 matches the source
  model on G26 (0.87x, SNR ≈ 50), but ζ = z2/z1, the power-invariant detuning measurement the UKF leans on, spans
  only about twice its noise: z2 scales with (dither / linewidth)², and G26's line is 1.6x wider.
- **Steady state is real tracking error.** The faithful TLC holds 2-4 pm rms; removing its dither tones from the
  error barely changes it.
- **Where it breaks:** the A1 aggressor (lock lost, ~1 nm), cold start (the ACQUIRE ramp stops at the first fringe,
  usually a foreign line), squelch. Each was predicted in the consultation brief and is now measured.
- **Model trust matters:** under the TEC-held package corner the TLC rides through A1. Conclusions are stated per
  corner in the report.
