# The MRM model, explained

This project is a **virtual test bench for a controller that keeps optical rings tuned**.
It simulates eight microring modulators in a Lightmatter G26 transmit bank: how they respond to light,
heat, electrical drive and your controller's commands.

The question it helps answer is: **can your controller keep every ring at a useful operating point,
on the right laser wavelength, while conditions change?**

## Picture the device

Eight laser wavelengths travel along one shared waveguide. Beside it sit eight tiny silicon rings,
each assigned to one wavelength. A ring interacts strongly with light near its **resonance**—its
preferred wavelength. An electrical data signal shifts that resonance rapidly, changing how much
light continues along the waveguide. That is how the ring puts data onto light.

Temperature also shifts the resonance. Left alone, a ring drifts away from the operating point that
makes modulation work well. Each ring therefore has a heater for tuning and a photodiode that measures
light leaving its drop port, a separate output from the ring.

Your lock controller closes this loop:

```text
heater command → heater power → ring temperature → resonance → monitor reading
      ↑                                                               │
      └──────────────── controller adjusts the heater ────────────────┘
```

The fast data signal carries the bits; the slower heater loop keeps the ring tuned. The default link
uses 50 Gb/s two-level data. The simulator advances in 1 µs ticks, and the controller updates every
10 µs (100 kHz).

## What “locked” means

A good operating point is usually on the **side of the resonance**, rather than at its peak.
The monitor reading changes with position along that slope, giving the controller a signal to follow.
The shipped PI controller measures the monitor peak and adjusts heater power to hold a fraction of
that reading on the stable side. That fraction is a **tunable hyperparameter**,
`controller.params.fraction`, with a default of 0.7 (70% of the measured drop-monitor peak).
It is an operating-point choice, not a physical constant or a guarantee of maximum slope or modulation amplitude.

Two distinctions matter:

- **Lock error is movement away from the intended operating point.** A ring can be correctly locked
while its resonance is offset from the laser. In the example experiments, error is measured against
that ring's own steady operating point.
- **A steady monitor reading does not prove the ring is on the right wavelength.** Rings have repeated
resonances and can capture another laser line. Check wavelength assignment as well as lock error.

The model defines **detuning = ring resonance wavelength − laser wavelength**. Positive detuning means
that the ring resonance is at a longer wavelength (“red” of the laser). This is the stable side used by
the shipped PI lock. The textbook uses the opposite sign for its δ.

## Why the controller has a difficult job



### Light heats the ring too

Some of the circulating light is absorbed and becomes heat. That heat shifts the resonance, changing
how much light is absorbed. The ring's temperature and its optical response therefore affect each other.

On the stable side, this feedback softens both heater adjustments and disturbances: in the nominal
model, their effect on resonance is reduced by roughly 3–7 times. Near the end of that stable branch,
the ring can suddenly jump to a much colder state and lose lock. The same heater command can support
two stable temperatures; this is **thermal bistability**.

This is why simply seeking the largest monitor reading can fail. It is also why laser power changes
can move the lock point even when the laser wavelength stays fixed.

### The heater has limits

The default heater is driven by a 16-bit voltage DAC, with commands from 0 to 65,535. Power roughly
follows V²/R, and resistance rises as the heater warms. A change of 100 codes therefore has a different
effect at low and high drive. All eight heaters also share supply limits.

Most importantly, a heater can only add heat. Reducing its command helps only until it reaches zero.
With the present model parameters, the full A1 heat disturbance pushes ring 0 beyond that limit:
its heater stays at zero for about 15 ms, and lock error reaches roughly 455–473 pm. That particular
failure needs more cooling or tuning range; changing controller gain cannot remove it.

### Heat spreads through the bank

A ring feels heat from its own heater, absorbed light, the data driver, neighbouring rings and an
external heat source called the **aggressor**. Each path has its own strength and delay. Adjusting one
ring can therefore disturb another later.

The default model has fast heater responses around 3 and 13 µs, a small 3 ms tail, and slower crosstalk
between rings. The nearest neighbour receives about 3% of the heater's own-ring temperature response.
These dynamics come from other devices and assumptions; they have not been measured for G26.

A1 represents the published on-die heating experiment. The simulator applies heat at an aggressor site,
so rings, monitor photodiodes and temperature sensors all feel it through their respective thermal paths.
Other scenarios include ambient warming, laser drift, power steps, traffic changes and faults.

### The measurement is imperfect

The controller sees photodiode readings converted into ADC codes, with integration time, delay, noise,
uncertain gain and possible clipping. The photodiode also produces **dark current**, a background signal
that grows as it warms. A higher reading can therefore mean something other than a resonance shift.

The default monitor integrates for 2 µs, adds 2 µs of ADC latency, and samples at 250 kHz. Heater commands
have another 1 µs of latency. These delays matter when choosing how aggressively to correct an error.

### Bias and traffic also change the ring

The PN junction carries the fast data signal. Its DC bias changes resonance and optical loss; the data
and nearby driver also generate heat. Starting or stopping traffic can move the ring's temperature.

Bias can be a second actuator, but it is unexpectedly weak near the default model's lock point:
changes in absorbed heat cancel much of its direct wavelength shift. This remains an open physics
check. By default, bias stays at 1.05 V and the controller drives only the heaters.

## Which numbers should you believe?

The default preset is [`g26.yaml`](../mrm/params/presets/g26.yaml). It combines G26 publications, earlier
G25 measurements, foundry information, other groups' devices, simulations of assumed geometry and
explicit assumptions. **It is a working hypothesis of the device.**

A few useful scales, for the nominal preset:


| Quantity                          | Approximate value                             | Why it matters                                               |
| --------------------------------- | --------------------------------------------- | ------------------------------------------------------------ |
| Laser grid                        | 8 lines, 400 GHz apart, near 1310 nm          | Each ring must stay with its assigned line.                  |
| Optical power at the first ring   | +4 dBm per line                               | Sets monitor signal and optical self-heating.                |
| Resonance linewidth               | 338 pm, loaded Q ≈ 3850                       | Sets the width and slope of the optical response.            |
| Wavelength shift with temperature | 65 pm/K at 25 °C, rising to 74 pm/K near lock | Converts heating into wavelength drift.                      |
| Cold resonance offset             | 7.6 nm below its assigned line                | Requires substantial heating; nominal lock is around 134 °C. |


Every parameter records where it came from: directly published, read from a figure, inferred by
arithmetic, computed by a solver, taken from other literature, or assumed. See
[PARAMETERS.md](PARAMETERS.md) for the full table and sources.

When a value is unknown, the model can try several plausible choices. A **world** is one complete
choice of those uncertain values. The **nominal world** uses the defaults; **corners** are selected
challenging combinations; **sampled worlds** explore more combinations. Passing the nominal world
shows that the controller works for that choice. A robustness claim needs all tested worlds to pass.

### Why some values differ from the textbook

The largest differences are:


| Topic                                  | Textbook              | Default model                                            |
| -------------------------------------- | --------------------- | -------------------------------------------------------- |
| Temperature sensitivity                | Constant 86 pm/K      | About 65–74 pm/K over the nominal heating range          |
| Fraction of optical loss becoming heat | 0.3                   | About 0.85, with stronger heating per mW than the heater |
| Main heater response times             | 10 and 300 µs         | 3.18 and 13 µs, plus a small 3 ms tail                   |
| Nearest-neighbour thermal crosstalk    | 10%, arriving quickly | 3%, arriving over milliseconds                           |
| Monitor background                     | No dark current       | Dark current grows with photodiode temperature           |


These choices materially change controller behaviour. Neither set has been verified against your
actual device. Several textbook choices remain available as alternative worlds. For the evidence
behind each disagreement, see the comparison in [PARAMETERS.md](PARAMETERS.md) and the
[textbook audit](research/06_textbook_audit.md).

## Try it

Run these commands from the repository. The memory guard stops a job above 3 GB; initial compilation
can take a few minutes.

```sh
cd /Users/kenny/mrm-model

# First: the shipped controller at constant ambient temperature, for 40 ms.
.venv/bin/python tools/memguard.py -- .venv/bin/mrm run experiments/examples/steady_pi_lock.yaml

# Next: the same controller under the A1 heat disturbance.
.venv/bin/python tools/memguard.py -- .venv/bin/mrm run experiments/examples/a1_pi_lock.yaml

# Finally: nominal, corner and 128 sampled worlds under A1.
.venv/bin/python tools/memguard.py -- .venv/bin/mrm run experiments/examples/robust_pi_lock.yaml
```

These examples use a **warm start**: the simulator supplies a settled initial tuning state. They test
maintaining lock; startup and finding the right laser line require separate testing. The robust run
prints `FAIL` and exits with code 1 if a criterion fails. That can be an experiment result, not a crash.
Add `--dry-run` to estimate a run's cost before executing it.

### Use your own controller

Copy [`template_python_controller.py`](../examples/controllers/template_python_controller.py) to
`examples/controllers/my_lock.py`. Give its `@register(...)` entry and `name` attribute a unique name,
then edit `step()`. The template uses an assumed peak reading; supply a measured or calibrated peak
for serious use. The shipped PI controller includes its own peak-mapping procedure.

Your controller reads the observation and interface specification, then returns integer heater codes.
The default interface gives it the laser grid and ring assignments; other device properties must be
learned from measurements. Internal simulator “Truth” is for evaluating the controller.

```sh
.venv/bin/python tools/memguard.py -- .venv/bin/python -m mrm.cli run experiments/examples/steady_pi_lock.yaml \
  --set controller.name=examples.controllers.my_lock:MyIntegralLock
```

Change the class name after the colon if you rename it. `python -m mrm.cli` lets Python import your controller from the repository. Use `--set PATH=VALUE` for a one-run override, or `--alt PATH=NAME` for a stored alternative. The [controller guide](../examples/controllers/README.md) covers the other templates and bias control. Use experiment files for comparisons: calling a scenario by name through the Python runner currently uses different signal defaults, including 1.0 V bias instead of 1.05 V.

### Read the result

Each run saves a directory under `runs/`, including `report.txt`, the resolved configuration and
`trace.h5` with recorded traces. Start with these report fields:


| Field                   | Meaning                                                                                 |
| ----------------------- | --------------------------------------------------------------------------------------- |
| `max_abs_lock_error_pm` | Largest departure from the ring's lock reference; lower is better.                      |
| `unlocked_rings`        | How many rings are unlocked at the end.                                                 |
| `wrong_line_rings`      | How many rings finish on the wrong laser line.                                          |
| `oma_penalty_db_worst`  | Worst loss of optical modulation amplitude relative to the best point; lower is better. |


Check both lock quality and modulation quality: a steady ring can still transmit poorly. The
[engine guide](ENGINE.md) explains trace access and reproducible reruns.

## What the model can tell you today

It is useful for studying feedback, comparing controller ideas and finding failure modes under stated
assumptions. Its present results have several limits:

- **Calibration and validation reports are stale.** Use the `g26` preset for these examples; the presets
named `*_calibrated` need refitting. Do not quote the old validation reports as current evidence.
- **Several published measurements are still missed**, including heater behaviour and modulation
amplitude. Some physics, especially bias tuning and thermal dynamics, needs further checking.
- **Some scenarios exceed the hardware limits assumed by the model.** A1 exhausts ring 0's ability to
reduce heater power. Assumed shared-supply limits also prevent lock in some sampled worlds.
- **Startup and robustness remain open.** The supplied wavelength supervisor can accept a wrong line
in some worlds, and even warm-start runs can lose rings when uncertain parameters are sampled.

[KNOWN_ISSUES.md](KNOWN_ISSUES.md) records the details. For implementation contracts see
[ARCHITECTURE.md](ARCHITECTURE.md); for engine behaviour see [ENGINE.md](ENGINE.md).