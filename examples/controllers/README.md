# Controller templates and the feedback-lock baselines

Templates (copy one to start a controller):

| file | what | API |
|---|---|---|
| `template_python_controller.py` | integral lock of the drop monitor to a fraction of a known peak | `BaseController` (vectorized over B, N) |
| `template_per_ring.py` | perturb-and-observe peak seeker written for one ring | `PerRing` adapter |
| `template_compiled_controller.py` | the integral lock as a numba controller (flat int64 ABI), with the fast-kernel signature | `CompiledController` |

`tests/verification/test_ctrl_a_examples.py` runs all three on a toy ring. `pi_lock` is also run on every named
corner preset of the uncertainty set (`tests/verification/test_ctrl_a_corners.py`, slow).

## The feedback-lock baselines (`mrm/control/baselines/`)

All of them see only the `ControllerSpec` and the `Observation` (ADC codes, sample ticks, temperature-sensor codes,
applied codes). Model knowledge comes only from `spec.nominal` for the tokens listed in
`controller_interface.knowledge` (`pi_lock.NominalModel`; tokens `heater_law`, `heater_r_nominal`,
`heater_efficiency`, `dlambda_dt_nominal`, `linewidth`, `thermal_dynamics`, `bias_efficiency`, besides `line_grid`
and `ring_to_line`). g26 lists only `line_grid` and `ring_to_line`, so the loops tune themselves from what they
measure (the hand-off map of the peak and the slope, a bias or heater identification step).

| registry name | module | idea | actuators | compiled |
|---|---|---|---|---|
| `pi_lock`, `pi_lock_c` | `pi_lock.py` | drop-port PI at a fraction of the measured peak, red (stable) side; MAP / RETURN / fall-off recapture; gain schedule on the driver law; optional type 2 (`k2`) | heater | yes (parity tested) |
| `dither_lockin`, `dither_lockin_c` | `dither_lockin.py` | three-level heater dither, drift-free 1f/2f demodulation; peak (1f = 0) or inflection (2f = 0) lock | heater | yes (parity tested) |
| `bias_dither` | `bias_dither.py` | bias dither demodulated in the `fast_kernel` per ADC sample; the heater integrates the 1f/DC-ratio (or 2f) error | heater + bias | fast_kernel + step |
| `es_po`, `es_sine` | `extremum_seeking.py` | perturb and observe; sinusoidal extremum seeking (peak of the drop monitor) | heater | no |
| `feedforward_pi` | `feedforward.py` | `pi_lock` plus a die temperature-sensor feed-forward (first-order hold between sensor samples) | heater | no |
| `midranging` | `midranging.py` | bias holds the lock (fast), the heater re-centres the bias (slow) | heater + bias | no |

Hand-off contract: at reset the ring sits on the stable (hot) branch of its assigned line, red of the laser (an
acquisition controller, the WDM supervisor or a warm start put it there); the heater code is `init_codes` or the
latched register `tx.heater.applied`. `pi_lock` and its derivatives then map the peak themselves (`start='map'`), or
take a factory table (`start='lock'`, `peak_codes`, `slope_per_code`), or ramp to a capture first
(`start='capture'`; it locks on whatever line it meets first, so in a bank it is the supervisor's job to pick the
line). `start_delay_s` holds the hand-off code while the thermal state settles (cold-state warm starts).

Bias controllers need `controller_interface.bind.tx.bias: controller` (and no absolute `scenario.bias_v`).

## Physics that shapes them (g26 nominal; numbers from the tests, run with `-s` to see the CTRL_A_METRIC lines)

- Self-heating (07 P0-1): bistability ratio ~6. The lock must be red of the laser; the heater-to-detuning gain is
  divided by the self-heating relief (3-7x), and so is every disturbance. The fold sits ~9 pm blue of the peak:
  peak locks (1f dither, extremum seeking) fall off it; a fall moves the resonance ~1.5 nm blue, where rings 4-7
  meet line n-4 with their next FSR order (BND-1), so the recapture logic ignores captures below a code guard.
- The bias barely tunes the resonance in this model (a 0.2 V step: 0.1-0.5 pm, the FCA self-heating change cancels
  most of the electro-optic shift) and changes the loss: the bias-dither and mid-ranging controllers work, but
  mostly through the loss term (measured slopes of the monitor against bias are positive on several rings). Under
  the A1 rate the bias saturates at once; mid-ranging then falls back to the heater PI (`saturation_fallback`,
  9.7 pm max at the victim; without it the lock was lost).
- The monitor peak can exceed the ADC full scale (monitor gain_scale up to 2; +8 dBm at the ring in the
  `g26_hot_selfheat` corner puts it at ~2.5x): `pi_lock` then takes the full scale as the peak (telemetry
  `peak_codes` = 65535) and locks further red; when the monitor is clipped from the hand-off on it first ramps red
  until it reads below f x full scale (state `declip`), then locks (~275-315 pm red in that corner: a large OMA cost).
- The fall-off detectors (every lock here) look for a monitor drop of more than 35 % of the peak within 3 steps to
  below 40 % of it. A fold fall does that; so would a RED step of the line or ring of ~200 pm within ~30 us, which
  is then mistaken for a fall (the recapture ramps the wrong way and gives up: FAULT). A1/A2 rates are far slower;
  `test_ctrl_a_pi.py::test_fast_red_step_is_not_mistaken_for_a_fall` is a strict xfail recording it.
- The slide-10 aggressor (A1) at its peak rate (2127 degC/s, 183 pm/ms at the victim) is held by the PI within
  ~7-8 pm (type 2: 6.8 pm); the heater dither cannot follow it (~110 pm). Over the full A1 trajectory the aggressor
  heats the monitor PD ~6x more than the ring in the interim layout; its dark current then moves the fraction-of-peak
  lock point red until lock is lost near the cusp (handoff controllers_a, request 4).
- The inflection (2f) dither lock is nearly power independent (-9 pm for -1 dB, against -57 pm for the
  fraction-of-peak PI).
