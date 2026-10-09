# Delivery and verification audit

Checked against the supplied request on 2026-10-09. The delivered artifact is
the MRM simulation/control laboratory in this repository. The capstone's later
FPGA deployment is supported by a measured-code/timestamp controller boundary;
RTL and a proprietary end-to-end Lightmatter link are not claimed here.

| Requested outcome | Current implementation and inspected evidence |
| --- | --- |
| Model informed by the supplied paper, public Lightmatter information and local references | [Research](research.md) separates OFC/G25/G26/later presentation constraints from assumptions. It references the supplied three-page paper and exact local textbook helpers; neither reference repository was modified. |
| Most defensible feasible accuracy without paid solver access | Exact periodic add-drop optics, passive thermal networks, real hardware timing/noise/limits, and a separate dynamic cavity model. Independent SAX component-network checks and sparse thermal/vector-mode extractions are recorded in [validation](validation.md) and [extraction](extraction.md). Missing process data is not replaced with claims of hardware accuracy. |
| One interface to vary the model and compare multiple algorithms | Strict `ExperimentConfig` JSON is shared by Python, CLI and Streamlit. Single-ring, eight-ring and FEM-derived profiles load and round-trip. Open-loop, PI, adaptive dither and local MPC run through the same measured observation/command protocol; custom controllers use it too. |
| Meaningful variation and fair comparison | Nominal calibration and detuning targets remain fixed across jointly specified corners. Independent deterministic noise streams, identical initial conditions and common objectives are exercised by integration tests. True detuning/temperature is confined to evaluation, not controller observations. |
| Exhaustive usable input/parameter explanation, including uncertainty | [Parameter inventory](parameter-inventory.md) covers all 131 fields across 12 public parameter/context/observation dataclasses, lower-level waveform/extraction/fit options, derived quantities and unimplemented conditional physics. Public constraints and assumed numeric profiles are distinct. |
| Intuitive explanation and justified textbook disagreements | [Model guide](model-guide.md) explains the light/heat/electronics chain, units and limits. [Research](research.md) anchors disagreements in actual add-drop linewidth, ambient-memory, carrier-change and rail-polarity implementations and primary thermo-optic literature. |
| Realistic route to device accuracy | Bounded optical/thermal fitting, residual and identifiability diagnostics, held-out sample support, passive ROM imports and [a prioritized Lightmatter data request](lightmatter-data-request.md). Synthetic recovery tests are not presented as hardware calibration. |
| RAM-aware, usable software | Preflight trace-byte/work/DOF limits, preallocated/decimatable traces, bounded waveform arrays and optional offline solvers. Recorded vector extraction used about 757 MiB peak RAM. Locked installation resolves; source distribution and wheel build; live lab health and actual app callbacks work. |

## Current checks

The final full environment ran **103 tests successfully**, with no skips, and
passed Ruff and bytecode compilation. Optional SAX, sparse FEM and Femwell are
installed and their test paths were exercised. The wheel and source distribution
were built with `uv build --out-dir outputs/dist`. `uv sync --locked --dry-run`
resolved the checked-in dependency lock; the separately documented minimal
Femwell installation follows that synchronization.

The current single-ring preset completed all four 10 ms CLI runs. Assigned
detuning RMS was approximately 18.430 GHz open-loop, 1.990 GHz PI, 1.653 GHz
dither and 0.244 GHz MPC. Those numbers describe one synthetic 2 K boundary-ramp
scenario, not a universal controller ranking or a reproduction of the paper's
local-aggressor test. Configurations, warnings and traces are in
`outputs/final-comparison`; outputs are intentionally Git-ignored and can be
regenerated with the README commands.

The final audit also tests explicit initial-state calibration, current/power
compliance and TCR, saturated-ADC observability, signed modal heat inputs,
fractional command/sample delays, and decimation-independent transient warnings.
Streamlit callback tests include stale waveform inputs; waveform inputs are
downloadable separately from arrays. The live `/_stcore/health` endpoint returned
`ok`. No in-app browser backend was available, so visual layout has not been
verified by browser screenshots.

## What these checks do not prove

The model is numerically verified under declared assumptions, not calibrated
to manufactured Lightmatter hardware. Coupling/loss, voltage response, thermal
and package behavior, electronics and manufacturing correlations still require
measurements or a shareable process-specific model. Synthetic fit recovery,
solver convergence and a good tracking score do not close that gap.

Slow runs do not compute BER. Short-waveform Gaussian BER is only a labeled
proxy; waveform nonlinearities are not fully coupled back into the slow thermal
plant. Conditional microscopic/link effects and extrapolation limits are
explicit in the inventory and guide. There is no FPGA RTL or deployed controller.
At the time of this completion audit, all changes were local and no GitHub push
or external message had been made. Subsequent publication is a separate action.
