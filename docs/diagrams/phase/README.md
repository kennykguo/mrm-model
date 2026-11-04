# phase.py — visual companion

[open the gallery](index.html) · [complete editable atlas](phase-walkthrough.excalidraw)

17 focused diagrams plus your supplied ring reference. no equations.
all artwork uses native Excalidraw shapes, paths, and text. Excalifont at size 36 follows the supplied drawing.
each topic has an editable file and SVG/PNG exports. the complete atlas uses named frames.

## reading map

| drawing | code connection |
|---|---|
| [00 — supplied ring reference](drawings/00-supplied-ring-reference.excalidraw) | original 39 elements, preserved unchanged |
| [01 — reference state](drawings/01-reference-state.excalidraw) | `PhaseParams` and `from_plant_params` |
| [02 — bus segments](drawings/02-bus-segments.excalidraw) | `calculate_bus_segment_power_transmission` |
| [03 — thermal tuning](drawings/03-thermal-tuning.excalidraw) | `calculate_thermal_resonance_shift_m` |
| [04 — rebase the thermal law](drawings/04-rebase-thermal-law.excalidraw) | `rebase_thermal_tuning_coefficients` |
| [05 — junction device](drawings/05-junction-device.excalidraw) | junction depletion and partial ring coverage |
| [06 — voltage tables](drawings/06-voltage-tables.excalidraw) | `junction_eval_row`, reference subtraction, and clipping |
| [07 — combined shifts](drawings/07-combined-shifts.excalidraw) | `evaluate_ring_resonance_shifts_with_workspace` |
| [08 — loss and heat](drawings/08-loss-and-heat.excalidraw) | temperature multiplier and the loss-budget part of `write_ring_optical_inputs_for_data_level` |
| [09 — shifted phase curve](drawings/09-shifted-phase-curve.excalidraw) | reference-curve coordinate in `write_ring_optical_inputs_for_data_level` |
| [10 — wavelength dispersion](drawings/10-wavelength-dispersion.excalidraw) | quadratic index and linear group-index expansions |
| [11 — shared delay](drawings/11-shared-delay.excalidraw) | level-zero delay retained for both voltage cases |
| [12 — two data levels](drawings/12-two-data-levels.excalidraw) | `evaluate_binary_ring_optical_inputs_with_workspace` |
| [13 — evanescent overlap](drawings/13-evanescent-overlap.excalidraw) | chapter 3/4 physical picture behind wavelength-sensitive coupling |
| [14 — coherent power exchange](drawings/14-coherent-power-exchange.excalidraw) | ideal chapter 4 coupler, distinct from the runtime wavelength fit |
| [15 — local coupling model](drawings/15-local-coupling-model.excalidraw) | `evaluate_coupler_power_fractions` and the offline extracted slope |
| [16 — rigid resonance comb](drawings/16-rigid-resonance-comb.excalidraw) | shared shift for every order and omitted thermal change in spacing |
| [17 — optical-input handoff](drawings/17-optical-input-handoff.excalidraw) | `evaluate_ring_optical_inputs` |

source — [phase.py](../../../mrm/plant/phase.py), [junction lookup](../../../mrm/plant/electrical.py), [coupling extraction](../../../mrm/extraction/coupling.py).

## how to use the drawings

open the gallery directly from disk. select a topic or use the arrow keys. enlarge a diagram to inspect labels, then drag to pan.
open an individual `.excalidraw` file in Excalidraw, or load the complete atlas and select a named frame.
SVG previews embed the drawing font and work offline.

the curves and device cross-sections are schematic. proportions and delay differences are exaggerated where needed.
they explain the implemented assumptions. they are not measured device data or substitutes for the extraction results.
the junction resistance curve only illustrates table lookup; its precise trend depends on the supplied table.
loss changes and coupling changes need not make a ring notch shallower or deeper in every coupling regime.

## construction

[design and coverage](DESIGN.md) · [layout generator](../../../tools/build_phase_diagrams.py) · [style guide](../../../EXCALIDRAW.md)

the generator writes native Excalidraw elements. previews are exported with the official Excalidraw renderer after measuring the supplied font.
`render-report.json`, `validation.json`, and `viewer-validation.json` record export and layout checks.
no production simulation code was changed.

to rebuild, run `python3 tools/build_phase_diagrams.py` from the repository root.
then use [the renderer](../../../tools/render_excalidraw_atlas.mjs) with `--packages` pointing to a separate `node_modules` directory containing `@excalidraw/excalidraw@0.18.0` and `vite@6.1.0`.
that directory also needs `playwright`, or supply its directory with `--browser-packages`.
the renderer uses installed Chrome by default. dependencies are not added to the simulation environment.
