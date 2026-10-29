# ring optics - visual atlas

open [the gallery](index.html) to browse one drawing at a time. use the arrow keys to move between drawings. enlarge a drawing when you want to inspect its labels.

open [the complete atlas](optics-walkthrough.excalidraw) in Excalidraw, or drag in a file from [drawings](drawings/). the atlas has a named frame for each topic. the individual files are easier to edit without nearby material.

there are 26 drawings - your supplied ring drawing and 25 new diagrams. the new drawings use the reference's handwritten font, canvas font size 36, dark strokes, and transparent shapes. text, paths, and shapes remain editable. no mathematical equations were added.

the [SVG and PNG previews](previews/) were exported through Excalidraw's renderer. the gallery works from a local file and needs no server or external scripts.

## what does each group cover?

| drawings | concepts | code to read alongside them |
| --- | --- | --- |
| 00–06 | ports, resonance orders, field interference, backscatter, travel directions, and field closure | [optics.py](../../../mrm/plant/optics.py) - `_exact_line`, `_exact_line_d`, `_phase_parts`, `_interpolated_scatterer_amplitude` |
| 07–10 | energy and heat, both data levels, eight-ring propagation, and returned reflections | [optics.py](../../../mrm/plant/optics.py) - `evaluate_ring_optical_outputs`, `_bus_pout_nb`, `_bus_refl_nb` - [electrical.py](../../../mrm/plant/electrical.py) - `nrz_levels` |
| 11–17 | heat paths, feedback, equilibrium arrows, folds, thermal memory, trial temperatures, and sensitivity bounds | [thermal.py](../../../mrm/plant/thermal.py), [selfheat.py](../../../mrm/plant/selfheat.py), and the trial, commit, and bound functions in [optics.py](../../../mrm/plant/optics.py) |
| 18–20 | monitor target, measurement imperfections, and statistical link quality | [PI controller](../../../mrm/control/baselines/pi_lock.py), [sensing.py](../../../mrm/plant/sensing.py), [datapath.py](../../../mrm/plant/datapath.py) |
| 21–22 | verification versus device evidence, and the data, plant, and controller time scales | [verification checks](../../../tests/verification/test_optics.py), [kernel.py](../../../mrm/sim/kernel.py), [known issues](../../KNOWN_ISSUES.md) |
| 23 | thermal mass and resistance - read before 11 if you want the thermal foundation first | [thermal.py](../../../mrm/plant/thermal.py) |
| 24 | the production ring calculation and the archived steady-state TCMT alternative | [optics.py](../../../mrm/plant/optics.py), [archived TCMT](../../../archive/optics/tcmt.py), [time-domain waveform model](../../../mrm/waveform/tcmt_time.py) |
| 25 | the connection from the transmission notch to peak seeking, a fold, and loss of lock | [selfheat.py](../../../mrm/plant/selfheat.py) and [PI controller](../../../mrm/control/baselines/pi_lock.py) |

## how should the curves be read?

the curves are schematic. they explain directions, branches, and model assumptions. they are not calibrated device predictions. the measurement dots in drawing 21 are illustrative, rather than experimental data.

in drawing 01, the laser sits on the left flank of a wavelength scan. the ring resonance is at a longer wavelength. this is the stable-side convention discussed in the walkthrough. the opposite flank is not automatically unstable - feedback must overcome cooling for instability to occur.

in drawing 13, arrows show the direction of temperature motion. solid curves show absorbed heating. dashed lines show the heat loss still to be supplied after other heating. drawing 14 follows equilibria as heater input changes. drawing 16 instead concerns a finite-step endpoint solve. its trial iterations do not advance physical time.

## where are the editable sources?

the `.excalidraw` files are the editable deliverables. [the generator](../../../tools/build_excalidraw_atlas.py) records the layout and labels. running it recreates native drawings from those definitions and the supplied reference. it does not regenerate the rendered previews or combined atlas - export those again through Excalidraw after source changes.

the reference drawing came from the attachment supplied in this chat. its elements are preserved without redrawing. the production simulation code was not changed.
