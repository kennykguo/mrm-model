# phase.py visual companion

build native, editable Excalidraw diagrams from the current phase.py walkthrough.
use EXCALIDRAW.md and the supplied 39-element ring drawing as the style reference.
preserve that supplied drawing. use Excalifont, size 36, dark sketch strokes, and transparent fills.
add no equations. use short labels, physical geometry, directions, and schematic graphs.
keep physical views separate from the final calculation-flow drawing.

## coverage

1. reference temperature, bias, resonance, and stored junction baseline.
2. bus attenuation, ring encounter order, input loss, and unmodeled final segment.
3. thermal wavelength shift and changing sensitivity.
4. rebasing the same thermal curve around the previous temperature.
5. junction depletion, index, loss, capacitance, resistance, and partial ring coverage.
6. voltage-table interpolation, baseline subtraction, and endpoint clipping.
7. combined thermal and electrical resonance shifts.
8. distinct ring/heater temperatures, intrinsic loss, absorbed loss, and fixed nonabsorptive loss.
9. shifted reference phase curve and fixed physical laser wavelength.
10. wavelength dispersion, group delay, and the shared level-zero delay approximation.
11. two voltage cases at shared temperatures and separate phase/loss outputs.
12. evanescent overlap, gap, wavelength, and interaction length.
13. coherent power exchange along the coupling region.
14. local exponential wavelength model, extracted slope, and limited extrapolation.
15. rigid resonance-comb translation versus changing spacing in a fuller model.
16. optical-input assembly and handoff to optics.py and the thermal loop.

## delivery and verification

provide individual drawings, a combined atlas with named frames, SVG/PNG previews, and an offline gallery.
map drawings to functions in a concise README instead of filling the artwork with code names.
render through the official Excalidraw package with the supplied font loaded.
inspect all drawings, check measured text bounds, validate element references, and exercise gallery navigation.
label all graphs as schematic. distinguish a model assumption from device evidence.
do not modify production simulation code or existing diagram collections.
