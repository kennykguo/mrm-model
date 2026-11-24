"""Apply component colors to the existing CAD assembly and render an angled overhead view."""
from pathlib import Path
import json

import cadquery as cq

OUT = Path(__file__).resolve().parent
original = cq.Assembly.load(str(OUT / "eight-ring-thermal-assembly.step"))
colors = json.loads((OUT / "model-details.json").read_text())["display_colors"]
colors.update({
    "si_sub": (0.17, 0.19, 0.22, 1),
    "box": (0.34, 0.38, 0.43, 1),
    "si_slab": (0.20, 0.29, 0.36, 1),
    "si_film": (0.30, 0.69, 0.85, 1),
    "si_heater": (0.91, 0.30, 0.24, 1),
    "cu": (0.76, 0.80, 0.84, 1),
})
assembly = cq.Assembly(name="Eight_ring_thermal_model")
original_volume = 0.0
for shape, name, location, _ in original:
    component = name.rsplit("/", 1)[-1]
    if component == "sub":
        material = "si_sub"
    elif component == "box":
        material = "box"
    elif component == "clad":
        material = "clad"
    elif component == "pass":
        material = "pass"
    elif component.startswith("ko_lo"):
        material = "beol_lo_ko"
    elif component.startswith("ko_hi"):
        material = "beol_hi_ko"
    elif component.startswith("slab"):
        material = "si_slab"
    elif component.startswith(("rib", "bus")):
        material = "si_film"
    elif component.startswith("heater"):
        material = "si_heater"
    elif component.startswith(("via", "m1_")):
        material = "cu"
    else:
        material = component
    original_volume += shape.Volume()
    assembly.add(shape, name=component, loc=location, color=cq.Color(*colors[material]))

output = OUT / "eight-ring-photonics-colors.step"
assembly.export(str(output), exportType="STEP")
reopened = cq.Assembly.load(str(output))
assert len(list(reopened)) == 79
assert abs(sum(shape.Volume() for shape, *_ in reopened) / original_volume - 1) < 1e-10
for _, name, _, color in reopened:
    if name.rsplit("/", 1)[-1].startswith("heater"):
        assert color is not None
        assert color.toTuple()[0] > 0.85 and color.toTuple()[1] < 0.4

(OUT / "photonics-palette.json").write_text(json.dumps({"display_colors": colors}, indent=2) + "\n")
print(f"Verified {output}", flush=True)
