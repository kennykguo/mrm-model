"""Export the existing eight-ring thermal geometry as a colored STEP assembly."""
from pathlib import Path
import json

import cadquery as cq
from mrm.extraction.thermal_fem import bank_geometry3d, default_config

OUT = Path(__file__).resolve().parent
config = default_config("inring_noundercut", 160.0, bottom_bc="dirichlet", sub_um=200.0)
centres = [(config.pitch_um * index, 0.0) for index in range(8)]
regions, _ = bank_geometry3d(config, centres, margin_um=400.0, leads=True, bus=True)

# These are display colors. Material coefficients remain in the thermal model.
colors = {
    "si_sub": (0.20, 0.22, 0.25, 1),
    "box": (0.61, 0.58, 0.49, 1),
    "si_slab": (0.39, 0.37, 0.32, 1),
    "si_film": (0.66, 0.61, 0.46, 1),
    "si_heater": (0.80, 0.57, 0.23, 1),
    "cu": (0.74, 0.73, 0.68, 1),
    "clad": (0.77, 0.74, 0.65, 0.08),
    "beol_lo": (0.62, 0.60, 0.55, 0.08),
    "beol_hi": (0.67, 0.65, 0.59, 0.08),
    "beol_lo_ko": (0.75, 0.73, 0.66, 0.04),
    "beol_hi_ko": (0.75, 0.73, 0.66, 0.04),
    "pass": (0.83, 0.81, 0.75, 0.06),
}

def primitive(region):
    if region.kind == "box":
        x0, x1, y0, y1, z0, z1 = region.par
        return cq.Solid.makeBox(x1-x0, y1-y0, z1-z0, cq.Vector(x0, y0, z0))
    assert region.kind == "annulus"
    cx, cy, inner, outer, z0, z1 = region.par
    location = cq.Vector(cx, cy, z0)
    result = cq.Solid.makeCylinder(outer, z1-z0, location)
    if inner:
        result = result.cut(cq.Solid.makeCylinder(inner, z1-z0, location))
    return result

def overlaps(a, b):
    aa, bb = a.BoundingBox(), b.BoundingBox()
    return all(max(getattr(aa, axis+"min"), getattr(bb, axis+"min")) <
               min(getattr(aa, axis+"max"), getattr(bb, axis+"max"))
               for axis in "xyz")

primitives = [primitive(region) for region in regions]
assembly = cq.Assembly(name="Eight_ring_thermal_model")
bodies = []
total_volume_um3 = 0.0
for index, (region, shape) in enumerate(zip(regions, primitives)):
    cutters = [
        other for other_index, (other_region, other) in enumerate(zip(regions, primitives))
        if (other_region.priority > region.priority or
            (other_region.priority == region.priority and other_index > index))
        and overlaps(shape, other)
    ]
    solid = shape.cut(*cutters) if cutters else shape
    assert solid.isValid() and solid.Volume() > 0, region.name
    total_volume_um3 += solid.Volume()
    # CAD coordinates and STEP length units are millimetres.
    solid = solid.scale(0.001)
    rgba = colors[region.material]
    assembly.add(solid, name=region.name, color=cq.Color(*rgba))
    bodies.append((region, solid, rgba))
    print(f"Built {region.name}", flush=True)

# The material regions must partition the original simulation box.
expected_volume_um3 = 1150 * 800 * 212
assert abs(total_volume_um3 / expected_volume_um3 - 1) < 1e-9
path = OUT / "eight-ring-thermal-assembly.step"
assembly.export(str(path), exportType="STEP")
loaded = cq.importers.importStep(str(path)).val()
assert loaded.isValid()
assert abs(loaded.Volume() / (expected_volume_um3 * 1e-9) - 1) < 1e-8
bounds = loaded.BoundingBox()
assert abs(bounds.xlen - 1.15) < 1e-6
assert abs(bounds.ylen - .8) < 1e-6
assert abs(bounds.zlen - .212) < 1e-6

record = {
    "model": "Existing eight-ring FEM bank geometry",
    "source_function": "mrm.extraction.thermal_fem.bank_geometry3d",
    "configuration": config.to_dict(),
    "cad_units": "mm",
    "bounding_dimensions_mm": [bounds.xlen, bounds.ylen, bounds.zlen],
    "components": len(bodies),
    "display_colors": colors,
    "appearance_reference": "https://lightmatter.co/wp-content/uploads/2025/04/OFC-2025-submission.pdf",
    "appearance_note": "Neutral silicon and warm metal palette inspired by the public chip photo. Display colors are illustrative.",
    "geometry_note": "Uses our simulation assumptions. This is not a manufacturer-provided CAD model.",
    "verification": "Every body valid, total region volume matches the full simulation box, STEP reimport and dimensions verified.",
}
(OUT / "model-details.json").write_text(json.dumps(record, indent=2)+"\n")
print(f"Verified {path}", flush=True)

# A sectioned rendering shows the ring bank without hiding it behind the covering layers.
# All CAD bodies retain their full simulated dimensions in the STEP assembly.
import vtk

window = vtk.vtkRenderWindow()
window.SetSize(3840, 1600)
window.SetOffScreenRendering(1)
window.SetMultiSamples(8)
scene = vtk.vtkRenderer()
scene.SetBackground(0.065, 0.070, 0.080)
scene.UseFXAAOn()
window.AddRenderer(scene)
view_bounds = (-16, 366, -20, 80, -4, 4)
section = cq.Solid.makeBox(
    view_bounds[1]-view_bounds[0], view_bounds[3]-view_bounds[2],
    view_bounds[5]-view_bounds[4], cq.Vector(view_bounds[0], view_bounds[2], view_bounds[4])
).scale(0.001)

for region, shape, rgba in bodies:
    if region.material not in ("si_sub", "box", "si_slab", "si_film", "si_heater", "cu"):
        continue
    clipped = shape.intersect(section)
    if clipped.Volume() < 1e-18:
        continue
    vertices, faces = clipped.scale(1000).tessellate(.02, .12)
    points = vtk.vtkPoints()
    for vertex in vertices:
        points.InsertNextPoint(vertex.x, vertex.y, vertex.z)
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(3, face)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.SetFeatureAngle(35)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*rgba[:3])
    prop.SetAmbient(.28)
    prop.SetDiffuse(.65)
    prop.SetSpecular(.25)
    prop.SetSpecularPower(40)
    scene.AddActor(actor)

light = vtk.vtkLightKit()
light.SetKeyLightIntensity(.85)
light.SetKeyToFillRatio(2)
light.AddLightsToRenderer(scene)
camera = scene.GetActiveCamera()
camera.SetPosition(214, -215, 305)
camera.SetFocalPoint(175, 27, 0)
camera.SetViewUp(0, 0, 1)
camera.ParallelProjectionOn()
camera.SetParallelScale(92)
scene.ResetCameraClippingRange()
window.Render()
capture = vtk.vtkWindowToImageFilter()
capture.SetInput(window)
capture.ReadFrontBufferOff()
capture.Update()
writer = vtk.vtkPNGWriter()
writer.SetFileName(str(OUT / "eight-ring-cad-cutaway.png"))
writer.SetInputConnection(capture.GetOutputPort())
writer.Write()
window.Finalize()
print("Saved CAD cutaway preview.", flush=True)
