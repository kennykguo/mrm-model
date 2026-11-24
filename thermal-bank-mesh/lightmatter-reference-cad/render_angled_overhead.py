"""Render a slightly tilted overhead cutaway from the STEP assembly."""
from pathlib import Path
import cadquery as cq
import json

OUT = Path(__file__).resolve().parent
colors = json.loads((OUT / "photonics-palette.json").read_text())["display_colors"]
assembly = cq.Assembly.load(str(OUT / "eight-ring-photonics-colors.step"))

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

for shape, name, location, color in assembly:
    if not name.rsplit("/", 1)[-1].startswith(("sub", "box", "slab", "rib", "heater", "via", "m1_", "bus")):
        continue
    rgba = color.toTuple() if color else colors["si_slab"]
    clipped = shape.located(location).intersect(section)
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
camera.SetPosition(175, -373, 440)
camera.SetFocalPoint(175, 27, 0)
camera.SetViewUp(0, 0, 1)
camera.ParallelProjectionOn()
camera.SetParallelScale(90)
scene.ResetCameraClippingRange()
window.Render()
capture = vtk.vtkWindowToImageFilter()
capture.SetInput(window)
capture.ReadFrontBufferOff()
capture.Update()
writer = vtk.vtkPNGWriter()
writer.SetFileName(str(OUT / "eight-ring-angled-overhead.png"))
writer.SetInputConnection(capture.GetOutputPort())
writer.Write()
window.Finalize()
print(OUT / "eight-ring-angled-overhead.png")
