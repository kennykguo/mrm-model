"""Render the bank on a laterally cropped substrate with upper layers hidden."""
from pathlib import Path

import vtk

ROOT = Path(__file__).resolve().parent
PARTS = ROOT.parent / "stl-parts"
window = vtk.vtkRenderWindow()
window.SetSize(3840, 2160)
window.SetOffScreenRendering(1)
window.SetMultiSamples(8)
scene = vtk.vtkRenderer()
scene.SetBackground(0.045, 0.055, 0.075)
scene.UseFXAAOn()
window.AddRenderer(scene)

def add_part(filename, color, metallic=0.15, roughness=0.35):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(PARTS / filename))
    reader.Update()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.SetFeatureAngle(45)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetInterpolationToPhong()
    prop.SetAmbient(0.35)
    prop.SetDiffuse(0.65)
    prop.SetSpecular(0.25)
    prop.SetSpecularPower(35)
    scene.AddActor(actor)

add_part("08-silicon-waveguides-and-slabs.stl", (0.23, 0.63, 0.87), 0.12, 0.36)
add_part("09-heaters.stl", (0.98, 0.63, 0.22), 0.48, 0.32)
add_part("10-vias.stl", (0.71, 0.53, 0.90), 0.38, 0.32)
add_part("11-metal-leads.stl", (0.77, 0.82, 0.89), 0.62, 0.3)

# Keep the original device coordinates and crop only the empty bus extensions.
plane_left = vtk.vtkPlane()
plane_left.SetOrigin(-16, 0, 0)
plane_left.SetNormal(1, 0, 0)
plane_right = vtk.vtkPlane()
plane_right.SetOrigin(366, 0, 0)
plane_right.SetNormal(-1, 0, 0)
for actor in scene.GetActors():
    actor.GetMapper().AddClippingPlane(plane_left)
    actor.GetMapper().AddClippingPlane(plane_right)


# Matching dimensions from the bank FEM, with lateral margins cropped for visibility.
for bounds, color in [
    ((-16, 366, -22, 85, -202, -2), (0.22, 0.28, 0.36)),
    ((-16, 366, -22, 85, -2, 0), (0.30, 0.38, 0.48)),
]:
    cube = vtk.vtkCubeSource()
    cube.SetBounds(*bounds)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cube.GetOutputPort())
    support = vtk.vtkActor()
    support.SetMapper(mapper)
    support.GetProperty().SetColor(*color)
    support.GetProperty().SetAmbient(0.3)
    support.GetProperty().SetDiffuse(0.7)
    scene.AddActor(support)

lights = vtk.vtkLightKit()
lights.SetKeyLightIntensity(1.2)
lights.SetKeyToFillRatio(1.6)
lights.SetKeyToHeadRatio(2.0)
lights.AddLightsToRenderer(scene)

camera = scene.GetActiveCamera()
camera.SetPosition(320, -420, 320)
camera.SetFocalPoint(175, 25, -92)
camera.SetViewUp(0, 0, 1)
camera.ParallelProjectionOn()
camera.SetParallelScale(190)
scene.ResetCameraClippingRange()
window.Render()
capture = vtk.vtkWindowToImageFilter()
capture.SetInput(window)
capture.ReadFrontBufferOff()
capture.Update()
writer = vtk.vtkPNGWriter()
writer.SetFileName(str(ROOT / "eight-ring-bank-substrate.png"))
writer.SetInputConnection(capture.GetOutputPort())
writer.Write()
window.Finalize()
print(ROOT / "eight-ring-bank-substrate.png")
