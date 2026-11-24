"""Render the solved thermal field and original device mesh."""
from pathlib import Path
import json

import matplotlib
import numpy as np
import vtk

ROOT = Path(__file__).resolve().parent
summary = json.loads((ROOT / "solve-summary.json").read_text())
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(str(ROOT / "heater-4-steady-temperature.vtu"))
reader.Update()
grid = reader.GetOutput()
grid.GetPointData().SetActiveScalars("temperature_rise_K")
names = summary["region_names"]
background = (0.035, 0.045, 0.065)
foreground = (0.90, 0.93, 0.97)
muted = (0.57, 0.65, 0.74)
blue = (0.34, 0.74, 1.0)
gold = (1.0, 0.77, 0.36)

lookup = vtk.vtkLookupTable()
lookup.SetNumberOfTableValues(256)
lookup.SetTableRange(0.003, summary["peak_temperature_rise_K"])
lookup.SetScaleToLog10()
cmap = matplotlib.colormaps["inferno"]
for i in range(256):
    rgb = np.array(cmap(i / 255)[:3])
    rgb = 0.86 * rgb + 0.14 * np.array((0.07, 0.15, 0.25))
    lookup.SetTableValue(i, *rgb, 1)
lookup.Build()

def crop(data, bounds):
    result = data
    for axis in range(3):
        for side, direction in ((0, 1), (1, -1)):
            origin = [0., 0., 0.]
            normal = [0., 0., 0.]
            origin[axis] = bounds[2 * axis + side]
            normal[axis] = direction
            plane = vtk.vtkPlane()
            plane.SetOrigin(*origin)
            plane.SetNormal(*normal)
            clip = vtk.vtkClipDataSet()
            clip.SetInputData(result)
            clip.SetClipFunction(plane)
            clip.Update()
            result = clip.GetOutput()
    return result

def region(prefixes):
    indices = [i for i, name in enumerate(names) if name.startswith(prefixes)]
    mask = vtk.vtkIdList()
    values = grid.GetCellData().GetArray("region_index")
    for i in range(grid.GetNumberOfCells()):
        if int(values.GetTuple1(i)) in indices:
            mask.InsertNextId(i)
    extract = vtk.vtkExtractCells()
    extract.SetInputData(grid)
    extract.SetCellList(mask)
    extract.Update()
    return extract.GetOutput()

def actor(data, color=None, opacity=1, edges=False):
    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputData(data)
    surface.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface.GetOutput())
    if color:
        mapper.ScalarVisibilityOff()
    else:
        mapper.SetLookupTable(lookup)
        mapper.SetScalarRange(lookup.GetTableRange())
        mapper.InterpolateScalarsBeforeMappingOn()
    result = vtk.vtkActor()
    result.SetMapper(mapper)
    prop = result.GetProperty()
    if color:
        prop.SetColor(*color)
        prop.SetAmbient(0.42)
        prop.SetDiffuse(0.58)
        prop.SetSpecular(0.28)
        prop.SetSpecularPower(40)
    else:
        prop.LightingOff()
    prop.SetOpacity(opacity)
    if edges:
        prop.EdgeVisibilityOn()
        prop.SetEdgeColor(0.19, 0.25, 0.31)
        prop.SetLineWidth(0.4)
    return result

def text(renderer, content, x, y, size, color=foreground):
    label = vtk.vtkTextActor()
    label.SetInput(content)
    label.SetPosition(x, y)
    prop = label.GetTextProperty()
    prop.SetFontFamily(vtk.VTK_FONT_FILE)
    prop.SetFontFile("/System/Library/Fonts/SFNSMono.ttf")
    prop.SetFontSize(size)
    prop.SetColor(*color)
    overlay.AddViewProp(label)
    return label

window = vtk.vtkRenderWindow()
window.SetSize(3200, 2000)
window.SetOffScreenRendering(1)
window.SetMultiSamples(8)
window.SetNumberOfLayers(3)
base = vtk.vtkRenderer()
base.SetBackground(*background)
window.AddRenderer(base)
scene = vtk.vtkRenderer()
scene.SetLayer(1)
scene.SetBackground(*background)
scene.SetViewport(0, 0.16, 1, 0.86)
window.AddRenderer(scene)
overlay = vtk.vtkRenderer()
overlay.SetLayer(2)
overlay.SetViewport(0, 0, 1, 1)
overlay.InteractiveOff()
window.AddRenderer(overlay)

# Cut through the oxide below the device, retaining actual dimensions.
bounds = (72, 228, -43, 79, -24, 0)
thermal = crop(grid, bounds)
scene.AddActor(actor(thermal))
nearby = (72, 228, -43, 79, -0.001, 6)
silicon = crop(region(("rib", "slab", "bus")), nearby)
heaters = crop(region(("heater",)), nearby)
metal = crop(region(("via", "m1_")), nearby)
scene.AddActor(actor(silicon, blue))
scene.AddActor(actor(heaters, gold))
scene.AddActor(actor(metal, (0.69, 0.75, 0.83)))

camera = scene.GetActiveCamera()
camera.SetPosition(236, -183, 166)
camera.SetFocalPoint(150, 13, -5)
camera.SetViewUp(0, 0, 1)
camera.ParallelProjectionOn()
camera.SetParallelScale(76)
scene.ResetCameraClippingRange()

text(scene, "Heat spreading in an eight-ring modulator", 116, 1850, 47)
text(scene, "Full 3-D FEM  /  steady state  /  heater 4 at 1 mW", 119, 1788, 25, muted)
text(scene, "TEMPERATURE RISE ABOVE THE FIXED HEAT SINK", 119, 162, 20, muted)

bar = vtk.vtkScalarBarActor()
bar.SetLookupTable(lookup)
bar.SetOrientationToHorizontal()
bar.SetPosition(0.038, 0.041)
bar.SetWidth(0.34)
bar.SetHeight(0.035)
ticks = vtk.vtkDoubleArray()
for value in (0.003, 0.01, 0.1, 1, summary["peak_temperature_rise_K"]):
    ticks.InsertNextValue(value)
bar.SetCustomLabels(ticks)
bar.UseCustomLabelsOn()
bar.SetLabelFormat("%.3g")
bar.SetTextPositionToSucceedScalarBar()
bar.SetUnconstrainedFontSize(True)
bar.GetLabelTextProperty().SetFontFamily(vtk.VTK_FONT_FILE)
bar.GetLabelTextProperty().SetFontFile("/System/Library/Fonts/SFNSMono.ttf")
bar.GetLabelTextProperty().SetFontSize(22)
bar.GetLabelTextProperty().SetColor(*foreground)
bar.GetLabelTextProperty().ShadowOff()
overlay.AddViewProp(bar)
text(scene, "K  /  logarithmic color scale", 122, 32, 19, muted)

text(scene, "Silicon waveguides", 2220, 270, 23, blue)
text(scene, "Heaters", 2220, 224, 23, gold)
text(scene, "Electrical contacts and leads", 2220, 178, 23, (0.69, 0.75, 0.83))
text(scene, "Cutaway through the oxide below the rings", 119, 280, 23, foreground)
text(scene, "True dimensions  /  all other heat inputs zero", 119, 236, 20, muted)
text(scene, "110,832 nodes  /  634,996 tetrahedra", 2220, 80, 19, muted)

# Independent top-view inset locates the heated ring in the complete bank.
overview = vtk.vtkRenderer()
overview.SetViewport(0.61, 0.73, 0.965, 0.86)
overview.SetLayer(2)
overview.SetBackground(*background)
window.AddRenderer(overview)
plane = vtk.vtkPlane()
plane.SetOrigin(0, 0, 0)
plane.SetNormal(0, 0, 1)
cut = vtk.vtkCutter()
cut.SetInputData(grid)
cut.SetCutFunction(plane)
cut.Update()
inset_field = crop(cut.GetOutput(), (-17, 367, -20, 20, -1, 1))
overview.AddActor(actor(inset_field))
overview.AddActor(actor(crop(region(("rib", "bus")), (-17, 367, -20, 20, -1, 1)), blue))
overview.AddActor(actor(crop(region(("heater3_",)), (-17, 367, -20, 20, -1, 1)), gold))
ocam = overview.GetActiveCamera()
ocam.SetPosition(175, 0, 500)
ocam.SetFocalPoint(175, 0, 0)
ocam.SetViewUp(0, 1, 0)
ocam.ParallelProjectionOn()
ocam.SetParallelScale(48)
overview.ResetCameraClippingRange()
text(scene, "Eight-ring bank  /  heated ring highlighted", 1985, 1760, 19, muted)

window.Render()
capture = vtk.vtkWindowToImageFilter()
capture.SetInput(window)
capture.ReadFrontBufferOff()
capture.Update()
writer = vtk.vtkPNGWriter()
writer.SetFileName(str(ROOT / "thermal-fem.png"))
writer.SetInputConnection(capture.GetOutputPort())
writer.Write()
print(ROOT / "thermal-fem.png")
window.Finalize()
