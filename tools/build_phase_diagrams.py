#!/usr/bin/env python3
# build the equation-free phase.py companion with native Excalidraw elements.
from __future__ import annotations

import json
import math
from pathlib import Path

from build_excalidraw_atlas import Drawing, curve, linspace, scene

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/diagrams/phase"
REFERENCE = OUT / "drawings/00-supplied-ring-reference.excalidraw"
BOARDS = []


class Board(Drawing):
    def __init__(self, slug, title, code, note):
        super().__init__(slug, title, "", code)
        self.elements.clear()
        self.label(90, 65, title)
        self.note = note
        BOARDS.append(self)

    def label(self, x, y, value, center=False):
        item = self.text(x, y, value, center)
        item["customData"] = {"labelAnchorX": x, "labelCentered": center}
        return item

    def path(self, points, arrow=False, dashed=False):
        # native dashed paths keep curves editable as one element.
        x, y = points[0]
        return self.element(
            "arrow" if arrow else "line", x, y,
            max(p[0] for p in points)-min(p[0] for p in points),
            max(p[1] for p in points)-min(p[1] for p in points),
            points=[[px-x, py-y] for px, py in points],
            lastCommittedPoint=None, startBinding=None, endBinding=None,
            startArrowhead=None, endArrowhead="arrow" if arrow else None,
            strokeStyle="dashed" if dashed else "solid", elbowed=False,
        )

    def graph(self, x, y, width, height, xlabel, ylabel):
        self.arrow(x, y, x+width, y)
        self.arrow(x, y, x, y-height)
        self.label(x, y-height-85, ylabel)
        self.label(x+width/2, y+65, xlabel, True)

    def ring(self, cx, cy, r=160, junction=False):
        self.circle(cx, cy, r)
        for side in (-1, 1):
            self.circle(cx+side*r, cy, 17)
        if junction:
            # the emphasized arc is a junction-covered section, not a second ring.
            self.arc(cx, cy, r+10, -1.4, .1)["strokeWidth"] = 8

    def bracket(self, x1, x2, y, label, above=False):
        self.line(x1, y-14, x1, y+14)
        self.line(x2, y-14, x2, y+14)
        self.line(x1, y, x2, y)
        self.label((x1+x2)/2, y-65 if above else y+30, label, True)

    def pulse(self, x, y, width, center, dashed=False):
        self.line(x, y, x+width, y)
        return self.path([(x+width*t, y-100*math.exp(-((t-center)/.055)**2))
                          for t in linspace(0, 1, 100)], dashed=dashed)

    def block(self, x, y, width, height, text):
        self.shape("rectangle", x, y, width, height)
        self.label(x+width/2, y+(height-45*len(text.splitlines()))/2, text, True)


def notch(board, x, y, width, height, center, dashed=False):
    return curve(board, x, y, width, height,
                 lambda t: .87-.70/(1+((t-center)/.075)**2), dashed)


def make_boards():
    b = Board("01-reference-state", "what does the reference ring already contain?", "PhaseParams.from_plant_params",
              "reference bias is already included in the configured resonance and loss")
    b.ring(590, 680, 215, True)
    b.label(590, 350, "reference device", True)
    b.label(125, 1030, "reference ring temperature")
    b.label(125, 1100, "reference reverse bias")
    b.label(125, 1170, "negligible optical self-heating")
    b.label(1030, 430, "known reference resonance")
    b.path([(1040, 510), (940, 510), (790, 570)], arrow=True)
    b.label(1030, 680, "known intrinsic loss")
    b.path([(1040, 750), (940, 750), (800, 790)], arrow=True)
    b.label(1590, 910, "junction property", True)
    b.line(1180, 1130, 2070, 1130)
    b.dot(1320, 1130)
    b.dot(1880, 1130)
    b.label(1320, 1190, "stored baseline", True)
    b.label(1880, 1190, "current value", True)
    b.bracket(1320, 1880, 1080, "only the change is added", True)

    b = Board("02-bus-segments", "how much light reaches the next ring?", "calculate_bus_segment_power_transmission",
              "equal supplied spacing and power attenuation; chip input loss handled upstream; final segment unmodeled")
    b.label(160, 290, "chip input loss")
    b.line(180, 540, 2260, 540)
    for cx, title in [(600, "first encountered"), (1230, "next ring"), (1860, "next ring")]:
        b.ring(cx, 760, 150)
        b.line(cx, 540, cx, 590, True)
        b.label(cx, 1040, title, True)
    for x in (300, 870, 1500, 2130):
        b.arrow(x, 540, x+95, 540)
    for x1, x2 in [(670, 1160), (1300, 1790)]:
        b.bracket(x1, x2, 390, "same spacing", True)
    b.label(390, 1180, "no extra input loss", True)
    b.label(1050, 1180, "segment loss", True)
    b.label(1660, 1180, "segment loss", True)
    b.label(1940, 290, "no final-segment\nloss modeled")
    b.graph(230, 1460, 1750, 150, "distance along one segment", "power remaining")
    curve(b, 230, 1460, 1750, 135, lambda t: .95*math.exp(-1.2*t))

    b = Board("03-thermal-tuning", "how does warming move the resonance?", "calculate_thermal_resonance_shift_m",
              "quadratic wavelength tuning; the slope anchor and reference resonance temperature need not coincide")
    b.graph(230, 1030, 1850, 620, "ring temperature", "resonance wavelength")
    fn = lambda t: .12+.35*t+.42*t*t
    curve(b, 230, 1030, 1850, 620, fn)
    for t, text, ty in [(.38, "reference resonance", 1180), (.82, "warmer ring", 1180)]:
        x, y = 230+1850*t, 1030-620*fn(t)
        b.dot(x, y)
        b.line(x, y+22, x, 1050, True)
        b.label(x, ty, text, True)
    b.arrow(1600, 490, 1840, 390)
    b.label(1160, 260, "sensitivity can increase with temperature")
    anchor_x=230+1850*.08
    anchor_y=1030-620*fn(.08)
    b.circle(anchor_x,anchor_y,12)
    b.line(anchor_x,anchor_y+20,anchor_x,1050,True)
    b.label(anchor_x,1290,"slope anchor",True)
    b.label(1050,1390,"the two reference temperatures can differ")

    b = Board("04-rebase-thermal-law", "what changes when we rebase the thermal law?", "rebase_thermal_tuning_coefficients",
              "same quadratic curve and curvature; only the expansion point and coefficient representation change")
    b.graph(230, 1080, 1850, 650, "ring temperature", "resonance wavelength")
    fn = lambda t: .10+.35*t+.42*t*t
    curve(b, 230, 1080, 1850, 650, fn)
    xs=[]
    for t, label in [(.14,"original reference"),(.57,"previous temperature"),(.83,"trial temperature")]:
        x,y=230+1850*t,1080-650*fn(t)
        xs.append(x)
        b.dot(x,y)
        b.line(x,y+20,x,1090,True)
        b.label(x,1210,label,True)
    b.bracket(xs[1],xs[2],1300,"next local change")
    b.label(320,260,"one unchanged curve")
    b.label(1240,260,"slope and curvature both retained")

    b = Board("05-junction-device", "what does reverse bias change physically?", "evaluate_ring_resonance_shifts; junction_eval_row",
              "voltage tables describe carrier-dependent index, absorption, capacitance, and resistance; fill fraction scales the junction-covered length")
    for x, title, gap in [(180,"lower reverse bias",100),(1320,"higher reverse bias",250)]:
        b.label(x+420,280,title,True)
        b.shape("rectangle",x,450,840,280)
        left=x+420-gap/2;right=x+420+gap/2
        b.line(left,450,left,730,True)
        b.line(right,450,right,730,True)
        b.label(x+150,375,"p region",True)
        b.label(x+690,375,"n region",True)
        for yy in (510,600,680):
            for xx in (x+65,x+170,x+275):
                if xx<left-25:b.circle(xx,yy,9)
            for xx in (x+565,x+670,x+775):
                if xx>right+25:b.dot(xx,yy,9)
        b.bracket(left,right,820,"depleted region")
    b.label(190,1020,"more free-carrier absorption")
    b.label(1330,1020,"less free-carrier absorption")
    b.label(190,1100,"junction stores charge")
    b.label(1330,1100,"lower capacitance in this model")
    b.ring(620,1390,130,True)
    b.label(900,1290,"junction covers part of the ring")
    b.label(900,1360,"index and loss are averaged over the ring")
    b.label(900,1430,"capacitance grows with junction length")
    b.label(900,1500,"parallel conduction lowers resistance")

    b = Board("06-voltage-tables", "how does a voltage become device properties?", "junction_eval_row; PhaseParams.from_plant_params",
              "piecewise-linear interpolation; values outside the table hold the endpoint and flag clipping; the reference index and loss are subtracted downstream")
    props=[("index change",lambda t:.12+.65*math.sqrt(t)), ("junction loss",lambda t:.9-.62*math.sqrt(t)),
           ("capacitance",lambda t:.86-.65*math.sqrt(t)), ("series resistance",lambda t:.22+.5*t*t)]
    for index,(label,fn) in enumerate(props):
        x=180+(index%2)*1190;y=680+(index//2)*720
        b.graph(x,y,850,340,"reverse bias",label)
        samples=[(.12,fn(.12)),(.38,fn(.38)),(.64,fn(.64)),(.85,fn(.85))]
        b.path([(x+850*t,y-340*v) for t,v in samples])
        for t,v in samples:b.dot(x+850*t,y-340*v,8)
        b.line(x,y-340*samples[0][1],x+850*.12,y-340*samples[0][1],True)
        b.line(x+850*.85,y-340*samples[-1][1],x+850,y-340*samples[-1][1],True)
        t=.51
        interpolated_value=(fn(.38)+fn(.64))/2
        b.circle(x+850*t,y-340*interpolated_value,14)
        if index==0:
            b.label(x+280,180,"table samples")
            b.label(x+415,330,"interpolated value")
        if index==1:b.label(x+570,280,"endpoint held")
    b.label(170,1560,"stored reference bias anchors the index and loss changes")

    b = Board("07-combined-shifts", "how do heat and bias move the same resonance?", "evaluate_ring_resonance_shifts_with_workspace",
              "thermal and electrical shifts are added; the reference bias contribution is already in the original resonance")
    rows=[(460,.35,"reference state"),(870,.50,"thermal shift"),(1280,.59,"thermal and voltage shifts")]
    for y,center,label in rows:
        b.label(160,y-220,label)
        b.line(180,y,2200,y)
        notch(b,180,y,2020,170,center)
        xx=180+2020*center
        b.line(xx,y-170,xx,y+20,True)
    b.arrow(887,570,1190,570)
    b.label(1480,535,"warming moves the resonance")
    b.arrow(1190,990,1372,990)
    b.label(1500,950,"bias adds another shift")
    b.label(750,1430,"wavelength",True)
    b.label(160,1500,"illustrated bias direction matches the configured depletion model")

    b = Board("08-loss-and-heat", "which optical loss becomes heat?", "calculate_intrinsic_loss_temperature_factor; write_ring_optical_inputs_for_data_level",
              "two temperature sensitivities scale total intrinsic loss; nonabsorptive loss stays fixed; negative loss and invalid heat shares are clamped")
    b.ring(480,610,175)
    b.shape("rectangle",350,345,260,60)
    b.label(480,230,"heater-resistor temperature",True)
    b.line(480,300,480,345)
    b.label(480,920,"optical-mode temperature",True)
    b.line(480,900,480,785)
    b.label(1200,300,"separate temperature sensitivities")
    b.label(1200,400,"same multiplier for both data levels")
    b.label(1200,500,"base G26 coefficients are zero")
    for y,absorbed,label in [(990,500,"reference loss"),(1280,770,"illustrative increased loss")]:
        b.label(1220,y-95,label)
        b.shape("rectangle",1200,y,absorbed,100)
        b.shape("rectangle",1200+absorbed,y,210,100)
        for xx in range(1208,1200+absorbed,24):b.line(xx,y+8,xx,y+92)
    b.label(1320,1470,"absorbed loss")
    b.label(1880,1470,"nonabsorptive loss")
    b.label(1220,1540,"extra loss is assigned to absorption")
    b.label(150,1130,"field retention follows total loss")
    b.label(150,1220,"heat fraction follows the loss budget")
    b.label(150,1400,"scattering and radiation do not heat\nthe modeled ring through this budget")

    b = Board("09-shifted-phase-curve", "where does a fixed laser sit on the shifted phase curve?", "write_ring_optical_inputs_for_data_level",
              "translate the reference phase curve; the physical laser wavelength stays fixed; reference phase already includes coupler phase")
    b.graph(220,1050,1900,640,"wavelength","round-trip phase")
    fn=lambda t:.88-.66*t+.10*t*t
    curve(b,220,1050,1900,640,fn,True)
    curve(b,220,1050,1900,640,lambda t:fn(t-.22))
    laser=.70;lookup=.48
    for t,text in [(lookup,"reference lookup"),(laser,"actual laser")]:
        xx=220+1900*t
        b.line(xx,1060,xx,500,True)
        b.label(xx,1170,text,True)
    y=1050-640*fn(lookup)
    b.dot(220+1900*lookup,y)
    b.dot(220+1900*laser,y)
    b.arrow(220+1900*lookup+25,y,220+1900*laser-25,y)
    b.label(430,250,"reference curve - dashed")
    b.label(1330,300,"shifted curve - solid")
    b.bracket(220+1900*lookup,220+1900*laser,1320,"resonance shift")
    b.label(220,1480,"same phase value at two different curve coordinates")

    b = Board("10-wavelength-dispersion", "what wavelength dependence does the reference curve retain?", "write_ring_optical_inputs_for_data_level",
              "quadratic effective-index expansion and linear group-index expansion about the reference wavelength; these are local wavelength approximations, not time integration")
    b.graph(160,1040,900,620,"reference-curve wavelength","effective index")
    b.graph(1340,1040,900,620,"reference-curve wavelength","group index")
    curve(b,160,1040,900,620,lambda t:.85-.36*t-.15*t*t)
    curve(b,1340,1040,900,620,lambda t:.33+.30*t)
    for x in (160,1340):
        b.line(x+360,1050,x+360,470,True)
        b.label(x+360,1200,"reference wavelength",True)
    b.label(220,280,"slope and curvature retained")
    b.label(1430,280,"linear wavelength trend retained")
    b.label(230,1380,"phase follows optical path length")
    b.label(1350,1380,"group index sets propagation delay")
    b.label(180,1510,"the two expansions are consistent locally, not exact global dispersion curves")

    b = Board("11-shared-delay", "what does sharing the level-zero delay mean?", "evaluate_binary_ring_optical_inputs_with_workspace",
              "the two voltage cases can sample different group indices; only the level-zero group delay is retained for both")
    for x,title in [(180,"separate delay estimates"),(1340,"current implementation")]:
        b.label(x+430,290,title,True)
        for row in (0,1):
            y=650+row*460
            b.label(x,y-230,"data level zero" if row==0 else "data level one")
            center=.45 if x>1000 or row==0 else .65
            b.pulse(x,y,850,center)
        b.arrow(x,1270,x+850,1270)
        b.label(x+425,1350,"arrival time",True)
    b.label(180,1480,"voltage can change group velocity")
    b.label(1340,1480,"level-zero delay reused")
    b.label(180,1570,"delay difference is exaggerated here for visibility")

    b = Board("12-two-data-levels", "which inputs differ between zero and one?", "evaluate_binary_ring_optical_inputs_with_workspace",
              "same instantaneous temperatures; separate voltage-dependent outputs; probability weighting occurs later in optics.py")
    b.label(1170,270,"shared ring and heater temperatures",True)
    b.line(600,400,1760,400)
    b.arrow(600,400,600,540)
    b.arrow(1760,400,1760,540)
    for x,title in [(600,"level-zero reverse bias"),(1760,"level-one reverse bias")]:
        b.label(x,600,title,True)
        b.ring(x,950,160,True)
        b.label(x,1200,"phase and resonance shift",True)
        b.label(x,1270,"field retention and heat fraction",True)
        b.label(x,1340,"capacitance and resistance",True)
    b.label(1170,1490,"one common round-trip delay from level zero",True)
    b.label(1170,1580,"the field solver and bit weighting come next",True)

    b = Board("13-evanescent-overlap", "why can wavelength change coupling?", "evaluate_coupler_power_fractions; offline coupling extraction",
              "evanescent overlap motivates wavelength dependence; this picture holds geometry and index factors fixed to isolate the tail mechanism")
    for x,title,spread in [(170,"shorter wavelength",75),(1350,"longer wavelength",150)]:
        b.label(x+420,220,title,True)
        b.shape("rectangle",x+100,430,140,780)
        b.shape("rectangle",x+580,430,140,780)
        b.label(x+170,1320,"bus",True)
        b.label(x+650,1320,"neighbor",True)
        b.bracket(x+240,x+580,350,"same gap",True)
        for dy in (0,160,320):
            b.line(x+240,600+dy,x+580,600+dy,True)
            pts=[(x+240+t*340,600+dy-100*math.exp(-t*340/spread)) for t in linspace(0,1,60)]
            b.path(pts)
        b.arrow(x+170,1180,x+170,970)
        b.label(x+80,1480,"field decays across the gap")
        if x>1000:b.label(x+50,1570,"a longer tail can increase overlap")

    b = Board("14-coherent-power-exchange", "how does power transfer along a coupler?", "textbook chapter 4; offline coupling extraction",
              "ideal phase-matched lossless uniform coupler; power exchanges sinusoidally with distance and can return to the launch guide")
    b.graph(200,1060,1910,650,"distance along the coupling region","power fraction")
    curve(b,200,1060,1910,650,lambda t:math.cos(math.pi*t)**2)
    curve(b,200,1060,1910,650,lambda t:math.sin(math.pi*t)**2,True)
    b.label(250,270,"launch guide - solid")
    b.label(1420,270,"neighbor guide - dashed")
    b.line(200+1910*.22,1080,200+1910*.22,430,True)
    b.label(200+1910*.22,1190,"coupler ends here",True)
    b.label(350,1400,"only the fraction at the physical exit is used")
    b.label(350,1490,"spatial power exchange is not exponential wavelength growth")

    b = Board("15-local-coupling-model", "why does runtime coupling use a local exponential?", "evaluate_coupler_power_fractions; extraction.coupler_block",
              "first-order approximation to logarithmic wavelength dependence; matches reference value and slope, but omits changing relative slope")
    b.graph(220,1080,1900,700,"actual laser wavelength","coupled power fraction")
    fn=lambda t:.12+.38*math.sin(1.2*t+.15)**2
    ref=.4;c0=fn(ref);slope=.38*1.2*math.sin(2*(1.2*ref+.15))/c0
    curve(b,220,1080,1900,700,lambda t:fn(t)/.8)
    curve(b,220,1080,1900,700,lambda t:c0*math.exp(slope*(t-ref))/.8,True)
    xr=220+1900*ref
    b.line(xr,1100,xr,390,True)
    b.dot(xr,1080-700*c0/.8)
    b.label(xr,1240,"reference wavelength",True)
    b.label(220,210,"same value and local slope")
    b.label(1510,360,"runtime model - dashed")
    b.label(1460,850,"detailed trend - solid")
    b.bracket(xr-140,xr+140,1340,"local agreement")
    b.label(180,1490,"no direct temperature or bias dependence of coupling")
    b.label(180,1570,"the physical laser wavelength is used, not the phase lookup coordinate")

    b = Board("16-rigid-resonance-comb", "what does sliding every resonance together miss?", "write_ring_optical_inputs_for_data_level",
              "all orders receive one wavelength shift; a fuller thermal-dispersion model also changes spacing; anchoring one order does not verify the others")
    rows=[(490,"reference resonance comb",[.20,.43,.66,.89]),
          (890,"runtime translation",[.26,.49,.72,.95]),
          (1290,"fuller temperature-dependent spacing",[.235,.49,.745,.98])]
    for y,title,peaks in rows:
        b.label(160,y-220,title)
        b.arrow(180,y,2200,y)
        for i,t in enumerate(peaks):
            x=180+1900*t
            b.line(x,y,x,y-150)
            if i==1:b.circle(x,y-170,15)
    b.line(180+1900*.49,640,180+1900*.49,1320,True)
    b.label(1500,620,"same anchored order")
    b.label(1450,1050,"other orders can move apart")
    b.label(190,1490,"the selected resonance can agree while neighboring resonances differ")

    b = Board("17-optical-input-handoff", "how do the phase.py calculations connect?", "evaluate_ring_optical_inputs",
              "this is an input-assembly map, separate from the physical schematics; time advancement belongs to thermal.py and selfheat.py")
    b.block(130,300,670,170,"ring and heater\ntemperatures")
    b.block(130,650,670,170,"two junction voltages")
    b.block(130,1000,670,170,"actual laser wavelengths")
    b.block(1120,300,1060,170,"thermal and electrical\nresonance shifts")
    b.block(1120,650,1060,170,"two phase and loss cases\none shared group delay")
    b.block(1120,1000,1060,170,"coupling at every\nlaser wavelength")
    b.arrow(800,385,1120,385)
    b.path([(800,735),(950,735),(950,460),(1120,460)],arrow=True)
    b.arrow(1650,470,1650,650)
    b.arrow(800,1085,1120,1085)
    b.path([(800,1125),(990,1125),(990,780),(1120,780)],arrow=True)
    b.label(180,1340,"reference parameters and tables supply each stage")
    b.path([(2180,735),(2300,735),(2300,1250),(1650,1250),(1650,1310)],arrow=True)
    b.line(2180,1085,2300,1085)
    b.label(1650,1360,"optics.py",True)
    b.label(1650,1430,"fields, monitor power, absorbed heat",True)
    b.label(1650,1540,"thermal update closes the feedback loop",True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"drawings").mkdir(exist_ok=True)
    make_boards()
    original=json.loads(REFERENCE.read_text())
    reference=scene(original["elements"])
    reference["files"]=original.get("files",{})
    reference["source"]="supplied ring reference - original elements preserved"
    (OUT/"drawings/00-supplied-ring-reference.excalidraw").write_text(json.dumps(reference,indent=2)+"\n")
    manifest=[dict(slug="00-supplied-ring-reference",title="your ring reference",code="optics.py - supplied style reference",subtitle="original supplied elements, unchanged",reference=True)]
    for b in BOARDS:
        data=scene(b.elements)
        data["source"]="mrm-model phase.py visual companion"
        (OUT/"drawings"/f"{b.slug}.excalidraw").write_text(json.dumps(data,indent=2)+"\n")
        manifest.append(dict(slug=b.slug,title=b.title,code=b.code,subtitle=b.note))
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (OUT/"manifest.js").write_text("window.PHASE_DIAGRAMS = "+json.dumps(manifest,indent=2)+";\n")
    print(f"wrote {len(manifest)} drawings to {OUT}")


if __name__=="__main__":
    main()
