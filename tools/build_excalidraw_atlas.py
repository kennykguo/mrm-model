#!/usr/bin/env python3
# build editable, equation-free diagrams for the optics walkthrough.
# each drawing uses the supplied reference's font, stroke, and monochrome style.
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/diagrams/optics"
REFERENCE = Path("/Users/kenny/.codex/attachments/77f624fa-6c9e-4144-9923-0a5f3e2b826b/Pasted text.txt")
INK = "#1e1e1e"
W, H = 2400, 1500
BOARDS = []


class Drawing:
    def __init__(self, slug, title, subtitle, code):
        self.slug, self.title, self.subtitle, self.code = slug, title, subtitle, code
        self.elements = []
        self.text(80, 65, title)
        self.text(80, 135, subtitle)
        BOARDS.append(self)

    def element(self, kind, x, y, width, height, **extra):
        ordinal = len(self.elements)
        ident = f"{self.slug}-{ordinal:03d}"
        digest = int(hashlib.sha256(ident.encode()).hexdigest()[:7], 16)
        element = dict(
            id=ident, type=kind, x=x, y=y, width=width, height=height,
            angle=0, strokeColor=INK, backgroundColor="transparent", fillStyle="solid",
            strokeWidth=4, strokeStyle="solid", roughness=1, opacity=100,
            groupIds=[], frameId=None, roundness=None, seed=digest,
            version=1, versionNonce=digest, isDeleted=False, boundElements=[],
            updated=1793188800000, link=None, locked=False,
        )
        element.update(extra)
        self.elements.append(element)
        return element

    def text(self, x, y, text, center=False):
        # the renderer replaces estimated dimensions with measured font metrics.
        lines = text.splitlines() or [""]
        width = max(len(s) for s in lines) * 21.0
        return self.element("text", x-width/2 if center else x, y, width, 45*len(lines),
                            text=text, originalText=text, fontSize=36, fontFamily=5,
                            textAlign="center" if center else "left", verticalAlign="top",
                            containerId=None, autoResize=True, lineHeight=1.25)

    def shape(self, kind, x, y, width, height, dashed=False):
        return self.element(kind, x, y, width, height,
                            strokeStyle="dashed" if dashed else "solid")

    def box(self, x, y, width, height, label):
        self.shape("rectangle", x, y, width, height)
        self.text(x+width/2, y+(height-45*len(label.splitlines()))/2, label, True)

    def path(self, points, arrow=False, dashed=False):
        if dashed and len(points)>2 and not arrow:
            # use visible dash segments for curved paths in native SVG exports.
            group=f"{self.slug}-dash-group-{len(self.elements)}"
            remaining=28.0
            visible=True
            segment=[points[0]]
            result=None
            def emit(chunk):
                if len(chunk)>1:
                    item=self.path(chunk)
                    item["groupIds"]=[group]
                    return item
            for a,b in zip(points,points[1:]):
                start=a
                distance=math.dist(start,b)
                while distance>=remaining and distance>0:
                    f=remaining/distance
                    stop=(start[0]+f*(b[0]-start[0]),start[1]+f*(b[1]-start[1]))
                    if visible:
                        segment.append(stop)
                        result=emit(segment)
                    visible=not visible
                    remaining=28.0 if visible else 20.0
                    segment=[stop] if visible else []
                    start=stop
                    distance=math.dist(start,b)
                if visible:segment.append(b)
                remaining-=distance
            if visible:result=emit(segment) or result
            return result
        x, y = points[0]
        rel = [[px-x, py-y] for px, py in points]
        width = max(px for px, _ in points)-min(px for px, _ in points)
        height = max(py for _, py in points)-min(py for _, py in points)
        return self.element("arrow" if arrow else "line", x, y, width, height,
                            points=rel, lastCommittedPoint=None,
                            startBinding=None, endBinding=None, startArrowhead=None,
                            endArrowhead="arrow" if arrow else None,
                            strokeStyle="dashed" if dashed else "solid", elbowed=False)

    def arrow(self, x1, y1, x2, y2, dashed=False):
        return self.path([(x1, y1), (x2, y2)], True, dashed)

    def line(self, x1, y1, x2, y2, dashed=False):
        return self.path([(x1, y1), (x2, y2)], False, dashed)

    def circle(self, cx, cy, radius, label=None):
        self.shape("ellipse", cx-radius, cy-radius, 2*radius, 2*radius)
        if label:
            self.text(cx, cy-23, label, True)

    def dot(self, x, y, radius=10):
        self.element("ellipse", x-radius, y-radius, radius*2, radius*2,
                     backgroundColor=INK, roughness=0)

    def arc(self, cx, cy, radius, start, end, arrow=False):
        return self.path([(cx+radius*math.cos(t), cy+radius*math.sin(t))
                          for t in linspace(start, end, 25)], arrow)

    def wave(self, x, y, width, amplitude, cycles, phase=0, dashed=False):
        return self.path([(x+width*t, y-amplitude*math.sin(2*math.pi*cycles*t+phase))
                          for t in linspace(0, 1, 100)], dashed=dashed)

    def axes(self, x, y, width, height, xlabel, ylabel, xlabel_offset=40):
        self.arrow(x, y, x+width, y)
        self.arrow(x, y, x, y-height)
        self.text(x, y-height-85, ylabel)
        self.text(x+width/2, y+xlabel_offset, xlabel, True)


def linspace(start, end, count):
    return [start+(end-start)*i/(count-1) for i in range(count)]


def lorentz(t, center, width):
    return 1/(1+((t-center)/width)**2)


def curve(d, x, bottom, width, height, fn, dashed=False):
    return d.path([(x+width*t, bottom-height*fn(t)) for t in linspace(0, 1, 140)], dashed=dashed)


def ring(d, cx, cy, r, scatterers=False, directions=True):
    d.circle(cx, cy, r)
    if scatterers:
        for sy in [cy-r, cy+r]:
            d.shape("diamond", cx-24, sy-24, 48, 48)
    if directions:
        d.arc(cx, cy, r-55, -1.2, -.1, True)
        d.arc(cx, cy, r-105, 2.7, 1.5, True)
        d.text(cx+35, cy-90, "CW")
        d.text(cx-110, cy+35, "CCW")


def build():
    d = Drawing("01-operating-side", "where does the operating point sit?",
                "wavelength scan - hold the ring temperature fixed for this snapshot", "evaluate_exact_ring_response; _set_trial_temperature_phase")
    d.axes(160, 850, 980, 490, "laser wavelength", "through power", xlabel_offset=110)
    curve(d, 160, 850, 980, 420, lambda t: .9-.78*lorentz(t, .57, .075))
    laser_x = 160+980*.48
    laser_y = 850-420*(.9-.78*lorentz(.48, .57, .075))
    d.line(laser_x, 850, laser_x, 375, True)
    d.dot(laser_x, laser_y)
    d.text(170, 1070, "laser on the left flank")
    d.text(170, 1140, "ring resonance at a longer wavelength")
    d.text(160+980*.57, 875, "ring resonance", True)
    d.text(1370, 320, "now hold the laser wavelength fixed")
    for row, shift in enumerate([0, 130]):
        y = 630+row*480
        d.line(1390, y, 2240, y)
        curve(d, 1390, y, 850, 210, lambda t, s=shift: .9-.78*lorentz(t, .54+s/850, .08))
        d.line(1730, y+20, 1730, y-235, True)
        if row==0:d.text(1760,400,"fixed laser")
        d.dot(1730, y-210*(.9-.78*lorentz(.4, .54+shift/850, .08)))
    d.arrow(1850, 735, 1980, 735)
    d.text(1900, 790, "warmer ring - resonance moves right", True)
    d.text(1370, 1165, "less light enters the ring")
    d.text(1370, 1225, "less absorption opposes the extra warmth")
    d.text(160, 1350, "the other flank can amplify warming - instability depends on feedback strength")

    d = Drawing("02-resonance-orders", "why can one ring have many resonances?",
                "one transverse mode can support many round-trip resonance orders", "calculate_propagation_and_scatterer_coefficients; interpolate_scatterer_reflection_and_phase_derivative")
    d.text(130, 270, "cross-section")
    d.shape("rectangle", 150, 385, 420, 240)
    d.shape("ellipse", 275, 435, 170, 140)
    d.text(120, 705, "one transverse pattern")
    d.text(120, 765, "this is the mode family")
    d.text(850, 270, "unwrap one trip around the ring")
    for i, cycles in enumerate([5, 6, 7]):
        y = 420+i*210
        d.line(900, y-75, 900, y+75, True)
        d.line(2040, y-75, 2040, y+75, True)
        d.wave(900, y, 1140, 60, cycles)
        d.text(2100, y-23, ["one order", "next order", "next order"][i])
    d.text(900, 1040, "each row fits a whole number of waves into one trip")
    d.axes(200, 1320, 1950, 160, "wavelength", "ring response")
    curve(d, 200, 1320, 1950, 130, lambda t: sum(.8*lorentz(t,c,.013) for c in [.13,.36,.59,.82]))
    d.text(1180, 1125, "repeated resonance peaks", True)

    d = Drawing("03-coupling-and-interference", "why do fields need direction and phase?",
                "a coupler mixes fields - matching outgoing waves then interfere", "solve_exact_ring_fields_and_energy; solve_exact_ring_fields_and_phase_derivatives")
    d.text(160, 280, "light stays in the bus")
    d.arrow(180, 460, 1030, 460)
    d.text(160, 565, "light returns from the ring")
    d.path([(180, 740), (590, 740), (820, 490)], True)
    d.circle(850, 460, 28)
    d.arrow(910, 460, 1150, 460)
    d.text(760, 860, "same outgoing path", True)
    d.text(1370, 270, "aligned fields")
    d.wave(1400, 420, 670, 55, 2)
    d.wave(1400, 590, 670, 55, 2)
    d.arrow(2150, 455, 2150, 610)
    d.text(1370, 700, "larger outgoing power")
    d.text(1370, 870, "opposed fields")
    d.wave(1400, 1020, 670, 55, 2)
    d.wave(1400, 1190, 670, 55, 2, math.pi)
    d.text(1370, 1300, "partial or complete cancellation")
    d.text(140, 1070, "phase tracks where each wave is in its cycle")
    d.text(140, 1150, "add fields before calculating their power")
    d.text(140, 1230, "coupling itself does not create energy")

    d = Drawing("04-backscatter-and-splitting", "what does backscatter change?",
                "part of the circulating light turns around and couples to the opposite direction", "interpolate_scatterer_reflection_and_phase_derivative; solve_exact_ring_fields_and_energy")
    ring(d, 590, 610, 250, True)
    d.text(590, 280, "effective scatterer A", True)
    d.text(590, 920, "effective scatterer B", True)
    d.arrow(640, 370, 790, 445)
    d.arrow(540, 390, 385, 460)
    d.text(170, 1070, "two equivalent scatterers represent distributed roughness")
    d.text(170, 1140, "their positions are a model choice")
    d.text(1440, 290, "weak backscatter")
    d.axes(1400, 690, 780, 260, "wavelength", "drop power")
    curve(d,1400,690,780,230,lambda t:.9*lorentz(t,.5,.09))
    d.text(1440, 870, "resolved splitting")
    d.axes(1400, 1270, 780, 260, "wavelength", "drop power")
    curve(d,1400,1270,780,230,lambda t:.76*lorentz(t,.37,.055)+.76*lorentz(t,.63,.055))
    d.text(160, 1300, "two coupled travelling waves can form two resonant patterns")

    d = Drawing("05-travel-and-coordinate", "which way does each field travel?",
                "physical travel and the order used to walk the ring are different ideas", "solve_exact_ring_fields_and_energy; solve_ring_fields_at_all_laser_wavelengths")
    ring(d, 690, 750, 350, True)
    d.circle(340, 750, 27)
    d.circle(1040, 750, 27)
    d.text(120, 700, "coupler 1")
    d.text(1100, 700, "coupler 2")
    d.text(690, 310, "scatterer A", True)
    d.text(690, 1170, "scatterer B", True)
    d.text(170, 1310, "four quarter-trip sections - not four resonance orders")
    d.text(1530, 300, "walk clockwise through the model")
    d.box(1550, 430, 620, 160, "carry both travelling fields")
    d.arrow(1860, 625, 1860, 735)
    d.box(1550, 770, 620, 160, "forward field follows its travel")
    d.arrow(1860, 965, 1860, 1070)
    d.box(1550, 1100, 620, 160, "backward field is traced\nagainst its travel")
    d.text(1450, 1330, "undoing attenuation in a transfer rule\nis not physical gain")

    d = Drawing("06-field-closure", "how are the circulating fields found?",
                "at a fixed temperature, the field must agree with itself after one complete trip", "solve_exact_ring_fields_and_energy; solve_exact_ring_fields_and_phase_derivatives")
    d.box(180, 380, 650, 180, "fields just after\nthe input coupler")
    d.arrow(875, 470, 1320, 470)
    d.text(1090, 370, "propagate and scatter", True)
    d.box(1370, 380, 750, 180, "fields returning\nto that same coupler")
    d.arrow(1745, 605, 1745, 830)
    d.text(1800, 675, "mix with incoming light")
    d.box(1370, 875, 750, 180, "fields after the coupler\nmust match the starting fields")
    d.path([(1320, 965), (505, 965), (505, 605)], True)
    d.text(180, 1090, "solve both directions together")
    d.text(180, 1160, "this is a small simultaneous linear field calculation")
    d.text(180, 1300, "the closure is solved directly - these arrows are not time-integration steps")

    d = Drawing("07-energy-and-heating", "where does incoming optical power go?",
                "distinguish travelling fields, circulating energy, optical loss, and deposited heat", "calculate_quarter_ring_energy_per_power_and_loss_fraction; evaluate_ring_optical_outputs; RingResponse")
    d.arrow(120, 650, 480, 650)
    d.text(130, 545, "incoming light")
    ring(d, 800, 650, 250, directions=False)
    d.text(800, 585, "stored optical\nenergy", True)
    d.arrow(1110, 480, 1570, 330)
    d.text(1640, 295, "through, drop, add, reflection")
    d.arrow(1110, 805, 1570, 960)
    d.text(1640, 925, "intrinsic optical loss")
    d.path([(1810, 1040), (1810, 1150), (1510, 1240)], True)
    d.path([(1890, 1040), (1890, 1150), (2190, 1240)], True)
    d.text(1320, 1300, "absorbed in the ring\nbecomes heat", True)
    d.text(2150, 1300, "other loss\nleaves the optical system", True)
    d.text(120, 1090, "field - amplitude and phase")
    d.text(120, 1160, "power - energy passing per time")
    d.text(120, 1230, "stored energy - light present in the ring")

    d = Drawing("08-two-data-levels", "how does fast binary data enter the model?",
                "calculate both voltage states at the same temperature before averaging their powers", "nrz_levels; evaluate_ring_optical_outputs; average_binary_data_levels")
    for x, label in [(180,"zero-bit voltage"),(1300,"one-bit voltage")]:
        d.box(x,300,780,125,label)
        d.arrow(x+390,460,x+390,540)
        d.box(x,575,780,165,"index and optical loss\nat this voltage")
        d.arrow(x+390,775,x+390,850)
        d.box(x,885,780,165,"phase, circulating fields,\nand optical powers")
    d.path([(570,1090),(570,1190),(1010,1190)],True)
    d.path([(1690,1090),(1690,1190),(1360,1190)],True)
    d.text(1200,1160,"weight by time spent\nin each data level",True)
    d.text(160,1370,"averaged absorption drives slow heating - retain the separate levels for link metrics")

    d = Drawing("09-eight-ring-bus", "how do eight rings share the light?",
                "eight laser lines share the bus - upstream rings and waveguides remove power", "propagate_bus_optical_power_into_output_buffers; write_bus_output_powers; BusResult")
    d.text(100,250,"drop monitors - one at each ring")
    d.arrow(100,640,2300,640)
    for i in range(8):
        cx=300+i*260
        d.circle(cx,510,90)
        d.circle(cx,615,15)
        d.arrow(cx,400,cx,350)
        d.text(cx,790,f"ring {i}",True)
        d.text(cx,900,f"lane {i}",True)
        if i<7:
            d.text(cx+130,705,"loss",True)
    d.text(170,1010,"each monitor collects its ring's drop power across all laser lines")
    d.text(170,1090,"each ring heats from its absorbed share across all laser lines")
    d.text(170,1170,"different laser lines contribute powers independently in this model")
    d.text(170,1340,"bus order is physical encounter order - output arrays retain physical ring indices")

    d = Drawing("10-reflections", "which returned light is included?",
                "follow the actual optical path before deciding how its power is combined", "write_returned_reflection_power_and_forward_error_bound; write_bus_output_powers")
    d.text(130,280,"included - one ring reflects light toward the input")
    for cx in [650,1250,1850]:
        d.circle(cx,430,75)
    d.arrow(140,570,2200,570)
    d.path([(1820,620),(1250,620),(650,620),(160,620)],True)
    d.text(130,730,"upstream transmission is encountered again on the return trip")
    d.text(130,800,"the current binary model uses the same upstream bit state on both crossings")
    d.text(130,870,"the omitted-bounce estimate is diagnostic - it does not correct forward power")
    d.text(130,990,"omitted - repeated reflections between rings")
    for cx in [650,1850]:
        d.circle(cx,1140,75)
    d.line(140,1270,2200,1270)
    d.arrow(700,1225,1800,1225,True)
    d.arrow(1800,1320,700,1320,True)
    d.text(880,1140,"multiple bounces",True)
    d.text(130,1410,"returned powers are added - interference between different rings' returns is omitted")

    d = Drawing("11-heat-paths", "what determines a ring's temperature?",
                "heat arrives through several paths - each path retains its own thermal memory", "ThermalSystem; mrm/plant/thermal.py")
    d.box(120,320,480,150,"heater")
    d.box(120,700,480,150,"absorbed light")
    d.box(120,1080,480,150,"nearby heat sources")
    for y in [395,775,1155]:
        d.arrow(640,y,1030,y)
        d.text(825,y-80,"thermal path",True)
        d.shape("rectangle",1070,y-60,330,120)
        d.text(1235,y-23,"stored warmth",True)
        d.arrow(1440,y,1740,775)
    d.circle(1940,775,150)
    d.text(1940,720,"ring\ntemperature",True)
    d.text(1650,280,"background base temperature")
    d.arrow(1940,395,1940,580)
    d.arrow(1940,975,1940,1130)
    d.text(1940,1210,"heat escapes toward\nthe surrounding structure",True)
    d.text(110,1365,"an observer reports a chosen local temperature - it is not a new heat source")

    d = Drawing("12-self-heating-loop", "how can temperature affect its own heating?",
                "follow the loop with the laser wavelength held fixed", "_set_trial_temperature_phase; absorbed_power_at_trial_temperature")
    nodes=[(150,360,700,170,"ring temperature"),(1400,360,760,170,"resonance position\nand round-trip phase"),
           (1400,970,760,170,"circulating light\nand absorbed power"),(150,970,700,170,"new thermal response")]
    for args in nodes:d.box(*args)
    d.arrow(900,445,1350,445)
    d.arrow(1780,580,1780,920)
    d.arrow(1350,1055,900,1055)
    d.arrow(500,920,500,580)
    d.text(1100,655,"stable-side response",True)
    d.text(1100,725,"extra warmth moves the resonance away from the laser",True)
    d.text(1100,795,"absorption falls and opposes that warmth",True)
    d.text(160,1300,"temperature sets the optical response - absorbed light then feeds the thermal dynamics")

    d = Drawing("13-heating-cooling-vector-fields", "which temperatures attract or repel?",
                "compare absorbed heating with heat loss after accounting for other heating", "selfheat.py - equilibrium interpretation")
    d.text(150,250,"solid hump - absorbed power     dashed line - heat loss still to be supplied")
    # a dimensionless illustrative heat budget - no device calibration is implied.
    def p(t): return .08+.72*lorentz(t,.56,.105)
    cases=[(0,1.5,"one cold equilibrium"),(0,.9,"two stable equilibria"),(.35,.9,"one hot equilibrium")]
    for j,(offset,loss_slope,label) in enumerate(cases):
        x=160+j*770; y=950; width=610; height=430
        d.text(x,370,label)
        d.axes(x,y,width,height,"temperature","power")
        curve(d,x,y,width,height,p)
        cool=lambda t,off=offset,k=loss_slope: k*t-off
        d.path([(x+width*t,y-height*cool(t)) for t in linspace(0,1,140) if 0<=cool(t)<=1],dashed=True)
        roots=[]; ts=linspace(0,1,1200)
        for a,b in zip(ts,ts[1:]):
            if (p(a)-cool(a))*(p(b)-cool(b))<0:
                roots.append((a+b)/2)
        # arrow direction follows the heat surplus along the temperature axis.
        boundaries=[0]+roots+[1]
        for left,right in zip(boundaries,boundaries[1:]):
            mid=(left+right)/2
            if right-left>.06:
                xa=x+width*(mid-.035); xb=x+width*(mid+.035)
                d.arrow(xa if p(mid)>cool(mid) else xb,1080,xb if p(mid)>cool(mid) else xa,1080)
        for r in roots:
            rx=x+width*r; ry=y-height*p(r)
            d.dot(rx,ry)
            if ry<900:d.line(rx,ry+22,rx,930,True)
        d.text(x,1170,"arrows show temperature motion")
    d.text(160,1330,"heating above cooling - warmer     heating below cooling - cooler")
    d.text(160,1400,"arrows toward a crossing - stable     arrows away - unstable     schematic curves")

    d = Drawing("14-folds-and-history", "why can the ring suddenly jump?",
                "follow steady temperatures as the heater input changes slowly", "selfheat.py - branch history; lorentzian_folds")
    x0,y0,pw,ph=250,1200,1300,800
    d.axes(x0,y0,pw,ph,"heater input","equilibrium temperature")
    # parameterize the branch by temperature so the folds remain geometrically consistent.
    ts=linspace(0,8,1400)
    xs=[t-3.5*lorentz(t,4,1) for t in ts]
    turning=[]
    for i in range(1,len(ts)-1):
        if (xs[i]-xs[i-1])*(xs[i+1]-xs[i])<0:turning.append(i)
    def pt(i):return (x0+100+(xs[i]+.3)*145,y0-60-ts[i]*85)
    cuts=[0]+turning+[len(ts)-1]
    for j,(a,b) in enumerate(zip(cuts,cuts[1:])):
        d.path([pt(i) for i in range(a,b+1,3)]+[pt(b)],dashed=j==1)
    for i in turning:d.dot(*pt(i),12)
    cold_fold,hot_fold=turning
    coldx,coldy=pt(cold_fold); hotx,hoty=pt(hot_fold)
    hot_dest=min(range(hot_fold,len(ts)),key=lambda i:abs(xs[i]-xs[cold_fold]))
    cold_dest=min(range(0,cold_fold),key=lambda i:abs(xs[i]-xs[hot_fold]))
    d.arrow(coldx+20,coldy-40,coldx+20,pt(hot_dest)[1]+35)
    d.arrow(hotx-20,hoty+40,hotx-20,pt(cold_dest)[1]-35)
    d.text(540,1120,"cold stable branch")
    d.text(1080,440,"hot stable branch")
    d.text(790,770,"unstable branch")
    d.text(1740,390,"same heater input")
    d.line(1800,510,1800,1120,True)
    d.dot(1800,600,13);d.dot(1800,1010,13)
    d.text(1870,570,"hot state")
    d.text(1870,980,"cold state")
    d.text(1670,1220,"history selects the branch")
    d.text(170,1380,"a stable branch ends at a fold - the ring moves to the remaining attracting state")

    d = Drawing("15-thermal-memory", "what does a thermal step preserve?",
                "temperature approaches a new balance gradually - earlier warmth is retained", "thermal.py - lag_coeffs; selfheat.py - implicit_step_b")
    d.axes(160,880,960,440,"time","temperature rise")
    curve(d,160,880,960,400,lambda t:.18+.65*(1-math.exp(-4*t)))
    d.line(160,880-400*.83,1120,880-400*.83,True)
    d.dot(160,880-400*.18)
    d.text(170,1020,"start with the existing stored warmth")
    d.text(170,1090,"held heating gives an exponential response")
    d.text(1390,300,"optical heating changes during a real step")
    d.axes(1400,880,760,440,"time within one step","absorbed power")
    curve(d,1400,880,760,400,lambda t:.2+.58*t*t)
    d.line(1400,568,2160,568,True)
    d.text(1390,1020,"solid - continuously changing heating")
    d.text(1390,1090,"dashed - endpoint power held across the step")
    d.text(160,1290,"exact for held power - an approximation when the actual power changes within the step")
    d.text(160,1360,"a step does not reset the ring to thermal equilibrium")

    d = Drawing("16-trial-temperature", "what happens inside the temperature solve?",
                "candidate temperatures refer to one endpoint - solver iterations do not advance physical time", "solve_ring_temperature; absorbed_power_at_trial_temperature; commit_optics_at_solved_temperature")
    d.box(130,330,630,160,"choose a trial\nendpoint temperature")
    d.arrow(810,410,1330,410)
    d.box(1380,330,780,160,"calculate absorption\nat that trial temperature")
    d.arrow(1770,535,1770,730)
    d.box(1380,775,780,170,"calculate the thermal endpoint\nusing that absorbed power")
    d.arrow(1330,860,810,860)
    d.box(130,775,630,170,"compare trial and\npredicted temperatures")
    d.path([(445,730),(445,535)],True)
    d.text(520,600,"disagree - revise the trial")
    d.arrow(445,990,445,1140)
    d.text(140,1210,"agree - accept the endpoint")
    d.arrow(840,1240,1330,1240)
    d.text(1390,1185,"update thermal memory\nand pass light to the next ring")

    d = Drawing("17-local-slope-and-bounds", "what do derivatives and bounds tell the solver?",
                "a local slope describes the current sensitivity - a bound covers a whole specified interval", "evaluate_ring_absorbed_power_and_resonance_wavelength_derivative; bound_ring_absorbed_power_and_phase_curvature; bound_ring_absorbed_power_phase_derivative; bound_absorbed_power_temperature_response_from_ring_parameters")
    d.axes(170,850,950,440,"trial temperature","absorbed power")
    fn=lambda t:.82*lorentz(t,.56,.12)
    curve(d,170,850,950,400,fn)
    t=.68; xx=170+950*t; yy=850-400*fn(t)
    slope=(fn(t+.0001)-fn(t-.0001))/.0002
    d.line(xx-90,yy+400*slope*90/950,xx+90,yy-400*slope*90/950)
    d.dot(xx,yy)
    d.text(150,1000,"tangent - local change in heating")
    d.text(150,1070,"the next trial still uses the full optical calculation")
    d.text(1390,320,"temperature interval to be covered")
    d.line(1420,470,2190,470)
    d.line(1420,435,1420,505);d.line(2190,435,2190,505)
    d.text(1390,640,"limits needed across that whole interval")
    for y,label in [(760,"largest absorbed power"),(900,"largest slope magnitude"),(1040,"largest curvature magnitude")]:
        d.circle(1430,y+23,12)
        d.text(1480,y,label)
    for x,label in [(150,"temperature"),(760,"resonance"),(1310,"phase"),(1810,"absorption")]:
        d.text(x,1290,label)
    for x1,x2 in [(450,710),(1030,1260),(1480,1760)]:d.arrow(x1,1315,x2,1315)
    d.text(150,1380,"bounds support branch-aware searches - their tested range is not an arbitrary-backscatter proof")

    d = Drawing("18-controller-target", "why hold a fraction of the monitor peak?",
                "the target is a tunable hyperparameter - it is not a fixed transmission or maximum-slope rule", "mrm/control - PI controller; controller.params.fraction")
    d.box(130,330,550,150,"sweep the heater")
    d.arrow(725,405,925,405)
    d.box(970,330,550,150,"measure monitor peak")
    d.arrow(1565,405,1765,405)
    d.box(1810,330,450,150,"choose target")
    d.text(1810,540,"default - 70% of peak")
    d.text(130,700,"when laser power falls, the old target stays fixed until refreshed")
    for x,peak,label in [(350,430,"earlier peak - 100"),(1330,344,"new peak - 80")]:
        bottom=1290
        d.line(x,bottom,x+620,bottom)
        d.shape("rectangle",x+100,bottom-peak,180,peak)
        d.line(x,bottom-301,x+620,bottom-301,True)
        d.text(x,bottom-peak-90,label)
        d.text(x+330,bottom-280,"target - 70")
    d.text(160,1380,"the same target is now nearer the new peak - the controller can move closer to resonance")

    d = Drawing("19-monitor-measurement", "what does the controller actually measure?",
                "a larger ADC reading can come from more than a resonance shift", "mrm/plant - monitor model; mrm/sim/kernel.py - sensor pipeline")
    d.box(130,330,550,170,"drop light\nfrom every laser line")
    d.arrow(725,415,930,415)
    d.box(980,330,500,170,"photodiode\nlight becomes current")
    d.arrow(1525,415,1730,415)
    d.box(1780,330,490,170,"integration\naverages over time")
    d.text(940,630,"warm photodiode")
    d.arrow(1230,700,1230,555)
    d.text(940,770,"dark current adds background")
    d.path([(2025,550),(2025,950),(1540,950)],True)
    d.box(980,865,500,170,"ADC\nfinite codes and clipping")
    d.arrow(930,950,725,950)
    d.box(130,865,550,170,"delayed reading\nreaches the controller")
    d.text(1690,1080,"gain uncertainty and noise")
    d.text(140,1240,"default integration - 2 microseconds     ADC latency - 2 microseconds")
    d.text(140,1320,"default sampling - 250 kHz     heater-command latency - 1 microsecond")

    d = Drawing("20-link-metrics", "how can link quality be estimated without tracing every bit?",
                "retain the zero and one output powers - their separation predicts receiver margin", "write_bus_output_powers; mrm/plant/datapath.py - ber_from_oma; errors_tick_row")
    d.text(140,300,"zero-level output")
    d.text(140,670,"one-level output")
    d.line(150,490,700,490)
    d.line(150,860,700,860)
    d.arrow(760,490,1080,650)
    d.arrow(760,860,1080,700)
    d.text(920,360,"power separation",True)
    d.box(1130,570,530,180,"optical modulation\namplitude")
    d.arrow(1710,660,1910,660)
    d.text(1980,590,"link loss\nand receiver model")
    d.axes(350,1300,1150,280,"received sample level","probability")
    curve(d,350,1300,1150,250,lambda t:.9*math.exp(-((t-.3)/.11)**2))
    curve(d,350,1300,1150,250,lambda t:.9*math.exp(-((t-.7)/.11)**2))
    d.line(925,1300,925,990,True)
    d.text(925,930,"decision threshold",True)
    d.text(1670,1070,"overlap predicts errors")
    d.text(1670,1150,"accumulate expected errors")
    d.text(1670,1230,"sample the reported error count")
    d.text(140,1410,"schematic Gaussian receiver model - fast waveform distortion is not resolved by the slow loop")

    d = Drawing("21-model-and-evidence", "what establishes accuracy?",
                "calculation checks establish consistency - measurements test agreement with the device", "tests/verification/test_optics.py; docs/KNOWN_ISSUES.md; docs/OPTICS_CONSOLIDATION.md")
    d.text(150,300,"check the mathematical implementation")
    d.circle(600,670,160)
    d.text(600,625,"ring\ncalculation",True)
    d.arrow(140,670,390,670)
    d.text(150,545,"input power")
    d.arrow(810,610,1120,470)
    d.arrow(810,750,1120,890)
    d.text(900,370,"outgoing power")
    d.text(960,945,"loss")
    d.text(180,1100,"power balance, reciprocity, and slope checks")
    d.line(1260,300,1260,1240,True)
    d.text(1440,300,"compare with device measurements")
    d.axes(1450,1000,740,460,"heater input","temperature")
    fn=lambda t:.12+.72*t-.16*math.sin(4*t)
    curve(d,1450,1000,740,420,fn)
    for i,t in enumerate([.1,.23,.39,.54,.72,.87]):
        d.dot(1450+740*t,1000-420*(fn(t)+[.02,-.03,.02,.04,-.02,.02][i]),9)
    d.text(1420,1140,"line - model     dots - measurements")
    d.text(1420,1210,"illustration only - not calibration data")
    d.text(160,1360,"backscatter, heat paths, and sensor parameters need credible device evidence")

    d = Drawing("22-one-control-interval", "how do all the time scales fit together?",
                "fast data changes the optical state - slow thermal and control updates use its statistics", "mrm/plant/plant.py; mrm/sim/kernel.py; mrm/plant/datapath.py")
    d.text(150,310,"data - 50 Gb/s")
    pts=[]
    for i in range(24):
        x=580+i*65; level=450 if [0,1,1,0,1,0,0,1][i%8] else 530
        if pts:pts.append((x,pts[-1][1]))
        pts.append((x,level));pts.append((x+65,level))
    d.path(pts)
    d.text(150,690,"plant - 1 microsecond steps")
    d.arrow(580,850,2200,850)
    for i in range(11):
        x=580+i*155;d.line(x,820,x,880)
    d.text(580,930,"about 50,000 bits within each plant step")
    d.text(150,1110,"controller - every 10 microseconds")
    d.arrow(580,1250,2200,1250)
    for x in [580,2130]:d.circle(x,1250,18)
    d.text(1450,1110,"read delayed sensors - update heater",True)
    d.text(150,1410,"optics settles rapidly at each trial temperature - inner iterations are outside this time axis")

    d = Drawing("23-stored-heat-and-cooling", "why does temperature take time to change?",
                "incoming heat changes stored thermal energy while heat also escapes", "mrm/plant/thermal.py - thermal lag states")
    d.arrow(120,650,610,650)
    d.text(150,520,"heat input")
    d.shape("rectangle",680,420,720,550)
    d.text(1040,590,"stored thermal energy",True)
    d.text(1040,710,"thermal mass",True)
    d.arrow(1450,650,1870,650)
    d.text(1570,510,"heat escape path")
    d.text(1940,590,"surroundings")
    d.text(150,1130,"larger thermal mass")
    d.arrow(630,1155,820,1155)
    d.text(900,1130,"more energy needed for the same temperature rise")
    d.text(150,1280,"larger thermal resistance")
    d.arrow(720,1305,890,1305)
    d.text(970,1280,"harder for heat to escape")

    d = Drawing("24-ring-model-choice", "what does the chosen ring model resolve?",
                "one production path - solve the fields travelling in both directions", "solve_exact_ring_fields_and_energy; archive/optics/tcmt.py; mrm/waveform/tcmt_time.py")
    ring(d,600,650,260,True)
    d.text(600,1030,"propagate around the ring",True)
    d.text(600,1100,"apply couplers and scatterers",True)
    d.text(600,1170,"close the circulating fields",True)
    d.line(1230,300,1230,1250,True)
    d.text(1490,320,"resonant-mode approximation")
    d.axes(1460,900,760,350,"wavelength","response")
    curve(d,1460,900,760,300,lambda t:.85*lorentz(t,.5,.09))
    d.line(1720,975,1720,500,True)
    d.line(1960,975,1960,500,True)
    d.text(1840,1070,"focus near one resonance",True)
    d.text(1450,1200,"TCMT steady-state alternative is archived")
    d.text(140,1370,"the separate time-domain waveform model still studies fast optical dynamics")

    d = Drawing("25-peak-seeking-and-lock-loss", "why can seeking the largest monitor reading lose lock?",
                "with strong self-heating, the high-reading state can sit close to a fold", "selfheat.py - branch following; PI peak-fraction target")
    labels=["stable hot operating point","move toward a larger reading","cross the fold - ring cools"]
    centers=[.65,.52,.16]
    for j,(label,center) in enumerate(zip(labels,centers)):
        x=150+j*780
        d.text(x,350,label)
        d.axes(x,860,590,300,"wavelength","through power")
        curve(d,x,860,590,270,lambda t,c=center:.9-.78*lorentz(t,c,.075))
        lx=x+590*.4
        d.line(lx,890,lx,520,True)
        d.text(lx,990,"fixed laser",True)
        d.dot(lx,860-270*(.9-.78*lorentz(.4,center,.075)))
    d.arrow(750,1100,890,1100)
    d.arrow(1530,1100,1670,1100)
    d.text(150,1220,"resonance is red of the laser")
    d.text(930,1220,"absorption and monitor rise")
    d.text(1710,1220,"absorption collapses")
    d.text(150,1380,"cooling moves the resonance away across the laser - a larger-reading search can trigger the jump")


def scene(elements):
    return dict(type="excalidraw", version=2, source="mrm-model optical walkthrough",
                elements=elements, appState=dict(viewBackgroundColor="#ffffff", gridSize=None), files={})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "drawings").mkdir(exist_ok=True)
    build()
    reference=json.loads(REFERENCE.read_text())
    original=scene(reference["elements"])
    original["files"]=reference.get("files",{})
    (OUT/"drawings/00-ring-model-reference.excalidraw").write_text(json.dumps(original,indent=2)+"\n")
    manifest=[dict(slug="00-ring-model-reference",title="ring model - your reference",subtitle="the supplied drawing, preserved without redrawing",code="solve_exact_ring_fields_and_energy",reference=True)]
    for board in BOARDS:
        (OUT/"drawings"/f"{board.slug}.excalidraw").write_text(json.dumps(scene(board.elements),indent=2)+"\n")
        manifest.append(dict(slug=board.slug,title=board.title,subtitle=board.subtitle,code=board.code))
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (OUT/"manifest.js").write_text("window.OPTICS_DIAGRAMS = "+json.dumps(manifest,indent=2)+";\n")
    print(f"wrote {len(manifest)} editable diagrams to {OUT}")


if __name__ == "__main__":
    main()
