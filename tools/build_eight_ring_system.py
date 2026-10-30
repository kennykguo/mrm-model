#!/usr/bin/env python3
# build one labelled, equation-free overview of the nominal eight-ring system.
from __future__ import annotations

import json
import math
from pathlib import Path

from build_excalidraw_atlas import Drawing, linspace, scene

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/diagrams/eight-ring-system"
CENTERS = [1150+430*i for i in range(8)]
LASER_COLORS = ["#2563eb", "#00879a", "#15803d", "#779219", "#af7a00", "#d65a15", "#c02670", "#7c3aed"]
ELECTRICAL = "#475569"
THERMAL = "#b45309"


class SystemDrawing(Drawing):
    def label(self, x, y, text, size=36, center=False):
        e=self.text(x,y,text,center)
        anchor=e["x"]+e["width"]/2 if center else x
        e["fontSize"]=size
        e["width"]*=size/36
        e["height"]*=size/36
        e["x"]=anchor-e["width"]/2 if center else x
        return e

    def block(self,x,y,width,height,text,size=36,color=None):
        border=self.shape("rectangle",x,y,width,height)
        if color:border["strokeColor"]=color
        self.label(x+width/2,y+(height-size*1.25*len(text.splitlines()))/2,text,size,True)

    def wire(self,points,tip=False):
        # curved crossover segments are grouped dashed paths.
        first=len(self.elements)
        if tip:
            self.path(points[:-1],dashed=True)
            self.arrow(*points[-2],*points[-1],dashed=True)
        else:self.path(points,dashed=True)
        for element in self.elements[first:]:
            element["strokeColor"]=ELECTRICAL
            element["strokeWidth"]=3

    def heat(self,x1,y1,x2,y2,tip=False):
        dx,dy=x2-x1,y2-y1
        length=math.hypot(dx,dy)
        if length==0:return
        amplitude=8
        cycles=max(2,int(length/85))
        points=[]
        for t in linspace(0,1,max(35,cycles*14)):
            wave=amplitude*math.sin(2*math.pi*cycles*t)
            points.append((x1+dx*t-dy/length*wave,y1+dy*t+dx/length*wave))
        element=self.path(points,arrow=tip)
        element["strokeColor"]=THERMAL
        element["strokeWidth"]=3
        return element

    def resistor(self,x,y,height=140):
        points=[(x,y)]
        for i in range(1,13):points.append((x+(11 if i%2 else -11),y+height*i/13))
        points.append((x,y+height))
        element=self.path(points)
        element["strokeColor"]=THERMAL

    def spectrum(self,x1,x2,y):
        for i,color in enumerate(LASER_COLORS):
            line=self.line(x1,y+(i-3.5)*2,x2,y+(i-3.5)*2)
            line.update(strokeColor=color,strokeWidth=2,roughness=0)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"drawings").mkdir(exist_ok=True)
    d=SystemDrawing("eight-ring-system","eight-ring system","","")
    d.elements.clear()
    d.label(100,70,"eight-ring system",56)
    d.label(100,160,"mrm-model · nominal G26",36)

    # controller and the independent diagnostic sink.
    d.shape("rectangle",2200,180,1000,340)["strokeColor"]=ELECTRICAL
    d.label(2700,220,"lock controller",48,True)
    d.label(2700,305,"peak map / PI",36,True)
    d.label(2700,385,"70% peak fraction · hyperparameter",36,True)
    d.label(2700,455,"100 kHz",32,True)
    d.wire([(2700,520),(2700,610)])
    d.label(2820,545,"heater commands",32)
    d.wire([(CENTERS[0],610),(CENTERS[-1],610)])
    d.block(4580,300,900,250,"diagnostics\nOMA · BER · error counts",40,ELECTRICAL)
    # heater command and shared-supply rows.
    d.label(100,650,"heater actuation",44)
    d.label(100,730,"16-bit DACs\n1 µs latency",36)
    d.block(420,825,350,130,"shared supply\nV / I / P limits",32,ELECTRICAL)
    d.wire([(770,890),(4290,890)])

    # per-lane data inputs and electrical connections.
    d.label(100,1030,"data drive",44)
    d.label(100,1115,"8 independent lanes\n50 Gb/s NRZ\n8 bias DACs · 1.05 V",36)
    for i,cx in enumerate(CENTERS):
        d.wire([(cx,610),(cx,740)],True)
        d.block(cx-135,740,270,120,f"heater DAC {i}",32,ELECTRICAL)
        d.dot(cx+70,890,5)
        d.wire([(cx+70,890),(cx+70,860)],True)
        d.wire([(cx-65,860),(cx-65,870),(cx-85,890),(cx-65,910),
                (cx-65,940),(cx-185,940),(cx-185,1367),
                (cx-208,1390),(cx-185,1413),(cx-185,1490)])
        d.resistor(cx-185,1490)
        d.heat(cx-172,1550,cx-145,1550,True)
        # a labelled two-level pulse train belongs to each lane.
        pattern=[0,1,1,0,1,0,0,1]
        bits=[pattern[(b+3*i)%8] for b in range(8)]
        pts=[]
        for b,bit in enumerate(bits):
            x=cx-112+b*28;y=1025-(40 if bit else 0)
            if pts:pts.append((x,pts[-1][1]))
            pts.extend([(x,y),(x+28,y)])
        d.path(pts)["strokeColor"]=ELECTRICAL
        d.wire([(cx,1040),(cx,1110)],True)
        d.block(cx-135,1110,270,100,f"bias / NRZ {i}",32,ELECTRICAL)
        d.wire([(cx+65,1210),(cx+65,1270),(cx+185,1270),(cx+185,1367),
                (cx+208,1390),(cx+185,1413),(cx+185,1660),(cx+119,1660)],True)

    # light source and eight wavelengths.
    d.shape("rectangle",140,1270,520,400)
    d.label(400,1300,"laser bank",44,True)
    d.line(205,1490,600,1490)
    for i in range(8):
        x=225+i*48
        tooth=d.line(x,1490,x,1400)
        tooth.update(strokeColor=LASER_COLORS[i],strokeWidth=7)
        d.label(x,1510,str(i),28,True)
    d.label(400,1570,"400 GHz · near 1310 nm",32,True)
    d.label(100,1720,"returned light",32)
    d.arrow(960,1390,690,1390)
    d.label(1020,1300,"8-line shared bus\nwaveguide loss",28)
    d.arrow(660,1390,4440,1390)
    d.spectrum(660,4440,1390)

    # chip boundary is a grouping boundary, not an optical path.
    boundary=d.shape("rectangle",940,1290,3440,720,dashed=True)
    boundary["strokeWidth"]=2
    boundary["opacity"]=55
    d.label(100,1820,"photonic die",44)
    d.label(100,1910,"drop photodiodes\nall laser lines",36)

    for i,cx in enumerate(CENTERS):
        cy=1560;r=140
        body=d.shape("ellipse",cx-r,cy-r,2*r,2*r)
        body.update(strokeColor=LASER_COLORS[i],strokeWidth=5,customData=dict(role="ring",lane=i))
        d.label(cx,cy-23,str(i),36,True)
        d.arc(cx,cy,105,-1.2,.35,True)
        d.arc(cx,cy,73,2.9,1.55,True)
        for sx in [cx-r,cx+r]:d.shape("diamond",sx-12,cy-12,24,24)
        d.circle(cx,cy-r,13)
        d.line(cx,1390,cx,cy-r,True)
        d.circle(cx,cy+r,13)
        d.line(cx,cy+r,cx,1730,True)
        d.line(cx-130,1730,cx+130,1730)
        d.path([(cx+80,1730),(cx-130,1730),(cx-130,1866)],True)
        # lower-right PN electrode pair.
        d.line(cx+83,1675,cx+111,1647)
        d.line(cx+95,1687,cx+123,1659)
        # monitor current has a separate electrical path.
        d.circle(cx-130,1900,34)
        d.label(cx-130,1883,"PD",26,True)
        d.wire([(cx-130,1934),(cx-130,2170)],True)
        d.block(cx-265,2170,270,130,f"AFE / ADC\n{i}",32,ELECTRICAL)
        d.wire([(cx-130,2300),(cx-130,2410)])

    # monitor path, common readout rail, and the closed heater loop.
    d.label(100,2140,"monitor readout",44)
    d.label(100,2230,"2 µs integration\n2 µs ADC latency\n250 kHz · 16-bit\ndark current\nnoise · gain · clipping",36)
    d.wire([(850,2410),(4190,2410)])
    d.label(1270,2470,"monitor codes",36)
    d.wire([(850,2410),(850,1414),(827,1390),(850,1366),
            (850,914),(827,890),(850,866),(850,350),(2200,350)],True)
    d.block(170,2490,430,110,"die0 sensor · 1 kHz",32,ELECTRICAL)
    d.wire([(600,2545),(715,2545)],True)
    d.path([(725,2500),(840,2500),(870,2545),(840,2590),(725,2590),(725,2500)])
    d.label(790,2525,"die0",32,True)
    d.path([(4270,380),(4385,380),(4415,425),(4385,470),(4270,470),(4270,380)])
    d.label(4335,405,"die0",32,True)
    d.wire([(4415,425),(4580,425)],True)
    d.dot(850,2410,5)

    # functional link model - lane separation and a statistical receiver.
    d.block(4480,1300,310,180,"link loss",36)
    d.arrow(4440,1390,4480,1390)
    d.spectrum(4440,4480,1390)
    d.arrow(4790,1390,4890,1390)
    d.spectrum(4790,4890,1390)
    d.shape("rectangle",4890,1020,570,1080)
    d.label(5175,1055,"8-lane RX model",40,True)
    d.label(5175,1125,"per-line detection",32,True)
    d.line(4890,1390,4950,1390)
    d.line(4950,1200,4950,1935)
    for i in range(8):
        y=1200+i*105
        d.arrow(4950,y,5290,y)["strokeColor"]=LASER_COLORS[i]
        d.label(5320,y-20,str(i),32)
        # the receiver model includes detection and noise statistically.
        d.shape("ellipse",5108,y-22,44,44)["strokeColor"]=LASER_COLORS[i]
    d.label(4610,1125,"modulated\nWDM output",32)
    d.block(4650,2180,810,170,"received OMA\nBER · Poisson error counts",36,ELECTRICAL)
    d.label(4920,2390,"2 s error window",28)
    d.wire([(5175,2100),(5175,2180)],True)
    d.wire([(5460,2265),(5630,2265),(5630,425),(5480,425)],True)

    # thermal network and its die-temperature feedback.
    d.label(100,2690,"thermal environment",44)
    d.block(150,2790,450,130,"aggressor heat",36,THERMAL)
    d.block(150,3020,450,130,"package / ambient",36,THERMAL)
    d.shape("rectangle",940,2670,3410,490,dashed=True)["strokeColor"]=THERMAL
    d.label(1050,2710,"thermal crosstalk",36)
    d.label(2570,2710,"heater · absorption · driver heat",36)
    for i,cx in enumerate(CENTERS):
        d.shape("ellipse",cx-45,2815,90,90)["strokeColor"]=THERMAL
        d.label(cx,2838,str(i),36,True)
        d.line(cx-105,3030,cx+105,3030)
        d.line(cx-105,3030,cx-105,2940)
        for rate in [6,2,.65]:
            d.path([(cx-105+210*t,3030-85*math.exp(-rate*t)) for t in linspace(0,1,32)])["strokeColor"]=THERMAL
        if i<7:
            coupling=d.heat(cx+65,2860,CENTERS[i+1]-65,2860,True)
            coupling["startArrowhead"]="arrow"
    d.label(1110,3080,"fast poles · slow tails · stored heat",36)
    d.label(3210,3080,"1 µs thermal step",36)
    d.heat(600,2855,940,2855,True)
    bath=d.heat(600,3085,940,3085,True)
    bath["startArrowhead"]="arrow"
    d.heat(940,2730,760,2640)
    d.heat(760,2640,385,2640)
    d.heat(385,2640,385,2600,True)
    d.heat(4230,2860,4430,2860)
    d.heat(4430,2860,4430,2010)
    d.heat(4430,2010,4380,2010,True)
    d.label(4490,2450,"ring / PD temperature",32)

    # one common detail inset avoids repeating long labels eight times.
    d.label(4770,2570,"ring detail",44)
    cx,cy,r=5100,2910,180
    d.circle(cx,cy,r)
    d.line(4790,2690,5420,2690)
    d.arrow(4810,2690,5400,2690)
    d.spectrum(4790,5420,2690)
    d.circle(cx,cy-r,16)
    d.line(cx,2690,cx,cy-r,True)
    d.label(5360,2740,"bus coupling",32)
    d.line(5350,2800,5120,2730)
    d.line(4790,3130,5420,3130)
    d.arrow(5380,3130,4830,3130)
    d.circle(cx,cy+r,16)
    d.line(cx,cy+r,cx,3130,True)
    for sx in [cx-r,cx+r]:d.shape("diamond",sx-16,cy-16,32,32)
    d.arc(cx,cy,130,-1.2,.3,True)
    d.arc(cx,cy,90,2.8,1.3,True)
    d.label(cx+15,cy-90,"CW",32)
    d.label(cx-50,cy+10,"CCW",32)
    d.resistor(4850,2830,150)
    d.label(4560,2710,"resistive heater",32)
    d.line(4680,2760,4830,2830)
    d.line(5210,3052,5240,3022)
    d.line(5225,3067,5255,3037)
    d.label(5360,2990,"PN junction",32)
    d.line(5350,3050,5265,3050)
    d.label(4880,3190,"drop coupling",32)
    d.label(4600,3340,"equivalent scatterers · off by default",32)
    # symbols and line styles only - no descriptive text.
    d.line(1030,3350,1190,3350)
    d.spectrum(1030,1190,3350)
    d.label(1240,3330,"optical · lane color",32)
    d.wire([(1740,3350),(1900,3350)])
    d.label(1950,3330,"electrical",32)
    d.heat(2440,3350,2600,3350)
    d.label(2650,3330,"thermal",32)
    d.shape("diamond",3090,3334,32,32)
    d.label(3170,3330,"backscatter site",32)
    d.circle(3880,3350,16)
    d.label(3940,3330,"coupler",32)

    data=scene(d.elements)
    (OUT/"drawings/eight-ring-system.excalidraw").write_text(json.dumps(data,indent=2)+"\n")
    (OUT/"manifest.json").write_text(json.dumps([dict(slug="eight-ring-system",title="eight-ring system",subtitle="",code="")],indent=2)+"\n")
    print(f"wrote {len(d.elements)} editable elements")


if __name__=="__main__":main()
