#!/usr/bin/env python3
# build concise LaTeX sheets corresponding to the phase.py visual companion.
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from build_excalidraw_atlas import Drawing, scene
from build_excalidraw_equations import definitions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/diagrams/phase-math"
TOPICS = []


def topic(slug, title, symbols, *groups):
    TOPICS.append(dict(slug=slug, title=title, symbols=symbols, groups=groups))


def prepare():
    topic("01-reference-state", "reference state", [
        (r"T_{\rm ref}\ [\mathrm K]", "reference temperature"),
        (r"V_{\rm ref}\ [\mathrm V]", "reference reverse bias"),
        (r"\lambda_0,L,\Delta\lambda_r\ [\mathrm m]", "reference resonance, length, shift"),
        (r"n_0,n_{g0}\ [1]", "reference effective, group index"),
        (r"m\in\mathbb Z", "longitudinal order"),
        (r"f,q(V),n_J(V)\ [1]", "coverage, table index, ring index"),
        (r"\alpha_J,\alpha_{\rm ref}\ [\mathrm{m^{-1}}]", "junction, total intrinsic loss")],
        ("resonance", r"n_0L=m\lambda_0,\qquad\phi_{\rm ref}(\lambda_0)=2\pi m"),
        ("junction baseline", r"\begin{aligned}n_{J,\rm ref}&=f q(V_{\rm ref})\\\alpha_{J,\rm ref}&=\alpha_J(V_{\rm ref})\end{aligned}"),
        ("reference shift", r"\Delta\lambda_r(T_{\rm ref},V_{\rm ref})=0"))

    topic("02-bus-segments", "bus attenuation", [
        (r"P(z),P_{{\rm in},k}\ [\mathrm W]", "waveguide, ring input power"),
        (r"z,d_k\ [\mathrm m]", "distance, segment length"),
        (r"\alpha_{\rm bus}\ [\mathrm{m^{-1}}]", "power attenuation"),
        (r"B_k\ [1]", "segment power transmission"),
        (r"k\in\{0,\ldots,N-1\}", "bus position"),
        (r"\bar q_k\ [1]", "mean ring through fraction")],
        ("propagation", r"\frac{dP}{dz}=-\alpha_{\rm bus}P,\qquad P(z+d)=P(z)e^{-\alpha_{\rm bus}d}"),
        ("segments", r"B_0=1,\qquad B_k=e^{-\alpha_{\rm bus}d_k}\ (k\ge1),\qquad B_{\rm out}=1"),
        ("power handoff", r"P_{{\rm in},k+1}=B_{k+1}\bar q_kP_{{\rm in},k}"))

    topic("03-thermal-tuning", "thermal tuning", [
        (r"T,T_{\rm ref},T_0\ [\mathrm K]", "ring, resonance anchor, slope anchor"),
        (r"\lambda_r,\Delta\lambda_T\ [\mathrm m]", "resonance, thermal shift"),
        (r"s(T),s_0\ [\mathrm{m/K}]", "local, anchor tuning sensitivity"),
        (r"\beta\ [\mathrm{K^{-1}}]", "relative tuning curvature"),
        (r"u\ [\mathrm K]", "integration temperature")],
        ("sensitivity", r"s(T)=\frac{d\lambda_r}{dT}=s_0[1+\beta(T-T_0)]"),
        ("shift", r"\Delta\lambda_T(T)=\int_{T_{\rm ref}}^T s(u)\,du"),
        ("integrated law", r"\begin{aligned}\Delta\lambda_T(T)=s_0\biggl[&(T-T_{\rm ref})\\&+\frac{\beta}{2}\bigl((T-T_0)^2-(T_{\rm ref}-T_0)^2\bigr)\biggr]\end{aligned}"),
        ("reference", r"\Delta\lambda_T(T_{\rm ref})=0,\qquad s(T_0)=s_0"))

    topic("04-rebase-thermal-law", "thermal rebasing", [
        (r"T_{\rm prev},T_0,\delta T\ [\mathrm K]", "previous, anchor, increment"),
        (r"s_0,s_{\rm loc}\ [\mathrm{m/K}]", "anchor, local sensitivity"),
        (r"\beta,\beta_{\rm loc}\ [\mathrm{K^{-1}}]", "anchor, local relative curvature"),
        (r"r_s\ [1]", "sensitivity ratio"),
        (r"\lambda_r\ [\mathrm m]", "resonance wavelength")],
        ("coefficients", r"\begin{aligned}r_s&=1+\beta(T_{\rm prev}-T_0)\ne0\\s_{\rm loc}&=s_0r_s,\qquad\beta_{\rm loc}=\frac{\beta}{r_s}\end{aligned}"),
        ("local shift", r"\begin{aligned}\delta T&=T-T_{\rm prev}\\\lambda_r(T)-\lambda_r(T_{\rm prev})&=s_{\rm loc}\left(\delta T+\frac{\beta_{\rm loc}}2\delta T^2\right)\end{aligned}"),
        ("curvature", r"s_{\rm loc}\beta_{\rm loc}=s_0\beta=\frac{d^2\lambda_r}{dT^2}"))

    topic("05-junction-device", "junction coverage", [
        (r"V\ [\mathrm V],\quad f\in[0,1]", "reverse bias, junction coverage"),
        (r"L,L_J\ [\mathrm m]", "ring, junction length"),
        (r"q(V),n_J(V)\ [1]", "table, ring index contribution"),
        (r"\alpha_{\rm FCA},\alpha_J\ [\mathrm{m^{-1}}]", "table, ring junction loss"),
        (r"c'(V)\ [\mathrm{F/m}]", "table capacitance density"),
        (r"r'(V)\ [\Omega\,\mathrm m]", "table resistance-length product"),
        (r"C_J\ [\mathrm F],\quad R_s\ [\Omega]", "junction capacitance, resistance")],
        ("coverage", r"L_J=fL"),
        ("optical properties", r"n_J(V)=fq(V),\qquad\alpha_J(V)=f\alpha_{\rm FCA}(V)"),
        ("electrical properties", r"C_J(V)=L_Jc'(V),\qquad R_s(V)=\begin{cases}r'(V)/L_J&L_J>0\\+\infty&L_J=0\end{cases}"))

    topic("06-voltage-tables", "junction table interpolation", [
        (r"V,V_i,V_{\rm ref}\ [\mathrm V]", "query, grid, reference bias"),
        (r"i\in\{0,\ldots,G-2\}", "grid interval"),
        (r"y_i,\mathcal I_y(V)", "table, interpolated property"),
        (r"w_i\ [1]", "interpolation weight"),
        (r"\chi_{\rm clip}\in\{0,1\}", "clipping flag"),
        (r"n_J\ [1],\quad\alpha_J\ [\mathrm{m^{-1}}]", "ring index, junction loss")],
        ("interior", r"w_i=\frac{V-V_i}{V_{i+1}-V_i},\qquad\mathcal I_y(V)=y_i+w_i(y_{i+1}-y_i)"),
        ("endpoints", r"\mathcal I_y(V)=\begin{cases}y_0&V\le V_0\\y_{G-1}&V\ge V_{G-1}\end{cases}"),
        ("clipping", r"\chi_{\rm clip}=\begin{cases}1&V<V_0\ \lor\ V>V_{G-1}\\0&V_0\le V\le V_{G-1}\end{cases}"),
        ("reference differences", r"\begin{aligned}\Delta n_J&=n_J(V)-n_J(V_{\rm ref})\\\Delta\alpha_J&=\alpha_J(V)-\alpha_J(V_{\rm ref})\end{aligned}"))

    topic("07-combined-shifts", "thermal and voltage shifts", [
        (r"T\ [\mathrm K],\quad V,V_{\rm ref}\ [\mathrm V]", "temperature, current and reference bias"),
        (r"\lambda_0,\lambda_r\ [\mathrm m]", "reference, shifted resonance"),
        (r"\Delta\lambda_T,\Delta\lambda_V\ [\mathrm m]", "thermal, voltage shift"),
        (r"\Delta n_J,n_{g0}\ [1]", "junction index difference, group index"),
        (r"n_{\rm eff}\ [1]", "effective index")],
        ("dispersion", r"n_g=n_{\rm eff}-\lambda\frac{dn_{\rm eff}}{d\lambda}"),
        ("voltage conversion", r"\Delta\lambda_V\simeq\frac{\lambda_0}{n_{g0}}\Delta n_J"),
        ("combined model", r"\begin{aligned}\Delta\lambda_r(T,V)&=\Delta\lambda_T(T)+\frac{\lambda_0}{n_{g0}}[n_J(V)-n_J(V_{\rm ref})]\\\lambda_r(T,V)&=\lambda_0+\Delta\lambda_r(T,V)\end{aligned}"))

    topic("08-loss-and-heat", "intrinsic loss and heat", [
        (r"T_r,T_h,T_{\rm ref}\ [\mathrm K]", "ring, heater, reference temperature"),
        (r"c_r,c_h\ [\mathrm{K^{-1}}]", "fractional temperature sensitivities"),
        (r"\alpha_{\rm ref},\Delta\alpha_J\ [\mathrm{m^{-1}}]", "reference loss, junction loss change"),
        (r"\alpha,\alpha_{\rm nh},\alpha_{\rm heat}\ [\mathrm{m^{-1}}]", "total, nonabsorptive, absorptive loss"),
        (r"s_\alpha,a,\eta\ [1]", "loss factor, field retention, heat share"),
        (r"L\ [\mathrm m]", "round-trip length")],
        ("temperature factor", r"s_\alpha=\exp\!\left[c_r(T_r-T_{\rm ref})+c_h(T_h-T_{\rm ref})\right]"),
        ("loss budget", r"\begin{aligned}\alpha&=\max\!\left(0,(\alpha_{\rm ref}+\Delta\alpha_J)s_\alpha\right)\\\alpha_{\rm nh}&=\alpha_{\rm nh,ref}\\\alpha_{\rm heat}&=\max(0,\alpha-\alpha_{\rm nh})\end{aligned}"),
        ("optical inputs", r"a=e^{-\alpha L/2},\qquad\eta=\begin{cases}\min(1,\alpha_{\rm heat}/\alpha)&\alpha>0\\0&\alpha=0\end{cases}"),
        ("base G26", r"c_r=c_h=0\quad\Longrightarrow\quad s_\alpha=1"))

    topic("09-shifted-phase-curve", "shifted reference phase", [
        (r"\lambda_L,\lambda_0\ [\mathrm m]", "laser, reference resonance"),
        (r"\Delta\lambda_r\ [\mathrm m]", "total resonance shift"),
        (r"x,d\ [\mathrm m]", "reference coordinate, offset"),
        (r"n_{\rm cold}(x)\ [1]", "reference effective index"),
        (r"L\ [\mathrm m],\quad m\in\mathbb Z", "ring length, resonance order"),
        (r"\phi,\psi_{\rm add}\ [\mathrm{rad}]", "round-trip phase, added coupler phase")],
        ("coordinates", r"x=\lambda_L-\Delta\lambda_r,\qquad d=x-\lambda_0"),
        ("translated curve", r"\phi(\lambda_L;\Delta\lambda_r)=\phi_{\rm ref}(x)=\frac{2\pi L n_{\rm cold}(x)}{x}"),
        ("resonance", r"\lambda_L=\lambda_0+\Delta\lambda_r\quad\Longrightarrow\quad x=\lambda_0,\quad\phi=2\pi m"),
        ("coupler phase", r"\psi_{\rm add}=0"))

    topic("10-wavelength-dispersion", "reference dispersion", [
        (r"x,\lambda_0,d=x-\lambda_0\ [\mathrm m]", "coordinate, reference, offset"),
        (r"n_0,n_{g0}\ [1]", "reference effective, group index"),
        (r"D\ [\mathrm{m^{-1}}]", "reference group-index slope"),
        (r"n_{\rm cold}(x),\widetilde n_g(x)\ [1]", "quadratic index, linear group index")],
        ("reference derivatives", r"D=\left.\frac{dn_g}{d\lambda}\right|_{\lambda_0},\qquad n'_0=\frac{n_0-n_{g0}}{\lambda_0},\qquad n''_0=-\frac{D}{\lambda_0}"),
        ("index model", r"n_{\rm cold}(x)=n_0+\frac{n_0-n_{g0}}{\lambda_0}d-\frac{D}{2\lambda_0}d^2"),
        ("group-index model", r"\widetilde n_g(x)=n_{g0}+Dd"),
        ("polynomial consistency", r"n_{\rm cold}(x)-x n'_{\rm cold}(x)=\widetilde n_g(x)+\frac{D}{2\lambda_0}d^2"))

    topic("11-shared-delay", "shared level-zero delay", [
        (r"b\in\{0,1\}", "data level"),
        (r"\lambda_L,\lambda_0,\Delta\lambda_b,x_b\ [\mathrm m]", "laser, reference, shift, coordinate"),
        (r"n_{g0}\ [1],\quad D\ [\mathrm{m^{-1}}]", "reference group index, slope"),
        (r"L\ [\mathrm m],\quad c\ [\mathrm{m/s}]", "ring length, vacuum light speed"),
        (r"\widetilde\tau_b,\tau_{{\rm used},b}\ [\mathrm s]", "estimated, retained delay")],
        ("per-level estimate", r"x_b=\lambda_L-\Delta\lambda_b,\qquad\widetilde\tau_b=\frac{L}{c}[n_{g0}+D(x_b-\lambda_0)]"),
        ("difference", r"\widetilde\tau_1-\widetilde\tau_0=-\frac{DL}{c}(\Delta\lambda_1-\Delta\lambda_0)"),
        ("implementation", r"\tau_{{\rm used},0}=\tau_{{\rm used},1}=\widetilde\tau_0"),
        ("equal-delay limit", r"D(\Delta\lambda_1-\Delta\lambda_0)=0\quad\Longrightarrow\quad\widetilde\tau_1=\widetilde\tau_0"))

    topic("12-two-data-levels", "two data levels", [
        (r"b\in\{0,1\},\quad\ell\in\{0,\ldots,M-1\}", "data level, laser line"),
        (r"V_b\ [\mathrm V],\quad T_r,T_h\ [\mathrm K]", "bias, shared temperatures"),
        (r"\phi_{b\ell}\ [\mathrm{rad}]", "round-trip phase"),
        (r"a_{b\ell},\eta_{b\ell}\ [1]", "retention, absorbed share"),
        (r"\Delta\lambda_b\ [\mathrm m],\quad C_{Jb}\ [\mathrm F]", "shift, capacitance"),
        (r"R_{sb}\ [\Omega],\quad p_b\ [1]", "resistance, bit probability"),
        (r"P_b,\overline P\ [\mathrm W]", "level, mean optical power")],
        ("shared state", r"T_{r,0}=T_{r,1}=T_r,\qquad T_{h,0}=T_{h,1}=T_h"),
        ("per-level inputs", r"\mathcal F(T_r,T_h,V_b,\lambda_\ell)\mapsto\left(\phi_{b\ell},a_{b\ell},\eta_{b\ell},\Delta\lambda_b,C_{Jb},R_{sb}\right)"),
        ("optics.py", r"p_0=1-p_1,\qquad\overline P=p_0P_0+p_1P_1"))

    topic("13-evanescent-overlap", "evanescent overlap", [
        (r"y,g,\lambda,\ell\ [\mathrm m]", "distance, gap, wavelength, length"),
        (r"E(y),E_0\ [\mathrm{V/m}]", "field, wall field"),
        (r"n_{\rm eff},n_{\rm clad}\ [1]", "effective, cladding index"),
        (r"\sigma,\kappa_c,F\ [\mathrm{m^{-1}}]", "decay, coupling, prefactor"),
        (r"C\ [1]", "cross-coupled power fraction")],
        ("field tail", r"E(y)=E_0e^{-\sigma y},\qquad\sigma(\lambda)=\frac{2\pi}{\lambda}\sqrt{n_{\rm eff}^2-n_{\rm clad}^2}"),
        ("overlap approximation", r"\kappa_c(\lambda,g)\simeq F(\lambda)e^{-\sigma(\lambda)g}"),
        ("weak transfer", r"|\kappa_c\ell|\ll1\quad\Longrightarrow\quad C\simeq(\kappa_c\ell)^2\simeq[F(\lambda)\ell]^2e^{-2\sigma(\lambda)g}"),
        ("fixed indices", r"\frac{dn_{\rm eff}}{d\lambda}=\frac{dn_{\rm clad}}{d\lambda}=0\quad\Longrightarrow\quad\frac{d\sigma}{d\lambda}=-\frac{\sigma}{\lambda}"))

    topic("14-coherent-power-exchange", "coherent power exchange", [
        (r"z,\ell\ [\mathrm m]", "propagation distance, interaction length"),
        (r"A(z),B(z)\ [1]", "normalized guide amplitudes"),
        (r"\kappa_c\ [\mathrm{m^{-1}}]", "spatial amplitude coupling"),
        (r"\Delta\beta\ [\mathrm{m^{-1}}]", "propagation mismatch"),
        (r"C\ [1],\quad i^2=-1", "transferred power fraction")],
        ("phase-matched, lossless", r"\Delta\beta=0,\qquad\frac{dA}{dz}=-i\kappa_c B,\qquad\frac{dB}{dz}=-i\kappa_c A"),
        ("unit launch", r"A(0)=1,\quad B(0)=0,\qquad A''+\kappa_c^2A=0"),
        ("fields", r"A(z)=\cos(\kappa_c z),\qquad B(z)=-i\sin(\kappa_c z)"),
        ("power", r"C=|B(\ell)|^2=\sin^2(\kappa_c\ell),\qquad|A|^2+|B|^2=1"))

    topic("15-local-coupling-model", "local coupling model", [
        (r"C(\lambda)=K^2(\lambda),\ C_0\ [1]", "coupled fraction, reference value"),
        (r"\lambda,\lambda_0,h\ [\mathrm m]", "laser, reference, extraction offset"),
        (r"\gamma_0\ [\mathrm{m^{-1}}]", "fractional wavelength slope"),
        (r"C_{\rm raw},C_{\rm used}\ [1]", "exponential, retained coupling")],
        ("local coefficients", r"C_0=C(\lambda_0),\qquad\gamma_0=\left.\frac{d\ln C}{d\lambda}\right|_{\lambda_0}=\frac{C'(\lambda_0)}{C_0}"),
        ("logarithmic tangent", r"\begin{aligned}\ln C(\lambda)&\simeq\ln C_0+\gamma_0(\lambda-\lambda_0)\\C_{\rm raw}(\lambda)&=C_0e^{\gamma_0(\lambda-\lambda_0)}\end{aligned}"),
        ("extracted slope", r"\gamma_0\simeq\frac{\ln C(\lambda_0+h)-\ln C(\lambda_0-h)}{2h}"),
        ("runtime bound", r"C_{\rm used}=\begin{cases}C_{\rm raw}&C_{\rm raw}<1\\1-10^{-15}&C_{\rm raw}\ge1\end{cases}"))

    topic("16-rigid-resonance-comb", "rigid resonance comb", [
        (r"m,m_*\in\mathbb Z", "resonance order, anchored order"),
        (r"\lambda_m^0,\lambda_m^{\rm rigid},\lambda_m^{\rm full}\ [\mathrm m]", "reference, translated, fuller resonance"),
        (r"\Delta\lambda_r,\epsilon_m\ [\mathrm m]", "common shift, comparison error"),
        (r"L(T)\ [\mathrm m],\quad n_{\rm eff}\ [1]", "temperature-dependent length, index")],
        ("runtime translation", r"\lambda_m^{\rm rigid}=\lambda_m^0+\Delta\lambda_r"),
        ("wavelength spacing", r"\lambda_{m+1}^{\rm rigid}-\lambda_m^{\rm rigid}=\lambda_{m+1}^0-\lambda_m^0"),
        ("fuller resonance condition", r"n_{\rm eff}(\lambda_m^{\rm full},T,V)L(T)=m\lambda_m^{\rm full}"),
        ("anchored comparison", r"\begin{aligned}\epsilon_m&=\lambda_m^{\rm full}-\lambda_m^{\rm rigid}\\\lambda_{m_*}^{\rm full}=\lambda_{m_*}^{\rm rigid}&\quad\Longrightarrow\quad\epsilon_{m_*}=0\\\epsilon_{m_*}=0&\quad\not\Longrightarrow\quad\epsilon_m=0\quad(m\ne m_*)\end{aligned}"))

    topic("17-optical-input-handoff", "optical-input handoff", [
        (r"T_r,T_h,T_{n+1},T_{\rm other}\ [\mathrm K]", "ring, heater, endpoint, remainder"),
        (r"V_0,V_1\ [\mathrm V],\quad\lambda_\ell\ [\mathrm m]", "binary biases, laser wavelength"),
        (r"\phi_{b\ell}\ [\mathrm{rad}]", "round-trip phase"),
        (r"a_{b\ell},\eta_{b\ell}\ [1]", "retention, absorbed share"),
        (r"K_{t\ell}^2,K_{d\ell}^2\ [1],\quad\tau_\ell\ [\mathrm s]", "coupler fractions, common delay"),
        (r"P_{\rm in},P_{\rm mon},P_{\rm abs}\ [\mathrm W]", "input, monitor, absorbed power"),
        (r"g\ [\mathrm{K/W}]", "substep optical-heat gain")],
        ("phase.py", r"\begin{aligned}\mathbf p_{b\ell}&=(\phi_{b\ell},a_{b\ell},\eta_{b\ell},K_{t\ell}^2,K_{d\ell}^2,\tau_\ell)\\\mathbf p&=\mathcal F_{\rm phase}(T_r,T_h,V_0,V_1,\{\lambda_\ell\})\end{aligned}"),
        ("optics.py", r"(\mathbf P_{\rm mon},\mathbf P_{\rm abs})=\mathcal F_{\rm optics}(\mathbf p,\mathbf P_{\rm in})"),
        ("thermal closure", r"T_{n+1}=T_{\rm other}+gP_{\rm abs}(T_{n+1})"))


def main():
    prepare()
    for folder in ("drawings", "equations", "latex"):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    expressions=[]
    for item in TOPICS:
        item["definitions"]=definitions(item["symbols"])
        item["groups"]=[(label,r"\displaystyle "+tex) for label,tex in item["groups"]]
        expressions.extend([item["definitions"],*(tex for _,tex in item["groups"])])
    process=subprocess.run(["node",str(ROOT/"tools/katex/tex2svg.js")],input=json.dumps(expressions),text=True,capture_output=True,check=True)
    typesetting=json.loads(process.stdout)
    if typesetting["errors"]:raise RuntimeError(typesetting["errors"])
    rendered={item["tex"]:item for item in typesetting["items"]}
    manifest=[];sources=[];failures=[];used_file_ids=set()
    for item in TOPICS:
        drawing=Drawing(item["slug"],item["title"],"","")
        drawing.elements.clear()
        drawing.text(90,65,item["title"])
        drawing.text(90,190,"symbols")
        files={}
        def formula(tex,x,y,maximum):
            result=rendered[tex]
            width,height=result["w"]*36,result["h"]*36
            if width>maximum:failures.append((item["slug"],round(width),maximum,tex))
            ident=hashlib.sha256(tex.encode()).hexdigest()
            used_file_ids.add(ident)
            svg=result["svg"].replace('<svg ',f'<svg width="{width}" height="{height}" color="#1e1e1e" ',1)
            ET.fromstring(svg)
            (OUT/"equations"/f"{ident}.svg").write_text(svg)
            files[ident]=dict(id=ident,mimeType="image/svg+xml",created=1793664000000,dataURL="data:image/svg+xml;base64,"+base64.b64encode(svg.encode()).decode())
            drawing.element("image",x,y,width,height,fileId=ident,status="saved",scale=[1,1],crop=None,customData=dict(latex=tex))
            return height
        formula(item["definitions"],90,275,940)
        y=190
        for label,tex in item["groups"]:
            drawing.text(1140,y,label)
            y+=82
            y+=formula(tex,1140,y,1700)+90
        data=scene(drawing.elements);data["files"]=files
        data["source"]="mrm-model phase.py equation companion - LaTeX vector images"
        (OUT/"drawings"/f"{item['slug']}.excalidraw").write_text(json.dumps(data,indent=2)+"\n")
        blocks=[rf"\section*{{{item['title']}}}",r"\[",item["definitions"],r"\]"]
        for label,tex in item["groups"]:blocks.extend([rf"\subsection*{{{label}}}",r"\[",tex,r"\]"])
        source="\n".join(blocks)+"\n"
        (OUT/"latex"/f"{item['slug']}.tex").write_text(source)
        sources.append(source)
        manifest.append(dict(slug=item["slug"],title=item["title"],subtitle="",code=""))
    if failures:raise RuntimeError(json.dumps(failures,indent=2))
    for obsolete in (OUT/"equations").glob("*.svg"):
        if obsolete.stem not in used_file_ids:obsolete.unlink()
    document="\\documentclass[11pt]{article}\n\\usepackage{amsmath,amssymb}\n\\usepackage[a4paper,margin=18mm]{geometry}\n\\begin{document}\n"+"\n\\clearpage\n".join(sources)+"\\end{document}\n"
    (OUT/"equations.tex").write_text(document)
    (OUT/"source.json").write_text(json.dumps(TOPICS,indent=2)+"\n")
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (OUT/"manifest.js").write_text("window.PHASE_DIAGRAMS = "+json.dumps(manifest,indent=2)+";\n")
    (OUT/"typesetting.json").write_text(json.dumps(dict(renderer="MathJax",version=typesetting["version"],sheets=len(TOPICS),expressions=len(expressions),errors=[]),indent=2)+"\n")
    print(f"rendered {len(expressions)} expressions on {len(TOPICS)} sheets")


if __name__=="__main__":
    main()
