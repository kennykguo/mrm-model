#!/usr/bin/env python3
# render equation-only companions to the visual atlas.
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path

from build_excalidraw_atlas import Drawing, scene

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/diagrams/optics-math"
TOPICS = []


def topic(slug, title, symbols, *groups):
    TOPICS.append(dict(slug=slug, title=title, symbols=symbols, groups=groups))


def definitions(rows):
    return (r"\begin{array}{ll}" +
            r"\\[0.6em]".join(symbol + r"&\text{" + label + "}" for symbol, label in rows) +
            r"\end{array}")


def prepare():
    topic("00-ring-model-reference", "ports and travelling fields", [
        (r"s_j,\ o_j\ [\sqrt{\mathrm W}]", "incoming, outgoing"),
        (r"u,v\ [\sqrt{\mathrm W}]", "CW, CCW fields"),
        (r"j\in\{0,1,2,3\}", "port index"),
        (r"k_j,t_j\ [1]", "cross, self coupling"),
        (r"\mathbf S\ [1]", "scattering matrix")],
        ("ports", r"\begin{aligned}0&\quad\text{input}\\1&\quad\text{through}\\2&\quad\text{drop}\\3&\quad\text{add}\end{aligned}"),
        ("field vectors", r"\mathbf x=\begin{pmatrix}u\\v\end{pmatrix},\qquad \mathbf o=\mathbf S\mathbf s"),
        ("power", r"P_j=|s_j|^2,\qquad P_{\mathrm{out},j}=|o_j|^2"),
        ("couplers", r"t_j^2+k_j^2=1"))

    topic("01-operating-side", "transmission and operating side", [
        (r"\lambda_L,\lambda_r\ [\mathrm m]", "laser, resonance"),
        (r"\gamma\ [\mathrm m]", "Lorentzian HWHM"),
        (r"\phi\ [\mathrm{rad}]", "round-trip phase"),
        (r"a\ [1]", "round-trip field retention"),
        (r"t_j,k_j\ [1]", "coupler coefficients"),
        (r"h_{\mathrm{th}}\ [1]", "through field ratio"),
        (r"\mathcal T,\mathcal D\ [1]", "through, drop fractions")],
        ("zero backscatter", r"r_b=0,\qquad r=a t_1t_2"),
        ("Airy response", r"\begin{aligned}h_{\mathrm{th}}(\phi)&=\frac{t_1-a t_2e^{-i\phi}}{1-r e^{-i\phi}}\\\mathcal T(\phi)&=|h_{\mathrm{th}}(\phi)|^2\\\mathcal D(\phi)&=\frac{k_1^2k_2^2a}{|1-r e^{-i\phi}|^2}\end{aligned}"),
        ("stable-side convention", r"\delta\lambda=\lambda_r-\lambda_L>0"),
        ("Lorentzian maximum slope", r"\begin{aligned}x&=\frac{\lambda_L-\lambda_r}{\gamma},\qquad\frac{\mathcal D}{\mathcal D_{\mathrm{pk}}}=\frac1{1+x^2}\\|x|_{\mathrm{max\ slope}}&=\frac1{\sqrt3},\qquad\frac{\mathcal D}{\mathcal D_{\mathrm{pk}}}=\frac34\end{aligned}"))

    topic("02-resonance-orders", "resonance orders", [
        (r"L\ [\mathrm m]", "ring length"),
        (r"n_{\mathrm{eff}},n_g\ [1]", "effective, group index"),
        (r"m,n\in\mathbb Z", "longitudinal orders"),
        (r"c\ [\mathrm{m/s}]", "vacuum light speed"),
        (r"T_{\mathrm{rt}}\ [\mathrm s]", "round-trip group delay")],
        ("round-trip resonance", r"\phi(\lambda_m)=\frac{2\pi n_{\mathrm{eff}}(\lambda_m)L}{\lambda_m}=2\pi m"),
        ("group delay and spacing", r"\begin{aligned}n_g&=n_{\mathrm{eff}}-\lambda\frac{dn_{\mathrm{eff}}}{d\lambda}\\T_{\mathrm{rt}}&=\frac{n_gL}{c}\\\mathrm{FSR}_{\nu}&\simeq\frac1{T_{\mathrm{rt}}},\qquad\mathrm{FSR}_{\lambda}\simeq\frac{\lambda^2}{n_gL}\end{aligned}"),
        ("order and residual phase", r"\begin{aligned}n&=\left\lfloor\frac{\phi}{2\pi}\right\rfloor\\\theta&=\phi-2\pi n\in[0,2\pi)\end{aligned}"))

    topic("03-coupling-and-interference", "coupling and interference", [
        (r"E_1,E_2\ [\sqrt{\mathrm W}]", "coherent fields"),
        (r"\psi\ [\mathrm{rad}]", "relative phase"),
        (r"t,k\ [1]", "coupler coefficients"),
        (r"b,u\ [\sqrt{\mathrm W}]", "bus, ring fields"),
        (r"\mathbf C\ [1]", "coupler matrix")],
        ("field addition", r"\begin{aligned}E_{\mathrm{out}}&=E_1+E_2e^{i\psi}\\P_{\mathrm{out}}&=|E_1|^2+|E_2|^2\\&\quad+2\operatorname{Re}\!\left(E_1^*E_2e^{i\psi}\right)\end{aligned}"),
        ("lossless coupler", r"\mathbf C=\begin{pmatrix}t&-ik\\-ik&t\end{pmatrix},\qquad\begin{pmatrix}b_{\mathrm{out}}\\u_{\mathrm{out}}\end{pmatrix}=\mathbf C\begin{pmatrix}b_{\mathrm{in}}\\u_{\mathrm{in}}\end{pmatrix}"),
        ("conservation", r"t^2+k^2=1,\qquad \mathbf C^{\dagger}\mathbf C=\mathbf I"))

    topic("04-backscatter-and-splitting", "backscatter and resonance splitting", [
        (r"r_{b,n}\ [1]", "round-trip backscatter"),
        (r"\rho_n,\rho\ [1]", "scatterer reflection"),
        (r"\tau\ [1]", "scatterer transmission"),
        (r"\theta\ [\mathrm{rad}]", "phase within an order"),
        (r"\phi\ [\mathrm{rad}]", "round-trip phase"),
        (r"g\ [\mathrm{rad/s}]", "mode coupling rate"),
        (r"\omega_0,\omega_\pm\ [\mathrm{rad/s}]", "resonant frequencies"),
        (r"T_{\mathrm{rt}}\ [\mathrm s]", "round-trip group delay")],
        ("equivalent scatterers", r"\begin{aligned}\phi&=2\pi n+\theta,\quad n\in\mathbb Z,\quad0\le\theta<2\pi\\\rho_n&=-\frac{i}{2}(-1)^n r_{b,n}^*,\qquad\tau=\sqrt{1-|\rho|^2}\end{aligned}"),
        ("between orders", r"\begin{aligned}\rho(\phi)&=\cos^2\!\left(\frac\theta4\right)\rho_n+\sin^2\!\left(\frac\theta4\right)\rho_{n+1}\\\frac{d\rho}{d\phi}&=\frac14\sin\!\left(\frac\theta2\right)(\rho_{n+1}-\rho_n)\end{aligned}"),
        ("near-resonance TCMT", r"g=\frac{r_{b,n}}{T_{\mathrm{rt}}},\qquad\omega_\pm=\omega_0\pm|g|"))

    topic("05-travel-and-coordinate", "propagation and coordinate direction", [
        (r"a\ [1]", "round-trip field retention"),
        (r"\phi\ [\mathrm{rad}]", "round-trip phase"),
        (r"\rho,\tau\ [1]", "scatterer reflection, transmission"),
        (r"u,v\ [\sqrt{\mathrm W}]", "CW, CCW fields"),
        (r"p\ [1]", "quarter-trip factor")],
        ("quarter trip", r"\xi=-\ln a,\qquad p=e^{-(\xi+i\phi)/4},\qquad\mathbf Q=\begin{pmatrix}p&0\\0&p^{-1}\end{pmatrix}"),
        ("physical scatterer", r"\begin{aligned}u_{\mathrm R}&=\tau u_{\mathrm L}-\rho^*v_{\mathrm R}\\v_{\mathrm L}&=\rho u_{\mathrm L}+\tau v_{\mathrm R}\end{aligned}"),
        ("clockwise-coordinate transfer", r"\begin{aligned}\mathbf S_{\mathrm{tr}}&=\frac1\tau\begin{pmatrix}1&-\rho^*\\-\rho&1\end{pmatrix}\\\mathbf H=\mathbf Q\mathbf S_{\mathrm{tr}}\mathbf Q&=\frac1\tau\begin{pmatrix}p^2&-\rho^*\\-\rho&p^{-2}\end{pmatrix}\end{aligned}"))

    topic("06-field-closure", "field closure and derivatives", [
        (r"\mathbf x\ [\sqrt{\mathrm W}]", "circulating field pair"),
        (r"\mathbf H,\mathbf C_j\ [1]", "half-ring, coupler transfers"),
        (r"s_j\ [\sqrt{\mathrm W}]", "incoming port fields"),
        (r"t_j,k_j\ [1]", "coupler coefficients"),
        (r"\phi\ [\mathrm{rad}]", "round-trip phase")],
        ("couplers and drive", r"\begin{aligned}\mathbf C_j&=\operatorname{diag}(t_j,t_j^{-1})\\\mathbf e&=\begin{pmatrix}-ik_2s_3\\ik_2s_2/t_2\end{pmatrix},\quad\mathbf d=\begin{pmatrix}-ik_1s_0\\ik_1s_1/t_1\end{pmatrix}\end{aligned}"),
        ("closed loop", r"\begin{aligned}\mathbf M&=\mathbf C_1\mathbf H\mathbf C_2\mathbf H\\\mathbf c&=\mathbf C_1\mathbf H\mathbf e+\mathbf d\\(\mathbf I-\mathbf M)\mathbf x&=\mathbf c\end{aligned}"),
        ("analytic phase derivative", r"\frac{d\mathbf x}{d\phi}=(\mathbf I-\mathbf M)^{-1}\!\left(\frac{d\mathbf c}{d\phi}+\frac{d\mathbf M}{d\phi}\mathbf x\right)"))

    topic("07-energy-and-heating", "energy, loss, and absorbed heat", [
        (r"s_j,o_j\ [\sqrt{\mathrm W}]", "incoming, outgoing fields"),
        (r"\eta\ [1]", "absorbed share of loss"),
        (r"\xi=-\ln a\ [1]", "round-trip field loss"),
        (r"\Pi_d\ [\mathrm W]", "segment-entry power sum"),
        (r"E_{d,q}\ [\sqrt{\mathrm W}]", "segment-entry field"),
        (r"d\in\{\mathrm{CW},\mathrm{CCW}\}", "travel direction"),
        (r"U_d\ [\mathrm J]", "stored energy"),
        (r"T_{\mathrm{rt}}\ [\mathrm s]", "round-trip group delay")],
        ("power balance", r"\sum_{j=0}^{3}|s_j|^2=\sum_{j=0}^{3}|o_j|^2+P_{\mathrm{loss}}"),
        ("quarter-ring integration", r"\begin{aligned}\ell_q&=1-e^{-\xi/2}\\u_q&=\frac{T_{\mathrm{rt}}}{4}\frac{1-e^{-\xi/2}}{\xi/2},\qquad\lim_{\xi\to0}u_q=\frac{T_{\mathrm{rt}}}{4}\\\Pi_d&=\sum_{q=1}^{4}|E_{d,q}|^2,\qquad U_d=u_q\Pi_d\end{aligned}"),
        ("loss and heat", r"\begin{aligned}P_{\mathrm{loss}}&=\ell_q(\Pi_{\mathrm{CW}}+\Pi_{\mathrm{CCW}})\\P_{\mathrm{abs}}&=\eta P_{\mathrm{loss}},\qquad0\le\eta\le1\end{aligned}"))

    topic("08-two-data-levels", "binary modulation and averaging", [
        (r"V_b\ [\mathrm V]", "bit-level voltage"),
        (r"p_b\ [1]", "bit probability"),
        (r"X_b", "level quantity"),
        (r"n_g,\Delta n\ [1]", "group index, index change"),
        (r"\lambda_{\mathrm{ref}}\ [\mathrm m]", "reference wavelength"),
        (r"D_\ell\ [\mathrm{rad/m}]", "phase sensitivity"),
        (r"\mathcal L_{b\ell}\ [1]", "intrinsic-loss fraction"),
        (r"P_{\mathrm{in},\ell}\ [\mathrm W]", "arriving line power"),
        (r"\eta_{b\ell}\ [1]", "absorbed loss share")],
        ("voltage tuning", r"\begin{aligned}b&\in\{0,1\},\qquad p_0=1-p_1\\\Delta\lambda_b&\simeq\frac{\lambda_{\mathrm{ref}}}{n_g}\Delta n(V_b)\\\phi_{b\ell}&=\phi_{\mathrm{ref},\ell}+D_\ell\Delta\lambda_b\end{aligned}"),
        ("power average", r"\overline X=p_0X_0+p_1X_1"),
        ("thermal input", r"\overline P_{\mathrm{abs}}=\sum_{\ell}\sum_{b=0}^{1}p_bP_{\mathrm{in},\ell}\eta_{b\ell}\mathcal L_{b\ell}"))

    topic("09-eight-ring-bus", "shared-bus propagation", [
        (r"k\in\{0,\ldots,7\}", "bus position"),
        (r"\ell\in\{0,\ldots,7\}", "laser line"),
        (r"B_k,B_{\mathrm{out}}\ [1]", "segment power transmission"),
        (r"\overline q_{k\ell}\ [1]", "mean through fraction"),
        (r"P_{\mathrm{in},k\ell}\ [\mathrm W]", "arriving power"),
        (r"P_{L,\ell}\ [\mathrm W]", "laser input power")],
        ("forward walk", r"\begin{aligned}P_{\mathrm{in},0\ell}&=B_0P_{L,\ell}\\P_{\mathrm{in},k+1,\ell}&=B_{k+1}\overline q_{k\ell}P_{\mathrm{in},k\ell}\\P_{\mathrm{out},\ell}&=B_{\mathrm{out}}P_{L,\ell}\prod_{k=0}^{7}B_k\overline q_{k\ell}\end{aligned}"),
        ("per-ring totals", r"\begin{aligned}P_{\mathrm{monitor},k}&=\sum_\ell\overline P_{\mathrm{drop},k\ell}\\P_{\mathrm{abs},k}&=\sum_\ell\overline P_{\mathrm{abs},k\ell}\end{aligned}"),
        ("independent data", r"\mathbb E\!\left[\prod_k q_k\right]=\prod_k\mathbb E[q_k]"))

    topic("10-reflections", "returned reflections", [
        (r"q_k,R_k\ [1]", "through, reflection fractions"),
        (r"B_k\ [1]", "segment power transmission"),
        (r"P_L\ [\mathrm W]", "input line power"),
        (r"M_b,\overline M\ [\mathrm W]", "level, mean through power"),
        (r"P_{\mathrm{in}}\ [\mathrm W]", "local arriving power"),
        (r"p_1\ [1]", "one-bit probability"),
        (r"\epsilon,E_{\mathrm{diag}}\ [1]", "pair sum, diagnostic")],
        ("single-return power", r"P_{\mathrm{return}}=P_L\sum_k\mathbb E[R_k]\prod_{j\le k}B_j^2\prod_{j<k}\mathbb E[q_j^2]"),
        ("same bit on both crossings", r"\begin{aligned}\mathbb E[q^2]&=\frac{(1-p_1)M_0^2+p_1M_1^2}{P_{\mathrm{in}}^2}\\&=\frac{M_0^2+(\overline M-M_0)(M_1+M_0)}{P_{\mathrm{in}}^2}\end{aligned}"),
        ("omitted-bounce diagnostic", r"\begin{aligned}r_k^{\mathrm{rms}}&=\sqrt{\mathbb E[R_k]}\\\epsilon&=\sum_{i<j}r_i^{\mathrm{rms}}r_j^{\mathrm{rms}}\\E_{\mathrm{diag}}&=\begin{cases}(1-\epsilon)^{-2}-1&\epsilon<1\\+\infty&\epsilon\ge1\end{cases}\end{aligned}"),
        ("NRZ scope", r"\text{data-averaged}\qquad O(\epsilon)"))

    topic("11-heat-paths", "thermal paths and observer temperatures", [
        (r"x_j\ [\mathrm K]", "thermal lag rise"),
        (r"R_j\ [\mathrm{K/W}]", "lag thermal resistance"),
        (r"\tau_j\ [\mathrm s]", "lag time constant"),
        (r"P_j\ [\mathrm W]", "lag heat input"),
        (r"T_{\mathrm b}\ [\mathrm K]", "background base"),
        (r"T_o\ [\mathrm K]", "observer temperature"),
        (r"w_{oj}\ [1]", "observer weights")],
        ("lag dynamics", r"\tau_j\dot x_j=-x_j+R_jP_j"),
        ("observer", r"T_o=T_{\mathrm b}+\sum_jw_{oj}x_j"),
        ("heat sources", r"P_j\in\left\{P_{\mathrm{heater}},P_{\mathrm{abs}},P_{\mathrm{nearby}}\right\}"))

    topic("12-self-heating-loop", "thermal tuning and optical feedback", [
        (r"T,T_{\mathrm{ref}}\ [\mathrm K]", "ring, reference temperature"),
        (r"s_{\mathrm{ref}}\ [\mathrm{m/K}]", "reference tuning slope"),
        (r"\beta\ [\mathrm{K}^{-1}]", "relative tuning curvature"),
        (r"D_\ell\ [\mathrm{rad/m}]", "phase sensitivity"),
        (r"c\ [\mathrm{m/s}]", "vacuum light speed"),
        (r"T_{\mathrm{rt}}\ [\mathrm s]", "round-trip group delay"),
        (r"\lambda_L\ [\mathrm m]", "fixed laser wavelength"),
        (r"P_{\mathrm{abs}}\ [\mathrm W]", "absorbed heat")],
        ("thermal tuning", r"\begin{aligned}\delta T&=T-T_{\mathrm{ref}}\\\Delta\lambda_r(T)&=s_{\mathrm{ref}}\delta T+\tfrac12s_{\mathrm{ref}}\beta\delta T^2\\s(T)&=s_{\mathrm{ref}}(1+\beta\delta T)\end{aligned}"),
        ("phase conversion", r"\begin{aligned}D_\ell&\simeq\frac{2\pi cT_{\mathrm{rt}}}{\lambda_{L,\ell}^2}\\\phi_{b\ell}(T)&=\phi_{b\ell,\mathrm{ref}}+D_\ell\Delta\lambda_r(T)\end{aligned}"),
        ("temperature sensitivity", r"\frac{dP_{\mathrm{abs}}}{dT}=s(T)\frac{\partial P_{\mathrm{abs}}}{\partial\lambda_r}"))

    topic("13-heating-cooling-vector-fields", "thermal vector field and stability", [
        (r"T,T_{\mathrm b}\ [\mathrm K]", "ring, base temperature"),
        (r"C\ [\mathrm{J/K}]", "thermal capacitance"),
        (r"R\ [\mathrm{K/W}]", "thermal resistance"),
        (r"P_h,P_{\mathrm{abs}}\ [\mathrm W]", "heater, absorbed power"),
        (r"f(T)\ [\mathrm{K/s}]", "temperature vector field"),
        (r"T_*\ [\mathrm K]", "equilibrium")],
        ("single thermal state", r"C\dot T=P_h+P_{\mathrm{abs}}(T)-\frac{T-T_{\mathrm b}}R"),
        ("heating gap", r"\begin{aligned}P_{\mathrm{req}}(T)&=\frac{T-T_{\mathrm b}}R-P_h\\\dot T=f(T)&=\frac{P_{\mathrm{abs}}(T)-P_{\mathrm{req}}(T)}C\end{aligned}"),
        ("equilibrium and local stability", r"\begin{aligned}f(T_*)&=0\\f'(T_*)&<0\quad\text{stable}\\f'(T_*)&>0\quad\text{unstable}\end{aligned}"),
        ("direction", r"f(T)>0\Rightarrow\dot T>0,\qquad f(T)<0\Rightarrow\dot T<0"))

    topic("14-folds-and-history", "folds and bistability", [
        (r"T,T_{\mathrm b}\ [\mathrm K]", "ring, base temperature"),
        (r"R\ [\mathrm{K/W}]", "thermal resistance"),
        (r"P_h,P_{\mathrm{pk}}\ [\mathrm W]", "heater, peak absorption"),
        (r"P_{\mathrm{abs}}\ [\mathrm W]", "absorbed optical power"),
        (r"\gamma\ [\mathrm m]", "resonance HWHM"),
        (r"G_{\mathrm{sh}}\ [\mathrm{m/W}]", "optical thermal tuning"),
        (r"s\ [\mathrm{m/K}]", "constant tuning slope"),
        (r"\lambda_0\ [\mathrm m]", "resonance without optical heat"),
        (r"\lambda_L,\lambda_r\ [\mathrm m]", "laser, hot resonance")],
        ("single-state steady residual", r"\mathcal E(T;P_h)=T-T_{\mathrm b}-R\bigl[P_h+P_{\mathrm{abs}}(T)\bigr]"),
        ("fold", r"\mathcal E=0,\qquad\frac{\partial\mathcal E}{\partial T}=0\quad\Longleftrightarrow\quad R\frac{dP_{\mathrm{abs}}}{dT}=1"),
        ("single-line Lorentzian - linear tuning", r"\begin{aligned}G_{\mathrm{sh}}&=Rs\\y&=\frac{\lambda_r-\lambda_L}{\gamma},\quad y_0=\frac{\lambda_0-\lambda_L}{\gamma}\\A&=\frac{G_{\mathrm{sh}}P_{\mathrm{pk}}}{\gamma}\\y-y_0&=\frac{A}{1+y^2}\end{aligned}"),
        ("Lorentzian folds", r"\begin{aligned}\mathcal F&=\{y_f\in\mathbb R:(1+y_f^2)^2+2Ay_f=0\}\\y_{0,\min}&=\min_{y_f\in\mathcal F}\left[y_f-\frac A{1+y_f^2}\right]\\y_{0,\max}&=\max_{y_f\in\mathcal F}\left[y_f-\frac A{1+y_f^2}\right]\end{aligned}"),
        ("bistable interval", r"A>\frac{8}{3\sqrt3},\quad y_{0,\min}<y_0<y_{0,\max},\quad\mathrm{FWHM}=2\gamma"))

    topic("15-thermal-memory", "thermal time discretization", [
        (r"x_n,x_{n+1}\ [\mathrm K]", "old, new lag rise"),
        (r"h,\tau\ [\mathrm s]", "step, thermal time constant"),
        (r"R\ [\mathrm{K/W}]", "thermal resistance"),
        (r"P\ [\mathrm W]", "held heat input"),
        (r"\alpha\ [1]", "memory coefficient"),
        (r"P_{\mathrm{abs}}(t)\ [\mathrm W]", "continuous optical heating")],
        ("held-power solution", r"\begin{aligned}\tau\dot x&=-x+RP\\\alpha&=e^{-h/\tau}\\x_{n+1}&=\alpha x_n+(1-\alpha)RP\end{aligned}"),
        ("continuously changing input", r"x_{n+1}=\alpha x_n+\frac R\tau\int_0^h e^{-(h-u)/\tau}P(t_n+u)\,du"),
        ("linear power ramp", r"\begin{aligned}P(t_n+u)&=P_n+(P_{n+1}-P_n)\frac uh\\b_0&=\frac{1-\alpha}{h/\tau}-\alpha,\quad b_1=1-\frac{1-\alpha}{h/\tau}\\x_{n+1}&=\alpha x_n+R(b_0P_n+b_1P_{n+1})\end{aligned}"),
        ("endpoint optical heating", r"P_{\mathrm{abs}}(t_n+u)\approx P_{\mathrm{abs}}(T_{n+1}),\qquad0\le u\le h"))

    topic("16-trial-temperature", "implicit endpoint temperature", [
        (r"T,T_{\mathrm{other}}\ [\mathrm K]", "trial, known contribution"),
        (r"g\ [\mathrm{K/W}]", "finite-step thermal gain"),
        (r"P(T)\ [\mathrm W]", "absorbed-power callback"),
        (r"R_j\ [\mathrm{K/W}]", "self-heating lag resistance"),
        (r"h,\tau_j\ [\mathrm s]", "step, lag time constant"),
        (r"P_{\max}\ [\mathrm W]", "power bound"),
        (r"B_{TT}\ [\mathrm{W/K^2}]", "absorption curvature bound"),
        (r"\varepsilon_T\ [\mathrm K]", "solve tolerance")],
        ("finite-step gain", r"g=\sum_{j\in\mathrm{self}}(1-e^{-h/\tau_j})R_j"),
        ("endpoint residual", r"\begin{aligned}F(T)&=T-T_{\mathrm{other}}-gP(T)\\F(T_{n+1})&=0\\F'(T)&=1-gP'(T)\end{aligned}"),
        ("bracket - nonnegative gain", r"g\ge0,\quad0\le P\le P_{\max}\quad\Rightarrow\quad T_{\mathrm{other}}\le T\le T_{\mathrm{other}}+gP_{\max}"),
        ("curvature-limited candidate", r"\begin{aligned}K_2&=|g|B_{TT}>0\\d_{\min}&=\frac{2|F|}{F'+\sqrt{(F')^2+2K_2|F|}}\\T_{\mathrm{trial}}&=T-\operatorname{sgn}(F)d_{\min}\end{aligned}"),
        ("Newton candidate - safeguarded in code", r"T^{(k+1)}_{\mathrm N}=T^{(k)}-\frac{F(T^{(k)})}{F'(T^{(k)})},\qquad |F(T)|\le\varepsilon_T"))

    topic("17-local-slope-and-bounds", "temperature sensitivity bounds", [
        (r"T,T_{\mathrm{ref}}\ [\mathrm K]", "trial, reference temperature"),
        (r"P_\ell\ [\mathrm W]", "absorbed line power"),
        (r"\Delta T_{\max}\ [\mathrm K]", "covered temperature interval"),
        (r"s_{\mathrm{ref}}\ [\mathrm{m/K}]", "reference tuning slope"),
        (r"\beta\ [\mathrm{K}^{-1}]", "relative tuning curvature"),
        (r"D_{\max}\ [\mathrm{rad/m}]", "largest phase sensitivity"),
        (r"B_\phi\ [\mathrm{W/rad}]", "phase-slope bound"),
        (r"B_{\phi\phi}\ [\mathrm{W/rad^2}]", "phase-curvature bound"),
        (r"B_T\ [\mathrm{W/K}]", "temperature-slope bound"),
        (r"B_{TT}\ [\mathrm{W/K^2}]", "temperature-curvature bound")],
        ("interval and tuning", r"\begin{aligned}|T-T_{\mathrm{ref}}|&\le\Delta T_{\max}\\s_{\max}&=|s_{\mathrm{ref}}|(1+|\beta|\Delta T_{\max})\\D_{\max}&=\max_\ell|D_\ell|\end{aligned}"),
        ("per-line chain rule", r"\begin{aligned}P&=\sum_\ell P_\ell\\\frac{dP_\ell}{dT}&=\frac{\partial P_\ell}{\partial\phi_\ell}\frac{d\phi_\ell}{dT}\\\frac{d^2P_\ell}{dT^2}&=\frac{\partial^2P_\ell}{\partial\phi_\ell^2}\left(\frac{d\phi_\ell}{dT}\right)^2+\frac{\partial P_\ell}{\partial\phi_\ell}\frac{d^2\phi_\ell}{dT^2}\end{aligned}"),
        ("converted envelopes", r"\begin{aligned}B_T&=B_\phi D_{\max}s_{\max}\\B_{TT}&=B_{\phi\phi}(D_{\max}s_{\max})^2\\&\quad+B_\phi D_{\max}|s_{\mathrm{ref}}\beta|\end{aligned}"),
        ("solver curvature", r"|P'(T)|\le B_T,\quad|P''(T)|\le B_{TT},\quad|F''(T)|\le |g|B_{TT}"))

    topic("17a-phase-envelopes", "phase envelopes", [
        (r"a,t_j,k_j\ [1]", "ring, coupler coefficients"),
        (r"\phi\ [\mathrm{rad}]", "round-trip phase"),
        (r"P_{\mathrm{in}}\ [\mathrm W]", "arriving power"),
        (r"\eta\ [1]", "absorbed share of loss"),
        (r"B_\phi\ [\mathrm{W/rad}]", "phase-slope bound"),
        (r"B_{\phi\phi}\ [\mathrm{W/rad^2}]", "phase-curvature bound"),
        (r"r_{b,n}\ [1]", "backscatter by order")],
        ("single line - one data level", r"\begin{aligned}r&=at_1t_2\in[0,1),\quad\mathcal N=k_1^2(1-a)(1+t_2^2a)\\\mathcal A(\phi)&=1-2r\cos\phi+r^2\\\mathcal L(\phi)&=\frac{\mathcal N}{\mathcal A(\phi)}\quad(r_b=0)\end{aligned}"),
        ("power and curvature envelopes", r"\begin{aligned}P_{\max}&=P_{\mathrm{in}}\eta\min\!\left\{1,\frac{1.05\mathcal N}{(1-r)^2}\right\}\\B_{\phi\phi}&=1.25P_{\mathrm{in}}\eta\frac{2r\mathcal N}{(1-r)^4}\end{aligned}"),
        ("slope envelope", r"\begin{aligned}c_*&=\frac{8r}{\sqrt{(1+r^2)^2+32r^2}+1+r^2}\\B_\phi&=1.10P_{\mathrm{in}}\eta\frac{2r\mathcal N\sqrt{1-c_*^2}}{(1-2rc_*+r^2)^2}\end{aligned}"),
        ("verified backscatter domain", r"\max_n|r_{b,n}|\le1"))

    topic("18-controller-target", "monitor target", [
        (r"y,\widehat y_{\mathrm{pk}}\ [\mathrm{codes}]", "reading, measured peak"),
        (r"y_*\ [\mathrm{codes}]", "monitor target"),
        (r"f\ [1]", "target hyperparameter"),
        (r"P_L\ [\mathrm W]", "laser power"),
        (r"\lambda_L,\lambda_r\ [\mathrm m]", "laser, resonance wavelength"),
        (r"T\ [\mathrm K]", "ring temperature")],
        ("target hyperparameter", r"y_*=f\widehat y_{\mathrm{pk}},\qquad f_{\mathrm{default}}=0.70"),
        ("changed peak", r"\frac{y_*}{\widehat y_{\mathrm{pk,new}}}=f\frac{\widehat y_{\mathrm{pk,old}}}{\widehat y_{\mathrm{pk,new}}}"),
        ("example", r"\widehat y_{\mathrm{pk,old}}=100,\quad y_*=70,\quad\widehat y_{\mathrm{pk,new}}=80,\quad\frac{70}{80}=87.5\%"),
        ("power drift", r"d\lambda_L=0,\quad dP_L\ne0\quad\Longrightarrow\quad dT\ne0,\quad d\lambda_r\ne0\quad\text{in general}"))

    topic("19-monitor-measurement", "photodiode and ADC", [
        (r"P_{\mathrm{monitor}}\ [\mathrm W]", "drop optical power"),
        (r"I_{\mathrm{PD}}\ [\mathrm A]", "photodiode current"),
        (r"y\ [\mathrm{codes}]", "ADC reading"),
        (r"\mathcal R\ [\mathrm{A/W}]", "responsivity"),
        (r"I_0,I_{\mathrm d}\ [\mathrm A]", "reference, dark current"),
        (r"E_a\ [\mathrm J]", "activation energy"),
        (r"k_B\ [\mathrm{J/K}]", "Boltzmann constant"),
        (r"T_{\mathrm{PD}},T_0\ [\mathrm K]", "diode, reference temperature"),
        (r"h_{\mathrm{int}},\tau_{\mathrm{ADC}}\ [\mathrm s]", "integration, latency"),
        (r"G\ [\mathrm{codes/A}]", "conversion gain"),
        (r"\nu\ [\mathrm{codes}]", "measurement noise"),
        (r"b_{\mathrm{ADC}}\ [1]", "ADC bit count")],
        ("light and dark current", r"\begin{aligned}I_{\mathrm{PD}}(t)&=\mathcal R P_{\mathrm{monitor}}(t)+I_{\mathrm d}(T_{\mathrm{PD}})\\I_{\mathrm d}(T)&=I_0\exp\!\left[-\frac{E_a}{k_B}\left(\frac1T-\frac1{T_0}\right)\right]\end{aligned}"),
        ("integration", r"\overline I(t)=\frac1{h_{\mathrm{int}}}\int_{t-h_{\mathrm{int}}}^{t}I_{\mathrm{PD}}(u)\,du"),
        ("measurement abstraction", r"y(t)=\operatorname{clip}_{[0,\,2^{b_{\mathrm{ADC}}}-1]}\!\left(\operatorname{round}\!\left[G\overline I(t-\tau_{\mathrm{ADC}})+\nu\right]\right)"),
        ("default timing", r"h_{\mathrm{int}}=2\,\mu\mathrm s,\quad\tau_{\mathrm{ADC}}=2\,\mu\mathrm s,\quad f_s=250\,\mathrm{kHz}"))

    topic("20-link-metrics", "optical modulation and BER", [
        (r"P_0,P_1\ [\mathrm W]", "zero, one output levels"),
        (r"L_{\mathrm{link}}\ [\mathrm{dB}]", "link loss"),
        (r"\mathrm{OMA}_{\mathrm{sens}}\ [\mathrm W]", "receiver sensitivity"),
        (r"\mathrm{BER}_{\mathrm{ref}}\ [1]", "reference error probability"),
        (r"N_{\mathrm{bits}}\ [1]", "transmitted bit count"),
        (r"\Lambda\ [1]", "expected error count"),
        (r"\mathrm{BER}_{\mathrm{floor}}\ [1]", "error floor")],
        ("modulation amplitude", r"\begin{aligned}\mathrm{OMA}_{\mathrm{tx}}&=P_1-P_0\\\mathrm{OMA}_{\mathrm{rx}}&=\mathrm{OMA}_{\mathrm{tx}}10^{-L_{\mathrm{link}}/10}\end{aligned}"),
        ("Gaussian receiver - positive levels", r"\begin{aligned}Q_{\mathrm{ref}}&=\sqrt2\operatorname{erfc}^{-1}(2\mathrm{BER}_{\mathrm{ref}})\\Q&=Q_{\mathrm{ref}}\frac{\mathrm{OMA}_{\mathrm{rx}}}{\mathrm{OMA}_{\mathrm{sens}}}\\\mathrm{BER}&=\max\!\left\{\mathrm{BER}_{\mathrm{floor}},\frac12\operatorname{erfc}\!\left(\frac Q{\sqrt2}\right)\right\}\end{aligned}"),
        ("degenerate levels", r"\min(\mathrm{OMA}_{\mathrm{rx}},\mathrm{OMA}_{\mathrm{sens}})\le0\quad\Rightarrow\quad\mathrm{BER}=\tfrac12"),
        ("error counter", r"\Lambda=\sum_n\mathrm{BER}_n N_{\mathrm{bits},n},\qquad N_{\mathrm{err}}\sim\operatorname{Poisson}(\Lambda)"))

    topic("21-model-and-evidence", "mathematical verification", [
        (r"\mathbf S\ [1]", "port scattering matrix"),
        (r"a\ [1]", "round-trip field retention"),
        (r"s_j,o_j\ [\sqrt{\mathrm W}]", "incoming, outgoing fields"),
        (r"P,P_{\mathrm{loss}}\ [\mathrm W]", "power, intrinsic loss"),
        (r"\phi,\epsilon\ [\mathrm{rad}]", "phase, perturbation")],
        ("power balance", r"\sum_j|s_j|^2-\sum_j|o_j|^2-P_{\mathrm{loss}}\simeq0"),
        ("reciprocity and passivity", r"\mathbf S=\mathbf S^{\mathsf T},\qquad\mathbf S^\dagger\mathbf S\preceq\mathbf I"),
        ("lossless limit", r"a=1\quad\Longrightarrow\quad\mathbf S^\dagger\mathbf S=\mathbf I"),
        ("derivative comparison", r"\frac{\partial P}{\partial\phi}\simeq\frac{P(\phi+\epsilon)-P(\phi-\epsilon)}{2\epsilon}"))

    topic("22-one-control-interval", "time scales", [
        (r"R_b\ [\mathrm{bits/s}]", "bit rate"),
        (r"T_b\ [\mathrm s]", "bit period"),
        (r"h\ [\mathrm s]", "plant step"),
        (r"h_c\ [\mathrm s]", "controller interval"),
        (r"N_{\mathrm{bits}}\ [1]", "bits per plant step"),
        (r"\mathbf A\ [\sqrt{\mathrm J}]", "optical mode amplitudes"),
        (r"V_b\ [\mathrm V]", "bit-level voltage"),
        (r"\tau_{\mathrm{opt}},\tau_{\mathrm{th}}\ [\mathrm s]", "optical, thermal time scales")],
        ("binary data", r"R_b=50\times10^9\,\mathrm{bits/s},\qquad T_b=R_b^{-1}=20\,\mathrm{ps}"),
        ("plant and controller", r"h=1\,\mu\mathrm s,\qquad h_c=10\,\mu\mathrm s=10h"),
        ("bits per step", r"N_{\mathrm{bits}}=R_bh=50{,}000"),
        ("thermal-scale optical solve", r"\tau_{\mathrm{opt}}\ll\tau_{\mathrm{th}},\qquad\left.\dot{\mathbf A}\right|_{T,V_b}\approx\mathbf0"))

    topic("23-stored-heat-and-cooling", "thermal storage and resistance", [
        (r"x=T-T_{\mathrm b}\ [\mathrm K]", "temperature rise"),
        (r"C\ [\mathrm{J/K}]", "thermal capacitance"),
        (r"R\ [\mathrm{K/W}]", "thermal resistance"),
        (r"P_{\mathrm{in}},P_{\mathrm{out}}\ [\mathrm W]", "heat flow rates"),
        (r"U_{\mathrm{th}}\ [\mathrm J]", "excess thermal energy")],
        ("stored energy", r"U_{\mathrm{th}}=Cx"),
        ("heat balance", r"\frac{dU_{\mathrm{th}}}{dt}=P_{\mathrm{in}}-P_{\mathrm{out}},\qquad P_{\mathrm{out}}=\frac xR"),
        ("first-order thermal response", r"C\dot x=P_{\mathrm{in}}-\frac xR,\qquad\tau=RC"),
        ("steady rise", r"x_*=RP_{\mathrm{in}}"))

    topic("24-ring-model-choice", "field model and near-resonance TCMT", [
        (r"a\ [1]", "round-trip field retention"),
        (r"k_1,k_2\ [1]", "cross-coupling amplitudes"),
        (r"T_{\mathrm{rt}}\ [\mathrm s]", "round-trip group delay"),
        (r"\kappa_i,\kappa_e,\kappa_d\ [\mathrm{s}^{-1}]", "energy decay rates"),
        (r"g\ [\mathrm{rad/s}]", "CW-CCW mode coupling"),
        (r"r_{b,n}\ [1]", "round-trip backscatter"),
        (r"\phi\ [\mathrm{rad}]", "round-trip phase"),
        (r"n\in\mathbb Z", "longitudinal order"),
        (r"A_+,A_-\ [\sqrt{\mathrm J}]", "TCMT mode amplitudes"),
        (r"d_+,d_-\ [\sqrt{\mathrm J}/\mathrm s]", "mode drives")],
        ("rate conversion", r"\begin{aligned}\kappa_i&=-\frac{2\ln a}{T_{\mathrm{rt}}}\\\kappa_e&=-\frac{\ln(1-k_1^2)}{T_{\mathrm{rt}}},\quad\kappa_d=-\frac{\ln(1-k_2^2)}{T_{\mathrm{rt}}}\\\kappa&=\kappa_i+\kappa_e+\kappa_d\end{aligned}"),
        ("near-resonance variables", r"\Delta_n=\frac{\phi-2\pi n}{T_{\mathrm{rt}}},\quad g=\frac{r_{b,n}}{T_{\mathrm{rt}}},\quad d=\frac\kappa2+i\Delta_n"),
        ("archived steady-state TCMT", r"\begin{pmatrix}d&ig\\ig^*&d\end{pmatrix}\begin{pmatrix}A_+\\A_-\end{pmatrix}=\begin{pmatrix}d_+\\d_-\end{pmatrix}"),
        ("energy normalization", r"U_{\mathrm{CW}}=|A_+|^2,\qquad U_{\mathrm{CCW}}=|A_-|^2"))

    topic("25-peak-seeking-and-lock-loss", "self-stabilization and fold sensitivity", [
        (r"P_{\mathrm{abs}}\ [\mathrm W]", "absorbed optical power"),
        (r"T,T_{\mathrm b}\ [\mathrm K]", "ring, base temperature"),
        (r"P_h\ [\mathrm W]", "heater power"),
        (r"R\ [\mathrm{K/W}]", "thermal resistance"),
        (r"\lambda_r,\lambda_L\ [\mathrm m]", "resonance, laser wavelength"),
        (r"\mathcal S\ [1]", "thermal suppression factor")],
        ("single-state steady sensitivity", r"\frac{dT_*}{dP_h}=\frac R{1-RP'_{\mathrm{abs}}(T_*)}"),
        ("negative optical feedback", r"\begin{aligned}\lambda_r&>\lambda_L,\qquad P'_{\mathrm{abs}}(T_*)<0\\\mathcal S&=1-RP'_{\mathrm{abs}}(T_*)>1\end{aligned}"),
        ("nominal suppression", r"\mathcal S\sim3\text{--}7,\qquad\frac{dT_*}{dP_h}=\frac R{\mathcal S}"),
        ("fold", r"1-RP'_{\mathrm{abs}}(T_*)=0,\qquad\left|\frac{dT_*}{dP_h}\right|\to\infty"))


def main():
    prepare()
    for name in ["drawings", "latex", "equations", "previews"]:
        (OUT/name).mkdir(parents=True, exist_ok=True)
    expressions = []
    for item in TOPICS:
        item["definitions"] = definitions(item["symbols"])
        item["groups"] = [(label, r"\displaystyle "+tex) for label,tex in item["groups"]]
        expressions.append(item["definitions"])
        expressions.extend(tex for _,tex in item["groups"])
    proc = subprocess.run(["node", str(ROOT/"tools/katex/tex2svg.js")], input=json.dumps(expressions),
                          text=True, capture_output=True, check=True)
    result = json.loads(proc.stdout)
    if result["errors"]:
        raise RuntimeError(result["errors"])
    rendered = {item["tex"]: item for item in result["items"]}
    manifest, all_tex = [], []
    failures=[]
    for item in TOPICS:
        drawing=Drawing(item["slug"],item["title"],"","")
        drawing.elements.pop()  # no explanatory subtitle on equation sheets.
        drawing.text(90,190,"symbols")
        files={}
        def formula(tex,x,y,max_width):
            render=rendered[tex]
            width,height=render["w"]*36,render["h"]*36
            if width>max_width:failures.append((item["slug"],round(width),max_width,tex))
            ident=hashlib.sha256(tex.encode()).hexdigest()
            svg=render["svg"].replace('<svg ',f'<svg width="{width}" height="{height}" color="#1e1e1e" ')
            (OUT/"equations"/f"{ident}.svg").write_text(svg)
            files[ident]=dict(id=ident,mimeType="image/svg+xml",created=1793188800000,
                             dataURL="data:image/svg+xml;base64,"+base64.b64encode(svg.encode()).decode())
            drawing.element("image",x,y,width,height,fileId=ident,status="saved",scale=[1,1],crop=None,
                            customData=dict(latex=tex))
            return height
        formula(item["definitions"],90,280,810)
        y=190
        for label,tex in item["groups"]:
            drawing.text(1030,y,label)
            y+=82
            y+=formula(tex,1030,y,1500)+85
        data=scene(drawing.elements);data["files"]=files
        (OUT/"drawings"/f"{item['slug']}.excalidraw").write_text(json.dumps(data,indent=2)+"\n")
        blocks=[rf"\section*{{{item['slug'][:2]} - {item['title']}}}","",r"\[",item["definitions"],r"\]",""]
        for label,tex in item["groups"]:
            blocks.extend([rf"\subsection*{{{label}}}",r"\[",tex,r"\]",""])
        source="\n".join(blocks)
        (OUT/"latex"/f"{item['slug']}.tex").write_text(source)
        all_tex.append(source)
        manifest.append(dict(slug=item["slug"],title=item["title"],subtitle="",code=""))
    if failures:
        print(json.dumps(failures,indent=2))
        raise RuntimeError("equation width exceeds the layout")
    (OUT/"equations.tex").write_text("\n\\clearpage\n".join(all_tex))
    (OUT/"source.json").write_text(json.dumps(TOPICS,indent=2)+"\n")
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (OUT/"manifest.js").write_text("window.OPTICS_DIAGRAMS = "+json.dumps(manifest,indent=2)+";\n")
    (OUT/"typesetting.json").write_text(json.dumps(dict(renderer="MathJax",version=result["version"],expressions=len(expressions),errors=[]),indent=2)+"\n")
    print(f"rendered {len(expressions)} expressions across {len(TOPICS)} sheets")


if __name__=="__main__":
    main()
