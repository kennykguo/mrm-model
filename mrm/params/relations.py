# solve the algebraic relations that keep the ring parameters consistent.
# `solve` updates a `RelationContext` in dependency order.
# `residuals` checks the resulting equations over the batch.
# all values are SI, even when their config path ends in nm, um, or mW.
# arrays broadcast over worlds and rings.
# the relation groups connect geometry and free spectral range, temperature and resonance, heater power and tuning, optical loss and Q, then I/O latency.
# an anchor is an input value.
# a free member is an absent value or one marked for derivation.
# each solver needs enough anchors to compute its free members.
# too few inputs raise `UnderdeterminedError`.
# conflicting inputs raise `OverdeterminedError`.
# an explicit relative tolerance permits recomputing the member with the weakest provenance.
# without a stated tolerance, fully specified inputs must agree within 1e-9.
# `realize` separately requires final closure residuals at or below 1e-12.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from mrm.params.symbols import SYMBOLS
from mrm.units import C0

__all__ = ["Relation", "RELATIONS", "RELATION_NAMES", "OverdeterminedError", "UnderdeterminedError",
           "RelationContext", "Derivation", "solve", "residuals", "LATENCY_OWNERS", "integral_s", "solve_t_for_shift"]

_TWO_PI = 2.0 * math.pi
# relative residual accepted as "every member given and consistent" when the preset states no tolerance_rel (-)
_EXACT_RTOL = 1e-9

# member config paths
# P_<name> is the config path of one relation member.
# Inside a RelationContext every value is SI, whatever unit the key suffix names (for example P_RADIUS holds the radius in m, not um).

# geometry and optics (R3)
P_RADIUS = "ring.geometry.radius_um"                        # ring radius R (m)
P_NG = "ring.optics.n_g"                                    # group index
P_NEFF = "ring.optics.n_eff"                                # effective (phase) index
P_FSR_NM = "ring.optics.fsr_nm"                             # FSR in wavelength at the line (m)
P_FSR_THZ = "ring.optics.fsr_thz"                           # FSR in frequency (Hz)
P_OFFSET = "ring.optics.resonance_offset_nm"                # cold resonance offset blue of the line (m in SI)
P_OFFSET_T = "ring.optics.resonance_offset_ref_temp_c"      # temperature at which the offset is stated (K in SI)
P_ORDER = "ring.optics.order"                               # longitudinal mode order m
# thermo-optic slope (R1) — s(T) = s0 * (1 + beta * (T - T0))
P_S0 = "ring.thermal.dlambda_dt_pm_per_k"                   # s0 - dlambda/dT at T0, the thermo-optic anchor (m/K)
P_T0 = "ring.thermal.dlambda_dt_ref_ring_temp_c"            # T0 - ring temperature at which s0 is stated (K)
P_BETA = "ring.thermal.dlambda_dt_beta_per_k"               # beta - relative change of s per kelvin (1/K)
P_DNDT = "ring.optics.dn_eff_dt_per_k"                      # dn_eff/dT at T0 (1/K) - derived from s0, never given
P_ALPHA_L = "ring.optics.alpha_l_per_k"                     # linear thermal expansion of the ring length (1/K)
P_ALPHA_L_SLOPE = "ring.optics.alpha_l_slope_per_k2"        # dalpha_L/dT (1/K^2)
# heater (R2)
P_PFSR = "ring.heater.p_fsr_mw"                             # P_FSR - heater power for one FSR of tuning (W)
P_PFSR_T = "ring.heater.p_fsr_ref_temp_c"                   # temperature at which P_FSR was measured (K)
P_PFSR_LAM = "ring.heater.p_fsr_ref_wavelength_nm"          # wavelength of the line on which P_FSR was measured (m)
P_EFF = "ring.heater.efficiency_nm_per_mw"                  # heater tuning efficiency (m/W)
P_Z = "ring.thermal.z_heater_ring_k_per_mw"                 # Z_heater->ring(0) - DC ring temperature rise per watt of heater power (K/W)
P_TLOCK = "ring.thermal.t_ring_lock_c"                      # ring temperature at lock (K)
# quality factors and loss (R4)
P_QL = "ring.optics.q_loaded"                               # loaded Q
P_QI = "ring.optics.q_intrinsic"                            # intrinsic Q (from the intrinsic loss alone)
P_QT = "ring.optics.q_through"                              # Q of the through (input) coupler
P_QD = "ring.optics.q_drop"                                 # Q of the drop coupler
P_XD = "ring.optics.kappa_d_over_kappa"                     # x_d = Q_L / Q_d, the drop share of the total decay
P_ETA = "ring.optics.heat_fraction"                         # eta - share of the intrinsic loss that heats the ring
# loss-budget components — power attenuation alpha, in 1/m in SI (the keys are in dB/cm)
_LOSS_BUDGET = "ring.optics.loss_budget."
P_A_FCA = _LOSS_BUDGET + "alpha_fca_db_per_cm"
P_A_OTH = _LOSS_BUDGET + "alpha_other_abs_db_per_cm"
P_A_SCA = _LOSS_BUDGET + "alpha_scatter_db_per_cm"
P_A_RAD = _LOSS_BUDGET + "alpha_bend_radiation_db_per_cm"
LOSS_COMPONENTS = (P_A_FCA, P_A_OTH, P_A_SCA, P_A_RAD)
# the line wavelength and the cold-resonance wavelength (realize records how they come from the channel plan)
_LAMBDA_LINE = "derived.optics.lambda_line_m"
_LAMBDA_RES_COLD = "derived.optics.lambda_res_cold_m"

# R5 assigns one config key to each I/O channel latency.
# temperature sensors use the temp_sensors table.
# the controller declares Controller.latency_ticks in code.
LATENCY_OWNERS = {
    "tx.monitor": "ring.monitor.adc_latency_s",
    "tx.heater": "ring.heater.latency_s",
    "tx.bias": "ring.bias_dac.latency_s",
    "heater.readback": "ring.heater.readback.latency_s",
    "rx.error_count": "datapath.error_count_latency_s",
}


# shared LaTeX formulas use the symbols defined in mrm.params.symbols
_TEX_R = r"R = \frac{c}{2\pi\, n_g\, \Delta f_{\mathrm{FSR}}}"
_TEX_NG = r"n_g = \frac{c}{2\pi R\, \Delta f_{\mathrm{FSR}}}"
_TEX_FSR = r"\Delta f_{\mathrm{FSR}} = \frac{c}{n_g L}"
_TEX_EFF = r"\eta_{\mathrm{tune}} = \frac{\Delta\lambda_{\mathrm{FSR,meas}}}{P_{\mathrm{FSR}}}"
_TEX_PFSR = r"P_{\mathrm{FSR}} = \frac{\Delta\lambda_{\mathrm{FSR,meas}}}{\eta_{\mathrm{tune}}}"
_TEX_QI = r"Q_{\mathrm{int}} = \frac{2\pi\, n_g}{\lambda_{r0}\,\alpha_{\mathrm{int}}}"
_TEX_ETA = r"\eta_{\mathrm{heat}} = \frac{\alpha_{\mathrm{FCA}} + \alpha_{\mathrm{other}}}{\alpha_{\mathrm{int}}}"
_TEX_QL = r"\frac{1}{Q_L} = \frac{1}{Q_{\mathrm{int}}} + \frac{1}{Q_t} + \frac{1}{Q_d}"
_TEX_OMEGA = r"\omega = 2\pi c/\lambda_{r0}"


# errors and records

# given relation members disagree beyond the allowed relative tolerance.
class OverdeterminedError(ValueError):
    pass


# a relation has more free members than equations.
class UnderdeterminedError(ValueError):
    pass


# how one derived quantity was computed (`mrm params explain` prints it).
@dataclass
class Derivation:
    path: str                   # config path or derived.* name of the value
    relation: str               # name of the relation that computed it
    inputs: tuple[str, ...]     # paths of the values it was computed from (its provenance is the weakest of theirs)
    formula: str                # the formula in plain text
    note: str = ""
    # the same formula in LaTeX (symbols of mrm.params.symbols - the variable map renders it) and the number of the textbook equation it is, or follows from (lightmatter-capstone chapters 0-5 - "" when it is not there)
    formula_tex: str = ""
    eq: str = ""


# SI values, known paths, and options shared by the relation solvers.
# `vals` maps config paths and derived names to arrays, usually shaped `(B, N)` for B worlds and N rings.
# `given` includes original inputs and values already solved.
# `prov` stores provenance.
# each computed value also gets a `Derivation` record for the parameter table and explain command.
# geometry holds come from the nominal device in each world.
# they keep the mask radius and mode order fixed while device errors change optical properties.
@dataclass
class RelationContext:
    vals: dict[str, np.ndarray]
    given: set[str]
    prov: dict[str, str]
    order2: bool = True                     # fidelity.thermo_optic_order == 2 (False - beta and dalpha_L/dT are 0)
    heater_anchor: str = "lock_slope"       # fidelity.heater_anchor - "lock_slope" or "fsr_integral" (R2_heater_z)
    tolerance: dict[str, float] = field(default_factory=dict)   # relation name -> stated tolerance_rel
    dt: float | None = None                 # plant tick sim.dt_plant_s (s), for the latencies in ticks (R5)
    latency_scale: np.ndarray | float = 1.0  # per-world factor on every I/O latency (uncertainty)
    derivations: dict[str, Derivation] = field(default_factory=dict)
    # relation name -> how an overdetermined relation was settled (the member that was recomputed)
    notes: dict[str, str] = field(default_factory=dict)
    # realize supplies anchors from the nominal-geometry relation solve.
    # the radius anchor uses m, and every sampled device retains the same integer order.
    # anchor arrays have shape (B,N).
    # kappa2_through and kappa2_drop are power fractions (1).
    # alpha_i_per_m uses 1/m, and dln_kappa2_through and dln_kappa2_drop use 1/m.
    radius_hold: np.ndarray | None = None
    order_hold: np.ndarray | None = None
    coupling_anchor: dict | None = None

    # true when path `p` has a value (given in the config or already computed).
    def known(self, p: str) -> bool:
        return p in self.vals and p in self.given

    # store an SI array and the record of how it was computed.
    # `formula` is plain text.
    # `tex` is the same equation in LaTeX.
    # `eq` is an optional textbook equation number.
    # record the input paths for provenance.
    def set(self, p: str, v, relation: str, inputs, formula: str, note: str = "", *, tex: str, eq: str = "") -> None:
        self.vals[p] = np.asarray(v)
        self.given.add(p)
        self.derivations[p] = Derivation(p, relation, tuple(inputs), formula, note, tex, eq)


# one group of related values with a solver and a residual check.
# `members` lists config paths.
# the solver fills free members in place.
# the residual reports the largest relative equation error over the batch.
@dataclass(frozen=True)
class Relation:
    name: str
    members: tuple[str, ...]
    solver: Callable[[RelationContext], None]
    residual: Callable[[dict], float]
    doc: str


# helpers

# max over the batch of |a - b| / |b| (0.0 for empty arrays).
def _max_rel_diff(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    denominator = np.maximum(np.abs(b), 1e-300)
    return float(np.max(np.abs(a - b) / denominator)) if a.size else 0.0


# the member with the weakest provenance (a member without one counts as 'assumed').
def _weakest_member(ctx: RelationContext, members) -> str:
    from mrm.params.config import provenance_rank
    return max(members, key=lambda m: provenance_rank(ctx.prov.get(m, "assumed")))


# check a relation with every member given and resolve allowed disagreement.
# without a stated tolerance, accept residuals at or below 1e-9 without changing the inputs.
# with a stated tolerance, recompute the weakest member whenever the residual is inside it.
# record that decision.
# otherwise raise an error.
def _resolve_overdetermined(ctx: RelationContext, name: str, resid: float, members,
                            recompute: Callable[[str], None]) -> None:
    tolerance = ctx.tolerance.get(name)
    if tolerance is None:
        if resid <= _EXACT_RTOL:
            return
        raise OverdeterminedError(
            f"{name}: every member is given ({', '.join(members)}) and the relative residual is {resid:.3g}; leave one "
            f"member null (or tag it derived), or state relations.{name}.tolerance_rel")
    if resid > tolerance:
        raise OverdeterminedError(f"{name}: relative residual {resid:.3g} exceeds tolerance_rel {tolerance:g} "
                                  f"(members {', '.join(members)})")
    weakest = _weakest_member(ctx, members)
    recompute(weakest)
    derivation = ctx.derivations.get(weakest)
    if derivation is not None:
        derivation.note = (f"overdetermined within tolerance_rel={tolerance:g} (residual {resid:.3g}); recomputed this member, the "
                  f"weakest provenance ({ctx.prov.get(weakest)})")
    ctx.notes[name] = f"residual {resid:.3g} <= tolerance_rel {tolerance:g}; recomputed {weakest}"


# return resonance shift in m from reference temperature t0 to t.
# integrate `s(T) = s0 * (1 + beta * (T - t0))` analytically.
# temperatures are in K, s0 is in m/K, and beta is in 1/K.
def integral_s(t, t0, s0, beta):
    x = t - t0
    return s0 * (x + 0.5 * beta * x * x)


# return the temperature in K that produces the requested wavelength shift in m.
# integrate the linear temperature sensitivity from `t_start`.
# writing `x = T - t0` gives `beta*x*x/2 + x = c`.
# use the root `x = 2*c / (1 + sqrt(1 + 2*beta*c))` to avoid subtracting nearly equal numbers when beta is small.
# reject shifts with a negative discriminant.
def solve_t_for_shift(t_start, shift, t0, s0, beta):
    c = (integral_s(t_start, t0, s0, beta) + shift) / s0
    discriminant = 1.0 + 2.0 * beta * c
    if np.any(discriminant < 0):
        raise ValueError("no temperature reaches the requested shift: s(T) changes sign first (check beta)")
    return t0 + 2.0 * c / (1.0 + np.sqrt(discriminant))


# return the given FSR in Hz and its config path, or two None values.
# prefer the frequency input.
# convert a wavelength FSR at the assigned laser line using `FSR_hz = c * FSR_m / wavelength_m**2`.
def _fsr_hz_from_given(ctx: RelationContext):
    wavelength_m = ctx.vals["derived.optics.lambda_line_m"]
    if ctx.known(P_FSR_THZ):
        return ctx.vals[P_FSR_THZ], P_FSR_THZ
    if ctx.known(P_FSR_NM):
        return C0 * ctx.vals[P_FSR_NM] / (wavelength_m * wavelength_m), P_FSR_NM
    return None, None


# R3 geometry

# solve geometry and FSR, then derive length, mode order, and phase index.
# `FSR_hz = c / (n_g * 2*pi*R)` leaves one of radius, group index, or FSR free.
# with sampled geometry, hold the nominal mask radius.
# each device's FSR then follows its own group index.
# the cold resonance is line wavelength minus offset.
# round `n_eff * L / lambda_res` to get the nominal integer mode order.
# set `n_eff_phase = order * lambda_res / L` so this order resonates exactly at the configured cold resonance.
# the nominal correction is at most `lambda_res / (2*L)`, about 0.016.
# sampled devices keep the nominal order because their geometry Jacobians use a fixed reference radius and order.
def _solve_r3_geometry(ctx: RelationContext) -> None:
    name = "R3_geometry"
    line_wavelength_m = ctx.vals["derived.optics.lambda_line_m"]
    if ctx.known(P_FSR_THZ) and ctx.known(P_FSR_NM):
        fsr_frequency_hz = ctx.vals[P_FSR_THZ]
        fsr_wavelength_hz = C0 * ctx.vals[P_FSR_NM] / (line_wavelength_m * line_wavelength_m)
        fsr_disagreement = _max_rel_diff(fsr_frequency_hz, fsr_wavelength_hz)
        if fsr_disagreement > max(ctx.tolerance.get(name, 0.0), _EXACT_RTOL):
            raise OverdeterminedError(f"{name}: fsr_thz and fsr_nm are both given and disagree by {fsr_disagreement:.3g}")
    fsr, fsr_key = _fsr_hz_from_given(ctx)
    have_r, have_ng = ctx.known(P_RADIUS), ctx.known(P_NG)
    free = [m for m, h in ((P_RADIUS, have_r), (P_NG, have_ng), ("fsr", fsr is not None)) if not h]
    if len(free) > 1:
        raise UnderdeterminedError(f"{name}: {len(free)} of radius, n_g, FSR are free ({free}); give all but one")
    held = ctx.radius_hold is not None and free == [P_RADIUS]
    if held:
        # hold the nominal mask radius within each world.
        # let the device's FSR follow its own group index.
        ctx.set(P_RADIUS, np.asarray(ctx.radius_hold, float), name, (P_NG, fsr_key),
                "R = c/(2 pi n_g FSR) of the nominal-geometry device, held for every device of the world",
                "a geometry error moves n_g at a fixed mask radius, so the FSR follows the device's n_g",
                tex=_TEX_R + r"\ \text{(nominal-geometry device, held for the world)}",
                eq="5.14")
    elif not free:
        L = _TWO_PI * ctx.vals[P_RADIUS]
        residual = _max_rel_diff(fsr * ctx.vals[P_NG] * L / C0, 1.0)

        def recompute(m):
            current_length_m = _TWO_PI * ctx.vals[P_RADIUS]
            if m == P_RADIUS:
                ctx.set(P_RADIUS, C0 / (ctx.vals[P_NG] * fsr * _TWO_PI), name, (P_NG, fsr_key), "R = c/(2 pi n_g FSR)",
                        tex=_TEX_R, eq="5.14")
            elif m == P_NG:
                ctx.set(P_NG, C0 / (fsr * current_length_m), name, (P_RADIUS, fsr_key), "n_g = c/(FSR * 2 pi R)",
                    tex=_TEX_NG,
                        eq="5.14")
            else:
                recomputed_fsr_hz = C0 / (ctx.vals[P_NG] * current_length_m)
                if fsr_key == P_FSR_THZ:
                    ctx.set(P_FSR_THZ, recomputed_fsr_hz, name, (P_NG, P_RADIUS), "FSR = c/(n_g 2 pi R)",
                            tex=r"\Delta f_{\mathrm{FSR}} = \frac{c}{n_g\, 2\pi R}", eq="5.14")
                else:
                    ctx.set(P_FSR_NM, recomputed_fsr_hz * line_wavelength_m ** 2 / C0, name, (P_NG, P_RADIUS),
                            "FSR_lambda = lambda^2/(n_g 2 pi R)",
                            tex=r"\Delta\lambda_{\mathrm{FSR}} = \frac{\lambda_L^2}{n_g\, 2\pi R}", eq="5.13")
        _resolve_overdetermined(ctx, name, residual, [P_RADIUS, P_NG, fsr_key], recompute)
        fsr, fsr_key = _fsr_hz_from_given(ctx)
    elif free[0] == P_RADIUS:
        ctx.set(P_RADIUS, C0 / (ctx.vals[P_NG] * fsr * _TWO_PI), name, (P_NG, fsr_key), "R = c/(2 pi n_g FSR)",
                tex=_TEX_R, eq="5.14")
    elif free[0] == P_NG:
        ctx.set(P_NG, C0 / (fsr * _TWO_PI * ctx.vals[P_RADIUS]), name, (P_RADIUS, fsr_key), "n_g = c/(FSR 2 pi R)",
                tex=_TEX_NG, eq="5.14")
    R = ctx.vals[P_RADIUS]
    ng = ctx.vals[P_NG]
    L = _TWO_PI * R
    fsr_hz = np.broadcast_to(C0 / (ng * L), np.broadcast_shapes(np.shape(ng), np.shape(L))).copy()
    # inherit the given FSR provenance because it anchors the geometry solve.
    # a held radius also needs the device's group-index provenance.
    inputs = (fsr_key, P_NG) if held else ((fsr_key,) if fsr is not None else (P_NG, P_RADIUS))
    ctx.set("derived.optics.length_m", L, name, (P_RADIUS,), "L = 2 pi R", tex=r"L = 2\pi R")
    ctx.set("derived.optics.fsr_hz", fsr_hz, name, inputs, "FSR = c/(n_g L)", tex=_TEX_FSR, eq="5.14")
    ctx.set("derived.optics.fsr_m", fsr_hz * line_wavelength_m ** 2 / C0, name,
            ("derived.optics.fsr_hz", "derived.optics.lambda_line_m"), "FSR_lambda = lambda_line^2 FSR/c",
            tex=r"\Delta\lambda_{\mathrm{FSR}} = \frac{\lambda_L^2}{c}\,\Delta f_{\mathrm{FSR}}", eq="0.2, 5.13")
    if fsr is None:
        ctx.set(P_FSR_THZ, fsr_hz, name, inputs, "FSR = c/(n_g L)", tex=_TEX_FSR, eq="5.14")
    ctx.set("derived.optics.round_trip_time_s", ng * L / C0, name, inputs, "T_rt = n_g L/c = 1/FSR",
            tex=r"\tau_{\mathrm{rt}} = \frac{n_g L}{c} = \frac{1}{\Delta f_{\mathrm{FSR}}}", eq="3.31")
    # the order m, and the n_eff that puts order m exactly on the cold resonance
    if not ctx.known(P_NEFF):
        raise UnderdeterminedError(f"{name}: ring.optics.n_eff is required")
    lam_res = ctx.vals["derived.optics.lambda_res_cold_m"]
    neff = ctx.vals[P_NEFF]
    if ctx.order_hold is not None:
        # keep the nominal order used by the geometry Jacobians.
        # adjust the phase index at this resonance.
        m = np.broadcast_to(np.asarray(ctx.order_hold, float), np.broadcast_shapes(np.shape(neff), np.shape(L)))
        m_formula, m_note = "m of the nominal-geometry device (fixed order)", \
            "a geometry error moves the resonance of the same mode order"
        m_tex, m_eq = r"m = m_{\text{nominal-geometry device}}\ \text{(fixed order)}", ""
    else:
        m = np.rint(neff * L / lam_res)
        m_formula, m_note = "m = round(n_eff L / lambda_res)", ""
        m_tex, m_eq = r"m = \mathrm{round}\!\left(\frac{n_{\mathrm{eff}} L}{\lambda_{r0}}\right)", "5.9"
    if np.any(m < 1):
        raise ValueError(f"{name}: resonance order below 1 (n_eff*L/lambda = {np.min(neff * L / lam_res):.3g})")
    ctx.set(P_ORDER, np.asarray(m).astype(np.int64), name, (P_NEFF, P_RADIUS, P_OFFSET), m_formula, m_note,
            tex=m_tex, eq=m_eq)
    neff_phase = m * lam_res / L
    ctx.set("derived.optics.n_eff_phase", neff_phase, name, (P_ORDER, P_OFFSET, P_RADIUS),
            "n_eff_phase = m lambda_res / L (exact cold resonance at the configured offset)",
            tex=r"n_{\mathrm{eff,phase}} = \frac{m\,\lambda_{r0}}{L}", eq="5.9")
    ctx.set("derived.optics.n_eff_correction", neff_phase - neff, name, (P_NEFF,), "n_eff_phase - n_eff",
            "absorbs the order rounding; |x| <= lambda/(2L) for the nominal-geometry device (a device of a sampled "
            "geometry keeps that order, so its correction also holds the n_eff Jacobian's fixed-wavelength term)",
            tex=r"n_{\mathrm{eff,phase}} - n_{\mathrm{eff}}")


# return the largest FSR, phase-order, or circumference closure error.
def _residual_r3_geometry(v: dict) -> float:
    L = v["derived.optics.length_m"]
    r_fsr = _max_rel_diff(v["derived.optics.fsr_hz"] * v[P_NG] * L / C0, 1.0)
    r_order = _max_rel_diff(v["derived.optics.n_eff_phase"] * L, v[P_ORDER] * v["derived.optics.lambda_res_cold_m"])
    r_length = _max_rel_diff(L, _TWO_PI * v[P_RADIUS])
    return max(r_fsr, r_order, r_length)


# R1 thermo-optic

# return beta in 1/K, or zeros when the thermo-optic model is first order.
def _thermo_optic_beta(ctx: RelationContext):
    if not ctx.order2:
        return np.zeros_like(np.asarray(ctx.vals[P_S0], float)[..., :1])
    if not ctx.known(P_BETA):
        raise UnderdeterminedError("R1_thermo_optic: thermo_optic_order 2 needs ring.thermal.dlambda_dt_beta_per_k")
    return ctx.vals[P_BETA]


# derive effective-index temperature slopes from the measured resonance slope.
# the anchor s0 is `d(lambda)/dT` at t0.
# resonance shifts through both index change and length expansion.
# subtract the expansion contribution to solve `dn_eff/dT = s0*n_g/lambda - n_eff*alpha_L`.
# second order uses beta and the expansion slope to derive `d2n_eff/dT2`.
# first order sets both slopes of the temperature coefficients to zero.
def _solve_r1_thermo_optic(ctx: RelationContext) -> None:
    name = "R1_thermo_optic"
    wavelength_m = ctx.vals["derived.optics.lambda_line_m"]
    ng = ctx.vals[P_NG]
    neff = ctx.vals["derived.optics.n_eff_phase"]
    alpha_l = ctx.vals[P_ALPHA_L]
    if not ctx.known(P_T0):
        raise UnderdeterminedError(f"{name}: ring.thermal.dlambda_dt_ref_ring_temp_c (T0) is required")
    beta = _thermo_optic_beta(ctx)
    ctx.set(
        "derived.thermal.beta_eff_per_k",
        beta,
        name,
        (P_BETA,) if ctx.order2 else (),
        "beta (0 when fidelity.thermo_optic_order = 1)",
        tex=r"\beta_T\quad \text{(0 when fidelity.thermo\_optic\_order = 1)}",
    )
    have_s, have_dn = ctx.known(P_S0), ctx.known(P_DNDT)
    if have_s and have_dn:
        residual = _max_rel_diff((wavelength_m / ng) * (ctx.vals[P_DNDT] + neff * alpha_l), ctx.vals[P_S0])

        def recompute(m):
            if m == P_S0:
                ctx.set(
                    P_S0,
                    (wavelength_m / ng) * (ctx.vals[P_DNDT] + neff * alpha_l),
                    name,
                    (P_DNDT, P_NG, P_ALPHA_L),
                    "s0 = (lambda/n_g)(dn_eff/dT + n_eff alpha_L)",
                    tex=r"s_0 = \frac{\lambda_L}{n_g}\left(\frac{\partial n_{\mathrm{eff}}}{\partial T} + "
                    r"n_{\mathrm{eff}}\,\alpha_L\right)",
                    eq="3.43",
                )
            else:
                ctx.set(
                    P_DNDT,
                    ctx.vals[P_S0] * ng / wavelength_m - neff * alpha_l,
                    name,
                    (P_S0, P_NG, P_ALPHA_L),
                    "dn_eff/dT = s0 n_g/lambda - n_eff alpha_L",
                    tex=r"\partial n_{\mathrm{eff}}/\partial T = \frac{n_g}{\lambda_L}\,s_0 - "
                    r"n_{\mathrm{eff}}\,\alpha_L",
                    eq="3.43",
                )

        _resolve_overdetermined(ctx, name, residual, [P_S0, P_DNDT], recompute)
    elif have_s:
        ctx.set(
            P_DNDT,
            ctx.vals[P_S0] * ng / wavelength_m - neff * alpha_l,
            name,
            (P_S0, P_NG, P_NEFF, P_ALPHA_L, _LAMBDA_LINE),
            "dn_eff/dT(T0) = s0 n_g/lambda - n_eff alpha_L(T0)",
            tex=r"\partial n_{\mathrm{eff}}/\partial T\,(T_0) = \frac{n_g}{\lambda_L}\,s_0 - "
            r"n_{\mathrm{eff}}\,\alpha_L(T_0)",
            eq="3.43",
        )
    else:
        raise UnderdeterminedError(
            f"{name}: ring.thermal.dlambda_dt_pm_per_k is the single thermo-optic anchor and "
            f"must be given (setting dn_eff_dt_per_k is a load error)"
        )
    d_expansion_coefficient_dtemp_per_k2 = ctx.vals.get(P_ALPHA_L_SLOPE, 0.0) if ctx.order2 else 0.0
    ctx.set(
        "derived.optics.alpha_l_slope_eff_per_k2",
        np.asarray(d_expansion_coefficient_dtemp_per_k2, float),
        name,
        (P_ALPHA_L_SLOPE,),
        "dalpha_L/dT (0 when thermo_optic_order = 1)",
        tex=r"\mathrm{d}\alpha_L/\mathrm{d}T\quad \text{(0 when thermo\_optic\_order = 1)}",
    )
    ctx.set(
        "derived.optics.d2n_eff_dt2_per_k2",
        ctx.vals[P_S0] * beta * ng / wavelength_m - neff * d_expansion_coefficient_dtemp_per_k2,
        name,
        (P_S0, P_BETA, P_NG, P_ALPHA_L_SLOPE, _LAMBDA_LINE),
        "d2n_eff/dT2 = s0 beta n_g/lambda - n_eff dalpha_L/dT",
        tex=r"\partial^2 n_{\mathrm{eff}}/\partial T^2 = \frac{n_g}{\lambda_L}\,s_0\,\beta_T - "
        r"n_{\mathrm{eff}}\,\frac{\mathrm{d}\alpha_L}{\mathrm{d}T}",
    )


# check the first- and second-order thermo-optic identities over the batch.
def _residual_r1_thermo_optic(v: dict) -> float:
    lam = v["derived.optics.lambda_line_m"]
    s = (lam / v[P_NG]) * (v[P_DNDT] + v["derived.optics.n_eff_phase"] * v[P_ALPHA_L])
    r_first = _max_rel_diff(s, v[P_S0])
    beta = v["derived.thermal.beta_eff_per_k"]
    d_expansion_coefficient_dtemp_per_k2 = v["derived.optics.alpha_l_slope_eff_per_k2"]
    # second-order identity — (lambda/n_g)(d2n_eff/dT2 + n_eff dalpha_L/dT) = s0 beta, relative to |s0|*max(|beta|, 1e-6)
    ds = (lam / v[P_NG]) * (
        v["derived.optics.d2n_eff_dt2_per_k2"]
        + v["derived.optics.n_eff_phase"] * d_expansion_coefficient_dtemp_per_k2
    )
    scale = np.abs(v[P_S0]) * np.maximum(np.abs(beta), 1e-6)
    r_second = float(np.max(np.abs(ds - v[P_S0] * beta) / scale))
    return max(r_first, r_second)


# R2 heater

P_FSR_SPAN = "derived.heater.p_fsr_span_m"      # FSR_lambda of the P_FSR measurement (m)


# return the wavelength FSR in m at the heater measurement wavelength.
# use the stated reference wavelength when present.
# this removes the line's wavelength-squared scaling from the heater anchor.
# otherwise use each ring's assigned line.
# record the span for the heater efficiency and integral solve.
def _p_fsr_span(ctx: RelationContext):
    if ctx.known(P_PFSR_LAM):
        wavelength_m = ctx.vals[P_PFSR_LAM]
        span = wavelength_m * wavelength_m * ctx.vals["derived.optics.fsr_hz"] / C0
        ctx.set(P_FSR_SPAN, span, "R2_heater_efficiency", ("derived.optics.fsr_hz", P_PFSR_LAM),
                "FSR_lambda at the P_FSR reference wavelength = lambda_ref^2 FSR/c",
                tex=r"\Delta\lambda_{\mathrm{FSR,meas}} = \frac{\lambda_{\mathrm{meas}}^2}{c}\,\Delta f_{\mathrm{FSR}}",
                eq="0.2")
    else:
        span = ctx.vals["derived.optics.fsr_m"]
        ctx.set(P_FSR_SPAN, span, "R2_heater_efficiency", ("derived.optics.fsr_m",),
                "FSR_lambda at the ring's line (no ring.heater.p_fsr_ref_wavelength_nm)",
                tex=r"\Delta\lambda_{\mathrm{FSR,meas}} = \Delta\lambda_{\mathrm{FSR}}\ \text{(at the ring's line)}")
    return span


# connect heater power per FSR with tuning efficiency in m/W.
# `efficiency = FSR_span_m / P_FSR`.
# use the FSR span at the measurement wavelength.
# solve the missing member or check a fully specified pair.
def _solve_r2_heater_efficiency(ctx: RelationContext) -> None:
    name = "R2_heater_efficiency"
    span_m = _p_fsr_span(ctx)
    span_path = P_FSR_SPAN
    have_power, have_efficiency = ctx.known(P_PFSR), ctx.known(P_EFF)
    if have_power and have_efficiency:
        residual = _max_rel_diff(ctx.vals[P_EFF] * ctx.vals[P_PFSR], span_m)

        def recompute(m):
            if m == P_EFF:
                ctx.set(P_EFF, span_m / ctx.vals[P_PFSR], name, (span_path, P_PFSR), "eff = FSR/P_FSR", tex=_TEX_EFF)
            else:
                ctx.set(P_PFSR, span_m / ctx.vals[P_EFF], name, (span_path, P_EFF), "P_FSR = FSR/eff", tex=_TEX_PFSR)
        _resolve_overdetermined(ctx, name, residual, [P_PFSR, P_EFF], recompute)
    elif have_power:
        ctx.set(P_EFF, span_m / ctx.vals[P_PFSR], name, (span_path, P_PFSR),
                "efficiency = FSR_lambda / P_FSR (P_FSR is the anchor; FSR_lambda of the P_FSR measurement)",
                tex=_TEX_EFF)
    elif have_efficiency:
        ctx.set(P_PFSR, span_m / ctx.vals[P_EFF], name, (span_path, P_EFF), "P_FSR = FSR_lambda/efficiency",
                tex=_TEX_PFSR)
    else:
        raise UnderdeterminedError(f"{name}: ring.heater.p_fsr_mw (the heater anchor) is required")


# derive the lock temperature and heater-to-ring thermal resistance in K/W.
# heating red-shifts resonance.
# take the cold offset modulo one FSR to select the nearest resonance that heating can reach.
# integrate the temperature sensitivity from the offset reference temperature to find the lock temperature.
# `lock_slope` sets resistance to efficiency divided by sensitivity at lock.
# `fsr_integral` instead finds the temperature rise for one full FSR at the measurement temperature, then divides by the measured heater power per FSR.
# at a fixed measurement wavelength, this removes the assigned line's scaling.
def _solve_r2_heater_z(ctx: RelationContext) -> None:
    name = "R2_heater_z"
    s0, t0 = ctx.vals[P_S0], ctx.vals[P_T0]
    beta = ctx.vals["derived.thermal.beta_eff_per_k"]
    t_ref = ctx.vals[P_OFFSET_T]
    # heating red-shifts resonance.
    # modulo one FSR selects the nearest reachable order when the sampled offset is negative or exceeds an FSR.
    offset_mod_fsr = np.mod(ctx.vals[P_OFFSET], ctx.vals["derived.optics.fsr_m"])
    ctx.set("derived.optics.lock_offset_m", offset_mod_fsr, name, (P_OFFSET, "derived.optics.fsr_m"),
            "resonance_offset mod FSR_lambda (heater-reachable order)",
            tex=r"\delta_0 \bmod \Delta\lambda_{\mathrm{FSR}}\ \text{(heater-reachable order)}")
    try:
        t_lock = solve_t_for_shift(t_ref, offset_mod_fsr, t0, s0, beta)
    except ValueError as e:
        raise ValueError(f"{name}: T_lock: {e}") from None
    ctx.set(P_TLOCK, t_lock, name, ("derived.optics.lock_offset_m", P_OFFSET_T, P_S0, P_T0, P_BETA),
            "int_{T_ref}^{T_lock} s(T) dT = resonance_offset mod FSR_lambda",
            tex=r"\int_{T_{\mathrm{ref}}}^{T_{\mathrm{lock}}} s(T)\,\mathrm{d}T = "
                r"\delta_0 \bmod \Delta\lambda_{\mathrm{FSR}}")
    s_lock = s0 * (1.0 + beta * (t_lock - t0))
    ctx.set("derived.thermal.s_lock_m_per_k", s_lock, name, (P_S0, P_BETA, P_TLOCK), "s(T_lock)",
            tex=r"s(T_{\mathrm{lock}}) = s_0\,\bigl(1 + \beta_T\,(T_{\mathrm{lock}} - T_0)\bigr)")
    efficiency_m_per_w = ctx.vals[P_EFF]
    if ctx.heater_anchor == "lock_slope":
        resistance_k_per_w = efficiency_m_per_w / s_lock
        formula = "Z_heater->ring(0) = efficiency / s(T_lock)  (R2 lock_slope)"
        formula_tex, eq = r"R_{\mathrm{th}} = \frac{\eta_{\mathrm{tune}}}{s(T_{\mathrm{lock}})}", "0.5"
        inputs = (P_EFF, P_S0, P_BETA, P_TLOCK)
    else:
        if not ctx.known(P_PFSR_T):
            raise UnderdeterminedError(f"{name}: fsr_integral needs ring.heater.p_fsr_ref_temp_c")
        t_meas = ctx.vals[P_PFSR_T]
        t_hot = solve_t_for_shift(t_meas, ctx.vals[P_FSR_SPAN], t0, s0, beta)
        ctx.set("derived.thermal.dt_fsr_k", t_hot - t_meas, name, (P_FSR_SPAN, P_PFSR_T, P_S0, P_BETA),
                "int_{T_meas}^{T_meas+dT} s dT = FSR_lambda of the P_FSR measurement",
                tex=r"\int_{T_{\mathrm{meas}}}^{T_{\mathrm{meas}} + \Delta T_{\mathrm{FSR}}} s(T)\,\mathrm{d}T = "
                    r"\Delta\lambda_{\mathrm{FSR,meas}}")
        resistance_k_per_w = (t_hot - t_meas) / ctx.vals[P_PFSR]
        formula = "Z_heater->ring(0) = dT_FSR / P_FSR"
        formula_tex, eq = r"R_{\mathrm{th}} = \frac{\Delta T_{\mathrm{FSR}}}{P_{\mathrm{FSR}}}", "2.12"
        inputs = ("derived.thermal.dt_fsr_k", P_PFSR)
    if ctx.known(P_Z):
        residual = _max_rel_diff(ctx.vals[P_Z], resistance_k_per_w)
        _resolve_overdetermined(ctx, name, residual, [P_Z],
                                lambda m: ctx.set(P_Z, resistance_k_per_w, name, inputs, formula, tex=formula_tex,
                                    eq=eq))
    else:
        ctx.set(P_Z, resistance_k_per_w, name, inputs, formula, tex=formula_tex, eq=eq)


# check heater efficiency, lock temperature, and thermal resistance closure.
def _residual_r2_heater(v: dict) -> float:
    span = v.get(P_FSR_SPAN, v["derived.optics.fsr_m"])
    r_eff = _max_rel_diff(v[P_EFF] * v[P_PFSR], span)
    s0, t0, beta = v[P_S0], v[P_T0], v["derived.thermal.beta_eff_per_k"]
    shift = integral_s(v[P_TLOCK], t0, s0, beta) - integral_s(v[P_OFFSET_T], t0, s0, beta)
    r_lock = float(np.max(np.abs(shift - np.mod(v[P_OFFSET], v["derived.optics.fsr_m"])) / v["derived.optics.fsr_m"]))
    if "derived.thermal.dt_fsr_k" in v:
        t_meas = v[P_PFSR_T]
        shift = integral_s(t_meas + v[P_Z] * v[P_PFSR], t0, s0, beta) - integral_s(t_meas, t0, s0, beta)
        r_z = _max_rel_diff(shift, span)
    else:
        r_z = _max_rel_diff(v[P_Z] * v["derived.thermal.s_lock_m_per_k"], v[P_EFF])
    return max(r_eff, r_lock, r_z)


# R4 Q and loss budget

# angular frequency of the cold resonance, omega = 2 pi c / lambda_res (rad/s).
def _cold_resonance_omega(ctx_or_vals) -> np.ndarray:
    values = ctx_or_vals.vals if isinstance(ctx_or_vals, RelationContext) else ctx_or_vals
    return _TWO_PI * C0 / values["derived.optics.lambda_res_cold_m"]


# derive intrinsic Q and heating fraction from the four loss components.
# all loss coefficients are power attenuation in 1/m.
# sum absorption, scattering, and radiation for total loss.
# only free-carrier and other absorption heat the ring.
# require all four components when any is present.
# otherwise skip this solve.
def _solve_r4_loss_budget(ctx: RelationContext) -> None:
    name = "R4_loss_budget"
    known_components = [ctx.known(p) for p in LOSS_COMPONENTS]
    if not any(known_components):
        return
    if not all(known_components):
        missing = [p for p, h in zip(LOSS_COMPONENTS, known_components) if not h]
        raise UnderdeterminedError(f"{name}: give all four loss-budget components or none (missing {missing})")
    wavelength_m = ctx.vals["derived.optics.lambda_res_cold_m"]
    ng = ctx.vals[P_NG]
    loss_components = [ctx.vals[p] for p in LOSS_COMPONENTS]         # alpha_fca, alpha_other_abs, alpha_scatter, alpha_rad (1/m)
    alpha_i = loss_components[0] + loss_components[1] + loss_components[2] + loss_components[3]
    if np.any(alpha_i <= 0):
        raise ValueError(f"{name}: total intrinsic loss must be > 0")
    qi = _TWO_PI * ng / (wavelength_m * alpha_i)
    eta = (loss_components[0] + loss_components[1]) / alpha_i
    if ctx.known(P_QI):
        residual = _max_rel_diff(ctx.vals[P_QI], qi)
        _resolve_overdetermined(ctx, name, residual, [P_QI] + list(LOSS_COMPONENTS),
                                lambda m: ctx.set(P_QI, qi, name, LOSS_COMPONENTS + (P_NG, _LAMBDA_RES_COLD),
                                                  "Q_i = 2 pi n_g/(lambda alpha_i)", tex=_TEX_QI, eq="5.24, 3.46"))
        qi = ctx.vals[P_QI]
    else:
        ctx.set(P_QI, qi, name, LOSS_COMPONENTS + (P_NG, _LAMBDA_RES_COLD),
                "Q_i = 2 pi n_g/(lambda_res alpha_i), alpha_i = sum",
                tex=_TEX_QI + r",\quad \alpha_{\mathrm{int}} = \alpha_{\mathrm{FCA}} + \alpha_{\mathrm{other}} + "
                r"\alpha_{\mathrm{scat}} + \alpha_{\mathrm{rad}}", eq="5.24, 3.46")
    if ctx.known(P_ETA):
        residual = _max_rel_diff(ctx.vals[P_ETA], eta)
        _resolve_overdetermined(ctx, name, residual, [P_ETA] + list(LOSS_COMPONENTS),
                                lambda m: ctx.set(P_ETA, eta, name, LOSS_COMPONENTS, "eta = (fca + other)/alpha_i",
                                                  tex=_TEX_ETA))
    else:
        ctx.set(P_ETA, eta, name, LOSS_COMPONENTS,
                "eta = (alpha_fca + alpha_other_abs)/alpha_i (the heat fraction comes from the loss budget)",
                tex=_TEX_ETA)
    ctx.set("derived.optics.alpha_heat_per_m", loss_components[0] + loss_components[1], name, (P_A_FCA, P_A_OTH),
        "alpha_fca + alpha_other_abs",
            tex=r"\alpha_{\mathrm{heat}} = \alpha_{\mathrm{FCA}} + \alpha_{\mathrm{other}}")
    ctx.set("derived.optics.alpha_nonheat_per_m", loss_components[2] + loss_components[3], name, (P_A_SCA, P_A_RAD),
        "alpha_scatter + alpha_rad",
            tex=r"\alpha_{\mathrm{nonheat}} = \alpha_{\mathrm{scat}} + \alpha_{\mathrm{rad}}")


# split intrinsic loss into heating and non-heating parts using the given fraction.
# run after the Q solve has produced intrinsic loss.
# skip this step when the loss-budget solve has already supplied both parts.
def _solve_r4_heat_fraction(ctx: RelationContext) -> None:
    name = "R4_heat_fraction"
    if "derived.optics.alpha_heat_per_m" in ctx.vals:
        return
    if not ctx.known(P_ETA):
        raise UnderdeterminedError(f"{name}: give the loss budget (four components) or ring.optics.heat_fraction "
                                   f"(the heat fraction comes from a loss budget)")
    alpha_i = ctx.vals["derived.optics.alpha_i_per_m"]
    eta = ctx.vals[P_ETA]
    ctx.set("derived.optics.alpha_heat_per_m", eta * alpha_i, name, (P_ETA, P_QI), "eta alpha_i",
            tex=r"\alpha_{\mathrm{heat}} = \eta_{\mathrm{heat}}\,\alpha_{\mathrm{int}}")
    ctx.set("derived.optics.alpha_nonheat_per_m", (1.0 - eta) * alpha_i, name, (P_ETA, P_QI), "(1 - eta) alpha_i",
            tex=r"\alpha_{\mathrm{nonheat}} = (1 - \eta_{\mathrm{heat}})\,\alpha_{\mathrm{int}}")


# solve Q values and convert decay rates to exact per-pass couplings.
# use given anchors or geometry-adjusted couplers.
# the energy fraction coupled per pass is `1 - exp(-omega*T_rt/Q_port)`.
# intrinsic round-trip amplitude is `exp(-omega*T_rt/(2*Q_i))`.
# these exponential forms keep the rate sum consistent between the Airy ring and temporal coupled-mode models.
def _solve_r4_q(ctx: RelationContext) -> None:
    name = "R4_q"
    if ctx.coupling_anchor is not None:
        _solve_r4_q_sampled_geometry(ctx, name, ctx.coupling_anchor)
    else:
        _solve_r4_q_from_anchors(ctx, name)
    # convert continuous decay rates to the exact energy loss in one round trip.
    values = ctx.vals
    round_trip_s = values["derived.optics.round_trip_time_s"]
    omega_rad_per_s = _cold_resonance_omega(ctx)
    inputs = (P_NG, P_RADIUS, _LAMBDA_RES_COLD)
    ctx.set("derived.optics.kappa2_through", -np.expm1(-omega_rad_per_s * round_trip_s / values[P_QT]), name, (P_QT,
        ) + inputs,
            "K_t^2 = 1 - exp(-omega T_rt/Q_t)",
            tex=r"\kappa_t^2 = 1 - \exp\!\left(-\frac{\omega\,\tau_{\mathrm{rt}}}{Q_t}\right),\quad " + _TEX_OMEGA)
    ctx.set("derived.optics.kappa2_drop", -np.expm1(-omega_rad_per_s * round_trip_s / values[P_QD]), name, (P_QD,
        ) + inputs,
            "K_d^2 = 1 - exp(-omega T_rt/Q_d)",
            tex=r"\kappa_d^2 = 1 - \exp\!\left(-\frac{\omega\,\tau_{\mathrm{rt}}}{Q_d}\right),\quad " + _TEX_OMEGA)
    ctx.set("derived.optics.round_trip_amplitude", np.exp(-0.5 * omega_rad_per_s * round_trip_s / values[P_QI]),
        name, (P_QI,) + inputs,
            "a = exp(-omega T_rt/(2 Q_i))",
            tex=r"a = \exp\!\left(-\frac{\omega\,\tau_{\mathrm{rt}}}{2\,Q_{\mathrm{int}}}\right) = "
                r"e^{-\alpha_{\mathrm{int}} L/2},\quad " + _TEX_OMEGA, eq="3.46, 5.2")
    ctx.set("derived.optics.alpha_i_per_m", _TWO_PI * values[P_NG] / (values["derived.optics.lambda_res_cold_m"] * values[P_QI]),
            name, (P_QI, P_NG, _LAMBDA_RES_COLD), "alpha_i = 2 pi n_g/(lambda Q_i)",
            tex=r"\alpha_{\mathrm{int}} = \frac{2\pi\, n_g}{\lambda_{r0}\, Q_{\mathrm{int}}}", eq="5.24, 3.46")


# apply a device's geometry errors to the nominal couplers, then derive its Q.
# the anchor contains per-pass couplings solved at nominal geometry.
# multiply each by `exp(dln_kappa2)` for the device's geometry.
# convert these couplings back to Q using the device's frequency and round-trip time.
# combine them with intrinsic loss to obtain loaded Q and drop share.
# a loss or group-index change changes Q without changing the coupler's per-pass fraction.
def _solve_r4_q_sampled_geometry(ctx: RelationContext, name: str, anchor: dict) -> None:
    values = ctx.vals
    round_trip_s = values["derived.optics.round_trip_time_s"]
    omega_rad_per_s = _cold_resonance_omega(ctx)
    anchors = tuple(p for p in (P_QL, P_XD, P_QT, P_QD, P_QI) if p in ctx.prov)
    if not ctx.known(P_QI):
        # without a loss budget or given intrinsic Q, keep the nominal loss per length.
        ctx.set(P_QI, _TWO_PI * values[P_NG] / (values[_LAMBDA_RES_COLD] * anchor["alpha_i_per_m"]), name,
                anchors + (P_NG, _LAMBDA_RES_COLD),
                "Q_i = 2 pi n_g/(lambda_res alpha_i) with the alpha_i of the nominal-geometry device",
                tex=_TEX_QI + r"\ \text{(}\alpha_{\mathrm{int}}\text{ of the nominal-geometry device)}",
                eq="5.24, 3.46")
    port_q = {}
    for port, q_path in (("through", P_QT), ("drop", P_QD)):
        coupling_fraction = np.asarray(anchor[f"kappa2_{port}"],
            float) * np.exp(np.asarray(anchor[f"dln_kappa2_{port}"], float))
        # non-finite inputs fail the later finite-value check.
        invalid_coupling = (np.isfinite(coupling_fraction)
                            & ((coupling_fraction <= 0.0) | (coupling_fraction >= 1.0)))
        if np.any(invalid_coupling):
            raise ValueError(f"{name}: a sampled geometry moves the {port} per-pass coupling outside (0, 1) (worst "
                             f"{float(coupling_fraction[invalid_coupling].flat[0]):.3g}); check the coupling Jacobian and the gap sigmas")
        port_q[q_path] = omega_rad_per_s * round_trip_s / (-np.log1p(-coupling_fraction))
        port_letter = "t" if port == "through" else "d"
        ctx.set(q_path, port_q[q_path], name, anchors + ("derived.geometry.dln_kappa2_" + port, P_NG, P_RADIUS,
                                                    _LAMBDA_RES_COLD),
                f"Q = omega T_rt/(-ln(1 - K^2)), K^2 = K_a^2 exp(dln_kappa2_{port}): the nominal-geometry device's "
                f"{port} coupler moved by the device's geometry",
                tex=(r"Q_%s = \frac{\omega\,\tau_{\mathrm{rt}}}{-\ln(1 - \kappa_%s^2)},\quad \kappa_%s^2 = "
                     r"\kappa_{%s,\mathrm{nom}}^2\, e^{\Delta\ln\kappa_%s^2}") % ((port_letter,) * 5))
    loaded_q = 1.0 / (1.0 / values[P_QI] + 1.0 / port_q[P_QT] + 1.0 / port_q[P_QD])
    ctx.set(P_QL, loaded_q, name, (P_QI, P_QT, P_QD), "1/Q_L = 1/Q_i + 1/Q_t + 1/Q_d (the device's couplers and loss)",
            tex=_TEX_QL, eq="5.24")
    ctx.set(P_XD, loaded_q / port_q[P_QD], name, (P_QL, P_QD),
        "kappa_d/kappa = Q_L/Q_d (the device's couplers and loss)",
            tex=r"x_d = \frac{Q_L}{Q_d}")


# solve the free Q using the sum of intrinsic and coupling decay rates.
# with a given drop share, leave one of loaded, intrinsic, or through Q free.
# derive drop Q from the share.
# without that share, leave one of the four Q values free.
# check the extra constraint if both drop Q and drop share are given.
def _solve_r4_q_from_anchors(ctx: RelationContext, name: str) -> None:
    have_q = {p: ctx.known(p) for p in (P_QL, P_QI, P_QT, P_QD)}
    use_drop_share = ctx.known(P_XD)
    values = ctx.vals

    # check the extra equation when drop Q and drop share are both given.
    def check_drop_share() -> None:
        residual = _max_rel_diff(values[P_QL] / values[P_QD], values[P_XD])
        if residual > max(ctx.tolerance.get(name, 0.0), _EXACT_RTOL):
            raise OverdeterminedError(f"{name}: q_drop and kappa_d_over_kappa are both given and disagree "
                                      f"({residual:.3g})")

    check_drop_share_after_solve = False
    if use_drop_share and have_q[P_QD]:
        if have_q[P_QL]:
            check_drop_share()
        else:
            check_drop_share_after_solve = True     # Q_L comes from the rate sum below
        use_drop_share = False
    if use_drop_share:
        x = values[P_XD]
        free = [p for p in (P_QL, P_QI, P_QT) if not have_q[p]]
        if len(free) != 1:
            raise (UnderdeterminedError if len(free) > 1 else OverdeterminedError)(
                f"{name}: with kappa_d_over_kappa given, exactly one of q_loaded, q_intrinsic, q_through must be free "
                f"(free: {free})")
        f = free[0]  # config path of the one free Q member
        if f == P_QL:
            loaded_q = (1.0 - x) / (1.0 / values[P_QI] + 1.0 / values[P_QT])
            ctx.set(P_QL, loaded_q, name, (P_QI, P_QT, P_XD), "(1 - x_d)/Q_L = 1/Q_i + 1/Q_t",
                    tex=r"\frac{1 - x_d}{Q_L} = \frac{1}{Q_{\mathrm{int}}} + \frac{1}{Q_t}", eq="5.24")
        else:
            other = P_QT if f == P_QI else P_QI
            inverse_q = (1.0 - x) / values[P_QL] - 1.0 / values[other]
            if np.any(inverse_q <= 0):
                raise ValueError(f"{name}: inconsistent Q: (1 - x_d)/Q_L <= 1/{other.rsplit('.', 1)[1]} "
                                 f"(min {float(np.min(inverse_q)):.3g}); the coupling and loss cannot be solved")
            ctx.set(f, 1.0 / inverse_q, name, (P_QL, other, P_XD), "1/Q = (1 - x_d)/Q_L - 1/Q_other",
                    tex=r"\frac{1}{%s} = \frac{1 - x_d}{Q_L} - \frac{1}{%s}" % (SYMBOLS[f], SYMBOLS[other]),
                        eq="5.24")
        ctx.set(P_QD, values[P_QL] / x, name, (P_QL, P_XD), "Q_d = Q_L / x_d", tex=r"Q_d = \frac{Q_L}{x_d}")
    else:
        free = [p for p, h in have_q.items() if not h]
        if len(free) > 1:
            raise UnderdeterminedError(f"{name}: {len(free)} of q_loaded, q_intrinsic, q_through, q_drop are free "
                                       f"({free}); give all but one (or kappa_d_over_kappa)")
        if not free:
            residual = _max_rel_diff(1.0 / values[P_QL], 1.0 / values[P_QI] + 1.0 / values[P_QT] + 1.0 / values[P_QD])

            def recompute(m):
                _solve_one_free_q(ctx, m, name)
            _resolve_overdetermined(ctx, name, residual, [P_QL, P_QI, P_QT, P_QD], recompute)
        else:
            _solve_one_free_q(ctx, free[0], name)
        if check_drop_share_after_solve:
            check_drop_share()


# solve the rate sum 1/Q_L = 1/Q_i + 1/Q_t + 1/Q_d for the one free Q at path `f`.
def _solve_one_free_q(ctx: RelationContext, f: str, name: str) -> None:
    values = ctx.vals
    others = [p for p in (P_QI, P_QT, P_QD) if p != f]
    if f == P_QL:
        ctx.set(P_QL, 1.0 / (1.0 / values[P_QI] + 1.0 / values[P_QT] + 1.0 / values[P_QD]), name, (P_QI, P_QT, P_QD),
                "1/Q_L = 1/Q_i + 1/Q_t + 1/Q_d", tex=_TEX_QL, eq="5.24")
        return
    inverse_q = 1.0 / values[P_QL] - sum(1.0 / values[p] for p in others if p != P_QL)
    if np.any(inverse_q <= 0):
        raise ValueError(f"{name}: inconsistent Q: 1/Q_L <= sum of the other 1/Q (min {float(np.min(inverse_q)):.3g}); "
                         f"{f} cannot be solved")
    ctx.set(f, 1.0 / inverse_q, name, tuple([P_QL] + [p for p in others if p != P_QL]), "1/Q = 1/Q_L - sum(other 1/Q)",
            tex=(r"\frac{1}{%s} = \frac{1}{Q_L} - " % SYMBOLS[f]
                 + " - ".join(r"\frac{1}{%s}" % SYMBOLS[p] for p in others)),
            eq="5.24")


# check decay rates, per-pass couplings, intrinsic amplitude, and loss partition.
# return the largest relative closure error across these equations and the batch.
def _residual_r4_q(v: dict) -> float:
    r_rate_sum = _max_rel_diff(1.0 / v[P_QL], 1.0 / v[P_QI] + 1.0 / v[P_QT] + 1.0 / v[P_QD])
    omega_rad_per_s = _cold_resonance_omega(v)
    round_trip_s = v["derived.optics.round_trip_time_s"]
    r_k2_through = _max_rel_diff(-np.log1p(-v["derived.optics.kappa2_through"]),
        omega_rad_per_s * round_trip_s / v[P_QT])
    r_k2_drop = _max_rel_diff(-np.log1p(-v["derived.optics.kappa2_drop"]), omega_rad_per_s * round_trip_s / v[P_QD])
    r_amplitude = _max_rel_diff(-2.0 * np.log(v["derived.optics.round_trip_amplitude"]),
        omega_rad_per_s * round_trip_s / v[P_QI])
    r_budget = 0.0
    if all(p in v for p in LOSS_COMPONENTS):
        total_loss = sum(v[p] for p in LOSS_COMPONENTS)
        r_budget = max(_max_rel_diff(v[P_QI], _TWO_PI * v[P_NG] / (v["derived.optics.lambda_res_cold_m"] * total_loss)),
                       _max_rel_diff(v[P_ETA], (v[P_A_FCA] + v[P_A_OTH]) / total_loss))
    r_heat_split = _max_rel_diff(v["derived.optics.alpha_heat_per_m"] + v["derived.optics.alpha_nonheat_per_m"],
                                 v["derived.optics.alpha_i_per_m"])
    return max(r_rate_sum, r_k2_through, r_k2_drop, r_amplitude, r_budget, r_heat_split)


# R5 latency

# convert each owned I/O latency from seconds to integer plant ticks.
# apply the world's latency scale before rounding.
# a config key cannot own two channels.
# temperature-sensor and controller latency are handled separately.
def _solve_r5_latency(ctx: RelationContext) -> None:
    name = "R5_latency"
    if ctx.dt is None:
        raise UnderdeterminedError(f"{name}: sim.dt_plant_s is required")
    owners: dict[str, str] = {}
    for channel, key in LATENCY_OWNERS.items():
        if key in owners:
            raise OverdeterminedError(f"{name}: {key} owns two channels ({owners[key]}, {channel})")
        owners[key] = channel
        if ctx.known(key):
            ticks = np.rint(np.asarray(ctx.vals[key]) * ctx.latency_scale / ctx.dt).astype(np.int64)
            if np.any(ticks < 0):
                raise ValueError(f"{name}: negative latency for {channel}")
            ctx.set(f"derived.timing.{channel}.latency_ticks", ticks, name, (key, "uncertainty: latency scale"),
                    "round(latency_s * latency_scale / dt_plant_s)",
                    tex=r"\mathrm{round}\!\left(\frac{\text{latency} \times \text{latency scale}}{\Delta t}\right)")


# return zero because latency conversion has no separate closure equation.
def _residual_r5_latency(v: dict) -> float:
    return 0.0


# registry and entry points

RELATIONS: tuple[Relation, ...] = (
    Relation("R3_geometry", (P_RADIUS, P_NG, P_FSR_NM, P_FSR_THZ, P_NEFF, P_OFFSET, P_ORDER), _solve_r3_geometry,
             _residual_r3_geometry, "FSR = c/(n_g 2 pi R); m = round(n_eff L/lambda_res)"),
    Relation("R1_thermo_optic", (P_S0, P_T0, P_DNDT, P_NG, P_BETA), _solve_r1_thermo_optic, _residual_r1_thermo_optic,
             "dlambda/dT(T) = (lambda/n_g)(dn_eff/dT(T) + n_eff alpha_L(T)) = s0 (1 + beta (T - T0))"),
    Relation("R2_heater_efficiency", (P_PFSR, P_EFF), _solve_r2_heater_efficiency, lambda v: 0.0,
             "efficiency = FSR/P_FSR"),
    Relation("R2_heater_z", (P_Z, P_EFF, P_S0, P_TLOCK), _solve_r2_heater_z, _residual_r2_heater,
             "Z_heater->ring(0) = dT_FSR/P_FSR (fsr_integral, default) or efficiency/s(T_lock) (lock_slope)"),
    Relation("R4_loss_budget", LOSS_COMPONENTS + (P_QI, P_ETA), _solve_r4_loss_budget, lambda v: 0.0,
             "alpha_i = sum of components; Q_i = 2 pi n_g/(lambda alpha_i); eta = (fca + other)/alpha_i"),
    Relation("R4_q", (P_QL, P_QI, P_QT, P_QD, P_XD), _solve_r4_q, _residual_r4_q, "1/Q_L = 1/Q_i + 1/Q_t + 1/Q_d"),
    Relation("R4_heat_fraction", (P_ETA, P_QI), _solve_r4_heat_fraction, lambda v: 0.0,
             "alpha_heat = eta alpha_i when no budget component is given"),
    Relation("R5_latency", tuple(LATENCY_OWNERS.values()), _solve_r5_latency, _residual_r5_latency,
             "one owner per I/O channel; ticks"),
)
RELATION_NAMES = [r.name for r in RELATIONS]


# run every relation in dependency order on the context (in place) and return it.
def solve(ctx: RelationContext) -> RelationContext:
    for relation in RELATIONS:
        relation.solver(ctx)
    return ctx


# return each relation's largest relative residual after solving.
# `realize` requires residuals at or below 1e-12.
# a missing required value produces NaN so that the closure check rejects the incomplete result.
def residuals(vals: dict) -> dict[str, float]:
    out = {}
    for relation in RELATIONS:
        try:
            out[relation.name] = float(relation.residual(vals))
        except KeyError:
            # optional members are handled inside each residual.
            # a missing required member must fail the closure check.
            out[relation.name] = float("nan")
    return out
