"""Relation groups R1..R5 of 07 P0-3: one independent anchor per physical relation, closure enforced on arrays.

Each relation lists its members (config paths). In a config, every relation must have exactly one *free* member
(absent, ``null`` or tagged ``derived``) per equation; the solver computes it with NumPy broadcasting over the batch
(B, N) arrays that ``realize()`` builds after sampling (07 P0-4). If every member is given, the residual is computed
and :class:`OverdeterminedError` is raised unless the preset states ``relations.<name>.tolerance_rel`` and the
residual is inside it; then the member with the weakest provenance is recomputed so that the relation holds exactly.
Too many free members raise :class:`UnderdeterminedError`.

All values here are SI (m, s, W, K, Hz, 1/m). Members are keyed by their config path; quantities that have no config
key live under ``derived.*``. Equations (07 P0-3 unless stated):

R3_geometry      FSR_f = c / (n_g * L),  L = 2*pi*R;  m = round(n_eff * L / lambda_res);  n_eff_phase = m*lambda_res/L
                 (lambda_res = lambda_line - resonance_offset, the cold resonance; R = r_ref of 07 P1-6). The
                 rounding residual moves into n_eff_phase (|correction| <= lambda/(2L) ~ 0.016), so the cold
                 resonance sits exactly at the configured offset and the FSR keeps its hypothesis.
                 Sampled geometry (``RelationContext.radius_hold`` / ``order_hold``, set by realize from the solve of
                 the nominal-geometry device): a free radius is solved once from the nominal n_g and the FSR
                 hypothesis and held for every device of the world (a mask dimension), so a device's FSR follows its
                 own n_g (07 P1-16: one geometry error moves the resonance, the FSR and the coupling together); the
                 order is the nominal device's (the geometry Jacobians are taken at fixed r_ref and order).
R1_thermo_optic  s(T) = (lambda/n_g) * (dn_eff/dT(T) + n_eff * alpha_L(T)),  s(T) = s0 * (1 + beta*(T - T0))
                 => dn_eff/dT(T0) = s0*n_g/lambda - n_eff*alpha_L(T0),
                    d2n_eff/dT2  = s0*beta*n_g/lambda - n_eff*dalpha_L/dT   (beta = 0 when thermo_optic_order = 1).
R2_heater_efficiency  efficiency = FSR_lambda / P_FSR  (07 D16: P_FSR is the anchor). FSR_lambda is taken at
                 ``ring.heater.p_fsr_ref_wavelength_nm`` (the line the P_FSR measurement was made on) when given,
                 else at each ring's line.
R2_heater_z      lock_slope (07 as written):  Z_heater->ring(0) = efficiency / s(T_lock), where T_lock solves
                 int_{T_ref}^{T_lock} s(T) dT = resonance_offset mod FSR_lambda (ambient at the offset reference
                 temperature; the heater only red-shifts, so a ring with a negative or > FSR offset locks the
                 nearest order blue of its line).
                 fsr_integral: Z = dT_FSR / P_FSR with int_{T_meas}^{T_meas + dT_FSR} s(T) dT = FSR_lambda, the
                 FSR_lambda of R2_heater_efficiency: at the P_FSR reference wavelength Z is a stack property, the same
                 on every line (s0 is one value for every line); at each ring's line it follows lambda_line^2.
R4_loss_budget   alpha_i = alpha_fca + alpha_other_abs + alpha_scatter + alpha_rad,  Q_i = 2*pi*n_g/(lambda_res*alpha_i),
                 eta = (alpha_fca + alpha_other_abs)/alpha_i  (07 P0-2). Or, with no component given:
                 eta and Q_i given -> alpha_heat = eta*alpha_i, alpha_nonheat = (1 - eta)*alpha_i.
R4_q             1/Q_L = 1/Q_i + 1/Q_t + 1/Q_d  (optional kappa_d_over_kappa = Q_L/Q_d). Derived per-pass couplings
                 use the exact energy decay per round trip: K_j^2 = 1 - exp(-omega*T_rt/Q_j), a = exp(-omega*T_rt/(2*Q_i)),
                 T_rt = n_g*L/c, so the rate sum is exact for the Airy ring (07 P2-4: one parameter set for both forms).
                 Sampled geometry (``RelationContext.coupling_anchor``): the anchors describe the nominal-geometry
                 device. Its per-pass couplings K_j,a^2 (solved by realize from the anchors with the nominal n_g,
                 length and loss) are moved by the device's geometry, K_j^2 = K_j,a^2 exp(dln_kappa2_j); then
                 Q_j = omega*T_rt/(-ln(1 - K_j^2)) with the device's own omega and T_rt, Q_i from the device's loss
                 budget, and Q_L and kappa_d_over_kappa follow from the rate sum (07 P1-16).
R5_latency       one owner per I/O channel (07 P0-6); latency_ticks = round(latency_s * latency_scale / dt).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from mrm.units import C0

__all__ = ["Relation", "RELATIONS", "RELATION_NAMES", "OverdeterminedError", "UnderdeterminedError",
           "RelationContext", "Derivation", "solve", "residuals", "LATENCY_OWNERS", "integral_s", "solve_t_for_shift"]

_TWO_PI = 2.0 * math.pi
#: numerical slack for "every member given and consistent" when no tolerance_rel is stated
_EXACT_RTOL = 1e-9

# config paths of the members (SI values inside the context)
P_RADIUS = "ring.geometry.radius_um"
P_NG = "ring.optics.n_g"
P_NEFF = "ring.optics.n_eff"
P_FSR_NM = "ring.optics.fsr_nm"
P_FSR_THZ = "ring.optics.fsr_thz"
P_OFFSET = "ring.optics.resonance_offset_nm"
P_OFFSET_T = "ring.optics.resonance_offset_ref_temp_c"
P_ORDER = "ring.optics.order"
P_S0 = "ring.thermal.dlambda_dt_pm_per_k"
P_T0 = "ring.thermal.dlambda_dt_ref_ring_temp_c"
P_BETA = "ring.thermal.dlambda_dt_beta_per_k"
P_DNDT = "ring.optics.dn_eff_dt_per_k"
P_ALPHA_L = "ring.optics.alpha_l_per_k"
P_ALPHA_L_SLOPE = "ring.optics.alpha_l_slope_per_k2"
P_PFSR = "ring.heater.p_fsr_mw"
P_PFSR_T = "ring.heater.p_fsr_ref_temp_c"
P_PFSR_LAM = "ring.heater.p_fsr_ref_wavelength_nm"
P_EFF = "ring.heater.efficiency_nm_per_mw"
P_Z = "ring.thermal.z_heater_ring_k_per_mw"
P_TLOCK = "ring.thermal.t_ring_lock_c"
P_QL = "ring.optics.q_loaded"
P_QI = "ring.optics.q_intrinsic"
P_QT = "ring.optics.q_through"
P_QD = "ring.optics.q_drop"
P_XD = "ring.optics.kappa_d_over_kappa"
P_ETA = "ring.optics.heat_fraction"
_LB = "ring.optics.loss_budget."
P_A_FCA = _LB + "alpha_fca_db_per_cm"
P_A_OTH = _LB + "alpha_other_abs_db_per_cm"
P_A_SCA = _LB + "alpha_scatter_db_per_cm"
P_A_RAD = _LB + "alpha_bend_radiation_db_per_cm"
LOSS_COMPONENTS = (P_A_FCA, P_A_OTH, P_A_SCA, P_A_RAD)
#: the line and cold-resonance wavelengths (realize records how they come from the channel plan, 07 P1-14)
_LAM_LINE = "derived.optics.lambda_line_m"
_LAM_RES = "derived.optics.lambda_res_cold_m"

#: R5: the one config key that owns the latency of each I/O channel (07 P0-6). Temperature sensors own theirs in
#: the temp_sensors table (field latency_s); the controller declares Controller.latency_ticks in code.
LATENCY_OWNERS = {
    "tx.monitor": "ring.monitor.adc_latency_s",
    "tx.heater": "ring.heater.latency_s",
    "tx.bias": "ring.bias_dac.latency_s",
    "heater.readback": "ring.heater.readback.latency_s",
    "rx.error_count": "datapath.error_count_latency_s",
}


class OverdeterminedError(ValueError):
    """Every member of a relation is given and the residual exceeds the stated tolerance_rel (default: exact)."""


class UnderdeterminedError(ValueError):
    """A relation has more free members than equations."""


@dataclass
class Derivation:
    """How a derived quantity was computed (used by ``mrm params explain``)."""
    path: str
    relation: str
    inputs: tuple[str, ...]
    formula: str
    note: str = ""


@dataclass
class RelationContext:
    """Arrays and flags a relation solver works on. ``vals`` maps config path (or ``derived.*``) to SI arrays."""
    vals: dict[str, np.ndarray]
    given: set[str]
    prov: dict[str, str]
    order2: bool = True
    heater_anchor: str = "lock_slope"
    tolerance: dict[str, float] = field(default_factory=dict)
    dt: float | None = None
    latency_scale: np.ndarray | float = 1.0
    derivations: dict[str, Derivation] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    #: sampled geometry (set by realize from the relation solve of the nominal-geometry device, 07 P1-16): the radius
    #: (m) a free radius is held at, the order every device keeps, and the anchor per-pass couplings
    #: {"kappa2_through", "kappa2_drop", "alpha_i_per_m", "dln_kappa2_through", "dln_kappa2_drop"} (B, N)
    radius_hold: np.ndarray | None = None
    order_hold: np.ndarray | None = None
    coupling_anchor: dict | None = None

    def known(self, p: str) -> bool:
        return p in self.vals and p in self.given

    def set(self, p: str, v, relation: str, inputs, formula: str, note: str = "") -> None:
        self.vals[p] = np.asarray(v)
        self.given.add(p)
        self.derivations[p] = Derivation(p, relation, tuple(inputs), formula, note)


@dataclass(frozen=True)
class Relation:
    """A relation group: ``members`` (config paths), a solver that fills the free member(s) of a context, and a
    residual function that returns the max relative residual over the batch once every member has a value."""
    name: str
    members: tuple[str, ...]
    solver: Callable[[RelationContext], None]
    residual: Callable[[dict], float]
    doc: str


# ------------------------------------------------------------------------------------------------ helpers

def _rel(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    den = np.maximum(np.abs(b), 1e-300)
    return float(np.max(np.abs(a - b) / den)) if a.size else 0.0


def _weakest_member(ctx: RelationContext, members) -> str:
    from mrm.params.config import provenance_rank
    return max(members, key=lambda m: provenance_rank(ctx.prov.get(m, "assumed")))


def _overdetermined(ctx: RelationContext, name: str, resid: float, members, recompute: Callable[[str], None]) -> None:
    tol = ctx.tolerance.get(name)
    if tol is None:
        if resid <= _EXACT_RTOL:
            return
        raise OverdeterminedError(
            f"{name}: every member is given ({', '.join(members)}) and the relative residual is {resid:.3g}; leave one "
            f"member null (or tag it derived), or state relations.{name}.tolerance_rel (07 P0-3)")
    if resid > tol:
        raise OverdeterminedError(f"{name}: relative residual {resid:.3g} exceeds tolerance_rel {tol:g} "
                                  f"(members {', '.join(members)})")
    weakest = _weakest_member(ctx, members)
    recompute(weakest)
    d = ctx.derivations.get(weakest)
    if d is not None:
        d.note = (f"overdetermined within tolerance_rel={tol:g} (residual {resid:.3g}); recomputed this member, the "
                  f"weakest provenance ({ctx.prov.get(weakest)})")
    ctx.notes[name] = f"residual {resid:.3g} <= tolerance_rel {tol:g}; recomputed {weakest}"


def integral_s(t, t0, s0, beta):
    """F(T) = int_{T0}^{T} s0*(1 + beta*(T' - T0)) dT' = s0*((T - T0) + beta*(T - T0)^2/2). SI (m)."""
    x = t - t0
    return s0 * (x + 0.5 * beta * x * x)


def solve_t_for_shift(t_start, shift, t0, s0, beta):
    """Temperature T with int_{t_start}^{T} s(T') dT' = shift (07 P0-3 shape). Stable quadratic root; SI.

    beta*x^2/2 + x - c = 0 with c = (F(t_start) + shift)/s0 and x = T - T0, so x = 2c/(1 + sqrt(1 + 2*beta*c)).
    Raises ValueError where no root exists (s(T) would change sign before the shift is reached).
    """
    c = (integral_s(t_start, t0, s0, beta) + shift) / s0
    disc = 1.0 + 2.0 * beta * c
    if np.any(disc < 0):
        raise ValueError("no temperature reaches the requested shift: s(T) changes sign first (check beta)")
    return t0 + 2.0 * c / (1.0 + np.sqrt(disc))


def _fsr_hz_from_given(ctx: RelationContext):
    """FSR in Hz from fsr_thz or fsr_nm (at the assigned line wavelength), whichever is given."""
    lam = ctx.vals["derived.optics.lambda_line_m"]
    if ctx.known(P_FSR_THZ):
        return ctx.vals[P_FSR_THZ], P_FSR_THZ
    if ctx.known(P_FSR_NM):
        return C0 * ctx.vals[P_FSR_NM] / (lam * lam), P_FSR_NM
    return None, None


# ------------------------------------------------------------------------------------------------ R3 geometry

def _r3(ctx: RelationContext) -> None:
    name = "R3_geometry"
    lam_line = ctx.vals["derived.optics.lambda_line_m"]
    if ctx.known(P_FSR_THZ) and ctx.known(P_FSR_NM):
        f1 = ctx.vals[P_FSR_THZ]
        f2 = C0 * ctx.vals[P_FSR_NM] / (lam_line * lam_line)
        r = _rel(f1, f2)
        if r > max(ctx.tolerance.get(name, 0.0), _EXACT_RTOL):
            raise OverdeterminedError(f"{name}: fsr_thz and fsr_nm are both given and disagree by {r:.3g}")
    fsr, fsr_key = _fsr_hz_from_given(ctx)
    have_r, have_ng = ctx.known(P_RADIUS), ctx.known(P_NG)
    free = [m for m, h in ((P_RADIUS, have_r), (P_NG, have_ng), ("fsr", fsr is not None)) if not h]
    if len(free) > 1:
        raise UnderdeterminedError(f"{name}: {len(free)} of radius, n_g, FSR are free ({free}); give all but one")
    held = ctx.radius_hold is not None and free == [P_RADIUS]
    if held:
        # sampled geometry: the radius of the nominal-geometry device (solved from its n_g and the FSR hypothesis)
        # is a mask dimension, the same for every device of the world; the device's FSR follows its own n_g
        ctx.set(P_RADIUS, np.asarray(ctx.radius_hold, float), name, (P_NG, fsr_key),
                "R = c/(2 pi n_g FSR) of the nominal-geometry device, held for every device of the world",
                "07 P1-16: a geometry error moves n_g at a fixed mask radius, so the FSR follows the device's n_g")
    elif not free:
        L = _TWO_PI * ctx.vals[P_RADIUS]
        resid = _rel(fsr * ctx.vals[P_NG] * L / C0, 1.0)

        def recompute(m):
            Lr = _TWO_PI * ctx.vals[P_RADIUS]
            if m == P_RADIUS:
                ctx.set(P_RADIUS, C0 / (ctx.vals[P_NG] * fsr * _TWO_PI), name, (P_NG, fsr_key), "R = c/(2 pi n_g FSR)")
            elif m == P_NG:
                ctx.set(P_NG, C0 / (fsr * Lr), name, (P_RADIUS, fsr_key), "n_g = c/(FSR * 2 pi R)")
            else:
                f = C0 / (ctx.vals[P_NG] * Lr)
                if fsr_key == P_FSR_THZ:
                    ctx.set(P_FSR_THZ, f, name, (P_NG, P_RADIUS), "FSR = c/(n_g 2 pi R)")
                else:
                    ctx.set(P_FSR_NM, f * lam_line ** 2 / C0, name, (P_NG, P_RADIUS), "FSR_lambda = lambda^2/(n_g 2 pi R)")
        _overdetermined(ctx, name, resid, [P_RADIUS, P_NG, fsr_key], recompute)
        fsr, fsr_key = _fsr_hz_from_given(ctx)
    elif free[0] == P_RADIUS:
        ctx.set(P_RADIUS, C0 / (ctx.vals[P_NG] * fsr * _TWO_PI), name, (P_NG, fsr_key), "R = c/(2 pi n_g FSR)")
    elif free[0] == P_NG:
        ctx.set(P_NG, C0 / (fsr * _TWO_PI * ctx.vals[P_RADIUS]), name, (P_RADIUS, fsr_key), "n_g = c/(FSR 2 pi R)")
    R = ctx.vals[P_RADIUS]
    ng = ctx.vals[P_NG]
    L = _TWO_PI * R
    fsr_hz = np.broadcast_to(C0 / (ng * L), np.broadcast_shapes(np.shape(ng), np.shape(L))).copy()
    # provenance inputs: a given FSR is the FSR (R or n_g was solved from it), so the FSR-derived quantities inherit
    # the FSR hypothesis' provenance (07 D16), not the weaker n_g's; with a held radius the device's n_g enters too
    ins = (fsr_key, P_NG) if held else ((fsr_key,) if fsr is not None else (P_NG, P_RADIUS))
    ctx.set("derived.optics.length_m", L, name, (P_RADIUS,), "L = 2 pi R")
    ctx.set("derived.optics.fsr_hz", fsr_hz, name, ins, "FSR = c/(n_g L)")
    ctx.set("derived.optics.fsr_m", fsr_hz * lam_line ** 2 / C0, name,
            ("derived.optics.fsr_hz", "derived.optics.lambda_line_m"), "FSR_lambda = lambda_line^2 FSR/c")
    if fsr is None:
        ctx.set(P_FSR_THZ, fsr_hz, name, ins, "FSR = c/(n_g L)")
    ctx.set("derived.optics.round_trip_time_s", ng * L / C0, name, ins, "T_rt = n_g L/c = 1/FSR")
    # order and phase-consistent n_eff
    if not ctx.known(P_NEFF):
        raise UnderdeterminedError(f"{name}: ring.optics.n_eff is required (07 P1-6)")
    lam_res = ctx.vals["derived.optics.lambda_res_cold_m"]
    neff = ctx.vals[P_NEFF]
    if ctx.order_hold is not None:
        # sampled geometry: the Jacobians are taken at fixed r_ref and order, so the device keeps the order of the
        # nominal-geometry device; n_eff_phase = m lambda_res/L is then the device's n_eff at its own resonance
        m = np.broadcast_to(np.asarray(ctx.order_hold, float), np.broadcast_shapes(np.shape(neff), np.shape(L)))
        m_formula, m_note = "m of the nominal-geometry device (07 P1-16: fixed order)", \
            "a geometry error moves the resonance of the same mode order"
    else:
        m = np.rint(neff * L / lam_res)
        m_formula, m_note = "m = round(n_eff L / lambda_res)", ""
    if np.any(m < 1):
        raise ValueError(f"{name}: resonance order below 1 (n_eff*L/lambda = {np.min(neff * L / lam_res):.3g})")
    ctx.set(P_ORDER, np.asarray(m).astype(np.int64), name, (P_NEFF, P_RADIUS, P_OFFSET), m_formula, m_note)
    neff_ph = m * lam_res / L
    ctx.set("derived.optics.n_eff_phase", neff_ph, name, (P_ORDER, P_OFFSET, P_RADIUS),
            "n_eff_phase = m lambda_res / L (exact cold resonance at the configured offset)")
    ctx.set("derived.optics.n_eff_correction", neff_ph - neff, name, (P_NEFF,), "n_eff_phase - n_eff",
            "absorbs the order rounding; |x| <= lambda/(2L) for the nominal-geometry device (a device of a sampled "
            "geometry keeps that order, so its correction also holds the n_eff Jacobian's fixed-wavelength term)")


def _r3_resid(v: dict) -> float:
    L = v["derived.optics.length_m"]
    r1 = _rel(v["derived.optics.fsr_hz"] * v[P_NG] * L / C0, 1.0)
    r2 = _rel(v["derived.optics.n_eff_phase"] * L, v[P_ORDER] * v["derived.optics.lambda_res_cold_m"])
    r3 = _rel(L, _TWO_PI * v[P_RADIUS])
    return max(r1, r2, r3)


# ------------------------------------------------------------------------------------------------ R1 thermo-optic

def _beta(ctx: RelationContext):
    if not ctx.order2:
        return np.zeros_like(np.asarray(ctx.vals[P_S0], float)[..., :1])
    if not ctx.known(P_BETA):
        raise UnderdeterminedError("R1_thermo_optic: thermo_optic_order 2 needs ring.thermal.dlambda_dt_beta_per_k")
    return ctx.vals[P_BETA]


def _r1(ctx: RelationContext) -> None:
    name = "R1_thermo_optic"
    lam = ctx.vals["derived.optics.lambda_line_m"]
    ng = ctx.vals[P_NG]
    neff = ctx.vals["derived.optics.n_eff_phase"]
    aL = ctx.vals[P_ALPHA_L]
    if not ctx.known(P_T0):
        raise UnderdeterminedError(f"{name}: ring.thermal.dlambda_dt_ref_ring_temp_c (T0) is required")
    beta = _beta(ctx)
    ctx.set("derived.thermal.beta_eff_per_k", beta, name, (P_BETA,) if ctx.order2 else (),
            "beta (0 when fidelity.thermo_optic_order = 1)")
    have_s, have_dn = ctx.known(P_S0), ctx.known(P_DNDT)
    if have_s and have_dn:
        resid = _rel((lam / ng) * (ctx.vals[P_DNDT] + neff * aL), ctx.vals[P_S0])

        def recompute(m):
            if m == P_S0:
                ctx.set(P_S0, (lam / ng) * (ctx.vals[P_DNDT] + neff * aL), name, (P_DNDT, P_NG, P_ALPHA_L),
                        "s0 = (lambda/n_g)(dn_eff/dT + n_eff alpha_L)")
            else:
                ctx.set(P_DNDT, ctx.vals[P_S0] * ng / lam - neff * aL, name, (P_S0, P_NG, P_ALPHA_L),
                        "dn_eff/dT = s0 n_g/lambda - n_eff alpha_L")
        _overdetermined(ctx, name, resid, [P_S0, P_DNDT], recompute)
    elif have_s:
        ctx.set(P_DNDT, ctx.vals[P_S0] * ng / lam - neff * aL, name, (P_S0, P_NG, P_NEFF, P_ALPHA_L, _LAM_LINE),
                "dn_eff/dT(T0) = s0 n_g/lambda - n_eff alpha_L(T0)")
    else:
        raise UnderdeterminedError(f"{name}: ring.thermal.dlambda_dt_pm_per_k is the single thermo-optic anchor and "
                                   f"must be given (setting dn_eff_dt_per_k is a load error, 07 P0-3)")
    slope = ctx.vals.get(P_ALPHA_L_SLOPE, 0.0) if ctx.order2 else 0.0
    ctx.set("derived.optics.alpha_l_slope_eff_per_k2", np.asarray(slope, float), name, (P_ALPHA_L_SLOPE,),
            "dalpha_L/dT (0 when thermo_optic_order = 1)")
    ctx.set("derived.optics.d2n_eff_dt2_per_k2", ctx.vals[P_S0] * beta * ng / lam - neff * slope, name,
            (P_S0, P_BETA, P_NG, P_ALPHA_L_SLOPE, _LAM_LINE), "d2n_eff/dT2 = s0 beta n_g/lambda - n_eff dalpha_L/dT")


def _r1_resid(v: dict) -> float:
    lam = v["derived.optics.lambda_line_m"]
    s = (lam / v[P_NG]) * (v[P_DNDT] + v["derived.optics.n_eff_phase"] * v[P_ALPHA_L])
    r = _rel(s, v[P_S0])
    beta = v["derived.thermal.beta_eff_per_k"]
    slope = v["derived.optics.alpha_l_slope_eff_per_k2"]
    # second-order identity: (lambda/n_g)(d2n_eff/dT2 + n_eff dalpha_L/dT) = s0 beta, relative to |s0|*max(|beta|, 1e-6)
    ds = (lam / v[P_NG]) * (v["derived.optics.d2n_eff_dt2_per_k2"] + v["derived.optics.n_eff_phase"] * slope)
    scale = np.abs(v[P_S0]) * np.maximum(np.abs(beta), 1e-6)
    r2 = float(np.max(np.abs(ds - v[P_S0] * beta) / scale))
    return max(r, r2)


# ------------------------------------------------------------------------------------------------ R2 heater

P_FSR_SPAN = "derived.heater.p_fsr_span_m"


def _p_fsr_span(ctx: RelationContext):
    """FSR_lambda of the P_FSR measurement: at ``ring.heater.p_fsr_ref_wavelength_nm`` when given (one line for the
    whole bank, so the efficiency and Z_heater->ring are the same on every line), else at each ring's line."""
    if ctx.known(P_PFSR_LAM):
        lam = ctx.vals[P_PFSR_LAM]
        span = lam * lam * ctx.vals["derived.optics.fsr_hz"] / C0
        ctx.set(P_FSR_SPAN, span, "R2_heater_efficiency", ("derived.optics.fsr_hz", P_PFSR_LAM),
                "FSR_lambda at the P_FSR reference wavelength = lambda_ref^2 FSR/c")
    else:
        span = ctx.vals["derived.optics.fsr_m"]
        ctx.set(P_FSR_SPAN, span, "R2_heater_efficiency", ("derived.optics.fsr_m",),
                "FSR_lambda at the ring's line (no ring.heater.p_fsr_ref_wavelength_nm)")
    return span


def _r2_eff(ctx: RelationContext) -> None:
    name = "R2_heater_efficiency"
    fsr_m = _p_fsr_span(ctx)
    sp = P_FSR_SPAN
    have_p, have_e = ctx.known(P_PFSR), ctx.known(P_EFF)
    if have_p and have_e:
        resid = _rel(ctx.vals[P_EFF] * ctx.vals[P_PFSR], fsr_m)

        def recompute(m):
            if m == P_EFF:
                ctx.set(P_EFF, fsr_m / ctx.vals[P_PFSR], name, (sp, P_PFSR), "eff = FSR/P_FSR")
            else:
                ctx.set(P_PFSR, fsr_m / ctx.vals[P_EFF], name, (sp, P_EFF), "P_FSR = FSR/eff")
        _overdetermined(ctx, name, resid, [P_PFSR, P_EFF], recompute)
    elif have_p:
        ctx.set(P_EFF, fsr_m / ctx.vals[P_PFSR], name, (sp, P_PFSR),
                "efficiency = FSR_lambda / P_FSR (07 D16; FSR_lambda of the P_FSR measurement)")
    elif have_e:
        ctx.set(P_PFSR, fsr_m / ctx.vals[P_EFF], name, (sp, P_EFF), "P_FSR = FSR_lambda/efficiency")
    else:
        raise UnderdeterminedError(f"{name}: ring.heater.p_fsr_mw (the heater anchor) is required")


def _r2_z(ctx: RelationContext) -> None:
    name = "R2_heater_z"
    s0, t0 = ctx.vals[P_S0], ctx.vals[P_T0]
    beta = ctx.vals["derived.thermal.beta_eff_per_k"]
    t_ref = ctx.vals[P_OFFSET_T]
    # The heater only heats (red shift), so the lock is on the order whose cold resonance is the nearest one blue of
    # the line: offset mod FSR_lambda. Equal to the offset for 0 <= offset < FSR (the designed case, 7.6 nm); a
    # sampled ring with a negative offset (resonance red of its line) or one above an FSR locks one order over.
    off = np.mod(ctx.vals[P_OFFSET], ctx.vals["derived.optics.fsr_m"])
    ctx.set("derived.optics.lock_offset_m", off, name, (P_OFFSET, "derived.optics.fsr_m"),
            "resonance_offset mod FSR_lambda (heater-reachable order)")
    try:
        t_lock = solve_t_for_shift(t_ref, off, t0, s0, beta)
    except ValueError as e:
        raise ValueError(f"{name}: T_lock: {e}") from None
    ctx.set(P_TLOCK, t_lock, name, ("derived.optics.lock_offset_m", P_OFFSET_T, P_S0, P_T0, P_BETA),
            "int_{T_ref}^{T_lock} s(T) dT = resonance_offset mod FSR_lambda")
    s_lock = s0 * (1.0 + beta * (t_lock - t0))
    ctx.set("derived.thermal.s_lock_m_per_k", s_lock, name, (P_S0, P_BETA, P_TLOCK), "s(T_lock)")
    eff = ctx.vals[P_EFF]
    if ctx.heater_anchor == "lock_slope":
        z = eff / s_lock
        formula = "Z_heater->ring(0) = efficiency / s(T_lock)  (07 P0-3 R2)"
        ins = (P_EFF, P_S0, P_BETA, P_TLOCK)
    else:
        if not ctx.known(P_PFSR_T):
            raise UnderdeterminedError(f"{name}: fsr_integral needs ring.heater.p_fsr_ref_temp_c")
        tm = ctx.vals[P_PFSR_T]
        t_hot = solve_t_for_shift(tm, ctx.vals[P_FSR_SPAN], t0, s0, beta)
        ctx.set("derived.thermal.dt_fsr_k", t_hot - tm, name, (P_FSR_SPAN, P_PFSR_T, P_S0, P_BETA),
                "int_{T_meas}^{T_meas+dT} s dT = FSR_lambda of the P_FSR measurement")
        z = (t_hot - tm) / ctx.vals[P_PFSR]
        formula = "Z_heater->ring(0) = dT_FSR / P_FSR"
        ins = ("derived.thermal.dt_fsr_k", P_PFSR)
    if ctx.known(P_Z):
        resid = _rel(ctx.vals[P_Z], z)
        _overdetermined(ctx, name, resid, [P_Z], lambda m: ctx.set(P_Z, z, name, ins, formula))
    else:
        ctx.set(P_Z, z, name, ins, formula)


def _r2_resid(v: dict) -> float:
    span = v.get(P_FSR_SPAN, v["derived.optics.fsr_m"])
    r1 = _rel(v[P_EFF] * v[P_PFSR], span)
    s0, t0, beta = v[P_S0], v[P_T0], v["derived.thermal.beta_eff_per_k"]
    shift = integral_s(v[P_TLOCK], t0, s0, beta) - integral_s(v[P_OFFSET_T], t0, s0, beta)
    r2 = float(np.max(np.abs(shift - np.mod(v[P_OFFSET], v["derived.optics.fsr_m"])) / v["derived.optics.fsr_m"]))
    if "derived.thermal.dt_fsr_k" in v:
        tm = v[P_PFSR_T]
        shift = integral_s(tm + v[P_Z] * v[P_PFSR], t0, s0, beta) - integral_s(tm, t0, s0, beta)
        r3 = _rel(shift, span)
    else:
        r3 = _rel(v[P_Z] * v["derived.thermal.s_lock_m_per_k"], v[P_EFF])
    return max(r1, r2, r3)


# ------------------------------------------------------------------------------------------------ R4 Q and loss budget

def _omega(ctx_or_vals) -> np.ndarray:
    vals = ctx_or_vals.vals if isinstance(ctx_or_vals, RelationContext) else ctx_or_vals
    return _TWO_PI * C0 / vals["derived.optics.lambda_res_cold_m"]


def _r4_budget_components(ctx: RelationContext) -> None:
    name = "R4_loss_budget"
    have = [ctx.known(p) for p in LOSS_COMPONENTS]
    if not any(have):
        return
    if not all(have):
        miss = [p for p, h in zip(LOSS_COMPONENTS, have) if not h]
        raise UnderdeterminedError(f"{name}: give all four loss-budget components or none (missing {miss})")
    lam = ctx.vals["derived.optics.lambda_res_cold_m"]
    ng = ctx.vals[P_NG]
    a = [ctx.vals[p] for p in LOSS_COMPONENTS]
    alpha_i = a[0] + a[1] + a[2] + a[3]
    if np.any(alpha_i <= 0):
        raise ValueError(f"{name}: total intrinsic loss must be > 0")
    qi = _TWO_PI * ng / (lam * alpha_i)
    eta = (a[0] + a[1]) / alpha_i
    if ctx.known(P_QI):
        resid = _rel(ctx.vals[P_QI], qi)
        _overdetermined(ctx, name, resid, [P_QI] + list(LOSS_COMPONENTS),
                        lambda m: ctx.set(P_QI, qi, name, LOSS_COMPONENTS + (P_NG, _LAM_RES),
                                          "Q_i = 2 pi n_g/(lambda alpha_i)"))
        qi = ctx.vals[P_QI]
    else:
        ctx.set(P_QI, qi, name, LOSS_COMPONENTS + (P_NG, _LAM_RES), "Q_i = 2 pi n_g/(lambda_res alpha_i), alpha_i = sum")
    if ctx.known(P_ETA):
        resid = _rel(ctx.vals[P_ETA], eta)
        _overdetermined(ctx, name, resid, [P_ETA] + list(LOSS_COMPONENTS),
                        lambda m: ctx.set(P_ETA, eta, name, LOSS_COMPONENTS, "eta = (fca + other)/alpha_i"))
    else:
        ctx.set(P_ETA, eta, name, LOSS_COMPONENTS, "eta = (alpha_fca + alpha_other_abs)/alpha_i (07 P0-2)")
    ctx.set("derived.optics.alpha_heat_per_m", a[0] + a[1], name, (P_A_FCA, P_A_OTH), "alpha_fca + alpha_other_abs")
    ctx.set("derived.optics.alpha_nonheat_per_m", a[2] + a[3], name, (P_A_SCA, P_A_RAD), "alpha_scatter + alpha_rad")


def _r4_budget_eta(ctx: RelationContext) -> None:
    name = "R4_loss_budget"
    if "derived.optics.alpha_heat_per_m" in ctx.vals:
        return
    if not ctx.known(P_ETA):
        raise UnderdeterminedError(f"{name}: give the loss budget (four components) or ring.optics.heat_fraction "
                                   f"(07 P0-2: the heat fraction comes from a loss budget)")
    alpha_i = ctx.vals["derived.optics.alpha_i_per_m"]
    eta = ctx.vals[P_ETA]
    ctx.set("derived.optics.alpha_heat_per_m", eta * alpha_i, name, (P_ETA, P_QI), "eta alpha_i")
    ctx.set("derived.optics.alpha_nonheat_per_m", (1.0 - eta) * alpha_i, name, (P_ETA, P_QI), "(1 - eta) alpha_i")


def _r4_q(ctx: RelationContext) -> None:
    name = "R4_q"
    if ctx.coupling_anchor is not None:
        _r4_q_geometry(ctx, name, ctx.coupling_anchor)
    else:
        _r4_q_anchors(ctx, name)
    # exact per-pass couplings and round-trip amplitude (07 P2-4)
    v = ctx.vals
    trt = v["derived.optics.round_trip_time_s"]
    w = _omega(ctx)
    ins = (P_NG, P_RADIUS, _LAM_RES)
    ctx.set("derived.optics.kappa2_through", -np.expm1(-w * trt / v[P_QT]), name, (P_QT,) + ins,
            "K_t^2 = 1 - exp(-omega T_rt/Q_t)")
    ctx.set("derived.optics.kappa2_drop", -np.expm1(-w * trt / v[P_QD]), name, (P_QD,) + ins,
            "K_d^2 = 1 - exp(-omega T_rt/Q_d)")
    ctx.set("derived.optics.round_trip_amplitude", np.exp(-0.5 * w * trt / v[P_QI]), name, (P_QI,) + ins,
            "a = exp(-omega T_rt/(2 Q_i))")
    ctx.set("derived.optics.alpha_i_per_m", _TWO_PI * v[P_NG] / (v["derived.optics.lambda_res_cold_m"] * v[P_QI]),
            name, (P_QI, P_NG, _LAM_RES), "alpha_i = 2 pi n_g/(lambda Q_i)")


def _r4_q_geometry(ctx: RelationContext, name: str, ca: dict) -> None:
    """R4_q for a device of a sampled geometry (07 P1-16): the anchors (q_loaded, kappa_d_over_kappa, or whichever
    members the config gives) describe the nominal-geometry device, whose per-pass couplings realize solved with the
    nominal n_g, length and loss (``ca``). The device's couplers are those couplers moved by its geometry error,
    K^2 = K_a^2 exp(dln_kappa2), and its Q_L and drop share follow from them and from its own loss: a coupler does not
    change because the loss or n_g of the device changed, and Q_L does."""
    v = ctx.vals
    trt = v["derived.optics.round_trip_time_s"]
    w = _omega(ctx)
    anchors = tuple(p for p in (P_QL, P_XD, P_QT, P_QD, P_QI) if p in ctx.prov)
    if not ctx.known(P_QI):
        # no loss budget and no given Q_i: the device keeps the nominal device's loss per length
        ctx.set(P_QI, _TWO_PI * v[P_NG] / (v[_LAM_RES] * ca["alpha_i_per_m"]), name, anchors + (P_NG, _LAM_RES),
                "Q_i = 2 pi n_g/(lambda_res alpha_i) with the alpha_i of the nominal-geometry device")
    q = {}
    for j, pq in (("through", P_QT), ("drop", P_QD)):
        k2 = np.asarray(ca[f"kappa2_{j}"], float) * np.exp(np.asarray(ca[f"dln_kappa2_{j}"], float))
        bad = np.isfinite(k2) & ((k2 <= 0.0) | (k2 >= 1.0))     # a non-finite input fails realize's finite check
        if np.any(bad):
            raise ValueError(f"{name}: a sampled geometry moves the {j} per-pass coupling outside (0, 1) (worst "
                             f"{float(k2[bad].flat[0]):.3g}); check the coupling Jacobian and the gap sigmas")
        q[pq] = w * trt / (-np.log1p(-k2))
        ctx.set(pq, q[pq], name, anchors + ("derived.geometry.dln_kappa2_" + j, P_NG, P_RADIUS, _LAM_RES),
                f"Q = omega T_rt/(-ln(1 - K^2)), K^2 = K_a^2 exp(dln_kappa2_{j}): the nominal-geometry device's "
                f"{j} coupler moved by the device's geometry (07 P1-16)")
    ql = 1.0 / (1.0 / v[P_QI] + 1.0 / q[P_QT] + 1.0 / q[P_QD])
    ctx.set(P_QL, ql, name, (P_QI, P_QT, P_QD), "1/Q_L = 1/Q_i + 1/Q_t + 1/Q_d (the device's couplers and loss)")
    ctx.set(P_XD, ql / q[P_QD], name, (P_QL, P_QD), "kappa_d/kappa = Q_L/Q_d (the device's couplers and loss)")


def _r4_q_anchors(ctx: RelationContext, name: str) -> None:
    qs = {p: ctx.known(p) for p in (P_QL, P_QI, P_QT, P_QD)}
    xd = ctx.known(P_XD)
    v = ctx.vals
    if xd and qs[P_QD]:
        if qs[P_QL]:
            resid = _rel(v[P_QL] / v[P_QD], v[P_XD])
            if resid > max(ctx.tolerance.get(name, 0.0), _EXACT_RTOL):
                raise OverdeterminedError(f"{name}: q_drop and kappa_d_over_kappa are both given and disagree "
                                          f"({resid:.3g})")
        xd = False
    if xd:
        x = v[P_XD]
        free = [p for p in (P_QL, P_QI, P_QT) if not qs[p]]
        if len(free) != 1:
            raise (UnderdeterminedError if len(free) > 1 else OverdeterminedError)(
                f"{name}: with kappa_d_over_kappa given, exactly one of q_loaded, q_intrinsic, q_through must be free "
                f"(free: {free})")
        f = free[0]
        if f == P_QL:
            ql = (1.0 - x) / (1.0 / v[P_QI] + 1.0 / v[P_QT])
            ctx.set(P_QL, ql, name, (P_QI, P_QT, P_XD), "(1 - x_d)/Q_L = 1/Q_i + 1/Q_t")
        else:
            other = P_QT if f == P_QI else P_QI
            inv = (1.0 - x) / v[P_QL] - 1.0 / v[other]
            if np.any(inv <= 0):
                raise ValueError(f"{name}: inconsistent Q: (1 - x_d)/Q_L <= 1/{other.rsplit('.', 1)[1]} "
                                 f"(min {float(np.min(inv)):.3g}); the coupling and loss cannot be solved")
            ctx.set(f, 1.0 / inv, name, (P_QL, other, P_XD), f"1/Q = (1 - x_d)/Q_L - 1/Q_other")
        ctx.set(P_QD, v[P_QL] / x, name, (P_QL, P_XD), "Q_d = Q_L / x_d")
    else:
        free = [p for p, h in qs.items() if not h]
        if len(free) > 1:
            raise UnderdeterminedError(f"{name}: {len(free)} of q_loaded, q_intrinsic, q_through, q_drop are free "
                                       f"({free}); give all but one (or kappa_d_over_kappa)")
        if not free:
            resid = _rel(1.0 / v[P_QL], 1.0 / v[P_QI] + 1.0 / v[P_QT] + 1.0 / v[P_QD])

            def recompute(m):
                _solve_q(ctx, m, name)
            _overdetermined(ctx, name, resid, [P_QL, P_QI, P_QT, P_QD], recompute)
        else:
            _solve_q(ctx, free[0], name)


def _solve_q(ctx: RelationContext, f: str, name: str) -> None:
    v = ctx.vals
    others = [p for p in (P_QI, P_QT, P_QD) if p != f]
    if f == P_QL:
        ctx.set(P_QL, 1.0 / (1.0 / v[P_QI] + 1.0 / v[P_QT] + 1.0 / v[P_QD]), name, (P_QI, P_QT, P_QD),
                "1/Q_L = 1/Q_i + 1/Q_t + 1/Q_d")
        return
    inv = 1.0 / v[P_QL] - sum(1.0 / v[p] for p in others if p != P_QL)
    if np.any(inv <= 0):
        raise ValueError(f"{name}: inconsistent Q: 1/Q_L <= sum of the other 1/Q (min {float(np.min(inv)):.3g}); "
                         f"{f} cannot be solved")
    ctx.set(f, 1.0 / inv, name, tuple([P_QL] + [p for p in others if p != P_QL]), "1/Q = 1/Q_L - sum(other 1/Q)")


def _r4_resid(v: dict) -> float:
    r1 = _rel(1.0 / v[P_QL], 1.0 / v[P_QI] + 1.0 / v[P_QT] + 1.0 / v[P_QD])
    w = _omega(v)
    trt = v["derived.optics.round_trip_time_s"]
    r2 = _rel(-np.log1p(-v["derived.optics.kappa2_through"]), w * trt / v[P_QT])
    r3 = _rel(-np.log1p(-v["derived.optics.kappa2_drop"]), w * trt / v[P_QD])
    r4 = _rel(-2.0 * np.log(v["derived.optics.round_trip_amplitude"]), w * trt / v[P_QI])
    r5 = 0.0
    if all(p in v for p in LOSS_COMPONENTS):
        a = sum(v[p] for p in LOSS_COMPONENTS)
        r5 = max(_rel(v[P_QI], _TWO_PI * v[P_NG] / (v["derived.optics.lambda_res_cold_m"] * a)),
                 _rel(v[P_ETA], (v[P_A_FCA] + v[P_A_OTH]) / a))
    r6 = _rel(v["derived.optics.alpha_heat_per_m"] + v["derived.optics.alpha_nonheat_per_m"],
              v["derived.optics.alpha_i_per_m"])
    return max(r1, r2, r3, r4, r5, r6)


# ------------------------------------------------------------------------------------------------ R5 latency

def _r5(ctx: RelationContext) -> None:
    name = "R5_latency"
    if ctx.dt is None:
        raise UnderdeterminedError(f"{name}: sim.dt_plant_s is required")
    owners: dict[str, str] = {}
    for ch, key in LATENCY_OWNERS.items():
        if key in owners:
            raise OverdeterminedError(f"{name}: {key} owns two channels ({owners[key]}, {ch})")
        owners[key] = ch
        if ctx.known(key):
            ticks = np.rint(np.asarray(ctx.vals[key]) * ctx.latency_scale / ctx.dt).astype(np.int64)
            if np.any(ticks < 0):
                raise ValueError(f"{name}: negative latency for {ch}")
            ctx.set(f"derived.timing.{ch}.latency_ticks", ticks, name, (key, "uncertainty: latency scale"),
                    "round(latency_s * latency_scale / dt_plant_s)")


def _r5_resid(v: dict) -> float:
    return 0.0


RELATIONS: tuple[Relation, ...] = (
    Relation("R3_geometry", (P_RADIUS, P_NG, P_FSR_NM, P_FSR_THZ, P_NEFF, P_OFFSET, P_ORDER), _r3, _r3_resid,
             "FSR = c/(n_g 2 pi R); m = round(n_eff L/lambda_res)"),
    Relation("R1_thermo_optic", (P_S0, P_T0, P_DNDT, P_NG, P_BETA), _r1, _r1_resid,
             "dlambda/dT(T) = (lambda/n_g)(dn_eff/dT(T) + n_eff alpha_L(T)) = s0 (1 + beta (T - T0))"),
    Relation("R2_heater_efficiency", (P_PFSR, P_EFF), _r2_eff, lambda v: 0.0, "efficiency = FSR/P_FSR"),
    Relation("R2_heater_z", (P_Z, P_EFF, P_S0, P_TLOCK), _r2_z, _r2_resid,
             "Z_heater->ring(0) = efficiency/dlambda/dT(T_lock)"),
    Relation("R4_loss_budget", LOSS_COMPONENTS + (P_QI, P_ETA), _r4_budget_components, lambda v: 0.0,
             "alpha_i = sum of components; Q_i = 2 pi n_g/(lambda alpha_i); eta = (fca + other)/alpha_i"),
    Relation("R4_q", (P_QL, P_QI, P_QT, P_QD, P_XD), _r4_q, _r4_resid, "1/Q_L = 1/Q_i + 1/Q_t + 1/Q_d"),
    Relation("R4_heat_fraction", (P_ETA, P_QI), _r4_budget_eta, lambda v: 0.0,
             "alpha_heat = eta alpha_i when no budget component is given"),
    Relation("R5_latency", tuple(LATENCY_OWNERS.values()), _r5, _r5_resid, "one owner per I/O channel; ticks"),
)
RELATION_NAMES = [r.name for r in RELATIONS]


def solve(ctx: RelationContext) -> RelationContext:
    """Run every relation in dependency order on the context (in place) and return it."""
    for rel in RELATIONS:
        rel.solver(ctx)
    return ctx


def residuals(vals: dict) -> dict[str, float]:
    """Max relative residual of every relation over the batch, after solving (07 P0-3 test: <= 1e-12)."""
    out = {}
    for rel in RELATIONS:
        try:
            out[rel.name] = float(rel.residual(vals))
        except KeyError as e:   # a relation whose optional members are absent
            out[rel.name] = float("nan") if str(e) else 0.0
    return out
