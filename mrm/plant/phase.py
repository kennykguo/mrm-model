"""Ring state -> optics inputs: the phase model that joins PlantParams, the thermal state and the junction to
:mod:`mrm.plant.optics` (07 P1-5, P1-6, P1-7, P0-2, P0-3 R1).

Owner: integrator (foundation stage). The optics takes, per ring k and laser line l, the absolute round-trip phase
``phi`` (without the coupler phase), the per-pass couplings ``k2_th``/``k2_dr`` at the LINE wavelength, the
round-trip amplitude ``a``, the heat fraction ``eta`` and the group delay ``t_rt``, each for the two NRZ junction
levels (docs/handoff/plant_optics.md section 1). No leaf module computed them from PlantParams; this module does, so
the kernel owner only calls :func:`ring_optics_row` per world.

Equations (SI; every input from :class:`mrm.params.realize.PlantParams`, docs/PARAMETERS_FORMAT.md section 10)
--------------------------------------------------------------------------------------------------------------
Reference state: ring temperature ``optics.t_ref_k``, reverse bias ``optics.bias_ref_v``, heater off, P_in -> 0. The
cold resonance of order m sits at ``lambda_res = optics.lambda_res_cold_m`` with n_eff L = m lambda_res exactly
(``optics.n_eff`` is the phase-consistent index, R3).

1. Resonance shift of ring k (07 P0-3 R1 for the temperature, 07 P1-7 for the junction):
       delta = s0 [(T - T_ref) + beta ((T - T0)^2 - (T_ref - T0)^2)/2] + lambda_res (f dneff(V) - f dneff(V_ref))/n_g
   with s0 = thermal.dlambda_dt_s0_m_per_k, beta = thermal.dlambda_dt_beta_per_k, T0 = thermal.dlambda_dt_t0_k,
   T = the ring's mode-weighted temperature (ROM observer ``ring_mode_weighted``), f dneff(V) the ring-level index
   change of the junction table (mrm.plant.electrical.junction_eval_row; 03 section 5.2: delta lambda/lambda =
   delta n_eff/n_g at fixed length).
2. Round-trip phase at the line wavelength lambda_l: the cold phase curve translated rigidly by delta,
       phi = 2 pi n_eff(x) L / x,   x = lambda_l - delta,   Delta = x - lambda_res,
       n_eff(x) = n_eff + (n_eff - n_g) Delta/lambda_res - (dn_g/dlambda) Delta^2 / (2 lambda_res)
   (second-order Taylor: dn_eff/dlambda = (n_eff - n_g)/lambda, d2n_eff/dlambda2 = -(dn_g/dlambda)/lambda, 07 P1-6
   "n_eff(lambda) to second order"). phi = 2 pi m exactly when the shifted resonance reaches the line, for any shift,
   and d phi/d delta = 2 pi n_g(x) L / x^2 (the optics' dphi_dlambda_res). The coupler phase is part of the measured
   cold resonance, so ``psi`` = 0 (passing a nonzero psi would count it twice).
3. Group delay t_rt = n_g(x) L / c with n_g(x) = n_g + (dn_g/dlambda) Delta (FSR = 1/t_rt).
4. Couplings at the LINE (07 P1-5; never at the resonance and never per heater code):
       K^2_j(lambda_l) = K^2_j,0 exp(dln_kappa2_j_dlambda (lambda_l - lambda_res))  (K^2_j,0 = optics.kappa2_*).
5. Loss and heat fraction (07 P0-2, P1-5):
       alpha_i = [alpha_i,ref + f alpha_fca(V) - f alpha_fca(V_ref)] exp(c_ring (T - T_ref) + c_heater (T_H - T_ref))
       alpha_heat = alpha_i - (alpha_i,ref - alpha_heat,ref)       (scattering and radiation do not change)
       a = exp(-alpha_i L/2),  eta = alpha_heat/alpha_i
   c_ring, c_heater = optics.dln_alpha_dt_ring_per_k / dln_alpha_dt_heater_per_k (d ln alpha/dT, 07 P1-5), T_H the
   ROM heater-resistor temperature. Integrator choice (documented in docs/handoff/ENGINE_BRIEF.md): the factor
   multiplies the whole alpha_i (the key is d ln alpha/dT) and the added loss is absorptive, so it heats.

Self-heating adapter (mrm.plant.optics.pack_selfheat_args): pass phi_ref = phi at the start-of-substep ring
temperature T_prev, t_ref_k = T_prev and the local law s(T_prev) = s0 (1 + beta (T_prev - T0)),
beta' = beta / (1 + beta (T_prev - T0)); the adapter's s0' [dT + beta' dT^2/2] is then exactly R1 about T_prev
(:func:`local_thermo_optic`).

Kernels are @njit(cache=True), take arrays and scalars only and run unchanged under NUMBA_DISABLE_JIT=1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from mrm.plant.electrical import JunctionTable, junction_eval_row, junction_table_from_plant_params
from mrm.units import C0

__all__ = ["RP_LAM_RES", "RP_N_EFF", "RP_N_G", "RP_DNG", "RP_LEN", "RP_T_REF", "RP_S0", "RP_BETA", "RP_T0",
           "RP_K2T", "RP_K2D", "RP_DLNK2T", "RP_DLNK2D", "RP_ALPHA_I", "RP_ALPHA_NONHEAT", "RP_CT_RING",
           "RP_CT_HEATER", "RP_FILL", "RP_V_REF", "RP_FDN_REF", "RP_FAL_REF", "N_RP", "PhaseParams",
           "ring_optics_row", "resonance_shift_row", "local_thermo_optic", "thermal_shift_m"]

# column indices of PhaseParams.ring (B, N, N_RP); all SI
RP_LAM_RES = 0       # cold resonance lambda_res (m)
RP_N_EFF = 1         # phase-consistent n_eff at lambda_res, T_ref, V_ref
RP_N_G = 2           # group index at lambda_res
RP_DNG = 3           # dn_g/dlambda (1/m)
RP_LEN = 4           # round-trip length L = 2 pi r_ref (m)
RP_T_REF = 5         # reference temperature of lambda_res (K)
RP_S0 = 6            # s0 of R1 at T0 (m/K)
RP_BETA = 7          # beta of R1 (1/K)
RP_T0 = 8            # T0 of R1 (K)
RP_K2T = 9           # through per-pass K^2 at lambda_res
RP_K2D = 10          # drop per-pass K^2 at lambda_res
RP_DLNK2T = 11       # d ln K^2_t / d lambda (1/m)
RP_DLNK2D = 12       # d ln K^2_d / d lambda (1/m)
RP_ALPHA_I = 13      # alpha_i at the reference state (1/m)
RP_ALPHA_NONHEAT = 14  # alpha_i - alpha_heat at the reference state (1/m)
RP_CT_RING = 15      # d ln alpha / dT_ring (1/K)
RP_CT_HEATER = 16    # d ln alpha / dT_heater (1/K)
RP_FILL = 17         # junction fill fraction
RP_V_REF = 18        # reference reverse bias (V)
RP_FDN_REF = 19      # ring-level f dneff(V_ref) of the junction table
RP_FAL_REF = 20      # ring-level f alpha_fca(V_ref) of the junction table (1/m)
N_RP = 21

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class PhaseParams:
    """Packed per-ring phase-model parameters for the kernel (one row per world and ring) plus the junction table.

    ring: (B, N, N_RP) float64 (columns RP_*). Junction table (mrm.plant.electrical.JunctionTable): grid (G,),
    dneff / alpha_fca / capacitance / r_series columns (Bt, G) with Bt = 1 (LUT) or B (analytic, one row per world).
    """
    ring: np.ndarray
    grid: np.ndarray
    dneff: np.ndarray
    alpha_fca: np.ndarray
    capacitance: np.ndarray
    r_series: np.ndarray

    @property
    def batch(self) -> int:
        return int(self.ring.shape[0])

    def table_row(self, b: int) -> int:
        """Row of the junction table for world b (0 for a shared LUT)."""
        return b if self.dneff.shape[0] > 1 else 0

    @classmethod
    def from_plant_params(cls, pp, table: JunctionTable | None = None) -> "PhaseParams":
        """Pack PlantParams (optics, thermal, junction) into the kernel layout. ``table`` defaults to
        :func:`mrm.plant.electrical.junction_table_from_plant_params`."""
        o, t, j = pp.optics, pp.thermal, pp.junction
        B, N = int(pp.batch), int(pp.n_rings)
        tab = junction_table_from_plant_params(pp) if table is None else table
        ring = np.zeros((B, N, N_RP))

        def bn(x):
            return np.broadcast_to(np.asarray(x, dtype=float), (B, N))
        ring[:, :, RP_LAM_RES] = bn(o.lambda_res_cold_m)
        ring[:, :, RP_N_EFF] = bn(o.n_eff)
        ring[:, :, RP_N_G] = bn(o.n_g)
        ring[:, :, RP_DNG] = bn(o.dn_g_dlambda_per_m)
        ring[:, :, RP_LEN] = bn(o.length_m)
        ring[:, :, RP_T_REF] = bn(o.t_ref_k)
        ring[:, :, RP_S0] = bn(t.dlambda_dt_s0_m_per_k)
        ring[:, :, RP_BETA] = bn(t.dlambda_dt_beta_per_k)
        ring[:, :, RP_T0] = bn(t.dlambda_dt_t0_k)
        ring[:, :, RP_K2T] = bn(o.kappa2_through)
        ring[:, :, RP_K2D] = bn(o.kappa2_drop)
        ring[:, :, RP_DLNK2T] = bn(o.dln_kappa2_through_dlambda_per_m)
        ring[:, :, RP_DLNK2D] = bn(o.dln_kappa2_drop_dlambda_per_m)
        ring[:, :, RP_ALPHA_I] = bn(o.alpha_i_per_m)
        ring[:, :, RP_ALPHA_NONHEAT] = bn(o.alpha_i_per_m) - bn(o.alpha_heat_per_m)
        ring[:, :, RP_CT_RING] = bn(o.dln_alpha_dt_ring_per_k)
        ring[:, :, RP_CT_HEATER] = bn(o.dln_alpha_dt_heater_per_k)
        ring[:, :, RP_FILL] = bn(j.fill_fraction)
        ring[:, :, RP_V_REF] = bn(o.bias_ref_v)
        # the table at the reference bias (ring level), so the reference state reproduces lambda_res and alpha_i
        g = np.asarray(tab.v_reverse_v, float)
        for b in range(B):
            r = b if tab.dneff.shape[0] > 1 else 0
            out = np.zeros((4, N))
            fl = np.zeros(N, np.int8)
            junction_eval_row(np.ascontiguousarray(ring[b, :, RP_V_REF]), g, tab.dneff[r], tab.alpha_fca_per_m[r],
                              tab.capacitance_f_per_m[r], tab.r_series_ohm_m[r], np.ascontiguousarray(ring[b, :, RP_FILL]),
                              np.ascontiguousarray(ring[b, :, RP_LEN]), out[0], out[1], out[2], out[3], fl)
            ring[b, :, RP_FDN_REF] = out[0]
            ring[b, :, RP_FAL_REF] = out[1]
        if not np.all(np.isfinite(ring)):
            bad = sorted({i for i in range(N_RP) if not np.all(np.isfinite(ring[:, :, i]))})
            raise ValueError(f"PhaseParams: non-finite columns {bad} (check optics/thermal/junction in PlantParams)")
        return cls(ring=np.ascontiguousarray(ring), grid=g, dneff=np.ascontiguousarray(tab.dneff),
                   alpha_fca=np.ascontiguousarray(tab.alpha_fca_per_m),
                   capacitance=np.ascontiguousarray(tab.capacitance_f_per_m),
                   r_series=np.ascontiguousarray(tab.r_series_ohm_m))


@njit(cache=True)
def thermal_shift_m(t_k, t_ref_k, s0, beta, t0):
    """R1 resonance shift (m) from T_ref to T: s0 [(T - T_ref) + beta ((T - T0)^2 - (T_ref - T0)^2)/2]."""
    return s0 * ((t_k - t_ref_k) + 0.5 * beta * ((t_k - t0) ** 2 - (t_ref_k - t0) ** 2))


@njit(cache=True)
def local_thermo_optic(t_prev_k, s0, beta, t0):
    """(s_local, beta_local) of R1 expanded about t_prev: s(t_prev) = s0 (1 + beta (t_prev - T0)) and
    beta / (1 + beta (t_prev - T0)), so s_local [dT + beta_local dT^2/2] = F(t_prev + dT) - F(t_prev) exactly.
    These are the s0 / beta the optics self-heating adapter takes when t_ref_k = t_prev."""
    g = 1.0 + beta * (t_prev_k - t0)
    return s0 * g, beta / g


@njit(cache=True)
def resonance_shift_row(rp, t_ring_k, v, grid, dn_tab, al_tab, c_tab, rs_tab, out_shift, out_fal, out_cj, out_rs,
                        out_flags):
    """Resonance shift (m) of every ring of one world at ring temperature t_ring_k (N,) and reverse bias v (N,)
    (equation 1 of the module docstring), with the ring-level junction FCA f alpha_fca(V) (1/m), the junction
    capacitance C_j (F) and series resistance R_s (ohm) at that bias. rp (N, N_RP). out_flags gets the junction
    FLAG_GRID_CLIP bit (mrm.plant.electrical) when v leaves the table."""
    n = rp.shape[0]
    fdn = np.empty(n)
    junction_eval_row(v, grid, dn_tab, al_tab, c_tab, rs_tab, rp[:, RP_FILL], rp[:, RP_LEN], fdn, out_fal, out_cj,
                      out_rs, out_flags)
    for k in range(n):
        dth = thermal_shift_m(t_ring_k[k], rp[k, RP_T_REF], rp[k, RP_S0], rp[k, RP_BETA], rp[k, RP_T0])
        out_shift[k] = dth + rp[k, RP_LAM_RES] * (fdn[k] - rp[k, RP_FDN_REF]) / rp[k, RP_N_G]


@njit(cache=True)
def _fill_level(rp, k, lam_l, shift, fal, t_ring, t_heater, out_phi, out_a, out_eta, out_trt, with_trt):
    lam_r = rp[k, RP_LAM_RES]
    ne = rp[k, RP_N_EFF]
    ng = rp[k, RP_N_G]
    dng = rp[k, RP_DNG]
    ln = rp[k, RP_LEN]
    t_ref = rp[k, RP_T_REF]
    alpha = rp[k, RP_ALPHA_I] + fal - rp[k, RP_FAL_REF]
    alpha *= math.exp(rp[k, RP_CT_RING] * (t_ring - t_ref) + rp[k, RP_CT_HEATER] * (t_heater - t_ref))
    if alpha < 0.0:
        alpha = 0.0
    a_rt = math.exp(-0.5 * alpha * ln)
    heat = alpha - rp[k, RP_ALPHA_NONHEAT]
    if heat < 0.0:
        heat = 0.0
    eta = heat / alpha if alpha > 0.0 else 0.0
    if eta > 1.0:
        eta = 1.0
    for li in range(lam_l.shape[0]):
        x = lam_l[li] - shift
        d = x - lam_r
        n_x = ne + (ne - ng) * d / lam_r - 0.5 * dng * d * d / lam_r
        out_phi[k, li] = _TWO_PI * n_x * ln / x
        out_a[k, li] = a_rt
        out_eta[k, li] = eta
        if with_trt:
            out_trt[k, li] = (ng + dng * d) * ln / C0


@njit(cache=True)
def ring_optics_row(rp, lam_l, t_ring_k, t_heater_k, v0, v1, grid, dn_tab, al_tab, c_tab, rs_tab,
                    out_phi0, out_phi1, out_a0, out_a1, out_eta0, out_eta1, out_k2t, out_k2d, out_trt,
                    out_shift, out_cj, out_rs, out_flags):
    """Optics inputs of one world (equations 1-5 of the module docstring).

    rp (N, N_RP) = PhaseParams.ring[b]; lam_l (L,) vacuum wavelengths of the lines (m) after the laser static error
    and the scenario's laser.dfreq_hz; t_ring_k (N,) mode-weighted ring temperatures (K); t_heater_k (N,) heater
    resistor temperatures (K; only for dln_alpha_dt_heater); v0, v1 (N,) reverse bias of the NRZ levels 0 and 1 (V;
    pass v1 = v0 for CW or a squelched driver); the junction table row of this world. Outputs (N, L): phi0/phi1
    (rad, absolute, no coupler phase), a0/a1, eta0/eta1, k2t/k2d (at the line), t_rt (s, level 0); (N, 2):
    out_shift (resonance shift per level, m), out_cj (C_j per level, F), out_rs (R_s per level, ohm); out_flags (N,)
    int8 junction flags (FLAG_GRID_CLIP)."""
    n = rp.shape[0]
    shift = np.empty(n)
    fal = np.empty(n)
    cj = np.empty(n)
    rs = np.empty(n)
    for lvl in range(2):
        v = v0 if lvl == 0 else v1
        resonance_shift_row(rp, t_ring_k, v, grid, dn_tab, al_tab, c_tab, rs_tab, shift, fal, cj, rs, out_flags)
        for k in range(n):
            out_shift[k, lvl] = shift[k]
            out_cj[k, lvl] = cj[k]
            out_rs[k, lvl] = rs[k]
            if lvl == 0:
                _fill_level(rp, k, lam_l, shift[k], fal[k], t_ring_k[k], t_heater_k[k], out_phi0, out_a0, out_eta0,
                            out_trt, True)
            else:
                _fill_level(rp, k, lam_l, shift[k], fal[k], t_ring_k[k], t_heater_k[k], out_phi1, out_a1, out_eta1,
                            out_trt, False)
    for k in range(n):
        lam_r = rp[k, RP_LAM_RES]
        for li in range(lam_l.shape[0]):
            dl = lam_l[li] - lam_r
            kt = rp[k, RP_K2T] * math.exp(rp[k, RP_DLNK2T] * dl)
            kd = rp[k, RP_K2D] * math.exp(rp[k, RP_DLNK2D] * dl)
            out_k2t[k, li] = kt if kt < 1.0 else 1.0 - 1e-15
            out_k2d[k, li] = kd if kd < 1.0 else 1.0 - 1e-15
