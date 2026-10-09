"""Linearized small-signal model of a ring's heater lock loop, built ONLY from the nominal design (07 P1-1).

``linearize(nominal_cfg, operating_point, Ts) -> LinearModel``, which unpacks as ``(A, B, C, D, delay_ticks)``.

Knowledge (07 P1-1, D13)
------------------------
A model-based controller may use the nominal, unsampled design and nothing else. The model is built from
``realize(nominal_cfg, batch=1, uncertainty="nominal", variation="off")``: every uncertainty entry at its nominal
choice, no per-device variation. The controller gets ``nominal_cfg`` from ``ControllerSpec.nominal`` and must also find
the token :data:`KNOWLEDGE_TOKEN` (``"model_nominal"``) in ``controller_interface.knowledge``
(:func:`check_knowledge`); g26 lists only ``[line_grid, ring_to_line]``, so an experiment that runs a model-based
controller adds the token explicitly. This module never sees a sampled world: the same functions are reused by the
Truth-fed oracle (``mrm.validation.oracle``) with the TRUE PlantParams, which is the oracle's business, not a
controller's.

Operating point (:class:`OperatingPoint`, :func:`static_point`)
--------------------------------------------------------------
The steady state of the whole nominal bank (B = 1, N rings, L lines) with every ring at a target detuning
``delta = lambda_res - lambda_line`` (resonance minus its assigned line; > 0 = resonance red of the laser, the
self-locking side of 07 P0-1). ``lambda_res`` is the mean of the two NRZ levels' resonances (mrm.plant.phase eq. 1).
Default target: the OMA-optimal detuning among open-loop-stable points of the nominal model (07 P0-1 item 7,
:func:`oma_optimal_detuning`). The solve uses the plant's own static laws:

1. ring temperature T_R from the R1 law (``mrm.plant.phase.resonance_shift_row``) so that delta = target;
2. exact bus optics at the ring temperatures (``ring_optics_row`` + ``bus_propagate``, NRZ-averaged with the ones
   density): absorbed power P_abs, drop power P_drop, bus power, OMA per ring;
3. data-activity heat (``junction_swing_capacitance_row``, ``data_activity_heat_row``);
4. thermal DC gains (``mrm.plant.thermal.dc_gain``): heater powers P_h such that every ring observer is at T_R
   (one N x N linear solve, iterated with the heater-resistor temperature for the loss-vs-T term);
5. heater drive from the driver law with the TCR at the heater-resistor temperature
   (``electrical.heater_power_row`` with g_self = 0, ideal DAC transfer), inverted for the code;
6. monitor code from ``MonitorHw.ideal_code`` (PD current + dark current at the PD observer temperature).

Linear model (one ring, deviations from the operating point; SI, codes)
-----------------------------------------------------------------------
Inputs u = [heater code (codes), d (K)], with d an additive RING-TEMPERATURE disturbance (ambient, aggressor; a
laser wavelength drift is represented by the d that moves delta equally). It enters every ring-temperature path
(resonance, absorbed and drop power) exactly as the thermal states do, so a large disturbance that the heater
cancels leaves no residue in either output (a rigid-shift disturbance would not: the drop power's dependence on
T_R and on the line wavelength differ by the coupler dispersion, about 0.4 %, which a nm-scale offset turns into
tens of pm). Outputs y = [monitor code (codes), delta (m)].

- Driver law with TCR: dP_h = k_u du + k_T dT_H, k_u = dP/dcode at fixed T_H (W/code), k_T = dP/dT_H at fixed code
  (W/K; < 0 for voltage drive, 07 P2-5).
- Thermal (the ring's own block of the ThermalSystem: observers ring_mode_weighted R and heater_resistor H, sources
  heater h and waveguide_absorption w; ``thermal.state_space``): dx/dt = A_th x + b_h dP_h + b_w dP_abs,
  dT_R = c_R x, dT_H = c_H x. Crosstalk from the other rings is left out (1.5-3 % with ms poles, 07 P1-2).
- Optics (finite differences of the exact bus solution): dP_abs = a_R (dT_R + d) + a_H dT_H,
  dP_drop = m_R (dT_R + d) + m_H dT_H; s_R = d lambda_res / dT_R. (a_d, m_d: the rigid-shift slopes d/d delta at
  fixed temperatures, kept in the StaticPoint for reports.)
- Monitor: dy = g_y dP_drop with g_y = codes per W at the PD (``MonitorHw.codes_per_w``).

    A = A_th + b_h k_T c_H + b_w (a_R c_R + a_H c_H)      B = [b_h k_u, b_w a_R]
    C = [g_y (m_R c_R + m_H c_H); s_R c_R]                 D = [[0, g_y m_R], [0, s_R]]

The b_w a_R c_R term is the self-heating feedback of 07 P0-1: on the stable side a_R < 0, so the DC gain from heater
power to detuning is divided by 1 - Z_opt(0) a_R = 1 + G|P'| (G = s_R Z_opt, P' = dP_abs/d lambda_res)
(``LinearModel.selfheat_factor``). The discrete A, B are the exact zero-order hold at Ts (``scipy.linalg.expm`` of
the augmented matrix); C, D are unchanged.

Delay (07 P0-6). The command of edge k acts from tick k + tau_u, tau_u = controller latency + heater DAC latency (+ the
nominal thermal transport delay, which the heater FIFO applies). A step consumes the monitor samples released since
the last edge; an integrate-and-dump sample with sample tick s averages the plant intervals [s - W, s), whose optics
uses the end-of-interval temperature, so it reads the state at s - (W - 1)/2 on average; it is released at s + ADC
latency. tau_m is the mean age of those samples over the controller/ADC phase pattern (:func:`measurement_age_ticks`;
for an rc AFE the filter time constant is added; per ring, since an ADC mux samples the rings that share a converter
at staggered phases). The model moves the whole loop delay tau = tau_u + tau_m to the
input (exact for this LTI path); ``delay_ticks`` is round(tau) and ``delay_exact_ticks`` keeps the fraction.
:meth:`LinearModel.delayed` gives the exact fractional-delay discretization (modified z-transform; Franklin, Powell
and Workman, Digital Control of Dynamic Systems, section 4.3.3) with the past inputs as extra states.

Units: SI everywhere (m, s, W, K), heater input in DAC codes, monitor output in ADC codes.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import scipy.linalg as sla

__all__ = ["KNOWLEDGE_TOKEN", "KnowledgeError", "check_knowledge", "OperatingPoint", "Conditions", "StaticPoint",
           "LinearModel", "linearize", "linearize_bank", "nominal_plant_params", "conditions_from_config",
           "static_point", "oma_curve", "oma_optimal_detuning", "measurement_age_ticks", "discretize_zoh",
           "delay_discretize", "resonance_detuning_m", "bank_optics", "ring_block_lags"]

#: knowledge token (controller_interface.knowledge) that lets a controller realize the nominal design as its model
KNOWLEDGE_TOKEN = "model_nominal"

#: finite-difference steps for the static slopes
_DT_FD_K = 2e-3            # ring / heater temperature step, K
_DLAM_FD_M = 1e-13         # rigid detuning step, m (0.1 pm)
_DCODE_FD = 8.0            # heater code step, codes


class KnowledgeError(PermissionError):
    """The controller asked for nominal-model knowledge that controller_interface.knowledge does not grant."""


def check_knowledge(knowledge: Iterable[str], required: Sequence[str] = (KNOWLEDGE_TOKEN,), who: str = "controller"
                    ) -> None:
    """Raise :class:`KnowledgeError` unless every token of ``required`` is in ``knowledge`` (07 P1-1 filter)."""
    have = set(knowledge or ())
    missing = [t for t in required if t not in have]
    if missing:
        raise KnowledgeError(
            f"{who} needs controller_interface.knowledge to contain {missing} (it has {sorted(have)}): a model-based "
            f"controller realizes the nominal design (uncertainty nominal, variation off) as its model. Add it with "
            f"the override controller_interface.knowledge=[{', '.join(sorted(have | set(missing)))}]")


# ---------------------------------------------------------------------------------------------- nominal realize
_PP_CACHE: dict[str, Any] = {}


def _cfg_key(cfg) -> str | None:
    fp = getattr(cfg, "fingerprint", None)
    return fp() if callable(fp) else None


def nominal_plant_params(cfg):
    """``realize(cfg, 1, uncertainty="nominal", variation="off")`` (cached by the config fingerprint).

    The nominal design only: every ``uncertainty`` entry at its nominal choice, no per-device variation (07 P1-1).
    """
    from mrm.params.realize import realize       # local import: the nominal design, never a sampled world
    key = _cfg_key(cfg)
    if key is not None and key in _PP_CACHE:
        return _PP_CACHE[key]
    pp = realize(cfg, 1, seeds=0, uncertainty="nominal", variation="off")
    if pp.thermal.system is None:
        raise ValueError("linearize needs the assembled thermal system (realize thermal_hook='auto')")
    if key is not None:
        if len(_PP_CACHE) > 16:
            _PP_CACHE.clear()
        _PP_CACHE[key] = pp
    return pp


# ---------------------------------------------------------------------------------------------- operating point
@dataclass(frozen=True)
class OperatingPoint:
    """Where to linearize (07 P1-1 ``operating_point``).

    ring: ring index of the SISO model. detuning_m: target delta = lambda_res - lambda_line (m), scalar or (N,);
    None = the OMA-optimal open-loop-stable detuning of the nominal model. detuning_frac_of_peak: alternative target,
    the drop power as a fraction of its peak on the stable side (0 < f < 1); ignored when detuning_m is given.
    t_base_k: die base temperature (K); None = scenario.ambient_c of the config (else 298.15 K). bias_v: DC reverse
    bias (V); None = scenario.bias_v (else optics.bias_ref_v). ones_density, transition_density: data statistics
    (transition density None = datapath.transition_density or 2 p (1 - p)). laser_power_scale: factor on every nominal
    line power. pga_select: monitor PGA setting.
    """
    ring: int = 0
    detuning_m: float | Sequence[float] | None = None
    detuning_frac_of_peak: float | None = None
    t_base_k: float | None = None
    bias_v: float | None = None
    ones_density: float = 0.5
    transition_density: float | None = None
    laser_power_scale: float = 1.0
    pga_select: int = 0


@dataclass(frozen=True)
class Conditions:
    """Operating conditions of one world for the static solve (SI): line wavelengths lam_m (L,), chip input power
    per line p_chip_w (L,), base temperature t_base_k, NRZ junction levels v_lo/v_hi (N,) (V reverse), ones density
    rho (N,), transition density (N,), the aggressor power (A,) W (held at DC), data_enable (N,), and t_extra_obs_k
    (O,) an additive temperature at every thermal observer (K; e.g. the non-steady part of a disturbance)."""
    lam_m: np.ndarray
    p_chip_w: np.ndarray
    t_base_k: float
    v_lo: np.ndarray
    v_hi: np.ndarray
    rho: np.ndarray
    transition_density: np.ndarray
    aggressor_w: np.ndarray | None = None
    data_enable: np.ndarray | None = None
    t_extra_obs_k: np.ndarray | None = None


def _cfg_si(cfg, key, default=None):
    si = getattr(cfg, "si", None)
    if callable(si):
        try:
            v = si(key, None)
        except Exception:      # noqa: BLE001 - an absent key in a strict schema
            v = None
        return default if v is None else v
    return default


def conditions_from_config(pp, b: int = 0, *, cfg=None, op: OperatingPoint | None = None) -> Conditions:
    """Nominal operating conditions of world b: laser grid and power (``pp.laser``, chip input transmission),
    ambient (``op.t_base_k`` or scenario.ambient_c), bias (``op.bias_v`` or scenario.bias_v or the reference bias)
    mapped to the two NRZ levels (``electrical.nrz_levels_row`` with ``datapath.swing_pp_v`` and the level
    convention), data statistics from ``op``."""
    from mrm.plant import electrical as el
    from mrm.units import C0
    op = OperatingPoint() if op is None else op
    N, L = int(pp.n_rings), int(pp.n_lines)
    lam = C0 / np.asarray(pp.laser.line_freq_hz, float)[b]
    p = np.asarray(pp.laser.power_w, float)[b] * float(np.asarray(pp.laser.input_transmission, float).reshape(-1)[
        b if np.size(pp.laser.input_transmission) > 1 else 0]) * float(op.laser_power_scale)
    t_base = op.t_base_k if op.t_base_k is not None else _cfg_si(cfg, "scenario.ambient_c", 298.15)
    vb = op.bias_v if op.bias_v is not None else _cfg_si(cfg, "scenario.bias_v", None)
    if vb is None:
        vb = float(np.asarray(pp.optics.bias_ref_v, float).reshape(-1)[0])
    swing = np.broadcast_to(np.asarray(pp.datapath.swing_pp_v, float)[b] if np.ndim(pp.datapath.swing_pp_v) > 0
                            else float(pp.datapath.swing_pp_v), (N,)).astype(float)
    conv = int(np.asarray(pp.datapath.level_convention).reshape(-1)[b if np.size(pp.datapath.level_convention) > 1
                                                                        else 0])
    v_lo, v_hi, fl = np.zeros(N), np.zeros(N), np.zeros(N, np.int8)
    el.nrz_levels_row(np.full(N, float(vb)), np.ascontiguousarray(swing), conv, el.JUNCTION_GRID_V_MIN,
                      el.JUNCTION_GRID_V_MAX, v_lo, v_hi, fl)
    rho = np.full(N, float(op.ones_density))
    td = op.transition_density
    if td is None:
        td = _cfg_si(cfg, "datapath.transition_density", None)
    if td is None:
        td = 2.0 * op.ones_density * (1.0 - op.ones_density)
    return Conditions(lam_m=np.ascontiguousarray(lam), p_chip_w=np.ascontiguousarray(p), t_base_k=float(t_base),
                      v_lo=v_lo, v_hi=v_hi, rho=rho, transition_density=np.full(N, float(td)))


def _phase(pp):
    from mrm.plant import phase as ph
    cache = getattr(pp, "_ctrl_c_phase", None)
    if cache is None:
        cache = ph.PhaseParams.from_plant_params(pp)
        try:
            object.__setattr__(pp, "_ctrl_c_phase", cache)
        except Exception:      # noqa: BLE001 - a frozen object without __dict__: recompute each time
            pass
    return cache


def resonance_detuning_m(pp, b: int, t_ring_k, cond: Conditions, line_of_ring=None) -> np.ndarray:
    """delta = lambda_res - lambda_line (N,) m for ring temperatures t_ring_k (N,): lambda_res is the mean of the two
    NRZ levels' resonances (mrm.plant.phase eq. 1), lambda_line the ring's assigned line (``layout.ring_to_line``)."""
    from mrm.plant import phase as ph
    P = _phase(pp)
    rp = P.ring[b]
    r = P.table_row(b)
    N = rp.shape[0]
    t = np.ascontiguousarray(np.broadcast_to(np.asarray(t_ring_k, float), (N,)))
    sh = np.zeros((2, N))
    for lv, v in enumerate((cond.v_lo, cond.v_hi)):
        fal, cj, rs, fl = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N, np.int8)
        ph.resonance_shift_row(rp, t, np.ascontiguousarray(v, dtype=float), P.grid, P.dneff[r], P.alpha_fca[r],
                               P.capacitance[r], P.r_series[r], sh[lv], fal, cj, rs, fl)
    line = np.asarray(pp.layout.ring_to_line if line_of_ring is None else line_of_ring, np.int64).reshape(-1)
    return rp[:, ph.RP_LAM_RES] + sh.mean(axis=0) - cond.lam_m[line]


def _ring_temperature_for(pp, b, target_m, cond: Conditions, t0=None) -> np.ndarray:
    """Ring temperatures (N,) that put every ring at the detuning target (N,) m (Newton on the smooth R1 law)."""
    N = int(pp.n_rings)
    tgt = np.broadcast_to(np.asarray(target_m, float), (N,))
    t = np.full(N, 400.0) if t0 is None else np.array(t0, float)
    for _ in range(50):
        f = resonance_detuning_m(pp, b, t, cond) - tgt
        d = (resonance_detuning_m(pp, b, t + 1e-2, cond) - resonance_detuning_m(pp, b, t - 1e-2, cond)) / 2e-2
        step = f / d
        t = t - step
        if np.max(np.abs(step)) < 1e-10:
            break
    return t


def bank_optics(pp, b: int, t_ring_k, t_heater_k, cond: Conditions, lam_m=None):
    """Exact bus optics of world b at ring/heater temperatures (N,): returns (BusResult, rs (N,) ohm, cj_eff (N,) F).

    Uses mrm.plant.phase.ring_optics_row and mrm.plant.optics.bus_propagate (NRZ-averaged with cond.rho), the same
    laws as the plant tick."""
    from mrm.plant import electrical as el
    from mrm.plant import optics as op
    from mrm.plant import phase as ph
    P = _phase(pp)
    rp = P.ring[b]
    r = P.table_row(b)
    N = rp.shape[0]
    lam = np.ascontiguousarray(cond.lam_m if lam_m is None else lam_m, dtype=float)
    L = lam.shape[0]
    o = {n: np.zeros((N, L)) for n in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
    sh, cj, rs, fl = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N, np.int8)
    ph.ring_optics_row(rp, lam, np.ascontiguousarray(t_ring_k, dtype=float),
                       np.ascontiguousarray(t_heater_k, dtype=float), np.ascontiguousarray(cond.v_lo, dtype=float),
                       np.ascontiguousarray(cond.v_hi, dtype=float), P.grid, P.dneff[r], P.alpha_fca[r],
                       P.capacitance[r], P.r_series[r], o["phi0"], o["phi1"], o["a0"], o["a1"], o["e0"], o["e1"],
                       o["k2t"], o["k2d"], o["trt"], sh, cj, rs, fl)
    rb = np.asarray(pp.optics.backscatter_rb)[b]
    bus_t2 = None
    method = "tcmt" if int(pp.flags.optics_model) else "exact"
    res = op.bus_propagate(np.ascontiguousarray(cond.p_chip_w, dtype=float), o["phi0"], o["k2t"], o["k2d"], o["a0"],
                           o["trt"], o["e0"], bus_order=np.asarray(pp.layout.bus_order, np.int64), bus_t2=bus_t2,
                           rb=rb if np.any(rb != 0) else None,
                           nrz={"p1": np.ascontiguousarray(cond.rho, dtype=float), "phi1": o["phi1"], "a1": o["a1"],
                                "eta1": o["e1"]},
                           method=method, m_orders=int(pp.optics.tcmt_orders))
    c_eff = np.zeros(N)
    el.junction_swing_capacitance_row(np.ascontiguousarray(cond.v_lo, dtype=float),
                                      np.ascontiguousarray(cond.v_hi, dtype=float), P.grid, P.capacitance[r],
                                      np.ascontiguousarray(rp[:, ph.RP_FILL]), np.ascontiguousarray(rp[:, ph.RP_LEN]),
                                      c_eff)
    return res, 0.5 * (rs[:, 0] + rs[:, 1]), c_eff


def _data_heat(pp, b, cond: Conditions, rs, c_eff):
    from mrm.plant import electrical as el
    N = int(pp.n_rings)
    d = pp.datapath
    pj, pd = np.zeros(N), np.zeros(N)
    en = np.ones(N) if cond.data_enable is None else np.asarray(cond.data_enable, float)

    def row(x):
        a = np.asarray(x, float)
        a = a[b] if a.ndim > 1 or (a.ndim == 1 and a.shape[0] == int(pp.batch) and int(pp.batch) > 1) else a
        return np.ascontiguousarray(np.broadcast_to(a.reshape(-1) if a.ndim else a, (N,)), dtype=float)
    el.data_activity_heat_row(np.ascontiguousarray(c_eff), np.ascontiguousarray(cond.v_hi - cond.v_lo),
                              float(np.asarray(d.bitrate_hz, float).reshape(int(pp.batch), -1)[b, 0]),
                              np.ascontiguousarray(cond.transition_density, dtype=float), en,
                              np.ascontiguousarray(rs, dtype=float), row(d.driver_output_resistance_ohm[b]),
                              row(d.driver_static_power_w[b]), row(d.driver_energy_per_transition_j[b]), 0, pj, pd)
    return pj, pd


# ---------------------------------------------------------------------------------------------- heater law
@dataclass(frozen=True)
class _HeaterLaw:
    """Ideal-DAC driver law of one world (mrm.plant.electrical): code -> drive u -> power at a heater temperature."""
    law: int
    bits: int
    pwm_bits: int
    r0: np.ndarray
    alpha: np.ndarray
    t0: np.ndarray
    v_fs: np.ndarray
    i_fs: np.ndarray
    p_fs: np.ndarray
    v_comp: np.ndarray
    code_min: int
    code_max: int

    @classmethod
    def from_pp(cls, pp, b: int = 0) -> "_HeaterLaw":
        h = pp.heater
        N = int(pp.n_rings)

        def rown(x):
            a = np.asarray(x, float)
            if a.ndim == 0:
                a = a.reshape(1, 1)
            a = a.reshape(a.shape[0], -1)
            a = a[b if a.shape[0] > 1 else 0]
            a = np.where(np.isfinite(a), a, np.inf)
            return np.ascontiguousarray(np.broadcast_to(a, (N,)), dtype=float)
        pwm_bits = int(h.pwm_bits) if int(h.pwm_bits) > 0 else 8
        return cls(law=int(np.asarray(h.driver_mode).reshape(-1)[b if np.size(h.driver_mode) > 1 else 0]),
                   bits=int(h.dac_bits), pwm_bits=pwm_bits, r0=rown(h.r_cold_ohm), alpha=rown(h.tcr_per_k),
                   t0=rown(h.r_ref_temp_k), v_fs=rown(h.full_scale_v), i_fs=rown(h.full_scale_a),
                   p_fs=rown(h.full_scale_w), v_comp=rown(h.compliance_v), code_min=int(h.code_min),
                   code_max=int(h.code_max))

    def drive(self, code) -> np.ndarray:
        """Ideal-DAC drive u (N,) for float codes (linear interpolation between integer codes)."""
        from mrm.plant import electrical as el
        c = np.asarray(code, float)
        lo = np.clip(np.floor(c), 0, (1 << self.bits) - 1).astype(np.int64)
        hi = np.clip(lo + 1, 0, (1 << self.bits) - 1)
        f = np.clip(c - lo, 0.0, 1.0)
        n_err = self.bits - self.pwm_bits if self.law == el.LAW_DAC_PWM else self.bits
        err = np.zeros((1, max(n_err, 1)))
        bow = np.zeros(1)
        ul, uh, rip = np.zeros(c.size), np.zeros(c.size), np.zeros(c.size)
        el.heater_target_row(np.ascontiguousarray(lo.reshape(-1)), self.law, self.bits, self.pwm_bits, err, bow, 0, 0,
                             1, ul, rip)
        el.heater_target_row(np.ascontiguousarray(hi.reshape(-1)), self.law, self.bits, self.pwm_bits, err, bow, 0, 0,
                             1, uh, rip)
        return (ul + f.reshape(-1) * (uh - ul)).reshape(c.shape)

    def power(self, code, t_heater_k) -> np.ndarray:
        """Heater power (N,) W at float codes (N,) and heater-resistor temperatures (N,) K (g_self = 0)."""
        return self.power_current(code, t_heater_k)[0]

    def power_current(self, code, t_heater_k) -> tuple[np.ndarray, np.ndarray]:
        """(power W, current A) (N,) at float codes and heater-resistor temperatures (g_self = 0; dac_pwm: rms
        current, as the plant's supply limit uses it)."""
        from mrm.plant import electrical as el
        u = np.ascontiguousarray(np.broadcast_to(self.drive(code), self.r0.shape), dtype=float)
        n = u.shape[0]
        out = [np.zeros(n) for _ in range(4)]
        fl = np.zeros(n, np.int8)
        el.heater_power_row(u, self.law, np.ascontiguousarray(np.broadcast_to(t_heater_k, (n,)), dtype=float),
                            np.zeros(n), self.r0, self.alpha, self.t0, self.v_fs, self.i_fs, self.p_fs, self.v_comp,
                            out[0], out[1], out[2], out[3], fl)
        return out[0], out[2]

    def code_for_power(self, p_w, t_heater_k) -> np.ndarray:
        """Float codes (N,) giving heater power p_w (N,) at heater temperatures (N,) (bisection; clipped to the code
        range, where the result is the range end)."""
        p = np.asarray(p_w, float)
        lo = np.full(p.shape, float(self.code_min))
        hi = np.full(p.shape, float(self.code_max))
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            up = self.power(mid, t_heater_k) < p
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------------------------- static point
@dataclass(frozen=True)
class StaticPoint:
    """Steady state of one world's bank at a detuning target, plus the small-signal slopes (all (N,), SI/codes).

    detuning_m: delta = lambda_res - lambda_line. t_ring_k, t_heater_k, t_pd_k: observer temperatures.
    heater_power_w, heater_code (float, ideal DAC), p_abs_w, p_drop_w, p_in_w (bus power at the ring on its line),
    oma_w (|P_out1 - P_out0| of its line at the bus output), y_code (ideal monitor code), y_sigma_codes (white noise
    plus quantization of one sample), codes_per_w (monitor gain). Slopes: s_m_per_k = d lambda_res/dT_R;
    dpabs_dtr / dpabs_dth (W/K), dpabs_dd (W/m, rigid detuning), dpdrop_dtr / dpdrop_dth / dpdrop_dd; dp_dcode
    (W/code at fixed T_H), dp_dth_heater (W/K at fixed code); z_opt_k_per_w (DC waveguide->ring),
    z_heater_k_per_w (DC heater->ring); loop_gain_opt = z_opt * dpabs_dtr (07 P0-1: open-loop stable iff < 1);
    p_drop_peak_w (drop power with the ring at delta = 0), p_junction_w / p_driver_w (data-activity heat),
    y_zero_code (the code with no drop light: dark current, background, offset), y_peak_code (the code at the peak),
    heater_current_a (driver current at the point, A; rms for dac_pwm), supply_scale (the plant's array-supply droop
    factor at this point, ``electrical.supply_limit_row``: min(1, sqrt(P_max / sum P), I_max / sum I); < 1 means the
    supply cannot deliver the point, 07 P1-12).
    """
    detuning_m: np.ndarray
    t_ring_k: np.ndarray
    t_heater_k: np.ndarray
    t_pd_k: np.ndarray
    heater_power_w: np.ndarray
    heater_code: np.ndarray
    p_abs_w: np.ndarray
    p_drop_w: np.ndarray
    p_in_w: np.ndarray
    oma_w: np.ndarray
    y_code: np.ndarray
    y_sigma_codes: np.ndarray
    codes_per_w: np.ndarray
    s_m_per_k: np.ndarray
    dpabs_dtr: np.ndarray
    dpabs_dth: np.ndarray
    dpabs_dd: np.ndarray
    dpdrop_dtr: np.ndarray
    dpdrop_dth: np.ndarray
    dpdrop_dd: np.ndarray
    dp_dcode: np.ndarray
    dp_dth_heater: np.ndarray
    z_opt_k_per_w: np.ndarray
    z_heater_k_per_w: np.ndarray
    loop_gain_opt: np.ndarray
    cond: Conditions | None = None
    p_drop_peak_w: np.ndarray | None = None
    p_junction_w: np.ndarray | None = None
    p_driver_w: np.ndarray | None = None
    y_zero_code: np.ndarray | None = None
    y_peak_code: np.ndarray | None = None
    heater_current_a: np.ndarray | None = None
    supply_scale: float = 1.0

    @property
    def selfheat_factor(self) -> np.ndarray:
        """1 - Z_opt(0) dP_abs/dT_R: the factor that divides the heater-to-detuning DC gain (07 P0-1)."""
        return 1.0 - self.loop_gain_opt


def _thermal_dc(pp, b):
    from mrm.plant.thermal import dc_gain
    sysm = pp.thermal.system
    return dc_gain(sysm)[b if sysm.batch > 1 else 0]          # (O, S) K/W


def _solve_heaters(pp, b, cond: Conditions, t_ring, p_abs, pj, pd, t_heater_guess=None):
    """Heater powers (N,) and the observer temperatures (O,) that put every ring observer at t_ring (DC)."""
    sysm = pp.thermal.system
    G = _thermal_dc(pp, b)
    S = sysm.n_src
    u = np.zeros(S)
    u[sysm.wg_src] = p_abs
    u[sysm.junction_src] = pj
    if sysm.driver_src.size:
        u[sysm.driver_src] = pd[:sysm.driver_src.size]
    if sysm.aggressor_src.size and cond.aggressor_w is not None:
        u[sysm.aggressor_src] = cond.aggressor_w
    extra = np.zeros(sysm.n_obs) if cond.t_extra_obs_k is None else np.asarray(cond.t_extra_obs_k, float)
    rhs = np.asarray(t_ring, float) - cond.t_base_k - extra[sysm.ring_obs] - G[sysm.ring_obs] @ u
    p_h = np.linalg.solve(G[np.ix_(sysm.ring_obs, sysm.heater_src)], rhs)
    u[sysm.heater_src] = p_h
    t_obs = cond.t_base_k + extra + G @ u
    return p_h, t_obs


def static_point(pp, b: int = 0, detuning_m=None, *, cond: Conditions | None = None, cfg=None,
                 op: OperatingPoint | None = None, pga_select: int = 0, with_slopes: bool = True) -> StaticPoint:
    """Steady state of world b's bank with every ring at the detuning target (N,) (m), and its slopes.

    ``pp`` is any PlantParams (the nominal one for a controller, the TRUE one for the oracle). ``cond`` defaults to
    :func:`conditions_from_config`. ``detuning_m`` None: the OMA-optimal stable detuning (:func:`oma_optimal_detuning`)."""
    from mrm.plant.sensing import MonitorHw
    N = int(pp.n_rings)
    cond = conditions_from_config(pp, b, cfg=cfg, op=op) if cond is None else cond
    if detuning_m is None:
        detuning_m = oma_optimal_detuning(pp, b, cond=cond)[0]
    target = np.ascontiguousarray(np.broadcast_to(np.asarray(detuning_m, float), (N,)))
    sysm = pp.thermal.system
    t_ring = _ring_temperature_for(pp, b, target, cond)
    t_h = t_ring.copy()
    for _ in range(4):           # the loss may depend on the heater temperature (er_mechanism heater_t_loss)
        res, rs, c_eff = bank_optics(pp, b, t_ring, t_h, cond)
        pj, pd = _data_heat(pp, b, cond, rs, c_eff)
        p_h, t_obs = _solve_heaters(pp, b, cond, t_ring, res.abs_sum, pj, pd)
        t_h_new = t_obs[sysm.heater_obs]
        if np.max(np.abs(t_h_new - t_h)) < 1e-9:
            t_h = t_h_new
            break
        t_h = t_h_new
    res, rs, c_eff = bank_optics(pp, b, t_ring, t_h, cond)
    t_pd = t_obs[sysm.pd_obs] if np.all(sysm.pd_obs >= 0) else t_ring
    law = _HeaterLaw.from_pp(pp, b)
    code = law.code_for_power(p_h, t_h)
    p_at_code, i_h = law.power_current(code, t_h)
    p_max = float(np.asarray(getattr(pp.heater, "supply_power_max_w", math.inf), float).reshape(-1)[0])
    i_max = float(np.asarray(getattr(pp.heater, "supply_current_max_a", math.inf), float).reshape(-1)[0])
    p_sum, i_sum = float(np.sum(np.maximum(p_at_code, 0.0))), float(np.sum(np.maximum(i_h, 0.0)))
    supply = 1.0
    if p_sum > p_max and p_sum > 0:
        supply = min(supply, math.sqrt(p_max / p_sum))
    if i_sum > i_max and i_sum > 0:
        supply = min(supply, i_max / i_sum)
    line = np.asarray(pp.layout.ring_to_line, np.int64).reshape(-1)
    idx = np.arange(N)
    oma = np.abs(res.pout1[idx, line] - res.pout0[idx, line])
    # monitor (nominal hardware of this PlantParams; one world)
    hw = MonitorHw.from_plant_params(pp)
    bb = b if hw.B > 1 else 0
    p_bus = res.p_in.sum(axis=1)
    i_sig, i_dark = hw.photocurrent_a(np.broadcast_to(res.drop_sum, (hw.B, N)), np.broadcast_to(p_bus, (hw.B, N)),
                                      np.broadcast_to(t_pd, (hw.B, N)))
    y = hw.ideal_code(i_sig + i_dark, pga_select)[bb]
    gy = hw.codes_per_w(pga_select)[bb]
    cpa = (hw.ideal_code(1.0, pga_select) - hw.ideal_code(0.0, pga_select))[bb]
    sig = hw.white_sigma_a(i_sig, i_dark)[bb] * np.abs(cpa)
    y_sigma = np.sqrt(sig ** 2 + 1.0 / 12.0)
    G = _thermal_dc(pp, b)
    z_opt = G[sysm.ring_obs, sysm.wg_src]
    z_h = G[sysm.ring_obs, sysm.heater_src]
    kw: dict[str, Any] = {}
    if with_slopes:
        def fd_t(which):
            h = _DT_FD_K
            up_r, dn_r = t_ring.copy(), t_ring.copy()
            up_h, dn_h = t_h.copy(), t_h.copy()
            if which == "r":
                up_r += h
                dn_r -= h
            else:
                up_h += h
                dn_h -= h
            ru = bank_optics(pp, b, up_r, up_h, cond)[0]
            rd = bank_optics(pp, b, dn_r, dn_h, cond)[0]
            return (ru.abs_sum - rd.abs_sum) / (2 * h), (ru.drop_sum - rd.drop_sum) / (2 * h)
        a_r, m_r = fd_t("r")
        a_h, m_h = fd_t("h")
        # rigid detuning d: every line moves by -d (ring-own effect; other rings see the same shift of their lines)
        h = _DLAM_FD_M
        ru = bank_optics(pp, b, t_ring, t_h, cond, lam_m=cond.lam_m - h)[0]
        rd = bank_optics(pp, b, t_ring, t_h, cond, lam_m=cond.lam_m + h)[0]
        a_d = (ru.abs_sum - rd.abs_sum) / (2 * h)
        m_d = (ru.drop_sum - rd.drop_sum) / (2 * h)
        s = (resonance_detuning_m(pp, b, t_ring + _DT_FD_K, cond)
             - resonance_detuning_m(pp, b, t_ring - _DT_FD_K, cond)) / (2 * _DT_FD_K)
        # one-sided at the ends of the code range (a centred step past code_max would read the clipped DAC and halve
        # the slope); a point beyond the range (code_for_power clipped) keeps the slope at the range end
        c_hi = np.minimum(code + _DCODE_FD, float(law.code_max))
        c_lo = np.maximum(c_hi - 2 * _DCODE_FD, float(law.code_min))
        c_hi = np.minimum(c_lo + 2 * _DCODE_FD, float(law.code_max))
        dpc = (law.power(c_hi, t_h) - law.power(c_lo, t_h)) / np.maximum(c_hi - c_lo, 1e-12)
        dpt = (law.power(code, t_h + _DT_FD_K) - law.power(code, t_h - _DT_FD_K)) / (2 * _DT_FD_K)
        t_pk = _ring_temperature_for(pp, b, np.zeros(N), cond, t0=t_ring)
        pk = bank_optics(pp, b, t_pk, _heater_t_for(pp, b, t_pk, cond), cond)[0].drop_sum
        kw = dict(s_m_per_k=s, dpabs_dtr=a_r, dpabs_dth=a_h, dpabs_dd=a_d, dpdrop_dtr=m_r, dpdrop_dth=m_h,
                  dpdrop_dd=m_d, dp_dcode=dpc, dp_dth_heater=dpt, loop_gain_opt=z_opt * a_r, p_drop_peak_w=pk)
    else:
        nan = np.full(N, np.nan)
        kw = dict(s_m_per_k=nan, dpabs_dtr=nan, dpabs_dth=nan, dpabs_dd=nan, dpdrop_dtr=nan, dpdrop_dth=nan,
                  dpdrop_dd=nan, dp_dcode=nan, dp_dth_heater=nan, loop_gain_opt=nan)
    i0_sig, _ = hw.photocurrent_a(np.zeros((hw.B, N)), np.broadcast_to(p_bus, (hw.B, N)),
                                  np.broadcast_to(t_pd, (hw.B, N)))
    y_zero = hw.ideal_code(i0_sig + i_dark, pga_select)[bb]
    if kw.get("p_drop_peak_w") is not None:
        kw["y_peak_code"] = y + gy * (kw["p_drop_peak_w"] - res.drop_sum)
    return StaticPoint(detuning_m=resonance_detuning_m(pp, b, t_ring, cond), t_ring_k=t_ring, t_heater_k=t_h,
                       t_pd_k=np.asarray(t_pd, float), heater_power_w=p_h, heater_code=code, p_abs_w=res.abs_sum,
                       p_drop_w=res.drop_sum, p_in_w=res.p_in[idx, line], oma_w=oma, y_code=y, y_sigma_codes=y_sigma,
                       codes_per_w=gy, z_opt_k_per_w=z_opt, z_heater_k_per_w=z_h, cond=cond, p_junction_w=pj,
                       p_driver_w=pd, y_zero_code=y_zero, heater_current_a=i_h, supply_scale=supply, **kw)


def _heater_t_for(pp, b: int, t_ring, cond: Conditions, n_iter: int = 3) -> np.ndarray:
    """Heater-resistor temperatures (N,) of the static point with the rings at t_ring (N,) K.

    They matter to the optics only through d ln alpha / dT_heater (07 P1-5, the ``heater_t_loss`` ER hypothesis);
    when that coefficient is 0 in world b the ring temperature is returned (no solve). Otherwise a few fixed-point
    passes of the static heater solve (as :func:`static_point`)."""
    from mrm.plant import phase as ph
    t_ring = np.ascontiguousarray(t_ring, dtype=float)
    if not np.any(_phase(pp).ring[b][:, ph.RP_CT_HEATER] != 0.0):
        return t_ring
    sysm = pp.thermal.system
    t_h = t_ring.copy()
    for _ in range(n_iter):
        res, rs, c_eff = bank_optics(pp, b, t_ring, t_h, cond)
        pj, pd = _data_heat(pp, b, cond, rs, c_eff)
        _, t_obs = _solve_heaters(pp, b, cond, t_ring, res.abs_sum, pj, pd)
        t_h = np.ascontiguousarray(t_obs[sysm.heater_obs])
    return t_h


def oma_curve(pp, b: int = 0, detunings_m=None, *, cond: Conditions | None = None, cfg=None,
              op: OperatingPoint | None = None):
    """OMA and open-loop stability of every ring over a grid of detunings (07 P0-1 item 7).

    Returns (detunings (D,), oma_w (D, N), loop_gain (D, N), p_drop_w (D, N), p_abs_w (D, N)): at each detuning the
    rings sit at that delta (the heater supplies whatever it takes); loop_gain = Z_opt(0) dP_abs/dT_R, and a point
    is open-loop stable (heater held) iff loop_gain < 1. The optics of every ring is evaluated with the other rings at
    the same detuning of their own lines, at the heater temperatures of that static point when the loss depends on
    them (:func:`_heater_t_for`).
    """
    cond = conditions_from_config(pp, b, cfg=cfg, op=op) if cond is None else cond
    N = int(pp.n_rings)
    if detunings_m is None:
        fwhm = float(np.max(np.asarray(pp.optics.fwhm_hz, float)[b if np.ndim(pp.optics.fwhm_hz) > 1 else 0]))
        lam = float(np.mean(cond.lam_m))
        hw = lam * lam * fwhm / 299792458.0
        detunings_m = np.linspace(-1.2 * hw, 1.5 * hw, 109)
    d = np.asarray(detunings_m, float)
    sysm = pp.thermal.system
    z_opt = _thermal_dc(pp, b)[sysm.ring_obs, sysm.wg_src]
    line = np.asarray(pp.layout.ring_to_line, np.int64).reshape(-1)
    idx = np.arange(N)
    oma = np.zeros((d.size, N))
    lg = np.zeros((d.size, N))
    pdrop = np.zeros((d.size, N))
    pabs = np.zeros((d.size, N))
    t_prev = None
    for i, x in enumerate(d):
        t = _ring_temperature_for(pp, b, np.full(N, x), cond, t0=t_prev)
        t_prev = t
        th = _heater_t_for(pp, b, t, cond)
        res = bank_optics(pp, b, t, th, cond)[0]
        ru = bank_optics(pp, b, t + _DT_FD_K, th, cond)[0]
        rd = bank_optics(pp, b, t - _DT_FD_K, th, cond)[0]
        oma[i] = np.abs(res.pout1[idx, line] - res.pout0[idx, line])
        lg[i] = z_opt * (ru.abs_sum - rd.abs_sum) / (2 * _DT_FD_K)
        pdrop[i] = res.drop_sum
        pabs[i] = res.abs_sum
    return d, oma, lg, pdrop, pabs


def oma_optimal_detuning(pp, b: int = 0, *, cond: Conditions | None = None, cfg=None, op: OperatingPoint | None = None,
                         stable_margin: float = 0.0, refine: bool = True):
    """OMA-optimal open-loop-stable detuning of every ring (07 P0-1 item 7, P1-1 oracle target).

    Returns (delta_opt (N,) m, oma_opt (N,) W, oma_unrestricted (N,) W, delta_unrestricted (N,) m). Stable means
    Z_opt(0) dP_abs/dT_R < 1 - stable_margin. The grid optimum is refined by golden-section search on the stable
    interval around it.
    """
    cond = conditions_from_config(pp, b, cfg=cfg, op=op) if cond is None else cond
    d, oma, lg, _, _ = oma_curve(pp, b, cond=cond)
    N = oma.shape[1]
    stable = lg < 1.0 - stable_margin
    om = np.where(stable, oma, -np.inf)
    i_s = np.argmax(om, axis=0)
    i_u = np.argmax(oma, axis=0)
    best = d[i_s].copy()
    best_oma = oma[i_s, np.arange(N)].copy()
    if refine:
        step = d[1] - d[0]
        sysm = pp.thermal.system
        z_opt = _thermal_dc(pp, b)[sysm.ring_obs, sysm.wg_src]
        line = np.asarray(pp.layout.ring_to_line, np.int64).reshape(-1)
        idx = np.arange(N)

        def f(x):
            t = _ring_temperature_for(pp, b, x, cond)
            th = _heater_t_for(pp, b, t, cond)
            res = bank_optics(pp, b, t, th, cond)[0]
            ru = bank_optics(pp, b, t + _DT_FD_K, th, cond)[0]
            rd = bank_optics(pp, b, t - _DT_FD_K, th, cond)[0]
            g = z_opt * (ru.abs_sum - rd.abs_sum) / (2 * _DT_FD_K)
            v = np.abs(res.pout1[idx, line] - res.pout0[idx, line])
            return np.where(g < 1.0 - stable_margin, v, -np.inf)
        lo = best - step
        hi = best + step
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = hi - gr * (hi - lo)
        e = lo + gr * (hi - lo)
        fc, fe = f(c), f(e)
        for _ in range(30):
            left = fc >= fe
            hi = np.where(left, e, hi)
            lo = np.where(left, lo, c)
            c_new = hi - gr * (hi - lo)
            e_new = lo + gr * (hi - lo)
            c, e = c_new, e_new
            fc, fe = f(c), f(e)
        x = 0.5 * (lo + hi)
        fx = f(x)
        better = fx > best_oma
        best = np.where(better, x, best)
        best_oma = np.where(better, fx, best_oma)
    ou, du = oma[i_u, np.arange(N)], d[i_u]
    use = best_oma >= ou                        # the refined stable optimum may beat the unrefined unrestricted grid
    return best, best_oma, np.where(use, best_oma, ou), np.where(use, best, du)


# ---------------------------------------------------------------------------------------------- delays
def measurement_age_ticks(period_ticks: int, adc_period_ticks: int, adc_latency_ticks: int, window_ticks: int, *,
                          adc_phase_ticks: int = 0, ctrl_phase_ticks: int = 0, mode: str = "mean",
                          afe_tau_ticks: float = 0.0) -> float:
    """Mean age (ticks) at a controller edge of the monitor information a step consumes (07 P0-6, P0-7).

    A sample with sample tick s (s = adc_phase + j * adc_period) averages the plant intervals [s - W, s), whose
    optics use the end-of-interval state, so it reads the state at s - (W - 1)/2 on average (W = window_ticks; an rc
    AFE has W = 1 plus its filter delay afe_tau_ticks). It is released at s + adc_latency, and an edge k receives every
    sample released in (k - period, k]. mode "mean": the age of the block mean; "latest": the age of the newest
    sample. Averaged over one full phase pattern (lcm of the two periods)."""
    if mode not in ("mean", "latest"):
        raise ValueError(f"measurement mode {mode!r}: mean | latest")
    P, Pa = int(period_ticks), int(adc_period_ticks)
    W = max(int(window_ticks), 1)
    lcm = P * Pa // math.gcd(P, Pa)
    ages = []
    for i in range(1, lcm // P + 1):
        k = ctrl_phase_ticks + i * P + lcm      # far from t = 0
        s_hi = k - adc_latency_ticks           # released at or before k
        s_lo = k - P - adc_latency_ticks       # released after the previous edge
        js = [s for s in range(s_lo + 1, s_hi + 1) if (s - adc_phase_ticks) % Pa == 0]
        if not js:
            continue
        a = [k - (s - 0.5 * (W - 1)) + afe_tau_ticks for s in js]
        ages.append(float(np.mean(a)) if mode == "mean" else float(min(a)))
    if not ages:
        raise ValueError("no monitor sample reaches the controller (ADC period longer than the lcm pattern?)")
    return float(np.mean(ages))


def discretize_zoh(Ac, Bc, Ts: float):
    """Exact zero-order-hold discretization: (Phi, Gamma) = (e^{A Ts}, int_0^Ts e^{A s} ds B)."""
    n, m = Bc.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = Ac * Ts
    M[:n, n:] = Bc * Ts
    E = sla.expm(M)
    return E[:n, :n], E[:n, n:]


def delay_discretize(Ac, Bc, Ts: float, delay_s: float):
    """Exact discretization of dx/dt = A x + B u(t - tau) with u held over each period (modified z-transform).

    tau = q Ts + r, 0 <= r < Ts: x[k+1] = Phi x[k] + Gamma0 u[k - q] + Gamma1 u[k - q - 1] with
    Gamma0 = int_0^{Ts - r} e^{As} ds B and Gamma1 = e^{A (Ts - r)} int_0^{r} e^{As} ds B (Franklin, Powell, Workman
    section 4.3.3). Returns (Phi, Gamma0, Gamma1, q, r)."""
    if delay_s < 0:
        raise ValueError("delay must be >= 0")
    q = int(math.floor(delay_s / Ts + 1e-12))
    r = delay_s - q * Ts
    if r < 1e-15 * Ts:
        r = 0.0
    Phi, _ = discretize_zoh(Ac, Bc, Ts)
    P1, G0 = discretize_zoh(Ac, Bc, Ts - r)
    if r > 0:
        _, Gr = discretize_zoh(Ac, Bc, r)
        G1 = P1 @ Gr
    else:
        G1 = np.zeros_like(G0)
    return Phi, G0, G1, q, r


# ---------------------------------------------------------------------------------------------- linear model
@dataclass(frozen=True)
class LinearModel:
    """Small-signal model of one ring (module docstring). Iterating gives (A, B, C, D, delay_ticks).

    A, B, C, D: exact ZOH discretization at Ts (no delay). Ac, Bc, Cc, Dc: continuous model. delay_ticks: round of
    delay_exact_ticks = tau_u + tau_m (plant ticks); tau_u_ticks / tau_m_ticks the parts. Inputs (u_names):
    heater_code (codes), ring_temperature_disturbance_k (K). Outputs (y_names): monitor_code (codes),
    detuning_m (m).
    op: the StaticPoint of the bank (arrays (N,); this model is ring ``ring``); dt_s: plant tick.
    """
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    delay_ticks: int
    Ac: np.ndarray
    Bc: np.ndarray
    Cc: np.ndarray
    Dc: np.ndarray
    Ts: float
    dt_s: float
    delay_exact_ticks: float
    tau_u_ticks: float
    tau_m_ticks: float
    ring: int
    op: StaticPoint
    x_names: tuple = ()
    u_names: tuple = ("heater_code", "ring_temperature_disturbance_k")
    y_names: tuple = ("monitor_code", "detuning_m")
    meta: dict = field(default_factory=dict)

    def __iter__(self) -> Iterator:
        return iter((self.A, self.B, self.C, self.D, self.delay_ticks))

    # ---- operating-point scalars of this ring
    @property
    def u0(self) -> float:
        """Heater code at the operating point (float, ideal DAC)."""
        return float(self.op.heater_code[self.ring])

    @property
    def y0(self) -> float:
        """Ideal monitor code at the operating point."""
        return float(self.op.y_code[self.ring])

    @property
    def detuning0_m(self) -> float:
        return float(self.op.detuning_m[self.ring])

    @property
    def selfheat_factor(self) -> float:
        """1 + G|P'| at the operating point (07 P0-1): the factor dividing the heater-to-detuning DC gain."""
        return float(self.op.selfheat_factor[self.ring])

    @property
    def delay_s(self) -> float:
        return self.delay_exact_ticks * self.dt_s

    def dc_gain(self) -> np.ndarray:
        """Continuous DC gain (2 outputs x 2 inputs) = D - C A^-1 B."""
        return self.Dc - self.Cc @ np.linalg.solve(self.Ac, self.Bc)

    def frequency_response(self, f_hz, with_delay: bool = True) -> np.ndarray:
        """Continuous transfer matrix (F, 2, 2) at f_hz; the heater column carries e^{-s tau} when with_delay."""
        f = np.atleast_1d(np.asarray(f_hz, float))
        n = self.Ac.shape[0]
        out = np.zeros((f.size, self.Cc.shape[0], self.Bc.shape[1]), complex)
        for i, fi in enumerate(f):
            s = 2j * math.pi * fi
            out[i] = self.Dc + self.Cc @ np.linalg.solve(s * np.eye(n) - self.Ac, self.Bc)
            if with_delay:
                out[i, :, 0] *= np.exp(-s * self.delay_s)
        return out

    def step_response(self, t_s, input_index: int = 0, with_delay: bool = True) -> np.ndarray:
        """Continuous unit-step responses (T, 2) of both outputs to input ``input_index`` (heater: per code)."""
        t = np.atleast_1d(np.asarray(t_s, float))
        tau = self.delay_s if (with_delay and input_index == 0) else 0.0
        out = np.zeros((t.size, self.Cc.shape[0]))
        b = self.Bc[:, input_index:input_index + 1]
        for i, ti in enumerate(t):
            te = ti - tau
            if te <= 0:
                continue
            _, g = discretize_zoh(self.Ac, b, te)
            out[i] = (self.Cc @ g)[:, 0] + self.Dc[:, input_index]
        return out

    def n_delay(self, Ts: float | None = None) -> int:
        """Length q + 1 of the heater-code delay line of :meth:`delayed` (q = floor(tau / Ts))."""
        Ts = self.Ts if Ts is None else float(Ts)
        return int(math.floor(self.delay_s / Ts + 1e-12)) + 1

    def delayed(self, Ts: float | None = None, outputs: Sequence[int] = (0, 1), n_delay: int | None = None):
        """Exact discrete model with the loop delay on the heater input (fractional part included).

        State z = [x (n); u[k-1]; ...; u[k-nd]] (the heater codes still in the delay line, nd = max(q + 1, n_delay):
        a longer line than the delay needs only pads unused states, so rings or worlds with different delays share
        one state size); the disturbance input (column 1) is not delayed. Returns (Phi (nz, nz), Gu (nz,), Gd (nz,),
        C (ny, nz), Dd (ny,), nd) with z[k+1] = Phi z[k] + Gu u[k] + Gd d[k], y[k] = C z[k] + Dd d[k]."""
        Ts = self.Ts if Ts is None else float(Ts)
        Ac, Bc = self.Ac, self.Bc
        n = Ac.shape[0]
        Phi, G0, G1, q, r = delay_discretize(Ac, Bc[:, :1], Ts, self.delay_s)
        _, Gd = discretize_zoh(Ac, Bc[:, 1:2], Ts)
        nd = max(q + 1, 0 if n_delay is None else int(n_delay))      # u[k-1] .. u[k-q-1] (+ padding)
        nz = n + nd
        Pz = np.zeros((nz, nz))
        Pz[:n, :n] = Phi
        Gu = np.zeros(nz)
        # u[k - q] enters through G0, u[k - q - 1] through G1
        if q == 0:
            Gu[:n] = G0[:, 0]
            Pz[:n, n] = G1[:, 0]                     # u[k-1] is state n
        else:
            Pz[:n, n + q - 1] = G0[:, 0]             # u[k-q]   is state n + q - 1
            Pz[:n, n + q] = G1[:, 0]                 # u[k-q-1] is state n + q
        # shift register: u[k-1] <- u[k]; u[k-j-1] <- u[k-j]
        Gu[n] = 1.0
        for j in range(1, nd):
            Pz[n + j, n + j - 1] = 1.0
        Gdz = np.zeros(nz)
        Gdz[:n] = Gd[:, 0]
        oi = list(outputs)
        C = np.zeros((len(oi), nz))
        C[:, :n] = self.Cc[oi]
        Dd = self.Dc[oi, 1].copy()
        return Pz, Gu, Gdz, C, Dd, nd


def ring_block_lags(pp, b: int, n: int):
    """Map the grouped states of ring n's own thermal block (``_ring_thermal_block`` order) to the plant's lags.

    Returns (lag_index (k,), residue (k,) K/W, source (k,) 0 heater / 1 waveguide): grouped state i equals
    x[lag_index[i]] / residue[i] for a plant lag state x (B, M) started from rest or a steady state (every lag of
    one (source, tau) then evolves in proportion to its residue). The ring observer's lag is chosen when it exists."""
    sysm = pp.thermal.system
    bb = b if sysm.batch > 1 else 0
    obs = [int(sysm.ring_obs[n]), int(sysm.heater_obs[n])]
    src = [int(sysm.heater_src[n]), int(sysm.wg_src[n])]
    groups: dict[tuple[int, float], list[int]] = {}
    for m in range(sysm.n_lags):
        if int(sysm.lag_obs[m]) not in obs or int(sysm.lag_src[m]) not in src:
            continue
        r = float(sysm.res_k_per_w[bb, m])
        if r == 0.0:
            continue
        s = src.index(int(sysm.lag_src[m]))
        groups.setdefault((s, float(np.round(sysm.taus_s[bb, m], 15))), []).append(m)
    keys = sorted(groups, key=lambda kv: (kv[0], kv[1]))
    idx, res, srcs = [], [], []
    for key in keys:
        ms = groups[key]
        ring_first = [m for m in ms if int(sysm.lag_obs[m]) == obs[0]] or ms
        m = ring_first[0]
        idx.append(m)
        res.append(float(sysm.res_k_per_w[bb, m]))
        srcs.append(key[0])
    return np.array(idx, np.int64), np.array(res), np.array(srcs, np.int64)


def _ring_thermal_block(pp, b, n):
    """The ring's own thermal block (observers ring R, heater resistor H; sources heater h, waveguide w) as a minimal
    realization: the lags of one (source, tau) share one unit state z (dz/dt = (u_src - z)/tau, exact: they have the
    same dynamics and input), and each observer reads sum r z. Returns (A, B (k, 2), C (2, k), poles)."""
    from mrm.plant.thermal import state_space
    sysm = pp.thermal.system
    obs = [int(sysm.ring_obs[n]), int(sysm.heater_obs[n])]
    src = [int(sysm.heater_src[n]), int(sysm.wg_src[n])]
    A, Bm, C, _ = state_space(sysm, b if sysm.batch > 1 else 0, obs=obs, src=src)
    tau = -1.0 / np.diag(A)
    groups: dict[tuple[int, float], list[int]] = {}
    for m in range(A.shape[0]):
        s = int(np.argmax(np.abs(Bm[m])))
        if Bm[m, s] == 0.0:
            continue
        key = (s, float(np.round(tau[m], 15)))
        groups.setdefault(key, []).append(m)
    keys = sorted(groups, key=lambda kv: (kv[0], kv[1]))
    k = len(keys)
    Az = np.zeros((k, k))
    Bz = np.zeros((k, 2))
    Cz = np.zeros((2, k))
    for i, key in enumerate(keys):
        s, t = key
        Az[i, i] = -1.0 / t
        Bz[i, s] = 1.0 / t
        for m in groups[key]:
            r = Bm[m, s] * t                    # residue r_m (K/W): Bm = r / tau
            Cz[:, i] += C[:, m] * r
    return Az, Bz, Cz, np.diag(Az)


def _model_for_ring(sp: StaticPoint, pp, b: int, n: int, Ts: float, tau_u: float, tau_m: float, dt: float,
                    settle_tau_s: float = 0.0) -> LinearModel:
    A_th, B_th, C_th, poles = _ring_thermal_block(pp, b, n)
    cR, cH = C_th[0], C_th[1]
    bh, bw = B_th[:, 0], B_th[:, 1]
    k_u, k_T = sp.dp_dcode[n], sp.dp_dth_heater[n]
    aR, aH, ad = sp.dpabs_dtr[n], sp.dpabs_dth[n], sp.dpabs_dd[n]
    mR, mH, md = sp.dpdrop_dtr[n], sp.dpdrop_dth[n], sp.dpdrop_dd[n]
    gy, s = sp.codes_per_w[n], sp.s_m_per_k[n]
    A = A_th + np.outer(bh, k_T * cH) + np.outer(bw, aR * cR + aH * cH)
    Bc = np.stack([bh * k_u, bw * aR], axis=1)
    Cc = np.stack([gy * (mR * cR + mH * cH), s * cR], axis=0)
    Dc = np.array([[0.0, gy * mR], [0.0, s]])
    names = tuple(f"{'hw'[int(np.argmax(np.abs(B_th[i])))]}_tau{(-1.0 / p) * 1e6:.3g}us" for i, p in enumerate(poles))
    if settle_tau_s > 0:                 # first-order settle filter of the heater DAC on the code
        m = A.shape[0]
        A2 = np.zeros((m + 1, m + 1))
        A2[:m, :m] = A
        A2[:m, m] = Bc[:, 0]
        A2[m, m] = -1.0 / settle_tau_s
        B2 = np.zeros((m + 1, 2))
        B2[m, 0] = 1.0 / settle_tau_s
        B2[:m, 1] = Bc[:, 1]
        Cc = np.concatenate([Cc, np.zeros((2, 1))], axis=1)
        A, Bc = A2, B2
        names = names + ("settle",)
    Ad, Bd = discretize_zoh(A, Bc, Ts)
    tau = tau_u + tau_m
    return LinearModel(A=Ad, B=Bd, C=Cc.copy(), D=Dc.copy(), delay_ticks=int(round(tau)), Ac=A, Bc=Bc, Cc=Cc, Dc=Dc,
                       Ts=float(Ts), dt_s=float(dt), delay_exact_ticks=float(tau), tau_u_ticks=float(tau_u),
                       tau_m_ticks=float(tau_m), ring=int(n), op=sp, x_names=names)


def _adc_phases(pp) -> np.ndarray:
    """(N,) sample phase of every ring's monitor ADC in ticks (``ring.monitor.adc_mux``: round_robin staggers the
    rings that share a converter; the plant's own rule, ``mrm.plant.sensing.MonitorHw.phase``)."""
    from mrm.plant.sensing import MonitorHw
    return np.asarray(MonitorHw.from_plant_params(pp).phase, np.int64).reshape(-1)


def _timing(pp, cfg, Ts, latency_ticks, measurement, transport_delay: bool = True, b: int = 0,
            adc_phase_ticks: int = 0):
    """(Ts, tau_u, tau_m, dt) of world b: controller period, input delay (controller latency + heater DAC latency +
    the thermal transport delay in whole ticks, as the heater FIFO applies it) and the mean measurement age of a ring
    whose ADC samples at phase ``adc_phase_ticks`` (controller edges at phase 0, as ControllerSpec.from_config)."""
    from mrm.plant.sensing import AFE_INTEGRATE_DUMP
    t = pp.timing
    dt = float(t.dt_s)

    def wb(x):
        a = np.asarray(x, float).reshape(-1)
        return float(a[b if a.size > 1 else 0])
    if Ts is None:
        rate = _cfg_si(cfg, "controller_interface.update_rate_hz", None)
        Ts = (1.0 / rate) if rate else int(t.controller_period_ticks) * dt
    period = int(round(Ts / dt))
    if abs(period * dt - Ts) > 1e-9 * Ts:
        raise ValueError(f"Ts = {Ts} s is not a whole number of plant ticks ({dt} s)")
    dac = wb(t.heater_latency_ticks)
    if transport_delay:
        td = getattr(pp.thermal.system, "transport_delay_s", None)
        if td is not None:
            dac += float(np.rint(wb(td) / dt))
    tau_u = float(latency_ticks) + dac
    m = pp.monitor
    afe_id = int(np.asarray(m.afe_type).reshape(-1)[0])
    win = int(wb(t.integration_window_ticks)) if afe_id == AFE_INTEGRATE_DUMP else 1
    afe_tau = 0.0 if afe_id == AFE_INTEGRATE_DUMP else wb(m.afe_tau_s) / dt
    lat = int(wb(t.monitor_latency_ticks))
    if measurement == "none":            # full state at the edge (the Truth-fed oracle): actuation delay only
        tau_m = 0.0
    else:
        tau_m = measurement_age_ticks(period, int(t.adc_period_ticks), lat, win, mode=measurement,
                                      afe_tau_ticks=afe_tau, adc_phase_ticks=int(adc_phase_ticks))
    return float(period * dt), tau_u, tau_m, dt


_BANK_CACHE: dict[tuple, list] = {}


def linearize_bank(nominal_cfg, operating_point: OperatingPoint | None = None, Ts: float | None = None, *,
                   latency_ticks: int = 0, measurement: str = "mean", knowledge: Iterable[str] | None = None,
                   pp=None, b: int = 0, cond: Conditions | None = None) -> list[LinearModel]:
    """:func:`linearize` for every ring of the bank (one static solve). Returns N LinearModels.

    ``pp`` (default: :func:`nominal_plant_params` of ``nominal_cfg``), ``b`` and ``cond`` (default:
    :func:`conditions_from_config`) exist for verification against a known world and for the Truth-fed oracle; a
    controller never passes them. Results for the nominal design are cached (config fingerprint,
    operating point, Ts, latency, measurement)."""
    if knowledge is not None:
        check_knowledge(knowledge, who="linearize")
    op = OperatingPoint() if operating_point is None else operating_point
    key = None
    if pp is None and cond is None:
        fp = _cfg_key(nominal_cfg)
        if fp is not None:
            det = op.detuning_m if op.detuning_m is None or np.ndim(op.detuning_m) == 0 else tuple(
                np.asarray(op.detuning_m, float).ravel())
            key = (fp, replace(op, ring=0, detuning_m=None), det, Ts, int(latency_ticks), measurement)
            if key in _BANK_CACHE:
                return list(_BANK_CACHE[key])
    if pp is None:
        pp = nominal_plant_params(nominal_cfg)
    models = _linearize_bank(nominal_cfg, op, Ts, latency_ticks, measurement, pp, b, cond)
    if key is not None:
        if len(_BANK_CACHE) > 32:
            _BANK_CACHE.clear()
        _BANK_CACHE[key] = list(models)
    return models


def _linearize_bank(nominal_cfg, op, Ts, latency_ticks, measurement, pp, b, cond=None):
    cond = conditions_from_config(pp, b, cfg=nominal_cfg, op=op) if cond is None else cond
    N = int(pp.n_rings)
    if op.detuning_m is not None:
        target = np.broadcast_to(np.asarray(op.detuning_m, float), (N,))
    elif op.detuning_frac_of_peak is not None:
        target = _detuning_for_fraction(pp, b, cond, float(op.detuning_frac_of_peak))
    else:
        target = oma_optimal_detuning(pp, b, cond=cond)[0]
    sp = static_point(pp, b, target, cond=cond, pga_select=op.pga_select)
    st_ = np.asarray(pp.heater.settle_tau_s, float).reshape(-1)
    settle = float(st_[b if st_.size > 1 else 0])
    phases = _adc_phases(pp) if measurement != "none" else np.zeros(N, np.int64)
    out = []
    for n in range(N):        # tau_m per ring: an ADC mux samples the rings that share a converter at staggered phases
        Ts_, tau_u, tau_m, dt = _timing(pp, nominal_cfg, Ts, latency_ticks, measurement, b=b,
                                        adc_phase_ticks=int(phases[n]))
        out.append(_model_for_ring(sp, pp, b, n, Ts_, tau_u, tau_m, dt, settle))
    return out


def _detuning_for_fraction(pp, b, cond, frac):
    """Stable-side detuning (> 0) where the drop power is frac of its peak (bisection per ring)."""
    if not 0.0 < frac < 1.0:
        raise ValueError("detuning_frac_of_peak must be in (0, 1)")
    N = int(pp.n_rings)

    def drop(x):
        t = _ring_temperature_for(pp, b, x, cond)
        return bank_optics(pp, b, t, _heater_t_for(pp, b, t, cond), cond)[0].drop_sum
    pk = drop(np.zeros(N))
    lo = np.zeros(N)
    hi = np.full(N, 2e-9)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        above = drop(mid) > frac * pk
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return 0.5 * (lo + hi)


def linearize(nominal_cfg, operating_point: OperatingPoint | None = None, Ts: float | None = None, *,
              latency_ticks: int = 0, measurement: str = "mean", knowledge: Iterable[str] | None = None
              ) -> LinearModel:
    """Small-signal model of ring ``operating_point.ring`` around the nominal operating point (07 P1-1).

    Args:
        nominal_cfg: the nominal design (``ControllerSpec.nominal``; an ``mrm.params`` Config). It is realized with
            uncertainty nominal and variation off (:func:`nominal_plant_params`).
        operating_point: :class:`OperatingPoint` (default: ring 0 at the OMA-optimal stable detuning, scenario
            ambient and bias).
        Ts: controller period in s (default 1 / controller_interface.update_rate_hz); a whole number of plant ticks.
        latency_ticks: the controller's own pipeline depth (``Controller.latency_ticks``), added to tau_u.
        measurement: "mean" (the controller averages the samples of a step), "latest" (newest sample only) or
            "none" (no measurement delay: a full-state, Truth-fed user such as the oracle).
        knowledge: if given, :func:`check_knowledge` runs on it first.
    Returns:
        :class:`LinearModel`; ``A, B, C, D, delay_ticks = linearize(...)``.
    """
    op = OperatingPoint() if operating_point is None else operating_point
    models = linearize_bank(nominal_cfg, op, Ts, latency_ticks=latency_ticks, measurement=measurement,
                            knowledge=knowledge)
    if not 0 <= op.ring < len(models):
        raise ValueError(f"operating_point.ring = {op.ring} outside 0..{len(models) - 1}")
    return models[op.ring]
