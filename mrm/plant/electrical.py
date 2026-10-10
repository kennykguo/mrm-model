"""Electrical actuator plant: heater DAC and driver law with TCR, bias DAC, junction runtime, data-activity heat.

Scope (docs/ARCHITECTURE.md section 2; 07 = docs/research/07_design_critique.md): 07 P1-11, P1-7 (runtime side),
P1-8, P1-10 (DAC INL/DNL, compliance, readback values), P1-12 (array supply limit), P2-5 (electrothermal gain),
P0-6 (actuator path order).

Units are SI everywhere (V, A, W, ohm, K, s, F, m, Hz). Temperatures enter only as differences (T - T0), so
kelvin or Celsius work as long as T and T0 use the same scale.

Event order of one actuator channel, per plant tick k (07 P0-6, items 4 and 5)::

    controller command --(Controller.latency_ticks: owned by the controller/runner, not here)-->
    DAC register latch (clamp to [code_min, code_max]; max update rate) == applied_codes (register readback)
      --> DAC latency FIFO (ring.heater.latency_s, whole ticks)
      --> DAC transfer: code -> normalized drive u (binary-weighted bit errors + INL bow; dac_pwm: period average)
      --> slew limit on u (slew_codes_per_s / (2^bits - 1) per second)
      --> settle filter on u (first order, exact solution for the piecewise-linear slewed input)
      --> [+ PWM ripple term, dac_pwm only]
      --> driver law with TCR: R = R0*(1 + alpha*(T_heater - T0)), compliance
      --> array supply limit (common droop factor per world; clip and flag)
      --> heater power (ROM source 'heater'), and V, I, R for heater readback (sampled by mrm/plant/sensing.py)

Tick convention (same as mrm/core/timing.py): tick k is the instant k*dt; ``*_step(tick=k, cmd)`` is called after
the controller edge at instant k and computes the actuator output for plant interval [k, k+1). ``cmd`` is the command
whose latch tick (edge + Controller.latency_ticks) is k. The register latched at k feeds the FIFO at k, so the analog
value first acts over [k_act, k_act + 1) with k_act = k_latch + dac_latency_ticks, as in timing.py step (5). The power
handed to the zero-order-hold thermal ROM is the driver law evaluated at the interval average of the settled drive, so
a register impulse latched at k reaches the heater power over interval k + latency_ticks exactly, never one tick late.
The FIFO is indexed by the absolute tick: call every tick in order (see actuator_quiescent_row for P0-10 skipping).

Normalized drive u, per driver law (07 P1-11, uncertainty.heater.driver_mode):
    voltage:  V = u*V_FS,             P = V^2 / R(T)
    current:  I = u*I_FS,             P = I^2 * R(T), V = I*R limited by the driver compliance V_comp
    power:    P = u*P_FS (regulated), V = sqrt(P*R) limited by V_comp
    dac_pwm:  u = <(V/V_FS)^2> over the 2.56 us PWM period; P = u*V_FS^2 / R(T)
              (US11409045 = PAT-2 in docs/research/01 section 4: 8-bit DAC + 8-bit PWM, 1/256 steps over 2.56 us,
              N/N+1 dithering; the inner DAC is assumed to be a voltage DAC, which the patent does not state.)

Kernels: every per-tick function is numba @njit, takes arrays and scalars only, and runs under NUMBA_DISABLE_JIT=1.
Each comes in two forms: ``*_row`` works on one uncertainty world (arrays (N,), parameters (N,) or (1,)), for use
inside a ``prange`` over B; the batched form loops over b for (B, N) arrays, with parameters (B, N), (B, 1), (1, N) or
(1, 1) broadcast. The Python classes ``HeaterBank`` and ``BiasBank`` hold the state and call the batched kernels.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numba import njit

from mrm.core.timing import rate_to_period_ticks, seconds_to_ticks

__all__ = [
    "LAW_VOLTAGE", "LAW_CURRENT", "LAW_POWER", "LAW_DAC_PWM", "DRIVER_LAWS",
    "LEVELS_CENTERED", "LEVELS_ONE_SIDED", "LEVEL_MODES",
    "FLAG_COMPLIANCE", "FLAG_SUPPLY", "FLAG_RUNAWAY", "FLAG_FORWARD_CLIP", "FLAG_GRID_CLIP",
    "PAT2_PWM_PERIOD_S", "PAT2_PWM_BITS", "PAT2_DAC_BITS", "G26_BIAS_V_MIN", "G26_BIAS_V_MAX",
    "JUNCTION_GRID_V_MIN", "JUNCTION_GRID_V_MAX",
    # row kernels
    "latch_row", "fifo_row", "dac_frac", "heater_target_row", "heater_target_one", "bias_target_row", "bias_target_one",
    "slew_settle_row",
    "actuator_quiescent_row", "settle_advance_row",
    "heater_power_row", "supply_limit_row", "heater_bank_step_row", "bias_bank_step_row", "nrz_levels_row",
    "junction_eval_row", "junction_swing_capacitance_row", "data_activity_heat_row",
    # batched kernels
    "heater_bank_step", "bias_bank_step", "nrz_levels", "junction_eval", "junction_swing_capacitance",
    "data_activity_heat",
    # Python API
    "HeaterDriverParams", "HeaterBank", "HeaterTickOut", "BiasDacParams", "BiasBank", "BiasTickOut",
    "JunctionTable", "load_junction_table", "analytic_junction_table", "analytic_junction_table_from_ring",
    "depletion_shift_m",
    "junction_efficiency_m_per_v", "junction_dqi_dv_frac_per_v", "heater_steady_state", "heater_gain_w_per_code", "electrothermal_loop_gain",
    "closed_loop_gain_factor", "compliance_headroom_v", "pwm_ripple_pp_k", "dac_bit_errors_from_spec",
    "dac_transfer_curve", "data_activity_heat_w",
]

# --------------------------------------------------------------------------------------------- enums and flags
LAW_VOLTAGE = 0
LAW_CURRENT = 1
LAW_POWER = 2
LAW_DAC_PWM = 3
DRIVER_LAWS = {"voltage": LAW_VOLTAGE, "current": LAW_CURRENT, "power": LAW_POWER, "dac_pwm": LAW_DAC_PWM}

LEVELS_CENTERED = 0   # V_lo,hi = V_bias -+ Vpp/2 (07 P1-7: "bias - Vpp/2 can reach" the forward region)
LEVELS_ONE_SIDED = 1  # V_lo = V_bias - Vpp, V_hi = V_bias (DC on one terminal, rail-to-rail driver on the other)
LEVEL_MODES = {"centered": LEVELS_CENTERED, "one_sided": LEVELS_ONE_SIDED}

FLAG_COMPLIANCE = 1    # driver output hit its compliance voltage
FLAG_SUPPLY = 2        # array heater supply limit scaled this world's heaters down
FLAG_RUNAWAY = 4       # electrothermal runaway guard (current drive with I^2*R0*alpha*g >= 1, or alpha < 0 voltage)
FLAG_FORWARD_CLIP = 8  # an NRZ level was clamped at the forward-bias floor of the junction grid
FLAG_GRID_CLIP = 16    # a junction lookup fell outside the table and was held at the end value

# ------------------------------------------------------------------- hardware constants (provenance stated)
PAT2_PWM_PERIOD_S = 2.56e-6  # printed, PAT-2 (US11409045): "1/256 increments over a 2.560 us period"
PAT2_PWM_BITS = 8            # printed, PAT-2: 8-bit PWM
PAT2_DAC_BITS = 8            # printed, PAT-2: low-precision 8-bit DAC (8 + 8 = "about 16 bits")
G26_BIAS_V_MIN = 1.0         # printed (axis), G26: LM26-S slide 9 / HC slide 16 bias axis 1.0 to 3.75 V
G26_BIAS_V_MAX = 3.75        # printed (axis), same source
JUNCTION_GRID_V_MIN = -0.5   # contract, ARCHITECTURE 5.4 and 07 P1-7: grid from -0.5 V (forward) ...
JUNCTION_GRID_V_MAX = 4.0    # ... to 4.0 V reverse

_NEVER = -(1 << 62)  # "last update" tick before any update: makes the first register write always pass
_U = np.uint64       # unsigned index cast of the per-tick lookups (no negative-index wraparound; perf.md R2-1)


# ============================================================================================ small helpers
@njit(cache=True)
def _ix(a, i):
    """Broadcast index along the first axis: i if the axis has length > 1, else 0."""
    return i if a.shape[0] > 1 else 0


@njit(cache=True, _nrt=False)
def dac_frac(code, n_bits, err_row, bow_lsb):
    """Endpoint-referenced transfer of a binary-weighted DAC: integer code -> fraction of full scale in [0, 1].

    y(c) = sum_i b_i(c) * (2^i + e_i) + bow * 4x(1 - x),   x = c/(2^n - 1),   frac = y(c) / y(2^n - 1)

    e_i (LSB) are bit-weight errors (unit-element mismatch gives DNL at the major carries, 07 P1-10 dac_dnl_lsb);
    bow (LSB) is a smooth, endpoint-free INL term (dac_inl_lsb). The gain error is removed by the endpoint
    reference (it is indistinguishable from a heater-resistance or full-scale error). With e = 0, bow = 0 the
    transfer is exactly c/(2^n - 1). err_row must have at least n_bits entries.
    """
    y = 0.0
    ytot = 0.0
    for i in range(n_bits):
        w = float(1 << i) + err_row[i]
        ytot += w
        if (code >> i) & 1:
            y += w
    cmax = float((1 << n_bits) - 1)
    x = code / cmax
    y += bow_lsb * 4.0 * x * (1.0 - x)
    return y / ytot


@njit(cache=True)
def _pwm_high_clocks(c, d, period_clocks):
    """Number of PWM clocks in [0, c) during which a left-aligned PWM of duty d/period is high."""
    q = c // period_clocks
    r = c - q * period_clocks
    return q * d + (r if r < d else d)


# ============================================================================================ register and FIFO
@njit(cache=True, _nrt=False)
def latch_row(cmd, cmd_valid, reg, pending, has_pending, last_update, tick, min_update_ticks, code_lo, code_hi):
    """DAC register latch for one world (07 P0-6 item 4; P1-10 dac_max_update_hz).

    A valid command is clamped to [code_lo, code_hi] (ring.heater.code_min/code_max, hardware range; 07 P1-11) and
    queued; it is written to the register when at least min_update_ticks ticks have passed since the last write (a
    DAC with a maximum update rate; a newer command replaces a queued one). The register is the ``applied_codes``
    readback. Arrays (N,): cmd int64, cmd_valid bool/uint8, reg int64, pending int64, has_pending int8,
    last_update int64 (all updated in place).
    """
    for n in range(reg.shape[0]):
        if cmd_valid[n]:
            c = cmd[n]
            if c < code_lo:
                c = code_lo
            elif c > code_hi:
                c = code_hi
            pending[n] = c
            has_pending[n] = 1
        if has_pending[n] != 0 and tick - last_update[n] >= min_update_ticks:
            reg[n] = pending[n]
            last_update[n] = tick
            has_pending[n] = 0


@njit(cache=True, _nrt=False)
def fifo_row(reg, fifo, tick, latency_ticks, out_code):
    """DAC latency FIFO for one world: out_code(k) = reg(k - latency_ticks), exactly (07 P0-6 item 5).

    fifo (N, D) int64 is a ring buffer indexed by the absolute tick (slot = tick mod D), so it must be called once
    per tick, in order, with D >= latency_ticks + 1. latency_ticks = 0 passes the register straight through.
    """
    d = fifo.shape[1]
    w = tick % d
    r = (tick - latency_ticks) % d
    for n in range(reg.shape[0]):
        fifo[n, w] = reg[n]
        out_code[n] = fifo[n, r]


# ============================================================================================ DAC transfer
@njit(cache=True)
def heater_target_row(code, law, bits, pwm_bits, bit_err, bow, pwm_ripple, tick, dt_clk, out_u, out_ripple):
    """Heater DAC transfer for one world: code -> normalized drive u (period average) and the PWM ripple term.

    voltage/current/power: u = dac_frac(code, bits) (V/V_FS, I/I_FS or P/P_FS).
    dac_pwm (PAT-2, docs/research/01 section 4): code = 2^p*N + d with p = pwm_bits; an inner (bits - p)-bit voltage
    DAC outputs level N+1 for d of the 2^p PWM clocks and level N for the rest, so the period average is
        u = (1 - f)*x_N^2 + f*x_{N+1}^2,   f = d/2^p,   x_N = dac_frac(N, bits - p)
    (x_{N+1} is held at full scale for N = 2^(bits-p) - 1). With pwm_ripple = 1 the ripple term is the exact tick
    average of the PWM waveform minus the period average:
        ripple = (x_{N+1}^2 - x_N^2) * (h - f),  h = fraction of the tick [k*dt, (k+1)*dt) with the PWM high,
    counted in whole PWM clocks (dt must be dt_clk PWM clocks, PWM clock = 2.56 us / 256 = 10 ns). Its mean over
    a PWM period is zero. bit_err (Nx, K) LSB of the DAC that the law uses; bow (Nx,) LSB.
    Approximations (dac_pwm): the ripple is added after the settle filter, so a settle_tau_s comparable to the PWM
    clock does not attenuate it (overestimate); the slew limit acts on the period-average power fraction u, so a
    slew in codes/s maps to u only approximately (du/dcode ~ 2*code/code_max^2 varies across the range).
    """
    for n in range(code.shape[0]):
        u, rip = heater_target_one(code[n], law, bits, pwm_bits, bit_err[_ix(bit_err, n)], bow[_ix(bow, n)],
                                   pwm_ripple, tick, dt_clk)
        out_u[n] = u
        out_ripple[n] = rip


@njit(cache=True, _nrt=False)
def heater_target_one(c, law, bits, pwm_bits, er, bw, pwm_ripple, tick, dt_clk):
    """One channel of :func:`heater_target_row`: (u, ripple) for code c, bit errors er (K,) and bow bw (LSB). Without
    the PWM ripple the result depends on the code alone (a caller may keep it while the code holds)."""
    if law == LAW_DAC_PWM:
        nbc = bits - pwm_bits
        cmax_c = (1 << nbc) - 1
        pclk = 1 << pwm_bits
        nc = c >> pwm_bits
        d = c - (nc << pwm_bits)
        f = d / pclk
        x0 = dac_frac(nc, nbc, er, bw)
        n1 = nc + 1 if nc < cmax_c else cmax_c
        x1 = dac_frac(n1, nbc, er, bw)
        u = (1.0 - f) * x0 * x0 + f * x1 * x1
        if pwm_ripple != 0:
            c0 = tick * dt_clk
            h = (_pwm_high_clocks(c0 + dt_clk, d, pclk) - _pwm_high_clocks(c0, d, pclk)) / dt_clk
            return u, (x1 * x1 - x0 * x0) * (h - f)
        return u, 0.0
    return dac_frac(c, bits, er, bw), 0.0


@njit(cache=True)
def bias_target_row(code, bits, bit_err, bow, v_min, v_max, out_v):
    """Bias DAC transfer for one world: V = v_min + (v_max - v_min)*dac_frac(code, bits)   [V].

    v_min/v_max (Nx,): the printed G26 bias axis is 1.0 to 3.75 V (07 P0-7, ring.bias_dac; LM26-S slide 9).
    """
    for n in range(code.shape[0]):
        out_v[n] = bias_target_one(code[n], bits, bit_err[_ix(bit_err, n)], bow[_ix(bow, n)], v_min[_ix(v_min, n)],
                                   v_max[_ix(v_max, n)])


@njit(cache=True, _nrt=False)
def bias_target_one(c, bits, er, bw, lo, hi):
    """One channel of :func:`bias_target_row` (a function of the code alone)."""
    return lo + (hi - lo) * dac_frac(c, bits, er, bw)


# ============================================================================================ slew and settle
@njit(cache=True)
def _x_plus_expm1_neg(x):
    """x - (1 - e^(-x)) = x + expm1(-x) for x >= 0, accurate for small x (Taylor series below 1e-3)."""
    if x < 1e-3:
        return x * x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0 - x / 120.0)))
    return x + math.expm1(-x)


@njit(cache=True, _nrt=False)
def slew_settle_row(target, y_slew, y_out, rate, tau, dt, out_avg):
    """Slew limit, then first-order settle filter, solved exactly over one tick (07 P0-6 item 5).

    Slew: the slewed signal s(t) moves from s0 toward the target at |ds/dt| = rate until it arrives at
    t1 = |target - s0|/rate (or stays a ramp for the whole tick). Settle: tau*dy/dt = s(t) - y. For the
    piecewise-linear s(t) the solution is exact (no step-size error):
        on [0, t1]:  y(t) = s0 + r t - r tau + (y0 - s0 + r tau) e^(-t/tau)
        on [t1, dt]: y(t) = s1 + (y(t1) - s1) e^(-(t - t1)/tau)
    tau <= 0 means no settle filter (y = s). rate = inf means no slew limit. Updates y_slew and y_out to their
    end-of-tick values and writes the exact tick average of y to out_avg (the value a zero-order-hold load sees).
    Units: whatever target is in (normalized drive, or volts for the bias DAC); rate in those units per second.
    """
    for n in range(target.shape[0]):
        rt = rate[_ix(rate, n)]
        ta = tau[_ix(tau, n)]
        s0 = y_slew[n]
        y0 = y_out[n]
        tg = target[n]
        dlt = tg - s0
        if dlt == 0.0 or math.isinf(rt):
            t1 = 0.0
            r = 0.0
            s1 = tg
        elif rt * dt >= abs(dlt):
            t1 = abs(dlt) / rt
            r = math.copysign(rt, dlt)
            s1 = tg
        else:
            t1 = dt
            r = math.copysign(rt, dlt)
            s1 = s0 + r * dt
        if ta <= 0.0:
            integ = s0 * t1 + 0.5 * r * t1 * t1 + s1 * (dt - t1)
            y_end = s1
        else:
            if t1 > 0.0:
                om1 = -math.expm1(-t1 / ta)  # 1 - e^(-t1/tau), accurate for small t1/tau
                e1 = ta * _x_plus_expm1_neg(t1 / ta)  # t1 - tau*(1 - e^(-t1/tau)) >= 0, no cancellation
                # same solution as the docstring, rearranged so that the large terms +-r*tau do not cancel:
                y1 = y0 + (s0 - y0) * om1 + r * e1
                int1 = s0 * t1 + 0.5 * r * t1 * t1 + (y0 - s0) * ta * om1 - r * ta * e1
            else:
                y1 = y0
                int1 = 0.0
            t2 = dt - t1
            if t2 > 0.0:
                om2 = -math.expm1(-t2 / ta)
                y_end = s1 + (y1 - s1) * (1.0 - om2)
                int2 = s1 * t2 + (y1 - s1) * ta * om2
            else:
                y_end = y1
                int2 = 0.0
            integ = int1 + int2
        y_slew[n] = s1
        y_out[n] = y_end
        out_avg[n] = integ / dt


@njit(cache=True, _nrt=False)
def actuator_quiescent_row(reg, has_pending, fifo, y_slew, y_out, target, tol):
    """True when every ring's actuator path is at rest, so that P0-10 multi-tick stepping may skip ticks.

    Conditions per ring: no command is waiting in the latch (has_pending == 0; a DAC with a maximum update rate
    writes it on a later tick), every FIFO slot holds the register value (the FIFO output cannot change), the
    slewed value equals the DAC target of that code, and |y_out - target| <= tol. target is the last u_target (or
    v_target) of the same tick. While quiescent the FIFO may be skipped; advance the settle filter with
    settle_advance_row. Not covered here (the caller must check them): a dac_pwm ripple term (varies every tick
    unless dt is a whole number of PWM periods) and bias-DAC noise (redrawn every noise period).
    """
    for n in range(reg.shape[0]):
        if has_pending[n] != 0:
            return False
        for j in range(fifo.shape[1]):
            if fifo[n, j] != reg[n]:
                return False
        if y_slew[n] != target[n] or abs(y_out[n] - target[n]) > tol:
            return False
    return True


@njit(cache=True, _nrt=False)
def settle_advance_row(target, y_slew, y_out, tau, dt, n_ticks):
    """Exact advance of the settle filter over n_ticks ticks with a constant, already-reached slew target:
    y <- target + (y - target) * exp(-n_ticks*dt/tau)  (tau <= 0: y <- target)."""
    for n in range(target.shape[0]):
        ta = tau[_ix(tau, n)]
        y_slew[n] = target[n]
        if ta <= 0.0:
            y_out[n] = target[n]
        else:
            y_out[n] = target[n] + (y_out[n] - target[n]) * math.exp(-n_ticks * dt / ta)


# ============================================================================================ driver law + TCR
@njit(cache=True)
def _p_from_v2(v2, r_o, k):
    """Implicit TCR solve for a voltage source: P*(r_o + k*P) = v2  ->  P = 2 v2 / (r_o + sqrt(r_o^2 + 4 k v2)).

    Returns (P, runaway_flag). k = R0*alpha*g (ohm/W) is the in-tick resistance rise per watt; k < 0 with a
    negative discriminant (negative TCR, voltage drive) falls back to the explicit P = v2/r_o and sets the flag.
    """
    disc = r_o * r_o + 4.0 * k * v2
    if disc < 0.0:
        return v2 / r_o, 1
    return 2.0 * v2 / (r_o + math.sqrt(disc)), 0


@njit(cache=True, _nrt=False, inline="always")
def heater_power_row(u, law, t_pre, g_self, r0, alpha, t0, v_fs, i_fs, p_fs, v_comp,
                     out_p, out_v, out_i, out_r, out_flags):
    """Driver law with the heater TCR, for one world (07 P1-11, P1-2 bullet 3, P1-10 compliance; 06 E1).

    Heater resistance: R = R0*(1 + alpha*(T_H - T0)), with T_H the heater-resistor observer of the thermal ROM.
    The observer at the end of the tick is T_H = t_pre + g_self*P, where t_pre holds everything except this tick's
    own heater power (ambient, neighbours, aggressor, the decayed own state) and g_self = sum_j (1 - a_j) r_j (K/W)
    is the instantaneous heater->heater-resistor self gain of the ZOH ROM. Then R = r_o + k*P with
    r_o = R0*(1 + alpha*(t_pre - T0)) and k = R0*alpha*g_self, and the law is solved implicitly:
        voltage: P = V^2/(r_o + kP)            -> P = 2V^2/(r_o + sqrt(r_o^2 + 4kV^2))
        current: P = I^2 (r_o + kP)            -> P = I^2 r_o/(1 - k I^2)    (runaway if k I^2 >= 1)
        power:   P = u*P_FS,  R = r_o + kP,  V = sqrt(P R)
        dac_pwm: P = <V^2>/(r_o + kP), <V^2> = u*V_FS^2 (V, I reported as rms)
    Compliance: if the required V exceeds v_comp, the output saturates as a voltage source at v_comp
    (FLAG_COMPLIANCE). Current-drive runaway (k I^2 >= 1, FLAG_RUNAWAY) ends in compliance; with no compliance
    limit (v_comp = inf) the explicit P = I^2 r_o is returned, so the runaway grows tick by tick and stays finite.
    g_self = 0 gives the explicit (lagged) coupling with T_H = t_pre: for a single pole its
    error multiplier is a + (1 - a)*L (L = electrothermal_loop_gain, about -0.19 for voltage drive at 25 mW), stable
    for |L| < 1 but with a decaying period-2 ringing when a -> 0 (tested); the implicit form (g_self > 0) has the
    multiplier a/(1 - (1 - a)L) and no ringing (same reasoning as 07 P0-1 / D11 for self-heating). With several
    poles these single-pole forms are estimates per pole.
    With g_self = R_th,H and t_pre = T0 it is the static textbook law P(1 + yP) = V^2/R0, y = alpha*R_th,H
    (textbook Ch11 eq. 11.15).
    Inputs (Nx,) arrays except law (int). Units: u dimensionless, T in K (or C, differences only), g_self K/W,
    R0 ohm, alpha 1/K, V_FS V, I_FS A, P_FS W, v_comp V. Outputs: P W, V V, I A, R ohm, flags (bitfield, OR-ed).
    """
    for n in range(u.shape[0]):
        un = u[n]
        if un < 0.0:
            un = 0.0
        rr0 = r0[_ix(r0, n)]
        al = alpha[_ix(alpha, n)]
        r_o = rr0 * (1.0 + al * (t_pre[_ix(t_pre, n)] - t0[_ix(t0, n)]))
        if r_o < 1e-3 * rr0:
            r_o = 1e-3 * rr0  # guard against an unphysical negative resistance from a large negative TCR swing
        k = rr0 * al * g_self[_ix(g_self, n)]
        vc = v_comp[_ix(v_comp, n)]
        fl = 0
        if law == LAW_CURRENT:
            cur = un * i_fs[_ix(i_fs, n)]
            x = k * cur * cur
            sat = False
            runaway_free = False
            if x >= 1.0:
                fl |= FLAG_RUNAWAY
                if math.isinf(vc):
                    # no compliance limit: the implicit solve has no finite root within the tick. Fall back to
                    # the explicit (lagged) power at the start-of-tick resistance, so the runaway shows as growth
                    # over the next ticks instead of a NaN that would poison the thermal state.
                    runaway_free = True
                else:
                    sat = True
                p = cur * cur * r_o
                res = r_o
            else:
                p = cur * cur * r_o / (1.0 - x)
                res = r_o + k * p
                if cur * res > vc:
                    sat = True
            if runaway_free:
                volt = cur * res
            elif sat:
                fl |= FLAG_COMPLIANCE
                p, rf = _p_from_v2(vc * vc, r_o, k)
                if rf:
                    fl |= FLAG_RUNAWAY
                res = r_o + k * p
                volt = vc
                cur = vc / res
            else:
                volt = cur * res
        elif law == LAW_POWER:
            p = un * p_fs[_ix(p_fs, n)]
            res = r_o + k * p
            volt = math.sqrt(p * res)
            if volt > vc:
                fl |= FLAG_COMPLIANCE
                p, rf = _p_from_v2(vc * vc, r_o, k)
                if rf:
                    fl |= FLAG_RUNAWAY
                res = r_o + k * p
                volt = vc
            cur = volt / res
        else:
            vf = v_fs[_ix(v_fs, n)]
            if law == LAW_DAC_PWM:
                if un > 1.0:
                    un = 1.0
                v2 = un * vf * vf
            else:
                v2 = (un * vf) * (un * vf)
            if v2 > vc * vc:
                fl |= FLAG_COMPLIANCE
                v2 = vc * vc
            p, rf = _p_from_v2(v2, r_o, k)
            if rf:
                fl |= FLAG_RUNAWAY
            res = r_o + k * p
            volt = math.sqrt(v2)
            cur = volt / res
        out_p[n] = p
        out_v[n] = volt
        out_i[n] = cur
        out_r[n] = res
        out_flags[n] = out_flags[n] | fl


@njit(cache=True, _nrt=False)
def supply_limit_row(p, v, i, p_max, i_max, flags):
    """Array heater supply limit for one world (07 P1-12): clip and flag.

    Model: a shared heater supply at its limit droops, which scales every heater's drive by one factor s <= 1
    (resistive loads: I and V scale by s, P by s^2):
        s = min(1, sqrt(p_max / sum P), i_max / sum I)
    Use inf for "no limit". The scaled powers are not re-solved for the TCR within the tick (second order; the
    next tick sees the cooler heater). For dac_pwm, i is the rms current, so sum I is the sum of rms currents, an
    upper bound on the mean supply current. Returns s; sets FLAG_SUPPLY on every ring when s < 1.
    """
    ptot = 0.0
    itot = 0.0
    for n in range(p.shape[0]):
        ptot += p[n]
        itot += i[n]
    s = 1.0
    if ptot > p_max and ptot > 0.0:
        s = min(s, math.sqrt(p_max / ptot))
    if itot > i_max and itot > 0.0:
        s = min(s, i_max / itot)
    if s < 1.0:
        for n in range(p.shape[0]):
            p[n] *= s * s
            v[n] *= s
            i[n] *= s
            flags[n] = flags[n] | FLAG_SUPPLY
    return s


# ============================================================================================ heater bank
@njit(cache=True)
def heater_bank_step_row(tick, cmd, cmd_valid,
                         reg, pending, has_pending, last_update, fifo, y_slew, y_out,
                         law, bits, pwm_bits, pwm_ripple, dt_clk, bit_err, bow, code_lo, code_hi,
                         latency_ticks, min_update_ticks, slew_u_per_s, settle_tau_s, dt,
                         r0, alpha, t0, v_fs, i_fs, p_fs, v_comp, p_max, i_max,
                         t_heater_pre, g_self,
                         code_dac, u_target, ripple, u_avg, out_p, out_v, out_i, out_r, out_flags, code_add=None):
    """One plant tick of the heater actuator path for one world, in the 07 P0-6 order (see the module docstring).

    State (N,) arrays, updated in place: reg (= applied_codes), pending, has_pending, last_update, fifo (N, D),
    y_slew, y_out (normalized drive u). Scalars: tick, law, bits, pwm_bits, pwm_ripple (0/1), dt_clk (PWM clocks per
    tick), code_lo/code_hi (register clamp range), latency_ticks, min_update_ticks, dt (s), p_max (W), i_max (A).
    Parameter rows (Nx,): bit_err (Nx, K), bow, slew_u_per_s (1/s), settle_tau_s (s), r0 (ohm), alpha (1/K), t0,
    v_fs (V), i_fs (A), p_fs (W), v_comp (V).
    Thermal inputs (Nx,): t_heater_pre (K) and g_self (K/W), see heater_power_row. Outputs (N,): code_dac (code
    leaving the FIFO), u_target, ripple, u_avg (tick-average drive incl. ripple), out_p (W), out_v (V), out_i (A),
    out_r (ohm), out_flags (reset here). Returns the supply droop factor s.
    code_add (Nx,) int64 or None: a per-tick additive code on the code LEAVING the latency FIFO, clipped to the DAC
    range 0 .. 2^bits - 1 (07 P1-10 dac_glitch fault; scenario signal fault.dac_glitch_codes masked by
    mrm.plant.sensing.FAULT_DAC_GLITCH; integrator addition). The latched register (applied_codes) never shows it.
    """
    latch_row(cmd, cmd_valid, reg, pending, has_pending, last_update, tick, min_update_ticks, code_lo, code_hi)
    fifo_row(reg, fifo, tick, latency_ticks, code_dac)
    if code_add is not None:
        full = (1 << bits) - 1
        for n in range(code_dac.shape[0]):
            c = code_dac[n] + code_add[_ix(code_add, n)]
            code_dac[n] = 0 if c < 0 else (full if c > full else c)
    heater_target_row(code_dac, law, bits, pwm_bits, bit_err, bow, pwm_ripple, tick, dt_clk, u_target, ripple)
    slew_settle_row(u_target, y_slew, y_out, slew_u_per_s, settle_tau_s, dt, u_avg)
    for n in range(u_avg.shape[0]):
        u_avg[n] = u_avg[n] + ripple[n]
        out_flags[n] = 0
    heater_power_row(u_avg, law, t_heater_pre, g_self, r0, alpha, t0, v_fs, i_fs, p_fs, v_comp,
                     out_p, out_v, out_i, out_r, out_flags)
    return supply_limit_row(out_p, out_v, out_i, p_max, i_max, out_flags)


@njit(cache=True)
def heater_bank_step(tick, cmd, cmd_valid,
                     reg, pending, has_pending, last_update, fifo, y_slew, y_out,
                     law, bits, pwm_bits, pwm_ripple, dt_clk, bit_err, bow, code_lo, code_hi,
                     latency_ticks, min_update_ticks, slew_u_per_s, settle_tau_s, dt,
                     r0, alpha, t0, v_fs, i_fs, p_fs, v_comp, p_max, i_max,
                     t_heater_pre, g_self,
                     code_dac, u_target, ripple, u_avg, out_p, out_v, out_i, out_r, out_flags, out_scale,
                     code_add=None):
    """Batched heater_bank_step_row over worlds b (state and outputs (B, N); fifo (B, N, D)).

    law (Bx,) int64, latency_ticks (Bx,) int64, p_max/i_max (Bx,) float; bit_err (Bx, Nx, K); every other
    parameter (Bx, Nx). out_scale (B,) receives the supply droop factor. code_add (Bx, Nx) int64 or None: the
    dac_glitch code offset of heater_bank_step_row. No reduction across b.
    """
    for b in range(reg.shape[0]):
        if code_add is None:
            out_scale[b] = heater_bank_step_row(
                tick, cmd[b], cmd_valid[b],
                reg[b], pending[b], has_pending[b], last_update[b], fifo[b], y_slew[b], y_out[b],
                law[_ix(law, b)], bits, pwm_bits, pwm_ripple, dt_clk, bit_err[_ix(bit_err, b)], bow[_ix(bow, b)],
                code_lo, code_hi, latency_ticks[_ix(latency_ticks, b)], min_update_ticks,
                slew_u_per_s[_ix(slew_u_per_s, b)], settle_tau_s[_ix(settle_tau_s, b)], dt,
                r0[_ix(r0, b)], alpha[_ix(alpha, b)], t0[_ix(t0, b)], v_fs[_ix(v_fs, b)], i_fs[_ix(i_fs, b)],
                p_fs[_ix(p_fs, b)], v_comp[_ix(v_comp, b)], p_max[_ix(p_max, b)], i_max[_ix(i_max, b)],
                t_heater_pre[_ix(t_heater_pre, b)], g_self[_ix(g_self, b)],
                code_dac[b], u_target[b], ripple[b], u_avg[b], out_p[b], out_v[b], out_i[b], out_r[b], out_flags[b])
        else:
            out_scale[b] = heater_bank_step_row(
                tick, cmd[b], cmd_valid[b],
                reg[b], pending[b], has_pending[b], last_update[b], fifo[b], y_slew[b], y_out[b],
                law[_ix(law, b)], bits, pwm_bits, pwm_ripple, dt_clk, bit_err[_ix(bit_err, b)], bow[_ix(bow, b)],
                code_lo, code_hi, latency_ticks[_ix(latency_ticks, b)], min_update_ticks,
                slew_u_per_s[_ix(slew_u_per_s, b)], settle_tau_s[_ix(settle_tau_s, b)], dt,
                r0[_ix(r0, b)], alpha[_ix(alpha, b)], t0[_ix(t0, b)], v_fs[_ix(v_fs, b)], i_fs[_ix(i_fs, b)],
                p_fs[_ix(p_fs, b)], v_comp[_ix(v_comp, b)], p_max[_ix(p_max, b)], i_max[_ix(i_max, b)],
                t_heater_pre[_ix(t_heater_pre, b)], g_self[_ix(g_self, b)],
                code_dac[b], u_target[b], ripple[b], u_avg[b], out_p[b], out_v[b], out_i[b], out_r[b], out_flags[b],
                code_add[_ix(code_add, b)])


# ============================================================================================ bias bank
@njit(cache=True)
def bias_bank_step_row(tick, cmd, cmd_valid,
                       reg, pending, has_pending, last_update, fifo, y_slew, y_out, noise_hold,
                       bits, bit_err, bow, v_min, v_max, latency_ticks, min_update_ticks,
                       noise_period_ticks, noise_rms_codes, z,
                       slew_v_per_s, settle_tau_s, dt, v_offset,
                       code_dac, v_target, out_v_avg, out_v_end):
    """One plant tick of the bias DAC path for one world (07 P0-7 ring.bias_dac, P0-6 order).

    register latch -> latency FIFO -> DAC transfer V = v_min + (v_max - v_min)*frac(code) -> + held DAC noise ->
    slew (slew_v_per_s, V/s) -> settle (settle_tau_s) -> + v_offset (scenario bias_offset_v, additive) -> V_bias.
    Noise: noise_rms_codes (Nx,) rms in codes of this DAC, i.e. V_n = noise_rms_codes*LSB*z with
    LSB = (v_max - v_min)/(2^bits - 1); a new draw from z (N,) (standard normal, from the counter RNG of
    mrm/core/rng.py) is taken on ticks with tick % noise_period_ticks == 0 and held in between, so its spectrum does
    not depend on dt when the period is set by a finite update rate (07 P1-13; with noise_period_ticks = 1 only its
    rms is dt-invariant). z may be (N,) or (1,). Outputs (N,): code_dac, v_target, out_v_avg (tick average, V),
    out_v_end (V).
    """
    code_max = (1 << bits) - 1
    latch_row(cmd, cmd_valid, reg, pending, has_pending, last_update, tick, min_update_ticks, 0, code_max)
    fifo_row(reg, fifo, tick, latency_ticks, code_dac)
    bias_target_row(code_dac, bits, bit_err, bow, v_min, v_max, v_target)
    if tick % noise_period_ticks == 0:
        for n in range(noise_hold.shape[0]):
            lsb = (v_max[_ix(v_max, n)] - v_min[_ix(v_min, n)]) / code_max
            noise_hold[n] = noise_rms_codes[_ix(noise_rms_codes, n)] * lsb * z[_ix(z, n)]
    for n in range(v_target.shape[0]):
        v_target[n] = v_target[n] + noise_hold[n]
    slew_settle_row(v_target, y_slew, y_out, slew_v_per_s, settle_tau_s, dt, out_v_avg)
    for n in range(out_v_avg.shape[0]):
        off = v_offset[_ix(v_offset, n)]
        out_v_avg[n] = out_v_avg[n] + off
        out_v_end[n] = y_out[n] + off


@njit(cache=True)
def bias_bank_step(tick, cmd, cmd_valid,
                   reg, pending, has_pending, last_update, fifo, y_slew, y_out, noise_hold,
                   bits, bit_err, bow, v_min, v_max, latency_ticks, min_update_ticks,
                   noise_period_ticks, noise_rms_codes, z,
                   slew_v_per_s, settle_tau_s, dt, v_offset,
                   code_dac, v_target, out_v_avg, out_v_end):
    """Batched bias_bank_step_row over worlds b; shapes as in heater_bank_step (z and v_offset (Bx, Nx))."""
    for b in range(reg.shape[0]):
        bias_bank_step_row(
            tick, cmd[b], cmd_valid[b],
            reg[b], pending[b], has_pending[b], last_update[b], fifo[b], y_slew[b], y_out[b], noise_hold[b],
            bits, bit_err[_ix(bit_err, b)], bow[_ix(bow, b)], v_min[_ix(v_min, b)], v_max[_ix(v_max, b)],
            latency_ticks[_ix(latency_ticks, b)], min_update_ticks,
            noise_period_ticks, noise_rms_codes[_ix(noise_rms_codes, b)], z[_ix(z, b)],
            slew_v_per_s[_ix(slew_v_per_s, b)], settle_tau_s[_ix(settle_tau_s, b)], dt, v_offset[_ix(v_offset, b)],
            code_dac[b], v_target[b], out_v_avg[b], out_v_end[b])


# ============================================================================================ NRZ levels
@njit(cache=True, _nrt=False)
def nrz_levels_row(v_bias, swing_vpp, mode, v_floor, v_ceil, out_lo, out_hi, out_flags):
    """Junction reverse voltage of the two NRZ levels for one world (07 P1-7; forward bias is negative).

    mode LEVELS_CENTERED:  V_lo = V_bias - Vpp/2, V_hi = V_bias + Vpp/2   (07 P1-7 wording)
    mode LEVELS_ONE_SIDED: V_lo = V_bias - Vpp,   V_hi = V_bias           (bias on one terminal, rail-to-rail driver
                                                                           0..Vpp on the other)
    Levels below v_floor (the forward end of the junction grid, -0.5 V) are clamped there with FLAG_FORWARD_CLIP:
    a depletion LUT cannot describe forward conduction. Levels above v_ceil are clamped with FLAG_GRID_CLIP.
    Units V. v_bias, swing_vpp (Nx,); outputs (N,); flags OR-ed into out_flags.
    """
    for n in range(out_lo.shape[0]):
        vb = v_bias[_ix(v_bias, n)]
        vpp = swing_vpp[_ix(swing_vpp, n)]
        if mode == LEVELS_ONE_SIDED:
            lo = vb - vpp
            hi = vb
        else:
            lo = vb - 0.5 * vpp
            hi = vb + 0.5 * vpp
        fl = 0
        if lo < v_floor:
            lo = v_floor
            fl |= FLAG_FORWARD_CLIP
        if hi < v_floor:
            hi = v_floor
            fl |= FLAG_FORWARD_CLIP
        if hi > v_ceil:
            hi = v_ceil
            fl |= FLAG_GRID_CLIP
        if lo > v_ceil:
            lo = v_ceil
            fl |= FLAG_GRID_CLIP
        out_lo[n] = lo
        out_hi[n] = hi
        out_flags[n] = out_flags[n] | fl


@njit(cache=True)
def nrz_levels(v_bias, swing_vpp, mode, v_floor, v_ceil, out_lo, out_hi, out_flags):
    """Batched nrz_levels_row; v_bias and swing_vpp (Bx, Nx), outputs (B, N)."""
    for b in range(out_lo.shape[0]):
        nrz_levels_row(v_bias[_ix(v_bias, b)], swing_vpp[_ix(swing_vpp, b)], mode, v_floor, v_ceil,
                       out_lo[b], out_hi[b], out_flags[b])


# ============================================================================================ junction runtime
@njit(cache=True, inline="always")
def grid_index(x, grid):
    """np.searchsorted(grid, x) - 1 for grid[0] < x < grid[-1] on a non-decreasing grid, i.e. the unique j with
    grid[j] < x <= grid[j + 1]: guessed from the mean spacing and confirmed by that test (or one step either side),
    else the binary search. The same index without the ~9-step search on the 10 mV junction grid (docs/handoff/perf.md
    P10)."""
    g = grid.shape[0]
    span = grid[_U(g - 1)] - grid[0]
    j = np.int64(0)
    if span > 0.0:
        u = (x - grid[0]) * ((g - 1) / span)
        if u > 0.0:
            j = np.int64(u) if u < g - 2 else np.int64(g - 2)
    # (j >= 0 here: unsigned indices, no wraparound code; docs/handoff/perf.md R2-1)
    if grid[_U(j)] < x and x <= grid[_U(j + 1)]:
        return j
    if j + 2 < g and grid[_U(j + 1)] < x and x <= grid[_U(j + 2)]:
        return j + 1
    if j >= 1 and grid[_U(j - 1)] < x and x <= grid[_U(j)]:
        return j - 1
    return np.int64(np.searchsorted(grid, x) - 1)


@njit(cache=True, _nrt=False)
def _interp_lin(x, grid, col):
    """Linear interpolation on an increasing grid; outside it the end value is held. Returns (value, clipped).
    A NaN input returns (NaN, True), so it is flagged instead of indexing past the end of the grid."""
    g = grid.shape[0]
    if x != x:
        return np.nan, True
    if x <= grid[0]:
        return col[0], x < grid[0]
    if x >= grid[g - 1]:
        return col[g - 1], x > grid[g - 1]
    j = grid_index(x, grid)
    gj = grid[_U(j)]
    w = (x - gj) / (grid[_U(j + 1)] - gj)
    cj = col[_U(j)]
    return cj + w * (col[_U(j + 1)] - cj), False


@njit(cache=True, inline="always")
def _interp_bracket(x, grid):
    """(j, w, clipped) of :func:`_interp_lin` at x; j = -1 outside the grid interior (end value held) and for NaN."""
    g = grid.shape[0]
    if x != x:
        return np.int64(-1), 0.0, True
    if x <= grid[0]:
        return np.int64(-1), 0.0, x < grid[0]
    if x >= grid[g - 1]:
        return np.int64(-1), 0.0, x > grid[g - 1]
    j = grid_index(x, grid)
    gj = grid[_U(j)]
    return j, (x - gj) / (grid[_U(j + 1)] - gj), False


@njit(cache=True, inline="always")
def _interp_at(x, j, w, grid, col):
    """The value of :func:`_interp_lin` at x from its bracket (j, w) of :func:`_interp_bracket`."""
    if x != x:
        return np.nan
    if j < 0:
        return col[0] if x <= grid[0] else col[grid.shape[0] - 1]
    cj = col[_U(j)]
    return cj + w * (col[_U(j + 1)] - cj)


@njit(cache=True)
def _integ_lin(a, b, grid, col):
    """Exact integral of the piecewise-linear (end-held) interpolant of col between a and b."""
    sgn = 1.0
    if b < a:
        a, b = b, a
        sgn = -1.0
    total = 0.0
    g = grid.shape[0]
    # below the grid: held at col[0]
    if a < grid[0]:
        hi = b if b < grid[0] else grid[0]
        total += col[0] * (hi - a)
        a = hi
    # above the grid: held at col[-1]
    if b > grid[g - 1]:
        lo = a if a > grid[g - 1] else grid[g - 1]
        total += col[g - 1] * (b - lo)
        b = lo
    if b > a:
        j = np.searchsorted(grid, a, side="right") - 1
        if j < 0:
            j = 0
        x0 = a
        f0, _ = _interp_lin(x0, grid, col)
        while x0 < b:
            x1 = grid[j + 1] if j + 1 < g else b
            if x1 > b:
                x1 = b
            f1, _ = _interp_lin(x1, grid, col)
            total += 0.5 * (f0 + f1) * (x1 - x0)
            x0 = x1
            f0 = f1
            j += 1
    return sgn * total


@njit(cache=True, _nrt=False)
def junction_eval_row(v, grid, dneff_col, alpha_col, c_col, rs_col, fill, length_m,
                      out_dneff, out_alpha, out_c_f, out_rs_ohm, out_flags):
    """Junction LUT lookup for one world, scaled to the ring (ARCHITECTURE 5.4; 07 P1-7).

    The table is for a junction over the full circumference, at reverse voltage grid (V, forward negative):
    dneff (relative to 0 V), alpha_fca (1/m, absolute), c' (F/m), r_s' (ohm*m). With fill fraction f and ring
    length L (m) the ring-level quantities are
        dneff_ring = f * dneff(V)       (round-trip phase 2*pi/lambda * [n_eff L + f L dneff(V)], 07 P1-7)
        alpha_ring = f * alpha_fca(V)   (ring-averaged free-carrier loss, 1/m; feeds Q_i(V) and eta(V), 07 P0-2)
        C_j        = f * L * c'(V)      (F)
        R_s        = r_s'(V) / (f * L)  (ohm; the junction conducts in parallel along its length)
    Linear interpolation; outside the grid the end value is held and FLAG_GRID_CLIP is set.
    v, fill, length_m (Nx,); outputs (N,).
    Optional temperature axis (07 P1-7): in the depletion approximation every column depends on V + V_bi, so a
    built-in-voltage drift is applied by evaluating at v + dV_bi/dT * (T_junction - T_ref) (dV_bi/dT from
    mrm.materials.dvbi_dt_v_per_k, about -1.1 to -1.2 mV/K at 1e17..1e18 cm^-3, 03 section 5.1; 07 P1-7 quotes
    about -2 mV/K). The caller adds that shift to v (and to both NRZ levels).
    """
    for n in range(out_dneff.shape[0]):
        vv = v[_ix(v, n)]
        f = fill[_ix(fill, n)]
        ln = length_m[_ix(length_m, n)]
        # one bracket for the four columns: the values of four _interp_lin calls (docs/handoff/perf.md P10)
        j, w, c1 = _interp_bracket(vv, grid)
        dn = _interp_at(vv, j, w, grid, dneff_col)
        al = _interp_at(vv, j, w, grid, alpha_col)
        cp = _interp_at(vv, j, w, grid, c_col)
        rsp = _interp_at(vv, j, w, grid, rs_col)
        out_dneff[n] = f * dn
        out_alpha[n] = f * al
        out_c_f[n] = f * ln * cp
        fl_len = f * ln
        out_rs_ohm[n] = rsp / fl_len if fl_len > 0.0 else np.inf
        if c1:
            out_flags[n] = out_flags[n] | FLAG_GRID_CLIP


@njit(cache=True)
def junction_eval(v, grid, dneff_tab, alpha_tab, c_tab, rs_tab, fill, length_m,
                  out_dneff, out_alpha, out_c_f, out_rs_ohm, out_flags):
    """Batched junction_eval_row. Table columns (Bt, G) with Bt in {1, B} (one table per cross-section world);
    v, fill, length_m (Bx, Nx); outputs (B, N)."""
    for b in range(out_dneff.shape[0]):
        junction_eval_row(v[_ix(v, b)], grid, dneff_tab[_ix(dneff_tab, b)], alpha_tab[_ix(alpha_tab, b)],
                          c_tab[_ix(c_tab, b)], rs_tab[_ix(rs_tab, b)], fill[_ix(fill, b)],
                          length_m[_ix(length_m, b)],
                          out_dneff[b], out_alpha[b], out_c_f[b], out_rs_ohm[b], out_flags[b])


@njit(cache=True)
def junction_swing_capacitance_row(v_lo, v_hi, grid, c_col, fill, length_m, out_c_eff):
    """Charge-equivalent junction capacitance over the NRZ swing, for one world (F).

    C_eff = f L (1/(V_hi - V_lo)) * integral_{V_lo}^{V_hi} c'(V) dV = Delta Q / Vpp.
    For a nonlinear C_j(V) driven by a step source through resistors, the energy dissipated per transition,
    averaged over up and down transitions, is exactly 0.5*Vpp*Delta Q = 0.5*C_eff*Vpp^2, so C_eff is the C_j(V)
    that makes the 07 P1-8 formula exact for a large swing. For Vpp -> 0 it is f L c'(V).
    """
    for n in range(out_c_eff.shape[0]):
        lo = v_lo[_ix(v_lo, n)]
        hi = v_hi[_ix(v_hi, n)]
        fl = fill[_ix(fill, n)] * length_m[_ix(length_m, n)]
        if abs(hi - lo) > 1e-9:
            out_c_eff[n] = fl * _integ_lin(lo, hi, grid, c_col) / (hi - lo)
        else:
            cp, _ = _interp_lin(0.5 * (lo + hi), grid, c_col)
            out_c_eff[n] = fl * cp


@njit(cache=True)
def junction_swing_capacitance(v_lo, v_hi, grid, c_tab, fill, length_m, out_c_eff):
    """Batched junction_swing_capacitance_row; c_tab (Bt, G); other inputs (Bx, Nx); output (B, N)."""
    for b in range(out_c_eff.shape[0]):
        junction_swing_capacitance_row(v_lo[_ix(v_lo, b)], v_hi[_ix(v_hi, b)], grid, c_tab[_ix(c_tab, b)],
                                       fill[_ix(fill, b)], length_m[_ix(length_m, b)], out_c_eff[b])


# ============================================================================================ data-activity heat
@njit(cache=True, _nrt=False)
def data_activity_heat_row(c_eff, swing_vpp, bitrate_hz, transition_density, enable, r_s, r_drv,
                           p_static_w, e_tr_j, add_load_share, out_pj, out_pd):
    """Data-activity heat for one world (07 P1-8; docs/research/03 section 6.2).

    Junction (ROM source 'junction'):
        P_j = 0.5 * C_j * Vpp^2 * bitrate * transition_density * R_s/(R_s + R_drv)       [W]
    Driver (ROM source 'driver', at driver.site):
        P_d = P_static + E_tr * bitrate * transition_density                             [W]
    Both are multiplied by driver_enable (0/1); squelch is swing_vpp = 0 with the driver enabled. C_j should be the
    charge-equivalent capacitance over the swing (junction_swing_capacitance_row). With add_load_share = 1 the driver
    also takes the R_drv share of the junction charging energy, 0.5*C_j*Vpp^2*bitrate*td*R_drv/(R_s + R_drv);
    leave it 0 when E_tr already includes the load (the contract form). Random NRZ has transition_density 0.5.
    Pass swing_vpp = V_hi - V_lo of the junction levels actually applied (after nrz_levels_row clamping), the same
    interval over which C_eff was integrated; the dissipation per transition is 0.5*(V_hi - V_lo)*Delta Q.
    Inputs (Nx,): c_eff F, swing_vpp V, transition_density, enable, r_s ohm, r_drv ohm, p_static_w W, e_tr_j J;
    bitrate_hz scalar (bits per second). Outputs (N,) W.
    """
    for n in range(out_pj.shape[0]):
        en = enable[_ix(enable, n)]
        vpp = swing_vpp[_ix(swing_vpp, n)]
        td = transition_density[_ix(transition_density, n)]
        rs = r_s[_ix(r_s, n)]
        rd = r_drv[_ix(r_drv, n)]
        e_load = 0.5 * c_eff[_ix(c_eff, n)] * vpp * vpp * bitrate_hz * td
        share = rs / (rs + rd) if (rs + rd) > 0.0 else 1.0
        if math.isinf(rs):
            share = 1.0
        out_pj[n] = en * e_load * share
        pd = p_static_w[_ix(p_static_w, n)] + e_tr_j[_ix(e_tr_j, n)] * bitrate_hz * td
        if add_load_share != 0:
            pd += e_load * (1.0 - share)
        out_pd[n] = en * pd


@njit(cache=True)
def data_activity_heat(c_eff, swing_vpp, bitrate_hz, transition_density, enable, r_s, r_drv,
                       p_static_w, e_tr_j, add_load_share, out_pj, out_pd):
    """Batched data_activity_heat_row; inputs (Bx, Nx), outputs (B, N)."""
    for b in range(out_pj.shape[0]):
        data_activity_heat_row(c_eff[_ix(c_eff, b)], swing_vpp[_ix(swing_vpp, b)], bitrate_hz,
                               transition_density[_ix(transition_density, b)], enable[_ix(enable, b)],
                               r_s[_ix(r_s, b)], r_drv[_ix(r_drv, b)], p_static_w[_ix(p_static_w, b)],
                               e_tr_j[_ix(e_tr_j, b)], add_load_share, out_pj[b], out_pd[b])


# ================================================================================================ Python API
def _f2(x, B, N, name):
    """Float array broadcastable to (B, N), returned with shape (1|B, 1|N).

    A 1-D array of length N is per ring (1, N); of length B (!= N) per world (B, 1). Pass 2-D to be explicit.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a.reshape(1, -1) if a.shape[0] == N else a.reshape(-1, 1)
    if a.ndim != 2 or a.shape[0] not in (1, B) or a.shape[1] not in (1, N):
        raise ValueError(f"{name}: shape {np.shape(x)} does not broadcast to (B={B}, N={N})")
    return np.ascontiguousarray(a)


def _f1(x, B, name, dtype=np.float64):
    a = np.atleast_1d(np.asarray(x, dtype=dtype))
    if a.ndim != 1 or a.shape[0] not in (1, B):
        raise ValueError(f"{name}: shape {np.shape(x)} is not scalar or (B={B},)")
    return np.ascontiguousarray(a)


def _ticks(seconds, dt_s, name):
    """Quantize a hardware latency/period to whole ticks with the one tick convention of mrm.core.timing
    (seconds_to_ticks: same tolerance, rejects negative, non-finite and non-whole values, names the key; 07 P0-6).
    Raises TimingError, a ValueError."""
    s = np.atleast_1d(np.asarray(seconds, dtype=np.float64))
    return np.array([seconds_to_ticks(float(x), dt_s, name) for x in s.reshape(-1)], dtype=np.int64).reshape(s.shape)


def _rate_ticks(rate_hz, dt_s, name):
    """Period of a rate in whole ticks (>= 1), mrm.core.timing.rate_to_period_ticks; inf (no limit) -> 1."""
    f = float(rate_hz)
    if math.isinf(f) and f > 0:
        return 1
    return rate_to_period_ticks(f, dt_s, name)


def _check_full_scales(law_codes, B, N, v_fs, i_fs, p_fs):
    """Each world's driver law needs a finite, positive full scale for every ring (0 * inf would give NaN power)."""
    need = {LAW_VOLTAGE: ("v_full_scale_v", v_fs), LAW_DAC_PWM: ("v_full_scale_v", v_fs),
            LAW_CURRENT: ("i_full_scale_a", i_fs), LAW_POWER: ("p_full_scale_w", p_fs)}
    for b in range(B):
        lc = int(law_codes[min(b, law_codes.shape[0] - 1)])
        name, arr = need[lc]
        row = arr[min(b, arr.shape[0] - 1)]
        if not np.all(np.isfinite(row) & (row > 0)):
            law_name = {v: k for k, v in DRIVER_LAWS.items()}[lc]
            raise ValueError(f"driver law {law_name!r} of world {b} needs a finite, positive {name} (got {row})")


def _bit_err(bit_err_lsb, B, N, n_bits, name):
    if bit_err_lsb is None:
        return np.zeros((1, 1, max(n_bits, 1)))
    a = np.asarray(bit_err_lsb, dtype=np.float64)
    while a.ndim < 3:
        a = a[None]
    if a.shape[0] not in (1, B) or a.shape[1] not in (1, N) or a.shape[2] < n_bits:
        raise ValueError(f"{name}: shape {a.shape} must be (1|B, 1|N, >= {n_bits})")
    return np.ascontiguousarray(a)


@dataclass
class HeaterDriverParams:
    """Parameters of the heater actuator path for B worlds and N rings, SI, already realized (no sampling here).

    Every array broadcasts over (B, N) as described in heater_bank_step. Build with ``HeaterDriverParams.build``.
    law: (B,) int64 codes of DRIVER_LAWS. bits: DAC code width (16: G26 codes exceed 2^15, inferred, docs/research/01
    section 2.4). pwm_bits: PWM part of the code for dac_pwm (8, PAT-2). latency_ticks: (B,) DAC latency.
    bit_err_lsb (Bx, Nx, K): bit-weight errors of the DAC the law uses, in that DAC's LSB; for dac_pwm the first
    bits - pwm_bits entries belong to the coarse 8-bit DAC (the PWM itself is clock-exact). slew_u_per_s is the
    slew in normalized drive per second (slew_codes_per_s / code_max); for dac_pwm it acts on the period-average
    power fraction u, not on the code.
    """

    dt_s: float
    law: np.ndarray
    bits: int
    pwm_bits: int
    pwm_ripple: int
    dt_clk: int
    code_lo: int
    code_hi: int
    bit_err_lsb: np.ndarray
    inl_bow_lsb: np.ndarray
    latency_ticks: np.ndarray
    fifo_depth: int
    min_update_ticks: int
    slew_u_per_s: np.ndarray
    settle_tau_s: np.ndarray
    r_cold_ohm: np.ndarray
    tcr_per_k: np.ndarray
    tcr_ref_k: np.ndarray
    v_full_scale_v: np.ndarray
    i_full_scale_a: np.ndarray
    p_full_scale_w: np.ndarray
    compliance_v: np.ndarray
    supply_power_max_w: np.ndarray
    supply_current_max_a: np.ndarray
    B: int = 1
    N: int = 1

    @property
    def code_full_scale(self) -> int:
        """Full-scale code 2^bits - 1 (normalization of u); the register clamp range is [code_lo, code_hi]."""
        return (1 << self.bits) - 1

    @classmethod
    def build(cls, B: int, N: int, dt_s: float, *, law="voltage", bits: int = 16, pwm_bits: int = PAT2_PWM_BITS,
              pwm_period_s: float = PAT2_PWM_PERIOD_S, pwm_ripple: bool = False,
              r_cold_ohm, tcr_per_k, tcr_ref_k=298.15,
              v_full_scale_v=np.inf, i_full_scale_a=np.inf, p_full_scale_w=np.inf, compliance_v=np.inf,
              code_min: int = 0, code_max: int | None = None,
              latency_s=0.0, max_latency_s=None, dac_max_update_hz=np.inf, slew_codes_per_s=np.inf,
              settle_tau_s=0.0, bit_err_lsb=None, inl_bow_lsb=0.0,
              supply_power_max_w=np.inf, supply_current_max_a=np.inf) -> "HeaterDriverParams":
        """Build from SI values (scalars or arrays broadcastable to (B, N); law/latency per world (B,)).

        law: name or code, or a sequence of them per world. code_min/code_max: register clamp range (default the
        full 0 .. 2^bits - 1, 07 P1-11; never REF.heaterDacCodeRange). slew_codes_per_s: inf = unlimited (07 P1-11
        default; never REF.heaterDacSlewCodesPerMs). latency_s: DAC latency (whole ticks). max_latency_s sizes the FIFO
        (defaults to the largest latency). dac_max_update_hz: register update rate limit (inf = every tick).
        Full scales that the chosen law does not use may stay inf; the law's own full scale must be finite (checked).
        tcr_ref_k: temperature at which R = r_cold_ohm (default 298.15 K = 25 C, assumed: the G25 wafer maps are
        at 25 C, 07 4.3 CAL-3; the preset key ring.heater.r_ref_temp_c overrides it).
        """
        laws = np.atleast_1d(np.asarray(law, dtype=object))
        law_codes = np.array([DRIVER_LAWS[x] if isinstance(x, str) else int(x) for x in laws], dtype=np.int64)
        law_codes = _f1(law_codes, B, "law", np.int64)
        if np.any((law_codes < 0) | (law_codes > 3)):
            raise ValueError("law must be one of voltage, current, power, dac_pwm")
        if not 1 <= bits <= 30:
            raise ValueError("bits must be in 1..30")
        if np.any(law_codes == LAW_DAC_PWM) and not 1 <= pwm_bits < bits:
            raise ValueError("dac_pwm needs 1 <= pwm_bits < bits")
        dt_clk = 1
        if pwm_ripple and np.any(law_codes == LAW_DAC_PWM):
            clk = pwm_period_s / (1 << pwm_bits)
            dt_clk = int(round(dt_s / clk))
            if dt_clk < 1 or abs(dt_clk * clk - dt_s) > 1e-6 * clk:
                raise ValueError(f"pwm_ripple needs dt ({dt_s} s) to be a whole number of PWM clocks ({clk} s)")
        lat = _f1(_ticks(latency_s, dt_s, "ring.heater.latency_s"), B, "latency_s", np.int64)
        depth = int(lat.max()) + 1
        if max_latency_s is not None:
            depth = max(depth, int(_ticks(max_latency_s, dt_s, "max_latency_s").max()) + 1)
        min_upd = _rate_ticks(dac_max_update_hz, dt_s, "ring.heater.dac_max_update_hz")
        n_dac = bits - pwm_bits if np.all(law_codes == LAW_DAC_PWM) else bits
        full = (1 << bits) - 1
        hi = full if code_max is None else int(code_max)
        if not 0 <= int(code_min) <= hi <= full:
            raise ValueError(f"need 0 <= code_min <= code_max <= {full}")
        v_fs = _f2(v_full_scale_v, B, N, "v_full_scale_v")
        i_fs = _f2(i_full_scale_a, B, N, "i_full_scale_a")
        p_fs = _f2(p_full_scale_w, B, N, "p_full_scale_w")
        _check_full_scales(law_codes, B, N, v_fs, i_fs, p_fs)
        r0 = _f2(r_cold_ohm, B, N, "r_cold_ohm")
        slew = _f2(np.asarray(slew_codes_per_s, dtype=np.float64) / full, B, N, "slew_codes_per_s")
        settle = _f2(settle_tau_s, B, N, "settle_tau_s")
        comp = _f2(compliance_v, B, N, "compliance_v")
        if not (np.all(r0 > 0) and np.all(slew > 0) and np.all(settle >= 0) and np.all(comp > 0)):
            raise ValueError("need r_cold_ohm > 0, slew_codes_per_s > 0, settle_tau_s >= 0, compliance_v > 0")
        if pwm_ripple and np.any(law_codes == LAW_DAC_PWM) and np.any(settle > 0):
            warnings.warn("pwm_ripple with settle_tau_s > 0: the PWM ripple term is added after the settle filter, "
                          "so it is not attenuated by it (the ripple is overestimated)", stacklevel=2)
        return cls(
            dt_s=float(dt_s), law=law_codes, bits=int(bits), pwm_bits=int(pwm_bits), pwm_ripple=int(bool(pwm_ripple)),
            dt_clk=dt_clk, code_lo=int(code_min), code_hi=hi,
            bit_err_lsb=_bit_err(bit_err_lsb, B, N, n_dac, "bit_err_lsb"),
            inl_bow_lsb=_f2(inl_bow_lsb, B, N, "inl_bow_lsb"), latency_ticks=lat, fifo_depth=depth,
            min_update_ticks=max(min_upd, 1), slew_u_per_s=slew, settle_tau_s=settle, r_cold_ohm=r0,
            tcr_per_k=_f2(tcr_per_k, B, N, "tcr_per_k"), tcr_ref_k=_f2(tcr_ref_k, B, N, "tcr_ref_k"),
            v_full_scale_v=v_fs, i_full_scale_a=i_fs, p_full_scale_w=p_fs, compliance_v=comp,
            supply_power_max_w=_f1(supply_power_max_w, B, "supply_power_max_w"),
            supply_current_max_a=_f1(supply_current_max_a, B, "supply_current_max_a"),
            B=int(B), N=int(N))

    @classmethod
    def from_plant_params(cls, pp, *, transport_delay: bool = True, max_latency_s=None,
                          **overrides) -> "HeaterDriverParams":
        """Build from ``mrm.params.realize`` output (PlantParams; docs/PARAMETERS_FORMAT.md section 10).

        Field map (integrator, docs/handoff/ENGINE_BRIEF.md): law = heater.driver_mode (B,), bits = heater.dac_bits,
        pwm_bits / pwm_period_s = heater.pwm_bits / pwm_period_s (PAT-2 defaults when absent), pwm_ripple =
        flags.heater_pwm_ripple, r_cold_ohm (B, N), tcr_per_k, tcr_ref_k = heater.r_ref_temp_k, full scales
        (nan -> inf; the law's own full scale is checked), compliance_v, code_min/code_max, dac_max_update_hz =
        heater.max_update_hz, slew_codes_per_s, settle_tau_s, supply limits. latency_s = (timing.heater_latency_ticks
        + the thermal transport delay in whole ticks) * dt: 07 P1-2 / 4.8 put a 0-2 us heater-to-ring transport delay
        per world in ThermalSystem.transport_delay_s, and the thermal owner asks the heater FIFO to apply it (one
        owner per delay, 07 P0-6); it is rounded to the nearest tick (pass transport_delay=False to leave it out).
        DAC errors: bit_err_lsb = dac_bit_errors_from_spec(dac_dnl_lsb, n, heater.dac_bit_err_z[..., :n]) with
        n = bits - pwm_bits for a dac_pwm world (the coarse DAC) and n = bits otherwise; inl_bow_lsb =
        heater.inl_bow_u * dac_inl_lsb. Keyword arguments override any :meth:`build` argument.
        """
        h, t = pp.heater, pp.timing
        B, N = int(pp.batch), int(pp.n_rings)
        dt = float(t.dt_s)
        law = np.asarray(h.driver_mode, dtype=np.int64).reshape(-1)
        bits = int(h.dac_bits)
        pwm_bits = int(h.pwm_bits) if int(h.pwm_bits) > 0 else PAT2_PWM_BITS
        pwm_period = np.asarray(h.pwm_period_s, dtype=float).reshape(-1)
        pwm_period = float(pwm_period[np.isfinite(pwm_period)][0]) if np.any(np.isfinite(pwm_period)) \
            else PAT2_PWM_PERIOD_S
        lat = np.asarray(t.heater_latency_ticks, dtype=np.int64).reshape(-1)
        sysm = getattr(pp.thermal, "system", None)
        td = getattr(sysm, "transport_delay_s", None) if sysm is not None else None
        if transport_delay and td is not None:
            lat = lat + np.rint(np.broadcast_to(np.asarray(td, dtype=float).reshape(-1), (B,)) / dt).astype(np.int64)
        z = np.asarray(h.dac_bit_err_z, dtype=float)
        dnl = np.broadcast_to(np.asarray(h.dac_dnl_lsb, dtype=float).reshape(-1), (B,))
        err = np.zeros((B, N, bits))
        for b in range(B):
            nb = bits - pwm_bits if law[b] == LAW_DAC_PWM else bits
            if dnl[b] > 0.0:
                err[b, :, :nb] = dac_bit_errors_from_spec(float(dnl[b]), nb, z[b, :, :nb])
        bow = np.asarray(h.inl_bow_u, dtype=float) * np.asarray(h.dac_inl_lsb, dtype=float).reshape(-1, 1)

        def fs(x):
            x = np.asarray(x, dtype=float).reshape(-1, 1)
            return np.where(np.isfinite(x), x, np.inf)
        kw = dict(law=law, bits=bits, pwm_bits=pwm_bits, pwm_period_s=pwm_period,
                  pwm_ripple=bool(pp.flags.heater_pwm_ripple), r_cold_ohm=np.asarray(h.r_cold_ohm, dtype=float),
                  tcr_per_k=np.asarray(h.tcr_per_k, dtype=float), tcr_ref_k=np.asarray(h.r_ref_temp_k, dtype=float),
                  v_full_scale_v=fs(h.full_scale_v), i_full_scale_a=fs(h.full_scale_a), p_full_scale_w=fs(h.full_scale_w),
                  compliance_v=fs(h.compliance_v), code_min=int(h.code_min), code_max=int(h.code_max),
                  latency_s=lat * dt, max_latency_s=max_latency_s, dac_max_update_hz=float(h.max_update_hz),
                  slew_codes_per_s=np.asarray(h.slew_codes_per_s, dtype=float),
                  settle_tau_s=np.asarray(h.settle_tau_s, dtype=float), bit_err_lsb=err, inl_bow_lsb=bow,
                  supply_power_max_w=float(h.supply_power_max_w), supply_current_max_a=float(h.supply_current_max_a))
        kw.update(overrides)
        return cls.build(B, N, dt, **kw)


class HeaterTickOut(NamedTuple):
    """Outputs of one heater tick, (B, N) unless noted. Views of internal buffers: copy them to keep them."""

    p_w: np.ndarray            # heater electrical power, W (ROM source 'heater')
    v_v: np.ndarray            # heater voltage (rms for dac_pwm), V  (readback source)
    i_a: np.ndarray            # heater current (rms for dac_pwm), A  (readback source)
    r_ohm: np.ndarray          # heater resistance at the end of the tick, ohm
    u: np.ndarray              # tick-average normalized drive after slew/settle (+ PWM ripple)
    code_dac: np.ndarray       # code leaving the latency FIFO
    applied_codes: np.ndarray  # DAC register readback (latched value), int64
    flags: np.ndarray          # FLAG_* bitfield, int8
    supply_scale: np.ndarray   # (B,) supply droop factor (1 = no limit hit)


class HeaterBank:
    """State and stepping of the heater actuator path for (B, N) rings (07 P0-6, P1-10, P1-11, P1-12).

    Usage::
        prm = HeaterDriverParams.build(B, N, 1e-6, law="voltage", v_full_scale_v=1.6, r_cold_ohm=45.4,
                                       tcr_per_k=1e-3, tcr_ref_k=298.15, latency_s=2e-6)
        bank = HeaterBank(prm, initial_code=0)
        out = bank.step(tick, cmd, cmd_valid, t_heater_pre_k, g_self_k_per_w)   # once per plant tick, in order
    t_heater_pre_k / g_self_k_per_w come from the thermal ROM's heater_resistor observer (see heater_power_row);
    pass g_self = 0 and the current observer value for the explicit coupling.
    """

    def __init__(self, params: HeaterDriverParams, initial_code=0):
        p = self.p = params
        B, N = p.B, p.N
        code0 = np.broadcast_to(np.asarray(initial_code, dtype=np.int64), (B, N)).copy()
        code0 = np.clip(code0, p.code_lo, p.code_hi)
        self.reg = code0.copy()
        self.pending = code0.copy()
        self.has_pending = np.zeros((B, N), np.int8)
        self.last_update = np.full((B, N), _NEVER, np.int64)
        self.fifo = np.repeat(code0[:, :, None], p.fifo_depth, axis=2)
        u0 = np.empty((B, N))
        rip = np.empty((B, N))
        for b in range(B):
            heater_target_row(code0[b], int(p.law[min(b, p.law.shape[0] - 1)]), p.bits, p.pwm_bits,
                              p.bit_err_lsb[min(b, p.bit_err_lsb.shape[0] - 1)],
                              p.inl_bow_lsb[min(b, p.inl_bow_lsb.shape[0] - 1)], 0, 0, 1, u0[b], rip[b])
        self.y_slew = u0.copy()
        self.y_out = u0.copy()
        self._code_dac = np.zeros((B, N), np.int64)
        self._u_target = np.zeros((B, N))
        self._ripple = np.zeros((B, N))
        self._u_avg = np.zeros((B, N))
        self._p = np.zeros((B, N))
        self._v = np.zeros((B, N))
        self._i = np.zeros((B, N))
        self._r = np.zeros((B, N))
        self._flags = np.zeros((B, N), np.int8)
        self._scale = np.ones(B)
        self._no_cmd = np.zeros((B, N), np.int64)
        self._no_valid = np.zeros((B, N), np.bool_)

    @property
    def applied_codes(self) -> np.ndarray:
        """DAC register readback (latched value), (B, N) int64 (07 P0-6)."""
        return self.reg

    def step(self, tick: int, cmd=None, cmd_valid=None, t_heater_pre_k=None, g_self_k_per_w=0.0,
             code_add=None) -> HeaterTickOut:
        """Advance one tick. cmd (B, N) int codes with cmd_valid (B, N) bool (default: all valid when cmd is given).
        t_heater_pre_k (B|1, N|1) K defaults to the TCR reference temperature (no heating). code_add (B|1, N|1) int
        codes added to the code leaving the latency FIFO this tick (07 P1-10 dac_glitch; None = no glitch)."""
        p = self.p
        B, N = p.B, p.N
        if cmd is None:
            cmd_a, valid = self._no_cmd, self._no_valid
        else:
            cmd_a = np.ascontiguousarray(np.broadcast_to(np.asarray(cmd, dtype=np.int64), (B, N)))
            valid = (np.ones((B, N), np.bool_) if cmd_valid is None
                     else np.ascontiguousarray(np.broadcast_to(np.asarray(cmd_valid, dtype=np.bool_), (B, N))))
        t_pre = p.tcr_ref_k if t_heater_pre_k is None else _f2(t_heater_pre_k, B, N, "t_heater_pre_k")
        g = _f2(g_self_k_per_w, B, N, "g_self_k_per_w")
        heater_bank_step(np.int64(tick), cmd_a, valid, self.reg, self.pending, self.has_pending, self.last_update,
                         self.fifo, self.y_slew, self.y_out, p.law, p.bits, p.pwm_bits, p.pwm_ripple, p.dt_clk,
                         p.bit_err_lsb, p.inl_bow_lsb, p.code_lo, p.code_hi, p.latency_ticks, p.min_update_ticks,
                         p.slew_u_per_s,
                         p.settle_tau_s, p.dt_s, p.r_cold_ohm, p.tcr_per_k, p.tcr_ref_k, p.v_full_scale_v,
                         p.i_full_scale_a, p.p_full_scale_w, p.compliance_v, p.supply_power_max_w,
                         p.supply_current_max_a, t_pre, g, self._code_dac, self._u_target, self._ripple, self._u_avg,
                         self._p, self._v, self._i, self._r, self._flags, self._scale,
                         None if code_add is None else np.ascontiguousarray(np.broadcast_to(
                             np.asarray(code_add, dtype=np.int64), (B, N))))
        return HeaterTickOut(self._p, self._v, self._i, self._r, self._u_avg, self._code_dac, self.reg, self._flags,
                             self._scale)


@dataclass
class BiasDacParams:
    """Parameters of the bias DAC path (07 P0-7 ring.bias_dac), realized, SI. Build with ``BiasDacParams.build``."""

    dt_s: float
    bits: int
    bit_err_lsb: np.ndarray
    inl_bow_lsb: np.ndarray
    v_min_v: np.ndarray
    v_max_v: np.ndarray
    latency_ticks: np.ndarray
    fifo_depth: int
    min_update_ticks: int
    noise_period_ticks: int
    noise_rms_codes: np.ndarray
    slew_v_per_s: np.ndarray
    settle_tau_s: np.ndarray
    B: int = 1
    N: int = 1

    @property
    def code_max(self) -> int:
        return (1 << self.bits) - 1

    @property
    def lsb_v(self) -> np.ndarray:
        return (self.v_max_v - self.v_min_v) / self.code_max

    def code_for_voltage(self, v) -> np.ndarray:
        """Nearest code for a bias voltage (ideal transfer), clipped to the code range."""
        c = np.rint((np.asarray(v, dtype=np.float64) - self.v_min_v) / self.lsb_v)
        return np.clip(c, 0, self.code_max).astype(np.int64)

    @classmethod
    def build(cls, B: int, N: int, dt_s: float, *, bits: int, v_min_v=G26_BIAS_V_MIN, v_max_v=G26_BIAS_V_MAX,
              update_rate_hz=np.inf, latency_s=0.0, max_latency_s=None, slew_v_per_s=np.inf, settle_tau_s=0.0,
              noise_rms_codes=0.0, bit_err_lsb=None, inl_bow_lsb=0.0) -> "BiasDacParams":
        """Build from SI values. v_min_v/v_max_v default to the printed G26 axis (1.0 / 3.75 V); bits, rate,
        latency, slew, settle and noise have no Lightmatter source (tag them assumed in the preset, 07 P0-7).

        update_rate_hz sets both the register update interval and the noise redraw period. With update_rate_hz =
        inf the noise is redrawn every plant tick: its rms (noise_rms_codes) is then dt-invariant, but its spectral
        density scales with dt. Give a finite update rate for a dt-invariant noise spectrum (07 P1-13)."""
        if not 1 <= int(bits) <= 30:
            raise ValueError("bias DAC bits must be in 1..30")
        lat = _f1(_ticks(latency_s, dt_s, "ring.bias_dac.latency_s"), B, "latency_s", np.int64)
        depth = int(lat.max()) + 1
        if max_latency_s is not None:
            depth = max(depth, int(_ticks(max_latency_s, dt_s, "max_latency_s").max()) + 1)
        per = _rate_ticks(update_rate_hz, dt_s, "ring.bias_dac.update_rate_hz")
        vmin = _f2(v_min_v, B, N, "v_min_v")
        vmax = _f2(v_max_v, B, N, "v_max_v")
        slew = _f2(slew_v_per_s, B, N, "slew_v_per_s")
        settle = _f2(settle_tau_s, B, N, "settle_tau_s")
        noise = _f2(noise_rms_codes, B, N, "noise_rms_codes")
        if not (np.all(vmax > vmin) and np.all(slew > 0) and np.all(settle >= 0) and np.all(noise >= 0)):
            raise ValueError("need v_max_v > v_min_v, slew_v_per_s > 0, settle_tau_s >= 0, noise_rms_codes >= 0")
        return cls(dt_s=float(dt_s), bits=int(bits), bit_err_lsb=_bit_err(bit_err_lsb, B, N, bits, "bit_err_lsb"),
                   inl_bow_lsb=_f2(inl_bow_lsb, B, N, "inl_bow_lsb"), v_min_v=vmin, v_max_v=vmax,
                   latency_ticks=lat, fifo_depth=depth, min_update_ticks=max(per, 1), noise_period_ticks=max(per, 1),
                   noise_rms_codes=noise, slew_v_per_s=slew, settle_tau_s=settle, B=int(B), N=int(N))

    @classmethod
    def from_plant_params(cls, pp, *, max_latency_s=None, **overrides) -> "BiasDacParams":
        """Build from PlantParams.bias_dac and timing (docs/PARAMETERS_FORMAT.md section 10): bits, v_min_v,
        v_max_v, update rate = 1/(update_period_ticks dt) (realize gives 1 tick when ring.bias_dac.update_rate_hz is
        absent: one update and one noise redraw per tick), latency_s = timing.bias_latency_ticks dt, slew_v_per_s,
        settle_tau_s, noise_rms_codes. The bias DAC has no INL/DNL keys (ideal transfer). Keyword arguments override
        any :meth:`build` argument."""
        d, t = pp.bias_dac, pp.timing
        B, N = int(pp.batch), int(pp.n_rings)
        dt = float(t.dt_s)
        per = max(int(d.update_period_ticks), 1)
        kw = dict(bits=int(d.bits), v_min_v=np.asarray(d.v_min_v, dtype=float),
                  v_max_v=np.asarray(d.v_max_v, dtype=float), update_rate_hz=1.0 / (per * dt),
                  latency_s=np.asarray(t.bias_latency_ticks, dtype=np.int64).reshape(-1) * dt,
                  max_latency_s=max_latency_s, slew_v_per_s=np.asarray(d.slew_v_per_s, dtype=float),
                  settle_tau_s=np.asarray(d.settle_tau_s, dtype=float),
                  noise_rms_codes=np.asarray(d.noise_rms_codes, dtype=float))
        kw.update(overrides)
        return cls.build(B, N, dt, **kw)


class BiasTickOut(NamedTuple):
    """Outputs of one bias tick, (B, N). Views of internal buffers."""

    v_avg: np.ndarray          # tick-average bias voltage incl. noise and offset, V (reverse positive)
    v_end: np.ndarray          # end-of-tick bias voltage, V
    code_dac: np.ndarray       # code leaving the latency FIFO
    applied_codes: np.ndarray  # register readback (latched value), int64


class BiasBank:
    """State and stepping of the bias DAC path for (B, N) rings (07 P0-7 tx.bias; P0-6 order)."""

    def __init__(self, params: BiasDacParams, initial_code=0):
        p = self.p = params
        B, N = p.B, p.N
        code0 = np.clip(np.broadcast_to(np.asarray(initial_code, dtype=np.int64), (B, N)).copy(), 0, p.code_max)
        self.reg = code0.copy()
        self.pending = code0.copy()
        self.has_pending = np.zeros((B, N), np.int8)
        self.last_update = np.full((B, N), _NEVER, np.int64)
        self.fifo = np.repeat(code0[:, :, None], p.fifo_depth, axis=2)
        v0 = np.empty((B, N))
        for b in range(B):
            bias_target_row(code0[b], p.bits, p.bit_err_lsb[min(b, p.bit_err_lsb.shape[0] - 1)],
                            p.inl_bow_lsb[min(b, p.inl_bow_lsb.shape[0] - 1)],
                            p.v_min_v[min(b, p.v_min_v.shape[0] - 1)], p.v_max_v[min(b, p.v_max_v.shape[0] - 1)],
                            v0[b])
        self.y_slew = v0.copy()
        self.y_out = v0.copy()
        self.noise_hold = np.zeros((B, N))
        self._code_dac = np.zeros((B, N), np.int64)
        self._v_target = np.zeros((B, N))
        self._v_avg = np.zeros((B, N))
        self._v_end = np.zeros((B, N))
        self._zero = np.zeros((1, 1))
        self._zz = np.zeros((B, N))
        self._no_cmd = np.zeros((B, N), np.int64)
        self._no_valid = np.zeros((B, N), np.bool_)

    @property
    def applied_codes(self) -> np.ndarray:
        return self.reg

    def step(self, tick: int, cmd=None, cmd_valid=None, z=None, v_offset=None) -> BiasTickOut:
        """Advance one tick. z (B, N) standard-normal draws for the DAC noise (from the counter RNG);
        v_offset (B|1, N|1) V, the scenario's additive bias_offset_v."""
        p = self.p
        B, N = p.B, p.N
        if cmd is None:
            cmd_a, valid = self._no_cmd, self._no_valid
        else:
            cmd_a = np.ascontiguousarray(np.broadcast_to(np.asarray(cmd, dtype=np.int64), (B, N)))
            valid = (np.ones((B, N), np.bool_) if cmd_valid is None
                     else np.ascontiguousarray(np.broadcast_to(np.asarray(cmd_valid, dtype=np.bool_), (B, N))))
        zz = self._zz if z is None else np.ascontiguousarray(np.broadcast_to(np.asarray(z, np.float64), (B, N)))
        off = self._zero if v_offset is None else _f2(v_offset, B, N, "v_offset")
        bias_bank_step(np.int64(tick), cmd_a, valid, self.reg, self.pending, self.has_pending, self.last_update,
                       self.fifo, self.y_slew, self.y_out, self.noise_hold, p.bits, p.bit_err_lsb, p.inl_bow_lsb,
                       p.v_min_v, p.v_max_v, p.latency_ticks, p.min_update_ticks, p.noise_period_ticks,
                       p.noise_rms_codes, zz, p.slew_v_per_s, p.settle_tau_s, p.dt_s, off,
                       self._code_dac, self._v_target, self._v_avg, self._v_end)
        return BiasTickOut(self._v_avg, self._v_end, self._code_dac, self.reg)


# ------------------------------------------------------------------------------------------ junction tables
@dataclass
class JunctionTable:
    """Junction LUT for a full-circumference junction (ARCHITECTURE 5.4, schema junction/1).

    v_reverse_v (G,) increasing, forward negative; dneff (Bt, G) relative to 0 V; alpha_fca_per_m (Bt, G) absolute;
    capacitance_f_per_m (Bt, G); r_series_ohm_m (Bt, G). Bt = 1 for one cross-section, or one row per world.
    """

    v_reverse_v: np.ndarray
    dneff: np.ndarray
    alpha_fca_per_m: np.ndarray
    capacitance_f_per_m: np.ndarray
    r_series_ohm_m: np.ndarray
    method: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        g = np.ascontiguousarray(np.asarray(self.v_reverse_v, dtype=np.float64))
        if g.ndim != 1 or g.size < 2 or np.any(np.diff(g) <= 0):
            raise ValueError("v_reverse_v must be a strictly increasing 1-D grid")
        self.v_reverse_v = g

        def col(x, name):
            a = np.asarray(x, dtype=np.float64)
            if a.ndim == 0:
                a = np.full(g.shape, float(a))
            a = np.atleast_2d(a)
            if a.shape[1] != g.size:
                raise ValueError(f"{name}: last axis {a.shape[1]} != grid size {g.size}")
            return np.ascontiguousarray(a)

        self.dneff = col(self.dneff, "dneff")
        self.alpha_fca_per_m = col(self.alpha_fca_per_m, "alpha_fca_per_m")
        self.capacitance_f_per_m = col(self.capacitance_f_per_m, "capacitance_f_per_m")
        self.r_series_ohm_m = col(self.r_series_ohm_m, "r_series_ohm_m")
        for name in ("dneff", "alpha_fca_per_m", "capacitance_f_per_m", "r_series_ohm_m"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name}: non-finite values")
        if np.any(self.alpha_fca_per_m < 0) or np.any(self.capacitance_f_per_m <= 0) or np.any(self.r_series_ohm_m < 0):
            raise ValueError("need alpha_fca_per_m >= 0 (absolute), capacitance_f_per_m > 0, r_series_ohm_m >= 0")
        if g[0] <= 0.0 <= g[-1]:
            # ARCHITECTURE 5.4: dneff is relative to 0 V. An absolute n_eff (~2.5) or an offset table fails here.
            d0 = np.array([np.interp(0.0, g, row) for row in self.dneff])
            scale = np.max(np.abs(self.dneff), axis=1)
            if np.any(np.abs(d0) > 1e-2 * scale + 1e-12):
                raise ValueError(f"dneff must be relative to 0 V (ARCHITECTURE 5.4); dneff(0 V) = {d0}")

    def eval(self, v, fill, length_m):
        """NumPy convenience over junction_eval: returns dict of ring-level dneff, alpha_per_m, c_f, r_s_ohm, flags
        for v (B, N) (or broadcastable), fill and length_m broadcastable to it."""
        v2 = np.atleast_2d(np.asarray(v, dtype=np.float64))
        B, N = v2.shape
        outs = [np.zeros((B, N)) for _ in range(4)]
        flags = np.zeros((B, N), np.int8)
        junction_eval(np.ascontiguousarray(v2), self.v_reverse_v, self.dneff, self.alpha_fca_per_m,
                      self.capacitance_f_per_m, self.r_series_ohm_m, _f2(fill, B, N, "fill"),
                      _f2(length_m, B, N, "length_m"), outs[0], outs[1], outs[2], outs[3], flags)
        return {"dneff": outs[0], "alpha_per_m": outs[1], "c_f": outs[2], "r_s_ohm": outs[3], "flags": flags}


def load_junction_table(path) -> JunctionTable:
    """Read a junction artefact ``junction_<xsec>.json`` (ARCHITECTURE 5.4, schema ``junction/1``).

    Expected keys (top level, or under "table"): v_reverse_v, dneff, alpha_fca_per_m, capacitance_f_per_m and
    r_series_ohm_m (ohm*m for a full circumference, scalar or per grid point; R_s = r'/(f L) at runtime).
    By the unit-suffix grammar (07 P1-15) a key ``r_series_ohm`` is in ohm, not ohm*m: it is accepted only for the
    full-circumference junction together with ``junction_length_m`` (the full circumference it refers to), and is
    converted as r' = R_s,full * L. The grid should span -0.5 .. 4.0 V (ARCHITECTURE 5.4); a narrower grid gives a
    warning (lookups outside it are end-held and flagged). The sha256 against the sidecar is verified by
    mrm.params.realize, not here.
    """
    d = json.loads(Path(path).read_text())
    schema = d.get("schema")
    if schema is not None and not str(schema).startswith("junction/"):
        raise ValueError(f"{path}: schema {schema!r} is not junction/*")
    t = d.get("table", d)
    if "r_series_ohm_m" in t:
        rs = t["r_series_ohm_m"]
    elif "r_series_ohm" in t:
        length = t.get("junction_length_m", d.get("junction_length_m"))
        if length is None:
            raise ValueError(f"{path}: r_series_ohm (ohm) needs junction_length_m to convert to r_series_ohm_m "
                             "(ohm*m); or write r_series_ohm_m directly")
        rs = np.asarray(t["r_series_ohm"], dtype=np.float64) * float(length)
    else:
        raise ValueError(f"{path}: no r_series_ohm_m")
    jt = JunctionTable(t["v_reverse_v"], t["dneff"], t["alpha_fca_per_m"], t["capacitance_f_per_m"], rs,
                       method=str(d.get("method", t.get("method", "unknown"))), meta={"path": str(path)})
    if jt.v_reverse_v[0] > JUNCTION_GRID_V_MIN + 1e-9 or jt.v_reverse_v[-1] < JUNCTION_GRID_V_MAX - 1e-9:
        warnings.warn(f"{path}: junction grid {jt.v_reverse_v[0]}..{jt.v_reverse_v[-1]} V does not span "
                      f"{JUNCTION_GRID_V_MIN}..{JUNCTION_GRID_V_MAX} V (ARCHITECTURE 5.4)", stacklevel=2)
    return jt


@njit(cache=True)
def depletion_shift_m(v, shift_amp_m, v_bi, m):
    """Analytic depletion law for the resonance shift (m), 07 P1-7 / 06 M2 / 03 section 5.3:

        dlambda(V) = A * ((V + V_bi)^m - V_bi^m),   V reverse (forward negative), red shift positive.

    A (m / V^m), V_bi (V). The contract states m in [1/3, 1/2]. Note (docs/handoff/plant_electrical.md): the
    charge-control derivation of 03 section 5.3 gives the exponent 1 - m_C (m_C the C-V grading coefficient, 1/2
    abrupt, 1/3 linearly graded), i.e. 1/2 to 2/3; m = 1/2 is where both readings agree. V + V_bi is floored at
    1e-6 V (forward conduction is outside a depletion model).
    """
    x = v + v_bi
    if x < 1e-6:
        x = 1e-6
    return shift_amp_m * (x ** m - v_bi ** m)


def analytic_junction_table(*, shift_amp_m: float, v_bi_v: float, m: float = 0.5, lambda_m: float, n_g: float,
                            c0_f_per_m: float, alpha0_per_m: float, loss_ratio_q: float,
                            r_series_ohm_m: float, m_cap: float | None = None,
                            v_grid=None) -> JunctionTable:
    """Analytic fallback table (method 'analytic') on the junction grid, for a full-circumference junction.

    dneff(V)   = (n_g/lambda) * dlambda(V)       with dlambda from depletion_shift_m (Delta lambda = lambda/n_g *
                                                  dneff, full circumference; 03 section 5.2)
    C'(V)      = c0' * (1 + V/V_bi)^(-m_cap),     m_cap = 1 - m by default, so that dlambda/dV is proportional to
                                                  C(V) (charge control, 03 section 5.3)
    alpha(V)   = alpha0 - (4 pi q/lambda) * dneff(V), floored at 0  (loss ratio q = lambda dalpha/(4 pi dn),
                                                  03 section 5.4, q ~ 0.06 derived from Nedeljkovic 2011)
    r_s'(V)    = r_series_ohm_m (constant)
    shift_amp_m: A for the FULL circumference (a ring fit gives A_fit/fill). Units SI.
    """
    if not (1.0 / 3.0 - 1e-12 <= m <= 2.0 / 3.0 + 1e-12):
        raise ValueError("m must be in [1/3, 2/3] (contract [1/3, 1/2]; charge control [1/2, 2/3])")
    # 451 points = 10 mV spacing (design choice: linear-interpolation error of the sqrt law below 1e-4 relative)
    g = (np.linspace(JUNCTION_GRID_V_MIN, JUNCTION_GRID_V_MAX, 451) if v_grid is None
         else np.asarray(v_grid, dtype=np.float64))
    if np.any(g + v_bi_v <= 0.0):
        raise ValueError("grid reaches V <= -V_bi (forward conduction)")
    mc = 1.0 - m if m_cap is None else m_cap
    dl = shift_amp_m * ((g + v_bi_v) ** m - v_bi_v ** m)
    dneff = n_g / lambda_m * dl
    cap = c0_f_per_m * (1.0 + g / v_bi_v) ** (-mc)
    alpha = np.maximum(alpha0_per_m - 4.0 * np.pi * loss_ratio_q / lambda_m * dneff, 0.0)
    return JunctionTable(g, dneff, alpha, cap, np.full(g.shape, float(r_series_ohm_m)), method="analytic",
                         meta={"shift_amp_m": shift_amp_m, "v_bi_v": v_bi_v, "m": m, "m_cap": mc,
                               "lambda_m": lambda_m, "n_g": n_g, "loss_ratio_q": loss_ratio_q})


def analytic_junction_table_from_ring(*, eff_at_0v_m_per_v: float, v_bi_v: float, m: float, fill_fraction: float,
                                      ring_length_m: float, lambda_m: float, n_g: float, capacitance_at_0v_f: float,
                                      series_resistance_ohm: float, alpha_fca_ring_per_m: float,
                                      loss_ratio_q: float | None = None, dqi_dv_frac_per_v: float | None = None,
                                      alpha_i_total_per_m: float | None = None, m_cap: float | None = None,
                                      v_grid=None) -> JunctionTable:
    """Analytic table from the ring-level schema keys (fidelity.junction_model = analytic; schema ring.junction.*).

    The runtime scales the table by fill f and length L (junction_eval_row), so the full-circumference values are
        A_full   = eff0 * V_bi^(1 - m) / (m f)       from eff0 = d(dlambda)/dV at 0 V = A m V_bi^(m - 1) (ring)
        c0'      = C_ring(0)/(f L),   r_s' = R_s,ring * f L,   alpha0_full = alpha_fca,ring(0 V)/f
    and junction_eval_row reproduces eff0, C_ring(0), R_s,ring and alpha_fca,ring at 0 V exactly.
    Loss modulation: give loss_ratio_q (03 section 5.4: 0.06 derived, range 0.03..0.15), or dqi_dv_frac_per_v with
    the ring's total intrinsic loss alpha_i_total_per_m, converted by dQi/Qi/dV = -d(alpha_ring)/dV / alpha_i and
    d(alpha_ring)/dV = -(4 pi q / lambda) (n_g/lambda) eff0  ->  q = dqi * alpha_i * lambda^2 / (4 pi n_g eff0).
    Units SI (eff0 in m/V, C in F, alpha in 1/m).
    """
    if not 0.0 < fill_fraction <= 1.0:
        raise ValueError("fill_fraction must be in (0, 1]")
    if loss_ratio_q is None:
        if dqi_dv_frac_per_v is None or alpha_i_total_per_m is None:
            raise ValueError("give loss_ratio_q, or dqi_dv_frac_per_v together with alpha_i_total_per_m")
        loss_ratio_q = dqi_dv_frac_per_v * alpha_i_total_per_m * lambda_m ** 2 / (4 * np.pi * n_g * eff_at_0v_m_per_v)
    fl = fill_fraction * ring_length_m
    return analytic_junction_table(
        shift_amp_m=eff_at_0v_m_per_v * v_bi_v ** (1.0 - m) / (m * fill_fraction), v_bi_v=v_bi_v, m=m,
        lambda_m=lambda_m, n_g=n_g, c0_f_per_m=capacitance_at_0v_f / fl,
        alpha0_per_m=alpha_fca_ring_per_m / fill_fraction, loss_ratio_q=float(loss_ratio_q),
        r_series_ohm_m=series_resistance_ohm * fl, m_cap=m_cap, v_grid=v_grid)


def junction_table_from_plant_params(pp, *, rtol: float = 1e-9) -> JunctionTable:
    """The junction table the runtime uses (junction_eval_row), built from PlantParams.junction and optics.

    LUT mode (fidelity.junction_model = lut): the artefact table of the cross-section, one row (Bt = 1).
    Analytic mode: one full-circumference table per world (Bt = B) from :func:`analytic_junction_table_from_ring`
    with the world's ring values. The table holds the carrier-induced index change dneff(V) of the junction, which
    does not depend on the ring's wavelength; eff_at_0v (a resonance shift per volt, measured on G25) is converted
    with the bank-mean lambda_res_cold / n_g, so a ring at another line shifts by (lambda/n_g)_ring /
    (lambda/n_g)_mean x eff_at_0v (about +-0.6 % across a 2.8 THz bank). alpha_fca_per_m (realize: the ring FCA at
    the reference bias bias_ref_v) is moved to its 0 V value through the loss law alpha(V) = alpha(0) -
    (4 pi q/lambda) dneff(V); without a loss budget (alpha_fca nan) the table has no FCA and no loss modulation.
    q is junction.loss_ratio_q, else derived from dqi_dv_frac_per_v with alpha_i, else 0. The per-length capacitance
    c' = C/(f L) and r_s' = R_s f L use the world's mean over rings (the ring length L varies slightly with the sampled
    n_g at a fixed FSR, so ring k then gets C = c' f L_k, proportional to its length, which is the physics of a
    junction along the ring). Raises if eff_at_0v / fill or fill differ between the rings of a world beyond ``rtol``,
    or c', r_s' by more than 1 %: a per-world table cannot hold per-ring junction physics (no preset varies them).
    """
    j, o = pp.junction, pp.optics
    if int(j.model) == 1:
        return JunctionTable(np.asarray(j.lut_v_reverse_v, float), np.asarray(j.lut_dneff, float)[None],
                             np.asarray(j.lut_alpha_fca_per_m, float)[None],
                             np.asarray(j.lut_capacitance_f_per_m, float)[None],
                             np.asarray(j.lut_r_series_ohm_m, float)[None], method="lut")
    B = int(pp.batch)

    def uniform(name, x, tol=rtol):
        x = np.asarray(x, dtype=float)
        x2 = x.reshape(x.shape[0], -1)
        ref = np.mean(x2, axis=1, keepdims=True)
        if np.any(np.abs(x2 - ref) > tol * np.maximum(np.abs(ref), 1e-300)):
            raise ValueError(f"junction_table_from_plant_params: {name} differs between the rings of a world; the "
                             f"per-world junction table cannot represent it")
        return ref[:, 0]
    fill = uniform("fill_fraction", j.fill_fraction)
    eff_per_fill = uniform("eff_at_0v / fill", np.asarray(j.eff_at_0v_m_per_v) / np.asarray(j.fill_fraction))
    length = np.asarray(o.length_m, float)
    c_per_len = uniform("capacitance / (fill length)", np.asarray(j.capacitance_at_0v_f) / (j.fill_fraction * length),
                        1e-2)
    rs_len = uniform("series resistance * fill length", np.asarray(j.series_resistance_ohm) * j.fill_fraction * length,
                     1e-2)
    vbi = np.asarray(j.v_bi_v, float).reshape(-1)
    m = np.asarray(j.depletion_exponent, float).reshape(-1)
    v_ref = np.asarray(o.bias_ref_v, float).reshape(-1)
    lam = np.mean(np.asarray(o.lambda_res_cold_m, float), axis=1)
    ng = np.mean(np.asarray(o.n_g, float), axis=1)
    a_fca = np.asarray(o.alpha_fca_per_m, float)
    q_given = np.asarray(j.loss_ratio_q, float).reshape(-1)
    dqi = np.asarray(j.dqi_dv_frac_per_v, float)
    rows = []
    for b in range(B):
        f, L1 = float(fill[b]), 1.0     # tables are per unit length: c' = C/(f L), r_s' = R_s f L
        a_full = eff_per_fill[b] * vbi[b] ** (1.0 - m[b]) / m[b]          # A_full = eff0 V_bi^(1-m)/(m f)
        if np.isfinite(q_given[b]):
            q = float(q_given[b])
        elif np.any(dqi[b] != 0.0) and np.all(np.isfinite(a_fca[b])):
            eff0 = eff_per_fill[b] * f
            q = float(np.mean(dqi[b]) * np.mean(np.asarray(o.alpha_i_per_m)[b]) * lam[b] ** 2
                      / (4.0 * np.pi * ng[b] * eff0))
        else:
            q = 0.0
        alpha_ref = float(np.mean(a_fca[b])) if np.all(np.isfinite(a_fca[b])) else 0.0
        # alpha_full(0) from the ring FCA at the reference bias: alpha(V) = alpha(0) - (4 pi q/lambda) dneff(V)
        dneff_ref = ng[b] / lam[b] * a_full * ((v_ref[b] + vbi[b]) ** m[b] - vbi[b] ** m[b])
        alpha0_full = alpha_ref / f + 4.0 * np.pi * q / lam[b] * dneff_ref
        rows.append(analytic_junction_table(
            shift_amp_m=float(a_full), v_bi_v=float(vbi[b]), m=float(m[b]), lambda_m=float(lam[b]), n_g=float(ng[b]),
            c0_f_per_m=float(c_per_len[b]) * L1, alpha0_per_m=float(alpha0_full), loss_ratio_q=q,
            r_series_ohm_m=float(rs_len[b])))
    g = rows[0].v_reverse_v
    return JunctionTable(g, np.stack([r.dneff[0] for r in rows]), np.stack([r.alpha_fca_per_m[0] for r in rows]),
                         np.stack([r.capacitance_f_per_m[0] for r in rows]),
                         np.stack([r.r_series_ohm_m[0] for r in rows]), method="analytic",
                         meta={"rows": "per world", "source": "PlantParams.junction (analytic)"})


def junction_efficiency_m_per_v(table: JunctionTable, v, fill, lambda_m: float, n_g: float, row: int = 0,
                                dv: float = 1e-3):
    """Report-only derived efficiency dlambda/dV (m/V) at reverse voltage v: (lambda/n_g)*fill*d(dneff)/dV
    (07 P1-7: eff_at_0v_pm_per_v becomes derived in LUT mode). Central difference over +-dv."""
    v = np.asarray(v, dtype=np.float64)
    hi = np.interp(v + dv, table.v_reverse_v, table.dneff[row])
    lo = np.interp(v - dv, table.v_reverse_v, table.dneff[row])
    return lambda_m / n_g * fill * (hi - lo) / (2 * dv)


def junction_dqi_dv_frac_per_v(table: JunctionTable, v, fill, alpha_i_total_per_m, row: int = 0,
                               dv: float = 1e-3):
    """Report-only derived relative Q_i change per volt (1/V) at reverse voltage v (07 P1-7: dqi_dv_frac_per_v
    becomes derived in LUT mode): dQ_i/Q_i/dV = -fill * d(alpha_fca)/dV / alpha_i, since Q_i is proportional to
    1/alpha_i (03 section 5.4). alpha_i_total_per_m is the ring's total intrinsic loss at v (1/m). Central
    difference over +-dv."""
    v = np.asarray(v, dtype=np.float64)
    hi = np.interp(v + dv, table.v_reverse_v, table.alpha_fca_per_m[row])
    lo = np.interp(v - dv, table.v_reverse_v, table.alpha_fca_per_m[row])
    return -fill * (hi - lo) / (2 * dv) / np.asarray(alpha_i_total_per_m, dtype=np.float64)


# ------------------------------------------------------------------------------------------ static helpers
def heater_steady_state(code=None, *, law="voltage", drive_frac=None, bits: int = 16, pwm_bits: int = PAT2_PWM_BITS,
                        v_full_scale_v=np.inf, i_full_scale_a=np.inf, p_full_scale_w=np.inf, compliance_v=np.inf,
                        r_cold_ohm, tcr_per_k, r_th_heater_k_per_w, t_base_minus_ref_k=0.0,
                        bit_err_lsb=None, inl_bow_lsb=0.0) -> dict:
    """Static heater operating point with the TCR feedback solved exactly (vectorized over code or drive_frac).

    Steady state of the heater-resistor observer: T_H - T0 = t_base_minus_ref_k + R_th,H*P. This is the same
    implicit solve as heater_power_row with g_self = R_th,H. For voltage drive and t_base = 0 it is the textbook
    law P(1 + yP) = V^2/R0 with y = alpha*R_th,H (Ch11 eq. 11.15; REF.heaterSelfHeatPerMw = 0.00904 /mW).
    Returns dict p_w, v_v, i_a, r_ohm, u, flags (arrays shaped like the input).
    """
    law_c = DRIVER_LAWS[law] if isinstance(law, str) else int(law)
    _check_full_scales(np.array([law_c]), 1, 1, np.atleast_2d(np.asarray(v_full_scale_v, float)),
                       np.atleast_2d(np.asarray(i_full_scale_a, float)),
                       np.atleast_2d(np.asarray(p_full_scale_w, float)))
    if drive_frac is None:
        c = np.asarray(code, dtype=np.int64)
        shape = c.shape
        flat = np.ascontiguousarray(c.reshape(-1))
        u = np.empty(flat.size)
        rip = np.empty(flat.size)
        nd = bits - pwm_bits if law_c == LAW_DAC_PWM else bits
        be = _bit_err(bit_err_lsb, 1, 1, nd, "bit_err_lsb")[0]
        heater_target_row(flat, law_c, bits, pwm_bits, be, np.atleast_1d(np.asarray(inl_bow_lsb, float)), 0, 0, 1,
                          u, rip)
    else:
        u = np.ascontiguousarray(np.asarray(drive_frac, dtype=np.float64).reshape(-1))
        shape = np.shape(drive_frac)
    M = u.size

    def row(x):
        return np.ascontiguousarray(np.broadcast_to(np.asarray(x, dtype=np.float64), (M,)))

    outs = [np.zeros(M) for _ in range(4)]
    flags = np.zeros(M, np.int8)
    heater_power_row(u, law_c, row(t_base_minus_ref_k), row(r_th_heater_k_per_w), row(r_cold_ohm), row(tcr_per_k),
                     np.zeros(1), row(v_full_scale_v), row(i_full_scale_a), row(p_full_scale_w), row(compliance_v),
                     outs[0], outs[1], outs[2], outs[3], flags)
    return {"p_w": outs[0].reshape(shape), "v_v": outs[1].reshape(shape), "i_a": outs[2].reshape(shape),
            "r_ohm": outs[3].reshape(shape), "u": u.reshape(shape), "flags": flags.reshape(shape)}


def heater_gain_w_per_code(code, h: int = 1, **kw):
    """Static incremental heater gain dP/dcode (W/code) including the TCR feedback, by a central difference of
    heater_steady_state over +-h codes. The gain-scheduling quantity of 07 P1-11 (P ~ code^2 for voltage/current,
    flat for power)."""
    c = np.asarray(code, dtype=np.int64)
    bits = kw.get("bits", 16)
    hi = np.minimum(c + h, (1 << bits) - 1)
    lo = np.maximum(c - h, 0)
    return (heater_steady_state(hi, **kw)["p_w"] - heater_steady_state(lo, **kw)["p_w"]) / (hi - lo)


def electrothermal_loop_gain(law, p_w, tcr_per_k, r_th_heater_k_per_w, t_heater_minus_ref_k=None):
    """Electrothermal loop gain of the heater, L = (dP/dT_H) * R_th,H (07 P2-5; 06 E1, N8).

        voltage, dac_pwm: L = -alpha R_th,H P / (1 + alpha (T_H - T0))   (negative feedback)
        current:          L = +alpha R_th,H P / (1 + alpha (T_H - T0))   (positive feedback)
        power:            L = 0                                           (regulated power)
    T_H - T0 defaults to R_th,H*P (heater self-heating only). The static small-signal heater gain with the TCR
    loop closed is 1/(1 - L) times the gain at frozen R (closed_loop_gain_factor). For current drive this equals
    1 + alpha R_th,H P exactly; 07 P2-5's "about 29 %" at P*TCR*R_th,H ~ 0.23 is the approximation 1/(1 - 0.23);
    the exact value at y = 0.00904 /mW and 25.5 mW is +23 %. Units: p_w W, tcr_per_k 1/K, R_th,H K/W.
    """
    law_c = DRIVER_LAWS[law] if isinstance(law, str) else int(law)
    p = np.asarray(p_w, dtype=np.float64)
    a = np.asarray(tcr_per_k, dtype=np.float64)
    rth = np.asarray(r_th_heater_k_per_w, dtype=np.float64)
    dth = rth * p if t_heater_minus_ref_k is None else np.asarray(t_heater_minus_ref_k, dtype=np.float64)
    mag = a * rth * p / (1.0 + a * dth)
    if law_c == LAW_CURRENT:
        return mag
    if law_c == LAW_POWER:
        return np.zeros_like(mag)
    return -mag


def closed_loop_gain_factor(loop_gain):
    """Static gain multiplier 1/(1 - L) of the electrothermal loop (07 P2-5)."""
    return 1.0 / (1.0 - np.asarray(loop_gain, dtype=np.float64))


def compliance_headroom_v(i_a, r_cold_ohm, tcr_per_k, t_heater_minus_ref_k, compliance_v):
    """Current-drive compliance headroom V_comp - I*R0*(1 + alpha (T_H - T0)) in V (07 P1-10 driver_compliance_v).
    Negative means the driver is in compliance (the current source saturates)."""
    r = np.asarray(r_cold_ohm, float) * (1.0 + np.asarray(tcr_per_k, float) * np.asarray(t_heater_minus_ref_k, float))
    return np.asarray(compliance_v, float) - np.asarray(i_a, float) * r


def pwm_ripple_pp_k(delta_p_w, duty, period_s, residues_k_per_w, taus_s):
    """Steady-state peak-to-peak temperature ripple of a PWM heater (K), for a Foster network sum_k r_k/(1 + s tau_k).

    A square wave of power step delta_p (W), duty D and period T drives each first-order section to the periodic
    ripple r_k*dP*(1 - e^(-DT/tau))(1 - e^(-(1-D)T/tau))/(1 - e^(-T/tau)); every section peaks at the falling edge
    and bottoms at the rising edge, so the peak-to-peak values add (for positive residues). PAT-2 requires the
    thermal time constant to exceed the 2.56 us period; this quantifies what is left when it barely does.
    """
    r = np.atleast_1d(np.asarray(residues_k_per_w, float))
    tau = np.atleast_1d(np.asarray(taus_s, float))
    D = float(duty)
    T = float(period_s)
    pp = r * (-np.expm1(-D * T / tau)) * (-np.expm1(-(1 - D) * T / tau)) / (-np.expm1(-T / tau))
    return float(delta_p_w) * float(np.sum(pp))


def dac_bit_errors_from_spec(dnl_sigma_lsb: float, n_bits: int, z) -> np.ndarray:
    """Bit-weight errors (LSB) of a binary-weighted DAC with unit-element mismatch, from a DNL spec.

    Bit i is 2^i unit elements with independent mismatch sigma_u, so e_i = z_i * sigma_u * sqrt(2^i). The DNL at
    the MSB carry is e_{n-1} - sum_{i<n-1} e_i with variance sigma_u^2 (2^n - 1), so sigma_u =
    dnl_sigma_lsb / sqrt(2^n - 1) makes dnl_sigma_lsb the 1-sigma mid-scale DNL (07 P1-10 dac_dnl_lsb).
    z: standard-normal draws, shape (..., n_bits) (from the counter RNG, stream 'variation').
    """
    z = np.asarray(z, dtype=np.float64)
    if z.shape[-1] != n_bits:
        raise ValueError("z must end in an axis of length n_bits")
    su = dnl_sigma_lsb / math.sqrt((1 << n_bits) - 1)
    return z * su * np.sqrt(2.0 ** np.arange(n_bits))


def dac_transfer_curve(n_bits: int, bit_err_lsb=None, inl_bow_lsb: float = 0.0):
    """Full DAC transfer for reports: returns (frac (2^n,), inl_lsb (2^n,), dnl_lsb (2^n - 1,)), endpoint INL."""
    codes = np.arange(1 << n_bits, dtype=np.int64)
    er = np.zeros(n_bits) if bit_err_lsb is None else np.asarray(bit_err_lsb, float)
    frac = np.array([dac_frac(int(c), n_bits, er, float(inl_bow_lsb)) for c in codes]) if n_bits <= 12 else None
    if frac is None:  # vectorized path for wide DACs
        w = 2.0 ** np.arange(n_bits) + er
        bitsm = ((codes[:, None] >> np.arange(n_bits)[None, :]) & 1).astype(np.float64)
        x = codes / ((1 << n_bits) - 1)
        frac = (bitsm @ w + inl_bow_lsb * 4 * x * (1 - x)) / w.sum()
    cmax = (1 << n_bits) - 1
    y = frac * cmax
    inl = y - codes
    dnl = np.diff(y) - 1.0
    return frac, inl, dnl


def data_activity_heat_w(c_eff_f, swing_vpp, bitrate_hz, transition_density, r_s_ohm, r_drv_ohm,
                         p_static_w=0.0, e_tr_j=0.0, enable=1.0, add_load_share=False):
    """NumPy convenience for data_activity_heat_row (07 P1-8). Returns (P_j, P_d) in W, broadcast shapes."""
    arrs = np.broadcast_arrays(*[np.asarray(x, dtype=np.float64) for x in
                                 (c_eff_f, swing_vpp, transition_density, enable, r_s_ohm, r_drv_ohm, p_static_w,
                                  e_tr_j)])
    shape = arrs[0].shape
    flat = [np.ascontiguousarray(a.reshape(-1)) for a in arrs]
    pj = np.zeros(flat[0].size)
    pd = np.zeros(flat[0].size)
    data_activity_heat_row(flat[0], flat[1], float(bitrate_hz), flat[2], flat[3], flat[4], flat[5], flat[6], flat[7],
                           int(bool(add_load_share)), pj, pd)
    return pj.reshape(shape), pd.reshape(shape)
