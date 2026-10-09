"""Verification tests for mrm/plant/electrical.py (07 section 4.1 class: verification; checks code against hand
calculations and analytic limits, not against hardware).

Covers 07 P1-11 (driver laws, dac_pwm), P0-6 (FIFO -> slew -> settle order, applied_codes), P1-10 (DAC INL/DNL,
compliance), P1-12 (supply limit), P2-5 (electrothermal gain), P1-7 (junction runtime), P1-8 (data-activity heat).
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from mrm.plant import electrical as el

DT = 1e-6  # s, default plant tick (ARCHITECTURE section 3)

# Textbook G25 heater numbers (READ-ONLY textbook src/physics/index.ts, REF):
Y_PER_W = 0.00904e3        # derived (fitted) in the textbook: REF.heaterSelfHeatPerMw = 0.00904 /mW
R_HOT_OHM = 55.9           # inferred, textbook REF.heaterOhm (wafer-map median at the FSR drive)
V_FSR_V = 1.195            # not re-checked (talk), textbook REF.heaterVFsr
TCR_PER_K = 1e-3           # assumed, textbook REF.heaterTcrPerK
R_COLD_OHM = R_HOT_OHM / (1 + Y_PER_W * V_FSR_V ** 2 / R_HOT_OHM)   # textbook HEATER_OHM_COLD (~45.4 ohm)
RTH_H = Y_PER_W / TCR_PER_K                                          # K/W, R_th,H = y/alpha (~9.04 K/mW)
# read-off, G25 heater V -> resonance (textbook HEATER_V_LAMBDA, OFC 2025 talk, +-5 pm)
G25_V_LAMBDA = np.array([
    [0.0, 1301.9], [0.1, 1301.98], [0.2, 1302.21], [0.3, 1302.58], [0.4, 1303.06], [0.5, 1303.7],
    [0.6, 1304.45], [0.7, 1305.31], [0.8, 1306.27], [0.9, 1307.3], [1.0, 1308.43], [1.1, 1309.6],
    [1.2, 1310.84], [1.3, 1312.15], [1.4, 1313.49], [1.5, 1314.9]])


def _heater(B=1, N=1, **kw):
    base = dict(law="voltage", v_full_scale_v=1.6, r_cold_ohm=R_COLD_OHM, tcr_per_k=TCR_PER_K, tcr_ref_k=298.15)
    base.update(kw)
    return el.HeaterDriverParams.build(B, N, DT, **base)


def _power_row(u, law, t_pre=298.15, g=0.0, r0=50.0, alpha=1e-3, t0=298.15, v_fs=1.6, i_fs=0.03, p_fs=0.03,
               v_comp=np.inf):
    u = np.atleast_1d(np.asarray(u, float))
    n = u.size
    a = lambda x: np.atleast_1d(np.asarray(x, float))
    out = [np.zeros(n) for _ in range(4)]
    fl = np.zeros(n, np.int8)
    el.heater_power_row(u, law, a(t_pre), a(g), a(r0), a(alpha), a(t0), a(v_fs), a(i_fs), a(p_fs), a(v_comp),
                        out[0], out[1], out[2], out[3], fl)
    return out[0], out[1], out[2], out[3], fl


# ============================================================================================ driver laws
def test_voltage_law_hand():
    # explicit (g = 0): P = (u V_FS)^2 / (R0 (1 + alpha (T - T0)))
    p, v, i, r, fl = _power_row([0.5, 1.0], el.LAW_VOLTAGE, t_pre=298.15 + 50.0)
    r_hand = 50.0 * (1 + 1e-3 * 50.0)
    np.testing.assert_allclose(r, r_hand, rtol=1e-14)
    np.testing.assert_allclose(p, (np.array([0.8, 1.6]) ** 2) / r_hand, rtol=1e-14)
    np.testing.assert_allclose(i, np.array([0.8, 1.6]) / r_hand, rtol=1e-14)
    assert not fl.any()
    # implicit: R = r_o + k P, k = R0 alpha g -> k P^2 + r_o P - V^2 = 0 (positive root)
    g = 9000.0
    p, v, i, r, fl = _power_row([0.75], el.LAW_VOLTAGE, g=g)
    k = 50.0 * 1e-3 * g
    p_hand = (-50.0 + math.sqrt(50.0 ** 2 + 4 * k * 1.2 ** 2)) / (2 * k)
    assert p[0] == pytest.approx(p_hand, rel=1e-13)
    assert r[0] == pytest.approx(50.0 + k * p_hand, rel=1e-13)
    assert p[0] * r[0] == pytest.approx(1.2 ** 2, rel=1e-13)


def test_current_law_hand():
    p, v, i, r, fl = _power_row([0.5], el.LAW_CURRENT, t_pre=298.15 + 20.0)
    r_hand = 50.0 * 1.02
    assert p[0] == pytest.approx(0.015 ** 2 * r_hand, rel=1e-14)
    assert v[0] == pytest.approx(0.015 * r_hand, rel=1e-14)
    # implicit: P = I^2 r_o/(1 - k I^2)
    g = 9000.0
    k = 50.0 * 1e-3 * g
    p, v, i, r, fl = _power_row([1.0], el.LAW_CURRENT, g=g)
    assert p[0] == pytest.approx(0.03 ** 2 * 50.0 / (1 - k * 0.03 ** 2), rel=1e-13)
    assert p[0] == pytest.approx(0.03 ** 2 * r[0], rel=1e-13)


def test_power_law_hand():
    p, v, i, r, fl = _power_row([0.25, 1.0], el.LAW_POWER, t_pre=298.15 + 10.0, g=5000.0)
    np.testing.assert_allclose(p, [0.0075, 0.03], rtol=1e-14)
    r_hand = 50.0 * 1.01 + 50.0 * 1e-3 * 5000.0 * p
    np.testing.assert_allclose(r, r_hand, rtol=1e-14)
    np.testing.assert_allclose(v, np.sqrt(p * r_hand), rtol=1e-14)


def test_dac_pwm_law_hand():
    # code = 256 N + d: P = ((1 - f)(N/255)^2 + f ((N+1)/255)^2) V_FS^2 / R (PAT-2 period average)
    prm = _heater(law="dac_pwm", latency_s=0.0)
    codes = np.array([0, 255, 256, 128 * 256 + 64, 200 * 256 + 255, 255 * 256, 65535], np.int64)
    ss = el.heater_steady_state(codes, law="dac_pwm", v_full_scale_v=1.6, r_cold_ohm=50.0, tcr_per_k=1e-3,
                                r_th_heater_k_per_w=0.0)
    nn = codes >> 8
    f = (codes & 255) / 256.0
    n1 = np.minimum(nn + 1, 255)
    u = (1 - f) * (nn / 255) ** 2 + f * (n1 / 255) ** 2
    np.testing.assert_allclose(ss["u"], u, rtol=1e-14, atol=1e-18)
    np.testing.assert_allclose(ss["p_w"], u * 1.6 ** 2 / 50.0, rtol=1e-13, atol=1e-18)
    assert ss["p_w"][-1] == pytest.approx(1.6 ** 2 / 50.0, rel=1e-14)   # top codes saturate at full scale
    # the bank gives the same power with no ripple, one tick after a latency-0 latch
    bank = el.HeaterBank(_heater(N=codes.size, law="dac_pwm", r_cold_ohm=50.0), 0)
    out = bank.step(0, cmd=codes[None, :])
    np.testing.assert_allclose(out.p_w[0], u * 1.6 ** 2 / 50.0, rtol=1e-13, atol=1e-18)
    assert prm.fifo_depth == 1


def test_per_world_laws_broadcast():
    prm = _heater(B=4, N=2, law=["voltage", "current", "power", "dac_pwm"], i_full_scale_a=0.03,
                  p_full_scale_w=0.03, r_cold_ohm=50.0)
    bank = el.HeaterBank(prm, 0)
    code = 32768
    out = bank.step(0, cmd=np.full((4, 2), code))
    u = code / 65535
    r = 50.0
    np.testing.assert_allclose(out.p_w[0], (u * 1.6) ** 2 / r, rtol=1e-13)
    np.testing.assert_allclose(out.p_w[1], (u * 0.03) ** 2 * r, rtol=1e-13)
    np.testing.assert_allclose(out.p_w[2], u * 0.03, rtol=1e-13)
    nn, f = code >> 8, (code & 255) / 256
    np.testing.assert_allclose(out.p_w[3], ((1 - f) * (nn / 255) ** 2 + f * ((nn + 1) / 255) ** 2) * 1.6 ** 2 / r,
                               rtol=1e-13)


def test_gain_ratio_code_squared():
    # 07 P1-11: in voltage drive P ~ code^2, so dP/dcode varies 49300/19500 = 2.53x across the G26 code range
    kw = dict(law="voltage", v_full_scale_v=1.6, r_cold_ohm=R_COLD_OHM, tcr_per_k=0.0, r_th_heater_k_per_w=RTH_H)
    g = el.heater_gain_w_per_code(np.array([19500, 49300]), **kw)
    assert g[1] / g[0] == pytest.approx(49300 / 19500, rel=1e-6)
    kw["law"], kw["p_full_scale_w"] = "power", 0.05
    g = el.heater_gain_w_per_code(np.array([19500, 49300]), **kw)
    assert g[1] / g[0] == pytest.approx(1.0, rel=1e-9)


# ============================================================================================ dac_pwm period average
def test_dac_pwm_period_average_and_ripple():
    code = 137 * 256 + 77                                   # N = 137, d = 77
    nn, d = 137, 77
    f = d / 256
    x0, x1 = nn / 255, (nn + 1) / 255
    u_avg = (1 - f) * x0 ** 2 + f * x1 ** 2
    # dt = one PWM clock (10 ns): each tick is wholly low or high; 256 ticks = one period
    clk = el.PAT2_PWM_PERIOD_S / 256
    prm = el.HeaterDriverParams.build(1, 1, clk, law="dac_pwm", pwm_ripple=True, v_full_scale_v=1.6,
                                      r_cold_ohm=50.0, tcr_per_k=0.0)
    bank = el.HeaterBank(prm, code)
    us = np.array([bank.step(k).u[0, 0] for k in range(512)])
    assert set(np.round(us, 15)) <= {round(x0 ** 2, 15), round(x1 ** 2, 15)}
    assert np.sum(np.isclose(us[:256], x1 ** 2, rtol=0, atol=1e-15)) == d   # high for exactly d clocks per period
    assert us[:256].mean() == pytest.approx(u_avg, rel=1e-14)
    # dt = 1 us = 100 clocks: the tick averages differ from the period average (ripple), but their mean over
    # lcm(100, 256) = 6400 clocks (64 ticks, 25 periods) is the period average exactly
    prm = _heater(law="dac_pwm", pwm_ripple=True, r_cold_ohm=50.0, tcr_per_k=0.0)
    assert prm.dt_clk == 100
    bank = el.HeaterBank(prm, code)
    us = np.array([bank.step(k).u[0, 0] for k in range(128)])
    assert np.ptp(us) > 0.0
    assert us[:64].mean() == pytest.approx(u_avg, rel=1e-13)
    assert us[64:].mean() == pytest.approx(u_avg, rel=1e-13)
    assert np.all((us >= x0 ** 2 - 1e-15) & (us <= x1 ** 2 + 1e-15))


def test_pwm_ripple_pp_formula_against_simulation():
    # first-order heater pole driven by the PWM waveform at dt = one PWM clock (exact ZOH per clock)
    tau, r_k_per_w = 1.0e-6, 2000.0   # assumed test values (an in-ring heater pole near 1 us)
    clk = el.PAT2_PWM_PERIOD_S / 256
    d = 96
    p_lo, p_hi = 0.010, 0.0102
    a = math.exp(-clk / tau)
    temp = 0.0
    trace = []
    for k in range(256 * 40):
        pw = p_hi if (k % 256) < d else p_lo
        temp = a * temp + (1 - a) * r_k_per_w * pw
        trace.append(temp)
    last = np.array(trace[-256:])
    pp = el.pwm_ripple_pp_k(p_hi - p_lo, d / 256, el.PAT2_PWM_PERIOD_S, [r_k_per_w], [tau])
    assert np.ptp(last) == pytest.approx(pp, rel=1e-9)


# ============================================================================================ P0-6 timing order
@pytest.mark.parametrize("lat", [0, 1, 3, 7])
def test_heater_impulse_after_exact_latency(lat):
    prm = _heater(latency_s=lat * DT, max_latency_s=8 * DT)
    bank = el.HeaterBank(prm, 0)
    t0 = 5
    p = []
    for k in range(20):
        out = bank.step(k, cmd=np.array([[30000 if k == t0 else 0]]))
        p.append(out.p_w[0, 0])
    p = np.array(p)
    nz = np.flatnonzero(p > 0)
    assert nz.tolist() == [t0 + lat]
    assert p[t0 + lat] == pytest.approx(
        el.heater_steady_state(np.array([30000]), law="voltage", v_full_scale_v=1.6, r_cold_ohm=R_COLD_OHM,
                               tcr_per_k=TCR_PER_K, r_th_heater_k_per_w=0.0)["p_w"][0], rel=1e-13)


def test_bias_impulse_after_exact_latency():
    prm = el.BiasDacParams.build(1, 1, DT, bits=12, latency_s=4 * DT)
    c_lo, c_hi = prm.code_for_voltage(1.0)[0, 0], prm.code_for_voltage(2.0)[0, 0]
    bank = el.BiasBank(prm, c_lo)
    v = [bank.step(k, cmd=np.array([[c_hi if k == 2 else c_lo]])).v_avg[0, 0] for k in range(12)]
    v = np.array(v)
    assert np.flatnonzero(np.abs(v - v[0]) > 1e-12).tolist() == [6]


def test_slew_starts_after_latency_and_applied_codes_is_register():
    # FIFO first, then slew: the ramp starts exactly at t0 + latency; applied_codes jumps at the latch
    lat, slew = 3, 2.0e7                                     # ticks; codes/s -> 20 codes per tick
    prm = _heater(latency_s=lat * DT, slew_codes_per_s=slew)
    bank = el.HeaterBank(prm, 1000)
    t0, target = 4, 1500
    reg, u_end = [], []
    for k in range(60):
        out = bank.step(k, cmd=np.array([[target]]) if k == t0 else None)
        reg.append(int(out.applied_codes[0, 0]))
        u_end.append(bank.y_out[0, 0] * 65535)
    reg, u_end = np.array(reg), np.array(u_end)
    assert reg[:t0].tolist() == [1000] * t0 and np.all(reg[t0:] == target)
    assert np.allclose(u_end[:t0 + lat], 1000, atol=1e-9)
    steps = np.diff(u_end)
    ramp = steps[t0 + lat - 1: t0 + lat - 1 + 25]
    np.testing.assert_allclose(ramp, 20.0, rtol=1e-9)        # slew-limited ramp of 20 codes per tick
    assert u_end[t0 + lat - 1 + 25] == pytest.approx(target, abs=1e-9)
    assert np.allclose(u_end[t0 + lat + 25:], target, atol=1e-9)
    # while slewing, the register readback differs from the analog value
    assert reg[t0 + lat + 5] == target and abs(u_end[t0 + lat + 5] - target) > 100


def test_settle_exact_step_and_dt_invariance():
    tau = 0.7e-6
    # step response, no slew: y((k+1) dt) = 1 - e^(-(k+1)dt/tau); tick average known in closed form
    tgt = np.array([1.0])
    ys, yo, avg = np.zeros(1), np.zeros(1), np.zeros(1)
    for k in range(6):
        el.slew_settle_row(tgt, ys, yo, np.array([np.inf]), np.array([tau]), DT, avg)
        assert yo[0] == pytest.approx(1 - math.exp(-(k + 1) * DT / tau), rel=1e-14)
        avg_hand = 1 - tau / DT * (1 - math.exp(-DT / tau)) * math.exp(-k * DT / tau)
        assert avg[0] == pytest.approx(avg_hand, rel=1e-13)
    # slew + settle: stepping at dt and at dt/10 gives the same state at common times (exact discretization)
    rate = np.array([3.3e5])
    a = [np.zeros(1), np.zeros(1)]
    b = [np.zeros(1), np.zeros(1)]
    av_a, av_b = np.zeros(1), np.zeros(1)
    for k in range(12):
        el.slew_settle_row(tgt, a[0], a[1], rate, np.array([tau]), DT, av_a)
        acc = 0.0
        for _ in range(10):
            el.slew_settle_row(tgt, b[0], b[1], rate, np.array([tau]), DT / 10, av_b)
            acc += av_b[0] / 10
        assert a[1][0] == pytest.approx(b[1][0], rel=1e-12, abs=1e-15)
        assert a[0][0] == pytest.approx(b[0][0], rel=1e-12, abs=1e-15)
        assert av_a[0] == pytest.approx(acc, rel=1e-12, abs=1e-15)


def test_quiescence_and_multi_tick_settle_advance():
    tau = 2e-6
    prm = _heater(N=2, latency_s=2 * DT, settle_tau_s=tau)
    bank = el.HeaterBank(prm, 1000)
    tgt = np.full(2, 1000 / 65535)
    assert el.actuator_quiescent_row(bank.reg[0], bank.has_pending[0], bank.fifo[0], bank.y_slew[0], bank.y_out[0], tgt, 1e-12)
    bank.step(0, cmd=np.array([[2000, 2000]]))
    assert not el.actuator_quiescent_row(bank.reg[0], bank.has_pending[0], bank.fifo[0], bank.y_slew[0], bank.y_out[0],
                                         np.full(2, 2000 / 65535), 1e-12)
    for k in range(1, 4):
        bank.step(k)
    # FIFO now uniform, slew target reached; settle still moving: stepping 7 ticks == advancing 7 ticks at once
    y_ref = bank.y_out[0].copy()
    ys = bank.y_slew[0].copy()
    tgt = np.full(2, 2000 / 65535)
    assert el.actuator_quiescent_row(bank.reg[0], bank.has_pending[0], bank.fifo[0], bank.y_slew[0], bank.y_out[0], tgt, 1.0)
    for k in range(4, 11):
        bank.step(k)
    el.settle_advance_row(tgt, ys, y_ref, np.array([tau]), DT, 7)
    np.testing.assert_allclose(y_ref, bank.y_out[0], rtol=1e-13)


def test_dac_max_update_rate_latch():
    prm = _heater(dac_max_update_hz=1.0 / (4 * DT))
    assert prm.min_update_ticks == 4
    bank = el.HeaterBank(prm, 0)
    reg = [int(bank.step(k, cmd=np.array([[100 + k]])).applied_codes[0, 0]) for k in range(10)]
    # writes at ticks 0, 4, 8 with the latest pending command
    assert reg == [100, 100, 100, 100, 104, 104, 104, 104, 108, 108]


def test_command_clamped_to_code_range():
    bank = el.HeaterBank(_heater(), 0)
    assert bank.step(0, cmd=np.array([[70000]])).applied_codes[0, 0] == 65535
    assert bank.step(1, cmd=np.array([[-5]])).applied_codes[0, 0] == 0


# ============================================================================================ TCR and electrothermal
def test_textbook_tcr_droop_reproduced():
    # Ch11 eq. 11.15: P(1 + yP) = V^2/R0, y = alpha R_th,H = 0.00904 /mW, R0 backed out of R_hot = 55.9 ohm
    assert R_COLD_OHM == pytest.approx(45.4, abs=0.05)
    kw = dict(law="voltage", v_full_scale_v=1.6, r_cold_ohm=R_COLD_OHM, tcr_per_k=TCR_PER_K,
              r_th_heater_k_per_w=RTH_H)
    hot = el.heater_steady_state(drive_frac=np.array([V_FSR_V / 1.6]), **kw)
    assert hot["r_ohm"][0] == pytest.approx(R_HOT_OHM, rel=1e-12)         # 55.9 ohm at 1.195 V
    assert hot["p_w"][0] * 1e3 == pytest.approx(25.55, abs=0.01)           # printed 25.45 mW within REF rounding
    v = G25_V_LAMBDA[:, 0]
    p = el.heater_steady_state(drive_frac=v / 1.6, **kw)["p_w"]
    i3, i14 = 3, 14                                                         # 0.3 V and 1.4 V
    droop = 1 - (p[i14] / v[i14] ** 2) / (p[i3] / v[i3] ** 2)
    dl = G25_V_LAMBDA[:, 1] - G25_V_LAMBDA[0, 1]
    droop_meas = 1 - (dl[i14] / v[i14] ** 2) / (dl[i3] / v[i3] ** 2)
    assert 0.20 <= droop <= 0.30                                            # 07 CAL-4: 20 to 30 %
    assert droop == pytest.approx(droop_meas, abs=0.005)                    # textbook: 22 % from 0.3 to 1.4 V
    # one-parameter fit of the whole curve: lambda - lambda0 = eta * P(V)
    eta = np.sum(p * dl) / np.sum(p * p)
    rms_pm = 1e3 * np.sqrt(np.mean((eta * p - dl) ** 2))
    assert eta * 1e-3 == pytest.approx(0.348, abs=0.002)                    # 06 E4: 0.348 nm/mW swept device
    assert rms_pm < 40.0
    p0 = v ** 2 / R_COLD_OHM                                                # without TCR the fit is ~10x worse
    eta0 = np.sum(p0 * dl) / np.sum(p0 * p0)
    assert 1e3 * np.sqrt(np.mean((eta0 * p0 - dl) ** 2)) > 5 * rms_pm


def test_electrothermal_loop_gain_and_closed_loop_factor():
    p = 0.0255
    yp = Y_PER_W * p
    lv = el.electrothermal_loop_gain("voltage", p, TCR_PER_K, RTH_H)
    li = el.electrothermal_loop_gain("current", p, TCR_PER_K, RTH_H)
    assert el.electrothermal_loop_gain("power", p, TCR_PER_K, RTH_H) == 0.0
    assert lv == pytest.approx(-yp / (1 + yp), rel=1e-14)
    assert li == pytest.approx(+yp / (1 + yp), rel=1e-14)
    assert el.closed_loop_gain_factor(li) == pytest.approx(1 + yp, rel=1e-13)   # +23 % at 25.5 mW
    assert 0.22 < yp < 0.24                                                      # 07 P2-5: P*TCR*R_th,H ~ 0.23
    # numerical check: static dP/du with the loop closed over dP/du at frozen R equals 1/(1 - L)
    for law, fs in (("voltage", dict(v_full_scale_v=1.6)), ("current", dict(i_full_scale_a=0.03))):
        kw = dict(law=law, r_cold_ohm=R_COLD_OHM, tcr_per_k=TCR_PER_K, r_th_heater_k_per_w=RTH_H, **fs)
        u0, h = 0.7, 1e-6
        s = el.heater_steady_state(drive_frac=np.array([u0 - h, u0, u0 + h]), **kw)
        dpdu = (s["p_w"][2] - s["p_w"][0]) / (2 * h)
        r_op = s["r_ohm"][1]
        frozen = 2 * u0 * (1.6 ** 2 / r_op if law == "voltage" else 0.03 ** 2 * r_op)
        loop = el.electrothermal_loop_gain(law, s["p_w"][1], TCR_PER_K, RTH_H)
        assert dpdu / frozen == pytest.approx(el.closed_loop_gain_factor(loop), rel=1e-6)


@pytest.mark.parametrize("law", ["voltage", "current"])
def test_explicit_and_implicit_tcr_coupling_reach_static_point(law):
    # one-pole heater-resistor observer, exact ZOH: T(k+1) = a T(k) + (1 - a) R_th,H P(k)
    tau = 0.3e-6                                  # s, assumed fast in-ring pole (stress case: a = e^-3.3)
    a = math.exp(-DT / tau)
    fs = dict(v_full_scale_v=1.6) if law == "voltage" else dict(i_full_scale_a=0.03)
    prm = _heater(law=law, **fs)
    code = 50000
    target = el.heater_steady_state(np.array([code]), law=law, r_cold_ohm=R_COLD_OHM, tcr_per_k=TCR_PER_K,
                                    r_th_heater_k_per_w=RTH_H, **fs)["p_w"][0]
    for implicit in (False, True):
        bank = el.HeaterBank(prm, code)
        x = 0.0                                   # heater rise above T0, K
        ps = []
        for k in range(40):
            if implicit:
                out = bank.step(k, t_heater_pre_k=298.15 + a * x, g_self_k_per_w=(1 - a) * RTH_H)
            else:
                out = bank.step(k, t_heater_pre_k=298.15 + x)
            pk = out.p_w[0, 0]
            x = a * x + (1 - a) * RTH_H * pk
            ps.append(pk)
        assert ps[-1] == pytest.approx(target, rel=1e-10)
        if implicit:                                  # implicit: monotone approach, no period-2 ringing
            dps = np.diff(ps)
            assert np.all(dps * np.sign(dps[0]) >= -1e-15)


# ============================================================================================ compliance, supply
def test_current_compliance_clip_and_headroom():
    # I = 30 mA into 50 ohm needs 1.5 V; compliance 1.2 V -> voltage source at 1.2 V
    p, v, i, r, fl = _power_row([1.0, 0.5], el.LAW_CURRENT, v_comp=1.2)
    assert fl[0] & el.FLAG_COMPLIANCE and not fl[1]
    assert v[0] == 1.2 and p[0] == pytest.approx(1.44 / 50.0, rel=1e-14) and i[0] == pytest.approx(0.024)
    assert p[1] == pytest.approx(0.015 ** 2 * 50.0, rel=1e-14)
    hr = el.compliance_headroom_v([0.03, 0.015], 50.0, 1e-3, [0.0, 100.0], 1.2)
    np.testing.assert_allclose(hr, [1.2 - 1.5, 1.2 - 0.015 * 55.0], rtol=1e-14)
    # TCR pushes a current driver into compliance as the heater heats: 25 mA x 45.4 ohm x (1 + 1e-3 dT) > 1.2 V
    p, v, i, r, fl = _power_row([1.0], el.LAW_CURRENT, r0=45.4, i_fs=0.025, v_comp=1.2, t_pre=298.15 + 100.0)
    assert fl[0] & el.FLAG_COMPLIANCE and v[0] == 1.2
    # power law: P = 30 mW into 50 ohm needs 1.22 V
    p, v, i, r, fl = _power_row([1.0], el.LAW_POWER, v_comp=1.0)
    assert fl[0] & el.FLAG_COMPLIANCE and p[0] == pytest.approx(1.0 / 50.0, rel=1e-14)


def test_current_runaway_guard():
    # k I^2 >= 1: positive electrothermal feedback without bound -> runaway flag, compliance holds it
    p, v, i, r, fl = _power_row([1.0], el.LAW_CURRENT, g=1.0 / (50.0 * 1e-3 * 0.03 ** 2) * 1.5, v_comp=2.0)
    assert fl[0] & el.FLAG_RUNAWAY and fl[0] & el.FLAG_COMPLIANCE
    assert v[0] == 2.0 and np.isfinite(p[0]) and p[0] > 0


def test_supply_limit_clip_and_flag():
    prm = _heater(N=4, r_cold_ohm=50.0, tcr_per_k=0.0, supply_power_max_w=0.100)
    bank = el.HeaterBank(prm, 0)
    code = 58000                                   # 1.416 V into 50 ohm: 40.1 mW each, 160 mW total
    out = bank.step(0, cmd=np.full((1, 4), code))
    p_each = (code / 65535 * 1.6) ** 2 / 50.0
    s = math.sqrt(0.100 / (4 * p_each))
    assert out.supply_scale[0] == pytest.approx(s, rel=1e-13)
    assert out.p_w.sum() == pytest.approx(0.100, rel=1e-13)
    assert np.all(out.flags & el.FLAG_SUPPLY)
    # current limit: sum I = 4 x 28.3 mA; limit 80 mA -> s = 0.08 / sum I
    prm = _heater(N=4, r_cold_ohm=50.0, tcr_per_k=0.0, supply_current_max_a=0.08)
    out = el.HeaterBank(prm, 0).step(0, cmd=np.full((1, 4), code))
    assert out.i_a.sum() == pytest.approx(0.08, rel=1e-13)
    assert np.all(out.flags & el.FLAG_SUPPLY)
    # under the limits: no flag, no scaling
    prm = _heater(N=4, r_cold_ohm=50.0, tcr_per_k=0.0, supply_power_max_w=1.0, supply_current_max_a=1.0)
    out = el.HeaterBank(prm, 0).step(0, cmd=np.full((1, 4), code))
    assert out.supply_scale[0] == 1.0 and not np.any(out.flags & el.FLAG_SUPPLY)


# ============================================================================================ DAC INL / DNL
def test_dac_ideal_and_bit_error_dnl():
    frac, inl, dnl = el.dac_transfer_curve(10)
    np.testing.assert_allclose(frac, np.arange(1024) / 1023, rtol=0, atol=1e-15)
    assert np.abs(inl).max() < 1e-12 and np.abs(dnl).max() < 1e-12
    # a +0.8 LSB MSB error gives mid-scale DNL = e9 - sum(e_i<9) (before endpoint scaling)
    e = np.zeros(10)
    e[9] = 0.8
    frac, inl, dnl = el.dac_transfer_curve(10, e)
    scale = 1023 / (1023 + 0.8)
    assert dnl[511] == pytest.approx((512 + 0.8) * scale - 511 * scale - 1, rel=1e-12)
    assert inl[0] == 0.0 and abs(inl[-1]) < 1e-12                    # endpoint-referenced
    # bow INL: +2 LSB at mid-scale, 0 at the ends
    frac, inl, dnl = el.dac_transfer_curve(10, None, 2.0)
    assert inl[511] == pytest.approx(2.0 * 4 * (511 / 1023) * (512 / 1023), rel=1e-12)


def test_dac_bit_errors_from_spec_statistics():
    rng = np.random.default_rng(1)
    n, sig = 10, 0.5
    e = el.dac_bit_errors_from_spec(sig, n, rng.standard_normal((4000, n)))
    dnl_mid = e[:, n - 1] - e[:, : n - 1].sum(axis=1)
    assert dnl_mid.std() == pytest.approx(sig, rel=0.05)


def test_dac_errors_reach_heater_power():
    e = np.zeros(16)
    e[15] = 3.0                                   # MSB +3 LSB: a DNL step at the 32767 -> 32768 carry
    prm = _heater(N=2, bit_err_lsb=e, tcr_per_k=0.0)
    out = el.HeaterBank(prm, 0).step(0, cmd=np.array([[32767, 32768]]))
    ytot = 65535 + 3.0
    np.testing.assert_allclose(out.u[0], [32767 / ytot, (32768 + 3.0) / ytot], rtol=1e-14)


# ============================================================================================ bias DAC, NRZ levels
def test_bias_dac_transfer_noise_and_offset():
    prm = el.BiasDacParams.build(1, 2, DT, bits=12, update_rate_hz=1.0 / (5 * DT), noise_rms_codes=0.7)
    lsb = (3.75 - 1.0) / 4095
    assert prm.lsb_v[0, 0] == pytest.approx(lsb)
    bank = el.BiasBank(prm, np.array([[0, 4095]]))
    z = np.array([[1.0, -2.0]])
    out = bank.step(0, z=z, v_offset=np.array([[0.01, 0.0]]))
    np.testing.assert_allclose(out.v_avg[0], [1.0 + 0.7 * lsb * 1.0 + 0.01, 3.75 - 0.7 * lsb * 2.0], rtol=1e-14)
    # noise is held between DAC updates (every 5 ticks), redrawn on update ticks
    v = [bank.step(k, z=np.full((1, 2), float(k))).v_avg[0, 0] for k in range(1, 11)]
    np.testing.assert_allclose(v[:4], 1.0 + 0.7 * lsb, rtol=1e-14)            # ticks 1-4 hold the tick-0 draw
    np.testing.assert_allclose(v[4:9], 1.0 + 0.7 * lsb * 5.0, rtol=1e-14)     # tick 5 draws z = 5
    assert v[9] == pytest.approx(1.0 + 0.7 * lsb * 10.0, rel=1e-14)


def test_nrz_levels_and_forward_clip():
    lo, hi, fl = np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3), np.int8)
    vb = np.array([[1.0, 2.0, 0.2]])
    el.nrz_levels(vb, np.array([[1.3]]), el.LEVELS_CENTERED, -0.5, 4.0, lo, hi, fl)
    np.testing.assert_allclose(lo, [[0.35, 1.35, -0.45]])
    np.testing.assert_allclose(hi, [[1.65, 2.65, 0.85]])
    assert not fl.any()
    lo, hi, fl = np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3), np.int8)
    el.nrz_levels(vb, np.array([[1.3]]), el.LEVELS_ONE_SIDED, -0.5, 4.0, lo, hi, fl)
    np.testing.assert_allclose(lo, [[-0.3, 0.7, -0.5]])                          # forward region; last clamped
    np.testing.assert_allclose(hi, [[1.0, 2.0, 0.2]])
    assert fl[0].tolist() == [0, 0, el.FLAG_FORWARD_CLIP]


# ============================================================================================ junction runtime
LAM, NG = 1.31e-6, 3.9          # m; n_g assumed for the test (01: 3.91 to 4.04)
L_RING = 2 * math.pi * 7.5e-6   # m, "15 um diameter" (01, printed)
A_FIT = 137.6e-12               # m/V^0.5: derived fit to the G25 ladder (03 section 5.3, m = 1/2, V_bi = 0.9 V)


def _jt(**kw):
    base = dict(shift_amp_m=A_FIT, v_bi_v=0.9, m=0.5, lambda_m=LAM, n_g=NG, c0_f_per_m=25e-15 / L_RING,
                alpha0_per_m=870.0, loss_ratio_q=0.06, r_series_ohm_m=60.0 * L_RING)
    base.update(kw)
    return el.analytic_junction_table(**base)


def test_depletion_fallback_reproduces_03_fit():
    v = np.array([0.8, 1.2, 1.6])
    meas_pm = np.array([56.4, 66.8, 84.4])    # read-off, G25 ladder (textbook MOD_SHIFT_PM)
    sh = np.array([el.depletion_shift_m(x, A_FIT, 0.9, 0.5) for x in v]) * 1e12
    rms = math.sqrt((np.sum((sh - meas_pm) ** 2)) / 4)  # 4 points incl. (0, 0)
    assert rms == pytest.approx(4.12, abs=0.01)           # 03 section 5.3: rms 4.1 pm
    jt = _jt()
    eff0 = el.junction_efficiency_m_per_v(jt, 0.0, 1.0, LAM, NG) * 1e12
    assert eff0 == pytest.approx(72.5, abs=0.1)            # 03 table: 72.5 pm/V at 0 V
    # 07 P1-7: a +0.5 V reverse step near the G26 bias moves the resonance by about 25 pm (red)
    out = jt.eval(np.array([[1.0, 1.5]]), 1.0, L_RING)
    dlam_pm = LAM / NG * (out["dneff"][0, 1] - out["dneff"][0, 0]) * 1e12
    assert dlam_pm == pytest.approx(23.5, abs=0.1) and dlam_pm > 0
    with pytest.raises(ValueError):
        _jt(m=0.25)


def test_ring_level_analytic_builder_round_trip():
    # schema ring.junction.* (ring level) -> full-circumference table -> runtime fill scaling gives them back
    fill, eff0, c0, rs, a_ring = 0.55, 50e-12, 22e-15, 80.0, 1500.0
    jt = el.analytic_junction_table_from_ring(eff_at_0v_m_per_v=eff0, v_bi_v=0.9, m=0.5, fill_fraction=fill,
                                              ring_length_m=L_RING, lambda_m=LAM, n_g=NG, capacitance_at_0v_f=c0,
                                              series_resistance_ohm=rs, alpha_fca_ring_per_m=a_ring,
                                              dqi_dv_frac_per_v=0.02, alpha_i_total_per_m=2900.0)
    out = jt.eval(np.array([[0.0]]), fill, L_RING)
    assert el.junction_efficiency_m_per_v(jt, 0.0, fill, LAM, NG) == pytest.approx(eff0, rel=1e-4)
    assert out["c_f"][0, 0] == pytest.approx(c0, rel=1e-13)
    assert out["r_s_ohm"][0, 0] == pytest.approx(rs, rel=1e-13)
    assert out["alpha_per_m"][0, 0] == pytest.approx(a_ring, rel=1e-13)
    # the dQi/dV conversion: -d(alpha_ring)/dV / alpha_i at 0 V equals the requested 0.02 /V
    h = 0.01
    a = jt.eval(np.array([[-h, h]]), fill, L_RING)["alpha_per_m"][0]
    assert -(a[1] - a[0]) / (2 * h) / 2900.0 == pytest.approx(0.02, rel=1e-3)


def test_code_clamp_range():
    bank = el.HeaterBank(_heater(code_min=1000, code_max=60000), 0)
    assert bank.applied_codes[0, 0] == 1000                      # the initial code is clamped too
    assert bank.step(0, cmd=np.array([[65000]])).applied_codes[0, 0] == 60000
    with pytest.raises(ValueError):
        _heater(code_min=5, code_max=70000)


def test_junction_lut_interp_and_fill_scaling():
    g = np.array([-0.5, 0.0, 1.0, 2.0, 4.0])
    jt = el.JunctionTable(g, [-1e-4, 0.0, 2e-4, 3e-4, 4.5e-4], [900., 870., 800., 760., 700.],
                          [5e-10, 4e-10, 3e-10, 2.6e-10, 2e-10], 2e-3)
    fill, L = 0.6, 50e-6
    out = jt.eval(np.array([[0.0, 1.0, 1.5, -0.25, 5.0]]), fill, L)
    np.testing.assert_allclose(out["dneff"][0], fill * np.array([0.0, 2e-4, 2.5e-4, -0.5e-4, 4.5e-4]), rtol=1e-14)
    np.testing.assert_allclose(out["alpha_per_m"][0], fill * np.array([870., 800., 780., 885., 700.]), rtol=1e-14)
    np.testing.assert_allclose(out["c_f"][0], fill * L * np.array([4e-10, 3e-10, 2.8e-10, 4.5e-10, 2e-10]),
                               rtol=1e-14)
    np.testing.assert_allclose(out["r_s_ohm"][0], 2e-3 / (fill * L), rtol=1e-14)
    assert out["flags"][0].tolist() == [0, 0, 0, 0, el.FLAG_GRID_CLIP]


def test_junction_artefact_loader(tmp_path):
    jt = _jt()
    d = {"schema": "junction/1", "method": "analytic", "v_reverse_v": jt.v_reverse_v.tolist(),
         "dneff": jt.dneff[0].tolist(), "alpha_fca_per_m": jt.alpha_fca_per_m[0].tolist(),
         "capacitance_f_per_m": jt.capacitance_f_per_m[0].tolist(), "r_series_ohm_m": 60.0 * L_RING}
    path = tmp_path / "junction_test.json"
    path.write_text(json.dumps(d))
    jt2 = el.load_junction_table(path)
    v = np.array([[-0.3, 0.4, 2.2, 3.7]])
    a, b = jt.eval(v, 0.5, L_RING), jt2.eval(v, 0.5, L_RING)
    for key in ("dneff", "alpha_per_m", "c_f", "r_s_ohm"):
        np.testing.assert_allclose(a[key], b[key], rtol=1e-14)
    assert b["r_s_ohm"][0, 0] == pytest.approx(60.0 / 0.5)
    d["schema"] = "modes/1"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        el.load_junction_table(path)


def test_alpha_and_c_follow_depletion():
    jt = _jt()
    out = jt.eval(np.array([[0.0, 2.0]]), 1.0, L_RING)
    dn = out["dneff"][0, 1]
    assert out["alpha_per_m"][0, 1] == pytest.approx(870.0 - 4 * math.pi * 0.06 / LAM * dn, rel=1e-6)
    # C(V) = C0 (1 + V/V_bi)^-(1-m), m = 1/2
    assert out["c_f"][0, 1] == pytest.approx(25e-15 * (1 + 2.0 / 0.9) ** -0.5, rel=1e-4)


def test_swing_capacitance_exact_integral():
    jt = _jt()
    c0p = 25e-15 / L_RING
    lo, hi = np.array([[0.35, 1.0]]), np.array([[1.65, 1.0]])
    c = np.zeros((1, 2))
    el.junction_swing_capacitance(lo, hi, jt.v_reverse_v, jt.capacitance_f_per_m, np.array([[1.0]]),
                                  np.array([[L_RING]]), c)
    # closed form: integral of c0 (1 + v/Vbi)^-1/2 dv = 2 c0 Vbi sqrt(1 + v/Vbi)
    F = lambda v: 2 * c0p * 0.9 * math.sqrt(1 + v / 0.9)
    assert c[0, 0] == pytest.approx(L_RING * (F(1.65) - F(0.35)) / 1.3, rel=2e-5)
    assert c[0, 1] == pytest.approx(25e-15 * (1 + 1.0 / 0.9) ** -0.5, rel=2e-5)   # zero swing -> C(V)
    # reversed limits and a swing inside one grid segment, also below the grid (held end value)
    c2 = np.zeros((1, 3))
    el.junction_swing_capacitance(np.array([[1.65, 1.001, -0.9]]), np.array([[0.35, 1.004, -0.6]]),
                                  jt.v_reverse_v, jt.capacitance_f_per_m, np.array([[1.0]]), np.array([[L_RING]]), c2)
    assert c2[0, 0] == pytest.approx(c[0, 0], rel=1e-14)
    cmid = np.interp(1.0025, jt.v_reverse_v, jt.capacitance_f_per_m[0]) * L_RING
    assert c2[0, 1] == pytest.approx(cmid, rel=1e-12)
    assert c2[0, 2] == pytest.approx(jt.capacitance_f_per_m[0, 0] * L_RING, rel=1e-14)


def test_junction_per_world_tables():
    j160, j200 = _jt(shift_amp_m=A_FIT), _jt(shift_amp_m=2 * A_FIT)   # one table per thickness world (D21)
    tabs = [np.vstack([getattr(j160, k), getattr(j200, k)]) for k in
            ("dneff", "alpha_fca_per_m", "capacitance_f_per_m", "r_series_ohm_m")]
    out = [np.zeros((2, 1)) for _ in range(4)]
    fl = np.zeros((2, 1), np.int8)
    el.junction_eval(np.array([[1.5]]), j160.v_reverse_v, *tabs, np.array([[1.0]]), np.array([[L_RING]]),
                     *out, fl)
    assert out[0][1, 0] == pytest.approx(2 * out[0][0, 0], rel=1e-14)


# ============================================================================================ data-activity heat
def test_data_activity_heat_formulas():
    pj, pd = el.data_activity_heat_w(20e-15, 1.0, 50e9, 0.5, 100.0, 100.0, p_static_w=1e-3, e_tr_j=50e-15)
    assert pj == pytest.approx(0.5 * 20e-15 * 1.0 * 50e9 * 0.5 * 0.5, rel=1e-14)    # 0.125 mW
    assert pd == pytest.approx(1e-3 + 50e-15 * 50e9 * 0.5, rel=1e-14)
    pj2, _ = el.data_activity_heat_w(20e-15, 1.0, 50e9, 0.25, 100.0, 100.0)
    assert pj2 == pytest.approx(pj / 2, rel=1e-14)                                  # 07 P1-8 test
    # 07 P1-8: random NRZ draws C V^2/4 per bit: 4 to 17 fJ/bit, 0.2 to 0.85 mW at 50 Gb/s (all in R_s)
    pj_lo, _ = el.data_activity_heat_w(20e-15, 0.9, 50e9, 0.5, 100.0, 0.0)
    pj_hi, _ = el.data_activity_heat_w(40e-15, 1.3, 50e9, 0.5, 100.0, 0.0)
    assert pj_lo == pytest.approx(0.2e-3, rel=0.02) and pj_hi == pytest.approx(0.845e-3, rel=0.01)
    # driver off, squelch, and the optional load share
    assert el.data_activity_heat_w(20e-15, 1.0, 50e9, 0.5, 100.0, 100.0, 1e-3, 5e-14, enable=0.0) == (0.0, 0.0)
    pj_s, pd_s = el.data_activity_heat_w(20e-15, 0.0, 50e9, 0.5, 100.0, 100.0, 1e-3, 5e-14)
    assert pj_s == 0.0 and pd_s == pytest.approx(1e-3 + 5e-14 * 50e9 * 0.5)
    _, pd_l = el.data_activity_heat_w(20e-15, 1.0, 50e9, 0.5, 100.0, 300.0, add_load_share=True)
    assert pd_l == pytest.approx(0.5 * 20e-15 * 50e9 * 0.5 * 300.0 / 400.0, rel=1e-14)


# ============================================================================================ NUMBA_DISABLE_JIT
_NOJIT_SCRIPT = r"""
import json, numpy as np
from mrm.plant import electrical as el
prm = el.HeaterDriverParams.build(2, 3, 1e-6, law=["current", "dac_pwm"], v_full_scale_v=1.6, i_full_scale_a=0.03,
    r_cold_ohm=45.4, tcr_per_k=1e-3, tcr_ref_k=298.15, latency_s=2e-6, pwm_ripple=True, slew_codes_per_s=5e7,
    settle_tau_s=0.3e-6, compliance_v=1.5, supply_power_max_w=0.05)
bank = el.HeaterBank(prm, 1000)
ps = []
for k in range(8):
    o = bank.step(k, cmd=np.full((2, 3), 40000 + 1000 * k), t_heater_pre_k=300.0 + k, g_self_k_per_w=2000.0)
    ps.append(o.p_w.tolist())
bp = el.BiasDacParams.build(1, 2, 1e-6, bits=10, latency_s=1e-6, slew_v_per_s=1e6, settle_tau_s=1e-7,
                            noise_rms_codes=0.5)
bb = el.BiasBank(bp, 0)
vs = [bb.step(k, cmd=np.array([[500, 900]]), z=np.full((1, 2), 0.3)).v_avg.tolist() for k in range(4)]
jt = el.analytic_junction_table(shift_amp_m=1e-10, v_bi_v=0.9, lambda_m=1.31e-6, n_g=3.9, c0_f_per_m=5e-10,
                                alpha0_per_m=800., loss_ratio_q=0.06, r_series_ohm_m=3e-3)
j = jt.eval(np.array([[-0.2, 1.7]]), 0.7, 47e-6)
print(json.dumps({"p": ps, "v": vs, "dn": j["dneff"].tolist(), "c": j["c_f"].tolist()}))
"""


@pytest.mark.filterwarnings("ignore:pwm_ripple with settle_tau_s")
def test_runs_under_numba_disable_jit():
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    res = subprocess.run([sys.executable, "-c", _NOJIT_SCRIPT], capture_output=True, text=True, env=env,
                         timeout=120)
    assert res.returncode == 0, res.stderr
    nojit = json.loads(res.stdout.strip().splitlines()[-1])
    ns = {}
    exec(compile(_NOJIT_SCRIPT.replace("print(json.dumps(", "RESULT = (("), "<jit>", "exec"), ns)
    jit = ns["RESULT"]
    for key in ("p", "v", "dn", "c"):
        np.testing.assert_allclose(np.array(nojit[key]), np.array(jit[key]), rtol=1e-12, atol=1e-300)


# ============================================================================================ review additions
def test_current_runaway_without_compliance_is_finite_and_grows():
    # k I^2 >= 1 with no compliance limit: no finite implicit root. The kernel must not return NaN (it would poison
    # the thermal state); it returns the explicit P = I^2 r_o, flags runaway, and the runaway grows tick by tick.
    g = 1.0 / (50.0 * 1e-3 * 0.03 ** 2) * 1.5
    p, v, i, r, fl = _power_row([1.0], el.LAW_CURRENT, g=g)
    assert np.isfinite(p[0]) and np.isfinite(v[0]) and np.isfinite(r[0])
    assert fl[0] & el.FLAG_RUNAWAY and not fl[0] & el.FLAG_COMPLIANCE
    assert p[0] == pytest.approx(0.03 ** 2 * 50.0, rel=1e-14) and v[0] == pytest.approx(0.03 * 50.0, rel=1e-14)
    # explicit coupling of a static, unstable heater (y P_cold > 1): power grows every tick, stays finite
    prm = _heater(law="current", i_full_scale_a=0.05, r_cold_ohm=50.0, tcr_per_k=1e-3)
    bank = el.HeaterBank(prm, 65535)
    x, ps = 0.0, []
    for k in range(30):
        out = bank.step(k, t_heater_pre_k=298.15 + x)
        ps.append(out.p_w[0, 0])
        x = 0.5 * x + 0.5 * 20000.0 * out.p_w[0, 0]     # one pole, R_th,H = 20 K/mW: y P_cold = 2.5
    assert np.all(np.isfinite(ps)) and np.all(np.diff(ps) > 0)


def test_build_validates_full_scale_latency_and_ranges():
    with pytest.raises(ValueError, match="v_full_scale_v"):           # voltage law with no full scale -> NaN power
        el.HeaterDriverParams.build(1, 1, DT, law="voltage", r_cold_ohm=50.0, tcr_per_k=1e-3)
    with pytest.raises(ValueError, match="i_full_scale_a"):
        el.HeaterDriverParams.build(2, 1, DT, law=["voltage", "current"], v_full_scale_v=1.6, r_cold_ohm=50.0,
                                    tcr_per_k=1e-3)
    with pytest.raises(ValueError, match="p_full_scale_w"):
        el.heater_steady_state(np.array([100]), law="power", r_cold_ohm=50.0, tcr_per_k=1e-3, r_th_heater_k_per_w=0.0)
    with pytest.raises(ValueError):                                     # negative latency (would read the future)
        _heater(latency_s=-DT)
    with pytest.raises(ValueError, match="latency_s"):                  # not a whole number of ticks (07 P0-6)
        _heater(latency_s=2.5 * DT)
    with pytest.raises(ValueError, match="dac_max_update_hz"):
        _heater(dac_max_update_hz=3e6)
    with pytest.raises(ValueError):
        _heater(slew_codes_per_s=-1.0)
    with pytest.raises(ValueError):
        el.BiasDacParams.build(1, 1, DT, bits=12, v_min_v=2.0, v_max_v=1.0)
    with pytest.raises(ValueError):
        el.BiasDacParams.build(1, 1, DT, bits=12, latency_s=-DT)
    # the same tick convention as mrm.core.timing (one owner of the rule)
    from mrm.core.timing import seconds_to_ticks
    assert _heater(latency_s=3 * DT).latency_ticks[0] == seconds_to_ticks(3 * DT, DT, "x") == 3


def test_quiescence_false_while_command_pending_and_skip_is_exact():
    # P0-10: a command waiting in the latch (DAC max update rate) must block multi-tick skipping
    prm = _heater(dac_max_update_hz=1.0 / (10 * DT), settle_tau_s=2e-6)
    bank = el.HeaterBank(prm, 1000)
    bank.step(0, cmd=np.array([[2000]]))
    for k in range(1, 5):
        bank.step(k)
    bank.step(5, cmd=np.array([[3000]]))                                # pending until tick 10
    assert bank.has_pending[0, 0] == 1 and bank.reg[0, 0] == 2000
    tgt = np.array([2000 / 65535])
    assert not el.actuator_quiescent_row(bank.reg[0], bank.has_pending[0], bank.fifo[0], bank.y_slew[0],
                                         bank.y_out[0], tgt, 1.0)
    # tick-by-tick reference vs "step until quiescent, then skip with settle_advance_row"
    ref = el.HeaterBank(prm, 1000)
    alt = el.HeaterBank(prm, 1000)
    cmds = {0: 2000, 5: 3000}
    for k in range(40):
        c = np.array([[cmds[k]]]) if k in cmds else None
        ref.step(k, cmd=c)
    k = 0
    while k < 40:
        c = np.array([[cmds[k]]]) if k in cmds else None
        alt.step(k, cmd=c)
        k += 1
        if k < 40 and k not in cmds and el.actuator_quiescent_row(
                alt.reg[0], alt.has_pending[0], alt.fifo[0], alt.y_slew[0], alt.y_out[0], alt._u_target[0], 1.0):
            nxt = min([t for t in cmds if t >= k] + [40])
            el.settle_advance_row(alt._u_target[0], alt.y_slew[0], alt.y_out[0], prm.settle_tau_s[0], DT, nxt - k)
            k = nxt
    assert alt.reg[0, 0] == ref.reg[0, 0] == 3000
    np.testing.assert_allclose(alt.y_out, ref.y_out, rtol=1e-12)


def test_bias_noise_draw_broadcasts_over_rings():
    # z of shape (1, 1) or (B, 1) must reach every ring (the row kernel indexes it with broadcasting)
    prm = el.BiasDacParams.build(2, 3, DT, bits=10, noise_rms_codes=1.0)
    bank = el.BiasBank(prm, 100)
    lsb = prm.lsb_v[0, 0]
    out = bank.step(0, z=np.array([[2.0], [-1.0]]))
    np.testing.assert_allclose(out.v_avg - (1.0 + 100 * lsb), [[2 * lsb] * 3, [-lsb] * 3], rtol=1e-12)
    zk = np.ones((1, 1))
    bank2 = el.BiasBank(prm, 100)
    args = [np.int64(0), bank2._no_cmd, bank2._no_valid, bank2.reg, bank2.pending, bank2.has_pending,
            bank2.last_update, bank2.fifo, bank2.y_slew, bank2.y_out, bank2.noise_hold, prm.bits, prm.bit_err_lsb,
            prm.inl_bow_lsb, prm.v_min_v, prm.v_max_v, prm.latency_ticks, prm.min_update_ticks,
            prm.noise_period_ticks, prm.noise_rms_codes, zk, prm.slew_v_per_s, prm.settle_tau_s, prm.dt_s,
            np.zeros((1, 1)), bank2._code_dac, bank2._v_target, bank2._v_avg, bank2._v_end]
    el.bias_bank_step(*args)
    np.testing.assert_allclose(bank2.noise_hold, lsb, rtol=1e-12)


def test_bias_slew_and_settle_in_volts():
    rate = 0.1 / DT                                                         # 0.1 V per tick
    prm = el.BiasDacParams.build(1, 1, DT, bits=12, slew_v_per_s=rate)
    c1, c2 = prm.code_for_voltage(1.0)[0, 0], prm.code_for_voltage(2.0)[0, 0]
    bank = el.BiasBank(prm, c1)
    v1 = 1.0 + c1 * prm.lsb_v[0, 0] - 0.0
    v2 = 1.0 + c2 * prm.lsb_v[0, 0]
    ends = [bank.step(k, cmd=np.array([[c2]]) if k == 0 else None).v_end[0, 0] for k in range(14)]
    n_ramp = int(math.floor((v2 - v1) / 0.1))
    np.testing.assert_allclose(np.diff([v1] + ends)[:n_ramp], 0.1, rtol=1e-9)
    assert ends[-1] == pytest.approx(v2, abs=1e-12)


def test_per_world_latency_and_batch_invariance():
    prm = el.HeaterDriverParams.build(3, 1, DT, law="voltage", v_full_scale_v=1.6, r_cold_ohm=50.0, tcr_per_k=0.0,
                                      latency_s=[0.0, 2 * DT, 5 * DT])
    bank = el.HeaterBank(prm, 0)
    P = np.array([bank.step(k, cmd=np.full((3, 1), 30000 if k == 3 else 0)).p_w[:, 0].copy() for k in range(12)])
    assert [np.flatnonzero(P[:, b] > 0).tolist() for b in range(3)] == [[3], [5], [8]]
    # element b of a B = 3 run equals the B = 1 run with world b's parameters (07 4.2 batch invariance)
    r0 = np.array([[45.0], [50.0], [55.0]])
    laws, lats = ["voltage", "current", "dac_pwm"], [DT, 2 * DT, 0.0]
    kw = dict(v_full_scale_v=1.6, i_full_scale_a=0.03, tcr_per_k=1e-3, slew_codes_per_s=3e7, pwm_ripple=True,
              supply_power_max_w=0.06)
    b3 = el.HeaterBank(el.HeaterDriverParams.build(3, 2, DT, law=laws, r_cold_ohm=r0, latency_s=lats, **kw), 500)
    singles = [el.HeaterBank(el.HeaterDriverParams.build(1, 2, DT, law=laws[b], r_cold_ohm=r0[b, 0],
                                                         latency_s=lats[b], **kw), 500) for b in range(3)]
    rng = np.random.default_rng(0)
    for k in range(30):
        cmd = rng.integers(0, 65536, (3, 2))
        o3 = b3.step(k, cmd=cmd, t_heater_pre_k=300.0 + k, g_self_k_per_w=1000.0)
        for b in range(3):
            o1 = singles[b].step(k, cmd=cmd[b:b + 1], t_heater_pre_k=300.0 + k, g_self_k_per_w=1000.0)
            assert np.array_equal(o1.p_w[0], o3.p_w[b]) and o1.supply_scale[0] == o3.supply_scale[b]


def test_row_kernels_inside_prange_match_batched():
    import numba as nb

    @nb.njit(parallel=True)
    def run(tick, cmd, valid, st, law, prm_s, prm_a, lat, tpre, g, outs_f, cd, fl, sc):
        reg, pend, hasp, last, fifo, ys, yo = st
        bits, pwmb, rip, dtc, lo, hi, mu = prm_s
        be, bow, slew, tau, r0, al, t0, vfs, ifs, pfs, vc, pmax, imax, dt = prm_a
        for b in nb.prange(reg.shape[0]):
            sc[b] = el.heater_bank_step_row(
                tick, cmd[b], valid[b], reg[b], pend[b], hasp[b], last[b], fifo[b], ys[b], yo[b],
                law[b], bits, pwmb, rip, dtc, be[0], bow[0], lo, hi, lat[b], mu, slew[0], tau[0], dt,
                r0[b], al[0], t0[0], vfs[0], ifs[0], pfs[0], vc[0], pmax[0], imax[0], tpre[b], g[0],
                cd[b], outs_f[0][b], outs_f[1][b], outs_f[2][b], outs_f[3][b], outs_f[4][b], outs_f[5][b],
                outs_f[6][b], fl[b])

    B, N = 4, 3
    kw = dict(law=["voltage", "current", "power", "dac_pwm"], v_full_scale_v=1.6, i_full_scale_a=0.03,
              p_full_scale_w=0.03, r_cold_ohm=np.array([[44.0], [45.0], [46.0], [47.0]]), tcr_per_k=1e-3,
              latency_s=[0.0, DT, 2 * DT, 3 * DT], pwm_ripple=True, slew_codes_per_s=4e7, supply_power_max_w=0.05)
    prm = el.HeaterDriverParams.build(B, N, DT, **kw)
    ref, par = el.HeaterBank(prm, 1000), el.HeaterBank(prm, 1000)
    outs_f = tuple(np.zeros((B, N)) for _ in range(7))
    cd, fl, sc = np.zeros((B, N), np.int64), np.zeros((B, N), np.int8), np.zeros(B)
    st = (par.reg, par.pending, par.has_pending, par.last_update, par.fifo, par.y_slew, par.y_out)
    prm_s = (prm.bits, prm.pwm_bits, prm.pwm_ripple, prm.dt_clk, prm.code_lo, prm.code_hi, prm.min_update_ticks)
    prm_a = (prm.bit_err_lsb, prm.inl_bow_lsb, prm.slew_u_per_s, prm.settle_tau_s, prm.r_cold_ohm, prm.tcr_per_k,
             prm.tcr_ref_k, prm.v_full_scale_v, prm.i_full_scale_a, prm.p_full_scale_w, prm.compliance_v,
             prm.supply_power_max_w, prm.supply_current_max_a, prm.dt_s)
    rng = np.random.default_rng(3)
    tpre, g = np.full((B, 1), 301.0), np.full((1, 1), 1500.0)
    for k in range(10):
        cmd = rng.integers(0, 65536, (B, N)).astype(np.int64)
        o = ref.step(k, cmd=cmd, t_heater_pre_k=tpre, g_self_k_per_w=g)
        run(np.int64(k), cmd, np.ones((B, N), np.bool_), st, prm.law, prm_s, prm_a, prm.latency_ticks, tpre, g,
            outs_f, cd, fl, sc)
        assert np.array_equal(outs_f[3], o.p_w) and np.array_equal(sc, o.supply_scale)
        assert np.array_equal(outs_f[4], o.v_v) and np.array_equal(fl, o.flags)


def test_tcr_shapes_gain_across_code_range():
    # 07 P1-11: dP/dcode varies 49300/19500 = 2.53x in voltage/current drive without TCR. With the textbook's
    # y = 0.00904 /mW (fitted to the G25 droop), the self-heating TCR compresses the voltage-drive ratio to 1.87 and
    # stretches the current-drive ratio to 4.55, far more than 07's "about +-10 %", which matches only the ambient
    # term: +80 K of die temperature lowers the power at a fixed voltage code by about 6 % (06 E1: "about 8 %").
    codes = np.array([19500, 49300])
    kw = dict(r_cold_ohm=R_COLD_OHM, tcr_per_k=TCR_PER_K, r_th_heater_k_per_w=RTH_H)
    gv = el.heater_gain_w_per_code(codes, law="voltage", v_full_scale_v=1.6, **kw)
    gi = el.heater_gain_w_per_code(codes, law="current", i_full_scale_a=1.6 / R_COLD_OHM, **kw)
    gp = el.heater_gain_w_per_code(codes, law="power", p_full_scale_w=0.05, **kw)
    assert gv[1] / gv[0] == pytest.approx(1.872, abs=0.002)
    assert gi[1] / gi[0] == pytest.approx(4.552, abs=0.002)
    assert gp[1] / gp[0] == pytest.approx(1.0, abs=1e-9)
    s = [el.heater_steady_state(np.array([40000]), law="voltage", v_full_scale_v=1.6, t_base_minus_ref_k=dtk,
                                **kw)["p_w"][0] for dtk in (0.0, 80.0)]
    assert 1 - s[1] / s[0] == pytest.approx(0.057, abs=0.002)


def test_explicit_coupling_ringing_ratio_is_loop_gain():
    # heater_power_row docstring: explicit coupling multiplier a + (1 - a) L; with a -> 0 the error alternates
    # sign with ratio L = -yP/(1 + yP) (about -0.19 at 25 mW), the implicit one does not ring
    code = int(round(V_FSR_V / 1.6 * 65535))
    p_star = el.heater_steady_state(np.array([code]), law="voltage", v_full_scale_v=1.6, r_cold_ohm=R_COLD_OHM,
                                    tcr_per_k=TCR_PER_K, r_th_heater_k_per_w=RTH_H)["p_w"][0]
    bank = el.HeaterBank(_heater(law="voltage"), code)
    x = RTH_H * p_star * 1.01                                   # start 1 % off the fixed point, a = 0 (tau << dt)
    errs = []
    for k in range(6):
        p = bank.step(k, t_heater_pre_k=298.15 + x).p_w[0, 0]
        x = RTH_H * p
        errs.append(x - RTH_H * p_star)
    ratios = np.array(errs[1:]) / np.array(errs[:-1])
    L = el.electrothermal_loop_gain("voltage", p_star, TCR_PER_K, RTH_H)
    np.testing.assert_allclose(ratios, L, rtol=2e-2)
    assert -0.20 < L < -0.17


def test_nonlinear_junction_dissipation_matches_rc_simulation():
    # junction_swing_capacitance_row claim: for C_j(V) charged through R from a step source, the energy dissipated
    # per transition, averaged over up and down, is 0.5*Vpp*Delta Q = 0.5*C_eff*Vpp^2. Check against an ODE.
    from scipy.integrate import solve_ivp
    vbi, c0p, r_tot = 0.9, 25e-15 / L_RING, 200.0
    c0 = c0p * L_RING
    q_of_v = lambda v: 2 * c0 * vbi * (math.sqrt(1 + v / vbi) - 1)           # integral of c0 (1 + v/vbi)^-1/2
    v_of_q = lambda q: vbi * ((1 + q / (2 * c0 * vbi)) ** 2 - 1)
    v_lo, v_hi = 0.35, 1.65
    tau = r_tot * c0

    def rhs(t, y, vs):
        i = (vs - v_of_q(y[0])) / r_tot
        return [i, i * i * r_tot]

    e = []
    for q0, vs in ((q_of_v(v_lo), v_hi), (q_of_v(v_hi), v_lo)):
        sol = solve_ivp(rhs, (0, 60 * tau), [q0, 0.0], args=(vs,), method="Radau", rtol=1e-11, atol=[1e-30, 1e-30])
        e.append(sol.y[1, -1])
    jt = _jt()
    c_eff = np.zeros((1, 1))
    el.junction_swing_capacitance(np.array([[v_lo]]), np.array([[v_hi]]), jt.v_reverse_v, jt.capacitance_f_per_m,
                                  np.array([[1.0]]), np.array([[L_RING]]), c_eff)
    assert 0.5 * (e[0] + e[1]) == pytest.approx(0.5 * c_eff[0, 0] * (v_hi - v_lo) ** 2, rel=1e-4)
    assert abs(e[0] - e[1]) / e[0] > 0.01                     # up and down differ for a nonlinear C; the mean does not


def test_junction_loader_units_and_validation(tmp_path):
    jt = _jt()
    base = {"schema": "junction/1", "method": "analytic", "v_reverse_v": jt.v_reverse_v.tolist(),
            "dneff": jt.dneff[0].tolist(), "alpha_fca_per_m": jt.alpha_fca_per_m[0].tolist(),
            "capacitance_f_per_m": jt.capacitance_f_per_m[0].tolist()}
    path = tmp_path / "junction_x.json"
    # r_series_ohm is in ohm (07 P1-15 suffix grammar): it needs the circumference it refers to
    path.write_text(json.dumps(dict(base, r_series_ohm=60.0)))
    with pytest.raises(ValueError, match="junction_length_m"):
        el.load_junction_table(path)
    path.write_text(json.dumps(dict(base, r_series_ohm=60.0, junction_length_m=L_RING)))
    out = el.load_junction_table(path).eval(np.array([[1.0]]), 0.5, L_RING)
    assert out["r_s_ohm"][0, 0] == pytest.approx(60.0 / 0.5, rel=1e-13)
    # an absolute n_eff in place of dneff relative to 0 V is rejected
    path.write_text(json.dumps(dict(base, dneff=(2.5 + jt.dneff[0]).tolist(), r_series_ohm_m=1e-3)))
    with pytest.raises(ValueError, match="relative to 0 V"):
        el.load_junction_table(path)
    # a grid narrower than -0.5 .. 4.0 V warns
    g = np.linspace(0.0, 2.0, 21)
    path.write_text(json.dumps({"schema": "junction/1", "v_reverse_v": g.tolist(), "dneff": (1e-4 * g).tolist(),
                                "alpha_fca_per_m": [800.0] * 21, "capacitance_f_per_m": [5e-10] * 21,
                                "r_series_ohm_m": 1e-3}))
    with pytest.warns(UserWarning, match="does not span"):
        el.load_junction_table(path)


def test_junction_nan_voltage_is_flagged_not_read_past_grid():
    out = _jt().eval(np.array([[np.nan, 1.0]]), 1.0, L_RING)
    assert np.isnan(out["dneff"][0, 0]) and out["flags"][0, 0] & el.FLAG_GRID_CLIP
    assert np.isfinite(out["dneff"][0, 1]) and out["flags"][0, 1] == 0


def test_junction_dqi_dv_derived_helper():
    # analytic table: d(alpha_fca)/dV = -(4 pi q/lambda) d(dneff)/dV, so dQi/Qi/dV = f (4 pi q/lambda)(n_g/lambda)
    # * eff_full / alpha_i, with eff_full the full-circumference d(dlambda)/dV
    jt, fill, a_i = _jt(), 0.6, 2900.0
    eff_ring = el.junction_efficiency_m_per_v(jt, 1.0, fill, LAM, NG)
    want = 4 * math.pi * 0.06 / LAM * NG / LAM * eff_ring / a_i
    assert el.junction_dqi_dv_frac_per_v(jt, 1.0, fill, a_i) == pytest.approx(want, rel=1e-6)


def test_implicit_tcr_with_thermal_observer_split():
    # Interface with mrm.plant.thermal (07 P1-2): observer_split_b gives (pre, g) for the heater_resistor observer;
    # heater_power_row(t_pre = T_base + pre, g_self = g) then makes the power consistent with the END-of-tick
    # resistance exactly, and the loop converges to the static operating point.
    from mrm.plant import thermal as th
    taus = np.array([0.3e-6, 3.2e-6, 13.0e-6, 1.0e-6])            # s; three heater poles + one other-source lag
    res = np.array([1.0e3, 3.5e3, 4.54e3, 500.0])                  # K/W; heater sum = R_th,H = 9.04 K/mW
    a, c0, c1 = th.foh_coefficients(taus, res, DT)
    lag_src = np.array([0, 0, 0, 1], np.int64)
    obs_ptr = np.array([0, 4], np.int64)
    scale = np.ones(2)
    x = np.zeros(4)
    t_base = 298.15
    p_other = 0.002                                                # W on source 1: +1 K at steady state
    code = 50000
    bank = el.HeaterBank(_heater(), code)
    for k in range(400):
        u = np.array([0.0, p_other])
        pre, g = th.observer_split_b(x, a, c0, c1, lag_src, obs_ptr, 0, 0, u, u, scale)
        out = bank.step(k, t_heater_pre_k=t_base + pre, g_self_k_per_w=g)
        u[0] = out.p_w[0, 0]
        th.thermal_step_b(x, a, c0, c1, lag_src, u, u, scale)
        t_end = t_base + x.sum()
        assert out.r_ohm[0, 0] == pytest.approx(R_COLD_OHM * (1 + TCR_PER_K * (t_end - 298.15)), rel=1e-12)
    stat = el.heater_steady_state(np.array([code]), law="voltage", v_full_scale_v=1.6, r_cold_ohm=R_COLD_OHM,
                                  tcr_per_k=TCR_PER_K, r_th_heater_k_per_w=res[:3].sum(),
                                  t_base_minus_ref_k=res[3] * p_other)
    assert out.p_w[0, 0] == pytest.approx(stat["p_w"][0], rel=1e-9)


@pytest.mark.parametrize("dt", [1e-6, 1e-7])
def test_dac_latency_in_seconds_is_invariant_to_dt(dt):
    # 07 P0-6: latencies are owned in seconds and converted to whole ticks once; the delay in seconds is the same
    lat_s, t_cmd_s = 3e-6, 2e-6
    prm = el.HeaterDriverParams.build(1, 1, dt, law="voltage", v_full_scale_v=1.6, r_cold_ohm=50.0, tcr_per_k=0.0,
                                      latency_s=lat_s)
    bank = el.HeaterBank(prm, 0)
    k_cmd = int(round(t_cmd_s / dt))
    first = next(k for k in range(int(round(10e-6 / dt)))
                 if bank.step(k, cmd=np.array([[30000]]) if k == k_cmd else None).p_w[0, 0] > 0)
    assert first * dt == pytest.approx(t_cmd_s + lat_s, rel=1e-12)
