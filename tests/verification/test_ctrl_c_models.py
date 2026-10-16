"""mrm.control.models.linearize: the small-signal model of the heater lock loop from the nominal design (07 P1-1).

Checks:
1. API: unpacks as (A, B, C, D, delay_ticks); knowledge rule (``model_nominal`` required); nominal only (the same
   model whatever world the plant samples).
2. Self-heating in the linear model (07 P0-1): the heater-to-detuning DC gain is the self-heating-free gain divided by
   1 - Z_opt(0) dP_abs/dT_R = 1 + G|P'| (about 6 at the g26 OMA point).
3. Against the nonlinear plant (tests/verification/test_plant_wiring.py::ReferencePlant, nominal world, warm start at
   the same operating point): a 60-code heater step (about 3 pm) within 0.2 pm of the plant at every tick from 3 us
   on and within 1 % at steady state; a heater dither at 2, 10 and 25 kHz within 3 % in amplitude and 4 degrees in
   phase for both the detuning (Truth) and the monitor code (the ADC samples at their window centres).
4. Delay bookkeeping: tau_u = controller latency + heater DAC latency; tau_m from the sample pattern; the exact
   fractional-delay discretization reproduces the continuous step response with delay at the sample instants.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mrm.control import models as M
from mrm.control.api import ControllerSpec

from test_ctrl_c_harness import build_rig, hold, load_cfg, realize


@pytest.fixture(scope="module")
def cfg():
    return load_cfg()


@pytest.fixture(scope="module")
def lm(cfg):
    return M.linearize(cfg, M.OperatingPoint(ring=0), latency_ticks=0)


def test_api_shapes_and_knowledge(cfg, lm):
    A, B, C, D, delay = lm
    n = A.shape[0]
    assert B.shape == (n, 2) and C.shape == (2, n) and D.shape == (2, 2)
    assert isinstance(delay, int) and delay == round(lm.delay_exact_ticks)
    assert lm.u_names == ("heater_code", "ring_temperature_disturbance_k")
    assert lm.Ts == pytest.approx(1e-5)
    assert np.all(np.abs(np.linalg.eigvals(A)) < 1.0)                 # the self-locked operating point is stable
    with pytest.raises(M.KnowledgeError):
        M.linearize(cfg, knowledge=["line_grid", "ring_to_line"])
    M.check_knowledge(["line_grid", M.KNOWLEDGE_TOKEN])
    # nominal only: the same model as linearizing the nominal PlantParams explicitly, and independent of any
    # sampled world the experiment draws (the controller never receives one)
    pp_nom = M.nominal_plant_params(cfg)
    lm2 = M.linearize_bank(cfg, M.OperatingPoint(), latency_ticks=0, pp=pp_nom, b=0)[0]
    np.testing.assert_allclose(lm2.A, lm.A, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(lm2.dc_gain(), lm.dc_gain(), rtol=1e-12)
    # the operating point: stable side, open-loop stable, monitor inside the ADC range
    assert lm.detuning0_m > 0 and lm.op.loop_gain_opt[0] < 1.0
    assert 0 < lm.y0 < 65535 and 0 < lm.u0 < 65535


def test_selfheating_divides_the_heater_gain(cfg, lm):
    """07 P0-1: the optical feedback divides the heater-power-to-detuning DC gain by 1 - Z_opt(0) a_R = 1 + G|P'|.
    Exactly so with the heater TCR loop frozen (k_T = 0); with it, the absorbed power also warms the heater resistor
    and the closed form below (DC gains of the ring's thermal block) holds to rounding."""
    from dataclasses import replace
    sp = lm.op
    pp = M.nominal_plant_params(cfg)

    def gain(spx):
        return M._model_for_ring(spx, pp, 0, 0, lm.Ts, lm.tau_u_ticks, lm.tau_m_ticks, lm.dt_s).dc_gain()[1, 0]
    z = np.zeros_like(sp.dpabs_dtr)
    no_tcr = replace(sp, dp_dth_heater=z, dpabs_dth=z)
    ratio = gain(replace(no_tcr, dpabs_dtr=z)) / gain(no_tcr)
    factor = 1.0 - sp.z_opt_k_per_w[0] * sp.dpabs_dtr[0]
    assert ratio == pytest.approx(factor, rel=1e-9)
    assert lm.selfheat_factor == pytest.approx(factor)
    assert 3.0 < factor < 10.0                                      # g26 nominal at the OMA point: about 6
    # full model: delta/code = s Z_RH k_u / [(1 - Z_RW a_R)(1 - k_T Z_HH) - k_T Z_HW a_R Z_RH]   (a_H = 0 here)
    A_th, B_th, C_th, _ = M._ring_thermal_block(pp, 0, 0)
    Z = -C_th @ np.linalg.solve(A_th, B_th)                         # [[Z_RH, Z_RW], [Z_HH, Z_HW]] K/W
    (zrh, zrw), (zhh, zhw) = Z
    a_r, k_u, k_t, s = sp.dpabs_dtr[0], sp.dp_dcode[0], sp.dp_dth_heater[0], sp.s_m_per_k[0]
    want = s * zrh * k_u / ((1 - zrw * a_r) * (1 - k_t * zhh) - k_t * zhw * a_r * zrh)
    assert gain(replace(sp, dpabs_dth=z)) == pytest.approx(want, rel=1e-9)
    assert zrw == pytest.approx(sp.z_opt_k_per_w[0], rel=1e-12) and zrh == pytest.approx(sp.z_heater_k_per_w[0],
                                                                                          rel=1e-12)
    # signs: heating moves the resonance red, away from the laser, and the drop code falls
    assert lm.dc_gain()[0, 0] < 0 and lm.dc_gain()[1, 0] > 0


def test_step_response_matches_the_reference_plant(cfg):
    rig = build_rig(cfg)
    lm = M.linearize_bank(cfg, M.OperatingPoint(), latency_ticks=0)[0]
    base = hold(rig, 200)
    d0 = base[-1, 0, 0]
    cmd = np.array(rig.plant.heater.applied_codes)
    cmd[0, 0] += 60
    det = hold(rig, 400, cmd=cmd)
    resp = (det[:, 0, 0] - d0) * 1e12
    # the command latches at the first tick and acts from the next one (heater DAC latency 1 tick); det[i] is the
    # state at the end of tick i, i.e. i ticks after the code reached the heater
    t = np.arange(400) * 1e-6
    mod = lm.step_response(t, 0, with_delay=False)[:, 1] * 60 * 1e12
    err = np.abs(resp[3:] - mod[3:])
    print(f"\nstep: plant {resp[-1]:.3f} pm, model {mod[-1]:.3f} pm, max |err| after 3 us {err.max():.3f} pm")
    assert err.max() < 0.2
    assert resp[-1] == pytest.approx(mod[-1], rel=0.01)


def test_frequency_response_matches_the_reference_plant(cfg):
    rig = build_rig(cfg)
    lm = M.linearize_bank(cfg, M.OperatingPoint(), latency_ticks=0)[0]
    hold(rig, 100)
    code0 = np.array(rig.plant.heater.applied_codes)
    amp = 40
    freqs = {0: 2000.0, 2: 10000.0, 4: 25000.0}                     # non-adjacent rings, crosstalk negligible
    nT = 2000
    det = np.zeros((nT, rig.N))
    samples = {n: ([], []) for n in freqs}
    k0 = rig.k
    for i in range(nT):
        k = rig.k
        if k % rig.spec.period_ticks == 0:
            s, _ = rig.plant.release(k)
            for n in freqs:
                c = int(s["tx.monitor.count"][0, n])
                samples[n][0].append(s["tx.monitor.sample_ticks"][0, n, :c].astype(float))
                samples[n][1].append(s["tx.monitor.samples"][0, n, :c].astype(float))
        cmd = code0.copy()
        for n, f in freqs.items():
            cmd[0, n] = code0[0, n] + int(round(amp * math.sin(2 * math.pi * f * (k - k0) * 1e-6)))
        rig.plant.tick(k, heater_cmd=cmd)
        rig.k += 1
        det[i] = rig.detuning_m()[0]
    W = rig.spec.adc.integration_window_ticks
    for n, f in freqs.items():
        w = 2 * math.pi * f
        # input: code sampled at tick k - k0 = i acts on [i + 1, i + 2) (DAC latency 1): continuous equivalent is the
        # ZOH of the sampled sine, i.e. a half-tick delay and a sinc
        zoh = np.exp(-1j * w * 1.5e-6) * np.sinc(f * 1e-6)
        t = (np.arange(500, nT) + 1) * 1e-6
        X = np.stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)], 1)
        c, *_ = np.linalg.lstsq(X, det[500:, n], rcond=None)
        hp = complex(c[0], c[1]) * 1e12
        hm = lm.frequency_response([f], with_delay=False)[0, 1, 0] * amp * zoh * 1e12
        tt = np.concatenate(samples[n][0])
        vv = np.concatenate(samples[n][1])
        keep = tt > k0 + 500
        tc = (tt[keep] - k0 - 0.5 * (W - 1)) * 1e-6                  # window centre
        X = np.stack([np.sin(w * tc), np.cos(w * tc), np.ones_like(tc)], 1)
        c2, *_ = np.linalg.lstsq(X, vv[keep], rcond=None)
        yp = complex(c2[0], c2[1])
        ym = lm.frequency_response([f], with_delay=False)[0, 0, 0] * amp * zoh * np.sinc(f * W * 1e-6)
        print(f"\n{f / 1e3:.0f} kHz: detuning plant {abs(hp):.3f} pm {math.degrees(np.angle(hp)):.1f} deg, model "
              f"{abs(hm):.3f} pm {math.degrees(np.angle(hm)):.1f} deg; monitor plant {abs(yp):.1f} codes "
              f"{math.degrees(np.angle(yp)):.1f} deg, model {abs(ym):.1f} {math.degrees(np.angle(ym)):.1f} deg")
        for p_, m_ in ((hp, hm), (yp, ym)):
            assert abs(abs(p_) / abs(m_) - 1) < 0.03
            assert abs(math.degrees(np.angle(p_ / m_))) < 4.0


def test_delay_bookkeeping(cfg):
    spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
    lat = 1
    lm = M.linearize(cfg, latency_ticks=lat)
    assert lm.tau_u_ticks == lat + spec.heater_dac.latency_ticks
    # g26: controller period 10, ADC period 4, window 2, ADC latency 2: edges alternate 3 and 2 samples, whose mean
    # age (window centre s - (W - 1)/2) is 6.5 ticks at both kinds of edge; the newest sample is 2.5 / 4.5 old
    assert M.measurement_age_ticks(10, 4, 2, 2) == pytest.approx(6.5)
    assert M.measurement_age_ticks(10, 4, 2, 2, mode="latest") == pytest.approx(3.5)
    assert lm.tau_m_ticks == pytest.approx(6.5)
    lmn = M.linearize_bank(cfg, M.OperatingPoint(), latency_ticks=lat, measurement="none")[0]
    assert lmn.tau_m_ticks == 0.0
    # exact fractional delay: the delayed discrete model at the sample instants = the continuous step with delay
    Pz, Gu, Gd, C, Dd, nd = lm.delayed()
    z = np.zeros(Pz.shape[0])
    ys = []
    for k in range(30):
        ys.append(C[1] @ z)
        z = Pz @ z + Gu * 1.0
    t = np.arange(30) * lm.Ts
    ref = lm.step_response(t, 0, with_delay=True)[:, 1]
    np.testing.assert_allclose(ys, ref, rtol=1e-9, atol=1e-9 * np.max(np.abs(ref)))
    assert nd == int(math.floor(lm.delay_exact_ticks * lm.dt_s / lm.Ts)) + 1


def test_adc_mux_gives_each_ring_its_own_measurement_delay():
    """07 P0-7 adc_mux: with channels_per_adc = 4 (round_robin) the rings sharing a converter are sampled at phases
    0..3 ticks, so the age of the samples an edge consumes differs per ring. linearize uses the plant's own phase rule
    (MonitorHw.phase) per ring, and the LQG design pads the delay lines to one state size. Four rings take turns on
    one integrating converter, so their windows must fit in the 4-tick ADC period: 1 us each (the g26 2 us window
    would overlap them; physics_sensing review)."""
    from mrm.control.baselines.kalman_lqg import KalmanLQG
    from mrm.plant.sensing import MonitorHw
    cfg4 = load_cfg(overrides=["ring.monitor.adc_mux.channels_per_adc=4", "ring.monitor.integration_window_s=1e-6"])
    pp = M.nominal_plant_params(cfg4)
    phase = MonitorHw.from_plant_params(pp).phase
    assert len(set(phase.tolist())) > 1
    lms = M.linearize_bank(cfg4, M.OperatingPoint(), latency_ticks=1)
    spec = ControllerSpec.from_config(cfg4, n_batch=1, nominal=cfg4)
    W = spec.adc.integration_window_ticks
    for n, lm in enumerate(lms):
        want = M.measurement_age_ticks(spec.period_ticks, spec.adc.period_ticks, spec.adc.latency_ticks, W,
                                       adc_phase_ticks=int(phase[n]))
        assert lm.tau_m_ticks == pytest.approx(want)
    assert len({round(lm.tau_m_ticks, 9) for lm in lms}) > 1
    d = KalmanLQG().build_design(spec)                       # rings with different delays stack (padded lines)
    assert d.Phi.shape[0] == spec.n_rings
    assert max(lm.n_delay() for lm in lms) == d.nc - lms[0].Ac.shape[0]
    # padding a delay line is harmless: the same commands and the same closed loop
    from mrm.control.baselines.kalman_lqg import closed_loop_spectral_radius, design_lqg
    d1 = design_lqg(lms[:2])
    d3 = design_lqg(lms[:2], n_delay=d1.nc - lms[0].Ac.shape[0] + 2)
    assert d3.nz == d1.nz + 2
    for g in (0.5, 1.0, 3.0):
        assert closed_loop_spectral_radius(d3, 0, lms[0], g) == pytest.approx(
            closed_loop_spectral_radius(d1, 0, lms[0], g), rel=1e-6)
    Pz, Gu, *_ = lms[0].delayed(n_delay=4)
    z = np.zeros(Pz.shape[0])
    z1 = np.zeros(lms[0].delayed()[0].shape[0])
    P1, G1, _, C1, *_ = lms[0].delayed()
    C4 = lms[0].delayed(n_delay=4)[3]
    for k in range(25):                                       # identical step responses
        assert C4[1] @ z == pytest.approx(C1[1] @ z1, rel=1e-12, abs=1e-24)
        z, z1 = Pz @ z + Gu, P1 @ z1 + G1


def test_linearize_sampled_world_and_ring_lag_map(cfg):
    """The same functions run on a sampled world (as the oracle and the robustness tests use them), and the grouped
    thermal states map onto the plant's lag states (ring_block_lags)."""
    pp = realize(cfg, 3, uncertainty="lhs", variation="sample", seeds=5)
    for b in range(pp.batch):
        lm = M.linearize_bank(cfg, M.OperatingPoint(), 1e-5, pp=pp, b=b)[3]
        idx, res, src = M.ring_block_lags(pp, b, 3)
        assert idx.size == lm.Ac.shape[0] - (1 if "settle" in lm.x_names else 0)
        sysm = pp.thermal.system
        # a steady state from heater power P: the grouped heater states equal P, the waveguide states 0
        u = np.zeros((pp.batch, sysm.n_src))
        u[:, sysm.heater_src[3]] = 0.01
        x = sysm.steady_state(u)
        z = x[b, idx] / res
        np.testing.assert_allclose(z[src == 0], 0.01, rtol=1e-12)
        np.testing.assert_allclose(z[src == 1], 0.0, atol=1e-15)


def test_no_truth_imports():
    """07 P1-1 (scoped to these files): the baselines import no plant, simulator or validation module; models.py
    uses the plant's LEAF libraries and realize only to build the nominal design (inside functions), never
    mrm.plant.plant (Truth), mrm.sim or mrm.validation."""
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for f, banned in (("mrm/control/baselines/kalman_lqg.py", ("mrm.plant", "mrm.sim", "mrm.validation",
                                                              "mrm.params.realize")),
                      ("mrm/control/baselines/mpc.py", ("mrm.plant", "mrm.sim", "mrm.validation",
                                                       "mrm.params.realize")),
                      ("mrm/control/models.py", ("mrm.plant.plant", "mrm.sim", "mrm.validation"))):
        for node in ast.walk(ast.parse((root / f).read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for n in names:
                assert not n.startswith(banned), (f, n)
                assert "truth" not in n.lower(), (f, n)


def test_design_depends_on_the_nominal_design_only(cfg):
    """07 P1-1 test_spec_independent_of_seed_and_uncertainty, for the model: the same config gives the same model
    whatever batch the experiment runs, and a sampled PlantParams never enters linearize (no argument takes one)."""
    from mrm.control.baselines.kalman_lqg import KalmanLQG
    d = []
    for B in (1, 3):
        spec = ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg)
        c = KalmanLQG()
        d.append(c.build_design(spec))
    np.testing.assert_array_equal(d[0].Kf, d[1].Kf)
    np.testing.assert_array_equal(d[0].Lk, d[1].Lk)
