"""Checks the bias-dither lock-in controller (mrm/control/baselines/bias_dither.py) and its per-sample fast kernel.

Read this first
---------------
BiasDither dithers the junction bias (tx.bias) and demodulates the monitor signal per sample in a fast kernel. The
heater then integrates the lock-in error. The tests check:

- the fast kernel (fed one released sample at a time) and a plain-Python reference demodulator give bit-identical
  per-cycle values (S1, S2, DC, r);
- on a toy ring where the bias moves the detuning, the heater holds the hand-off point against a drift;
- on g26 nominal (ReferencePlant), the honest result: the bias barely moves the resonance there, so the lock is weak.
  The CTRL_A_METRIC lines report it (run with -s).
"""
from __future__ import annotations

import numpy as np
import pytest

import test_ctrl_a_harness as harness
from mrm.control import registry
from mrm.control.api import ActionValidator, fast_act_size, fast_obs_size
from mrm.control.baselines import bias_dither
from mrm.control.baselines.bias_dither import BiasDither, bias_fast, demod_reference


def report(tag, m: harness.LockMetrics, **extra):
    s = m.summary()
    s.update(extra)
    print("CTRL_A_METRIC", tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})


def test_fast_kernel_equals_python_reference_bit_for_bit():
    """Identical per-cycle accumulators from the fast kernel (fed one released sample at a time) and from the
    plain-Python demodulator, for every ring and both start phases."""
    spec = harness.toy_spec(1, 3, bias=True)
    ctrl = BiasDither(samples_per_phase=3, blank_samples=1, dither_codes=200.0, stagger=True)
    ctrl.reset(spec)
    p_i, p_f, s_i, s_f = (x[0] for x in ctrl.state)
    N = spec.n_rings
    rng = np.random.default_rng(3)
    codes = rng.integers(20000, 40000, size=(400, N))
    fobs = np.zeros(fast_obs_size(N), np.int64)
    fact = np.zeros(fast_act_size(N), np.int64)
    got = [[] for _ in range(N)]
    R = bias_dither._FS_HIST + int(p_i[bias_dither._IP_J])
    # fast-kernel observation: [tick, N, new flag[N], monitor code[N], sample tick[N]] (api.fast_obs_size)
    for t in range(codes.shape[0]):
        fobs[0], fobs[1] = 4 * t, N
        fobs[2:2 + N] = 1
        fobs[2 + N:2 + 2 * N] = codes[t]
        fobs[2 + 2 * N:2 + 3 * N] = 4 * t
        before = [int(s_i[1 + bias_dither._IS_PER_RING * i + bias_dither._IS_CYCLES]) for i in range(N)]
        bias_fast(p_i, p_f, s_i, s_f, fobs, fact)
        for i in range(N):
            if int(s_i[1 + bias_dither._IS_PER_RING * i + bias_dither._IS_CYCLES]) != before[i]:
                b = i * R
                got[i].append((s_f[b + bias_dither._FS_S1], s_f[b + bias_dither._FS_S2],
                               s_f[b + bias_dither._FS_Y0], s_f[b + bias_dither._FS_RATIO]))
    for i in range(N):
        want = demod_reference(codes[:, i], 3, 1, 200.0, start_phase=i % 4)
        assert len(want) == len(got[i]) > 20
        for g, w in zip(got[i], want):
            assert g == w, (i, g, w)                       # bit-identical, not approximately equal


def toy_fast_loop(ctrl, spec, toy: harness.ToyRings, n_steps: int, m: int = 3, drift=None):
    """A toy closed loop with the fast kernel: m samples per controller step, each released to the fast kernel,
    whose bias offset acts on the next sample (as the plant's DAC latency would)."""
    v = ActionValidator(spec)
    N = spec.n_rings
    fobs = np.zeros((spec.n_batch, fast_obs_size(N)), np.int64)
    fact = np.zeros((spec.n_batch, fast_act_size(N)), np.int64)
    off = np.zeros((spec.n_batch, N), np.int64)
    xs = []
    for k in range(n_steps):
        if drift is not None:
            drift(toy, k)
        blocks = np.zeros((spec.n_batch, N, spec.samples_per_step))
        for j in range(m):
            bias_save = toy.bias.copy()
            toy.bias = toy.bias + off
            toy.advance()
            toy.bias = bias_save
            y = toy.M / (1.0 + toy.x ** 2)
            code = np.rint(y + toy.noise * np.array([[toy._rand() for _ in range(N)] for _ in range(spec.n_batch)]))
            blocks[..., j] = code
            p_i, p_f, s_i, s_f = ctrl.state
            for b in range(spec.n_batch):
                fobs[b, 0], fobs[b, 1] = toy.tick, N
                fobs[b, 2:2 + N] = 1
                fobs[b, 2 + N:2 + 2 * N] = code[b]
                bias_fast(p_i[b], p_f[b], s_i[b], s_f[b], fobs[b], fact[b])
            off = fact[:, N:].copy()                        # fast action: [heater offset[N], bias offset[N]]
            toy.tick += 3
        obs = toy.observation()
        obs.sensors["tx.monitor.samples"]                      # (read-only views: rebuild with our block)
        sens = {k2: np.array(v2) for k2, v2 in obs.sensors.items()}
        sens["tx.monitor.samples"][..., :m] = blocks
        sens["tx.monitor.count"][...] = m
        from mrm.control.api import Observation
        act = v.validate(ctrl.step(Observation(toy.tick, spec.tick_s, sens, dict(obs.new))) or {})
        toy.heater = act["tx.heater"].astype(np.int64)
        toy.bias = act["tx.bias"].astype(np.int64)
        xs.append(toy.x.copy())
    return np.array(xs)


@pytest.mark.parametrize("mode", ["ratio", "inflection"])
def test_toy_lock_holds_against_drift(mode):
    """Toy ring where the bias shifts the detuning (bias_gain): HOLD, heater identification, LOCK; a slow drift of the
    line is followed by the heater (ratio mode holds the hand-off detuning, inflection mode goes to x = 1/sqrt(3))."""
    spec = harness.toy_spec(1, 2, bias=True)
    u0 = np.array([[30000.0, 31000.0]])
    # the heater alone gives (u - u0)/W = 0.8; the bias adds bias_gain x 1191 = 596 codes, so x ~ 2.2 at hand-off
    # (g = 0.5: no bistability)
    init = u0[0] + 450.0 * 0.8 - 450 * 0.0
    toy = harness.ToyRings(spec, u0=u0, g=0.5, W=450.0, noise=1.0, bias_gain=0.5, u_init=init, bias_init=1191)
    ctrl = BiasDither(mode=mode, dither_codes=120.0, samples_per_phase=2, hold_cycles=8, id_cycles=8,
                      id_codes=30.0, gain=0.5, init_codes=init)
    ctrl.reset(spec)
    x_hand = toy.x.copy()

    def drift(t, k):
        if k >= 400:
            t.u0 += 2.0                                  # the line moves red relative to the ring: the heater follows
    xs = toy_fast_loop(ctrl, spec, toy, 1200, drift=drift)
    tel = ctrl.telemetry()
    assert np.all(tel["state"] == bias_dither.B_LOCK), tel["state"]
    final = xs[-100:].mean(axis=0)
    if mode == "ratio":
        np.testing.assert_allclose(final, x_hand, atol=0.08)
    else:
        np.testing.assert_allclose(final, 1.0 / np.sqrt(3.0), atol=0.1)
    assert np.all(tel["g_dc_per_code"] < 0.0)


def test_g26_bias_dither_lock(capsys):
    """g26 nominal, bias bound to the controller at 1.8 V. Measured: how well the hand-off detuning is held. In this
    plant a 0.2 V bias step moves the resonance only ~0.1-0.5 pm: the change of free-carrier-absorption self-heating
    cancels most of the electro-optic shift. The loss change also modulates the monitor more than the detuning does.
    So the lock-in sensitivity is small (g_r ~ -4e-6 per heater code)."""
    cfg, pp, spec, plant = harness.placed(bias_v=1.8, delta_pm=128.0)
    harness.use_fast_monitor(plant)
    ctrl = BiasDither()
    loop = harness.ClosedLoop(plant, spec, ctrl)
    rec = loop.run(6000)
    tel = ctrl.telemetry()
    assert np.all(tel["state"] == bias_dither.B_LOCK), tel["state"]
    d = rec["delta_pm"].astype(float)
    m = harness.lock_metrics(rec, slice(400, 900), slice(2500, None))
    assert np.all(m.max_err_pm < 15.0), m.max_err_pm               # held near the hand-off detuning
    report("bias_dither ratio g26 nominal steady (1.8 V)", m, g_ratio_per_code=float(np.mean(tel["g_ratio_per_code"])),
           ratio_set=float(np.mean(tel["ratio_set"])), dither_cycles=float(np.mean(tel["cycles"])))


def test_requires_bias_binding_and_checks():
    spec = harness.toy_spec(1, 2)            # the scenario owns tx.bias here
    with pytest.raises(ValueError, match="bind.tx.bias"):
        BiasDither().reset(spec)
    with pytest.raises(ValueError):
        BiasDither(mode="peak")
    with pytest.raises(TypeError):
        BiasDither(nope=1)
    with pytest.raises(ValueError, match="range"):
        BiasDither(bias_v=1.05).reset(harness.toy_spec(1, 2, bias=True))
    assert isinstance(registry.create("bias_dither", {"gain": 0.2}), BiasDither)


def test_fall_during_identification_recaptures_and_keeps_r_set():
    """Bistable toy ring (g = 3) pushed past the fold while the heater identification runs. Before the fix the fall
    was not detected in the identification states: the identification measured the cold branch (dy0/du > 0) and the
    ring ended in FAULT on the cold branch. Now the fall starts a recapture, HOLD and the identification restart, and
    the ratio lock returns to the hand-off detuning (the r_set measured at the hand-off is kept: a re-measured one
    would sit just red of the peak, next to the fold)."""
    spec = harness.toy_spec(1, 1, bias=True)
    u0 = np.array([[30000.0]])
    init = u0[0] - 300.0
    toy = harness.ToyRings(spec, u0=u0, g=3.0, W=450.0, noise=1.0, bias_gain=0.5, u_init=init, bias_init=1191)
    ctrl = BiasDither(dither_codes=120.0, samples_per_phase=2, hold_cycles=8, id_cycles=8, id_codes=30.0, gain=0.5,
                      init_codes=init)
    ctrl.reset(spec)
    x_hand = toy.x.copy()
    states = []

    def drift(t, k):
        if k == 40:
            t.u0 += 3000.0
        states.append(int(ctrl.ring_values(0, True)[0, 0]))
    xs = toy_fast_loop(ctrl, spec, toy, 1500, drift=drift)
    assert states[40] in (bias_dither.B_IDP, bias_dither.B_IDM)              # the kick fell inside the identification
    tel = ctrl.telemetry()
    assert tel["state"][0, 0] == bias_dither.B_LOCK and tel["fall_count"][0, 0] == 1
    assert tel["g_dc_per_code"][0, 0] < 0.0
    np.testing.assert_allclose(xs[-100:].mean(axis=0), x_hand, atol=0.05)
