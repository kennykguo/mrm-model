"""Checks the bit-statistics lock (Sun et al.) on the g26 reference plant, closed loop.

Read this first
---------------
BitStatLock reads ``datapath.bitstat_above`` (how many samples of a window are above the threshold) and drives the
threshold actuator ``datapath.bitstat_threshold``. Two time-multiplexed threshold trackers find the medians of the
two data levels (the eye). The inner loop holds the eye centre at a set-point. By default (mode='ratio') the outer
loop holds the eye ratio captured after settling, which starts at the eye maximum (alpha = 3/4). CtrlBPlant
(test_ctrl_b_plant.py) wires the threshold to the sampler.

The scenarios here: a steady lock at the eye maximum (OMA penalty against the static optimum and against the
half-maximum monitor lock), and squelch (no data: the bit statistics freeze and the lock holds). The ones-density
steps are in test_ctrl_b_bitstat_density.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from mrm.control import registry
from mrm.control.api import ControllerSpec, validate_telemetry
from mrm.control.baselines.bitstat_lock import BitStatLock, BitStatParams
from mrm.control.baselines.ramp_acquire import ST_LOCK
from mrm.plant import optics
from mrm.scenarios import library

from test_ctrl_b_plant import CtrlBPlant, load_g26, make_spec, realize, run_closed_loop

KNOW = ["line_grid", "ring_to_line", "heater_nominal", "optics_nominal", "dlambda_dt_nominal"]
LINE = 7                                # ring 0's first resonance from blue in nominal g26


@pytest.fixture(scope="module")
def nominal():
    cfg = load_g26(knowledge=KNOW)
    pp = realize(cfg, 1, nominal=True)
    spec = make_spec(cfg, 1)
    return cfg, pp, spec


def test_registered_and_needs_the_threshold(nominal):
    cfg, _, spec = nominal
    c = registry.create("bitstat_lock", {"po_step": 0.02})
    assert isinstance(c, BitStatLock) and c.params.po_step == 0.02
    c.reset(spec)
    validate_telemetry(c.telemetry(), spec)
    cfg2 = load_g26(overrides=["controller_interface.bind.datapath.bitstat_threshold=scenario"], knowledge=KNOW)
    spec2 = make_spec(cfg2, 1)
    with pytest.raises(ValueError, match="bitstat_threshold"):
        BitStatLock().reset(spec2)


def test_threshold_trackers_find_the_level_medians(nominal):
    """Unit test of Sun's statistic in threshold form: with the window counts drawn as the plant draws them
    (Binomial(n1, p1) + Binomial(n0, p0), mrm.plant.datapath), the two time-multiplexed sign-adaptive trackers
    converge to the class medians mu_hi, mu_lo from a start 15 % of the amplitude off."""
    _, _, spec = nominal
    c = BitStatLock(rings=[0])
    c.reset(spec)
    rng = np.random.default_rng(3)
    mu_hi, mu_lo, s = 34000.0, 24000.0, 700.0
    W, amp = c.W, 40000.0
    c.t_hi[:] = 30000.0
    c.t_lo[:] = 27000.0
    c.s_hi[:] = c.s_lo[:] = 0.02 * amp
    sel = np.zeros((1, spec.n_rings), bool)
    sel[0, 0] = True
    from scipy.stats import norm
    for k in range(400):
        which = k % 2
        t = c.t_hi if which == 0 else c.t_lo
        p_hi = 1.0 - norm.cdf((t[0, 0] + 0.5 - mu_hi) / s)
        p_lo = 1.0 - norm.cdf((t[0, 0] + 0.5 - mu_lo) / s)
        above = rng.binomial(int(W / 2), p_hi) + rng.binomial(int(W / 2), p_lo)
        target = 0.25 * W if which == 0 else 0.75 * W
        c._update_tracker(sel, which, np.full((1, spec.n_rings), float(above)), target, amp)
    assert abs(c.t_hi[0, 0] - mu_hi) < 0.5 * s
    assert abs(c.t_lo[0, 0] - mu_lo) < 0.5 * s


def _truth_levels(plant):
    """Truth: the two data levels at the monitor through the bit-statistics sampler (codes) (tests only)."""
    last = plant.last
    mu1, mu0, _, _ = plant.mon.level_codes(
        last["out_r"][:, optics.RING_OUTPUT_ROW_LEVEL_ONE_TOTAL_CLOCKWISE_DROP_POWER],
        last["out_r"][:, optics.RING_OUTPUT_ROW_LEVEL_ZERO_TOTAL_CLOCKWISE_DROP_POWER],
        0.0,
        None,
        None,
        None,
        tau_s=plant.dp.hw.bitstat_tau_s,
        noise_a_per_rt_hz=plant.dp.hw.bitstat_noise_a_per_rt_hz,
        rin_per_hz=np.asarray(plant.pp.laser.rin_per_hz).reshape(-1),
    )
    return mu1, mu0


@pytest.fixture(scope="module")
def eye_run(nominal):
    cfg, pp, spec = nominal
    plant = CtrlBPlant(pp, cfg, spec)
    plant.assignment[0, 0] = LINE
    ctrl = BitStatLock(rings=[0])
    rec, val = run_closed_loop(plant, ctrl, 3500)
    return plant, ctrl, rec, val


def test_eye_maximum_lock(eye_run):
    plant, ctrl, rec, val = eye_run
    f = ctrl.fsm
    assert f.state[0, 0] == ST_LOCK and plant.truth()["nearest_line"][0, 0] == LINE
    asg = np.full((1, plant.N), -1)
    asg[0, 0] = LINE
    pen, _ = plant.oma_penalty_db(assignment=asg)
    # the eye maximum is the OMA maximum (same derivative of the Lorentzian): a much smaller penalty than the
    # half-maximum monitor lock (0.68 dB in test_ctrl_b_ramp.py)
    assert pen[0, 0] < 0.25, pen
    # the eye estimate (threshold trackers) against Truth's level codes
    mu1, mu0 = _truth_levels(plant)
    d_true = abs(mu1[0, 0] - mu0[0, 0])
    d_est = rec.telemetry[-1]["eye_codes"][0, 0]
    assert abs(d_est - d_true) < 0.1 * d_true, (d_est, d_true)
    # upper level = zeros on the red side (one_is_red convention)
    assert mu0[0, 0] > mu1[0, 0]
    d = rec.det_m[:, 0, 0] * 1e12
    tail = d[rec.ticks > 2000]
    print(f"\nbitstat lock: OMA penalty {pen[0, 0]:.3f} dB, detuning {tail.mean():.1f} pm (wander "
          f"{np.ptp(tail):.1f} pm p-p), eye estimate {d_est:.0f} vs Truth {d_true:.0f} codes, alpha "
          f"{ctrl.alpha_eye[0, 0]:.3f}")


def test_squelch_freezes_and_holds(nominal):
    """Squelch (B1: swing to 0, average kept) from 2.5 ms to 4 ms: no eye, so the statistic and the eye P&O freeze;
    the inner lock holds; after the squelch the eye loop resumes."""
    cfg, pp, spec = nominal
    sc = library.squelch(t_on_s=2.5e-3, t_off_s=4.0e-3, rings=[0])
    plant = CtrlBPlant(pp, cfg, spec, scenario=sc)
    plant.assignment[0, 0] = LINE
    ctrl = BitStatLock(rings=[0])
    rec, _ = run_closed_loop(plant, ctrl, 5000)
    edges = np.array(rec.edge_ticks)
    ok = np.array([t["data_ok"][0, 0] for t in rec.telemetry])
    st = np.array([t["state"][0, 0] for t in rec.telemetry])
    alpha = np.array([t["alpha_eye"][0, 0] for t in rec.telemetry])
    in_sq = (edges > 3000) & (edges < 4000)               # the trackers need ~0.3 ms to collapse
    assert not np.any(ok[in_sq])
    assert np.ptp(alpha[in_sq]) == 0.0                             # P&O frozen
    assert np.all(st[edges > 1500] == ST_LOCK)                     # never lost
    assert np.any(ok[edges > 4600])                                # resumes
    d = rec.det_m[:, 0, 0] * 1e12
    before = d[(rec.ticks > 2000) & (rec.ticks < 2500)].mean()
    during = d[(rec.ticks > 3000) & (rec.ticks < 4000)]
    print(f"\nsquelch: detuning {before:.1f} pm before, {during.min():.1f}..{during.max():.1f} pm during")
    assert np.max(np.abs(during - before)) < 40.0


def _locked_bitstat(spec, latency_ticks=0, mode="ratio"):
    c = BitStatLock(rings=[0], mode=mode)
    c.latency_ticks = latency_ticks
    c.reset(spec)
    f = c.fsm
    f.state[:] = ST_LOCK
    f.y_pk[:], f.y_bg[:], f.setpoint[:], f.ki[:] = 40000.0, 5000.0, 22500.0, 0.01
    c.started, c.t0 = True, 0
    return c


def test_threshold_pairing_follows_the_controller_latency(nominal):
    """The threshold sent at edge j latches at j + latency_ticks. The window released at the first edge after the
    latch uses it (the event order is: release, then latch). With latency 0 the window of edge j + 1 uses it; with a
    latency of one period the window of edge j + 2. The tracker update must pair each window with the tracker that
    set its threshold, else the two time-multiplexed trackers chase each other's targets."""
    from test_ctrl_b_ramp import synthetic_obs
    _, _, spec = nominal
    B, N = spec.n_batch, spec.n_rings
    for lat, first_update in ((0, 2), (spec.period_ticks, 3)):
        c = _locked_bitstat(spec, latency_ticks=lat)
        assert c.n_hist == 1 + lat // spec.period_ticks
        hi0 = None
        for j in range(1, 6):
            obs = synthetic_obs(spec, j * spec.period_ticks, y=22000.0,
                                **{"datapath.bitstat_above": np.full((B, N), c.W),        # every sample above
                                   "tx.ones_count": np.full((B, N), 500), "tx.bits": np.full((B, N), 1000)})
            c.step(obs)
            if j == 1:
                hi0, lo0 = float(c.t_hi[0, 0]), float(c.t_lo[0, 0])
            elif j < first_update:
                assert c.t_hi[0, 0] == hi0 and c.t_lo[0, 0] == lo0, (lat, j)
            elif j == first_update:
                # the first window measured with a known threshold is the upper tracker's (sent first): only t_hi
                assert c.t_hi[0, 0] > hi0 and c.t_lo[0, 0] == lo0, (lat, j)


def test_ratio_mode_keeps_alpha_above_the_ratio_maximum(nominal):
    """R = 2 x dx / (1 + x^2) peaks at x = 1 (alpha = 1/2); below it an integral law on R runs away. A ratio that
    stays below its reference (a swing loss beyond ~13 %) drives alpha down to the floor 0.55, not to alpha_min."""
    _, _, spec = nominal
    c = _locked_bitstat(spec)
    assert c.alpha_floor == pytest.approx(0.55)
    assert _locked_bitstat(spec, mode="po").alpha_floor == pytest.approx(BitStatParams().alpha_min)
