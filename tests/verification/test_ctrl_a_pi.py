"""Checks the drop-port fraction-of-peak PI lock (mrm/control/baselines/pi_lock.py).

Read this first
---------------
PiLock first maps the drop line (MAP: it lowers the heater from the hand-off code and records the monitor until it
passes the peak), then locks the monitor at a fraction f of the peak on the red, stable side with a PI loop. The
tests run:

- the closed loop on the ReferencePlant (through test_ctrl_a_harness.py) and a bit-identical replay of the compiled
  port (PiLockCompiled);
- toy bistable rings for parity with falls and recaptures, anti-windup, capture starts and ADC clipping;
- the knowledge gate of the nominal model, and that the controller modules do not import Truth.

The fraction 0.7 (70 %) used below is the default controller hyperparameter, not a physical constant. Only the TEST
reads Truth (detuning to the assigned line, OMA). Lines that start with CTRL_A_METRIC are the reported numbers (run
with -s to see them).
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

import test_ctrl_a_harness as harness
from mrm.control import registry
from mrm.control.api import ActionValidator, Observation
from mrm.control.baselines import pi_lock
from mrm.control.baselines.pi_lock import NominalModel, PiLock, PiLockCompiled

ROOT = Path(__file__).resolve().parents[2]
BASELINE_FILES = ["pi_lock.py", "dither_lockin.py", "bias_dither.py", "extremum_seeking.py", "feedforward.py",
                  "midranging.py"]


def report(tag, m: harness.LockMetrics, **extra):
    s = m.summary()
    s.update(extra)
    print("CTRL_A_METRIC", tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


@pytest.fixture(scope="module")
def locked():
    """g26 nominal, B = 1, 8 rings handed off at +170 pm on the stable branch; PiLock maps and locks (5.5 ms)."""
    cfg, pp, spec, plant = harness.placed()
    ctrl = PiLock()
    loop = harness.ClosedLoop(plant, spec, ctrl)
    seen = []
    orig = ctrl.step

    def step(obs):
        out = orig(obs)
        seen.append((obs.tick, {k: np.array(v) for k, v in obs.sensors.items()},
                     {k: np.array(v) for k, v in obs.new.items()}, out["tx.heater"].copy()))
        return out
    ctrl.step = step
    rec = loop.run(5500)
    ctrl.step = orig
    return dict(cfg=cfg, pp=pp, spec=spec, plant=plant, ctrl=ctrl, loop=loop, rec=rec, seen=seen)


def test_map_then_lock_on_the_stable_side(locked):
    rec, ctrl = locked["rec"], locked["ctrl"]
    tel = ctrl.telemetry()
    assert np.all(tel["state"] == pi_lock.ST_LOCK), tel["state"]
    assert np.all(tel["locked"])
    d = rec["delta_pm"].astype(float)
    # the MAP stops at the peak: it never crosses the self-heating fold (~9 pm blue of the peak in g26 nominal)
    assert np.all(d.min(axis=0) > -8.0), d.min(axis=0)
    # locked red of the line, at the same detuning on every ring (the set-point is a fraction of each ring's peak)
    final = np.median(d[-500:], axis=0)
    assert np.all((final > 80.0) & (final < 180.0)), final
    assert np.ptp(final) < 8.0, final
    # measured lock-point slope: negative (stable side), with the self-heating relief folded in
    assert np.all(tel["slope_per_code"] < 0.0)
    t_lock = harness.time_to_lock_us(rec)
    m = harness.lock_metrics(rec, slice(-1200, -800), slice(-800, None))
    assert np.all(m.excursions == 0) and np.all(m.max_err_pm < 1.5), m.max_err_pm
    report("pi_lock steady g26 nominal", m, time_to_lock_us=float(np.nanmax(t_lock)),
           lock_delta_pm=float(np.mean(final)), peak_codes=float(np.mean(tel["peak_codes"])),
           slope_per_code=float(np.mean(tel["slope_per_code"])))


def test_compiled_replay_is_bit_identical(locked):
    """The compiled port, fed the observation sequence of the Python run, returns the same codes at every step
    (MAP, peak detection, slope fit, PI), and ends in the same state."""
    spec = locked["spec"]
    cc = PiLockCompiled()
    cc.reset(spec)
    for tick, sens, new, want in locked["seen"]:
        got = cc.step(Observation(tick, spec.tick_s, sens, new))["tx.heater"]
        assert np.array_equal(got, want), (tick, got, want)
    tp, tc = locked["ctrl"].telemetry(), cc.telemetry()
    for k in ("state", "integrator", "peak_codes", "slope_per_code"):
        np.testing.assert_array_equal(tp[k], tc[k])


@pytest.mark.parametrize("kick", [0, 1])
def test_toy_parity_with_falls_and_recaptures(kick):
    """Python and compiled PI on a toy bistable ring (folds, captures, invalid samples, a disturbance that pushes
    rings past the fold): identical codes at every step, and the falls are recovered."""
    spec = harness.toy_spec(2, 3)
    u0 = np.array([[30000, 30500, 31000], [29800, 30200, 30900]], float)
    g = np.array([[4.0, 6.0, 0.8], [3.0, 8.0, 1.5]])
    init = [30000 + 1400, 30500 + 1400, 31000 + 600]

    def disturb(toy, k):
        if kick and 700 <= k < 760:
            toy.u0 -= 25.0               # the resonance moves red (x up): the PI lowers the heater and lags
        if kick and 800 <= k < 815:
            toy.u0 += 250.0              # a fast blue move (x down): the ring is pushed past the fold
        toy.valid[:] = not (400 <= k < 405)

    runs = []
    for ctrl in (PiLock(init_codes=init, k2=0.002), PiLockCompiled(init_codes=init, k2=0.002)):
        toy = harness.ToyRings(spec, u0=u0, g=g, u_init=np.array(init), x0=3.0)
        ctrl.reset(spec)
        h, _ = toy.run(ctrl, 1600, on_step=disturb)
        runs.append((h, ctrl.telemetry(), toy.x.copy()))
    (ha, ta, xa), (hb, tb, xb) = runs
    assert np.array_equal(ha, hb)
    np.testing.assert_array_equal(ta["state"], tb["state"])
    np.testing.assert_array_equal(ta["fall_count"], tb["fall_count"])
    assert np.all(ta["state"] == pi_lock.ST_LOCK), ta["state"]
    np.testing.assert_allclose(xa, math.sqrt(1 / 0.7 - 1), atol=0.03)       # y = 0.7 M on the red side
    if kick:
        assert ta["fall_count"].sum() >= 1                                   # the kick crossed the fold somewhere


@pytest.mark.parametrize("delta_db", [-1.0, +1.0])
def test_laser_power_step_moves_the_lock_point(locked, delta_db):
    """Known failure mode of a side-of-fringe lock, measured: the peak is not re-measured, so a laser power step moves
    the set-point along the line (toward the peak for a drop in power). The loop stays on the stable side (no
    fall)."""
    cfg, spec = locked["cfg"], locked["spec"]
    plant = harness.snapshot(locked["plant"])
    ctrl = locked["ctrl"]
    tel = ctrl.telemetry()
    c = PiLock(start="lock", peak_codes=tel["peak_codes"], slope_per_code=tel["slope_per_code"],
               init_codes=plant.lat_heater.copy())
    loop = harness.ClosedLoop(plant, spec, c)
    t_s = (plant.k_next + 500) * plant.dt
    plant.set_scenario(harness.laser_power_step_scenario(cfg, t_s, delta_db))
    rec = loop.run(3500)
    tc = c.telemetry()
    assert np.all(tc["state"] == pi_lock.ST_LOCK) and np.all(tc["fall_count"] == 0)
    d = rec["delta_pm"].astype(float)
    before, after = np.median(d[300:480], axis=0), np.median(d[-500:], axis=0)
    shift = after - before
    # fraction-of-peak arithmetic for a Lorentzian: f_new = f * 10**(-dB/10); here f = 0.7
    if delta_db < 0:
        assert np.all((shift < -15.0) & (shift > -90.0)), shift
    else:
        assert np.all((shift > 10.0) & (shift < 90.0)), shift
    m = harness.lock_metrics(rec, slice(300, 480), slice(500, None))
    report(f"pi_lock laser power step {delta_db:+.0f} dB", m, lock_shift_pm=float(np.mean(shift)))


def test_capture_start_single_ring():
    """start='capture' (no hand-off on the stable branch): ramp up until the monitor jumps (capture), MAP, lock.
    A single ring and line, so no other line or order can capture it (in g26 that is the supervisor's job)."""
    ov = ["array.n_rings=1", "channel_plan.n_lines=1", "array.bus_order=[0]", "channel_plan.ring_to_line=[0]"]
    cfg = harness.load_cfg(ov)
    pp = harness.realize(cfg, 1)
    spec = harness.make_spec(cfg, 1)
    plant = harness.LoopPlant(pp, cfg, period_ticks=spec.period_ticks)
    d = harness.place_on_hot_branch(plant, delta_pm=170.0, n_ticks=2000)
    code_lock = int(plant.lat_heater[0, 0])
    # cold side: 1500 codes below the capture point; the ring falls to the cold branch
    plant.lat_heater[:] = code_lock - 3500
    for k in range(plant.k_next, plant.k_next + 400):
        plant.tick(k)
    assert plant.delta_pm[0, 0] < -200.0
    c = PiLock(start="capture", ramp_step_codes=24.0)
    loop = harness.ClosedLoop(plant, spec, c)
    rec = loop.run(7000)
    tel = c.telemetry()
    assert tel["state"][0, 0] == pi_lock.ST_LOCK, tel["state"]
    final = float(np.median(rec["delta_pm"][-300:, 0, 0]))
    assert 80.0 < final < 180.0, final
    m = harness.lock_metrics(rec, slice(-800, -400), slice(-400, None))
    assert m.max_err_pm[0, 0] < 1.0, m.max_err_pm               # settled within the run (5.2 ms measured)
    report("pi_lock capture start (N = L = 1)", m, time_to_lock_us=float(harness.time_to_lock_us(rec)[0, 0]),
           lock_delta_pm=final)


def test_nominal_model_is_knowledge_gated():
    cfg = harness.load_cfg()
    spec = harness.make_spec(cfg, 1)
    nm = NominalModel.from_spec(spec)          # g26: knowledge = [line_grid, ring_to_line]
    assert nm.law is None and nm.pm_per_mw is None and nm.fwhm_pm is None and nm.pm_per_code(30000) is None
    assert nm.lambda_m == pytest.approx(1310e-9, rel=0.01)
    know = ["controller_interface.knowledge=[line_grid, ring_to_line, heater_law, heater_efficiency, linewidth, "
            "thermal_dynamics, dlambda_dt_nominal]"]
    spec2 = harness.make_spec(harness.load_cfg(know), 1)
    nm2 = NominalModel.from_spec(spec2)
    assert nm2.law == "voltage" and nm2.tau_slow_s == pytest.approx(13.01e-6)
    assert nm2.fwhm_pm == pytest.approx(1310.0 / 3850 * 1e3, rel=0.01)
    assert nm2.dlambda_dt_pm_per_k == pytest.approx(66.0, rel=0.03)      # the H_solver nominal (si of the leaf)
    # the nominal open-loop heater gain (pm/mW) equals derive()'s diagnostic: both are the heater anchor FSR/P_FSR
    # (relation R2) of the same preset, so they must agree
    from mrm.params.derive import derive
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dg = derive(harness.load_cfg()).diagnostics
    assert nm2.pm_per_mw == pytest.approx(1e3 * float(np.mean(dg["heater_efficiency_nm_per_mw"])), rel=0.01)
    # derive's heater_pm_per_code uses the slope at the lock temperature (beta > 0), the anchor ratio FSR/P_FSR is
    # the sweep average: they differ by a few per cent
    want = float(np.mean(dg["heater_pm_per_code"][2]))          # at code 32768
    assert float(nm2.pm_per_code(32768.0)) == pytest.approx(want, rel=0.06)
    # the PI zero follows the nominal slow pole when the token is given
    c = PiLock()
    c.reset(spec2)
    assert c.t_ratio == pytest.approx(spec2.period_s / 13.01e-6)


def test_controller_modules_do_not_import_truth():
    """Truth isolation: the baseline modules import nothing from the plant, the simulator, the validation oracle,
    realize or derive (the sampled worlds). mrm.params.units (unit conversion only) is allowed, as in
    mrm.control.api."""
    for f in BASELINE_FILES:
        path = ROOT / "mrm/control/baselines" / f
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for n in names:
                assert not n.startswith(("mrm.plant", "mrm.sim", "mrm.validation", "mrm.params.realize",
                                         "mrm.params.derive", "mrm.params.relations")), (f, n)
                assert "truth" not in n.lower(), (f, n)


def test_registry_and_parameter_checks():
    assert registry.resolve("pi_lock") is PiLock
    c = registry.create("pi_lock_c", {"fraction": 0.6})
    assert isinstance(c, PiLockCompiled) and c.params.fraction == 0.6
    with pytest.raises(TypeError):
        PiLock(unknown=1)
    with pytest.raises(ValueError):
        PiLock(fraction=1.2)
    with pytest.raises(ValueError):
        PiLock(start="nope")
    spec = harness.toy_spec(1, 2)
    with pytest.raises(ValueError, match="factory table"):
        PiLock(start="lock").reset(spec)
    a = ActionValidator(spec)
    toy = harness.ToyRings(spec, u0=30000, u_init=31000)
    c = PiLock(init_codes=31000)
    c.reset(spec)
    out = a.validate(c.step(toy.observation()))
    assert out["tx.heater"].dtype == np.uint16


# --------------------------------------------------------------------- regression tests: each one failed before its fix

def test_type2_velocity_does_not_wind_up_at_the_rail():
    """Anti-windup: a disturbance drives the heater to full scale for ~300 steps and then reverses. The type-2
    velocity v used to keep integrating while the heater sat at the rail, and held it there ~80 steps (0.8 ms) after
    the type-1 PI had left it. Now v is reset while it pushes into the clamp: both leave the rail together. Python and
    compiled stay identical."""
    spec = harness.toy_spec(1, 1)

    def dist(toy, k):
        if 300 <= k < 600:
            toy.u0 += 20.0               # the ring needs more heat than the DAC has
        if 900 <= k < 1200:
            toy.u0 -= 20.0               # and back
    last = {}
    for k2 in (0.0, 0.002):
        hs = []
        for cls in (PiLock, PiLockCompiled):
            c = cls(init_codes=[61400], k2=k2)
            toy = harness.ToyRings(spec, u0=[[60000.0]], g=0.8, u_init=[61400], x0=3.0)
            c.reset(spec)
            h, _ = toy.run(c, 2000, on_step=dist)
            hs.append(h)
            assert c.telemetry()["state"][0, 0] == pi_lock.ST_LOCK
            np.testing.assert_allclose(toy.x, math.sqrt(1 / 0.7 - 1), atol=0.03)
        assert np.array_equal(hs[0], hs[1])
        sat = np.flatnonzero(hs[0][:, 0, 0] >= spec.heater_dac.code_max)
        assert sat.size > 100, sat.size                  # the rail was reached
        last[k2] = int(sat[-1])
    assert last[0.002] <= last[0.0] + 5, last


@pytest.mark.parametrize("g", [0.5, 4.0])
def test_capture_start_with_and_without_bistability(g):
    """start='capture' from the cold (blue) side. g = 4: bistable, the ring jumps onto the hot branch; g = 0.5: no
    bistability, the monitor rises and falls smoothly. The old capture test (a jump within jump_steps steps) never
    fired for g = 0.5: the heater ramped to full scale and parked there. Both now lock at y = 0.7 M on the red side,
    Python and compiled identical, and the heater never reaches full scale."""
    spec = harness.toy_spec(1, 2)
    u0 = np.array([[30000.0, 31000.0]])
    runs = []
    for cls in (PiLock, PiLockCompiled):
        c = cls(start="capture", init_codes=[27000, 28000])
        toy = harness.ToyRings(spec, u0=u0, g=g, u_init=[27000, 28000], x0=-3.0)
        c.reset(spec)
        h, _ = toy.run(c, 3000)
        runs.append((h, c.telemetry(), toy.x.copy()))
    (ha, ta, xa), (hb, tb, xb) = runs
    assert np.array_equal(ha, hb)
    assert np.all(ta["state"] == pi_lock.ST_LOCK), ta["state"]
    np.testing.assert_allclose(xa, math.sqrt(1 / 0.7 - 1), atol=0.03)
    assert ha.max() < spec.heater_dac.code_max


def test_capture_without_a_resonance_faults_back_to_the_start_code():
    """No resonance within the heater range: CAPTURE ramps to full scale once, then FAULT with the heater back at the
    start code (never parked at full scale: in a bank that would load the shared supply)."""
    spec = harness.toy_spec(1, 1)
    for cls in (PiLock, PiLockCompiled):
        c = cls(start="capture", init_codes=[27000])
        toy = harness.ToyRings(spec, u0=[[200000.0]], g=0.5, u_init=[27000], x0=-3.0)
        c.reset(spec)
        h, _ = toy.run(c, 3000)
        assert c.telemetry()["state"][0, 0] == pi_lock.ST_FAULT
        assert h[-1, 0, 0] == 27000 and h.max() == spec.heater_dac.code_max


def test_python_init_codes_are_whole_codes_like_the_compiled_port():
    """A fractional init code is rounded by both versions (the compiled ABI carries int64 init codes); the Python
    version used to start from the fractional code and differed at 209 of 600 outputs here."""
    spec = harness.toy_spec(1, 2)
    hs = []
    for cls in (PiLock, PiLockCompiled):
        c = cls(start="lock", init_codes=[31700.4, 32700.6], peak_codes=50000.0, slope_per_code=-50.0)
        toy = harness.ToyRings(spec, u0=[[30000.0, 31000.0]], g=0.8, u_init=[31700, 32701], x0=3.0)
        c.reset(spec)
        hs.append(toy.run(c, 300)[0])
    assert np.array_equal(hs[0], hs[1])


@pytest.mark.parametrize("case", ["clipped_from_the_start", "clips_near_the_top"])
def test_monitor_clipped_at_the_adc_full_scale(case):
    """ADC clipping: the drop peak above the 16-bit ADC full scale (the g26_hot_selfheat corner puts
    it at ~2.5x). Clipped from the hand-off on: DECLIP ramps red until the monitor reads below f x full scale and locks
    there (it used to FAULT: no unclipped record to fit). Clipping only near the top (the MAP's existing branch): the
    set-point is f x full scale too. Both: y = 0.7 x 65535 on the red side, Python and compiled identical."""
    spec = harness.toy_spec(1, 2)
    u0 = np.array([[30000.0, 31000.0]])
    M, init = (150000.0, u0[0] + 200.0) if case == "clipped_from_the_start" else (70000.0, u0[0] + 680.0)
    runs = []
    for cls in (PiLock, PiLockCompiled):
        c = cls(init_codes=init)
        toy = harness.ToyRings(spec, u0=u0, g=0.8, M=M, u_init=init, x0=3.0)
        y0 = M / (1.0 + toy.x ** 2)
        assert np.all(y0 >= 65535) == (case == "clipped_from_the_start")
        c.reset(spec)
        h, _ = toy.run(c, 2500)
        runs.append((h, c.telemetry(), toy.x.copy()))
    (ha, ta, xa), (hb, tb, xb) = runs
    assert np.array_equal(ha, hb)
    assert np.all(ta["state"] == pi_lock.ST_LOCK), ta["state"]
    assert np.all(ta["peak_codes"] == 65535.0)
    np.testing.assert_allclose(M / (1.0 + xa ** 2), 0.7 * 65535, rtol=0.01)
    assert np.all(xa > 0.0)


@pytest.mark.xfail(strict=True, reason="known limitation: the jump-based fall "
                   "detector reads a fast RED step of more than ~0.3 M_pk in jump_steps steps as a fold fall; "
                   "RECAPTURE then ramps the heater red, never sees a peak and gives up (FAULT at the fall code)")
def test_fast_red_step_is_not_mistaken_for_a_fall():
    """Measured on a toy ring: the line moves 700 heater codes red in one step (x 0.65 -> 1.96, y 0.7 -> 0.21 M_pk,
    still on the red side). The desired behaviour is to stay in LOCK and walk back; the PI family (pi_lock,
    feedforward_pi, midranging) and the dither/ES/bias locks share the detector. Slower red moves (A1/A2 rates) are
    followed; this needs a ~200 pm step within ~30 us."""
    spec = harness.toy_spec(1, 1)
    c = PiLock(init_codes=[30450])
    toy = harness.ToyRings(spec, u0=[[30000.0]], g=0.5, u_init=[30450], x0=3.0)
    c.reset(spec)

    def kick(t, k):
        if k == 1500:
            t.u0 -= 700.0
    toy.run(c, 3500, on_step=kick)
    tel = c.telemetry()
    assert tel["fall_count"][0, 0] == 0 and tel["state"][0, 0] == pi_lock.ST_LOCK
