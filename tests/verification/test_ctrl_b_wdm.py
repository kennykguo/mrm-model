"""WDM bank supervisor: cold-start map, line labels, assignment, sequential acquisition (07 P1-12), closed loop on the
g26 reference plant, plus fast unit tests of the planning logic.

The cold-ring hazard of docs/handoff/integrator.md section 4 is present in the run (Truth at turn-on: most rings sit
on a foreign line); the supervisor must end with every ring on its declared line, flag the captured rings, and keep
the die supply within its limit. Slower variants (barrel shift under a tight supply, warm start from the factory
table, a sampled world) are in test_ctrl_b_wdm_slow.py.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from mrm.control import registry
from mrm.control.api import validate_telemetry
from mrm.control.baselines.ramp_acquire import ST_LOCK
from mrm.control.baselines.wdm_supervisor import PH_TRACK, WdmSupervisor
from mrm.control.calibration import code_of_w, heater_current_a
from mrm.plant import electrical as el

from test_ctrl_b_plant import CtrlBPlant, load_g26, make_spec, realize, run_closed_loop
from test_ctrl_b_ramp import steady_metrics

KNOW = ["line_grid", "ring_to_line", "heater_nominal", "optics_nominal", "dlambda_dt_nominal"]
N_TICKS = 20000


@pytest.fixture(scope="module")
def cold_run():
    cfg = load_g26(knowledge=KNOW)
    pp = realize(cfg, 1, nominal=True)
    spec = make_spec(cfg, 1)
    plant = CtrlBPlant(pp, cfg, spec)
    # Truth at turn-on (before any heating): which line each ring sits on
    for k in range(60):
        plant.tick(k)
    t0 = plant.truth()
    plant = CtrlBPlant(pp, cfg, spec)
    ctrl = WdmSupervisor()
    flags = []
    orig_tick = plant.tick

    def tick(k, cmd=None):
        out = orig_tick(k, cmd)
        flags.append(int(np.any(out["heater"].flags & el.FLAG_SUPPLY)))
        return out
    plant.tick = tick
    rec, val = run_closed_loop(plant, ctrl, N_TICKS, record_every=5)
    return plant, ctrl, rec, val, t0, np.array(flags)


def test_registered(cold_run):
    plant, ctrl, rec, val, _, _ = cold_run
    c = registry.create("wdm_supervisor", {"policy": "fixed", "map_w_max": 0.8})
    assert isinstance(c, WdmSupervisor) and c.params.map_w_max == 0.8
    validate_telemetry(ctrl.telemetry(), plant.spec)


def test_hazard_present_at_turn_on(cold_run):
    """Truth at laser turn-on, heaters at 0: rings sit on lines that are not theirs (g26 nominal: n + 3)."""
    _, ctrl, _, _, t0, _ = cold_run
    near, det = t0["nearest_line"][0], t0["nearest_det_m"][0]
    captured = np.abs(det) < 0.35e-9                    # within ~1 FWHM of a line, self-heated onto it
    foreign = captured & (near != np.arange(near.size))
    print(f"\nturn-on: nearest lines {near.tolist()}, detuning {np.round(det * 1e12).astype(int).tolist()} pm")
    assert foreign.sum() >= 5
    # the supervisor flags them from its own map (monitor high at w = 0, then falling)
    flagged = ctrl.captured[0]
    assert np.sum(flagged & captured) >= foreign.sum() - 1


def test_cold_start_locks_every_ring_on_its_line(cold_run):
    plant, ctrl, rec, val, _, flags = cold_run
    f = ctrl.fsm
    assert ctrl.phase[0] == PH_TRACK
    assert np.all(f.state[0] == ST_LOCK), f.state
    assert np.all(ctrl.map_ok[0]), ctrl.map_ok
    t = plant.truth()
    want = np.arange(plant.N)                           # g26: fixed identity map (tx.assignment bound to the scenario)
    np.testing.assert_array_equal(ctrl.assigned[0], want)
    np.testing.assert_array_equal(t["nearest_line"][0], want)
    assert not np.any(ctrl.wrong[0]) and not np.any(ctrl.supply_limited[0])
    # the plant's supply limit was never hit (07 P1-12: the plant clips and flags)
    assert flags.sum() == 0
    # the nominal heater model (heater_nominal knowledge) predicts the real heater power within 3 %
    p_est = rec.telemetry[-1]["p_est_mw"][0]
    np.testing.assert_allclose(p_est, plant.u_heater[0] * 1e3, rtol=0.03)
    # per-ring lock metrics after the last acquisition
    edges = np.array(rec.edge_ticks)
    ph = np.array([tt["phase"][0, 0] for tt in rec.telemetry])
    t_track = int(edges[np.flatnonzero(ph == PH_TRACK)[0]])
    d = rec.det_m[:, 0, :] * 1e12
    rows = []
    late = rec.ticks > t_track
    for n in range(plant.N):
        m = steady_metrics(d[late, n], rec.ticks[late], t_from=t_track + 300)
        rows.append(m)
        assert m["rms_pm"] < 1.0 and m["excursions"] == 0, (n, m)
    pen, _ = plant.oma_penalty_db()
    print(f"\ncold start: map done, all rings in TRACK at {t_track * 1e-3:.2f} ms; total heater "
          f"{plant.u_heater[0].sum() * 1e3:.1f} mW")
    for n, m in enumerate(rows):
        print(f"  ring {n}: line {t['nearest_line'][0, n]}, detuning {m['d_ss_pm']:.1f} pm, jitter rms "
              f"{m['rms_pm']:.3f} pm, max {m['max_pm']:.2f} pm, OMA penalty {pen[0, n]:.2f} dB")


# ---------------------------------------------------------------------------------------------------------------
# planning logic (fast, synthetic maps)
# ---------------------------------------------------------------------------------------------------------------

def _synthetic(spec, policy, supply_a=None, bound=True, w0=0.06):
    """A supervisor whose maps are set by hand: ring n meets line (n + 3 + j) mod 8 at w = w0 + 0.04 j (w0 above the
    supervisor's headroom floor w_min_target unless a test asks otherwise)."""
    ctrl = WdmSupervisor(policy=policy)
    ctrl.reset(spec)
    N, L = spec.n_rings, spec.n_lines
    for n in range(N):
        order = [(n + 3 + j) % L for j in range(L)]                 # line n + 3 first, then n + 4, ...
        w = w0 + 0.04 * np.arange(L)
        ctrl.events[0][n] = w
        ctrl.labels[0][n] = np.array(order)
        ctrl.ev_floor_w[0][n] = w - 0.02
        ctrl.ev_height[0][n] = np.full(L, 30000.0)
    if supply_a is not None:
        import dataclasses
        ctrl.d = dataclasses.replace(ctrl.d, supply_current_max_a=supply_a)
    return ctrl


def test_plan_fixed_cyclic_any():
    cfg = load_g26(overrides=["controller_interface.bind.tx.assignment=controller"], knowledge=KNOW)
    spec = make_spec(cfg, 1)
    # fixed: the declared identity, whatever it costs
    c = _synthetic(spec, "fixed")
    c._plan_assignment(0)
    np.testing.assert_array_equal(c.target_line[0], np.arange(8))
    # cyclic with plenty of supply: the declared map (shift 0) is kept
    c = _synthetic(spec, "cyclic")
    c._plan_assignment(0)
    assert c.shift[0] == 0
    # cyclic with a tight supply: a barrel shift (PAT-1) that fits; every ring moves by the same number of channels
    c = _synthetic(spec, "cyclic", supply_a=0.08)
    c._plan_assignment(0)
    s = int(c.shift[0])
    assert s != 0
    np.testing.assert_array_equal(c.target_line[0], (np.arange(8) + s) % 8)
    w = np.array([c.events[0][n][c.target_ev[0, n]] for n in range(8)])
    codes = code_of_w(w + 2 * c.dw_acq, c.fsm.code_max, c.fsm.q)
    assert heater_current_a(codes, c.d).sum() <= 0.95 * 0.08
    # any: minimum total drive (here every ring takes its first resonance, line n + 3)
    c = _synthetic(spec, "any")
    c._plan_assignment(0)
    np.testing.assert_array_equal(c.target_line[0], (np.arange(8) + 3) % 8)
    # optional headroom floor (PAT-1 [0050]): a resonance below w_min_target is never a target; with one resonance
    # per line in these maps, a first resonance at w = 0.02 leaves line n + 3 without a target, so 'any' takes n + 4
    c = _synthetic(spec, "any", w0=0.02)
    assert c.w_min_target == 0.0                    # off by default: under a tight supply w ~ 0 is the cheapest map
    c.params.w_min_target = c.w_min_target = 0.045
    c._plan_assignment(0)
    np.testing.assert_array_equal(c.target_line[0], (np.arange(8) + 4) % 8)
    assert c._target_index(0, 0, 3) == -1 and c._target_index(0, 0, 3, w_min=0.0) == 0


def test_online_barrel_shift_logic():
    """PAT-1 online step in TRACK (synthetic map, no plant): when the locked bank's estimated supply stays above the
    budget for excess_steps steps, the cyclic policy moves every ring to the next bluer line (line + 1, the
    resonance just below its present drive in map coordinates) and re-acquires ring by ring, each window shifted by
    that ring's own drift since its acquisition."""
    from mrm.control.baselines.wdm_supervisor import PH_ACQUIRE
    cfg = load_g26(overrides=["controller_interface.bind.tx.assignment=controller"], knowledge=KNOW)
    spec = make_spec(cfg, 1)
    from mrm.control.calibration import predicted_pattern
    c = WdmSupervisor(policy="cyclic")
    c.reset(spec)
    grid = np.array(spec.line_grid_hz)
    for n in range(8):                                       # the g26 heating order: line n + 3 first, 9 crossings
        v, lines, _ = predicted_pattern(grid, 1.8e12, (n + 3) % 8, 9)
        c.events[0][n] = 0.02 + 0.35 * v
        c.labels[0][n] = lines
        c.ev_floor_w[0][n] = c.events[0][n] - 0.02
        c.ev_height[0][n] = np.full(9, 30000.0)
    c._plan_assignment(0)
    assert c.shift[0] == 0
    f = c.fsm
    w_lock = np.array([c.events[0][n][c.target_ev[0, n]] for n in range(8)]) + 0.005
    c.w_lock0[0] = w_lock                                   # acquired (map coordinates)
    drift = 0.05                                            # the chip cooled: every ring now needs more drive
    f.w[0] = w_lock + drift
    f.code[0] = code_of_w(f.w[0], f.code_max, f.q)
    f.state[0] = ST_LOCK
    c.phase[0] = PH_TRACK
    import dataclasses
    c.d = dataclasses.replace(c.d, supply_current_max_a=0.9 * float(heater_current_a(f.code[0], c.d).sum()))
    for _ in range(c.params.excess_steps):
        c._track_step(0)
    assert c.shift[0] == 1 and c.phase[0] == PH_ACQUIRE and c.retarget[0]
    np.testing.assert_array_equal(c.target_line[0], (np.arange(8) + 1) % 8)
    for n in range(8):
        w = c.events[0][n]
        k = int(c.target_ev[0, n])
        assert c.labels[0][n][k] == (n + 1) % 8 and w[k] < w_lock[n]          # one line bluer: less drive
    # the first ring's new window carries its own drift
    c._start_ring(0, 0)
    k = int(c.target_ev[0, 0])
    assert c.drift_used[0, 0] == pytest.approx(drift)
    assert f.w_lo[0, 0] < c.events[0][0][k] + drift < f.w_hi[0, 0]


def test_policy_falls_back_to_fixed_without_the_binding():
    """g26 binds tx.assignment to the scenario: the supervisor may not remap (07 P1-12), whatever its params."""
    cfg = load_g26(knowledge=KNOW)
    spec = make_spec(cfg, 1)
    c = WdmSupervisor(policy="cyclic")
    c.reset(spec)
    assert c.policy == "fixed" and not c.bound_assignment
    cfg = load_g26(overrides=["controller_interface.bind.tx.assignment=controller"], knowledge=KNOW)
    c.reset(make_spec(cfg, 1))
    assert c.policy == "cyclic" and c.bound_assignment


def test_controllers_b_do_not_import_truth():
    """07 P1-1 test_control_does_not_import_truth for the files of this owner: no import of the plant, realize,
    the simulation engine or the validation oracle."""
    root = Path(__file__).resolve().parents[2] / "mrm" / "control"
    files = [root / "calibration.py"] + [root / "baselines" / f for f in
                                         ("ramp_acquire.py", "wdm_supervisor.py", "bitstat_lock.py")]
    bad = ("mrm.plant", "mrm.sim", "mrm.params.realize", "mrm.validation")
    for f in files:
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                assert not any(m == b or m.startswith(b + ".") for b in bad), (f.name, m)


# ---------------------------------------------------------------------------------------------------------------
# review additions: per-world state, reset, lock verification, target headroom
# ---------------------------------------------------------------------------------------------------------------

def _cyclic_spec(batch):
    cfg = load_g26(overrides=["controller_interface.bind.tx.assignment=controller"], knowledge=KNOW)
    return make_spec(cfg, batch)


def _set_pattern_maps(c, b, fsr_hz, rng, n_events=11):
    """World b: ring n meets its lines in the predicted order of a heater-up sweep from line n + 3, at
    w = 0.06 + 0.55 v (v in FSR units) plus 1 mdrive noise."""
    from mrm.control.calibration import predicted_pattern
    grid = np.array(c.spec.line_grid_hz)
    want = []
    for n in range(c.N):
        v, lines, _ = predicted_pattern(grid, fsr_hz, (n + 3) % c.L, n_events)
        c.events[b][n] = 0.06 + 0.55 * v + rng.normal(0.0, 0.001, v.size)
        c.ev_floor_w[b][n] = c.events[b][n] - 0.02
        c.ev_height[b][n] = np.full(v.size, 30000.0)
        want.append(lines)
    return want


def test_state_is_per_world_and_reset_clears_it():
    """B = 2 with different FSRs (window B 1.8 THz, window A 1.47 THz): each world's fresh-map labels use its OWN
    bank fit (the bank fit was one attribute shared by all worlds, so world 0 was labelled with world 1's FSR);
    the up-sweep step is per world; reset clears the first-step sync and the bank fits."""
    spec = _cyclic_spec(2)
    c = WdmSupervisor(policy="cyclic")
    c.reset(spec)
    rng = np.random.default_rng(4)
    want0 = _set_pattern_maps(c, 0, 1.8e12, rng)
    want1 = _set_pattern_maps(c, 1, 1.47e12, rng)
    c._label(0)
    c._label(1)
    assert c.fits[0][0]["fsr_hz"] == pytest.approx(1.8e12, rel=0.02)
    assert c.fits[1][0]["fsr_hz"] == pytest.approx(1.47e12, rel=0.02)
    for n in (0, 3):
        c._label_fresh(0, n)
        np.testing.assert_array_equal(c.labels[0][n], want0[n])
        c._label_fresh(1, n)
        np.testing.assert_array_equal(c.labels[1][n], want1[n])
    c.fsm.dw_up[0] = 0.01
    assert np.all(c.fsm.dw_up[1] == c.fsm.dw_up0)
    c._synced = True
    c.reset(spec)
    assert c._synced is False and c.fits == [None, None] and c.t0 is None


def test_verify_lock_flags_a_displaced_comb():
    """07 P1-20: a lock is vouched for only if the found resonance and the last resonance the up-search passed
    sit where the map put them (common drift within 0.3 of the local spacing). A comb shifted by half a spacing (the
    ring warmer than during its map) is flagged; a 6 mdrive drift (one map step, typical) is not."""
    spec = _cyclic_spec(1)
    c = WdmSupervisor(policy="fixed")
    c.reset(spec)
    rng = np.random.default_rng(2)
    _set_pattern_maps(c, 0, 1.8e12, rng)
    c.map_ok[0] = True
    f = c.fsm
    n, k = 2, 6
    c.target_ev[0, n] = k
    w = c.events[0][n]
    for shift, flagged in ((-0.006, False), (-0.5 * (w[k] - w[k - 1]), True)):
        f.sk_w[0, n, :k] = w[:k] + shift
        f.sk_n[0, n] = k
        f.w_event[0, n] = w[k] - 0.006               # the found resonance looks right in both cases
        c.drift_used[0, n] = 0.0
        c._verify_lock(0, n)
        assert bool(c.acq_uncertain[0, n]) == flagged, (shift, c.drift_meas[0, n])
    f.locked[0, n] = True
    assert c.telemetry()["line_confident"][0, n] == 0.0
    c.acq_uncertain[0, n] = False
    assert c.telemetry()["line_confident"][0, n] == 1.0


def test_planner_without_a_feasible_shift_takes_the_least_drive():
    """Cyclic policy, a supply so tight that no barrel shift fits: the planner takes the shift with the least total
    drive (the supply sequencing then flags what does not fit), not silently the declared map (the costliest one
    in these synthetic maps: line n is the last of ring n's lines)."""
    cfg = load_g26(overrides=["controller_interface.bind.tx.assignment=controller"], knowledge=KNOW)
    spec = make_spec(cfg, 1)
    c = _synthetic(spec, "cyclic", supply_a=0.01)
    c._plan_assignment(0)
    assert c.shift[0] == 3                          # every ring on its first resonance, line n + 3
    np.testing.assert_array_equal(c.target_line[0], (np.arange(8) + 3) % 8)


def test_track_rearms_assigned_rings_in_fail():
    """In TRACK a ring whose lock was lost and whose re-search FAILed (its line dark for a while) is re-armed after
    fail_rearm_s, in its own window; a ring that never got a line (assigned -1) stays parked. Before the review a
    FAIL in TRACK was final."""
    from mrm.control.baselines.ramp_acquire import ST_FAIL, ST_UP
    spec = make_spec(load_g26(knowledge=KNOW), 1)
    c = WdmSupervisor(policy="fixed")
    c.reset(spec)
    f = c.fsm
    c.phase[0] = PH_TRACK
    f.state[0] = ST_LOCK
    mask = np.zeros((1, 8), bool)
    mask[0, [2, 5]] = True
    f.start(mask, w_from=0.2, w_lo=0.25, w_hi=0.3)           # their last search: from 0.2 in [0.25, 0.3]
    f._enter_fail(mask)
    f.w[0, [2, 5]] = 0.25
    c.assigned[0, 5] = -1
    f.fail_steps[0, [2, 5]] = f.rearm_steps - 1
    c._track_step(0)
    assert f.state[0, 2] == ST_FAIL                           # not yet
    f.fail_steps[0, [2, 5]] = f.rearm_steps
    c._track_step(0)
    assert f.state[0, 2] == ST_UP and f.w[0, 2] == pytest.approx(0.2) and f.w_lo[0, 2] == 0.25
    assert f.state[0, 5] == ST_FAIL
