"""Robustness of the controllers-B locks on the engine (mrm.sim.runner, Python controller path): model mismatch
from the uncertainty set (sampled worlds, the named corner presets), a data-swing loss for the bit-statistics
ratio lock. Truth is read in the TEST only: the engine reports the detuning to the DECLARED line, so each case runs
twice, the second time declaring (tx.assignment) the line the first run ended on; the controllers are
deterministic, so the second run reproduces the first and its Truth is exact.

What is asserted is what the controller claims: a ring in LOCK sits on the red (self-heating-stable) side of a line,
steadily. Which line the ramp acquisition takes is not asserted (the first resonance met from blue, a different
line in each world: the WDM supervisor exists for the assignment). Printed (pytest -s): per world the line, the
detuning, the jitter, and the cases the controller handles in a degraded way (``partial``: the peak lies below
w = 0, the set-point is a fraction of a lower bound; ``clipped``: the monitor ADC saturated at the peak, so the
set-point is a fraction of the clipped amplitude). Both put the lock further red, with a larger OMA penalty.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

runner = pytest.importorskip("mrm.sim.runner")

from mrm.control.baselines.bitstat_lock import BitStatLock
from mrm.control.baselines.ramp_acquire import RampAcquire, ST_LOCK
from mrm.scenarios import generators as g
from mrm.scenarios import library as lib
from mrm.units import C0


KNOW = ["line_grid", "ring_to_line", "heater_nominal", "optics_nominal", "dlambda_dt_nominal"]
BIND = "controller_interface.bind.tx.assignment=controller"


class Declare:
    """Adapter (tests only): forwards a controller and writes a fixed per-world ``tx.assignment`` (B, N), so the
    engine's Truth detuning refers to those lines."""

    def __init__(self, ctrl, assignment):
        self.ctrl = ctrl
        self.asg = np.asarray(assignment, dtype=np.int64)
        self.name = getattr(ctrl, "name", "declared")
        self.latency_ticks = int(getattr(ctrl, "latency_ticks", 0))

    def reset(self, spec):
        self.ctrl.reset(spec)

    def step(self, obs):
        out = dict(self.ctrl.step(obs) or {})
        out["tx.assignment"] = self.asg.copy()
        return out

    def telemetry(self):
        return self.ctrl.telemetry()


def _load(preset, ov):
    from mrm import params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load(preset, overrides=list(ov) + ["controller_interface.knowledge=[" + ",".join(KNOW) + "]"])


def _run(make_ctrl, preset, duration_s, batch=1, sampled=False, scenario=None, seeds=7, assignment=None):
    cfg = _load(preset, [BIND] if assignment is not None else [])
    ctrl = make_ctrl()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = runner.run(cfg, controller=ctrl if assignment is None else Declare(ctrl, assignment),
                         scenario=lib.static() if scenario is None else scenario, duration_s=duration_s, batch=batch,
                         seeds=seeds, uncertainty="sample" if sampled else "nominal",
                         variation="sample" if sampled else "off", reset_protocol="branch_low",
                         record=[("lam_res_m", 50, "max"), ("lam_res_m", 50, "min")])
    return ctrl, res


def _nearest_line(res, ring):
    """Line nearest the resonance of ``ring`` (B,) from the run's final Truth: lam_res (order nearest the declared
    line) against every line, wrapped by the local FSR; good to a fraction of the line spacing (enough to pick the
    line; the exact detuning comes from the declared second run)."""
    pp = res.plant.pp
    tr = res.truth_final
    lam = tr.fields["lam_res_m"][:, ring]
    lam_l = tr.lam_m
    fsr_hz = np.asarray(pp.optics.fsr_hz, float).reshape(lam.size, -1)[:, 0]
    fsr_m = fsr_hz * lam ** 2 / C0
    d = lam[:, None] - lam_l
    d = d - fsr_m[:, None] * np.round(d / fsr_m[:, None])
    return np.argmin(np.abs(d), axis=1)


def _exact(make_ctrl, preset, duration_s, batch=1, sampled=False, scenario=None, ring=0):
    """(controller of the second run, result, line (B,), detuning to that line (B,) pm, p-p of lam_res over the last
    quarter (B,) pm)."""
    _, res1 = _run(make_ctrl, preset, duration_s, batch, sampled, scenario)
    line = _nearest_line(res1, ring)
    asg = np.repeat(np.arange(8)[None, :], batch, axis=0)
    asg[:, ring] = line
    ctrl, res = _run(make_ctrl, preset, duration_s, batch, sampled, scenario, assignment=asg)
    det = res.truth_final.fields["detuning_m"][:, ring] * 1e12
    mx = [v for k, v in res.trace.items() if k.startswith("lam_res_m") and k.endswith("max")][0]["value"][:, :, ring]
    mn = [v for k, v in res.trace.items() if k.startswith("lam_res_m") and k.endswith("min")][0]["value"][:, :, ring]
    a = int(0.75 * mx.shape[0])
    pp_pm = (mx[a:].max(axis=0) - mn[a:].min(axis=0)) * 1e12
    return ctrl, res, line, det, pp_pm


def test_ramp_acquire_in_sampled_worlds():
    """Eight worlds of the g26 uncertainty set (07 4.8: FSR window, driver law, coupling regime, self-heating eta,
    thermal family, monitor gain, ...): ring 0 ends in LOCK on the red side of a line, steady (p-p < 1 pm)."""
    B = 8
    ctrl, res, line, det, pp_pm = _exact(lambda: RampAcquire(rings=[0]), "g26", 5e-3, batch=B, sampled=True)
    f = ctrl.fsm
    for b in range(B):
        print(f"\nworld {b}: line {line[b]}, detuning {det[b]:.1f} pm (red > 0), p-p {pp_pm[b]:.2f} pm, partial "
              f"{bool(f.partial[b, 0])}, clipped {bool(f.clipped[b, 0])}, jumps {f.jumps[b, 0]}", end="")
    assert np.all(f.state[:, 0] == ST_LOCK)
    assert np.all(det > 0.0) and np.all(det < 600.0), det
    assert np.all(pp_pm < 1.0), pp_pm


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["g26_hot_selfheat", "g26_low_eta", "g26_undercut", "g26_fsr_window_a",
                                    "g26_textbook_thermal", "g26_slow_thermal", "g26_overcoupled"])
def test_ramp_acquire_on_the_corner_presets(preset):
    """The named corners of 07 4.8: LOCK on the red side of a line, steady. hot_selfheat, low_eta and
    textbook_thermal end ``partial`` (the first line met has its peak below w = 0: a ring captured at turn-on), so
    their set-point is half of a lower bound of the peak (lock further red, reported)."""
    ctrl, res, line, det, pp_pm = _exact(lambda: RampAcquire(rings=[0]), preset, 4e-3)
    f = ctrl.fsm
    print(f"\n{preset}: line {line[0]}, detuning {det[0]:.1f} pm, p-p {pp_pm[0]:.2f} pm, partial "
          f"{bool(f.partial[0, 0])}, retries {f.retries[0, 0]}")
    assert f.state[0, 0] == ST_LOCK
    assert 0.0 < det[0] < 600.0 and pp_pm[0] < 1.0


def test_bitstat_lock_in_sampled_worlds():
    """The same eight worlds with the bit-statistics lock (ratio mode): LOCK on the red side, steady, the eye loop
    running (data_ok). The eye statistic is noisier than the monitor mean: a dim world (9 dB chip input loss) wanders
    ~2.5 pm p-p; a world whose monitor clips at the peak computes the eye ratio on a clipped amplitude (y_bg near
    full scale, R meaningless) and wanders ~6 pm p-p; reported, the clipped ones not bounded here."""
    B = 8
    ctrl, res, line, det, pp_pm = _exact(lambda: BitStatLock(rings=[0]), "g26", 6e-3, batch=B, sampled=True)
    f = ctrl.fsm
    for b in range(B):
        print(f"\nworld {b}: line {line[b]}, detuning {det[b]:.1f} pm, p-p {pp_pm[b]:.2f} pm, alpha "
              f"{ctrl.alpha_eye[b, 0]:.3f}, eye ratio {ctrl.ratio[b, 0]:.3f} (ref {ctrl.r_ref[b, 0]:.3f}), "
              f"clipped {bool(f.clipped[b, 0])}", end="")
    assert np.all(f.state[:, 0] == ST_LOCK) and np.all(ctrl.data_ok[:, 0])
    assert np.all(det > 0.0) and np.all(det < 600.0), det
    unclipped = ~f.clipped[:, 0]
    assert np.all(pp_pm[unclipped] < 5.0), pp_pm


def test_bitstat_ratio_lock_under_a_swing_loss():
    """A 20 % loss of the data swing at 3.5 ms (data.swing_scale 1 -> 0.8; a driver or bias change). The eye ratio
    R = 2 x dx / (1 + x^2) scales with the swing, and R_ref cannot be reached any more (the maximum of R/dx at x = 1
    is 1/0.866 of its value at alpha 3/4): the ratio loop lowers alpha to its floor 0.55 (the R maximum is at
    alpha 1/2; below it the law would run away). The inner lock holds; the lock point moves red (reported)."""
    sc = g.Scenario(name="swing", components=(g.step("data.swing_scale", 3.5e-3, -0.2),))
    ctrl, res, line, det, pp_pm = _exact(lambda: BitStatLock(rings=[0]), "g26", 9e-3, scenario=sc)
    mx = [v for k, v in res.trace.items() if k.startswith("lam_res_m") and k.endswith("max")][0]
    t = mx["t_s"]
    lam = mx["value"][:, 0, 0]
    before = float(np.mean(lam[(t > 2.5e-3) & (t < 3.5e-3)]))
    print(f"\nswing -20 %: alpha {ctrl.alpha_eye[0, 0]:.3f} (floor {ctrl.alpha_floor}), final detuning {det[0]:.1f} pm"
          f" (moved {(lam[-1] - before) * 1e12:+.1f} pm), p-p {pp_pm[0]:.2f} pm")
    assert ctrl.fsm.state[0, 0] == ST_LOCK
    assert ctrl.alpha_eye[0, 0] >= ctrl.alpha_floor - 1e-12
    # p-p of the last quarter (2.25 ms after the step): the ratio loop is still lowering alpha, so the lock point
    # still creeps. Measured 1.97 pm with the pre-modes_rib160 loss budget (eta 0.8546) and 2.98 pm after the g26
    # budget was re-solved for the landed artefact (eta 0.848; 2.79 pm at 12 ms): bound 4 pm (integrator, 2026-10-09)
    assert 0.0 < det[0] < 300.0 and pp_pm[0] < 4.0
