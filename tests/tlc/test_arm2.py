"""Second arm (mrm/control/tlc/arm2.py, hybrid.py): options parse, the default arm equals the faithful core, the
stiff-scale step decorrelates p, and the tlc_wdm hybrid is registered with a warm-start-only TLC."""
from __future__ import annotations

import numpy as np
import pytest

import mrm.params as P
from mrm.control import registry
from mrm.control.api import ControllerSpec
from mrm.control.tlc.arm2 import Arm2Options, Arm2TLC, _stiffen_scale
from mrm.control.tlc.core import TLC
from mrm.control.tlc.faithful_core import FaithfulCore
from mrm.control.tlc.priors import g26_priors

TIER = "controller_interface.knowledge=[line_grid,ring_to_line,model_nominal]"


@pytest.fixture(scope="module")
def priors():
    cfg = P.load("g26", overrides=[TIER])
    return g26_priors(ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg))


def test_unknown_option_is_refused():
    with pytest.raises(ValueError, match="unknown options"):
        Arm2Options.from_dict({"stiff": 0.1})
    assert Arm2Options.from_dict({"red_wait_s": 0.05}).red_wait_s == 0.05


def test_default_options_reproduce_the_faithful_core(priors):
    a, b = TLC(priors.pp, priors.cp), Arm2TLC(priors.pp, priors.cp, options=Arm2Options())
    u0 = np.full(priors.cp.n_rings, 0.28)
    for c in (a, b):
        c.warm_start(u0, c.sp.targets.copy())
    rng = np.random.default_rng(0)
    n = int(round(priors.cp.ts * priors.pp.pd_rate))
    for k in range(40):
        pd = 0.6 + 0.01 * rng.standard_normal((n, priors.cp.n_rings))
        np.testing.assert_array_equal(a.step(pd, k * priors.cp.ts), b.step(pd, k * priors.cp.ts))
    np.testing.assert_array_equal(a.ukf.x, b.ukf.x)


def test_stiffen_scale_decorrelates_and_caps_p(priors):
    c = TLC(priors.pp, priors.cp)
    n = c.N
    c.ukf.P[:] = 0.5 + np.eye(3 * n)
    rings = np.zeros(n, bool)
    rings[2] = True
    _stiffen_scale(c.ukf, 0.005, rings)
    s = 2 * n + 2
    assert c.ukf.P[s, s] == pytest.approx((0.005 * abs(c.ukf.x[s])) ** 2)
    assert not np.any(np.delete(c.ukf.P[s], s)) and not np.any(np.delete(c.ukf.P[:, s], s))
    assert c.ukf.P[2 * n + 3, 0] == 0.5                       # other rings untouched


def test_faithful_core_builds_the_arm2_tlc_only_when_asked():
    cfg = P.load("g26", overrides=[TIER])
    spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
    assert type(FaithfulCore(spec, ts=1e-4, n_out=10).ctl) is TLC
    assert isinstance(FaithfulCore(spec, ts=1e-4, n_out=10, arm2={"red_wait_s": 0.05}).ctl, Arm2TLC)
    with pytest.raises(ValueError):
        FaithfulCore(spec, ts=1e-4, n_out=10, kind="pi", arm2={"red_wait_s": 0.05})


def test_hybrid_is_registered_and_warm_starts_the_tlc():
    from mrm.control.tlc.hybrid import TlcWdmController
    assert registry.resolve("tlc_wdm") is TlcWdmController
    cfg = P.load("g26", overrides=["controller_interface.knowledge=[line_grid,ring_to_line,model_nominal,heater_nominal,"
                                   "optics_nominal,dlambda_dt_nominal,monitor_nominal]"])
    spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
    with pytest.raises(ValueError, match="warm"):
        TlcWdmController({"tlc": {"core": "faithful", "start_mode": "cold"}}).reset(spec)
