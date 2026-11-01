"""Checks the compiled kernel against the pure-NumPy reference engine (mrm.sim.kernel_ref.ReferenceEngine).

Read this first
---------------
The reference is written independently (bank classes, NumPy Truth, NumPy oracle with mrm.plant.optics.ring_exact,
metrics from the full trace). It must agree with the kernel bit for bit on every obs vector a controller receives
and on the Truth fields in COMPARED (ber to 1e-12 relative), and to rounding level on the accumulated metrics and
the oracle values (1e-12 to 1e-9 relative; see _assert_parity). The comparison runs open loop, with a Python
controller (slow path) and with a compiled controller (kernel path). The same comparison runs in a subprocess under
NUMBA_DISABLE_JIT=1 (the kernel is then plain Python, which is what lets a student step through it in pdb).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest
from numba import njit

from mrm import params
from mrm.control.api import (SPEC_I_N_RINGS, CompiledController, ControllerSpec, act_offset, obs_offset, user_i64)
from mrm.core.channels import ActId, ObsId
from mrm.plant.plant import Plant
from mrm.scenarios import registry as scenario_registry
from mrm.sim import kernel
from mrm.sim.kernel_ref import ReferenceEngine
from mrm.sim.metrics import TRUTH_FIELDS, MetricsSpec
from mrm.sim.runner import _run_python

LATEST = int(ObsId.TX_MONITOR_LATEST)   # observation id of the latest monitor code
HEATER = int(ActId.TX_HEATER)           # actuator id of the heater commands
# the Truth fields that _assert_parity compares exactly (ber to 1e-12 relative); oracle_reloma is compared
# separately to 1e-9 relative
COMPARED = [f for f in TRUTH_FIELDS if f not in ("er_db", "il_db", "monitor_code", "oracle_reloma")]


# ----------------------------------------------------------------------------------------------------- test controllers

# A compiled integrating controller: s accumulates (set-point - latest monitor code), and the heater command is
# 15000 + (s >> shift) - (2^30 >> shift). Its integer parameters p: [n_rings, latest-code offset in obs, heater offset
# in act, set-point code, gain shift]. s starts at 2^30.
@njit(cache=True)
def integrator_init(si, sf):
    p = np.zeros(5, np.int64)
    p[0] = si[SPEC_I_N_RINGS]
    p[1] = obs_offset(si, LATEST)
    p[2] = act_offset(si, HEATER)
    p[3] = user_i64(si, 0)          # set-point code
    p[4] = user_i64(si, 1)          # gain shift
    return p, np.zeros(1), np.full(p[0], 1 << 30, np.int64), np.zeros(1)


@njit(cache=True)
def integrator_step(p, pf, s, sf, obs, act):
    for i in range(p[0]):
        s[i] += p[3] - obs[p[1] + i]
        c = (s[i] >> p[4]) - (1 << (30 - p[4])) + 15000
        act[p[2] + i] = c                   # may leave [0, 65535]: the kernel clips and counts


class PyStair:
    """Deterministic Python controller: a heater staircase driven by the released monitor code."""
    name = "py_stair"
    latency_ticks = 2

    def reset(self, spec):
        self.spec = spec
        self.n = 0

    def step(self, obs):
        self.n += 1
        lat = np.asarray(obs["tx.monitor.latest"]).astype(np.int64)
        h = 6000 + 50 * self.n + (lat % 97)
        return {"tx.heater": h.astype(np.int64)}


# -------------------------------------------------------------------------------------------------------------- helpers

def _setup(B=2, n_ticks=400, lat=0, ovr=(), plant_kw=None, scenario=None, **metrics_kw):
    """(cfg, ControllerSpec, Plant) of g26 with B sampled worlds that records every Truth field for n_ticks.
    lat: controller latency in ticks; ovr: config overrides; plant_kw: extra Plant arguments; scenario: added to the
    config's scenario; metrics_kw: MetricsSpec options (oracle every 50 ticks)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26", overrides=["sim.reset_ramp_s=2e-4", *ovr])
        pp = params.realize(cfg, B, seeds=7)
    sc = scenario_registry.from_config(cfg)
    if scenario is not None:
        sc = sc + scenario
    spec = ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)
    plant = Plant(pp, sc, spec, noise_seed=3, scenario_seed=5, controller_latency_ticks=lat,
                  record=[(f, 1, "last") for f in TRUTH_FIELDS], record_ticks=n_ticks,
                  metrics=MetricsSpec.from_config(cfg, oracle_period_ticks=50, **metrics_kw), **(plant_kw or {}))
    return cfg, spec, plant


def _assert_parity(plant, ref, n_ticks):
    """Assert that the plant and the reference agree. The COMPARED Truth fields agree bit for bit (ber to 1e-12
    relative). oracle_reloma, the lock error, the lock reference, the expected errors and the OMA penalty agree to
    1e-9 relative, and the other metrics to 1e-12 relative."""
    tr = plant.traces()
    for f in COMPARED:
        k = tr[f"{f}@1:last"]["value"]
        r = ref.stacked(f)
        assert k.shape == r.shape == (n_ticks, plant.B, plant.N)
        if f == "ber":
            np.testing.assert_allclose(k, r, rtol=1e-12, atol=1e-200, err_msg=f)
        else:
            np.testing.assert_array_equal(k, r, err_msg=f)
    np.testing.assert_allclose(tr["oracle_reloma@1:last"]["value"], ref.stacked("oracle_reloma"), rtol=1e-9,
                               atol=1e-15)
    mk, mr = plant.metrics(), ref.metrics()
    for key in ("max_abs_detuning_pm", "mean_detuning_pm", "excursion_count", "time_unlocked_s", "first_lock_s",
                "wrong_line_events", "fsr_hop_events", "order_changes", "heater_energy_j", "code_max_used",
                "peak_slew_codes_per_s", "mean_reloma", "mean_p_abs_w", "heater_sat_s"):
        np.testing.assert_allclose(np.asarray(mk[key], float), np.asarray(mr[key], float), rtol=1e-12, atol=0,
                                   equal_nan=True, err_msg=key)
    # the lock error (oracle reference: the oracle's scan agrees to ~1e-15, not bit for bit)
    for key in ("max_abs_lock_error_pm", "rms_lock_error_pm"):
        np.testing.assert_allclose(mk[key], mr[key], rtol=1e-9, atol=1e-9, err_msg=key)
    np.testing.assert_allclose(mk["lock_reference_pm"], ref.lock_ref * 1e12, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(mk["expected_errors"], mr["expected_errors"], rtol=1e-9, atol=1e-300)
    np.testing.assert_allclose(mk["oma_penalty_db"], mr["oma_penalty_db"], rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------------------------------------- parity tests

def test_open_loop_parity():
    n = 400
    cfg, spec, plant = _setup(n_ticks=n)
    ref = ReferenceEngine(plant)
    ref.run(n)
    plant.advance(None, n)
    _assert_parity(plant, ref, n)
    last = max(ref.obs_log)
    np.testing.assert_array_equal(ref.obs_log[last], plant.S.obs)


def test_python_controller_parity():
    """Slow path: obs at every edge and the closed loop (controller latency 2 ticks) agree bit for bit."""
    n = 400
    cfg, spec, plant = _setup(n_ticks=n, lat=2)
    ref = ReferenceEngine(plant)
    c_ref = PyStair()
    c_ref.reset(spec)
    ref.run(n, c_ref)
    seen = {}

    class Spy(PyStair):
        def step(self, obs):
            seen[obs.tick] = spec.layout.pack_obs(obs.tick, obs.sensors, obs.new)
            return super().step(obs)
    c = Spy()
    c.reset(spec)
    _run_python(plant, c, spec, n, None)
    assert sorted(seen) == sorted(ref.obs_log)
    for k in seen:
        np.testing.assert_array_equal(seen[k], ref.obs_log[k], err_msg=f"obs at tick {k}")
    _assert_parity(plant, ref, n)


def test_compiled_controller_parity():
    """Kernel path (step inside the kernel, actions clipped and counted there) against the reference running the
    same CompiledController through its Python protocol."""
    n = 400
    cfg, spec, plant = _setup(n_ticks=n)
    ref = ReferenceEngine(plant)
    c_ref = CompiledController("integ", integrator_init, integrator_step, params_i64=[2000, 3])
    c_ref.reset(spec)
    ref.run(n, c_ref)
    c = CompiledController("integ", integrator_init, integrator_step, params_i64=[2000, 3])
    c.reset(spec)
    plant.bind_compiled(c)
    plant.run(n, mode=kernel.MODE_COMPILED, step_fn=c.step_fn)
    _assert_parity(plant, ref, n)
    np.testing.assert_array_equal(plant.S.obs, ref.obs_log[max(ref.obs_log)])
    np.testing.assert_array_equal(c.state[2], c_ref.state[2])
    sat_k = plant.S.sat[:, :plant.maps.act_size]
    assert sat_k.sum() == sum(v.sum() for v in plant.validator.saturation_counts().values())


def _reset_pulse(t_s=210.5e-6):
    """A die-common scenario controller reset (fault.controller_reset) rising at t_s (s), 3 us long."""
    from mrm.scenarios import generators
    return generators.Scenario(name="controller_reset", components=(
        generators.pwl("fault.controller_reset", [0.0, t_s, t_s, t_s + 3e-6, t_s + 3e-6, 1.0], [0, 0, 1, 1, 0, 0],
                       absolute=True),))


def _rich_scenario():
    """Every scenario path of the interval inputs at once: a square aggressor (FOH power), a laser failover with a dark
    gap (line enable, frequency and power steps), a fibre loss step, traffic start and stop (driver enable, swing,
    transition and ones density), a bias offset step, the four sensor and actuator faults and a controller reset."""
    from mrm.scenarios import generators
    from mrm.scenarios import library
    faults = generators.Scenario(name="faults", components=(
        generators.pulse("fault.adc_drop", 120e-6, 140e-6, 1.0, elements=(1,)),
        generators.pulse("fault.adc_stuck", 160e-6, 180e-6, 1.0, elements=(2,)),
        generators.pulse("fault.adc_saturate", 190e-6, 200e-6, 1.0, elements=(3,)),
        generators.pulse("fault.dac_glitch_codes", 230e-6, 240e-6, 500.0, elements=(4,))))
    return (library.aggressor_direct(power_w=0.1, period_s=200e-6, duty=0.5)
            + library.laser_failover(line=0, t_fail_s=100e-6, gap_s=20e-6)
            + library.fibre_loss_step(t_s=150e-6) + library.traffic_start_stop(t_start_s=50e-6, t_stop_s=300e-6)
            + library.bias_step(t_s=250e-6) + faults + _reset_pulse())


def test_compiled_controller_parity_through_a_controller_reset():
    """fault.controller_reset: the kernel restores the compiled controller's post-init state and syncs its held
    command at the first edge after the rising edge; the reference calls the controller's reset there (its
    CompiledController protocol re-runs init and syncs from the latched registers). A reference that ignored the
    fault would differ by 12684 heater codes from tick 220."""
    n = 400
    cfg, spec, plant = _setup(n_ticks=n, scenario=_reset_pulse())
    ref = ReferenceEngine(plant)
    c_ref = CompiledController("integ", integrator_init, integrator_step, params_i64=[2000, 3])
    c_ref.reset(spec)
    ref.run(n, c_ref)
    c = CompiledController("integ", integrator_init, integrator_step, params_i64=[2000, 3])
    c.reset(spec)
    plant.bind_compiled(c)
    plant.run(n, mode=kernel.MODE_COMPILED, step_fn=c.step_fn)
    _assert_parity(plant, ref, n)
    np.testing.assert_array_equal(c.state[2], c_ref.state[2])
    # the reset is visible: the integrator restarts at the edge after 210.5 us
    h = plant.traces()["heater_code@1:last"]["value"][:, 0, 0]
    assert abs(float(h[220]) - float(h[219])) > 1000.0, h[215:225]


@pytest.mark.parametrize("ctrl", ["python", "compiled"])
def test_parity_with_every_scenario_path(ctrl):
    """The parity under a scenario that drives every interval input the kernel reads from the scenario, including a
    controller reset (the other parity tests use only the config scenario)."""
    n = 400
    cfg, spec, plant = _setup(n_ticks=n, lat=1 if ctrl == "python" else 0, scenario=_rich_scenario())
    ref = ReferenceEngine(plant)
    if ctrl == "python":
        c_ref = PyStair()
        c_ref.reset(spec)
        ref.run(n, c_ref)
        c = PyStair()
        c.reset(spec)
        _run_python(plant, c, spec, n, None)
    else:
        c_ref = CompiledController("integ", integrator_init, integrator_step, params_i64=[2000, 3])
        c_ref.reset(spec)
        ref.run(n, c_ref)
        c = CompiledController("integ", integrator_init, integrator_step, params_i64=[2000, 3])
        c.reset(spec)
        plant.bind_compiled(c)
        plant.run(n, mode=kernel.MODE_COMPILED, step_fn=c.step_fn)
    _assert_parity(plant, ref, n)
    np.testing.assert_array_equal(plant.S.obs, ref.obs_log[max(ref.obs_log)])


@pytest.mark.parametrize("ctrl", ["python", "compiled"])
def test_round_robin_parity(ctrl):
    """controller_interface.schedule round_robin: the kernel (both paths) and the reference pack the active ring j % N
    of edge j into the observation and apply only that ring's heater command (the others hold), also through a
    controller reset (the inactive rings keep the held command of the edge on every path), with controllers that write
    every ring at every edge."""
    n = 400
    cfg, spec, plant = _setup(n_ticks=n, lat=1 if ctrl == "python" else 0, scenario=_reset_pulse(),
                              ovr=("controller_interface.schedule=round_robin",))
    assert spec.schedule == "round_robin"
    ref = ReferenceEngine(plant)
    if ctrl == "python":
        c_ref = PyStair()
        c_ref.reset(spec)
        ref.run(n, c_ref)
        c = PyStair()
        c.reset(spec)
        _run_python(plant, c, spec, n, None)
    else:
        c_ref = CompiledController("integ", integrator_init, integrator_step, params_i64=[2000, 3])
        c_ref.reset(spec)
        ref.run(n, c_ref)
        c = CompiledController("integ", integrator_init, integrator_step, params_i64=[2000, 3])
        c.reset(spec)
        plant.bind_compiled(c)
        plant.run(n, mode=kernel.MODE_COMPILED, step_fn=c.step_fn)
        np.testing.assert_array_equal(c.act, c_ref.act)
        sat_k = plant.S.sat[:, :plant.maps.act_size]
        assert sat_k.sum() == sum(v.sum() for v in plant.validator.saturation_counts().values())
    _assert_parity(plant, ref, n)
    for k, o in ref.obs_log.items():
        assert o[0, 1] == ((k - spec.phase_ticks) // spec.period_ticks) % spec.n_rings      # OBS_ACTIVE_RING
    np.testing.assert_array_equal(plant.S.obs, ref.obs_log[max(ref.obs_log)])


# the Python-controller parity of this module, short; test_parity_without_jit runs it with NUMBA_DISABLE_JIT=1
_NO_JIT_SCRIPT = textwrap.dedent("""
    import numpy as np, warnings
    warnings.simplefilter("ignore")
    import tests.verification.test_kernel_parity as parity_tests
    n = 25
    cfg, spec, plant = parity_tests._setup(B=2, n_ticks=n, lat=1)
    ref = parity_tests.ReferenceEngine(plant)
    c_ref = parity_tests.PyStair(); c_ref.reset(spec); ref.run(n, c_ref)
    c = parity_tests.PyStair(); c.reset(spec)
    parity_tests._run_python(plant, c, spec, n, None)
    parity_tests._assert_parity(plant, ref, n)
    print("PARITY-OK")
""")


@pytest.mark.parametrize("ovr,metrics_kw", [
    (["fidelity.pattern_noise=prbs7"], {}),                        # PRBS data pattern: its own period
    (['fidelity.backscatter="on"'], {}),   # exact ring with backscatter
    (['fidelity.self_heating="off"'], {}),                         # explicit bus pass, no implicit solve
    ([], {"lock_reference": "oracle"}),                            # lock error about the oracle's operating point
], ids=["prbs7", "exact_backscatter", "selfheat_off", "lock_ref_oracle"])
def test_parity_fidelity_variants(ovr, metrics_kw):
    """The same parity under the other fidelity switches the kernel branches on (Python controller, latency 1)."""
    n = 300
    cfg, spec, plant = _setup(n_ticks=n, lat=1, ovr=ovr, **metrics_kw)
    ref = ReferenceEngine(plant)
    c_ref = PyStair()
    c_ref.reset(spec)
    ref.run(n, c_ref)
    c = PyStair()
    c.reset(spec)
    _run_python(plant, c, spec, n, None)
    _assert_parity(plant, ref, n)
    if metrics_kw.get("lock_reference") == "oracle":
        assert np.all(np.abs(plant.metrics()["lock_reference_pm"]) > 1.0)        # the operating point is off-line




def test_parity_without_jit():
    """The same parity with NUMBA_DISABLE_JIT=1 (kernel and reference both plain Python)."""
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, NUMBA_DISABLE_JIT="1", PYTHONPATH=str(root))
    res = subprocess.run([sys.executable, "-c", _NO_JIT_SCRIPT], cwd=root, env=env, capture_output=True, text=True,
                         timeout=600)
    assert "PARITY-OK" in res.stdout, res.stdout[-2000:] + res.stderr[-4000:]


def test_swing_capacitance_equals_electrical():
    """The engine's cumulative-table C_eff (kernel.swing_capacitance_row) equals
    mrm.plant.electrical.junction_swing_capacitance_row, inside, across and outside the grid."""
    from mrm.plant import electrical
    rng = np.random.default_rng(0)
    grid = np.linspace(-0.5, 4.0, 451)
    c = ((1.0 + 0.3 * np.cos(3 * grid)) * 1e-10)[None, :]
    cum = kernel.cumulative_capacitance(grid, c)
    for _ in range(20):
        lo = rng.uniform(-1.0, 4.5, 8)
        hi = lo + rng.uniform(0.0, 2.0, 8)
        hi[0] = lo[0] + 1e-12                         # the vanishing-swing branch
        a, b = np.zeros(8), np.zeros(8)
        electrical.junction_swing_capacitance_row(lo, hi, grid, c[0], np.full(8, 0.6), np.full(8, 4e-5), a)
        kernel.swing_capacitance_row(lo, hi, grid, c[0], cum[0], np.full(8, 0.6), np.full(8, 4e-5), b)
        np.testing.assert_allclose(b, a, rtol=1e-12)


def test_kernel_is_disk_cacheable():
    """cache=True: the compiled chunk function has no dynamic globals (dispatcher addresses embedded in the
    code would make numba refuse to cache it, and every process would compile the plant again)."""
    cfg, spec, plant = _setup(B=1, n_ticks=0)
    plant.advance(None, 2)
    if not kernel._RUN_CHUNKS:
        pytest.skip("NUMBA_DISABLE_JIT")
    for chunk in kernel._RUN_CHUNKS.values():
        for cres in chunk.overloads.values():
            assert not cres.library.has_dynamic_globals


def test_parity_with_general_optics():
    """the compiled and reference event loops share one physical ring model and agree within floating-point tolerance."""
    n = 400
    cfg, spec, plant = _setup(n_ticks=n)
    ref = ReferenceEngine(plant)
    ref.run(n)
    plant.advance(None, n)
    tr = plant.traces()
    exact = ("heater_code", "order", "line_on", "declared_line")
    for f in COMPARED:
        k = tr[f"{f}@1:last"]["value"]
        r = ref.stacked(f)
        if f in exact:
            np.testing.assert_array_equal(k, r, err_msg=f)
            continue
        scale = np.max(np.abs(r[np.isfinite(r)])) if np.isfinite(r).any() else 0.0
        np.testing.assert_allclose(k, r, rtol=0, atol=1e-9 * scale + 1e-300, err_msg=f)
    np.testing.assert_array_equal(ref.obs_log[max(ref.obs_log)], plant.S.obs)


# ------------------------------------------------------------------------------- reference counts (runtime-free kernel)

# A fast kernel that counts the new monitor samples of each ring and toggles a +-40 code dither on each one.
@njit(cache=True)
def _count_samples_fast(p, pf, s, sf, fobs, fact):
    n = fobs[1]                     # fobs = [tick, N, new[N], code[N], sample_tick[N]] (mrm.control.api)
    for i in range(n):
        if fobs[2 + i] != 0:
            s[i] += 1
            fact[i] = 40 if s[i] % 2 == 0 else -40


def _array_refcounts(pl):
    """The Python reference count of every KParams / KState array of the plant."""
    return [sys.getrefcount(a) for a in (*pl.P, *pl.S) if isinstance(a, np.ndarray)]


@pytest.mark.parametrize("ovr,plant_kw", [
    ((), {}),                                                       # Airy path, oracle every 50 ticks
    ((), {"substep_policy": "auto"}),                               # auto policy (_auto_substep_a)
    (('fidelity.backscatter="on"',), {}),   # the general optics path
    (('fidelity.self_heating="off"',), {}),                        # explicit bus pass
], ids=["airy", "auto", "exact_backscatter", "selfheat_off"])
def test_kernel_releases_every_array(ovr, plant_kw):
    """The tick loop and the hot leaves are compiled without the numba runtime (docs/ENGINE.md, "Runtime-free hot
    loops"). Such a function must not allocate (numba refuses to compile that) and must not call a function that returns
    an array: it would never release that reference, and the plant's arrays would outlive the Plant. A function with the
    runtime called from it must not create views of its arguments either (measured: numba 0.68 then leaks one reference
    per block). Every KParams / KState array keeps its Python reference count over blocks of ticks: open loop (oracle
    every 50 ticks), with a compiled controller (edges, the controller step), with a Python controller (the kernel stops
    and resumes at every edge) and with a fast_kernel, and over Truth."""
    n = 300
    cfg, spec, plant = _setup(n_ticks=n, ovr=ovr, plant_kw=plant_kw)
    plant.advance(None, 20)                                         # compile and warm up
    before = _array_refcounts(plant)
    plant.advance(None, n)
    plant.truth()
    assert _array_refcounts(plant) == before
    if plant_kw or ovr:
        return
    cfg, spec, plant = _setup(n_ticks=n, plant_kw={"fast_kernel": True})
    c = CompiledController("integ", integrator_init, integrator_step, params_i64=[2000, 3], fast_fn=_count_samples_fast)
    c.reset(spec)
    plant.bind_compiled(c)
    plant.run(20, mode=kernel.MODE_COMPILED, step_fn=c.step_fn, fast_fn=c.fast_fn)
    before = _array_refcounts(plant)
    plant.run(n, mode=kernel.MODE_COMPILED, step_fn=c.step_fn, fast_fn=c.fast_fn)
    assert _array_refcounts(plant) == before
    assert int(plant.S.world_int[0, kernel.WI_FAST_CALLS]) > 0
    cfg, spec, plant = _setup(n_ticks=n, lat=1)
    py = PyStair()
    py.reset(spec)
    _run_python(plant, py, spec, 20, None)
    before = _array_refcounts(plant)
    _run_python(plant, py, spec, n, None)
    assert _array_refcounts(plant) == before


@pytest.mark.parametrize("ovr", [(), ('fidelity.backscatter="on"',)], ids=["airy", "backscatter"])
def test_full_optics_outputs_after_every_block(ovr):
    """all optical outputs, including heating sensitivities, agree between compiled and reference loops after each block."""
    from mrm.plant import optics

    blocks = (1, 2, 7, 13, 1, 64, 3, 29)
    cfg, spec, plant = _setup(n_ticks=sum(blocks), ovr=ovr)
    ref = ReferenceEngine(plant)
    for m in blocks:
        ref.run(m)
        plant.advance(None, m)
        for b in range(plant.B):
            vk = optics.selfheat_views(plant.S.selfheat_float[b], plant.P.selfheat_int[b])
            vr = optics.selfheat_views(ref.packed_f[b], ref.packed_i[b])
            for i, name, skip in (
                (
                    optics.SELF_HEATING_VIEW_INDEX_RING_LINE_OUTPUTS,
                    "out_rl",
                    optics.RING_LINE_OUTPUT_ROW_ABSORBED_POWER_ROUND_TRIP_PHASE_DERIVATIVE,
                ),
                (
                    optics.SELF_HEATING_VIEW_INDEX_RING_OUTPUTS,
                    "out_r",
                    optics.RING_OUTPUT_ROW_ABSORBED_POWER_RESONANCE_WAVELENGTH_DERIVATIVE,
                ),
                (optics.SELF_HEATING_VIEW_INDEX_LASER_LINE_OUTPUTS, "out_l", -1),
            ):
                for f in range(vk[i].shape[0]):
                    np.testing.assert_array_equal(
                        vk[i][f], vr[i][f], err_msg=f"{name}[{f}] after a block of {m} ticks (world {b})"
                    )
