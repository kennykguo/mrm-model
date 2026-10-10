"""07 P1-18: the compiled kernel against the pure-NumPy reference (mrm.sim.kernel_ref.ReferenceEngine; owner: engine).

The reference is written independently (bank classes, NumPy Truth, NumPy oracle with mrm.plant.optics.ring_exact,
metrics from the full trace) and must agree with the kernel bit for bit on every Truth field, every obs vector a
controller receives, and the accumulated metrics; open loop, with a Python controller (slow path) and with a
compiled controller (kernel path). The same comparison runs in a subprocess under NUMBA_DISABLE_JIT=1 (the kernel is
then plain Python, which is what lets a student step through it in pdb).
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
from mrm.scenarios import registry as scen_registry
from mrm.sim import kernel as K
from mrm.sim.kernel_ref import ReferenceEngine
from mrm.sim.metrics import TRUTH_FIELDS, MetricsSpec
from mrm.sim.runner import _run_python

LATEST = int(ObsId.TX_MONITOR_LATEST)
HEATER = int(ActId.TX_HEATER)
COMPARED = [f for f in TRUTH_FIELDS if f not in ("er_db", "il_db", "monitor_code", "oracle_reloma")]


@njit(cache=True)
def integ_init(si, sf):
    p = np.zeros(5, np.int64)
    p[0] = si[SPEC_I_N_RINGS]
    p[1] = obs_offset(si, LATEST)
    p[2] = act_offset(si, HEATER)
    p[3] = user_i64(si, 0)          # set-point code
    p[4] = user_i64(si, 1)          # gain shift
    return p, np.zeros(1), np.full(p[0], 1 << 30, np.int64), np.zeros(1)


@njit(cache=True)
def integ_step(p, pf, s, sf, obs, act):
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


def _setup(B=2, n_ticks=400, lat=0, ovr=(), plant_kw=None, **metrics_kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26", overrides=["sim.reset_ramp_s=2e-4", *ovr])
        pp = params.realize(cfg, B, seeds=7)
    sc = scen_registry.from_config(cfg)
    spec = ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)
    plant = Plant(pp, sc, spec, noise_seed=3, scenario_seed=5, controller_latency_ticks=lat,
                  record=[(f, 1, "last") for f in TRUTH_FIELDS], record_ticks=n_ticks,
                  metrics=MetricsSpec.from_config(cfg, oracle_period_ticks=50, **metrics_kw), **(plant_kw or {}))
    return cfg, spec, plant


def _compare(plant, ref, n_ticks):
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
                "wrong_line_events", "fsr_hop_events", "heater_energy_j", "code_max_used", "peak_slew_codes_per_s",
                "mean_reloma", "mean_p_abs_w", "heater_sat_s"):
        np.testing.assert_allclose(np.asarray(mk[key], float), np.asarray(mr[key], float), rtol=1e-12, atol=0,
                                   equal_nan=True, err_msg=key)
    # the lock error (oracle reference: the oracle's scan agrees to ~1e-15, not bit for bit)
    for key in ("max_abs_lock_error_pm", "rms_lock_error_pm"):
        np.testing.assert_allclose(mk[key], mr[key], rtol=1e-9, atol=1e-9, err_msg=key)
    np.testing.assert_allclose(mk["lock_reference_pm"], ref.lock_ref * 1e12, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(mk["expected_errors"], mr["expected_errors"], rtol=1e-9, atol=1e-300)
    np.testing.assert_allclose(mk["oma_penalty_db"], mr["oma_penalty_db"], rtol=1e-9, atol=1e-12)


def test_open_loop_parity():
    n = 400
    cfg, spec, plant = _setup(n_ticks=n)
    ref = ReferenceEngine(plant)
    ref.run(n)
    plant.advance(None, n)
    _compare(plant, ref, n)
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
    _compare(plant, ref, n)


def test_compiled_controller_parity():
    """Kernel path (step inside the kernel, actions clipped and counted there) against the reference running the
    same CompiledController through its Python protocol."""
    n = 400
    cfg, spec, plant = _setup(n_ticks=n)
    ref = ReferenceEngine(plant)
    c_ref = CompiledController("integ", integ_init, integ_step, params_i64=[2000, 3])
    c_ref.reset(spec)
    ref.run(n, c_ref)
    c = CompiledController("integ", integ_init, integ_step, params_i64=[2000, 3])
    c.reset(spec)
    plant.bind_compiled(c)
    plant.run(n, mode=K.MODE_COMPILED, step_fn=c.step_fn)
    _compare(plant, ref, n)
    np.testing.assert_array_equal(plant.S.obs, ref.obs_log[max(ref.obs_log)])
    np.testing.assert_array_equal(c.state[2], c_ref.state[2])
    sat_k = plant.S.sat[:, :plant.maps.act_size]
    assert sat_k.sum() == sum(v.sum() for v in plant.validator.saturation_counts().values())


_SUBPROCESS = textwrap.dedent("""
    import numpy as np, warnings
    warnings.simplefilter("ignore")
    import tests.verification.test_kernel_parity as T
    n = 25
    cfg, spec, plant = T._setup(B=2, n_ticks=n, lat=1)
    ref = T.ReferenceEngine(plant)
    c_ref = T.PyStair(); c_ref.reset(spec); ref.run(n, c_ref)
    c = T.PyStair(); c.reset(spec)
    T._run_python(plant, c, spec, n, None)
    T._compare(plant, ref, n)
    print("PARITY-OK")
""")


@pytest.mark.parametrize("ovr,metrics_kw", [
    (["fidelity.pattern_noise=prbs7"], {}),                        # PRBS pattern (07 P0-11): its own period
    (['fidelity.backscatter="on"', "fidelity.optics_model=tcmt"], {}),   # TCMT with backscatter (07 P2-2, P2-4)
    (['fidelity.self_heating="off"'], {}),                         # explicit bus pass, no implicit solve
    ([], {"lock_reference": "oracle"}),                            # lock error about the oracle's operating point
], ids=["prbs7", "tcmt_backscatter", "selfheat_off", "lock_ref_oracle"])
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
    _compare(plant, ref, n)
    if metrics_kw.get("lock_reference") == "oracle":
        assert np.all(np.abs(plant.metrics()["lock_reference_pm"]) > 1.0)        # the operating point is off-line


def test_parity_with_far_line_hold():
    """The far-line hold of the self-heating optics (Plant(optics_far_hold=True), a bounded approximation;
    docs/handoff/perf.md P5) lives in the shared optics adapter, so the kernel and the reference must still agree bit
    for bit with it on (Python controller, latency 1)."""
    n = 300
    cfg, spec, plant = _setup(n_ticks=n, lat=1, plant_kw={"optics_far_hold": True})
    ref = ReferenceEngine(plant)
    c_ref = PyStair()
    c_ref.reset(spec)
    ref.run(n, c_ref)
    c = PyStair()
    c.reset(spec)
    _run_python(plant, c, spec, n, None)
    _compare(plant, ref, n)


def test_parity_without_jit():
    """07 P1-18: the same parity with NUMBA_DISABLE_JIT=1 (kernel and reference both plain Python)."""
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, NUMBA_DISABLE_JIT="1", PYTHONPATH=str(root))
    res = subprocess.run([sys.executable, "-c", _SUBPROCESS], cwd=root, env=env, capture_output=True, text=True,
                         timeout=600)
    assert "PARITY-OK" in res.stdout, res.stdout[-2000:] + res.stderr[-4000:]


def test_swing_capacitance_equals_electrical():
    """The engine's cumulative-table C_eff (kernel.swing_capacitance_row) equals
    mrm.plant.electrical.junction_swing_capacitance_row (07 P1-8), inside, across and outside the grid."""
    from mrm.plant import electrical as el
    rng = np.random.default_rng(0)
    grid = np.linspace(-0.5, 4.0, 451)
    c = ((1.0 + 0.3 * np.cos(3 * grid)) * 1e-10)[None, :]
    cum = K.cumulative_capacitance(grid, c)
    for _ in range(20):
        lo = rng.uniform(-1.0, 4.5, 8)
        hi = lo + rng.uniform(0.0, 2.0, 8)
        hi[0] = lo[0] + 1e-12                         # the vanishing-swing branch
        a, b = np.zeros(8), np.zeros(8)
        el.junction_swing_capacitance_row(lo, hi, grid, c[0], np.full(8, 0.6), np.full(8, 4e-5), a)
        K.swing_capacitance_row(lo, hi, grid, c[0], cum[0], np.full(8, 0.6), np.full(8, 4e-5), b)
        np.testing.assert_allclose(b, a, rtol=1e-12)


def test_kernel_is_disk_cacheable():
    """07 P0-9 cache=True: the compiled chunk function has no dynamic globals (dispatcher addresses embedded in the
    code would make numba refuse to cache it, and every process would compile the plant again)."""
    cfg, spec, plant = _setup(B=1, n_ticks=0)
    plant.advance(None, 2)
    if not K._RUN_CHUNKS:
        pytest.skip("NUMBA_DISABLE_JIT")
    for chunk in K._RUN_CHUNKS.values():
        for cres in chunk.overloads.values():
            assert not cres.library.has_dynamic_globals


def test_parity_against_independent_optics():
    """The bit-for-bit parity above runs the reference with the engine's own Airy path of the self-heating optics
    (shared code, mrm.plant.optics sh_fast_*), so it does not check that path against an independent implementation
    (review of performance round 1, docs/handoff/perf.md section 6, item 4). Here the reference runs the general ring
    kernels instead (ReferenceEngine(airy=False): the closed forms of _airy_line and the separate bound kernels), and
    the engine agrees with it to floating-point rounding: every Truth field within 1e-9 of the field's largest value
    (measured ~1e-11; the Newton root tolerance is 1e-7 K), every heater code, order and line identical, and the
    controller observations (ADC codes) of the last edge identical."""
    n = 400
    cfg, spec, plant = _setup(n_ticks=n)
    ref = ReferenceEngine(plant, airy=False)
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



@njit(cache=True)
def _fast_count(p, pf, s, sf, fobs, fact):
    n = fobs[1]
    for i in range(n):
        if fobs[2 + i] != 0:
            s[i] += 1
            fact[i] = 40 if s[i] % 2 == 0 else -40


def _refcounts(pl):
    return [sys.getrefcount(a) for a in (*pl.P, *pl.S) if isinstance(a, np.ndarray)]


@pytest.mark.parametrize("ovr,plant_kw", [
    ((), {}),                                                       # Airy path, oracle every 50 ticks
    ((), {"substep_policy": "auto"}),                               # auto policy (_auto_h_a)
    (('fidelity.backscatter="on"', "fidelity.optics_model=tcmt"), {}),   # the general optics path
    (('fidelity.self_heating="off"',), {}),                        # explicit bus pass
], ids=["airy", "auto", "tcmt_backscatter", "selfheat_off"])
def test_kernel_releases_every_array(ovr, plant_kw):
    """The tick loop and the hot leaves are compiled without the numba runtime (docs/handoff/perf.md R2-11). Such a
    function must not allocate (numba refuses to compile that) and must not call a function that returns an array:
    it would never release that reference, and the plant's arrays would outlive the Plant. A function with the
    runtime called from it must not create views of its arguments either (measured: numba 0.68 then leaks one
    reference per block). Every KParams / KState array keeps its Python reference count over blocks of ticks: open
    loop (oracle every 50 ticks), with a compiled controller (edges, the controller step), with a Python controller
    (the kernel stops and resumes at every edge) and with a fast_kernel, and over Truth."""
    n = 300
    cfg, spec, plant = _setup(n_ticks=n, ovr=ovr, plant_kw=plant_kw)
    plant.advance(None, 20)                                         # compile and warm up
    before = _refcounts(plant)
    plant.advance(None, n)
    plant.truth()
    assert _refcounts(plant) == before
    if plant_kw or ovr:
        return
    cfg, spec, plant = _setup(n_ticks=n, plant_kw={"fast_kernel": True})
    c = CompiledController("integ", integ_init, integ_step, params_i64=[2000, 3], fast_fn=_fast_count)
    c.reset(spec)
    plant.bind_compiled(c)
    plant.run(20, mode=K.MODE_COMPILED, step_fn=c.step_fn, fast_fn=c.fast_fn)
    before = _refcounts(plant)
    plant.run(n, mode=K.MODE_COMPILED, step_fn=c.step_fn, fast_fn=c.fast_fn)
    assert _refcounts(plant) == before
    assert int(plant.S.wi[0, K.WI_FAST_CALLS]) > 0
    cfg, spec, plant = _setup(n_ticks=n, lat=1)
    py = PyStair()
    py.reset(spec)
    _run_python(plant, py, spec, 20, None)
    before = _refcounts(plant)
    _run_python(plant, py, spec, n, None)
    assert _refcounts(plant) == before


@pytest.mark.parametrize("ovr", [(), ('fidelity.backscatter="on"',)], ids=["airy", "backscatter"])
def test_full_optics_outputs_after_every_block(ovr):
    """docs/handoff/perf.md R2-3: inside a block the engine commits only the optics outputs a tick reads, and the last
    step of every block commits in full. After every block, every optics output (out_rl, out_r, out_l, also the
    fields no tick reads: RL_DROP, RL_REFL, RL_UCW, RL_DABS_DPHI, R_LOSS_SUM, R_DABS_DLAM, L_REFL_IN, L_BOUND, ...)
    must equal the reference's (which commits in full at every step) bit for bit, for blocks of irregular length
    (1 tick to 64 ticks; Plant.truth() reads these arrays after a block). The one known difference: with backscatter
    (the general path) the engine also writes the slope outputs RL_DABS_DPHI / R_DABS_DLAM, which the optics adapter
    the reference runs leaves at zero (kernel.py, comment above EXT_I); they are compared on the Airy path only."""
    from mrm.plant import optics as op
    blocks = (1, 2, 7, 13, 1, 64, 3, 29)
    cfg, spec, plant = _setup(n_ticks=sum(blocks), ovr=ovr)
    ref = ReferenceEngine(plant)
    general = bool(ovr)
    for m in blocks:
        ref.run(m)
        plant.advance(None, m)
        for b in range(plant.B):
            vk = op.selfheat_views(plant.S.sh_args_f[b], plant.P.sh_args_i[b])
            vr = op.selfheat_views(ref.args_f[b], ref.args_i[b])
            for i, name, skip in ((22, "out_rl", op.RL_DABS_DPHI), (23, "out_r", op.R_DABS_DLAM), (24, "out_l", -1)):
                for f in range(vk[i].shape[0]):
                    if general and f == skip:
                        continue
                    np.testing.assert_array_equal(vk[i][f], vr[i][f],
                                                  err_msg=f"{name}[{f}] after a block of {m} ticks (world {b})")
