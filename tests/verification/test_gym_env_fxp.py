"""Checks the fixed-point MLP helper mrm.control.fxp_mlp and the compiled MLP policy of the gym environment.

Read this first
---------------
A trained policy network (a small MLP) can run inside the engine as a compiled controller, in integer (Q format)
arithmetic as an FPGA would run it. The tests check the helper bit for bit against an independent big-integer
reference in every rounding and overflow mode, the accumulator-width proof, the float-vs-fixed error bound, the
packed parameter vector, and NUMBA_DISABLE_JIT. Then they check that the compiled policy inside the kernel rebuilds
the env's observation exactly and gives the same commands as the env (env == engine, command for command).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from fractions import Fraction

import numba as nb
import numpy as np
import pytest

from mrm.control import fxp_mlp
from mrm.control.fxp import Overflow, Rounding
from mrm.sim import runner
from mrm.sim.gym_env import MRMEnv, fxp_policy_action, mlp_policy_step

JIT_OFF = bool(nb.config.DISABLE_JIT)   # this process runs with numba off (NUMBA_DISABLE_JIT=1)


# ---------------------------------------------------------------------- exact reference (Python integers and fractions)

def _ref_round(q: Fraction, mode: Rounding) -> int:
    """The exact integer that rounding mode ``mode`` gives for the rational q."""
    f = math.floor(q)
    d = q - f
    if mode == Rounding.FLOOR:
        return f
    if mode == Rounding.HALF_UP:
        return f + 1 if d >= Fraction(1, 2) else f
    if mode == Rounding.HALF_EVEN:
        return f + 1 if (d > Fraction(1, 2) or (d == Fraction(1, 2) and f % 2 == 1)) else f
    if mode == Rounding.TOWARD_ZERO:
        return math.trunc(q)
    a = abs(q)
    r = math.floor(a + Fraction(1, 2))
    return r if q >= 0 else -r


def _ref_fit(x: int, bits: int, ovf: Overflow) -> int:
    """x fitted into a signed bits-wide integer by overflow mode ovf (OverflowError for CHECKED)."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if lo <= x <= hi:
        return x
    if ovf == Overflow.SATURATE:
        return lo if x < lo else hi
    if ovf == Overflow.CHECKED:
        raise OverflowError
    y = x % (1 << bits)
    return y - (1 << bits) if y >= (1 << (bits - 1)) else y


def ref_forward(mlp: fxp_mlp.FxpMlp, x_raw) -> list[int]:
    """The module docstring's semantics with Python integers and fractions (independent of the numba code)."""
    rnd, ovf = Rounding(mlp.rounding), Overflow(mlp.overflow)
    x = [int(v) for v in x_raw]
    xf = mlp.in_fmt.frac_bits
    for layer in mlp.layers:
        f_acc = layer.w_fmt.frac_bits + xf
        out = []
        for o in range(layer.n_out):
            s = f_acc - layer.b_frac
            b = int(layer.b_raw[o])
            acc = _ref_fit(b << s if s >= 0 else _ref_round(Fraction(b, 1 << -s), rnd), layer.acc_bits, ovf)
            for i in range(layer.n_in):
                acc = _ref_fit(acc + _ref_fit(int(layer.w_raw[o, i]) * x[i], layer.acc_bits, ovf), layer.acc_bits, ovf)
            z = _ref_round(Fraction(acc, 1 << (f_acc - layer.out_fmt.frac_bits)), rnd)
            if layer.activation == fxp_mlp.ACT_HARDTANH:
                one = 1 << layer.out_fmt.frac_bits
                out.append(_ref_fit(min(max(z, -one), one), layer.out_fmt.bits, Overflow.SATURATE))
            else:
                if layer.activation == fxp_mlp.ACT_RELU:
                    z = max(z, 0)
                out.append(_ref_fit(z, layer.out_fmt.bits, ovf))
        x = out
        xf = layer.out_fmt.frac_bits
    return x


def _random_network(rng, sizes, acts, scale=1.0):
    """Random float weights and biases for layer widths ``sizes``; returns (weights, biases, activations)."""
    ws = [rng.normal(0, scale / np.sqrt(sizes[i]), (sizes[i + 1], sizes[i])) for i in range(len(sizes) - 1)]
    bs = [rng.normal(0, 0.2 * scale, sizes[i + 1]) for i in range(len(sizes) - 1)]
    return ws, bs, acts


# ------------------------------------------------------------------------------------------- the fixed-point MLP helper

@pytest.mark.parametrize("rounding", [r.name.lower() for r in Rounding])
def test_forward_is_bit_exact_against_the_big_integer_reference(rounding):
    rng = np.random.default_rng(hash(rounding) % 2**32)
    for trial in range(6):
        sizes = [int(v) for v in rng.integers(2, 9, 4)]
        acts = [str(a) for a in rng.choice(["linear", "relu", "hardtanh"], 3)]
        ws, bs, acts = _random_network(rng, sizes, acts, scale=float(rng.uniform(0.3, 3.0)))
        w_bits, act_bits = int(rng.choice([8, 12, 16])), int(rng.choice([8, 12, 16]))
        m = fxp_mlp.quantize_mlp(ws, bs, acts, w_bits=w_bits, act_bits=act_bits, rounding=rounding)
        assert m.proven
        fin = m.in_fmt
        x = rng.integers(fin.raw_min, fin.raw_max + 1, (40, sizes[0]))
        x[0] = fin.raw_min
        x[1] = fin.raw_max
        y = m.forward(x)
        for b in range(x.shape[0]):
            assert list(y[b]) == ref_forward(m, x[b])
        # the per-addition fitted path gives the same numbers when no overflow is possible
        slow = dataclasses.replace(m, proven=False, _packed=None)
        assert np.array_equal(slow.forward(x), y)


@pytest.mark.parametrize("overflow", ["wrap", "saturate"])
def test_narrow_accumulators_wrap_or_saturate_like_the_reference(overflow):
    """Accumulators 1-3 bits narrower than the proof (the bias, held at the accumulator's fraction bits, still
    fits): the per-addition fit of the numba path equals the reference, and some inputs do overflow."""
    rng = np.random.default_rng(3 if overflow == "wrap" else 4)
    differ = 0
    for trial in range(8):
        ws, bs, acts = _random_network(rng, [6, 5, 3], ["relu", "linear"], scale=4.0)
        m0 = fxp_mlp.quantize_mlp(ws, bs, acts, overflow=overflow)
        narrow = [max(layer.acc_bits_needed - int(rng.integers(1, 4)), int(np.abs(layer.b_raw).max()).bit_length() + 1)
                  for layer in m0.layers]
        m = fxp_mlp.quantize_mlp(ws, bs, acts, overflow=overflow, acc_bits=narrow)
        assert not m.proven
        x = rng.integers(m.in_fmt.raw_min, m.in_fmt.raw_max + 1, (60, 6))
        x[:6] = np.where(m.layers[0].w_raw >= 0, m.in_fmt.raw_max, m.in_fmt.raw_min)[:5].repeat(2, 0)[:6]
        y = m.forward(x)
        for b in range(x.shape[0]):
            assert list(y[b]) == ref_forward(m, x[b])
        differ += int(not np.array_equal(y, m0.forward(x)))
    assert differ >= 2                                       # overflow was exercised


def test_checked_overflow_raises():
    rng = np.random.default_rng(9)
    ws, bs, acts = _random_network(rng, [8, 4], ["linear"], scale=8.0)
    m0 = fxp_mlp.quantize_mlp(ws, bs, acts)
    m = fxp_mlp.quantize_mlp(ws, bs, acts, overflow="checked", acc_bits=[m0.layers[0].acc_bits_needed - 2])
    x = np.where(np.asarray(m.layers[0].w_raw[0]) >= 0, m.in_fmt.raw_max, m.in_fmt.raw_min)[None, :]
    with pytest.raises(OverflowError):
        m.forward(x)


def test_accumulator_proof_is_tight_and_exact():
    """The worst-case input (every input at the format's end that maximizes |sum|) reaches the proven width: one bit
    less would overflow, the proven width never does."""
    rng = np.random.default_rng(11)
    ws, bs, acts = _random_network(rng, [10, 4], ["linear"], scale=2.0)
    m = fxp_mlp.quantize_mlp(ws, bs, acts)
    layer = m.layers[0]
    worst = 0
    for o in range(layer.n_out):
        sgn = np.where(layer.w_raw[o] >= 0, -1, 1)              # -2^(bits-1) is the largest magnitude
        x = np.where(sgn < 0, m.in_fmt.raw_min, m.in_fmt.raw_max)
        acc = int(layer.b_raw[o]) + sum(int(w) * int(v) for w, v in zip(layer.w_raw[o], x))
        worst = max(worst, abs(acc))
    assert worst <= (1 << (layer.acc_bits - 1)) - 1
    assert worst > (1 << (layer.acc_bits - 3)) - 1              # within a factor 2 of the bound (the bias and sign mix)


@pytest.mark.parametrize("rounding", ["half_even", "floor"])
def test_float_vs_fixed_error_is_within_the_bound(rounding):
    rng = np.random.default_rng(21 if rounding == "floor" else 22)
    for trial in range(5):
        sizes = [12, 16, 8, 3]
        ws, bs, acts = _random_network(rng, sizes, ["relu", "relu", "hardtanh"], scale=1.5)
        m = fxp_mlp.quantize_mlp(ws, bs, acts, rounding=rounding)
        lo, hi = -0.6, 0.6
        eb = fxp_mlp.error_bound(m, ws, bs, acts, lo, hi)
        assert eb.saturation_free
        x = rng.uniform(lo, hi, (3000, sizes[0]))
        x[:64] = np.where(rng.random((64, sizes[0])) < 0.5, lo, hi)          # box vertices
        y_fx = m.forward_real(x)
        y_fl = fxp_mlp.mlp_forward_float(ws, bs, acts, x)
        err = np.abs(y_fx - y_fl).max(axis=0)
        assert np.all(err <= eb.bound + 1e-12), (err, eb.bound)
        # exactly representable inputs (the env's observations): the bound without the input LSB still holds
        xq = m.quantize_input(x) * m.in_fmt.lsb
        eb0 = fxp_mlp.error_bound(m, ws, bs, acts, lo - m.in_fmt.lsb, hi + m.in_fmt.lsb, input_exact=True)
        err0 = np.abs(m.forward_real(xq) - fxp_mlp.mlp_forward_float(ws, bs, acts, xq)).max(axis=0)
        assert np.all(err0 <= eb0.bound + 1e-12)


def test_packed_form_memh_and_resources():
    rng = np.random.default_rng(5)
    ws, bs, acts = _random_network(rng, [5, 7, 2], ["relu", "hardtanh"])
    m = fxp_mlp.quantize_mlp(ws, bs, acts, w_bits=12)
    p = m.pack()
    assert fxp_mlp.fxp_mlp_check(p, 0) == 0
    assert (fxp_mlp.packed_n_in(p) == 5 and fxp_mlp.packed_n_out(p) == 2
            and fxp_mlp.packed_out_frac(p) == m.out_fmt.frac_bits)
    bad = p.copy()
    bad[fxp_mlp.H_MAGIC] += 1
    assert fxp_mlp.fxp_mlp_check(bad, 0) == 2
    # embedded at an offset inside a longer vector (as in a controller's parameter row)
    q = np.concatenate([np.full(9, -7, np.int64), p])
    x = m.quantize_input(rng.uniform(-1, 0.99, 5))
    buf = np.zeros(5 + 2 + 2 * m.max_width + 3, np.int64)
    buf[3:8] = x
    assert fxp_mlp.fxp_mlp_eval(q, 9, buf, 3, 8, 10) == 2
    assert list(buf[8:10]) == list(m.forward(x))
    mh = m.memh()
    assert len(mh["layer0_w"]) == 35 and all(len(s) == 3 for s in mh["layer0_w"])
    back = [int(s, 16) - (1 << 12) if int(s, 16) >= (1 << 11) else int(s, 16) for s in mh["layer0_w"]]
    assert back == [int(v) for v in m.layers[0].w_raw.ravel()]
    r = m.resources()
    assert r["macs"] == 5 * 7 + 7 * 2 and r["max_acc_bits"] == max(layer.acc_bits for layer in m.layers)
    with pytest.raises(ValueError):
        fxp_mlp.memh_lines([1 << 12], 12)


def test_packed_check_rejects_vectors_that_could_raise_in_the_kernel():
    """fxp_mlp_check (run by the compiled policy's init) refuses every packed field that would make fxp_mlp_eval
    raise inside the engine's tick loop: modes, activation, fraction bits (negative shifts), widths, a weight whose
    product with the largest input reaches 2^62 (fx_mul's limit)."""
    rng = np.random.default_rng(6)
    ws, bs, acts = _random_network(rng, [5, 4, 2], ["relu", "hardtanh"])
    p = fxp_mlp.quantize_mlp(ws, bs, acts).pack()
    assert fxp_mlp.fxp_mlp_check(p, 0) == 0
    r1 = fxp_mlp.H_SIZE + fxp_mlp.L_SIZE                                 # the second layer's record

    def bad(i, v):
        q = p.copy()
        q[i] = v
        return int(fxp_mlp.fxp_mlp_check(q, 0))
    assert bad(fxp_mlp.H_RND, 5) == 9 and bad(fxp_mlp.H_OVF, 3) == 9 and bad(fxp_mlp.H_OVF, -1) == 9
    assert bad(r1 + fxp_mlp.L_ACT, 3) == 10
    assert bad(r1 + fxp_mlp.L_OUT_FRAC, 61) == 11                         # out fraction bits > the accumulator's
    assert bad(fxp_mlp.H_SIZE + fxp_mlp.L_W_FRAC, -1) == 11
    assert bad(r1 + fxp_mlp.L_OUT_BITS, 63) == 7
    assert bad(fxp_mlp.H_IN_BITS, 1) == 12
    assert bad(int(p[fxp_mlp.H_SIZE + fxp_mlp.L_W_OFF]), 1 << 50) == 13  # 2^50 x 2^15 (Q0.15 input) > 2^62


def test_compiled_policy_releases_every_array():
    """mlp_policy_step runs inside the engine's runtime-free tick loop (docs/ENGINE.md, "Runtime-free hot loops"): it
    and fxp_mlp_eval must neither allocate nor make views, or the plant's arrays leak one reference per call."""
    import sys
    from mrm.sim import kernel
    env = MRMEnv("g26", num_envs=2, max_episode_steps=None, max_delta_heater_codes=16)
    mlp, *_ = _random_policy(env, 8)
    env.reset(seed=1)
    ctrl = env.compiled_policy(mlp)
    ctrl.reset(env.ctrl_spec)
    pl = env.plant
    pl.bind_compiled(ctrl)

    def refcounts():
        return [sys.getrefcount(a) for a in (*pl.P, *pl.S) if isinstance(a, np.ndarray)]
    pl.run(20, mode=kernel.MODE_COMPILED, step_fn=ctrl.step_fn)
    before = refcounts()
    pl.run(300, mode=kernel.MODE_COMPILED, step_fn=ctrl.step_fn)
    assert refcounts() == before


def test_quantizer_rejects_bad_networks():
    with pytest.raises(ValueError):
        fxp_mlp.quantize_mlp([np.ones((2, 3))], [np.ones(3)], ["relu"])
    with pytest.raises(ValueError):
        fxp_mlp.quantize_mlp([np.ones((2, 3))], [np.ones(2)], ["tanh"])
    with pytest.raises(ValueError):
        fxp_mlp.quantize_mlp([np.full((2, 3), np.nan)], [np.ones(2)], ["relu"])
    with pytest.raises(ValueError):
        fxp_mlp.quantize_mlp([np.ones((2, 3))], [np.ones(2)], ["relu"], in_fmt="UQ0.15")


# MLP forward passes in several rounding and overflow modes; the test runs it with numba on and off
_NOJIT_SCRIPT = r"""
import json, sys
import numpy as np
from mrm.control import fxp_mlp
rng = np.random.default_rng(31)
ws = [rng.normal(0, 0.6, (6, 9)), rng.normal(0, 0.6, (3, 6))]
bs = [rng.normal(0, 0.2, 6), rng.normal(0, 0.2, 3)]
out = {}
for rnd in ("half_even", "floor", "half_away"):
    m0 = fxp_mlp.quantize_mlp(ws, bs, ["relu", "hardtanh"], rounding=rnd)
    narrow = [l.acc_bits_needed - 2 for l in m0.layers]
    for ovf, acc in (("saturate", None), ("wrap", narrow), ("saturate", narrow)):
        m = fxp_mlp.quantize_mlp(ws, bs, ["relu", "hardtanh"], rounding=rnd, overflow=ovf, acc_bits=acc)
        x = rng.integers(m.in_fmt.raw_min, m.in_fmt.raw_max + 1, (50, 9))
        out[f"{rnd}-{ovf}-{acc}"] = m.forward(x).tolist()
print(json.dumps(out))
"""


@pytest.mark.skipif(JIT_OFF, reason="compares the JIT run with a NUMBA_DISABLE_JIT=1 subprocess")
def test_same_results_without_jit():
    run = lambda env: subprocess.run([sys.executable, "-c", _NOJIT_SCRIPT], capture_output=True, text=True,
                                     env=env, check=True).stdout
    base = dict(os.environ)
    a = json.loads(run({**base, "NUMBA_DISABLE_JIT": "0"}))
    b = json.loads(run({**base, "NUMBA_DISABLE_JIT": "1"}))
    assert a == b and len(a) == 9


# -------------------------------------------------------------------------------- the compiled MLP policy on the engine

def _random_policy(env, seed, scale=0.4):
    """A random 12-unit MLP for the env's observation and action sizes, quantized. Returns (mlp, ws, bs, acts)."""
    rng = np.random.default_rng(seed)
    D, A = env.obs_layout.size, env.action_codec.size
    ws = [rng.normal(0, scale / np.sqrt(D), (12, D)), rng.normal(0, 1.0, (A, 12))]
    bs = [rng.normal(0, 0.1, 12), rng.normal(0, 0.3, A)]
    acts = ["relu", "hardtanh"]
    return fxp_mlp.quantize_mlp(ws, bs, acts), ws, bs, acts


def test_compiled_policy_builds_the_env_observation_exactly():
    """Step the compiled policy's numba step on the env's own obs_i64 at every edge: its integer features are the
    env observation times 2^f_in, and its decoded commands are the env's commands for fxp_policy_action."""
    B = 2
    env = MRMEnv("g26", num_envs=B, max_episode_steps=None, max_delta_heater_codes=16)
    mlp, *_ = _random_policy(env, 1)
    ctrl = env.compiled_policy(mlp)
    obs, _ = env.reset(seed=3)
    ctrl.reset(env.ctrl_spec)
    p_i, p_f, s_i, s_f = ctrl.state
    act = env.plant.S.act[:, :env.ctrl_spec.layout.act_size].copy()
    N, K, M = env.ctrl_spec.n_rings, env.history_blocks, env.ctrl_spec.samples_per_step   # K: history blocks
    b0 = 1 + N * K * M
    D = env.obs_layout.size
    for k in range(12):
        o_i64 = env.plant.S.obs.copy()
        for b in range(B):
            mlp_policy_step(p_i[b], p_f[b], s_i[b], s_f[b], o_i64[b], act[b])
        feats = s_i[:, b0:b0 + D]
        assert np.array_equal(feats, np.rint(obs.astype(np.float64) * 2 ** mlp.in_fmt.frac_bits).astype(np.int64))
        a = fxp_policy_action(mlp, obs)
        obs, *_ = env.step(a)
        held = env.plant.S.act[:, :env.ctrl_spec.layout.act_size]
        assert np.array_equal(np.clip(act, 0, 65535), held)          # the kernel clips the env's command
        act = held.copy()


def _reset_fault_scenario(t_s: float = 150.5e-6):
    """g26's own scenario plus a scenario controller reset (fault.controller_reset, 3 us long) at t_s (s)."""
    from mrm import params
    from mrm.scenarios import generators, registry as scenario_registry
    comp = generators.pwl("fault.controller_reset", [0.0, t_s, t_s, t_s + 3e-6, t_s + 3e-6, 1.0], [0, 0, 1, 1, 0, 0],
                          absolute=True)
    return (scenario_registry.from_config(params.load("g26"))
            + generators.Scenario(name="controller_reset", components=(comp,)))


@pytest.mark.filterwarnings("ignore:warm_start_codes")
@pytest.mark.parametrize("case", ["hardtanh", "linear_beyond_unit", "controller_reset"])
def test_compiled_policy_on_the_engine_equals_the_env(case):
    """The same fixed-point policy: driven through the env (Python path, one kernel call per step) and run as a
    CompiledController inside the kernel by mrm.sim.runner.run: every metric and Truth field bit-identical.
    'linear_beyond_unit': a linear output layer whose outputs reach |y| > 1 (the env clips actions to [-1, 1]; the
    compiled policy must clip the same way). 'controller_reset': a scenario controller reset mid-run (the kernel
    restarts a compiled controller there; the env must restart its history and held command the same way)."""
    B, n = 2, 40
    seeds = {"device": 60, "uncertainty": 60, "noise": 61, "scenario": 62}
    sc = _reset_fault_scenario() if case == "controller_reset" else None
    env = MRMEnv("g26", num_envs=B, max_episode_steps=None, max_delta_heater_codes=16, scenario=sc)
    mlp, ws, bs, acts = _random_policy(env, 2)
    if case == "linear_beyond_unit":
        mlp = fxp_mlp.quantize_mlp(ws, [bs[0], bs[1] * 6.0], ["relu", "linear"])
        assert mlp.out_fmt.int_bits >= 2
    obs, _ = env.reset(options={"seeds": seeds})
    moved, a_max, resets = 0, 0.0, []
    for k in range(n):
        a = fxp_policy_action(mlp, obs)
        moved += int(np.count_nonzero(np.rint(np.clip(a, -1, 1) * 16)))
        a_max = max(a_max, float(np.abs(a).max()))
        obs, r, term, trunc, info = env.step(a)
        if info["controller_reset"].any():
            resets.append((k, info["controller_reset"].tolist()))
    assert moved > n                                                  # the policy acts
    if case == "linear_beyond_unit":
        assert a_max > 2.0                                            # the clip is exercised
    if case == "controller_reset":
        assert resets == [(15, [True, True])]                         # the edge at 160 us, every world
    else:
        assert not resets
    ctrl = env.compiled_policy(mlp)
    r = runner.run("g26", ctrl, sc, duration_s=n * env.step_s, batch=B, seeds=seeds, uncertainty="sample",
                   variation="sample", heater_code="locked_steady_state", settle=True,
                   metrics={"lock_reference": "oracle"}, histogram=False)
    assert r.manifest_inputs["controller_path"] == "compiled" and r.manifest_inputs["n_ticks"] == env.plant.tick
    m_env = env.plant.metrics()
    for k in ("rms_lock_error_pm", "max_abs_lock_error_pm", "mean_reloma", "heater_energy_j", "code_min_used",
              "code_max_used", "peak_slew_codes_per_s", "excursion_count"):
        assert np.array_equal(np.asarray(m_env[k]), np.asarray(r.metrics[k]), equal_nan=True), k
    t_env = env.plant.truth()
    for f, v in r.truth_final.fields.items():
        assert np.array_equal(t_env.fields[f], v, equal_nan=True), f
    assert np.array_equal(ctrl.act, env.plant.S.act[:, :env.ctrl_spec.layout.act_size])


def test_compiled_policy_checks_its_inputs():
    env = MRMEnv("g26", num_envs=1, max_episode_steps=None)
    mlp, ws, bs, acts = _random_policy(env, 4)
    with pytest.raises(ValueError):
        env.compiled_policy(fxp_mlp.quantize_mlp(ws, bs, acts, in_fmt="Q0.11"))       # 16-bit codes need >= 15 bits
    with pytest.raises(ValueError):
        env.compiled_policy(fxp_mlp.quantize_mlp(ws, bs, acts, overflow="checked"))
    with pytest.raises(ValueError):
        env.compiled_policy(fxp_mlp.quantize_mlp([w[:, :5] for w in ws[:1]] + ws[1:], bs, acts))
    d = MRMEnv("g26", num_envs=1, action_type="discrete")
    with pytest.raises(ValueError):
        d.compiled_policy(mlp)


# the env and the compiled policy on the engine for 6 steps; the test runs it with numba on and off
_NOJIT_ENV_SCRIPT = r"""
import json
import numpy as np
from mrm.control import fxp_mlp
from mrm.sim import runner
from mrm.sim.gym_env import MRMEnv, fxp_policy_action
env = MRMEnv("g26", num_envs=1, max_episode_steps=None, max_delta_heater_codes=16)
rng = np.random.default_rng(2)
D, A = env.obs_layout.size, env.action_codec.size
mlp = fxp_mlp.quantize_mlp([rng.normal(0, 0.04, (6, D)), rng.normal(0, 1.0, (A, 6))],
                           [rng.normal(0, 0.1, 6), rng.normal(0, 0.3, A)], ["relu", "hardtanh"])
seeds = {"device": 5, "uncertainty": 5, "noise": 6, "scenario": 7}
obs, _ = env.reset(options={"seeds": seeds})
out = {"obs": [obs.tolist()], "r": []}
for k in range(6):
    obs, r, *_ = env.step(fxp_policy_action(mlp, obs))
    out["obs"].append(obs.tolist())
    out["r"].append(r.tolist())
res = runner.run("g26", env.compiled_policy(mlp), None, duration_s=6 * env.step_s, batch=1, seeds=seeds,
                 heater_code="locked_steady_state", settle=True, metrics={"lock_reference": "oracle"}, histogram=False)
out["engine"] = {f: res.truth_final.fields[f].tolist() for f in ("detuning_m", "heater_code", "t_ring_k")}
out["env"] = {f: env.plant.truth().fields[f].tolist() for f in ("detuning_m", "heater_code", "t_ring_k")}
print(json.dumps(out))
"""


@pytest.mark.skipif(JIT_OFF, reason="compares the JIT run with a NUMBA_DISABLE_JIT=1 subprocess")
def test_env_and_compiled_policy_without_jit():
    """The env (kernel on the Python path) and the compiled MLP policy (mlp_policy_step inside the kernel) give the
    same observations, rewards and Truth with NUMBA_DISABLE_JIT=1 as compiled, and agree with each other."""
    run = lambda env: json.loads(subprocess.run([sys.executable, "-c", _NOJIT_ENV_SCRIPT], capture_output=True,
                                                text=True, env=env, check=True).stdout)
    base = dict(os.environ)
    a = run({**base, "NUMBA_DISABLE_JIT": "0"})
    b = run({**base, "NUMBA_DISABLE_JIT": "1"})
    assert a == b
    assert a["engine"] == a["env"]
