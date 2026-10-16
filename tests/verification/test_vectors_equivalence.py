"""Equivalence of the fixed-point reference controller (07 P1-21: "float against fixed point within bounds;
fixed point against RTL, bit-exact").

- Fixed point against itself: fx_pi_update against an exact big-integer reference on random operands (every
  rounding mode, the accumulator saturation, the integrator clamp and the output clamp); the closed loop on the
  compiled path, on the Python path and on the Python path without JIT (test_vectors.py) is bit-identical.
- Float against fixed point: the float64 twin (same structure, unquantized gains, no accumulator width) on the SAME
  observations stays within ``mrm.control.fxp.pi_float_bound_codes`` = 1 + 2^-(F+1) (sum |e| + |e_k|) codes at every
  step (derived in its docstring: the gains are the only quantized operands; docs/research states no numeric
  float-vs-fixed rule). In closed loop the two runs see the same noise draws (counter-based RNG, 07 P1-13) and
  differ only through the controller: measured and bounded below.
- The failure mode 07 P1-21 cites from textbook Ch17 ("the integrator's fraction bits are the usual failure"): an
  integrator rounded to whole codes ignores errors with |ki e| < 1/2 code and leaves a steady-state error; FxpPi's
  integrator keeps F = 20 fraction bits.
"""
from __future__ import annotations

import math
import random
import warnings
from fractions import Fraction

import numba as nb
import numpy as np
import pytest

from mrm import params
from mrm.control import api, fxp
from mrm.control.fxp import FxpPi, Overflow, QFormat, Rounding
from mrm.scenarios import library as lib
from mrm.sim import runner, vectors
from mrm.sim.metrics import RecordSpec

RUN = dict(uncertainty="nominal", variation="off", seeds=0, heater_code="quasi_static", settle=True)
DURATION_S = 4e-3
REC = [RecordSpec("detuning_m", 1, "last"), RecordSpec("heater_code", 10, "last")]


def scenario():
    return lib.aggressor_direct(waveform="square", power_w=0.05, period_s=2e-3, duty=0.5, phase_s=1e-3)


# ---- exact reference of fx_pi_update (Python integers, Fraction rounding) -----------------------------------------

def _round(q: Fraction, rnd: int) -> int:
    f = math.floor(q)
    d = q - f
    if rnd == Rounding.FLOOR:
        return f
    if rnd == Rounding.HALF_UP:
        return f + (d >= Fraction(1, 2))
    if rnd == Rounding.HALF_EVEN:
        return f + (d > Fraction(1, 2) or (d == Fraction(1, 2) and f % 2 == 1))
    if rnd == Rounding.TOWARD_ZERO:
        return math.trunc(q)
    if rnd == Rounding.HALF_AWAY:
        r = math.floor(abs(q) + Fraction(1, 2))
        return r if q >= 0 else -r
    raise ValueError(rnd)


def ref_pi_update(e, integ, kp, ki, u0, frac, acc_bits, lo, hi, rnd):
    a = integ + e * ki
    a = max(-(1 << (acc_bits - 1)), min((1 << (acc_bits - 1)) - 1, a))
    a = max((lo - u0) << frac, min((hi - u0) << frac, a))
    v = (u0 << frac) + e * kp + a
    u = _round(Fraction(v, 1 << frac), rnd)
    return max(lo, min(hi, u)), a


def test_fx_pi_update_matches_exact_reference():
    rng = random.Random(20261010)
    n = 0
    for _ in range(20000):
        frac = rng.choice((0, 1, 8, 16, 20, 24))
        gbits = rng.randint(frac + 2, min(frac + 12, 40))
        acc_bits = rng.randint(max(frac + 2, 4), min(frac + 24, 61))
        kp = rng.randint(-(1 << (gbits - 1)), (1 << (gbits - 1)) - 1)
        ki = rng.randint(-(1 << (gbits - 1)), (1 << (gbits - 1)) - 1)
        lo = rng.randint(0, 30000)
        hi = rng.randint(lo, 65535)
        u0 = rng.randint(lo, hi)
        e = rng.randint(-65535, 65535) if rng.random() < 0.7 else rng.randint(-50, 50)
        integ = rng.randint(-(1 << (acc_bits - 1)), (1 << (acc_bits - 1)) - 1)
        rnd = rng.randint(0, 4)
        got = fxp.fx_pi_update(e, integ, kp, ki, u0, frac, acc_bits, lo, hi, rnd)
        assert (int(got[0]), int(got[1])) == ref_pi_update(e, integ, kp, ki, u0, frac, acc_bits, lo, hi, rnd), \
            (e, integ, kp, ki, u0, frac, acc_bits, lo, hi, rnd)
        n += 1
    assert n == 20000


def test_fxp_pi_quantizes_and_checks_its_formats():
    c = FxpPi()
    assert c.frac_bits == 20 and str(c.gain_format) == "Q3.20" and c.acc_format.bits == 40
    assert c.kp_raw == round(0.0193 * 2**20) and c.ki_raw == round(0.0148 * 2**20)
    assert abs(c.kp_q - 0.0193) <= 2**-21 and abs(c.ki_q - 0.0148) <= 2**-21
    with pytest.raises(OverflowError):
        FxpPi(kp=9.0)                                   # outside Q3.20 (|k| < 8): a design error, never saturated
    with pytest.raises(ValueError, match="fraction bits"):
        FxpPi(gain_format="Q3.20", acc_format="Q19.16")
    with pytest.raises(ValueError, match="int64"):
        FxpPi(gain_format="Q30.20", acc_format="Q19.20")
    d = FxpPi(setpoint_codes=[40000] * 8).describe()
    assert d["setpoint_codes"] == (40000,) * 8 and d["rounding"] == "half_up"


def test_fxp_pi_refuses_a_monitor_wider_than_its_error_register():
    """The int64 exactness of fx_pi_update rests on a 17-bit error (a monitor spanning < 2^16 codes): an 18-bit ADC
    is refused at reset instead of silently overflowing the sized products."""
    from mrm.sim import bridge
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c18 = params.load("g26", overrides=["ring.monitor.adc_bits=18"])
    spec = bridge.spec_for(c18, scenario(), 1)
    assert spec.channels["tx.monitor.latest"].max == 2**18 - 1
    with pytest.raises(ValueError, match="17 bits"):
        FxpPi().reset(spec)


# ---- float twin ------------------------------------------------------------------------------------------------------

@nb.njit(cache=True)
def float_pi_init(spec_i64, spec_f64):
    p, _pf, s, _sf = fxp.fxp_pi_init(spec_i64, spec_f64)
    n = p[0]
    pf = np.zeros(2)
    pf[0] = api.user_f64(spec_f64, 0)
    pf[1] = api.user_f64(spec_f64, 1)
    return p, pf, s, np.zeros(n)


@nb.njit(cache=True)
def float_pi_step(p, pf, s, sf, obs, act):
    """FxpPi's structure in float64: a = clamp(I + ki e, lo - u0, hi - u0); u = clamp(floor(u0 + kp e + a + 1/2))."""
    n = p[0]
    if s[0] == 0:
        for i in range(n):
            u0 = obs[p[4] + i] if p[4] >= 0 else act[p[5] + i]
            s[1 + 2 * n + i] = u0
            sf[i] = 0.0
        s[0] = 1
    for i in range(n):
        if obs[p[2] + i] == 0:
            continue
        if p[3] >= 0 and obs[p[3] + i] == 0:
            continue
        y = obs[p[1] + i]
        if s[1 + i] == 0:
            s[1 + n + i] = y
            s[1 + i] = 1
            continue
        e = float(y - s[1 + n + i])
        u0 = float(s[1 + 2 * n + i])
        lo, hi = float(p[10]), float(p[11])
        a = min(max(sf[i] + pf[1] * e, lo - u0), hi - u0)
        sf[i] = a
        u = math.floor(u0 + pf[0] * e + a + 0.5)
        act[p[5] + i] = np.int64(min(max(u, lo), hi))


def float_pi(kp=0.0193, ki=0.0148):
    c = FxpPi()
    ctrl = api.CompiledController("float_pi", float_pi_init, float_pi_step, params_i64=c.params_i64,
                                  params_f64=[kp, ki])
    return ctrl


@pytest.fixture(scope="module")
def cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26")


@pytest.fixture(scope="module")
def recorded(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vectors.run_recorded(cfg, FxpPi(), scenario(), duration_s=DURATION_S, record=REC, **RUN)


def test_float_vs_fixed_on_the_same_observations_within_the_derived_bound(recorded):
    """Replay the fixed-point run's observations through the float twin: |u_fixed - u_float| <= 1 + 2^-(F+1)
    (sum |e| + |e_k|) at every update (pi_float_bound_codes). Measured on this run: at most 1 code."""
    v = recorded.vectors
    c = FxpPi()
    res = vectors.replay_abi(float_pi_init, float_pi_step, v,
                             spec_f64=np.concatenate([v.spec_f64[:api.SPEC_F_HEADER], [c.kp, c.ki]]))
    n = 8
    u_fx = v.channel("tx.heater", "act")[:, 0, :].astype(np.int64)
    u_fl = res.act[:, 0, :n]
    # the error stream exactly as the controllers form it: e = y - y_set on updates (new and valid samples)
    y = v.channel("tx.monitor.latest")[:, 0, :]
    new = v.channel("tx.monitor.latest", "new")[:, 0, :].astype(bool)
    ok = v.channel("monitor_valid")[:, 0, :].astype(bool)
    upd = new & ok
    first = upd.argmax(axis=0)                       # the capture sample (no update)
    e = np.zeros_like(y, dtype=np.float64)
    for i in range(n):
        ks = np.flatnonzero(upd[:, i])
        e[ks[1:], i] = y[ks[1:], i] - y[first[i], i]
    abs_sum = np.cumsum(np.abs(e), axis=0)
    bound = np.vectorize(fxp.pi_float_bound_codes)(abs_sum, np.abs(e), c.frac_bits)
    # a held step keeps the bound of the ring's last update (forward fill along k)
    last = np.where(upd, np.arange(len(y))[:, None], 0)
    last = np.maximum.accumulate(last, axis=0)
    bound = np.take_along_axis(bound, last, axis=0)
    diff = np.abs(u_fx - u_fl)
    assert (diff <= bound).all(), (diff.max(), bound.min())
    assert diff.max() <= 1, diff.max()               # measured: one rounding boundary at most
    # the worst-case gain-quantization part grows with sum |e| (ring 0 reaches ~4e5 ADC codes in 400 steps):
    # derived bound 1.28 codes at the end of this run (1.19 before the 2026-10-16 crosstalk-distance fix, whose
    # faster aggressor -> ring 0 pole, 0.62 instead of 1.03 ms, gives larger errors; the recording was regenerated,
    # docs/CHANGELOG.md); the measured difference stays at one rounding boundary
    assert 1.0 <= bound.max() < 1.32, bound.max()


def test_float_vs_fixed_closed_loop(cfg, recorded):
    """Closed loop on the same plant and noise draws: the float and fixed-point loops part only when an output
    rounds to another code. Measured (2026-10-10, this scenario): heater codes within 1, detuning within 0.045 pm.
    Bound asserted: 2 codes and 0.1 pm (one heater code moves the resonance ~0.053 pm at the g26 lock; the closed
    loop is contractive, so a difference stays at the level of a code instead of growing)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf = runner.run(cfg, float_pi(), scenario(), duration_s=DURATION_S, record=REC, **RUN)
    hx = np.asarray(recorded.trace["heater_code@10:last"]["value"])
    hf = np.asarray(rf.trace["heater_code@10:last"]["value"])
    dx = np.asarray(recorded.trace["detuning_m@1:last"]["value"])
    df = np.asarray(rf.trace["detuning_m@1:last"]["value"])
    dcode = np.abs(hx - hf).max()
    dpm = np.abs(dx - df).max() * 1e12
    print(f"\nfloat vs fixed closed loop: max |d heater| {dcode:.0f} codes, max |d detuning| {dpm:.4f} pm")
    assert dcode <= 2 and dpm <= 0.1, (dcode, dpm)
    assert np.allclose(recorded.metrics["rms_detuning_pm"], rf.metrics["rms_detuning_pm"], atol=0.05)


# ---- the Ch17 failure mode: integrator fraction bits ---------------------------------------------------------------

def _integrate(e_seq, ki, int_frac, gain_frac=20):
    """Integrator kept with ``int_frac`` fraction bits: each ki*e (exact at gain_frac bits) is rounded to int_frac
    bits (half-even) before it is accumulated, as an RTL integrator register of that width would be."""
    kq = QFormat(3, gain_frac)
    ki_raw = kq.quantize(ki)
    acc = 0
    out = []
    for e in e_seq:
        inc = fxp.fx_cast(np.int64(e) * ki_raw, gain_frac, 40, int_frac, True, int(Rounding.HALF_EVEN),
                          int(Overflow.SATURATE))
        acc += int(inc)
        out.append(acc * 2.0 ** -int_frac)
    return np.array(out)


def test_integrator_without_fraction_bits_stalls():
    """|ki e| = 0.0148 x 20 = 0.30 code per step: rounded to whole codes it is lost every step (dead zone
    |e| < 1/(2 ki) = 34 ADC codes = 3.2 heater-code equivalents at the g26 slope); with 20 fraction bits the
    integrator follows the float sum within 2^-21 per step."""
    e = np.full(200, 20)
    i0 = _integrate(e, 0.0148, 0)
    i20 = _integrate(e, 0.0148, 20)
    exact = 0.0148 * np.cumsum(e)
    assert np.all(i0 == 0.0)                         # stalls: a steady-state error of 20 ADC codes survives
    assert np.max(np.abs(i20 - exact)) <= 200 * 20 * 2.0 ** -21 + 1e-12
    assert i20[-1] == pytest.approx(59.2, abs=1e-3)
    # FxpPi itself: the same constant error accumulates in its Q19.20 integrator (u0 = 30000)
    c = FxpPi()
    a = 0
    for _ in range(200):
        u, a = fxp.fx_pi_update(20, a, c.kp_raw, c.ki_raw, 30000, 20, 40, 0, 65535, int(Rounding.HALF_UP))
    assert int(u) == 30000 + round(c.kp_q * 20 + c.ki_q * 20 * 200)
