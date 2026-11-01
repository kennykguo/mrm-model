"""Checks the laser model of mrm.plant.laser: line tables, frequency error and drift, RIN and scenario inputs.

Read this first
---------------
The laser bank gives each line its optical frequency (Hz) and power (W) at every tick. These verification tests check
the line tables of G26 and L20 against the printed grids, the static frequency error per device, the frequency and
power drift (linear and random walk; a pure function of scenario time, so the step size and time compression do not
change it), the relative intensity noise (RIN) discretization, the scenario inputs (line enable, power scale,
frequency offset, tuning), the batch and thread invariance of the counter-based random streams, and identical results
with numba on and off.
"""
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from mrm.core import rng
from mrm.plant import laser
from mrm.units import C0

DT = 1e-6   # s, plant tick


def _laser_bank(B=1, lines=None, seeds=0, tcomp=1.0, **kw):
    """A LaserBank of B worlds (default: the G26 TX lines at 10 mW) with seed base ``seeds`` and time compression
    tcomp. Keyword arguments override the LaserHw.build inputs."""
    lines = laser.g26_line_table().select("tx") if lines is None else lines
    base = dict(power_w=10e-3)
    base.update(kw)
    hw = laser.LaserHw.build(B, DT, lines, **base)
    return laser.LaserBank(hw, rng.batch_seeds(seeds, B), time_compression=tcomp)


# ---------------------------------------------------------------------------------------------------------- line tables

def test_g26_line_table_grid():
    """G26: 2 x 8 lines, 400 GHz grid per direction, the two sets offset by 200 GHz, centred at 1310 nm (printed,
    LM26-P section 2). The comb spans 1301.47-1318.64 nm (read-off 1301.5-1318.9 nm, HC slide 17), and its 8
    highest-frequency lines are the PAT-4 example grid (printed to 0.01 nm)."""
    t = laser.g26_line_table()
    assert t.n_lines == 16 and np.all(np.diff(t.freq_hz) > 0)
    assert np.mean(t.freq_hz) == pytest.approx(C0 / 1310e-9, rel=1e-14)
    np.testing.assert_allclose(np.diff(t.freq_hz), 200e9, rtol=1e-12)
    for d in ("tx", "rx"):
        s = t.select(d)
        assert s.n_lines == 8 and s.n_lasers == 1
        np.testing.assert_allclose(np.diff(s.freq_hz), 400e9, rtol=1e-12)
    np.testing.assert_allclose(t.select("rx").freq_hz - t.select("tx").freq_hz, 200e9, rtol=1e-12)
    lam_nm = np.sort(t.wavelength_m * 1e9)
    assert abs(lam_nm[0] - 1301.5) < 0.3 and abs(lam_nm[-1] - 1318.9) < 0.3
    pat4 = [1301.47, 1302.60, 1303.73, 1304.87, 1306.01, 1307.14, 1308.28, 1309.43]   # printed, PAT-4
    np.testing.assert_allclose(lam_nm[:8], pat4, atol=0.006)
    # TX lines come from the local module, RX lines from the far end's
    assert set(t.source_laser[t.direction == laser.DIR_TX]) == {0} and t.laser_names == ("local", "remote")
    assert np.all(t.pol == laser.POL_TE)


def test_l20_and_grid_tables():
    t = laser.l20_line_table()
    lam = t.wavelength_m
    assert t.n_lines == 2
    assert lam[t.direction == laser.DIR_TX][0] == pytest.approx(1311e-9) and lam[t.direction == laser.DIR_RX][0] == \
        pytest.approx(1331e-9)
    sw = laser.l20_line_table(tx_m=1331e-9, rx_m=1311e-9)
    assert sw.wavelength_m[sw.direction == laser.DIR_TX][0] == pytest.approx(1331e-9)
    g = laser.grid_line_table(228.849e12, 400e9, 4)
    np.testing.assert_allclose(g.freq_hz, 228.849e12 + np.array([-1.5, -0.5, 0.5, 1.5]) * 400e9)


def test_line_table_records_roundtrip():
    t = laser.g26_line_table()
    back = laser.LineTable.from_records(t.records())
    np.testing.assert_allclose(back.freq_hz, t.freq_hz, rtol=1e-15)
    np.testing.assert_array_equal(back.direction, t.direction)
    assert back.laser_names == t.laser_names
    with pytest.raises(ValueError):
        laser.LineTable(np.array([1e14, 2e14]), np.array([0]), np.array([0, 0]), np.array([0, 0]))


# ----------------------------------------------------------------------------------------------- static frequency error

def test_static_error_bounds_and_module_correlation():
    lines = laser.g26_line_table()
    seeds = np.arange(500)
    e = laser.draw_static_error_hz(seeds, lines, 20e9)
    assert e.shape == (500, 16) and np.abs(e).max() <= 20e9 and np.abs(e).max() > 18e9
    assert e.std() == pytest.approx(20e9 / math.sqrt(3), rel=0.05)
    np.testing.assert_array_equal(e, laser.draw_static_error_hz(seeds, lines, 20e9))     # device-keyed
    c = laser.draw_static_error_hz(seeds, lines, 15e9, common_weight=1.0)
    tx = lines.direction == laser.DIR_TX
    assert np.all(c[:, tx] == c[:, tx][:, :1]) and np.abs(c).max() <= 15e9
    h = laser.draw_static_error_hz(seeds, lines, 15e9, common_weight=0.5, dist="normal3sigma")
    assert h.std() == pytest.approx(5e9, rel=0.06)
    r = np.corrcoef(h[:, 0], h[:, 2])[0, 1]                 # same module: correlation = common weight
    assert r == pytest.approx(0.5, abs=0.1)


# -------------------------------------------------------------------------------------------- frequency and power drift

def test_levy_brownian_motion_is_path_independent_with_unit_variance():
    key = np.uint64(rng.stream_key(3, int(rng.Stream.LASER_DRIFT)))
    js = [0, 1, 2, 3, 7, 1000, 12345, 2 ** 20 + 17]
    a = [laser.levy_bm(key, 2, 0, j, 30) for j in js]
    b = [laser.levy_bm(key, 2, 0, j, 30) for j in reversed(js)][::-1]
    assert a == b and a[0] == 0.0
    keys = [np.uint64(rng.stream_key(s, int(rng.Stream.LASER_DRIFT))) for s in range(3000)]
    w1 = np.array([laser.levy_bm(k, 0, 0, 100, 30) for k in keys])
    w2 = np.array([laser.levy_bm(k, 0, 0, 300, 30) for k in keys])
    assert w1.var() == pytest.approx(100, rel=0.08) and (w2 - w1).var() == pytest.approx(200, rel=0.08)
    assert abs(np.corrcoef(w1, w2 - w1)[0, 1]) < 0.06


def test_linear_drift_and_power_drift_are_exact():
    r_f, r_p = 4.0e4, 2.9e-7                       # 3.5 GHz and 2.5 % per 24 h (worst_linear of the printed bounds)
    bank = _laser_bank(2, f_rate_hz_per_s=r_f, p_rate_per_s=r_p)
    f0 = bank.hw.f0_hz.copy()
    for k in (0, 10 ** 6, 3 * 10 ** 9):
        f, p = bank.tick(k)
        t = (k + 0.5) * DT
        np.testing.assert_allclose(f - f0, r_f * t, rtol=1e-9, atol=0.1)   # float64 ulp at 230 THz: 0.03 Hz
        np.testing.assert_allclose(p, 10e-3 * (1 + r_p * t), rtol=1e-12)
    rate, rw = laser.drift_from_24h_bound(0.02e-9 * C0 / 1310e-9 ** 2, "worst_linear")
    assert rate * 86400 == pytest.approx(3.494e9, rel=1e-3) and rw == 0.0
    big = _laser_bank(1, p_rate_per_s=-1.0)
    assert big.tick(10 ** 7)[1].max() == 0.0       # power clamped at zero


def test_random_walk_statistics_and_module_correlation():
    """f - f0 = D sqrt(G) [sqrt(rho) W_m + sqrt(1 - rho) W_l]: rms D sqrt(t), and two lines of one module correlate
    with coefficient rho (the correlated-drift option)."""
    B = 600
    D = 1e6                                        # Hz/sqrt(s)
    bank = _laser_bank(B, seeds=100, f_rw_hz_per_rt_s=D, rho_f=0.6)
    k = 2 * 10 ** 6 - 1                            # t = 2 s - 0.5 us; grid 1 ms -> W at j = 1999.9995
    f, _ = bank.tick(k)
    d = f - bank.hw.f0_hz
    t = (k + 0.5) * DT
    assert d.std() == pytest.approx(D * math.sqrt(t), rel=0.08)
    r = np.corrcoef(d[:, 0], d[:, 3])[0, 1]
    assert r == pytest.approx(0.6, abs=0.08)
    rate, rw = laser.drift_from_24h_bound(3.5e9, "random_walk")
    assert rw * math.sqrt(86400) == pytest.approx(1.75e9)


def test_drift_is_independent_of_stepping_and_compression():
    """The drift at tick k is a pure function of scenario time: stepping every tick, jumping, or reaching
    it from another cache state gives the same bits; compression c only rescales time."""
    kw = dict(f_rw_hz_per_rt_s=5e5, p_rw_per_rt_s=1e-4, rho_f=0.3, rho_p=0.2)
    a = _laser_bank(2, seeds=7, **kw)
    seq = [a.tick(k)[0].copy() for k in range(2990, 3010)]
    b = _laser_bank(2, seeds=7, **kw)
    b.tick(10 ** 8)
    jump = [b.tick(k)[0].copy() for k in range(2990, 3010)]
    np.testing.assert_array_equal(np.array(seq), np.array(jump))
    c10 = _laser_bank(1, seeds=7, tcomp=10.0, **kw)
    c1 = _laser_bank(1, seeds=7, **kw)
    # compressed tick k at time (k + 0.5) dt 10 equals uncompressed tick 10 k + 4 at (10 k + 4.5) dt + 0.5 dt
    f10 = c10.tick(1234)[0][0] - c10.hw.f0_hz[0]
    f1a = c1.tick(12344)[0][0] - c1.hw.f0_hz[0]
    f1b = c1.tick(12345)[0][0] - c1.hw.f0_hz[0]
    np.testing.assert_allclose(f10, 0.5 * (f1a + f1b), rtol=1e-9, atol=0.1)   # 0.03 Hz ulp at 230 THz
    with pytest.raises(ValueError):
        _laser_bank(1, tcomp=0.5)


# --------------------------------------------------------------------------------------- relative intensity noise (RIN)

def test_rin_per_tick_and_window_average_are_dt_invariant():
    """RIN: the tick average of white relative intensity noise has variance RIN/(2 dt); its average over a fixed
    window (2 us) has variance RIN/(2 T) for any dt."""
    rin = 10 ** (-13.0)                               # 1e-13 /Hz (-130 dB/Hz; large so the test is quick)
    for dt in (0.25e-6, 1e-6):
        B = 2000
        hw = laser.LaserHw.build(B, dt, laser.grid_line_table(2.29e14, 400e9, 2), power_w=1e-3, rin_per_hz=rin)
        bank = laser.LaserBank(hw, rng.batch_seeds(5, B))
        nw = int(round(2e-6 / dt))
        ps = np.array([bank.tick(k)[1].copy() for k in range(nw)])
        rel = ps / 1e-3 - 1.0
        assert rel.std() == pytest.approx(math.sqrt(rin / (2 * dt)), rel=0.05)
        assert rel.mean(0).std() == pytest.approx(math.sqrt(rin / (2 * 2e-6)), rel=0.05)
        assert abs(np.corrcoef(rel[0, :, 0], rel[0, :, 1])[0, 1]) < 0.06      # independent lines


# ------------------------------------------------------------------------------------------------------ scenario inputs

def test_enable_scale_offset_and_tune():
    hw = laser.LaserHw.build(1, DT, laser.grid_line_table(2.29e14, 400e9, 3), power_w=[1e-3, 2e-3, 3e-3],
                             tune_lsb_hz=1e6)
    bank = laser.LaserBank(hw, [0])
    f, p = bank.tick(5, enable=[[True, False, True]], f_offset_hz=[[0.0, 0.0, -150e9]], p_scale=[[1.0, 1.0, 0.5]],
                     tune_codes=[[10, 0, -3]])
    np.testing.assert_allclose(p, [[1e-3, 0.0, 1.5e-3]])
    np.testing.assert_allclose(f - hw.f0_hz, [[1e7, 0.0, -150e9 - 3e6]])


def test_enable_is_a_power_factor_like_the_scenario_signal():
    """The scenario's laser.line_enable is a power factor in [0, 1] (a ramp is a turn-on, mrm.scenarios): a value of
    0.25 must give a quarter of the power, not full power; bool input keeps its on/off meaning; a negative power
    scale clamps to zero (optical power cannot go negative)."""
    hw = laser.LaserHw.build(1, DT, laser.grid_line_table(2.29e14, 400e9, 4), power_w=2e-3)
    bank = laser.LaserBank(hw, [0])
    _, p = bank.tick(0, enable=[[1.0, 0.25, 0.0, 1.7]], p_scale=[[1.0, 2.0, 1.0, 1.0]])
    np.testing.assert_allclose(p, [[2e-3, 1e-3, 0.0, 2e-3]])
    _, p = bank.tick(1, enable=[[True, False, True, True]], p_scale=[[1.0, 1.0, -0.5, 0.5]])
    np.testing.assert_allclose(p, [[2e-3, 0.0, 0.0, 1e-3]])


# ----------------------------------------------------------------------- batch invariance, parameter adapter, numba off

def test_batch_and_thread_invariance():
    kw = dict(f_rw_hz_per_rt_s=1e6, p_rw_per_rt_s=1e-4, rho_f=0.5, rin_per_hz=1e-14)

    def run(B, base, parallel=False):
        bank = _laser_bank(B, seeds=base, **kw)
        return np.array([np.concatenate(bank.tick(k, parallel=parallel), axis=1).copy()
                         for k in range(0, 50000, 997)])

    big = run(4, 20)
    for b in range(4):
        np.testing.assert_array_equal(big[:, b], run(1, 20 + b)[:, 0])
    import numba
    numba.set_num_threads(1)
    one = run(4, 20, parallel=True)
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    many = run(4, 20, parallel=True)
    np.testing.assert_array_equal(one, big)
    np.testing.assert_array_equal(many, big)


def test_from_plant_params_mini():
    from mrm.params import load, realize
    pp = realize(load(os.path.join(os.path.dirname(__file__), "..", "data", "mini.yaml")), batch=2, seeds=0)
    hw = laser.LaserHw.from_plant_params(pp)
    assert (hw.B, hw.L) == (2, 4)
    np.testing.assert_allclose(hw.f0_hz, pp.laser.line_freq_hz, rtol=1e-15)
    bank = laser.LaserBank(hw, [0, 1])
    f, p = bank.tick(0)
    np.testing.assert_allclose(f, pp.laser.line_freq_hz, rtol=1e-15)
    np.testing.assert_allclose(p, pp.laser.power_w, rtol=1e-15)


# a short drifting, noisy laser bank; the test below runs it with numba on and off
_NOJIT_SCRIPT = r"""
import numpy as np
from mrm.core import rng
from mrm.plant import laser
hw = laser.LaserHw.build(2, 1e-6, laser.g26_line_table(), power_w=5e-3, f_rw_hz_per_rt_s=1e6, p_rw_per_rt_s=1e-4,
                         rho_f=0.4, rin_per_hz=1e-14, depth=30)
bank = laser.LaserBank(hw, rng.batch_seeds(4, 2))
out = [np.concatenate(bank.tick(k), axis=1).copy() for k in range(0, 20000, 1999)]
np.save(OUT, np.array(out))
"""


def test_kernels_identical_without_jit(tmp_path):
    res = {}
    for mode in ("0", "1"):
        out = tmp_path / f"out{mode}.npy"
        env = dict(os.environ, NUMBA_DISABLE_JIT=mode)
        subprocess.run([sys.executable, "-c", f"OUT = {str(out)!r}\n" + _NOJIT_SCRIPT], env=env, check=True,
                       timeout=120)
        res[mode] = np.load(out)
    np.testing.assert_array_equal(res["0"], res["1"])
