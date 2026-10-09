"""Engine-level physics tests that need the whole plant (owner: engine; 07 P0-1 (a)-(d), P1-2, P1-7, P1-8, P1-9,
P1-13, P2-1, and the P0-10 quasi-static survey).

Most tests use a one-ring, one-line g26 bank (``ONE``) with a power-linear heater driver (no TCR feedback in the
heater power, so the static self-heating gain is exactly G = s(T) Z_opt,self), the 1-10 ms thermal tails (the ring
tail and the 3 ms package tail of the uncertainty set) and the driver heat (a ms site path) switched off (so "steady"
means ~200 us), the nominal world, no bias-DAC noise and no laser RIN. These are test settings, not presets.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from mrm import params
from mrm.control.api import ControllerSpec
from mrm.plant import optics as op
from mrm.plant import phase as ph
from mrm.plant import thermal as th
from mrm.plant.plant import Plant
from mrm.scenarios import generators as sg
from mrm.scenarios import registry as scen_registry
from mrm.sim import kernel as K
from mrm.sim.runner import quasi_static, run
from mrm.units import C0

ONE = ["array.n_rings=1", "channel_plan.n_lines=1", "array.bus_order=[0]", "channel_plan.ring_to_line=[0]"]
TWO = ["array.n_rings=2", "channel_plan.n_lines=2", "array.bus_order=[0,1]", "channel_plan.ring_to_line=[0,1]"]
CLEAN = ["ring.heater.driver_mode=power", "ring.thermal.tail_share_frac=0.0", "ring.bias_dac.noise_rms_codes=0.0",
         "uncertainty.thermal.package_tail_share_frac.low=0.0",
         "uncertainty.thermal.package_tail_share_frac.nominal=0.0",
         "laser.rin_db_per_hz=null", "driver.static_power_mw=0.0", "driver.energy_per_transition_fj=0.0"]
#: the monolithic driver's heat reaches the ring through a site path with ms poles (07 P1-8, interim site model), so
#: CLEAN switches it off and only the traffic test (which waits 45 ms) keeps it
DRIVER_ON = [o for o in CLEAN if not o.startswith("driver.")]


def _cfg(*ov):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26", overrides=list(ov))


def _plant(cfg, scenario=None, *, record=(), n_rec=0, seeds=1, protocol="branch_low", heater_code=None,
           oracle=False, B=1, uncertainty="nominal", variation="off", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = params.realize(cfg, B, seeds=seeds, uncertainty=uncertainty, variation=variation)
    sc = scen_registry.from_config(cfg) if scenario is None else scenario
    spec = ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)
    pl = Plant(pp, sc, spec, record=list(record), record_ticks=n_rec, oracle=oracle, histogram=False, **kw)
    pl.reset(protocol=protocol, heater_code=heater_code)
    return pl


def _with_defaults(cfg, *components):
    """The config's scenario (ambient, bias) plus extra components."""
    base = scen_registry.from_config(cfg)
    return base.replace(components=tuple(base.components) + tuple(components))


def _rows(pl, b=0):
    return op.selfheat_views(pl.S.sh_args_f[b], pl.P.sh_args_i[b])


def _line_inputs(pl, n=0, b=0):
    """Optics inputs of ring n at its line (level phases, a, eta, K^2, t_rt, dphi/dlambda) and the static self-heating
    gain G = s(T) Z_opt,self (m/W) at the present state; rp (the phase-model row) and xm (the cold-curve coordinate
    of the order-m resonance nearest the line) map a detuning to the phase at the line exactly (:func:`_phase_at`)."""
    v = _rows(pl, b)
    l = int(pl.S.ri[b, K.RI_DECL, n])
    tref, s0, beta = v[17][n], v[18][n], v[19][n]
    dT = pl.S.rf[b, K.RF_T_RING, n] - tref
    s_now = s0 * (1.0 + beta * dT)
    lam = pl.S.lf[b, K.LF_LAM, l]
    rp = pl.phase.ring[b, n]
    shift = 0.5 * (pl.S.jx[b, 0, n, 0] + pl.S.jx[b, 0, n, 1]) + s0 * (dT + 0.5 * beta * dT * dT)
    m = int(pl.S.ri[b, K.RI_ORDER, n])
    xm = K.cold_coordinate(rp, lam - shift, float(m))
    return dict(phi0=v[4][n, l], phi1=v[5][n, l], a0=v[8][n, l], a1=v[9][n, l], e0=v[10][n, l], e1=v[11][n, l],
                k2t=v[13][n, l], k2d=v[14][n, l], trt=v[15][n, l], dpl=v[16][n, l], pin=v[22][op.RL_PIN, n, l],
                rho=pl.S.rf[b, K.RF_RHO, n], s=s_now, G=float(pl.P.z_opt_self[b, n]) * s_now, line=l,
                lam=lam, fwhm=lam ** 2 * pl.P.fwhm_hz[b, n] / C0, rp=rp, xm=xm, m=m)


def _phase_at(li, d):
    """Mean level phase at the line (rad) and d phase / d d (rad/m) for the mean-resonance detuning d (m), exactly on
    the phase model's cold curve (mrm.plant.phase equation 2): x = x_m - d, phi = 2 pi n(x) L / x,
    dphi/dd = 2 pi n_g(x) L / x^2."""
    rp = li["rp"]
    lam_r, ne, ng, dng, ln = (rp[ph.RP_LAM_RES], rp[ph.RP_N_EFF], rp[ph.RP_N_G], rp[ph.RP_DNG], rp[ph.RP_LEN])
    x = li["xm"] - np.asarray(d, float)
    dd = x - lam_r
    n_x = ne + (ne - ng) * dd / lam_r - 0.5 * dng * dd * dd / lam_r
    ng_x = ng + dng * dd + 0.5 * dng * dd * dd / lam_r
    return 2 * math.pi * n_x * ln / x, 2 * math.pi * ng_x * ln / (x * x)


def _pabs_curve(li, d):
    """P_abs(d) and dP_abs/dd (W, W/m) of the mean-resonance detuning d (m) at fixed input, both NRZ levels (the
    exact detuning -> phase map of the plant's phase model, :func:`_phase_at`)."""
    dphi = li["phi1"] - li["phi0"]
    base, dpd = _phase_at(li, d)
    r0 = op.ring_exact(base - 0.5 * dphi, li["k2t"], li["k2d"], li["a0"], li["trt"])
    r1 = op.ring_exact(base + 0.5 * dphi, li["k2t"], li["k2d"], li["a1"], li["trt"])
    q1 = li["rho"]
    p = li["pin"] * ((1 - q1) * li["e0"] * r0.p_loss + q1 * li["e1"] * r1.p_loss)
    dp = li["pin"] * ((1 - q1) * li["e0"] * r0.dloss_dphi + q1 * li["e1"] * r1.dloss_dphi) * dpd
    return p, dp, r0, r1


def _lock_code(cfg, bias=1.05):
    qs = quasi_static(cfg, ambient_c=[25.0], bias_v=[bias], batch=1, uncertainty="nominal", variation="off")
    return qs, int(qs.heater_code[0, 0, 0, 0])


def _settle_on_upper_branch(pl, code, n_up=300, n_ramp=300, n_hold=300, over=1.3):
    """Reach the self-locked (upper, red-side) branch at ``code``: start above the up-fold, ramp down, hold."""
    hi = int(min(pl.P.he_hi, code * over))
    pl.advance({"tx.heater": np.full((pl.B, pl.N), hi, np.int64)}, n_up)
    for c in np.linspace(hi, code, 30).astype(np.int64):
        pl.advance({"tx.heater": np.full((pl.B, pl.N), c, np.int64)}, n_ramp // 30)
    pl.advance({"tx.heater": np.full((pl.B, pl.N), code, np.int64)}, n_hold)


# ================================================================================================ 07 P0-1
@pytest.fixture(scope="module")
def lock1():
    cfg = _cfg(*ONE, *CLEAN)
    qs, code = _lock_code(cfg)
    return cfg, qs, code


def test_quasi_static_lock_holds_open_loop(lock1):
    """07 P0-10 quasi_static: the heater code it reports, applied open loop to the dynamic plant from the upper
    branch, holds the ring at the reported oracle detuning (the static map and the dynamic plant agree), and the
    reported relOMA is the plant's."""
    cfg, qs, code = lock1
    assert qs.code_ok[0, 0, 0, 0]
    pl = _plant(cfg)
    _settle_on_upper_branch(pl, code)
    tr = pl.truth()
    li = _line_inputs(pl)
    d_target = qs.detuning_target_m[0, 0, 0, 0]
    # the code is the smallest one reaching the target power: one code step is ~0.5 pm
    assert abs(tr.detuning_m[0, 0] - d_target) < 0.02 * li["fwhm"]
    assert 10 * math.log10(tr.reloma[0, 0]) == pytest.approx(qs.reloma_db[0, 0, 0, 0], abs=0.05)


def _sweep(pl, c0, c1, ticks_per_step, step=1):
    """Heater staircase from c0 to c1 (both included), ``ticks_per_step`` ticks per code step; the detuning, absorbed
    power and the optics inputs (``_line_inputs``) at the end of every step."""
    codes = np.arange(c0, c1 + 1, step) if c1 >= c0 else np.arange(c0, c1 - 1, -step)
    det, pab, lis = [], [], []
    for c in codes:
        pl.advance({"tx.heater": np.full((1, 1), c, np.int64)}, ticks_per_step)
        det.append(pl.S.rf[0, K.RF_DET, 0])
        pab.append(pl.S.rf[0, K.RF_P_ABS, 0])
        lis.append(_line_inputs(pl))
    return codes, np.array(det), np.array(pab), lis


def _onset(det, hw):
    """Index of the last quasi-static sample before the first jump (> 0.1 HWHM in one step)."""
    dd = np.abs(np.diff(det))
    big = np.flatnonzero(dd > 0.1 * hw)
    assert big.size, "no jump"
    return int(big[0])


def _folds(li, G, hw):
    """Fold detunings (G dP_abs/dd = 1) of the exact ring and their cold-detuning coordinate y_o = d - G P_abs."""
    d = np.linspace(-6.0, 6.0, 24001) * hw
    p, dp, _, _ = _pabs_curve(li, d)
    f = G * dp - 1.0
    idx = np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:]))
    out = []
    for i in idx:
        x = d[i] - f[i] * (d[i + 1] - d[i]) / (f[i + 1] - f[i])
        pp, _, _, _ = _pabs_curve(li, np.array([x]))
        out.append((x, x - G * pp[0]))
    return sorted(out, key=lambda t: t[0])


def test_selfheat_hysteresis_and_fold_edges(lock1):
    """07 P0-1 (a): at +4 dBm slow heater sweeps up and down show hysteresis whose edges sit where
    G dP_abs/dlambda_r = 1 (the static folds of the exact ring at the state just before each jump, compared in the
    cold-detuning coordinate y_o = d - G P_abs, within 2 % of the HWHM); at -10 dBm, swept across the line, there
    is none.

    With y_o(d) = d - G P_abs(d) the lower branch ends at the local maximum of y_o (fold a, further blue) and the
    upper (self-locked) branch at the local minimum (fold b): the up-sweep jumps at y_o(a) and the down-sweep at
    y_o(b) < y_o(a). For this ring (G ~ 0.72 pm/uW, HWHM ~ 171 pm, 0.25 pm of y_o per code) fold a sits ~1600 codes
    above the lock code and fold b ~1700 below it, hence the +-2500-code sweep (2 codes per 40 us step)."""
    cfg, qs, code = lock1
    pl = _plant(cfg)
    lo, hi = code - 2500, code + 2500
    pl.advance({"tx.heater": np.full((1, 1), lo, np.int64)}, 300)
    hw = 0.5 * _line_inputs(pl)["fwhm"]
    cu, du, pu, lu = _sweep(pl, lo, hi, 40, 2)
    cd, dd, pd, ld = _sweep(pl, hi, lo, 40, 2)
    iu, idn = _onset(du, hw), _onset(dd, hw)
    assert cu[iu] > cd[idn]                                      # hysteresis: jumps at different heater powers
    # up-sweep edge: fold a of the ring at the last sample before the jump
    G = lu[iu]["G"]
    folds = _folds(lu[iu], G, hw)
    assert len(folds) == 2
    yo_up = du[iu] - G * pu[iu]
    assert abs(yo_up - folds[0][1]) < 0.02 * hw, (yo_up / hw, folds[0][1] / hw)
    # down-sweep edge: fold b
    G = ld[idn]["G"]
    folds = _folds(ld[idn], G, hw)
    assert len(folds) == 2
    yo_dn = dd[idn] - G * pd[idn]
    assert abs(yo_dn - folds[1][1]) < 0.02 * hw, (yo_dn / hw, folds[1][1] / hw)
    # -10 dBm at the ring (G P_abs,max ~ 0.4 HWHM, below the bistability onset): swept across the line (y_o from
    # about -1.6 to +1.6 HWHM), up and down sweeps coincide and nothing jumps
    low = _with_defaults(cfg, sg.const("laser.power_delta_db", -14.0))
    pl2 = _plant(cfg, low)
    a, b = code + 3000, code + 5200
    pl2.advance({"tx.heater": np.full((1, 1), a, np.int64)}, 300)
    _, d2u, _, _ = _sweep(pl2, a, b, 30, 2)
    _, d2d, _, _ = _sweep(pl2, b, a, 30, 2)
    assert d2u.min() < -1.0 * hw and d2u.max() > 1.0 * hw       # the sweep crosses the line
    assert np.max(np.abs(np.diff(d2u))) < 0.05 * hw
    np.testing.assert_allclose(d2u, d2d[::-1], atol=0.01 * hw)


def test_selfheat_dt_convergence():
    """07 P0-1 (b): the implicit solve at dt = 1 us agrees with a 10 ns reference to better than 1 pm RMS on the
    stable branch (a slow heater ramp after locking), and at +9 dBm the steady detuning changes by less than 1 % of
    the FWHM between dt and dt/8."""
    base = [*ONE, *CLEAN]
    cfg1 = _cfg(*base)
    qs, code = _lock_code(cfg1)
    traces = {}
    for dt in (1e-6, 1e-8):
        cfg = _cfg(*base, f"sim.dt_plant_s={dt}")
        per_us = int(round(1e-6 / dt))
        pl = _plant(cfg)
        _settle_on_upper_branch(pl, code, n_up=200 * per_us, n_ramp=300 * per_us, n_hold=200 * per_us)
        out = []
        for c in np.linspace(code, int(code * 1.04), 200).astype(np.int64):
            pl.advance({"tx.heater": np.full((1, 1), c, np.int64)}, per_us)
            out.append(pl.S.rf[0, K.RF_DET, 0])
        traces[dt] = np.array(out)
    rms = np.sqrt(np.mean((traces[1e-6] - traces[1e-8]) ** 2))
    assert rms < 1e-12, rms
    # +9 dBm: steady detuning at dt and dt/8
    finals = []
    for dt in (1e-6, 1.25e-7):
        cfg = _cfg(*base, f"sim.dt_plant_s={dt}")
        per_us = int(round(1e-6 / dt))
        sc = _with_defaults(cfg, sg.const("laser.power_delta_db", 5.0))
        pl = _plant(cfg, sc)
        _settle_on_upper_branch(pl, int(code * 1.1), n_up=200 * per_us, n_ramp=300 * per_us, n_hold=400 * per_us)
        finals.append(pl.S.rf[0, K.RF_DET, 0])
        fwhm = _line_inputs(pl)["fwhm"]
    assert abs(finals[0] - finals[1]) < 0.01 * fwhm


def _self_lags(pl, n=0):
    """The ring's own optical-heating lags (tau_m s, r_m K/W) of the thermal system."""
    sysm = pl.sys
    ro = sysm.ring_obs[n]
    m = np.flatnonzero((sysm.lag_kind == th.LAG_OPT_SELF) & (sysm.lag_obs == ro))
    return sysm.taus_s[0, m], sysm.res_k_per_w[0, m]


def test_selfheat_no_oscillation_on_stable_branch(lock1):
    """07 P0-1 (c), stable side: at +9 dBm and dt = 1 us, where an explicit coupling (P_abs of the previous step
    into the fastest self-heating lag) would oscillate (its multiplier a - (1 - a) r |dP_abs/dT| is below -1), a
    heater step up and one down on the self-locked branch relax without changing direction from tick to tick."""
    cfg, qs, code = lock1
    sc = _with_defaults(cfg, sg.const("laser.power_delta_db", 5.0))
    pl = _plant(cfg, sc, record=[("detuning_m", 1, "last")], n_rec=3000)
    c9 = int(code * 1.1)
    _settle_on_upper_branch(pl, c9, n_up=300, n_ramp=300, n_hold=400)
    # the explicit multiplier at the operating point (fastest own lag)
    _, kslope = K.eng_pabs(pl.S.rf[0, K.RF_T_RING, 0], 0, pl.S.sh_args_f[0], pl.P.sh_args_i[0])
    tau, r = _self_lags(pl)
    f = int(np.argmin(tau))
    a = math.exp(-1e-6 / tau[f])
    assert kslope < 0                                                  # stable (red) side
    assert a - (1 - a) * r[f] * abs(kslope) < -1.0
    for dc in (+300, -600):
        pl.advance({"tx.heater": np.full((1, 1), c9 + dc, np.int64)}, 200)
    d = pl.traces()["detuning_m@1:last"]["value"][:, 0, 0]
    for start in (1000, 1200):
        seg = np.diff(d[start:start + 120])
        assert np.max(np.abs(seg)) > 1e-12                               # the step moved the resonance
        seg = seg[np.abs(seg) > 1e-17]                                   # below: solver tolerance (1e-7 K)
        assert np.all(np.sign(seg) == np.sign(seg[0])), "detuning changed direction tick to tick"


def _static_detuning(li, G, y_o, d_start, scale=1.0):
    """Self-heated static detuning d solving d = y_o + G P_abs(d) with the line input scaled by ``scale`` (Newton
    from d_start on the branch it starts on)."""
    lj = dict(li, pin=li["pin"] * scale)
    d = d_start
    for _ in range(100):
        p, dp, _, _ = _pabs_curve(lj, np.array([d]))
        step = (d - y_o - G * p[0]) / (1.0 - G * dp[0])
        d -= step
        if abs(step) < 1e-18:
            break
    return d


def test_power_step_formula():
    """07 P0-1 (d): a laser-power step with the heater held moves the self-locked resonance by
    d_lambda = G (dP_abs/dP_in) dP_in / (1 - G dP_abs/dlambda_r) to first order. The dynamic plant lands on the exact
    static solution of d = y_o + G P_abs(d; P_in (1 + f)) (y_o held) within 1 % at f = +10 % and +1 %; the first-order
    formula matches within 1.5 % at +1 % (its error is O(f): ~8 % at +10 %, docs/handoff/plant_thermal.md 6.4)."""
    cfg = _cfg(*ONE, *CLEAN)
    qs, code = _lock_code(cfg)
    t_step = 1500
    for frac in (0.10, 0.01):
        sc = _with_defaults(cfg, sg.step("laser.power_delta_db", t_step * 1e-6, 10 * math.log10(1 + frac)))
        pl = _plant(cfg, sc)
        _settle_on_upper_branch(pl, code, n_up=300, n_ramp=300, n_hold=t_step - 600 - 1)
        li = _line_inputs(pl)
        d0 = pl.S.rf[0, K.RF_DET, 0]
        p0 = pl.S.rf[0, K.RF_P_ABS, 0]
        p_chk, dp, _, _ = _pabs_curve(li, np.array([d0]))
        # the helper is the plant's absorption (to 2e-5: the plant's two level phases average to the phase at the
        # mean resonance only to second order in the 25 pm level split, ~0.2 fm of detuning)
        assert p_chk[0] == pytest.approx(p0, rel=2e-5)
        assert pl.tick < t_step
        pl.advance(None, 600)
        d1 = pl.S.rf[0, K.RF_DET, 0]
        G = li["G"]
        exact = _static_detuning(li, G, d0 - G * p0, d0, 1.0 + frac) - d0
        linear = G * p0 * frac / (1.0 - G * dp[0])
        assert exact > 0 and (d1 - d0) == pytest.approx(exact, rel=0.01)
        if frac <= 0.01:
            assert (d1 - d0) == pytest.approx(linear, rel=0.015)


# ================================================================================================ 07 P1-2
def test_neighbour_lock_steps_by_cross_residue():
    """07 P1-2: ring 0 locking onto its line steps ring 1 by the cross optical residue (ring 0's absorbed power
    heating ring 1). Line 0 comes on at t0 and ring 0 is brought onto its self-locked branch (heater above the
    up-fold, then down to its lock code); ring 1 sits between lines (heater code 15000, ~4 nm from line 0 in either
    order, absorbing ~1 uW). The same plant with the light off and the same heater programme is the counterfactual,
    so the heaters' own warm-up cancels. The whole response of ring 1 equals the thermal system driven by the
    recorded sources (absorbed and heater powers; junction and driver heat held) within 0.1 % of the step, and the
    step equals dc_gain[ring1, wg0] dP_abs,0 + dc_gain[ring1, wg1] dP_abs,1 within 1 %, carried by the cross path."""
    cfg = _cfg(*TWO, *CLEAN)
    qs = quasi_static(cfg, ambient_c=[25.0], bias_v=[1.05], batch=1, uncertainty="nominal", variation="off")
    code0 = int(qs.heater_code[0, 0, 0, 0])
    c1 = 15000
    t0 = 400
    n = 25000
    off = (sg.const("laser.line_enable", 0.0, absolute=True, elements=(1,)),
           sg.const("laser.line_enable", 0.0, absolute=True, elements=(0,)))
    rec = [("t_ring_k", 1, "last"), ("p_abs_w", 1, "last"), ("p_heater_w", 1, "last")]
    runs = {}
    for name, comps in (("light", off + (sg.step("laser.line_enable", t0 * 1e-6, 1.0, elements=(0,)),)),
                        ("dark", off)):
        pl = _plant(cfg, _with_defaults(cfg, *comps), record=rec, n_rec=n, heater_code=np.array([[code0, c1]]))
        pl.advance(None, t0)
        pl.advance({"tx.heater": np.array([[int(code0 * 1.3), c1]])}, 300)
        for c in np.linspace(int(code0 * 1.3), code0, 30).astype(np.int64):
            pl.advance({"tx.heater": np.array([[c, c1]])}, 10)
        pl.advance({"tx.heater": np.array([[code0, c1]])}, n - pl.tick)
        assert pl.tick == n
        tr = pl.traces()
        runs[name] = (pl, tr["t_ring_k@1:last"]["value"][:, 0, :], tr["p_abs_w@1:last"]["value"][:, 0, :],
                      tr["p_heater_w@1:last"]["value"][:, 0, :])
    pl, T, Pa, Ph = runs["light"]
    _, T_d, Pa_d, Ph_d = runs["dark"]
    np.testing.assert_array_equal(Ph, Ph_d)
    assert np.all(Pa_d == 0.0)
    assert Pa[-1, 0] > 1e-3 and Pa[-1, 1] < 1e-2 * Pa[-1, 0]           # ring 0 self-locked, ring 1 between lines
    assert abs(pl.S.rf[0, K.RF_DET, 0]) < _line_inputs(pl)["fwhm"]
    sysm = pl.sys
    u = np.zeros((n, sysm.n_src))
    u[:, sysm.heater_src] = Ph
    u[:, sysm.wg_src] = Pa
    for i in range(2):
        u[:, sysm.junction_src[i]] = pl.S.rf[0, K.RF_PJ, i]
        u[:, sysm.driver_src[i]] = pl.S.rf[0, K.RF_PD, i]
    u = np.concatenate([u, u[-1:]])                                    # simulate takes n + 1 boundary samples
    t_base = float(pl.S.wf[0, K.WF_T_BASE1])
    t_pred = th.simulate(sysm, u, 1e-6, hold="zoh", t_base_k=t_base)[1:, 0][:, sysm.ring_obs]
    step = T[-1, 1] - T_d[-1, 1]
    assert step > 0.1
    assert np.max(np.abs(t_pred[:, 1] - T[:, 1])) < 1e-3 * step
    np.testing.assert_array_equal(T[:t0, 1], T_d[:t0, 1])               # nothing before the light comes on
    G = th.dc_gain(sysm)[0]
    dP = Pa[-1] - Pa_d[-1]
    want = G[sysm.ring_obs[1], sysm.wg_src[0]] * dP[0] + G[sysm.ring_obs[1], sysm.wg_src[1]] * dP[1]
    assert step == pytest.approx(want, rel=0.01)
    assert G[sysm.ring_obs[1], sysm.wg_src[0]] * dP[0] > 0.9 * step       # the cross path carries the step


# ================================================================================================ 07 P1-7 / P1-8 / P1-9
def test_bias_step_instant_shift_then_selfheat_relaxation(lock1):
    """07 P1-7: a +0.5 V reverse bias step at lock, heater held, shifts the resonance red at once by the junction
    shift (about 25 pm; equal to the junction table's ring-level shift), then the self-heating relaxes it by
    s (Z_opt dP_abs + Z_junction dP_j + Z_driver dP_d) (blue: the ring absorbs less)."""
    cfg, qs, code = lock1
    t0 = 1500
    sc = _with_defaults(cfg, sg.step("tx.bias_v", t0 * 1e-6, 0.5))
    pl = _plant(cfg, sc, record=[("detuning_m", 1, "last"), ("bias_v", 1, "last"), ("t_ring_k", 1, "last")],
                n_rec=3000)
    _settle_on_upper_branch(pl, code, n_up=300, n_ramp=300, n_hold=t0 - 600)
    pre = {k: pl.S.rf[0, i, 0] for k, i in (("pabs", K.RF_P_ABS), ("pj", K.RF_PJ), ("pd", K.RF_PD))}
    pl.advance(None, 1500)
    tr = pl.traces()
    d = tr["detuning_m@1:last"]["value"][:, 0, 0]
    v = tr["bias_v@1:last"]["value"][:, 0, 0]
    k_on = int(np.flatnonzero(np.abs(np.diff(v)) > 0.1)[0]) + 1          # first interval at the new bias
    jump = d[k_on] - d[k_on - 1]
    # junction-table shift between the two biases at the ring temperature (mean of the two NRZ levels)
    rp = pl.phase.ring[0]
    r = pl.phase.table_row(0)
    out = []
    for vb in (v[k_on - 1], v[k_on]):
        sw = float(pl.P.drv_swing[0, 0])
        lo, hi = (vb - 0.5 * sw, vb + 0.5 * sw) if int(pl.P.drv_level[0]) == 0 else (vb - sw, vb)
        sh = np.zeros(1)
        f, c, rs, fl = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1, np.int8)
        tot = 0.0
        for vv in (lo, hi):
            ph.resonance_shift_row(rp, np.array([pl.S.rf[0, K.RF_T_RING, 0]]), np.array([vv]), pl.phase.grid,
                                   pl.phase.dneff[r], pl.phase.alpha_fca[r], pl.phase.capacitance[r],
                                   pl.phase.r_series[r], sh, f, c, rs, fl)
            tot += 0.5 * sh[0]
        out.append(tot)
    want_jump = out[1] - out[0]
    assert 10e-12 < want_jump < 50e-12                                    # "about 25 pm"
    t = tr["t_ring_k@1:last"]["value"][:, 0, 0]
    s = _line_inputs(pl)["s"]
    # the instant part: the detuning step minus what the ring temperature did in that tick (the sub-us self-heating
    # pole already answers within the tick)
    inst = (d[k_on] - d[k_on - 1]) - s * (t[k_on] - t[k_on - 1])
    assert inst == pytest.approx(want_jump, rel=0.01)
    # then the self-heating and data-heat relaxation to the new static state: s (Z_opt dP_abs + Z_j dP_j + Z_d dP_d)
    post = {k: pl.S.rf[0, i, 0] for k, i in (("pabs", K.RF_P_ABS), ("pj", K.RF_PJ), ("pd", K.RF_PD))}
    sysm = pl.sys
    Gd = th.dc_gain(sysm)[0]
    ro = sysm.ring_obs[0]
    dT = (Gd[ro, sysm.wg_src[0]] * (post["pabs"] - pre["pabs"])
          + Gd[ro, sysm.junction_src[0]] * (post["pj"] - pre["pj"])
          + Gd[ro, sysm.driver_src[0]] * (post["pd"] - pre["pd"]))
    relax = d[-1] - d[k_on - 1] - want_jump
    assert post["pabs"] < pre["pabs"] and relax < 0                       # blue relaxation: the ring absorbs less
    assert relax == pytest.approx(s * dT, rel=0.05)


def test_traffic_enable_step_and_transition_density(lock1):
    """07 P1-8: a traffic-enable step at lock moves the hot resonance by the analytic amount
    s (Z_junction P_j + Z_driver P_d) / (1 - G dP_abs/dlambda) within 5 %; halving the transition density halves P_j."""
    _, qs, code = lock1
    cfg = _cfg(*ONE, *DRIVER_ON)
    t0 = 1500
    sc = _with_defaults(cfg, sg.const("data.driver_enable", 0.0, absolute=True),
                        sg.step("data.driver_enable", t0 * 1e-6, 1.0))
    pl = _plant(cfg, sc)
    _settle_on_upper_branch(pl, code, n_up=300, n_ramp=300, n_hold=t0 - 601)
    d0 = pl.S.rf[0, K.RF_DET, 0]
    li = _line_inputs(pl)
    _, dp, _, _ = _pabs_curve(li, np.array([d0]))
    assert pl.S.rf[0, K.RF_PJ, 0] == 0.0
    pl.advance(None, 45000)                                   # the driver site path has ms poles
    pj, pd = pl.S.rf[0, K.RF_PJ, 0], pl.S.rf[0, K.RF_PD, 0]
    assert pj > 0 and pd > 0
    sysm = pl.sys
    Gd = th.dc_gain(sysm)[0]
    ro = sysm.ring_obs[0]
    dT = Gd[ro, sysm.junction_src[0]] * pj + Gd[ro, sysm.driver_src[0]] * pd
    want = li["s"] * dT / (1.0 - li["G"] * dp[0])
    assert (pl.S.rf[0, K.RF_DET, 0] - d0) == pytest.approx(want, rel=0.05)
    # halving the transition density halves the junction CV^2 heat (the driver static part stays)
    cfg_h = _cfg(*ONE, *DRIVER_ON, "datapath.transition_density=0.25")
    pl_h = _plant(cfg_h)
    pl_f = _plant(_cfg(*ONE, *DRIVER_ON, "datapath.transition_density=0.5"))
    pl_h.advance(None, 5)
    pl_f.advance(None, 5)
    assert pl_h.S.rf[0, K.RF_PJ, 0] == pytest.approx(0.5 * pl_f.S.rf[0, K.RF_PJ, 0], rel=1e-9)


def test_monitor_tracks_intracavity_energy():
    """07 P1-9 test_monitor_tracks_intracavity_energy: for one CW line the monitor current (minus dark current) times
    kappa_abs/kappa_d equals R P_abs, with kappa_abs/kappa_d = eta (1 - a)(1 + t_d^2 a)/(K_d^2 a) of the exact ring,
    at any detuning; and with the lasers off the monitor reads the Arrhenius dark current at the PD observer
    temperature, which the heater warms (test_dark_current_vs_heater)."""
    cfg = _cfg(*ONE, *CLEAN)
    sc = _with_defaults(cfg, sg.const("data.swing_scale", 0.0, absolute=True))     # squelch: one junction level
    pl = _plant(cfg, sc)
    mhw = pl.monitor_hw
    for code in (0, 15000, 22000, 26000):
        pl.advance({"tx.heater": np.full((1, 1), code, np.int64)}, 150)
        tr = pl.truth()
        li = _line_inputs(pl)
        assert li["phi0"] == li["phi1"]
        _, i_dark = mhw.photocurrent_a(0.0, 0.0, tr.t_pd_k)
        resp = mhw.pr[0, 0, 0] * mhw.pw[0, 0]                               # R x gain_scale
        a, t2 = li["a0"], math.sqrt(1.0 - li["k2d"])
        ratio = li["e0"] * (1 - a) * (1 + t2 * t2 * a) / (li["k2d"] * a)
        lhs = (tr.i_monitor_a[0, 0] - i_dark[0, 0]) * ratio
        assert lhs == pytest.approx(resp * tr.p_abs_w[0, 0], rel=1e-9)
    dark = _with_defaults(cfg, sg.const("laser.line_enable", 0.0, absolute=True))
    i_m = []
    for code in (0, 30000):
        pl2 = _plant(cfg, dark, heater_code=np.full((1, 1), code))
        pl2.advance(None, 400)
        tr = pl2.truth()
        _, i_d = mhw.photocurrent_a(0.0, 0.0, tr.t_pd_k)
        assert tr.i_monitor_a[0, 0] == pytest.approx(i_d[0, 0], rel=1e-12)
        i_m.append((tr.t_pd_k[0, 0], tr.i_monitor_a[0, 0]))
    assert i_m[1][0] > i_m[0][0] + 0.5 and i_m[1][1] > i_m[0][1]


# ================================================================================================ 07 P1-13
def test_monitor_noise_sigma_invariant_to_dt():
    """07 P1-13: the monitor noise at the ADC (sample-to-sample code differences of a steady plant) has the same
    sigma within 5 % at dt = 1 us and 0.5 us (noise drawn per sample by exact discretization)."""
    sig = []
    for dt in (1e-6, 5e-7):
        cfg = _cfg("ring.bias_dac.noise_rms_codes=0.0", "ring.thermal.tail_share_frac=0.0", f"sim.dt_plant_s={dt}")
        per_us = int(round(1e-6 / dt))
        pl = _plant(cfg)
        pl.advance({"tx.heater": np.zeros((1, 8), np.int64)}, 300 * per_us)
        series = [[] for _ in range(8)]
        for _ in range(1200):
            pl.advance(None, 10 * per_us)                 # one controller period: the block of the last edge
            s, _ = pl.sample_sensors()
            cnt = s["tx.monitor.count"][0]
            for n in range(8):
                series[n].extend(int(c) for c in s["tx.monitor.samples"][0, n, :cnt[n]])
        x = np.concatenate([np.diff(np.asarray(c, float)) for c in series])
        assert x.size > 20000
        sig.append(np.std(x))
    assert sig[0] > 1.0
    assert sig[1] == pytest.approx(sig[0], rel=0.05)


def test_batch_and_thread_invariance():
    """07 P1-13 at plant level: element b of a B = 4 run equals the B = 1 run with seeds base + b (device,
    uncertainty, noise, scenario), for any thread count."""
    cfg = _cfg("sim.reset_ramp_s=1e-4")
    rec = [("t_ring_k", 1, "last"), ("monitor_code", 1, "last"), ("p_heater_w", 1, "last")]
    kw = dict(duration_s=300e-6, record=rec, oracle=False, histogram=False)
    r4a = run(cfg, None, batch=4, seeds={"device": 7, "uncertainty": 7, "noise": 3, "scenario": 5}, n_threads=1,
              **kw)
    r4b = run(cfg, None, batch=4, seeds={"device": 7, "uncertainty": 7, "noise": 3, "scenario": 5}, n_threads=4,
              **kw)
    r1 = run(cfg, None, batch=1, seeds={"device": 9, "uncertainty": 9, "noise": 5, "scenario": 7}, **kw)
    for key in r4a.trace:
        np.testing.assert_array_equal(r4a.trace[key]["value"], r4b.trace[key]["value"])
        np.testing.assert_array_equal(r4a.trace[key]["value"][:, 2], r1.trace[key]["value"][:, 0])


# ================================================================================================ 07 P2-1
def _nonlinear_magnitudes(beta_m_per_w, tau_fc_s, n2_m2_per_w, a_eff_m2=0.094e-12, q_target=4200.0, p_dbm=9.0):
    """TPA loss (dB/cm), TPA free-carrier dispersion shift (m, magnitude), Kerr shift (m) and the thermal self-heating
    shift G P_abs (m) of the one-ring CLEAN plant on resonance at ``p_dbm`` in the bus, with the intracavity power of
    the plant's own exact ring (u_cw / T_rt) scaled linearly to Q_L = q_target (U ~ Q_L at a fixed coupling ratio).
    Intensity I = P_circ / A_eff (A_eff 0.094 um^2 for the 400 x 200 nm strip, 03 section 7.2; 07 quotes ~0.15);
    alpha_TPA = beta I; carriers N = beta I^2 tau / (2 h nu) (03 section 7.1, Johnson et al. 2006 Eq. 25 with
    V = A_eff L); Soref-Bennett at 1.3 um: dn = 6.2e-22 N + 6.0e-18 N^0.8 (N in cm^-3); shifts lambda dn / n_g with
    n_g from the plant's T_rt; also alpha_i (dB/cm) from the round-trip amplitude a."""
    cfg = _cfg(*ONE, *CLEAN)
    pl = _plant(cfg)
    pl.advance(None, 3)
    li = _line_inputs(pl)
    p_ring = 10 ** (p_dbm / 10) * 1e-3
    r = op.ring_exact(2 * math.pi * 1000.0, li["k2t"], li["k2d"], li["a0"], li["trt"])
    q_l = float(pl.pp.optics.q_loaded[0, 0])
    p_circ = float(r.u_cw) * p_ring / li["trt"] * (q_target / q_l)
    inten = p_circ / a_eff_m2
    length_cm = 2 * math.pi * float(pl.pp.optics.radius_m[0, 0]) * 100.0
    alpha_i_db_cm = -20.0 * math.log10(li["a0"]) / length_cm
    alpha_tpa_db_cm = 10 * math.log10(math.e) * beta_m_per_w * inten / 100.0
    hnu = 6.62607015e-34 * C0 / li["lam"]
    n_g = li["trt"] * C0 / (2 * math.pi * float(pl.pp.optics.radius_m[0, 0]))
    n_cm3 = beta_m_per_w * inten ** 2 * tau_fc_s / (2 * hnu) * 1e-6
    dn = 6.2e-22 * n_cm3 + 6.0e-18 * n_cm3 ** 0.8
    thermal = li["G"] * float(r.p_loss) * p_ring
    return dict(alpha_tpa_db_cm=alpha_tpa_db_cm, alpha_i_db_cm=alpha_i_db_cm, fcd_m=li["lam"] * dn / n_g,
                kerr_m=li["lam"] * n2_m2_per_w * inten / n_g, thermal_m=thermal, i_gw_cm2=inten / 1e13,
                buildup=p_circ / p_ring)


def test_nonlinear_bounds():
    """07 P2-1 with the plant's own ring: at +9 dBm, Q_L 4200, beta_TPA 0.5-2 cm/GW and n2 up to 6.7e-18 m^2/W
    (Bristow 2007 upper end) the intensity is ~0.09 GW/cm^2 (build-up ~10), and:
    TPA loss 0.2-0.8 dB/cm, below 1 % of alpha_i (~110 dB/cm); Kerr shift <= 2 pm, below 0.1 % of the thermal
    self-heating shift (~5 nm); TPA free-carrier shift 0.4-1.2 pm under junction sweep-out (tau_fc = 10 ps,
    03 section 7.1; 2.6-8.1 pm at 0.1 ns), and at most 1.5 % of the thermal shift even at tau_fc = 1 ns
    (17-55 pm). The exclusion holds (every effect is a percent-level correction to the thermal pull the loop
    already tracks), but 07's literal numbers do not: see test_nonlinear_bounds_07_literal."""
    for beta in (0.5e-11, 2.0e-11):
        m = _nonlinear_magnitudes(beta, 1e-11, 6.7e-18)
        assert 0.07 < m["i_gw_cm2"] < 0.12
        assert m["alpha_tpa_db_cm"] < 0.01 * m["alpha_i_db_cm"]
        assert 90.0 < m["alpha_i_db_cm"] < 130.0
        assert m["kerr_m"] < 2e-12 and m["kerr_m"] < 1e-3 * m["thermal_m"]
        assert m["fcd_m"] < 1.5e-12
        m1 = _nonlinear_magnitudes(beta, 1e-9, 6.7e-18)
        assert m1["fcd_m"] < 0.015 * m1["thermal_m"]


@pytest.mark.xfail(strict=True, reason="07 P2-1's literal bounds (TPA < 0.2 dB/cm, TPA-carrier shift < 5 pm at "
                   "1 ns and < 0.5 pm at 0.1 ns, Kerr < 1 pm) fail at +9 dBm, Q 4200, beta 2 cm/GW; 03 section 7.2 "
                   "gives the same magnitudes (9-28 pm at 8 mW, 1 ns). See docs/handoff/engine.md.")
def test_nonlinear_bounds_07_literal():
    m = _nonlinear_magnitudes(2.0e-11, 1e-9, 6.7e-18, a_eff_m2=0.15e-12)
    m01 = _nonlinear_magnitudes(2.0e-11, 1e-10, 6.7e-18, a_eff_m2=0.15e-12)
    assert m["alpha_tpa_db_cm"] < 0.2
    assert m["fcd_m"] < 5e-12
    assert m01["fcd_m"] < 0.5e-12
    assert m["kerr_m"] < 1e-12


# ================================================================================================ quasi-static survey
def test_quasi_static_survey_g26():
    """07 P0-10 quasi_static over the 8-ring g26 bank: shapes, code-range coverage, FSR-hop and wrong-line risk, and
    the BND-4 oracle relOMA variation over ambient at each bias."""
    cfg = _cfg()
    qs = quasi_static(cfg, ambient_c=[0.0, 50.0, 105.0], bias_v=[1.05, 3.0], batch=1, uncertainty="nominal",
                      variation="off", n_scan=151, iterations=3)
    assert qs.heater_code.shape == (3, 2, 1, 8)
    assert qs.bnd4_reloma_variation_db.shape == (2, 1)
    assert 0.0 <= qs.code_range_coverage[0] <= 1.0
    assert np.all(np.isfinite(qs.reloma_db))
    # the g26 rings sit 7.6 nm blue of their lines with a 2.29 nm grid: a heater-up sweep crosses three foreign
    # lines (n+1, n+2, n+3; docs/handoff/integrator.md section 4) wherever the line exists
    assert np.max(qs.wrong_line_risk) >= 3
    # the lock temperature is set by the line (fixed laser grid), not by the ambient: the heater supplies the rest
    ok = qs.code_ok[0] & qs.code_ok[2]
    assert ok.any()
    assert np.all(np.abs(qs.t_ring_k[2] - qs.t_ring_k[0])[ok] < 0.05)
    assert np.all((qs.p_heater_w[2] < qs.p_heater_w[0])[ok])
    assert np.all((qs.heater_code[2] < qs.heater_code[0])[ok])


# ================================================================================================ 07 P0-10 auto policy
def test_auto_substeps_match_fixed():
    """07 P0-10 substep_policy auto: on the self-locked branch under a slow base-temperature ramp (0.2 K/ms, a soak
    edge), with a 10 kHz controller grid, the multi-tick steps (exact FOH thermal advance, implicit self-heating per
    substep, sensors per tick) follow the fixed 1 us steps within max_detuning_step (0.2 pm) plus 0.1 pm, in fewer
    than half the substeps; the heater energy and mean relOMA agree within 1e-3 relative."""
    cfg = _cfg(*ONE, *CLEAN, "controller_interface.update_rate_hz=10000.0")
    qs, code = _lock_code(cfg)
    res = {}
    for pol in ("fixed", "auto"):
        sc = _with_defaults(cfg, sg.pwl("thermal.t_base_k", [0.0, 1.5e-3, 6.5e-3], [0.0, 0.0, 1.0]))
        pl = _plant(cfg, sc, record=[("detuning_m", 100, "mean"), ("t_ring_k", 100, "last")], n_rec=6500,
                    substep_policy=pol)
        _settle_on_upper_branch(pl, code, n_up=300, n_ramp=300, n_hold=900)
        pl.advance(None, 5000)
        res[pol] = (pl.traces(), pl.metrics(), pl.diagnostics())
    tf, ta = res["fixed"][0], res["auto"][0]
    d_f = tf["detuning_m@100:mean"]["value"][15:, 0, 0]
    d_a = ta["detuning_m@100:mean"]["value"][15:, 0, 0]
    assert np.max(np.abs(d_f - d_a)) < 0.3e-12
    T_f = tf["t_ring_k@100:last"]["value"][15:, 0, 0]
    T_a = ta["t_ring_k@100:last"]["value"][15:, 0, 0]
    assert np.max(np.abs(T_f - T_a)) < 0.3e-12 / 60e-12
    da = res["auto"][2]
    assert int(da["substeps"][0]) < 0.5 * int(da["ticks"][0])
    assert int(da["largest_substep_ticks"][0]) >= 4
    for k in ("heater_energy_j", "mean_reloma"):
        assert res["auto"][1][k][0, 0] == pytest.approx(res["fixed"][1][k][0, 0], rel=1e-3)


def _growth_factor(k, a, r, k_rows=0.0):
    """Per-step growth factor mu > 1 of the implicit scheme linearized at an equilibrium:
    1 = (k mu + k_rows) sum (1 - a) r / (mu - a) over the ring's own optical-heating lags (derived in
    test_selfheat_divergence_rate). A finite root needs k sum (1 - a) r < 1; above that the step jumps to another
    root at once."""
    assert k * np.sum((1 - a) * r) < 1.0
    lo, hi = 1.0 + 1e-12, 1e6
    f = lambda mu: (k * mu + k_rows) * np.sum((1 - a) * r / (mu - a)) - 1.0      # noqa: E731
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def _rows_slope(pl, t, dT=1e-4):
    """dP_abs/dT of ring 0 through the optics rows (loss, coupling, junction shift: prepared at the previous tick's
    ring temperature, an explicit one-tick-delayed term next to the implicit one) at the solve temperature t: rows
    prepared at T_ring + dT minus rows at T_ring. The plant state and tick are restored."""
    S = pl.S
    snap = [np.array(a, copy=True) for a in S]
    tick = pl.tick
    out = []
    for d in (dT, 0.0):
        S.rf[0, K.RF_T_RING, 0] += d
        pl.advance(None, 1)
        out.append(K.eng_pabs(t, 0, S.sh_args_f[0], pl.P.sh_args_i[0])[0])
        for arr, c in zip(S, snap):
            arr[...] = c
        pl.tick = tick
    return (out[0] - out[1]) / dT


def _equilibria(pl, p_heater):
    """Static ring temperatures T = T_rest + Z_opt,self P_abs(T) at heater power ``p_heater`` (junction and driver
    heat as in the plant): (t_rest, z_or, roots on a 3000-point grid)."""
    sysm = pl.sys
    G = th.dc_gain(sysm)[0]
    ro = sysm.ring_obs[0]
    S, P = pl.S, pl.P
    t_rest = (S.wf[0, K.WF_T_BASE1] + G[ro, sysm.heater_src[0]] * p_heater
              + G[ro, sysm.junction_src[0]] * S.rf[0, K.RF_PJ, 0] + G[ro, sysm.driver_src[0]] * S.rf[0, K.RF_PD, 0])
    z_or = G[ro, sysm.wg_src[0]]
    a_f, a_i = S.sh_args_f[0], P.sh_args_i[0]
    T = np.linspace(t_rest, t_rest + z_or * 3e-3, 3001)
    F = np.array([t - t_rest - z_or * K.eng_pabs(t, 0, a_f, a_i)[0] for t in T])
    return t_rest, z_or, T[:-1][np.sign(F[:-1]) != np.sign(F[1:])]


@pytest.mark.parametrize("dt", [1e-6, 1e-7])
def test_selfheat_divergence_rate(dt):
    """07 P0-1 (c), unstable side: held at the middle (open-loop unstable) equilibrium of the bistable region, with
    the resonance blue of the laser, a 10 uK perturbation grows exponentially at the linearized rate. With the
    ring's own optical-heating lags (r_m, tau_m) and k = dP_abs/dT at the equilibrium, the implicit step's
    linearization (x_m' = a_m x_m + (1 - a_m) r_m (k T' + k_rows T), k_rows the slope through the optics rows
    prepared at the previous tick's temperature, ~1 % of k) gives the per-step factor mu from
    1 = (k mu + k_rows) sum (1 - a_m) r_m/(mu - a_m), a_m = exp(-dt/tau_m); the simulated log growth rate over
    10 uK - 1 mK matches ln mu within 2 %, and at dt = 0.1 us it is within 5 % of the continuous rate s*
    (1 = (k + k_rows) sum r_m/(1 + s tau_m); the single-pole form is 07's (G P' - 1)/tau_fast). The heater code
    is chosen just above the down-fold, where Z_opt k ~ 1.2-2 at the middle root, so that dt = 1 us still resolves
    the growth (k sum (1 - a) r < 1)."""
    cfg = _cfg(*ONE, *CLEAN, f"sim.dt_plant_s={dt}")
    qs, code = _lock_code(_cfg(*ONE, *CLEAN))
    probe = _plant(_cfg(*ONE, *CLEAN))                        # optics rows prepared near the lock temperature
    _settle_on_upper_branch(probe, code)
    w_per_code = probe.S.rf[0, K.RF_HE_P, 0] / code                 # power-linear driver (to ~1e-5)
    a_f, a_i = probe.S.sh_args_f[0], probe.P.sh_args_i[0]
    lo, hi = int(code * 0.85), code                           # one root (lower branch) / three (lock code)
    assert _equilibria(probe, w_per_code * lo)[2].size == 1 and _equilibria(probe, w_per_code * hi)[2].size == 3
    while hi - lo > 1:                                        # the down-fold code
        mid = (lo + hi) // 2
        if _equilibria(probe, w_per_code * mid)[2].size == 3:
            hi = mid
        else:
            lo = mid
    c_sel = None
    for c in range(hi, hi + 200, 2):                          # first code with Z k > 1.5 at the middle root
        t_rest, z_or, roots = _equilibria(probe, w_per_code * c)
        if roots.size == 3:
            t_mid = roots[1]
            for _ in range(50):
                p, dp = K.eng_pabs(t_mid, 0, a_f, a_i)
                t_mid -= (t_mid - t_rest - z_or * p) / (1.0 - z_or * dp)
            if z_or * K.eng_pabs(t_mid, 0, a_f, a_i)[1] > 1.5:
                c_sel = c
                break
    assert c_sel is not None
    per_us = int(round(1e-6 / dt))
    pl = _plant(cfg, sh_tol_k=1e-11)                          # upper branch at c_sel (rows near t_mid)
    _settle_on_upper_branch(pl, code, n_up=300 * per_us, n_ramp=300 * per_us, n_hold=300 * per_us)
    pl.advance({"tx.heater": np.full((1, 1), c_sel, np.int64)}, 300 * per_us)
    S, P = pl.S, pl.P
    sysm = pl.sys
    ro = sysm.ring_obs[0]
    args_f, args_i = S.sh_args_f[0], P.sh_args_i[0]
    tau, r = _self_lags(pl)
    # the optics rows (loss, coupling, phases) are prepared at the previous tick's ring temperature: inject the
    # equilibrium, step once so the rows are prepared there and solve again until it stops moving (the fixed point
    # contracts ~60x per pass), then add the perturbation
    mids = []
    final = False
    for it in range(12):
        p_h = S.rf[0, K.RF_HE_P, 0]
        t_rest, z_or, roots = _equilibria(pl, p_h)
        assert roots.size == 3, "the code is not inside the bistable region"
        t_mid = roots[1]
        for _ in range(50):                                   # refine the middle root (Newton)
            p, dp = K.eng_pabs(t_mid, 0, args_f, args_i)
            t_mid -= (t_mid - t_rest - z_or * p) / (1.0 - z_or * dp)
        p_mid, kslope = K.eng_pabs(t_mid, 0, args_f, args_i)
        mids.append(t_mid)
        final = it > 0 and abs(mids[-1] - mids[-2]) < 1e-9
        # inject the static state (every lag at its steady value), with a 10 uK perturbation on the last pass
        dP = 1e-5 / z_or if final else 0.0
        u = np.zeros((1, sysm.n_src))
        u[0, sysm.heater_src[0]] = p_h
        u[0, sysm.junction_src[0]] = S.rf[0, K.RF_PJ, 0]
        u[0, sysm.driver_src[0]] = S.rf[0, K.RF_PD, 0]
        u[0, sysm.wg_src[0]] = p_mid + dP
        x = sysm.steady_state(u)
        S.th_x[0] = x[0]
        S.th_t_obs[0] = sysm.outputs(x, S.wf[0, K.WF_T_BASE1])[0]
        for idx in (K.RF_T_RING, K.RF_T_BRANCH, K.RF_T_NEW):
            S.rf[0, idx, 0] = S.th_t_obs[0, ro]
        S.rf[0, K.RF_P_PREV, 0] = p_mid + dP
        S.rf[0, K.RF_T_HEATER, 0] = S.th_t_obs[0, sysm.heater_obs[0]]
        if final:
            break
        pl.advance(None, 1)
    assert final, mids
    k_rows = _rows_slope(pl, t_mid)
    assert abs(k_rows) < 0.05 * kslope
    assert S.rf[0, K.RF_T_RING, 0] - t_mid == pytest.approx(1e-5, rel=1e-3)
    assert 1.2 < z_or * kslope < 2.0                          # open-loop unstable, resolvable at 1 us
    temps = []
    for _ in range(int(round(1e-6 / dt)) * 40):
        pl.advance(None, 1)
        temps.append(S.rf[0, K.RF_T_RING, 0])
    d = np.array(temps) - t_mid
    assert d[0] > 0 and np.all(np.diff(d[d < 0.05]) > 0)      # monotone growth, no oscillation
    w = np.flatnonzero((d > 3e-5) & (d < 1e-3))
    assert w.size >= 10
    rate = np.polyfit(w, np.log(d[w]), 1)[0]                   # per step
    mu = _growth_factor(kslope, np.exp(-dt / tau), r, k_rows)
    assert rate == pytest.approx(math.log(mu), rel=0.02)
    if dt <= 1e-7:
        lo, hi = 1.0, 1e12
        for _ in range(200):
            s = math.sqrt(lo * hi)
            if (kslope + k_rows) * np.sum(r / (1 + s * tau)) > 1.0:
                lo = s
            else:
                hi = s
        assert rate / dt == pytest.approx(s, rel=0.05)
    assert _line_inputs(pl)["dpl"] > 0


# ================================================================================================ Truth detuning
def test_truth_detuning_is_the_phase_model_resonance():
    """ARCHITECTURE Truth: detuning_m is lambda_res - lambda_line of the order nearest the declared line, where
    lambda_res is the mean of the two NRZ level resonances of the plant's own phase model (mrm.plant.phase: the cold
    curve translated by the R1 + junction shift). Solved here independently (root of the mean level phase at the
    line over wavelength, ring_optics_row at the end-of-tick temperature) at detunings from -3 nm to +0.3 nm: the
    kernel agrees to 1e-3 pm. (The slope 2 pi c t_rt / lambda_line^2 that it used before is the slope of the
    UNshifted curve, 2 shift / lambda = 1.2 % too small after the 7.6 nm a g26 ring is heated by: -3069 pm against
    -3040 pm, 3.6 pm at +300 pm.) Also: the in-tick thermo-optic phase slope of the optics rows is that exact slope."""
    cfg = _cfg()
    pl = _plant(cfg, heater_code=np.full((1, 8), 30000))
    pl.advance(None, 300)
    rp = pl.phase.ring[0]
    r = pl.phase.table_row(0)
    N = pl.N
    seen = []
    for code in (30000, 36000, 40000):
        pl.advance({"tx.heater": np.full((1, 8), code, np.int64)}, 400)
        tr = pl.truth()
        S = pl.S
        t_ring, t_h = S.rf[0, K.RF_T_RING].copy(), S.rf[0, K.RF_T_HEATER].copy()
        vlo, vhi = S.rf[0, K.RF_V_LO].copy(), S.rf[0, K.RF_V_HI].copy()

        def mean_phase(lam, n):
            o = {k: np.zeros((N, 1)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
            sh, cj, rs, jf = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N, np.int8)
            ph.ring_optics_row(rp, np.array([lam]), t_ring, t_h, vlo, vhi, pl.phase.grid, pl.phase.dneff[r],
                               pl.phase.alpha_fca[r], pl.phase.capacitance[r], pl.phase.r_series[r], o["phi0"],
                               o["phi1"], o["a0"], o["a1"], o["e0"], o["e1"], o["k2t"], o["k2d"], o["trt"], sh, cj, rs,
                               jf)
            return 0.5 * (o["phi0"][n, 0] + o["phi1"][n, 0]), sh[n], o["trt"][n, 0]
        for n in range(N):
            l = int(pl.P.ring_to_line[n])
            lam_l = S.lf[0, K.LF_LAM, l]
            phl, shl, trt = mean_phase(lam_l, n)
            m = math.floor(phl / (2 * math.pi) + 0.5)
            a, b_ = lam_l, lam_l + float(tr.detuning_m[0, n]) + 1e-12
            fa, fb = phl - 2 * math.pi * m, mean_phase(b_, n)[0] - 2 * math.pi * m
            for _ in range(40):
                if fb == fa:
                    break
                c = b_ - fb * (b_ - a) / (fb - fa)
                a, fa = b_, fb
                b_, fb = c, mean_phase(c, n)[0] - 2 * math.pi * m
            exact = b_ - lam_l
            assert abs(tr.detuning_m[0, n] - exact) < 1e-15, (code, n, tr.detuning_m[0, n], exact)
            assert int(tr.order[0, n]) == m
            seen.append(exact)
            # the optics rows' d phi / d lambda_res is the slope at the shifted coordinate x (to 1e-5: the rows are
            # prepared at the previous tick's temperature); the unshifted slope is 1-1.2 % off
            x = lam_l - shl[0]
            v = _rows(pl)
            want = 2 * math.pi * C0 * trt / x ** 2
            assert v[16][n, l] == pytest.approx(want, rel=1e-5)
            assert abs(2 * math.pi * C0 * trt / lam_l ** 2 / want - 1.0) > 2e-3
    seen = np.asarray(seen)
    assert seen.min() < -2.5e-9 and seen.max() > 2e-10                  # the range the test claims
    tr = pl.truth()
    assert tr.x.shape == (1, pl.sys.lag_src.shape[0]) and len(tr.conditions) == 1
    assert np.array_equal(tr.conditions[0].lam_m, pl.S.lf[0, K.LF_LAM])
    np.testing.assert_array_equal(tr.heater_applied, tr.heater_code)


# ================================================================================================ 07 P0-10 auto at events
def test_auto_substeps_resolve_disturbance_steps():
    """07 P0-10 auto, at the tick a disturbance starts (a +1 dB laser-power step and a traffic-enable step at lock):
    the monitor samples a 10 kHz controller receives equal the fixed policy's within 10 codes (the step itself is
    ~2400 codes; common random numbers, so only the auto approximations differ). The auto rules only looked back,
    so the first substep after the step was up to 64 ticks and the monitor saw a linear ramp to the end value
    (43138 -> 44505) instead of the jump to 45478 and its relaxation. Fewer than 0.6 substeps per tick."""
    cfg = _cfg(*ONE, *CLEAN, "controller_interface.update_rate_hz=10000.0")
    qs, code = _lock_code(cfg)
    t_step = 1630
    for comp in (sg.step("laser.power_delta_db", t_step * 1e-6, 1.0),
                 (sg.const("data.driver_enable", 0.0, absolute=True),
                  sg.step("data.driver_enable", t_step * 1e-6, 1.0))):
        comps = comp if isinstance(comp, tuple) else (comp,)
        res = {}
        for pol in ("fixed", "auto"):
            pl = _plant(cfg, _with_defaults(cfg, *comps), substep_policy=pol)
            _settle_on_upper_branch(pl, code, n_up=300, n_ramp=300, n_hold=900)
            blocks = []
            while pl.tick < 2600:
                pl.advance(None, 100)
                s, _ = pl.sample_sensors()
                c = int(s["tx.monitor.count"][0, 0])
                blocks.append(np.asarray(s["tx.monitor.samples"][0, 0, :c], float))
            res[pol] = (np.concatenate(blocks), pl.diagnostics())
        f, a = res["fixed"][0], res["auto"][0]
        assert f.shape == a.shape
        assert np.max(f) - np.min(f) > 300                         # the step is visible
        assert np.max(np.abs(f - a)) <= 10, np.max(np.abs(f - a))
        d = res["auto"][1]
        assert int(d["substeps"][0]) < 0.6 * int(d["ticks"][0])
        assert int(d["largest_substep_ticks"][0]) >= 16


def test_truth_fed_oracle_runs_on_the_engine():
    """07 P1-1: the Truth-fed oracle (mrm.validation.oracle.Oracle, which needs the thermal lag states and the
    per-world conditions in Truth) holds the g26 bank through mrm.sim.runner.run, and three independent operating
    points agree: the oracle's own target, the engine's oracle lock reference (MetricsSpec lock_reference='oracle',
    parabola-refined) and the quasi-static survey (07 P0-10) within 1.5 pm; the held rings end within 3 pm of the
    engine's reference (the two oracles differ in their OMA port: the ring's own through port vs the bus output)."""
    from mrm.sim.runner import load_inputs
    from mrm.validation.oracle import Oracle
    cfg, pp, _ = load_inputs(_cfg(), batch=1, seeds=1, uncertainty="nominal", variation="off")
    qs = quasi_static(pp, cfg=cfg, ambient_c=[25.0], bias_v=[1.05])
    codes = qs.heater_code[0, 0].astype(np.int64)
    orc = Oracle(pp, cfg)
    r = run(pp, orc, cfg=cfg, duration_s=3e-3, batch=1, seeds=1, heater_code=codes, reset_protocol="branch_high",
            metrics={"lock_reference": "oracle"}, histogram=False)
    ref = r.metrics["lock_reference_pm"][0]
    assert np.all((ref > 50.0) & (ref < 200.0))                       # red of the line, on the self-locked side
    np.testing.assert_allclose(qs.detuning_target_m[0, 0, 0] * 1e12, ref, atol=1.5)
    np.testing.assert_allclose(orc.det_target[0] * 1e12, ref, atol=1.5)
    np.testing.assert_allclose(r.truth_final.detuning_m[0] * 1e12, ref, atol=3.0)


@pytest.mark.parametrize("self_heating", ["on", "off"])
def test_metrics_invariant_to_dt(self_heating):
    """07 4.2: the metrics agree within 1 % between dt = 1 us and 0.25 us, with self-heating on and off (one ring,
    CLEAN settings; a cold start, the heater above the up-fold, a ramp down onto the self-locked branch and a hold,
    the same programme in physical time)."""
    base = [*ONE, *CLEAN, f'fidelity.self_heating="{self_heating}"']
    _, code = _lock_code(_cfg(*ONE, *CLEAN))
    out = []
    for dt in (1e-6, 2.5e-7):
        cfg = _cfg(*base, f"sim.dt_plant_s={dt}")
        per_us = int(round(1e-6 / dt))
        pl = _plant(cfg)
        _settle_on_upper_branch(pl, code, n_up=300 * per_us, n_ramp=300 * per_us, n_hold=600 * per_us)
        out.append(pl.metrics())
    # (max |detuning| is left out: the heater sweeps the cold ring through half an FSR, where the nearest order
    # changes and |d| peaks at ~FSR/2 for one sample; that sample's value depends on where the tick grid falls)
    for key in ("mean_detuning_pm", "rms_detuning_pm", "mean_reloma", "mean_p_abs_w", "heater_energy_j"):
        a, b = float(out[0][key][0, 0]), float(out[1][key][0, 0])
        assert b == pytest.approx(a, rel=0.01), (key, a, b)
    assert float(out[0]["mean_p_abs_w"][0, 0]) > 1e-4                   # the light is on and absorbed
