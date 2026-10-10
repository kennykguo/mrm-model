"""The self-heating adapter on real optics rows of the g26 plant (owner: engine; docs/handoff/perf.md P1, P4).

1. The engine's adapter (mrm.sim.kernel eng_precompute / eng_pabs / eng_commit, optics.sh_bounds: caches computed once per
   substep, no checks) against the optics adapter the reference engine runs (mrm.plant.optics selfheat_pabs /
   selfheat_commit / selfheat_bounds: caches re-derived when their inputs change), bit for bit: every P_abs, slope
   and committed output, so the kernel/reference parity can stay bit for bit.
2. The Airy path (cached cos/sin of the reference phase rotated by the small temperature-dependent angle, one
   reciprocal) against the closed forms it replaces (mrm.plant.optics._airy_line through dpabs_dlambda_res_nb and
   ring_lines_nb): equal up to floating-point rounding (the absolute phase phi ~ 200 rad carries ~3e-14 rad of
   rounding, which a resonance with finesse ~30 turns into ~1e-12 relative in P_abs).
The rows come from the NumPy reference engine (mrm.sim.kernel_ref), so the test does not need the compiled kernel.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mrm import params
from mrm.control.api import ControllerSpec
from mrm.plant import optics as op
from mrm.plant import phase as ph
from mrm.plant.plant import Plant
from mrm.scenarios import registry as scen_registry
from mrm.sim import kernel as K
from mrm.sim.kernel_ref import ReferenceEngine


def _reference(ovr=(), B=2, n=60):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26", overrides=["sim.reset_ramp_s=2e-5", *ovr])
        pp = params.realize(cfg, B, seeds=7)
    sc = scen_registry.from_config(cfg)
    spec = ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)
    plant = Plant(pp, sc, spec, noise_seed=3, scenario_seed=5, histogram=False)
    ref = ReferenceEngine(plant, oracle=False)
    ref.run(n)
    return plant, ref


def _outputs(af, ai):
    orl, orr, ol = op._sh_outputs(af, ai)
    return orl.copy(), orr.copy(), ol.copy()


@pytest.mark.parametrize("ovr", [(), ('fidelity.backscatter="on"', "fidelity.optics_model=tcmt"),
                                 ('fidelity.backscatter="on"',)],
                         ids=["airy_nrz", "tcmt_backscatter", "exact_backscatter"])
def test_engine_adapter_is_bit_identical(ovr):
    plant, ref = _reference(ovr)
    order = np.asarray(plant.P.bus_order)
    for b in range(plant.B):
        ai = np.ascontiguousarray(ref.args_i[b])
        ai_slope = ai.copy()
        ai_slope[op.SH_SLOPE] = 1                           # the optics adapter writes the slope outputs too
        a_new = np.ascontiguousarray(ref.args_f[b]).copy()
        a_old = a_new.copy()
        lam = np.asarray(plant.pp.optics.lambda_line_m, float)[b] if np.ndim(plant.pp.optics.lambda_line_m) > 1 \
            else np.asarray(plant.pp.optics.lambda_line_m, float)
        assert K.eng_lines_changed(a_new, ai, np.ascontiguousarray(lam))
        assert not K.eng_lines_changed(a_new, ai, np.ascontiguousarray(lam))
        K.eng_precompute(a_new, ai)
        o_fast = int(ai[op.SH_O_FF])
        airy = not ovr                                       # exact ring without backscatter: the hoisted path
        assert np.all(a_new[o_fast:o_fast + plant.N] == (1.0 if airy else 0.0))
        # bounds
        N = plant.N
        bn = [np.zeros(N) for _ in range(3)]
        bo = [np.zeros(N) for _ in range(3)]
        op.sh_bounds(a_new, ai, 3.0, *bn)
        op.selfheat_bounds(a_old, ai, 3.0, *bo)
        for x, y in zip(bn, bo):
            np.testing.assert_array_equal(x, y)
        # one walk along the bus: several trial temperatures per ring, commit at the last (reused) and, for every
        # other ring, at a temperature not evaluated last (the fallback path)
        t_ring = ref.t_ring[b]
        for q, k in enumerate(order):
            ts = [t_ring[k] - 0.3, t_ring[k], t_ring[k] + 1e-3, t_ring[k] + 0.7]
            for t in ts:
                pn = K.eng_pabs(float(t), int(k), a_new, ai)
                po = op.selfheat_pabs(float(t), int(k), a_old, ai_slope)
                assert pn[0] == po[0] and pn[1] == po[1], (b, k, t, pn, po)
            tc = float(ts[-1]) if q % 2 == 0 else float(t_ring[k] + 0.2)
            K.eng_commit(tc, int(k), a_new, ai)
            op.selfheat_commit(tc, int(k), a_old, ai_slope if q % 2 == 0 else ai)
            on, oo = _outputs(a_new, ai), _outputs(a_old, ai)
            for x, y in zip(on, oo):
                np.testing.assert_array_equal(x, y, err_msg=f"world {b} ring {k} commit {q}")
            np.testing.assert_array_equal(a_new[ai[op.SH_O_PCUR]:ai[op.SH_O_PCUR] + plant.L],
                                          a_old[ai[op.SH_O_PCUR]:ai[op.SH_O_PCUR] + plant.L])


def test_ring_optics_row_split():
    """phase.ring_optics_row = ring_optics_levels_row + coupling_row (the kernel calls coupling_row only when the
    line wavelengths change)."""
    plant, ref = _reference(n=5)
    phs = ph.PhaseParams.from_plant_params(plant.pp)
    N, L = plant.N, plant.L
    lam = np.linspace(1.3026e-6, 1.3186e-6, L)
    rng = np.random.default_rng(1)
    t = 300.0 + rng.uniform(0, 60, N)
    th = t + 5.0
    v0 = rng.uniform(0.5, 2.0, N)
    v1 = v0 + 0.8
    outs = [np.zeros((N, L)) for _ in range(9)] + [np.zeros((N, 2)) for _ in range(3)] + [np.zeros(N, np.int8)]
    outs2 = [np.zeros((N, L)) for _ in range(9)] + [np.zeros((N, 2)) for _ in range(3)] + [np.zeros(N, np.int8)]
    r = phs.table_row(0)
    args = (phs.ring[0], lam, t, th, v0, v1, phs.grid, phs.dneff[r], phs.alpha_fca[r], phs.capacitance[r],
            phs.r_series[r])
    ph.ring_optics_row(*args, *outs)
    o = outs2
    ph.ring_optics_levels_row(*args, o[0], o[1], o[2], o[3], o[4], o[5], o[8], o[9], o[10], o[11], o[12])
    ph.coupling_row(phs.ring[0], lam, o[6], o[7])
    for x, y in zip(outs, outs2):
        np.testing.assert_array_equal(x, y)


def _old_pabs(t, k, af, ai):
    """P_abs and dP_abs/dT of ring k at t from the closed forms (_airy_line via dpabs_dlambda_res_nb), on a copy."""
    af = af.copy()
    p_in, pcur, rg, nb, rb, cw, fw = op._sh_core(af, ai)
    n = ai[op.SH_N]
    dlam_dt = op._sh_prepare(t, k, ai[op.SH_HDR + n + k], p_in, pcur, rg, nb)
    p, sl = op.dpabs_dlambda_res_nb(k, pcur, rg[op.SHR_P1, k], nb[op.SHN_PHI0], nb[op.SHN_PHI1], nb[op.SHN_A0],
                                    nb[op.SHN_A1], nb[op.SHN_E0], nb[op.SHN_E1], nb[op.SHN_PSI], nb[op.SHN_K2T],
                                    nb[op.SHN_K2D], nb[op.SHN_TRT], rb, nb[op.SHN_DPL], ai[op.SH_TWO] != 0, False,
                                    ai[op.SH_MORD], cw, fw)
    return p, sl * dlam_dt


def _old_outputs(t, k, af, ai):
    af = af.copy()
    p_in, pcur, rg, nb, rb, cw, fw = op._sh_core(af, ai)
    orl, orr, ol = op._sh_outputs(af, ai)
    n = ai[op.SH_N]
    op._sh_prepare(t, k, ai[op.SH_HDR + n + k], p_in, pcur, rg, nb)
    op.ring_lines_nb(k, pcur, rg[op.SHR_P1, k], nb[op.SHN_PHI0], nb[op.SHN_PHI1], nb[op.SHN_A0], nb[op.SHN_A1],
                     nb[op.SHN_E0], nb[op.SHN_E1], nb[op.SHN_PSI], nb[op.SHN_K2T], nb[op.SHN_K2D], nb[op.SHN_TRT],
                     rb, nb[op.SHN_DPL], ai[op.SH_TWO] != 0, False, ai[op.SH_MORD], True, orl, orr, pcur.copy(), cw,
                     fw)
    return orl[:, k].copy(), orr[:, k].copy()


def test_airy_path_equals_the_closed_forms_to_rounding():
    """The hoisted Airy path vs _airy_line, across the trial temperatures of a solve (small rotation) and far
    beyond it (the libm fallback for |b| > ROT_MAX), every ring and both worlds (far-line hold off: the exact path)."""
    plant, ref = _reference()
    worst_p = worst_d = worst_o = 0.0
    for b in range(plant.B):
        ai = np.ascontiguousarray(ref.args_i[b]).copy()
        ai[op.SH_FARHOLD] = 0
        af = np.ascontiguousarray(ref.args_f[b]).copy()
        K.eng_precompute(af, ai)
        for k in range(plant.N):
            for dt in (0.0, 1e-6, 1e-3, 0.03, 0.4, 2.5, -1.7):
                t = float(ref.t_ring[b, k] + dt)
                # the walk input of ring k: use the bus input for every ring (pos 0 resets it; others read p_cur)
                pn = op.sh_fast_pabs(t, int(k), af, ai)
                po = _old_pabs(t, int(k), af, ai)
                worst_p = max(worst_p, abs(pn[0] - po[0]) / abs(po[0]))
                worst_d = max(worst_d, abs(pn[1] - po[1]) / max(abs(po[1]), 1e-30 + 1e-6 * abs(po[0])))
                orl_o, orr_o = _old_outputs(t, int(k), af, ai)
                a2 = af.copy()
                op.sh_fast_outputs(int(k), a2, ai)
                orl_n, orr_n = op._sh_outputs(a2, ai)[0][:, k], op._sh_outputs(a2, ai)[1][:, k]
                for f in (op.RL_PIN, op.RL_THRU, op.RL_DROP, op.RL_LOSS, op.RL_ABS, op.RL_UCW, op.RL_THRU1,
                          op.RL_THRU0, op.RL_DROP1, op.RL_DROP0):
                    scale = np.abs(orl_o[op.RL_PIN]).max()
                    worst_o = max(worst_o, np.abs(orl_n[f] - orl_o[f]).max() / scale)
    print(f"\nAiry path vs closed forms: P_abs {worst_p:.2e}, slope {worst_d:.2e}, outputs {worst_o:.2e} (relative)")
    assert worst_p < 1e-11, worst_p
    assert worst_d < 1e-10, worst_d
    assert worst_o < 1e-12, worst_o


def test_incremental_phase_cache_stays_at_rounding_level():
    """The cached cos/sin of the reference phase is updated incrementally (rotated by each change of the phase,
    re-derived with libm every ROT_REFRESH updates or after a jump): over 3 x ROT_REFRESH substeps of a random phase
    walk (with jumps) it stays within 2e-14 of libm cos/sin of the same phase, at both levels."""
    plant, ref = _reference(n=5)
    ai = np.ascontiguousarray(ref.args_i[0]).copy()
    ai[op.SH_FARHOLD] = 0                                    # every entry is then updated each substep
    af = np.ascontiguousarray(ref.args_f[0]).copy()
    v = op.selfheat_views(af, ai)
    phi0, phi1, psi = v[4], v[5], v[12]
    rng = np.random.default_rng(3)
    K.eng_precompute(af, ai)
    o_fc = int(ai[op.SH_O_FC])
    N, L = plant.N, plant.L
    worst = 0.0
    for it in range(3 * op.ROT_REFRESH):
        step = rng.normal(0.0, 3e-3, (N, L))
        if it % 997 == 0:
            step += 0.2                                       # a jump beyond ROT_MAX: the libm path
        phi0 += step
        phi1 += step + rng.normal(0.0, 1e-4, (N, L))
        K.eng_precompute(af, ai)
        if it % 64 == 0 or it > 3 * op.ROT_REFRESH - 4:
            fc = af[o_fc:o_fc + 2 * N * L * op.N_FC].reshape(2, N, L, op.N_FC)
            for lvl, ph in ((0, phi0), (1, phi1)):
                worst = max(worst, np.abs(fc[lvl, :, :, op.FC_C] - np.cos(ph + psi)).max(),
                            np.abs(fc[lvl, :, :, op.FC_S] - np.sin(ph + psi)).max())
    print(f"\nincremental cos/sin cache vs libm: {worst:.2e}")
    assert worst < 2e-14, worst


@pytest.mark.parametrize("step", [1e-8, 1e-6])
def test_incremental_phase_cache_constant_step_drift(step):
    """A constant phase step per substep (a steady drift of the ring temperature or of the laser) repeats the same
    rounding at every incremental update, so the cached cos/sin drifts systematically instead of as a random walk.
    Derived bound over ROT_REFRESH updates: per update cos b is rounded (<= ulp/2; truncation < 1e-17) and the two
    products and the sum of the rotation round (<= 3 ulp), so <= ROT_REFRESH x 4 ulp(1) = 1.8e-12 of libm (measured
    ~3e-13 at 1e-8 to 1e-6 rad per update). At resonance that is <= 2 r 1.8e-12 / (1 - r)^2 ~ 7e-10 relative in P_abs
    (g26: (1 - r)^2 ~ 5e-3), ~5e-9 K at the ~4800 K/W self-heating gain: far below the 1e-7 K root tolerance."""
    plant, ref = _reference(n=5)
    ai = np.ascontiguousarray(ref.args_i[0]).copy()
    ai[op.SH_FARHOLD] = 0
    af = np.ascontiguousarray(ref.args_f[0]).copy()
    v = op.selfheat_views(af, ai)
    phi0, phi1, psi = v[4], v[5], v[12]
    N, L = plant.N, plant.L
    o_fc = int(ai[op.SH_O_FC])
    fc = af[o_fc:o_fc + 2 * N * L * op.N_FC].reshape(2, N, L, op.N_FC)
    fc[..., op.FC_CNT] = np.nan                               # the next update is a libm refresh, the rest incremental
    for _ in range(op.ROT_REFRESH):
        phi0 += step
        phi1 += step
        K.eng_precompute(af, ai)
    fc = af[o_fc:o_fc + 2 * N * L * op.N_FC].reshape(2, N, L, op.N_FC)
    assert np.all(fc[..., op.FC_CNT] == op.ROT_REFRESH - 1)
    worst = 0.0
    for lvl, ph in ((0, phi0), (1, phi1)):
        worst = max(worst, np.abs(fc[lvl, :, :, op.FC_C] - np.cos(ph + psi)).max(),
                    np.abs(fc[lvl, :, :, op.FC_S] - np.sin(ph + psi)).max())
    bound = op.ROT_REFRESH * 4 * 2.0 ** -53
    print(f"\nincremental cos/sin cache, constant step {step:g} rad: {worst:.2e} after {op.ROT_REFRESH - 1} updates "
          f"(bound {bound:.1e})")
    assert worst <= bound, worst


def test_airy_flag_follows_the_backscatter_row():
    """The optics adapter derives a ring's Airy-path flag from its current backscatter row: a caller that writes a
    backscatter coefficient into a packed world after an evaluation gets the general (backscatter) path, as the rows
    say, and the Airy path again when it clears it."""
    plant, ref = _reference(n=5)
    ai = np.ascontiguousarray(ref.args_i[0]).copy()
    af = np.ascontiguousarray(ref.args_f[0]).copy()
    o_ff = int(ai[op.SH_O_FF])
    k = int(np.asarray(plant.P.bus_order)[0])                 # first in bus order: its input is the bus input
    t = float(ref.t_ring[0, k] + 0.01)
    p_airy = op.selfheat_pabs(t, k, af, ai)
    assert af[o_ff + k] == 1.0
    v = op.selfheat_views(af, ai)
    v[20][k, :] = 0.05 + 0.02j
    p_bs = op.selfheat_pabs(t, k, af, ai)
    assert af[o_ff + k] == 0.0
    p_gen = _old_pabs(t, k, af, ai)                           # the general path's closed forms with this rb
    assert p_bs[0] == p_gen[0] and p_bs[1] == p_gen[1]
    assert p_bs[0] != p_airy[0]
    v[20][k, :] = 0.0
    assert op.selfheat_pabs(t, k, af, ai) == p_airy
    assert af[o_ff + k] == 1.0


@pytest.mark.parametrize("phase_step", [3e-5, 2e-4])
def test_far_line_hold_error_is_within_its_bound(phase_step):
    """optics_far_hold against the exact Airy path on real g26 rows over 2000 substeps of a random walk of the
    reference phases (per-substep steps of phase_step rad, as the bias noise and the ring temperature move them)
    and of the loss (a steps of 1e-6): at every substep P_abs, its slope and every output of every ring after a
    full walk along the bus. Bound (mrm.plant.optics module comment at FAR_U): P_abs <= 3e-9 W per ring; the
    through, drop and loss outputs <= 1e-6 of the ring's input power at that line."""
    plant, ref = _reference(n=60)
    ai_h = np.ascontiguousarray(ref.args_i[0]).copy()
    ai_h[op.SH_FARHOLD] = 1
    ai_x = ai_h.copy()
    ai_x[op.SH_FARHOLD] = 0
    af_h = np.ascontiguousarray(ref.args_f[0]).copy()
    af_x = af_h.copy()
    N, L = plant.N, plant.L
    order = np.asarray(plant.P.bus_order)
    rng = np.random.default_rng(11)
    worst_p = worst_o = 0.0
    held = 0
    for it in range(2000):
        dphi = rng.normal(0.0, phase_step, (N, 1))            # rigid in the lines of a ring, like a shift
        da = rng.normal(0.0, 1e-6, (N, 1))
        for af in (af_h, af_x):
            v = op.selfheat_views(af, ai_h)
            v[4][...] += dphi
            v[5][...] += dphi
            v[8][...] += da
            v[9][...] += da
        K.eng_precompute(af_h, ai_h)
        K.eng_precompute(af_x, ai_x)
        o_fh = int(ai_h[op.SH_O_FH])
        held += int((af_h[o_fh:o_fh + N * L * op.N_FH].reshape(N, L, op.N_FH)[:, :, op.FH_STATE] != 0).sum())
        t_ring = ref.t_ring[0] + rng.normal(0.0, 2e-3, N)    # trial temperatures within a typical solve
        for k in order:
            ph = op.sh_fast_pabs(float(t_ring[k]), int(k), af_h, ai_h)
            px = op.sh_fast_pabs(float(t_ring[k]), int(k), af_x, ai_x)
            worst_p = max(worst_p, abs(ph[0] - px[0]))
            op.sh_fast_outputs(int(k), af_h, ai_h)
            op.sh_fast_outputs(int(k), af_x, ai_x)
            orl_h = op._sh_outputs(af_h, ai_h)[0][:, k]
            orl_x = op._sh_outputs(af_x, ai_x)[0][:, k]
            pin = orl_x[op.RL_PIN]
            for f in (op.RL_THRU, op.RL_DROP, op.RL_LOSS, op.RL_ABS, op.RL_THRU1, op.RL_THRU0, op.RL_DROP1,
                      op.RL_DROP0):
                worst_o = max(worst_o, float(np.max(np.abs(orl_h[f] - orl_x[f]) / pin)))
            op.sh_pass_on(int(k), af_h, ai_h)
            op.sh_pass_on(int(k), af_x, ai_x)
            np.copyto(af_h[ai_h[op.SH_O_PCUR]:ai_h[op.SH_O_PCUR] + L], af_x[ai_x[op.SH_O_PCUR]:ai_x[op.SH_O_PCUR] + L])
    frac = held / (2000 * N * L)
    print(f"\nfar-line hold, phase steps {phase_step:g} rad: held {frac:.2f} of the lines, P_abs error {worst_p:.2e} W, "
          f"outputs {worst_o:.2e} of P_in")
    assert frac > 0.5                                         # the hold is in use (14 of 16 lines are far in g26)
    assert worst_p <= 3e-9, worst_p
    assert worst_o <= 1e-6, worst_o
