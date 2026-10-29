# check the general field adapter against analytic equations and explicit archived cache references.
from __future__ import annotations

from archive.optics import retired_cached_approximations as archived_optics

import warnings

import numpy as np
import pytest

from mrm import params
from mrm.control.api import ControllerSpec
from mrm.plant import optics
from mrm.plant import phase
from mrm.plant.plant import Plant
from mrm.scenarios import registry as scenario_registry
from mrm.sim import kernel
from mrm.sim.kernel_ref import ReferenceEngine


def _run_reference_engine(ovr=(), B=2, n=60):
    """The g26 plant (B worlds, a 20 us reset ramp) after n ticks of the NumPy reference engine. ovr: extra
    config overrides. Returns (plant, reference engine)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26", overrides=["sim.reset_ramp_s=2e-5", *ovr])
        pp = params.realize(cfg, B, seeds=7)
    sc = scenario_registry.from_config(cfg)
    spec = ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)
    plant = Plant(pp, sc, spec, noise_seed=3, scenario_seed=5, histogram=False)
    ref = ReferenceEngine(plant, oracle=False)
    ref.run(n)
    return plant, ref


def _copy_outputs(af, ai):
    """Copies of the committed outputs of one packed world: (per ring and line, per ring, per line)."""
    orl, orr, ol = optics._sh_outputs(af, ai)
    return orl.copy(), orr.copy(), ol.copy()


@pytest.mark.parametrize("ovr", [(), ('fidelity.backscatter="on"',)], ids=["airy_nrz", "exact_backscatter"])
def test_engine_adapter_is_bit_identical(ovr):
    plant, ref = _run_reference_engine(ovr)
    order = np.asarray(plant.P.bus_order)
    for b in range(plant.B):
        ai = np.ascontiguousarray(ref.packed_i[b])
        ai_slope = ai.copy()
        ai_slope[optics.SELF_HEATING_HEADER_INDEX_COMMIT_OPTICAL_DERIVATIVES_FLAG] = (
            1  # the optics adapter writes the slope outputs too
        )
        a_new = np.ascontiguousarray(ref.packed_f[b]).copy()
        a_old = a_new.copy()
        lam = (
            np.asarray(plant.pp.optics.lambda_line_m, float)[b]
            if np.ndim(plant.pp.optics.lambda_line_m) > 1
            else np.asarray(plant.pp.optics.lambda_line_m, float)
        )
        assert kernel.cache_update_laser_wavelengths(a_new, ai, np.ascontiguousarray(lam))
        assert not kernel.cache_update_laser_wavelengths(a_new, ai, np.ascontiguousarray(lam))
        # bounds
        N = plant.N
        bn = [np.zeros(N) for _ in range(3)]
        bo = [np.zeros(N) for _ in range(3)]
        optics.sh_bounds(a_new, ai, 3.0, *bn)
        optics.selfheat_bounds(a_old, ai, 3.0, *bo)
        for x, y in zip(bn, bo):
            np.testing.assert_array_equal(x, y)
        # one walk along the bus: several trial temperatures per ring, commit at the last (reused) and, for every
        # other ring, at a temperature not evaluated last (the fallback path)
        t_ring = ref.t_ring[b]
        for q, k in enumerate(order):
            ts = [t_ring[k] - 0.3, t_ring[k], t_ring[k] + 1e-3, t_ring[k] + 0.7]
            for t in ts:
                pn = kernel.eng_pabs(float(t), int(k), a_new, ai)
                po = optics.selfheat_pabs(float(t), int(k), a_old, ai_slope)
                assert pn[0] == po[0] and pn[1] == po[1], (b, k, t, pn, po)
            tc = float(ts[-1]) if q % 2 == 0 else float(t_ring[k] + 0.2)
            kernel.eng_commit(tc, int(k), a_new, ai, True)
            optics.selfheat_commit(tc, int(k), a_old, ai_slope if q % 2 == 0 else ai)
            on, oo = _copy_outputs(a_new, ai), _copy_outputs(a_old, ai)
            for x, y in zip(on, oo):
                np.testing.assert_array_equal(x, y, err_msg=f"world {b} ring {k} commit {q}")
            np.testing.assert_array_equal(
                a_new[
                    ai[optics.SELF_HEATING_HEADER_INDEX_CURRENT_RING_INPUT_POWER_OFFSET] : ai[
                        optics.SELF_HEATING_HEADER_INDEX_CURRENT_RING_INPUT_POWER_OFFSET
                    ]
                    + plant.L
                ],
                a_old[
                    ai[optics.SELF_HEATING_HEADER_INDEX_CURRENT_RING_INPUT_POWER_OFFSET] : ai[
                        optics.SELF_HEATING_HEADER_INDEX_CURRENT_RING_INPUT_POWER_OFFSET
                    ]
                    + plant.L
                ],
            )



def test_ring_optics_row_split():
    """phase.ring_optics_row = ring_optics_levels_row + coupling_row (the kernel calls coupling_row only when the
    line wavelengths change)."""
    plant, ref = _run_reference_engine(n=5)
    phase_params = phase.PhaseParams.from_plant_params(plant.pp)
    N, L = plant.N, plant.L
    lam = np.linspace(1.3026e-6, 1.3186e-6, L)
    rng = np.random.default_rng(1)
    t = 300.0 + rng.uniform(0, 60, N)
    t_heater = t + 5.0
    v0 = rng.uniform(0.5, 2.0, N)
    v1 = v0 + 0.8
    outs = [np.zeros((N, L)) for _ in range(9)] + [np.zeros((N, 2)) for _ in range(3)] + [np.zeros(N, np.int8)]
    outs2 = [np.zeros((N, L)) for _ in range(9)] + [np.zeros((N, 2)) for _ in range(3)] + [np.zeros(N, np.int8)]
    r = phase_params.table_row(0)
    args = (phase_params.ring[0], lam, t, t_heater, v0, v1, phase_params.grid, phase_params.dneff[r],
            phase_params.alpha_fca[r], phase_params.capacitance[r], phase_params.r_series[r])
    phase.ring_optics_row(*args, *outs)
    o = outs2
    phase.ring_optics_levels_row(*args, o[0], o[1], o[2], o[3], o[4], o[5], o[8], o[9], o[10], o[11], o[12])
    phase.coupling_row(phase_params.ring[0], lam, o[6], o[7])
    for x, y in zip(outs, outs2):
        np.testing.assert_array_equal(x, y)


def _closed_form_pabs(t, k, af, ai):
    """P_abs and dP_abs/dT of ring k at t from the closed forms (_airy_line via dpabs_dlambda_res_nb), on a copy."""
    af = af.copy()
    p_in, pcur, rg, nb, rb, cw, fw = optics._sh_core(af, ai)  # rg: ring rows, nb: (ring, line) rows
    n = ai[optics.SELF_HEATING_HEADER_INDEX_RING_COUNT]
    dlam_dt = optics._sh_prepare(
        t, k, ai[optics.SELF_HEATING_HEADER_FIXED_FIELD_COUNT + n + k], p_in, pcur, rg, nb
    )  # SELF_HEATING_HEADER_FIXED_FIELD_COUNT + n + k: bus position
    p, sl = optics.dpabs_dlambda_res_nb(
        k,
        pcur,
        rg[optics.SELF_HEATING_RING_ROW_PROBABILITY_OF_ONE, k],
        nb[optics.SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_CURRENT_ROUND_TRIP_PHASE],
        nb[optics.SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_CURRENT_ROUND_TRIP_PHASE],
        nb[optics.SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION],
        nb[optics.SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION],
        nb[optics.SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
        nb[optics.SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
        nb[optics.SELF_HEATING_RING_LINE_ROW_COUPLER_PHASE],
        nb[optics.SELF_HEATING_RING_LINE_ROW_THROUGH_COUPLER_POWER_COUPLING],
        nb[optics.SELF_HEATING_RING_LINE_ROW_DROP_COUPLER_POWER_COUPLING],
        nb[optics.SELF_HEATING_RING_LINE_ROW_ROUND_TRIP_GROUP_DELAY],
        rb,
        nb[optics.SELF_HEATING_RING_LINE_ROW_PHASE_RESONANCE_WAVELENGTH_DERIVATIVE],
        ai[optics.SELF_HEATING_HEADER_INDEX_TWO_DATA_LEVELS_FLAG] != 0,
        False,
        ai[optics.SELF_HEATING_HEADER_INDEX_RESERVED_MODE_ORDER_COUNT],
        cw,
        fw,
    )
    return p, sl * dlam_dt


def _closed_form_outputs(t, k, af, ai):
    """The per-(ring, line) and per-ring outputs of ring k at t from the closed forms (evaluate_ring_optical_outputs), on a copy."""
    af = af.copy()
    p_in, pcur, rg, nb, rb, cw, fw = optics._sh_core(af, ai)
    orl, orr, ol = optics._sh_outputs(af, ai)
    n = ai[optics.SELF_HEATING_HEADER_INDEX_RING_COUNT]
    optics._sh_prepare(t, k, ai[optics.SELF_HEATING_HEADER_FIXED_FIELD_COUNT + n + k], p_in, pcur, rg, nb)
    optics.evaluate_ring_optical_outputs(
        k,
        pcur,
        rg[optics.SELF_HEATING_RING_ROW_PROBABILITY_OF_ONE, k],
        nb[optics.SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_CURRENT_ROUND_TRIP_PHASE],
        nb[optics.SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_CURRENT_ROUND_TRIP_PHASE],
        nb[optics.SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION],
        nb[optics.SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION],
        nb[optics.SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
        nb[optics.SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
        nb[optics.SELF_HEATING_RING_LINE_ROW_COUPLER_PHASE],
        nb[optics.SELF_HEATING_RING_LINE_ROW_THROUGH_COUPLER_POWER_COUPLING],
        nb[optics.SELF_HEATING_RING_LINE_ROW_DROP_COUPLER_POWER_COUPLING],
        nb[optics.SELF_HEATING_RING_LINE_ROW_ROUND_TRIP_GROUP_DELAY],
        rb,
        nb[optics.SELF_HEATING_RING_LINE_ROW_PHASE_RESONANCE_WAVELENGTH_DERIVATIVE],
        ai[optics.SELF_HEATING_HEADER_INDEX_TWO_DATA_LEVELS_FLAG] != 0,
        False,
        ai[optics.SELF_HEATING_HEADER_INDEX_RESERVED_MODE_ORDER_COUNT],
        True,
        orl,
        orr,
        pcur.copy(),
        cw,
        fw,
    )
    return orl[:, k].copy(), orr[:, k].copy()


def test_airy_path_equals_the_closed_forms_to_rounding():
    """The hoisted Airy path vs _airy_line, across the trial temperatures of a solve (small rotation) and far
    beyond it (the libm fallback for |b| > PHASE_ROTATION_POLYNOMIAL_MAX_ANGLE_RAD), every ring and both worlds (far-line hold off: the exact path)."""
    optics = archived_optics
    plant, ref = _run_reference_engine()
    worst_p = worst_d = worst_o = 0.0
    for b in range(plant.B):
        ai = np.ascontiguousarray(ref.packed_i[b]).copy()
        ai[optics.SELF_HEATING_HEADER_INDEX_FAR_LINE_HOLD_ENABLED_FLAG] = 0
        af = np.ascontiguousarray(ref.packed_f[b]).copy()
        optics.sh_fast_precompute(af, ai)
        for k in range(plant.N):
            for dt in (0.0, 1e-6, 1e-3, 0.03, 0.4, 2.5, -1.7):
                t = float(ref.t_ring[b, k] + dt)
                # the walk input of ring k: use the bus input for every ring (pos 0 resets it; others read p_cur)
                pn = optics.sh_fast_pabs(t, int(k), af, ai)
                po = _closed_form_pabs(t, int(k), af, ai)
                worst_p = max(worst_p, abs(pn[0] - po[0]) / abs(po[0]))
                worst_d = max(worst_d, abs(pn[1] - po[1]) / max(abs(po[1]), 1e-30 + 1e-6 * abs(po[0])))
                orl_o, orr_o = _closed_form_outputs(t, int(k), af, ai)
                a2 = af.copy()
                optics.cache_write_optical_outputs(int(k), a2, ai)
                orl_n, orr_n = optics._sh_outputs(a2, ai)[0][:, k], optics._sh_outputs(a2, ai)[1][:, k]
                for f in (optics.RING_LINE_OUTPUT_ROW_INPUT_POWER, optics.RING_LINE_OUTPUT_ROW_THROUGH_POWER, optics.RING_LINE_OUTPUT_ROW_CLOCKWISE_DROP_POWER, optics.RING_LINE_OUTPUT_ROW_INTRINSIC_LOSS_POWER, optics.RING_LINE_OUTPUT_ROW_ABSORBED_POWER, optics.RING_LINE_OUTPUT_ROW_CLOCKWISE_STORED_ENERGY,
                          optics.RING_LINE_OUTPUT_ROW_LEVEL_ONE_THROUGH_POWER, optics.RING_LINE_OUTPUT_ROW_LEVEL_ZERO_THROUGH_POWER, optics.RING_LINE_OUTPUT_ROW_LEVEL_ONE_CLOCKWISE_DROP_POWER, optics.RING_LINE_OUTPUT_ROW_LEVEL_ZERO_CLOCKWISE_DROP_POWER):
                    scale = np.abs(orl_o[optics.RING_LINE_OUTPUT_ROW_INPUT_POWER]).max()
                    worst_o = max(worst_o, np.abs(orl_n[f] - orl_o[f]).max() / scale)
    print(f"\nAiry path vs closed forms: P_abs {worst_p:.2e}, slope {worst_d:.2e}, outputs {worst_o:.2e} (relative)")
    assert worst_p < 1e-11, worst_p
    assert worst_d < 1e-10, worst_d
    assert worst_o < 1e-12, worst_o



def test_engine_solve_reports_a_nan_gain_as_failed():
    """The engine copy of the ring solve returns t_other with STATUS_NAN_GAIN for a NaN gain (it reported converged),
    and keeps STATUS_CONVERGED for g = 0."""
    plant, ref = _run_reference_engine(B=1, n=5)
    ai = np.ascontiguousarray(ref.packed_i[0])
    af = np.ascontiguousarray(ref.packed_f[0]).copy()
    t_other = float(ref.t_ring[0][0])
    p_ref = kernel.eng_pabs(t_other, 0, af, ai)[0]
    for g, status in ((np.nan, kernel.STATUS_NAN_GAIN), (0.0, kernel.STATUS_CONVERGED)):
        t, p, d, it, st = kernel._solve_ring_eng(0, af, ai, t_other, g, t_other, 1e-3, 1.0, 10, 1e-9)
        assert t == t_other and p == p_ref and it == 0 and st == status, (g, st)



def test_incremental_phase_cache_stays_at_rounding_level():
    """The cached cos/sin of the reference phase is updated incrementally (rotated by each change of the phase,
    re-derived with libm every PHASE_ROTATION_EXACT_REFRESH_INTERVAL updates or after a jump): over 3 x PHASE_ROTATION_EXACT_REFRESH_INTERVAL substeps of a random phase
    walk (with jumps) it stays within 2e-14 of libm cos/sin of the same phase, at both levels."""
    optics = archived_optics
    plant, ref = _run_reference_engine(n=5)
    ai = np.ascontiguousarray(ref.packed_i[0]).copy()
    ai[optics.SELF_HEATING_HEADER_INDEX_FAR_LINE_HOLD_ENABLED_FLAG] = (
        0  # every entry is then updated each substep
    )
    af = np.ascontiguousarray(ref.packed_f[0]).copy()
    v = optics.selfheat_views(af, ai)
    phi0, phi1, psi = (
        v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_REFERENCE_ROUND_TRIP_PHASE],
        v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ONE_REFERENCE_ROUND_TRIP_PHASE],
        v[optics.SELF_HEATING_VIEW_INDEX_COUPLER_PHASE],
    )
    rng = np.random.default_rng(3)
    optics.sh_fast_precompute(af, ai)
    o_fc = int(ai[optics.SELF_HEATING_HEADER_INDEX_AIRY_CACHE_OFFSET])
    N, L = plant.N, plant.L
    worst = 0.0
    for it in range(3 * optics.PHASE_ROTATION_EXACT_REFRESH_INTERVAL):
        step = rng.normal(0.0, 3e-3, (N, L))
        if it % 997 == 0:
            step += 0.2  # a jump beyond PHASE_ROTATION_POLYNOMIAL_MAX_ANGLE_RAD: the libm path
        phi0 += step
        phi1 += step + rng.normal(0.0, 1e-4, (N, L))
        optics.sh_fast_precompute(af, ai)
        if it % 64 == 0 or it > 3 * optics.PHASE_ROTATION_EXACT_REFRESH_INTERVAL - 4:
            fc = af[o_fc : o_fc + 2 * N * L * optics.AIRY_CACHE_FIELD_COUNT].reshape(
                2, N, L, optics.AIRY_CACHE_FIELD_COUNT
            )
            for lvl, phi in ((0, phi0), (1, phi1)):
                worst = max(
                    worst,
                    np.abs(
                        fc[lvl, :, :, optics.AIRY_CACHE_FIELD_REFERENCE_TOTAL_PHASE_COSINE]
                        - np.cos(phi + psi)
                    ).max(),
                    np.abs(
                        fc[lvl, :, :, optics.AIRY_CACHE_FIELD_REFERENCE_TOTAL_PHASE_SINE] - np.sin(phi + psi)
                    ).max(),
                )
    print(f"\nincremental cos/sin cache vs libm: {worst:.2e}")
    assert worst < 2e-14, worst



@pytest.mark.parametrize("step", [1e-8, 1e-6])
def test_incremental_phase_cache_constant_step_drift(step):
    """A constant phase step per substep (a steady drift of the ring temperature or of the laser) repeats the same
    rounding at every incremental update, so the cached cos/sin drifts systematically instead of as a random walk.
    Derived bound over PHASE_ROTATION_EXACT_REFRESH_INTERVAL updates: per update cos b is rounded (<= ulp/2; truncation < 1e-17) and the two
    products and the sum of the rotation round (<= 3 ulp), so <= PHASE_ROTATION_EXACT_REFRESH_INTERVAL x 4 ulp(1) = 1.8e-12 of libm (measured
    ~3e-13 at 1e-8 to 1e-6 rad per update). At resonance that is <= 2 r 1.8e-12 / (1 - r)^2 ~ 7e-10 relative in P_abs
    (g26: (1 - r)^2 ~ 5e-3), ~5e-9 K at the ~4800 K/W self-heating gain: far below the 1e-7 K root tolerance."""
    optics = archived_optics
    plant, ref = _run_reference_engine(n=5)
    ai = np.ascontiguousarray(ref.packed_i[0]).copy()
    ai[optics.SELF_HEATING_HEADER_INDEX_FAR_LINE_HOLD_ENABLED_FLAG] = 0
    af = np.ascontiguousarray(ref.packed_f[0]).copy()
    v = optics.selfheat_views(af, ai)
    phi0, phi1, psi = v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_REFERENCE_ROUND_TRIP_PHASE], v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ONE_REFERENCE_ROUND_TRIP_PHASE], v[optics.SELF_HEATING_VIEW_INDEX_COUPLER_PHASE]
    N, L = plant.N, plant.L
    o_fc = int(ai[optics.SELF_HEATING_HEADER_INDEX_AIRY_CACHE_OFFSET])
    fc = af[o_fc:o_fc + 2 * N * L * optics.AIRY_CACHE_FIELD_COUNT].reshape(2, N, L, optics.AIRY_CACHE_FIELD_COUNT)
    fc[..., optics.AIRY_CACHE_FIELD_PHASE_ROTATION_UPDATE_COUNT] = np.nan                           # the next update is a libm refresh, the rest incremental
    for _ in range(optics.PHASE_ROTATION_EXACT_REFRESH_INTERVAL):
        phi0 += step
        phi1 += step
        optics.sh_fast_precompute(af, ai)
    fc = af[o_fc:o_fc + 2 * N * L * optics.AIRY_CACHE_FIELD_COUNT].reshape(2, N, L, optics.AIRY_CACHE_FIELD_COUNT)
    assert np.all(fc[..., optics.AIRY_CACHE_FIELD_PHASE_ROTATION_UPDATE_COUNT] == optics.PHASE_ROTATION_EXACT_REFRESH_INTERVAL - 1)
    worst = 0.0
    for lvl, phi in ((0, phi0), (1, phi1)):
        worst = max(worst, np.abs(fc[lvl, :, :, optics.AIRY_CACHE_FIELD_REFERENCE_TOTAL_PHASE_COSINE] - np.cos(phi + psi)).max(),
                    np.abs(fc[lvl, :, :, optics.AIRY_CACHE_FIELD_REFERENCE_TOTAL_PHASE_SINE] - np.sin(phi + psi)).max())
    bound = optics.PHASE_ROTATION_EXACT_REFRESH_INTERVAL * 4 * 2.0 ** -53
    print(f"\nincremental cos/sin cache, constant step {step:g} rad: {worst:.2e} after {optics.PHASE_ROTATION_EXACT_REFRESH_INTERVAL - 1} "
          f"updates (bound {bound:.1e})")
    assert worst <= bound, worst



def test_airy_flag_follows_the_backscatter_row():
    """The optics adapter derives a ring's Airy-path flag from its current backscatter row: a caller that writes a
    backscatter coefficient into a packed world after an evaluation gets the general (backscatter) path, as the rows
    say, and the Airy path again when it clears it."""
    optics = archived_optics
    plant, ref = _run_reference_engine(n=5)
    ai = np.ascontiguousarray(ref.packed_i[0]).copy()
    af = np.ascontiguousarray(ref.packed_f[0]).copy()
    o_ff = int(ai[optics.SELF_HEATING_HEADER_INDEX_NO_BACKSCATTER_FLAGS_OFFSET])
    k = int(np.asarray(plant.P.bus_order)[0])  # first in bus order: its input is the bus input
    t = float(ref.t_ring[0, k] + 0.01)
    p_airy = optics.selfheat_pabs(t, k, af, ai)
    assert af[o_ff + k] == 1.0
    v = optics.selfheat_views(af, ai)
    v[optics.SELF_HEATING_VIEW_INDEX_ROUND_TRIP_BACKSCATTER_COEFFICIENTS][k, :] = 0.05 + 0.02j
    p_bs = optics.selfheat_pabs(t, k, af, ai)
    assert af[o_ff + k] == 0.0
    p_gen = _closed_form_pabs(t, k, af, ai)  # the general path's closed forms with this rb
    assert p_bs[0] == p_gen[0] and p_bs[1] == p_gen[1]
    assert p_bs[0] != p_airy[0]
    v[optics.SELF_HEATING_VIEW_INDEX_ROUND_TRIP_BACKSCATTER_COEFFICIENTS][k, :] = 0.0
    assert optics.selfheat_pabs(t, k, af, ai) == p_airy
    assert af[o_ff + k] == 1.0



@pytest.mark.parametrize("phase_step", [3e-5, 2e-4])
def test_far_line_hold_error_is_within_its_bound(phase_step):
    """optics_far_hold against the exact Airy path on real g26 rows, over 2000 substeps of a random walk of the
    reference phases (steps of phase_step rad per substep, as the bias noise and the ring temperature move them)
    and of the round-trip amplitude (steps of 1e-6). At every substep the test compares P_abs, its slope and every
    output of every ring after a full walk along the bus. Bound (mrm.plant.optics module comment at FAR_LINE_HOLD_MIN_DETUNING_IN_HALF_WIDTHS):
    P_abs <= 3e-9 W per ring; the through, drop and loss outputs <= 1e-6 of the ring's input power at that line."""
    optics = archived_optics
    plant, ref = _run_reference_engine(n=60)
    ai_h = np.ascontiguousarray(ref.packed_i[0]).copy()
    ai_h[optics.SELF_HEATING_HEADER_INDEX_FAR_LINE_HOLD_ENABLED_FLAG] = 1
    ai_x = ai_h.copy()
    ai_x[optics.SELF_HEATING_HEADER_INDEX_FAR_LINE_HOLD_ENABLED_FLAG] = 0
    af_h = np.ascontiguousarray(ref.packed_f[0]).copy()
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
            v = optics.selfheat_views(af, ai_h)
            v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_REFERENCE_ROUND_TRIP_PHASE][...] += dphi
            v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ONE_REFERENCE_ROUND_TRIP_PHASE][...] += dphi
            v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION][...] += da
            v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION][...] += da
        optics.sh_fast_precompute(af_h, ai_h)
        optics.sh_fast_precompute(af_x, ai_x)
        o_fh = int(ai_h[optics.SELF_HEATING_HEADER_INDEX_FAR_LINE_CACHE_OFFSET])
        held += int((af_h[o_fh:o_fh + N * L * optics.FAR_LINE_CACHE_FIELD_COUNT].reshape(N, L, optics.FAR_LINE_CACHE_FIELD_COUNT)[:, :, optics.FAR_LINE_CACHE_FIELD_HOLD_ENABLED_FLAG] != 0)
                    .sum())
        t_ring = ref.t_ring[0] + rng.normal(0.0, 2e-3, N)    # trial temperatures within a typical solve
        for k in order:
            p_held = optics.sh_fast_pabs(float(t_ring[k]), int(k), af_h, ai_h)
            p_exact = optics.sh_fast_pabs(float(t_ring[k]), int(k), af_x, ai_x)
            worst_p = max(worst_p, abs(p_held[0] - p_exact[0]))
            optics.cache_write_optical_outputs(int(k), af_h, ai_h)
            optics.cache_write_optical_outputs(int(k), af_x, ai_x)
            orl_h = optics._sh_outputs(af_h, ai_h)[0][:, k]
            orl_x = optics._sh_outputs(af_x, ai_x)[0][:, k]
            pin = orl_x[optics.RING_LINE_OUTPUT_ROW_INPUT_POWER]
            for f in (optics.RING_LINE_OUTPUT_ROW_THROUGH_POWER, optics.RING_LINE_OUTPUT_ROW_CLOCKWISE_DROP_POWER, optics.RING_LINE_OUTPUT_ROW_INTRINSIC_LOSS_POWER, optics.RING_LINE_OUTPUT_ROW_ABSORBED_POWER, optics.RING_LINE_OUTPUT_ROW_LEVEL_ONE_THROUGH_POWER, optics.RING_LINE_OUTPUT_ROW_LEVEL_ZERO_THROUGH_POWER,
                      optics.RING_LINE_OUTPUT_ROW_LEVEL_ONE_CLOCKWISE_DROP_POWER, optics.RING_LINE_OUTPUT_ROW_LEVEL_ZERO_CLOCKWISE_DROP_POWER):
                worst_o = max(worst_o, float(np.max(np.abs(orl_h[f] - orl_x[f]) / pin)))
            optics.sh_pass_on(int(k), af_h, ai_h)
            optics.sh_pass_on(int(k), af_x, ai_x)
            np.copyto(af_h[ai_h[optics.SELF_HEATING_HEADER_INDEX_CURRENT_RING_INPUT_POWER_OFFSET]:ai_h[optics.SELF_HEATING_HEADER_INDEX_CURRENT_RING_INPUT_POWER_OFFSET] + L],
                      af_x[ai_x[optics.SELF_HEATING_HEADER_INDEX_CURRENT_RING_INPUT_POWER_OFFSET]:ai_x[optics.SELF_HEATING_HEADER_INDEX_CURRENT_RING_INPUT_POWER_OFFSET] + L])
    frac = held / (2000 * N * L)
    print(f"\nfar-line hold, phase steps {phase_step:g} rad: held {frac:.2f} of the lines, "
          f"P_abs error {worst_p:.2e} W, outputs {worst_o:.2e} of P_in")
    assert frac > 0.5                                         # the hold is in use (14 of 16 lines are far in g26)
    assert worst_p <= 3e-9, worst_p
    assert worst_o <= 1e-6, worst_o



def test_degenerate_row_raises_on_both_paths():
    """A ring without coupling and without loss (K^2 = 0, a = 1) at resonance has den = 1 - 2 r cos(phi) + r^2 = 0:
    the closed forms of the general path (_airy_line) and the Airy path (with or without the far-line hold) both raise
    ZeroDivisionError there, explicitly. They must not return inf / NaN, and the result must not depend on which
    caller's numba error model compiled and cached the function first. Every ring with r = t1 t2 a < 1 has
    den >= (1 - r)^2 > 0."""
    plant, ref = _run_reference_engine(n=5)
    for hold in (0,):
        ai = np.ascontiguousarray(ref.packed_i[0]).copy()
        ai[optics.SELF_HEATING_HEADER_INDEX_FAR_LINE_HOLD_ENABLED_FLAG] = hold
        af = np.ascontiguousarray(ref.packed_f[0]).copy()
        k = int(np.asarray(plant.P.bus_order)[0])
        v = optics.selfheat_views(af, ai)
        for row in (
            optics.SELF_HEATING_VIEW_INDEX_THROUGH_COUPLER_POWER_COUPLING,
            optics.SELF_HEATING_VIEW_INDEX_DROP_COUPLER_POWER_COUPLING,
        ):
            v[row][k, :] = 0.0
        for row in (
            optics.SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION,
            optics.SELF_HEATING_VIEW_INDEX_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION,
        ):
            v[row][k, :] = 1.0
        for row in (
            optics.SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_REFERENCE_ROUND_TRIP_PHASE,
            optics.SELF_HEATING_VIEW_INDEX_LEVEL_ONE_REFERENCE_ROUND_TRIP_PHASE,
            optics.SELF_HEATING_VIEW_INDEX_COUPLER_PHASE,
        ):  # cos = 1
            v[row][k, :] = 0.0
        t = float(
            v[optics.SELF_HEATING_VIEW_INDEX_REFERENCE_TEMPERATURE][k]
        )  # the reference temperature: no in-solve rotation
        with pytest.raises(ZeroDivisionError):
            optics.selfheat_pabs(t, k, af, ai)
        with pytest.raises(ZeroDivisionError):
            optics.selfheat_pabs_general(t, k, af.copy(), ai)



def test_general_adapter_functions_match_archived_airy_calculation():
    """The independent general-path adapter functions (selfheat_pabs_general / selfheat_commit_general /
    selfheat_bounds_general) against the archived Airy calculation on real
    g26 rows: P_abs to 1e-11 relative, its slope to 1e-11 of P_abs per K, every committed output to 1e-11 of its
    field's largest value, the bounds bit for bit (sh_bounds fuses the same expressions in the same order)."""
    plant, ref = _run_reference_engine(n=60)
    order = np.asarray(plant.P.bus_order)
    for b in range(plant.B):
        ai = np.ascontiguousarray(ref.packed_i[b]).copy()
        ai[optics.SELF_HEATING_HEADER_INDEX_COMMIT_OPTICAL_DERIVATIVES_FLAG] = 1
        a_air = np.ascontiguousarray(ref.packed_f[b]).copy()
        a_gen = a_air.copy()
        N = plant.N
        bn = [np.zeros(N) for _ in range(3)]
        bo = [np.zeros(N) for _ in range(3)]
        archived_optics.selfheat_bounds(a_air, ai, 3.0, *bn)
        optics.selfheat_bounds_general(a_gen, ai, 3.0, *bo)
        for x, y in zip(bn, bo):
            np.testing.assert_array_equal(x, y)
        t_ring = ref.t_ring[b]
        for q, k in enumerate(order):
            for t in (t_ring[k] - 0.2, t_ring[k], t_ring[k] + 1e-4, t_ring[k] + 0.5):
                pa = archived_optics.selfheat_pabs(float(t), int(k), a_air, ai)
                pg = optics.selfheat_pabs_general(float(t), int(k), a_gen, ai)
                np.testing.assert_allclose(pa[0], pg[0], rtol=1e-11, atol=0)
                # the slope sums terms of both signs: relative to |P_abs| per K
                np.testing.assert_allclose(pa[1], pg[1], rtol=1e-11, atol=1e-11 * abs(pg[0]))
            tc = float(t_ring[k] + 1e-4)
            archived_optics.selfheat_commit(tc, int(k), a_air, ai)
            optics.selfheat_commit_general(tc, int(k), a_gen, ai)
            for x, y in zip(_copy_outputs(a_air, ai), _copy_outputs(a_gen, ai)):
                scale = np.max(np.abs(y), axis=tuple(range(1, y.ndim)), keepdims=True)
                np.testing.assert_array_less(np.abs(x - y), np.broadcast_to(1e-11 * scale + 1e-300, x.shape))



# retired selectors must fail rather than silently choose a different calculation
@pytest.mark.parametrize("entry_point", ["plant", "packed", "runner", "raw_header", "reference"])
def test_retired_optics_path_cannot_be_selected(entry_point):
    if entry_point == "plant":
        with pytest.raises(TypeError, match="optics_far_hold"):
            Plant(None, optics_far_hold=True)
    elif entry_point == "packed":
        with pytest.raises(TypeError, match="far_hold"):
            optics.pack_selfheat_args([1e-3], [[0.1]], 0.1, 0.01, 0.95, 1e-12, 0.8, 1310e-9,
                                     300.0, 75e-12, far_hold=True)
    elif entry_point == "runner":
        from mrm.sim import runner
        with pytest.raises(ValueError, match="optics_far_hold is retired"):
            runner.run(None, optics_far_hold=True)
    elif entry_point == "reference":
        with pytest.raises(TypeError, match="airy"):
            ReferenceEngine(None, airy=True)
    else:
        af, ai = optics.pack_selfheat_args([1e-3], [[0.1]], 0.1, 0.01, 0.95, 1e-12, 0.8,
                                         1310e-9, 300.0, 75e-12)
        ai[optics.SELF_HEATING_HEADER_INDEX_FAR_LINE_HOLD_ENABLED_FLAG] = 1
        with pytest.raises(ValueError, match="far-line hold is retired"):
            optics.selfheat_pabs(300.0, 0, af, ai)
