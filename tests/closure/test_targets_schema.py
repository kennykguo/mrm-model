"""Closure tests of the frozen target files: calibration, validation (held out), bounds and excluded.

Read this first. The targets live in mrm/validation/targets/*.yaml. The split between the calibration targets (the
only ones a fit may read) and the held-out ones was frozen before any fitting, so that a later agreement is a real
test. These tests check that:

- the four files obey the schema and the split rules (no data point is both calibration and held-out data);
- every target id of the validation plan is present (CAL-1..14, VAL-1..8, BND-1..4, EXC-1..8);
- the numbers in the files agree with the research reports they cite (re-derived here: unit conversions, the
  rectified-sine aggressor, the heater-sweep differences, the BND-1 FSR windows);
- the tolerance semantics of ``targets_io.evaluate`` hold on synthetic predictions.

The research reports in docs/research/ are cited by number: report 01 = 01_lightmatter_public.md (the read-off of
the public Lightmatter data), 04 = 04_thermal_and_control.md, 06 = 06_textbook_audit.md and 07 =
07_design_critique.md (the validation plan, its section 4). LM26-P, YOLE and the like are source keys of
mrm/params/sources.yaml.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import re

import numpy as np
import pytest
import yaml

from mrm.validation import targets_io

C0 = 299_792_458.0          # speed of light, m/s (exact, SI definition)
LAMBDA0_M = 1310e-9         # O-band centre of the G26 grid, m (LM26-P section 2, printed)


@pytest.fixture(scope="module")
def sets():
    """{role: tuple of Targets} of all four target files."""
    return targets_io.load_all()


@pytest.fixture(scope="module")
def by_id(sets):
    """{target id: Target} over all roles."""
    out = {}
    for role in targets_io.ROLES:
        for t in sets[role]:
            out[t.id] = t
    return out


# ----------------------------------------------------------------------------------------------------- schema and split
@pytest.mark.parametrize("role", targets_io.ROLES)
def test_file_passes_schema(role):
    errs = targets_io.schema_errors(role, targets_io.read_raw(role))
    assert errs == [], "\n".join(errs)


def test_split_rules_hold(sets):
    errs = targets_io.split_errors(sets)
    assert errs == [], "\n".join(errs)


def test_contract_ids_of_07_section_4_present(sets):
    """Every target row of the validation plan (report 07, sections 4.3, 4.4, 4.5 and 4.7: calibration, held-out,
    bounds, excluded) has at least one entry (sub-ids a, b, ... allowed)."""
    def bases(role):
        return {re.sub(r"[a-z]$", "", t.id) for t in sets[role]}
    assert {f"CAL-{i}" for i in range(1, 15)} <= bases("calibration")
    assert {f"VAL-{i}" for i in range(1, 9)} <= bases("validation")
    assert {f"BND-{i}" for i in range(1, 5)} <= bases("bounds")
    assert {f"EXC-{i}" for i in range(1, 9)} <= bases("excluded")


def test_calibration_and_held_out_data_disjoint(sets):
    cal = {d for t in sets["calibration"] for d in t.data_points}
    for role in targets_io.HELD_OUT_ROLES:
        for t in sets[role]:
            if t.status == "deferred":
                continue
            shared = cal & set(t.data_points)
            assert not shared, f"{t.id} tests calibration data {shared}"


def test_split_check_detects_an_overlap(sets):
    """The overlap rule is not vacuous: copying a calibration data point into a validation entry is caught."""
    cal0 = sets["calibration"][1]
    val0 = sets["validation"][0]
    leaked = targets_io.Target(**{**val0.__dict__, "data_points": val0.data_points + (cal0.data_points[0],)})
    bad = dict(sets)
    bad["validation"] = (leaked,) + tuple(sets["validation"][1:])
    errs = targets_io.split_errors(bad)
    assert any("also calibration data" in e for e in errs)


def test_deferred_entries_declare_blockers(sets):
    for role in targets_io.HELD_OUT_ROLES:
        for t in sets[role]:
            if t.status == "deferred":
                assert t.raw.get("blocked_by"), t.id
                assert t.tolerance["kind"] == "pending"
                assert not t.scored


def test_every_scored_target_has_a_real_tolerance(sets):
    for role in targets_io.TARGET_ROLES:
        for t in sets[role]:
            if t.status == "active" and not (role == "calibration" and not t.fit):
                assert t.tolerance["kind"] not in ("none", "pending"), t.id


def test_unverified_and_cross_check_are_not_scored(sets):
    for role in targets_io.TARGET_ROLES:
        for t in sets[role]:
            if t.status in ("unverified", "cross_check", "deferred"):
                assert not t.scored, t.id


def test_replica_names_are_declared(sets):
    for role in targets_io.TARGET_ROLES:
        for t in sets[role]:
            assert t.replica in targets_io.REPLICA_NAMES, t.id


def test_replicas_module_implements_every_protocol():
    """Once mrm/validation/replicas.py exists it must provide every replica named by the targets."""
    if importlib.util.find_spec("mrm.validation.replicas") is None:
        pytest.skip("mrm/validation/replicas.py not written yet")
    mod = importlib.import_module("mrm.validation.replicas")
    missing = [n for n in targets_io.REPLICA_NAMES if not callable(getattr(mod, n, None))]
    assert not missing, f"replicas.py lacks {missing}"


def test_source_keys_exist_in_sources_yaml(sets):
    if not targets_io.SOURCES_YAML.exists():
        pytest.skip("mrm/params/sources.yaml not written yet")
    doc = yaml.safe_load(targets_io.SOURCES_YAML.read_text(encoding="utf-8")) or {}
    body = doc.get("sources", doc) if isinstance(doc, dict) else doc
    if isinstance(body, dict):
        keys = set(body)
    else:                                   # a list of {key: ...} / {id: ...} records
        keys = {str(r.get("key", r.get("id"))) for r in body if isinstance(r, dict)}
    used = {s for role in targets_io.ROLES for t in sets[role] for s in t.source}
    missing = sorted(used - keys)
    assert not missing, f"source keys missing from sources.yaml: {missing}"


def _strings(x):
    """Every string inside a nested dict / list structure."""
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


def test_cross_references_name_existing_ids(sets, by_id):
    """Every CAL-/VAL-/BND-/EXC- id mentioned in any text field exists (a base id like CAL-9 may stand for CAL-9a-c)."""
    bases = {re.sub(r"[a-z]$", "", i) for i in by_id}
    for role in targets_io.ROLES:
        for t in sets[role]:
            for s in _strings(t.raw):
                for ref in re.findall(r"\b(?:CAL|VAL|BND|EXC)-\d+[a-z]?\b", s):
                    assert ref in by_id or ref in bases, f"{t.id} mentions unknown {ref}"


def test_schema_rejects_bad_entries():
    doc = targets_io.read_raw("validation")
    bad = yaml.safe_load(yaml.safe_dump(doc))
    e = bad["targets"][0]
    e["provenance"] = "measured"            # not a provenance word
    e["generation"] = "G27"                 # not a generation tag
    e["unit"] = "furlong"                   # not a unit token
    e["tolerance"] = {"kind": "abs"}        # missing value
    e["conditions"]["colour"] = "red"       # unknown condition key
    errs = "\n".join(targets_io.schema_errors("validation", bad))
    for word in ("provenance", "generation", "furlong", "tolerance kind abs needs", "unknown condition keys"):
        assert word in errs


def test_schema_rejects_typos_that_would_silently_drop_a_check():
    """A misspelt optional field or tolerance key must be an error, not ignored (it could drop a split check)."""
    doc = yaml.safe_load(yaml.safe_dump(targets_io.read_raw("bounds")))
    e = doc["targets"][0]                                   # BND-1
    e["use"] = e.pop("uses")                                # misspelt: would skip the uses check
    e["tolerance"]["hardvalue"] = e["tolerance"].pop("hard_value")
    del e["conditions"]["bias_v"]
    del e["sigma"]
    errs = "\n".join(targets_io.schema_errors("bounds", doc))
    for word in ("unknown fields ['use']", "does not take ['hardvalue']", "conditions lack ['bias_v']",
                 "missing field 'sigma'"):
        assert word in errs, word


# --------------------------------------------------------------------------------------------------- re-derived numbers
def test_cal6_is_15_ghz_per_c_at_1310_nm(by_id):
    """dlambda = df * lambda^2 / c: 15 GHz at 1310 nm is 85.86 pm (report 01, verification log item 1)."""
    pm = 15e9 * LAMBDA0_M ** 2 / C0 * 1e12
    assert by_id["CAL-6"].value == pytest.approx(pm, abs=0.01)


def test_cal13a_rates_follow_from_the_rectified_sine(by_id):
    """T = Tmax - dT |sin(pi t / P)|: peak |dT/dt| = pi dT / P, mean = 2 dT / P (docs/research/01_lightmatter_public.md,
    "The 808 degC/s definition", and its verification log item 5)."""
    v = by_id["CAL-13a"].value
    p, d = v["params"], v["derived"]
    assert d["peak_rate_k_per_s"] == pytest.approx(math.pi * p["delta_t_k"] / p["period_s"], rel=2e-3)
    assert d["mean_rate_k_per_s"] == pytest.approx(2 * p["delta_t_k"] / p["period_s"], rel=2e-3)
    assert d["t_min_c"] == pytest.approx(p["t_max_c"] - p["delta_t_k"], abs=0.05)
    t = np.linspace(0.0, p["period_s"], 20001)
    temp = targets_io.rectified_sine(t, p["t_max_c"], p["delta_t_k"], p["period_s"])
    rate = np.abs(np.diff(temp) / np.diff(t))
    assert rate.max() == pytest.approx(d["peak_rate_k_per_s"], rel=2e-3)
    assert rate.mean() == pytest.approx(d["mean_rate_k_per_s"], rel=2e-3)


def test_heater_sweep_points_are_consistent(by_id):
    """CAL-4 shifts are the report 01 read-off positions minus the 0 V position; CAL-7 and VAL-1 use the same 0 V
    point."""
    c4 = by_id["CAL-4"]
    pos = np.asarray(c4.raw["absolute_positions_nm"])
    assert np.allclose(pos[1:] - pos[0], c4.value["y"], atol=1e-9)
    assert pos[0] == by_id["CAL-7"].raw["components"]["resonance_0v_nm"]
    assert by_id["CAL-7"].value == pytest.approx(
        by_id["CAL-7"].raw["components"]["operating_line_nm"] - pos[0], abs=0.05)
    # docs/research/01_lightmatter_public.md 2.2, the 1.5 V trace
    assert by_id["VAL-1"].value == pytest.approx(1314.89 - pos[0], abs=1e-9)
    assert max(c4.value["x"]) < 1.5


def test_notch_points_cover_the_g25_read_off_once(by_id):
    """CAL-2 (0 to 1.2 V) plus VAL-2 (1.4, 1.5 V) are the 8 notch points of docs/research/01 section 2.2, none
    twice: the top two points are held out, not fitted."""
    pts = dict(zip(by_id["CAL-2"].value["x"], by_id["CAL-2"].value["y"]))
    pts[1.4] = by_id["VAL-2a"].value
    pts[1.5] = by_id["VAL-2b"].value
    expected = {0.0: 27.9, 0.2: 29.8, 0.5: 29.8, 0.8: 24.4, 1.0: 22.1, 1.2: 19.8, 1.4: 18.0, 1.5: 16.9}
    assert pts == expected
    assert max(by_id["CAL-2"].value["x"]) <= 1.2


def test_cal5_average_efficiency(by_id):
    v = by_id["CAL-5"].value
    assert v["y"][-1] / v["x"][-1] == pytest.approx(52.8, abs=0.1)     # printed average, LM-T via REF


def test_cal3_implies_about_56_ohm(by_id):
    r = by_id["CAL-3b"].value ** 2 / (by_id["CAL-3a"].value * 1e-3)
    assert 52.8 <= r <= 70.5                                              # textbook REF.heaterOhmRange (inferred)
    assert r == pytest.approx(56.1, abs=0.1)


def test_val3b_ratios(by_id):
    c = by_id["VAL-3b"].raw["components"]
    assert c["legend"] == pytest.approx(35.4 / 31.8, abs=1e-3)
    assert c["caption"] == pytest.approx(34.1 / 31.8, abs=1e-3)
    assert min(c.values()) <= by_id["VAL-3b"].value <= max(c.values())


def test_val5_offsets_from_read_off_positions(by_id):
    t = by_id["VAL-5"]
    peak = t.raw["components"]["peak_c"]
    cont = t.raw["components"]["contours_c"]
    below = [peak - cont["m16db"][0], peak - cont["m24db"][0]]
    above = [cont["m16db"][1] - peak, cont["m24db"][1] - peak]
    assert np.allclose(below, t.value["below_peak_k"], atol=1e-9)
    assert np.allclose(above, t.value["above_peak_k"], atol=1e-9)
    assert all(b > a for b, a in zip(below, above))                      # cold side wider


def test_val7c_gamma_th_is_a_rate(by_id):
    """gamma_th = 2.4 MHz read as 1/tau gives 0.417 us, matching the printed 0.5 us; 1/(2 pi f) gives 66 ns."""
    c = by_id["VAL-7c"].raw["components"]
    assert c["from_gamma_th_2p4mhz_as_rate"] == pytest.approx(1e6 / 2.4e6, abs=1e-3)
    assert 1e6 / (2 * math.pi * 2.4e6) < 0.1                               # the angular reading, in us
    assert by_id["VAL-7c"].unit == "us"


def test_val8e_t63_point():
    """63 % point of 0.427(1 - exp(-t/3.18 us)) + 0.573(1 - exp(-t/13.01 us)) is 7.35 us (report 04, item V5,
    prints 7.3)."""
    t = np.linspace(0.0, 50.0, 500001)
    step = 0.427 * (1 - np.exp(-t / 3.18)) + 0.573 * (1 - np.exp(-t / 13.01))
    t63 = t[np.searchsorted(step, 1 - math.exp(-1))]
    assert t63 == pytest.approx(7.35, abs=0.01)


GRID_HZ = 400e9             # G26 per-direction TX grid spacing, Hz (LM26-P section 2, printed)
N_TX_LINES = 8              # TX lines per direction (LM26-P section 2, printed)


def _next_order_distance_ghz(fsr_hz: float, max_order: int = 3) -> float:
    """Smallest distance, over every ring and every other order m' != m (|m' - m| <= max_order), from a resonance
    order to another TX line of the bank, in GHz. Lines sit at k x 400 GHz, k = 0..7 (a 2.8 THz span)."""
    lines = np.arange(N_TX_LINES) * GRID_HZ
    best = np.inf
    for i, f_i in enumerate(lines):
        for k in range(-max_order, max_order + 1):
            if k == 0:
                continue
            others = np.delete(lines, i)
            best = min(best, float(np.min(np.abs(f_i + k * fsr_hz - others))))
    return best / 1e9


def _first_order_only_distance_ghz(fsr_hz: float) -> float:
    """What the validation plan (report 07, section 4.5) checked: FSR modulo the grid (the first order only)."""
    m = fsr_hz % GRID_HZ
    return min(m, GRID_HZ - m) / 1e9


def test_bnd1_examples_and_windows(by_id):
    """BND-1 re-derived on the 8-line, 400 GHz TX grid with FWHM = 60 GHz, scanning every order inside the span."""
    fwhm = 60.0
    assert _next_order_distance_ghz(1.8e12) / fwhm == pytest.approx(3.33, abs=0.01)
    fsr_8p8 = C0 * 8.8e-9 / LAMBDA0_M ** 2
    assert _next_order_distance_ghz(fsr_8p8) / fwhm == pytest.approx(1.04, abs=0.02)
    for n_g in (3.9, 4.04):
        fsr = C0 / (n_g * 2 * math.pi * 7.5e-6)
        assert _next_order_distance_ghz(fsr) / fwhm < 0.55
    windows = ((1.32e12, 1.34e12), (1.46e12, 1.48e12), (1.72e12, 1.88e12), (2.12e12, 2.28e12), (2.52e12, 2.68e12))
    for lo, hi in windows:
        for fsr in np.linspace(lo, hi, 33):
            assert _next_order_distance_ghz(fsr) >= 2 * fwhm - 1e-6, fsr
    # Lorentzian through-port loss at 2 FWHM, T_min = 0.01: 1 - 0.99/(1 + 4^2) -> 0.26 dB (report 07, section 4.5)
    il = -10 * math.log10(1 - 0.99 / (1 + 4.0 ** 2))
    assert il == pytest.approx(0.26, abs=0.005)
    assert by_id["BND-1"].tolerance["value"] == 2.0
    assert "all" in str(by_id["BND-1"].replica_args.get("orders", ""))


def test_bnd1_window_a_interior_fails_on_the_second_order(by_id):
    """The validation plan's window A (1.32-1.48 THz; report 07, section 4.5) checked only the first order. Inside the
    2.8 THz span the second order also counts: at 1.40 THz the second order of the ring on line 0 lands exactly on
    line 7, and 1.34-1.46 THz is within 2 FWHM of a line. The bound file records the corrected windows."""
    fwhm = 60.0
    assert _first_order_only_distance_ghz(1.40e12) / fwhm == pytest.approx(3.33, abs=0.01)   # the plan's reading
    assert _next_order_distance_ghz(1.40e12) == pytest.approx(0.0, abs=1e-6)                  # all orders
    for fsr in np.linspace(1.35e12, 1.45e12, 21):
        assert _next_order_distance_ghz(fsr) < 2 * fwhm
    text = by_id["BND-1"].raw["verification"]
    assert "1.32-1.34" in text and "1.46-1.48" in text


def test_bnd2_margins():
    """BND-2 margin from 7.6 nm over 25 -> 105 degC: H86 leaves 0.72 nm (report 07, section 4.5). H_solver, with
    dlambda/dT rising linearly from 66-69 pm/K at 25 degC to 74-77 pm/K (73-76 after the correction of report 06) at
    105 degC, leaves 1.8-2.0 nm."""
    assert 7.6 - 0.086 * 80.0 == pytest.approx(0.72, abs=1e-9)
    margins = [7.6 - 80.0 * (s25 + s105) / 2 * 1e-3 for s25, s105 in ((66, 74), (69, 77), (66, 73), (69, 76))]
    assert min(margins) == pytest.approx(1.76, abs=0.01)
    assert max(margins) == pytest.approx(2.04, abs=0.01)


def test_exc4_secant_is_about_0p65_fwhm():
    """2.85 K x 86 pm/K against FWHM = 1310 nm / 3500 (report 06, item V5)."""
    assert 2.85 * 86.0 / (1310.0 / 3500 * 1000) == pytest.approx(0.65, abs=0.01)


# ---------------------------------------------------------------------------------------------------------------- units
@pytest.mark.parametrize("unit,factor", [
    ("nm", 1e-9), ("pm_per_k", 1e-12), ("pm_per_mw", 1e-9), ("k_per_s", 1.0), ("a_per_w", 1.0),
    ("ua", 1e-6), ("ghz", 1e9), ("codes", 1.0), ("pct", 1e-2), ("1", 1.0), ("pa_per_rt_hz", 1e-12),
    ("um2", 1e-12), ("w_per_m_k", 1.0), ("pm_per_k_per_mw", 1e-9),
])
def test_unit_factors(unit, factor):
    assert targets_io.unit_factor(unit) == pytest.approx(factor)


@pytest.mark.parametrize("bad", ["c_per_s", "code", "furlong", "per_s", "nm_per", "db_per_c", "rt"])
def test_unit_grammar_rejects(bad):
    with pytest.raises(ValueError):
        targets_io.parse_unit(bad)


def test_to_si_special_units():
    assert targets_io.to_si(25.0, "c") == pytest.approx(298.15)
    assert targets_io.to_si(0.0, "dbm") == pytest.approx(1e-3)
    assert targets_io.to_si(-2.1, "db") == pytest.approx(-2.1 * math.log(10) / 10)    # natural-log power ratio
    assert np.allclose(targets_io.to_si([1.0, 2.0], "us"), [1e-6, 2e-6])
    for bad in ("db", "c", "mixed"):
        with pytest.raises(ValueError):
            targets_io.unit_factor(bad)


def test_units_agree_with_params_grammar(sets):
    """Every unit used by the target files converts exactly as mrm.params.units converts a key with that suffix."""
    if importlib.util.find_spec("mrm.params.units") is None:
        pytest.skip("mrm/params/units.py not written yet")
    pu = importlib.import_module("mrm.params.units")
    units = {t.unit for role in targets_io.TARGET_ROLES for t in sets[role]} - {"1", "mixed"}
    for u in sorted(units):
        x = np.array([0.5, 2.0])
        theirs = pu.to_si(f"quantity_{u}", x)
        assert np.allclose(targets_io.to_si(x, u), theirs, rtol=1e-12), u


# ----------------------------------------------------------------------------------------------------------- evaluate()
def test_evaluate_abs_curve(by_id):
    t = by_id["CAL-2"]
    y = np.asarray(t.value["y"])
    assert targets_io.evaluate(t, y + 1.0).passed
    ev = targets_io.evaluate(t, y + np.r_[0, 0, 0, 0, 0, 2.0])
    assert not ev.passed and ev.max_abs_residual == pytest.approx(2.0)
    assert ev.z == pytest.approx(4.0)                                     # 2.0 dB / sigma 0.5 dB


def test_evaluate_rel_and_set(by_id):
    t = by_id["VAL-8c"]
    assert targets_io.evaluate(t, 222.0 * 1.2).passed
    assert not targets_io.evaluate(t, 222.0 * 1.3).passed
    s = by_id["CAL-9b"]
    assert targets_io.evaluate(s, {"at_code_maxima": 7000, "at_code_minima": 10500}).passed
    assert not targets_io.evaluate(s, {"at_code_maxima": 7000, "at_code_minima": 11000}).passed
    with pytest.raises(ValueError):
        targets_io.evaluate(s, {"at_code_maxima": 7000})


def test_evaluate_interval_and_inequality(by_id):
    q = by_id["CAL-1"]
    assert targets_io.evaluate(q, 3800).passed
    assert not targets_io.evaluate(q, [3400, 3900]).passed
    b1 = by_id["BND-1"]
    assert targets_io.evaluate(b1, [3.3, 2.5]).level == "pass"
    assert targets_io.evaluate(b1, [3.3, 1.5]).level == "soft_fail"
    assert targets_io.evaluate(b1, [3.3, 0.5]).level == "fail"
    assert targets_io.evaluate(by_id["BND-4"], 0.8).passed
    assert not targets_io.evaluate(by_id["BND-4"], 1.2).passed


def test_evaluate_band(by_id):
    b3 = by_id["BND-3a"]
    assert targets_io.evaluate(b3, [-2.1, 3.2]).passed                           # inside with read-off slack
    assert not targets_io.evaluate(b3, [-3.5, 1.0]).passed                       # outside in both orientations
    mirrored = targets_io.evaluate(b3, [-3.2, 2.1])                       # the slide's sign convention is unknown
    assert mirrored.passed and "mirrored" in mirrored.detail
    with pytest.raises(ValueError):
        targets_io.evaluate(b3, [3.0, -1.0])                                     # [max, min] is a replica bug
    b3b = by_id["BND-3b"]
    assert targets_io.evaluate(b3b, [-0.4, 0.45]).passed
    ev = targets_io.evaluate(b3b, [-0.2, 0.7])
    assert not ev.passed and ev.max_abs_residual == pytest.approx(0.2)


def test_evaluate_rms_parametric(by_id):
    t = by_id["CAL-13a"]
    p = t.value["params"]
    ts = np.linspace(0, 2 * p["period_s"], 2001)
    ref = targets_io.rectified_sine(ts, p["t_max_c"], p["delta_t_k"], p["period_s"])
    assert targets_io.evaluate(t, {"t_s": ts, "y": ref + 0.5}).passed
    ev = targets_io.evaluate(t, {"t_s": ts, "y": ref + 1.5})
    assert not ev.passed and ev.max_abs_residual == pytest.approx(1.5)


def test_evaluate_contour_asymmetry(by_id):
    t = by_id["VAL-5"]
    good = {"below_peak_k": [8.0, 12.0], "above_peak_k": [5.5, 9.0]}
    assert targets_io.evaluate(t, good).passed
    mirrored = {"below_peak_k": [4.9, 9.4], "above_peak_k": [7.6, 12.6]}
    assert not targets_io.evaluate(t, mirrored).passed                           # wrong side wider
    assert not targets_io.evaluate(t, {"below_peak_k": [11.0, 12.0], "above_peak_k": [5.0, 9.0]}).passed


def test_evaluate_not_scored(by_id):
    assert targets_io.evaluate(by_id["CAL-6"], 70.0).level == "not_scored"
    assert targets_io.evaluate(by_id["VAL-6"], 30000).level == "not_scored"
    ev = targets_io.evaluate(by_id["VAL-7a"], 2.73)                              # unverified: computed, not scored
    assert ev.passed and not ev.scored
    assert not targets_io.evaluate(by_id["VAL-8e"], {"tau1_us": 3.18, "tau2_us": 13.01, "share1_frac": 0.427,
                                              "share2_frac": 0.573}).scored


# ------------------------------------------------------------------------------------------------ contract completeness
def test_every_entry_states_the_p0_5_conditions(sets):
    """Every entry states its measurement conditions {T_c, heater_v or code, bias_v, p_in_dbm, port} (null = not
    printed), and its sigma."""
    for role in targets_io.TARGET_ROLES:
        for t in sets[role]:
            for k in targets_io.REQUIRED_CONDITION_KEYS:
                assert k in t.conditions, (t.id, k)
            assert any(k in t.conditions for k in targets_io.HEATER_CONDITION_KEYS), t.id
            assert "sigma" in t.raw, t.id


def test_read_off_entries_record_digitizer_and_crop(sets):
    """Read-off targets (numbers digitized from a published plot) carry their digitization sigma and record the
    digitizer and the crop, so the read-off can be redone."""
    for role in targets_io.TARGET_ROLES:
        for t in sets[role]:
            if t.provenance != "read-off":
                continue
            ro = t.raw.get("readoff")
            assert isinstance(ro, dict) and all(str(ro.get(k, "")).strip() for k in targets_io.READOFF_KEYS), t.id
            if role != "bounds" and t.status == "active":
                assert t.sigma is not None, f"{t.id}: an active read-off target needs its read-off sigma"


def test_tolerances_are_not_tighter_than_two_sigma(sets):
    """A tolerance below 2 sigma would fail a perfect model on read-off noise alone."""
    for role in targets_io.TARGET_ROLES:
        for t in sets[role]:
            k = t.tolerance.get("kind")
            sig = [float(x) for x in targets_io._numbers(t.sigma)] if t.sigma is not None else []
            if not sig or k not in ("abs", "rel") or t.kind == "parametric":
                continue
            vals = [abs(float(v)) for v in targets_io._numbers(t.value)]
            allowed = t.tolerance["value"] * (min(vals) if k == "rel" else 1.0)
            assert allowed >= 2 * max(sig) - 1e-12, (t.id, allowed, max(sig))


def test_cal12b_is_a_population_not_a_sigma_bound(by_id):
    """+-15 GHz is the Guide yield limit; the plotted population has outliers to -21 GHz, so it is neither a hard
    bound on |error| nor a standard deviation. The target carries the quartiles instead."""
    t = by_id["CAL-12b"]
    v, c = t.value, t.raw["components"]
    assert v["q25"] < v["median"] < v["q75"]
    assert c["yield_limit_printed_ghz"] == 15.0 and c["outliers_to_ghz"] < -c["yield_limit_printed_ghz"]
    assert t.tolerance["kind"] == "abs"


def test_cal12d_has_a_source(by_id):
    """CAL-12d cites its source (YOLE) and is active, not marked as rechecked."""
    t = by_id["CAL-12d"]
    assert t.source == ("YOLE",) and t.status == "active" and t.rechecked is False


def test_calibration_loader_never_opens_held_out_files(tmp_path):
    """load_calibration_targets works with calibration.yaml alone, so it cannot depend on the held-out files."""
    (tmp_path / "calibration.yaml").write_bytes(targets_io.targets_path("calibration").read_bytes())
    ts = targets_io.load_calibration_targets(tmp_path)
    assert ts and all(t.role == "calibration" for t in ts)
    assert {t.id for t in targets_io.load_calibration_targets(tmp_path, active_only=True)} <= {t.id for t in ts}


# ------------------------------------------------------------------------------------------------ evaluate() edge cases
def test_evaluate_non_finite_prediction_fails(by_id):
    for tid, pred in (("CAL-2", [np.nan] * 6), ("CAL-1", np.inf), ("BND-1", [np.nan, 3.0]), ("BND-3a", [np.nan, 1.0]),
                      ("VAL-5", {"below_peak_k": [np.nan, 12.0], "above_peak_k": [5.0, 9.0]}),
                      ("CAL-9b", {"at_code_maxima": np.nan, "at_code_minima": 9500})):
        ev = targets_io.evaluate(by_id[tid], pred)
        assert ev.passed is False and ev.level == "fail", tid


def test_evaluate_z_uses_each_points_sigma(by_id):
    """CAL-4 has per-point sigma (0.04 nm, 0.06 nm at the '~' points); z must use the point's own sigma."""
    t = by_id["CAL-4"]
    y = np.asarray(t.value["y"], dtype=float)
    bump = np.zeros_like(y)
    bump[0] = 0.08                                                        # sigma 0.04 there
    ev = targets_io.evaluate(t, y + bump)
    assert ev.z == pytest.approx(2.0)
    s = by_id["CAL-9b"]                                                   # sigma {7500: 200, 9500: 400}
    ev = targets_io.evaluate(s, {"at_code_maxima": 7900, "at_code_minima": 9500})
    assert ev.z == pytest.approx(2.0)


def test_evaluate_inequality_residual_is_the_violation(by_id):
    b1 = by_id["BND-1"]
    ok = targets_io.evaluate(b1, [3.3, 2.5])
    assert ok.passed and ok.max_abs_residual == 0.0 and "margin +0.5" in ok.detail
    bad = targets_io.evaluate(b1, [3.3, 1.5])
    assert bad.max_abs_residual == pytest.approx(0.5)
