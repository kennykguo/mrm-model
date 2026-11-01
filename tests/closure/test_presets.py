"""Closure tests of the shipped presets: every preset loads, closes its relations and realizes.

Read this first. A closure test asks whether the published numbers of a preset agree with each other once derive()
has computed the derived ones. These tests block CI. For every preset in ``mrm/params/presets/*.yaml`` they check:

1. it loads and passes the schema validation of :func:`mrm.params.load` (strict provenance, units, fidelity rules,
   uncertainty and variation entry formats);
2. every relation R1-R5 closes after :func:`mrm.params.realize` (residual <= 1e-12), for the nominal world and for
   a sampled batch of worlds and devices, and every hypothesis choice of every uncertainty entry realizes;
3. every corner preset (``meta.kind: corner``) is a thin ``extends:`` file that only selects alternatives (never a
   copy of a preset with other numbers), and every selection it makes changes the realized plant;
4. the completeness inventory ``meta.inventory`` of g26 covers every item of
   ``docs/research/00_prior_notes_checklist.md`` (key, excluded with a reason, or unknown with a note), and every key
   it names exists;
5. the solver-derived leaves use the ``{from: "extracted/..."}`` form with a ``fallback`` alternative, and every
   source key exists in ``mrm/params/sources.yaml``;
6. calibration consistency of the nominal worlds: the exact G25 ring reproduces the CAL-2 0 V notch in the labelled
   coupling regime (also in the 200 nm world), every eta choice keeps Q_i, the monitor current matches CAL-8 without
   clipping the ADC, the laser ramp at reset lasts at least 10 slow thermal time constants, the G25 bench heater is
   voltage driven, the geometry draws reproduce the Berkeley 6 nm / 2 nm resonance spreads, and every corner pins
   what it says in every sampled world.

Missing extraction artefacts. A leaf whose ``from:`` artefact file does not exist takes its ``fallback`` alternative,
with a warning. :mod:`mrm.params.config` does this itself now (``config.ARTEFACT_FALLBACK_ALT``). The autouse fixture
:func:`fallback_loader` installs :func:`patched_make_leaf`, a copy of ``config._make_leaf`` with that behaviour, only
for a loader without it. With the current loader the adapter is not installed and its two tests skip. No artefact is
fabricated.
"""
from __future__ import annotations

import contextlib
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from mrm.params import config as pconfig
from mrm.params import schema

ROOT = Path(__file__).resolve().parents[2]
PRESET_DIR = ROOT / "mrm" / "params" / "presets"
CHECKLIST = ROOT / "docs" / "research" / "00_prior_notes_checklist.md"
PRESETS = sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))
CORNERS = [p for p in PRESETS if (yaml.safe_load((PRESET_DIR / f"{p}.yaml").read_text()).get("meta") or {}).get(
    "kind") == "corner"]
FALLBACK = "fallback"          # the alternative a leaf takes when its artefact file is missing

#: the corner presets of the uncertainty set (the worlds controllers are judged on), plus their parents
REQUIRED_PRESETS = {"g26", "g25_test_vehicle", "teaching_basic", "g26_slow_thermal", "g26_fast_thermal",
                    "g26_undercut", "g26_h86", "g26_h86_sensor", "g26_overcoupled", "g26_heater_loss",
                    "g26_hot_selfheat", "g26_low_eta", "g26_fsr_window_a", "g26_fsr_edge", "g26_textbook_thermal"}

#: leaves whose value must come from an extraction artefact (docs/ARCHITECTURE.md, "Artefact formats")
SOLVER_LEAVES = ("ring.optics.n_eff", "ring.optics.n_g", "ring.optics.dn_g_dlambda_per_nm",
                 "ring.thermal.dlambda_dt_pm_per_k",
                 "ring.optics.loss_budget.alpha_bend_radiation_db_per_cm", "ring.geometry.jacobian",
                 "ring.geometry.jacobian_coupling",
                 "ring.optics.dln_kappa2_through_dlambda_per_nm", "ring.optics.dln_kappa2_drop_dlambda_per_nm")
#: solver leaves computed from an artefact table (not a from-form): the number is recomputed by a closure test
SOLVER_COMPUTED_LEAVES = {"ring.thermal.dlambda_dt_beta_per_k": "tests/closure/test_params_notes.py"}


# ----------------------------------------------------------------------------------------------------- fallback adapter

def _artefact_missing(err: Exception) -> bool:
    """True only for an artefact FILE that does not exist (a bad sha256, sidecar or key stays an error)."""
    return "not found under" in str(err)


def patched_make_leaf(path: str, d: dict, origin: list[str], cfg, errs: list[str]):
    """``mrm.params.config._make_leaf`` plus the artefact fallback, for a loader without it.

    When the ``from:`` artefact file does not exist and the leaf has an alternative named ``fallback`` (a mapping with
    a ``value``), the leaf takes that alternative's value, provenance, generation and source, ``selected_alt`` is
    ``fallback``, and a warning names the missing artefact. Everything else is a copy of the loader's code."""
    d = dict(d)
    art = None
    val = d.get("value")
    spec = d.get("from")
    at = d.get("at")
    fell_back = False
    if isinstance(val, dict) and "from" in val:
        spec = val["from"]
        at = val.get("at", at)
    if spec is not None:
        try:
            val, art = pconfig.resolve_from(spec, at, cfg.data_root)
        except pconfig.ArtefactError as e:
            fb = (d.get("alternatives") or {}).get(FALLBACK)
            if _artefact_missing(e) and isinstance(fb, dict) and "value" in fb:
                val = fb["value"]
                for f in ("provenance", "generation", "source"):
                    if fb.get(f) is not None:
                        d[f] = fb[f]
                fell_back = True
                cfg.warnings.append(f"{path}: {e}; using its '{FALLBACK}' alternative {val!r} "
                                    f"({d.get('provenance')}) until the artefact exists")
            else:
                errs.append(f"{path}: {e}")
                val = None
        if not fell_back:
            if "provenance" not in d:
                d["provenance"] = art["provenance"] if art is not None else "solver"
            elif art is not None and not str(d.get("provenance", "")).startswith("solver"):
                errs.append(f"{path}: a value taken from a solver artefact must have provenance 'solver' "
                            f"(got {d.get('provenance')!r})")
    prov = d.get("provenance")
    if prov is None:
        errs.append(f"{path}: leaf has no provenance (a number without a provenance word is a bug)")
        prov = "assumed"
    alts = {}
    for name, a in (d.get("alternatives") or {}).items():
        if isinstance(a, dict) and ("value" in a or "from" in a):
            bad = set(a) - pconfig._ALTERNATIVE_FIELDS
            if bad:
                errs.append(f"{path}: alternative {name!r} has unknown fields {sorted(bad)}")
            alts[str(name)] = dict(a)
        else:
            alts[str(name)] = {"value": a, "_bare": True}
    if alts and "default" not in alts:
        alts["default"] = {"value": val, "provenance": prov, "generation": d.get("generation"),
                           "source": d.get("source"), "_resolved_artefact": art}
    rng = d.get("range")
    leaf = pconfig.Leaf(path=path, value=val, provenance=prov, generation=d.get("generation"), source=d.get("source"),
                        range=tuple(rng) if isinstance(rng, (list, tuple)) else rng, alternatives=alts,
                        note=d.get("note"), textbook=d.get("textbook"), disagrees=d.get("disagrees"),
                        targets=d.get("targets"), residual=d.get("residual"), identifiable=d.get("identifiable"),
                        origin=list(origin), artefact=art)
    if fell_back:
        leaf.selected_alt = FALLBACK
        leaf.origin.append(f"artefact {spec!r} missing: alternative {FALLBACK!r}")
    return leaf


#: the loader's own _make_leaf, captured before any test installs the adapter (drift guard below)
_ORIG_MAKE_LEAF = pconfig._make_leaf


def loader_has_fallback() -> bool:
    """True when mrm.params.config implements the fallback itself (then no adapter is installed)."""
    return getattr(pconfig, "ARTEFACT_FALLBACK_ALT", None) == FALLBACK


@contextlib.contextmanager
def artefact_fallback():
    """Install :func:`patched_make_leaf` unless the loader already falls back (restores the original on exit)."""
    if loader_has_fallback():
        yield False
        return
    orig = pconfig._make_leaf
    pconfig._make_leaf = patched_make_leaf
    try:
        yield True
    finally:
        pconfig._make_leaf = orig


@pytest.fixture(autouse=True)
def fallback_loader():
    """Every test runs with the artefact fallback (the adapter only when the loader lacks it); yields "installed"."""
    with artefact_fallback() as installed:
        yield installed


_CACHE: dict = {}            # (preset name, fallback active) -> Config, for loads without keywords


def load(name, **kw):
    """Load a preset (warnings silenced). Plain loads are cached per adapter state; tests never mutate a Config."""
    key = (name, loader_has_fallback() or pconfig._make_leaf is patched_make_leaf) if not kw else None
    if key is not None and key in _CACHE:
        return _CACHE[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = pconfig.load(name, **kw)
    if key is not None:
        _CACHE[key] = cfg
    return cfg


def realize(cfg, batch=1, **kw):
    """mrm.params.realize with the warnings silenced."""
    from mrm.params.realize import realize as _realize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _realize(cfg, batch, **kw)


# -------------------------------------------------------------------------------------------------------------- helpers

def _hypothesis_choices(entry) -> list[str]:
    """The choice names of a hypothesis uncertainty entry."""
    return list(pconfig._hypothesis_choice_map(entry))


def _max_residual(pp) -> float:
    """Largest |residual| of the relations R1-R5 over every world and ring of a realized PlantParams."""
    res = pp.diagnostics["relation_residuals"]
    vals = [float(np.max(np.abs(v))) if np.size(v) else 0.0 for v in res.values()] if isinstance(res, dict) else [
        float(res)]
    return max(vals) if vals else 0.0


def _checklist_items() -> dict[str, list[str]]:
    """Items of the checklist inventory (sections A-K) plus the OFC 2026 'what it does not give' list."""
    text = CHECKLIST.read_text()
    inv = text.split("## Inventory", 1)[1]
    out: dict[str, list[str]] = {}
    for para in re.split(r"\n\s*\n", inv):
        para = " ".join(para.split())
        m = re.match(r"^([A-K])\. ([^:]+): (.*)$", para)
        if m:
            out[m.group(1)] = [s.strip() for s in m.group(3).rstrip(".").split(";")]
    m = re.search(r"what it does not \(([^)]*)\)", " ".join(text.split()))
    assert m, "checklist: the OFC 2026 'what it does not' list moved"
    out["OFC2026_not_given"] = [s.strip() for s in m.group(1).split(",")]
    return out


def _key_exists(key: str, cfg) -> bool:
    """True when an inventory key names an uncertainty entry, a variation entry or a schema path of ``cfg``."""
    if key.startswith("uncertainty."):
        return key[len("uncertainty."):] in cfg.uncertainty
    if key.startswith("variation."):
        rest = key[len("variation."):]
        return (rest == "geometry" and bool(cfg.geometry)) or rest in cfg.variation
    return schema.lookup(key) is not None


# ---------------------------------------------------------------------------------------------------------------- tests

def test_required_presets_exist():
    missing = REQUIRED_PRESETS - set(PRESETS)
    assert not missing, f"presets missing: {sorted(missing)}"


@pytest.mark.parametrize("name", PRESETS)
def test_preset_loads_and_validates(name):
    cfg = load(name)
    assert cfg.kind in ("preset", "corner")
    for p, lf in cfg.leaves.items():
        assert lf.provenance, p
        if not lf.is_free:
            assert lf.generation is not None and lf.source is not None, p
            assert lf.source in cfg.sources, f"{p}: source {lf.source!r} not in sources.yaml"


@pytest.mark.parametrize("name", PRESETS)
def test_preset_closes_every_relation(name):
    cfg = load(name)
    nominal = realize(cfg, 1, seeds=0, uncertainty="nominal", variation="off")
    assert _max_residual(nominal) <= 1e-12
    sampled = realize(cfg, 16, seeds=3, uncertainty="lhs", variation="sample")
    assert _max_residual(sampled) <= 1e-12
    assert np.all(np.isfinite(sampled.optics.q_through)) and np.all(sampled.optics.q_through > 0)
    assert np.all((sampled.optics.heat_fraction > 0) & (sampled.optics.heat_fraction < 1))


@pytest.mark.slow
def test_g26_every_hypothesis_choice_realizes():
    """Pin each choice of each hypothesis in turn (each choice is a world of the uncertainty set): every world must
    realize and close."""
    base = load("g26")
    for ename, e in sorted(base.uncertainty.items()):
        if e.kind != "hypothesis":
            continue
        for c in _hypothesis_choices(e):
            cfg = load("g26", alts={f"uncertainty.{ename}": c})
            pp = realize(cfg, 1, seeds=1, uncertainty="nominal", variation="sample")
            assert _max_residual(pp) <= 1e-12, (ename, c)


@pytest.mark.slow
def test_g26_interval_corners_realize():
    """Pin each interval entry at its low and high end and at each named corner: every world must realize and close."""
    base = load("g26")
    for ename, e in sorted(base.uncertainty.items()):
        if e.kind == "hypothesis":
            continue
        for c in ["low", "high"] + sorted((e.fields.get("corners") or {})):
            cfg = load("g26", alts={f"uncertainty.{ename}": c})
            pp = realize(cfg, 1, seeds=1, uncertainty="nominal", variation="off")
            assert _max_residual(pp) <= 1e-12, (ename, c)


@pytest.mark.parametrize("name", CORNERS)
def test_corner_only_selects_alternatives(name):
    raw = yaml.safe_load((PRESET_DIR / f"{name}.yaml").read_text())
    assert set(raw) <= {"extends", "meta", "alts"}, f"{name}: a corner may hold only extends, meta and alts"
    assert raw.get("extends"), f"{name}: a corner must extend a preset"
    assert (raw.get("meta") or {}).get("kind") == "corner"
    assert raw.get("alts"), f"{name}: a corner must select at least one alternative"
    cfg = load(name)
    parent = load(raw["extends"])
    assert cfg.alts_applied, name
    for path, choice in raw["alts"].items():
        assert cfg.alts_applied.get(path) == str(choice), (name, path)
    # the selections must change the realized nominal plant (otherwise the corner is a no-op)
    a = realize(parent, 1, seeds=0, uncertainty="nominal", variation="off")
    b = realize(cfg, 1, seeds=0, uncertainty="nominal", variation="off")
    diffs = [k for k in a.values if k in b.values and np.shape(a.values[k]) == np.shape(b.values[k])
             and not np.allclose(np.asarray(a.values[k], float), np.asarray(b.values[k], float), equal_nan=True)]
    pinned_world = {k for k in b.world if not np.array_equal(np.asarray(a.world.get(k)), np.asarray(b.world[k]))}
    assert diffs or pinned_world, f"{name}: the alternatives it selects do not change the realized plant"


def test_corners_cover_07_section_4_8():
    """Every corner of the uncertainty set exists (docs/research/07_design_critique.md, "The uncertainty set, and how
    controllers are judged")."""
    assert {"g26_slow_thermal", "g26_fast_thermal", "g26_undercut", "g26_h86", "g26_h86_sensor", "g26_overcoupled",
            "g26_heater_loss", "g26_hot_selfheat", "g26_low_eta", "g26_fsr_window_a", "g26_fsr_edge",
            "g26_textbook_thermal"} <= set(CORNERS)


def test_teaching_basic_switches_off_selfheating_and_noise():
    cfg = load("teaching_basic")
    assert cfg.get("fidelity.self_heating") == "off"
    assert cfg.get("fidelity.shot_noise") == "off"
    assert cfg.get("experiment.uncertainty") == "nominal" and cfg.get("experiment.variation") == "off"
    pp = realize(cfg, 2, seeds=0)
    assert pp.flags.self_heating == 0 and pp.flags.shot_noise == 0
    assert np.all(pp.monitor.input_noise_a_per_rt_hz == 0)
    np.testing.assert_array_equal(pp.optics.q_loaded[0], pp.optics.q_loaded[1])   # one nominal world


def test_selfheating_on_in_every_non_teaching_preset():
    for name in PRESETS:
        if name.startswith("teaching"):
            continue
        assert load(name).get("fidelity.self_heating") == "on", name


def test_solver_leaves_use_the_from_form_with_a_fallback():
    raw = yaml.safe_load((PRESET_DIR / "g26.yaml").read_text())

    def node(path):
        n = raw
        for part in path.split("."):
            n = n[part]
        return n

    for path in SOLVER_LEAVES:
        leaf = node(path)
        spec = leaf.get("from") or (leaf.get("value") or {}).get("from")
        assert isinstance(spec, str) and spec.startswith("extracted/"), path
        assert FALLBACK in (leaf.get("alternatives") or {}), f"{path}: needs a '{FALLBACK}' alternative"
        fb = leaf["alternatives"][FALLBACK]
        assert "value" in fb and fb.get("provenance") in ("literature", "inferred", "assumed"), path
    for path, test in SOLVER_COMPUTED_LEAVES.items():
        # a number fitted to an artefact table (the secant beta over the operating range, CHANGELOG params-5):
        # solver provenance, a fallback, and a note naming the test that recomputes it from the artefact
        leaf = node(path)
        assert isinstance(leaf.get("value"), float) and str(leaf.get("provenance")).startswith("solver"), path
        assert FALLBACK in (leaf.get("alternatives") or {}), f"{path}: needs a '{FALLBACK}' alternative"
        assert test in leaf.get("note", "") and (ROOT / test).exists(), path


def test_inventory_covers_the_checklist():
    cfg = load("g26")
    inv = cfg.meta.get("inventory")
    assert isinstance(inv, dict), "g26 meta.inventory is missing"
    items = _checklist_items()
    assert sum(len(v) for v in items.values()) >= 180
    problems = []
    for sec, its in items.items():
        got = inv.get(sec) or {}
        for it in its:
            if it not in got:
                problems.append(f"{sec}: {it!r} not in meta.inventory")
                continue
            e = got[it]
            st = e.get("status")
            if st not in ("key", "excluded", "unknown"):
                problems.append(f"{sec}: {it!r} status {st!r}")
            if st == "key" and not e.get("keys"):
                problems.append(f"{sec}: {it!r} is 'key' but lists no keys")
            if st in ("excluded", "unknown") and not e.get("reason"):
                problems.append(f"{sec}: {it!r} is {st!r} without a reason")
            for k in e.get("keys") or []:
                if not _key_exists(k, cfg):
                    problems.append(f"{sec}: {it!r} names unknown key {k!r}")
        extra = set(got) - set(its)
        if extra:
            problems.append(f"{sec}: inventory items not in the checklist: {sorted(extra)}")
    assert not problems, "\n".join(problems)


def test_every_source_key_is_declared():
    known = pconfig.load_sources()
    assert known is not None, "mrm/params/sources.yaml is missing"
    needed = {"BU-45CLO", "CO22-JLT", "CO22-TCPMT", "CO24-JOM", "DEC19", "GF-ECIO", "HC", "HOTI26", "LM-P", "LM-T",
              "LM26-P", "LM26-S", "OCP", "PAT-4", "PECC", "SIB", "SMTAI", "YOLE", "textbook-REF"}
    assert needed <= known, sorted(needed - known)       # the source keys that the validation targets cite
    for name in PRESETS:
        cfg = load(name)
        for n, e in cfg.uncertainty.items():
            assert e.fields.get("source") in known, (name, n)
        for p, e in cfg.variation.items():
            assert e.fields.get("source") in known, (name, p)
        for p, lf in cfg.leaves.items():                     # leaf alternatives carry their own source keys
            for an, a in (lf.alternatives or {}).items():
                assert a.get("source") is None or a["source"] in known, (name, p, an)


def test_textbook_disagreements_are_recorded():
    """The presets carry the textbook field and the reason wherever the model deliberately differs from the textbook
    (the validation report lists these disagreements from the ``disagrees:`` fields)."""
    cfg = load("g26")
    dis = {p for p, lf in cfg.leaves.items() if lf.disagrees}
    for p in ("ring.thermal.dlambda_dt_pm_per_k", "ring.thermal.tau_fast_s", "ring.thermal.xt_nearest_frac",
              "ring.optics.loss_budget.alpha_fca_db_per_cm", "ring.geometry.film_nm", "ring.heater.code_min"):
        assert p in dis, p
    for p in dis:
        assert cfg.leaves[p].textbook, f"{p}: a disagreement must name the textbook field"


def test_g26_nominal_physics_sanity():
    """Order-of-magnitude anchors for the nominal G26 world: heat fraction, intrinsic Q, heater efficiency, lock
    temperature and the self-heating bistability."""
    from mrm.params.derive import derive
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = derive(load("g26"))
    pp = d.params
    eta = float(np.mean(pp.optics.heat_fraction))
    assert 0.8 < eta < 0.9                                              # heat fraction from the loss budget, ~0.85
    qi = float(np.mean(pp.optics.q_intrinsic))
    assert 7000 < qi < 8400                                             # intrinsic Q near critical coupling
    eff = float(np.mean(pp.thermal.heater_efficiency_m_per_w)) * 1e9 / 1e3   # nm/mW
    assert 0.38 < eff < 0.42                                            # FSR / P_FSR = 10.3 nm / 25.45 mW
    tl = float(np.mean(pp.thermal.t_ring_lock_k)) - 273.15
    assert 110 < tl < 150                                               # lock temperature, ~132 degC (H_solver)
    assert float(np.max(d.diagnostics["bistability_ratio"])) > 1.0      # self-heating bistable at +4 dBm


# ---------------------------------------------------------------------------------------------- calibration consistency
# What the nominal worlds must reproduce, and what each corner must pin.

G25_NOTCH_DB = 27.9          # CAL-2: G25 notch depth at 0 V heater, dB (read off the published plot)
CAL2_SIGMA_DB = 0.5          # CAL-2 read-off sigma, dB
CAL8_UA = (15.0, 16.5)       # CAL-8: G26 drop monitor current at lock, uA (read off as 15.5 +- 0.5 uA)


def _exact_ring(pp, b: int = 0, i: int = 0) -> tuple[float, str, float]:
    """Exact add-drop ring at its cold resonance from the realized per-pass power couplings K1^2, K2^2:
    through minimum T = (t1 - a t2)^2 / (1 - a t1 t2)^2 and drop peak D = K1^2 K2^2 a / (1 - a t1 t2)^2, with
    t = sqrt(1 - K^2) and a the round-trip amplitude. Returns (notch in dB, 'under' if t1 > a t2 else 'over', D)."""
    o = pp.optics
    k1, k2 = float(o.kappa2_through[b, i]), float(o.kappa2_drop[b, i])
    a = float(o.round_trip_amplitude[b, i])
    t1, t2 = math.sqrt(1.0 - k1), math.sqrt(1.0 - k2)
    den = (1.0 - a * t1 * t2) ** 2
    return -10.0 * math.log10((t1 - a * t2) ** 2 / den), ("under" if t1 > a * t2 else "over"), k1 * k2 * a / den


@pytest.mark.parametrize("alts, regime, tol_db", [
    ({}, "under", CAL2_SIGMA_DB),
    ({"uncertainty.coupling_regime": "over"}, "over", CAL2_SIGMA_DB),
    ({"uncertainty.film_thickness": "200"}, "under", 1.0),
    ({"uncertainty.film_thickness": "200", "uncertainty.coupling_regime": "over"}, "over", 1.0),
])
def test_g25_ring_reproduces_the_cal2_notch_in_its_labelled_regime(alts, regime, tol_db):
    """Coupling regime: at the nominal Q_L both coupling branches (under- and over-coupled) give the G25 27.9 dB notch
    at 0 V. The 200 nm film world (the rib200 artefact) keeps the measured Q_i (rib200_q_anchor) instead of turning
    both branches over-coupled."""
    pp = realize(load("g25_test_vehicle", alts=alts) if alts else load("g25_test_vehicle"), 1, seeds=0,
                 uncertainty="nominal", variation="off")
    notch, got, _ = _exact_ring(pp)
    assert got == regime, (alts, notch, got)
    assert abs(notch - G25_NOTCH_DB) <= tol_db, (alts, notch)


def test_eta_choices_keep_q_intrinsic_and_hit_their_label():
    """The eta hypothesis moves only the heating share of a fixed intrinsic loss alpha_i (the loss budget fixes
    alpha_i). So Q_i, hence the notch and the regime, is the same in every eta world, and each choice realizes its
    labelled heat fraction."""
    labels = {"nominal": 0.848, "low": 0.5, "mid_low": 0.65, "mid": 0.75, "high": 0.95, "stress_0p3": 0.3}
    base = realize(load("g25_test_vehicle"), 1, seeds=0, uncertainty="nominal", variation="off")
    entry = load("g26").uncertainty["self_heating.eta"]
    assert set(_hypothesis_choices(entry)) == set(labels)
    for c, eta in labels.items():
        pp = realize(load("g25_test_vehicle", alts={"uncertainty.self_heating.eta": c}), 1, seeds=0,
                     uncertainty="nominal", variation="off")
        assert abs(float(pp.optics.heat_fraction[0, 0]) - eta) < 0.005, (c, float(pp.optics.heat_fraction[0, 0]))
        np.testing.assert_allclose(pp.optics.q_intrinsic, base.optics.q_intrinsic, rtol=1e-9, err_msg=c)


def test_monitor_current_is_consistent_with_cal8_and_does_not_clip():
    """Drop coupler: the nominal drop coupling reproduces the ~15.5 uA G26 monitor current at a half-maximum lock
    (+4 dBm at the G25 ring), and the drop PEAK of every nominal g26 ring stays below the ADC full scale, so a
    peak-search acquisition (PAT-3) does not saturate in the nominal world."""
    pp = realize(load("g25_test_vehicle"), 1, seeds=0, uncertainty="nominal", variation="off")
    _, _, drop = _exact_ring(pp)
    p_in = float(pp.laser.power_w[0, 0] * pp.laser.input_transmission[0, 0])
    i_peak = drop * p_in * float(pp.monitor.responsivity_a_per_w[0, 0]) * float(pp.monitor.gain_scale[0, 0])
    assert CAL8_UA[0] <= 0.5e6 * i_peak <= CAL8_UA[1], 1e6 * i_peak
    pp = realize(load("g26"), 1, seeds=0, uncertainty="nominal", variation="off")
    for i in range(pp.optics.q_loaded.shape[1]):
        line = int(pp.layout.ring_to_line[i])
        p_in = float(pp.laser.power_w[0, line] * pp.laser.input_transmission[0, 0])   # no bus loss: conservative
        i_peak = _exact_ring(pp, 0, i)[2] * p_in * float(pp.monitor.responsivity_a_per_w[0, i])
        assert i_peak < float(pp.monitor.adc_full_scale_a[0, 0]), (i, 1e6 * i_peak)


def _reachable_choices(entry) -> list[str]:
    """Hypothesis choices a world can take: the pinned or restricted ones, else those with a positive weight."""
    if entry.pinned:
        return [str(entry.pinned)]
    if entry.restrict:
        return [str(c) for c in entry.restrict]
    w = {str(k): float(v) for k, v in (entry.fields.get("weights") or {}).items()}
    return [c for c in _hypothesis_choices(entry) if w.get(c, 1.0) > 0]


def _interval_max(entry) -> float:
    """Largest value an interval entry takes in any world (its pinned value, which the loader stores as a number or a
    name, else its high end)."""
    f = entry.fields
    if entry.pinned is not None and entry.pinned != "":
        if isinstance(entry.pinned, (int, float)) and not isinstance(entry.pinned, bool):
            return float(entry.pinned)
        named = {"low": f.get("low"), "high": f.get("high"), "nominal": f.get("nominal")}
        named.update(f.get("corners") or {})
        return float(named[str(entry.pinned)])
    return float(f["high"])


@pytest.mark.parametrize("name", PRESETS)
def test_reset_ramp_covers_ten_slow_thermal_poles(name):
    """Start-up history: the laser_ramp reset turns the lasers on over at least 10 tau_slow, in every world the preset
    can draw (each reachable heater family at the largest slow time scale), so the slow thermal pole is settled."""
    cfg = load(name)
    if cfg.get("sim.reset_protocol") != "laser_ramp":
        pytest.skip("not a laser_ramp reset")
    ramp = float(cfg.get("sim.reset_ramp_s"))
    ts = _interval_max(cfg.uncertainty["thermal.time_scale_slow"])
    fam = cfg.uncertainty["thermal.family"]
    leaf = cfg.leaves["ring.thermal.tau_slow_s"]
    cmap = pconfig._hypothesis_choice_map(fam)
    for c in _reachable_choices(fam):
        sel = (cmap.get(c) or {}).get("ring.thermal.tau_slow_s")
        if fam.pinned or sel is None:
            tau_slow = float(np.max(leaf.value))             # the config as loaded (a pin already selected it)
        elif isinstance(sel, dict):
            tau_slow = float(np.max(sel["value"]))
        else:
            tau_slow = float(np.max(leaf.alternatives[str(sel)]["value"]))
        assert ramp >= 10.0 * tau_slow * ts, (name, c, ramp, tau_slow, ts)


def test_g25_bench_heater_is_voltage_driven_in_every_world():
    """The G25 heater data (CAL-3, CAL-4) are printed against a bench voltage, so g25_test_vehicle pins the heater
    driver-law hypothesis (voltage, current, power or DAC + PWM drive) to voltage (code 0) while g26 keeps it open."""
    pp = realize(load("g25_test_vehicle"), 32, seeds=5, uncertainty="lhs", variation="off")
    assert np.all(np.asarray(pp.heater.driver_mode) == 0)
    pp = realize(load("g26"), 32, seeds=5, uncertainty="lhs", variation="off")
    assert len(np.unique(np.asarray(pp.heater.driver_mode))) > 1


def test_geometry_variation_reproduces_the_berkeley_resonance_spreads():
    """Geometry variation: width and film draws through the Jacobian give the 6 nm die-level and 2 nm ring-level
    resonance spreads the preset is sized to (textbook fabOffsetGlobalNm / fabOffsetLocalNm, BER-FAB)."""
    pp = realize(load("g26"), 512, seeds=21, uncertainty="nominal", variation="sample")
    off = np.asarray(pp.optics.resonance_offset_m) * 1e9
    n = off.shape[1]
    die = float(np.std(off.mean(axis=1)))
    ring = float(np.std(off - off.mean(axis=1, keepdims=True)) * math.sqrt(n / (n - 1)))
    assert abs(die - 6.0) < 0.6 and abs(ring - 2.0) < 0.2, (die, ring)


def _corner_checks():
    """Per corner: (description, function of the realized PlantParams returning the per-world values, expected)."""
    v = lambda path: (lambda pp: np.asarray(pp.values[path], float))          # noqa: E731 (SI value by path)
    return {
        "g26_h86": [("s0 [pm/K]", lambda pp: np.asarray(pp.thermal.dlambda_dt_s0_m_per_k) * 1e12, 86.0),
                    ("beta", lambda pp: np.asarray(pp.thermal.dlambda_dt_beta_per_k), 0.0)],
        "g26_h86_sensor": [("reference_rise_frac", v("ring.thermal.reference_rise_frac"), 0.8)],
        "g26_hot_selfheat": [("eta", lambda pp: np.asarray(pp.optics.heat_fraction), 0.95),
                             ("z_ratio", lambda pp: np.asarray(pp.thermal.z_ratio_opt_heater), 2.0),
                             ("P at the ring [dBm]", lambda pp: 10 * np.log10(
                                 np.asarray(pp.laser.power_w) * np.asarray(pp.laser.input_transmission) * 1e3), 8.0)],
        "g26_low_eta": [("eta", lambda pp: np.asarray(pp.optics.heat_fraction), 0.3)],
        "g26_fsr_window_a": [("FSR [THz]", lambda pp: np.asarray(pp.optics.fsr_hz) * 1e-12, 1.47)],
        "g26_fsr_edge": [("FSR [THz]", lambda pp: np.asarray(pp.optics.fsr_hz) * 1e-12, 1.53)],
        "g26_slow_thermal": [("time_scale_fast", lambda pp: pp.thermal.inputs.scales["time_scale_fast"], 2.0),
                             ("time_scale_slow", lambda pp: pp.thermal.inputs.scales["time_scale_slow"], 2.0)],
        "g26_fast_thermal": [("time_scale_fast", lambda pp: pp.thermal.inputs.scales["time_scale_fast"], 0.5),
                             ("time_scale_slow", lambda pp: pp.thermal.inputs.scales["time_scale_slow"], 0.5)],
        "g26_undercut": [("tau_fast [us]", lambda pp: pp.thermal.inputs.interim["tau_fast_s"] * 1e6, 6.08),
                         ("tau_slow [us]", lambda pp: pp.thermal.inputs.interim["tau_slow_s"] * 1e6, 45.67),
                         ("tcr", lambda pp: np.asarray(pp.heater.tcr_per_k), 2.0e-3)],
        "g26_overcoupled": [("Q_L", lambda pp: np.asarray(pp.optics.q_loaded), 3850.0),
                            ("over-coupled", lambda pp: np.asarray(
                                [_exact_ring(pp, b, i)[1] == "over" for b in range(pp.optics.q_loaded.shape[0])
                                 for i in range(pp.optics.q_loaded.shape[1])], float), 1.0)],
        "g26_heater_loss": [("dln_alpha/dT_heater", lambda pp: np.asarray(pp.optics.dln_alpha_dt_heater_per_k),
                             1.6e-3)],
        "g26_textbook_thermal": [("tau_fast [us]", lambda pp: pp.thermal.inputs.interim["tau_fast_s"] * 1e6, 10.0),
                                 ("share_fast", lambda pp: pp.thermal.inputs.interim["share_fast_frac"], 0.75),
                                 ("tau_slow [us]", lambda pp: pp.thermal.inputs.interim["tau_slow_s"] * 1e6, 300.0),
                                 ("xt_nearest", lambda pp: pp.thermal.inputs.interim["xt_nearest_frac"], 0.1),
                                 ("tail share", lambda pp: pp.thermal.inputs.scales["package_tail_share_frac"], 0.0)],
        "g26_tec_package": [("xt_nearest", lambda pp: pp.thermal.inputs.interim["xt_nearest_frac"], 0.0102),
                            ("xt_floor", lambda pp: pp.thermal.inputs.interim["xt_floor_frac"], 0.0026),
                            ("site_floor", lambda pp: pp.thermal.inputs.interim["site_floor_frac"], 0.0009),
                            ("xt_tau_ref [us]", lambda pp: pp.thermal.inputs.interim["xt_tau_ref_s"] * 1e6, 345.0)],
    }


def test_every_corner_has_a_pin_check():
    """_corner_checks has an entry for every corner preset."""
    assert set(_corner_checks()) == set(CORNERS)


@pytest.mark.parametrize("name", sorted(_corner_checks()))
def test_corner_pins_what_it_says_in_every_world(name):
    """A corner must hold its stated values in every sampled world, not only in the nominal one."""
    pp = realize(load(name), 8, seeds=7, uncertainty="lhs", variation="off")
    for what, fn, want in _corner_checks()[name]:
        got = np.asarray(fn(pp), float)
        np.testing.assert_allclose(got, want, rtol=2e-3, atol=1e-12, err_msg=f"{name}: {what}")


# -------------------------------------------------------------------------------------------------- adapter drift guard

def _mini_with_n_g(spec: str) -> dict:
    """tests/data/mini.yaml with ring.optics.n_g read from artefact ``spec`` (and a fallback alternative of 4.0)."""
    raw = pconfig._read_yaml(ROOT / "tests" / "data" / "mini.yaml")      # the loader's YAML (1.0e6 is a float)
    raw["ring"]["optics"]["n_g"] = {
        "value": {"from": spec}, "provenance": "solver(assumed-geometry)", "generation": "none", "source": "R01",
        "range": [3.8, 4.4],
        "alternatives": {FALLBACK: {"value": 4.0, "provenance": "inferred", "generation": "none", "source": "R01"}}}
    return raw


@pytest.mark.skipif(loader_has_fallback(), reason="the loader falls back itself; no adapter is installed")
def test_adapter_equals_the_loader_except_for_the_fallback():
    """The test-local adapter is a copy of config._make_leaf plus the artefact fallback. It must
    give the same Config as the real loader whenever no artefact is missing, and the fallback only for a missing
    artefact file, so the preset tests cannot pass on stale loader logic."""
    present = _mini_with_n_g("extracted/modes_strip400x200.json:n_g")
    missing = _mini_with_n_g("extracted/modes_does_not_exist.json:n_g")

    def raw_load(d):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pconfig.load(d)
    try:
        pconfig._make_leaf = _ORIG_MAKE_LEAF
        a = raw_load(present)
        with pytest.raises(pconfig.ConfigError):
            raw_load(missing)
        pconfig._make_leaf = patched_make_leaf
        b = raw_load(present)
        c = raw_load(missing)
    finally:
        pconfig._make_leaf = patched_make_leaf
    assert a.hash() == b.hash()
    assert {p: lf.value for p, lf in a.leaves.items()} == {p: lf.value for p, lf in b.leaves.items()}
    lf = c.leaves["ring.optics.n_g"]
    assert lf.value == 4.0 and lf.selected_alt == FALLBACK and lf.provenance == "inferred"
    assert any("modes_does_not_exist" in w for w in c.warnings)


@pytest.mark.skipif(loader_has_fallback(),
                    reason="the loader falls back itself: tests/verification/test_params_cli.py checks the file")
def test_parameters_md_is_current_under_the_fallback():
    """Staleness check of docs/PARAMETERS.md for a loader without the fallback: the committed file must equal
    `mrm params table g26` as the fallback adapter renders it (so the doc is never silently stale)."""
    from mrm.params.table import table
    doc = ROOT / "docs" / "PARAMETERS.md"
    assert doc.read_text() == table(load("g26")), "regenerate: mrm params table g26 --out docs/PARAMETERS.md"


@pytest.mark.slow
@pytest.mark.parametrize("name", PRESETS)
def test_preset_realizes_a_monte_carlo_batch(name):
    """Controllers are judged over 256-512 Latin-hypercube worlds with device variation, so every preset must realize
    such a batch: the self-heating lag bound (lag gain < 0.05), the finite checks and the relations hold in every
    world. A 16-world batch once missed the textbook corner's lag gain of 0.050 (with its former 20 us crosstalk
    pole)."""
    pp = realize(load(name), 512, seeds=101, uncertainty="lhs", variation="sample")
    assert _max_residual(pp) <= 1e-12
    lag = pp.diagnostics.get("selfheat_lag_gain")
    assert lag is None or float(np.max(lag)) < 0.05
