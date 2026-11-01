"""The registry of runtime values (mrm/sim/state_spec.py) agrees with the code, and the generated variable map
(docs/VARIABLES.md, docs/variables/data.json, docs/variables/symbols.json; tools/gen_variable_map.py) is current.

Each test names what to change when it fails. Regenerate the map with ``.venv/bin/python tools/gen_variable_map.py``.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import re
import warnings
from pathlib import Path

import pytest

from mrm.sim import state_spec

ROOT = Path(__file__).resolve().parents[2]                   # repository root
_KATEX_DIR = ROOT / "tools" / "katex"                        # pinned KaTeX / MathJax for the renderer checks
REGEN = ".venv/bin/python tools/gen_variable_map.py"         # the command that regenerates the map
# a kernel index constant: block prefix (RF, RI, R8, RB, WI, WF, LF) + "_" + row name
_CONST_RX = re.compile(r"\b(?:RF|RI|R8|RB|WI|WF|LF)_[A-Z0-9_]*[A-Z0-9]\b")


def _gen():
    """The module tools/gen_variable_map.py, loaded from its file (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location("gen_variable_map", ROOT / "tools" / "gen_variable_map.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sources(*globs):
    """{path: text} of every file matching the globs, except state_spec.py and __pycache__."""
    out = {}
    for g in globs:
        for p in sorted(ROOT.glob(g)):
            if p.name != "state_spec.py" and "__pycache__" not in p.parts:
                out[p] = p.read_text()
    return out


def _plant_call_keywords(name):
    """Keyword names and expressions of the kernel.<name>(...) call in Plant._build (mrm/plant/plant.py)."""
    tree = ast.parse((ROOT / "mrm" / "plant" / "plant.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == name
                and isinstance(node.func.value, ast.Name) and node.func.value.id in ("K", "kernel")):
            return {kw.arg: kw.value for kw in node.keywords}
    raise AssertionError(f"no kernel.{name}(...) call in mrm/plant/plant.py")


# ------------------------------------------------------------------------------------------------ (a) code <-> registry

def test_kernel_layout_comes_from_the_registry():
    """The kernel's row names, index constants and namedtuple fields are the registry's (nothing redefined)."""
    from mrm.sim import kernel
    for b in state_spec.BLOCKS:
        short = b.prefix[:-1]                       # RF, RI, ... (the index-constant prefix without "_")
        assert getattr(kernel, f"_{short}_NAMES") == b.names
        assert getattr(kernel, short) == b.index
    assert kernel.KParams._fields == state_spec.KPARAMS_NAMES
    assert kernel.KState._fields == state_spec.KSTATE_NAMES
    for name, value in state_spec.index_constants().items():
        assert getattr(kernel, name) == value, name
    assert kernel._LAYOUT_SIG == state_spec.layout_signature()


def test_compiled_kernel_has_the_registry_layout():
    """A kernel compiled (and disk-cached) with another layout would use stale index constants: run
    tools/clear_numba_cache.py after changing the order or set of names in mrm/sim/state_spec.py."""
    from mrm.sim import kernel
    assert kernel._layout_sig() == state_spec.layout_signature(), "stale numba cache: run tools/clear_numba_cache.py"


@pytest.mark.parametrize("what, names", [("KParams", state_spec.KPARAMS_NAMES), ("KState", state_spec.KSTATE_NAMES)])
def test_plant_builds_exactly_the_registry_fields(what, names):
    """Every field Plant._build fills has a registry entry and every registry entry is filled."""
    kws = set(_plant_call_keywords(what))
    assert not kws - set(names), f"{what} fields without a registry entry (add them to mrm/sim/state_spec.py)"
    assert not set(names) - kws, f"registry {what} entries that Plant._build does not fill"


def test_every_index_constant_in_the_code_is_registered():
    """The kernel's own uses (bare names in kernel.py) and every ``K.X`` / ``kernel.X`` use elsewhere."""
    consts = state_spec.index_constants()
    bad = {}
    for p, text in _sources("mrm/**/*.py", "tools/*.py", "tests/**/*.py").items():
        if p.name == "kernel.py":
            used = set(_CONST_RX.findall(text))
        else:
            used = {m.split(".", 1)[1]
                    for m in re.findall(r"\b(?:K|kernel)\.(?:RF|RI|R8|RB|WI|WF|LF)_[A-Z0-9_]*[A-Z0-9]\b", text)}
        for c in used - set(consts):
            bad.setdefault(c, str(p.relative_to(ROOT)))
    assert not bad, f"index constants that the registry does not define: {bad}"


def test_unused_rows_are_unused_and_every_other_row_is_used():
    """kind 'unused' means no code reads or writes the row; every other row is referenced by code."""
    texts = _sources("mrm/**/*.py")
    for b in state_spec.BLOCKS:
        for f in b.fields:
            c = b.const(f)
            n = sum(len(re.findall(r"\b" + c + r"\b", t)) for t in texts.values())
            if f.kind == "unused":
                assert n == 0, f"{b.key}.{f.name} is marked unused but {c} appears in the code: update its entry"
            else:
                assert n > 0, f"{b.key}.{f.name} ({c}) is not referenced by any code: mark it unused or remove it"
    for a in state_spec.KSTATE:
        if a.kind == "unused":
            n = sum(len(re.findall(r"\." + a.name + r"\b", t)) for t in texts.values())
            assert n == 0, f"S.{a.name} is marked unused but the code reads it"


# -------------------------------------------------------------------------------------------------------------- entries

def _all_entries():
    """(id, meaning, unit, kind, updated_by, depends_on) of every registry entry: block rows, KState fields outside a
    block, and the self-heating rows."""
    for b in state_spec.BLOCKS:
        for f in b.fields:
            yield f"{b.key}.{f.name}", f.meaning, f.unit, f.kind, f.updated_by, f.depends_on
    for a in state_spec.KSTATE:
        if a.block is None:
            yield f"S.{a.name}", a.meaning, a.unit, a.kind, a.updated_by, a.depends_on
    for r in state_spec.SH_ROWS:
        yield f"selfheat.{r.name}", r.meaning, r.unit, r.kind, r.updated_by, r.depends_on


def _resolve(fn: str):
    """The object named by 'module_key.attr[.attr]' (module_key from state_spec.MODULES)."""
    mod, _, rest = fn.partition(".")
    obj = importlib.import_module(state_spec.MODULES[mod])
    for part in rest.split("."):
        obj = getattr(obj, part)
    return obj


def test_every_entry_is_complete_and_names_real_code():
    """Meaning, unit and kind are set; updated_by names a step section and functions that exist."""
    for eid, meaning, unit, kind, by, _deps in _all_entries():
        assert kind in state_spec.KINDS, f"{eid}: kind {kind!r}"
        assert isinstance(unit, str), eid
        if not eid.startswith("selfheat."):
            assert meaning and meaning.strip(), f"{eid}: no meaning"
        sec, _, fns = by.partition(":")
        assert sec.strip() in state_spec.SECTION_KEYS, f"{eid}: section {sec!r} not in state_spec.SECTIONS"
        if kind == "unused":
            assert sec.strip() == "never", f"{eid}: an unused row has no writer"
            continue
        if eid.startswith(tuple(b.key + "." for b in state_spec.BLOCKS)) or not by.startswith("build"):
            assert fns.strip(), f"{eid}: updated_by names no function"
        for fn in (x.strip() for x in fns.split(",") if x.strip()):
            if fn.startswith("controller."):
                continue
            try:
                _resolve(fn)
            except (AttributeError, KeyError, ImportError) as e:
                raise AssertionError(f"{eid}: updated_by function {fn} does not exist ({e})") from None
    for p in state_spec.KPARAMS:
        assert p.meaning and p.src, f"P.{p.name}: meaning and sources are required"


def _known_ids():
    """Every value id of the model: registry rows, KState and KParams fields, Truth fields, actuators, PlantParams."""
    from mrm.core import channels
    from mrm.sim import metrics
    ids = {f"{b.key}.{f.name}" for b in state_spec.BLOCKS for f in b.fields}
    ids |= {f"S.{a.name}" for a in state_spec.KSTATE} | {f"selfheat.{r.name}" for r in state_spec.SH_ROWS}
    ids |= {f"P.{n}" for n in state_spec.KPARAMS_NAMES} | {f"truth.{n}" for n in metrics.TRUTH_FIELDS}
    ids |= {f"act.{r['name']}" for r in channels.V1_CHANNEL_TABLE if r["kind"] == "actuator"}
    realize_module = importlib.import_module("mrm.params.realize")
    for f in dataclasses.fields(realize_module.PlantParams):
        cls = getattr(realize_module, f.type if isinstance(f.type, str) else f.type.__name__, None)
        if dataclasses.is_dataclass(cls):
            ids |= {f"pp.{f.name}.{g.name}" for g in dataclasses.fields(cls)}
    return ids


def test_depends_on_names_exist():
    """(b) Every name in depends_on (and in the output maps) is a value of the model, and no entry depends on
    itself."""
    known = _known_ids()
    bad = []
    for eid, _m, _u, _k, _by, deps in _all_entries():
        for d in deps:
            if d not in known or d == eid:
                bad.append((eid, d))
    for table in (state_spec.TRUTH_SOURCES, state_spec.OBS_SOURCES, state_spec.ACT_TARGETS):
        for key, deps in table.items():
            bad += [(key, d) for d in deps.split() if d not in known]
    assert not bad, f"unknown or self dependencies (fix mrm/sim/state_spec.py): {bad}"


def test_self_heating_rows_exist_and_have_a_meaning():
    gen = _gen()
    comments = gen._comments(ROOT / "mrm" / "plant" / "optics.py", r"^([A-Z][A-Z0-9_]*) = \d+\s*#\s*(.*)$")
    for r in state_spec.SH_ROWS:
        mod, _, name = r.const.partition(".")
        assert isinstance(getattr(importlib.import_module(state_spec.MODULES[mod]), name), int), r.const
        assert r.meaning or comments.get(name), f"selfheat.{r.name}: no meaning and no comment on {r.const}"


def test_output_maps_cover_the_channels_and_truth_fields():
    from mrm.core import channels
    from mrm.sim import metrics
    assert set(state_spec.TRUTH_SOURCES) == set(metrics.TRUTH_FIELDS), \
        "state_spec.TRUTH_SOURCES must list every Truth field"
    sensors = {r["name"] for r in channels.V1_CHANNEL_TABLE if r["kind"] == "sensor"}
    actuators = {r["name"] for r in channels.V1_CHANNEL_TABLE if r["kind"] == "actuator"}
    assert set(state_spec.OBS_SOURCES) == sensors, "state_spec.OBS_SOURCES must list every sensor channel"
    assert set(state_spec.ACT_TARGETS) == actuators, "state_spec.ACT_TARGETS must list every actuator channel"


def test_kparam_sources_exist_and_cover_plant_build():
    """pp.* sources are PlantParams fields, cfg.* sources schema keys, unc.* g26 uncertainty entries; every
    pp.<group>.<field> that Plant._build reads directly for a KParams field is among that field's sources."""
    from mrm import params
    from mrm.params import schema
    realize_module = importlib.import_module("mrm.params.realize")
    groups = {}
    for f in dataclasses.fields(realize_module.PlantParams):
        t = f.type if isinstance(f.type, str) else f.type.__name__
        cls = getattr(realize_module, t, None)
        groups[f.name] = {g.name for g in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26")
    for p in state_spec.KPARAMS:
        for s in p.src:
            parts = s.split(".")
            if parts[0] == "pp":
                assert parts[1] in groups, f"P.{p.name}: {s}"
                if len(parts) == 3:
                    assert groups[parts[1]] is not None and parts[2] in groups[parts[1]], f"P.{p.name}: {s}"
            elif parts[0] == "cfg":
                assert s[4:] in schema.SCHEMA, f"P.{p.name}: {s} is not a schema key"
            elif parts[0] == "unc":
                assert s[4:] in cfg.uncertainty, f"P.{p.name}: {s} is not an uncertainty entry"
    for name, expr in _plant_call_keywords("KParams").items():
        reads = set()
        for n in ast.walk(expr):
            if isinstance(n, ast.Attribute):
                chain, cur = [n.attr], n.value
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id == "pp":
                    chain.reverse()
                    reads.add("pp." + ".".join(chain[:2]))
        src = set(dict(zip(state_spec.KPARAMS_NAMES, (p.src for p in state_spec.KPARAMS)))[name])
        missing = {r for r in reads if r not in src and not any(s.startswith(r + ".") for s in src)}
        assert not missing, f"P.{name}: Plant._build reads {sorted(missing)}; add them to its src in state_spec"


# -------------------------------------------------------------------------------------------------- (c) generated files

@pytest.mark.parametrize("name, unit", [
    ("s_lock_m_per_k", "m/K"), ("z_heater_ring_k_per_w", "K/W"), ("heater_efficiency_m_per_w", "m/W"),
    ("responsivity_a_per_w", "A/W"), ("alpha_per_k2", "1/K^2"), ("alpha_per_k", "1/K"), ("t_ring_k", "K"),
    ("ref_frac", "1"), ("n_rings", "")])
def test_unit_from_name_takes_the_longest_suffix(name, unit):
    """A compound unit suffix wins over its tail: ``_m_per_k`` is m/K, not the 1/K of ``_per_k``."""
    assert _gen()._unit_from_name(name) == unit


@pytest.fixture(scope="module")
def model():
    """The variable map of g26 as built by tools/gen_variable_map.py."""
    return _gen().build("g26")


def test_variables_md_is_current(model):
    gen = _gen()
    assert (ROOT / "docs" / "VARIABLES.md").read_text() == gen.render_markdown(model), f"run: {REGEN}"


def test_viewer_data_is_current(model):
    gen = _gen()
    assert (ROOT / "docs" / "variables" / "data.json").read_text() == gen.render_json(model), f"run: {REGEN}"


def _symbols_json():
    """(path, parsed content) of docs/variables/symbols.json."""
    import json
    path = ROOT / "docs" / "variables" / "symbols.json"
    assert path.exists(), f"docs/variables/symbols.json is missing; run: {REGEN}"
    return path, json.loads(path.read_text())


def test_box_symbol_svgs_cover_the_map(model):
    """docs/variables/symbols.json (the typeset symbol of each box) has one SVG for every distinct symbol of the map
    and no other, made with the MathJax version that tools/katex/package.json pins (no node needed)."""
    gen = _gen()
    _path, js = _symbols_json()
    have = [x["tex"] for x in js["symbols"]]
    want = gen.symbol_texts(model)
    missing, extra = sorted(set(want) - set(have)), sorted(set(have) - set(want))
    assert not missing and not extra, (f"symbols.json is stale (no SVG for {missing}; SVG of no symbol {extra}); "
                                       f"run: {REGEN}")
    assert have == want, f"symbols.json is not in the generator's order; run: {REGEN}"
    assert js["meta"]["version"] == gen.mathjax_version(), f"symbols.json was made with another MathJax; run: {REGEN}"
    n_glyphs = len(js["glyphs"])
    for x in js["symbols"]:
        refs = [int(i) for i in re.findall(r'<path data-g="(\d+)"></path>', x["svg"])]
        assert x["svg"].startswith("<svg ") and "currentColor" in x["svg"] and refs, f"bad SVG of {x['tex']!r}"
        assert all(0 <= i < n_glyphs for i in refs), f"SVG of {x['tex']!r} names a missing glyph; run: {REGEN}"
        assert x["w"] > 0 and x["h"] > 0, x["tex"]


def test_box_symbol_svgs_are_current(model):
    """A fresh MathJax rendering of every symbol (node, ``npm ci --prefix tools/katex``) equals symbols.json."""
    import shutil
    if shutil.which("node") is None or not (_KATEX_DIR / "node_modules" / "mathjax-full").is_dir():
        pytest.skip("node or tools/katex/node_modules missing (run: npm ci --prefix tools/katex)")
    gen = _gen()
    path, _js = _symbols_json()
    assert path.read_text() == gen.render_symbols_json(gen.render_symbol_svgs(model)), (
        f"docs/variables/symbols.json differs from a fresh MathJax rendering; run: {REGEN}")


def test_map_is_connected_from_the_constants(model):
    """Every frozen parameter and every per-tick value that code writes reaches back to the model's inputs or to a
    run setting (no orphan), except the listed constants of Plant._build."""
    nodes = {n["id"]: n for n in model["nodes"]}
    pred = {}
    for a, b, _t in model["edges"]:
        pred.setdefault(b, []).append(a)
    for nid in ("P.he_r0", "P.th_a", "ring_float.t_ring", "ring_float.det", "obs.tx.monitor.samples",
                "metric.max_abs_detuning_pm"):
        seen, stack = set(), [nid]
        while stack:
            x = stack.pop()
            for y in pred.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        assert any(nodes[s]["layer"] == 1 for s in seen), f"{nid} does not reach a preset input"


# ------------------------------------------------------------------------------------------------ (d) symbols, formulas

def _branch_tex() -> list:
    """Every LaTeX formula that mrm/params/relations.py (``ctx.set(..., tex=...)``) and mrm/params/realize.py
    (``Derivation(..., formula_tex=...)``) can record, in every branch: each tex expression is evaluated with the
    module's _TEX_* constants and SYMBOLS, the tex variables of its function (``formula_tex, eq = r"...", ...``) and
    every value its branch variables (the Q member solved, the coupler) can take."""
    from mrm.params import relations
    from mrm.params.symbols import SYMBOLS
    qs = [relations.P_QI, relations.P_QT, relations.P_QD]
    samples = {"port_letter": ["t", "d"], "f": qs, "other": [relations.P_QT, relations.P_QI],
               "others": [[q for q in qs if q != f] for f in qs],
               "t": ["derived.geometry.dln_kappa2_through", "derived.geometry.dln_kappa2_drop"]}
    out = []
    for path, kw in (("mrm/params/relations.py", "tex"), ("mrm/params/realize.py", "formula_tex")):
        tree = ast.parse((ROOT / path).read_text())
        env = {"SYMBOLS": SYMBOLS}
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.startswith("_TEX_")):
                env[node.targets[0].id] = ast.literal_eval(node.value)
        for node in ast.walk(tree):
            # tex variables of a branch: formula_tex, eq = r"...", "..."
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple) and isinstance(node.value,
                                                                                                    ast.Tuple):
                for t, v in zip(node.targets[0].elts, node.value.elts):
                    if isinstance(t, ast.Name) and t.id.endswith("tex"):
                        out.append(ast.literal_eval(v))
            if not isinstance(node, ast.Call):
                continue
            for k in node.keywords:
                if k.arg != kw or (isinstance(k.value, ast.Name) and k.value.id.endswith("tex")):
                    continue
                code = compile(ast.Expression(k.value), path, "eval")
                names = {n.id for n in ast.walk(k.value) if isinstance(n, ast.Name)} & set(samples)
                combos = [{}]
                for nm in sorted(names):
                    combos = [dict(c, **{nm: v}) for c in combos for v in samples[nm]]
                for c in combos:
                    out.append(eval(code, dict(env, **c)))  # noqa: S307 - our own source, constants only
    return out


def _model_tex(model) -> list:
    """(node id, field, LaTeX string) for every symbol and formula of the map."""
    return [(n["id"], k, n[k]) for n in model["nodes"] for k in ("symbol", "formula_tex") if n.get(k)]


def test_every_symbol_and_formula_passes_the_strict_tex_check(model):
    """Balanced braces, only known KaTeX commands, no '|' or '$' (Markdown tables): mrm.params.symbols.check_tex."""
    from mrm.params import symbols
    strings = [(i, k, t) for i, k, t in _model_tex(model)] + [("relations/realize", "formula_tex", t)
                                                              for t in _branch_tex()]
    tables = [symbols.SYMBOLS, symbols.PLANTPARAMS_SYMBOLS, state_spec.TRUTH_SYMBOLS, state_spec.OBS_SYMBOLS,
              state_spec.ACT_SYMBOLS, state_spec.METRIC_SYMBOLS]
    strings += [(k, "symbol", t) for tb in tables for k, t in tb.items()]
    bad = [(i, k, t, symbols.check_tex(t)) for i, k, t in strings if symbols.check_tex(t)]
    assert not bad, f"LaTeX that would not render (fix mrm/params/symbols.py or the tex= of the formula): {bad}"


def test_every_symbol_and_formula_renders_in_katex(model):
    """Render every string with KaTeX 0.16.11 (the viewer's version; strict mode). Needs node and
    ``npm ci --prefix tools/katex``."""
    import json
    import shutil
    import subprocess
    if shutil.which("node") is None or not (_KATEX_DIR / "node_modules" / "katex").is_dir():
        pytest.skip("node or tools/katex/node_modules missing (run: npm ci --prefix tools/katex)")
    strings = sorted({t for _i, _k, t in _model_tex(model)} | set(_branch_tex()))
    r = subprocess.run(["node", str(_KATEX_DIR / "render.js")], input=json.dumps(strings), capture_output=True,
                       text=True, timeout=120, check=True)
    res = json.loads(r.stdout)
    assert res["version"] == "0.16.11", "tools/katex must pin the KaTeX version that docs/variables/index.html loads"
    assert not res["errors"], f"KaTeX cannot render: {res['errors']}"
    html = (ROOT / "docs" / "variables" / "index.html").read_text()
    assert "KaTeX/0.16.11/katex.min.js" in html and "KaTeX/0.16.11/katex.min.css" in html


def test_every_derived_value_has_a_latex_formula(model):
    """Every derived constant of every preset carries formula_tex, every ctx.set of relations.py passes tex= and
    every Derivation(...) of realize.py passes formula_tex= (so no branch can record a formula without one)."""
    import glob
    from mrm import params
    missing = [n["id"] for n in model["nodes"] if n["layer"] == 2 and not n.get("formula_tex")]
    assert not missing, f"derived values without formula_tex: {missing}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in sorted(glob.glob(str(ROOT / "mrm" / "params" / "presets" / "*.yaml"))):
            name = Path(f).stem
            ders = params.derive(params.load(name)).params.derivations
            bad = [p for p, d in ders.items() if not d.formula_tex]
            assert not bad, f"{name}: derivations without formula_tex: {bad}"
    for path, call, kw in (("mrm/params/relations.py", "set", "tex"),
                           ("mrm/params/realize.py", "Derivation", "formula_tex")):
        for node in ast.walk(ast.parse((ROOT / path).read_text())):
            if isinstance(node, ast.Call):
                fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if fn == call and (call != "set" or getattr(node.func.value, "id", "") == "ctx"):
                    assert any(k.arg == kw for k in node.keywords), f"{path}:{node.lineno}: {call}(...) without {kw}="


def test_formula_names_the_symbol_of_its_value(model):
    """A derived value's LaTeX formula contains the value's own symbol (the formula and the symbol table agree)."""
    bad = [(n["id"], n["symbol"]) for n in model["nodes"]
           if n["layer"] == 2 and n.get("symbol") and n["symbol"] not in n.get("formula_tex", "")]
    assert not bad, f"formula_tex does not contain the symbol of its value (mrm/params/symbols.py): {bad}"


def test_symbols_do_not_mislead(model):
    """Two values share a symbol only when an edge of the map links them (the same quantity passed on), or when
    mrm.params.symbols.INTENTIONAL_SHARED says why; every INTENTIONAL_SHARED entry is still needed."""
    from collections import defaultdict
    from mrm.params.symbols import INTENTIONAL_SHARED
    by = defaultdict(set)
    for n in model["nodes"]:
        if n.get("symbol"):
            by[n["symbol"]].add(n["id"])
    adj = defaultdict(set)
    for a, b, _t in model["edges"]:
        adj[a].add(b)
        adj[b].add(a)
    shared = {}
    for sym, ids in by.items():
        comps, seen = [], set()
        for i in sorted(ids):
            if i in seen:
                continue
            comp, stack = set(), [i]
            while stack:
                x = stack.pop()
                if x not in comp:
                    comp.add(x)
                    stack += [y for y in adj[x] if y in ids]
            seen |= comp
            comps.append(sorted(comp))
        if len(comps) > 1:
            shared[sym] = comps
    unmarked = {s: c for s, c in shared.items() if s not in INTENTIONAL_SHARED}
    assert not unmarked, ("one symbol for unlinked values: give one of them a distinct symbol, or add the symbol to "
                          f"INTENTIONAL_SHARED with the reason (mrm/params/symbols.py): {unmarked}")
    stale = sorted(set(INTENTIONAL_SHARED) - set(shared))
    assert not stale, f"INTENTIONAL_SHARED entries that no unlinked values share any more (remove them): {stale}"


def test_symbol_tables_name_real_values(model):
    """Every key of the symbol tables is a value of the model, and every node label text is plain (no LaTeX)."""
    from mrm.core import channels
    from mrm.params import schema
    from mrm.params import symbols
    from mrm.sim import metrics
    src = (ROOT / "mrm" / "params" / "relations.py").read_text() + (ROOT / "mrm" / "params" / "realize.py").read_text()
    bad = [k for k in symbols.SYMBOLS if k not in schema.SCHEMA and not (k.startswith("derived.") and f'"{k}"' in src)]
    ids = {n["id"] for n in model["nodes"]}
    bad += [k for k in symbols.PLANTPARAMS_SYMBOLS if ("pp." + k) not in ids]
    bad += [k for k in state_spec.TRUTH_SYMBOLS if k not in metrics.TRUTH_FIELDS]
    names = {r["name"] for r in channels.V1_CHANNEL_TABLE}
    bad += [k for k in list(state_spec.OBS_SYMBOLS) + list(state_spec.ACT_SYMBOLS) if k not in names]
    bad += [k for k in state_spec.METRIC_SYMBOLS if ("metric." + k) not in ids]
    assert not bad, f"symbol-table keys that name no value: {bad}"
    for n in model["nodes"]:
        if n.get("symbol"):
            t = n.get("symbol_text", "")
            assert t and not set(t) & set("\\{}"), f"{n['id']}: plain label {t!r} of {n['symbol']!r}"
