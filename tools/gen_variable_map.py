"""Generate the map of every value of the model, from the preset constants to the values that change every tick.

    .venv/bin/python tools/gen_variable_map.py             # write docs/VARIABLES.md, docs/variables/*.json
    .venv/bin/python tools/gen_variable_map.py --check     # exit 1 when a committed file differs from a fresh build

Read this first. The map is a graph: each node is one value of the model, and each edge points from a value to
a value computed from it. build() makes the graph; render_markdown(), render_json() and render_symbols_json() write
the three files. The tool also typesets every distinct symbol to SVG with MathJax (tools/katex/tex2svg.js; needs
node and ``npm ci --prefix tools/katex``) for docs/variables/symbols.json: the viewer draws each one inside its
box. Both commands fail when they cannot render the symbols.

The map has five layers, top to bottom:

1. inputs: the leaves and settings of the g26 preset (value, unit, provenance, source; mrm.params.load), with the
   uncertainty and variation draws that change them marked on them, and the uncertainty entries that have no target
   leaf;
2. derived constants: the derivations of mrm.params.realize (relation R1-R5, formula, inputs; mrm.params.derive);
3. frozen per-run parameters: the PlantParams fields (their config sources read from mrm/params/realize.py with the
   ast module) and the KParams fields of the kernel (mrm/sim/state_spec.py KPARAM_GROUPS);
4. per-tick state: the rows and arrays of KState and the self-heating rows (state_spec), grouped by the part of the
   plant step that writes them;
5. outputs: the observation channels and commands of mrm.core.channels, Truth (mrm.sim.metrics.TRUTH_FIELDS) and the
   summary metrics (mrm.sim.metrics.summarize), none of which a controller sees except the observations.

Each value carries its LaTeX symbol where one is natural (mrm/params/symbols.py for the preset, derived and
PlantParams values; state_spec for the runtime values and outputs), its plain-text form (the viewer's search, and
its box label when symbols.json is missing) and, in docs/variables/symbols.json, its typeset SVG (the box label);
each derived constant carries its formula in LaTeX (Derivation.formula_tex) and its textbook equation number.

Nothing here is a hand-written list of values: every node comes from the code (the registry state_spec, the preset,
the realize/derive records, the channel table, the metrics module) and tests/verification/test_variable_map.py fails
when the committed outputs differ from a fresh build. The viewer is docs/variables/index.html (README there).
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MD_PATH = ROOT / "docs" / "VARIABLES.md"
JSON_PATH = ROOT / "docs" / "variables" / "data.json"
SYM_PATH = ROOT / "docs" / "variables" / "symbols.json"
KATEX_DIR = ROOT / "tools" / "katex"
TEX2SVG = KATEX_DIR / "tex2svg.js"
REGEN = ".venv/bin/python tools/gen_variable_map.py"
NPM_CI = "npm ci --prefix tools/katex"

LAYERS = (
    (1, "Inputs", "Leaves and settings of the preset: the constants you set. Uncertainty and variation draws are "
                  "marked on the inputs they change."),
    (2, "Derived constants", "Values that mrm.params.realize computes from the inputs with the relations R1-R5."),
    (3, "Frozen per-run parameters", "PlantParams (realize output, SI arrays) and the kernel parameters KParams "
                                     "(Plant._build). Fixed for a run."),
    (4, "Per-tick state", "Values the kernel writes in each plant step, grouped by the part of the step that writes "
                          "them (event order, top to bottom)."),
    (5, "Outputs", "What the controller sees (observations), what it writes (commands), and Truth and the metrics "
                   "(never seen by a controller)."),
)

# -------------------------------------------------------------------------------------------------------------- helpers

_UNIT_SUFFIX = (("_per_k2", "1/K^2"), ("_per_k", "1/K"), ("_per_m", "1/m"), ("_per_w", "1/W"), ("_m_per_k", "m/K"),
                ("_k_per_w", "K/W"), ("_m_per_w", "m/W"), ("_m_per_v", "m/V"), ("_per_s", "1/s"),
                ("_per_hz", "1/Hz"), ("_a_per_w", "A/W"), ("_ohm", "ohm"), ("_hz", "Hz"), ("_m", "m"), ("_k", "K"),
                ("_s", "s"), ("_w", "W"), ("_v", "V"), ("_a", "A"), ("_f", "F"), ("_j", "J"), ("_frac", "1"),
                ("_ticks", "tick"), ("_codes", "code"), ("_lsb", "LSB"), ("_db", "dB"))


def _unit_from_name(name: str) -> str:
    """The unit that a name's suffix states (``t_ring_k`` -> "K"), or "" when the suffix is not a unit.

    The longest matching suffix in _UNIT_SUFFIX wins, so a compound suffix such as ``_m_per_w`` gives "m/W", not
    the "1/W" of its tail ``_per_w``.
    """
    for suf, u in sorted(_UNIT_SUFFIX, key=lambda pair: -len(pair[0])):
        if name.endswith(suf):
            return u
    return ""


def _value_text(v) -> str:
    """Compact text of a value (mrm.params.table.format_value), never an object repr (no memory addresses)."""
    from mrm.params.table import format_value
    if v is None:
        return ""
    if isinstance(v, (bool, int, float, str, np.integer, np.floating, np.bool_)):
        return format_value(v)
    if isinstance(v, np.ndarray):
        if v.dtype.kind in "biuf":
            return format_value(v.ravel())
        if v.dtype.kind == "c":
            return f"complex array {tuple(v.shape)}"
        return f"array {tuple(v.shape)}"
    if isinstance(v, (list, tuple)):
        if all(isinstance(x, (int, float, str, np.integer, np.floating)) for x in v):
            return format_value(list(v))
        return f"<{len(v)} items>"
    if isinstance(v, dict):
        return format_value({str(k): 0 for k in v})
    return f"<{type(v).__name__}>"


def _comments(path: Path, pattern: str) -> dict:
    """{NAME: comment} of the lines ``NAME = value  # comment`` (or the pattern's two groups) of a source file."""
    rx = re.compile(pattern)
    out = {}
    for line in path.read_text().splitlines():
        m = rx.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


# read a constant's description from its assignment or the comment directly above it.
def _constant_comments(path: Path) -> dict:
    import io
    import tokenize

    source = path.read_text()
    lines = source.splitlines()
    comments = {
        token.start[0]: token.string.removeprefix("#").strip()
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    }
    result = {}
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        inline = next((comments[line] for line in range(node.lineno, node.end_lineno + 1) if line in comments), "")
        leading = comments.get(node.lineno - 1, "") if node.lineno > 1 and lines[node.lineno - 2].lstrip().startswith("#") else ""
        description = inline or leading
        for target in targets:
            if isinstance(target, ast.Name) and re.fullmatch(r"[A-Z][A-Z0-9_]*", target.id):
                result[target.id] = description
    return result


# A design-review code such as "07 P1-10" or "07 P0-7, P1-7" (docs/research/07_design_critique.md).
_REVIEW_CODE = r"07 [PD]\d+(?:-\d+)?(?:, [PD]\d+(?:-\d+)?)*"


def _plain_channel_doc(doc: str) -> str:
    """The doc string of a channel without its design-review codes, for display.

    The doc strings in mrm/core/channels.py keep their codes, because they enter the ControllerSpec fingerprint and the
    committed golden vectors. The map shows the plain text: "(07 P1-10)" is dropped, "(x; 07 P1-10)" becomes "(x)" and
    "(07 P0-6 applied codes)" becomes "(applied codes)"."""
    doc = re.sub(r"\s*\(" + _REVIEW_CODE + r"\)", "", doc)          # the whole parenthesis is a code
    doc = re.sub(r";\s*" + _REVIEW_CODE + r"(?=\))", "", doc)        # "(text; code)" -> "(text)"
    doc = re.sub(r"\(" + _REVIEW_CODE + r",?\s+", "(", doc)          # "(code, text)" or "(code text)" -> "(text)"
    return doc


def _split_updated_by(s: str) -> tuple[str, list[str]]:
    """Split a registry ``updated_by`` string "section: fn1, fn2" into ("section", ["fn1", "fn2"])."""
    sec, _, fns = s.partition(":")
    return sec.strip(), [f.strip() for f in fns.split(",") if f.strip()]


def _md_cell(s) -> str:
    """Text of ``s`` safe for one Markdown table cell ("|" escaped, newlines as spaces; "" for None)."""
    return str(s).replace("|", "\\|").replace("\n", " ") if s is not None else ""


def _md_math(s) -> str:
    """Inline math of a LaTeX string for a Markdown table ("" when none). mrm.params.symbols.check_tex keeps
    ``|`` and ``$`` out of every symbol and formula, so the cell needs no escape."""
    return f"${s}$" if s else ""


# ------------------------------------------------------------------------------------------------------ realize sources

#: names in realize.py that are not followed to a config source: helpers, shapes, modules, seeds and diagnostics
_SKIP_NAMES = {"vals", "cfg", "optional", "per_ring", "per_die", "BN", "B1", "BL", "B", "N", "L", "np", "math",
               "relations", "schema", "self", "element_seeds", "seeds", "diag", "ctx", "residuals", "thermal_module",
               "fields", "_freeze", "float", "int", "bool", "str", "tuple", "dict", "list", "range", "len", "max",
               "min", "abs", "round", "to_si", "rng", "C0", "math"}
#: names of PlantParams group objects inside realize.py: ``timing.adc_period_ticks`` refers to that group's field
_GROUP_VARS = {"timing": "timing", "layout": "layout", "optics": "optics", "thermal": "thermal", "heater": "heater",
               "junction": "junction", "monitor": "monitor", "laser": "laser", "datapath": "datapath",
               "temps": "temp_sensors", "bias": "bias_dac", "flags": "flags"}
_DRAW_FUNCS = {"_uniform", "_normal"}


class _Scope:
    """The assignments, parameters and nested functions of one function of realize.py (nested bodies excluded)."""

    def __init__(self, fn, outer=None, subst=None, call_line=10 ** 9, bind=None):
        self.fn, self.outer, self.subst, self.call_line = fn, outer, dict(subst or {}), call_line
        self.bind = dict(bind or {})              # parameter -> (argument expression, caller scope, call line)
        a = fn.args
        self.params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
        self.assigns = defaultdict(list)          # name -> [(lineno, expr)]
        self.subs = defaultdict(list)             # dict name -> [(key, expr)]
        self.local_funcs = {}
        stack = list(fn.body)
        while stack:
            node = stack.pop()
            if isinstance(node, ast.FunctionDef):
                self.local_funcs[node.name] = node
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if (isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple)
                            and len(t.elts) == len(node.value.elts)):
                        pairs = list(zip(t.elts, node.value.elts))        # a, b = x, y: pairwise
                    else:
                        pairs = [(t, node.value)]
                    for tt, vv in pairs:
                        for nm in ast.walk(tt):
                            if isinstance(nm, ast.Name):
                                self.assigns[nm.id].append((node.lineno, vv))
                    if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                            and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str)):
                        self.subs[t.value.id].append((t.slice.value, node.value))
            elif isinstance(node, (ast.For, ast.comprehension)):
                for nm in ast.walk(node.target):                                   # for x in it: x <- it
                    if isinstance(nm, ast.Name):
                        self.assigns[nm.id].append((getattr(node, "lineno", 0), node.iter))
            elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name)
                  and node.value.func.attr in ("append", "extend", "update", "add")):
                self.assigns[node.value.func.value.id].append((node.lineno, node.value))     # x.append(v)
            stack.extend(ast.iter_child_nodes(node))

    def func(self, name):
        s = self
        while s is not None:
            if name in s.local_funcs:
                return s.local_funcs[name], s
            s = s.outer
        return None, None


def realize_sources(known: set, unc_names: set) -> dict:
    """{"group.field": set of sources} for every PlantParams field, read from the calls that build the groups in
    mrm/params/realize.py (ast). A source is a config path (a string constant of the keyword expression that is a
    preset or schema path, or a ``relations.P_*`` constant), ``unc.<entry>`` (an uncertainty entry name) or
    ``draw:<name>`` (a per-device random draw of realize). Local names are followed to their last assignment
    before the use, helper calls into their bodies (with constant arguments substituted), a function parameter to
    the arguments of its call sites, and ``<group>.<field>`` of a group object (``timing.adc_period_ticks``) to that
    field's sources."""
    realize_mod = importlib.import_module("mrm.params.realize")    # (mrm.params.realize is also a function name)
    relations_mod = importlib.import_module("mrm.params.relations")
    tree = ast.parse((ROOT / "mrm" / "params" / "realize.py").read_text())
    mod_funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    group_of = {}
    for f in dataclasses.fields(realize_mod.PlantParams):
        t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
        if t.endswith("Params") or t == "FidelityFlags":
            group_of[t] = f.name
    param_src: dict = defaultdict(set)              # (function, parameter) -> sources from its call sites
    # cfg.<property> (Config.n_rings, n_lines): the paths its property body reads (mrm/params/config.py)
    cfg_props = {}
    ctree = ast.parse((ROOT / "mrm" / "params" / "config.py").read_text())
    for c in ctree.body:
        if isinstance(c, ast.ClassDef) and c.name == "Config":
            for f in c.body:
                if isinstance(f, ast.FunctionDef) and any(isinstance(d, ast.Name) and d.id == "property"
                                                          for d in f.decorator_list):
                    cfg_props[f.name] = {n.value for n in ast.walk(f) if isinstance(n, ast.Constant)
                                         and isinstance(n.value, str) and n.value in known}

    def collect(expr, line, scope, depth=0, noskip=False, seen=frozenset()):
        found = set()
        if expr is None or depth > 8:
            return found
        stack = [expr]
        while stack:
            n = stack.pop()
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                if n.value in known:
                    found.add(n.value)
                elif n.value in unc_names:
                    found.add("unc." + n.value)
            elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                if n.value.id == "relations" and isinstance(getattr(relations_mod, n.attr, None), str):
                    if getattr(relations_mod, n.attr) in known:
                        found.add(getattr(relations_mod, n.attr))
                    continue
                if n.value.id in _GROUP_VARS:
                    found.add(f"@{_GROUP_VARS[n.value.id]}.{n.attr}")
                    continue
                if n.value.id == "cfg" and n.attr in cfg_props:
                    found |= cfg_props[n.attr]
                    continue
            elif isinstance(n, ast.JoinedStr) and scope.subst:
                parts = []
                for v in n.values:
                    if isinstance(v, ast.Constant):
                        parts.append(str(v.value))
                    elif (isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name)
                          and v.value.id in scope.subst):
                        parts.append(scope.subst[v.value.id])
                    else:
                        parts = None
                        break
                if parts is not None and "".join(parts) in known:
                    found.add("".join(parts))
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                found |= resolve(n.id, line, scope, depth, noskip, seen)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                name = n.func.id
                if name in _DRAW_FUNCS:
                    for arg in n.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            found.add("draw:" + arg.value)
                    continue
                callee, owner = scope.func(name)
                if callee is None and name in mod_funcs:
                    callee, owner = mod_funcs[name], None
                if callee is not None and (name, line) not in seen and callee is not scope.fn:
                    sub, bind = {}, {}
                    for a_, arg in zip(callee.args.args, n.args):
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            sub[a_.arg] = arg.value
                        else:
                            bind[a_.arg] = (arg, scope, line)
                    for kw in n.keywords:
                        if kw.arg is not None:
                            bind[kw.arg] = (kw.value, scope, line)
                    inner = _Scope(callee, outer=owner, subst=sub, call_line=line, bind=bind)
                    for st in callee.body:
                        found |= collect(st, 10 ** 9, inner, depth + 1, False, seen | {(name, line)})
            stack.extend(ast.iter_child_nodes(n))
        return found

    def resolve(name, line, scope, depth, noskip, seen):
        if name in _SKIP_NAMES and not noskip:
            return set()
        if name in scope.params:
            out_ = set()
            if name in scope.subst and scope.subst[name] in known:
                out_.add(scope.subst[name])
            if name in scope.bind:
                arg, caller, ln = scope.bind[name]
                out_ |= collect(arg, ln, caller, depth + 1, False, seen)
            elif scope.outer is None and not scope.subst and not scope.bind:
                out_ |= param_src.get((scope.fn.name, name), set())
            return out_
        prev = [(ln, e) for ln, e in scope.assigns.get(name, ()) if ln < line]
        if prev:
            out_ = set()
            for ln, e in prev:                     # every earlier assignment (branches and later refinements)
                if (name, ln) not in seen:
                    out_ |= collect(e, ln, scope, depth + 1, False, seen | {(name, ln)})
            return out_
        if scope.outer is not None:
            return resolve(name, scope.call_line, scope.outer, depth + 1, noskip, seen)
        return set()

    out: dict = defaultdict(set)
    for _pass in range(2):                       # pass 1 fills param_src; pass 2 uses it
        out.clear()
        new_params: dict = defaultdict(set)
        for fn in mod_funcs.values():
            scope = _Scope(fn)
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                name = node.func.id
                if name in mod_funcs and name != fn.name:
                    for a_, arg in zip(mod_funcs[name].args.args, node.args):
                        if not isinstance(arg, ast.Constant):
                            new_params[(name, a_.arg)] |= collect(arg, node.lineno, scope)
                if name in group_of:
                    grp = group_of[name]
                    for kw in node.keywords:
                        if kw.arg is not None:
                            out[f"{grp}.{kw.arg}"] |= collect(kw.value, node.lineno, scope)
                        elif isinstance(kw.value, ast.Name):
                            for ln, e in scope.assigns.get(kw.value.id, ()):
                                if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "dict":
                                    for kw2 in e.keywords:
                                        out[f"{grp}.{kw2.arg}"] |= collect(kw2.value, ln, scope)
                            for key, e in scope.subs.get(kw.value.id, ()):
                                out[f"{grp}.{key}"] |= collect(e, node.lineno, scope)
                        elif (isinstance(kw.value, ast.Call) and isinstance(kw.value.func, ast.Name)
                              and kw.value.func.id in mod_funcs):
                            # **helper(...): the keys of the dicts it returns, each from the whole helper body
                            h = mod_funcs[kw.value.func.id]
                            body = collect(kw.value, node.lineno, scope)
                            for d in ast.walk(h):
                                if isinstance(d, ast.Dict):
                                    for k in d.keys:
                                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                            out[f"{grp}.{k.value}"] |= body
                if name == "PlantParams":
                    for kw in node.keywords:
                        if kw.arg in ("n_rings", "n_lines", "batch"):
                            out[kw.arg] |= collect(kw.value, node.lineno, scope, noskip=True)
        param_src = new_params
    # group references (@timing.adc_period_ticks) -> that field's sources, to a fixed point
    for _ in range(6):
        changed = False
        for key, ss_ in out.items():
            refs = {x for x in ss_ if x.startswith("@")}
            if not refs:
                continue
            add = set()
            for r in refs:
                add |= {x for x in out.get(r[1:], set()) if x != "@" + key}
            new = (ss_ - refs) | add
            if new != ss_:
                out[key] = new
                changed = True
        if not changed:
            break
    return {k: {x for x in v if not x.startswith("@")} for k, v in out.items()}


def plantparams_fields() -> list:
    """[(group, field, comment)] of every PlantParams group field (the comments of mrm/params/realize.py)."""
    realize_mod = importlib.import_module("mrm.params.realize")
    src = (ROOT / "mrm" / "params" / "realize.py").read_text().splitlines()
    tree = ast.parse("\n".join(src))
    comments = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for st in node.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    line = src[st.lineno - 1]
                    comments[(node.name, st.target.id)] = line.split("#", 1)[1].strip() if "#" in line else ""
    out = [("top", n, c) for n, c in (("batch", "number of worlds B"), ("n_rings", "number of rings N"),
                                      ("n_lines", "number of laser lines L"))]
    for f in dataclasses.fields(realize_mod.PlantParams):
        t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
        cls = getattr(realize_mod, t, None)
        if cls is None or not dataclasses.is_dataclass(cls) or not (t.endswith("Params") or t == "FidelityFlags"):
            continue
        for g in dataclasses.fields(cls):
            out.append((f.name, g.name, comments.get((t, g.name), "")))
    return out


# ------------------------------------------------------------------------------------------------------------ the model

def build(preset: str = "g26") -> dict:
    """The graph model {meta, layers, kinds, sections, groups, nodes, edges} of ``preset`` (deterministic)."""
    from mrm import params
    from mrm.core import channels
    from mrm.params import schema
    from mrm.params.config import _hypothesis_choice_map
    from mrm.params.realize import JACOBIAN_TARGETS
    from mrm.params import symbols
    from mrm.params.table import _unit_label
    from mrm.sim import metrics
    from mrm.sim import state_spec

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load(preset)
        der = params.derive(cfg)
    pp = der.params
    derivations = pp.derivations
    nodes: dict = {}
    groups: dict = {}
    edges: set = set()

    def group(gid, layer, title, desc=""):
        if gid not in groups:
            groups[gid] = {"id": gid, "layer": layer, "title": title, "desc": desc, "order": len(groups)}
        return gid

    def node(nid, layer, grp, **kw):
        n = {"id": nid, "layer": layer, "group": grp}
        n.update({k: v for k, v in kw.items() if v not in (None, "", (), [])})
        if n.get("symbol"):                       # plain text: the viewer's search and its fallback box label
            n["symbol_text"] = symbols.to_unicode(n["symbol"])
        nodes[nid] = n
        return n

    # -------------------------------------------------------------------------------------------------- layer 1: inputs
    def section(path):
        parts = path.split(".")
        return ".".join(parts[:2]) if parts[0] == "ring" and len(parts) > 2 else parts[0]

    draws = defaultdict(list)
    for name, e in sorted(cfg.uncertainty.items()):
        f = e.fields
        if e.kind == "hypothesis":
            paths = sorted({p for m in _hypothesis_choice_map(e).values() for p in m})
            for p in paths:
                draws[p].append(f"uncertainty {name} (hypothesis: {', '.join(sorted(_hypothesis_choice_map(e)))})")
        elif e.target:
            if e.kind == "interval":
                d = f"[{f.get('low')}, {f.get('high')}] {f.get('dist', 'uniform')} {f.get('mode', 'absolute')}"
            else:
                d = f"{f.get('dist', 'normal')} mean {f.get('mean')} sigma {f.get('sigma')}"
            draws[e.target].append(f"uncertainty {name} ({e.kind} {d})")
    for p, e in sorted(cfg.variation.items()):
        f = e.fields
        sig = ", ".join(f"{k}={f[k]}" for k in ("die_sigma", "ring_sigma", "line_sigma", "rel_sigma") if k in f)
        draws[p].append(f"variation per device ({f.get('level', '')}, {f.get('dist', 'normal')} {sig})")
    if cfg.geometry:
        dims = [d for d in ("width", "thickness", "slab", "gap_through", "gap_drop") if d in cfg.geometry]
        for out_name, (target, how, _sgn) in sorted(JACOBIAN_TARGETS.items()):
            if target in cfg.leaves or target in derivations:
                draws[target].append(f"variation of the geometry ({', '.join(dims)}) through the Jacobian "
                                     f"{out_name} ({how})")

    leaf_ids = set()
    for p, lf in sorted(cfg.leaves.items()):
        if lf.is_free or p in derivations:
            continue
        sec = section(p)
        g = group("in:" + sec, 1, sec)
        node("cfg." + p, 1, g, label=p.split(".", 2)[-1] if p.startswith("ring.") else p.split(".", 1)[-1] or p,
             path=p, type="leaf", symbol=symbols.symbol(p), value=_value_text(lf.value), unit=_unit_label(p),
             provenance=lf.provenance,
             generation=lf.generation, source=lf.source, note=lf.note,
             meaning=(schema.SCHEMA[p].doc if p in schema.SCHEMA else ""), draws=draws.get(p))
        leaf_ids.add(p)
    for p, v in sorted(cfg.settings.items()):
        sec = section(p)
        g = group("in:" + sec, 1, sec)
        node("cfg." + p, 1, g, label=p.split(".", 2)[-1] if p.startswith("ring.") else p.split(".", 1)[-1] or p,
             path=p, type="setting", symbol=symbols.symbol(p), value=_value_text(v),
             meaning=(schema.SCHEMA[p].doc if p in schema.SCHEMA else ""),
             draws=draws.get(p))
    # uncertainty entries that change no leaf (world-level scales and switches)
    g_unc = None
    for name, e in sorted(cfg.uncertainty.items()):
        targets = e.target or sorted({p for m in _hypothesis_choice_map(e).values() for p in m})
        if targets:
            continue
        g_unc = group("in:uncertainty", 1, "uncertainty (no target leaf)",
                      "Uncertainty entries drawn once per world that change no preset leaf: thermal scales, the "
                      "latency scale, scenario hypotheses.")
        f = e.fields
        detail = (f"[{f.get('low')}, {f.get('high')}]" if e.kind == "interval"
                  else ", ".join(sorted(_hypothesis_choice_map(e))) if e.kind == "hypothesis" else "")
        node("unc." + name, 1, g_unc, label=name, type="uncertainty", value=f"{e.kind} {detail}".strip(),
             provenance=f.get("provenance", ""), source=f.get("source", ""), note=f.get("note", ""),
             meaning="Uncertainty draw (one per world) without a target leaf.",
             draws=[f"uncertainty {name} ({e.kind} {detail})"])

    # --------------------------------------------------------------------------------------- layer 2: derived constants
    def cfg_ref(token):
        """A derivation input or a realize source as a node id (None when the model has no node for it)."""
        if token.startswith("uncertainty: latency"):
            return "unc.timing.latency_scale" if "unc.timing.latency_scale" in nodes else None
        if token.startswith("unc."):
            return token if token in nodes else None
        return "cfg." + token if ("cfg." + token) in nodes or token in derivations else None

    for p, d in sorted(derivations.items()):
        g = group("der:" + d.relation, 2, d.relation)
        v = pp.values.get(p)
        unit = _unit_label(p) if schema.lookup(p) is not None else "SI"
        disp = v
        if v is not None and schema.lookup(p) is not None:
            try:
                disp = params.from_si(p, np.asarray(v, float))
            except Exception:  # noqa: BLE001 - a non-numeric value keeps its SI form
                disp = v
        node("cfg." + p, 2, g, label=p.split(".", 1)[-1], path=p, type="derived", relation=d.relation,
             symbol=symbols.symbol(p), formula=d.formula, formula_tex=d.formula_tex, eq=d.eq,
             value=_value_text(np.asarray(disp)) if disp is not None else "", unit=unit,
             provenance=params.effective_provenance(p, cfg, derivations), note=d.note,
             meaning=(schema.SCHEMA[p].doc if p in schema.SCHEMA else ""), draws=draws.get(p),
             inputs_text=list(d.inputs))
    for p, d in sorted(derivations.items()):
        for i in d.inputs:
            r = cfg_ref(i)
            if r is not None:
                edges.add((r, "cfg." + p, "dep"))

    # --------------------------------------------------------------------------------- layer 3: PlantParams and KParams
    known = set(cfg.leaves) | set(cfg.settings) | set(derivations) | set(schema.SCHEMA)
    srcs = realize_sources(known, set(cfg.uncertainty))
    tin = pp.thermal_inputs
    if tin is not None:
        for k in sorted(tin.interim):
            for cand in (f"ring.thermal.{k}", f"ring.thermal.{k[:-2]}_um" if k.endswith("_m") else None):
                if cand and cand in known:
                    srcs["thermal.system"].add(cand)
        for k in sorted(tin.scales):
            if f"thermal.{k}" in cfg.uncertainty:
                srcs["thermal.system"].add(f"unc.thermal.{k}")
    for grp, fld, comment in plantparams_fields():
        pid = f"pp.{fld}" if grp == "top" else f"pp.{grp}.{fld}"
        key = fld if grp == "top" else f"{grp}.{fld}"
        g = group("pp:" + grp, 3, f"PlantParams.{grp}" if grp != "top" else "PlantParams (sizes)",
                  "realize() output: SI arrays (B, N) / (B, 1) / (B, L) per world.")
        obj = pp if grp == "top" else getattr(getattr(pp, grp), fld, None) if grp != "top" else None
        if grp == "top":
            obj = getattr(pp, fld)
        sources = sorted(srcs.get(key, ()))
        rnd = [f"per-device random draw '{s[5:]}' of realize (seed 'device')" for s in sources
               if s.startswith("draw:")]
        sources = [s for s in sources if not s.startswith("draw:")]
        present = [s for s in sources if cfg_ref(s)]
        absent = [s for s in sources if not cfg_ref(s)]
        node(pid, 3, g, label=fld, type="plantparams", symbol=symbols.PLANTPARAMS_SYMBOLS.get(key, ""), meaning=comment,
             unit=_unit_from_name(fld),
             value=_value_text(obj) if not dataclasses.is_dataclass(obj) else f"<{type(obj).__name__}>",
             sources=present, absent_sources=absent, draws=rnd)
        for s in present:
            edges.add((cfg_ref(s), pid, "dep"))
    for gp in state_spec.KPARAM_GROUPS:
        g = group("P:" + gp.key, 3, f"KParams: {gp.title}", "Kernel parameters built by Plant._build (frozen "
                  "for a run; mrm/sim/state_spec.py KPARAM_GROUPS).")
        for prm in gp.params:
            pp_src = [s for s in prm.src if s.startswith("pp.") and s in nodes]
            cfg_src = [s for s in prm.src if s.startswith("cfg.") and s in nodes]
            unc_src = [s for s in prm.src if s.startswith("unc.") and s in nodes]
            other = [s for s in prm.src if s not in pp_src + cfg_src + unc_src]
            node("P." + prm.name, 3, g, label=prm.name, type="kparam", symbol=prm.symbol, meaning=prm.meaning,
                 unit=prm.unit,
                 shape=prm.shape, sources=pp_src + cfg_src + unc_src, other_sources=other, note=prm.note)
            for s in pp_src + cfg_src + unc_src:
                edges.add((s, "P." + prm.name, "dep"))

    # ------------------------------------------------------------------------------------------- layer 4 (and 5): state
    sec_title = {s.key: s.title for s in state_spec.SECTIONS}
    sec_what = {s.key: s.what for s in state_spec.SECTIONS}
    optics_comments = _constant_comments(ROOT / "mrm" / "plant" / "optics.py")
    deps_of: dict = {}

    def state_node(nid, storage, entry_kind, meaning, unit, shape, by, deps, note="", init="", symbol=""):
        sec, fns = _split_updated_by(by)
        layer = 5 if entry_kind == "truth" else 4
        if layer == 5:
            g = group("out:truth_rows", 5, "Truth and metric rows of the kernel",
                      "Kernel values that only Truth, the metrics and the records read (never a controller).")
        else:
            g = group("st:" + sec, 4, sec_title.get(sec, sec), sec_what.get(sec, ""))
        node(nid, layer, g, label=nid, type="state", storage=storage, kind=entry_kind, symbol=symbol, meaning=meaning,
             unit=unit, shape=shape, section=sec, updated_by=fns, note=note, init=init)
        deps_of[nid] = deps

    for b in state_spec.BLOCKS:
        for i, f in enumerate(b.fields):
            state_node(f"{b.key}.{f.name}", f"S.{b.key}[:, {b.const(f)}] (row {i})", f.kind, f.meaning, f.unit,
                       b.shape, f.updated_by, f.depends_on, f.note, f.init, f.symbol)
    for r in state_spec.SH_ROWS:
        cname = r.const.split(".", 1)[1]
        meaning = r.meaning or optics_comments.get(cname, "")
        state_node("selfheat." + r.name, f"S.selfheat_float row {r.const}", r.kind, meaning, r.unit, r.shape,
                   r.updated_by, r.depends_on, r.note, symbol=r.symbol)
    containers = {"selfheat_float", "truth"} | {b.key for b in state_spec.BLOCKS}
    for a in state_spec.KSTATE:
        if a.name in containers:
            continue
        state_node("S." + a.name, f"S.{a.name}", a.kind, a.meaning, a.unit, a.shape, a.updated_by, a.depends_on,
                   a.note, a.init, a.symbol)

    # ------------------------------------------------------------------------------------------------- layer 5: outputs
    g_obs = group("out:obs", 5, "Observations (controller sees)", "Sensor channels of mrm.core.channels packed "
                  "into obs_i64 at each controller edge (io.pack_obs_row).")
    g_ctrl = group("out:ctrl", 5, "Controller", "The controller reads every observation channel and writes every "
                   "command channel.")
    g_act = group("out:act", 5, "Commands (controller writes)", "Actuator channels of mrm.core.channels: the "
                  "controller writes them, the plant latches them after the controller latency.")
    node("ctrl.controller", 5, g_ctrl, label="controller", type="controller",
         meaning="The controller under test (compiled step_fn or Python step). It sees only the observations.")
    for row in channels.V1_CHANNEL_TABLE:
        name = row["name"]
        if row["kind"] == "sensor":
            node("obs." + name, 5, g_obs, label=name, type="channel", symbol=state_spec.OBS_SYMBOLS.get(name, ""),
                 meaning=_plain_channel_doc(row.get("doc", "")),
                 unit=row.get("unit", ""), shape="(" + ", ".join(map(str, row["shape"])) + ")")
            deps_of["obs." + name] = state_spec.OBS_SOURCES.get(name, "").split()
            edges.add(("obs." + name, "ctrl.controller", "obs"))
        else:
            node("act." + name, 5, g_act, label=name, type="channel", symbol=state_spec.ACT_SYMBOLS.get(name, ""),
                 meaning=_plain_channel_doc(row.get("doc", "")),
                 unit=row.get("unit", ""), shape="(" + ", ".join(map(str, row["shape"])) + ")",
                 reaches=state_spec.ACT_TARGETS.get(name, "").split())
            edges.add(("ctrl.controller", "act." + name, "cmd"))
    truth_comments = _comments(ROOT / "mrm" / "sim" / "metrics.py", r'^\s*"(\w+)",\s*#\s*\d+\s+(.*)$')
    g_tr = group("out:truth", 5, "Truth (never seen by a controller)", "Per-ring Truth fields of every step "
                 "(mrm.sim.metrics.TRUTH_FIELDS, S.truth).")
    for name in metrics.TRUTH_FIELDS:
        node("truth." + name, 5, g_tr, label=name, type="truth", symbol=state_spec.TRUTH_SYMBOLS.get(name, ""),
             meaning=truth_comments.get(name, ""),
             unit=_unit_from_name(name), storage=f"S.truth[:, {metrics.TR[name]}]", updated_by=["kernel._truth_a"])
        deps_of["truth." + name] = state_spec.TRUTH_SOURCES.get(name, "").split()
    slot_comments = _comments(ROOT / "mrm" / "sim" / "metrics.py", r"^(A[FI]_[A-Z0-9_]+) = [^#]*#\s*(.*)$")
    g_m = group("out:metrics", 5, "Summary metrics", "Per-ring results of mrm.sim.metrics.summarize from the "
                "streaming accumulators.")
    for key, used, slots in _summary_metrics():
        meaning = "Summary metric (mrm.sim.metrics.summarize)."
        if slots:
            meaning += " Reads " + "; ".join(f"{s} ({slot_comments[s]})" if s in slot_comments else s
                                             for s in slots) + "."
        node("metric." + key, 5, g_m, label=key, type="metric", symbol=state_spec.METRIC_SYMBOLS.get(key, ""),
             meaning=meaning, unit=_unit_from_name(key))
        deps_of["metric." + key] = used

    # ------------------------------------------------------------------------------- dependency edges of layers 4 and 5
    for nid, deps in deps_of.items():
        through = any(f.startswith("controller.") for f in nodes[nid].get("updated_by", []))
        for d in deps:
            if d in nodes:
                # the observation vectors feed a value the controller writes: that edge passes through the controller
                edges.add((d, nid, "cmd" if through and d in ("S.obs", "S.fobs") else "dep"))
    for name in state_spec.ACT_TARGETS:
        if "act." + name in nodes and "S.act" in nodes:
            edges.add(("act." + name, "S.act", "cmd"))
    edges = {e for e in edges if not (e[1] == "S.act" and e[0].startswith("act.") and e[2] == "dep")}

    # ----------------------------------------------------------------------------------- counts and the uncertain notes
    lay_of = {n["id"]: n["layer"] for n in nodes.values()}
    counts = {}
    for L, _, _ in LAYERS:
        ne = sum(1 for e in edges if lay_of[e[1]] == L)
        counts[str(L)] = {"nodes": sum(1 for n in nodes.values() if n["layer"] == L),
                          "groups": sum(1 for g in groups.values() if g["layer"] == L), "edges_into": ne}
    uncertain = [{"id": n["id"], "note": n["note"]} for n in nodes.values() if "uncertain:" in n.get("note", "")]
    order = {gid: i for i, gid in enumerate(sorted(groups, key=lambda x: (groups[x]["layer"],
                                                                          _group_rank(x, state_spec),
                                                                          groups[x]["order"])))}
    for gid, g in groups.items():
        g["order"] = order[gid]
        g["count"] = sum(1 for n in nodes.values() if n["group"] == gid)
    return {
        "meta": {"preset": preset, "config_hash": cfg.hash()[:16], "generator": "tools/gen_variable_map.py",
                 "registry": "mrm/sim/state_spec.py", "notation": "mrm/params/symbols.py", "regenerate": REGEN,
                 "counts": counts, "symbols": sum(1 for n in nodes.values() if n.get("symbol")),
                 "uncertain": uncertain},
        "layers": [{"id": L, "title": t, "desc": d} for L, t, d in LAYERS],
        "kinds": dict(state_spec.KINDS),
        "sections": [{"key": s.key, "title": s.title, "what": s.what} for s in state_spec.SECTIONS],
        "groups": sorted(groups.values(), key=lambda g: g["order"]),
        "nodes": sorted(nodes.values(), key=lambda n: (n["layer"], order[n["group"]], n["id"])),
        "edges": sorted([list(e) for e in edges]),
    }


def _group_rank(gid: str, state_spec) -> int:
    """Order of the groups inside a layer: the event order for the state sections, else 0 (then creation order)."""
    if gid.startswith("st:"):
        k = gid[3:]
        return state_spec.SECTION_KEYS.index(k) if k in state_spec.SECTION_KEYS else 99
    rank = {"pp:top": 0, "out:obs": 0, "out:ctrl": 1, "out:act": 2, "out:truth_rows": 3, "out:truth": 4,
            "out:metrics": 5, "in:uncertainty": 98}
    if gid.startswith("P:"):
        return 50
    return rank.get(gid, 10)


def _summary_metrics() -> list:
    """[(key, [state ids], [AF_/AI_ slots])] of every out["key"] = ... of mrm.sim.metrics.summarize (ast)."""
    tree = ast.parse((ROOT / "mrm" / "sim" / "metrics.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "summarize")
    local = {}
    res = []
    for st in fn.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1:
            t = st.targets[0]
            if isinstance(t, ast.Name):
                local[t.id] = st.value
            elif isinstance(t, ast.Tuple):
                for el in t.elts:
                    if isinstance(el, ast.Name):
                        local[el.id] = st.value
            elif (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "out"
                  and isinstance(t.slice, ast.Constant) and t.slice.value not in ("world", "settings")):
                names, slots = set(), set()

                def walk(e, depth=0):
                    for n in ast.walk(e):
                        if isinstance(n, ast.Name):
                            if re.match(r"^A[FI]_", n.id):
                                slots.add(n.id)
                            names.add(n.id)
                            if n.id in local and depth < 3 and n.id not in ("af", "ai", "dt", "spec", "state"):
                                walk(local[n.id], depth + 1)
                        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "state":
                            names.add("state." + n.attr)
                walk(st.value)
                used = []
                for nm, sid in (("af", "S.m_af"), ("ai", "S.m_ai"), ("state.runhist", "S.m_runhist"),
                                ("state.hist", "S.m_hist"), ("state.ref", "S.m_ref"), ("t0", "S.m_ai"),
                                ("ticks", "S.m_ai"), ("tk", "S.m_ai")):
                    if nm in names and sid not in used:
                        used.append(sid)
                res.append((t.slice.value, used, sorted(slots)))
    return res


# ------------------------------------------------------------------------------------------------------------ renderers

def render_json(model: dict) -> str:
    """One object per line in the long lists, so a change shows as a short diff."""
    def block(name, items):
        return f' "{name}": [\n' + ",\n".join("  " + json.dumps(x, ensure_ascii=False, sort_keys=True)
                                               for x in items) + "\n ]"
    head = {k: model[k] for k in ("meta", "layers", "kinds", "sections")}
    parts = [f' "{k}": ' + json.dumps(v, ensure_ascii=False, sort_keys=True) for k, v in head.items()]
    parts += [block("groups", model["groups"]), block("nodes", model["nodes"]), block("edges", model["edges"])]
    return "{\n" + ",\n".join(parts) + "\n}\n"


class SymbolRenderError(RuntimeError):
    """The symbols cannot be typeset to SVG (no node, no tools/katex/node_modules, or a MathJax error)."""


def mathjax_version() -> str:
    """The mathjax-full version that tools/katex/package.json pins (the one symbols.json must be made with)."""
    return json.loads((KATEX_DIR / "package.json").read_text())["dependencies"]["mathjax-full"]


def symbol_texts(model: dict) -> list:
    """Every distinct symbol of the map (sorted): the strings that symbols.json has an SVG for."""
    return sorted({n["symbol"] for n in model["nodes"] if n.get("symbol")})


def render_symbol_svgs(model: dict) -> list:
    """[{tex, svg, w, h}] of every distinct symbol, typeset by MathJax in node (tools/katex/tex2svg.js). Raises
    SymbolRenderError when node or the npm packages are missing or a symbol does not typeset."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None or not (KATEX_DIR / "node_modules" / "mathjax-full").is_dir():
        raise SymbolRenderError("cannot typeset the box symbols: node or tools/katex/node_modules/mathjax-full is "
                                f"missing (install node, then run: {NPM_CI})")
    texts = symbol_texts(model)
    r = subprocess.run([node, str(TEX2SVG)], input=json.dumps(texts), capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise SymbolRenderError(f"tools/katex/tex2svg.js failed: {r.stderr.strip()[-2000:]}")
    res = json.loads(r.stdout)
    if res["errors"]:
        raise SymbolRenderError(f"MathJax cannot typeset (fix mrm/params/symbols.py or state_spec): {res['errors']}")
    if res["version"] != mathjax_version():
        raise SymbolRenderError(f"tools/katex/node_modules has mathjax-full {res['version']}, package.json pins "
                                f"{mathjax_version()}: run {NPM_CI}")
    items = sorted(res["items"], key=lambda x: x["tex"])
    if [x["tex"] for x in items] != texts:
        raise SymbolRenderError("tools/katex/tex2svg.js did not return one SVG per symbol")
    return items


_GLYPH_RX = re.compile(r'<path d="([^"]*)"></path>')


def render_symbols_json(items: list) -> str:
    """docs/variables/symbols.json: the glyph outlines once ("glyphs", one per line) and one symbol per line (tex,
    svg, w and h in em). In a symbol's svg, ``<path data-g="N"></path>`` stands for the path of glyph N (the viewer
    puts ``d="..."`` back); most glyphs appear in many symbols, so the file is about four times smaller."""
    glyphs = sorted({d for x in items for d in _GLYPH_RX.findall(x["svg"])})
    gid = {d: k for k, d in enumerate(glyphs)}
    syms = [dict(x, svg=_GLYPH_RX.sub(lambda m: f'<path data-g="{gid[m.group(1)]}"></path>', x["svg"]))
            for x in items]
    meta = {"generator": "tools/gen_variable_map.py", "renderer": "mathjax-full", "version": mathjax_version(),
            "script": "tools/katex/tex2svg.js", "regenerate": REGEN, "count": len(items),
            "about": "SVG of every distinct box symbol of data.json, drawn in currentColor; w and h in em; "
                     '<path data-g="N"></path> is the path of glyphs[N]'}
    return ("{\n \"meta\": " + json.dumps(meta, ensure_ascii=False, sort_keys=True)
            + ",\n \"glyphs\": [\n" + ",\n".join("  " + json.dumps(d) for d in glyphs) + "\n ]"
            + ",\n \"symbols\": [\n" + ",\n".join("  " + json.dumps(x, ensure_ascii=False, sort_keys=True)
                                                 for x in syms) + "\n ]\n}\n")


def render_markdown(model: dict) -> str:
    from mrm.sim import state_spec
    nodes = {n["id"]: n for n in model["nodes"]}
    into = defaultdict(list)
    outof = defaultdict(list)
    for a, b, _t in model["edges"]:
        into[b].append(a)
        outof[a].append(b)
    m = model["meta"]
    out = ["# Variables: every value of the model, from the constants to the per-tick state", "",
           f"Generated by `{m['regenerate']}` from the registry `{m['registry']}`, the `{m['preset']}` preset, the "
           "derivations of mrm.params.realize, mrm.core.channels and mrm.sim.metrics. Do not edit by hand: "
           "tests/verification/test_variable_map.py fails when this file differs from a fresh generation. "
           f"Config hash `{m['config_hash']}`.", "",
           "The same graph, drawn top to bottom with search and a detail panel: docs/variables/index.html (open it "
           "through a local web server, see docs/variables/README.md).", "",
           "Five layers, top to bottom. An arrow goes from a value to the values computed from it.", "",
           f"{m['symbols']} values have a math symbol (column *symbol*; the notation of `{m['notation']}`, which "
           "follows the textbook's chapters 0-5). A value without a symbol (a code, flag, counter, key or table) is "
           "named only. *eq.* is the number of the textbook equation that a formula is or follows from.", ""]
    out += ["| layer | name | groups | nodes | edges into the layer |", "|---|---|---|---|---|"]
    for L in model["layers"]:
        c = m["counts"][str(L["id"])]
        out.append(f"| {L['id']} | {L['title']} | {c['groups']} | {c['nodes']} | {c['edges_into']} |")
    out += ["", "Kinds of runtime values:", ""] + [f"- `{k}`: {v}." for k, v in model["kinds"].items()]

    # layer 1
    out += ["", "## 1. Inputs", "", "Every leaf and setting of the preset, with its value, unit, provenance and "
            "source, is in docs/PARAMETERS.md (`mrm params table g26`). Counts per section:", "",
            "| section | leaves | settings |", "|---|---|---|"]
    cnt = defaultdict(lambda: [0, 0])
    for n in model["nodes"]:
        if n["layer"] == 1 and n.get("type") in ("leaf", "setting"):
            cnt[n["group"][3:]][0 if n["type"] == "leaf" else 1] += 1
    for sec, (a, b) in sorted(cnt.items()):
        out.append(f"| `{sec}` | {a} | {b} |")
    out += ["", "Inputs that an uncertainty or variation draw changes (one draw per world or per device):", "",
            "| input | draws |", "|---|---|"]
    for n in model["nodes"]:
        if n["layer"] in (1, 2) and n.get("draws") and n.get("type") != "uncertainty":
            out.append(f"| `{n['path']}` | {_md_cell('; '.join(n['draws']))} |")
    out += ["", "Symbols of the inputs:", "", "| input | symbol | meaning | value | unit |", "|---|---|---|---|---|"]
    for n in model["nodes"]:
        if n["layer"] == 1 and n.get("symbol"):
            out.append(f"| `{n['path']}` | {_md_math(n['symbol'])} | {_md_cell(n.get('meaning', ''))} | "
                       f"{_md_cell(n.get('value', ''))} | {n.get('unit', '')} |")
    out += ["", "Uncertainty entries that change no leaf:", "", "| entry | draw | feeds |", "|---|---|---|"]
    for n in model["nodes"]:
        if n.get("type") == "uncertainty":
            out.append(f"| `{n['label']}` | {_md_cell(n.get('value', ''))} | "
                       f"{', '.join('`' + x + '`' for x in sorted(outof[n['id']])) or 'no engine value (kept in PlantParams.world)'} |")

    # layer 2
    out += ["", "## 2. Derived constants", "",
            "| quantity | symbol | relation | formula | eq. | formula (plain text) | inputs | value | unit |",
            "|---|---|---|---|---|---|---|---|---|"]
    for n in model["nodes"]:
        if n["layer"] == 2:
            ins = ", ".join(f"`{i}`" for i in n.get("inputs_text", []))
            out.append(f"| `{n['path']}` | {_md_math(n.get('symbol'))} | {n['relation']} | "
                       f"{_md_math(n.get('formula_tex'))} | {n.get('eq', '')} | {_md_cell(n.get('formula', ''))} | "
                       f"{ins} | {_md_cell(n.get('value', ''))} | {n.get('unit', '')} |")

    # layer 3
    out += ["", "## 3. Frozen per-run parameters", "", "### PlantParams (mrm.params.realize)", "",
            "Config sources are read from the code that builds each group in mrm/params/realize.py.", "",
            "| field | symbol | meaning | config sources |", "|---|---|---|---|"]
    for n in model["nodes"]:
        if n.get("type") == "plantparams":
            s = ", ".join(f"`{x[4:] if x.startswith('cfg.') else x}`" for x in n.get("sources", []))
            if n.get("absent_sources"):
                s += (", " if s else "") + "absent in the preset: " + ", ".join(f"`{x}`" for x in
                                                                                 n["absent_sources"])
            out.append(f"| `{n['id'][3:]}` | {_md_math(n.get('symbol'))} | {_md_cell(n.get('meaning', ''))} | {s} |")
    out += ["", "### KParams (Plant._build; registry: state_spec.KPARAM_GROUPS)", ""]
    for gp in state_spec.KPARAM_GROUPS:
        out += [f"#### {gp.title}", "", "| field | symbol | meaning | unit | shape | from |",
                "|---|---|---|---|---|---|"]
        for prm in gp.params:
            n = nodes["P." + prm.name]
            fr = ", ".join(f"`{x}`" for x in n.get("sources", []) + n.get("other_sources", []))
            note = f" {prm.note}" if prm.note else ""
            out.append(f"| `{prm.name}` | {_md_math(prm.symbol)} | {_md_cell(prm.meaning + note)} | {prm.unit} | "
                       f"{prm.shape} | {fr} |")
        out.append("")

    # layer 4
    out += ["## 4. Per-tick state", "", "One plant step [k, k + h) runs these parts in this order "
            "(kernel._world_loop). Each table lists the values the part writes.", "",
            "| # | key | part | what it does |", "|---|---|---|---|"]
    for i, s in enumerate(model["sections"]):
        out.append(f"| {i} | `{s['key']}` | {s['title']} | {_md_cell(s['what'])} |")
    by_sec = defaultdict(list)
    for n in model["nodes"]:
        if n.get("type") == "state":
            by_sec[n["section"]].append(n)
    for s in model["sections"]:
        rows = by_sec.get(s["key"])
        if not rows:
            continue
        out += ["", f"### {s['title']}", "",
                "| value | symbol | kind | meaning | unit | shape | updated by | depends on |",
                "|---|---|---|---|---|---|---|---|"]
        for n in rows:
            dep = " ".join(f"`{d}`" for d in sorted(into[n["id"]]))
            note = f" {n['note']}" if n.get("note") else ""
            init = f" Initial value: {n['init']}." if n.get("init") else ""
            out.append(f"| `{n['id']}` | {_md_math(n.get('symbol'))} | {n.get('kind', '')} | "
                       f"{_md_cell(n.get('meaning', '') + note + init)} | "
                       f"{n.get('unit', '')} | {n.get('shape', '')} | {_md_cell(', '.join(n.get('updated_by', [])))} | "
                       f"{dep} |")

    # layer 5
    out += ["", "## 5. Outputs", "", "### Observations (what the controller sees)", "",
            "| channel | symbol | meaning | packed from |", "|---|---|---|---|"]
    for n in model["nodes"]:
        if n["id"].startswith("obs."):
            out.append(f"| `{n['label']}` | {_md_math(n.get('symbol'))} | {_md_cell(n.get('meaning', ''))} | "
                       f"{' '.join(f'`{d}`' for d in sorted(into[n['id']]))} |")
    out += ["", "### Commands (what the controller writes)", "", "| channel | symbol | meaning | reaches |",
            "|---|---|---|---|"]
    for n in model["nodes"]:
        if n["id"].startswith("act."):
            out.append(f"| `{n['label']}` | {_md_math(n.get('symbol'))} | {_md_cell(n.get('meaning', ''))} | "
                       f"{' '.join(f'`{d}`' for d in n.get('reaches', []))} |")
    out += ["", "### Truth (never seen by a controller)", "", "| field | symbol | meaning | unit | computed from |",
            "|---|---|---|---|---|"]
    for n in model["nodes"]:
        if n["id"].startswith("truth."):
            out.append(f"| `{n['label']}` | {_md_math(n.get('symbol'))} | {_md_cell(n.get('meaning', ''))} | "
                       f"{n.get('unit', '')} | {' '.join(f'`{d}`' for d in sorted(into[n['id']]))} |")
    out += ["", "### Summary metrics", "", "| metric | symbol | reads |", "|---|---|---|"]
    for n in model["nodes"]:
        if n["id"].startswith("metric."):
            out.append(f"| `{n['label']}` | {_md_math(n.get('symbol'))} | {_md_cell(n.get('meaning', ''))} |")
    out += ["", "## Dependencies that are not certain", "",
            "Registry notes that start with `uncertain:` (mrm/sim/state_spec.py):", ""]
    out += [f"- `{u['id']}`: {_md_cell(u['note'])}" for u in m["uncertain"]] or ["- none"]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 when a committed file differs from a fresh build")
    ap.add_argument("--preset", default="g26")
    a = ap.parse_args(argv)
    model = build(a.preset)
    try:
        svgs = render_symbol_svgs(model)
    except SymbolRenderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    outs = {MD_PATH: render_markdown(model), JSON_PATH: render_json(model), SYM_PATH: render_symbols_json(svgs)}
    if a.check:
        stale = [str(p.relative_to(ROOT)) for p, text in outs.items() if not p.exists() or p.read_text() != text]
        if stale:
            print(f"stale: {', '.join(stale)}; run: {REGEN}", file=sys.stderr)
            return 1
        print("docs/VARIABLES.md, docs/variables/data.json and docs/variables/symbols.json are current")
        return 0
    for p, text in outs.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"wrote {p.relative_to(ROOT)} ({len(text) // 1024} KiB)")
    c = model["meta"]["counts"]
    print("layers: " + "; ".join(f"{k}: {v['nodes']} nodes, {v['edges_into']} edges in" for k, v in c.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
