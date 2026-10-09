"""Human views of a config: ``mrm params show | table | explain | diff | validate`` (07 P1-14, P2-12).

- :func:`show` prints the resolved tree with each leaf's value, unit and provenance.
- :func:`table` renders the Markdown of ``docs/PARAMETERS.md`` (leaves, derived quantities with the weakest input
  provenance, diagnostics, uncertainty entries, variation entries, textbook disagreements). It is deterministic.
- :func:`explain` prints the provenance chain of a leaf, a derived quantity or a diagnostic (07 P1-14: derived
  values inherit the weakest input provenance).
- :func:`diff` compares two configs leaf by leaf.
- :func:`validate` loads, derives and realizes a small sampled batch, and returns the warnings.

:func:`register_cli` adds the ``params`` command group to the registry of :mod:`mrm.cli`.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from mrm.params import schema as sch
from mrm.params.config import Config, ConfigError, load
from mrm.params.derive import DIAGNOSTIC_UNITS, Derived, derive, effective_provenance
from mrm.params.realize import realize
from mrm.params.units import from_si, parse_key

__all__ = ["show", "table", "explain", "diff", "validate", "register_cli", "format_value", "keys_markdown"]


def _unit_label(path: str) -> str:
    try:
        _, spec = parse_key(path)
    except ValueError:
        return ""
    return spec.expr or ""


def _fmt_num(x) -> str:
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if x == 0 or (1e-3 <= abs(x) < 1e5):
            return f"{x:.6g}"
        return f"{x:.4g}"
    return str(x)


def format_value(v: Any, max_items: int = 8) -> str:
    """Compact text of a value (scalars, lists, arrays, mappings)."""
    if v is None:
        return "null"
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        flat = np.asarray(v, dtype=object).ravel().tolist() if v and not isinstance(v[0], dict) else v
        if flat and all(isinstance(x, (int, float, np.integer, np.floating)) for x in flat):
            arr = np.asarray(flat, float)
            if arr.size > 1 and np.allclose(arr, arr[0], rtol=0, atol=0):
                return f"{_fmt_num(flat[0])} (x{arr.size})"
            items = [_fmt_num(x) for x in flat[:max_items]]
            return "[" + ", ".join(items) + (", ..." if len(flat) > max_items else "") + "]"
        return f"<{len(v)} items>"
    if isinstance(v, dict):
        return "{" + ", ".join(list(map(str, v))[:max_items]) + (", ..." if len(v) > max_items else "") + "}"
    return _fmt_num(v)


def _md(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ") if s is not None else ""


# ------------------------------------------------------------------------------------------------ show

def show(cfg: Config) -> str:
    """Resolved tree: ``path = value unit  [provenance, generation, source] (alt: name)``."""
    lines = [f"# {cfg.name} ({cfg.kind}); files: {', '.join(cfg.files)}", f"# config hash {cfg.hash()[:12]}"]
    paths = sorted(set(cfg.leaves) | set(cfg.settings) | set(cfg.open_values))
    for p in paths:
        if p in cfg.leaves:
            lf = cfg.leaves[p]
            if lf.is_free:
                lines.append(f"{p} = <derived>")
                continue
            tag = ", ".join(str(x) for x in (lf.provenance, lf.generation, lf.source) if x)
            alt = f" (alt: {lf.selected_alt})" if lf.selected_alt else ""
            unit = _unit_label(p)
            lines.append(f"{p} = {format_value(lf.value)}{' ' + unit if unit else ''}  [{tag}]{alt}")
        elif p in cfg.settings:
            lines.append(f"{p} = {format_value(cfg.settings[p])}")
        else:
            lines.append(f"{p} = {format_value(cfg.open_values[p])}  (open)")
    for n, e in sorted(cfg.uncertainty.items()):
        pin = f" pinned={e.pinned}" if e.pinned is not None else ""
        lines.append(f"uncertainty.{n}: {e.kind} target={e.target}{pin}")
    for p, e in sorted(cfg.variation.items()):
        lines.append(f"variation.{p}: " + ", ".join(f"{k}={v}" for k, v in e.fields.items()
                                                       if k not in ("provenance", "generation", "source", "note")))
    if cfg.geometry:
        lines.append("variation.geometry: " + ", ".join(k for k in cfg.geometry if k in
                                                        ("width", "thickness", "slab", "gap_through", "gap_drop")))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ table

def _derived_rows(cfg: Config, d: Derived) -> list[tuple]:
    rows = []
    vals = d.params.values
    for path, der in sorted(d.params.derivations.items()):
        v = vals.get(path)
        if v is None or path.startswith("derived.timing"):
            continue
        key = sch.lookup(path)
        if key is not None:
            disp = from_si(path, np.asarray(v, float))
            unit = _unit_label(path)
        else:
            disp, unit = np.asarray(v), "SI"
        rows.append((path, format_value(np.asarray(disp).ravel()), unit,
                     effective_provenance(path, cfg, d.params.derivations), der.relation, der.formula))
    return rows


def table(cfg: Config, *, derived: bool = True, materials: bool = True, thermal_hook="auto") -> str:
    """Markdown of docs/PARAMETERS.md for ``cfg`` (deterministic; no timestamps)."""
    out = [f"# Parameters: `{cfg.name}`", "",
           f"Generated by `mrm params table {cfg.name}` from {', '.join(f'`{f}`' for f in cfg.files)}. Do not edit "
           f"by hand. Config hash `{cfg.hash()[:16]}`.", "",
           "Provenance ranks, strongest first: printed > private > read-off > calibrated > inferred > solver > "
           "literature > assumed > override. A derived value inherits the weakest provenance of its inputs.", ""]
    out += ["## Switches and settings", "", "| key | value |", "|---|---|"]
    for p, v in sorted(cfg.settings.items()):
        out.append(f"| `{p}` | {_md(format_value(v))} |")
    out += ["", "## Leaves", "",
            "| key | value | unit | provenance | generation | source | range | alternatives | note |",
            "|---|---|---|---|---|---|---|---|---|"]
    for p, lf in sorted(cfg.leaves.items()):
        if lf.is_free:
            continue
        alts = ", ".join(("**" + a + "**" if a == lf.selected_alt else a) for a in sorted(lf.alternatives))
        rng = format_value(list(lf.range)) if lf.range is not None else ""
        out.append(f"| `{p}` | {_md(format_value(lf.value))} | {_unit_label(p)} | {lf.provenance} | "
                   f"{lf.generation or ''} | {lf.source or ''} | {rng} | {_md(alts)} | {_md(lf.note)} |")
    if derived:
        d = derive(cfg, thermal_hook=thermal_hook)
        out += ["", "## Derived (nominal world, relations of 07 P0-3)", "",
                "| quantity | value | unit | provenance (weakest input) | relation | formula |",
                "|---|---|---|---|---|---|"]
        for r in _derived_rows(cfg, d):
            out.append("| `" + r[0] + "` | " + " | ".join(_md(x) for x in r[1:]) + " |")
        out += ["", "## Diagnostics (07 P0-1, P0-3, P1-6, P1-9)", "", "| diagnostic | value | unit | provenance |",
                "|---|---|---|---|"]
        for k, v in d.diagnostics.items():
            out.append(f"| `{k}` | {_md(format_value(np.asarray(v).ravel(), 10))} | {DIAGNOSTIC_UNITS.get(k, '')} | "
                       f"{d.provenance.get(k, '')} |")
        if d.warnings:
            out += ["", "Warnings:", ""] + [f"- {w}" for w in d.warnings]
    if cfg.uncertainty:
        out += ["", "## Uncertainty (one draw per world, 07 P0-4, 4.8)", "",
                "| entry | kind | target / choices | values | nominal | provenance | source |", "|---|---|---|---|---|---|---|"]
        from mrm.params.config import _hyp_choice_map
        for n, e in sorted(cfg.uncertainty.items()):
            f = e.fields
            if e.kind == "hypothesis":
                tgt = ", ".join(sorted(_hyp_choice_map(e)))
                w = f.get("weights")
                vals = ("weights " + ", ".join(f"{k}: {_fmt_num(x)}" for k, x in w.items())) if w else "equal weights"
            elif e.kind == "interval":
                tgt = e.target or "(world variable)"
                vals = f"[{f.get('low')}, {f.get('high')}] {f.get('dist', 'uniform')} {f.get('mode', 'absolute')}"
            else:
                tgt = e.target or "(world variable)"
                vals = f"{f.get('dist', 'normal')} mean {f.get('mean')} sigma {f.get('sigma')}"
            nom = e.pinned if e.pinned is not None else f.get("nominal", "leaf")
            out.append(f"| `{n}` | {e.kind} | {_md(tgt)} | {_md(vals)} | {_md(nom)} | {f.get('provenance', '')} | "
                       f"{f.get('source', '')} |")
    if cfg.variation or cfg.geometry:
        out += ["", "## Variation (per device, 07 P1-16)", "", "| path | level | dist | sigmas | provenance | source |",
                "|---|---|---|---|---|---|"]
        for p, e in sorted(cfg.variation.items()):
            f = e.fields
            sig = ", ".join(f"{k}={f[k]}" for k in ("die_sigma", "ring_sigma", "line_sigma", "rel_sigma",
                                                     "corr_length_um") if k in f)
            out.append(f"| `{p}` | {f.get('level', '')} | {f.get('dist', 'normal')} | {_md(sig)} | "
                       f"{f.get('provenance', '')} | {f.get('source', '')} |")
        if cfg.geometry:
            g = cfg.geometry
            for dim in ("width", "thickness", "slab", "gap_through", "gap_drop"):
                if dim in g:
                    out.append(f"| geometry.{dim} | die+ring | normal (Jacobian) | {_md(format_value(g[dim]))} | "
                               f"{g.get('provenance', '')} | {g.get('source', '')} |")
    if materials:
        try:
            from mrm.materials import material_table_markdown
        except ImportError:  # pragma: no cover - materials is part of the package
            material_table_markdown = None
        if material_table_markdown is not None:
            out += ["", "## Materials (mrm.materials)", "", material_table_markdown().rstrip("\n")]
    # meta.generation_mix and meta.inventory (docs/handoff/params_content.md section 2; 07 4.9 item 6 and
    # docs/research/00_prior_notes_checklist.md: every item modelled, excluded with a reason, or unknown)
    gm = (cfg.meta or {}).get("generation_mix")
    if gm:
        out += ["", "## Generation mix (07 4.9 item 6)", "", _md(gm)]
    inv = (cfg.meta or {}).get("inventory")
    if isinstance(inv, dict) and inv:
        out += ["", "## Inventory of docs/research/00_prior_notes_checklist.md", "",
                "Every item of the prior-notes checklist is a key, excluded with a reason, or unknown (with what stands "
                "in for it).", "", "| section | item | status | keys | reason |", "|---|---|---|---|---|"]
        for sec, items in inv.items():
            for item, e in (items or {}).items():
                e = e or {}
                keys = ", ".join(f"`{k}`" for k in (e.get("keys") or []))
                out.append(f"| {_md(sec)} | {_md(item)} | {_md(e.get('status', ''))} | {keys} | "
                           f"{_md(e.get('reason', ''))} |")
    dis = [(p, lf) for p, lf in sorted(cfg.leaves.items()) if lf.textbook or lf.disagrees]
    if dis:
        out += ["", "## Textbook comparison", "", "| key | value here | textbook field | disagreement |", "|---|---|---|---|"]
        for p, lf in dis:
            out.append(f"| `{p}` | {_md(format_value(lf.value))} {_unit_label(p)} | {_md(lf.textbook)} | "
                       f"{_md(lf.disagrees)} |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ explain

def _explain_derived(path: str, cfg: Config, d: Derived, depth: int, max_depth: int, lines: list[str],
                     stack: frozenset) -> None:
    pad = "  " * depth
    der = d.params.derivations.get(path)
    v = d.params.values.get(path)
    if path in cfg.leaves and not cfg.leaves[path].is_free:
        lf = cfg.leaves[path]
        lines.append(f"{pad}{path} = {format_value(lf.value)} {_unit_label(path)}  "
                     f"[{lf.provenance}, {lf.generation}, {lf.source}]")
        return
    if der is None:
        lines.append(f"{pad}{path}: (not set)")
        return
    if sch.lookup(path) is not None and v is not None:
        shown = f"{format_value(np.asarray(from_si(path, np.asarray(v, float))).ravel())} {_unit_label(path)}"
    else:
        shown = f"{format_value(np.asarray(v).ravel())} SI" if v is not None else "?"
    prov = effective_provenance(path, cfg, d.params.derivations)
    lines.append(f"{pad}{path} = {shown}  [derived by {der.relation}; provenance {prov} (weakest input)]")
    lines.append(f"{pad}  formula: {der.formula}")
    if der.note:
        lines.append(f"{pad}  note: {der.note}")
    if depth >= max_depth or path in stack:
        return
    for i in der.inputs:
        if i in cfg.leaves or i in d.params.derivations:
            _explain_derived(i, cfg, d, depth + 1, max_depth, lines, stack | {path})


def explain(cfg: Config, path: str, *, max_depth: int = 6, thermal_hook="auto") -> str:
    """Provenance chain of ``path``: a leaf, a relation output, a ``derived.*`` quantity, a derive() diagnostic or
    an ``uncertainty.<entry>``."""
    lines: list[str] = []
    if path.startswith("uncertainty."):
        e = cfg.uncertainty.get(path[len("uncertainty."):])
        if e is None:
            raise ConfigError(f"no uncertainty entry {path!r}")
        lines.append(f"{path}: {e.kind}" + (f", pinned to {e.pinned}" if e.pinned is not None else ""))
        for k, v in e.fields.items():
            lines.append(f"  {k}: {format_value(v)}")
        lines.append(f"  set in: {', '.join(e.origin)}")
        return "\n".join(lines)
    leaf = cfg.leaves.get(path)
    if leaf is not None and not leaf.is_free:
        lines.append(f"{path} = {format_value(leaf.value)} {_unit_label(path)}")
        lines.append(f"  provenance: {leaf.provenance}" + (f" (before override: {leaf.base_provenance})"
                                                            if leaf.base_provenance else ""))
        for f in ("generation", "source", "range", "note", "textbook", "disagrees", "targets", "residual",
                  "identifiable"):
            v = getattr(leaf, f)
            if v is not None:
                lines.append(f"  {f}: {format_value(v) if not isinstance(v, str) else v}")
        if leaf.artefact:
            a = leaf.artefact
            lines.append(f"  artefact: {a.get('file')}:{a.get('key')} sha256 {str(a.get('sha256'))[:12]}... "
                         f"(sidecar {a.get('sidecar')})")
        if leaf.alternatives:
            for n, a in sorted(leaf.alternatives.items()):
                mark = " <- selected" if n == leaf.selected_alt else ""
                av = a.get("value") if "value" in a else a.get("from")
                lines.append(f"  alternative {n}: {format_value(av)} ({a.get('provenance', leaf.provenance)}){mark}")
        lines.append("  chain: " + " -> ".join(leaf.origin))
        key = sch.lookup(path)
        if key is not None and key.relation:
            lines.append(f"  member of: {', '.join(key.relation)}")
        for n, e in sorted(cfg.uncertainty.items()):
            from mrm.params.config import _hyp_choice_map
            if e.target == path or (e.kind == "hypothesis" and any(path in m for m in _hyp_choice_map(e).values())):
                lines.append(f"  varied per world by uncertainty.{n} ({e.kind})")
        if path in cfg.variation:
            lines.append(f"  varied per device by variation.{path}")
        return "\n".join(lines)
    d = derive(cfg, thermal_hook=thermal_hook)
    if path in d.diagnostics:
        v = d.diagnostics[path]
        lines.append(f"{path} = {format_value(np.asarray(v).ravel())} {DIAGNOSTIC_UNITS.get(path, '')}  "
                     f"[diagnostic; provenance {d.provenance.get(path)} (weakest input)]")
        for i in d.inputs.get(path, []):
            if i in cfg.leaves or i in d.params.derivations:
                _explain_derived(i, cfg, d, 1, max_depth, lines, frozenset())
        return "\n".join(lines)
    if path in d.params.derivations:
        _explain_derived(path, cfg, d, 0, max_depth, lines, frozenset())
        return "\n".join(lines)
    sugg = sch.suggest(path)
    raise ConfigError(f"{path!r} is not a leaf, derived quantity or diagnostic of this config"
                      + (f" (did you mean {', '.join(sugg)}?)" if sugg else ""))


# ------------------------------------------------------------------------------------------------ diff

def diff(a: Config, b: Config) -> str:
    """Leaf-by-leaf differences between two configs (value, provenance, selected alternative), settings, and the
    uncertainty and variation entries. Empty string when they are equal."""
    out = []
    for p in sorted(set(a.leaves) | set(b.leaves)):
        la, lb = a.leaves.get(p), b.leaves.get(p)
        if la is None:
            out.append(f"+ {p} = {format_value(lb.value)} [{lb.provenance}]")
        elif lb is None:
            out.append(f"- {p} = {format_value(la.value)} [{la.provenance}]")
        else:
            va, vb = np.asarray(la.value, dtype=object), np.asarray(lb.value, dtype=object)
            same = va.shape == vb.shape and bool(np.all(va == vb))
            if not same or la.provenance != lb.provenance or la.selected_alt != lb.selected_alt:
                sa = f" alt={la.selected_alt}" if la.selected_alt else ""
                sb = f" alt={lb.selected_alt}" if lb.selected_alt else ""
                out.append(f"~ {p}: {format_value(la.value)} [{la.provenance}{sa}] -> "
                           f"{format_value(lb.value)} [{lb.provenance}{sb}]")
    for p in sorted(set(a.settings) | set(b.settings)):
        va, vb = a.settings.get(p), b.settings.get(p)
        if va != vb:
            out.append(f"~ {p}: {format_value(va)} -> {format_value(vb)}")
    for n in sorted(set(a.uncertainty) | set(b.uncertainty)):
        ea, eb = a.uncertainty.get(n), b.uncertainty.get(n)
        fa = (ea.fields, ea.pinned) if ea else None
        fb = (eb.fields, eb.pinned) if eb else None
        if fa != fb:
            out.append(f"~ uncertainty.{n}: {fa[1] if fa else 'absent'} -> {fb[1] if fb else 'absent'}"
                       if fa and fb and fa[0] == fb[0] else f"~ uncertainty.{n} differs")
    for p in sorted(set(a.variation) | set(b.variation)):
        ea, eb = a.variation.get(p), b.variation.get(p)
        if (ea.fields if ea else None) != (eb.fields if eb else None):
            out.append(f"~ variation.{p} differs")
    if a.geometry != b.geometry:
        out.append("~ variation.geometry differs")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ validate

def validate(cfg: Config, *, batch: int = 4, thermal_hook="auto") -> list[str]:
    """Derive the nominal view and realize a sampled batch (worlds and devices); return the warnings. Errors raise
    (ConfigError, OverdeterminedError, RealizeError)."""
    msgs = list(cfg.warnings)
    d = derive(cfg, thermal_hook=thermal_hook)
    msgs += d.warnings
    p = realize(cfg, batch, 0, thermal_hook=thermal_hook)
    if p.diagnostics.get("clipped"):
        msgs.append(f"variation clipped to the leaf range: {p.diagnostics['clipped']}")
    used = p.diagnostics.get("defaults_used", [])
    if used:
        msgs.append(f"{len(used)} optional keys absent (neutral values used): " + "; ".join(used))
    return msgs


# ------------------------------------------------------------------------------------------------ key list

def keys_markdown() -> str:
    """The schema key list as a Markdown table (the appendix of docs/PARAMETERS_FORMAT.md)."""
    def cond(c):
        return " and ".join(f"{p}={'|'.join(map(str, v))}" for p, v in c)
    out = ["| key | type | level | unit | rules | meaning |", "|---|---|---|---|---|---|"]
    for k in sch.SCHEMA.values():
        unit = parse_key(k.path)[1].expr or ""
        rules = []
        if not k.physical:
            rules.append("setting")
        if k.required:
            rules.append("required")
        if k.required_if:
            rules.append("required if " + cond(k.required_if))
        if k.forbidden_if:
            rules.append("forbidden if " + cond(k.forbidden_if))
        if k.derived_only:
            rules.append("derived only")
        if k.per_world:
            rules.append("per-world switch")
        if k.tick:
            rules.append(f"whole ticks ({k.tick})")
        if k.enum:
            rules.append("enum: " + ", ".join(map(str, k.enum)))
        if k.default is not None and not k.physical:
            rules.append(f"default {k.default}")
        t = k.type + (f" ({k.table_kind}: " + ", ".join(f.name for f in k.fields) + ")" if k.fields else "")
        out.append(f"| `{k.path}` | {t} | {k.level if k.physical else ''} | {unit} | {'; '.join(rules)} | "
                   f"{_md(k.doc)} |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ CLI

def _common(sp: argparse.ArgumentParser, preset: bool = True) -> None:
    if preset:
        sp.add_argument("preset", help="preset name or YAML path")
    sp.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE",
                    help="override a value (tagged provenance 'override'); repeatable")
    sp.add_argument("--alt", dest="alts", action="append", default=[], metavar="PATH=NAME",
                    help="select a leaf alternative or pin an uncertainty entry; repeatable")
    sp.add_argument("--sources", default=None, help="sources.yaml to check source keys against")
    sp.add_argument("--data-root", default=None, help="directory that holds extracted/ (default mrm/data)")
    sp.add_argument("--no-thermal-hook", action="store_true", help="do not call mrm.plant.thermal.assemble")


def _load(args, preset=None, extra_set=(), extra_alt=()):
    return load(preset or args.preset, list(args.overrides) + list(extra_set), list(args.alts) + list(extra_alt),
                sources=args.sources, data_root=args.data_root)


def _hook(args):
    return None if getattr(args, "no_thermal_hook", False) else "auto"


def _run_params(args) -> int:
    cmd = args.params_cmd
    try:
        if cmd == "show":
            print(show(_load(args)))
        elif cmd == "table":
            text = table(_load(args), derived=not args.no_derived, thermal_hook=_hook(args))
            if args.out:
                with open(args.out, "w") as f:
                    f.write(text)
                print(f"wrote {args.out}")
            else:
                sys.stdout.write(text)
        elif cmd == "explain":
            print(explain(_load(args), args.path, thermal_hook=_hook(args)))
        elif cmd == "diff":
            a = _load(args, args.preset)
            b = _load(args, args.other, args.set_b, args.alt_b)
            text = diff(a, b)
            print(text if text else "(no differences)")
        elif cmd == "keys":
            sys.stdout.write(keys_markdown())
        elif cmd == "validate":
            rc = 0
            for pr in args.presets:
                try:
                    msgs = validate(_load(args, pr), batch=args.batch, thermal_hook=_hook(args))
                    print(f"{pr}: OK")
                    for m in msgs:
                        print(f"  warning: {m}")
                except (ConfigError, ValueError) as e:
                    rc = 1
                    print(f"{pr}: FAILED\n  {e}")
            return rc
        else:  # pragma: no cover - argparse enforces the choices
            raise SystemExit(f"unknown params command {cmd}")
    except (ConfigError, ValueError) as e:     # ValueError: RealizeError, Over/UnderdeterminedError
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


def _setup_params(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="params_cmd", required=True)
    s = sub.add_parser("show", help="print the resolved config with provenance")
    _common(s)
    s = sub.add_parser("table", help="print the PARAMETERS.md table")
    _common(s)
    s.add_argument("--out", default=None, help="write to this file (e.g. docs/PARAMETERS.md)")
    s.add_argument("--no-derived", action="store_true", help="leaves only (no derive())")
    s = sub.add_parser("explain", help="print the provenance chain of a key, derived value or diagnostic")
    _common(s)
    s.add_argument("path", help="config path, derived.* name, diagnostic name or uncertainty.<entry>")
    s = sub.add_parser("diff", help="compare two configs")
    _common(s)
    s.add_argument("other", help="second preset name or YAML path")
    s.add_argument("--set-b", dest="set_b", action="append", default=[], help="override for the second config only")
    s.add_argument("--alt-b", dest="alt_b", action="append", default=[], help="alternative for the second config only")
    sub.add_parser("keys", help="print the schema key list (appendix of docs/PARAMETERS_FORMAT.md)")
    s = sub.add_parser("validate", help="load, derive and realize; exit 1 on any error")
    _common(s, preset=False)
    s.add_argument("presets", nargs="+", help="preset names or YAML paths")
    s.add_argument("--batch", type=int, default=4, help="sampled batch size for the realize check")


def register_cli(register) -> None:
    """Register the ``params`` command group with :func:`mrm.cli.register`."""
    register("params", "inspect presets: show | table | explain | diff | validate | keys", _setup_params,
             _run_params)
