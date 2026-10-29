# format parameter values, provenance, and checks for people reading a config.
# `show` lists resolved inputs.
# `table` builds deterministic Markdown with inputs, nominal derived values, diagnostics, and uncertainty.
# `explain` traces a value to its sources.
# `diff` compares two loaded configs.
# `validate` runs nominal derivation and a small sampled batch, then returns warnings.
# `keys_markdown` lists schema keys and their rules.
# `register_cli` exposes these functions through `mrm params`.
# stored config values use config units.
# derived plant values use SI and are converted back for display when a schema key defines their unit.
# diagnostics already use the units in `DIAGNOSTIC_UNITS`.

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from mrm.params import schema
from mrm.params.config import Config, ConfigError, load
from mrm.params.derive import DIAGNOSTIC_UNITS, Derived, derive, effective_provenance
from mrm.params.realize import realize
from mrm.params.units import from_si, parse_key

__all__ = ["show", "table", "explain", "diff", "validate", "register_cli", "format_value", "keys_markdown"]


# the unit suffix of a config path as written in the key (`pm_per_k`), or "" when it has none.
def _unit_label(path: str) -> str:
    try:
        _, spec = parse_key(path)
    except ValueError:
        return ""
    return spec.expr or ""


# short text of one number — 6 significant digits in [1e-3, 1e5), else 4.
def _format_number(x) -> str:
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


# format a scalar, list, array, or mapping for a compact text display.
# collapse exactly equal numeric values to one value and a count.
# otherwise show at most `max_items` values.
# summarize structured lists by item count and mappings by their keys.
def format_value(v: Any, max_items: int = 8) -> str:
    if v is None:
        return "null"
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        flat = np.asarray(v, dtype=object).ravel().tolist() if v and not isinstance(v[0], dict) else v
        if flat and all(isinstance(x, (int, float, np.integer, np.floating)) for x in flat):
            arr = np.asarray(flat, float)
            if arr.size > 1 and np.allclose(arr, arr[0], rtol=0, atol=0):
                return f"{_format_number(flat[0])} (x{arr.size})"
            items = [_format_number(x) for x in flat[:max_items]]
            return "[" + ", ".join(items) + (", ..." if len(flat) > max_items else "") + "]"
        return f"<{len(v)} items>"
    if isinstance(v, dict):
        return "{" + ", ".join(list(map(str, v))[:max_items]) + (", ..." if len(v) > max_items else "") + "}"
    return _format_number(v)


# text safe for one Markdown table cell (`|` escaped, newlines removed. None gives "").
def _md_cell(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ") if s is not None else ""


# show

# list resolved inputs with units, provenance, selections, and uncertainty entries.
def show(cfg: Config) -> str:
    lines = [f"# {cfg.name} ({cfg.kind}); files: {', '.join(cfg.files)}", f"# config hash {cfg.hash()[:12]}"]
    paths = sorted(set(cfg.leaves) | set(cfg.settings) | set(cfg.open_values))
    for path in paths:
        if path in cfg.leaves:
            leaf = cfg.leaves[path]
            if leaf.is_free:
                lines.append(f"{path} = <derived>")
                continue
            tag = ", ".join(str(x) for x in (leaf.provenance, leaf.generation, leaf.source) if x)
            alt = f" (alt: {leaf.selected_alt})" if leaf.selected_alt else ""
            unit = _unit_label(path)
            lines.append(f"{path} = {format_value(leaf.value)}{' ' + unit if unit else ''}  [{tag}]{alt}")
        elif path in cfg.settings:
            lines.append(f"{path} = {format_value(cfg.settings[path])}")
        else:
            lines.append(f"{path} = {format_value(cfg.open_values[path])}  (open)")
    for name, entry in sorted(cfg.uncertainty.items()):
        pin = f" pinned={entry.pinned}" if entry.pinned is not None else ""
        lines.append(f"uncertainty.{name}: {entry.kind} target={entry.target}{pin}")
    for path, entry in sorted(cfg.variation.items()):
        lines.append(f"variation.{path}: " + ", ".join(f"{k}={v}" for k, v in entry.fields.items()
                                                       if k not in ("provenance", "generation", "source", "note")))
    if cfg.geometry:
        lines.append("variation.geometry: " + ", ".join(k for k in cfg.geometry if k in
                                                        ("width", "thickness", "slab", "gap_through", "gap_drop")))
    return "\n".join(lines)


# table

# return display rows for derived values, excluding latency-in-ticks records.
# each row gives path, value, unit, provenance, relation, and formula.
# convert SI values back to config units for schema keys.
# keep other values in SI.
def _derived_rows(cfg: Config, d: Derived) -> list[tuple]:
    rows = []
    values = d.params.values
    for path, derivation in sorted(d.params.derivations.items()):
        value = values.get(path)
        if value is None or path.startswith("derived.timing"):
            continue
        key = schema.lookup(path)
        if key is not None:
            display_value = from_si(path, np.asarray(value, float))
            unit = _unit_label(path)
        else:
            display_value, unit = np.asarray(value), "SI"
        rows.append((path, format_value(np.asarray(display_value).ravel()), unit,
                     effective_provenance(path, cfg, d.params.derivations), derivation.relation, derivation.formula))
    return rows


# build deterministic Markdown for the parameter reference page.
# `derived` includes the nominal relation solve and diagnostics.
# `materials` includes the material reference table.
# also show uncertainty, variation, source inventory, and textbook disagreements when present.
# add no timestamps.
def table(cfg: Config, *, derived: bool = True, materials: bool = True, thermal_hook="auto") -> str:
    out = [f"# Parameters: `{cfg.name}`", "",
           f"Generated by `mrm params table {cfg.name}` from {', '.join(f'`{fields}`' for fields in cfg.files)}. Do not edit "
           f"by hand. Config hash `{cfg.hash()[:16]}`.", "",
           "Provenance ranks, strongest first: printed > private > read-off > calibrated > inferred > solver > "
           "literature > assumed > override. A derived value inherits the weakest provenance of its inputs.", ""]
    out += ["## Switches and settings", "", "| key | value |", "|---|---|"]
    for path, v in sorted(cfg.settings.items()):
        out.append(f"| `{path}` | {_md_cell(format_value(v))} |")
    out += ["", "## Leaves", "",
            "| key | value | unit | provenance | generation | source | range | alternatives | note |",
            "|---|---|---|---|---|---|---|---|---|"]
    for path, leaf in sorted(cfg.leaves.items()):
        if leaf.is_free:
            continue
        alternatives = ", ".join(("**" + a + "**" if a == leaf.selected_alt else a) for a in sorted(leaf.alternatives))
        range_text = format_value(list(leaf.range)) if leaf.range is not None else ""
        out.append(f"| `{path}` | {_md_cell(format_value(leaf.value))} | {_unit_label(path)} | {leaf.provenance} | "
                   f"{leaf.generation or ''} | {leaf.source or ''} | {range_text} | {_md_cell(alternatives)} | {_md_cell(leaf.note)} |")
    if derived:
        nominal_view = derive(cfg, thermal_hook=thermal_hook)
        out += ["", "## Derived (nominal world, relations R1-R5)", "",
                "| quantity | value | unit | provenance (weakest input) | relation | formula |",
                "|---|---|---|---|---|---|"]
        for row in _derived_rows(cfg, nominal_view):
            out.append("| `" + row[0] + "` | " + " | ".join(_md_cell(x) for x in row[1:]) + " |")
        out += ["", "## Diagnostics (mrm.params.derive)", "", "| diagnostic | value | unit | provenance |",
                "|---|---|---|---|"]
        for k, v in nominal_view.diagnostics.items():
            out.append(f"| `{k}` | {_md_cell(format_value(np.asarray(v).ravel(), 10))} | "
                       f"{DIAGNOSTIC_UNITS.get(k, '')} | {nominal_view.provenance.get(k, '')} |")
        if nominal_view.warnings:
            out += ["", "Warnings:", ""] + [f"- {w}" for w in nominal_view.warnings]
    if cfg.uncertainty:
        out += ["", "## Uncertainty (one draw per world)", "",
                "| entry | kind | target / choices | values | nominal | provenance | source |",
                "|---|---|---|---|---|---|---|"]
        from mrm.params.config import _hypothesis_choice_map
        for name, entry in sorted(cfg.uncertainty.items()):
            fields = entry.fields
            if entry.kind == "hypothesis":
                target_text = ", ".join(sorted(_hypothesis_choice_map(entry)))
                w = fields.get("weights")
                values_text = (("weights " + ", ".join(f"{k}: {_format_number(x)}" for k, x in w.items())) if w
                        else "equal weights")
            elif entry.kind == "interval":
                target_text = entry.target or "(world variable)"
                values_text = f"[{fields.get('low')}, {fields.get('high')}] {fields.get('dist',
                    'uniform')} {fields.get('mode', 'absolute')}"
            else:
                target_text = entry.target or "(world variable)"
                values_text = f"{fields.get('dist', 'normal')} mean {fields.get('mean')} sigma {fields.get('sigma')}"
            nominal = entry.pinned if entry.pinned is not None else fields.get("nominal", "leaf")
            out.append(f"| `{name}` | {entry.kind} | {_md_cell(target_text)} | {_md_cell(values_text)} | {_md_cell(nominal)} | "
                       f"{fields.get('provenance', '')} | {fields.get('source', '')} |")
    if cfg.variation or cfg.geometry:
        out += ["", "## Variation (per device)", "", "| path | level | dist | sigmas | provenance | source |",
                "|---|---|---|---|---|---|"]
        for path, entry in sorted(cfg.variation.items()):
            fields = entry.fields
            spreads = ", ".join(f"{k}={fields[k]}" for k in ("die_sigma", "ring_sigma", "line_sigma", "rel_sigma",
                                                     "corr_length_um") if k in fields)
            out.append(f"| `{path}` | {fields.get('level', '')} | {fields.get('dist',
                'normal')} | {_md_cell(spreads)} | "
                       f"{fields.get('provenance', '')} | {fields.get('source', '')} |")
        if cfg.geometry:
            g = cfg.geometry
            for dim in ("width", "thickness", "slab", "gap_through", "gap_drop"):
                if dim in g:
                    out.append(f"| geometry.{dim} | die+ring | normal (Jacobian) | "
                               f"{_md_cell(format_value(g[dim]))} | {g.get('provenance', '')} | "
                               f"{g.get('source', '')} |")
    if materials:
        try:
            from mrm.materials import material_table_markdown
        except ImportError:  # pragma: no cover - materials is part of the package
            material_table_markdown = None
        if material_table_markdown is not None:
            out += ["", "## Materials (mrm.materials)", "", material_table_markdown().rstrip("\n")]
    # show hardware generations and the coverage of the prior-notes checklist.
    # inventory entries state whether each item is modeled, excluded, or still unknown.
    generation_mix = (cfg.meta or {}).get("generation_mix")
    if generation_mix:
        out += ["", "## Generation mix", "", _md_cell(generation_mix)]
    inventory = (cfg.meta or {}).get("inventory")
    if isinstance(inventory, dict) and inventory:
        out += ["", "## Inventory of docs/research/00_prior_notes_checklist.md", "",
                "Every item of the prior-notes checklist is a key, excluded with a reason, or unknown (with what "
                "stands in for it).", "", "| section | item | status | keys | reason |", "|---|---|---|---|---|"]
        for section, items in inventory.items():
            for item, entry in (items or {}).items():
                entry = entry or {}
                keys = ", ".join(f"`{k}`" for k in (entry.get("keys") or []))
                out.append(f"| {_md_cell(section)} | {_md_cell(item)} | {_md_cell(entry.get('status', ''))} | {keys} | "
                           f"{_md_cell(entry.get('reason', ''))} |")
    comparisons = [(path, leaf) for path, leaf in sorted(cfg.leaves.items()) if leaf.textbook or leaf.disagrees]
    if comparisons:
        out += ["", "## Textbook comparison", "", "| key | value here | textbook field | disagreement |",
                "|---|---|---|---|"]
        for path, leaf in comparisons:
            out.append(f"| `{path}` | {_md_cell(format_value(leaf.value))} {_unit_label(path)} | {_md_cell(leaf.textbook)} | "
                       f"{_md_cell(leaf.disagrees)} |")
    return "\n".join(out) + "\n"


# explain

# append a value and recursively explain its inputs with indentation.
# stop at a given leaf, the maximum depth, or a dependency cycle.
# convert derived values to config units when their schema key defines those units.
def _explain_derived(path: str, cfg: Config, d: Derived, depth: int, max_depth: int, lines: list[str],
                     stack: frozenset) -> None:
    pad = "  " * depth
    derivation = d.params.derivations.get(path)
    value = d.params.values.get(path)
    if path in cfg.leaves and not cfg.leaves[path].is_free:
        leaf = cfg.leaves[path]
        lines.append(f"{pad}{path} = {format_value(leaf.value)} {_unit_label(path)}  "
                     f"[{leaf.provenance}, {leaf.generation}, {leaf.source}]")
        return
    if derivation is None:
        lines.append(f"{pad}{path}: (not set)")
        return
    if schema.lookup(path) is not None and value is not None:
        shown = f"{format_value(np.asarray(from_si(path, np.asarray(value, float))).ravel())} {_unit_label(path)}"
    else:
        shown = f"{format_value(np.asarray(value).ravel())} SI" if value is not None else "?"
    provenance = effective_provenance(path, cfg, d.params.derivations)
    lines.append(f"{pad}{path} = {shown}  [derived by {derivation.relation}; provenance {provenance} (weakest input)]")
    lines.append(f"{pad}  formula: {derivation.formula}")
    if derivation.note:
        lines.append(f"{pad}  note: {derivation.note}")
    if depth >= max_depth or path in stack:
        return
    for i in derivation.inputs:
        if i in cfg.leaves or i in d.params.derivations:
            _explain_derived(i, cfg, d, depth + 1, max_depth, lines, stack | {path})


# explain a physical leaf, derived value, diagnostic, or uncertainty entry.
# given leaves show metadata and their source history.
# derived values and diagnostics show the nominal calculation and recursively list their inputs.
# `max_depth` limits that trace.
# reject unknown paths with a name suggestion.
def explain(cfg: Config, path: str, *, max_depth: int = 6, thermal_hook="auto") -> str:
    lines: list[str] = []
    if path.startswith("uncertainty."):
        entry = cfg.uncertainty.get(path[len("uncertainty."):])
        if entry is None:
            raise ConfigError(f"no uncertainty entry {path!r}")
        lines.append(f"{path}: {entry.kind}" + (f", pinned to {entry.pinned}" if entry.pinned is not None else ""))
        for k, v in entry.fields.items():
            lines.append(f"  {k}: {format_value(v)}")
        lines.append(f"  set in: {', '.join(entry.origin)}")
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
                alternative_value = a.get("value") if "value" in a else a.get("from")
                lines.append(f"  alternative {n}: {format_value(alternative_value)} ({a.get('provenance',
                    leaf.provenance)}){mark}")
        lines.append("  chain: " + " -> ".join(leaf.origin))
        key = schema.lookup(path)
        if key is not None and key.relation:
            lines.append(f"  member of: {', '.join(key.relation)}")
        for n, entry in sorted(cfg.uncertainty.items()):
            from mrm.params.config import _hypothesis_choice_map
            if entry.target == path or (entry.kind == "hypothesis"
                                    and any(path in m for m in _hypothesis_choice_map(entry).values())):
                lines.append(f"  varied per world by uncertainty.{n} ({entry.kind})")
        if path in cfg.variation:
            lines.append(f"  varied per device by variation.{path}")
        return "\n".join(lines)
    nominal_view = derive(cfg, thermal_hook=thermal_hook)
    if path in nominal_view.diagnostics:
        v = nominal_view.diagnostics[path]
        lines.append(f"{path} = {format_value(np.asarray(v).ravel())} {DIAGNOSTIC_UNITS.get(path, '')}  "
                     f"[diagnostic; provenance {nominal_view.provenance.get(path)} (weakest input)]")
        for i in nominal_view.inputs.get(path, []):
            if i in cfg.leaves or i in nominal_view.params.derivations:
                _explain_derived(i, cfg, nominal_view, 1, max_depth, lines, frozenset())
        return "\n".join(lines)
    if path in nominal_view.params.derivations:
        _explain_derived(path, cfg, nominal_view, 0, max_depth, lines, frozenset())
        return "\n".join(lines)
    suggestions = schema.suggest(path)
    raise ConfigError(f"{path!r} is not a leaf, derived quantity or diagnostic of this config"
                      + (f" (did you mean {', '.join(suggestions)}?)" if suggestions else ""))


# diff

# compare physical values, provenance, selections, settings, and uncertainty.
# also compare device variation and geometry settings.
# return an empty string when these compared fields agree.
# this is not a comparison of every metadata field in the two configs.
def diff(a: Config, b: Config) -> str:
    out = []
    for p in sorted(set(a.leaves) | set(b.leaves)):
        left_leaf, right_leaf = a.leaves.get(p), b.leaves.get(p)
        if left_leaf is None:
            out.append(f"+ {p} = {format_value(right_leaf.value)} [{right_leaf.provenance}]")
        elif right_leaf is None:
            out.append(f"- {p} = {format_value(left_leaf.value)} [{left_leaf.provenance}]")
        else:
            left_value, right_value = np.asarray(left_leaf.value, dtype=object), np.asarray(right_leaf.value,
                dtype=object)
            same = left_value.shape == right_value.shape and bool(np.all(left_value == right_value))
            if not same or left_leaf.provenance != right_leaf.provenance or left_leaf.selected_alt != right_leaf.selected_alt:
                left_selection = f" alt={left_leaf.selected_alt}" if left_leaf.selected_alt else ""
                right_selection = f" alt={right_leaf.selected_alt}" if right_leaf.selected_alt else ""
                out.append(f"~ {p}: {format_value(left_leaf.value)} [{left_leaf.provenance}{left_selection}] -> "
                           f"{format_value(right_leaf.value)} [{right_leaf.provenance}{right_selection}]")
    for p in sorted(set(a.settings) | set(b.settings)):
        left_value, right_value = a.settings.get(p), b.settings.get(p)
        if left_value != right_value:
            out.append(f"~ {p}: {format_value(left_value)} -> {format_value(right_value)}")
    for n in sorted(set(a.uncertainty) | set(b.uncertainty)):
        left_entry, right_entry = a.uncertainty.get(n), b.uncertainty.get(n)
        left_spec = (left_entry.fields, left_entry.pinned) if left_entry else None
        right_spec = (right_entry.fields, right_entry.pinned) if right_entry else None
        if left_spec != right_spec:
            out.append(f"~ uncertainty.{n}: {left_spec[1] if left_spec else 'absent'} -> {right_spec[1] if right_spec else 'absent'}"
                       if left_spec and right_spec and left_spec[0] == right_spec[0] else f"~ uncertainty.{n} differs")
    for p in sorted(set(a.variation) | set(b.variation)):
        left_entry, right_entry = a.variation.get(p), b.variation.get(p)
        if (left_entry.fields if left_entry else None) != (right_entry.fields if right_entry else None):
            out.append(f"~ variation.{p} differs")
    if a.geometry != b.geometry:
        out.append("~ variation.geometry differs")
    return "\n".join(out)


# validate

# derive nominal values and realize a small batch using the experiment modes.
# return config and derivation warnings, clipped variation fractions, and the optional defaults used.
# invalid inputs raise their config or relation error.
def validate(cfg: Config, *, batch: int = 4, thermal_hook="auto") -> list[str]:
    messages = list(cfg.warnings)
    nominal_view = derive(cfg, thermal_hook=thermal_hook)
    messages += nominal_view.warnings
    params = realize(cfg, batch, 0, thermal_hook=thermal_hook)
    if params.diagnostics.get("clipped"):
        messages.append(f"variation clipped to the leaf range: {params.diagnostics['clipped']}")
    used = params.diagnostics.get("defaults_used", [])
    if used:
        messages.append(f"{len(used)} optional keys absent (neutral values used): " + "; ".join(used))
    return messages


# key list

# the schema key list as a Markdown table (the appendix of docs/PARAMETERS_FORMAT.md).
def keys_markdown() -> str:
    def cond(c):
        return " and ".join(f"{p}={'|'.join(map(str, v))}" for p, v in c)
    out = ["| key | type | level | unit | rules | meaning |", "|---|---|---|---|---|---|"]
    for key in schema.SCHEMA.values():
        unit = parse_key(key.path)[1].expr or ""
        rules = []
        if not key.physical:
            rules.append("setting")
        if key.required:
            rules.append("required")
        if key.required_if:
            rules.append("required if " + cond(key.required_if))
        if key.forbidden_if:
            rules.append("forbidden if " + cond(key.forbidden_if))
        if key.derived_only:
            rules.append("derived only")
        if key.per_world:
            rules.append("per-world switch")
        if key.tick:
            rules.append(f"whole ticks ({key.tick})")
        if key.enum:
            rules.append("enum: " + ", ".join(map(str, key.enum)))
        if key.default is not None and not key.physical:
            rules.append(f"default {key.default}")
        type_text = key.type + (f" ({key.table_kind}: " + ", ".join(f.name for f in key.fields) + ")" if key.fields else "")
        out.append(f"| `{key.path}` | {type_text} | {key.level if key.physical else ''} | {unit} | {'; '.join(rules)} | "
                   f"{_md_cell(key.doc)} |")
    return "\n".join(out) + "\n"


# CLI

# add the shared arguments (the preset unless preset=False, --set, --alt, --sources, --data-root, --no-thermal-hook) to one `params` subcommand parser.
def _add_common_args(sp: argparse.ArgumentParser, preset: bool = True) -> None:
    if preset:
        sp.add_argument("preset", help="preset name or YAML path")
    sp.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE",
                    help="override a value (tagged provenance 'override'); repeatable")
    sp.add_argument("--alt", dest="alts", action="append", default=[], metavar="PATH=NAME",
                    help="select a leaf alternative or pin an uncertainty entry; repeatable")
    sp.add_argument("--sources", default=None, help="sources.yaml to check source keys against")
    sp.add_argument("--data-root", default=None, help="directory that holds extracted/ (default mrm/data)")
    sp.add_argument("--no-thermal-hook", action="store_true", help="do not call mrm.plant.thermal.assemble")


# load the selected preset with CLI alternatives, overrides, and extra selections.
def _load_config(args, preset=None, extra_set=(), extra_alt=()):
    return load(preset or args.preset, list(args.overrides) + list(extra_set), list(args.alts) + list(extra_alt),
                sources=args.sources, data_root=args.data_root)


# the `thermal_hook` argument for derive and realize — None with --no-thermal-hook, else "auto".
def _thermal_hook_arg(args):
    return None if getattr(args, "no_thermal_hook", False) else "auto"


# dispatch a parameter subcommand and return 1 when a config or value check fails.
def _run_params(args) -> int:
    command = args.params_cmd
    try:
        if command == "show":
            print(show(_load_config(args)))
        elif command == "table":
            text = table(_load_config(args), derived=not args.no_derived, thermal_hook=_thermal_hook_arg(args))
            if args.out:
                with open(args.out, "w") as stream:
                    stream.write(text)
                print(f"wrote {args.out}")
            else:
                sys.stdout.write(text)
        elif command == "explain":
            print(explain(_load_config(args), args.path, thermal_hook=_thermal_hook_arg(args)))
        elif command == "diff":
            a = _load_config(args, args.preset)
            b = _load_config(args, args.other, args.set_b, args.alt_b)
            text = diff(a, b)
            print(text if text else "(no differences)")
        elif command == "keys":
            sys.stdout.write(keys_markdown())
        elif command == "validate":
            exit_code = 0
            for preset_name in args.presets:
                try:
                    messages = validate(_load_config(args, preset_name), batch=args.batch,
                        thermal_hook=_thermal_hook_arg(args))
                    print(f"{preset_name}: OK")
                    for m in messages:
                        print(f"  warning: {m}")
                except (ConfigError, ValueError) as e:
                    exit_code = 1
                    print(f"{preset_name}: FAILED\n  {e}")
            return exit_code
        else:  # pragma: no cover - argparse enforces the choices
            raise SystemExit(f"unknown params command {command}")
    except (ConfigError, ValueError) as e:     # value errors include realization and relation failures
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


# build the argument parsers of the `params` subcommands.
def _setup_params(p: argparse.ArgumentParser) -> None:
    commands = p.add_subparsers(dest="params_cmd", required=True)
    command_parser = commands.add_parser("show", help="print the resolved config with provenance")
    _add_common_args(command_parser)
    command_parser = commands.add_parser("table", help="print the PARAMETERS.md table")
    _add_common_args(command_parser)
    command_parser.add_argument("--out", default=None, help="write to this file (e.g. docs/PARAMETERS.md)")
    command_parser.add_argument("--no-derived", action="store_true", help="leaves only (no derive())")
    command_parser = commands.add_parser("explain",
        help="print the provenance chain of a key, derived value or diagnostic")
    _add_common_args(command_parser)
    command_parser.add_argument("path", help="config path, derived.* name, diagnostic name or uncertainty.<entry>")
    command_parser = commands.add_parser("diff", help="compare two configs")
    _add_common_args(command_parser)
    command_parser.add_argument("other", help="second preset name or YAML path")
    command_parser.add_argument("--set-b", dest="set_b", action="append", default=[],
        help="override for the second config only")
    command_parser.add_argument("--alt-b", dest="alt_b", action="append", default=[],
        help="alternative for the second config only")
    commands.add_parser("keys", help="print the schema key list (appendix of docs/PARAMETERS_FORMAT.md)")
    command_parser = commands.add_parser("validate", help="load, derive and realize; exit 1 on any error")
    _add_common_args(command_parser, preset=False)
    command_parser.add_argument("presets", nargs="+", help="preset names or YAML paths")
    command_parser.add_argument("--batch", type=int, default=4, help="sampled batch size for the realize check")


# register the `params` command group with `mrm.cli.register`.
def register_cli(register) -> None:
    register("params", "inspect presets: show | table | explain | diff | validate | keys", _setup_params,
             _run_params)
