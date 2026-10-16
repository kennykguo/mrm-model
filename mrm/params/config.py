"""Config loader with enforced provenance: ``load(preset|path, overrides, alts) -> Config`` (07 P1-14, P1-15).

What ``load`` does, in order:

1. Reads the YAML file (a preset name resolves to ``mrm/params/presets/<name>.yaml`` or ``presets/private/``),
   follows ``extends:`` (a name or a list; parents merge left to right, then the child) and merges leaf mappings
   field by field. A child that changes a leaf's ``value`` must restate ``provenance`` and ``generation`` (07 P1-14).
2. Enforces the leaf format. In a preset every physical number is a mapping ``{value, provenance, generation,
   source, range?, alternatives?, note?, textbook?, disagrees?, targets?, residual?, identifiable?}``. Bare physical
   numbers are allowed only in experiment files (``meta.kind: experiment``) and in overrides, and are tagged
   ``override``. A ``corner`` file may contain only ``extends``, ``meta`` and ``alts`` (07 D3).
3. Resolves ``{from: "extracted/<file>:<dotted.key>"}`` values from solver artefacts (provenance ``solver``) after a
   sha256 check against the JSON sidecar.
4. Applies ``alts`` ({path: name}; from the files, then the argument) and then ``overrides`` (a dict or a list of
   ``"path=value"`` strings).
5. Validates against :mod:`mrm.params.schema`: unknown keys (with did-you-mean), types, enums, fidelity rules,
   derived-only keys, ranges and hard bounds, the uncertainty and variation entry formats, and the timing rule that
   every hardware period and latency is a whole number of ticks (07 P0-6).

Values stay in the units of their key suffix inside a Config; :meth:`Config.si` and ``realize()`` convert to SI
exactly once.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from mrm.params import schema as sch
from mrm.params.units import parse_key, to_si

__all__ = [
    "Config", "Leaf", "UncertaintyEntry", "VariationEntry", "ConfigError", "ArtefactError", "load", "PROVENANCE_RANK",
    "PROVENANCE_WORDS", "GENERATIONS", "LEAF_FIELDS", "weakest", "provenance_rank", "PRESET_DIR", "DATA_ROOT",
    "parse_overrides", "load_sources", "ARTEFACT_FALLBACK_ALT",
]

PACKAGE_DIR = Path(__file__).resolve().parent
PRESET_DIR = PACKAGE_DIR / "presets"
DATA_ROOT = PACKAGE_DIR.parent / "data"           # "extracted/<file>" resolves under mrm/data/
SOURCES_FILE = PACKAGE_DIR / "sources.yaml"

#: closed provenance set, ranked strongest first (ARCHITECTURE section 1). ``private`` (NDA answers from
#: Lightmatter, 07 P1-14) ranks with measured data, just below printed; ``override`` ranks weakest because nothing
#: backs it. ``derived`` is not ranked: it resolves to the weakest of its inputs.
PROVENANCE_RANK = ("printed", "private", "read-off", "calibrated", "inferred", "solver", "literature", "assumed",
                   "override")
PROVENANCE_WORDS = frozenset(PROVENANCE_RANK) | {"derived"}
GENERATIONS = frozenset({"G25", "G26", "EVK100", "L20", "M1000", "Guide", "OCP", "OCI", "GF", "PAT", "none"})
LEAF_FIELDS = frozenset({"value", "from", "at", "provenance", "generation", "source", "range", "alternatives", "note",
                         "textbook", "disagrees", "targets", "residual", "identifiable"})
_ALT_FIELDS = frozenset({"value", "from", "at", "provenance", "generation", "source", "note"})
_STRICT_KINDS = ("preset", "corner", "private", "test")
#: name of the leaf alternative a missing ``from:`` artefact falls back to (docs/handoff/params_content.md 1)
ARTEFACT_FALLBACK_ALT = "fallback"
_PROV_RE = re.compile(r"^([a-z-]+)(?:\((.*)\))?$")

_UNC_COMMON = {"kind", "provenance", "generation", "source", "note", "textbook", "disagrees"}
_UNC_FIELDS = {
    "hypothesis": _UNC_COMMON | {"choices", "target", "nominal", "weights", "select"},
    "interval": _UNC_COMMON | {"target", "mode", "low", "high", "dist", "nominal", "corners", "select"},
    "dist": _UNC_COMMON | {"target", "mode", "dist", "mean", "sigma", "truncate", "nominal", "corners", "select"},
}
_VAR_FIELDS = {"level", "dist", "mode", "die_sigma", "ring_sigma", "line_sigma", "rel_sigma", "corr_length_um",
               "truncate", "provenance", "generation", "source", "note"}
_GEOM_DIMS = ("width", "thickness", "slab", "gap_through", "gap_drop")
_GEOM_DIM_FIELDS = {"die_sigma_nm", "ring_sigma_nm", "corr_length_um"}
_GEOM_FIELDS = set(_GEOM_DIMS) | {"jacobians", "width_closes_gaps", "provenance", "generation", "source", "note"}


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads 1e6 and 1.0e6 as floats (YAML 1.1 needs a sign in the exponent)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                  |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                  |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?)$""", re.X),
    list("-+0123456789."))


def yaml_load(text: str):
    """Parse YAML text with :class:`_Loader`."""
    return yaml.load(text, Loader=_Loader)


class ConfigError(ValueError):
    """One or more problems in a config. ``errors`` lists them all."""

    def __init__(self, errors: list[str] | str):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("\n  - " + "\n  - ".join(self.errors) if len(self.errors) > 1 else self.errors[0])


class ArtefactError(ConfigError):
    """A ``from:`` artefact is missing, has no sidecar, or fails its sha256 check."""


# ------------------------------------------------------------------------------------------------ provenance helpers

def provenance_rank(prov: str) -> int:
    """Rank of a provenance string (0 = strongest). ``word(qualifier)`` takes the weaker of the word and any
    provenance word named inside the qualifier, e.g. ``solver(assumed-geometry)`` ranks as ``assumed``."""
    m = _PROV_RE.match(prov or "")
    if not m:
        return len(PROVENANCE_RANK)
    word, qual = m.group(1), m.group(2)
    r = PROVENANCE_RANK.index(word) if word in PROVENANCE_RANK else len(PROVENANCE_RANK) - 1
    if qual:
        for tok in re.split(r"[^a-z-]+", qual):
            for w in PROVENANCE_RANK:
                if tok == w or tok.startswith(w + "-"):
                    r = max(r, PROVENANCE_RANK.index(w))
    return r


def weakest(provs: Iterable[str]) -> str:
    """The weakest provenance of a set (derived values inherit it, 07 P1-14)."""
    provs = [p for p in provs if p]
    if not provs:
        return "assumed"
    return max(provs, key=provenance_rank)


def _check_provenance_word(prov: Any, where: str, errs: list[str]) -> None:
    if not isinstance(prov, str):
        errs.append(f"{where}: provenance must be a string, got {prov!r}")
        return
    m = _PROV_RE.match(prov)
    if not m or m.group(1) not in PROVENANCE_WORDS:
        sugg = difflib.get_close_matches(prov, sorted(PROVENANCE_WORDS), n=1)
        hint = f" (did you mean {sugg[0]!r}?)" if sugg else ""
        errs.append(f"{where}: provenance {prov!r} is not one of {sorted(PROVENANCE_WORDS)}{hint}")


# ------------------------------------------------------------------------------------------------ data classes

@dataclass
class Leaf:
    """One physical leaf after merging, alternative selection and overrides. ``value`` is in config units."""
    path: str
    value: Any
    provenance: str
    generation: str | None = None
    source: str | None = None
    range: tuple | None = None
    alternatives: dict[str, dict] = field(default_factory=dict)
    note: str | None = None
    textbook: str | None = None
    disagrees: str | None = None
    targets: list | None = None
    residual: Any = None
    identifiable: Any = None
    origin: list[str] = field(default_factory=list)
    artefact: dict | None = None
    selected_alt: str | None = None
    base_provenance: str | None = None   # provenance before an override replaced it

    @property
    def is_free(self) -> bool:
        """True when a relation must compute this leaf (value null or provenance ``derived``)."""
        return self.value is None or self.provenance == "derived"


@dataclass
class UncertaintyEntry:
    """One epistemic entry: drawn once per world and shared by every ring (07 P0-4)."""
    name: str
    fields: dict
    origin: list[str] = field(default_factory=list)
    pinned: Any = None                 # a choice name (hypothesis) or a value (interval/dist) used in every world
    restrict: list[str] | None = None  # hypothesis choices still allowed after a leaf alternative was selected

    @property
    def kind(self) -> str:
        return self.fields.get("kind")

    @property
    def target(self) -> str | None:
        return self.fields.get("target")


@dataclass
class VariationEntry:
    """One per-device variation entry keyed by config path (07 P1-16)."""
    path: str
    fields: dict
    origin: list[str] = field(default_factory=list)


class Config:
    """A resolved, validated configuration. Values are in config units; :meth:`si` converts to SI."""

    def __init__(self):
        self.name: str = ""
        self.kind: str = "preset"
        self.files: list[str] = []
        self.leaves: dict[str, Leaf] = {}
        self.settings: dict[str, Any] = {}
        self.open_values: dict[str, Any] = {}
        self.setting_leaves: dict[str, Leaf] = {}     # settings that were written as provenance mappings
        self.uncertainty: dict[str, UncertaintyEntry] = {}
        self.variation: dict[str, VariationEntry] = {}
        self.geometry: dict | None = None
        self.relations_cfg: dict[str, dict] = {}
        self.alts_applied: dict[str, str] = {}
        self.overrides_applied: dict[str, Any] = {}
        self.sources: set[str] | None = None
        self.data_root: Path = DATA_ROOT
        self.meta: dict = {}
        self.warnings: list[str] = []

    # -------------------------------------------------------------------------------------------- access
    def has(self, path: str) -> bool:
        """True if ``path`` is set (a leaf, a setting or an open value)."""
        return path in self.leaves or path in self.settings or path in self.open_values

    def is_given(self, path: str) -> bool:
        """True if a physical leaf is present with a value (not null, not ``derived``)."""
        leaf = self.leaves.get(path)
        return leaf is not None and not leaf.is_free

    def get(self, path: str, default: Any = None) -> Any:
        """Value of a leaf (config units), a setting (or its schema default), or an open value."""
        if path in self.leaves:
            return self.leaves[path].value
        if path in self.settings:
            return self.settings[path]
        if path in self.open_values:
            return self.open_values[path]
        key = sch.lookup(path)
        if key is not None and key.default is not None:
            return key.default
        prefix = path + "."
        sub = {k[len(prefix):]: v for k, v in self.open_values.items() if k.startswith(prefix)}
        if sub:
            return sub
        return default

    def leaf(self, path: str) -> Leaf:
        """The Leaf at ``path`` (KeyError if absent)."""
        return self.leaves[path]

    def si(self, path: str, default: Any = None):
        """Value of ``path`` converted to SI (the single conversion, see mrm.params.units)."""
        v = self.get(path, None)
        if v is None:
            return default
        return to_si(path, v)

    def provenance(self, path: str) -> str:
        """Provenance word of a leaf."""
        return self.leaves[path].provenance

    @property
    def n_rings(self) -> int:
        """Number of rings N (array.n_rings)."""
        return int(self.get("array.n_rings"))

    @property
    def n_lines(self) -> int:
        """Number of laser lines L (channel_plan.lines table or channel_plan.n_lines)."""
        lines = self.get("channel_plan.lines")
        if lines:
            return len(lines)
        return int(self.get("channel_plan.n_lines"))

    def alt_value(self, path: str, name: str) -> dict:
        """Resolved alternative ``name`` of a leaf: {value, provenance, generation, source, artefact}."""
        leaf = self.leaves[path]
        if name not in leaf.alternatives:
            raise ConfigError(f"{path}: no alternative {name!r}; known: {sorted(leaf.alternatives)}")
        alt = leaf.alternatives[name]
        if "from" in alt or (isinstance(alt.get("value"), dict) and "from" in alt["value"]):
            spec = alt.get("from", alt.get("value", {}).get("from") if isinstance(alt.get("value"), dict) else None)
            at = alt.get("at")
            if isinstance(alt.get("value"), dict):
                at = alt["value"].get("at", at)
            val, info = resolve_from(spec, at, self.data_root)      # cached by file and mtime
            # a resolved copy: the stored alternative keeps its from: spec, so the resolved tree and the config
            # hash do not change when an alternative is first drawn
            alt = dict(alt, value=val, provenance=alt.get("provenance") or info["provenance"], artefact=info)
        return {"value": alt.get("value"), "provenance": alt.get("provenance", leaf.provenance),
                "generation": alt.get("generation", leaf.generation), "source": alt.get("source", leaf.source),
                "artefact": alt.get("artefact") or alt.get("_resolved_artefact")}

    # -------------------------------------------------------------------------------------------- export
    def resolved_tree(self) -> dict:
        """Nested dict of the resolved config: leaves as mappings with their provenance, settings bare."""
        out: dict = {}

        def put(path, v):
            node = out
            parts = path.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v

        for p, leaf in sorted(self.leaves.items()):
            d = {"value": _plain(leaf.value), "provenance": leaf.provenance}
            for f in ("generation", "source", "range", "note", "textbook", "disagrees", "targets"):
                v = getattr(leaf, f)
                if v is not None:
                    d[f] = _plain(v)
            if leaf.selected_alt:
                d["selected_alt"] = leaf.selected_alt
            if leaf.artefact:
                d["artefact"] = {k: leaf.artefact[k] for k in ("file", "key", "sha256") if k in leaf.artefact}
            if leaf.alternatives:
                # alternatives reach the plant through hypothesis draws, so they belong to the resolved config
                # (and to its hash: two configs that realize differently must not share a hash)
                alts = {}
                for an, a in sorted(leaf.alternatives.items()):
                    ad = {k: _plain(a[k]) for k in ("value", "from", "at", "provenance", "generation", "source")
                          if a.get(k) is not None}
                    art = a.get("artefact") or a.get("_resolved_artefact")
                    if art:
                        ad["artefact"] = {k: art[k] for k in ("file", "key", "sha256") if k in art}
                    alts[an] = ad
                d["alternatives"] = alts
            put(p, d)
        for p, v in sorted(self.settings.items()):
            put(p, _plain(v))
        for p, v in sorted(self.open_values.items()):
            put(p, _plain(v))
        if self.uncertainty:
            out["uncertainty"] = {n: dict(_plain(e.fields), **({"pinned": e.pinned} if e.pinned is not None else {}))
                                  for n, e in sorted(self.uncertainty.items())}
        if self.variation or self.geometry:
            var = {p: _plain(e.fields) for p, e in sorted(self.variation.items())}
            if self.geometry:
                var["geometry"] = _plain(self.geometry)
            out["variation"] = var
        if self.relations_cfg:
            out["relations"] = _plain(self.relations_cfg)
        return out

    def dump_yaml(self) -> str:
        """YAML text of :meth:`resolved_tree` (the experiment.resolved.yaml of 07 P1-17)."""
        return yaml.safe_dump(self.resolved_tree(), sort_keys=True, default_flow_style=None, width=120)

    def hash(self) -> str:
        """sha256 hex digest of the canonical resolved tree (for run manifests)."""
        blob = json.dumps(self.resolved_tree(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    def fingerprint(self) -> str:
        """Alias of :meth:`hash` (used by mrm.control.api.ControllerSpec.fingerprint)."""
        return self.hash()

    def to_dict(self) -> dict:
        """Alias of :meth:`resolved_tree`."""
        return self.resolved_tree()

    def without(self, prefixes, keep=()) -> "Config":
        """A copy without the leaves, settings and open values whose path starts with one of ``prefixes`` (e.g.
        ``"experiment."``), except the paths in ``keep``. The uncertainty, variation, geometry and relation tables
        are shared, not copied (read them only). Used for the nominal model a controller may realize
        (mrm.control.knowledge.NominalView.model_config, 07 D13): no experiment section, so no seeds."""
        prefixes = tuple(prefixes)
        keep = frozenset(keep)

        def kept(path: str) -> bool:
            return path in keep or not path.startswith(prefixes)
        c = copy.copy(self)
        c.leaves = {p: v for p, v in self.leaves.items() if kept(p)}
        c.settings = {p: v for p, v in self.settings.items() if kept(p)}
        c.open_values = {p: v for p, v in self.open_values.items() if kept(p)}
        c.setting_leaves = {p: v for p, v in self.setting_leaves.items() if kept(p)}
        c.overrides_applied = {p: v for p, v in self.overrides_applied.items() if kept(p)}
        return c

    def __repr__(self) -> str:
        return f"Config({self.name!r}, {len(self.leaves)} leaves, {len(self.uncertainty)} uncertainty entries)"


def _plain(v):
    """Make a value YAML/JSON-friendly (numpy -> python, tuples -> lists)."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        # string keys: YAML may give numeric keys (weights {160: 0.7}), and a mix of int and str keys cannot be
        # sorted by json.dumps(sort_keys=True) in Config.hash()
        return {str(k): _plain(x) for k, x in v.items()}
    return v


# ------------------------------------------------------------------------------------------------ artefacts

_ART_CACHE: dict[tuple, tuple] = {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_candidates(path: Path) -> list[Path]:
    """Sidecar file names accepted for an artefact, in order (ARCHITECTURE section 5)."""
    out = [path.with_name(path.name + ".sidecar.json"), path.with_name(path.stem + ".sidecar.json"),
           path.with_name(path.name + ".meta.json"), path.with_name(path.stem + ".meta.json")]
    if path.suffix != ".json":
        out.append(path.with_suffix(".json"))      # mrm.plant.thermal convention: <stem>.json beside the .npz
    return out


def resolve_from(spec: str, at: Mapping | None, data_root: Path) -> tuple[Any, dict]:
    """Load ``"extracted/<file>:<dotted.key>"`` from ``data_root``, verify its sha256, and return (value, info).

    ``at: {x_name: x}`` linearly interpolates a table node ``{x_name: [...], value: [...]}`` at x (in the table's
    units). ``info`` holds file, key, sha256, sidecar path and the provenance string (``solver`` or
    ``solver(<input provenance>-<input>)`` when the sidecar records the provenance of its inputs).
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ArtefactError(f"from: expects 'extracted/<file>:<dotted.key>', got {spec!r}")
    rel, key = spec.rsplit(":", 1)
    path = (Path(data_root) / rel).resolve()
    if not path.exists():
        raise ArtefactError(f"artefact {rel!r} not found under {data_root}")
    sidecar = next((p for p in sidecar_candidates(path) if p.exists()), None)
    if sidecar is None:
        raise ArtefactError(f"artefact {rel!r} has no sidecar (tried {[p.name for p in sidecar_candidates(path)]})")
    mtime = (path.stat().st_mtime_ns, sidecar.stat().st_mtime_ns)
    cache_key = (str(path), mtime)
    if cache_key in _ART_CACHE:
        data, meta, digest = _ART_CACHE[cache_key]
    else:
        meta = json.loads(sidecar.read_text())
        digest = _sha256(path)
        if meta.get("sha256") != digest:
            raise ArtefactError(f"artefact {rel!r}: sha256 {digest[:12]}... does not match its sidecar "
                                f"{str(meta.get('sha256'))[:12]}... ({sidecar.name})")
        if path.suffix == ".json":
            data = json.loads(path.read_text())
        elif path.suffix in (".yaml", ".yml"):
            data = yaml_load(path.read_text())
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
        else:
            raise ArtefactError(f"artefact {rel!r}: unsupported format {path.suffix}")
        _ART_CACHE[cache_key] = (data, meta, digest)
    node: Any = data
    for part in key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, (list, tuple)) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise ArtefactError(f"artefact {rel!r}: key {key!r} not found (failed at {part!r})")
    if at:
        if not isinstance(node, dict) or "value" not in node or len(at) != 1:
            raise ArtefactError(f"{spec}: 'at' needs a table {{x: [...], value: [...]}} and one x")
        (xname, x), = at.items()
        if xname not in node:
            raise ArtefactError(f"{spec}: table has no axis {xname!r}")
        xs = np.asarray(node[xname], float)
        ys = np.asarray(node["value"], float)
        if x < xs.min() or x > xs.max():
            raise ArtefactError(f"{spec}: {xname}={x} is outside the table range [{xs.min()}, {xs.max()}]")
        node = float(np.interp(x, xs, ys))
    if isinstance(node, np.ndarray):
        node = node.tolist()
    qual = []
    for iname, inp in (meta.get("inputs") or {}).items():
        if isinstance(inp, dict) and isinstance(inp.get("provenance"), str):
            qual.append(f"{inp['provenance']}-{iname}")
    prov = f"solver({', '.join(qual)})" if qual else "solver"
    info = {"file": rel, "key": key, "sha256": digest, "sidecar": sidecar.name, "provenance": prov,
            "schema": meta.get("schema"), "at": dict(at) if at else None}
    return copy.deepcopy(node), info


# ------------------------------------------------------------------------------------------------ file reading

def _resolve_file(name: str | Path, base: Path | None = None) -> Path:
    p = Path(name)
    cands = []
    if p.suffix in (".yaml", ".yml"):
        cands += [p] + ([base / p] if base and not p.is_absolute() else []) + [PRESET_DIR / p.name]
    else:
        cands += [PRESET_DIR / f"{name}.yaml", PRESET_DIR / "private" / f"{name}.yaml"]
        if base:
            cands.insert(0, base / f"{name}.yaml")
    for c in cands:
        if c.exists():
            return c.resolve()
    raise ConfigError(f"preset or file {str(name)!r} not found (looked in {[str(c) for c in cands]})")


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml_load(f.read()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def load_sources(path: Path | None = None) -> set[str] | None:
    """Source keys of ``mrm/params/sources.yaml`` (a mapping key -> citation, a list of keys, or a list of
    {key: ...}). Returns None when the file does not exist yet."""
    path = Path(path) if path else SOURCES_FILE
    if not path.exists():
        return None
    data = yaml_load(path.read_text()) or {}
    if isinstance(data, dict):
        if "sources" in data and isinstance(data["sources"], (dict, list)):
            data = data["sources"]
        if isinstance(data, dict):
            return set(map(str, data))
    keys = set()
    for item in data:
        if isinstance(item, str):
            keys.add(item)
        elif isinstance(item, dict):
            keys.add(str(item.get("key") or next(iter(item))))
    return keys


def _is_leaf_mapping(v: Any) -> bool:
    return isinstance(v, dict) and ("value" in v or "from" in v)


@dataclass
class _Node:
    """A flattened value during merging: a leaf mapping (``leaf``) or a bare value (``bare``)."""
    leaf: dict | None = None
    bare: Any = None
    is_bare: bool = False
    origin: list[str] = field(default_factory=list)


@dataclass
class _Flat:
    nodes: dict[str, _Node] = field(default_factory=dict)
    uncertainty: dict[str, dict] = field(default_factory=dict)
    unc_origin: dict[str, list[str]] = field(default_factory=dict)
    variation: dict[str, dict] = field(default_factory=dict)
    var_origin: dict[str, list[str]] = field(default_factory=dict)
    geometry: dict | None = None
    alts: dict[str, str] = field(default_factory=dict)
    relations: dict[str, dict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    files: list[str] = field(default_factory=list)


def _flatten_uncertainty(tree: dict, prefix: str, out: dict, errs: list[str], where: str) -> None:
    for k, v in tree.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict) and "kind" in v:
            out[name] = copy.deepcopy(v)
        elif isinstance(v, dict):
            _flatten_uncertainty(v, name + ".", out, errs, where)
        else:
            errs.append(f"{where}: uncertainty.{name} must be an entry mapping with 'kind' (07 P0-4), got {v!r}")


def _flatten_file(data: dict, fname: str, errs: list[str]) -> _Flat:
    """Flatten one file's tree into path -> node (leaf mapping or bare value)."""
    flat = _Flat(files=[fname])
    for top, v in data.items():
        if top == "extends":
            continue
        if top == "alts":
            if not isinstance(v, dict):
                errs.append(f"{fname}: alts must be a mapping {{path: alternative}}")
            else:
                flat.alts.update({str(a): str(b) for a, b in v.items()})
            continue
        if top == "relations":
            if not isinstance(v, dict):
                errs.append(f"{fname}: relations must be a mapping {{relation: {{tolerance_rel: ...}}}}")
            else:
                flat.relations.update(copy.deepcopy(v))
            continue
        if top == "uncertainty":
            if v:
                _flatten_uncertainty(v, "", flat.uncertainty, errs, fname)
                for n in flat.uncertainty:
                    flat.unc_origin.setdefault(n, []).append(fname)
            continue
        if top == "variation":
            for p, e in (v or {}).items():
                if p == "geometry":
                    flat.geometry = copy.deepcopy(e)
                elif isinstance(e, dict):
                    flat.variation[str(p)] = copy.deepcopy(e)
                    flat.var_origin.setdefault(str(p), []).append(fname)
                else:
                    errs.append(f"{fname}: variation.{p} must be an entry mapping keyed by config path, e.g. "
                                f"variation.ring.optics.resonance_offset_nm: {{die_sigma: 6.0, ring_sigma: 2.0, "
                                f"provenance: ...}} (07 P1-16; the v1 *_sigma_* keys are replaced by such entries)")
            continue
        if top == "meta":
            flat.meta = copy.deepcopy(v or {})
            continue
        if top not in sch.TOP_SECTIONS:
            sugg = difflib.get_close_matches(top, list(sch.TOP_SECTIONS) + list(sch.SPECIAL_TOP), n=1)
            hint = f" (did you mean {sugg[0]!r}?)" if sugg else ""
            msg = sch.DELETED.get(top)
            errs.append(f"{fname}: unknown top-level section {top!r}{hint}" + (f": {msg}" if msg else ""))
            continue
        _flatten_tree(v, top, flat, fname, errs)
    return flat


def _flatten_tree(v: Any, path: str, flat: _Flat, fname: str, errs: list[str]) -> None:
    key = sch.lookup(path)
    if sch.is_open(path) or (key is not None and key.type == "map" and not key.physical):
        flat.nodes[path] = _Node(bare=copy.deepcopy(v), is_bare=True, origin=[fname])
        return
    if _is_leaf_mapping(v):
        bad = set(v) - LEAF_FIELDS
        if bad:
            for b in sorted(bad):
                s = difflib.get_close_matches(b, sorted(LEAF_FIELDS), n=1)
                errs.append(f"{fname}: {path}: unknown leaf field {b!r}" + (f" (did you mean {s[0]!r}?)" if s else ""))
        flat.nodes[path] = _Node(leaf=copy.deepcopy(v), origin=[fname])
        return
    if isinstance(v, dict) and not (key is not None and key.type == "table"):
        if not v:
            return
        for k, sub in v.items():
            _flatten_tree(sub, f"{path}.{k}", flat, fname, errs)
        return
    flat.nodes[path] = _Node(bare=copy.deepcopy(v), is_bare=True, origin=[fname])


def _merge(parent: _Flat, child: _Flat, child_kind: str, errs: list[str]) -> _Flat:
    """Merge ``child`` over ``parent`` (07 P1-14: a child that changes a value restates provenance and generation)."""
    out = _Flat(nodes=dict(parent.nodes), uncertainty=copy.deepcopy(parent.uncertainty),
                unc_origin={k: list(v) for k, v in parent.unc_origin.items()},
                variation=copy.deepcopy(parent.variation), var_origin={k: list(v) for k, v in parent.var_origin.items()},
                geometry=copy.deepcopy(parent.geometry), alts=dict(parent.alts),
                relations=copy.deepcopy(parent.relations), meta=copy.deepcopy(parent.meta),
                files=parent.files + child.files)
    fname = child.files[0]
    for path, node in child.nodes.items():
        prev = out.nodes.get(path)
        if prev is None or prev.is_bare or node.is_bare:
            if prev is not None and not prev.is_bare and node.is_bare and child_kind in _STRICT_KINDS:
                key = sch.lookup(path)
                if key is not None and key.physical:
                    errs.append(f"{fname}: {path}: bare value replaces a provenance leaf; restate it as "
                                f"{{value, provenance, generation, source}} (07 P1-14)")
            new = _Node(leaf=copy.deepcopy(node.leaf), bare=copy.deepcopy(node.bare), is_bare=node.is_bare,
                        origin=(prev.origin if prev else []) + node.origin)
            out.nodes[path] = new
            continue
        merged = dict(prev.leaf)
        changed = False
        for vf in ("value", "from", "at"):
            if vf in node.leaf and node.leaf[vf] != prev.leaf.get(vf):
                changed = True
        if changed:
            missing = [f for f in ("provenance", "generation") if f not in node.leaf]
            if missing:
                errs.append(f"{fname}: {path}: changes the value of a parent leaf without restating "
                            f"{' and '.join(missing)} (07 P1-14)")
            for vf in ("value", "from", "at"):
                merged.pop(vf, None)
        if "alternatives" in node.leaf and "alternatives" in prev.leaf:
            alts = dict(prev.leaf["alternatives"] or {})
            alts.update(node.leaf["alternatives"] or {})
            # a child's `name: null` removes the parent's alternative (one whose meaning the child changed, e.g. a
            # leaf restated at another reference plane)
            for an in [an for an, a in (node.leaf["alternatives"] or {}).items() if a is None]:
                if an not in (prev.leaf["alternatives"] or {}):
                    errs.append(f"{fname}: {path}: alternative {an!r} is null, but the parent has no such alternative")
                alts.pop(an, None)
            merged.update(node.leaf)
            merged["alternatives"] = alts
        else:
            merged.update(node.leaf)
        out.nodes[path] = _Node(leaf=merged, origin=prev.origin + [f"{fname}" + (" (value changed)" if changed else "")])
    for n, e in child.uncertainty.items():
        if n in out.uncertainty and e.get("kind", out.uncertainty[n].get("kind")) == out.uncertainty[n].get("kind"):
            out.uncertainty[n].update(e)
        else:
            out.uncertainty[n] = e
        out.unc_origin.setdefault(n, []).append(fname)
    for p, e in child.variation.items():
        out.variation.setdefault(p, {}).update(e)
        out.var_origin.setdefault(p, []).append(fname)
    if child.geometry is not None:
        if out.geometry is None:
            out.geometry = child.geometry
        else:
            for k, v in child.geometry.items():
                if isinstance(v, dict) and isinstance(out.geometry.get(k), dict):
                    out.geometry[k].update(v)
                else:
                    out.geometry[k] = v
    out.alts.update(child.alts)
    for r, d in child.relations.items():
        out.relations.setdefault(r, {}).update(d if isinstance(d, dict) else {"value": d})
    srcs = dict(out.meta.get("sources") or {})
    srcs.update(child.meta.get("sources") or {})
    out.meta.update(child.meta)
    if srcs:
        out.meta["sources"] = srcs
    return out


def _load_chain(path: Path, stack: tuple[Path, ...], errs: list[str]) -> tuple[_Flat, str]:
    if path in stack:
        raise ConfigError(f"extends cycle: {' -> '.join(str(p.name) for p in stack + (path,))}")
    data = _read_yaml(path)
    kind = (data.get("meta") or {}).get("kind", "preset")
    if kind not in sch.SCHEMA["meta.kind"].enum:
        errs.append(f"{path.name}: meta.kind {kind!r} is not one of {sch.SCHEMA['meta.kind'].enum}")
    if kind == "corner":
        extra = set(data) - {"extends", "meta", "alts"}
        if extra:
            errs.append(f"{path.name}: a corner preset may contain only extends, meta and alts (07 D3); "
                        f"found {sorted(extra)}")
    try:
        rel = str(path.relative_to(PACKAGE_DIR.parent.parent))
    except ValueError:
        rel = str(path)
    flat = _flatten_file(data, rel, errs)
    for p, node in flat.nodes.items():
        if node.is_bare and kind in _STRICT_KINDS:
            key = sch.lookup(p)
            if key is not None and key.physical and node.bare is not None:
                errs.append(f"{rel}: {p}: bare value in a preset; write {{value, provenance, generation, source}} "
                            f"(07 P1-14; bare numbers are allowed only in overrides)")
    ext = data.get("extends")
    parents = [ext] if isinstance(ext, str) else list(ext or [])
    merged: _Flat | None = None
    for par in parents:
        ppath = _resolve_file(par, path.parent)
        pflat, _ = _load_chain(ppath, stack + (path,), errs)
        merged = pflat if merged is None else _merge(merged, pflat, "preset", errs)
    if merged is None:
        return flat, kind
    return _merge(merged, flat, kind, errs), kind


# ------------------------------------------------------------------------------------------------ overrides

def parse_overrides(overrides: Mapping[str, Any] | Iterable[str] | None) -> dict[str, Any]:
    """Normalize overrides: a dict {path: value} or strings "path=value" (value parsed as YAML)."""
    if overrides is None:
        return {}
    if isinstance(overrides, Mapping):
        return dict(overrides)
    out = {}
    for s in overrides:
        if not isinstance(s, str) or "=" not in s:
            raise ConfigError(f"override {s!r} must be 'path=value'")
        k, v = s.split("=", 1)
        out[k.strip()] = _enum_word(k.strip(), yaml_load(v))
    return out


def _enum_word(path: str, v: Any) -> Any:
    """YAML 1.1 reads off / on / no / yes as booleans. For an enum setting whose words do not include the boolean
    but include exactly one word of that meaning, the word is meant: ``--set experiment.variation=off`` is 'off',
    not False (experiments handoff request 8). Other values are returned unchanged."""
    if not isinstance(v, bool):
        return v
    key = sch.lookup(path)
    if key is None or key.type != "enum" or v in key.enum:
        return v
    words = [w for w in key.enum if isinstance(w, str) and w.lower() in (("on", "yes", "true") if v else
                                                                         ("off", "no", "false"))]
    return words[0] if len(words) == 1 else v


def _parse_alts(alts: Mapping[str, str] | Iterable[str] | None) -> dict[str, str]:
    if alts is None:
        return {}
    if isinstance(alts, Mapping):
        return {str(k): str(v) for k, v in alts.items()}
    out = {}
    for s in alts:
        if "=" not in s:
            raise ConfigError(f"alt {s!r} must be 'path=name'")
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ------------------------------------------------------------------------------------------------ load

def load(preset: str | Path | Mapping, overrides: Mapping[str, Any] | Iterable[str] | None = None,
         alts: Mapping[str, str] | Iterable[str] | None = None, *, sources: Iterable[str] | Path | None = None,
         data_root: str | Path | None = None) -> Config:
    """Load and validate a config.

    Args:
        preset: a preset name (``mrm/params/presets/<name>.yaml``), a YAML path, or an already-parsed mapping.
        overrides: ``{path: value}`` or ``["path=value", ...]``; physical values are tagged ``override``.
        alts: ``{path: alternative}`` or ``["path=name", ...]``. A leaf path selects one of its alternatives; an
            ``uncertainty.<entry>`` path pins that entry (a hypothesis choice, or low/high/nominal/<corner>).
        sources: allowed source keys (a set or a path); default ``mrm/params/sources.yaml`` plus ``meta.sources``.
        data_root: directory that holds ``extracted/`` (default ``mrm/data``).

    Raises:
        ConfigError (with every problem listed), ArtefactError.
    """
    errs: list[str] = []
    cfg = Config()
    cfg.data_root = Path(data_root) if data_root else DATA_ROOT
    if isinstance(preset, Mapping):
        data = copy.deepcopy(dict(preset))
        kind = (data.get("meta") or {}).get("kind", "preset")
        if data.get("extends"):
            raise ConfigError("a mapping preset cannot use extends:; write it to a file")
        flat = _flatten_file(data, "<mapping>", errs)
        for p, node in flat.nodes.items():
            key = sch.lookup(p)
            if node.is_bare and kind in _STRICT_KINDS and key is not None and key.physical and node.bare is not None:
                errs.append(f"<mapping>: {p}: bare value in a preset (07 P1-14)")
        cfg.name = str(data.get("meta", {}).get("name", "<mapping>"))
    else:
        path = _resolve_file(preset)
        flat, kind = _load_chain(path, (), errs)
        cfg.name = str(flat.meta.get("name", path.stem))
    cfg.kind = kind
    cfg.files = flat.files
    cfg.meta = flat.meta
    cfg.relations_cfg = flat.relations

    # sources
    if sources is None:
        known = load_sources()
    elif isinstance(sources, (str, Path)):
        known = load_sources(Path(sources))
    else:
        known = set(sources)
    extra = set((flat.meta.get("sources") or {}).keys())
    if known is None and kind == "test" and extra:
        cfg.sources = extra                        # a self-contained test preset checks against its own keys
    elif known is None:
        cfg.sources = None
        cfg.warnings.append("mrm/params/sources.yaml does not exist yet: source keys are not checked")
    else:
        cfg.sources = known | extra

    # leaves and settings
    for path, node in flat.nodes.items():
        key = sch.lookup(path)
        if sch.is_open(path):
            cfg.open_values[path] = node.bare
            continue
        if not node.is_bare and key is not None and not key.physical:
            lf = _make_leaf(path, node.leaf, node.origin, cfg, errs)
            cfg.settings[path] = lf.value
            cfg.setting_leaves[path] = lf
            continue
        if node.is_bare:
            if key is not None and key.physical and node.bare is not None:
                cfg.leaves[path] = Leaf(path=path, value=node.bare, provenance="override",
                                        origin=node.origin + ["bare value in an experiment file (tagged override)"])
            else:
                cfg.settings[path] = node.bare
            continue
        cfg.leaves[path] = _make_leaf(path, node.leaf, node.origin, cfg, errs)

    cfg.uncertainty = {n: UncertaintyEntry(n, e, flat.unc_origin.get(n, [])) for n, e in flat.uncertainty.items()}
    cfg.variation = {p: VariationEntry(p, e, flat.var_origin.get(p, [])) for p, e in flat.variation.items()}
    cfg.geometry = flat.geometry
    # bare physical numbers of an experiment file are overrides (07 P1-14): they hold in every world, like
    # overrides= values
    for path, leaf in cfg.leaves.items():
        if leaf.provenance == "override" and leaf.value is not None:
            _pin_for_leaf_override(cfg, path)

    # alts: files first, then the argument
    for p, name in list(flat.alts.items()) + list(_parse_alts(alts).items()):
        _apply_alt(cfg, p, name, errs)
    # overrides
    for p, v in parse_overrides(overrides).items():
        _apply_override(cfg, p, v, errs)

    try:
        _validate(cfg, errs)
    except Exception as e:  # noqa: BLE001 - a broken config may break validation; report what was found
        if not errs:
            raise
        errs.append(f"(validation stopped early: {type(e).__name__}: {e})")
    if errs:
        if any("artefact" in m for m in errs) and all(("artefact" in m) or ("from:" in m) for m in errs):
            raise ArtefactError(errs)
        raise ConfigError(errs)
    for w in cfg.warnings:
        warnings.warn(w, stacklevel=2)
    return cfg


def _make_leaf(path: str, d: dict, origin: list[str], cfg: Config, errs: list[str]) -> Leaf:
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
            val, art = resolve_from(spec, at, cfg.data_root)
        except ArtefactError as e:
            # a missing artefact FILE (not a bad sha256, sidecar or key) falls back to the leaf's 'fallback'
            # alternative, with a warning, until the extraction has produced it (docs/handoff/params_content.md 1)
            fb = (d.get("alternatives") or {}).get(ARTEFACT_FALLBACK_ALT)
            if "not found under" in str(e) and isinstance(fb, dict) and "value" in fb:
                val = fb["value"]
                for f in ("provenance", "generation", "source"):
                    if fb.get(f) is not None:
                        d[f] = fb[f]
                fell_back = True
                cfg.warnings.append(f"{path}: {e}; using its '{ARTEFACT_FALLBACK_ALT}' alternative {val!r} "
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
        errs.append(f"{path}: leaf has no provenance (ARCHITECTURE rule 1: a number without a provenance word is a bug)")
        prov = "assumed"
    alts = {}
    for name, a in (d.get("alternatives") or {}).items():
        if a is None:
            errs.append(f"{path}: alternative {name!r} is null; null only removes an inherited alternative in a child "
                        f"preset")
            continue
        if isinstance(a, dict) and ("value" in a or "from" in a):
            bad = set(a) - _ALT_FIELDS
            if bad:
                errs.append(f"{path}: alternative {name!r} has unknown fields {sorted(bad)}")
            alts[str(name)] = dict(a)
        else:
            alts[str(name)] = {"value": a, "_bare": True}
    if alts and "default" not in alts:
        alts["default"] = {"value": val, "provenance": prov, "generation": d.get("generation"),
                           "source": d.get("source"), "_resolved_artefact": art}
    rng = d.get("range")
    leaf = Leaf(path=path, value=val, provenance=prov, generation=d.get("generation"), source=d.get("source"),
                range=tuple(rng) if isinstance(rng, (list, tuple)) else rng, alternatives=alts, note=d.get("note"),
                textbook=d.get("textbook"), disagrees=d.get("disagrees"), targets=d.get("targets"),
                residual=d.get("residual"), identifiable=d.get("identifiable"), origin=list(origin), artefact=art)
    if fell_back:
        leaf.selected_alt = ARTEFACT_FALLBACK_ALT
        leaf.origin.append(f"artefact {spec!r} missing: alternative {ARTEFACT_FALLBACK_ALT!r}")
    return leaf


def _hyp_choice_map(entry: UncertaintyEntry) -> dict[str, dict[str, Any]]:
    """{choice: {path: alternative name | value}} of a hypothesis entry (normalizes the target + list form)."""
    ch = entry.fields.get("choices")
    tgt = entry.fields.get("target")
    if isinstance(ch, list):
        if not tgt:
            return {}
        return {str(c): {tgt: str(c)} for c in ch}
    if isinstance(ch, dict):
        out = {}
        for c, m in ch.items():
            if isinstance(m, dict):
                out[str(c)] = dict(m)
            elif tgt:
                out[str(c)] = {tgt: m}
            else:
                out[str(c)] = {}
        return out
    return {}


def _apply_alt(cfg: Config, path: str, name: str, errs: list[str]) -> None:
    if path.startswith("uncertainty."):
        ename = path[len("uncertainty."):]
        e = cfg.uncertainty.get(ename)
        if e is None:
            errs.append(f"alts: no uncertainty entry {ename!r}; known: {sorted(cfg.uncertainty)}")
            return
        if e.kind == "hypothesis":
            choices = _hyp_choice_map(e)
            if name not in choices:
                errs.append(f"alts: uncertainty.{ename} has no choice {name!r}; known: {sorted(choices)}")
                return
            e.pinned = name
            for p, a in choices[name].items():
                if p in cfg.leaves and isinstance(a, str):
                    _select_leaf_alt(cfg, p, a, f"pinned by uncertainty.{ename}={name}", errs)
                elif p in cfg.leaves and isinstance(a, dict):
                    leaf = cfg.leaves[p]
                    leaf.value = a.get("value")
                    leaf.provenance = a.get("provenance", leaf.provenance)
                    leaf.origin.append(f"set by uncertainty.{ename}={name}")
                elif sch.lookup(p) is not None and not sch.lookup(p).physical:
                    cfg.settings[p] = a
        else:
            corners = dict(e.fields.get("corners") or {})
            if name in ("low", "high"):
                if name not in e.fields:
                    errs.append(f"alts: uncertainty.{ename} has no {name!r}")
                    return
                e.pinned = e.fields[name]
            elif name == "nominal":
                e.pinned = "nominal"
            elif name in corners:
                c = corners[name]
                e.pinned = c["value"] if isinstance(c, dict) else c
            else:
                errs.append(f"alts: uncertainty.{ename} has no corner {name!r}; known: low, high, nominal, "
                            f"{sorted(corners)}")
                return
        cfg.alts_applied[path] = name
        return
    if path not in cfg.leaves:
        sugg = difflib.get_close_matches(path, list(cfg.leaves), n=1)
        errs.append(f"alts: {path!r} is not a leaf of this config" + (f" (did you mean {sugg[0]!r}?)" if sugg else ""))
        return
    _select_leaf_alt(cfg, path, name, "alt", errs)
    # keep hypothesis entries that set this leaf consistent with the selection
    for e in cfg.uncertainty.values():
        if e.kind != "hypothesis":
            if e.target == path and e.fields.get("mode", "absolute") == "absolute":
                e.pinned = "override"         # an absolute draw would replace the selected value: keep the selection
            continue
        cm = _hyp_choice_map(e)
        touching = {c: m[path] for c, m in cm.items() if path in m}
        if not touching:
            continue
        ok = [c for c, a in touching.items() if a == name]
        if not ok:
            errs.append(f"alts: {path}={name} is not reachable from uncertainty.{e.name} (choices "
                        f"{touching}); select the hypothesis with uncertainty.{e.name}=<choice> instead")
        elif len(ok) == 1:
            e.pinned = ok[0]
        else:
            e.restrict = ok
    cfg.alts_applied[path] = name


def _select_leaf_alt(cfg: Config, path: str, name: str, why: str, errs: list[str]) -> None:
    leaf = cfg.leaves[path]
    if name not in leaf.alternatives:
        errs.append(f"alts: {path} has no alternative {name!r}; known: {sorted(leaf.alternatives)}")
        return
    try:
        a = cfg.alt_value(path, name)
    except ArtefactError as e:
        errs.append(f"{path}: alternative {name!r}: {e}")
        return
    leaf.value = a["value"]
    leaf.provenance = a["provenance"]
    leaf.generation = a["generation"]
    leaf.source = a["source"]
    leaf.artefact = a["artefact"] if a["artefact"] else (leaf.artefact if name == "default" else None)
    leaf.selected_alt = name
    leaf.origin.append(f"{why}: alternative {name!r}")


def _pin_for_leaf_override(cfg: Config, path: str) -> None:
    """An override of a leaf (an ``overrides=`` value or a bare number in an experiment file) pins every uncertainty
    entry that targets it, so the user's value holds in every world."""
    for e in cfg.uncertainty.values():
        if e.target == path and e.kind in ("interval", "dist"):
            e.pinned = "override"
        elif e.kind == "hypothesis" and any(path in m for m in _hyp_choice_map(e).values()):
            cfg.warnings.append(f"override of {path} conflicts with hypothesis uncertainty.{e.name}; the "
                                f"hypothesis is pinned to its nominal choice and {path} keeps the override")
            e.pinned = e.fields.get("nominal")
            e.fields = dict(e.fields, _override_paths=sorted(set(e.fields.get("_override_paths", [])) | {path}))


def _apply_override(cfg: Config, path: str, v: Any, errs: list[str]) -> None:
    if path.startswith("uncertainty."):
        rest = path[len("uncertainty."):]
        for ename in sorted(cfg.uncertainty, key=len, reverse=True):
            if rest.startswith(ename + "."):
                fld = rest[len(ename) + 1:]
                cfg.uncertainty[ename].fields[fld] = v
                cfg.uncertainty[ename].origin.append(f"override {fld}={v!r}")
                cfg.overrides_applied[path] = v
                return
        errs.append(f"override {path!r}: no uncertainty entry; known: {sorted(cfg.uncertainty)}")
        return
    if path.startswith("variation."):
        rest = path[len("variation."):]
        if rest.startswith("geometry."):
            parts = rest.split(".")[1:]
            node = cfg.geometry if cfg.geometry is not None else {}
            cfg.geometry = node
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v
            cfg.overrides_applied[path] = v
            return
        for vp in sorted(cfg.variation, key=len, reverse=True):
            if rest.startswith(vp + "."):
                cfg.variation[vp].fields[rest[len(vp) + 1:]] = v
                cfg.variation[vp].origin.append(f"override {rest[len(vp) + 1:]}={v!r}")
                cfg.overrides_applied[path] = v
                return
        fld = rest.rsplit(".", 1)
        if len(fld) == 2 and fld[1] in _VAR_FIELDS:
            cfg.variation[fld[0]] = VariationEntry(fld[0], {fld[1]: v, "provenance": "override"}, ["override"])
            cfg.overrides_applied[path] = v
            return
        errs.append(f"override {path!r}: not a variation entry field")
        return
    key = sch.lookup(path)
    if sch.is_open(path):
        cfg.open_values[path] = v
        cfg.overrides_applied[path] = v
        return
    if key is None:
        # an override inside an open map whose parent is known (e.g. controller.params.kp)
        msg = sch.DELETED.get(path)
        if msg:
            errs.append(f"override {path!r}: {msg}")
        else:
            sugg = sch.suggest(path)
            errs.append(f"override {path!r}: unknown key" + (f" (did you mean {', '.join(sugg)}?)" if sugg else ""))
        return
    if key.physical:
        leaf = cfg.leaves.get(path)
        val = v["value"] if isinstance(v, dict) and "value" in v else v
        if leaf is None:
            cfg.leaves[path] = Leaf(path=path, value=val, provenance="override", origin=["override"])
        else:
            leaf.base_provenance = leaf.base_provenance or leaf.provenance
            leaf.origin.append(f"override (was {leaf.value!r}, {leaf.provenance})")
            leaf.value = val
            leaf.provenance = "override"
            leaf.selected_alt = None
            leaf.artefact = None
        _pin_for_leaf_override(cfg, path)
    else:
        cfg.settings[path] = v
        if key.per_world:
            # a per-world switch set by the user must hold in every world: pin each hypothesis that sets it to the
            # choice(s) giving this value (their other leaves come along); if no choice gives it, pin the hypothesis
            # to its nominal and keep the user's value for this switch (as for an overridden leaf)
            for e in cfg.uncertainty.values():
                if e.kind != "hypothesis":
                    continue
                cm = _hyp_choice_map(e)
                touching = {c: m[path] for c, m in cm.items() if path in m}
                if not touching:
                    continue
                ok = [c for c, a in touching.items() if a == v]
                if len(ok) == 1:
                    e.pinned = ok[0]
                elif ok:
                    e.restrict = ok
                else:
                    cfg.warnings.append(f"override {path}={v!r} is not a choice of hypothesis uncertainty.{e.name}; "
                                        f"the hypothesis is pinned to its nominal and {path} keeps the override")
                    e.pinned = e.fields.get("nominal")
                    e.fields = dict(e.fields, _override_paths=sorted(set(e.fields.get("_override_paths", []))
                                                                     | {path}))
    cfg.overrides_applied[path] = v


# ------------------------------------------------------------------------------------------------ validation

def _check_type(key: sch.Key, v: Any, where: str, n: dict[str, int], errs: list[str]) -> None:
    t = key.type

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

    if v is None:
        return
    if t == "float":
        if key.level in ("ring", "line") and isinstance(v, list):
            want = n.get(key.level)
            if want is not None and len(v) != want:
                errs.append(f"{where}: {key.level}-level list has {len(v)} values, expected {want}")
            if not all(num(x) for x in v):
                errs.append(f"{where}: expected numbers, got {v!r}")
        elif not num(v):
            errs.append(f"{where}: expected a number, got {v!r}")
    elif t == "int":
        vals = v if (isinstance(v, list) and key.level in ("ring", "line")) else [v]
        for x in vals:
            if not (isinstance(x, int) and not isinstance(x, bool)) and not (isinstance(x, float) and x.is_integer()):
                errs.append(f"{where}: expected an integer, got {x!r}")
    elif t == "bool":
        if not isinstance(v, bool):
            errs.append(f"{where}: expected true/false, got {v!r}")
    elif t == "str":
        if not isinstance(v, str):
            errs.append(f"{where}: expected a string, got {v!r}")
    elif t == "enum":
        if v not in key.enum:
            sugg = difflib.get_close_matches(str(v), [str(e) for e in key.enum], n=1)
            errs.append(f"{where}: {v!r} is not one of {list(key.enum)}" + (f" (did you mean {sugg[0]!r}?)" if sugg else ""))
    elif t.startswith("list_"):
        inner = {"list_float": num, "list_int": lambda x: isinstance(x, int) and not isinstance(x, bool),
                 "list_str": lambda x: isinstance(x, str)}[t]
        if not isinstance(v, list) or not all(inner(x) for x in v):
            errs.append(f"{where}: expected a list of {t[5:]}, got {v!r}")
    elif t == "map":
        if not isinstance(v, dict):
            errs.append(f"{where}: expected a mapping, got {v!r}")
    elif t == "table":
        recs = None
        if key.table_kind == "list":
            if not isinstance(v, list):
                errs.append(f"{where}: expected a list of records")
                return
            recs = [(f"[{i}]", r) for i, r in enumerate(v)]
        elif key.table_kind == "map":
            if not isinstance(v, dict):
                errs.append(f"{where}: expected a mapping of records")
                return
            if not key.fields:
                return
            recs = [(f".{k}", r) for k, r in v.items()]
        for tag, r in recs or []:
            if not isinstance(r, dict):
                errs.append(f"{where}{tag}: expected a record mapping")
                continue
            for fname, fv in r.items():
                fk = sch.table_field(key, fname)
                if fk is None:
                    sugg = difflib.get_close_matches(fname, [f.name for f in key.fields], n=1)
                    errs.append(f"{where}{tag}: unknown field {fname!r}" + (f" (did you mean {sugg[0]!r}?)" if sugg else ""))
                else:
                    _check_type(fk, fv, f"{where}{tag}.{fname}", n, errs)


def _in_bounds(v, lo, hi) -> bool:
    arr = np.asarray(v, dtype=float)
    return bool(np.all(arr >= lo) and np.all(arr <= hi))


def _numeric(v) -> bool:
    """True for a finite number or a flat list of finite numbers (bools excluded)."""
    def num(x):
        return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and math.isfinite(x)
    if isinstance(v, (list, tuple)):
        return len(v) > 0 and all(num(x) for x in v)
    return num(v)


def _check_tick(path: str, kind: str, v: Any, dt: float, errs: list[str]) -> None:
    if v is None or dt is None:
        return
    arr = np.atleast_1d(np.asarray(to_si(path, v), dtype=float))
    for x in arr:
        if kind == "rate":
            if x <= 0:
                continue
            ticks = 1.0 / x / dt
        else:
            ticks = x / dt
        if abs(ticks - round(ticks)) > 1e-6 * max(1.0, abs(ticks)):
            what = "period 1/" if kind == "rate" else ""
            errs.append(f"{path}: {what}{v!r} is not a whole number of ticks of sim.dt_plant_s={dt:g} s "
                        f"({ticks:.6g} ticks; 07 P0-6)")
            return


def _sizes(cfg: Config) -> dict[str, int]:
    """{'ring': N, 'line': L} as far as the config states them (for list-length checks)."""
    n = {}
    try:
        n["ring"] = cfg.n_rings
    except (TypeError, ValueError):
        pass
    try:
        n["line"] = cfg.n_lines
    except (TypeError, ValueError):
        pass
    return n


def _validate(cfg: Config, errs: list[str]) -> None:
    dt = cfg.get("sim.dt_plant_s")
    n = _sizes(cfg)
    strict = cfg.kind in _STRICT_KINDS
    # unknown keys, types, provenance fields
    for path, leaf in cfg.leaves.items():
        key = sch.lookup(path)
        if key is None:
            msg = sch.DELETED.get(path)
            if msg:
                errs.append(f"{path}: {msg}")
            else:
                sugg = sch.suggest(path)
                errs.append(f"unknown key {path!r}" + (f" (did you mean {', '.join(sugg)}?)" if sugg else ""))
            continue
        if not key.physical:  # pragma: no cover - settings never become leaves (see load)
            continue
        if key.derived_only and not leaf.is_free:
            rel = ", ".join(key.relation) or "a relation"
            errs.append(f"{path}: is computed by {rel}; a preset must not set it (07 P0-3)")
        if leaf.provenance == "derived" and leaf.value is not None:
            errs.append(f"{path}: a leaf tagged 'derived' must not carry a value; the relations compute it")
        if leaf.provenance == "derived" and not key.relation and not key.derived_only:
            errs.append(f"{path}: tagged 'derived' but no relation computes it (relation members are listed in "
                        f"docs/PARAMETERS_FORMAT.md section 8); give a value or leave the key out")
        _check_provenance_word(leaf.provenance, path, errs)
        if leaf.provenance.startswith("private") and cfg.kind != "private" and not any(
                "/private/" in f or f.startswith("private") for f in cfg.files):
            errs.append(f"{path}: provenance 'private' is allowed only in presets/private/ (07 P1-14)")
        if strict and leaf.provenance != "override" and not leaf.is_free:
            if leaf.generation is None:
                errs.append(f"{path}: leaf has no generation (one of {sorted(GENERATIONS)})")
            if leaf.source is None:
                errs.append(f"{path}: leaf has no source (a key of mrm/params/sources.yaml)")
        if leaf.generation is not None and leaf.generation not in GENERATIONS:
            sugg = difflib.get_close_matches(str(leaf.generation), sorted(GENERATIONS), n=1)
            errs.append(f"{path}: generation {leaf.generation!r} not in {sorted(GENERATIONS)}"
                        + (f" (did you mean {sugg[0]!r}?)" if sugg else ""))
        if leaf.source is not None and cfg.sources is not None and str(leaf.source) not in cfg.sources:
            sugg = difflib.get_close_matches(str(leaf.source), sorted(cfg.sources), n=1)
            errs.append(f"{path}: source {leaf.source!r} is not a key of sources.yaml or meta.sources"
                        + (f" (did you mean {sugg[0]!r}?)" if sugg else ""))
        if leaf.provenance.startswith("calibrated") and not leaf.targets:
            errs.append(f"{path}: a calibrated leaf must list its targets: [CAL-ids] (07 P0-5)")
        for an, a in leaf.alternatives.items():
            if a.get("_bare") and strict:
                errs.append(f"{path}: alternative {an!r} is a bare value; give {{value, provenance}} (07 P1-14)")
            elif "provenance" in a and a["provenance"] is not None:
                _check_provenance_word(a["provenance"], f"{path}.alternatives.{an}", errs)
            av = a.get("value")
            if an != "default" and av is not None and not (isinstance(av, dict) and "from" in av) and "from" not in a:
                # an alternative is a value of the same key: same type, list length and hard bounds. A selected
                # alternative is range-checked like any value, so in a preset the leaf's range must include every
                # alternative (params_content handoff 5.7: the range spans the stress values; the plausible span is
                # in the note); one outside could never be selected
                _check_type(key, av, f"{path}.alternatives.{an}", n, errs)
                if key.type in ("float", "int") and key.bounds is not None and _numeric(av) and not _in_bounds(
                        av, *key.bounds):
                    errs.append(f"{path}.alternatives.{an}: value {av!r} is outside the physical bounds "
                                f"{list(key.bounds)}")
                rng = leaf.range
                if (strict and key.type in ("float", "int") and isinstance(rng, tuple) and len(rng) == 2
                        and _numeric(list(rng)) and _numeric(av) and not _in_bounds(av, *rng)):
                    errs.append(f"{path}.alternatives.{an}: value {av!r} lies outside the leaf's range {list(rng)}, "
                                f"so it can never be selected; widen the range, or remove the alternative (null in a "
                                f"child preset)")
        if leaf.range is not None and not (isinstance(leaf.range, tuple) and len(leaf.range) == 2
                                           and all(_numeric(x) and not isinstance(x, list) for x in leaf.range)):
            errs.append(f"{path}: range must be [low, high] with two numbers, got {leaf.range!r}")
            leaf.range = None
        if leaf.value is None:
            continue
        _check_type(key, leaf.value, path, n, errs)
        if key.type in ("float", "int") and _numeric(leaf.value):
            if leaf.range is not None:
                lo, hi = leaf.range
                if lo > hi:
                    errs.append(f"{path}: range {list(leaf.range)} has low > high")
                elif not _in_bounds(leaf.value, lo, hi):
                    errs.append(f"{path}: value {leaf.value!r} is outside its own range {list(leaf.range)}")
            if key.bounds is not None and not _in_bounds(leaf.value, *key.bounds):
                errs.append(f"{path}: value {leaf.value!r} is outside the physical bounds {list(key.bounds)}")
        if key.tick and dt:
            _check_tick(path, key.tick, leaf.value, dt, errs)
        if key.type == "table" and key.fields and dt:
            recs = leaf.value if isinstance(leaf.value, list) else list((leaf.value or {}).values())
            for r in recs:
                if isinstance(r, dict):
                    for fname, fv in r.items():
                        fk = sch.table_field(key, fname)
                        if fk is not None and fk.tick:
                            _check_tick(f"{path}.{fname}", fk.tick, fv, dt, errs)
    for path, v in cfg.settings.items():
        key = sch.lookup(path)
        if key is None:
            msg = sch.DELETED.get(path)
            if msg:
                errs.append(f"{path}: {msg}")
            else:
                sugg = sch.suggest(path)
                errs.append(f"unknown key {path!r}" + (f" (did you mean {', '.join(sugg)}?)" if sugg else ""))
            continue
        if key.physical:
            continue
        _check_type(key, v, path, n, errs)
        if key.bounds is not None and isinstance(v, (int, float)) and not _in_bounds(v, *key.bounds):
            errs.append(f"{path}: {v!r} is outside {list(key.bounds)}")
        if key.tick and dt and v is not None:
            _check_tick(path, key.tick, v, dt, errs)
    # the line grid: a lines table, or centre + spacing + count
    if not cfg.get("channel_plan.lines") and any(cfg.get(k) is None for k in (
            "channel_plan.center_thz", "channel_plan.spacing_ghz", "channel_plan.n_lines")):
        errs.append("channel_plan: give channel_plan.lines, or center_thz, spacing_ghz and n_lines")
    # fidelity rules
    present = {p for p, lf in cfg.leaves.items() if lf.value is not None} | {
        p for p, v in cfg.settings.items() if v is not None}
    base_errs = sch.check_conditions(lambda p: cfg.get(p), present)
    errs.extend(base_errs)
    # the same rules in every world a hypothesis can reach: a per-world switch (e.g. ring.heater.driver_mode =
    # current) must find the keys it requires in the worlds that select it, not only in the nominal one
    for ename, e in sorted(cfg.uncertainty.items()):
        if e.kind != "hypothesis":
            continue
        for c, m in _hyp_choice_map(e).items():
            sw = {p: a for p, a in m.items() if (k := sch.lookup(p)) is not None and k.per_world and not k.physical}
            if not sw:
                continue
            for msg in sch.check_conditions(lambda p, sw=sw: sw[p] if p in sw else cfg.get(p), present):
                if msg not in base_errs:
                    errs.append(f"uncertainty.{ename}={c}: {msg}")
    # relations section
    from mrm.params.relations import RELATION_NAMES
    for r, d in cfg.relations_cfg.items():
        if r not in RELATION_NAMES:
            sugg = difflib.get_close_matches(r, RELATION_NAMES, n=1)
            errs.append(f"relations.{r}: unknown relation" + (f" (did you mean {sugg[0]!r}?)" if sugg else ""))
        elif not isinstance(d, dict) or set(d) - {"tolerance_rel", "note", "source"}:
            errs.append(f"relations.{r}: allowed fields are tolerance_rel, note, source")
        elif "tolerance_rel" in d and not (isinstance(d["tolerance_rel"], (int, float)) and d["tolerance_rel"] >= 0):
            errs.append(f"relations.{r}.tolerance_rel must be a number >= 0")
    _validate_uncertainty(cfg, strict, errs)
    _validate_variation(cfg, strict, errs)


def _entry_prov(fields: dict, where: str, strict: bool, cfg: Config, errs: list[str]) -> None:
    prov = fields.get("provenance")
    if prov is None:
        if strict:
            errs.append(f"{where}: entry has no provenance")
        return
    _check_provenance_word(prov, where, errs)
    if strict and prov != "override":
        for f in ("generation", "source"):
            if fields.get(f) is None:
                errs.append(f"{where}: entry has no {f}")
    g = fields.get("generation")
    if g is not None and g not in GENERATIONS:
        errs.append(f"{where}: generation {g!r} not in {sorted(GENERATIONS)}")
    s = fields.get("source")
    if s is not None and cfg.sources is not None and str(s) not in cfg.sources:
        errs.append(f"{where}: source {s!r} is not a known source key")


def _validate_uncertainty(cfg: Config, strict: bool, errs: list[str]) -> None:
    for name, e in cfg.uncertainty.items():
        where = f"uncertainty.{name}"
        kind = e.kind
        if kind not in _UNC_FIELDS:
            errs.append(f"{where}: kind {kind!r} must be one of {sorted(_UNC_FIELDS)}")
            continue
        bad = set(e.fields) - _UNC_FIELDS[kind] - {"_override_paths"}
        if bad:
            for b in sorted(bad):
                s = difflib.get_close_matches(b, sorted(_UNC_FIELDS[kind]), n=1)
                errs.append(f"{where}: unknown field {b!r} for kind {kind}" + (f" (did you mean {s[0]!r}?)" if s else ""))
        _entry_prov(e.fields, where, strict, cfg, errs)
        tgt = e.target
        if tgt is not None:
            key = sch.lookup(tgt)
            if key is None:
                sugg = sch.suggest(tgt)
                errs.append(f"{where}: target {tgt!r} is not a schema key" + (f" (did you mean {', '.join(sugg)}?)" if sugg else ""))
                continue
            if key.derived_only:
                errs.append(f"{where}: target {tgt!r} is derived; vary an input instead (07 P0-3)")
            if kind != "hypothesis" and key.physical and tgt not in cfg.leaves:
                errs.append(f"{where}: target {tgt!r} is not set in this config")
            if kind != "hypothesis" and cfg.is_given(tgt) is False and tgt in cfg.leaves:
                errs.append(f"{where}: target {tgt!r} is a free relation member (null or derived); vary an input")
        if kind == "hypothesis":
            cm = _hyp_choice_map(e)
            if not cm:
                errs.append(f"{where}: needs choices (a list with target:, or {{choice: {{path: alternative}}}})")
                continue
            nom = e.fields.get("nominal")
            if nom is None or str(nom) not in cm:
                errs.append(f"{where}: nominal {nom!r} must be one of the choices {sorted(cm)}")
            elif e.pinned is None and e.restrict is None:
                # the nominal world (derive(), the PARAMETERS table, explain) must be the config as written: the
                # nominal choice may only select what the leaves and switches already hold
                skip = set(e.fields.get("_override_paths", []))
                for p, a in cm[str(nom)].items():
                    if p in skip:
                        continue
                    k = sch.lookup(p)
                    if k is None:
                        continue
                    if not k.physical:
                        held = cfg.get(p)
                        ok = held == a
                    elif p in cfg.leaves and isinstance(a, str):
                        held = cfg.leaves[p].selected_alt or "default"
                        ok = held == a
                    elif p in cfg.leaves and isinstance(a, dict):
                        held = cfg.leaves[p].value
                        ok = held == a.get("value")
                    else:
                        continue
                    if not ok:
                        errs.append(f"{where}: the nominal choice {nom!r} sets {p} to {a!r} but the config holds "
                                    f"{held!r}; make the leaf (or switch) the nominal choice's value, or pin the "
                                    f"entry with alts (the nominal world must be the config as written)")
            w = e.fields.get("weights")
            if w is not None:
                if not isinstance(w, dict) or set(map(str, w)) - set(cm) or any(
                        not isinstance(x, (int, float)) or x < 0 for x in w.values()):
                    errs.append(f"{where}: weights must map choices to numbers >= 0")
            for c, m in cm.items():
                for p, a in m.items():
                    key = sch.lookup(p)
                    if key is None:
                        errs.append(f"{where}.choices.{c}: {p!r} is not a schema key")
                        continue
                    if key.physical and key.type != "enum":
                        if p not in cfg.leaves:
                            errs.append(f"{where}.choices.{c}: {p!r} is not set in this config")
                        elif cfg.leaves[p].is_free and key.type != "table":
                            errs.append(f"{where}.choices.{c}: {p} is a free relation member (null or derived); "
                                        f"a hypothesis can only set given leaves")
                        elif isinstance(a, str) and a not in cfg.leaves[p].alternatives:
                            errs.append(f"{where}.choices.{c}: {p} has no alternative {a!r}; known: "
                                        f"{sorted(cfg.leaves[p].alternatives)}")
                        elif not isinstance(a, (str, dict)) and strict:
                            errs.append(f"{where}.choices.{c}: give an alternative name or {{value, provenance}} "
                                        f"for {p}, not a bare number")
                        elif isinstance(a, dict):
                            if a.get("value") is None:
                                errs.append(f"{where}.choices.{c}: {p} needs {{value, provenance}}")
                            else:
                                _check_type(key, a["value"], f"{where}.choices.{c}.{p}", _sizes(cfg), errs)
                                if "provenance" in a:
                                    _check_provenance_word(a["provenance"], f"{where}.choices.{c}.{p}", errs)
                    elif not key.physical:
                        if not key.per_world:
                            errs.append(f"{where}.choices.{c}: {p} is not a per-world switch")
                        elif a not in key.enum:
                            errs.append(f"{where}.choices.{c}: {p}={a!r} not in {list(key.enum)}")
        else:
            mode = e.fields.get("mode", "absolute")
            if mode not in ("absolute", "scale", "offset"):
                errs.append(f"{where}: mode {mode!r} must be absolute, scale or offset")
            tkey = sch.lookup(tgt) if tgt is not None else None
            if tkey is not None and mode == "scale" and parse_key(tgt)[1].kind in ("celsius", "dbm"):
                errs.append(f"{where}: mode scale multiplies {tgt}, an affine (_c) or logarithmic (_dbm) value; use "
                            f"mode offset (07 P1-15)")
            lohi = []
            if kind == "interval":
                lo, hi = e.fields.get("low"), e.fields.get("high")
                if not all(_numeric(x) for x in (lo, hi)) or lo > hi:
                    errs.append(f"{where}: needs numbers low <= high")
                else:
                    lohi = [lo, hi]
                d = e.fields.get("dist", "uniform")
                if d not in ("uniform", "loguniform", "triangular"):
                    errs.append(f"{where}: dist {d!r} must be uniform, loguniform or triangular")
                if d == "loguniform" and _numeric(lo) and lo <= 0:
                    errs.append(f"{where}: loguniform needs low > 0")
                nom = e.fields.get("nominal")
                if d == "triangular" and lohi and _numeric(nom) and not lohi[0] <= nom <= lohi[1]:
                    errs.append(f"{where}: the triangular mode (nominal {nom}) must lie in [low, high]")
            else:
                d = e.fields.get("dist", "normal")
                if d not in ("normal", "lognormal"):
                    errs.append(f"{where}: dist {d!r} must be normal or lognormal")
                if not _numeric(e.fields.get("mean")) or not _numeric(e.fields.get("sigma")) or e.fields["sigma"] < 0:
                    errs.append(f"{where}: needs numbers mean and sigma >= 0")
                elif d == "lognormal" and e.fields["mean"] <= 0:
                    errs.append(f"{where}: lognormal needs mean > 0 (x = mean exp(sigma z); mean is the median)")
                tr = e.fields.get("truncate")
                if tr is not None:
                    if not (isinstance(tr, (list, tuple)) and len(tr) == 2 and _numeric(list(tr)) and tr[0] < tr[1]):
                        errs.append(f"{where}: truncate must be [low, high] with low < high")
                    else:
                        lohi = list(tr)
            vals = lohi + [c["value"] if isinstance(c, dict) else c for c in (e.fields.get("corners") or {}).values()]
            vals = [x for x in vals if _numeric(x)]
            if tkey is not None and mode == "absolute" and tkey.bounds is not None:
                if vals and not _in_bounds(vals, *tkey.bounds):
                    errs.append(f"{where}: values {vals} reach outside the physical bounds {list(tkey.bounds)} of {tgt}")
            rng = cfg.leaves[tgt].range if (tkey is not None and tgt in cfg.leaves) else None
            if mode == "absolute" and isinstance(rng, tuple) and len(rng) == 2 and _numeric(list(rng)):
                # a world's value of the target is the drawn or pinned value itself, so the leaf's declared range
                # must hold every one of them (params_content handoff 5.7: ranges include every selectable value)
                out = [x for x in vals if not float(rng[0]) <= float(x) <= float(rng[1])]
                if out:
                    errs.append(f"{where}: values {out} lie outside the range {list(rng)} of {tgt}; a leaf's range "
                                f"must include every value a world can take (widen it or move the bound/corner)")
            if tgt is None and e.fields.get("nominal") is None:
                errs.append(f"{where}: an entry without target needs a nominal value")
            nom = e.fields.get("nominal")
            if tgt is not None and nom is not None and not isinstance(nom, (int, float)):
                errs.append(f"{where}: nominal must be a number")
            for cn, c in (e.fields.get("corners") or {}).items():
                cv = c["value"] if isinstance(c, dict) else c
                if not isinstance(cv, (int, float)):
                    errs.append(f"{where}.corners.{cn}: must be a number or {{value, ...}}")


def _validate_variation(cfg: Config, strict: bool, errs: list[str]) -> None:
    for path, e in cfg.variation.items():
        where = f"variation.{path}"
        bad = set(e.fields) - _VAR_FIELDS
        for b in sorted(bad):
            s = difflib.get_close_matches(b, sorted(_VAR_FIELDS), n=1)
            errs.append(f"{where}: unknown field {b!r}" + (f" (did you mean {s[0]!r}?)" if s else ""))
        _entry_prov(e.fields, where, strict, cfg, errs)
        key = sch.lookup(path)
        if key is None:
            sugg = sch.suggest(path)
            errs.append(f"{where}: {path!r} is not a schema key (variation is keyed by config path, 07 P1-16)"
                        + (f" (did you mean {', '.join(sugg)}?)" if sugg else ""))
            continue
        if key.derived_only:
            errs.append(f"{where}: {path} is derived; vary an input (07 P0-3)")
        if path not in cfg.leaves:
            errs.append(f"{where}: {path} is not set in this config")
        elif cfg.leaves[path].is_free:
            errs.append(f"{where}: {path} is a free relation member; vary an input")
        if key.type != "float" or key.level == "global":
            errs.append(f"{where}: only die-, ring- or line-level float leaves can vary (level of {path}: {key.level})")
        lvl = e.fields.get("level", key.level)
        allowed = {"die": ("die",), "ring": ("die", "ring"), "line": ("die", "line")}.get(key.level, ())
        if lvl not in allowed:
            errs.append(f"{where}: level {lvl!r} not allowed for a {key.level}-level leaf (allowed {list(allowed)})")
        dist = e.fields.get("dist", "normal")
        if dist not in ("normal", "lognormal", "uniform"):
            errs.append(f"{where}: dist {dist!r} must be normal, lognormal or uniform")
        mode = e.fields.get("mode", "scale" if (dist == "lognormal" or "rel_sigma" in e.fields) else "offset")
        if mode not in ("offset", "scale"):
            errs.append(f"{where}: mode {mode!r} must be offset or scale")
        if dist == "lognormal" and mode != "scale":
            errs.append(f"{where}: lognormal variation is multiplicative (mode: scale)")
        if mode == "scale" and parse_key(path)[1].kind in ("celsius", "dbm"):
            errs.append(f"{where}: a multiplicative (scale, lognormal or rel_sigma) variation of an affine (_c) or "
                        f"logarithmic (_dbm) value has no meaning; use mode offset with sigmas in K or dB (07 P1-15)")
        corr = e.fields.get("corr_length_um")
        if corr is not None and not (_numeric(corr) and corr > 0):
            errs.append(f"{where}.corr_length_um: must be a number > 0")
        sig = [f for f in ("die_sigma", "ring_sigma", "line_sigma", "rel_sigma") if f in e.fields]
        if not sig:
            errs.append(f"{where}: needs die_sigma, ring_sigma/line_sigma or rel_sigma")
        for f in sig:
            if not isinstance(e.fields[f], (int, float)) or e.fields[f] < 0:
                errs.append(f"{where}.{f}: must be a number >= 0")
        if "ring_sigma" in e.fields and lvl != "ring":
            errs.append(f"{where}: ring_sigma needs level ring")
        if "line_sigma" in e.fields and lvl != "line":
            errs.append(f"{where}: line_sigma needs level line")
        if "rel_sigma" in e.fields and mode != "scale":
            errs.append(f"{where}: rel_sigma is a relative (scale) variation")
    g = cfg.geometry
    if g:
        bad = set(g) - _GEOM_FIELDS
        for b in sorted(bad):
            s = difflib.get_close_matches(b, sorted(_GEOM_FIELDS), n=1)
            errs.append(f"variation.geometry: unknown field {b!r}" + (f" (did you mean {s[0]!r}?)" if s else ""))
        _entry_prov(g, "variation.geometry", strict, cfg, errs)
        for d in _GEOM_DIMS:
            if d in g:
                if not isinstance(g[d], dict) or set(g[d]) - _GEOM_DIM_FIELDS:
                    errs.append(f"variation.geometry.{d}: fields must be {sorted(_GEOM_DIM_FIELDS)}")
                    continue
                for f, x in g[d].items():
                    if not _numeric(x) or x < 0 or (f == "corr_length_um" and x == 0):
                        errs.append(f"variation.geometry.{d}.{f}: must be a number >= 0 (corr_length_um > 0)")
        if "width_closes_gaps" in g and not isinstance(g["width_closes_gaps"], bool):
            errs.append("variation.geometry.width_closes_gaps: must be true or false")
        jac = g.get("jacobians", ["ring.geometry.jacobian"])
        if isinstance(jac, str):
            jac = [jac]
        for jp in jac:
            if jp not in cfg.leaves:
                errs.append(f"variation.geometry: Jacobian leaf {jp!r} is not set (07 P1-16: extraction stores the "
                            f"Jacobians; inline them with provenance until it does)")
