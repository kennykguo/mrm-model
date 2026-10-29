# load and check the physical parameters and settings used by a simulation.
# `load` accepts a preset name, a YAML path, or a mapping.
# it returns a `Config`.
# a physical parameter is stored as a `Leaf` with its value and provenance — the record of where that value came from.
# the load order is inheritance, artefact resolution, alternative selection, overrides, then validation.
# later stages take precedence.
# parents merge from left to right before the child is applied.
# a changed physical value must state its provenance and generation again in a preset.
# values in `Config` keep the units named by their key suffix.
# `Config.si` returns an SI conversion without changing the stored value.
# `realize` builds the SI arrays that the plant uses.
# validation checks names, types, bounds, provenance, uncertainty, variation, and periods expressed in plant ticks.

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

from mrm.params import schema
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

# the closed set of provenance words, ranked strongest first.
# `private` (NDA answers from Lightmatter) ranks with measured data, just below printed.
# `override` has no supporting source.
# `derived` is not ranked — it takes the weakest rank of its inputs.
PROVENANCE_RANK = ("printed", "private", "read-off", "calibrated", "inferred", "solver", "literature", "assumed",
                   "override")
PROVENANCE_WORDS = frozenset(PROVENANCE_RANK) | {"derived"}
# hardware-generation tags a leaf may carry ("none" for no generation)
GENERATIONS = frozenset({"G25", "G26", "EVK100", "L20", "M1000", "Guide", "OCP", "OCI", "GF", "PAT", "none"})
# fields a provenance leaf mapping may have
LEAF_FIELDS = frozenset({"value", "from", "at", "provenance", "generation", "source", "range", "alternatives", "note",
                         "textbook", "disagrees", "targets", "residual", "identifiable"})
# allowed fields for a leaf alternative
_ALTERNATIVE_FIELDS = frozenset({"value", "from", "at", "provenance", "generation", "source", "note"})
# these file kinds need provenance, generation, and source for physical values.
# experiment files are not strict
_STRICT_KINDS = ("preset", "corner", "private", "test")
# name of the leaf alternative that a missing `from:` artefact falls back to, until the extraction has made it
ARTEFACT_FALLBACK_ALT = "fallback"
# provenance string — a word with an optional qualifier in parentheses
_PROVENANCE_RE = re.compile(r"^([a-z-]+)(?:\((.*)\))?$")

# allowed fields of an uncertainty entry, by kind
_UNCERTAINTY_COMMON_FIELDS = {"kind", "provenance", "generation", "source", "note", "textbook", "disagrees"}
_UNCERTAINTY_FIELDS = {
    "hypothesis": _UNCERTAINTY_COMMON_FIELDS | {"choices", "target", "nominal", "weights", "select", "follows", "map"},
    "interval": _UNCERTAINTY_COMMON_FIELDS | {"target", "mode", "low", "high", "dist", "nominal", "corners", "select"},
    "dist": _UNCERTAINTY_COMMON_FIELDS | {"target", "mode", "dist", "mean", "sigma", "truncate", "nominal", "corners",
                                          "select"},
}
# allowed fields of a path-keyed variation entry
_VARIATION_FIELDS = {"level", "dist", "mode", "die_sigma", "ring_sigma", "line_sigma", "rel_sigma", "corr_length_um",
                     "truncate", "provenance", "generation", "source", "note"}
# the correlated geometry errors of variation.geometry, and the fields of each one and of the whole block
_GEOMETRY_DIMS = ("width", "thickness", "slab", "gap_through", "gap_drop")
_GEOMETRY_DIM_FIELDS = {"die_sigma_nm", "ring_sigma_nm", "corr_length_um"}
_GEOMETRY_FIELDS = set(_GEOMETRY_DIMS) | {"jacobians", "width_closes_gaps", "provenance", "generation", "source",
                                          "note"}


# safeLoader that also reads 1e6 and 1.0e6 as floats (YAML 1.1 needs a sign in the exponent).
class _Loader(yaml.SafeLoader):
    pass


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                  |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                  |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?)$""", re.X),
    list("-+0123456789."))


# parse YAML text with `_Loader`.
def yaml_load(text: str):
    return yaml.load(text, Loader=_Loader)


# one or more problems in a config.
# `errors` lists them all.
class ConfigError(ValueError):
    def __init__(self, errors: list[str] | str):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("\n  - " + "\n  - ".join(self.errors) if len(self.errors) > 1 else self.errors[0])


# a `from:` artefact is missing, has no sidecar, or fails its sha256 check.
class ArtefactError(ConfigError):
    pass


# provenance helpers

# return the provenance rank, where zero is strongest.
# use the weakest word found in the base word or its qualifier.
# for example, `solver(assumed-geometry)` has the rank of `assumed`.
def provenance_rank(prov: str) -> int:
    match = _PROVENANCE_RE.match(prov or "")
    if not match:
        return len(PROVENANCE_RANK)
    word, qualifier = match.group(1), match.group(2)
    rank = PROVENANCE_RANK.index(word) if word in PROVENANCE_RANK else len(PROVENANCE_RANK) - 1
    if qualifier:
        for token in re.split(r"[^a-z-]+", qualifier):
            for candidate in PROVENANCE_RANK:
                if token == candidate or token.startswith(candidate + "-"):
                    rank = max(rank, PROVENANCE_RANK.index(candidate))
    return rank


# return the weakest input provenance, or `assumed` if no input is known.
def weakest(provs: Iterable[str]) -> str:
    provs = [p for p in provs if p]
    if not provs:
        return "assumed"
    return max(provs, key=provenance_rank)


# append an error when `prov` is not a known provenance word (with an optional `(qualifier)`).
def _check_provenance_word(prov: Any, where: str, errs: list[str]) -> None:
    if not isinstance(prov, str):
        errs.append(f"{where}: provenance must be a string, got {prov!r}")
        return
    match = _PROVENANCE_RE.match(prov)
    if not match or match.group(1) not in PROVENANCE_WORDS:
        suggestions = difflib.get_close_matches(prov, sorted(PROVENANCE_WORDS), n=1)
        hint = f" (did you mean {suggestions[0]!r}?)" if suggestions else ""
        errs.append(f"{where}: provenance {prov!r} is not one of {sorted(PROVENANCE_WORDS)}{hint}")


# data classes

# one physical leaf after merging, alternative selection and overrides.
# `value` is in config units.
@dataclass
class Leaf:
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

    # true when a relation must compute this leaf (value null or provenance `derived`).
    @property
    def is_free(self) -> bool:
        return self.value is None or self.provenance == "derived"


# a model uncertainty shared by all rings in one simulated world.
# a world is one possible device model.
# a normal entry is drawn once per world.
# a label entry follows another hypothesis choice and needs no separate draw.
@dataclass
class UncertaintyEntry:
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

    # return the leader hypothesis name, or None for an independent entry.
    @property
    def follows(self) -> str | None:
        leader = self.fields.get("follows")
        return str(leader) if leader is not None else None

    # return the label implied by the leader hypothesis choice.
    def label_of(self, leader_choice: str) -> str:
        return str((self.fields.get("map") or {})[str(leader_choice)])


# one per-device variation entry, keyed by the config path it varies.
@dataclass
class VariationEntry:
    path: str
    fields: dict
    origin: list[str] = field(default_factory=list)


# a validated config with values in config units and metadata for each physical leaf.
class Config:

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

    # access
    # return whether the path is explicitly stored.
    # schema defaults do not count.
    def has(self, path: str) -> bool:
        return path in self.leaves or path in self.settings or path in self.open_values

    # true if a physical leaf is present with a value (not null, not `derived`).
    def is_given(self, path: str) -> bool:
        leaf = self.leaves.get(path)
        return leaf is not None and not leaf.is_free

    # return a stored value in config units, or its schema default.
    # lookup order is physical leaf, setting, open value, then schema default.
    # an open subtree can also be read as a mapping.
    # return `default` if absent.
    def get(self, path: str, default: Any = None) -> Any:
        if path in self.leaves:
            return self.leaves[path].value
        if path in self.settings:
            return self.settings[path]
        if path in self.open_values:
            return self.open_values[path]
        key = schema.lookup(path)
        if key is not None and key.default is not None:
            return key.default
        prefix = path + "."
        subtree = {k[len(prefix):]: v for k, v in self.open_values.items() if k.startswith(prefix)}
        if subtree:
            return subtree
        return default

    # the Leaf at `path` (KeyError if absent).
    def leaf(self, path: str) -> Leaf:
        return self.leaves[path]

    # return an SI conversion of the stored value.
    # leave the config unchanged.
    def si(self, path: str, default: Any = None):
        value = self.get(path, None)
        if value is None:
            return default
        return to_si(path, value)

    # provenance word of a leaf.
    def provenance(self, path: str) -> str:
        return self.leaves[path].provenance

    # number of rings N (array.n_rings).
    @property
    def n_rings(self) -> int:
        return int(self.get("array.n_rings"))

    # number of laser lines L (channel_plan.lines table or channel_plan.n_lines).
    @property
    def n_lines(self) -> int:
        lines = self.get("channel_plan.lines")
        if lines:
            return len(lines)
        return int(self.get("channel_plan.n_lines"))

    # return an alternative with its resolved value and provenance metadata.
    # resolve artefacts into a copy.
    # keep the stored reference unchanged so that reading an alternative does not change the config hash.
    def alt_value(self, path: str, name: str) -> dict:
        leaf = self.leaves[path]
        if name not in leaf.alternatives:
            raise ConfigError(f"{path}: no alternative {name!r}; known: {sorted(leaf.alternatives)}")
        alt = leaf.alternatives[name]
        if "from" in alt or (isinstance(alt.get("value"), dict) and "from" in alt["value"]):
            spec = alt.get("from", alt.get("value", {}).get("from") if isinstance(alt.get("value"), dict) else None)
            at = alt.get("at")
            if isinstance(alt.get("value"), dict):
                at = alt["value"].get("at", at)
            value, info = resolve_from(spec, at, self.data_root)      # cached by file and mtime
            # keep the stored artefact reference.
            # resolving this copy must not change the config hash.
            alt = dict(alt, value=value, provenance=alt.get("provenance") or info["provenance"], artefact=info)
        return {"value": alt.get("value"), "provenance": alt.get("provenance", leaf.provenance),
                "generation": alt.get("generation", leaf.generation), "source": alt.get("source", leaf.source),
                "artefact": alt.get("artefact") or alt.get("_resolved_artefact")}

    # export
    # return a nested, serializable tree of values and provenance.
    # include alternatives and uncertainty because they can change a simulated world.
    # this tree defines the config hash.
    # omit file-loading history.
    def resolved_tree(self) -> dict:
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
                # alternatives reach the plant through hypothesis draws, so they belong to the resolved config and its hash.
                # configs that produce different worlds must have different hashes.
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

    # yAML text of `resolved_tree` (written as experiment.resolved.yaml in a run directory).
    def dump_yaml(self) -> str:
        return yaml.safe_dump(self.resolved_tree(), sort_keys=True, default_flow_style=None, width=120)

    # sha256 hex digest of the canonical resolved tree (for run manifests).
    def hash(self) -> str:
        blob = json.dumps(self.resolved_tree(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    # alias of `hash` (used by mrm.control.api.ControllerSpec.fingerprint).
    def fingerprint(self) -> str:
        return self.hash()

    # alias of `resolved_tree`.
    def to_dict(self) -> dict:
        return self.resolved_tree()

    # return a shallow view with the requested path prefixes removed.
    # paths in `keep` remain.
    # leaf objects and the uncertainty, variation, geometry, and relation tables are shared.
    # treat them as read-only.
    # controllers use this view to exclude experiment settings, including random seeds.
    def without(self, prefixes, keep=()) -> "Config":
        prefixes = tuple(prefixes)
        keep = frozenset(keep)

        def kept(path: str) -> bool:
            return path in keep or not path.startswith(prefixes)
        view = copy.copy(self)
        view.leaves = {p: v for p, v in self.leaves.items() if kept(p)}
        view.settings = {p: v for p, v in self.settings.items() if kept(p)}
        view.open_values = {p: v for p, v in self.open_values.items() if kept(p)}
        view.setting_leaves = {p: v for p, v in self.setting_leaves.items() if kept(p)}
        view.overrides_applied = {p: v for p, v in self.overrides_applied.items() if kept(p)}
        return view

    def __repr__(self) -> str:
        return f"Config({self.name!r}, {len(self.leaves)} leaves, {len(self.uncertainty)} uncertainty entries)"


# make a value YAML/JSON-friendly (numpy -> python, tuples -> lists).
def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        # YAML weights can have numeric keys.
        # use strings so the config hash can sort all keys.
        return {str(k): _plain(x) for k, x in v.items()}
    return v


# artefacts

# cache key — absolute path and file/sidecar modification times cache value — data, sidecar metadata, and SHA-256
_ARTEFACT_CACHE: dict[tuple, tuple] = {}


# sha256 hex digest of a file.
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# sidecar file names accepted for an artefact, in the order they are tried.
def sidecar_candidates(path: Path) -> list[Path]:
    out = [path.with_name(path.name + ".sidecar.json"), path.with_name(path.stem + ".sidecar.json"),
           path.with_name(path.name + ".meta.json"), path.with_name(path.stem + ".meta.json")]
    if path.suffix != ".json":
        out.append(path.with_suffix(".json"))      # thermal artefacts store <stem>.json beside the .npz
    return out


# verify an artefact and return its value plus file and provenance metadata.
# `spec` contains a file path and a dotted key, separated by a colon.
# the JSON sidecar must match the file SHA-256.
# cached data is reused while the file and sidecar modification times stay the same.
# return a copy of the value.
# `at` selects one coordinate of a table by linear interpolation.
# coordinates and values keep the table units.
# values outside the axis range are rejected.
def resolve_from(spec: str, at: Mapping | None, data_root: Path) -> tuple[Any, dict]:
    if not isinstance(spec, str) or ":" not in spec:
        raise ArtefactError(f"from: expects 'extracted/<file>:<dotted.key>', got {spec!r}")
    relative_path, key = spec.rsplit(":", 1)
    path = (Path(data_root) / relative_path).resolve()
    if not path.exists():
        raise ArtefactError(f"artefact {relative_path!r} not found under {data_root}")
    sidecar = next((p for p in sidecar_candidates(path) if p.exists()), None)
    if sidecar is None:
        raise ArtefactError(f"artefact {relative_path!r} has no sidecar (tried {[p.name for p in sidecar_candidates(path)]})")
    mtime = (path.stat().st_mtime_ns, sidecar.stat().st_mtime_ns)
    cache_key = (str(path), mtime)
    if cache_key in _ARTEFACT_CACHE:
        data, metadata, digest = _ARTEFACT_CACHE[cache_key]
    else:
        metadata = json.loads(sidecar.read_text())
        digest = _sha256(path)
        if metadata.get("sha256") != digest:
            raise ArtefactError(f"artefact {relative_path!r}: sha256 {digest[:12]}... does not match its sidecar "
                                f"{str(metadata.get('sha256'))[:12]}... ({sidecar.name})")
        if path.suffix == ".json":
            data = json.loads(path.read_text())
        elif path.suffix in (".yaml", ".yml"):
            data = yaml_load(path.read_text())
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files}
        else:
            raise ArtefactError(f"artefact {relative_path!r}: unsupported format {path.suffix}")
        _ARTEFACT_CACHE[cache_key] = (data, metadata, digest)
    node: Any = data
    for part in key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, (list, tuple)) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise ArtefactError(f"artefact {relative_path!r}: key {key!r} not found (failed at {part!r})")
    if at:
        if not isinstance(node, dict) or "value" not in node or len(at) != 1:
            raise ArtefactError(f"{spec}: 'at' needs a table {{x: [...], value: [...]}} and one x")
        (axis_name, x), = at.items()
        if axis_name not in node:
            raise ArtefactError(f"{spec}: table has no axis {axis_name!r}")
        axis_values = np.asarray(node[axis_name], float)
        table_values = np.asarray(node["value"], float)
        if x < axis_values.min() or x > axis_values.max():
            raise ArtefactError(f"{spec}: {axis_name}={x} is outside the table range [{axis_values.min()}, {axis_values.max()}]")
        node = float(np.interp(x, axis_values, table_values))
    if isinstance(node, np.ndarray):
        node = node.tolist()
    qualifiers = []
    for input_name, input_record in (metadata.get("inputs") or {}).items():
        if isinstance(input_record, dict) and isinstance(input_record.get("provenance"), str):
            qualifiers.append(f"{input_record['provenance']}-{input_name}")
    provenance = f"solver({', '.join(qualifiers)})" if qualifiers else "solver"
    info = {"file": relative_path, "key": key, "sha256": digest, "sidecar": sidecar.name, "provenance": provenance,
            "schema": metadata.get("schema"), "at": dict(at) if at else None}
    return copy.deepcopy(node), info


# file reading

# path of a preset name or a YAML file.
# a name `x` is tried as `<base>/x.yaml`, `presets/x.yaml`, then `presets/private/x.yaml`.
# a relative `.yaml` path is tried beside `base` first (the directory of the file that extends it), then as given (relative to the working directory), then as `presets/<file name>`.
# an absolute path is tried as given, then as `presets/<file name>`.
def _resolve_file(name: str | Path, base: Path | None = None) -> Path:
    path = Path(name)
    candidates = []
    if path.suffix in (".yaml", ".yml"):
        candidates += ([base / path] if base and not path.is_absolute() else []) + [path] + [PRESET_DIR / path.name]
    else:
        candidates += [PRESET_DIR / f"{name}.yaml", PRESET_DIR / "private" / f"{name}.yaml"]
        if base:
            candidates.insert(0, base / f"{name}.yaml")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise ConfigError(f"preset or file {str(name)!r} not found (looked in {[str(candidate) for candidate in candidates]})")


# the top-level mapping of a YAML file (an empty file gives {}).
def _read_yaml(path: Path) -> dict:
    with open(path) as stream:
        data = yaml_load(stream.read()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


# source keys of `mrm/params/sources.yaml` (a mapping key -> citation, a list of keys, or a list of {key — ...}).
# returns None when the file does not exist yet.
def load_sources(path: Path | None = None) -> set[str] | None:
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


# true for a provenance leaf — a mapping with `value` or `from`.
def _is_leaf_mapping(v: Any) -> bool:
    return isinstance(v, dict) and ("value" in v or "from" in v)


# a flattened value during merging — a leaf mapping (`leaf`) or a bare value (`bare`).
@dataclass
class _FlatNode:
    leaf: dict | None = None
    bare: Any = None
    is_bare: bool = False
    origin: list[str] = field(default_factory=list)


# one file (or a merged chain of files) flattened to dotted paths, plus its special sections.
@dataclass
class _FlatTree:
    nodes: dict[str, _FlatNode] = field(default_factory=dict)
    uncertainty: dict[str, dict] = field(default_factory=dict)
    uncertainty_origin: dict[str, list[str]] = field(default_factory=dict)
    variation: dict[str, dict] = field(default_factory=dict)
    variation_origin: dict[str, list[str]] = field(default_factory=dict)
    geometry: dict | None = None
    alts: dict[str, str] = field(default_factory=dict)
    relations: dict[str, dict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    files: list[str] = field(default_factory=list)


# collect the uncertainty entries (mappings with `kind`) of a nested tree by dotted name.
def _flatten_uncertainty(tree: dict, prefix: str, out: dict, errs: list[str], where: str) -> None:
    for k, v in tree.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict) and "kind" in v:
            out[name] = copy.deepcopy(v)
        elif isinstance(v, dict):
            _flatten_uncertainty(v, name + ".", out, errs, where)
        else:
            errs.append(f"{where}: uncertainty.{name} must be an entry mapping with 'kind', got {v!r}")


# flatten one file's tree into path -> node (leaf mapping or bare value).
def _flatten_file(data: dict, fname: str, errs: list[str]) -> _FlatTree:
    flat = _FlatTree(files=[fname])
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
                    flat.uncertainty_origin.setdefault(n, []).append(fname)
            continue
        if top == "variation":
            for p, e in (v or {}).items():
                if p == "geometry":
                    flat.geometry = copy.deepcopy(e)
                elif isinstance(e, dict):
                    flat.variation[str(p)] = copy.deepcopy(e)
                    flat.variation_origin.setdefault(str(p), []).append(fname)
                else:
                    errs.append(f"{fname}: variation.{p} must be an entry mapping keyed by config path, e.g. "
                                f"variation.ring.optics.resonance_offset_nm: {{die_sigma: 6.0, ring_sigma: 2.0, "
                                f"provenance: ...}} (the old *_sigma_* keys are replaced by such entries)")
            continue
        if top == "meta":
            flat.meta = copy.deepcopy(v or {})
            continue
        if top not in schema.TOP_SECTIONS:
            sugg = difflib.get_close_matches(top, list(schema.TOP_SECTIONS) + list(schema.SPECIAL_TOP), n=1)
            hint = f" (did you mean {sugg[0]!r}?)" if sugg else ""
            msg = schema.DELETED.get(top)
            errs.append(f"{fname}: unknown top-level section {top!r}{hint}" + (f": {msg}" if msg else ""))
            continue
        _flatten_tree(v, top, flat, fname, errs)
    return flat


# flatten one section — stop at a provenance leaf, a table, an open subtree or a bare value.
def _flatten_tree(v: Any, path: str, flat: _FlatTree, fname: str, errs: list[str]) -> None:
    key = schema.lookup(path)
    if schema.is_open(path) or (key is not None and key.type == "map" and not key.physical):
        flat.nodes[path] = _FlatNode(bare=copy.deepcopy(v), is_bare=True, origin=[fname])
        return
    if _is_leaf_mapping(v):
        bad = set(v) - LEAF_FIELDS
        if bad:
            for b in sorted(bad):
                s = difflib.get_close_matches(b, sorted(LEAF_FIELDS), n=1)
                errs.append(f"{fname}: {path}: unknown leaf field {b!r}" + (f" (did you mean {s[0]!r}?)" if s else ""))
        flat.nodes[path] = _FlatNode(leaf=copy.deepcopy(v), origin=[fname])
        return
    if isinstance(v, dict) and not (key is not None and key.type == "table"):
        if not v:
            return
        for k, sub in v.items():
            _flatten_tree(sub, f"{path}.{k}", flat, fname, errs)
        return
    flat.nodes[path] = _FlatNode(bare=copy.deepcopy(v), is_bare=True, origin=[fname])


# merge the child over the parent and retain the source history.
# a changed physical value must restate provenance and generation.
# remove the old value reference before merging its replacement.
# merge alternatives by name.
# a null child alternative removes that inherited alternative.
def _merge(parent: _FlatTree, child: _FlatTree, child_kind: str, errs: list[str]) -> _FlatTree:
    out = _FlatTree(nodes=dict(parent.nodes), uncertainty=copy.deepcopy(parent.uncertainty),
                    uncertainty_origin={k: list(v) for k, v in parent.uncertainty_origin.items()},
                    variation=copy.deepcopy(parent.variation),
                    variation_origin={k: list(v) for k, v in parent.variation_origin.items()},
                    geometry=copy.deepcopy(parent.geometry), alts=dict(parent.alts),
                    relations=copy.deepcopy(parent.relations), meta=copy.deepcopy(parent.meta),
                    files=parent.files + child.files)
    filename = child.files[0]
    for path, node in child.nodes.items():
        previous = out.nodes.get(path)
        if previous is None or previous.is_bare or node.is_bare:
            if previous is not None and not previous.is_bare and node.is_bare and child_kind in _STRICT_KINDS:
                key = schema.lookup(path)
                if key is not None and key.physical:
                    errs.append(f"{filename}: {path}: bare value replaces a provenance leaf; restate it as "
                                f"{{value, provenance, generation, source}}")
            new = _FlatNode(leaf=copy.deepcopy(node.leaf), bare=copy.deepcopy(node.bare), is_bare=node.is_bare,
                            origin=(previous.origin if previous else []) + node.origin)
            out.nodes[path] = new
            continue
        merged = dict(previous.leaf)
        changed = False
        for value_field in ("value", "from", "at"):
            if value_field in node.leaf and node.leaf[value_field] != previous.leaf.get(value_field):
                changed = True
        if changed:
            missing = [f for f in ("provenance", "generation") if f not in node.leaf]
            if missing:
                errs.append(f"{filename}: {path}: changes the value of a parent leaf without restating "
                            f"{' and '.join(missing)}")
            for value_field in ("value", "from", "at"):
                merged.pop(value_field, None)
        if "alternatives" in node.leaf and "alternatives" in previous.leaf:
            alternatives = dict(previous.leaf["alternatives"] or {})
            alternatives.update(node.leaf["alternatives"] or {})
            # a null child alternative removes an inherited choice whose meaning no longer applies.
            for alternative_name in [alternative_name for alternative_name,
                a in (node.leaf["alternatives"] or {}).items() if a is None]:
                if alternative_name not in (previous.leaf["alternatives"] or {}):
                    errs.append(f"{filename}: {path}: alternative {alternative_name!r} is null, but the parent has no such alternative")
                alternatives.pop(alternative_name, None)
            merged.update(node.leaf)
            merged["alternatives"] = alternatives
        else:
            merged.update(node.leaf)
        out.nodes[path] = _FlatNode(leaf=merged,
                                    origin=previous.origin + [f"{filename}" + (" (value changed)" if changed else "")])
    for n, e in child.uncertainty.items():
        if n in out.uncertainty and e.get("kind", out.uncertainty[n].get("kind")) == out.uncertainty[n].get("kind"):
            out.uncertainty[n].update(e)
        else:
            out.uncertainty[n] = e
        out.uncertainty_origin.setdefault(n, []).append(filename)
    for p, e in child.variation.items():
        out.variation.setdefault(p, {}).update(e)
        out.variation_origin.setdefault(p, []).append(filename)
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
    sources = dict(out.meta.get("sources") or {})
    sources.update(child.meta.get("sources") or {})
    out.meta.update(child.meta)
    if sources:
        out.meta["sources"] = sources
    return out


# load parents recursively, then merge the child.
# return the tree and file kind.
# merge multiple parents from left to right.
# `stack` detects inheritance cycles.
# validate each file's bare physical values before inheritance can hide them.
def _load_chain(path: Path, stack: tuple[Path, ...], errs: list[str]) -> tuple[_FlatTree, str]:
    if path in stack:
        raise ConfigError(f"extends cycle: {' -> '.join(str(p.name) for p in stack + (path,))}")
    data = _read_yaml(path)
    kind = (data.get("meta") or {}).get("kind", "preset")
    if kind not in schema.SCHEMA["meta.kind"].enum:
        errs.append(f"{path.name}: meta.kind {kind!r} is not one of {schema.SCHEMA['meta.kind'].enum}")
    if kind == "corner":
        extra = set(data) - {"extends", "meta", "alts"}
        if extra:
            errs.append(f"{path.name}: a corner preset may contain only extends, meta and alts; "
                        f"found {sorted(extra)}")
    try:
        source_path = str(path.relative_to(PACKAGE_DIR.parent.parent))
    except ValueError:
        source_path = str(path)
    flat = _flatten_file(data, source_path, errs)
    for p, node in flat.nodes.items():
        if node.is_bare and kind in _STRICT_KINDS:
            key = schema.lookup(p)
            if key is not None and key.physical and node.bare is not None:
                errs.append(f"{source_path}: {p}: bare value in a preset; write {{value, provenance, generation, source}} "
                            f"(bare numbers are allowed only in overrides)")
    extends = data.get("extends")
    parents = [extends] if isinstance(extends, str) else list(extends or [])
    merged: _FlatTree | None = None
    for parent_name in parents:
        parent_path = _resolve_file(parent_name, path.parent)
        parent_tree, _ = _load_chain(parent_path, stack + (path,), errs)
        merged = parent_tree if merged is None else _merge(merged, parent_tree, "preset", errs)
    if merged is None:
        return flat, kind
    return _merge(merged, flat, kind, errs), kind


# overrides

# normalize overrides — a dict {path — value} or strings "path=value" (value parsed as YAML).
def parse_overrides(overrides: Mapping[str, Any] | Iterable[str] | None) -> dict[str, Any]:
    if overrides is None:
        return {}
    if isinstance(overrides, Mapping):
        return dict(overrides)
    out = {}
    for s in overrides:
        if not isinstance(s, str) or "=" not in s:
            raise ConfigError(f"override {s!r} must be 'path=value'")
        k, v = s.split("=", 1)
        out[k.strip()] = _bool_as_enum_word(k.strip(), yaml_load(v))
    return out


# yAML 1.1 reads off / on / no / yes as booleans.
# for an enum setting whose words do not include the boolean but include exactly one word of that meaning, the word is meant — `--set experiment.variation=off` is 'off', not False.
# Other values are returned unchanged.
def _bool_as_enum_word(path: str, v: Any) -> Any:
    if not isinstance(v, bool):
        return v
    key = schema.lookup(path)
    if key is None or key.type != "enum" or v in key.enum:
        return v
    words = [w for w in key.enum if isinstance(w, str) and w.lower() in (("on", "yes", "true") if v else
                                                                         ("off", "no", "false"))]
    return words[0] if len(words) == 1 else v


# normalize alternatives — a dict {path — name} or strings "path=name".
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


# load

# load a config, apply selections and overrides, then validate it.
# `preset` is a preset name, YAML path, or parsed mapping.
# mappings cannot use inheritance.
# `alts` selects a leaf alternative or pins an uncertainty choice.
# `overrides` sets values last and holds them in every world.
# both arguments accept mappings or lists of `path=value` strings.
# `sources` supplies allowed citation keys.
# its default is the source registry plus keys in the file metadata.
# `data_root` locates extracted artefacts and defaults to `mrm/data`.
# physical values remain in config units.
# collect validation errors in `ConfigError`.
# use `ArtefactError` when all reported errors concern artefact resolution.
def load(preset: str | Path | Mapping, overrides: Mapping[str, Any] | Iterable[str] | None = None,
         alts: Mapping[str, str] | Iterable[str] | None = None, *, sources: Iterable[str] | Path | None = None,
         data_root: str | Path | None = None) -> Config:
    errors: list[str] = []
    cfg = Config()
    cfg.data_root = Path(data_root) if data_root else DATA_ROOT
    if isinstance(preset, Mapping):
        data = copy.deepcopy(dict(preset))
        kind = (data.get("meta") or {}).get("kind", "preset")
        if data.get("extends"):
            raise ConfigError("a mapping preset cannot use extends:; write it to a file")
        flat = _flatten_file(data, "<mapping>", errors)
        for p, node in flat.nodes.items():
            key = schema.lookup(p)
            if node.is_bare and kind in _STRICT_KINDS and key is not None and key.physical and node.bare is not None:
                errors.append(f"<mapping>: {p}: bare value in a preset")
        cfg.name = str((data.get("meta") or {}).get("name", "<mapping>"))
    else:
        path = _resolve_file(preset)
        flat, kind = _load_chain(path, (), errors)
        cfg.name = str(flat.meta.get("name", path.stem))
    cfg.kind = kind
    cfg.files = flat.files
    cfg.meta = flat.meta
    cfg.relations_cfg = flat.relations

    # sources
    if sources is None:
        source_keys = load_sources()
    elif isinstance(sources, (str, Path)):
        source_keys = load_sources(Path(sources))
    else:
        source_keys = set(sources)
    extra_source_keys = set((flat.meta.get("sources") or {}).keys())
    if source_keys is None and kind == "test" and extra_source_keys:
        cfg.sources = extra_source_keys                        # a self-contained test preset checks against its own keys
    elif source_keys is None:
        cfg.sources = None
        cfg.warnings.append("mrm/params/sources.yaml does not exist yet: source keys are not checked")
    else:
        cfg.sources = source_keys | extra_source_keys

    # leaves and settings
    for path, node in flat.nodes.items():
        key = schema.lookup(path)
        if schema.is_open(path):
            cfg.open_values[path] = node.bare
            continue
        if not node.is_bare and key is not None and not key.physical:
            setting_leaf = _make_leaf(path, node.leaf, node.origin, cfg, errors)
            cfg.settings[path] = setting_leaf.value
            cfg.setting_leaves[path] = setting_leaf
            continue
        if node.is_bare:
            if key is not None and key.physical and node.bare is not None:
                cfg.leaves[path] = Leaf(path=path, value=node.bare, provenance="override",
                                        origin=node.origin + ["bare value in an experiment file (tagged override)"])
            else:
                cfg.settings[path] = node.bare
            continue
        cfg.leaves[path] = _make_leaf(path, node.leaf, node.origin, cfg, errors)

    cfg.uncertainty = {n: UncertaintyEntry(n, e, flat.uncertainty_origin.get(n, []))
                       for n, e in flat.uncertainty.items()}
    cfg.variation = {p: VariationEntry(p, e, flat.variation_origin.get(p, [])) for p, e in flat.variation.items()}
    cfg.geometry = flat.geometry
    # bare physical values in experiments are overrides.
    # hold them in every world.
    for path, leaf in cfg.leaves.items():
        if leaf.provenance == "override" and leaf.value is not None:
            _pin_for_leaf_override(cfg, path)

    # select alternatives from files, then from the argument
    for p, name in list(flat.alts.items()) + list(_parse_alts(alts).items()):
        _apply_alt(cfg, p, name, errors)
    # overrides
    for p, v in parse_overrides(overrides).items():
        _apply_override(cfg, p, v, errors)
    # a pinned label entry pins (or restricts) the entry it follows
    _resolve_label_entries(cfg, errors)
    _apply_enum_aliases(cfg)

    try:
        _validate(cfg, errors)
    except Exception as e:  # noqa: BLE001 - a broken config may break validation - report what was found
        if not errors:
            raise
        errors.append(f"(validation stopped early: {type(e).__name__}: {e})")
    if errors:
        if any("artefact" in m for m in errors) and all(("artefact" in m) or ("from:" in m) for m in errors):
            raise ArtefactError(errors)
        raise ConfigError(errors)
    for w in cfg.warnings:
        warnings.warn(w, stacklevel=2)
    return cfg


# replace a renamed enum word (schema.ENUM_ALIASES) with its new word, in the settings and their leaves.
def _apply_enum_aliases(cfg: Config) -> None:
    for path, word_map in schema.ENUM_ALIASES.items():
        v = cfg.settings.get(path)
        if isinstance(v, str) and v in word_map:
            cfg.settings[path] = word_map[v]
            leaf = cfg.setting_leaves.get(path)
            if leaf is not None:
                leaf.value = word_map[v]


# build a physical leaf, resolve its artefact, and collect its alternatives.
# add a default alternative for the original value when other alternatives exist.
# a missing artefact file can use the declared fallback.
# an invalid checksum, sidecar, or dotted key remains an error.
def _make_leaf(path: str, d: dict, origin: list[str], cfg: Config, errs: list[str]) -> Leaf:
    d = dict(d)
    artefact = None
    value = d.get("value")
    from_spec = d.get("from")
    at = d.get("at")
    fell_back = False
    if isinstance(value, dict) and "from" in value:
        from_spec = value["from"]
        at = value.get("at", at)
    if from_spec is not None:
        try:
            value, artefact = resolve_from(from_spec, at, cfg.data_root)
        except ArtefactError as e:
            # use a fallback only while the artefact file is missing.
            # report invalid existing files.
            fallback = (d.get("alternatives") or {}).get(ARTEFACT_FALLBACK_ALT)
            if "not found under" in str(e) and isinstance(fallback, dict) and "value" in fallback:
                value = fallback["value"]
                for f in ("provenance", "generation", "source"):
                    if fallback.get(f) is not None:
                        d[f] = fallback[f]
                fell_back = True
                cfg.warnings.append(f"{path}: {e}; using its '{ARTEFACT_FALLBACK_ALT}' alternative {value!r} "
                                    f"({d.get('provenance')}) until the artefact exists")
            else:
                errs.append(f"{path}: {e}")
                value = None
        if not fell_back:
            if "provenance" not in d:
                d["provenance"] = artefact["provenance"] if artefact is not None else "solver"
            elif artefact is not None and not str(d.get("provenance", "")).startswith("solver"):
                errs.append(f"{path}: a value taken from a solver artefact must have provenance 'solver' "
                            f"(got {d.get('provenance')!r})")
    provenance = d.get("provenance")
    if provenance is None:
        errs.append(f"{path}: leaf has no provenance (a number without a provenance word is a bug)")
        provenance = "assumed"
    alternatives = {}
    for name, alternative in (d.get("alternatives") or {}).items():
        if alternative is None:
            errs.append(f"{path}: alternative {name!r} is null; null only removes an inherited alternative in a child "
                        f"preset")
            continue
        if isinstance(alternative, dict) and ("value" in alternative or "from" in alternative):
            bad = set(alternative) - _ALTERNATIVE_FIELDS
            if bad:
                errs.append(f"{path}: alternative {name!r} has unknown fields {sorted(bad)}")
            alternatives[str(name)] = dict(alternative)
        else:
            alternatives[str(name)] = {"value": alternative, "_bare": True}
    if alternatives and "default" not in alternatives:
        alternatives["default"] = {"value": value, "provenance": provenance, "generation": d.get("generation"),
                           "source": d.get("source"), "_resolved_artefact": artefact}
    leaf_range = d.get("range")
    leaf = Leaf(path=path, value=value, provenance=provenance, generation=d.get("generation"), source=d.get("source"),
                range=tuple(leaf_range) if isinstance(leaf_range, (list, tuple)) else leaf_range,
                    alternatives=alternatives,
                note=d.get("note"), textbook=d.get("textbook"), disagrees=d.get("disagrees"), targets=d.get("targets"),
                residual=d.get("residual"), identifiable=d.get("identifiable"), origin=list(origin), artefact=artefact)
    if fell_back:
        leaf.selected_alt = ARTEFACT_FALLBACK_ALT
        leaf.origin.append(f"artefact {from_spec!r} missing: alternative {ARTEFACT_FALLBACK_ALT!r}")
    return leaf


# return each hypothesis choice as a mapping of paths to selections.
# normalize the short target-plus-list form into the same structure as the explicit choice mappings.
# selections can be alternative names or values.
def _hypothesis_choice_map(entry: UncertaintyEntry) -> dict[str, dict[str, Any]]:
    choices = entry.fields.get("choices")
    target = entry.fields.get("target")
    if isinstance(choices, list):
        if not target:
            return {}
        return {str(choice): {target: str(choice)} for choice in choices}
    if isinstance(choices, dict):
        out = {}
        for choice, selections in choices.items():
            if isinstance(selections, dict):
                out[str(choice)] = dict(selections)
            elif target:
                out[str(choice)] = {target: selections}
            else:
                out[str(choice)] = {}
        return out
    return {}


# select a leaf alternative or pin an uncertainty choice.
# keep any hypothesis that controls the leaf consistent with this selection.
# pin an absolute numeric uncertainty so that a draw cannot replace the value.
def _apply_alt(cfg: Config, path: str, name: str, errs: list[str]) -> None:
    if path.startswith("uncertainty."):
        entry_name = path[len("uncertainty."):]
        entry = cfg.uncertainty.get(entry_name)
        if entry is None:
            errs.append(f"alts: no uncertainty entry {entry_name!r}; known: {sorted(cfg.uncertainty)}")
            return
        if entry.kind == "hypothesis":
            choices = _hypothesis_choice_map(entry)
            if name not in choices:
                errs.append(f"alts: uncertainty.{entry_name} has no choice {name!r}; known: {sorted(choices)}")
                return
            entry.pinned = name
            for p, a in choices[name].items():
                if p in cfg.leaves and isinstance(a, str):
                    _select_leaf_alt(cfg, p, a, f"pinned by uncertainty.{entry_name}={name}", errs)
                elif p in cfg.leaves and isinstance(a, dict):
                    leaf = cfg.leaves[p]
                    leaf.value = a.get("value")
                    leaf.provenance = a.get("provenance", leaf.provenance)
                    leaf.origin.append(f"set by uncertainty.{entry_name}={name}")
                elif schema.lookup(p) is not None and not schema.lookup(p).physical:
                    cfg.settings[p] = a
        else:
            corners = dict(entry.fields.get("corners") or {})
            if name in ("low", "high"):
                if name not in entry.fields:
                    errs.append(f"alts: uncertainty.{entry_name} has no {name!r}")
                    return
                entry.pinned = entry.fields[name]
            elif name == "nominal":
                entry.pinned = "nominal"
            elif name in corners:
                c = corners[name]
                entry.pinned = c["value"] if isinstance(c, dict) else c
            else:
                errs.append(f"alts: uncertainty.{entry_name} has no corner {name!r}; known: low, high, nominal, "
                            f"{sorted(corners)}")
                return
        cfg.alts_applied[path] = name
        return
    if path not in cfg.leaves:
        suggestions = difflib.get_close_matches(path, list(cfg.leaves), n=1)
        errs.append(f"alts: {path!r} is not a leaf of this config" + (f" (did you mean {suggestions[0]!r}?)" if suggestions else ""))
        return
    _select_leaf_alt(cfg, path, name, "alt", errs)
    # keep hypothesis entries that set this leaf consistent with the selection
    for entry in cfg.uncertainty.values():
        if entry.kind != "hypothesis":
            if entry.target == path and entry.fields.get("mode", "absolute") == "absolute":
                # pin this entry so a later draw cannot replace the selected value.
                entry.pinned = "override"
            continue
        choice_map = _hypothesis_choice_map(entry)
        touching = {c: m[path] for c, m in choice_map.items() if path in m}
        if not touching:
            continue
        matching_choices = [c for c, a in touching.items() if a == name]
        if not matching_choices:
            errs.append(f"alts: {path}={name} is not reachable from uncertainty.{entry.name} (choices "
                        f"{touching}); select the hypothesis with uncertainty.{entry.name}=<choice> instead")
        elif len(matching_choices) == 1:
            entry.pinned = matching_choices[0]
        else:
            entry.restrict = matching_choices
    cfg.alts_applied[path] = name


# check a hypothesis label that follows a leader choice.
# the label changes no parameters and has no draw or weights of its own.
# its map must cover the leader choices and agree with the nominal label.
def _validate_label_entry(cfg: Config, entry: UncertaintyEntry, choice_map: dict, where: str, errs: list[str]) -> None:
    leader_name, label_map = entry.fields.get("follows"), entry.fields.get("map")
    leader = cfg.uncertainty.get(str(leader_name)) if leader_name is not None else None
    if leader is None or leader.kind != "hypothesis":
        errs.append(f"{where}: follows {leader_name!r} must name another hypothesis entry")
        return
    if leader.name == entry.name or leader.follows is not None:
        errs.append(f"{where}: follows uncertainty.{leader.name}, which is itself a label entry (or this entry)")
        return
    if not isinstance(label_map, dict):
        errs.append(f"{where}: a label entry needs map: {{<choice of uncertainty.{leader.name}>: <choice>}}")
        return
    leader_choices = list(_hypothesis_choice_map(leader))
    label_map = {str(k): str(v) for k, v in label_map.items()}
    if set(label_map) != set(leader_choices):
        errs.append(f"{where}: map must name every choice of uncertainty.{leader.name} exactly once "
                    f"({leader_choices}), got {sorted(label_map)}")
    unknown_labels = set(label_map.values()) - set(choice_map)
    if unknown_labels:
        errs.append(f"{where}: map gives labels {sorted(unknown_labels)} that are not choices {sorted(choice_map)}")
    unused = set(choice_map) - set(label_map.values())
    if unused:
        errs.append(f"{where}: choices {sorted(unused)} are never implied by uncertainty.{leader.name} (map)")
    if any(choice_map[c] for c in choice_map):
        errs.append(f"{where}: the choices of a label entry set nothing (its leader's choices do): "
                    f"{ {c: v for c, v in choice_map.items() if v} }")
    if entry.fields.get("weights") is not None or entry.target is not None:
        errs.append(f"{where}: a label entry is not drawn: no weights or target")
    leader_nominal = leader.fields.get("nominal")
    if (leader_nominal is not None and str(leader_nominal) in label_map
            and str(entry.fields.get("nominal")) != label_map[str(leader_nominal)]):
        errs.append(f"{where}: nominal {entry.fields.get('nominal')!r} must be the label of "
                    f"uncertainty.{leader.name}'s nominal {leader_nominal!r}: {label_map[str(leader_nominal)]!r}")


# translate a pinned label into allowed choices of its leader hypothesis.
# one allowed choice pins the leader.
# several choices restrict its draw.
# a conflicting leader pin is an error.
# leave malformed entries to validation.
def _resolve_label_entries(cfg: Config, errs: list[str]) -> None:
    for entry in cfg.uncertainty.values():
        if entry.kind != "hypothesis" or entry.follows is None or entry.pinned is None:
            continue
        leader = cfg.uncertainty.get(entry.follows)
        label_map = entry.fields.get("map")
        if (leader is None or leader.kind != "hypothesis" or leader.follows is not None
                or not isinstance(label_map, dict)):
            continue
        allowed = [c for c in _hypothesis_choice_map(leader) if str(label_map.get(c)) == str(entry.pinned)]
        if leader.restrict is not None:
            allowed = [c for c in allowed if c in leader.restrict]
        if leader.pinned is not None:
            if str(leader.pinned) not in allowed:
                errs.append(f"alts: uncertainty.{entry.name}={entry.pinned} needs uncertainty.{leader.name} in "
                            f"{allowed}, but it is pinned to {leader.pinned!r} (uncertainty.{entry.name} follows "
                            f"uncertainty.{leader.name})")
            continue
        if not allowed:
            errs.append(f"alts: uncertainty.{entry.name}={entry.pinned}: no allowed choice of "
                        f"uncertainty.{leader.name} implies it (map {label_map})")
        elif len(allowed) == 1:
            _apply_alt(cfg, f"uncertainty.{leader.name}", allowed[0], errs)
            leader.origin.append(f"pinned by uncertainty.{entry.name}={entry.pinned} (a label entry)")
        else:
            leader.restrict = allowed
            leader.origin.append(f"restricted to {allowed} by uncertainty.{entry.name}={entry.pinned} (a label entry)")


# make alternative `name` the value of leaf `path`.
# `why` is recorded in the leaf's origin.
def _select_leaf_alt(cfg: Config, path: str, name: str, why: str, errs: list[str]) -> None:
    leaf = cfg.leaves[path]
    if name not in leaf.alternatives:
        errs.append(f"alts: {path} has no alternative {name!r}; known: {sorted(leaf.alternatives)}")
        return
    try:
        alternative = cfg.alt_value(path, name)
    except ArtefactError as e:
        errs.append(f"{path}: alternative {name!r}: {e}")
        return
    leaf.value = alternative["value"]
    leaf.provenance = alternative["provenance"]
    leaf.generation = alternative["generation"]
    leaf.source = alternative["source"]
    leaf.artefact = alternative["artefact"] if alternative["artefact"] else (leaf.artefact if name == "default" else None)
    leaf.selected_alt = name
    leaf.origin.append(f"{why}: alternative {name!r}")


# hold the user's physical override in every world.
# disable numeric draws for this leaf.
# pin a touching hypothesis to its nominal choice and record this path so that the choice cannot overwrite the override.
def _pin_for_leaf_override(cfg: Config, path: str) -> None:
    for entry in cfg.uncertainty.values():
        if entry.target == path and entry.kind in ("interval", "dist"):
            entry.pinned = "override"
        elif entry.kind == "hypothesis" and any(path in m for m in _hypothesis_choice_map(entry).values()):
            cfg.warnings.append(f"override of {path} conflicts with hypothesis uncertainty.{entry.name}; the "
                                f"hypothesis is pinned to its nominal choice and {path} keeps the override")
            entry.pinned = entry.fields.get("nominal")
            override_paths = sorted(set(entry.fields.get("_override_paths", [])) | {path})
            entry.fields = dict(entry.fields, _override_paths=override_paths)


# apply a final value to an entry field, open value, physical leaf, or setting.
# match entry names from longest to shortest because names can contain dots.
# tag physical values as overrides and hold them across world draws.
def _apply_override(cfg: Config, path: str, v: Any, errs: list[str]) -> None:
    if path.startswith("uncertainty."):
        rest = path[len("uncertainty."):]
        for entry_name in sorted(cfg.uncertainty, key=len, reverse=True):
            if rest.startswith(entry_name + "."):
                field_path = rest[len(entry_name) + 1:]
                cfg.uncertainty[entry_name].fields[field_path] = v
                cfg.uncertainty[entry_name].origin.append(f"override {field_path}={v!r}")
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
        for variation_path in sorted(cfg.variation, key=len, reverse=True):
            if rest.startswith(variation_path + "."):
                cfg.variation[variation_path].fields[rest[len(variation_path) + 1:]] = v
                cfg.variation[variation_path].origin.append(f"override {rest[len(variation_path) + 1:]}={v!r}")
                cfg.overrides_applied[path] = v
                return
        field_path = rest.rsplit(".", 1)
        if len(field_path) == 2 and field_path[1] in _VARIATION_FIELDS:
            cfg.variation[field_path[0]] = VariationEntry(field_path[0], {field_path[1]: v,
                "provenance": "override"}, ["override"])
            cfg.overrides_applied[path] = v
            return
        errs.append(f"override {path!r}: not a variation entry field")
        return
    key = schema.lookup(path)
    if schema.is_open(path):
        cfg.open_values[path] = v
        cfg.overrides_applied[path] = v
        return
    if key is None:
        # an override inside an open map whose parent is known (e.g. controller.params.kp)
        msg = schema.DELETED.get(path)
        if msg:
            errs.append(f"override {path!r}: {msg}")
        else:
            suggestions = schema.suggest(path)
            errs.append(f"override {path!r}: unknown key" + (f" (did you mean {', '.join(suggestions)}?)" if suggestions else ""))
        return
    if key.physical:
        leaf = cfg.leaves.get(path)
        value = v["value"] if isinstance(v, dict) and "value" in v else v
        if leaf is None:
            cfg.leaves[path] = Leaf(path=path, value=value, provenance="override", origin=["override"])
        else:
            leaf.base_provenance = leaf.base_provenance or leaf.provenance
            leaf.origin.append(f"override (was {leaf.value!r}, {leaf.provenance})")
            leaf.value = value
            leaf.provenance = "override"
            leaf.selected_alt = None
            leaf.artefact = None
        _pin_for_leaf_override(cfg, path)
    else:
        cfg.settings[path] = v
        if key.per_world:
            # keep choices that agree with the user's switch value.
            # if none agree, use the nominal hypothesis for its other leaves and preserve this switch override.
            for entry in cfg.uncertainty.values():
                if entry.kind != "hypothesis":
                    continue
                choice_map = _hypothesis_choice_map(entry)
                touching = {c: m[path] for c, m in choice_map.items() if path in m}
                if not touching:
                    continue
                matching_choices = [c for c, a in touching.items() if a == v]
                if len(matching_choices) == 1:
                    entry.pinned = matching_choices[0]
                elif matching_choices:
                    entry.restrict = matching_choices
                else:
                    cfg.warnings.append(f"override {path}={v!r} is not a choice of hypothesis "
                                        f"uncertainty.{entry.name}; the hypothesis is pinned to its nominal and "
                                        f"{path} keeps the override")
                    entry.pinned = entry.fields.get("nominal")
                    override_paths = sorted(set(entry.fields.get("_override_paths", [])) | {path})
                    entry.fields = dict(entry.fields, _override_paths=override_paths)
    cfg.overrides_applied[path] = v


# validation

# check a value against its schema type and append errors.
# `level_sizes` gives the ring and line counts for list-length checks.
# check table records field by field.
# accept None for later presence checks.
def _check_type(key: schema.Key, v: Any, where: str, level_sizes: dict[str, int], errs: list[str]) -> None:
    type_name = key.type

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

    if v is None:
        return
    if type_name == "float":
        if key.level in ("ring", "line") and isinstance(v, list):
            expected_length = level_sizes.get(key.level)
            if expected_length is not None and len(v) != expected_length:
                errs.append(f"{where}: {key.level}-level list has {len(v)} values, expected {expected_length}")
            if not all(num(x) for x in v):
                errs.append(f"{where}: expected numbers, got {v!r}")
        elif not num(v):
            errs.append(f"{where}: expected a number, got {v!r}")
    elif type_name == "int":
        values = v if (isinstance(v, list) and key.level in ("ring", "line")) else [v]
        for x in values:
            if not (isinstance(x, int) and not isinstance(x, bool)) and not (isinstance(x, float) and x.is_integer()):
                errs.append(f"{where}: expected an integer, got {x!r}")
    elif type_name == "bool":
        if not isinstance(v, bool):
            errs.append(f"{where}: expected true/false, got {v!r}")
    elif type_name == "str":
        if not isinstance(v, str):
            errs.append(f"{where}: expected a string, got {v!r}")
    elif type_name == "enum":
        if v not in key.enum:
            suggestions = difflib.get_close_matches(str(v), [str(e) for e in key.enum], n=1)
            errs.append(f"{where}: {v!r} is not one of {list(key.enum)}"
                        + (f" (did you mean {suggestions[0]!r}?)" if suggestions else ""))
    elif type_name.startswith("list_"):
        inner = {"list_float": num, "list_int": lambda x: isinstance(x, int) and not isinstance(x, bool),
                 "list_str": lambda x: isinstance(x, str)}[type_name]
        if not isinstance(v, list) or not all(inner(x) for x in v):
            errs.append(f"{where}: expected a list of {type_name[5:]}, got {v!r}")
    elif type_name == "map":
        if not isinstance(v, dict):
            errs.append(f"{where}: expected a mapping, got {v!r}")
    elif type_name == "table":
        records = None
        if key.table_kind == "list":
            if not isinstance(v, list):
                errs.append(f"{where}: expected a list of records")
                return
            records = [(f"[{i}]", record) for i, record in enumerate(v)]
        elif key.table_kind == "map":
            if not isinstance(v, dict):
                errs.append(f"{where}: expected a mapping of records")
                return
            if not key.fields:
                return
            records = [(f".{k}", record) for k, record in v.items()]
        for tag, record in records or []:
            if not isinstance(record, dict):
                errs.append(f"{where}{tag}: expected a record mapping")
                continue
            for field_name, field_value in record.items():
                field_key = schema.table_field(key, field_name)
                if field_key is None:
                    suggestions = difflib.get_close_matches(field_name, [f.name for f in key.fields], n=1)
                    errs.append(f"{where}{tag}: unknown field {field_name!r}"
                                + (f" (did you mean {suggestions[0]!r}?)" if suggestions else ""))
                else:
                    _check_type(field_key, field_value, f"{where}{tag}.{field_name}", level_sizes, errs)


# true when every element of `v` lies in [lo, hi].
def _in_bounds(v, lo, hi) -> bool:
    arr = np.asarray(v, dtype=float)
    return bool(np.all(arr >= lo) and np.all(arr <= hi))


# true for a finite number or a flat list of finite numbers (bools excluded).
def _numeric(v) -> bool:
    def num(x):
        return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and math.isfinite(x)
    if isinstance(v, (list, tuple)):
        return len(v) > 0 and all(num(x) for x in v)
    return num(v)


# check that a hardware period fits an integer number of plant ticks.
# convert the value to SI.
# for a rate, test its reciprocal period.
# `dt` is the plant tick duration in seconds.
# the tick-count tolerance is `1e-6 * max(1, abs(ticks))`.
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
                        f"({ticks:.6g} ticks)")
            return


# {'ring' — N, 'line' — L} as far as the config states them (for list-length checks).
def _level_sizes(cfg: Config) -> dict[str, int]:
    sizes = {}
    try:
        sizes["ring"] = cfg.n_rings
    except (TypeError, ValueError):
        pass
    try:
        sizes["line"] = cfg.n_lines
    except (TypeError, ValueError):
        pass
    return sizes


# check the resolved config and append all detected errors.
# check physical leaves, settings, timing, channel layout, and fidelity rules.
# also check required inputs for switches reachable through hypotheses.
# then validate relation options, uncertainty, and per-device variation.
def _validate(cfg: Config, errs: list[str]) -> None:
    dt = cfg.get("sim.dt_plant_s")
    level_sizes = _level_sizes(cfg)
    strict = cfg.kind in _STRICT_KINDS
    # unknown keys, types, provenance fields
    for path, leaf in cfg.leaves.items():
        key = schema.lookup(path)
        if key is None:
            msg = schema.DELETED.get(path)
            if msg:
                errs.append(f"{path}: {msg}")
            else:
                suggestions = schema.suggest(path)
                errs.append(f"unknown key {path!r}" + (f" (did you mean {', '.join(suggestions)}?)" if suggestions else ""))
            continue
        if not key.physical:  # pragma: no cover - settings never become leaves (see load)
            continue
        if key.derived_only and not leaf.is_free:
            rel = ", ".join(key.relation) or "a relation"
            errs.append(f"{path}: is computed by {rel}; a preset must not set it")
        if leaf.provenance == "derived" and leaf.value is not None:
            errs.append(f"{path}: a leaf tagged 'derived' must not carry a value; the relations compute it")
        if leaf.provenance == "derived" and not key.relation and not key.derived_only:
            errs.append(f"{path}: tagged 'derived' but no relation computes it (relation members are listed in "
                        f"docs/PARAMETERS_FORMAT.md section 8); give a value or leave the key out")
        _check_provenance_word(leaf.provenance, path, errs)
        if leaf.provenance.startswith("private") and cfg.kind != "private" and not any(
                "/private/" in f or f.startswith("private") for f in cfg.files):
            errs.append(f"{path}: provenance 'private' is allowed only in presets/private/")
        if strict and leaf.provenance != "override" and not leaf.is_free:
            if leaf.generation is None:
                errs.append(f"{path}: leaf has no generation (one of {sorted(GENERATIONS)})")
            if leaf.source is None:
                errs.append(f"{path}: leaf has no source (a key of mrm/params/sources.yaml)")
        if leaf.generation is not None and leaf.generation not in GENERATIONS:
            suggestions = difflib.get_close_matches(str(leaf.generation), sorted(GENERATIONS), n=1)
            errs.append(f"{path}: generation {leaf.generation!r} not in {sorted(GENERATIONS)}"
                        + (f" (did you mean {suggestions[0]!r}?)" if suggestions else ""))
        if leaf.source is not None and cfg.sources is not None and str(leaf.source) not in cfg.sources:
            suggestions = difflib.get_close_matches(str(leaf.source), sorted(cfg.sources), n=1)
            errs.append(f"{path}: source {leaf.source!r} is not a key of sources.yaml or meta.sources"
                        + (f" (did you mean {suggestions[0]!r}?)" if suggestions else ""))
        if leaf.provenance.startswith("calibrated") and not leaf.targets:
            errs.append(f"{path}: a calibrated leaf must list its targets: [CAL-ids]")
        for alternative_name, a in leaf.alternatives.items():
            if a.get("_bare") and strict:
                errs.append(f"{path}: alternative {alternative_name!r} is a bare value; give {{value, provenance}}")
            elif "provenance" in a and a["provenance"] is not None:
                _check_provenance_word(a["provenance"], f"{path}.alternatives.{alternative_name}", errs)
            alternative_value = a.get("value")
            if (alternative_name != "default" and alternative_value is not None
                    and not (isinstance(alternative_value, dict) and "from" in alternative_value)
                    and "from" not in a):
                # alternatives must satisfy the same type, shape, and bounds as the base value.
                # include each selectable alternative in the declared leaf range.
                _check_type(key, alternative_value, f"{path}.alternatives.{alternative_name}", level_sizes, errs)
                if key.type in ("float", "int") and key.bounds is not None and _numeric(alternative_value) and not _in_bounds(
                        alternative_value, *key.bounds):
                    errs.append(f"{path}.alternatives.{alternative_name}: value {alternative_value!r} is outside the physical bounds "
                                f"{list(key.bounds)}")
                leaf_range = leaf.range
                if (strict and key.type in ("float", "int") and isinstance(leaf_range, tuple) and len(leaf_range) == 2
                        and _numeric(list(leaf_range)) and _numeric(alternative_value) and not _in_bounds(alternative_value, *leaf_range)):
                    errs.append(f"{path}.alternatives.{alternative_name}: value {alternative_value!r} lies outside the leaf's range "
                                f"{list(leaf_range)}, so it can never be selected; widen the range, or remove the "
                                f"alternative (null in a child preset)")
        if leaf.range is not None and not (isinstance(leaf.range, tuple) and len(leaf.range) == 2
                                           and all(_numeric(x) and not isinstance(x, list) for x in leaf.range)):
            errs.append(f"{path}: range must be [low, high] with two numbers, got {leaf.range!r}")
            leaf.range = None
        if leaf.value is None:
            continue
        _check_type(key, leaf.value, path, level_sizes, errs)
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
            records = leaf.value if isinstance(leaf.value, list) else list((leaf.value or {}).values())
            for r in records:
                if isinstance(r, dict):
                    for fname, field_value in r.items():
                        field_key = schema.table_field(key, fname)
                        if field_key is not None and field_key.tick:
                            _check_tick(f"{path}.{fname}", field_key.tick, field_value, dt, errs)
    for path, v in cfg.settings.items():
        key = schema.lookup(path)
        if key is None:
            msg = schema.DELETED.get(path)
            if msg:
                errs.append(f"{path}: {msg}")
            else:
                suggestions = schema.suggest(path)
                errs.append(f"unknown key {path!r}" + (f" (did you mean {', '.join(suggestions)}?)" if suggestions else ""))
            continue
        if key.physical:
            continue
        _check_type(key, v, path, level_sizes, errs)
        if key.bounds is not None and isinstance(v, (int, float)) and not _in_bounds(v, *key.bounds):
            errs.append(f"{path}: {v!r} is outside {list(key.bounds)}")
        if key.tick and dt and v is not None:
            _check_tick(path, key.tick, v, dt, errs)
    # the line grid - a lines table, or centre + spacing + count
    if not cfg.get("channel_plan.lines") and any(cfg.get(k) is None for k in (
            "channel_plan.center_thz", "channel_plan.spacing_ghz", "channel_plan.n_lines")):
        errs.append("channel_plan: give channel_plan.lines, or center_thz, spacing_ghz and n_lines")
    # fidelity rules
    present = {p for p, lf in cfg.leaves.items() if lf.value is not None} | {
        p for p, v in cfg.settings.items() if v is not None}
    base_errors = schema.check_conditions(lambda p: cfg.get(p), present)
    errs.extend(base_errors)
    # check required inputs for every reachable switch choice, including non-nominal worlds.
    for entry_name, entry in sorted(cfg.uncertainty.items()):
        if entry.kind != "hypothesis":
            continue
        for c, m in _hypothesis_choice_map(entry).items():
            switches = {p: a for p, a in m.items()
                        if (k := schema.lookup(p)) is not None and k.per_world and not k.physical}
            if not switches:
                continue
            for msg in schema.check_conditions(
                    lambda p, switches=switches: switches[p] if p in switches else cfg.get(p), present):
                if msg not in base_errors:
                    errs.append(f"uncertainty.{entry_name}={c}: {msg}")
    # relations section
    from mrm.params.relations import RELATION_NAMES
    for r, d in cfg.relations_cfg.items():
        if r not in RELATION_NAMES:
            suggestions = difflib.get_close_matches(r, RELATION_NAMES, n=1)
            errs.append(f"relations.{r}: unknown relation" + (f" (did you mean {suggestions[0]!r}?)" if suggestions else ""))
        elif not isinstance(d, dict) or set(d) - {"tolerance_rel", "note", "source"}:
            errs.append(f"relations.{r}: allowed fields are tolerance_rel, note, source")
        elif "tolerance_rel" in d and not (isinstance(d["tolerance_rel"], (int, float)) and d["tolerance_rel"] >= 0):
            errs.append(f"relations.{r}.tolerance_rel must be a number >= 0")
    _validate_uncertainty(cfg, strict, errs)
    _validate_variation(cfg, strict, errs)


# check the provenance, generation and source of an uncertainty or variation entry.
def _check_entry_provenance(fields: dict, where: str, strict: bool, cfg: Config, errs: list[str]) -> None:
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
    generation = fields.get("generation")
    if generation is not None and generation not in GENERATIONS:
        errs.append(f"{where}: generation {generation!r} not in {sorted(GENERATIONS)}")
    source = fields.get("source")
    if source is not None and cfg.sources is not None and str(source) not in cfg.sources:
        errs.append(f"{where}: source {source!r} is not a known source key")


# check shared model uncertainty, its targets, and its possible values.
# hypotheses must define valid choices consistent with the nominal config.
# numeric entries must use valid distributions, bounds, and units.
# absolute draws and corners must fit the target's physical bounds and declared range.
def _validate_uncertainty(cfg: Config, strict: bool, errs: list[str]) -> None:
    for name, entry in cfg.uncertainty.items():
        where = f"uncertainty.{name}"
        kind = entry.kind
        if kind not in _UNCERTAINTY_FIELDS:
            errs.append(f"{where}: kind {kind!r} must be one of {sorted(_UNCERTAINTY_FIELDS)}")
            continue
        bad = set(entry.fields) - _UNCERTAINTY_FIELDS[kind] - {"_override_paths"}
        if bad:
            for b in sorted(bad):
                s = difflib.get_close_matches(b, sorted(_UNCERTAINTY_FIELDS[kind]), n=1)
                errs.append(f"{where}: unknown field {b!r} for kind {kind}"
                            + (f" (did you mean {s[0]!r}?)" if s else ""))
        _check_entry_provenance(entry.fields, where, strict, cfg, errs)
        target = entry.target
        if target is not None:
            key = schema.lookup(target)
            if key is None:
                sugg = schema.suggest(target)
                errs.append(f"{where}: target {target!r} is not a schema key"
                            + (f" (did you mean {', '.join(sugg)}?)" if sugg else ""))
                continue
            if key.derived_only:
                errs.append(f"{where}: target {target!r} is derived; vary an input instead")
            if kind != "hypothesis" and key.physical and target not in cfg.leaves:
                errs.append(f"{where}: target {target!r} is not set in this config")
            if kind != "hypothesis" and cfg.is_given(target) is False and target in cfg.leaves:
                errs.append(f"{where}: target {target!r} is a free relation member (null or derived); vary an input")
        if kind == "hypothesis":
            choice_map = _hypothesis_choice_map(entry)
            if not choice_map:
                errs.append(f"{where}: needs choices (a list with target:, or {{choice: {{path: alternative}}}})")
                continue
            if "follows" in entry.fields or "map" in entry.fields:
                _validate_label_entry(cfg, entry, choice_map, where, errs)
            nominal = entry.fields.get("nominal")
            if nominal is None or str(nominal) not in choice_map:
                errs.append(f"{where}: nominal {nominal!r} must be one of the choices {sorted(choice_map)}")
            elif entry.pinned is None and entry.restrict is None:
                # derive(), the parameter table, and explain must use the config as written.
                # the nominal choice may only select what the leaves and switches already hold
                skip = set(entry.fields.get("_override_paths", []))
                for p, a in choice_map[str(nominal)].items():
                    if p in skip:
                        continue
                    k = schema.lookup(p)
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
                        errs.append(f"{where}: the nominal choice {nominal!r} sets {p} to {a!r} but the config holds "
                                    f"{held!r}; make the leaf (or switch) the nominal choice's value, or pin the "
                                    f"entry with alts (the nominal world must be the config as written)")
            w = entry.fields.get("weights")
            if w is not None:
                if not isinstance(w, dict) or set(map(str, w)) - set(choice_map) or any(
                        not isinstance(x, (int, float)) or x < 0 for x in w.values()):
                    errs.append(f"{where}: weights must map choices to numbers >= 0")
            for c, m in choice_map.items():
                for p, a in m.items():
                    key = schema.lookup(p)
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
                                _check_type(key, a["value"], f"{where}.choices.{c}.{p}", _level_sizes(cfg), errs)
                                if "provenance" in a:
                                    _check_provenance_word(a["provenance"], f"{where}.choices.{c}.{p}", errs)
                    elif not key.physical:
                        if not key.per_world:
                            errs.append(f"{where}.choices.{c}: {p} is not a per-world switch")
                        elif a not in key.enum:
                            errs.append(f"{where}.choices.{c}: {p}={a!r} not in {list(key.enum)}")
        else:
            mode = entry.fields.get("mode", "absolute")
            if mode not in ("absolute", "scale", "offset"):
                errs.append(f"{where}: mode {mode!r} must be absolute, scale or offset")
            target_key = schema.lookup(target) if target is not None else None
            if target_key is not None and mode == "scale" and parse_key(target)[1].kind in ("celsius", "dbm"):
                errs.append(f"{where}: mode scale multiplies {target}, an affine (_c) or logarithmic (_dbm) value; use "
                            f"mode offset")
            bounds = []
            if kind == "interval":
                lo, hi = entry.fields.get("low"), entry.fields.get("high")
                if not all(_numeric(x) for x in (lo, hi)) or lo > hi:
                    errs.append(f"{where}: needs numbers low <= high")
                else:
                    bounds = [lo, hi]
                d = entry.fields.get("dist", "uniform")
                if d not in ("uniform", "loguniform", "triangular"):
                    errs.append(f"{where}: dist {d!r} must be uniform, loguniform or triangular")
                if d == "loguniform" and _numeric(lo) and lo <= 0:
                    errs.append(f"{where}: loguniform needs low > 0")
                nominal = entry.fields.get("nominal")
                if d == "triangular" and bounds and _numeric(nominal) and not bounds[0] <= nominal <= bounds[1]:
                    errs.append(f"{where}: the triangular mode (nominal {nominal}) must lie in [low, high]")
            else:
                d = entry.fields.get("dist", "normal")
                if d not in ("normal", "lognormal"):
                    errs.append(f"{where}: dist {d!r} must be normal or lognormal")
                if (not _numeric(entry.fields.get("mean")) or not _numeric(entry.fields.get("sigma"))
                        or entry.fields["sigma"] < 0):
                    errs.append(f"{where}: needs numbers mean and sigma >= 0")
                elif d == "lognormal" and entry.fields["mean"] <= 0:
                    errs.append(f"{where}: lognormal needs mean > 0 (x = mean exp(sigma z); mean is the median)")
                truncate = entry.fields.get("truncate")
                if truncate is not None:
                    if not (isinstance(truncate, (list,
                        tuple)) and len(truncate) == 2 and _numeric(list(truncate)) and truncate[0] < truncate[1]):
                        errs.append(f"{where}: truncate must be [low, high] with low < high")
                    else:
                        bounds = list(truncate)
            corner_specs = (entry.fields.get("corners") or {}).values()
            values = bounds + [c["value"] if isinstance(c, dict) else c for c in corner_specs]
            values = [x for x in values if _numeric(x)]
            if target_key is not None and mode == "absolute" and target_key.bounds is not None:
                if values and not _in_bounds(values, *target_key.bounds):
                    errs.append(f"{where}: values {values} reach outside the physical bounds {list(target_key.bounds)} "
                                f"of {target}")
            leaf_range = cfg.leaves[target].range if (target_key is not None and target in cfg.leaves) else None
            if (mode == "absolute" and isinstance(leaf_range, tuple) and len(leaf_range) == 2
                    and _numeric(list(leaf_range))):
                # a world's value of the target is the drawn or pinned value itself, so the leaf's declared range must hold every one of them (a range includes every selectable value)
                out = [x for x in values if not float(leaf_range[0]) <= float(x) <= float(leaf_range[1])]
                if out:
                    errs.append(f"{where}: values {out} lie outside the range {list(leaf_range)} of {target}; a "
                                f"leaf's range must include every value a world can take (widen it or move the "
                                f"bound/corner)")
            if target is None and entry.fields.get("nominal") is None:
                errs.append(f"{where}: an entry without target needs a nominal value")
            nominal = entry.fields.get("nominal")
            if target is not None and nominal is not None and not isinstance(nominal, (int, float)):
                errs.append(f"{where}: nominal must be a number")
            for corner_name, c in (entry.fields.get("corners") or {}).items():
                corner_value = c["value"] if isinstance(c, dict) else c
                if not isinstance(corner_value, (int, float)):
                    errs.append(f"{where}.corners.{corner_name}: must be a number or {{value, ...}}")


# check per-device variation and shared geometry-error settings.
# check target levels, distributions, nonnegative spreads, and correlation lengths.
# offset spreads use the target units.
# scale spreads multiply the target and cannot be applied to Celsius or dBm values.
# geometry spreads use nm.
def _validate_variation(cfg: Config, strict: bool, errs: list[str]) -> None:
    for path, entry in cfg.variation.items():
        where = f"variation.{path}"
        bad = set(entry.fields) - _VARIATION_FIELDS
        for b in sorted(bad):
            s = difflib.get_close_matches(b, sorted(_VARIATION_FIELDS), n=1)
            errs.append(f"{where}: unknown field {b!r}" + (f" (did you mean {s[0]!r}?)" if s else ""))
        _check_entry_provenance(entry.fields, where, strict, cfg, errs)
        key = schema.lookup(path)
        if key is None:
            suggestions = schema.suggest(path)
            errs.append(f"{where}: {path!r} is not a schema key (variation is keyed by config path)"
                        + (f" (did you mean {', '.join(suggestions)}?)" if suggestions else ""))
            continue
        if key.derived_only:
            errs.append(f"{where}: {path} is derived; vary an input")
        if path not in cfg.leaves:
            errs.append(f"{where}: {path} is not set in this config")
        elif cfg.leaves[path].is_free:
            errs.append(f"{where}: {path} is a free relation member; vary an input")
        if key.type != "float" or key.level == "global":
            errs.append(f"{where}: only die-, ring- or line-level float leaves can vary (level of {path}: {key.level})")
        level = entry.fields.get("level", key.level)
        allowed = {"die": ("die",), "ring": ("die", "ring"), "line": ("die", "line")}.get(key.level, ())
        if level not in allowed:
            errs.append(f"{where}: level {level!r} not allowed for a {key.level}-level leaf (allowed {list(allowed)})")
        dist = entry.fields.get("dist", "normal")
        if dist not in ("normal", "lognormal", "uniform"):
            errs.append(f"{where}: dist {dist!r} must be normal, lognormal or uniform")
        mode = entry.fields.get("mode", "scale" if (dist == "lognormal" or "rel_sigma" in entry.fields) else "offset")
        if mode not in ("offset", "scale"):
            errs.append(f"{where}: mode {mode!r} must be offset or scale")
        if dist == "lognormal" and mode != "scale":
            errs.append(f"{where}: lognormal variation is multiplicative (mode: scale)")
        if mode == "scale" and parse_key(path)[1].kind in ("celsius", "dbm"):
            errs.append(f"{where}: a multiplicative (scale, lognormal or rel_sigma) variation of an affine (_c) or "
                        f"logarithmic (_dbm) value has no meaning; use mode offset with sigmas in K or dB")
        corr = entry.fields.get("corr_length_um")
        if corr is not None and not (_numeric(corr) and corr > 0):
            errs.append(f"{where}.corr_length_um: must be a number > 0")
        sigma_fields = [f for f in ("die_sigma", "ring_sigma", "line_sigma", "rel_sigma") if f in entry.fields]
        if not sigma_fields:
            errs.append(f"{where}: needs die_sigma, ring_sigma/line_sigma or rel_sigma")
        for f in sigma_fields:
            if not isinstance(entry.fields[f], (int, float)) or entry.fields[f] < 0:
                errs.append(f"{where}.{f}: must be a number >= 0")
        if "ring_sigma" in entry.fields and level != "ring":
            errs.append(f"{where}: ring_sigma needs level ring")
        if "line_sigma" in entry.fields and level != "line":
            errs.append(f"{where}: line_sigma needs level line")
        if "rel_sigma" in entry.fields and mode != "scale":
            errs.append(f"{where}: rel_sigma is a relative (scale) variation")
    geometry = cfg.geometry
    if geometry:
        bad = set(geometry) - _GEOMETRY_FIELDS
        for b in sorted(bad):
            s = difflib.get_close_matches(b, sorted(_GEOMETRY_FIELDS), n=1)
            errs.append(f"variation.geometry: unknown field {b!r}" + (f" (did you mean {s[0]!r}?)" if s else ""))
        _check_entry_provenance(geometry, "variation.geometry", strict, cfg, errs)
        for d in _GEOMETRY_DIMS:
            if d in geometry:
                if not isinstance(geometry[d], dict) or set(geometry[d]) - _GEOMETRY_DIM_FIELDS:
                    errs.append(f"variation.geometry.{d}: fields must be {sorted(_GEOMETRY_DIM_FIELDS)}")
                    continue
                for f, x in geometry[d].items():
                    if not _numeric(x) or x < 0 or (f == "corr_length_um" and x == 0):
                        errs.append(f"variation.geometry.{d}.{f}: must be a number >= 0 (corr_length_um > 0)")
        if "width_closes_gaps" in geometry and not isinstance(geometry["width_closes_gaps"], bool):
            errs.append("variation.geometry.width_closes_gaps: must be true or false")
        jacobians = geometry.get("jacobians", ["ring.geometry.jacobian"])
        if isinstance(jacobians, str):
            jacobians = [jacobians]
        for jacobian_path in jacobians:
            if jacobian_path not in cfg.leaves:
                errs.append(f"variation.geometry: Jacobian leaf {jacobian_path!r} is not set (the extraction stores the "
                            f"Jacobians; inline them with provenance until it does)")
