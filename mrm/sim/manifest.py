"""Run directories and manifests (07 P1-17): ``runs/<UTC>_<hash8>/`` with the resolved experiment, a manifest, the
metrics and the traces, so that every result can be reproduced from its directory (``mrm rerun runs/<id>``).

Owner: experiments (docs/handoff/experiments.md). Contract: docs/research/07_design_critique.md P1-17 (normative),
P2-11 (hashed artefacts, pinned environment), 4.9 (reproducibility).

Layout of one run directory::

    runs/20261010T120000Z_1a2b3c4d/
        experiment.resolved.yaml   every config leaf with its provenance (mrm.params Config.resolved_tree), the
                                   experiment options with their defaults filled in, and the source (files,
                                   overrides, alternatives)
        manifest.json              schema mrm.run/1: run id, kind, experiment hash, git commit and dirty-diff hash,
                                   numpy / scipy / numba / llvmlite / h5py / pyyaml versions, platform, argv, seeds,
                                   the sha256 of every artefact the config uses and of every input file, the per-case
                                   runner manifests (aggressor calibration, warm start, diagnostics), the wall time
        metrics.json               the summary of the experiment plus every metric of every case (NaN -> null,
                                   +-inf -> "inf" / "-inf"; floats are written with repr, so they round-trip)
        trace.h5                   exact arrays: /metrics/<case>/..., /trace/<case>/<record key>/{t_s, value},
                                   /truth_final/<case>/<field>, and kind-specific groups (/quasi_static, ...)
        inputs/<file>.yaml         a copy of the experiment file (extends: of non-preset parents made absolute), the
                                   file :func:`mrm.sim.experiment.rerun` reloads
        report.txt                 the human-readable report (compare / robust / acceptance / quasi_static)

The directory name is the UTC start time and the first 8 hex digits of the experiment hash (the sha256 of the
resolved config and the resolved experiment options, :attr:`mrm.sim.experiment.Experiment.hash`). Two runs of the
same experiment in the same second get a ``_2``, ``_3`` ... suffix.

Bit-for-bit reproduction needs the same code (git commit and dirty diff), the same environment (numba and numpy
change floating-point results, 07 P2-11), the same config hash and the same seeds: :func:`compare_runs` compares two
run directories array by array (every metric and trace in trace.h5, including the sign of zero and NaN positions).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = ["MANIFEST_SCHEMA", "TRACE_SCHEMA", "REPO_ROOT", "git_state", "environment", "artefacts_used",
           "file_sha256", "new_run_dir", "to_jsonable", "write_json", "write_trace_h5", "read_trace_h5",
           "read_run", "compare_runs", "runs_root"]

MANIFEST_SCHEMA = "mrm.run/1"
TRACE_SCHEMA = "mrm.trace/1"
#: the repository root (the directory that holds the ``mrm`` package)
REPO_ROOT = Path(__file__).resolve().parents[2]
#: data root of the solver artefacts (``extracted/<file>`` in a config resolves under it)
DATA_ROOT = REPO_ROOT / "mrm" / "data"


# ================================================================================================ hashing helpers
def file_sha256(path: str | Path) -> str:
    """sha256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str], root: Path) -> tuple[int, bytes]:
    try:
        p = subprocess.run(["git", *args], cwd=root, capture_output=True, timeout=60)
        return p.returncode, p.stdout
    except (OSError, subprocess.TimeoutExpired):
        return 1, b""


def git_state(root: str | Path | None = None) -> dict:
    """Git commit and dirty-diff hash of the repository that holds the code (07 P1-17).

    ``commit``: HEAD; ``dirty``: any tracked change or untracked (not ignored) file; ``diff_sha256``: sha256 of
    ``git diff HEAD --binary`` followed by the path and sha256 of every untracked, not ignored file (so a new module
    changes it too); ``status``: the porcelain status lines (at most 200). Without git (or outside a work tree) every
    field is None and ``available`` is False.
    """
    root = Path(root) if root is not None else REPO_ROOT
    rc, out = _git(["rev-parse", "HEAD"], root)
    if rc != 0:
        return {"available": False, "commit": None, "dirty": None, "diff_sha256": None, "branch": None,
                "status": []}
    commit = out.decode().strip()
    _, br = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    _, diff = _git(["diff", "HEAD", "--binary"], root)
    _, st = _git(["status", "--porcelain", "--untracked-files=all"], root)
    status = [ln for ln in st.decode(errors="replace").splitlines() if ln.strip()]
    h = hashlib.sha256(diff)
    for ln in sorted(status):
        if ln.startswith("??"):
            rel = ln[3:].strip().strip('"')
            p = root / rel
            h.update(b"\0untracked\0" + rel.encode())
            if p.is_file():
                try:
                    h.update(file_sha256(p).encode())
                except OSError:
                    pass
    return {"available": True, "commit": commit, "branch": br.decode().strip() or None, "dirty": bool(status),
            "diff_sha256": h.hexdigest(), "status": status[:200]}


def environment() -> dict:
    """Versions and platform (07 P1-17, P2-11: numba, llvmlite and numpy change floating-point results)."""
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:                                   # noqa: BLE001 - a missing package is recorded as None
            return None
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "numpy": ver("numpy"), "scipy": ver("scipy"), "numba": ver("numba"), "llvmlite": ver("llvmlite"),
            "h5py": ver("h5py"), "pyyaml": ver("yaml"), "mrm": ver("mrm"),
            "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "byteorder": sys.byteorder,
            "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT", "0"),
            "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS")}


def _artefact_file(spec: str) -> str | None:
    """'extracted/modes_rib160.json:jacobian' -> 'extracted/modes_rib160.json'."""
    if not isinstance(spec, str) or not spec:
        return None
    return spec.split(":", 1)[0]


def artefacts_used(cfg, pp=None, *, data_root: str | Path | None = None) -> list[dict]:
    """The solver artefacts and data files a config (and its realized PlantParams) uses, with their sha256 now.

    Collected from every leaf that took its value from an artefact (``artefact`` of the leaf), every alternative that
    names one (an uncertainty hypothesis may draw it in some world), the thermal ROM of a realized plant in ROM mode,
    and mrm/data/materials.json (read by mrm.materials). Each entry: ``file`` (relative to the data root),
    ``sha256`` (of the file now), ``sidecar`` / ``sidecar_sha256`` when a sidecar exists, ``recorded_sha256`` (what
    the loader verified, when known), ``used_by`` (config paths), ``selected`` (True when a selected value comes from
    it, False when only an alternative names it).
    """
    root = Path(data_root) if data_root is not None else Path(getattr(cfg, "data_root", DATA_ROOT))
    found: dict[str, dict] = {}

    def add(rel: str | None, path: str, selected: bool, recorded: str | None = None):
        if not rel:
            return
        e = found.setdefault(rel, {"file": rel, "used_by": [], "selected": False, "recorded_sha256": None})
        if path not in e["used_by"]:
            e["used_by"].append(path)
        e["selected"] = e["selected"] or selected
        if recorded and not e["recorded_sha256"]:
            e["recorded_sha256"] = recorded

    for p, leaf in sorted(getattr(cfg, "leaves", {}).items()):
        art = getattr(leaf, "artefact", None)
        if art:
            add(art.get("file"), p, True, art.get("sha256"))
        for an, a in sorted((getattr(leaf, "alternatives", None) or {}).items()):
            a_art = a.get("artefact") or a.get("_resolved_artefact")
            if a_art:
                add(a_art.get("file"), f"{p}#{an}", False, a_art.get("sha256"))
            spec = a.get("from")
            if spec is None and isinstance(a.get("value"), dict):
                spec = a["value"].get("from")
            add(_artefact_file(spec), f"{p}#{an}", False)
    # settings that name a data file: the thermal ROM (fidelity.thermal_model rom) and the junction LUT
    # (fidelity.junction_model lut); read the way realize() reads them
    for key, switch, value in (("ring.thermal.rom_file", "fidelity.thermal_model", "rom"),
                               ("ring.junction.lut_file", "fidelity.junction_model", "lut")):
        try:
            f, sw = cfg.get(key), cfg.get(switch)
        except Exception:                                   # noqa: BLE001 - a mapping config
            f, sw = None, None
        if f and sw == value:
            add(_artefact_file(str(f)), key, True)
    ti = getattr(pp, "thermal_inputs", None) if pp is not None else None
    if ti is not None and getattr(ti, "rom_path", None):
        rp = Path(ti.rom_path)
        try:
            rel = str(rp.relative_to(root))
        except ValueError:
            rel = str(rp)
        add(rel, "thermal_rom", True, getattr(ti, "sha256", None))
    mat = root / "materials.json"
    if mat.exists():
        add("materials.json", "mrm.materials", True)
    out = []
    from mrm.params.config import sidecar_candidates
    for rel, e in sorted(found.items()):
        p = Path(rel) if Path(rel).is_absolute() else root / rel
        e["exists"] = p.exists()
        e["sha256"] = file_sha256(p) if p.is_file() else None
        side = None
        for c in sidecar_candidates(p):
            if c.is_file() and c != p:
                side = c
                break
        e["sidecar"] = side.name if side is not None else None
        e["sidecar_sha256"] = file_sha256(side) if side is not None else None
        if e["recorded_sha256"] and e["sha256"] and e["recorded_sha256"] != e["sha256"]:
            e["mismatch"] = True
        out.append(e)
    return out


# ================================================================================================ run directories
def runs_root(runs_dir: str | Path | None = None) -> Path:
    """The directory that holds run directories: ``runs_dir``, else $MRM_RUNS_DIR, else ./runs (the current working
    directory; the repository's runs/ is git-ignored)."""
    if runs_dir is not None:
        return Path(runs_dir)
    env = os.environ.get("MRM_RUNS_DIR")
    return Path(env) if env else Path.cwd() / "runs"


def new_run_dir(runs_dir: str | Path | None, hash8: str, *, now: _dt.datetime | None = None) -> Path:
    """Create ``<runs_dir>/<UTC>_<hash8>/`` (``_2``, ``_3`` ... when it exists) and its ``inputs/`` folder."""
    root = runs_root(runs_dir)
    root.mkdir(parents=True, exist_ok=True)
    t = (now or _dt.datetime.now(_dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    base = f"{t}_{hash8}"
    d = root / base
    k = 2
    while True:
        try:
            d.mkdir()
            break
        except FileExistsError:
            d = root / f"{base}_{k}"
            k += 1
    (d / "inputs").mkdir()
    return d


# ================================================================================================ JSON
def to_jsonable(x: Any) -> Any:
    """Plain JSON value: numpy arrays -> nested lists, NaN -> None, +-inf -> 'inf' / '-inf', tuples -> lists,
    dataclasses and objects with ``to_dict`` -> dicts, anything else -> str."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "fc":
            return [to_jsonable(v) for v in x.tolist()] if x.ndim == 1 else [to_jsonable(v) for v in x]
        if x.ndim == 0:
            return to_jsonable(x.item())
        return [to_jsonable(v) for v in x] if x.ndim > 1 else [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, Mapping):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in x]
    if hasattr(x, "to_dict") and callable(x.to_dict):
        return to_jsonable(x.to_dict())
    import dataclasses
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return to_jsonable(dataclasses.asdict(x))
    if isinstance(x, Path):
        return str(x)
    return str(x)


def write_json(path: str | Path, obj: Any) -> None:
    """Write ``to_jsonable(obj)`` as indented JSON (strict: no NaN tokens)."""
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=1, sort_keys=False, allow_nan=False)
        f.write("\n")


# ================================================================================================ HDF5
def _h5_put(g, name: str, v: Any) -> None:
    """Store v under group g: arrays and numbers as datasets, mappings as sub-groups, strings as attributes,
    lists of numbers as arrays, other lists as JSON attributes."""
    name = str(name).replace("/", "|")
    if v is None:
        g.attrs[name] = "null"
        return
    if isinstance(v, Mapping):
        sg = g.require_group(name)
        for k, x in v.items():
            _h5_put(sg, k, x)
        return
    if isinstance(v, str):
        g.attrs[name] = v
        return
    if isinstance(v, (bool, np.bool_, int, float, np.integer, np.floating)):
        g.create_dataset(name, data=np.asarray(v))
        return
    if isinstance(v, np.ndarray):
        if v.dtype.kind in "biufc":
            g.create_dataset(name, data=v)
        else:
            g.attrs[name] = json.dumps(to_jsonable(v))
        return
    if isinstance(v, (list, tuple)):
        try:
            a = np.asarray(v)
        except (ValueError, TypeError):
            a = None
        if a is not None and a.dtype.kind in "biuf" and a.dtype != object:
            g.create_dataset(name, data=a)
        else:
            g.attrs[name] = json.dumps(to_jsonable(v))
        return
    g.attrs[name] = json.dumps(to_jsonable(v))


def write_trace_h5(path: str | Path, groups: Mapping[str, Any], *, attrs: Mapping[str, Any] | None = None) -> None:
    """Write the exact arrays of a run to HDF5 (07 P1-17 trace.h5; h5py). ``groups``: {top-level name: nested
    mapping of arrays}. Datasets keep their dtype and every bit (NaN, the sign of zero)."""
    import h5py
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = TRACE_SCHEMA
        for k, v in (attrs or {}).items():
            f.attrs[str(k)] = v if isinstance(v, (str, int, float)) else json.dumps(to_jsonable(v))
        for name, tree in groups.items():
            if tree is None:
                continue
            _h5_put(f, name, tree)


def _h5_get(g) -> dict:
    import h5py
    out: dict = {}
    for k, v in g.items():
        out[k] = _h5_get(v) if isinstance(v, h5py.Group) else v[()]
    for k, v in g.attrs.items():
        out.setdefault(f"@{k}", v)
    return out


def read_trace_h5(path: str | Path) -> dict:
    """Every dataset of a trace.h5 as nested dicts of numpy arrays (attributes under '@name')."""
    import h5py
    with h5py.File(path, "r") as f:
        return _h5_get(f)


def read_run(run_dir: str | Path) -> dict:
    """{manifest, metrics, trace (read_trace_h5), resolved_yaml (text), dir} of a run directory."""
    d = Path(run_dir)
    if not (d / "manifest.json").exists():
        raise FileNotFoundError(f"{d}: not a run directory (no manifest.json)")
    with open(d / "manifest.json") as f:
        man = json.load(f)
    met = None
    if (d / "metrics.json").exists():
        with open(d / "metrics.json") as f:
            met = json.load(f)
    tr = read_trace_h5(d / "trace.h5") if (d / "trace.h5").exists() else {}
    res = (d / "experiment.resolved.yaml").read_text() if (d / "experiment.resolved.yaml").exists() else None
    return {"dir": d, "manifest": man, "metrics": met, "trace": tr, "resolved_yaml": res}


def _flat_arrays(tree: Mapping, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in tree.items():
        p = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(_flat_arrays(v, p))
        else:
            out[p] = v
    return out


def _bit_equal(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.dtype.kind in "fc":
            return a.tobytes() == b.tobytes()
        if a.dtype == object:
            return a.tolist() == b.tolist()
        return np.array_equal(a, b)
    if isinstance(a, float) and isinstance(b, float):
        return np.asarray(a).tobytes() == np.asarray(b).tobytes()
    return a == b


#: trace.h5 attributes that legitimately differ between a run and its rerun
_VOLATILE_ATTRS = {"@run_id", "@created_utc", "@wall_s", "@rerun_of"}


def compare_runs(a: str | Path | Mapping, b: str | Path | Mapping, *, ignore: Iterable[str] = ()) -> dict:
    """Bit-for-bit comparison of the trace.h5 arrays of two runs (paths or :func:`read_run` results).

    Returns {identical, n_compared, differing: [path, ...], only_in_a, only_in_b, max_abs_diff: {path: value}}.
    Timing (wall time) and run ids are not compared; everything else in trace.h5 is (metrics, traces, final Truth,
    kind-specific arrays)."""
    ra = read_run(a) if not isinstance(a, Mapping) else a
    rb = read_run(b) if not isinstance(b, Mapping) else b
    fa = {k: v for k, v in _flat_arrays(ra["trace"]).items() if k.rsplit("/", 1)[-1] not in _VOLATILE_ATTRS}
    fb = {k: v for k, v in _flat_arrays(rb["trace"]).items() if k.rsplit("/", 1)[-1] not in _VOLATILE_ATTRS}
    ign = tuple(ignore)
    keys_a = {k for k in fa if not any(k.startswith(i) for i in ign)}
    keys_b = {k for k in fb if not any(k.startswith(i) for i in ign)}
    diff, mad = [], {}
    for k in sorted(keys_a & keys_b):
        if not _bit_equal(fa[k], fb[k]):
            diff.append(k)
            try:
                x, y = np.asarray(fa[k], float), np.asarray(fb[k], float)
                if x.shape == y.shape:
                    with np.errstate(invalid="ignore"):
                        mad[k] = float(np.nanmax(np.abs(x - y))) if x.size else 0.0
            except (TypeError, ValueError):
                pass
    return {"identical": not diff and keys_a == keys_b, "n_compared": len(keys_a & keys_b), "differing": diff,
            "only_in_a": sorted(keys_a - keys_b), "only_in_b": sorted(keys_b - keys_a), "max_abs_diff": mad}
