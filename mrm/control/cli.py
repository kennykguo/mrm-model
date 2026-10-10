"""Controller and hardware utilities of the ``mrm`` command line: ``mrm golden``, ``mrm bridge``, ``mrm vectors``
(07 P1-21, section 4.9). Registered through ``mrm.cli.PROVIDERS`` (``mrm.control.cli:register_cli``).

Golden traces (07 4.9, 4.1)
---------------------------
``tests/golden/<preset>__<scenario>.npz`` hold the METRICS (``mrm.sim.metrics.summarize``, no traces) of fixed runs
on the engine. They answer 07 4.1's question for ``tests/golden/``: "has a refactor changed the plant that earlier
controller results were scored on?", and the golden tests block CI. A file changes only through
``mrm golden --update --reason "..."``, which prints the CHANGELOG line to add; tests/golden/test_golden.py fails
until docs/CHANGELOG.md contains each file's data hash (first 12 hex digits of ``meta['data_sha256']``).

Cases (:data:`GOLDEN_CASES`; every run: g26 nominal world (uncertainty nominal, variation off), seeds 0, B = 1,
heaters at the quasi-static oracle lock codes with the thermal state settled; 07 P1-1 warm start):

- ``g26__static`` (static lock): scenario ``static`` (25 degC), 20 ms, fixed 1 us ticks;
- ``g26__a1_slide10_aggressor`` (the fast aggressor, 07 D6): A1 with ``phase_s = P/2`` (t = 0 at the cool minimum,
  the hot cusp at 65.35 ms), one period (130.7 ms), fixed 1 us ticks, aggressor power calibrated by the engine;
- ``g26__soak_compressed`` (the compressed soak, 07 P0-10): one soak cycle (33 -> 108 -> 33 degC, 20.3 min) at
  time compression 100 (12.18 s simulated), substep policy auto, metrics from 5 ms (``score_from_s``), pi_lock_c
  only. This is exactly the c = 100 case of the 07 P0-10 convergence experiment of
  experiments/acceptance/a4_soak.yaml, whose certificate (one cycle at c = 1, 10, 100 agreeing on max detuning,
  wrong-line and FSR-hop events, heater range and integrated errors) validates compression 100 for this plant,
  controller, start and world (docs/handoff/experiments.md section 7). The open loop and fxp_pi are not run here:
  no convergence experiment covers them, and 07 P0-10 allows no compressed result without one. 95 s idle, ~3.5 min
  under load.

``g26__static`` and ``g26__a1_slide10_aggressor`` run three controllers: ``open_loop`` (no controller: the plant
alone, the most direct refactor detector), ``pi_lock_c`` (the registered compiled baseline of the controller
scorecard) and ``fxp_pi`` (mrm.control.fxp.FxpPi, the fixed-point reference of the golden vectors and the bridge).
Comparison: integer and
bool metrics exactly; float metrics within ``|a - b| <= rtol max(|a|, |b|) + atol_scale max|ref|`` with rtol = 1e-9
and atol_scale = 1e-12 (any physics change and any change of a code, a lock state or a count fails; relative
differences below 1e-9, e.g. another libm's last bits in an open-loop metric, pass); non-finite values only
exactly (NaN = NaN, +inf = +inf; the histogram edges are +-inf); text settings exactly. The tolerance is not a
statement that results are expected to move: the engine's refactors are bit-identical (docs/handoff/perf.md), and a
rounding-level change of the engine's floating-point path (perf round 1: 2e-8 relative) fails here on purpose, so it is
acknowledged with ``--update`` and a CHANGELOG line. Closed loops can turn a last-bit change into another, equally
valid trajectory (one ADC code flips; perf.md 5.4), so a golden failure after an intended numerical change is
expected; the metadata records the environment (Python, NumPy, numba, llvmlite, platform) to tell such a case
from a plant change.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = ["register_cli", "GOLDEN_FORMAT", "CONFIG_SCENARIO", "GoldenRun", "GoldenCase", "GoldenData",
           "GOLDEN_CASES", "DEFAULT_RUN_KWARGS", "RTOL", "ATOL_SCALE", "golden_dir", "golden_path", "run_golden_case",
           "save_golden", "load_golden", "compare_golden", "check_golden", "update_golden", "changelog_line",
           "changelog_path", "flatten_metrics", "data_sha256"]

GOLDEN_FORMAT = "mrm.golden/1"
RTOL = 1e-9
ATOL_SCALE = 1e-12
_A1_PERIOD_S = 0.1307            # mrm.scenarios.library.A1_PERIOD_S (read-off; checked in tests/golden)
_SOAK_CYCLE_S = 20.3 * 60        # mrm.scenarios.library.SOAK_CYCLE_S (read-off; checked in tests/golden)
_SOAK_COMPRESSION = 100.0        # experiments/acceptance/a4_soak.yaml (07 P0-10 limit ~173 for the soak)
#: GoldenCase.scenario value: the scenario of the case's config (mrm.scenarios.registry.from_config)
CONFIG_SCENARIO = "config"

#: run options shared by every golden run (07 P1-1 warm start: the oracle lock codes, settled thermal state)
DEFAULT_RUN_KWARGS: dict = {"uncertainty": "nominal", "variation": "off", "seeds": 0, "batch": 1,
                            "heater_code": "quasi_static", "settle": True}


@dataclass(frozen=True)
class GoldenRun:
    """One controller of a case: ``label`` (key prefix in the file), ``controller`` (registry name or
    package.module:Class, None = open loop) and its parameters."""
    label: str
    controller: str | None
    controller_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GoldenCase:
    """A golden trace: ``name`` = file stem ``<preset>__<scenario>``; the scenario is a registry name with keyword
    parameters, or :data:`CONFIG_SCENARIO` ('config'): the scenario section of the case's config (preset +
    ``overrides``) as mrm.scenarios.registry.from_config builds it, i.e. as an experiment file runs it, with the
    config's signal defaults (scenario.ambient_c, scenario.bias_v: 1.05 V in g26, transition density). A registry
    name gets no defaults: the bias is then not driven and the bias DAC sits at its initial code (1.0 V in g26).
    ``pending``: why the case cannot run yet (it is listed and skipped, never silently dropped)."""
    name: str
    preset: str
    scenario: str
    duration_s: float
    runs: tuple
    scenario_params: dict = field(default_factory=dict)
    overrides: tuple = ()
    run_kwargs: dict = field(default_factory=lambda: dict(DEFAULT_RUN_KWARGS))
    description: str = ""
    pending: str | None = None

    def definition(self) -> dict:
        """JSON-serializable definition (stored in the file; a changed definition is reported by the check)."""
        d = asdict(self)
        d["runs"] = [asdict(r) if not isinstance(r, dict) else r for r in self.runs]
        d["overrides"] = list(self.overrides)
        return d


_RUNS = (GoldenRun("open_loop", None), GoldenRun("pi_lock_c", "pi_lock_c"),
         GoldenRun("fxp_pi", "mrm.control.fxp:FxpPi"))

GOLDEN_CASES: dict[str, GoldenCase] = {
    "g26__static": GoldenCase(
        name="g26__static", preset="g26", scenario="static", duration_s=20e-3, runs=_RUNS,
        description="static lock: g26 nominal world, 25 degC, settled warm start at the quasi-static lock codes"),
    "g26__a1_slide10_aggressor": GoldenCase(
        name="g26__a1_slide10_aggressor", preset="g26", scenario="a1_slide10_aggressor",
        scenario_params={"phase_s": _A1_PERIOD_S / 2}, duration_s=_A1_PERIOD_S, runs=_RUNS,
        description="the fast aggressor (07 D6): A1 slide-10 at the g26 default site, one period from the cool "
                    "minimum, settled warm start"),
    # 07 4.9 "and the compressed soak" (integrator, 2026-10-10): the c = 100 case of the A4 convergence experiment
    # (experiments/acceptance/a4_soak.yaml), whose 07 P0-10 certificate validated compression 100 for pi_lock_c on
    # this plant, start and world; only that controller, since no convergence experiment covers the others. The
    # overrides make the config the A4 file's (controller.name only names the controller in the config, the run
    # passes it): at creation mrm.sim.experiment.plant_fingerprint of this config (start warm, the A4 worlds) and
    # the scenario fingerprint equal the certificate's (8e979786..., 79d9c265...)
    "g26__soak_compressed": GoldenCase(
        name="g26__soak_compressed", preset="g26", scenario=CONFIG_SCENARIO,
        duration_s=_SOAK_CYCLE_S / _SOAK_COMPRESSION, runs=(GoldenRun("pi_lock_c", "pi_lock_c"),),
        overrides=("controller.name=pi_lock_c", "scenario.name=soak",
                   f"scenario.time_compression={_SOAK_COMPRESSION}", "sim.substep_policy=auto"),
        run_kwargs=dict(DEFAULT_RUN_KWARGS, score_from_s=0.005),
        description="the compressed soak (07 P0-10): one 20.3 min soak cycle 33 -> 108 -> 33 degC at time "
                    "compression 100 (12.18 s), auto substeps, settled warm start, pi_lock_c scored from 5 ms; the "
                    "c = 100 case of the A4 convergence experiment, which validated compression 100"),
}


@dataclass
class GoldenData:
    """The content of a golden file: numeric metric arrays, text settings and metadata."""
    arrays: dict
    text: dict
    meta: dict


# ------------------------------------------------------------------------------------------------ paths
def _repo_root() -> Path:
    import mrm
    return Path(mrm.__file__).resolve().parents[1]


def golden_dir() -> Path:
    """``$MRM_GOLDEN_DIR`` or ``<repo>/tests/golden``."""
    env = os.environ.get("MRM_GOLDEN_DIR")
    return Path(env) if env else _repo_root() / "tests" / "golden"


def golden_path(name: str, directory: Path | str | None = None) -> Path:
    return Path(directory or golden_dir()) / f"{name}.npz"


def changelog_path() -> Path:
    return _repo_root() / "docs" / "CHANGELOG.md"


# ------------------------------------------------------------------------------------------------ data
def flatten_metrics(m: Mapping[str, Any], prefix: str = "") -> tuple[dict, dict]:
    """(numeric arrays, text values) of a nested metrics dict, keys joined with '/'."""
    arrays, text = {}, {}
    for k, v in m.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            a, t = flatten_metrics(v, key + "/")
            arrays.update(a)
            text.update(t)
        elif v is None or isinstance(v, str):
            text[key] = v
        else:
            a = np.asarray(v)
            if a.dtype.kind in "OUS":
                text[key] = json.dumps(a.tolist())
            else:
                arrays[key] = np.ascontiguousarray(a)
    return arrays, text


def data_sha256(arrays: Mapping[str, np.ndarray], text: Mapping[str, Any]) -> str:
    """sha256 over the sorted metric arrays (name, dtype, shape, bytes) and the text settings."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    h.update(json.dumps(dict(text), sort_keys=True).encode())
    return h.hexdigest()


def _environment() -> dict:
    import numba
    import llvmlite
    return {"python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__,
            "llvmlite": llvmlite.__version__, "machine": platform.machine(), "system": platform.system()}


def run_golden_case(case: GoldenCase, *, progress: Callable[[str], None] | None = None) -> GoldenData:
    """Run every controller of ``case`` and collect its metrics (keys ``<label>/<metric>``)."""
    if case.pending:
        raise RuntimeError(f"golden case {case.name} is pending: {case.pending}")
    from mrm import params
    from mrm.control import registry
    from mrm.scenarios import registry as scen_registry
    from mrm.sim import runner
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        cfg = params.load(case.preset, overrides=list(case.overrides) or None)
    arrays: dict = {}
    text: dict = {}
    runs_meta = {}
    for r in case.runs:
        t0 = time.perf_counter()
        if case.scenario == CONFIG_SCENARIO:
            if case.scenario_params:
                raise ValueError(f"golden case {case.name}: scenario 'config' takes its parameters from the "
                                 f"overrides (scenario.params.<p>), not scenario_params")
            sc = scen_registry.from_config(cfg)
        else:
            sc = scen_registry.create(case.scenario, **dict(case.scenario_params))
        ctrl = None if r.controller is None else registry.create(r.controller, dict(r.controller_params))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            res = runner.run(cfg, ctrl, sc, duration_s=case.duration_s, **dict(case.run_kwargs))
        a, t = flatten_metrics(res.metrics, f"{r.label}/")
        arrays.update(a)
        text.update(t)
        mi = res.manifest_inputs
        runs_meta[r.label] = {"config_fingerprint": mi.get("config_fingerprint"),
                              "scenario_fingerprint": mi.get("scenario_fingerprint"),
                              "spec_fingerprint": mi.get("spec_fingerprint"), "n_ticks": mi.get("n_ticks"),
                              "controller": mi.get("controller"), "controller_path": mi.get("controller_path"),
                              "aggressor_calibration": _jsonable(mi.get("aggressor_calibration")),
                              "settle": _jsonable((mi.get("warm_start") or {}).get("settle")),
                              "wall_s": round(time.perf_counter() - t0, 3)}
        if progress:
            progress(f"{case.name}/{r.label}: {time.perf_counter() - t0:.1f} s")
    meta = {"format": GOLDEN_FORMAT, "case": case.definition(), "runs": runs_meta, "environment": _environment(),
            "data_sha256": data_sha256(arrays, text), "tolerance": {"rtol": RTOL, "atol_scale": ATOL_SCALE}}
    return GoldenData(arrays=arrays, text=text, meta=meta)


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else
                                 (o.item() if isinstance(o, np.generic) else repr(o))))


def save_golden(path: Path | str, data: GoldenData, *, reason: str) -> Path:
    """Write a golden file (metrics arrays + ``__text__`` + ``__meta__`` JSON). ``reason`` is recorded."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(data.meta, reason=str(reason),
                created_utc=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    payload = {f"m:{k}": v for k, v in data.arrays.items()}
    payload["__text__"] = np.array(json.dumps(data.text, sort_keys=True))
    payload["__meta__"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez_compressed(p, **payload)
    return p


def load_golden(path: Path | str) -> GoldenData:
    """Read a golden file; checks the format and the data hash."""
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        text = json.loads(str(z["__text__"]))
        arrays = {k[2:]: np.array(z[k]) for k in z.files if k.startswith("m:")}
    if meta.get("format") != GOLDEN_FORMAT:
        raise ValueError(f"{p}: format {meta.get('format')!r}, expected {GOLDEN_FORMAT!r}")
    if data_sha256(arrays, text) != meta.get("data_sha256"):
        raise ValueError(f"{p}: the metric arrays do not match the recorded data_sha256 (edited by hand?); "
                         f"regenerate with `mrm golden --update`")
    return GoldenData(arrays=arrays, text=text, meta=meta)


def compare_golden(ref: GoldenData, new: GoldenData, *, rtol: float = RTOL,
                   atol_scale: float = ATOL_SCALE) -> list[str]:
    """Differences of ``new`` against ``ref`` (empty when equal within the stated rule; module docstring)."""
    out: list[str] = []
    for k in sorted(set(ref.arrays) | set(new.arrays)):
        if k not in new.arrays:
            out.append(f"{k}: missing from the new run")
            continue
        if k not in ref.arrays:
            out.append(f"{k}: new metric (not in the golden file)")
            continue
        a, b = ref.arrays[k], new.arrays[k]
        if a.shape != b.shape:
            out.append(f"{k}: shape {a.shape} -> {b.shape}")
            continue
        if a.dtype.kind in "biu" and b.dtype.kind in "biu":
            bad = a != b
            if bad.any():
                i = tuple(int(x) for x in np.argwhere(bad)[0])
                out.append(f"{k}: {int(bad.sum())} of {a.size} differ, first {list(i)}: {a[i].item()} -> "
                           f"{b[i].item()}")
            continue
        fa, fb = a.astype(np.float64), b.astype(np.float64)
        both_nan = np.isnan(fa) & np.isnan(fb)
        fin = np.isfinite(fa)
        both_fin = fin & np.isfinite(fb)
        scale = float(np.max(np.abs(fa[fin]))) if fin.any() else 0.0
        with np.errstate(invalid="ignore", over="ignore"):
            tol = rtol * np.maximum(np.abs(fa), np.abs(fb)) + atol_scale * scale
            # non-finite values match only exactly (NaN = NaN, +inf = +inf): a tolerance scaled by an infinity
            # would accept finite -> inf and -inf -> +inf
            ok = both_nan | (fa == fb) | (both_fin & (np.abs(fa - fb) <= tol))
        if not ok.all():
            i = tuple(int(x) for x in np.argwhere(~ok)[0])
            with np.errstate(invalid="ignore", over="ignore"):
                d = np.abs(fa - fb)[~ok]
                big = np.nanmax(d) if np.isfinite(d).any() or np.isinf(d).any() else d.max()
            out.append(f"{k}: {int((~ok).sum())} of {a.size} differ, first {list(i)}: {fa[i]!r} -> {fb[i]!r} "
                       f"(largest |diff| {big!r})")
    for k in sorted(set(ref.text) | set(new.text)):
        if ref.text.get(k) != new.text.get(k):
            out.append(f"{k}: {ref.text.get(k)!r} -> {new.text.get(k)!r}")
    if out:
        info = []
        if ref.meta.get("case") != new.meta.get("case"):
            info.append("note: the case definition changed since the golden file was written")
        for lab, rm in (ref.meta.get("runs") or {}).items():
            nm = (new.meta.get("runs") or {}).get(lab, {})
            for key in ("config_fingerprint", "scenario_fingerprint", "spec_fingerprint"):
                if rm.get(key) != nm.get(key):
                    info.append(f"note: {lab} {key} {rm.get(key)} -> {nm.get(key)}")
        if ref.meta.get("environment") != new.meta.get("environment"):
            info.append(f"note: environment {ref.meta.get('environment')} -> {new.meta.get('environment')} "
                        f"(a closed loop can follow another trajectory after a last-bit change)")
        out += info
    return out


def changelog_line(name: str, data: GoldenData, reason: str) -> str:
    """The CHANGELOG line that must accompany a golden update (contains the 12-digit data hash)."""
    return (f"- golden: tests/golden/{name}.npz updated (data sha256 {data.meta['data_sha256'][:12]}): "
            f"{reason}")


def _select(names: Sequence[str] | None) -> list[GoldenCase]:
    if not names:
        return list(GOLDEN_CASES.values())
    bad = [n for n in names if n not in GOLDEN_CASES]
    if bad:
        raise KeyError(f"unknown golden case(s) {bad}; known: {sorted(GOLDEN_CASES)}")
    return [GOLDEN_CASES[n] for n in names]


def check_golden(names: Sequence[str] | None = None, directory: Path | str | None = None, *,
                 progress: Callable[[str], None] | None = None) -> dict[str, list[str] | str]:
    """Run the cases and compare them with their files: {case: differences (empty = equal) | 'pending: ...' |
    'missing: ...'}."""
    out: dict[str, list[str] | str] = {}
    for case in _select(names):
        if case.pending:
            out[case.name] = f"pending: {case.pending}"
            continue
        p = golden_path(case.name, directory)
        if not p.exists():
            out[case.name] = f"missing: {p} (create it with `mrm golden --update --case {case.name} --reason ...`)"
            continue
        out[case.name] = compare_golden(load_golden(p), run_golden_case(case, progress=progress))
    return out


def update_golden(names: Sequence[str] | None, reason: str, directory: Path | str | None = None, *,
                  progress: Callable[[str], None] | None = None) -> list[str]:
    """Recompute and write the golden files of ``names`` (all non-pending cases if None). Returns the CHANGELOG
    lines to add (one per file whose data changed)."""
    if not reason or not reason.strip():
        raise ValueError("a golden update needs a reason (it goes into the file and the CHANGELOG line)")
    lines = []
    for case in _select(names):
        if case.pending:
            if names:
                raise RuntimeError(f"golden case {case.name} is pending: {case.pending}")
            continue
        p = golden_path(case.name, directory)
        old = load_golden(p) if p.exists() else None
        data = run_golden_case(case, progress=progress)
        if old is not None and old.meta.get("data_sha256") == data.meta["data_sha256"]:
            if old.meta.get("case") != json.loads(json.dumps(data.meta.get("case"))):
                # same metrics, new case definition (e.g. a description): rewrite the metadata; the CHANGELOG
                # already names this data hash, so no new line is needed
                save_golden(p, data, reason=f"{old.meta.get('reason', '')} [definition updated: {reason}]")
                if progress:
                    progress(f"{case.name}: data unchanged, case definition updated in the file")
            elif progress:
                progress(f"{case.name}: unchanged (data sha256 {data.meta['data_sha256'][:12]})")
            continue
        save_golden(p, data, reason=reason)
        lines.append(changelog_line(case.name, data, reason))
    return lines


# ------------------------------------------------------------------------------------------------ commands
def _golden_setup(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--update", action="store_true", help="recompute and rewrite the golden files")
    ap.add_argument("--reason", default=None, help="why the golden files change (required with --update)")
    ap.add_argument("--case", action="append", default=None, help="limit to these cases (repeatable)")
    ap.add_argument("--list", action="store_true", help="list the cases and exit")
    ap.add_argument("--dir", default=None, help="golden directory (default tests/golden or $MRM_GOLDEN_DIR)")


def _golden_run(args: argparse.Namespace) -> int:
    def prog(msg):
        print(msg, file=sys.stderr, flush=True)
    if args.list:
        for c in GOLDEN_CASES.values():
            state = f"PENDING ({c.pending})" if c.pending else str(golden_path(c.name, args.dir))
            print(f"{c.name}: {c.description} [{', '.join(r.label for r in c.runs)}] -> {state}")
        return 0
    if args.update:
        if not args.reason:
            print("mrm golden --update needs --reason \"...\" (it is recorded with the CHANGELOG line)",
                  file=sys.stderr)
            return 2
        lines = update_golden(args.case, args.reason, args.dir, progress=prog)
        if lines:
            print("Add these lines to docs/CHANGELOG.md (tests/golden/test_golden.py checks for the hashes):")
            print("\n".join(lines))
        else:
            print("golden files unchanged")
        return 0
    res = check_golden(args.case, args.dir, progress=prog)
    bad = 0
    for name, r in res.items():
        if isinstance(r, str):
            print(f"{name}: {r}")
            bad += r.startswith("missing")
        elif r:
            bad += 1
            print(f"{name}: DIFFERS ({len(r)} lines)")
            for line in r[:40]:
                print("   ", line)
        else:
            print(f"{name}: matches")
    return 1 if bad else 0


def _bridge_setup(ap: argparse.ArgumentParser) -> None:
    from mrm.sim import bridge
    bridge.add_arguments(ap)


def _bridge_run(args: argparse.Namespace) -> int:
    from mrm.sim import bridge
    return bridge.run_command(args)


def _vectors_setup(ap: argparse.ArgumentParser) -> None:
    sub = ap.add_subparsers(dest="vectors_cmd", required=True)
    r = sub.add_parser("record", help="run a controller on the engine and write its golden vectors")
    r.add_argument("--preset", default="g26")
    r.add_argument("--override", action="append", default=[])
    r.add_argument("--scenario", default="static")
    r.add_argument("--scenario-params", default=None, help="JSON keyword arguments of the scenario builder")
    r.add_argument("--controller", default="mrm.control.fxp:FxpPi", help="registry name or package.module:Class")
    r.add_argument("--params", default=None, help="JSON controller parameters")
    r.add_argument("--duration-s", type=float, default=2e-3)
    r.add_argument("--seeds", type=int, default=0)
    r.add_argument("--cold", action="store_true", help="start cold (default: settled warm start at the lock)")
    r.add_argument("--out", required=True, help="output stem (writes STEM.npz, STEM.*.hex, STEM.layout.vh)")
    c = sub.add_parser("replay", help="replay recorded observations through a compiled controller (ABI)")
    c.add_argument("vectors", help="a .npz written by record")
    c.add_argument("--controller", default="mrm.control.fxp:FxpPi")
    c.add_argument("--params", default=None)
    h = sub.add_parser("hex", help="(re)write the .hex and .vh files of a .npz")
    h.add_argument("vectors")
    h.add_argument("--out", default=None, help="output stem (default: next to the .npz)")


def _vectors_run(args: argparse.Namespace) -> int:
    from mrm.control import registry
    from mrm.sim import vectors
    if args.vectors_cmd == "record":
        from mrm import params
        from mrm.scenarios import registry as scen_registry
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            cfg = params.load(args.preset, overrides=list(args.override) or None)
        sc = scen_registry.create(args.scenario, **(json.loads(args.scenario_params) if args.scenario_params
                                                    else {}))
        ctrl = registry.create(args.controller, json.loads(args.params) if args.params else {})
        kw = dict(uncertainty="nominal", variation="off", seeds=args.seeds)
        if not args.cold:
            kw.update(heater_code="quasi_static", settle=True)
        rr = vectors.run_recorded(cfg, ctrl, sc, duration_s=args.duration_s, **kw)
        paths = vectors.record_vectors(rr, args.out, meta={"preset": args.preset, "overrides": args.override,
                                                           "scenario_params": args.scenario_params})
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=1))
        return 0
    if args.vectors_cmd == "replay":
        vec = vectors.load_vectors(args.vectors)
        ctrl = registry.create(args.controller, json.loads(args.params) if args.params else {})
        if not hasattr(ctrl, "init_fn") or vec.spec_i64 is None:
            print("mrm vectors replay needs a CompiledController and vectors recorded from one (spec arrays)",
                  file=sys.stderr)
            return 2
        res = vectors.replay_abi(ctrl.init_fn, ctrl.step_fn, vec, spec_i64=_ctrl_spec_i64(ctrl, vec),
                                 spec_f64=vec.spec_f64)
        print(f"{vec.n_records} records x {vec.n_batch} worlds: {res.mismatches} mismatches"
              + (f"; first {res.first}" if res.first else ""))
        return 0 if res.ok else 1
    vec = vectors.load_vectors(args.vectors)
    stem = args.out or str(Path(args.vectors).with_suffix(""))
    paths = vectors.record_vectors(vec, stem)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=1))
    return 0


def _ctrl_spec_i64(ctrl, vec) -> np.ndarray:
    """The recorded spec_i64 with the user parameters of ``ctrl`` (a CompiledController) in place of the recorded
    ones: replaying recorded observations through another parameter set of the same controller."""
    si = np.array(vec.spec_i64, dtype=np.int64, copy=True)
    from mrm.control.api import SPEC_I_N_USER, SPEC_I_USER
    start, n = int(si[SPEC_I_USER]), int(si[SPEC_I_N_USER])
    user = np.asarray(ctrl.params_i64, dtype=np.int64)
    if user.size != n:
        raise ValueError(f"controller has {user.size} integer parameters, the recording {n}")
    si[start:start + n] = user
    return si


def register_cli(register: Callable) -> None:
    """``mrm golden``, ``mrm bridge``, ``mrm vectors`` (mrm.cli provider)."""
    register("golden", "check or update the golden traces (tests/golden; 07 4.9)", _golden_setup, _golden_run)
    register("bridge", "binary lockstep bridge: serve a controller, generate the C header (07 P1-21)",
             _bridge_setup, _bridge_run)
    register("vectors", "golden vectors: record, replay, hex (07 P1-21)", _vectors_setup, _vectors_run)
