"""Experiments: ``run | sweep | mc | compare | robust | acceptance | quasi_static`` from one file (07 P0-10, P1-17,
section 4.8), with run manifests (:mod:`mrm.sim.manifest`) and the rules for judging a controller.

Owner: experiments (docs/handoff/experiments.md). Normative: docs/research/07_design_critique.md P1-17 (experiment
files, run directories, ``mrm.run``), P0-10 (quasi-static mode, time compression and its mandatory convergence
test, ``--dry-run``), section 4.8 (how controllers are judged), P2-12 (CLI, in :mod:`mrm.sim.cli`).

Experiment files
----------------
An experiment file is an mrm.params config (``meta.kind: experiment``; bare physical numbers are overrides) that
``extends:`` a preset and adds the sections of 07 P1-17::

    extends: g26
    meta: {name: a1_pi, kind: experiment}
    controller: {name: pi_lock_c, params: {}}
    scenario: {name: a1, params: {phase_s: 0.06535}}
    metrics: {lock_threshold_pm: 5.0, hysteresis_pm: 1.0}
    experiment:
      kind: run                 # run | sweep | mc | compare | robust | acceptance | quasi_static
      duration_s: 0.1307
      batch: 1
      seeds: {device: 0, uncertainty: 0, noise: 0, scenario: 0}
      uncertainty: nominal      # nominal | sample | lhs (realize)
      variation: "off"          # off | sample (quoted: YAML 1.1 reads a bare off as false)
      start: warm               # extension keys (below)
      score_from_s: 0.005

``controller``, ``scenario``, ``metrics`` and the ``experiment`` keys ``kind duration_s batch n_mc seeds uncertainty
variation grid controllers`` are mrm.params schema keys (strict). The experiment-layer keys
(:data:`EXTENSION_KEYS`: start, record, corners, n_threads, self_reference, acceptance, convergence_certificate,
compression_factors, histogram, oracle, p_in_w, max_worlds_per_plant, rank_metrics, score_from_s) are mrm.params
schema keys too since the integration of 2026-10-10 (record, corners, self_reference and acceptance as open
subtrees), so ``mrm.params.load`` accepts a complete experiment file; this module reads them back from the Config,
validates them again strictly (:data:`EXTENSION_KEYS`: types, enums, criteria, record syntax) and keeps them on
:attr:`Experiment.options`. (Before, :func:`load` stripped them out of the file; it still does for a key that the
schema does not know, so the two stay compatible.) The experiment hash counts them once, as options
(:func:`config_hash_without_options`). ``--set experiment.<key>=...`` works on all of them.

Kinds
-----
- ``run``: one runner.run of ``batch`` worlds; records Truth traces (default a standard set at an automatic
  decimation). A time-compressed scenario (``scenario.time_compression`` > 1, e.g. the soak) first needs a
  convergence certificate (07 P0-10): from ``experiment.convergence_certificate`` (an earlier run directory), or
  from the convergence experiment this run performs first (one slow cycle at each of ``compression_factors``,
  default 1, 10, 100, on the same plant, controller, start and seeds; per world, every factor must agree on max
  detuning, wrong-line and FSR-hop events, heater range used and integrated errors within
  mrm.scenarios.DEFAULT_CONVERGENCE_TOLERANCES). Without a certificate that covers the compression the run is
  refused (:class:`mrm.scenarios.CompressionError`); the convergence results are still written.
- ``sweep``: the Cartesian product of ``experiment.grid`` axes, one run per point. Axis keys: a config path (values
  in config units, applied as overrides; ``controller.params.<p>`` and ``scenario.params.<p>`` included),
  ``alt:<path>`` (an alternative of a leaf, or an ``uncertainty.<entry>`` pin), ``corner`` (a corner preset name),
  ``controller`` (a controller entry).
- ``mc``: ``n_mc`` worlds (default ``batch``) of ``experiment.uncertainty`` (sample or lhs) and variation, metrics
  only (07 P0-8), in chunks of ``max_worlds_per_plant``; quantiles over worlds.
- ``compare`` (07 4.8): every controller of ``experiment.controllers`` on the nominal world, on every named corner
  and on ``n_mc`` (default 256) Latin-hypercube worlds with metrics-only recording; per controller the nominal
  result, the worst named corner and quantiles over the worlds; the Kendall-tau stability of the ranking across
  worlds for every rank metric. The first line of the report is :data:`Q1_CAVEAT`.
- ``robust``: the same for the one ``controller``, plus the acceptance criteria evaluated over the whole set (a
  claim holds only if it holds in every world, 07 4.8).
- ``acceptance``: the main case (nominal world unless ``experiment.uncertainty`` is set; ``batch`` worlds), plus the
  corners and the ``n_mc`` worlds when given; every criterion of ``experiment.acceptance`` must hold in every world
  of every case. ``experiments/acceptance/*.yaml`` hold the acceptance table.
- ``quasi_static`` (07 P0-10, BND-4): runner.quasi_static over ``experiment.grid.ambient_c`` (default 0..105 degC
  in 5 K steps) and ``experiment.grid.bias_v`` (default 1.0..3.75 V in 0.25 V steps): code-range coverage, FSR-hop
  and wrong-line risk, the oracle relOMA map and its BND-4 variation over ambient (< 1 dB asked).

Judging a controller (07 4.8)
-----------------------------
Primary metrics (per world, :func:`world_summary`): the OMA penalty against the oracle at the same instant
(worst lane and bank mean, dB), the lock-excursion count and the longest excursion (07 P0-8 hysteresis), the
wrong-line flag (rings on a foreign line at the end) and the time to lock (the last ring; inf when a ring ends
unlocked). Absolute relOMA (bank mean, dB, 07 D9 OMA/P_in) is reported. A lock is judged about the reference of
``metrics.lock_reference`` (line, oracle or target); ``experiment.self_reference`` measures it about each ring's own
steady operating point instead (a side-of-fringe lock sits 0.3-0.5 FWHM from its line and ~20 pm from the oracle's
optimum by design, so neither 'line' nor 'oracle' judges its loop dynamics).

Python entry points: :func:`load`, :func:`run` (``mrm.run`` calls it), :func:`estimate` (``--dry-run``),
:func:`rerun`. Results: :class:`ExperimentResult`.
"""
from __future__ import annotations

import atexit
import copy
import dataclasses
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mrm.sim.metrics import TR, TRUTH_FIELDS, RecordSpec

__all__ = [
    "KINDS", "STARTS", "NAMED_CORNERS", "Q1_CAVEAT", "PERF_REFERENCE", "EXTENSION_KEYS", "OptionKey", "RANK_METRICS",
    "ExperimentError", "Experiment", "Criterion", "Case", "CaseResult", "ExperimentResult", "load", "from_config",
    "run", "estimate", "rerun", "world_summary", "rank_stability", "evaluate_criteria", "parse_records",
    "default_records", "plant_fingerprint", "load_certificate", "RANKABLE", "config_hash_without_options",
]

#: First line of every controller report (07 4.8): the thermal dynamics are assumed until Lightmatter answers Q1.
Q1_CAVEAT = ("Performance is conditional on the assumed thermal dynamics (heater, crosstalk and aggressor poles) "
             "until Q1 (the heater step response at the lock point, 07 section 5) is answered.")
KINDS = ("run", "sweep", "mc", "compare", "robust", "acceptance", "quasi_static")
#: start states (see EXTENSION_KEYS['start'])
STARTS = ("cold", "cold_settled", "warm")
#: the corner presets 07 section 4.8 names (thin extends: files of mrm/params/presets that select alternatives)
NAMED_CORNERS = ("g26_slow_thermal", "g26_fast_thermal", "g26_undercut", "g26_h86", "g26_h86_sensor",
                 "g26_overcoupled", "g26_heater_loss", "g26_hot_selfheat", "g26_low_eta", "g26_fsr_window_a",
                 "g26_fsr_edge", "g26_textbook_thermal")
#: world-level metrics a compare ranks the controllers by (lower is better); 07 4.8 primary metrics
RANK_METRICS = ("oma_penalty_db_worst", "excursion_count", "excursion_max_s", "wrong_line_rings", "time_to_lock_s")
#: world_summary metrics a ranking may use (lower is better; relOMA, where higher is better, is reported only)
RANKABLE = ("oma_penalty_db_worst", "oma_penalty_db_mean", "excursion_count", "excursion_max_s", "time_unlocked_s",
            "wrong_line_rings", "wrong_line_events", "unlocked_rings", "max_abs_lock_error_pm", "rms_lock_error_pm",
            "fsr_hop_events", "ber_mean_worst", "heater_sat_s", "heater_low_rail_s", "time_to_lock_s",
            "monitor_dev_ua_max")

#: Measured engine speed used by :func:`estimate` (docs/handoff/perf.md section 8.5, 8-core M2 on AC power, warm
#: numba cache; docs/ENGINE.md section 9) and the memory model (measured for this module, g26, histogram off/on).
PERF_REFERENCE = {
    "source": "docs/handoff/perf.md section 8.5 (round 2, 2026-10-10) and docs/ENGINE.md sections 5 and 9",
    "us_per_tick_b1_fixed": 9.3,          # B = 1, fixed 1 us tick: 9.1-9.4 us
    "us_per_world_tick_parallel": 2.1,    # B = 64, 8 threads: 2.0-2.3 us per world-tick
    "auto_factor_b1_10khz": 2.95 / 9.3,   # B = 1, auto policy, 10 kHz controller: 2.9-3.0 us per tick
    "auto_factor_b1_100khz": 4.9 / 9.3,   # B = 1, auto policy, 100 kHz controller: 4.8-5.1 us per tick
    "auto_factor_parallel": 1.9 / 2.1,    # B = 64, auto: 1.9 us per world-tick
    "auto_factor_b1_100khz_ramp": 6.5 / 9.3,  # measured here: B = 1, auto, 100 kHz pi_lock_c, soak ramp (A4
                                          # convergence: 1.218e9 ticks in 7905 s, load 2-3)
    "python_us_per_edge": 300.0,          # Python controller path: ~100-300 us per edge (ENGINE.md 5; pi_lock ~0.3 ms
                                          # measured from the scorecard); controller dependent: wdm_supervisor at
                                          # B = 8 measured ~2 ms per edge (acceptance B5)
    "python_us_per_edge_per_world": 2.0,  # assumed growth of the per-edge cost with B (observation packing)
    "setup_s": 1.0,                       # realize + Plant build + settle (measured 0.6-1.7 s for B = 1..64)
    "quasi_static_s_per_point_world": 0.012,  # runner.quasi_static, measured: 144 points of g26 in 1.7 s (A7)
    "cold_compile_s": 110.0,              # empty numba cache: 104-110 s, peak RSS 2.0-2.3 GB
    "cold_compile_gb": 2.3,
    "ram_base_gb": 0.24,                  # measured: B = 1 run 0.24 GB peak RSS
    "ram_mb_per_world": 1.55,             # measured: B = 512 1.03 GB (histogram off)
    "ram_mb_per_world_histogram": 2.1,    # measured: B = 128 with the histogram 0.70 GB
}


class ExperimentError(ValueError):
    """An experiment file or option is invalid, or an experiment cannot run as asked."""


# ================================================================================================ extension schema
@dataclass(frozen=True)
class OptionKey:
    """One experiment-layer key ``experiment.<name>``: this module's strict validation of it (the key is also in
    mrm/params/schema.py since 2026-10-10, with the same default)."""
    name: str
    type: str                 # bool | int | float | str | enum | list | map
    default: Any = None
    doc: str = ""
    enum: tuple = ()


EXTENSION_KEYS: tuple[OptionKey, ...] = (
    OptionKey("start", "enum", None,
              "how the plant starts: 'cold' = heaters at code_min, reset protocol sim.reset_protocol (g26: "
              "laser_ramp), cold thermal lags (a power-up); 'cold_settled' = heaters at code_min, thermal state "
              "settled from the cold branch, lasers on at t = 0 (branch_low); 'warm' = the quasi-static oracle lock "
              "codes with the thermal state settled on the hot branch (a steady lock; runner heater_code="
              "'quasi_static', settle=True). Default: the scenario's ResetSpec (warm_start -> warm, cold_start -> "
              "cold), else cold", STARTS),
    OptionKey("record", "list", None,
              "Truth records for trace.h5: strings 'field@decimation:reducer' (decimation in ticks or 'auto'; "
              "reducer mean | min | max | last | envelope) or {channel, decimation, reducer} maps; [] records "
              "nothing. Default: kind run -> default_records(); every other kind -> nothing (metrics only)"),
    OptionKey("corners", "list", None,
              "named corners: corner preset names or {name, alts: {path: alternative}} maps. Default: compare and "
              "robust -> NAMED_CORNERS (07 4.8); other kinds -> none"),
    OptionKey("n_threads", "int", None, "kernel worker threads (default os.cpu_count())"),
    OptionKey("self_reference", "map", None,
              "{duration_s (default 0.02), window_s (default 0.005)}: a steady pre-run of the same worlds, "
              "controller and start under the static scenario with the main scenario's signal defaults (ambient, "
              "bias) gives each ring's own operating point (median detuning over the last window_s) and monitor "
              "current; the main run judges the lock about it (metrics.lock_reference = target) and adds the "
              "monitor-error metrics. Not allowed for scenarios that drive the base temperature with a component "
              "(soak, cold_start, reloma_plane)"),
    OptionKey("acceptance", "list", None,
              "criteria {id, metric, op, value, reduce, rings, cases, source, note}: metric a name of the metrics "
              "dict, 'world.<name>', 'time_above_s.<thr>', a world_summary name or a derived metric; op <= < >= > "
              "== != band band_either_sign; reduce over rings max | min | sum | mean | all | any | none (default "
              "max); cases 'all' (default) or case labels (exact, or a prefix ending in '/' or '*')"),
    OptionKey("convergence_certificate", "str", None,
              "a run directory (or its convergence.json) whose 07 P0-10 certificate covers this compressed "
              "scenario and plant; without it a compressed run performs the convergence experiment first"),
    OptionKey("compression_factors", "list", (1.0, 10.0, 100.0),
              "time-compression factors of the convergence experiment (07 P0-10: 1, 10, 100)"),
    OptionKey("histogram", "bool", False, "detuning histogram accumulators (07 P0-8; ~2.1 MB per world)"),
    OptionKey("oracle", "bool", True, "the in-kernel oracle (OMA penalty; needed by lock_reference oracle)"),
    OptionKey("p_in_w", "float", None,
              "quasi_static: power per line at the chip in W (default the plant's own; BND-4 is stated at 1 mW "
              "into the MRM)"),
    OptionKey("max_worlds_per_plant", "int", 512,
              "largest B in one Plant (~1.55 MB per world); sample-mode Monte Carlo runs in chunks with advanced "
              "seed bases (the same worlds, 07 P1-13); lhs needs every world in one Plant"),
    OptionKey("rank_metrics", "list", None, "compare: world-level metrics to rank by (default RANK_METRICS)"),
    OptionKey("score_from_s", "float", None,
              "the metrics accumulate from this time on (runner option score_from_s): a controller's start-up "
              "(pi_lock maps the peak through the resonance in its first ~3 ms) is left out of every metric of the "
              "main and convergence runs; traces still cover the whole run. Default 0 (score everything)"),
)
_EXT = {k.name: k for k in EXTENSION_KEYS}


def _is_extension(name: str) -> bool:
    """True when ``experiment.<name>`` is one of this module's keys and mrm.params does not know it yet."""
    from mrm.params import schema as sch
    path = f"experiment.{name}"
    return name in _EXT and sch.lookup(path) is None and not sch.is_open(path)


def _check_experiment_key(name: str, where: str) -> None:
    """Raise for ``experiment.<name>`` that neither mrm.params nor this module knows, with a did-you-mean over
    both key sets."""
    import difflib
    from mrm.params import schema as sch
    path = f"experiment.{name}"
    if name in _EXT or sch.lookup(path) is not None or sch.is_open(path) or any(
            k.startswith(path + ".") for k in sch.SCHEMA):
        return
    known = sorted({k.split(".")[1] for k in sch.SCHEMA if k.startswith("experiment.")} | set(_EXT))
    near = difflib.get_close_matches(name, known, n=2)
    raise ExperimentError(f"{where}: unknown key experiment.{name}" + (f" (did you mean {near[0]!r}?)" if near else
                                                                        f"; known: {known}"))


def _check_option(k: OptionKey, v: Any) -> Any:
    """Validate one extension value against its OptionKey (strict); returns the normalized value."""
    if v is None:
        return None
    t = k.type
    bad = ExperimentError(f"experiment.{k.name}: expected {t}, got {v!r}")
    if t == "bool":
        if not isinstance(v, (bool, np.bool_)):
            raise bad
        return bool(v)
    if t == "int":
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise bad
        return int(v)
    if t == "float":
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise bad
        return float(v)
    if t == "str":
        if not isinstance(v, (str, Path)):
            raise bad
        return str(v)
    if t == "enum":
        if v not in k.enum:
            raise ExperimentError(f"experiment.{k.name}: {v!r} is not one of {list(k.enum)}")
        return v
    if t == "list":
        if isinstance(v, (str, bytes)) or not isinstance(v, (list, tuple)):
            raise bad
        return list(v)
    if t == "map":
        if not isinstance(v, Mapping):
            raise bad
        return dict(v)
    raise ExperimentError(f"experiment.{k.name}: unknown option type {t!r}")  # pragma: no cover


def _set_nested(d: dict, parts: Sequence[str], v: Any) -> None:
    for p in parts[:-1]:
        d = d.setdefault(p, {})
        if not isinstance(d, dict):
            raise ExperimentError(f"cannot set {'.'.join(parts)}: {p} is not a mapping")
    d[parts[-1]] = v


# ================================================================================================ loading
_TMP_DIR: Path | None = None


def _tmpdir() -> Path:
    global _TMP_DIR
    if _TMP_DIR is None:
        _TMP_DIR = Path(tempfile.mkdtemp(prefix="mrm-exp-"))
        atexit.register(shutil.rmtree, _TMP_DIR, True)
    return _TMP_DIR


def _is_preset_file(p: Path) -> bool:
    from mrm.params.config import PRESET_DIR
    try:
        p.resolve().relative_to(PRESET_DIR.resolve())
        return True
    except ValueError:
        return False


def _strip_file(path: Path, stack: tuple = ()) -> tuple[Path, dict, list[dict]]:
    """Copy ``path`` without its extension keys (written to a temp file with every ``extends:`` parent made
    absolute; non-preset parents are stripped too). Returns (stripped path, extension options (parents first,
    the child wins), chain [{file, sha256}])."""
    import yaml
    from mrm.params.config import _resolve_file, yaml_load
    from mrm.sim.manifest import file_sha256
    path = Path(path).resolve()
    if path in stack:
        raise ExperimentError(f"extends cycle at {path}")
    text = path.read_text()
    data = yaml_load(text) or {}
    if not isinstance(data, dict):
        raise ExperimentError(f"{path}: top level must be a mapping")
    chain = [{"file": str(path), "sha256": file_sha256(path)}]
    ext: dict = {}
    exp_sec = data.get("experiment")
    if isinstance(exp_sec, dict):
        for k in list(exp_sec):
            if "." not in str(k):
                _check_experiment_key(str(k), str(path))
            if _is_extension(str(k)):
                ext[str(k)] = exp_sec.pop(k)
        if not exp_sec:
            data.pop("experiment")
    parents = data.get("extends")
    plist = [parents] if isinstance(parents, str) else list(parents or [])
    new_par: list[str] = []
    ext_par: dict = {}
    for par in plist:
        pp = _resolve_file(par, path.parent)
        if _is_preset_file(pp):
            new_par.append(str(pp))
            chain.append({"file": str(pp), "sha256": file_sha256(pp)})
        else:
            sp, pe, pc = _strip_file(pp, stack + (path,))
            ext_par.update(pe)
            new_par.append(str(sp))
            chain.extend(pc)
    if plist:
        data["extends"] = new_par if isinstance(parents, list) else new_par[0]
    meta = data.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        data["meta"] = meta
    meta.setdefault("name", path.stem)
    out_text = yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=120)
    out = _tmpdir() / f"{path.stem}.{hashlib.sha256(out_text.encode()).hexdigest()[:16]}.yaml"
    if not out.exists():
        out.write_text(out_text)
    _STRIPPED_ORIGIN[str(out.resolve())] = str(path)
    return out, {**ext_par, **ext}, chain


#: stripped temp file -> the experiment file it was made from (Config.files shows the originals)
_STRIPPED_ORIGIN: dict[str, str] = {}


def _input_copies(path: Path) -> list[dict]:
    """The files of runs/<id>/inputs/ (07 P1-17: a run is reproduced from its directory alone): the experiment file
    first, then every file of its ``extends:`` chain that is not a preset, as ``_extends_<i>_<name>``. Each copy's
    ``extends:`` names the copies of its non-preset parents (a preset given by path is named by its preset name), so
    the run directory reruns, and can be sent elsewhere, without the original files. A file whose parents are all
    preset names is copied byte for byte. Returns [{name, text, original, sha256 (of the original's bytes)}]. Taken
    when the experiment is loaded (an edit of the file during a long run must not change what the run records)."""
    import yaml
    from mrm.params.config import PRESET_DIR, _resolve_file, yaml_load
    out: list[dict | None] = []

    def visit(p: Path, name: str, stack: tuple) -> None:
        if p in stack:
            raise ExperimentError(f"extends cycle at {p}")
        slot = len(out)
        out.append(None)
        raw = p.read_bytes()
        text = raw.decode()
        data = yaml_load(text) or {}
        parents = data.get("extends") if isinstance(data, dict) else None
        plist = [parents] if isinstance(parents, str) else list(parents or [])
        new, changed = [], False
        for par in plist:
            pp = _resolve_file(par, p.parent).resolve()
            if _is_preset_file(pp):
                if str(par).endswith((".yaml", ".yml")):
                    new.append(str(pp.relative_to(PRESET_DIR.resolve()).with_suffix("")))
                    changed = True
                else:
                    new.append(par)
            else:
                pname = f"_extends_{len(out)}_{pp.name}"
                visit(pp, pname, stack + (p,))
                new.append(pname)
                changed = True
        if changed:
            data["extends"] = new if isinstance(parents, list) else new[0]
            text = ("# copied by mrm.sim.experiment: extends: names the copies in this folder (or preset names)\n"
                    + yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=120))
        out[slot] = {"name": name, "text": text, "original": str(p), "sha256": hashlib.sha256(raw).hexdigest()}

    p0 = Path(path).resolve()
    visit(p0, p0.name, ())
    return out  # type: ignore[return-value]


def _enum_word(path: str, v: Any) -> Any:
    """YAML 1.1 reads off / on / no / yes as booleans; for an enum setting whose words include exactly one of them,
    the word is meant (``--set experiment.variation=off``)."""
    from mrm.params import schema as sch
    key = sch.lookup(path)
    if isinstance(v, bool) and key is not None and key.type == "enum" and v not in key.enum:
        words = [w for w in key.enum if isinstance(w, str) and w in (("on", "yes", "true") if v else
                                                                        ("off", "no", "false"))]
        if len(words) == 1:
            return words[0]
    return v


def _param_overrides(items: Sequence[str]) -> dict:
    """'path=value' strings -> {path: value} for mrm.params.load (values parsed once, enum words kept)."""
    from mrm.params.config import parse_overrides
    return {p: _enum_word(p, v) for p, v in parse_overrides(list(items)).items()}


def _ov_str(path: str, v: Any) -> str:
    """'path=value' with the value as YAML flow text (JSON for numbers, lists and maps; strings as they are)."""
    if isinstance(v, str):
        return f"{path}={v}"
    from mrm.sim.manifest import to_jsonable
    return f"{path}={json.dumps(to_jsonable(v))}"


def _route_overrides(overrides) -> tuple[list[str], dict]:
    """Split overrides into mrm.params overrides ("path=value" strings) and extension options (nested dict)."""
    from mrm.params.config import parse_overrides
    po = parse_overrides(overrides)
    plain: list[str] = []
    ext: dict = {}
    for p, v in po.items():
        parts = p.split(".")
        v = _enum_word(p, v)
        if len(parts) >= 2 and parts[0] == "experiment":
            _check_experiment_key(parts[1], f"override {p!r}")
            if len(parts) == 2 and parts[1] in _EXT:
                _check_option(_EXT[parts[1]], v)     # this module's strict check and messages, in both states
        if len(parts) >= 2 and parts[0] == "experiment" and _is_extension(parts[1]):
            _set_nested(ext, parts[1:], v)
        else:
            plain.append(_ov_str(p, v))
    return plain, ext


def _plain_override_list(overrides) -> list[str]:
    """Normalize overrides to a list of "path=value" strings (the form manifests store)."""
    from mrm.params.config import parse_overrides
    return [_ov_str(p, v) for p, v in parse_overrides(overrides).items()]


@dataclass
class Experiment:
    """A loaded experiment: the validated Config, the extension options and where they came from.

    ``source``: {path (the experiment file or None), stripped (the temp file mrm.params loaded), overrides (list of
    "path=value"), alts ({path: name}), chain ([{file, sha256}] of the file and its parents)}.
    """
    cfg: Any
    options: dict
    source: dict
    _cache: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------------------------------- config access
    def config(self, overrides: Sequence[str] = (), alts: Mapping[str, str] | None = None):
        """The experiment's Config with extra overrides and alternatives (a corner, a sweep point, a controller
        entry's bindings); cached."""
        if not overrides and not alts:
            return self.cfg
        key = (tuple(overrides), tuple(sorted((alts or {}).items())))
        if key not in self._cache:
            from mrm import params
            src = self.source.get("stripped")
            base_ov = list(self.source.get("param_overrides") or [])
            base_alts = dict(self.source.get("alts") or {})
            base_alts.update(alts or {})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                if src is None:
                    raise ExperimentError("this experiment has no source file; corners and sweeps need one")
                self._cache[key] = params.load(src, overrides=_param_overrides(base_ov + list(overrides)),
                                               alts=base_alts)
        return self._cache[key]

    def option(self, name: str) -> Any:
        """Resolved extension option (the given value or its default)."""
        v = self.options.get(name)
        return _EXT[name].default if v is None else v

    # ---------------------------------------------------------------------------------------- core settings
    def get(self, key: str, default: Any = None) -> Any:
        """A setting of the experiment's config (mrm.params Config.get)."""
        return self.cfg.get(key, default)

    @property
    def kind(self) -> str:
        return str(self.get("experiment.kind", "run"))

    @property
    def name(self) -> str:
        return str(self.cfg.name)

    @property
    def seeds(self) -> dict:
        out = {}
        for k in ("device", "uncertainty", "noise", "scenario"):
            out[k] = int(self.get(f"experiment.seeds.{k}", 0) or 0)
        return out

    @property
    def batch(self) -> int:
        return int(self.get("experiment.batch", 1) or 1)

    @property
    def n_mc(self) -> int | None:
        v = self.get("experiment.n_mc", None)
        return None if v is None else int(v)

    @property
    def duration_s(self) -> float | None:
        v = self.get("experiment.duration_s", None)
        return None if v is None else float(v)

    def given(self, key: str) -> bool:
        """True when the config sets ``key`` explicitly (not the schema default)."""
        return bool(self.cfg.has(key))

    @property
    def grid(self) -> dict:
        """experiment.grid with --set experiment.grid.<path>=... overrides merged in (they win)."""
        g = self.cfg.open_values.get("experiment.grid")
        out = dict(g) if isinstance(g, Mapping) else {}
        for k, v in self.cfg.open_values.items():
            if k.startswith("experiment.grid."):
                out[k[len("experiment.grid."):]] = v
        return out

    @property
    def n_threads(self) -> int:
        v = self.option("n_threads")
        return int(v) if v else int(os.cpu_count() or 1)

    @property
    def hash(self) -> str:
        """sha256 of the resolved config and the resolved extension options (the run directory's hash8).

        The config part is :func:`config_hash_without_options`: the extension keys are hashed once, as resolved
        options, whether mrm.params validated them (they are in mrm/params/schema.py since 2026-10-10) or this
        module stripped them (before); so the hash of an experiment did not change when the keys moved into the
        schema, and recorded run directories still rerun."""
        opts = {k.name: self.option(k.name) for k in EXTENSION_KEYS}
        from mrm.sim.manifest import to_jsonable
        blob = json.dumps({"config": config_hash_without_options(self.cfg), "options": to_jsonable(opts)},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def resolved_options(self) -> dict:
        """Every extension option with its default filled in, and whether it was given."""
        return {k.name: {"value": self.option(k.name), "given": self.options.get(k.name) is not None}
                for k in EXTENSION_KEYS}


def load(source, overrides=None, alts=None) -> Experiment:
    """Load an experiment from a YAML path (or preset name), a Config, or an Experiment (returned with the extra
    overrides/alts applied).

    ``overrides``: {path: value} or ["path=value", ...]; ``experiment.<extension>`` paths set extension options,
    the rest are mrm.params overrides. ``alts``: {path: alternative} as for mrm.params.load.
    """
    from mrm import params
    from mrm.params.config import Config, _resolve_file
    if isinstance(source, Experiment):
        if not overrides and not alts:
            return source
        if source.source.get("path") is None:
            raise ExperimentError("cannot re-load an experiment without a source file")
        ov = list(source.source.get("overrides") or []) + _plain_override_list(overrides)
        al = dict(source.source.get("alts") or {})
        al.update(dict(alts or {}))
        return load(source.source["path"], ov, al)
    if isinstance(source, Config):
        if overrides or alts:
            raise ExperimentError("pass a file path to apply overrides; a Config is already resolved")
        return from_config(source)
    if isinstance(source, Mapping):
        raise ExperimentError("an experiment needs a file (extends: and run reproduction); write the mapping to a "
                              "YAML file")
    path = _resolve_file(source)
    stripped, ext, chain = _strip_file(path)
    plain, ext_ov = _route_overrides(overrides)
    alts_d = {}
    if alts:
        from mrm.params.config import _parse_alts
        alts_d = _parse_alts(alts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        cfg = params.load(stripped, overrides=_param_overrides(plain), alts=alts_d)
    cfg.files = [_STRIPPED_ORIGIN.get(str(Path(f).resolve()), f) if Path(f).is_absolute() else f
                 for f in cfg.files]
    opts = _options_from_cfg(cfg)
    for k, v in ext.items():
        opts[k] = v
    for k, v in ext_ov.items():
        if isinstance(v, dict) and isinstance(opts.get(k), dict):
            merged = dict(opts[k])
            merged.update(v)
            opts[k] = merged
        else:
            opts[k] = v
    options = _validate_options(opts)
    return Experiment(cfg=cfg, options=options,
                      source={"path": str(path), "stripped": str(stripped), "overrides": _plain_override_list(
                          overrides), "param_overrides": plain, "alts": dict(alts_d), "chain": chain,
                          "input_copies": _input_copies(path)})


def from_config(cfg, options: Mapping | None = None) -> Experiment:
    """An Experiment from a loaded Config (``mrm.run(mrm.params.load('exp.yaml'))``; extension options from
    ``options`` or their defaults). Corners and sweeps reload the config's last file with its applied overrides
    and alternatives."""
    opts = _validate_options({**_options_from_cfg(cfg), **dict(options or {})})
    src: dict = {"path": None, "stripped": None, "overrides": [], "param_overrides": [],
                 "alts": dict(getattr(cfg, "alts_applied", {}) or {}), "chain": []}
    files = [f for f in getattr(cfg, "files", []) if f and not f.startswith("<")]
    if files:
        from mrm.params.config import _resolve_file
        try:
            last = _resolve_file(files[-1])
        except Exception:                                       # noqa: BLE001
            last = None
        if last is not None and last.exists():
            ov = _plain_override_list(getattr(cfg, "overrides_applied", {}) or {})
            stripped, ext, chain = _strip_file(last)
            if ext:
                raise ExperimentError(f"{last} uses experiment extension keys; load it with "
                                      f"mrm.sim.experiment.load (or mrm.run(path))")
            src.update(path=str(last), stripped=str(stripped), overrides=ov, param_overrides=ov, chain=chain,
                       input_copies=_input_copies(last))
    return Experiment(cfg=cfg, options=opts, source=src)


def _deep_merge(base: dict, over: Mapping) -> dict:
    """``over`` merged into a copy of ``base`` (nested mappings merged, other values replaced)."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def config_hash_without_options(cfg) -> str:
    """sha256 of ``cfg``'s resolved tree without the experiment-layer keys of :data:`EXTENSION_KEYS` (the same
    serialization as mrm.params Config.hash). It equals the Config hash of the same file with those keys stripped,
    which is how experiments were loaded before the keys entered mrm/params/schema.py."""
    tree = copy.deepcopy(cfg.resolved_tree())
    sec = tree.get("experiment")
    if isinstance(sec, dict):
        for k in EXTENSION_KEYS:
            sec.pop(k.name, None)
        if not sec:
            tree.pop("experiment")
    return hashlib.sha256(json.dumps(tree, sort_keys=True, default=str).encode()).hexdigest()


def _options_from_cfg(cfg) -> dict:
    """Extension options that mrm.params validated (they are mrm/params/schema.py keys since 2026-10-10, so they stay
    in the Config instead of being stripped): read back from the Config so they are never ignored. A map option
    (self_reference) set in the file and refined by dotted overrides (``--set experiment.self_reference.window_s=...``,
    stored flattened by mrm.params) gets both, the overrides winning."""
    from mrm.params import schema as sch
    out = {}
    open_values = getattr(cfg, "open_values", {}) or {}
    for k in EXTENSION_KEYS:
        path = f"experiment.{k.name}"
        if sch.lookup(path) is None and not sch.is_open(path):
            continue
        base = cfg.get(path) if cfg.has(path) else None
        sub = {kk[len(path) + 1:]: v for kk, v in open_values.items() if kk.startswith(path + ".")}
        if sub:
            from mrm.scenarios.registry import unflatten
            over = unflatten(sub)
            if base is None:
                base = over
            elif isinstance(base, Mapping):
                base = _deep_merge(dict(base), over)
            else:
                raise ExperimentError(f"experiment.{k.name}: dotted overrides {sorted(sub)} need a mapping, the "
                                      f"config has {base!r}")
        if base is None:
            continue
        if k.type == "int" and isinstance(base, float) and base.is_integer():
            base = int(base)                    # mrm.params accepts 4.0 for an int setting
        out[k.name] = base
    return out


def _validate_options(opts: Mapping) -> dict:
    out = {}
    for k, v in opts.items():
        if k not in _EXT:
            import difflib
            near = difflib.get_close_matches(str(k), list(_EXT), n=1)
            raise ExperimentError(f"unknown experiment key experiment.{k}" + (f" (did you mean {near[0]!r}?)"
                                                                              if near else ""))
        out[k] = _check_option(_EXT[k], v)
    if out.get("self_reference") is not None:
        sr = out["self_reference"]
        bad = set(sr) - {"duration_s", "window_s"}
        if bad:
            raise ExperimentError(f"experiment.self_reference: unknown keys {sorted(bad)} (duration_s, window_s)")
        vals = {"duration_s": 0.02, "window_s": 0.005}
        for k in ("duration_s", "window_s"):
            v = sr.get(k)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not v > 0:
                raise ExperimentError(f"experiment.self_reference.{k}: a positive number of seconds, got {v!r}")
            vals[k] = float(v)
        if vals["window_s"] > vals["duration_s"]:
            raise ExperimentError(f"experiment.self_reference: window_s {vals['window_s']} s exceeds the pre-run "
                                  f"duration_s {vals['duration_s']} s")
    if out.get("record") is not None:
        parse_records(out["record"], n_ticks=1000, period_ticks=10)   # syntax check only
    if out.get("acceptance") is not None:
        _criteria(out["acceptance"])
    if out.get("rank_metrics") is not None:
        bad = [str(k) for k in out["rank_metrics"] if k not in RANKABLE]
        if bad or not out["rank_metrics"]:
            raise ExperimentError(f"experiment.rank_metrics: {bad or 'empty'}; a ranking uses world metrics where "
                                  f"lower is better: {list(RANKABLE)}")
    if out.get("compression_factors") is not None:
        cf = [float(x) for x in out["compression_factors"]]
        if not cf or min(cf) < 1.0:
            raise ExperimentError("experiment.compression_factors must be >= 1")
        out["compression_factors"] = cf
    return out


# ================================================================================================ records
_STANDARD_RECORDS = (("detuning_m", "mean"), ("detuning_m", "min"), ("detuning_m", "max"), ("reloma", "mean"),
                     ("oracle_reloma", "mean"), ("heater_code", "last"), ("monitor_code", "last"),
                     ("i_monitor_a", "mean"), ("t_ring_k", "mean"), ("line_on", "last"))


def _auto_decimation(n_ticks: int, period_ticks: int, max_samples: int = 10_000) -> int:
    """Smallest multiple of the controller period that keeps a record at <= max_samples windows."""
    p = max(int(period_ticks), 1)
    return int(max(p, p * math.ceil(max(n_ticks, 1) / (max_samples * p))))


def default_records(n_ticks: int, period_ticks: int) -> list[RecordSpec]:
    """The standard record of a ``run``: detuning (mean and min/max envelope, 07 P0-8), relOMA, oracle relOMA,
    heater and monitor codes, monitor current, ring temperature and the line each ring sits on, at the smallest
    multiple of the controller period that keeps <= 10 000 windows."""
    dec = _auto_decimation(n_ticks, period_ticks)
    return [RecordSpec(ch, dec, red) for ch, red in _STANDARD_RECORDS]


def parse_records(spec, *, n_ticks: int, period_ticks: int) -> list[RecordSpec]:
    """experiment.record entries -> RecordSpec list ('field@dec:reducer', dec 'auto', reducer 'envelope' = min and
    max)."""
    out: list[RecordSpec] = []
    for r in spec or []:
        if isinstance(r, Mapping):
            ch, dec, red = r.get("channel"), r.get("decimation", "auto"), r.get("reducer", "mean")
        elif isinstance(r, str):
            ch, _, rest = r.partition("@")
            dec, _, red = rest.partition(":") if rest else ("auto", "", "mean")
            red = red or "mean"
            dec = dec or "auto"
        else:
            raise ExperimentError(f"experiment.record entry {r!r}: a 'field@dec:reducer' string or a mapping")
        if ch not in TR:
            raise ExperimentError(f"experiment.record: {ch!r} is not a Truth field; known: {list(TRUTH_FIELDS)}")
        d = _auto_decimation(n_ticks, period_ticks) if dec in ("auto", None) else int(dec)
        if d < 1:
            raise ExperimentError(f"experiment.record: decimation {dec!r} must be >= 1 tick")
        reds = ("min", "max") if red == "envelope" else (red,)
        for rr in reds:
            if rr not in ("mean", "min", "max", "last"):
                raise ExperimentError(f"experiment.record: reducer {red!r} (mean | min | max | last | envelope)")
            rs = RecordSpec(ch, d, rr)
            if rs.key not in {x.key for x in out}:
                out.append(rs)
    return out


# ================================================================================================ criteria
_OPS = ("<=", "<", ">=", ">", "==", "!=", "band", "band_either_sign")
_REDUCE = ("max", "min", "sum", "mean", "all", "any", "none")


@dataclass(frozen=True)
class Criterion:
    """One acceptance criterion: ``reduce(metric[rings])`` per world, then ``op value`` in every world of every
    case (``cases``: 'all' or a list of case-label prefixes). band ops take value [lo, hi] and need the metrics
    <metric>_min and <metric>_max; band_either_sign also accepts the mirrored band [-hi, -lo] (an unpublished sign
    convention, BND-3a)."""
    id: str
    metric: str
    op: str
    value: Any
    reduce: str = "max"
    rings: tuple | None = None
    cases: Any = "all"
    source: str = ""
    note: str = ""

    def __post_init__(self):
        if self.op not in _OPS:
            raise ExperimentError(f"criterion {self.id}: op {self.op!r} not in {_OPS}")
        if self.reduce not in _REDUCE:
            raise ExperimentError(f"criterion {self.id}: reduce {self.reduce!r} not in {_REDUCE}")
        if self.op.startswith("band") and (not isinstance(self.value, (list, tuple)) or len(self.value) != 2):
            raise ExperimentError(f"criterion {self.id}: op {self.op} needs value [lo, hi]")


def _criteria(spec) -> list[Criterion]:
    out = []
    for i, c in enumerate(spec or []):
        if not isinstance(c, Mapping):
            raise ExperimentError(f"experiment.acceptance[{i}] must be a mapping")
        bad = set(c) - {"id", "metric", "op", "value", "reduce", "rings", "cases", "source", "note"}
        if bad:
            raise ExperimentError(f"experiment.acceptance[{i}]: unknown keys {sorted(bad)}")
        for need in ("metric", "op", "value"):
            if need not in c:
                raise ExperimentError(f"experiment.acceptance[{i}]: missing {need!r}")
        rings = c.get("rings")
        out.append(Criterion(id=str(c.get("id", f"C{i + 1}")), metric=str(c["metric"]), op=str(c["op"]),
                             value=c["value"], reduce=str(c.get("reduce", "max")),
                             rings=None if rings is None else tuple(int(r) for r in rings),
                             cases=c.get("cases", "all"), source=str(c.get("source", "")),
                             note=str(c.get("note", ""))))
    ids = [c.id for c in out]
    if len(set(ids)) != len(ids):
        raise ExperimentError(f"experiment.acceptance: duplicate ids {ids}")
    return out


def _metric_array(m: Mapping, name: str):
    """A metric by name: a key of the metrics dict, a dotted path into it, or a world_summary name."""
    if name in m:
        return m[name]
    cur: Any = m
    for p in name.split("."):
        if isinstance(cur, Mapping) and p in cur:
            cur = cur[p]
        else:
            cur = None
            break
    if cur is not None:
        return cur
    try:
        ws = world_summary(m)
    except KeyError:                       # not an engine metrics dict (e.g. a quasi-static survey)
        ws = {}
    if name in ws:
        return ws[name]
    raise ExperimentError(f"unknown metric {name!r}")


def _reduce_rings(a, c: Criterion) -> np.ndarray:
    x = np.asarray(a)
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim == 1:
        return x
    if c.rings is not None:
        x = x[:, list(c.rings)]
    if c.reduce == "none":
        return x
    if c.reduce == "max":
        return np.max(x, axis=1)
    if c.reduce == "min":
        return np.min(x, axis=1)
    if c.reduce == "sum":
        return np.sum(x, axis=1)
    if c.reduce == "mean":
        return np.mean(x, axis=1)
    if c.reduce == "all":
        return np.all(x, axis=1)
    return np.any(x, axis=1)


def _holds(v: np.ndarray, op: str, ref) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    with np.errstate(invalid="ignore"):
        if op == "<=":
            return v <= ref
        if op == "<":
            return v < ref
        if op == ">=":
            return v >= ref
        if op == ">":
            return v > ref
        if op == "==":
            return v == ref
        return v != ref


def _case_selected(c: Criterion, label: str) -> bool:
    """``cases``: 'all' (default), or labels: an exact case label, or a prefix ending in '/' or '*'
    (e.g. 'pi_lock_c/' or 'quasi_static/corner/*')."""
    if c.cases == "all" or c.cases is None:
        return True
    sel = [c.cases] if isinstance(c.cases, str) else list(c.cases)
    for s in sel:
        s = str(s)
        if s.endswith("*"):
            if label.startswith(s[:-1]):
                return True
        elif s.endswith("/"):
            if label.startswith(s):
                return True
        elif label == s:
            return True
    return False


def evaluate_criteria(criteria: Sequence[Criterion], cases: Mapping[str, Mapping]) -> list[dict]:
    """Evaluate every criterion on every selected case (metrics dicts, with derived metrics merged in). A criterion
    passes when it holds in every world of every selected case; NaN never holds (e.g. a time to lock of a ring that
    ends unlocked). Returns one dict per criterion: id, metric, op, value, pass, worst (the value furthest on the
    wrong side, or the extreme), per_case {label: {values, pass}}."""
    out = []
    for c in criteria:
        per = {}
        allv = []
        ok_all = True
        for label, m in cases.items():
            if not _case_selected(c, label):
                continue
            try:
                if c.op.startswith("band"):
                    lo_v = _reduce_rings(_metric_array(m, c.metric + "_min"), dataclasses.replace(c, reduce="min"))
                    hi_v = _reduce_rings(_metric_array(m, c.metric + "_max"), dataclasses.replace(c, reduce="max"))
                    lo, hi = float(c.value[0]), float(c.value[1])
                    with np.errstate(invalid="ignore"):
                        ok = (lo_v >= lo) & (hi_v <= hi)
                        if c.op == "band_either_sign":
                            ok = ok | ((lo_v >= -hi) & (hi_v <= -lo))
                    vals = np.stack([lo_v, hi_v], axis=-1)
                else:
                    vals = _reduce_rings(_metric_array(m, c.metric), c)
                    ok = _holds(vals, c.op, float(c.value) if not isinstance(c.value, bool) else c.value)
            except ExperimentError as e:
                per[label] = {"error": str(e), "pass": False}
                ok_all = False
                continue
            p = bool(np.all(ok))
            per[label] = {"values": np.asarray(vals).tolist(), "pass": p}
            ok_all = ok_all and p
            allv.append(np.asarray(vals, float).reshape(-1))
        worst = None
        if allv:
            v = np.concatenate(allv)
            if v.size:
                if c.op in ("<=", "<"):
                    worst = float(np.nanmax(v)) if np.any(np.isfinite(v)) else float("nan")
                elif c.op in (">=", ">"):
                    worst = float(np.nanmin(v)) if np.any(np.isfinite(v)) else float("nan")
                else:
                    worst = [float(np.nanmin(v)), float(np.nanmax(v))] if np.any(np.isfinite(v)) else None
                if np.any(np.isnan(v)):
                    worst = float("nan") if not isinstance(worst, list) else worst
        out.append({"id": c.id, "metric": c.metric, "op": c.op, "value": c.value, "reduce": c.reduce,
                    "rings": None if c.rings is None else list(c.rings), "pass": ok_all and bool(per),
                    "worst": worst, "per_case": per, "source": c.source, "note": c.note})
    return out


# ================================================================================================ summaries
def world_summary(m: Mapping) -> dict[str, np.ndarray]:
    """World-level (B,) values of the 07 4.8 primary metrics and the reported ones, from one summarize() dict.

    oma_penalty_db_worst / _mean (dB, worst lane / mean over rings of the OMA penalty against the oracle at the same
    instant); excursion_count (sum over rings); excursion_max_s (longest excursion of any ring); time_unlocked_s
    (sum over rings); wrong_line_rings (rings on a foreign line at the end) and wrong_line_events (sum);
    time_to_lock_s (the last ring's; inf when any ring ends unlocked); unlocked_rings; max_abs_lock_error_pm and
    rms_lock_error_pm (worst ring); mean_reloma_db and oracle_reloma_db (bank means, dB; absolute relOMA = OMA/P_in,
    07 D9); fsr_hop_events (sum); ber_mean_worst; heater_sat_s / heater_low_rail_s (worst ring, s);
    heater_range_used_codes (largest ring range). From the warm start, when the run had one:
    warm_start_no_code_rings (rings that started at code_min), warm_start_order_hopped_rings (rings locked one order
    bluer than the heater-up order) and warm_start_supply_limited (1 when the world's lock powers exceed the array
    heater supply limit at the start, so the bank cannot hold every lock)."""
    def a(k):
        return np.asarray(m[k])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pen = a("oma_penalty_db").astype(float)
        out = {
            "oma_penalty_db_worst": np.nanmax(np.where(np.isnan(pen), -np.inf, pen), axis=1),
            "oma_penalty_db_mean": np.nanmean(pen, axis=1),
            "excursion_count": a("excursion_count").sum(axis=1).astype(float),
            "excursion_max_s": a("excursion_max_s").max(axis=1).astype(float),
            "time_unlocked_s": a("time_unlocked_s").sum(axis=1).astype(float),
            "wrong_line_rings": a("wrong_line_final").sum(axis=1).astype(float),
            "wrong_line_events": a("wrong_line_events").sum(axis=1).astype(float),
            "unlocked_rings": (~a("locked_final").astype(bool)).sum(axis=1).astype(float),
            "max_abs_lock_error_pm": a("max_abs_lock_error_pm").max(axis=1).astype(float),
            "rms_lock_error_pm": a("rms_lock_error_pm").max(axis=1).astype(float),
            "mean_reloma_db": np.nanmean(a("mean_reloma_db").astype(float), axis=1),
            "oracle_reloma_db": np.nanmean(a("oracle_reloma_db").astype(float), axis=1),
            "fsr_hop_events": a("fsr_hop_events").sum(axis=1).astype(float),
            "ber_mean_worst": np.nanmax(np.where(np.isnan(a("ber_mean").astype(float)), -np.inf,
                                                 a("ber_mean").astype(float)), axis=1),
            "heater_sat_s": a("heater_sat_s").max(axis=1).astype(float),
            "heater_low_rail_s": a("heater_low_rail_s").max(axis=1).astype(float),
            "heater_range_used_codes": a("heater_range_used_codes").max(axis=1).astype(float),
        }
        ttl = a("time_to_lock_s").astype(float)
        out["time_to_lock_s"] = np.where(np.any(np.isnan(ttl), axis=1), np.inf, np.max(np.nan_to_num(ttl), axis=1))
    out["oma_penalty_db_worst"] = np.where(np.isneginf(out["oma_penalty_db_worst"]), np.nan,
                                           out["oma_penalty_db_worst"])
    out["ber_mean_worst"] = np.where(np.isneginf(out["ber_mean_worst"]), np.nan, out["ber_mean_worst"])
    if "warm_start_no_code" in m:
        out["warm_start_no_code_rings"] = np.asarray(m["warm_start_no_code"], bool).sum(axis=1).astype(float)
    if "warm_start_supply_limited" in m:
        out["warm_start_supply_limited"] = np.asarray(m["warm_start_supply_limited"], bool).astype(float)
    if "warm_start_order_hopped" in m:
        out["warm_start_order_hopped_rings"] = np.asarray(m["warm_start_order_hopped"], bool).sum(axis=1).astype(
            float)
    for k in ("monitor_error_pct_min", "monitor_error_pct_max", "monitor_dev_ua_max"):
        if k in m:
            x = np.asarray(m[k], float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)        # a world with no reference: NaN
                out[k] = np.nanmin(x, axis=1) if k.endswith("_min") else np.nanmax(x, axis=1)
    return out


def _quantiles(x: np.ndarray, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    x = np.asarray(x, float)
    fin = x[~np.isnan(x)]
    d = {"n": int(x.size), "nan": int(np.isnan(x).sum())}
    if fin.size:
        for q in qs:                       # sample quantiles (no interpolation: +inf values stay +inf)
            d[f"p{int(round(q * 100))}"] = float(np.quantile(fin, q, method="inverted_cdf"))
        d["min"] = float(fin.min())
        d["max"] = float(fin.max())
        d["mean"] = float(fin[np.isfinite(fin)].mean()) if np.any(np.isfinite(fin)) else float("nan")
    return d


def rank_stability(values: Mapping[str, np.ndarray], nominal: Mapping[str, float]) -> dict:
    """Kendall-tau stability of a ranking of controllers across worlds (07 4.8).

    ``values``: {controller: (W,) metric per world (lower is better)}; ``nominal``: {controller: value on the
    nominal world}. For every world w, tau_b(w) = Kendall's tau-b between the world's values and the nominal values
    (scipy.stats.kendalltau; NaN counts as worst, ties allowed). Returns {tau_mean, tau_min, tau_median,
    frac_same_order (tau == 1), frac_defined, n_worlds, nominal_order, median_order, tau_median_order (tau between
    the order of the per-controller medians and the nominal order)}. With fewer than two controllers or a nominal
    with all ties, tau is undefined (NaN)."""
    from scipy.stats import kendalltau
    names = list(values)
    if len(names) < 2:
        return {"n_worlds": 0, "note": "fewer than two controllers"}
    V = np.stack([np.asarray(values[n], float) for n in names])            # (C, W)
    V = np.where(np.isnan(V), np.inf, V)
    nom = np.array([nominal.get(n, np.nan) for n in names], float)
    nom = np.where(np.isnan(nom), np.inf, nom)
    taus = []
    for w in range(V.shape[1]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = kendalltau(_finite_rank(V[:, w]), _finite_rank(nom)).statistic
        taus.append(t)
    taus = np.asarray(taus, float)
    med = np.median(np.where(np.isinf(V), 1e300, V), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tm = kendalltau(_finite_rank(med), _finite_rank(nom)).statistic
    d = np.isfinite(taus)
    return {"n_worlds": int(V.shape[1]), "frac_defined": float(d.mean()) if taus.size else float("nan"),
            "tau_mean": float(np.mean(taus[d])) if d.any() else float("nan"),
            "tau_median": float(np.median(taus[d])) if d.any() else float("nan"),
            "tau_min": float(np.min(taus[d])) if d.any() else float("nan"),
            "frac_same_order": float(np.mean(taus[d] == 1.0)) if d.any() else float("nan"),
            "nominal_order": [names[i] for i in np.argsort(nom, kind="stable")],
            "median_order": [names[i] for i in np.argsort(med, kind="stable")],
            "tau_median_order": float(tm)}


def _finite_rank(x: np.ndarray) -> np.ndarray:
    """Ranks with +inf (worst) kept as the largest, so kendalltau sees ties among infinities as ties."""
    from scipy.stats import rankdata
    return rankdata(np.where(np.isinf(x), np.finfo(float).max, x), method="average")


# ================================================================================================ cases
@dataclass
class Case:
    """One runner call of an experiment (also what :func:`estimate` costs)."""
    label: str
    controller: dict | None = None            # {name, params, overrides, alts, start, label}
    overrides: tuple = ()
    alts: dict = field(default_factory=dict)
    batch: int = 1
    seeds: dict | None = None
    uncertainty: str = "nominal"
    variation: str = "off"
    scenario: Any = None                      # None: the config's scenario section; or a Scenario
    duration_s: float = 0.0
    record: list | None = None                # RecordSpec list or None (resolved at execution when 'default')
    record_default: bool = False
    start: str = "cold"
    self_reference: dict | None = None
    group: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class CaseResult:
    """Outcome of one case: the summarize() metrics (plus derived metrics), traces, final Truth fields, the
    runner's manifest inputs, warnings and wall time."""
    label: str
    metrics: dict
    trace: dict | None
    truth_final: dict
    manifest_inputs: dict
    wall_s: float
    warnings: list
    reference: dict | None = None
    group: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """What :func:`run` returns. ``summary``: the kind-specific reduction (also in metrics.json); ``report``: the
    text report; ``passed``: acceptance / robust criteria outcome (None when the kind has none); ``status``: 'ok',
    'failed' (criteria) or 'refused' (an unvalidated compression); ``run_dir``: the run directory (None when not
    written)."""
    kind: str
    name: str
    experiment_hash: str
    cases: dict
    summary: dict
    report: str
    passed: bool | None
    status: str
    run_dir: Path | None
    wall_s: float
    manifest: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def metrics(self) -> dict:
        """The metrics of the main case (kind run) or of the only case; else {label: metrics}."""
        if "main" in self.cases:
            return self.cases["main"].metrics
        if len(self.cases) == 1:
            return next(iter(self.cases.values())).metrics
        return {k: v.metrics for k, v in self.cases.items()}


def _controller_entry(e) -> dict:
    if isinstance(e, str):
        return {"name": e, "params": {}, "overrides": [], "alts": {}, "start": None, "label": e}
    if not isinstance(e, Mapping) or "name" not in e:
        raise ExperimentError(f"controller entry {e!r}: a name or {{name, params, overrides, alts, start, label}}")
    bad = set(e) - {"name", "params", "overrides", "alts", "start", "label"}
    if bad:
        raise ExperimentError(f"controller entry {e.get('name')}: unknown keys {sorted(bad)}")
    st = e.get("start")
    if st is not None and st not in STARTS:
        raise ExperimentError(f"controller entry {e['name']}: start {st!r} not in {STARTS}")
    ov = e.get("overrides") or []
    if isinstance(ov, Mapping):
        ov = [f"{k}={v}" for k, v in ov.items()]
    return {"name": str(e["name"]), "params": dict(e.get("params") or {}), "overrides": [str(x) for x in ov],
            "alts": dict(e.get("alts") or {}), "start": st, "label": str(e.get("label", e["name"]))}


def _controller_from_cfg(cfg) -> dict | None:
    """The ``controller`` section of a Config as an entry (``from_config``: rebuilt from each case's own Config, so a
    sweep over ``controller.params.<p>`` or ``controller.name`` reaches the controller)."""
    name = cfg.get("controller.name")
    if not name:
        return None
    from mrm.scenarios.registry import unflatten
    p = cfg.open_values.get("controller.params")
    prm = dict(p) if isinstance(p, Mapping) else {}
    flat = {k[len("controller.params."):]: v for k, v in cfg.open_values.items()
            if k.startswith("controller.params.")}
    prm.update(unflatten(flat))
    return {"name": str(name), "params": prm, "overrides": [], "alts": {}, "start": None, "label": str(name),
            "from_config": True}


def _main_controller(exp: Experiment) -> dict | None:
    return _controller_from_cfg(exp.cfg)


def _corners(exp: Experiment, default: Sequence[str]) -> list[tuple[str, dict]]:
    """[(name, alts)] of experiment.corners (preset names or {name, alts}), or the defaults."""
    import yaml
    from mrm.params.config import _resolve_file
    spec = exp.option("corners")
    items = list(default) if spec is None else list(spec)
    out = []
    for it in items:
        if isinstance(it, Mapping):
            if "name" not in it or "alts" not in it:
                raise ExperimentError(f"corner {it!r}: needs name and alts")
            out.append((str(it["name"]), dict(it["alts"])))
            continue
        p = _resolve_file(str(it))
        data = yaml.safe_load(p.read_text()) or {}
        if (data.get("meta") or {}).get("kind") != "corner" or set(data) - {"extends", "meta", "alts"}:
            raise ExperimentError(f"corner {it!r}: {p} is not a corner preset (meta.kind corner, only alts)")
        out.append((str(it), dict(data.get("alts") or {})))
    return out


def _start_of(exp: Experiment, entry: dict | None, sc) -> str:
    if entry is not None and entry.get("start"):
        return entry["start"]
    s = exp.options.get("start")
    if s:
        return s
    rs = getattr(sc, "reset", None)
    if rs is not None:
        return "warm" if rs.start == "warm" else "cold"
    return "cold"


def _start_kwargs(start: str) -> dict:
    if start == "warm":
        # the quasi-static oracle lock codes, a ring that its heater-up order cannot lock (negative heater power)
        # moved to the next, bluer order (integrator 2026-10-10; identical codes when every ring has one, e.g. the
        # nominal g26 world)
        return {"heater_code": "quasi_static_reachable", "settle": True, "reset_protocol": "branch_high"}
    if start == "cold_settled":
        return {"settle": True, "reset_protocol": "branch_low"}
    return {}


def _period_ticks(cfg) -> int:
    dt = float(cfg.get("sim.dt_plant_s"))
    rate = cfg.get("controller_interface.update_rate_hz")
    return max(1, int(round(1.0 / (float(rate) * dt)))) if rate else 1


def _scenario(cfg):
    from mrm.scenarios import registry as scen_registry
    return scen_registry.from_config(cfg)


def _drives_base_temperature(sc) -> bool:
    return any(c.signal in ("thermal.t_base_k",) and getattr(c, "absolute", False) for c in sc.all_components)


def _self_ref_scenario(sc):
    """The static scenario with the main scenario's signal defaults (ambient, bias, transition density)."""
    from mrm.scenarios import library as lib
    if _drives_base_temperature(sc):
        raise ExperimentError(f"experiment.self_reference: scenario {sc.name!r} drives the base temperature with "
                              f"a component (its operating point at t = 0 is not a static one); use "
                              f"metrics.lock_reference oracle or line")
    return lib.static().replace(defaults=dict(sc.defaults), reset=sc.reset)


def _make_controller(entry: dict | None):
    if entry is None:
        return None
    from mrm.control import registry
    return registry.create(entry["name"], dict(entry.get("params") or {}))


def _truth_dict(t) -> dict:
    out = {k: np.asarray(v) for k, v in (t.fields or {}).items()}
    for k in ("p_chip_w", "p_out_w", "lam_m", "t_obs"):
        v = getattr(t, k, None)
        if v is not None:
            out[k] = np.asarray(v)
    out["tick"] = np.asarray(t.tick)
    return out


def _monitor_decimation(n_ticks: int, sf_ticks: int, max_windows: int = 10_000) -> int:
    """Window length (ticks) of the monitor min / max record of a self-referenced run.

    The windows must start at the scoring start ``sf_ticks`` (a window that straddles it would mix the unscored
    start-up into the extremes) and should end at ``n_ticks`` (the record keeps complete windows only, so a remainder
    would leave the last ticks unjudged). The largest common divisor g = gcd(n_ticks, sf_ticks) does both with the
    fewest windows; it is used when it keeps <= max_windows windows. Otherwise the smallest divisor of sf_ticks that
    keeps <= max_windows windows (the last n_ticks mod d ticks are then not covered), else that minimum length (the
    window holding the scoring start is dropped)."""
    n, sf = int(n_ticks), int(sf_ticks)
    lo = max(1, -(-n // max_windows))
    g = math.gcd(n, sf) if sf > 0 else n
    if g >= lo:
        return g
    if sf >= lo:
        return min(d for d in range(lo, sf + 1) if sf % d == 0)
    return lo


def _runner_kwargs(exp: Experiment) -> dict:
    return {"n_threads": exp.n_threads, "histogram": bool(exp.option("histogram")), "oracle": bool(exp.option(
        "oracle"))}


def _execute(exp: Experiment, case: Case, log: Callable = None) -> CaseResult:
    """Run one case (and its self-reference pre-run) on the engine."""
    from mrm.sim import runner
    t0 = time.perf_counter()
    # the case's own Config and controller (a from_config entry is rebuilt from it: sweeps over controller.params)
    cfg, entry, ov, alts = _case_config(exp, case)
    sc = case.scenario if case.scenario is not None else _scenario(cfg)
    start_kw = _start_kwargs(case.start)
    seeds = case.seeds if case.seeds is not None else exp.seeds
    n_ticks = int(round(case.duration_s / float(cfg.get("sim.dt_plant_s"))))
    record = case.record
    if case.record_default:
        record = default_records(n_ticks, _period_ticks(cfg))
    ref = None
    metrics_kw: dict = {}
    caught: list[str] = []
    rkw = _runner_kwargs(exp)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        if case.self_reference is not None:
            sr = dict(case.self_reference)
            dur = float(sr.get("duration_s", 0.02))
            win = float(sr.get("window_s", 0.005))
            dt_s = float(cfg.get("sim.dt_plant_s"))
            per = _period_ticks(cfg)
            # ~10 window means inside the reference window (whole controller periods): bounded memory at any B
            dec = max(per, per * int(round(win / dt_s / 10.0 / per)))
            rec = [RecordSpec("detuning_m", dec, "mean"), RecordSpec("i_monitor_a", dec, "mean")]
            pre = runner.run(cfg, _make_controller(entry), _self_ref_scenario(sc), duration_s=dur,
                             batch=case.batch, seeds=seeds, record=rec, uncertainty=case.uncertainty,
                             variation=case.variation, **start_kw, **rkw)
            tr = pre.trace
            t = np.asarray(tr[rec[0].key]["t_s"])                  # window END times
            # the windows that lie inside the last window_s of the pre-run (start >= end - window_s)
            sel = (t - dec * dt_s) >= (t[-1] - win) - 0.5 * dt_s if t.size else np.zeros(0, bool)
            if not np.any(sel):
                raise ExperimentError(f"experiment.self_reference: no complete {dec}-tick window inside the last "
                                      f"window_s = {win} s of the {dur} s pre-run")
            d_ref = np.median(np.asarray(tr[rec[0].key]["value"])[sel], axis=0) * 1e12      # (B, N) pm
            i_ref = np.median(np.asarray(tr[rec[1].key]["value"])[sel], axis=0)             # (B, N) A
            ref = {"d_ref_pm": d_ref, "i_monitor_ref_a": i_ref, "duration_s": dur, "window_s": win,
                   "wall_s": pre.wall_s, "n_windows": int(np.sum(sel))}
            metrics_kw = {"lock_reference": "target", "lock_target_pm": d_ref}
            # monitor envelope for the monitor-error metrics: windows aligned on the scoring start and on the end
            # of the run (the min / max of window extremes is exact at any decimation), at most ~10 000 windows
            sf_ticks = int(round(float(exp.option("score_from_s") or 0.0) / dt_s))
            dmon = _monitor_decimation(n_ticks, sf_ticks)
            mon = [RecordSpec("i_monitor_a", dmon, "min"), RecordSpec("i_monitor_a", dmon, "max")]
            have = {r.key for r in (record or [])}
            record = list(record or []) + [r for r in mon if r.key not in have]
        sf = exp.option("score_from_s")
        res = runner.run(cfg, _make_controller(entry), sc, duration_s=case.duration_s, batch=case.batch,
                         seeds=seeds, record=record or None, metrics=metrics_kw or None,
                         uncertainty=case.uncertainty, variation=case.variation,
                         **({"score_from_s": sf} if sf else {}), **start_kw, **rkw)
    for w in wl:
        msg = f"{w.category.__name__}: {w.message}"
        if msg not in caught:
            caught.append(msg)
    m = dict(res.metrics)
    ws_info = (res.manifest_inputs.get("warm_start") or {})
    if ws_info.get("no_code") is not None:
        m["warm_start_no_code"] = np.asarray(ws_info["no_code"], bool)
    if ws_info.get("supply_limited") is not None:
        m["warm_start_supply_limited"] = np.asarray(ws_info["supply_limited"], bool)            # (B,)
    if ws_info.get("order_hopped") is not None:
        m["warm_start_order_hopped"] = np.asarray(ws_info["order_hopped"], bool)
    if ref is not None and res.trace:
        t_w = np.asarray(res.trace[mon[0].key]["t_s"])
        keep = (t_w - dmon * dt_s) >= float(exp.option("score_from_s") or 0.0) - 0.5 * dt_s   # windows after it
        lo = np.asarray(res.trace[mon[0].key]["value"])[keep]
        hi = np.asarray(res.trace[mon[1].key]["value"])[keep]
        iref = ref["i_monitor_ref_a"]
        with np.errstate(divide="ignore", invalid="ignore"):
            m["monitor_error_pct_min"] = (lo.min(axis=0) - iref) / iref * 100.0 if lo.size else np.full_like(iref,
                                                                                                         np.nan)
            m["monitor_error_pct_max"] = (hi.max(axis=0) - iref) / iref * 100.0 if hi.size else np.full_like(iref,
                                                                                                         np.nan)
            m["monitor_dev_ua_max"] = (np.maximum(np.abs(hi.max(axis=0) - iref), np.abs(lo.min(axis=0) - iref))
                                       * 1e6 if lo.size else np.full_like(iref, np.nan))
    wall = time.perf_counter() - t0
    mi = dict(res.manifest_inputs)
    mi["start"] = case.start
    mi["controller_entry"] = entry
    mi["overrides"] = ov
    mi["alts"] = alts
    mi["case_config_hash"] = cfg.hash()
    if log:
        log(f"  {case.label}: B = {case.batch}, {case.duration_s:g} s simulated, {wall:.1f} s wall")
    return CaseResult(label=case.label, metrics=m, trace=res.trace, truth_final=_truth_dict(res.truth_final),
                      manifest_inputs=mi, wall_s=wall, warnings=caught, reference=ref, group=case.group,
                      meta=dict(case.meta))


# ================================================================================================ planning
def _duration(exp: Experiment, sc=None) -> float:
    d = exp.duration_s
    if d is not None:
        return d
    if sc is not None and getattr(sc, "duration_s", None):
        dt = float(exp.cfg.get("sim.dt_plant_s"))
        return round(float(sc.duration_s) / float(sc.time_compression) / dt) * dt
    raise ExperimentError("experiment.duration_s is required (the scenario declares no duration)")


def _unc_var(exp: Experiment, *, default_unc: str, default_var: str) -> tuple[str, str]:
    u = exp.get("experiment.uncertainty") if exp.given("experiment.uncertainty") else default_unc
    v = exp.get("experiment.variation") if exp.given("experiment.variation") else default_var
    return str(u), str(v)


def _plan(exp: Experiment) -> list[Case]:
    """The cases an experiment runs (kind-specific; the convergence cases of a compressed run are added by
    _kind_run itself and costed by :func:`estimate`)."""
    kind = exp.kind
    if kind not in KINDS:
        raise ExperimentError(f"experiment.kind {kind!r} not in {KINDS}")
    cfg = exp.cfg
    sc = _scenario(cfg)
    if kind == "quasi_static":
        return []
    ctrl = _main_controller(exp)
    sref = exp.option("self_reference")
    if kind == "run":
        u, v = _unc_var(exp, default_unc=str(exp.get("experiment.uncertainty")),
                        default_var=str(exp.get("experiment.variation")))
        rec = exp.option("record")
        n_ticks = int(round(_duration(exp, sc) / float(cfg.get("sim.dt_plant_s"))))
        return [Case("main", controller=ctrl, batch=exp.batch, uncertainty=u, variation=v,
                     duration_s=_duration(exp, sc), record=None if rec is None else parse_records(
                         rec, n_ticks=n_ticks, period_ticks=_period_ticks(cfg)),
                     record_default=rec is None, start=_start_of(exp, ctrl, sc), self_reference=sref)]
    dur = _duration(exp, sc)
    n_ticks = int(round(dur / float(cfg.get("sim.dt_plant_s"))))
    rec_opt = exp.option("record")
    rec = parse_records(rec_opt, n_ticks=n_ticks, period_ticks=_period_ticks(cfg)) if rec_opt else None
    if kind == "sweep":
        cases = []
        for i, (assign, ov, alts, centry) in enumerate(_grid_points(exp)):
            c = centry if centry is not None else ctrl
            u, v = _unc_var(exp, default_unc=str(exp.get("experiment.uncertainty")),
                            default_var=str(exp.get("experiment.variation")))
            cases.append(Case(f"point_{i:03d}", controller=c, overrides=tuple(ov), alts=alts, batch=exp.batch,
                              uncertainty=u, variation=v, duration_s=dur, record=rec,
                              start=_start_of(exp, c, sc), self_reference=sref, group="sweep",
                              meta={"assignment": assign}))
        return cases
    if kind == "mc":
        n = exp.n_mc or exp.batch
        u, v = _unc_var(exp, default_unc=str(exp.get("experiment.uncertainty")),
                        default_var=str(exp.get("experiment.variation")))
        if u == "nominal":
            raise ExperimentError("experiment.kind mc needs experiment.uncertainty sample or lhs")
        cap = int(exp.option("max_worlds_per_plant"))
        if u == "lhs" and n > cap:
            raise ExperimentError(f"a Latin hypercube of {n} worlds must run in one Plant (the design depends on "
                                  f"B); raise experiment.max_worlds_per_plant above {cap} or use fewer worlds")
        cases = []
        for j, b0 in enumerate(range(0, n, cap)):
            B = min(cap, n - b0)
            sd = {k: v0 + b0 for k, v0 in exp.seeds.items()}
            cases.append(Case(f"worlds_{j:02d}" if n > cap else "worlds", controller=ctrl, batch=B, seeds=sd,
                              uncertainty=u, variation=v, duration_s=dur, record=rec,
                              start=_start_of(exp, ctrl, sc), self_reference=sref, group="mc",
                              meta={"first_world": b0}))
        return cases
    if kind in ("compare", "robust", "acceptance"):
        if kind == "compare":
            ents = exp.get("experiment.controllers")
            entries = [_controller_entry(e) for e in (ents or [])]
            if not entries:
                if ctrl is None:
                    raise ExperimentError("compare needs experiment.controllers (or a controller)")
                entries = [ctrl]
            labels = [e["label"] for e in entries]
            if len(set(labels)) != len(labels):
                raise ExperimentError(f"compare: duplicate controller labels {labels}; set label: on the entries")
        else:
            if ctrl is None and kind == "robust":
                raise ExperimentError("robust needs a controller section")
            entries = [ctrl]
        corners = _corners(exp, NAMED_CORNERS if kind in ("compare", "robust") else ())
        n_mc = exp.n_mc if exp.n_mc is not None else (256 if kind in ("compare", "robust") else 0)
        cases = []
        for e in entries:
            lab = e["label"] if e is not None else "open_loop"
            st = _start_of(exp, e, sc)
            if kind == "acceptance":
                u, v = _unc_var(exp, default_unc="nominal", default_var="off")
                cases.append(Case("main" if len(entries) == 1 else f"{lab}/main", controller=e, batch=exp.batch,
                                  uncertainty=u, variation=v, duration_s=dur, record=rec, start=st,
                                  self_reference=sref, group=lab, meta={"set": "main"}))
            else:
                cases.append(Case(f"{lab}/nominal", controller=e, batch=1, uncertainty="nominal", variation="off",
                                  duration_s=dur, record=rec, start=st, self_reference=sref, group=lab,
                                  meta={"set": "nominal"}))
            for cn, calts in corners:
                cases.append(Case(f"{lab}/corner/{cn}", controller=e, alts=calts, batch=1, uncertainty="nominal",
                                  variation="off", duration_s=dur, record=rec, start=st, self_reference=sref,
                                  group=lab, meta={"set": "corner", "corner": cn}))
            if n_mc:
                u, v = _unc_var(exp, default_unc="lhs", default_var=str(exp.get("experiment.variation")))
                if kind == "acceptance" and not exp.given("experiment.uncertainty"):
                    u = "lhs"
                if u == "nominal":
                    raise ExperimentError(f"{kind}: n_mc worlds need experiment.uncertainty lhs or sample")
                cases.append(Case(f"{lab}/worlds", controller=e, batch=int(n_mc), uncertainty=u, variation=v,
                                  duration_s=dur, record=None, start=st, self_reference=sref, group=lab,
                                  meta={"set": "worlds"}))
        return cases
    raise ExperimentError(f"unhandled kind {kind}")  # pragma: no cover


def _grid_points(exp: Experiment) -> list[tuple[dict, list[str], dict, dict | None]]:
    """[(assignment, overrides, alts, controller entry)] for every point of the sweep grid (Cartesian product, in
    the order of the grid keys)."""
    grid = exp.grid
    if not grid:
        raise ExperimentError("experiment.kind sweep needs experiment.grid {key: [values]}")
    keys = list(grid)
    vals = []
    for k in keys:
        v = grid[k]
        if not isinstance(v, (list, tuple)) or not v:
            raise ExperimentError(f"experiment.grid.{k}: a non-empty list of values")
        vals.append(list(v))
    pts = []
    for combo in itertools.product(*vals):
        ov, alts, centry = [], {}, None
        assign = {}
        for k, v in zip(keys, combo):
            assign[k] = v
            if k == "corner":
                alts.update(_corners_from_names([v])[0][1])
            elif k == "controller":
                centry = _controller_entry(v)
            elif k.startswith("alt:"):
                alts[k[4:]] = str(v)
            else:
                ov.append(_ov_str(k, v))
        pts.append((assign, ov, alts, centry))
    return pts


def _corners_from_names(names) -> list[tuple[str, dict]]:
    import yaml
    from mrm.params.config import _resolve_file
    out = []
    for n in names:
        p = _resolve_file(str(n))
        data = yaml.safe_load(p.read_text()) or {}
        if (data.get("meta") or {}).get("kind") != "corner":
            raise ExperimentError(f"grid corner {n!r} is not a corner preset")
        out.append((str(n), dict(data.get("alts") or {})))
    return out


# ================================================================================================ convergence
def plant_fingerprint(cfg, start: str, worlds: Mapping | None = None) -> str:
    """Fingerprint of everything a compression convergence certificate covers (07 P0-10): the resolved config
    without the ``experiment`` section and without ``scenario.time_compression`` (so the certificate of one
    experiment holds for another of the same plant, controller and scenario, e.g. another duration), plus the start
    state and ``worlds``: what defines the worlds the convergence was shown on and how they were scored ({batch,
    uncertainty, variation, seeds, score_from_s}; a certificate shown on the nominal world does not cover sampled
    worlds)."""
    from mrm.sim.manifest import to_jsonable
    tree = copy.deepcopy(cfg.resolved_tree())
    tree.pop("experiment", None)
    if isinstance(tree.get("scenario"), dict):
        tree["scenario"].pop("time_compression", None)
    blob = json.dumps({"tree": tree, "start": start, "worlds": to_jsonable(dict(worlds or {}))}, sort_keys=True,
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _worlds_of(exp: Experiment, case: Case) -> dict:
    """The world-defining settings of a case (plant_fingerprint ``worlds``)."""
    return {"batch": int(case.batch), "uncertainty": case.uncertainty, "variation": case.variation,
            "seeds": dict(case.seeds if case.seeds is not None else exp.seeds),
            "score_from_s": float(exp.option("score_from_s") or 0.0)}


def load_certificate(path: str | Path):
    """A ConvergenceCertificate from a run directory or its convergence.json."""
    from mrm.scenarios import ConvergenceCertificate
    p = Path(path)
    if p.is_dir():
        p = p / "convergence.json"
    if not p.exists():
        raise ExperimentError(f"no convergence certificate at {p}")
    with open(p) as f:
        d = json.load(f)
    cert = d.get("certificate")
    if not cert:
        raise ExperimentError(f"{p}: the convergence experiment did not produce a certificate (it failed)")
    return ConvergenceCertificate.from_dict(cert)


def _convergence(exp: Experiment, main: Case, sc, log) -> tuple[Any, dict, list[CaseResult]]:
    """The mandatory 07 P0-10 convergence experiment for a compressed scenario: one slow cycle at every factor on
    the same plant, controller, start, worlds and seeds; per world, every factor's CONVERGENCE_METRICS must agree
    with factor 1. Returns (certificate or None, report dict, case results)."""
    from mrm.scenarios import ConvergenceCertificate, assess_convergence, convergence_plan
    from mrm.scenarios.generators import CONVERGENCE_METRICS
    factors = [float(x) for x in exp.option("compression_factors")]
    plan = convergence_plan(sc, factors)
    fp = plant_fingerprint(exp.cfg, main.start, _worlds_of(exp, main))
    results = []
    world = {}
    dt = float(exp.cfg.get("sim.dt_plant_s"))
    for c, (sc_c, dur) in sorted(plan.items()):
        dur = round(float(dur) / dt) * dt                       # one slow cycle, whole ticks
        case = dataclasses.replace(main, label=f"convergence_c{c:g}", scenario=sc_c, duration_s=dur,
                                   record=None, record_default=False, group="convergence", meta={"factor": c})
        r = _execute(exp, case, log)
        results.append(r)
        world[c] = {m: np.asarray(r.metrics["world"][m], float) for m in CONVERGENCE_METRICS}
    B = main.batch
    reports = []
    max_ok = math.inf
    for b in range(B):
        rep = assess_convergence(sc, {c: {m: float(world[c][m][b]) for m in CONVERGENCE_METRICS} for c in world},
                                 plant_fingerprint=fp)
        reports.append({"world": b, "passed": rep.passed, "failures": list(rep.failures),
                        "details": {str(k): v for k, v in rep.details.items()},
                        "max_validated_factor": rep.certificate.max_validated_factor if rep.certificate else 1.0})
        max_ok = min(max_ok, rep.certificate.max_validated_factor if rep.certificate else 1.0)
    cert = None
    if max_ok > 1.0:
        from mrm.scenarios.generators import DEFAULT_CONVERGENCE_TOLERANCES
        cert = ConvergenceCertificate(
            scenario_fingerprint=sc.fingerprint(include_compression=False), plant_fingerprint=fp,
            factors=tuple(sorted(world)), max_validated_factor=float(max_ok),
            metrics={str(c): {m: world[c][m].tolist() for m in CONVERGENCE_METRICS} for c in world},
            tolerances=dict(DEFAULT_CONVERGENCE_TOLERANCES))
    rep = {"factors": sorted(world), "plant_fingerprint": fp,
           "scenario_fingerprint": sc.fingerprint(include_compression=False),
           "passed": all(r["passed"] for r in reports), "max_validated_factor": float(max_ok),
           "worlds": reports, "certificate": cert.to_dict() if cert else None,
           "cycle_s": sc.slow_cycle_s,
           "note": ("07 P0-10: one slow cycle at each factor; per world, max detuning, wrong-line and FSR-hop "
                    "events, heater range used and integrated errors must agree with factor 1 "
                    "(mrm.scenarios.DEFAULT_CONVERGENCE_TOLERANCES, design choices). Integrated errors are compared "
                    "raw (not scaled by the compression).")}
    return cert, rep, results


# ================================================================================================ kinds
def _fmt(x, nd=3) -> str:
    if x is None:
        return "-"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_fmt(v, nd) for v in x) + "]"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        return f"{v:.{nd}g}"
    return f"{v:.{nd}f}".rstrip("0").rstrip(".") if "." in f"{v:.{nd}f}" else f"{v:.{nd}f}"


_REPORT_METRICS = ("oma_penalty_db_worst", "oma_penalty_db_mean", "excursion_count", "excursion_max_s",
                   "time_unlocked_s", "wrong_line_rings", "time_to_lock_s", "unlocked_rings", "max_abs_lock_error_pm",
                   "mean_reloma_db", "oracle_reloma_db", "fsr_hop_events", "ber_mean_worst",
                   "warm_start_no_code_rings", "warm_start_order_hopped_rings", "warm_start_supply_limited")


def _header(exp: Experiment, controller_report: bool) -> list[str]:
    lines = [Q1_CAVEAT] if controller_report else []
    lines += [f"experiment {exp.name} (kind {exp.kind}), config {exp.cfg.hash()[:12]}, experiment hash "
              f"{exp.hash[:12]}", f"files: {', '.join(exp.cfg.files)}"]
    return lines


def _compression_gate(exp: Experiment, main: Case, log) -> tuple[bool, list[CaseResult], dict, list[str]]:
    """07 P0-10 gate for a time-compressed scenario: a certificate from experiment.convergence_certificate, or the
    convergence experiment run here on the main case's plant, controller, start, worlds and seeds. Returns (allowed,
    convergence case results, extra {'convergence': report}, report lines). An uncompressed scenario passes."""
    from mrm.scenarios import CompressionError, require_compression_validated
    sc = main.scenario if main.scenario is not None else _scenario(exp.config(tuple(main.overrides), main.alts))
    if float(sc.time_compression) == 1.0:
        return True, [], {}, []
    fp = plant_fingerprint(exp.cfg, main.start, _worlds_of(exp, main))
    cpath = exp.option("convergence_certificate")
    results: list[CaseResult] = []
    extra: dict = {}
    if cpath:
        cert = load_certificate(cpath)
        extra["convergence"] = {"certificate": cert.to_dict(), "from": str(cpath)}
    else:
        log(f"time compression {sc.time_compression:g}: running the 07 P0-10 convergence experiment "
            f"(one {sc.slow_cycle_s} s cycle at {exp.option('compression_factors')})")
        cert, rep, results = _convergence(exp, main, sc, log)
        extra["convergence"] = rep
    try:
        require_compression_validated(sc, cert, plant_fingerprint=fp)
    except CompressionError as e:
        return False, results, extra, [f"REFUSED: {e}"]
    return True, results, extra, [f"time compression {sc.time_compression:g}: validated up to "
                                  f"{cert.max_validated_factor:g} (07 P0-10 convergence certificate)"]


def _refuse_compression(exp: Experiment, kind: str) -> None:
    sc = _scenario(exp.cfg)
    if float(sc.time_compression) != 1.0:
        raise ExperimentError(f"a time-compressed scenario needs a convergence certificate per plant and "
                              f"controller (07 P0-10); kind {kind} would need one for every case: use kind run or "
                              f"acceptance (main case only)")


def _criteria_step(exp: Experiment, results: Sequence[CaseResult], lines: list[str], summary: dict,
                   title: str = "") -> tuple[bool | None, str]:
    """Evaluate experiment.acceptance on the case results (every kind); appends the table to the report."""
    crit = _criteria(exp.option("acceptance")) if exp.option("acceptance") else []
    if not crit:
        return None, "ok"
    ev = evaluate_criteria(crit, {r.label: r.metrics for r in results if r.group != "convergence"})
    summary["criteria"] = ev
    passed = all(e["pass"] for e in ev)
    lines.append("")
    lines += _criteria_lines(ev, title)
    return passed, ("ok" if passed else "failed")


def _kind_run(exp: Experiment, log) -> tuple[dict, list[CaseResult], list[str], bool | None, str, dict]:
    (main,) = _plan(exp)
    lines = _header(exp, main.controller is not None)
    ok, results, extra, glines = _compression_gate(exp, main, log)
    lines += glines
    if not ok:
        summary = {"status": "refused", "reason": glines[-1], "convergence": extra.get("convergence")}
        return summary, results, lines, None, "refused", extra
    r = _execute(exp, main, log)
    results.append(r)
    ws = world_summary(r.metrics)
    summary = {"status": "ok", "world_summary": {k: v.tolist() for k, v in ws.items()},
               "start": main.start, "batch": main.batch, "duration_s": main.duration_s}
    lines.append(f"start {main.start}; B = {main.batch}; {main.duration_s:g} s simulated; wall {r.wall_s:.1f} s")
    lines += _world_table({"main": ws})
    passed, status = _criteria_step(exp, [r], lines, summary)
    summary["status"] = status
    return summary, results, lines, passed, status, extra


def _case_config(exp: Experiment, case: Case):
    """(Config, controller entry) of one case: the experiment config with the case's and its controller entry's
    overrides and alternatives; a from_config entry is rebuilt from that Config (sweeps over controller.params)."""
    entry = case.controller
    ov = list(case.overrides) + (list(entry.get("overrides", [])) if entry else [])
    alts = dict(case.alts)
    if entry:
        alts.update(entry.get("alts") or {})
    cfg = exp.config(tuple(ov), alts)
    if entry is not None and entry.get("from_config"):
        entry = _controller_from_cfg(cfg)
    return cfg, entry, ov, alts


def _prebuild(exp: Experiment, cases: Sequence[Case]) -> None:
    """Build every case's Config, scenario and controller before anything runs, so that a bad sweep value, corner or
    controller entry fails at once instead of after the earlier cases ran (the Configs are cached for the run)."""
    seen = set()
    for c in cases:
        cfg, entry, _, _ = _case_config(exp, c)
        key = (id(cfg), json.dumps(entry, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        if c.scenario is None:
            _scenario(cfg)
        _make_controller(entry)


def _world_table(rows: Mapping[str, Mapping[str, np.ndarray]], metrics=_REPORT_METRICS) -> list[str]:
    """Rows: label -> world_summary; one line per label with the median over worlds (and the worst, when B > 1)."""
    out = ["| case | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for lab, ws in rows.items():
        cells = []
        for k in metrics:
            v = np.asarray(ws[k] if k in ws else np.nan, float)
            if v.size > 1:
                cells.append(f"{_fmt(np.median(v))} (max {_fmt(np.max(v))})")
            else:
                cells.append(_fmt(v.reshape(-1)[0] if v.size else np.nan))
        out.append(f"| {lab} | " + " | ".join(cells) + " |")
    return out


def _kind_sweep(exp, log):
    _refuse_compression(exp, "sweep")
    cases = _plan(exp)
    _prebuild(exp, cases)
    results = []
    rows = {}
    pts = []
    for c in cases:
        r = _execute(exp, c, log)
        results.append(r)
        rows[f"{c.label} {json.dumps(c.meta['assignment'], default=str)}"] = world_summary(r.metrics)
        pts.append({"label": c.label, "assignment": c.meta["assignment"],
                    "world_summary": {k: v.tolist() for k, v in world_summary(r.metrics).items()}})
    lines = _header(exp, any(c.controller is not None for c in cases)) + _world_table(rows)
    summary = {"points": pts}
    passed, status = _criteria_step(exp, results, lines, summary, "every point")
    return summary, results, lines, passed, status, {}


def _kind_mc(exp, log):
    _refuse_compression(exp, "mc")
    cases = _plan(exp)
    _prebuild(exp, cases)
    results = [_execute(exp, c, log) for c in cases]
    ws = {}
    for r in results:
        for k, v in world_summary(r.metrics).items():
            ws.setdefault(k, []).append(v)
    ws = {k: np.concatenate(v) for k, v in ws.items()}
    q = {k: _quantiles(v) for k, v in ws.items()}
    lines = _header(exp, any(c.controller is not None for c in cases))
    lines.append(f"{len(ws['excursion_count'])} worlds ({cases[0].uncertainty}, variation {cases[0].variation})")
    lines.append("| metric | p5 | p25 | p50 | p75 | p95 | max | nan |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k in [k for k in _REPORT_METRICS if k in q]:
        d = q.get(k, {})
        lines.append(f"| {k} | " + " | ".join(_fmt(d.get(x)) for x in ("p5", "p25", "p50", "p75", "p95", "max"))
                     + f" | {d.get('nan', 0)} |")
    summary = {"quantiles": q, "world_summary": {k: v.tolist() for k, v in ws.items()}}
    passed, status = _criteria_step(exp, results, lines, summary, "every world")
    return summary, results, lines, passed, status, {}


def _compare_like(exp, log, kind):
    cases = _plan(exp)
    _prebuild(exp, cases)
    results = []
    extra: dict = {}
    pre_lines: list[str] = []
    if float(_scenario(exp.cfg).time_compression) != 1.0:
        if kind != "acceptance" or len(cases) != 1:
            _refuse_compression(exp, kind if kind != "acceptance" else "acceptance with corners or worlds")
        ok, conv, extra, pre_lines = _compression_gate(exp, cases[0], log)
        results.extend(conv)
        if not ok:
            lines = _header(exp, any(c.controller is not None for c in cases)) + pre_lines
            return ({"status": "refused", "reason": pre_lines[-1], "convergence": extra.get("convergence")},
                    results, lines, None, "refused", extra)
    by_group: dict[str, dict] = {}
    for c in cases:
        r = _execute(exp, c, log)
        results.append(r)
        g = by_group.setdefault(c.group, {"nominal": None, "corners": {}, "worlds": None, "main": None})
        ws = world_summary(r.metrics)
        s = c.meta.get("set")
        if s == "nominal":
            g["nominal"] = ws
        elif s == "corner":
            g["corners"][c.meta["corner"]] = ws
        elif s == "worlds":
            g["worlds"] = ws
        elif s == "main":
            g["main"] = ws
    lines = _header(exp, any(c.controller is not None for c in cases)) + pre_lines
    rank_keys = tuple(exp.option("rank_metrics") or RANK_METRICS)
    summary: dict = {"controllers": {}, "rank_metrics": list(rank_keys)}
    for lab, g in by_group.items():
        d: dict = {}
        if g["nominal"] is not None:
            d["nominal"] = {k: float(v[0]) for k, v in g["nominal"].items()}
        if g["main"] is not None:
            d["main"] = {k: v.tolist() for k, v in g["main"].items()}
        if g["corners"]:
            d["corners"] = {cn: {k: float(v[0]) for k, v in ws.items()} for cn, ws in g["corners"].items()}
            worst = {}
            for k in [k for k in _REPORT_METRICS if all(k in ws for ws in g["corners"].values())]:
                lower_better = k not in ("mean_reloma_db", "oracle_reloma_db")
                vals = {cn: float(ws[k][0]) for cn, ws in g["corners"].items()}
                key = (lambda cn: (-np.inf if np.isnan(vals[cn]) else vals[cn])) if lower_better else (
                    lambda cn: (-np.inf if np.isnan(vals[cn]) else -vals[cn]))
                cn = max(vals, key=key)
                worst[k] = {"corner": cn, "value": vals[cn]}
            d["worst_corner"] = worst
            d["worst_corner_overall"] = _worst_corner_overall(g["corners"])
        if g["worlds"] is not None:
            d["quantiles"] = {k: _quantiles(v) for k, v in g["worlds"].items()}
        summary["controllers"][lab] = d
    # ranking stability (07 4.8)
    if kind == "compare" and len(by_group) >= 2:
        summary["rank_stability"] = {}
        for k in rank_keys:
            vals, nom = {}, {}
            for lab, g in by_group.items():
                parts = []
                if g["worlds"] is not None:
                    parts.append(np.asarray(g["worlds"][k], float))
                for cn in sorted(g["corners"]):
                    parts.append(np.asarray(g["corners"][cn][k], float))
                vals[lab] = np.concatenate(parts) if parts else np.zeros(0)
                nom[lab] = float(g["nominal"][k][0]) if g["nominal"] is not None else np.nan
            if all(v.size for v in vals.values()):
                summary["rank_stability"][k] = rank_stability(vals, nom)
    # report
    for lab, d in summary["controllers"].items():
        lines.append("")
        lines.append(f"## {lab}")
        if "nominal" in d:
            lines.append("nominal world: " + ", ".join(f"{k} {_fmt(d['nominal'][k])}" for k in _REPORT_METRICS
                                                       if k in d["nominal"]))
        if "worst_corner_overall" in d:
            w = d["worst_corner_overall"]
            lines.append(f"worst named corner: {w['corner']} ({w['why']})")
            lines.append("worst corner per metric: " + ", ".join(
                f"{k} {_fmt(v['value'])} ({v['corner']})" for k, v in d["worst_corner"].items()
                if k in RANK_METRICS))
        if "quantiles" in d:
            q = d["quantiles"]
            n = q["excursion_count"]["n"]
            lines.append(f"{n} worlds: median [p5, p95] (max)")
            for k in [k for k in _REPORT_METRICS if k in q]:
                qq = q.get(k, {})
                lines.append(f"  {k}: {_fmt(qq.get('p50'))} [{_fmt(qq.get('p5'))}, {_fmt(qq.get('p95'))}] "
                             f"(max {_fmt(qq.get('max'))}, nan {qq.get('nan', 0)})")
        if "main" in d:
            lines.append("main case (median over its worlds): " + ", ".join(
                f"{k} {_fmt(np.median(d['main'][k]))}" for k in _REPORT_METRICS if k in d["main"]))
    if summary.get("rank_stability"):
        lines.append("")
        lines.append("## Ranking stability (Kendall tau-b of each world's ranking against the nominal ranking)")
        lines.append("| metric | nominal order (best first) | tau mean | tau min | same order | defined |")
        lines.append("|---|---|---|---|---|---|")
        for k, rs in summary["rank_stability"].items():
            lines.append(f"| {k} | {' < '.join(rs['nominal_order'])} | {_fmt(rs['tau_mean'])} | "
                         f"{_fmt(rs['tau_min'])} | {_fmt(rs['frac_same_order'])} | {_fmt(rs['frac_defined'])} |")
    passed, status = _criteria_step(exp, results, lines, summary,
                                    "over the whole set; 07 4.8: a claim holds only if it holds in every world"
                                    if kind != "acceptance" else "")
    if "convergence" in extra:
        summary["convergence"] = {k: extra["convergence"].get(k) for k in ("passed", "max_validated_factor",
                                                                             "factors", "from")
                                  if k in extra["convergence"]}
    return summary, results, lines, passed, status, extra


def _worst_corner_overall(corners: Mapping[str, Mapping[str, np.ndarray]]) -> dict:
    """The worst named corner: most rings on a foreign line, then most rings unlocked at the end, then the largest
    worst-lane OMA penalty, then the most excursions."""
    def key(cn):
        ws = corners[cn]
        pen = float(ws["oma_penalty_db_worst"][0])
        return (float(ws["wrong_line_rings"][0]), float(ws["unlocked_rings"][0]),
                -np.inf if math.isnan(pen) else pen, float(ws["excursion_count"][0]))
    cn = max(corners, key=key)
    k = key(cn)
    return {"corner": cn, "why": f"wrong-line rings {_fmt(k[0])}, unlocked rings {_fmt(k[1])}, worst-lane OMA "
                                 f"penalty {_fmt(k[2])} dB, excursions {_fmt(k[3])}"}


def _criteria_lines(ev: Sequence[dict], title: str) -> list[str]:
    out = ["## Criteria" + (f" ({title})" if title else ""),
           "| id | metric | op | value | worst | pass | source |", "|---|---|---|---|---|---|---|"]
    for e in ev:
        rings = "" if e["rings"] is None else f" rings {e['rings']}"
        out.append(f"| {e['id']} | {e['metric']} ({e['reduce']}{rings}) | {e['op']} | {_fmt(e['value'])} | "
                   f"{_fmt(e['worst'])} | {'PASS' if e['pass'] else 'FAIL'} | {e['source']} |")
    return out


def _qs_one(exp: Experiment, cfg, amb, bv, u, v, label: str, log) -> tuple[CaseResult, dict, dict]:
    """One quasi-static survey (runner.quasi_static) -> (CaseResult, summary, arrays)."""
    from mrm.sim import runner
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        qs = runner.quasi_static(cfg, ambient_c=amb, bias_v=bv, batch=exp.batch, seeds=exp.seeds,
                                 p_in_w=exp.option("p_in_w"), uncertainty=u, variation=v)
    wall = time.perf_counter() - t0
    log(f"  {label}: {amb.size} ambient x {bv.size} bias x B = {exp.batch}, {wall:.1f} s wall")
    arrays = {k: np.asarray(getattr(qs, k)) for k in ("ambient_c", "bias_v", "detuning_target_m", "reloma_db",
                                                       "reloma_unrestricted_db", "t_ring_k", "p_abs_w",
                                                       "p_heater_w", "heater_code", "code_ok", "fsr_hop_risk",
                                                       "wrong_line_risk", "code_range_coverage",
                                                       "bnd4_reloma_variation_db", "supply_scale", "supply_limited")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bank = 10 * np.log10(np.nanmean(10 ** (arrays["reloma_db"] / 10), axis=3))          # (A, V, B)
    arrays["oracle_reloma_bank_db"] = bank
    fin = np.isfinite(arrays["heater_code"])
    summary = {
        "code_range_coverage": arrays["code_range_coverage"].tolist(),
        "points_without_code": int((~arrays["code_ok"]).sum()),
        "fsr_hop_risk_points": int(arrays["fsr_hop_risk"].sum()),
        "supply_limited_points": int(arrays["supply_limited"].sum()),
        "supply_scale_min": float(arrays["supply_scale"].min()) if arrays["supply_scale"].size else 1.0,
        "wrong_line_risk_max": int(arrays["wrong_line_risk"].max()) if arrays["wrong_line_risk"].size else 0,
        "wrong_line_risk_mean": float(arrays["wrong_line_risk"].mean()) if arrays["wrong_line_risk"].size else 0.0,
        "bnd4_reloma_variation_db": arrays["bnd4_reloma_variation_db"].tolist(),
        "bnd4_max_db": float(np.nanmax(arrays["bnd4_reloma_variation_db"])),
        "oracle_reloma_bank_db": bank.tolist(),
        "heater_code_min": float(np.nanmin(arrays["heater_code"])) if fin.any() else float("nan"),
        "heater_code_max": float(np.nanmax(arrays["heater_code"])) if fin.any() else float("nan"),
        "notes": list(qs.notes), "wall_s": wall,
        "warnings": sorted({f"{w.category.__name__}: {w.message}" for w in wl}),
    }
    metrics = {"world": {"code_range_coverage": arrays["code_range_coverage"],
                         "bnd4_max_db": np.nanmax(arrays["bnd4_reloma_variation_db"], axis=0)},
               "bnd4_reloma_variation_db": arrays["bnd4_reloma_variation_db"],
               "code_range_coverage": arrays["code_range_coverage"],
               "fsr_hop_risk_points": arrays["fsr_hop_risk"].sum(axis=(0, 1)),
               "supply_limited_points": arrays["supply_limited"].sum(axis=(0, 1)),
               "wrong_line_risk_max": arrays["wrong_line_risk"].max(axis=(0, 1)),
               "no_code_points": (~arrays["code_ok"]).sum(axis=(0, 1))}
    res = CaseResult(label=label, metrics=metrics, trace=None, truth_final={},
                     manifest_inputs={"quasi_static": True, "uncertainty": u, "variation": v, "batch": exp.batch,
                                      "seeds": exp.seeds, "p_in_w": exp.option("p_in_w"),
                                      "config_hash": cfg.hash(), "case_config_hash": cfg.hash()},
                     wall_s=wall, warnings=summary["warnings"], group="quasi_static")
    return res, summary, arrays


def _kind_quasi_static(exp, log):
    g = exp.grid
    amb = np.asarray(g.get("ambient_c", np.arange(0.0, 105.0 + 1e-9, 5.0)), float)
    bv = np.asarray(g.get("bias_v", np.arange(1.0, 3.75 + 1e-9, 0.25)), float)
    extra_keys = set(g) - {"ambient_c", "bias_v"}
    if extra_keys:
        raise ExperimentError(f"quasi_static grid takes ambient_c and bias_v only, got {sorted(extra_keys)}")
    u, v = _unc_var(exp, default_unc="nominal", default_var="off")
    cases = [("quasi_static", {})] + [(f"quasi_static/corner/{cn}", calts) for cn, calts in _corners(exp, ())]
    results, per, arrays_all = [], {}, {}
    for label, calts in cases:
        cfg = exp.config((), calts) if calts else exp.cfg
        r, sm, ar = _qs_one(exp, cfg, amb, bv, u, v, label, log)
        results.append(r)
        per[label] = sm
        arrays_all[label.replace("/", "|")] = ar
    main = per["quasi_static"]
    arrays = arrays_all["quasi_static"]
    bank = arrays["oracle_reloma_bank_db"]
    summary = {"ambient_c": amb.tolist(), "bias_v": bv.tolist(), "batch": exp.batch, "uncertainty": u,
               "variation": v, **main, "cases": {k: {kk: vv for kk, vv in sm.items() if kk != "oracle_reloma_bank_db"}
                                                 for k, sm in per.items()}}
    lines = _header(exp, False)
    lines.append(f"quasi-static locked steady state (07 P0-10, BND-4): {amb.size} ambient points "
                 f"{_fmt(amb.min())}..{_fmt(amb.max())} degC x {bv.size} bias points {_fmt(bv.min())}..{_fmt(bv.max())}"
                 f" V, B = {exp.batch} ({u}, variation {v})")
    lines.append("| case | code-range coverage | points without a code | FSR-hop risk points | supply-limited points "
                 "(min droop) | wrong-line risk max (mean) | heater codes | BND-4 max (dB) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for lab, sm in per.items():
        lines.append(f"| {lab} | {_fmt(sm['code_range_coverage'])} | {sm['points_without_code']} | "
                     f"{sm['fsr_hop_risk_points']} | {sm['supply_limited_points']} ({_fmt(sm['supply_scale_min'])}) | "
                     f"{sm['wrong_line_risk_max']} ({_fmt(sm['wrong_line_risk_mean'])}) "
                     f"| {_fmt(sm['heater_code_min'])}..{_fmt(sm['heater_code_max'])} | {_fmt(sm['bnd4_max_db'])} |")
    lines.append("code-range coverage: fraction of (ambient, bias) points with every ring reachable within the array "
                 "heater supply limit; points without a code count rings (a supply-limited point counts all of its "
                 "rings); supply-limited: the bank's lock powers exceed the supply (07 P1-12), with the smallest "
                 "droop factor; wrong-line risk: foreign lines a heater-up sweep from code_min crosses; BND-4: oracle "
                 "relOMA variation over ambient at a fixed bias (BND-4 asks < 1 dB; only a heater-temperature loss "
                 "mechanism can move it, 07 4.5)")
    lines.append("oracle relOMA, bank mean (dB, OMA / P_in, 07 D9), world 0, rows ambient, columns bias:")
    lines.append("| degC | " + " | ".join(_fmt(b) for b in bv) + " |")
    lines.append("|---" * (bv.size + 1) + "|")
    for ia, a in enumerate(amb):
        lines.append(f"| {_fmt(a)} | " + " | ".join(_fmt(bank[ia, iv, 0], 2) for iv in range(bv.size)) + " |")
    passed, status = _criteria_step(exp, results, lines, summary)
    # one layout with or without corners: /quasi_static/<case label, '/' -> '|'>/<array>
    extra = {"quasi_static": arrays_all}
    return summary, results, lines, passed, status, extra


_KINDS = {"run": _kind_run, "sweep": _kind_sweep, "mc": _kind_mc,
          "compare": lambda e, l: _compare_like(e, l, "compare"),
          "robust": lambda e, l: _compare_like(e, l, "robust"),
          "acceptance": lambda e, l: _compare_like(e, l, "acceptance"),
          "quasi_static": _kind_quasi_static}


# ================================================================================================ run
def _as_experiment(x, overrides=None, alts=None, options=None) -> Experiment:
    from mrm.params.config import Config
    if isinstance(x, Experiment):
        e = load(x, overrides, alts) if (overrides or alts) else x
    elif isinstance(x, Config):
        if overrides or alts:
            raise ExperimentError("pass a file path to apply overrides; a Config is already resolved")
        e = from_config(x, options)
        options = None
    else:
        e = load(x, overrides, alts)
    if options:
        opts = dict(e.options)
        opts.update(options)
        e = dataclasses.replace(e, options=_validate_options(opts), _cache=e._cache)
    return e


def run(experiment, *, overrides=None, alts=None, options: Mapping | None = None, kind: str | None = None,
        runs_dir: str | Path | None = None, write: bool = True, argv: Sequence[str] | None = None,
        log: Callable[[str], None] | None = None, rerun_of: str | None = None) -> ExperimentResult:
    """Run an experiment and write its run directory ``<runs_dir>/<UTC>_<hash8>/`` (07 P1-17).

    ``experiment``: a YAML path or preset name, a Config (``mrm.run(mrm.params.load('exp.yaml'))``), or an
    Experiment. ``overrides`` ({path: value} or ["path=value"]) and ``alts`` ({path: alternative}) as for
    :func:`load`; ``experiment.<extension>`` overrides set extension options. ``options``: extension options from
    Python (e.g. {"n_threads": 4}). ``kind``: force experiment.kind (``mrm mc file.yaml``). ``runs_dir``: default
    $MRM_RUNS_DIR or ./runs. ``write`` False: no run directory. ``argv``: recorded in the manifest (default
    sys.argv). ``log``: progress lines (default: stderr). ``rerun_of``: the run id a rerun reproduces. Returns an
    :class:`ExperimentResult`.
    """
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    ov = list(_plain_override_list(overrides)) if overrides else []
    if kind is not None:
        if kind not in KINDS:
            raise ExperimentError(f"kind {kind!r} not in {KINDS}")
        ov.append(f"experiment.kind={kind}")
    exp = _as_experiment(experiment, ov or None, alts, options)
    t0 = time.perf_counter()
    import datetime as dt
    started = dt.datetime.now(dt.timezone.utc)
    log(f"mrm experiment {exp.name}: kind {exp.kind}, hash {exp.hash[:8]}")
    from mrm.sim import manifest as mf
    state0 = {"git": mf.git_state(), "environment": mf.environment()} if write else {}
    summary, results, lines, passed, status, extra = _KINDS[exp.kind](exp, log)
    wall = time.perf_counter() - t0
    if passed is not None:
        lines.insert(1 if lines and lines[0] == Q1_CAVEAT else 0, f"RESULT: {'PASS' if passed else 'FAIL'}")
    elif status != "ok":
        lines.insert(1 if lines and lines[0] == Q1_CAVEAT else 0, f"RESULT: {status.upper()}")
    lines.append(f"wall time {wall:.1f} s")
    report = "\n".join(lines) + "\n"
    cases = {r.label: r for r in results}
    res = ExperimentResult(kind=exp.kind, name=exp.name, experiment_hash=exp.hash, cases=cases, summary=summary,
                           report=report, passed=passed, status=status, run_dir=None, wall_s=wall, extra=extra)
    if write:
        res.run_dir = _write_run(exp, res, started, argv, rerun_of, runs_dir, state0)
        log(f"run directory: {res.run_dir}")
    return res


# ================================================================================================ writing
def _metrics_for_json(m: Mapping) -> dict:
    out = {}
    for k, v in m.items():
        if k in ("detuning_histogram", "excursion_run_hist_log2"):
            continue
        out[k] = v
    return out


def _artefacts_of_cases(exp: Experiment) -> list[dict]:
    """mrm.sim.manifest.artefacts_used over the experiment's Config and every case Config it built (union by file:
    ``used_by`` merged, ``selected`` when any case selects a value from the file)."""
    from mrm.sim import manifest as mf
    merged: dict[str, dict] = {}
    for cfg in [exp.cfg, *exp._cache.values()]:
        for a in mf.artefacts_used(cfg):
            e = merged.get(a["file"])
            if e is None:
                merged[a["file"]] = dict(a, used_by=list(a["used_by"]))
                continue
            e["selected"] = bool(e["selected"] or a["selected"])
            e["used_by"] += [u for u in a["used_by"] if u not in e["used_by"]]
            if a.get("mismatch"):
                e["mismatch"] = True
    return [merged[k] for k in sorted(merged)]


def _write_run(exp: Experiment, res: ExperimentResult, started, argv, rerun_of, runs_dir,
               state0: Mapping | None = None) -> Path:
    from mrm.sim import manifest as mf
    d = mf.new_run_dir(runs_dir, exp.hash[:8], now=started)
    run_id = d.name
    src = exp.source
    inputs = []
    if src.get("path"):
        copies = src.get("input_copies") or _input_copies(Path(src["path"]))
        for c in copies:
            (d / "inputs" / c["name"]).write_text(c["text"])
            inputs.append({"file": f"inputs/{c['name']}", "original": c["original"], "sha256": c["sha256"]})
    # resolved yaml: the config tree with provenance, the resolved options and the source
    import yaml
    tree = exp.cfg.resolved_tree()
    tree.setdefault("experiment", {})
    for k, v in exp.resolved_options().items():
        tree["experiment"][k] = mf.to_jsonable(v["value"])
    tree["_source"] = {"files": list(exp.cfg.files), "overrides": list(src.get("overrides") or []),
                       "alts": dict(src.get("alts") or {}), "options_given": sorted(
                           k for k, v in exp.resolved_options().items() if v["given"])}
    (d / "experiment.resolved.yaml").write_text(
        "# experiment.resolved.yaml (07 P1-17): every config leaf with its provenance, the resolved experiment\n"
        "# options and the source; written by mrm.sim.experiment\n"
        + yaml.safe_dump(mf.to_jsonable(tree), sort_keys=True, default_flow_style=None, width=120))
    # artefacts: of the experiment's config and of every case config (corners and sweep points select their own
    # alternatives and settings, e.g. a thermal ROM)
    arts = _artefacts_of_cases(exp)
    case_hashes = sorted({r.manifest_inputs.get("case_config_hash") for r in res.cases.values()
                          if r.manifest_inputs.get("case_config_hash")})
    man = {
        "schema": mf.MANIFEST_SCHEMA, "run_id": run_id, "kind": exp.kind, "name": exp.name,
        "status": res.status, "passed": res.passed,
        "experiment_hash": exp.hash, "config_hash": exp.cfg.hash(), "case_config_hashes": case_hashes,
        "created_utc": started.isoformat(), "wall_s": res.wall_s, "rerun_of": rerun_of,
        # captured when the run started: the code and environment that ran (an edit during a long run must not
        # change what the manifest records)
        "git": (state0 or {}).get("git") or mf.git_state(),
        "environment": (state0 or {}).get("environment") or mf.environment(),
        "argv": list(argv) if argv is not None else list(sys.argv),
        "seeds": exp.seeds, "source": {"path": src.get("path"), "overrides": list(src.get("overrides") or []),
                                       "alts": dict(src.get("alts") or {}), "chain": src.get("chain") or [],
                                       "inputs": inputs},
        "options": {k: v["value"] for k, v in exp.resolved_options().items()},
        "artefacts": arts,
        "cases": {lab: {"manifest_inputs": r.manifest_inputs, "wall_s": r.wall_s, "warnings": r.warnings,
                        "group": r.group, "meta": r.meta,
                        "self_reference": None if r.reference is None else {
                            "duration_s": r.reference["duration_s"], "window_s": r.reference["window_s"]}}
                  for lab, r in res.cases.items()},
        "perf_reference": PERF_REFERENCE["source"],
    }
    if "convergence" in res.extra:
        mf.write_json(d / "convergence.json", res.extra["convergence"])
        man["convergence"] = {"file": "convergence.json",
                              "max_validated_factor": res.extra["convergence"].get("max_validated_factor")
                              if "max_validated_factor" in res.extra["convergence"]
                              else (res.extra["convergence"].get("certificate") or {}).get("max_validated_factor")}
    res.manifest = man
    mf.write_json(d / "manifest.json", man)
    mf.write_json(d / "metrics.json", {"kind": exp.kind, "name": exp.name, "status": res.status,
                                       "passed": res.passed, "summary": res.summary,
                                       "cases": {lab: _metrics_for_json(r.metrics) for lab, r in res.cases.items()}})
    groups: dict = {"metrics": {}, "trace": {}, "truth_final": {}, "self_reference": {}}
    for lab, r in res.cases.items():
        key = lab.replace("/", "|")
        mm = {k: v for k, v in r.metrics.items() if k != "settings"}
        if "detuning_histogram" in mm:
            mm["detuning_histogram"] = dict(mm["detuning_histogram"])
        groups["metrics"][key] = mm
        if r.trace:
            groups["trace"][key] = {k: {"t_s": v["t_s"], "value": v["value"]} for k, v in r.trace.items()}
        if r.truth_final:
            groups["truth_final"][key] = r.truth_final
        if r.reference is not None:
            groups["self_reference"][key] = {"d_ref_pm": r.reference["d_ref_pm"],
                                             "i_monitor_ref_a": r.reference["i_monitor_ref_a"]}
    if "quasi_static" in res.extra:
        groups["quasi_static"] = res.extra["quasi_static"]
    mf.write_trace_h5(d / "trace.h5", groups, attrs={"run_id": run_id, "kind": exp.kind,
                                                      "experiment_hash": exp.hash})
    (d / "report.txt").write_text(res.report)
    return d


# ================================================================================================ rerun
def rerun(run_dir: str | Path, *, runs_dir: str | Path | None = None, write: bool = True, compare: bool = True,
          log: Callable[[str], None] | None = None, allow_changed_config: bool = False) -> tuple[ExperimentResult,
                                                                                                dict | None]:
    """Reproduce a run from its directory (07 P1-17 ``mrm rerun runs/<id>``): reload the copied experiment file
    with the recorded overrides, alternatives and options, check that the experiment hash is the recorded one
    (else the presets, artefacts or code changed: ExperimentError unless ``allow_changed_config``), run it into a
    new run directory (``rerun_of`` in its manifest) and compare every array of trace.h5 bit for bit
    (:func:`mrm.sim.manifest.compare_runs`). Warns when the git commit, dirty diff or environment differ (results
    may then differ at rounding level). Returns (result, comparison or None)."""
    from mrm.sim import manifest as mf
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    d = Path(run_dir)
    old = mf.read_run(d)
    man = old["manifest"]
    src = man.get("source") or {}
    ins = src.get("inputs") or []
    if not ins:
        raise ExperimentError(f"{d}: the run has no copied experiment file (it was run from a Config without a "
                              f"source file); it cannot be rerun")
    path = d / ins[0]["file"]
    opts = {}
    for k, v in (man.get("options") or {}).items():
        if k in _EXT and v is not None and v != _EXT[k].default:
            opts[k] = v
    exp = load(path, overrides=src.get("overrides") or None, alts=src.get("alts") or None)
    exp = dataclasses.replace(exp, options=_validate_options({**exp.options, **opts}), _cache={})
    exp.source["path"] = str(path)
    if exp.kind != man.get("kind"):
        raise ExperimentError(f"rerun: kind {exp.kind} != recorded {man.get('kind')}")
    if exp.hash != man.get("experiment_hash"):
        msg = (f"rerun: experiment hash {exp.hash[:12]} != recorded {str(man.get('experiment_hash'))[:12]}: the "
               f"presets, artefacts or the loader changed since the run")
        if not allow_changed_config:
            raise ExperimentError(msg)
        log("WARNING " + msg)
    g_now, g_then = mf.git_state(), man.get("git") or {}
    if (g_now.get("commit"), g_now.get("diff_sha256")) != (g_then.get("commit"), g_then.get("diff_sha256")):
        log(f"WARNING rerun: code differs from the run (commit {str(g_then.get('commit'))[:10]} dirty-diff "
            f"{str(g_then.get('diff_sha256'))[:10]} -> {str(g_now.get('commit'))[:10]} "
            f"{str(g_now.get('diff_sha256'))[:10]}); a bit-for-bit match is not guaranteed")
    env_now, env_then = mf.environment(), man.get("environment") or {}
    for k in ("numpy", "scipy", "numba", "llvmlite", "python"):
        if env_now.get(k) != env_then.get(k):
            log(f"WARNING rerun: {k} {env_then.get(k)} -> {env_now.get(k)}")
    res = run(exp, runs_dir=runs_dir, write=write, log=log, rerun_of=man.get("run_id"),
              argv=["mrm", "rerun", str(d)])
    cmp = None
    if compare and write:
        cmp = mf.compare_runs(d, res.run_dir)
        log("rerun: bit-for-bit IDENTICAL" if cmp["identical"] else
            f"rerun: {len(cmp['differing'])} arrays differ (e.g. {cmp['differing'][:5]})")
        mf.write_json(Path(res.run_dir) / "rerun_comparison.json", {"original": str(d), **cmp})
    return res, cmp


# ================================================================================================ dry run
def estimate(experiment, *, overrides=None, alts=None, kind: str | None = None,
             options: Mapping | None = None) -> dict:
    """``mrm run --dry-run`` (07 P0-10): B, step counts, estimated RAM and wall time of every case, from the
    measured engine speed of docs/handoff/perf.md section 8 (:data:`PERF_REFERENCE`); nothing is run. Arguments as
    for :func:`run` (``options``: extension options from Python, e.g. {"n_threads": 4} from ``--threads``)."""
    from mrm.control.api import CompiledController
    ov = list(_plain_override_list(overrides)) if overrides else []
    if kind is not None:
        ov.append(f"experiment.kind={kind}")
    exp = _as_experiment(experiment, ov or None, alts, options)
    P = PERF_REFERENCE
    cfg = exp.cfg
    dt = float(cfg.get("sim.dt_plant_s"))
    policy = str(cfg.get("sim.substep_policy", "fixed"))
    rate = cfg.get("controller_interface.update_rate_hz")
    T = exp.n_threads
    rows = []
    cases = _plan(exp)
    sc = _scenario(cfg)
    if exp.kind in ("run", "acceptance") and cases and float(sc.time_compression) != 1.0 \
            and not exp.option("convergence_certificate"):
        from mrm.scenarios import convergence_plan
        for c, (sc_c, d) in sorted(convergence_plan(sc, exp.option("compression_factors")).items(), reverse=True):
            cases.insert(0, dataclasses.replace(cases[-1], label=f"convergence_c{c:g}", duration_s=round(
                float(d) / dt) * dt,
                                                record=None, record_default=False))
    for c in cases:
        # the case's own config and controller (a sweep may change the controller, its rate or the policy)
        cfg_c, entry, _, _ = _case_config(exp, c)
        ctrl = _make_controller(entry) if entry else None
        rate_c = cfg_c.get("controller_interface.update_rate_hz")
        policy_c = str(cfg_c.get("sim.substep_policy", "fixed"))
        path = "open" if ctrl is None else ("compiled" if isinstance(ctrl, CompiledController) else "python")
        n_ticks = int(round(c.duration_s / dt))
        pre_ticks = int(round(float((c.self_reference or {}).get("duration_s", 0.02)) / dt)) if c.self_reference \
            else 0
        B = int(c.batch)
        us_tick = max(B * P["us_per_world_tick_parallel"], math.ceil(B / T) * P["us_per_tick_b1_fixed"])
        if policy_c == "auto":
            if B == 1:
                lr = math.log10(float(rate_c)) if rate_c else 4.0
                f100 = P["auto_factor_b1_100khz_ramp"] if any(cc.slow for cc in sc.all_components) \
                    else P["auto_factor_b1_100khz"]
                f = np.interp(lr, [4.0, 5.0], [P["auto_factor_b1_10khz"], f100])
            else:
                f = P["auto_factor_parallel"]
            us_tick *= float(f)
        edges = (n_ticks + pre_ticks) * dt * float(rate_c) if (rate_c and path == "python") else 0.0
        t_py = edges * (P["python_us_per_edge"] + P["python_us_per_edge_per_world"] * B) * 1e-6
        t_setup = P["setup_s"] * (2 if c.self_reference else 1) + (
            P["quasi_static_s_per_point_world"] * B * (2 if c.self_reference else 1) if c.start == "warm" else 0.0)
        wall = (n_ticks + pre_ticks) * us_tick * 1e-6 + t_py + t_setup
        rec = c.record if c.record is not None else (default_records(n_ticks, _period_ticks(cfg_c))
                                                     if c.record_default else [])
        N = cfg.n_rings
        rec_bytes = sum((n_ticks // r.decimation) * B * N * 8 for r in rec)
        ram = (P["ram_base_gb"] + B * (P["ram_mb_per_world"] + (P["ram_mb_per_world_histogram"] if
                                                                exp.option("histogram") else 0.0)) / 1e3
               + rec_bytes / 1e9)
        rows.append({"case": c.label, "B": B, "ticks": n_ticks, "pre_ticks": pre_ticks,
                     "controller": None if entry is None else entry["name"], "path": path,
                     "controller_edges": int(round((n_ticks + pre_ticks) * dt * float(rate_c))) if rate_c and ctrl
                     else 0, "policy": policy_c, "start": c.start, "records": len(rec),
                     "record_mb": rec_bytes / 1e6, "ram_gb": ram, "wall_s": wall})
    if exp.kind == "quasi_static":
        g = exp.grid
        A = len(g.get("ambient_c", np.arange(0.0, 105.0 + 1e-9, 5.0)))
        V = len(g.get("bias_v", np.arange(1.0, 3.75 + 1e-9, 0.25)))
        wall = A * V * exp.batch * P["quasi_static_s_per_point_world"]
        for lab in ["quasi_static"] + [f"quasi_static/corner/{cn}" for cn, _ in _corners(exp, ())]:
            rows.append({"case": lab, "B": exp.batch, "ticks": 0, "points": A * V, "path": "static",
                         "ram_gb": P["ram_base_gb"] + exp.batch * P["ram_mb_per_world"] / 1e3, "wall_s": wall})
    from mrm.sim import kernel as _k
    cache = Path(_k.__file__).parent / "__pycache__"
    warm = any(p.name.startswith("kernel.") and p.suffix == ".nbi" for p in cache.iterdir()) if cache.is_dir() \
        else False
    total = sum(r["wall_s"] for r in rows) + (0.0 if warm else P["cold_compile_s"])
    ram = max([r["ram_gb"] for r in rows] + [0.0])
    ram_total = max(ram, P["cold_compile_gb"] if not warm else 0.0) + sum(r.get("record_mb", 0.0) for r in rows) / 1e3
    return {"experiment": exp.name, "kind": exp.kind, "experiment_hash": exp.hash, "dt_s": dt,
            "substep_policy": policy, "controller_rate_hz": rate, "n_threads": T, "cases": rows,
            "total_ticks": int(sum(r["ticks"] + r.get("pre_ticks", 0) for r in rows)),
            "world_ticks": int(sum((r["ticks"] + r.get("pre_ticks", 0)) * r["B"] for r in rows)),
            "est_wall_s": total, "est_peak_ram_gb": ram_total, "numba_cache_warm": warm,
            "basis": (f"estimated from the measured engine speed in {P['source']}: B = 1 fixed "
                      f"{P['us_per_tick_b1_fixed']} us per tick, B >= 8 on 8 threads {P['us_per_world_tick_parallel']} "
                      f"us per world-tick, auto policy x{P['auto_factor_b1_100khz']:.2f} (100 kHz; "
                      f"x{P['auto_factor_b1_100khz_ramp']:.2f} measured under a soak ramp) to "
                      f"x{P['auto_factor_b1_10khz']:.2f} (10 kHz), Python controllers +{P['python_us_per_edge']:g} "
                      f"us per edge; RAM {P['ram_base_gb']} GB + {P['ram_mb_per_world']} MB per world (measured); "
                      + ("warm numba cache" if warm else
                         f"cold numba cache: +{P['cold_compile_s']:g} s compile, {P['cold_compile_gb']:.1f} GB"))}
