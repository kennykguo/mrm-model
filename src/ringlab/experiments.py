"""Reproducible runs and paired controller comparisons with small memory use."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from numbers import Integral
from pathlib import Path
import platform
import subprocess

import numpy as np
import yaml

from .config import load
from .controllers import create, plugin_source_hashes
from .plant import simulate
from .provenance import PROJECT, require_unchanged_sources, source_hashes

DISCLAIMER = ("Performance is conditional on this model's documented assumptions and uncertain parameters. "
              "These runs do not validate Lightmatter hardware or predict its BER.")


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json(path, value):
    Path(path).write_text(json.dumps(plain(value), sort_keys=True, indent=2, allow_nan=False)+"\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parameter_json(value):
    """JSON constructor data must round-trip without replacing invalid numbers."""
    if isinstance(value, np.ndarray):
        return _parameter_json(value.tolist())
    if isinstance(value, np.generic):
        return _parameter_json(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("controller parameters must use text mapping keys")
        return {key: _parameter_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_parameter_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError("saved controller parameters must be finite JSON-compatible constructor data")


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def versions():
    return {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pyyaml", "matplotlib")}


def _git_revision():
    process = subprocess.run(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return process.stdout.strip() if process.returncode == 0 else None


def directory(root, digest):
    path = Path(root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")+"_"+digest[:8])
    path.mkdir(parents=True)
    return path


def save(result, output="runs", *, plot=True):
    from .junction import DepletionJunction, MeasuredJunctionTable
    require_unchanged_sources(result.source_sha256)
    if result.controller_spec is None:
        raise ValueError("cannot save an unknown injected controller; use an importable module:Class configuration "
                         "or supply controller_configuration with explicit constructor parameters")
    controller_spec = _parameter_json(result.controller_spec)
    if controller_spec["kind"] == "importable" and plugin_source_hashes(controller_spec["name"]) != controller_spec["source_sha256"]:
        raise ValueError("controller plugin source changed since simulation; reload it in a fresh process and rerun")
    if not isinstance(result.junction_model, (DepletionJunction, MeasuredJunctionTable)):
        raise ValueError("saved replay supports DepletionJunction or MeasuredJunctionTable; custom junction needs a serialization adapter")
    path = directory(output, result.configuration.digest())
    (path/"resolved.yaml").write_text(yaml.safe_dump(result.configuration.resolved(), sort_keys=True))
    _json(path/"metrics.json", result.metrics)
    np.savez_compressed(path/"trace.npz", **result.trace)
    # Save the actual injected/calibrated models. The resolved configuration
    # alone cannot describe a caller-supplied replacement thermal/junction model.
    thermal = result.thermal_model
    np.savez_compressed(path/"thermal_model.npz", **{name: getattr(thermal, name) for name in
                        ("capacity_j_per_k", "conductance_w_per_k", "source_map", "boundary_conductance_w_per_k",
                         "observer_matrix", "node_names", "source_names", "observer_names")})
    _json(path/"junction_model.json", {"type": type(result.junction_model).__name__,
                                      "parameters": asdict(result.junction_model)})
    _json(path/"controller.json", controller_spec)
    manifest = {"schema": "ringlab-run/1", "claim": DISCLAIMER, "git_commit": _git_revision(),
                "source_sha256": result.source_sha256, "versions": versions(), "python": platform.python_version(),
                "platform": platform.platform(), "config_sha256": result.configuration.digest(),
                "diagnostics": result.diagnostics,
                "controller_sha256": sha256(path/"controller.json"),
                "replay_scope": "Project source, configured numerical dependencies, saved physical models, and controller constructor settings. Plugin entry/factory source is fingerprinted; other plugin imports, external resources and prior object state are not captured.",
                "physical_models": {"thermal_model.npz": sha256(path/"thermal_model.npz"),
                                    "junction_model.json": sha256(path/"junction_model.json")},
                "nonfinite": "JSON null means unavailable/non-finite, never passing evidence"}
    _json(path/"manifest.json", manifest)
    if plot:
        plot_trace(result.trace, path/"response.png")
    return path


def plot_trace(trace, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = trace["time_s"]*1e3
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
    for n in range(trace["ring_temperature_k"].shape[1]):
        axes[0].plot(t, trace["ring_temperature_k"][:, n]-273.15, label=f"Ring {n}")
        axes[1].plot(t, trace["detuning_hz"][:, n]/1e9)
        axes[2].plot(t, trace["monitor_code"][:, n])
        axes[3].plot(t, trace["oma_w"][:, n]*1e6)
    for ax, label in zip(axes, ("Ring temperature (°C)", "Resonance − laser (GHz)", "Released monitor (ADC code)", "Bus-output OMA (µW)")):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Synthetic MRM control experiment — conditional on configured parameters")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rerun(path, output="runs", *, allow_changed_model=False):
    from .junction import DepletionJunction, MeasuredJunctionTable
    from .thermal import ThermalNetwork
    path = Path(path)
    manifest = json.loads((path/"manifest.json").read_text())
    if manifest.get("schema") != "ringlab-run/1":
        raise ValueError("unsupported run schema")
    config = load(path/"resolved.yaml")
    if config.digest() != manifest["config_sha256"]:
        raise ValueError("saved configuration has changed")
    require_unchanged_sources()
    if "controller_sha256" not in manifest:
        raise ValueError("saved run has no actual-controller specification; exact controller replay is unavailable")
    if sha256(path/"controller.json") != manifest["controller_sha256"]:
        raise ValueError("saved controller specification changed")
    controller_spec = json.loads((path/"controller.json").read_text())
    changes = []
    if source_hashes() != manifest["source_sha256"]:
        changes.append("model source")
    if versions() != manifest["versions"]:
        changes.append("numerical dependencies")
    if platform.python_version() != manifest["python"]:
        changes.append("Python version")
    if platform.platform() != manifest["platform"]:
        changes.append("platform")
    if controller_spec["kind"] == "importable" and plugin_source_hashes(controller_spec["name"]) != controller_spec["source_sha256"]:
        changes.append("controller plugin source")
    if changes and not allow_changed_model:
        raise ValueError("replay mismatch in "+", ".join(changes)+"; explicitly allow a changed-model comparison")
    for name, expected in manifest["physical_models"].items():
        if sha256(path/name) != expected:
            raise ValueError(f"saved physical model changed: {name}")
    with np.load(path/"thermal_model.npz", allow_pickle=False) as stored:
        thermal = ThermalNetwork(**{k: (tuple(stored[k]) if k.endswith("_names") else stored[k]) for k in stored.files})
    junction_data = json.loads((path/"junction_model.json").read_text())
    junction_cls = {"DepletionJunction": DepletionJunction, "MeasuredJunctionTable": MeasuredJunctionTable}.get(junction_data["type"])
    if junction_cls is None:
        raise ValueError("unsupported saved junction model")
    junction = junction_cls(**junction_data["parameters"])
    controller = create(controller_spec["name"], controller_spec["parameters"])
    result = simulate(config, controller=controller, thermal_network=thermal, junction_model=junction,
                      controller_configuration={key: controller_spec[key] for key in ("name", "parameters")})
    result.diagnostics["replay_changes"] = changes
    result.diagnostics["replay_of"] = str(path.resolve())
    return save(result, output)


def dry_run(config):
    c = load(config)
    n = c.array.count
    sample_events = int(np.ceil(c.simulation.duration_s/c.sensor.sample_period_s))
    records = int(np.ceil(c.simulation.duration_s/c.simulation.record_period_s))+1
    return {"rings": n, "duration_s": c.simulation.duration_s, "sample_events": sample_events,
            "record_count": records, "estimated_trace_bytes": records*(10*n+2)*8,
            "wall_time": "not estimated; run a short benchmark with the same model and solver tolerances",
            "model": c.data.averaging, "config_sha256": c.digest(), "claim": DISCLAIMER}


def draw_uncertainty(spec, samples, seed):
    """Reproducible named independent draws. Correlations must be supplied as cases.

    Ranges are sensitivity assumptions, not measured device population priors.
    """
    samples = _nonnegative_integer(samples, "uncertainty sample count")
    seed = _nonnegative_integer(seed, "uncertainty seed")
    rng = np.random.default_rng(seed)
    draws = [{} for _ in range(samples)]
    for path, dist in spec.items():
        if set(dist) - {"kind", "low", "high"}:
            raise ValueError(f"uncertainty {path}: expected kind, low, high")
        lo, hi = float(dist["low"]), float(dist["high"])
        if not np.isfinite(lo+hi) or lo > hi:
            raise ValueError(f"invalid range for {path}")
        # Latin hypercube: stratify each marginal, shuffle independently.
        u = (np.arange(samples)+rng.random(samples))/max(samples, 1)
        rng.shuffle(u)
        if dist.get("kind", "uniform") == "uniform":
            values = lo+(hi-lo)*u
        elif dist["kind"] == "loguniform" and lo > 0:
            values = np.exp(np.log(lo)+(np.log(hi)-np.log(lo))*u)
        else:
            raise ValueError(f"unsupported uncertainty distribution for {path}")
        for draw, value in zip(draws, values):
            draw[path] = float(value)
    return draws


def compare(study_path, output="runs", *, preview=False):
    """Run paired cases sequentially, saving each result before freeing its arrays."""
    study_path = Path(study_path)
    study = yaml.safe_load(study_path.read_text())
    allowed = {"model", "controllers", "cases", "uncertainty", "samples", "seed", "overrides"}
    if not isinstance(study, dict) or set(study)-allowed:
        raise ValueError("invalid study keys")
    model_path = (study_path.parent/study["model"]).resolve()
    base = load(model_path, study.get("overrides"))
    controllers = study.get("controllers", [])
    if not controllers or any(not isinstance(v, dict) or "name" not in v for v in controllers):
        raise ValueError("study.controllers must contain name/parameters mappings")
    cases = study.get("cases", {"nominal": {}})
    if not isinstance(cases, dict) or any(not isinstance(key, str) or not key or not isinstance(value, dict)
                                          for key, value in cases.items()):
        raise ValueError("study.cases must map nonempty case names to override mappings")
    cases = dict(cases)
    seed = _nonnegative_integer(study.get("seed", 1), "study.seed")
    samples = _nonnegative_integer(study.get("samples", 0), "study.samples")
    for i, values in enumerate(draw_uncertainty(study.get("uncertainty", {}), samples, seed)):
        label = f"sample_{i:04d}"
        if label in cases:
            raise ValueError(f"generated uncertainty case {label!r} conflicts with a supplied case")
        cases[label] = values
    plan = []
    for i, (label, overrides) in enumerate(cases.items()):
        for ctrl in controllers:
            cfg = load(base, dict(overrides, **{"controller.name": ctrl["name"],
                                               "controller.parameters": ctrl.get("parameters", {}),
                                               "simulation.seed": seed+i}))
            plan.append((label, ctrl["name"], cfg))
    if preview:
        return {"cases": [{"case": label, "controller": ctrl, **dry_run(cfg)} for label, ctrl, cfg in plan]}
    root = directory(output, base.digest())
    (root/"study.yaml").write_text(yaml.safe_dump(study))
    rows = []
    for label, ctrl, cfg in plan:
        result = simulate(cfg)
        path = save(result, root, plot=False)
        rows.append({"case": label, "controller": ctrl, "run": str(path.relative_to(root)),
                     "metrics": plain(result.metrics), "diagnostics": result.diagnostics})
        _json(root/"comparison.json", {"claim": DISCLAIMER, "runs": rows, "total_runs": len(plan),
                                      "completed": len(rows), "uncertainty": study.get("uncertainty", {}),
                                      "sampling": "paired seeds; independent Latin-hypercube marginals"})
        del result
    lines = [DISCLAIMER, "", "| Case | Controller | Mean OMA (µW) | Wrong-line time (ms, worst ring) | Heater energy (µJ) |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['case']} | {row['controller']} | {np.mean(m['mean_oma_w'])*1e6:.4g} | "
                     f"{np.max(m['wrong_line_time_s'])*1e3:.4g} | {np.sum(m['heater_energy_j'])*1e6:.4g} |")
    lines += ["", "Includes acquisition. Controllers may target different operating points; OMA, range and recovery",
              "must be considered together. No universal ranking or hardware performance claim is inferred.", ""]
    (root/"comparison.md").write_text("\n".join(lines))
    return {"directory": str(root), "runs": len(rows), "claim": DISCLAIMER}
