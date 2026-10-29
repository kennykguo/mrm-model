# compare calibration-target predictions before and after applying fitted values.
# `before_after` runs each target's primary replica protocol twice.
# G25 targets use the original and calibrated bench.
# other targets use the original and calibrated G26 bank.
# `format_table` displays residuals.
# only calibration targets are read.
# the fit adds an instrument floor for the notch data and uses a uniform aggressor for the DAC data.
# this report uses the target file's primary protocol, so its after column tests the calibrated plant under that protocol.
from __future__ import annotations

import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mrm.calibration import preset
from mrm.calibration.core import calibration_targets, jsonable
from mrm.validation import replicas
from mrm.validation.targets_io import evaluate

__all__ = ["before_after", "format_table"]


# {target id — {"before" — record, "after" — record, "plant_after" — name, "fit" — bool}} for every calibration target (or only `ids`).
# `results` is the fit record.
# it builds the calibrated bench when `bench_preset_path` is None or missing.
# a record holds the replica status, value, first notes and runtime (s), and, for a successful run, the tolerance verdict and the residuals.
def before_after(results: Mapping[str, Any], *, ids: Sequence[str] | None = None,
                 preset_path: str | Path = preset.PRESET_PATH,
                 bench_preset_path: str | Path | None = preset.BENCH_PRESET_PATH, log=print) -> dict[str, dict]:
    targets = calibration_targets()
    if bench_preset_path is not None and Path(bench_preset_path).exists():
        bench, bench_name = preset.bank_config(bench_preset_path), Path(bench_preset_path).stem
    else:
        bench, bench_name = preset.bench_config(results), "g25_test_vehicle + calibrated device leaves"
    bank = preset.bank_config(preset_path)
    table = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for target_id, target in targets.items():
            if ids and target_id not in ids:
                continue
            before = _replica_record(target, None)
            after = _replica_record(target, bench if target.generation == "G25" else bank)
            table[target_id] = {"before": before, "after": after, "plant_after": bench_name
                        if target.generation == "G25" else Path(preset_path).stem, "fit": bool(target.fit)}
            log(f"  {target_id}: before {before.get('level')} (|r| {_fmt_residual(before.get('max_abs_residual'))}) -> "
                f"after {after.get('level')} (|r| {_fmt_residual(after.get('max_abs_residual'))})")
    return table


# a plain-text table of the before/after verdicts and residuals (`table` — the result of `before_after`).
def format_table(table: Mapping[str, dict]) -> str:
    rows = [f"{'id':8s} {'before':>10s} {'|res| before':>14s} {'after':>10s} {'|res| after':>13s}"]
    for target_id, record in table.items():
        before, after = record["before"], record["after"]
        rows.append(f"{target_id:8s} {str(before.get('level')):>10s} {_fmt_residual(before.get('max_abs_residual')):>14s} "
                    f"{str(after.get('level')):>10s} {_fmt_residual(after.get('max_abs_residual')):>13s}")
    return "\n".join(rows)


# markdown tables of a fit record — per group, the hypothesis comparison and the leaf decisions.
def summary_markdown(results: Mapping[str, Any]) -> str:
    out = []
    for g, group in results["groups"].items():
        out.append(f"### {g} ({', '.join(group['targets'])}), nominal `{group['nominal']}`\n")
        out.append("| hypothesis | chi2 | dof | k | LR | LR (Birge) | dAIC |")
        out.append("|---|---|---|---|---|---|---|")
        for r in group["comparison"]["rows"]:
            out.append(f"| {r['hypothesis']} | {r['chi2']:.4g} | {r['dof']} | {r['n_params']} | "
                       f"{r['likelihood_ratio']:.3g} | {r['likelihood_ratio_birge']:.3g} | {r['delta_aic']:.3g} |")
        out.append("")
        out.append("| leaf | value | was | provenance | 95 % profile hull | width/prior | rule |")
        out.append("|---|---|---|---|---|---|---|")
        for decision in group["decisions"]:
            ident = decision["identifiable"]
            hull = ident.get("profile_95")
            hull_text = f"{hull[0]:.4g} .. {hull[1]:.4g}" if hull else "-"
            v = decision["value"]
            value_text = f"{v:.5g}" if isinstance(v, (int, float)) else str(v)
            previous_value = decision["previous_value"]
            previous_text = f"{previous_value:.5g}" if isinstance(previous_value, (int, float)) else str(previous_value)
            width_frac = ident.get("width_frac")
            out.append(f"| `{decision['path']}` | {value_text}{'' if decision.get('write_value',
                True) else ' (not written)'} | "
                       f"{previous_text} ({decision['previous_provenance']}) | {decision['provenance']} | {hull_text} | "
                       f"{'-' if width_frac is None else f'{width_frac:.2f}'} | {ident.get('rule', '')} |")
            for an, a in (decision.get("alternatives") or {}).items():
                out.append(f"| &nbsp;&nbsp;alt `{an}` | {a['value']:.5g} | | {a.get('provenance')} | | | |")
        out.append("")
    return "\n".join(out)


# run target `t` through its replica on `plant` (None — the target's own uncalibrated preset).
def _replica_record(t, plant) -> dict:
    started = time.time()
    result = replicas.run_target(t, plant)
    record: dict[str, Any] = {"status": result.status, "value": jsonable(result.value), "notes": list(result.notes)[:3],
                           "runtime_s": time.time() - started}
    if result.status == "ok":
        evaluation = evaluate(t, result.value)
        record.update(level=evaluation.level, passed=evaluation.passed,
            max_abs_residual=evaluation.max_abs_residual, z=evaluation.z)
    return record


# a residual to 4 significant digits, or "-" when there is none.
def _fmt_residual(x) -> str:
    return "-" if x is None else f"{x:.4g}"
