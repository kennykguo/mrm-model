"""Before/after residuals of every calibration target: the replicas (their own protocols and primary hypotheses,
the arguments of the target file) on the uncalibrated presets and on the calibrated ones.

Owner: calibration. Calibration targets only (07 P0-5): the G25 targets run on ``g25_test_vehicle`` (before) and on
``g25_test_vehicle`` with the calibrated device leaves (after, :func:`mrm.calibration.preset.bench_config`); every
other target on ``g26`` (before) and ``g26_calibrated`` (after). The fit protocols differ from the replicas' primary
in two stated places (an instrument floor on CAL-2, a uniform aggressor and one die for CAL-9): the "after" column
here is the replica's own protocol, so it shows what the target file's protocol gives on the calibrated plant.
"""
from __future__ import annotations

import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mrm.calibration import preset as P
from mrm.calibration.core import calibration_targets, jsonable
from mrm.validation import replicas as R
from mrm.validation.targets_io import evaluate

__all__ = ["before_after", "format_table"]


def _one(t, plant) -> dict:
    t0 = time.time()
    res = R.run_target(t, plant)
    rec: dict[str, Any] = {"status": res.status, "value": jsonable(res.value), "notes": list(res.notes)[:3],
                           "runtime_s": time.time() - t0}
    if res.status == "ok":
        ev = evaluate(t, res.value)
        rec.update(level=ev.level, passed=ev.passed, max_abs_residual=ev.max_abs_residual, z=ev.z)
    return rec


def before_after(results: Mapping[str, Any], *, ids: Sequence[str] | None = None,
                 preset_path: str | Path = P.PRESET_PATH, log=print) -> dict[str, dict]:
    """{target id: {"before": rec, "after": rec}} for every calibration target (or ``ids``)."""
    T = calibration_targets()
    bench = P.bench_config(results)
    bank = P.bank_config(preset_path)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tid, t in T.items():
            if ids and tid not in ids:
                continue
            before = _one(t, None)
            after = _one(t, bench if t.generation == "G25" else bank)
            out[tid] = {"before": before, "after": after, "plant_after": "g25_test_vehicle + calibrated device leaves"
                        if t.generation == "G25" else P.PRESET_NAME, "fit": bool(t.fit)}
            log(f"  {tid}: before {before.get('level')} (|r| {_short(before.get('max_abs_residual'))}) -> after "
                f"{after.get('level')} (|r| {_short(after.get('max_abs_residual'))})")
    return out


def _short(x) -> str:
    return "-" if x is None else f"{x:.4g}"


def format_table(ba: Mapping[str, dict]) -> str:
    """A plain-text table of the before/after verdicts and residuals."""
    rows = [f"{'id':8s} {'before':>10s} {'|res| before':>14s} {'after':>10s} {'|res| after':>13s}"]
    for tid, r in ba.items():
        b, a = r["before"], r["after"]
        rows.append(f"{tid:8s} {str(b.get('level')):>10s} {_short(b.get('max_abs_residual')):>14s} "
                    f"{str(a.get('level')):>10s} {_short(a.get('max_abs_residual')):>13s}")
    return "\n".join(rows)


def summary_markdown(results: Mapping[str, Any]) -> str:
    """Markdown tables of a fit record: per group the hypothesis comparison and the leaf decisions."""
    out = []
    for g, gr in results["groups"].items():
        out.append(f"### {g} ({', '.join(gr['targets'])}), nominal `{gr['nominal']}`\n")
        out.append("| hypothesis | chi2 | dof | k | LR | LR (Birge) | dAIC |")
        out.append("|---|---|---|---|---|---|---|")
        for r in gr["comparison"]["rows"]:
            out.append(f"| {r['hypothesis']} | {r['chi2']:.4g} | {r['dof']} | {r['n_params']} | "
                       f"{r['likelihood_ratio']:.3g} | {r['likelihood_ratio_birge']:.3g} | {r['delta_aic']:.3g} |")
        out.append("")
        out.append("| leaf | value | was | provenance | 95 % profile hull | width/prior | rule |")
        out.append("|---|---|---|---|---|---|---|")
        for d in gr["decisions"]:
            idt = d["identifiable"]
            hull = idt.get("profile_95")
            hs = f"{hull[0]:.4g} .. {hull[1]:.4g}" if hull else "-"
            v = d["value"]
            vs = f"{v:.5g}" if isinstance(v, (int, float)) else str(v)
            pv = d["previous_value"]
            ps = f"{pv:.5g}" if isinstance(pv, (int, float)) else str(pv)
            wf = idt.get("width_frac")
            out.append(f"| `{d['path']}` | {vs}{'' if d.get('write_value', True) else ' (not written)'} | {ps} "
                       f"({d['previous_provenance']}) | {d['provenance']} | {hs} | "
                       f"{'-' if wf is None else f'{wf:.2f}'} | {idt.get('rule', '')} |")
            for an, a in (d.get("alternatives") or {}).items():
                out.append(f"| &nbsp;&nbsp;alt `{an}` | {a['value']:.5g} | | {a.get('provenance')} | | | |")
        out.append("")
    return "\n".join(out)
