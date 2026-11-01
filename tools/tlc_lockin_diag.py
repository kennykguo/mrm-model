"""Open-loop lock-in diagnostic for the TLC on G26 (spec Phase 3; informational, not a pass/fail gate).

    .venv/bin/python tools/tlc_lockin_diag.py [--ring 0] [--sweep-du 0.02] [--period-s 0.04] [--duration-s 0.08]
                                              [--out docs/tlc/lockin_diag.md]

The faithful core runs as usual (dither, B2 conditioner, B3 lock-in), but the heater's DC level does not follow
its control law: it is held at the warm-start command plus a slow sine sweep of ``sweep_du`` (power fraction), so
the true detuning walks across the red flank. The dither stays in the stream. Each core step logs the lock-in
outputs z1 = y1/y0, z2 = y2/y0 and zeta = z2/z1 with their validity; the true detuning comes from the plant trace,
averaged with the lock-in's own Hann window (4 ms) so both describe the same interval.

The report bins the true detuning and compares the measured z1, z2, zeta (mean and spread per bin) with the
source's model ``optics.lockin_ratios`` at the core's priors and dither amplitude: how well the source's
measurement model and its SNR carry over to G26.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import mrm.params as P
from mrm.control.api import ControllerSpec
from mrm.control.tlc import controller as tlc_controller
from mrm.control.tlc.controller import TlcController
from mrm.control.tlc.core.optics import lockin_ratios
from mrm.control.tlc.faithful_core import FaithfulCore
from mrm.control.tlc.units import pm_per_ghz
from mrm.sim import runner
from mrm.sim.metrics import RecordSpec

TIER = "controller_interface.knowledge=[line_grid,ring_to_line,model_nominal]"
REC_DEC_TICKS = 100           # one detuning sample per 100 us core step


class SweepProbeCore(FaithfulCore):
    """FaithfulCore whose output DC is the warm command plus a slow sine; logs the lock-in outputs."""

    def __init__(self, spec, *, ts, n_out, sweep_du, period_s):
        super().__init__(spec, ts=ts, n_out=n_out, kind="tlc")
        self.sweep_du, self.period_s, self.history = sweep_du, period_s, []

    def warm_start(self, u0):
        super().warm_start(u0)
        self.u0 = np.asarray(u0, dtype=np.float64).copy()

    def step(self, pd, t_block_start):
        seq = super().step(pd, t_block_start)
        t_next = t_block_start + self.ctl.cp.ts
        u_dc = self.u0 + self.sweep_du * np.sin(2 * np.pi * t_next / self.period_s)
        L = self.ctl.last
        self.history.append({"t": t_next, "z1": np.array(L["z1"]), "z2": np.array(L["z2"]),
                             "valid": np.array(L["valid"]), "amp": np.array(L["amp"])})
        return np.clip(seq - self.ctl.u_cmd[None, :] + u_dc[None, :], 0.0, 1.0)


def run_probe(sweep_du: float, period_s: float, duration_s: float) -> tuple[SweepProbeCore, dict]:
    tlc_controller.CORES["sweep_probe"] = lambda spec, *, ts, n_out, params: SweepProbeCore(
        spec, ts=ts, n_out=n_out, sweep_du=sweep_du, period_s=period_s)
    cfg = P.load("g26", overrides=[TIER])
    ctrl = TlcController(core="sweep_probe")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = runner.run(cfg, ctrl, duration_s=duration_s, batch=1, uncertainty="nominal", variation="off",
                         heater_code="quasi_static_reachable", settle=True, path="python",
                         record=[RecordSpec("detuning_m", REC_DEC_TICKS, "mean")])
    tr = res.trace[f"detuning_m@{REC_DEC_TICKS}:mean"]
    return ctrl.cores[0], {"t_s": np.asarray(tr["t_s"]), "detuning_pm": np.asarray(tr["value"])[:, 0, :] * 1e12}


def window_average(t_trace, x, t_eval, window_s: float) -> np.ndarray:
    """Hann-weighted mean of x over [t - window, t) at each t_eval (the lock-in's view of a slow signal)."""
    out = np.full((len(t_eval),) + x.shape[1:], np.nan)
    for k, t in enumerate(t_eval):
        sel = (t_trace > t - window_s) & (t_trace <= t)
        if sel.sum() >= 3:
            w = np.hanning(sel.sum() + 2)[1:-1]
            out[k] = (w[:, None] * x[sel]).sum(0) / w.sum() if x.ndim > 1 else (w * x[sel]).sum() / w.sum()
    return out


def analyse(core: SweepProbeCore, trace: dict, ring: int, n_bins: int = 12) -> dict:
    pr = core.priors
    h = core.history
    t = np.array([r["t"] for r in h])
    z1 = np.array([r["z1"][ring] for r in h])
    z2 = np.array([r["z2"][ring] for r in h])
    ok = np.array([r["valid"][ring] for r in h])
    amp = np.array([r["amp"][ring] for r in h])
    d_pm = window_average(trace["t_s"], trace["detuning_pm"][:, ring], t, pr.cp.lockin_window)
    keep = ok & np.isfinite(d_pm)
    edges = np.linspace(np.nanmin(d_pm[keep]), np.nanmax(d_pm[keep]), n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = keep & (d_pm >= lo) & (d_pm < hi)
        if sel.sum() < 3:
            continue
        dc = 0.5 * (lo + hi)
        rows.append({"detuning_pm": dc, "n": int(sel.sum()),
                     "z1_mean": float(z1[sel].mean()), "z1_std": float(z1[sel].std()),
                     "z2_mean": float(z2[sel].mean()), "z2_std": float(z2[sel].std()),
                     "zeta_mean": float((z2[sel] / z1[sel]).mean()), "zeta_std": float((z2[sel] / z1[sel]).std()),
                     "amp_ghz": float(amp[sel].mean())})
    return {"ring": ring, "valid_fraction": float(ok.mean()), "rows": rows, "n_steps": int(len(t))}


def add_model(summary: dict, core: SweepProbeCore, pm_per_ghz: float) -> dict:
    pp, i = core.priors.pp, summary["ring"]
    for r in summary["rows"]:
        z1m, z2m = lockin_ratios(r["detuning_pm"] / pm_per_ghz, r["amp_ghz"], pp.hwhm[i], pp.mod_shift)
        r.update(z1_model=float(z1m), z2_model=float(z2m), zeta_model=float(z2m / z1m))
    return summary


def to_markdown(s: dict, args) -> str:
    cols = ["detuning_pm", "n", "z1_mean", "z1_std", "z1_model", "z2_mean", "z2_std", "z2_model", "zeta_mean",
            "zeta_std", "zeta_model"]
    out = ["# TLC lock-in diagnostic on G26 (open loop)", "",
           f"Generated by `tools/tlc_lockin_diag.py`: ring {s['ring']}, heater DC = warm command + "
           f"{args.sweep_du} sin(2 pi t / {args.period_s} s), {args.duration_s} s, {s['n_steps']} core steps, "
           f"lock-in valid {100 * s['valid_fraction']:.0f} % of steps. True detuning: plant trace averaged with the "
           f"4 ms Hann window. Model: the source's `optics.lockin_ratios` at the G26 priors and the scheduled "
           f"dither amplitude.", "",
           "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |"
            for r in s["rows"]]
    q = s["quality"]
    out += ["", "## Reading", "",
            f"- z1 (slope channel): measured / model = {q['z1_ratio_median']:.2f} (median over bins); "
            f"SNR |z1| / std = {q['z1_snr_median']:.0f}.",
            f"- zeta = z2 / z1 (the UKF's power-invariant detuning measurement): model span over the sweep "
            f"{q['zeta_model_span']:.3g}; measured spread per bin (median) {q['zeta_std_median']:.3g}, so the "
            f"span is {q['zeta_span_over_noise']:.1f}x the per-step noise; worst bias |measured - model| = "
            f"{q['zeta_bias_max']:.3g}.", ""]
    return "\n".join(out) + "\n"


def quality(s: dict) -> dict:
    r = s["rows"]
    z1_ratio = np.median([x["z1_mean"] / x["z1_model"] for x in r])
    zeta_model = np.array([x["zeta_model"] for x in r])
    zeta_std = np.median([x["zeta_std"] for x in r])
    return {"z1_ratio_median": float(z1_ratio),
            "z1_snr_median": float(np.median([abs(x["z1_mean"]) / x["z1_std"] for x in r])),
            "zeta_model_span": float(np.ptp(zeta_model)), "zeta_std_median": float(zeta_std),
            "zeta_span_over_noise": float(np.ptp(zeta_model) / zeta_std),
            "zeta_bias_max": float(max(abs(x["zeta_mean"] - x["zeta_model"]) for x in r))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ring", type=int, default=0)
    ap.add_argument("--sweep-du", type=float, default=0.02)
    ap.add_argument("--period-s", type=float, default=0.04)
    ap.add_argument("--duration-s", type=float, default=0.08)
    ap.add_argument("--out", default="docs/tlc/lockin_diag.md")
    args = ap.parse_args(argv)
    core, trace = run_probe(args.sweep_du, args.period_s, args.duration_s)
    cfg = P.load("g26")
    spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
    ppg = float(pm_per_ghz(np.asarray(spec.line_grid_hz)[np.asarray(spec.ring_to_line)])[args.ring])
    s = add_model(analyse(core, trace, args.ring), core, ppg)
    s["quality"] = quality(s)
    out = Path(args.out)
    out.write_text(to_markdown(s, args))
    out.with_suffix(".json").write_text(json.dumps(s, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
