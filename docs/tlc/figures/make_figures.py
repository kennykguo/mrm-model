"""Static figures for docs/tlc/CLIENT_REPORT.md, from docs/tlc/results.json and the evaluation page's data.json.

    .venv/bin/python docs/tlc/figures/make_figures.py

Palette: the dataviz reference categorical slots 1-4 (validated: adjacent CVD dE 9.1, normal-vision 22.9). Slots 3-4
sit below 3:1 contrast on white, so every figure is paired with a table of its values in the report.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TUNED, FAITHFUL, PI, DITHER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTE, GRID, OK, WARN = "#0b0b0b", "#52514e", "#e4e3df", "#1e7a3c", "#a15c00"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTE,
                     "xtick.color": MUTE, "ytick.color": MUTE, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.dpi": 160, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})


def _runs():
    return json.loads((REPO / "docs/tlc/results.json").read_text())["runs"]


def _pk(r):
    return max(x["peak_err_pm"] for x in r["rings"][1:])


def _lost(r):
    return _pk(r) > 100 and (any(x.get("wrong_line_final") for x in r["rings"][1:])
                             or sum(x["rms_err_pm"] for x in r["rings"][1:]) / 7 > 15)


def scenarios(runs):
    order = [("steady_tlc_faithful", "Steady state"), ("tlc_xpu_step", "Die-temperature ramp"),
             ("tlc_laser_drift", "Laser wavelength drift"), ("tlc_ext_common_aggressor", "Die-wide aggressor"),
             ("tlc_a1", "Local aggressor (A1)"), ("tlc_squelch", "Squelch"), ("tlc_ext_kick", "+10 K thermal kick"),
             ("tlc_power_step", "−3 dB laser power step")]
    series = [("tlc_arm2", "TLC, adapted", TUNED), ("tlc", "TLC, as designed", FAITHFUL),
              ("pi_lock_c", "Reference PI", PI), ("dither_lockin_c", "Reference dither lock-in", DITHER)]
    by = {(r["experiment"], r["controller"]): r for r in runs if not r["corner"]}
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    h = 0.19
    for j, (c, label, color) in enumerate(series):
        ys, xs, wrong = [], [], []
        for i, (e, _) in enumerate(order):
            r = by[(e, c)]
            ys.append(i + (j - 1.5) * h)
            xs.append(_pk(r))
            wrong.append(any(x.get("wrong_line_final") for x in r["rings"][1:]))
        ax.barh(ys, xs, height=h * 0.86, color=color, label=label, zorder=2)
        for y, x, w in zip(ys, xs, wrong):
            if w:
                ax.text(x * 1.12, y, "✕ wrong line", va="center", fontsize=7, color=MUTE)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 60000)
    ax.set_yticks(range(len(order)), [n for _, n in order])
    ax.invert_yaxis()
    for v, c, n, ha, dx in ((5, OK, "5 pm budget ", "right", 0.97), (11.4, WARN, " 11.4 pm (2 GHz)", "left", 1.03)):
        ax.axvline(v, color=c, lw=1, ls="--", zorder=1)
        ax.text(v * dx, len(order) - 0.45, n, color=c, fontsize=7, ha=ha, va="top")
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel("worst peak tracking error, rings 1–7 (pm, log scale)")
    ax.set_title("Peak error by scenario (nominal world)", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=8, handlelength=1.2)
    fig.savefig(HERE / "fig_scenarios.png")
    plt.close(fig)


def traces():
    data = json.loads((REPO / "docs/tlc/evaluation_src/data.json").read_text())
    tr = {(t["exp"], t["ctrl"]): t for t in data["traces"]}
    panels = [(("steady_tlc_faithful", "tlc"), "Steady state · as designed", FAITHFUL),
              (("steady_tlc_faithful", "tlc_arm2"), "Steady state · adapted", TUNED),
              (("tlc_ext_kick", "tlc"), "+10 K kick at 10 ms · as designed", FAITHFUL),
              (("tlc_ext_kick", "tlc_arm2"), "+10 K kick at 10 ms · adapted", TUNED)]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.8))
    for ax, (key, title, color) in zip(axs.flat, panels):
        t = tr[key]
        tm = np.asarray(t["t_ms"])
        for i, e in enumerate(t["err"]):
            ax.plot(tm, e, color=color, lw=0.8, alpha=0.35 + 0.08 * i)
        lim = 12 if key[0] == "steady_tlc_faithful" else 80
        ax.set_ylim(-lim, lim)
        ax.axhspan(-5, 5, color=OK, alpha=0.07, lw=0)
        ax.axvspan(0, t["score_ms"], color=GRID, alpha=0.6, lw=0)
        ax.set_title(title, loc="left", fontsize=9)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("error (pm)")
    fig.suptitle("Tracking error over time, all 8 rings (shaded: ±5 pm; grey: warm-up, not scored)", x=0.01,
                 ha="left", fontsize=9, color=MUTE)
    fig.tight_layout()
    fig.savefig(HERE / "fig_traces.png")
    plt.close(fig)


def cold(runs):
    camp = {(r["experiment"], r["controller"]): r["campaign"]["wrong_line"]["ring_trials"] for r in runs if r["campaign"]}
    rows = [("TLC, as designed (own acquisition)", camp[("tlc_cold_campaign", "tlc")], FAITHFUL),
            ("TLC, adapted (line-finding acquisition)", camp[("tlc_cold_campaign_hybrid", "tlc_wdm")], TUNED),
            ("Reference WDM supervisor, 8 worlds", camp[("tlc_cold_campaign_wdm", "wdm_supervisor")], PI),
            ("Reference WDM supervisor, 300 worlds", [12, 2400, 0.0026, 0.0087], PI)]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    for i, (label, (k, n, lo, hi), color) in enumerate(rows):
        ax.plot([lo * 100, hi * 100], [i, i], color=color, lw=3, solid_capstyle="round")
        ax.plot([k / n * 100], [i], "o", ms=8, color=color, mec="white", mew=1.5)
        ax.text(100, i, f"{k} / {n}", va="center", ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel("ring trials ending on a foreign laser line (%), with 95 % Clopper–Pearson interval")
    ax.set_title("Cold start: wrong-line locking", loc="left")
    fig.savefig(HERE / "fig_cold_start.png")
    plt.close(fig)


def lockin():
    rows = json.loads((REPO / "docs/tlc/lockin_diag.json").read_text())["rows"]
    d = np.array([r["detuning_pm"] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, key, title, lim in ((axs[0], "z1", "z1: slope signal", None), (axs[1], "zeta", "ζ = z2/z1", (-0.02, 0.04))):
        mu = np.array([r[key + "_mean"] for r in rows])
        sd = np.array([r[key + "_std"] for r in rows])
        ax.plot(d, [r[key + "_model"] for r in rows], color=MUTE, lw=1.2, ls="--", label="model")
        ax.errorbar(d, mu, yerr=sd, fmt="o", ms=4, color=FAITHFUL, ecolor=FAITHFUL, elinewidth=1, label="measured ±1σ")
        if lim:
            ax.set_ylim(*lim)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("ring 0 detuning (pm)")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "fig_lockin.png")
    plt.close(fig)


if __name__ == "__main__":
    r = _runs()
    scenarios(r)
    traces()
    cold(r)
    lockin()
    print("wrote", sorted(p.name for p in HERE.glob("*.png")))
