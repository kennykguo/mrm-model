"""Derived numbers of the webviz site that need the engine run itself (imported by generate.py).

Read this first. generate.py calls these functions with its run record R (see generate.run_scenario):
  stats_records()          the extra 10 us records that every scenario run carries (STATS_RECORDS)
  stats_json(R, ...)       stats_<id>.json: statistics, time in each state, counts and the limits reached
  xt_index, xt_json        the thermal-crosstalk split of each ring and PD temperature by heat source
  coupling_json(sysm)      the thermal system as tables: DC gains, lags and step-response times
  inputs_json(R, S)        every scenario input signal, reduced over the frames
  controller_timing(R)     the tick timing of the controller, the ADC and the DAC
  hardware_limits(R0, ...) the hardware limits the model assumes, with value, unit and provenance
  describe(R, ...)         the plain paragraph that describes each scenario
Nothing here adds physics. Every number comes from the engine (records, the device tap, Truth snapshots, the
thermal system's own lags) or from the model's own laws (the heater driver law at full code). The bases:

  statistics   one sample per 10 us window (SAMPLE ticks, one controller step): the engine's RecordSpec records at
               that decimation (window means over every 1 us tick; min / max reducers where the name says so) and
               the device tap's kernel state at each controller edge. Extremes of the detuning and of the monitor
               code use the min / max reducers, so they are the tick extremes. The rms and the percentiles are of
               the 10 us values (a window mean removes some tick-to-tick noise).
  time in a state   number of 10 us samples in that state x 10 us; the engine's own per-tick metrics are given
               next to them where the engine has one (time unlocked, at a rail, on a foreign line).
  limits       margin = distance from the limit (in the limit's unit) at every 10 us sample; the smallest margin
               (negative once the limit is passed), its time, the time at the limit and the first time the limit
               was reached.
  crosstalk    the engine's thermal lag states (each observer = base + the sum of its lags) grouped by heat source,
               read at every controller edge and averaged over each frame; the DC gains and the lags (poles and
               residues) of the model's thermal system.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from mrm.plant import electrical
from mrm.plant import thermal
from mrm.scenarios import generators
from mrm.sim.metrics import RecordSpec

#: one sample every SAMPLE ticks (10 ticks of 1 us = 10 us: one g26 controller period); generate.py uses this value
SAMPLE = 10


# --------------------------------------------------------------------------------------------------------- JSON helpers
# The same helpers as in generate.py (js: rounded nested list, jint: integers, jnum: one number; NaN -> null).
# Two differences: jint here always gives floats (12.0, not 12), and jnum here turns a bool into 1.0 / 0.0
# (generate.py keeps a bool).
def round_sig(a, sig: int = 4, dec: int | None = None) -> np.ndarray:
    """Round to `sig` significant digits (and to `dec` decimals when given, whichever is coarser)."""
    x = np.array(a, dtype=float)
    fin = np.isfinite(x) & (x != 0.0)
    if np.any(fin):
        mag = np.floor(np.log10(np.abs(x[fin])))
        f = 10.0 ** (sig - 1 - mag)
        x[fin] = np.round(x[fin] * f) / f
    if dec is not None:
        x[np.isfinite(x)] = np.round(x[np.isfinite(x)], dec)
    return x


def _nan_to_null(o):
    """Nested lists with every non-finite float replaced by None (JSON null)."""
    if isinstance(o, list):
        return [_nan_to_null(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def js(a, sig: int = 4, dec: int | None = None):
    """JSON-ready nested lists, rounded to `sig` significant digits (and `dec` decimals), NaN -> null."""
    return _nan_to_null(round_sig(a, sig, dec).tolist())


def jint(a):
    """JSON-ready nested lists of whole numbers (rounded; always floats), NaN and inf -> null."""
    a = np.asarray(a, float)
    return _nan_to_null(np.where(np.isfinite(a), np.rint(a), np.nan).tolist()) if a.dtype.kind == "f" else a.tolist()


def jnum(v, sig=6):
    """One JSON-ready number rounded to `sig` digits; None, NaN and inf -> None."""
    if v is None:
        return None
    v = float(v)
    if not math.isfinite(v):
        return None
    return float(round_sig(v, sig))


# --------------------------------------------------------------------------------------------------- statistics records
#: Truth fields recorded every SAMPLE ticks for the statistics: (field, reducer over the 10 us window)
STATS_RECORDS = (
    ("detuning_m", "mean"), ("detuning_m", "min"), ("detuning_m", "max"),
    ("t_ring_k", "mean"), ("t_heater_k", "mean"), ("t_pd_k", "mean"), ("t_base_k", "mean"),
    ("p_abs_w", "mean"), ("p_in_w", "mean"), ("p_drop_w", "mean"), ("i_monitor_a", "mean"),
    ("monitor_code", "last"), ("monitor_code", "min"), ("monitor_code", "max"),
    ("reloma", "mean"), ("oracle_reloma", "mean"), ("p_heater_w", "mean"),
    ("ber", "mean"), ("oma_w", "mean"), ("er_db", "mean"), ("il_db", "mean"), ("bias_v", "mean"),
    ("line_on", "last"), ("declared_line", "last"),
)


def stats_records():
    """The RecordSpec list of STATS_RECORDS: one window every SAMPLE ticks."""
    return [RecordSpec(f, SAMPLE, red) for f, red in STATS_RECORDS]


#: the statistics of every quantity (p5 / p50 / p95: percentiles; std: standard deviation about the mean)
STATS = ("min", "p5", "p50", "mean", "p95", "max", "rms", "std")


def _statistics(x, lo=None, hi=None, lin_db=False, no_spread=False):
    """Statistics over axis 0 of x (T, C). lo / hi: arrays whose min / max give the extremes (tick envelopes).
    lin_db: x is linear, the result is in dB (10 log10) except rms and std (not given). Returns {stat: (C,)}."""
    x = np.asarray(x, float)
    if x.shape[0] == 0:
        return {k: np.full(x.shape[1:], np.nan) for k in STATS}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p = np.nanpercentile(x, (5, 50, 95), axis=0)
        out = {"min": np.nanmin(x if lo is None else lo, axis=0), "max": np.nanmax(x if hi is None else hi, axis=0),
               "p5": p[0], "p50": p[1], "p95": p[2], "mean": np.nanmean(x, axis=0),
               "rms": np.sqrt(np.nanmean(x ** 2, axis=0)), "std": np.nanstd(x, axis=0)}
    if lin_db:
        for k in ("min", "max", "p5", "p50", "p95", "mean"):
            out[k] = 10 * np.log10(np.maximum(out[k], 1e-300))
    if lin_db or no_spread:
        out["rms"] = np.full_like(out["mean"], np.nan)
        out["std"] = np.full_like(out["mean"], np.nan)
    return out


def _pack_stats(st, sig=5):
    """The statistics dict as JSON lists rounded to `sig` digits."""
    return {k: js(v, sig) for k, v in st.items()}


class _SampleSeries:
    """The 10 us series of one run: records at SAMPLE ticks and the device rows aligned with them.

    t_end (T,): window end times (s). rec: {(field, reducer): (T, N) values}. dev: {device-row key: (T, ...)}.
    reg_prev (T, N): the heater register one sample earlier. sf: scoring start (s). scored (T,): windows entirely
    after the scoring start. win: {"run": all samples, "scored": scored samples} (boolean masks)."""

    def __init__(self, R):
        tr = R["cr"].trace
        self.dt = R["dt"]
        key = RecordSpec("t_ring_k", SAMPLE, "mean").key
        self.t_end = np.asarray(tr[key]["t_s"], float)                      # window ends (s)
        self.T = self.t_end.size
        self.rec = {(f, red): np.asarray(tr[RecordSpec(f, SAMPLE, red).key]["value"], float)[:, 0, :]
                    for f, red in STATS_RECORDS}
        plant = R["plant"]
        u_t = np.asarray(plant.u_ticks)
        if u_t[0] != 0 or not np.all(np.diff(u_t) == SAMPLE) or len(plant.dev) < 1 + self.T:
            raise RuntimeError("device rows do not line up with the 10 us statistics records")
        rows = plant.dev[1:1 + self.T]          # row j is the state at the end of record window j
        self.dev = {k: np.array([r[k] for r in rows], float) for k in rows[0] if k != "sat"}
        self.dev["sat"] = np.array([r["sat"] for r in rows], float)
        prev = plant.dev[0:self.T]
        self.reg_prev = np.array([r["he_reg"] for r in prev], float)
        sf = float(R["cr"].manifest_inputs.get("score_from_s") or 0.0)
        self.sf = sf
        self.scored = (self.t_end - SAMPLE * self.dt) >= sf - 1e-12          # windows entirely after the start
        self.win = {"run": np.ones(self.T, bool), "scored": self.scored}

    def record(self, f, red="mean"):
        """The (T, N) values of Truth field f with reducer red."""
        return self.rec[(f, red)]


def _quantity_entry(key, part, label, unit, note, x, per="ring", lo=None, hi=None, lin_db=False, no_spread=False,
                    basis="records", win=None, sig=6):
    """One row of stats_<id>.json "quantities": the statistics of x (T, C) for every window of win.

    lo / hi: the arrays whose min / max are the extremes (tick envelopes). per: "ring", "line" or "bank".
    basis: "records" (engine records) or "device" (the tap's kernel state). Rows per ring or line also get the
    statistics of all columns pooled ("<window>_pooled")."""
    out = {"key": key, "part": part, "label": label, "unit": unit, "note": note, "per": per, "basis": basis}
    for w, sel in win.items():
        out[w] = _pack_stats(_statistics(x[sel], None if lo is None else lo[sel], None if hi is None else hi[sel],
                                         lin_db, no_spread), sig)
        if per in ("ring", "line"):
            flat = x[sel].reshape(-1, 1)
            pl = _statistics(flat, None if lo is None else lo[sel].reshape(-1, 1),
                             None if hi is None else hi[sel].reshape(-1, 1), lin_db, no_spread)
            out[w + "_pooled"] = {k: jnum(v[0], sig) for k, v in pl.items()}
    return out


def heater_full_power_w(plant, t_heater_k):
    """The heater driver law at code full scale and the given heater-resistor temperatures (T, N), no further
    self-heating within the tick (g_self = 0): the most power the heater can give at that temperature.
    t_heater_k in K; returns W, same shape."""
    hp = plant.heater_params
    N = plant.N
    law = int(np.ravel(hp.law)[0])
    # a per-ring parameter as a contiguous (N,) float64 row: its last N values, or its one value repeated
    row = lambda a: np.ascontiguousarray(np.broadcast_to(np.asarray(a, float).reshape(-1)[-N:] if np.size(a) >= N  # noqa: E731
                                                          else np.asarray(a, float).reshape(-1)[:1], (N,)), np.float64)
    args = [row(hp.r_cold_ohm), row(hp.tcr_per_k), row(hp.tcr_ref_k), row(hp.v_full_scale_v),
            row(hp.i_full_scale_a), row(hp.p_full_scale_w), row(hp.compliance_v)]
    t = np.atleast_2d(np.asarray(t_heater_k, float))
    out = np.zeros_like(t)
    tmp = [np.zeros(N) for _ in range(4)]
    fl = np.zeros(N, np.int8)
    for f in range(t.shape[0]):
        electrical.heater_power_row(np.ones(N), law, np.ascontiguousarray(t[f]), np.zeros(N), *args,
                                    tmp[0], tmp[1], tmp[2], tmp[3], fl)
        out[f] = tmp[0]
    return out


def stats_json(R, sj, devj, specj, checks):
    """stats_<id>.json: statistics, time in each state, counts and the limits reached, from the 10 us series."""
    sd, plant = R["sd"], R["plant"]
    m = R["cr"].metrics
    series = _SampleSeries(R)
    win = series.win
    T, dt = series.T, series.dt
    N, L = plant.N, plant.L
    hp, mh, lhw, dph = plant.heater_params, plant.monitor_hw, plant.laser_hw, plant.datapath_hw
    code_lo, code_hi = int(hp.code_lo), int(hp.code_hi)
    cmax = int(mh.code_max)
    ref = R["cr"].reference                                  # full precision (the JSON copy is rounded)
    d_ref = np.asarray(ref["d_ref_pm"], float)[0] if ref is not None else np.zeros(N)
    thr = float(sj["summary"]["lock_threshold_pm"])
    ber_limit = float(sj["summary"]["ber_limit"])
    link_t = float(np.ravel(dph.link_t)[0])
    sens = np.asarray(dph.oma_sens_w, float).reshape(-1)[-N:] if np.size(dph.oma_sens_w) >= N else \
        np.full(N, float(np.ravel(dph.oma_sens_w)[0]))
    K0 = 273.15
    det = series.record("detuning_m") * 1e12
    det_lo, det_hi = series.record("detuning_m", "min") * 1e12, series.record("detuning_m", "max") * 1e12
    err, err_lo, err_hi = det - d_ref, det_lo - d_ref, det_hi - d_ref
    abs_env = np.maximum(np.abs(err_lo), np.abs(err_hi))
    reg = series.dev["he_reg"]
    p_h = series.record("p_heater_w") * 1e3
    t_h = series.record("t_heater_k")
    p_full = heater_full_power_w(plant, t_h) * 1e3
    rel, orc = series.record("reloma"), series.record("oracle_reloma")
    rel_db = 10 * np.log10(np.maximum(rel, 1e-300))
    orc_db = 10 * np.log10(np.maximum(orc, 1e-300))
    mon = series.record("monitor_code", "last")
    mon_lo, mon_hi = series.record("monitor_code", "min"), series.record("monitor_code", "max")
    f0 = np.asarray(lhw.f0_hz, float)[0]
    bank_p = series.dev["he_p"].sum(axis=1, keepdims=True) * 1e3
    bank_i = series.dev["he_i"].sum(axis=1, keepdims=True) * 1e3
    pmax_mw = float(np.ravel(hp.supply_power_max_w)[0]) * 1e3
    imax_ma = float(np.ravel(hp.supply_current_max_a)[0]) * 1e3
    # add_quantity(key, part, label, unit, note, x, **options): one row of "quantities" (see _quantity_entry)
    quantities = []
    add_quantity = lambda *a, **k: quantities.append(_quantity_entry(*a, win=win, **k))                     # noqa: E731
    # ---- ring
    add_quantity("det", "ring", "Detuning (true)", "pm", "Resonance minus the declared line; + is red of the line.",
                 det, lo=det_lo, hi=det_hi)
    add_quantity("err", "ring", "Lock error", "pm", "Detuning minus the lock reference.", err, lo=err_lo, hi=err_hi)
    add_quantity("abs_err", "ring", "|Lock error|", "pm", "Size of the lock error; max is the tick extreme.",
                 np.abs(err), lo=np.zeros_like(err), hi=abs_env)
    add_quantity("t_ring", "ring", "Ring temperature", "°C", "The mode-weighted ring observer.",
                 series.record("t_ring_k") - K0, sig=7)
    add_quantity("p_abs", "ring", "Absorbed light", "mW",
                 "Optical power absorbed in the ring, all lines; it heats the ring.", series.record("p_abs_w") * 1e3)
    add_quantity("p_in", "ring", "Light in on its line", "mW", "Power of the declared line at the ring input.",
                 series.record("p_in_w") * 1e3)
    add_quantity("reloma", "ring", "relOMA", "dB", "OMA / P_in at the through port. Mean: 10 log10 of the linear mean.",
                 rel, lin_db=True)
    add_quantity("oracle", "ring", "Oracle relOMA", "dB", "The best stable relOMA at the same instant.", orc,
                 lin_db=True)
    add_quantity("oma_lost", "ring", "OMA lost to the oracle", "dB", "Oracle relOMA minus relOMA, per 10 µs sample.",
                 orc_db - rel_db)
    # ---- heater
    add_quantity("code", "heater", "Heater code (register)", "code",
                 f"Latched DAC register; rails {code_lo} and {code_hi}.", reg, basis="device")
    add_quantity("headroom", "heater", "Headroom to the nearest rail", "code", "The smaller of (code − low rail) and "
                 "(high rail − code).", np.minimum(reg - code_lo, code_hi - reg), basis="device")
    add_quantity("v", "heater", "Heater voltage", "V", "Driver output voltage.", series.dev["he_v"], basis="device")
    add_quantity("i", "heater", "Heater current", "mA", "I = V / R.", series.dev["he_i"] * 1e3, basis="device")
    add_quantity("r", "heater", "Heater resistance", "Ω", "Changes with the heater temperature (TCR).",
                 series.dev["he_r"], basis="device")
    add_quantity("p", "heater", "Heater power", "mW", "Electrical power into the heater.", p_h)
    add_quantity("p_left", "heater", "Power left at full code", "mW",
                 "The driver law at full code and the present heater temperature, minus the power now.",
                 p_full - p_h)
    add_quantity("t_heater", "heater", "Heater temperature", "°C", "The heater-resistor observer.", t_h - K0, sig=7)
    add_quantity("step", "heater", "Code step per controller step", "code",
                 "|register change| from one controller step to the next.", np.abs(reg - series.reg_prev),
                 basis="device")
    add_quantity("bank_p", "heater", "Bank heater power", "mW",
                 f"Sum of the {N} heaters (supply limit {pmax_mw:g} mW).", bank_p, per="bank", basis="device")
    add_quantity("bank_i", "heater", "Bank heater current", "mA", f"Sum of the {N} heater currents (supply limit "
                 f"{imax_ma:g} mA).", bank_i, per="bank", basis="device")
    # ---- monitor PD and ADC
    add_quantity("p_drop", "pd", "Drop power into the PD", "µW", "All lines and orders, averaged over the data.",
                 series.record("p_drop_w") * 1e6)
    add_quantity("i_mon", "pd", "PD current (signal + dark)", "µA", "The current that the ADC integrates.",
                 series.record("i_monitor_a") * 1e6)
    add_quantity("i_dark", "pd", "Dark current", "nA", "Grows with the PD temperature.", series.dev["i_dark"] * 1e9,
                 basis="device")
    add_quantity("t_pd", "pd", "PD temperature", "°C", "The PD thermal observer.", series.record("t_pd_k") - K0, sig=7)
    add_quantity("mon", "pd", "ADC code", "code",
                 f"Latest released code; full scale {cmax}. Min and max: tick extremes.", mon, lo=mon_lo, hi=mon_hi)
    add_quantity("mon_left", "pd", "ADC headroom to full scale", "code",
                 "Full scale minus the highest code in each 10 µs.", cmax - mon_hi, lo=cmax - mon_hi, hi=cmax - mon_hi)
    # ---- laser, WDM and bus (per line)
    add_quantity("p_chip", "laser", "Light into the chip", "mW", "Per laser line, after the chip input loss.",
                 series.dev["lf_pchip"] * 1e3, per="line", basis="device", sig=6)
    add_quantity("p_laser", "laser", "Laser output", "mW", "Per laser line, at the laser module.",
                 series.dev["lf_p"] * 1e3, per="line", basis="device", sig=6)
    add_quantity("f_dev", "laser", "Frequency offset from the grid", "MHz", "Per laser line (0 in the nominal world).",
                 (series.dev["lf_f"] - f0) * 1e-6, per="line", basis="device")
    # ---- controller I/O
    add_quantity("cmd", "ctrl", "Heater command", "code", "Controller output after the channel clip.",
                 series.dev["he_cmd"], basis="device")
    add_quantity("cmd_dac", "ctrl", "|Command − DAC code|", "code", "The DAC lags the command by its latency.",
                 np.abs(series.dev["he_cmd"] - series.dev["he_code"]), basis="device")
    # ---- data path
    ber = series.record("ber")
    add_quantity("ber", "data", "BER (instantaneous)", "",
                 "BER of the declared line from the bus-output OMA. Mean = the time mean (expected errors / bits).",
                 ber, no_spread=True, sig=4)
    add_quantity("oma_rx", "data", "OMA at the receiver", "µW", f"Bus-output OMA × {link_t:.3g} link transmission.",
                 series.record("oma_w") * link_t * 1e6)
    add_quantity("er", "data", "Extinction ratio", "dB", "At the ring's through port.", series.record("er_db"))
    add_quantity("il", "data", "Insertion loss", "dB", "Of the high level at the through port.", series.record("il_db"))
    add_quantity("bias", "data", "Junction bias", "V", "Mean of the two NRZ levels.", series.record("bias_v"))
    add_quantity("rho", "data", "Ones density", "", "Fraction of ones in the data.", series.dev["rho"], basis="device")

    # ---- time in each state (ms) and counts
    ms_per_sample = SAMPLE * dt * 1e3                                                   # ms per sample
    line_on = series.record("line_on", "last")
    decl = series.record("declared_line", "last")
    foreign = (line_on >= 0) & (line_on != decl)
    offline = line_on < 0
    over = series.dev["mo_over"] > 0
    states = []

    def add_state(key, label, note, mask, engine=None, per="ring"):
        """One row of "states": the time (ms) with mask (T, N) true and the number of entries into it."""
        e = {"key": key, "label": label, "note": note, "per": per}
        for w, sel in win.items():
            e[w] = js(mask[sel].sum(axis=0) * ms_per_sample, 5)
            e[w + "_entries"] = jint(((mask[sel][1:] & ~mask[sel][:-1]).sum(axis=0)
                                      + mask[sel][0].astype(int)).astype(float)) if sel.any() else None
        if engine is not None:
            e["engine_scored"] = js(engine, 5)
        states.append(e)

    metric_row = lambda k: np.asarray(m[k], float)[0] if k in m else None     # noqa: E731
    scored_ms = float(np.sum(series.scored)) * ms_per_sample
    tu = metric_row("time_unlocked_s")
    fl = metric_row("first_lock_s")
    locked_ms = None
    if tu is not None and fl is not None:
        before = np.where(np.isfinite(fl), np.clip(fl * 1e3 - series.sf * 1e3, 0.0, scored_ms), scored_ms)
        locked_ms = scored_ms - tu * 1e3 - before
    states.append({"key": "locked", "label": "Locked (engine)", "per": "ring",
                   "note": "Engine lock state with hysteresis, every tick: scored time minus the time before the first lock "
                           "minus the time unlocked after it.", "run": None, "scored": js(locked_ms, 5)})
    states.append({"key": "unlocked", "label": "Unlocked after the first lock (engine)", "per": "ring",
                   "note": "The engine's time unlocked, every tick, with hysteresis.", "run": None,
                   "scored": js(tu * 1e3, 5) if tu is not None else None})
    add_state("out_band", "Lock error outside the band", f"10 µs means with |lock error| > {thr:g} pm (no hysteresis).",
              np.abs(err) > thr)
    add_state("rail_lo", "Heater at code 0", "Register at the low rail: the heater cannot cool the ring more.",
              reg <= code_lo,
              engine=metric_row("heater_low_rail_s") * 1e3 if metric_row("heater_low_rail_s") is not None else None)
    add_state("rail_hi", "Heater at full scale", "Register at the high rail: the heater cannot heat the ring more.",
              reg >= code_hi,
              engine=metric_row("heater_sat_s") * 1e3 if metric_row("heater_sat_s") is not None else None)
    add_state("adc_clip", "ADC at full scale (clipped)", f"Highest code in the 10 µs at {cmax}.", mon_hi >= cmax)
    add_state("adc_over", "ADC over-range flag", "The ADC flagged its last sample as over its range.", over)
    add_state("adc_zero", "ADC at code 0", "Lowest code in the 10 µs at 0 (no light or an under-range).", mon_lo <= 0)
    add_state("foreign", "On a foreign line", "Within 1 FWHM of a line that is not its declared line.", foreign,
              engine=metric_row("wrong_line_s") * 1e3 if metric_row("wrong_line_s") is not None else None)
    add_state("offline", "On no line", "Not within 1 FWHM of any line.", offline)
    add_state("ber_over", "BER above the limit", f"BER of the 10 µs window > {ber_limit:g}.", ber > ber_limit)
    add_state("supply", "Bank supply limited", "The bank supply limit scaled the heater drive down.",
              (series.dev["he_scale"] < 1.0)[:, None], per="bank")
    counts = []
    for key, label, note in (
            ("excursion_count", "Lock excursions", "Locked → unlocked after the first lock (engine, every tick)."),
            ("excursion_max_s", "Longest excursion (ms)", "Longest unlocked run after the first lock (engine)."),
            ("wrong_line_events", "Foreign-line entries", "Moves onto a foreign line (engine)."),
            ("fsr_hop_events", "FSR hops", "Relocks on another resonance order (engine)."),
            ("order_changes", "Resonance order changes", "Steps at whose end the nearest order changed (engine)."),
            ("ber_windows", "BER windows completed", f"Complete {sj['summary']['ber_sample_period_s']} s BER "
                                                     "windows (engine); 0 when the run is shorter.")):
        v = metric_row(key)
        if v is not None:
            counts.append({"key": key, "label": label, "note": note, "scored": js(v * (1e3 if key.endswith("_s") else 1), 5)})
    sat = sj["summary"]["saturation_counts"]["tx_heater"]
    counts.append({"key": "clipped_cmds", "label": "Clipped heater commands", "note": "Commands outside the code "
                   "range, clipped (engine clip counter, whole run).", "run": sat})
    ov = over.astype(float)
    counts.append({"key": "adc_over_n", "label": "ADC over-range samples", "note": "Controller steps whose last ADC "
                   "sample was flagged over-range.", "run": jint(ov.sum(axis=0)),
                   "scored": jint(ov[series.scored].sum(axis=0))})

    # ---- limits: margin to each hardware limit at every 10 us sample
    limits = []

    def add_limit(key, label, unit, value, margin, at, note, per="ring", t=None, sig=5):
        """One row of "limits": the smallest margin (T, N) and when it occurred (negative once the limit is
        passed), the time at the limit (at: T x N booleans) and the first time it was reached. t: sample times (s)
        when not the 10 us series."""
        key_frames = t is not None
        t = series.t_end if t is None else np.asarray(t, float)
        wins = win if not key_frames else {"run": np.ones(t.size, bool), "scored": t >= series.sf - 1e-12}
        e = {"key": key, "label": label, "unit": unit, "value": value, "note": note, "per": per,
             "basis": "key frames" if key_frames else "10 us samples"}
        mg = np.asarray(margin, float)
        at = np.asarray(at, bool)
        for w, sel in wins.items():
            if not sel.any():
                e[w] = None
                continue
            mm = mg[sel]
            i = np.nanargmin(mm, axis=0)
            hit = at[sel].any(axis=0)
            first = np.where(hit, t[sel][np.argmax(at[sel], axis=0)], np.nan)
            e[w] = {"margin_min": js(np.nanmin(mm, axis=0), sig), "t_margin_min_ms": js(t[sel][i] * 1e3, 6),
                    "time_at_ms": None if key_frames else js(at[sel].sum(axis=0) * ms_per_sample, 5),
                    "reached": [bool(v) for v in hit], "first_ms": js(first * 1e3, 6)}
        limits.append(e)
    add_limit("heater_lo", "Heater code at the low rail", "code", code_lo, reg - code_lo, reg <= code_lo,
              "Margin = code − low rail. At the rail the heater is off and cannot cool the ring further.")
    add_limit("heater_hi", "Heater code at the high rail", "code", code_hi, code_hi - reg, reg >= code_hi,
              "Margin = high rail − code. At the rail the heater cannot heat the ring further.")
    add_limit("heater_p", "Heater power at full code", "mW", None, p_full - p_h, reg >= code_hi,
              "Margin = power at full code (at the present heater temperature) − power now.")
    add_limit("compliance", "Heater driver compliance", "V", float(np.ravel(hp.compliance_v)[0]),
              float(np.ravel(hp.compliance_v)[0]) - series.dev["he_v"],
              series.dev["he_v"] >= float(np.ravel(hp.compliance_v)[0]),
              "Margin = compliance − heater voltage. The voltage DAC full scale is below the compliance.")
    add_limit("bank_p", "Bank supply power", "mW", pmax_mw, pmax_mw - bank_p, (series.dev["he_scale"] < 1.0)[:, None] |
              (bank_p >= pmax_mw), "Margin = supply limit − sum of the heater powers.", per="bank")
    add_limit("bank_i", "Bank supply current", "mA", imax_ma, imax_ma - bank_i, bank_i >= imax_ma,
              "Margin = supply limit − sum of the heater currents.", per="bank")
    add_limit("adc_fs", "ADC full scale", "code", cmax, cmax - mon_hi, mon_hi >= cmax,
              "Margin = full scale − the highest code (tick extreme). At full scale the code clips.")
    add_limit("adc_zero", "ADC code 0", "code", 0, mon_lo, mon_lo <= 0, "Margin = the lowest code (tick extreme).")
    bias = series.record("bias_v")
    add_limit("lock_band", "Lock band", "pm", thr, thr - abs_env, np.abs(err) > thr,
              "Margin = band − |lock error| (tick extreme). Time at the limit: 10 µs means outside the band.")
    add_limit("ber", "BER limit", "decades", ber_limit,
              np.log10(ber_limit) - np.log10(np.maximum(ber, 1e-300)), ber > ber_limit,
              "Margin = log10(limit) − log10(BER): decades below the limit (BER below 1e-300 counts as 1e-300).", sig=4)
    key_data = devj["key"]
    lg = np.array(key_data["selfheat_loop_gain"], float)
    add_limit("selfheat", "Self-heating stability (loop gain 1)", "", 1.0, 1.0 - lg, lg >= 1.0,
              "Margin = 1 − loop gain at the key frames only. At 1 the ring's own heating runs away.",
              t=np.array(key_data["t_s"], float))
    # bias range: the junction bias set point against the bias DAC range (constant in these runs)
    rng = R.get("bias_range_v")
    if rng is not None:
        add_limit("bias_lo", "Bias DAC low end", "V", rng[0], bias - rng[0], bias <= rng[0],
                  "Margin = junction bias − bias DAC minimum.")
        add_limit("bias_hi", "Bias DAC high end", "V", rng[1], rng[1] - bias, bias >= rng[1],
                  "Margin = bias DAC maximum − junction bias.")

    # ---- junction levels and the resonance shift between them (key frames)
    lv = specj["levels"]
    resl = np.array(specj["resonance_level_nm"], float)
    junction = {"v0_v": js(np.median(np.array(lv["v_lo_v"], float), axis=0), 4),
                "v1_v": js(np.median(np.array(lv["v_hi_v"], float), axis=0), 4),
                "shift_pm": js(np.median((resl[:, :, 1] - resl[:, :, 0]) * 1e3, axis=0), 4),
                "note": "median over the key frames: the junction voltage of NRZ level 0 and level 1 (Truth), and "
                        "the exact resonance of level 1 minus that of level 0 (phase model, exact ring)"}

    # ---- checks of the new numbers against the engine
    c = checks[sd["id"]].setdefault("stats", {})
    mx = np.asarray(m["max_abs_lock_error_pm"], float)[0]
    c["max_abs_lock_error_from_10us_envelopes_minus_metric_pm"] = jnum(
        np.max(np.abs(abs_env[series.scored].max(axis=0) - mx)), 4)
    if metric_row("heater_low_rail_s") is not None:
        c["rail_lo_time_10us_minus_engine_ms"] = jnum(np.max(np.abs(
            (reg[series.scored] <= code_lo).sum(axis=0) * ms_per_sample - metric_row("heater_low_rail_s") * 1e3)), 4)
    bm = metric_row("ber_mean")
    if bm is not None:
        mean_ber = ber[series.scored].mean(axis=0)
        ok = np.isfinite(bm) & (bm > 0)
        c["ber_mean_10us_vs_engine_rel_max"] = jnum(np.max(np.abs(mean_ber[ok] - bm[ok]) / bm[ok]), 4) if ok.any() else None
    c["samples"] = int(T)
    return {
        "id": sd["id"], "sample_s": SAMPLE * dt, "samples": int(T), "score_from_s": jnum(series.sf),
        "windows": {"run": [0.0, jnum(series.t_end[-1])], "scored": [jnum(series.sf), jnum(series.t_end[-1])]},
        "stats": list(STATS), "quantities": quantities, "states": states, "counts": counts, "limits": limits,
        "junction": junction,
        "note": ("statistics over 10 µs samples (one per controller step): window means of the engine's records (every "
                 "1 µs tick) and the kernel state read at each controller edge ('device'). min / max of the detuning, "
                 "the lock error and the ADC code are tick extremes (min / max reducers). The rms and the percentiles "
                 "are of the 10 µs values. 'scored': from the scoring start; '_pooled': all rings (or lines) "
                 "together. Times are in ms."),
    }


# ------------------------------------------------------------------------------------------------------------ crosstalk
#: heat-source groups of the crosstalk split, seen from one ring: its own heater, own absorbed light, own junction
#: and driver ("jd"), the same three of the other rings, and the aggressor sites
XT_GROUPS = ("own_heater", "own_light", "own_jd", "other_heaters", "other_light", "other_jd", "aggressor")
_XT_OTHER_OFFSET = 3                             # XT_GROUPS index of other_* = index of own_* + 3
_XT_AGGRESSOR = XT_GROUPS.index("aggressor")     # 6


def xt_index(sysm):
    """Index arrays that sum the lag states of the ring and PD observers by heat source in one bincount.

    Returns (lag, idx, shape): np.bincount(idx, weights=x[lag], minlength=shape[0] * shape[1]).reshape(shape),
    with shape = (2N observers, n_src), gives for each ring observer (first N rows) and PD observer (last N rows)
    the sum of its lags of each source (K). minlength is needed: without it the reshape fails when the last
    (observer, source) pair has no lag."""
    obs = [int(o) for o in np.asarray(sysm.ring_obs)] + [int(o) for o in np.asarray(sysm.pd_obs).ravel()]
    ptr, src = np.asarray(sysm.obs_ptr), np.asarray(sysm.lag_src)
    lag, idx = [], []
    for i, o in enumerate(obs):
        mm = np.arange(ptr[o], ptr[o + 1])
        lag.append(mm)
        idx.append(i * sysm.n_src + src[mm])
    return np.concatenate(lag), np.concatenate(idx), (len(obs), int(sysm.n_src))


def group_matrix(src_names, N):
    """(N, S, G) 0/1 weights that put each source into XT_GROUPS for each observing ring."""
    import re
    S = len(src_names)
    W = np.zeros((N, S, len(XT_GROUPS)))
    for s, nm in enumerate(src_names):
        mt = re.match(r"^(heater|waveguide_absorption|junction|driver)(\d+)$", nm)
        for r in range(N):
            if not mt:                                   # not a per-ring source: an aggressor site
                W[r, s, _XT_AGGRESSOR] = 1
                continue
            own = int(mt.group(2)) == r
            kind = {"heater": 0, "waveguide_absorption": 1}.get(mt.group(1), 2)     # own_heater / own_light / own_jd
            W[r, s, kind + (0 if own else _XT_OTHER_OFFSET)] = 1
    return W


def xt_json(R, frames_s, checks):
    """Per frame: each ring's and PD's rise above the base split into XT_GROUPS (frame means of the lag states at
    every controller edge); whole-run and scored means of the full source split."""
    sd, plant = R["sd"], R["plant"]
    sysm = plant.sys
    N = plant.N
    S = len(frames_s)
    sums, cnt = plant.xt_sum, plant.xt_n
    if any(f not in sums for f in range(S)):
        raise RuntimeError("crosstalk tap does not cover every frame")
    mean = np.array([sums[f] / cnt[f] for f in range(S)])              # (S, 2N, n_src)
    W = group_matrix(list(sysm.src_names), N)
    ring = np.einsum("frs,rsg->frg", mean[:, :N], W)
    pd = np.einsum("frs,rsg->frg", mean[:, N:], W)
    sf = float(R["cr"].manifest_inputs.get("score_from_s") or 0.0)
    fs = sd["dec"] * R["dt"]
    t = np.asarray(frames_s, float)
    sel = (t - fs) >= sf - 1e-12
    tr = R["cr"].trace
    t_ring = np.asarray(tr[RecordSpec("t_ring_k", sd["dec"], "mean").key]["value"], float)[:, 0, :]
    t_base = np.asarray(tr[RecordSpec("t_base_k", sd["dec"], "mean").key]["value"], float)[:, 0, :]
    d = ring.sum(axis=2) + t_base - t_ring
    checks[sd["id"]].setdefault("crosstalk", {})["frame_split_sum_plus_base_minus_t_ring_k"] = {
        "max_abs": jnum(np.max(np.abs(d)), 3), "median_abs": jnum(np.median(np.abs(d)), 3),
        "note": "lag states at the controller edges (10 us) against the tick-mean ring temperature"}
    return {
        "groups": list(XT_GROUPS), "ring_k": js(ring, 6, 4), "pd_k": js(pd, 6, 4),
        "ring_by_source_mean_k": {"run": js(mean[:, :N].mean(axis=0), 4, 5),
                                  "scored": js(mean[sel, :N].mean(axis=0), 4, 5) if sel.any() else None},
        "pd_by_source_mean_k": {"run": js(mean[:, N:].mean(axis=0), 4, 5),
                                "scored": js(mean[sel, N:].mean(axis=0), 4, 5) if sel.any() else None},
        "src_names": list(sysm.src_names),
        "note": ("rise above the base temperature, K, split by heat source: the engine's thermal lag states of each "
                 "ring (and monitor PD) observer, read at every controller edge (10 us) and averaged over the frame; "
                 "groups: the ring's own heater, own absorbed light, own junction + driver, the other rings' heaters, "
                 "light and junctions + drivers, and the aggressor sites"),
    }


def _step_response_time(taus, res, frac, delay=0.0):
    """First time the step response sum r (1 - exp(-t / tau)) reaches frac of its final value (None if never)."""
    taus, res = np.asarray(taus, float), np.asarray(res, float)
    fin = float(res.sum())
    if not fin > 0:
        return None
    t = np.logspace(-9, 2, 4401)
    y = (res[None, :] * (1 - np.exp(-t[:, None] / taus[None, :]))).sum(axis=1)
    i = int(np.argmax(y >= frac * fin))
    if y[i] < frac * fin:
        return None
    if i == 0:
        return float(t[0]) + delay
    lt0, lt1 = math.log(t[i - 1]), math.log(t[i])
    a = (frac * fin - y[i - 1]) / (y[i] - y[i - 1])
    return float(math.exp(lt0 + a * (lt1 - lt0))) + delay


def coupling_json(sysm):
    """The model's thermal system as tables: DC gains of every observer from every source (K/mW), the lags of each
    (observer, source) pair (poles and residues), and the 10 / 63 / 90 % step-response times."""
    G = thermal.dc_gain(sysm)[0] * 1e-3
    lo, ls = np.asarray(sysm.lag_obs), np.asarray(sysm.lag_src)
    tau, res = np.asarray(sysm.taus_s, float)[0], np.asarray(sysm.res_k_per_w, float)[0] * 1e-3
    O, Sn = int(sysm.n_obs), int(sysm.n_src)
    lags = [[[] for _ in range(Sn)] for _ in range(O)]
    for mi in range(lo.size):
        lags[int(lo[mi])][int(ls[mi])].append([jnum(tau[mi], 5), jnum(res[mi], 5)])
    tq = {k: [[None] * Sn for _ in range(O)] for k in ("t10_s", "t63_s", "t90_s")}
    for o in range(O):
        for s in range(Sn):
            m = (lo == o) & (ls == s)
            if not m.any():
                continue
            for k, fr in (("t10_s", 0.1), ("t63_s", 1 - math.exp(-1)), ("t90_s", 0.9)):
                v = _step_response_time(tau[m], res[m], fr)
                tq[k][o][s] = jnum(v, 4)
    td = np.asarray(sysm.transport_delay_s if sysm.transport_delay_s is not None else [0.0], float).ravel()
    return {"obs_names": list(sysm.obs_names), "src_names": list(sysm.src_names),
            "g_k_per_mw": js(G, 4), "lags": lags, **tq,
            "transport_delay_s": jnum(td[0]),
            "note": ("every observer (rings, heater resistors, monitor PDs, die sensor) against every heat source: DC "
                     "gain Z(0) = sum of the residues (mrm.plant.thermal.dc_gain), K per mW into the source; lags "
                     "[tau_s, residue K/mW] (step response sum r (1 - exp(-t/tau))); t10/t63/t90: the first time the "
                     "step response reaches 10 / 63.2 / 90 % of its DC value (none when the DC gain is not positive). "
                     "The heat transport delay is not applied by the thermal kernels (0 here).")}


# ------------------------------------------------------------------------------------------------------ scenario inputs
LEVEL_NOTE = {"die": "one value for the die", "line": "one value per laser line", "ring": "one value per ring",
              "aggressor": "one value per aggressor site"}


def inputs_json(R, S_frames):
    """Every signal of the compiled scenario, evaluated every SAMPLE ticks and reduced over the frames (mean, min,
    max), or one constant value when it does not change in the run."""
    sd, plant, dt = R["sd"], R["plant"], R["dt"]
    arr = plant.scen_arrays
    dec = sd["dec"]
    per = dec // SAMPLE
    n_end = S_frames * dec
    ticks = np.arange(0, n_end + 1, SAMPLE, dtype=np.int64)
    fr = arr.render(ticks=ticks)
    active = set(arr.active_signals)
    out = []
    for spec in generators.SIGNALS:
        if spec.name not in arr.layout:
            continue
        v = np.asarray(fr.signals[spec.name], float)[:, 0, :]                 # (T+1, W)
        e = {"name": spec.name, "level": spec.level, "unit": spec.unit, "default": jnum(spec.default),
             "active": spec.name in active, "doc": spec.doc, "path": spec.path}
        if spec.name == "thermal.aggressor_w":
            e["columns"] = list(arr.aggressor_sites)
        if np.all(v == v[0:1]):
            e["constant"] = js(v[0], 6)
        else:
            w = v[1:1 + S_frames * per].reshape(S_frames, per, -1)
            e["mean"] = js(w.mean(axis=1), 6)
            e["min"] = js(w.min(axis=1), 6)
            e["max"] = js(w.max(axis=1), 6)
            e["range"] = [jnum(v.min()), jnum(v.max())]
        out.append(e)
    return {"signals": out, "components": list(arr.component_labels), "time_compression": jnum(arr.compression),
            "note": ("every input signal of the compiled scenario (mrm.scenarios.generators.SIGNALS) evaluated every "
                     "10 us with the scenario's own evaluator and reduced over each frame (mean, min, max); a signal "
                     "that does not change in the run is given once ('constant'). 'active': the scenario drives it "
                     "(it has components); otherwise it holds its default or the config's value")}


def controller_timing(R):
    """The tick timing of the controller steps, the ADC samples and the heater DAC (for the timing diagram)."""
    plant, cr = R["plant"], R["cr"]
    mh, hp = plant.monitor_hw, plant.heater_params
    return {"dt_s": R["dt"], "controller_period_ticks": int(plant.P.period), "controller_phase_ticks": int(plant.P.phase),
            "controller_jitter_ticks": int(plant.P.jitter),
            "controller_latency_ticks": cr.manifest_inputs.get("controller_latency_ticks"),
            "adc_period_ticks": int(mh.period_ticks), "adc_window_ticks": int(mh.window_ticks),
            "adc_latency_ticks": int(np.ravel(mh.latency_ticks)[0]), "adc_phase_ticks": np.asarray(mh.phase).tolist(),
            "dac_latency_ticks": int(np.ravel(hp.latency_ticks)[0]),
            "note": ("ticks of dt_s. Controller steps at phase + j x period; ADC samples of ring n end at "
                     "phase[n] + j x adc_period after a window of adc_window ticks and are released adc_latency ticks "
                     "later; a heater command reaches the DAC output dac_latency ticks after it is latched")}


# ------------------------------------------------------------------------------------------------------ hardware limits
def hardware_limits(R0, steady_junction=None):
    """The hardware limits the model assumes (g26 preset, realized plant), each with value, unit and provenance."""
    plant, cfg = R0["plant"], R0["exp"].cfg
    hp, mh, lhw, dph, pp = plant.heater_params, plant.monitor_hw, plant.laser_hw, plant.datapath_hw, plant.pp
    dt = R0["dt"]

    def cfg_value(key):
        """The config value of a key, or None when the preset has no such key."""
        try:
            return cfg.get(key)
        except Exception:                                    # noqa: BLE001
            return None

    def provenance_of(key):
        """The preset's provenance tag of a key, or None."""
        try:
            return cfg.provenance(key)
        except Exception:                                    # noqa: BLE001
            return None
    rows = []

    def row(group, label, value, unit, key=None, prov=None, note=""):
        """One table row; the provenance is the preset's tag of `key` unless `prov` is given."""
        rows.append({"group": group, "label": label, "value": value, "unit": unit, "key": key,
                     "provenance": prov if prov is not None else (provenance_of(key) if key else None), "note": note})
    N = plant.N
    p_full_25 = heater_full_power_w(plant, np.full((1, N), 298.15))[0, 0] * 1e3
    row("Heater", "DAC code range", [int(hp.code_lo), int(hp.code_hi)], "code", "ring.heater.code_max",
        note=f"{int(hp.bits)}-bit voltage DAC; code 0 = heater off.")
    row("Heater", "DAC resolution", int(hp.bits), "bit", "ring.heater.dac_bits")
    row("Heater", "Full-scale voltage", jnum(np.ravel(hp.v_full_scale_v)[0]), "V", "ring.heater.full_scale_v",
        note="Heater voltage at the highest code: V = code / code_max × V_FS.")
    row("Heater", "Maximum power per heater", jnum(p_full_25, 4), "mW", prov="derived",
        note="Driver law at full code with the heater at 25 °C (V_FS² / R). It falls as the heater warms (TCR).")
    row("Heater", "Heater resistance at 25 °C", jnum(np.ravel(hp.r_cold_ohm)[0]), "Ω", "ring.heater.r_cold_ohm")
    row("Heater", "Resistance temperature coefficient (TCR)", jnum(np.ravel(hp.tcr_per_k)[0]), "1/K",
        "ring.heater.tcr_per_k", note="R = R0 (1 + TCR × (T_heater − 25 °C)).")
    row("Heater", "Driver compliance", jnum(np.ravel(hp.compliance_v)[0]), "V", "ring.heater.driver_compliance_v")
    row("Heater", "Bank supply power limit", jnum(np.ravel(hp.supply_power_max_w)[0] * 1e3), "mW",
        "array.heater_total_power_max_w", note="Sum of the 8 heaters; above it the drive is scaled down.")
    row("Heater", "Bank supply current limit", jnum(np.ravel(hp.supply_current_max_a)[0] * 1e3), "mA",
        "array.heater_supply_current_max_a")
    row("Heater", "Heater power for one FSR", cfg_value("ring.heater.p_fsr_mw"), "mW", "ring.heater.p_fsr_mw",
        note="Tuning efficiency anchor: the power that moves the resonance by one free spectral range.")
    row("Heater", "DAC latency", jnum(float(np.ravel(hp.latency_ticks)[0]) * dt * 1e6), "µs", "ring.heater.latency_s")
    j = steady_junction or {}
    row("Junction and data", "Bias DAC range", [cfg_value("ring.bias_dac.v_min_v"), cfg_value("ring.bias_dac.v_max_v")],
        "V", "ring.bias_dac.v_min_v", note="Range of the junction bias set point.")
    row("Junction and data", "Bias DAC resolution", cfg_value("ring.bias_dac.bits"), "bit", "ring.bias_dac.bits")
    row("Junction and data", "Bias set point", cfg_value("scenario.bias_v"), "V", prov="preset setting",
        note="scenario.bias_v: the mean of the two data levels.")
    row("Junction and data", "Data swing", cfg_value("driver.swing_pp_v"), "Vpp", "driver.swing_pp_v",
        note=f"Peak to peak, {cfg_value('driver.level_convention') or 'centered'} on the bias.")
    if j:
        v0, v1, sh = (np.array(j[k], float) for k in ("v0_v", "v1_v", "shift_pm"))
        row("Junction and data", "Junction level 0 (V0)", jnum(np.median(v0), 3), "V", prov="derived",
            note="Model output (Truth, steady run, median of the rings): bias − swing / 2.")
        row("Junction and data", "Junction level 1 (V1)", jnum(np.median(v1), 3), "V", prov="derived",
            note="Model output (Truth, steady run, median of the rings): bias + swing / 2.")
        row("Junction and data", "Resonance shift V0 → V1", jnum(np.median(sh), 3), "pm", prov="derived",
            note="Model output (phase model, steady run): level-1 resonance minus level-0 resonance.")
    row("Junction and data", "Junction efficiency at 0 V", cfg_value("ring.junction.eff_at_0v_pm_per_v"), "pm/V",
        "ring.junction.eff_at_0v_pm_per_v")
    row("Monitor PD and ADC", "PD responsivity", cfg_value("ring.monitor.responsivity_a_per_w"), "A/W",
        "ring.monitor.responsivity_a_per_w")
    row("Monitor PD and ADC", "PD dark current at 25 °C", cfg_value("ring.monitor.dark_i0_na"), "nA",
        "ring.monitor.dark_i0_na",
        note=f"Grows with temperature, activation energy {cfg_value('ring.monitor.dark_activation_ev')} eV.")
    row("Monitor PD and ADC", "ADC resolution", int(mh.adc_bits), "bit", "ring.monitor.adc_bits")
    row("Monitor PD and ADC", "ADC full scale (PGA gain 1)", cfg_value("ring.monitor.adc_full_scale_ua"), "µA",
        "ring.monitor.adc_full_scale_ua", note=f"Codes 0 to {int(mh.code_max)}; full scale / gain at higher gains.")
    row("Monitor PD and ADC", "PGA gains", cfg_value("ring.monitor.pga_gains"), "×", "ring.monitor.pga_gains")
    row("Monitor PD and ADC", "Input noise", cfg_value("ring.monitor.input_noise_pa_per_rt_hz"), "pA/√Hz",
        "ring.monitor.input_noise_pa_per_rt_hz", note="Plus shot noise.")
    row("Timing", "ADC integration window", jnum(int(mh.window_ticks) * dt * 1e6), "µs",
        "ring.monitor.integration_window_s")
    row("Timing", "ADC sample period", jnum(int(mh.period_ticks) * dt * 1e6), "µs", "ring.monitor.adc_rate_hz",
        note=f"{cfg_value('ring.monitor.adc_rate_hz') / 1e3:g} kHz sample rate.")
    row("Timing", "ADC latency", jnum(float(np.ravel(mh.latency_ticks)[0]) * dt * 1e6), "µs",
        "ring.monitor.adc_latency_s")
    row("Timing", "Controller step period", jnum(int(plant.P.period) * dt * 1e6), "µs", prov="preset setting",
        note="controller_interface.update_rate_hz.")
    lat = R0["cr"].manifest_inputs.get("controller_latency_ticks")
    row("Timing", "Controller latency", jnum((lat or 0) * dt * 1e6), "µs", prov="model setting",
        note="From the latch of the inputs to the command (pi_lock_c).")
    row("Timing", "Heater DAC latency", jnum(float(np.ravel(hp.latency_ticks)[0]) * dt * 1e6), "µs",
        "ring.heater.latency_s")
    row("Laser and optics", "Laser power per line", cfg_value("laser.power_per_line_dbm"), "dBm",
        "laser.power_per_line_dbm", note=f"{float(np.asarray(lhw.p0_w, float)[0, 0]) * 1e3:g} mW at the laser module.")
    row("Laser and optics", "Chip input loss", cfg_value("optical_path.chip_input_loss_db"), "dB",
        "optical_path.chip_input_loss_db",
        note=f"Light into the chip: × {float(np.ravel(pp.laser.input_transmission)[0]):.4g} per line.")
    row("Laser and optics", "Bus waveguide loss", cfg_value("optical_path.bus_loss_db_per_mm"), "dB/mm",
        "optical_path.bus_loss_db_per_mm")
    row("Laser and optics", "Laser RIN", cfg_value("laser.rin_db_per_hz"), "dB/Hz", "laser.rin_db_per_hz")
    sens = np.asarray(dph.oma_sens_w, float).ravel() * 1e6
    row("Data path", "Bit rate", cfg_value("datapath.bitrate_gbps"), "Gb/s", "datapath.bitrate_gbps")
    s_dbm = 10 * math.log10(sens.max() * 1e-3)
    row("Data path", "Receiver sensitivity (OMA at BER 1e-9)",
        jnum(sens[0], 4) if np.all(sens == sens[0]) else [jnum(sens.min(), 4), jnum(sens.max(), 4)], "µW",
        prov="model default", note=f"{s_dbm:.1f} dBm OMA on every lane: the worst lane of the G26 range -14.0 to "
                                   "-12.8 dBm, read off a plot (mrm.plant.datapath).")
    lt = float(np.ravel(dph.link_t)[0])
    row("Data path", "Link transmission", jnum(lt, 4), "×", prov="printed",
        note=f"{-10 * math.log10(lt):.3g} dB chip to chip, bus output to the receiver (LM26-P; the "
             "mrm.plant.datapath default).")
    row("Data path", "BER limit", 1e-9, "", prov="model setting", note="metrics.ber_limit (the G26 criterion).")
    row("Data path", "BER counting window", cfg_value("datapath.error_count_period_s"), "s",
        "datapath.error_count_period_s")
    return {"rows": rows, "note": "the g26 preset's values as the realized plant uses them (nominal world); "
                                  "provenance: the preset's tag for that key, or 'derived' (model output), "
                                  "'model default' (a module default), 'preset setting' / 'model setting' (no tag)"}


# ---------------------------------------------------------------------------------------------------------- description
def _sig_str(v, d=3):
    """v as text with d significant digits (format g)."""
    return f"{float(v):.{d}g}"


def describe(R, sj, stats, devj, meta_geo, period_s=None):
    """One plain paragraph about the scenario; every number in it comes from the run."""
    sd = R["sd"]
    sm = sj["summary"]
    pr = sm["per_ring"]
    N = len(pr["max_abs_lock_error_pm"])
    ctrl = sm["controller"]
    dur = float(sm["duration_s"]) * 1e3
    base = float(np.mean(sj["drives"]["t_base_k"])) - 273.15
    pchip = np.array(devj["laser"]["p_chip_mw"], float)
    thr = sm["lock_threshold_pm"]
    worst = int(np.nanargmax(np.array(pr["max_abs_lock_error_pm"], float)))
    wmax = float(pr["max_abs_lock_error_pm"][worst])
    sf = float(sm["score_from_s"] or 0) * 1e3
    q = {d["key"]: d for d in stats["quantities"]}
    if sd["id"] == "steady":
        return (f"The controller {ctrl} holds all {N} rings on their laser lines for {dur:g} ms. The run starts from a "
                f"settled warm state at {base:.3g} °C. Nothing disturbs the bank: there is no aggressor, the laser "
                f"power is constant ({_sig_str(np.median(pchip))} mW per line into the chip) and every ring sends "
                f"data. This run shows the noise floor of the lock: after the start-up ({sf:g} ms, not scored) the largest "
                f"lock error is {_sig_str(wmax)} pm (ring {worst}), against a lock band of ±{thr:g} pm.")
    if sd["id"] == "a1_aggressor":
        a = np.array(sj["drives"]["aggressor_w"]["aggressor"], float)
        ip = int(np.argmax(a))
        d0 = meta_geo["ring_to_site_um"]["aggressor"]
        dmin = int(np.argmin(d0))                                  # the ring nearest the aggressor: it is quoted
        lo = next(x for x in stats["limits"] if x["key"] == "heater_lo")["scored"]
        rail_ms = float(lo["time_at_ms"][dmin])
        first = lo["first_ms"][dmin]
        err_pm = np.array(pr["max_abs_lock_error_pm"], float)
        return (f"An on-die heater (the A1 aggressor) sits {_sig_str(d0[dmin])} µm from ring {dmin} and "
                f"{_sig_str(max(d0))} µm from the farthest ring. Its power follows one "
                + (f"{period_s * 1e3:.4g} ms " if period_s else "") + f"period of a "
                f"rectified sine from its cool minimum and peaks at {_sig_str(a.max())} W (frame mean) at "
                f"{sj['t_s'][ip] * 1e3:.4g} ms. The controller {ctrl} must cool each ring by lowering its own heater "
                f"while the aggressor heats it. Ring {dmin} gets the most heat: its heater reaches code 0"
                + (f" at {first:.4g} ms and stays at that rail for {rail_ms:.3g} ms in total" if first is not None else "")
                + f", so the ring leaves its operating point by up to "
                  f"{_sig_str(err_pm[dmin])} pm. The other rings stay within "
                  f"{_sig_str(np.max(np.delete(err_pm, dmin)))} pm.")
    if sd["id"] == "laser_step":
        f_ = np.array(sj["drives"]["laser_power_factor"], float)[:, 0]
        return (f"The controller {ctrl} holds all {N} rings at {base:.3g} °C. At {sd['step_s'] * 1e3:g} ms every "
                f"laser line steps up by {100 * (f_.max() / f_[0] - 1):.3g} % in power "
                f"({10 * math.log10(f_.max() / f_[0]):.3g} dB), as a laser failover or drift would. More light heats "
                f"each ring more (self-heating) and changes what the monitor PD sees. The lock settles at a new point: "
                f"the lock error is measured about the point the lock held before the step, so the rings end outside "
                f"the ±{thr:g} pm band (largest error {_sig_str(wmax)} pm, ring {worst}).")
    if sd["id"] == "cold_start":
        ttl = np.array([np.nan if v is None else v for v in pr["time_to_lock_s"]], float) * 1e3
        decl = sj["rings"]["declared_line"][-1]
        p_end = pchip[-1].mean()
        k = int(np.argmax(pchip.mean(axis=1) >= 0.99 * p_end))
        return (f"The bank starts cold at {base:.3g} °C: every heater is at code 0 and the lasers turn on over the "
                f"first {sj['t_s'][k] * 1e3:.3g} ms (the light into the chip reaches 99 % of its final "
                f"{_sig_str(p_end)} mW per line). The supervisor {ctrl} sweeps each ring's heater to find the laser "
                f"lines, gives each ring a line (at the end rings 0 to {N - 1} sit on lines {', '.join(f'L{x}' for x in decl)}) "
                f"and locks the rings one by one in bus order. "
                f"The rings lock between {np.nanmin(ttl):.3g} and {np.nanmax(ttl):.3g} ms. The lock band is "
                f"±{thr:g} pm about the declared line.")
    return sj.get("description", "")
