"""Dither tone plans: the source's collision rule and a search for N-ring plans (spec Phase 3).

The source's B3 lock-in integrates over 4 ms, so frequencies live on a 250 Hz bin grid. The rule, from
``thermal_control/params.py`` (ControllerParams.tone_freqs): each ring demodulates its tone f_i and its 2nd
harmonic 2 f_i, and both must be at least ``min_sep`` bins from

- every other ring's tone and 2nd harmonic,
- every 2nd-order intermodulation product f_j +/- f_k (j != k),
- the control-update ripple (multiples of 10 kHz),

and every tone must be at least ``min_dc`` bins above DC (the Hann window's leakage of slow drifts). The source's
4-ring plan (2.0, 3.0, 3.5, 4.5 kHz) passes, and :func:`search` finds exactly it for 4 rings; the withdrawn rev 1
grid did not pass.
"""
from __future__ import annotations

from itertools import combinations

BIN_HZ = 250.0              # 1 / lockin_window (4 ms)
RIPPLE_HZ = 10e3            # control update rate (100 us)

#: search(8): the 8-ring plan used on G26 (26 s to find, so it is stored). Its top tone (13.5 kHz) is 3x the
#: source's 4.5 kHz, so the same dither amplitude needs 3x the heater slew (R19 limit; Phase 3 measures it)
EIGHT_RING_BINS = (16, 24, 28, 34, 36, 42, 46, 54)


def _lines(bins: tuple[int, ...]) -> list[tuple[int, str]]:
    """Every spectral line the tones put on a monitor: (bin, label)."""
    out = [(b, f"f{i}") for i, b in enumerate(bins)] + [(2 * b, f"2f{i}") for i, b in enumerate(bins)]
    for (i, a), (j, b) in combinations(enumerate(bins), 2):
        out += [(a + b, f"f{i}+f{j}"), (abs(a - b), f"|f{i}-f{j}|")]
    return out


def collisions(bins: tuple[int, ...], *, min_sep: int = 2, min_dc: int = 8, max_bin: int = 400) -> list[str]:
    """Rule violations of a plan given in 250 Hz bins; [] means collision-free."""
    bad = [f"f{i} = {b} bins is closer than {min_dc} bins to DC" for i, b in enumerate(bins) if b < min_dc]
    ripple = [k * int(RIPPLE_HZ / BIN_HZ) for k in range(1, max_bin // int(RIPPLE_HZ / BIN_HZ) + 2)]
    lines = _lines(bins)
    for i, b in enumerate(bins):
        own = {f"f{i}", f"2f{i}"}
        for name, x in ((f"f{i}", b), (f"2f{i}", 2 * b)):
            for y, label in lines:
                if label not in own and abs(x - y) < min_sep:
                    bad.append(f"{name} = {x} bins is within {min_sep} bins of {label} = {y}")
            for r in ripple:
                if abs(x - r) < min_sep:
                    bad.append(f"{name} = {x} bins is within {min_sep} bins of the {r * BIN_HZ / 1e3:g} kHz ripple")
    return bad


def search(n: int, *, lo: int = 8, hi: int = 120, min_sep: int = 2, min_dc: int = 8) -> tuple[int, ...]:
    """The collision-free n-tone plan (in bins) with the lowest highest tone, then lexicographically smallest.

    Lower tones are better: the source's R19 slew limit caps dither amplitude x frequency. Depth-first search
    over increasing bins with pruning; raises ValueError if no plan fits below ``hi``.
    """
    for top in range(lo + n - 1, hi + 1):
        found = _extend((), n, lo, top, min_sep, min_dc)
        if found is not None:
            return found
    raise ValueError(f"no collision-free {n}-tone plan with tones in bins {lo}..{hi}")


def _extend(plan, n, lo, top, min_sep, min_dc):
    if len(plan) == n:
        return plan if plan[-1] == top else None
    start = plan[-1] + 1 if plan else lo
    for b in range(start, top + 1):
        cand = plan + (b,)
        if not collisions(cand, min_sep=min_sep, min_dc=min_dc, max_bin=4 * top):
            found = _extend(cand, n, lo, top, min_sep, min_dc)
            if found is not None:
                return found
    return None


def to_hz(bins: tuple[int, ...]) -> tuple[float, ...]:
    return tuple(b * BIN_HZ for b in bins)


# --- sampled dither shapes (square, three-level, triangle, qsine; source ae9556b) ------------------------------
#: the source's tones for sampled shapes (params.SQUARE_TONES): out_rate / N with N even, so every half period is a
#: whole number of 10 us output samples. :func:`sampled_collisions` with every ring in reach finds exactly these.
SAMPLED_N = (26, 22, 16, 14)
OUT_RATE_HZ = 100e3
SAMPLED_HARMONICS = (1, 3, 5, 7, 9)     # the harmonics B1/B4 model for a sampled shape
SAMPLED_SEP_HZ = 500.0                  # source rule: >= 500 Hz between every demodulated line and every other line


def sampled_collisions(periods: tuple[int, ...], *, reach: int, sep_hz: float = SAMPLED_SEP_HZ,
                       harmonics: tuple[int, ...] = SAMPLED_HARMONICS) -> list[str]:
    """Rule violations of a per-ring plan of sampled tones (``periods``: output samples per tone period).

    The source's rule (params.SQUARE_TONES), applied per monitor: ring i demodulates f_i and 2 f_i, and both must be
    ``sep_hz`` from every line that reaches ring i's detuning: harmonics 1..9 and 2nd harmonics of the tones of the
    rings within ``reach`` (thermal crosstalk), their sums and differences, the own tone's higher harmonics and the
    10 kHz ripple. With reach >= n - 1 this is the source's global rule. Rings further apart than ``reach`` may
    share a tone: their crosstalk is outside the core's model (coupling has ``reach`` entries).
    """
    f = [OUT_RATE_HZ / p for p in periods]
    bad = []
    for i in range(len(f)):
        near = [j for j in range(len(f)) if abs(i - j) <= reach]
        lines = [(k * f[i], f"{k}f{i}") for k in harmonics if k > 2]
        for j in (j for j in near if j != i):
            lines += [(k * f[j], f"{k}f{j}") for k in harmonics] + [(2 * f[j], f"2f{j}")]
        for a, b in combinations(near, 2):
            lines += [(f[a] + f[b], f"f{a}+f{b}"), (abs(f[a] - f[b]), f"|f{a}-f{b}|")]
        lines += [(m * RIPPLE_HZ, f"{m * 10} kHz ripple") for m in range(1, 13)]
        for name, x in ((f"f{i}", f[i]), (f"2f{i}", 2 * f[i])):
            bad += [f"ring {i}: {name} = {x:.0f} Hz is within {sep_hz:g} Hz of {lab} = {y:.0f} Hz"
                    for y, lab in lines if abs(x - y) < sep_hz]
    return bad


def sampled_plan(n: int) -> tuple[float, ...]:
    """Tones (Hz) for a sampled dither on n rings: the source's four tones, repeated every four rings.

    Every even-N tone set for 8 distinct tones within reach 3 collides (no plan exists), but four tones tiled
    passes with reach 3, the length of the G26 coupling prior: rings i and i + 4 share a tone. On G26 the
    crosstalk four rings away is 1.5 % at DC and its ms-scale pole leaves about 1e-4 of the own response at these
    tones. The plan also tops out at 7.1 kHz against 13.5 kHz for the sine plan (EIGHT_RING_BINS), which halves
    the slew a given dither amplitude needs (R19).
    """
    periods = tuple(SAMPLED_N[i % len(SAMPLED_N)] for i in range(n))
    reach = min(n - 1, len(SAMPLED_N) - 1)
    bad = sampled_collisions(periods, reach=reach)
    if bad:
        raise ValueError(f"sampled tone plan for {n} rings collides: {bad[:3]}")
    return tuple(OUT_RATE_HZ / p for p in periods)
