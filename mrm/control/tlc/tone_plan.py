"""Dither tone plans: the source's collision rule and a search for N-ring plans (spec Phase 3).

The source's B3 lock-in integrates over 4 ms, so frequencies live on a 250 Hz bin grid. The rule, from
``thermal_control/params.py`` (ControllerParams.tone_freqs): each ring demodulates its tone f_i and its 2nd
harmonic 2 f_i, and both must be at least ``min_sep`` bins from

- every other ring's tone and 2nd harmonic,
- every 2nd-order intermodulation product f_j +/- f_k (j != k),
- the control-update ripple (multiples of 10 kHz),

and every tone must be at least ``min_dc`` bins above DC (the Hann window's leakage of slow drifts). The source's
4-ring plan (2.0, 3.0, 3.5, 4.5 kHz) passes, and :func:`search` finds exactly it for 4 rings; the withdrawn rev 1
grid did not pass (docs/tlc/REV1_REVIEW.md).
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
