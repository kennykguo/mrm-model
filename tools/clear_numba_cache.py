"""Delete numba disk-cache files so that the next run recompiles from the current sources.

numba keys a cached function on its OWN source file only (numba/core/caching.py, get_source_stamp: mtime and size
of the defining .py file). An edit to an njit function in a leaf module (mrm/plant/*.py, mrm/sim/metrics.py,
mrm/scenarios/generators.py, mrm/core/*.py) therefore does NOT invalidate the cached kernel in mrm/sim/kernel.py,
which keeps running the old compiled code (docs/ENGINE.md section 1). Run this after every edit of njit code and
before any timing or "exact" comparison.

Usage:
    .venv/bin/python tools/clear_numba_cache.py                 # the kernel + every mrm leaf module (default)
    .venv/bin/python tools/clear_numba_cache.py kernel optics    # only these modules (by file stem)
    .venv/bin/python tools/clear_numba_cache.py --all            # every .nbi/.nbc under mrm/ and tests/
    .venv/bin/python tools/clear_numba_cache.py --dry-run ...    # list only

The default set is the kernel and every module whose njit functions the kernel inlines or calls, so the next
process recompiles the whole plant (5-8 min, ~2.3 GB peak RSS: run it through tools/memguard.py --heavy).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# every module the compiled kernel calls into (directly or through another leaf)
DEFAULT = {
    "mrm/sim": ["kernel", "metrics"],
    "mrm/plant": ["selfheat", "optics", "phase", "thermal", "electrical", "sensing", "datapath", "laser", "io"],
    "mrm/scenarios": ["generators"],
    "mrm/core": ["rng", "timing", "channels"],
    "tools": ["perf_profile", "perf_tick_prof"],
}


def _caches(pkg_dir: Path, stem: str):
    d = pkg_dir / "__pycache__"
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.name.startswith(stem + ".") and p.suffix in (".nbi", ".nbc"))


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]
    files: list[Path] = []
    if "--all" in argv:
        for base in ("mrm", "tests"):
            files += sorted(p for p in (ROOT / base).rglob("*") if p.suffix in (".nbi", ".nbc"))
    else:
        stems = set(argv)
        for pkg, mods in DEFAULT.items():
            for m in mods:
                if not stems or m in stems:
                    files += _caches(ROOT / pkg, m)
        if stems:
            known = {m for mods in DEFAULT.values() for m in mods}
            for s in stems - known:
                for p in (ROOT / "mrm").rglob(f"__pycache__/{s}.*"):
                    if p.suffix in (".nbi", ".nbc"):
                        files.append(p)
    n = 0
    size = 0
    for p in files:
        size += p.stat().st_size
        n += 1
        if not dry:
            p.unlink()
    print(f"[clear_numba_cache] {'would delete' if dry else 'deleted'} {n} files ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
