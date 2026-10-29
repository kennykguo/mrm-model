"""Delete numba disk-cache files so that the next run recompiles from the current sources.

Usage:
    .venv/bin/python tools/clear_numba_cache.py                 # the default set (KERNEL_MODULES below)
    .venv/bin/python tools/clear_numba_cache.py kernel optics    # only these modules (by file stem)
    .venv/bin/python tools/clear_numba_cache.py --all            # every .nbi/.nbc under mrm/ and tests/
    .venv/bin/python tools/clear_numba_cache.py --dry-run ...    # list only

Read this first. numba checks a cached function against its OWN source file only (numba/core/caching.py,
get_source_stamp: the modification time and size of the .py file that defines it). So an edit to an njit function
in a leaf module (mrm/plant/*.py, mrm/sim/metrics.py, mrm/scenarios/generators.py, mrm/core/*.py) does NOT
invalidate the cached kernel in mrm/sim/kernel.py. The kernel then keeps running the old compiled code (see
docs/ENGINE.md, "Clear the cache after editing compiled code"). Run this tool after every edit of njit code and
before any timing or "exact" comparison.

The default set is the kernel, every module whose njit functions the kernel inlines or calls, and
tools/perf_profile.py. The next process then recompiles the whole plant (5-8 min, about 2.3 GB peak RSS: run it
through tools/memguard.py --heavy).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: package directory -> module stems whose njit code the compiled kernel calls (directly or through another leaf).
#: The two tools are here because they compile their own copies of kernel components. perf_tick_prof compiles
#: without a disk cache, so its entry only guards against a future cache=True.
KERNEL_MODULES = {
    "mrm/sim": ["kernel", "metrics"],
    "mrm/plant": ["selfheat", "optics", "phase", "thermal", "electrical", "sensing", "datapath", "laser", "io"],
    "mrm/scenarios": ["generators"],
    "mrm/core": ["rng", "timing", "channels"],
    "tools": ["perf_profile", "perf_tick_prof"],
}


def _cache_files(pkg_dir: Path, stem: str):
    """The numba cache files (index .nbi and data .nbc) of module ``stem`` in ``pkg_dir/__pycache__``."""
    cache_dir = pkg_dir / "__pycache__"
    if not cache_dir.is_dir():
        return []
    return sorted(p for p in cache_dir.iterdir() if p.name.startswith(stem + ".") and p.suffix in (".nbi", ".nbc"))


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]
    files: list[Path] = []
    if "--all" in argv:
        for base in ("mrm", "tests"):
            files += sorted(p for p in (ROOT / base).rglob("*") if p.suffix in (".nbi", ".nbc"))
    else:
        stems = set(argv)
        for pkg, mods in KERNEL_MODULES.items():
            for m in mods:
                if not stems or m in stems:
                    files += _cache_files(ROOT / pkg, m)
        if stems:
            # a named module outside the default set: search every package of mrm/ for it
            known = {m for mods in KERNEL_MODULES.values() for m in mods}
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
