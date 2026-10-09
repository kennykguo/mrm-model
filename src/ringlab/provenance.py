"""Conservative source provenance for code loaded in this Python process.

The package imports this module first. A later disk edit must not be labelled as
the source that produced a result from modules already loaded in memory.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def source_hashes():
    paths = sorted((PROJECT / "src").rglob("*.py")) + [PROJECT / "pyproject.toml", PROJECT / "uv.lock"]
    return {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


_LOADED_SOURCE_HASHES = source_hashes()


def loaded_source_hashes():
    return dict(_LOADED_SOURCE_HASHES)


def require_unchanged_sources(result_sources=None):
    if source_hashes() != _LOADED_SOURCE_HASHES:
        raise ValueError("project source changed after package import; start a fresh Python process and rerun before saving")
    if result_sources is not None and result_sources != _LOADED_SOURCE_HASHES:
        raise ValueError("result source snapshot differs from this loaded model; rerun before saving")

