"""Parameters: suffix grammar, strict schema, loader with provenance, relations, realize and derive (07 P0-3,
P0-4, P1-14, P1-15, P1-16, P2-12). The format is documented in docs/PARAMETERS_FORMAT.md.

Typical use::

    from mrm import params
    cfg = params.load("g26", overrides=["ring.optics.q_loaded=4000"],
                      alts={"uncertainty.thermo_optic": "H86"})
    pp = params.realize(cfg, batch=64, seeds=0)        # PlantParams: SI arrays (B, N) / (B, 1) / (B, L)
    nominal = params.derive(cfg)                        # B = 1 view + diagnostics
    print(params.explain(cfg, "ring.heater.efficiency_nm_per_mw"))
"""
from mrm.params.config import (ArtefactError, Config, ConfigError, Leaf, UncertaintyEntry, VariationEntry, load,
                               provenance_rank, weakest)
from mrm.params.derive import Derived, derive, effective_provenance
from mrm.params.realize import PlantParams, RealizeError, realize
from mrm.params.relations import OverdeterminedError, UnderdeterminedError, residuals
from mrm.params.units import from_si, parse_key, to_si

__all__ = ["load", "Config", "Leaf", "UncertaintyEntry", "VariationEntry", "ConfigError", "ArtefactError", "realize",
           "PlantParams", "RealizeError", "derive", "Derived", "effective_provenance", "OverdeterminedError",
           "UnderdeterminedError", "residuals", "to_si", "from_si", "parse_key", "provenance_rank", "weakest",
           "explain", "table", "show", "diff", "validate"]


def explain(cfg, path, **kw):
    """Provenance chain of a key, derived value or diagnostic (see :func:`mrm.params.table.explain`)."""
    from mrm.params.table import explain as _e
    return _e(cfg, path, **kw)


def table(cfg, **kw):
    """PARAMETERS.md Markdown (see :func:`mrm.params.table.table`)."""
    from mrm.params.table import table as _t
    return _t(cfg, **kw)


def show(cfg):
    """Resolved tree with provenance (see :func:`mrm.params.table.show`)."""
    from mrm.params.table import show as _s
    return _s(cfg)


def diff(a, b):
    """Leaf-by-leaf differences (see :func:`mrm.params.table.diff`)."""
    from mrm.params.table import diff as _d
    return _d(a, b)


def validate(cfg, **kw):
    """Derive and realize a small batch; return the warnings (see :func:`mrm.params.table.validate`)."""
    from mrm.params.table import validate as _v
    return _v(cfg, **kw)
