# load documented design parameters and realize SI arrays for the plant.
# a preset states independent numbers with units in key suffixes and provenance describing their sources.
# load returns Config in those config units.
# realize draws uncertain worlds and device variation, converts to SI, solves derived relations, and returns PlantParams arrays such as (B, N), (B, 1), or (B, L).
# derive gives a nominal single-world view with diagnostics.
# see docs/PARAMETERS_FORMAT.md.
# units.py parses suffix units.
# schema.py defines permitted keys, bounds, levels, and fidelity rules.
# config.py loads YAML inheritance, alternatives, and overrides, then checks schema/provenance.
# relations.py defines thermo-optic tuning, heater efficiency, geometry/FSR/order, loss/Q, and tick timing.
# realize.py samples and assembles plant parameters.
# derive.py adds nominal diagnostics.
# symbols.py maps quantities to LaTeX.
# table.py renders parameter views and serves params CLI commands.
# example —
# from mrm import params
# cfg = params.load("g26")
# pp = params.realize(cfg, batch=64, seeds=0)
# nominal = params.derive(cfg)
# view helpers below import their renderers on demand.
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


# the views below live in mrm.params.table.
# each one is imported when it is called, so importing mrm.params stays light.

# return the provenance chain for a key, derived value, or diagnostic.
# see params.table.explain.
def explain(cfg, path, **kw):
    from mrm.params.table import explain as explain_path
    return explain_path(cfg, path, **kw)


# render the parameter table as Markdown.
# see params.table.table.
def table(cfg, **kw):
    from mrm.params.table import table as render_table
    return render_table(cfg, **kw)


# return the resolved parameter tree with provenance.
# see params.table.show.
def show(cfg):
    from mrm.params.table import show as show_tree
    return show_tree(cfg)


# compare config leaves.
# see params.table.diff.
def diff(a, b):
    from mrm.params.table import diff as diff_configs
    return diff_configs(a, b)


# derive and realize a small batch, then return warnings.
# see params.table.validate.
def validate(cfg, **kw):
    from mrm.params.table import validate as validate_config
    return validate_config(cfg, **kw)
