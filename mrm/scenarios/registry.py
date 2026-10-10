"""Scenario registry: name -> builder, ``module:callable`` resolution, and the config entry point (07 P1-17).

``create(name, **params)`` builds a Scenario from the library (or any registered or importable builder);
``from_config(cfg)`` reads the ``scenario`` section of a config (``name | stack``, ``params``, ``time_compression``,
``ambient_c``, ``bias_v``, ``bias_offset_v``, ``aggressor``, ``target``) plus ``datapath.transition_density``.

Config forms
------------
    scenario: {name: a1, params: {period_s: 0.1307, referencing: temperature}}
    scenario: {stack: [a1, laser_drift], params: {a1: {...}, laser_drift: {mode: random_walk}}}
    scenario: {aggressor: {site: agg0}, target: {units: resonance, trajectory: a2, reading: one_sided_sawtooth}}
    scenario: {name: soak, time_compression: 100}
"""
from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from mrm.scenarios import generators as g
from mrm.scenarios import library as lib

__all__ = ["register", "resolve", "create", "available", "describe", "from_config", "unflatten", "ALIASES"]

_REGISTRY: dict[str, Callable[..., g.Scenario]] = {}
#: Short names: the textbook's scenario ids (Ch17 groups A and B) and common abbreviations.
ALIASES: dict[str, str] = {}


def register(name: str, *, aliases: tuple[str, ...] = (), overwrite: bool = False):
    """Decorator: register a builder ``fn(**params) -> Scenario`` under ``name`` (and ``aliases``)."""
    def deco(fn):
        for n in (name,) + tuple(aliases):
            if not overwrite and (n in _REGISTRY or n in ALIASES) and _REGISTRY.get(n, None) is not fn:
                raise ValueError(f"scenario name {n!r} is already registered")
        _REGISTRY[name] = fn
        for a in aliases:
            ALIASES[a] = name
        return fn
    return deco


for _name, _fn, _al in (
        ("static", lib.static, ("none",)),
        ("a1_slide10_aggressor", lib.a1_slide10_aggressor, ("a1",)),
        ("a2_808_aggressor", lib.a2_808_aggressor, ("a2",)),
        # STRESS scenarios (monitor dark-current runaway), not a realistic layout: A1 / A2 at the layout site
        # aggressor_pd_adjacent (g26: 5 um from ring 0's monitor PD)
        ("a1_slide10_aggressor_pd_adjacent", lib.a1_slide10_aggressor_pd_adjacent, ("a1_pd_adjacent",)),
        ("a2_808_aggressor_pd_adjacent", lib.a2_808_aggressor_pd_adjacent, ("a2_pd_adjacent",)),
        ("laser_sweep_emulation", lib.laser_sweep_emulation, ("a3",)),
        ("soak", lib.soak, ("a4",)),
        ("connector_remate", lib.connector_remate, ("a6",)),
        ("reloma_plane", lib.reloma_plane, ("a7",)),
        ("squelch", lib.squelch, ("b1",)),
        ("laser_failover", lib.laser_failover, ("b2",)),
        ("xpu_power_step", lib.xpu_power_step, ("b3",)),
        ("cold_start", lib.cold_start, ("b5",)),
        ("warm_start", lib.warm_start, ()),
        ("laser_drift", lib.laser_drift, ()),
        ("fibre_loss_step", lib.fibre_loss_step, ()),
        ("traffic_start_stop", lib.traffic_start_stop, ("traffic",)),
        ("idle", lib.idle, ()),
        ("training_pattern", lib.training_pattern, ("training",)),
        ("ones_density_steps", lib.ones_density_steps, ("sun_ones_density",)),
        ("bias_ripple", lib.bias_ripple, ()),
        ("bias_step", lib.bias_step, ()),
        ("line_enable", lib.line_enable, ()),
        ("aggressor_direct", lib.aggressor_direct, ())):
    register(_name, aliases=_al)(_fn)


def resolve(name: str) -> Callable[..., g.Scenario]:
    """Builder for a registered name, an alias, or ``module:callable``."""
    if not isinstance(name, str) or not name:
        raise g.ScenarioError(f"scenario name must be a non-empty string, got {name!r}")
    key = name.strip()
    if key in _REGISTRY:
        return _REGISTRY[key]
    if key.lower() in ALIASES:
        return _REGISTRY[ALIASES[key.lower()]]
    if ":" in key:
        mod, attr = key.split(":", 1)
        try:
            fn = getattr(importlib.import_module(mod), attr)
        except (ImportError, AttributeError) as e:
            raise g.ScenarioError(f"cannot import scenario builder {key!r}: {e}") from None
        if not callable(fn):
            raise g.ScenarioError(f"{key!r} is not callable")
        return fn
    import difflib
    near = difflib.get_close_matches(key, available() + sorted(ALIASES), n=3)
    raise g.ScenarioError(f"unknown scenario {name!r}" + (f"; did you mean {near}?" if near else
                          f"; known: {available()}"))


def available() -> list[str]:
    """Registered scenario names (without aliases)."""
    return sorted(_REGISTRY)


def describe(name: str) -> str:
    """The builder's docstring and signature (provenance of its defaults)."""
    fn = resolve(name)
    return f"{getattr(fn, '__name__', name)}{inspect.signature(fn)}\n\n{inspect.getdoc(fn) or ''}"


def create(name: str, **params) -> g.Scenario:
    """Build scenario ``name`` with keyword ``params`` (a ScenarioError names unknown parameters)."""
    fn = resolve(name)
    try:
        sig = inspect.signature(fn)
        sig.bind(**params)
    except TypeError as e:
        raise g.ScenarioError(f"scenario {name!r}: {e}; signature {name}{inspect.signature(fn)}") from None
    except ValueError:
        pass
    sc = fn(**params)
    if not isinstance(sc, g.Scenario):
        raise g.ScenarioError(f"builder {name!r} returned {type(sc).__name__}, not a Scenario")
    return sc


def unflatten(d: Mapping[str, Any] | None) -> dict:
    """{'a1.period_s': 0.1} -> {'a1': {'period_s': 0.1}} (Config stores open sub-trees flattened)."""
    out: dict = {}
    for k, v in (d or {}).items():
        parts = str(k).split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
            if not isinstance(cur, dict):
                raise g.ScenarioError(f"conflicting scenario parameter {k!r}")
        if isinstance(v, Mapping):
            cur.setdefault(parts[-1], {}).update(unflatten(v))
        else:
            cur[parts[-1]] = v
    return out


def _getter(cfg) -> Callable[[str, Any], Any]:
    if hasattr(cfg, "get") and hasattr(cfg, "leaves"):          # mrm.params.config.Config
        return lambda path, default=None: cfg.get(path, default)
    if isinstance(cfg, Mapping):
        def get(path, default=None):
            if path in cfg:
                v = cfg[path]
            else:
                v = cfg
                for p in path.split("."):
                    if isinstance(v, Mapping) and p in v:
                        v = v[p]
                    else:
                        return default
            if isinstance(v, Mapping) and "value" in v and "provenance" in v:      # a provenance leaf
                return v["value"]
            return v
        return get
    raise TypeError("cfg must be a mrm.params Config or a mapping")


def from_config(cfg) -> g.Scenario:
    """The scenario of a config (07 P1-17). Reads ``scenario.name`` or ``scenario.stack``, ``scenario.params``
    (per name for a stack), ``scenario.aggressor`` / ``scenario.target`` (07 P1-4), ``scenario.time_compression``
    (07 P0-10), ``scenario.ambient_c`` (base temperature default; it replaces a scenario's own default, such as A1's
    25 degC or warm_start's ambient, and has no effect where a component drives the base absolutely, e.g. soak or
    cold_start), ``scenario.bias_v`` (absolute bias default when
    the scenario owns tx.bias), ``scenario.bias_offset_v`` (constant offset) and ``datapath.transition_density``
    (default transition density). With nothing set it returns the static scenario. The 07 P0-10 compression rule
    is checked here (CompressionError) as well as at compile."""
    get = _getter(cfg)
    name = get("scenario.name")
    stack_names = get("scenario.stack")
    params = unflatten(get("scenario.params") or {})
    if name and stack_names:
        raise g.ScenarioError("give scenario.name or scenario.stack, not both")
    names = list(stack_names or ([name] if name else []))
    parts: list[g.Scenario] = []
    for nm in names:
        if stack_names:
            p = params.get(nm, {})
        else:
            canonical = ALIASES.get(str(nm).lower(), nm)
            sub = params.get(nm, params.get(canonical))
            p = sub if isinstance(sub, dict) and len(params) == 1 else params
        if not isinstance(p, dict):
            raise g.ScenarioError(f"scenario.params for {nm!r} must be a mapping")
        parts.append(create(nm, **p))
    if stack_names:
        extra = set(params) - set(stack_names)
        if extra:
            raise g.ScenarioError(f"scenario.params keys {sorted(extra)} are not in scenario.stack {stack_names}")
    agg = unflatten(get("scenario.aggressor") or {})
    tgt = unflatten(get("scenario.target") or {})
    if agg or tgt:
        parts.append(lib.aggressor_from_config(agg, tgt))
    sc = g.stack(parts, name="+".join(names) or None) if parts else lib.static()
    defaults = dict(sc.defaults)
    from mrm.params.units import to_si                # the single config-units -> SI conversion (07 P1-15)
    for key, sig in (("scenario.ambient_c", "thermal.t_base_k"), ("scenario.bias_v", "tx.bias_v"),
                     ("scenario.bias_offset_v", "tx.bias_offset_v"),
                     ("datapath.transition_density", "data.transition_density")):
        v = get(key)
        if v is not None:
            defaults[sig] = to_si(key, v)
    tc = get("scenario.time_compression")
    if tc is not None and float(tc) != 1.0:
        if sc.time_compression not in (1.0, float(tc)):
            raise g.CompressionError("scenario.time_compression disagrees with the scenario's own compression")
        sc = sc.with_compression(float(tc))
    sc = sc.replace(defaults=defaults)
    g.check_time_compression(sc)          # 07 P0-10 load-time rule: fail when the config is read, not only at compile
    return sc
