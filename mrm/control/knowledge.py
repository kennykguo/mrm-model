"""What a controller may know (07 D13, P1-1): the knowledge vocabulary and the filtered view ``ControllerSpec.nominal``.

``controller_interface.knowledge`` lists tokens. Each token releases a set of config keys of the UNSAMPLED design
(:func:`token_keys`): the two vocabularies in use, the coarse one of :data:`mrm.control.calibration.KNOWLEDGE_TOKENS`
(heater_nominal, optics_nominal, dlambda_dt_nominal, monitor_nominal, line_grid, ring_to_line) and the fine one of
:data:`FINE_TOKENS` (the controllers-A helpers: heater_law, heater_r_nominal, heater_efficiency, dlambda_dt_nominal,
linewidth, thermal_dynamics, bias_efficiency; ``mrm.control.baselines.pi_lock.KNOWLEDGE_KEYS``). The token
``model_nominal`` (:data:`MODEL_NOMINAL`) releases the whole nominal model, which a model-based controller realizes
with uncertainty nominal and variation off (``mrm.control.models.linearize``): every key except the experiment
section and the scenario section, whose nominal operating point (``scenario.ambient_c``, ``scenario.bias_v``,
``scenario.bias_offset_v``) it keeps.

:class:`NominalView` is what ``ControllerSpec.nominal`` holds (``ControllerSpec`` wraps any nominal design it is
given): ``get`` / ``si`` of a released key delegate to the design; any other key raises :class:`KnowledgeError`, so a
controller cannot read a nominal number its knowledge does not grant, nor the experiment's seeds and world choices,
from which the sampled worlds (Truth) could be rebuilt. ``model_config()`` (only with ``model_nominal``) is the
design without the experiment section and without the scenario section beyond its operating point.
``fingerprint()`` covers the tokens and the released values only, so the spec is identical across seeds and
uncertainty worlds, also when those are config leaves (07 D13).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from mrm.control.calibration import KNOWLEDGE_TOKENS

__all__ = ["KnowledgeError", "MODEL_NOMINAL", "FINE_TOKENS", "MODEL_EXCLUDED_PREFIXES", "MODEL_SCENARIO_KEYS",
           "token_keys", "released_keys", "NominalView", "knowledge_view"]

#: the token that releases the whole nominal model (mrm.control.models.KNOWLEDGE_TOKEN)
MODEL_NOMINAL = "model_nominal"

#: the fine vocabulary of the controllers-A helpers (07 P1-1 names line_grid, ring_to_line, heater_r_nominal and
#: dlambda_dt_nominal; the others are proposed in docs/handoff/controllers_a.md). Re-exported as
#: ``mrm.control.baselines.pi_lock.KNOWLEDGE_KEYS``.
FINE_TOKENS: dict[str, tuple[str, ...]] = {
    "heater_law": ("ring.heater.driver_mode", "ring.heater.full_scale_v", "ring.heater.full_scale_a",
                   "ring.heater.full_scale_w", "ring.heater.r_cold_ohm", "ring.heater.dac_bits"),
    "heater_r_nominal": ("ring.heater.r_cold_ohm",),
    "heater_efficiency": ("ring.heater.p_fsr_mw", "ring.optics.fsr_thz"),
    "dlambda_dt_nominal": ("ring.thermal.dlambda_dt_pm_per_k",),
    "linewidth": ("ring.optics.q_loaded",),
    "thermal_dynamics": ("ring.thermal.tau_fast_s", "ring.thermal.tau_slow_s", "ring.thermal.share_fast_frac"),
    "bias_efficiency": ("ring.junction.eff_at_0v_pm_per_v", "ring.junction.v_bi_v",
                        "ring.junction.depletion_exponent", "ring.junction.fill_fraction"),
}

#: what model_nominal never releases: the experiment section (seeds, world choices: they rebuild the sampled worlds)
#: and the scenario section (the disturbances a run will apply), except its nominal operating point below
MODEL_EXCLUDED_PREFIXES = ("experiment.", "scenario.")
MODEL_SCENARIO_KEYS = ("scenario.ambient_c", "scenario.bias_v", "scenario.bias_offset_v")


class KnowledgeError(PermissionError):
    """The controller asked for nominal-model knowledge that controller_interface.knowledge does not grant."""


def token_keys() -> dict[str, tuple[str, ...]]:
    """token -> the config keys it releases (both vocabularies merged; model_nominal is not listed: it releases the
    whole design but the excluded sections)."""
    out: dict[str, tuple[str, ...]] = {}
    for t, (_, keys) in KNOWLEDGE_TOKENS.items():
        out[t] = tuple(keys)
    for t, keys in FINE_TOKENS.items():
        out[t] = tuple(dict.fromkeys(out.get(t, ()) + tuple(keys)))
    return out


def released_keys(knowledge: Iterable[str]) -> frozenset:
    """The config keys the tokens of ``knowledge`` release (without model_nominal's whole design)."""
    tk = token_keys()
    out: set[str] = set()
    for t in knowledge or ():
        out.update(tk.get(t, ()))
    return frozenset(out)


def _model_excluded(key: str) -> bool:
    return key.startswith(MODEL_EXCLUDED_PREFIXES) and key not in MODEL_SCENARIO_KEYS


def _who_releases(key: str) -> list[str]:
    return sorted(t for t, keys in token_keys().items() if key in keys)


def _plain(v):
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


class NominalView:
    """The knowledge-filtered view of the unsampled design (module docstring; 07 D13, P1-1).

    ``design`` is an ``mrm.params`` Config (``get``, ``si``, ``leaves``) or a plain mapping of dotted keys in config
    units; ``knowledge`` the granted tokens. Read with ``get(key, default)`` and ``si(key, default)`` (SI); a key
    that no granted token releases raises :class:`KnowledgeError`. ``leaves`` is the design's leaves of the released
    keys (their declared ranges). ``model_config()`` needs ``model_nominal``.
    """

    __slots__ = ("_design", "_knowledge", "_keys", "_model", "_model_cfg", "_fp")

    def __init__(self, design: Any, knowledge: Iterable[str] = ()):
        if isinstance(design, NominalView):
            design = design._design
        self._design = design
        self._knowledge = tuple(knowledge or ())
        self._model = MODEL_NOMINAL in self._knowledge
        self._keys = released_keys(self._knowledge)
        self._model_cfg = None
        self._fp = None

    # -- access
    @property
    def knowledge(self) -> tuple:
        return self._knowledge

    def released(self, key: str) -> bool:
        """True if a granted token releases ``key``."""
        if key in self._keys:
            return True
        return self._model and not _model_excluded(key)

    def _check(self, key: str) -> None:
        if not self.released(key):
            who = _who_releases(key)
            if _model_excluded(key):
                why = ("the experiment section (seeds, world choices) and the scenario are never part of the nominal "
                       "design a controller may read")
            elif who:
                why = f"grant one of {who} (or {MODEL_NOMINAL}) in controller_interface.knowledge"
            else:
                why = f"only {MODEL_NOMINAL} (the whole nominal model) releases it"
            raise KnowledgeError(f"spec.nominal: {key!r} is not released by controller_interface.knowledge "
                                 f"{list(self._knowledge)}: {why} (07 D13, P1-1)")

    def get(self, key: str, default: Any = None) -> Any:
        """The design's value of a released key (config units, as the design's own ``get``)."""
        self._check(key)
        get = getattr(self._design, "get", None)
        if not callable(get):
            return default
        return get(key, default)

    def si(self, key: str, default: Any = None):
        """The design's value of a released key in SI."""
        self._check(key)
        si = getattr(self._design, "si", None)
        if callable(si):
            return si(key, default)
        v = self.get(key, None)
        if isinstance(v, Mapping) and "value" in v:
            v = v["value"]
        if v is None:
            return default
        from mrm.params.units import to_si
        return to_si(key, v)

    @property
    def leaves(self) -> dict:
        """The design's leaves of the released keys (a Config's ``Leaf`` objects, with their ranges)."""
        lv = getattr(self._design, "leaves", None)
        if not isinstance(lv, Mapping):
            return {}
        return {k: v for k, v in lv.items() if self.released(k)}

    def model_config(self):
        """The nominal model for ``mrm.params.realize`` (``model_nominal`` only): the design without the experiment
        section and without the scenario section beyond its operating point (cached)."""
        if not self._model:
            raise KnowledgeError(f"spec.nominal.model_config() needs {MODEL_NOMINAL!r} in controller_interface."
                                 f"knowledge (it has {list(self._knowledge)}); add it with the override "
                                 f"controller_interface.knowledge=[{', '.join([*self._knowledge, MODEL_NOMINAL])}]")
        if self._model_cfg is None:
            d = self._design
            without = getattr(d, "without", None)
            if callable(without):
                self._model_cfg = without(MODEL_EXCLUDED_PREFIXES, keep=MODEL_SCENARIO_KEYS)
            elif isinstance(d, Mapping):
                self._model_cfg = {k: v for k, v in d.items() if not _model_excluded(str(k))}
            else:
                raise TypeError(f"spec.nominal: cannot build the nominal model from {type(d).__name__}")
        return self._model_cfg

    # -- identity
    def fingerprint(self) -> str:
        """sha256 (16 hex) of the tokens and of what they release (the model config's hash under model_nominal):
        independent of the experiment section, so of the seeds and world choices (07 D13)."""
        if self._fp is None:
            body: dict[str, Any] = {"knowledge": sorted(self._knowledge)}
            if self._model:
                m = self.model_config()
                h = getattr(m, "hash", None)
                body["model"] = h() if callable(h) else json.dumps(_plain(m), sort_keys=True, default=str)
            vals = {}
            lv = getattr(self._design, "leaves", None)
            for k in sorted(self._keys):
                try:
                    v = self._design.get(k, None)
                except Exception:      # noqa: BLE001 - a key this design does not carry
                    v = None
                rng = getattr(lv.get(k), "range", None) if isinstance(lv, Mapping) and k in lv else None
                vals[k] = [_plain(v), _plain(rng)]
            body["released"] = vals
            blob = json.dumps(body, sort_keys=True, default=str)
            self._fp = hashlib.sha256(blob.encode()).hexdigest()[:16]
        return self._fp

    def to_dict(self) -> dict:
        return {"knowledge": list(self._knowledge), "fingerprint": self.fingerprint()}

    def __eq__(self, other) -> bool:
        """Equal views: the same tokens over equal designs (a Config compares by identity)."""
        if not isinstance(other, NominalView):
            return NotImplemented
        return self._knowledge == other._knowledge and (self._design is other._design
                                                        or self._design == other._design)

    def __hash__(self) -> int:
        return hash((self._knowledge, self.fingerprint()))

    def __repr__(self) -> str:
        return f"NominalView(knowledge={list(self._knowledge)}, design={type(self._design).__name__})"


def knowledge_view(design: Any, knowledge: Iterable[str]) -> NominalView | None:
    """``design`` filtered by ``knowledge`` (None stays None; a view is re-filtered by the new tokens)."""
    if design is None:
        return None
    know = tuple(knowledge or ())
    if isinstance(design, NominalView) and design.knowledge == know:
        return design
    return NominalView(design, know)
