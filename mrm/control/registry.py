"""Controller registry: name -> class, or a lazy "module:Class" string (07 P1-17).

``controller: {name, params}`` in an experiment file resolves through :func:`create`. A name is either registered
here (by :func:`register`, usually as a decorator in the controller's module) or a ``"package.module:Class"`` path,
so a student's controller outside the package works without registration.

Built-in baselines register themselves when ``mrm.control.baselines`` is imported; :func:`resolve` imports that
package once on the first miss. The oracle (``mrm.validation.oracle``) is Truth-fed and must never be registered
(07 P1-1): any class with a true ``truth_fed`` attribute is refused.
"""
from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["RegistryError", "register", "unregister", "resolve", "create", "available"]

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_REGISTRY: dict[str, Any] = {
    "composite": "mrm.control.api:CompositeController",
}
_RESOLVED: dict[str, Any] = {}
_BUILTINS_LOADED = False
_BASELINES_PACKAGE = "mrm.control.baselines"     # imported once, on the first registry miss


class RegistryError(KeyError):
    """Unknown, duplicate or forbidden registry entry."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def _check_not_truth_fed(name: str, target: Any) -> None:
    if getattr(target, "truth_fed", False):
        raise RegistryError(f"{name!r} is Truth-fed (truth_fed = True); the oracle is not a registrable "
                            f"controller (07 P1-1)")


def register(name: str, target: Any = None, *, overwrite: bool = False) -> Callable | Any:
    """Register ``target`` (a class, a factory or a "module:Class" string) under ``name``.

    Use as ``register("pi", PIController)`` or as a decorator ``@register("pi")``. Raises RegistryError for a bad
    name, a different target already registered under the name (unless ``overwrite``), or a Truth-fed target.
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise RegistryError(f"controller name {name!r} must be a lower-case identifier")

    def _do(t):
        if not isinstance(t, str):
            _check_not_truth_fed(name, t)
        prev = _REGISTRY.get(name)
        if prev is not None and prev is not t and prev != t and not overwrite:
            raise RegistryError(f"controller {name!r} is already registered as {prev!r}")
        _REGISTRY[name] = t
        _RESOLVED.pop(name, None)
        return t

    if target is None:
        return _do
    return _do(target)


def unregister(name: str) -> None:
    """Remove ``name`` (no error if absent)."""
    _REGISTRY.pop(name, None)
    _RESOLVED.pop(name, None)


def _import_path(path: str) -> Any:
    mod_name, _, attr = path.partition(":")
    if not mod_name or not attr:
        raise RegistryError(f"{path!r}: expected 'package.module:Class'")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise RegistryError(f"{path!r}: cannot import {mod_name!r} ({e})") from None
    obj = mod
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise RegistryError(f"{path!r}: {mod_name!r} has no attribute {attr!r}") from None
    return obj


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    parts = _BASELINES_PACKAGE.split(".")
    own = {".".join(parts[:i]) for i in range(1, len(parts) + 1)}
    try:
        importlib.import_module(_BASELINES_PACKAGE)
    except ModuleNotFoundError as e:   # only a missing package is silent; a broken baseline module must surface
        if e.name not in own:
            _BUILTINS_LOADED = False
            raise


def resolve(name_or_path: str) -> Any:
    """The class (or factory) for a registered name or a "module:Class" path."""
    if not isinstance(name_or_path, str) or not name_or_path:
        raise RegistryError(f"controller name must be a non-empty string, got {name_or_path!r}")
    if name_or_path in _RESOLVED:
        return _RESOLVED[name_or_path]
    target = _REGISTRY.get(name_or_path)
    if target is None and ":" not in name_or_path:
        _load_builtins()
        target = _REGISTRY.get(name_or_path)
    if target is None:
        if ":" in name_or_path:
            target = name_or_path
        else:
            raise RegistryError(f"unknown controller {name_or_path!r}; registered: {available()}; or give "
                                f"'package.module:Class'")
    obj = _import_path(target) if isinstance(target, str) else target
    _check_not_truth_fed(name_or_path, obj)
    _RESOLVED[name_or_path] = obj
    return obj


def create(name_or_path: str, params: Mapping[str, Any] | None = None) -> Any:
    """Instantiate a controller from ``controller: {name, params}``.

    Order: ``cls.from_params(params)`` if defined; a :class:`mrm.control.api.BaseController` subclass gets
    ``cls(params)`` (its Params dataclass validates the keys); anything else gets ``cls(**params)``.
    """
    from mrm.control.api import BaseController
    obj = resolve(name_or_path)
    params = dict(params or {})
    if hasattr(obj, "from_params") and callable(obj.from_params):
        ctrl = obj.from_params(params)
    elif isinstance(obj, type) and issubclass(obj, BaseController):
        ctrl = obj(params)
    else:
        ctrl = obj(**params)
    _check_not_truth_fed(name_or_path, ctrl)      # a factory may return a Truth-fed instance
    return ctrl


def available() -> list[str]:
    """Registered names, built-ins included, sorted."""
    _load_builtins()
    return sorted(_REGISTRY)
