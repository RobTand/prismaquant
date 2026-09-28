"""The Tessera lane's plugin: the hooks core reaches this lane through.

``lane_specs/tessera.json`` names this module as the lane's ``plugin`` and
declares the ``tessera`` format family (``TESSERA_`` names) as data. Core
modules -- the format registry, the render path, the cache-miss fallbacks,
the serving profiles -- never import a ``tessera_*`` module; they look a hook
up here through ``lane_spec`` (decoupling step 6, PQ #1550).

Two rules keep the seam honest:

* **The import is free.** This module imports no ``tessera_*`` module and not
  the ``tessera`` package at module scope, so resolving a hook costs a stock
  run nothing. Every hook imports what it needs when it is called.
* **Hooks bind at call time.** Each hook reads the lane function through its
  module when it runs, exactly as the in-function imports it replaces did, so
  a test that substitutes ``tessera_menu.route_admission`` still reaches every
  caller.
"""
from __future__ import annotations

from typing import Any

#: The family this lane declares in ``lane_specs/tessera.json``.
FAMILY_ID = "tessera"


def is_tessera_format_name(name: object) -> bool:
    """True for a name the Tessera family claims, without importing Tessera.

    The family's name grammar is data in the lane spec; this reads it there.
    For lane modules that need the question by name. Core asks
    ``format_registry.format_family_of`` instead.
    """
    from .lane_spec import format_family_for_name

    family = format_family_for_name(name)
    return family is not None and family.id == FAMILY_ID


# -- format family hooks (format_registry) ---------------------------------

def synthesize_format(canonical: str):
    """A ``FormatSpec`` for one Tessera rung, or ``None`` for a non-rung."""
    from .tessera_render import synthesize_tessera_spec

    return synthesize_tessera_spec(canonical)


def format_admitted_in_contexts(canonical: str, contexts) -> bool:
    """Does at least one of these serving contexts admit ``canonical``?

    Shared-menu intake only: each candidate still passes its own unit's scope.
    """
    from .tessera_menu import menu_mode, route_admission

    return any(route_admission(canonical, serving_context=context).admits(menu_mode())
               for context in contexts)


def render_production(weight, fmt: str, *, qname, activations, levers):
    """The production render: the H-aware encode that ships, decoded."""
    from .tessera_render import render_tessera_production

    return render_tessera_production(
        weight,
        fmt,
        qname=qname,
        activations=activations,
        levers=levers,
    )


# -- serving profile hooks (serving_profiles) --------------------------------

def resolved_serving_lane(fmt: str, *, runtime_version: str,
                          serving_context: Any = None):
    """The route the Tessera admission seam resolves for one rung."""
    from .tessera_menu import tessera_resolved_serving_lane

    return tessera_resolved_serving_lane(
        fmt, runtime_version=runtime_version,
        **({"serving_context": serving_context}
           if serving_context is not None else {}))


def require_canonical_subfamily(value: object, *, owner: str) -> str:
    """Refuse a value that is not a canonical Tessera family name."""
    from .tessera_formats import get_tessera_family, TesseraFormatError

    try:
        family = get_tessera_family(value)
    except TesseraFormatError as exc:
        raise ValueError(f"{owner}: invalid Tessera family {value!r}") from exc
    if not isinstance(value, str) or family.name != value:
        raise ValueError(f"{owner}: expected a canonical Tessera family, got {value!r}")
    return value


def format_subfamily(canonical: str) -> str | None:
    """The Tessera family a rung name belongs to, or ``None`` if it parses to none."""
    from .tessera_formats import parse_tessera_format_name, TesseraFormatError

    try:
        parsed = parse_tessera_format_name(canonical)
    except TesseraFormatError:
        return None
    return None if parsed is None else parsed[0].name


# -- the pinned serving runtime (serving_profiles) ---------------------------

class ServingRuntimePinError(ValueError):
    """The lane's serving-runtime pin exists and cannot be read."""


def serving_runtime_pin_path():
    from .tessera_serving_runtime_pin import tessera_serving_runtime_pin_path

    return tessera_serving_runtime_pin_path()


def load_serving_runtime_pin():
    """The pin, or :class:`ServingRuntimePinError` when it is malformed."""
    from .tessera_serving_runtime_pin import (
        TesseraServingRuntimePinError, load_tessera_serving_runtime_pin,
    )

    try:
        return load_tessera_serving_runtime_pin()
    except TesseraServingRuntimePinError as exc:
        raise ServingRuntimePinError(str(exc)) from exc


def serving_runtime_contract_path():
    """The ``runtime_contract.json`` the importable serving runtime packages."""
    from .tessera_runtime_contract import contract_path

    return contract_path()
