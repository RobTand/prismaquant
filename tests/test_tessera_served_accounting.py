"""CPU estimates use the existing structure-specific wire, not a new encoding."""
from fractions import Fraction
from types import SimpleNamespace

import pytest

from prismaquant import tessera_menu as menu
from prismaquant.lane_eligibility import ServingContext
from prismaquant.tessera_footprint import (
    tessera_exact_bits_for_shape, tessera_tensor_payload_breakdown,
    validate_tessera_tensor_payload_breakdown,
)
from prismaquant.tessera_formats import (
    TesseraFormatError, TesseraRouteRefused, get_tessera_family,
    tessera_served_wire_recipe, tessera_wire_recipe,
)
from prismaquant.tessera_legal_domain import byte_account

FAMILY = "TESSERA_E2M1_K2"
SHAPE = (64, 256)


def _payload(*, recipe=None, structure=None, rung=640):
    return tessera_tensor_payload_breakdown(
        SHAPE, family=FAMILY, body_rate_q256=rung,
        recipe=recipe, structure=structure,
    )


def _served(rung=640):
    return tessera_served_wire_recipe(FAMILY, rung, structure="routed_moe")


@pytest.mark.parametrize("rung", [640, 768, 896])
def test_routed_breakdown_prices_the_existing_served_recipe(rung):
    expected = _payload(recipe=_served(rung), rung=rung)
    actual = _payload(structure="routed_moe", rung=rung)
    assert actual == expected
    assert (actual["body_kind"], actual["trellis_span"]) == ("tcq", 2)
    assert validate_tessera_tensor_payload_breakdown(actual) == actual


def test_structure_free_and_explicit_recipe_prices_are_unchanged():
    research = tessera_wire_recipe(FAMILY, 640)
    default = _payload()
    assert default == _payload(recipe=research)
    # Explicit recipe describes existing bytes even in a serving context.
    assert _payload(recipe=research, structure="routed_moe") == default
    assert default["body_kind"] == "window"
    assert _payload(structure="routed_moe")["total_bytes"] != default["total_bytes"]


@pytest.mark.parametrize("shape,experts", [(SHAPE, 1), ((3, *SHAPE), 3)])
def test_exact_bits_prices_each_served_expert_unit(shape, experts):
    total_bytes = _payload(recipe=_served())["total_bytes"]
    assert isinstance(total_bytes, int)
    expected = Fraction(total_bytes * 8 * experts)
    assert tessera_exact_bits_for_shape(
        FAMILY, 640, shape, structure="routed_moe") == expected


def test_legal_domain_byte_account_uses_the_requested_structure():
    expected = _payload(recipe=_served())
    actual = byte_account(FAMILY, 640, SHAPE, structure="routed_moe")
    assert actual.total_bytes == expected["total_bytes"]
    assert actual.body_kind == "tcq"
    assert sum(actual.components().values()) == actual.total_bytes
    assert byte_account(FAMILY, 640, SHAPE).body_kind == "window"


def _allow_menu(monkeypatch):
    # Isolate byte/geometry selection from the independently tested admission gate.
    monkeypatch.setattr(menu, "route_admission", lambda *_a, **_kw: SimpleNamespace(
        admits=lambda _mode: True, act_bits=4))
    monkeypatch.setattr(menu, "tessera_tp_axis_legal", lambda *_a, **_kw: (True, ""))
    return ServingContext("cuda", "routed_moe", "resident",
                          "fixture@sha256:" + "a" * 64, "eager")


def test_context_scoped_menu_prices_the_served_wire(monkeypatch):
    context = _allow_menu(monkeypatch)
    rows = menu.expand_tessera_menu(
        SHAPE, mode=menu.MENU_RESEARCH, families=[get_tessera_family(FAMILY)],
        step_q256=128, serving_context=context,
    )
    row = next(row for row in rows if row.body_rate_q256 == 640)
    assert row.memory_bytes == _payload(recipe=_served())["total_bytes"]
    assert row.bits_per_param == Fraction(row.memory_bytes * 8, SHAPE[0] * SHAPE[1])


def test_routed_shape_and_tp_checks_require_whole_span_two_units(monkeypatch):
    _allow_menu(monkeypatch)
    assert menu.tessera_shape_legal(FAMILY, 640, (6, 256))[0]
    legal, reason = menu.tessera_shape_legal(
        FAMILY, 640, (6, 256), structure="routed_moe")
    assert not legal and "span" in reason
    legal, reason = menu.tessera_tp_legal(
        FAMILY, 640, (12, 256), tp_degree=2,
        parallel_kind=menu.PARALLEL_COLUMN, structure="routed_moe")
    assert not legal and "granularity" in reason


def test_menu_does_not_price_a_shape_that_cannot_hold_the_served_span(monkeypatch):
    context = _allow_menu(monkeypatch)
    assert menu.expand_tessera_menu(
        (6, 256), mode=menu.MENU_RESEARCH, families=[get_tessera_family(FAMILY)],
        step_q256=128, serving_context=context) == []


def test_explicit_dense_unattested_wire_is_not_silently_priced():
    with pytest.raises(TesseraRouteRefused, match="dense"):
        _payload(structure="dense")
    with pytest.raises(TesseraFormatError, match="structure"):
        _payload(structure="stacked")
