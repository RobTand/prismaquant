"""Scoped rendering and its price resolve the routed wire together (#1504)."""
from types import SimpleNamespace

import pytest
import torch

from prismaquant import tessera_render as render
from prismaquant.lane_eligibility import ServingContext
from prismaquant.tessera_footprint import tessera_tensor_payload_breakdown
from prismaquant.tessera_formats import (
    TesseraRouteRefused, parse_tessera_format_name,
    tessera_served_wire_recipe, tessera_wire_recipe,
)

NAME = "TESSERA_E2M1_K2_R640"
FAMILY, RUNG = parse_tessera_format_name(NAME)
STRUCTURE = "routed_moe"
SHAPE = (4, 64)


def _weight():
    return torch.randn(SHAPE, generator=torch.Generator().manual_seed(1504)).bfloat16()


def _served():
    return tessera_served_wire_recipe(FAMILY, RUNG, structure=STRUCTURE)


def test_scoped_planned_facts_describe_the_served_body():
    planned = render.planned_wire_facts(FAMILY, RUNG, structure=STRUCTURE)
    assert (planned["body"], planned["window_bits"]) == ("TCQ", 0)
    assert planned["structure"] == STRUCTURE


def test_scoped_encode_and_render_match_the_served_wire_and_byte_price():
    from tessera.container import parse
    from tessera.fused import pack_fused
    from tessera.serving.scheme import wire_facts_of_parsed

    weight = _weight()
    unit, _ = render._encode_planned_unit(weight, NAME, structure=STRUCTURE)
    on_bytes = wire_facts_of_parsed(SimpleNamespace(unit=unit, grid=render._grid_for(FAMILY)))
    planned = render.planned_wire_facts(FAMILY, RUNG, structure=STRUCTURE)
    assert set(planned["rates"]) == set(on_bytes["rates"])
    for key in set(on_bytes) - {"rates"}:
        assert planned[key] == on_bytes[key], key

    expected, blob = render.encode_tessera_unit(
        weight, NAME, hessian_required=False, recipe=_served())
    actual = render.render_tessera_weight(weight, NAME, structure=STRUCTURE)
    assert torch.equal(actual, expected)
    price = tessera_tensor_payload_breakdown(
        SHAPE, family=FAMILY, body_rate_q256=RUNG, structure=STRUCTURE,
        member_name="down_proj")
    # The plane extent is exact; the existing whole-container estimate retains
    # its separately documented manifest-ratio slack (#1609).
    assert price["payload_bytes"] == len(parse(blob).plane_region)
    fused = pack_fused([("down_proj", SHAPE[0], blob)])
    assert 0 <= price["total_bytes"] - len(fused) <= 64


def test_synthesis_prices_and_renders_the_contexts_served_recipe(monkeypatch):
    monkeypatch.setattr(render, "_producer_eligible", lambda *_a, **_kw: True)
    context = ServingContext("sm_121", STRUCTURE, "resident",
                             "fixture@sha256:" + "a" * 64, "eager")
    spec = render.synthesize_tessera_spec(NAME, serving_context=context)
    price = tessera_tensor_payload_breakdown(
        SHAPE, family=FAMILY, body_rate_q256=RUNG, structure=STRUCTURE)
    assert spec.memory_bytes_for_shape(SHAPE) == price["total_bytes"]
    expected = render.synthesize_tessera_spec(NAME, recipe=_served())
    assert torch.equal(spec.quantize_dequantize(_weight()),
                       expected.quantize_dequantize(_weight()))


def test_production_cache_render_uses_the_units_declared_structure():
    weight = _weight()
    actual = render.render_tessera_production(
        weight, NAME, qname="expert.down", activations=None,
        levers={"tessera_weights_only": True,
                "tessera_structure_by_unit": {"expert.down": STRUCTURE}})
    expected, _ = render.encode_tessera_unit(
        weight, NAME, hessian_required=False, recipe=_served())
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("mapping", [{}, {"other": STRUCTURE}, {"expert.down": None}, []])
def test_declared_production_structure_map_cannot_silently_fall_back(mapping):
    with pytest.raises(ValueError, match="structure"):
        render.render_tessera_production(
            _weight(), NAME, qname="expert.down", activations=None,
            levers={"tessera_weights_only": True, "tessera_structure_by_unit": mapping})


@pytest.mark.parametrize("operation", ["plan", "encode", "render", "production"])
def test_dense_unattested_plan_and_render_refuse_before_encoding(operation, monkeypatch):
    def unexpected_plan(*_args, **_kwargs):
        pytest.fail("an unattested dense wire reached the encoder plan")

    monkeypatch.setattr(render, "_plan", unexpected_plan)
    with pytest.raises(TesseraRouteRefused, match="dense"):
        if operation == "plan":
            render.planned_wire_facts(FAMILY, RUNG, structure="dense")
        elif operation == "encode":
            render._encode_planned_unit(_weight(), NAME, structure="dense")
        elif operation == "render":
            render.render_tessera_weight(_weight(), NAME, structure="dense")
        else:
            render.render_tessera_production(
                _weight(), NAME, qname="dense.down", activations=None,
                levers={"tessera_weights_only": True,
                        "tessera_structure_by_unit": {"dense.down": "dense"}})


def test_unattested_scoped_spec_reports_ineligibility_and_preserves_wire_price(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "attested")
    context = ServingContext("sm_121", "dense", "resident",
                             "fixture@sha256:" + "a" * 64, "eager")
    name = "TESSERA_E2M1_K2_R512"
    family, rung = parse_tessera_format_name(name)
    spec = render.synthesize_tessera_spec(name, serving_context=context)
    assert spec.producer_eligible is False
    wire = tessera_served_wire_recipe(
        family, rung, structure="dense", refuse_unattested=False)
    price = tessera_tensor_payload_breakdown(
        SHAPE, family=family, body_rate_q256=rung, recipe=wire)
    assert spec.memory_bytes_for_shape(SHAPE) == price["total_bytes"]


@pytest.mark.parametrize("structure", ["routed-moe", [], 0])
@pytest.mark.parametrize("explicit_recipe", [False, True])
def test_spec_query_does_not_mask_invalid_structure(structure, explicit_recipe):
    context = SimpleNamespace(structure=structure)
    with pytest.raises(ValueError, match="structure"):
        render.synthesize_tessera_spec(
            NAME, serving_context=context,
            recipe=_served() if explicit_recipe else None)


def test_structure_free_and_explicit_research_recipe_are_preserved():
    research = tessera_wire_recipe(FAMILY, RUNG)
    bare = render.planned_wire_facts(FAMILY, RUNG)
    assert bare == render.planned_wire_facts(FAMILY, RUNG, recipe=research)
    assert bare["body"] == "WINDOW"
    explicit = render.planned_wire_facts(FAMILY, RUNG, recipe=research, structure=STRUCTURE)
    assert explicit == {**bare, "structure": STRUCTURE}
