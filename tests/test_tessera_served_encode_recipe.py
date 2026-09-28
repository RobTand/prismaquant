"""The recipe a campaign encodes is the served wire the pinned contract attests (#1502).

``tessera.export.wire_recipe`` is the research default: below the E2M1x2 cap it
is the WINDOW body, span 1, ``window_bits`` 12. The pinned contract's
``formats[].attested_wire`` stamps the served routed wire, TCQ span 2, at every
E2M1_K2 rung from 128 to 896, and the routed cells attest those rungs. A
routed unit priced on the research wire is priced on bytes the routed decoder
cannot read; G2 spent w02r row-0045 on exactly that.
"""
import pytest

pytest.importorskip("tessera")

from prismaquant.joint_catalog_extension import added_format_recipe
from prismaquant.tessera_formats import (
    TesseraRouteRefused, parse_tessera_format_name, tessera_served_wire_recipe,
    tessera_wire_recipe,
)


def _served(fmt, structure):
    family, rung = parse_tessera_format_name(fmt)
    return tessera_served_wire_recipe(family, rung, structure=structure)


def _attested(fmt):
    recipe = dict(added_format_recipe(fmt))
    recipe.pop("grid")
    recipe.pop("q256")
    return recipe


@pytest.mark.parametrize("fmt", ["TESSERA_E2M1_K2_R640", "TESSERA_E2M1_K2_R768"])
def test_a_routed_subcap_e2m1_unit_encodes_the_attested_wire(fmt):
    assert _served(fmt, "routed_moe").to_config() == _attested(fmt)


@pytest.mark.parametrize("fmt", [
    "TESSERA_E2M1_K2_R896",
    *(f"TESSERA_E4M3_K1_R{q}" for q in (768, 896, 1152, 1408, 1536, 2048)),
    *(f"TESSERA_BF16_K1_R{q}" for q in (960, 1088, 1152, 1408, 1536, 2048)),
])
@pytest.mark.parametrize("structure", ["dense", "routed_moe"])
def test_where_the_research_wire_is_attested_the_encode_is_unchanged(fmt, structure):
    family, rung = parse_tessera_format_name(fmt)
    assert _served(fmt, structure) == tessera_wire_recipe(family, rung)
    assert _served(fmt, structure).to_config() == _attested(fmt)


@pytest.mark.parametrize("fmt", ["TESSERA_E2M1_K2_R640", "TESSERA_E2M1_K2_R768"])
def test_a_dense_subcap_e2m1_unit_is_refused_as_a_route_fact(fmt):
    # The attested wire differs from the research wire at this rung, and no
    # dense cell attests the rung (the E2M1 dense cells publish [896]): the
    # dense route has no attested reading of either spelling.
    with pytest.raises(TesseraRouteRefused, match="dense"):
        _served(fmt, "dense")


# ---------------------------------------------------------------------------
# GREEN: the resolved wire is the encoded wire, and the campaign plans it
# ---------------------------------------------------------------------------

def _weight():
    import torch
    return torch.randn(64, 64, generator=torch.Generator().manual_seed(1502)).bfloat16()


def test_the_planning_reader_returns_the_refusal_it_would_raise():
    from prismaquant.tessera_formats import tessera_served_route_refusal
    family, rung = parse_tessera_format_name("TESSERA_E2M1_K2_R640")
    assert tessera_served_route_refusal(family, rung, structure=None) is None
    assert tessera_served_route_refusal(family, rung, structure="routed_moe") is None
    reason = tessera_served_route_refusal(family, rung, structure="dense")
    assert "dense" in reason and "stays in the menu" in reason
    # Not refused where the reader range ends: the export route gate owns it.
    k1, k1_rung = parse_tessera_format_name("TESSERA_E4M3_K1_R1024")
    assert tessera_served_route_refusal(k1, k1_rung, structure="dense") is None
    with pytest.raises(Exception, match="structure 'stacked'"):
        tessera_served_route_refusal(family, rung, structure="stacked")


def test_a_receipt_reader_resolves_the_dense_wire_without_the_refusal():
    family, rung = parse_tessera_format_name("TESSERA_E2M1_K2_R640")
    assert tessera_served_wire_recipe(
        family, rung, structure="dense", refuse_unattested=False) == tessera_wire_recipe(family, rung)


# BF16_K1 is the family whose recipe sets a source spread (``window_sigma``).
@pytest.mark.parametrize("fmt", ["TESSERA_E2M1_K2_R640", "TESSERA_E2M1_K2_R896",
                                 "TESSERA_E4M3_K1_R1024", "TESSERA_BF16_K1_R1088"])
def test_forwarding_the_research_recipe_writes_the_bytes_the_default_wrote(fmt):
    from prismaquant import tessera_render
    family, rung = parse_tessera_format_name(fmt)
    weight = _weight()
    _render, blob = tessera_render.encode_tessera_unit(weight, fmt, hessian_required=False)
    default = tessera_render._tessera_export.encode_linear(
        weight, grid=family.payload_grid(), q256=int(rung), name=fmt, verify=False).blob
    assert blob == default


@pytest.mark.parametrize("fmt", ["TESSERA_E2M1_K2_R640", "TESSERA_E2M1_K2_R768"])
def test_a_routed_subcap_encode_writes_the_served_wire(fmt):
    from tessera.container import parse
    from prismaquant import tessera_render
    served = _served(fmt, "routed_moe")
    _render, blob = tessera_render.encode_tessera_unit(
        _weight(), fmt, hessian_required=False, recipe=served)
    manifest = parse(blob).manifest
    assert (manifest.body, manifest.span, manifest.scale_plane.kind, manifest.window_bits) == (
        served.body, served.span, served.scale_plane, served.window_bits)
    batch = tessera_render.encode_tessera_units(
        [_weight(), _weight()], fmt, hessian_required=False, recipe=served)
    assert [blob_ for _r, blob_ in batch] == [blob, blob]


def test_batches_never_join_units_served_on_different_structures():
    import torch
    from prismaquant import tessera_campaign as campaign
    weights = {name: torch.empty((64, 64)) for name in ("dense", "routed", "routed2")}
    pending = [(name, "TESSERA_E2M1_K2", 896) for name in weights]
    structures = {"dense": "dense", "routed": "routed_moe", "routed2": "routed_moe"}
    batches = campaign._anchor_batches(pending, weights=weights, batch_size=4,
                                       structures=structures)
    assert sorted(map(sorted, batches)) == sorted([
        [("dense", "TESSERA_E2M1_K2", 896)],
        [("routed", "TESSERA_E2M1_K2", 896), ("routed2", "TESSERA_E2M1_K2", 896)]])
    # Without declared structures the historical key is unchanged: one batch.
    assert len(campaign._anchor_batches(pending, weights=weights, batch_size=4)) == 1


def test_the_campaign_plans_route_refusals_as_contract_facts():
    from prismaquant import tessera_campaign as campaign
    rungs = {640, 768, 896}
    plan = campaign._served_route_refusals(
        "TESSERA_E2M1_K2", rungs, ["up", "gate"],
        encode_structure={"up": "dense", "gate": "dense"}, projected_units={})
    assert sorted(plan) == ["640", "768"]
    assert plan["640"]["members"] == ["gate", "up"] and "dense" in plan["640"]["reason"]
    # A projected routed unit is served on TCQ: nothing refused.
    assert campaign._served_route_refusals(
        "TESSERA_E2M1_K2", rungs, ["e0"], encode_structure={"e0": "routed_moe"},
        projected_units={"e0": {}}) == {}
    # A routed unit with no projection cannot be adopted on the routed wire.
    unprojected = campaign._served_route_refusals(
        "TESSERA_E2M1_K2", rungs, ["e0"], encode_structure={"e0": "routed_moe"},
        projected_units={})
    assert sorted(unprojected) == ["640", "768"]
    assert "no producer projection" in unprojected["640"]["reason"]
    # No declared structure and K1 families: never refused.
    assert campaign._served_route_refusals(
        "TESSERA_E2M1_K2", rungs, ["x"], encode_structure=None, projected_units={}) == {}
    assert campaign._served_route_refusals(
        "TESSERA_E4M3_K1", {1024, 1408}, ["x"], encode_structure={"x": "dense"},
        projected_units={}) == {}


def test_a_routed_checkpoint_identity_stamps_the_served_wire(monkeypatch):
    from types import SimpleNamespace
    from tessera import cached_unit
    from prismaquant import tessera_campaign as campaign
    fmt = "TESSERA_E2M1_K2_R640"
    family, rung = parse_tessera_format_name(fmt)
    weight = _weight()
    anchor = SimpleNamespace(qname="u", format_name=fmt, family=family.name,
                             body_rate_q256=rung, hessian_applied=False,
                             input_global_scale=None)
    kwargs = dict(weights={"u": weight}, menus={"u": [SimpleNamespace(format_name=fmt)]},
                  calibration_source=None, static_scales={})
    # The A-side gate is not under test here.
    monkeypatch.setattr(campaign, "_require_resumable_anchor", lambda *_a, **_k: None)
    research = campaign._checkpoint_anchor_identity(anchor, **kwargs)
    assert research == cached_unit.encoding_input_identity(
        weight, "u", family.payload_grid(), int(rung))
    assert campaign._checkpoint_anchor_identity(anchor, structure="dense", **kwargs) == research
    tensor = "model.layers.3.mlp.experts.0.gate_proj.weight"
    projection = {"tensor": tensor, "source_tensor": tensor, "source_layout": "unpacked_per_expert",
                  "expert": 0, "source_slice": {"expert": 0, "selector": "whole",
                                                "transpose": False},
                  "projection": "gate_proj", "group": "w13", "rows": 64, "cols": 64}
    routed = campaign._checkpoint_anchor_identity(
        anchor, projected_units={"u": projection}, structure="routed_moe", **kwargs)
    assert routed["recipe"] == {"grid": family.payload_grid().name, "q256": int(rung),
                                **_served(fmt, "routed_moe").to_config()}
    # The predicate the overlay catalog applies to every cell
    # (tools/build_t4_overlay_catalog.py: "recipe is not the pinned
    # contract's"): a fresh routed receipt now passes it; the research one
    # the campaign stamped before #1502 does not.
    assert routed["recipe"] == dict(added_format_recipe(fmt))
    assert research["recipe"] != dict(added_format_recipe(fmt))


def test_a_resumed_row_stamped_on_the_research_wire_is_diverted_not_adopted():
    from types import SimpleNamespace
    from prismaquant import tessera_campaign as campaign
    fmt = "TESSERA_E2M1_K2_R640"
    family, rung = parse_tessera_format_name(fmt)
    anchor = SimpleNamespace(format_name=fmt)

    def record(recipe):
        return {"identity": {"recipe": {"grid": "E2M1x2", "q256": int(rung),
                                        **recipe.to_config()}}}

    research = record(tessera_wire_recipe(family, rung))
    served = record(_served(fmt, "routed_moe"))
    reason = campaign._stale_served_wire(anchor, research, structure="routed_moe")
    assert reason is not None and "#1502" in reason
    assert campaign._stale_served_wire(anchor, served, structure="routed_moe") is None
    # No declared structure, or a dense unit on its own wire: adopted as before.
    assert campaign._stale_served_wire(anchor, research, structure=None) is None
    assert campaign._stale_served_wire(anchor, research, structure="dense") is None
    # An unreadable receipt is left to the full identity check, which refuses it.
    assert campaign._stale_served_wire(anchor, {}, structure="routed_moe") is None
