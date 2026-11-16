"""The v68 served E2M1x2 wire is WINDOW L14 on dense and routed units."""
from prismaquant import tessera_cell_measurement as cell_measurement
import pytest

pytest.importorskip("tessera")

from prismaquant.joint_catalog_extension import added_format_recipe
from prismaquant.tessera_formats import (
    parse_tessera_format_name, tessera_served_wire_recipe,
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


@pytest.mark.parametrize("structure", ["dense", "routed_moe"])
def test_e2m1_r896_encodes_the_v68_attested_window_wire(structure):
    recipe = _served("TESSERA_E2M1_K2_R896", structure)
    assert recipe.to_config() == _attested("TESSERA_E2M1_K2_R896")
    assert recipe.span == 1 and recipe.window_bits == 14


@pytest.mark.parametrize("fmt", [
    *(f"TESSERA_E4M3_K1_R{q}" for q in (768, 896, 1152, 1408, 1536, 2048)),
    *(f"TESSERA_BF16_K1_R{q}" for q in (960, 1088, 1152, 1408, 1536, 2048)),
])
@pytest.mark.parametrize("structure", ["dense", "routed_moe"])
def test_where_the_research_wire_is_attested_the_encode_is_unchanged(fmt, structure):
    family, rung = parse_tessera_format_name(fmt)
    assert _served(fmt, structure) == tessera_wire_recipe(family, rung)
    assert _served(fmt, structure).to_config() == _attested(fmt)


@pytest.mark.parametrize("rung", [640, 768])
def test_subcap_wire_planning_grants_no_dense_route(rung):
    from prismaquant.tessera_formats import tessera_served_route_refusal
    from prismaquant import tessera_runtime_contract as contract
    assert tessera_served_route_refusal("TESSERA_E2M1_K2", rung, structure="dense") is None
    table = contract.installed_runtime_snapshot().eligibility
    assert not any(c.family == "TESSERA_E2M1_K2" and c.covers_rate(rung) for c in table.cells)


def test_served_wire_mismatch_refuses_with_the_planning_reason(monkeypatch):
    from dataclasses import replace
    from prismaquant import tessera_formats as formats

    recipe = _served("TESSERA_E2M1_K2_R896", "routed_moe")
    mismatch = replace(recipe, window_bits=recipe.window_bits + 1)
    monkeypatch.setattr(formats, "_served_recipe_for", lambda *_args: mismatch)
    reason = formats.tessera_served_route_refusal(
        "TESSERA_E2M1_K2", 896, structure="routed_moe")
    assert reason is not None and "is not the wire the pinned contract attests" in reason
    with pytest.raises(formats.TesseraRouteRefused) as refused:
        _served("TESSERA_E2M1_K2_R896", "routed_moe")
    assert str(refused.value) == reason


def _weight():
    import torch
    return torch.randn(64, 64, generator=torch.Generator().manual_seed(1502)).bfloat16()


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


@pytest.mark.parametrize("structure", ["dense", "routed_moe"])
def test_e2m1_r896_writes_the_served_wire(structure):
    fmt = "TESSERA_E2M1_K2_R896"
    from tessera.container import parse
    from prismaquant import tessera_render
    served = _served(fmt, structure)
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
    batches = cell_measurement.compatible_batches(pending, weights=weights, batch_size=4,
                                       structures=structures)
    assert sorted(map(sorted, batches)) == sorted([
        [("dense", "TESSERA_E2M1_K2", 896)],
        [("routed", "TESSERA_E2M1_K2", 896), ("routed2", "TESSERA_E2M1_K2", 896)]])
    # Without declared structures the historical key is unchanged: one batch.
    assert len(cell_measurement.compatible_batches(pending, weights=weights, batch_size=4)) == 1


def test_the_campaign_uses_the_shared_dense_and_routed_window_wire():
    from prismaquant import tessera_campaign as campaign
    for structure in ("dense", "routed_moe"):
        assert campaign._served_route_refusals(
            "TESSERA_E2M1_K2", {896}, ["u"],
            encode_structure={"u": structure}, projected_units={"u": {}}) == {}
    refusal = campaign._served_route_refusals(
        "TESSERA_E2M1_K2", {896}, ["u"],
        encode_structure={"u": "routed_moe"}, projected_units={})
    assert "no producer projection" in refusal["896"]["reason"]


def test_a_routed_checkpoint_identity_stamps_the_served_wire(monkeypatch):
    from types import SimpleNamespace
    from tessera import cached_unit
    from prismaquant import tessera_campaign as campaign
    fmt = "TESSERA_E2M1_K2_R896"
    family, rung = parse_tessera_format_name(fmt)
    weight = _weight()
    anchor = SimpleNamespace(qname="u", format_name=fmt, family=family.name,
                             body_rate_q256=rung, hessian_applied=False,
                             input_global_scale=None)
    kwargs = dict(weights={"u": weight}, menus={"u": [SimpleNamespace(format_name=fmt)]},
                  calibration_source=None, static_scales={})
    # The A-side gate is not under test here.
    monkeypatch.setattr(cell_measurement, "require_resumable_cell", lambda *_a, **_k: None)
    default = campaign._checkpoint_anchor_identity(anchor, **kwargs)
    assert default == cached_unit.encoding_input_identity(
        weight, "u", family.payload_grid(), int(rung))
    assert campaign._checkpoint_anchor_identity(anchor, structure="dense", **kwargs)["recipe"] == default["recipe"]
    tensor = "model.layers.3.mlp.experts.0.gate_proj.weight"
    projection = {"tensor": tensor, "source_tensor": tensor, "source_layout": "unpacked_per_expert",
                  "expert": 0, "source_slice": {"expert": 0, "selector": "whole",
                                                "transpose": False},
                  "projection": "gate_proj", "group": "w13", "rows": 64, "cols": 64}
    routed = campaign._checkpoint_anchor_identity(
        anchor, projected_units={"u": projection}, structure="routed_moe", **kwargs)
    assert routed["recipe"] == {"grid": family.payload_grid().name, "q256": int(rung),
                                **_served(fmt, "routed_moe").to_config()}
    assert routed["recipe"] == dict(added_format_recipe(fmt))
    assert default["recipe"] == dict(added_format_recipe(fmt))


def test_a_resumed_row_stamped_on_the_research_wire_is_diverted_not_adopted():
    from types import SimpleNamespace
    from prismaquant import tessera_campaign as campaign
    fmt = "TESSERA_E2M1_K2_R896"
    family, rung = parse_tessera_format_name(fmt)
    anchor = SimpleNamespace(format_name=fmt)

    def record(recipe):
        return {"identity": {"recipe": {"grid": "E2M1x2", "q256": int(rung),
                                        **recipe.to_config()}}}

    research = record(tessera_wire_recipe(family, rung))
    served = record(_served(fmt, "routed_moe"))
    reason = cell_measurement.stale_wire_reason(anchor, research, structure="routed_moe")
    assert reason is not None and "#1502" in reason
    assert cell_measurement.stale_wire_reason(anchor, served, structure="routed_moe") is None
    # A declared dense unit also requires the new served wire.
    assert cell_measurement.stale_wire_reason(anchor, research, structure=None) is None
    assert cell_measurement.stale_wire_reason(anchor, research, structure="dense") is not None
    # An unreadable receipt is left to the full identity check, which refuses it.
    assert cell_measurement.stale_wire_reason(anchor, {}, structure="routed_moe") is None
