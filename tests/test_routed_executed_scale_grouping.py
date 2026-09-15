"""The routed NVFP4 static activation-scale export guard (RobTand/prismaquant#624).

The campaign prices one static ``input_global_scale`` per unit, so on the
Tessera routed NVFP4 wire it prices one per expert projection.  The routed
stage takes one per ``(module, stage)`` (vLLM FLASHINFER_CUTLASS behind Tessera
#507's ``nvfp4_moe_route``).  Until routed cells are rescored under that
grouping, a routed allocation must declare ``per_unit.v1`` explicitly, and the
export records it as not qualified.  The rescoring itself is a design
(``docs/design/routed_executed_scale_grouping_2026-09-14.md``).
"""
import hashlib
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from prismaquant import nvfp4_activation_contract as nac
from prismaquant import tessera_export_lane as export

ROUTED_FMT = "TESSERA_E2M1_K2_R896"
A8_FMT = "TESSERA_E4M3_K1_R1024"
DENSE = "model.layers.0.self_attn.o_proj"


def _expert(layer, index, proj):
    return f"model.layers.{layer}.mlp.experts.{index}.{proj}"


def _per_unit_stamp():
    return {"schema": nac.ROUTED_EXECUTED_SCALE_GROUPING_SCHEMA,
            "grouping": nac.ACTIVATION_SCALE_GROUPING_PER_UNIT}


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------

def test_gate_and_up_share_one_group_and_down_is_its_own():
    """w13 (gate+up) and w2 (down) are two different executed tensors."""
    w13 = nac.routed_expert_scale_group(_expert(3, 0, "gate_proj"))
    w13_up = nac.routed_expert_scale_group(_expert(3, 7, "up_proj"))
    w2 = nac.routed_expert_scale_group(_expert(3, 0, "down_proj"))
    assert w13 is not None and w2 is not None
    assert w13[2] == "w13" and w2[2] == "w2"
    assert w13[0] == w13_up[0]
    assert w13[0] != w2[0]
    assert w13[1] == w2[1] == "model.layers.3.mlp.experts"


def test_the_group_key_separates_layers_and_stages():
    down3 = nac.routed_expert_scale_group(_expert(3, 0, "down_proj"))[0]
    down4 = nac.routed_expert_scale_group(_expert(4, 0, "down_proj"))[0]
    gate3 = nac.routed_expert_scale_group(_expert(3, 0, "gate_proj"))[0]
    assert len({down3, down4, gate3}) == 3


def test_dense_and_native_packed_names_are_not_per_expert():
    for name in ("model.layers.0.mlp.down_proj",
                 "model.layers.3.mlp.experts.gate_up_proj",
                 "model.layers.3.mlp.experts.down_proj"):
        assert not nac.is_routed_expert_projection_name(name)
        assert nac.routed_expert_scale_group(name) is None


def test_an_unknown_per_expert_leaf_is_per_expert_but_ungroupable():
    """The shape and the group are separate answers, so callers can refuse."""
    name = _expert(3, 0, "bogus_proj")
    assert nac.is_routed_expert_projection_name(name)
    assert nac.routed_expert_scale_group(name) is None
    with pytest.raises(ValueError, match="no routed-MoE"):
        nac.routed_static_scale_grouping([DENSE, name])


def test_the_declaration_producer():
    assert nac.routed_static_scale_grouping([DENSE]) is None
    assert nac.routed_static_scale_grouping([]) is None
    assert nac.routed_static_scale_grouping(
        [DENSE, _expert(3, 0, "down_proj")]) == _per_unit_stamp()


# ---------------------------------------------------------------------------
# The export gate
# ---------------------------------------------------------------------------

def _layer_config(tmp_path, *, units, grouping, fmt=ROUTED_FMT):
    payload = {name: {"data_type": "tessera", "bits": 4,
                      "tessera_format": fmt} for name in units}
    scales_block = {"schema": export.PRICED_STATIC_SCALES_SCHEMA,
                    "units": dict(units)}
    if grouping is not None:
        scales_block["activation_scale_grouping"] = grouping
    payload["__prismaquant__"] = {
        "tessera_hessian": {"supplied": False, "text_sha256": "a" * 64,
                            "fit_ids_sha256": "b" * 64, "fit_tokens": 4096},
        "tessera_activation_static_scales": scales_block,
    }
    path = tmp_path / "layer_config.json"
    path.write_text(json.dumps(payload))
    return path


def _scales_file(tmp_path, units):
    from prismaquant.tessera_campaign import write_export_inputs

    _capture, scales, _digest = write_export_inputs(
        tmp_path, hessians=None, hessian_rows={}, hessian_identity={},
        static_scales=dict(units),
        static_scale_policy="legacy_6_over_calibration_amax.v1")
    return scales


def _routed_units():
    return {_expert(3, 0, "down_proj"): 0.5, _expert(3, 1, "down_proj"): 2.0}


def test_an_undeclared_routed_allocation_is_refused(tmp_path):
    """The regression #624 is about: absence was the silent serving claim."""
    units = _routed_units()
    config = _layer_config(tmp_path, units=units, grouping=None)
    scales = _scales_file(tmp_path, units)
    with pytest.raises(export.TesseraExportLaneError,
                       match="does not declare which activation-scale grouping"):
        export.require_priced_export_inputs(config, input_scales_path=scales)


def test_an_explicit_per_unit_declaration_exports_unqualified(tmp_path):
    units = _routed_units()
    config = _layer_config(tmp_path, units=units, grouping=_per_unit_stamp())
    scales = _scales_file(tmp_path, units)
    report = export.require_priced_export_inputs(config, input_scales_path=scales)
    assert report["activation_scale_grouping"] == (
        nac.ACTIVATION_SCALE_GROUPING_PER_UNIT)
    assert report["activation_scale_grouping_qualified"] is False
    assert report["activation_scale_grouping_routed_units"] == 2


def test_any_other_declaration_is_refused(tmp_path):
    """Including a collapsed/served label: nothing rescored or attests it."""
    units = _routed_units()
    for grouping, match in (
            ({**_per_unit_stamp(), "grouping": "executed_routed_collapse.v1"},
             "exports only"),
            ({**_per_unit_stamp(), "grouping": None}, "exports only"),
            ({"grouping": nac.ACTIVATION_SCALE_GROUPING_PER_UNIT}, "schema is"),
            ("per_unit.v1", "must be an object")):
        config = _layer_config(tmp_path, units=units, grouping=grouping)
        scales = _scales_file(tmp_path, units)
        with pytest.raises(export.TesseraExportLaneError, match=match):
            export.require_priced_export_inputs(config, input_scales_path=scales)


def test_an_ungroupable_per_expert_unit_is_refused_even_when_declared(tmp_path):
    """Fail closed: a per-expert spelling with no group is never read as dense."""
    units = {_expert(3, 0, "bogus_proj"): 0.5}
    config = _layer_config(tmp_path, units=units, grouping=_per_unit_stamp())
    scales = _scales_file(tmp_path, units)
    with pytest.raises(export.TesseraExportLaneError, match="resolve to no routed-MoE"):
        export.require_priced_export_inputs(config, input_scales_path=scales)


def test_a_dense_only_allocation_needs_no_declaration(tmp_path):
    units = {DENSE: 0.25}
    config = _layer_config(tmp_path, units=units, grouping=None)
    scales = _scales_file(tmp_path, units)
    report = export.require_priced_export_inputs(config, input_scales_path=scales)
    assert "activation_scale_grouping" not in report


def test_an_a8_routed_allocation_never_reaches_the_guard(tmp_path):
    """No static activation contract on the E4M3 route: no scales, no guard."""
    payload = {_expert(3, 0, "down_proj"): {"data_type": "tessera", "bits": 8,
                                            "tessera_format": A8_FMT}}
    payload["__prismaquant__"] = {
        "tessera_hessian": {"supplied": False, "text_sha256": "a" * 64,
                            "fit_ids_sha256": "b" * 64, "fit_tokens": 4096}}
    config = tmp_path / "layer_config.json"
    config.write_text(json.dumps(payload))
    report = export.require_priced_export_inputs(config)
    assert report["static_activation_contract_units"] == 0
    assert "activation_scale_grouping" not in report


def test_the_build_anchor_carries_the_unqualified_stamp_beside_priced_inputs(
        tmp_path, monkeypatch, capsys):
    """The ship record sees it; Tessera's closed priced_inputs block does not change."""
    lane = export
    for attr, value in (("require_declared_structure", lambda _: "dense"),
                        ("require_serving_target", lambda _: None),
                        ("require_executes_derived_from_contract", lambda: ()),
                        ("require_producer_tools", lambda: ()),
                        ("require_producer_repo_is_pinned", lambda: ()),
                        ("require_release_pin", lambda: None),
                        ("require_assignment_scope", lambda *a, **k: None)):
        monkeypatch.setattr(lane, attr, value)
    from prismaquant import tessera_serving_runtime_pin as pin
    monkeypatch.setattr(pin, "load_tessera_serving_runtime_pin",
                        lambda: SimpleNamespace(version="fixture", commit="f" * 40))
    units = _routed_units()
    config = _layer_config(tmp_path, units=units, grouping=_per_unit_stamp())
    scales = _scales_file(tmp_path, units)
    build = tmp_path / "build.json"
    capsys.readouterr()
    assert lane.main(["--model", str(tmp_path), "--assignment", str(config),
                      "--write-build-json", str(build), "--print-build-sha256",
                      "--input-scales", str(scales)]) == 0
    assert capsys.readouterr().out.strip() == hashlib.sha256(
        build.read_bytes()).hexdigest()
    anchor = json.loads(build.read_bytes())
    assert set(anchor["priced_inputs"]) == {
        "schema", "hessian_capture_sha256", "input_global_scales"}
    assert anchor["tessera_activation_scale_grouping"] == {
        "schema": nac.ROUTED_EXECUTED_SCALE_GROUPING_SCHEMA,
        "grouping": nac.ACTIVATION_SCALE_GROUPING_PER_UNIT,
        "qualified": False, "routed_units": 2}


# ---------------------------------------------------------------------------
# The producers declare what they priced
# ---------------------------------------------------------------------------

def test_the_allocator_block_declares_per_unit_for_routed_static_rows_only():
    from prismaquant.tessera_menu import priced_static_scales

    routed = _expert(3, 0, "down_proj")
    assignment = {routed: ROUTED_FMT, DENSE: ROUTED_FMT}
    costs = {routed: {ROUTED_FMT: {"input_global_scale": 0.5}},
             DENSE: {ROUTED_FMT: {"input_global_scale": 0.25}}}
    assert priced_static_scales(assignment, costs)[
        "activation_scale_grouping"] == _per_unit_stamp()
    # Dense-only: one scale per executed tensor already, no declaration.
    assert "activation_scale_grouping" not in priced_static_scales(
        {DENSE: ROUTED_FMT}, costs)
    # An A8 routed row carries no static scale, so the block is unchanged.
    a8 = {routed: {A8_FMT: {"output_mse": 0.1}}}
    assert priced_static_scales({routed: A8_FMT}, a8) == {
        "schema": export.PRICED_STATIC_SCALES_SCHEMA, "units": {}}


def test_selected_wire_completion_declares_only_for_static_contract_selections():
    from prismaquant.tessera_materialization import _static_scale_grouping

    routed = _expert(3, 0, "down_proj")
    scale_units = {routed: 0.5, DENSE: 0.25}
    assert _static_scale_grouping(scale_units, {routed: ROUTED_FMT, DENSE: A8_FMT}) == (
        _per_unit_stamp())
    # The campaign's scale table covers every unit; an A8 selection adds no key.
    assert _static_scale_grouping(scale_units, {routed: A8_FMT, DENSE: A8_FMT}) is None
    assert _static_scale_grouping(scale_units, {routed: "BF16"}) is None
