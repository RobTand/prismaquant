"""Explicit CLI targets reach Tessera admission with each unit's topology."""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest
from prismaquant import tessera_calibration_forward as calibration_forward
from lane_table_fixtures import (
    down_convert_lane_table, lane_cells_on_one_image,
    project_lane_cells_onto_structures)


IMAGE = "example/runtime@sha256:" + "a" * 64
DENSE = "model.layers.0.self_attn.o_proj"
EXPERT = "model.layers.0.mlp.experts.0.down_proj"
SHARED = "model.layers.0.mlp.shared_expert.down_proj"


def _campaign_outputs(tmp_path, *, units=(DENSE, EXPERT, SHARED)):
    """The two files the campaign leaves beside its --cache-dir, and the
    digest it stamps on every row (#204)."""
    import torch
    from prismaquant.tessera_campaign import write_export_inputs
    cache = tmp_path / "tessera-cache"
    cache.mkdir(exist_ok=True)
    return write_export_inputs(
        cache, hessians={name: torch.eye(4) for name in units},
        hessian_rows={name: 4 for name in units}, hessian_identity=dict(_TRIPLE),
        static_scales={name: _scale() for name in units},
        static_scale_policy="legacy_6_over_calibration_amax.v1")


def _stamp_rows(argv, *, fmt, capture_sha256, units=(DENSE, EXPERT, SHARED)):
    """Give a fixture cost table the capture digest and static scale that the
    campaign's rows carry."""
    import pickle
    from pathlib import Path
    costs_path = Path(argv[argv.index("--costs") + 1])
    payload = pickle.loads(costs_path.read_bytes())
    payload.setdefault("provenance", {})["activation_static_scales"] = {
        "policy": "legacy_6_over_calibration_amax.v1",
        "units": {name: _scale() for name in units}}
    for rows in payload["costs"].values():
        rows[fmt]["hessian_identity"] = {
            "supplied": True, **_TRIPLE, "capture_sha256": capture_sha256}
        rows[fmt]["input_global_scale"] = _scale()
    costs_path.write_bytes(pickle.dumps(payload))


_TRIPLE = {"text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64, "fit_tokens": 4096}


def _scale():
    from prismaquant.nvfp4_activation_contract import input_global_scale_tensor
    return float(input_global_scale_tensor(448.0 * 6.0 / 37.5).item())


def _scope():
    from prismaquant import tessera_serving_scope
    return tessera_serving_scope


def _args(**changed):
    values = dict(tessera_runtime_image=IMAGE, tessera_execution_mode="eager",
                  tessera_residency="resident", tessera_platform="sm_121")
    values.update(changed)
    return argparse.Namespace(**values)


def _profile():
    from prismaquant.model_profiles import Qwen3Profile
    return Qwen3Profile()


def _stats():
    return {
        DENSE: dict(router_path=None, expert_id=None),
        EXPERT: dict(router_path="model.layers.0.mlp.gate", expert_id="0"),
        SHARED: dict(router_path=None, expert_id=None),
    }


def test_scope_cli_has_no_implicit_runtime_defaults():
    parser = argparse.ArgumentParser()
    _scope().add_serving_scope_arguments(parser)
    assert _scope().serving_target_from_args(parser.parse_args([])) is None


@pytest.mark.parametrize("field", [
    "tessera_runtime_image", "tessera_execution_mode", "tessera_residency",
    "tessera_platform",
])
def test_partial_runtime_target_refuses_named_missing_field(field):
    with pytest.raises(ValueError, match=field.removeprefix("tessera_")):
        _scope().serving_target_from_args(_args(**{field: None}))


def test_platform_comes_from_selected_profile_not_probe_device():
    scope = _scope()
    target = scope.serving_target_from_args(
        _args(tessera_platform=None), target_platform="sm_121")
    assert target.platform == "sm_121"
    with pytest.raises(ValueError, match="platform.*conflict"):
        scope.serving_target_from_args(_args(), target_platform="sm_120")


def test_unit_topology_preserves_dense_shared_and_routed_split():
    scope = _scope()
    contexts = scope.context_by_unit_from_stats(
        scope.serving_target_from_args(_args()), _stats(), _profile())
    assert {name: ctx.structure for name, ctx in contexts.items()} == {
        DENSE: "dense", EXPERT: "routed_moe", SHARED: "dense"}
    assert all(ctx.runtime_image == IMAGE for ctx in contexts.values())


def test_packed_expert_and_grouped_attention_are_not_the_same_structure():
    scope = _scope()
    packed = "model.layers.0.mlp.experts.gate_up_proj"
    rows = {
        packed: dict(num_experts=2, _packed_experts_module="model.layers.0.mlp.experts",
                     router_path=None, expert_id=None),
        DENSE: dict(num_groups=2, router_path=None, expert_id=None),
    }
    contexts = scope.context_by_unit_from_stats(
        scope.serving_target_from_args(_args()), rows, _profile())
    assert contexts[packed].structure == "routed_moe"
    assert contexts[DENSE].structure == "dense"


@pytest.mark.parametrize("name,row", [
    (DENSE, {}),
    (EXPERT, dict(router_path=None, expert_id=None)),
    (DENSE, dict(router_path="router", expert_id=None)),
    (DENSE, dict(router_path=None, expert_id="0")),
])
def test_ambiguous_or_conflicting_unit_topology_is_not_guessed(name, row):
    scope = _scope()
    with pytest.raises(ValueError, match=name):
        scope.context_by_unit_from_stats(
            scope.serving_target_from_args(_args()), {name: row}, _profile())


def test_unscoped_legacy_call_does_not_require_new_topology_fields():
    assert _scope().context_by_unit_from_stats(None, {"old.row": {}}, None) is None


def _cli_scope():
    return ["--tessera-runtime-image", IMAGE, "--tessera-execution-mode", "eager",
            "--tessera-residency", "resident", "--tessera-platform", "sm_121"]


class EndpointReached(Exception):
    pass


def test_campaign_main_passes_live_model_topology_to_real_menu_boundary(tmp_path, monkeypatch):
    import torch
    from prismaquant import tessera_campaign as campaign
    from prismaquant import tessera_render as render
    from prismaquant import sensitivity_probe
    import transformers

    modules = {name: torch.nn.Linear(32, 32, bias=False) for name in (DENSE, EXPERT)}
    # The campaign reads the nn.Module API: named_modules for its dense walk
    # and named_parameters for the packed-population refusal (#183), so the
    # stand-in presents both, and the parameters are those of its modules.
    model = SimpleNamespace(
        named_modules=lambda: modules.items(), eval=lambda: None,
        named_parameters=lambda: (
            (f"{name}.{attr}", parameter)
            for name, module in modules.items()
            for attr, parameter in module.named_parameters()))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(render, "tessera_encoder_hessian_status", lambda: {"accepted": True})
    monkeypatch.setattr(campaign, "_calibration_tokens", lambda *a: ([torch.ones(1, 1, dtype=torch.int64)], "fixture"))
    monkeypatch.setattr(calibration_forward, "capture_activation_statistics", lambda *a, **k: ({}, {}, {}, {}))
    monkeypatch.setattr(sensitivity_probe, "discover_moe_structure", lambda *a, **k: {
        EXPERT: ("model.layers.0.mlp.gate", "0")})
    seen = {}

    def menus(weights, targets, **kwargs):
        seen.update(kwargs)
        raise EndpointReached

    monkeypatch.setattr(campaign, "expand_menus_for_targets", menus)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]}))
    with pytest.raises(EndpointReached):
        campaign.main(["--model", str(model_dir), "--out", str(tmp_path / "cost.pkl"),
                       "--cache-dir", str(tmp_path / "cache"), "--hessian", "off", *_cli_scope()])
    assert seen["context_by_unit"][DENSE].structure == "dense"
    assert seen["context_by_unit"][EXPERT].structure == "routed_moe"


def _v5_contract(monkeypatch, *, with_experts=True):
    from prismaquant import tessera_runtime_contract as contract
    from prismaquant import tessera_menu as menu
    # A v5 fixture this file owns: the installed contract is v10, whose cells
    # carry an ``evidence`` block v5 cannot express. The population is the
    # packaged scopes re-addressed onto the structures under test (see
    # ``project_lane_cells_onto_structures``): this test is about SCOPE, not
    # about evidence, so it decides for itself whether an expert cell
    # exists. It used to relabel the dense cells as routed_moe, which lost
    # TESSERA_E4M3_K1 -- the rung these tests select -- entirely once
    # contract v23 (lane schema v10) made coverage structure-specific: the
    # dense cells publish TESSERA_E2M1_K2 only and E4M3 only as routed_moe.
    # One real image's roster first: this fixture selects A4 as well as A8,
    # so choose the uniquely published A4 image, not the v39 default (A8 only).
    # Every kept cell is flattened onto IMAGE below; no cross-image union.
    payload = down_convert_lane_table(
        project_lane_cells_onto_structures(
            lane_cells_on_one_image(
                json.loads(contract.contract_path().read_text()),
                family="TESSERA_E2M1_K2"),
            ("dense", "routed_moe") if with_experts else ("dense",)),
        "tessera.lane-eligibility.v5")
    block = payload["lane_eligibility"]
    for cell in block["cells"]:
        cell["runtime"] = {"image": IMAGE, "execution_modes": ["eager"]}
    parsed = contract._parse(payload, commit="fixture", sha="fixture", path="fixture")
    monkeypatch.setattr(menu, "tessera_runtime_contract", lambda: parsed)
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "attested")
    return payload
