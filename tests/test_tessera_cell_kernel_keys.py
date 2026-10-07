"""The producer uses kernel build and module kind for cell scope."""
from types import SimpleNamespace
from prismaquant import lane_eligibility as lane


def test_matching_kernel_build_does_not_require_the_old_image():
    cell = SimpleNamespace(platform="sm_121", structure="dense", residency_modes=("resident",),
        execution_modes=("eager",), runtime_image="example/old@sha256:" + "a" * 64,
        runtime_kernel_build="build-a")
    context = lane.ServingContext(platform="sm_121", structure="dense", residency="resident",
        runtime_image="example/new@sha256:" + "b" * 64, execution_mode="eager", kernel_build="build-a")
    assert lane.cell_matches_serving_context(cell, context, serving_source_sha256=None)
    cell.runtime_kernel_build = "build-b"
    assert not lane.cell_matches_serving_context(cell, context, serving_source_sha256=None)
    cell.runtime_kernel_build = "build-a"
    cell.structure = "routed_moe"
    assert not lane.cell_matches_serving_context(cell, context, serving_source_sha256=None)


def test_legacy_context_uses_the_compatibility_map():
    context = lane.ServingContext(platform="sm_121", structure="dense", residency="resident",
        runtime_image="example/old@sha256:" + "a" * 64, execution_mode="eager")
    cell = SimpleNamespace(platform="sm_121", structure="dense", residency_modes=("resident",),
        execution_modes=("eager",), runtime_image=context.runtime_image, runtime_kernel_build="build-a")
    assert lane.cell_matches_serving_context(cell, context, serving_source_sha256=None)


def test_cell_parser_preserves_build_and_compatibility_key():
    payload = {"id": "old_cell", "platform": "sm_121", "family": "TESSERA_E4M3_K1",
        "structure": "dense", "regime": "decode", "rungs_q256": [1024],
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"], "predicates": [],
        "requires_plugin": "tessera", "activation_contract": "fp8_per_token_dynamic",
        "executes": [{"symbol": "native.kernel", "decoder": "native_decoder"}],
        "runtime": {"image": "example/old@sha256:" + "a" * 64,
                    "execution_modes": ["eager"], "kernel_build": "build-a"}}
    cell = lane.EligibilityCell.from_dict(payload, "cell", trellis_families=frozenset({"TESSERA_E4M3_K1"}),
        schema=lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V5, residency_modes=("resident", "streamed"))
    assert cell.runtime_kernel_build == "build-a"
    assert cell.as_dict()["runtime"]["kernel_build"] == "build-a"
    assert lane.cell_key_compatibility([cell])["old_cell"][:2] == ("build-a", "dense")


def test_development_projection_and_review_answer_keep_the_build_name():
    import json
    from prismaquant import tessera_runtime_contract as runtime
    payload = json.loads(runtime.contract_path().read_text())
    payload["lane_eligibility"]["cells"][0]["runtime"]["kernel_build"] = "build-a"
    parsed = runtime._parse(payload, commit="fixture", sha="fixture", path="fixture")
    assert parsed.cells[0].runtime_kernel_build == "build-a"
    answer = runtime.contract_answer(parsed)
    assert any(isinstance(value, dict) and value.get("kernel_build") == "build-a"
               for row in answer["cells"] for value in row)


def test_legacy_compatibility_key_selects_its_historical_cell():
    cell = SimpleNamespace(id="historical_cell", platform="sm_121", structure="dense",
        residency_modes=("resident",), execution_modes=("eager",),
        runtime_image="example/old@sha256:" + "a" * 64, runtime_kernel_build="",
        family="TESSERA_E4M3_K1", regime="decode")
    key = lane.cell_key_compatibility([cell])[cell.id]
    context = lane.ServingContext(platform=cell.platform, structure=cell.structure,
        residency="resident", runtime_image=cell.runtime_image, execution_mode="eager", kernel_build=key[0])
    assert lane.cell_matches_serving_context(cell, context, serving_source_sha256=None)


def test_image_only_scope_cannot_overlap_across_build_names():
    import copy
    import json
    import pytest
    from prismaquant import tessera_runtime_contract as runtime
    payload = json.loads(runtime.contract_path().read_text())
    cell = payload["lane_eligibility"]["cells"][0]
    cell["runtime"]["kernel_build"] = "build-a"
    variant = copy.deepcopy(cell)
    variant["id"] += "_other_build"
    variant["runtime"]["kernel_build"] = "build-b"
    payload["lane_eligibility"]["cells"].append(variant)
    with pytest.raises(lane.LaneEligibilityError, match="overlapping serving scopes"):
        runtime._parse(payload, commit="fixture", sha="fixture", path="fixture")
