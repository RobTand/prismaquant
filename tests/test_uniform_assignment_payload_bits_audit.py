"""Real uniform assignment accounting; no model load, render or serving claim."""
import pytest

from prismaquant import format_registry as formats
from prismaquant.model_profiles.structure import build_model_graph
from tools.make_uniform_assignment import build_uniform_assignment, layer_config_payload
from test_make_uniform_assignment import _ToyModel, _ToyProfile, _bf16_sources


@pytest.mark.parametrize("format_name", ["FP8_DYNAMIC", "NVFP4"])
def test_uniform_report_charges_actual_covered_payload_shapes(format_name):
    profile = _ToyProfile()
    graph = build_model_graph(_ToyModel(), profile)
    result = build_uniform_assignment(
        graph, format_name, profile=profile, target_profile="research",
        source_kinds=_bf16_sources(graph),
    )
    assert result.assignment
    assert len(set(result.assignment.values())) == 1
    covered = [tensor for tensor in graph.quantizable_tensors()
               if tensor.recipe_name.removesuffix(".weight") in result.assignment]
    params = sum(tensor.shape[0] * tensor.shape[1] for tensor in covered)
    assert params == sum(result.params_by_format.values()) > 0
    expected_bits = sum(
        formats.get_format(result.assignment[tensor.recipe_name.removesuffix(".weight")])
        .memory_bytes_for_shape(tuple(tensor.shape)) * 8
        for tensor in covered
    )
    # Match the actual CLI's metadata construction and real output parser.
    payload = layer_config_payload(result, meta={
        "target_bits": result.achieved_bits,
        "achieved_bits": result.achieved_bits,
    })
    reported = payload["__prismaquant__"]
    assert reported["target_bits"] == reported["achieved_bits"]
    reported_bits = reported["achieved_bits"] * params
    assert reported_bits == pytest.approx(expected_bits, rel=1e-12, abs=1e-6), (
        format_name, reported_bits, expected_bits, params,
    )


def _small_model(width):
    import torch.nn as nn

    model = nn.Module()
    model.proj = nn.Linear(width, 8, bias=False)
    model.lm_head = nn.Linear(width, 4, bias=False)
    return model


@pytest.mark.parametrize("width", [32, 128])
def test_fp8_payload_rate_depends_on_covered_input_width(width):
    profile = _ToyProfile()
    graph = build_model_graph(_small_model(width), profile)
    result = build_uniform_assignment(
        graph, "FP8_DYNAMIC", profile=profile, target_profile="research",
        source_kinds=_bf16_sources(graph),
    )
    assert result.assignment == {"proj": formats.get_format("FP8_DYNAMIC").name}
    assert sum(result.params_by_format.values()) == 8 * width
    # Eight weight rows each carry one FP32 scale; the pinned head is omitted.
    assert result.payload_bytes == 8 * width + 8 * 4
    assert result.achieved_bits == pytest.approx(8 + 32 / width)


def test_mixed_fallback_payload_omits_structurally_excluded_members():
    profile = _ToyProfile()
    graph = build_model_graph(_ToyModel(), profile)
    result = build_uniform_assignment(
        graph, "MXFP8_E4M3", profile=profile, target_profile="research",
        source_kinds=_bf16_sources(graph),
    )
    assert set(result.assignment.values()) == {"MXFP8_E4M3", "BF16"}
    assert len(result.demoted_units) == 1
    assert result.assignment["model.layers.1.mlp.gate_proj"] == "BF16"
    assert result.assignment["model.layers.1.mlp.up_proj"] == "BF16"
    assert "model.layers.2.self_attn.q_proj" not in result.assignment
    assert "model.layers.2.self_attn.k_proj" not in result.assignment
    assert "lm_head" not in result.assignment
    covered = [tensor for tensor in graph.quantizable_tensors()
               if tensor.recipe_name.removesuffix(".weight") in result.assignment]
    expected_bytes = sum(
        formats.get_format(result.assignment[tensor.recipe_name.removesuffix(".weight")])
        .memory_bytes_for_shape(tuple(tensor.shape)) for tensor in covered
    )
    params = sum(tensor.shape[0] * tensor.shape[1] for tensor in covered)
    assert result.payload_bytes == expected_bytes
    assert sum(result.params_by_format.values()) == params
    assert result.achieved_bits == pytest.approx(8 * expected_bytes / params)


def test_empty_assignment_has_no_payload_or_covered_parameters():
    import torch.nn as nn

    model = nn.Module()
    model.lm_head = nn.Linear(32, 4, bias=False)
    profile = _ToyProfile()
    graph = build_model_graph(model, profile)
    result = build_uniform_assignment(
        graph, "FP8_DYNAMIC", profile=profile, target_profile="research",
        source_kinds=_bf16_sources(graph),
    )
    assert result.assignment == {}
    assert result.params_by_format == {}
    assert result.payload_bytes == 0
    assert result.achieved_bits == 0.0


def test_real_cli_writes_shape_aware_payload_rate_to_both_metadata_outputs(
        tmp_path, monkeypatch):
    import json

    from prismaquant.layer_config import canonicalize_assignment
    from prismaquant.model_profiles import registry
    from prismaquant import tied_embeddings
    from tools import make_uniform_assignment as uniform

    profile = _ToyProfile()
    model = _small_model(32)
    graph = build_model_graph(model, profile)
    monkeypatch.setattr(registry, "detect_profile", lambda path: profile)
    monkeypatch.setattr(uniform, "_meta_skeleton", lambda path: model)
    monkeypatch.setattr(uniform, "_scan_source_dtype_manifest",
                        lambda path, selected_profile: _bf16_sources(graph))
    monkeypatch.setattr(tied_embeddings, "lm_head_is_tied_alias",
                        lambda path, *, profile: False)
    output = tmp_path / "layer_config.json"
    report_path = tmp_path / "report.json"
    assert uniform.main([
        "--model", str(tmp_path / "source"), "--format", "FP8_DYNAMIC",
        "--target-profile", "research", "--out", str(output),
        "--report", str(report_path),
    ]) == 0
    payload = json.loads(output.read_bytes())
    report = json.loads(report_path.read_bytes())
    assert canonicalize_assignment(payload) == {
        "proj": formats.get_format("FP8_DYNAMIC").name,
    }
    assert payload["__prismaquant__"]["target_bits"] == 9.0
    assert payload["__prismaquant__"]["achieved_bits"] == 9.0
    assert report["achieved_bits"] == 9.0
    assert report["meta"] == payload["__prismaquant__"]
    assert sum(report["params_by_format"].values()) == 8 * 32
