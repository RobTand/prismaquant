"""Declared Gemma visual namespaces must reach the real allocator policy."""
import json

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import allocator
from prismaquant.layer_config import load_assignment
from prismaquant.model_profiles.registry import detect_profile
import test_visual_allocator_units_1936 as visual_fixture
from test_visual_shard_coverage import CASES, _config

VISION = "model.vision_tower.vision_model.encoder.layers.0.attn.qkv"
PROJECTOR = "model.embed_vision.projection"


def _gemma_inputs(tmp_path, monkeypatch, **kwargs):
    # Reuse the existing measured-row/contract control and change only the
    # declared profile/visual names. No measured production inputs are invented.
    monkeypatch.setattr(visual_fixture, "VISION", VISION)
    monkeypatch.setattr(visual_fixture, "MERGER", PROJECTOR)
    output, probe = visual_fixture._inputs(tmp_path, monkeypatch, **kwargs)
    (tmp_path / "model/config.json").write_text(json.dumps({
        "model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
        "vision_config": {"num_hidden_layers": 1}}))
    return output, probe


def test_gemma_measured_vision_and_projector_keep_independent_choices(tmp_path, monkeypatch):
    output, probe = _gemma_inputs(tmp_path, monkeypatch)
    allocator.main()
    assignment = load_assignment(output)
    assert assignment[VISION] == visual_fixture.DENSE_RUNG
    assert assignment[PROJECTOR] == visual_fixture.VALUE_RUNG
    attribution = json.loads((tmp_path / "attribution.json").read_text())
    assert attribution["n_body_linears"] == len(probe["stats"])


def test_gemma_partial_measured_population_refuses_uniform_substitution(tmp_path, monkeypatch):
    output, _ = _gemma_inputs(tmp_path, monkeypatch, missing_cost=PROJECTOR)
    with pytest.raises(SystemExit, match="visual Fisher.*incomplete.*embed_vision"):
        allocator.main()
    assert not output.exists()


def test_gemma_uniform_control_excludes_visual_rows_from_decision_accounting(tmp_path, monkeypatch):
    output, _ = _gemma_inputs(tmp_path, monkeypatch, sensitivity="uniform")
    allocator.main()
    assignment = load_assignment(output)
    assert assignment[VISION] == assignment[PROJECTOR] == "BF16"
    attribution = json.loads((tmp_path / "attribution.json").read_text())
    assert attribution["n_body_linears"] == 3


def test_gemma_mismatched_source_shape_refuses_before_selection(tmp_path, monkeypatch):
    output, _ = _gemma_inputs(tmp_path, monkeypatch, source_shape_mismatch=True)
    with pytest.raises(SystemExit, match="visual Fisher source roster.*mismatched.*embed_vision"):
        allocator.main()
    assert not output.exists()


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_visual_classifier_uses_declared_roots_and_existing_model_alias(tmp_path, case):
    model_type, architecture, block_prefix, extras = case
    profile = detect_profile(_config(tmp_path, model_type, architecture))
    for name in (f"{block_prefix}.0.attn.qkv", *extras):
        assert allocator._is_visual_linear(name, profile)
        assert allocator._is_visual_linear(name.removeprefix("model."), profile)
    for name in ("model.layers.0.mlp.up_proj", "model.audio_tower.projection",
                 "model.embed_audio.projection", "model.visual_extra.projection",
                 "model.vision_tower_extra.projection", "vis_attn_qkv", "merger_fc1"):
        # vis_*/merger_* are role tags, not declared module namespaces.
        assert not allocator._is_visual_linear(name, profile)


@pytest.mark.parametrize("indexed", [False, True])
def test_gemma_source_header_census_covers_both_roots_and_exact_shapes(tmp_path, indexed):
    profile = detect_profile(_config(tmp_path, "gemma4", "Gemma4ForConditionalGeneration"))
    tensors = {
        VISION + ".weight": torch.zeros((12, 4), dtype=torch.bfloat16),
        PROJECTOR + ".weight": torch.zeros((6, 4), dtype=torch.bfloat16),
        "model.vision_tower.vision_model.pos_embed.weight": torch.zeros((8, 4)),
        "model.vision_tower.vision_model.norm.weight": torch.zeros((4,)),
        "model.vision_tower.vision_model.patch_conv.weight": torch.zeros((4, 3, 2, 2)),
        "model.embed_audio.projection.weight": torch.zeros((6, 4)),
        "model.layers.0.mlp.up_proj.weight": torch.zeros((6, 4)),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    if indexed:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in tensors}}))
    stats = allocator.discover_visual_linear_stats_from_source(str(tmp_path), strict=True,
                                                               profile=profile)
    assert set(stats) == {VISION, PROJECTOR}
    assert (stats[VISION]["out_features"], stats[VISION]["in_features"]) == (12, 4)
    assert stats[PROJECTOR]["n_params"] == 24
    assert allocator.discover_visual_linears_from_source(str(tmp_path), profile=profile) == sorted(stats)
