"""Tests for the probe's shard schedule: which body, MTP, visual and head
regexes it scopes for a model."""
from __future__ import annotations

import json

from prismaquant.incremental_probe import (
    build_extended_shard_regexes,
    build_shard_schedule,
)


def test_extended_shards_skip_declared_mtp_when_index_has_no_mtp(tmp_path, capsys):
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
    }))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00001.safetensors",
        }
    }))

    regexes = build_extended_shard_regexes(
        str(tmp_path),
        1,
        include_visual=False,
        include_lm_head=False,
    )

    assert regexes == [r"model\.layers\.0\.", r"model\.layers\.1\."]
    assert not any("mtp" in regex for regex in regexes)
    assert "skipping MTP shards" in capsys.readouterr().out


def test_extended_shards_include_mtp_when_index_has_mtp_but_config_omits_count(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 2,
    }))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00001.safetensors",
            "mtp.fc.weight": "model-00001.safetensors",
            "mtp.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
        }
    }))

    regexes = build_extended_shard_regexes(
        str(tmp_path),
        1,
        include_body=False,
        include_visual=False,
        include_lm_head=False,
    )

    assert regexes == [r"(?:mtp\.fc|mtp\.layers\.0\.)"]


def test_extended_shards_use_profile_visual_prefix(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "gemma4",
        "num_hidden_layers": 2,
        "vision_config": {"num_hidden_layers": 3},
    }))

    regexes = build_extended_shard_regexes(
        str(tmp_path),
        2,
        include_body=False,
        include_mtp=False,
        include_lm_head=False,
    )

    assert regexes == [
        r"model\.vision_tower\.vision_model\.encoder\.layers\.(?:0|1|2)\.",
        r"^(?!model\.vision_tower\.vision_model\.encoder\.layers\.\d+\.)"
        r"(?:model\.vision_tower|model\.embed_vision)\.",
    ]


def test_shard_schedule_uses_profile_body_and_head_names(tmp_path, capsys):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
    }))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "layers.0.attn.wkv.weight": "model-00001.safetensors",
            "layers.1.attn.wkv.weight": "model-00001.safetensors",
            "mtp.0.attn.wkv.weight": "model-00001.safetensors",
        }
    }))

    schedule = build_shard_schedule(
        model_path=str(tmp_path),
        num_body_layers=2,
        body_layers_per_shard=1,
        body_layer_range=(0, 2),
        include_mtp=True,
        include_visual=False,
        include_lm_head=True,
        unified_body_sweep=False,
    )

    assert schedule.regexes() == [
        r"layers\.0\.",
        r"layers\.1\.",
        r"mtp\.0\.",
        r"^head$",
    ]
    assert schedule.body_layer_indices() == frozenset({0, 1})
    assert "skipping MTP" not in capsys.readouterr().out
