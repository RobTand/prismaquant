from __future__ import annotations

import json

import pytest

from prismaquant.export_native_compressed import _load_prune_manifest
from prismaquant.schemas import (
    SchemaValidationError,
    validate_cost_payload,
    validate_layer_config_payload,
    validate_probe_payload,
    validate_prune_manifest_payload,
)


def _probe_payload():
    return {
        "stats": {
            "model.layers.0.mlp.experts.gate_up_proj": {
                "h_trace": 0.25,
                "n_params": 64,
                "in_features": 8,
                "out_features": 8,
                "num_experts": 4,
            }
        },
        "meta": {"model": "/models/tiny", "top_k": 2},
        "expert_saliency": {
            "model.layers.0.mlp.gate": {"0": 0.1, "1": 0.2, "2": 0.0, "3": 0.4}
        },
        "expert_info": {
            "model.layers.0.mlp.experts.0.gate_proj": (
                "model.layers.0.mlp.gate",
                "0",
            )
        },
    }


def _cost_payload():
    return {
        "formats": ["NVFP4", "BF16"],
        "costs": {
            "model.layers.0.mlp.experts.gate_up_proj": {
                "NVFP4": {"weight_mse": 0.01, "predicted_dloss": 0.001},
                "BF16": {"weight_mse": 0.0},
            }
        },
    }


def _layer_config_payload():
    return {
        "model.layers.0.mlp.experts.gate_up_proj": {
            "data_type": "nv_fp",
            "bits": 4,
            "group_size": 16,
        },
        "model.layers.0.self_attn.o_proj": "BF16",
        "model.layers.0.mlp.down_proj": 8,
    }


def _prune_manifest_payload():
    return {
        "model.layers.0.mlp.gate": {
            "num_experts_orig": 4,
            "num_experts_kept": 2,
            "pruned_expert_ids": [1, 3],
            "kept_expert_ids": [0, 2],
            "orig_to_new_eid": {"0": 0, "2": 1},
        }
    }


def test_valid_handoff_payloads_pass():
    assert validate_probe_payload(_probe_payload(), "probe.pkl") is not None
    assert validate_cost_payload(_cost_payload(), "cost.pkl") is not None
    assert validate_layer_config_payload(_layer_config_payload(), "layer.json") is not None
    assert validate_prune_manifest_payload(_prune_manifest_payload(), "prune.json") is not None


def test_probe_payload_rejects_missing_required_stat_field():
    payload = _probe_payload()
    del payload["stats"]["model.layers.0.mlp.experts.gate_up_proj"]["h_trace"]
    with pytest.raises(SchemaValidationError, match="h_trace"):
        validate_probe_payload(payload, "probe.pkl")


def test_cost_payload_rejects_usable_entry_without_error_or_signal():
    payload = _cost_payload()
    payload["costs"]["model.layers.0.mlp.experts.gate_up_proj"]["NVFP4"] = {}
    with pytest.raises(SchemaValidationError, match="usable cost entry"):
        validate_cost_payload(payload, "cost.pkl")


def test_layer_config_rejects_malformed_dict_entry():
    payload = _layer_config_payload()
    payload["model.layers.0.self_attn.q_proj"] = {"bits": 4}
    with pytest.raises(SchemaValidationError, match="data_type"):
        validate_layer_config_payload(payload, "layer.json")


def test_prune_manifest_rejects_inconsistent_reindex_map():
    payload = _prune_manifest_payload()
    payload["model.layers.0.mlp.gate"]["orig_to_new_eid"] = {"0": 0, "1": 1}
    with pytest.raises(SchemaValidationError, match="keys must exactly match"):
        validate_prune_manifest_payload(payload, "prune.json")


def test_exporter_prune_loader_validates_manifest_schema(tmp_path):
    path = tmp_path / "bad.prune.json"
    payload = _prune_manifest_payload()
    payload["model.layers.0.mlp.gate"]["kept_expert_ids"] = [0, 1]
    path.write_text(json.dumps(payload))
    with pytest.raises(SchemaValidationError, match="overlap"):
        _load_prune_manifest(path)
