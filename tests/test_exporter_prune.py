"""Unit tests for the exporter's prune-manifest plumbing.

Covers the pure helpers in `export_native_compressed`:
  - manifest load (missing file → empty; valid JSON round-trip)
  - parent-index build from router-keyed manifest
  - resolve-action for per-expert Linears (router / drop / reindex /
    passthrough)
  - router-weight out-dim shrink + size-mismatch guard
  - config.json expert-count field patching (including nested
    `text_config` common in multimodal-umbrella archs)

Does NOT attempt to spin up a real model — the surgery on the
streaming loop is covered end-to-end by the Phase 3 Qwen3.5 smoke
run, not by unit tests (would require a full MoE skeleton on disk).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from prismaquant.export_native_compressed import (
    _index_prune_by_parent,
    _load_prune_manifest,
    _resolve_linear_prune_action,
    _resolve_packed_experts_prune,
    _shrink_router_weight,
    write_config_with_quantization,
)


def _manifest_fixture():
    """4-expert MoE on layer 0, one entry keyed by router_qname."""
    return {
        "model.layers.0.mlp.gate": {
            "num_experts_orig": 4,
            "num_experts_kept": 2,
            "pruned_expert_ids": [1, 3],
            "kept_expert_ids": [0, 2],
            "orig_to_new_eid": {"0": 0, "2": 1},
        }
    }


def test_load_prune_manifest_missing_file(tmp_path):
    assert _load_prune_manifest(None) == {}
    assert _load_prune_manifest(tmp_path / "does_not_exist.json") == {}


def test_load_prune_manifest_valid_json(tmp_path):
    m = _manifest_fixture()
    p = tmp_path / "m.json"
    with open(p, "w") as f:
        json.dump(m, f)
    got = _load_prune_manifest(p)
    assert got == m


def test_load_prune_manifest_rejects_non_object(tmp_path):
    p = tmp_path / "bad.json"
    with open(p, "w") as f:
        json.dump([1, 2, 3], f)
    with pytest.raises(ValueError, match="not a JSON object"):
        _load_prune_manifest(p)


def test_index_prune_by_parent_groups_router_under_shared_parent():
    m = _manifest_fixture()
    by_parent = _index_prune_by_parent(m)
    assert "model.layers.0.mlp" in by_parent
    entry = by_parent["model.layers.0.mlp"]
    assert entry["router_qname"] == "model.layers.0.mlp.gate"
    assert entry["num_experts_orig"] == 4
    assert entry["kept_expert_ids"] == [0, 2]


def test_resolve_action_router():
    by_parent = _index_prune_by_parent(_manifest_fixture())
    got = _resolve_linear_prune_action("model.layers.0.mlp.gate", by_parent)
    assert got is not None
    kind, entry = got
    assert kind == "router"
    assert entry["router_qname"] == "model.layers.0.mlp.gate"


def test_resolve_action_drop_pruned_expert():
    by_parent = _index_prune_by_parent(_manifest_fixture())
    for qname in (
        "model.layers.0.mlp.experts.1.gate_proj",
        "model.layers.0.mlp.experts.3.up_proj",
        "model.layers.0.mlp.experts.1.down_proj",
    ):
        got = _resolve_linear_prune_action(qname, by_parent)
        assert got is not None, qname
        kind, _ = got
        assert kind == "drop", qname


def test_resolve_action_reindex_kept_expert():
    by_parent = _index_prune_by_parent(_manifest_fixture())
    got = _resolve_linear_prune_action(
        "model.layers.0.mlp.experts.2.gate_proj", by_parent,
    )
    assert got is not None
    kind, entry = got
    assert kind == "reindex"
    # orig eid 2 → new eid 1 per fixture
    assert entry["orig_eid"] == 2
    assert entry["new_eid"] == 1
    assert entry["new_full"] == "model.layers.0.mlp.experts.1.gate_proj"


def test_resolve_action_returns_none_for_unrelated_qname():
    by_parent = _index_prune_by_parent(_manifest_fixture())
    # A dense attention Linear — no prune.
    assert _resolve_linear_prune_action(
        "model.layers.0.self_attn.q_proj", by_parent,
    ) is None
    # A layer not in the manifest.
    assert _resolve_linear_prune_action(
        "model.layers.5.mlp.experts.0.gate_proj", by_parent,
    ) is None


def test_resolve_action_empty_manifest_is_noop():
    # Guard: no manifest means every qname resolves to None, so the
    # non-prune export path is identical to the pre-Phase-2c behavior.
    assert _resolve_linear_prune_action(
        "model.layers.0.mlp.experts.0.gate_proj", {},
    ) is None


def test_resolve_packed_experts_prune():
    by_parent = _index_prune_by_parent(_manifest_fixture())
    got = _resolve_packed_experts_prune("model.layers.0.mlp.experts", by_parent)
    assert got is not None
    assert got["num_experts_orig"] == 4
    # A layer not in the manifest.
    assert _resolve_packed_experts_prune(
        "model.layers.7.mlp.experts", by_parent,
    ) is None


def test_shrink_router_weight_keeps_correct_rows():
    # Router for 4 experts, 8-dim hidden. Rows labeled by expert id.
    w = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    mod = nn.Linear(8, 4, bias=False)
    mod.weight.data = w.clone()
    entry = _manifest_fixture()["model.layers.0.mlp.gate"]
    shrunk = _shrink_router_weight(mod, entry)
    assert shrunk.shape == (2, 8)
    # kept = [0, 2], so shrunk must equal original rows 0 and 2.
    assert torch.equal(shrunk[0], w[0])
    assert torch.equal(shrunk[1], w[2])


def test_shrink_router_weight_rejects_size_mismatch():
    # Manifest says 4 experts, but router has 5 rows. Silent shrink
    # would produce an unservable artifact — must loud-crash.
    w = torch.randn(5, 8)
    mod = nn.Linear(8, 5, bias=False)
    mod.weight.data = w
    entry = _manifest_fixture()["model.layers.0.mlp.gate"]
    with pytest.raises(RuntimeError, match="num_experts_orig"):
        _shrink_router_weight(mod, entry)


def test_write_config_patches_num_experts_fields(tmp_path):
    # Fake source model dir with a minimal config.json carrying both
    # flat and text_config nested expert counts.
    src = tmp_path / "src"
    src.mkdir()
    src_cfg = {
        "model_type": "qwen3_5_moe",
        "num_experts": 4,
        "num_routed_experts": 4,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_local_experts": 4,
        },
    }
    with open(src / "config.json", "w") as f:
        json.dump(src_cfg, f)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Monkey-patch detect_profile since we haven't built a real
    # model; write_config_with_quantization only uses it to get a
    # profile argument for build_quantization_config.
    import prismaquant.export_native_compressed as enc
    from prismaquant.model_profiles import DefaultProfile

    orig = enc.__dict__.get("detect_profile")

    def _fake_detect_profile(_p):
        return DefaultProfile()

    # Insert the name into the module namespace the function looks up
    # at call time (it uses `from .model_profiles import detect_profile`
    # inside the function body, so we patch the module it imports from).
    import prismaquant.model_profiles as mp
    mp.detect_profile = _fake_detect_profile
    try:
        write_config_with_quantization(
            str(src), out_dir,
            assignment={},  # empty → no quantization_config injected
            bf16_passthrough=set(),
            prune_manifest=_manifest_fixture(),
        )
    finally:
        # Restore (best-effort — tests run in-process, don't leak).
        if orig is not None:
            mp.detect_profile = orig

    with open(out_dir / "config.json") as f:
        patched = json.load(f)
    # 4 → 2 in all three fields
    assert patched["num_experts"] == 2
    assert patched["num_routed_experts"] == 2
    assert patched["text_config"]["num_local_experts"] == 2


def test_write_config_rejects_mixed_kept_counts(tmp_path):
    """All MoE layers must share a kept count — the HF config holds
    a single scalar per field."""
    src = tmp_path / "src"
    src.mkdir()
    with open(src / "config.json", "w") as f:
        json.dump({"num_experts": 4}, f)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    mixed = {
        "model.layers.0.mlp.gate": {
            "num_experts_orig": 4, "num_experts_kept": 2,
            "pruned_expert_ids": [1, 3], "kept_expert_ids": [0, 2],
            "orig_to_new_eid": {"0": 0, "2": 1},
        },
        "model.layers.1.mlp.gate": {
            "num_experts_orig": 4, "num_experts_kept": 3,
            "pruned_expert_ids": [2], "kept_expert_ids": [0, 1, 3],
            "orig_to_new_eid": {"0": 0, "1": 1, "3": 2},
        },
    }
    import prismaquant.model_profiles as mp
    from prismaquant.model_profiles import DefaultProfile
    mp.detect_profile = lambda _p: DefaultProfile()
    with pytest.raises(RuntimeError, match="inconsistent"):
        write_config_with_quantization(
            str(src), out_dir,
            assignment={}, bf16_passthrough=set(),
            prune_manifest=mixed,
        )
