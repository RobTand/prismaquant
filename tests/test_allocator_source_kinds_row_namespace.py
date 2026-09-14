"""Live-keyed probe rows resolve to the recipe-keyed source-dtype manifest.

The GLM-5.3 census allocation refused every row with
``source_kind='unknown' has no exact source footprint owner``: the scan keyed
kinds by recipe unit (``model.layers.N.*``) while glm5_next probe and cost
rows are live (``model.language_model.layers.N.*``).
"""
from __future__ import annotations

import pytest
import torch

from prismaquant.allocator_candidates import (
    _scan_source_dtype_manifest,
    source_footprint_owner_for_kind,
    source_kinds_in_row_namespace,
)
from prismaquant.model_profiles import DefaultProfile
from prismaquant.model_profiles.glm5_next import Glm5NextProfile

LIVE_DOWN = "model.language_model.layers.0.mlp.down_proj"
LIVE_FORGET = "model.language_model.layers.0.self_attn.forget_gate.f_a_proj"


def _checkpoint(tmp_path) -> str:
    from safetensors.torch import save_file

    save_file({
        f"{LIVE_DOWN}.weight": torch.zeros(4, 2, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.f_a_proj.weight":
            torch.zeros(2, 2, dtype=torch.float32),
    }, str(tmp_path / "model.safetensors"))
    return str(tmp_path)


def test_live_glm_rows_take_their_recipe_unit_kind(tmp_path):
    profile = Glm5NextProfile()
    manifest = _scan_source_dtype_manifest(_checkpoint(tmp_path), profile)
    assert manifest == {"model.layers.0.mlp.down_proj": "bf16",
                        "model.layers.0.self_attn.f_a_proj": "f32"}
    # The allocator's former lookup: the live row has no owner and refuses.
    assert source_footprint_owner_for_kind(manifest.get(LIVE_DOWN, "unknown")) is None

    resolved = source_kinds_in_row_namespace(manifest, [LIVE_DOWN, LIVE_FORGET], profile)
    assert resolved[LIVE_DOWN] == "bf16"
    assert resolved[LIVE_FORGET] == "f32"
    assert source_footprint_owner_for_kind(resolved[LIVE_DOWN]) is not None
    assert {k: resolved[k] for k in manifest} == manifest


def test_unclassified_recipe_unit_stays_absent(tmp_path):
    profile = Glm5NextProfile()
    manifest = _scan_source_dtype_manifest(_checkpoint(tmp_path), profile)
    row = "model.language_model.layers.0.mlp.gate_proj"
    assert row not in source_kinds_in_row_namespace(manifest, [row], profile)


def test_conflicting_row_kind_refuses():
    manifest = {"model.layers.0.mlp.down_proj": "bf16", LIVE_DOWN: "fp8"}
    with pytest.raises(ValueError, match="refusing to choose one"):
        source_kinds_in_row_namespace(manifest, [LIVE_DOWN], Glm5NextProfile())


def test_identity_profile_leaves_the_manifest_unchanged():
    manifest = {"model.layers.0.mlp.down_proj": "bf16"}
    assert source_kinds_in_row_namespace(
        manifest, list(manifest), DefaultProfile()) == manifest
