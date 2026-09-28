"""Owner-digest binding for the selected-cache assignment (PQ #1613-family P1).

The selected assignment's identity is
``footprint.assignment_serialization_sha256`` over the canonical
unit-to-format mapping -- the same digest the allocator stamps as
``selection_assignment_sha256`` -- not the layer_config file's raw bytes.
These tests pin that: a pristine allocator-emitted file binds by its stamp,
a post-stamp-diverged file refuses with both digests named, and an
unstamped file refuses instead of binding nothing.
"""
from __future__ import annotations

import json
import os

import pytest

from tools.build_tessera_selected_cache import _bind_selected_assignment


A8_BODY_MTP = (
    "/mnt/shared/tessera-runs/moe/glm53-pact-uniform-arms-20260927"
    "/a8/body-mtp-v39/layer_config.json"
)
A8_STAMP = "66b321d8e0e9c7465861fd5f30a261f8fcf7a74d3b8d5ee278a7b7e76b92d520"

STALE_STAMP_PICK = (  # pre-#1632 output: stamp binds the body without layer 45
    "/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928"
    "/corrected/accuracy/layer_config.json"
)
STALE_STAMP = "086c73026b18518c8cfe2784282fb384ed27e74fb970106f106238c56b6ebc6e"

RESTAMPED_PICK = (  # the release pick re-allocated on the #1632 fix (PB 39b2cf0c)
    "/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928"
    "/corrected/accuracy-1632/layer_config.json"
)
RESTAMPED_STAMP = "69d7c894"  # prefix; the full digest is read from the stamp below

requires_shared_fixtures = pytest.mark.skipif(
    not (os.path.exists(A8_BODY_MTP) and os.path.exists(STALE_STAMP_PICK) and os.path.exists(RESTAMPED_PICK)),
    reason="needs /mnt/shared fixtures (fleet-only; CI has no shared mount)",
)


@requires_shared_fixtures
def test_pristine_allocator_output_binds_by_stamp():
    stamp, assignment, _meta = _bind_selected_assignment(A8_BODY_MTP, A8_STAMP)
    assert assignment
    assert stamp["selection_assignment_sha256"] == A8_STAMP
    assert stamp["budget_bytes"] == 175642157752


@requires_shared_fixtures
def test_diverged_pick_refuses_naming_both_digests():
    with pytest.raises(ValueError, match="086c7302.*69d7c894"):
        _bind_selected_assignment(STALE_STAMP_PICK, STALE_STAMP)


def test_unstamped_selection_refuses(tmp_path):
    payload = {"model.language_model.layers.0.mlp.down_proj": {
        "data_type": "tessera", "bits": 5,
        "tessera_format": "TESSERA_E4M3_K1_R1024"}}
    path = tmp_path / "layer_config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="no whole-artifact budget stamp"):
        _bind_selected_assignment(str(path), "0" * 64)


@requires_shared_fixtures
def test_flag_must_name_the_stamped_digest():
    with pytest.raises(ValueError, match="flag names"):
        _bind_selected_assignment(A8_BODY_MTP, "0" * 64)


@requires_shared_fixtures
def test_restamped_release_pick_binds_its_mtp_bearing_digest():
    """After #1632 the stamp covers layer 45, so the release pick binds."""
    stamped = json.loads(open(RESTAMPED_PICK).read())["__prismaquant__"][
        "whole_artifact_budget"]["selection_assignment_sha256"]
    assert stamped.startswith(RESTAMPED_STAMP)
    stamp, assignment, _meta = _bind_selected_assignment(RESTAMPED_PICK, stamped)
    assert stamp["selection_assignment_sha256"] == stamped
    assert any(".layers.45." in name for name in assignment)
