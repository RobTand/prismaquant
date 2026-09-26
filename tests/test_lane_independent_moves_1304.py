"""Pin the helpers moved out of the retired codebook lane's modules (#1304).

The whole-artifact budget helpers moved from ``nvfp4_cb_footprint`` to
``footprint``, and the probe-derived imatrix helpers moved from ``cb_imatrix``
to ``moe_imatrix``. The move is a pure refactor: every expected value below was
computed by the pre-move functions on ``origin/main`` 253cc885c28, so a drift
in any digest, stamp field, or provenance record fails here.
"""
from __future__ import annotations

import importlib.util

import pytest
import torch

from prismaquant import footprint, moe_imatrix

ASSIGNMENT = {
    "model.layers.0.mlp.down_proj": "nvfp4",
    "model.layers.0.self_attn.q_proj": "FP8_DYNAMIC",
    "lm_head": "bf16",
}
ASSIGNMENT_SHA256 = (
    "7e0ffd2a41f1fcfa26675840e9b4570f84be0e6ef83d1435b9784f9d1a2da560"
)
STAMP = {
    "budget_bytes": 10000,
    "excluded_source_prefixes": ["mtp.", "visual."],
    "final_contract": "stat_all_regular_files_recursive_fail_closed",
    "schema": "prismaquant.whole_artifact_budget.v2",
    "scope": "all_regular_files_recursive",
    "selection_assignment_sha256": ASSIGNMENT_SHA256,
    "selection_contract": (
        "tensor_payload_plus_operator_supplied_non_tensor_reserve"
    ),
    "selection_non_tensor_reserve_bytes": 1000,
    "selection_tensor_payload_bytes": 6000,
    "selection_whole_artifact_upper_bound_bytes": 7000,
}


def _stamp():
    return footprint.whole_artifact_budget_stamp(
        budget_bytes=10_000,
        selection_tensor_payload_bytes=6000,
        selection_non_tensor_reserve_bytes=1000,
        selection_assignment=ASSIGNMENT,
        excluded_source_prefixes=("mtp.", "visual.", "mtp."),
    )


def _imatrix_sha():
    return moe_imatrix.canonical_imatrix_sha256({
        "a": torch.tensor([1.0, 2.0, 3.0]),
        "b": torch.tensor([[0.5, 0.25], [4.0, 8.0]]),
    })


def _probe_provenance():
    _values, provenance = moe_imatrix.imatrix_from_probe_stats({
        "dense": {
            "act_sq_sum": torch.tensor([4.0, 8.0, 12.0]),
            "n_tokens_seen": 4,
            "in_features": 3,
        },
        "experts": {
            "expert_act_sq_sum": torch.tensor([[2.0, 4.0], [9.0, 12.0]]),
            "expert_tokens": torch.tensor([2, 3]),
        },
    })
    return provenance


@pytest.mark.parametrize(
    ("name", "compute", "expected"),
    [
        (
            "assignment_serialization_sha256",
            lambda: footprint.assignment_serialization_sha256(ASSIGNMENT),
            ASSIGNMENT_SHA256,
        ),
        ("whole_artifact_budget_stamp", _stamp, STAMP),
        (
            "budget_stamp_excluded_prefixes",
            lambda: footprint.budget_stamp_excluded_prefixes(_stamp()),
            ("mtp.", "visual."),
        ),
        (
            "whole_artifact_budget_from_assignment_payload",
            lambda: footprint.whole_artifact_budget_from_assignment_payload(
                {"__prismaquant__": {"whole_artifact_budget": _stamp()}},
                where="pin",
                assignment=ASSIGNMENT,
            ),
            STAMP,
        ),
        (
            "canonical_imatrix_sha256",
            _imatrix_sha,
            "97b7c5655039ec23c746b151392838f889e93b3a98481fc9d6a39b08039564ce",
        ),
        (
            "imatrix_from_probe_stats",
            _probe_provenance,
            {
                "dense_entries": 1,
                "packed_entries": 1,
                "schema": (
                    "prismaquant.cb_imatrix.probe_act_sq_sum_over_tokens.v1"
                ),
                "skipped_missing_entries": 0,
                "value_sha256": (
                    "14582372dfa35bd941d2bcb9df91ea59"
                    "bbdd58c5b237652c12db79d3adef3c30"
                ),
            },
        ),
    ],
)
def test_moved_helper_output_is_unchanged(name, compute, expected):
    assert compute() == expected, name


def test_enforce_whole_artifact_budget_measures_every_regular_file(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 3000)
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 4000)
    payload = {"whole_artifact_budget": _stamp()}
    assert footprint.recursive_regular_file_bytes(tmp_path) == 7000
    assert footprint.enforce_whole_artifact_budget(
        tmp_path, payload, where="pin", assignment=ASSIGNMENT
    ) == {
        "scope": "all_regular_files_recursive",
        "artifact_path": str(tmp_path),
        "actual_bytes": 7000,
        "budget_bytes": 10000,
        "headroom_bytes": 3000,
        "within_budget": True,
    }
    footprint.assert_exclusions_match_budget_stamp(
        _stamp(), ["visual.", "mtp."], where="pin"
    )
    with pytest.raises(ValueError, match="namespace exclusions disagree"):
        footprint.assert_exclusions_match_budget_stamp(
            _stamp(), ["mtp."], where="pin"
        )


def test_the_helpers_have_one_owner():
    # The retired codebook lane's modules were archived in step 2 of #1304.
    for name in ("cb_imatrix", "nvfp4_cb_footprint"):
        assert importlib.util.find_spec(f"prismaquant.{name}") is None, name
