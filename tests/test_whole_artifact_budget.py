"""The whole-artifact byte budget: stamp, reader, and final recursive stat.

Carried out of ``test_cb_serialization_contract.py`` when the retired codebook
lane's tests were archived on 2026-09-25 (#1304). The budget is
lane-independent: the allocator writes the stamp, and the compressed-tensors
and GGUF exporters enforce it (``prismaquant/footprint.py``).
"""
from __future__ import annotations

import pytest

from prismaquant.footprint import (
    enforce_whole_artifact_budget,
    whole_artifact_budget_stamp,
)


def test_generic_export_budget_gate_measures_files_recursively(tmp_path):
    artifact = tmp_path / "artifact"
    (artifact / "nested").mkdir(parents=True)
    (artifact / "a.bin").write_bytes(b"a" * 7)
    (artifact / "nested" / "b.bin").write_bytes(b"b" * 5)
    assignment = {"layer.q_proj": "BF16"}
    payload = {
        "whole_artifact_budget": whole_artifact_budget_stamp(
            budget_bytes=12,
            selection_tensor_payload_bytes=8,
            selection_non_tensor_reserve_bytes=4,
            selection_assignment=assignment,
        ),
    }
    attestation = enforce_whole_artifact_budget(
        artifact, payload, where="unit export", assignment=assignment
    )
    assert attestation["actual_bytes"] == 12
    assert attestation["within_budget"]

    payload["whole_artifact_budget"] = whole_artifact_budget_stamp(
        budget_bytes=11,
        selection_tensor_payload_bytes=8,
        selection_non_tensor_reserve_bytes=3,
        selection_assignment=assignment,
    )
    with pytest.raises(RuntimeError, match="exact completed artifact size"):
        enforce_whole_artifact_budget(
            artifact, payload, where="unit export", assignment=assignment
        )


def test_whole_artifact_budget_rejects_assignment_drift(tmp_path):
    assignment = {"layer.q_proj": "BF16"}
    payload = {
        "whole_artifact_budget": whole_artifact_budget_stamp(
            budget_bytes=1,
            selection_tensor_payload_bytes=1,
            selection_non_tensor_reserve_bytes=0,
            selection_assignment=assignment,
        ),
    }
    (tmp_path / "artifact.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="assignment being consumed"):
        enforce_whole_artifact_budget(
            tmp_path,
            payload,
            where="unit export",
            assignment={"layer.q_proj": "NVFP4"},
        )



def _stamp(assignment, *, excluded=()):
    from prismaquant.footprint import whole_artifact_budget_stamp

    return whole_artifact_budget_stamp(
        budget_bytes=1000,
        selection_tensor_payload_bytes=8,
        selection_non_tensor_reserve_bytes=4,
        selection_assignment=assignment,
        excluded_source_prefixes=excluded,
    )


def test_budget_stamp_without_exclusions_is_byte_identical():
    """A run that excludes nothing must write exactly the stamp it always did."""
    assignment = {"layer.q_proj": "BF16"}
    assert _stamp(assignment) == _stamp(assignment, excluded=())
    assert "excluded_source_prefixes" not in _stamp(assignment)
    # Blank/whitespace entries are not exclusions.
    assert "excluded_source_prefixes" not in _stamp(assignment, excluded=["", "  "])


def test_budget_stamp_records_and_dedupes_exclusions():
    from prismaquant.footprint import budget_stamp_excluded_prefixes

    stamp = _stamp({"layer.q_proj": "BF16"}, excluded=["mtp.", " mtp.", "visual."])
    assert stamp["excluded_source_prefixes"] == ["mtp.", "visual."]
    assert budget_stamp_excluded_prefixes(stamp) == ("mtp.", "visual.")


def test_exclusions_must_match_the_price_that_bought_them():
    from prismaquant.footprint import assert_exclusions_match_budget_stamp

    assignment = {"layer.q_proj": "BF16"}
    priced = _stamp(assignment, excluded=["mtp."])

    # Agreement in both spellings of "nothing" and in the real case.
    assert_exclusions_match_budget_stamp(priced, ["mtp."], where="unit")
    assert_exclusions_match_budget_stamp(
        _stamp(assignment), [], where="unit")
    # No stamp is no claim: exclusion stands on its own without a budget.
    assert_exclusions_match_budget_stamp(None, ["mtp."], where="unit")

    # OVERSHOOT: priced without mtp, but the export writes it anyway.
    with pytest.raises(ValueError, match="overshoots"):
        assert_exclusions_match_budget_stamp(priced, [], where="unit")

    # UNDERSHOOT -- the direction nothing else catches: the price charged for
    # mtp, the export drops it, and the artifact silently ships under budget.
    with pytest.raises(ValueError, match="under budget"):
        assert_exclusions_match_budget_stamp(
            _stamp(assignment), ["mtp."], where="unit")


def test_a_malformed_exclusion_record_is_loud():
    from prismaquant.footprint import (
        whole_artifact_budget_from_assignment_payload,
    )

    assignment = {"layer.q_proj": "BF16"}
    stamp = dict(_stamp(assignment, excluded=["mtp."]))
    stamp["excluded_source_prefixes"] = "mtp."      # a string, not a list
    with pytest.raises(ValueError, match="excluded_source_prefixes"):
        whole_artifact_budget_from_assignment_payload(
            {"whole_artifact_budget": stamp},
            where="unit", assignment=assignment)

