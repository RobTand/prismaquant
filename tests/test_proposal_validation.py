from __future__ import annotations

from dataclasses import replace

import pytest

from prismaquant.proposal_validation import (
    PairedBootstrapConfig,
    ProposalValidationBinding,
    SequenceLoss,
    validate_sampled_proposal,
)


def _sha(character: str) -> str:
    return character * 64


def _binding(**overrides) -> ProposalValidationBinding:
    values = {
        "source_sha256": _sha("a"),
        "teacher_sha256": _sha("b"),
        "tokenizer_sha256": _sha("c"),
        "heldout_population_sha256": _sha("d"),
        "selected_assignment_sha256": _sha("e"),
        "pilot_training_sha256": _sha("f"),
        "pilot_evaluation_sha256": _sha("0"),
        "heldout_training_overlap_audit_status": "verified_disjoint",
        "heldout_training_overlap_audit_sha256": _sha("1"),
        "sequence_independence_declared": True,
        "resampling_scope": "one independent held-out sequence per cluster",
        "resampling_limitations": (),
    }
    values.update(overrides)
    return ProposalValidationBinding(**values)


def _config(**overrides) -> PairedBootstrapConfig:
    values = {
        "seed": 17,
        "bootstrap_replicates": 401,
        "confidence": 0.95,
        "noninferiority_margin": 0.01,
    }
    values.update(overrides)
    return PairedBootstrapConfig(**values)


def _rows():
    candidate = (
        SequenceLoss("s-1", 10, "complete", mean_loss=1.02),
        SequenceLoss("s-2", 30, "complete", mean_loss=1.03),
        SequenceLoss("s-3", 60, "complete", mean_loss=1.00),
    )
    incumbent = (
        SequenceLoss("s-1", 10, "complete", mean_loss=1.00),
        SequenceLoss("s-2", 30, "complete", mean_loss=1.00),
        SequenceLoss("s-3", 60, "complete", mean_loss=1.00),
    )
    return candidate, incumbent


def test_token_weighted_paired_difference_and_pass_report():
    candidate, incumbent = _rows()

    report = validate_sampled_proposal(
        candidate, incumbent, binding=_binding(), bootstrap=_config()
    )

    # Equal-sequence weighting would be 0.0167.  The scored-token contract is
    # instead (10*.02 + 30*.03 + 60*0) / 100 = 0.011.
    assert report.mean_paired_loss_difference == pytest.approx(0.011)
    assert report.bootstrap_interval is not None
    assert report.bootstrap_interval.approximate is True
    assert report.bootstrap_interval.method == "paired_manifest_cluster_bootstrap_token_weighted"
    assert report.coverage["complete_pair_count"] == 3
    assert report.coverage["resampling_cluster_count"] == 3
    assert report.verdict in {"pass", "inconclusive", "regression"}
    assert report.validity == "independent_heldout"
    assert report.promotion == report.verdict
    assert report.to_dict()["binding"]["selected_assignment_sha256"] == _sha("e")


def test_bootstrap_is_deterministic_for_the_explicit_seed():
    candidate, incumbent = _rows()
    first = validate_sampled_proposal(
        candidate, incumbent, binding=_binding(), bootstrap=_config()
    )
    second = validate_sampled_proposal(
        candidate, incumbent, binding=_binding(), bootstrap=_config()
    )

    assert first.input_sha256 == second.input_sha256
    assert first.bootstrap_interval == second.bootstrap_interval


def test_pairing_and_scored_token_mismatches_refuse():
    candidate, incumbent = _rows()
    with pytest.raises(ValueError, match="identical sequence IDs"):
        validate_sampled_proposal(
            candidate[:-1], incumbent, binding=_binding(), bootstrap=_config()
        )
    mismatched = list(incumbent)
    mismatched[1] = replace(mismatched[1], scored_token_count=31)
    with pytest.raises(ValueError, match="scored_token_count differs"):
        validate_sampled_proposal(
            candidate, mismatched, binding=_binding(), bootstrap=_config()
        )


def test_failure_and_timeout_rows_are_retained_and_cannot_pass():
    candidate, incumbent = _rows()
    failed_candidate = list(candidate)
    failed_candidate[1] = SequenceLoss(
        "s-2", 30, "timed_out", failure_detail="worker deadline"
    )

    report = validate_sampled_proposal(
        failed_candidate, incumbent, binding=_binding(), bootstrap=_config()
    )

    assert report.verdict == "inconclusive"
    assert report.promotion == "inconclusive"
    assert report.mean_paired_loss_difference is None
    assert report.coverage["candidate_timeout_count"] == 1
    assert report.failures["candidate"][0].sequence_id == "s-2"
    assert "no complete-row subset" in report.decision_reason


def test_exact_training_overlap_and_population_reuse_refuse():
    candidate, incumbent = _rows()
    with pytest.raises(ValueError, match="overlap exact training"):
        validate_sampled_proposal(
            candidate,
            incumbent,
            binding=_binding(training_sequence_ids=("train-0", "s-2")),
            bootstrap=_config(),
        )
    with pytest.raises(ValueError, match="equals the exact training population"):
        _binding(training_population_sha256=_sha("d"))


def test_unverified_overlap_audit_cannot_pass_on_different_population_hashes():
    candidate, incumbent = _rows()

    report = validate_sampled_proposal(
        candidate,
        incumbent,
        binding=_binding(heldout_training_overlap_audit_status="unverified"),
        bootstrap=_config(noninferiority_margin=1.0),
    )

    assert report.verdict == "inconclusive"
    assert report.promotion == "inconclusive"
    assert report.validity == "descriptive_only_overlap_unverified"
    assert report.mean_paired_loss_difference == pytest.approx(0.011)
    assert report.bootstrap_interval is not None
    assert "different population hashes" in report.decision_reason


def test_margin_drives_the_only_verdict_tradeoff():
    candidate, incumbent = _rows()
    better = validate_sampled_proposal(
        candidate,
        incumbent,
        binding=_binding(),
        bootstrap=_config(noninferiority_margin=1.0),
    )
    worse = validate_sampled_proposal(
        candidate,
        incumbent,
        binding=_binding(),
        bootstrap=_config(noninferiority_margin=-1.0),
    )

    assert better.verdict == "pass"
    assert worse.verdict == "regression"


def test_no_implicit_statistical_defaults_or_invalid_digests():
    with pytest.raises(TypeError):
        PairedBootstrapConfig()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _binding(source_sha256="not-a-digest")


def test_shared_document_windows_resample_as_whole_manifest_clusters():
    candidate, incumbent = _rows()
    candidate = (
        replace(candidate[0], resampling_cluster_id="document-a"),
        replace(candidate[1], resampling_cluster_id="document-a"),
        replace(candidate[2], resampling_cluster_id="document-b"),
    )
    incumbent = tuple(
        replace(row, resampling_cluster_id=cluster)
        for row, cluster in zip(incumbent, ("document-a", "document-a", "document-b"))
    )

    report = validate_sampled_proposal(
        candidate,
        incumbent,
        binding=_binding(
            sequence_independence_declared=False,
            resampling_scope="packed held-out windows clustered by original document",
            resampling_limitations=("document membership is the finest audited packing unit",),
        ),
        bootstrap=_config(),
    )

    assert report.coverage["resampling_cluster_count"] == 2
    assert report.coverage["resampling_scope"] == "packed held-out windows clustered by original document"
    assert report.bootstrap_interval is not None


def test_cluster_id_is_required_when_sequence_independence_is_not_declared():
    candidate, incumbent = _rows()
    with pytest.raises(ValueError, match="resampling_cluster_id is required"):
        validate_sampled_proposal(
            candidate,
            incumbent,
            binding=_binding(sequence_independence_declared=False),
            bootstrap=_config(),
        )


def test_one_shared_document_cluster_is_inconclusive():
    candidate, incumbent = _rows()
    candidate = tuple(replace(row, resampling_cluster_id="document-a") for row in candidate)
    incumbent = tuple(replace(row, resampling_cluster_id="document-a") for row in incumbent)

    report = validate_sampled_proposal(
        candidate,
        incumbent,
        binding=_binding(sequence_independence_declared=False),
        bootstrap=_config(),
    )

    assert report.verdict == "inconclusive"
    assert report.bootstrap_interval is None
    assert report.coverage["resampling_cluster_count"] == 1
