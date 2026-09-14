"""Identity-bound paired validation for sampled whole-stack proposals.

The allocator may propose a sampled whole-stack assignment, but this module
does not promote it, export it, or infer a latency result.  It evaluates only
an independently selected assignment's held-out, per-sequence losses.  The
uncertainty interval is a deterministic *manifest-declared cluster bootstrap*; it is
an approximation and is deliberately not a Hessian/probe error estimate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, Sequence

from .cost_stage_checkpoint import canonical_json_sha256


SAMPLED_PROPOSAL_VALIDATION_SCHEMA = "prismaquant.sampled_proposal_validation.v1"
SequenceStatus = Literal["complete", "failed", "timed_out"]
ValidationVerdict = Literal["pass", "inconclusive", "regression"]
_STATUSES = frozenset({"complete", "failed", "timed_out"})
_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class SequenceLoss:
    """One selected-assignment held-out sequence outcome.

    ``mean_loss`` is a mean over exactly ``scored_token_count`` tokens.  A
    failed or timed-out row remains in the input and carries no loss; it is
    evidence that the comparison is inconclusive, never a row to filter out.
    """

    sequence_id: str
    scored_token_count: int
    status: SequenceStatus
    mean_loss: float | None = None
    failure_detail: str | None = None
    resampling_cluster_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("sequence_id must be a nonempty string")
        if isinstance(self.scored_token_count, bool) or self.scored_token_count <= 0:
            raise ValueError(
                f"sequence {self.sequence_id!r} scored_token_count must be positive"
            )
        if self.status not in _STATUSES:
            raise ValueError(
                f"sequence {self.sequence_id!r} has unknown status {self.status!r}"
            )
        if self.resampling_cluster_id is not None and (
            not isinstance(self.resampling_cluster_id, str)
            or not self.resampling_cluster_id
        ):
            raise ValueError(
                f"sequence {self.sequence_id!r} resampling_cluster_id must be a nonempty string"
            )
        if self.status == "complete":
            if self.mean_loss is None or not math.isfinite(float(self.mean_loss)):
                raise ValueError(
                    f"complete sequence {self.sequence_id!r} needs a finite mean_loss"
                )
            if self.failure_detail is not None:
                raise ValueError(
                    f"complete sequence {self.sequence_id!r} cannot have failure_detail"
                )
        else:
            if self.mean_loss is not None:
                raise ValueError(
                    f"incomplete sequence {self.sequence_id!r} cannot have mean_loss"
                )
            if not isinstance(self.failure_detail, str) or not self.failure_detail:
                raise ValueError(
                    f"incomplete sequence {self.sequence_id!r} needs failure_detail"
                )


@dataclass(frozen=True)
class ProposalValidationBinding:
    """All identities required to compare a selected proposal independently."""

    source_sha256: str
    teacher_sha256: str
    tokenizer_sha256: str
    heldout_population_sha256: str
    selected_assignment_sha256: str
    pilot_training_sha256: str
    pilot_evaluation_sha256: str
    heldout_training_overlap_audit_status: Literal["verified_disjoint", "unverified"]
    heldout_training_overlap_audit_sha256: str
    sequence_independence_declared: bool
    resampling_scope: str
    resampling_limitations: tuple[str, ...]
    training_population_sha256: str | None = None
    training_sequence_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for field in (
            "source_sha256",
            "teacher_sha256",
            "tokenizer_sha256",
            "heldout_population_sha256",
            "selected_assignment_sha256",
            "pilot_training_sha256",
            "pilot_evaluation_sha256",
            "heldout_training_overlap_audit_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if self.heldout_training_overlap_audit_status not in {
            "verified_disjoint",
            "unverified",
        }:
            raise ValueError(
                "heldout_training_overlap_audit_status must be "
                "'verified_disjoint' or 'unverified'"
            )
        if not isinstance(self.sequence_independence_declared, bool):
            raise ValueError("sequence_independence_declared must be a boolean")
        if not isinstance(self.resampling_scope, str) or not self.resampling_scope:
            raise ValueError("resampling_scope must be a nonempty manifest-declared string")
        if not isinstance(self.resampling_limitations, tuple) or any(
            not isinstance(limitation, str) or not limitation
            for limitation in self.resampling_limitations
        ):
            raise ValueError("resampling_limitations must be a tuple of nonempty strings")
        if self.training_population_sha256 is not None:
            _require_sha256(self.training_population_sha256, "training_population_sha256")
            if self.training_population_sha256 == self.heldout_population_sha256:
                raise ValueError(
                    "held-out population equals the exact training population; refusing comparison"
                )
        if self.training_sequence_ids is not None:
            if not self.training_sequence_ids:
                raise ValueError("training_sequence_ids must be omitted when exact IDs are unavailable")
            if len(set(self.training_sequence_ids)) != len(self.training_sequence_ids):
                raise ValueError("training_sequence_ids contains duplicates")
            if any(not isinstance(value, str) or not value for value in self.training_sequence_ids):
                raise ValueError("training_sequence_ids must contain nonempty strings")


@dataclass(frozen=True)
class PairedBootstrapConfig:
    """Explicit statistical choices; there are intentionally no defaults."""

    seed: int
    bootstrap_replicates: int
    confidence: float
    noninferiority_margin: float

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if (
            isinstance(self.bootstrap_replicates, bool)
            or not isinstance(self.bootstrap_replicates, int)
            or self.bootstrap_replicates < 2
        ):
            raise ValueError("bootstrap_replicates must be an integer of at least 2")
        if not 0.0 < float(self.confidence) < 1.0:
            raise ValueError("confidence must be strictly between zero and one")
        if not math.isfinite(float(self.noninferiority_margin)):
            raise ValueError("noninferiority_margin must be finite")


@dataclass(frozen=True)
class BootstrapInterval:
    lower: float
    upper: float
    confidence: float
    approximate: bool = True
    method: str = "paired_manifest_cluster_bootstrap_token_weighted"


@dataclass(frozen=True)
class SampledProposalValidationReport:
    schema: str
    binding: ProposalValidationBinding
    bootstrap: PairedBootstrapConfig
    input_sha256: str
    mean_paired_loss_difference: float | None
    bootstrap_interval: BootstrapInterval | None
    validity: Literal[
        "independent_heldout",
        "descriptive_only_overlap_unverified",
    ]
    verdict: ValidationVerdict
    promotion: ValidationVerdict
    decision_reason: str
    coverage: dict[str, object]
    failures: dict[str, tuple[SequenceLoss, ...]]

    def to_dict(self) -> dict[str, object]:
        """Return an artifact-ready JSON payload with all raw failure rows."""
        return asdict(self)


def validate_sampled_proposal(
    candidate_rows: Sequence[SequenceLoss],
    incumbent_rows: Sequence[SequenceLoss],
    *,
    binding: ProposalValidationBinding,
    bootstrap: PairedBootstrapConfig,
) -> SampledProposalValidationReport:
    """Compare paired selected-assignment losses with a sequence bootstrap.

    The loss difference is candidate minus incumbent, so a positive result is
    a candidate regression.  A pass requires the approximate interval's upper
    bound to be no greater than the supplied non-inferiority margin.  A
    regression requires its lower bound to be larger than that margin.
    """
    candidates = _indexed_rows(candidate_rows, arm="candidate")
    incumbents = _indexed_rows(incumbent_rows, arm="incumbent")
    _require_same_sequence_contract(candidates, incumbents, binding)
    _require_training_disjointness(candidates, binding)

    ordered_ids = tuple(sorted(candidates))
    candidate_failures = tuple(
        candidates[sequence_id]
        for sequence_id in ordered_ids
        if candidates[sequence_id].status != "complete"
    )
    incumbent_failures = tuple(
        incumbents[sequence_id]
        for sequence_id in ordered_ids
        if incumbents[sequence_id].status != "complete"
    )
    coverage = {
        "sequence_count": len(ordered_ids),
        "scored_token_count": sum(
            candidates[sequence_id].scored_token_count for sequence_id in ordered_ids
        ),
        "complete_pair_count": sum(
            candidates[sequence_id].status == "complete"
            and incumbents[sequence_id].status == "complete"
            for sequence_id in ordered_ids
        ),
        "candidate_failure_count": len(candidate_failures),
        "candidate_timeout_count": sum(row.status == "timed_out" for row in candidate_failures),
        "incumbent_failure_count": len(incumbent_failures),
        "incumbent_timeout_count": sum(row.status == "timed_out" for row in incumbent_failures),
        "sequence_ids": ordered_ids,
        "resampling_cluster_count": len(
            {_resampling_cluster_id(candidates[sequence_id], binding) for sequence_id in ordered_ids}
        ),
        "resampling_scope": binding.resampling_scope,
        "resampling_limitations": binding.resampling_limitations,
    }
    input_sha256 = canonical_json_sha256(
        {
            "schema": SAMPLED_PROPOSAL_VALIDATION_SCHEMA,
            "binding": asdict(binding),
            "bootstrap": asdict(bootstrap),
            "candidate_rows": [asdict(candidates[key]) for key in ordered_ids],
            "incumbent_rows": [asdict(incumbents[key]) for key in ordered_ids],
        },
        where="sampled proposal validation input",
    )
    failures = {
        "candidate": candidate_failures,
        "incumbent": incumbent_failures,
    }
    if candidate_failures or incumbent_failures:
        return SampledProposalValidationReport(
            schema=SAMPLED_PROPOSAL_VALIDATION_SCHEMA,
            binding=binding,
            bootstrap=bootstrap,
            input_sha256=input_sha256,
            mean_paired_loss_difference=None,
            bootstrap_interval=None,
            validity=_validity(binding),
            verdict="inconclusive",
            promotion="inconclusive",
            decision_reason="incomplete held-out rows retained; no complete-row subset was tested",
            coverage=coverage,
            failures=failures,
        )
    if len(ordered_ids) < 2:
        return SampledProposalValidationReport(
            schema=SAMPLED_PROPOSAL_VALIDATION_SCHEMA,
            binding=binding,
            bootstrap=bootstrap,
            input_sha256=input_sha256,
            mean_paired_loss_difference=None,
            bootstrap_interval=None,
            validity=_validity(binding),
            verdict="inconclusive",
            promotion="inconclusive",
            decision_reason="at least two complete held-out sequences are required for sequence-level uncertainty",
            coverage=coverage,
            failures=failures,
        )

    pairs = tuple((candidates[key], incumbents[key]) for key in ordered_ids)
    clusters = _cluster_pairs(pairs, binding)
    if len(clusters) < 2:
        return SampledProposalValidationReport(
            schema=SAMPLED_PROPOSAL_VALIDATION_SCHEMA,
            binding=binding,
            bootstrap=bootstrap,
            input_sha256=input_sha256,
            mean_paired_loss_difference=None,
            bootstrap_interval=None,
            validity=_validity(binding),
            verdict="inconclusive",
            promotion="inconclusive",
            decision_reason=(
                "at least two complete held-out resampling clusters are required for "
                "cluster-level uncertainty"
            ),
            coverage=coverage,
            failures=failures,
        )
    mean_difference = _token_weighted_difference(pairs)
    draws = _bootstrap_differences(clusters, bootstrap)
    alpha = (1.0 - float(bootstrap.confidence)) / 2.0
    interval = BootstrapInterval(
        lower=_quantile(draws, alpha),
        upper=_quantile(draws, 1.0 - alpha),
        confidence=float(bootstrap.confidence),
    )
    if interval.upper <= float(bootstrap.noninferiority_margin):
        verdict: ValidationVerdict = "pass"
        reason = "approximate upper interval bound is within the supplied non-inferiority margin"
    elif interval.lower > float(bootstrap.noninferiority_margin):
        verdict = "regression"
        reason = "approximate lower interval bound exceeds the supplied non-inferiority margin"
    else:
        verdict = "inconclusive"
        reason = "approximate interval crosses the supplied non-inferiority margin"
    if binding.heldout_training_overlap_audit_status != "verified_disjoint":
        validity = "descriptive_only_overlap_unverified"
        promotion: ValidationVerdict = "inconclusive"
        reason = (
            "held-out/training overlap audit is unverified; paired loss and "
            "bootstrap interval are descriptive only because different population "
            "hashes do not establish disjointness"
        )
        verdict = "inconclusive"
    else:
        validity = "independent_heldout"
        promotion = verdict
    return SampledProposalValidationReport(
        schema=SAMPLED_PROPOSAL_VALIDATION_SCHEMA,
        binding=binding,
        bootstrap=bootstrap,
        input_sha256=input_sha256,
        mean_paired_loss_difference=mean_difference,
        bootstrap_interval=interval,
        validity=validity,
        verdict=verdict,
        promotion=promotion,
        decision_reason=reason,
        coverage=coverage,
        failures=failures,
    )


def _require_sha256(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


def _indexed_rows(rows: Sequence[SequenceLoss], *, arm: str) -> dict[str, SequenceLoss]:
    indexed: dict[str, SequenceLoss] = {}
    for row in rows:
        if not isinstance(row, SequenceLoss):
            raise TypeError(f"{arm} rows must be SequenceLoss instances")
        if row.sequence_id in indexed:
            raise ValueError(f"{arm} rows duplicate sequence_id {row.sequence_id!r}")
        indexed[row.sequence_id] = row
    if not indexed:
        raise ValueError(f"{arm} rows must not be empty")
    return indexed


def _require_same_sequence_contract(
    candidates: dict[str, SequenceLoss],
    incumbents: dict[str, SequenceLoss],
    binding: ProposalValidationBinding,
) -> None:
    candidate_ids, incumbent_ids = set(candidates), set(incumbents)
    if candidate_ids != incumbent_ids:
        raise ValueError(
            "candidate and incumbent must have identical sequence IDs; "
            f"candidate_only={sorted(candidate_ids - incumbent_ids)!r}, "
            f"incumbent_only={sorted(incumbent_ids - candidate_ids)!r}"
        )
    mismatched_counts = [
        sequence_id
        for sequence_id in sorted(candidate_ids)
        if candidates[sequence_id].scored_token_count
        != incumbents[sequence_id].scored_token_count
    ]
    if mismatched_counts:
        raise ValueError(
            "candidate and incumbent scored_token_count differs for sequence IDs "
            f"{mismatched_counts!r}"
        )
    mismatched_clusters = [
        sequence_id
        for sequence_id in sorted(candidate_ids)
        if _resampling_cluster_id(candidates[sequence_id], binding)
        != _resampling_cluster_id(incumbents[sequence_id], binding)
    ]
    if mismatched_clusters:
        raise ValueError(
            "candidate and incumbent resampling cluster differs for sequence IDs "
            f"{mismatched_clusters!r}"
        )


def _require_training_disjointness(
    candidates: dict[str, SequenceLoss], binding: ProposalValidationBinding
) -> None:
    if binding.training_sequence_ids is None:
        return
    overlap = sorted(set(candidates).intersection(binding.training_sequence_ids))
    if overlap:
        raise ValueError(
            "held-out sequence IDs overlap exact training sequence IDs; "
            f"refusing comparison: {overlap!r}"
        )


def _token_weighted_difference(pairs: Sequence[tuple[SequenceLoss, SequenceLoss]]) -> float:
    numerator = sum(
        (float(candidate.mean_loss) - float(incumbent.mean_loss))
        * candidate.scored_token_count
        for candidate, incumbent in pairs
    )
    denominator = sum(candidate.scored_token_count for candidate, _ in pairs)
    return numerator / denominator


def _resampling_cluster_id(
    row: SequenceLoss, binding: ProposalValidationBinding
) -> str:
    if row.resampling_cluster_id is not None:
        return row.resampling_cluster_id
    if binding.sequence_independence_declared:
        return row.sequence_id
    raise ValueError(
        "resampling_cluster_id is required unless the manifest explicitly declares "
        f"sequence independence (sequence {row.sequence_id!r})"
    )


def _cluster_pairs(
    pairs: Sequence[tuple[SequenceLoss, SequenceLoss]],
    binding: ProposalValidationBinding,
) -> tuple[tuple[tuple[SequenceLoss, SequenceLoss], ...], ...]:
    grouped: dict[str, list[tuple[SequenceLoss, SequenceLoss]]] = {}
    for candidate, incumbent in pairs:
        grouped.setdefault(_resampling_cluster_id(candidate, binding), []).append(
            (candidate, incumbent)
        )
    return tuple(tuple(grouped[cluster_id]) for cluster_id in sorted(grouped))


def _bootstrap_differences(
    clusters: Sequence[Sequence[tuple[SequenceLoss, SequenceLoss]]],
    config: PairedBootstrapConfig,
) -> tuple[float, ...]:
    """Use a fixed local LCG so bootstrap sampling is replayable everywhere."""
    state = int(config.seed) & _MASK64
    count = len(clusters)
    draws: list[float] = []
    for _ in range(config.bootstrap_replicates):
        sampled: list[tuple[SequenceLoss, SequenceLoss]] = []
        for _ in range(count):
            state = (state * 6364136223846793005 + 1442695040888963407) & _MASK64
            sampled.extend(clusters[state % count])
        draws.append(_token_weighted_difference(sampled))
    return tuple(draws)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of no bootstrap draws")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _validity(
    binding: ProposalValidationBinding,
) -> Literal["independent_heldout", "descriptive_only_overlap_unverified"]:
    return (
        "independent_heldout"
        if binding.heldout_training_overlap_audit_status == "verified_disjoint"
        else "descriptive_only_overlap_unverified"
    )


__all__ = [
    "BootstrapInterval",
    "PairedBootstrapConfig",
    "ProposalValidationBinding",
    "SAMPLED_PROPOSAL_VALIDATION_SCHEMA",
    "SampledProposalValidationReport",
    "SequenceLoss",
    "validate_sampled_proposal",
]
