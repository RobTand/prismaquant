#!/usr/bin/env python3
"""Value-bearing contracts for a streamed all-position KL teacher.

The DSv4 BF16 checkpoint is larger than one Spark's unified memory, so the
release gold lane builds its teacher distribution with PrismaQuant's existing
layer streamer.  This module keeps that payload independently replayable:
tensor bytes, source checkpoint, tokenized calibration windows, and the
serialized file are all digest-bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import torch


TEACHER_PAYLOAD_SCHEMA = "prismaquant.full_kl_teacher_payload/1"
TEACHER_META_SCHEMA = "prismaquant.full_kl_teacher_meta/1"
TEACHER_EVIDENCE_SCHEMA = "prismaquant.full_kl_teacher_evidence/1"
TEACHER_PAYLOAD_V2_SCHEMA = "prismaquant.full_kl_teacher_payload/2"
TEACHER_META_V2_SCHEMA = "prismaquant.full_kl_teacher_meta/2"
TEACHER_EVIDENCE_V2_SCHEMA = "prismaquant.full_kl_teacher_evidence/2"
CALIBRATION_SCHEMA = "prismaquant.wikitext_gold_calibration/1"
TOKENIZER_IDENTITY_SCHEMA = "prismaquant.tokenizer_identity/1"

WIKITEXT_DATASET = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_SPLIT = "train"
# Immutable commit currently backing the repository's historical gold lane.
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
N_SAMPLES = 8
SEQLEN = 512
WINDOW_SEED = 42
# Measured, not chosen. At K=1024 the DSv4-Flash BF16 teacher misses the 0.90
# per-position floor on 34 of these 4,088 positions (worst 0.6943), so the gate
# refused every teacher this lane could build. A single sweep over the pinned
# calibration -- topk returns sorted, so one pass at 16384 yields every smaller
# K -- gives the coverage minimum as a function of K:
#
#     K       min     mean    #<0.90
#     1024  0.6943  0.9905      34
#     2048  0.7829  0.9949      18
#     4096  0.8606  0.9975       8
#     8192  0.9231  0.9990       0
#     16384 0.9658  0.9997       0
#
# 8192 is the smallest K clearing the floor. The floor itself is unchanged at
# 0.90: this widens the teacher's support until it can meet the guarantee,
# rather than lowering the guarantee to fit a support that could not. These
# windows are pinned (seed 42, fixed starts, fixed digest), so the margin above
# is exact for the shipping workload and not a sample that could drift.
PROMPT_TOP_K = 8192
EXPECTED_POSITIONS = N_SAMPLES * (SEQLEN - 1)

# FP32 log-softmax values can round the reconstructed probability sum a few
# ulps above one.  One part per million is a deliberately tight allowance for
# that representation effect; it is not permission for an over-normalized
# teacher distribution.
TOPK_PROBABILITY_MASS_ABS_TOLERANCE = 1e-6
# The tail bucket makes the statistic defined below this point, but allowing a
# mostly-tail distribution would make the top-K comparison uninformative.
# Requiring 90% support caps the declared aggregated teacher tail at 10% per
# scored position while remaining conservative for a language-model top-K.
TOPK_MINIMUM_COVERAGE = 0.90
TOPK_COVERAGE_POLICY_SCHEMA = "prismaquant.topk_tail_coverage_policy/1"

FORWARD_FIDELITY_POLICY_SCHEMA = "prismaquant.teacher_forward_fidelity_policy/1"
# Continued-fraction iteration cap for the regularized incomplete beta below.
# This is a fail-closed numerical guard, not a decision threshold: exceeding it
# raises rather than returning an unconverged tail probability.
_BETA_CF_MAX_ITERATIONS = 1024
# Largest NLL whose perplexity is representable.  Reporting-only: a mean NLL
# past this point is shown as an infinite perplexity instead of raising an
# overflow.  Derived from the float64 range, not chosen.
_MAX_REPORTABLE_NLL = math.log(float(torch.finfo(torch.float64).max))

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TENSOR_KEYS = ("calib_ids", "topk_ids", "topk_lps")
_V2_FIELDS = {"final_logprobs", "source_execution", "producer_identity", "fit_overlap_status",
              "wikitext_inputs_sha256", "model_identity"}
_TOKENIZER_FILENAMES = (
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


class TeacherPayloadError(ValueError):
    """The teacher payload or one of its value-bearing contracts is invalid."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode strict canonical JSON used by every digest in this contract."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TeacherPayloadError("value is not strict canonical JSON") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TeacherPayloadError(f"{where} is not a lowercase SHA256")
    return value


def topk_coverage_policy() -> dict[str, object]:
    """Return the closed acceptance policy carried by metadata/evidence."""
    return {
        "schema": TOPK_COVERAGE_POLICY_SCHEMA,
        "top_k": PROMPT_TOP_K,
        "minimum_probability_mass_per_position": TOPK_MINIMUM_COVERAGE,
        "maximum_probability_mass": 1.0,
        "probability_mass_absolute_tolerance": (
            TOPK_PROBABILITY_MASS_ABS_TOLERANCE
        ),
        "maximum_declared_tail_mass_per_position": (
            1.0 - TOPK_MINIMUM_COVERAGE
        ),
        "tail_bucket": True,
    }


def topk_coverage_summary(
    topk_ids: torch.Tensor,
    topk_lps: torch.Tensor,
    *,
    vocab_size: int,
) -> dict[str, object]:
    """Validate every top-K row and derive coverage from tensor values.

    The returned values are computed in float64 from the serialized float32
    log probabilities.  No caller-provided summary participates in this
    calculation.
    """
    if topk_ids.dtype != torch.int32 or topk_lps.dtype != torch.float32:
        raise TeacherPayloadError("teacher top-k tensor dtypes are invalid")
    if topk_ids.shape != topk_lps.shape or topk_ids.ndim != 3:
        raise TeacherPayloadError("teacher top-k tensor shapes differ")
    if not torch.isfinite(topk_lps).all():
        raise TeacherPayloadError("teacher topk_lps contains non-finite values")
    if int(topk_ids.min()) < 0 or int(topk_ids.max()) >= vocab_size:
        raise TeacherPayloadError(
            "teacher top-k token ids are non-finite or out of range"
        )

    # Sorting ids, rather than relying on their probability order, makes the
    # uniqueness check exact and vectorized over all 4,088 release rows.
    ids_by_value = torch.sort(topk_ids, dim=-1).values
    if bool((ids_by_value[..., 1:] == ids_by_value[..., :-1]).any()):
        raise TeacherPayloadError("teacher top-k rows contain duplicate token ids")
    if bool((topk_lps[..., 1:] > topk_lps[..., :-1]).any()):
        raise TeacherPayloadError(
            "teacher top-k log probabilities are not nonincreasing"
        )
    if bool((topk_lps > 0.0).any()):
        raise TeacherPayloadError("teacher log probabilities exceed zero")

    coverage = topk_lps.to(dtype=torch.float64).exp().sum(dim=-1)
    if not torch.isfinite(coverage).all():
        raise TeacherPayloadError("teacher top-k probability mass is non-finite")
    coverage_mean = float(coverage.mean().item())
    coverage_min = float(coverage.min().item())
    coverage_max = float(coverage.max().item())
    if coverage_max > 1.0 + TOPK_PROBABILITY_MASS_ABS_TOLERANCE:
        raise TeacherPayloadError(
            "teacher top-k probability mass exceeds one beyond the "
            f"{TOPK_PROBABILITY_MASS_ABS_TOLERANCE:g} absolute tolerance"
        )
    if coverage_min < TOPK_MINIMUM_COVERAGE:
        # Report the shortfall, not just its existence. This refusal costs a
        # full streamed teacher pass over the source model, and the payload is
        # not written when it fires, so a bare "below 0.90" leaves the operator
        # with no way to tell a near-miss (raise K) from a genuinely flat
        # predictive distribution (the top-K formulation does not fit this
        # model) without paying for the pass again.
        below = coverage < TOPK_MINIMUM_COVERAGE
        n_below = int(below.sum().item())
        n_total = int(coverage.numel())
        quantiles = torch.tensor([0.001, 0.01, 0.05, 0.50], dtype=torch.float64)
        q = torch.quantile(coverage.flatten(), quantiles).tolist()
        raise TeacherPayloadError(
            "teacher top-k coverage falls below the declared "
            f"{TOPK_MINIMUM_COVERAGE:.2f} per-position minimum: "
            f"K={int(topk_ids.shape[-1])} over vocab={vocab_size}; "
            f"min={coverage_min:.4f} mean={coverage_mean:.4f}; "
            f"{n_below}/{n_total} positions short "
            f"({100.0 * n_below / max(n_total, 1):.2f}%); "
            f"coverage quantiles p0.1={q[0]:.4f} p1={q[1]:.4f} "
            f"p5={q[2]:.4f} p50={q[3]:.4f}"
        )
    return {
        "topk_coverage_mean": coverage_mean,
        "topk_coverage_min": coverage_min,
        "topk_coverage_policy": topk_coverage_policy(),
    }


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz evaluation of the continued fraction for the incomplete beta."""
    eps = float(torch.finfo(torch.float64).eps)
    tiny = float(torch.finfo(torch.float64).tiny)
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, _BETA_CF_MAX_ITERATIONS + 1):
        m2 = 2 * m
        for numerator in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + numerator * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + numerator / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) <= eps:
            return h
    raise TeacherPayloadError(
        "incomplete beta continued fraction did not converge to float64 "
        f"precision within {_BETA_CF_MAX_ITERATIONS} iterations"
    )


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Return I_x(a, b) in float64 without depending on SciPy."""
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0.0 or b <= 0.0:
        raise TeacherPayloadError("incomplete beta parameters are invalid")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_upper_tail(statistic: float, degrees_of_freedom: float) -> float:
    """Return P(T > statistic) for Student's t with the given d.o.f.

    Implemented here rather than imported so the gate is deterministic and has
    no optional dependency: a fidelity gate that silently weakens when SciPy is
    absent is not a gate.  ``tests`` cross-check it against ``scipy.stats.t``.
    """
    if degrees_of_freedom <= 0.0 or not math.isfinite(degrees_of_freedom):
        raise TeacherPayloadError("Student-t degrees of freedom are invalid")
    if math.isnan(statistic):
        raise TeacherPayloadError("Student-t statistic is not a number")
    if math.isinf(statistic):
        return 0.0 if statistic > 0.0 else 1.0
    half = 0.5 * _regularized_incomplete_beta(
        0.5 * degrees_of_freedom,
        0.5,
        degrees_of_freedom / (degrees_of_freedom + statistic * statistic),
    )
    return half if statistic >= 0.0 else 1.0 - half


def _octave_position_blocks(scored_positions: int) -> list[tuple[int, int]]:
    """Partition scored indices into octaves of available context length.

    Scored index ``t`` predicts the token after a prefix of ``t + 1`` tokens,
    and an autoregressive language model's expected NLL falls roughly like a
    power law in that prefix length.  The scale-free partition of a power law
    is by octave, so the blocks are context lengths ``[2**j, 2**(j+1))``.  The
    block count is ``floor(log2(scored_positions)) + 1`` -- fully determined by
    ``SEQLEN``, with no partition choice left to the caller.
    """
    if scored_positions < 1:
        raise TeacherPayloadError("teacher payload scores no positions")
    blocks: list[tuple[int, int]] = []
    lower = 1
    while lower <= scored_positions:
        upper = min(2 * lower, scored_positions + 1)
        # Stored as scored-index bounds; context length is index + 1.
        blocks.append((lower - 1, upper - 1))
        lower = upper
    return blocks


def teacher_forward_fidelity_policy(
    *,
    scored_positions: int,
    comparisons: int,
) -> dict[str, object]:
    """Return the closed acceptance policy for the context-monotonicity gate."""
    family_alpha = 1.0 / float(scored_positions)
    return {
        "schema": FORWARD_FIDELITY_POLICY_SCHEMA,
        "statistic": "per-position teacher-forced NLL, nats",
        "partition": "octaves of available context length",
        "test": "Welch one-sided t, later block mean > earlier block mean",
        "scored_positions": int(scored_positions),
        "comparisons": int(comparisons),
        "family_wise_alpha": family_alpha,
        "per_comparison_alpha": family_alpha / float(max(comparisons, 1)),
        "absolute_ceiling": "ln(vocab_size)",
    }


def teacher_forward_nll_per_position(
    topk_ids: torch.Tensor,
    topk_lps: torch.Tensor,
    calib_ids: torch.Tensor,
    *,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the teacher's own teacher-forced NLL from the payload alone.

    Scored row ``t`` is the distribution over the token at ``t + 1``, so the
    target log probability is already serialized whenever the target survives
    the top-K truncation.  When it does not, the payload still bounds the
    target's probability from above by two exact facts about the same row:

    * it cannot exceed the tail mass ``1 - sum(exp(topk_lps))`` -- everything
      outside the support shares that mass; and
    * it cannot exceed ``exp(topk_lps[..., -1])`` -- a larger probability would
      have placed the token inside a sorted top-K.

    The NLL is therefore imputed as ``-log(min(tail, p_K))``, a *lower* bound.
    Imputing a lower bound can only understate degradation, so a refusal is
    never manufactured by the imputation.  The bound is floored at
    ``K * eps(float32)``, the resolution at which a tail mass reconstructed
    from ``K`` serialized float32 log probabilities is still meaningful; that
    floor comes from the storage dtype, not from a chosen probability.

    Returns ``(nll, missing)`` shaped like the scored grid.
    """
    if calib_ids.ndim != 2 or topk_ids.ndim != 3:
        raise TeacherPayloadError("teacher fidelity inputs have invalid rank")
    n_samples, sequence = (int(value) for value in calib_ids.shape)
    if list(topk_ids.shape[:2]) != [n_samples, sequence - 1]:
        raise TeacherPayloadError(
            "teacher top-k grid does not match the calibration windows"
        )
    if int(calib_ids.min()) < 0 or int(calib_ids.max()) >= vocab_size:
        raise TeacherPayloadError("teacher calibration ids are out of vocabulary")
    support = int(topk_ids.shape[-1])
    # The reconstructed tail is a sum of `support` float32 terms; below this
    # magnitude it is rounding, not probability.
    tail_resolution = support * float(torch.finfo(torch.float32).eps)

    nll = torch.empty((n_samples, sequence - 1), dtype=torch.float64)
    missing = torch.empty((n_samples, sequence - 1), dtype=torch.bool)
    for sample in range(n_samples):
        ids = topk_ids[sample]
        lps = topk_lps[sample].to(dtype=torch.float64)
        targets = calib_ids[sample, 1:].to(dtype=ids.dtype).unsqueeze(-1)
        hit = ids == targets
        found = hit.any(dim=-1)
        # Top-K ids are already validated unique, so at most one term survives.
        target_lp = torch.where(hit, lps, torch.zeros((), dtype=torch.float64))
        target_lp = target_lp.sum(dim=-1)
        tail = (1.0 - lps.exp().sum(dim=-1)).clamp_min(0.0)
        smallest_in_support = lps[..., -1].exp()
        bound = torch.minimum(tail, smallest_in_support).clamp_min(tail_resolution)
        nll[sample] = torch.where(found, -target_lp, -bound.log())
        missing[sample] = ~found
    if not torch.isfinite(nll).all():
        raise TeacherPayloadError("teacher per-position NLL is non-finite")
    return nll, missing


def _block_statistics(
    nll: torch.Tensor,
    missing: torch.Tensor,
    blocks: Sequence[tuple[int, int]],
) -> list[dict[str, object]]:
    profile: list[dict[str, object]] = []
    for first, last in blocks:
        values = nll[:, first:last].reshape(-1)
        count = int(values.numel())
        mean = float(values.mean().item())
        variance = (
            float(values.var(unbiased=True).item()) if count > 1 else float("nan")
        )
        profile.append({
            "context_first": first + 1,
            "context_last": last,
            "positions": count,
            "nll_mean": mean,
            "perplexity": math.exp(mean) if mean < _MAX_REPORTABLE_NLL else math.inf,
            "nll_stdev": math.sqrt(variance) if count > 1 else None,
            "out_of_support_targets": int(
                missing[:, first:last].sum().item()
            ),
            "variance": variance,
        })
    return profile


def format_forward_fidelity_profile(summary: Mapping[str, Any]) -> str:
    """Render the per-block context profile as a fixed-width table."""
    lines = [
        "[teacher-fidelity] per-position teacher-forced NLL by context octave",
        "  context      positions    NLL     PPL   out-of-support",
    ]
    for block in summary["blocks"]:
        stdev = block["nll_stdev"]
        lines.append(
            f"  {block['context_first']:>5d}-{block['context_last']:<5d} "
            f"{block['positions']:>9d} "
            f"{block['nll_mean']:>7.3f} "
            f"{block['perplexity']:>9.2f} "
            f"{block['out_of_support_targets']:>10d}"
            + (f"   (sd {stdev:.3f})" if stdev is not None else "")
        )
    lines.append(
        f"  overall NLL {summary['nll_mean']:.4f} "
        f"PPL {summary['perplexity']:.3f} over "
        f"{summary['scored_positions']} positions; "
        f"uniform-vocabulary ceiling ln(V)={summary['uniform_nll_ceiling']:.4f}"
    )
    worst = summary.get("worst_comparison")
    if worst is not None:
        lines.append(
            "  worst context-monotonicity comparison: octave "
            f"{worst['earlier_context_first']}-{worst['earlier_context_last']} "
            f"(NLL {worst['earlier_nll_mean']:.3f}) -> "
            f"{worst['later_context_first']}-{worst['later_context_last']} "
            f"(NLL {worst['later_nll_mean']:.3f}); "
            f"welch_t={worst['welch_t']:.3f} df={worst['degrees_of_freedom']:.1f} "
            f"p={worst['p_value']:.3e} vs alpha={summary['per_comparison_alpha']:.3e}"
        )
    return "\n".join(lines)


def teacher_forward_fidelity_summary(
    topk_ids: torch.Tensor,
    topk_lps: torch.Tensor,
    calib_ids: torch.Tensor,
    *,
    vocab_size: int,
) -> dict[str, object]:
    """Refuse a teacher whose own NLL degrades as its context grows.

    Top-K coverage says nothing about faithfulness -- a confidently *wrong*
    distribution is still sharply peaked, which is exactly how a streamed
    teacher whose own perplexity was 262 passed every existing gate on
    2026-08-16 and was used as a KL reference for a 9.05-PPL student.  The
    signature of that defect is structural rather than absolute: teacher NLL
    got monotonically *worse* with more context (PPL 7.6 at 32-64 tokens,
    1013.2 at 384-511).  A correct autoregressive model on contiguous natural
    text improves with context and never inverts that way, whatever its
    absolute quality, so context-monotonicity is the property to enforce.

    The statistic is the teacher's own teacher-forced NLL per scored position,
    recovered from the payload alone (see
    :func:`teacher_forward_nll_per_position`); no additional model forward is
    required.  Positions are partitioned into octaves of available context
    (see :func:`_octave_position_blocks`), and every ordered pair of octaves
    ``(earlier, later)`` is compared with a one-sided Welch t-test on the
    per-position NLL.

    "Materially worse" is therefore measured in units of the payload's own
    dispersion -- the acceptance region is ``t_crit`` pooled standard errors
    wide, and the standard errors come from the per-position NLL spread -- and
    never as a ratio or an absolute NLL.  The one remaining convention is the
    significance level, and it is fixed by the payload's shape rather than
    chosen: the family-wise level is ``1 / scored_positions``, the finest
    false-alarm rate a payload of this size can meaningfully assert, split
    across the ``C(B, 2)`` ordered comparisons by Bonferroni.  The verdict is
    insensitive to that convention by many decades -- on the 2026-08-16
    payload's profile the worst pair's t statistic is far past the critical
    value at any level between 0.05 and 1e-12 -- so the gate does not rest on
    it.  A deterministic parametric test is used rather than a bootstrap
    because this repository quarantines irreproducible numbers, and a
    resampled p-value cannot resolve a level this small in any case.

    Octaves holding fewer than two positions cannot supply dispersion and are
    reported but excluded from the comparison family.

    A second, absolute check uses ``ln(vocab_size)``: the NLL of the uniform
    distribution over the teacher's own vocabulary.  A teacher no more
    informative than uniform is refused outright.  That anchor is derived from
    the payload's vocabulary, not picked.

    Raises ``TeacherPayloadError`` -- carrying the full block profile -- when
    either check fails; otherwise returns the profile.
    """
    nll, missing = teacher_forward_nll_per_position(
        topk_ids, topk_lps, calib_ids, vocab_size=vocab_size
    )
    scored_positions = int(nll.numel())
    blocks = _octave_position_blocks(int(nll.shape[1]))
    profile = _block_statistics(nll, missing, blocks)
    testable = [
        (index, block)
        for index, block in enumerate(profile)
        if int(block["positions"]) > 1
    ]
    comparisons: list[dict[str, object]] = []
    for position, (_, earlier) in enumerate(testable):
        for _, later in testable[position + 1:]:
            n_earlier = int(earlier["positions"])
            n_later = int(later["positions"])
            spread = (
                float(earlier["variance"]) / n_earlier
                + float(later["variance"]) / n_later
            )
            difference = float(later["nll_mean"]) - float(earlier["nll_mean"])
            if spread <= 0.0:
                # Exactly separated blocks with no within-block dispersion.
                welch_t = math.inf if difference > 0.0 else -math.inf
                degrees_of_freedom = math.inf
                p_value = 0.0 if difference > 0.0 else 1.0
            else:
                welch_t = difference / math.sqrt(spread)
                degrees_of_freedom = (spread * spread) / (
                    (float(earlier["variance"]) / n_earlier) ** 2 / (n_earlier - 1)
                    + (float(later["variance"]) / n_later) ** 2 / (n_later - 1)
                )
                p_value = student_t_upper_tail(welch_t, degrees_of_freedom)
            comparisons.append({
                "earlier_context_first": earlier["context_first"],
                "earlier_context_last": earlier["context_last"],
                "earlier_nll_mean": float(earlier["nll_mean"]),
                "later_context_first": later["context_first"],
                "later_context_last": later["context_last"],
                "later_nll_mean": float(later["nll_mean"]),
                "nll_increase": difference,
                "welch_t": welch_t,
                "degrees_of_freedom": degrees_of_freedom,
                "p_value": p_value,
            })

    policy = teacher_forward_fidelity_policy(
        scored_positions=scored_positions,
        comparisons=len(comparisons),
    )
    per_comparison_alpha = float(policy["per_comparison_alpha"])
    overall_mean = float(nll.mean().item())
    summary: dict[str, object] = {
        "schema": FORWARD_FIDELITY_POLICY_SCHEMA,
        "scored_positions": scored_positions,
        "nll_mean": overall_mean,
        "perplexity": (
            math.exp(overall_mean)
            if overall_mean < _MAX_REPORTABLE_NLL
            else math.inf
        ),
        "out_of_support_targets": int(missing.sum().item()),
        "uniform_nll_ceiling": math.log(float(vocab_size)),
        "per_comparison_alpha": per_comparison_alpha,
        "blocks": [
            {key: value for key, value in block.items() if key != "variance"}
            for block in profile
        ],
        "worst_comparison": (
            min(comparisons, key=lambda item: float(item["p_value"]))
            if comparisons
            else None
        ),
        "forward_fidelity_policy": policy,
    }

    uniform_ceiling = float(summary["uniform_nll_ceiling"])
    above_ceiling = [
        block for block in profile if float(block["nll_mean"]) >= uniform_ceiling
    ]
    if overall_mean >= uniform_ceiling or above_ceiling:
        raise TeacherPayloadError(
            "teacher is no more informative than the uniform distribution over "
            f"its own {vocab_size}-token vocabulary "
            f"(ln(V)={uniform_ceiling:.4f} nats): overall NLL "
            f"{overall_mean:.4f}, {len(above_ceiling)} context octave(s) at or "
            "above the ceiling\n" + format_forward_fidelity_profile(summary)
        )

    regressions = [
        item for item in comparisons
        if float(item["p_value"]) < per_comparison_alpha
    ]
    if regressions:
        detail = "\n".join(
            "    octave "
            f"{item['earlier_context_first']}-{item['earlier_context_last']} "
            f"NLL {item['earlier_nll_mean']:.3f} -> "
            f"{item['later_context_first']}-{item['later_context_last']} "
            f"NLL {item['later_nll_mean']:.3f} "
            f"(+{item['nll_increase']:.3f} nats, welch_t={item['welch_t']:.3f}, "
            f"df={item['degrees_of_freedom']:.1f}, p={item['p_value']:.3e})"
            for item in sorted(regressions, key=lambda item: float(item["p_value"]))
        )
        raise TeacherPayloadError(
            "teacher forward fidelity is not context-monotone: "
            f"{len(regressions)} of {len(comparisons)} ordered octave "
            "comparisons show later context significantly worse than earlier "
            f"at the derived per-comparison alpha "
            f"{per_comparison_alpha:.3e} "
            f"(family-wise 1/{scored_positions} over {len(comparisons)} "
            "comparisons). A correct autoregressive teacher improves with "
            "context; this one degrades, so any KL measured against it is "
            "meaningless.\n"
            + format_forward_fidelity_profile(summary)
            + "\n  significant regressions:\n"
            + detail
        )
    return summary


def safe_load_torch_payload(path: str | os.PathLike) -> object:
    """Deserialize tensors/primitives without permitting pickle execution."""
    payload_path = Path(path).resolve(strict=True)
    try:
        return torch.load(
            payload_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise TeacherPayloadError(
            "could not safely load teacher tensor payload"
        ) from exc


def tensor_descriptor(value: torch.Tensor) -> dict[str, object]:
    """Hash a tensor's contiguous CPU storage, independent of torch.save."""
    if not isinstance(value, torch.Tensor):
        raise TeacherPayloadError("tensor descriptor requires a torch.Tensor")
    tensor = value.detach().to("cpu").contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(dimension) for dimension in tensor.shape],
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def compact_source_model_identity(identity: Mapping[str, Any]) -> dict[str, object]:
    """Project a full streamed identity to the artifact provenance schema."""
    shards = identity.get("shards")
    checkpoint_weight_map = identity.get("checkpoint_weight_map")
    if not isinstance(shards, list) or not shards:
        raise TeacherPayloadError("source model identity has no checkpoint shards")
    if not isinstance(checkpoint_weight_map, Mapping) or not checkpoint_weight_map:
        raise TeacherPayloadError("source model identity has no checkpoint tensor map")
    compact = {
        "schema": identity.get("schema"),
        "content_sha256": identity.get("content_sha256"),
        "resolved_commit": identity.get("resolved_commit"),
        "checkpoint_shards": len(shards),
        "checkpoint_tensors": len(checkpoint_weight_map),
    }
    _require_sha256(compact["content_sha256"], where="source content_sha256")
    return compact


def tokenizer_identity(model_dir: str | os.PathLike) -> dict[str, object]:
    """Bind the exact local files that define tokenization for this lane."""
    root = Path(model_dir).resolve(strict=True)
    files: dict[str, dict[str, object]] = {}
    for name in _TOKENIZER_FILENAMES:
        path = root / name
        if path.is_file():
            files[name] = {
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
    if not files:
        raise TeacherPayloadError(f"no tokenizer files found under {root}")
    value_bearing = {"files": files}
    return {
        "schema": TOKENIZER_IDENTITY_SCHEMA,
        "content_sha256": canonical_sha256(value_bearing),
        **value_bearing,
    }


def tensor_semantic_projection(payload: Mapping[str, Any]) -> dict[str, object]:
    """Replace tensor bodies with byte descriptors for a stable payload hash."""
    expected = set(payload) - {"payload_semantic_sha256"}
    tensor_keys = teacher_tensor_keys(payload)
    missing = set(tensor_keys) - expected
    if missing:
        raise TeacherPayloadError(
            f"teacher payload misses semantic tensors: {sorted(missing)}"
        )
    projection: dict[str, object] = {}
    for key in sorted(expected):
        value = payload[key]
        projection[key] = tensor_descriptor(value) if key in tensor_keys else value
    return projection


def teacher_tensor_keys(payload: Mapping[str, Any]) -> tuple[str, ...]:
    if payload.get("schema") == TEACHER_PAYLOAD_V2_SCHEMA and payload.get("final_logprobs") is not None:
        return _TENSOR_KEYS + ("final_logprobs",)
    return _TENSOR_KEYS


def validate_final_logprobs(value: object, *, n_samples: int, vocab_size: int) -> torch.Tensor:
    """Validate a full normalized FP32 row without top-K or tail substitution."""
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or list(value.shape) != [n_samples, vocab_size]):
        raise TeacherPayloadError("final companion shape/dtype is invalid")
    if not bool(torch.isfinite(value).all()) or bool((value > 0).any()):
        raise TeacherPayloadError("final companion logprobs must be finite and nonpositive")
    mass = value.double().exp().sum(dim=-1)
    if bool((torch.abs(mass - 1) > TOPK_PROBABILITY_MASS_ABS_TOLERANCE).any()):
        raise TeacherPayloadError("final companion must carry the full normalized vocabulary")
    return value


def _validate_v2_source(payload: Mapping[str, Any]) -> None:
    _require_sha256(payload.get("wikitext_inputs_sha256"), where="teacher WikiText input file")
    try:
        from .dsv4_wikitext_inputs import normalize_wikitext_model_identity
    except ImportError:
        from dsv4_wikitext_inputs import normalize_wikitext_model_identity
    model = payload.get("model_identity")
    if not isinstance(model, Mapping) or set(model) != {"schema", "model_type", "text_model_type", "vocab_size"}:
        raise TeacherPayloadError("teacher input model identity fields are not closed")
    normalized = normalize_wikitext_model_identity({"model_type": model["model_type"],
        "text_config": {"model_type": model["text_model_type"], "vocab_size": model["vocab_size"]}})
    if model != normalized or model["vocab_size"] != payload["vocab_size"]:
        raise TeacherPayloadError("teacher input model identity/vocabulary differs")
    execution = payload.get("source_execution")
    if not isinstance(execution, Mapping):
        raise TeacherPayloadError("teacher source execution is missing")
    schema = execution.get("schema")
    expected = {"schema", "modules"}
    if schema == "prismaquant.joint_aura.source_execution.v2":
        expected.add("source_derivative")
    elif schema != "prismaquant.joint_aura.source_execution.v1":
        raise TeacherPayloadError("teacher source execution schema is unsupported")
    if set(execution) != expected or not isinstance(execution.get("modules"), Mapping):
        raise TeacherPayloadError("teacher source execution fields are not closed")
    for name, selectors in execution["modules"].items():
        if (not isinstance(name, str) or not isinstance(selectors, Mapping)
                or not selectors or not set(selectors) <= {"attention", "experts"}):
            raise TeacherPayloadError("teacher source execution selectors are invalid")
    if "source_derivative" in execution:
        from prismaquant.glm_source_derivative import (
            declaration, CORRECTED_MODELING_SHA256, CORRECTED_IMAGE_CONTENT_SHA256,
            ORIGINAL_HUB_KERNELS_SHA256, ORIGINAL_ACCELERATE_INTEGRATION_SHA256,
        )
        derivative = execution["source_derivative"]
        fixed = {"declaration": declaration(), "modeling_sha256": CORRECTED_MODELING_SHA256,
                 "image_content_sha256": CORRECTED_IMAGE_CONTENT_SHA256,
                 "hub_kernels_sha256": ORIGINAL_HUB_KERNELS_SHA256,
                 "accelerate_integration_sha256": ORIGINAL_ACCELERATE_INTEGRATION_SHA256}
        if (not isinstance(derivative, Mapping)
                or set(derivative) != set(fixed) | {"gates", "image_build_sha256"}
                or any(derivative.get(key) != value for key, value in fixed.items())
                or not isinstance(derivative.get("gates"), Mapping) or not derivative["gates"]):
            raise TeacherPayloadError("teacher source derivative identity differs")
        _require_sha256(derivative["image_build_sha256"], where="teacher derivative image build")
    canonical_json_bytes(execution)
    producer = payload.get("producer_identity")
    if not isinstance(producer, Mapping) or set(producer) != {"tools", "prismaquant_source_sha256"}:
        raise TeacherPayloadError("teacher producer identity fields are not closed")
    _require_sha256(producer["prismaquant_source_sha256"], where="teacher source package")
    from prismaquant.shipcard import _verify_gold_producer_identity
    tools = producer["tools"]
    problems = _verify_gold_producer_identity(
        "streamed teacher", {"git_commit": tools.get("git_commit") if isinstance(tools, Mapping) else None},
        {"measurement_tool": "build_streamed_full_kl_teacher", "producer_identity": tools},
        canonical_sha=canonical_sha256,
    )
    if problems:
        raise TeacherPayloadError("; ".join(problems))
    if payload.get("fit_overlap_status") != "unverified":
        raise TeacherPayloadError("fixed gold draw fitting overlap has not been verified")


def payload_semantic_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(tensor_semantic_projection(payload))


def build_calibration_contract(
    *,
    dataset_fingerprint: str,
    corpus_sha256: str,
    tokenizer: Mapping[str, Any],
    starts: list[int],
    total_tokens: int,
    calib_ids: torch.Tensor,
) -> dict[str, object]:
    """Create the closed WikiText window/tokenization contract."""
    if len(starts) != N_SAMPLES or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in starts
    ):
        raise TeacherPayloadError("calibration starts must be eight nonnegative ints")
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
        raise TeacherPayloadError("dataset fingerprint is missing")
    _require_sha256(corpus_sha256, where="calibration corpus_sha256")
    tokenizer_sha = tokenizer.get("content_sha256")
    _require_sha256(tokenizer_sha, where="tokenizer content_sha256")
    contract = {
        "schema": CALIBRATION_SCHEMA,
        "dataset": {
            "name": WIKITEXT_DATASET,
            "config": WIKITEXT_CONFIG,
            "split": WIKITEXT_SPLIT,
            "revision": WIKITEXT_REVISION,
            "fingerprint": dataset_fingerprint,
            "corpus_sha256": corpus_sha256,
        },
        "corpus_construction": {
            "row_filter": "include iff bool(text.strip()); preserve text verbatim",
            "join_separator": "\n\n",
            "normalization": "none",
        },
        "tokenizer": {
            "identity_sha256": tokenizer_sha,
            "trust_remote_code": True,
            "add_special_tokens": False,
        },
        "window_seed": WINDOW_SEED,
        "sampler": "python.random.Random(seed).sample(range(max_start), n_samples)/v1",
        "n_samples": N_SAMPLES,
        "seqlen": SEQLEN,
        "starts": list(starts),
        "total_tokens": int(total_tokens),
        "calib_ids_sha256": tensor_descriptor(calib_ids)["sha256"],
        "scoring": {
            "positions": "all",
            "prompt_top_k": PROMPT_TOP_K,
            "logprob_dtype": "float32",
            "tail_bucket": True,
        },
    }
    validate_calibration_contract(contract, calib_ids=calib_ids)
    return contract


def validate_calibration_contract(
    contract: object,
    *,
    calib_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise TeacherPayloadError("calibration contract is not an object")
    expected_keys = {
        "schema", "dataset", "corpus_construction", "tokenizer",
        "window_seed", "sampler", "n_samples", "seqlen", "starts",
        "total_tokens", "calib_ids_sha256", "scoring",
    }
    if set(contract) != expected_keys:
        raise TeacherPayloadError("calibration contract fields are not closed")
    if contract.get("schema") != CALIBRATION_SCHEMA:
        raise TeacherPayloadError("unsupported calibration contract schema")
    expected_dataset = {
        "name": WIKITEXT_DATASET,
        "config": WIKITEXT_CONFIG,
        "split": WIKITEXT_SPLIT,
        "revision": WIKITEXT_REVISION,
    }
    dataset = contract.get("dataset")
    if not isinstance(dataset, Mapping) or any(
        dataset.get(key) != value for key, value in expected_dataset.items()
    ) or set(dataset) != {*expected_dataset, "fingerprint", "corpus_sha256"}:
        raise TeacherPayloadError("calibration dataset identity differs")
    if not isinstance(dataset.get("fingerprint"), str) or not dataset.get(
        "fingerprint"
    ):
        raise TeacherPayloadError("calibration dataset fingerprint is missing")
    _require_sha256(dataset.get("corpus_sha256"), where="calibration corpus_sha256")
    if contract.get("corpus_construction") != {
        "row_filter": "include iff bool(text.strip()); preserve text verbatim",
        "join_separator": "\n\n",
        "normalization": "none",
    }:
        raise TeacherPayloadError("calibration corpus construction differs")
    tokenizer = contract.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or set(tokenizer) != {
        "identity_sha256", "trust_remote_code", "add_special_tokens"
    } or tokenizer.get("trust_remote_code") is not True or tokenizer.get(
        "add_special_tokens"
    ) is not False:
        raise TeacherPayloadError("calibration tokenizer contract differs")
    _require_sha256(tokenizer.get("identity_sha256"), where="tokenizer identity")
    if (
        contract.get("window_seed") != WINDOW_SEED
        or contract.get("sampler")
        != "python.random.Random(seed).sample(range(max_start), n_samples)/v1"
        or contract.get("n_samples") != N_SAMPLES
        or contract.get("seqlen") != SEQLEN
    ):
        raise TeacherPayloadError("calibration window contract differs")
    starts = contract.get("starts")
    if not isinstance(starts, list) or len(starts) != N_SAMPLES or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in starts
    ):
        raise TeacherPayloadError("calibration starts are malformed")
    if isinstance(contract.get("total_tokens"), bool) or not isinstance(
        contract.get("total_tokens"), int
    ) or int(contract["total_tokens"]) < SEQLEN + 1:
        raise TeacherPayloadError("calibration total token count is invalid")
    _require_sha256(contract.get("calib_ids_sha256"), where="calib_ids_sha256")
    if contract.get("scoring") != {
        "positions": "all",
        "prompt_top_k": PROMPT_TOP_K,
        "logprob_dtype": "float32",
        "tail_bucket": True,
    }:
        raise TeacherPayloadError("calibration scoring contract differs")
    if calib_ids is not None:
        if list(calib_ids.shape) != [N_SAMPLES, SEQLEN] or calib_ids.dtype != torch.long:
            raise TeacherPayloadError("calib_ids shape/dtype differs from contract")
        if tensor_descriptor(calib_ids)["sha256"] != contract.get(
            "calib_ids_sha256"
        ):
            raise TeacherPayloadError("calib_ids bytes differ from contract")
    return dict(contract)


def validate_teacher_payload(payload: object) -> dict[str, Any]:
    """Fail closed over every payload field and tensor byte descriptor."""
    if not isinstance(payload, Mapping):
        raise TeacherPayloadError("teacher payload is not an object")
    expected_keys = {
        "schema", "score_positions", "prompt_top_k", "topk_ids", "topk_lps",
        "calib_ids", "starts", "model", "n_samples", "seqlen", "vocab_size",
        "source_model_identity", "source_model", "source_model_identity_sha256",
        "calibration_contract", "calibration_contract_sha256",
        "payload_semantic_sha256",
    }
    v2 = payload.get("schema") == TEACHER_PAYLOAD_V2_SCHEMA
    if v2:
        expected_keys |= _V2_FIELDS
    if set(payload) != expected_keys:
        raise TeacherPayloadError("teacher payload fields are not closed")
    if payload.get("schema") not in {TEACHER_PAYLOAD_SCHEMA, TEACHER_PAYLOAD_V2_SCHEMA}:
        raise TeacherPayloadError("unsupported teacher payload schema")
    if (
        payload.get("score_positions") != "all"
        or payload.get("prompt_top_k") != PROMPT_TOP_K
        or payload.get("n_samples") != N_SAMPLES
        or payload.get("seqlen") != SEQLEN
    ):
        raise TeacherPayloadError("teacher scoring dimensions differ")
    vocab_size = payload.get("vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= PROMPT_TOP_K:
        raise TeacherPayloadError("teacher vocab_size is invalid")
    if v2:
        _validate_v2_source(payload)
        if payload["final_logprobs"] is not None:
            validate_final_logprobs(payload["final_logprobs"], n_samples=N_SAMPLES, vocab_size=vocab_size)
    calib_ids = payload.get("calib_ids")
    topk_ids = payload.get("topk_ids")
    topk_lps = payload.get("topk_lps")
    if not isinstance(calib_ids, torch.Tensor) or calib_ids.dtype != torch.long or list(
        calib_ids.shape
    ) != [N_SAMPLES, SEQLEN]:
        raise TeacherPayloadError("teacher calib_ids shape/dtype is invalid")
    expected_topk_shape = [N_SAMPLES, SEQLEN - 1, PROMPT_TOP_K]
    if not isinstance(topk_ids, torch.Tensor) or topk_ids.dtype != torch.int32 or list(
        topk_ids.shape
    ) != expected_topk_shape:
        raise TeacherPayloadError("teacher topk_ids shape/dtype is invalid")
    if not isinstance(topk_lps, torch.Tensor) or topk_lps.dtype != torch.float32 or list(
        topk_lps.shape
    ) != expected_topk_shape:
        raise TeacherPayloadError("teacher topk_lps shape/dtype is invalid")
    topk_coverage_summary(topk_ids, topk_lps, vocab_size=vocab_size)
    # Coverage cannot see an unfaithful forward: a confidently wrong teacher is
    # still sharply peaked.  The context-monotonicity gate can, and it runs on
    # every validation -- build, sidecar and replay -- so a payload that fails
    # it is never written and never grades a student.
    teacher_forward_fidelity_summary(
        topk_ids, topk_lps, calib_ids, vocab_size=vocab_size
    )
    identity = payload.get("source_model_identity")
    try:
        from prismaquant.cost_streaming import validate_streamed_model_identity

        full_identity = validate_streamed_model_identity(
            identity, where="full KL teacher payload"
        )
    except Exception as exc:
        raise TeacherPayloadError("teacher source-model identity is invalid") from exc
    expected_identity_sha = canonical_sha256(full_identity)
    if payload.get("source_model_identity_sha256") != expected_identity_sha:
        raise TeacherPayloadError("teacher full source identity digest differs")
    if payload.get("source_model") != compact_source_model_identity(full_identity):
        raise TeacherPayloadError("teacher compact source identity differs")
    contract = validate_calibration_contract(
        payload.get("calibration_contract"), calib_ids=calib_ids
    )
    if payload.get("calibration_contract_sha256") != canonical_sha256(contract):
        raise TeacherPayloadError("teacher calibration contract digest differs")
    if payload.get("starts") != contract.get("starts"):
        raise TeacherPayloadError("teacher starts differ from calibration contract")
    observed_semantic = payload_semantic_sha256(payload)
    if payload.get("payload_semantic_sha256") != observed_semantic:
        raise TeacherPayloadError("teacher semantic payload digest differs")
    return dict(payload)


def _v2_evidence_fields(payload: Mapping[str, Any]) -> dict[str, object]:
    if payload.get("schema") != TEACHER_PAYLOAD_V2_SCHEMA:
        return {}
    final = payload["final_logprobs"]
    return {
        "source_execution": payload["source_execution"],
        "producer_identity": payload["producer_identity"],
        "fit_overlap_status": payload["fit_overlap_status"],
        "wikitext_inputs_sha256": payload["wikitext_inputs_sha256"],
        "model_identity": payload["model_identity"],
        "final_logprobs_descriptor": None if final is None else tensor_descriptor(final),
    }


def teacher_meta(
    *,
    payload_path: str | os.PathLike,
    elapsed_s: float,
) -> dict[str, object]:
    """Construct a sidecar strictly from the serialized tensor payload."""
    path = Path(payload_path).resolve(strict=True)
    validated = validate_teacher_payload(safe_load_torch_payload(path))
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise TeacherPayloadError("teacher elapsed time is invalid")
    coverage = topk_coverage_summary(
        validated["topk_ids"],
        validated["topk_lps"],
        vocab_size=int(validated["vocab_size"]),
    )
    return {
        "schema": (TEACHER_META_V2_SCHEMA if validated["schema"] == TEACHER_PAYLOAD_V2_SCHEMA
                   else TEACHER_META_SCHEMA),
        "payload": str(path),
        "payload_sha256": file_sha256(path),
        "payload_bytes": int(path.stat().st_size),
        "payload_semantic_sha256": validated["payload_semantic_sha256"],
        "source_model": validated["source_model"],
        "source_model_identity_sha256": validated["source_model_identity_sha256"],
        "calibration_contract": validated["calibration_contract"],
        "calibration_contract_sha256": validated["calibration_contract_sha256"],
        "tensor_descriptors": {
            key: tensor_descriptor(validated[key]) for key in teacher_tensor_keys(validated)
        },
        **_v2_evidence_fields(validated),
        "teacher_shape": list(validated["topk_lps"].shape),
        **coverage,
        "elapsed_s": float(elapsed_s),
    }


def atomic_torch_save(payload: object, path: str | os.PathLike) -> None:
    """Publish a torch payload by rename, never as a partial final file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise TeacherPayloadError(f"temporary payload already exists: {temporary}")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_write(payload: object, path: str | os.PathLike) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise TeacherPayloadError(f"temporary metadata already exists: {temporary}")
    data = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_teacher_evidence(
    payload_path: str | os.PathLike,
    meta_path: str | os.PathLike,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Load/replay teacher bytes and return shipcard-sized evidence."""
    payload_file = Path(payload_path).resolve(strict=True)
    meta_file = Path(meta_path).resolve(strict=True)
    try:
        payload = safe_load_torch_payload(payload_file)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except TeacherPayloadError:
        raise
    except Exception as exc:
        raise TeacherPayloadError("could not load teacher payload evidence") from exc
    validated = validate_teacher_payload(payload)
    v2 = validated["schema"] == TEACHER_PAYLOAD_V2_SCHEMA
    if not isinstance(meta, Mapping) or meta.get("schema") != (
            TEACHER_META_V2_SCHEMA if v2 else TEACHER_META_SCHEMA):
        raise TeacherPayloadError("unsupported teacher metadata schema")
    if meta.get("payload_sha256") != file_sha256(payload_file) or meta.get(
        "payload_bytes"
    ) != payload_file.stat().st_size:
        raise TeacherPayloadError("teacher serialized payload bytes differ")
    expected_meta_fields = {
        "schema", "payload", "payload_sha256", "payload_bytes",
        "payload_semantic_sha256", "source_model",
        "source_model_identity_sha256", "calibration_contract",
        "calibration_contract_sha256", "tensor_descriptors", "teacher_shape",
        "topk_coverage_mean", "topk_coverage_min", "topk_coverage_policy",
        "elapsed_s",
    }
    expected_meta_fields |= set(_v2_evidence_fields(validated))
    if set(meta) != expected_meta_fields:
        raise TeacherPayloadError("teacher metadata fields are not closed")
    comparisons = {
        "payload_semantic_sha256": validated["payload_semantic_sha256"],
        "source_model": validated["source_model"],
        "source_model_identity_sha256": validated["source_model_identity_sha256"],
        "calibration_contract": validated["calibration_contract"],
        "calibration_contract_sha256": validated["calibration_contract_sha256"],
        "tensor_descriptors": {
            key: tensor_descriptor(validated[key]) for key in teacher_tensor_keys(validated)
        },
        **_v2_evidence_fields(validated),
        "teacher_shape": list(validated["topk_lps"].shape),
        **topk_coverage_summary(
            validated["topk_ids"],
            validated["topk_lps"],
            vocab_size=int(validated["vocab_size"]),
        ),
    }
    if any(meta.get(key) != value for key, value in comparisons.items()):
        raise TeacherPayloadError("teacher metadata differs from payload semantics")
    elapsed_s = meta.get("elapsed_s")
    if (
        isinstance(elapsed_s, bool)
        or not isinstance(elapsed_s, (int, float))
        or not math.isfinite(float(elapsed_s))
        or float(elapsed_s) < 0.0
    ):
        raise TeacherPayloadError("teacher metadata elapsed time is invalid")
    evidence = {
        "schema": TEACHER_EVIDENCE_V2_SCHEMA if v2 else TEACHER_EVIDENCE_SCHEMA,
        "payload_sha256": meta["payload_sha256"],
        "payload_bytes": meta["payload_bytes"],
        "payload_semantic_sha256": meta["payload_semantic_sha256"],
        "meta_sha256": file_sha256(meta_file),
        "source_model": meta["source_model"],
        "source_model_identity_sha256": meta["source_model_identity_sha256"],
        "calibration_contract": meta["calibration_contract"],
        "calibration_contract_sha256": meta["calibration_contract_sha256"],
        "topk_coverage_mean": meta["topk_coverage_mean"],
        "topk_coverage_min": meta["topk_coverage_min"],
        "topk_coverage_policy": meta["topk_coverage_policy"],
        **_v2_evidence_fields(validated),
    }
    return validated, evidence


__all__ = [
    "CALIBRATION_SCHEMA",
    "EXPECTED_POSITIONS",
    "FORWARD_FIDELITY_POLICY_SCHEMA",
    "N_SAMPLES",
    "PROMPT_TOP_K",
    "SEQLEN",
    "TEACHER_EVIDENCE_SCHEMA",
    "TEACHER_META_SCHEMA",
    "TEACHER_PAYLOAD_SCHEMA",
    "TEACHER_PAYLOAD_V2_SCHEMA",
    "TEACHER_META_V2_SCHEMA",
    "TEACHER_EVIDENCE_V2_SCHEMA",
    "TeacherPayloadError",
    "TOPK_COVERAGE_POLICY_SCHEMA",
    "TOPK_MINIMUM_COVERAGE",
    "TOPK_PROBABILITY_MASS_ABS_TOLERANCE",
    "WIKITEXT_CONFIG",
    "WIKITEXT_DATASET",
    "WIKITEXT_REVISION",
    "WIKITEXT_SPLIT",
    "WINDOW_SEED",
    "atomic_json_write",
    "atomic_torch_save",
    "build_calibration_contract",
    "canonical_sha256",
    "compact_source_model_identity",
    "file_sha256",
    "format_forward_fidelity_profile",
    "load_teacher_evidence",
    "payload_semantic_sha256",
    "safe_load_torch_payload",
    "student_t_upper_tail",
    "teacher_forward_fidelity_policy",
    "teacher_forward_fidelity_summary",
    "teacher_forward_nll_per_position",
    "teacher_meta",
    "tensor_descriptor",
    "teacher_tensor_keys",
    "topk_coverage_policy",
    "topk_coverage_summary",
    "tokenizer_identity",
    "validate_calibration_contract",
    "validate_teacher_payload",
    "validate_final_logprobs",
]
