"""One make, preflight and replay contract for each ship-record slot."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

BOUNDARY_ENDPOINT = "/v1/chat/completions"
BOUNDARY_REQUEST_SCHEMA = "prismaquant.boundary_chat_request/1"
BOUNDARY_RESPONSE_SCHEMA = "prismaquant.boundary_chat_response/1"
DEFAULT_MAX_PPL = 25.0
DEFAULT_MAX_P99_NLL = 6.0
DEFAULT_MAX_MEAN_NLL = 3.0
DEFAULT_MIN_GEN_LEN = 30               # chars in each generated completion
DEFAULT_MIN_MTP_ACCEPT_P0 = 0.60       # position-0 accept fraction
# These boundary values remain historical fail-closed defaults pending the
# paired-control decision in issue #87. They are not universal calibration claims.
DEFAULT_MAX_BOUNDARY_DEFECTS = 0
DEFAULT_BOUNDARY_TEMPERATURE = 1.0
DEFAULT_BOUNDARY_MAX_TOKENS = 64
DEFAULT_BOUNDARY_REPS = 6

#: Every slot all serving lanes must close before an artifact is shippable.
#: Keep this base tuple stable: no container inherits a plugin-only gate
#: merely because one lane adds one.
REQUIRED_SLOTS = ("native_export.eager", "native_export.graph", "ship_gate", "gold.kl", "gold.ppl")
GOLD_SLOTS = frozenset({"gold.kl", "gold.ppl"})
#: The byte-matched uniform control's verdict (RobTand/prismaquant#121).
UNIFORM_CONTROL_SLOT = "uniform_control"
OPTIONAL_SLOTS = (UNIFORM_CONTROL_SLOT,)
#: Principle 14's serve-side leg on the COMPRESSED-TENSORS lane (#631).  Stock
#: vLLM emits no route telemetry, so this lane's served side is read off the
#: running engine through vLLM's own ``LLM.apply_model`` by
#: ``validate_native_export --route-sweep-out``: the resolved quantization
#: method, scheme, scheme attributes and forward-dispatch count per module.
#: A separate slot from `route.trace` because it is a different observation
#: from a different producer -- a resolved-state sweep, not the serve's own
#: counters -- and reading the two through one verifier would hide which one
#: an artifact actually closed.  The record carries the sweeps and the config
#: text, and `verify` replays the comparison (`compressed_route_sweep_gate`).
ROUTE_SWEEP_SLOT = "route.sweep"
PRIMARY_METRIC_KEYS = ("kl_confident_mean", "kl_mean", "ppl", "mean_nll", "last_token_kl")
CARRIED_METRIC_KEYS = PRIMARY_METRIC_KEYS + (
    "kl_p99", "kl_max", "n_positions", "n_confident", "n_samples", "seqlen",
    "n_tokens_scored", "per_chunk_mean_nll", "max_chunk_mean_nll", "quantization",
    "score_positions", "serve_manifest", "teacher_evidence", "mode", "prompt_top_k",
    "vocab_size", "split", "n_tokens_requested", "calibration_contract", "calibration_contract_sha256")
GOLD_PRIMARY_METRICS = {"gold.kl": ("kl_confident_mean", "kl_mean"),
                        "gold.ppl": ("ppl", "mean_nll")}

def verify_gold_record(
    slot: str,
    record: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
    require_current_artifact_path: bool = True,
) -> list[str]:
    """Require a slot-specific finite gold metric plus measurement identity."""
    problems: list[str] = []
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return [f"{slot}: missing structured gold metrics"]
    allowed = GOLD_PRIMARY_METRICS[slot]
    finite = [
        key for key in allowed
        if key in metrics
        and not isinstance(metrics[key], bool)
        and isinstance(metrics[key], (int, float))
        and math.isfinite(float(metrics[key]))
        and float(metrics[key]) >= 0
    ]
    if not finite:
        problems.append(
            f"{slot}: carries no finite non-negative slot-specific metric "
            f"from {allowed}"
        )
    fingerprint = record.get("serve_fingerprint")
    if not isinstance(fingerprint, str) or re.fullmatch(
        r"[0-9a-f]{64}", fingerprint
    ) is None:
        problems.append(f"{slot}: missing exact serve fingerprint")
    commit = record.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
    ) is None:
        problems.append(f"{slot}: missing full producer git commit")
    tool = record.get("tool")
    if not isinstance(tool, str) or not tool:
        problems.append(f"{slot}: missing measurement tool identity")
    if slot == "gold.kl":
        if not any(
            isinstance(metrics.get(key), int)
            and not isinstance(metrics.get(key), bool)
            and metrics.get(key, 0) > 0
            for key in ("n_positions", "n_samples")
        ):
            problems.append(f"{slot}: missing positive KL sample/position count")
        # A positive count was never enough. `--score-positions final` scores
        # ONE position per sequence -- the window-final context -- which is the
        # cheap last-token "hook KL" screen that is triage-only and must never
        # be a promotion number (§5). It reports n_samples=8 and sails through
        # the count check above, so the card could not tell an 8-position screen
        # from a 4088-position gold measurement. Caught 2026-08-14 on the
        # Qwen3.8-27B lane, where the driver simply omitted the flag and the
        # tool's default is `final`.
        if metrics.get("score_positions") != "all":
            problems.append(
                f"{slot}: score_positions={metrics.get('score_positions')!r} — "
                "the gold KL contract is every prompt position "
                "(--score-positions all, n_positions = n_samples*(seqlen-1)). "
                "'final' is the last-token hook screen: triage only, never a "
                "promotion metric."
            )
    elif not isinstance(metrics.get("n_tokens_scored"), int) or isinstance(
        metrics.get("n_tokens_scored"), bool
    ) or metrics.get("n_tokens_scored", 0) <= 0:
        problems.append(f"{slot}: missing positive scored-token count")
    return problems


def _verify_native_export_record(
    slot: str,
    record: Mapping[str, Any],
) -> list[str]:
    """Replay the smoke arm's own stamped metrics against the slot it closes.

    `validate_native_export._record_arm` stamps the arm it actually ran
    (`metrics.arm`), the residency it ran under (`enforce_eager`), and how
    much it generated (`generated_chars`) — but `verify` compared only the
    slot key against `record["slot"]`, so an eager receipt closed the graph
    slot and a pass that generated nothing verified clean. The only site
    that knows the truth already writes it; this reads it.
    """
    problems: list[str] = []
    arm = slot.split(".", 1)[1] if "." in slot else slot
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return [f"{slot}: missing structured arm metrics"]
    if metrics.get("arm") != arm:
        problems.append(
            f"{slot}: metrics.arm={metrics.get('arm')!r} does not match the "
            f"slot arm {arm!r} — a mislabeled receipt, not a result"
        )
    # `validate_native_export._run_arm`: `arm = "eager" if enforce_eager else
    # "graph"`. The two residencies are different numeric objects on the
    # Tessera lane, exercised by separate gates.
    expected_eager = {"eager": True, "graph": False}.get(arm)
    if (
        expected_eager is not None
        and metrics.get("enforce_eager") is not expected_eager
    ):
        problems.append(
            f"{slot}: enforce_eager={metrics.get('enforce_eager')!r} "
            f"disagrees with the {arm} arm — this receipt ran under the "
            "other residency"
        )
    produced = metrics.get("generated_chars")
    if record.get("passed") is True:
        if (
            isinstance(produced, bool)
            or not isinstance(produced, int)
            or produced <= 0
        ):
            problems.append(
                f"{slot}: passed but generated_chars={produced!r} — a smoke "
                "pass that generated nothing is not evidence"
            )
    elif (
        not isinstance(produced, bool)
        and isinstance(produced, int)
        and produced > 0
    ):
        problems.append(
            f"{slot}: failed yet claims generated_chars={produced!r}"
        )
    max_new_tokens = metrics.get("max_new_tokens")
    if max_new_tokens is not None and (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        problems.append(
            f"{slot}: max_new_tokens={max_new_tokens!r} is not a positive "
            "integer"
        )
    return problems


def _verify_ship_gate_record(
    slot: str,
    record: Mapping[str, Any],
) -> list[str]:
    """Replay the fixed catastrophic-quality thresholds, the check ledger
    with its token evidence, and the endpoint binding.

    The binding half is presence-and-shape, stated honestly: an offline
    `verify` has no live session to compare against, so it refuses a record
    that names no server, no served model and no artifact path -- but a
    well-formed binding to the WRONG server still passes here. Catching that
    needs the nonce-bound live-session check (the retired CB eager driver
    did exactly that before teardown); see the `ship_gate` paragraph in
    §7.2 of docs/ARCHITECTURE.md.
    """
    problems: list[str] = []
    if record.get("tool") != "validate_quantized_model.py":
        problems.append(f"{slot}: not filled by validate_quantized_model.py")
    commit = record.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
    ) is None:
        problems.append(f"{slot}: missing full producer git commit")
    endpoint = record.get("base_url")
    if (
        not isinstance(endpoint, str)
        or not endpoint.startswith(("http://", "https://"))
    ):
        problems.append(f"{slot}: missing endpoint binding (base_url)")
    served = record.get("served_model_name")
    if not isinstance(served, str) or not served.strip():
        problems.append(
            f"{slot}: missing served-model binding (served_model_name)")
    source = record.get("model_sha_source")
    if not isinstance(source, str) or not source.strip():
        problems.append(
            f"{slot}: missing artifact-path binding (model_sha_source)")
    metrics = record.get("metrics")
    thresholds = record.get("thresholds")
    expected_checks = {
        "serve_ready", "generation_sanity", "perplexity", "mtp_acceptance",
        "boundary_behavior",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != expected_checks:
        return problems + [f"{slot}: validation check ledger is incomplete"]
    for name, row in metrics.items():
        if not isinstance(row, Mapping) or row.get("passed") is not True:
            problems.append(f"{slot}: check {name} is not a structured pass")
    expected_thresholds = {
        "max_ppl": DEFAULT_MAX_PPL,
        "max_mean_nll": DEFAULT_MAX_MEAN_NLL,
        "max_p99_nll": DEFAULT_MAX_P99_NLL,
        "min_gen_len": DEFAULT_MIN_GEN_LEN,
        "min_mtp_accept_p0": DEFAULT_MIN_MTP_ACCEPT_P0,
        "max_boundary_defects": DEFAULT_MAX_BOUNDARY_DEFECTS,
        "boundary_temperature": DEFAULT_BOUNDARY_TEMPERATURE,
        "boundary_max_tokens": DEFAULT_BOUNDARY_MAX_TOKENS,
        "boundary_reps": DEFAULT_BOUNDARY_REPS,
    }
    if not isinstance(thresholds, Mapping):
        problems.append(f"{slot}: missing threshold contract")
    else:
        for key, expected in expected_thresholds.items():
            value = thresholds.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != float(expected)
            ):
                problems.append(
                    f"{slot}: threshold {key}={value!r}, expected {expected!r}"
                )
    perplexity = metrics.get("perplexity")
    if isinstance(perplexity, Mapping):
        numeric = {
            "perplexity": ("max_ppl", DEFAULT_MAX_PPL),
            "mean_nll_per_tok": ("max_mean_nll", DEFAULT_MAX_MEAN_NLL),
            "max_nll_per_tok": ("max_p99_nll", DEFAULT_MAX_P99_NLL),
        }
        for key, (_threshold_key, limit) in numeric.items():
            value = perplexity.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                or float(value) > limit
            ):
                problems.append(
                    f"{slot}: perplexity metric {key} does not clear {limit}"
                )
        tokens = perplexity.get("n_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            problems.append(f"{slot}: perplexity check scored no tokens")
        if perplexity.get("spec_decode_detected") is True:
            problems.append(f"{slot}: perplexity ran under speculative decode")
        elif perplexity.get("spec_decode_detected") is not False:
            problems.append(
                f"{slot}: perplexity speculative-decode state is unknown"
            )
    else:
        problems.append(f"{slot}: missing structured perplexity evidence")
    boundary = metrics.get("boundary_behavior")
    if isinstance(boundary, Mapping):
        # The behavioral replay: raw `/v1/completions` never applies the chat
        # template, so a clean-looking count from that endpoint is not
        # boundary evidence.  Pin the producer's request and extraction
        # schemas independently here, alongside the numerical replay below.
        endpoint = boundary.get("endpoint")
        if endpoint != BOUNDARY_ENDPOINT:
            problems.append(
                f"{slot}: boundary endpoint={endpoint!r}, expected "
                "'/v1/chat/completions'"
            )
        request_schema = boundary.get("request_schema")
        if request_schema != BOUNDARY_REQUEST_SCHEMA:
            problems.append(
                f"{slot}: boundary request schema={request_schema!r}, "
                "expected 'prismaquant.boundary_chat_request/1'"
            )
        response_schema = boundary.get("response_schema")
        if response_schema != BOUNDARY_RESPONSE_SCHEMA:
            problems.append(
                f"{slot}: boundary response schema={response_schema!r}, "
                "expected 'prismaquant.boundary_chat_response/1'"
            )

        # A sampled check with no generations, run at temp 0, or with defects
        # over the zero-tolerance bound is not evidence — it is the old
        # argmax-blind gate wearing the new name.  The zero/64 values remain
        # fail-closed historical defaults pending #87's paired-control policy;
        # replaying them here does not claim they are calibrated universally.
        generations = boundary.get("n_generations")
        if (
            isinstance(generations, bool)
            or not isinstance(generations, int)
            or generations <= 0
        ):
            problems.append(
                f"{slot}: boundary check sampled no generations")
        defects = boundary.get("n_defects")
        if (
            isinstance(defects, bool)
            or not isinstance(defects, int)
            or defects < 0
            or defects > 0
        ):
            problems.append(
                f"{slot}: boundary check reports {defects!r} defective "
                "generations — the historical bound is zero; "
                "its control-relative replacement remains pending #87"
            )
        temperature = boundary.get("temperature")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0
        ):
            problems.append(
                f"{slot}: boundary check temperature={temperature!r} is not "
                "sampling — the defect is invisible at temp 0"
            )
    else:
        problems.append(f"{slot}: missing structured boundary evidence")
    return problems


# ---------------------------------------------------------------------------
# The byte-matched uniform control (#121, closing prismaquant#117 / tessera#1)
# ---------------------------------------------------------------------------
# An allocation over a rate *axis* is a claim: that choosing a rung per Linear
# beats spending the same bytes everywhere.  On 2026-09-02 that claim was
# measured and it was false -- a PrismaQuant-allocated Tessera checkpoint
# served 2.00x worse KL than a byte-matched uniform arm at 4.0 bpp (0.3485 vs
# 0.1746; 2.33x at 3.0, 2.88x at 5.0) while every other check the pipeline
# owns passed: 196/196 units matched the plan's bytes, a layer-0 re-encode was
# byte-identical, census read 112/112.  The pipeline computed one artifact
# correctly.  It was the wrong artifact.
#
# On 2026-09-03 an ORACLE handed the measured per-unit KL table reached only
# 0.941x the uniform control at 4.0 bpp, P(worse) = 0.075 -- not significant.
# So the closure is a REFUSAL, not a better cost model: there is no prize at
# that rung to fund one with.  Below the knee the axis is real (3.0 bpp:
# oracle 0.748x, AURA 0.780x, both P = 0.000), which is why this is a gate on
# the claim and not a verdict on allocation.
#
# Principle 3 applied to allocation itself: an allocation that cannot beat its
# byte-matched uniform control has not earned its allocation, and does not
# ship.  The lesson this is written against is that provenance nothing
# consumes is a confession log -- 73.7% of a 92 GB body once rode an
# `arch::Sm80` fallback on Blackwell, recorded in its own selection.json,
# refused by nothing.  So this REFUSES, in `verify`, at publication.
#
# How the record gets here:
#   1. Tessera builds and prices the control:
#        python -m tessera.uniform_control plan <candidate plan> --model <source>
#      then serves it, and `verify` re-asserts the byte match on the two
#      exported manifests.
#   2. The SAME KL tool that filled `gold.kl` is run on the control checkpoint
#      -- the control arm's record is gold-record shaped and is replayed
#      through `verify_gold_record` above, so a last-token or weight-space
#      number cannot close this slot.
#   3. `python -m prismaquant.shipcard_cli fill-control` writes both, plus the
def uniform_control_schemas() -> tuple[str, ...]:
    """The uniform-control block schemas a ship record accepts.

    A rate-axis format family names the block its producer emits
    (``uniform_control_schema`` in its lane spec; the Tessera lane's is
    ``tessera.control.control_block()``'s).  The block is carried verbatim.
    The shipcard never imports the producer, so the schema string is the whole
    of what is taken on trust: every number in the block is replayed here from
    integers.
    """
    from prismaquant.lane_spec import rate_axis_lanes

    return tuple(dict.fromkeys(
        family.uniform_control_schema
        for spec in rate_axis_lanes()
        for family in spec.format_families
        if family.rate_axis and family.uniform_control_schema))


def _schema_names(schemas: Sequence[str]) -> str:
    return " or ".join(repr(schema) for schema in schemas) or "(no rate-axis lane)"


#: The override that lets a measured loss ship anyway.
UNIFORM_CONTROL_OVERRIDE_SCHEMA = "prismaquant.uniform_control_override/1"
#: The widest a control's bytes may miss the candidate's and still be a
#: control, as a :class:`~fractions.Fraction` of the candidate's own bits.
#: This is ``tessera.control.DEFAULT_MAX_RELATIVE_SLACK``, restated here
#: because a block that widened its own tolerance would otherwise certify
#: itself: the carried ``max_relative_slack`` is checked against this ceiling
#: before it is used.  Zero slack is unreachable -- the rate axis is discrete,
#: one rung quantum is ~0.0039 bpp (~0.1% at 4 bpp) -- so "exact" here means
#: exact integer arithmetic against an explicit ceiling, not zero slack.
MAX_CONTROL_RELATIVE_SLACK = Fraction(1, 1000)
#: Measurement-contract keys the two arms must agree on when the candidate's
#: own ``gold.kl`` record carries them.  Driven by the candidate rather than
#: by a fixed list so the control cannot dodge a key by omitting it: the
#: candidate side is the card's published gold number and is gated separately.
UNIFORM_CONTROL_EXACT_CONTRACT_KEYS = (
    "measurement_fidelity", "calibration_contract_sha256", "teacher_evidence",
)
UNIFORM_CONTROL_CONTRACT_KEYS = (
    "n_samples",
    "seqlen",
    "n_positions",
    "score_positions",
    "corpus_sha256",
) + UNIFORM_CONTROL_EXACT_CONTRACT_KEYS
#: Which gold metric the two arms are compared on.  Both must quote the same
#: one, and it must be a KL: the gate's whole point is the serving metric.
UNIFORM_CONTROL_METRIC_KEYS = ("kl_mean", "kl_confident_mean")


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _close(a: float, b: float, *, rel: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=0.0)


def _replay_byte_match(
    slot: str,
    match: Any,
) -> tuple[list[str], bool]:
    """Recompute the byte match from the integers the block carries.

    "Byte-matched" is checked here, never read: a boolean the producer wrote
    is exactly the kind of claim principle 14 refuses to consume.  Everything
    below is exact :class:`~fractions.Fraction` arithmetic on integer bit
    totals, and the carried ``byte_matched`` flag has to agree with it.
    """
    problems: list[str] = []
    if not isinstance(match, Mapping):
        return [f"{slot}: the control block carries no byte match"], False
    candidate = _positive_int(match.get("candidate_bits"))
    control = _positive_int(match.get("control_bits"))
    params = _positive_int(match.get("varying_params"))
    if candidate is None or control is None or params is None:
        return [
            f"{slot}: byte match needs positive integer candidate_bits, "
            f"control_bits and varying_params; got "
            f"{match.get('candidate_bits')!r}, {match.get('control_bits')!r}, "
            f"{match.get('varying_params')!r}"
        ], False

    tolerance = match.get("max_relative_slack")
    if (
        not isinstance(tolerance, (list, tuple))
        or len(tolerance) != 2
        or any(_positive_int(part) is None for part in tolerance)
    ):
        return [
            f"{slot}: byte match declares no [numerator, denominator] "
            f"max_relative_slack; got {tolerance!r}"
        ], False
    tolerance = Fraction(int(tolerance[0]), int(tolerance[1]))
    if tolerance > MAX_CONTROL_RELATIVE_SLACK:
        problems.append(
            f"{slot}: the control widened its own tolerance to "
            f"{float(tolerance) * 100:.4f}%, over the "
            f"{float(MAX_CONTROL_RELATIVE_SLACK) * 100:.4f}% a control may be. "
            "Two arms that differ by more than that are two products, not an "
            "arm and its control."
        )
        tolerance = MAX_CONTROL_RELATIVE_SLACK

    slack_bits = control - candidate
    relative = Fraction(abs(slack_bits), candidate)
    matched = relative <= tolerance
    carried = match.get("byte_matched")
    if carried is not None and bool(carried) is not matched:
        problems.append(
            f"{slot}: the block claims byte_matched={carried!r} but its own "
            f"integers replay to {matched} "
            f"({int(slack_bits)} bits apart, {float(relative) * 1e6:.1f} ppm)"
        )
    carried_slack = match.get("slack_bits")
    if carried_slack is not None and (
        isinstance(carried_slack, bool)
        or not isinstance(carried_slack, int)
        or carried_slack != slack_bits
    ):
        problems.append(
            f"{slot}: slack_bits={carried_slack!r} but control_bits - "
            f"candidate_bits = {int(slack_bits)}"
        )
    for key, bits in (("candidate_bpp", candidate), ("control_bpp", control)):
        reported = _finite_float(match.get(key))
        if reported is not None and not _close(reported, bits / params):
            problems.append(
                f"{slot}: {key}={reported!r} but {bits} bits over {params} "
                f"parameters is {bits / params!r}"
            )
    if not matched:
        problems.append(
            f"{slot}: the arms are NOT byte-matched — the control weighs "
            f"{control} bits against the candidate's {candidate}, "
            f"{float(relative) * 100:.4f}% apart, over the "
            f"{float(tolerance) * 100:.4f}% a control may be. The "
            f"{'control' if slack_bits > 0 else 'candidate'} arm is the fatter "
            "one, so the comparison would price those bytes as quality."
        )
    return problems, matched


def _replay_control_arms(
    slot: str,
    record: Mapping[str, Any],
    verdict: Mapping[str, Any],
    *,
    card: Mapping[str, Any],
) -> list[str]:
    """Bind both arms to the serving metric, structurally.

    The candidate arm is not accepted as a number at all: it must BE the
    card's own ``gold.kl``, which is already gated to served KL
    with ``score_positions=all`` on a no-spec-decode serve. So a
    last-token hook screen or a weight-space error cannot reach this slot
    through the candidate leg, and a block measured on some other allocation
    cannot be pasted onto this artifact -- its candidate KL would not be this
    card's gold number.

    The control arm carries its own gold-shaped record and is replayed through
    the same :func:`verify_gold_record`, then held to the candidate's
    measurement contract key by key.
    """
    problems: list[str] = []
    metrics = record.get("metrics")
    key = (metrics or {}).get("gold_metric_key") if isinstance(
        metrics, Mapping) else None
    if key not in UNIFORM_CONTROL_METRIC_KEYS:
        return [
            f"{slot}: metrics.gold_metric_key={key!r} — the record must name "
            f"which of {list(UNIFORM_CONTROL_METRIC_KEYS)} both arms are "
            "compared on, so the comparison is not a choice made after the "
            "fact"
        ]

    gold = ((card.get("slots") or {}).get("gold.kl"))
    if not isinstance(gold, Mapping):
        return [
            f"{slot}: the card carries no gold.kl record, so the candidate "
            "arm has nothing to be. The candidate leg of this comparison is "
            "the artifact's own gold KL, not a number handed to the card."
        ]
    gold_metrics = gold.get("metrics")
    if not isinstance(gold_metrics, Mapping):
        return [f"{slot}: the card's gold.kl record carries no metrics"]
    gold_value = _finite_float(gold_metrics.get(key))
    candidate = _finite_float(verdict.get("candidate"))
    if gold_value is None:
        problems.append(
            f"{slot}: the card's gold.kl carries no finite {key}, so the "
            "candidate arm cannot be bound to it"
        )
    elif candidate is None or not _close(candidate, gold_value):
        problems.append(
            f"{slot}: the verdict's candidate {verdict.get('candidate')!r} is "
            f"not the card's own gold.kl {key}={gold_value!r}. The candidate "
            "arm must be the artifact's served gold KL — a comparison against "
            "some other measurement of some other artifact is not this "
            "artifact's control."
        )

    arm = record.get("control_arm")
    if not isinstance(arm, Mapping):
        return problems + [
            f"{slot}: carries no control_arm record. The control's KL needs "
            "the same evidence the candidate's does: it is a second served "
            "checkpoint, not a number."
        ]
    problems.extend(
        f"{slot}: control arm: {item.split(': ', 1)[-1]}"
        for item in verify_gold_record(
            "gold.kl", arm, model_dir=None, require_current_artifact_path=False,
        )
    )
    spec = arm.get("spec_decode_detected")
    if spec is None:
        problems.append(
            f"{slot}: control arm: spec_decode_detected is unknown — a gold "
            "number measured against a spec-decode serve is the draft model's "
            "NLL (§7.5)"
        )
    elif spec:
        problems.append(
            f"{slot}: control arm: spec_decode_detected is TRUE — that is "
            "draft-model NLL, not the control's"
        )

    arm_metrics = arm.get("metrics")
    arm_metrics = arm_metrics if isinstance(arm_metrics, Mapping) else {}
    arm_value = _finite_float(arm_metrics.get(key))
    control_kl = _finite_float(verdict.get("control"))
    if arm_value is None:
        problems.append(
            f"{slot}: control arm: carries no finite {key}, the metric this "
            "verdict is stated on"
        )
    elif control_kl is None or not _close(arm_value, control_kl):
        problems.append(
            f"{slot}: the verdict's control {verdict.get('control')!r} is not "
            f"the control arm's own {key}={arm_value!r}"
        )

    arm_sha = arm.get("model_sha")
    if not isinstance(arm_sha, str) or re.fullmatch(
        r"[0-9a-f]{64}", arm_sha
    ) is None:
        problems.append(
            f"{slot}: control arm: model_sha is not one lowercase SHA-256 — "
            "the control is a checkpoint that was built and served, and it is "
            "identified like one"
        )
    elif arm_sha == card.get("model_sha"):
        problems.append(
            f"{slot}: control arm: model_sha equals the candidate's — an arm "
            "compared against itself is not a control"
        )

    gold_tool = gold.get("tool")
    if arm.get("tool") != gold_tool:
        problems.append(
            f"{slot}: control arm: measured by {arm.get('tool')!r} but the "
            f"candidate by {gold_tool!r} — two evaluators are two metrics"
        )
    for contract_key in UNIFORM_CONTROL_CONTRACT_KEYS:
        if contract_key not in gold_metrics:
            continue
        want = gold_metrics.get(contract_key)
        got = arm_metrics.get(contract_key)
        same = got == want
        if contract_key in UNIFORM_CONTROL_EXACT_CONTRACT_KEYS:
            from .digests import DIRECT_ASCII_STRICT

            try:
                same = (contract_key in arm_metrics
                        and DIRECT_ASCII_STRICT.text(got) == DIRECT_ASCII_STRICT.text(want))
            except (TypeError, ValueError):
                same = False
        if not same:
            if contract_key in UNIFORM_CONTROL_EXACT_CONTRACT_KEYS:
                problems.append(
                    f"{slot}: control arm: missing or different {contract_key}; "
                    "the arms did not run the same measurement contract")
                continue
            problems.append(
                f"{slot}: control arm: {contract_key}={got!r} but the "
                f"candidate's gold.kl says {want!r} — the arms did not run "
                "the same measurement contract"
            )
    return problems


def _verify_uniform_control_override(
    slot: str,
    override: Any,
    *,
    card: Mapping[str, Any],
    model_dir: str | os.PathLike | None,
    ratio: float | None,
) -> tuple[bool, list[str]]:
    """Is this override a deliberate act, bound to this artifact and verdict?

    Same bar as ``publish_artifact.py --force-unverified``: the artifact
    directory's basename has to have been re-typed, and the act is stamped
    into the bytes that get uploaded.  Two bindings on top of it, because this
    override outlives the command that made it: it names the artifact's
    ``model_sha`` and the exact ratio it forgives, so a re-export or a
    re-measurement voids it rather than inheriting it.
    """
    if not isinstance(override, Mapping):
        return False, [
            f"{slot}: override is not an object ({type(override).__name__})"
        ]
    problems: list[str] = []
    if override.get("schema") != UNIFORM_CONTROL_OVERRIDE_SCHEMA:
        return False, [
            f"{slot}: override schema {override.get('schema')!r} != "
            f"{UNIFORM_CONTROL_OVERRIDE_SCHEMA!r}"
        ]
    for key in ("reason", "authorized_by", "stamped_at"):
        value = override.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{slot}: override {key} is empty")
    sha = override.get("model_sha")
    if sha != card.get("model_sha"):
        problems.append(
            f"{slot}: override names model_sha {str(sha)[:12]} but this card "
            f"is {str(card.get('model_sha'))[:12]} — an override does not "
            "survive a re-export"
        )
    typed = override.get("confirmed_artifact_name")
    if not isinstance(typed, str) or not typed.strip():
        problems.append(
            f"{slot}: override records no re-typed artifact directory name"
        )
    # The re-typed name is checked against the directory ONCE, when it is
    # typed (``shipcard_cli override-control``), exactly as ``--force-unverified``
    # checks its re-type at publish time.  It is not re-checked here: ``verify``
    # runs on the publisher's snapshot copy under a randomised basename and on
    # downloaded artifacts wherever they land, and a stamp that stops verifying
    # when the directory moves is a stamp bound to a path, not to the card.
    # What binds the override to THIS card is below: model_sha and the ratio.
    forgiven = _finite_float(override.get("candidate_over_control"))
    if forgiven is None or ratio is None or not _close(forgiven, ratio):
        problems.append(
            f"{slot}: override forgives a "
            f"{override.get('candidate_over_control')!r}x loss but the record "
            f"states {ratio!r}x — an override does not survive a "
            "re-measurement"
        )
    return (not problems), problems


def _verify_uniform_control_record(
    slot: str,
    record: Mapping[str, Any],
    *,
    card: Mapping[str, Any],
    model_dir: str | os.PathLike | None = None,
) -> list[str]:
    """Refuse an allocation that lost to spending the same bytes uniformly."""
    problems: list[str] = []
    block = record.get("uniform_control")
    schemas = uniform_control_schemas()
    if not isinstance(block, Mapping):
        return [
            f"{slot}: carries no uniform_control block "
            f"(the rate-axis producer's control block, schema "
            f"{_schema_names(schemas)}), so there is nothing to replay"
        ]
    if block.get("schema") not in schemas:
        return [
            f"{slot}: block schema {block.get('schema')!r} != "
            f"{_schema_names(schemas)}"
        ]

    control = block.get("control")
    control = control if isinstance(control, Mapping) else {}
    byte_problems, byte_matched = _replay_byte_match(slot, control.get("match"))
    problems.extend(byte_problems)
    dominated = control.get("dominated_by")
    if dominated is not None:
        problems.append(
            f"{slot}: the control sits on R{control.get('q256')} while "
            f"R{dominated} weighs no more on this plan's shapes — a "
            "handicapped uniform arm. Beating a rung the allocator should not "
            "have been offered either is not beating the control "
            "(tessera#43)."
        )

    verdict = block.get("verdict")
    if not isinstance(verdict, Mapping):
        return problems + [f"{slot}: the block carries no verdict"]
    if verdict.get("metric") != "kl_vs_bf16":
        problems.append(
            f"{slot}: verdict metric {verdict.get('metric')!r} — this gate is "
            "on the serving metric, exact full-vocab vLLM KL-vs-BF16 "
            "('kl_vs_bf16'), and on nothing else"
        )
    if verdict.get("measured") is not True:
        # Deliberately NOT overridable, and deliberately not silence: this is
        # the `{"measured": false}` variant `control_block()` emits for a
        # control that was built and priced but never served.  A built control
        # is not a passed gate.
        return problems + [
            f"{slot}: the uniform control was built and priced but never "
            f"SERVED ({verdict.get('detail') or 'no detail'}). Missing is not "
            "passing: serve the control checkpoint on the same corpus that "
            "produced this card's gold.kl and fill the verdict, or the "
            "allocation's claim to be worth allocating is untested."
        ]

    candidate = _finite_float(verdict.get("candidate"))
    control_kl = _finite_float(verdict.get("control"))
    if candidate is None or candidate < 0 or control_kl is None or (
        control_kl <= 0
    ):
        return problems + [
            f"{slot}: verdict needs a finite non-negative candidate KL and a "
            f"finite positive control KL; got {verdict.get('candidate')!r} and "
            f"{verdict.get('control')!r}"
        ]
    ratio = candidate / control_kl
    beat = candidate < control_kl
    reported_ratio = _finite_float(verdict.get("candidate_over_control"))
    if reported_ratio is None or not _close(reported_ratio, ratio):
        problems.append(
            f"{slot}: candidate_over_control="
            f"{verdict.get('candidate_over_control')!r} but "
            f"{candidate!r} / {control_kl!r} = {ratio!r}"
        )
    if verdict.get("beat_control") is not None and bool(
        verdict.get("beat_control")
    ) is not beat:
        problems.append(
            f"{slot}: the block claims beat_control="
            f"{verdict.get('beat_control')!r} but its own two KLs replay to "
            f"{beat}"
        )
    if record.get("passed") is not beat:
        # The flag is not the gate -- the two KLs are -- but a record whose
        # flag disagrees with its own numbers is lying, and saying so is
        # cheaper than letting a hand-edited `passed: true` look ordinary.
        problems.append(
            f"{slot}: record says passed={record.get('passed')!r} while its "
            f"own KLs say the candidate {'beat' if beat else 'did NOT beat'} "
            "the control"
        )

    problems.extend(_replay_control_arms(slot, record, verdict, card=card))

    if beat:
        return problems

    losing = (
        f"{slot}: the allocation LOST to its byte-matched uniform control — "
        f"{candidate:.6g} against {control_kl:.6g} KL-vs-BF16 at matched "
        f"bytes ({ratio:.4g}x worse) on {control.get('grid')} "
        f"R{control.get('q256')}. An allocation that cannot beat spending the "
        "same bytes at one rung has not earned its allocation and does not "
        "ship (prismaquant#117, tessera#1)."
    )
    override = record.get("override")
    if override is None:
        return problems + [losing]
    if problems or not byte_matched:
        # An override forgives a MEASURED loss on a REAL control.  It does not
        # forgive a control that was not byte-matched or a record that does
        # not replay -- those are not results to accept, they are results that
        # do not exist yet.  `--force-unverified` is the (stamped) instrument
        # for those, and it is meant to feel heavier than this.
        return problems + [
            losing + " The override on this record does not apply: an "
            "override forgives a measured loss on a byte-matched control, and "
            "this record has unresolved problems above."
        ]
    ok, override_problems = _verify_uniform_control_override(
        slot, override, card=card, model_dir=model_dir, ratio=ratio,
    )
    if not ok:
        return problems + override_problems + [losing]
    return problems


def make_uniform_control_record(
    *,
    tool: str,
    model_sha: str | None,
    control_block: Mapping[str, Any],
    control_arm: Mapping[str, Any],
    gold_metric_key: str,
    git_commit: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    """One byte-matched-uniform-control verdict block.

    ``control_block`` is stored verbatim -- it is Tessera's accountant's
    output and this module does not import Tessera -- and every number in it
    is replayed by :func:`verify`.  ``control_arm`` is the gold-shaped record
    of the CONTROL checkpoint's own served KL, produced by the same tool that
    filled ``gold.kl`` on this artifact.
    """
    if gold_metric_key not in UNIFORM_CONTROL_METRIC_KEYS:
        raise ValueError(
            f"gold_metric_key must be one of {list(UNIFORM_CONTROL_METRIC_KEYS)}"
        )
    block = json.loads(json.dumps(dict(control_block)))
    schemas = uniform_control_schemas()
    if block.get("schema") not in schemas:
        raise ValueError(
            f"control block schema {block.get('schema')!r} != "
            f"{_schema_names(schemas)}"
        )
    verdict = block.get("verdict") or {}
    beat = bool(verdict.get("measured")) and bool(verdict.get("beat_control"))
    if detail is None:
        detail = str(
            verdict.get("detail")
            or "the control was built and priced; neither arm was served"
        )
    return make_record(
        slot=UNIFORM_CONTROL_SLOT,
        tool=tool,
        passed=beat,
        model_sha=model_sha,
        metrics={"gold_metric_key": gold_metric_key},
        detail=detail,
        git_commit=git_commit,
        extra={
            "uniform_control": block,
            "control_arm": json.loads(json.dumps(dict(control_arm))),
        },
    )


def make_route_sweep_record(
    *,
    tool: str,
    model_sha: str | None,
    sweeps: Sequence[tuple[str, Any]],
    expected_ranks: int,
    config_json: str,
    serve_fingerprint: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Close `route.sweep` from every rank's served route sweep (#631).

    Raises ``RouteSweepNotVerified`` when no usable observation exists and
    ``CompressedRouteSweepError`` when the observation disagrees with the
    price.  Neither produces a record, so the slot stays unfilled and
    publication refuses; only an agreeing verdict is written, and `verify`
    replays it from the carried bytes.
    """
    from copy import deepcopy

    from prismaquant import compressed_route_sweep_gate as gate

    try:
        config = json.loads(config_json)
    except ValueError as exc:
        raise gate.CompressedRouteSweepError(
            f"config.json is not JSON: {exc}") from exc
    verdict = gate.compare_route_sweeps(
        list(sweeps), expected_ranks=expected_ranks, config=config)
    if verdict["status"] == gate.NOT_VERIFIED:
        raise gate.RouteSweepNotVerified(verdict["detail"])
    if verdict["status"] != gate.AGREE:
        raise gate.CompressedRouteSweepError(verdict["detail"])
    carried = []
    for label, payload in sweeps:
        if isinstance(payload, (str, bytes)):
            payload = json.loads(payload)
        carried.append({"rank": label, "sweep": deepcopy(payload)})
    return make_record(
        slot=ROUTE_SWEEP_SLOT,
        tool=tool,
        passed=True,
        model_sha=model_sha,
        metrics={
            "n_sweeps": len(carried),
            "n_priced_modules": sum(verdict["histogram"].values()),
            "activation_contracts": verdict["histogram"],
        },
        detail=verdict["detail"],
        serve_fingerprint=serve_fingerprint,
        git_commit=git_commit,
        extra={
            "route_sweeps": carried,
            "expected_ranks": expected_ranks,
            "config_json": config_json,
            "sweep_verdict": verdict,
        },
    )


def _verify_route_sweep_record(
    slot: str,
    record: Mapping[str, Any],
    *,
    card: Mapping[str, Any] | None = None,
    model_dir: str | os.PathLike | None = None,
) -> list[str]:
    """Replay the served-vs-priced module comparison from the carried sweeps."""
    from prismaquant import compressed_route_sweep_gate as gate

    sweeps = record.get("route_sweeps")
    config_json = record.get("config_json")
    expected = record.get("expected_ranks")
    problems: list[str] = []
    if not isinstance(sweeps, list) or not sweeps or not all(
            isinstance(row, Mapping) and isinstance(row.get("rank"), str)
            for row in sweeps):
        problems.append(
            f"{slot}: record carries no route_sweeps; an absent observation is "
            "not a clean bill")
    if not isinstance(config_json, str) or not config_json:
        problems.append(f"{slot}: record carries no config_json to price against")
    if type(expected) is not int:
        problems.append(f"{slot}: record carries no expected_ranks")
    if problems:
        return problems
    if model_dir is not None:
        try:
            on_disk = (Path(model_dir) / "config.json").read_bytes().decode("utf-8")
        except (OSError, ValueError) as exc:
            return [f"{slot}: cannot read the artifact's config.json: {exc}"]
        if on_disk != config_json:
            problems.append(
                f"{slot}: carried config_json differs from the artifact's "
                "config.json; the sweeps were compared against another price")
    try:
        verdict = gate.compare_route_sweeps(
            [(row["rank"], json.dumps(row.get("sweep"))) for row in sweeps],
            expected_ranks=expected, config=json.loads(config_json))
    except (gate.CompressedRouteSweepError, ValueError) as exc:
        return problems + [f"{slot}: REFUSED on replay: {exc}"]
    if verdict["status"] != gate.AGREE:
        problems.append(f"{slot}: {verdict['detail']}")
    if record.get("passed") is not True:
        problems.append(f"{slot}: record carries passed={record.get('passed')!r}")
    if record.get("sweep_verdict") != verdict:
        problems.append(
            f"{slot}: carried sweep_verdict differs from the replay against "
            "the carried sweeps and the artifact's own config.json")
    return problems


def _gold_replay(slot, record, *, card=None, model_dir=None):
    problems = verify_gold_record(slot, record, model_dir=model_dir,
                                  require_current_artifact_path=False)
    spec = record.get("spec_decode_detected")
    if spec is None:
        problems.append(f"{slot}: spec_decode_detected is unknown — a gold number "
                        "measured against a spec-decode serve is the draft model's "
                        "NLL, so 'unknown' is not acceptable (§7.5)")
    elif spec:
        problems.append(f"{slot}: spec_decode_detected is TRUE — this is draft-model "
                        "NLL, not the artifact's; re-measure on a no-spec serve")
    return problems


def _native_replay(slot, record, **context):
    return _verify_native_export_record(slot, record)


def _ship_gate_replay(slot, record, **context):
    return _verify_ship_gate_record(slot, record)


@dataclass(frozen=True)
class ShipSlot:
    name: str
    policy: Callable
    owns_pass: bool = False

    def make(self, *, tool, passed, model_sha, metrics=None, detail="",
             spec_decode_detected=None, serve_fingerprint=None, git_commit=None, extra=None):
        record = {"slot": self.name, "tool": tool,
                  "filled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                  "passed": bool(passed), "model_sha": model_sha,
                  "spec_decode_detected": spec_decode_detected,
                  "serve_fingerprint": serve_fingerprint, "git_commit": git_commit,
                  "detail": detail, "metrics": dict(metrics or {})}
        if extra:
            record.update(dict(extra))
        return record

    def replay(self, record, *, card=None, model_dir=None):
        problems = []
        if not self.owns_pass and record.get("passed") is not True:
            problems.append(f"{self.name}: FAILED — {record.get('detail') or 'no detail'}")
        return problems + self.policy(self.name, record, card=card, model_dir=model_dir)

    def preflight(self, record, *, card=None, model_dir=None):
        return self.replay(record, card=card, model_dir=model_dir)


_BASE_SLOTS = {
    **{name: ShipSlot(name, _native_replay) for name in REQUIRED_SLOTS if name.startswith("native_export.")},
    **{name: ShipSlot(name, _gold_replay) for name in GOLD_SLOTS},
    "ship_gate": ShipSlot("ship_gate", _ship_gate_replay),
    UNIFORM_CONTROL_SLOT: ShipSlot(UNIFORM_CONTROL_SLOT, _verify_uniform_control_record,
                                   owns_pass=True),
}

#: The replays core itself owns for lane-declared slots, keyed by slot name.
#: ``route.sweep`` is here because the compressed-tensors lane is data only
#: and its sweep gate is a core module.  A lane with a plugin registers its
#: slots' replays there instead (``shipcard_slot_verifiers``, decoupling step
#: 6, PQ #1553), and :func:`lane_slot_verifiers` is the merged view every
#: check reads.  The slot's vocabulary membership is derived from the lane's
#: own spec (:func:`lane_scoped_slots`); a derived slot with no replay in the
#: merged view is REFUSED wherever it is declared, filled, or verified
#: (RobTand/prismaquant#162).
#:
#: Core entries replay after the generic ``passed`` check. A plugin's replays
#: own the slot's whole verdict (``owns_pass``), so :meth:`ShipSlot.replay`
#: skips that check for them.
CORE_LANE_SLOTS = {ROUTE_SWEEP_SLOT: ShipSlot(ROUTE_SWEEP_SLOT, _verify_route_sweep_record)}


def ship_slot_registry():
    from .lane_spec import lane_hooks
    found = {**_BASE_SLOTS, **CORE_LANE_SLOTS}
    owners = {name: "shipcard" for name in found}
    for lane, hook in lane_hooks("shipcard_slot_verifiers"):
        for name, replay in dict(hook()).items():
            if name in owners:
                raise KeyError(f"shipcard slot {name!r} has two replays, from "
                               f"{owners[name]!r} and lane {lane!r}; one slot has one verifier")
            owners[name] = lane
            found[name] = ShipSlot(name, replay, owns_pass=True)
    return found


def ship_slot(slot):
    registry = ship_slot_registry()
    try:
        return registry[slot]
    except KeyError:
        raise KeyError(f"unknown shipcard slot {slot!r}; known: {sorted(registry)}") from None


def make_record(*, slot, **fields):
    return ship_slot(slot).make(**fields)


class GoldSpecDecodeRefusal(ValueError):
    """A gold number whose serve did not report ``spec_decode_detected: false``."""


@dataclass(frozen=True)
class GoldRecordDraft:
    record: dict
    primary: object
    structurally_valid: bool
    problems: tuple[str, ...]


def make_gold_record(payload, *, slot, model_sha, record_path, model_dir=None,
                     tool=None, detail=None, passed=None, allow_spec_decode=False):
    """Prepare a gold payload for CLI or configured-gate attachment.

    Under ``--speculative-config`` vLLM routes echo+logprobs through the draft
    model, so a gold number is the artifact's only when its serve reported
    ``spec_decode_detected: false`` (§7.5). True, unknown or malformed is
    refused here unless the caller explicitly records it anyway
    (``allow_spec_decode``); publication replay still refuses such a record.
    """
    spec = payload.get("spec_decode_detected")
    if spec is not False and not allow_spec_decode:
        raise GoldSpecDecodeRefusal(
            f"{slot}: {record_path} reports spec_decode_detected={spec!r}. A gold "
            "number measured against a spec-decode serve is the DRAFT model's "
            "NLL, not the artifact's (§7.5). Re-measure on a no-spec serve.")

    def finite(value):
        try:
            return math.isfinite(float(value))
        except Exception:
            return False

    primary = next((payload[key] for key in GOLD_PRIMARY_METRICS[slot]
                    if key in payload and finite(payload[key])), None)
    if slot == "gold.kl":
        count_ok = any(isinstance(payload.get(key), int)
                       and not isinstance(payload.get(key), bool)
                       and payload.get(key, 0) > 0 for key in ("n_positions", "n_samples"))
    else:
        count_ok = (isinstance(payload.get("n_tokens_scored"), int)
                    and not isinstance(payload.get("n_tokens_scored"), bool)
                    and payload.get("n_tokens_scored", 0) > 0)
    fingerprint = payload.get("serve_fingerprint")
    commit = payload.get("git_commit") or (payload.get("git") or {}).get("commit")
    valid = (primary is not None and float(primary) >= 0 and count_ok
             and isinstance(fingerprint, str) and len(fingerprint) == 64
             and all(c in "0123456789abcdef" for c in fingerprint)
             and isinstance(commit, str) and len(commit) in {40, 64}
             and all(c in "0123456789abcdef" for c in commit))
    record = ship_slot(slot).make(
        tool=tool or f"record:{Path(record_path).name}",
        passed=valid if passed is None else passed, model_sha=model_sha,
        metrics={key: payload[key] for key in CARRIED_METRIC_KEYS if key in payload},
        detail=detail or f"from {record_path}",
        spec_decode_detected=payload.get("spec_decode_detected"),
        serve_fingerprint=fingerprint, git_commit=commit,
        extra={"record_path": str(Path(record_path).resolve()),
               "measured_model": payload.get("model")})
    problems = verify_gold_record(slot, record, model_dir=model_dir,
                                  require_current_artifact_path=True) if valid else []
    return GoldRecordDraft(record, primary, valid, tuple(problems))


def lane_slot_verifiers():
    return {name: adapter.policy for name, adapter in ship_slot_registry().items()
            if name not in _BASE_SLOTS}


def lane_plugin_slot_verifiers():
    return {name: replay for name, replay in lane_slot_verifiers().items() if name not in CORE_LANE_SLOTS}
