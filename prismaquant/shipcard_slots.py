"""One make, preflight and replay contract for each ship-record slot."""
from __future__ import annotations
from dataclasses import dataclass
import math
import os
import re
import time
from typing import Any, Callable, Mapping

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

REQUIRED_SLOTS = ("native_export.eager", "native_export.graph", "ship_gate", "gold.kl", "gold.ppl")
GOLD_SLOTS = frozenset({"gold.kl", "gold.ppl"})
OPTIONAL_SLOTS = ("uniform_control",)
PRIMARY_METRIC_KEYS = ("kl_confident_mean", "kl_mean", "ppl", "mean_nll", "last_token_kl")
CARRIED_METRIC_KEYS = PRIMARY_METRIC_KEYS + (
    "kl_p99", "kl_max", "n_positions", "n_confident", "n_samples", "seqlen",
    "n_tokens_scored", "per_chunk_mean_nll", "max_chunk_mean_nll", "quantization",
    "score_positions", "serve_manifest", "teacher_evidence", "mode", "prompt_top_k",
    "vocab_size", "split", "n_tokens_requested", "calibration_contract", "calibration_contract_sha256")
GOLD_PRIMARY_METRICS = {"gold.kl": ("kl_confident_mean", "kl_mean"),
                        "gold.ppl": ("ppl", "mean_nll")}

def _verify_gold_record(
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


def _gold_replay(slot, record, *, card=None, model_dir=None):
    problems = _verify_gold_record(slot, record, model_dir=model_dir,
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


def _control_replay(slot, record, **context):
    from .shipcard import _verify_uniform_control_record
    return _verify_uniform_control_record(slot, record, **context)


def _sweep_replay(slot, record, **context):
    from .shipcard import _verify_route_sweep_record
    return _verify_route_sweep_record(slot, record)


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
    "uniform_control": ShipSlot("uniform_control", _control_replay, owns_pass=True),
}
CORE_LANE_SLOTS = {"route.sweep": ShipSlot("route.sweep", _sweep_replay)}


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


@dataclass(frozen=True)
class GoldRecordDraft:
    record: dict
    primary: object
    structurally_valid: bool
    problems: tuple[str, ...]


def make_gold_record(payload, *, slot, model_sha, record_path, model_dir=None,
                     tool=None, detail=None, passed=None):
    """Prepare a gold payload for CLI or configured-gate attachment."""
    from pathlib import Path

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
    problems = _verify_gold_record(slot, record, model_dir=model_dir,
                                   require_current_artifact_path=True) if valid else []
    return GoldRecordDraft(record, primary, valid, tuple(problems))


def lane_slot_verifiers():
    return {name: adapter.policy for name, adapter in ship_slot_registry().items()
            if name not in _BASE_SLOTS}


def lane_plugin_slot_verifiers():
    return {name: replay for name, replay in lane_slot_verifiers().items() if name not in CORE_LANE_SLOTS}
