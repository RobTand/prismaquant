#!/usr/bin/env python3
"""Build a BF16 gold teacher with one streamed GPU model.

This is the one-Spark source-teacher path.  It deliberately extends the
repository's existing StreamingContext instead of inventing a second offload
or residency mechanism.  The complete BF16 source never has to be resident at
once; one decoder layer is installed at a time and logits are reduced to the
fixed top-K gold support (PROMPT_TOP_K) on GPU. An explicit v2 option retains
full-vocabulary final rows from that same forward; model-bound inputs and
source execution remain authenticated separately from fitting-overlap claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

_TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_ROOT))
from prismaquant_source_bootstrap import activate_prismaquant_source

activate_prismaquant_source()

try:  # package mode
    from .full_kl_teacher_payload import (
        N_SAMPLES,
        PROMPT_TOP_K,
        SEQLEN,
        TEACHER_PAYLOAD_SCHEMA,
        TEACHER_PAYLOAD_V2_SCHEMA,
        WIKITEXT_CONFIG,
        WIKITEXT_DATASET,
        WIKITEXT_REVISION,
        WIKITEXT_SPLIT,
        WINDOW_SEED,
        atomic_json_write,
        atomic_torch_save,
        build_calibration_contract,
        canonical_sha256,
        compact_source_model_identity,
        format_forward_fidelity_profile,
        file_sha256,
        payload_semantic_sha256,
        teacher_forward_fidelity_summary,
        teacher_meta,
        tokenizer_identity,
        validate_teacher_payload,
    )
except ImportError:  # direct script mode
    from full_kl_teacher_payload import (  # type: ignore
        N_SAMPLES,
        PROMPT_TOP_K,
        SEQLEN,
        TEACHER_PAYLOAD_SCHEMA,
        TEACHER_PAYLOAD_V2_SCHEMA,
        WIKITEXT_CONFIG,
        WIKITEXT_DATASET,
        WIKITEXT_REVISION,
        WIKITEXT_SPLIT,
        WINDOW_SEED,
        atomic_json_write,
        atomic_torch_save,
        build_calibration_contract,
        canonical_sha256,
        compact_source_model_identity,
        format_forward_fidelity_profile,
        file_sha256,
        payload_semantic_sha256,
        teacher_forward_fidelity_summary,
        teacher_meta,
        tokenizer_identity,
        validate_teacher_payload,
    )

try:  # package mode
    from .dsv4_wikitext_inputs import load_dsv4_wikitext_inputs
except ImportError:  # direct script mode
    from dsv4_wikitext_inputs import load_dsv4_wikitext_inputs  # type: ignore


def _tokenizer_vocab_size(tokenizer) -> int:
    """Return the output-vocabulary cardinality, including added tokens."""
    size = len(tokenizer)
    if isinstance(size, bool) or not isinstance(size, int) or size <= PROMPT_TOP_K:
        raise RuntimeError(f"invalid tokenizer vocabulary size: {size!r}")
    return size


def _final_logprobs(logits: torch.Tensor) -> torch.Tensor:
    """The next-token distribution after the entire window, from this forward."""
    if logits.ndim != 3 or min(logits.shape) <= 0:
        raise RuntimeError("teacher logits must be nonempty [batch, sequence, vocabulary]")
    return torch.log_softmax(logits[:, -1, :].float(), dim=-1).to("cpu").contiguous()


def _source_derivative_policy(args):
    path = getattr(args, "source_derivative_json", None)
    if path is None:
        return None
    from prismaquant.glm_source_derivative import bound_json, normalize_source_derivative
    return normalize_source_derivative(bound_json(
        {"path": path, "sha256": args.source_derivative_sha256}, "gold source derivative"))


def _teacher_producer_identity() -> dict:
    from serve_fingerprint import gold_producer_identity
    from prismaquant.production_weight_cache import _production_cache_source_sha256
    return {"tools": gold_producer_identity("build_streamed_full_kl_teacher"),
            "prismaquant_source_sha256": _production_cache_source_sha256()}


def _require_source_execution_policy(execution, policy) -> None:
    if policy is None:
        if execution.get("schema") != "prismaquant.joint_aura.source_execution.v1" or "source_derivative" in execution:
            raise RuntimeError("teacher original source requires an observed execution without a derivative")
    elif (execution.get("schema") != "prismaquant.joint_aura.source_execution.v2"
          or execution.get("source_derivative", {}).get("image_build_sha256") != policy["image_build"]["sha256"]):
        raise RuntimeError("teacher source derivative requires the declared observed image-build binding")


def _load_gold_inputs(args, model_path, tokenizer_attestation):
    expected = getattr(args, "wikitext_inputs_sha256", None)
    if expected is None:
        return load_dsv4_wikitext_inputs(
            args.wikitext_inputs, expected_tokenizer_identity=tokenizer_attestation)
    try:
        from .dsv4_wikitext_inputs import load_wikitext_inputs, wikitext_model_identity
    except ImportError:
        from dsv4_wikitext_inputs import load_wikitext_inputs, wikitext_model_identity
    return load_wikitext_inputs(
        args.wikitext_inputs, expected_tokenizer_identity=tokenizer_attestation,
        expected_model_identity=wikitext_model_identity(model_path), expected_sha256=expected)


def _topk_all_positions(
    logits: torch.Tensor,
    *,
    top_k: int = PROMPT_TOP_K,
    chunk_rows: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce causal logits to fp32 all-position top-K distributions on GPU."""
    if logits.ndim != 3:
        raise RuntimeError(f"teacher logits must be rank 3, got {logits.shape}")
    batch, sequence, vocab = (int(value) for value in logits.shape)
    if batch != N_SAMPLES or sequence != SEQLEN:
        raise RuntimeError(
            f"teacher logits must be [{N_SAMPLES},{SEQLEN},V], got "
            f"{list(logits.shape)}"
        )
    if top_k != PROMPT_TOP_K or vocab <= top_k:
        raise RuntimeError("teacher top-K/vocabulary differs from gold contract")
    if chunk_rows <= 0:
        raise RuntimeError("--logits-chunk-rows must be positive")

    # prompt_logprobs[position] predicts token[position] from the prefix ending
    # at position-1, so the matching HF causal row is logits[position-1].
    rows = logits[:, :-1, :].reshape(batch * (sequence - 1), vocab)
    ids_cpu = torch.empty((rows.size(0), top_k), dtype=torch.int32)
    lps_cpu = torch.empty((rows.size(0), top_k), dtype=torch.float32)
    for first in range(0, int(rows.size(0)), chunk_rows):
        last = min(first + chunk_rows, int(rows.size(0)))
        log_probs = torch.log_softmax(rows[first:last].float(), dim=-1)
        values, indices = torch.topk(
            log_probs, k=top_k, dim=-1, largest=True, sorted=True
        )
        ids_cpu[first:last].copy_(
            indices.to(device="cpu", dtype=torch.int32)
        )
        lps_cpu[first:last].copy_(
            values.to(device="cpu", dtype=torch.float32)
        )
        print(
            f"[streamed-teacher] reduced rows {last}/{rows.size(0)}",
            flush=True,
        )
        del log_probs, values, indices
    return (
        ids_cpu.reshape(batch, sequence - 1, top_k).contiguous(),
        lps_cpu.reshape(batch, sequence - 1, top_k).contiguous(),
    )


def _build_payload(args: argparse.Namespace) -> dict:
    from prismaquant.cost_streaming import (
        build_streamed_causal_lm,
        validate_cached_streamed_model_identity,
    )
    from prismaquant.gpu_guard import require_cuda_hot_path
    from prismaquant.model_profiles import detect_profile

    device = require_cuda_hot_path("build_streamed_full_kl_teacher", "cuda")
    include_final = getattr(args, "include_final_logprobs", False)
    explicit_derivative = getattr(args, "source_derivative_json", None) is not None
    v2 = include_final or explicit_derivative or getattr(args, "wikitext_inputs_sha256", None) is not None
    derivative_policy = _source_derivative_policy(args)
    producer_identity = _teacher_producer_identity() if v2 else None
    model_path = Path(args.model).resolve(strict=True)
    if not model_path.is_dir():
        raise RuntimeError(f"source model is not a directory: {model_path}")
    tokenizer_attestation = tokenizer_identity(model_path)
    # Reject an absent/tampered 156-KiB token input before walking the much
    # larger checkpoint identity or constructing any model/tokenizer runtime.
    input_sha256 = file_sha256(args.wikitext_inputs) if v2 else None
    wikitext_inputs = _load_gold_inputs(args, model_path, tokenizer_attestation)
    full_kl_inputs = wikitext_inputs["full_kl"]
    calibration = torch.tensor(
        full_kl_inputs["token_ids"], dtype=torch.long
    ).contiguous()
    starts = list(full_kl_inputs["selection"]["starts"])
    dataset_evidence = full_kl_inputs["dataset"]
    calibration_contract = build_calibration_contract(
        dataset_fingerprint=dataset_evidence["fingerprint"],
        corpus_sha256=dataset_evidence["corpus_sha256"],
        tokenizer=tokenizer_attestation,
        starts=starts,
        total_tokens=dataset_evidence["total_tokens"],
        calib_ids=calibration,
    )
    full_identity = validate_cached_streamed_model_identity(
        model_path,
        args.identity_cache,
        require_complete_checkpoint=True,
    )
    compact_identity = compact_source_model_identity(full_identity)
    # Detection bootstraps the vendored DSv4 config with Transformers.  It
    # must precede AutoTokenizer: the source is local and intentionally has no
    # remote-code Python files of its own.
    profile = detect_profile(str(model_path))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    print(
        "[streamed-teacher] "
        f"source={compact_identity['content_sha256']} "
        f"calibration={canonical_sha256(calibration_contract)} "
        f"shape={list(calibration.shape)}",
        flush=True,
    )

    runner = build_streamed_causal_lm(
        str(model_path),
        device=device,
        dtype=torch.bfloat16,
        offload_folder=str(Path(args.offload_folder).resolve()),
        profile=profile,
        cache_headroom_gb=float(args.cache_headroom_gb),
        max_cache_slots=1,
        prefetch_workers=1,
        prefetch_lookahead=0,
        **({"source_derivative": derivative_policy, "attn_implementation": "eager"}
           if explicit_derivative else {}),
    )
    try:
        if runner.context.max_cache_slots != 1 or runner.prefetch_lookahead != 0:
            raise RuntimeError(
                "streamed teacher source-cache policy is not fail-closed "
                f"(slots={runner.context.max_cache_slots}, "
                f"lookahead={runner.prefetch_lookahead})"
            )
        if v2:
            from prismaquant.joint_aura import source_execution_identity
            source_execution = source_execution_identity(runner.model)
            _require_source_execution_policy(source_execution, derivative_policy)
        with torch.inference_mode():
            output = runner(calibration.to(device, non_blocking=True))
            logits = output.logits.detach()
            vocab_size = _tokenizer_vocab_size(tokenizer)
            if int(logits.shape[-1]) != vocab_size:
                raise RuntimeError(
                    "source logits vocabulary differs from tokenizer vocabulary"
                )
            topk_ids, topk_lps = _topk_all_positions(
                logits,
                chunk_rows=int(args.logits_chunk_rows),
            )
            final_logprobs = _final_logprobs(logits) if include_final else None
            del output, logits
        if v2:
            if validate_cached_streamed_model_identity(
                    model_path, args.identity_cache, require_complete_checkpoint=True) != full_identity:
                raise RuntimeError("teacher source identity changed during the forward")
            if source_execution_identity(runner.model) != source_execution:
                raise RuntimeError("teacher source execution changed during the forward")
            if tokenizer_identity(model_path) != tokenizer_attestation:
                raise RuntimeError("teacher tokenizer identity changed during the forward")
            if _source_derivative_policy(args) != derivative_policy:
                raise RuntimeError("teacher source derivative input changed during the forward")
            if _teacher_producer_identity() != producer_identity:
                raise RuntimeError("teacher producer source changed during the forward")
            if file_sha256(args.wikitext_inputs) != input_sha256:
                raise RuntimeError("teacher WikiText input changed during the forward")
    finally:
        runner.shutdown()

    payload: dict = {
        "schema": TEACHER_PAYLOAD_V2_SCHEMA if v2 else TEACHER_PAYLOAD_SCHEMA,
        "score_positions": "all",
        "prompt_top_k": PROMPT_TOP_K,
        "topk_ids": topk_ids,
        "topk_lps": topk_lps,
        "calib_ids": calibration,
        "starts": starts,
        "model": str(model_path),
        "n_samples": N_SAMPLES,
        "seqlen": SEQLEN,
        "vocab_size": _tokenizer_vocab_size(tokenizer),
        "source_model_identity": full_identity,
        "source_model": compact_identity,
        "source_model_identity_sha256": canonical_sha256(full_identity),
        "calibration_contract": calibration_contract,
        "calibration_contract_sha256": canonical_sha256(calibration_contract),
        **({"final_logprobs": final_logprobs, "source_execution": source_execution,
            "producer_identity": producer_identity, "fit_overlap_status": "unverified",
            "wikitext_inputs_sha256": input_sha256} if v2 else {}),
    }
    payload["payload_semantic_sha256"] = payload_semantic_sha256(payload)
    validate_teacher_payload(payload)
    # Report the forward-fidelity profile on every build, refusal or not, so
    # the teacher's own teacher-forced NLL is in the log alongside the payload
    # it graded students with.  validate_teacher_payload has already run this
    # gate; a refusal propagates from there and main() never reaches the write.
    print(
        format_forward_fidelity_profile(
            teacher_forward_fidelity_summary(
                payload["topk_ids"],
                payload["topk_lps"],
                payload["calib_ids"],
                vocab_size=int(payload["vocab_size"]),
            )
        ),
        flush=True,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--identity-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-output", required=True)
    parser.add_argument("--offload-folder", required=True)
    parser.add_argument(
        "--wikitext-inputs",
        required=True,
        help=(
            "offline payload from tools/prepare_dsv4_wikitext_inputs.py; "
            "the GPU teacher environment never imports datasets"
        ),
    )
    parser.add_argument("--cache-headroom-gb", type=float, default=100.0)
    parser.add_argument("--logits-chunk-rows", type=int, default=32)
    parser.add_argument("--wikitext-inputs-sha256",
                        help="independent file SHA required for model-bound v2 WikiText input")
    parser.add_argument("--include-final-logprobs", action="store_true",
                        help="add full-vocabulary final rows from the same forward in a v2 payload")
    parser.add_argument("--source-derivative-json",
                        help="hash-bound existing GLM derivative policy, or JSON null for original; uses eager attention")
    parser.add_argument("--source-derivative-sha256")
    args = parser.parse_args()
    if (args.source_derivative_json is None) != (args.source_derivative_sha256 is None):
        parser.error("--source-derivative-json and --source-derivative-sha256 must be supplied together")

    output = Path(args.output)
    meta_output = Path(args.meta_output)
    if output.exists() or meta_output.exists():
        parser.error(
            "refusing to overwrite an existing teacher payload or metadata file"
        )
    if output.resolve() == meta_output.resolve():
        parser.error("--output and --meta-output must be distinct")

    started = time.monotonic()
    payload = _build_payload(args)
    atomic_torch_save(payload, output)
    meta = teacher_meta(
        payload_path=output,
        elapsed_s=time.monotonic() - started,
    )
    atomic_json_write(meta, meta_output)
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
