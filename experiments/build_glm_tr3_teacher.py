#!/usr/bin/env python3
"""Emit the sealed final panel's raw BF16-reference logits through one source pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from prismaquant_source_bootstrap import activate_prismaquant_source
activate_prismaquant_source()

import numpy as np
import torch

from experiments.glm_tr3_full_vocab import (
    PANEL_SHA256, REFERENCE_REVISION, TOKENIZER_SHA256, VOCAB_SIZE,
    CONTEXT_LENGTH, bound_json, cached_checkpoint_identity, load_panel, sha256,
)
from tools.build_streamed_full_kl_teacher import (
    _source_derivative_policy, _require_source_execution_policy,
    _teacher_producer_identity,
)
from tools.full_kl_teacher_payload import atomic_json_write, canonical_sha256, tokenizer_identity


def producer_identity():
    return {"gold_source": _teacher_producer_identity(),
            "experiment_files": {name: sha256(ROOT / "experiments" / name)
                                 for name in ("glm_tr3_full_vocab.py", "build_glm_tr3_teacher.py",
                                              "measure_glm_tr3_vllm.py", "glm_full_capture_profile.py",
                                              "workspace_netdata.py")}}


def require_reference_binding(binding, identity, model):
    """The independent upstream audit supplies the immutable revision mapping."""
    if (binding.get("schema") != "root-source-revision-binding-v1"
            or binding.get("repo") != "zai-org/GLM-5.3-Flash-BF16"
            or binding.get("revision") != REFERENCE_REVISION
            or binding.get("all_matched") is not True or binding.get("safetensors_count") != 120):
        raise ValueError("original BF16 checkpoint lacks the required upstream revision binding")
    roster = binding.get("source_files", [])
    if len(roster) != 126 or len({r["name"] for r in roster}) != 126:
        raise ValueError("upstream source roster must contain every original file exactly once")
    shards = {r["name"]: r for r in identity["shards"]}
    covered = set()
    for row in roster:
        name = row["name"]
        if Path(name).name != name or row["capture_sha256"] != row["upstream_sha256"]:
            raise ValueError("upstream source filename or digest mismatch")
        path = Path(model) / name
        if name.endswith(".safetensors"):
            actual = shards.get(name)
            if actual is None or actual["sha256"] != row["upstream_sha256"] or actual["size"] != row["size_bytes"]:
                raise ValueError("current complete source identity differs from upstream shard roster")
            covered.add(name)
        elif sha256(path) != row["upstream_sha256"] or path.stat().st_size != row["size_bytes"]:
            raise ValueError("current source metadata differs from immutable upstream")
    if covered != set(shards):
        raise ValueError("upstream binding does not cover the complete source identity")


def visit_panel(runner, inputs, consumer, profiler=None):
    """One existing resident layer visit over all ordered, independent windows."""
    next_output = 0
    def visitor(_layer, forward_batch):
        for tokens in inputs:
            forward_batch(tokens)
        if profiler is not None:
            profiler.step()
    def output(index, logits):
        nonlocal next_output
        if index != next_output:
            raise ValueError("teacher output window order changed")
        consumer(index, logits)
        next_output += 1
    runner.visit_layer_batches(inputs, visitor, output_consumer=output)
    if next_output != len(inputs):
        raise ValueError("teacher traversal omitted final windows")


def build(args):
    from prismaquant.cost_streaming import build_streamed_causal_lm
    from prismaquant.gpu_guard import require_cuda_hot_path
    from prismaquant.model_profiles import detect_profile
    from prismaquant.joint_aura import source_execution_identity

    panel, inputs = load_panel(args.panel, arrays_root=args.arrays_root)
    model = Path(args.model).resolve(strict=True)
    if sha256(model / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("reference tokenizer.json differs from the sealed panel")
    identity = cached_checkpoint_identity(model, args.identity_cache)
    binding = bound_json(args.reference_binding, args.reference_binding_sha256)
    require_reference_binding(binding, identity, model)
    tokenizer = tokenizer_identity(model)
    producer = producer_identity()
    policy = _source_derivative_policy(args)
    # Explicit JSON null is required to mean original execution, avoiding an
    # implicit default whose meaning could vary with the installed source.
    if args.source_derivative_json is None:
        raise ValueError("declare the original/null or corrected source execution explicitly")
    device = require_cuda_hot_path("build_glm_tr3_teacher", "cuda")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner = build_streamed_causal_lm(
        str(model), device=device, dtype=torch.bfloat16,
        offload_folder=args.offload_folder, profile=detect_profile(str(model)),
        cache_headroom_gb=args.cache_headroom_gb, max_cache_slots=2,
        prefetch_workers=1, prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation="eager", source_derivative=policy,
    )
    rows = []
    started = time.monotonic()
    try:
        if runner.context.max_cache_slots != 2 or runner.prefetch_lookahead != 1 or not runner.require_prefetched_residency:
            raise ValueError("teacher requires the existing two-slot resident prefetch policy")
        runner.model.eval()
        runner.context.begin_source_initialization_audit()
        source_execution = source_execution_identity(runner.model)
        _require_source_execution_policy(source_execution, policy)
        resident_plan = {"estimated_layer_bytes": runner.context.estimated_layer_bytes,
                         "cache_max_bytes": runner.context.layer_cache.max_bytes,
                         "cache_slots": runner.context.max_cache_slots,
                         "prefetch_workers": runner.context.prefetch_workers,
                         "prefetch_lookahead": runner.prefetch_lookahead,
                         "prefetch_min_available_bytes": runner.context.prefetch_min_available_bytes,
                         "cuda_allocated_before_windows": torch.cuda.memory_allocated(device),
                         "cuda_reserved_before_windows": torch.cuda.memory_reserved(device),
                         "hidden_state_bf16_bytes": 25 * 2048 * 4 * 4096 * 2,
                         "hidden_state_fp32_bytes": 25 * 2048 * 4 * 4096 * 4}
        atomic_json_write(resident_plan, output / "resident-plan.json")
        def consume(index, logits):
            if tuple(logits.shape) != (1, CONTEXT_LENGTH, VOCAB_SIZE) or logits.device.type != "cuda":
                raise ValueError("teacher raw logits have the wrong causal geometry/device")
            # Retain raw logits, not FP32-normalized logprobs: upstream performs
            # normalization in FP64. The unscored final-context row is omitted.
            raw = logits[0, :-1].float()
            if not bool(torch.isfinite(raw).all()):
                raise ValueError("teacher emitted nonfinite logits")
            window = panel["windows"][index]
            filename = f"{window['window_id']}.logits.npy"
            cpu = raw.cpu().numpy()
            np.save(output / filename, cpu, allow_pickle=False)
            rows.append({"window_id": window["window_id"], "path": filename,
                         "sha256": sha256(output / filename),
                         "bytes": (output / filename).stat().st_size,
                         "shape": list(cpu.shape), "dtype": "float32",
                         "tokens_sha256": window["tokens_sha256"]})
            print(f"[tr3-teacher] wrote {window['window_id']} raw logits", flush=True)
        # Trace one layer's complete panel traversal. This records actual work
        # without retaining the whole model run's potentially unbounded events.
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                on_trace_ready=lambda p: p.export_chrome_trace(str(output / "first-layer.trace.json")),
                record_shapes=False, profile_memory=True) as profiler:
            with torch.inference_mode():
                visit_panel(runner, inputs, consume, profiler)
        initialization_contract = runner.context.source_initialization_contract()
        if cached_checkpoint_identity(model, args.identity_cache) != identity:
            raise ValueError("source checkpoint identity changed during teacher emission")
        require_reference_binding(binding, identity, model)
        if source_execution_identity(runner.model) != source_execution:
            raise ValueError("source execution changed during teacher emission")
        if (tokenizer_identity(model) != tokenizer or producer_identity() != producer
                or _source_derivative_policy(args) != policy
                or bound_json(args.reference_binding, args.reference_binding_sha256) != binding):
            raise ValueError("teacher source/tokenizer/producer binding changed")
        load_panel(args.panel, arrays_root=args.arrays_root)
    finally:
        runner.shutdown()
    result = {"schema": "prismaquant.glm_tr3_raw_teacher/1", "panel_sha256": PANEL_SHA256,
              "source_model_identity": identity, "source_model_identity_sha256": canonical_sha256(identity),
              "reference_binding": binding, "reference_binding_sha256": args.reference_binding_sha256,
              "source_execution": source_execution, "producer_identity": producer,
              "source_initialization_contract": initialization_contract,
              "resident_plan": resident_plan,
              "tokenizer_identity": tokenizer, "tokenizer_json_sha256": TOKENIZER_SHA256,
              "vocab_size": VOCAB_SIZE, "context_length": CONTEXT_LENGTH,
              "score_positions": "causal-0-through-2046", "raw_logits_dtype": "float32",
              "windows": rows, "elapsed_seconds": time.monotonic() - started,
              "profile": {"path": "first-layer.trace.json", "sha256": sha256(output / "first-layer.trace.json"),
                          "scope": "first resident layer over all 25 windows"},
              "fit_overlap_status": "unverified", "use": "sealed final benchmark; never allocator tuning"}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "identity-cache", "panel", "reference-binding", "reference-binding-sha256",
                 "source-derivative-json", "source-derivative-sha256", "output-dir", "offload-folder"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--arrays-root")
    p.add_argument("--cache-headroom-gb", type=float, default=12.)
    args = p.parse_args()
    from experiments.glm_full_capture_profile import CaptureObserver
    from experiments.workspace_netdata import sample_netdata
    telemetry = Path(args.output_dir).resolve().parent / "telemetry"
    # Reuse the existing bounded five-second both-box sampler and one-second
    # main-thread stack/IO observer. Synchronous samples bracket the entire
    # build, including source initialization; its context always stops readers.
    with CaptureObserver(telemetry, profile_layers=()) as observer:
        atomic_json_write([sample_netdata(host) for host in ("sparky", "sparklina")],
                          telemetry / "before.json")
        result = build(args)
        atomic_json_write([sample_netdata(host) for host in ("sparky", "sparklina")],
                          telemetry / "after.json")
    result["host_telemetry"] = {
        "hosts": ["sparky", "sparklina"], "status": observer.result["status"],
        "files": {name: {"path": "../telemetry/" + name, "sha256": sha256(telemetry / name)}
                  for name in ("netdata.jsonl", "python_sampler.jsonl", "result.json",
                               "before.json", "after.json")}}
    # Only this final manifest admits the complete artifact, after both source
    # and observer completion. Failed partial directories have no manifest.
    atomic_json_write(result, Path(args.output_dir) / "teacher.json")


if __name__ == "__main__":
    main()
