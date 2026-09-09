#!/usr/bin/env python3
"""Experimental stock-vLLM full-vocabulary KL on the sealed TR3 final panel.

Run a one-window native hook qualification first, then replay its exact
candidate/runtime/topology binding for the whole panel. No core patches.
"""
from __future__ import annotations

import argparse
import copy
from functools import partial
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from prismaquant_source_bootstrap import activate_prismaquant_source
activate_prismaquant_source()

import numpy as np
import torch

from experiments.glm_tr3_full_vocab import (
    CONTEXT_LENGTH, LOGITS_LAYOUTS, PANEL_SHA256, TOKENIZER_SHA256, VOCAB_SIZE,
    PromptLogitsCapture, bound_json, cached_checkpoint_identity, collect_tp_result, load_panel, sha256, summarize_panel,
)
from experiments.build_glm_tr3_teacher import producer_identity
from tools.full_kl_teacher_payload import atomic_json_write, canonical_sha256, tokenizer_identity
from tools.gold_engine_options import add_gold_engine_arguments, gold_engine_kwargs
from tools.serve_fingerprint import self_manifest
from tools.spec_decode_guard import refuse_if_spec_decode


def load_teacher(path, digest, panel):
    value = bound_json(path, digest)
    if (value.get("schema") != "prismaquant.glm_tr3_raw_teacher/1"
            or value.get("panel_sha256") != PANEL_SHA256
            or value.get("tokenizer_json_sha256") != TOKENIZER_SHA256
            or value.get("vocab_size") != VOCAB_SIZE
            or value.get("context_length") != CONTEXT_LENGTH
            or value.get("raw_logits_dtype") != "float32"
            or value.get("score_positions") != "causal-0-through-2046"
            or len(value.get("windows", [])) != len(panel["windows"])):
        raise ValueError("teacher does not bind the exact raw-logits final panel")
    for row, window in zip(value["windows"], panel["windows"]):
        if (row["window_id"] != window["window_id"] or row["tokens_sha256"] != window["tokens_sha256"]
                or row["shape"] != [CONTEXT_LENGTH - 1, VOCAB_SIZE] or row["dtype"] != "float32"
                or row["path"] != window["window_id"] + ".logits.npy"):
            raise ValueError("teacher window order/geometry/token binding mismatch")
    return value


def language_model_owner(model):
    """Resolve only the two published GLM5Next wrappers through their contract."""
    name = type(model).__name__
    if name == "Glm5NextForCausalLM":
        return model, "self"
    if name == "Glm5NextForConditionalGeneration":
        accessor = getattr(model, "get_language_model", None)
        owner = accessor() if callable(accessor) else None
        if (owner is not getattr(model, "language_model", None)
                or type(owner).__name__ != "Glm5NextForCausalLM"):
            raise ValueError("GLM conditional wrapper did not expose its declared language model")
        return owner, "get_language_model() == language_model"
    raise ValueError(f"unsupported native logits owner: {name}")


def attention_runtime(text_model):
    result = []
    for name, module in text_model.named_modules():
        accessor = getattr(module, "get_attn_backend", None)
        if callable(accessor):
            backend = accessor()
            if not inspect.isclass(backend):
                raise ValueError("attention backend accessor did not return its implementation class")
            cache = getattr(module, "kv_cache", None)
            result.append({"module": name, "backend": backend.__module__ + "." + backend.__qualname__,
                           "backend_source_sha256": sha256(inspect.getfile(backend)),
                           "kv_cache_dtype": str(getattr(module, "kv_cache_dtype", None)),
                           "cache_dtype": str(getattr(module, "cache_dtype", None)),
                           "allocated_kv_cache": ({"dtype": str(cache.dtype), "shape": list(cache.shape),
                                                   "device": str(cache.device)}
                                                  if isinstance(cache, torch.Tensor) else None)})
    if not result:
        raise ValueError("native GLM exposes no observable attention backends")
    return result


def install_capture(model, *, tile_rows, logits_layout="legacy_single"):
    """Public apply_model control RPC; hooks return no replacement tensor."""
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    import vllm
    import vllm.v1.worker.gpu_model_runner as gpu_runner
    text_model, owner_path = language_model_owner(model)
    processor = getattr(text_model, "logits_processor", None)
    if (not isinstance(processor, torch.nn.Module) or not hasattr(processor, "register_forward_hook")
            or getattr(processor, "scale", None) != 1. or getattr(processor, "soft_cap", None) is not None
            or getattr(processor, "vocab_size", None) != VOCAB_SIZE):
        raise ValueError("candidate lacks the unscaled full-vocabulary logits module")
    if hasattr(model, "_tr3_capture"):
        raise ValueError("candidate already has a TR3 capture hook")
    state = PromptLogitsCapture(rank=get_tensor_model_parallel_rank(),
                               world_size=get_tensor_model_parallel_world_size(),
                               rows=CONTEXT_LENGTH - 1, vocab_size=VOCAB_SIZE,
                               tile_rows=tile_rows, logits_layout=logits_layout)
    model._tr3_capture = state
    model._tr3_capture_handle = processor.register_forward_hook(state)
    return {"rank": state.rank, "world_size": state.world_size, "torch": torch.__version__,
            "vllm": vllm.__version__, "model_class": type(model).__qualname__,
            "text_model_class": type(text_model).__qualname__, "logits_owner": owner_path,
            "logits_module_class": type(processor).__qualname__,
            "attention_runtime": attention_runtime(text_model),
            "source_files": {"model": sha256(inspect.getfile(type(model))),
                             "text_model": sha256(inspect.getfile(type(text_model))),
                             "logits_module": sha256(inspect.getfile(type(processor))),
                             "gpu_model_runner": sha256(inspect.getfile(gpu_runner))}}


def arm_capture(model, *, index, window_id, descriptor, teacher_root, target_ids):
    state = model._tr3_capture
    teacher = targets = None
    if state.rank == 0:
        path = Path(teacher_root) / descriptor["path"]
        raw = path.read_bytes()
        if len(raw) != descriptor["bytes"] or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
            raise ValueError("teacher window bytes changed before GPU preload")
        a = np.load(io.BytesIO(raw), allow_pickle=False)
        if a.dtype != np.float32 or list(a.shape) != descriptor["shape"]:
            raise ValueError("teacher array geometry/dtype mismatch")
        teacher = torch.from_numpy(a).to("cuda")
        targets = torch.tensor(target_ids, dtype=torch.long, device="cuda")
        torch.cuda.synchronize()
    state.arm(index, window_id, teacher, targets)
    return {"rank": state.rank, "window_id": window_id, "teacher_resident": state.rank == 0}


def finish_capture(model, *, window_id):
    return model._tr3_capture.finish(window_id)


def remove_capture(model):
    model._tr3_capture_handle.remove()
    del model._tr3_capture_handle, model._tr3_capture
    return True


def observed_engine_configuration(llm, *, expected_kv_cache_dtype, requested_kv_cache_dtype=None):
    config = getattr(llm.llm_engine, "vllm_config", None)
    # Worker-local MLA construction can promote the cache without mutating the
    # coordinator's copied configuration. Keep the requested and resolved
    # states explicit; workers must independently prove the resolved state.
    actual = getattr(getattr(config, "cache_config", None), "cache_dtype", None)
    allowed = {expected_kv_cache_dtype}
    if requested_kv_cache_dtype is not None:
        allowed.add(requested_kv_cache_dtype)
    if actual not in allowed:
        raise ValueError("native coordinator cache_config is neither requested nor declared resolved dtype")
    return observed_configuration(config, expected_kv_cache_dtype=actual)


def observed_configuration(config, *, expected_kv_cache_dtype):
    if config is None:
        raise ValueError("cannot observe the native engine configuration")
    required = {"model_config": {"enforce_eager": True, "max_model_len": CONTEXT_LENGTH + 1,
                                 "logprobs_mode": "raw_logprobs"},
                "cache_config": {"enable_prefix_caching": False, "cache_dtype": expected_kv_cache_dtype},
                "scheduler_config": {"enable_chunked_prefill": False, "max_num_seqs": 1,
                                     "max_num_batched_tokens": CONTEXT_LENGTH + 1},
                "parallel_config": {"pipeline_parallel_size": 1, "data_parallel_size": 1}}
    observed = {}
    for group, fields in required.items():
        owner = getattr(config, group, None)
        observed[group] = {key: getattr(owner, key, None) for key in fields}
        if observed[group] != fields:
            raise ValueError(f"native engine configuration differs from isolated prompt contract: {group}")
    if getattr(config, "speculative_config", None) is not None:
        raise ValueError("native engine has speculative decoding configured")
    if getattr(getattr(config.model_config, "multimodal_config", None), "language_model_only", None) is not True:
        raise ValueError("native engine did not observe the explicit language-model-only contract")
    observed["language_model_only"] = True
    return observed


def observed_worker_configuration(worker, *, expected_kv_cache_dtype, logits_layout="legacy_single"):
    """Public control RPC observes worker-local promotion after model loading."""
    config = observed_configuration(getattr(worker, "vllm_config", None),
                                    expected_kv_cache_dtype=expected_kv_cache_dtype)
    runner = getattr(worker, "model_runner", None)
    worker_dtype = getattr(getattr(worker, "cache_config", None), "cache_dtype", None)
    runner_dtype = getattr(getattr(runner, "cache_config", None), "cache_dtype", None)
    storage_dtype = getattr(runner, "kv_cache_dtype", None)
    if (worker_dtype != expected_kv_cache_dtype or runner_dtype != expected_kv_cache_dtype
            or not isinstance(storage_dtype, torch.dtype)):
        raise ValueError("native worker/runner KV cache differs from the declared observed dtype")
    state = getattr(getattr(runner, "model", None), "_tr3_capture", None)
    if state is None:
        raise ValueError("native worker configuration has no installed capture owner")
    layout_source = None
    if logits_layout == "vllm_v2_chunk1024":
        prompt_worker = getattr(runner, "prompt_logprobs_worker", None)
        if (type(runner).__module__ != "vllm.v1.worker.gpu.model_runner"
                or type(runner).__name__ != "GPUModelRunner"
                or type(prompt_worker).__module__ != "vllm.v1.worker.gpu.sample.prompt_logprob"
                or type(prompt_worker).__name__ != "PromptLogprobsWorker"):
            raise ValueError("declared V2 prompt layout requires the native V2 prompt worker")
        layout_source = {"runner_class": type(runner).__module__ + "." + type(runner).__name__,
                         "runner_source_sha256": sha256(inspect.getfile(type(runner))),
                         "prompt_worker_class": type(prompt_worker).__module__ + "." + type(prompt_worker).__name__,
                         "prompt_worker_source_sha256": sha256(inspect.getfile(type(prompt_worker)))}
    return {"rank": state.rank, "configuration": config,
            "worker_cache_dtype": worker_dtype, "runner_cache_dtype": runner_dtype,
            # Assigned before model construction in the pinned GPU runner;
            # unlike each attention module's allocated cache, this may predate
            # sparse MLA's worker-local dtype promotion.
            "runner_initial_kv_dtype": str(storage_dtype),
            "logits_layout": logits_layout, "prompt_layout_source": layout_source}


def route_diagnostics(model, *, require_exl3):
    # Observe only an already loaded module. Importing a plugin for measurement
    # would itself change the serving process's extension residency.
    module = sys.modules.get("vllm.model_executor.layers.quantization.exl3")
    function = getattr(module, "exl3_fat_diag", None)
    if require_exl3 and not callable(function):
        raise ValueError("EXL3 worker has no loaded exl3_fat_diag observer")
    return {"rank": model._tr3_capture.rank,
            "exl3": function() if callable(function) else None,
            "exl3_source_sha256": sha256(inspect.getfile(module)) if module is not None else None}


def verify_exl3_route_delta(before, after, *, world_size):
    if (len(before) != world_size or len(after) != world_size
            or sorted(r["rank"] for r in before) != list(range(world_size))
            or sorted(r["rank"] for r in after) != list(range(world_size))):
        raise ValueError("EXL3 route receipt requires every TP rank")
    prior = {r["rank"]: r for r in before}
    for row in after:
        old = prior[row["rank"]]
        diag, initial = row["exl3"], old["exl3"]
        if (not isinstance(diag, dict) or not isinstance(initial, dict)
                or row["exl3_source_sha256"] != old["exl3_source_sha256"]
                or diag.get("tp_rank") != row["rank"] or diag.get("tp_size") != world_size
                or diag.get("prefill_layer_calls", 0) <= initial.get("prefill_layer_calls", 0)):
            raise ValueError("EXL3 route receipt has no attributable scored-prefill counter delta")


def verify_prompt_alignment(output, tokens, reports):
    if list(output.prompt_token_ids) != tokens or len(output.prompt_logprobs or []) != len(tokens):
        raise ValueError("vLLM returned different prompt token order/length")
    owner = next(row for row in reports if row["rank"] == 0)
    hook_values = owner.get("target_logprobs")
    if hook_values is None or len(hook_values) != len(tokens) - 1:
        raise ValueError("missing native prompt alignment evidence")
    emitted = []
    for index in range(1, len(tokens)):
        entry = output.prompt_logprobs[index]
        if entry is None or tokens[index] not in entry:
            raise ValueError("vLLM omitted a target prompt logprob")
        emitted.append(float(entry[tokens[index]].logprob))
    actual, expected = np.asarray(hook_values), np.asarray(emitted)
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("nonfinite native prompt alignment evidence")
    error = float(np.max(np.abs(actual - expected)))
    if error > 1e-4:
        raise ValueError(f"hook causal row alignment disagrees with vLLM prompt scores: {error}")
    return {"positions": len(emitted), "max_abs_target_logprob_error": error,
            "tolerance": 1e-4, "passed": True}


def write_runtime_observation(output, runtime_binding):
    """Retain initialized state even when later native scoring fails closed."""
    output = Path(output)
    digest = canonical_sha256(runtime_binding)
    path = output.with_name(f"{output.stem}.runtime-{digest}.json")
    atomic_json_write({"schema": "prismaquant.glm_tr3_runtime_observation/1",
                       "stage": "initialized_before_scoring", "scored_windows": 0,
                       "runtime_binding_sha256": digest, "runtime_binding": runtime_binding}, path)
    return path


def qualification_runtime_matches(qualified, observed):
    """Compare replay semantics without treating free-memory capacity as layout.

    These native backends declare ``get_kv_cache_shape(num_blocks, ...)``
    with num_blocks as dimension zero. Keep their raw allocations in both
    receipts; only this comparison ignores that positive block count. Every
    other field, including backend source, dtype and remaining dimensions,
    still compares exactly. Unknown backend layouts receive no exception.
    """
    capacity_axis_backends = {
        "vllm.v1.attention.backends.mla.indexer.DeepseekV32IndexerBackend": 3,
        "vllm.v1.attention.backends.mla.indexer.KpoolTailBackend": 4,
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse.FlashInferMLASparseSM120Backend": 3,
    }

    def semantics(binding):
        if not isinstance(binding, dict):
            raise ValueError("runtime binding must be an object")
        value = copy.deepcopy(binding)
        for worker in value["worker_runtime"]:
            for attention in worker["attention_runtime"]:
                rank = capacity_axis_backends.get(attention["backend"])
                cache = attention["allocated_kv_cache"]
                if rank is None or cache is None:
                    continue
                shape = cache["shape"]
                if (not isinstance(shape, list) or len(shape) != rank
                        or any(type(size) is not int or size <= 0 for size in shape)):
                    raise ValueError("native allocated KV shape must have positive dimensions")
                shape[0] = None
        return value

    try:
        return semantics(qualified) == semantics(observed)
    except (KeyError, TypeError, ValueError):
        return False


def measure(args):
    panel, inputs = load_panel(args.panel, arrays_root=args.arrays_root)
    teacher = load_teacher(args.teacher, args.teacher_sha256, panel)
    model = Path(args.model).resolve(strict=True)
    if sha256(model / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("candidate tokenizer.json differs from sealed reference vocabulary")
    token_identity = tokenizer_identity(model)
    if token_identity != teacher["tokenizer_identity"]:
        raise ValueError("candidate tokenizer files differ from teacher")
    candidate_identity = cached_checkpoint_identity(model, args.candidate_digest_cache)
    producer = producer_identity()
    topology = gold_engine_kwargs(args)
    kwargs = {"model": str(model), "trust_remote_code": True, "dtype": "bfloat16",
              "language_model_only": True, "kv_cache_dtype": args.kv_cache_dtype,
              "enforce_eager": True, "enable_prefix_caching": False, "enable_chunked_prefill": False,
              "max_model_len": CONTEXT_LENGTH + 1, "max_num_batched_tokens": CONTEXT_LENGTH + 1,
              "max_num_seqs": 1, "max_logprobs": 1, "disable_log_stats": True,
              "logprobs_mode": "raw_logprobs",
              "gpu_memory_utilization": args.gpu_memory_utilization, **topology}
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if not args.qualify_hook and (args.qualification is None or args.qualification_sha256 is None):
        raise ValueError("whole panel requires the matching native one-window hook qualification")
    qualification = (bound_json(args.qualification, args.qualification_sha256)
                     if not args.qualify_hook else None)
    from vllm import LLM, SamplingParams
    llm = LLM(**kwargs)
    installed = False
    try:
        if refuse_if_spec_decode(llm=llm, context="TR3 full-vocabulary") is not False:
            raise ValueError("speculative decoding must be observed disabled")
        observed_configuration = observed_engine_configuration(
            llm, expected_kv_cache_dtype=args.expected_kv_cache_dtype,
            requested_kv_cache_dtype=args.kv_cache_dtype)
        worker_runtime = llm.apply_model(partial(install_capture, tile_rows=args.tile_rows,
                                                logits_layout=args.logits_layout))
        installed = True
        worker_runtime.sort(key=lambda row: row["rank"])
        if ([row["rank"] for row in worker_runtime] != list(range(topology["tensor_parallel_size"]))
                or any(row["world_size"] != topology["tensor_parallel_size"] for row in worker_runtime)):
            raise ValueError("native hook installation lacks complete TP rank evidence")
        worker_configuration = llm.collective_rpc(
            observed_worker_configuration,
            kwargs={"expected_kv_cache_dtype": args.expected_kv_cache_dtype,
                    "logits_layout": args.logits_layout})
        worker_configuration.sort(key=lambda row: row["rank"])
        if [row["rank"] for row in worker_configuration] != list(range(topology["tensor_parallel_size"])):
            raise ValueError("native configuration observation lacks every TP worker")
        runtime_binding = {"worker_runtime": worker_runtime, "engine_kwargs": kwargs,
                           "observed_engine_configuration": observed_configuration,
                           "observed_worker_configuration": worker_configuration,
                           "serve_image": args.serve_image, "producer_identity": producer,
                           "candidate_identity": candidate_identity,
                           "teacher_sha256": args.teacher_sha256, "panel_sha256": PANEL_SHA256,
                           "logits_layout": args.logits_layout}
        diagnostics_before = llm.apply_model(partial(route_diagnostics, require_exl3=args.require_exl3_diag))
        runtime_binding["require_exl3_diag"] = args.require_exl3_diag
        observation = write_runtime_observation(args.output, runtime_binding)
        print(f"[tr3-full-kl] initialized runtime observation {observation}", flush=True)
        if qualification is not None and (
                qualification.get("schema") != "prismaquant.glm_tr3_hook_qualification/1"
                or not qualification_runtime_matches(qualification.get("runtime_binding"), runtime_binding)
                or qualification.get("passed") is not True):
            raise ValueError("native qualification differs from this candidate/runtime/teacher/topology")
        vectors, alignment, rank_calls = [], [], []
        count = 1 if args.qualify_hook else len(inputs)
        for index in range(count):
            window, row = panel["windows"][index], teacher["windows"][index]
            tokens = inputs[index][0].tolist()
            armed = llm.apply_model(partial(arm_capture, index=index, window_id=window["window_id"],
                                           descriptor=row, teacher_root=str(Path(args.teacher).resolve().parent),
                                           target_ids=tokens[1:]))
            if sum(r["teacher_resident"] for r in armed) != 1:
                raise ValueError("resident teacher must have exactly one TP owner")
            outputs = llm.generate([{"prompt_token_ids": tokens}],
                                   SamplingParams(max_tokens=1, temperature=1., top_p=1., seed=42,
                                                  prompt_logprobs=1, detokenize=False, ignore_eos=True),
                                   use_tqdm=False)
            reports = llm.apply_model(partial(finish_capture, window_id=window["window_id"]))
            if len(outputs) != 1:
                raise ValueError("native hook requires one outstanding request")
            vectors.append(collect_tp_result(reports, window_id=window["window_id"],
                                            world_size=topology["tensor_parallel_size"],
                                            rows=CONTEXT_LENGTH - 1, vocab_size=VOCAB_SIZE,
                                            logits_layout=args.logits_layout))
            alignment.append(verify_prompt_alignment(outputs[0], tokens, reports))
            rank_calls.append([{k: r[k] for k in ("rank", "world_size", "window_id", "calls", "logits_layout")} for r in reports])
            print(f"[tr3-full-kl] measured {window['window_id']}", flush=True)
        if cached_checkpoint_identity(model, args.candidate_digest_cache) != candidate_identity:
            raise ValueError("candidate checkpoint changed while scoring")
        if (tokenizer_identity(model) != token_identity or producer_identity() != producer
                or bound_json(args.teacher, args.teacher_sha256) != teacher):
            raise ValueError("teacher/tokenizer/producer changed while scoring")
        load_panel(args.panel, arrays_root=args.arrays_root)
        diagnostics_after = llm.apply_model(partial(route_diagnostics, require_exl3=args.require_exl3_diag))
        if args.require_exl3_diag:
            verify_exl3_route_delta(diagnostics_before, diagnostics_after,
                                   world_size=topology["tensor_parallel_size"])
        if observed_engine_configuration(
                llm, expected_kv_cache_dtype=args.expected_kv_cache_dtype,
                requested_kv_cache_dtype=args.kv_cache_dtype) != observed_configuration:
            raise ValueError("native engine configuration changed during scoring")
        manifest = self_manifest(image=args.serve_image,
                                 extra={"measurement_tool": "experimental_glm_tr3_full_vocabulary",
                                        "runtime_binding": runtime_binding})
        result = {"schema": ("prismaquant.glm_tr3_hook_qualification/1" if args.qualify_hook
                              else "prismaquant.glm_tr3_full_vocabulary_kl/1"),
                  "passed": True, "runtime_binding": runtime_binding, "serve_manifest": manifest,
                  "estimator": "KL(reference||candidate), raw logits normalized and summed in FP64 over full vocabulary",
                  "per_position_kl": vectors, "prompt_alignment": alignment, "rank_calls": rank_calls,
                  "route_diagnostics": {"before": diagnostics_before, "after": diagnostics_after},
                  "teacher_source_execution": teacher["source_execution"],
                  "summary": summarize_panel({"windows": panel["windows"][:count]}, vectors)}
        atomic_json_write(result, args.output)
        return result
    finally:
        if installed:
            llm.apply_model(remove_capture)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "candidate-digest-cache", "panel", "teacher", "teacher-sha256", "serve-image", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--arrays-root")
    p.add_argument("--quantization")
    p.add_argument("--kv-cache-dtype", choices=("auto", "bfloat16", "fp8_ds_mla"), required=True)
    p.add_argument("--expected-kv-cache-dtype", choices=("auto", "bfloat16", "fp8_ds_mla"), required=True,
                   help="observed native cache dtype; an automatic promotion must be declared explicitly")
    p.add_argument("--require-exl3-diag", action="store_true")
    p.add_argument("--gpu-memory-utilization", type=float, default=.9)
    p.add_argument("--tile-rows", type=int, default=32)
    p.add_argument("--logits-layout", choices=LOGITS_LAYOUTS, default="legacy_single",
                   help="explicit native logits-call layout; scheduler chunked prefill stays disabled")
    p.add_argument("--qualify-hook", action="store_true")
    p.add_argument("--qualification")
    p.add_argument("--qualification-sha256")
    add_gold_engine_arguments(p)
    args = p.parse_args()
    if args.tile_rows <= 0 or args.tile_rows > 64:
        p.error("tile rows must be 1..64")
    if "@sha256:" not in args.serve_image:
        p.error("serve image must be immutable digest-qualified")
    measure(args)


if __name__ == "__main__":
    main()
