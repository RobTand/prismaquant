"""Opt-in single-window EXL3 diagnostic orchestration, never qualification.

The public worker/report driver delegates its CLI here. Native authentication,
engine construction and prompt capture remain owned by measure_glm_tr3_vllm;
this module deliberately never calls its full-panel measure().
"""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
from uuid import uuid4

from tools.exl3_grouped_down_replay import (
    DEFAULT_MODULE, DEFAULT_REPEATS, DEFAULT_REQUIRED_ROWS,
)


def parse_args(argv=None):
    """Parse a bounded diagnostic request without importing vLLM."""
    from tools.gold_engine_options import add_gold_engine_arguments, gold_engine_kwargs
    from experiments.measure_glm_tr3_vllm import (
        EXECUTION_MODES, LOGITS_LAYOUTS, declared_compilation,
    )

    parser = argparse.ArgumentParser(description=(
        "Diagnostic-only replay of the first sealed TR3 window's live EXL3 "
        "grouped-down call. Not a qualification, gold measurement or determinism test."),
        allow_abbrev=False)
    for name in ("model", "candidate-digest-cache", "panel", "teacher",
                 "teacher-sha256", "serve-image", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--arrays-root")
    parser.add_argument("--module", default=DEFAULT_MODULE,
                        help="exact named MoE module boundary (default: %(default)s)")
    parser.add_argument("--required-rows", type=int, default=DEFAULT_REQUIRED_ROWS,
                        help="only this MoE input row count can spend the one shot")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="replays after the real call, 2..16 (default: %(default)s)")
    parser.add_argument("--quantization", choices=("exl3",), default="exl3")
    for name in ("kv-cache-dtype", "expected-kv-cache-dtype"):
        parser.add_argument("--" + name, choices=("auto", "bfloat16", "fp8_ds_mla"),
                            required=True)
    parser.add_argument("--attention-backend", choices=("CUSTOM",), default=None)
    parser.add_argument("--kernel-config", help="JSON object forwarded to the native engine")
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, default="eager")
    parser.add_argument("--compilation-config", help="native scorer's explicit compiled configuration JSON")
    parser.add_argument("--gpu-memory-utilization", type=float, default=.9)
    parser.add_argument("--tile-rows", type=int, default=32)
    parser.add_argument("--logits-layout", choices=LOGITS_LAYOUTS, default="legacy_single")
    # Route evidence is mandatory here; accept the familiar explicit spelling too.
    parser.add_argument("--require-exl3-diag", action="store_true", default=True,
                        help="mandatory for this diagnostic (always enabled)")
    add_gold_engine_arguments(parser)
    args = parser.parse_args(argv)
    if not args.module.strip():
        parser.error("--module must be a nonempty exact module name")
    if not 2 <= args.repeats <= 16:
        parser.error("--repeats must be 2..16")
    if args.required_rows <= 0:
        parser.error("--required-rows must be positive")
    if not 1 <= args.tile_rows <= 64:
        parser.error("--tile-rows must be 1..64")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if "@sha256:" not in args.serve_image:
        parser.error("--serve-image must be immutable digest-qualified")
    if (len(args.teacher_sha256) != 64
            or any(c not in "0123456789abcdef" for c in args.teacher_sha256)):
        parser.error("--teacher-sha256 must be a lowercase SHA-256 hex digest")
    if args.kernel_config is not None:
        try:
            kernel = json.loads(args.kernel_config)
        except json.JSONDecodeError as exc:
            parser.error(f"kernel config must be JSON: {exc.msg}")
        if not isinstance(kernel, dict):
            parser.error("kernel config must be a JSON object")
    try:
        declared_compilation(args)
        gold_engine_kwargs(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _install_native_capture(model, *, owner, tile_rows, logits_layout):
    from experiments.measure_glm_tr3_vllm import install_capture

    if hasattr(model, "_tr3_capture") or hasattr(model, "_tr3_diagnostic_capture_owner"):
        raise ValueError("candidate already has a native capture owner")
    model._tr3_diagnostic_capture_owner = owner
    return install_capture(model, tile_rows=tile_rows, logits_layout=logits_layout)


def _cleanup_native_capture(model, *, owner=None):
    """Remove only this scope's native hook, including partial installation."""
    from experiments.measure_glm_tr3_vllm import remove_capture

    if owner is None or getattr(model, "_tr3_diagnostic_capture_owner", None) != owner:
        return False
    if hasattr(model, "_tr3_capture_handle"):
        remove_capture(model)
    elif hasattr(model, "_tr3_capture"):
        del model._tr3_capture
    del model._tr3_diagnostic_capture_owner
    return True


def run(args):
    """Authenticate once, run exactly window zero, recheck, publish diagnostics.

    Worker replay installation/removal is owned by replay_request, including
    partial-install failures. Native capture removal is owned by this scope.
    Exceptions never publish successful evidence; a completed but non-firing
    replay is retained by build_report as invalid diagnostic evidence.
    """
    from experiments import measure_glm_tr3_vllm as native
    from experiments import diagnose_exl3_grouped_down_replay as driver
    from tools.gold_engine_options import gold_engine_kwargs
    from vllm import LLM, SamplingParams

    if "@sha256:" not in args.serve_image or not driver._digest(args.serve_image.split("@sha256:")[-1]):
        raise ValueError("diagnostic execution requires an immutable --serve-image digest")
    panel, inputs = native.load_panel(args.panel, arrays_root=args.arrays_root)
    teacher = native.load_teacher(args.teacher, args.teacher_sha256, panel)
    model = Path(args.model).resolve(strict=True)
    if native.sha256(model / "tokenizer.json") != native.TOKENIZER_SHA256:
        raise ValueError("candidate tokenizer.json differs from sealed reference vocabulary")
    token_identity = native.tokenizer_identity(model)
    if token_identity != teacher["tokenizer_identity"]:
        raise ValueError("candidate tokenizer files differ from teacher")
    candidate_identity = native.cached_checkpoint_identity(model, args.candidate_digest_cache)
    producer = driver.producer_identity()
    topology = gold_engine_kwargs(args)
    world_size = topology["tensor_parallel_size"]
    compilation = native.declared_compilation(args)
    kwargs = native.scorer_engine_kwargs(args, model=model, topology=topology)
    llm = LLM(**kwargs)
    capture_attempted = False
    capture_owner = uuid4().hex
    try:
        if native.refuse_if_spec_decode(llm=llm, context="EXL3 grouped-down diagnostic") is not False:
            raise ValueError("speculative decoding must be observed disabled")
        observation_kwargs = dict(expected_kv_cache_dtype=args.expected_kv_cache_dtype,
                                  requested_kv_cache_dtype=args.kv_cache_dtype,
                                  compilation=compilation)
        observed = native.observed_engine_configuration(llm, **observation_kwargs)
        capture_attempted = True
        worker_runtime = llm.apply_model(partial(_install_native_capture, owner=capture_owner,
                                                 tile_rows=args.tile_rows, logits_layout=args.logits_layout))
        worker_runtime.sort(key=lambda row: row["rank"])
        if ([row["rank"] for row in worker_runtime] != list(range(world_size))
                or any(row["world_size"] != world_size for row in worker_runtime)):
            raise ValueError("native capture installation lacks complete TP rank evidence")
        worker_kwargs = dict(expected_kv_cache_dtype=args.expected_kv_cache_dtype,
                             logits_layout=args.logits_layout)
        if compilation is not None:
            worker_kwargs["compilation"] = compilation
        worker_configuration = llm.collective_rpc(native.observed_worker_configuration,
                                                  kwargs=worker_kwargs)
        worker_configuration.sort(key=lambda row: row["rank"])
        if [row["rank"] for row in worker_configuration] != list(range(world_size)):
            raise ValueError("native configuration observation lacks every TP worker")
        binding = dict(worker_runtime=worker_runtime, engine_kwargs=kwargs,
                       observed_engine_configuration=observed,
                       observed_worker_configuration=worker_configuration,
                       serve_image=args.serve_image, producer_identity=producer,
                       candidate_identity=candidate_identity, teacher_sha256=args.teacher_sha256,
                       panel_sha256=native.PANEL_SHA256, logits_layout=args.logits_layout,
                       require_exl3_diag=True)
        if compilation is not None:
            binding["execution_mode"] = "compiled"
        before = llm.apply_model(partial(native.route_diagnostics, require_exl3=True))
        # This sidecar is initialization evidence, not a qualification or result.
        native.write_runtime_observation(args.output, binding)
        window = panel["windows"][0]
        window_id = window["window_id"]
        tokens = inputs[0][0].tolist()
        outputs = []

        def generate():
            result = llm.generate(
                [{"prompt_token_ids": tokens}],
                SamplingParams(max_tokens=1, temperature=1., top_p=1., seed=42,
                               prompt_logprobs=1, detokenize=False, ignore_eos=True),
                use_tqdm=False)
            outputs.extend(result)
            return result

        reports = driver.replay_request(
            llm, window_id=window_id,
            arm_kwargs=dict(index=0, window_id=window_id, descriptor=teacher["windows"][0],
                            teacher_root=str(Path(args.teacher).resolve().parent),
                            target_ids=tokens[1:]),
            generate=generate, module_name=args.module, repeats=args.repeats,
            required_rows=args.required_rows)
        if len(outputs) != 1:
            raise ValueError("native diagnostic requires one outstanding request")
        # Native completeness and prompt alignment are checked independently of
        # replay evidence: a successful kernel probe cannot excuse a bad request.
        native.collect_tp_result(reports, window_id=window_id, world_size=world_size,
                                 rows=native.CONTEXT_LENGTH - 1, vocab_size=native.VOCAB_SIZE,
                                 logits_layout=args.logits_layout)
        alignment = native.verify_prompt_alignment(outputs[0], tokens, reports)
        if native.cached_checkpoint_identity(model, args.candidate_digest_cache) != candidate_identity:
            raise ValueError("candidate checkpoint changed during diagnostic")
        if (native.tokenizer_identity(model) != token_identity
                or driver.producer_identity() != producer
                or native.bound_json(args.teacher, args.teacher_sha256) != teacher):
            raise ValueError("teacher/tokenizer/producer changed during diagnostic")
        native.load_panel(args.panel, arrays_root=args.arrays_root)
        if native.refuse_if_spec_decode(llm=llm, context="EXL3 grouped-down diagnostic") is not False:
            raise ValueError("speculative decoding changed during diagnostic")
        if native.observed_engine_configuration(llm, **observation_kwargs) != observed:
            raise ValueError("native engine configuration changed during diagnostic")
        after_workers = llm.collective_rpc(native.observed_worker_configuration, kwargs=worker_kwargs)
        after_workers.sort(key=lambda row: row["rank"])
        if after_workers != worker_configuration:
            raise ValueError("native worker configuration changed during diagnostic")
        after = llm.apply_model(partial(native.route_diagnostics, require_exl3=True))
        result = driver.build_report(
            reports=reports, runtime_binding=binding, diagnostics_before=before,
            diagnostics_after=after, world_size=world_size, window_id=window_id,
            module_name=args.module, repeats=args.repeats, required_rows=args.required_rows)
        result["prompt_alignment"] = alignment
        cleaned = llm.apply_model(partial(_cleanup_native_capture, owner=capture_owner))
        if len(cleaned) != world_size or any(value is not True for value in cleaned):
            raise ValueError("native capture cleanup lacks complete ownership receipts")
        capture_attempted = False
        native.atomic_json_write(result, args.output)
        return result
    except Exception as exc:
        # A request/collection failure is diagnostic evidence, never a pass.
        failure = {"schema": "prismaquant.exl3_grouped_down_diagnostic/1",
                   "diagnostic_only": True, "evidence_valid": False,
                   "invalid_reasons": [f"{type(exc).__name__}: {exc}"],
                   "runtime_binding": locals().get("binding"),
                   "reports": locals().get("reports", []),
                   "diagnostics_before": locals().get("before", []),
                   "diagnostics_after": locals().get("after", [])}
        native.atomic_json_write(driver._strict(failure), args.output)
        raise
    finally:
        # Also run after partial capture installation or a failed RPC. Replay
        # cleanup has already run inside replay_request before native removal.
        if capture_attempted:
            llm.apply_model(partial(_cleanup_native_capture, owner=capture_owner))


def main(argv=None):
    """Entry point lazily exported by diagnose_exl3_grouped_down_replay."""
    result = run(parse_args(argv))
    if not result.get("evidence_valid", False):
        raise SystemExit(1)
    return result
