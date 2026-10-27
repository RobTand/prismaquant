"""Opt-in bounded EXL3 observer; never qualification or a determinism verdict."""
import copy
import functools
import math
import json
from types import SimpleNamespace
from pathlib import Path
import sys
from uuid import uuid4

from experiments import measure_glm_tr3_vllm as served
from tools import exl3_grouped_down_replay as probe

EXL3 = "vllm.model_executor.layers.quantization.exl3"


def _file_identity(path):
    if not path:
        return {"unavailable_reason": "no observable file"}
    try:
        return {"path": str(path), "sha256": served.sha256(Path(path))}
    except OSError as exc:
        return {"path": str(path), "unavailable_reason": str(exc)}


def install_replay(model, *, module_name=probe.DEFAULT_MODULE,
                   repeats=probe.DEFAULT_REPEATS, required_rows=probe.DEFAULT_REQUIRED_ROWS,
                   owner=None):
    if not 2 <= repeats <= 16 or required_rows < 1:
        raise ValueError("positive rows and 2..16 repeats required")
    state = getattr(model, "_tr3_capture", None)
    module = sys.modules.get(EXL3)
    cache = getattr(module, "_FAT_MOE_EXT_CACHE", None)
    if state is None or not cache or cache[0] is None:
        raise ValueError("requires native capture and already-loaded EXL3 extension")
    if getattr(model, "_tr3_replay_extension", None) is not None:
        raise ValueError("replay already installed")
    ext = cache[0]
    model._tr3_replay_extension = ext
    model._tr3_replay_owner = owner
    model._tr3_replay_identity = {
        "source": _file_identity(getattr(module, "__file__", None)),
        "extension": _file_identity(getattr(ext, "__file__", None)),
    }
    try:
        return probe.install(model, ext, state, module_name=module_name,
                             repeats=repeats, required_rows=required_rows)
    except BaseException:
        remove_replay(model, owner=owner)
        raise


def remove_replay(model, *, owner=None):
    if getattr(model, "_tr3_replay_owner", None) != owner:
        return
    ext = getattr(model, "_tr3_replay_extension", None)
    if ext is not None:
        probe.uninstall(model, ext)
    for attr in ("_tr3_replay_extension", "_tr3_replay_identity", "_tr3_replay_owner"):
        if hasattr(model, attr):
            delattr(model, attr)


def finish_replay_capture(model, *, window_id):
    replay = getattr(model, "_tr3_grouped_down_replay", None)
    # Snapshot while still armed: native finish clears the window and teacher.
    record = copy.deepcopy(replay.record) if replay is not None else None
    declined = list(replay.declined) if replay is not None else ["probe not installed"]
    identity = copy.deepcopy(getattr(model, "_tr3_replay_identity", {}))
    result = served.finish_capture(model, window_id=window_id)
    return dict(result, replay=record, replay_declined=declined, loaded_exl3_identity=identity)


def replay_request(llm, *, window_id, arm_kwargs, generate, module_name,
                   repeats, required_rows):
    owner = uuid4().hex
    try:
        llm.apply_model(functools.partial(install_replay, module_name=module_name,
                                        repeats=repeats, required_rows=required_rows, owner=owner))
        llm.apply_model(functools.partial(served.arm_capture, **arm_kwargs))
        generate()
        return llm.apply_model(functools.partial(finish_replay_capture, window_id=window_id))
    finally:
        # Also runs if only a subset of workers installed successfully.
        llm.apply_model(functools.partial(remove_replay, owner=owner))


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _strict(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {k: _strict(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_strict(v) for v in value)
    if isinstance(value, list):
        return [_strict(v) for v in value]
    return value


def producer_identity():
    """Bind the native producer and every file in the diagnostic execution path."""
    root = Path(__file__).resolve().parents[1]
    return dict(served.producer_identity(), diagnostic_source_files={
        name: served.sha256(root / name) for name in (
            "experiments/diagnose_exl3_grouped_down_replay.py",
            "experiments/exl3_replay_cli.py", "tools/exl3_grouped_down_replay.py")})


def _ranked(items, world_size, label):
    if (not isinstance(items, list) or len(items) != world_size
            or any(not isinstance(row, dict) or type(row.get("rank")) is not int
                   for row in items)
            or {row["rank"] for row in items} != set(range(world_size))):
        raise ValueError(f"{label}: missing, duplicate or misattributed TP ranks")
    return {row["rank"]: row for row in items}


def _configuration(value, compilation):
    config = SimpleNamespace(**{
        group: SimpleNamespace(**value[group]) for group in (
            "model_config", "cache_config", "scheduler_config", "parallel_config")},
        speculative_config=None)
    config.model_config.multimodal_config = SimpleNamespace(
        language_model_only=value["language_model_only"])
    dtype = value["cache_config"]["cache_dtype"]
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("recorded configuration has no cache dtype")
    if compilation is not None:
        config.compilation_config = SimpleNamespace(**value["compilation_config"])
    observed = served.observed_configuration(config, expected_kv_cache_dtype=dtype,
                                              compilation=compilation)
    if served.canonical_sha256(observed) != served.canonical_sha256(value):
        raise ValueError("recorded configuration differs from native observation schema")


def _validate_runtime(binding, world_size):
    from prismaquant.streamed_model_identity import (
        SOURCE_MODEL_BINDING_SCHEMA, validate_source_model_binding,
    )
    from tools.gold_engine_options import gold_engine_kwargs

    if served.canonical_sha256(binding["producer_identity"]) != served.canonical_sha256(producer_identity()):
        raise ValueError("producer identity does not match the diagnostic source")
    kwargs = binding["engine_kwargs"]
    args = SimpleNamespace(**kwargs)
    args.execution_mode = binding.get("execution_mode", "eager")
    args.compilation_config = (json.dumps(kwargs["compilation_config"])
                               if "compilation_config" in kwargs else None)
    args.attention_backend = kwargs.get("attention_backend")
    args.kernel_config = (json.dumps(kwargs["kernel_config"])
                          if "kernel_config" in kwargs else None)
    fraction = kwargs["gpu_memory_utilization"]
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("runtime GPU-memory fraction is invalid")
    if "kernel_config" in kwargs and not isinstance(kwargs["kernel_config"], dict):
        raise ValueError("runtime kernel configuration is not an object")
    topology = gold_engine_kwargs(args)
    if topology["tensor_parallel_size"] != world_size or kwargs.get("quantization") != "exl3":
        raise ValueError("runtime request has a different TP world or quantization")
    expected = served.scorer_engine_kwargs(args, model=kwargs["model"], topology=topology)
    if (served.canonical_sha256(kwargs) != served.canonical_sha256(expected)
            or not isinstance(kwargs["model"], str) or not kwargs["model"]):
        raise ValueError("runtime request differs from the native scorer contract")
    validate_source_model_binding({"schema": SOURCE_MODEL_BINDING_SCHEMA,
        "effective_dtype": "torch." + kwargs["dtype"],
        "checkpoint": binding["candidate_identity"]}, where="EXL3 candidate")
    compilation = served.declared_compilation(args)
    _configuration(binding["observed_engine_configuration"], compilation)
    layout = binding["logits_layout"]
    if layout not in served.LOGITS_LAYOUTS or binding.get("require_exl3_diag") is not True:
        raise ValueError("runtime lacks the declared native layout and EXL3 observation")
    captures = _ranked(binding["worker_runtime"], world_size, "native captures")
    workers = _ranked(binding["observed_worker_configuration"], world_size, "native workers")
    worker_dtypes = set()
    for rank in range(world_size):
        worker, capture = workers[rank], captures[rank]
        _configuration(worker["configuration"], compilation)
        dtype = worker["configuration"]["cache_config"]["cache_dtype"]
        if (worker["worker_cache_dtype"] != dtype or worker["runner_cache_dtype"] != dtype
                or worker["logits_layout"] != layout):
            raise ValueError(f"rank {rank}: worker KV dtype or prompt layout differs")
        worker_dtypes.add(dtype)
        initial_dtype = worker["runner_initial_kv_dtype"]
        if (not isinstance(initial_dtype, str) or not initial_dtype.startswith("torch.")
                or not isinstance(getattr(served.torch, initial_dtype[6:], None), served.torch.dtype)):
            raise ValueError(f"rank {rank}: initial runner dtype unavailable")
        source = worker["prompt_layout_source"]
        if layout == "vllm_v2_chunk1024":
            if (source["runner_class"] != "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
                    or source["prompt_worker_class"] != "vllm.v1.worker.gpu.sample.prompt_logprob.PromptLogprobsWorker"
                    or not all(_digest(source[key]) for key in (
                        "runner_source_sha256", "prompt_worker_source_sha256"))):
                raise ValueError(f"rank {rank}: V2 prompt implementation unavailable")
        elif source is not None:
            raise ValueError(f"rank {rank}: legacy prompt layout has unexpected source")
        if capture["world_size"] != world_size:
            raise ValueError(f"rank {rank}: native capture TP world differs")
        for field in ("torch", "vllm", "model_class", "text_model_class", "logits_owner", "logits_module_class"):
            if not isinstance(capture[field], str) or not capture[field]:
                raise ValueError(f"rank {rank}: native {field} unavailable")
        sources = capture["source_files"]
        if (set(sources) != {"model", "text_model", "logits_module", "gpu_model_runner"}
                or not all(_digest(digest) for digest in sources.values())):
            raise ValueError(f"rank {rank}: native implementation source incomplete")
        attention = capture["attention_runtime"]
        if (not isinstance(attention, list) or not attention
                or len({row["module"] for row in attention}) != len(attention)):
            raise ValueError(f"rank {rank}: attention observation incomplete")
        for row in attention:
            if (not all(isinstance(row[field], str) and row[field]
                        for field in ("module", "backend", "kv_cache_dtype", "cache_dtype"))
                    or not _digest(row["backend_source_sha256"])):
                raise ValueError(f"rank {rank}: attention implementation unavailable")
            cache = row["allocated_kv_cache"]
            if cache is not None and (
                    not isinstance(cache["shape"], list) or not cache["shape"]
                    or any(type(size) is not int or size <= 0 for size in cache["shape"])
                    or not all(isinstance(cache[field], str) and cache[field]
                               for field in ("dtype", "device"))):
                raise ValueError(f"rank {rank}: allocated attention cache incomplete")
    if len(worker_dtypes) != 1:
        raise ValueError("native workers resolved different KV dtypes")


def _report_invalid_reasons(*, reports, runtime_binding, diagnostics_before, diagnostics_after,
                            world_size, window_id, module_name, repeats, required_rows):
    reasons = []

    def require(condition, reason):
        if not condition:
            reasons.append(reason)

    if type(world_size) is not int or world_size < 1:
        raise ValueError("invalid TP world size")
    if (type(repeats) is not int or not 2 <= repeats <= 16
            or type(required_rows) is not int or required_rows < 1):
        raise ValueError("invalid replay bounds")
    for key in ("panel_sha256", "teacher_sha256"):
        require(_digest(runtime_binding.get(key)), f"missing {key}")
    require(runtime_binding.get("panel_sha256") == served.PANEL_SHA256, "panel identity differs from sealed panel")
    image = runtime_binding.get("serve_image", "")
    require(isinstance(image, str) and "@sha256:" in image and _digest(image.split("@sha256:")[-1]), "runtime image incomplete")
    _validate_runtime(runtime_binding, world_size)

    before = _ranked(diagnostics_before, world_size, "before routes")
    after = _ranked(diagnostics_after, world_size, "after routes")
    _ranked(reports, world_size, "replay reports")
    for rank in range(world_size):
        b, a = before.get(rank, {}), after.get(rank, {})
        for label, observation in (("before", b), ("after", a)):
            route = observation.get("exl3", {})
            require(route.get("tp_rank") == rank and route.get("tp_size") == world_size,
                    f"rank {rank} {label}: internal TP identity missing/mismatched")
            require(type(route.get("prefill_layer_calls")) is int and route["prefill_layer_calls"] >= 0,
                    f"rank {rank} {label}: missing prefill counter")
            require(_digest(observation.get("exl3_source_sha256")), f"rank {rank} {label}: missing overlay identity")
        require(b.get("exl3_source_sha256") == a.get("exl3_source_sha256"), f"rank {rank}: overlay changed")
        bc, ac = b.get("exl3", {}).get("prefill_layer_calls"), a.get("exl3", {}).get("prefill_layer_calls")
        require(type(bc) is int and type(ac) is int and ac > bc, f"rank {rank}: no route activity")

    for report in reports:
        rank = report.get("rank")
        prefix = f"rank {rank}: "
        require(report.get("window_id") == window_id, prefix + "native window mismatch")
        require(report.get("world_size") == world_size, prefix + "native TP world mismatch")
        identity = report.get("loaded_exl3_identity", {})
        require(isinstance(identity, dict) and all(
            isinstance(identity.get(k), dict) and bool(identity[k].get("path"))
            and _digest(identity[k].get("sha256")) and not identity[k].get("unavailable_reason")
            for k in ("source", "extension")), prefix + "loaded EXL3 identity unavailable")
        require(identity.get("source", {}).get("sha256") == before.get(rank, {}).get("exl3_source_sha256")
                == after.get(rank, {}).get("exl3_source_sha256"),
                prefix + "loaded overlay differs from route source")
        r = report.get("replay")
        if not isinstance(r, dict):
            require(False, prefix + "probe did not fire: " + str(report.get("replay_declined")))
            continue
        for key, expected in (("rank", rank), ("world_size", world_size), ("window_id", window_id),
                              ("module", module_name), ("required_rows", required_rows), ("repeats", repeats), ("schema", probe.REPLAY_SCHEMA)):
            require(r.get(key) == expected, prefix + key + " attribution mismatch")
        ib, ia = r.get("inputs_before", {}), r.get("inputs_after", {})
        require(set(ib) == set(probe.ARGUMENT_NAMES) and set(ia) == set(probe.ARGUMENT_NAMES), prefix + "incomplete arguments")
        require(r.get("inputs_stable") is True
                and {name: value for name, value in ib.items() if name != "out"}
                == {name: value for name, value in ia.items() if name != "out"}, prefix + "changed arguments")
        for name in probe.ARGUMENT_NAMES:
            for side in (ib, ia):
                value = side.get(name, {})
                require(_digest(value.get("sha256")) and value.get("hashed") is True and bool(value.get("shape"))
                        and value.get("name") == name and bool(value.get("dtype"))
                        and type(value.get("nbytes")) is int and value["nbytes"] > 0,
                        prefix + name + " missing hash/shape/metadata")
        ob, oa = r.get("original_out_before", {}), r.get("original_out_after", {})
        require(r.get("original_out_unchanged_after_replays") is True and ob == oa, prefix + "served output changed")
        for original in (ob, oa):
            require(all(_digest(original.get(k)) for k in ("fp32_sha256", "bf16_sha256")) and original.get("fp32_finite", {}).get("all_finite") is True, prefix + "original nonfinite/incomplete")
        initial_digest = r.get("replay_initial_out_sha256")
        require(_digest(initial_digest) and ib.get("out", {}).get("sha256") == initial_digest,
                prefix + "replay initial destination does not match invocation")
        require(ia.get("out", {}).get("sha256") == oa.get("fp32_sha256"),
                prefix + "served output metadata hash mismatch")
        require({key: value for key, value in ib.get("out", {}).items() if key != "sha256"}
                == {key: value for key, value in ia.get("out", {}).items() if key != "sha256"},
                prefix + "destination metadata changed")
        shape = ib.get("out", {}).get("shape", [])
        require(len(shape) == 2 and shape[0] == required_rows, prefix + "actual output row mismatch")
        entries = r.get("replays", [])
        require(len(entries) == repeats, prefix + "incomplete replays")
        for index, entry in enumerate(entries):
            require(entry.get("initial_out_sha256") == initial_digest,
                    prefix + "replay destination seed differs from invocation")
            require(entry.get("index") == index and _digest(entry.get("sha256")) and entry.get("finite", {}).get("all_finite") is True, prefix + "replay hash/finiteness/index incomplete")
            for key in (("vs_original",) if index == 0 else ("vs_original", "vs_first_replay")):
                delta = entry.get(key, {})
                require(delta.get("candidate_sha256") == entry.get("sha256") and delta.get("reference_sha256") == (ob.get("fp32_sha256") if key == "vs_original" else entries[0].get("sha256")), prefix + "comparison hash mismatch")
                for metric in ("fp32_max_abs_delta", "bf16_max_abs_delta", "bf16_max_ulp_delta", "fp32_differing_elements", "bf16_differing_elements"):
                    v = delta.get(metric)
                    require(type(v) in (int, float) and math.isfinite(v) and v >= 0, prefix + "missing/nonfinite delta " + metric)
                for finite in ("reference_finite", "candidate_finite", "reference_bf16_finite", "candidate_bf16_finite"):
                    require(delta.get(finite, {}).get("all_finite") is True, prefix + "nonfinite comparison " + finite)
        require(r.get("distinct_replay_digests") == len({e.get("sha256") for e in entries}), prefix + "distinct digest count mismatch")
        require(r.get("pointed_to_weight_bytes_hashed") is False, prefix + "pointer limitation missing")
    return reasons


def build_report(*, reports, runtime_binding, diagnostics_before, diagnostics_after,
                 world_size, window_id, module_name, repeats, required_rows):
    """Validate external worker JSON without discarding malformed diagnostics."""
    raw = dict(reports=reports, runtime_binding=runtime_binding,
               diagnostics_before=diagnostics_before, diagnostics_after=diagnostics_after,
               world_size=world_size, window_id=window_id, module_name=module_name,
               repeats=repeats, required_rows=required_rows)
    try:
        reasons = _report_invalid_reasons(**raw)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        reasons = [f"invalid diagnostic evidence: {type(exc).__name__}: {exc}"]
    return _strict({"schema": "prismaquant.exl3_grouped_down_diagnostic/1", "diagnostic_only": True,
                    "evidence_valid": not reasons, "invalid_reasons": reasons,
                    "runtime_binding": copy.deepcopy(runtime_binding), "reports": copy.deepcopy(reports),
                    "diagnostics_before": copy.deepcopy(diagnostics_before), "diagnostics_after": copy.deepcopy(diagnostics_after),
                    "limitations": ["Identical bounded replays are not a determinism verdict.",
                                    "Observer synchronization can mask the original behavior.",
                                    "Pointer-table hashes do not hash pointed-to weight bytes.",
                                    "P1 remains unresolved; historical full25 remains non-comparable."]})


def parse_args(argv=None):
    from experiments.exl3_replay_cli import parse_args as parse
    return parse(argv)


def main(argv=None):
    from experiments.exl3_replay_cli import main as run
    return run(argv)


if __name__ == "__main__":
    main()
