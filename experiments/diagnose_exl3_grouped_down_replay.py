"""Opt-in bounded EXL3 observer; never qualification or a determinism verdict."""
import copy
import functools
import hashlib
import math
from pathlib import Path
import sys

from experiments import measure_glm_tr3_vllm as served
from tools import exl3_grouped_down_replay as probe

EXL3 = "vllm.model_executor.layers.quantization.exl3"


def _file_identity(path):
    if not path:
        return {"unavailable_reason": "no observable file"}
    try:
        return {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    except OSError as exc:
        return {"path": str(path), "unavailable_reason": str(exc)}


def install_replay(model, *, module_name=probe.DEFAULT_MODULE,
                   repeats=probe.DEFAULT_REPEATS, required_rows=probe.DEFAULT_REQUIRED_ROWS):
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
    model._tr3_replay_identity = {
        "source": _file_identity(getattr(module, "__file__", None)),
        "extension": _file_identity(getattr(ext, "__file__", None)),
    }
    try:
        return probe.install(model, ext, state, module_name=module_name,
                             repeats=repeats, required_rows=required_rows)
    except BaseException:
        remove_replay(model)
        raise


def remove_replay(model):
    ext = getattr(model, "_tr3_replay_extension", None)
    if ext is not None:
        probe.uninstall(model, ext)
    for attr in ("_tr3_replay_extension", "_tr3_replay_identity"):
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
    try:
        llm.apply_model(functools.partial(install_replay, module_name=module_name,
                                        repeats=repeats, required_rows=required_rows))
        llm.apply_model(functools.partial(served.arm_capture, **arm_kwargs))
        generate()
        return llm.apply_model(functools.partial(finish_replay_capture, window_id=window_id))
    finally:
        # Also runs if only a subset of workers installed successfully.
        llm.apply_model(remove_replay)


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


def _has_digest(value):
    if isinstance(value, dict):
        return not value.get("unavailable_reason") and any(_has_digest(v) for v in value.values())
    if isinstance(value, list):
        return bool(value) and all(_has_digest(v) for v in value)
    return _digest(value)


def build_report(*, reports, runtime_binding, diagnostics_before, diagnostics_after,
                 world_size, window_id, module_name, repeats, required_rows):
    reasons = []

    def require(condition, reason):
        if not condition:
            reasons.append(reason)

    require(type(world_size) is int and world_size > 0, "invalid TP world size")
    require(2 <= repeats <= 16 and required_rows > 0, "invalid replay bounds")
    for key in ("panel_sha256", "teacher_sha256"):
        require(_digest(runtime_binding.get(key)), f"missing {key}")
    candidate = runtime_binding.get("candidate_identity", {})
    require(isinstance(candidate, dict) and _digest(candidate.get("content_sha256"))
            and not candidate.get("unavailable_reason"), "candidate identity incomplete")
    require(_has_digest(runtime_binding.get("producer_identity")), "producer identity incomplete")
    image = runtime_binding.get("serve_image", "")
    require(isinstance(image, str) and "@sha256:" in image and _digest(image.split("@sha256:")[-1]), "runtime image incomplete")
    for key in ("engine_kwargs", "observed_engine_configuration", "observed_worker_configuration", "worker_runtime", "logits_layout"):
        require(bool(runtime_binding.get(key)), f"runtime {key} incomplete")

    def rank_map(items, label):
        ranks = [item.get("rank") for item in items if isinstance(item, dict)]
        require(len(ranks) == world_size and set(ranks) == set(range(world_size)), f"{label}: missing, duplicate or misattributed TP ranks")
        return {item.get("rank"): item for item in items if isinstance(item, dict)}

    before = rank_map(diagnostics_before, "before routes")
    after = rank_map(diagnostics_after, "after routes")
    rank_map(reports, "replay reports")
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
        r = report.get("replay")
        if not isinstance(r, dict):
            require(False, prefix + "probe did not fire: " + str(report.get("replay_declined")))
            continue
        for key, expected in (("rank", rank), ("world_size", world_size), ("window_id", window_id),
                              ("module", module_name), ("required_rows", required_rows), ("repeats", repeats), ("schema", probe.REPLAY_SCHEMA)):
            require(r.get(key) == expected, prefix + key + " attribution mismatch")
        ib, ia = r.get("inputs_before", {}), r.get("inputs_after", {})
        require(set(ib) == set(probe.ARGUMENT_NAMES) and set(ia) == set(probe.ARGUMENT_NAMES), prefix + "incomplete arguments")
        require(r.get("inputs_stable") is True and ib == ia, prefix + "changed arguments")
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
        require(ib.get("out", {}).get("sha256") == ob.get("fp32_sha256") and ia.get("out", {}).get("sha256") == oa.get("fp32_sha256"), prefix + "original metadata hash mismatch")
        shape = ib.get("out", {}).get("shape", [])
        require(len(shape) == 2 and shape[0] == required_rows, prefix + "actual output row mismatch")
        entries = r.get("replays", [])
        require(len(entries) == repeats, prefix + "incomplete replays")
        for index, entry in enumerate(entries):
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
