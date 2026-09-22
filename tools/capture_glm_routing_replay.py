"""PB-admitted original routing acquisition from a fresh prefix or owned boundary."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from prismaquant.calibration_data import load_calibration_input
from prismaquant.cost_stage_checkpoint import publish_new_bytes
from prismaquant.cost_streaming import build_streamed_model_identity
from prismaquant.glm_routing_replay import capture_replayed_glm_routes, require_cached_prefix_source_identity
from prismaquant.joint_adjoint_checkpoints import reference_from_record
from prismaquant.joint_aura import source_execution_identity
from prismaquant.joint_cost_quantum import build_quantum_source_runner
from prismaquant.perturbed_x_cache import prefetch_exact_activation_cache_entries
from prismaquant.tessera_joint_allocation import _read_bound
from prismaquant.tessera_joint_aura import _seed_source_identity_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="bound plan/prepared/capture and owned boundary-copy bindings")
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device-bytes", required=True, type=int)
    args = parser.parse_args()
    spec_binding = {"path": args.spec, "sha256": args.spec_sha256}
    spec = json.loads(_read_bound(spec_binding, "routing replay spec"))
    plan = json.loads(_read_bound(spec["plan"], "routing replay plan"))
    prepared = json.loads(_read_bound(spec["prepared"], "routing replay prepared"))
    if prepared["plan_sha256"] != spec["plan"]["sha256"]:
        raise ValueError("routing source plan/prepared identity differs")
    mode = spec.get("mode", "completed_boundary_replay")
    layer = spec["layer"]
    reference = entry = matches = copied = capture = None
    if mode == "fresh_source_prefix":
        if layer != 3 or spec.get("sample") != 0:
            raise ValueError("fresh routing acquisition is exactly sample zero through first MoE layer3")
    elif mode == "completed_boundary_replay":
        capture = json.loads(_read_bound(spec["capture"], "routing replay completed capture"))
        if capture.get("status") != "complete" or capture.get("schema") != "prismaquant.joint_adjoint_capture.v1":
            raise ValueError("routing replay requires a completed immutable parent capture")
        run = capture["run_identity"]
        if run["plan_sha256"] != spec["plan"]["sha256"] or run["prepared_sha256"] != spec["prepared"]["sha256"]:
            raise ValueError("routing replay parent plan/prepared identity differs")
        if type(layer) is not int or layer != 8:
            raise ValueError("this retained acquisition is GLM layer-eight boundary")
        matches = [v for v in capture["boundary_entries"][str(layer)] if v["name"] == f"boundary-0-{layer}-at-{layer}"]
        if len(matches) != 1:
            raise ValueError("parent capture has no unique original sample-zero boundary")
        entry = copy.deepcopy(matches[0]);copied = spec["owned_boundary_copy"]
        if copied["sha256"] != entry["sha256"] or copied["path"] == entry["path"]:
            raise ValueError("replay needs a separately owned byte-identical boundary copy")
        entry["path"] = copied["path"];reference = reference_from_record(entry)
    else:
        raise ValueError("unknown routing acquisition mode")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "routing_boundary.pt").exists() or (root / "receipt.json").exists():
        raise ValueError("routing replay output namespace must be fresh")
    ids, calibration = load_calibration_input(plan["calibration_input"]["path"],
        expected_sha256=plan["calibration_input"]["sha256"], n_samples=512, seqlen=512)
    if calibration != prepared["calibration_input"] or (capture is not None and calibration["calibration_sha256"] != capture["run_identity"]["calibration_sha256"]):
        raise ValueError("routing replay calibration differs from parent/source preparation")
    census = json.loads(_read_bound(plan["inputs"]["census"], "routing replay producer census"))
    producer_source = census["expert_projection"]["producer"]["source"]
    expected_files = {Path(v["path"]).name: v["sha256"] for v in prepared["source_model_identity"]["shards"]}
    if (producer_source["files"] != expected_files
            or producer_source["tensors"] != prepared["source_model_identity"]["checkpoint_weight_map"]):
        raise ValueError("native producer file/tensor identity differs from original qualified source")
    cache_path = _seed_source_identity_cache(plan, root)
    require_cached_prefix_source_identity(plan["model"], cache_path, prepared["source_model_identity"])
    from prismaquant.memory_management import enforce_device_envelope
    envelope = enforce_device_envelope("cuda", args.device_bytes, where="native GLM routing prefix")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    from prismaquant.staged_tier_policy import activate_staged_tier_policy
    from prismaquant.staged_lease import resolve_sealed_readset, load_sealed_readset
    from prismaquant.residency_map import bind_residency_manifest
    activate_staged_tier_policy("ram,ssd")
    _cas, digest, _size = resolve_sealed_readset();load_sealed_readset(digest);bind_residency_manifest(digest)
    phases = spec.get("read_phases")
    from prismaquant.prismabuild_progress import report
    if phases:
        from prismaquant.joint_run_progress import declared_phases
        if phases != ['head','layer-0','layer-1','layer-2','layer-3'] or tuple(phases) != declared_phases():
            raise ValueError("routing source phases differ from the admitted progress contract")
        if not report('head', 0, unit='routing_captures'):
            raise ValueError("routing source phase channel is unavailable")
    runner = None
    try:
        runner = build_quantum_source_runner(plan, offload_folder=root / "offload")
        source = build_streamed_model_identity(runner, plan["model"],
            identity_cache_path=cache_path)
        if source != prepared["source_model_identity"]:
            raise ValueError("replayed BF16 source identity differs")
        if source_execution_identity(runner.model) != prepared["source_execution"]:
            raise ValueError("replayed source execution/derivative differs")
        if mode == "fresh_source_prefix":
            from prismaquant.cost_streaming import StreamedForwardBoundaries
            batch_ids, positions, hidden, embeddings, mask = runner._prepare(ids[:1])
            batch = StreamedForwardBoundaries(batch_ids, positions, embeddings, mask, [], None)
            pass_state = runner.profile.new_forward_pass_state()
            with torch.no_grad():
                for depth in range(layer):
                    if phases and not report(f"layer-{depth}", 0, unit="routing_captures"):
                        raise ValueError("routing source phase report failed")
                    runner.context.schedule_prefetch(depth)
                    runner.context.install(depth, require_prefetched=True, prefetch_following=False)
                    try:
                        hidden = runner.isolated_layer(batch, depth, hidden, pass_state=pass_state)
                    finally:
                        runner.context.unload(depth)
            shared = runner.profile.capture_forward_pass_state(pass_state)
            if shared:
                raise ValueError("fresh routing prefix requires an explicit cross-layer shared-state binding")
            if phases and not report(f"layer-{layer}", 0, unit="routing_captures"):
                raise ValueError("routing source phase report failed")
            runner.context.schedule_prefetch(layer);runner.context.install(layer, require_prefetched=True, prefetch_following=False)
            result = capture_replayed_glm_routes(runner, hidden, ids[:1], layer=layer,
                calibration=calibration, producer_source=producer_source,
                parent_boundary={"spec":spec_binding,"source":"fresh_original_bf16_prefix",
                                 "completed_layers":list(range(layer)),"sample":0})
            result["metadata"]["replay"]["source"] = "fresh_original_bf16_prefix_0_through_3"
        else:
            runner.context.schedule_prefetch(layer);runner.context.install(layer, require_prefetched=True)
            with prefetch_exact_activation_cache_entries([reference],
                    max_tensor_bytes=reference.tensor_bytes,
                    expected_session=entry["metadata"]["identity"]["session"]) as window:
                boundary = window.get(reference)
                result = capture_replayed_glm_routes(runner, boundary, ids[:1], layer=layer,
                    calibration=calibration, producer_source=producer_source,
                    parent_boundary={"capture":spec["capture"],"original_entry":matches[0],
                                     "owned_copy":copied,"spec":spec_binding})
        from prismaquant.dev_mode import dev_stamp
        result["metadata"].update(dev_stamp(timestamped=False))
        result["metadata"]["source_cache_reuse"] = {"binding":plan["source_identity_cache"], "validator":"validate_cached_streamed_model_identity", "complete_checkpoint":True, "content_sha256":source["content_sha256"]}
        result["metadata"]["capture_device_envelope"] = envelope
        buffer = io.BytesIO()
        torch.save(result, buffer)
        raw = buffer.getvalue()
        target = root / "routing_boundary.pt"
        if not publish_new_bytes(target, raw):
            raise ValueError("routing capture output already exists")
        receipt = {"schema": "prismaquant.glm_routing_replay_receipt.v1", "status": "complete",
                   "spec": spec_binding, "boundary": {"path": str(target), "sha256": hashlib.sha256(raw).hexdigest()},
                   "metadata": result["metadata"]}
        receipt.update(dev_stamp(timestamped=False))
        if not publish_new_bytes(root / "receipt.json", (json.dumps(receipt, sort_keys=True) + "\n").encode()):
            raise ValueError("routing capture receipt already exists")
        if phases:report(phases[-1], 1, unit="routing_captures")
        print(json.dumps({"status": "complete", "boundary": receipt["boundary"]}, sort_keys=True))
    finally:
        if runner is not None:
            runner.shutdown()


if __name__ == "__main__":
    main()
