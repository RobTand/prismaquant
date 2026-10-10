"""PB-admitted single traversal for GLM native routed inputs, layers 3..44.

The legacy Stage B DEV/cache/canonical-v2 branch is retained unchanged. The
exclusive current-original branch takes independently bound authority and plan
inputs through the existing source owner. It remains nonactivating: missing
qualification/root admission and the original material CUDA guard refuse before
material/profile/device/output work. Both branches execute only sample zero of
the complete 512x512 calibration; neither grants full-draw quality or prices.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import resource
import sys
import time
from pathlib import Path

if __package__:
    from prismaquant.digests import bytes_sha256hex
else:
    # Stdlib-only script use binds the same tree's owner by file path, so the
    # host needs no installed package. digests.py is stdlib-only: register the
    # file module under a name carrying that path's exact bytes, so two
    # checkouts in one process never share a cache entry.
    _digest_owner_path = Path(__file__).resolve().parents[1] / "prismaquant" / "digests.py"
    _digest_owner_name = "_prismaquant_standalone_digest_owner_" + os.fsencode(_digest_owner_path).hex()
    if _digest_owner_name not in sys.modules:
        _digest_owner_spec = importlib.util.spec_from_file_location(_digest_owner_name, _digest_owner_path)
        _digest_owner_module = importlib.util.module_from_spec(_digest_owner_spec)
        sys.modules[_digest_owner_name] = _digest_owner_module
        try:
            _digest_owner_spec.loader.exec_module(_digest_owner_module)
        except BaseException:
            del sys.modules[_digest_owner_name]
            raise
    bytes_sha256hex = sys.modules[_digest_owner_name].bytes_sha256hex


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--original-authority")
    parser.add_argument("--original-authority-sha256")
    parser.add_argument("--session-preparation")
    parser.add_argument("--session-preparation-sha256")
    args = parser.parse_args(argv)
    if (args.original_authority is None) != (args.original_authority_sha256 is None):
        parser.error("original authority requires both its path and independently supplied SHA256")
    if (args.session_preparation is None) != (args.session_preparation_sha256 is None):
        parser.error("original session preparation requires both its path and independently supplied SHA256")
    if args.original_authority is None and args.session_preparation is not None:
        parser.error("original session preparation cannot be used by legacy DEV/cache capture")
    if args.original_authority is not None and args.session_preparation is None:
        parser.error("original capture requires its independently bound existing session preparation")
    if args.original_authority is not None:
        return _capture_original(args)
    if os.environ.get("PRISMAQUANT_DEV_MODE") != "1":
        raise ValueError("fresh routed capture retains its explicit DEV source authority")

    import torch

    from prismaquant.calibration_data import load_calibration_input
    from prismaquant.cost_stage_checkpoint import publish_new_bytes
    from prismaquant.cost_streaming import validate_cached_streamed_model_identity
    from prismaquant.glm_routing_capture import capture_streamed_glm_routes
    from prismaquant.joint_aura import source_execution_identity
    from prismaquant.joint_cost_quantum import build_quantum_source_runner
    from prismaquant.matmul_arithmetic import pin_matmul_arithmetic
    from prismaquant.memory_management import enforce_device_envelope
    from prismaquant.native_moe_panel import routed_boundary_inputs
    from prismaquant.prismabuild_progress import report
    from prismaquant.tessera_calibration_cache import (
        authenticate_selected_capture_source,
        require_capture_contract,
    )
    from prismaquant.stage_inputs import read_bound as _read_bound

    started = time.monotonic()
    plan_ref = {"path": args.plan, "sha256": args.plan_sha256}
    prepared_ref = {"path": args.prepared, "sha256": args.prepared_sha256}
    plan = json.loads(_read_bound(plan_ref, "routed capture plan"))
    prepared = json.loads(_read_bound(prepared_ref, "routed capture preparation"))
    if prepared["plan_sha256"] != args.plan_sha256 or prepared.get("status") != "complete":
        raise ValueError("routed capture preparation differs from its complete Stage B plan")
    # Keep the existing Stage B bounded source window; do not turn a traversal
    # into an all-source preload or let it fall back to synchronous cold reads.
    prefetch = plan["source_prefetch"]
    if (prefetch.get("require_prefetched_residency") is not True
            or prefetch.get("max_cache_slots") != 2
            or prefetch.get("prefetch_lookahead") != 1
            or prefetch.get("prefetch_workers") != 1):
        raise ValueError("routed capture needs the Stage B two-slot, one-ahead source window")
    ids, calibration = load_calibration_input(plan["calibration_input"]["path"],
        expected_sha256=plan["calibration_input"]["sha256"], n_samples=512, seqlen=512)
    if calibration != prepared["calibration_input"]:
        raise ValueError("routed capture calibration differs from Stage B")
    # Read the bound identity cache, but never scan all checkpoint payloads
    # before the first layer. The source owner below authenticates on demand.
    cache_binding = plan["source_identity_cache"]
    _read_bound(cache_binding, "routed capture complete source cache")
    source = validate_cached_streamed_model_identity(plan["model"], cache_binding["path"])
    if source != prepared["source_model_identity"]:
        raise ValueError("routed capture source differs from Stage B")
    census = json.loads(_read_bound(plan["inputs"]["census"], "routed capture census"))
    producer = census["expert_projection"]["producer"]["source"]
    shards = source["shards"]
    if not isinstance(shards, list):
        raise ValueError("routed capture source must declare checkpoint shards")
    if (producer["tensors"] != source["checkpoint_weight_map"]
            or producer["files"] != {Path(row["path"]).name: row["sha256"] for row in shards}):
        raise ValueError("routed capture producer differs from the complete source")
    canonical = plan["canonical_capture"]
    manifest = require_capture_contract(canonical["path"], expected_sha256=canonical["sha256"])
    acquisition = {
        "dev_uncertified": True, "dev_mode": {"PRISMAQUANT_DEV_MODE": "1"},
        "source_cache_reuse": {"binding": cache_binding, "complete_checkpoint": True,
            "validator": "validate_cached_streamed_model_identity",
            "content_sha256": source["content_sha256"]},
    }
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    pin_matmul_arithmetic()
    envelope = enforce_device_envelope("cuda", plan["max_gpu_bytes"],
                                       where="streamed GLM routed capture")
    entries = []

    def publish(layer, result):
        # Exercise the actual intake for every file before it is published.
        transported = {"source": "routed_boundary_capture",
                       "boundary_metadata": result["metadata"], **result["tensors"]}
        routed_boundary_inputs(transported, calibration_receipt=calibration,
            capture_manifest=manifest, device="cpu", source_model_identity=source)
        buffer = io.BytesIO()
        torch.save(result, buffer)
        raw = buffer.getvalue()
        target = root / f"layer-{layer:03d}.pt"
        if not publish_new_bytes(target, raw):
            raise ValueError(f"routed capture output already exists: {target}")
        entries.append({"layer": layer, "unit": result["metadata"]["unit"],
                        "path": str(target), "sha256": bytes_sha256hex(raw),
                        "bytes": len(raw)})
        report("capture", len(entries), unit=result["metadata"]["unit"])
        print(json.dumps({"captured": layer, "count": len(entries),
                          "elapsed_s": time.monotonic() - started}), flush=True)

    owner = authenticate_selected_capture_source(
        plan["inputs"]["census"]["path"], canonical["path"], expected_sha256=canonical["sha256"],
        model=plan["model"], max_act_rows=manifest["identity"]["max_act_rows"],
        attention_implementation="eager", release_read_pages=True)
    runner = None
    try:
        runner = build_quantum_source_runner(plan, offload_folder=root / "offload",
                                             source_authentication=owner)
        if source_execution_identity(runner.model) != prepared["source_execution"]:
            raise ValueError("routed capture source execution differs from Stage B")
        layers = capture_streamed_glm_routes(runner, ids, calibration=calibration,
            producer_source=producer, source_acquisition=acquisition, consume=publish)
        receipt = {
            "schema": "prismaquant.glm_routed_capture_pass.v1", "status": "complete",
            "plan": plan_ref, "prepared": prepared_ref, "calibration": calibration,
            "scope": "sample zero of the bound 512x512 draw; not a quality measurement",
            "layers": layers, "entries": entries, "source_acquisition": acquisition,
            "source_authentication": owner.receipt(),
            "source_prefetch": runner.context.prefetch_summary(),
            "device_envelope": envelope,
            "aggregate_memory_bound_bytes": plan["aggregate_memory_bytes"],
            "elapsed_s": time.monotonic() - started,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    finally:
        if runner is not None:
            runner.shutdown()
        owner.close()
    raw = (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode()
    if not publish_new_bytes(root / "receipt.json", raw):
        raise ValueError("routed capture receipt already exists")
    print(json.dumps({"status": "complete", "captures": len(entries),
                      "receipt": str(root / "receipt.json")}), flush=True)
    return 0


class _BoundedCaptureBuffer(io.BytesIO):
    """Use the existing Torch bytes serializer inside its admitted CPU cap."""
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise RuntimeError("original routed serialization exceeds its admitted byte envelope")
        return super().write(data)


def _pin_original_arithmetic(runtime):
    import torch

    from prismaquant.matmul_arithmetic import bf16_reduction_from_environment, pin_matmul_arithmetic
    from prismaquant.source_generation import validate_original_source_runtime

    validate_original_source_runtime(runtime, runtime)
    expected = runtime["arithmetic"]
    planned = {"matmul_precision": "highest", "allow_tf32": False,
               "allow_bf16_reduced_precision_reduction": bf16_reduction_from_environment(os.environ)}
    if expected != planned:
        raise ValueError("original routed arithmetic/environment differs from its independently sealed runtime")
    pin_matmul_arithmetic()
    actual = {"matmul_precision": torch.get_float32_matmul_precision(),
              "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
              "allow_bf16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)}
    if actual != expected:
        raise ValueError("original routed arithmetic pin differs from the observed sealed selectors")


def _capture_original(args):
    import torch

    from prismaquant.calibration_data import load_calibration_input
    from prismaquant.cost_stage_checkpoint import publish_new_bytes
    from prismaquant.cost_streaming import (StreamedBoundaryArtifacts, build_streamed_model_identity,
                                            check_boundary_storage, LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA)
    from prismaquant.dev_mode import dev_mode_enabled
    from prismaquant.digests import DIRECT_ASCII_SPACED_STRICT, bytes_sha256hex
    from prismaquant.glm_routing_capture import capture_streamed_glm_routes
    from prismaquant.joint_adjoint_checkpoints import adjoint_space, boundary_entry_directory
    from prismaquant.joint_aura import source_execution_identity
    from prismaquant.joint_cost_quantum import build_quantum_source_runner
    from prismaquant.memory_management import CaptureMemoryGuard, enforce_device_envelope
    from prismaquant.native_moe_panel import routed_boundary_inputs, FIRST_SEQUENCE_ORIGINAL_SCHEMA
    from prismaquant.prismabuild_progress import report
    from prismaquant.source_generation import (_control, _resources, observe_original_source_execution,
        original_source_runtime, validate_original_source_runtime)
    from prismaquant.stage_a_selected_row_diagnostic import load_original_diagnostic_issued_context
    from prismaquant.tessera_calibration_cache import CaptureSourceAuthentication, require_original_source_authority

    if dev_mode_enabled():
        raise ValueError("original authority and legacy DEV/cache source capture are mutually exclusive")
    started = time.monotonic()
    authority_ref = {"path": args.original_authority, "sha256": args.original_authority_sha256}
    plan_ref = {"path": args.plan, "sha256": args.plan_sha256}
    prepared_ref = {"path": args.prepared, "sha256": args.prepared_sha256}
    session_preparation_ref = {"path": args.session_preparation, "sha256": args.session_preparation_sha256}
    _, control = _control(authority_ref, "original routed authority")
    _, plan = _control(plan_ref, "original routed final plan")
    bindings = plan.get("original_source", {})
    if bindings.get("authority") != authority_ref or bindings.get("prepared") != prepared_ref:
        raise ValueError("original routed inputs differ from the independently issued final plan")
    if plan.get("original_session_preparation") != session_preparation_ref:
        raise ValueError("original routed session preparation differs from its independently issued final plan")
    _, prepared = _control(prepared_ref, "original routed preparation")
    _, base = _control(bindings["base_plan"], "original routed base plan")
    if str(Path(args.output_root)) != base["output_root"]:
        raise ValueError("original routed output differs from the independently issued base plan")
    _, resources = _control(control["resources"], "original routed resources")
    resources = _resources(resources)
    _, expected_runtime = _control(control["runtime"], "original routed expected runtime")
    # CPU/cgroup-only control admission is real. No CUDA observation or source
    # material is permitted before the public authority requirement below.
    guard = CaptureMemoryGuard("cpu", host_floor_bytes=resources["host_floor_bytes"])
    if guard.cpu_cap_bytes != resources["cpu_bytes"] or guard.margin_bytes != resources["margin_bytes"]:
        raise ValueError("original routed CPU reservation differs from its exact finite envelope")

    def resource_check(label, *, reserve_bytes=0, reserve_device_bytes=0):
        return guard.check(label, reserve_bytes=reserve_bytes, reserve_device_bytes=reserve_device_bytes)

    resource_check.separates_cpu_and_device_reservations = guard.separate_reservations
    # Configure only validated process-local selector flags. Ambient choices
    # that differ from sealed arithmetic refuse; the helper only compares.
    _pin_original_arithmetic(expected_runtime)
    admitted = observe_original_source_execution(authority_ref, plan_ref, resource_check=resource_check)
    authority = require_original_source_authority(None, authority_ref, plan_ref, admitted)
    context = load_original_diagnostic_issued_context(session_preparation_ref,
        base_plan_input=bindings["base_plan"], authority=authority)
    if (context["base_plan"] != base or context["prepared"] != prepared
            or context["session_identity"] != admitted["session_identity"]
            or context["resources"] != resources):
        raise ValueError("original routed context differs from the independently issued owning session")
    _, producer = _control(authority["producer"], "original routed complete producer")
    _, source_paths = _control(authority["source_paths"], "original routed complete source paths")
    execution = context["execution"]
    storage_policy = check_boundary_storage(execution["boundary_storage"])
    if (storage_policy["schema"] != LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA
            or Path(storage_policy["directory"]) != boundary_entry_directory(adjoint_space(base["output_root"]))):
        raise ValueError("original routed session lacks its existing exact artifact policy/namespace")
    for name in ("source_identity_cache", "source_digest_cache", "canonical_capture"):
        if plan.get(name) is not None:
            raise ValueError("original routed capture accepts no legacy source/cache/canonical authority")
    if execution.get("source_derivative") is not None:
        raise ValueError("original routed capture cannot substitute a source derivative")
    if resources["gpu_bytes"] <= 0:
        raise ValueError("original routed capture requires its finite independently bound GPU envelope")
    owner = CaptureSourceAuthentication.qualified_original_material(base["model"], producer,
        publisher_input=authority["publisher"]["input"], publisher_id=authority["publisher"]["id"],
        publisher_revision=authority["publisher"]["revision"], readset_input=authority["readset"],
        source_paths=source_paths, max_material_bytes=resources["material_bytes"], resource_check=resource_check)
    runner = None
    entries = []
    total_artifact_bytes = 0
    root = Path(args.output_root)
    try:
        require_original_source_authority(owner, authority_ref, plan_ref, admitted)
        # This unchanged refusal precedes the GPU envelope, model/profile,
        # calibration material and output namespace. This cutover cannot lift it.
        owner.require_material_device("cuda")
        guard = CaptureMemoryGuard("cuda", device_bytes=resources["gpu_bytes"],
                                   host_floor_bytes=resources["host_floor_bytes"])
        resource_check.separates_cpu_and_device_reservations = guard.separate_reservations
        envelope = enforce_device_envelope("cuda", resources["gpu_bytes"], where="original routed capture")
        guard.check("before_original_routed_runner")
        torch.set_num_threads(1)
        ids, calibration = load_calibration_input(base["calibration_input"]["path"],
            expected_sha256=base["calibration_input"]["sha256"], n_samples=512, seqlen=512)
        if calibration != authority["calibration"] or calibration != prepared["calibration"]:
            raise ValueError("original routed calibration differs from its full bound draw")
        runner_config = {**plan, "execution": execution}
        runner = build_quantum_source_runner(runner_config, offload_folder=root / "offload",
            sealed_head_tensors=prepared["head_source"]["tensors"], source_authentication=owner)
        source = build_streamed_model_identity(runner, base["model"])
        if source != authority["source_model_identity"] or source_execution_identity(runner.model) != authority["source_execution"]:
            raise ValueError("original routed live source/execution differs from its issued authority")
        validate_original_source_runtime(original_source_runtime(runner, owner), authority["runtime"])
        manifest = {"schema": FIRST_SEQUENCE_ORIGINAL_SCHEMA, "scope": "first_sequence_original_capture",
                    "authority": authority_ref, "session": authority["session"], "calibration": calibration}

        with StreamedBoundaryArtifacts(storage_policy) as artifacts:
            artifacts.rebind(authority["session"], identity=admitted["session_identity"], n_probes=1,
                             check_memory=resource_check, owner_label="original-routed-capture")

            def publish(layer, result):
                nonlocal total_artifact_bytes
                if time.monotonic() - started >= resources["deadline_seconds"]:
                    raise RuntimeError("original routed capture exceeded its admitted runtime deadline")
                transported = {"source": "routed_boundary_capture", "boundary_metadata": result["metadata"], **result["tensors"]}
                routed_boundary_inputs(transported, calibration_receipt=calibration, capture_manifest=manifest,
                    device="cpu", source_model_identity=source, expected_original_authority=authority_ref,
                    expected_original_session=artifacts.session, original_source_authority=authority)
                resource_check("before_original_routed_serialization", reserve_bytes=resources["serialization_bytes"])
                # Two held byte representations (Torch buffer plus publication)
                # fit the declared serialization ceiling; no unbounded BytesIO.
                with _BoundedCaptureBuffer(resources["serialization_bytes"] // 2) as buffer:
                    torch.save(result, buffer)
                    raw = buffer.getvalue()
                    if total_artifact_bytes + len(raw) > resources["artifact_bytes"]:
                        raise RuntimeError("original routed artifacts exceed their admitted byte envelope")
                    target = root / f"layer-{layer:03d}.pt"
                    if not publish_new_bytes(target, raw):
                        raise ValueError(f"original routed capture output already exists: {target}")
                    total_artifact_bytes += len(raw)
                    entries.append({"layer": layer, "unit": result["metadata"]["unit"], "path": str(target),
                                    "sha256": bytes_sha256hex(raw), "bytes": len(raw)})
                report("capture", len(entries), unit=result["metadata"]["unit"])

            layers = capture_streamed_glm_routes(runner, ids, calibration=calibration, producer_source=producer,
                consume=publish, original_authority_input=authority_ref, original_plan_input=plan_ref,
                admitted_execution=admitted, artifact_owner=artifacts)
            prefetch = runner.context.prefetch_summary()
            runner.shutdown()
            runner = None
            final_material = owner.receipt()
            owner.close()
            receipt = {"schema": "prismaquant.glm_routed_capture_pass.v1", "status": "complete",
                "scope": "first_sequence_original_capture", "plan": plan_ref, "prepared": prepared_ref,
                "capture_authority": manifest, "layers": layers, "entries": entries,
                "calibration": calibration, "source_authentication": final_material,
                "source_prefetch": prefetch, "device_envelope": envelope,
                "source_owner_closed": owner._closed, "elapsed_s": time.monotonic() - started}
            raw = (DIRECT_ASCII_SPACED_STRICT.text(receipt) + "\n").encode()
            if total_artifact_bytes + len(raw) > resources["artifact_bytes"]:
                raise RuntimeError("original routed receipt exceeds its admitted artifact envelope")
            if not publish_new_bytes(root / "receipt.json", raw):
                raise ValueError("original routed capture receipt already exists")
    finally:
        try:
            if runner is not None:
                runner.shutdown()
        finally:
            runner = None
            owner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
