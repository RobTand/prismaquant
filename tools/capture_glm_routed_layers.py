"""PB-admitted single traversal for GLM native routed inputs, layers 3..44.

Runs in the Stage B source-derivative environment. It binds the complete
512x512 calibration and captures its first sequence at every routed boundary;
this is not a full-draw quality measurement. No new source or activation store
is introduced: the runner owns source prefetch, the canonical capture owner
verifies each consumed shard before exposing a tensor, and each output is the
existing raw native boundary payload.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import resource
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
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
                        "path": str(target), "sha256": hashlib.sha256(raw).hexdigest(),
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


if __name__ == "__main__":
    raise SystemExit(main())
