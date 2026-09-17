"""Prove the bounded PREPARE row's child environment before the loader runs.

Run INSIDE the admitted container, through ``tools.tessera_campaign_container``:

    python3 -m tools.tessera_campaign_container --spec "<sealed a2v4 spec>" \
        --cpu-only -- python3 -m tools.joint_prepare_startup_probe \
        --plan <sealed joint plan> --plan-sha256 <sha>

The joint pass refuses a bounded row whose environment is not the bounded
capture contract, and it does so at its first bounded step -- which on this
campaign is minutes into loading the model. This probe makes that refusal, and
the two memory budgets the guard holds apart, observable in seconds and without
a device: it asserts the environment, imports PrismaQuant, constructs the
configuration from the sealed plan, and constructs the capture guard. It reads
no model bytes and never touches CUDA, so it is a CPU probe by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", default=None)
    args = parser.parse_args(argv)

    from prismaquant import autoscale, memory_management, tessera_joint_aura

    # 1. The environment the pass will require at its first bounded step.
    observed_env = {name: os.environ.get(name)
                    for name in autoscale.BOUNDED_CAPTURE_ENV}
    autoscale.require_bounded_capture_environment(os.environ)

    # 2. Where PrismaQuant resolved from, which is what the launcher predicted.
    import prismaquant

    # 3. The sealed plan, through the pass's own config construction.
    plan_path = Path(args.plan)
    if args.plan_sha256 is not None:
        actual = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        if actual != args.plan_sha256:
            raise SystemExit(
                f"plan {plan_path} hashes to {actual}, not {args.plan_sha256}")
    config = tessera_joint_aura._load_plan(plan_path, args.plan_sha256,
                                           projection_runtime=False)

    # 4. The two budgets, held apart. ``check`` needs a device, so nothing here
    # calls it: the point is what the guard was constructed to enforce.
    guard = memory_management.CaptureMemoryGuard(
        "cuda", device_bytes=config["max_gpu_bytes"])
    record = {
        "schema": "prismaquant.joint_prepare_startup_probe.v1",
        "bounded_capture_env_observed": observed_env,
        "bounded_capture_env_required": dict(autoscale.BOUNDED_CAPTURE_ENV),
        "prismaquant_import": str(Path(prismaquant.__file__).resolve()),
        "python": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "plan": str(plan_path),
        "plan_sha256": args.plan_sha256,
        "plan_fields": {"max_gpu_bytes": config["max_gpu_bytes"],
                        "max_render_bytes": config["max_render_bytes"],
                        "min_free_gib": config["min_free_gib"],
                        "n_calib_samples": config["execution"]["n_calib_samples"],
                        "calib_seqlen": config["execution"]["calib_seqlen"]},
        "guard": {"cpu_cap_bytes": guard.cpu_cap_bytes,
                  "device_envelope_bytes": guard.device_bytes,
                  "aggregate_envelope_bytes": (guard.cpu_cap_bytes
                                               + guard.device_bytes),
                  "cgroup_scope": str(guard.scope),
                  "margin_bytes": guard.margin_bytes,
                  "host_floor_bytes": guard.host_floor_bytes,
                  "min_host_floor_bytes": memory_management.MIN_HOST_FLOOR_BYTES},
        "cgroup_memory_max": (guard.scope / "memory.max").read_text().strip(),
    }
    assert record["bounded_capture_env_observed"] == record["bounded_capture_env_required"]
    assert record["guard"]["host_floor_bytes"] >= record["guard"]["min_host_floor_bytes"]
    print(json.dumps(record, indent=1, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
