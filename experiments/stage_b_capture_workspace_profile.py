"""Measure the Stage B capture workspace on one real layer quantum (PQ #1151).

Runs a layer quantum exactly as ``python3 -m prismaquant.joint_cost_quantum``
does -- the same arguments, the same identity checks, the same head,
checkpoint load, source load and retained window -- with two differences:

* ``PRISMAQUANT_STAGE_B_WORKSPACE_PROFILE`` is set, so probe 0's spill capture
  runs the measurement ladder of :mod:`prismaquant.stage_b_workspace_profile`
  and stops the quantum once the receipt is written;
* the record's ``output_space`` is moved under ``--profile-root``, so the
  checkpoint manifest and the failure counters of this measurement never
  touch the campaign's own layer directory. The record file is only read.

Usage: the quantum's own argument list, plus ``--profile-root DIR``,
``--profile-out FILE.json`` and optionally ``--profile-groups N``. Submit it
through PrismaBuild with the row's own residency, image and phases.

Exit status 0 means the receipt was written; its path and sha256 are the last
line of stdout.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path


def moved_output_space(record, root):
    """``record`` with every ``output_space`` path under ``root``."""
    root = Path(root)
    moved = copy.deepcopy(record)
    layer_root = root / f"layer-{int(record['layer']):03d}"
    moved["output_space"] = {
        "root": str(layer_root),
        "checkpoint_dir": str(layer_root / "checkpoints"),
        "cost_payload": str(layer_root / "cost.pkl"),
        "counters": str(layer_root / "counters.json"),
        "results": str(layer_root / "results.json"),
    }
    return moved


def main(argv=None) -> int:
    from prismaquant import joint_cost_quantum as quantum
    from prismaquant.stage_b_workspace_profile import (
        DEFAULT_GROUPS, GROUPS_ENV, PROFILE_ENV, CaptureWorkspaceProfiled)

    parser = quantum.build_parser()
    parser.add_argument("--profile-root", required=True,
                        help="directory the measurement's own output space goes under")
    parser.add_argument("--profile-out", required=True,
                        help="absolute path of the JSON receipt")
    parser.add_argument("--profile-groups", type=int, default=DEFAULT_GROUPS)
    args = parser.parse_args(argv)
    if args.device != "cuda":
        parser.error("the profile measures the GPU hot path; --device must be cuda")
    out = Path(args.profile_out)
    root = Path(args.profile_root)
    if not out.is_absolute() or not root.is_absolute():
        parser.error("--profile-out and --profile-root must be absolute paths")
    if out.exists():
        parser.error(f"{out} already exists; a receipt is never overwritten")
    campaign_root = Path(args.output_root).resolve()
    if root.resolve() == campaign_root or campaign_root in root.resolve().parents:
        parser.error("--profile-root must lie outside the campaign's output root")
    os.environ[PROFILE_ENV] = str(out)
    os.environ[GROUPS_ENV] = str(args.profile_groups)

    quantum.require_dev_mode("stage_b_capture_workspace_profile")
    try:
        record, adjoint_slice = quantum.verify_quantum_identity(
            quantum_path=args.quantum, quantum_sha256=args.quantum_sha256,
            plan_path=args.plan, plan_sha256=args.plan_sha256,
            prepared_path=args.prepared, prepared_sha256=args.prepared_sha256,
            adjoint_path=args.adjoint_slice, adjoint_sha256=args.adjoint_slice_sha256,
            output_root=args.output_root)
        quantum.require_chain_readset(record, data_manifest_sha256=args.data_manifest_sha256)
    except quantum.QuantumIdentityRefused as exc:
        print(f"{quantum.IDENTITY_REFUSED_MARKER}: {exc}", flush=True)
        return quantum.EXIT_IDENTITY_REFUSED
    from prismaquant.staged_tier_policy import activate_staged_tier_policy
    from prismaquant.tessera_joint_aura import _load_plan

    config = _load_plan(args.plan, args.plan_sha256, defer_pool_reads=True)
    if Path(config["output_root"]).resolve() != campaign_root:
        print(f"{quantum.IDENTITY_REFUSED_MARKER}: plan output_root "
              f"{config['output_root']} is not --output-root {args.output_root}", flush=True)
        return quantum.EXIT_IDENTITY_REFUSED
    allowed = activate_staged_tier_policy(args.allowed_tiers)
    print(f"[STAGED-TIER] bulk inputs serve from {','.join(sorted(allowed))}", flush=True)
    record = moved_output_space(record, root)
    print(f"[WORKSPACE-PROFILE] output space {record['output_space']['root']}; "
          f"receipt {out}", flush=True)
    try:
        quantum.run_layer_quantum(
            config, record=record, adjoint_slice=adjoint_slice, plan_sha256=args.plan_sha256,
            prepared={"path": str(args.prepared), "sha256": args.prepared_sha256},
            output_root=args.output_root,
            data_manifest_sha256=args.data_manifest_sha256, resume=args.resume,
            progress_grace=quantum.progress_grace_stamps(args.progress_grace_derivation))
    except quantum.QuantumIdentityRefused as exc:
        print(f"{quantum.IDENTITY_REFUSED_MARKER}: {exc}", flush=True)
        return quantum.EXIT_IDENTITY_REFUSED
    except BaseException as error:
        # A wrapper between the capture and here may re-raise the stop as
        # another type; the receipt is written either way.
        done = CaptureWorkspaceProfiled.found_in(error)
        if done is None:
            raise
        print(json.dumps({"status": "profiled", "profile": done.path,
                          "sha256": done.sha256}), flush=True)
        return 0
    print("the quantum ended without reaching probe 0's spill capture; no receipt",
          flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
