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

``--stop-after-chain`` (PQ #1163) measures the chain phase instead: every
chain roll is admitted, measured in-process (CUDA allocated and reserved
peaks, wall time, cgroup peak, lowest MemAvailable) and the quantum stops
after its chain with the receipt at ``--profile-out``. A plan whose retained
budget prices no chain phase is admitted under an explicitly declared owner:
the largest workspace the budget admits beside the chain's fixed owners,
stated as declared in the receipt.

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


def declare_chain_owner(config, adjoint_slice, chain_layers):
    """Admit the chain under a declared owner when the plan prices none.

    The owner is the largest chain workspace the plan's own retained budget
    admits beside the chain phase's resident owners at their declared caps
    (:meth:`RetainedWindowBudget.declared_chain_residents`): the smaller of
    the device envelope less the device-side residents, and the physical
    budget less its margin less every resident. It is recorded in the
    receipt as declared. A plan that already prices the chain keeps its
    owner.
    """
    from prismaquant.joint_adjoint_slices import chain_regime_of
    from prismaquant.joint_retained_window_plan import (
        ChainRetainedWindowBudget, RetainedWindowBudget)
    from prismaquant.stage_b_workspace_profile import CHAIN_OWNER_ENV

    retained = config["execution"]["retained_operator_windows"]
    budget = RetainedWindowBudget.from_dict(retained["budget"])
    regime = chain_regime_of(adjoint_slice["run_identity"])
    if isinstance(budget, ChainRetainedWindowBudget):
        owner = {"source": "plan", "chain_workspace_reserve_bytes":
                 budget.chain_workspace_bytes(regime["batch_size"],
                                              fused=regime["probe_fusion"]),
                 "chain_layers": list(budget.chain_layers)}
    else:
        residents = budget.declared_chain_residents(
            retained["source_reserve_bytes"],
            loading_bytes=retained["source_loading_reserve_bytes"])
        device_limit = config["max_gpu_bytes"]
        bound = budget.physical_limit_bytes - budget.safety_margin_bytes
        headroom = min(device_limit - residents["device_resident_bytes"],
                       bound - residents["device_resident_bytes"]
                       - residents["host_committed_bytes"])
        if headroom <= 0:
            raise SystemExit(f"the chain phase's declared resident owners {residents} leave "
                             f"no workspace under the {device_limit}-byte device envelope "
                             f"and {bound} bytes of physical budget less margin")
        layers = sorted(int(layer) for layer in chain_layers)
        planned = ChainRetainedWindowBudget(
            **{name: getattr(budget, name) for name in RetainedWindowBudget.__dataclass_fields__},
            chain_batch_size=regime["batch_size"], chain_probe_fusion=regime["probe_fusion"],
            chain_workspace_reserve_bytes=headroom,
            chain_device_resident_bytes=residents["device_resident_bytes"],
            chain_host_committed_bytes=residents["host_committed_bytes"],
            chain_device_limit_bytes=device_limit, chain_layers=tuple(layers))
        planned.require_chain_fits()
        retained["budget"] = planned.as_dict()
        owner = {"source": "declared", "chain_workspace_reserve_bytes": headroom,
                 "chain_layers": layers, "regime": dict(regime), **residents,
                 "device_limit_bytes": device_limit, "bound_bytes": bound,
                 "basis": ("the largest chain workspace the plan's retained budget admits "
                           "beside the chain phase's resident owners at their declared "
                           "caps: min(device envelope - device residents, physical_limit_bytes "
                           "- safety_margin_bytes - device residents - host residents)")}
    os.environ[CHAIN_OWNER_ENV] = json.dumps(owner, sort_keys=True)
    print(f"[CHAIN-PROFILE] chain admitted under {json.dumps(owner, sort_keys=True)}",
          flush=True)
    return owner


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
    parser.add_argument("--stop-after-chain", action="store_true",
                        help="measure every chain roll and stop after the chain (PQ #1163)")
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
    if args.stop_after_chain:
        from prismaquant.stage_b_workspace_profile import CHAIN_PROFILE_ENV
        os.environ[CHAIN_PROFILE_ENV] = str(out)
    else:
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
    if args.stop_after_chain:
        if not record["adjoint"]["chain_layers"]:
            parser.error("--stop-after-chain needs a quantum whose record walks a chain")
        declare_chain_owner(config, adjoint_slice, record["adjoint"]["chain_layers"])
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
    print("the quantum ended without reaching its measurement; no receipt", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
