#!/usr/bin/env python3
"""Publish the selected, closed Tessera dense+expert wire manifest.

This consumes a completed joint handoff and a final allocation. It does not
encode, interpolate bytes, change a source cache, or qualify a serving lane.
The exporter's --cached-units intake remains the current-byte verifier.

Head journaling is opt-in: --head-checkpoint supplies the existing reader's
checkpoint path, and --head-resume reuses its identity-bound, reverified
prefix. Neither option declares a synthesis phase or changes manifest bytes.
The paired --head-progress-phase / --head-progress-allowance-s options report
through the reader's existing cadence under a phase and stall allowance that
the submitting PB action already declared. Without them, no phase is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cluster_campaign import _atomic_write_new_bytes
from prismaquant.footprint import whole_artifact_budget_from_assignment_payload
from prismaquant.schemas import strict_json_loads
from prismaquant.layer_config import (
    canonicalize_assignment, layer_config_metadata, validate_layer_config_payload)
from prismaquant.tessera_export_lane import (read_cached_unit_bundle,
                                             selected_cached_units_manifest)
from prismaquant.tessera_joint_aura import (load_measured_anchor_input,
                                          _head_walk_worker_count)
from tessera.cached_unit import CACHE_SCHEMA


def _bound(path: str, digest: str, label: str) -> bytes:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{label} SHA-256 differs from the selected receipt")
    return raw


def _bind_selected_assignment(assignment_path: str, expected_digest: str
                              ) -> tuple[dict, dict[str, str], dict]:
    """Bind the selected assignment by the allocator's own digest primitive.

    The identity of a selected assignment is
    ``footprint.assignment_serialization_sha256`` over the canonical
    unit-to-format mapping (meta excluded) -- the same digest the allocator
    stamps as ``selection_assignment_sha256``. Raw file bytes are NOT the
    identity: allocator stamps, meta blocks and JSON re-serialization all
    change bytes without changing the selection. The file is read ONCE and
    the assignment and metadata returned are parsed from the bytes that were
    checked, never from a second read. Returns ``(stamp, assignment, metadata)``.
    """
    payload = json.loads(Path(assignment_path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"selected assignment {assignment_path} is not a JSON object")
    validate_layer_config_payload(payload, assignment_path)
    assignment = canonicalize_assignment(payload)
    stamp = whole_artifact_budget_from_assignment_payload(
        payload, where=f"selected assignment {assignment_path}",
        assignment=assignment)
    if stamp is None:
        raise ValueError(
            f"selected assignment {assignment_path} carries no whole-artifact "
            "budget stamp; refusing unstamped selection")
    stamped = stamp["selection_assignment_sha256"]
    if expected_digest != stamped:
        raise ValueError(
            f"selected assignment flag names {expected_digest} but the "
            f"assignment's budget stamp binds {stamped}")
    return stamp, assignment, layer_config_metadata(payload)


def _bind_plan_encoder_reuse(plan_path: str | None, plan_sha256: str | None,
                             joint: dict):
    """The original plan's historical encoder allowance, or None (#1488).

    The loader refuses a recorded encoder seal the installed package cannot
    re-derive unless the plan that priced the checkpoint named it. This reads
    that plan by digest and admits it only when it is the plan the handoff
    was joined under -- the same digest and the same original inputs -- as
    the allocation handoff does (``tessera_joint_allocation``). Nothing is
    inferred: without ``--plan`` the loader's strict default holds.
    """
    if plan_path is None:
        return None
    plan = json.loads(_bound(plan_path, plan_sha256, "joint plan"))
    if joint.get("plan_sha256") != plan_sha256:
        raise ValueError(
            f"joint handoff was joined under plan {joint.get('plan_sha256')}, "
            f"not the supplied plan {plan_sha256}")
    if plan.get("inputs") != joint.get("inputs"):
        raise ValueError(
            "joint plan original inputs differ from the handoff's bound inputs")
    return plan.get("historical_encoder_reuse")


def _split_child_units(assignment: dict[str, str], child_path: str | None,
                       child_sha256: str | None) -> tuple[dict[str, str], dict | None]:
    """Leave out the selected units a byte-bound child cache supplies (#1641).

    A handoff's census roster covers only the units its campaign measured; a
    selection may also name units whose wires live in another cached-units
    child (GLM-5.3's MTP layer 45). Those units are removed here and composed
    back by ``tools/compose_tessera_cached_units.py``, so the census rule
    still judges everything the handoff supplies. A child must name only
    selected, non-BF16 units: a passthrough has no wire to supply. Format and
    identity are verified at export intake (``verify_cached_unit``), as for
    every other cached unit. Returns ``(handoff_assignment, child_binding)``.
    """
    if child_path is None:
        return assignment, None

    def duplicate(key):
        return ValueError(f"child manifest repeats key {key!r}")

    document = strict_json_loads(_bound(child_path, child_sha256, "child manifest"),
                                 duplicate=duplicate)
    units = document.get("units") if isinstance(document, dict) else None
    if not isinstance(units, dict) or not units:
        raise ValueError(f"child manifest {child_path} names no units")
    missing = sorted(set(units) - set(assignment))
    if missing:
        raise ValueError(
            f"child manifest names {len(missing)} unit(s) not in the selected "
            f"assignment (first: {missing[0]})")
    passthrough = sorted(name for name in units if assignment[name] == "BF16")
    if passthrough:
        raise ValueError(
            f"child manifest supplies {len(passthrough)} BF16 passthrough "
            f"unit(s), which have no wire (first: {passthrough[0]})")
    remainder = {name: fmt for name, fmt in assignment.items() if name not in units}
    return remainder, {"path": str(Path(child_path).resolve()), "sha256": child_sha256,
                       "units": len(units)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--handoff-sha256", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--assignment-sha256", required=True,
                        help="the assignment's owner digest (its stamped selection_assignment_sha256), not the file bytes")
    parser.add_argument("--out", required=True)
    parser.add_argument("--head-walk-workers", type=int, default=None,
                        help="explicit bounded head I/O concurrency (1-16), independent of PB's CPU reservation")
    parser.add_argument("--head-checkpoint", default=None,
                        help="opt-in existing head-walk journal; declare this path as writable PB output")
    parser.add_argument("--head-resume", action="store_true",
                        help="resume that journal through the reader's existing identity and drift checks")
    parser.add_argument("--head-progress-phase", default=None,
                        help="opt-in head progress under a phase already declared by the submitting PB action")
    parser.add_argument("--head-progress-allowance-s", type=float, default=None,
                        help="that phase's declared positive stall allowance; required with --head-progress-phase")
    parser.add_argument("--read-paths-out", help="new JSON file listing all rooted export inputs for PB staging")
    parser.add_argument("--catalog-extension")
    parser.add_argument("--catalog-extension-sha256")
    parser.add_argument("--producer-packages", help="bound JSON mapping exact encoder seals to archived packages")
    parser.add_argument("--producer-packages-sha256")
    parser.add_argument("--plan", default=None,
                        help="the original joint plan the handoff was joined under; only its historical_encoder_reuse is read")
    parser.add_argument("--plan-sha256", default=None)
    parser.add_argument("--child-manifest", default=None,
                        help="a cached-units child supplying selected units outside the handoff roster; compose it afterwards")
    parser.add_argument("--child-manifest-sha256", default=None)
    parser.add_argument("--research-proposal", default=None,
                        help="explicit sampled-pilot research proposal for validation export")
    parser.add_argument("--research-proposal-sha256", default=None)
    args = parser.parse_args(argv)
    if args.head_walk_workers is not None:
        _head_walk_worker_count(args.head_walk_workers)
    if args.head_resume and not args.head_checkpoint:
        raise ValueError("head resume requires --head-checkpoint")
    if (args.head_progress_phase is None) != (args.head_progress_allowance_s is None):
        raise ValueError("head progress phase and allowance must be supplied together")
    if args.head_progress_phase is not None:
        if not args.head_progress_phase.strip():
            raise ValueError("head progress phase must be nonempty")
        if (not math.isfinite(args.head_progress_allowance_s)
                or args.head_progress_allowance_s <= 0):
            raise ValueError("head progress allowance must be finite and positive")
    if bool(args.research_proposal) != bool(args.research_proposal_sha256):
        raise ValueError('research proposal path and SHA-256 must be supplied together')
    extension = None
    packages = None
    if args.plan and args.research_proposal:
        raise ValueError("a research proposal binds its own pilot plan; do not also pass --plan")
    for name in ("plan", "child_manifest", "catalog_extension", "producer_packages"):
        if bool(getattr(args, name)) != bool(getattr(args, name + "_sha256")):
            raise ValueError(name + " path and SHA-256 must be supplied together")
    if bool(args.catalog_extension) != bool(args.producer_packages):
        raise ValueError("rooted cache needs both extension and producer package bindings")
    if args.catalog_extension:
        extension = {"path": args.catalog_extension, "sha256": args.catalog_extension_sha256}
        _bound(args.catalog_extension, args.catalog_extension_sha256, "catalog extension")
        packages = json.loads(_bound(args.producer_packages, args.producer_packages_sha256, "producer packages"))
    pilot_binding = {'path': args.handoff, 'sha256': args.handoff_sha256}
    research = None
    if args.research_proposal:
        from prismaquant.tessera_sampled_stack_proposal import (
            bind_pilot_from_inputs, require_research_proposal_assignment)
        research = json.loads(_bound(args.research_proposal,
                                    args.research_proposal_sha256, 'research proposal'))
        if research.get('input_bindings', {}).get('pilot_joint_cost') != pilot_binding:
            raise ValueError('research proposal does not bind the supplied pilot cost')
        handoff, _plan = bind_pilot_from_inputs(joint_binding=pilot_binding,
            plan_binding=research['input_bindings']['pilot_plan'])
    else:
        handoff = pickle.loads(_bound(args.handoff, args.handoff_sha256, "joint handoff"))
    _stamp, assignment, metadata = _bind_selected_assignment(
        args.assignment, args.assignment_sha256)
    if research is not None:
        binding = require_research_proposal_assignment(research, assignment,
                                                        pilot_joint_binding=pilot_binding)
        if metadata.get('sampled_joint_proposal') != {
                'schema': binding['schema'],
                'proposal_sha256': args.research_proposal_sha256,
                'selected_assignment_sha256': binding['selected_assignment_sha256']}:
            raise ValueError('selected cache assignment lacks exact research proposal marker')
    elif 'sampled_joint_proposal' in metadata:
        raise ValueError('selected cache pilot assignment requires explicit research proposal')
    assignment, child = _split_child_units(
        assignment, args.child_manifest, args.child_manifest_sha256)
    provenance = handoff.get("provenance", {})
    joint = provenance.get("tessera_joint_anchors", {})
    inputs = joint.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("joint handoff has no bound original campaign inputs")
    reuse = _bind_plan_encoder_reuse(args.plan, args.plan_sha256, joint)
    # This reader declares no phase. Report only the caller's explicit phase
    # and allowance, or none; never invent an undeclared synthesis phase (#678).
    data = load_measured_anchor_input(inputs, verify_payloads=False,
                                      require_existing_renders=True,
                                      historical_encoder_reuse=reuse,
                                      progress_phase=args.head_progress_phase,
                                      progress_allowance_s=args.head_progress_allowance_s,
                                      head_walk_workers=args.head_walk_workers,
                                      head_checkpoint=args.head_checkpoint,
                                      head_resume=args.head_resume)
    manifest = selected_cached_units_manifest(
        assignment, metadata, handoff, data,
        schema="tessera.cached_units.v2" if extension else CACHE_SCHEMA,
        research_proposal=research, catalog_extension=extension, producer_packages=packages)
    directory = Path(provenance["wire_dir"]).resolve()
    out = Path(args.out)
    if out.is_symlink() or (extension is None and out.resolve().parent != directory):
        raise ValueError("selected manifest must be a new file in the original wire directory")
    bundle = read_cached_unit_bundle(
        manifest, directory, set(manifest["units"]), manifest["source"])
    raw = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    paths_raw = None
    if args.read_paths_out:
        if not extension:
            raise ValueError("read-paths output requires a rooted selected manifest")
        from prismaquant.joint_catalog_extension import selected_cache_read_paths
        paths = sorted({str(out.resolve()), *selected_cache_read_paths(manifest)})
        paths_raw = (json.dumps({"schema": "prismaquant.selected_cache_read_paths.v1", "paths": paths},
                                sort_keys=True, indent=2) + "\n").encode()
    _atomic_write_new_bytes(out, raw)
    if paths_raw is not None:
        _atomic_write_new_bytes(Path(args.read_paths_out), paths_raw)
    print(json.dumps({"schema": "prismaquant.tessera_selected_cache_handoff.v1",
                      "status": "research_wires_only", "manifest": str(out.resolve()),
                      "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                      "assignment_sha256": args.assignment_sha256,
                      "handoff_sha256": args.handoff_sha256,
                      "plan_sha256": args.plan_sha256,
                      "child_manifest": child,
                      "research_proposal_sha256": args.research_proposal_sha256,
                      "catalog_extension": extension,
                      "producer_packages_sha256": args.producer_packages_sha256,
                      "units": len(manifest["units"]),
                      "encoder_source_proof_mode": bundle.encoder_source_proof_mode,
                      "warnings": bundle.warnings,
                      "export_qualified": False, "serving_qualified": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
