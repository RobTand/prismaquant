#!/usr/bin/env python3
"""Publish the selected, closed Tessera dense+expert wire manifest.

This consumes a completed joint handoff and a final allocation. It does not
encode, interpolate bytes, change a source cache, or qualify a serving lane.
The exporter's --cached-units intake remains the current-byte verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cluster_campaign import _atomic_write_new_bytes
from prismaquant.layer_config import load_assignment, read_layer_config_metadata
from prismaquant.tessera_export_lane import selected_cached_units_manifest
from prismaquant.tessera_joint_aura import load_measured_anchor_input
from tessera.cached_unit import CACHE_SCHEMA, CachedUnitBundle


def _bound(path: str, digest: str, label: str) -> bytes:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{label} SHA-256 differs from the selected receipt")
    return raw


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--handoff-sha256", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--read-paths-out", help="new JSON file listing all rooted export inputs for PB staging")
    parser.add_argument("--catalog-extension")
    parser.add_argument("--catalog-extension-sha256")
    parser.add_argument("--producer-packages", help="bound JSON mapping exact encoder seals to archived packages")
    parser.add_argument("--producer-packages-sha256")
    parser.add_argument("--research-proposal", default=None,
                        help="explicit sampled-pilot research proposal for validation export")
    parser.add_argument("--research-proposal-sha256", default=None)
    args = parser.parse_args(argv)
    if bool(args.research_proposal) != bool(args.research_proposal_sha256):
        raise ValueError('research proposal path and SHA-256 must be supplied together')
    extension = None
    packages = None
    for name in ("catalog_extension", "producer_packages"):
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
    _bound(args.assignment, args.assignment_sha256, "selected assignment")
    assignment = load_assignment(args.assignment)
    metadata = read_layer_config_metadata(args.assignment)
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
    provenance = handoff.get("provenance", {})
    joint = provenance.get("tessera_joint_anchors", {})
    inputs = joint.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("joint handoff has no bound original campaign inputs")
    # A reader, not the synthesis stage: it declares no PrismaBuild phase, so
    # it reports under none rather than under a name nothing declared (#678).
    data = load_measured_anchor_input(inputs, verify_payloads=False,
                                      require_existing_renders=True,
                                      progress_phase=None)
    manifest = selected_cached_units_manifest(
        assignment, metadata, handoff, data,
        schema="tessera.cached_units.v2" if extension else CACHE_SCHEMA,
        research_proposal=research, catalog_extension=extension, producer_packages=packages)
    directory = Path(provenance["wire_dir"]).resolve()
    out = Path(args.out)
    if out.is_symlink() or (extension is None and out.resolve().parent != directory):
        raise ValueError("selected manifest must be a new file in the original wire directory")
    CachedUnitBundle(manifest, directory, set(manifest["units"]), manifest["source"])
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
                      "research_proposal_sha256": args.research_proposal_sha256,
                      "catalog_extension": extension,
                      "producer_packages_sha256": args.producer_packages_sha256,
                      "units": len(manifest["units"]),
                      "export_qualified": False, "serving_qualified": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
