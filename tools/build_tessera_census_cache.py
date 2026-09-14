#!/usr/bin/env python3
"""Close the census wires an assignment selects into a ``tessera.cached_units.v1`` manifest.

Input is either an allocator ``layer_config.json`` (``--layer-config``) or a
hand-written uniform control (``--uniform-format``), for which the layer
config is written here with the allocator's own expert-projection block.
No joint handoff is read, nothing is encoded, and every referenced blob is
re-hashed.  Research wires only: this qualifies neither export nor serving.

The exporter reads blobs from its manifest file's parent directory, so the
manifest written to ``--out-dir`` is a record; an export copies it into the
census wire directory (the ``wire_dir`` printed in the summary) unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cluster_campaign import _atomic_write_new_bytes
from prismaquant.layer_config import (
    LAYER_CONFIG_META_KEY, load_assignment, read_layer_config_metadata,
)
from prismaquant.tessera_census_cache import (
    census_layer_config, census_selected_cached_units_manifest,
    load_selected_wire_records, selected_census_assignment, uniform_assignment,
)
from tessera.cached_unit import (
    CACHE_SCHEMA, ENCODING_INPUT_SCHEMA, INPUT_SCHEMA, CachedUnitBundle,
)


def _bound(path: str, digest: str, label: str) -> bytes:
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != digest:
        raise SystemExit(f"{label} {path}: SHA-256 {actual}, expected {digest}")
    return raw


def _write(path: Path, raw: bytes) -> str:
    _atomic_write_new_bytes(path, raw)
    return hashlib.sha256(raw).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costs", required=True)
    parser.add_argument("--costs-sha256", required=True)
    parser.add_argument("--census", required=True, help="census.json (unit_shapes)")
    parser.add_argument("--census-sha256", required=True)
    parser.add_argument("--roster", required=True, help="seal_tessera_census_identity.py output")
    parser.add_argument("--roster-sha256", required=True)
    parser.add_argument("--checkpoint-parts", required=True,
                        help="the census cost.anchors.json.parts journal directory")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--layer-config")
    choice.add_argument("--uniform-format")
    parser.add_argument("--layer-config-sha256")
    parser.add_argument("--target-profile", help="stamped into a uniform layer config")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    started = time.monotonic()
    out = Path(args.out_dir)

    cost = pickle.loads(_bound(args.costs, args.costs_sha256, "cost table"))
    census = json.loads(_bound(args.census, args.census_sha256, "census"))
    roster = json.loads(_bound(args.roster, args.roster_sha256, "seal roster"))
    shas = {"costs": args.costs_sha256, "census": args.census_sha256, "roster": args.roster_sha256}

    if args.uniform_format:
        if not args.target_profile:
            raise SystemExit("--uniform-format needs --target-profile")
        from prismaquant import format_registry as fr
        written = uniform_assignment(cost, args.uniform_format)
        config = census_layer_config(
            cost, written, target_profile=args.target_profile,
            entry_for=lambda fmt: fr.get_format(fmt).autoround_config())
        layer_config = out / "layer_config.json"
        shas["layer_config"] = _write(
            layer_config, (json.dumps(config, indent=2, allow_nan=False) + "\n").encode())
        if load_assignment(layer_config) != written:
            raise SystemExit("written layer config does not parse back to the uniform assignment")
    else:
        if not args.layer_config_sha256:
            raise SystemExit("--layer-config needs --layer-config-sha256")
        layer_config = Path(args.layer_config)
        _bound(args.layer_config, args.layer_config_sha256, "layer config")
        shas["layer_config"] = args.layer_config_sha256
    assignment = load_assignment(layer_config)
    metadata = read_layer_config_metadata(layer_config)

    selected, *_ = selected_census_assignment(assignment, metadata, cost)
    records = load_selected_wire_records(Path(args.checkpoint_parts), selected,
                                         identity_sha256=roster["identity_sha256"],
                                         workers=args.workers)
    loaded = time.monotonic()
    manifest = census_selected_cached_units_manifest(
        assignment, metadata, cost, roster, census["unit_shapes"], records,
        input_schema=INPUT_SCHEMA, encoding_input_schema=ENCODING_INPUT_SCHEMA,
        cache_schema=CACHE_SCHEMA, hash_workers=args.workers)
    wire_dir = Path(cost["provenance"]["wire_dir"]).resolve()
    CachedUnitBundle(manifest, wire_dir, set(manifest["units"]), manifest["source"])
    raw = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    shas["manifest"] = _write(out / "manifest.json", raw)
    summary = {
        "schema": "prismaquant.tessera_census_cache_build.v1",
        "status": "research_wires_only", "export_qualified": False, "serving_qualified": False,
        "sha256": shas, "units": len(manifest["units"]),
        "unselected_bf16": sorted(n for n, f in selected.items() if f == "BF16"),
        "formats": dict(sorted(Counter(selected.values()).items())),
        "blob_bytes": sum(int(r["blob_bytes"]) for r in manifest["units"].values()),
        "wire_dir": str(wire_dir), "seal_identity_sha256": roster["identity_sha256"],
        "stack_formats": metadata.get("tessera_expert_stack_formats"),
        "exporter_reads_blobs_from": "the manifest file's parent directory; copy manifest.json "
                                     "into wire_dir unchanged before export",
        "seconds": {"journals": round(loaded - started, 1),
                    "total": round(time.monotonic() - started, 1)},
        "layer_config_meta_keys": sorted(metadata),
    }
    shas["summary"] = _write(out / "summary.json",
                             (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({key: summary[key] for key in ("units", "formats", "blob_bytes", "sha256",
                                                     "seconds")}, sort_keys=True))
    print(json.dumps({"summary_sha256": shas["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
