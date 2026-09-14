#!/usr/bin/env python3
"""Recompute a Tessera census checkpoint seal and write its compact per-unit roster.

The census checkpoint manifest (``cost.anchors.json``) is gigabytes of JSON,
almost all of it the sealed identity.  This reads it once, recomputes the
identity digest against the stored one, and writes only what a selected-wire
manifest checks per unit (source weight and Hessian identity) plus the
digest every checkpoint journal envelope must carry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cluster_campaign import _atomic_write_new_bytes
from prismaquant.tessera_census_cache import seal_roster


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="census cost.anchors.json")
    parser.add_argument("--out", required=True, help="new roster JSON path")
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    digest = hashlib.sha256()
    with manifest_path.open("rb") as handle:
        raw = handle.read()
    digest.update(raw)
    manifest = json.loads(raw)
    del raw
    roster = seal_roster(manifest)
    roster["checkpoint_manifest"] = {"path": str(manifest_path), "sha256": digest.hexdigest(),
                                     "bytes": manifest_path.stat().st_size}
    encoded = (json.dumps(roster, sort_keys=True, indent=1, allow_nan=False) + "\n").encode()
    out = Path(args.out)
    _atomic_write_new_bytes(out, encoded)
    sha = hashlib.sha256(encoded).hexdigest()
    _atomic_write_new_bytes(out.with_name(out.name + ".sha256"), f"{sha}  {out.name}\n".encode())
    print(json.dumps({"roster": str(out.resolve()), "roster_sha256": sha,
                      "identity_sha256": roster["identity_sha256"],
                      "units": len(roster["units"]),
                      "checkpoint_manifest_sha256": roster["checkpoint_manifest"]["sha256"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
