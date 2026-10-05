#!/usr/bin/env python3
"""Compose existing disjoint Tessera Hessian references without copying H.

The installed Tessera reader owns the collection schema, child authentication
and combined capture seal. This tool only authors a new closed document from
already published v1 references; it never opens Hessian payloads.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cost_stage_checkpoint import publish_new_bytes
from prismaquant.tessera_reuse_authority import CANONICAL_CAPTURE
from prismaquant.digests import DIRECT_ASCII_SPACED_LAX


def collection_document(paths) -> dict:
    from tessera.hessian_capture import (COLLECTION_SCHEMA, ReferenceHessians,
                                         capture_sha256_from_units)

    names = sorted(str(Path(path).resolve()) for path in paths)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Hessian collection needs at least two distinct references")
    with ExitStack() as owners:
        references, commitments, provenance = [], {}, None
        for path in names:
            child = owners.enter_context(ReferenceHessians(path, canonical_capture=CANONICAL_CAPTURE))
            if provenance is None:
                provenance = child.provenance
            elif child.provenance != provenance:
                raise ValueError(f"Hessian reference {path} has different calibration")
            overlap = set(commitments) & set(child)
            if overlap:
                raise ValueError(f"Hessian references overlap on {sorted(overlap)[0]}")
            commitments.update(child.committed_units())
            references.append({"path": path, "sha256": child.document_sha256})
        return {"schema": COLLECTION_SCHEMA, "references": references,
                "units": sorted(commitments),
                "capture_sha256": capture_sha256_from_units(provenance, commitments)}


def publish_collection(paths, output: str | Path) -> dict:
    from tessera import hessian_capture as reader
    from tessera.hessian_capture import ReferenceHessianCollection

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Hessian collection output exists: {output}")
    document = collection_document(paths)
    raw = (json.dumps(document, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate the exact bytes under the producer's reader before publication.
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".collection.references.json") as temp:
        temp.write(raw)
        temp.flush()
        with ReferenceHessianCollection(temp.name, canonical_capture=CANONICAL_CAPTURE) as checked:
            if sorted(checked) != document["units"]:
                raise ValueError("Tessera reader disagrees with the authored collection")
    if not publish_new_bytes(output, raw):
        raise FileExistsError(f"Hessian collection output exists: {output}")
    with ReferenceHessianCollection(output, canonical_capture=CANONICAL_CAPTURE) as checked:
        return {"path": str(output.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                "capture_sha256": document["capture_sha256"],
                "reference_binding": checked.binding(),
                "units": len(checked), "reader_file": str(Path(reader.__file__).resolve()),
                "reader_sha256": hashlib.sha256(Path(reader.__file__).read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(DIRECT_ASCII_SPACED_LAX.text(publish_collection(args.reference, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
