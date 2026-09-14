#!/usr/bin/env python3
"""Write a Fisher probe keyed exactly to a Tessera census's cost rows.

Research input for an allocation over a census cost table: packed-expert rows
become the census's per-expert rows (``prismaquant.tessera_census_stats``).
Both inputs are read by their SHA-256; the output is a new file beside a
``.sha256`` sidecar, never an overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cluster_campaign import _atomic_write_new_bytes
from prismaquant.model_profiles.structure import load_structure_spec
from prismaquant.schemas import validate_probe_payload
from prismaquant.tessera_census_stats import META_KEY, expand_probe_onto_census


def _bound(path: str, digest: str, label: str):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != digest:
        raise SystemExit(f"{label} {path}: SHA-256 {actual}, expected {digest}")
    return pickle.loads(raw)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--probe-sha256", required=True)
    parser.add_argument("--costs", required=True)
    parser.add_argument("--costs-sha256", required=True)
    parser.add_argument("--structure-spec", required=True,
                        help="model_profiles/specs id, e.g. glm5_next")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    structure = load_structure_spec(args.structure_spec)
    if structure is None:
        raise SystemExit(f"no structure spec {args.structure_spec!r}")
    probe = _bound(args.probe, args.probe_sha256, "probe")
    cost = _bound(args.costs, args.costs_sha256, "cost table")
    expanded = expand_probe_onto_census(probe, cost, structure=structure)
    expanded["meta"][META_KEY]["inputs"] = {
        "probe": {"path": str(Path(args.probe).resolve()), "sha256": args.probe_sha256},
        "costs": {"path": str(Path(args.costs).resolve()), "sha256": args.costs_sha256},
        "structure_spec": args.structure_spec,
    }
    out = Path(args.out)
    validate_probe_payload(expanded, str(out))
    raw = pickle.dumps(expanded, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(raw).hexdigest()
    _atomic_write_new_bytes(out, raw)
    _atomic_write_new_bytes(out.with_name(out.name + ".sha256"),
                            f"{digest}  {out.name}\n".encode())
    summary = {key: value for key, value in expanded["meta"][META_KEY].items()
               if key != "dropped_rows"}
    summary.update({"out": str(out.resolve()), "sha256": digest, "bytes": len(raw),
                    "rows": len(expanded["stats"]),
                    "dropped_row_count": len(expanded["meta"][META_KEY]["dropped_rows"]),
                    "n_params": sum(int(r["n_params"]) for r in expanded["stats"].values())})
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
