"""Re-stamp a cost table's missing per-unit topology from the profile grammar.

A scoped Tessera allocation classifies every unit from facts its stats row
carries (``tessera_serving_scope.unit_structure_from_stats``). AURA tables
built before PQ #1278 carry none, so the scope refuses them. This tool reads
such a table by path and SHA-256, stamps each row without producer topology
with ``unit_structure`` from ``tessera_serving_scope.unit_structure_from_profile``
and ``unit_topology_source="profile_grammar"``, and publishes the result as a
new table next to a receipt. The input table is never modified, and a row
whose producer recorded topology keeps it unchanged.

Usage::

    python -m prismaquant.unit_topology_restamp \\
        --table joint-allocation.pkl --table-sha256 <hex> \\
        --output joint-allocation.topology.pkl [--model /path/to/model]

``--model`` defaults to the table's ``provenance.model``; the profile is the
one ``model_profiles.detect_profile`` resolves for it, as in the allocator.
The receipt ``<output>.receipt.json`` binds input and output by SHA-256 and
records the model, the profile and the per-source and per-structure counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

RECEIPT_SCHEMA = "prismaquant.unit_topology_restamp.receipt.v1"


def restamp_table(*, table: str, table_sha256: str, output: str,
                  model: str | None = None) -> dict:
    from . import model_profiles
    from .cluster_campaign import _atomic_write_new_bytes
    from .tessera_serving_scope import restamp_unit_topology

    source = Path(table)
    target = Path(output)
    receipt_path = target.with_suffix(target.suffix + ".receipt.json")
    if target.exists() or receipt_path.exists():
        raise FileExistsError(f"restamp output already exists: {target}")
    if target.resolve() == source.resolve():
        raise ValueError("restamp output must be a new table, not its input")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != table_sha256:
        raise ValueError(f"{source}: sha256 {digest} differs from the bound {table_sha256}")
    payload = pickle.loads(raw)
    del raw
    model_path = model or (payload.get("provenance") or {}).get("model")
    if not model_path:
        raise ValueError("no model: pass --model or restamp a table whose provenance names one")
    profile = model_profiles.detect_profile(model_path)
    result, summary = restamp_unit_topology(payload, profile, input_sha256=digest)
    encoded = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "input": {"path": str(source.resolve()), "sha256": digest},
        "output": {"path": str(target.resolve()), "sha256": hashlib.sha256(encoded).hexdigest()},
        "model": str(model_path),
        "summary": summary,
        "units": len(result["stats"]),
        "fields_changed": ["stats[*].unit_structure", "stats[*].unit_topology_source",
                           "provenance.unit_topology_restamp"],
    }
    _atomic_write_new_bytes(target, encoded)
    _atomic_write_new_bytes(receipt_path,
                            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True, help="cost table to re-stamp (read only)")
    parser.add_argument("--table-sha256", required=True, help="SHA-256 the table must have")
    parser.add_argument("--output", required=True, help="new table path; must not exist")
    parser.add_argument("--model", default=None,
                        help="model whose profile declares the grammar; default provenance.model")
    args = parser.parse_args(argv)
    print(json.dumps(restamp_table(table=args.table, table_sha256=args.table_sha256,
                                   output=args.output, model=args.model), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
