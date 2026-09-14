#!/usr/bin/env python3
"""Read-only, bounded audit of two real campaign wire receipts.

The dense and routed names are explicit inputs. This does not claim a joint
handoff or selected allocation exists, and does not write campaign files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cost_stage_checkpoint import _load_unit, unit_path
from prismaquant.tessera_joint_aura import STAGE
from tessera.cached_unit import verify_cached_unit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-parts", type=Path, required=True)
    parser.add_argument("--wire-dir", type=Path, required=True)
    parser.add_argument("--dense", required=True)
    parser.add_argument("--expert", required=True)
    parser.add_argument("--dense-format", required=True)
    parser.add_argument("--expert-format", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    results = []
    seals = set()
    for name, fmt in ((args.dense, args.dense_format), (args.expert, args.expert_format)):
        path = unit_path(args.checkpoint_parts, name)
        with path.open("rb") as handle:
            envelope = pickle.load(handle)
        seal = envelope["identity_sha256"]
        seals.add(seal)
        state = _load_unit(path, stage=STAGE, qname=name, identity_sha256=seal)
        record = state["wire_records"][fmt]
        observed = record["identity"]
        wire = args.wire_dir / record["file"]
        if wire.is_symlink() or wire.resolve().parent != args.wire_dir.resolve():
            raise ValueError(f"{name}@{fmt}: wire escaped campaign directory")
        blob = wire.read_bytes()
        accepted = verify_cached_unit(blob, record, observed)
        results.append({"unit": name, "format": fmt, "blob_sha256": hashlib.sha256(blob).hexdigest(),
                        "bytes": len(blob), "wire_bytes": accepted.wire_bytes,
                        "source_hessian_encoder_claimed": True,
                        "current_source_hessian_encoder_rederived": False})
    if len(seals) != 1:
        raise ValueError("sampled campaign units were sealed under different checkpoint identities")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "prismaquant.tessera_selected_cache_slice_audit.v1",
                                    "checkpoint_identity_sha256": next(iter(seals)),
                                    "scope": "two self-receipt wire parses; root checkpoint and source/H not re-read",
                                    "units": results}, indent=2) + "\n")
    print(json.dumps({"status": "passed", "units": len(results), "out": str(args.out)}))


if __name__ == "__main__":
    main()
