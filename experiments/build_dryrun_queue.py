"""Build a throwaway pool queue whose 16 ready rows carry the campaign's real data manifests.

The resume rows must never be submitted to the live fleet, so the only honest
way to ask the prewarm loop what it would warm is to give it a private queue
holding the same sealed shape: one action per row, each carrying the row's real
``pbcampaign.data-manifest`` input, published in the manifest's own order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True,
                        help="runtime generation whose src/ defines the manifest input")
    parser.add_argument("--campaign-manifest", required=True,
                        help="resume-overlap-manifest.with-data.json")
    parser.add_argument("--root", required=True,
                        help="scratch directory for the queue and its CAS")
    parser.add_argument("--priority", type=int, default=0)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(args.runtime) / "src"))
    import prismabuild.core as pb
    from prismabuild import pool

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    queue = pool.PoolQueue(root / "pb-queue")
    queue.ensure_layout()
    cas_root = root / "cas"
    cas = pb.PrismaBuildCAS(cas_root)

    campaign = json.loads(Path(args.campaign_manifest).read_text())
    rows = campaign["rows"] if isinstance(campaign, dict) else campaign

    built = []
    for index, row in enumerate(rows):
        manifest_path = row.get("data_manifest")
        if not manifest_path:
            print(f"row at rank {index}: no data_manifest, skipped")
            continue
        manifest = pb.load_data_manifest(manifest_path)
        row_id = ((manifest.get("annotations") or {}).get("row_id")
                  or os.path.basename(manifest_path).split(".")[0])
        entry, _ = cas.ingest_input(
            manifest_path, input_id=pb.PBCAMPAIGN_DATA_MANIFEST_INPUT_ID)
        action_key = hashlib.sha256(
            f"dryrun:{row_id}:{entry['sha256']}".encode()).hexdigest()
        request = cas_root / "requests" / action_key[:2] / f"{action_key}.json"
        request.parent.mkdir(parents=True, exist_ok=True)
        request.write_text(json.dumps({
            "action_key": action_key,
            "inputs": [entry],
            "params": {
                "command": ["true"],
                "row_id": row_id,
                "data_manifest": {
                    "input": entry,
                    "mount_prefix": manifest["mount_prefix"],
                    "entry_count": manifest["entry_count"],
                    "total_bytes": manifest["total_bytes"],
                },
            },
        }))
        queue.publish(action_key=action_key, cas_root=cas_root,
                      worker_script=str(root / "worker.py"),
                      checkout_root=str(root), priority=args.priority)
        built.append({"rank": index, "row_id": row_id,
                      "action_key": action_key,
                      "manifest_sha256": entry["sha256"],
                      "total_bytes": manifest["total_bytes"]})
        time.sleep(0.02)

    out = {"root": str(root), "rows": built,
           "total_bytes": sum(b["total_bytes"] for b in built)}
    print(json.dumps(out, indent=1))
    (root / "built.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
