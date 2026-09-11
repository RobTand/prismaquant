"""Show the deployed runtime carrying a prewarm record into a done row.

A fleet worker claims a published action within a second, which is faster than
any external process can write the sidecar the claim reads, so the live
demonstration is done against a private queue: the code under test is the
deployed ``pool.py``, the queue is a real ``PoolQueue``, and only the worker is
this script instead of a fleet loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.runtime) / "src"))
    import prismabuild.core as pb
    from prismabuild import pool

    root = Path(args.root)
    if root.exists():
        subprocess.run(["rm", "-rf", str(root)], check=True)
    root.mkdir(parents=True)
    queue = pool.PoolQueue(root / "pb-queue")
    queue.ensure_layout()
    cas_root = root / "cas"
    cas = pb.PrismaBuildCAS(cas_root)

    manifest = pb.load_data_manifest(args.manifest)
    entry, _ = cas.ingest_input(args.manifest,
                                input_id=pb.PBCAMPAIGN_DATA_MANIFEST_INPUT_ID)
    action_key = hashlib.sha256(f"receipt-demo:{entry['sha256']}".encode()).hexdigest()
    request = cas_root / "requests" / action_key[:2] / f"{action_key}.json"
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text(json.dumps({
        "action_key": action_key, "inputs": [entry],
        "params": {"command": ["true"], "data_manifest": {
            "input": entry, "mount_prefix": manifest["mount_prefix"],
            "entry_count": manifest["entry_count"],
            "total_bytes": manifest["total_bytes"]}}}))
    queue.publish(action_key=action_key, cas_root=cas_root,
                  worker_script=str(root / "worker.py"),
                  checkout_root=str(root))
    print("published", action_key[:12], flush=True)

    loop = [sys.executable, str(Path(args.runtime) / "tools" / "fleet" / "prewarm_loop.py"),
            "--pool-root", str(queue.root), "--cas-root", str(cas_root),
            "--mount-map", "/mnt/shared=/storage_pool/shared",
            "--arcstats", "/proc/spl/kstat/zfs/arcstats",
            "--readers", "4", "--lookahead", "2", "--min-manifest-bytes", "0",
            "--poll-s", "1", "--claim-grace-min", "20", "--once"]
    run = subprocess.run(loop, capture_output=True, text=True)
    print("loop rc", run.returncode)
    print((run.stdout or run.stderr).strip()[-900:], flush=True)

    sidecar = queue.root / "prewarm" / f"{action_key}.json"
    print("sidecar exists:", sidecar.exists())
    if sidecar.exists():
        print("sidecar:", json.dumps({k: v for k, v in json.loads(sidecar.read_text()).items()
                                      if k not in ("arc_before", "arc_after")}, indent=1))

    claimed = queue.claim(owner=f"receipt-demo:{os.getpid()}",
                          capacity={"cpu": 8, "mem_gb": 8})
    if claimed is None:
        print("nothing claimed")
        return 1
    print("claimed carries prewarm:", "prewarm" in claimed)
    queue.finish(action_key, status="executed", detail={"note": "receipt demonstration"})
    done = json.loads((queue.root / "done" / f"{action_key}.json").read_text())
    got = done.get("detail", {}).get("prewarm")
    print("done row detail.prewarm:",
          json.dumps({k: v for k, v in got.items()
                      if k not in ("arc_before", "arc_after")}, indent=1)
          if got else None)
    return 0 if got else 2


if __name__ == "__main__":
    sys.exit(main())
