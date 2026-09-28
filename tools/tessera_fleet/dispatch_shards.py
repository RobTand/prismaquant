"""Submit the GLM-5.3 Tessera export, one PrismaBuild action per input shard.

Moved from PrismaBuild's ``tools/fleet/dispatch_tessera_shards.py`` on
2026-09-28 (RobTand/prismabuild#1076), and rewired onto ``pbcampaign``.

* **One sealed tree.** The wrapper, the plan and every ``.py`` of the encoder
  are copied from the source checkout into a new Git workspace, and ``pbrun``
  seals that workspace's snapshot into every action. Both boxes therefore run
  the same bytes: the two halves of the 2026-09-01 export were once written
  by encoders seven commits apart, and a sealed snapshot refuses that drift
  instead of shipping it.

* **Re-submitting is free.** A row whose receipt is already in the CAS is a
  lookup, not a run, so every shard of the range is submitted rather than only
  the ones known to be missing. Keys sealed by the old PrismaBuild driver are
  not these keys: that driver sealed its own action shape, and this one uses
  ``pbrun``'s, so a shard it finished is encoded again once.

* **One action per shard, each to its own directory.** Shards share no state,
  and a shared output directory would race on the exporter's report and aux
  copies. ``merge_tessera_parts.py`` takes ``nargs="+"``, so 120
  self-consistent parts merge exactly as two would.

``--dry-run`` prints the manifest rows and stages and submits nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.tessera_fleet import common

CHECKOUT = Path("/mnt/shared/prismabuild-fleet/checkout")
SOURCE = "/mnt/shared/models/GLM-5.3-Flash-BF16"
PLAN = ("/mnt/shared/dq-runs/glm53-tessera-alloc-20260901/artifacts/"
        "glm53_tessera_plan.json")
PARTS = "/mnt/shared/models/GLM-5.3-Flash-Tessera-E2M1K2-20260901-parts"
PYTHON = "/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python"
WRAPPER = "tessera_export_shard.py"

#: How many shards the export is cut into. Declared here rather than shared
#: with the ladder driver: each driver states the shape of its own run.
OF_SHARDS = 120

#: Measured: one exporter holds about 8 GB resident, and four concurrent took
#: a GB10 from 116 GB free to 55 GB. 16 GB is the honest cost of one shard.
RESOURCE_DEMAND = {"cpu": 1, "gpu": 1, "mem_gb": 16}

#: Relative to the tree the action runs in, so the import lands on the sealed
#: encoder bytes wherever the snapshot is materialized. Never /tmp: an OOM
#: cleared it once and took artifacts with it.
ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONPATH": "tessera/src",
    "TMPDIR": "/home/rob/tmp",
    # The default triton cache is root-owned on these boxes.
    "TRITON_CACHE_DIR": "/home/rob/.triton-cache",
}


def row(workspace: Path, shard: int, *, python: str, source: str, parts: str) -> dict:
    return {
        "cwd": str(workspace),
        "argv": [python, WRAPPER, "--shard", str(shard), "--source", source,
                 "--plan", "plan.json", "--out", f"{parts}/shard-{shard:05d}",
                 "--result", f"results/glm53-tessera/shard-{shard:05d}.json"],
        "demand": dict(RESOURCE_DEMAND),
        "tags": ["gb10"],
        "env": dict(ENVIRONMENT),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", required=True, type=common.shard_range(OF_SHARDS),
                    help=f"which shards to submit: one number, or an inclusive "
                         f"LO-HI range within 1-{OF_SHARDS} (e.g. 61 or 1-{OF_SHARDS})")
    ap.add_argument("--workspace", type=Path,
                    help="new directory the sealed tree and the submission records "
                         "are written to; required unless --dry-run")
    ap.add_argument("--checkout", type=Path, default=CHECKOUT,
                    help="tree holding the wrapper and the tessera/ encoder sources")
    ap.add_argument("--plan", type=Path, default=Path(PLAN),
                    help="allocation plan, copied into the sealed tree")
    ap.add_argument("--source", default=SOURCE, help="source checkpoint every shard reads")
    ap.add_argument("--parts", default=PARTS, help="shared directory the shard parts go to")
    ap.add_argument("--python", default=PYTHON, help="interpreter on the GPU boxes")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the manifest rows; stage and submit nothing")
    args = ap.parse_args(argv)
    if args.workspace is None and not args.dry_run:
        ap.error("--workspace is required unless --dry-run")
    workspace = (args.workspace or Path("<workspace>")).resolve()
    rows = [row(workspace, shard, python=args.python, source=args.source,
                parts=args.parts) for shard in args.shards]
    if args.dry_run:
        print(json.dumps(rows, indent=1))
        return 0
    common.stage_workspace(
        workspace,
        common.encoder_workspace_files(args.checkout, args.checkout / WRAPPER, WRAPPER,
                                       {"plan.json": args.plan}),
        message="PrismaQuant sealed Tessera export shards")
    submitted = common.submit_detached(rows, workspace, "export", wait_s=600.0)
    print(f"submitted {len(submitted)} shard action(s); "
          f"status: python3 -m tools.tessera_fleet.status --workspace {workspace}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"dispatch_shards: {exc}")
