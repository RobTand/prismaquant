"""Plan Tessera rate-band probe rows for native Flux.

Source and output contracts are preserved independently of native execution
facts. GPU/container admission remains unqualified in the CPU baseline.
Existing pb-result.json scientific records keep their historical name/schema;
only a native terminal success can admit them for downstream validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.tessera_fleet import common
CHECKOUT = Path("/mnt/shared/prismabuild-fleet/checkout")
SOURCE = "/mnt/shared/models/GLM-5.3-Flash-BF16"
PYTHON = "/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python"
ENVIRONMENT = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "PYTHONPATH": "tessera/src", "TMPDIR": "/home/rob/tmp",
    "TRITON_CACHE_DIR": "/home/rob/.triton-cache",
}

WRAPPER = "tessera_ladder_probe.py"

#: How many shards the probe is cut into, declared for this run alone: a probe
#: later cut differently must not move the export's domain with it.
OF_SHARDS = 120

#: A probe holds one weight plus its forests and re-decodes in place -- lighter
#: than an export shard. Re-measure before trusting this if the rung changes.
RESOURCE_DEMAND = {"cpu": 1, "gpu": 1, "mem_gb": 12}


def probe_row(workspace: Path, shard: int, *, python: str, source: str, rung: int,
              calibrate_every: int) -> dict:
    return common.gb10_row(
        workspace,
        [python, WRAPPER, "--shard", str(shard), "--source", source,
         "--rung", str(rung), "--calibrate-every", str(calibrate_every),
         "--result", str(Path(workspace) / "results" / f"glm53-tessera-ladder/rung{rung}/shard-{shard:05d}.json")],
        demand=RESOURCE_DEMAND, env=ENVIRONMENT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", required=True, type=common.shard_range(OF_SHARDS),
                    help=f"which shards to probe: one number, or an inclusive "
                         f"LO-HI range within 1-{OF_SHARDS} (e.g. 1 or 1-{OF_SHARDS})")
    ap.add_argument("--rung", type=int, default=4,
                    help="body rate per code; one encode prices the whole "
                         "[rung/arity, cap/arity] bpp band")
    ap.add_argument("--calibrate-every", type=int, default=32,
                    help="every Nth unit also gets native encodes, so the "
                         "truncation bias in the band is measured")
    ap.add_argument("--wrapper", type=Path,
                    help=f"probe source file; defaults to {WRAPPER} in --checkout")
    ap.add_argument("--workspace", type=Path,
                    help="new directory the sealed tree and the submission records "
                         "are written to; required unless --dry-run")
    ap.add_argument("--checkout", type=Path, default=CHECKOUT,
                    help="tree holding the tessera/ encoder sources")
    ap.add_argument("--source", default=SOURCE, help="source checkpoint every shard reads")
    ap.add_argument("--python", default=PYTHON, help="interpreter on the GPU boxes")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the manifest rows; stage and submit nothing")
    args = ap.parse_args(argv)
    if args.workspace is None and not args.dry_run:
        ap.error("--workspace is required unless --dry-run")
    workspace = (args.workspace or Path("<workspace>")).resolve()
    rows = [probe_row(workspace, shard, python=args.python, source=args.source,
                rung=args.rung, calibrate_every=args.calibrate_every)
            for shard in args.shards]
    if args.dry_run:
        print(json.dumps(rows, indent=1))
        return 0
    # Refuse unqualified GPU/environment requirements before staging a new tree.
    common.campaign_rows.validate_row(rows[0])
    wrapper = args.wrapper or args.checkout / WRAPPER
    common.stage_workspace(
        workspace, common.encoder_workspace_files(args.checkout, wrapper, WRAPPER),
        message="PrismaQuant sealed Tessera ladder probe")
    submitted = common.submit_detached(rows, workspace, "ladder", wait_s=600.0)
    print(f"submitted {len(submitted)} probe action(s); "
          f"status: python3 -m tools.tessera_fleet.status --workspace {workspace} "
          f"--stage ladder")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"dispatch_ladder: {exc}")
