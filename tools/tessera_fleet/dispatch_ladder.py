"""Submit the Tessera rate-band probe, one PrismaBuild action per input shard.

What this buys, and why it is a separate stage from the export: the
completion axis is embedded, so ONE encode per Linear prices the whole band
``[rung/arity, cap/arity]`` bpp instead of one encode per rate point. At rung
4 that is four rate points from one Viterbi, at a quarter of the encodes.

It is a *generator*, not a shipping path. Measured 2026-09-01 on a real GLM
expert: truncation costs 1.03x-1.28x, but choosing a low rung to widen the band
costs up to 2.15x at equal bpp. So the band is priced by truncation and the
selected point is re-encoded natively.

Moved from PrismaBuild's ``tools/fleet/dispatch_tessera_ladder.py`` on
2026-09-28 (RobTand/prismabuild#1076), and rewired onto ``pbcampaign``. The
probe source is copied, with every ``.py`` of the encoder, into a new Git
workspace that ``pbrun`` seals into each action, so two dispatches with
different probe bytes are different actions and never overwrite each other's
staged copy. ``--dry-run`` prints the manifest rows and stages and submits
nothing, so it cannot touch the source checkout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.tessera_fleet import common
from tools.tessera_fleet.dispatch_shards import CHECKOUT, ENVIRONMENT, PYTHON, SOURCE

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
         "--result", f"results/glm53-tessera-ladder/rung{rung}/shard-{shard:05d}.json"],
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
