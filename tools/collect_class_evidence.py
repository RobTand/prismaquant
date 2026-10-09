"""Collect one worker's platform/accelerator packet for a class measurement.

Thin in-tree wrapper over the fleet's ``pbevidence.py`` contract
(PrismaBuild #1598): run this tool as a normal PrismaBuild action on a
worker of the target class, then hand its stdout to
``pbrun --target-evidence``. The packet carries the worker's live platform
and accelerator facts; each claiming worker still checks them against its
own facts before it runs, so a wrong packet fails closed there.

``--out`` writes the packet to that path in one atomic step. Without it
the packet goes to stdout. The tool exits 1 and writes nothing when this
box cannot attest an accelerator.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

FLEET_PROBE = Path(
    "/mnt/shared/prismabuild-fleet/repo/tools/fleet/pbevidence.py")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--recorder-python", default=None)
    args = parser.parse_args(argv)
    module = runpy.run_path(str(FLEET_PROBE))
    packet = module["collect_packet"](
        **({"recorder_python": args.recorder_python}
           if args.recorder_python is not None else {}))
    raw = (json.dumps(packet, sort_keys=True, indent=2,
                      allow_nan=False) + "\n").encode()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_name(args.out.name + ".tmp")
        tmp.write_bytes(raw)
        tmp.replace(args.out)
    else:
        sys.stdout.write(raw.decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
