"""Where a Tessera fleet dispatch got to, asked of ``pbwait``.

Reads the keys a driver recorded under ``<workspace>/.pb-state/`` and runs
the published ``pbwait.py --wait-s 0`` over them, which waits for nothing and
prints one table of endings. The table is PrismaBuild's own account of each
action under whichever transport carried it, so this screen neither lists
queue directories nor reads the CAS.

Moved from PrismaBuild's ``tools/fleet/tessera_status.py`` on 2026-09-28
(RobTand/prismabuild#1076). That screen counted receipts in the fleet CAS by
the old drivers' definition ID; the drivers now seal through ``pbrun``, so a
dispatch is named by its workspace instead.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from prismaquant.pbwait_table import DONE_STATUSES, parse_pbwait_table
from tools.tessera_fleet.common import PUBLISHED_TOOLS, STATE

EXIT_DONE = 0
EXIT_FAILED = 1
EXIT_NOTHING = 3
EXIT_RUNNING = 4

_EPILOG = """\
exit status:
  0  every recorded action ended with its work done
  1  an action ended without its work done, or its ending could not be read
  2  the command line was wrong
  3  nothing to read: the workspace records no submission for this stage
  4  nothing failed, and at least one action is still waiting
"""


def recorded_keys(workspace: Path, stage: str) -> list[str]:
    """The keys ``<stage>-submissions.json`` or ``<stage>-endings.json`` hold."""

    state = Path(workspace) / STATE
    for name, field in ((f"{stage}-submissions.json", "rows"),
                        (f"{stage}-endings.json", "rows")):
        path = state / name
        if path.is_file():
            rows = json.loads(path.read_text()).get(field) or []
            return [str(row["action_key"]) for row in rows if row.get("action_key")]
    return []


def exit_status(table: list[dict], expected: int) -> int:
    statuses = [row.get("status", "") for row in table]
    if len(table) != expected or any(s not in DONE_STATUSES and s != "waiting"
                                     for s in statuses):
        return EXIT_FAILED
    if "waiting" in statuses:
        return EXIT_RUNNING
    return EXIT_DONE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=_EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", type=Path, required=True,
                    help="the directory a tessera_fleet driver was given")
    ap.add_argument("--stage", default="export",
                    help="export, ladder, or a dispatch_model stage (prepare, encode, "
                         "assemble)")
    ap.add_argument("--pbwait", type=Path, default=PUBLISHED_TOOLS / "pbwait.py",
                    help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    keys = recorded_keys(args.workspace, args.stage)
    if not keys:
        print(f"no {args.stage} submissions recorded under {args.workspace / STATE}")
        return EXIT_NOTHING
    completed = subprocess.run(
        [sys.executable, str(args.pbwait), "--wait-s", "0", *keys],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    table = parse_pbwait_table(completed.stdout)
    print(completed.stdout, end="")
    counts = Counter(row.get("status", "?") for row in table)
    print(f"{args.stage}: {len(keys)} recorded, "
          + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())))
    return exit_status(table, len(keys))


if __name__ == "__main__":
    raise SystemExit(main())
