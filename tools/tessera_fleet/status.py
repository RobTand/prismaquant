"""Collect native Flux facts for the exact saved Tessera fleet stage IDs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from prismaquant.tessera_campaign_rows import collect_rows
from tools.tessera_fleet.common import STATE, stage_state

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


def exit_status(table: list[dict], expected: int) -> int:
    statuses = [row.get("status", "") for row in table]
    if len(table) != expected or any(s not in {"succeeded", "running"}
                                     for s in statuses):
        return EXIT_FAILED
    if "running" in statuses:
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
    args = ap.parse_args(argv)
    state = stage_state(args.workspace, args.stage)
    if not (state / 'manifest.json').exists():
        print(f"no {args.stage} submissions recorded under {args.workspace / STATE}")
        return EXIT_NOTHING
    try:
        table = collect_rows(state, wait=False)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'native status query failed: {exc}')
        return EXIT_FAILED
    expected = len(json.loads((state / 'manifest.json').read_text())['rows'])
    print(json.dumps(table, indent=2))
    counts = Counter(row.get("status", "?") for row in table)
    print(f"{args.stage}: {expected} recorded, "
          + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())))
    return exit_status(table, expected)


if __name__ == "__main__":
    raise SystemExit(main())
