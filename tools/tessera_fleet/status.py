"""Collect native Flux facts for the exact saved Tessera fleet stage IDs."""
from __future__ import annotations

import argparse
from pathlib import Path
from prismaquant.tessera_campaign_rows import collect_rows
from tools.tessera_fleet.common import (  # noqa: F401
    EXIT_DONE, EXIT_FAILED, EXIT_NOTHING, EXIT_RUNNING, exit_status, show_status)

_EPILOG = """\
exit status:
  0  every recorded action ended with its work done
  1  an action ended without its work done, or its ending could not be read
  2  the command line was wrong
  3  nothing to read: the workspace records no submission for this stage
  4  nothing failed, and at least one action is still waiting
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=_EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", type=Path, required=True,
                    help="the directory a tessera_fleet driver was given")
    ap.add_argument("--stage", default="export",
                    help="export, ladder, or a dispatch_model stage (prepare, encode, "
                         "assemble)")
    args = ap.parse_args(argv)
    return show_status(args.workspace, args.stage, collect_rows=collect_rows)


if __name__ == "__main__":
    raise SystemExit(main())
