#!/usr/bin/env python3
"""Union merged Tessera census tables that priced disjoint rungs of one census."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.tessera_census_union import UnionRefused, build_plan, write_plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", action="append", required=True, metavar="NAME=DIR",
                        help="a merged census table directory (repeat; order sets anchor order)")
    parser.add_argument("--out", required=True, help="union table directory to create")
    parser.add_argument("--dry-run", action="store_true", help="run every check, print the plan, write nothing")
    parser.add_argument("--resume", action="store_true",
                        help="continue into an existing --out whose files verify against the plan")
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        if (out.exists() or out.is_symlink()) and not args.resume and not args.dry_run:
            raise UnionRefused(f"--out {out} exists; pass --resume to continue into it")
        plan = build_plan(args.table, out)
        summary = dict(plan.summary)
        if args.dry_run:
            summary["dry_run"] = True
            summary["writes"] = 0
        else:
            summary["written"] = write_plan(plan, resume=args.resume)
    except UnionRefused as exc:
        print(f"union refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
