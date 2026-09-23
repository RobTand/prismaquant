#!/usr/bin/env python3
"""Retire a superseded Stage A run's checkpoints and pinned entries (PQ #1073).

Run it as a PrismaBuild action on a checkout of this repository, for example::

    pbrun --cwd <checkout> --cpus 1 --demand mem_gb=4 --priority -10 -- \\
      <python> tools/retire_stage_a_run.py \\
        --output-root <superseded run's Stage A output root> \\
        --superseded-by <successor chain-state.json or adjoint-capture.json> \\
        --superseded-by-sha256 <its sha256> \\
        --binding-root <directory where live seed specs, bands and Stage B
                        metadata are kept> [--binding-root ...] [--dry-run]

``--dry-run`` checks everything and prints the plan; nothing is written. See
``prismaquant.stage_a_retirement`` for the proof, the plan and the order of the
removal. Exit 3 on a refusal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismaquant.stage_a_retirement import (  # noqa: E402
    RetirementRefused, apply_retirement, plan_retirement)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--superseded-by", type=Path, required=True)
    parser.add_argument("--superseded-by-sha256", required=True)
    parser.add_argument("--binding-root", type=Path, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = plan_retirement(args.output_root, successor_path=args.superseded_by,
                               successor_sha256=args.superseded_by_sha256,
                               binding_roots=args.binding_root)
        if args.dry_run:
            print(json.dumps({
                "status": "planned", "record": plan.record,
                "batches": [row["batch_id"] for *_, row in plan.batches],
                "untouched_batches": plan.untouched,
                "bindings_read": plan.bindings_read}, sort_keys=True))
            return 0
        report = apply_retirement(plan)
    except (RetirementRefused, OSError, ValueError) as exc:
        print(f"Stage A retirement refused: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({"status": "retired", **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
