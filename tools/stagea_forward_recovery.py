"""Freeze a finite forward recovery capsule from an explicit native selection."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.joint_forward_resume import _read, build_forward_recovery
from prismaquant.digests import DIRECT_ASCII_STRICT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--specification', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frontier', type=int, required=True)
    parser.add_argument('--checkpoint-exit', type=int)
    parser.add_argument('--bind-current-implementation', action='store_true')
    args = parser.parse_args(argv)
    result = build_forward_recovery(selection=_read(args.selection)[0],
        specification=_read(args.specification)[0], output=args.output,
        frontier=args.frontier, checkpoint_exit=args.checkpoint_exit,
        bind_current_implementation=args.bind_current_implementation)
    print(DIRECT_ASCII_STRICT.text(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
