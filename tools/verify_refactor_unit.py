#!/usr/bin/env python3
"""Run a refactor unit's compile and behavior checks inside a PB action."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", nargs="+", required=True, dest="modules")
    parser.add_argument("--tests", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--base")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.proof is not None:
        if args.base is None:
            parser.error("--proof requires --base")
        subprocess.run([sys.executable, str(args.proof), "--base", args.base], cwd=root, check=True)
    for command in (
        [sys.executable, "-m", "compileall", "-q", *args.modules],
        [sys.executable, "-m", "pytest", "-q", "-n", str(args.workers), *args.tests],
    ):
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
