#!/usr/bin/env python3
"""Verify definition equivalence, imports, and streamed behavior after the split."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_TESTS = (
    "tests/test_streamed_boundary_artifacts.py",
    "tests/test_layer_major_boundary_capture.py",
    "tests/test_boundary_partition_ranges_738.py",
    "tests/test_original_source_identity.py",
    "tests/test_source_identity_portable_device.py",
    "tests/test_source_identity_validate_derivation.py",
    "tests/test_source_checkpoint_identity_parallel_hash.py",
    "tests/test_joint_aura_streamed.py",
    "tests/test_streamed_cost_checkpoints.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    commands = []
    if not args.baseline:
        commands.append([sys.executable, "tools/refactor_cost_streaming.py", "--base", args.base])
        commands.append([sys.executable, "-m", "compileall", "-q", "prismaquant", "tools", "tests", "experiments"])
    commands.append([sys.executable, "-m", "pytest", "-q", "-n", str(args.workers), *BEHAVIOR_TESTS])
    for command in commands:
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
