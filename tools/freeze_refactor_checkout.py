#!/usr/bin/env python3
"""Freeze source edits in a separate checkout before PrismaBuild submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def freeze_refactor_checkout(workspace: Path, target: Path) -> dict:
    workspace = workspace.resolve()
    target = target.resolve()
    conflicts = subprocess.check_output(
        ["git", "ls-files", "--unmerged", "-z"], cwd=workspace)
    if conflicts:
        raise ValueError("cannot freeze source with unresolved merge conflicts")
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=workspace)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z",
         "prismaquant", "tools", "tests", "experiments", "docs"], cwd=workspace)
    subprocess.run(["git", "worktree", "add", "--detach", "--quiet", str(target), head],
                   cwd=workspace, check=True)
    if patch:
        subprocess.run(["git", "apply", "--binary"], cwd=target, input=patch, check=True)
    copied = []
    for raw in untracked.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode())
        source = workspace / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            destination.symlink_to(source.readlink())
        else:
            shutil.copy2(source, destination)
        copied.append(relative.as_posix())
    return {"checkout": str(target), "base": head, "patch_bytes": len(patch), "untracked": copied}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_refactor_checkout(args.cwd, args.output)))


if __name__ == "__main__":
    main()
