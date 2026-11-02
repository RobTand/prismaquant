#!/usr/bin/env python3
"""Materialize PQ's reviewed PB source bundle, offline, from a pinned tree.

Accept a Git checkout or the unpacked pinned GitHub tarball. Authenticate the
complete Git tree, including executable bits and symlink targets, before
copying anything. This builds test data, never a deployable PB runtime.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import stat
import subprocess
import tempfile

PIN_PATH = Path(__file__).resolve().parents[1] / "tests/pb_runtime_generation_pin.json"

# Load the existing stdlib-only owner before the test environment is installed.
_digests = runpy.run_path(str(PIN_PATH.parents[1] / "prismaquant/digests.py"))
git_blob_sha1hex = _digests["git_blob_sha1hex"]
git_tree_sha1hex = _digests["git_tree_sha1hex"]
file_sha256hex = _digests["file_sha256hex"]


def _tree_digest(root: Path, *, top: bool = True) -> bytes:
    entries = []
    for path in root.iterdir():
        if top and path.name == ".git":
            continue
        mode = path.lstat().st_mode
        name = os.fsencode(path.name)
        if stat.S_ISLNK(mode):
            git_mode = b"120000"
            digest = bytes.fromhex(git_blob_sha1hex(os.fsencode(os.readlink(path))))
        elif stat.S_ISDIR(mode):
            git_mode, digest = b"40000", _tree_digest(path, top=False)
        elif stat.S_ISREG(mode):
            git_mode = b"100755" if mode & 0o111 else b"100644"
            digest = bytes.fromhex(git_blob_sha1hex(path.read_bytes()))
        else:
            raise ValueError(f"unsupported source tree member: {path}")
        entries.append((name + (b"/" if git_mode == b"40000" else b""),
                        git_mode + b" " + name + b"\0" + digest))
    return bytes.fromhex(git_tree_sha1hex(b"".join(record for _, record in sorted(entries))))


def build_bundle(source: Path, destination: Path, pin: dict) -> Path:
    """Write a sealed generation under destination after authenticating source."""
    source = Path(source).resolve(strict=True)
    commit = pin["candidate_commit"]
    if pin["runtime_generation"] != commit:
        raise ValueError("bundle generation differs from its commit")
    if (source / ".git").exists():
        head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if head != commit:
            raise ValueError(f"source commit {head} differs from {commit}")
    if _tree_digest(source).hex() != pin["source_tree"]:
        raise ValueError("source tree differs from the pinned Git tree")
    for name, digest in pin["files"].items():
        if file_sha256hex(source / name) != digest:
            raise ValueError(f"source digest differs: {name}")
    destination = Path(destination).resolve()
    if destination.is_relative_to(source):
        raise ValueError("bundle destination must be outside the source tree")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / commit
    with tempfile.TemporaryDirectory(prefix=".pb-bundle-", dir=destination) as temporary:
        staged = Path(temporary) / commit
        shutil.copytree(source, staged, symlinks=True,
                        ignore=lambda directory, names: [".git"] if Path(directory) == source else [])
        # Authenticate the copy too, before publishing it.
        if _tree_digest(staged).hex() != pin["source_tree"]:
            raise ValueError("source tree changed while copying")
        (staged / ".pinned-source.json").write_text(json.dumps({
            "commit": commit, "git_archive_sha256": pin["git_archive_sha256"]}) + "\n")
        for path in sorted(staged.rglob("*"), reverse=True):
            mode = path.lstat().st_mode
            if not stat.S_ISLNK(mode):
                path.chmod(0o555 if stat.S_ISDIR(mode) or mode & 0o111 else 0o444)
            os.utime(path, (pin["source_timestamp"],) * 2, follow_symlinks=False)
        staged.rename(output)
        output.chmod(0o555)
        os.utime(output, (pin["source_timestamp"],) * 2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pin", type=Path, default=PIN_PATH)
    args = parser.parse_args()
    pin = json.loads(args.pin.read_text())
    print(build_bundle(args.source, args.output, pin))


if __name__ == "__main__":
    main()
