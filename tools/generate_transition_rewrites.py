#!/usr/bin/env python3
"""Emit the GENERATED REWRITES block of a closed joint source transition.

A transition module reconstructs the sealed producer package by reversing exact
old/new snippets and hashing the result to the prepared record's
``implementation_sha256``. The snippets are literals, so they have to be exact,
and they were hand-assembled -- which is fine once and a trap the second time.

This writes them. Give it the sealed package (any Git revision, or a directory)
and the package that will execute, and it prints the block, each hunk carrying
the smallest line context that makes both sides unique in their own file. It
decides nothing: a file that changed produces a hunk, and whether that change
belongs in a transition at all is the author's judgement, recorded in the
module and its design note.

It verifies what it emits. The generated table is applied in reverse to the
executing package and the result must hash, with the transition's own complete-
package algorithm, to the sealed package's digest -- the same check
``source_proof`` makes at run time.

    python3 tools/generate_transition_rewrites.py \\
        --sealed-rev 208d340d27 --package prismaquant \\
        --new-file joint_aura_run_transition.py \\
        --new-file joint_aura_transitions.py \\
        --new-file joint_aura_retained_budget_transition.py
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path
import pprint
import subprocess
import sys
import tempfile


def _package_digest(files):
    """The complete-package hash every transition's ``source_proof`` computes."""
    digest = hashlib.sha256()
    for name, payload in sorted(files.items()):
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _read_tree(root):
    files = {}
    for path in sorted(Path(root).rglob("*")):
        if (not path.is_file() or "__pycache__" in path.relative_to(root).parts
                or path.suffix in {".pyc", ".pyo"}):
            continue
        files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def _sealed_tree(rev, package, repo):
    """The package as of one revision, materialized without touching the checkout."""
    listing = subprocess.run(["git", "ls-tree", "-r", "--name-only", f"{rev}:{package}"],
                             cwd=repo, check=True, capture_output=True, text=True).stdout.split()
    files = {}
    for name in listing:
        if name.endswith((".pyc", ".pyo")) or "__pycache__" in name.split("/"):
            continue
        files[name] = subprocess.run(["git", "show", f"{rev}:{package}/{name}"],
                                     cwd=repo, check=True, capture_output=True).stdout
    return files


def _hunks(old_text, new_text):
    """Exact old/new snippets, each grown until both sides are unique in their file."""
    old_lines, new_lines = old_text.splitlines(True), new_text.splitlines(True)
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    pairs = []
    for group in matcher.get_grouped_opcodes(0):
        old_start, old_stop = group[0][1], group[-1][2]
        new_start, new_stop = group[0][3], group[-1][4]
        for context in range(0, max(len(old_lines), len(new_lines)) + 1):
            a0, a1 = max(0, old_start - context), min(len(old_lines), old_stop + context)
            b0, b1 = max(0, new_start - context), min(len(new_lines), new_stop + context)
            old_snippet, new_snippet = "".join(old_lines[a0:a1]), "".join(new_lines[b0:b1])
            if not old_snippet and not new_snippet:
                continue
            if old_text.count(old_snippet) == 1 and new_text.count(new_snippet) == 1:
                pairs.append((old_snippet, new_snippet))
                break
        else:
            raise SystemExit("no unique context exists for a hunk; the files repeat themselves")
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sealed-rev", help="the Git revision holding the sealed package")
    parser.add_argument("--sealed-dir", type=Path, help="a directory holding the sealed package")
    parser.add_argument("--package", default="prismaquant", help="the package path inside the revision")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--executing-dir", type=Path,
                        help="the package that will execute; defaults to --repo/--package")
    parser.add_argument("--new-file", action="append", default=[],
                        help="a file the sealed package does not have; repeatable")
    parser.add_argument("--expect-sealed-sha256", help="refuse unless the sealed package hashes to this")
    args = parser.parse_args(argv)

    if (args.sealed_rev is None) == (args.sealed_dir is None):
        raise SystemExit("pass exactly one of --sealed-rev and --sealed-dir")
    sealed = (_read_tree(args.sealed_dir) if args.sealed_dir is not None
              else _sealed_tree(args.sealed_rev, args.package, args.repo))
    executing_root = args.executing_dir or (args.repo / args.package)
    executing = _read_tree(executing_root)

    sealed_digest = _package_digest(sealed)
    if args.expect_sealed_sha256 and sealed_digest != args.expect_sealed_sha256:
        raise SystemExit(f"sealed package hashes to {sealed_digest}, not {args.expect_sealed_sha256}")

    new_files = set(args.new_file)
    appeared = set(executing) - set(sealed)
    if appeared != new_files:
        raise SystemExit(
            f"the executing package adds {sorted(appeared)}; --new-file names {sorted(new_files)}")
    missing = set(sealed) - set(executing)
    if missing:
        raise SystemExit(f"the executing package dropped {sorted(missing)}; a transition cannot restore a file")

    rewrites = {}
    for name in sorted(sealed):
        if sealed[name] == executing[name]:
            continue
        rewrites[name] = _hunks(sealed[name].decode(), executing[name].decode())

    # Apply in reverse, exactly as source_proof does, and require the sealed digest.
    reconstructed = {}
    for name, payload in executing.items():
        if name in new_files:
            continue
        for old, new in reversed(rewrites.get(name, ())):
            if payload.count(new.encode()) != 1:
                raise SystemExit(f"generated hunk is not unique in {name}")
            payload = payload.replace(new.encode(), old.encode(), 1)
        reconstructed[name] = payload
    digest = _package_digest(reconstructed)
    if digest != sealed_digest:
        raise SystemExit(f"generated table reconstructs {digest}, not the sealed {sealed_digest}")

    print("# BEGIN GENERATED REWRITES")
    print("_SOURCE_REWRITES = " + pprint.pformat(rewrites, width=100))
    print("# END GENERATED REWRITES")
    print(f"# sealed package sha256 {sealed_digest}", file=sys.stderr)
    print(f"# {len(rewrites)} rewritten files, "
          f"{sum(len(value) for value in rewrites.values())} hunks, "
          f"{len(new_files)} new files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
