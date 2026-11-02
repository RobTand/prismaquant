"""Build the reviewed PB tools locally instead of requiring a fleet mount."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

import pytest


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "plain").write_bytes(b"pinned fixture\n")
    (root / "run").write_bytes(b"#!/bin/sh\nexit 0\n")
    (root / "run").chmod(0o755)
    (root / "link").symlink_to("src/plain")
    for name in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{name}_NAME", "Bundle fixture")
        monkeypatch.setenv(f"GIT_{name}_EMAIL", "fixture@example.invalid")
        monkeypatch.setenv(f"GIT_{name}_DATE", "2000-01-01T00:00:00Z")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    pin = {
        "candidate_commit": _git(root, "rev-parse", "HEAD"),
        "runtime_generation": _git(root, "rev-parse", "HEAD"),
        "source_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "source_timestamp": 946684800,
        "git_archive_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "-C", str(root), "archive", "HEAD"])).hexdigest(),
        "files": {"src/plain": hashlib.sha256(b"pinned fixture\n").hexdigest()},
    }
    return root, pin


def _inventory(root):
    return [(str(p.relative_to(root)), stat.S_IMODE(p.lstat().st_mode),
             p.lstat().st_mtime_ns,
             os.readlink(p) if p.is_symlink() else p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))]


def test_builder_seals_a_byte_stable_tree_from_checkout_or_tarball(source, tmp_path):
    from tools.build_prismabuild_source_bundle import build_bundle

    root, pin = source
    first = build_bundle(root, tmp_path / "one", pin)
    unpacked = tmp_path / "unpacked"
    shutil.copytree(root, unpacked, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    os.utime(unpacked / "src/plain", (123, 123))
    second = build_bundle(unpacked, tmp_path / "two", pin)
    assert first.name == second.name == pin["candidate_commit"]
    assert _inventory(first) == _inventory(second)
    assert (first / "src/plain").read_bytes() == b"pinned fixture\n"
    assert (first / "run").stat().st_mode & 0o777 == 0o555
    assert (first / "src/plain").stat().st_mode & 0o777 == 0o444
    assert os.readlink(first / "link") == "src/plain"
    assert json.loads((first / ".pinned-source.json").read_text()) == {
        "commit": pin["candidate_commit"], "git_archive_sha256": pin["git_archive_sha256"]}


def test_builder_refuses_a_checkout_at_the_wrong_commit(source, tmp_path):
    from tools.build_prismabuild_source_bundle import build_bundle

    root, pin = source
    _git(root, "commit", "--allow-empty", "-qm", "wrong commit, same tree")
    with pytest.raises(ValueError, match="commit"):
        build_bundle(root, tmp_path / "out", pin)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["bytes", "mode", "extra"])
def test_builder_refuses_a_dirty_or_wrong_tree(source, tmp_path, change):
    from tools.build_prismabuild_source_bundle import build_bundle

    root, pin = source
    if change == "bytes":
        (root / "src/plain").write_bytes(b"wrong\n")
    elif change == "mode":
        (root / "run").chmod(0o644)
    else:
        (root / "extra").write_bytes(b"untracked\n")
    with pytest.raises(ValueError, match="tree"):
        build_bundle(root, tmp_path / "out", pin)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("workflow", ["gate.yml", "ci.yml"])
def test_hosted_suites_build_the_bundle_and_export_its_locator(workflow):
    import yaml

    root = Path(__file__).resolve().parents[1]
    jobs = yaml.safe_load((root / ".github/workflows" / workflow).read_text())["jobs"]
    job = jobs["shards" if workflow == "gate.yml" else "tests"]
    steps = job["steps"]
    checkout = next(step for step in steps if step.get("with", {}).get("repository") == "RobTand/prismabuild")
    assert checkout["with"]["ref"] == "${{ steps.prismabuild-pin.outputs.commit }}"
    build = next(step for step in steps if "build_prismabuild_source_bundle.py" in step.get("run", ""))
    assert '"$RUNNER_TEMP/pb-source"' in build["run"]
    assert "PRISMAQUANT_PB_SOURCE_BUNDLE" in build["run"]
    assert '"$GITHUB_ENV"' in build["run"]
