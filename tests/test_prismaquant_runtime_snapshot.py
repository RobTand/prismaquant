"""Content-addressed PrismaQuant runtime snapshot boundary."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "prismaquant_runtime_snapshot.py"


def _run(*args: object):
    return subprocess.run(
        [sys.executable, str(TOOL), *(str(value) for value in args)],
        check=False,
        capture_output=True,
        text=True,
    )


def _repository(root: Path) -> tuple[Path, str]:
    source = root / "source"
    (source / "prismaquant").mkdir(parents=True)
    (source / "tools").mkdir()
    (source / "prismaquant" / "__init__.py").write_text("VALUE = 1\n")
    (source / "prismaquant" / "container_runtime_identity.py").write_text("# tool\n")
    (source / "tools" / "prismaquant_runtime_snapshot.py").write_text("# self\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(source), "-c", "user.name=test",
            "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return source, commit


def test_materialize_is_content_addressed_and_ignores_live_worktree(tmp_path):
    source, commit = _repository(tmp_path)
    cache = tmp_path / "cache"
    first = _run(
        "materialize", "--source-root", source,
        "--cache-root", cache, "--commit", commit,
    )
    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout)
    snapshot = Path(payload["snapshot"])
    assert snapshot.is_dir()
    assert payload["commit"] == commit
    assert (snapshot / "prismaquant" / "__init__.py").read_text() == "VALUE = 1\n"

    (source / "prismaquant" / "__init__.py").write_text("VALUE = 2\n")
    repeated = _run(
        "materialize", "--source-root", source,
        "--cache-root", cache, "--commit", commit,
    )
    assert repeated.returncode == 0, repeated.stderr
    assert json.loads(repeated.stdout) == payload
    assert (snapshot / "prismaquant" / "__init__.py").read_text() == "VALUE = 1\n"


def test_verify_refuses_mutation_extra_files_and_wrong_transport_hash(tmp_path):
    source, commit = _repository(tmp_path)
    created = _run(
        "materialize", "--source-root", source,
        "--cache-root", tmp_path / "cache", "--commit", commit,
    )
    assert created.returncode == 0, created.stderr
    payload = json.loads(created.stdout)
    verify_args = (
        "verify", "--snapshot", payload["snapshot"],
        "--expected-commit", commit,
        "--expected-tree", payload["tree"],
        "--expected-closure-sha256", payload["closure_sha256"],
    )
    exact = _run(*verify_args)
    assert exact.returncode == 0, exact.stderr

    wrong = list(verify_args)
    wrong[-1] = "0" * 64
    refused_hash = _run(*wrong)
    assert refused_hash.returncode == 2
    assert "caller-attested" in refused_hash.stderr

    snapshot = Path(payload["snapshot"])
    (snapshot / "untracked.txt").write_text("unexpected\n")
    refused_extra = _run(*verify_args)
    assert refused_extra.returncode == 2
    assert "files differ" in refused_extra.stderr


def test_materialize_refuses_abbreviated_commit(tmp_path):
    source, commit = _repository(tmp_path)
    result = _run(
        "materialize", "--source-root", source,
        "--cache-root", tmp_path / "cache", "--commit", commit[:12],
    )
    assert result.returncode == 2
    assert "full lowercase 40-hex" in result.stderr
