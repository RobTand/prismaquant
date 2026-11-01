"""A private receipt demo must refuse an existing root without deleting it."""
from __future__ import annotations

from pathlib import Path

import pytest

from experiments import receipt_demo as demo


pytestmark = pytest.mark.own_process


class _DeletionAttempt(Exception):
    pass


class _ReachedQueueBoundary(Exception):
    pass


@pytest.fixture
def guarded_demo(tmp_path, monkeypatch):
    import prismabuild.core as pb
    from prismabuild import pool

    queue_calls = []
    process_calls = []

    def process(command, **kwargs):
        process_calls.append(command)
        raise _DeletionAttempt(f"demo requested an external process before admission: {command}")

    def queue(path):
        queue_calls.append(Path(path))
        raise _ReachedQueueBoundary

    def forbidden_build(*args, **kwargs):
        raise AssertionError("receipt-root fixture must stop before build/CAS execution")

    monkeypatch.setattr(demo.subprocess, "run", process)
    monkeypatch.setattr(pool, "PoolQueue", queue)
    monkeypatch.setattr(pb, "PrismaBuildCAS", forbidden_build)
    monkeypatch.setattr(demo.sys, "path", list(demo.sys.path))

    def run(root):
        monkeypatch.setattr(demo.sys, "argv", [
            "receipt_demo.py", "--runtime", str(tmp_path / "unused-runtime"),
            "--root", str(root), "--manifest", str(tmp_path / "not-read.json"),
        ])
        return demo.main()

    return run, queue_calls, process_calls


def test_receipt_demo_refuses_existing_root_without_deleting_payload(tmp_path, guarded_demo):
    run, queue_calls, process_calls = guarded_demo
    root = tmp_path / "existing"
    root.mkdir()
    sentinel = root / "irreplaceable.txt"
    sentinel.write_bytes(b"preserve this evidence\n")
    before = sentinel.read_bytes()
    try:
        with pytest.raises(FileExistsError):
            run(root)
    finally:
        assert sentinel.read_bytes() == before
    assert process_calls == []
    assert queue_calls == []
    assert sorted(path.name for path in root.iterdir()) == ["irreplaceable.txt"]


def test_receipt_demo_fresh_root_reaches_only_private_queue_boundary(tmp_path, guarded_demo):
    run, queue_calls, process_calls = guarded_demo
    root = tmp_path / "fresh"
    assert not root.exists()
    with pytest.raises(_ReachedQueueBoundary):
        run(root)
    assert root.is_dir()
    assert list(root.iterdir()) == []
    assert queue_calls == [root / "pb-queue"]
    assert process_calls == []
