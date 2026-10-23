"""The opt-in Stage A benchmark must own a run root before deleting it."""
from pathlib import Path

import pytest

import bench_stage_a_io_overlap as bench


pytestmark = pytest.mark.own_process


class _CaptureBoundaryReached(Exception):
    pass


@pytest.fixture
def guarded_capture(tmp_path, monkeypatch):
    import test_stage_a_chain_resume as capture

    calls = []

    def stop_before_capture(root, patch):
        root = Path(root)
        calls.append(root)
        raise _CaptureBoundaryReached(
            f"expensive capture boundary reached; root_exists={root.exists()}")

    monkeypatch.setattr(capture, "_run", stop_before_capture)
    monkeypatch.setenv(bench.OUT_ENV, str(tmp_path / "bench-results"))

    def run(root):
        monkeypatch.setenv(bench.RUN_ROOT_ENV, str(root))
        return bench.test_capture_manifest(tmp_path, monkeypatch)

    return run, calls


def test_stage_a_bench_refuses_existing_unowned_run_root(tmp_path, guarded_capture):
    run, calls = guarded_capture
    root = tmp_path / "existing-run"
    root.mkdir()
    sentinel = root / "original-evidence.txt"
    original = b"preserve existing evidence\n"
    sentinel.write_bytes(original)

    try:
        with pytest.raises(FileExistsError):
            run(root)
    finally:
        assert sentinel.exists(), "the real benchmark deleted a preexisting root"
        assert sentinel.read_bytes() == original
    assert calls == []
    assert sorted(path.name for path in root.iterdir()) == [sentinel.name]


def test_stage_a_bench_fresh_run_root_reaches_capture_boundary(tmp_path, guarded_capture):
    run, calls = guarded_capture
    root = tmp_path / "fresh-run"
    assert not root.exists()
    with pytest.raises(_CaptureBoundaryReached):
        run(root)
    assert calls == [root]


def test_owned_root_repeats_at_one_path_and_preserves_report_evidence(tmp_path, monkeypatch):
    import hashlib
    import json
    import test_stage_a_chain_resume as capture

    root = tmp_path / "owned-run"
    output = tmp_path / "report"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "retain.txt"
    sentinel.write_bytes(b"external evidence")
    calls = []

    def tiny_capture(path, patch):
        path = Path(path)
        assert path == root and path.is_dir()
        assert list(path.iterdir()) == [], "the second capture retained the first tree"
        calls.append(path)
        (path / "nested").mkdir()
        (path / "nested" / "payload.bin").write_bytes(b"stable capture")
        (path / "metadata.json").write_text(json.dumps({
            "root": str(path), "telemetry": {"attempt": len(calls)}}))
        (path / "external-link").symlink_to(external, target_is_directory=True)

    monkeypatch.setattr(capture, "_run", tiny_capture)
    monkeypatch.setenv(bench.OUT_ENV, str(output))
    monkeypatch.setenv(bench.RUN_ROOT_ENV, str(root))
    bench.test_capture_manifest(tmp_path, monkeypatch)

    assert calls == [root, root] and not root.exists()
    report = json.loads((output / "capture.json").read_text())
    assert report["repeat_identical"] is True and report["root"] == str(root)
    assert report["files"]["nested/payload.bin"] == hashlib.sha256(b"stable capture").hexdigest()
    assert (output / "capture-run" / "nested" / "payload.bin").read_bytes() == b"stable capture"
    assert sentinel.read_bytes() == b"external evidence"


def test_competing_invocation_cannot_remove_an_active_owned_root(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import test_stage_a_chain_resume as capture

    root = tmp_path / "active-run"
    calls = []

    def tiny_capture(path, patch):
        calls.append(Path(path))
        sentinel = Path(path) / "active-evidence.txt"
        sentinel.write_bytes(b"owned by the first invocation")
        if len(calls) == 1:
            with ThreadPoolExecutor(max_workers=1) as pool:
                competitor = pool.submit(bench.test_capture_manifest, tmp_path, monkeypatch)
                with pytest.raises(FileExistsError):
                    competitor.result(timeout=5)
            assert sentinel.read_bytes() == b"owned by the first invocation"

    monkeypatch.setattr(capture, "_run", tiny_capture)
    monkeypatch.setenv(bench.OUT_ENV, str(tmp_path / "report"))
    monkeypatch.setenv(bench.RUN_ROOT_ENV, str(root))
    bench.test_capture_manifest(tmp_path, monkeypatch)
    assert calls == [root, root] and not root.exists()


@pytest.mark.parametrize("replacement", [False, True], ids=["renamed", "replaced"])
def test_changed_root_is_refused_without_deleting_either_tree(tmp_path, monkeypatch, replacement):
    import test_stage_a_chain_resume as capture

    root = tmp_path / "run"
    moved = tmp_path / "moved-owned-run"
    calls = []

    def move_root(path, patch):
        calls.append(Path(path))
        (Path(path) / "owned.txt").write_bytes(b"owned evidence")
        Path(path).rename(moved)
        if replacement:
            Path(path).mkdir()
            (Path(path) / "foreign.txt").write_bytes(b"foreign evidence")

    monkeypatch.setattr(capture, "_run", move_root)
    monkeypatch.setenv(bench.OUT_ENV, str(tmp_path / "report"))
    monkeypatch.setenv(bench.RUN_ROOT_ENV, str(root))
    with pytest.raises(RuntimeError, match="run root changed"):
        bench.test_capture_manifest(tmp_path, monkeypatch)
    assert calls == [root]
    assert (moved / "owned.txt").read_bytes() == b"owned evidence"
    if replacement:
        assert (root / "foreign.txt").read_bytes() == b"foreign evidence"


@pytest.mark.parametrize("nested", [False, True], ids=["equal", "nested"])
def test_report_inside_run_root_refuses_before_creating_directories(tmp_path, monkeypatch, nested):
    import test_stage_a_chain_resume as capture

    root = tmp_path / "run"
    output = root / "report" if nested else root
    monkeypatch.setattr(capture, "_run", lambda *args: pytest.fail("capture started"))
    monkeypatch.setenv(bench.OUT_ENV, str(output))
    monkeypatch.setenv(bench.RUN_ROOT_ENV, str(root))
    with pytest.raises(ValueError, match="must be outside"):
        bench.test_capture_manifest(tmp_path, monkeypatch)
    assert not root.exists() and not output.exists()


def test_failed_second_capture_retains_the_first_report_copy(tmp_path, monkeypatch):
    import test_stage_a_chain_resume as capture

    root = tmp_path / "run"
    output = tmp_path / "report"
    calls = []

    def tiny_capture(path, patch):
        calls.append(Path(path))
        if len(calls) == 2:
            raise _CaptureBoundaryReached("second capture stopped")
        (Path(path) / "first.txt").write_bytes(b"retained first capture")

    monkeypatch.setattr(capture, "_run", tiny_capture)
    monkeypatch.setenv(bench.OUT_ENV, str(output))
    monkeypatch.setenv(bench.RUN_ROOT_ENV, str(root))
    with pytest.raises(_CaptureBoundaryReached, match="second capture stopped"):
        bench.test_capture_manifest(tmp_path, monkeypatch)
    assert calls == [root, root] and not root.exists()
    assert (output / "capture-run" / "first.txt").read_bytes() == b"retained first capture"
    assert not (output / "capture.json").exists()
