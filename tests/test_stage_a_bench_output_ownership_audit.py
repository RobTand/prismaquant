"""Real capture entry must not pair an old retained tree with a new report.

Only capture._run is doubled. These tests perform no Stage A workload,
subprocess, GPU work or recursive PrismaBuild submission.
"""
import hashlib
import json
from pathlib import Path

import pytest

import bench_stage_a_io_overlap as bench

pytestmark = pytest.mark.own_process


def _tiny_capture_entry(tmp_path, monkeypatch):
    import test_stage_a_chain_resume as capture

    calls = []
    payload = [b"first capture bytes"]

    def tiny_capture(root, patch):
        root = Path(root)
        calls.append(root)
        # The historical entry gives capture an absent root; the held root
        # repair creates an empty owned root first. Both are real contracts.
        root.mkdir(parents=True, exist_ok=True)
        (root / "payload.bin").write_bytes(payload[0])

    monkeypatch.setattr(capture, "_run", tiny_capture)

    def run(root, output):
        monkeypatch.setenv(bench.RUN_ROOT_ENV, str(root))
        monkeypatch.setenv(bench.OUT_ENV, str(output))
        bench.test_capture_manifest(tmp_path, monkeypatch)

    return run, calls, payload


def test_same_output_refuses_a_new_capture_and_preserves_first_evidence(tmp_path, monkeypatch):
    run, calls, payload = _tiny_capture_entry(tmp_path, monkeypatch)
    first_root, second_root = tmp_path / "first-run", tmp_path / "second-run"
    output = tmp_path / "shared-report"
    run(first_root, output)
    first_report = (output / "capture.json").read_bytes()
    first_payload = (output / "capture-run" / "payload.bin").read_bytes()
    assert json.loads(first_report)["files"]["payload.bin"] == hashlib.sha256(first_payload).hexdigest()
    assert calls == [first_root, first_root]

    payload[0] = b"different second capture bytes"
    try:
        run(second_root, output)
    except FileExistsError:
        pass
    else:
        report = json.loads((output / "capture.json").read_bytes())
        retained = (output / "capture-run" / "payload.bin").read_bytes()
        assert report["files"]["payload.bin"] == hashlib.sha256(retained).hexdigest(), (
            "the real capture entry attributed new payload bytes to the old retained capture-run")
        pytest.fail("a second capture must refuse preexisting capture-owned output products")

    assert (output / "capture.json").read_bytes() == first_report
    assert (output / "capture-run" / "payload.bin").read_bytes() == first_payload
    assert calls == [first_root, first_root], "refusal must precede expensive capture"
    assert not second_root.exists(), "refusal must precede run-root mutation"


def test_separate_outputs_keep_each_capture_report_bound_to_its_tree(tmp_path, monkeypatch):
    run, calls, payload = _tiny_capture_entry(tmp_path, monkeypatch)
    first_root, second_root = tmp_path / "first-run", tmp_path / "second-run"
    first_out, second_out = tmp_path / "base-report", tmp_path / "head-report"
    # Other benchmark products are allowed in the shared output directory.
    first_out.mkdir()
    unrelated = first_out / "chain.json"
    unrelated.write_bytes(b"unrelated benchmark evidence")
    run(first_root, first_out)
    payload[0] = b"different second capture bytes"
    run(second_root, second_out)

    for output, expected in ((first_out, b"first capture bytes"),
                             (second_out, b"different second capture bytes")):
        report = json.loads((output / "capture.json").read_bytes())
        retained = (output / "capture-run" / "payload.bin").read_bytes()
        assert retained == expected
        assert report["repeat_identical"] is True
        assert report["files"]["payload.bin"] == hashlib.sha256(retained).hexdigest()
    assert unrelated.read_bytes() == b"unrelated benchmark evidence"
    assert calls == [first_root, first_root, second_root, second_root]


@pytest.mark.parametrize("product", ["report", "tree", "tree-link", "dangling-report-link"])
def test_existing_capture_product_refuses_before_root_or_capture_mutation(tmp_path, monkeypatch, product):
    run, calls, payload = _tiny_capture_entry(tmp_path, monkeypatch)
    output = tmp_path / "shared-report"
    output.mkdir()
    external = tmp_path / "external-evidence"
    external.mkdir()
    sentinel = external / "retain.txt"
    sentinel.write_bytes(b"independently owned bytes")
    if product == "report":
        (output / "capture.json").write_bytes(b"prior report")
    elif product == "tree":
        (output / "capture-run").mkdir()
        (output / "capture-run" / "prior.bin").write_bytes(b"prior capture")
    elif product == "tree-link":
        (output / "capture-run").symlink_to(external, target_is_directory=True)
    else:
        (output / "capture.json").symlink_to(tmp_path / "absent-report")
    root = tmp_path / "must-stay-absent"
    with pytest.raises(FileExistsError):
        run(root, output)
    assert calls == [] and not root.exists()
    assert sentinel.read_bytes() == b"independently owned bytes"
    if product == "report":
        assert (output / "capture.json").read_bytes() == b"prior report"
        assert not (output / "capture-run").exists()
    elif product == "tree":
        assert (output / "capture-run" / "prior.bin").read_bytes() == b"prior capture"
    else:
        name = "capture-run" if product == "tree-link" else "capture.json"
        assert (output / name).is_symlink()


def test_distinct_run_roots_cannot_capture_concurrently_into_one_output(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import test_stage_a_chain_resume as capture

    first_root, competing_root = tmp_path / "first", tmp_path / "competing"
    output = tmp_path / "shared-report"
    monkeypatch.setenv(bench.OUT_ENV, str(output))
    monkeypatch.setenv(bench.RUN_ROOT_ENV, str(first_root))
    calls = []

    def tiny_capture(root, patch):
        root = Path(root)
        calls.append(root)
        assert root == first_root, "competing invocation crossed capture admission"
        (root / "payload.bin").write_bytes(b"first invocation evidence")
        if len(calls) == 1:
            # Both callers share OUT, while the second gets a genuinely absent
            # independent root. Its refusal must come from output ownership.
            with monkeypatch.context() as competing:
                competing.setenv(bench.RUN_ROOT_ENV, str(competing_root))
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(bench.test_capture_manifest, tmp_path, competing)
                    with pytest.raises(FileExistsError):
                        future.result(timeout=5)
            assert not competing_root.exists()

    monkeypatch.setattr(capture, "_run", tiny_capture)
    bench.test_capture_manifest(tmp_path, monkeypatch)
    assert calls == [first_root, first_root]
    report = json.loads((output / "capture.json").read_bytes())
    retained = (output / "capture-run" / "payload.bin").read_bytes()
    assert retained == b"first invocation evidence"
    assert report["files"]["payload.bin"] == hashlib.sha256(retained).hexdigest()


def test_late_report_publication_collision_preserves_both_owners_bytes(tmp_path, monkeypatch):
    import test_stage_a_chain_resume as capture

    root, output = tmp_path / "run", tmp_path / "output"
    foreign_report = b'{"independent": "late publisher"}\n'
    calls = []

    def tiny_capture(path, patch):
        calls.append(Path(path))
        (Path(path) / "payload.bin").write_bytes(b"captured bytes")
        if len(calls) == 2:
            (output / "capture.json").write_bytes(foreign_report)

    monkeypatch.setattr(capture, "_run", tiny_capture)
    monkeypatch.setenv(bench.OUT_ENV, str(output))
    monkeypatch.setenv(bench.RUN_ROOT_ENV, str(root))
    with pytest.raises(FileExistsError):
        bench.test_capture_manifest(tmp_path, monkeypatch)
    assert calls == [root, root] and not root.exists()
    assert (output / "capture.json").read_bytes() == foreign_report
    assert (output / "capture-run" / "payload.bin").read_bytes() == b"captured bytes"


@pytest.mark.parametrize("replacement", ["retained-tree", "output-directory"])
def test_replaced_capture_output_refuses_without_writing_foreign_tree(tmp_path, monkeypatch, replacement):
    import test_stage_a_chain_resume as capture

    root, output = tmp_path / "run", tmp_path / "output"
    moved = tmp_path / "moved-claim"
    calls = []

    def replace_output(path, patch):
        calls.append(Path(path))
        (Path(path) / "payload.bin").write_bytes(b"our capture")
        owned = output / "capture-run" if replacement == "retained-tree" else output
        owned.rename(moved)
        if replacement == "output-directory":
            output.mkdir()
        (output / "capture-run").mkdir()
        (output / "capture-run" / "foreign.bin").write_bytes(b"foreign capture")

    monkeypatch.setattr(capture, "_run", replace_output)
    monkeypatch.setenv(bench.OUT_ENV, str(output))
    monkeypatch.setenv(bench.RUN_ROOT_ENV, str(root))
    with pytest.raises(RuntimeError, match="capture output changed"):
        bench.test_capture_manifest(tmp_path, monkeypatch)
    assert calls == [root] and not root.exists()
    moved_tree = moved if replacement == "retained-tree" else moved / "capture-run"
    assert moved_tree.is_dir() and list(moved_tree.iterdir()) == []
    assert list((output / "capture-run").iterdir()) == [output / "capture-run" / "foreign.bin"]
    assert (output / "capture-run" / "foreign.bin").read_bytes() == b"foreign capture"
    assert not (output / "capture.json").exists()
