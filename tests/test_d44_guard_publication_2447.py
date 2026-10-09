"""D44 guard publication ownership for PQ #2447."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path


def _load_guard(monkeypatch, tmp_path, *, path=None):
    """Load the guard with stub stage1/D30 owners."""
    target = Path(path) if path is not None else (
        Path(__file__).resolve().parents[1] / "prismaquant/d44_guard/campaign_launch.py")

    stage1 = types.ModuleType("stage1")

    def _sha(p) -> str:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def _save(p, value) -> None:
        out = Path(p)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(value, sort_keys=True))

    def _require(condition, message):
        if not condition:
            raise AssertionError(message)

    owner = tmp_path / "owner"
    owner.mkdir(parents=True, exist_ok=True)
    template = owner / "launch.json"
    template.write_text(json.dumps({"spec": {"container": {"mounts": []}, "env": {}}}))
    (owner / "v2_launch.py").write_text(
        f"TEMPLATE = {str(template)!r}\n"
        "def available():\n"
        "    return 16 * 2**30\n"
    )
    stage1.load = lambda p: json.loads(Path(p).read_text())  # type: ignore[attr-defined]
    stage1.save = _save  # type: ignore[attr-defined]
    stage1.sha = _sha  # type: ignore[attr-defined]
    stage1.OWNERS = owner  # type: ignore[attr-defined]
    stage1.require = _require  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stage1", stage1)

    residency = types.ModuleType("g3_residency")
    residency.container_contract = lambda: ([], {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "g3_residency", residency)

    import importlib.util

    spec = importlib.util.spec_from_file_location("d44_guard_campaign_launch", target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "d44_guard_campaign_launch", module)
    spec.loader.exec_module(module)
    return module


def _batch(root: Path, *, manifest: str) -> Path:
    receipt = root / "receipts" / "q.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text("{}\n")
    batch = {
        "parent_key": "a" * 64,
        "plan_key": "b" * 64,
        "child_ordinal": 0,
        "result_manifest_path": str(root / manifest),
        "tasks": [{"id": "t0", "output_id": "o0",
                   "payload": {"qname": "q"}}],
    }
    path = root / "batch.json"
    path.write_text(json.dumps(batch))
    return path


def _run_cpu_entry(monkeypatch, guard, root: Path, batch: Path, *, dry_run: bool,
                   child_rc: int = 0):
    """Run the CPU caller path of main. Report its return code."""
    monkeypatch.setattr(guard.subprocess, "call", lambda argv: child_rc)
    argv = ["campaign_launch.py", "--device", "cpu", "--root", str(root),
            "--stage", "encode", "--batch", str(batch)]
    if dry_run:
        argv.append("--dry-run")
    monkeypatch.setattr(sys, "argv", argv)
    return guard.main()


def _install_approved_copy(tmp_path) -> Path:
    """Write the approved defective source to a side path. Return it."""
    approved = Path("/home/rob/tmp/cl2447-recover-campaign_launch.py")
    assert approved.exists(), "approved owner source is absent"
    digest = hashlib.sha256(approved.read_bytes()).hexdigest()
    assert digest == "38ca21ef9156eaf0f706643dd14dd3258db658aa652a98a2832c440f1d99d2e9"
    side = tmp_path / "approved-campaign_launch.py"
    side.write_bytes(approved.read_bytes())
    return side


class _Child:
    """A finished container client. Tests set the return code."""

    def __init__(self, rc: int = 0):
        self.pid = 12345
        self._rc = rc

    def wait(self, timeout=None):
        return self._rc

    def poll(self):
        return self._rc


def _run_container_entry(monkeypatch, guard, root: Path, batch: Path, *, dry_run: bool,
                         child_rc: int = 0):
    """Run the container caller path of main. Report its return code."""
    monkeypatch.setattr(guard, "start_container_client", lambda argv: _Child(child_rc))
    monkeypatch.setattr(guard, "halt_child",
                        lambda child, owner, reason, **kw: {"ok": True, "halt": reason})
    monkeypatch.setenv(guard.OWNER_ENV, "0" * 64)
    argv = ["campaign_launch.py", "--device", "cuda", "--root", str(root),
            "--stage", "encode", "--batch", str(batch)]
    if dry_run:
        argv.append("--dry-run")
    monkeypatch.setattr(sys, "argv", argv)
    return guard.main()


def _guard_record(root: Path) -> dict:
    records = list(root.glob("encode-guard-*.json"))
    assert len(records) == 1
    return json.loads(records[0].read_text())


def test_dry_run_publishes_nothing_and_reports_false(monkeypatch, tmp_path):
    """A dry run writes no manifest. The flag stays false."""
    guard = _load_guard(monkeypatch, tmp_path)
    root = tmp_path / "root"
    batch = _batch(root, manifest="child.json")

    assert guard._publish_batch_result(batch, root, "encode", dry_run=True) is False
    assert not (root / "child.json").exists()


def test_real_run_writes_manifest_and_reports_true(monkeypatch, tmp_path):
    """A real run writes the manifest. The flag turns true."""
    guard = _load_guard(monkeypatch, tmp_path)
    root = tmp_path / "root"
    batch = _batch(root, manifest="child.json")

    assert guard._publish_batch_result(batch, root, "encode", dry_run=False) is True
    manifest = json.loads((root / "child.json").read_text())
    assert manifest["schema"] == "prismabuild.child_result_manifest.v1"
    assert manifest["results"][0]["output_id"] == "o0"


def test_failed_publication_keeps_published_false(monkeypatch, tmp_path):
    """A missing receipt fails publication. The caller keeps false."""
    guard = _load_guard(monkeypatch, tmp_path)
    root = tmp_path / "root"
    batch = _batch(root, manifest="child.json")
    (root / "receipts" / "q.json").unlink()

    published = False
    try:
        published = guard._publish_batch_result(batch, root, "encode", dry_run=False)
    except (OSError, AssertionError):
        published = False
    assert published is False
    assert not (root / "child.json").exists()


def test_cpu_dry_run_caller_writes_no_manifest(monkeypatch, tmp_path):
    """The CPU caller publishes nothing on a dry run."""
    guard = _load_guard(monkeypatch, tmp_path)
    root = tmp_path / "cpu-dry"
    batch = _batch(root, manifest="child.json")

    assert _run_cpu_entry(monkeypatch, guard, root, batch, dry_run=True) == 0
    assert not (root / "child.json").exists()


def test_cpu_real_run_caller_writes_manifest(monkeypatch, tmp_path):
    """The CPU caller publishes the manifest on a real run."""
    guard = _load_guard(monkeypatch, tmp_path)
    root = tmp_path / "cpu-real"
    batch = _batch(root, manifest="child.json")

    assert _run_cpu_entry(monkeypatch, guard, root, batch, dry_run=False) == 0
    manifest = json.loads((root / "child.json").read_text())
    assert manifest["schema"] == "prismabuild.child_result_manifest.v1"


def test_cpu_failed_child_caller_writes_no_manifest(monkeypatch, tmp_path):
    """The CPU caller publishes nothing when the child fails."""
    guard = _load_guard(monkeypatch, tmp_path)
    root = tmp_path / "cpu-fail"
    batch = _batch(root, manifest="child.json")

    assert _run_cpu_entry(monkeypatch, guard, root, batch, dry_run=False,
                          child_rc=1) == 1
    assert not (root / "child.json").exists()


def test_container_dry_run_caller_reports_unpublished(monkeypatch, tmp_path):
    """The container caller reports false after a dry run."""
    guard = _load_guard(monkeypatch, tmp_path)
    root = tmp_path / "guard-dry"
    root.mkdir(parents=True, exist_ok=True)
    batch = _batch(root, manifest="child.json")

    assert _run_container_entry(monkeypatch, guard, root, batch, dry_run=True) == 0
    saved = _guard_record(root)
    assert saved["published"] is False
    assert saved["returncode"] == 0
    assert not (root / "child.json").exists()


def test_container_real_run_caller_reports_published(monkeypatch, tmp_path):
    """The container caller reports true after it writes the manifest."""
    guard = _load_guard(monkeypatch, tmp_path)
    root = tmp_path / "guard-real"
    root.mkdir(parents=True, exist_ok=True)
    batch = _batch(root, manifest="child.json")

    assert _run_container_entry(monkeypatch, guard, root, batch, dry_run=False) == 0
    saved = _guard_record(root)
    assert saved["published"] is True
    manifest = json.loads((root / "child.json").read_text())
    assert manifest["schema"] == "prismabuild.child_result_manifest.v1"


def test_container_failed_child_caller_reports_unpublished(monkeypatch, tmp_path):
    """The container caller reports false when the child fails."""
    guard = _load_guard(monkeypatch, tmp_path)
    root = tmp_path / "guard-fail"
    root.mkdir(parents=True, exist_ok=True)
    batch = _batch(root, manifest="child.json")

    assert _run_container_entry(monkeypatch, guard, root, batch, dry_run=False,
                                child_rc=1) == 1
    saved = _guard_record(root)
    assert saved["published"] is False
    assert not (root / "child.json").exists()


def test_route_plan_selects_corrected_guard(monkeypatch, tmp_path):
    """Route-plan points the frozen entry at campaign_launch."""
    guard = _load_guard(monkeypatch, tmp_path)
    plan = {"common": {"argv": ["python3", "/workspace/encode_launch.py", "--device", "cpu"]},
            "roster": {"tasks": [{"residency_key": "GPU-A"}]},
            "batch_policy": {"residencies": [{"key": "GPU-A"}]}}
    src = tmp_path / "plan.json"
    out = tmp_path / "routed.json"
    src.write_text(json.dumps(plan))

    package = types.ModuleType("prismabuild")
    decomposition = types.ModuleType("prismabuild.decomposition")
    decomposition.validate_logical_request = lambda value: {}  # type: ignore[attr-defined]
    package.decomposition = decomposition  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prismabuild", package)
    monkeypatch.setitem(sys.modules, "prismabuild.decomposition", decomposition)
    guard.route_plan(src, out)
    routed = json.loads(out.read_text())
    assert Path(routed["common"]["argv"][1]).name == "campaign_launch.py"
    assert routed["roster"]["tasks"][0]["residency_key"] == "gpu-a"
    assert routed["batch_policy"]["residencies"][0]["key"] == "gpu-a"


def test_approved_caller_reports_published_for_dry_run(monkeypatch, tmp_path):
    """The approved caller is the red proof. It marks dry runs published."""
    side = _install_approved_copy(tmp_path)
    approved = _load_guard(monkeypatch, tmp_path, path=side)
    root = tmp_path / "approved-dry"
    root.mkdir(parents=True, exist_ok=True)
    batch = _batch(root, manifest="child.json")

    assert _run_container_entry(monkeypatch, approved, root, batch, dry_run=True) == 0
    saved = _guard_record(root)
    assert saved["published"] is True
    assert not (root / "child.json").exists()


def test_routed_entry_runs_corrected_guard(monkeypatch, tmp_path):
    """Route-plan selects a guard that reports dry runs unpublished."""
    guard = _load_guard(monkeypatch, tmp_path)
    entry = tmp_path / "owner-tree" / "encode_launch.py"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("# frozen entry\n")
    plan = {"common": {"argv": ["python3", str(entry), "--device", "cpu"]},
            "roster": {"tasks": [{"residency_key": "GPU-A"}]},
            "batch_policy": {"residencies": [{"key": "GPU-A"}]}}
    src = tmp_path / "plan.json"
    out = tmp_path / "routed.json"
    src.write_text(json.dumps(plan))

    package = types.ModuleType("prismabuild")
    decomposition = types.ModuleType("prismabuild.decomposition")
    decomposition.validate_logical_request = lambda value: {}  # type: ignore[attr-defined]
    package.decomposition = decomposition  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prismabuild", package)
    monkeypatch.setitem(sys.modules, "prismabuild.decomposition", decomposition)
    guard.route_plan(src, out)
    routed_argv = json.loads(out.read_text())["common"]["argv"]
    routed_entry = Path(routed_argv[1])
    assert routed_entry.name == "campaign_launch.py"
    fixed = Path(__file__).resolve().parents[1] / "prismaquant/d44_guard/campaign_launch.py"
    routed_entry.write_bytes(fixed.read_bytes())
    selected = _load_guard(monkeypatch, tmp_path, path=routed_entry)
    root = tmp_path / "routed-dry"
    root.mkdir(parents=True, exist_ok=True)
    batch = _batch(root, manifest="child.json")

    assert _run_container_entry(monkeypatch, selected, root, batch, dry_run=True) == 0
    saved = _guard_record(root)
    assert saved["published"] is False
    assert not (root / "child.json").exists()


def test_fix_changes_only_the_publication_lines(tmp_path):
    """The fix differs from approved source in three lines only."""
    _ = tmp_path
    approved_path = Path("/home/rob/tmp/cl2447-recover-campaign_launch.py")
    approved = approved_path.read_text()
    assert hashlib.sha256(approved.encode()).hexdigest() == (
        "38ca21ef9156eaf0f706643dd14dd3258db658aa652a98a2832c440f1d99d2e9")
    fixed_path = Path(__file__).resolve().parents[1] / "prismaquant/d44_guard/campaign_launch.py"
    fixed = fixed_path.read_text()
    expect = approved.replace(
        "    if dry_run:\n        return\n",
        '    """Write the owned scientific result manifest. Report if the write ran."""\n'
        "    if dry_run:\n        return False\n",
    ).replace(
        "                **{key: batch[key] for key in ('parent_key','plan_key','child_ordinal')}, 'results':results})\n",
        "                **{key: batch[key] for key in ('parent_key','plan_key','child_ordinal')}, 'results':results})\n"
        "    return True\n",
    ).replace(
        "            _publish_batch_result(args.batch, args.root, args.stage, dry_run=args.dry_run)\n"
        "            published = True\n",
        "            published = _publish_batch_result(args.batch, args.root, args.stage, dry_run=args.dry_run)\n",
    )
    assert fixed == expect
