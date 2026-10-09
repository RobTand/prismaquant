"""D44 guard publication ownership for PQ #2447."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path


def _load_guard(monkeypatch, tmp_path):
    """Load the guard with stub stage1/D30 owners.

    The guard imports the frozen G3 owner modules at load. The test
    owns minimal stubs with the same ``load/save/sha/require`` contract.
    """
    path = Path(__file__).resolve().parents[1] / "prismaquant/d44_guard/campaign_launch.py"

    stage1 = types.ModuleType("stage1")

    def _sha(p) -> str:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def _save(p, value) -> None:
        target = Path(p)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, sort_keys=True))

    def _require(condition, message):
        if not condition:
            raise AssertionError(message)

    stage1.load = lambda p: json.loads(Path(p).read_text())  # type: ignore[attr-defined]
    stage1.save = _save  # type: ignore[attr-defined]
    stage1.sha = _sha  # type: ignore[attr-defined]
    stage1.OWNERS = tmp_path / "owner"  # type: ignore[attr-defined]
    stage1.require = _require  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stage1", stage1)
    (tmp_path / "owner").mkdir(parents=True, exist_ok=True)
    (tmp_path / "owner" / "v2_launch.py").write_text(
        "TEMPLATE = 't'\n"
        "def available():\n"
        "    return 16 * 2**30\n"
    )

    import importlib.util

    spec = importlib.util.spec_from_file_location("d44_guard_campaign_launch", path)
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
