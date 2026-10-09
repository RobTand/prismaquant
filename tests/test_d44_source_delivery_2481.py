"""D44 standalone source delivery through a repository-owned path (PQ #2481)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parents[1] / "prismaquant" / "d44_guard"


def _load_delivery():
    name = "prismaquant_d44_guard_source_delivery"
    found = sys.modules.get(name)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(
        name, GUARD_DIR / "source_delivery.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stub_stage1(monkeypatch, tmp_path):
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
    return stage1


def _load_guard(monkeypatch, tmp_path, *, path):
    _stub_stage1(monkeypatch, tmp_path)
    name = "d44_delivered_guard_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    saved_path = list(sys.path)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "path", saved_path, raising=False)
    sys.path[:] = saved_path
    return module


def _batch(root: Path) -> Path:
    receipt = root / "receipts" / "q.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text("{}\n")
    batch = {
        "parent_key": "a" * 64,
        "plan_key": "b" * 64,
        "child_ordinal": 0,
        "result_manifest_path": str(root / "child.json"),
        "tasks": [{"id": "t0", "output_id": "o0",
                   "payload": {"qname": "q"}}],
    }
    path = root / "batch.json"
    path.write_text(json.dumps(batch))
    return path


class _Child:
    """A finished container client. Tests set the return code."""

    def __init__(self, rc: int = 0):
        self.pid = 12345
        self._rc = rc

    def wait(self, timeout=None):
        return self._rc

    def poll(self):
        return self._rc


def test_pin_binds_accepted_head_and_bundle():
    """The pin names the accepted head and its bundle digest."""
    delivery = _load_delivery()
    pin = delivery.load_pin()
    assert pin["accepted_head"] == "fa3775151f77dc713fd78882daf6e147ab471243"
    assert pin["accepted_bundle_sha256"] == (
        "7ea79d62000dc4a41d940a9c72eb925e6dbd2b752e49dccd25b22b3a78c998e0")
    assert pin["accepted_source_branch"] == (
        "campaign/d44-autonomous-phase-2449-20261009")


def test_delivery_verifies_every_vendored_owner():
    """Each vendored owner matches its pinned digest."""
    delivery = _load_delivery()
    digests = delivery.verify_vendored()
    assert set(digests) == {
        "campaign_launch.py", "stage1.py", "d44_training.py",
        "d44_subsample.py", "v2_launch.py", "encode_launch.py"}


def test_staged_tree_carries_guard_beside_owners(tmp_path):
    """The staged tree holds the guard beside its sibling owners."""
    delivery = _load_delivery()
    guard = delivery.stage_tree(tmp_path / "tree")
    assert guard.name == "campaign_launch.py"
    for name in ("stage1.py", "d44_training.py", "d44_subsample.py",
                 "v2_launch.py"):
        assert (guard.parent / name).is_file()
    assert hashlib.sha256(guard.read_bytes()).hexdigest() == (
        delivery.load_pin()["corrected_guard_sha256"])


def test_routed_entry_selects_delivered_guard(monkeypatch, tmp_path):
    """Route-plan stages the delivered guard. No test copies source."""
    delivery = _load_delivery()
    _stub_stage1(monkeypatch, tmp_path)
    entry = tmp_path / "owner-tree" / "encode_launch.py"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("# frozen entry\n")
    staged_dir = tmp_path / "staged"
    saved_path = list(sys.path)
    guard = delivery.load_staged_guard(staged_dir)
    sys.path[:] = saved_path
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
    routed_entry = Path(json.loads(out.read_text())["common"]["argv"][1])
    assert routed_entry.name == "campaign_launch.py"
    assert routed_entry.parent == entry.parent
    assert hashlib.sha256(routed_entry.read_bytes()).hexdigest() == (
        delivery.load_pin()["corrected_guard_sha256"])
    for name in ("stage1.py", "d44_training.py", "d44_subsample.py",
                 "v2_launch.py"):
        assert (routed_entry.parent / name).is_file()

    selected = _load_guard(monkeypatch, tmp_path, path=routed_entry)
    assert selected._DELIVERY is not None
    root = tmp_path / "routed-dry"
    root.mkdir(parents=True, exist_ok=True)
    batch = _batch(root)
    monkeypatch.setattr(selected, "start_container_client",
                        lambda argv: _Child(0))
    monkeypatch.setattr(selected, "halt_child",
                        lambda child, owner, reason, **kw: {"ok": True, "halt": reason})
    monkeypatch.setenv(selected.OWNER_ENV, "0" * 64)
    monkeypatch.setattr(sys, "argv",
                        ["campaign_launch.py", "--device", "cuda", "--root", str(root),
                         "--stage", "encode", "--batch", str(batch), "--dry-run"])
    assert selected.main() == 0
    records = list(root.glob("encode-guard-*.json"))
    assert len(records) == 1
    saved = json.loads(records[0].read_text())
    assert saved["published"] is False
    assert not (root / "child.json").exists()


def test_container_path_refuses_without_residency(monkeypatch, tmp_path):
    """The container gate refuses when no host supplies g3_residency."""
    delivery = _load_delivery()
    monkeypatch.delitem(sys.modules, "g3_residency", raising=False)
    empty = tmp_path / "empty-path"
    empty.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sys, "path", [str(empty)], raising=False)
    try:
        delivery.require_container_residency()
    except ImportError:
        return
    raise AssertionError("delivery admits the container path without g3_residency")


def test_pin_records_missing_residency_binding():
    """The pin records the absent g3_residency owner explicitly."""
    delivery = _load_delivery()
    missing = delivery.load_pin()["missing_owner_bindings"]
    assert "g3_residency.py" in missing
