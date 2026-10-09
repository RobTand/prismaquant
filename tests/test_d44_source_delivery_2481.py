"""D44 standalone source delivery through a repository-owned path (PQ #2481)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUARD_DIR = REPO / "prismaquant" / "d44_guard"
CPU_PYTHON = "/home/rob/venvs/pb-cpu/bin/python"

NEEDS_FLEET = pytest.mark.skipif(
    not Path("/mnt/shared/tessera-measurements/g3-v2-rebaseline-20261005/source/v2_launch.py").is_file(),
    reason="the bound external owners live on the fleet mount",
)


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


def _plan(entry: Path) -> dict:
    return {
        "schema": "prismabuild.logical_request.v1",
        "common": {
            "argv": ["python3", str(entry), "--device", "cpu",
                     "--batch", "{pb.task_batch}"],
            "cwd": ".",
            "demand": {"cpu": 1},
            "gpu_memory_gb": None,
            "data_manifest": None,
            "env": {},
        },
        "roster": {
            "schema": "prismabuild.logical_task_roster.v1",
            "tasks": [{"id": "t0", "payload": {}, "residency_key": "gpu-a",
                       "estimated_seconds": 1.0,
                       "estimate_evidence": "delivery proof",
                       "output_id": "o0"}],
        },
        "batch_policy": {
            "schema": "prismabuild.roster_batch_policy.v1",
            "residencies": [{"key": "gpu-a", "setup_seconds": 0.0,
                             "setup_evidence": "delivery proof"}],
            "max_setup_fraction": 0.5,
            "max_estimated_wall_seconds": 60.0,
        },
    }


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
        "d44_subsample.py", "d44.py", "d44_g3.py", "d44_weight_leg.py",
        "codec_factorial.py", "cd2_alignment.py", "source_file.py",
        "d44-frozen-method.json", "v2_launch.py", "v2_score.py",
        "encode_launch.py"}


def test_delivery_binds_every_external_owner():
    """Each external owner has an immutable digest binding."""
    delivery = _load_delivery()
    bindings = delivery.load_pin()["external_owner_bindings"]
    assert set(bindings["files"]) == {
        "v2_launch.py", "g3_residency.py", "v2_score.py",
        "g3_offline_decoded_kl.py", "g3_lib.py", "g3_readset.py",
        "exl3_torch.py"}
    assert bindings["container_template"]["sha256"] == (
        "ca1e014058f9c48ee88e16a471af59b301ff15afae1a5731649a8b79fd08734e")


@NEEDS_FLEET
def test_delivery_verifies_external_owners_against_bindings():
    """Each bound external owner matches its recorded digest."""
    delivery = _load_delivery()
    digests = delivery.verify_external_bindings()
    assert set(digests) == {
        "v2_launch.py", "g3_residency.py", "v2_score.py",
        "g3_offline_decoded_kl.py", "g3_lib.py", "g3_readset.py",
        "exl3_torch.py", "container_template"}


def test_recorded_source_seal_stamps_in_dev_mode(monkeypatch, tmp_path):
    """A drifted recorded head stamps in dev mode, not a refusal."""
    delivery = _load_delivery()
    pin = json.loads((GUARD_DIR / "standalone_source_pin.json").read_text())
    pin["accepted_head"] = "0" * 40
    drifted = tmp_path / "standalone_source_pin.json"
    drifted.write_text(json.dumps(pin))
    monkeypatch.setattr(delivery, "PIN_PATH", drifted)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    assert delivery.check_recorded_source() is False


def test_recorded_source_seal_refuses_in_certified_mode(monkeypatch, tmp_path):
    """A drifted recorded head refuses in certified mode."""
    delivery = _load_delivery()
    pin = json.loads((GUARD_DIR / "standalone_source_pin.json").read_text())
    pin["accepted_bundle_sha256"] = "0" * 64
    drifted = tmp_path / "standalone_source_pin.json"
    drifted.write_text(json.dumps(pin))
    monkeypatch.setattr(delivery, "PIN_PATH", drifted)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(ValueError, match="bundle digest drifted"):
        delivery.check_recorded_source()


@NEEDS_FLEET
def test_staged_tree_carries_guard_beside_owners(tmp_path):
    """The staged tree holds the guard beside its sibling owners."""
    delivery = _load_delivery()
    guard = delivery.stage_tree(tmp_path / "tree")
    assert guard.name == "campaign_launch.py"
    for name in ("stage1.py", "d44_training.py", "d44_subsample.py",
                 "d44.py", "d44_g3.py", "d44_weight_leg.py",
                 "codec_factorial.py", "cd2_alignment.py", "source_file.py",
                 "d44-frozen-method.json", "v2_launch.py",
                 "d44_v2_score_adapter.py", "encode_launch.py"):
        assert (guard.parent / name).is_file()
    assert hashlib.sha256(guard.read_bytes()).hexdigest() == (
        delivery.load_pin()["corrected_guard_sha256"])


def test_stage_refuses_an_existing_path(tmp_path):
    """Staging refuses an existing directory. It never overwrites."""
    delivery = _load_delivery()
    dest = tmp_path / "tree"
    dest.mkdir()
    marker = dest / "marker.txt"
    marker.write_text("active tree\n")
    with pytest.raises(FileExistsError, match="existing stage path"):
        delivery.stage_tree(dest)
    assert marker.read_text() == "active tree\n"


@NEEDS_FLEET
def test_staged_tree_imports_every_required_owner(tmp_path):
    """Each required owner imports from the staged or bound path."""
    delivery = _load_delivery()
    guard = delivery.stage_tree(tmp_path / "tree")
    staged = str(guard.parent.resolve())
    owners = str(delivery.owners_dir().resolve())
    script = (
        "import importlib, sys\n"
        f"sys.path.insert(0, {staged!r})\n"
        f"sys.path.insert(0, {owners!r})\n"
        f"for name in {list(delivery.REQUIRED_STAGED_IMPORTS)!r}:\n"
        "    module = importlib.import_module(name)\n"
        f"    assert module.__file__.startswith({staged!r} + '/'), name\n"
        f"for name in {list(delivery.REQUIRED_EXTERNAL_IMPORTS)!r}:\n"
        "    module = importlib.import_module(name)\n"
        f"    assert module.__file__.startswith({owners!r} + '/'), name\n"
        "print('owners-ok')\n"
    )
    done = subprocess.run(
        [CPU_PYTHON, "-c", script], capture_output=True, text=True,
        timeout=300)
    assert done.returncode == 0, done.stderr
    assert "owners-ok" in done.stdout


@NEEDS_FLEET
def test_bound_d30_names_the_bound_template():
    """The bound D30 launcher names the bound container template."""
    delivery = _load_delivery()
    template = delivery.check_d30_binding()
    assert template == Path(
        "/mnt/shared/tessera-measurements/g3-v2-teacher-20261005/"
        "gpu-score/probe-v2teacher.launch.json")


@NEEDS_FLEET
def test_bound_residency_supplies_container_contract():
    """The bound g3_residency owner supplies container_contract."""
    delivery = _load_delivery()
    delivery.require_container_residency()
    import g3_residency

    assert g3_residency.__file__.startswith(
        str(delivery.owners_dir()) + "/")
    mounts, env = g3_residency.container_contract()
    assert isinstance(mounts, list) and isinstance(env, dict)


@NEEDS_FLEET
def test_routed_entry_selects_delivered_guard_in_subprocess(tmp_path):
    """Route-plan runs the delivered guard in a fresh subprocess.

    The subprocess imports the staged guard with real owners. No test
    replaces stage1, D30, residency, or the child client. The routed
    entry reports a dry run unpublished.
    """
    delivery = _load_delivery()
    entry = tmp_path / "owner-tree" / "encode_launch.py"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("# frozen entry\n")
    before = entry.read_bytes()
    src = tmp_path / "plan.json"
    out = tmp_path / "routed.json"
    src.write_text(json.dumps(_plan(entry)))
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(GUARD_DIR)!r})\n"
        "import source_delivery as delivery\n"
        "from pathlib import Path\n"
        "guard = delivery.stage_fresh()\n"
        "print(str(guard))\n"
    )
    done = subprocess.run(
        [CPU_PYTHON, "-c", script], capture_output=True, text=True,
        timeout=300, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr
    staged_guard = Path(done.stdout.strip())
    assert staged_guard.name == "campaign_launch.py"
    assert staged_guard.parent != entry.parent
    assert entry.read_bytes() == before
    assert list(entry.parent.iterdir()) == [entry]

    run_script = (
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(staged_guard.parent)!r})\n"
        f"sys.path.insert(0, {str(delivery.owners_dir())!r})\n"
        "import campaign_launch as guard\n"
        f"assert guard.__file__ == {str(staged_guard)!r}\n"
        f"assert str(guard.OWNER) == {str(delivery.owners_dir())!r}\n"
        f"root = Path({str(tmp_path / 'routed-dry')!r})\n"
        "root.mkdir(parents=True, exist_ok=True)\n"
        "receipt = root / 'receipts' / 'q.json'\n"
        "receipt.parent.mkdir(parents=True, exist_ok=True)\n"
        "receipt.write_text('{}\\n')\n"
        "batch = {'parent_key': 'a' * 64, 'plan_key': 'b' * 64,\n"
        "         'child_ordinal': 0,\n"
        "         'result_manifest_path': str(root / 'child.json'),\n"
        "         'tasks': [{'id': 't0', 'output_id': 'o0',\n"
        "                    'payload': {'qname': 'q'}}]}\n"
        "batch_path = root / 'batch.json'\n"
        "batch_path.write_text(json.dumps(batch))\n"
        "published = guard._publish_batch_result(\n"
        "    batch_path, root, 'encode', dry_run=True)\n"
        "assert published is False\n"
        "assert not (root / 'child.json').exists()\n"
        "print('routed-guard-ok')\n"
    )
    run = subprocess.run(
        [CPU_PYTHON, "-c", run_script], capture_output=True, text=True,
        timeout=300, cwd=str(tmp_path))
    assert run.returncode == 0, run.stderr
    assert "routed-guard-ok" in run.stdout
