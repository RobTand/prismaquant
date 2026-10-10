"""D44 standalone source delivery through a repository-owned path (PQ #2481)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUARD_DIR = REPO / "prismaquant" / "d44_guard"
CPU_PYTHON = sys.executable

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
            "argv": [CPU_PYTHON, str(entry), "--device", "cpu",
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
@pytest.mark.parametrize("stage,child", [
    ("encode", "stage1.py"),
    ("select-unit", "d44_training.py"),
    ("encode-cost", "d44_subsample.py"),
])
def test_routed_entry_selects_delivered_guard_in_subprocess(tmp_path, stage, child):
    """The real route and guard execute the delivered child's argument parser."""
    delivery = _load_delivery()
    original = tmp_path / "active"
    original.mkdir()
    for name in ("encode_launch.py", child, "v2_launch.py"):
        (original / name).write_text("raise SystemExit('old source selected')\n")
    before = {p.name: p.read_bytes() for p in original.iterdir()}
    root = tmp_path / "output"
    plan = _plan(original / "encode_launch.py")
    plan["common"]["cwd"] = str(original)
    plan["common"]["argv"] += [
        "--stage", stage, "--root", str(root), "--dry-run", "--capture"]
    plan["roster"]["tasks"][0]["residency_key"] = "GPU-A"
    plan["batch_policy"]["residencies"][0]["key"] = "GPU-A"
    source = tmp_path / "plan.json"
    out = tmp_path / "routed.json"
    source.write_text(json.dumps(plan))
    routed = subprocess.run(
        [CPU_PYTHON, str(GUARD_DIR / "campaign_launch.py"),
         "route-plan", str(source), "--out", str(out)],
        cwd=original, capture_output=True, text=True, timeout=300)
    assert routed.returncode == 0, routed.stderr
    request = json.loads(out.read_text())
    entry = Path(request["common"]["cwd"]) / request["common"]["argv"][1]
    assert entry == GUARD_DIR / "campaign_launch.py"
    assert request["common"]["cwd"] == str(REPO)
    assert hashlib.sha256(entry.read_bytes()).hexdigest() == (
        delivery.load_pin()["corrected_guard_sha256"])
    assert request["roster"]["tasks"][0]["residency_key"] == "gpu-a"
    assert request["batch_policy"]["residencies"][0]["key"] == "gpu-a"
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps({"result_manifest_path": str(root / "child.json")}))
    argv = [str(batch) if token == "{pb.task_batch}" else token
            for token in request["common"]["argv"]]
    env = {**os.environ, **request["common"]["env"]}
    run = subprocess.run(argv, cwd=request["common"]["cwd"], env=env,
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 2, run.stderr
    assert f"{child} {stage}:" in run.stderr
    assert "argument --capture: expected one argument" in run.stderr
    assert "old source selected" not in run.stderr
    assert not root.exists()
    assert {p.name: p.read_bytes() for p in original.iterdir()} == before
    assert json.loads(source.read_text()) == plan


@NEEDS_FLEET
def test_native_refusal_precedes_source_staging(tmp_path):
    """Native request validation refuses before a source tree or output exists."""
    stage_root = tmp_path / "delivery"
    plan = _plan(tmp_path / "encode_launch.py")
    plan["roster"]["tasks"][0]["estimated_seconds"] = -1
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    out = tmp_path / "routed.json"
    done = subprocess.run(
        [CPU_PYTHON, str(GUARD_DIR / "campaign_launch.py"),
         "route-plan", str(source), "--out", str(out)],
        env={**os.environ, "D44_DELIVERY_STAGE_PARENT": str(stage_root)},
        capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert "estimated_seconds must be greater than zero" in done.stderr
    assert not out.exists()
    assert not stage_root.exists()
    assert json.loads(source.read_text()) == plan


@NEEDS_FLEET
@pytest.mark.parametrize("binding", ["owner", "template"])
@pytest.mark.parametrize("mode", ["1", "0"])
def test_external_provenance_follows_d32(binding, mode, monkeypatch, tmp_path, capsys):
    """Recorded source drift stamps in dev mode and refuses in certified mode."""
    delivery = _load_delivery()
    pin = delivery.load_pin()
    if binding == "owner":
        pin["external_owner_bindings"]["files"]["g3_residency.py"] = "0" * 64
    else:
        pin["external_owner_bindings"]["container_template"]["sha256"] = "0" * 64
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(pin))
    monkeypatch.setattr(delivery, "PIN_PATH", path)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    if mode == "0":
        with pytest.raises(ValueError, match="changed"):
            delivery.verify_external_bindings()
    else:
        actual = delivery.verify_external_bindings()
        assert actual["g3_residency.py"] == (
            "22c2068a92c516c8bae2564b589e8228b47d5ce145c5e03ba0d898773e0548e8")
        assert actual["container_template"] == (
            "ca1e014058f9c48ee88e16a471af59b301ff15afae1a5731649a8b79fd08734e")
        assert "[DEV-MODE]" in capsys.readouterr().out


@NEEDS_FLEET
@pytest.mark.parametrize("mode", ["1", "0"])
def test_residency_module_identity_follows_d32(mode, monkeypatch, capsys):
    """A module path mismatch follows D32 without a module replacement."""
    delivery = _load_delivery()
    delivery.require_container_residency()
    module = sys.modules["g3_residency"]
    monkeypatch.setattr(module, "__file__", "/different/g3_residency.py")
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    if mode == "0":
        with pytest.raises(ImportError, match="module path changed"):
            delivery.require_container_residency()
    else:
        delivery.require_container_residency()
        assert sys.modules["g3_residency"] is module
        assert "[DEV-MODE]" in capsys.readouterr().out


@NEEDS_FLEET
@pytest.mark.parametrize("mode", ["1", "0"])
def test_missing_external_dependency_refuses_in_both_modes(mode, monkeypatch, tmp_path):
    """Neither mode permits an absent owner."""
    delivery = _load_delivery()
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    with pytest.raises(FileNotFoundError, match="external owner is absent"):
        delivery.stage_tree(tmp_path / "stage", owners_dir=tmp_path)
    assert not (tmp_path / "stage").exists()


@NEEDS_FLEET
def test_container_client_selects_delivered_workspace(tmp_path):
    """The real client boundary supplies the delivered read-only workspace."""
    delivery = _load_delivery()
    entry = delivery.stage_tree(tmp_path / "delivered")
    output = tmp_path / "output"
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(entry.parent)!r})\n"
        "import campaign_launch as guard\n"
        f"root = Path({str(output)!r})\n"
        "container, command, client = guard.container_launch(\n"
        "    ['stage1.py', 'encode', '--device', 'cuda', '--root', str(root)], root)\n"
        f"assert client[1] == {str(entry.parent / 'v2_launch.py')!r}\n"
        "assert command[1] == '/workspace/stage1.py'\n"
        "probe = guard.start_container_client([sys.executable, '-c',\n"
        "    \"from pathlib import Path; print('workspace=' + str(Path.cwd()))\"])\n"
        "assert probe.wait(timeout=30) == 0\n"
        "sys.path.insert(0, '/mnt/shared/tessera-measurements/"
        "surrogate-diag-20260929/src/pq-7882eda3')\n"
        "from tools import tessera_campaign_container as adapter\n"
        "docker = adapter.docker_command(container, command,\n"
        "    cwd=str(guard.CHILD_SOURCE), uid=1000, gid=1000,\n"
        "    image_id='sha256:' + 'a' * 64, with_gpu=False)\n"
        f"assert 'type=bind,src={entry.parent},dst=/workspace,readonly' in docker\n"
        "assert not root.exists()\n"
        "print('container-workspace-ok')\n"
    )
    done = subprocess.run([CPU_PYTHON, "-c", script], cwd=tmp_path,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert f"workspace={entry.parent}" in done.stdout
    assert "container-workspace-ok" in done.stdout
