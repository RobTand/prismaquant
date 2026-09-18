"""CPU gates for the sealed-prepare run transition; no external checkpoint is touched."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from prismaquant import aura_cost as aura
from prismaquant import joint_aura_run_transition as transition
from prismaquant import joint_aura_transitions as transitions
from prismaquant import joint_aura_source_transition as resume_transition
from prismaquant.production_weight_cache import _production_cache_source_sha256
from test_joint_aura_streamed import _fixture, _run


def _write_json(path, value):
    path.write_bytes(transition._canonical(value) + b"\n")
    return {"path": str(path), "sha256": transition._sha(path)}


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    """A tiny sealed package: an old tree hashed into the contract, a fixed tree that reconstructs it."""
    new_git = "3" * 40
    package = tmp_path / "package"
    package.mkdir()
    (package / "aura_cost.py").write_bytes(b"new-math\nnew-glue\n")
    (package / "tessera_joint_aura.py").write_bytes(b"new-import\n")
    (package / "dependency.py").write_bytes(b"unchanged\n")
    (package / "joint_aura_run_transition.py").write_bytes(b"verifier\n")
    (package / "joint_aura_transitions.py").write_bytes(b"dispatcher\n")
    old = tmp_path / "original"
    old.mkdir()
    (old / "aura_cost.py").write_bytes(b"old-math\nold-glue\n")
    (old / "tessera_joint_aura.py").write_bytes(b"old-import\n")
    (old / "dependency.py").write_bytes(b"unchanged\n")
    old_source = _production_cache_source_sha256(old)
    for path in old.iterdir():
        path.unlink()
    old.rmdir()
    monkeypatch.setattr(transition, "_SOURCE_REWRITES", {
        "aura_cost.py": [("old-math", "new-math"), ("old-glue", "new-glue")],
        "tessera_joint_aura.py": [("old-import", "new-import")]})
    plan_config = {"output_root": str(tmp_path), "calibration": "fixed", "backend": "fixed"}
    plan = _write_json(tmp_path / "plan.json", plan_config)
    cache_file = tmp_path / "production.pkl"
    cache_file.write_bytes(b"prepared production cache fixture")
    prepared = _write_json(tmp_path / "prepared.json", {
        "schema": transition.PREPARED_SCHEMA, "status": "complete", "measured_cells": 6,
        "implementation_sha256": old_source, "plan_sha256": plan["sha256"],
        "production_cache": {"path": str(cache_file), "sha256": transition._sha(cache_file)}})
    identity = _write_json(tmp_path / "identity.json", {"schema": "campaign identity fixture"})
    contract = {"source_sha256": old_source, "git_commit": "1" * 40, "plan_sha256": plan["sha256"],
                "prepared_sha256": prepared["sha256"], "production_cache_sha256": transition._sha(cache_file),
                "campaign_identity_sha256": identity["sha256"], "measured_cells": 6}
    monkeypatch.setattr(transition, "_CONTRACT", contract)
    state = {"commit": new_git, "package": package}

    def actual_execution():
        return {"git_commit": state["commit"], **transition.source_proof(state["package"])}
    monkeypatch.setattr(transition, "_actual_execution", actual_execution)
    monkeypatch.setattr(transition, "_committed_package", lambda root: {"git_parent_commit": "5" * 40})
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: state["commit"])
    monkeypatch.setattr(aura, "_aura_source_sha256",
                        lambda: transition.source_proof(state["package"])["producer_source_sha256"])
    bindings = {"plan": plan, "prepared": prepared, "campaign_identity": identity}
    receipt = transition.create_transition(bindings=bindings, output=tmp_path / "transition.json")
    checkpoints = tmp_path / "checkpoints"

    def load(**overrides):
        arguments = {"config": plan_config, "plan_sha256": plan["sha256"], "prepared": prepared,
                     "checkpoint_dir": checkpoints, **overrides}
        return transitions.load_transition(receipt, **arguments)
    return {"receipt": receipt, "bindings": bindings, "load": load, "state": state, "config": plan_config,
            "checkpoints": checkpoints, "old_source": old_source, "package": package}


@pytest.mark.parametrize("mutation", ["other_file", "math", "glue", "import", "missing_module",
                                      "missing_dispatcher", "extra_file"])
def test_source_proof_rejects_every_change_outside_the_rewrites(sealed, mutation):
    package = sealed["package"]
    transition.source_proof(package)
    if mutation == "other_file":
        (package / "dependency.py").write_bytes(b"unchanged\n ")
    elif mutation == "math":
        (package / "aura_cost.py").write_bytes(b"wrong-math\nnew-glue\n")
    elif mutation == "glue":
        (package / "aura_cost.py").write_bytes(b"new-math\nwrong-glue\n")
    elif mutation == "import":
        (package / "tessera_joint_aura.py").write_bytes(b"wrong-import\n")
    elif mutation == "missing_module":
        (package / "joint_aura_run_transition.py").unlink()
    elif mutation == "missing_dispatcher":
        (package / "joint_aura_transitions.py").unlink()
    else:
        (package / "extra.py").write_bytes(b"new dependency")
    with pytest.raises(ValueError, match="source|package"):
        transition.source_proof(package)


@pytest.mark.skipif(os.environ.get("PRISMAQUANT_RUN_TRANSITION_REAL_PACKAGE") != "1",
                    reason="the closed contract binds one package's bytes (the campaign branch); "
                           "the campaign preflight runs this with the variable set")
def test_real_package_reconstructs_the_sealed_prepare_package():
    """The live rewrites reverse to the campaign's prepared implementation_sha256."""
    proof = transition.source_proof()
    assert proof["reconstructed_source_sha256"] == transition._CONTRACT["source_sha256"]
    assert proof["producer_source_sha256"] == _production_cache_source_sha256()
    assert proof["producer_source_sha256"] != proof["reconstructed_source_sha256"]


def test_contract_names_the_live_prepared_schema():
    from prismaquant.tessera_joint_aura import PREPARED_SCHEMA
    assert transition.PREPARED_SCHEMA == PREPARED_SCHEMA


def test_fresh_run_carries_the_receipt_and_rewrites_the_measurement_source(sealed):
    cap = sealed["load"]()
    assert isinstance(cap, transition.VerifiedRunTransition)
    assert cap.measurement_source_sha256 == sealed["old_source"]
    _, context, runner, cache = _fixture()
    result = _run(runner, cache, checkpoint_dir=sealed["checkpoints"], resume=True, source_transition=cap)
    manifest = json.loads((sealed["checkpoints"] / "manifest.json").read_bytes())
    assert manifest["identity"]["git_commit"] == "1" * 40
    assert manifest["identity"]["producer_source_sha256"] == sealed["old_source"]
    provenance = result["provenance"]["source_transition"]
    assert provenance["units"] == len(result["costs"])
    assert provenance["observed_git_commit"] == sealed["state"]["commit"]
    assert provenance["receipt"]["sha256"] == sealed["receipt"]["sha256"]
    assert result["provenance"]["joint_activation"] is True
    for name in result["costs"]:
        state = aura._load_aura_unit_checkpoint(aura._aura_unit_checkpoint_path(sealed["checkpoints"], name),
                                               qname=name, identity_sha256=manifest["identity_sha256"])
        assert state["execution_provenance"] == cap.execution_provenance
    # An interruption resumes under the same receipt without recomputation.
    cap2 = sealed["load"]()
    _, context, runner, cache = _fixture()
    again = _run(runner, cache, checkpoint_dir=sealed["checkpoints"], resume=True, source_transition=cap2)
    assert again["costs"] == result["costs"]
    assert context.install_calls == 0


def test_receipt_created_under_one_snapshot_commit_loads_under_another_with_identical_bytes(sealed):
    """PrismaBuild seals a distinct commit per submission; the binding is the package bytes."""
    sealed["state"]["commit"] = "6" * 40
    cap = sealed["load"]()
    assert json.loads(cap._receipt_bytes)["execution"]["git_commit"] == "3" * 40
    assert cap._observed_git_commit == "6" * 40
    _, _, runner, cache = _fixture()
    result = _run(runner, cache, checkpoint_dir=sealed["checkpoints"], resume=True, source_transition=cap)
    assert result["provenance"]["source_transition"]["observed_git_commit"] == "6" * 40


def test_a_checkpoint_manifest_from_another_source_is_refused(sealed):
    _, _, runner, cache = _fixture()
    _run(runner, cache, checkpoint_dir=sealed["checkpoints"])
    with pytest.raises(ValueError, match="another measurement source"):
        sealed["load"]()


@pytest.mark.parametrize("target", ["plan", "prepared", "campaign_identity", "cache", "receipt"])
def test_load_rejects_changed_bound_artifact(sealed, target):
    if target == "cache":
        path = Path(sealed["bindings"]["prepared"]["path"]).parent / "production.pkl"
    elif target == "receipt":
        path = Path(sealed["receipt"]["path"])
    else:
        path = Path(sealed["bindings"][target]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        sealed["load"]()


def test_load_rejects_a_package_whose_bytes_differ_from_the_receipt(sealed):
    (sealed["package"] / "dependency.py").write_bytes(b"unchanged\n ")
    with pytest.raises(ValueError, match="package"):
        sealed["load"]()


def test_load_rejects_a_resealed_receipt(sealed):
    path = Path(sealed["receipt"]["path"])
    record = json.loads(path.read_bytes())
    record["execution"]["producer_source_sha256"] = "9" * 64
    sealed["receipt"].update(_write_json(path, record))
    with pytest.raises(ValueError, match="execution source"):
        sealed["load"]()


def test_load_rejects_changed_runtime_plan_or_prepared_binding(sealed):
    with pytest.raises(ValueError, match="runtime plan"):
        sealed["load"](config={**sealed["config"], "backend": "different"})
    with pytest.raises(ValueError, match="prepared binding"):
        sealed["load"](prepared={**sealed["bindings"]["prepared"], "path": "/elsewhere/prepared.json"})


def test_transition_requires_joint_resume(sealed):
    cap = sealed["load"]()
    _, context, runner, cache = _fixture()
    with pytest.raises(ValueError, match="bound joint resume"):
        _run(runner, cache, checkpoint_dir=sealed["checkpoints"], resume=False, source_transition=cap)
    assert context.install_calls == 0


@pytest.mark.parametrize("kind", ["dict", "forged", "other_version"])
def test_caller_cannot_supply_identity_override(sealed, kind):
    cap = sealed["load"]()
    if kind == "dict":
        bad = cap.execution_provenance
    elif kind == "forged":
        bad = transition.VerifiedRunTransition(cap._receipt_bytes, cap._receipt_path, cap._receipt_sha256,
                                               cap._checkpoint_dir, cap._observed_git_commit)
    else:
        bad = resume_transition.VerifiedTransition(cap._receipt_bytes, cap._receipt_path, cap._receipt_sha256,
                                                   cap._checkpoint_dir, b"{}")
    _, context, runner, cache = _fixture()
    with pytest.raises(ValueError, match="verified receipt loader"):
        _run(runner, cache, checkpoint_dir=sealed["checkpoints"], resume=True, source_transition=bad)
    assert context.install_calls == 0


def test_source_changed_after_admission_is_refused(sealed):
    cap = sealed["load"]()
    (sealed["package"] / "dependency.py").write_bytes(b"unchanged\n ")
    _, context, runner, cache = _fixture()
    with pytest.raises(ValueError, match="package"):
        _run(runner, cache, checkpoint_dir=sealed["checkpoints"], resume=True, source_transition=cap)
    assert context.install_calls == 0


def test_receipt_cannot_be_overwritten(sealed):
    with pytest.raises(FileExistsError):
        transition.create_transition(bindings=sealed["bindings"], output=sealed["receipt"]["path"])


def test_dispatcher_routes_by_version_and_refuses_unknown_or_missing_receipts(sealed, tmp_path):
    assert transitions.receipt_version(sealed["receipt"]) == transition.VERSION
    with pytest.raises(ValueError, match="missing"):
        transitions.load_transition({"path": str(tmp_path / "absent.json"), "sha256": "0" * 64})
    path = Path(sealed["receipt"]["path"])
    record = json.loads(path.read_bytes())
    record["version"] = "not_a_version"
    bound = _write_json(path, record)
    with pytest.raises(ValueError, match="unknown transition version"):
        transitions.load_transition(bound)


def test_actual_execution_binds_the_checkpoint_commit_to_the_sealed_head(tmp_path, monkeypatch):
    """The override the launcher supplies must be the checkout's own HEAD, read as a file."""
    root = tmp_path / "repo"
    (root / "prismaquant").mkdir(parents=True)
    (root / "prismaquant" / "aura_cost.py").write_text("x = 1\n")
    _git(root, "init", "-q")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "seal")
    head = _git(root, "rev-parse", "HEAD")
    assert transition.checkout_head_commit(root) == head
    _git(root, "checkout", "-q", "--detach")
    assert transition.checkout_head_commit(root) == head
    monkeypatch.setattr(transition, "source_proof", lambda: {key: "7" * 64 for key in transition._BYTES})
    monkeypatch.setattr(transition, "__file__", str(root / "prismaquant" / "joint_aura_run_transition.py"))
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: head)
    assert transition._actual_execution()["git_commit"] == head
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "8" * 40)
    with pytest.raises(ValueError, match="contradicts the sealed checkout HEAD"):
        transition._actual_execution()


def test_execute_preflight_precedes_cuda_and_model_work():
    from prismaquant.tessera_joint_aura import execute
    with pytest.raises(ValueError, match="joint source transition"):
        execute("run", {"output_root": "/does/not/exist"}, plan_sha256="0" * 64,
                resume=True, source_transition={"path": "/does/not/exist", "sha256": "0" * 64})
