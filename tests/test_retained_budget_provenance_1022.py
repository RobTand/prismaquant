"""Stage B metadata refuses a retained budget not derived from its roster (PQ #1022).

The base GLM plan sealed an operator-declared retained budget whose 4 MiB
candidate delta no matrix in the roster fits, so every Stage B quantum built
on it would have refused at its preflight. The resource policy
(``joint_stageb_resources.derive_policy``) derives that budget from the
roster. Two producer-side guards:

* ``tools/prepare_extended_joint_quanta.py`` refuses an extended plan that
  binds no resource policy, or seals a budget other than the policy's.
* ``tools/regenerate_joint_quanta.py --executable-readsets`` settles every
  layer's retained admission before the head intake; a declared budget the
  roster does not fit refuses there, naming the plan digest and the fix.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools", ROOT / "tests"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from test_joint_stageb_resources import resource_fixture  # noqa: E402,F401
from test_strict_reader_tier_enforcement import _forget_state  # noqa: E402,F401


# -- prepare_extended_joint_quanta ------------------------------------------

def test_prepare_accepts_the_derived_overlay(resource_fixture):
    from tools.prepare_extended_joint_quanta import require_derived_budget
    _, _, extended, _, policy = resource_fixture
    assert require_derived_budget(extended, plan_sha256="e" * 64) == policy


def test_prepare_refuses_a_plan_without_a_resource_policy(resource_fixture):
    from tools.prepare_extended_joint_quanta import require_derived_budget
    _, original, _, _, _ = resource_fixture
    with pytest.raises(ValueError) as refused:
        require_derived_budget(original, plan_sha256="b" * 64)
    message = str(refused.value)
    assert "b" * 64 in message
    assert "prismaquant.joint_stageb_resources.derive_policy" in message


def test_prepare_refuses_a_budget_other_than_the_policys(resource_fixture):
    from tools.prepare_extended_joint_quanta import require_derived_budget
    _, _, extended, binding, _ = resource_fixture
    forged = copy.deepcopy(extended)
    forged["execution"]["retained_operator_windows"]["budget"][
        "candidate_delta_bytes"] += 1
    with pytest.raises(ValueError, match="different retained budget"):
        require_derived_budget(forged, plan_sha256="c" * 64)


def test_dev_mode_prints_a_budget_other_than_the_policys_and_continues(
        resource_fixture, monkeypatch, capsys):
    """PQ #1147: with PRISMAQUANT_DEV_MODE unset the budget seal prints both
    budgets and returns the policy instead of refusing."""
    from tools.prepare_extended_joint_quanta import require_derived_budget
    _, _, extended, _, policy = resource_fixture
    forged = copy.deepcopy(extended)
    forged["execution"]["retained_operator_windows"]["budget"][
        "candidate_delta_bytes"] += 1
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE")
    assert require_derived_budget(forged, plan_sha256="c" * 64) == policy
    out = capsys.readouterr().out
    assert "[DEV-MODE] seal retained budget differs at candidate_delta_bytes" in out


def test_prepare_refuses_a_plan_without_retained_windows(resource_fixture):
    from tools.prepare_extended_joint_quanta import require_derived_budget
    _, _, extended, _, _ = resource_fixture
    bare = copy.deepcopy(extended)
    del bare["execution"]["retained_operator_windows"]
    with pytest.raises(ValueError, match="no retained operator windows"):
        require_derived_budget(bare, plan_sha256="d" * 64)


def test_prepare_refusal_is_a_refusal_not_a_traceback(tmp_path, resource_fixture, monkeypatch, capsys):
    """The CLI exits 3 with the named refusal; it used to raise KeyError."""
    import tools.prepare_extended_joint_quanta as pe
    _, original, _, _, _ = resource_fixture
    plan_path = tmp_path / "declared-plan.json"
    plan_path.write_text(json.dumps(original))
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    prepared_path = tmp_path / "declared-prepared.json"
    prepared_path.write_text("{}")
    inputs = {"extended_plan": {"path": str(plan_path), "sha256": plan_sha},
              "extended_prepared": {"path": str(prepared_path),
                                    "sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest()}}
    pair = tmp_path / "pair.json"
    pair.write_text(json.dumps(inputs))
    stub = tmp_path / "stub.json"
    stub.write_text("{}")
    stub_sha = hashlib.sha256(stub.read_bytes()).hexdigest()
    band = tmp_path / "band.json"
    band.write_text("{}")
    argv = ["--pair-inputs", str(pair), "--pair-inputs-sha256",
            hashlib.sha256(pair.read_bytes()).hexdigest(),
            "--parent-manifest", str(stub), "--parent-manifest-sha256", stub_sha,
            "--derivation", str(stub), "--derivation-sha256", stub_sha,
            "--spec", str(stub), "--spec-sha256", stub_sha,
            "--adjoint-band", str(band), "--adjoint-band-sha256",
            hashlib.sha256(band.read_bytes()).hexdigest(),
            "--metadata-root", str(tmp_path / "meta")]
    import prismaquant.joint_adjoint_slices as slices
    monkeypatch.setattr(slices, "load_stage_a_receipt_like",
                        lambda path, digest=None: {"status": "band", "band": {"boundary": 45}})
    with pytest.raises(SystemExit) as exited:
        pe.main(argv)
    assert exited.value.code == 3
    assert plan_sha in capsys.readouterr().err
    assert not (tmp_path / "meta").exists()


# -- regenerate_joint_quanta ------------------------------------------------

def _rebind_receipt(layout, plan_sha):
    """The Stage A receipt answers for one plan; bind it to the edited one."""
    receipt = Path(layout["receipt_path"])
    receipt.write_text(receipt.read_text().replace(
        layout["campaign"]["plan_sha256"], plan_sha))


def _declared_plan(tmp_path, *, candidate_delta_bytes):
    from test_stageb_prepared_inputs_bridge import _campaign_files
    layout = _campaign_files(tmp_path)
    plan = copy.deepcopy(layout["plan"])
    plan["execution"]["retained_operator_windows"]["budget"][
        "candidate_delta_bytes"] = candidate_delta_bytes
    plan_path = tmp_path / "plan-declared.json"
    plan_path.write_text(json.dumps(plan))
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    _rebind_receipt(layout, plan_sha)
    return layout, plan_path, plan_sha


def _regen(layout, plan_path, plan_sha, records_out, run_root, *extra):
    import regenerate_joint_quanta as regen
    return regen.main([
        "--plan", str(plan_path), "--plan-sha256", plan_sha,
        "--prepared", str(layout["prepared_path"]),
        "--prepared-sha256", layout["campaign"]["prepared_sha256"],
        "--parent-manifest", str(layout["parent_path"]),
        "--parent-manifest-sha256", layout["parent_sha"],
        "--derivation", str(layout["derivation_path"]),
        "--partition", str(layout["partition_path"]),
        "--records-out", str(records_out),
        "--output-root", str(run_root),
        "--adjoint-receipt", str(layout["receipt_path"]),
        "--executable-readsets", *extra])


def test_regen_refuses_an_undelivered_declared_budget_before_the_head_intake(
        tmp_path, capsys, monkeypatch):
    import regenerate_joint_quanta as regen
    from test_stageb_prepared_inputs_bridge import SHAPE
    # Every fixture matrix needs a 4 * rows * cols fp32 delta; declare half.
    delta = 4 * SHAPE[0] * SHAPE[1]
    layout, plan_path, plan_sha = _declared_plan(
        tmp_path, candidate_delta_bytes=delta // 2)

    def no_head_intake(*args, **kwargs):
        raise AssertionError("the head intake ran before admission settled")
    monkeypatch.setattr(regen, "_build_head_slices", no_head_intake)
    records_out = tmp_path / "regen-declared" / "records"
    code = _regen(layout, plan_path, plan_sha, records_out,
                  tmp_path / "run-declared", "--head-slices")
    assert code == 3
    err = capsys.readouterr().err
    assert plan_sha in err
    assert "operator-declared retained budget" in err
    assert "indivisible target" in err
    assert "prismaquant.joint_stageb_resources.derive_policy" in err
    assert list(records_out.glob("layer-*.json")) == []


def test_regen_accepts_a_declared_budget_its_roster_fits(tmp_path):
    """A declared budget is refused only when it does not admit the roster."""
    from test_stageb_prepared_inputs_bridge import SHAPE
    delta = 4 * SHAPE[0] * SHAPE[1]
    layout, plan_path, plan_sha = _declared_plan(
        tmp_path, candidate_delta_bytes=delta)
    records_out = tmp_path / "regen-fits" / "records"
    assert _regen(layout, plan_path, plan_sha, records_out,
                  tmp_path / "run-fits") == 0
    assert len(list(records_out.glob("layer-*.json"))) > 0


def test_regen_refuses_a_plan_whose_budget_is_not_its_policys(
        tmp_path, capsys, monkeypatch):
    import regenerate_joint_quanta as regen
    import prismaquant.joint_stageb_resources as resources
    from test_stageb_prepared_inputs_bridge import _campaign_files
    layout = _campaign_files(tmp_path)
    plan = copy.deepcopy(layout["plan"])
    derived = copy.deepcopy(plan["execution"]["retained_operator_windows"]["budget"])
    derived["candidate_delta_bytes"] += 1
    plan["stage_b_resource_policy"] = {"path": "/policy.json", "sha256": "f" * 64}
    plan_path = tmp_path / "plan-policy.json"
    plan_path.write_text(json.dumps(plan))
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    _rebind_receipt(layout, plan_sha)
    monkeypatch.setattr(resources, "verify_policy",
                        lambda binding: {"budget": derived})
    records_out = tmp_path / "regen-policy" / "records"
    assert _regen(layout, plan_path, plan_sha, records_out,
                  tmp_path / "run-policy") == 3
    assert "different retained budget" in capsys.readouterr().err
    assert list(records_out.glob("layer-*.json")) == []
