"""CPU gates for the retained-budget run transition; no external checkpoint is touched."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from prismaquant import aura_cost as aura
from prismaquant import joint_aura_retained_budget_transition as transition
from prismaquant import joint_aura_run_transition as run_transition
from prismaquant import joint_aura_source_transition as resume_transition
from prismaquant import joint_aura_transitions as transitions
from prismaquant.production_weight_cache import _production_cache_source_sha256
from test_joint_aura_streamed import _fixture, _run

#: The sealed plan's own caps, and the caps #745 derives from the campaign
#: roster. The two that refused the run are ``candidate_delta_bytes`` and
#: ``max_windows_per_layer``.
_SEALED_BUDGET = {
    "schema": transition.BUDGET_SCHEMA,
    "auxiliary_reserve_bytes": 2147483648, "boundary_reserve_bytes": 2281701376,
    "candidate_delta_bytes": 4194304, "load_buffer_bytes": 536870912,
    "max_windows_per_layer": 2, "metadata_reserve_bytes": 21474836480,
    "physical_limit_bytes": 111669149696, "read_page_reserve_bytes": 4096,
    "retained_render_cap_bytes": 2147483648, "runtime_reserve_bytes": 4294967296,
    "safety_margin_bytes": 2147483648, "statistics_cap_bytes": 34359738368,
    "workspace_reserve_bytes": 17179869184,
}
_CORRECTED = {"candidate_delta_bytes": 201326592, "max_windows_per_layer": 78,
              "load_buffer_bytes": 402662764}
_BUDGET_FIELDS = tuple(sorted(set(_SEALED_BUDGET) - {"schema"}))


def _plan(tmp_path, budget=None, **overrides):
    """A plan shaped like the campaign's: the transition reads structure, not physics."""
    plan = {
        "schema": "prismaquant.tessera_joint_aura.plan.v1",
        "output_root": str(tmp_path), "model": "/models/fixed", "max_gpu_bytes": 85899345920,
        "inputs": {"campaign_plan": {"path": "/mnt/shared/campaign", "sha256": "a" * 64}},
        "calibration_input": {"text_sha256": "b" * 64},
        "reader": None, "historical_encoder_reuse": None,
        "execution": {
            "n_probes": 3, "seed_base": 7000, "projection_backend": {"binary_sha256": "c" * 64},
            "operator_windows": {"max_candidate_bytes": 268435456, "prefetch_workers": 4},
            "boundary_storage": {"capture_order": "layer_major"},
            "retained_operator_windows": {
                "schema": "prismaquant.joint_retained_execution.v1",
                "budget": dict(_SEALED_BUDGET if budget is None else budget),
                "source_reserve_bytes": 33285996544,
                "source_loading_reserve_bytes": 536870912},
        },
    }
    plan.update(copy.deepcopy(overrides))
    return plan


def _write_json(path, value):
    path.write_bytes(transition._canonical(value) + b"\n")
    return {"path": str(path), "sha256": transition._sha(path)}


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    """A tiny sealed package, the plan its prepare was made against, and a corrected plan."""
    new_git = "3" * 40
    package = tmp_path / "package"
    package.mkdir()
    (package / "aura_cost.py").write_bytes(b"new-math\nnew-glue\n")
    (package / "tessera_joint_aura.py").write_bytes(b"new-import\n")
    (package / "dependency.py").write_bytes(b"unchanged\n")
    for name in sorted(transition._NEW_FILES):
        (package / name).write_bytes(f"{name}\n".encode())
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

    prepared_config = _plan(tmp_path)
    run_config = _plan(tmp_path, budget={**_SEALED_BUDGET, **_CORRECTED})
    prepared_plan = _write_json(tmp_path / "prepared-plan.json", prepared_config)
    run_plan = _write_json(tmp_path / "run-plan.json", run_config)
    cache_file = tmp_path / "production.pkl"
    cache_file.write_bytes(b"prepared production cache fixture")
    prepared = _write_json(tmp_path / "prepared.json", {
        "schema": transition.PREPARED_SCHEMA, "status": "complete", "measured_cells": 6,
        "implementation_sha256": old_source, "plan_sha256": prepared_plan["sha256"],
        "production_cache": {"path": str(cache_file), "sha256": transition._sha(cache_file)}})
    identity = _write_json(tmp_path / "identity.json", {"schema": "campaign identity fixture"})
    contract = {**transition._CONTRACT, "source_sha256": old_source, "git_commit": "1" * 40,
                "prepared_plan_sha256": prepared_plan["sha256"], "prepared_sha256": prepared["sha256"],
                "production_cache_sha256": transition._sha(cache_file),
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
    checkpoints = tmp_path / "checkpoints"

    def bind(config, name):
        return _write_json(tmp_path / name, config)

    def seal(config=None, output="transition.json"):
        """Write a run plan, bind it and create a receipt over it."""
        bound = run_plan if config is None else bind(config, "candidate-plan.json")
        bindings = {"prepared_plan": prepared_plan, "run_plan": bound,
                    "prepared": prepared, "campaign_identity": identity}
        receipt = transition.create_transition(bindings=bindings, output=tmp_path / output)
        return receipt, bindings, bound

    receipt, bindings, _ = seal()

    def load(**overrides):
        arguments = {"config": run_config, "plan_sha256": run_plan["sha256"], "prepared": prepared,
                     "checkpoint_dir": checkpoints, **overrides}
        return transitions.load_transition(receipt, **arguments)
    return {"receipt": receipt, "bindings": bindings, "load": load, "state": state,
            "prepared_config": prepared_config, "run_config": run_config, "checkpoints": checkpoints,
            "old_source": old_source, "package": package, "seal": seal, "tmp_path": tmp_path,
            "prepared_plan": prepared_plan, "run_plan": run_plan, "prepared": prepared}


# ---------------------------------------------------------------- the key set

def test_the_enumerated_key_set_is_the_whole_retained_budget_and_nothing_else():
    """The contract's literal paths, checked against the budget the plan loader requires."""
    from prismaquant.joint_retained_window_plan import RetainedWindowBudget
    budget_paths = {tuple(path) for path in transition._CONTRACT["admitted_budget_keys"]}
    assert budget_paths == {("execution", "retained_operator_windows", "budget", name)
                            for name in RetainedWindowBudget.__dataclass_fields__}
    assert ("execution", "retained_operator_windows", "budget", "schema") not in budget_paths
    assert [tuple(path) for path in transition._CONTRACT["admitted_record_keys"]] == [
        ("retained_window_budget_derivation",)]
    assert set(_SEALED_BUDGET) == set(RetainedWindowBudget.__dataclass_fields__) | {"schema"}


@pytest.mark.parametrize("field", _BUDGET_FIELDS)
def test_every_enumerated_budget_key_is_admitted_alone(sealed, field):
    changed = _plan(sealed["tmp_path"], budget={**_SEALED_BUDGET, field: _SEALED_BUDGET[field] + 1})
    difference = transition.plan_difference(sealed["prepared_config"], changed)
    assert difference == [{"path": ["execution", "retained_operator_windows", "budget", field],
                           "prepared_plan": {"present": True, "value": _SEALED_BUDGET[field]},
                           "run_plan": {"present": True, "value": _SEALED_BUDGET[field] + 1}}]


def test_the_derivation_record_rides_and_is_recorded_without_being_read(sealed):
    record = {"schema": "prismaquant.joint_retained_window_budget_derivation.v1",
              "superseded_budget": dict(_SEALED_BUDGET), "windows_by_layer": {"44": 78}}
    changed = _plan(sealed["tmp_path"], budget={**_SEALED_BUDGET, **_CORRECTED},
                    retained_window_budget_derivation=record)
    difference = transition.plan_difference(sealed["prepared_config"], changed)
    entry = [row for row in difference if row["path"] == ["retained_window_budget_derivation"]]
    digest = hashlib.sha256(transition._canonical(record)).hexdigest()
    assert entry == [{"path": ["retained_window_budget_derivation"],
                      "prepared_plan": {"present": False},
                      "run_plan": {"present": True, "canonical_sha256": digest}}]
    # Recorded, never interpreted: a record whose content differs is a
    # different record, and nothing here reads what is inside it.
    changed["retained_window_budget_derivation"] = {**record, "windows_by_layer": {"44": 7}}
    other = transition.plan_difference(sealed["prepared_config"], changed)
    assert [row for row in other if row["path"] == ["retained_window_budget_derivation"]] != entry


def test_the_corrected_campaign_caps_are_admitted_and_reported(sealed):
    difference = transition.plan_difference(sealed["prepared_config"], sealed["run_config"])
    assert {row["path"][-1] for row in difference} == set(_CORRECTED)
    assert {row["path"][-1]: row["run_plan"]["value"] for row in difference} == _CORRECTED


# ------------------------------------------------- everything else is refused

@pytest.mark.parametrize("mutation", [
    "top_level_value", "top_level_added", "top_level_removed", "nested_execution_value",
    "sibling_of_the_budget", "budget_schema", "budget_key_removed", "retained_schema",
    "deep_input_value", "list_member",
])
def test_a_difference_outside_the_enumerated_keys_is_refused(sealed, mutation):
    changed = _plan(sealed["tmp_path"], budget={**_SEALED_BUDGET, **_CORRECTED})
    if mutation == "top_level_value":
        changed["max_gpu_bytes"] = 56 * 1024 ** 3
    elif mutation == "top_level_added":
        changed["aggregate_memory_bytes"] = 1
    elif mutation == "top_level_removed":
        del changed["reader"]
    elif mutation == "nested_execution_value":
        changed["execution"]["seed_base"] = 7001
    elif mutation == "sibling_of_the_budget":
        changed["execution"]["retained_operator_windows"]["source_reserve_bytes"] += 1
    elif mutation == "budget_schema":
        changed["execution"]["retained_operator_windows"]["budget"]["schema"] = "other.v2"
    elif mutation == "budget_key_removed":
        del changed["execution"]["retained_operator_windows"]["budget"]["read_page_reserve_bytes"]
    elif mutation == "retained_schema":
        changed["execution"]["retained_operator_windows"]["schema"] = "other.v2"
    elif mutation == "deep_input_value":
        changed["inputs"]["campaign_plan"]["sha256"] = "d" * 64
    else:
        changed["execution"]["operator_windows"]["prefetch_workers"] = [4]
    with pytest.raises(ValueError, match="outside the admitted retained-budget keys|exact integer"):
        transition.plan_difference(sealed["prepared_config"], changed)


@pytest.mark.parametrize("value", [4194304.0, "4194304", True, None])
def test_a_budget_cap_that_is_not_an_exact_integer_is_refused(sealed, value):
    changed = _plan(sealed["tmp_path"], budget={**_SEALED_BUDGET, "candidate_delta_bytes": value})
    with pytest.raises(ValueError, match="exact integer"):
        transition.plan_difference(sealed["prepared_config"], changed)


def test_dropping_the_retained_block_from_one_plan_is_a_residue_change(sealed):
    """The block's own schema and reserves are not admitted keys, so they refuse first."""
    changed = copy.deepcopy(sealed["prepared_config"])
    del changed["execution"]["retained_operator_windows"]
    with pytest.raises(ValueError, match="outside the admitted retained-budget keys"):
        transition.plan_difference(sealed["prepared_config"], changed)


def test_two_plans_that_both_declare_no_retained_budget_are_refused(sealed):
    """A residue can be identical and still not be a plan this transition admits."""
    bare = copy.deepcopy(sealed["prepared_config"])
    del bare["execution"]["retained_operator_windows"]
    with pytest.raises(ValueError, match="exact integer"):
        transition.plan_difference(bare, copy.deepcopy(bare))


def test_a_retained_block_whose_budget_is_not_a_mapping_is_refused(sealed):
    changed = copy.deepcopy(sealed["prepared_config"])
    changed["execution"]["retained_operator_windows"]["budget"] = "sealed"
    with pytest.raises(ValueError, match="does not run through mappings"):
        transition.plan_difference(sealed["prepared_config"], changed)


def test_an_identical_plan_is_admitted_with_an_empty_difference(sealed):
    assert transition.plan_difference(sealed["prepared_config"], sealed["prepared_config"]) == []


def test_reformatting_alone_is_the_same_plan(sealed):
    """The pass consumes the parsed plan, and both files are pinned by SHA256 besides."""
    reordered = json.loads(json.dumps(sealed["run_config"], sort_keys=True, indent=3))
    assert transition.plan_difference(sealed["prepared_config"], reordered) == \
        transition.plan_difference(sealed["prepared_config"], sealed["run_config"])


# ------------------------------------------------------------ RED, end to end

def test_a_run_plan_changed_outside_the_budget_keys_cannot_be_sealed(sealed):
    changed = _plan(sealed["tmp_path"], budget={**_SEALED_BUDGET, **_CORRECTED})
    changed["execution"]["seed_base"] = 7001
    with pytest.raises(ValueError, match="outside the admitted retained-budget keys"):
        sealed["seal"](changed, output="refused.json")


def test_a_budget_only_run_plan_is_admitted_and_the_receipt_carries_both_plans(sealed):
    cap = sealed["load"]()
    assert isinstance(cap, transition.VerifiedRetainedBudgetTransition)
    receipt = json.loads(Path(sealed["receipt"]["path"]).read_bytes())
    assert receipt["inputs"]["prepared_plan"]["sha256"] == sealed["prepared_plan"]["sha256"]
    assert receipt["inputs"]["run_plan"]["sha256"] == sealed["run_plan"]["sha256"]
    assert receipt["inputs"]["run_plan"]["sha256"] != receipt["inputs"]["prepared_plan"]["sha256"]
    assert {row["path"][-1]: [row["prepared_plan"]["value"], row["run_plan"]["value"]]
            for row in receipt["plan_difference"]} == {
        name: [_SEALED_BUDGET[name], value] for name, value in _CORRECTED.items()}
    provenance = cap.execution_provenance
    assert provenance["prepared_plan_sha256"] == sealed["prepared_plan"]["sha256"]
    assert provenance["run_plan_sha256"] == sealed["run_plan"]["sha256"]
    assert provenance["plan_difference"] == receipt["plan_difference"]
    assert cap.prepared_plan_sha256 == sealed["prepared_plan"]["sha256"]
    assert cap.measurement_source_sha256 == sealed["old_source"]


def test_a_receipt_whose_recorded_difference_was_edited_is_refused(sealed):
    path = Path(sealed["receipt"]["path"])
    record = json.loads(path.read_bytes())
    record["plan_difference"] = []
    sealed["receipt"].update(_write_json(path, record))
    with pytest.raises(ValueError, match="recorded plan difference"):
        sealed["load"]()


def test_a_run_plan_swapped_after_the_receipt_is_refused(sealed):
    other = _plan(sealed["tmp_path"], budget={**_SEALED_BUDGET, "candidate_delta_bytes": 33554432})
    Path(sealed["run_plan"]["path"]).write_bytes(transition._canonical(other) + b"\n")
    with pytest.raises(ValueError, match="run plan bytes changed"):
        sealed["load"]()


def test_the_running_plan_must_be_the_sealed_run_plan(sealed):
    with pytest.raises(ValueError, match="runtime plan"):
        sealed["load"](config=sealed["prepared_config"])
    with pytest.raises(ValueError, match="runtime plan"):
        sealed["load"](plan_sha256=sealed["prepared_plan"]["sha256"])


# --------------------------------------------------- the sibling stays sealed

def test_the_prepared_record_keeps_naming_the_plan_the_prepare_was_made_against(sealed):
    """The one prepared field a budget-only plan change reaches, and the seam that answers it."""
    from prismaquant.tessera_joint_aura import _preflight_run_prepared
    cap = sealed["load"]()
    arguments = {"implementation_sha256": sealed["old_source"], "reader_identity": None,
                 "projection_backend": None}
    prepared = dict(sealed["prepared"])
    with pytest.raises(ValueError, match="prepared plan_sha256"):
        _preflight_run_prepared(prepared, plan_sha256=sealed["run_plan"]["sha256"], **arguments)
    completion = _preflight_run_prepared(
        prepared, plan_sha256=transitions.transition_prepared_plan_sha256(
            cap, plan_sha256=sealed["run_plan"]["sha256"]), **arguments)
    assert completion["plan_sha256"] == sealed["prepared_plan"]["sha256"]


def test_every_other_transition_binds_the_running_plan(sealed):
    cap = sealed["load"]()
    assert transitions.transition_prepared_plan_sha256(cap, plan_sha256="z" * 64) == \
        sealed["prepared_plan"]["sha256"]
    for verified_type, _ in transitions._VERIFIED:
        assert verified_type in transitions._PREPARED_PLAN
    other = run_transition.VerifiedRunTransition(b"{}", "p", "s", "d", "c")
    assert transitions.transition_prepared_plan_sha256(other, plan_sha256="z" * 64) == "z" * 64
    resume = resume_transition.VerifiedTransition(b"{}", "p", "s", "d", b"{}")
    assert transitions.transition_prepared_plan_sha256(resume, plan_sha256="z" * 64) == "z" * 64
    with pytest.raises(ValueError, match="verified receipt loader"):
        transitions.transition_prepared_plan_sha256(object(), plan_sha256="z" * 64)


def test_the_sibling_transition_refuses_the_corrected_plan(sealed, tmp_path, monkeypatch):
    """RED: what this version exists for. v1 pins one plan digest, and it is the old one."""
    monkeypatch.setattr(run_transition, "_SOURCE_REWRITES", transition._SOURCE_REWRITES)
    monkeypatch.setattr(run_transition, "_CONTRACT", {
        "source_sha256": sealed["old_source"], "git_commit": "1" * 40,
        "plan_sha256": sealed["prepared_plan"]["sha256"],
        "prepared_sha256": sealed["prepared"]["sha256"],
        "production_cache_sha256": transition._CONTRACT["production_cache_sha256"],
        "campaign_identity_sha256": transition._CONTRACT["campaign_identity_sha256"],
        "measured_cells": 6})
    corrected = {"plan": sealed["run_plan"], "prepared": sealed["prepared"],
                 "campaign_identity": sealed["bindings"]["campaign_identity"]}
    with pytest.raises(ValueError, match="unapproved plan"):
        run_transition._load_inputs(corrected)
    # The same corrected plan, through this version: admitted, and the difference said.
    assert transition.plan_difference(sealed["prepared_config"], sealed["run_config"])


def test_the_sibling_contract_is_untouched_and_still_binds_one_plan():
    assert run_transition.VERSION == "meta_skeleton_render_proof_v1"
    assert set(run_transition._CONTRACT) == {
        "source_sha256", "git_commit", "plan_sha256", "prepared_sha256",
        "production_cache_sha256", "campaign_identity_sha256", "measured_cells"}
    assert run_transition._NEW_FILES == frozenset({
        "joint_aura_run_transition.py", "joint_aura_transitions.py"})
    assert transition.VERSION != run_transition.VERSION
    assert transition.SCHEMA != run_transition.SCHEMA
    assert transition._CONTRACT["source_sha256"] == run_transition._CONTRACT["source_sha256"]
    assert transition._CONTRACT["prepared_plan_sha256"] == run_transition._CONTRACT["plan_sha256"]
    # Its own literals, not a reference: an edit to the sibling's table cannot
    # move this proof.
    assert transition._SOURCE_REWRITES is not run_transition._SOURCE_REWRITES


# ------------------------------------------------- the package and the wiring

@pytest.mark.parametrize("mutation", ["other_file", "math", "glue", "import", "missing_module",
                                      "missing_sibling", "missing_dispatcher", "extra_file"])
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
        (package / "joint_aura_retained_budget_transition.py").unlink()
    elif mutation == "missing_sibling":
        (package / "joint_aura_run_transition.py").unlink()
    elif mutation == "missing_dispatcher":
        (package / "joint_aura_transitions.py").unlink()
    else:
        (package / "extra.py").write_bytes(b"new dependency")
    with pytest.raises(ValueError, match="source|package"):
        transition.source_proof(package)


def test_the_rewrite_table_carries_the_prepared_plan_substitution_and_matches_the_package():
    """The wiring is the table: source_proof requires each hunk present exactly once."""
    hunks = transition._SOURCE_REWRITES["tessera_joint_aura.py"]
    wiring = [new for _, new in hunks if "prepared_plan_sha256" in new]
    assert len(wiring) == 3
    assert any("plan_sha256=prepared_plan_sha256" in new for new in wiring)
    assert any('("plan_sha256", prepared_plan_sha256)' in new for new in wiring)
    assert any("transition_prepared_plan_sha256" in new for new in wiring)
    source = (Path(transition.__file__).resolve().parent / "tessera_joint_aura.py").read_text()
    for _, new in hunks:
        assert source.count(new) == 1


@pytest.mark.skipif(os.environ.get("PRISMAQUANT_BUDGET_TRANSITION_REAL_PACKAGE") != "1",
                    reason="the closed contract binds one package's bytes (the campaign branch); "
                           "the campaign preflight runs this with the variable set")
def test_real_package_reconstructs_the_sealed_prepare_package():
    """The live rewrites reverse to the campaign's prepared implementation_sha256."""
    proof = transition.source_proof()
    assert proof["reconstructed_source_sha256"] == transition._CONTRACT["source_sha256"]
    assert proof["producer_source_sha256"] == _production_cache_source_sha256()
    assert proof["producer_source_sha256"] != proof["reconstructed_source_sha256"]


def test_contract_names_the_live_prepared_and_budget_schemas():
    from prismaquant.joint_retained_window_plan import SCHEMA as BUDGET_SCHEMA
    from prismaquant.tessera_joint_aura import PREPARED_SCHEMA
    assert transition.PREPARED_SCHEMA == PREPARED_SCHEMA
    assert transition.BUDGET_SCHEMA == BUDGET_SCHEMA


# ------------------------------------------------------- the admitted run

def test_a_fresh_run_carries_the_receipt_and_rewrites_the_measurement_source(sealed):
    cap = sealed["load"]()
    _, context, runner, cache = _fixture()
    result = _run(runner, cache, checkpoint_dir=sealed["checkpoints"], resume=True, source_transition=cap)
    manifest = json.loads((sealed["checkpoints"] / "manifest.json").read_bytes())
    assert manifest["identity"]["git_commit"] == "1" * 40
    assert manifest["identity"]["producer_source_sha256"] == sealed["old_source"]
    provenance = result["provenance"]["source_transition"]
    assert provenance["units"] == len(result["costs"])
    assert provenance["run_plan_sha256"] == sealed["run_plan"]["sha256"]
    assert provenance["prepared_plan_sha256"] == sealed["prepared_plan"]["sha256"]
    assert provenance["observed_git_commit"] == sealed["state"]["commit"]
    for name in result["costs"]:
        state = aura._load_aura_unit_checkpoint(aura._aura_unit_checkpoint_path(sealed["checkpoints"], name),
                                                qname=name, identity_sha256=manifest["identity_sha256"])
        assert state["execution_provenance"] == cap.execution_provenance


@pytest.mark.parametrize("target", ["prepared_plan", "run_plan", "prepared", "campaign_identity",
                                    "cache", "receipt"])
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


def test_transition_requires_joint_resume(sealed):
    cap = sealed["load"]()
    _, context, runner, cache = _fixture()
    with pytest.raises(ValueError, match="bound joint resume"):
        _run(runner, cache, checkpoint_dir=sealed["checkpoints"], resume=False, source_transition=cap)
    assert context.install_calls == 0


@pytest.mark.parametrize("kind", ["dict", "forged", "sibling_version"])
def test_caller_cannot_supply_identity_override(sealed, kind):
    cap = sealed["load"]()
    if kind == "dict":
        bad = cap.execution_provenance
    elif kind == "forged":
        bad = transition.VerifiedRetainedBudgetTransition(
            cap._receipt_bytes, cap._receipt_path, cap._receipt_sha256,
            cap._checkpoint_dir, cap._observed_git_commit)
    else:
        bad = run_transition.VerifiedRunTransition(
            cap._receipt_bytes, cap._receipt_path, cap._receipt_sha256,
            cap._checkpoint_dir, cap._observed_git_commit)
    _, context, runner, cache = _fixture()
    with pytest.raises(ValueError, match="verified receipt loader"):
        _run(runner, cache, checkpoint_dir=sealed["checkpoints"], resume=True, source_transition=bad)
    assert context.install_calls == 0


def test_receipt_cannot_be_overwritten(sealed):
    with pytest.raises(FileExistsError):
        transition.create_transition(bindings=sealed["bindings"], output=sealed["receipt"]["path"])


def test_dispatcher_routes_by_version(sealed, tmp_path):
    assert transitions.receipt_version(sealed["receipt"]) == transition.VERSION
    assert transitions._LOADERS[transition.VERSION] is transition
    path = Path(sealed["receipt"]["path"])
    record = json.loads(path.read_bytes())
    record["version"] = "not_a_version"
    bound = _write_json(path, record)
    with pytest.raises(ValueError, match="unknown transition version"):
        transitions.load_transition(bound)
