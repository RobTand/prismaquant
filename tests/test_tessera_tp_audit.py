"""``prismaquant.tessera_tp_audit``: audit an assignment, never edit it.

The audit is the Step 5b check the GLM-5.3 TP2 export needs before Step 6:
given an assignment the allocator already produced, does the pinned runtime's
loader accept the cut the target profile says tensor parallelism makes on each
Tessera unit? It refuses and reports; the answer to a refusal is a re-run of
the allocation with the offending rung excluded by that measured fact, never a
hand edit of the assignment.
"""
from __future__ import annotations

import hashlib
import json
from importlib.resources import as_file
from pathlib import Path

import pytest

import prismaquant.tessera_runtime_contract as trc
from prismaquant import tessera_tp_audit as audit

FIXTURE = str(Path(__file__).parent / "fixtures"
              / "tessera_tp_audit_layer_config.json")
PROFILE = "tessera_research_sm121"

#: The one column-parallel K2 Linear in the fixture: a column-parallel cut
#: splits the output features, which are this unit's rows, and the contract
#: refuses the K2 loader a row shard on every rank.
REFUSED_UNIT = "model.layers.0.self_attn.q_proj"


@pytest.fixture
def contract_file(tmp_path):
    """The installed contract, as a file the CLI can be pointed at."""
    with as_file(trc.contract_path()) as path:
        copy = tmp_path / "runtime_contract.json"
        copy.write_bytes(Path(path).read_bytes())
        return str(copy)


def _run(tmp_path, contract_file, *, tp, out="receipt.json"):
    target = str(tmp_path / out)
    code = audit.main([
        "--layer-config", FIXTURE, "--target-profile", PROFILE,
        "--tp", str(tp), "--contract", contract_file, "--out", target,
    ])
    return code, json.loads(Path(target).read_text(encoding="utf-8"))


def test_a_whole_unit_world_has_nothing_to_refuse(tmp_path, contract_file):
    """At tp=1 no axis is cut, so every unit passes and the CLI exits 0."""
    code, receipt = _run(tmp_path, contract_file, tp=1)
    assert code == audit.EXIT_OK
    assert receipt["verdict"] == "pass"
    assert receipt["summary"]["refusals"] == 0
    assert {unit["cut_axis"] for unit in receipt["units"]} == {None}


def test_the_column_parallel_k2_linear_is_refused_at_tp2(tmp_path, contract_file):
    """One refusal, named, and a non-zero exit."""
    code, receipt = _run(tmp_path, contract_file, tp=2)
    assert code == audit.EXIT_REFUSED
    assert receipt["verdict"] == "refused"

    refused = [unit for unit in receipt["units"] if unit["verdict"] == "refused"]
    assert [unit["unit"] for unit in refused] == [REFUSED_UNIT]
    assert refused[0]["reason"] == (
        f"tp_axis_refused:TESSERA_E2M1_K2:{REFUSED_UNIT}:row")
    assert refused[0]["cut_kind"] == "column"
    assert refused[0]["axis_status"] == "refused"
    assert receipt["summary"]["refusals_by_reason"] == {"tp_axis_refused": 1}


def test_the_other_cut_direction_and_the_other_families_pass(tmp_path,
                                                             contract_file):
    """The refusal is per axis and per family, not per family alone."""
    _, receipt = _run(tmp_path, contract_file, tp=2)
    by_unit = {unit["unit"]: unit for unit in receipt["units"]}

    row_parallel_k2 = by_unit["model.layers.0.self_attn.o_proj"]
    assert row_parallel_k2["cut_axis"] == "column"
    assert row_parallel_k2["axis_status"] == "sharded"
    assert row_parallel_k2["verdict"] == "pass"

    column_parallel_fp8 = by_unit["model.layers.0.mlp.gate_up_proj"]
    assert column_parallel_fp8["cut_axis"] == "row"
    assert column_parallel_fp8["verdict"] == "pass"


def test_a_packed_expert_resolves_to_no_cut_at_all(tmp_path, contract_file):
    """Expert parallelism cuts the stack; each expert's 2-D unit stays whole.

    The fixture's packed experts carry the family whose row axis the loader
    refuses, so a packed expert read as column-parallel would be refused
    here. The receipt records the profile rule's answer beside the effective
    one, because a disagreement is worth seeing and is not a refusal.
    """
    _, receipt = _run(tmp_path, contract_file, tp=2)
    packed = [unit for unit in receipt["units"] if unit["packed_expert"]]
    assert {unit["unit"] for unit in packed} == {
        "model.layers.1.mlp.experts.gate_up_proj",
        "model.layers.1.mlp.experts.down_proj",
    }
    for unit in packed:
        assert unit["family"] == "TESSERA_E2M1_K2"
        assert unit["cut_kind"] == "none"
        assert unit["cut_kind_source"] == "expert_parallel_construction"
        assert unit["cut_axis"] is None
        assert unit["verdict"] == "pass"
    rule_kinds = {unit["unit"]: unit["profile_rule_kind"] for unit in packed}
    assert rule_kinds == {
        "model.layers.1.mlp.experts.gate_up_proj": "column",
        "model.layers.1.mlp.experts.down_proj": "row",
    }

    shared = next(unit for unit in receipt["units"]
                  if unit["unit"].endswith("shared_experts.gate_up_proj"))
    assert shared["packed_expert"] is False, (
        "a shared expert is an ordinary dense Linear and TP cuts it"
    )
    assert shared["cut_kind"] == "column"


def test_the_receipt_says_which_legs_it_did_not_audit(tmp_path, contract_file):
    """A receipt that hid the unaudited legs would read as a clearance."""
    _, receipt = _run(tmp_path, contract_file, tp=2)
    assert receipt["schema"] == audit.AUDIT_SCHEMA
    assert receipt["legs"]["loader_axis"] == "audited"
    assert receipt["legs"]["shard_geometry"].startswith("not audited")
    assert receipt["legs"]["attested_world"].startswith("not audited")
    assert receipt["require_attested_world"] is False
    assert receipt["contract"]["sha256"] == hashlib.sha256(
        Path(contract_file).read_bytes()).hexdigest()
    assert receipt["summary"]["non_tessera_units"] == 1, "lm_head is BF16"
    assert receipt["units"][0]["max_world_size"] == 1, (
        "what the contract does attest, recorded beside what it refuses"
    )


def test_the_audit_never_writes_the_assignment(tmp_path, contract_file):
    before = hashlib.sha256(Path(FIXTURE).read_bytes()).hexdigest()
    _run(tmp_path, contract_file, tp=2)
    assert hashlib.sha256(Path(FIXTURE).read_bytes()).hexdigest() == before


def test_the_receipt_can_be_written_to_stdout(tmp_path, contract_file, capsys):
    """``--out -`` so a receipt survives a run on a machine you do not keep."""
    code = audit.main([
        "--layer-config", FIXTURE, "--target-profile", PROFILE,
        "--tp", "2", "--contract", contract_file, "--out", "-",
    ])
    assert code == audit.EXIT_REFUSED
    captured = capsys.readouterr()
    assert json.loads(captured.out)["verdict"] == "refused"
    assert "1 refusal(s)" in captured.err


def test_with_no_table_the_audit_refuses_to_run(tmp_path, monkeypatch):
    """Certifying an assignment from silence is the failure to avoid."""
    monkeypatch.delenv(trc.TESSERA_DEV_PIN_ENV, raising=False)
    code = audit.main([
        "--layer-config", FIXTURE, "--target-profile", PROFILE,
        "--tp", "2", "--out", str(tmp_path / "receipt.json"),
    ])
    assert code == audit.EXIT_CANNOT_RUN


def test_an_unknown_profile_is_a_refusal_to_run_not_a_pass(tmp_path,
                                                           contract_file):
    code = audit.main([
        "--layer-config", FIXTURE, "--target-profile", "no_such_profile",
        "--tp", "2", "--contract", contract_file,
        "--out", str(tmp_path / "receipt.json"),
    ])
    assert code == audit.EXIT_CANNOT_RUN
