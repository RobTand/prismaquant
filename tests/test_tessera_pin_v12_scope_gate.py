"""Pin-update gate on v12 per-launch scope proof (PQ #2600).

A pin update to ``prismaquant/tessera_runtime/tessera_serving_runtime_pin.json``
is refused unless the four named v12 scope fixture classes -- census, derived,
absent and malformed -- all pass against the reader. The checks reuse the
fixture in ``tests/test_tessera_lane_v12.py``, not a new copy.
``check_pin_update`` reads the tracked pin file, so the gate below runs
against the real pin, not a caller-built dict.

A producer schema change is refused unless its PR links consumer
compatibility work and reports explicit passing fixture-result records.
That rule lives in
``prismaquant/tessera_pin_scope_gate.py`` beside the pin gate, as a rule
plus unit tests: no caller wires it into a PR check in this repository.
One test per scope class breaks the real check (not a ``_CHECKS`` stub)
and asserts the refusal reason.
"""
import dataclasses
import json

import pytest

from prismaquant import tessera_pin_scope_gate as gate


PRODUCER_RESULTS = "\n".join(
    f"fixture-result: {name} PASS https://ci.example/results/v12#{name}"
    for name in gate.REQUIRED_SCOPE_CLASSES
)


def test_pin_update_passes_with_tracked_pin_and_passing_fixtures():
    pin = gate.check_pin_update()
    assert pin.commit
    results = gate.run_v12_scope_proof()
    assert set(results) == set(gate.REQUIRED_SCOPE_CLASSES)
    assert all(proof.passed for proof in results.values())
    assert all(proof.reason == "" for proof in results.values())
    gate.require_v12_scope_proof(results)


@pytest.mark.parametrize("missing", list(gate.REQUIRED_SCOPE_CLASSES))
def test_pin_update_refuses_a_missing_scope_class(monkeypatch, missing):
    monkeypatch.setattr(
        gate, "_CHECKS",
        {k: v for k, v in gate._CHECKS.items() if k != missing},
    )
    with pytest.raises(gate.TesseraPinScopeGateError, match=missing):
        gate.check_pin_update()


@pytest.mark.parametrize("failing", list(gate.REQUIRED_SCOPE_CLASSES))
def test_pin_update_refuses_a_failing_scope_class(monkeypatch, failing):
    def _boom():
        raise ValueError(f"boom detail for {failing}")

    monkeypatch.setitem(gate._CHECKS, failing, _boom)
    with pytest.raises(gate.TesseraPinScopeGateError, match=failing) as refused:
        gate.check_pin_update()
    assert f"boom detail for {failing}" in str(refused.value)


def test_scope_proof_keeps_the_failure_reason(monkeypatch):
    def _boom():
        raise ValueError("boom detail")

    monkeypatch.setitem(gate._CHECKS, "census", _boom)
    results = gate.run_v12_scope_proof()
    assert results["census"].passed is False
    assert "boom detail" in results["census"].reason
    with pytest.raises(gate.TesseraPinScopeGateError, match="boom detail"):
        gate.require_v12_scope_proof(results)


def test_pin_gate_refuses_an_empty_proof():
    with pytest.raises(gate.TesseraPinScopeGateError):
        gate.require_v12_scope_proof({})


def test_malformed_class_does_not_count_a_broken_baseline_as_scope_refusal(monkeypatch):
    def broken_parser(*args, **kwargs):
        raise gate.lane.LaneEligibilityError("unrelated parser failure")

    monkeypatch.setattr(gate.lane, "_parse_table", broken_parser)
    proof = gate.run_v12_scope_proof()["malformed"]
    assert not proof.passed
    assert "unrelated parser failure" in proof.reason


@pytest.mark.parametrize("linked", [[], ["", "  "]])
def test_producer_schema_pr_refuses_without_linked_consumer_work(linked):
    with pytest.raises(gate.ProducerSchemaPRError, match="consumer"):
        gate.check_producer_schema_pr(body=PRODUCER_RESULTS, linked=linked)


def test_producer_schema_pr_refuses_without_fixture_results():
    with pytest.raises(gate.ProducerSchemaPRError, match="[Ff]ixture"):
        gate.check_producer_schema_pr(
            body="see the linked consumer issue",
            linked=["prismaquant#2600"],
        )


@pytest.mark.parametrize("missing", gate.REQUIRED_SCOPE_CLASSES)
def test_producer_schema_pr_refuses_when_one_scope_result_is_missing(missing):
    body = "\n".join(
        line for line in PRODUCER_RESULTS.splitlines()
        if not line.startswith(f"fixture-result: {missing} ")
    )
    with pytest.raises(gate.ProducerSchemaPRError, match=missing):
        gate.check_producer_schema_pr(body=body, linked=["prismaquant#2600"])


@pytest.mark.parametrize("body", [
    "No fixture results exist yet. TODO: census, derived, absent, malformed.",
    "fixtures: census pass, derived pass, absent pass, malformed pass",
    "fixture-result: census PASS\nTODO: derived, absent, malformed",
])
def test_producer_schema_pr_refuses_mentions_without_result_records(body):
    with pytest.raises(gate.ProducerSchemaPRError, match="fixture"):
        gate.check_producer_schema_pr(body=body, linked=["prismaquant#2600"])


@pytest.mark.parametrize("name", gate.REQUIRED_SCOPE_CLASSES)
@pytest.mark.parametrize("replacement", ["FAIL", "TODO", "SKIP"])
def test_producer_schema_pr_refuses_nonpassing_results(name, replacement):
    body = PRODUCER_RESULTS.replace(f"{name} PASS", f"{name} {replacement}")
    with pytest.raises(gate.ProducerSchemaPRError, match=f"{name} is not PASS"):
        gate.check_producer_schema_pr(body=body, linked=["prismaquant#2600"])


@pytest.mark.parametrize("evidence", ["TODO", "https:///missing-host", "http://ci.example/run"])
def test_producer_schema_pr_refuses_results_without_evidence_links(evidence):
    body = PRODUCER_RESULTS.replace("https://ci.example/results/v12#derived", evidence)
    with pytest.raises(gate.ProducerSchemaPRError, match="derived.*HTTPS"):
        gate.check_producer_schema_pr(body=body, linked=["prismaquant#2600"])


def test_producer_schema_pr_refuses_duplicate_results():
    body = PRODUCER_RESULTS + "\n" + PRODUCER_RESULTS.splitlines()[0]
    with pytest.raises(gate.ProducerSchemaPRError, match="duplicate.*census"):
        gate.check_producer_schema_pr(body=body, linked=["prismaquant#2600"])


def test_producer_schema_pr_admits_with_linked_work_and_all_fixture_results():
    gate.check_producer_schema_pr(
        body=f"## Scope proof\n```text\n{PRODUCER_RESULTS}\n```",
        linked=["prismaquant#2600"],
    )


def test_pin_gate_refuses_a_real_census_regression(monkeypatch):
    assert gate.check_pin_update().commit
    real_payload = gate._payload

    def bad_payload():
        payload = real_payload()
        payload["contract_version"] = 65
        return payload

    monkeypatch.setattr(gate, "_payload", bad_payload)
    with pytest.raises(gate.TesseraPinScopeGateError, match="census") as refused:
        gate.check_pin_update()
    assert "contract version is not 66" in str(refused.value)


def test_pin_gate_refuses_a_real_derived_regression(monkeypatch):
    assert gate.check_pin_update().commit
    real_parsed = gate._parsed

    def bad_parsed(payload=None):
        parsed = real_parsed(payload)
        cells = []
        for cell in parsed.cells:
            if cell.cell_id == "tessera_e4m3_k1_routed_moe_sm121_decode_resident":
                covered = tuple(
                    tuple(r for r in entry if r != 800)
                    if entry is not None else None
                    for entry in cell.launch_covered_rungs_q256
                )
                cell = dataclasses.replace(
                    cell, launch_covered_rungs_q256=covered)
            cells.append(cell)
        return dataclasses.replace(parsed, cells=tuple(cells))

    monkeypatch.setattr(gate, "_parsed", bad_parsed)
    with pytest.raises(
        gate.TesseraPinScopeGateError,
        match="derived: route cell loses the derived launch coverage",
    ):
        gate.check_pin_update()


def test_pin_gate_refuses_a_real_absent_leak(monkeypatch):
    assert gate.check_pin_update().commit
    real_answer = gate.runtime.contract_answer

    def leaking_answer(contract):
        answer = real_answer(contract)
        cells = []
        for row in answer["cells"]:
            row = list(row)
            if not any(
                isinstance(entry, dict) and "launch_scopes" in entry
                for entry in row
            ):
                row.append({"launch_scopes": [
                    {"symbol": "leak", "decoder": "leak",
                     "rungs_q256": [768]}]})
            cells.append(row)
        return {**answer, "cells": cells}

    monkeypatch.setattr(gate.runtime, "contract_answer", leaking_answer)
    with pytest.raises(gate.TesseraPinScopeGateError, match="absent") as refused:
        gate.check_pin_update()
    assert "absent: v11 answer row" in str(refused.value)


def test_pin_gate_refuses_a_real_malformed_regression(monkeypatch):
    assert gate.check_pin_update().commit
    real_parse_table = gate.lane._parse_table
    good = json.loads(gate.V12_FIXTURE.read_bytes())

    def soft_parse_table(*args, **kwargs):
        try:
            return real_parse_table(*args, **kwargs)
        except gate.lane.LaneEligibilityError:
            return real_parse_table(
                good["lane_eligibility"], good["formats"],
                "fixture", "fixture", "fixture",
                native_extensions=good["native_extensions"],
            )

    monkeypatch.setattr(gate.lane, "_parse_table", soft_parse_table)
    with pytest.raises(
        gate.TesseraPinScopeGateError, match="malformed"
    ) as refused:
        gate.check_pin_update()
    assert "malformed scope admitted" in str(refused.value)
