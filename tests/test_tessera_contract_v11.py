"""V11 rule coverage at the exact public v56 pin, with unchanged refusals."""
import hashlib
import json

import pytest

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_runtime_contract as runtime
from prismaquant import tessera_serving_runtime_pin as pin


def _parse(payload):
    return runtime._parse(payload, commit="fixture", sha="fixture", path="fixture")


def test_active_v56_answer_and_pin_match_the_installed_contract():
    payload = json.loads(runtime.contract_path().read_bytes())
    assert payload["contract_version"] == 56
    assert runtime._answer_drift(runtime.TESSERA_DEV_PIN_ANSWER,
                                runtime.contract_answer(_parse(payload))) == []


def _payload():
    payload = json.loads(runtime.contract_path().read_bytes())
    for family in payload["formats"]:
        family.pop("allowable_rungs", None)
    payload["lane_eligibility"]["schema"] = "tessera.lane-eligibility.v11"
    row = next(e for e in payload["formats"] if e["family"] == "TESSERA_E4M3_K1")
    row["allowable_rungs"] = {
        "rule": "window_rate_set", "code_arity": 1,
        "range_q256": [256, 2048], "step_q256": 1,
        "run_tables": sorted([[r] for r in range(1, 9)] + [[r, r+1] for r in range(1, 8)]),
        "excluded_run_tables": [], "excluded_q256": [],
        "wire": {k: v for k, v in row["attested_wire"][0].items() if k != "q256"},
        "evidence": ["docs/measurements/fixture.json"],
    }
    for cell in payload["lane_eligibility"]["cells"]:
        if cell["family"] == row["family"]:
            cell["run_tables"] = sorted({tuple([q//256] if q%256 == 0 else [q//256, q//256+1])
                                          for q in cell["rungs_q256"]})
            cell["run_tables"] = [list(t) for t in cell["run_tables"]]
        else:
            cell["run_tables"] = []
    return payload


def _row(payload):
    return next(e for e in payload["formats"] if e["family"] == "TESSERA_E4M3_K1")


def test_v11_preserves_census_and_derives_only_censused_tables():
    payload = _payload()
    parsed = _parse(payload)
    for raw, cell in zip(payload["lane_eligibility"]["cells"], parsed.cells):
        assert cell.rungs_q256 == frozenset(raw["rungs_q256"])
        if cell.family != "TESSERA_E4M3_K1":
            assert cell.covered_rungs_q256 == frozenset()
            continue
        tables = {tuple(t) for t in raw["run_tables"]}
        for q in range(255, 2050):
            table = (q//256,) if q%256 == 0 else (q//256, q//256+1)
            assert cell.covers_rate(q) == (q in raw["rungs_q256"] or
                                          256 <= q <= 2048 and table in tables)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(rule="unknown_rule"), "unknown allowable"),
    (lambda r: r.update(code_arity=2), "code_arity"),
    (lambda r: r.update(range_q256=[255, 2048]), "reader grid"),
    (lambda r: r.update(step_q256=0), "step_q256"),
    (lambda r: r.update(run_tables=[[1, 3]]), "invalid run table"),
    (lambda r: r.update(run_tables=[[2], [1]]), "ascending"),
    (lambda r: r.update(excluded_run_tables=[[1]]), "both allowed"),
    (lambda r: r.update(excluded_q256=[255]), "outside the rule"),
    (lambda r: r.update(evidence=["../receipt.json"]), "repository path"),
    (lambda r: r["wire"].update(plane="lut16"), "wire stamp"),
    (lambda r: r["wire"].update(seed=237), "census stamp"),
])
def test_v11_refuses_malformed_or_ununderstood_rule(mutation, match):
    payload = _payload()
    mutation(_row(payload)["allowable_rungs"])
    with pytest.raises(runtime.TesseraContractError, match=match):
        _parse(payload)


def test_v11_cell_cannot_claim_an_uncensused_run_table():
    payload = _payload()
    cell = next(c for c in payload["lane_eligibility"]["cells"]
                if c["family"] == "TESSERA_E4M3_K1")
    cell["run_tables"] = []
    with pytest.raises(runtime.TesseraContractError, match="allowable census"):
        _parse(payload)


@pytest.mark.parametrize("rungs", [None, [], [True], [[896]], [896, 896]])
def test_v11_malformed_census_is_a_named_refusal(rungs):
    payload = _payload()
    payload["lane_eligibility"]["cells"][0]["rungs_q256"] = rungs
    with pytest.raises(runtime.TesseraContractError, match="rungs_q256"):
        _parse(payload)


def test_v11_does_not_remove_the_plugin_requirement():
    payload = _payload()
    del payload["lane_eligibility"]["cells"][0]["requires_plugin"]
    with pytest.raises(runtime.TesseraContractError, match="requires_plugin"):
        _parse(payload)



def test_v11_exclusions_change_coverage_and_reviewed_answer():
    payload = _payload()
    before = _parse(payload)
    _row(payload)["allowable_rungs"]["excluded_q256"] = [897]
    after = _parse(payload)
    assert any(c.covers_rate(897) for c in before.cells if c.family == "TESSERA_E4M3_K1")
    assert not any(c.covers_rate(897) for c in after.cells if c.family == "TESSERA_E4M3_K1")
    assert runtime._answer_drift(runtime.contract_answer(before), runtime.contract_answer(after))
    assert runtime._answer_drift(runtime.TESSERA_DEV_PIN_ANSWER, runtime.contract_answer(before))


def test_v11_coverage_keeps_exact_context_and_compiled_refusal():
    parsed = _parse(_payload())
    assert parsed.native_cells("TESSERA_E4M3_K1", 897) == ()
    cell = next(c for c in parsed.cells if c.family == "TESSERA_E4M3_K1" and c.covers_rate(897))
    compiled = lane.ServingContext(cell.platform, cell.structure, cell.residency_modes[0],
                                   cell.runtime_image, "compiled")
    assert parsed.native_cells(cell.family, 897, serving_context=compiled) == ()


def test_v11_still_refuses_a_foreign_installed_contract():
    candidate = _parse(_payload())
    assert runtime.TESSERA_DEV_PIN_COMMIT == "2dbac1910c88254d9c6391f02a34c4b07e516803"
    assert pin.load_tessera_serving_runtime_pin().commit == runtime.TESSERA_DEV_PIN_COMMIT
    with pytest.raises(pin.TesseraServingRuntimePinError):
        pin.require_exact_tessera_runtime_pin(pin.load_tessera_serving_runtime_pin(),
                                              installed_contract_sha256="0"*64)
    assert candidate.lane_schema == "tessera.lane-eligibility.v11"


def test_installed_immutable_v56_package_scope():
    raw = runtime.contract_path().read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "47f180efaf97faa5c411df5d48f9da7dff4b9c9fc0c3ddbf9f815bcd4d0aed78"
    payload = json.loads(raw)
    parsed = _parse(payload)
    assert parsed.contract_version == 56
    assert len(parsed.cells) == 22
    assert all("compiled" not in c.execution_modes for c in parsed.cells)
    assert all(c.requires_plugin == "tessera" for c in parsed.cells)
    assert runtime._answer_drift(runtime.TESSERA_DEV_PIN_ANSWER, runtime.contract_answer(parsed)) == []
