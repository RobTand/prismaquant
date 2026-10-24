"""Portable D43 acceptance, not another scientific implementation (#2885-2887).

The implementer supplies fixtures/pact_d43/binding.json and cases.json. Keep the
historical source under oracle/ test-only. The binding is custody evidence, not
Campaign acceptance. Cases invoke the designated owners' real public Python
entries (including CLI main(argv)); no test driver or parallel solver is loaded.
All arguments/outputs are JSON, with $FIXTURE/ and $OUTPUT/ path substitutions.
Missing accepted inputs must be reported, never filled with synthetic prices.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "pact_d43"
CANONICAL = {
    "frontier.py": "233c5d8341326a506ec539e759f08fb25c83602673485333efffd38b515cd305",
    "manifest.json": "12cb8430a837723b1c965331998998b6b22c0733e07aed1ef74b63c9ee4d6cf2",
    "test_frontier_bound.py": "f2a5bddf31fe5da91712f65dc2066248ef97b03be364fcb4ea1d9bfc31c6f7c5",
}
ROLES = {
    "price_tables", "from_format_anchors", "d41_index", "d41_api", "d41_tables",
    "byte_audit", "heldout_preregistration", "stack_scope", "weight_side_bound",
    "thresholds", "predictor", "lever_pairs", "calibration", "teacher",
}
DIMENSIONS = {"objectives", "prices", "wire_bytes", "assignments", "replay_inputs", "refusals"}


def _json(path):
    assert path.is_file(), (
        f"Missing portable D43 acceptance artifact: {path}. "
        "Bind the authoritative source closure; fleet-path availability is not acceptance."
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _local(base, relative):
    assert isinstance(relative, str) and relative and not Path(relative).is_absolute()
    path = (base / relative).resolve()
    assert path.is_relative_to(base.resolve()), f"Escaping fixture/output path: {relative}"
    return path


def _sha(path):
    assert path.is_file(), f"Missing bound fixture bytes: {path}"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle():
    binding = _json(FIXTURE / "binding.json")
    assert binding["schema"] == "prismaquant.pact_d43_binding.v1"
    return binding


def _oracle():
    # No git extraction, fleet module, installed-current D41 substitution or download.
    path = FIXTURE / "oracle" / "frontier.py"
    assert _sha(path) == CANONICAL["frontier.py"]
    name = "_pact_d43_historical_test_oracle"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def test_complete_source_closure_is_bound_and_acceptance_is_explicit():
    binding = _bundle()
    for name, digest in CANONICAL.items():
        assert _sha(FIXTURE / "oracle" / name) == digest
    assert binding["standing"] == "research_only"
    assert binding["scientific_currency"] != "operator_runtime_ms"
    assert ROLES <= set(binding["roles"])
    sources = binding["sources"]
    assert sources
    for role in ROLES:
        ids = binding["roles"][role]
        assert isinstance(ids, list) and ids, f"Unbound {role}"
        assert all(key in sources for key in ids)
    for key, source in sources.items():
        assert source["source_identity"], key
        assert len(source["sha256"]) == 64 and int(source["sha256"], 16) >= 0
        assert source["bytes"] > 0
        assert all(dep in sources for dep in source["dependencies"]), key
        acceptance = source["campaign_acceptance"]
        assert acceptance["status"] in {"accepted", "unaccepted"}
        assert acceptance["owner"] and acceptance["evidence"], key
        projection = source["fixture"]
        assert _sha(_local(FIXTURE, projection["path"])) == projection["sha256"]
        # A subset needs source selectors/derivation, never an invented replacement.
        assert projection["derivation"], key
    manifest = _json(FIXTURE / "oracle" / "manifest.json")
    overlays = [sources[key] for key in binding["roles"]["price_tables"]]
    assert [row["source_identity"] for row in overlays] == [
        row["path"] for row in manifest["price_tables"]
    ], "Overlay precedence is part of the scientific input identity"
    assert binding["thresholds"] == manifest["thresholds"]
    assert binding["predictor"] == manifest["predictor"]
    assert binding["authority"] == manifest["authority"]
    assert binding["price_statuses"] == [row["status"] for row in manifest["price_tables"]]
    assert binding["qualification"]["native"] == "independent"
    assert binding["qualification"]["export_serving"] == "independent"


def test_source_bound_canonical_results_and_invalid_rows_are_reproducible():
    binding = _bundle()
    oracle = _oracle()
    cases = _json(FIXTURE / "cases.json")["canonical"]
    assert cases, "Commit at least one real, small two-class GLM population"
    saw_over_bound = False
    for case in cases:
        assert case["profile"] == "glm5_next"
        assert case["source_ids"] and all(key in binding["sources"] for key in case["source_ids"])
        # Canonical options retain original unit/rate/price/byte/timing data.
        # Case derivation must identify each row and any reduced-population budget.
        assert case["derivation"]
        grouped, count = oracle.exact_two_class(
            case["options"], case["time_classes"], case["baseline_bytes"],
            case["cap"], case["beam"],
        )
        assert count == case["expected_candidate_count"]
        assert _plain(grouped) == case["expected_grouped"]
        actual = []
        for point in case["point_inputs"]:
            row = oracle.point_row(**point)
            actual.append(row)
            if row["remaining_budget"] < 0:
                saw_over_bound = True
                assert row["invalid"] is True and row["threshold_feasible"] is False
                assert row["weight_side_bound_would_bind"] is True
                assert row["remaining_budget_after_bound"] == row["remaining_budget"]
        assert _plain(actual) == case["expected_points"]
        assert DIMENSIONS <= set(case["expected_contract"])
        assert case["expected_contract"]["wire_bytes"], "Use actual serialized wire evidence"
    assert saw_over_bound, "Retain an actual over-bound invalid row; do not clip it"


def _expand(value, output):
    if isinstance(value, str):
        if value.startswith("$FIXTURE/"):
            return str(_local(FIXTURE, value[len("$FIXTURE/"):]))
        if value.startswith("$OUTPUT/"):
            return str(_local(output, value[len("$OUTPUT/"):]))
        return value
    if isinstance(value, list):
        return [_expand(item, output) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, output) for key, item in value.items()}
    return value


def _entry(entry):
    # Declared *real* owner APIs; fixture Python drivers are deliberately forbidden.
    assert entry["module"].startswith(("prismaquant.", "experiments."))
    module = importlib.import_module(entry["module"])
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / "tests")
    assert not entry["function"].startswith("_")
    return getattr(module, entry["function"])


def _observe(case, result, output):
    """Project native API/file results in TESTS; require no production result shim.

    observations maps each contract dimension to {file: relative_path, pointer:
    [JSON object keys / array indices]}, or {return: true, pointer: [...]}.
    """
    observed = {}
    for dimension, selector in case["observations"].items():
        value = result if selector.get("return") else _json(_local(output, selector["file"]))
        for key in selector["pointer"]:
            value = value[key]
        observed[dimension] = _plain(value)
    return observed


@pytest.mark.parametrize("profile", ["glm5_next", "qwen3"])
def test_actual_frontend_harness_price_frontier_flow(profile, tmp_path, monkeypatch):
    binding = _bundle()
    population = _json(FIXTURE / "cases.json")
    cases = [case for case in population["consumers"] if case["profile"] == profile]
    assert cases, f"No real public-entry CPU cases for {profile}"
    # D43's exact two-class solver is not the operator-runtime hull/sweep path.
    # Their shared selection/interval preservation is covered by existing suites.
    assert any(not case["refusal"] for case in cases), "Missing successful public-entry flow"
    assert any(case["refusal"] for case in cases), "Missing scope must refuse, not borrow GLM"
    canonical = {case["id"]: case for case in population["canonical"]}
    # Working directory is scratch, so an accidental cwd-relative external read fails.
    monkeypatch.chdir(tmp_path)
    for index, case in enumerate(cases):
        output = tmp_path / str(index)
        output.mkdir()
        owners = {role: _entry(binding["consumer_entries"][role]) for role in
                  ("frontend", "harness", "price", "frontier")}
        called = set()
        def trace(frame, event, arg):
            if event == "call":
                for role, function in owners.items():
                    if frame.f_code is function.__code__:
                        called.add(role)
                if frame.f_globals.get("__name__") == "prismaquant.pact_frontier_profile":
                    called.add("profile")
                assert not str(frame.f_globals.get("__name__", "")).startswith(
                    "_pact_d43_historical_test_oracle"
                ), "The live consumer must not execute historical test-only code"
        previous = sys.getprofile()
        try:
            sys.setprofile(trace)
            if case.get("exception"):
                assert case["refusal"]
                exceptions = {"ValueError": ValueError, "RuntimeError": RuntimeError,
                              "SystemExit": SystemExit}
                with pytest.raises(exceptions[case["exception"]["type"]]) as caught:
                    owners["frontend"](
                        *_expand(case["args"], output), **_expand(case["kwargs"], output)
                    )
                assert str(caught.value) == case["exception"]["message"]
                result = None
            else:
                result = owners["frontend"](
                    *_expand(case["args"], output), **_expand(case["kwargs"], output)
                )
        finally:
            sys.setprofile(previous)
        assert "frontend" in called and "profile" in called
        if case.get("exception"):
            continue
        if not case["refusal"]:
            assert {"harness", "price", "frontier"} <= called
        actual = _observe(case, result, output)
        expected = (canonical[case["canonical_id"]]["expected_contract"]
                    if profile == "glm5_next" and not case["refusal"] else case["expected_contract"])
        assert DIMENSIONS <= set(expected)
        assert actual == expected
        if case["refusal"]:
            assert actual["refusals"] and not actual["assignments"]
        for wire in case["wire_outputs"]:
            assert _local(output, wire["output"]).read_bytes() == _local(FIXTURE, wire["fixture"]).read_bytes()
        if not case["refusal"]:
            assert case["wire_outputs"], "Count actual wire bytes, not a resident-memory estimate"
