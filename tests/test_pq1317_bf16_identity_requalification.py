"""CPU-only regression checks for BF16 retention, NOT GPU qualification evidence.

Mock pytest items expose real torch CPU operands in live Python frames. Synthetic
CUDA-labelled JSON below tests only the refusal/acceptance contract of the merger;
it is fabricated schema data, not a claim that any CUDA operation was performed.
No candidate code, native probe, container runner or production path is changed.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "docs/measurements/pq1317-gpu-tests/harness"
spec = importlib.util.spec_from_file_location("pq1317_bf16_oracle", HARNESS / "bf16_oracle.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

torch = pytest.importorskip("torch")
CASES = [(nodeid, actual, reference, m)
         for nodeid, comparisons in oracle.EXPECTED_COMPARISONS.items()
         for actual, reference, m in comparisons]
SINGLE_NODE = next(iter(oracle.EXPECTED_COMPARISONS))
GEMV_NODE = f"{oracle.TEST_FILE}::test_gemv_and_torch_agree_with_measured_differences"
RATE_NODE = f"{oracle.TEST_FILE}::test_rate1_columns_keep_dense_dispatch_across_decode_sizes"


def capture(nodeid, actual_name, reference_name, actual, reference, bounds, *, orig=None):
    """A mocked item, but real live locals and real CPU tensor arithmetic."""
    expression = orig or f"bool((({actual_name}.float() - {reference_name}).abs() <= bound).all())"

    def live(got=None, exact=None, ref=None, y_gemv=None, y_torch=None,
             y_eager=None, y_cap=None, bound=None):
        return oracle.observe_bf16_assertion(item, sys._getframe().f_lineno, expression)

    item = SimpleNamespace(nodeid=nodeid, obj=live)
    return live(**{actual_name: actual, reference_name: reference, "bound": bounds})


def cpu_operands(m=2):
    actual = torch.ones((m, 64), dtype=torch.bfloat16)
    reference = actual.float()
    reference[0, 0] -= 0.125
    bound = torch.full((m, 64), 0.25, dtype=torch.float64)
    bound[0, 0] = 0.2
    return actual, reference, bound


def record_for(nodeid, measurements):
    return {"kind": "test", "nodeid": nodeid, "when": "call", "outcome": "passed",
            "assertions_dropped": 0,
            "assertions": [{"numerical": measurement} for measurement in measurements]}


def synthetic_cuda_records():
    """Fabricated merge fixtures. No CPU measurement is relabelled as GPU proof."""
    records = []
    for nodeid, comparisons in oracle.EXPECTED_COMPARISONS.items():
        measurements = []
        for actual, reference, m in comparisons:
            measurements.append({
                "schema": oracle.SCHEMA, "nodeid": nodeid, "line": 1,
                "orig": f"bool((({actual}.float() - {reference}).abs() <= bound).all())",
                "actual": actual, "reference": reference, "bound": "bound",
                "shape": [m, 64], "numel": m * 64, "device": "cuda:3",
                "actual_dtype": "torch.bfloat16", "reference_dtype": "torch.float32",
                "bound_dtype": "torch.float64", "finite": True,
                "max_abs_error": 0.125, "mean_abs_error": 0.125 / (m * 64),
                "bound_min": 0.2, "bound_max": 0.25, "max_violation": 0.125 - 0.2,
                "violation_count": 0,
            })
        records.append(record_for(nodeid, measurements))
    return records


def test_retention_scope_is_19_nodes_23_comparisons_in_current_roster():
    assert len(oracle.EXPECTED_COMPARISONS) == 19
    assert len(CASES) == 23
    roster = json.loads((HARNESS.parent / "roster.json").read_text())
    ids = {n["id"] for n in roster["nodes"]}
    assert set(oracle.EXPECTED_COMPARISONS) <= ids
    assert len(oracle.EXPECTED_COMPARISONS[GEMV_NODE]) == 2
    assert len(oracle.EXPECTED_COMPARISONS[RATE_NODE]) == 2


@pytest.mark.parametrize("nodeid,actual_name,reference_name,m", CASES)
def test_each_live_comparison_retains_cpu_errors_and_actual_bound(nodeid, actual_name, reference_name, m):
    operands = cpu_operands(m)
    before = [t.clone() for t in operands]
    measured = capture(nodeid, actual_name, reference_name, *operands)
    assert measured["device"] == "cpu"  # Explicitly NOT GPU evidence.
    assert measured["finite"] is True
    assert measured["shape"] == [m, 64]
    assert measured["numel"] == m * 64
    assert measured["max_abs_error"] == 0.125
    assert measured["mean_abs_error"] == 0.125 / (m * 64)
    assert measured["bound_min"] == 0.2
    assert measured["bound_max"] == 0.25
    assert measured["max_violation"] == pytest.approx(-0.075)
    assert measured["violation_count"] == 0
    assert all(type(measured[key]) is float for key in (
        "max_abs_error", "mean_abs_error", "bound_min", "bound_max", "max_violation"))
    assert all(torch.equal(a, b) for a, b in zip(operands, before))
    assert json.loads(json.dumps(measured, allow_nan=False)) == measured
    reasons = oracle.bf16_oracle_errors([record_for(nodeid, [measured])], [nodeid])
    assert any("not from a CUDA device" in reason for reason in reasons)


def test_two_outputs_retain_distinct_live_errors_not_only_the_last_output():
    actual, ref, bound = cpu_operands(4)
    first = capture(GEMV_NODE, "y_gemv", "ref", actual, ref, bound)
    second = capture(GEMV_NODE, "y_torch", "ref", actual + 0.5, ref, bound)
    record = record_for(GEMV_NODE, [first, second])
    assert [a["numerical"]["actual"] for a in record["assertions"]] == ["y_gemv", "y_torch"]
    assert first["max_abs_error"] == 0.125
    assert second["max_abs_error"] == 0.625
    assert second["violation_count"] == 4 * 64
    assert second["max_violation"] == pytest.approx(0.425)


def test_elementwise_violation_is_not_hidden_by_largest_bound():
    actual, ref, bound = cpu_operands()
    bound[0, 0] = 0.1
    bound[0, 1] = 100.0
    measured = capture(SINGLE_NODE, "got", "exact", actual, ref, bound)
    assert measured["max_abs_error"] < measured["bound_max"]
    assert measured["violation_count"] == 1
    assert measured["max_violation"] == pytest.approx(0.025)


@pytest.mark.parametrize("target", ["actual", "reference", "bound"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_live_operand_is_an_explicit_error_record(target, value):
    operands = dict(zip(("actual", "reference", "bound"), cpu_operands()))
    operands[target][0, 0] = value
    measured = capture(SINGLE_NODE, "got", "exact", *operands.values())
    assert "nonfinite" in measured["error"]
    assert "max_abs_error" not in measured
    json.dumps(measured, allow_nan=False)
    assert oracle.bf16_oracle_errors([record_for(SINGLE_NODE, [measured])], [SINGLE_NODE])


@pytest.mark.parametrize("problem", ["none", "dtype", "shape", "empty", "negative", "device"])
def test_invalid_live_operands_fail_closed(problem):
    actual, ref, bound = cpu_operands()
    if problem == "none":
        actual = None
    elif problem == "dtype":
        actual = actual.float()
    elif problem == "shape":
        bound = bound[:1]  # Broadcasting must not silently qualify.
    elif problem == "empty":
        actual, ref, bound = actual[:0], ref[:0], bound[:0]
    elif problem == "negative":
        bound[0, 0] = -1
    else:
        bound = torch.empty(bound.shape, dtype=torch.float64, device="meta")
    measured = capture(SINGLE_NODE, "got", "exact", actual, ref, bound)
    assert measured.get("error")
    assert "max_abs_error" not in measured


def test_only_known_node_and_exact_elementwise_expression_are_observed():
    operands = cpu_operands()
    assert capture("other.py::test_anything", "got", "exact", *operands) is None
    for orig in ("layer.tessera_native is not None", "bool((got == exact).all())",
                 "bool(((got.float() - exact).abs() < bound).all())"):
        assert capture(SINGLE_NODE, "got", "exact", *operands, orig=orig) is None
    assert capture(SINGLE_NODE, "y_gemv", "ref", *operands) is None


def test_missing_live_frame_or_local_is_not_text_based_success():
    def not_running():
        pass

    item = SimpleNamespace(nodeid=SINGLE_NODE, obj=not_running)
    orig = "bool(((got.float() - exact).abs() <= bound).all())"
    measured = oracle.observe_bf16_assertion(item, 1, orig)
    assert "frame/line unavailable" in measured["error"]

    def missing_local():
        return oracle.observe_bf16_assertion(item, sys._getframe().f_lineno, orig)

    item.obj = missing_local
    assert "KeyError" in missing_local()["error"]


def test_complete_synthetic_merge_fixture_accepts_all_23_independent_comparisons():
    records = synthetic_cuda_records()
    records = json.loads(json.dumps(records, allow_nan=False))
    assert oracle.bf16_oracle_errors(records, oracle.EXPECTED_COMPARISONS) == []
    assert sum(len(record["assertions"]) for record in records) == 23
    assert oracle.bf16_oracle_errors(records, [*oracle.EXPECTED_COMPARISONS, "other.py::test_other"]) == []


@pytest.mark.parametrize("problem", ["node", "comparison", "duplicate-node", "duplicate-comparison",
                                    "replace-loop-iteration", "failed", "setup", "dropped", "no-assertions",
                                    "boolean-text-only", "observer-error"])
def test_gate_refuses_absent_or_incomplete_required_evidence(problem):
    records = synthetic_cuda_records()
    node = next(r for r in records if r["nodeid"] == RATE_NODE)
    if problem == "node":
        records.remove(node)
    elif problem == "comparison":
        node["assertions"].pop()
    elif problem == "duplicate-node":
        records.append(copy.deepcopy(node))
    elif problem == "duplicate-comparison":
        node["assertions"].append(copy.deepcopy(node["assertions"][0]))
    elif problem == "replace-loop-iteration":
        node["assertions"][1] = copy.deepcopy(node["assertions"][0])
    elif problem == "failed":
        node["outcome"] = "failed"
    elif problem == "setup":
        node["when"] = "setup"
    elif problem == "dropped":
        node["assertions_dropped"] = 1
    elif problem == "no-assertions":
        del node["assertions"]
    elif problem == "boolean-text-only":
        node["assertions"] = [{"orig": "bool(...)", "evaluated": "True"}]
    else:
        node["assertions"][0]["numerical"]["error"] = "operands unavailable"
    assert oracle.bf16_oracle_errors(records, oracle.EXPECTED_COMPARISONS)


@pytest.mark.parametrize("field,value", [
    ("schema", "unknown"), ("nodeid", "wrong-node"), ("line", 0), ("orig", "True"),
    ("actual", "other"), ("bound", "guessed"), ("device", "cpu"), ("device", "cuda"),
    ("actual_dtype", "torch.float32"), ("reference_dtype", "torch.float64"),
    ("bound_dtype", "torch.float32"), ("shape", [2, 1]), ("numel", 1), ("finite", False),
    ("max_abs_error", True), ("mean_abs_error", float("nan")), ("bound_min", float("inf")),
    ("bound_max", -1.0), ("max_violation", 0.01), ("max_violation", -100.0),
    ("violation_count", 1), ("violation_count", False),
])
def test_gate_rejects_invalid_or_nonfinite_numerical_fields(field, value):
    record = synthetic_cuda_records()[0]
    record["assertions"][0]["numerical"][field] = value
    assert oracle.bf16_oracle_errors([record], [record["nodeid"]])


@pytest.mark.parametrize("field", ["max_abs_error", "mean_abs_error", "bound_min", "bound_max",
                                   "max_violation", "violation_count", "finite", "device", "shape"])
def test_gate_requires_each_numerical_evidence_field(field):
    record = synthetic_cuda_records()[0]
    del record["assertions"][0]["numerical"][field]
    assert oracle.bf16_oracle_errors([record], [record["nodeid"]])


def test_gate_does_not_infer_expected_nodes_from_observed_records():
    assert len(oracle.bf16_oracle_errors([], oracle.EXPECTED_COMPARISONS)) == 19
    record = synthetic_cuda_records()[0]
    nodeid = record["nodeid"]
    record["assertions"] = [{"numerical": None}]
    assert oracle.bf16_oracle_errors([record], [nodeid])
