"""Retain the live BF16 elementwise oracle, without rewriting candidate tests.

Scope: tests/test_serving_bf16_gemv.py at candidate fca4c6ce0; 19 nodes,
23 comparisons. These are numerical component checks, not serving qualification.
The supplied candidate has no tests/test_dense_serving_dispatch.py.

Integration (not installed by this module): inside pytest_assertion_pass, call
``observe_bf16_assertion(item, lineno, orig)`` BEFORE any text-based early return.
If non-None, append it as ``{"numerical": measurement}`` to that node's assertions
(or add ``numerical`` to the existing entry). Never replace an earlier comparison.
At merge, extend refusal reasons with ``bf16_oracle_errors(test_records, expected)``;
``expected`` is the existing suite roster, not ids inferred from observed records.
This gate supplements, never replaces, source/native identity and pytest gates.

Torch is imported lazily. Observation uses existing tensors only, does not run a
forward/reference builder, and never evaluates assertion text. CPU observations
are useful for unit tests but are ALWAYS refused by the qualification gate.
"""
from __future__ import annotations

from collections import Counter
import math
import sys

SCHEMA = "pq1317.bf16-oracle.v1"
TEST_FILE = "tests/test_serving_bf16_gemv.py"

# (actual local, reference local, M). All retained outputs have 64 columns and
# compare against the LIVE local `bound`, including its fp32 accumulation term.
EXPECTED_COMPARISONS = {
    f"{TEST_FILE}::test_rung_outside_reference_gemv_range_uses_dense_native": (("got", "exact", 2),),
    **{
        f"{TEST_FILE}::test_decode_regime_serves_the_folded_native_window_gemm[{m}-{lane}]":
        (("got", "exact", m),)
        for m in (1, 2, 3, 4, 5, 8) for lane in ("fused", "triton")
    },
    **{
        f"{TEST_FILE}::test_prefill_keeps_the_folded_native_window_path[{m}]": (("got", "exact", m),)
        for m in (16, 64)
    },
    f"{TEST_FILE}::test_gemv_and_torch_agree_with_measured_differences":
        (("y_gemv", "ref", 4), ("y_torch", "ref", 4)),
    f"{TEST_FILE}::test_rate1_columns_keep_dense_dispatch_across_decode_sizes":
        (("got", "exact", 2), ("got", "exact", 4)),
    **{
        f"{TEST_FILE}::test_the_materialised_branch_survives_cuda_graph_capture[{param}]":
        (("y_eager", "exact", m), ("y_cap", "exact", m))
        for param, m in (("prefill-m16", 16), ("rate1-fallback-m4", 4))
    },
}


def _expression(actual: str, reference: str) -> str:
    return f"bool((({actual}.float()-{reference}).abs()<=bound).all())"


_OPERANDS = {_expression(actual, reference): (actual, reference)
             for comparisons in EXPECTED_COMPARISONS.values()
             for actual, reference, _ in comparisons}


def observe_bf16_assertion(item, lineno: int, orig: str) -> dict | None:
    """Called synchronously from the assertion-pass hook while test locals live.

    Return None for unrelated assertions. Recognized assertions with missing or
    invalid operands yield an error record, never a successful numerical record.
    No frames/tensors are retained, and no candidate tensor is mutated.
    """
    nodeid = item.nodeid
    operands = _OPERANDS.get("".join(orig.split()))
    expected = EXPECTED_COMPARISONS.get(nodeid, ())
    if operands is None or not any(operands == pair[:2] for pair in expected):
        return None
    actual_name, reference_name = operands
    record = {"schema": SCHEMA, "nodeid": nodeid, "line": lineno, "orig": orig,
              "actual": actual_name, "reference": reference_name, "bound": "bound"}
    frame = sys._getframe(1)
    try:
        code = item.obj.__code__
        while frame is not None and frame.f_code is not code:
            frame = frame.f_back
        if frame is None or frame.f_lineno != lineno:
            raise ValueError("live assertion frame/line unavailable")
        local = frame.f_locals
        record.update(_measure(local[actual_name], local[reference_name], local["bound"]))
    except Exception as error:  # Observation failure is evidence of refusal, not a test rewrite.
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        del frame
    return record


def _measure(actual, reference, bound) -> dict:
    import torch

    if not all(isinstance(t, torch.Tensor) for t in (actual, reference, bound)):
        raise TypeError("actual, reference and bound must be tensors")
    if actual.dtype != torch.bfloat16 or reference.dtype != torch.float32 or bound.dtype != torch.float64:
        raise ValueError("expected live BF16 output, fp32 reference and fp64 bound")
    if actual.device != reference.device or actual.device != bound.device:
        raise ValueError("oracle operands are on different devices")
    if actual.shape != reference.shape or actual.shape != bound.shape or actual.numel() == 0:
        raise ValueError("oracle operands must have equal nonempty shapes (no broadcasting)")
    with torch.no_grad():
        # Match the assertion's arithmetic: subtract in fp32, compare with the
        # existing fp64 elementwise bound. Do not replace it with a global max.
        error = (actual.detach().float() - reference.detach()).abs()
        live_bound = bound.detach()
        if not all(bool(torch.isfinite(t).all()) for t in (actual, reference, live_bound, error)):
            raise ValueError("nonfinite oracle operand/error")
        if bool((live_bound < 0).any()):
            raise ValueError("negative oracle bound")
        return {
            "shape": list(actual.shape), "numel": actual.numel(), "device": str(actual.device),
            "actual_dtype": str(actual.dtype), "reference_dtype": str(reference.dtype),
            "bound_dtype": str(bound.dtype), "finite": True,
            "max_abs_error": float(error.max()), "mean_abs_error": float(error.double().mean()),
            "bound_min": float(live_bound.min()), "bound_max": float(live_bound.max()),
            "max_violation": float((error - live_bound).max()),
            "violation_count": int((error > live_bound).sum()),
        }


def _measurement_error(measurement: dict, nodeid: str) -> str | None:
    """Validate a JSON-roundtrippable observation, including explicit CUDA evidence."""
    if measurement.get("schema") != SCHEMA or measurement.get("nodeid") != nodeid:
        return "missing/mismatched numerical schema or nodeid"
    if measurement.get("error"):
        return f"numerical observation failed: {measurement['error']}"
    if type(measurement.get("line")) is not int or measurement["line"] <= 0:
        return "missing assertion line"
    operands = _OPERANDS.get("".join(str(measurement.get("orig", "")).split()))
    if operands != (measurement.get("actual"), measurement.get("reference")) or operands is None:
        return "missing/mismatched numerical assertion operands"
    if measurement.get("bound") != "bound":
        return "missing live elementwise bound"
    device = measurement.get("device", "")
    if not (isinstance(device, str) and device.startswith("cuda:") and device[5:].isdigit()):
        return "numerical evidence is not from a CUDA device"
    if (measurement.get("actual_dtype"), measurement.get("reference_dtype"), measurement.get("bound_dtype")) != (
            "torch.bfloat16", "torch.float32", "torch.float64"):
        return "missing/mismatched oracle dtypes"
    shape = measurement.get("shape")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(n) is not int or n <= 0 for n in shape) or shape[1] != 64
            or type(measurement.get("numel")) is not int or measurement["numel"] != math.prod(shape)):
        return "missing/mismatched oracle shape/numel"
    fields = ("max_abs_error", "mean_abs_error", "bound_min", "bound_max", "max_violation")
    if measurement.get("finite") is not True or any(
            type(measurement.get(key)) is not float or not math.isfinite(measurement[key]) for key in fields):
        return "missing/nonfinite measured errors or bounds (floats required)"
    if not (0 <= measurement["mean_abs_error"] <= measurement["max_abs_error"]
            and 0 <= measurement["bound_min"] <= measurement["bound_max"]):
        return "inconsistent measured errors/bounds"
    # A max violation is signed: negative means every element is inside its bound.
    violation = measurement["max_violation"]
    if (type(measurement.get("violation_count")) is not int or measurement["violation_count"] != 0
            or violation > 0 or measurement["max_abs_error"] > measurement["bound_max"]):
        return "elementwise numerical bound violated"
    if not (measurement["max_abs_error"] - measurement["bound_max"] <= violation
            <= measurement["max_abs_error"] - measurement["bound_min"]):
        return "inconsistent maximum elementwise violation"
    return None


def bf16_oracle_errors(node_records, expected_nodeids) -> list[str]:
    """Return refusal reasons for required BF16 nodes in the explicit suite roster.

    Input records use native_probe's test schema, with each observation appended
    under assertions[*].numerical. Unrelated suite nodes are ignored. Missing,
    duplicate, dropped, failed, CPU-only or nonfinite observations cannot qualify.
    Keeping a list (not one measurement per node) preserves all 23 comparisons.
    """
    expected = set(expected_nodeids) & EXPECTED_COMPARISONS.keys()
    by_node = {nodeid: [] for nodeid in expected}
    for record in node_records:
        if record.get("nodeid") in by_node:
            by_node[record["nodeid"]].append(record)
    errors = []
    for nodeid in sorted(expected):
        records = by_node[nodeid]
        if len(records) != 1:
            errors.append(f"{nodeid}: expected one node record, got {len(records)}")
            continue
        record = records[0]
        if record.get("kind") != "test" or record.get("when") != "call" or record.get("outcome") != "passed":
            errors.append(f"{nodeid}: required node did not pass its call")
        if type(record.get("assertions_dropped")) is not int or record["assertions_dropped"] != 0:
            errors.append(f"{nodeid}: missing/dropped assertion evidence")
        assertions = record.get("assertions")
        if not isinstance(assertions, list):
            errors.append(f"{nodeid}: missing assertion records")
            continue
        seen = Counter()
        for assertion in assertions:
            if not isinstance(assertion, dict) or "numerical" not in assertion:
                continue
            measurement = assertion["numerical"]
            reason = (_measurement_error(measurement, nodeid) if isinstance(measurement, dict)
                      else "invalid numerical observation")
            if reason:
                errors.append(f"{nodeid}: {reason}")
                continue
            seen[(measurement["actual"], measurement["reference"], measurement["shape"][0])] += 1
        if seen != Counter(EXPECTED_COMPARISONS[nodeid]):
            errors.append(f"{nodeid}: missing/duplicate/unexpected numerical comparisons; "
                          f"expected {list(EXPECTED_COMPARISONS[nodeid])}, observed {list(seen.elements())}")
    return errors
