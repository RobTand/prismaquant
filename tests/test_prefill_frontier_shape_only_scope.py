"""A named scope that reads none of the fixed terms no gate admits.

`admit_fixed_resources` refuses every real table today (PQ debt D37: the
transient charge boundary is versioned now, and what still refuses is the four
observations the report consumer recomputes nothing from), so the prefill
frontier sweep -- which reaches the solver through three consumers of
`fixed_resources` -- has never emitted a curve from a real table.

`--measured-runtime-fixed-scope shape-only` is not a relaxation of that gate.
The gate still refuses, `fixed_resources_admitted` stays false everywhere it is
read, and the scope ADDS refusals on every path that would read what the gate
refused. These tests hold that line from both sides: the scope emits a curve
that is assignment-for-assignment the curve the same table produces with no
scope at all, and each refusal is demonstrated by mutating the driver -- the
budget, the declared term, the caller -- rather than by breaking the fixture.

The gate stand-ins are `monkeypatch`ed exactly as `test_runtime_admission_split`
does: what is under test here is the scope, not the gates, and the refusal text
is the real one those gates produce. Nothing here is GPU evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import allocator, prefill_frontier
from prismaquant.measured_runtime_prices import (
    PROVENANCE_CONTEXT_SCHEMA, PROVENANCE_IDENTITY_KIND, PROVENANCE_TABLE_SCHEMA,
    RuntimePriceError, RuntimeResources, SHAPE_ONLY_SCOPE, shape_only_fixed_resources,
)
from prismaquant.serve_constraints import (
    ServeConstraintError, ServeSLOs, evaluate_measured_assignment,
)
from test_allocator_measured_runtime_cli import _resources
from test_prefill_frontier import _curve_fixture

#: What `admit_fixed_resources` says about a v2 table that names no transient
#: charge boundary, wrapped in the gate's own preamble (D37: the boundary is
#: versioned now, and a table that carries no name is refused by this one).
D37 = ("no qualified recomputable full-engine resource partition: "
       "the table declares no transient charge boundary, so no candidate activation or "
       "scratch term may be compared to a priced row")


def _stand_in_for_the_gates(monkeypatch, *, fixed_refusal=D37):
    """Native rows admitted, fixed resources refused -- the state of every real table."""
    import prismaquant.runtime_provenance as rp

    monkeypatch.setattr(rp, "load_runtime_relation", lambda *a, **k: {"relation": True})
    monkeypatch.setattr(rp, "admit_native_rows", lambda table, relation: None)

    def fixed(table, relation):
        if fixed_refusal is not None:
            raise RuntimePriceError(fixed_refusal)

    monkeypatch.setattr(rp, "admit_fixed_resources", fixed)


def _promote_to_v2(tmp_path, **fixed_overrides):
    """Re-emit the curve fixture's table at the v2 schema, numbers unchanged."""
    context_path, table_path = tmp_path / "context.json", tmp_path / "runtime.json"
    context = json.loads(context_path.read_text())
    context["schema"] = PROVENANCE_CONTEXT_SCHEMA
    context["runtime_identity_kind"] = PROVENANCE_IDENTITY_KIND
    context_path.write_text(json.dumps(context))
    table = json.loads(table_path.read_text())
    table["schema"] = PROVENANCE_TABLE_SCHEMA
    table["context"] = context
    table["runtime_provenance"] = {"path": "relation.json", "sha256": "e" * 64}
    table["native_receipt_bindings"] = []
    table["fixed_resources"].update(fixed_overrides)
    table_path.write_text(json.dumps(table))


def _sweep(tmp_path, allocator_argv, *, scope=True, own=("--slo-grid", "auto")):
    output = tmp_path / "frontier.json"
    argv = [*own, "--output", str(output), "--",
            *allocator_argv,
            *(["--measured-runtime-fixed-scope", SHAPE_ONLY_SCOPE] if scope else [])]
    code = prefill_frontier.main(argv)
    return code, json.loads(output.read_text())


def _v2_sweep_fixture(tmp_path, monkeypatch, **fixed_overrides):
    _, allocator_argv = _curve_fixture(tmp_path)
    _promote_to_v2(tmp_path, **fixed_overrides)
    _stand_in_for_the_gates(monkeypatch)
    return allocator_argv


# --------------------------------------------------------------------------- #
# The success path
# --------------------------------------------------------------------------- #

def test_the_scope_emits_the_curve_and_says_what_it_did_not_price(tmp_path, monkeypatch):
    allocator_argv = _v2_sweep_fixture(tmp_path, monkeypatch)
    code, doc = _sweep(tmp_path, allocator_argv)

    assert code == 0 and doc["n_feasible"] == doc["n_points"] >= 3
    # The two scopes are stated separately and never mixed.
    assert doc["fixed_resources_admitted"] is False
    assert doc["certifies_placement"] is False and doc["certifies_p95"] is False
    scope = doc["fixed_resource_scope"]
    assert scope["scope"] == SHAPE_ONLY_SCOPE
    # The admission gate's own refusal, verbatim, on the document's face.
    assert scope["fixed_resources_refusal"] == D37
    assert scope["withheld_terms"] == ["activation_bytes", "kv_bytes",
                                       "non_step_transient_peak_bytes",
                                       "peak_scratch_bytes", "resident_bytes"]
    assert scope["read_terms"] == {"prefill_ms": 0, "decode_ms": 0, "serialized_bytes": 0}
    # No device number reaches the document, at any level.
    assert all(point["device_memory_bytes"] is None for point in doc["points"])
    for point in doc["points"]:
        constraints = point["serve_constraints"]
        assert constraints["predicted"]["device_memory_bytes"] is None
        assert constraints["fixed_resource_scope"] == SHAPE_ONLY_SCOPE
        assert constraints["coverage"]["memory"]["scope"].startswith("candidate_only")
        assert [check["constraint"] for check in constraints["checks"]] == ["operator_sum_prefill_ms"]


def test_the_scope_changes_no_chosen_assignment(tmp_path, monkeypatch):
    """The curve under the scope IS the curve without it, point for point.

    The claim the scope rests on is that nothing it withholds enters the
    answer: the fixed timing terms are zero and cancel on the SLO axis anyway,
    and every device filter in `allocator_solver` is guarded on
    `max_device_bytes is not None`, which no budget leaves unset. So the same
    table, priced once with its fixed charge admitted and once at v2 under
    the scope, must return the same assignments at the same budgets (PQ #560
    defect 3: pricing with no provenance at all is refused, so the plain arm
    carries admission too). Everything but the withheld device number is
    compared.
    """
    plain, scoped = tmp_path / "plain", tmp_path / "scoped"
    plain.mkdir(), scoped.mkdir()
    _, plain_argv = _curve_fixture(plain)
    _, scoped_argv = _curve_fixture(scoped)
    _promote_to_v2(plain)
    _promote_to_v2(scoped)
    _stand_in_for_the_gates(monkeypatch)
    _, under = _sweep(scoped, scoped_argv, scope=True)
    # The plain arm reads the fixed charge, so it needs the admission the
    # scope arm is defined by refusing; re-stub between the two sweeps.
    _stand_in_for_the_gates(monkeypatch, fixed_refusal=None)
    _, without = _sweep(plain, plain_argv, scope=False)

    compared = ("slo_ms", "feasible", "predicted_dloss", "achieved_bits", "payload_bytes",
                "attained_prefill_ms", "attained_decode_ms", "assignment_sha256", "nondominated")
    assert ([{key: point[key] for key in compared} for point in under["points"]]
            == [{key: point[key] for key in compared} for point in without["points"]])
    assert under["saturation"]["slo_ms"] == without["saturation"]["slo_ms"]
    assert under["slo_axis"]["lower_bound_ms"] == without["slo_axis"]["lower_bound_ms"]
    # ... and the one thing that does differ is the number with a charge missing.
    assert all(point["device_memory_bytes"] is not None for point in without["points"])
    assert all(point["device_memory_bytes"] is None for point in under["points"])
    assert without["fixed_resource_scope"] is None


# --------------------------------------------------------------------------- #
# The refusals, each demonstrated by mutating the driver
# --------------------------------------------------------------------------- #

def test_a_device_budget_refuses_the_scope(tmp_path, monkeypatch, capsys):
    """The one input that makes the solver's byte axes bind."""
    allocator_argv = _v2_sweep_fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        _sweep(tmp_path, [*allocator_argv, "--serve-device-budget-bytes", "1000000000"])
    assert "cannot evaluate --serve-device-budget-bytes" in capsys.readouterr().err


def test_a_declared_fixed_prefill_term_refuses_the_scope(tmp_path, monkeypatch):
    """The check that bites: the scope READS this term, so it re-runs the gate's rule.

    The fixture is unchanged apart from the one number the scope is allowed to
    read. A table that declares fixed prefill work has no evidence for it --
    the report schema observes no timing at all -- and the gate refuses it by
    name. So does the scope, rather than carrying it onto the SLO axis.
    """
    allocator_argv = _v2_sweep_fixture(tmp_path, monkeypatch, prefill_ms=4.0)
    with pytest.raises(SystemExit) as raised:
        _sweep(tmp_path, allocator_argv)
    assert "scope reads prefill_ms" in str(raised.value)
    assert "fixed prefill_ms (4.0) has no evidence" in str(raised.value)
    assert not (tmp_path / "frontier.json").exists()


def test_a_single_solve_refuses_the_scope(tmp_path, monkeypatch, capsys):
    """A lone run writes a layer config the export path reads; a sweep does not."""
    _, allocator_argv = _curve_fixture(tmp_path)
    _promote_to_v2(tmp_path)
    _stand_in_for_the_gates(monkeypatch)
    with pytest.raises(SystemExit):
        allocator.main([*allocator_argv, "--slo-prefill-p95-ttft-ms", "24",
                        "--measured-runtime-fixed-scope", SHAPE_ONLY_SCOPE])
    assert "available only to the prefill frontier sweep" in capsys.readouterr().err
    assert not (tmp_path / "layer.json").exists()


def test_without_the_flag_the_sweep_still_refuses(tmp_path, monkeypatch):
    """No silent change: the default path reads the fixed charge and is refused."""
    allocator_argv = _v2_sweep_fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimePriceError, match="require full-engine producer admission"):
        _sweep(tmp_path, allocator_argv, scope=False)
    assert not (tmp_path / "frontier.json").exists()


def test_the_scope_refuses_a_table_no_gate_ever_judged(tmp_path, monkeypatch, capsys):
    """A v1 table calls neither gate, so there is no refusal for the scope to narrow."""
    _, allocator_argv = _curve_fixture(tmp_path)          # left at v1 on purpose
    _stand_in_for_the_gates(monkeypatch)
    with pytest.raises(SystemExit) as raised:
        _sweep(tmp_path, allocator_argv)
    assert "narrows a refusal this table never received" in str(raised.value)


def test_the_scope_refuses_a_table_whose_fixed_charge_is_admitted(tmp_path, monkeypatch):
    """It must not degrade as the ledger improves: evidence is read, not withheld."""
    _, allocator_argv = _curve_fixture(tmp_path)
    _promote_to_v2(tmp_path)
    _stand_in_for_the_gates(monkeypatch, fixed_refusal=None)
    with pytest.raises(SystemExit) as raised:
        _sweep(tmp_path, allocator_argv)
    assert "its fixed resources are admitted, so read them" in str(raised.value)


# --------------------------------------------------------------------------- #
# The two consumers, refusing directly
# --------------------------------------------------------------------------- #

def test_the_evaluator_withholds_the_device_sum_under_a_scope():
    common = dict(option_assignments={("u", "A"): {"u": "A"}},
                  resources={("u", "A"): _resources(prefill_ms=3, decode_ms=2, resident_bytes=800,
                                                    activation_bytes=40, peak_scratch_bytes=20)},
                  fixed_assignment={}, fixed_resources=_resources(),
                  table_identity={"synthetic": True})
    verdict = evaluate_measured_assignment({"u": "A"}, slos=ServeSLOs(p95_ttft_ms=4),
                                           fixed_resource_scope=SHAPE_ONLY_SCOPE, **common)
    assert verdict.predicted["device_memory_bytes"] is None
    assert verdict.predicted["operator_sum_prefill_ms"] == 3
    # The candidate-side components survive, under a name that says what they are.
    assert verdict.coverage["memory"]["resident_bytes"] == 800
    assert "candidate_only" in verdict.coverage["memory"]["scope"]
    with pytest.raises(ServeConstraintError, match="can evaluate no device budget"):
        evaluate_measured_assignment({"u": "A"},
                                     slos=ServeSLOs(p95_ttft_ms=4, device_budget_bytes=10**9),
                                     fixed_resource_scope=SHAPE_ONLY_SCOPE, **common)


def test_a_point_that_carries_a_device_number_under_a_scope_is_refused(tmp_path):
    """Mutate the driver: hand the point builder a solve that priced a device.

    The withholding happens in the evaluator. This is the second reader
    refusing to trust it, because the whole content of the scope is that no
    device term reaches a document.
    """
    _, stamp = shape_only_fixed_resources(SimpleNamespace(
        runtime_provenance={"path": "p", "sha256": "0" * 64}, fixed_resources_admitted=False,
        fixed_resources_refusal=D37,
        fixed_resources=RuntimeResources(prefill_ms=0.0, decode_ms=0.0, serialized_bytes=0,
                                         resident_bytes=0, peak_scratch_bytes=0,
                                         activation_bytes=0)))
    record = {"slo_ms": 1.0, "target_bits": 4.0, "feasible": True, "assignment": {"u": "A"},
              "predicted_dloss": 1.0, "achieved_bits": 4.0, "payload_bytes": 8,
              "serve_constraints": {"predicted": {"operator_sum_prefill_ms": 1.0,
                                                  "operator_sum_decode_ms": None,
                                                  "device_memory_bytes": 4096}},
              "diagnostics": {}}
    with pytest.raises(prefill_frontier.PrefillFrontierError, match="may carry no device_memory_bytes"):
        prefill_frontier._point_record(record, Path(tmp_path), provenance_stub={},
                                       fixed_resource_scope=stamp)
    # Without the scope the same record is an ordinary point.
    assert prefill_frontier._point_record(record, Path(tmp_path),
                                          provenance_stub={})["device_memory_bytes"] == 4096
