"""Contracts for the versioned native/full-engine transient charge boundary (D37).

Every fixture here is synthetic and says so, except the last test, which reads
the 2026-09-18 full-engine report by reference. A positive fixture proves that
the boundary's own refusals are lifted and nothing else: the fixed charge is
still not admitted from any v1 fixture, because the four observations v1 owes
stay null, and the gate keeps naming them. Each negative test is a
discrimination: the refusal it names is absent from the agreeing baseline and
appears only when its property is broken.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import allocator_solver as solver
from prismaquant.full_engine_resource_report import (
    REPORT_SCHEMA_V2, consume_full_engine_resource_report, read_full_engine_resource_report,
)
from prismaquant.measured_runtime_prices import (
    RuntimePriceError, RuntimeResources, admitted_charge_boundary, parse_runtime_context,
)
from prismaquant.runtime_provenance import admit_fixed_resources
from prismaquant.serve_constraints import (
    ServeConstraintError, ServeSLOs, evaluate_measured_assignment,
)
from prismaquant.transient_charge_boundary import (
    BOUNDARIES, BOUNDARY_V1, FULL_ENGINE_PARTITION_SCHEMA, KV_CAPACITY_PIN_FIELDS,
    NATIVE_BOUND_COMPOSITION, require_boundary, reservation_slack_bytes,
)
from test_full_engine_resource_report import CLOSED_STEPS, closed_report, kv_record, written
from test_runtime_fixed_resource_admission import (
    CALIBRATION_SHA, CONFIGURATION_SHA, GPU_IDENTITY, PREFIX, RELATION, ROW_RESIDENT_BYTES,
    RUNTIME_MANIFEST_SHA, SOURCE_SHA, agreeing_resources, context_payload,
)

V1 = BOUNDARIES[BOUNDARY_V1]
NO_BOUNDARY = ("the table declares no transient charge boundary, so no candidate activation or "
               "scratch term may be compared to a priced row")
#: The two allocator readings of the 2026-09-18 capture, restated as the
#: fixture's reservation-slack witness.
ALLOCATED, RESERVED = 1_098_421_248, 1_201_668_096
#: Hand-computed from `CLOSED_ROWS` once the fixed scratch rows are moved out
#: of unit.a's interval: unit.a holds 700[11,13) + 800[12,16) + its 500 B
#: output [12,40) at once at index 12; unit.b holds 1200[30,34) + 900[31,36),
#: then 1300[34,38) + 900 after the free at 34 settles.
UNIT_PEAK = {"unit.a": 2000, "unit.b": 2200}
UNIT_OUTPUT = {"unit.a": 500, "unit.b": 900}
FIXED_IN_UNIT_A = ("0:2000:1", "0:3000:1", "0:4000:1")


def v1_report(*, selected=(("unit.a", "NVFP4"), ("unit.b", "NVFP4"))):
    """`closed_report` re-identified as the table's own run and made v1-clean.

    `CLOSED_ROWS` allocates its fixed scratch inside unit.a's interval, which
    v1 refuses by construction (check 1); here those rows are moved out of the
    interval and classified by the declared steps instead, so the fixture's
    numbers (`CLOSED_FIXED_SCRATCH` 5000, the per-unit scratch) are unchanged.
    """
    report = closed_report(steps=CLOSED_STEPS)
    identity = dict(report["identity"]["run"])
    identity.update(model_sha256=SOURCE_SHA, configuration_sha256=CONFIGURATION_SHA,
                    runtime_manifest_sha256=RUNTIME_MANIFEST_SHA, device_uuid=GPU_IDENTITY)
    report["identity"]["run"] = identity
    report["partition"]["identity"] = dict(identity)
    report["reference"]["selected_rows"] = [{"unit": unit, "format": fmt} for unit, fmt in selected]
    report["workload"]["calibration"] = {"sha256": CALIBRATION_SHA}
    for allocation in report["observations"]["torch_allocations"]:
        if allocation["allocation_id"] in FIXED_IN_UNIT_A:
            allocation.update(scope_stack=[], unit_invocation=None, lifetime_scope="outside_units")
    for row in report["partition"]["membership"]:
        if row["allocation_id"] in FIXED_IN_UNIT_A:
            row["unit"] = None
    policy = {"values": {"kv_cache_memory_bytes": 267_911_168, "num_gpu_blocks_override": None}}
    # `runtime_admission` False keeps `cache_capacity` open, exactly as the
    # baseline is; the route-class check reads only the capacity policy.
    report["observations"]["kv_observations"] = [kv_record(capacity_policy=policy,
                                                            runtime_admission=False)]
    report["observations"]["reservation_slack"] = {
        "allocated_bytes": ALLOCATED, "reserved_bytes": RESERVED,
        "sampled_at": "ready_for_workload",
        "scope": "torch.cuda.memory_allocated() and memory_reserved() at one instant"}
    return report


def row(unit, fmt, *, peak=None, output=True, route=None):
    resources = RuntimeResources(
        prefill_ms=1.0, decode_ms=0.5, serialized_bytes=2048, resident_bytes=ROW_RESIDENT_BYTES,
        peak_scratch_bytes=UNIT_PEAK[unit] if peak is None else peak, activation_bytes=500,
        kv_bytes=0, output_bytes={"prefill": UNIT_OUTPUT[unit]} if output else None)
    binding = None if route is False else SimpleNamespace(operator_route=route or f"route:{fmt}")
    return SimpleNamespace(unit=unit, fmt=fmt, resources=resources, binding=binding)


def table(tmp_path, report, *, boundary=BOUNDARY_V1, rows=None):
    reference = written(tmp_path, report, "report.json")
    receipt = written(tmp_path, {"full_model_resources": reference}, "fixed.json")
    payload = context_payload()
    if boundary is not None:
        payload["transient_charge_boundary"] = boundary
    return SimpleNamespace(
        source_path=str(tmp_path / "table.json"),
        fixed_resources_receipt_path=receipt["path"],
        fixed_resources_receipt_sha256=receipt["sha256"],
        context=parse_runtime_context(payload),
        rows=tuple(rows if rows is not None else (row("unit.a", "NVFP4"), row("unit.b", "NVFP4"))),
        fixed_resources=agreeing_resources(), fixed_assignment={})


def refusals(tmp_path, report=None, **table_kwargs):
    built = table(tmp_path, v1_report() if report is None else report, **table_kwargs)
    with pytest.raises(RuntimePriceError) as caught:
        admit_fixed_resources(built, RELATION)
    message = str(caught.value)
    assert message.startswith(PREFIX), message
    return message.removeprefix(PREFIX).split("; ")


def added(tmp_path, mutate=None, **table_kwargs):
    """The refusals a mutation adds over the v1-clean baseline."""
    baseline = refusals(tmp_path)
    report = v1_report()
    if mutate is not None:
        mutate(report)
    return [reason for reason in refusals(tmp_path, report, **table_kwargs)
            if reason not in baseline]


def boundary_scoped(reasons):
    """Refusals only a named boundary can produce."""
    markers = ("route class", "operator route binding", "capacity policy", "KV observation",
               "interval", "escapes", "output_bytes", "block extent", "held", "resident "
               "candidate bytes", "reservation_slack", "registered boundary",
               "partition declares schema")
    return [reason for reason in reasons if any(marker in reason for marker in markers)]


# --------------------------------------------------------------------------
# The registry and the decision it carries.
# --------------------------------------------------------------------------

def test_v1_pairs_the_two_producer_rules_and_charges_the_allocated_composition():
    assert V1.native_row["bound_composition"] == NATIVE_BOUND_COMPOSITION
    assert V1.full_engine["partition_schema"] == FULL_ENGINE_PARTITION_SCHEMA
    assert V1.row_terms_charged == ("resident_bytes", "peak_scratch_bytes")
    assert V1.row_terms_witnessed == ("activation_bytes",)
    assert V1.charges_row_activation is False
    assert V1.device_budget == {"search_charges": "allocated_block_composition",
                                "ship_gate_compares": "reserved_peak_bytes",
                                "witness": "reservation_slack_bytes"}
    assert V1.ownership["reservation_slack"] == "witnessed_at_ship_gate"
    assert tuple(V1.kv_capacity_pin["pinned_when_any_of"]) == KV_CAPACITY_PIN_FIELDS
    assert V1.as_dict()["kv_capacity_pin"]["pinned_when_any_of"] == list(KV_CAPACITY_PIN_FIELDS)
    # The registry is data a gate compares verbatim, so it serializes as such.
    assert json.loads(json.dumps(V1.as_dict()))["schema"] == BOUNDARY_V1
    with pytest.raises(TypeError):
        BOUNDARIES["other"] = V1


@pytest.mark.parametrize("table_name,partition_schema,report_name,expected", [
    (None, FULL_ENGINE_PARTITION_SCHEMA, None, NO_BOUNDARY),
    ("prismaquant.transient_charge_boundary.v0", FULL_ENGINE_PARTITION_SCHEMA, None,
     "is not a registered boundary"),
    (BOUNDARY_V1, "tessera.full_engine_resource_partition.v2", None,
     "the partition declares schema 'tessera.full_engine_resource_partition.v2'"),
    (BOUNDARY_V1, FULL_ENGINE_PARTITION_SCHEMA, "prismaquant.transient_charge_boundary.v0",
     "one assignment is priced under one owner map"),
])
def test_require_boundary_names_each_side_that_is_missing_or_differs(
        table_name, partition_schema, report_name, expected):
    with pytest.raises(RuntimePriceError, match=expected):
        require_boundary(table_name, partition_schema=partition_schema, report_boundary=report_name)


def test_require_boundary_returns_the_registered_spec():
    spec = require_boundary(BOUNDARY_V1, partition_schema=FULL_ENGINE_PARTITION_SCHEMA)
    assert spec is V1
    assert require_boundary(BOUNDARY_V1, partition_schema=FULL_ENGINE_PARTITION_SCHEMA,
                            report_boundary=BOUNDARY_V1) is V1


# --------------------------------------------------------------------------
# The gate: an unversioned table refuses by name, a versioned one is checked.
# --------------------------------------------------------------------------

def test_an_unversioned_table_refuses_by_name_and_runs_no_boundary_check(tmp_path):
    reasons = refusals(tmp_path, boundary=None)
    assert NO_BOUNDARY in reasons
    assert boundary_scoped(reasons) == []


def test_a_versioned_table_lifts_exactly_the_boundary_refusal(tmp_path):
    """Under v1 the identity, route-class and witness checks run and find
    nothing on the v1-clean fixture; what remains is what the producer still
    owes (the null observations), untouched by the boundary."""
    unversioned = refusals(tmp_path, boundary=None)
    versioned = refusals(tmp_path)
    assert NO_BOUNDARY not in versioned
    assert boundary_scoped(versioned) == []
    assert sorted(versioned) == sorted(reason for reason in unversioned if reason != NO_BOUNDARY)
    assert any("the capture observes no timing_captures" in reason for reason in versioned)


def test_an_unregistered_boundary_refuses_and_admits_nothing(tmp_path):
    reasons = refusals(tmp_path, boundary="prismaquant.transient_charge_boundary.v0")
    assert any("is not a registered boundary" in reason for reason in reasons)
    assert NO_BOUNDARY not in reasons


def test_a_multi_format_pact_table_is_admitted_under_route_class_scope(tmp_path):
    """One measured assignment prices every assignment in the route classes it
    exercised: unit.a ran NVFP4 and unit.b ran FP8_DYNAMIC, so a menu of both
    formats for both units is inside the scope, and the retired sentence about
    "more than one format" never appears."""
    rows = [row("unit.a", "NVFP4"), row("unit.a", "FP8_DYNAMIC"),
            row("unit.b", "NVFP4"), row("unit.b", "FP8_DYNAMIC")]
    report = v1_report(selected=(("unit.a", "NVFP4"), ("unit.b", "FP8_DYNAMIC")))
    reasons = refusals(tmp_path, report, rows=rows)
    assert not any("prices more than one format" in reason for reason in reasons)
    assert not any("route class" in reason for reason in reasons)


def test_a_route_class_the_run_did_not_exercise_refuses_with_its_units(tmp_path):
    rows = [row("unit.a", "NVFP4"), row("unit.a", "FP8_DYNAMIC"),
            row("unit.b", "NVFP4"), row("unit.b", "FP8_DYNAMIC")]
    assert ("route class 'route:FP8_DYNAMIC' is priced for ['unit.a@FP8_DYNAMIC', "
            "'unit.b@FP8_DYNAMIC'] and the full-engine run exercised no row of that class, so "
            "the fixed charge measured there does not transfer to it") in added(tmp_path, rows=rows)


def test_a_row_without_a_route_binding_refuses(tmp_path):
    rows = [row("unit.a", "NVFP4", route=False), row("unit.b", "NVFP4")]
    assert ("row ('unit.a', 'NVFP4') carries no operator route binding, so its route class "
            "cannot be checked against the full-engine run") in added(tmp_path, rows=rows)


def test_an_unpinned_kv_capacity_refuses(tmp_path):
    def unpin(report):
        record = report["observations"]["kv_observations"][0]
        record["capacity_policy"] = {"values": {name: None for name in KV_CAPACITY_PIN_FIELDS}}
    reasons = added(tmp_path, unpin)
    assert any("pins no capacity" in reason and "fixed_kv is sized from free memory" in reason
               for reason in reasons)

    def drop(report):
        report["observations"]["kv_observations"] = None
    assert any("carries no KV observation" in reason for reason in added(tmp_path, drop))


# --------------------------------------------------------------------------
# "Equal" under v1: the five identity checks, each a discrimination.
# --------------------------------------------------------------------------

def test_a_fixed_owned_allocation_inside_a_unit_interval_refuses(tmp_path):
    def restore(report):
        for allocation in report["observations"]["torch_allocations"]:
            if allocation["allocation_id"] == "0:2000:1":
                allocation.update(scope_stack=["unit.a"], unit_invocation="unit.a:0",
                                  lifetime_scope="inside_unit")
        for member in report["partition"]["membership"]:
            if member["allocation_id"] == "0:2000:1":
                member["unit"] = "unit.a"
    assert any(reason.startswith("fixed-owned allocation 0:2000:1 (1000 B) was made inside the "
                                 "interval of unit 'unit.a'") for reason in added(tmp_path, restore))


def test_a_candidate_allocation_attributed_across_intervals_refuses(tmp_path):
    def reattribute(report):
        for member in report["partition"]["membership"]:
            if member["allocation_id"] == "0:5000:1":
                member["unit"] = "unit.b"
    reasons = added(tmp_path, reattribute)
    assert ("candidate-owned allocation 0:5000:1 was made inside the interval of unit 'unit.a' "
            "and is attributed to 'unit.b'") in reasons
    assert ("candidate-owned scratch allocation 0:5000:1 is attributed to unit 'unit.b' and was "
            "allocated outside its interval") in reasons


def test_two_escaping_allocations_refuse(tmp_path):
    def second_escape(report):
        for allocation in report["observations"]["torch_allocations"]:
            if allocation["allocation_id"] == "0:6000:1":
                allocation["lifetime_scope"] = "escapes_unit"
        for member in report["partition"]["membership"]:
            if member["allocation_id"] == "0:6000:1":
                member["lifetime_class"] = "activation"
    assert any(reason.startswith("invocation unit.a:0 escapes 2 allocations")
               for reason in added(tmp_path, second_escape))


def test_an_escape_of_another_size_than_the_returned_output_refuses(tmp_path):
    rows = [row("unit.a", "NVFP4"), row("unit.b", "NVFP4")]
    rows[0].resources = RuntimeResources(**(rows[0].resources.as_dict() | {"output_bytes": {"prefill": 501}}))
    assert ("invocation unit.a:0 escapes allocation 0:9000:1 of 500 B where the row's returned "
            "output is {'prefill': 501} B") in added(tmp_path, rows=rows)


def test_a_row_without_output_bytes_cannot_identify_its_escape(tmp_path):
    rows = [row("unit.a", "NVFP4", output=False), row("unit.b", "NVFP4")]
    assert ("the priced row ('unit.a', 'NVFP4') carries no output_bytes, so the allocation "
            "escaping invocation unit.a:0 cannot be identified as its returned output"
            ) in added(tmp_path, rows=rows)


def test_an_engine_that_exceeds_the_priced_cover_refuses(tmp_path):
    rows = [row("unit.a", "NVFP4", peak=UNIT_PEAK["unit.a"] - 1), row("unit.b", "NVFP4")]
    assert ("the engine held 2000 B of the allocations of unit 'unit.a' at once in invocation "
            "unit.a:0 where the priced row bounds 1999 B -- the native bound does not cover what "
            "the engine did") in added(tmp_path, rows=rows)
    # A row that exceeds the engine is disclosed conservatism, not a refusal.
    assert not any("held" in reason for reason in
                   added(tmp_path, rows=[row("unit.a", "NVFP4", peak=10 ** 6), row("unit.b", "NVFP4")]))


def test_an_unobserved_block_extent_refuses_the_cover_check_per_invocation(tmp_path):
    def unobserved(report):
        for allocation in report["observations"]["torch_allocations"]:
            if allocation["allocation_id"] in ("0:5000:1", "0:6000:1"):
                allocation["allocator_block_bytes_observed"] = []
    assert ("2 transient allocations of unit 'unit.a' inside invocation unit.a:0 observed no "
            "allocator block extent, so the cover check has no rounded extent to sum"
            ) in added(tmp_path, unobserved)


def test_resident_candidate_bytes_above_the_priced_row_are_a_workspace(tmp_path):
    def workspace(report):
        report["observations"]["torch_allocations"].append({
            "address": 9300, "allocate_index": 13, "allocation_id": "0:9300:1",
            "allocator_block_bytes_observed": [100], "bytes": 100, "free_completed_index": None,
            "free_requested_index": None, "generation": 1, "lifetime_scope": "inside_unit",
            "observed_categories": ["candidate"], "observed_owners": ["owner.workspace"],
            "scope_stack": ["unit.a"], "unit_invocation": "unit.a:0"})
        report["partition"]["membership"].append({
            "allocate_index": 13, "allocation_id": "0:9300:1", "bytes": 100,
            "free_completed_index": None, "lifetime_class": "resident",
            "owner_class": "candidate", "unit": "unit.a"})
    rows = [row("unit.a", "NVFP4"), row("unit.b", "NVFP4")]
    rows[0].resources = RuntimeResources(**(rows[0].resources.as_dict() | {"resident_bytes": 0}))
    assert any(reason.startswith("unit 'unit.a' holds 100 B of resident candidate bytes the priced "
                                 "row does not price (allocations inside its interval: "
                                 "['0:9300:1'])") for reason in added(tmp_path, workspace, rows=rows))


def test_a_capture_without_the_reserved_witness_refuses_by_name(tmp_path):
    def drop(report):
        del report["observations"]["reservation_slack"]
    assert ("the capture observes no reservation_slack, so the reserved-extent witness "
            "reservation_slack_bytes has no evidence") in added(tmp_path, drop)
    assert reservation_slack_bytes(v1_report()) == RESERVED - ALLOCATED == 103_246_848


def test_the_reader_refuses_a_slack_below_zero(tmp_path):
    report = v1_report()
    report["observations"]["reservation_slack"]["reserved_bytes"] = ALLOCATED - 1
    reference = written(tmp_path, report)
    with pytest.raises(RuntimePriceError, match="reserved bytes .* are below allocated bytes"):
        read_full_engine_resource_report(reference, root=tmp_path)


# --------------------------------------------------------------------------
# The search and the evaluator: one constant added once, activation a witness.
# --------------------------------------------------------------------------

def _candidate(fmt, memory, loss):
    return solver.Candidate(fmt, memory * 8 / 1000, memory, loss)


def _price(c, *, resident, scratch, activation):
    return SimpleNamespace(serialized_bytes=c.memory_bytes, prefill_ms=1.0, decode_ms=1.0,
                           resident_bytes=resident, peak_scratch_bytes=scratch,
                           activation_bytes=activation)


def test_the_search_refuses_a_fixed_charge_with_no_boundary():
    c = _candidate("A", 1, 0.0)
    resources = {("u", "A"): _price(c, resident=1, scratch=1, activation=1)}
    with pytest.raises(ValueError, match="no transient charge boundary"):
        solver.solve_runtime_frontier({"u": [c]}, resources, max_memory_bytes=1,
                                      max_prefill_ms=1, max_device_bytes=10, fixed_device_bytes=1)
    with pytest.raises(ValueError, match="no transient charge boundary"):
        solver.solve_runtime_frontier({"u": [c]}, resources, max_memory_bytes=1,
                                      max_prefill_ms=1, max_device_bytes=10,
                                      fixed_non_step_peak_bytes=1)


def test_the_search_charges_no_row_activation_under_v1_and_says_so():
    c = _candidate("A", 1, 0.0)
    resources = {("u", "A"): _price(c, resident=4, scratch=3, activation=100)}
    diag = {}
    kwargs = dict(max_memory_bytes=1, max_prefill_ms=1, fixed_device_bytes=2, boundary=V1,
                  diagnostics=diag)
    result = solver.solve_runtime_frontier({"u": [c]}, resources, max_device_bytes=9, **kwargs)
    assert len(result) == 1
    assert result[0].device_bytes == 4 + 3 + 2
    assert result[0].activation_bytes == 100
    assert diag["transient_charge_boundary"] == BOUNDARY_V1
    assert diag["row_activation_charged"] is False
    # The witness is still tracked: the same budget refuses once resident
    # bytes, and not activation bytes, push the composition over it.
    assert solver.solve_runtime_frontier({"u": [c]}, resources, max_device_bytes=8, **kwargs) == []


def _resources(**updates):
    return SimpleNamespace(**(dict(prefill_ms=0.0, decode_ms=0.0, serialized_bytes=0,
                                   resident_bytes=0, activation_bytes=0, peak_scratch_bytes=0,
                                   kv_bytes=0) | updates))


def _evaluate(**kwargs):
    return evaluate_measured_assignment(
        {"q": "A", "head": "BF16"},
        option_assignments={("q", "A"): {"q": "A"}},
        resources={("q", "A"): _resources(resident_bytes=800, activation_bytes=40,
                                          peak_scratch_bytes=20)},
        fixed_assignment={"head": "BF16"},
        fixed_resources=_resources(resident_bytes=500, activation_bytes=10, peak_scratch_bytes=30,
                                   kv_bytes=50),
        slos=ServeSLOs(device_budget_bytes=1400), table_identity={"synthetic": True}, **kwargs)


def test_the_evaluator_refuses_a_fixed_charge_with_no_boundary():
    with pytest.raises(ServeConstraintError, match="no transient charge boundary"):
        _evaluate()


def test_the_evaluator_witnesses_row_activation_under_v1():
    verdict = _evaluate(boundary=V1)
    # resident 800 + 500, scratch 30 + 20, fixed activation 10, kv 50; the
    # row's 40 activation bytes are witnessed, not added.
    assert verdict.predicted["device_memory_bytes"] == 1300 + 50 + 10 + 50
    memory = verdict.coverage["memory"]
    assert memory["row_activation_bytes_witness"] == 40
    assert memory["row_activation_charged"] is False
    assert memory["transient_charge_boundary"] == BOUNDARY_V1


# --------------------------------------------------------------------------
# The name travels: context and row resources round-trip it.
# --------------------------------------------------------------------------

def test_the_context_carries_the_boundary_name_only_when_set():
    payload = context_payload()
    assert "transient_charge_boundary" not in parse_runtime_context(payload).as_dict()
    payload["transient_charge_boundary"] = BOUNDARY_V1
    context = parse_runtime_context(payload)
    assert context.transient_charge_boundary == BOUNDARY_V1
    assert parse_runtime_context(context.as_dict()) == context
    payload["transient_charge_boundary"] = None
    with pytest.raises(RuntimePriceError):
        parse_runtime_context(payload)


def test_admitted_charge_boundary_reads_the_table_context(tmp_path):
    def stub(boundary):
        payload = context_payload()
        if boundary is not None:
            payload["transient_charge_boundary"] = boundary
        return SimpleNamespace(context=parse_runtime_context(payload), fixed_resources=None,
                               fixed_resources_receipt_path=None, rows=())
    import prismaquant.measured_runtime_prices as prices
    original = prices.admitted_fixed_resources
    prices.admitted_fixed_resources = lambda table: None
    try:
        assert admitted_charge_boundary(stub(BOUNDARY_V1)) is V1
        with pytest.raises(RuntimePriceError, match=NO_BOUNDARY.split(",")[0]):
            admitted_charge_boundary(stub(None))
        with pytest.raises(RuntimePriceError, match="not a registered boundary"):
            admitted_charge_boundary(stub("prismaquant.transient_charge_boundary.v0"))
    finally:
        prices.admitted_fixed_resources = original


def test_row_resources_round_trip_the_returned_output():
    resources = RuntimeResources(prefill_ms=1.0, decode_ms=0.5, serialized_bytes=1,
                                 resident_bytes=2, peak_scratch_bytes=3, activation_bytes=4,
                                 output_bytes={"prefill": 500, "decode": 8})
    assert RuntimeResources.from_dict(resources.as_dict()) == resources
    assert "output_bytes" not in RuntimeResources(prefill_ms=1.0, decode_ms=0.5, serialized_bytes=1,
                                                  resident_bytes=2, peak_scratch_bytes=3,
                                                  activation_bytes=4).as_dict()
    with pytest.raises(RuntimePriceError):
        RuntimeResources(prefill_ms=1.0, decode_ms=0.5, serialized_bytes=1, resident_bytes=2,
                         peak_scratch_bytes=3, activation_bytes=4, output_bytes={})


# --------------------------------------------------------------------------
# The real report, by reference: v2 reads, the versioned boundary is admitted,
# the unversioned one still refuses, and what remains is named.
# --------------------------------------------------------------------------

REAL_REPORT_ROOT = Path("/mnt/shared/tessera-runs/receipts/frontier-qwen3-0.6b-20260918-399/"
                        "report-resources")
REAL_FORMATS = ("TESSERA_BF16", "TESSERA_FP8", "TESSERA_NVFP4")


def _real_reference():
    path = REAL_REPORT_ROOT / "report.json"
    if not path.exists():
        pytest.skip(f"the 2026-09-18 full-engine report is not mounted at {path}; this test "
                    "certifies nothing when skipped")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture(scope="module")
def real_report():
    reference = _real_reference()
    return reference, read_full_engine_resource_report(reference, root=REAL_REPORT_ROOT)


def _real_table(report, boundary, rows):
    identity = report["identity"]["run"]
    payload = context_payload(gpu_identity=identity["device_uuid"],
                              source_sha256=identity["model_sha256"],
                              calibration_sha256=report["workload"]["calibration"]["sha256"])
    if boundary is not None:
        payload["transient_charge_boundary"] = boundary

    def resources():
        return RuntimeResources(prefill_ms=1.0, decode_ms=0.5, serialized_bytes=0,
                                resident_bytes=0, peak_scratch_bytes=0, activation_bytes=0)
    return SimpleNamespace(
        source_path="synthetic-table-over-the-real-report.json",
        fixed_resources_receipt_path=str(REAL_REPORT_ROOT / "unused.json"),
        fixed_resources_receipt_sha256="0" * 64,
        context=parse_runtime_context(payload),
        rows=tuple(SimpleNamespace(unit=unit, fmt=fmt, resources=resources(),
                                   binding=SimpleNamespace(operator_route=f"route:{fmt}"))
                   for unit, fmt in rows),
        fixed_resources=RuntimeResources(prefill_ms=0.0, decode_ms=0.0, serialized_bytes=0,
                                         resident_bytes=0, peak_scratch_bytes=0,
                                         activation_bytes=0),
        fixed_assignment={})


def _real_refusals(reference, report, boundary, rows):
    import prismaquant.runtime_provenance as rp
    identity = report["identity"]["run"]
    relation = {"runs": {"engine": {"sha256": identity["runtime_manifest_sha256"]}},
                "full_engine_run_id": "engine",
                "configuration_sha256": identity["configuration_sha256"]}
    return rp._fixed_resource_refusals(_real_table(report, boundary, rows), relation, reference,
                                       root=REAL_REPORT_ROOT)


def test_the_real_v2_report_reads_and_names_what_this_consumer_does_not_recompute(real_report):
    reference, report = real_report
    assert report["schema"] == REPORT_SCHEMA_V2
    assert report["partition"]["schema"] == FULL_ENGINE_PARTITION_SCHEMA
    verdict = consume_full_engine_resource_report(reference, root=REAL_REPORT_ROOT)
    assert verdict.schema == REPORT_SCHEMA_V2
    assert "worker_startup" in verdict.open_domains
    for name in ("provenance_admission", "timing_partition"):
        assert any(f"domain {name} is closed by the producer on" in reason
                   and "recomputes nothing from" in reason for reason in verdict.blocking), name
    assert any(reason.startswith("the partition classifies on owner_views (rules [")
               for reason in verdict.blocking)


def test_the_real_report_admits_the_versioned_boundary_and_refuses_the_unversioned(real_report):
    reference, report = real_report
    selected = [(item["unit"], item["format"]) for item in report["reference"]["selected_rows"]]
    assert sorted({fmt for _, fmt in selected}) == list(REAL_FORMATS)
    unversioned = _real_refusals(reference, report, None, selected)
    versioned = _real_refusals(reference, report, BOUNDARY_V1, selected)
    assert NO_BOUNDARY in unversioned and NO_BOUNDARY not in versioned
    assert not any("registered boundary" in r or "partition declares schema" in r for r in versioned)
    assert set(unversioned) - set(versioned) == {NO_BOUNDARY}
    only_versioned = set(versioned) - set(unversioned)
    # Everything the boundary adds on this capture names what is still owed:
    # rows priced with no returned output, in-unit transients whose block
    # extent was not sampled, the absent reserved-extent witness, and -- since
    # these synthetic rows price zero resident bytes -- every unit's resident
    # candidate bytes named as unpriced.
    assert only_versioned, "the boundary checks ran"
    owed = ("carries no output_bytes", "observed no allocator block extent", "reservation_slack",
            "resident candidate bytes the priced row does not price")
    unexpected = [r for r in only_versioned if not any(marker in r for marker in owed)]
    assert unexpected == [], unexpected[:5]
    assert any("observed no allocator block extent" in r for r in only_versioned)
    assert any("reservation_slack" in r for r in only_versioned)
    assert not any("route class" in r for r in versioned)
    # The producer's own refusals survive the boundary untouched, and the v2
    # observations are named as carried rather than read.
    assert any("domain worker_startup is not closed" in r for r in versioned)
    assert any(r.startswith("the capture carries owner_views, and this consumer recomputes")
               for r in versioned)


def test_the_real_report_prices_every_pact_format_in_its_route_classes(real_report):
    reference, report = real_report
    units = [item["unit"] for item in report["reference"]["selected_rows"]]
    rows = [(unit, fmt) for unit in units for fmt in REAL_FORMATS]
    reasons = _real_refusals(reference, report, BOUNDARY_V1, rows)
    assert not any("route class" in r or "prices more than one format" in r for r in reasons)
    narrowed = _real_refusals(reference, report, BOUNDARY_V1,
                              [(unit, fmt) for unit in units for fmt in REAL_FORMATS
                               if fmt != "TESSERA_BF16"])
    assert not any("route class" in r for r in narrowed)
