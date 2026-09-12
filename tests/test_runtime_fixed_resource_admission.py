"""Contracts for the fixed-resource admission gate, `admit_fixed_resources`.

Every fixture here is synthetic and says so. A positive fixture proves the
recomputation and agreement contract only; it establishes no measurement, and
this gate admits nothing at the producer's current schema version -- what it
refuses is derived rather than unconditional, which is the whole change.

Each test below is a discrimination, not an assertion that a refusal exists:
it names a refusal the agreeing baseline does not carry and shows it appears
only when its property is broken. A check that cannot fail is not a check, so
the refusals this schema version can never lift -- the absent timing
partition, the unversioned native/full-engine charge boundary, the fixed
member roster the partition does not name -- are asserted as *disclosures* in
one test that says so, never dressed up as checks.
"""
import copy
from types import SimpleNamespace

import pytest

from prismaquant.measured_runtime_prices import (
    RuntimePriceError, RuntimeResources, parse_runtime_context,
)
from prismaquant.runtime_provenance import admit_fixed_resources
from test_full_engine_resource_report import (
    CLOSED_CANDIDATE_SCRATCH, CLOSED_FIXED_SCRATCH, CLOSED_UNITS, closed_report, written,
)

PREFIX = "no qualified recomputable full-engine resource partition: "
SOURCE_SHA = "c" * 64
CALIBRATION_SHA = "d" * 64
CONFIGURATION_SHA = "a" * 64
RUNTIME_MANIFEST_SHA = "e" * 64
GPU_IDENTITY = "synthetic-gpu"

#: What `load_runtime_relation` returns, reduced to the three members this gate
#: reads. No run is handed another run's digest: the full-engine manifest below
#: is the full-engine run's own.
RELATION = {"runs": {"engine": {"sha256": RUNTIME_MANIFEST_SHA},
                     "dense": {"sha256": "f" * 64}},
            "full_engine_run_id": "engine",
            "configuration_sha256": CONFIGURATION_SHA}

ROW_RESIDENT_BYTES = 1600


def context_payload(**overrides):
    serving = {"platform": "sm_121", "structure": "dense", "residency": "resident",
               "runtime_image": "example.invalid/runtime@sha256:" + "a" * 64,
               "execution_mode": "eager"}
    serving.update(overrides.pop("serving_context", {}))
    payload = {
        "schema": "prismaquant.measured_runtime_context.v2",
        "runtime_identity_kind": "prismaquant.runtime_provenance_relation.v1",
        "serving_context": serving, "gpu_identity": GPU_IDENTITY,
        "runtime_sha256": "b" * 64, "source_sha256": SOURCE_SHA,
        "calibration_sha256": CALIBRATION_SHA, "prompt_tokens": 512, "batch_size": 1,
        "tensor_parallel": 1, "graph_mode": "eager",
        "operator_routes": {"layer": {"FP8": "synthetic"}},
    }
    payload.update(overrides)
    return payload


def agreeing_report():
    """The arithmetic fixture, re-identified as this table's own measurement.

    It keeps its `fixture_provenance`, so every baseline below carries the
    gate's synthetic refusal; a constant refusal masks nothing, because the
    tests compare against that same baseline.
    """
    report = closed_report()
    identity = dict(report["identity"]["run"])
    identity.update(model_sha256=SOURCE_SHA, configuration_sha256=CONFIGURATION_SHA,
                    runtime_manifest_sha256=RUNTIME_MANIFEST_SHA, device_uuid=GPU_IDENTITY)
    report["identity"]["run"] = identity
    report["partition"]["identity"] = dict(identity)
    report["reference"]["selected_rows"] = [{"unit": unit, "format": "NVFP4"}
                                            for unit in CLOSED_UNITS]
    report["workload"]["calibration"] = {"sha256": CALIBRATION_SHA}
    return report


def agreeing_resources(**overrides):
    """A fixed charge whose one recomputable field agrees with the report."""
    fields = {"prefill_ms": 12.5, "decode_ms": 0.75, "serialized_bytes": 4096,
              "resident_bytes": 8192, "peak_scratch_bytes": CLOSED_FIXED_SCRATCH,
              "activation_bytes": 1024, "kv_bytes": 2048}
    fields.update(overrides)
    return RuntimeResources(**fields)


def build_table(tmp_path, report, *, context=None, rows=None, fixed_resources=None,
                fixed_assignment=None):
    reference = written(tmp_path, report, "report.json")
    receipt = written(tmp_path, {"full_model_resources": reference}, "fixed.json")
    if rows is None:
        rows = [(unit, "NVFP4") for unit in CLOSED_UNITS]
    return SimpleNamespace(
        source_path=str(tmp_path / "table.json"),
        fixed_resources_receipt_path=receipt["path"],
        fixed_resources_receipt_sha256=receipt["sha256"],
        context=parse_runtime_context(context or context_payload()),
        rows=tuple(SimpleNamespace(unit=unit, fmt=fmt,
                                   resources=RuntimeResources(
                                       prefill_ms=1.0, decode_ms=0.5, serialized_bytes=2048,
                                       resident_bytes=ROW_RESIDENT_BYTES, peak_scratch_bytes=1500,
                                       activation_bytes=500, kv_bytes=0))
                   for unit, fmt in rows),
        fixed_resources=fixed_resources or agreeing_resources(),
        fixed_assignment=fixed_assignment or {"model.embed_tokens": "BF16"})


def refusals(tmp_path, report=None, *, relation=None, **table_kwargs):
    """Run the gate and return every refusal it named, in order."""
    table = build_table(tmp_path, agreeing_report() if report is None else report,
                        **table_kwargs)
    with pytest.raises(RuntimePriceError) as caught:
        admit_fixed_resources(table, RELATION if relation is None else relation)
    message = str(caught.value)
    assert message.startswith(PREFIX), message
    return message.removeprefix(PREFIX).split("; ")


def added(tmp_path, mutate, **table_kwargs):
    """The refusals a report mutation adds, and nothing the baseline carries."""
    baseline = refusals(tmp_path)
    report = agreeing_report()
    mutate(report)
    return [reason for reason in refusals(tmp_path, report, **table_kwargs)
            if reason not in baseline]


# --------------------------------------------------------------------------
# The gate reads the recomputation, never the producer's `derived` block.
# --------------------------------------------------------------------------

def test_the_table_is_checked_against_the_recomputation_and_not_against_derived(tmp_path):
    """The keystone. A producer that restates the table's own wrong number in
    `derived` still refuses, because the comparison is against what this
    consumer recomputes from `observations` and `partition`. A gate that read
    `derived` would admit this exact pair."""
    wrong = CLOSED_FIXED_SCRATCH + 1000
    report = agreeing_report()
    assert report["derived"]["terms"]["fixed_scratch"] == CLOSED_FIXED_SCRATCH
    report["derived"]["terms"]["fixed_scratch"] = wrong
    reasons = refusals(tmp_path, report,
                       fixed_resources=agreeing_resources(peak_scratch_bytes=wrong))
    assert (f"this table declares fixed peak_scratch_bytes {wrong} where the recomputed "
            f"fixed_scratch is {CLOSED_FIXED_SCRATCH}") in reasons
    assert any(f"derived term fixed_scratch is {wrong}" in reason for reason in reasons)


def test_the_one_recomputable_term_agrees_on_the_baseline(tmp_path):
    """Without it the disagreement test above could pass on a gate that
    refuses every table, agreeing or not."""
    reasons = refusals(tmp_path)
    assert not [reason for reason in reasons if "peak_scratch_bytes" in reason]
    assert not [reason for reason in reasons if "recomputed fixed_scratch" in reason]


@pytest.mark.parametrize("declared", [0, CLOSED_FIXED_SCRATCH - 1, CLOSED_FIXED_SCRATCH + 1,
                                      sum(CLOSED_CANDIDATE_SCRATCH.values())])
def test_a_declared_fixed_scratch_that_differs_refuses(tmp_path, declared):
    """Including the two numbers a wrong composition would most plausibly
    produce: zero, and the candidate charge summed instead of maximised."""
    reasons = refusals(tmp_path, fixed_resources=agreeing_resources(peak_scratch_bytes=declared))
    assert (f"this table declares fixed peak_scratch_bytes {declared} where the recomputed "
            f"fixed_scratch is {CLOSED_FIXED_SCRATCH}") in reasons


@pytest.mark.parametrize("term,field", [("fixed_resident", "resident_bytes"),
                                        ("fixed_activation", "activation_bytes"),
                                        ("fixed_kv", "kv_bytes")])
def test_an_unrecomputable_term_blocks_rather_than_disagreeing(tmp_path, term, field):
    """Absence of evidence is not a disagreement. A null term compared as a
    number would read `None != 8192` as a producer error and hide that the
    schema version simply cannot express it."""
    reasons = refusals(tmp_path)
    value = getattr(agreeing_resources(), field)
    assert f"no {term} is recomputable, so this table's fixed {field} ({value}) has no evidence" in reasons
    assert not [reason for reason in reasons if f"recomputed {term} is" in reason]


# --------------------------------------------------------------------------
# Stale identity: the report was measured on other bytes.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["model_sha256", "configuration_sha256",
                                 "runtime_manifest_sha256"])
def test_a_stale_source_configuration_or_runtime_refuses(tmp_path, key):
    def mutate(report):
        report["identity"]["run"][key] = "9" * 64
        report["partition"]["identity"][key] = "9" * 64
    assert added(tmp_path, mutate) == [f"stale {key}: the report was measured on other bytes"]


def test_a_foreign_device_refuses(tmp_path):
    def mutate(report):
        report["identity"]["run"]["device_uuid"] = "another-gpu"
        report["partition"]["identity"]["device_uuid"] = "another-gpu"
    assert added(tmp_path, mutate) == ["stale device_uuid: the report was measured on other bytes"]


def test_a_stale_calibration_refuses(tmp_path):
    def mutate(report):
        report["workload"]["calibration"] = {"sha256": "9" * 64}
    assert added(tmp_path, mutate) == ["the workload names another calibration than this table's"]


def test_a_workload_without_a_calibration_identity_refuses(tmp_path):
    def mutate(report):
        report["workload"]["calibration"] = None
    assert (f"the workload names no calibration, so this table's calibration "
            f"{CALIBRATION_SHA} is bound to nothing") in added(tmp_path, mutate)


def test_a_relation_that_names_another_full_engine_run_refuses(tmp_path):
    """The digest comes from the relation's own full-engine run, so pointing
    the relation at the native run refuses rather than admitting its manifest."""
    relation = copy.deepcopy(RELATION)
    relation["full_engine_run_id"] = "dense"
    assert ("stale runtime_manifest_sha256: the report was measured on other bytes"
            in refusals(tmp_path, relation=relation))


def test_admission_requires_the_loaded_relation(tmp_path):
    table = build_table(tmp_path, agreeing_report())
    with pytest.raises(RuntimePriceError, match="requires the loaded runtime relation"):
        admit_fixed_resources(table, {})


# --------------------------------------------------------------------------
# Unsupported boundary: the scalar budget is TP1, one device, resident, eager,
# one request, and nothing else is projected.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("overrides,expected", [
    ({"graph_mode": "cudagraph"},
     "the table's graph_mode is 'cudagraph', and a recomputed partition covers only 'eager'"),
    ({"serving_context": {"residency": "streamed"}},
     "the table's residency is 'streamed', and a recomputed partition covers only 'resident'"),
    ({"tensor_parallel": 2},
     "the table's topology is 'tp2', and a recomputed partition covers only 'tp1'"),
    ({"batch_size": 4},
     "the table's batch size is 4, and a recomputed partition covers only one request"),
])
def test_an_unsupported_boundary_refuses(tmp_path, overrides, expected):
    baseline = refusals(tmp_path)
    assert expected not in baseline
    assert expected in refusals(tmp_path, context=context_payload(**overrides))


# --------------------------------------------------------------------------
# Assignment-dependent shared state, and the roster the reference partitions.
# --------------------------------------------------------------------------

def test_a_second_priced_format_for_one_unit_refuses(tmp_path):
    """One measured assignment cannot establish that shared workspace, cache
    policy or persistent buffers stay unchanged under an alternative."""
    rows = [("unit.a", "NVFP4"), ("unit.a", "FP8_DYNAMIC"), ("unit.b", "NVFP4")]
    expected = ("the table prices more than one format for ['unit.a'], and one measured "
                "assignment establishes no invariant fixed charge under the others")
    assert expected not in refusals(tmp_path)
    assert expected in refusals(tmp_path, rows=rows)


def test_a_census_that_omits_a_priced_unit_refuses(tmp_path):
    def mutate(report):
        report["reference"]["canonical_census"]["units"] = ["unit.a"]
    assert ("the canonical census names units ['unit.a'] where this table prices "
            "['unit.a', 'unit.b']") in added(tmp_path, mutate)


def test_a_selected_row_omitted_for_a_census_unit_refuses(tmp_path):
    def mutate(report):
        report["reference"]["selected_rows"] = [{"unit": "unit.a", "format": "NVFP4"}]
    assert ("the selected rows cover units ['unit.a'] where the census names "
            "['unit.a', 'unit.b']") in added(tmp_path, mutate)


def test_two_selected_rows_for_one_unit_refuse(tmp_path):
    def mutate(report):
        report["reference"]["selected_rows"].append({"unit": "unit.a", "format": "FP8_DYNAMIC"})
    assert ("unit 'unit.a' carries more than one selected row, so the report measures no "
            "single complete assignment") in added(tmp_path, mutate)


def test_a_selected_row_that_names_no_format_refuses(tmp_path):
    """What the producer emits today, which is why no selected row binds to a
    priced table row yet."""
    def mutate(report):
        report["reference"]["selected_rows"] = [{"unit": unit} for unit in CLOSED_UNITS]
    assert added(tmp_path, mutate) == [
        f"the selected row for unit {unit!r} names no format, so it binds to no priced table row"
        for unit in CLOSED_UNITS]


def test_a_selected_row_this_table_does_not_price_refuses(tmp_path):
    def mutate(report):
        report["reference"]["selected_rows"][0]["format"] = "MXFP4"
    assert added(tmp_path, mutate) == [
        "selected row ('unit.a', 'MXFP4') is not a row this table prices"]


@pytest.mark.parametrize("row", [{"unit": "unit.a", "format": "NVFP4", "bytes": 1},
                                 {"format": "NVFP4"}, "unit.a"])
def test_a_selected_row_that_is_not_a_closed_reference_refuses(tmp_path, row):
    def mutate(report):
        report["reference"]["selected_rows"][0] = row
    assert "selected row 0 is not a unit-and-format reference" in added(tmp_path, mutate)


def test_a_reference_without_a_canonical_census_refuses(tmp_path):
    def mutate(report):
        report["reference"]["canonical_census"] = None
    assert ("the reference carries no canonical census naming its units, so it partitions no "
            "model roster") in added(tmp_path, mutate)


# --------------------------------------------------------------------------
# A synthetic capture admits nothing.
# --------------------------------------------------------------------------

def test_a_synthetic_capture_refuses_on_this_gate_s_own_authority(tmp_path):
    """`fixture_provenance` travels from capture to ledger to `identity`, so a
    synthetic artifact can never read as a measurement. Clearing it removes
    this refusal and no other, which is what makes it a check rather than a
    restatement of the fixture's own comment."""
    provenance = agreeing_report()["identity"]["fixture_provenance"]
    expected = (f"the report declares fixture provenance {provenance!r}, and a synthetic "
                "capture admits no measured table")
    assert expected in refusals(tmp_path)
    report = agreeing_report()
    report["identity"]["fixture_provenance"] = None
    assert expected not in refusals(tmp_path, report)


def test_the_receipt_must_reference_a_report_rather_than_claim_a_result(tmp_path):
    """Inline numbers, status flags and proof digests are not recomputable, so
    they refuse before the relation is read at all."""
    for claim in ({}, {"resident_bytes": 0}, {"status": "qualified_complete"},
                  {"path": "report.json"}, {"path": "report.json", "sha256": "a" * 64,
                                            "admitted": True}, "report.json", 0):
        receipt = written(tmp_path, {"full_model_resources": claim}, "fixed.json")
        table = SimpleNamespace(source_path=str(tmp_path / "table.json"),
                                fixed_resources_receipt_path=receipt["path"],
                                fixed_resources_receipt_sha256=receipt["sha256"])
        with pytest.raises(RuntimePriceError, match="must reference one recomputable"):
            admit_fixed_resources(table, RELATION)


def test_a_receipt_with_no_resources_keeps_the_incomplete_refusal(tmp_path):
    receipt = written(tmp_path, {"full_model_resources": None}, "fixed.json")
    table = SimpleNamespace(source_path=str(tmp_path / "table.json"),
                            fixed_resources_receipt_path=receipt["path"],
                            fixed_resources_receipt_sha256=receipt["sha256"])
    with pytest.raises(RuntimePriceError, match="producer admission is incomplete"):
        admit_fixed_resources(table, RELATION)


def test_a_tampered_report_that_keeps_the_receipt_s_hash_refuses(tmp_path):
    reference = written(tmp_path, agreeing_report(), "report.json")
    receipt = written(tmp_path, {"full_model_resources": reference}, "fixed.json")
    (tmp_path / "report.json").write_bytes(b'{"schema": "tessera.full_engine_resource_report.v1"}')
    table = SimpleNamespace(source_path=str(tmp_path / "table.json"),
                            fixed_resources_receipt_path=receipt["path"],
                            fixed_resources_receipt_sha256=receipt["sha256"],
                            context=parse_runtime_context(context_payload()),
                            rows=(), fixed_resources=agreeing_resources(), fixed_assignment={})
    with pytest.raises(RuntimePriceError, match="artifact SHA-256"):
        admit_fixed_resources(table, RELATION)


# --------------------------------------------------------------------------
# What this schema version cannot express. These are disclosures, not checks:
# no input lifts them, which is exactly why nothing here pretends otherwise.
# --------------------------------------------------------------------------

def test_the_gate_names_what_the_producer_still_owes(tmp_path):
    reasons = refusals(tmp_path)
    owed = ["the capture observes no timing_captures, so the fixed prefill and decode charge "
            "has no evidence",
            "the capture observes no worker_startup_records, so the fixed resident and "
            "activation charge has no evidence",
            "the capture observes no kv_observations, so the fixed KV charge has no evidence",
            "the capture observes no owner_views, so the fixed member roster and each "
            "candidate's retained weights has no evidence",
            "the native-row and full-engine transient charge boundary is not versioned, so no "
            "candidate activation or scratch term may be compared to a priced row",
            "the partition names no fixed member, so this table's fixed_assignment binds to no "
            "observed allocation"]
    for reason in owed:
        assert reason in reasons


def test_the_absent_timing_partition_leaves_the_prefill_charge_unevidenced(tmp_path):
    """The prefill axis the measured table exists to serve. `evaluate_measured_
    assignment` sums `fixed_resources.prefill_ms` into the operator-sum budget
    `--slo-prefill-p95-ttft-ms` gates, and this schema version emits no timing
    observation at all, so a perfect capture still admits no prefill charge."""
    resources = agreeing_resources()
    reasons = refusals(tmp_path)
    for field in ("prefill_ms", "decode_ms"):
        assert (f"the report carries no timing partition, so this table's fixed {field} "
                f"({getattr(resources, field)}) has no evidence") in reasons


def test_no_candidate_row_resident_charge_is_expressible(tmp_path):
    reasons = refusals(tmp_path)
    assert ("no candidate_resident is recomputable, so no priced row's resident bytes has "
            "evidence") in reasons
    assert not [reason for reason in reasons if "resident bytes 1600" in reason]


def test_the_gate_returns_no_admission_token(tmp_path):
    """The loader sets `producer_admitted` only because this raised nothing;
    there is no success value for a caller to read, mistake for a proof, or
    carry past the gate."""
    import ast
    from pathlib import Path
    import prismaquant.runtime_provenance as module
    assert refusals(tmp_path)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    gate = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "admit_fixed_resources")
    assert [node for node in ast.walk(gate) if isinstance(node, ast.Return)] == []
