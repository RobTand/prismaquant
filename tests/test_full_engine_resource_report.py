"""Contracts for the pure artifact consumer, never runtime or GPU measurements.

Every fixture here is synthetic and says so. A positive fixture proves the
parser and the recomputation contract only; it establishes no measurement and
admits nothing. Each tampering test restates the outer artifact hash, so what
catches the defect is the semantic consumer rather than the checksum reader.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

import pytest

from prismaquant.measured_runtime_prices import RuntimePriceError
from prismaquant import full_engine_resource_report as consumer
from prismaquant.full_engine_resource_report import (
    DOMAINS, ENVELOPE_MEMBERS, REPORT_SCHEMA, TERMS,
    consume_full_engine_resource_report, read_full_engine_resource_report,
)

MODULE = Path(consumer.__file__)
SUPPLIED = Path(__file__).resolve().parent / "fixtures" / "full_engine_resource_report.synthetic.json"


def written(tmp_path, report, name="report.json"):
    """Write a report and restate its hash, so tampering reaches the consumer."""
    raw = (report if isinstance(report, bytes)
           else json.dumps(report, sort_keys=True, allow_nan=False).encode())
    path = tmp_path / name
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def supplied():
    return json.loads(SUPPLIED.read_bytes())


def consume(tmp_path, report, **kwargs):
    return consume_full_engine_resource_report(written(tmp_path, report), root=tmp_path, **kwargs)


def mutated(mutate):
    report = supplied()
    mutate(report)
    return report


# --------------------------------------------------------------------------
# The supplied artifact. Its correct verdict is a refusal that names why.
# --------------------------------------------------------------------------

def test_the_supplied_synthetic_report_refuses_and_names_its_open_domains_and_unclassified_allocation(tmp_path):
    verdict = consume(tmp_path, supplied())
    assert verdict.open_domains == ("worker_startup", "external_closure",
                                    "provenance_admission", "cache_capacity", "timing_partition")
    assert verdict.unclassified_allocations == ("0:4608:1",)
    assert verdict.expressible_terms == ()
    assert verdict.recomputed_scalar_budget_bytes is None
    for name in verdict.open_domains:
        assert any(name in reason for reason in verdict.blocking)
    assert any("0:4608:1" in reason and "unclassified" in reason for reason in verdict.blocking)
    assert "synthetic" in verdict.fixture_provenance


def test_the_consumer_recomputes_the_supplied_report_without_disagreeing_with_it(tmp_path):
    """Every number the producer declares reproduces from the raw observations."""
    assert consume(tmp_path, supplied()).disagreements == ()


def test_the_consumer_reads_the_artifact_and_never_the_serving_runtime():
    """The artifact travels; the code does not (AGENTS.md principle 5)."""
    source = MODULE.read_text(encoding="utf-8")
    assert re.search(r"^\s*(import|from)\s+\S*tessera", source, flags=re.MULTILINE) is None
    assert "subprocess" not in source and "importlib" not in source
    mentions = re.findall(r"tessera\S*", source)
    assert mentions, "the schema the artifact declares about itself should be named"
    for mention in mentions:
        # Every mention is part of a schema identifier, never a module path.
        assert re.match(r"tessera\.full_engine_resource_(report|identity|partition)\.v1",
                        mention), mention


def test_admission_stays_closed_and_no_gate_reads_the_recomputed_partition(tmp_path):
    from types import SimpleNamespace
    from prismaquant.runtime_provenance import admit_fixed_resources
    reference = written(tmp_path, {"full_model_resources": {"resident_bytes": 0}}, "fixed.json")
    table = SimpleNamespace(source_path=str(tmp_path / "table.json"),
                            fixed_resources_receipt_path=reference["path"],
                            fixed_resources_receipt_sha256=reference["sha256"])
    with pytest.raises(RuntimePriceError, match="no qualified recomputable"):
        admit_fixed_resources(table, {})
    root = MODULE.parent
    importers = [path.name for path in sorted(root.glob("*.py"))
                 if "full_engine_resource_report" in path.read_text(encoding="utf-8")
                 and path != MODULE]
    assert importers == []


# --------------------------------------------------------------------------
# The keystone: `derived` is a claim, never an input.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("term", TERMS)
def test_a_changed_derived_term_refuses_because_the_recomputation_disagrees(tmp_path, term):
    def mutate(report):
        report["derived"]["terms"][term] = 4096
        report["derived"]["scope"]["unavailable_terms"].remove(term)
    verdict = consume(tmp_path, mutated(mutate))
    assert any(f"derived term {term} is 4096" in reason for reason in verdict.disagreements)
    assert any("unavailable terms" in reason for reason in verdict.disagreements)


def test_a_changed_derived_scalar_budget_refuses_because_the_recomputation_disagrees(tmp_path):
    verdict = consume(tmp_path, mutated(lambda r: r["derived"].update(scalar_budget_bytes=20 * 1024**3)))
    assert any("scalar budget is 21474836480" in reason and "recomputes None" in reason
               for reason in verdict.disagreements)


def test_a_zero_derived_total_is_not_an_unknown_one(tmp_path):
    """`None` and `0` are different claims and the comparison is type-strict."""
    verdict = consume(tmp_path, mutated(lambda r: r["derived"]["terms"].update(fixed_scratch=0)))
    assert any("derived term fixed_scratch is 0" in reason for reason in verdict.disagreements)


def test_a_changed_observed_live_peak_refuses_as_not_the_simultaneous_maximum(tmp_path):
    """512 + 512 + 1024 is a sum of per-allocation maxima, not a peak."""
    verdict = consume(tmp_path, mutated(
        lambda r: r["observations"].update(torch_observed_live_peak_bytes=2048)))
    assert any("maximum simultaneous sum" in reason for reason in verdict.disagreements)


# --------------------------------------------------------------------------
# Envelope and reader refusals, before any arithmetic.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("schema", [None, "", "tessera.full_engine_resource_capture.v1",
                                    "tessera.full_engine_raw_resource_ledger.v1",
                                    "tessera.full_engine_resource_report.v2"])
def test_a_foreign_schema_refuses(tmp_path, schema):
    report = supplied()
    if schema is None:
        del report["schema"]
    else:
        report["schema"] = schema
    with pytest.raises(RuntimePriceError, match="unsupported schema"):
        consume(tmp_path, report)


@pytest.mark.parametrize("member", ENVELOPE_MEMBERS)
def test_a_missing_envelope_member_refuses(tmp_path, member):
    with pytest.raises(RuntimePriceError, match="missing envelope member " + member):
        consume(tmp_path, mutated(lambda r: r.pop(member)))


def test_an_unnamed_envelope_field_refuses(tmp_path):
    with pytest.raises(RuntimePriceError, match="expected exactly fields"):
        consume(tmp_path, mutated(lambda r: r.update(admission="qualified")))


def test_a_boolean_where_an_integer_is_declared_refuses(tmp_path):
    """`True` is an instance of `int` in Python. It is not one here."""
    with pytest.raises(RuntimePriceError, match="expected integer"):
        consume(tmp_path, mutated(
            lambda r: r["observations"]["torch_allocations"][0].update(bytes=True)))


@pytest.mark.parametrize("literal", [b"NaN", b"Infinity", b"-Infinity"])
def test_a_nonfinite_numeric_field_refuses(tmp_path, literal):
    """`json.loads` accepts these by default; the artifact reader does not."""
    raw = SUPPLIED.read_bytes().replace(b'"torch_observed_live_peak_bytes": 1536',
                                        b'"torch_observed_live_peak_bytes": ' + literal)
    assert literal in raw
    with pytest.raises(RuntimePriceError, match="invalid JSON artifact"):
        consume_full_engine_resource_report(written(tmp_path, raw), root=tmp_path)


def test_a_duplicate_json_key_refuses(tmp_path):
    raw = SUPPLIED.read_bytes().replace(b'"schema": "tessera.full_engine_resource_report.v1"',
                                        b'"schema": "tessera.full_engine_resource_report.v1",'
                                        b' "schema": "tessera.full_engine_resource_report.v1"', 1)
    with pytest.raises(RuntimePriceError, match="duplicate JSON key"):
        consume_full_engine_resource_report(written(tmp_path, raw), root=tmp_path)


def test_a_negative_size_refuses(tmp_path):
    with pytest.raises(RuntimePriceError, match="expected integer >= 0"):
        consume(tmp_path, mutated(
            lambda r: r["observations"]["torch_allocations"][0].update(bytes=-512)))


def test_an_unknown_enum_value_refuses(tmp_path):
    with pytest.raises(RuntimePriceError, match="unknown value"):
        consume(tmp_path, mutated(
            lambda r: r["observations"]["torch_allocations"][0].update(lifetime_scope="maybe")))


def test_a_tampered_report_that_keeps_its_old_hash_refuses_at_the_digest(tmp_path):
    report = mutated(lambda r: r["derived"].update(scalar_budget_bytes=1))
    reference = written(tmp_path, report)
    reference["sha256"] = hashlib.sha256(SUPPLIED.read_bytes()).hexdigest()
    with pytest.raises(RuntimePriceError, match="artifact SHA-256"):
        consume_full_engine_resource_report(reference, root=tmp_path)


# --------------------------------------------------------------------------
# The named refusals of the merged design.
# --------------------------------------------------------------------------

def test_a_reused_pointer_generation_refuses(tmp_path):
    def mutate(report):
        rows = report["observations"]["torch_allocations"]
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(RuntimePriceError, match="pointer generation is reused"):
        consume(tmp_path, mutated(mutate))


def test_an_allocation_identity_that_hides_its_generation_refuses(tmp_path):
    with pytest.raises(RuntimePriceError, match="does not name its device, address and generation"):
        consume(tmp_path, mutated(
            lambda r: r["observations"]["torch_allocations"][2].update(allocation_id="0:4608")))


@pytest.mark.parametrize("mutation", ["completed_without_requested", "freed_before_allocated",
                                      "settled_out_of_order", "live_at_a_later_checkpoint"])
def test_a_live_free_mismatch_refuses(tmp_path, mutation):
    report = supplied()
    row = report["observations"]["torch_allocations"][2]
    if mutation == "completed_without_requested":
        row["free_requested_index"] = None
        with pytest.raises(RuntimePriceError, match="requested free"):
            consume(tmp_path, report)
    elif mutation == "freed_before_allocated":
        row.update(free_requested_index=2, free_completed_index=3)
        with pytest.raises(RuntimePriceError, match="must follow the allocation"):
            consume(tmp_path, report)
    elif mutation == "settled_out_of_order":
        row.update(free_requested_index=10, free_completed_index=9)
        with pytest.raises(RuntimePriceError, match="settle in order"):
            consume(tmp_path, report)
    else:
        # Freed at 7, yet the `unit_end` checkpoint at index 8 still owns it.
        row.update(free_requested_index=7, free_completed_index=7)
        verdict = consume(tmp_path, report)
        assert any("does not list the owned storages live at its index" in reason
                   for reason in verdict.disagreements)


@pytest.mark.parametrize("mutation", ["omitted_extent", "omitted_owner", "omitted_unit"])
def test_an_omitted_unit_owner_or_extent_refuses(tmp_path, mutation):
    report = supplied()
    if mutation == "omitted_extent":
        report["partition"]["membership"] = [report["partition"]["membership"][0]]
        expected = "not the ones this consumer classifies"
    elif mutation == "omitted_owner":
        report["observations"]["torch_allocations"][0]["observed_owners"] = ["embedding.weight"]
        expected = "another owner count"
    else:
        report["partition"]["units"] = []
        expected = "names a unit the partition omits"
    assert any(expected in reason for reason in consume(tmp_path, report).disagreements)


def test_a_duplicate_alias_refuses(tmp_path):
    def mutate(report):
        rows = report["observations"]["torch_allocations"]
        rows[2]["observed_owners"] = ["lm_head.weight"]
    assert any("duplicate alias claimed by two backings" in reason
               for reason in consume(tmp_path, mutated(mutate)).disagreements)


def test_a_repeated_alias_inside_one_storage_refuses(tmp_path):
    with pytest.raises(RuntimePriceError, match="duplicate entry"):
        consume(tmp_path, mutated(
            lambda r: r["observations"]["checkpoints"][0]["storages"][0].update(
                owners=["embedding.weight", "embedding.weight"])))


def test_a_candidate_and_fixed_overlap_refuses(tmp_path):
    def mutate(report):
        report["observations"]["torch_allocations"][0]["observed_categories"] = ["candidate", "fixed"]
    verdict = consume(tmp_path, mutated(mutate))
    assert any("claimed by both a candidate and a fixed owner" in reason
               for reason in verdict.disagreements)
    # It also stops being classified at all, which nulls every term.
    assert "0:4096:1" in verdict.unclassified_allocations


def test_a_missing_parent_segment_refuses(tmp_path):
    """A Torch suballocation and its CUDA segment are two views of one backing."""
    def mutate(report):
        report["observations"]["torch_allocations"][0]["allocator_block_bytes_observed"] = [256]
    assert any("no observed parent segment large enough" in reason
               for reason in consume(tmp_path, mutated(mutate)).disagreements)


@pytest.mark.parametrize("mutation", ["unknown_status", "handled_while_unavailable", "domain_issue"])
def test_an_unknown_allocation_api_refuses(tmp_path, mutation):
    report = supplied()
    domains = report["observations"]["cuda_argument_domains"]
    if mutation == "unknown_status":
        domains["status"] = "partially_observed"
        with pytest.raises(RuntimePriceError, match="unknown value"):
            consume(tmp_path, report)
        return
    if mutation == "handled_while_unavailable":
        domains["handled_api_keys"] = ["cudaMallocAsync"]
        expected = "declares itself unavailable"
    else:
        domains["issues"] = ["unhandled allocation API"]
        expected = "unresolved issues"
    assert any(expected in reason for reason in consume(tmp_path, report).disagreements)


@pytest.mark.parametrize("key", ["model_sha256", "runtime_manifest_sha256", "workload_sha256",
                                 "configuration_sha256", "assignment_sha256"])
def test_a_stale_source_workload_or_runtime_refuses(tmp_path, key):
    verdict = consume(tmp_path, supplied(), expected_run_identity={key: "b" * 64})
    assert any(f"stale {key}" in reason for reason in verdict.disagreements)


def test_a_calibration_identity_is_not_expressible_so_its_absence_blocks(tmp_path):
    """`identity.run` carries no `calibration_sha256`, so stale calibration can
    only be refused through the workload member, which carries none either."""
    assert "calibration_sha256" not in supplied()["identity"]["run"]
    verdict = consume(tmp_path, supplied(), expected_run_identity={"calibration_sha256": "b" * 64})
    assert any("declares no calibration_sha256" in reason for reason in verdict.disagreements)
    assert any("no calibration identity" in reason for reason in verdict.blocking)


def test_a_partition_of_another_run_refuses(tmp_path):
    def mutate(report):
        report["partition"]["identity"]["model_sha256"] = "b" * 64
    assert any("differs from the report identity" in reason
               for reason in consume(tmp_path, mutated(mutate)).disagreements)


def test_a_partition_of_another_capture_refuses(tmp_path):
    def mutate(report):
        report["partition"]["capture_sha256"] = "b" * 64
    assert any("names a different capture" in reason
               for reason in consume(tmp_path, mutated(mutate)).disagreements)


@pytest.mark.parametrize("domain", [name for name in DOMAINS if name != "history_join"])
def test_a_domain_that_closes_where_the_consumer_cannot_refuses(tmp_path, domain):
    """Altered cache capacity, a missing timing tail and overlapping streams
    all reach this consumer only as a domain state, because the report carries
    no KV, timing or stream observation to recompute."""
    def mutate(report):
        for member in ("derived", "partition"):
            report[member]["domains"][domain] = {"state": "closed", "reason": None,
                                                 "evidence": ["asserted"]}
    verdict = consume(tmp_path, mutated(mutate))
    assert any(f"calls domain {domain} closed where this consumer recomputes it as not closed"
               in reason for reason in verdict.disagreements)


def test_the_two_checkable_domains_close_only_on_their_own_observed_condition(tmp_path):
    """`history_join` reads the unattributed external records and
    `external_closure` reads the external native peak. The other four have no
    condition in this schema, which is why they stay open above."""
    assert consume(tmp_path, supplied()).disagreements == ()

    def unattributed(report):
        report["observations"]["unattributed_external_records"] = [{"record": 1}]
    assert any("calls domain history_join closed where this consumer recomputes it as not closed"
               in reason for reason in consume(tmp_path, mutated(unattributed)).disagreements)

    def external(report):
        report["observations"]["external_native_peak_bytes"] = 4096
    assert any("calls domain external_closure open where this consumer recomputes it as closed"
               in reason for reason in consume(tmp_path, mutated(external)).disagreements)


def test_a_domain_that_closes_without_evidence_refuses(tmp_path):
    def mutate(report):
        report["derived"]["domains"]["cache_capacity"] = {"state": "closed", "reason": None,
                                                          "evidence": []}
    with pytest.raises(RuntimePriceError, match="evidence must be present exactly when"):
        consume(tmp_path, mutated(mutate))


def test_a_foreign_rank_or_device_refuses(tmp_path):
    def mutate(report):
        rows = report["observations"]["torch_allocations"]
        rows[0]["allocation_id"] = "1:4096:1"
        for checkpoint in report["observations"]["checkpoints"]:
            for storage in checkpoint["storages"]:
                if storage["allocation_id"] == "0:4096:1":
                    storage["allocation_id"] = "1:4096:1"
        for row in report["partition"]["membership"]:
            if row["allocation_id"] == "0:4096:1":
                row["allocation_id"] = "1:4096:1"
    assert any("is on a foreign device" in reason
               for reason in consume(tmp_path, mutated(mutate)).disagreements)


@pytest.mark.parametrize("member,key,value", [
    ("execution", "topology", "tp2"),
    ("execution", "graph_mode", "graph"),
    ("execution", "residency", "streamed"),
])
def test_an_unsupported_boundary_refuses(tmp_path, member, key, value):
    with pytest.raises(RuntimePriceError, match="unsupported " + key):
        consume(tmp_path, mutated(lambda r: r[member].update({key: value})))


@pytest.mark.parametrize("key,value", [
    ("topology", "tp2_rank_sum"),
    ("allocation_scope", "host_and_device"),
    ("invariance", "every assignment the allocator may pick"),
])
def test_an_unsupported_derived_scope_refuses(tmp_path, key, value):
    with pytest.raises(RuntimePriceError, match="unsupported " + key):
        consume(tmp_path, mutated(lambda r: r["derived"]["scope"].update({key: value})))


@pytest.mark.parametrize("label", ["shared", "unknown"])
def test_assignment_dependent_shared_state_stays_unclassified_and_nulls_every_term(tmp_path, label):
    def mutate(report):
        report["observations"]["torch_allocations"][0]["observed_categories"] = [label]
    verdict = consume(tmp_path, mutated(mutate))
    assert "0:4096:1" in verdict.unclassified_allocations
    assert verdict.recomputed_terms == {name: None for name in TERMS}
    assert any("neither classification nor invariance" in reason for reason in verdict.blocking)


def test_a_transient_freed_outside_every_unit_interval_stays_unclassified(tmp_path):
    """No declared step boundary, so once-per-step and never-again are both fills."""
    def mutate(report):
        report["observations"]["torch_allocations"][0].update(
            free_requested_index=11, free_completed_index=12)
        report["observations"]["checkpoints"] = []
        report["partition"]["membership"] = [
            row for row in report["partition"]["membership"] if row["allocation_id"] != "0:4096:1"]
    verdict = consume(tmp_path, mutated(mutate))
    assert "0:4096:1" in verdict.unclassified_allocations
    assert any("no declared step boundary" in reason for reason in verdict.blocking)


# --------------------------------------------------------------------------
# The arithmetic, on a fixture whose expected numbers are written by hand.
# --------------------------------------------------------------------------

#  id           alloc  free  bytes  lifetime_scope   owner        lifetime    unit
CLOSED_ROWS = [
    ("0:1000:1",     1, None,  4096, "outside_units", "fixed",     "resident",  None),
    ("0:2000:1",    10,   14,  1000, "inside_unit",   "fixed",     "scratch",   None),
    ("0:3000:1",    12,   20,  2000, "inside_unit",   "fixed",     "scratch",   None),
    ("0:4000:1",    14,   18,  3000, "inside_unit",   "fixed",     "scratch",   None),
    ("0:5000:1",    11,   13,   700, "inside_unit",   "candidate", "scratch",   "unit.a"),
    ("0:6000:1",    12,   16,   800, "inside_unit",   "candidate", "scratch",   "unit.a"),
    ("0:7000:1",    30,   34,  1200, "inside_unit",   "candidate", "scratch",   "unit.b"),
    ("0:8000:1",    34,   38,  1300, "inside_unit",   "candidate", "scratch",   "unit.b"),
    ("0:9000:1",    12,   40,   500, "escapes_unit",  "candidate", "activation", "unit.a"),
]
# Hand-computed from the intervals above, not from the module under test:
#   fixed scratch  1000[10,14) 2000[12,20) 3000[14,18): 10>1000 12>3000
#                  14>free 1000 then take 3000 = 5000, 18>2000, 20>0.  Peak 5000.
#                  A sum of per-allocation maxima would be 6000.
#   unit.a scratch  700[11,13) 800[12,16): 11>700 12>1500 13>800.       Peak 1500.
#   unit.b scratch 1200[30,34) 1300[34,38): the free at 34 settles first. Peak 1300.
#   candidate scratch is the largest unit peak, 1500, not the 2800 sum.
#   whole-capture peak: 4096 +1000 +700 +(2000+800+500) -700 -1000 +3000 = 10396.
CLOSED_FIXED_SCRATCH = 5000
CLOSED_CANDIDATE_SCRATCH = 1500
CLOSED_LIVE_PEAK = 10396


def closed_report():
    allocations, membership = [], []
    for index, (ident, alloc, free, size, scope, owner, lifetime, unit) in enumerate(CLOSED_ROWS):
        device, address, generation = ident.split(":")
        allocations.append({
            "address": int(address), "allocate_index": alloc, "allocation_id": ident,
            "allocator_block_bytes_observed": [size], "bytes": size,
            "free_completed_index": free,
            "free_requested_index": None if free is None else free - 1,
            "generation": int(generation), "lifetime_scope": scope,
            "observed_categories": [owner], "observed_owners": [f"owner.{index}"],
            "scope_stack": [] if scope == "outside_units" else ["unit"],
            "unit_invocation": None if unit is None and scope == "outside_units" else f"{unit or 'unit.a'}:0",
        })
        membership.append({"allocate_index": alloc, "allocation_id": ident, "bytes": size,
                           "free_completed_index": free, "lifetime_class": lifetime,
                           "owner_class": owner, "unit": unit if owner == "candidate" else None})
    terms = {name: None for name in TERMS}
    terms["fixed_scratch"] = CLOSED_FIXED_SCRATCH
    terms["candidate_scratch"] = CLOSED_CANDIDATE_SCRATCH
    domains = {name: {"state": "open", "evidence": [], "reason": "no implemented check"}
               for name in DOMAINS}
    for name in ("history_join", "external_closure"):
        domains[name] = {"state": "closed", "evidence": [name], "reason": None}
    scope = {"allocation_scope": "gpu_allocations_only", "expressible": False,
             "invariance": "one complete assignment, one row per unit",
             "topology": "tp1_single_device_resident_eager",
             "unavailable_terms": sorted(name for name in TERMS if terms[name] is None),
             "unclassified_allocation_count": 0}
    identity = {"assignment_sha256": "a" * 64, "canonical_units_sha256": "a" * 64,
                "configuration_sha256": "a" * 64, "device_id": 0, "device_uuid": "synthetic-device",
                "model_sha256": "a" * 64, "runtime_manifest_sha256": "a" * 64,
                "schema": "tessera.full_engine_resource_identity.v1", "workload_sha256": "a" * 64}
    return {
        "schema": REPORT_SCHEMA,
        "identity": {"capture_sha256": "c" * 64,
                     "fixture_provenance": "synthetic CPU-only arithmetic fixture",
                     "run": copy.deepcopy(identity)},
        "reference": {"canonical_census": {"units": ["unit.a", "unit.b"]}, "note": "synthetic",
                      "runtime_binding": {"member_formats": {}},
                      "selected_rows": [{"unit": "unit.a"}, {"unit": "unit.b"}]},
        "workload": {"calibration": {"sha256": "a" * 64}, "note": "synthetic",
                     "prompt_ids": [0, 1], "sampling": {"greedy": True}},
        "execution": {"graph_mode": "eager", "note": "synthetic", "residency": "resident",
                      "topology": "tp1"},
        "observations": {
            "artifacts": [], "capture_sha256": "c" * 64, "checkpoints": [],
            "cuda_argument_domains": {"handled_api_keys": ["cudaMalloc"], "host_allocations": [],
                                      "host_mappings": [], "issues": [], "null_device_frees": [],
                                      "scope": "observed pinned-host lifetimes", "status": "observed"},
            "external_native_peak_bytes": 2048, "issues": [],
            "torch_allocations": allocations,
            "torch_observed_live_peak_bytes": CLOSED_LIVE_PEAK,
            "torch_observed_live_peak_scope": "requested_allocation_bytes_excluding_allocator_rounding",
            "unattributed_external_records": []},
        "partition": {"capture_sha256": "c" * 64, "domains": copy.deepcopy(domains),
                      "identity": copy.deepcopy(identity), "membership": membership,
                      "schema": "tessera.full_engine_resource_partition.v1",
                      "scope": copy.deepcopy(scope), "terms": copy.deepcopy(terms),
                      "unclassified_allocations": [], "units": ["unit.a", "unit.b"]},
        "derived": {"domains": domains, "scalar_budget_bytes": None, "scope": scope,
                    "terms": terms},
    }


def test_a_peak_is_a_sweep_and_not_a_sum_of_per_allocation_maxima(tmp_path):
    verdict = consume(tmp_path, closed_report())
    assert verdict.disagreements == ()
    assert verdict.recomputed_terms["fixed_scratch"] == CLOSED_FIXED_SCRATCH == 5000
    assert verdict.recomputed_terms["fixed_scratch"] < 1000 + 2000 + 3000


def test_frees_settle_before_allocations_at_the_same_index(tmp_path):
    """unit.b frees 1200 at index 34 and allocates 1300 at index 34."""
    verdict = consume(tmp_path, closed_report())
    assert verdict.recomputed_terms["candidate_scratch"] == CLOSED_CANDIDATE_SCRATCH == 1500
    assert verdict.recomputed_terms["candidate_scratch"] < 1500 + 1300


def test_a_candidate_term_is_the_largest_unit_peak_and_not_their_sum(tmp_path):
    verdict = consume(tmp_path, closed_report())
    assert verdict.recomputed_terms["candidate_scratch"] == max(1500, 1300)


def test_a_fully_recomputing_report_still_expresses_no_scalar_budget(tmp_path):
    """Four of the six domains have no closing condition this schema emits, so
    at v1 the resident, activation and KV terms can never become numbers and
    the composition can never complete."""
    verdict = consume(tmp_path, closed_report())
    assert verdict.disagreements == ()
    assert verdict.expressible_terms == ("fixed_scratch", "candidate_scratch")
    assert verdict.recomputed_scalar_budget_bytes is None
    assert any("no scalar device budget is expressible" in reason for reason in verdict.blocking)


@pytest.mark.parametrize("term,wrong", [("fixed_scratch", 6000), ("candidate_scratch", 2800),
                                        ("fixed_scratch", CLOSED_FIXED_SCRATCH + 1)])
def test_a_changed_derived_total_on_an_agreeing_report_refuses(tmp_path, term, wrong):
    report = closed_report()
    report["derived"]["terms"][term] = wrong
    verdict = consume(tmp_path, report)
    assert any(f"derived term {term} is {wrong}" in reason for reason in verdict.disagreements)


def test_an_unrecomputable_kv_charge_refuses(tmp_path):
    """`fixed_kv` has a declared term and no observation to recompute it from."""
    report = closed_report()
    report["derived"]["terms"]["fixed_kv"] = 1024
    report["derived"]["scope"]["unavailable_terms"].remove("fixed_kv")
    verdict = consume(tmp_path, report)
    assert any("no KV observation to recompute" in reason for reason in verdict.disagreements)


def test_the_reader_returns_the_report_it_validated(tmp_path):
    report = closed_report()
    assert read_full_engine_resource_report(written(tmp_path, report), root=tmp_path) == report
