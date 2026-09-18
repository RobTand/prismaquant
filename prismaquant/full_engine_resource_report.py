"""Pure artifact consumer for ``tessera.full_engine_resource_report.v1`` and ``.v2``.

This module reads one producer-emitted JSON report and nothing else. It does
not import, vendor, launch or link the serving runtime, and it adds no
dependency on it (AGENTS.md principle 5); the only such name in this file is
the schema string the artifact declares about itself. The artifact travels,
the code does not.

``derived`` is a claim, never an input. Every quantity reported here is
recomputed from ``partition`` and ``observations``, and a producer number that
disagrees with that recomputation is a refusal. A producer that certifies its
own partition is the failure this consumer exists to catch, so no branch here
reads a number out of ``derived`` and uses it (AGENTS.md principle 14: a claim
about another runtime is attested, never asserted).

``runtime_provenance.admit_fixed_resources`` calls this module and admits only
when it returns no refusal. This module still admits nothing itself: it reports
what it recomputed and every reason the report is not usable, and the gate
decides. Nothing here reads a number out of ``derived`` and uses it.

The producer prices a row that is live during no declared engine step
separately, as ``non_step_transient_peak_bytes``, because the seven composition
terms price one engine step and those bytes are live during none of them. The
obligation a placement has to satisfy is therefore
``max(scalar_budget_bytes, non_step_transient_peak_bytes)``, and this consumer
recomputes both sides of that maximum.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .measured_runtime_prices import RuntimePriceError, _integer, _object, _sha, _string
from .runtime_provenance import ArtifactReader, _equal

REPORT_SCHEMA = "tessera.full_engine_resource_report.v1"
#: The producer's second envelope (the runtime's PR 556): the same seven members, plus
#: an observer-allocation class in the partition and three producer-derived
#: blocks (``admission``, ``fixed_resources``, ``timing_terms``) this consumer
#: validates as carried and reads nothing from. The four observations v1 owed
#: as null may be carried; this consumer recomputes no term and closes no
#: domain from them, and says so by name where a domain is closed on one.
REPORT_SCHEMA_V2 = "tessera.full_engine_resource_report.v2"
REPORT_SCHEMAS = (REPORT_SCHEMA, REPORT_SCHEMA_V2)
IDENTITY_SCHEMA = "tessera.full_engine_resource_identity.v1"
PARTITION_SCHEMA = "tessera.full_engine_resource_partition.v1"

#: The seven envelope members, spelled as the consumer design spells them.
ENVELOPE_MEMBERS = ("identity", "reference", "workload", "execution",
                    "observations", "partition", "derived")

DOMAINS = ("worker_startup", "history_join", "external_closure",
           "provenance_admission", "cache_capacity", "timing_partition")
DOMAIN_STATES = ("closed", "open", "refused")

#: Domains whose closing condition this consumer can itself check against the
#: raw observations this schema emits. Every other domain stays open here
#: whatever the report says: a state the consumer cannot verify is
#: ``qualified: true`` spelled differently. The first two close on the raw
#: ledger alone; the last two close on the shape of the startup and KV
#: observations, which is what makes the fixed resident, activation and KV
#: terms recomputable rather than permanently owed.
CHECKABLE_DOMAINS = ("worker_startup", "history_join", "external_closure",
                     "cache_capacity")

#: The observations this schema version names and sets to null, so a consumer
#: can tell "this capture did not observe it" from "the producer dropped it".
#: They are what the four domains with no consumer-checkable closing condition
#: would close on, which is why the scalar composition cannot complete at v1
#: even on a flawless capture: only `fixed_scratch` and `candidate_scratch` are
#: reachable from a capture that supplies none of them. A non-null value here
#: is refused rather than read -- this consumer recomputes nothing from an
#: observation whose shape v1 does not define, and a domain that closed on one
#: would be certifying itself.
#:
#: ``worker_startup_records`` and ``kv_observations`` left this list when their
#: shapes were defined below: they are now *checked* observations that close
#: ``worker_startup`` and ``cache_capacity``, which is what makes
#: ``fixed_resident``, ``fixed_activation`` and ``fixed_kv`` recomputable from a
#: capture that actually observed them, instead of terms no schema version
#: could express.
OWED_OBSERVATIONS = ("observer_qualification", "owner_views",
                     "runtime_provenance_relation", "timing_captures")

#: One rank's resident-after-load observation: the engine-side sample the
#: producer's own contract names for ``fixed_resident``. ``rank`` is the
#: process that measured it (not the world it was configured for),
#: ``memory_allocated_bytes`` is ``torch.cuda.memory_allocated()`` sampled after
#: ``process_weights_after_loading`` and after ``lock_workspace()``,
#: ``receipt_resident_bytes`` is the receipt's own ``resources.resident_bytes``
#: and ``workspace_resident_bytes`` the ``resident_bytes`` of the producer's own
#: ``tessera.native_moe_workspace.v1`` record
#: (``bench_native_moe_operator.observe_workspace``), which is the sum of that
#: record's distinct slot storages. ``worker_startup`` closes when the workspace
#: is locked and the ledger's fixed-owned resident rows sum to the receipt's
#: figure, which is what ties the engine-side sample to the allocation ledger
#: this consumer recomputes from. ``scope`` names the sample in the producer's
#: own words, exactly as the KV record does, so the record says what it observed
#: rather than leaving a reader to infer it.
WORKER_STARTUP_RECORD_FIELDS = ("memory_allocated_bytes", "rank",
                                "receipt_resident_bytes", "scope", "workspace_locked",
                                "workspace_resident_bytes")

#: One rank's KV-pool accounting, and the engine's rather than this operator's.
#: This is the record the runtime's own read-only KV observer already returns
#: (``experiments/full_engine_kv.py:inspect_worker_kv``), and it is taken as it
#: is emitted rather than projected into a smaller spelling of the same pool:
#: ``num_blocks``, one ``group_page_size_bytes`` per cache group -- a hybrid or
#: multi-group config has more than one and its groups do not share a page
#: geometry, so no single block size is the pool -- and the sensor's own
#: ``storage`` block, whose ``storages`` are the physical backings deduplicated
#: by ``(device_type, device_id, address)`` with their ``owners`` views and
#: whose ``unique_physical_storage_bytes`` this consumer re-adds. The
#: scheduler's ``resolved_limits`` carry ``max_num_batched_tokens`` and
#: ``max_num_seqs``. Every other coordinate the observer emits (``tensors``,
#: ``received``, ``runner_resolved``, ``kernel_block_sizes``,
#: ``capacity_policy``, ``group_details``, ``capacity_assertions``,
#: ``received_argument_scope``) travels as evidence.
#:
#: ``runtime_admission`` is the producer's own closure attestation, and the
#: reason it is required rather than assumed: the intrusive snapshot pass sets
#: it ``False`` and says in ``scope`` that the pass is timing- and
#: admission-ineligible, so a consumer that closed this domain on that pass
#: would be pricing bytes the harness warns are not admissible. It never closes
#: the domain by itself -- the recomputed storage arithmetic and the ledger
#: equality below do that -- but a record that does not carry the attestation is
#: refused rather than read.
#:
#: Unlike the other observations this one is an *open* mapping: it carries the
#: runtime's own resolved descriptors, whose field set the runtime owns. The
#: consumer requires the coordinates it recomputes from and keeps the rest as
#: evidence.
KV_OBSERVATION_REQUIRED_FIELDS = ("group_page_size_bytes", "num_blocks",
                                  "resolved_limits", "runtime_admission", "scope", "storage")
KV_RESOLVED_LIMIT_FIELDS = ("max_num_batched_tokens", "max_num_seqs")
KV_STORAGE_SET_FIELDS = ("storages", "unique_physical_storage_bytes")
KV_STORAGE_FIELDS = ("address", "bytes", "device_id", "device_type", "owners")

TERMS = ("fixed_resident", "candidate_resident", "fixed_activation",
         "candidate_activation", "fixed_scratch", "candidate_scratch", "fixed_kv")

#: The candidate terms are per-unit mappings, not scalars. The declared
#: invariance is "one complete assignment, one row per unit", and it is the
#: per-unit breakdown that makes `sum(candidate_resident)` and
#: `max(candidate_activation)` in the composition mean anything: the reduction
#: belongs to the composition, not to the term. Comparing mapping to mapping
#: is also strictly stronger than comparing the reductions, which agree by
#: coincidence whenever two units trade the same bytes.
PER_UNIT_TERMS = ("candidate_resident", "candidate_activation", "candidate_scratch")

TERM_DEPENDENCIES = {
    "fixed_resident": ("worker_startup", "history_join", "external_closure"),
    "candidate_resident": ("worker_startup", "history_join", "external_closure"),
    "fixed_activation": ("worker_startup", "history_join"),
    "candidate_activation": ("worker_startup", "history_join"),
    "fixed_scratch": ("history_join", "external_closure"),
    "candidate_scratch": ("history_join", "external_closure"),
    "fixed_kv": ("cache_capacity",),
}

#: Every term is recomputed from the observations; what gates each one is its
#: domain dependency, not this consumer's reach.
RECOMPUTABLE_TERMS = TERMS

#: The three owner classes the producer accepts. `kv` is one of them: a KV
#: backing is an observed allocation like any other and `fixed_kv` is the sum
#: of the resident ones, so dropping it here would unclassify a real row and
#: null every term on any capture that has one. What `fixed_kv` waits on is its
#: domain, `cache_capacity`, which closes on a KV observation this capture must
#: carry rather than on any producer flag.
OWNER_CLASSES = ("fixed", "candidate", "kv")
#: Neither label supplies a classification or an invariance, so a row carrying
#: one is unclassified and named, never bucketed.
UNSUPPORTED_OWNER_LABELS = ("shared", "unknown")
LIFETIME_SCOPES = ("outside_units", "inside_unit", "escapes_unit")
#: ``non_step`` is the class no composition term charges. A row proven live
#: during no declared engine step is not a term the composition forgot; it is
#: priced beside the composition, because an engine still has to fit its
#: startup peak on a box where no step ever reaches it.
LIFETIME_CLASSES = ("resident", "activation", "scratch", "non_step")

#: The domains an off-step price depends on. A peak over rows the history join
#: may have missed, or whose external bytes were never closed, is a floor
#: wearing a total's name.
NON_STEP_PEAK_DOMAINS = ("history_join", "external_closure")

#: The states a capture may declare about its own step coverage. Only
#: ``complete`` licenses the off-step classification: under ``partial`` an
#: allocation live during no *declared* step may still be live during an
#: undeclared one, so "off-step" stops being a proof and becomes a guess, and
#: ``unobserved`` declares no step at all.
STEP_COVERAGE_STATES = ("complete", "partial", "unobserved")
ADMITTED_STEP_COVERAGE = "complete"

#: The placement obligation, spelled exactly as the producer freezes it. This
#: is a contract string a gate reads, not prose, so it is compared verbatim.
PLACEMENT_OBLIGATION = "max(scalar_budget_bytes, non_step_transient_peak_bytes)"

#: How the partition came by its domain states. The producer derives them from
#: the ledger, and it also accepts a caller handing them in so its own
#: arithmetic stays testable. A handed-in table closes every domain because a
#: caller said so, which is the `qualified: true` failure this whole design
#: exists to prevent, so this consumer admits only the derived spelling. The
#: producer records the fact; declining to read it is this consumer's job.
DOMAINS_SOURCES = ("derived", "supplied")
ADMITTED_DOMAINS_SOURCE = "derived"

SUPPORTED_SCOPE = {
    "allocation_scope": "gpu_allocations_only",
    "topology": "tp1_single_device_resident_eager",
    "invariance": "one complete assignment, one row per unit",
}
SUPPORTED_EXECUTION = {"graph_mode": "eager", "residency": "resident", "topology": "tp1"}
SUPPORTED_ARGUMENT_DOMAIN_STATUS = ("unavailable", "observed")

_TOP_FIELDS = tuple(sorted(ENVELOPE_MEMBERS + ("schema",)))
_IDENTITY_FIELDS = ("capture_sha256", "fixture_provenance", "run")
#: The rank scope a capture taken on one rank of a tensor-parallel world adds to
#: its run identity. Optional and all-or-nothing: a scalar capture names only
#: the device it observed, and a capture that names a rank names the world it
#: belongs to, so a per-rank charge can bind each rank's own four fixed terms
#: to that rank's own sealed report instead of to a redistribution of a world
#: total.
_RUN_RANK_FIELDS = ("rank", "world_size")
_RUN_FIELDS = ("assignment_sha256", "canonical_units_sha256", "configuration_sha256",
               "device_id", "device_uuid", "model_sha256", "runtime_manifest_sha256",
               "schema", "workload_sha256")
_RUN_SCOPED_FIELDS = tuple(sorted(set(_RUN_FIELDS) | set(_RUN_RANK_FIELDS)))
_RUN_DIGESTS = ("assignment_sha256", "canonical_units_sha256", "configuration_sha256",
                "model_sha256", "runtime_manifest_sha256", "workload_sha256")
_EXECUTION_FIELDS = ("graph_mode", "residency", "topology")
_REFERENCE_FIELDS = ("canonical_census", "runtime_binding", "selected_rows")
#: The boundary the partition was classified against (design §2.2). Optional
#: on the wire; compared verbatim by the admission gate.
REFERENCE_BOUNDARY_FIELD = "transient_charge_boundary"
#: The reserved-extent witness (design §3.7): both allocator readings at one
#: instant. Optional on the wire; a boundary that requires it refuses by name.
RESERVATION_SLACK_FIELD = "reservation_slack"
RESERVATION_SLACK_FIELDS = ("allocated_bytes", "reserved_bytes", "sampled_at", "scope")
_WORKLOAD_FIELDS = ("calibration", "prompt_ids", "sampling")
_OBSERVATION_FIELDS = ("artifacts", "capture_sha256", "checkpoints", "cuda_argument_domains",
                       "external_native_peak_bytes", "issues", "kv_observations",
                       "observer_qualification", "owner_views", "runtime_provenance_relation",
                       "timing_captures", "torch_allocations",
                       "step_coverage", "step_intervals",
                       "torch_observed_live_peak_bytes", "torch_observed_live_peak_scope",
                       "unattributed_external_records", "worker_startup_records")
#: A declared step, named and bounded by history index. The issue text spells
#: these ``begin_trace_index``/``end_trace_index``; the producer emits
#: ``begin_index``/``end_index`` and the producer is what travels.
_STEP_INTERVAL_FIELDS = ("begin_index", "end_index", "step_id")
#: Two shapes, one field apart. A capture that declared steps says what its
#: coverage claim is scoped to; a capture that declared none carries no scope
#: because there is nothing for one to be about. ``_object`` demands an exact
#: field set, so the shape is selected by the state rather than unioned.
_STEP_COVERAGE_FIELDS = ("declared", "executed", "reason", "scope", "state")
_UNOBSERVED_COVERAGE_FIELDS = ("declared", "executed", "reason", "state")
_ARGUMENT_DOMAIN_FIELDS = ("handled_api_keys", "host_allocations", "host_mappings", "issues",
                           "null_device_frees", "scope", "status")
_ALLOCATION_FIELDS = ("address", "allocate_index", "allocation_id",
                      "allocator_block_bytes_observed", "bytes", "free_completed_index",
                      "free_requested_index", "generation", "lifetime_scope",
                      "observed_categories", "observed_owners", "scope_stack",
                      "unit_invocation")
_CHECKPOINT_FIELDS = ("label", "owner_count", "pinned_host_storages", "storages",
                      "trace_index", "unique_owned_storage_bytes",
                      "unique_pinned_host_backing_bytes", "unmatched_storage_observations")
_STORAGE_FIELDS = ("address", "allocation_id", "bytes", "category", "owner_categories", "owners")
_PARTITION_FIELDS = ("capture_sha256", "domains", "domains_source", "identity", "membership",
                     "non_step_allocations", "non_step_transient_peak_bytes",
                     "non_step_transient_peak_scope", "schema", "scope", "terms",
                     "uncharged_allocations", "unclassified_allocations", "units")
_MEMBERSHIP_FIELDS = ("allocate_index", "allocation_id", "bytes", "free_completed_index",
                      "lifetime_class", "owner_class", "unit")
_UNCLASSIFIED_FIELDS = ("allocation_id", "bytes", "lifetime_scope", "observed_categories", "reason")
#: A classified row that no composition term charges. It carries its cell and
#: its unit, which is what the consumer recomputes, plus the producer's prose
#: reason, which no check reads.
_UNCHARGED_FIELDS = ("allocation_id", "bytes", "lifetime_class", "owner_class", "reason", "unit")
_DERIVED_FIELDS = ("non_step_transient_peak_bytes", "non_step_transient_peak_scope",
                   "placement_obligation", "scalar_budget_bytes", "scope", "terms")
_SCOPE_FIELDS = ("allocation_scope", "expressible", "invariance",
                 "non_step_allocation_count", "step_coverage", "topology",
                 "unavailable_terms", "uncharged_allocation_count",
                 "unclassified_allocation_count")
_DOMAIN_FIELDS = ("evidence", "reason", "state")
#: What v2 adds to each member, by name. ``observer_allocations`` is the
#: producer's own observer's allocations, set aside from every class; this
#: consumer defines no shape for one, so a non-empty list refuses by name.
_PARTITION_FIELDS_V2 = _PARTITION_FIELDS + ("observer_allocations",)
_SCOPE_FIELDS_V2 = _SCOPE_FIELDS + ("observer_allocation_count",)
_DERIVED_FIELDS_V2 = _DERIVED_FIELDS + ("admission", "fixed_resources", "timing_terms")


# --------------------------------------------------------------------------
# Reader. Structural faults raise before any arithmetic runs, exactly as the
# producer schema declares: unknown fields, missing fields, duplicate JSON
# keys, nonfinite values, negative sizes, booleans where an integer is
# declared, duplicate IDs and unknown enum values all refuse here.
# --------------------------------------------------------------------------

def _list(value: Any, where: str) -> list:
    if not isinstance(value, list):
        raise RuntimePriceError(f"{where}: expected a list")
    return value


def _bool(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise RuntimePriceError(f"{where}: expected a boolean")
    return value


def _index(value: Any, where: str) -> int:
    """A trace index or byte count. ``True`` is an ``int`` in Python; it is not
    one here, and ``json.loads`` accepts ``NaN``/``Infinity`` unless told not
    to, which :class:`ArtifactReader` is."""
    return _integer(value, where)


def _optional_index(value: Any, where: str):
    return None if value is None else _index(value, where)


def _optional_string(value: Any, where: str):
    return None if value is None else _string(value, where)


def _required(value: Any, fields: Sequence[str], where: str) -> Mapping:
    """A mapping that must carry these coordinates, and may carry others.

    The two observations that carry the runtime's own resolved descriptors are
    the exception to this module's closed field sets: their extra fields belong
    to the runtime, and refusing them would refuse every real capture. The
    coordinates this consumer recomputes from are required and validated; the
    rest travel as evidence.
    """
    if not isinstance(value, Mapping):
        raise RuntimePriceError(f"{where}: expected an object")
    missing = [name for name in fields if name not in value]
    if missing:
        raise RuntimePriceError(f"{where}: missing required fields {sorted(missing)}")
    return value


def _string_list(value: Any, where: str, *, unique: bool = True) -> list[str]:
    items = [_string(item, where + " entry") for item in _list(value, where)]
    if unique and len(set(items)) != len(items):
        raise RuntimePriceError(f"{where}: duplicate entry")
    return items


def _enum(value: Any, allowed: Sequence[str], where: str) -> str:
    text = _string(value, where)
    if text not in allowed:
        raise RuntimePriceError(f"{where}: unknown value {text!r}")
    return text


def _domain_record(value: Any, where: str) -> Mapping:
    record = _object(value, _DOMAIN_FIELDS, where)
    state = _enum(record["state"], DOMAIN_STATES, where + " state")
    evidence = _string_list(record["evidence"], where + " evidence")
    _optional_string(record["reason"], where + " reason")
    # A domain that closes with no evidence, or carries evidence without
    # closing, is a status flag rather than an observation.
    if bool(evidence) != (state == "closed"):
        raise RuntimePriceError(f"{where}: evidence must be present exactly when the domain is closed")
    if (record["reason"] is None) != (state == "closed"):
        raise RuntimePriceError(f"{where}: a reason is required exactly when the domain is not closed")
    return record


def _domains(value: Any, where: str) -> dict[str, Mapping]:
    table = _object(value, DOMAINS, where)
    return {name: _domain_record(table[name], f"{where} {name}") for name in DOMAINS}


def _unit_charges(value: Any, where: str) -> Mapping[str, int] | None:
    """A per-unit charge table, or null when the term is not expressible."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimePriceError(f"{where}: expected a per-unit charge table")
    return {_string(unit, where + " unit"): _index(charge, f"{where} {unit}")
            for unit, charge in value.items()}


def _terms(value: Any, where: str) -> dict[str, Any]:
    table = _object(value, TERMS, where)
    return {name: (_unit_charges(table[name], f"{where} {name}") if name in PER_UNIT_TERMS
                   else _optional_index(table[name], f"{where} {name}"))
            for name in TERMS}


def _scope(value: Any, where: str, *, schema: str = REPORT_SCHEMA) -> Mapping:
    scope = _object(value, _SCOPE_FIELDS_V2 if schema == REPORT_SCHEMA_V2 else _SCOPE_FIELDS,
                    where)
    if schema == REPORT_SCHEMA_V2:
        _index(scope["observer_allocation_count"], where + " observer allocation count")
    for key, supported in SUPPORTED_SCOPE.items():
        _string(scope[key], f"{where} {key}")
        if scope[key] != supported:
            raise RuntimePriceError(
                f"{where}: unsupported {key} {scope[key]!r}; this consumer covers only {supported!r}")
    _bool(scope["expressible"], where + " expressible")
    _index(scope["unclassified_allocation_count"], where + " unclassified allocation count")
    _index(scope["uncharged_allocation_count"], where + " uncharged allocation count")
    _index(scope["non_step_allocation_count"], where + " non-step allocation count")
    _enum(scope["step_coverage"], STEP_COVERAGE_STATES, where + " step coverage")
    for name in _string_list(scope["unavailable_terms"], where + " unavailable terms"):
        if name not in TERMS:
            raise RuntimePriceError(f"{where}: unknown term {name!r}")
    return scope


def _step_coverage(value: Any, where: str) -> Mapping:
    """The capture's coverage claim over its own engine steps.

    The state selects the field set, because a capture that declared no step
    carries no ``scope`` for a claim it never made. Reading the state first is
    not a relaxation: an object whose state is unknown, or which carries no
    state at all, refuses before either shape is tried.
    """
    if not isinstance(value, Mapping) or "state" not in value:
        raise RuntimePriceError(f"{where}: expected a step coverage claim naming its state")
    state = _enum(value["state"], STEP_COVERAGE_STATES, where + " state")
    fields = _UNOBSERVED_COVERAGE_FIELDS if state == "unobserved" else _STEP_COVERAGE_FIELDS
    coverage = _object(value, fields, where)
    _optional_index(coverage["declared"], where + " declared")
    _optional_index(coverage["executed"], where + " executed")
    _optional_string(coverage["reason"], where + " reason")
    if state != "unobserved":
        _string(coverage["scope"], where + " scope")
    elif coverage["declared"] is not None or coverage["executed"] is not None:
        raise RuntimePriceError(
            f"{where}: an unobserved coverage claim counts declared or executed steps")
    # A reason is owed exactly when the coverage does not license the off-step
    # classification, mirroring the rule every domain record already follows.
    if (coverage["reason"] is None) != (state == ADMITTED_STEP_COVERAGE):
        raise RuntimePriceError(
            f"{where}: a reason is required exactly when step coverage is not "
            f"{ADMITTED_STEP_COVERAGE!r}")
    return coverage


def _step_intervals(value: Any, where: str):
    """The declared engine steps, or null when the capture declared none.

    Ordering and separation are checked here rather than assumed. Two declared
    steps that overlap contradict each other about which step an allocation was
    live during, so a declared interval spanning another step's extent refuses
    instead of being classified against either one.
    """
    if value is None:
        return None
    rows, seen = [], set()
    for item in _list(value, where):
        row = _object(item, _STEP_INTERVAL_FIELDS, where + " row")
        step_id = _string(row["step_id"], where + " step id")
        begin = _index(row["begin_index"], where + " begin index")
        end = _index(row["end_index"], where + " end index")
        if step_id in seen:
            raise RuntimePriceError(f"{where}: duplicate step {step_id!r}")
        seen.add(step_id)
        if begin >= end:
            raise RuntimePriceError(f"{where}: step {step_id!r} is reversed or unclosed")
        rows.append((begin, end, step_id))
    for (first_begin, first_end, first_id), (second_begin, _, second_id) in zip(rows, rows[1:]):
        if second_begin < first_begin:
            raise RuntimePriceError(f"{where}: step {second_id!r} is declared out of order")
        if second_begin < first_end:
            raise RuntimePriceError(
                f"{where}: step {first_id!r} spans the extent of step {second_id!r}, so an "
                "allocation live in one is live in the other and neither bounds it")
    return rows


def _recomputed_coverage_state(intervals, executed: Any) -> str:
    """The coverage state the declared counts support.

    ``complete`` is a claim like any other: it is true when the capture said
    how many steps it executed and declared an interval for every one of them,
    and a report that asserts it on other counts is asserting the very license
    the classification runs on.
    """
    if intervals and executed is not None and executed == len(intervals):
        return ADMITTED_STEP_COVERAGE
    return "partial"


def _run_identity(value: Any, where: str) -> Mapping:
    """One capture's run identity, optionally scoped to a rank of a world.

    A scalar capture names the device it observed and nothing else. A capture
    taken on one rank of a tensor-parallel world additionally names which rank
    and world it observed, because a rank's own fixed terms are only that
    rank's if the report that carries them says which rank it is. The scope is
    all-or-nothing and the two shapes are separate field sets, so a document
    cannot carry a rank without a world for a consumer to check it against.
    """
    if not isinstance(value, Mapping):
        raise RuntimePriceError(f"{where}: expected an object")
    fields = set(value)
    if fields == set(_RUN_FIELDS):
        run = _object(value, _RUN_FIELDS, where)
    elif fields == set(_RUN_SCOPED_FIELDS):
        run = _object(value, _RUN_SCOPED_FIELDS, where)
        world = _index(run["world_size"], where + " world size")
        rank = _index(run["rank"], where + " rank")
        if world < 1 or rank >= world:
            raise RuntimePriceError(
                f"{where}: rank {rank} is not inside a world of {world}")
    else:
        raise RuntimePriceError(
            f"{where}: expected exactly fields {sorted(_RUN_FIELDS)} or "
            f"{sorted(_RUN_SCOPED_FIELDS)}")
    _equal(run["schema"], IDENTITY_SCHEMA, where + " schema")
    for key in _RUN_DIGESTS:
        _sha(run[key], f"{where} {key}")
    _index(run["device_id"], where + " device id")
    _string(run["device_uuid"], where + " device uuid")
    return run


def _allocation(value: Any, where: str) -> Mapping:
    row = _object(value, _ALLOCATION_FIELDS, where)
    address = _index(row["address"], where + " address")
    generation = _integer(row["generation"], where + " generation", minimum=1)
    allocation_id = _string(row["allocation_id"], where + " allocation id")
    _index(row["bytes"], where + " bytes")
    allocate = _index(row["allocate_index"], where + " allocate index")
    requested = _optional_index(row["free_requested_index"], where + " free requested index")
    completed = _optional_index(row["free_completed_index"], where + " free completed index")
    _enum(row["lifetime_scope"], LIFETIME_SCOPES, where + " lifetime scope")
    _string_list(row["observed_categories"], where + " observed categories")
    _string_list(row["observed_owners"], where + " observed owners")
    _string_list(row["scope_stack"], where + " scope stack", unique=False)
    _optional_string(row["unit_invocation"], where + " unit invocation")
    for block in _list(row["allocator_block_bytes_observed"], where + " allocator blocks"):
        _index(block, where + " allocator block bytes")
    # A pointer identifies a lifetime only with its generation, and the
    # composed identifier is what every other member refers to.
    device, _, rest = allocation_id.partition(":")
    if rest != f"{address}:{generation}" or not device.isdigit():
        raise RuntimePriceError(f"{where}: allocation id {allocation_id!r} does not name its device, address and generation")
    if (completed is None) != (requested is None):
        raise RuntimePriceError(f"{where}: a completed free needs its requested free, and the reverse")
    if completed is not None and not allocate < requested <= completed:
        raise RuntimePriceError(f"{where}: free indices must follow the allocation and settle in order")
    return row


def _storage(value: Any, where: str) -> Mapping:
    row = _object(value, _STORAGE_FIELDS, where)
    _index(row["address"], where + " address")
    _index(row["bytes"], where + " bytes")
    _string(row["allocation_id"], where + " allocation id")
    _string(row["category"], where + " category")
    owners = _string_list(row["owners"], where + " owners")
    categories = _object(row["owner_categories"], tuple(owners), where + " owner categories")
    for owner in owners:
        _string(categories[owner], where + " owner category")
    return row


def _checkpoint(value: Any, where: str) -> Mapping:
    row = _object(value, _CHECKPOINT_FIELDS, where)
    _string(row["label"], where + " label")
    _index(row["trace_index"], where + " trace index")
    _index(row["owner_count"], where + " owner count")
    _index(row["unique_owned_storage_bytes"], where + " owned storage bytes")
    _index(row["unique_pinned_host_backing_bytes"], where + " pinned host backing bytes")
    _list(row["pinned_host_storages"], where + " pinned host storages")
    _list(row["unmatched_storage_observations"], where + " unmatched storage observations")
    storages = [_storage(item, where + " storage") for item in _list(row["storages"], where + " storages")]
    if len({item["allocation_id"] for item in storages}) != len(storages):
        raise RuntimePriceError(f"{where}: duplicate storage identity")
    return row


def _reservation_slack(value: Any, where: str) -> Mapping:
    """One instant's two allocator readings, so the slack is recomputable.

    ``reserved_bytes`` is what the caching allocator holds in segments and what
    the box holds; ``allocated_bytes`` is the live block extent at the same
    instant. The consumer recomputes their difference and never reads a slack
    the producer wrote.
    """
    record = _object(value, RESERVATION_SLACK_FIELDS, where)
    allocated = _index(record["allocated_bytes"], where + " allocated bytes")
    reserved = _index(record["reserved_bytes"], where + " reserved bytes")
    if reserved < allocated:
        raise RuntimePriceError(
            f"{where}: reserved bytes {reserved} are below allocated bytes {allocated}; every "
            "allocated block lives inside a reserved segment")
    _string(record["sampled_at"], where + " sampled at")
    _string(record["scope"], where + " scope")
    return record


def _observations(value: Any, where: str, *, schema: str = REPORT_SCHEMA) -> Mapping:
    # `reservation_slack` is optional on the wire for the same reason the
    # reference boundary is: a report emitted before it existed still reads,
    # and a boundary that requires it refuses its absence by name.
    fields = _OBSERVATION_FIELDS + (
        (RESERVATION_SLACK_FIELD,) if isinstance(value, Mapping) and RESERVATION_SLACK_FIELD in value else ())
    observations = _object(value, fields, where)
    if RESERVATION_SLACK_FIELD in observations and observations[RESERVATION_SLACK_FIELD] is not None:
        _reservation_slack(observations[RESERVATION_SLACK_FIELD], where + " reservation slack")
    _sha(observations["capture_sha256"], where + " capture digest")
    _list(observations["artifacts"], where + " artifacts")
    _list(observations["issues"], where + " issues")
    _list(observations["unattributed_external_records"], where + " unattributed external records")
    _optional_index(observations["external_native_peak_bytes"], where + " external native peak bytes")
    _index(observations["torch_observed_live_peak_bytes"], where + " observed live peak bytes")
    _string(observations["torch_observed_live_peak_scope"], where + " observed live peak scope")
    domains = _object(observations["cuda_argument_domains"], _ARGUMENT_DOMAIN_FIELDS,
                      where + " argument domains")
    _enum(domains["status"], SUPPORTED_ARGUMENT_DOMAIN_STATUS, where + " argument domain status")
    _string(domains["scope"], where + " argument domain scope")
    _string_list(domains["handled_api_keys"], where + " handled API keys")
    for key in ("host_allocations", "host_mappings", "issues", "null_device_frees"):
        _list(domains[key], where + " argument domain " + key)
    allocations = [_allocation(item, where + " allocation")
                   for item in _list(observations["torch_allocations"], where + " allocations")]
    if len({item["allocation_id"] for item in allocations}) != len(allocations):
        raise RuntimePriceError(f"{where}: a pointer generation is reused by two allocations")
    for item in _list(observations["checkpoints"], where + " checkpoints"):
        _checkpoint(item, where + " checkpoint")
    coverage = _step_coverage(observations["step_coverage"], where + " step coverage")
    intervals = _step_intervals(observations["step_intervals"], where + " step intervals")
    if (intervals is None) != (coverage["state"] == "unobserved"):
        raise RuntimePriceError(
            f"{where}: step intervals are carried exactly when the coverage state is not "
            "'unobserved'")
    if intervals is not None:
        if coverage["declared"] != len(intervals):
            raise RuntimePriceError(
                f"{where}: step coverage declares {coverage['declared']!r} steps where the "
                f"report carries {len(intervals)}")
        recomputed = _recomputed_coverage_state(intervals, coverage["executed"])
        if coverage["state"] != recomputed:
            raise RuntimePriceError(
                f"{where}: step coverage claims {coverage['state']!r} where its own declared "
                f"and executed counts support {recomputed!r}")
    for name in OWED_OBSERVATIONS:
        if observations[name] is None:
            continue
        if schema != REPORT_SCHEMA_V2:
            raise RuntimePriceError(
                f"{where}: {name} is not null, but this schema version defines no shape for it, "
                "so nothing here can recompute a term or close a domain from it")
        # v2 carries these as the producer's own records. They are read as
        # carried -- a record naming its own schema -- and nothing below
        # recomputes a term or closes a domain from one.
        record = observations[name]
        if not isinstance(record, Mapping) or not isinstance(record.get("schema"), str):
            raise RuntimePriceError(
                f"{where}: {name} is carried but names no schema, so it cannot be identified")
    startup = observations["worker_startup_records"]
    if startup is not None:
        for item in _list(startup, where + " worker startup records"):
            record = _object(item, WORKER_STARTUP_RECORD_FIELDS, where + " worker startup record")
            _index(record["rank"], where + " worker startup rank")
            _index(record["memory_allocated_bytes"], where + " worker startup allocated bytes")
            _index(record["receipt_resident_bytes"], where + " worker startup receipt resident bytes")
            _index(record["workspace_resident_bytes"],
                   where + " worker startup workspace resident bytes")
            _bool(record["workspace_locked"], where + " worker startup workspace locked")
            _string(record["scope"], where + " worker startup scope")
    kv = observations["kv_observations"]
    if kv is not None:
        for item in _list(kv, where + " kv observations"):
            record = _required(item, KV_OBSERVATION_REQUIRED_FIELDS, where + " kv observation")
            _index(record["num_blocks"], where + " kv blocks")
            _bool(record["runtime_admission"], where + " kv runtime admission")
            _string(record["scope"], where + " kv scope")
            for page in _list(record["group_page_size_bytes"], where + " kv group page sizes"):
                _index(page, where + " kv group page size")
            limits = _required(record["resolved_limits"], KV_RESOLVED_LIMIT_FIELDS,
                               where + " kv resolved limits")
            _index(limits["max_num_batched_tokens"], where + " kv batched tokens")
            _index(limits["max_num_seqs"], where + " kv sequences")
            storage_set = _required(record["storage"], KV_STORAGE_SET_FIELDS,
                                    where + " kv storage observation")
            _index(storage_set["unique_physical_storage_bytes"],
                   where + " kv unique physical storage bytes")
            for storage in _list(storage_set["storages"], where + " kv storages"):
                row = _object(storage, KV_STORAGE_FIELDS, where + " kv storage")
                _string(row["device_type"], where + " kv storage device type")
                _index(row["device_id"], where + " kv storage device id")
                _index(row["address"], where + " kv storage address")
                _index(row["bytes"], where + " kv storage bytes")
                _string_list(row["owners"], where + " kv storage owners")
    return observations


def _resolved_evidence(domains: Mapping, observations: Mapping, where: str) -> None:
    """Every evidence id names an observation this envelope actually carries.

    A domain citing an id nothing answers to is a status flag with a footnote:
    the consumer is told where to look and finds nothing there.
    """
    for name in DOMAINS:
        for item in domains[name]["evidence"]:
            if item not in observations:
                raise RuntimePriceError(
                    f"{where} {name}: evidence {item!r} names no observation this report carries")


def _partition(value: Any, where: str, *, schema: str = REPORT_SCHEMA) -> Mapping:
    partition = _object(value, _PARTITION_FIELDS_V2 if schema == REPORT_SCHEMA_V2
                        else _PARTITION_FIELDS, where)
    if schema == REPORT_SCHEMA_V2:
        observer = _list(partition["observer_allocations"], where + " observer allocations")
        if observer:
            raise RuntimePriceError(
                f"{where}: carries {len(observer)} observer allocations, and this consumer "
                "defines no shape for one, so it cannot tell them from the classes it recomputes")
        if partition["scope"].get("observer_allocation_count") != len(observer):
            raise RuntimePriceError(
                f"{where}: scope claims another observer allocation count than the partition "
                "carries")
    _equal(partition["schema"], PARTITION_SCHEMA, where + " schema")
    _sha(partition["capture_sha256"], where + " capture digest")
    _run_identity(partition["identity"], where + " identity")
    _domains(partition["domains"], where + " domains")
    # A partition whose domains were handed in by a caller has had nothing
    # checked: every domain closes because an argument said so, and every term
    # is then emitted on that word. It is a well-formed value of the frozen
    # schema and it is refused by name, exactly as an unsupported topology is.
    source = _enum(partition["domains_source"], DOMAINS_SOURCES, where + " domains source")
    if source != ADMITTED_DOMAINS_SOURCE:
        raise RuntimePriceError(
            f"{where}: domains_source is {source!r}; this consumer reads only a partition whose "
            f"domains are {ADMITTED_DOMAINS_SOURCE!r}, because supplied domains close on a "
            f"caller's word rather than on evidence")
    _terms(partition["terms"], where + " terms")
    _scope(partition["scope"], where + " scope", schema=schema)
    _string_list(partition["units"], where + " units")
    rows = []
    for item in _list(partition["membership"], where + " membership"):
        row = _object(item, _MEMBERSHIP_FIELDS, where + " membership row")
        _string(row["allocation_id"], where + " membership allocation id")
        _index(row["bytes"], where + " membership bytes")
        _index(row["allocate_index"], where + " membership allocate index")
        _optional_index(row["free_completed_index"], where + " membership free completed index")
        _enum(row["owner_class"], OWNER_CLASSES, where + " membership owner class")
        _enum(row["lifetime_class"], LIFETIME_CLASSES, where + " membership lifetime class")
        _optional_string(row["unit"], where + " membership unit")
        rows.append(row)
    unclassified = []
    for item in _list(partition["unclassified_allocations"], where + " unclassified allocations"):
        row = _object(item, _UNCLASSIFIED_FIELDS, where + " unclassified row")
        _string(row["allocation_id"], where + " unclassified allocation id")
        _index(row["bytes"], where + " unclassified bytes")
        _enum(row["lifetime_scope"], LIFETIME_SCOPES, where + " unclassified lifetime scope")
        _string_list(row["observed_categories"], where + " unclassified observed categories")
        _string(row["reason"], where + " unclassified reason")
        unclassified.append(row)
    non_step = []
    for item in _list(partition["non_step_allocations"], where + " non-step allocations"):
        row = _object(item, _MEMBERSHIP_FIELDS, where + " non-step row")
        _string(row["allocation_id"], where + " non-step allocation id")
        _index(row["bytes"], where + " non-step bytes")
        _index(row["allocate_index"], where + " non-step allocate index")
        _optional_index(row["free_completed_index"], where + " non-step free completed index")
        # Off-step rows are the one place a null owner class is well formed: no
        # composition term charges them, so none reads the class (Tessera #478).
        if row["owner_class"] is not None:
            _enum(row["owner_class"], OWNER_CLASSES, where + " non-step owner class")
        _equal(row["lifetime_class"], "non_step", where + " non-step lifetime class")
        _optional_string(row["unit"], where + " non-step unit")
        non_step.append(row)
    _optional_index(partition["non_step_transient_peak_bytes"],
                    where + " non-step transient peak bytes")
    _string(partition["non_step_transient_peak_scope"], where + " non-step transient peak scope")
    identities = [row["allocation_id"] for row in rows + unclassified + non_step]
    if len(set(identities)) != len(identities):
        raise RuntimePriceError(f"{where}: an allocation is partitioned more than once")
    # Uncharged rows are classified rows the seven terms do not reach, so each
    # one restates a membership row rather than adding an allocation.
    uncharged_ids = []
    for item in _list(partition["uncharged_allocations"], where + " uncharged allocations"):
        row = _object(item, _UNCHARGED_FIELDS, where + " uncharged row")
        _string(row["allocation_id"], where + " uncharged allocation id")
        _index(row["bytes"], where + " uncharged bytes")
        _enum(row["owner_class"], OWNER_CLASSES, where + " uncharged owner class")
        _enum(row["lifetime_class"], LIFETIME_CLASSES, where + " uncharged lifetime class")
        _optional_string(row["unit"], where + " uncharged unit")
        _string(row["reason"], where + " uncharged reason")
        uncharged_ids.append(row["allocation_id"])
    if len(set(uncharged_ids)) != len(uncharged_ids):
        raise RuntimePriceError(f"{where}: an allocation is named uncharged more than once")
    orphans = sorted(set(uncharged_ids) - {row["allocation_id"] for row in rows})
    if orphans:
        raise RuntimePriceError(
            f"{where}: uncharged allocation {orphans[0]!r} is not a classified membership row")
    return partition


def read_full_engine_resource_report(reference: Mapping, *, root: Path) -> Mapping:
    """Rehash and structurally validate one report, before any arithmetic.

    ``reference`` is the existing ``{path, sha256}`` artifact form, so a
    tampered file is caught by its digest and a tampering test must restate
    the outer hash to reach the semantic checks below.
    """
    _, report = ArtifactReader(Path(root)).json(reference, "full-engine resource report")
    schema = report.get("schema")
    if schema not in REPORT_SCHEMAS:
        raise RuntimePriceError(
            f"full-engine resource report: unsupported schema {schema!r}; "
            f"this consumer reads only {', '.join(REPORT_SCHEMAS)}")
    missing = [name for name in ENVELOPE_MEMBERS if name not in report]
    if missing:
        raise RuntimePriceError("full-engine resource report: missing envelope member "
                                + ", ".join(sorted(missing)))
    _object(report, _TOP_FIELDS, "full-engine resource report")
    identity = _object(report["identity"], _IDENTITY_FIELDS, "report identity")
    _sha(identity["capture_sha256"], "report identity capture digest")
    _optional_string(identity["fixture_provenance"], "report fixture provenance")
    _run_identity(identity["run"], "report identity run")
    execution = _object(report["execution"], _EXECUTION_FIELDS, "report execution")
    for key, supported in SUPPORTED_EXECUTION.items():
        if _string(execution[key], "report execution " + key) != supported:
            raise RuntimePriceError(
                f"report execution: unsupported {key} {execution[key]!r}; the scalar device budget "
                f"covers only {supported!r}")
    # The boundary name is optional on the wire so every report emitted before
    # it existed still reads; a report that carries it names the owner map its
    # partition was classified against, and the admission gate compares it
    # verbatim to the table's (`transient_charge_boundary.require_boundary`).
    reference_fields = _REFERENCE_FIELDS + (
        (REFERENCE_BOUNDARY_FIELD,) if REFERENCE_BOUNDARY_FIELD in report["reference"] else ())
    reference_member = _object(report["reference"], reference_fields, "report reference")
    _list(reference_member["selected_rows"], "report reference selected rows")
    if REFERENCE_BOUNDARY_FIELD in reference_member:
        _optional_string(reference_member[REFERENCE_BOUNDARY_FIELD],
                         "report reference transient charge boundary")
    workload = _object(report["workload"], _WORKLOAD_FIELDS, "report workload")
    _list(workload["prompt_ids"], "report workload prompt ids")
    observations = _observations(report["observations"], "report observations", schema=schema)
    partition = _partition(report["partition"], "report partition", schema=schema)
    _resolved_evidence(partition["domains"], observations, "report partition domain")
    derived = _object(report["derived"],
                      _DERIVED_FIELDS_V2 if schema == REPORT_SCHEMA_V2 else _DERIVED_FIELDS,
                      "report derived")
    if schema == REPORT_SCHEMA_V2:
        # The producer's own admission verdict, fixed-resource composition and
        # timing terms. Carried, and read by nothing here: every number this
        # consumer admits is its own recomputation.
        for name in ("admission", "fixed_resources", "timing_terms"):
            if not isinstance(derived[name], Mapping):
                raise RuntimePriceError(f"report derived {name}: expected an object")
    _terms(derived["terms"], "report derived terms")
    _scope(derived["scope"], "report derived scope", schema=schema)
    _optional_index(derived["scalar_budget_bytes"], "report derived scalar budget bytes")
    _optional_index(derived["non_step_transient_peak_bytes"],
                    "report derived non-step transient peak bytes")
    _string(derived["non_step_transient_peak_scope"],
            "report derived non-step transient peak scope")
    _string(derived["placement_obligation"], "report derived placement obligation")
    return report


# --------------------------------------------------------------------------
# Independent recomputation. Nothing below reads `derived`.
# --------------------------------------------------------------------------

def _declared_steps(observations: Mapping):
    """The engine steps this classification may rely on, or ``None``.

    Index pairs come back only when the capture declared an interval for every
    step it executed. Partial coverage is not a smaller set of steps: an
    allocation from an undeclared step is live during a step nobody declared,
    so "live during no declared step" stops being a proof. Partial and
    unobserved therefore behave identically, which is how this consumer behaved
    before any step could be declared at all.
    """
    coverage = observations["step_coverage"]
    if coverage["state"] != ADMITTED_STEP_COVERAGE:
        return None
    return [(row["begin_index"], row["end_index"]) for row in observations["step_intervals"]]


def _allocations_whose_unit_crosses_a_step(observations: Mapping) -> list:
    """Allocation ids proving a unit invocation crossed a declared step boundary.

    A unit runs inside one engine step, so the producer refuses a unit interval
    that overlaps a declared step without being contained in it. The report
    carries no unit intervals, so that refusal cannot be re-read directly. It
    can still be derived: an ``inside_unit`` allocation is allocated and freed
    within its own unit interval, so one that overlaps a declared step without
    being contained in one proves that its unit overlapped that step without
    being contained in it either. Naming the allocation is the closest this
    consumer can come to naming the interval.
    """
    steps = _declared_steps(observations)
    if steps is None:
        return []
    crossing = []
    for allocation in observations["torch_allocations"]:
        begin, end = allocation["allocate_index"], allocation["free_completed_index"]
        if allocation["lifetime_scope"] != "inside_unit" or end is None:
            continue
        if (any(step_begin < end and begin < step_end for step_begin, step_end in steps)
                and not any(step_begin <= begin and end <= step_end
                            for step_begin, step_end in steps)):
            crossing.append(allocation["allocation_id"])
    return crossing


def _classify(allocation: Mapping, steps) -> tuple:
    """Return ``(owner_class, lifetime_class, unit)`` or ``(None, None, reason)``.

    Ownership and lifetime are separate questions, and the lifetime one is
    asked first because its answer decides whether the ownership one is asked
    at all. An owner class is an invariance claim, and only a composition term
    reads one: a row proven live during no declared engine step is charged by
    no term, so it is classified off-step on its lifetime with
    ``owner_class`` ``None``. Every other row needs a supported class, because
    a term that charges it needs the claim. ``shared`` and ``unknown`` answer
    neither question (Tessera #478).
    """
    categories = list(allocation["observed_categories"])
    freed = allocation["free_completed_index"] is not None
    scope = allocation["lifetime_scope"]
    if any(label in UNSUPPORTED_OWNER_LABELS for label in categories):
        owner, owner_reason = None, ("shared or unknown ownership supplies neither "
                                     "classification nor invariance")
    elif len(categories) != 1 or categories[0] not in OWNER_CLASSES:
        owner, owner_reason = None, "no single supported owner category"
    else:
        owner, owner_reason = categories[0], None
    if not freed:
        # Never freed within the capture, so it is live at the terminal
        # boundary whatever scope it was allocated in. Resident bytes add.
        lifetime = "resident"
    elif scope == "inside_unit":
        lifetime = "scratch"
    elif scope == "escapes_unit":
        lifetime = "activation"
    elif steps is None:
        # Freed with no unit interval containing either end. Charging it as
        # fixed scratch assumes once per step and treating it as startup
        # assumes never again; both are fills, and this capture declares no
        # complete step boundary to decide between them.
        return None, None, ("freed outside every unit interval, and no complete declared step "
                            "boundary covers this capture, so charging it would assume either "
                            "once per step or never again")
    else:
        # With a complete boundary the question is answered by liveness, not by
        # the allocation index alone. A buffer allocated before a step and freed
        # inside it is live during that step and has to be charged; a buffer
        # whose whole lifetime sits between steps is live during none of them.
        begin, end = allocation["allocate_index"], allocation["free_completed_index"]
        if any(step_begin <= begin and end <= step_end for step_begin, step_end in steps):
            lifetime = "scratch"
        elif any(step_begin < end and begin < step_end for step_begin, step_end in steps):
            # Live across a step boundary: carried, exactly as a row that
            # outlives its unit is carried. `fixed_activation` and
            # `fixed_scratch` are separate additive terms, so this neither
            # double-counts the bytes nor drops them.
            lifetime = "activation"
        else:
            lifetime = "non_step"
    if lifetime != "non_step" and owner is None:
        # The row is one some term charges, so which term is a question this
        # consumer has to answer, and the owner class is how it is answered.
        return None, None, owner_reason
    # The unit an allocation is charged to is the **outermost** scope it was
    # made in, not the innermost. The producer reads `unit_invocation` from the
    # outermost containing interval and decides `lifetime_scope` against that
    # same interval, so charging the innermost would split a row's lifetime
    # basis from its charge. Unit intervals may nest, and only crossing ones
    # refuse, so the innermost pick also puts simultaneously live rows into two
    # per-unit buckets the composition then takes a maximum between. Outermost
    # intervals cannot overlap, so a maximum over them is a maximum over
    # genuine alternatives. `unit_invocation` names the same unit from the
    # other side, so a row where the two disagree is not a row this consumer
    # can charge anywhere.
    stack = allocation["scope_stack"]
    unit = stack[0] if stack else None
    invocation = allocation["unit_invocation"]
    if invocation is not None and invocation.rsplit(":", 1)[0] != (unit or ""):
        return None, None, "the scope stack and the unit invocation name different units"
    if unit is not None and invocation is None:
        return None, None, "an allocation inside a unit scope names no unit invocation"
    return owner, lifetime, unit


def _simultaneous_peak(allocations: Sequence[Mapping]) -> int:
    """The maximum simultaneous sum over the declared interval.

    A sum of per-allocation maxima is not a peak, and neither is a difference
    of two independent peaks. Frees settle before allocations at the same
    index, so a byte freed at ``k`` is not charged against one allocated at
    ``k``. For allocations that are never freed this equals their sum.
    """
    events: list[tuple[int, int, int]] = []
    for allocation in allocations:
        events.append((allocation["allocate_index"], 1, allocation["bytes"]))
        if allocation["free_completed_index"] is not None:
            events.append((allocation["free_completed_index"], 0, -allocation["bytes"]))
    live = peak = 0
    for _, _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        live += delta
        peak = max(peak, live)
    return peak


def _recompute_domains(observations: Mapping, identity: Mapping) -> dict[str, bool]:
    """Which domains this consumer can itself see closed, keyed by name.

    Four of the six now have a closing condition expressible in the raw
    observations this schema emits: the two ledger conditions below, and the
    two observation shapes that name the startup prefix and the KV backings.
    The remaining two -- ``provenance_admission`` and ``timing_partition`` --
    have none, so they are open here regardless of the state the report
    declares: a domain that closes because a field was truthy is a status flag,
    and the consumer would be reading the producer's word for what the producer
    did.
    """
    blocked_by_issues = bool(observations["issues"])
    closed = {name: False for name in DOMAINS}
    closed["history_join"] = (not blocked_by_issues
                              and not observations["unattributed_external_records"])
    closed["external_closure"] = (not blocked_by_issues
                                  and observations["external_native_peak_bytes"] is not None)
    closed["worker_startup"] = (not blocked_by_issues
                                and _worker_startup_observed(observations, identity))
    closed["cache_capacity"] = (not blocked_by_issues
                                and _cache_capacity_observed(observations))
    return closed


def _worker_startup_observed(observations: Mapping, identity: Mapping) -> bool:
    """Whether this rank's resident-after-load observation is present.

    The producer's own contract names the source: a
    ``torch.cuda.memory_allocated()`` sample taken after
    ``process_weights_after_loading`` and after ``lock_workspace()``, beside the
    receipt's ``resources.resident_bytes`` and the ``resident_bytes`` of the
    producer's own ``tessera.native_moe_workspace.v1`` record. The domain closes
    when the record names this report's own rank (the process that measured it,
    not the world it was configured for), the workspace is locked -- which is
    what makes its resident storage a stable charge -- and the ledger's
    fixed-owned resident rows sum to exactly the receipt's figure. That last
    equality is what ties the engine-side sample to the allocation ledger this
    consumer recomputes from, so neither side is trusted about the other; a zero
    charge is a real state and closes on the same equality rather than being
    refused for looking small.

    The allocator's own sample must cover the receipt's resident plus its
    workspace slots, which is the one direction the two sides can be compared
    without either being a claim about the other's internals.
    """
    records = observations["worker_startup_records"]
    if not records or len(records) != 1:
        return False
    record = records[0]
    measured_rank = identity["run"].get("rank")
    if measured_rank is not None and record["rank"] != measured_rank:
        return False
    if record["workspace_locked"] is not True:
        return False
    if record["memory_allocated_bytes"] < (record["receipt_resident_bytes"]
                                           + record["workspace_resident_bytes"]):
        return False
    ledger = sum(row["bytes"] for row in observations["torch_allocations"]
                 if list(row["observed_categories"]) == ["fixed"]
                 and row["free_completed_index"] is None)
    return ledger == record["receipt_resident_bytes"]


def _cache_capacity_observed(observations: Mapping) -> bool:
    """Whether the engine's own KV-pool accounting is present.

    ``fixed_kv`` is the engine's, not this operator's: the source is the block
    manager's resolved ``num_blocks`` and per-group page geometry beside the
    observer's own deduplicated physical ``storage`` block. The domain closes on
    the physical backings, not on a block-count formula: the distinct storages
    must not overlap, their own bytes must add up to
    ``unique_physical_storage_bytes``, and the ledger's ``kv``-owned resident
    rows must sum to that same extent. A hybrid or multi-group config therefore
    compares its actual backings rather than assuming every group's blocks are
    identical, and a pool of zero blocks is a real state rather than a refusal.

    ``runtime_admission`` is required and must be true: the intrusive snapshot
    pass the capture harness emits sets it false and names itself timing- and
    admission-ineligible in ``scope``, and closing this domain on that pass
    would be pricing bytes the harness warns are not admissible.
    """
    records = observations["kv_observations"]
    if not records or len(records) != 1:
        return False
    record = records[0]
    if record["runtime_admission"] is not True:
        return False
    limits = record["resolved_limits"]
    if limits["max_num_batched_tokens"] < 1 or limits["max_num_seqs"] < 1:
        return False
    pages = record["group_page_size_bytes"]
    if not pages or any(page < 1 for page in pages):
        return False
    if record["num_blocks"] < 0:
        return False
    storage_set = record["storage"]
    storages = storage_set["storages"]
    if bool(storages) != bool(record["num_blocks"]):
        return False
    spans = []
    for row in storages:
        if (row["device_type"] != "cuda" or row["address"] < 1 or row["bytes"] < 1
                or not row["owners"]):
            return False
        spans.append((row["device_type"], row["device_id"], row["address"],
                      row["address"] + row["bytes"]))
    ordered = sorted(spans)
    for first, second in zip(ordered, ordered[1:]):
        if (first[0], first[1]) == (second[0], second[1]) and first[3] > second[2]:
            return False
    if storage_set["unique_physical_storage_bytes"] != sum(row["bytes"] for row in storages):
        return False
    ledger = sum(row["bytes"] for row in observations["torch_allocations"]
                 if list(row["observed_categories"]) == ["kv"]
                 and row["free_completed_index"] is None)
    return ledger == storage_set["unique_physical_storage_bytes"]


def _recompute_membership(observations: Mapping) -> tuple[list[dict], list[dict], list[dict]]:
    """Split the observed allocations into charged, unclassified and off-step.

    The third list is classified and deliberately outside the composition: each
    row is proven, from the capture's own declared step intervals, to be live
    during no engine step, and the seven terms compose one step. Those rows are
    named and priced separately rather than nulling every term.
    """
    steps = _declared_steps(observations)
    membership, unclassified, non_step = [], [], []
    for allocation in observations["torch_allocations"]:
        owner, lifetime, detail = _classify(allocation, steps)
        if lifetime is None:
            # An unclassified row is one with no lifetime class or -- where a
            # term charges it -- no owner class. An off-step row carries a
            # lifetime and may carry no owner, and it is classified.
            unclassified.append({"allocation_id": allocation["allocation_id"],
                                 "bytes": allocation["bytes"], "reason": detail})
            continue
        entry = {"allocation_id": allocation["allocation_id"],
                 "allocate_index": allocation["allocate_index"],
                 "bytes": allocation["bytes"],
                 "free_completed_index": allocation["free_completed_index"],
                 "owner_class": owner, "lifetime_class": lifetime, "unit": detail}
        (non_step if lifetime == "non_step" else membership).append(entry)
    return membership, unclassified, non_step


def _charged(row: Mapping) -> bool:
    """Whether one of the seven composition terms charges this classified row.

    Three fixed cells, three candidate cells and resident KV are charged; a KV
    backing with a transient lifetime is not, and neither is a candidate
    allocation with no unit, because every candidate term is keyed by unit.
    A row that falls through is the one error direction that must never be
    silent: an overcount wastes headroom, an undercount hands a serving gate a
    budget smaller than the engine needs.
    """
    return (row["owner_class"] == "kv" and row["lifetime_class"] == "resident"
            or row["owner_class"] == "fixed"
            or row["owner_class"] == "candidate" and row["unit"] is not None)


def _charge(lifetime: str, rows: Sequence[Mapping]) -> int:
    """Resident bytes add; transient bytes peak.

    Resident allocations are live at the terminal boundary by definition, so
    their charge is the sum. A transient charge is the simultaneous sweep and
    never a sum of per-allocation maxima.
    """
    if lifetime == "resident":
        return sum(row["bytes"] for row in rows)
    return _simultaneous_peak(rows)


def _recompute_terms(membership: Sequence[Mapping], *, closed: Mapping[str, bool],
                     unclassified: int, uncharged: int) -> dict[str, Any]:
    """Recompute every term the emitted observations can express.

    An unclassified allocation nulls every term whatever the domains say: a
    row with neither an owner nor a lifetime could belong to any of them, so
    no term is complete while one exists. An uncharged row nulls them for the
    same reason from the other side: it is classified and no term reaches it,
    so a composition built over the seven terms omits bytes the engine holds.
    """
    def rows(owner: str, lifetime: str) -> list[Mapping]:
        return [row for row in membership
                if row["owner_class"] == owner and row["lifetime_class"] == lifetime]

    units = sorted({row["unit"] for row in membership if row["unit"] is not None})
    terms: dict[str, Any] = {name: None for name in TERMS}
    if unclassified or uncharged:
        return terms
    for term in RECOMPUTABLE_TERMS:
        if not all(closed[name] for name in TERM_DEPENDENCIES[term]):
            continue
        owner, _, lifetime = term.partition("_")
        if term == "fixed_kv":
            owner, lifetime = "kv", "resident"
        selected = rows(owner, lifetime)
        if term in PER_UNIT_TERMS:
            # One charge per unit, every unit present, including a unit whose
            # charge is zero: the reduction is the composition's business, and
            # a term that hid a unit would make it unauditable.
            terms[term] = {
                unit: _charge(lifetime, [row for row in selected if row["unit"] == unit])
                for unit in units}
        else:
            terms[term] = _charge(lifetime, selected)
    return terms


def _term_disagreements(name: str, claimed: Any, recomputed: Any) -> list[str]:
    """Every way the producer's term differs from this consumer's.

    A per-unit term is compared unit by unit rather than by its reduction:
    two units that trade the same bytes produce the same `max` and the same
    `sum` while each unit's own charge is wrong, and a comparison of totals
    cannot see it. Types are compared strictly, so a mapping never matches a
    scalar and ``True`` never matches ``1``.
    """
    if claimed is None or recomputed is None or type(claimed) is not type(recomputed):
        if claimed != recomputed or type(claimed) is not type(recomputed):
            return [f"derived term {name} is {claimed!r} where this consumer recomputes {recomputed!r}"]
        return []
    if name not in PER_UNIT_TERMS:
        return ([] if claimed == recomputed
                else [f"derived term {name} is {claimed!r} where this consumer recomputes {recomputed!r}"])
    messages = []
    for unit in sorted(set(claimed) | set(recomputed)):
        if unit not in claimed:
            messages.append(f"derived term {name} charges no unit {unit!r}, which this consumer "
                            f"charges {recomputed[unit]!r}")
        elif unit not in recomputed:
            messages.append(f"derived term {name} charges unit {unit!r} {claimed[unit]!r}, which is "
                            "not a unit this consumer charges")
        elif claimed[unit] != recomputed[unit] or type(claimed[unit]) is not type(recomputed[unit]):
            messages.append(f"derived term {name} charges unit {unit!r} {claimed[unit]!r} where this "
                            f"consumer recomputes {recomputed[unit]!r}")
    return messages


def _compose(terms: Mapping[str, Any]):
    """`fixed_resident + sum(candidate_resident) + fixed_activation
    + max(candidate_activation) + fixed_scratch + max(candidate_scratch)
    + fixed_KV`, or nothing at all when a term is unknown."""
    if any(terms[name] is None for name in TERMS):
        return None
    return (terms["fixed_resident"]
            + sum(terms["candidate_resident"].values())
            + terms["fixed_activation"]
            + max(terms["candidate_activation"].values(), default=0)
            + terms["fixed_scratch"]
            + max(terms["candidate_scratch"].values(), default=0)
            + terms["fixed_kv"])


def _non_step_peak(non_step: Sequence[Mapping], *, closed: Mapping[str, bool],
                   unclassified: int, uncharged: int):
    """The off-step transient peak, or nothing at all.

    An off-step price needs the same join a scratch term needs. It is a peak
    over rows the history join may have missed and whose external bytes may
    never have been closed, and an unclassified or uncharged row could itself
    be off-step, so the price goes null while the count stays readable.
    """
    if unclassified or uncharged or not all(closed[name] for name in NON_STEP_PEAK_DOMAINS):
        return None
    return _simultaneous_peak(non_step)


def _placement_obligation(budget: Any, non_step_peak: Any):
    """``max(scalar_budget_bytes, non_step_transient_peak_bytes)``, or nothing.

    Both sides are obligations on the same box. The seven terms price one
    engine step; the off-step peak prices what the engine still holds when no
    step is running. A placement that satisfies only one of them satisfies the
    smaller of two numbers. Neither side is defaulted to zero: an absent side
    is an absence of evidence, and a maximum taken against it would read as
    the other side having been checked.
    """
    if budget is None or non_step_peak is None:
        return None
    return max(budget, non_step_peak)


@dataclass(frozen=True)
class ReportVerdict:
    """What the consumer recomputed, and why nothing is admissible.

    ``disagreements`` are places the producer's ``derived`` claim differs from
    this consumer's own recomputation. ``blocking`` are the reasons no term is
    expressible even where the two agree. Both refuse. An empty ``refusals``
    means the report recomputes and every expressible term is present; it is
    still not an admission, which no code path in this module performs.
    """
    schema: str
    fixture_provenance: Any
    recomputed_terms: Mapping[str, Any]
    recomputed_scalar_budget_bytes: Any
    recomputed_non_step_transient_peak_bytes: Any
    recomputed_placement_obligation_bytes: Any
    open_domains: tuple
    unclassified_allocations: tuple
    disagreements: tuple = ()
    blocking: tuple = ()

    @property
    def refusals(self) -> tuple:
        return tuple(self.disagreements) + tuple(self.blocking)

    @property
    def expressible_terms(self) -> tuple:
        return tuple(name for name in TERMS if self.recomputed_terms[name] is not None)

    def describe(self) -> str:
        if not self.refusals:
            return "the report recomputes and every expressible term is present"
        return "; ".join(self.refusals)


def consume_full_engine_resource_report(reference: Mapping, *, root: Path,
                                        expected_run_identity: Mapping | None = None
                                        ) -> ReportVerdict:
    """Read one report, recompute it, and name every reason it is not usable.

    Structural faults raise :class:`RuntimePriceError` from the reader.
    Semantic faults are collected so a caller sees every one of them at once.
    """
    report = read_full_engine_resource_report(reference, root=root)
    identity, observations = report["identity"], report["observations"]
    partition, derived = report["partition"], report["derived"]
    disagreements: list[str] = []
    blocking: list[str] = []

    def disagree(message: str) -> None:
        disagreements.append(message)

    # Identity. The report's three capture digests name one capture, and the
    # partition is the partition of this run and no other.
    for member, digest in (("observations", observations["capture_sha256"]),
                           ("partition", partition["capture_sha256"])):
        if digest != identity["capture_sha256"]:
            disagree(f"{member} names a different capture than the report identity")
    if partition["identity"] != identity["run"]:
        disagree("the partition's run identity differs from the report identity")
    for key, expected in (expected_run_identity or {}).items():
        if key not in identity["run"]:
            disagree(f"the report identity declares no {key}")
        elif identity["run"][key] != expected:
            disagree(f"stale {key}: the report was measured on other bytes")
    if identity["fixture_provenance"] is not None:
        blocking.append(f"the report carries fixture provenance {identity['fixture_provenance']!r} "
                        "and is not a measurement")

    # Coordinates the raw ledger does not carry. Each must be supplied, and a
    # report that invents its own is what this recomputation exists to catch.
    if report["workload"]["calibration"] is None or not report["workload"]["prompt_ids"]:
        blocking.append("the workload carries no calibration identity or prompt roster, "
                        "so its digest is not recomputable")
    if report["reference"]["canonical_census"] is None or not report["reference"]["selected_rows"]:
        blocking.append("the reference carries no canonical census or selected rows, "
                        "so it partitions no model roster")

    device = str(identity["run"]["device_id"])
    owners_seen: dict[str, str] = {}
    allocations = {row["allocation_id"]: row for row in observations["torch_allocations"]}
    for allocation_id, allocation in allocations.items():
        if not allocation_id.startswith(device + ":"):
            disagree(f"allocation {allocation_id} is on a foreign device")
        blocks = allocation["allocator_block_bytes_observed"]
        if blocks and max(blocks) < allocation["bytes"]:
            disagree(f"allocation {allocation_id} has no observed parent segment large enough to back it")
        for owner in allocation["observed_owners"]:
            if owners_seen.setdefault(owner, allocation_id) != allocation_id:
                disagree(f"owner {owner!r} is a duplicate alias claimed by two backings")
        categories = set(allocation["observed_categories"]) & set(OWNER_CLASSES)
        if len(categories) > 1:
            disagree(f"allocation {allocation_id} is claimed by two owner classes "
                     f"({', '.join(sorted(categories))})")

    # Checkpoints are a second view of the same allocations, so they are
    # recomputed too rather than trusted.
    for checkpoint in observations["checkpoints"]:
        index = checkpoint["trace_index"]
        live = [row for row in allocations.values()
                if row["allocate_index"] <= index
                and (row["free_completed_index"] is None or index < row["free_completed_index"])
                and row["observed_owners"]]
        claimed = {storage["allocation_id"] for storage in checkpoint["storages"]}
        if claimed != {row["allocation_id"] for row in live}:
            disagree(f"checkpoint {checkpoint['label']!r} does not list the owned storages live at its index")
        if checkpoint["unique_owned_storage_bytes"] != sum(row["bytes"] for row in live):
            disagree(f"checkpoint {checkpoint['label']!r} claims other owned storage bytes than its live extents")
        owners = {owner for row in live for owner in row["observed_owners"]}
        if checkpoint["owner_count"] != len(owners):
            disagree(f"checkpoint {checkpoint['label']!r} claims another owner count than its live owners")
        for storage in checkpoint["storages"]:
            row = allocations.get(storage["allocation_id"])
            if row is None:
                disagree(f"checkpoint {checkpoint['label']!r} names an unobserved storage")
            elif storage["bytes"] != row["bytes"] or storage["address"] != row["address"]:
                disagree(f"checkpoint {checkpoint['label']!r} restates a storage extent")
        if checkpoint["unmatched_storage_observations"]:
            disagree(f"checkpoint {checkpoint['label']!r} leaves storage observations unmatched")
    if observations["torch_observed_live_peak_bytes"] != _simultaneous_peak(list(allocations.values())):
        disagree("the observed live peak is not the maximum simultaneous sum of the observed allocations")
    if observations["issues"] or observations["cuda_argument_domains"]["issues"]:
        disagree("the ledger carries unresolved issues")
    if (observations["cuda_argument_domains"]["status"] == "unavailable"
            and observations["cuda_argument_domains"]["handled_api_keys"]):
        disagree("allocation APIs are handled by an argument domain that declares itself unavailable")

    # A unit interval that crosses a declared step boundary contradicts both
    # declarations, and the producer refuses to emit one. The report carries no
    # unit intervals, so the contradiction is derived from the allocations that
    # ran inside them.
    for allocation_id in _allocations_whose_unit_crosses_a_step(observations):
        disagree(f"allocation {allocation_id} ran inside a unit invocation that spans a declared "
                 "step boundary, which no capture may declare")

    # The partition, recomputed from the observations it claims to derive from.
    membership, unclassified, non_step = _recompute_membership(observations)
    by_id = {row["allocation_id"]: row for row in membership}
    claimed_membership = {row["allocation_id"]: row for row in partition["membership"]}
    if set(by_id) != set(claimed_membership):
        disagree("the partition's classified allocations are not the ones this consumer classifies")
    for allocation_id, row in by_id.items():
        claimed = claimed_membership.get(allocation_id)
        if claimed is not None and any(claimed[key] != row[key] for key in _MEMBERSHIP_FIELDS):
            disagree(f"allocation {allocation_id} is partitioned differently than it recomputes")
    if (report["schema"] == REPORT_SCHEMA_V2 and observations["owner_views"] is not None
            and (set(by_id) != set(claimed_membership)
                 or any(claimed_membership[key][field] != by_id[key][field]
                        for key in set(by_id) & set(claimed_membership)
                        for field in _MEMBERSHIP_FIELDS))):
        # A v2 producer classifies on its ownership observation (site and
        # history rules over `owner_views`); this consumer recomputes
        # membership from `observed_categories` and `lifetime_scope` alone.
        # The disagreements above are that gap, named once so they are not
        # read as the producer contradicting its own ledger.
        views = observations["owner_views"].get("views")
        rules = sorted((views or {}).get("rules") or ()) if isinstance(views, Mapping) else []
        blocking.append(f"the partition classifies on owner_views (rules {rules}), which this "
                        f"consumer recomputes no membership from at {report['schema']}, so its "
                        "membership disagreements are this consumer's gap rather than the "
                        "producer's contradiction")
    claimed_unclassified = {row["allocation_id"]: row for row in partition["unclassified_allocations"]}
    if {row["allocation_id"] for row in unclassified} != set(claimed_unclassified):
        disagree("the partition's unclassified allocations are not the ones this consumer cannot classify")
    for row in unclassified:
        claimed = claimed_unclassified.get(row["allocation_id"])
        if claimed is not None and claimed["bytes"] != row["bytes"]:
            disagree(f"unclassified allocation {row['allocation_id']} restates its extent")
    # The off-step rows are recomputed from the same observations and compared
    # cell by cell, exactly as the charged rows are. A producer that moved one
    # row into this list moved it out of every term that charges it.
    by_non_step = {row["allocation_id"]: row for row in non_step}
    claimed_non_step = {row["allocation_id"]: row for row in partition["non_step_allocations"]}
    if set(by_non_step) != set(claimed_non_step):
        disagree("the partition's off-step allocations are not the ones this consumer finds "
                 "live during no declared engine step")
    for allocation_id, row in by_non_step.items():
        claimed = claimed_non_step.get(allocation_id)
        if claimed is not None and any(claimed[key] != row[key] for key in _MEMBERSHIP_FIELDS):
            disagree(f"off-step allocation {allocation_id} is partitioned differently than it "
                     "recomputes")
    units = {row["unit"] for row in membership if row["unit"] is not None}
    if units != set(partition["units"]):
        disagree("the partition's units are not the units this consumer recomputes")
    # The uncharged set is derived here from the same observations, never read
    # from the report: it is the one omission that hands a serving gate a
    # budget smaller than the engine needs, so the producer's list is a claim
    # to check. The cells are compared, the prose reasons are not.
    uncharged = [row for row in membership if not _charged(row)]
    for row in uncharged:
        blocking.append(f"allocation {row['allocation_id']} is classified "
                        f"({row['owner_class']}, {row['lifetime_class']}) but no term charges it")
    claimed_uncharged = {row["allocation_id"]: row for row in partition["uncharged_allocations"]}
    if {row["allocation_id"] for row in uncharged} != set(claimed_uncharged):
        disagree("the partition's uncharged allocations are not the ones this consumer finds "
                 "no term charges")
    for row in uncharged:
        claimed = claimed_uncharged.get(row["allocation_id"])
        if claimed is not None and any(claimed[key] != row[key]
                                       for key in ("owner_class", "lifetime_class", "unit")):
            disagree(f"uncharged allocation {row['allocation_id']} is placed in another cell "
                     f"than it recomputes")

    closed = _recompute_domains(observations, identity)
    for name in DOMAINS:
        claimed = partition["domains"][name]
        if (claimed["state"] == "closed") == closed[name]:
            continue
        # Only an observation v1 owed as null and v2 carries counts: a domain
        # closed on a checked observation this consumer does read is compared
        # as a claim, exactly as before.
        carried = [item for item in claimed["evidence"]
                   if item in OWED_OBSERVATIONS and observations.get(item) is not None]
        if claimed["state"] == "closed" and name not in CHECKABLE_DOMAINS and carried:
            # The producer closed it on an observation it carries and this
            # consumer defines no recomputation for. That is not the producer
            # contradicting itself; it is a closure this side cannot verify,
            # and it stays not closed here for that reason.
            blocking.append(f"domain {name} is closed by the producer on {carried}, which this "
                            f"consumer recomputes nothing from at {report['schema']}, so it "
                            "stays not closed here")
            continue
        disagree(f"the partition block calls domain {name} {claimed['state']} "
                 f"where this consumer recomputes it as "
                 f"{'closed' if closed[name] else 'not closed'}")
    open_domains = tuple(name for name in DOMAINS if not closed[name])
    for name in open_domains:
        blocking.append(f"domain {name} is not closed, so every term depending on it stays null")
    for row in unclassified:
        blocking.append(f"allocation {row['allocation_id']} is unclassified ({row['reason']}), "
                        "which nulls every term")

    terms = _recompute_terms(membership, closed=closed, unclassified=len(unclassified),
                             uncharged=len(uncharged))
    for name in TERMS:
        for message in _term_disagreements(name, derived["terms"][name], terms[name]):
            disagree(message)
    budget = _compose(terms)
    if derived["scalar_budget_bytes"] != budget or type(derived["scalar_budget_bytes"]) is not type(budget):
        disagree(f"derived scalar budget is {derived['scalar_budget_bytes']!r} where this consumer "
                 f"recomputes {budget!r}")
    # The off-step price, recomputed. It is carried in two places -- the
    # partition and `derived` -- and both are the producer's claim, so both are
    # compared to this consumer's own sweep and to each other.
    non_step_peak = _non_step_peak(non_step, closed=closed, unclassified=len(unclassified),
                                   uncharged=len(uncharged))
    for member, claimed in (("derived", derived["non_step_transient_peak_bytes"]),
                            ("partition", partition["non_step_transient_peak_bytes"])):
        if claimed != non_step_peak or type(claimed) is not type(non_step_peak):
            disagree(f"{member} non_step_transient_peak_bytes is {claimed!r} where this consumer "
                     f"recomputes {non_step_peak!r}")
    # The scope sentence is prose and no check reads it, but two copies of one
    # claim invite drift, so the two copies must at least be one claim.
    if derived["non_step_transient_peak_scope"] != partition["non_step_transient_peak_scope"]:
        disagree("the derived and partition off-step peak scopes are two different claims")
    # A frozen contract spelling a gate reads, not prose: compared verbatim.
    if derived["placement_obligation"] != PLACEMENT_OBLIGATION:
        disagree(f"derived placement_obligation is {derived['placement_obligation']!r} where this "
                 f"consumer applies {PLACEMENT_OBLIGATION!r}")

    unavailable = tuple(sorted(name for name in TERMS if terms[name] is None))
    coverage_state = observations["step_coverage"]["state"]
    for member, scope in (("derived", derived["scope"]), ("partition", partition["scope"])):
        if tuple(sorted(scope["unavailable_terms"])) != unavailable:
            disagree(f"the {member} scope names other unavailable terms than this consumer recomputes")
        if scope["expressible"] is not (not unavailable):
            disagree(f"the {member} scope claims a different expressibility than its terms support")
        if scope["unclassified_allocation_count"] != len(unclassified):
            disagree(f"the {member} scope claims another unclassified allocation count")
        if scope["uncharged_allocation_count"] != len(uncharged):
            disagree(f"the {member} scope claims another uncharged allocation count")
        if scope["non_step_allocation_count"] != len(non_step):
            disagree(f"the {member} scope claims another off-step allocation count")
        if scope["step_coverage"] != coverage_state:
            disagree(f"the {member} scope claims step coverage {scope['step_coverage']!r} where "
                     f"the observations declare {coverage_state!r}")
    if budget is None:
        blocking.append("no scalar device budget is expressible from this report")
    # The whole classification runs on this licence, so it is named as a
    # refusal in its own right rather than left to be inferred from the
    # unclassified rows a missing licence produces.
    if coverage_state != ADMITTED_STEP_COVERAGE:
        blocking.append(f"step coverage is {coverage_state!r}, so an allocation live during no "
                        "declared engine step may still be live during an undeclared one and "
                        "the off-step classification is not licensed")
    obligation = _placement_obligation(budget, non_step_peak)
    if obligation is None:
        blocking.append("no placement obligation is recomputable from this report")

    return ReportVerdict(schema=report["schema"], fixture_provenance=identity["fixture_provenance"],
                         recomputed_terms=terms, recomputed_scalar_budget_bytes=budget,
                         recomputed_non_step_transient_peak_bytes=non_step_peak,
                         recomputed_placement_obligation_bytes=obligation,
                         open_domains=open_domains,
                         unclassified_allocations=tuple(row["allocation_id"] for row in unclassified),
                         disagreements=tuple(disagreements), blocking=tuple(blocking))
