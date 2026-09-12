"""Pure artifact consumer for ``tessera.full_engine_resource_report.v1``.

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

This module admits nothing. ``runtime_provenance.admit_fixed_resources`` does
not call it and keeps its unconditional refusal. Wiring a gate to a recomputed
partition is the separate "Integrate allocator admission" prerequisite in
``docs/design/runtime_fixed_resource_admission.md``; #420 and the producer's
own issue both stay open until that row is built on qualified measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .measured_runtime_prices import RuntimePriceError, _integer, _object, _sha, _string
from .runtime_provenance import ArtifactReader, _equal

REPORT_SCHEMA = "tessera.full_engine_resource_report.v1"
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
#: ``qualified: true`` spelled differently.
CHECKABLE_DOMAINS = ("history_join", "external_closure")

#: The observations this schema version names and sets to null, so a consumer
#: can tell "this capture did not observe it" from "the producer dropped it".
#: They are what the other four domains would close on, which is why the
#: scalar composition cannot complete at v1 even on a flawless capture: only
#: `fixed_scratch` and `candidate_scratch` are reachable at all. A non-null
#: value here is refused rather than read -- this consumer recomputes nothing
#: from an observation whose shape v1 does not define, and a domain that
#: closed on one would be certifying itself.
OWED_OBSERVATIONS = ("kv_observations", "observer_qualification", "owner_views",
                     "runtime_provenance_relation", "timing_captures",
                     "worker_startup_records")

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
#: null every term on any capture that has one. What `fixed_kv` still waits on
#: is its domain, `cache_capacity`, which never closes at this schema version.
OWNER_CLASSES = ("fixed", "candidate", "kv")
#: Neither label supplies a classification or an invariance, so a row carrying
#: one is unclassified and named, never bucketed.
UNSUPPORTED_OWNER_LABELS = ("shared", "unknown")
LIFETIME_SCOPES = ("outside_units", "inside_unit", "escapes_unit")
LIFETIME_CLASSES = ("resident", "activation", "scratch")

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
_RUN_FIELDS = ("assignment_sha256", "canonical_units_sha256", "configuration_sha256",
               "device_id", "device_uuid", "model_sha256", "runtime_manifest_sha256",
               "schema", "workload_sha256")
_RUN_DIGESTS = ("assignment_sha256", "canonical_units_sha256", "configuration_sha256",
                "model_sha256", "runtime_manifest_sha256", "workload_sha256")
_EXECUTION_FIELDS = ("graph_mode", "residency", "topology")
_REFERENCE_FIELDS = ("canonical_census", "runtime_binding", "selected_rows")
_WORKLOAD_FIELDS = ("calibration", "prompt_ids", "sampling")
_OBSERVATION_FIELDS = ("artifacts", "capture_sha256", "checkpoints", "cuda_argument_domains",
                       "external_native_peak_bytes", "issues", "kv_observations",
                       "observer_qualification", "owner_views", "runtime_provenance_relation",
                       "timing_captures", "torch_allocations",
                       "torch_observed_live_peak_bytes", "torch_observed_live_peak_scope",
                       "unattributed_external_records", "worker_startup_records")
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
                     "schema", "scope", "terms", "uncharged_allocations",
                     "unclassified_allocations", "units")
_MEMBERSHIP_FIELDS = ("allocate_index", "allocation_id", "bytes", "free_completed_index",
                      "lifetime_class", "owner_class", "unit")
_UNCLASSIFIED_FIELDS = ("allocation_id", "bytes", "lifetime_scope", "observed_categories", "reason")
#: A classified row that no composition term charges. It carries its cell and
#: its unit, which is what the consumer recomputes, plus the producer's prose
#: reason, which no check reads.
_UNCHARGED_FIELDS = ("allocation_id", "bytes", "lifetime_class", "owner_class", "reason", "unit")
_DERIVED_FIELDS = ("scalar_budget_bytes", "scope", "terms")
_SCOPE_FIELDS = ("allocation_scope", "expressible", "invariance", "topology",
                 "unavailable_terms", "uncharged_allocation_count",
                 "unclassified_allocation_count")
_DOMAIN_FIELDS = ("evidence", "reason", "state")


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


def _scope(value: Any, where: str) -> Mapping:
    scope = _object(value, _SCOPE_FIELDS, where)
    for key, supported in SUPPORTED_SCOPE.items():
        _string(scope[key], f"{where} {key}")
        if scope[key] != supported:
            raise RuntimePriceError(
                f"{where}: unsupported {key} {scope[key]!r}; this consumer covers only {supported!r}")
    _bool(scope["expressible"], where + " expressible")
    _index(scope["unclassified_allocation_count"], where + " unclassified allocation count")
    _index(scope["uncharged_allocation_count"], where + " uncharged allocation count")
    for name in _string_list(scope["unavailable_terms"], where + " unavailable terms"):
        if name not in TERMS:
            raise RuntimePriceError(f"{where}: unknown term {name!r}")
    return scope


def _run_identity(value: Any, where: str) -> Mapping:
    run = _object(value, _RUN_FIELDS, where)
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


def _observations(value: Any, where: str) -> Mapping:
    observations = _object(value, _OBSERVATION_FIELDS, where)
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
    for name in OWED_OBSERVATIONS:
        if observations[name] is not None:
            raise RuntimePriceError(
                f"{where}: {name} is not null, but this schema version defines no shape for it, "
                "so nothing here can recompute a term or close a domain from it")
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


def _partition(value: Any, where: str) -> Mapping:
    partition = _object(value, _PARTITION_FIELDS, where)
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
    _scope(partition["scope"], where + " scope")
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
    identities = [row["allocation_id"] for row in rows + unclassified]
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
    if report.get("schema") != REPORT_SCHEMA:
        raise RuntimePriceError(
            f"full-engine resource report: unsupported schema {report.get('schema')!r}; "
            f"this consumer reads only {REPORT_SCHEMA}")
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
    reference_member = _object(report["reference"], _REFERENCE_FIELDS, "report reference")
    _list(reference_member["selected_rows"], "report reference selected rows")
    workload = _object(report["workload"], _WORKLOAD_FIELDS, "report workload")
    _list(workload["prompt_ids"], "report workload prompt ids")
    observations = _observations(report["observations"], "report observations")
    partition = _partition(report["partition"], "report partition")
    _resolved_evidence(partition["domains"], observations, "report partition domain")
    derived = _object(report["derived"], _DERIVED_FIELDS, "report derived")
    _terms(derived["terms"], "report derived terms")
    _scope(derived["scope"], "report derived scope")
    _optional_index(derived["scalar_budget_bytes"], "report derived scalar budget bytes")
    return report


# --------------------------------------------------------------------------
# Independent recomputation. Nothing below reads `derived`.
# --------------------------------------------------------------------------

def _classify(allocation: Mapping) -> tuple:
    """Return ``(owner_class, lifetime_class, unit)`` or ``(None, None, reason)``.

    Ownership and lifetime are separate questions and a row needs a supported
    answer to both. ``shared`` and ``unknown`` answer neither.
    """
    categories = list(allocation["observed_categories"])
    freed = allocation["free_completed_index"] is not None
    scope = allocation["lifetime_scope"]
    if any(label in UNSUPPORTED_OWNER_LABELS for label in categories):
        return None, None, "shared or unknown ownership supplies neither classification nor invariance"
    if len(categories) != 1 or categories[0] not in OWNER_CLASSES:
        return None, None, "no single supported owner category"
    owner = categories[0]
    if not freed:
        # Never freed within the capture, so it is live at the terminal
        # boundary whatever scope it was allocated in. Resident bytes add.
        lifetime = "resident"
    elif scope == "inside_unit":
        lifetime = "scratch"
    elif scope == "escapes_unit":
        lifetime = "activation"
    else:
        # Freed with no unit interval containing either end. Charging it as
        # fixed scratch assumes once per step and treating it as startup
        # assumes never again; both are fills, and the capture emits no
        # declared step boundary to decide between them.
        return None, None, "freed outside every unit interval with no declared step boundary"
    # The unit an allocation is charged to is the innermost scope it was made
    # in. `unit_invocation` names the same unit from the other side, so a row
    # where the two disagree is not a row this consumer can charge anywhere.
    stack = allocation["scope_stack"]
    unit = stack[-1] if stack else None
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


def _recompute_domains(observations: Mapping) -> dict[str, bool]:
    """Which domains this consumer can itself see closed, keyed by name.

    Only two of the six have a closing condition expressible in the raw
    observations this schema emits. The other four have none, so they are open
    here regardless of the state the report declares: a domain that closes
    because a field was truthy is a status flag, and the consumer would be
    reading the producer's word for what the producer did.
    """
    blocked_by_issues = bool(observations["issues"])
    closed = {name: False for name in DOMAINS}
    closed["history_join"] = (not blocked_by_issues
                              and not observations["unattributed_external_records"])
    closed["external_closure"] = (not blocked_by_issues
                                  and observations["external_native_peak_bytes"] is not None)
    return closed


def _recompute_membership(observations: Mapping) -> tuple[list[dict], list[dict]]:
    membership, unclassified = [], []
    for allocation in observations["torch_allocations"]:
        owner, lifetime, detail = _classify(allocation)
        if owner is None:
            unclassified.append({"allocation_id": allocation["allocation_id"],
                                 "bytes": allocation["bytes"], "reason": detail})
            continue
        membership.append({"allocation_id": allocation["allocation_id"],
                           "allocate_index": allocation["allocate_index"],
                           "bytes": allocation["bytes"],
                           "free_completed_index": allocation["free_completed_index"],
                           "owner_class": owner, "lifetime_class": lifetime, "unit": detail})
    return membership, unclassified


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

    # The partition, recomputed from the observations it claims to derive from.
    membership, unclassified = _recompute_membership(observations)
    by_id = {row["allocation_id"]: row for row in membership}
    claimed_membership = {row["allocation_id"]: row for row in partition["membership"]}
    if set(by_id) != set(claimed_membership):
        disagree("the partition's classified allocations are not the ones this consumer classifies")
    for allocation_id, row in by_id.items():
        claimed = claimed_membership.get(allocation_id)
        if claimed is not None and any(claimed[key] != row[key] for key in _MEMBERSHIP_FIELDS):
            disagree(f"allocation {allocation_id} is partitioned differently than it recomputes")
    claimed_unclassified = {row["allocation_id"]: row for row in partition["unclassified_allocations"]}
    if {row["allocation_id"] for row in unclassified} != set(claimed_unclassified):
        disagree("the partition's unclassified allocations are not the ones this consumer cannot classify")
    for row in unclassified:
        claimed = claimed_unclassified.get(row["allocation_id"])
        if claimed is not None and claimed["bytes"] != row["bytes"]:
            disagree(f"unclassified allocation {row['allocation_id']} restates its extent")
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

    closed = _recompute_domains(observations)
    for name in DOMAINS:
        claimed = partition["domains"][name]
        if (claimed["state"] == "closed") != closed[name]:
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
    unavailable = tuple(sorted(name for name in TERMS if terms[name] is None))
    for member, scope in (("derived", derived["scope"]), ("partition", partition["scope"])):
        if tuple(sorted(scope["unavailable_terms"])) != unavailable:
            disagree(f"the {member} scope names other unavailable terms than this consumer recomputes")
        if scope["expressible"] is not (not unavailable):
            disagree(f"the {member} scope claims a different expressibility than its terms support")
        if scope["unclassified_allocation_count"] != len(unclassified):
            disagree(f"the {member} scope claims another unclassified allocation count")
        if scope["uncharged_allocation_count"] != len(uncharged):
            disagree(f"the {member} scope claims another uncharged allocation count")
    if budget is None:
        blocking.append("no scalar device budget is expressible from this report")

    return ReportVerdict(schema=REPORT_SCHEMA, fixture_provenance=identity["fixture_provenance"],
                         recomputed_terms=terms, recomputed_scalar_budget_bytes=budget,
                         open_domains=open_domains,
                         unclassified_allocations=tuple(row["allocation_id"] for row in unclassified),
                         disagreements=tuple(disagreements), blocking=tuple(blocking))
