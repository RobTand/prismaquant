"""Pure, identity-bound contract for the Tessera quality-prefill experiment.

This module owns three things and performs no I/O:

* the frozen run manifest ``prismaquant.quality_prefill_experiment.v1``
  (:data:`MANIFEST_SCHEMA`), with the eleven closed sections of the
  specification's section 3.1;
* the seven evidence envelopes of section 3.2, and the one price-table object
  they share;
* the section 4 phase DAG and its explicit legal-transition table.

It follows ``cluster_campaign_contract.py``, which is the house exemplar for a
closed, strictly typed, canonically hashed contract, and it reuses that
module's hasher (``cost_stage_checkpoint.canonical_json_sha256``) rather than
introducing a second canonical JSON encoding.

Three rules the specification states and this module enforces mechanically:

* A path string is not an identity.  Every reference to an artifact another
  work package owns is an ``{path, sha256}`` object, validated once by
  :func:`_artifact_reference`.
* A price table carries exactly one currency.  The scalar screen currency
  ``output_mse_under_route_activation_contract`` and the measured joint
  currency ``joint_aura_predicted_dloss`` never appear in the same table, and
  a row whose currency differs from its table header is refused.
* A required value that nobody has chosen is an explicit unresolved sentinel,
  not an invented default.  A draft manifest may carry them; a frozen manifest
  may not, and :func:`unresolved_inputs` names every one of them at once so a
  refusal can report the complete set.

The driver ``experiments/tessera_quality_prefill.py`` does the I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Literal

from prismaquant.cost_stage_checkpoint import canonical_json_sha256


MANIFEST_SCHEMA = "prismaquant.quality_prefill_experiment.v1"
PRICE_TABLE_SCHEMA = "prismaquant.quality_prefill_experiment.price_table.v1"
CANDIDATE_SCHEMA = "prismaquant.quality_prefill_experiment.candidate.v1"
OBSERVATION_SCHEMA = "prismaquant.quality_prefill_experiment.observation.v1"
SCREEN_DECISION_SCHEMA = "prismaquant.quality_prefill_experiment.screen_decision.v1"
TRANSFER_SEAL_SCHEMA = "prismaquant.quality_prefill_experiment.transfer_seal.v1"
PHASE_RESULT_SCHEMA = "prismaquant.quality_prefill_experiment.phase_result.v1"
ASSIGNMENT_SCHEMA = "prismaquant.quality_prefill_experiment.assignment.v1"
FRONTIER_REPORT_SCHEMA = "prismaquant.quality_prefill_experiment.frontier_report.v1"


class QualityPrefillContractError(ValueError):
    """A manifest, envelope, or requested phase transition is invalid."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------

#: The two currencies of section 3.2.  They are never mixed in one table.
SCALAR_CURRENCY = "output_mse_under_route_activation_contract"
JOINT_CURRENCY = "joint_aura_predicted_dloss"
CURRENCIES = frozenset({SCALAR_CURRENCY, JOINT_CURRENCY})

#: The six coverage values.  One field carries them -- ``measurement_status``,
#: the name section 3.2 gives the per-row field -- so there is no second,
#: overlapping "coverage type" enum to drift against.
COVERAGE_TYPES = (
    "measured",
    "predicted_screen",
    "retained_pending_measurement",
    "pruned_by_declared_screen",
    "unsupported_route",
    "not_yet_acquired",
)
_COVERAGE_SET = frozenset(COVERAGE_TYPES)

#: Only a measured row may carry the joint currency; the scalar currency is a
#: screen.  Both statements are section 3.2's, enforced in
#: :func:`validate_price_table`.
_JOINT_STATUSES = frozenset({"measured"})

ROUTE_STATUSES = frozenset({"backed", "backed_with_serve_flag", "unbacked"})
DISPOSITIONS = frozenset({"retained", "pruned", "deferred"})
VALIDITY = frozenset({"valid", "unknown"})
TIE_POLICIES = frozenset({"lowest_bytes", "lowest_index", "refuse"})
GRAPH_MODES = frozenset({"eager", "graph"})


# ---------------------------------------------------------------------------
# Phase DAG (specification section 4) and its state machine
# ---------------------------------------------------------------------------

PHASE_STATES = (
    "planned",
    "waiting_dependencies",
    "ready",
    "submitted",
    "running",
    "collecting",
    "complete",
    "failed",
    "withdrawn",
    "requires_plan_revision",
)
_STATE_SET = frozenset(PHASE_STATES)

TERMINAL_STATES = frozenset({"complete", "withdrawn", "requires_plan_revision"})

#: The complete legal transition table.  Every pair absent from it refuses.
#: ``failed -> ready`` is the retry edge and is legal only inside the original
#: attempt budget, which :func:`transition` checks.
LEGAL_TRANSITIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "planned": frozenset(
            {"waiting_dependencies", "ready", "withdrawn", "requires_plan_revision"}
        ),
        "waiting_dependencies": frozenset(
            {"ready", "withdrawn", "requires_plan_revision"}
        ),
        "ready": frozenset({"submitted", "withdrawn", "requires_plan_revision"}),
        "submitted": frozenset({"running", "failed", "withdrawn"}),
        "running": frozenset({"collecting", "failed", "withdrawn"}),
        "collecting": frozenset({"complete", "failed", "requires_plan_revision"}),
        "failed": frozenset({"ready", "withdrawn", "requires_plan_revision"}),
        "complete": frozenset(),
        "withdrawn": frozenset(),
        "requires_plan_revision": frozenset(),
    }
)


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    """One node of the fixed section 4 DAG."""

    phase: str
    title: str
    dependencies: tuple[str, ...]


def _phase(phase: str, title: str, *dependencies: str) -> PhaseSpec:
    return PhaseSpec(phase=phase, title=title, dependencies=dependencies)


#: The specification's flowchart A..P, verbatim.  The order and the edges are
#: part of the v1 contract: a manifest may bind rosters and limits to these
#: phases but may not add, remove or rewire one.
PHASE_DAG: tuple[PhaseSpec, ...] = (
    _phase("inventory_and_freeze_inputs", "Inventory and freeze inputs"),
    _phase(
        "source_and_route_correctness",
        "Source and route correctness",
        "inventory_and_freeze_inputs",
    ),
    _phase(
        "boundary_and_activation_screen",
        "Boundary and activation screen",
        "source_and_route_correctness",
    ),
    _phase(
        "freeze_transfer_audit_targets",
        "Freeze transfer audit targets",
        "boundary_and_activation_screen",
    ),
    _phase(
        "measure_bf_sources_and_e4_endpoints",
        "Measure BF sources and E4 endpoints",
        "freeze_transfer_audit_targets",
    ),
    _phase(
        "seal_e4_predictions",
        "Seal E4 predictions",
        "measure_bf_sources_and_e4_endpoints",
    ),
    _phase(
        "measure_complementary_e4_targets_and_audit",
        "Measure complementary E4 targets and audit",
        "seal_e4_predictions",
    ),
    _phase(
        "prepare_measurement_producers",
        "Prepare native and full-engine measurement producers",
        "boundary_and_activation_screen",
    ),
    _phase(
        "freeze_retained_candidate_menu",
        "Freeze retained candidate menu",
        "measure_complementary_e4_targets_and_audit",
    ),
    _phase(
        "materialize_and_measure_joint_aura",
        "Materialize and measure joint AURA",
        "freeze_retained_candidate_menu",
    ),
    _phase(
        "measure_whole_operator_timing_and_memory",
        "Measure whole-operator timing and memory",
        "freeze_retained_candidate_menu",
        "prepare_measurement_producers",
    ),
    _phase(
        "admit_full_engine_fixed_partition",
        "Admit full-engine fixed partition",
        "prepare_measurement_producers",
    ),
    _phase(
        "validate_tables_and_solve_frontier",
        "Validate complete tables and solve frontier",
        "materialize_and_measure_joint_aura",
        "measure_whole_operator_timing_and_memory",
        "admit_full_engine_fixed_partition",
    ),
    _phase(
        "serve_development_knee_and_neighbors",
        "Serve development knee and neighbors",
        "validate_tables_and_solve_frontier",
    ),
    _phase(
        "freeze_selected_comparison_then_confirm",
        "Freeze selected comparison then confirm",
        "serve_development_knee_and_neighbors",
    ),
    _phase(
        "report_measured_frontier_and_limitations",
        "Report measured frontier and limitations",
        "freeze_selected_comparison_then_confirm",
    ),
)

PHASE_BY_NAME: Mapping[str, PhaseSpec] = MappingProxyType(
    {spec.phase: spec for spec in PHASE_DAG}
)
if len(PHASE_BY_NAME) != len(PHASE_DAG):  # pragma: no cover - static guard
    raise RuntimeError("quality-prefill phase DAG contains duplicate names")
for _spec in PHASE_DAG:  # pragma: no cover - static guard
    for _dependency in _spec.dependencies:
        if _dependency not in PHASE_BY_NAME:
            raise RuntimeError(f"phase {_spec.phase} depends on unknown {_dependency}")


# ---------------------------------------------------------------------------
# Strict primitives
# ---------------------------------------------------------------------------

_ID_RE = re.compile(r"[a-z0-9][a-z0-9._:-]{0,127}\Z")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
_IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_VERSION_RE = re.compile(r"v[0-9]+(\.[0-9]+){0,2}\Z")
_PATH_COMPONENT_RE = re.compile(r"[A-Za-z0-9._-]+\Z")

_ARTIFACT_REFERENCE_KEYS = frozenset({"path", "sha256"})
_UNRESOLVED_KEYS = frozenset({"unresolved"})


def _fail(message: str) -> None:
    raise QualityPrefillContractError(message)


def _exact_mapping(
    value: object,
    *,
    keys: frozenset[str],
    where: str,
) -> Mapping[str, object]:
    """Refuse anything but an object whose key set is exactly ``keys``."""

    if not isinstance(value, Mapping):
        _fail(f"{where} must be an object")
    if any(type(key) is not str for key in value):
        _fail(f"{where} keys must be strings")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        _fail(f"{where} fields differ: missing={missing}, extra={extra}")
    return value  # type: ignore[return-value]


def _string(
    value: object,
    *,
    where: str,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if type(value) is not str or not value:
        _fail(f"{where} must be a non-empty string")
    text = value  # type: ignore[assignment]
    if text != text.strip() or any(ord(char) < 32 for char in text):
        _fail(f"{where} contains whitespace padding or control characters")
    if pattern is not None and pattern.fullmatch(text) is None:
        _fail(f"{where} has an invalid value")
    return text


def _integer(
    value: object,
    *,
    where: str,
    minimum: int,
    maximum: int = 2**63 - 1,
) -> int:
    """``type(value) is int`` -- a ``bool`` never satisfies an integer field."""

    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{where} must be an integer in [{minimum}, {maximum}]")
    return value  # type: ignore[return-value]


def _finite_number(
    value: object,
    *,
    where: str,
    minimum: float | None = None,
) -> float:
    if type(value) not in (int, float):
        _fail(f"{where} must be a finite number")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        _fail(f"{where} must be a finite number")
    if minimum is not None and number < minimum:
        _fail(f"{where} must be at least {minimum}")
    return number


def _sha256(value: object, *, where: str) -> str:
    return _string(value, where=where, pattern=_SHA256_RE)


def _enum(value: object, *, where: str, allowed) -> str:
    text = _string(value, where=where)
    if text not in allowed:
        _fail(f"{where} must be one of {sorted(allowed)}")
    return text


def _sequence(value: object, *, where: str, minimum: int = 0) -> Sequence[object]:
    if type(value) is not list:
        _fail(f"{where} must be an array")
    if len(value) < minimum:  # type: ignore[arg-type]
        _fail(f"{where} must have at least {minimum} entries")
    return value  # type: ignore[return-value]


def _unique_ids(values: Sequence[object], *, where: str) -> tuple[str, ...]:
    out = tuple(
        _string(item, where=f"{where}[{index}]", pattern=_ID_RE)
        for index, item in enumerate(values)
    )
    if len(set(out)) != len(out):
        _fail(f"{where} contains duplicate identifiers")
    return out


def _absolute_posix_path(value: object, *, where: str) -> str:
    raw = _string(value, where=where)
    if not raw.startswith("/") or raw == "/":
        _fail(f"{where} must be a non-root absolute POSIX path")
    components = raw.split("/")[1:]
    if (
        not components
        or any(
            not component
            or component in {".", ".."}
            or _PATH_COMPONENT_RE.fullmatch(component) is None
            for component in components
        )
        or str(PurePosixPath(raw)) != raw
    ):
        _fail(f"{where} must be normalized and traversal-free")
    return raw


def _artifact_reference(value: object, *, where: str) -> Mapping[str, object]:
    """``{path, sha256}``.  A path alone is never an identity (section 3.2)."""

    body = _exact_mapping(value, keys=_ARTIFACT_REFERENCE_KEYS, where=where)
    _absolute_posix_path(body["path"], where=f"{where}.path")
    _sha256(body["sha256"], where=f"{where}.sha256")
    return body


def _named_version(value: object, *, where: str) -> Mapping[str, object]:
    body = _exact_mapping(value, keys=frozenset({"name", "version"}), where=where)
    _string(body["name"], where=f"{where}.name", pattern=_ID_RE)
    _string(body["version"], where=f"{where}.version", pattern=_VERSION_RE)
    return body


def _is_unresolved(value: object) -> bool:
    return isinstance(value, Mapping) and set(value) == {"unresolved"}


def _unresolved_sentinel(value: object, *, where: str) -> None:
    body = _exact_mapping(value, keys=_UNRESOLVED_KEYS, where=where)
    _string(body["unresolved"], where=f"{where}.unresolved")


# ---------------------------------------------------------------------------
# Canonical hashing (reused, never re-implemented)
# ---------------------------------------------------------------------------


def canonical_sha256(value: object) -> str:
    """SHA-256 of compact, sorted, strict canonical JSON of ``value``."""

    try:
        return canonical_json_sha256(value, where="quality-prefill value")
    except (TypeError, ValueError) as exc:
        raise QualityPrefillContractError(
            "quality-prefill value is not canonical JSON data"
        ) from exc


def canonical_json_bytes(value: object) -> bytes:
    """The exact bytes :func:`canonical_sha256` digests."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QualityPrefillContractError(
            "quality-prefill value is not canonical JSON data"
        ) from exc


def _seal(body: Mapping[str, object]) -> dict[str, object]:
    """Append the digest of the body.  A hash never covers its own field."""

    return {**body, "identity_sha256": canonical_sha256(dict(body))}


def _check_seal(value: Mapping[str, object], *, where: str) -> dict[str, object]:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    declared = _sha256(value["identity_sha256"], where=f"{where}.identity_sha256")
    if canonical_sha256(body) != declared:
        _fail(f"{where} identity_sha256 differs from its canonical body")
    return {**body, "identity_sha256": declared}


# ---------------------------------------------------------------------------
# Strict JSON decoding
# ---------------------------------------------------------------------------


def decode_strict_json(text: str, *, where: str) -> object:
    """Decode JSON refusing duplicate members and non-JSON constants."""

    if type(text) is not str:
        _fail(f"{where} must be text")

    def object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                _fail(f"{where} contains duplicate JSON member {key!r}")
            result[key] = item
        return result

    def reject_constant(value: str) -> object:
        _fail(f"{where} contains non-JSON constant {value}")
        raise AssertionError  # pragma: no cover - _fail always raises

    try:
        return json.loads(
            text,
            object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant,
        )
    except QualityPrefillContractError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise QualityPrefillContractError(f"{where} is not valid strict JSON") from exc


# ---------------------------------------------------------------------------
# The price table: one currency per table, always
# ---------------------------------------------------------------------------

_PRICE_TABLE_KEYS = frozenset(
    {"schema", "currency", "units", "context_sha256", "rows"}
)
_PRICE_ROW_KEYS = frozenset(
    {
        "candidate_id",
        "observation_id",
        "currency",
        "units",
        "measurement_status",
        "value",
        "provenance_sha256",
    }
)


def validate_price_table(value: object, *, where: str) -> Mapping[str, object]:
    """One table, one currency.

    Section 3.2: "Scalar artifacts never enter the joint cost payload... the
    solve gate ... rejects mixed currencies even when outer hashes and rosters
    match."  The header names the currency and every row must repeat it; a row
    that names the other currency is a refusal, not a warning.
    """

    body = _exact_mapping(value, keys=_PRICE_TABLE_KEYS, where=where)
    if body["schema"] != PRICE_TABLE_SCHEMA:
        _fail(f"{where}.schema must be {PRICE_TABLE_SCHEMA}")
    currency = _enum(body["currency"], where=f"{where}.currency", allowed=CURRENCIES)
    units = _string(body["units"], where=f"{where}.units", pattern=_TOKEN_RE)
    _sha256(body["context_sha256"], where=f"{where}.context_sha256")
    rows = _sequence(body["rows"], where=f"{where}.rows", minimum=1)
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row_where = f"{where}.rows[{index}]"
        row = _exact_mapping(raw, keys=_PRICE_ROW_KEYS, where=row_where)
        _string(row["candidate_id"], where=f"{row_where}.candidate_id", pattern=_ID_RE)
        observation_id = _string(
            row["observation_id"], where=f"{row_where}.observation_id", pattern=_ID_RE
        )
        if observation_id in seen:
            _fail(f"{row_where}.observation_id is duplicated in the table")
        seen.add(observation_id)
        row_currency = _enum(
            row["currency"], where=f"{row_where}.currency", allowed=CURRENCIES
        )
        if row_currency != currency:
            _fail(
                f"{row_where}.currency is {row_currency!r} in a {currency!r} table: "
                "one table carries one currency"
            )
        if _string(row["units"], where=f"{row_where}.units", pattern=_TOKEN_RE) != units:
            _fail(f"{row_where}.units differs from the table header units")
        status = _enum(
            row["measurement_status"],
            where=f"{row_where}.measurement_status",
            allowed=_COVERAGE_SET,
        )
        if currency == JOINT_CURRENCY and status not in _JOINT_STATUSES:
            _fail(
                f"{row_where}.measurement_status must be one of "
                f"{sorted(_JOINT_STATUSES)} for a {JOINT_CURRENCY} row"
            )
        _finite_number(row["value"], where=f"{row_where}.value", minimum=0.0)
        _sha256(row["provenance_sha256"], where=f"{row_where}.provenance_sha256")
    return body


# ---------------------------------------------------------------------------
# Manifest sections
# ---------------------------------------------------------------------------

MANIFEST_SECTIONS = (
    "source",
    "producer_reader",
    "data",
    "population",
    "menu",
    "screen",
    "quality",
    "runtime",
    "allocation",
    "validation",
    "execution",
)

_MANIFEST_BODY_KEYS = frozenset({"schema", "experiment_id", *MANIFEST_SECTIONS})
_MANIFEST_KEYS = _MANIFEST_BODY_KEYS | {"identity_sha256"}

#: The dotted paths whose value may be an unresolved sentinel in a draft.  Rob
#: has chosen none of them; the specification forbids inventing a default, so
#: the manifest carries the gap explicitly and ``freeze`` refuses on it.
RESOLVABLE_INPUTS = (
    "allocation.byte_budgets",
    "allocation.prefill_budget_sweep",
    "runtime.workload_manifests",
)

_SOURCE_KEYS = frozenset(
    {
        "model_content_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "profile",
        "tensor_map_sha256",
        "source_closure_sha256",
        "dependency_commits",
    }
)
_PRODUCER_READER_KEYS = frozenset(
    {
        "producer_commit",
        "producer_source_sha256",
        "reader_pin",
        "packaged_contract_sha256",
        "recipe_resolver_version",
    }
)
_DATA_KEYS = frozenset(
    {
        "calibration_id",
        "calibration_content_sha256",
        "capture_sha256",
        "census_sha256",
        "hessian_sha256",
        "source_execution_selectors",
        "development_dataset",
        "confirmation_dataset",
    }
)
_POPULATION_KEYS = frozenset(
    {
        "inventory",
        "eligibility_ledger",
        "pilot_roster",
        "confirmation_roster",
        "prior_exposure_ledger",
    }
)
_MENU_KEYS = frozenset(
    {
        "legal_roster",
        "recipe_partitions",
        "activation_route_map",
        "immutable_region_roster",
        "control_assignments",
    }
)
_SCREEN_KEYS = frozenset(
    {
        "aqua_settings",
        "transfer_forms",
        "audit_draws",
        "coverage_quotas",
        "uncertainty_rule",
        "refinement_policy",
    }
)
_QUALITY_KEYS = frozenset(
    {
        "joint_probe_count",
        "joint_probe_seed",
        "probe_identity_sha256",
        "calibration_normalization",
        "execution_microbatch",
        "arithmetic",
        "cache_settings",
        "source_residency",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "gpu_identity",
        "image_digest",
        "library_manifest_sha256",
        "workload_manifests",
        "numerical_policy",
        "timing_policy",
        "resource_policy",
    }
)
_ALLOCATION_KEYS = frozenset(
    {
        "byte_budgets",
        "prefill_budget_sweep",
        "device_constraints",
        "solver_limits",
        "tie_policy",
    }
)
_VALIDATION_KEYS = frozenset(
    {
        "baselines",
        "development_selection_rule",
        "confirmation_protocol",
        "quality_thresholds",
        "performance_thresholds",
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "phases",
        "setup_cost_reference",
        "pb_policy",
        "retry_policy",
        "withdrawal_policy",
    }
)

_DEPENDENCY_COMMIT_KEYS = frozenset({"name", "commit"})
_GPU_IDENTITY_KEYS = frozenset({"name", "uuid", "compute_capability"})
_WORKLOAD_KEYS = frozenset(
    {
        "workload_id",
        "prompt_tokens",
        "batch_size",
        "concurrency",
        "tensor_parallel",
        "graph_mode",
        "chunked_prefill",
        "manifest_sha256",
    }
)
_TIMING_POLICY_KEYS = frozenset({"warmup_iterations", "measured_iterations", "clock"})
_RESOURCE_POLICY_KEYS = frozenset(
    {"partition_producer", "independent_recomputation", "energy_attribution"}
)
_SOLVER_LIMIT_KEYS = frozenset({"max_seconds", "max_states"})
_PB_POLICY_KEYS = frozenset({"batch_policy", "progress_policy", "priority"})
_RETRY_POLICY_KEYS = frozenset({"max_attempts"})
_PHASE_ENTRY_KEYS = frozenset(
    {"task_roster", "dependencies", "work_limit", "cost_limit"}
)


def _validate_source(value: object, *, where: str) -> None:
    body = _exact_mapping(value, keys=_SOURCE_KEYS, where=where)
    for field in (
        "model_content_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "tensor_map_sha256",
        "source_closure_sha256",
    ):
        _sha256(body[field], where=f"{where}.{field}")
    _string(body["profile"], where=f"{where}.profile", pattern=_ID_RE)
    commits = _sequence(
        body["dependency_commits"], where=f"{where}.dependency_commits", minimum=1
    )
    names: set[str] = set()
    for index, raw in enumerate(commits):
        entry_where = f"{where}.dependency_commits[{index}]"
        entry = _exact_mapping(raw, keys=_DEPENDENCY_COMMIT_KEYS, where=entry_where)
        name = _string(entry["name"], where=f"{entry_where}.name", pattern=_ID_RE)
        if name in names:
            _fail(f"{entry_where}.name is duplicated")
        names.add(name)
        _string(entry["commit"], where=f"{entry_where}.commit", pattern=_GIT_SHA_RE)


def _validate_producer_reader(value: object, *, where: str) -> None:
    body = _exact_mapping(value, keys=_PRODUCER_READER_KEYS, where=where)
    _string(body["producer_commit"], where=f"{where}.producer_commit", pattern=_GIT_SHA_RE)
    _sha256(body["producer_source_sha256"], where=f"{where}.producer_source_sha256")
    _artifact_reference(body["reader_pin"], where=f"{where}.reader_pin")
    _sha256(body["packaged_contract_sha256"], where=f"{where}.packaged_contract_sha256")
    _string(
        body["recipe_resolver_version"],
        where=f"{where}.recipe_resolver_version",
        pattern=_VERSION_RE,
    )


def _validate_data(value: object, *, where: str) -> None:
    body = _exact_mapping(value, keys=_DATA_KEYS, where=where)
    _string(body["calibration_id"], where=f"{where}.calibration_id", pattern=_ID_RE)
    for field in (
        "calibration_content_sha256",
        "capture_sha256",
        "census_sha256",
        "hessian_sha256",
    ):
        _sha256(body[field], where=f"{where}.{field}")
    for field in (
        "source_execution_selectors",
        "development_dataset",
        "confirmation_dataset",
    ):
        _artifact_reference(body[field], where=f"{where}.{field}")
    if body["development_dataset"]["sha256"] == body["confirmation_dataset"]["sha256"]:
        _fail(
            f"{where}.confirmation_dataset must be untouched: it may not be the "
            "development dataset"
        )


def _validate_reference_section(
    value: object, *, keys: frozenset[str], where: str
) -> None:
    body = _exact_mapping(value, keys=keys, where=where)
    for field in sorted(keys):
        _artifact_reference(body[field], where=f"{where}.{field}")


def _validate_screen(value: object, *, where: str) -> None:
    body = _exact_mapping(value, keys=_SCREEN_KEYS, where=where)
    for field in ("aqua_settings", "transfer_forms", "audit_draws"):
        _artifact_reference(body[field], where=f"{where}.{field}")
    quotas = _exact_mapping(
        body["coverage_quotas"], keys=frozenset(COVERAGE_TYPES), where=f"{where}.coverage_quotas"
    )
    for coverage in COVERAGE_TYPES:
        _integer(
            quotas[coverage],
            where=f"{where}.coverage_quotas.{coverage}",
            minimum=0,
        )
    _named_version(body["uncertainty_rule"], where=f"{where}.uncertainty_rule")
    policy = _exact_mapping(
        body["refinement_policy"],
        keys=frozenset({"name", "version", "max_rounds"}),
        where=f"{where}.refinement_policy",
    )
    _string(policy["name"], where=f"{where}.refinement_policy.name", pattern=_ID_RE)
    _string(
        policy["version"], where=f"{where}.refinement_policy.version", pattern=_VERSION_RE
    )
    _integer(policy["max_rounds"], where=f"{where}.refinement_policy.max_rounds", minimum=0)


def _validate_quality(value: object, *, where: str) -> None:
    body = _exact_mapping(value, keys=_QUALITY_KEYS, where=where)
    _integer(body["joint_probe_count"], where=f"{where}.joint_probe_count", minimum=1)
    _integer(body["joint_probe_seed"], where=f"{where}.joint_probe_seed", minimum=0)
    _sha256(body["probe_identity_sha256"], where=f"{where}.probe_identity_sha256")
    _string(
        body["calibration_normalization"],
        where=f"{where}.calibration_normalization",
        pattern=_ID_RE,
    )
    _integer(body["execution_microbatch"], where=f"{where}.execution_microbatch", minimum=1)
    _string(body["arithmetic"], where=f"{where}.arithmetic", pattern=_TOKEN_RE)
    _artifact_reference(body["cache_settings"], where=f"{where}.cache_settings")
    _string(body["source_residency"], where=f"{where}.source_residency", pattern=_ID_RE)


def _validate_workload(value: object, *, where: str) -> str:
    body = _exact_mapping(value, keys=_WORKLOAD_KEYS, where=where)
    workload_id = _string(
        body["workload_id"], where=f"{where}.workload_id", pattern=_ID_RE
    )
    _integer(body["prompt_tokens"], where=f"{where}.prompt_tokens", minimum=1)
    for field in ("batch_size", "concurrency", "tensor_parallel"):
        _integer(body[field], where=f"{where}.{field}", minimum=1)
    _enum(body["graph_mode"], where=f"{where}.graph_mode", allowed=GRAPH_MODES)
    if type(body["chunked_prefill"]) is not bool:
        _fail(f"{where}.chunked_prefill must be a boolean")
    _sha256(body["manifest_sha256"], where=f"{where}.manifest_sha256")
    return workload_id


def _validate_runtime(value: object, *, where: str) -> None:
    body = _exact_mapping(value, keys=_RUNTIME_KEYS, where=where)
    gpu = _exact_mapping(
        body["gpu_identity"], keys=_GPU_IDENTITY_KEYS, where=f"{where}.gpu_identity"
    )
    _string(gpu["name"], where=f"{where}.gpu_identity.name")
    _string(gpu["uuid"], where=f"{where}.gpu_identity.uuid", pattern=_TOKEN_RE)
    capability = _sequence(
        gpu["compute_capability"], where=f"{where}.gpu_identity.compute_capability"
    )
    if len(capability) != 2:
        _fail(f"{where}.gpu_identity.compute_capability must be [major, minor]")
    for index, item in enumerate(capability):
        _integer(
            item,
            where=f"{where}.gpu_identity.compute_capability[{index}]",
            minimum=0,
            maximum=99,
        )
    _string(body["image_digest"], where=f"{where}.image_digest", pattern=_IMAGE_DIGEST_RE)
    _sha256(body["library_manifest_sha256"], where=f"{where}.library_manifest_sha256")
    workloads = body["workload_manifests"]
    if _is_unresolved(workloads):
        _unresolved_sentinel(workloads, where=f"{where}.workload_manifests")
    else:
        entries = _sequence(workloads, where=f"{where}.workload_manifests", minimum=1)
        seen: set[str] = set()
        for index, raw in enumerate(entries):
            workload_id = _validate_workload(
                raw, where=f"{where}.workload_manifests[{index}]"
            )
            if workload_id in seen:
                _fail(f"{where}.workload_manifests[{index}].workload_id is duplicated")
            seen.add(workload_id)
    _string(body["numerical_policy"], where=f"{where}.numerical_policy", pattern=_ID_RE)
    timing = _exact_mapping(
        body["timing_policy"], keys=_TIMING_POLICY_KEYS, where=f"{where}.timing_policy"
    )
    _integer(
        timing["warmup_iterations"],
        where=f"{where}.timing_policy.warmup_iterations",
        minimum=0,
    )
    _integer(
        timing["measured_iterations"],
        where=f"{where}.timing_policy.measured_iterations",
        minimum=1,
    )
    _string(timing["clock"], where=f"{where}.timing_policy.clock", pattern=_ID_RE)
    resource = _exact_mapping(
        body["resource_policy"],
        keys=_RESOURCE_POLICY_KEYS,
        where=f"{where}.resource_policy",
    )
    for field in sorted(_RESOURCE_POLICY_KEYS):
        _string(resource[field], where=f"{where}.resource_policy.{field}", pattern=_ID_RE)


def _validate_positive_integer_sweep(value: object, *, where: str) -> None:
    if _is_unresolved(value):
        _unresolved_sentinel(value, where=where)
        return
    entries = _sequence(value, where=where, minimum=1)
    seen: set[int] = set()
    for index, item in enumerate(entries):
        number = _integer(item, where=f"{where}[{index}]", minimum=1)
        if number in seen:
            _fail(f"{where}[{index}] repeats a value already in the sweep")
        seen.add(number)


def _validate_allocation(value: object, *, where: str) -> None:
    body = _exact_mapping(value, keys=_ALLOCATION_KEYS, where=where)
    _validate_positive_integer_sweep(body["byte_budgets"], where=f"{where}.byte_budgets")
    _validate_positive_integer_sweep(
        body["prefill_budget_sweep"], where=f"{where}.prefill_budget_sweep"
    )
    _artifact_reference(body["device_constraints"], where=f"{where}.device_constraints")
    limits = _exact_mapping(
        body["solver_limits"], keys=_SOLVER_LIMIT_KEYS, where=f"{where}.solver_limits"
    )
    _integer(limits["max_seconds"], where=f"{where}.solver_limits.max_seconds", minimum=1)
    _integer(limits["max_states"], where=f"{where}.solver_limits.max_states", minimum=1)
    _enum(body["tie_policy"], where=f"{where}.tie_policy", allowed=TIE_POLICIES)


def _validate_validation(value: object, *, where: str) -> None:
    body = _exact_mapping(value, keys=_VALIDATION_KEYS, where=where)
    for field in ("baselines", "quality_thresholds", "performance_thresholds"):
        _artifact_reference(body[field], where=f"{where}.{field}")
    _named_version(
        body["development_selection_rule"], where=f"{where}.development_selection_rule"
    )
    _named_version(body["confirmation_protocol"], where=f"{where}.confirmation_protocol")


def _validate_execution(value: object, *, where: str) -> None:
    body = _exact_mapping(value, keys=_EXECUTION_KEYS, where=where)
    phases = _exact_mapping(
        body["phases"], keys=frozenset(PHASE_BY_NAME), where=f"{where}.phases"
    )
    for name, spec in PHASE_BY_NAME.items():
        entry_where = f"{where}.phases.{name}"
        entry = _exact_mapping(phases[name], keys=_PHASE_ENTRY_KEYS, where=entry_where)
        _artifact_reference(entry["task_roster"], where=f"{entry_where}.task_roster")
        declared = _sequence(entry["dependencies"], where=f"{entry_where}.dependencies")
        declared_names = _unique_ids(declared, where=f"{entry_where}.dependencies")
        if declared_names != spec.dependencies:
            _fail(
                f"{entry_where}.dependencies must be {list(spec.dependencies)}: the "
                "phase DAG is part of the v1 contract"
            )
        _integer(entry["work_limit"], where=f"{entry_where}.work_limit", minimum=1)
        _integer(entry["cost_limit"], where=f"{entry_where}.cost_limit", minimum=1)
    _artifact_reference(
        body["setup_cost_reference"], where=f"{where}.setup_cost_reference"
    )
    pb_policy = _exact_mapping(
        body["pb_policy"], keys=_PB_POLICY_KEYS, where=f"{where}.pb_policy"
    )
    _string(pb_policy["batch_policy"], where=f"{where}.pb_policy.batch_policy", pattern=_ID_RE)
    _string(
        pb_policy["progress_policy"], where=f"{where}.pb_policy.progress_policy", pattern=_ID_RE
    )
    _integer(
        pb_policy["priority"], where=f"{where}.pb_policy.priority", minimum=-20, maximum=20
    )
    retry = _exact_mapping(
        body["retry_policy"], keys=_RETRY_POLICY_KEYS, where=f"{where}.retry_policy"
    )
    _integer(retry["max_attempts"], where=f"{where}.retry_policy.max_attempts", minimum=1)
    _string(body["withdrawal_policy"], where=f"{where}.withdrawal_policy", pattern=_ID_RE)


_SECTION_VALIDATORS = {
    "source": _validate_source,
    "producer_reader": _validate_producer_reader,
    "data": _validate_data,
    "population": lambda value, *, where: _validate_reference_section(
        value, keys=_POPULATION_KEYS, where=where
    ),
    "menu": lambda value, *, where: _validate_reference_section(
        value, keys=_MENU_KEYS, where=where
    ),
    "screen": _validate_screen,
    "quality": _validate_quality,
    "runtime": _validate_runtime,
    "allocation": _validate_allocation,
    "validation": _validate_validation,
    "execution": _validate_execution,
}
assert set(_SECTION_VALIDATORS) == set(MANIFEST_SECTIONS)


Mode = Literal["draft", "frozen"]


def validate_manifest(
    value: object, *, mode: Mode = "frozen"
) -> dict[str, object]:
    """Validate one run manifest.

    ``mode="draft"`` is an unsealed plan that may still carry unresolved
    sentinels in :data:`RESOLVABLE_INPUTS`.  ``mode="frozen"`` requires the
    ``identity_sha256`` seal and refuses every sentinel.
    """

    if mode not in ("draft", "frozen"):
        _fail("manifest validation mode must be 'draft' or 'frozen'")
    keys = _MANIFEST_KEYS if mode == "frozen" else _MANIFEST_BODY_KEYS
    body = _exact_mapping(value, keys=keys, where="manifest")
    if body["schema"] != MANIFEST_SCHEMA:
        _fail(f"manifest schema must be {MANIFEST_SCHEMA}")
    _string(body["experiment_id"], where="manifest.experiment_id", pattern=_ID_RE)
    for section in MANIFEST_SECTIONS:
        _SECTION_VALIDATORS[section](body[section], where=section)
    if mode == "frozen":
        unresolved = unresolved_inputs(body)
        if unresolved:
            _fail(
                "a frozen manifest cannot carry unresolved inputs: "
                + ", ".join(unresolved)
            )
        return _check_seal(body, where="manifest")
    return dict(body)


def parse_manifest(text: str, *, mode: Mode = "frozen") -> dict[str, object]:
    """Strictly decode manifest JSON, then validate it."""

    decoded = decode_strict_json(text, where="manifest")
    if not isinstance(decoded, Mapping):
        _fail("manifest JSON root must be an object")
    return validate_manifest(decoded, mode=mode)


def unresolved_inputs(manifest: Mapping[str, object]) -> list[str]:
    """Every unresolved dotted path, not just the first.

    A refusal must be able to name the complete set: the byte budget, the
    prefill budget and the workload feasibility inputs are all open today, and
    reporting them one at a time would make three round trips out of one.
    """

    found: list[str] = []
    for path in RESOLVABLE_INPUTS:
        section, field = path.split(".", 1)
        container = manifest.get(section)
        if not isinstance(container, Mapping):
            continue
        if _is_unresolved(container.get(field)):
            found.append(path)
    return found


def freeze_manifest(draft: Mapping[str, object]) -> dict[str, object]:
    """Seal a validated draft into immutable plan bytes.

    Refuses while any required value is unresolved, naming all of them.
    """

    body = validate_manifest(draft, mode="draft")
    unresolved = unresolved_inputs(body)
    if unresolved:
        reasons = "; ".join(
            f"{path} ({body[path.split('.', 1)[0]][path.split('.', 1)[1]]['unresolved']})"  # type: ignore[index]
            for path in unresolved
        )
        _fail(
            "cannot freeze: "
            + str(len(unresolved))
            + " required value(s) are unresolved: "
            + reasons
        )
    return validate_manifest(_seal(body), mode="frozen")


# ---------------------------------------------------------------------------
# Evidence envelopes (specification section 3.2)
# ---------------------------------------------------------------------------

_CANDIDATE_KEYS = frozenset(
    {
        "schema",
        "candidate_id",
        "unit_map",
        "family",
        "rate_q256",
        "recipe",
        "activation_contract",
        "route_status",
        "source_sha256",
        "footprint_bytes",
        "support_facts",
    }
)
_OBSERVATION_KEYS = frozenset(
    {
        "schema",
        "observation_id",
        "candidate_id",
        "context_sha256",
        "phase_id",
        "task_id",
        "inner_schema",
        "currency",
        "units",
        "measurement_status",
        "value",
        "artifacts",
        "action_key",
        "attempt",
        "receipt_sha256",
        "validity",
        "unknown_reason",
    }
)
_SCREEN_DECISION_KEYS = frozenset(
    {"schema", "decision_id", "rule", "dispositions"}
)
_DISPOSITION_KEYS = frozenset(
    {
        "candidate_id",
        "disposition",
        "measurement_status",
        "evidence_observation_ids",
        "reason",
        "uncertainty",
    }
)
_TRANSFER_SEAL_KEYS = frozenset(
    {
        "schema",
        "seal_id",
        "form",
        "segment_id",
        "source_candidate_ids",
        "endpoint_candidate_ids",
        "predictions",
        "prior_exposure_ledger",
        "target_plan_sha256",
    }
)
_PREDICTION_KEYS = frozenset(
    {"candidate_id", "predicted_value", "currency", "units", "measurement_status"}
)
_PHASE_RESULT_KEYS = frozenset(
    {
        "schema",
        "phase_id",
        "plan_identity_sha256",
        "parent_phase_ids",
        "expected_task_ids",
        "child_receipts",
        "merged_output_sha256",
        "terminal_state",
        "coverage",
        "refusal_reasons",
    }
)
_CHILD_RECEIPT_KEYS = frozenset({"task_id", "action_key", "receipt_sha256"})
_ASSIGNMENT_KEYS = frozenset(
    {
        "schema",
        "assignment_id",
        "member_recipes",
        "serving_unit_recipes",
        "total_bytes",
        "price_table",
        "context_sha256",
        "resource_proposal",
        "eligibility",
        "exported_artifact_sha256",
    }
)
_ELIGIBILITY_KEYS = frozenset({"route_status", "target_platform", "native"})
_FRONTIER_REPORT_KEYS = frozenset(
    {
        "schema",
        "report_id",
        "proposed_points",
        "served_points",
        "uncertainty",
        "selected_neighbors",
        "controls",
        "omissions",
        "evidence_links",
        "commands",
    }
)
_POINT_KEYS = frozenset(
    {
        "point_id",
        "assignment_id",
        "bytes",
        "prefill_budget",
        "quality_value",
        "currency",
        "units",
        "measurement_status",
    }
)

_TERMINAL_RESULT_STATES = frozenset(
    {"complete", "failed", "withdrawn", "requires_plan_revision"}
)


def _envelope(value: object, *, keys: frozenset[str], schema: str, where: str):
    body = _exact_mapping(value, keys=keys | {"identity_sha256"}, where=where)
    if body["schema"] != schema:
        _fail(f"{where}.schema must be {schema}")
    return body


def validate_candidate(value: object, *, where: str = "candidate") -> dict[str, object]:
    """A candidate identity contains no price (section 3.2)."""

    body = _envelope(value, keys=_CANDIDATE_KEYS, schema=CANDIDATE_SCHEMA, where=where)
    _string(body["candidate_id"], where=f"{where}.candidate_id", pattern=_ID_RE)
    _artifact_reference(body["unit_map"], where=f"{where}.unit_map")
    _string(body["family"], where=f"{where}.family", pattern=_TOKEN_RE)
    _integer(body["rate_q256"], where=f"{where}.rate_q256", minimum=1)
    _artifact_reference(body["recipe"], where=f"{where}.recipe")
    _string(
        body["activation_contract"], where=f"{where}.activation_contract", pattern=_TOKEN_RE
    )
    _enum(body["route_status"], where=f"{where}.route_status", allowed=ROUTE_STATUSES)
    _sha256(body["source_sha256"], where=f"{where}.source_sha256")
    _integer(body["footprint_bytes"], where=f"{where}.footprint_bytes", minimum=1)
    _artifact_reference(body["support_facts"], where=f"{where}.support_facts")
    return _check_seal(body, where=where)


def validate_observation(value: object, *, where: str = "observation") -> dict[str, object]:
    body = _envelope(
        value, keys=_OBSERVATION_KEYS, schema=OBSERVATION_SCHEMA, where=where
    )
    for field in ("observation_id", "candidate_id", "phase_id", "task_id"):
        _string(body[field], where=f"{where}.{field}", pattern=_ID_RE)
    if body["phase_id"] not in PHASE_BY_NAME:
        _fail(f"{where}.phase_id is not a phase of the v1 DAG")
    _sha256(body["context_sha256"], where=f"{where}.context_sha256")
    _string(body["inner_schema"], where=f"{where}.inner_schema", pattern=_TOKEN_RE)
    currency = _enum(body["currency"], where=f"{where}.currency", allowed=CURRENCIES)
    _string(body["units"], where=f"{where}.units", pattern=_TOKEN_RE)
    status = _enum(
        body["measurement_status"],
        where=f"{where}.measurement_status",
        allowed=_COVERAGE_SET,
    )
    if currency == JOINT_CURRENCY and status not in _JOINT_STATUSES:
        _fail(
            f"{where}.measurement_status must be one of {sorted(_JOINT_STATUSES)} "
            f"for a {JOINT_CURRENCY} observation"
        )
    _finite_number(body["value"], where=f"{where}.value", minimum=0.0)
    artifacts = _sequence(body["artifacts"], where=f"{where}.artifacts", minimum=1)
    for index, raw in enumerate(artifacts):
        _artifact_reference(raw, where=f"{where}.artifacts[{index}]")
    _string(body["action_key"], where=f"{where}.action_key", pattern=_TOKEN_RE)
    _integer(body["attempt"], where=f"{where}.attempt", minimum=1)
    _sha256(body["receipt_sha256"], where=f"{where}.receipt_sha256")
    validity = _enum(body["validity"], where=f"{where}.validity", allowed=VALIDITY)
    reason = body["unknown_reason"]
    if type(reason) is not str:
        _fail(f"{where}.unknown_reason must be a string")
    if validity == "unknown" and not reason:
        _fail(f"{where}.unknown_reason is required when validity is 'unknown'")
    if validity == "valid" and reason:
        _fail(f"{where}.unknown_reason must be empty when validity is 'valid'")
    return _check_seal(body, where=where)


def validate_screen_decision(
    value: object, *, where: str = "screen_decision"
) -> dict[str, object]:
    body = _envelope(
        value, keys=_SCREEN_DECISION_KEYS, schema=SCREEN_DECISION_SCHEMA, where=where
    )
    _string(body["decision_id"], where=f"{where}.decision_id", pattern=_ID_RE)
    _named_version(body["rule"], where=f"{where}.rule")
    dispositions = _sequence(
        body["dispositions"], where=f"{where}.dispositions", minimum=1
    )
    seen: set[str] = set()
    for index, raw in enumerate(dispositions):
        entry_where = f"{where}.dispositions[{index}]"
        entry = _exact_mapping(raw, keys=_DISPOSITION_KEYS, where=entry_where)
        candidate_id = _string(
            entry["candidate_id"], where=f"{entry_where}.candidate_id", pattern=_ID_RE
        )
        if candidate_id in seen:
            _fail(f"{entry_where}.candidate_id already has a disposition")
        seen.add(candidate_id)
        _enum(entry["disposition"], where=f"{entry_where}.disposition", allowed=DISPOSITIONS)
        _enum(
            entry["measurement_status"],
            where=f"{entry_where}.measurement_status",
            allowed=_COVERAGE_SET,
        )
        evidence = _sequence(
            entry["evidence_observation_ids"], where=f"{entry_where}.evidence_observation_ids"
        )
        _unique_ids(evidence, where=f"{entry_where}.evidence_observation_ids")
        _string(entry["reason"], where=f"{entry_where}.reason")
        _finite_number(entry["uncertainty"], where=f"{entry_where}.uncertainty", minimum=0.0)
    return _check_seal(body, where=where)


def validate_transfer_seal(
    value: object, *, where: str = "transfer_seal"
) -> dict[str, object]:
    """Sources and endpoints only before the seal; predictions are predicted."""

    body = _envelope(
        value, keys=_TRANSFER_SEAL_KEYS, schema=TRANSFER_SEAL_SCHEMA, where=where
    )
    _string(body["seal_id"], where=f"{where}.seal_id", pattern=_ID_RE)
    _named_version(body["form"], where=f"{where}.form")
    _string(body["segment_id"], where=f"{where}.segment_id", pattern=_ID_RE)
    sources = _unique_ids(
        _sequence(body["source_candidate_ids"], where=f"{where}.source_candidate_ids", minimum=1),
        where=f"{where}.source_candidate_ids",
    )
    endpoints = _unique_ids(
        _sequence(
            body["endpoint_candidate_ids"], where=f"{where}.endpoint_candidate_ids", minimum=1
        ),
        where=f"{where}.endpoint_candidate_ids",
    )
    if set(sources) & set(endpoints):
        _fail(f"{where} sources and endpoints must be separate boundaries")
    predictions = _sequence(body["predictions"], where=f"{where}.predictions", minimum=1)
    predicted: set[str] = set()
    for index, raw in enumerate(predictions):
        entry_where = f"{where}.predictions[{index}]"
        entry = _exact_mapping(raw, keys=_PREDICTION_KEYS, where=entry_where)
        candidate_id = _string(
            entry["candidate_id"], where=f"{entry_where}.candidate_id", pattern=_ID_RE
        )
        if candidate_id in predicted:
            _fail(f"{entry_where}.candidate_id is predicted twice")
        predicted.add(candidate_id)
        if candidate_id in set(sources) | set(endpoints):
            _fail(
                f"{entry_where}.candidate_id is a source or endpoint: a seal predicts "
                "its complementary targets, not its own inputs"
            )
        value_number = _finite_number(
            entry["predicted_value"], where=f"{entry_where}.predicted_value", minimum=0.0
        )
        if value_number <= 0.0:
            _fail(f"{entry_where}.predicted_value must be positive")
        _enum(entry["currency"], where=f"{entry_where}.currency", allowed=CURRENCIES)
        _string(entry["units"], where=f"{entry_where}.units", pattern=_TOKEN_RE)
        status = _enum(
            entry["measurement_status"],
            where=f"{entry_where}.measurement_status",
            allowed=_COVERAGE_SET,
        )
        if status != "predicted_screen":
            _fail(
                f"{entry_where}.measurement_status must be 'predicted_screen': a sealed "
                "prediction is a screen, never a measurement"
            )
    _artifact_reference(
        body["prior_exposure_ledger"], where=f"{where}.prior_exposure_ledger"
    )
    _sha256(body["target_plan_sha256"], where=f"{where}.target_plan_sha256")
    return _check_seal(body, where=where)


def validate_phase_result(
    value: object, *, where: str = "phase_result"
) -> dict[str, object]:
    """``complete`` requires an exact cover of the expected task set."""

    body = _envelope(
        value, keys=_PHASE_RESULT_KEYS, schema=PHASE_RESULT_SCHEMA, where=where
    )
    phase_id = _string(body["phase_id"], where=f"{where}.phase_id", pattern=_ID_RE)
    if phase_id not in PHASE_BY_NAME:
        _fail(f"{where}.phase_id is not a phase of the v1 DAG")
    _sha256(body["plan_identity_sha256"], where=f"{where}.plan_identity_sha256")
    parents = _unique_ids(
        _sequence(body["parent_phase_ids"], where=f"{where}.parent_phase_ids"),
        where=f"{where}.parent_phase_ids",
    )
    if parents != PHASE_BY_NAME[phase_id].dependencies:
        _fail(
            f"{where}.parent_phase_ids must be "
            f"{list(PHASE_BY_NAME[phase_id].dependencies)}"
        )
    expected = _unique_ids(
        _sequence(body["expected_task_ids"], where=f"{where}.expected_task_ids", minimum=1),
        where=f"{where}.expected_task_ids",
    )
    receipts = _sequence(body["child_receipts"], where=f"{where}.child_receipts")
    covered: list[str] = []
    for index, raw in enumerate(receipts):
        entry_where = f"{where}.child_receipts[{index}]"
        entry = _exact_mapping(raw, keys=_CHILD_RECEIPT_KEYS, where=entry_where)
        covered.append(_string(entry["task_id"], where=f"{entry_where}.task_id", pattern=_ID_RE))
        _string(entry["action_key"], where=f"{entry_where}.action_key", pattern=_TOKEN_RE)
        _sha256(entry["receipt_sha256"], where=f"{entry_where}.receipt_sha256")
    if len(set(covered)) != len(covered):
        _fail(f"{where}.child_receipts contains a duplicate task_id")
    foreign = sorted(set(covered) - set(expected))
    if foreign:
        _fail(f"{where}.child_receipts contains foreign task ids: {foreign}")
    _sha256(body["merged_output_sha256"], where=f"{where}.merged_output_sha256")
    terminal = _enum(
        body["terminal_state"], where=f"{where}.terminal_state", allowed=_TERMINAL_RESULT_STATES
    )
    if terminal == "complete" and set(covered) != set(expected):
        missing = sorted(set(expected) - set(covered))
        _fail(
            f"{where} cannot be 'complete' without an exact cover: missing {missing}"
        )
    coverage = _exact_mapping(
        body["coverage"], keys=frozenset(COVERAGE_TYPES), where=f"{where}.coverage"
    )
    for name in COVERAGE_TYPES:
        _integer(coverage[name], where=f"{where}.coverage.{name}", minimum=0)
    reasons = _sequence(body["refusal_reasons"], where=f"{where}.refusal_reasons")
    for index, item in enumerate(reasons):
        _string(item, where=f"{where}.refusal_reasons[{index}]")
    if terminal != "complete" and not reasons:
        _fail(f"{where}.refusal_reasons is required when the phase is not complete")
    return _check_seal(body, where=where)


def validate_assignment(value: object, *, where: str = "assignment") -> dict[str, object]:
    body = _envelope(value, keys=_ASSIGNMENT_KEYS, schema=ASSIGNMENT_SCHEMA, where=where)
    _string(body["assignment_id"], where=f"{where}.assignment_id", pattern=_ID_RE)
    _artifact_reference(body["member_recipes"], where=f"{where}.member_recipes")
    _artifact_reference(
        body["serving_unit_recipes"], where=f"{where}.serving_unit_recipes"
    )
    _integer(body["total_bytes"], where=f"{where}.total_bytes", minimum=1)
    table = validate_price_table(body["price_table"], where=f"{where}.price_table")
    _sha256(body["context_sha256"], where=f"{where}.context_sha256")
    if table["context_sha256"] != body["context_sha256"]:
        _fail(f"{where}.price_table.context_sha256 differs from the assignment context")
    _artifact_reference(body["resource_proposal"], where=f"{where}.resource_proposal")
    eligibility = _exact_mapping(
        body["eligibility"], keys=_ELIGIBILITY_KEYS, where=f"{where}.eligibility"
    )
    route_status = _enum(
        eligibility["route_status"],
        where=f"{where}.eligibility.route_status",
        allowed=ROUTE_STATUSES,
    )
    _string(
        eligibility["target_platform"],
        where=f"{where}.eligibility.target_platform",
        pattern=_TOKEN_RE,
    )
    if type(eligibility["native"]) is not bool:
        _fail(f"{where}.eligibility.native must be a boolean")
    if eligibility["native"] and route_status == "unbacked":
        _fail(
            f"{where}.eligibility cannot claim a native route while route_status "
            "is 'unbacked'"
        )
    _sha256(body["exported_artifact_sha256"], where=f"{where}.exported_artifact_sha256")
    return _check_seal(body, where=where)


def _validate_points(value: object, *, where: str) -> tuple[str, ...]:
    entries = _sequence(value, where=where)
    ids: list[str] = []
    for index, raw in enumerate(entries):
        entry_where = f"{where}[{index}]"
        entry = _exact_mapping(raw, keys=_POINT_KEYS, where=entry_where)
        ids.append(_string(entry["point_id"], where=f"{entry_where}.point_id", pattern=_ID_RE))
        _string(entry["assignment_id"], where=f"{entry_where}.assignment_id", pattern=_ID_RE)
        _integer(entry["bytes"], where=f"{entry_where}.bytes", minimum=1)
        _integer(entry["prefill_budget"], where=f"{entry_where}.prefill_budget", minimum=1)
        _finite_number(entry["quality_value"], where=f"{entry_where}.quality_value", minimum=0.0)
        _enum(entry["currency"], where=f"{entry_where}.currency", allowed=CURRENCIES)
        _string(entry["units"], where=f"{entry_where}.units", pattern=_TOKEN_RE)
        _enum(
            entry["measurement_status"],
            where=f"{entry_where}.measurement_status",
            allowed=_COVERAGE_SET,
        )
    if len(set(ids)) != len(ids):
        _fail(f"{where} contains duplicate point identifiers")
    return tuple(ids)


def validate_frontier_report(
    value: object, *, where: str = "frontier_report"
) -> dict[str, object]:
    """Proposed and served points stay separate tables (section 3.2)."""

    body = _envelope(
        value, keys=_FRONTIER_REPORT_KEYS, schema=FRONTIER_REPORT_SCHEMA, where=where
    )
    _string(body["report_id"], where=f"{where}.report_id", pattern=_ID_RE)
    proposed = _validate_points(body["proposed_points"], where=f"{where}.proposed_points")
    served = _validate_points(body["served_points"], where=f"{where}.served_points")
    if set(proposed) & set(served):
        _fail(
            f"{where} reports a point as both proposed and served: they are separate "
            "tables, not one table with a flag"
        )
    _artifact_reference(body["uncertainty"], where=f"{where}.uncertainty")
    _unique_ids(
        _sequence(body["selected_neighbors"], where=f"{where}.selected_neighbors"),
        where=f"{where}.selected_neighbors",
    )
    _unique_ids(
        _sequence(body["controls"], where=f"{where}.controls"),
        where=f"{where}.controls",
    )
    for index, item in enumerate(
        _sequence(body["omissions"], where=f"{where}.omissions")
    ):
        _string(item, where=f"{where}.omissions[{index}]")
    for index, raw in enumerate(
        _sequence(body["evidence_links"], where=f"{where}.evidence_links")
    ):
        _artifact_reference(raw, where=f"{where}.evidence_links[{index}]")
    commands = _sequence(body["commands"], where=f"{where}.commands", minimum=1)
    for index, item in enumerate(commands):
        _string(item, where=f"{where}.commands[{index}]")
    return _check_seal(body, where=where)


ENVELOPE_VALIDATORS: Mapping[str, object] = MappingProxyType(
    {
        "candidate": validate_candidate,
        "observation": validate_observation,
        "screen_decision": validate_screen_decision,
        "transfer_seal": validate_transfer_seal,
        "phase_result": validate_phase_result,
        "assignment": validate_assignment,
        "frontier_report": validate_frontier_report,
    }
)

ENVELOPE_SCHEMAS: Mapping[str, str] = MappingProxyType(
    {
        "candidate": CANDIDATE_SCHEMA,
        "observation": OBSERVATION_SCHEMA,
        "screen_decision": SCREEN_DECISION_SCHEMA,
        "transfer_seal": TRANSFER_SEAL_SCHEMA,
        "phase_result": PHASE_RESULT_SCHEMA,
        "assignment": ASSIGNMENT_SCHEMA,
        "frontier_report": FRONTIER_REPORT_SCHEMA,
    }
)


def seal_envelope(kind: str, body: Mapping[str, object]) -> dict[str, object]:
    """Seal and validate one envelope body of ``kind``."""

    if kind not in ENVELOPE_VALIDATORS:
        _fail(f"unknown evidence envelope {kind!r}")
    validator = ENVELOPE_VALIDATORS[kind]
    return validator(_seal(body), where=kind)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Phase state machine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhaseState:
    """One phase's durable state.  Advanced only by :func:`transition`."""

    phase: str
    state: str
    attempt: int
    reason: str


def initial_phase_states() -> dict[str, PhaseState]:
    """Every phase of the v1 DAG, ``planned``, on attempt 1."""

    return {
        spec.phase: PhaseState(phase=spec.phase, state="planned", attempt=1, reason="")
        for spec in PHASE_DAG
    }


def transition(
    current: PhaseState,
    target: str,
    *,
    max_attempts: int,
    receipts: Sequence[Mapping[str, object]] | None = None,
    expected_task_ids: Sequence[str] | None = None,
    reason: str = "",
) -> PhaseState:
    """Advance one phase, or refuse.

    Entering ``complete`` requires an exact cover of ``expected_task_ids`` by
    ``receipts``: a launch, a heartbeat or a subprocess message is not a
    completion (section 4).  Entering ``requires_plan_revision`` requires the
    missing capacity, source, route or input to be named.  ``failed -> ready``
    is a retry and stays inside the original attempt budget.
    """

    if current.phase not in PHASE_BY_NAME:
        _fail(f"{current.phase!r} is not a phase of the v1 DAG")
    if target not in _STATE_SET:
        _fail(f"{target!r} is not a phase state")
    allowed = LEGAL_TRANSITIONS[current.state]
    if target not in allowed:
        _fail(
            f"{current.phase}: {current.state} -> {target} is not a legal transition "
            f"(legal: {sorted(allowed) or 'none, terminal'})"
        )
    _integer(max_attempts, where="max_attempts", minimum=1)
    attempt = current.attempt
    if current.state == "failed" and target == "ready":
        attempt = current.attempt + 1
        if attempt > max_attempts:
            _fail(
                f"{current.phase}: retry {attempt} exceeds the attempt budget "
                f"{max_attempts}; this is a plan revision, not a retry"
            )
    if target == "complete":
        if receipts is None or expected_task_ids is None:
            _fail(
                f"{current.phase}: 'complete' requires its child receipts and the "
                "expected task set"
            )
        covered = [
            _string(entry["task_id"], where="receipt.task_id", pattern=_ID_RE)
            for entry in receipts
        ]
        if len(set(covered)) != len(covered):
            _fail(f"{current.phase}: duplicate child receipt task_id")
        expected = set(_unique_ids(list(expected_task_ids), where="expected_task_ids"))
        if set(covered) != expected:
            missing = sorted(expected - set(covered))
            foreign = sorted(set(covered) - expected)
            _fail(
                f"{current.phase}: 'complete' needs an exact cover "
                f"(missing={missing}, foreign={foreign})"
            )
    if target in ("requires_plan_revision", "withdrawn", "failed") and not reason:
        _fail(f"{current.phase}: entering {target!r} requires a named reason")
    return PhaseState(
        phase=current.phase, state=target, attempt=attempt, reason=reason
    )


def ready_phases(states: Mapping[str, PhaseState]) -> tuple[str, ...]:
    """Phases whose dependencies are all ``complete`` and are not yet started."""

    out: list[str] = []
    for spec in PHASE_DAG:
        state = states.get(spec.phase)
        if state is None or state.state not in ("planned", "waiting_dependencies"):
            continue
        if all(
            states[dependency].state == "complete" for dependency in spec.dependencies
        ):
            out.append(spec.phase)
    return tuple(out)
