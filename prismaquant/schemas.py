"""Runtime schema checks for PrismaQuant file handoffs.

The pipeline passes several pickle/JSON artifacts between long-running
steps.  These validators intentionally check only the structural contract
that downstream code relies on, so older artifacts with extra fields still
load while malformed artifacts fail before optimization or export begins.

The end of this module holds the checks that contract readers across the
package share (PQ #1300): the strict JSON reader (:func:`strict_json_loads`)
and one refusal vocabulary (:class:`Contract`). Each reader keeps its own
exception type and message text; what they share is the logic.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
from numbers import Integral, Real
from pathlib import PurePosixPath
import re
from typing import Any, NoReturn, NotRequired, TypedDict


class CostEntry(TypedDict, total=False):
    """Structural type for one per-format cost row.

    Older artifacts are allowed to omit ``cost_source``; live producers should
    set it when a cost was rewritten or comes from a non-default surrogate so
    allocator logs can explain which objective priced the decision.
    """

    weight_mse: float
    output_mse: float
    rel_output_mse: float
    predicted_dloss: float
    fisher_output_mse: float
    output_mse_measured: bool
    cost_source: NotRequired[str]
    weight_mse_per_expert: NotRequired[list[float]]
    cost_source_per_expert: NotRequired[list[str]]
    error: str


class SchemaValidationError(ValueError):
    """Raised when a PrismaQuant handoff artifact is structurally invalid."""


#: Shape of a retired codebook rung name (the Gridbook lane, archived
#: 2026-09-02; remnant cleanup 2026-09-25, #1304). A shape test only,
#: kept here so the torch-free readers (this module, ``layer_config``)
#: can spot one; the authority is
#: ``format_registry.RETIRED_CODEBOOK_FORMAT_RE``, reached through
#: ``get_format`` by :func:`refuse_retired_codebook_format`.
RETIRED_CODEBOOK_NAME_RE = re.compile(r"^(?:NVFP4_CB_K|FP8_CB_K)\d+$")
RETIRED_CODEBOOK_ARCHIVE = "archive/gridbook_lane_2026-09-02"
#: Cost-row fields only the retired codebook lane's min-chain encoder wrote.
_RETIRED_CODEBOOK_COST_FIELDS = (
    "cb_minchain_identity_per_expert",
    "cb_minchain_interpolation",
)


#: ``cost_source`` spellings only the retired codebook lane's RD-ladder
#: interpolation stamped (archived 2026-09-25, #1304). No live producer writes
#: them; the Tessera campaign's fitted rows use their own spelling.
RETIRED_LADDER_COST_SOURCES = frozenset({"band_interpolated", "mixed"})


#: Metadata keys that bound a cache or cost table to a source-class format
#: plan, a codebook-campaign feature archived with the lane (#1345).
RETIRED_FORMAT_PLAN_IDENTITY_KEYS = (
    "format_plan_identity_sha256",
    "source_format_plan_identity_sha256",
)


def refuse_retired_format_plan_identity(record, where: str) -> None:
    """Raise ``RetiredFormatError`` when ``record`` names a format-plan identity.

    ``source_class_format_plan`` scoped each qname's menu to its source class.
    It was archived with the codebook lane (#1345), so a cache or cost table
    built under a plan can no longer be read back against that plan. Reading
    it without the plan would price rows the plan withheld, so it refuses.
    A null value, which every artifact since then carries, passes.
    """
    if not _is_mapping(record):
        return
    for key in RETIRED_FORMAT_PLAN_IDENTITY_KEYS:
        if record.get(key) is None:
            continue
        from prismaquant.format_registry import RetiredFormatError

        raise RetiredFormatError(
            f"{where}: {key}={record[key]!r} binds it to a source-class format "
            f"plan; that feature was archived with the Gridbook codebook lane "
            f"(#1345). See {RETIRED_CODEBOOK_ARCHIVE}/README.md."
        )


def refuse_retired_ladder_cost_source(entry, where: str = "cost row") -> None:
    """Raise ``RetiredFormatError`` when ``entry`` was priced by the retired
    codebook lane's RD ladder; return for every other row.

    ``validate_cost_payload`` calls it on every row it admits, and
    ``allocator_candidates.cost_entry_is_band_interpolated`` calls it on the
    rows it prices, so a table that skipped validation still refuses.
    """
    source = entry.get("cost_source") if _is_mapping(entry) else None
    if source not in RETIRED_LADDER_COST_SOURCES:
        return
    from prismaquant.format_registry import RetiredFormatError

    raise RetiredFormatError(
        f"{where}: cost_source {source!r} was stamped by the retired Gridbook "
        f"codebook lane's RD-ladder interpolation (archived 2026-09-25, "
        f"#1304); a row carrying it cannot be priced. See "
        f"{RETIRED_CODEBOOK_ARCHIVE}/README.md."
    )


def refuse_retired_codebook_format(name: str) -> None:
    """Raise ``RetiredFormatError`` when ``name`` is a retired codebook rung.

    Returns for every other name. ``format_registry`` imports torch, so it is
    imported only on this refusal path and the caller stays torch-free.
    """
    if not RETIRED_CODEBOOK_NAME_RE.fullmatch(str(name).upper()):
        return
    from prismaquant.format_registry import get_format

    get_format(str(name))
    raise AssertionError(f"{name!r} is a retired codebook rung but resolved")


def _label(path: str | None) -> str:
    return str(path) if path else "<memory>"


def _fail(path: str | None, where: str, message: str) -> None:
    raise SchemaValidationError(f"{_label(path)}:{where}: {message}")


def _is_mapping(value) -> bool:
    return isinstance(value, Mapping)


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _as_non_negative_int(value, path: str | None, where: str) -> int:
    if isinstance(value, bool):
        _fail(path, where, "expected a non-negative integer")
    try:
        out = int(value)
    except (TypeError, ValueError):
        _fail(path, where, "expected a non-negative integer")
    if out < 0:
        _fail(path, where, "expected a non-negative integer")
    return out


def _as_number(value, path: str | None, where: str) -> float:
    if not _is_number(value):
        _fail(path, where, "expected a number")
    return float(value)


def _as_finite_cost_number(value, path: str | None, where: str) -> float:
    """A usable cost must be finite; other handoff schemas keep their rules."""
    out = _as_number(value, path, where)
    if not math.isfinite(out):
        _fail(path, where, "expected a finite number")
    return out


def _validate_router_number_map(
    payload,
    field: str,
    path: str | None,
    *,
    integral_values: bool = False,
) -> None:
    value = payload.get(field)
    if value is None:
        return
    if not _is_mapping(value):
        _fail(path, f".{field}", "must be a mapping when present")
    for router, values in value.items():
        if not isinstance(router, str):
            _fail(path, f".{field}", "router keys must be strings")
        if not _is_mapping(values):
            _fail(path, f".{field}[{router!r}]", "must be a mapping")
        for eid, count in values.items():
            if not isinstance(eid, (str, Integral)) or isinstance(eid, bool):
                _fail(path, f".{field}[{router!r}]", "expert ids must be strings or ints")
            if integral_values:
                _as_non_negative_int(count, path, f".{field}[{router!r}][{eid!r}]")
            else:
                _as_number(count, path, f".{field}[{router!r}][{eid!r}]")


def validate_probe_payload(payload, path: str | None = None):
    """Validate the merged sensitivity-probe pickle contract."""
    if not _is_mapping(payload):
        _fail(path, "", "probe payload is not a mapping")
    stats = payload.get("stats")
    if not _is_mapping(stats):
        _fail(path, ".stats", "missing or not a mapping")
    for name, entry in stats.items():
        if not isinstance(name, str):
            _fail(path, ".stats", "stat keys must be strings")
        if not _is_mapping(entry):
            _fail(path, f".stats[{name!r}]", "entry is not a mapping")
        if "h_trace" not in entry:
            _fail(path, f".stats[{name!r}].h_trace", "required field missing")
        if "n_params" not in entry:
            _fail(path, f".stats[{name!r}].n_params", "required field missing")
        _as_number(entry["h_trace"], path, f".stats[{name!r}].h_trace")
        _as_non_negative_int(entry["n_params"], path, f".stats[{name!r}].n_params")
        for optional in ("in_features", "out_features", "num_experts"):
            if optional in entry and entry[optional] is not None:
                _as_non_negative_int(
                    entry[optional], path, f".stats[{name!r}].{optional}"
                )
    meta = payload.get("meta", {})
    if meta is not None and not _is_mapping(meta):
        _fail(path, ".meta", "must be a mapping when present")
    _validate_router_number_map(payload, "router_counts", path)
    _validate_router_number_map(payload, "router_active_counts", path, integral_values=True)
    router_totals = payload.get("router_totals")
    if router_totals is not None:
        if not _is_mapping(router_totals):
            _fail(path, ".router_totals", "must be a mapping when present")
        for router, total in router_totals.items():
            if not isinstance(router, str):
                _fail(path, ".router_totals", "router keys must be strings")
            _as_non_negative_int(total, path, f".router_totals[{router!r}]")
    expert_info = payload.get("expert_info", {})
    if expert_info is not None:
        if not _is_mapping(expert_info):
            _fail(path, ".expert_info", "must be a mapping when present")
        for name, pair in expert_info.items():
            if not isinstance(name, str):
                _fail(path, ".expert_info", "expert-info keys must be strings")
            if (not isinstance(pair, Sequence)
                    or isinstance(pair, (str, bytes))
                    or len(pair) != 2):
                _fail(path, f".expert_info[{name!r}]", "must be a 2-item sequence")
            router, eid = pair
            if not isinstance(router, str):
                _fail(path, f".expert_info[{name!r}][0]", "router qname must be a string")
            if not isinstance(eid, (str, Integral)) or isinstance(eid, bool):
                _fail(path, f".expert_info[{name!r}][1]", "expert id must be a string or int")
    return payload


def validate_cost_payload(payload, path: str | None = None):
    """Validate cost structure and finite numeric signals on non-error rows."""
    if not _is_mapping(payload):
        _fail(path, "", "cost payload is not a mapping")
    costs = payload.get("costs")
    if not _is_mapping(costs):
        _fail(path, ".costs", "missing or not a mapping")
    for part in ("provenance", "meta"):
        refuse_retired_format_plan_identity(
            payload.get(part), f"{_label(path)}.{part}")
    formats = payload.get("formats", [])
    if formats is not None:
        if not isinstance(formats, Sequence) or isinstance(formats, (str, bytes)):
            _fail(path, ".formats", "must be a sequence of format names")
        for idx, fmt in enumerate(formats):
            if not isinstance(fmt, str):
                _fail(path, f".formats[{idx}]", "format name must be a string")
            refuse_retired_codebook_format(fmt)
    for name, layer_costs in costs.items():
        if not isinstance(name, str):
            _fail(path, ".costs", "layer keys must be strings")
        if not _is_mapping(layer_costs):
            _fail(path, f".costs[{name!r}]", "entry is not a mapping")
        for fmt, entry in layer_costs.items():
            if not isinstance(fmt, str):
                _fail(path, f".costs[{name!r}]", "format keys must be strings")
            refuse_retired_codebook_format(fmt)
            if not _is_mapping(entry):
                _fail(path, f".costs[{name!r}][{fmt!r}]", "entry is not a mapping")
            if "error" in entry:
                continue
            has_signal = False
            for field in (
                "weight_mse",
                "predicted_dloss",
                "output_mse",
                "fisher_output_mse",
            ):
                if field in entry:
                    _as_finite_cost_number(
                        entry[field], path, f".costs[{name!r}][{fmt!r}].{field}"
                    )
                    has_signal = True
            if "cost_source" in entry and not isinstance(entry["cost_source"], str):
                _fail(
                    path,
                    f".costs[{name!r}][{fmt!r}].cost_source",
                    "must be a string when present",
                )
            refuse_retired_ladder_cost_source(
                entry, f"{_label(path)}.costs[{name!r}][{fmt!r}]")
            if "weight_mse_per_expert" in entry:
                values = entry["weight_mse_per_expert"]
                if (not isinstance(values, Sequence)
                        or isinstance(values, (str, bytes))):
                    _fail(
                        path,
                        f".costs[{name!r}][{fmt!r}].weight_mse_per_expert",
                        "must be a sequence when present",
                    )
                for idx, value in enumerate(values):
                    _as_finite_cost_number(
                        value,
                        path,
                        f".costs[{name!r}][{fmt!r}]"
                        f".weight_mse_per_expert[{idx}]",
                    )
            if "cost_source_per_expert" in entry:
                values = entry["cost_source_per_expert"]
                if (not isinstance(values, Sequence)
                        or isinstance(values, (str, bytes))
                        or not all(isinstance(value, str) for value in values)):
                    _fail(
                        path,
                        f".costs[{name!r}][{fmt!r}].cost_source_per_expert",
                        "must be a sequence of strings when present",
                    )
                mse_values = entry.get("weight_mse_per_expert")
                if (isinstance(mse_values, Sequence)
                        and not isinstance(mse_values, (str, bytes))
                        and len(values) != len(mse_values)):
                    _fail(
                        path,
                        f".costs[{name!r}][{fmt!r}].cost_source_per_expert",
                        "must match weight_mse_per_expert length",
                    )
            for field in _RETIRED_CODEBOOK_COST_FIELDS:
                if field in entry:
                    _fail(
                        path,
                        f".costs[{name!r}][{fmt!r}].{field}",
                        "belongs to the retired Gridbook codebook lane's "
                        "min-chain encoder (archived 2026-09-25, #1304); a "
                        "row carrying it cannot be priced. See "
                        f"{RETIRED_CODEBOOK_ARCHIVE}/README.md.",
                    )
            if ("output_mse_measured" in entry
                    and not isinstance(entry["output_mse_measured"], bool)):
                _fail(
                    path,
                    f".costs[{name!r}][{fmt!r}].output_mse_measured",
                    "must be a boolean when present",
                )
            if not has_signal:
                _fail(
                    path,
                    f".costs[{name!r}][{fmt!r}]",
                    "usable cost entry needs weight_mse, predicted_dloss, or output_mse",
                )
    return payload


#: Exact (family, rate, row class) rows of the #1588 pre-dispatch packet
#: (PQ #2558). Fourteen (rung, unit-class) pairs over 11 unique
#: (family, rate) rates: BF16 R1152, E4M3 R1152 and E2M1 R768 each
#: appear once as routed and once as dense.
PQ1588_PREDISPATCH_ROWS = (
    ("TESSERA_BF16_K1", 960, "routed"),
    ("TESSERA_BF16_K1", 1088, "routed"),
    ("TESSERA_BF16_K1", 1152, "routed"),
    ("TESSERA_BF16_K1", 1152, "dense"),
    ("TESSERA_BF16_K1", 1408, "dense"),
    ("TESSERA_BF16_K1", 1792, "dense"),
    ("TESSERA_E4M3_K1", 768, "routed"),
    ("TESSERA_E4M3_K1", 1152, "routed"),
    ("TESSERA_E4M3_K1", 1152, "dense"),
    ("TESSERA_E4M3_K1", 1536, "dense"),
    ("TESSERA_E4M3_K1", 2048, "dense"),
    ("TESSERA_E2M1_K2", 640, "routed"),
    ("TESSERA_E2M1_K2", 768, "routed"),
    ("TESSERA_E2M1_K2", 768, "dense"),
)

#: Hour-math constants of the #1588 pre-dispatch packet (PQ #2558).
#: The 30-41 GPU-h receipt half is gone under per-shape time; the
#: encode-plus-joint cost half remains at 145-190 GPU-h. Shape-time
#: rows count (7 routed x 1 shape + 7 dense x 4 shapes) x 4 regimes.
PQ1588_PREDISPATCH_ENCODE_LOW = 145
PQ1588_PREDISPATCH_ENCODE_HIGH = 190
PQ1588_PREDISPATCH_JOINT_RECEIPT = 0
PQ1588_PREDISPATCH_SHAPE_TIME_ROWS = 140

_PQ1588_PIN_KEYS = (
    "reader_dev_pin_commit",
    "reader_dev_pin_contract_sha256",
    "serving_runtime_pinned_commit",
    "serving_runtime_pinned_version",
    "serving_runtime_pinned_contract_sha256",
    "producer_installed_contract_sha256",
)
_PQ1588_DIGEST_KEYS = (
    "tessera_export_sha256",
    "tessera_grammar_sha256",
    "installed_contract_sha256",
)
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def validate_pq1588_predispatch_packet(payload, path: str | None = None):
    """Validate the #1588 pre-dispatch packet shape and hour math (PQ #2558).

    This checks the packet's own structure: the exact 14-row set, the
    40-char source SHA bound to the namespace short id, the pin and
    digest key sets, and the hour-math constants. It does not check
    the values against the live repo state; the packet test re-derives
    pins, digests, legal rates and cell state through the repo APIs,
    so a stale value fails there.
    """
    if not _is_mapping(payload):
        _fail(path, "", "predispatch packet is not a mapping")
    if payload.get("schema") != "prismaquant.predispatch_packet.v1":
        _fail(path, ".schema", "must be prismaquant.predispatch_packet.v1")
    source = payload.get("source_commit")
    if not isinstance(source, str) or _GIT_SHA_RE.fullmatch(source) is None:
        _fail(path, ".source_commit", "must be a 40-char lowercase git SHA")
    provenance = payload.get("provenance")
    if not _is_mapping(provenance) or provenance.get("source_commit") != source:
        _fail(path, ".provenance.source_commit", "must equal .source_commit")
    short_id = payload.get("short_id", provenance.get("short_id"))
    if short_id != source[:8]:
        _fail(path, ".short_id", "must be the first 8 chars of .source_commit")
    pins = payload.get("pins")
    if not _is_mapping(pins) or set(pins) != set(_PQ1588_PIN_KEYS):
        _fail(path, ".pins", f"must hold exactly {sorted(_PQ1588_PIN_KEYS)}")
    digests = payload.get("input_digests")
    if not _is_mapping(digests) or set(digests) != set(_PQ1588_DIGEST_KEYS):
        _fail(path, ".input_digests", f"must hold exactly {sorted(_PQ1588_DIGEST_KEYS)}")
    for key in _PQ1588_DIGEST_KEYS:
        if not isinstance(digests[key], str) or _SHA256_RE.fullmatch(digests[key]) is None:
            _fail(path, f".input_digests.{key}", "must be a 64-char hex digest")
    rows = payload.get("requested_rows")
    if not isinstance(rows, list) or len(rows) != 14:
        _fail(path, ".requested_rows", "must list exactly 14 rows")
    seen = []
    for pos, row in enumerate(rows):
        where = f".requested_rows[{pos}]"
        if not _is_mapping(row):
            _fail(path, where, "row is not a mapping")
        for field in ("family", "rate", "row_class"):
            if field not in row:
                _fail(path, f"{where}.{field}", "required field missing")
        if row.get("producer_legal") is not True:
            _fail(path, f"{where}.producer_legal", "must be true")
        if row.get("priced_in_joint_pkl") is not False:
            _fail(path, f"{where}.priced_in_joint_pkl", "must be false")
        if not isinstance(row.get("qualified_structures"), list):
            _fail(path, f"{where}.qualified_structures", "must be a list")
        if not isinstance(row.get("cell_for_requested_class"), bool):
            _fail(path, f"{where}.cell_for_requested_class", "must be a boolean")
        seen.append((row["family"], row["rate"], row["row_class"]))
    if sorted(seen) != sorted(PQ1588_PREDISPATCH_ROWS):
        _fail(path, ".requested_rows", "row set differs from PQ1588_PREDISPATCH_ROWS")
    unique = payload.get("unique_rates")
    if not _is_mapping(unique):
        _fail(path, ".unique_rates", "missing or not a mapping")
    flat = sorted(
        (family, rate) for family, rates in unique.items() for rate in rates
    )
    expect_unique = sorted({(family, rate) for family, rate, _ in PQ1588_PREDISPATCH_ROWS})
    if flat != expect_unique or len(flat) != 11:
        _fail(path, ".unique_rates", "must hold the 11 unique packet rates")
    estimate = payload.get("estimate")
    if not _is_mapping(estimate):
        _fail(path, ".estimate", "missing or not a mapping")
    cost = estimate.get("cost_rows_gpu_h")
    if not _is_mapping(cost):
        _fail(path, ".estimate.cost_rows_gpu_h", "missing or not a mapping")
    if cost.get("encode_low") != PQ1588_PREDISPATCH_ENCODE_LOW:
        _fail(path, ".estimate.cost_rows_gpu_h.encode_low", "must be 145")
    if cost.get("encode_high") != PQ1588_PREDISPATCH_ENCODE_HIGH:
        _fail(path, ".estimate.cost_rows_gpu_h.encode_high", "must be 190")
    if cost.get("joint_receipt") != PQ1588_PREDISPATCH_JOINT_RECEIPT:
        _fail(path, ".estimate.cost_rows_gpu_h.joint_receipt", "must be 0")
    shape_rows = estimate.get("shape_time_rows")
    if not _is_mapping(shape_rows):
        _fail(path, ".estimate.shape_time_rows", "missing or not a mapping")
    routed = sum(1 for _, _, cls in PQ1588_PREDISPATCH_ROWS if cls == "routed")
    dense = sum(1 for _, _, cls in PQ1588_PREDISPATCH_ROWS if cls == "dense")
    if shape_rows.get("count") != (routed * 1 + dense * 4) * 4:
        _fail(path, ".estimate.shape_time_rows.count", "hour math is stale")
    if shape_rows.get("count") != PQ1588_PREDISPATCH_SHAPE_TIME_ROWS:
        _fail(path, ".estimate.shape_time_rows.count", "must be 140")
    namespace = payload.get("namespace")
    if not _is_mapping(namespace):
        _fail(path, ".namespace", "missing or not a mapping")
    if namespace.get("created") is not False:
        _fail(path, ".namespace.created", "must be false: the root stays a plan")
    root = namespace.get("proposed_root")
    if not isinstance(root, str) or source[:8] not in root:
        _fail(path, ".namespace.proposed_root", "must carry the packet short id")
    return payload


def validate_legacy_freeze(payload, path: str | None = None):
    """Validate the legacy anchors freeze file shape (PQ #2559).

    The freeze binds a legacy anchors path to the SHA-256 of its manifest
    bytes. It carries no legacy bytes, only the digest. Digest drift is
    refused by the reader, not here.
    """
    if not _is_mapping(payload):
        _fail(path, "", "legacy freeze is not a mapping")
    if payload.get("schema") != "prismaquant.tessera_legacy_freeze.v1":
        _fail(path, ".schema", "must be prismaquant.tessera_legacy_freeze.v1")
    frozen_path = payload.get("legacy_path")
    if not isinstance(frozen_path, str) or not frozen_path or not frozen_path.startswith("/"):
        _fail(path, ".legacy_path", "must be an absolute path string")
    digest = payload.get("sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        _fail(path, ".sha256", "must be a 64-char hex digest")
    return payload


def validate_layer_config_payload(payload, path: str | None = None):
    """Validate allocator/exporter layer_config JSON shape."""
    if not _is_mapping(payload):
        _fail(path, "", "layer_config is not a JSON object")
    for name, entry in payload.items():
        if not isinstance(name, str):
            _fail(path, "", "layer_config keys must be strings")
        if name == "__prismaquant__":
            # Reserved allocator-metadata block (layer_config.LAYER_CONFIG_META_KEY):
            # travels with the assignment, is not a tensor entry.
            if not _is_mapping(entry):
                _fail(path, f"[{name!r}]", "reserved metadata must be an object")
            continue
        where = f"[{name!r}]"
        if isinstance(entry, dict):
            dt = entry.get("data_type")
            if not isinstance(dt, str):
                _fail(path, f"{where}.data_type", "required string field missing")
            if "bits" in entry:
                _as_non_negative_int(entry["bits"], path, f"{where}.bits")
            if "group_size" in entry and entry["group_size"] is not None:
                _as_non_negative_int(entry["group_size"], path, f"{where}.group_size")
            continue
        if isinstance(entry, str):
            continue
        if isinstance(entry, int) and not isinstance(entry, bool):
            continue
        _fail(path, where, "entry must be a format dict, string, or integer")
    return payload


# -- Strict JSON and the shared refusal vocabulary (PQ #1300) --------------


def unique_json_object(
    duplicate: Callable[[str], BaseException],
) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
    """An ``object_pairs_hook`` that refuses a repeated key.

    ``json`` keeps the last of two equal keys, so one document would have two
    readings. The hook raises ``duplicate(key)``, the caller's own exception.
    """

    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise duplicate(key)
            result[key] = value
        return result

    return object_from_pairs


def strict_json_loads(
    text: str | bytes,
    *,
    duplicate: Callable[[str], BaseException],
    constant: Callable[[str], BaseException] | None = None,
) -> Any:
    """``json.loads`` that refuses a repeated key and, given ``constant``, NaN.

    ``duplicate(key)`` and ``constant(name)`` build the caller's exceptions;
    ``name`` is ``NaN``, ``Infinity`` or ``-Infinity``. Without ``constant``
    those three parse as floats, as ``json.loads`` parses them. ``text`` is
    ``str`` or ``bytes``, as for ``json.loads``. A malformed document raises
    ``json.JSONDecodeError`` (a ``ValueError``) and undecodable bytes raise
    ``UnicodeDecodeError``; the caller wraps both in its own refusal.
    """
    if constant is None:
        return json.loads(text, object_pairs_hook=unique_json_object(duplicate))

    def reject_constant(name: str) -> NoReturn:
        raise constant(name)

    return json.loads(text, object_pairs_hook=unique_json_object(duplicate),
                      parse_constant=reject_constant)


_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9._-]+\Z")


class Contract:
    """One module's refusal vocabulary: the exception it raises and its prefix.

    ``fail`` and ``require`` carry any message. The typed checks speak the
    ``"{where} must be ..."`` vocabulary that the cluster-campaign and
    quality-prefill contracts share word for word, so a module that binds
    them refuses exactly as its own copies did.
    """

    __slots__ = ("error", "prefix")

    def __init__(self, error: type[BaseException] = ValueError, prefix: str = "") -> None:
        self.error = error
        self.prefix = prefix

    def exception(self, message: str) -> BaseException:
        """The refusal ``fail`` raises, for a caller that raises it itself."""
        return self.error(f"{self.prefix}{message}")

    def fail(self, message: str) -> NoReturn:
        raise self.exception(message)

    def require(self, condition: object, message: str) -> None:
        if not condition:
            raise self.exception(message)

    def mapping(self, value: object, *, where: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            self.fail(f"{where} must be an object")
        return value

    def exact_mapping(
        self, value: object, *, keys: frozenset[str], where: str,
    ) -> Mapping[str, object]:
        """An object whose keys are strings and exactly ``keys``."""
        self.mapping(value, where=where)
        if any(type(key) is not str for key in value):
            self.fail(f"{where} keys must be strings")
        actual, expected = set(value), set(keys)
        if actual != expected:
            self.fail(f"{where} fields differ: missing={sorted(expected - actual)}, "
                      f"extra={sorted(actual - expected)}")
        return value

    def integer(
        self, value: object, *, where: str, minimum: int, maximum: int = 2**63 - 1,
    ) -> int:
        """``type(value) is int``: a ``bool`` never satisfies an integer field."""
        if type(value) is not int or not minimum <= value <= maximum:
            self.fail(f"{where} must be an integer in [{minimum}, {maximum}]")
        return value

    def string(
        self, value: object, *, where: str, pattern: re.Pattern[str] | None = None,
    ) -> str:
        """A non-empty string with no padding or control characters."""
        if type(value) is not str or not value:
            self.fail(f"{where} must be a non-empty string")
        if value != value.strip() or any(ord(char) < 32 for char in value):
            self.fail(f"{where} contains whitespace padding or control characters")
        if pattern is not None and pattern.fullmatch(value) is None:
            self.fail(f"{where} has an invalid value")
        return value

    def sha256(self, value: object, *, where: str) -> str:
        """A lowercase hex SHA-256 digest."""
        return self.string(value, where=where, pattern=_SHA256_HEX)

    def absolute_posix_path(self, value: object, *, where: str) -> str:
        """A non-root absolute POSIX path of plain, traversal-free components."""
        raw = self.string(value, where=where)
        if not raw.startswith("/") or raw == "/":
            self.fail(f"{where} must be a non-root absolute POSIX path")
        components = raw.split("/")[1:]
        if (
            not components
            or any(
                not component
                or component in {".", ".."}
                or _PATH_COMPONENT.fullmatch(component) is None
                for component in components
            )
            or str(PurePosixPath(raw)) != raw
        ):
            self.fail(f"{where} must be normalized and traversal-free")
        return raw
