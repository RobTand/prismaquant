"""Runtime schema checks for PrismaQuant file handoffs.

The pipeline passes several pickle/JSON artifacts between long-running
steps.  These validators intentionally check only the structural contract
that downstream code relies on, so older artifacts with extra fields still
load while malformed artifacts fail before optimization or export begins.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real


class SchemaValidationError(ValueError):
    """Raised when a PrismaQuant handoff artifact is structurally invalid."""


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
    saliency = payload.get("expert_saliency", {})
    if saliency is not None:
        if not _is_mapping(saliency):
            _fail(path, ".expert_saliency", "must be a mapping when present")
        for router, values in saliency.items():
            if not isinstance(router, str):
                _fail(path, ".expert_saliency", "router keys must be strings")
            if not _is_mapping(values):
                _fail(path, f".expert_saliency[{router!r}]", "must be a mapping")
            for eid, score in values.items():
                if not isinstance(eid, (str, Integral)) or isinstance(eid, bool):
                    _fail(path, f".expert_saliency[{router!r}]", "expert ids must be strings or ints")
                _as_number(score, path, f".expert_saliency[{router!r}][{eid!r}]")
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
    """Validate the measured quantization-cost pickle contract."""
    if not _is_mapping(payload):
        _fail(path, "", "cost payload is not a mapping")
    costs = payload.get("costs")
    if not _is_mapping(costs):
        _fail(path, ".costs", "missing or not a mapping")
    formats = payload.get("formats", [])
    if formats is not None:
        if not isinstance(formats, Sequence) or isinstance(formats, (str, bytes)):
            _fail(path, ".formats", "must be a sequence of format names")
        for idx, fmt in enumerate(formats):
            if not isinstance(fmt, str):
                _fail(path, f".formats[{idx}]", "format name must be a string")
    for name, layer_costs in costs.items():
        if not isinstance(name, str):
            _fail(path, ".costs", "layer keys must be strings")
        if not _is_mapping(layer_costs):
            _fail(path, f".costs[{name!r}]", "entry is not a mapping")
        for fmt, entry in layer_costs.items():
            if not isinstance(fmt, str):
                _fail(path, f".costs[{name!r}]", "format keys must be strings")
            if not _is_mapping(entry):
                _fail(path, f".costs[{name!r}][{fmt!r}]", "entry is not a mapping")
            if "error" in entry:
                continue
            has_signal = False
            for field in ("weight_mse", "predicted_dloss", "output_mse"):
                if field in entry:
                    _as_number(entry[field], path, f".costs[{name!r}][{fmt!r}].{field}")
                    has_signal = True
            if not has_signal:
                _fail(
                    path,
                    f".costs[{name!r}][{fmt!r}]",
                    "usable cost entry needs weight_mse, predicted_dloss, or output_mse",
                )
    return payload


def validate_layer_config_payload(payload, path: str | None = None):
    """Validate allocator/exporter layer_config JSON shape."""
    if not _is_mapping(payload):
        _fail(path, "", "layer_config is not a JSON object")
    for name, entry in payload.items():
        if not isinstance(name, str):
            _fail(path, "", "layer_config keys must be strings")
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


def validate_prune_manifest_payload(payload, path: str | None = None):
    """Validate the router-keyed expert prune manifest."""
    if not _is_mapping(payload):
        _fail(path, "", "prune manifest is not a JSON object")
    for router, entry in payload.items():
        if not isinstance(router, str):
            _fail(path, "", "router keys must be strings")
        if not _is_mapping(entry):
            _fail(path, f"[{router!r}]", "entry is not a mapping")
        for field in (
            "num_experts_orig",
            "num_experts_kept",
            "pruned_expert_ids",
            "kept_expert_ids",
            "orig_to_new_eid",
        ):
            if field not in entry:
                _fail(path, f"[{router!r}].{field}", "required field missing")
        num_orig = _as_non_negative_int(
            entry["num_experts_orig"], path, f"[{router!r}].num_experts_orig"
        )
        num_kept = _as_non_negative_int(
            entry["num_experts_kept"], path, f"[{router!r}].num_experts_kept"
        )
        pruned = _int_list(entry["pruned_expert_ids"], path, f"[{router!r}].pruned_expert_ids")
        kept = _int_list(entry["kept_expert_ids"], path, f"[{router!r}].kept_expert_ids")
        if len(set(pruned)) != len(pruned):
            _fail(path, f"[{router!r}].pruned_expert_ids", "contains duplicates")
        if len(set(kept)) != len(kept):
            _fail(path, f"[{router!r}].kept_expert_ids", "contains duplicates")
        if set(pruned) & set(kept):
            _fail(path, f"[{router!r}]", "pruned and kept expert ids overlap")
        if len(kept) != num_kept:
            _fail(path, f"[{router!r}].kept_expert_ids", "length does not match num_experts_kept")
        if len(kept) + len(pruned) != num_orig:
            _fail(path, f"[{router!r}]", "kept + pruned count does not match num_experts_orig")
        for eid in kept + pruned:
            if eid < 0 or eid >= num_orig:
                _fail(path, f"[{router!r}]", f"expert id {eid} is outside [0, {num_orig})")
        mapping = entry["orig_to_new_eid"]
        if not _is_mapping(mapping):
            _fail(path, f"[{router!r}].orig_to_new_eid", "must be a mapping")
        dense_values: list[int] = []
        seen_orig: set[int] = set()
        for orig, new in mapping.items():
            orig_int = _parse_eid(orig, path, f"[{router!r}].orig_to_new_eid key")
            new_int = _as_non_negative_int(
                new, path, f"[{router!r}].orig_to_new_eid[{orig!r}]"
            )
            seen_orig.add(orig_int)
            dense_values.append(new_int)
        if seen_orig != set(kept):
            _fail(path, f"[{router!r}].orig_to_new_eid", "keys must exactly match kept_expert_ids")
        if sorted(dense_values) != list(range(num_kept)):
            _fail(path, f"[{router!r}].orig_to_new_eid", "values must be dense 0..num_experts_kept-1")
    return payload


def _int_list(value, path: str | None, where: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(path, where, "expected a list of integers")
    out = [_as_non_negative_int(v, path, f"{where}[]") for v in value]
    return out


def _parse_eid(value, path: str | None, where: str) -> int:
    if isinstance(value, bool):
        _fail(path, where, "expected an expert id")
    try:
        out = int(value)
    except (TypeError, ValueError):
        _fail(path, where, "expected an expert id")
    if out < 0:
        _fail(path, where, "expected a non-negative expert id")
    return out
