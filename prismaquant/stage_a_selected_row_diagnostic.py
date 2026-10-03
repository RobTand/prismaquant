"""Explicit geometry for one fresh Stage A row/probe (PQ #2010/#2008).

The full calibration tensor remains the input and identity. Only execution is
narrowed; Fisher noise keeps its global row and full-draw normalization. This
module owns no model, source, cache, autograd graph or qualification override.
Original CUDA material remains refused by the existing source owner.
"""
from __future__ import annotations

from pathlib import Path

from .cost_stage_checkpoint import publish_new_bytes
from .dev_mode import dev_mode_enabled
from .digests import bytes_sha256hex, indent2_json_file_bytes
from .schemas import Contract, strict_json_loads
from .stage_inputs import read_bound

SPEC_SCHEMA = "prismaquant.stage_a.selected_row_diagnostic.v1"
RECEIPT_SCHEMA = "prismaquant.stage_a.selected_row_diagnostic_receipt.v1"
MARKER_NAME = "selected-row-diagnostic.json"
RECEIPT_NAME = "selected-row-diagnostic-receipt.json"
_FIELDS = frozenset({"schema", "calibration_shape", "calibration_dtype",
                     "calibration_tensor_sha256", "selected_global_row", "probe_seed",
                     "global_token_count", "vocab_size", "through"})


class SelectedRowDiagnosticRefused(ValueError):
    """The diagnostic's explicit coordinates or full draw do not agree."""


_contract = Contract(SelectedRowDiagnosticRefused, "selected-row diagnostic: ")


def normalize_diagnostic_spec(document) -> dict:
    """Closed, model-independent geometry; the enclosing plan binds authority."""
    _contract.require(dev_mode_enabled(), "selected-row diagnostic requires dev mode")
    _contract.exact_mapping(document, keys=_FIELDS, where="spec")
    _contract.require(document["schema"] == SPEC_SCHEMA, "diagnostic schema mismatch")
    shape = document["calibration_shape"]
    _contract.require(type(shape) is list and len(shape) == 2,
                      "calibration_shape must be [rows, sequence_length]")
    rows, tokens = [_contract.integer(v, where="calibration_shape", minimum=1) for v in shape]
    _contract.string(document["calibration_dtype"], where="calibration_dtype")
    _contract.sha256(document["calibration_tensor_sha256"], where="calibration_tensor_sha256")
    _contract.integer(document["selected_global_row"], where="selected_global_row",
                      minimum=0, maximum=rows - 1)
    for field in ("probe_seed", "through"):
        _contract.integer(document[field], where=field, minimum=0)
    _contract.integer(document["vocab_size"], where="vocab_size", minimum=1)
    _contract.integer(document["global_token_count"], where="global_token_count", minimum=1)
    _contract.require(document["global_token_count"] == rows * tokens,
                      "global_token_count must equal the full all-logit draw")
    return {**document, "calibration_shape": [rows, tokens]}


def load_diagnostic_spec(path, expected_sha256) -> dict:
    """Decode the same bound bytes that were authenticated; reject ambiguity."""
    _contract.sha256(expected_sha256, where="spec sha256")
    try:
        raw = read_bound({"path": str(path), "sha256": expected_sha256}, "selected-row diagnostic")
        return normalize_diagnostic_spec(strict_json_loads(
            raw, duplicate=lambda key: SelectedRowDiagnosticRefused(f"diagnostic duplicate key {key}"),
            constant=lambda key: SelectedRowDiagnosticRefused(f"diagnostic nonfinite constant {key}")))
    except (OSError, ValueError) as exc:
        raise SelectedRowDiagnosticRefused(f"selected-row diagnostic spec refused: {exc}") from exc


def bind_diagnostic_draw(spec, calib_ids, *, execution, num_layers) -> dict:
    """Validate before output creation; never treat the selected row as a draw."""
    from .stage_a_chain_seed import tensor_payload_sha256

    spec = normalize_diagnostic_spec(spec)
    _contract.require(list(calib_ids.shape) == spec["calibration_shape"],
                      "diagnostic full calibration shape differs")
    _contract.require(str(calib_ids.dtype) == spec["calibration_dtype"],
                      "diagnostic calibration dtype differs")
    _contract.require(tensor_payload_sha256(calib_ids) == spec["calibration_tensor_sha256"],
                      "diagnostic full calibration tensor differs")
    _contract.require(type(execution["n_probes"]) is int and execution["n_probes"] == 1,
                      "diagnostic needs exactly one probe")
    _contract.require(type(execution["seed_base"]) is int
                      and execution["seed_base"] == spec["probe_seed"], "diagnostic seed differs")
    _contract.require(type(execution.get("probe_microbatch")) is int
                      and execution["probe_microbatch"] == 1,
                      "diagnostic requires explicit probe_microbatch=1")
    _contract.require(execution.get("token_scope", "all") == "all"
                      and execution.get("temperature", 1.0) == 1.0,
                      "diagnostic requires all logits at temperature 1")
    _contract.require(spec["through"] < num_layers, "diagnostic through is outside the chain")
    return spec


def require_diagnostic_original_owner(owner, *, device, model=None) -> None:
    """Use the existing authority and device gate; never fabricate a provider."""
    from .tessera_calibration_cache import CaptureSourceAuthentication

    _contract.require(isinstance(owner, CaptureSourceAuthentication)
                      and owner.is_qualified_original_material,
                      "diagnostic requires the existing qualified original material owner")
    if model is not None:
        _contract.require(Path(model).resolve() == owner.root.resolve(),
                          "diagnostic model differs from the original owner's root")
    owner.require_material_device(device)


def diagnostic_marker_path(space) -> Path:
    return Path(space) / MARKER_NAME


def write_diagnostic_marker(space, spec) -> None:
    _contract.require(publish_new_bytes(diagnostic_marker_path(space), indent2_json_file_bytes(spec)),
                      "diagnostic root already holds a marker")


def write_diagnostic_receipt(space, receipt) -> dict:
    _contract.require(receipt.get("schema") == RECEIPT_SCHEMA
                      and diagnostic_marker_path(space).is_file(), "diagnostic receipt needs its marker")
    path = Path(space) / RECEIPT_NAME
    raw = indent2_json_file_bytes(receipt)
    _contract.require(publish_new_bytes(path, raw), "diagnostic receipt already exists")
    return {"path": str(path), "sha256": bytes_sha256hex(raw)}
