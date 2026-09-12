#!/usr/bin/env python3
"""Make a compact, identity-bound measured-rate snapshot for research.

This is deliberately an offline reader of campaign ``cost.pkl`` snapshots.  It
does not read weights, wires, journals, or the large campaign manifest, and it
does not produce joint-AURA or serving-admission input.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pickle
import re
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np


SCHEMA = "prismaquant.sparse_rate_dataset.v1"
CURRENCY = "output_mse_under_route_activation_contract"
CAMPAIGN_SCHEMA = "prismaquant.tessera_campaign_cost.v1"
DATASET_NAME = "sparse_rate_dataset.npz"
MANIFEST_NAME = "manifest.json"
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


class SparseRateDatasetError(ValueError):
    """A source snapshot is incomplete, ambiguous, or not campaign-measured."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SparseRateDatasetError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> tuple[bytes, Mapping]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SparseRateDatasetError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, Mapping), f"JSON object required: {path}")
    return raw, value


def _canonical(value: object) -> object:
    try:
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SparseRateDatasetError("identity is not canonical JSON") from exc


def _number(value: object, *, positive: bool, field: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             f"{field} must be finite")
    numeric = float(value)
    _require(numeric > 0 if positive else numeric >= 0,
             f"{field} must be {'positive' if positive else 'nonnegative'}")
    return numeric


def _integer(value: object, *, positive: bool, field: str) -> int:
    _require(type(value) is int and (value > 0 if positive else value >= 0),
             f"{field} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _structure(qname: str) -> str:
    _require("." in qname, f"qname has no structure: {qname}")
    return qname.rsplit(".", 1)[1]


def _role(qname: str) -> str:
    return "expert" if ".experts." in qname else "dense"


def _family_restriction_policy(provenance: Mapping) -> object:
    value = provenance.get("family_restriction")
    if value is None:
        return None
    _require(isinstance(value, Mapping), "invalid family-restriction provenance")
    _require("policy" in value, "family-restriction provenance lacks policy")
    return value["policy"]


def _current_source_pins(provenance: Mapping) -> object:
    """Use the payload's terminal reseal pins, never a historical source pin."""
    migration = provenance.get("identity_migration")
    if migration is None:
        return None
    _require(isinstance(migration, list) and migration, "invalid identity-migration provenance")
    terminal = migration[-1]
    _require(isinstance(terminal, Mapping) and isinstance(terminal.get("new_pins"), Mapping),
             "identity migration lacks current source pins")
    return terminal["new_pins"]


def _campaign_identity(payload: Mapping) -> Mapping:
    """The source/calibration identity common to all row payloads.

    Row-local paths and placement details do not identify the scored source, so
    they are deliberately absent.  Any selected field present in one payload
    must be present and identical in every payload.
    """
    provenance = payload.get("provenance")
    _require(isinstance(provenance, Mapping), "payload has no provenance object")
    hessian = provenance.get("hessian")
    _require(hessian is None or isinstance(hessian, Mapping), "invalid Hessian provenance")
    # ``selected_source_preparation``, ``population``, source/cache paths, and
    # identity-migration entries are intentionally row-local.  They cannot be
    # compared across a dense and an expert row.  These values describe the
    # common source and calibration contract that the payload actually records.
    return _canonical({
        "schema": payload.get("schema"), "currency": payload.get("currency"),
        "source": {
            **({field: provenance[field] for field in ("model", "tessera_commit")
                if field in provenance}),
            **({"current_source_pins": _current_source_pins(provenance)}
               if "identity_migration" in provenance else {}),
        },
        "calibration": {
            **({"calibration_cache": provenance["calibration_cache"]}
               if "calibration_cache" in provenance else {}),
            **({"hessian_reference_binding": hessian["reference_binding"]}
               if isinstance(hessian, Mapping) and "reference_binding" in hessian else {}),
        },
        "scoring": {field: provenance[field] for field in (
            "cost_mode", "nsamples", "seqlen", "max_act_rows", "layer_stride",
            "tp_degree", "rate_band", "menu_mode",
        ) if field in provenance},
        "family_restriction_policy": _family_restriction_policy(provenance),
    })


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Write stable ``.npz`` bytes, including a fixed ZIP timestamp."""
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name in sorted(arrays):
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            with archive.open(entry, "w") as handle:
                np.lib.format.write_array(handle, np.asarray(arrays[name]), allow_pickle=False)
    return result.getvalue()


def _atomic_create(path: Path, data: bytes) -> None:
    """Durably publish exactly one new regular file without overwriting it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.is_symlink() and not path.exists(), f"output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SparseRateDatasetError(f"output appeared concurrently: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _existing_is_exact(npz_path: Path, manifest_path: Path, expected_identity: str,
                       expected_npz_sha256: str) -> bool:
    if not npz_path.exists() and not manifest_path.exists():
        return False
    _require(npz_path.is_file() and manifest_path.is_file()
             and not npz_path.is_symlink() and not manifest_path.is_symlink(),
             "existing output is incomplete or unsafe")
    raw, manifest = _read_json(manifest_path)
    del raw
    _require(manifest.get("schema") == SCHEMA, "existing output has unsupported schema")
    _require(manifest.get("snapshot_identity_sha256") == expected_identity,
             "existing output identity differs; refusing overwrite")
    _require(manifest.get("npz_sha256") == expected_npz_sha256,
             "existing output bytes differ; refusing overwrite")
    _require(manifest.get("npz_sha256") == _sha256(npz_path.read_bytes()),
             "existing output hash mismatch")
    return True


def build_snapshot(plan_path: str | Path, census_path: str | Path, out: str | Path) -> Mapping:
    """Extract every measured campaign rate into a compact immutable snapshot."""
    plan_path, census_path, out = Path(plan_path), Path(census_path), Path(out)
    plan_bytes, plan = _read_json(plan_path)
    census_bytes, census = _read_json(census_path)
    _require(plan.get("schema") == "prismaquant.tessera_campaign_plan.v1", "unsupported plan schema")
    declared_census = plan.get("census")
    _require(isinstance(declared_census, str) and Path(declared_census).resolve() == census_path.resolve(),
             "plan census identity does not match --census")
    _require(isinstance(plan.get("rows"), list) and plan["rows"], "plan has no rows")
    _require(isinstance(census.get("counts"), Mapping)
             and isinstance(census.get("unit_shapes"), Mapping), "census lacks counts or shapes")
    _require(isinstance(census.get("dense_targets"), list)
             and isinstance(census.get("expert_targets"), list), "census lacks target identities")
    dense_targets, expert_targets = set(census["dense_targets"]), set(census["expert_targets"])

    input_files = {
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_bytes)},
        "census": {"path": str(census_path), "sha256": _sha256(census_bytes)},
    }
    rows: dict[tuple[str, str], dict] = {}
    observation_count = 0
    ignored_interpolated = 0
    measured_by_rate: dict[int, int] = {}
    global_identity = None
    seen_row_ids = set()

    for plan_row in plan["rows"]:
        _require(isinstance(plan_row, Mapping), "plan row is not an object")
        row_id, row_dir, members = plan_row.get("row_id"), plan_row.get("dir"), plan_row.get("members")
        _require(isinstance(row_id, str) and row_id and row_id not in seen_row_ids,
                 "duplicate or invalid plan row_id")
        seen_row_ids.add(row_id)
        _require(isinstance(row_dir, str) and row_dir, f"{row_id}: missing row directory")
        _require(isinstance(members, list) and all(isinstance(member, str) for member in members),
                 f"{row_id}: invalid members")
        cost_path = Path(row_dir) / "cost.pkl"
        _require(cost_path.is_file() and not cost_path.is_symlink(), f"{row_id}: missing cost.pkl")
        cost_bytes = cost_path.read_bytes()
        try:
            payload = pickle.loads(cost_bytes)
        except Exception as exc:
            raise SparseRateDatasetError(f"{row_id}: invalid cost.pkl") from exc
        _require(isinstance(payload, Mapping), f"{row_id}: payload is not an object")
        _require(payload.get("schema") == CAMPAIGN_SCHEMA and payload.get("currency") == CURRENCY,
                 f"{row_id}: unsupported campaign currency or schema")
        identity = _campaign_identity(payload)
        if global_identity is None:
            global_identity = identity
        else:
            _require(identity == global_identity, f"{row_id}: campaign source/calibration identity mismatch")
        costs = payload.get("costs")
        _require(isinstance(costs, Mapping), f"{row_id}: costs is not an object")
        _require(set(costs).issubset(set(members)), f"{row_id}: payload contains a non-planned unit")
        input_files["rows/" + row_id] = {"path": str(cost_path), "sha256": _sha256(cost_bytes)}

        for qname, cells in sorted(costs.items()):
            _require(isinstance(qname, str) and isinstance(cells, Mapping),
                     f"{row_id}: invalid cost unit")
            _require(qname in census["counts"] and qname in census["unit_shapes"],
                     f"{row_id}: census lacks {qname}")
            shape = census["unit_shapes"][qname]
            _require(isinstance(shape, list) and len(shape) == 2,
                     f"{row_id}: invalid shape for {qname}")
            shape_rows = _integer(shape[0], positive=True, field=f"{qname} shape rows")
            shape_cols = _integer(shape[1], positive=True, field=f"{qname} shape cols")
            count = _integer(census["counts"][qname], positive=False, field=f"{qname} count")
            layer_match = LAYER_RE.search(qname)
            _require(layer_match is not None, f"{row_id}: qname has no layer: {qname}")
            layer = int(layer_match.group(1))
            expected_targets = expert_targets if _role(qname) == "expert" else dense_targets
            _require(qname in expected_targets, f"{row_id}: target identity mismatch for {qname}")
            for format_name, cell in sorted(cells.items()):
                _require(isinstance(format_name, str) and isinstance(cell, Mapping),
                         f"{row_id}: invalid cell for {qname}")
                if cell.get("output_mse_measured") is not True:
                    ignored_interpolated += 1
                    continue
                _require(cell.get("currency") == CURRENCY
                         and cell.get("cost_source") == "tessera_campaign_measured"
                         and cell.get("tessera_provenance") == "measured",
                         f"{row_id}: untrusted measured cell {qname}/{format_name}")
                family, activation = cell.get("tessera_family"), cell.get("activation_contract")
                _require(isinstance(family, str) and family and isinstance(activation, str) and activation,
                         f"{row_id}: missing family or activation contract for {qname}/{format_name}")
                activation_quantized = cell.get("activation_quantized")
                _require(activation_quantized is None or type(activation_quantized) is bool,
                         f"{row_id}: invalid activation-quantized identity for {qname}/{format_name}")
                rate = _integer(cell.get("tessera_body_rate_q256"), positive=True,
                                field=f"{qname}/{format_name} rate")
                mse = _number(cell.get("output_mse"), positive=True,
                              field=f"{qname}/{format_name} output_mse")
                wire_bytes = _integer(cell.get("wire_bytes"), positive=True,
                                      field=f"{qname}/{format_name} wire_bytes")
                encode_seconds = _number(cell.get("encode_seconds"), positive=False,
                                         field=f"{qname}/{format_name} encode_seconds")
                key = (qname, family)
                record = rows.setdefault(key, {
                    "qname": qname, "family": family, "activation_contract": activation,
                    "activation_quantized": activation_quantized,
                    "layer": layer, "role": _role(qname), "structure": _structure(qname),
                    "rows": shape_rows, "cols": shape_cols, "count": count, "rates": {},
                })
                _require(record["activation_contract"] == activation
                         and record["activation_quantized"] == activation_quantized
                         and record["layer"] == layer and record["rows"] == shape_rows
                         and record["cols"] == shape_cols and record["count"] == count,
                         f"{row_id}: activation or shape identity mismatch for {qname}/{family}")
                _require(rate not in record["rates"],
                         f"{row_id}: duplicate measured rate for {qname}/{family}/{rate}")
                record["rates"][rate] = (mse, wire_bytes, encode_seconds)
                observation_count += 1
                measured_by_rate[rate] = measured_by_rate.get(rate, 0) + 1

    _require(rows, "no measured campaign observations")
    ordered = [rows[key] for key in sorted(rows)]
    rates = sorted({rate for row in ordered for rate in row["rates"]})
    values = np.full((len(ordered), len(rates)), np.nan, dtype=np.float64)
    wire_bytes = np.zeros((len(ordered), len(rates)), dtype=np.int64)
    encode_seconds = np.full((len(ordered), len(rates)), np.nan, dtype=np.float64)
    for row_index, row in enumerate(ordered):
        for rate_index, rate in enumerate(rates):
            if rate in row["rates"]:
                (values[row_index, rate_index], wire_bytes[row_index, rate_index],
                 encode_seconds[row_index, rate_index]) = row["rates"][rate]
    arrays = {
        "qnames": np.asarray([row["qname"] for row in ordered]),
        "families": np.asarray([row["family"] for row in ordered]),
        "activation_contracts": np.asarray([row["activation_contract"] for row in ordered]),
        "layers": np.asarray([row["layer"] for row in ordered], dtype=np.int64),
        "roles": np.asarray([row["role"] for row in ordered]),
        "structures": np.asarray([row["structure"] for row in ordered]),
        "rows": np.asarray([row["rows"] for row in ordered], dtype=np.int64),
        "cols": np.asarray([row["cols"] for row in ordered], dtype=np.int64),
        "counts": np.asarray([row["count"] for row in ordered], dtype=np.int64),
        "rates": np.asarray(rates, dtype=np.int64),
        "values": values,
        "wire_bytes": wire_bytes,
        "encode_seconds": encode_seconds,
    }
    npz = _npz_bytes(arrays)
    snapshot_identity = _sha256(json.dumps(_canonical({
        "schema": SCHEMA, "input_files": input_files, "campaign_identity": global_identity,
    }), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    manifest = {
        "schema": SCHEMA,
        "research_only": True,
        "not_joint_aura": True,
        "not_serving_admission": True,
        "currency": CURRENCY,
        "campaign_identity": global_identity,
        "source_identity": global_identity["source"],
        "calibration_identity": global_identity["calibration"],
        "activation_identities": [
            {key: value for key, value in {
                "family": row["family"], "activation_contract": row["activation_contract"],
                "activation_quantized": row["activation_quantized"],
            }.items() if value is not None}
            for row in ordered
        ],
        "input_files": input_files,
        "snapshot_identity_sha256": snapshot_identity,
        "npz_file": DATASET_NAME,
        "npz_sha256": _sha256(npz),
        "coverage": {
            "planned_rows": len(plan["rows"]), "payload_rows": len(seen_row_ids),
            "unit_family_rows": len(ordered), "measured_observations": observation_count,
            "ignored_unmeasured_or_interpolated": ignored_interpolated,
            "rates": rates, "measured_observations_by_rate": {str(k): v for k, v in sorted(measured_by_rate.items())},
        },
        "arrays": {name: {"dtype": str(array.dtype), "shape": list(array.shape)}
                   for name, array in arrays.items()},
    }
    npz_path, manifest_path = out / DATASET_NAME, out / MANIFEST_NAME
    if _existing_is_exact(npz_path, manifest_path, snapshot_identity, manifest["npz_sha256"]):
        return {"status": "verified_existing", "npz": str(npz_path), "manifest": str(manifest_path),
                "npz_sha256": manifest["npz_sha256"]}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False,
                                 allow_nan=False) + "\n").encode("utf-8")
    _atomic_create(npz_path, npz)
    try:
        _atomic_create(manifest_path, manifest_bytes)
    except Exception:
        # The data file is retained as bounded evidence of the interrupted publication.
        raise
    return {"status": "written", "npz": str(npz_path), "manifest": str(manifest_path),
            "npz_sha256": manifest["npz_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="campaign plan.json")
    parser.add_argument("--census", required=True, help="campaign census.json")
    parser.add_argument("--out", required=True, help="new or previously verified output directory")
    arguments = parser.parse_args()
    result = build_snapshot(arguments.plan, arguments.census, arguments.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
