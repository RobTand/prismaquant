# SPDX-License-Identifier: LicenseRef-PrismaQuant-Attribution-1.0
"""Canonical artifact telemetry for gated fixed-codebook LDLQ exports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any


SIDECAR_SCHEMA = "prismaquant.cb_ldlq_gate_telemetry.v1"
REFERENCE_SCHEMA = "prismaquant.cb_ldlq_gate_telemetry_ref.v1"
KERNEL_STAMP_SCHEMA = "prismaquant.cb_ldlq_kernel_stamp.v1"
SIDECAR_FILENAME = "cb_ldlq_gate_telemetry.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class LDLQTelemetryError(ValueError):
    """Telemetry is incomplete, contradictory, or not canonical JSON."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise LDLQTelemetryError(
            f"LDLQ telemetry is not finite canonical JSON: {exc}"
        ) from exc
    return text.encode("utf-8") + b"\n"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _qnames_sha256(qnames: Iterable[str]) -> str:
    return _sha256(
        "".join(f"{name}\n" for name in sorted(qnames)).encode("utf-8")
    )


def validate_kernel_stamp(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LDLQTelemetryError("kernel stamp must be an object")
    stamp = dict(value)
    if stamp.get("schema") != KERNEL_STAMP_SCHEMA:
        raise LDLQTelemetryError(
            f"unsupported LDLQ kernel stamp schema {stamp.get('schema')!r}"
        )
    required = (
        "abi",
        "implementation_sha256",
        "objective",
        "candidate_solver",
        "tie_break",
        "outer_tile_columns",
        "atom_size_by_grid",
        "packed_expert_kernel",
        "execution_environment",
    )
    missing = [key for key in required if key not in stamp]
    if missing:
        raise LDLQTelemetryError(f"kernel stamp missing {missing}")
    if not isinstance(stamp["abi"], str) or not stamp["abi"].strip():
        raise LDLQTelemetryError("kernel stamp abi must be non-empty")
    if _SHA256_RE.fullmatch(str(stamp["implementation_sha256"])) is None:
        raise LDLQTelemetryError("invalid implementation_sha256")
    if (
        isinstance(stamp["outer_tile_columns"], bool)
        or int(stamp["outer_tile_columns"]) <= 0
    ):
        raise LDLQTelemetryError("outer_tile_columns must be positive")
    atoms = stamp["atom_size_by_grid"]
    if not isinstance(atoms, Mapping) or {
        str(key): int(item) for key, item in atoms.items()
    } != {"fp4": 4, "fp8": 2}:
        raise LDLQTelemetryError("kernel stamp must attest fp4=4 and fp8=2")
    packed = stamp["packed_expert_kernel"]
    if not isinstance(packed, Mapping):
        raise LDLQTelemetryError("packed_expert_kernel must be an object")
    if str(packed.get("route")) != "e16_batched_v1":
        raise LDLQTelemetryError("packed LDLQ route must be e16_batched_v1")
    if int(packed.get("batch_size", 0)) != 16:
        raise LDLQTelemetryError("packed LDLQ batch size must be 16")
    if (
        isinstance(packed.get("streams"), bool)
        or int(packed.get("streams", 0)) <= 0
    ):
        raise LDLQTelemetryError("packed LDLQ streams must be positive")
    if packed.get("nondivisible_experts") != "refuse":
        raise LDLQTelemetryError("non-divisible packed experts must be refused")
    execution = stamp["execution_environment"]
    if not isinstance(execution, Mapping):
        raise LDLQTelemetryError("execution_environment must be an object")
    for key in (
        "torch_version",
        "cuda_version",
        "gpu_arch",
        "gpu_name",
        "producer_image_digest",
    ):
        if key not in execution or not str(execution[key]).strip():
            raise LDLQTelemetryError(
                f"execution_environment missing non-empty {key!r}"
            )
    return json.loads(_canonical_json_bytes(stamp))


def _optional_float(value: object, *, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise LDLQTelemetryError(f"{where} must be numeric or null")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LDLQTelemetryError(f"{where} must be numeric or null") from exc
    return result if math.isfinite(result) else None


def _float_vector(
    gate_info: Mapping[str, Any],
    *,
    plural: str,
    singular: str,
    count: int,
) -> list[float | None]:
    raw = gate_info.get(plural)
    if raw is None and singular in gate_info:
        raw = [gate_info.get(singular)]
    if raw is None:
        return [None] * count
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise LDLQTelemetryError(f"{plural} must be a sequence")
    if len(raw) != count:
        raise LDLQTelemetryError(
            f"{plural} has {len(raw)} values for {count} expert units"
        )
    return [
        _optional_float(item, where=f"{plural}[{index}]")
        for index, item in enumerate(raw)
    ]


def _index_set(
    gate_info: Mapping[str, Any], key: str, count: int
) -> list[int]:
    raw = gate_info.get(key, ())
    if raw is None:
        raw = ()
    if (
        not isinstance(raw, Iterable)
        or isinstance(raw, (str, bytes, Mapping))
    ):
        raise LDLQTelemetryError(f"{key} must be an integer sequence")
    result: set[int] = set()
    for item in raw:
        if isinstance(item, bool):
            raise LDLQTelemetryError(f"{key} contains a bool")
        index = int(item)
        if index < 0 or index >= count:
            raise LDLQTelemetryError(
                f"{key} contains out-of-range expert {index}/{count}"
            )
        result.add(index)
    return sorted(result)


def _kept_vector(gate_info: Mapping[str, Any], count: int) -> list[bool]:
    raw = gate_info.get("per_expert_kept")
    if raw is None:
        raw = gate_info.get("kept_ldlq", False)
    if isinstance(raw, bool):
        return [raw] * count
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != count
        or any(not isinstance(item, bool) for item in raw)
    ):
        raise LDLQTelemetryError(
            f"per-expert keep mask must contain {count} bool values"
        )
    return list(raw)


def normalize_gate_record(
    *,
    qname: str,
    shape: Sequence[int],
    grid: str,
    mode: str,
    k: int,
    gate_info: Mapping[str, Any],
) -> dict[str, Any]:
    name = str(qname).strip()
    dims = [int(dim) for dim in shape]
    if not name:
        raise LDLQTelemetryError("qname must be non-empty")
    if len(dims) not in (2, 3) or any(dim <= 0 for dim in dims):
        raise LDLQTelemetryError(f"{name}: invalid LDLQ shape {dims}")
    count = dims[0] if len(dims) == 3 else 1
    if not isinstance(gate_info, Mapping) or not str(
        gate_info.get("gate", "")
    ).strip():
        raise LDLQTelemetryError(f"{name}: missing gate outcome")

    kept = _kept_vector(gate_info, count)
    missing = _index_set(gate_info, "missing_experts", count)
    uncertifiable = _index_set(
        gate_info, "uncertifiable_experts", count
    )
    hessian_failed = _index_set(
        gate_info, "hessian_failed_experts", count
    )
    raw_mse = _float_vector(
        gate_info,
        plural="raw_mse_per_expert",
        singular="raw_mse",
        count=count,
    )
    ldlq_mse = _float_vector(
        gate_info,
        plural="ldlq_mse_per_expert",
        singular="ldlq_mse",
        count=count,
    )
    ratio = _float_vector(
        gate_info,
        plural="holdout_ratio_per_expert",
        singular="holdout_ratio",
        count=count,
    )
    excluded = set(missing) | set(uncertifiable) | set(hessian_failed)
    null_mse = {
        index
        for index, pair in enumerate(zip(raw_mse, ldlq_mse))
        if pair[0] is None or pair[1] is None
    }
    measured_any = any(item is not None for item in raw_mse + ldlq_mse)
    if measured_any and not null_mse.issubset(excluded):
        raise LDLQTelemetryError(
            f"{name}: unexplained null MSE for {sorted(null_mse - excluded)}"
        )
    for index, (raw, refined, observed) in enumerate(
        zip(raw_mse, ldlq_mse, ratio)
    ):
        if index in excluded:
            if observed is not None:
                raise LDLQTelemetryError(
                    f"{name}: excluded expert {index} must not claim a ratio"
                )
            continue
        expected = (
            refined / raw
            if raw not in (None, 0.0) and refined is not None
            else None
        )
        if expected is None:
            if observed is not None:
                raise LDLQTelemetryError(
                    f"{name}: ratio[{index}] lacks finite nonzero raw MSE"
                )
        elif observed is None or not math.isclose(
            observed, expected, rel_tol=1e-12, abs_tol=0.0
        ):
            raise LDLQTelemetryError(
                f"{name}: ratio[{index}] does not equal LDLQ/raw MSE"
            )
    return {
        "qname": name,
        "shape": dims,
        "grid": str(grid),
        "mode": str(mode),
        "k": int(k),
        "expert_count": count,
        "gate": str(gate_info["gate"]),
        "metric": (
            str(gate_info["metric"])
            if gate_info.get("metric") is not None
            else None
        ),
        "gate_mode": (
            str(gate_info["gate_mode"])
            if gate_info.get("gate_mode") is not None
            else None
        ),
        "per_expert": {
            "kept_ldlq": kept,
            "missing_activation": missing,
            "uncertifiable": uncertifiable,
            "hessian_failed": hessian_failed,
            "raw_mse": raw_mse,
            "ldlq_mse": ldlq_mse,
            "ldlq_over_raw_mse": ratio,
        },
        **(
            {"reason": str(gate_info["reason"])}
            if gate_info.get("reason") is not None
            else {}
        ),
    }


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result = {
        "tensor_records": len(records),
        "expert_units": 0,
        "kept_ldlq": 0,
        "kept_raw": 0,
        "missing_activation": 0,
        "uncertifiable": 0,
        "hessian_failed": 0,
    }
    for record in records:
        count = int(record["expert_count"])
        per = record["per_expert"]
        kept = sum(bool(item) for item in per["kept_ldlq"])
        result["expert_units"] += count
        result["kept_ldlq"] += kept
        result["kept_raw"] += count - kept
        for key in (
            "missing_activation",
            "uncertifiable",
            "hessian_failed",
        ):
            result[key] += len(per[key])
    return result


class LDLQGateTelemetryCollector:
    """Thread-safe exact-qname collector for selected encode results."""

    def __init__(
        self,
        *,
        expected_qnames: Iterable[str],
        kernel_stamp: Mapping[str, Any],
    ) -> None:
        self.expected_qnames = frozenset(
            str(name) for name in expected_qnames
        )
        if any(not name for name in self.expected_qnames):
            raise LDLQTelemetryError("expected qnames must be non-empty")
        self.kernel_stamp = validate_kernel_stamp(kernel_stamp)
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def record(
        self,
        *,
        qname: str,
        shape: Sequence[int],
        grid: str,
        mode: str,
        k: int,
        gate_info: Mapping[str, Any],
    ) -> None:
        observed = gate_info.get("kernel_stamp")
        if observed is None:
            raise LDLQTelemetryError(
                f"{qname}: selected pack has no executed kernel stamp"
            )
        if validate_kernel_stamp(observed) != self.kernel_stamp:
            raise LDLQTelemetryError(
                f"{qname}: executed kernel differs from artifact kernel"
            )
        record = normalize_gate_record(
            qname=qname,
            shape=shape,
            grid=grid,
            mode=mode,
            k=k,
            gate_info=gate_info,
        )
        name = record["qname"]
        if name not in self.expected_qnames:
            raise LDLQTelemetryError(f"unexpected telemetry qname {name!r}")
        with self._lock:
            previous = self._records.get(name)
            if previous is not None and previous != record:
                raise LDLQTelemetryError(
                    f"conflicting telemetry for qname {name!r}"
                )
            self._records[name] = record

    def payload(self) -> dict[str, Any]:
        with self._lock:
            observed = set(self._records)
            missing = sorted(self.expected_qnames - observed)
            extra = sorted(observed - self.expected_qnames)
            if missing or extra:
                raise LDLQTelemetryError(
                    f"telemetry coverage mismatch: missing={missing[:8]}, "
                    f"extra={extra[:8]}"
                )
            records = [
                self._records[name] for name in sorted(self._records)
            ]
        return {
            "schema": SIDECAR_SCHEMA,
            "kernel_stamp": self.kernel_stamp,
            "records": records,
            "summary": _summary(records),
        }

    def publish(
        self,
        out_dir: str | Path,
        quant_config: dict[str, Any],
        *,
        filename: str = SIDECAR_FILENAME,
    ) -> dict[str, Any] | None:
        if not self.expected_qnames:
            return None
        provenance = quant_config.setdefault("provenance", {})
        if not isinstance(provenance, dict):
            raise LDLQTelemetryError("quant_config provenance must be an object")
        if "ldlq_gate_telemetry" in provenance:
            raise LDLQTelemetryError("telemetry provenance already exists")
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise LDLQTelemetryError("unsafe telemetry filename")
        payload = self.payload()
        encoded = _canonical_json_bytes(payload)
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        final_path = root / filename
        pending_path = root / f".{filename}.pending"
        if os.path.lexists(final_path) or os.path.lexists(pending_path):
            raise LDLQTelemetryError(
                f"refusing to overwrite {final_path} or its pending sibling"
            )
        fd = os.open(
            pending_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
        owns_pending = True
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(pending_path, final_path)
            owns_pending = False
        finally:
            if owns_pending and os.path.lexists(pending_path):
                os.unlink(pending_path)
        kernel_bytes = _canonical_json_bytes(payload["kernel_stamp"])
        reference = {
            "schema": REFERENCE_SCHEMA,
            "sidecar_schema": SIDECAR_SCHEMA,
            "file": filename,
            "sha256": _sha256(encoded),
            "kernel_stamp_sha256": _sha256(kernel_bytes),
            "record_count": len(payload["records"]),
            "qnames_sha256": _qnames_sha256(self.expected_qnames),
            "summary": payload["summary"],
        }
        provenance["ldlq_gate_telemetry"] = reference
        return reference


def verify_sidecar_reference(
    out_dir: str | Path,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(reference, Mapping):
        raise LDLQTelemetryError("telemetry reference must be an object")
    if reference.get("schema") != REFERENCE_SCHEMA:
        raise LDLQTelemetryError("unsupported telemetry reference schema")
    filename = str(reference.get("file", ""))
    if Path(filename).name != filename or not filename:
        raise LDLQTelemetryError("unsafe telemetry reference filename")
    encoded = (Path(out_dir) / filename).read_bytes()
    if _sha256(encoded) != reference.get("sha256"):
        raise LDLQTelemetryError("telemetry sidecar sha256 mismatch")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise LDLQTelemetryError("telemetry sidecar is not JSON") from exc
    if _canonical_json_bytes(payload) != encoded:
        raise LDLQTelemetryError("telemetry sidecar is not canonical JSON")
    if payload.get("schema") != SIDECAR_SCHEMA:
        raise LDLQTelemetryError("unsupported telemetry sidecar schema")
    stamp = validate_kernel_stamp(payload.get("kernel_stamp"))
    if _sha256(_canonical_json_bytes(stamp)) != reference.get(
        "kernel_stamp_sha256"
    ):
        raise LDLQTelemetryError("telemetry kernel digest mismatch")
    records = payload.get("records")
    if not isinstance(records, list):
        raise LDLQTelemetryError("telemetry records must be a list")
    names = [
        record.get("qname")
        for record in records
        if isinstance(record, Mapping)
    ]
    if len(names) != len(records) or names != sorted(set(names)):
        raise LDLQTelemetryError("telemetry qnames must be unique and sorted")
    if len(records) != int(reference.get("record_count", -1)):
        raise LDLQTelemetryError("telemetry record count mismatch")
    if _qnames_sha256(names) != reference.get("qnames_sha256"):
        raise LDLQTelemetryError("telemetry qname digest mismatch")
    if (
        _summary(records) != payload.get("summary")
        or payload.get("summary") != reference.get("summary")
    ):
        raise LDLQTelemetryError("telemetry summary mismatch")
    return payload


__all__ = [
    "KERNEL_STAMP_SCHEMA",
    "LDLQGateTelemetryCollector",
    "LDLQTelemetryError",
    "REFERENCE_SCHEMA",
    "SIDECAR_FILENAME",
    "SIDECAR_SCHEMA",
    "normalize_gate_record",
    "validate_kernel_stamp",
    "verify_sidecar_reference",
]
