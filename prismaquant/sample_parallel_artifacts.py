"""Read-only admission contracts for recorded sample-parallel probe artifacts.

The execution lane is archived. Schema identifiers, digest bytes, token/shape
checks and activation priorities remain live because incremental resume reads
recorded artifacts. This module cannot prepare, probe, publish or merge workers.
"""
from __future__ import annotations
import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
import numpy as np
import torch
from .digests import DIRECT_UTF8_STRICT, bytes_sha256hex
from .probe_statistics import merge_marginals, finalize_fisher_stats

SAMPLE_PARALLEL_META_FIELDS = (
    "sample_parallel",
    "sample_parallel_importance",
    "sample_parallel_activation_scope",
    "sample_parallel_execution_identity",
    "sample_parallel_qname_census",
)

import hashlib

import re

import struct

import torch

from prismaquant.perturbed_x_cache import fused_subsample_group

ACTIVATION_CACHE_SHARD_SCHEMA = (
    "prismaquant.sample_parallel_probe.activation_cache_shard.v1"
)

ACTIVATION_PRIORITY_SCHEMA = (
    "blake2b64-keyed-fmix32x2-prp-global-row-fused-group-v2"
)

ROW_INDICES_SCOPE = "shard_local_flat_tokens"

ACTIVATION_PRIORITY_KEY_DOMAIN = (
    b"prismaquant.sample-parallel.activation-priority.v2\0"
)

ACTIVATION_PRIORITY_MAX_ROWS = 1 << 32

_U16_MASK = (1 << 16) - 1

_U32_MASK = (1 << 32) - 1

_FMIX32_C1 = 0x85EBCA6B

_FMIX32_C2 = 0xC2B2AE35

_CALIBRATION_HASH_RE = re.compile(r"[0-9a-f]{32}")

def activation_priority_group(qname: str) -> str:
    """Resolve the existing joint-consumer row-alignment group."""
    return fused_subsample_group(str(qname))

def validate_activation_priority_domain(
    global_samples: int,
    seqlen: int,
) -> int:
    """Return the global dense row count or refuse the uint32 PRP domain."""
    samples = int(global_samples)
    tokens = int(seqlen)
    total = samples * tokens
    if (
        samples < 1
        or tokens < 1
        or total < 1
        or total > ACTIVATION_PRIORITY_MAX_ROWS
    ):
        raise ValueError(
            "sample-parallel activation priority domain requires "
            "0 < global_samples * seqlen <= 2**32"
        )
    return total

def activation_priority_key(
    global_calibration_hash: str,
    qname_or_group: str,
    *,
    is_group: bool = False,
) -> tuple[int, int]:
    """Derive the exact little-endian uint32 key pair for one fused group."""
    calibration_hash = str(global_calibration_hash)
    if _CALIBRATION_HASH_RE.fullmatch(calibration_hash) is None:
        raise ValueError("global calibration hash must be 32 lowercase hex digits")
    group = (
        str(qname_or_group)
        if is_group
        else activation_priority_group(str(qname_or_group))
    )
    encoded_group = group.encode("utf-8")
    if len(encoded_group) > _U32_MASK:
        raise ValueError("activation priority group name is too long")
    material = (
        ACTIVATION_PRIORITY_KEY_DOMAIN
        + bytes.fromhex(calibration_hash)
        + struct.pack("<I", len(encoded_group))
        + encoded_group
    )
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return struct.unpack("<II", digest)

def _mullo32_scalar(value: int, constant: int) -> int:
    """Low uint32 product without a signed-overflowing intermediate."""
    x = int(value) & _U32_MASK
    c = int(constant) & _U32_MASK
    low = x & _U16_MASK
    high = x >> 16
    return (
        low * c + ((high * (c & _U16_MASK)) << 16)
    ) & _U32_MASK

def _fmix32_scalar(value: int) -> int:
    x = int(value) & _U32_MASK
    x ^= x >> 16
    x = _mullo32_scalar(x, _FMIX32_C1)
    x ^= x >> 13
    x = _mullo32_scalar(x, _FMIX32_C2)
    x ^= x >> 16
    return x & _U32_MASK

def activation_row_priority_scalar(
    global_calibration_hash: str,
    qname: str,
    global_row: int,
) -> int:
    """Normative arbitrary-precision reference for one global row."""
    row = int(global_row)
    if row < 0 or row >= ACTIVATION_PRIORITY_MAX_ROWS:
        raise ValueError("global activation row is outside the uint32 domain")
    k0, k1 = activation_priority_key(global_calibration_hash, qname)
    return _fmix32_scalar(_fmix32_scalar(row ^ k0) ^ k1)

def _mullo32_tensor(values: torch.Tensor, constant: int) -> torch.Tensor:
    """Vectorized low uint32 product with all int64 intermediates < 2**49."""
    low = torch.bitwise_and(values, _U16_MASK)
    high = torch.bitwise_right_shift(values, 16)
    return torch.bitwise_and(
        low * int(constant)
        + torch.bitwise_left_shift(
            high * int(constant & _U16_MASK), 16,
        ),
        _U32_MASK,
    )

def _fmix32_tensor(values: torch.Tensor) -> torch.Tensor:
    x = torch.bitwise_and(values, _U32_MASK)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    x = _mullo32_tensor(x, _FMIX32_C1)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 13))
    x = _mullo32_tensor(x, _FMIX32_C2)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    return torch.bitwise_and(x, _U32_MASK)

def activation_row_priorities(
    global_calibration_hash: str,
    qname: str,
    global_rows: torch.Tensor,
) -> torch.Tensor:
    """Return device-preserving, globally comparable uint32 priorities.

    ``global_rows`` must already be within the validated global uint32 domain.
    Producer callers construct those rows from a closed partition contract;
    the independent merger checks every persisted row and priority on CPU.
    """
    if not isinstance(global_rows, torch.Tensor) or global_rows.ndim != 1:
        raise ValueError("global activation rows must be a one-dimensional tensor")
    rows = global_rows.detach().to(dtype=torch.long)
    k0, k1 = activation_priority_key(global_calibration_hash, qname)
    first = _fmix32_tensor(torch.bitwise_xor(rows, k0))
    return _fmix32_tensor(torch.bitwise_xor(first, k1))

__all__ = [
    "ACTIVATION_CACHE_SHARD_SCHEMA",
    "ACTIVATION_PRIORITY_KEY_DOMAIN",
    "ACTIVATION_PRIORITY_MAX_ROWS",
    "ACTIVATION_PRIORITY_SCHEMA",
    "ROW_INDICES_SCOPE",
    "activation_priority_group",
    "activation_priority_key",
    "activation_row_priorities",
    "activation_row_priority_scalar",
    "validate_activation_priority_domain",
]

SOURCE_QNAME_CENSUS_SCHEMA = (
    "prismaquant.sample_parallel_probe.rtx4090_source_qname_census.v2"
)


SOURCE_MODEL_CONTENT_SCHEMA = (
    "prismaquant.sample_parallel_probe.model_content.v2"
)


QNAME_MANIFEST_SCHEMA = (
    "prismaquant.sample_parallel_probe.qname_manifest.v1"
)


BODY_STATS_AND_ACTIVATION = "dense_body_stats_and_activation"


LM_HEAD_STATS_ONLY = "terminal_bf16_lm_head_stats_only"


MTP_STATS_ONLY = "terminal_bf16_mtp_stats_only"


TEXT_EXCLUDED = "excluded_text_only_source_linear"


BODY_TOKEN_ROWS = "seqlen"


LM_HEAD_TOKEN_ROWS = "seqlen_minus_1"


MTP_TOKEN_ROWS = "seqlen_minus_2"


class SampleParallelProbeError(ValueError):
    """A calibration partition, sufficient statistic, or cache is invalid."""


def _sample_artifact_sha256(value: object, *, where: str) -> str:
    try:
        encoded = DIRECT_UTF8_STRICT.encoded(value)
    except (TypeError, ValueError) as exc:
        raise SampleParallelProbeError(
            f"{where} is not canonical JSON data"
        ) from exc
    return bytes_sha256hex(encoded)


def activation_cache_shard_stamp(
    partition_contract: Mapping[str, object],
    *,
    qname: str,
    rows_limit: int,
    candidate_rows: int,
    execution_identity_sha256: str,
) -> dict[str, object]:
    limit = int(rows_limit)
    candidates = int(candidate_rows)
    if limit < 1 or candidates < 1:
        raise SampleParallelProbeError(
            "activation-cache priority cardinalities must be positive"
        )
    execution_digest = str(execution_identity_sha256)
    if re.fullmatch(r"[0-9a-f]{64}", execution_digest) is None:
        raise SampleParallelProbeError(
            "activation-cache execution identity must be a SHA-256 digest"
        )
    return {
        "schema": ACTIVATION_CACHE_SHARD_SCHEMA,
        "priority_schema": ACTIVATION_PRIORITY_SCHEMA,
        "priority_group": activation_priority_group(qname),
        "selection": "local_top_r",
        "row_indices_scope": ROW_INDICES_SCOPE,
        "layout": "dense_linear",
        "global_calibration_hash": str(
            partition_contract["global_calibration_hash"]
        ),
        "execution_identity_sha256": execution_digest,
        "partition_index": int(partition_contract["partition_index"]),
        "rows_limit": limit,
        "candidate_rows": candidates,
    }


def _manifest_with_digest(
    *, kind: str, source_census_sha256: str,
    entries: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": QNAME_MANIFEST_SCHEMA,
        "kind": str(kind),
        "source_census_sha256": str(source_census_sha256),
        "entries": {
            str(name): dict(entry) for name, entry in sorted(entries.items())
        },
    }
    return {
        **body,
        "identity_sha256": _sample_artifact_sha256(
            body, where=f"{kind} qname manifest"
        ),
    }


def _validate_qname_manifest(
    raw: Mapping[str, object], *, kind: str, source_digest: str,
) -> dict[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema", "kind", "source_census_sha256", "entries",
        "identity_sha256",
    }:
        raise SampleParallelProbeError(f"{kind} qname manifest fields differ")
    body = dict(raw)
    digest = body.pop("identity_sha256", None)
    if (
        body.get("schema") != QNAME_MANIFEST_SCHEMA
        or body.get("kind") != kind
        or body.get("source_census_sha256") != source_digest
        or digest != _sample_artifact_sha256(body, where=f"{kind} qname manifest")
    ):
        raise SampleParallelProbeError(f"{kind} qname manifest identity differs")
    entries = body.get("entries")
    if not isinstance(entries, Mapping) or not entries:
        raise SampleParallelProbeError(f"{kind} qname manifest is empty")
    normalized_entries: dict[str, dict[str, object]] = {}
    entry_keys = {
        "source_tensor", "source_dtype", "shape", "disposition",
        "token_rows_per_sample", "terminal_format",
    }
    for raw_name, raw_entry in entries.items():
        name = str(raw_name)
        if not name or not isinstance(raw_entry, Mapping) or set(raw_entry) != entry_keys:
            raise SampleParallelProbeError(
                f"{kind} qname manifest entry {name!r} differs"
            )
        entry = dict(raw_entry)
        if (
            not str(entry["source_tensor"])
            or str(entry["source_dtype"]) != "BF16"
            or not isinstance(entry["shape"], Sequence)
            or isinstance(entry["shape"], (str, bytes))
            or len(entry["shape"]) != 2
            or any(type(value) is not int for value in entry["shape"])
            or any(int(value) < 1 for value in entry["shape"])
        ):
            raise SampleParallelProbeError(
                f"{kind} qname manifest entry {name!r} source shape differs"
            )
        normalized_entries[name] = {
            **entry,
            "source_tensor": str(entry["source_tensor"]),
            "source_dtype": str(entry["source_dtype"]),
            "shape": [int(value) for value in entry["shape"]],
        }
    return {**body, "entries": dict(sorted(normalized_entries.items())),
            "identity_sha256": str(digest)}


def validate_qname_census(raw: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "source_census", "probe_qname_manifest",
        "activation_qname_manifest", "terminal_qname_manifest",
        "identity_sha256",
    }:
        raise SampleParallelProbeError("qname census bundle fields differ")
    source_raw = raw.get("source_census")
    source_keys = {
        "schema", "model", "producer_profile_schema", "producer_profile_id",
        "source_layout", "source_config_sha256",
        "source_tensor_manifest_sha256", "source_weight_map_sha256",
        "source_tensor_count", "source_linear_count", "linear_entries",
        "source_model_identity",
        "identity_sha256",
    }
    if not isinstance(source_raw, Mapping) or set(source_raw) != source_keys:
        raise SampleParallelProbeError("source qname census fields differ")
    source_body = dict(source_raw)
    source_digest = source_body.pop("identity_sha256", None)
    # The producer-profile equality clauses checked this census against the
    # archived RTX4090 policy identity; that policy retired with the
    # Gridbook lane on 2026-09-02 (archive/gridbook_lane_2026-09-02/).
    # The key-set check above still requires the two recorded keys, because
    # every run contract on disk carries them.
    if (
        source_body.get("schema") != SOURCE_QNAME_CENSUS_SCHEMA
        or not str(source_body.get("model", ""))
        or source_body.get("source_layout") not in {
            "flattened_text", "official_wrapper"
        }
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(source_body.get(key, "")))
            for key in (
                "source_config_sha256", "source_tensor_manifest_sha256",
                "source_weight_map_sha256",
            )
        )
        or type(source_body.get("source_tensor_count")) is not int
        or type(source_body.get("source_linear_count")) is not int
        or not re.fullmatch(r"[0-9a-f]{64}", str(source_digest or ""))
        or source_digest != _sample_artifact_sha256(
            source_body, where="sample-parallel source qname census"
        )
    ):
        raise SampleParallelProbeError("source qname census identity differs")
    linear_entries = source_body.get("linear_entries")
    if not isinstance(linear_entries, Mapping) or int(
        source_body.get("source_linear_count", -1)
    ) != len(linear_entries) or int(
        source_body.get("source_tensor_count", -1)
    ) < len(linear_entries):
        raise SampleParallelProbeError("source qname census Linear cover differs")
    model_content = source_body.get("source_model_identity")
    model_content_keys = {
        "schema", "derivation", "upstream_content_sha256",
        "upstream_portable_content_sha256", "content_sha256",
        "resolved_commit", "checkpoint_tensors", "checkpoint_shards",
        "checkpoint_weight_map_sha256", "shards", "identity_sha256",
    }
    if not isinstance(model_content, Mapping) or set(model_content) != model_content_keys:
        raise SampleParallelProbeError("source model-content identity fields differ")
    model_content_body = dict(model_content)
    model_content_digest = model_content_body.pop("identity_sha256", None)
    model_shards = model_content_body.get("shards")
    upstream_content = model_content_body.get("upstream_content_sha256")
    upstream_portable_content = model_content_body.get(
        "upstream_portable_content_sha256"
    )
    if (
        model_content_body.get("schema")
        != SOURCE_MODEL_CONTENT_SCHEMA
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(model_content_body.get("content_sha256", ""))
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(model_content_body.get("checkpoint_weight_map_sha256", "")),
        )
        or not isinstance(model_shards, Sequence)
        or isinstance(model_shards, (str, bytes))
        or len(model_shards) != int(model_content_body.get("checkpoint_shards", -1))
        or type(model_content_body.get("checkpoint_shards")) is not int
        or type(model_content_body.get("checkpoint_tensors")) is not int
        or int(model_content_body.get("checkpoint_tensors", -1)) < 1
        or model_content_body.get("checkpoint_weight_map_sha256")
        != source_body.get("source_weight_map_sha256")
        or model_content_body.get("content_sha256") != _sample_artifact_sha256(
            {
                "source_config_sha256": source_body.get(
                    "source_config_sha256"
                ),
                "checkpoint_weight_map_sha256": model_content_body.get(
                    "checkpoint_weight_map_sha256"
                ),
                "shards": model_shards,
            },
            where="sample-parallel model content",
        )
        or model_content_digest != _sample_artifact_sha256(
            model_content_body, where="sample-parallel model-content identity"
        )
        or (
            upstream_content is not None
            and (
                type(upstream_content) is not str
                or re.fullmatch(r"[0-9a-f]{64}", upstream_content) is None
            )
        )
        or (
            upstream_portable_content is not None
            and (
                type(upstream_portable_content) is not str
                or re.fullmatch(
                    r"[0-9a-f]{64}", upstream_portable_content
                ) is None
            )
        )
        or (upstream_content is None) != (upstream_portable_content is None)
        or (
            model_content_body.get("derivation")
            == "validated_streamed_model_identity_cache_v1"
            and (
                upstream_content is None
                or upstream_portable_content is None
            )
        )
    ):
        raise SampleParallelProbeError("source model-content identity differs")
    for shard in model_shards:
        if (
            not isinstance(shard, Mapping)
            or set(shard) != {"name", "size", "sha256"}
            or type(shard.get("name")) is not str
            or type(shard.get("size")) is not int
            or type(shard.get("sha256")) is not str
            or Path(str(shard.get("name", ""))).name != shard.get("name")
            or int(shard.get("size", -1)) < 1
            or not re.fullmatch(r"[0-9a-f]{64}", str(shard.get("sha256", "")))
        ):
            raise SampleParallelProbeError(
                "source model-content shard identity differs"
            )
    # Reuse the entry validator through a temporary manifest; its digest is
    # derived here, so the source census remains the sole authority.
    all_manifest = _manifest_with_digest(
        kind="source_linear_census", source_census_sha256=str(source_digest),
        entries=linear_entries,  # type: ignore[arg-type]
    )
    all_entries = _validate_qname_manifest(
        all_manifest, kind="source_linear_census",
        source_digest=str(source_digest),
    )["entries"]
    allowed = {
        BODY_STATS_AND_ACTIVATION: (BODY_TOKEN_ROWS, None),
        LM_HEAD_STATS_ONLY: (LM_HEAD_TOKEN_ROWS, "BF16"),
        MTP_STATS_ONLY: (MTP_TOKEN_ROWS, "BF16"),
        TEXT_EXCLUDED: (None, None),
    }
    for name, entry in all_entries.items():
        disposition = entry["disposition"]
        expected = allowed.get(disposition)
        if expected is None or (
            entry["token_rows_per_sample"], entry["terminal_format"]
        ) != expected:
            raise SampleParallelProbeError(
                f"source qname {name!r} disposition contract differs"
            )
        name_disposition = (
            LM_HEAD_STATS_ONLY if name == "lm_head"
            else MTP_STATS_ONLY if name.startswith("mtp.")
            else TEXT_EXCLUDED if name.startswith("model.visual.")
            else BODY_STATS_AND_ACTIVATION
            if name.startswith("model.layers.") else None
        )
        if disposition != name_disposition:
            raise SampleParallelProbeError(
                f"source qname {name!r} namespace/disposition differs"
            )
    probe = _validate_qname_manifest(
        raw["probe_qname_manifest"], kind="full_probe",
        source_digest=str(source_digest),  # type: ignore[arg-type]
    )
    activation = _validate_qname_manifest(
        raw["activation_qname_manifest"], kind="dense_body_activation",
        source_digest=str(source_digest),  # type: ignore[arg-type]
    )
    terminal = _validate_qname_manifest(
        raw["terminal_qname_manifest"], kind="terminal_bf16_stats_only",
        source_digest=str(source_digest),  # type: ignore[arg-type]
    )
    expected_probe = {
        name: entry for name, entry in all_entries.items()
        if entry["disposition"] != TEXT_EXCLUDED
    }
    expected_activation = {
        name: entry for name, entry in all_entries.items()
        if entry["disposition"] == BODY_STATS_AND_ACTIVATION
    }
    expected_terminal = {
        name: entry for name, entry in all_entries.items()
        if entry["disposition"] in {LM_HEAD_STATS_ONLY, MTP_STATS_ONLY}
    }
    if probe["entries"] != expected_probe:
        raise SampleParallelProbeError(
            "full-probe qname manifest is not derived from the source census"
        )
    if activation["entries"] != expected_activation:
        raise SampleParallelProbeError(
            "activation qname manifest is not the dense-body source subset"
        )
    if terminal["entries"] != expected_terminal:
        raise SampleParallelProbeError(
            "terminal qname manifest is not the complete BF16 source subset"
        )
    if set(expected_terminal) != {
        name for name in expected_probe
        if name == "lm_head" or name.startswith("mtp.")
    } or "lm_head" not in expected_terminal or not any(
        name.startswith("mtp.") for name in expected_terminal
    ):
        raise SampleParallelProbeError("terminal BF16/MTP census is incomplete")
    source = {**source_body, "linear_entries": all_entries,
              "identity_sha256": str(source_digest)}
    body = {
        "source_census": source,
        "probe_qname_manifest": probe,
        "activation_qname_manifest": activation,
        "terminal_qname_manifest": terminal,
    }
    bundle_digest = raw.get("identity_sha256")
    if bundle_digest != _sample_artifact_sha256(
        body, where="sample-parallel qname census bundle"
    ):
        raise SampleParallelProbeError("qname census bundle digest differs")
    return {**body, "identity_sha256": str(bundle_digest)}

_SCALAR_SUM_FIELDS = ("h_trace_raw", "h_w2_sum_raw", "n_tokens_seen")


_DERIVED_FIELDS = frozenset({
    "h_trace",
    "h_w2_sum",
    "h_trace_per_expert",
    "h_trace_norm_tokens",
    "route_prob",
})


_DENSE_MARGINAL_FIELDS = frozenset({
    "fisher_row", "fisher_col", "g_sq_sum", "act_sq_sum", "act_absmax",
})


_FISHER_MARGINAL_REL_TOL = 1e-3


_PACKED_MARGINAL_FIELDS = frozenset({
    "expert_g_sq_sum", "expert_act_sq_sum", "expert_act_absmax",
    "expert_tokens", "h_trace_per_expert_raw", "h_trace_per_expert",
})


_ROUTED_NULLABLE_MARKERS = frozenset({"router_path", "expert_id", "route_prob"})


_PACKED_PRESENCE_MARKERS = frozenset({
    "num_experts", "expert_axis", "_packed_experts_module", "_packed_param",
    "packed_experts",
})


_DENSE_STAT_BASE_FIELDS = frozenset({
    "h_trace_raw", "h_w2_sum_raw", "h_trace", "h_w2_sum",
    "h_trace_norm_tokens", "w_max_abs", "w_norm_sq", "n_params",
    "in_features", "out_features", "n_tokens_seen", "route_prob",
    "router_path", "expert_id",
})


class SampleParallelMergeError(ValueError):
    """The declared sample partition or one of its payloads is inconsistent."""


def _array_shape(value: object) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _nonnegative_finite_array(value: object, *, where: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise SampleParallelMergeError(f"{where} is not an array") from exc
    if (
        array.dtype.kind not in "fiu"
        or not np.all(np.isfinite(array))
        or np.any(array < 0)
    ):
        raise SampleParallelMergeError(
            f"{where} must contain only finite nonnegative raw statistics"
        )
    return array


def _finite_nonnegative_scalar(value: object, *, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise SampleParallelMergeError(
            f"{where} must be a numeric scalar"
        )
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise SampleParallelMergeError(
            f"{where} must be finite and nonnegative"
        )
    return result


def _validate_fisher_marginal_trace(
    row: Mapping[str, object],
    *,
    where: str,
) -> None:
    """Require both Fisher marginals to reduce to their raw scalar trace."""
    trace = float(row["h_trace_raw"])
    for field in ("fisher_row", "fisher_col"):
        total = float(np.asarray(row[field], dtype=np.float64).sum())
        if not math.isclose(
            total,
            trace,
            rel_tol=_FISHER_MARGINAL_REL_TOL,
            abs_tol=0.0,
        ):
            raise SampleParallelMergeError(
                f"{where} sum({field})={total:.9g} differs from "
                f"h_trace_raw={trace:.9g} beyond the qualified fp32 "
                f"relative tolerance {_FISHER_MARGINAL_REL_TOL:g}"
            )


def _merge_stat_rows(
    rows: Sequence[Mapping[str, object]],
    global_tokens: int,
    *,
    expected_n_tokens: Sequence[int],
    qname: str,
    require_marginals: bool,
    expected_shape: tuple[int, int],
) -> dict:
    first = rows[0]
    if len(rows) != len(expected_n_tokens):
        raise AssertionError("stat/token-cover cardinality differs")
    expected_fields = set(_DENSE_STAT_BASE_FIELDS)
    if require_marginals:
        expected_fields.update(_DENSE_MARGINAL_FIELDS)
    expected_out, expected_in = (int(expected_shape[0]), int(expected_shape[1]))
    for field in _SCALAR_SUM_FIELDS:
        if any(field not in row for row in rows):
            raise SampleParallelMergeError(
                f"sample-parallel stat row lacks required raw field {field!r}"
            )
    for index, (row, expected_tokens) in enumerate(zip(
        rows, expected_n_tokens, strict=True
    )):
        if set(row) != expected_fields:
            raise SampleParallelMergeError(
                f"sample-parallel {qname} shard {index} stat fields differ "
                "from the closed dense schema"
            )
        for field in ("h_trace_raw", "h_w2_sum_raw"):
            try:
                value = _finite_nonnegative_scalar(
                    row[field],
                    where=(
                        f"sample-parallel {qname} shard {index} raw {field}"
                    ),
                )
            except KeyError as exc:
                raise SampleParallelMergeError(
                    f"sample-parallel {qname} shard {index} raw {field} is invalid"
                ) from exc
        for field in ("w_max_abs", "w_norm_sq"):
            try:
                value = _finite_nonnegative_scalar(
                    row[field],
                    where=f"sample-parallel {qname} shard {index} {field}",
                )
            except KeyError as exc:
                raise SampleParallelMergeError(
                    f"sample-parallel {qname} shard {index} {field} is invalid"
                ) from exc
        for field in ("n_params", "in_features", "out_features"):
            value = row.get(field)
            if type(value) is not int or value < 1:
                raise SampleParallelMergeError(
                    f"sample-parallel {qname} shard {index} {field} is invalid"
                )
        if (
            row["out_features"] != expected_out
            or row["in_features"] != expected_in
            or row["n_params"] != expected_out * expected_in
        ):
            raise SampleParallelMergeError(
                f"sample-parallel {qname} shard {index} geometry differs "
                "from the source qname manifest"
            )
        seen = row.get("n_tokens_seen")
        if type(seen) is not int or seen != int(expected_tokens):
            raise SampleParallelMergeError(
                f"sample-parallel {qname} shard {index} n_tokens_seen differs: "
                f"observed={seen!r}, expected={expected_tokens}"
            )
        forbidden = sorted(set(row) & _PACKED_MARGINAL_FIELDS)
        if forbidden:
            raise SampleParallelMergeError(
                f"sample-parallel v1 rejects routed/packed stats for {qname}: "
                f"{forbidden[:8]}"
            )
        routed = sorted(
            marker for marker in _ROUTED_NULLABLE_MARKERS
            if row[marker] is not None
        )
        routed.extend(sorted(set(row) & _PACKED_PRESENCE_MARKERS))
        if routed:
            raise SampleParallelMergeError(
                f"sample-parallel v1 rejects routed/packed markers for {qname}: "
                f"{routed[:8]}"
            )
        present_marginals = set(row) & _DENSE_MARGINAL_FIELDS
        if require_marginals and present_marginals != _DENSE_MARGINAL_FIELDS:
            raise SampleParallelMergeError(
                f"sample-parallel {qname} shard {index} dense marginals are incomplete"
            )
        if not require_marginals and present_marginals:
            raise SampleParallelMergeError(
                f"sample-parallel {qname} shard {index} emitted undeclared marginals"
            )
        for field in present_marginals:
            array = _nonnegative_finite_array(
                row[field], where=f"sample-parallel {qname} shard {index} {field}"
            )
            expected_length = (
                expected_out
                if field in {"fisher_row", "g_sq_sum"}
                else expected_in
            )
            if array.dtype != np.dtype(np.float32) or array.shape != (
                expected_length,
            ):
                raise SampleParallelMergeError(
                    f"sample-parallel {qname} shard {index} {field} "
                    "dtype/shape differs from the source qname manifest"
                )
        if require_marginals:
            _validate_fisher_marginal_trace(
                row,
                where=f"sample-parallel {qname} shard {index}",
            )
    excluded = (
        set(_SCALAR_SUM_FIELDS)
        | _DERIVED_FIELDS
        | _DENSE_MARGINAL_FIELDS
        | _PACKED_MARGINAL_FIELDS
    )
    static = {key: first[key] for key in first if key not in excluded}
    for row in rows[1:]:
        observed = {key: row[key] for key in row if key not in excluded}
        if observed != static:
            raise SampleParallelMergeError(
                "sample-parallel stat static metadata differs"
            )

    merged = copy.deepcopy(dict(first))
    for field in _DERIVED_FIELDS:
        merged.pop(field, None)
    merged["h_trace_raw"] = sum(float(row["h_trace_raw"]) for row in rows)
    merged["h_w2_sum_raw"] = sum(float(row["h_w2_sum_raw"]) for row in rows)
    merged["n_tokens_seen"] = sum(int(row["n_tokens_seen"]) for row in rows)
    if not np.isfinite(merged["h_trace_raw"]) or not np.isfinite(
        merged["h_w2_sum_raw"]
    ):
        raise SampleParallelMergeError(
            f"sample-parallel merged raw scalar overflow for {qname}"
        )

    for field in _DENSE_MARGINAL_FIELDS:
        present = [field in row for row in rows]
        if any(present) and not all(present):
            raise SampleParallelMergeError(
                f"sample-parallel dense marginal {field!r} presence differs"
            )
        if all(present):
            shapes = {_array_shape(row[field]) for row in rows}
            if len(shapes) != 1:
                raise SampleParallelMergeError(
                    f"sample-parallel dense marginal {field!r} shape differs"
                )
    for field in _DENSE_MARGINAL_FIELDS:
        merged.pop(field, None)
    for row in rows:
        merge_marginals(merged, row)
    for field in _DENSE_MARGINAL_FIELDS & set(merged):
        _nonnegative_finite_array(
            merged[field], where=f"sample-parallel merged {qname} {field}"
        )
    if require_marginals:
        _validate_fisher_marginal_trace(
            merged, where=f"sample-parallel merged {qname}"
        )

    finalize_fisher_stats({"row": merged}, global_tokens)
    for field in ("h_trace", "h_w2_sum"):
        value = float(merged.get(field, float("nan")))
        if not np.isfinite(value) or value < 0.0:
            raise SampleParallelMergeError(
                f"sample-parallel finalized {qname} {field} is invalid"
            )
    return merged


def validate_sample_parallel_stat_row(
    row: Mapping[str, object],
    *,
    qname: str,
    expected_tokens: int,
    expected_shape: tuple[int, int],
    require_marginals: bool,
) -> None:
    """Validate one worker row with the reducer's closed dense contract."""
    _merge_stat_rows(
        [row],
        int(expected_tokens),
        expected_n_tokens=[int(expected_tokens)],
        qname=str(qname),
        require_marginals=bool(require_marginals),
        expected_shape=tuple(int(value) for value in expected_shape),
    )
