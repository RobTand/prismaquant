"""Generic Stage A/B input checks (decoupling step 7, PQ #1555).

These helpers bind a campaign's inputs to the bytes and settings a stage
reads: a path held to its SHA-256, a memoized bound read behind a stat
fence, an identity comparison, the explicit source-prefetch settings, and
a producer's source identity record. They know no serving lane, so the
stage modules, the lane modules and the tools all read them from here.

Every refusal message is the one the helper carried before it moved, since
tests and receipts match on them.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

from .digests import bytes_sha256hex, file_sha256hex
from .file_identity import file_stat_signature
from .schemas import Contract


require = Contract(ValueError).require


def same(actual, expected, label, *, contract: Contract | None = None):
    """Actual two-thing comparability with the caller's refusal vocabulary."""
    check = require if contract is None else contract.require
    check(actual == expected, f"{label}: identity mismatch")


def recorded_same(actual, expected, label):
    """Recorded-versus-running provenance, stamped in default dev mode."""
    from .dev_mode import seal_check
    seal_check(label, expected, actual, where="Stage A/B recorded versus running provenance",
               refusal=lambda: ValueError(f"{label}: identity mismatch"))


def bound(record, label):
    """The path of an independently bound ``{"path", "sha256"}`` record."""
    require(isinstance(record, dict) and set(record) == {"path", "sha256"},
            f"{label}: independently bound path/SHA256 required")
    path = Path(record["path"])
    require(file_sha256hex(path) == record["sha256"], f"{label}: artifact checksum changed")
    return path


def bound_stat_fence(path):
    """The stat identity a memoized bound read trusts a hit on (P3 #682).

    ``(st_mode, *file_identity.file_stat_signature)``: the file type, then
    the one file stat identity every fence reads (PQ #1531). Index 3 is the
    size.
    """
    value = path.stat()
    return (value.st_mode, *file_stat_signature(value))


#: Process-scoped bound bytes by ``(path, sha256)`` with the stat fence the
#: bytes were verified under. The digest check is what authenticates the
#: bytes; the fence only admits reusing them without re-reading and
#: re-hashing. A fence drift re-reads and re-verifies, and a digest mismatch
#: still refuses -- a memo hit never authenticates anything.
_BOUND_BYTES = {}


#: A process-wide reader for :func:`read_bound`'s bytes, ``(path, sha256,
#: label) -> bytes``. None reads the path. The Stage B preparation installs
#: its strict staged reader here (``stage_b_prep_io.bind_staged_reads``,
#: PQ #1092), so the control documents it reads come off the stage.
BOUND_READER = None


def read_bound(record, label):
    """The bytes of a bound ``{"path", "sha256"}`` record, verified."""
    require(isinstance(record, dict) and set(record) == {'path', 'sha256'}, f'{label}: bound path and SHA256 required')
    path = Path(record['path'])
    key = (str(path), record['sha256'])
    try:
        fence = bound_stat_fence(path)
    except OSError:
        fence = None
    if fence is not None:
        hit = _BOUND_BYTES.get(key)
        if hit is not None and hit[0] == fence:
            return hit[1]
    raw = (path.read_bytes() if BOUND_READER is None
           else BOUND_READER(path, record['sha256'], label))
    same(bytes_sha256hex(raw), record['sha256'], f'{label}: owned bytes')
    if fence is not None and len(raw) == fence[3]:
        _BOUND_BYTES[key] = (fence, raw)
    return raw


def source_prefetch(config):
    """The plan's explicit, complete ``source_prefetch`` settings, validated."""
    prefetch = config.get("source_prefetch")
    fields = {"max_cache_slots", "prefetch_workers", "prefetch_lookahead",
              "cache_headroom_gb", "prefetch_min_available_gb",
              "require_prefetched_residency"}
    require(isinstance(prefetch, dict) and set(prefetch) == fields,
            "explicit complete source_prefetch settings required")
    require(prefetch["require_prefetched_residency"] is True,
            "source_prefetch must require prefetched residency")
    for name in ("max_cache_slots", "prefetch_workers", "prefetch_lookahead"):
        require(type(prefetch[name]) is int and prefetch[name] > 0,
                f"source_prefetch requires positive {name}")
    require(prefetch["prefetch_lookahead"] < prefetch["max_cache_slots"],
            "source_prefetch lookahead must fit the declared cache slots")
    for name in ("cache_headroom_gb", "prefetch_min_available_gb"):
        require(type(prefetch[name]) in (int, float) and
                math.isfinite(prefetch[name]) and prefetch[name] > 0,
                f"source_prefetch requires positive finite {name}")
    return dict(prefetch)


#: The keys of a producer's source identity record.
SOURCE_IDENTITY_KEYS = ("auxiliary_sha256", "config_sha256", "files", "tensors")


class ExpertProjectionError(RuntimeError):
    """The producer's projection is absent, malformed, or does not cover a unit."""


def require_source_identity(source: Any) -> dict:
    """A copy of a producer's source identity record, checked for shape."""
    if not isinstance(source, Mapping) or set(source) != set(SOURCE_IDENTITY_KEYS):
        raise ExpertProjectionError(
            "producer projection source identity must carry exactly "
            f"{sorted(SOURCE_IDENTITY_KEYS)}")
    for key in ("config_sha256",):
        if not isinstance(source[key], str) or not source[key]:
            raise ExpertProjectionError(f"producer projection source.{key} must be a sha256")
    for key in ("auxiliary_sha256", "files", "tensors"):
        if not isinstance(source[key], Mapping):
            raise ExpertProjectionError(f"producer projection source.{key} must be an object")
    for tensor, file in source["tensors"].items():
        if not isinstance(tensor, str) or not isinstance(file, str) or file not in source["files"]:
            raise ExpertProjectionError(
                f"producer projection source.tensors[{tensor!r}] must name a hashed file")
    return dict(source)
