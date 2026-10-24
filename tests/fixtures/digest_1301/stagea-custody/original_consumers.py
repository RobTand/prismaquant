# Selected original consumer functions; not an owner-derived oracle.
# Research base: 6d8288adc94b3c4b624c143ed372fb6513f32784
from __future__ import annotations
import hashlib
import json
import stat
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# prismaquant/joint_replay_frontier.py; blob 17f28517c5d94774bfdc7976ef8b43ff8a7dcd14
def replay_read_items(
    roster: Iterable[str], cells_by_unit: Mapping[str, Iterable[str]]
) -> list[tuple[str, str, str | None]]:
    """The exact order a replay reads, one complete unit at a time.

    ``_qualification_replay`` authenticates a unit's X/H capture and then each
    of its measured rungs' wire and render. The unit is the durable boundary,
    so the order is grouped by unit -- capture, then that unit's cells -- not
    two whole-roster passes. Items are ``(kind, qname, fmt)`` with ``fmt``
    ``None`` for the capture.

    A unit's cells are read in the order the caller's mapping lists them, which
    is the campaign's own cell order (``sorted(anchors)`` on both ends). The
    order is *sealed*, not re-derived: two mappings over the same cells in
    different orders are two different frontiers, and the digest has to tell
    them apart or a reordered read set would pass its own check.
    """
    items: list[tuple[str, str, str | None]] = []
    for name in sorted(roster):
        cells = cells_by_unit.get(name)
        if cells is None:
            raise ValueError(f"replay roster names a unit with no cells: {name}")
        items.append((CAPTURE, name, None))
        for fmt in cells:
            items.append((WIRE, name, fmt))
            items.append((RENDER, name, fmt))
    return items

def _framed(items: Sequence[tuple[str, str, str | None]]) -> bytes:
    """Length-framed items, so adjacent rows cannot be read two ways."""
    digest = hashlib.sha256(b"prismaquant.joint_prepare.replay_read_order.v1\n")
    for kind, name, fmt in items:
        row = json.dumps((kind, name, fmt), ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
        digest.update(len(row).to_bytes(8, "big"))
        digest.update(row)
    return digest.digest()

def roster_sha256(roster: Iterable[str]) -> str:
    digest = hashlib.sha256(b"prismaquant.joint_prepare.replay_roster.v1\n")
    for name in roster:
        row = json.dumps(name, ensure_ascii=False, separators=(",", ":")).encode()
        digest.update(len(row).to_bytes(8, "big"))
        digest.update(row)
    return digest.hexdigest()

def read_order_sha256(items: Sequence[tuple[str, str, str | None]]) -> str:
    return _framed(items).hex()

# prismaquant/joint_forward_resume.py; blob c23724f01823b67f0d71ca57280ec23325089624
SCHEMA = 'prismaquant.joint_forward_recovery.v1'
class ForwardRecoveryRefused(RuntimeError):
    pass

PROOF_READ_MAX_BYTES = 128 << 20
_GROUP_READS = 4
_HEADER_READS = 2
def capsule_byte_limit(document):
    """The most a capsule declaring these groups can occupy."""
    groups = document.get('groups')
    if not isinstance(groups, list):
        raise ForwardRecoveryRefused('recovery capsule declares no groups')
    return (_HEADER_READS + _GROUP_READS * len(groups)) * PROOF_READ_MAX_BYTES

def _read(path, expected=None):
    """Read one proof document; a pinned read verifies its digest first.

    An unpinned read is one PB record, bounded by ``PROOF_READ_MAX_BYTES``.
    A pinned read hashes the file in fixed chunks before loading it, so a
    wrong file is refused without being held in memory; a pinned capsule is
    then bounded by the groups it declares, not by one record's bound.
    """
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or (
            expected is None and before.st_size > PROOF_READ_MAX_BYTES):
        raise ForwardRecoveryRefused('recovery proof is not a bounded regular file')
    if expected is not None:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        if file_stat_signature(path.lstat()) != file_stat_signature(before):
            raise ForwardRecoveryRefused('recovery proof changed during read')
        if digest.hexdigest() != expected:
            raise ForwardRecoveryRefused('recovery proof SHA256 mismatch')
    raw = path.read_bytes()
    if file_stat_signature(path.lstat()) != file_stat_signature(before):
        raise ForwardRecoveryRefused('recovery proof changed during read')
    digest = bytes_sha256hex(raw)
    if expected is not None and digest != expected:
        raise ForwardRecoveryRefused('recovery proof SHA256 mismatch')
    document = json.loads(raw)
    if expected is not None and len(raw) > (
            capsule_byte_limit(document)
            if isinstance(document, dict) and document.get('schema') == SCHEMA
            else PROOF_READ_MAX_BYTES):
        raise ForwardRecoveryRefused(
            'recovery proof is larger than its declared groups allow')
    return document, digest

# prismaquant/stage_a_chain_seed.py; blob ddc95808ec42bd05e29dfcf583ba8158c1fd905c
def tensor_payload_sha256(tensor, *, chunk_bytes: int = 1 << 26) -> str:
    """sha256 of a tensor's contiguous payload bytes, dtype and shape aside.

    The bytes are copied to the host ``chunk_bytes`` at a time, so hashing a
    device tensor holds one chunk beside it, not a second copy of it.
    """
    import torch

    data = tensor.detach().contiguous().reshape(-1)
    digest = hashlib.sha256()
    step = max(1, int(chunk_bytes) // max(data.element_size(), 1))
    for start in range(0, data.numel(), step):
        digest.update(data[start:start + step].to("cpu").view(torch.uint8).numpy())
    return digest.hexdigest()


# Selected original dependencies from prismaquant/joint_replay_frontier.py at research base.
CAPTURE = "capture"

WIRE = "wire"

RENDER = "render"


# Selected original dependencies from prismaquant/digests.py at research base.
def bytes_sha256hex(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


# Selected original dependencies from prismaquant/file_identity.py at research base.
def file_stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    """``(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)`` of one stat."""
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)

