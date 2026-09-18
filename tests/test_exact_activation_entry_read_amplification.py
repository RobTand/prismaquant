"""Read amplification and digest identity for exact boundary activation entries.

RobTand/prismaquant#735. Each boundary entry used to be read four times: the
writer re-opened the published file to hash it, and the prefetch hashed the
file in full before ``torch.load`` read it again. These tests measure the
syscall-level read volume per entry (``/proc/self/io`` ``rchar``, which counts
bytes handed to read syscalls whether or not they came from page cache) and
pin every refusal the prefetch already owned.

This module imports only ``prismaquant.perturbed_x_cache`` so it carries no
sibling-test dependency.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from prismaquant.perturbed_x_cache import (
    ExactActivationReference,
    prefetch_exact_activation_cache_entries,
    write_exact_activation_cache_entry,
)

SESSION = "exact-activation-read-amplification"


def _rchar() -> int:
    with open("/proc/self/io", "rb") as handle:
        for line in handle:
            if line.startswith(b"rchar:"):
                return int(line.split()[1])
    raise RuntimeError("this platform does not report /proc/self/io rchar")


def _identity(slot: str, *, session: str = SESSION) -> dict:
    return {"session": session, "slot": slot, "kind": "boundary",
            "coordinates": {"batch": 0, "boundary": 0, "probe": None}}


def _write(directory: Path, name: str, tensor: torch.Tensor, *,
           session: str = SESSION, release_file_pages: bool = False):
    nbytes = tensor.numel() * tensor.element_size()
    return write_exact_activation_cache_entry(
        directory, name, tensor, identity=_identity(name, session=session),
        max_tensor_bytes=nbytes, max_file_bytes=nbytes + 65536,
        release_file_pages=release_file_pages)


def _prefetch(references, *, session: str = SESSION, release_file_pages: bool = False):
    budget = sum(ref.tensor_bytes for ref in references)
    return prefetch_exact_activation_cache_entries(
        references, max_tensor_bytes=budget, expected_session=session,
        release_file_pages=release_file_pages)


def _warm(directory: Path) -> None:
    """Pay the one-off import/lazy-init read cost before any measurement."""
    ref = _write(directory, "warm", torch.zeros(4, 8))
    with _prefetch([ref]) as window:
        window.get(ref)


def test_write_hashes_the_serialized_bytes_without_rereading_the_entry(tmp_path):
    """The writer must not read the file it just published (#735, write leg)."""
    _warm(tmp_path)
    tensor = torch.randn(1024, 1024)  # 4 MiB, far above any bookkeeping noise
    before = _rchar()
    ref = _write(tmp_path, "boundary-write", tensor)
    read_bytes = _rchar() - before
    assert ref.file_bytes >= tensor.numel() * tensor.element_size()
    assert read_bytes < ref.file_bytes // 4, (
        f"write re-read {read_bytes} bytes of a {ref.file_bytes}-byte entry")


def test_prefetch_reads_each_entry_once(tmp_path):
    """Hashing and deserializing must share one pass (#735, prefetch leg)."""
    _warm(tmp_path)
    tensor = torch.randn(1024, 1024)
    ref = _write(tmp_path, "boundary-prefetch", tensor)
    before = _rchar()
    with _prefetch([ref]) as window:
        restored = window.get(ref)
    read_bytes = _rchar() - before
    assert torch.equal(restored, tensor)
    assert read_bytes < ref.file_bytes * 3 // 2, (
        f"prefetch read {read_bytes} bytes of a {ref.file_bytes}-byte entry")


@pytest.mark.parametrize("factory", [
    lambda: torch.randn(3, 5, 7),
    lambda: torch.randn(2, 4, 8, 16),
    lambda: torch.randn(4, 16).to(torch.bfloat16),
    lambda: torch.randn(8, 16).t(),
    lambda: torch.randn(1, 1),
])
def test_receipt_digest_equals_the_published_file_digest(tmp_path, factory):
    """Old path vs new path.

    ``hashlib.file_digest`` over the published file is verbatim what
    ``origin/main`` computed for the receipt, so this compares the streamed
    digest against the pre-change one for the same tensor.
    """
    tensor = factory()
    ref = _write(tmp_path, f"digest-{tuple(tensor.shape)}-{tensor.dtype}", tensor)
    with open(ref.path, "rb") as handle:
        expected = hashlib.file_digest(handle, "sha256").hexdigest()
    assert ref.sha256 == expected
    assert ref.file_bytes == Path(ref.path).lstat().st_size


def test_realistic_entry_round_trips_through_the_single_pass_reader(tmp_path):
    """A production-sized 16 MiB entry, the size that exercises the reader."""
    tensor = torch.randn(2048, 2048)
    ref = _write(tmp_path, "boundary-realistic", tensor, release_file_pages=True)
    with open(ref.path, "rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == ref.sha256
    with _prefetch([ref], release_file_pages=True) as window:
        assert torch.equal(window.get(ref), tensor)


def test_prefetch_refuses_a_truncated_entry(tmp_path):
    ref = _write(tmp_path, "truncated", torch.randn(64, 64))
    data = Path(ref.path).read_bytes()
    Path(ref.path).write_bytes(data[:-16])
    with pytest.raises(RuntimeError, match="size changed"):
        with _prefetch([ref]):
            pytest.fail("truncated entry exposed as ready")


def test_prefetch_refuses_an_extended_entry(tmp_path):
    ref = _write(tmp_path, "extended", torch.randn(64, 64))
    with open(ref.path, "ab") as handle:
        handle.write(b"\0" * 32)
    with pytest.raises(RuntimeError, match="size changed"):
        with _prefetch([ref]):
            pytest.fail("extended entry exposed as ready")


def test_prefetch_refuses_a_same_size_content_change(tmp_path):
    ref = _write(tmp_path, "flipped", torch.randn(64, 64))
    data = bytearray(Path(ref.path).read_bytes())
    data[len(data) // 2] ^= 0x01
    Path(ref.path).write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="checksum changed"):
        with _prefetch([ref]):
            pytest.fail("rewritten entry exposed as ready")


def test_prefetch_refuses_a_receipt_whose_digest_disagrees(tmp_path):
    ref = _write(tmp_path, "receipt-digest", torch.randn(32, 32))
    bad = ExactActivationReference(ref.path, ref.name, ref.metadata_json, ref.shape,
                                   ref.dtype, ref.tensor_bytes, ref.file_bytes,
                                   "0" * 64)
    with pytest.raises(RuntimeError, match="checksum changed"):
        with _prefetch([bad]):
            pytest.fail("mismatched receipt exposed as ready")


@pytest.mark.parametrize("field,value", [
    ("shape", (1, 7)),
    ("dtype", "torch.float64"),
    ("tensor_bytes", 8),
    ("name", "someone-else"),
])
def test_prefetch_refuses_a_payload_that_differs_from_its_receipt(tmp_path, field, value):
    ref = _write(tmp_path, "payload", torch.randn(16, 16))
    fields = dict(path=ref.path, name=ref.name, metadata_json=ref.metadata_json,
                  shape=ref.shape, dtype=ref.dtype, tensor_bytes=ref.tensor_bytes,
                  file_bytes=ref.file_bytes, sha256=ref.sha256)
    fields[field] = value
    with pytest.raises(RuntimeError, match="differs from its receipt"):
        with _prefetch([ExactActivationReference(**fields)]):
            pytest.fail("receipt/payload disagreement exposed as ready")


def test_prefetch_refuses_a_foreign_session(tmp_path):
    ref = _write(tmp_path, "foreign", torch.randn(16, 16))
    with pytest.raises(RuntimeError, match="different session identity"):
        with _prefetch([ref], session="another-session"):
            pytest.fail("foreign-session entry exposed as ready")


def test_prefetch_refuses_a_metadata_rewrite(tmp_path):
    ref = _write(tmp_path, "metadata", torch.randn(16, 16))
    payload = torch.load(ref.path, map_location="cpu", weights_only=True)
    metadata = json.loads(ref.metadata_json)
    metadata["identity"]["slot"] = "elsewhere"
    payload["exact"] = metadata
    Path(ref.path).unlink()
    torch.save(payload, ref.path)
    fields = dict(path=ref.path, name=ref.name, metadata_json=ref.metadata_json,
                  shape=ref.shape, dtype=ref.dtype, tensor_bytes=ref.tensor_bytes,
                  file_bytes=Path(ref.path).lstat().st_size, sha256=hashlib.sha256(
                      Path(ref.path).read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="differs from its receipt"):
        with _prefetch([ExactActivationReference(**fields)]):
            pytest.fail("rewritten metadata exposed as ready")


def test_prefetch_refuses_a_missing_entry(tmp_path):
    ref = _write(tmp_path, "missing", torch.randn(16, 16))
    Path(ref.path).unlink()
    with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
        with _prefetch([ref]):
            pytest.fail("missing entry exposed as ready")


def test_read_block_is_the_block_the_stdlib_digest_used():
    """The single pass keeps ``hashlib.file_digest``'s own block size.

    A whole-file read removes the kernel readahead that used to overlap the
    next block with the hash of the current one, and measured slower.
    """
    import inspect

    from prismaquant.perturbed_x_cache import _ENTRY_READ_BLOCK_BYTES

    default = inspect.signature(hashlib.file_digest).parameters["_bufsize"].default
    assert _ENTRY_READ_BLOCK_BYTES == default


def test_a_shared_scratch_is_reused_across_windows(tmp_path):
    """A caller-owned buffer is allocated once, not once per window."""
    import tracemalloc

    from prismaquant.perturbed_x_cache import EntryReadScratch

    scratch = EntryReadScratch()
    refs = [_write(tmp_path, f"shared-{index}", torch.randn(1024, 1024)) for index in range(2)]
    with prefetch_exact_activation_cache_entries(
            [refs[0]], max_tensor_bytes=refs[0].tensor_bytes, expected_session=SESSION,
            release_file_pages=False, scratch=scratch) as window:
        window.get(refs[0])
    tracemalloc.start()
    try:
        with prefetch_exact_activation_cache_entries(
                [refs[1]], max_tensor_bytes=refs[1].tensor_bytes, expected_session=SESSION,
                release_file_pages=False, scratch=scratch) as window:
            window.get(refs[1])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < refs[1].file_bytes // 4, f"second window allocated {peak} bytes"
    scratch.release()


def test_the_boundary_owner_reuses_one_read_buffer_and_frees_it(tmp_path):
    """The same buffer serves every window of a generation, and is released."""
    import tracemalloc

    from prismaquant.cost_streaming import BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts

    tensor = torch.randn(1024, 1024)
    nbytes = tensor.numel() * tensor.element_size()
    policy = {"schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(tmp_path),
              "max_resident_bytes": nbytes, "max_auxiliary_bytes": 1 << 20,
              "max_artifact_bytes": 1 << 26, "prefetch_batches": 1}
    with StreamedBoundaryArtifacts(policy) as owner:
        owner.bind({"fixture": "scratch-reuse"}, n_probes=1)
        first = owner.write(tensor, batch_index=0, boundary_index=0)
        second = owner.write(tensor, batch_index=1, boundary_index=0)
        with owner.prefetch([first]) as window:
            owner.get(window, first)
        tracemalloc.start()
        try:
            with owner.prefetch([second]) as window:
                assert torch.equal(owner.get(window, second), tensor)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < second.file_bytes // 4, f"second window allocated {peak} bytes"
        owner.retire(first)
        owner.retire(second)
    assert owner._scratch is not None and len(owner._scratch.buffer(0)) == 0
