"""Calibration identities retain their bytes and reuse the owner (#1795)."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import calibration_data, digests


def _draw():
    # A noncontiguous int64 view: the draw hashes converted int32 bytes, while
    # the receipt hashes contiguous int64 bytes. These identities must differ.
    ids = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 2**31 - 1]]).t()
    values = [0, 1, 2, 3, 4, 5, 6, 2**31 - 1]
    raw32 = struct.pack("=8i", *values)
    raw64 = struct.pack("=8q", *values)
    provenance = {
        "fit_ids_sha256": hashlib.sha256(raw32).hexdigest(),
        "fit_tokens": 8, "nsamples": 4, "seqlen": 2,
    }
    return ids, provenance, raw32, raw64


def _artifact(tmp_path):
    ids, provenance, raw32, raw64 = _draw()
    path = tmp_path / "draw.safetensors"
    save_file({"calibration_ids": ids.contiguous()}, str(path),
              metadata={"calibration_provenance": json.dumps(provenance)})
    raw = path.read_bytes()
    return path, raw, ids, raw32, raw64


def test_draw_retains_both_native_byte_identities():
    ids, provenance, raw32, raw64 = _draw()
    result, receipt = calibration_data._validate_calibration_draw(
        ids, provenance, artifact_sha256="a" * 64, n_samples=4, seqlen=2)
    assert result is ids
    assert not ids.is_contiguous()
    assert receipt["calibration_sha256"] == hashlib.sha256(raw64).hexdigest()
    assert receipt["calibration_sha256"] != hashlib.sha256(raw32).hexdigest()
    assert receipt["artifact_sha256"] == "a" * 64
    assert receipt["provenance"] is provenance


def test_draw_routes_exact_buffers_through_byte_owner(monkeypatch):
    ids, provenance, raw32, raw64 = _draw()
    seen = []

    def record(raw):
        seen.append(raw)
        return digests.bytes_sha256hex(raw)

    monkeypatch.setattr(calibration_data, "bytes_sha256hex", record)
    calibration_data._validate_calibration_draw(
        ids, provenance, artifact_sha256="a" * 64, n_samples=4, seqlen=2)
    assert seen == [raw32, raw64]


def test_offline_routes_artifact_draw_tensor_and_mutation_fence(tmp_path, monkeypatch):
    path, raw, ids, raw32, raw64 = _artifact(tmp_path)
    seen = []

    def record(payload):
        seen.append(payload)
        return digests.bytes_sha256hex(payload)

    monkeypatch.setattr(calibration_data, "bytes_sha256hex", record)
    loaded, receipt = calibration_data.load_calibration_input(
        path, expected_sha256=hashlib.sha256(raw).hexdigest(), n_samples=4, seqlen=2)
    assert torch.equal(loaded, ids)
    assert seen == [raw, raw32, raw64, raw]
    assert receipt["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["calibration_sha256"] == hashlib.sha256(raw64).hexdigest()


def test_offline_mutation_fence_remains_a_second_read(tmp_path, monkeypatch):
    path, raw, _ids, _raw32, _raw64 = _artifact(tmp_path)
    reads = []

    def changing_read(given):
        assert given == path
        reads.append(given)
        return raw if len(reads) == 1 else raw + b"changed"

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    with pytest.raises(ValueError, match="^exact calibration input changed while loading$"):
        calibration_data.load_calibration_input(
            path, expected_sha256=hashlib.sha256(raw).hexdigest(), n_samples=4, seqlen=2)
    assert reads == [path, path]


def test_staged_routes_the_owned_artifact_without_path_reads(tmp_path, monkeypatch):
    from prismaquant import staged_tier_policy

    path, raw, ids, raw32, raw64 = _artifact(tmp_path)
    seen = []

    def record(payload):
        seen.append(payload)
        return digests.bytes_sha256hex(payload)

    def refuse_pool(given):
        pytest.fail(f"unexpected pool read: {given}")

    monkeypatch.setattr(staged_tier_policy, "policy_is_active", lambda: True)
    monkeypatch.setattr(calibration_data, "_read_calibration_payload", lambda *args: raw)
    monkeypatch.setattr(calibration_data, "bytes_sha256hex", record)
    monkeypatch.setattr(Path, "read_bytes", refuse_pool)
    loaded, receipt = calibration_data.load_calibration_input(
        path, expected_sha256=hashlib.sha256(raw).hexdigest(), n_samples=4, seqlen=2)
    assert torch.equal(loaded, ids)
    assert seen == [raw, raw32, raw64]
    assert receipt["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
