"""PQ #1713: PrismaSnap probe hashes delegate to the bytes owner.

Routing: _tensor_payload_sha256 + _importance_digest + _load_probe must call
bytes_sha256hex. _validate_source_identity + _validate_probe_binding_receipt
+ bind_legacy_text_probe share the same one-line recipe but need a live
_checkpoint and sealed binding docs to invoke; their rows are pinned by the
baseline tripwire. Values: byte-identical to the verbatim old spellings.
"""
import hashlib
import pickle

import numpy as np
import torch

from prismaquant import prismasnap_checkpoint as snap
from prismaquant.digests import bytes_sha256hex


def _spy(monkeypatch):
    calls = []
    real = getattr(snap, "bytes_sha256hex", None)

    def stand_in(value):
        calls.append(value)
        return real(value) if real is not None else None

    monkeypatch.setattr(snap, "bytes_sha256hex", stand_in, raising=False)
    return calls


def test_probe_hashes_route_to_owner(monkeypatch, tmp_path):
    hash_calls = _spy(monkeypatch)
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    digest = snap._tensor_payload_sha256(tensor, where="test")
    assert isinstance(digest, str) and len(digest) == 64
    assert snap._importance_digest([0.5, 1.0, 0.0]) == bytes_sha256hex(
        np.asarray([0.5, 1.0, 0.0], dtype=np.float32).tobytes(order="C"))
    probe = {"stats": {"lm_head": {"act_sq_sum": [1.0], "in_features": 1,
                                   "out_features": 1}},
             "meta": {}}
    probe_path = tmp_path / "probe.pkl"
    probe_path.write_bytes(pickle.dumps(probe))
    stats, meta, probe_sha = snap._load_probe(probe_path)
    assert stats and probe_sha == hashlib.sha256(probe_path.read_bytes()).hexdigest()
    assert len(hash_calls) == 3


def test_probe_values_match_verbatim_spellings():
    raw = b"\x00\x01prismasnap"
    assert bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()
    array = np.asarray([0.5, 1.0, 0.0], dtype=np.float32)
    assert bytes_sha256hex(array.tobytes(order="C")) == hashlib.sha256(
        array.tobytes(order="C")).hexdigest()
