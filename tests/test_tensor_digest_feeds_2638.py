import hashlib
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prismaquant import production_weight_cache, stage_a_chain_seed


@pytest.mark.parametrize("path", ["stage-a", "pwc"])
def test_tensor_digest_feeds_are_bounded(monkeypatch, path):
    tensor = torch.tensor(
        [[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, device="cpu"
    ).t()
    empty = torch.empty((2, 0), dtype=torch.float32, device="cpu")
    sha256 = hashlib.sha256
    transposed_digest = sha256(struct.pack("=6f", 1.0, 4.0, -2.0, 5.0, 3.0, 6.0)).hexdigest()
    empty_digest = sha256(b"").hexdigest()
    if sys.byteorder == "little":
        source = struct.pack("=6f", 1.0, -2.0, 3.0, 4.0, 5.0, 6.0)
        assert sha256(source).hexdigest() == "f96920007cbc44e611055595511408db6b56b8ceb06217b672ad160475604e6c"
        assert transposed_digest == "fce5c7856a9b4ae02cb960f4b20fbe335d437a18217005f1c7bb349451f07b4d"
    assert empty_digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    updates = []

    class ObservedDigest:
        def __init__(self):
            self.digest = sha256()

        def update(self, data):
            updates.append((type(data), memoryview(data).nbytes))
            self.digest.update(data)

        def hexdigest(self):
            return self.digest.hexdigest()

    if path == "stage-a":
        owner = stage_a_chain_seed
        assert owner.tensor_payload_sha256.__kwdefaults__["chunk_bytes"] == 64 << 20
        expected_updates = [(np.ndarray, 8)] * 3
    else:
        owner = production_weight_cache
        assert owner._TENSOR_DIGEST_CHUNK_BYTES == 8 << 20
        monkeypatch.setattr(owner, "_TENSOR_DIGEST_CHUNK_BYTES", 7)
        expected_updates = [(memoryview, 7)] * 3 + [(memoryview, 3)]
    monkeypatch.setattr(owner, "hashlib", SimpleNamespace(sha256=ObservedDigest))

    for data, shape, size, digest, feeds in (
        (tensor, [3, 2], 24, transposed_digest, expected_updates),
        (empty, [2, 0], 0, empty_digest, []),
    ):
        updates.clear()
        if path == "stage-a":
            assert owner.tensor_payload_sha256(data, chunk_bytes=9) == digest
        else:
            assert owner._cb_cache_tensor_identity(data) == {
                "shape": shape,
                "dtype": "torch.float32",
                "logical_bytes": size,
                "content_sha256": digest,
            }
        assert updates == feeds
