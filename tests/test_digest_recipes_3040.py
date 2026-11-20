"""Compare each recipe in the blocked-call ledger with its source form."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from prismaquant import digests, tensor_digests
from tests.golden_table import outcome


LEDGER = json.loads((Path(__file__).parent / "fixtures/digest_recipes_3040.json").read_bytes())
JSON_CASES = [
    pytest.param({"z": ["é", "😀", -0.0], "a": {10: "ten", 9: "nine"}}, id="order-and-unicode"),
    pytest.param({"z": 1, "a": {1: "number", "1": "text"}}, id="mixed-keys"),
    pytest.param({"z": 1, "a": [float("nan"), float("inf")]}, id="nonfinite"),
    pytest.param({"z": 1, "a": complex(1, 2)}, id="fallback"),
    pytest.param({"z": 1, "a": "\ud800"}, id="surrogate"),
]


def test_ledger_maps_all_73_calls():
    groups = [*LEDGER["json"].values(), *LEDGER["sha256"].values()]
    ids = [call for group in groups for call in group["calls"]]
    assert len(ids) == len(set(ids)) == 73
    for recipe, group in LEDGER["sha256"].items():
        assert callable(globals()[f"test_sha256_{recipe}"])
        for route in group["owners"]:
            owner, _, method = route.partition(".")
            api = getattr(digests, owner, None) or getattr(tensor_digests, owner)
            assert callable(getattr(api, method) if method else api)


@pytest.mark.parametrize("value", JSON_CASES)
@pytest.mark.parametrize("route", LEDGER["json"])
def test_json_recipe_matches_source(route, value):
    profile, method = route.split(".")
    encode = getattr(getattr(digests, profile), method)
    kwargs = LEDGER["json"][route]["kwargs"].copy()
    if kwargs.get("default") == "str":
        kwargs["default"] = str
    if method == "text":
        assert outcome(lambda: encode(value).encode("utf-8")) == outcome(
            lambda: json.dumps(value, **kwargs).encode("utf-8")
        )
        return

    class Stream:
        def __init__(self):
            self.chunks = []

        def write(self, chunk):
            self.chunks.append(chunk.encode("utf-8"))

    original, current = Stream(), Stream()
    assert outcome(lambda: encode(value, current)) == outcome(
        lambda: json.dump(value, original, **kwargs)
    )
    assert current.chunks == original.chunks


def test_sha256_binary():
    payload = b"band-serial live pair head"
    assert digests.bytes_sha256digest(payload) == hashlib.sha256(payload).digest()


def test_sha256_write_interleaved():
    sink = io.BytesIO()
    block = bytearray(b"\xff" * 32)
    original = hashlib.sha256()

    def written_chunks():
        for chunk in range(3):
            block[:16] = (7).to_bytes(8, "little") + chunk.to_bytes(8, "little")
            sink.write(block)
            original.update(block)
            yield memoryview(block)
        block[:] = b"\x00" * len(block)

    actual = digests.chunks_sha256hex(written_chunks())
    assert actual == original.hexdigest() == hashlib.sha256(sink.getvalue()).hexdigest()


@pytest.mark.parametrize("tensor", [
    pytest.param(torch.arange(6, dtype=torch.int16).reshape(2, 3).T, id="strided"),
    pytest.param(torch.empty((2, 0), dtype=torch.float32), id="empty"),
    pytest.param(torch.tensor(0.5, dtype=torch.bfloat16), id="scalar"),
])
def test_sha256_tensor_leaf(tensor):
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    prefix = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
    original = hashlib.sha256(prefix)
    original.update(flat.view(torch.uint8).numpy().tobytes())
    payload = tensor_digests.tensor_payload_buffer(flat)
    if flat.numel():
        assert payload.obj.ctypes.data == flat.data_ptr()
    assert digests.chunks_sha256hex((prefix, payload)) == original.hexdigest()


def test_sha256_array_leaf():
    obj = np.arange(6, dtype=">i2").reshape(2, 3).T
    array = np.ascontiguousarray(obj)
    prefix = f"{array.dtype}|{array.shape}|".encode()
    original = hashlib.sha256(prefix)
    original.update(array.tobytes())
    assert digests.chunks_sha256hex((prefix, array.tobytes())) == original.hexdigest()


def test_sha256_metadata_tensor():
    renders = [
        (("z", "fp4"), torch.arange(6, dtype=torch.float32).reshape(2, 3).T),
        (("a", "fp4"), torch.empty((2, 0), dtype=torch.int16)),
    ]

    def records():
        for key, tensor in renders:
            file_receipt = {"bytes": tensor.nbytes, "path": "local"}
            identity = {"source": "é"}
            yield [list(key), list(tensor.shape), str(tensor.dtype),
                   {k: v for k, v in file_receipt.items() if k != "path"}, identity], tensor

    original = hashlib.sha256()
    for record, tensor in records():
        original.update(json.dumps(record, sort_keys=True).encode())
        original.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())

    def chunks():
        for record, tensor in records():
            yield digests.DIRECT_ASCII_SPACED_LAX.encoded(record)
            yield tensor_digests.tensor_payload_buffer(tensor.contiguous())

    assert digests.chunks_sha256hex(chunks()) == original.hexdigest()


def test_tensor_payload_keeps_scalar_view_refusal():
    tensor = torch.tensor(0.5)
    original = outcome(lambda: tensor.view(torch.uint8).numpy().tobytes())
    assert "raised" in original
    assert outcome(lambda: tensor_digests.tensor_payload_buffer(tensor)) == original


def test_sha256_projection_callback():
    original = hashlib.sha256()
    current = digests.IncrementalSha256()

    def consume(probe_index, terms):
        for key in sorted(terms):
            current.update(repr((probe_index, key, sorted(terms[key].items()))).encode())

    consume(3, {"z": {"q": 0.5}, "a": {"z": -0.0, "a": 1}})
    original.update(b"(3, 'a', [('a', 1), ('z', -0.0)])")
    original.update(b"(3, 'z', [('q', 0.5)])")
    assert current.hexdigest() == original.hexdigest()
    consume(7, {"a": {"q": 2}})
    original.update(b"(7, 'a', [('q', 2)])")
    assert current.hexdigest() == original.hexdigest()
