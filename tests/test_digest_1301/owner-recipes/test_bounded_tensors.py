"""Independent research-base oracles for three distinct bounded tensor recipes.

Only the frozen snippets supply expected results. No original consumer import,
Git subprocess, owner helper, or historical worktree is needed at test runtime.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from prismaquant import tensor_digests as owner


_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_FROZEN = json.loads(
    (_FIXTURES / "digest_1301/owner-recipes/bounded_tensors.json").read_text()
)
_RECIPES = {recipe["api"]: recipe for recipe in _FROZEN["recipes"]}
_APIS = tuple(_RECIPES)


def _original(api, **dependencies):
    recipe = _RECIPES[api]
    namespace = {
        "torch": torch,
        "hashlib": hashlib,
        "math": math,
        **_FROZEN["dependencies"],
        **dependencies,
    }
    # Compile annotations as the original modules do (postponed evaluation).
    exec(compile("from __future__ import annotations\n" + recipe["source"],
                 recipe["source_file"], "exec"), namespace)
    return namespace[recipe["original_name"]]


def _outcome(function, *args, **kwargs):
    try:
        result = function(*args, **kwargs)
    except Exception as exc:
        return ("error", type(exc), str(exc))
    # Equality alone ignores dict insertion order, an observable contract.
    return ("ok", result, tuple(result) if isinstance(result, dict) else None)


def _tensor(record):
    values = [float(v) if isinstance(v, str) else v for v in record["values"]]
    tensor = torch.tensor(values, dtype=getattr(torch, record["dtype"]))
    tensor = tensor.reshape(record["shape"])
    layout = record.get("layout")
    if layout == "transpose":
        tensor = tensor.t()
    elif layout == "strided":
        tensor = tensor[:, 1::2]
    elif layout == "expanded":
        tensor = tensor.expand(3, -1)
    elif layout == "channels_last":
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def test_frozen_original_provenance():
    registry = json.loads((_FIXTURES / "digest_parent_completion_1301.json").read_text())
    assert _FROZEN["source_commit"] == registry["source_commit"]
    assert _APIS == ("tensor_payload_sha256", "source_weight_value_identity",
                     "cb_cache_tensor_identity")
    for api, recipe in _RECIPES.items():
        assert recipe["source_commit"] == registry["source_commit"]
        calls = [call for call in registry["calls"]
                 if call["file"] == recipe["source_file"]
                 and call["scope"] == recipe["original_name"]]
        assert recipe["original_ids"] == [call["id"] for call in calls]
        assert calls and all(call["source_blob"] == recipe["source_blob"]
                             for call in calls)
        assert hashlib.sha256(recipe["source"].encode()).hexdigest() == recipe["source_sha256"]
        tree = ast.parse(recipe["source"])
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        assert tree.body[0].name == recipe["original_name"]
        assert str(inspect.signature(getattr(owner, api))) == str(inspect.signature(_original(api)))


@pytest.mark.parametrize("api", _APIS)
@pytest.mark.parametrize("record", _FROZEN["records"], ids=lambda r: r["id"])
def test_tiny_cpu_records(api, record):
    tensor = _tensor(record)
    expected = _outcome(_original(api), tensor)
    assert _outcome(getattr(owner, api), tensor) == expected
    # Most records must succeed; only the original wide-scalar view refusal is
    # expected here (payload reshapes first; value identity converts to fp32).
    if api == "cb_cache_tensor_identity" and record["id"] in {"scalar", "scalar_bf16"}:
        assert expected[0:2] == ("error", RuntimeError)
    else:
        assert expected[0] == "ok"


@pytest.mark.parametrize("chunk_bytes", [1, 3, 4, 7, 12, 1 << 26, 0, -4, 2.9, "7", None, "bad"])
def test_payload_original_chunk_argument(chunk_bytes):
    tensor = torch.arange(15, dtype=torch.float32).reshape(3, 5).t()
    assert _outcome(owner.tensor_payload_sha256, tensor, chunk_bytes=chunk_bytes) == _outcome(
        _original("tensor_payload_sha256"), tensor, chunk_bytes=chunk_bytes
    )


@pytest.mark.parametrize("value", [[1, 2, 3], 2.5, [], None, object()])
def test_source_weight_original_coercion(value):
    assert _outcome(owner.source_weight_value_identity, value) == _outcome(
        _original("source_weight_value_identity"), value
    )


@pytest.mark.parametrize("api", _APIS)
@pytest.mark.parametrize("kind", ["requires_grad", "complex", "sparse", "non_tensor", "nan_bits"])
def test_original_edge_behavior(api, kind):
    if kind == "requires_grad":
        tensor = torch.tensor([[1.0, 2.0]], requires_grad=True)
    elif kind == "complex":
        tensor = torch.tensor([1 + 2j, -3 - 4j], dtype=torch.complex64)
    elif kind == "sparse":
        tensor = torch.sparse_coo_tensor([[0], [1]], [2.0], (2, 3))
    elif kind == "nan_bits":
        tensor = torch.tensor([0x7FC00001, 0x7FC00123, -2147483648], dtype=torch.int32).view(torch.float32)
    else:
        tensor = object()
    assert _outcome(getattr(owner, api), tensor) == _outcome(_original(api), tensor)


class _Trace:
    """Virtual tensor/array/view: only dimensions and offsets, never large data."""

    def __init__(self, fail=None):
        self.events = []
        self.fail = fail
        self.torch = SimpleNamespace(uint8="uint8", float32="float32", as_tensor=self.as_tensor)
        self.hashlib = SimpleNamespace(sha256=self.sha256)

    def event(self, *event):
        self.events.append(event)
        if event[0] == self.fail:
            raise RuntimeError("injected " + event[0])

    def as_tensor(self, tensor):
        self.event("as_tensor")
        return tensor

    def sha256(self):
        self.event("sha256")
        return self

    def update(self, payload):
        if isinstance(payload, _Array):
            token = ("array", payload.tensor.shape, payload.tensor.offset)
        else:
            token = payload
        self.event("update", token)

    def hexdigest(self):
        self.event("hexdigest")
        return "trace-digest"

    def memoryview(self, array):
        self.event("memoryview")
        return _View(self, math.prod(array.tensor.shape) * array.tensor.itemsize)


class _Tensor:
    def __init__(self, trace, shape, offset=0, itemsize=4, dtype="torch.float32"):
        self.trace = trace
        self.shape = shape
        self.offset = offset
        self.itemsize = itemsize
        self.dtype = dtype

    def detach(self):
        self.trace.event("detach")
        return self

    def contiguous(self):
        self.trace.event("contiguous")
        return self

    def reshape(self, *shape):
        self.trace.event("reshape", shape)
        assert shape == (-1,)
        return _Tensor(self.trace, (math.prod(self.shape),), self.offset, self.itemsize, self.dtype)

    def element_size(self):
        self.trace.event("element_size")
        return self.itemsize

    def numel(self):
        self.trace.event("numel", self.shape)
        return math.prod(self.shape)

    @property
    def ndim(self):
        self.trace.event("ndim")
        return len(self.shape)

    @property
    def nbytes(self):
        self.trace.event("nbytes")
        return math.prod(self.shape) * self.itemsize

    def __getitem__(self, key):
        trailing = math.prod(self.shape[1:])
        if isinstance(key, slice):
            self.trace.event("slice", self.shape, self.offset, key.start, key.stop, key.step)
            start, stop, step = key.indices(self.shape[0])
            assert step == 1
            shape = (max(stop - start, 0),) + self.shape[1:]
        else:
            self.trace.event("index", self.shape, self.offset, key)
            start, shape = key, self.shape[1:]
        return _Tensor(self.trace, shape, self.offset + start * trailing, self.itemsize, self.dtype)

    def to(self, *args, **kwargs):
        self.trace.event("to", args, tuple(kwargs.items()), self.shape, self.offset)
        return self

    def view(self, dtype):
        self.trace.event("view", dtype)
        return self

    def numpy(self):
        self.trace.event("numpy")
        return _Array(self)


class _Array:
    def __init__(self, tensor):
        self.tensor = tensor

    def astype(self, dtype, *, copy):
        self.tensor.trace.event("astype", dtype, copy)
        return self

    def tobytes(self, *, order):
        self.tensor.trace.event("tobytes", order, self.tensor.shape, self.tensor.offset)
        return ("bytes", self.tensor.shape, self.tensor.offset)


class _View:
    def __init__(self, trace, size):
        self.trace, self.size = trace, size

    def cast(self, format):
        self.trace.event("cast", format)
        return self

    def __len__(self):
        self.trace.event("len_view")
        return self.size

    def __getitem__(self, key):
        self.trace.event("window", key.start, key.stop, key.step)
        return ("window", key.start, min(key.stop, self.size))


def _run_trace(monkeypatch, api, shape, *, fail=None, itemsize=4, **kwargs):
    results = []
    for original in (True, False):
        trace = _Trace(fail)
        tensor = _Tensor(trace, shape, itemsize=itemsize)
        with monkeypatch.context() as patch:
            # stage_a's original and owner both import torch inside the function.
            patch.setitem(sys.modules, "torch", trace.torch)
            if original:
                function = _original(api, torch=trace.torch, hashlib=trace.hashlib,
                                     memoryview=trace.memoryview)
            else:
                patch.setattr(owner, "torch", trace.torch)
                patch.setattr(owner, "hashlib", trace.hashlib)
                patch.setattr(owner, "memoryview", trace.memoryview, raising=False)
                function = getattr(owner, api)
            results.append((_outcome(function, tensor, **kwargs), trace.events))
    assert results[0] == results[1]
    if fail is None:
        assert results[0][0][0] == "ok"
    else:
        assert results[0][0] == ("error", RuntimeError, "injected " + fail)
    return results[1][1]


@pytest.mark.parametrize("size", [0, 1, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 25) + 3])
def test_payload_chunk_boundary_trace(monkeypatch, size):
    events = _run_trace(monkeypatch, "tensor_payload_sha256", (size,))
    feeds = [event[1] for event in events if event[0] == "update"]
    step = (1 << 26) // 4
    assert feeds == [("array", (min(step, size - start),), start)
                     for start in range(0, size, step)]
    assert events[:4] == [("detach",), ("contiguous",), ("reshape", (-1,)), ("sha256",)]
    # Each chunk transfers to CPU before the byte view; no whole host copy.
    assert sum(event[0] == "to" for event in events) == len(feeds)


@pytest.mark.parametrize("shape", [(), (0,), (4 << 20,), ((4 << 20) + 1,),
                                         (3, 1 << 21), (2, 2, (4 << 20) + 1)])
def test_source_c_order_chunk_boundary_trace(monkeypatch, shape):
    events = _run_trace(monkeypatch, "source_weight_value_identity", shape)
    feeds = [event[1] for event in events if event[0] == "update"]
    assert feeds  # Empty source weights still convert and feed one empty chunk.
    end = 0
    for kind, dimensions, offset in feeds:
        assert kind == "bytes"
        assert offset == end
        count = math.prod(dimensions)
        assert count <= 4 << 20
        end += count
    assert end == math.prod(shape)
    for i, event in enumerate(events):
        if event[0] == "to":
            assert event[1:3] == ((), (("device", "cpu"), ("dtype", "float32")))
            assert events[i + 1:i + 4] == [("contiguous",), ("numpy",), ("astype", "<f4", False)]
            assert events[i + 4][0:2] == ("tobytes", "C")


@pytest.mark.parametrize("nbytes", [0, 1, (8 << 20) - 1, 8 << 20, (8 << 20) + 1, (16 << 20) + 3])
def test_cb_memoryview_window_boundary_trace(monkeypatch, nbytes):
    events = _run_trace(monkeypatch, "cb_cache_tensor_identity", (nbytes,), itemsize=1)
    feeds = [event[1] for event in events if event[0] == "update"]
    assert feeds == [("window", start, min(start + (8 << 20), nbytes))
                     for start in range(0, nbytes, 8 << 20)]
    assert events[:3] == [("detach",), ("to", (), (("device", "cpu"),), (nbytes,), 0),
                         ("contiguous",)]
    assert ("cast", "B") in events if nbytes else not any(e[0] == "view" for e in events)
    assert not any(e[0] == "tobytes" for e in events)


@pytest.mark.parametrize("api,fail", [
    ("tensor_payload_sha256", operation)
    for operation in ("detach", "contiguous", "reshape", "sha256", "element_size", "numel", "slice", "to", "view", "numpy", "update", "hexdigest")
] + [
    ("source_weight_value_identity", operation)
    for operation in ("as_tensor", "detach", "sha256", "numel", "to", "contiguous", "numpy", "astype", "tobytes", "update", "hexdigest")
] + [
    ("cb_cache_tensor_identity", operation)
    for operation in ("detach", "to", "contiguous", "nbytes", "sha256", "view", "numpy", "memoryview", "cast", "len_view", "window", "update", "hexdigest")
])
def test_original_first_error_operation_trace(monkeypatch, api, fail):
    _run_trace(monkeypatch, api, (3,), fail=fail)
