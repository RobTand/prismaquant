"""An empty NVFP4 activation batch is allocated, never launched.

WHAT THIS HOLDS.  ``torch.ops._C.scaled_fp4_quant`` -- the operator the served
static-activation contract prices against -- derives its launch grid from the
token count.  At zero rows that is an empty grid, and the failure is not a
return value: the call returns, ``torch.cuda.synchronize()`` reports nothing,
and the next CHECKED launch anywhere in the process raises ``CUDA error:
invalid argument``, because the error is sticky in the context.  Measured on
the pinned image for the same registered operator (Tessera's binding, 2026-09-17
``fp4-red-20260917T041842Z-4a35878a10dd``: M = 0 poisons the process, M = 1 and
M = 8 are clean, the FP8 control at M = 0 is clean).

The dispatch decision is what a CPU test can hold exactly: an empty activation
must not reach the operator, and a real one must still reach it once.  That the
operator's own codes are the served codes is a device question and is measured
on the retained real-wire inputs, not here.

STUBBED: the registered operator (a recorder) and the CUDA device (a tensor
stand-in whose ``device`` says ``cuda``), because a CPU box cannot allocate on
the device the operator requires.  The tensors this leg hands BACK are ordinary
CPU ones on purpose: what is priced here is which call the leg makes.
"""
from __future__ import annotations

import types

import torch

from prismaquant import nvfp4_activation_contract as owner


class _FakeCudaActivation:
    """The surface the registered-op leg reads before it launches.

    ``device`` is the only CUDA-shaped thing about it; ``reshape`` /
    ``contiguous`` / ``to`` hand back real CPU tensors so the leg's own
    arithmetic (which is device-free) can run, and ``new_empty`` records what
    the empty path allocated.
    """

    def __init__(self, shape, dtype=torch.bfloat16):
        self.device = torch.device("cuda")
        self.dtype = dtype
        self._shape = tuple(shape)
        self.allocations: list[tuple] = []

    @property
    def shape(self) -> tuple:
        return self._shape

    def numel(self) -> int:
        count = 1
        for dimension in self._shape:
            count *= int(dimension)
        return count

    def new_empty(self, shape):
        self.allocations.append(tuple(shape))
        return torch.empty(tuple(shape), dtype=self.dtype)

    def reshape(self, *shape):
        dims = [int(dimension) for dimension in shape]
        if -1 in dims:
            known = 1
            for dimension in dims:
                if dimension != -1:
                    known *= dimension
            if known == 0:
                raise RuntimeError("cannot resolve an unknown dimension here")
            dims[dims.index(-1)] = self.numel() // known
        return torch.empty(tuple(dims), dtype=self.dtype)

    def contiguous(self):
        return self

    def to(self, dtype):
        return torch.empty(self._shape, dtype=dtype)


def _op_spy(monkeypatch, rows: int, columns: int):
    """Replace the registered operator with a recorder for one test."""
    calls: list[dict] = []

    def _record(x, global_scale, swizzled):
        calls.append({"shape": tuple(x.shape), "swizzled": swizzled,
                      "global_scale_numel": int(global_scale.numel())})
        # The packed nibble plane the real op returns for [rows, columns]:
        # two codes per byte, one flat uint8 tensor.
        return (torch.zeros(rows * (columns // 2), dtype=torch.uint8),
                torch.zeros(0, dtype=torch.float8_e4m3fn))

    monkeypatch.setattr(torch, "ops", types.SimpleNamespace(
        _C=types.SimpleNamespace(scaled_fp4_quant=_record)))
    return calls


def _empty_call(x, g=1.0):
    """``(out, failure)`` -- the dispatch is the property under test here.

    What the operator DOES with an empty batch is not this file's question (and
    on the unfixed source the answer is a poisoned context, not a value), so a
    downstream refusal is caught and reported rather than allowed to hide the
    dispatch assertion.
    """
    try:
        return owner._nvfp4_activation_qdq_registered_op(x, g), None
    except Exception as exc:  # noqa: BLE001 -- reported by the caller
        return None, exc


def test_an_empty_activation_batch_is_allocated_and_never_launched(monkeypatch):
    """M = 0 is an allocation: the operator is not called at all."""
    calls = _op_spy(monkeypatch, rows=0, columns=1024)
    x = _FakeCudaActivation((0, 1024))

    out, failure = _empty_call(x)

    assert calls == [], "a zero-row activation must not launch the FP4 quantiser"
    assert out is not None, f"the empty path raised {failure!r}"
    assert x.allocations == [(0, 1024)]
    assert tuple(out.shape) == (0, 1024) and out.dtype == torch.bfloat16


def test_an_empty_last_dim_is_allocated_too(monkeypatch):
    """No elements is no launch, whichever axis is zero."""
    calls = _op_spy(monkeypatch, rows=2, columns=0)
    x = _FakeCudaActivation((2, 0))

    out, failure = _empty_call(x)

    assert calls == []
    assert out is not None, f"the empty path raised {failure!r}"
    assert tuple(out.shape) == (2, 0)


def test_the_empty_path_still_refuses_what_the_guards_refuse(monkeypatch):
    """The guards run FIRST: an empty tensor is not a licence to skip them."""
    calls = _op_spy(monkeypatch, rows=0, columns=1000)
    with torch.no_grad():
        try:
            owner._nvfp4_activation_qdq_registered_op(_FakeCudaActivation((0, 1000)), 1.0)
        except ValueError as refusal:
            assert "divisible by" in str(refusal)
        else:
            raise AssertionError("an empty activation with a bad last dim was accepted")
        try:
            owner._nvfp4_activation_qdq_registered_op(_FakeCudaActivation((0, 1024)), 0.0)
        except ValueError as refusal:
            assert "finite and > 0" in str(refusal)
        else:
            raise AssertionError("an empty activation with G = 0 was accepted")
    assert calls == []


def _cpu_dequant_kernels(monkeypatch):
    """The fused kernels (#1211) are CUDA-only; this file's tensors are CPU.

    What is priced here is which call the leg makes, so the kernels are
    replaced by CPU functions with their contracts; their arithmetic is held
    bit-identical to the Torch composition in
    ``tests/test_nvfp4_served_dequant_kernel.py``.
    """
    group = owner.FP4_GROUP_SIZE
    monkeypatch.setattr(owner, "_served_dequant_kernels", lambda: types.SimpleNamespace(
        group_abs_max=lambda rows: rows.float().reshape(
            rows.shape[0], -1, group).abs().amax(-1),
        dequantize_codes=lambda packed, stored, used, table, dtype: torch.zeros(
            packed.shape[0], packed.shape[1] * 2, dtype=dtype)))


def test_a_real_batch_still_launches_the_operator_once(monkeypatch):
    """The guard is a short-circuit, not a replacement for the operator."""
    calls = _op_spy(monkeypatch, rows=4, columns=1024)
    _cpu_dequant_kernels(monkeypatch)

    out = owner._nvfp4_activation_qdq_registered_op(_FakeCudaActivation((4, 1024)), 1.0)

    assert [c["shape"] for c in calls] == [(4, 1024)]
    assert calls[0]["swizzled"] is True and calls[0]["global_scale_numel"] == 1
    assert tuple(out.shape) == (4, 1024) and out.dtype == torch.bfloat16
