"""The served NVFP4 activation leg makes no per-call host sync (RobTand/prismaquant#1211).

``_nvfp4_activation_qdq_registered_op`` handed ``torch.ops._C.scaled_fp4_quant``
its global scale as ``torch.tensor([g], device=rows.device)``, built on every
call: a pageable host-to-device copy, which blocks the host until the stream
has drained.  On GLM-5.3 row 041 (PB ``4e1468a6d07e``) that one line held 498
of the 617 main-thread py-spy samples the leg took over the render windows.

The operand is now one device scalar per ``(device, G)``, built by the same
expression, so the value is the same float32 and only the first call at a G
pays the copy.  Held here two ways: the operator sees the SAME tensor on every
call at one G (CPU, a recording stand-in for the operator), and after the first
call at a G the whole leg runs under ``torch.cuda.set_sync_debug_mode("error")``
(CUDA, with a non-synchronising stand-in operator, and with the registered
operator where the vLLM extension is installed).
"""
from __future__ import annotations

import types

import pytest
import torch

from nvfp4_served_qdq_fixtures import (
    FakeCudaActivation, activation, cpu_kernels, real_operator_registered,
    stand_in_operator)
from prismaquant import nvfp4_activation_contract as owner


def test_the_operator_reads_g_from_one_cached_device_scalar(monkeypatch):
    """The operator's scale operand is the SAME tensor on every call at one G,
    holding the float32 the per-call expression built; another G gets its own."""
    seen: list[torch.Tensor] = []

    def _op(rows, global_scale, swizzled):
        seen.append(global_scale)
        return (torch.zeros(rows.shape[0], rows.shape[1] // 2, dtype=torch.uint8),
                torch.zeros(0, dtype=torch.float8_e4m3fn))

    monkeypatch.setattr(torch, "ops", types.SimpleNamespace(
        _C=types.SimpleNamespace(scaled_fp4_quant=_op)))
    # ``raising=False``: a tree without the fused kernels has nothing to stub,
    # and the test must then fail on its assertion, not on the stub.
    monkeypatch.setattr(owner, "_served_dequant_kernels", cpu_kernels, raising=False)
    monkeypatch.setattr(owner, "_SERVED_GLOBAL_SCALES", {}, raising=False)

    g = 3.5720930099487305
    for _ in range(3):
        owner._nvfp4_activation_qdq_registered_op(FakeCudaActivation((4, 64)), g)
    owner._nvfp4_activation_qdq_registered_op(FakeCudaActivation((4, 64)), 2.0)

    assert len(seen) == 4
    assert seen[0] is seen[1] is seen[2], "the G operand is rebuilt on every call"
    assert seen[3] is not seen[0]
    assert torch.equal(seen[0], torch.tensor([g], dtype=torch.float32))
    assert torch.equal(seen[3], torch.tensor([2.0], dtype=torch.float32))


def _assert_no_sync_after_first_call(x, g):
    owner._nvfp4_activation_qdq_registered_op(x, g)   # builds the scalar, compiles
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            owner._nvfp4_activation_qdq_registered_op(x, g)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_leg_makes_no_synchronising_call_after_its_first_at_a_g(monkeypatch):
    monkeypatch.setattr(torch, "ops", stand_in_operator())
    _assert_no_sync_after_first_call(activation(129, 4096, seed=9), 448.0 * 6.0 / 11.375)


@pytest.mark.skipif(not real_operator_registered(),
                    reason="needs the vLLM extension's registered scaled_fp4_quant")
def test_the_registered_operator_leg_makes_no_synchronising_call_after_its_first_at_a_g():
    _assert_no_sync_after_first_call(activation(129, 4096, seed=13), 448.0 * 6.0 / 11.375)
