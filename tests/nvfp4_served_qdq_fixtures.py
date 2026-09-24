"""Shared inputs for the served NVFP4 activation leg's tests (RobTand/prismaquant#1211).

A helper module, not a test module: ``tests/test_nvfp4_served_qdq_no_sync.py``
and ``tests/test_nvfp4_served_dequant_kernel.py`` import it.
"""
from __future__ import annotations

import math
import types

import torch

from prismaquant import nvfp4_activation_contract as owner

G16 = owner.FP4_GROUP_SIZE


class FakeCudaActivation:
    """A tensor stand-in whose ``device`` says CUDA; everything else is CPU."""

    def __init__(self, shape, dtype=torch.bfloat16):
        self.device = torch.device("cuda")
        self.dtype = dtype
        self.shape = tuple(shape)

    def numel(self) -> int:
        return math.prod(self.shape)

    def reshape(self, *shape):
        return torch.zeros(self.shape, dtype=self.dtype).reshape(*shape)


def cpu_kernels():
    """CPU stand-ins with the fused kernels' contracts, for dispatch-only tests."""

    def group_abs_max(rows):
        return rows.float().reshape(rows.shape[0], -1, G16).abs().amax(-1)

    def dequantize_codes(packed, stored, used, table, dtype):
        return torch.zeros(packed.shape[0], packed.shape[1] * 2, dtype=dtype)

    return types.SimpleNamespace(group_abs_max=group_abs_max,
                                 dequantize_codes=dequantize_codes)


def activation(rows: int, width: int, *, seed: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Replay-like CUDA rows: a heavy-tailed body, spikes, zero groups, a zero row."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, width, device="cuda", generator=generator)
    x = x * torch.exp(torch.randn(rows, 1, device="cuda", generator=generator))
    spikes = torch.rand(rows, width, device="cuda", generator=generator) < 0.002
    x = torch.where(spikes, x * 40.0, x)
    if width >= 2 * G16:
        x[:, G16:2 * G16] = 0.0            # an all-zero group in every row
    if rows > 3:
        x[3] = 0.0                         # an all-zero row
    return x.to(dtype)


def stand_in_operator():
    """A ``torch.ops`` stand-in whose ``_C.scaled_fp4_quant`` is PrismaQuant's
    own E2M1 code rule, packed the way the operator packs (low nibble first).

    It reads G from the device scalar it is handed and never from the host, so
    it makes no synchronising call of its own.
    """

    def _op(rows, global_scale, swizzled):
        assert swizzled is True and global_scale.numel() == 1
        g = global_scale.reshape(())
        grouped = rows.float().reshape(rows.shape[0], -1, G16)
        stored = owner.nvfp4_group_stored_scale(grouped, g).float()
        used = stored / g
        safe = torch.where(stored != 0, used, torch.ones_like(used))
        codes = owner.nvfp4_e2m1_code(owner.nvfp4_e2m1_normalize(grouped, safe))
        codes = codes.reshape(rows.shape[0], rows.shape[1] // 2, 2).to(torch.uint8)
        return codes[..., 0] | (codes[..., 1] << 4), torch.empty(0, device=rows.device)

    return types.SimpleNamespace(_C=types.SimpleNamespace(scaled_fp4_quant=_op))


def real_operator_registered() -> bool:
    """Whether this process has the vLLM extension's ``scaled_fp4_quant`` on CUDA."""
    if not torch.cuda.is_available():
        return False
    try:
        return bool(owner._register_served_quantizer_op())
    except Exception:
        return False
