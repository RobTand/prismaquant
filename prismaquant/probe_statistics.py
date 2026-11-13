"""Activation-row selection shared by the probe."""
from __future__ import annotations

import torch

_U16_MASK = (1 << 16) - 1
_U32_MASK = (1 << 32) - 1
_FMIX32_C1 = 0x85EBCA6B
_FMIX32_C2 = 0xC2B2AE35

# The rule that selects the act/ rows. The key is fixed: the first eight
# BLAKE2b bytes of this name, read as little-endian uint32 k0, k1
# (tests/test_body_fisher_sweep.py derives them again).
ACT_ROW_SELECTION = "blake2b64-fixed-key-fmix32x2-prp-global-row-v1"
_ACT_ROW_KEY = (0x497DDF44, 0xCCAD2C68)


def _mullo32_tensor(values: torch.Tensor, constant: int) -> torch.Tensor:
    """Vectorized low uint32 product with all int64 intermediates < 2**49."""
    low = torch.bitwise_and(values, _U16_MASK)
    high = torch.bitwise_right_shift(values, 16)
    return torch.bitwise_and(
        low * int(constant)
        + torch.bitwise_left_shift(high * int(constant & _U16_MASK), 16),
        _U32_MASK,
    )


def fmix32_tensor(values: torch.Tensor) -> torch.Tensor:
    """Murmur3 fmix32 on int64 tensors: a permutation of the uint32 domain."""
    x = torch.bitwise_and(values, _U32_MASK)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    x = _mullo32_tensor(x, _FMIX32_C1)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 13))
    x = _mullo32_tensor(x, _FMIX32_C2)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    return torch.bitwise_and(x, _U32_MASK)


def keep_act_rows(snaps: dict, indices: dict, name: str, flat: torch.Tensor,
                  rows: torch.Tensor, limit: int, *, to_cpu: bool = False) -> None:
    """Keep the ``limit`` act/ rows of ``name`` with the lowest priority.

    ``rows`` are the global row numbers (sample * seqlen + token) of
    ``flat``. The priority is a keyed permutation of the row number, so it
    reads no RNG state, gives the same result on each device, and has no
    ties. Thus the kept set, sorted by row, is the same for each split of
    the rows into hook calls or calibration slices (a top-K merge).
    """
    if snaps.get(name):
        flat = torch.cat([snaps[name][0].to(flat.device), flat])
        rows = torch.cat([indices[name][0].to(rows.device), rows])
    if flat.size(0) > limit:
        k0, k1 = _ACT_ROW_KEY
        priority = fmix32_tensor(
            torch.bitwise_xor(fmix32_tensor(torch.bitwise_xor(rows, k0)), k1))
        pick = torch.topk(priority, int(limit), largest=False).indices
        flat, rows = flat.index_select(0, pick), rows[pick]
    rows, order = torch.sort(rows)
    flat = flat.index_select(0, order)
    snaps[name] = [flat.to("cpu") if to_cpu else flat]
    indices[name] = [rows.to("cpu") if to_cpu else rows]
