"""Pure-torch decode of one EXL3 (exllamav3 0.0.43, MCG codebook, K=4) routed Linear.

Restates the published reader, without its compiled extension:
  * inner codes: exllamav3_ext quant/reconstruct.cu ``reconstruct_kernel<4, cb=1>`` with
    exl3_dq.cuh ``dq8_aligned_4bits`` and codebook.cuh ``decode_3inst<1>``:
      - each 16x16 tile is a circular MSB-first bitstream of 256 four-bit symbols, stored as
        64 int16 read as 32 little-endian uint32 words;
      - position t's 16-bit state is symbols t-3, t-2, t-1, t (mod 256), most significant first;
      - value(t) = fp16(lo) + fp16(hi) (correctly rounded) of
        x = ((state * MCG) mod 2^32 & 0x8fff8fff) ^ 0x3b603b60;
      - position t = 8*lane + j lands at (row, col) of the tile by the kernel's shuffle.
  * effective weight: ``LinearEXL3.get_weight_tensor`` = diag(suh) H w_inner H diag(svh) with
    H the natural-order Sylvester Hadamard of order 128 over each 128-block, normalised by
    1/sqrt(128) per side.  exllamav3 rounds to fp16 after every step; this decode keeps fp32
    throughout and rounds once, to bf16, so it is the stored operator's value rather than any
    one kernel's numerical result. The decoder preserves the accepted operation order.
    CPU parity does not establish native GPU equality.

Returns the HF-orientation [out, in] weight: the stored operator, which for routed experts is
P.Wg, P.Wu, Wd.P^T (an intermediate-channel permutation that preserves each expert's function).
"""
from __future__ import annotations

import torch

K_BITS = 4
TILE = 16
HAD = 128
MCG_KERNEL = 0xCBAC1FED   # codebook.cuh decode_3inst<1>: the multiplier the kernel applies


def _tile_positions():
    """(row, col) inside the 16x16 tile of trellis position t = 8*lane + j (reconstruct.cu)."""
    rc = [None] * 256
    for lane in range(32):
        if lane & 4:
            continue
        r0 = (lane % 4) * 2
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        c0 = lane // 8
        for src, col_off in ((lane, 0), (lane + 4, 1)):
            for j in range(8):
                r = rows[j % 4]
                c = (c0 if j < 4 else c0 + 4) * 2 + col_off
                rc[8 * src + j] = (r, c)
    assert all(v is not None for v in rc) and len(set(rc)) == 256
    return rc


_RC = _tile_positions()
#: flat index r*16+c for each trellis position t
TILE_PERM = torch.tensor([r * TILE + c for r, c in _RC], dtype=torch.long)
#: inverse: trellis position for each flat (r, c)
TILE_INV = torch.empty(256, dtype=torch.long)
TILE_INV[TILE_PERM] = torch.arange(256)


def inner_codes(trellis: torch.Tensor, mcg: int = MCG_KERNEL) -> torch.Tensor:
    """[kt, nt, 64] int16 trellis -> [16*kt, 16*nt] fp16 inner weight (reconstruct kernel)."""
    if trellis.dtype != torch.int16 or trellis.dim() != 3 or trellis.shape[-1] != 256 * K_BITS // 16:
        raise ValueError(f"trellis must be int16 [kt, nt, 64], got {trellis.dtype} {tuple(trellis.shape)}")
    kt, nt, _ = trellis.shape
    dev = trellis.device
    u16 = trellis.to(torch.int64) & 0xFFFF
    u32 = u16[..., 0::2] | (u16[..., 1::2] << 16)                       # [kt, nt, 32]
    shifts = torch.arange(28, -1, -4, device=dev, dtype=torch.int64)      # 28, 24, ..., 0
    nib = ((u32[..., :, None] >> shifts) & 0xF).reshape(kt, nt, 256)       # MSB-first symbols
    state = ((torch.roll(nib, 3, dims=-1) << 12) | (torch.roll(nib, 2, dims=-1) << 8)
             | (torch.roll(nib, 1, dims=-1) << 4) | nib)
    x = (state * (mcg & 0xFFFFFFFF)) & 0xFFFFFFFF
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    lo = (x & 0xFFFF).to(torch.int32).to(torch.int16)       # low 16 bits, two's complement
    hi = ((x >> 16) & 0xFFFF).to(torch.int32).to(torch.int16)
    val = (lo.view(torch.float16).float() + hi.view(torch.float16).float()).half()   # __hadd
    tile = val.index_select(-1, TILE_INV.to(dev)).reshape(kt, nt, TILE, TILE)
    return tile.permute(0, 2, 1, 3).reshape(kt * TILE, nt * TILE)


def fwht_rows(x: torch.Tensor) -> torch.Tensor:
    """Unnormalised natural-order Sylvester Hadamard over dim 0 in blocks of 128 (fp32)."""
    n, m = x.shape
    if n % HAD:
        raise ValueError(f"dim 0 ({n}) is not a multiple of {HAD}")
    y = x.reshape(n // HAD, HAD, m)
    h = 1
    while h < HAD:
        y = y.reshape(n // HAD, HAD // (2 * h), 2, h, m)
        a, b = y[:, :, 0], y[:, :, 1]
        y = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return y.reshape(n, m)


def effective_weight(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor,
                     mcg: int = MCG_KERNEL) -> torch.Tensor:
    """HF-orientation [out, in] fp32 weight of one stored EXL3 Linear."""
    w = inner_codes(trellis, mcg).float()                  # [in, out]
    k, n = w.shape
    if suh.numel() != k or svh.numel() != n:
        raise ValueError(f"suh {suh.numel()} / svh {svh.numel()} do not match inner {k}x{n}")
    w = fwht_rows(w)                                       # H . w   (unnormalised)
    w = w * suh.float().reshape(k, 1)
    w = fwht_rows(w.t().contiguous()).t()                  # (. H)   (unnormalised)
    w = w * svh.float().reshape(1, n)
    w = w * (1.0 / HAD)                                    # 1/sqrt(128) per side: exact power of two
    return w.t().contiguous()


# ------------------------------------------------------------------ G3 wire framing
# One EXL3 unit's wire, as the G3 manifest frames it: suh (fp16, in) || svh (fp16, out) ||
# trellis (int16) || mcg (int32), concatenated in that order from the published shards.
def pack_wire(suh: bytes, svh: bytes, trellis: bytes, mcg: bytes) -> bytes:
    return bytes(suh) + bytes(svh) + bytes(trellis) + bytes(mcg)


def decode_wire(blob: bytes, in_features: int, out_features: int, device="cpu",
                expect_mcg: int | None = None) -> torch.Tensor:
    """Decode a framed wire to the [out, in] weight (fp32).  Refuses a wire of the wrong length."""
    n_suh, n_svh = 2 * in_features, 2 * out_features
    n_tr = 2 * in_features * out_features * K_BITS // 16
    if len(blob) != n_suh + n_svh + n_tr + 4:
        raise ValueError(f"EXL3 wire has {len(blob)} bytes, framing needs {n_suh + n_svh + n_tr + 4}")
    buf = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
    suh = buf[:n_suh].view(torch.float16)
    svh = buf[n_suh:n_suh + n_svh].view(torch.float16)
    tr = buf[n_suh + n_svh:n_suh + n_svh + n_tr].view(torch.int16).reshape(
        in_features // TILE, out_features // TILE, 256 * K_BITS // 16)
    stored_mcg = int(buf[-4:].view(torch.int32)[0]) & 0xFFFFFFFF
    if expect_mcg is not None and stored_mcg != expect_mcg:
        raise ValueError(f"stored mcg 0x{stored_mcg:08X} != expected 0x{expect_mcg:08X}")
    return effective_weight(tr.to(device), suh.to(device), svh.to(device))
