#!/usr/bin/env python3
"""Ceiling estimates for the two remaining CB-encode candidates.

(C) Batch the two-tier E window into one scoring pass: does raising S from 16
    to W*16 amortize the (m, K) B-plane read, or does register pressure eat it?
(D) Fuse the _vq_assign epilogue (term2 - 2*term1 -> argmin) so the (m, K)
    distance plane is never materialized.

Reports throughput per unit of scored work so the two S values are comparable.
"""
from __future__ import annotations

import time

import torch

from prismaquant import nvfp4_cb_formats as F


def bench(fn, iters=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def cand_c(m, K, P, S_list):
    dev = "cuda"
    A = torch.randn(P, K, device=dev).abs()
    B = torch.randn(m, K, device=dev)
    print(f"\n[C] batched scan  m={m} K={K} P={P}")
    base = None
    for S in S_list:
        s = torch.rand(m, S, device=dev) + 0.5
        t = bench(lambda: F._score_min_batched(A, B, s))
        per_unit = t / S
        if base is None:
            base = per_unit
        print(f"   S={S:4d}  {t * 1e3:8.2f} ms   per-16-scales "
              f"{per_unit * 16 * 1e3:7.2f} ms   speedup_vs_S16 "
              f"{base / per_unit:5.2f}x")
        del s


def cand_d(m, K, d, P):
    dev = "cuda"
    x = torch.randn(m, d, device=dev)
    wq = torch.rand(P, d, device=dev).repeat(m // P, 1).contiguous()
    cb = torch.randn(K, d, device=dev)
    cb_t = cb.t().contiguous()
    cb_sq_t = (cb * cb).t().contiguous()
    term2 = wq[:P] @ cb_sq_t
    chunk = max(1, F._SCORE_CHUNK_ELEMS // K)
    r = min(chunk, m) // P

    def cur():
        term1 = (wq[:chunk] * x[:chunk]) @ cb_t
        d_ = term2.reshape(1, P, K) - 2.0 * term1.reshape(r, P, K)
        return d_.argmin(dim=-1).reshape(-1)

    def fused_eager(t2, t1):
        return (t2 - 2.0 * t1).argmin(dim=-1)

    comp = torch.compile(fused_eager, dynamic=True)

    def new():
        term1 = (wq[:chunk] * x[:chunk]) @ cb_t
        return comp(term2.reshape(1, P, K),
                    term1.reshape(r, P, K)).reshape(-1)

    a = bench(lambda: cur(), iters=5)
    ref = cur()
    try:
        b = bench(lambda: new(), iters=5)
        got = new()
        same = torch.equal(ref, got)
    except Exception as e:  # noqa: BLE001
        print(f"\n[D] compile failed: {e}")
        return
    print(f"\n[D] vq epilogue  m={min(chunk, m)} K={K} d={d}")
    print(f"   materialized {a * 1e3:7.2f} ms   fused {b * 1e3:7.2f} ms   "
          f"{a / b:5.2f}x   identical={same}")


if __name__ == "__main__":
    F._raise_encode_recompile_limit()
    # K14 gate/up row-chunk: m = 1024 rows * 512 vec/row, K=128, P=512.
    cand_c(1024 * 512, 128, 512, [16, 32, 64, 144])
    # K15 stream 0.
    cand_c(512 * 512, 256, 512, [16, 32, 64, 144])
    cand_d(1024 * 512, 128, 4, 512)
    cand_d(1024 * 512, 256, 4, 512)
