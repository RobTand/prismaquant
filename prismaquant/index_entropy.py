"""Index-stream entropy for NVFP4-CB encodings (Phase-0 experiment 2).

Decides whether entropy coding a fixed-rate ``k``-bit VQ index stream could
ever pay: how far below ``k`` is the empirical entropy ``H(indices)``, and is
there exploitable first-order serial correlation ``H(idx_t | idx_{t-1})``.

Sparse counting (``torch.unique`` / ``bincount`` over observed indices only)
so ``k`` up to 24 (2^24 symbols) never allocates a ``2^k`` histogram.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import torch


def _empirical_entropy_bits(counts: torch.Tensor) -> float:
    """Shannon entropy (bits) of a symbol distribution given integer counts."""
    counts = counts[counts > 0].to(torch.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(-(p * (p.log() / math.log(2.0))).sum().item())


def index_entropy(indices: torch.Tensor, k: int) -> dict:
    """Empirical entropy / redundancy of a VQ index stream.

    Args:
      indices: integer index tensor (any shape); flattened in row-major order,
        which for a VQ-encoded weight is the natural on-disk vector order.
      k: index width in bits (codebook has ``2^k`` entries).

    Returns dict with:
      ``H`` empirical entropy (bits),
      ``redundancy`` = ``k - H`` (bpw recoverable by ideal entropy coding),
      ``H_conditional`` first-order ``H(idx_t | idx_{t-1})`` (bits),
      ``conditional_gain`` = ``H - H_conditional`` (serial-correlation bpw),
      plus ``k``, ``n``, ``n_unique``.
    """
    flat = indices.reshape(-1).to(torch.long)
    n = int(flat.numel())
    if n == 0:
        return {
            "k": int(k), "n": 0, "n_unique": 0,
            "H": 0.0, "redundancy": float(k),
            "H_conditional": 0.0, "conditional_gain": 0.0,
        }

    # Marginal H via sparse counts over observed symbols only.
    uniq, counts = torch.unique(flat, return_counts=True)
    h_marginal = _empirical_entropy_bits(counts)

    # First-order conditional H(cur | prev) = H(prev, cur) - H(prev), computed
    # over consecutive pairs. Encode a pair as prev * n_unique + cur on the
    # densified symbol ids so the joint histogram is bounded by observed
    # symbols, never 2^k * 2^k.
    if n >= 2:
        n_uniq = int(uniq.numel())
        # Map raw symbols -> dense 0..n_uniq-1 via searchsorted on sorted uniq.
        dense = torch.searchsorted(uniq, flat)
        prev = dense[:-1]
        cur = dense[1:]
        pair_id = prev.to(torch.long) * n_uniq + cur.to(torch.long)
        _, pair_counts = torch.unique(pair_id, return_counts=True)
        h_joint = _empirical_entropy_bits(pair_counts)
        _, prev_counts = torch.unique(prev, return_counts=True)
        h_prev = _empirical_entropy_bits(prev_counts)
        h_conditional = max(h_joint - h_prev, 0.0)
    else:
        h_conditional = h_marginal

    return {
        "k": int(k),
        "n": n,
        "n_unique": int(uniq.numel()),
        "H": h_marginal,
        "redundancy": float(k) - h_marginal,
        "H_conditional": h_conditional,
        "conditional_gain": max(h_marginal - h_conditional, 0.0),
    }


def _extract_indices(payload) -> tuple[torch.Tensor, int]:
    """Pull an index tensor + k from a saved fields dict.

    Accepts a fields dict with an ``indices`` (or ``idx``) tensor and an
    optional ``k``; falls back to inferring ``k`` from the max index.
    """
    if isinstance(payload, torch.Tensor):
        idx = payload
        k = None
    elif isinstance(payload, dict):
        idx = None
        for key in ("indices", "idx", "index", "codes"):
            if key in payload:
                idx = payload[key]
                break
        if idx is None:
            raise KeyError(
                "index_entropy: no 'indices'/'idx' field in payload; "
                f"keys={sorted(payload)}")
        k = payload.get("k")
    else:
        raise TypeError(f"Unsupported payload type {type(payload)!r}")

    if not isinstance(idx, torch.Tensor):
        idx = torch.as_tensor(idx)
    if k is None:
        maxv = int(idx.reshape(-1).max().item()) if idx.numel() else 0
        k = max(1, (maxv).bit_length())
    return idx, int(k)


def _load_payload(path: Path):
    if path.suffix in (".pt", ".pth"):
        return torch.load(path, map_location="cpu", weights_only=False)
    with open(path, "rb") as f:
        return pickle.load(f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Empirical entropy of an NVFP4-CB index stream.")
    ap.add_argument("fields", help="saved fields dict (.pt/.pkl) or tensor")
    ap.add_argument("--k", type=int, default=None,
                    help="override index width in bits")
    ap.add_argument("--output", default=None, help="write JSON result here")
    args = ap.parse_args(argv)

    payload = _load_payload(Path(args.fields))
    idx, k = _extract_indices(payload)
    if args.k is not None:
        k = int(args.k)
    result = index_entropy(idx, k)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
