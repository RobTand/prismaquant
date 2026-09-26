#!/usr/bin/env python3
"""Byte-level encode identity check for the CB cost render.

Encodes real layer-0 Linears under the production context and emits, per
format, an exact fingerprint of every field the artifact carries (indices,
scales, two-tier super/sub codes) plus the dequantized W_hat and weight_mse.

Run it against two checkouts and diff the JSON: identical fingerprints ==
byte-identical encodings. ``--vs`` additionally loads a saved baseline blob
and asserts ``torch.equal`` on every tensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "cb_encode_replica",
    str(Path(__file__).resolve().parent / "cb_encode_replica.py"))
R = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(R)

# (expert, proj) covering both production expert shapes + an attention Linear.
DEFAULT_TARGETS = "0:gate_proj,0:down_proj,7:up_proj,131:gate_proj"


def _sha(t: torch.Tensor) -> str:
    x = t.detach().cpu().contiguous()
    return hashlib.sha256(
        x.numpy().tobytes() if x.dtype != torch.bfloat16
        else x.view(torch.uint8).numpy().tobytes()).hexdigest()[:32]


def fields_for(w, cw, fmt, device="cuda"):
    """Mirror of cb_quantize_dequantize_for_context's nvfp4_cb_fields call."""
    from prismaquant.nvfp4_cb_footprint import (
        cb_serialization_context_from_env, _cb_info,
    )
    from prismaquant.nvfp4_cb_formats import (
        SCALE_CODING_V1, nvfp4_cb_fields,
    )

    ctx = cb_serialization_context_from_env(require_explicit=True,
                                            where="bitcheck")
    grid, mode, k = _cb_info(fmt)
    dev = torch.device(device)
    ws = w.to(device=dev, dtype=torch.bfloat16)
    cws = cw.to(device=dev)
    coding = ctx.scale_coding if grid == "fp4" else SCALE_CODING_V1
    return nvfp4_cb_fields(
        ws, k, grid=grid, mode=mode, col_weights=cws, codebook=None,
        scale_sweep=ctx.scale_sweep, scale_coding=coding,
        encode_tier=ctx.encode_tier)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--formats", default="NVFP4_CB_K14,NVFP4_CB_K15,FP8_CB_K36")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-tensors", default="")
    ap.add_argument("--vs", default="")
    args = ap.parse_args()

    R.apply_prod_env()
    report = {}
    blobs = {}
    base = torch.load(args.vs, map_location="cpu") if args.vs else None
    mismatches = []

    for tgt in args.targets.split(","):
        e, proj = tgt.split(":")
        w = R.load_expert_weight(args.layer, int(e), proj)
        cw = R.load_col_weights(args.layer, int(e), proj)
        srow = R.shard_row(args.layer, int(e), proj)
        for fmt in args.formats.split(","):
            key = f"{tgt}|{fmt}"
            w_hat, mse, dt = R.encode_one(w, cw, fmt, device=args.device)
            f = fields_for(w, cw, fmt, device=args.device)
            # Production computes weight_mse as err.pow(2).mean(dim=(1,2))
            # on the STACKED (N, out, in) chunk tensor; a 2-D mean has a
            # different summation tree and can land 1 ULP away even when the
            # encoding is byte-identical. Report both.
            ws_dev = w.to(device=w_hat.device, dtype=torch.bfloat16)
            mse_stacked = float(
                (ws_dev.unsqueeze(0) - w_hat.unsqueeze(0)).float()
                .pow(2).mean(dim=(1, 2))[0])
            sref = srow[fmt]["weight_mse"]
            ent = {"weight_mse": mse,
                   "weight_mse_stacked": mse_stacked,
                   "shard_weight_mse": sref,
                   "shard_exact": mse == sref,
                   "shard_exact_stacked": mse_stacked == sref,
                   "secs": dt,
                   "w_hat_sha": _sha(w_hat)}
            tens = {"w_hat": w_hat.cpu()}
            for fk in ("indices", "scales", "scale_super", "scale_sub",
                       "signs"):
                if fk in f:
                    ent[f"{fk}_sha"] = _sha(f[fk])
                    ent[f"{fk}_shape"] = tuple(f[fk].shape)
                    tens[fk] = f[fk].cpu()
            report[key] = ent
            blobs[key] = tens
            if base is not None and key in base:
                for tk, tv in tens.items():
                    bv = base[key].get(tk)
                    if bv is None or not torch.equal(bv, tv):
                        mismatches.append(f"{key}:{tk}")
            print(f"{key:34s} mse={mse!r} "
                  f"exact2d={ent['shard_exact']} "
                  f"exact3d={ent['shard_exact_stacked']} {dt:.2f}s")

    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    if args.save_tensors:
        torch.save(blobs, args.save_tensors)
    if base is not None:
        if mismatches:
            print(f"\nBIT-IDENTITY FAIL: {len(mismatches)} tensors differ")
            for m in mismatches[:20]:
                print("  ", m)
            return 1
        print(f"\nBIT-IDENTICAL vs {args.vs} "
              f"({sum(len(v) for v in blobs.values())} tensors, torch.equal)")
    bad = [k for k, v in report.items()
           if not (v["shard_exact"] or v["shard_exact_stacked"])]
    if bad:
        print(f"\nSHARD MISMATCH on {len(bad)}: {bad}")
        return 1
    print(f"\nall {len(report)} (target, format) rows match the production "
          f"shard weight_mse exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
