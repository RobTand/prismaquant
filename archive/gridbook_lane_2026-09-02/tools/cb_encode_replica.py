#!/usr/bin/env python3
"""Standalone replica of ONE production CB cost-encode, bit-comparable to
the shipped shard rows.

Rebuilds exactly what ``measure_quant_cost.measure_batched_gpu`` does for a
single Linear of the DSv4-Flash 92 GB production cost run:

  * decode the MXFP4 source weight the way ``layer_streaming`` does,
  * build the per-item imatrix col-weights the way the batched path does
    (``a.float().pow(2).mean(dim=0)`` over the FULL fp32 act rows),
  * call ``_cb_cost_quantize_dequantize`` under the production env context,
  * report ``weight_mse`` and compare against the production shard row.

No GPU work happens on import; every entry point takes an explicit device.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import torch

RUN = Path("/home/rob/dq-runs/dsv4-flash-0731")
SRC = RUN / "source"
CAL = RUN / "prod-cal-0p6"
ACT = CAL / "act"
SHARDS = CAL / "work-prod" / "shards"

PROD_ENV = {
    "PRISMAQUANT_ACTIVATION_FAIR_PRICING": "1",
    "CB_CODEBOOK_SOURCE": "lattice",
    "CB_SCALE_CODING": "two_tier",
    "CB_SCALE_SWEEP": "1",
    "PRISMAQUANT_CB_ENCODE_TIER": "balanced",
    "PRISMAQUANT_CB_EXT_DIR": str(RUN / "ext"),
    "PRISMAQUANT_CB_COL_WEIGHTS": str(CAL / "artifacts" / "cb_col_weights.pkl"),
}

# (live name, checkpoint leaf) for the routed-expert projections.
PROJ_TO_W = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}

_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def apply_prod_env() -> None:
    for k, v in PROD_ENV.items():
        os.environ[k] = v


def _index_map() -> dict:
    with open(SRC / "model.safetensors.index.json") as fh:
        return json.load(fh)["weight_map"]


def decode_mxfp4(packed: torch.Tensor, scale_u8: torch.Tensor,
                 device: str = "cpu") -> torch.Tensor:
    """Byte-for-byte the ``layer_streaming`` MXFP4 decode (OCP MX v1.0)."""
    dev = torch.device(device)
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=dev)
    codes = torch.arange(256, device=dev)
    pair_lut = torch.stack([lut[codes & 0x0F], lut[codes >> 4]], dim=-1)
    wp = packed.to(device=dev).view(torch.uint8)
    rows, packed_in = wp.shape
    logical_in = packed_in * 2
    deq = pair_lut[wp.to(torch.int32)].reshape(rows, logical_in // 32, 32)
    sb = scale_u8.to(device=dev).view(torch.uint8)
    scale = torch.exp2(sb.to(torch.float32) - 127.0)
    scale = torch.where(sb == 0xFF, torch.full_like(scale, float("nan")), scale)
    deq.mul_(scale.unsqueeze(-1))
    return deq.to(torch.bfloat16).reshape(rows, logical_in).contiguous()


def load_expert_weight(layer: int, expert: int, proj: str,
                       device: str = "cpu") -> torch.Tensor:
    """Live-namespace bf16 weight for ``layers.L.ffn.experts.E.{w1,w2,w3}``."""
    from safetensors import safe_open

    leaf = PROJ_TO_W[proj]
    key_w = f"layers.{layer}.ffn.experts.{expert}.{leaf}.weight"
    key_s = f"layers.{layer}.ffn.experts.{expert}.{leaf}.scale"
    wmap = _index_map()
    shard = wmap[key_w]
    assert wmap[key_s] == shard, "weight/scale live in different shards"
    with safe_open(SRC / shard, "pt") as fh:
        packed = fh.get_tensor(key_w)
        scale = fh.get_tensor(key_s)
    return decode_mxfp4(packed, scale, device=device)


def load_col_weights(layer: int, expert: int, proj: str) -> torch.Tensor:
    """The batched path's per-item imatrix: mean of squared FULL fp32 act rows."""
    p = ACT / f"model__layers__{layer}__mlp__experts__{expert}__{proj}.pt"
    blob = torch.load(p, map_location="cpu", weights_only=False)
    act = blob["inputs"] if isinstance(blob, dict) else blob
    return act.float().pow(2).mean(dim=0).reshape(-1)


def live_name(layer: int, expert: int, proj: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{proj}"


def shard_row(layer: int, expert: int, proj: str) -> dict:
    with open(SHARDS / f"cost_shard_{layer:03d}.pkl", "rb") as fh:
        d = pickle.load(fh)
    return d["costs"][live_name(layer, expert, proj)]


def spec_for(fmt: str):
    from prismaquant import format_registry as fr
    return fr.get_format(fmt)


def encode_one(w: torch.Tensor, cw: torch.Tensor, fmt: str,
               device: str = "cuda"):
    """Run the production render for one Linear. Returns (W_hat, weight_mse)."""
    from prismaquant.measure_quant_cost import _cb_cost_quantize_dequantize

    dev = torch.device(device)
    ws = w.to(device=dev, dtype=torch.bfloat16)
    cws = cw.to(device=dev)
    t0 = time.time()
    w_hat = _cb_cost_quantize_dequantize(spec_for(fmt), ws.clone(),
                                         col_weights=cws)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    mse = (ws - w_hat).float().pow(2).mean()
    return w_hat, float(mse), dt


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--proj", default="gate_proj", choices=sorted(PROJ_TO_W))
    ap.add_argument("--formats", default="NVFP4_CB_K14,NVFP4_CB_K15,FP8_CB_K36")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    apply_prod_env()
    w = load_expert_weight(args.layer, args.expert, args.proj)
    cw = load_col_weights(args.layer, args.expert, args.proj)
    row = shard_row(args.layer, args.expert, args.proj)
    print(f"weight {tuple(w.shape)} {w.dtype}  col_weights {tuple(cw.shape)}")

    out = {}
    for fmt in args.formats.split(","):
        w_hat, mse, dt = encode_one(w, cw, fmt, device=args.device)
        ref = row[fmt]["weight_mse"]
        ok = (mse == ref)
        print(f"{fmt:14s} weight_mse={mse!r}  shard={ref!r}  "
              f"exact={ok}  rel={abs(mse - ref) / max(ref, 1e-30):.3e}  "
              f"{dt:.2f}s")
        out[fmt] = {"w_hat": w_hat.cpu(), "weight_mse": mse, "shard": ref}
    if args.save:
        torch.save(out, args.save)
        print(f"saved -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
