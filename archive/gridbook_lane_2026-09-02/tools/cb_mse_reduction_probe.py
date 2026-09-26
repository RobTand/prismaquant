#!/usr/bin/env python3
"""Is the residual replica-vs-shard weight_mse delta a REDUCTION artifact?

Takes the two rows whose fp32 2-D mean lands ~1 ULP off the production shard
and re-reduces the SAME W_hat several ways (fp32 2-D, fp32 stacked-3-D,
float64). If a different reduction of identical bytes reaches the shard value,
the encoding was never in question.
"""
import importlib.util as ilu
import sys
from pathlib import Path

import torch

spec = ilu.spec_from_file_location(
    "cb_encode_replica", str(Path(__file__).resolve().parent / "cb_encode_replica.py"))
R = ilu.module_from_spec(spec)
spec.loader.exec_module(R)

R.apply_prod_env()
TARGETS = [(0, "down_proj", "FP8_CB_K36"), (7, "up_proj", "NVFP4_CB_K15"),
           (0, "gate_proj", "NVFP4_CB_K14")]
for e, proj, fmt in TARGETS:
    w = R.load_expert_weight(0, e, proj)
    cw = R.load_col_weights(0, e, proj)
    ref = R.shard_row(0, e, proj)[fmt]["weight_mse"]
    w_hat, mse2d, _ = R.encode_one(w, cw, fmt, device="cuda")
    ws = w.to(device=w_hat.device, dtype=torch.bfloat16)
    err = (ws - w_hat).float()
    v = {
        "fp32_2d_mean": float(err.pow(2).mean()),
        "fp32_3d_mean": float(err.unsqueeze(0).pow(2).mean(dim=(1, 2))[0]),
        "fp32_sum_div": float(err.pow(2).sum() / err.numel()),
        "fp64_mean": float(err.double().pow(2).mean()),
        "fp64_as_f32": float(torch.tensor(
            float(err.double().pow(2).mean()), dtype=torch.float32)),
    }
    print(f"\n{e}:{proj}|{fmt}  shard={ref!r}")
    for k, x in v.items():
        print(f"   {k:16s} {x!r}  match={x == ref}")
