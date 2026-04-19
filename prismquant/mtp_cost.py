#!/usr/bin/env python3
"""mtp_cost.py — per-(MTP Linear, format) quantization error, full parity
with the body cost pipeline.

Requires `mtp_probe.py` to have been run first with
`--activation-cache-dir <dir>` — we reuse those cached activations to
drive the same `measure_batched_gpu` path the body uses, so both
weight_mse and output_mse are recorded.

Builds the MTP module from the body's `text_config`, wraps it under a
`mtp.` parent so Linear names (`mtp.fc`, `mtp.layers.0.self_attn.q_proj`,
...) match the probe and export conventions, loads `mtp.*` safetensors,
then runs the standard cost passes against it.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn

from . import format_registry as fr
from .measure_quant_cost import (
    ActivationIndex,
    _finalize_results,
    _measure_packed_experts,
    measure_batched_gpu,
    measure_unbatched,
)
from .mtp_probe import MtpModule, _load_into_mtp, _load_mtp_state_dict


def _load_body_text_config(model_path: str):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return getattr(cfg, "text_config", cfg)


def _build_wrapped_mtp(model_path: str, device: str, dtype: torch.dtype):
    text_config = _load_body_text_config(model_path)
    inner = MtpModule(text_config)
    wrapper = nn.Module()
    wrapper.add_module("mtp", inner)
    wrapper.to(device=device, dtype=dtype)
    raw = _load_mtp_state_dict(model_path)
    _load_into_mtp(inner, raw)
    wrapper.eval()
    return wrapper


def _collect_mtp_targets(wrapper: nn.Module) -> set[str]:
    targets: set[str] = set()
    for name, module in wrapper.named_modules():
        if isinstance(module, nn.Linear) and not name.endswith(".mlp.gate"):
            targets.add(name)
    for name, module in wrapper.named_modules():
        if not type(module).__name__.lower().endswith("experts"):
            continue
        for pn, p in module.named_parameters(recurse=False):
            if p.dim() == 3 and pn in {"gate_up_proj", "down_proj"}:
                targets.add(f"{name}.{pn}")
    return targets


def run_mtp_cost(model_path: str,
                 formats_csv: str,
                 output_path: str,
                 activation_cache_dir: str,
                 device: str,
                 dtype: torch.dtype,
                 mode: str,
                 chunk_size: int):
    wrapper = _build_wrapped_mtp(model_path, device=device, dtype=dtype)
    target_names = _collect_mtp_targets(wrapper)
    print(f"[mtp-cost] tracking {len(target_names)} MTP tensors", flush=True)
    for n in sorted(target_names):
        print(f"  {n}", flush=True)

    fmt_names = [s.strip() for s in formats_csv.split(",") if s.strip()]
    specs = [fr.get_format(n) for n in fmt_names]
    print(f"[mtp-cost] formats: {[s.name for s in specs]}", flush=True)

    act_cache = ActivationIndex(Path(activation_cache_dir), target_names)
    print(f"[mtp-cost] activation cache: {len(act_cache)} Linears mapped",
          flush=True)
    missing_acts = [n for n in target_names if n not in act_cache]
    if missing_acts:
        print(f"[mtp-cost] WARNING {len(missing_acts)} targets without activations:",
              flush=True)
        for n in missing_acts:
            print(f"  (no acts) {n}", flush=True)

    chosen_mode = mode
    if chosen_mode == "auto":
        chosen_mode = "batched" if device.startswith("cuda") else "unbatched"
    if chosen_mode == "batched":
        results = measure_batched_gpu(
            wrapper, act_cache, target_names, specs, device, dtype,
            chunk_size=chunk_size,
        )
    else:
        results = measure_unbatched(
            wrapper, act_cache, target_names, specs, device, dtype,
        )

    packed_accum: dict[str, dict] = {}
    _measure_packed_experts(wrapper, target_names, specs, device, dtype,
                            packed_accum)
    if packed_accum:
        results.update(_finalize_results(packed_accum))
        print(f"[mtp-cost] measured {len(packed_accum)} packed-expert tensors",
              flush=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "costs": results,
            "formats": fmt_names,
            "meta": {
                "mtp_cost": True,
                "n_tensors": len(results),
                "device": device,
                "mode": chosen_mode,
            },
        }, f)
    print(f"[mtp-cost] wrote {output_path} ({len(results)} entries)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--formats", default="NVFP4,MXFP8_E4M3,BF16")
    ap.add_argument("--output", required=True)
    ap.add_argument("--activation-cache-dir", required=True,
                    help="Directory where mtp_probe.py wrote cached activations.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--mode", choices=["auto", "batched", "unbatched"], default="auto")
    ap.add_argument("--chunk-size", type=int, default=256)
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    run_mtp_cost(
        model_path=args.model,
        formats_csv=args.formats,
        output_path=args.output,
        activation_cache_dir=args.activation_cache_dir,
        device=args.device,
        dtype=dtype,
        mode=args.mode,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
