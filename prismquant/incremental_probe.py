#!/usr/bin/env python3
"""incremental_probe.py — run the PrismQuant sensitivity probe in shards.

This keeps the live hook state bounded by probing only a subset of transformer
blocks at a time, then merges the per-shard probe artifacts into one final
`probe.pkl`. Activation snapshots are written into one shared cache dir so
later stages can consume them normally.

This does not magically eliminate model residency cost; it does eliminate the
"all layers hooked at once" part of the memory profile and makes long runs
resumable.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import torch

from .sensitivity_probe import (
    load_calibration,
    load_probe_model_and_tokenizer,
    run_probe_pass,
    stage_text_only,
)


def build_layer_shard_regexes(num_hidden_layers: int,
                              layers_per_shard: int,
                              layer_prefix: str = "model.layers") -> list[str]:
    regexes: list[str] = []
    for start in range(0, num_hidden_layers, layers_per_shard):
        end = min(start + layers_per_shard, num_hidden_layers)
        if end - start == 1:
            body = rf"{re.escape(layer_prefix)}\.{start}\."
        else:
            idxs = "|".join(str(i) for i in range(start, end))
            body = rf"{re.escape(layer_prefix)}\.(?:{idxs})\."
        regexes.append(body)
    return regexes


def build_extended_shard_regexes(
    model_path: str,
    layers_per_shard: int,
    *,
    include_body: bool = True,
    include_mtp: bool = True,
    include_visual: bool = True,
    include_lm_head: bool = True,
) -> list[str]:
    """Extended shard list covering everything quantizable in a
    multimodal Qwen3.5/3.6 checkpoint:

      - body transformer    (`model.layers.X.*`)         — N shards
      - MTP block(s)        (`mtp.layers.X.*`)           — 1 shard typically
      - visual ViT blocks   (`model.visual.blocks.X.*`)  — depth/N shards
      - lm_head             (`^lm_head$`)                — 1 shard
    """
    # Read the ORIGINAL config, not the text-only staged one. Staging
    # strips `vision_config`, which would make us miss every visual
    # block. For the staged-body layer count we still want text_config
    # when present.
    src_cfg_path = Path(model_path) / "config.json"
    with open(src_cfg_path) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    vis_cfg = cfg.get("vision_config", {})

    regexes: list[str] = []

    if include_body:
        n_body = int(text_cfg.get("num_hidden_layers", cfg.get("num_hidden_layers", 0)))
        regexes.extend(build_layer_shard_regexes(
            n_body, layers_per_shard, layer_prefix="model.layers"))

    if include_mtp:
        # MTP layer count is small (Qwen3.6 = 1). Try common config keys;
        # fall back to scanning the source safetensors index.
        n_mtp = int(
            text_cfg.get("num_nextn_predict_layers")
            or cfg.get("num_nextn_predict_layers")
            or text_cfg.get("num_mtp_layers")
            or cfg.get("num_mtp_layers")
            or _count_mtp_layers_from_safetensors(model_path)
            or 0
        )
        if n_mtp > 0:
            regexes.extend(build_layer_shard_regexes(
                n_mtp, layers_per_shard, layer_prefix="mtp.layers"))

    if include_visual:
        n_vis = int(vis_cfg.get("depth") or vis_cfg.get("num_hidden_layers") or 0)
        if n_vis > 0:
            # Visual blocks share a uniform shape, so a slightly larger
            # shard per group is fine (memory pressure per shard is
            # bounded by max-shard size, dominated by the body Linears
            # not the visual ones).
            vis_per_shard = max(layers_per_shard, 4)
            regexes.extend(build_layer_shard_regexes(
                n_vis, vis_per_shard, layer_prefix="model.visual.blocks"))

    if include_lm_head:
        # Single Linear at the model root — its own shard.
        regexes.append(r"^lm_head$")

    return regexes


def _count_mtp_layers_from_safetensors(model_path: str) -> int:
    """Fallback for when the config doesn't carry an MTP layer count:
    scan the source safetensors index and count `mtp.layers.<N>.` paths."""
    import os, re as _re
    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    if not idx_path.exists():
        # Fall back to listing tensors directly.
        try:
            from safetensors.torch import safe_open
            mtp_indices = set()
            for f in os.listdir(src):
                if not f.endswith(".safetensors"):
                    continue
                with safe_open(str(src / f), framework="pt") as sf:
                    for k in sf.keys():
                        m = _re.match(r"^mtp\.layers\.(\d+)\.", k)
                        if m:
                            mtp_indices.add(int(m.group(1)))
            return max(mtp_indices) + 1 if mtp_indices else 0
        except Exception:
            return 0
    with open(idx_path) as f:
        wm = json.load(f)["weight_map"]
    mtp_indices = set()
    for k in wm:
        m = _re.match(r"^mtp\.layers\.(\d+)\.", k)
        if m:
            mtp_indices.add(int(m.group(1)))
    return max(mtp_indices) + 1 if mtp_indices else 0


def _merge_nested_counts(dst: dict, src: dict):
    for key, sub in src.items():
        tgt = dst.setdefault(key, {})
        for sk, sv in sub.items():
            tgt[sk] = tgt.get(sk, 0.0) + float(sv)


def merge_probe_pickles(paths: list[Path], output_path: Path):
    merged = None
    merged_stats = {}
    merged_router_counts = {}
    merged_router_totals = defaultdict(int)
    merged_expert_info = {}
    shard_metas = []

    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if merged is None:
            merged = data
        overlap = set(merged_stats) & set(data["stats"])
        if overlap:
            raise ValueError(f"probe shards overlap on {len(overlap)} stats entries")
        merged_stats.update(data["stats"])
        _merge_nested_counts(merged_router_counts, data.get("router_counts", {}))
        for rk, rv in data.get("router_totals", {}).items():
            merged_router_totals[rk] += int(rv)
        merged_expert_info.update(data.get("expert_info", {}))
        shard_metas.append(data.get("meta", {}))

    if merged is None:
        raise ValueError("no probe shards to merge")

    merged["stats"] = merged_stats
    merged["router_counts"] = dict(merged_router_counts)
    merged["router_totals"] = dict(merged_router_totals)
    merged["expert_info"] = merged_expert_info
    merged["meta"] = {
        **merged.get("meta", {}),
        "incremental": True,
        "n_shards": len(paths),
        "shards": shard_metas,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(merged, f)


def load_num_hidden_layers(model_path: str) -> int:
    staged = stage_text_only(model_path)
    cfg_path = Path(staged) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    n = cfg.get("num_hidden_layers")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Could not infer num_hidden_layers from {cfg_path}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ultrachat_200k")
    ap.add_argument("--nsamples", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--output", required=True)
    ap.add_argument("--activation-cache-dir", required=True)
    ap.add_argument("--work-dir", required=True,
                    help="Stores shard logs/pickles; safe to resume.")
    ap.add_argument("--layers-per-shard", type=int, default=1)
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=None)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", action="store_false",
                    dest="gradient_checkpointing")
    ap.add_argument("--importance-weighting", action="store_true", default=True)
    ap.add_argument("--no-importance-weighting", action="store_false",
                    dest="importance_weighting")
    ap.add_argument("--include-mtp", action="store_true", default=True,
                    help="Probe MTP layers (`mtp.layers.X.*`).")
    ap.add_argument("--no-include-mtp", action="store_false", dest="include_mtp")
    ap.add_argument("--include-visual", action="store_true", default=True,
                    help="Probe visual encoder blocks (`model.visual.blocks.X.*`).")
    ap.add_argument("--no-include-visual", action="store_false", dest="include_visual")
    ap.add_argument("--include-lm-head", action="store_true", default=True,
                    help="Probe lm_head (`^lm_head$`).")
    ap.add_argument("--no-include-lm-head", action="store_false", dest="include_lm_head")
    args = ap.parse_args()

    # Body shard range may be partial; MTP / visual / lm_head are
    # always all-or-nothing (small enough that splitting them is silly).
    n_layers = load_num_hidden_layers(args.model)
    start = max(0, args.start_layer)
    end = n_layers if args.end_layer is None else min(args.end_layer, n_layers)
    if start >= end:
        raise SystemExit(f"empty layer range: start={start} end={end}")

    body_regexes = build_layer_shard_regexes(
        n_layers, args.layers_per_shard, layer_prefix="model.layers")
    first_shard = start // args.layers_per_shard
    last_shard = (end + args.layers_per_shard - 1) // args.layers_per_shard
    shard_regexes = body_regexes[first_shard:last_shard]

    # Append MTP / visual / lm_head shards beyond the body range. These
    # are unaffected by --start-layer / --end-layer (those are body-only).
    extra = build_extended_shard_regexes(
        args.model, args.layers_per_shard,
        include_body=False,
        include_mtp=args.include_mtp,
        include_visual=args.include_visual,
        include_lm_head=args.include_lm_head,
    )
    shard_regexes = shard_regexes + extra
    print(f"[incremental] shard regexes: {len(shard_regexes)} total "
          f"(body={len(body_regexes[first_shard:last_shard])}, extras={len(extra)})",
          flush=True)

    work_dir = Path(args.work_dir)
    shard_dir = work_dir / "shards"
    log_dir = work_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.activation_cache_dir).mkdir(parents=True, exist_ok=True)

    # Persistent runner: load once, calibrate once, sweep shard regexes.
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[incremental] loading model once for {len(shard_regexes)} shards", flush=True)
    _, tokenizer, model, exec_device, load_device_map = load_probe_model_and_tokenizer(
        args.model,
        requested_device=args.device,
        dtype=dtype,
        device_map=args.device_map,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    calib = load_calibration(tokenizer, args.dataset, args.nsamples, args.seqlen)
    print(f"[incremental] calibration ready: {tuple(calib.shape)}", flush=True)

    shard_paths = []
    for shard_idx, linear_include in enumerate(shard_regexes):
        shard_path = shard_dir / f"probe_shard_{shard_idx:03d}.pkl"
        shard_paths.append(shard_path)
        if shard_path.exists():
            print(f"[incremental] reuse shard {shard_idx}: {shard_path}", flush=True)
            continue
        print(f"[incremental] shard {shard_idx}: include={linear_include}", flush=True)
        run_probe_pass(
            model=model,
            tokenizer=tokenizer,
            calib=calib,
            model_name=args.model,
            dataset_name=args.dataset,
            seqlen=args.seqlen,
            dtype_name=args.dtype,
            requested_device=args.device,
            load_device_map=load_device_map,
            exec_device=exec_device,
            linear_include=linear_include,
            # Routers excluded; `lm_head` intentionally INCLUDED (the
            # allocator picks its format alongside the body).
            linear_exclude=r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|block_sparse_moe\.gate$)",
            importance_weighting=args.importance_weighting,
            activation_cache_dir=args.activation_cache_dir,
            output_path=str(shard_path),
        )
        if exec_device.type == "cuda":
            torch.cuda.empty_cache()

    merge_probe_pickles(shard_paths, Path(args.output))
    print(f"[incremental] wrote merged probe to {args.output}", flush=True)


if __name__ == "__main__":
    main()
