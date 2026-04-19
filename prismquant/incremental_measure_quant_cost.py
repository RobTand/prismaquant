#!/usr/bin/env python3
"""incremental_measure_quant_cost.py — shard cost measurement and merge outputs.

Persistent version: load the model once, keep the activation index open, sweep
layer shards in-process, and merge the shard pickles at the end.
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import torch

from .incremental_probe import (
    build_extended_shard_regexes,
    build_layer_shard_regexes,
    load_num_hidden_layers,
)
from .measure_quant_cost import (
    load_cost_model,
    prepare_cost_context,
    run_cost_pass,
    start_mem_watchdog,
)


def merge_cost_pickles(paths: list[Path], output_path: Path):
    merged_costs = {}
    merged_formats = None
    shard_metas = []
    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        costs = data["costs"]
        overlap = set(merged_costs) & set(costs)
        if overlap:
            raise ValueError(f"cost shards overlap on {len(overlap)} entries")
        merged_costs.update(costs)
        if merged_formats is None:
            merged_formats = data.get("formats", [])
        shard_metas.append(data.get("meta", {}))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "costs": merged_costs,
            "formats": merged_formats or [],
            "meta": {
                "incremental": True,
                "n_shards": len(paths),
                "shards": shard_metas,
            },
        }, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--activation-cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--formats", default="")
    ap.add_argument("--skip-missing-activations", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mode", choices=["auto", "batched", "unbatched"], default="auto")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--swap-grow-limit-mb", type=int, default=256)
    ap.add_argument("--min-mem-available-mb", type=int, default=2048)
    ap.add_argument("--no-watchdog", action="store_true")
    ap.add_argument("--layers-per-shard", type=int, default=1)
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=None)
    args = ap.parse_args()

    n_layers = load_num_hidden_layers(args.model)
    start = max(0, args.start_layer)
    end = n_layers if args.end_layer is None else min(args.end_layer, n_layers)
    if start >= end:
        raise SystemExit(f"empty layer range: start={start} end={end}")

    body_regexes = build_layer_shard_regexes(n_layers,
                                             args.layers_per_shard,
                                             layer_prefix="model.layers")
    first_shard = start // args.layers_per_shard
    last_shard = (end + args.layers_per_shard - 1) // args.layers_per_shard
    shard_regexes = body_regexes[first_shard:last_shard]
    # MTP / visual / lm_head shards match the probe's coverage so cost
    # has measurements for every probe stat entry.
    extra = build_extended_shard_regexes(
        args.model, args.layers_per_shard,
        include_body=False,
        include_mtp=True,
        include_visual=True,
        include_lm_head=True,
    )
    shard_regexes = shard_regexes + extra

    work_dir = Path(args.work_dir)
    shard_dir = work_dir / "shards"
    log_dir = work_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    _, _, act_cache, _, missing_act, _, specs = prepare_cost_context(
        probe_path=args.probe,
        activation_cache_dir=args.activation_cache_dir,
        formats_csv=args.formats,
        skip_missing_activations=args.skip_missing_activations,
    )
    print(f"[incremental-cost] loading model once for {len(shard_regexes)} shards", flush=True)
    model = load_cost_model(args.model, args.device, dtype,
                            device_map=args.device_map)
    if not args.no_watchdog:
        start_mem_watchdog(swap_grow_limit_mb=args.swap_grow_limit_mb,
                           min_mem_available_mb=args.min_mem_available_mb)

    with open(args.probe, "rb") as f:
        probe = pickle.load(f)

    shard_paths: list[Path] = []
    for shard_idx, linear_include in enumerate(shard_regexes):
        shard_path = shard_dir / f"cost_shard_{shard_idx:03d}.pkl"
        shard_paths.append(shard_path)
        if shard_path.exists():
            print(f"[incremental-cost] reuse shard {shard_idx}: {shard_path}", flush=True)
            continue

        shard_probe = shard_dir / f"probe_subset_{shard_idx:03d}.pkl"
        shard_probe_data = dict(probe)
        shard_probe_data["stats"] = {
            k: v for k, v in probe["stats"].items() if re.search(linear_include, k)
        }
        shard_probe = shard_dir / f"probe_subset_{shard_idx:03d}.pkl"
        with open(shard_probe, "wb") as f:
            pickle.dump(shard_probe_data, f)

        shard_target_names = set(shard_probe_data["stats"].keys())

        print(f"[incremental-cost] shard {shard_idx}: include={linear_include}", flush=True)
        run_cost_pass(
            model=model,
            act_cache=act_cache,
            target_names=shard_target_names,
            missing_act=[n for n in missing_act if n in shard_target_names],
            specs=specs,
            model_name=args.model,
            probe_path=str(shard_probe),
            device=args.device,
            dtype=dtype,
            mode=args.mode,
            chunk_size=args.chunk_size,
            output_path=str(shard_path),
        )
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    merge_cost_pickles(shard_paths, Path(args.output))
    print(f"[incremental-cost] wrote merged cost to {args.output}", flush=True)


if __name__ == "__main__":
    main()
