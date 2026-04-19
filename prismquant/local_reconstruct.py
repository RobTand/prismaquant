#!/usr/bin/env python3
"""local_reconstruct.py — improve critical native-format candidates locally.

For a small set of important layers, refine per-format costs by grid-searching
simple symmetric clipping factors on weights and activations, minimizing the
measured layer output MSE on cached activations.

This is intentionally conservative:
  - one layer at a time
  - one format at a time
  - tiny clip grids
  - no extra optimizer state
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from . import format_registry as fr
from .allocator import aggregate_moe_candidates, build_candidates, promote_fused, solve_allocation
from .calibrate_allocator import load_inputs
from .interaction_refine import build_refinement_units, select_critical_units
from .measure_quant_cost import ActivationIndex, _load_live_model


def _sym_clip(x: torch.Tensor, factor, group_size: int | None = None) -> torch.Tensor:
    if isinstance(factor, torch.Tensor):
        f = factor.to(device=x.device, dtype=x.dtype)
        if bool(torch.all(f >= 0.999999)):
            return x
    else:
        f = float(factor)
        if f >= 0.999999:
            return x
    if isinstance(f, torch.Tensor) and f.ndim >= 2 and f.shape[-1] > 1:
        if group_size is None or group_size <= 0:
            raise ValueError("groupwise clip requested without valid group_size")
        if x.shape[-1] % group_size != 0:
            raise ValueError("tensor width must be divisible by group_size for groupwise clip")
        groups = x.shape[-1] // group_size
        if f.shape[-2:] != (x.shape[0], groups):
            raise ValueError("groupwise clip tensor must have shape [rows, groups]")
        xg = x.view(x.shape[0], groups, group_size)
        max_abs = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        fg = f.view(x.shape[0], groups, 1)
        limit = max_abs * fg
        return xg.clamp(-limit, limit).view_as(x)
    max_abs = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    limit = max_abs * f
    return x.clamp(-limit, limit)


def _measure_entry(W: torch.Tensor, X: torch.Tensor, spec: fr.FormatSpec, w_clip: float, a_clip: float):
    W_in = _sym_clip(W, w_clip, group_size=spec.group_size)
    X_in = _sym_clip(X, a_clip)
    W_hat = spec.quantize_dequantize(W_in.clone())
    X_hat = spec.activation_quantize_dequantize(X_in.clone())
    y_ref = X @ W.T
    y_q = X_hat @ W_hat.T
    weight_mse = float((W - W_hat).float().pow(2).mean().item())
    out_err = (y_ref - y_q).float().pow(2)
    output_mse = float(out_err.mean().item())
    ref_energy = float(y_ref.float().pow(2).mean().item())
    return {
        "weight_mse": weight_mse,
        "output_mse": output_mse,
        "rel_output_mse": output_mse / max(ref_energy, 1e-12),
        "weight_clip": w_clip,
        "act_clip": a_clip,
        "per_output_mse": out_err.mean(dim=0).detach().cpu(),
    }


def _candidate_clip_values(best: float, step: float) -> list[float]:
    vals = {
        1.0,
        max(0.5, min(1.0, best)),
        max(0.5, min(1.0, best - step)),
        max(0.5, min(1.0, best + step)),
    }
    return sorted(vals, reverse=True)


def _entry_score(entry: dict) -> float:
    return float(entry["output_mse"])


def _summarize_weight_clip(weight_clip):
    if isinstance(weight_clip, torch.Tensor):
        clip_cpu = weight_clip.detach().cpu()
        flat = [float(x) for x in clip_cpu.view(-1).tolist()]
        mode = "groupwise" if clip_cpu.ndim >= 2 and clip_cpu.shape[-1] > 1 else "rowwise"
        return {
            "mode": mode,
            "default": flat[0] if flat else 1.0,
            "min": min(flat) if flat else 1.0,
            "max": max(flat) if flat else 1.0,
            "values": flat,
        }
    clip = float(weight_clip)
    return {
        "mode": "scalar",
        "default": clip,
        "min": clip,
        "max": clip,
    }


def _rowwise_refine_weight_clip(
    W: torch.Tensor,
    X: torch.Tensor,
    spec: fr.FormatSpec,
    best: dict,
    rowwise_topk: int,
    rowwise_rounds: int,
):
    if rowwise_topk <= 0 or rowwise_rounds <= 0:
        return best
    per_output = best.get("per_output_mse")
    if per_output is None or per_output.numel() <= 1:
        return best
    topk = min(int(rowwise_topk), int(per_output.numel()))
    top_rows = torch.topk(per_output, k=topk).indices.tolist()
    row_clips = torch.full((W.shape[0], 1), float(best["weight_clip"]), device=W.device, dtype=W.dtype)
    best_entry = dict(best)
    best_entry["weight_clip"] = row_clips.clone()
    step = 0.02
    for _ in range(max(rowwise_rounds, 0)):
        improved = False
        for row in top_rows:
            current = float(row_clips[row, 0].item())
            for candidate in _candidate_clip_values(current, step):
                trial = row_clips.clone()
                trial[row, 0] = candidate
                try:
                    entry = _measure_entry(W, X, spec, trial, float(best_entry["act_clip"]))
                except Exception:
                    continue
                if _entry_score(entry) + 1e-12 < _entry_score(best_entry):
                    row_clips = trial
                    best_entry = entry
                    best_entry["weight_clip"] = row_clips.clone()
                    improved = True
        step *= 0.5
        if not improved:
            continue
    return best_entry


def _select_top_weight_blocks(
    W: torch.Tensor,
    X: torch.Tensor,
    spec: fr.FormatSpec,
    weight_clip,
    topk: int,
) -> list[tuple[int, int]]:
    if topk <= 0 or spec.group_size <= 0 or W.shape[-1] % spec.group_size != 0:
        return []
    W_in = _sym_clip(W, weight_clip, group_size=spec.group_size)
    W_hat = spec.quantize_dequantize(W_in.clone())
    x_var = X.float().pow(2).mean(dim=0)
    groups = W.shape[-1] // spec.group_size
    weighted = (W.float() - W_hat.float()).pow(2) * x_var.unsqueeze(0)
    block_scores = weighted.view(W.shape[0], groups, spec.group_size).sum(dim=-1)
    topk = min(int(topk), int(block_scores.numel()))
    if topk <= 0:
        return []
    flat_idx = torch.topk(block_scores.reshape(-1), k=topk).indices.tolist()
    return [(idx // groups, idx % groups) for idx in flat_idx]


def _groupwise_refine_weight_clip(
    W: torch.Tensor,
    X: torch.Tensor,
    spec: fr.FormatSpec,
    best: dict,
    groupwise_topk: int,
    groupwise_rounds: int,
):
    if groupwise_topk <= 0 or groupwise_rounds <= 0 or spec.group_size <= 0:
        return best
    groups = W.shape[-1] // spec.group_size
    base = best["weight_clip"]
    if isinstance(base, torch.Tensor):
        if base.ndim >= 2 and base.shape[-1] == groups:
            group_clips = base.clone().to(device=W.device, dtype=W.dtype)
        elif base.ndim >= 2 and base.shape[-1] == 1:
            group_clips = base.expand(W.shape[0], groups).clone().to(device=W.device, dtype=W.dtype)
        else:
            return best
    else:
        group_clips = torch.full((W.shape[0], groups), float(base), device=W.device, dtype=W.dtype)
    best_entry = dict(best)
    best_entry["weight_clip"] = group_clips.clone()
    step = 0.02
    for _ in range(max(groupwise_rounds, 0)):
        improved = False
        targets = _select_top_weight_blocks(W, X, spec, best_entry["weight_clip"], groupwise_topk)
        for row, group in targets:
            current = float(group_clips[row, group].item())
            for candidate in _candidate_clip_values(current, step):
                trial = group_clips.clone()
                trial[row, group] = candidate
                try:
                    entry = _measure_entry(W, X, spec, trial, float(best_entry["act_clip"]))
                except Exception:
                    continue
                if _entry_score(entry) + 1e-12 < _entry_score(best_entry):
                    group_clips = trial
                    best_entry = entry
                    best_entry["weight_clip"] = group_clips.clone()
                    improved = True
        step *= 0.5
        if not improved:
            continue
    return best_entry


def _row_clip_for_weight_clip(weight_clip, row: int, groups: int):
    if isinstance(weight_clip, torch.Tensor):
        clip = weight_clip[row:row + 1]
        if clip.ndim >= 2 and clip.shape[-1] == groups:
            return clip.clone()
        return clip.clone()
    return float(weight_clip)


def _quantize_row_with_clip(row_vec: torch.Tensor, spec: fr.FormatSpec, row_clip):
    row_in = _sym_clip(row_vec.unsqueeze(0), row_clip, group_size=spec.group_size)
    row_hat = spec.quantize_dequantize(row_in.clone())
    return row_hat.squeeze(0)


def _measure_with_quantized_weight(
    W: torch.Tensor,
    X: torch.Tensor,
    spec: fr.FormatSpec,
    weight_clip,
    act_clip: float,
    W_hat: torch.Tensor,
):
    X_in = _sym_clip(X, act_clip)
    X_hat = spec.activation_quantize_dequantize(X_in.clone())
    y_ref = X @ W.T
    y_q = X_hat @ W_hat.T
    out_err = (y_ref - y_q).float().pow(2)
    weight_mse = float((W - W_hat).float().pow(2).mean().item())
    output_mse = float(out_err.mean().item())
    ref_energy = float(y_ref.float().pow(2).mean().item())
    return {
        "weight_mse": weight_mse,
        "output_mse": output_mse,
        "rel_output_mse": output_mse / max(ref_energy, 1e-12),
        "weight_clip": weight_clip,
        "act_clip": act_clip,
        "per_output_mse": out_err.mean(dim=0).detach().cpu(),
    }


def _gptq_lite_quantize_row(
    row_vec: torch.Tensor,
    X_hat: torch.Tensor,
    spec: fr.FormatSpec,
    row_clip,
    damping: float,
):
    n_in = row_vec.numel()
    if spec.group_size <= 0 or n_in % spec.group_size != 0:
        return _quantize_row_with_clip(row_vec, spec, row_clip)
    H = (X_hat.float().T @ X_hat.float()) / max(int(X_hat.shape[0]), 1)
    diag_mean = float(torch.diag(H).mean().item()) if H.numel() else 1.0
    H = H + torch.eye(n_in, device=H.device, dtype=H.dtype) * max(damping * diag_mean, 1e-8)
    row_work = row_vec.float().clone()
    q_row = torch.empty_like(row_work)
    group = spec.group_size
    groups = n_in // group
    for g in range(groups):
        start = g * group
        end = start + group
        q_block = _quantize_row_with_clip(row_work[start:end], spec, _row_clip_for_weight_clip(row_clip, 0, groups)[:, g:g + 1] if isinstance(row_clip, torch.Tensor) and row_clip.ndim >= 2 and row_clip.shape[-1] == groups else row_clip)
        q_row[start:end] = q_block
        if end >= n_in:
            continue
        err = q_block - row_work[start:end]
        H_ff = H[end:, end:]
        H_fb = H[end:, start:end]
        try:
            delta = -torch.linalg.solve(H_ff, H_fb @ err.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            continue
        row_work[end:] = row_work[end:] + delta.to(row_work.dtype)
    return q_row.to(row_vec.dtype)


def _gptq_lite_refine_rows(
    W: torch.Tensor,
    X: torch.Tensor,
    spec: fr.FormatSpec,
    best: dict,
    gptq_topk: int,
    gptq_damping: float,
):
    if gptq_topk <= 0 or spec.name == "BF16":
        return best
    per_output = best.get("per_output_mse")
    if per_output is None or per_output.numel() <= 1:
        return best
    topk = min(int(gptq_topk), int(per_output.numel()))
    top_rows = torch.topk(per_output, k=topk).indices.tolist()
    X_in = _sym_clip(X, float(best["act_clip"]))
    X_hat = spec.activation_quantize_dequantize(X_in.clone())
    base_weight_clip = best["weight_clip"]
    W_in = _sym_clip(W, base_weight_clip, group_size=spec.group_size)
    W_hat = spec.quantize_dequantize(W_in.clone())
    for row in top_rows:
        row_clip = _row_clip_for_weight_clip(base_weight_clip, row, W.shape[-1] // spec.group_size if spec.group_size > 0 else 1)
        W_hat[row] = _gptq_lite_quantize_row(W_in[row], X_hat, spec, row_clip, gptq_damping)
    entry = _measure_with_quantized_weight(W, X, spec, base_weight_clip, float(best["act_clip"]), W_hat)
    if _entry_score(entry) + 1e-12 < _entry_score(best):
        return entry
    return best


def _refine_measurement(
    W: torch.Tensor,
    X: torch.Tensor,
    spec: fr.FormatSpec,
    w_grid: list[float],
    a_grid: list[float],
    rounds: int,
    rowwise_topk: int,
    rowwise_rounds: int,
    groupwise_topk: int,
    groupwise_rounds: int,
    gptq_topk: int,
    gptq_damping: float,
):
    best = None
    for w_clip in w_grid:
        for a_clip in a_grid:
            try:
                entry = _measure_entry(W, X, spec, w_clip, a_clip)
            except Exception as exc:
                entry = {"error": str(exc), "weight_clip": w_clip, "act_clip": a_clip}
            if "error" in entry:
                continue
            if best is None or _entry_score(entry) < _entry_score(best):
                best = entry
    if best is None:
        return None

    step = 0.02
    for _ in range(max(rounds, 0)):
        improved = False
        for w_clip in _candidate_clip_values(best["weight_clip"], step):
            for a_clip in _candidate_clip_values(best["act_clip"], step):
                try:
                    entry = _measure_entry(W, X, spec, w_clip, a_clip)
                except Exception:
                    continue
                if _entry_score(entry) + 1e-12 < _entry_score(best):
                    best = entry
                    improved = True
        step *= 0.5
        if not improved:
            continue
    best = _rowwise_refine_weight_clip(
        W,
        X,
        spec,
        best,
        rowwise_topk=rowwise_topk,
        rowwise_rounds=rowwise_rounds,
    )
    best = _groupwise_refine_weight_clip(
        W,
        X,
        spec,
        best,
        groupwise_topk=groupwise_topk,
        groupwise_rounds=groupwise_rounds,
    )
    return _gptq_lite_refine_rows(
        W,
        X,
        spec,
        best,
        gptq_topk=gptq_topk,
        gptq_damping=gptq_damping,
    )


def expand_live_target_layers(critical_units, stats_alloc: dict) -> set[str]:
    target_layers = set()
    for unit in critical_units:
        for member in unit.members:
            fused_members = stats_alloc.get(member, {}).get("_fused_members")
            if fused_members:
                target_layers.update(fused_members)
            else:
                target_layers.add(member)
    return target_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--activation-cache-dir", required=True)
    ap.add_argument("--formats", required=True)
    ap.add_argument("--target-bits", type=float, required=True)
    ap.add_argument("--top-units", type=int, default=8)
    ap.add_argument("--unit-scope", choices=["sibling", "block", "hybrid", "layer"], default="sibling")
    ap.add_argument("--w-clip-grid", default="1.0,0.995,0.99,0.98,0.95,0.9")
    ap.add_argument("--a-clip-grid", default="1.0,0.995,0.99,0.98,0.95,0.9")
    ap.add_argument("--refine-rounds", type=int, default=2)
    ap.add_argument("--rowwise-topk", type=int, default=8)
    ap.add_argument("--rowwise-rounds", type=int, default=1)
    ap.add_argument("--groupwise-topk", type=int, default=16)
    ap.add_argument("--groupwise-rounds", type=int, default=1)
    ap.add_argument("--gptq-topk", type=int, default=8)
    ap.add_argument("--gptq-damping", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--expert-granularity", choices=["layer", "expert"], default="layer")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    fmt_names = [s.strip() for s in args.formats.split(",") if s.strip()]
    stats, costs, specs_sorted = load_inputs(Path(args.probe), Path(args.costs), fmt_names)
    candidates = build_candidates(stats, costs, specs_sorted)
    stats_alloc = stats
    if args.expert_granularity == "layer":
        stats_alloc, costs, candidates = aggregate_moe_candidates(stats, costs, specs_sorted, candidates)
    format_rank = {s.name: i for i, s in enumerate(specs_sorted)}
    assignment = solve_allocation(stats_alloc, candidates, args.target_bits, 0.001)
    if assignment is None:
        raise SystemExit("no feasible assignment at requested target")
    assignment = promote_fused(assignment, format_rank)
    units = build_refinement_units(stats_alloc, candidates, assignment, unit_scope=args.unit_scope)
    critical = select_critical_units(units, args.top_units)
    target_layers = expand_live_target_layers(critical, stats_alloc)

    with open(args.costs, "rb") as f:
        cost_blob = pickle.load(f)
    raw_costs = cost_blob["costs"]
    act_cache = ActivationIndex(Path(args.activation_cache_dir), target_layers)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = _load_live_model(args.model, args.device, dtype, unfuse_moe=True)

    module_map = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name in target_layers:
            module_map[name] = mod

    w_grid = [float(x) for x in args.w_clip_grid.split(",") if x.strip()]
    a_grid = [float(x) for x in args.a_clip_grid.split(",") if x.strip()]
    upgraded = {}
    for layer_name in sorted(target_layers):
        if layer_name not in module_map or layer_name not in act_cache:
            continue
        mod = module_map[layer_name]
        W = mod.weight.detach()
        X = act_cache.load(layer_name).to(W.dtype).to(W.device)
        per_fmt = {}
        for spec in specs_sorted:
            if spec.name not in raw_costs.get(layer_name, {}):
                continue
            best = _refine_measurement(
                W,
                X,
                spec,
                w_grid,
                a_grid,
                args.refine_rounds,
                rowwise_topk=args.rowwise_topk,
                rowwise_rounds=args.rowwise_rounds,
                groupwise_topk=args.groupwise_topk,
                groupwise_rounds=args.groupwise_rounds,
                gptq_topk=args.gptq_topk,
                gptq_damping=args.gptq_damping,
            )
            if best is not None:
                best["source"] = "local_reconstruct"
                best["weight_clip_summary"] = _summarize_weight_clip(best["weight_clip"])
                if "per_output_mse" in best:
                    best["per_output_mse"] = best["per_output_mse"].tolist()
                per_fmt[spec.name] = best
        if per_fmt:
            upgraded[layer_name] = per_fmt
            for fmt, entry in per_fmt.items():
                raw_costs.setdefault(layer_name, {})[fmt] = entry
                clip_summary = entry["weight_clip_summary"]
                print(
                    f"[reconstruct] {layer_name} {fmt} output_mse={entry['output_mse']:.4e} "
                    f"w_clip={clip_summary['default']:.3f}"
                    + (
                        f"[{clip_summary['min']:.3f},{clip_summary['max']:.3f}]"
                        if clip_summary["mode"] == "rowwise" else ""
                    )
                    + f" a_clip={entry['act_clip']:.3f}",
                    flush=True,
                )

    cost_blob["costs"] = raw_costs
    meta = dict(cost_blob.get("meta", {}))
    meta["local_reconstruct"] = {
        "target_bits": args.target_bits,
        "top_units": args.top_units,
        "formats": fmt_names,
        "unit_scope": args.unit_scope,
        "w_clip_grid": w_grid,
        "a_clip_grid": a_grid,
        "refine_rounds": args.refine_rounds,
        "rowwise_topk": args.rowwise_topk,
        "rowwise_rounds": args.rowwise_rounds,
        "groupwise_topk": args.groupwise_topk,
        "groupwise_rounds": args.groupwise_rounds,
        "gptq_topk": args.gptq_topk,
        "gptq_damping": args.gptq_damping,
        "layers_refined": sorted(upgraded),
    }
    cost_blob["meta"] = meta
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(cost_blob, f)
    print(f"[reconstruct] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
