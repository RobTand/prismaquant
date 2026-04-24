#!/usr/bin/env python3
"""measure_interactions.py — sparse pairwise interaction probing for PrismaQuant.

Build a base additive assignment at a target bit budget, select the most
important refinement units near that point, and measure actual KL deltas for
single-unit and pairwise deviations from the base recipe.

The output is designed to feed quadratic_refine_allocator.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from prismaquant import format_registry as fr
from prismaquant.allocator import (
    aggregate_moe_candidates,
    build_candidates,
    compute_achieved,
    expand_moe_assignment,
    promote_fused,
    solve_allocation,
)
from prismaquant.calibrate_allocator import (
    apply_recipe_in_place,
    build_module_param_map,
    install_activation_hooks,
    load_inputs,
    measure_avg_last_token_kl,
)
from prismaquant.interaction_refine import (
    build_refinement_units,
    expand_unit_assignment,
    make_pair_key,
    neighborhood_options,
    select_critical_units,
)
from prismaquant.build_rtn_cache import (
    cache_reference_log_probs,
    load_wikitext_calibration,
    stage_multimodal,
)


def _shape_bits_total(stats: dict, assignment: dict[str, str]) -> tuple[float, int]:
    total_bits = 0.0
    total_params = 0
    for name, fmt in assignment.items():
        spec = fr.get_format(fmt)
        n_params = stats[name]["n_params"]
        total_bits += spec.effective_bits_for_shape((
            stats[name].get("out_features", n_params),
            stats[name].get("in_features", 1),
        )) * n_params
        total_params += n_params
    return total_bits, total_params


def _predicted_dloss_total(stats: dict, costs: dict, assignment: dict[str, str]) -> float:
    total = 0.0
    for name, fmt in assignment.items():
        entry = costs[name].get(fmt, {})
        total += 0.5 * stats[name]["h_trace"] * entry.get("output_mse", 0.0) * stats[name]["out_features"]
    return total


def _measure_recipe(
    model,
    quant_map,
    calib_ids,
    ref_log_probs,
    device,
    assignment_expanded: dict[str, str],
):
    originals = apply_recipe_in_place(model, assignment_expanded, quant_map)
    hook_handles, active_hooks, skipped_hooks = install_activation_hooks(
        assignment_expanded, quant_map
    )
    try:
        actual_kl = measure_avg_last_token_kl(model, calib_ids, ref_log_probs, device)
    finally:
        for handle in hook_handles:
            handle.remove()
        from prismaquant.calibrate_allocator import restore_in_place

        restore_in_place(originals)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return actual_kl, active_hooks, skipped_hooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--formats", required=True)
    ap.add_argument("--target-bits", type=float, required=True)
    ap.add_argument("--top-units", type=int, default=16)
    ap.add_argument("--unit-scope", choices=["sibling", "block", "hybrid", "layer"], default="sibling")
    ap.add_argument("--neighbor-radius", type=int, default=1)
    ap.add_argument("--n-calib-samples", type=int, default=4)
    ap.add_argument("--calib-seqlen", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bit-precision", type=float, default=0.001)
    ap.add_argument("--expert-granularity", choices=["layer", "expert"], default="layer")
    ap.add_argument("--no-fused-promote", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    fmt_names = [s.strip() for s in args.formats.split(",") if s.strip()]
    stats, costs, specs_sorted = load_inputs(Path(args.probe), Path(args.costs), fmt_names)
    format_rank = {s.name: i for i, s in enumerate(specs_sorted)}
    format_specs = {s.name: s for s in specs_sorted}

    candidates = build_candidates(stats, costs, specs_sorted)
    stats_alloc = stats
    if args.expert_granularity == "layer":
        stats_alloc, costs, candidates = aggregate_moe_candidates(
            stats, costs, specs_sorted, candidates
        )

    result = solve_allocation(stats_alloc, candidates, args.target_bits, args.bit_precision)
    if result is None:
        raise SystemExit("no feasible assignment at requested target")
    assignment, _chosen = result
    if not args.no_fused_promote:
        assignment = promote_fused(assignment, format_rank)

    achieved_bits, _ = compute_achieved(stats_alloc, assignment, format_specs)
    units = build_refinement_units(stats_alloc, candidates, assignment, unit_scope=args.unit_scope)
    selected_units = select_critical_units(units, args.top_units)
    allowed = {unit.key: neighborhood_options(unit, args.neighbor_radius) for unit in selected_units}

    total_params = sum(stats_alloc[name]["n_params"] for name in assignment)
    target_total_bits = args.target_bits * total_params
    base_total_bits, _ = _shape_bits_total(stats_alloc, assignment)
    selected_member_keys = {member for unit in selected_units for member in unit.members}
    fixed_bits_total = 0.0
    fixed_predicted_dloss_total = 0.0
    for name, fmt in assignment.items():
        if name in selected_member_keys:
            continue
        spec = fr.get_format(fmt)
        shape = (
            stats_alloc[name].get("out_features", stats_alloc[name]["n_params"]),
            stats_alloc[name].get("in_features", 1),
        )
        fixed_bits_total += spec.effective_bits_for_shape(shape) * stats_alloc[name]["n_params"]
        entry = costs[name].get(fmt, {})
        fixed_predicted_dloss_total += (
            0.5 * stats_alloc[name]["h_trace"] * entry.get("output_mse", 0.0) * stats_alloc[name]["out_features"]
        )
    base_predicted_dloss = _predicted_dloss_total(stats_alloc, costs, assignment)

    model_arg = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    staged, cleanup = stage_multimodal(model_arg)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if args.device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = args.device
        device = torch.device(device_str)
        load_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer_kwargs = dict(trust_remote_code=True)
        if Path(staged).exists():
            load_kwargs["local_files_only"] = True
            tokenizer_kwargs["local_files_only"] = True
        if device.type == "cuda":
            load_kwargs["device_map"] = device_str

        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        if device.type != "cuda":
            model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(staged, **tokenizer_kwargs)
        quant_map = build_module_param_map(model)
        calib_ids = load_wikitext_calibration(tokenizer, args.n_calib_samples, args.calib_seqlen)
        ref_log_probs = cache_reference_log_probs(model, calib_ids, device)

        base_expanded = (
            expand_moe_assignment(assignment, stats_alloc)
            if args.expert_granularity == "layer"
            else assignment
        )
        base_kl, base_hooks, base_skipped = _measure_recipe(
            model, quant_map, calib_ids, ref_log_probs, device, base_expanded
        )

        unary = {unit.key: {unit.base_fmt: 0.0} for unit in selected_units}
        single_records = []
        for unit in selected_units:
            for opt in allowed[unit.key]:
                if opt.fmt == unit.base_fmt:
                    continue
                unit_choice = {unit.key: opt.fmt}
                override = expand_unit_assignment(selected_units, unit_choice)
                trial = dict(assignment)
                trial.update(override)
                trial_expanded = (
                    expand_moe_assignment(trial, stats_alloc)
                    if args.expert_granularity == "layer"
                    else trial
                )
                kl, active_hooks, skipped_hooks = _measure_recipe(
                    model, quant_map, calib_ids, ref_log_probs, device, trial_expanded
                )
                delta = kl - base_kl
                unary[unit.key][opt.fmt] = delta
                single_records.append(
                    {
                        "unit": unit.key,
                        "fmt": opt.fmt,
                        "actual_last_token_kl": kl,
                        "delta_kl": delta,
                        "activation_hook_count": len(active_hooks),
                        "activation_hook_skipped": skipped_hooks,
                    }
                )
                print(f"[int] single {unit.key} -> {opt.fmt} delta={delta:.4e}", flush=True)

        pairwise = {}
        pair_records = []
        for idx, left in enumerate(selected_units):
            for right in selected_units[idx + 1 :]:
                for left_opt in allowed[left.key]:
                    for right_opt in allowed[right.key]:
                        if left_opt.fmt == left.base_fmt or right_opt.fmt == right.base_fmt:
                            continue
                        unit_choice = {left.key: left_opt.fmt, right.key: right_opt.fmt}
                        override = expand_unit_assignment(selected_units, unit_choice)
                        trial = dict(assignment)
                        trial.update(override)
                        trial_expanded = (
                            expand_moe_assignment(trial, stats_alloc)
                            if args.expert_granularity == "layer"
                            else trial
                        )
                        kl, active_hooks, skipped_hooks = _measure_recipe(
                            model, quant_map, calib_ids, ref_log_probs, device, trial_expanded
                        )
                        interaction = kl - base_kl - unary[left.key][left_opt.fmt] - unary[right.key][right_opt.fmt]
                        key = make_pair_key(left.key, left_opt.fmt, right.key, right_opt.fmt)
                        pairwise[key] = interaction
                        pair_records.append(
                            {
                                "left_unit": left.key,
                                "left_fmt": left_opt.fmt,
                                "right_unit": right.key,
                                "right_fmt": right_opt.fmt,
                                "actual_last_token_kl": kl,
                                "interaction_delta": interaction,
                                "activation_hook_count": len(active_hooks),
                                "activation_hook_skipped": skipped_hooks,
                            }
                        )
                        print(
                            f"[int] pair {left.key}->{left_opt.fmt} + {right.key}->{right_opt.fmt} "
                            f"interaction={interaction:.4e}",
                            flush=True,
                        )

        payload = {
            "model": args.model,
            "formats": fmt_names,
            "target_bits": args.target_bits,
            "achieved_bits_base": achieved_bits,
            "target_total_bits": target_total_bits,
            "base_total_bits": base_total_bits,
            "fixed_bits_total": fixed_bits_total,
            "base_predicted_dloss": base_predicted_dloss,
            "fixed_predicted_dloss_total": fixed_predicted_dloss_total,
            "total_params": total_params,
            "expert_granularity": args.expert_granularity,
            "base_assignment": assignment,
            "base_last_token_kl": base_kl,
            "base_activation_hook_count": len(base_hooks),
            "base_activation_hook_skipped": base_skipped,
            "selected_units": [
                {
                    "key": unit.key,
                    "members": list(unit.members),
                    "base_fmt": unit.base_fmt,
                    "base_member_fmts": list(unit.base_member_fmts),
                    "options": [
                        {
                            "fmt": opt.fmt,
                            "bits_total": opt.bits_total,
                            "predicted_dloss": opt.predicted_dloss,
                            "allowed": opt in allowed[unit.key],
                        }
                        for opt in unit.options
                    ],
                }
                for unit in selected_units
            ],
            "unit_scope": args.unit_scope,
            "unary": unary,
            "pairwise": [
                {
                    "left_unit": key[0],
                    "left_fmt": key[1],
                    "right_unit": key[2],
                    "right_fmt": key[3],
                    "interaction_delta": value,
                }
                for key, value in sorted(pairwise.items())
            ],
            "single_records": single_records,
            "pair_records": pair_records,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[int] wrote {args.output}", flush=True)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
