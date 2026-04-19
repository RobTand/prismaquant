#!/usr/bin/env python3
"""oracle_search.py — exact local search for tiny-model mixed-format recipes.

Given an interaction payload produced by measure_interactions.py, exhaustively
enumerate all feasible assignments over the selected refinement units and score
them with actual KL on the live model. This is intentionally slow and should
only be used on very small models / very small local neighborhoods.

This provides a concrete oracle against which future heuristic refinements can
be compared.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
from pathlib import Path

import torch

from .build_rtn_cache import (
    cache_reference_log_probs,
    load_wikitext_calibration,
    stage_multimodal,
)
from prismquant.calibrate_allocator import (
    apply_recipe_in_place,
    build_module_param_map,
    install_activation_hooks,
    measure_avg_last_token_kl,
    restore_in_place,
)
from prismquant.interaction_refine import RefinementUnit, UnitOption, expand_unit_assignment


def _load_units(payload: dict):
    units = []
    allowed = {}
    for row in payload["selected_units"]:
        options = tuple(
            UnitOption(
                fmt=opt["fmt"],
                bits_total=float(opt["bits_total"]),
                predicted_dloss=float(opt["predicted_dloss"]),
            )
            for opt in row["options"]
        )
        base_member_fmts = tuple(tuple(x) for x in row.get("base_member_fmts", []))
        if not base_member_fmts:
            base_member_fmts = tuple((member, row["base_fmt"]) for member in row["members"])
        unit = RefinementUnit(
            key=row["key"],
            members=tuple(row["members"]),
            base_fmt=row["base_fmt"],
            base_member_fmts=base_member_fmts,
            options=options,
        )
        units.append(unit)
        allowed[unit.key] = tuple(opt for opt, raw in zip(options, row["options"]) if raw.get("allowed", True))
    return units, allowed


def enumerate_feasible_choices(
    units: list[RefinementUnit],
    allowed: dict[str, tuple[UnitOption, ...]],
    target_total_bits: float,
    fixed_bits_total: float,
):
    keys = [unit.key for unit in units]
    option_lists = [allowed[unit.key] for unit in units]
    for combo in itertools.product(*option_lists):
        bits_total = fixed_bits_total + sum(opt.bits_total for opt in combo)
        if bits_total > target_total_bits + 1e-6:
            continue
        yield {key: opt.fmt for key, opt in zip(keys, combo)}, bits_total


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
        restore_in_place(originals)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return actual_kl, active_hooks, skipped_hooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-calib-samples", type=int, default=2)
    ap.add_argument("--calib-seqlen", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-combos", type=int, default=4096)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.interactions) as f:
        payload = json.load(f)

    units, allowed = _load_units(payload)
    total_combos = math.prod(len(allowed[unit.key]) for unit in units)
    if total_combos > args.max_combos:
        raise SystemExit(
            f"oracle search would enumerate {total_combos} assignments; "
            f"raise --max-combos if you really want this"
        )

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

        results = []
        best = None
        for idx, (choices, bits_total) in enumerate(
            enumerate_feasible_choices(
                units,
                allowed,
                max(float(payload["target_total_bits"]), float(payload.get("base_total_bits", 0.0))),
                float(payload["fixed_bits_total"]),
            ),
            start=1,
        ):
            assignment = dict(payload["base_assignment"])
            assignment.update(expand_unit_assignment(units, choices))
            if payload["expert_granularity"] == "layer":
                from prismquant.allocator import expand_moe_assignment

                stats_alloc = {}
                for row in payload["selected_units"]:
                    for member in row["members"]:
                        stats_alloc[member] = {"_fused_members": []}
                assignment_expanded = assignment
            else:
                assignment_expanded = assignment
            kl, active_hooks, skipped_hooks = _measure_recipe(
                model, quant_map, calib_ids, ref_log_probs, device, assignment_expanded
            )
            result = {
                "index": idx,
                "choices": choices,
                "bits_total": bits_total,
                "bits_per_param": bits_total / max(float(payload["total_params"]), 1.0),
                "actual_last_token_kl": kl,
                "activation_hook_count": len(active_hooks),
                "activation_hook_skipped": skipped_hooks,
            }
            results.append(result)
            if best is None or kl < best["actual_last_token_kl"]:
                best = result
            print(
                f"[oracle] {idx}/{total_combos} bits={result['bits_per_param']:.4f} kl={kl:.4e}",
                flush=True,
            )

        frontier = []
        for result in sorted(results, key=lambda r: (r["bits_per_param"], r["actual_last_token_kl"])):
            if not frontier or result["actual_last_token_kl"] < frontier[-1]["actual_last_token_kl"] - 1e-12:
                frontier.append(result)

        out = {
            "source": args.interactions,
            "total_combos": total_combos,
            "best": best,
            "frontier": frontier,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[oracle] wrote {args.output}", flush=True)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
