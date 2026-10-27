#!/usr/bin/env python3
"""Build BF16-promotion assignment overlays for sensitivity calibration.

The output is intentionally just assignment JSON plus a manifest.  KL replay
stays in ``prismaquant.validate_assignments_kl`` so rendered weights and
activation perturbations use the production cache path.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from prismaquant import format_registry as fr
from prismaquant.priced_cell import cost_entry_predicted_dloss
from prismaquant.kl_measurement import assignment_bit_total
from prismaquant.layer_config import load_assignment
from prismaquant.schemas import validate_cost_payload


_LAYER_RE = re.compile(r"(?:^|[.])layers[.](\d+)[.]")
_VISUAL_BLOCK_RE = re.compile(r"(?:^|[.])visual[.]blocks[.](\d+)[.]")


def _load_pickle_mapping(path: str | Path, key: str) -> dict:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a mapping")
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} did not contain mapping key {key!r}")
    return dict(value)


def _load_costs(path: str | Path) -> dict:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    validate_cost_payload(payload, str(path))
    return dict(payload["costs"])


def _category(name: str) -> str:
    if ".visual." in name or name.startswith("model.visual."):
        return "visual"
    if ".mlp.shared_expert." in name or name.endswith(".mlp.shared_expert_gate"):
        return "shared_expert"
    if ".self_attn." in name:
        return "self_attn"
    if ".linear_attn." in name:
        return "linear_attn"
    if ".mlp.experts." in name:
        return "routed_experts"
    if name.startswith("mtp."):
        return "mtp"
    return "other"


def _layer_number(name: str) -> str:
    match = _LAYER_RE.search(name)
    if match:
        return match.group(1)
    match = _VISUAL_BLOCK_RE.search(name)
    if match:
        return match.group(1)
    return "unknown"


def _group_key(name: str) -> str:
    cat = _category(name)
    layer = _layer_number(name)
    if cat in {"shared_expert", "self_attn", "linear_attn", "routed_experts"}:
        return f"{cat}.layer_{layer}"
    if cat == "visual":
        if ".attn." in name:
            return f"visual.block_{layer}.attn"
        if ".mlp." in name:
            return f"visual.block_{layer}.mlp"
        if ".merger." in name:
            return "visual.merger"
        return f"visual.block_{layer}"
    return cat


def _cost_entry(costs: Mapping, name: str, fmt: str) -> Mapping | None:
    per_name = costs.get(name)
    if not isinstance(per_name, Mapping):
        return None
    seen: set[str] = set()
    for candidate in (fmt, fr.canonical_format_name(fmt), *fr.aliases_for(fmt)):
        key = str(candidate).strip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        entry = per_name.get(key)
        if isinstance(entry, Mapping) and "error" not in entry:
            return entry
    return None


def _specs_for_assignment(assignment: Mapping[str, str]) -> dict[str, fr.FormatSpec]:
    specs: dict[str, fr.FormatSpec] = {}
    for fmt in set(assignment.values()) | {"BF16", "NVFP4", "MXFP8_E4M3", "FP8_E4M3"}:
        try:
            spec = fr.get_format(fmt)
        except Exception:
            continue
        specs[spec.name] = spec
        specs[fr.canonical_format_name(spec.name)] = spec
    return specs


def _bpp(stats: Mapping, assignment: Mapping[str, str]) -> float:
    names = [name for name in assignment if name in stats]
    params = sum(int(stats[name].get("n_params", 0) or 0) for name in names)
    if params <= 0:
        return 0.0
    specs = _specs_for_assignment(assignment)
    return assignment_bit_total(
        stats,
        {name: assignment[name] for name in names},
        specs,
    ) / float(params)


def _summarize_names(
    names: Sequence[str],
    *,
    assignment: Mapping[str, str],
    stats: Mapping,
    costs: Mapping,
) -> dict[str, object]:
    predicted_saved = 0.0
    params = 0
    by_format: dict[str, int] = defaultdict(int)
    missing_cost: list[str] = []
    for name in names:
        fmt = fr.canonical_format_name(assignment[name])
        by_format[fmt] += 1
        params += int(stats.get(name, {}).get("n_params", 0) or 0)
        if fmt == "BF16":
            continue
        entry = _cost_entry(costs, name, fmt)
        if entry is None or name not in stats:
            missing_cost.append(name)
            continue
        predicted_saved += cost_entry_predicted_dloss(dict(stats[name]), dict(entry))
    return {
        "entry_count": len(names),
        "non_bf16_count": sum(
            1 for name in names if fr.canonical_format_name(assignment[name]) != "BF16"
        ),
        "params": int(params),
        "formats": dict(sorted(by_format.items())),
        "predicted_dloss_saved": float(predicted_saved),
        "missing_cost_count": len(missing_cost),
        "missing_cost_sample": missing_cost[:8],
    }


def _write_overlay(path: Path, names: Sequence[str]) -> None:
    payload = {name: "BF16" for name in sorted(names)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate BF16-promotion overlays for KL sensitivity replay."
    )
    parser.add_argument("--base-assignment", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--costs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--categories",
        default="shared_expert,self_attn,linear_attn,routed_experts",
        help="Comma-separated categories for class and group overlays.",
    )
    parser.add_argument("--top-groups-per-category", type=int, default=4)
    args = parser.parse_args(argv)

    assignment = load_assignment(args.base_assignment)
    stats = _load_pickle_mapping(args.probe, "stats")
    costs = _load_costs(args.costs)
    output_dir = Path(args.output_dir)
    assignment_dir = output_dir / "assignments"
    assignment_dir.mkdir(parents=True, exist_ok=True)

    base_bpp = _bpp(stats, assignment)
    categories = [part.strip() for part in args.categories.split(",") if part.strip()]
    quantized_by_category: dict[str, list[str]] = defaultdict(list)
    quantized_by_group: dict[str, list[str]] = defaultdict(list)
    for name, fmt in assignment.items():
        fmt_canon = fr.canonical_format_name(fmt)
        if fmt_canon == "BF16":
            continue
        cat = _category(name)
        if cat not in categories:
            continue
        quantized_by_category[cat].append(name)
        quantized_by_group[_group_key(name)].append(name)

    variants: list[dict[str, object]] = []

    baseline_path = assignment_dir / "baseline.json"
    baseline_path.write_text("{}\n")
    variants.append({
        "label": "baseline",
        "kind": "baseline",
        "category": "baseline",
        "path": str(baseline_path),
        "bpp_base": float(base_bpp),
        "bpp_after_promotion": float(base_bpp),
        "bpp_delta": 0.0,
        "entry_count": 0,
        "non_bf16_count": 0,
        "params": 0,
        "formats": {},
        "predicted_dloss_saved": 0.0,
        "missing_cost_count": 0,
        "missing_cost_sample": [],
    })

    def add_variant(label: str, names: Sequence[str], kind: str, category: str) -> None:
        if not names:
            return
        overlay_path = assignment_dir / f"{label}.json"
        _write_overlay(overlay_path, names)
        promoted = dict(assignment)
        for name in names:
            promoted[name] = "BF16"
        summary = _summarize_names(
            names,
            assignment=assignment,
            stats=stats,
            costs=costs,
        )
        variants.append({
            "label": label,
            "kind": kind,
            "category": category,
            "path": str(overlay_path),
            "bpp_base": float(base_bpp),
            "bpp_after_promotion": float(_bpp(stats, promoted)),
            "bpp_delta": float(_bpp(stats, promoted) - base_bpp),
            **summary,
        })

    for cat in categories:
        add_variant(
            f"promote_all_{cat}_to_bf16",
            sorted(quantized_by_category.get(cat, [])),
            "category",
            cat,
        )

    groups_by_category: dict[str, list[tuple[str, list[str], float]]] = defaultdict(list)
    for group, names in quantized_by_group.items():
        cat = group.split(".", 1)[0]
        summary = _summarize_names(
            names,
            assignment=assignment,
            stats=stats,
            costs=costs,
        )
        groups_by_category[cat].append(
            (group, sorted(names), float(summary["predicted_dloss_saved"]))
        )

    limit = max(int(args.top_groups_per_category), 0)
    for cat in categories:
        rows = sorted(groups_by_category.get(cat, []), key=lambda row: row[2], reverse=True)
        for group, names, _ in rows[:limit]:
            label = "promote_" + group.replace(".", "_") + "_to_bf16"
            add_variant(label, names, "group", cat)

    manifest = {
        "base_assignment": str(args.base_assignment),
        "probe": str(args.probe),
        "costs": str(args.costs),
        "base_bpp": float(base_bpp),
        "categories": categories,
        "top_groups_per_category": int(args.top_groups_per_category),
        "variants": variants,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")
    for variant in variants:
        print(
            f"{variant['label']}: changed={variant['non_bf16_count']} "
            f"bpp_delta={variant['bpp_delta']:.6f} "
            f"pred_saved={variant['predicted_dloss_saved']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
