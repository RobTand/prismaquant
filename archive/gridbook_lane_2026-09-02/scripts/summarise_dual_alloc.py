#!/usr/bin/env python3
"""Summarise the sanctioned b/c x 92/88 GB research allocation grid."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

MTP_BYTES = 10_862_838_300
BASELINES = {"b-92": 5376.61, "c-92": 1325.83}


def fmt_of(cfg: dict) -> str:
    dtype = cfg.get("data_type")
    if dtype == "nvfp4_cb":
        return f"NVFP4_CB_K{cfg['cb_k']}"
    if dtype == "fp8_cb":
        return f"FP8_CB_K{cfg['cb_k']}"
    if dtype == "fp4_e2m1":
        return "MXFP4_SOURCE"
    if dtype == "fp8_e4m3" and cfg.get("scale_fmt") == "ue8m0":
        return "FP8_BLOCK_UE8M0_SOURCE"
    return str(dtype)


def cell(root: Path, name: str) -> dict:
    selection = json.loads((root / name / "selection.json").read_text())
    config = json.loads((root / name / "layer_config.json").read_text())
    experts: dict[int, set[str]] = defaultdict(set)
    body: Counter[str] = Counter()
    for qname, cfg in config.items():
        if qname == "__prismaquant__":
            continue
        fmt = fmt_of(cfg)
        match = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.", qname)
        if match:
            experts[int(match.group(1))].add(fmt)
        else:
            body[fmt] += 1
    expert_map: dict[str, list[int]] = defaultdict(list)
    for layer, formats in sorted(experts.items()):
        if len(formats) != 1:
            raise ValueError(f"{name}: expert layer {layer} has {sorted(formats)}")
        expert_map[next(iter(formats))].append(layer)
    actual_gb = float(selection["predicted_whole_artifact_upper_bound_gb"])
    if name.startswith("c-"):
        actual_gb -= MTP_BYTES / 1e9
    baseline_key = "c-92" if name.startswith("c-") else "b-92"
    dloss = float(selection["predicted_dloss"])
    cost_provenance = selection.get("cost_provenance")
    if isinstance(cost_provenance, dict):
        cost_provenance = cost_provenance.get("cost_provenance")

    return {
        "predicted_dloss": dloss,
        "versus_oldmenu_92": dloss - BASELINES[baseline_key],
        "achieved_bits": float(selection["chosen_achieved_bits"]),
        "reported_artifact_gb": actual_gb,
        "effective_selection_gb": float(
            selection["predicted_whole_artifact_upper_bound_gb"]
        ),
        "byte_gate_passed": bool(selection["feasible"]),
        "selection_headroom_gb": float(selection["selection_headroom_gb"]),
        "expert_map": dict(sorted(expert_map.items())),
        "body_split": dict(body.most_common()),
        "cost_provenance": cost_provenance,
    }


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "research-alloc")
    cells = {name: cell(root, name) for name in ("b-92", "b-88", "c-92", "c-88")}
    for variant in ("b", "c"):
        high, low = cells[f"{variant}-92"], cells[f"{variant}-88"]
        delta = low["predicted_dloss"] - high["predicted_dloss"]
        freed_gb = high["reported_artifact_gb"] - low["reported_artifact_gb"]
        low["marginal_shave_price_from_92"] = {
            "dloss": delta,
            "achieved_gb_freed": freed_gb,
            "dloss_per_gb_freed": delta / freed_gb,
        }
        high["marginal_shave_price_to_88"] = dict(
            low["marginal_shave_price_from_92"]
        )
    output = {
        "schema": "prismaquant.research_accepted_allocation_grid.v1",
        "oldmenu_baselines": BASELINES,
        "mtp_bytes_released_in_c": MTP_BYTES,
        "cells": cells,
    }
    (root / "SUMMARY.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
