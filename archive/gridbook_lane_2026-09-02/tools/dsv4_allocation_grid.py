#!/usr/bin/env python3
"""Run the DSV4 collapsed and per-expert allocation study grid."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cb_layout import codebook_subtable_shapes
from prismaquant.layer_config import load_assignment
from prismaquant.tier2_per_expert_counterfactual import run_counterfactual
from tools.dsv4_ldlq_burn import (
    BURN_ROOT,
    COL_WEIGHTS,
    OVERHEAD_RESERVE_BYTES,
    PILOT_JSON,
    PROBE,
    RUNGS,
    SOURCE,
    atomic_json,
)


GRID_ROOT = BURN_ROOT / "allocation-grid"
MTP_BF16_BYTES = 10_862_838_300
MENU = tuple([
    *[f"NVFP4_CB_K{k}" for k in range(12, 19)],
    *[f"FP8_CB_K{k}" for k in RUNGS],
    "MXFP4_SOURCE", "FP8_BLOCK_UE8M0_SOURCE", "BF16",
])
PARETO_TARGETS = (
    "1.78,1.80,1.82,1.85,1.88,1.90,1.92,1.95,1.98,2.00,2.02,"
    "2.04,2.06,2.08,2.10,2.12,2.14,2.15,2.16,2.17,2.18,2.19,"
    "2.20,2.22,2.25,2.30,2.35,2.40,2.50,2.65,2.80,3.00,3.25,"
    "3.50,4.00,4.50,5.00,6.00"
)


def _unique_fp8_sidecar_bytes(assignment: Mapping[str, str], strict: bool) -> int:
    if not strict:
        return 0
    total = 0
    for fmt in assignment.values():
        if not fmt.startswith("FP8_CB_K"):
            continue
        k = int(fmt.rsplit("K", 1)[1])
        # Conservative chain-production gate: every selected FP8 tensor gets
        # a distinct flat FP16 table bundle. This safely covers unknown future
        # refine winners on law-predicted cells. The lattice sidecar already
        # counted by the allocator is deliberately not subtracted.
        total += sum(rows * dim * 2 for rows, dim in codebook_subtable_shapes(
            k, "product", 4
        ))
    return total


def _tee_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"allocator exited {code}; see {log_path}")


def _allocator_command(out: Path, budget_bytes: int) -> list[str]:
    return [
        sys.executable, "-m", "prismaquant.allocator",
        "--probe", str(PROBE),
        "--costs", str(BURN_ROOT / "cost_merged.pkl"),
        "--accept-research-cost-table",
        "--model-override", str(SOURCE),
        "--target-profile", "nvfp4_cb",
        "--target-disk-gb", f"{budget_bytes / 1e9:.9f}",
        "--artifact-overhead-reserve-bytes", str(OVERHEAD_RESERVE_BYTES),
        "--cb-scale-coding", "two_tier",
        "--cb-codebook-source", "lattice",
        "--cb-scale-sweep", "1",
        "--cb-ldlq", "1",
        "--cb-encode-tier", "balanced",
        "--cb-col-weights", str(COL_WEIGHTS),
        "--formats", ",".join(MENU),
        "--mtp-format", "BF16",
        "--visual-format", "BF16",
        "--threads", "16",
        "--pareto-targets", PARETO_TARGETS,
        "--layer-config", str(out / "layer_config.json"),
        "--pareto-csv", str(out / "pareto.csv"),
        "--pareto-output-dir", str(out / "pareto-points"),
        "--applicability-report", str(out / "format_applicability.json"),
        "--bit-attribution-json", str(out / "bit_attribution.json"),
        "--bit-attribution-csv", str(out / "bit_attribution.csv"),
    ]


def _collapsed_cell(name: str, requested_budget: int, strict: bool) -> dict:
    out = GRID_ROOT / name / "collapsed"
    out.mkdir(parents=True, exist_ok=True)
    effective = requested_budget
    attempts = []
    for attempt in range(1, 7):
        print(
            f"[grid] {name} collapsed attempt {attempt} effective="
            f"{effective / 1e9:.6f}GB", flush=True,
        )
        _tee_command(
            _allocator_command(out, effective),
            out / f"allocator-attempt-{attempt}.log",
        )
        selection = json.loads((out / "selection.json").read_text())
        assignment = load_assignment(out / "layer_config.json")
        extra = _unique_fp8_sidecar_bytes(assignment, strict)
        raw = int(selection["whole_artifact_budget"][
            "selection_whole_artifact_upper_bound_bytes"
        ])
        adjusted = raw + extra
        attempts.append({
            "attempt": attempt, "effective_budget_bytes": effective,
            "raw_upper_bound_bytes": raw,
            "conservative_unique_chain_sidecar_bytes": extra,
            "adjusted_upper_bound_bytes": adjusted,
        })
        if adjusted <= requested_budget:
            gate = {
                "pass": True, "requested_budget_bytes": requested_budget,
                "effective_allocator_budget_bytes": effective,
                "raw_upper_bound_bytes": raw,
                "conservative_unique_chain_sidecar_bytes": extra,
                "adjusted_upper_bound_bytes": adjusted,
                "headroom_bytes": requested_budget - adjusted,
                "attempts": attempts,
            }
            atomic_json(out / "BYTE_GATE.json", gate)
            return {"selection": selection, "byte_gate": gate, "directory": str(out)}
        new_effective = requested_budget - extra
        if new_effective <= 0 or new_effective >= effective:
            break
        effective = new_effective
    gate = {
        "pass": False, "requested_budget_bytes": requested_budget,
        "attempts": attempts,
    }
    atomic_json(out / "BYTE_GATE.json", gate)
    raise RuntimeError(f"{name}: collapsed byte gate failed")


def _per_expert_cell(
    name: str, requested_budget: int, strict: bool,
) -> dict:
    baseline = GRID_ROOT / name / "collapsed"
    out = GRID_ROOT / name / "per-expert"
    out.mkdir(parents=True, exist_ok=True)
    effective = requested_budget
    attempts = []
    result: dict[str, Any] | None = None
    for attempt in range(1, 7):
        print(
            f"[grid] {name} per-expert attempt {attempt} effective="
            f"{effective / 1e9:.6f}GB", flush=True,
        )
        result = run_counterfactual(
            baseline_dir=baseline,
            probe_path=PROBE,
            cost_path=BURN_ROOT / "cost_merged.pkl",
            budget_bytes=effective,
            menu=MENU,
            floor_unmeasured=False,
        )
        extra = _unique_fp8_sidecar_bytes(result["assignment"], strict)
        raw = int(result["exact_bytes"])
        adjusted = raw + extra
        attempts.append({
            "attempt": attempt, "effective_budget_bytes": effective,
            "raw_exact_bytes": raw,
            "conservative_unique_chain_sidecar_bytes": extra,
            "adjusted_exact_bytes": adjusted,
        })
        if adjusted <= requested_budget:
            result["requested_budget_bytes"] = requested_budget
            result["effective_solver_budget_bytes"] = effective
            result["conservative_unique_chain_sidecar_bytes"] = extra
            result["adjusted_exact_bytes"] = adjusted
            result["adjusted_headroom_bytes"] = requested_budget - adjusted
            result["adjusted_bytes_gate"] = True
            result["byte_gate_attempts"] = attempts
            atomic_json(out / "result.json", result)
            return result
        new_effective = requested_budget - extra
        if new_effective <= 0 or new_effective >= effective:
            break
        effective = new_effective
    atomic_json(out / "BYTE_GATE_FAIL.json", {
        "requested_budget_bytes": requested_budget, "attempts": attempts,
    })
    raise RuntimeError(f"{name}: per-expert byte gate failed")


def _knee_budget(reference: Mapping[str, Any]) -> tuple[int, dict]:
    directory = Path(reference["directory"])
    knees = json.loads((directory / "pareto.knees.json").read_text())
    knee = dict(knees[knees["primary"]])
    target = float(knee["target_bits"])
    selection = reference["selection"]
    matches = [
        point for point in selection["grid"]
        if abs(float(point["target_bits"]) - target) < 1e-9
    ]
    if len(matches) != 1:
        raise AssertionError(f"knee target {target} absent/duplicate in grid")
    budget = int(round(float(matches[0]["whole_artifact_upper_bound_gb"]) * 1e9))
    return budget, {"knee": knee, "source_point": matches[0]}


def run_grid() -> int:
    pilot = json.loads(PILOT_JSON.read_text())
    strict = pilot["burn_method"] == "MIN-CHAIN STRICT"
    cells: dict[str, Any] = {}
    requested = {
        "b-88": 88_000_000_000,
        "b-92": 92_000_000_000,
        "b-97": 97_000_000_000,
        "c-92-mtp": 102_862_838_300,
    }
    for name, budget in requested.items():
        collapsed = _collapsed_cell(name, budget, strict)
        per_expert = _per_expert_cell(name, budget, strict)
        cells[name] = {
            "requested_budget_bytes": budget,
            "mtp_bf16_bytes": MTP_BF16_BYTES if name == "c-92-mtp" else 0,
            "collapsed": collapsed,
            "per_expert": per_expert,
        }
    knee_budget, knee_source = _knee_budget(cells["b-92"]["collapsed"])
    name = "knee"
    collapsed = _collapsed_cell(name, knee_budget, strict)
    per_expert = _per_expert_cell(name, knee_budget, strict)
    cells[name] = {
        "requested_budget_bytes": knee_budget,
        "mtp_bf16_bytes": 0,
        "knee_source": knee_source,
        "collapsed": collapsed,
        "per_expert": per_expert,
    }
    report = {
        "schema": "prismaquant.dsv4_allocation_grid.v1",
        "method": pilot["burn_method"],
        "menu": list(MENU),
        "mtp_bf16_bytes": MTP_BF16_BYTES,
        "mtp_accounting_note": (
            "c-92-mtp uses the registered decimal-byte point; MTP remains "
            "fixed BF16 and excluded from quantizable-parameter bpp"
        ),
        "unique_chain_sidecar_policy": (
            "conservative one distinct flat FP16 bundle per FP8 assignment"
            if strict else "lattice shared sidecars"
        ),
        "cells": cells,
        "per_expert_solver_caveat": (
            "Lagrangian candidate generation plus exact-payload tidy is not "
            "a proof of global discrete-knapsack optimality; mixed-stack "
            "runtime overhead is unpriced."
        ),
    }
    atomic_json(GRID_ROOT / "GRID_REPORT.json", report)
    print(f"[grid] PASS -> {GRID_ROOT / 'GRID_REPORT.json'}", flush=True)
    return 0


def main() -> int:
    argparse.ArgumentParser().parse_args()
    return run_grid()


if __name__ == "__main__":
    raise SystemExit(main())
