#!/usr/bin/env python3
"""Collapsed/per-expert A-FAST allocation grid with MTP-in/out accounting."""
from __future__ import annotations

import argparse
import hashlib
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
from tools.dsv4_afast_burn import (
    BURN_ROOT,
    COL_WEIGHTS,
    MENU,
    MTP_BF16_BYTES,
    RUNGS,
    SOURCE,
    atomic_json,
    sha256_file,
)


GRID_ROOT = BURN_ROOT / "allocation-grid"
# The menu the allocator consumes is `cost_merged.pkl` PLUS the dense FP8
# completion: the 301 dense tensors carried only FP8_CB_K36, and that lone row
# was pre-LDLQ (ldlq=False) while every routed-expert FP8_CB row and this
# tool's own --cb-ldlq 1 stamp are ldlq=True. `cost_merged_dense_complete.pkl`
# adds K28-K38 at ldlq=1 and supersedes the off-basis K36, closing 3010
# coverage holes the allocator refuses to run with. It is a separate file, not
# a patch, because `dsv4_afast_burn.py merge` regenerates cost_merged.pkl from
# the shards and would silently revert an in-place edit.
# See DENSE_COVERAGE_GAP.md / DENSE_FP8_COMPOSITION.json.
COST_TABLE = BURN_ROOT / "cost_merged_dense_complete.pkl"
PROBE = Path(
    "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/"
    "artifacts-mxfp4/probe.pkl"
)
OVERHEAD_RESERVE_BYTES = 268_435_456
PARETO_TARGETS = (
    "1.78,1.80,1.82,1.85,1.88,1.90,1.92,1.95,1.98,2.00,2.02,"
    "2.04,2.06,2.08,2.10,2.12,2.14,2.15,2.16,2.17,2.18,2.19,"
    "2.20,2.22,2.25,2.30,2.35,2.40,2.50,2.65,2.80,3.00,3.25,"
    "3.50,4.00,4.50,5.00,6.00"
)


def _content_key(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _cell_identity(
    *, name: str, nominal_budget: int, mtp_in: bool, kind: str,
    collapsed_content_key: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": "prismaquant.dsv4_afast_grid_cell_identity.v1",
        "profile": "A-FAST",
        "name": name,
        "nominal_budget_bytes": int(nominal_budget),
        "mtp_in": bool(mtp_in),
        "kind": kind,
        "collapsed_content_key": collapsed_content_key,
        "mtp_bf16_bytes": MTP_BF16_BYTES,
        "menu": list(MENU),
        "pareto_targets": PARETO_TARGETS,
        "overhead_reserve_bytes": OVERHEAD_RESERVE_BYTES,
        "cost_sha256": sha256_file(COST_TABLE),
        "probe_sha256": sha256_file(PROBE),
        "source_index_sha256": sha256_file(
            SOURCE / "model.safetensors.index.json"
        ),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }


def _validated_completion(
    path: Path, identity: Mapping[str, Any], required: tuple[Path, ...],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != "prismaquant.dsv4_afast_grid_cell.v1"
        or payload.get("identity") != dict(identity)
        or payload.get("content_key") != _content_key(identity)
        or not isinstance(payload.get("result"), Mapping)
        or any(not item.is_file() for item in required)
    ):
        raise AssertionError(f"stale or corrupt grid completion: {path}")
    return dict(payload["result"])


def _complete(
    path: Path, identity: Mapping[str, Any], result: Mapping[str, Any],
) -> dict[str, Any]:
    content_key = _content_key(identity)
    completed = {**dict(result), "content_key": content_key}
    atomic_json(path, {
        "schema": "prismaquant.dsv4_afast_grid_cell.v1",
        "content_key": content_key,
        "identity": dict(identity),
        "result": completed,
    })
    return completed


def _unique_fp8_sidecar_bytes(assignment: Mapping[str, str]) -> int:
    total = 0
    for fmt in assignment.values():
        if not fmt.startswith("FP8_CB_K"):
            continue
        rung = int(fmt.rsplit("K", 1)[1])
        total += sum(
            rows * dim * 2
            for rows, dim in codebook_subtable_shapes(rung, "product", 4)
        )
    return total


def _tee(command: list[str], log_path: Path) -> None:
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


def _allocator_command(out: Path, solver_budget: int) -> list[str]:
    return [
        sys.executable, "-m", "prismaquant.allocator",
        "--probe", str(PROBE),
        "--costs", str(COST_TABLE),
        "--accept-research-cost-table",
        "--model-override", str(SOURCE),
        "--target-profile", "nvfp4_cb",
        "--target-disk-gb", f"{solver_budget / 1e9:.9f}",
        "--artifact-overhead-reserve-bytes", str(OVERHEAD_RESERVE_BYTES),
        "--cb-scale-coding", "two_tier",
        "--cb-codebook-source", "lattice",
        "--cb-scale-sweep", "1", "--cb-ldlq", "1",
        "--cb-encode-tier", "balanced",
        "--cb-col-weights", str(COL_WEIGHTS),
        "--formats", ",".join(MENU),
        "--mtp-format", "BF16", "--visual-format", "BF16",
        "--threads", "16", "--pareto-targets", PARETO_TARGETS,
        "--layer-config", str(out / "layer_config.json"),
        "--pareto-csv", str(out / "pareto.csv"),
        "--pareto-output-dir", str(out / "pareto-points"),
        "--applicability-report", str(out / "format_applicability.json"),
        "--bit-attribution-json", str(out / "bit_attribution.json"),
        "--bit-attribution-csv", str(out / "bit_attribution.csv"),
    ]


def _solver_budget(nominal: int, mtp_in: bool, sidecar: int = 0) -> int:
    return nominal + (0 if mtp_in else MTP_BF16_BYTES) - sidecar


def _reported_bytes(raw: int, mtp_in: bool, sidecar: int) -> int:
    return raw - (0 if mtp_in else MTP_BF16_BYTES) + sidecar


def _collapsed(
    name: str, nominal_budget: int, *, mtp_in: bool,
) -> dict[str, Any]:
    variant = "mtp-in" if mtp_in else "mtp-out"
    out = GRID_ROOT / name / variant / "collapsed"
    out.mkdir(parents=True, exist_ok=True)
    identity = _cell_identity(
        name=name, nominal_budget=nominal_budget, mtp_in=mtp_in,
        kind="collapsed",
    )
    completion_path = out / "CELL_COMPLETE.json"
    existing = _validated_completion(
        completion_path, identity,
        (out / "BYTE_GATE.json", out / "layer_config.json", out / "selection.json"),
    )
    if existing is not None:
        print(
            f"[afast-grid] content-resume {name} {variant} collapsed",
            flush=True,
        )
        return existing
    sidecar = 0
    attempts = []
    for attempt in range(1, 7):
        solver_budget = _solver_budget(nominal_budget, mtp_in, sidecar)
        print(
            f"[afast-grid] {name} {variant} collapsed attempt {attempt} "
            f"solver={solver_budget/1e9:.6f}GB", flush=True,
        )
        _tee(
            _allocator_command(out, solver_budget),
            out / f"allocator-attempt-{attempt}.log",
        )
        selection = json.loads((out / "selection.json").read_text())
        assignment = load_assignment(out / "layer_config.json")
        next_sidecar = _unique_fp8_sidecar_bytes(assignment)
        raw = int(selection["whole_artifact_budget"][
            "selection_whole_artifact_upper_bound_bytes"
        ])
        reported = _reported_bytes(raw, mtp_in, next_sidecar)
        attempts.append({
            "attempt": attempt, "solver_budget_bytes": solver_budget,
            "raw_upper_bound_bytes": raw,
            "unique_chain_sidecar_bytes": next_sidecar,
            "reported_total_bytes": reported,
        })
        if reported <= nominal_budget:
            gate = {
                "pass": True, "nominal_total_budget_bytes": nominal_budget,
                "mtp_in": mtp_in, "mtp_removed_bytes": 0 if mtp_in else MTP_BF16_BYTES,
                "solver_budget_bytes": solver_budget,
                "raw_upper_bound_bytes": raw,
                "unique_chain_sidecar_bytes": next_sidecar,
                "reported_total_bytes": reported,
                "headroom_bytes": nominal_budget - reported,
                "attempts": attempts,
            }
            atomic_json(out / "BYTE_GATE.json", gate)
            return _complete(completion_path, identity, {
                "selection": selection, "byte_gate": gate, "directory": str(out),
            })
        if next_sidecar == sidecar:
            break
        sidecar = next_sidecar
    atomic_json(out / "BYTE_GATE_FAIL.json", {"attempts": attempts})
    raise RuntimeError(f"{name} {variant}: collapsed byte gate failed")


def _per_expert(
    name: str, nominal_budget: int, *, mtp_in: bool,
    collapsed_content_key: str,
) -> dict[str, Any]:
    variant = "mtp-in" if mtp_in else "mtp-out"
    baseline = GRID_ROOT / name / variant / "collapsed"
    out = GRID_ROOT / name / variant / "per-expert"
    out.mkdir(parents=True, exist_ok=True)
    identity = _cell_identity(
        name=name, nominal_budget=nominal_budget, mtp_in=mtp_in,
        kind="per-expert", collapsed_content_key=collapsed_content_key,
    )
    completion_path = out / "CELL_COMPLETE.json"
    existing = _validated_completion(
        completion_path, identity, (out / "result.json",),
    )
    if existing is not None:
        print(
            f"[afast-grid] content-resume {name} {variant} per-expert",
            flush=True,
        )
        return existing
    sidecar = 0
    attempts = []
    for attempt in range(1, 7):
        solver_budget = _solver_budget(nominal_budget, mtp_in, sidecar)
        print(
            f"[afast-grid] {name} {variant} per-expert attempt {attempt} "
            f"solver={solver_budget/1e9:.6f}GB", flush=True,
        )
        result = run_counterfactual(
            baseline_dir=baseline, probe_path=PROBE,
            cost_path=COST_TABLE,
            budget_bytes=solver_budget, menu=MENU, floor_unmeasured=False,
        )
        next_sidecar = _unique_fp8_sidecar_bytes(result["assignment"])
        raw = int(result["exact_bytes"])
        reported = _reported_bytes(raw, mtp_in, next_sidecar)
        attempts.append({
            "attempt": attempt, "solver_budget_bytes": solver_budget,
            "raw_exact_bytes": raw, "unique_chain_sidecar_bytes": next_sidecar,
            "reported_total_bytes": reported,
        })
        if reported <= nominal_budget:
            result.update({
                "nominal_total_budget_bytes": nominal_budget,
                "mtp_in": mtp_in,
                "mtp_removed_bytes": 0 if mtp_in else MTP_BF16_BYTES,
                "solver_budget_bytes": solver_budget,
                "unique_chain_sidecar_bytes": next_sidecar,
                "reported_total_bytes": reported,
                "reported_headroom_bytes": nominal_budget - reported,
                "adjusted_bytes_gate": True, "byte_gate_attempts": attempts,
                "optimality_gap_certified": False,
                "optimality_gap_note": (
                    "Lagrangian candidate generation plus exact-payload tidy "
                    "does not certify a global discrete-knapsack bound."
                ),
            })
            atomic_json(out / "result.json", result)
            return _complete(completion_path, identity, result)
        if next_sidecar == sidecar:
            break
        sidecar = next_sidecar
    atomic_json(out / "BYTE_GATE_FAIL.json", {"attempts": attempts})
    raise RuntimeError(f"{name} {variant}: per-expert byte gate failed")


def _cell(name: str, budget: int) -> dict[str, Any]:
    variants = {}
    for mtp_in in (True, False):
        label = "mtp-in" if mtp_in else "mtp-out"
        collapsed = _collapsed(name, budget, mtp_in=mtp_in)
        per_expert = _per_expert(
            name, budget, mtp_in=mtp_in,
            collapsed_content_key=str(collapsed["content_key"]),
        )
        collapsed_loss = float(collapsed["selection"]["predicted_dloss"])
        expert_loss = float(per_expert["predicted_dloss"])
        variants[label] = {
            "collapsed": collapsed, "per_expert": per_expert,
            "observed_per_expert_solver_gap": {
                "absolute_dloss": collapsed_loss - expert_loss,
                "relative_to_collapsed": (
                    (collapsed_loss - expert_loss) / collapsed_loss
                    if collapsed_loss else 0.0
                ),
                "certified_optimality_gap": None,
            },
        }
    return {"nominal_total_budget_bytes": budget, "variants": variants}


def _knee_budget(reference: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    directory = Path(reference["directory"])
    knees = json.loads((directory / "pareto.knees.json").read_text())
    knee = dict(knees[knees["primary"]])
    target = float(knee["target_bits"])
    matches = [
        point for point in reference["selection"]["grid"]
        if abs(float(point["target_bits"]) - target) < 1e-9
    ]
    if len(matches) != 1:
        raise AssertionError(f"knee target {target} missing/duplicated")
    budget = int(round(float(matches[0]["whole_artifact_upper_bound_gb"]) * 1e9))
    return budget, {"knee": knee, "source_point": matches[0]}


def run() -> int:
    cells = {}
    for name, budget in (
        ("b-88", 88_000_000_000), ("b-92", 92_000_000_000),
        ("b-97", 97_000_000_000), ("b-102p8", 102_800_000_000),
    ):
        cells[name] = _cell(name, budget)
    knee_budget, knee_source = _knee_budget(
        cells["b-92"]["variants"]["mtp-in"]["collapsed"]
    )
    cells["knee"] = _cell("knee", knee_budget)
    cells["knee"]["knee_source"] = knee_source
    report_identity = {
        "schema": "prismaquant.dsv4_afast_grid_report_identity.v1",
        "cell_content_keys": {
            name: {
                variant: {
                    kind: data["content_key"]
                    for kind, data in variants.items()
                    if kind in {"collapsed", "per_expert"}
                }
                for variant, variants in cell["variants"].items()
            }
            for name, cell in cells.items()
        },
    }
    report = {
        "schema": "prismaquant.dsv4_afast_allocation_grid.v1",
        "content_key": _content_key(report_identity),
        "content_identity": report_identity,
        "profile": "A-FAST", "menu": list(MENU),
        "mtp_bf16_bytes": MTP_BF16_BYTES,
        "mtp_semantics": {
            "mtp-in": "fixed source-format carry counts inside nominal total bytes",
            "mtp-out": (
                "same nominal total bytes; 10.8628383GB removed from the fixed "
                "floor and made available to body choices"
            ),
        },
        "unique_chain_sidecar_policy": (
            "conservative distinct flat FP16 table bundle per selected FP8 tensor"
        ),
        "per_expert_solver_caveat": (
            "Observed collapsed-to-per-expert loss deltas are reported, but the "
            "Lagrangian/tidy solver has no certified global optimality gap."
        ),
        "cells": cells,
    }
    atomic_json(GRID_ROOT / "GRID_REPORT.json", report)
    print(f"[afast-grid] PASS -> {GRID_ROOT / 'GRID_REPORT.json'}", flush=True)
    return 0


def main() -> int:
    argparse.ArgumentParser().parse_args()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
