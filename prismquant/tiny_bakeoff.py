#!/usr/bin/env python3
"""tiny_bakeoff.py — one-command tiny-model PrismQuant regression bakeoff.

This orchestrates the canonical small-model validation loop:

  1. local_reconstruct
  2. measure_interactions
  3. calibrate_allocator
  4. quadratic_refine_allocator
  5. oracle_search
  6. bakeoff

It is designed to answer one question quickly and consistently:
    "Did the latest change earn its complexity?"

The script assumes a tiny model and precomputed probe/cost/cache by default,
but all paths are overridable.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)
DEFAULT_PROBE = "/tmp/tiny_probe.pkl"
DEFAULT_COSTS = "/tmp/tiny_cost.pkl"
DEFAULT_ACT_CACHE = "/tmp/tiny_act"
DEFAULT_OUTPUT_DIR = "/tmp/prismquant_tiny_bakeoff"


def _run(cmd: list[str], cwd: str, dry_run: bool):
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"[bakeoff-run] {rendered}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def _paths(out_dir: Path):
    return {
        "costs_refined": out_dir / "costs_refined.pkl",
        "interactions": out_dir / "interactions.json",
        "refined": out_dir / "refined.json",
        "calibration": out_dir / "calibration.json",
        "oracle": out_dir / "oracle.json",
        "decision": out_dir / "decision.json",
    }


def _target_list(args) -> list[float]:
    if getattr(args, "target_grid", None):
        return [float(x) for x in args.target_grid.split(",") if x.strip()]
    half = max(float(getattr(args, "target_band", 0.0)), 0.0)
    if half <= 0:
        return [float(args.target_bits)]
    return [
        round(float(args.target_bits) - half, 4),
        round(float(args.target_bits), 4),
        round(float(args.target_bits) + half, 4),
    ]


def _variant_dir(base: Path, target_bits: float) -> Path:
    return base / f"target_{target_bits:.4f}".replace(".", "p")


def build_bakeoff_commands(args) -> tuple[dict[str, Path], list[list[str]]]:
    commands = []
    paths_by_target = {}
    base_out_dir = Path(args.output_dir)
    for target_bits in _target_list(args):
        out_dir = _variant_dir(base_out_dir, target_bits)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = _paths(out_dir)
        paths_by_target[f"{target_bits:.4f}"] = paths
        commands.append([
            sys.executable,
            "-m",
            "prismquant.local_reconstruct",
            "--model", args.model,
            "--probe", args.probe,
            "--costs", args.costs,
            "--activation-cache-dir", args.activation_cache_dir,
            "--formats", args.formats,
            "--target-bits", str(target_bits),
            "--top-units", str(args.top_units),
            "--unit-scope", args.unit_scope,
            "--device", args.device,
            "--dtype", "bf16",
            "--refine-rounds", str(args.refine_rounds),
            "--rowwise-topk", str(args.rowwise_topk),
            "--rowwise-rounds", str(args.rowwise_rounds),
            "--groupwise-topk", str(args.groupwise_topk),
            "--groupwise-rounds", str(args.groupwise_rounds),
            "--gptq-topk", str(args.gptq_topk),
            "--gptq-damping", str(args.gptq_damping),
            "--output", str(paths["costs_refined"]),
        ])
        commands.append([
            sys.executable,
            "-m",
            "prismquant.measure_interactions",
            "--model", args.model,
            "--probe", args.probe,
            "--costs", str(paths["costs_refined"]),
            "--formats", args.formats,
            "--target-bits", str(target_bits),
            "--top-units", str(args.top_units),
            "--unit-scope", args.unit_scope,
            "--neighbor-radius", str(args.neighbor_radius),
            "--n-calib-samples", str(args.n_calib_samples),
            "--calib-seqlen", str(args.calib_seqlen),
            "--device", args.device,
            "--output", str(paths["interactions"]),
        ])
        commands.append([
            sys.executable,
            "-m",
            "prismquant.calibrate_allocator",
            "--model", args.model,
            "--probe", args.probe,
            "--costs", str(paths["costs_refined"]),
            "--formats", args.formats,
            "--pareto-targets", f"4.5,{target_bits},16.0",
            "--selection", "baseline,knee,high",
            "--n-calib-samples", str(args.n_calib_samples),
            "--calib-seqlen", str(args.calib_seqlen),
            "--device", args.device,
            "--output", str(paths["calibration"]),
        ])
        commands.append([
            sys.executable,
            "-m",
            "prismquant.quadratic_refine_allocator",
            "--interactions", str(paths["interactions"]),
            "--calibration", str(paths["calibration"]),
            "--output", str(paths["refined"]),
        ])
        if not args.skip_oracle:
            commands.append([
                sys.executable,
                "-m",
                "prismquant.oracle_search",
                "--interactions", str(paths["interactions"]),
                "--model", args.model,
                "--n-calib-samples", str(args.n_calib_samples),
                "--calib-seqlen", str(args.calib_seqlen),
                "--device", args.device,
                "--max-combos", str(args.oracle_max_combos),
                "--output", str(paths["oracle"]),
            ])
        bakeoff_cmd = [
            sys.executable,
            "-m",
            "prismquant.bakeoff",
            "--calibration", str(paths["calibration"]),
            "--candidate", "refined",
            "--refined", str(paths["refined"]),
            "--output", str(paths["decision"]),
        ]
        if not args.skip_oracle:
            bakeoff_cmd.extend(["--oracle", str(paths["oracle"])])
        commands.append(bakeoff_cmd)
    return paths_by_target, commands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--probe", default=DEFAULT_PROBE)
    ap.add_argument("--costs", default=DEFAULT_COSTS)
    ap.add_argument("--activation-cache-dir", default=DEFAULT_ACT_CACHE)
    ap.add_argument("--formats", default="NVFP4,MXFP8,BF16")
    ap.add_argument("--target-bits", type=float, default=4.8)
    ap.add_argument("--target-band", type=float, default=0.0,
                    help="If > 0, also run target_bits ± target_band")
    ap.add_argument("--target-grid", default="",
                    help="Explicit comma-separated target list; overrides --target-band")
    ap.add_argument("--top-units", type=int, default=6)
    ap.add_argument("--unit-scope", choices=["sibling", "block", "hybrid", "layer"], default="sibling")
    ap.add_argument("--neighbor-radius", type=int, default=1)
    ap.add_argument("--refine-rounds", type=int, default=2)
    ap.add_argument("--rowwise-topk", type=int, default=8)
    ap.add_argument("--rowwise-rounds", type=int, default=1)
    ap.add_argument("--groupwise-topk", type=int, default=16)
    ap.add_argument("--groupwise-rounds", type=int, default=1)
    ap.add_argument("--gptq-topk", type=int, default=8)
    ap.add_argument("--gptq-damping", type=float, default=1e-4)
    ap.add_argument("--n-calib-samples", type=int, default=2)
    ap.add_argument("--calib-seqlen", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--oracle-max-combos", type=int, default=1024)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths_by_target, commands = build_bakeoff_commands(args)
    cwd = os.getcwd()
    for cmd in commands:
        _run(cmd, cwd, args.dry_run)

    if args.dry_run:
        summary = {
            "output_dir": str(out_dir),
            "targets": {
                tgt: {k: str(v) for k, v in paths.items()}
                for tgt, paths in paths_by_target.items()
            },
            "oracle_enabled": not args.skip_oracle,
        }
        print(json.dumps(summary, indent=2))
    elif len(paths_by_target) > 1:
        scoreboard = []
        for tgt, paths in paths_by_target.items():
            if not paths["decision"].exists():
                continue
            with open(paths["decision"]) as f:
                decision = json.load(f)
            scoreboard.append({
                "target_bits": float(tgt),
                "candidate_bits": decision["candidate"]["bits"],
                "candidate_kl": decision["candidate"]["kl"],
                "delta_kl_vs_baseline": decision["delta_kl_vs_baseline"],
                "oracle_gap_abs": decision.get("oracle_gap_abs"),
                "decision": decision["decision"],
            })
        scoreboard.sort(key=lambda row: (row["candidate_kl"], row["candidate_bits"]))
        summary_path = out_dir / "scoreboard.json"
        with open(summary_path, "w") as f:
            json.dump(scoreboard, f, indent=2)
        print(json.dumps(scoreboard, indent=2))


if __name__ == "__main__":
    main()
