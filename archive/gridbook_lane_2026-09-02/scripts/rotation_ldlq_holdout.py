#!/usr/bin/env python3
"""Run the disjoint-row held-out gate for the fixed-CB LDLQ pilot."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from prismaquant.cb_layout import parse_format_name
from prismaquant.rotation_ldlq_pilot import (
    activation_row_split_indices,
    compare_output_sse,
    inverse_hessian_cholesky,
    reassign_product_cb,
)

SAMPLE_ROOT = Path("/home/rob/dq-runs/dsv4-flash-0731/tier3-sample")
SAMPLES = (
    "layers.40.attn.wq_b",
    "layers.40.experts.81.up_proj",
    "layers.20.experts.63.up_proj",
)
RUNGS = (12, 15, 18)
SPLIT_SEEDS = (0, 1, 2)
CAL_SIZES = (16, 32)
DECISION_CAL_ROWS = 32
HOLDOUT_ROWS = 32
PRODUCTION_ENV = {
    "PRISMAQUANT_ACTIVATION_FAIR_PRICING": "1",
    "CB_CODEBOOK_SOURCE": "lattice",
    "CB_SCALE_CODING": "two_tier",
    "CB_SCALE_SWEEP": "1",
    "PRISMAQUANT_CB_ENCODE_TIER": "balanced",
}


def _configure_environment(ext_dir: Path) -> None:
    for name, value in PRODUCTION_ENV.items():
        os.environ[name] = value
    os.environ["PRISMAQUANT_CB_EXT_DIR"] = str(ext_dir)


def _load_tensor(path: Path, device: torch.device) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} did not contain a bare tensor")
    return value.to(device=device).contiguous()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, fn):
    _sync(device)
    start = time.perf_counter()
    value = fn()
    _sync(device)
    return value, time.perf_counter() - start


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _production_fields(
    weight: torch.Tensor,
    col_weights: torch.Tensor,
    rung: int,
) -> dict:
    from prismaquant.nvfp4_cb_formats import nvfp4_cb_fields

    parsed = parse_format_name(f"NVFP4_CB_K{rung}")
    assert parsed is not None
    family, k = parsed
    return nvfp4_cb_fields(
        weight,
        k,
        grid=family.grid,
        mode=family.mode,
        col_weights=col_weights,
        scale_sweep=True,
        scale_coding="two_tier",
        encode_tier="balanced",
    )


def _reconstruct(fields: dict, rung: int, dtype: torch.dtype) -> torch.Tensor:
    from prismaquant.nvfp4_cb_formats import nvfp4_cb_reconstruct

    parsed = parse_format_name(f"NVFP4_CB_K{rung}")
    assert parsed is not None
    family, k = parsed
    return nvfp4_cb_reconstruct(fields, k, grid=family.grid, mode=family.mode).to(
        dtype
    )


def _verdict(cal_reduction: float, holdout_reduction: float) -> tuple[str, float]:
    retention = holdout_reduction / cal_reduction if cal_reduction > 0.0 else 0.0
    if retention > 0.50:
        return "VALIDATED-PENDING-MODEL-LEVEL", retention
    if holdout_reduction < 0.10:
        return "OVERFIT-ARTIFACT", retention
    return "PARTIAL", retention


def _release(*values: Any) -> None:
    del values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _new_results(
    sample_root: Path,
    ext_dir: Path,
    device: torch.device,
    *,
    block_size: int,
    damping_fraction: float,
) -> dict[str, Any]:
    return {
        "schema": "prismaquant.rotation_ldlq_holdout.v1",
        "branch": "feat/rotation-ldlq-pilot",
        "sample_root": str(sample_root),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "split_contract": {
            "activation_rows": 64,
            "decision_cal_rows": DECISION_CAL_ROWS,
            "holdout_rows": HOLDOUT_ROWS,
            "cal_size_probe_rows": list(CAL_SIZES),
            "split_seeds": list(SPLIT_SEEDS),
            "nesting": (
                "one CPU randperm per seed; CAL16=perm[:16], "
                "CAL32=perm[:32], HOLDOUT=perm[32:]"
            ),
        },
        "method": {
            "plain": (
                "production nvfp4_cb_fields assignment using CAL-only mean(X^2) "
                "column weights"
            ),
            "feedback": (
                "same fixed codebook/scales and CAL-only assignment metric, plus "
                "block-sequential LDLQ feedback from the CAL-only Hessian"
            ),
            "block_size": block_size,
            "damping_fraction": damping_fraction,
        },
        "production_encoder_context": {
            **PRODUCTION_ENV,
            "PRISMAQUANT_CB_EXT_DIR": str(ext_dir),
        },
        "samples": {},
    }


def run_holdout(
    out_dir: Path,
    sample_root: Path,
    ext_dir: Path,
    device: torch.device,
    *,
    block_size: int,
    damping_fraction: float,
) -> dict[str, Any]:
    result_path = out_dir / "results.json"
    results = _new_results(
        sample_root,
        ext_dir,
        device,
        block_size=block_size,
        damping_fraction=damping_fraction,
    )
    for sample in SAMPLES:
        print(f"[holdout] {sample}", flush=True)
        tensors = sample_root / sample / "tensors"
        weight = _load_tensor(tensors / "W_bf16_ref.pt", device)
        acts = _load_tensor(tensors / "X_acts.pt", device)
        if int(acts.shape[0]) != DECISION_CAL_ROWS + HOLDOUT_ROWS:
            raise ValueError(
                f"{sample}: expected 64 activation rows, got {acts.shape[0]}"
            )
        sample_record: dict[str, Any] = {
            "weight_shape": list(weight.shape),
            "activation_shape": list(acts.shape),
            "splits": {},
        }
        results["samples"][sample] = sample_record
        for split_seed in SPLIT_SEEDS:
            cal32_indices, holdout_indices = activation_row_split_indices(
                acts.shape[0], cal_rows=DECISION_CAL_ROWS, seed=split_seed
            )
            if holdout_indices.numel() != HOLDOUT_ROWS:
                raise AssertionError("decision split did not produce a 32-row holdout")
            x_holdout = acts.index_select(0, holdout_indices.to(device))
            split_record: dict[str, Any] = {
                "cal32_indices": cal32_indices.tolist(),
                "holdout_indices": holdout_indices.tolist(),
                "cal_sizes": {},
            }
            sample_record["splits"][str(split_seed)] = split_record
            print(f"  split={split_seed}", flush=True)
            for cal_rows in CAL_SIZES:
                cal_indices, _ = activation_row_split_indices(
                    acts.shape[0], cal_rows=cal_rows, seed=split_seed
                )
                if not torch.equal(cal_indices, cal32_indices[:cal_rows]):
                    raise AssertionError("CAL-size probe is not nested in CAL32")
                x_cal = acts.index_select(0, cal_indices.to(device))
                col_weights = x_cal.to(torch.float32).square().mean(dim=0)
                hessian, hessian_seconds = _timed(
                    device,
                    lambda: x_cal.to(torch.float32).T @ x_cal.to(torch.float32),
                )
                (upper, damping), factor_seconds = _timed(
                    device,
                    lambda: inverse_hessian_cholesky(
                        hessian, damping_fraction=damping_fraction
                    ),
                )
                del hessian
                cal_record: dict[str, Any] = {
                    "cal_rows": cal_rows,
                    "holdout_rows": HOLDOUT_ROWS,
                    "cal_indices": cal_indices.tolist(),
                    "hessian_seconds": hessian_seconds,
                    "factor_seconds": factor_seconds,
                    "damping_added": damping,
                    "rungs": {},
                }
                split_record["cal_sizes"][str(cal_rows)] = cal_record
                print(f"    CAL={cal_rows}", flush=True)
                for rung in RUNGS:
                    fields, encode_seconds = _timed(
                        device,
                        lambda rung=rung: _production_fields(
                            weight, col_weights, rung
                        ),
                    )
                    plain = _reconstruct(fields, rung, weight.dtype)
                    feedback, feedback_seconds = _timed(
                        device,
                        lambda: reassign_product_cb(
                            weight,
                            fields,
                            col_weights,
                            block_size=block_size,
                            upper_inverse_cholesky=upper,
                        ),
                    )
                    cal_eval = compare_output_sse(x_cal, weight, plain, feedback)
                    holdout_eval = compare_output_sse(
                        x_holdout, weight, plain, feedback
                    )
                    verdict, retention = _verdict(
                        cal_eval["reduction"], holdout_eval["reduction"]
                    )
                    cal_record["rungs"][str(rung)] = {
                        "cal": cal_eval,
                        "holdout": holdout_eval,
                        "generalization_gap": (
                            cal_eval["reduction"] - holdout_eval["reduction"]
                        ),
                        "heldout_gain_retention": retention,
                        "verdict": verdict,
                        "production_encode_seconds": encode_seconds,
                        "feedback_assignment_seconds": feedback_seconds,
                    }
                    print(
                        f"      K{rung}: cal={cal_eval['feedback_over_plain_ratio']:.4f} "
                        f"holdout={holdout_eval['feedback_over_plain_ratio']:.4f} "
                        f"{verdict}",
                        flush=True,
                    )
                    _save_json(result_path, results)
                    del fields, plain, feedback
                    _release()
                del upper, x_cal, col_weights
                _release()
            del x_holdout
        del weight, acts
        _release()
    _save_json(result_path, results)
    return results


def _records(results: dict[str, Any], cal_rows: int):
    for sample in SAMPLES:
        for split_seed in SPLIT_SEEDS:
            rungs = results["samples"][sample]["splits"][str(split_seed)][
                "cal_sizes"
            ][str(cal_rows)]["rungs"]
            for rung in RUNGS:
                yield sample, rung, split_seed, rungs[str(rung)]


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(out_dir: Path, results: dict[str, Any]) -> Path:
    decision_rows = list(_records(results, DECISION_CAL_ROWS))
    mean_cal = fmean(record["cal"]["reduction"] for *_, record in decision_rows)
    mean_holdout = fmean(
        record["holdout"]["reduction"] for *_, record in decision_rows
    )
    overall_verdict, overall_retention = _verdict(mean_cal, mean_holdout)
    verdict_counts = {
        verdict: sum(record["verdict"] == verdict for *_, record in decision_rows)
        for verdict in (
            "VALIDATED-PENDING-MODEL-LEVEL",
            "PARTIAL",
            "OVERFIT-ARTIFACT",
        )
    }

    size16 = {
        (sample, rung, seed): record
        for sample, rung, seed, record in _records(results, 16)
    }
    size32 = {
        (sample, rung, seed): record
        for sample, rung, seed, record in decision_rows
    }
    deltas = [
        size16[key]["holdout"]["reduction"]
        - size32[key]["holdout"]["reduction"]
        for key in size32
    ]
    grows_count = sum(delta > 0.0 for delta in deltas)
    mean_holdout16 = fmean(
        record["holdout"]["reduction"] for record in size16.values()
    )
    mean_holdout32 = mean_holdout

    lines = [
        "# Fixed-CB LDLQ held-out validation",
        "",
        "Date: 2026-08-03",
        "",
        "Branch: `feat/rotation-ldlq-pilot`",
        "",
        f"GPU: {results['device']}, torch {results['torch_version']}",
        "",
        "Raw results: [`results.json`](results.json)",
        "",
        "## Verdict",
        "",
        f"**{overall_verdict}.** Across the 27 CAL32 decisions, mean in-sample "
        f"reduction is {_percent(mean_cal)} and mean held-out reduction is "
        f"{_percent(mean_holdout)}, retaining {_percent(overall_retention)} of "
        "the in-sample gain. Cell verdicts: "
        f"{verdict_counts['VALIDATED-PENDING-MODEL-LEVEL']} validated, "
        f"{verdict_counts['PARTIAL']} partial, and "
        f"{verdict_counts['OVERFIT-ARTIFACT']} overfit-artifact.",
        "",
        "The decision rule is fixed in advance: held-out reduction greater than "
        "50% of the in-sample reduction is `VALIDATED-PENDING-MODEL-LEVEL`; "
        "held-out reduction below 10% is `OVERFIT-ARTIFACT`; intermediate "
        "outcomes are `PARTIAL` with retained gain reported.",
        "",
        "## Contract",
        "",
        "Each Linear has exactly 64 activation rows. For each of seeds 0, 1, "
        "and 2, a deterministic CPU random permutation defines disjoint CAL32 "
        "(first 32 rows) and HOLDOUT32 (last 32 rows) sets. There is no "
        "stratification. The CAL16 mechanics probe uses the first 16 rows of "
        "the same CAL32 permutation and is evaluated on the same HOLDOUT32.",
        "",
        "For every split and rung, the plain production encoder fits its "
        "codebook, two-tier scales, and assignments using only the CAL-derived "
        "`mean(X^2)` column weights. LDLQ freezes those fields, uses the same "
        "CAL-only assignment metric, and builds its damped Hessian from CAL "
        "only. Both reconstructed weights are then evaluated separately on CAL "
        "and HOLDOUT. Ratios below 1 and positive reductions favor LDLQ.",
        "",
        "## CAL32 decision table",
        "",
        "| Linear | K | split | CAL ratio | CAL reduction | HOLDOUT ratio | "
        "HOLDOUT reduction | gap (pp) | retained | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sample, rung, seed, record in decision_rows:
        cal = record["cal"]
        holdout = record["holdout"]
        retention = record["heldout_gain_retention"]
        retained = _percent(retention) if cal["reduction"] > 0.0 else "n/a"
        lines.append(
            f"| `{sample}` | {rung} | {seed} | "
            f"{cal['feedback_over_plain_ratio']:.4f} | {_percent(cal['reduction'])} | "
            f"{holdout['feedback_over_plain_ratio']:.4f} | "
            f"{_percent(holdout['reduction'])} | "
            f"{100.0 * record['generalization_gap']:.2f} | {retained} | "
            f"{record['verdict']} |"
        )

    lines.extend([
        "",
        "## CAL-size overfit-mechanics probe",
        "",
        "The comparison below holds the 32-row evaluation set fixed. A positive "
        "delta means held-out gain grew when calibration shrank from 32 to 16 "
        "rows, the classic overfit signature named in the experiment design.",
        "",
        "| Linear | K | split | HOLDOUT reduction CAL16 | HOLDOUT reduction "
        "CAL32 | CAL16-CAL32 (pp) | trend |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for key in size32:
        sample, rung, seed = key
        hold16 = size16[key]["holdout"]["reduction"]
        hold32 = size32[key]["holdout"]["reduction"]
        delta = hold16 - hold32
        trend = "grows as CAL shrinks" if delta > 0.0 else "stable/better at CAL32"
        lines.append(
            f"| `{sample}` | {rung} | {seed} | {_percent(hold16)} | "
            f"{_percent(hold32)} | {100.0 * delta:.2f} | {trend} |"
        )
    mechanics = (
        "This is an overfit signature"
        if mean_holdout16 > mean_holdout32
        else "This argues that feedback is capturing structure rather than benefiting from a smaller fit set"
    )
    lines.extend([
        "",
        f"Mean held-out reduction is {_percent(mean_holdout16)} at CAL16 versus "
        f"{_percent(mean_holdout32)} at CAL32; gain grows under smaller CAL in "
        f"{grows_count}/27 matched cases. **{mechanics}.**",
        "",
        "## Limitation and next gate",
        "",
        "This 32-row holdout comes from the same calibration distribution. It "
        "tests estimation overfit, not distribution shift. The stronger test is "
        "a future GPU probe that runs fresh text through the model and captures "
        "new activations; this experiment deliberately performs no new model "
        "forwards.",
        "",
        "Command:",
        "",
        "```bash",
        "export PYTHONPATH=/w PRISMAQUANT_CB_EXT_DIR=/w/.ext",
        "/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \\",
        "  scripts/rotation_ldlq_holdout.py --out rotpilot-out/holdout \\",
        "  --sample-root /home/rob/dq-runs/dsv4-flash-0731/tier3-sample \\",
        "  --ext-dir /w/.ext --block-size 64 --damping-fraction 0.01",
        "```",
        "",
    ])
    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("rotpilot-out/holdout"))
    parser.add_argument("--sample-root", type=Path, default=SAMPLE_ROOT)
    parser.add_argument("--ext-dir", type=Path, default=Path(".ext"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--damping-fraction", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sample_root = args.sample_root.resolve()
    ext_dir = args.ext_dir.resolve()
    _configure_environment(ext_dir)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real-tensor holdout pilot requires CUDA")
    args.out.mkdir(parents=True, exist_ok=True)
    results = run_holdout(
        args.out,
        sample_root,
        ext_dir,
        device,
        block_size=args.block_size,
        damping_fraction=args.damping_fraction,
    )
    report = write_report(args.out, results)
    print(f"wrote {args.out / 'results.json'}", flush=True)
    print(f"wrote {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
