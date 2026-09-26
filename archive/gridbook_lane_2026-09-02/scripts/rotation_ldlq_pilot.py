#!/usr/bin/env python3
"""Run the bounded DSv4 rotation and fixed-CB-feedback mini-experiments."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from prismaquant.cb_layout import parse_format_name
from prismaquant.row_dispersion import per_row_error, tail_metrics
from prismaquant.rotation_ldlq_pilot import (
    inverse_hessian_cholesky,
    random_hadamard_signs,
    randomized_hadamard_right,
    reassign_product_cb,
    relative_frobenius_gap,
)

SAMPLE_ROOT = Path("/home/rob/dq-runs/dsv4-flash-0731/tier3-sample")
SAMPLES = (
    "layers.40.attn.wq_b",
    "layers.40.experts.81.up_proj",
    "layers.20.experts.63.up_proj",
)
RUNGS = (12, 15, 18)
SEEDS = (0, 1)
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


def _output_record(
    x: torch.Tensor,
    weight: torch.Tensor,
    reconstructed: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    # The recovered row-dispersion metric library is intentionally CPU-only
    # for its percentile/Gini reductions.  Keep the large matmul on GPU and
    # transfer only the one-float64-per-output-row error vector.
    errors = per_row_error(x, weight, reconstructed).cpu()
    return float(errors.sum()), tail_metrics(errors)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _release(*values: Any) -> None:
    del values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sample_tensors(sample: str, device: torch.device):
    tensors = SAMPLE_ROOT / sample / "tensors"
    weight = _load_tensor(tensors / "W_bf16_ref.pt", device)
    acts = _load_tensor(tensors / "X_acts.pt", device)
    baselines = {
        rung: _load_tensor(tensors / f"What_NVFP4_CB_K{rung}.pt", device)
        for rung in RUNGS
    }
    return weight, acts, baselines


def _baseline_records(
    sample: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    baselines: dict[int, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    summary = json.loads(
        (SAMPLE_ROOT / sample / "summary.json").read_text(encoding="utf-8")
    )
    records: dict[str, dict[str, Any]] = {}
    for rung, reconstructed in baselines.items():
        total, metrics = _output_record(x, weight, reconstructed)
        prior = float(summary["formats"][f"K{rung}"]["total_error"])
        records[str(rung)] = {
            "total_error": total,
            "tail_metrics": metrics,
            "tier3_summary_total_error": prior,
            "tier3_summary_relative_drift": abs(total - prior) / max(prior, 1e-30),
        }
    return records


def run_rotation(out_dir: Path, device: torch.device, results: dict[str, Any]) -> None:
    from prismaquant import format_registry as fr
    from prismaquant.measure_quant_cost import _cb_cost_quantize_dequantize

    experiment = results.setdefault("experiment_a", {})
    experiment.update({
        "method": "right randomized Hadamard H = D F; bf16 folded W@H and X@H",
        "seeds": list(SEEDS),
        "rungs": list(RUNGS),
        "samples": experiment.get("samples", {}),
    })
    for sample in SAMPLES:
        print(f"[A] {sample}", flush=True)
        weight, acts, baselines = _sample_tensors(sample, device)
        sample_record = experiment["samples"].setdefault(sample, {})
        baseline_records = _baseline_records(sample, acts, weight, baselines)
        sample_record["shape"] = list(weight.shape)
        sample_record["activation_shape"] = list(acts.shape)
        sample_record["unrotated"] = baseline_records
        sample_record.setdefault("seeds", {})
        reference_output = acts.to(torch.float32) @ weight.to(torch.float32).T
        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)
            signs = random_hadamard_signs(weight.shape[1], seed=seed, device=device)
            rotated_weight, rotation_weight_seconds = _timed(
                device,
                lambda: randomized_hadamard_right(
                    weight, signs, output_dtype=torch.bfloat16
                ),
            )
            rotated_acts, rotation_acts_seconds = _timed(
                device,
                lambda: randomized_hadamard_right(
                    acts, signs, output_dtype=torch.bfloat16
                ),
            )
            rotated_output = rotated_acts.to(torch.float32) @ rotated_weight.to(
                torch.float32
            ).T
            equivalence_gap = relative_frobenius_gap(reference_output, rotated_output)
            if equivalence_gap > 5e-3:
                raise RuntimeError(
                    f"{sample} seed={seed}: serving-equivalence gap "
                    f"{equivalence_gap:.6g} exceeds 5e-3"
                )
            col_weights = rotated_acts.to(torch.float32).square().mean(dim=0)
            seed_record = {
                "serving_equivalence_relative_frobenius_gap": equivalence_gap,
                "rotation_weight_seconds": rotation_weight_seconds,
                "rotation_acts_seconds": rotation_acts_seconds,
                "rungs": {},
            }
            for rung in RUNGS:
                name = f"NVFP4_CB_K{rung}"
                reconstructed, encode_seconds = _timed(
                    device,
                    lambda name=name: _cb_cost_quantize_dequantize(
                        fr.get_format(name),
                        rotated_weight.clone(),
                        col_weights=col_weights,
                    ),
                )
                total, metrics = _output_record(
                    rotated_acts, rotated_weight, reconstructed
                )
                baseline = baseline_records[str(rung)]["total_error"]
                seed_record["rungs"][str(rung)] = {
                    "total_error": total,
                    "error_ratio_rotated_over_unrotated": total / baseline,
                    "tail_metrics": metrics,
                    "gini_before": baseline_records[str(rung)]["tail_metrics"]["gini"],
                    "gini_after": metrics["gini"],
                    "encode_seconds": encode_seconds,
                }
                print(
                    f"    K{rung}: ratio={total / baseline:.6f} "
                    f"gini={metrics['gini']:.4f} encode={encode_seconds:.2f}s",
                    flush=True,
                )
                del reconstructed
            sample_record["seeds"][str(seed)] = seed_record
            _save_json(out_dir / "results.json", results)
            del signs, rotated_weight, rotated_acts, rotated_output, col_weights
            _release()
        del reference_output, weight, acts, baselines
        _release()


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
    return nvfp4_cb_reconstruct(
        fields, k, grid=family.grid, mode=family.mode
    ).to(dtype)


def run_feedback(
    out_dir: Path,
    device: torch.device,
    results: dict[str, Any],
    *,
    block_size: int,
    damping_fraction: float,
) -> None:
    experiment = results.setdefault("experiment_b", {})
    experiment.update({
        "method": (
            "fixed production codebook+scales; 64-column block GPTQ update; "
            "nearest complete 8-wide product-CB vectors under production diagonal imatrix"
        ),
        "block_size": block_size,
        "damping_fraction": damping_fraction,
        "rungs": list(RUNGS),
        "samples": experiment.get("samples", {}),
    })
    for sample in SAMPLES:
        print(f"[B] {sample}", flush=True)
        weight, acts, baselines = _sample_tensors(sample, device)
        col_weights = acts.to(torch.float32).square().mean(dim=0)
        hessian, hessian_seconds = _timed(
            device,
            lambda: acts.to(torch.float32).T @ acts.to(torch.float32),
        )
        (upper, damping), factor_seconds = _timed(
            device,
            lambda: inverse_hessian_cholesky(
                hessian, damping_fraction=damping_fraction
            ),
        )
        del hessian
        sample_record = experiment["samples"].setdefault(sample, {})
        sample_record.update({
            "shape": list(weight.shape),
            "activation_shape": list(acts.shape),
            "hessian_seconds": hessian_seconds,
            "factor_seconds": factor_seconds,
            "damping_added": damping,
            "rungs": {},
        })
        for rung in RUNGS:
            print(f"  K{rung}", flush=True)
            fields, production_encode_seconds = _timed(
                device,
                lambda rung=rung: _production_fields(weight, col_weights, rung),
            )
            production_reconstruction = _reconstruct(fields, rung, weight.dtype)
            production_total, production_metrics = _output_record(
                acts, weight, production_reconstruction
            )
            saved_total, _ = _output_record(acts, weight, baselines[rung])
            fixed_render_gap = relative_frobenius_gap(
                baselines[rung], production_reconstruction
            )
            plain, plain_seconds = _timed(
                device,
                lambda: reassign_product_cb(
                    weight,
                    fields,
                    col_weights,
                    block_size=block_size,
                ),
            )
            plain_total, _ = _output_record(acts, weight, plain)
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
            feedback_total, feedback_metrics = _output_record(
                acts, weight, feedback
            )
            factor_share = (hessian_seconds + factor_seconds) / len(RUNGS)
            sample_record["rungs"][str(rung)] = {
                "saved_production_total_error": saved_total,
                "reencoded_production_total_error": production_total,
                "reencoded_over_saved_error_ratio": production_total / saved_total,
                "reencoded_weight_relative_frobenius_gap_vs_saved": fixed_render_gap,
                "plain_reassignment_total_error": plain_total,
                "plain_over_production_error_ratio": plain_total / production_total,
                "feedback_total_error": feedback_total,
                "feedback_over_production_error_ratio": feedback_total / production_total,
                "feedback_over_plain_error_ratio": feedback_total / plain_total,
                "production_tail_metrics": production_metrics,
                "feedback_tail_metrics": feedback_metrics,
                "production_encode_seconds": production_encode_seconds,
                "plain_assignment_seconds": plain_seconds,
                "feedback_assignment_seconds": feedback_seconds,
                "feedback_assignment_over_plain_time": feedback_seconds / plain_seconds,
                "amortized_factor_seconds": factor_share,
                "feedback_plus_amortized_factor_over_plain_time": (
                    feedback_seconds + factor_share
                ) / plain_seconds,
                "prototype_total_over_production_encode_time": (
                    production_encode_seconds + feedback_seconds + factor_share
                ) / production_encode_seconds,
            }
            print(
                f"    error ratio={feedback_total / production_total:.6f} "
                f"assign={feedback_seconds / plain_seconds:.2f}x "
                f"total={sample_record['rungs'][str(rung)]['prototype_total_over_production_encode_time']:.2f}x",
                flush=True,
            )
            _save_json(out_dir / "results.json", results)
            del fields, production_reconstruction, plain, feedback
            _release()
        del upper, weight, acts, baselines, col_weights
        _release()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("rotpilot-out"))
    parser.add_argument("--sample-root", type=Path, default=SAMPLE_ROOT)
    parser.add_argument("--ext-dir", type=Path, default=Path(".ext"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", choices=("a", "b", "both"), default="both")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--damping-fraction", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> int:
    global SAMPLE_ROOT
    args = _parser().parse_args(argv)
    SAMPLE_ROOT = args.sample_root.resolve()
    _configure_environment(args.ext_dir.resolve())
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real-tensor pilot requires CUDA")
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / "results.json"
    results = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.is_file()
        else {}
    )
    results.update({
        "schema": "prismaquant.rotation_ldlq_pilot.v1",
        "sample_root": str(SAMPLE_ROOT),
        "production_encoder_context": {
            **PRODUCTION_ENV,
            "PRISMAQUANT_CB_EXT_DIR": str(args.ext_dir.resolve()),
        },
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    })
    if args.phase in ("a", "both"):
        run_rotation(args.out, device, results)
    if args.phase in ("b", "both"):
        run_feedback(
            args.out,
            device,
            results,
            block_size=args.block_size,
            damping_fraction=args.damping_fraction,
        )
    _save_json(result_path, results)
    print(f"wrote {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
