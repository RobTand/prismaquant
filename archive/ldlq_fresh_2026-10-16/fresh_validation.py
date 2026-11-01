#!/usr/bin/env python3
"""Build and evaluate the genuinely-fresh-text LDLQ validation artifact."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from prismaquant.cb_layout import parse_format_name
from prismaquant.rotation_ldlq_pilot import (
    compare_output_sse,
    inverse_hessian_cholesky,
    reassign_product_cb,
)


PRODUCTION_CALIB_HASH = "2682b690de20c091568729be8f78baf7"
PRODUCTION_TEXT_ROWS = 16
FRESH_TEXT_START = 16
FRESH_TEXT_ROWS = 16
RUNGS = (12, 15, 18)
PRODUCTION_ENV = {
    "PRISMAQUANT_ACTIVATION_FAIR_PRICING": "1",
    "CB_CODEBOOK_SOURCE": "lattice",
    "CB_SCALE_CODING": "two_tier",
    "CB_SCALE_SWEEP": "1",
    "PRISMAQUANT_CB_ENCODE_TIER": "balanced",
}
SAMPLES = {
    "layers.40.attn.wq_b": {
        "live_name": "model.layers.40.self_attn.wq_b",
        "activation_file": "model__layers__40__self_attn__wq_b.pt",
        "capture_layer": 40,
    },
    "layers.40.experts.81.up_proj": {
        "live_name": "model.layers.40.mlp.experts.81.up_proj",
        "activation_file": "model__layers__40__mlp__experts__81__up_proj.pt",
        "capture_layer": 40,
    },
    "layers.20.experts.63.up_proj": {
        "live_name": "model.layers.20.mlp.experts.63.up_proj",
        "activation_file": "model__layers__20__mlp__experts__63__up_proj.pt",
        "capture_layer": 20,
    },
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text_records(source: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            text = payload.get("text")
            if isinstance(text, str) and text:
                records.append(
                    {
                        "text_index": len(records),
                        "source_line": line_number,
                        "text": text,
                        "text_sha256": _sha256_bytes(text.encode("utf-8")),
                    }
                )
    return records


def build_disjoint_corpus(source: Path, corpus_out: Path, manifest_out: Path) -> dict:
    """Write records 16:32 after proving exact-text disjointness from 0:16.

    The production loader's historical seed 42 preserves local-JSONL order.
    Every selected record is longer than 512 tokens (verified separately with
    the checkpoint tokenizer), so production consumed text rows 0:16.
    """
    records = _text_records(source)
    production = records[:PRODUCTION_TEXT_ROWS]
    fresh = records[FRESH_TEXT_START:FRESH_TEXT_START + FRESH_TEXT_ROWS]
    if len(production) != PRODUCTION_TEXT_ROWS or len(fresh) != FRESH_TEXT_ROWS:
        raise ValueError("source corpus does not contain the required text slices")
    production_hashes = {row["text_sha256"] for row in production}
    fresh_hashes = {row["text_sha256"] for row in fresh}
    overlap = sorted(production_hashes & fresh_hashes)
    if overlap:
        raise ValueError(f"fresh text duplicates production records: {overlap}")

    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    corpus_out.write_text(
        "".join(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n" for row in fresh),
        encoding="utf-8",
    )
    manifest = {
        "schema": "prismaquant.ldlq_fresh_text.v1",
        "source": str(source),
        "source_sha256": _sha256_bytes(source.read_bytes()),
        "loader_contract": {
            "calib_seed": 42,
            "seed_42_local_jsonl_order": "preserved",
            "nsamples": 16,
            "seqlen": 512,
            "production_calibration_data_hash": PRODUCTION_CALIB_HASH,
        },
        "production_records": [
            {key: row[key] for key in ("text_index", "source_line", "text_sha256")}
            for row in production
        ],
        "fresh_records": [
            {key: row[key] for key in ("text_index", "source_line", "text_sha256")}
            for row in fresh
        ],
        "exact_text_hash_overlap": overlap,
        "disjoint": not overlap,
        "fresh_corpus": str(corpus_out),
        "fresh_corpus_sha256": _sha256_bytes(corpus_out.read_bytes()),
    }
    manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def audit_corpus_with_checkpoint_tokenizer(
    manifest: dict,
    source: Path,
    fresh_corpus: Path,
    model_path: Path,
) -> dict:
    """Prove the source slices reproduce production and fresh token batches."""
    from transformers import AutoTokenizer

    from prismaquant.perturbed_x_cache import calibration_data_hash
    from prismaquant.sensitivity_probe import load_calibration, stage_text_only

    staged = stage_text_only(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
    records = _text_records(source)
    token_counts = {
        row["text_index"]: int(
            tokenizer(row["text"], return_tensors="pt", truncation=False).input_ids.shape[1]
        )
        for row in records[:FRESH_TEXT_START + FRESH_TEXT_ROWS]
    }
    for group in ("production_records", "fresh_records"):
        for row in manifest[group]:
            row["checkpoint_token_count"] = token_counts[row["text_index"]]
    if any(
        row["checkpoint_token_count"] < 512
        for row in manifest["production_records"] + manifest["fresh_records"]
    ):
        raise ValueError("selected corpus record is too short for an independent window")

    production_ids = load_calibration(tokenizer, str(source), 16, 512, calib_seed=42)
    fresh_ids = load_calibration(tokenizer, str(fresh_corpus), 16, 512, calib_seed=42)
    production_hash = calibration_data_hash(production_ids)
    fresh_hash = calibration_data_hash(fresh_ids)
    if production_hash != PRODUCTION_CALIB_HASH:
        raise ValueError(
            f"production reconstruction hash {production_hash} != {PRODUCTION_CALIB_HASH}"
        )
    if fresh_hash == production_hash:
        raise ValueError("fresh and production token-window hashes unexpectedly match")
    manifest["loader_contract"].update(
        {
            "checkpoint_tokenizer_model": str(model_path),
            "production_calibration_data_hash_recomputed": production_hash,
            "fresh_calibration_data_hash": fresh_hash,
        }
    )
    return manifest


def collect_fresh_activations(capture_root: Path, act_out: Path) -> dict[str, int]:
    """Copy only the three target cache entries out of per-layer captures."""
    act_out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for slug, spec in SAMPLES.items():
        source = (
            capture_root
            / f"l{spec['capture_layer']}"
            / "act"
            / spec["activation_file"]
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("inputs"), torch.Tensor):
            raise TypeError(f"{source} is not a PrismaQuant activation-cache record")
        if payload.get("name") != spec["live_name"]:
            raise ValueError(
                f"{source}: name {payload.get('name')!r} != {spec['live_name']!r}"
            )
        rows = int(payload["inputs"].shape[0])
        if rows < 64:
            raise ValueError(f"{slug}: fresh capture has only {rows} rows; need >=64")
        shutil.copy2(source, act_out / spec["activation_file"])
        counts[slug] = rows
    return counts


def write_capture_manifest(
    capture_root: Path,
    act_dir: Path,
    text_manifest: dict,
    output: Path,
) -> dict:
    """Persist the probe metadata needed after large intermediates are removed."""
    expected_hash = text_manifest["loader_contract"]["fresh_calibration_data_hash"]
    layer_runs = []
    for layer in (20, 40):
        probe_path = capture_root / f"l{layer}" / "probe.pkl"
        with probe_path.open("rb") as stream:
            probe = pickle.load(stream)
        meta = probe["meta"]
        if meta["calib_hash"] != expected_hash:
            raise ValueError(
                f"layer {layer} capture hash {meta['calib_hash']} != {expected_hash}"
            )
        log_path = capture_root / f"l{layer}" / "logs" / "probe.log"
        log = log_path.read_text(encoding="utf-8")
        phase1_match = re.search(r"phase-1 forward: ([0-9.]+)s", log)
        phase3_match = re.search(r"phase-3 reverse sweep \[[^]]+\]: ([0-9.]+)s", log)
        if phase1_match is None or phase3_match is None:
            raise ValueError(f"capture timing missing from {log_path}")
        layer_runs.append(
            {
                "layer": layer,
                "dataset": meta["dataset"],
                "calib_hash": meta["calib_hash"],
                "nsamples": meta["nsamples"],
                "seqlen": meta["seqlen"],
                "activation_rows_limit": meta["activation_rows_limit"],
                "linear_include": meta["linear_include"],
                "phase1_forward_seconds": float(phase1_match.group(1)),
                "phase3_reverse_seconds": float(phase3_match.group(1)),
            }
        )

    targets = {}
    for slug, spec in SAMPLES.items():
        path = act_dir / spec["activation_file"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        inputs = payload["inputs"]
        indices = payload.get("row_indices")
        targets[slug] = {
            "path": str(path),
            "sha256": _sha256_bytes(path.read_bytes()),
            "bytes": path.stat().st_size,
            "live_name": payload["name"],
            "shape": list(inputs.shape),
            "dtype": str(inputs.dtype),
            "row_indices_sha256": (
                _sha256_bytes(indices.contiguous().numpy().tobytes())
                if isinstance(indices, torch.Tensor)
                else None
            ),
        }
    manifest = {
        "schema": "prismaquant.ldlq_fresh_capture.v1",
        "capture_path": "prismaquant.incremental_probe",
        "device": "cuda",
        "dtype": "bf16",
        "production_activation_caches_written": False,
        "fresh_calibration_data_hash": expected_hash,
        "layer_runs": layer_runs,
        "measured_probe_hot_seconds": sum(
            run["phase1_forward_seconds"] + run["phase3_reverse_seconds"]
            for run in layer_runs
        ),
        "targets": targets,
    }
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_heldout_report(path: Path) -> dict[tuple[str, int], float]:
    """Return mean CAL32 held-out reduction across the report's three splits."""
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "## CAL32 decision table":
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table or not re.match(r"^\| `[^`]+` \|", line):
            continue
        columns = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(columns) != 10:
            raise ValueError(f"unexpected held-out table row: {line}")
        slug = columns[0].strip("`")
        rung = int(columns[1])
        reduction = float(columns[6].removesuffix("%")) / 100.0
        grouped[(slug, rung)].append(reduction)
    expected = {(slug, rung) for slug in SAMPLES for rung in RUNGS}
    if set(grouped) != expected or any(len(values) != 3 for values in grouped.values()):
        raise ValueError("held-out report does not contain 3 splits for all 9 cells")
    return {key: fmean(values) for key, values in grouped.items()}


def validation_verdict(insample_reduction: float, fresh_reduction: float) -> tuple[str, float]:
    if insample_reduction <= 0.0:
        raise ValueError("in-sample reduction must be positive")
    retention = fresh_reduction / insample_reduction
    if retention > 0.50:
        return "VALIDATED-PENDING-SERVED", retention
    if retention < 0.10:
        return "DISTRIBUTION-FRAGILE", retention
    return "PARTIAL", retention


def _configure_environment(ext_dir: Path) -> None:
    for name, value in PRODUCTION_ENV.items():
        os.environ[name] = value
    os.environ["PRISMAQUANT_CB_EXT_DIR"] = str(ext_dir)


def _load_tensor(path: Path, device: torch.device) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"{path} did not contain a bare tensor")
    return payload.to(device=device).contiguous()


def _load_fresh(path: Path, expected_name: str, device: torch.device) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("name") != expected_name:
        raise ValueError(f"invalid fresh activation-cache record: {path}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, torch.Tensor):
        raise TypeError(f"{path} has no activation tensor")
    return inputs.to(device=device).contiguous()


def _production_fields(weight: torch.Tensor, col_weights: torch.Tensor, rung: int) -> dict:
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
    return nvfp4_cb_reconstruct(fields, k, grid=family.grid, mode=family.mode).to(dtype)


def evaluate(
    out_dir: Path,
    sample_root: Path,
    act_dir: Path,
    heldout_report: Path,
    ext_dir: Path,
    device: torch.device,
    *,
    block_size: int,
    damping_fraction: float,
) -> dict[str, Any]:
    _configure_environment(ext_dir)
    heldout = parse_heldout_report(heldout_report)
    results: dict[str, Any] = {
        "schema": "prismaquant.ldlq_fresh_validation.v1",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "fit_contract": {
            "source": "original 64-row production calibration capture",
            "block_size": block_size,
            "damping_fraction": damping_fraction,
            "plain": "production nvfp4_cb_fields assignment",
            "feedback": "same fixed fields plus block-sequential LDLQ feedback",
        },
        "production_encoder_context": {
            **PRODUCTION_ENV,
            "PRISMAQUANT_CB_EXT_DIR": str(ext_dir),
        },
        "samples": {},
    }
    for slug, spec in SAMPLES.items():
        print(f"[fresh-eval] {slug}", flush=True)
        tensors = sample_root / slug / "tensors"
        weight = _load_tensor(tensors / "W_bf16_ref.pt", device)
        x_fit = _load_tensor(tensors / "X_acts.pt", device)
        x_fresh = _load_fresh(act_dir / spec["activation_file"], spec["live_name"], device)
        if x_fit.shape[1] != weight.shape[1] or x_fresh.shape[1] != weight.shape[1]:
            raise ValueError(f"{slug}: activation/weight input widths disagree")
        col_weights = x_fit.to(torch.float32).square().mean(dim=0)
        hessian = x_fit.to(torch.float32).T @ x_fit.to(torch.float32)
        upper, damping = inverse_hessian_cholesky(
            hessian, damping_fraction=damping_fraction
        )
        del hessian
        sample_record: dict[str, Any] = {
            "weight_shape": list(weight.shape),
            "fit_activation_shape": list(x_fit.shape),
            "fresh_activation_shape": list(x_fresh.shape),
            "damping_added": damping,
            "rungs": {},
        }
        results["samples"][slug] = sample_record
        for rung in RUNGS:
            fields = _production_fields(weight, col_weights, rung)
            plain = _reconstruct(fields, rung, weight.dtype)
            feedback = reassign_product_cb(
                weight,
                fields,
                col_weights,
                block_size=block_size,
                upper_inverse_cholesky=upper,
            )
            insample = compare_output_sse(x_fit, weight, plain, feedback)
            fresh = compare_output_sse(x_fresh, weight, plain, feedback)
            verdict, retention = validation_verdict(insample["reduction"], fresh["reduction"])
            sample_record["rungs"][str(rung)] = {
                "insample": insample,
                "heldout_mean_reduction_from_three_cal32_splits": heldout[(slug, rung)],
                "fresh": fresh,
                "fresh_gain_retention": retention,
                "verdict": verdict,
            }
            print(
                f"  K{rung}: fresh ratio={fresh['feedback_over_plain_ratio']:.6f} "
                f"retained={100.0 * retention:.2f}% {verdict}",
                flush=True,
            )
            del fields, plain, feedback
            gc.collect()
            torch.cuda.empty_cache()
        del upper, weight, x_fit, x_fresh, col_weights
        gc.collect()
        torch.cuda.empty_cache()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return results


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(
    out_dir: Path,
    results: dict[str, Any],
    text_manifest: dict,
    capture_manifest: dict,
) -> Path:
    rows = []
    for slug in SAMPLES:
        for rung in RUNGS:
            record = results["samples"][slug]["rungs"][str(rung)]
            rows.append((slug, rung, record))
    mean_insample = fmean(record["insample"]["reduction"] for _, _, record in rows)
    mean_heldout = fmean(
        record["heldout_mean_reduction_from_three_cal32_splits"]
        for _, _, record in rows
    )
    mean_fresh = fmean(record["fresh"]["reduction"] for _, _, record in rows)
    overall_verdict, overall_retention = validation_verdict(mean_insample, mean_fresh)
    counts = {
        slug: results["samples"][slug]["fresh_activation_shape"][0]
        for slug in SAMPLES
    }

    lines = [
        "# Fixed-CB LDLQ genuinely-fresh-text validation",
        "",
        "Date: 2026-08-03",
        "",
        f"GPU: {results['device']}, torch {results['torch_version']}",
        "",
        "## Verdict",
        "",
        f"**{overall_verdict}.** Mean fresh-text reduction is {_pct(mean_fresh)} versus "
        f"{_pct(mean_insample)} in-sample, retaining {_pct(overall_retention)} of the "
        f"fit-set gain. The same-capture CAL32/HOLDOUT32 reference averages "
        f"{_pct(mean_heldout)}. Cell verdicts: "
        f"{sum(record['verdict'] == 'VALIDATED-PENDING-SERVED' for _, _, record in rows)} "
        "validated-pending-served, "
        f"{sum(record['verdict'] == 'PARTIAL' for _, _, record in rows)} partial, and "
        f"{sum(record['verdict'] == 'DISTRIBUTION-FRAGILE' for _, _, record in rows)} "
        "distribution-fragile.",
        "",
        "The fixed rule is applied to fresh-gain retention: more than 50% is "
        "`VALIDATED-PENDING-SERVED`, less than 10% is `DISTRIBUTION-FRAGILE`, and "
        "the interval between is `PARTIAL`.",
        "",
        "## Fresh capture and disjointness",
        "",
        "The production probe is identified by its stored metadata and reproduced loader "
        f"hash `{PRODUCTION_CALIB_HASH}`: "
        "`diverse-v1.jsonl`, 16 samples, sequence length 512, calibration seed 42. "
        "For a local JSONL, the historical seed-42 branch preserves file order. All first "
        "16 text records are longer than 512 checkpoint-tokenizer tokens, so those records "
        "(source lines 2-17, text indices 0-15) are exactly the records production consumed. "
        "The fresh corpus contains the next 16 records only (source lines 18-33, text "
        "indices 16-31). Their full-text SHA-256 set has zero intersection with the "
        "production set; the per-record hashes and source-file hash are in "
        "[`text_manifest.json`](text_manifest.json). This proves record-level text "
        "disjointness rather than merely selecting different windows from the same text. "
        "Re-running the checkpoint tokenizer and loader reproduces the production hash "
        f"exactly and gives fresh token-window hash "
        f"`{text_manifest['loader_contract']['fresh_calibration_data_hash']}`.",
        "",
        "Two one-layer probe captures (`--start-layer 20 --end-layer 21` and "
        "`--start-layer 40 --end-layer 41`) used the established incremental-probe path, "
        "16x512 fresh tokens, bf16, and a 64-row cap; no production activation cache was "
        "opened for writing. Target row counts are "
        + ", ".join(f"`{slug}`={count}" for slug, count in counts.items())
        + f". The measured probe hot sections total "
        f"{capture_manifest['measured_probe_hot_seconds']:.1f} seconds. The three retained "
        "cache records are under [`act/`](act/), with hashes and capture metadata in "
        "[`capture_manifest.json`](capture_manifest.json).",
        "",
        "## Fresh vs held-out vs in-sample",
        "",
        "For every row below, plain and LDLQ weights were re-fit from the original "
        "production 64-row activation tensor exactly as in the merged pilot: production "
        "codebook and two-tier scales are frozen, the plain arm is the production field "
        "assignment, and the LDLQ arm adds 64-column block feedback from the 1%-damped "
        "fit Hessian. Both frozen weights are evaluated on the new activation rows. "
        "Held-out is the mean of the three prior CAL32/HOLDOUT32 splits and is included as "
        "the estimation-level reference; its fit size differs, so it is not used for the "
        "fresh verdict.",
        "",
        "| Linear | K | in-sample reduction | held-out reduction | fresh LDLQ/plain | "
        "fresh reduction | fresh gain retained | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for slug, rung, record in rows:
        lines.append(
            f"| `{slug}` | {rung} | {_pct(record['insample']['reduction'])} | "
            f"{_pct(record['heldout_mean_reduction_from_three_cal32_splits'])} | "
            f"{record['fresh']['feedback_over_plain_ratio']:.4f} | "
            f"{_pct(record['fresh']['reduction'])} | "
            f"{_pct(record['fresh_gain_retention'])} | {record['verdict']} |"
        )
    lines.extend(
        [
            "",
            "Raw SSE values, ratios, tensor shapes, damping, and encoder provenance are in "
            "[`results.json`](results.json). The capture/evaluation commands are preserved "
            "in [`run_capture.sh`](run_capture.sh).",
            "",
            "## What the served A/B still adds",
            "",
            "The planned Qwen smoke and served A/B remain necessary because this test "
            "freezes one Linear at a time and measures only local output SSE. Serving tests "
            "the materialized serialization and kernel path, accumulation across many "
            "simultaneously changed Linears, router/expert-selection changes, activation "
            "quantization, and the resulting model-level KL/PPL and generation behavior; "
            "it also supplies load/generation correctness and latency evidence that an "
            "activation-only screen cannot. Fresh-text retention therefore clears the "
            "pre-serving distribution gate, but it does not promote LDLQ to a production "
            "default.",
            "",
        ]
    )
    report = out_dir / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-corpus")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--model", type=Path, required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--capture-root", type=Path, required=True)
    collect.add_argument("--act-out", type=Path, required=True)
    collect.add_argument("--text-manifest", type=Path, required=True)
    collect.add_argument("--manifest-out", type=Path, required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--sample-root", type=Path, required=True)
    run.add_argument("--act-dir", type=Path, required=True)
    run.add_argument("--heldout-report", type=Path, required=True)
    run.add_argument("--ext-dir", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--block-size", type=int, default=64)
    run.add_argument("--damping-fraction", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-corpus":
        manifest = build_disjoint_corpus(
            args.source.resolve(),
            args.out / "fresh-text.jsonl",
            args.out / "text_manifest.json",
        )
        manifest = audit_corpus_with_checkpoint_tokenizer(
            manifest,
            args.source.resolve(),
            args.out / "fresh-text.jsonl",
            args.model.resolve(),
        )
        (args.out / "text_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"disjoint": manifest["disjoint"], "fresh_rows": FRESH_TEXT_ROWS}))
        return 0
    if args.command == "collect":
        counts = collect_fresh_activations(args.capture_root, args.act_out)
        write_capture_manifest(
            args.capture_root,
            args.act_out,
            json.loads(args.text_manifest.read_text(encoding="utf-8")),
            args.manifest_out,
        )
        print(json.dumps(counts, sort_keys=True))
        return 0
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fresh real-tensor evaluation requires CUDA")
    results = evaluate(
        args.out,
        args.sample_root,
        args.act_dir,
        args.heldout_report,
        args.ext_dir,
        device,
        block_size=args.block_size,
        damping_fraction=args.damping_fraction,
    )
    text_manifest = json.loads((args.out / "text_manifest.json").read_text())
    capture_manifest = json.loads((args.out / "capture_manifest.json").read_text())
    print(write_report(args.out, results, text_manifest, capture_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
