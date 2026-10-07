"""Provisional block concentration from existing A8S bytes and saved calibration.

CPU-only research screen. No encoding, new rungs, allocation, capture or held-out
claim. The score is isolated baseline-residual output energy per existing body
bit, not a measured rate-change marginal. Cross-input-block terms are omitted
from the nonnegative concentration score and retained in the full output error.
With --conditional-repair, also report the signed change from restoring one
block to source precision while keeping every other block fixed. That oracle
retains cross-block coupling; it is not an admitted rung or an equal-byte arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import time

import numpy as np
import torch
from safetensors import safe_open
from tessera.fused import parse_fused
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.stock import materialize_stock, stock_dequant
from tessera.unit_artifact import parse_unit_artifact


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def concentration(scores: torch.Tensor) -> dict:
    flat = scores.flatten()
    if not bool(torch.isfinite(flat).all()) or bool((flat < 0).any()):
        raise ValueError("nonfinite or negative isolated-block score")
    total = float(flat.sum())
    if total <= 0:
        raise ValueError("zero total sensitivity has no concentration fraction")
    ordered = torch.sort(flat, descending=True, stable=True).values
    cumulative = ordered.cumsum(0)
    result = {"blocks": flat.numel(), "total_score": total}
    for percent in (50, 80):
        threshold = torch.tensor(total * percent / 100, dtype=flat.dtype)
        count = int(torch.searchsorted(cumulative, threshold).item()) + 1
        if not 1 <= count <= flat.numel():
            raise ValueError("concentration threshold outside score population")
        result[f"blocks_for_{percent}pct"] = count
        result[f"fraction_for_{percent}pct"] = count / flat.numel()
    return result


def block_scores(error: torch.Tensor, h: torch.Tensor,
                 row_span: int, col_span: int) -> torch.Tensor:
    rows, cols = error.shape
    if rows % row_span or cols % col_span:
        raise ValueError(f"{rows}x{cols} cannot tile {row_span}x{col_span}")
    nr, nc = rows // row_span, cols // col_span
    blocks = error.reshape(nr, row_span, nc, col_span).permute(0, 2, 1, 3).contiguous()
    hblocks = torch.stack([h[c:c + col_span, c:c + col_span]
                           for c in range(0, cols, col_span)]).double()
    weighted = blocks @ hblocks
    return (weighted * blocks).sum(dim=(-1, -2))


def prefix_block_h(x: torch.Tensor, col_span: int) -> torch.Tensor:
    rows, cols = x.shape
    blocks = x.reshape(rows, cols // col_span, col_span).permute(1, 0, 2)
    local = blocks.transpose(1, 2) @ blocks
    # Place only the block diagonal. No second persistent activation cache.
    h = torch.zeros(cols, cols, dtype=torch.float64)
    for i, start in enumerate(range(0, cols, col_span)):
        h[start:start + col_span, start:start + col_span] = local[i]
    return h


def check_prefix_blocks(scores: torch.Tensor, error: torch.Tensor,
                        x: torch.Tensor, rb: int, cb: int) -> float:
    nr, nc = scores.shape
    max_relative = 0.0
    for ri, ci in ((0, 0), (nr // 2, nc // 2), (nr - 1, nc - 1)):
        e = error[ri * rb:(ri + 1) * rb, ci * cb:(ci + 1) * cb]
        y = x[:, ci * cb:(ci + 1) * cb] @ e.t()
        direct = y.square().sum()
        torch.testing.assert_close(scores[ri, ci], direct, rtol=1e-9, atol=1e-12)
        max_relative = max(max_relative, float((scores[ri, ci] - direct).abs() / direct))
    return max_relative


def repair_gains(error: torch.Tensor, gradient: torch.Tensor, isolated: torch.Tensor,
                 baseline: torch.Tensor, count: int, rb: int, cb: int):
    rows, cols = error.shape
    dot = (error * gradient).reshape(rows // rb, rb, cols // cb, cb).sum(dim=(1, 3))
    gains = 2 * dot - isolated
    if not bool(torch.isfinite(gains).all()):
        raise ValueError("nonfinite conditional repair gain")
    torch.testing.assert_close(gains.sum(), 2 * baseline - isolated.sum(),
                               rtol=1e-10, atol=1e-9)
    extra_bits = rb * cb * (16 - 4)
    per_bit = gains / (count * extra_bits)
    positive = per_bit > 0
    summary = {
        "blocks": gains.numel(), "positive_blocks": int(positive.sum()),
        "negative_blocks": int((per_bit < 0).sum()),
        "zero_blocks": int((per_bit == 0).sum()),
        "positive_fraction": float(positive.double().mean()),
        "ideal_added_precision_bits_per_block": extra_bits,
        "baseline_mean_output_error": float(baseline / count),
        "summed_single_block_gain_mean": float(gains.sum() / count),
        "summed_positive_single_block_gains_mean": float(gains[gains > 0].sum() / count),
        "sum_of_gains_is_not_joint_gain": True,
        "not_a_rung_quote_or_equal_byte_arm": True,
        "positive_gain_concentration": concentration(per_bit.clamp_min(0))
            if bool(positive.any()) else None,
        "concentration_population": "all blocks; only positive repair gains contribute",
        "no_positive_gain_reason": None if bool(positive.any())
            else "no single block improves this output objective when restored alone",
    }
    return gains, summary


def check_repair_blocks(gains: torch.Tensor, error: torch.Tensor, x: torch.Tensor,
                        output_error: torch.Tensor, rb: int, cb: int) -> float:
    nr, nc = gains.shape
    max_abs = 0.0
    for ri, ci in ((0, 0), (nr // 2, nc // 2), (nr - 1, nc - 1)):
        patch = error[ri * rb:(ri + 1) * rb, ci * cb:(ci + 1) * cb]
        correction = x[:, ci * cb:(ci + 1) * cb] @ patch.t()
        old = output_error[:, ri * rb:(ri + 1) * rb]
        direct = (old.square() - (old - correction).square()).sum()
        torch.testing.assert_close(gains[ri, ci], direct, rtol=1e-9, atol=1e-10)
        max_abs = max(max_abs, float((gains[ri, ci] - direct).abs()))
    return max_abs


def decode_existing(export: Path, weight_map: dict, qname: str):
    if ".shared_experts." in qname:
        module = qname.rsplit(".", 1)[0] + ".gate_up_proj"
        key = module + ".wire_bytes"
    else:
        key = qname + ".wire"
    with safe_open(str(export / weight_map[key]), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(key)
        if tensor.dtype != torch.uint8 or tensor.ndim != 1:
            raise ValueError("baseline wire must be one byte stream")
        blob = tensor.numpy().tobytes()
    role = qname.rsplit(".", 1)[1]
    members = parse_fused(blob)
    matching = [member for member in members if member.name == role]
    if len(matching) != 1:
        raise ValueError(f"{key}: expected one {role}, found {[m.name for m in members]}")
    member = matching[0]
    parsed = parse_unit_artifact(member.blob, device="cpu")
    if parsed.grid.name != "E4M3" or parsed.body is not BodyKind.WINDOW \
            or parsed.unit.scale_plane is not ScalePlaneKind.CHANNEL \
            or set(parsed.unit.rates) != {4}:
        raise ValueError("provisional baseline must be existing uniform A8S E4M3/CHANNEL R4")
    rendered = stock_dequant(materialize_stock(parsed.unit, parsed.forests, parsed.code))
    if rendered.shape[0] != member.rows:
        raise ValueError("decoded rows disagree with fused framing")
    return rendered, {"tensor_key": key, "shard": weight_map[key],
                      "container_bytes": len(blob), "member_bytes": len(member.blob),
                      "container_sha256": hashlib.sha256(blob).hexdigest(),
                      "member_sha256": hashlib.sha256(member.blob).hexdigest(),
                      "stored_root_q256": parsed.manifest.branch.root_q256,
                      "stored_unit_name": parsed.manifest.branch.unit_id}


def measure(args, qname: str, captures: dict, source_map: dict, export_map: dict) -> dict:
    started = time.monotonic()
    entry = captures[qname]
    capture_path = args.capture_root / entry["path"]
    actual_sha = file_sha256(capture_path)
    if actual_sha != entry["sha256"]:
        raise ValueError(f"{capture_path}: own-byte digest does not match capture receipt")
    saved = torch.load(capture_path, map_location="cpu", weights_only=True)
    if saved["name"] != qname:
        raise ValueError("capture qname disagrees with requested source tensor")
    x, h = saved["inputs"], saved["hessian"]
    if x.ndim != 2 or h.shape != (x.shape[1], x.shape[1]) \
            or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(h).all()):
        raise ValueError("invalid saved calibration X/H")
    x = x.double()
    with safe_open(str(args.source / source_map[qname + ".weight"]),
                   framework="pt", device="cpu") as handle:
        source_tensor = handle.get_tensor(qname + ".weight")
        if source_tensor.dtype != torch.bfloat16:
            raise ValueError("the source-precision repair oracle requires BF16 source weights")
        source = source_tensor.float()
        del source_tensor
    rendered, wire = decode_existing(args.export, export_map, qname)
    if rendered.shape != source.shape or source.shape[1] != x.shape[1]:
        raise ValueError("source, actual baseline and saved calibration shapes disagree")
    error = (rendered - source).double()
    del rendered
    prefix_output = x @ error.t()
    output_error = prefix_output.square().sum()
    reference_energy = (x @ source.double().t()).square().sum()
    count = int(saved["count"])
    if count < x.shape[0] or count <= 0 or reference_energy <= 0:
        raise ValueError("invalid full count or zero reference output energy")
    if args.conditional_repair:
        prefix_gradient = prefix_output.t() @ x
        full_gradient = error @ h.double()
        full_error = (full_gradient * error).sum()
        if not bool(torch.isfinite(full_error)) or full_error <= 0:
            raise ValueError("full saved Hessian gives an invalid residual objective")
    geometries = {}
    raw = {}
    for geometry in args.geometry:
        rb, cb = (int(part) for part in geometry.split("x"))
        if rb <= 0 or cb <= 0:
            raise ValueError("block dimensions must be positive")
        hprefix = prefix_block_h(x, cb)
        prefix = block_scores(error, hprefix, rb, cb)
        max_check = check_prefix_blocks(prefix, error, x, rb, cb)
        full_h = block_scores(error, h, rb, cb)
        bits = rb * cb * 4
        prefix_per_bit = prefix / (x.shape[0] * bits)
        full_per_bit = full_h / (count * bits)
        geometries[geometry] = {
            "weight_positions_per_block": rb * cb, "existing_body_bits_per_block": bits,
            "retained_rows": concentration(prefix_per_bit),
            "full_capture_H": concentration(full_per_bit),
            "spotcheck_max_relative_error": max_check,
            "sum_isolated_prefix_energy": float(prefix.sum()),
            "full_prefix_output_error": float(output_error),
            "cross_block_term": float(output_error - prefix.sum()),
            "sum_isolated_is_not_full_error": True}
        raw[geometry + "_prefix"] = prefix_per_bit.numpy()
        raw[geometry + "_full_H"] = full_per_bit.numpy()
        if args.conditional_repair:
            prefix_gains, prefix_repair = repair_gains(
                error, prefix_gradient, prefix, output_error, x.shape[0], rb, cb)
            full_gains, full_repair = repair_gains(
                error, full_gradient, full_h, full_error, count, rb, cb)
            prefix_repair["spotcheck_max_absolute_error"] = check_repair_blocks(
                prefix_gains, error, x, prefix_output, rb, cb)
            geometries[geometry]["conditional_repair_retained_rows"] = prefix_repair
            geometries[geometry]["conditional_repair_full_H"] = full_repair
            raw[geometry + "_repair_gain_prefix"] = prefix_gains.numpy()
            raw[geometry + "_repair_gain_full_H"] = full_gains.numpy()
        del hprefix, prefix, full_h
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.out_dir / (qname.replace(".", "__") + ".npz")
    np.savez(scores_path, **raw)
    return {"qname": qname, "shape": list(source.shape), "wire": wire,
            "capture": {"path": str(capture_path), "sha256": actual_sha,
                        "saved_source": saved["source"], "retained_rows": x.shape[0],
                        "full_draw_count": count, "max_abs": saved["max_abs"]},
            "full_prefix_relative_output_error": float(output_error / reference_energy),
            "geometries": geometries, "score_arrays": str(scores_path),
            "score_arrays_sha256": file_sha256(scores_path),
            "elapsed_seconds": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--qname", action="append", required=True)
    parser.add_argument("--geometry", action="append", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--conditional-repair", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        raise RuntimeError("this authorized provisional screen is CPU only")
    capture_manifest = args.capture_root / "capture_manifest.json"
    captures = json.loads(capture_manifest.read_bytes())["entries"]
    source_map = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    export_map = json.loads((args.export / "model.safetensors.index.json").read_bytes())["weight_map"]
    result = {"schema": "prismaquant.provisional_block_concentration.v1",
              "host": socket.gethostname(), "python": platform.python_version(),
              "torch": torch.__version__, "provisional": True,
              "in_domain": False, "held_out": False, "allocation_gain_measured": False,
              "score": "tr(E_block H_block E_block^T)/(calibration_count * existing_body_bits)",
              "scope": "existing canonical calibration; isolated A8S baseline residual blocks, not rate-change marginal; cross-input-block terms excluded from concentration",
              "conditional_repair": args.conditional_repair,
              "capture_manifest_sha256": file_sha256(capture_manifest), "units": []}
    for qname in args.qname:
        row = measure(args, qname, captures, source_map, export_map)
        result["units"].append(row)
        print(json.dumps({"provisional": True, "unit": row}), flush=True)
    output = args.out_dir / "result.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(output), "sha256": file_sha256(output),
                      "provisional": True, "units_measured": len(result["units"])}), flush=True)


if __name__ == "__main__":
    main()
