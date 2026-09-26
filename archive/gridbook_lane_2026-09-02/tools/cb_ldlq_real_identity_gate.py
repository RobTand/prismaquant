#!/usr/bin/env python3
"""Fail-closed serial/batched LDLQ identity gate on real expert weights."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open

from prismaquant import format_registry as fr
from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_layout import parse_format_name
from prismaquant.cb_ldlq import CBLDLQActivationLoader
from prismaquant.model_profiles import detect_profile
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_fields_for_context,
)


def _load_source_slice(
    model_dir: Path,
    source_key: str,
    expert_ids: list[int],
) -> torch.Tensor:
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    shard = model_dir / index[source_key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        value = handle.get_tensor(source_key)
    return value[expert_ids].contiguous()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--activation-cache-dir", required=True)
    parser.add_argument("--col-weights", required=True)
    parser.add_argument("--qname", required=True)
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--formats", default="FP8_CB_K28,FP8_CB_K38")
    parser.add_argument("--experts", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--encode-tier", default="fast")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    device = torch.device(args.device)
    expert_ids = [int(value) for value in args.experts.split(",") if value]
    formats = [value for value in args.formats.split(",") if value]
    if len(expert_ids) < 8:
        raise SystemExit("identity gate requires at least 8 real experts")
    if len(formats) < 2:
        raise SystemExit("identity gate requires at least two CB rungs")

    weight = _load_source_slice(
        model_dir, args.source_key, expert_ids
    ).to(device)
    with open(args.col_weights, "rb") as handle:
        col_weight_table = pickle.load(handle)
    col_weights = torch.as_tensor(col_weight_table[args.qname])[
        expert_ids
    ].to(device)
    profile = detect_profile(model_dir)
    activation_stack = CBLDLQActivationLoader(
        args.activation_cache_dir,
        model_dir=model_dir,
        profile=profile,
        replay_device=str(device),
    ).load(args.qname, stack_size=int(col_weight_table[args.qname].shape[0]))
    if not isinstance(activation_stack, tuple):
        raise SystemExit(
            f"{args.qname}: expected per-expert activation rows, got shared tensor"
        )
    activation_rows = tuple(activation_stack[index] for index in expert_ids)

    report = {
        "schema": "prismaquant.cb_ldlq_real_identity.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir.resolve()),
        "profile": profile.name,
        "qname": args.qname,
        "source_key": args.source_key,
        "expert_ids": expert_ids,
        "weight_shape": list(weight.shape),
        "activation_row_counts": [int(value.shape[0]) for value in activation_rows],
        "formats": {},
        "passed": True,
    }
    context = CBSerializationContext.production(
        encode_tier=args.encode_tier,
        ldlq=False,
    )

    for format_name in formats:
        family, k = parse_format_name(format_name)
        spec = fr.get_format(format_name)
        fields = cb_fields_for_context(
            spec,
            weight,
            context=context,
            col_weights=col_weights,
        )
        _synchronize(device)
        started = time.perf_counter()
        serial = cb.ldlq_reassign_cb_fields(
            weight,
            fields,
            col_weights,
            activation_rows,
            grid=family.grid,
            mode=family.mode,
            batch_experts=False,
        )
        _synchronize(device)
        serial_seconds = time.perf_counter() - started
        started = time.perf_counter()
        batched = cb.ldlq_reassign_cb_fields(
            weight,
            fields,
            col_weights,
            activation_rows,
            grid=family.grid,
            mode=family.mode,
            batch_experts=True,
        )
        _synchronize(device)
        batched_seconds = time.perf_counter() - started

        field_equal = {
            key: bool(torch.equal(serial[key], batched[key]))
            for key in ("indices", "scales", "scale_super", "scale_sub")
            if key in serial
        }
        serial_reconstruction = cb.nvfp4_cb_reconstruct(
            serial, k, grid=family.grid, mode=family.mode
        )
        batched_reconstruction = cb.nvfp4_cb_reconstruct(
            batched, k, grid=family.grid, mode=family.mode
        )
        reconstruction_equal = bool(
            torch.equal(serial_reconstruction, batched_reconstruction)
        )
        serial_mse = []
        batched_mse = []
        for local, rows in enumerate(activation_rows):
            rows = rows.to(device=device, dtype=torch.float32)
            serial_mse.append(
                (
                    rows
                    @ (
                        weight[local].float()
                        - serial_reconstruction[local].float()
                    ).T
                ).square().mean()
            )
            batched_mse.append(
                (
                    rows
                    @ (
                        weight[local].float()
                        - batched_reconstruction[local].float()
                    ).T
                ).square().mean()
            )
        serial_mse_tensor = torch.stack(serial_mse)
        batched_mse_tensor = torch.stack(batched_mse)
        mse_equal = bool(torch.equal(serial_mse_tensor, batched_mse_tensor))
        passed = all(field_equal.values()) and reconstruction_equal and mse_equal
        report["formats"][format_name] = {
            "field_equal": field_equal,
            "index_difference_count": int(
                (serial["indices"] != batched["indices"]).sum().item()
            ),
            "reconstruction_equal": reconstruction_equal,
            "per_unit_output_mse_equal": mse_equal,
            "serial_output_mse": serial_mse_tensor.cpu().tolist(),
            "batched_output_mse": batched_mse_tensor.cpu().tolist(),
            "serial_seconds": serial_seconds,
            "batched_seconds": batched_seconds,
            "speedup": serial_seconds / batched_seconds,
            "passed": passed,
        }
        report["passed"] = bool(report["passed"] and passed)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
