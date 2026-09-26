#!/usr/bin/env python3
"""Fail-closed integrity inventory for the completed DSV4 phase-A truth."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safetensors import safe_open

from prismaquant.cb_warm_state import warm_serialization_context
from prismaquant.nvfp4_cb_footprint import cb_serialization_context_stamp
from tools.dsv4_ldlq_cost_campaign import (
    COL_WEIGHTS,
    CONTEXT,
    PILOT_LAYER,
    PROJECTIONS,
    RUNGS,
    RUN_ROOT,
    SOURCE,
    atomic_json,
    atomic_text,
    canonical_cb_col_weights_sha256,
    sha256_file,
)


def _load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def run() -> int:
    manifest_path = RUN_ROOT / "PILOT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    checks = {
        "source_index_sha256": (
            sha256_file(SOURCE / "model.safetensors.index.json")
            == manifest["source_index_sha256"]
        ),
        "by_layer_sha256": (
            sha256_file(Path(manifest["by_layer"]["path"]))
            == manifest["by_layer"]["sha256"]
        ),
    }
    with COL_WEIGHTS.open("rb") as handle:
        col_weights = pickle.load(handle)
    checks["col_weights_sha256"] = (
        canonical_cb_col_weights_sha256(
            col_weights, sorted(
                qname for qname in col_weights
                if f".layers.{PILOT_LAYER}." in qname
            )
        ) == manifest["col_weights_sha256"]
    )
    formats = [f"FP8_CB_K{k}" for k in RUNGS]
    checks["serialization_context"] = (
        cb_serialization_context_stamp(CONTEXT, formats=formats)
        == manifest["cb_serialized_payload"]
    )
    inventory = []
    aggregate = hashlib.sha256(sha256_file(manifest_path).encode())
    for projection in PROJECTIONS:
        expected_qnames = [
            f"model.layers.{PILOT_LAYER}.mlp.experts.{expert}.{projection}"
            for expert in range(256)
        ]
        for rung in RUNGS:
            path = RUN_ROOT / "pilot-shards" / f"layer_021_{projection}_K{rung}.pkl"
            payload = _load(path)
            if payload.get("schema") != "prismaquant.dsv4_ldlq_projection_rung.v1":
                raise AssertionError(f"{path}: schema mismatch")
            if (
                payload.get("layer") != PILOT_LAYER
                or payload.get("projection") != projection
                or payload.get("rung") != rung
                or payload.get("format") != f"FP8_CB_K{rung}"
                or payload.get("qnames") != expected_qnames
            ):
                raise AssertionError(f"{path}: coordinate/content mismatch")
            for field in ("weight_mse_per_expert", "weighted_mse_per_expert"):
                values = payload.get(field)
                if not isinstance(values, list) or len(values) != 256:
                    raise AssertionError(f"{path}: {field} length mismatch")
                if any(not math.isfinite(float(v)) or float(v) < 0 for v in values):
                    raise AssertionError(f"{path}: {field} invalid")
            warm_path = Path(payload["warm_state_path"])
            with safe_open(warm_path, framework="pt", device="cpu") as handle:
                raw = (handle.metadata() or {}).get("prismaquant_cb_warm_state")
                metadata = json.loads(raw)
                if set(handle.keys()) != {"scales"}:
                    raise AssertionError(f"{warm_path}: unexpected tensor planes")
                source_shape = tuple(int(v) for v in metadata["source_shape"])
                if (
                    len(source_shape) != 3
                    or tuple(handle.get_tensor("scales").shape)
                    != (source_shape[0] * source_shape[1], 1)
                ):
                    raise AssertionError(f"{warm_path}: scale shape mismatch")
            logical = f"model.layers.{PILOT_LAYER}.mlp.experts.{projection}"
            if (
                metadata["qname"] != logical
                or metadata["format"] != f"FP8_CB_K{rung}"
                or metadata["serialization_context"]
                != warm_serialization_context(CONTEXT, f"FP8_CB_K{rung}")
            ):
                raise AssertionError(f"{warm_path}: warm identity mismatch")
            shard_sha = sha256_file(path)
            warm_sha = sha256_file(warm_path)
            aggregate.update(shard_sha.encode())
            aggregate.update(warm_sha.encode())
            inventory.append({
                "projection": projection, "rung": rung,
                "shard": str(path), "shard_sha256": shard_sha,
                "warm_state": str(warm_path), "warm_state_sha256": warm_sha,
                "source_digest": metadata["source_digest"],
                "col_weights_digest": metadata["col_weights_digest"],
            })
        mx = RUN_ROOT / "pilot-shards" / f"layer_021_{projection}_MXFP4.pkl"
        payload = _load(mx)
        if payload.get("format") != "MXFP4" or len(payload["weight_mse_per_expert"]) != 256:
            raise AssertionError(f"{mx}: MXFP4 menu row mismatch")
        mx_sha = sha256_file(mx)
        aggregate.update(mx_sha.encode())
        inventory.append({
            "projection": projection, "format": "MXFP4",
            "shard": str(mx), "shard_sha256": mx_sha,
        })
    for name in ("PILOT_FULL_MEASUREMENTS.pkl", "PILOT_FIT_REPORT.json", "PILOT_FIT_REPORT.md"):
        path = RUN_ROOT / name
        if not path.is_file():
            raise AssertionError(f"phase-A completion artifact absent: {path}")
        aggregate.update(sha256_file(path).encode())
    if not all(checks.values()):
        raise AssertionError(f"phase-A root checks failed: {checks}")
    report = {
        "schema": "prismaquant.dsv4_phase_a_integrity.v1",
        "result": "PASS",
        "layer": PILOT_LAYER,
        "root_checks": checks,
        "fp8_shards": len(PROJECTIONS) * len(RUNGS),
        "menu_shards": len(PROJECTIONS),
        "warm_states": len(PROJECTIONS) * len(RUNGS),
        "inventory_content_key": aggregate.hexdigest(),
        "inventory": inventory,
    }
    atomic_json(RUN_ROOT / "PHASE_A_INTEGRITY.json", report)
    atomic_text(RUN_ROOT / "PHASE_A_INTEGRITY.md", "\n".join([
        "# DSV4 Phase-A Integrity",
        "",
        "- Result: **PASS**",
        f"- Layer: {PILOT_LAYER}",
        f"- FP8 free-fit shards: {report['fp8_shards']}/{report['fp8_shards']}",
        f"- Menu shards: {report['menu_shards']}/{report['menu_shards']}",
        f"- Warm states: {report['warm_states']}/{report['warm_states']}",
        f"- Inventory content key: `{report['inventory_content_key']}`",
        "- Source index, by-layer base, canonical imatrix, serialization context, coordinates, vectors, warm metadata, and every file digest passed.",
        "",
    ]))
    print(f"[phase-a] integrity PASS key={report['inventory_content_key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
