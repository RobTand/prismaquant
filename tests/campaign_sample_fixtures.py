"""Build a small campaign for local tests."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from prismaquant.cost_stage_checkpoint import (
    MANIFEST_SCHEMA, canonical_json_sha256, unit_path, write_unit,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {"path": str(path), "sha256": sha(path)}


def installed_encoder_source_sha256():
    from tessera.cached_unit import encoder_source_sha256

    return encoder_source_sha256()


def fixture(tmp_path):
    names = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"]
    fmt = "TESSERA_E4M3_K1_R1024"
    root = tmp_path / "campaign"
    rowdir = root / "rows/row-0000"
    cache = rowdir / "cache"
    cache.mkdir(parents=True)
    wire = tmp_path / "merged/cache/wire"
    wire.mkdir(parents=True)
    from prismaquant.render_cache_paths import cache_weight_filename
    source = {"shape": [4, 4], "dtype": "bfloat16", "sha256": "a" * 64}
    encoder_seal = installed_encoder_source_sha256()
    identity = {"campaign_schema": "prismaquant.tessera_campaign_cost.v1",
                "currency": "output_mse_under_route_activation_contract",
                "prismaquant_source_sha256": "b" * 64, "encoder_source_sha256": encoder_seal,
                "calibration": {"fit_ids_sha256": "d" * 64},
                "units": {n: {"weight": source, "hessian": None,
                               "input_global_scale": None, "menu": [fmt],
                               "scoring_rows": {"shape": [2, 4], "sha256": "e" * 64}}
                          for n in names}}
    checkpoint = tmp_path / "merged/cost.anchors.json"
    parts = checkpoint.with_name(checkpoint.name + ".parts")
    seal = canonical_json_sha256(identity, where="fixture")
    manifest = {"schema": MANIFEST_SCHEMA, "stage": "Tessera campaign",
                "identity": identity, "identity_sha256": seal,
                "units": [{"qname": n, "file": str(unit_path(parts, n).relative_to(parts))}
                          for n in names]}
    checkpoint.write_text(json.dumps(manifest))
    costs, states = {}, {}
    for n in names:
        filename = n + ".tessera"
        blob = ("inert " + n).encode()
        (wire / filename).write_bytes(blob)
        (cache / cache_weight_filename(n, fmt)).write_bytes(b"inert render")
        anchor = {"qname": n, "format_name": fmt, "family": "TESSERA_E4M3_K1",
                  "body_rate_q256": 1024, "dloss": 0.25, "dloss_stderr": 0.0,
                  "memory_bytes": 16, "bits_per_param": 8.0,
                  "activation_contract": "fp8_e4m3", "activation_quantized": True,
                  "wire_bytes": len(blob), "seconds": 0.1, "hessian_applied": False,
                  "input_global_scale": None}
        record = {"file": filename, "blob_bytes": len(blob),
                  "blob_sha256": hashlib.sha256(blob).hexdigest(),
                  "identity": {"unit": n, "source": source, "calibration": None,
                               "encoder_source_sha256": encoder_seal,
                               "recipe": {"grid": "E4M3", "q256": 1024}}}
        states[n] = {"anchors": [anchor], "wire_records": {fmt: record}}
        write_unit(parts, stage="Tessera campaign", qname=n,
                   identity_sha256=seal, state=states[n])
        costs[n] = {fmt: {"output_mse": 0.25, "output_mse_measured": True,
                         "cost_source": "tessera_campaign_measured", "tessera_provenance": "measured",
                         "currency": identity["currency"], "tessera_family": anchor["family"],
                         "tessera_body_rate_q256": 1024, "activation_contract": "fp8_e4m3",
                         "activation_quantized": True, "wire_bytes": len(blob),
                         "hessian_identity": {"applied": False, "supplied": False}}}
    census = {"model": "/fixture/model", "unit_shapes": {n: [4, 4] for n in names},
              "anchor_groups": {"g:qk": names}, "max_abs": {names[0]: 1.0, names[1]: 2.0}}
    census_path = root / "census.json"
    census_path.write_text(json.dumps(census))
    plan = {"schema": "prismaquant.tessera_campaign_plan.v1", "model": census["model"],
            "census": str(census_path), "rows": [{"row_id": "row-0000", "members": names,
            "groups": ["g:qk"], "dir": str(rowdir)}]}
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan))
    receipts = root / "receipts.json"
    receipts.write_text(json.dumps({"returncode": 0, "rows": [{"key": "f" * 64,
        "rc": 0, "status": "executed", "host": "fixture"}]}))
    payload = {"schema": identity["campaign_schema"], "currency": identity["currency"],
               "costs": costs, "provenance": {"model": census["model"],
               "cost_mode": "production-render-score", "wire_dir": str(wire),
               "hessian": {"supplied": False}, "stopped_early": False,
               "campaign_fanout": {"schema": plan["schema"], "rows": {"row-0000": ["g:qk"]}},
               "activation_static_scales": {"units": {}, "policy": "fixture"}}}
    cost_path = tmp_path / "merged/cost.pkl"
    cost_path.write_bytes(pickle.dumps(payload))
    config = {"campaign_plan": bind(plan_path), "census": bind(census_path),
              "campaign_receipts": bind(receipts), "merged_cost": bind(cost_path),
              "merged_checkpoint": bind(checkpoint), "required_source_units": 2,
              "required_campaign_groups": 1}
    return config, names, fmt, payload, states
