"""The allocator's non-codebook output is unchanged by the lane excision (#1304).

Step 2 of #1304 removed the retired codebook lane's branches from the
allocator, the cost readers and the footprint. None of those branches could
run on a menu without a codebook rung, so a stock menu and a Tessera menu must
allocate to the same bytes as before. The digests below were computed on the
parent commit 1b29d4a6aff (PR #1332), before any branch was removed. They cover
``layer_config.json``, the Pareto CSV and, for the stock menu, every Pareto seed
file. Absolute temporary paths are replaced by ``<TMP>`` and the solver's
wall time is blanked before hashing.

One key changed on purpose. ``format_applicability.json`` carried
``cb_ladder_cross_family_verdict``, which was always ``null`` without a
codebook rung, and step 2 dropped it. Its two digests were computed on
1b29d4a6aff with that null key removed before hashing; every other file's
digest is the unmodified 1b29d4a6aff digest.

The Tessera ``layer.json`` digest was re-taken on 2026-09-26 for the Tessera
pin move to af7a86d43 (contract v38, PQ #1274). Diffing the normalised file
before and after, the allocation is unchanged and exactly three provenance
fields move: ``contract_version`` (34 to 38), ``reviewed_contract_sha256``, and
the routed unit's attesting cells. The last is the fixture's doing: since v38
``_v5_contract`` keeps only the default image's roster before flattening
images, so the routed unit is attested by the dense E2M1 pair's routed
projection (``resident|streamed``) instead of the routed E2M1 pair on its own
image (``resident``).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys

import pytest

STOCK_DIGESTS = {
    "format_applicability.json": (
        "d9c224a47508eceafdb39f3f1bbdf1b4c4ee9c37c9b228dff836c063c2743cba"
    ),
    "layer_config.json": (
        "8adac3a9655fbf2b783b98ad04142ee9ddac725ce600ad39bcb7fd312392d6a2"
    ),
    "pareto.csv": (
        "0805fbc4dbffb240025dc5bfb8a4e97ddb241eda8e3c80818f07cdec425cfb32"
    ),
    "pareto.knees.json": (
        "26a0603fe7a943444451b31617140b65294cabdaf377f981bb262c5f7eea5d17"
    ),
    "pareto_seeds/allocator_target_4p6000_achieved_4p5020_c3f5ef1e4ab4.json": (
        "3c9ba881cb0ada42cd400c070d6dc495a29da985f283e79e07ed3dfdd3c19e01"
    ),
    "pareto_seeds/allocator_target_5p5000_achieved_5p2783_81f5decbbbb0.json": (
        "cf93255cb16eb8efe540d5b2fb97257b53e2b3f5ab50434ec970d4b8e77eb2f2"
    ),
    "pareto_seeds/allocator_target_6p0000_achieved_5p7959_6ef0bae1963d.json": (
        "21e0fe09065f2056aead72a4c6c4d3441e00349488dfdfc39e92ea579df3c1d5"
    ),
    "pareto_seeds/allocator_target_8p0000_achieved_7p8662_9e13e69f37a9.json": (
        "53d1d5af6d11318048fe0056e8c54656476146f18a3a6a160d104350a814631b"
    ),
    "pareto_seeds/manifest.json": (
        "6c1d347bab305682a27c3527198dd5cf726d27b3d1a8f1aa6a7545b4cccf0d66"
    ),
}
TESSERA_DIGESTS = {
    "format_applicability.json": (
        "fe348e3503bc245e296cb22f9aeb3750ab96ed5f615eb5b690f46baccb52b7b1"
    ),
    "layer.json": (
        "658e614fe614281454b535be5b44bd631e2bdaf84d187fe15f3d75b928382ad2"
    ),
    "pareto.csv": (
        "0abb6a82a89d9cd9686c6a250368bb77603d3904dc2707ccb69a13c9eec70c5a"
    ),
    "pareto.knees.json": (
        "925991a1a2afcd461ccb3aa026a2135d1487506520f88e78c3bab8c64dedc114"
    ),
}


def _without_wall_time(value):
    # ``solve_diagnostics.*.solver_seconds`` is the DP's wall time; every
    # other byte of the JSON outputs is pinned, key order included.
    if isinstance(value, dict):
        return {
            key: None if key == "solver_seconds" else _without_wall_time(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_without_wall_time(item) for item in value]
    return value


def _digests(root):
    out = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.endswith((".pkl",)) or rel.startswith("model/"):
            continue
        text = path.read_bytes().replace(str(root).encode(), b"<TMP>")
        if rel.endswith(".json"):
            text = json.dumps(_without_wall_time(json.loads(text))).encode()
        out[rel] = hashlib.sha256(text).hexdigest()
    return out


def _stock_inputs(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
    }))
    names = [
        f"model.layers.{layer}.{leaf}"
        for layer in range(2)
        for leaf in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
            "mlp.down_proj",
        )
    ]
    stats, costs = {}, {}
    for idx, name in enumerate(names):
        stats[name] = {
            "h_trace": float(1 + (idx * 7) % 5),
            "n_params": 128 * 256,
            "in_features": 256,
            "out_features": 128,
        }
        costs[name] = {
            "NVFP4": {"predicted_dloss": 1.0 + 0.37 * idx},
            "FP8_DYNAMIC": {"predicted_dloss": 0.05 + 0.011 * idx},
            "BF16": {"predicted_dloss": 0.0},
        }
    (tmp_path / "probe.pkl").write_bytes(
        pickle.dumps({"stats": stats, "meta": {"model": str(model_dir)}}))
    (tmp_path / "cost.pkl").write_bytes(pickle.dumps({
        "costs": costs, "formats": ["NVFP4", "FP8_DYNAMIC", "BF16"]}))
    return [
        "--probe", str(tmp_path / "probe.pkl"),
        "--costs", str(tmp_path / "cost.pkl"),
        "--model-override", str(model_dir),
        "--formats", "NVFP4,FP8_DYNAMIC,BF16",
        "--target-bits", "6.0",
        "--pareto-targets", "4.6,5.5,6.0,8.0",
        "--bit-precision", "0.1",
        "--layer-config", str(tmp_path / "layer_config.json"),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
        "--pareto-output-dir", str(tmp_path / "pareto_seeds"),
    ]


def test_stock_menu_allocation_is_byte_identical(tmp_path, monkeypatch):
    from prismaquant import allocator

    monkeypatch.setattr(sys, "argv", ["allocator", *_stock_inputs(tmp_path)])
    allocator.main()
    got = _digests(tmp_path)
    assert got == STOCK_DIGESTS, json.dumps(got, indent=1, sort_keys=True)


def test_tessera_menu_allocation_is_byte_identical(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from test_tessera_scope_endpoints import (
        _allocator_inputs, _cli_scope, _v5_contract,
    )

    from prismaquant import allocator

    _v5_contract(monkeypatch)
    argv = _allocator_inputs(tmp_path, "TESSERA_E2M1_K2_R896")
    monkeypatch.setattr(sys, "argv", [
        "allocator", *argv, "--no-fused-aggregation",
        "--no-packed-aggregation", *_cli_scope()])
    allocator.main()
    got = _digests(tmp_path)
    assert got == TESSERA_DIGESTS, json.dumps(got, indent=1, sort_keys=True)
