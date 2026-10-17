"""A successful dense cache resume must still satisfy the exact union contract."""
from __future__ import annotations

import json
import pickle

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prismaquant import build_production_cache as builder
from prismaquant import union_production_cache as union
from prismaquant.model_profiles import DefaultProfile
from test_production_weight_cache import _TinyChain
from test_union_production_cache import CODE_IDENTITY, _source_identity


@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resumed"])
def test_dense_producer_preserves_gate_coverage_for_exact_union(
    tmp_path, monkeypatch, resume,
):
    model = _TinyChain().eval()
    profile = DefaultProfile()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calibration = torch.tensor([[0, 1]], dtype=torch.long)
    assignment = {"l1": "NVFP4", "l2": "NVFP4"}
    assignment_path = tmp_path / "layer_config.json"
    assignment_path.write_text(json.dumps(assignment))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(builder, "detect_profile_with_warning", lambda *args, **kwargs: profile)
    monkeypatch.setattr(builder, "require_cuda_hot_path", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(builder, "_load_cache_calibration", lambda *args: calibration)
    manifests = []

    for index, selected in enumerate(assignment):
        bundle = tmp_path / f"stripe-{index}"
        bundle.mkdir()
        weights_dir = bundle / "weights"
        include_path = bundle / "include.txt"
        include_path.write_text(selected + "\n")
        cache_path = bundle / "cache.pkl"
        argv = [
            "--model", str(model_dir), "--output", str(cache_path),
            "--cache-dir", str(weights_dir),
            "--render-scope", "assignment",
            "--render-layer-config", str(assignment_path),
            "--include-qnames-file", str(include_path),
            "--formats", "NVFP4", "--enable", "gptq", "--dtype", "fp32",
            "--max-act-rows", "8",
        ]
        assert builder.main(argv) == 0
        if resume:
            assert builder.main(argv) == 0
        cache = pickle.loads(cache_path.read_bytes())
        assert set(cache.weights) == {(selected, "NVFP4")}
        assert cache.metadata["render_scores"]["entries"] == 1
        if resume:
            gates = cache.metadata["render_gates"]
            assert gates["entries"] == 1
            assert gates["mechanisms"] == {}
            assert gates["records"] == [{
                "qname": selected,
                "format": "NVFP4",
                "render_format": "NVFP4",
                "trace": [],
                "progressive_gate": "not-run",
                "mechanism": "resumed_shard",
            }]
        manifest = bundle / "shard_manifest.json"
        union.create_shard_manifest(
            cache_path=cache_path,
            cache_dir=weights_dir,
            manifest_path=manifest,
            shard_id=f"stripe-{index}",
            source_model_identity=_source_identity(),
            settings={"schema": "test.dense_resume_union.v1", "max_act_rows": 8},
            assignment=assignment,
            code_identity=CODE_IDENTITY,
        )
        manifests.append(manifest)

    output = tmp_path / "union"
    union.union_shard_manifests(
        manifests,
        assignment=assignment,
        output_bundle=output,
        require_current_code=False,
    )
    _payload, merged = union.verify_union_manifest(
        output / "union_manifest.json",
        assignment=assignment,
        require_current_code=False,
    )
    assert set(merged.weights) == {(name, "NVFP4") for name in assignment}
    assert merged.metadata["render_gates"]["entries"] == len(assignment)
