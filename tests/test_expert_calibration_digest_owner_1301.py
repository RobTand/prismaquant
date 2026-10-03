"""Calibration identities retain exact acquired bytes and native refusals.

These tests call the existing checkpoint and CLI consumers. Model/GPU loading
and empirical measurement are isolated; the CLI's provenance writer is real.
Explicit packed-byte oracles distinguish both legacy tensor extraction rules.
"""
from __future__ import annotations

import hashlib
import pickle
import struct
from types import SimpleNamespace

import pytest
import torch

from prismaquant import expert_empirical_cost as ec
from prismaquant.digests import bytes_sha256hex, canonical_json_sha256


def _input(case):
    base = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    if case == "contiguous":
        return base, struct.pack("<12q", *range(12))
    if case == "transpose":
        return base.T, struct.pack("<12q", 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11)
    if case == "slice":
        return base[:, ::2], struct.pack("<6q", 0, 2, 4, 6, 8, 10)
    if case == "int32":
        return base.to(torch.int32), struct.pack("<12i", *range(12))
    if case == "empty":
        return base[:0], b""
    raise AssertionError(case)


def _identity(monkeypatch, calib):
    from prismaquant import production_weight_cache as pwc
    from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA

    monkeypatch.setattr(ec, "_git_commit", lambda: "test-commit")
    monkeypatch.setattr(pwc, "_production_cache_source_sha256", lambda: "a" * 64)
    source = {"config": {}, "weight_map": {}, "shards": [{"sha256": "b" * 64}]}
    model = {"schema": STREAMED_MODEL_IDENTITY_SCHEMA, **source,
             "content_sha256": canonical_json_sha256(source, where="test source")}
    return ec._expert_checkpoint_identity(
        runner=SimpleNamespace(dtype=torch.bfloat16), profile=None, calib_ids=calib,
        formats=["BF16"], col_weights={}, unit_identities=[], model_identity=model,
        expert_chunk=1, expert_sample=0, max_units=0, unit_filter=None, identity_extra=None,
    )


def _cli(monkeypatch, tmp_path, calib):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prismaquant import build_rtn_cache, calibration_data, gpu_guard, layer_streaming, model_profiles

    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *args: None)
    monkeypatch.setattr(build_rtn_cache, "stage_multimodal", lambda path: (path, None))
    monkeypatch.setattr(model_profiles, "detect_profile", lambda path: None)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: torch.nn.Linear(1, 1))
    monkeypatch.setattr(layer_streaming, "fill_packed_experts_from_source", lambda *a, **k: 0)
    monkeypatch.setattr(calibration_data, "load_wikitext_calibration_windowed", lambda *a, **k: calib)
    monkeypatch.setattr(ec, "measure_expert_unit_costs", lambda *a, **k: ({}, {}, {}))
    monkeypatch.setattr(ec, "_git_commit", lambda: "test-commit")
    out = tmp_path / "cost.pkl"
    assert ec.main(["--model", str(tmp_path), "--output", str(out),
                    "--device", "cpu", "--formats", "BF16"]) == 0
    return pickle.loads(out.read_bytes())


@pytest.mark.parametrize("site", ["checkpoint", "cli"])
@pytest.mark.parametrize("case", ["contiguous", "transpose", "slice", "int32", "empty"])
def test_legacy_consumers_keep_exact_calibration_hash(monkeypatch, tmp_path, site, case):
    calib, expected_bytes = _input(case)
    expected_sha = hashlib.sha256(expected_bytes).hexdigest()
    if site == "checkpoint":
        actual = _identity(monkeypatch, calib)["calibration"]
        assert actual["shape"] == list(calib.shape)
        assert actual["dtype"] == str(calib.dtype)
        assert actual["sha256"] == expected_sha
    else:
        actual = _cli(monkeypatch, tmp_path, calib)["provenance"]
        assert actual["n_calib_samples"] == calib.shape[0]
        assert actual["calib_seqlen"] == calib.shape[1]
        assert actual["calib_sha256"] == expected_sha


@pytest.mark.parametrize("site", ["checkpoint", "cli"])
def test_each_final_raw_hash_routes_its_already_extracted_bytes(monkeypatch, tmp_path, site):
    calib, expected_bytes = _input("transpose")
    acquired = []

    def record(raw):
        acquired.append(raw)
        return bytes_sha256hex(raw)

    monkeypatch.setattr(ec, "bytes_sha256hex", record, raising=False)
    if site == "checkpoint":
        _identity(monkeypatch, calib)
    else:
        _cli(monkeypatch, tmp_path, calib)
    assert acquired == [expected_bytes]


def test_cli_keeps_numpy_grad_and_bfloat16_refusals(monkeypatch, tmp_path):
    for calib in (torch.ones(2, 3, requires_grad=True), torch.ones(2, 3, dtype=torch.bfloat16)):
        with pytest.raises((RuntimeError, TypeError)) as old:
            calib.cpu().numpy().tobytes()
        with pytest.raises(type(old.value)) as actual:
            _cli(monkeypatch, tmp_path, calib)
        assert str(actual.value) == str(old.value)
        assert not (tmp_path / "cost.pkl").exists()


def test_checkpoint_keeps_wide_scalar_view_refusal(monkeypatch):
    calib = torch.tensor(3, dtype=torch.int64)
    with pytest.raises(RuntimeError) as old:
        calib.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    with pytest.raises(RuntimeError) as actual:
        _identity(monkeypatch, calib)
    assert str(actual.value) == str(old.value)
