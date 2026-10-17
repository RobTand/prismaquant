"""Packed retained costs must price the candidate actually stored."""
from __future__ import annotations

import json
import pickle

import pytest
import torch

from prismaquant import production_weight_cache as pwc
from test_dense_score_storage_identity_audit import SCORE_FIELDS
from test_packed_expert_cross_domain_gate import ASSIGNMENT, TinyLM


def _setup(dtype, directory):
    torch.manual_seed(13)
    model = TinyLM().eval().to(dtype)
    calib = torch.randint(0, 32, (2, 64))
    directory.mkdir()
    cache = pwc.ProductionWeightCache(weights={}, levers={"gptq": True}, cache_dir=str(directory))
    return model, calib, cache


def _fill(cache, model, calib, mode):
    return pwc.fill_packed_expert_cache_entries(
        cache, model, calib, render_assignment=ASSIGNMENT,
        levers={"gptq": True}, profile=None, module_token_budget=4096,
        eval_rows_per_expert=8, render_mode=mode, cache_dir=cache.cache_dir,
        progress=False,
    )


def _observe_scores(monkeypatch):
    real_score = pwc._packed_expert_render_score_record
    calls = {}

    def score(**values):
        calls[values["qname"]] = dict(values)
        return real_score(**values)

    monkeypatch.setattr(pwc, "_packed_expert_render_score_record", score)
    return real_score, calls


def _assert_scores(cache, real_score, calls):
    for qname in ASSIGNMENT:
        stored = cache.get(qname, "NVFP4")
        arguments = {**calls[qname], "rendered": stored, "stored_dtype": None}
        expected = real_score(**arguments)
        record = cache.metadata["render_scores"]["records"][qname + "|NVFP4"]
        assert record["scoring_contract"] == pwc.STORED_CANDIDATE_RENDER_SCORE_CONTRACT
        assert record["rendered_weight_dtype"] == str(stored.dtype)
        assert record["score_rows_source"] == "fit_corpus_tail"
        for field in SCORE_FIELDS:
            assert record[field] == expected[field], (qname, field)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("mode", ["batched", "per_expert"])
@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resume"])
def test_packed_retained_weight_cost_prices_stored_candidate(tmp_path, monkeypatch, dtype, mode, resume):
    model, calib, cache = _setup(dtype, tmp_path / "weights")
    real_score, calls = _observe_scores(monkeypatch)
    assert set(_fill(cache, model, calib, mode)) == set(ASSIGNMENT)
    if resume:
        _fill(cache, model, calib, mode)
    assert not cache.failed
    for qname in ASSIGNMENT:
        stored = cache.get(qname, "NVFP4")
        assert stored.dtype == (torch.bfloat16 if dtype == torch.float32 else dtype)
        reference = getattr(model.mlp.experts, qname.rsplit(".", 1)[1]).detach().float()
        expected = sum(
            float((reference[e] - stored[e].float()).pow(2).mean().item()) * reference[e].numel()
            for e in range(reference.shape[0])
        ) / reference.numel()
        assert cache.metadata["render_scores"]["records"][qname + "|NVFP4"]["weight_mse"] == expected
    _assert_scores(cache, real_score, calls)


@pytest.mark.parametrize("mode", ["batched", "per_expert"])
def test_legacy_packed_scores_are_completely_rescored_without_rerender(tmp_path, monkeypatch, mode):
    from prismaquant import export_batched_gptq

    model, calib, cache = _setup(torch.bfloat16, tmp_path / "weights")
    real_score, calls = _observe_scores(monkeypatch)
    _fill(cache, model, calib, mode)
    records = cache.metadata["render_scores"]["records"]
    differences = []
    for qname in ASSIGNMENT:
        legacy = real_score(**{**calls[qname], "stored_dtype": None})
        assert "scoring_contract" not in legacy
        if mode == "batched":
            stored_score = real_score(**{**calls[qname], "rendered": cache.get(qname, "NVFP4"), "stored_dtype": None})
            differences.append(legacy["weight_mse"] != stored_score["weight_mse"])
        records[qname + "|NVFP4"] = legacy
    if mode == "batched":
        assert any(differences), "fixture did not retain a wider-candidate cost"
    directory = tmp_path / "weights"
    (directory / "render_scores.json").write_text(json.dumps({"records": records}))
    cache = pickle.loads(pickle.dumps(cache))
    before = {q: (directory / pwc.cache_weight_filename(q, "NVFP4")).read_bytes() for q in ASSIGNMENT}
    captures = []
    real_collected = pwc._PackedExpertActivationCollector.collected

    def collect(collector):
        result = real_collected(collector)
        captures.append(set(result))
        return result

    def refuse_render(*args, **kwargs):
        pytest.fail("packed score migration re-rendered a stored candidate")

    monkeypatch.setattr(pwc._PackedExpertActivationCollector, "collected", collect)
    monkeypatch.setattr(pwc, "render_production_weight", refuse_render)
    monkeypatch.setattr(export_batched_gptq, "gptq_obs_rounding_nvfp4_batched", refuse_render)
    calls.clear()
    _fill(cache, model, calib, mode)
    assert captures == [{"mlp.experts"}]
    assert set(calls) == set(ASSIGNMENT)
    assert {q: (directory / pwc.cache_weight_filename(q, "NVFP4")).read_bytes() for q in ASSIGNMENT} == before
    _assert_scores(cache, real_score, calls)
    _fill(cache, model, calib, mode)
    assert captures == [{"mlp.experts"}], "qualified packed resume captured rows again"
