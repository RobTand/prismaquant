"""Retained decision scores must describe the canonical stored candidate."""
from __future__ import annotations

import json

import pytest
import torch

from prismaquant import production_render_cost as cost
from prismaquant import production_weight_cache as pwc
from prismaquant.native_render_levers import NativeRenderLevers
from test_production_weight_cache import _TinyChain


DTYPES = [torch.bfloat16, torch.float16, torch.float32]
SCORE_FIELDS = (
    "score", "score_sum", "raw_render_score", "raw_render_score_sum",
    "weight_mse", "weight_mse_sum", "normalizer", "activation_rows",
    "activation_quantized", "activation_clipped",
)
IDS = torch.tensor([[0, 1]], dtype=torch.long)


def _cost_baseline():
    return {
        "formats": ["FP8_E4M3"],
        "costs": {"l1": {"FP8_E4M3": {"weight_mse": 99.0}}},
    }


def _model(dtype):
    model = _TinyChain().eval()
    with torch.no_grad():
        values = (torch.arange(32 * 32).reshape(32, 32) % 37 - 18).float() / 37
        model.l1.weight.copy_(values)
    return model.to(dtype=dtype)


def _fill(model, directory):
    return pwc.fill_production_weight_cache(
        model, IDS, ["l1", "l2"], render_qnames=["l1"], formats=["FP8_E4M3"],
        levers={"gptq": False, "scale_sweep": False}, cache_dir=directory,
        max_act_rows=8, progress=False,
    )


def _score(model, rendered, levers):
    return pwc._render_score_record(
        qname="l1", fmt="FP8_E4M3", render_format="FP8_E4M3",
        reference_weight=model.l1.weight.detach(), rendered_weight=rendered,
        activations=model.embed(IDS).detach().float().reshape(-1, 32),
        activation_max_abs=None, levers=NativeRenderLevers.from_record(levers),
    )


def _assert_scores(cache, model, stored):
    expected = _score(model, stored, cache.levers)
    record = cache.metadata["render_scores"]["records"]["l1|FP8_E4M3"]
    assert record["scoring_contract"] == pwc.STORED_CANDIDATE_RENDER_SCORE_CONTRACT
    assert record["rendered_weight_dtype"] == str(stored.dtype)
    for field in SCORE_FIELDS:
        assert record[field] == expected[field], field
    for field, key in (("weight_mse", "weight_mse"), ("output_mse", "output_mse")):
        payload = cost.synthesize_production_render_cost_payload(cache, _cost_baseline(), score_field=field)
        assert payload["costs"]["l1"]["FP8_E4M3"][key] == expected[
            "raw_render_score" if field == "output_mse" else field]


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resume"])
def test_dense_retained_weight_cost_prices_stored_candidate(tmp_path, dtype, resume):
    model = _model(dtype)
    cache = _fill(model, tmp_path / "weights")
    if resume:
        cache = _fill(model, tmp_path / "weights")
    assert not cache.failed
    stored = cache.get("l1", "FP8_E4M3")
    assert stored.dtype == (torch.bfloat16 if dtype == torch.float32 else dtype)
    _assert_scores(cache, model, stored)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp16", "fp32"])
def test_legacy_dense_scores_are_completely_rescored_without_rerender(tmp_path, monkeypatch, dtype):
    model = _model(dtype)
    directory = tmp_path / "weights"
    cache = _fill(model, directory)
    stored = cache.get("l1", "FP8_E4M3")
    rows = model.embed(IDS).detach().float().reshape(-1, 32)
    wider = pwc.render_production_weight(
        model.l1.weight.detach(), "FP8_E4M3", qname="l1", activations={"l1": rows},
        levers=cache.levers,
    )
    legacy = _score(model, wider, cache.levers)
    assert "scoring_contract" not in legacy
    if dtype == torch.float32:
        assert legacy["weight_mse"] != _score(model, stored, cache.levers)["weight_mse"]
    sidecar = directory / "render_scores.json"
    payload = json.loads(sidecar.read_text())
    payload["records"]["l1|FP8_E4M3"] = legacy
    sidecar.write_text(json.dumps(payload))
    shard = directory / pwc.cache_weight_filename("l1", "FP8_E4M3")
    before = shard.read_bytes()
    captured = []
    real_collected = pwc._LinearActivationCollector.collected

    def collect(collector):
        result = real_collected(collector)
        captured.append(set(result))
        return result

    def refuse_render(*args, **kwargs):
        pytest.fail("score migration re-rendered a stored candidate")

    monkeypatch.setattr(pwc._LinearActivationCollector, "collected", collect)
    monkeypatch.setattr(pwc, "render_production_weight", refuse_render)
    repaired = _fill(model, directory)
    assert captured == [{"l1"}]
    assert shard.read_bytes() == before
    _assert_scores(repaired, model, repaired.get("l1", "FP8_E4M3"))
    assert repaired.metadata["render_gates"]["records"][0]["mechanism"] == "resumed_shard"
    _fill(model, directory)
    assert captured == [{"l1"}], "qualified full resume captured rows again"


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("retain", [True, False], ids=["retained", "transient"])
def test_streamed_dense_scores_price_the_consumed_candidate(tmp_path, dtype, retain):
    from prismaquant.measure_quant_cost import ActivationIndex
    from prismaquant.streaming_production_cache import _render_dense_layer

    model = _model(dtype)
    rows = model.embed(IDS).detach().float().reshape(-1, 32)
    torch.save({"inputs": rows, "name": "l1"}, tmp_path / "l1.pt")
    index = ActivationIndex(tmp_path, ["l1"])
    levers = pwc._resolve_production_render_levers({"gptq": False, "scale_sweep": False})
    cache = pwc.ProductionWeightCache(weights={}, levers=levers, activation_max_abs={})
    records = {}
    consumed = []

    def consume(**value):
        consumed.append(value["rendered_weight"])
        return {}

    assert _render_dense_layer(
        model, {"l1": model.l1}, render_formats_by_qname={"l1": ["FP8_E4M3"]},
        act_index=index, cache=cache, levers=levers, cache_dir_path=None,
        profile=None, device=torch.device("cpu"), fisher_rows=None,
        render_score_records=records, col_weights=None, retain_rendered=retain,
        consume_render=None if retain else consume, max_act_rows=8, progress=False,
    ) == 1
    stored = cache.get("l1", "FP8_E4M3") if retain else consumed[0]
    cache.metadata = {"render_scores": {"records": records}}
    assert stored.dtype == (torch.bfloat16 if dtype == torch.float32 else dtype)
    _assert_scores(cache, model, stored)


@pytest.mark.parametrize("fields", [
    {}, {"rendered_weight_dtype": "torch.bfloat16"},
    {"scoring_contract": pwc.STORED_CANDIDATE_RENDER_SCORE_CONTRACT},
    {"scoring_contract": "older", "rendered_weight_dtype": "torch.bfloat16"},
    {"scoring_contract": pwc.STORED_CANDIDATE_RENDER_SCORE_CONTRACT, "rendered_weight_dtype": "torch.int8"},
    {"scoring_contract": pwc.STORED_CANDIDATE_RENDER_SCORE_CONTRACT, "rendered_weight_dtype": []},
])
def test_present_unqualified_selected_cost_refuses_even_with_baseline_fallback(fields):
    record = {"qname": "l1", "format": "FP8_E4M3", "metric": "output_mse", "score": 1.0, "score_sum": 2.0, **fields}
    cache = pwc.ProductionWeightCache(weights={}, levers={}, metadata={"render_scores": {"records": {"l1|FP8_E4M3": record}}})
    with pytest.raises(ValueError, match="l1@FP8_E4M3.*resume.*rescore.*rebuild"):
        cost.synthesize_production_render_cost_payload(cache, _cost_baseline())


@pytest.mark.parametrize("record", [None, 17])
def test_malformed_present_selected_cost_cannot_become_missing_fallback(record):
    cache = pwc.ProductionWeightCache(
        weights={}, levers={},
        metadata={"render_scores": {"records": {"l1|FP8_E4M3": record}}},
    )
    with pytest.raises(ValueError, match="l1@FP8_E4M3.*resume.*rescore.*rebuild"):
        cost.synthesize_production_render_cost_payload(cache, _cost_baseline())


def test_unselected_unqualified_record_does_not_replace_missing_score_fallback():
    cache = pwc.ProductionWeightCache(weights={}, levers={}, metadata={"render_scores": {"records": {"outside|FP8_E4M3": {"qname": "outside", "format": "FP8_E4M3", "score": 1.0}}}})
    payload = cost.synthesize_production_render_cost_payload(cache, _cost_baseline())
    assert payload["costs"]["l1"]["FP8_E4M3"]["weight_mse"] == 99.0
    with pytest.raises(ValueError, match="missing 1 production render scores"):
        cost.synthesize_production_render_cost_payload(cache, _cost_baseline(), require_render_scores=True)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp16", "fp32"])
def test_mtp_shared_renderer_prices_the_stored_candidate(tmp_path, monkeypatch, dtype):
    from prismaquant.mtp_production_cache import fill_profile_mtp_production_cache
    from test_mtp_production_cache import _TinyMtpProfile, _patch_auto_config, _write_activation

    _patch_auto_config(monkeypatch)
    profile = _TinyMtpProfile()
    rows = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 7
    activation_dir = tmp_path / "activations"
    _write_activation(activation_dir, "mtp.proj", rows)
    levers = pwc._resolve_production_render_levers({"gptq": False, "scale_sweep": False})
    cache = pwc.ProductionWeightCache(weights={}, levers=levers)
    assert fill_profile_mtp_production_cache(
        cache, "/fake/model", profile=profile, activation_cache_dir=activation_dir,
        render_assignment={"mtp.proj": "FP8_E4M3"}, device="cpu", dtype=dtype,
        progress=False,
    ) == 1
    stored = cache.get("mtp.proj", "FP8_E4M3")
    expected = pwc._render_score_record(
        qname="mtp.proj", fmt="FP8_E4M3", render_format="FP8_E4M3",
        reference_weight=profile.source["proj.weight"].to(dtype), rendered_weight=stored,
        activations=rows, activation_max_abs=None,
        levers=NativeRenderLevers.from_record(cache.levers),
    )
    record = cache.metadata["render_scores"]["records"]["mtp.proj|FP8_E4M3"]
    assert record["scoring_contract"] == pwc.STORED_CANDIDATE_RENDER_SCORE_CONTRACT
    assert record["rendered_weight_dtype"] == str(stored.dtype)
    assert stored.dtype == (torch.bfloat16 if dtype == torch.float32 else dtype)
    for field in SCORE_FIELDS:
        assert record[field] == expected[field], field


@pytest.mark.parametrize("axis", ["static_scale", "served_quantizer"])
def test_unqualified_score_repair_preserves_existing_provenance_refusals(tmp_path, monkeypatch, axis):
    from prismaquant import nvfp4_activation_contract as owner

    model = _model(torch.float32)
    directory = tmp_path / "weights"
    cache = _fill(model, directory)
    legacy = _score(model, cache.get("l1", "FP8_E4M3"), cache.levers)
    identity = owner.ServedQuantizerIdentity(backend=owner.SERVED_QUANTIZER_BACKEND_MODEL)
    owner.bind_served_quantizer_identity(identity=identity, require=False)
    policy = pwc._render_levers_input_global_scale_policy(cache.levers)
    legacy.update({
        "activation_max_abs": 2.0,
        "input_global_scale": owner.input_global_scale_from_max_abs(2.0, policy=policy),
        "input_global_scale_policy": policy,
        "served_quantizer": identity.as_record(),
    })
    if axis == "static_scale":
        legacy["input_global_scale"] *= 2
        error = owner.ActivationScalePolicyMismatchError
    else:
        legacy["served_quantizer"] = {**identity.as_record(), "torch": "another-build"}
        error = owner.ServedQuantizerUnboundError
    sidecar = directory / "render_scores.json"
    payload = json.loads(sidecar.read_text())
    payload["records"]["l1|FP8_E4M3"] = legacy
    sidecar.write_text(json.dumps(payload))

    def refuse_capture(*args, **kwargs):
        pytest.fail("unqualified score repair erased an existing provenance refusal")

    monkeypatch.setattr(pwc._LinearActivationCollector, "collected", refuse_capture)
    with pytest.raises(error):
        _fill(model, directory)


def test_failed_legacy_score_repair_cannot_erase_the_record_into_fallback(tmp_path, monkeypatch):
    model = _model(torch.float32)
    directory = tmp_path / "weights"
    cache = _fill(model, directory)
    sidecar = directory / "render_scores.json"
    payload = json.loads(sidecar.read_text())
    payload["records"]["l1|FP8_E4M3"] = _score(model, cache.get("l1", "FP8_E4M3"), cache.levers)
    sidecar.write_text(json.dumps(payload))
    before = sidecar.read_bytes()

    def refuse_score(**kwargs):
        raise RuntimeError("intentional rescore failure")

    monkeypatch.setattr(pwc, "_render_score_record", refuse_score)
    with pytest.raises(RuntimeError, match="cannot rescore stored candidate l1@FP8_E4M3"):
        _fill(model, directory)
    assert sidecar.read_bytes() == before
