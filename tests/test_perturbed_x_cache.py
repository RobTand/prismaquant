import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from prismaquant import format_registry as fr
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    activation_cache_filename,
    capture_perturbed_activation_cache,
    stage_text_only_under_work_root,
)
from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
from prismaquant.production_weight_cache import ProductionWeightCache


class _TwoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def _load_cache(cache_dir: Path, name: str) -> torch.Tensor:
    blob = torch.load(
        cache_dir / activation_cache_filename(name),
        map_location="cpu",
        weights_only=False,
    )
    return blob["inputs"].to(torch.float32)


def test_perturbed_cache_captures_then_quantizes_for_forward(tmp_path):
    torch.manual_seed(0)
    model = _TwoLinear().eval()
    x = torch.randn(2, 64, dtype=torch.float32)
    fc1_w = model.fc1.weight.detach().clone()

    manifest = capture_perturbed_activation_cache(
        model,
        {"fc1": "NVFP4", "fc2": "BF16"},
        x,
        tmp_path,
        input_rows=8,
    )

    nvfp4 = fr.get_format("NVFP4")
    expected_fc2_input = F.linear(
        nvfp4.activation_quantize_dequantize(x),
        nvfp4.quantize_dequantize(fc1_w),
    )
    torch.testing.assert_close(_load_cache(tmp_path, "fc1"), x, rtol=0.01, atol=0.01)
    torch.testing.assert_close(
        _load_cache(tmp_path, "fc2"),
        expected_fc2_input,
        rtol=0.01,
        atol=0.01,
    )
    torch.testing.assert_close(model.fc1.weight, fc1_w)
    assert manifest["written"] == ["fc1", "fc2"]


class _SiblingInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(64, 64, bias=False)
        self.self_attn.k_proj = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.self_attn.q_proj(x) + self.self_attn.k_proj(x)


class _QwenLiveNameModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.language_model.layers[0]
        layer.mlp = nn.Module()
        layer.mlp.gate_proj = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.model.language_model.layers[0].mlp.gate_proj(x)


def test_perturbed_cache_shares_row_subsample_for_fused_siblings(tmp_path):
    model = _SiblingInputModel().eval()
    x = torch.arange(320, dtype=torch.float32).reshape(1, 5, 64)

    capture_perturbed_activation_cache(
        model,
        {"self_attn.q_proj": "BF16", "self_attn.k_proj": "BF16"},
        x,
        tmp_path,
        input_rows=2,
        cal_hash="fixed",
    )

    q_rows = _load_cache(tmp_path, "self_attn.q_proj")
    k_rows = _load_cache(tmp_path, "self_attn.k_proj")
    assert q_rows.shape == (2, 64)
    torch.testing.assert_close(q_rows, k_rows)


def test_perturbed_cache_uses_profile_live_to_recipe_names(tmp_path):
    model = _QwenLiveNameModel().eval()
    x = torch.randn(2, 64)

    manifest = capture_perturbed_activation_cache(
        model,
        {"model.layers.0.mlp.gate_proj": "BF16"},
        x,
        tmp_path,
        input_rows=4,
        profile=Qwen3_5Profile(),
    )

    assert manifest["missing"] == []
    assert manifest["written"] == ["model.layers.0.mlp.gate_proj"]
    torch.testing.assert_close(
        _load_cache(tmp_path, "model.layers.0.mlp.gate_proj"),
        x,
        rtol=0.01,
        atol=0.01,
    )


def test_perturbed_cache_can_skip_activation_quant_for_probe(tmp_path, monkeypatch):
    spec = fr.FormatSpec(
        name="ZERO_ACT_TEST",
        weight_bits=8,
        group_size=0,
        scale_bits=0,
        scale_dtype_name="none",
        weight_element_dtype="test",
        act_bits=4,
        quantize_dequantize=lambda w: w.clone(),
        activation_quantize_dequantize=lambda x: torch.zeros_like(x),
    )
    monkeypatch.setitem(fr.REGISTRY, spec.name, spec)
    model = nn.Sequential(nn.Linear(64, 64, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(64))
    x = torch.randn(2, 64)

    with_act = PerturbedActivationCache(
        model,
        {"0": spec.name},
        tmp_path / "with_act",
        input_rows=0,
        cal_hash="test",
        include_activation_quant=True,
    )
    with_act.install()
    try:
        torch.testing.assert_close(model(x), torch.zeros_like(x))
    finally:
        with_act.remove()

    without_act = PerturbedActivationCache(
        model,
        {"0": spec.name},
        tmp_path / "without_act",
        input_rows=0,
        cal_hash="test",
        include_activation_quant=False,
    )
    without_act.install()
    try:
        torch.testing.assert_close(model(x), x)
    finally:
        without_act.remove()


def test_perturbed_cache_production_cache_miss_is_strict_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_STRICT_PRODUCTION_CACHE", raising=False)
    model = nn.Sequential(nn.Linear(64, 64, bias=False)).eval()
    cache = PerturbedActivationCache(
        model,
        {"0": "NVFP4"},
        tmp_path,
        input_rows=0,
        cal_hash="test",
        production_weight_cache=ProductionWeightCache({}, levers={}),
    )

    cache.install()
    try:
        with pytest.raises(RuntimeError, match="production_weight_cache miss"):
            model(torch.randn(1, 64))
    finally:
        cache.remove()


def test_perturbed_cache_strict_miss_escape_allows_rtn(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_STRICT_PRODUCTION_CACHE", "0")
    model = nn.Sequential(nn.Linear(64, 64, bias=False)).eval()
    cache = PerturbedActivationCache(
        model,
        {"0": "NVFP4"},
        tmp_path,
        input_rows=0,
        cal_hash="test",
        production_weight_cache=ProductionWeightCache({}, levers={}),
    )

    cache.install()
    try:
        out = model(torch.randn(1, 64))
    finally:
        cache.remove()

    assert out.shape == (1, 64)


def test_perturbed_cache_refuses_a_retired_codebook_rung(
    tmp_path, monkeypatch,
):
    # The retired codebook lane, archived 2026-09-25, #1304: the refusal is
    # raised wherever the cache first resolves the name, never skipped.
    monkeypatch.setenv("PRISMAQUANT_STRICT_PRODUCTION_CACHE", "0")
    model = nn.Sequential(nn.Linear(256, 256, bias=False)).eval()
    cache = None
    try:
        with pytest.raises(fr.RetiredFormatError, match="gridbook_lane"):
            cache = PerturbedActivationCache(
                model,
                {"0": "NVFP4_CB_K16"},
                tmp_path,
                input_rows=0,
                cal_hash="test",
                production_weight_cache=ProductionWeightCache({}, levers={}),
            )
            cache.install()
            model(torch.randn(1, 256))
    finally:
        if cache is not None:
            cache.remove()


def test_perturbed_cache_can_disable_capture_for_inplace_replay(tmp_path):
    model = nn.Sequential(nn.Linear(64, 64, bias=False)).eval()
    x = torch.randn(2, 64)
    cache = PerturbedActivationCache(
        model,
        {"0": "BF16"},
        tmp_path,
        input_rows=8,
        cal_hash="test",
        capture_inputs=False,
    )

    cache.install()
    try:
        _ = model(x)
    finally:
        cache.remove()

    assert cache.max_abs == {}
    assert cache.finalize()["written"] == []


class _OneLinear(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.fc = nn.Linear(width, width, bias=False)

    def forward(self, x):
        return self.fc(x)


def _batches_with_markers(n_batches, rows_per_batch, width):
    """Calibration batches whose rows carry (batch_idx, row_idx) markers
    in features 0/1 so a loaded cache row can be attributed."""
    batches = []
    for b in range(n_batches):
        t = torch.randn(rows_per_batch, width)
        t[:, 0] = float(b)
        t[:, 1] = torch.arange(rows_per_batch, dtype=torch.float32)
        batches.append(t)
    return batches


def test_perturbed_cache_samples_uniformly_across_batches(tmp_path):
    """M8 regression: the capture must NOT keep the first input_rows rows
    of the stream ({limit, 0, 0, 0} per batch); a seeded priority
    reservoir keeps ~limit/n_batches rows from every batch."""
    torch.manual_seed(0)
    model = _OneLinear().eval()
    n_batches, rows_per_batch, limit = 4, 256, 128
    batches = _batches_with_markers(n_batches, rows_per_batch, 64)

    capture_perturbed_activation_cache(
        model,
        {"fc": "BF16"},
        batches,
        tmp_path,
        input_rows=limit,
        cal_hash="fixed-uniformity",
    )

    rows = _load_cache(tmp_path, "fc")
    assert rows.shape[0] == limit
    counts = [int((rows[:, 0] == float(b)).sum()) for b in range(n_batches)]
    assert sum(counts) == limit
    # Expected 32/batch; binomial sd ~4.9. ±5 sigma bounds — deterministic
    # under the fixed cal_hash, and {128, 0, 0, 0} fails loudly.
    for b, c in enumerate(counts):
        assert 8 <= c <= 56, f"batch {b} kept {c} rows: {counts}"


def test_perturbed_cache_reservoir_is_deterministic_under_cal_hash(tmp_path):
    model = _OneLinear().eval()
    batches = _batches_with_markers(4, 64, 64)
    for run in ("a", "b"):
        capture_perturbed_activation_cache(
            model,
            {"fc": "BF16"},
            batches,
            tmp_path / run,
            input_rows=32,
            cal_hash="fixed-repro",
        )
    torch.testing.assert_close(
        _load_cache(tmp_path / "a", "fc"),
        _load_cache(tmp_path / "b", "fc"),
    )


def test_perturbed_cache_siblings_share_rows_across_whole_stream(tmp_path):
    """Fused siblings must keep IDENTICAL row sets even when the reservoir
    replaces rows across many batches (not just within one call)."""
    model = _SiblingInputModel().eval()
    batches = _batches_with_markers(4, 64, 64)

    capture_perturbed_activation_cache(
        model,
        {"self_attn.q_proj": "BF16", "self_attn.k_proj": "BF16"},
        batches,
        tmp_path,
        input_rows=32,
        cal_hash="fixed-siblings",
    )

    q_rows = _load_cache(tmp_path, "self_attn.q_proj")
    k_rows = _load_cache(tmp_path, "self_attn.k_proj")
    assert q_rows.shape == (32, 64)
    torch.testing.assert_close(q_rows, k_rows)
    # And the shared sample must actually span multiple batches.
    assert len(set(q_rows[:, 0].tolist())) >= 2


def test_perturbed_cache_input_rows_zero_still_tracks_max_abs(tmp_path):
    model = _OneLinear().eval()
    cache = PerturbedActivationCache(
        model,
        {"fc": "BF16"},
        tmp_path,
        input_rows=0,
        cal_hash="test",
    )
    cache.install()
    try:
        _ = model(torch.full((2, 64), 3.0))
    finally:
        cache.remove()

    assert cache.max_abs["fc"] == pytest.approx(3.0)
    assert cache.finalize()["written"] == []


def test_stage_text_only_uses_work_root_for_tempdir(tmp_path):
    src = tmp_path / "model"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"placeholder")
    with open(src / "config.json", "w") as f:
        json.dump(
            {
                "vision_config": {},
                "text_config": {"hidden_size": 8, "model_type": "toy_text"},
                "architectures": ["ToyForConditionalGeneration"],
            },
            f,
        )
    work_root = tmp_path / "work"

    staged = Path(stage_text_only_under_work_root(str(src), work_root))

    assert staged.parent == work_root
    with open(staged / "config.json") as f:
        cfg = json.load(f)
    assert "vision_config" not in cfg
    assert cfg["hidden_size"] == 8
    assert cfg["architectures"] == ["ToyForCausalLM"]
