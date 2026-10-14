from prismaquant.calibration_data import load_calibration_repeats
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.validate_assignments_kl import (
    _assert_calibration_disjoint,
    _assignment_cost_summary,
    _calibration_provenance,
    _materialize_assignment_inplace,
    _production_cache_assignment_diagnostics,
    _upstream_calibration_hashes,
)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(32, 32, bias=False)

    def forward(self, x):
        return self.l(x)


def test_materialize_assignment_inplace_uses_production_cache_tensor():
    model = _TinyModel()
    with torch.no_grad():
        model.l.weight.zero_()
    rendered = torch.full_like(model.l.weight, 3.0)
    cache = ProductionWeightCache(
        weights={("l.weight", "NVFP4"): rendered.clone()},
        levers={},
    )

    stats = _materialize_assignment_inplace(
        model,
        {"l.weight": "NVFP4", "other": "BF16"},
        cache,
    )

    torch.testing.assert_close(model.l.weight, rendered)
    assert stats["copied"] == 1
    assert stats["format_counts"] == {"NVFP4": 1}


def test_assignment_cost_summary_reports_local_mse_and_aliases():
    costs = {
        "a": {
            "NVFP4": {
                "weight_mse": 1.0,
                "output_mse": 2.0,
                "fisher_output_mse": 1.5,
                "rel_output_mse": 0.2,
                "predicted_dloss": 3.0,
            },
        },
        "b": {
            "MXFP8_E4M3": {
                "weight_mse": 0.5,
                "output_mse": 0.25,
                "fisher_output_mse": 0.10,
                "rel_output_mse": 0.025,
            },
        },
    }
    summary = _assignment_cost_summary(
        costs,
        {"a": "NVFP4", "b": "MXFP8", "c": "BF16", "missing": "NVFP4"},
    )

    assert abs(summary["output_mse_sum"] - 2.25) < 1e-12
    assert abs(summary["fisher_output_mse_sum"] - 1.60) < 1e-12
    assert abs(summary["weight_mse_sum"] - 1.5) < 1e-12
    assert abs(summary["rel_output_mse_sum"] - 0.225) < 1e-12
    assert abs(summary["predicted_dloss_sum"] - 3.0) < 1e-12
    assert summary["counts"]["output_mse"] == 3
    assert summary["counts"]["fisher_output_mse"] == 2
    assert summary["counts"]["predicted_dloss"] == 1
    assert summary["missing_count"] == 1
    assert summary["missing_sample"] == ["missing"]


def test_calibration_provenance_hashes_repeats():
    repeat_a = torch.tensor([[1, 2, 3]], dtype=torch.long)
    repeat_b = torch.tensor([[4, 5, 6]], dtype=torch.long)

    single = _calibration_provenance([repeat_a])
    combined = _calibration_provenance([repeat_a, repeat_b])

    assert single["calib_hash"] == single["calib_repeat_hashes"][0]
    assert len(combined["calib_repeat_hashes"]) == 2
    assert combined["calib_hash"] != single["calib_hash"]


def test_production_cache_assignment_diagnostics_counts_misses(monkeypatch):
    cache = ProductionWeightCache(
        weights={("l.weight", "NVFP4"): torch.zeros(1, 1)},
        levers={},
    )
    assignment = {
        "l.weight": "NVFP4",
        "missing.weight": "NVFP4",
        "source.weight": "BF16",
    }

    strict = _production_cache_assignment_diagnostics(cache, assignment)
    assert strict["required_entries"] == 2
    assert strict["cache_hit_count"] == 1
    assert strict["cache_miss_count"] == 1
    assert strict["rtn_fallback_count"] == 0
    assert strict["strict"] is True

    monkeypatch.setenv("PRISMAQUANT_STRICT_PRODUCTION_CACHE", "0")
    permissive = _production_cache_assignment_diagnostics(cache, assignment)
    assert permissive["cache_hit_count"] == 1
    assert permissive["cache_miss_count"] == 1
    assert permissive["rtn_fallback_count"] == 1
    assert permissive["strict"] is False


def test_diagnostics_counts_packed_expert_misses_m4():
    # M4: the diagnostics must ask assignment_keys to INCLUDE packed experts,
    # otherwise packed-expert cache misses are silently skipped and the
    # validated-surrogate fail-fast never fires for MoE (it would abort later
    # in materialization with a less actionable message instead).
    class _FakeCache:
        def __init__(self):
            self.include_flag = None

        def assignment_keys(self, assignment, include_packed_experts=False):
            self.include_flag = include_packed_experts
            # the lone non-BF16 entry is a packed expert: counted as missing
            # only when the diagnostics opt into packed-expert coverage.
            missing = (
                [("model.layers.0.mlp.experts.gate_up_proj", "NVFP4")]
                if include_packed_experts else []
            )
            return [], missing

    cache = _FakeCache()
    diag = _production_cache_assignment_diagnostics(
        cache, {"model.layers.0.mlp.experts.gate_up_proj": "NVFP4"})
    assert cache.include_flag is True, \
        "diagnostics must pass include_packed_experts=True (M4)"
    assert diag["cache_miss_count"] == 1
    assert diag["missing_sample"][0][0].endswith("gate_up_proj")


# --------------------------------------------------------------------------
# R14 — held-out disjointness mechanized by hash
# --------------------------------------------------------------------------

def test_upstream_calibration_hashes_reads_meta_and_provenance():
    assert _upstream_calibration_hashes(
        {"meta": {"calib_hash": "aa"}}) == {"aa"}
    assert _upstream_calibration_hashes(
        {"provenance": {"calib_hashes": ["bb", "cc"]}}) == {"bb", "cc"}
    # Pre-R14 artifacts stamp nothing -> the check stays inert rather than guess.
    assert _upstream_calibration_hashes({"meta": {"nsamples": 8}}) == set()
    assert _upstream_calibration_hashes(None) == set()


def test_assert_calibration_disjoint_passes_on_disjoint_draws():
    checked = _assert_calibration_disjoint(
        ["sel1", "sel2"], {"probe": {"cost1"}, "cost": {"cost1", "cost2"}})
    assert checked["probe"]["overlap"] == []
    assert checked["cost"]["upstream_hashes"] == 2


def test_assert_calibration_disjoint_hard_errors_on_intersection():
    with pytest.raises(ValueError, match="held-out violation"):
        _assert_calibration_disjoint(["sel1", "shared"], {"cost": {"shared"}})


def test_calib_skip_first_on_wikitext_raises_instead_of_silently_skipping():
    """R14(iii): the wikitext branch computed `skip` and never applied it."""
    args = SimpleNamespace(
        calib_repeats=1, n_calib_samples=4, calib_skip_first=32,
        dataset="", calib_split="train", calib_seed=42, calib_seqlen=128,
        calib_repeat_seed_stride=1,
    )
    with pytest.raises(ValueError, match="only implemented for --dataset"):
        load_calibration_repeats(object(), args)

    # No skip requested -> the wikitext branch is untouched.
    args.calib_skip_first = 0
    calls = {}

    def _fake_loader(tokenizer, n, seqlen, *, split, seed):
        calls["seed"] = seed
        return torch.zeros(n, seqlen, dtype=torch.long)

    import prismaquant.validate_assignments_kl as mod
    from prismaquant import calibration_data
    original = calibration_data.load_wikitext_calibration_windowed
    calibration_data.load_wikitext_calibration_windowed = _fake_loader
    try:
        repeats = load_calibration_repeats(object(), args)
    finally:
        calibration_data.load_wikitext_calibration_windowed = original
    assert len(repeats) == 1 and repeats[0].shape == (4, 128)
    assert calls["seed"] == 42


def test_calibration_provenance_hashes_are_per_repeat():
    repeats = [torch.arange(8).reshape(2, 4), torch.arange(8, 16).reshape(2, 4)]
    prov = _calibration_provenance(repeats)
    assert len(prov["calib_repeat_hashes"]) == 2
    assert len(set(prov["calib_repeat_hashes"])) == 2
    # The combined digest is a different construction from the element hashes,
    # which is why the disjointness check intersects the ELEMENTS.
    assert prov["calib_hash"] not in prov["calib_repeat_hashes"]
