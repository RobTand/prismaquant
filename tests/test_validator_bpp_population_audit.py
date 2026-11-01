"""Assigned mutable tensors must not disappear from reported validator bpp."""
from __future__ import annotations

import json
import pickle

import pytest

torch = pytest.importorskip("torch")

from prismaquant import format_registry as fr
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.perturbed_x_cache import build_quantizable_map
from prismaquant.validate_assignments_kl import (
    _assignment_bpp_details,
    _materialize_assignment_inplace,
)


class _Profile:
    def is_pinned_name(self, name):
        return name == "lm_head"

    def source_passthrough_prefixes(self):
        return ("mtp.", "model.visual.")


class _TwoLinears(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.small = torch.nn.Linear(32, 32, bias=False)
        self.large = torch.nn.Linear(64, 64, bias=False)
        with torch.no_grad():
            self.small.weight.fill_(0.25)
            self.large.weight.fill_(0.5)


def _stats(model):
    return {name + ".weight": {
        "n_params": module.weight.numel(), "in_features": module.in_features,
        "out_features": module.out_features,
    } for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}


def _specs():
    return {name: fr.get_format(name) for name in ("NVFP4", "BF16")}


@pytest.mark.parametrize("missing", ["quantized", "bf16", "all"])
def test_installed_body_cannot_be_omitted_from_bpp(missing):
    model = _TwoLinears()
    large_format = "BF16" if missing == "bf16" else "NVFP4"
    assignment = {"small.weight": "NVFP4", "large.weight": large_format}
    rendered = {("small.weight", "NVFP4"): torch.full_like(model.small.weight, 0.125)}
    if large_format != "BF16":
        rendered[("large.weight", large_format)] = torch.full_like(model.large.weight, 0.375)
    cache = ProductionWeightCache(weights=rendered, levers={})
    targets = build_quantizable_map(model)
    assert all(name in targets for name in assignment)
    installed = _materialize_assignment_inplace(model, assignment, cache)
    assert installed["copied"] == len(rendered)
    assert torch.equal(model.small.weight, rendered[("small.weight", "NVFP4")])
    if large_format != "BF16":
        assert torch.equal(model.large.weight, rendered[("large.weight", large_format)])
    stats = _stats(model)
    stats = {} if missing == "all" else {"small.weight": stats["small.weight"]}
    with pytest.raises((ValueError, RuntimeError), match="(?i)(stats|population|coverage|missing)"):
        _assignment_bpp_details(stats, assignment, _specs(), profile=_Profile())


def test_complete_body_reports_every_mutable_parameter():
    model = _TwoLinears()
    details = _assignment_bpp_details(
        _stats(model), {"small.weight": "NVFP4", "large.weight": "BF16"},
        _specs(), profile=_Profile())
    small_bytes = fr.get_format("NVFP4").memory_bytes_for_shape((32, 32)) + 8
    expected_bits = 8 * small_bytes + 16 * model.large.weight.numel()
    assert details["quantizable_entries"] == 2
    assert details["quantizable_params"] == sum(p.numel() for p in model.parameters())
    assert details["bpp"] == expected_bits / details["quantizable_params"]


def test_explicit_fixed_and_auxiliary_entries_need_no_body_stats():
    model = _TwoLinears()
    assignment = {"small.weight": "NVFP4", "lm_head": "BF16",
                  "mtp.fc": "BF16", "model.visual.proj": "NVFP4"}
    details = _assignment_bpp_details(
        {"small.weight": _stats(model)["small.weight"]}, assignment, _specs(), profile=_Profile())
    assert details["quantizable_entries"] == 1
    assert details["quantizable_params"] == model.small.weight.numel()


@pytest.mark.parametrize("known_fixed_stats", [False, True], ids=["absent-stats", "present-stats"])
def test_all_explicitly_exempt_scopes_keep_zero_body_bpp(known_fixed_stats):
    assignment = {"lm_head": "BF16", "mtp.fc": "BF16",
                  "model.visual.proj": "NVFP4"}
    stats = ({name: {"n_params": 32 * 32, "in_features": 32, "out_features": 32}
              for name in assignment} if known_fixed_stats else {})

    details = _assignment_bpp_details(stats, assignment, _specs(), profile=_Profile())

    assert details == {"bpp": 0.0, "quantizable_entries": 0,
                       "excluded_entries": len(stats), "quantizable_params": 0}


def test_empty_assignment_has_no_required_body_population():
    assert _assignment_bpp_details({}, {}, _specs(), profile=_Profile()) == {
        "bpp": 0.0, "quantizable_entries": 0,
        "excluded_entries": 0, "quantizable_params": 0,
    }


def test_missing_body_diagnostic_uses_scope_before_stats_filtering():
    assignment = {"lm_head": "BF16", "mtp.fc": "BF16",
                  "model.visual.proj": "NVFP4", "large.weight": "BF16"}

    with pytest.raises(ValueError, match="frontier candidate") as error:
        _assignment_bpp_details({}, assignment, _specs(), profile=_Profile(),
                                where="frontier candidate")

    message = str(error.value)
    assert "1 mutable assigned" in message and "large.weight" in message
    assert "lm_head" not in message and "mtp.fc" not in message
    assert "model.visual.proj" not in message
    assert "Rebuild the probe stats" in message


class _StagingBoundary(RuntimeError):
    pass


@pytest.mark.parametrize("missing_stats", [True, False],
                         ids=["missing-body-refuses", "complete-body-reaches-staging"])
def test_main_checks_population_before_source_staging_or_model_payload(
    tmp_path, monkeypatch, missing_stats,
):
    from prismaquant import validate_assignments_kl as validator
    from tests.test_resident_source_decode_audit import _source

    root, expected = _source(tmp_path, native=False)
    name = next(name for name in expected if name.endswith(".self_attn.o_proj.weight"))
    weight = expected[name]
    stats = {} if missing_stats else {name: {
        "n_params": weight.numel(), "in_features": weight.shape[1],
        "out_features": weight.shape[0],
    }}
    probe = tmp_path / "probe.pkl"
    with probe.open("wb") as handle:
        pickle.dump({"stats": stats}, handle)
    base = tmp_path / "base.json"
    base.write_text(json.dumps({name: "BF16"}))
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    staging_calls = []

    def stop_at_staging(model_path):
        staging_calls.append(model_path)
        raise _StagingBoundary("after real BPP admission")

    # Only hardware admission and the staging-boundary stop are adapted.
    # Parsing, local source config, profile scope and BPP admission stay real.
    monkeypatch.setattr(validator, "require_cuda_hot_path",
                        lambda *args, **kwargs: torch.device("cpu"))
    monkeypatch.setattr(validator, "stage_multimodal", stop_at_staging)
    arguments = ["--model", str(root), "--probe", str(probe),
                 "--base-assignment", str(base), "--assignment", str(candidate),
                 "--output", str(tmp_path / "result.json"), "--device", "cpu",
                 "--work-dir", str(tmp_path / "work"), "--formats", "NVFP4,BF16"]

    if missing_stats:
        with pytest.raises(ValueError, match="missing probe stats") as error:
            validator.main(arguments)
        assert name in str(error.value)
        assert not staging_calls
    else:
        with pytest.raises(_StagingBoundary, match="after real BPP admission"):
            validator.main(arguments)
        assert staging_calls == [str(root)]
    assert not (tmp_path / "result.json").exists()
