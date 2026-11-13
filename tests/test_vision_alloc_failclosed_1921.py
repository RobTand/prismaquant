"""Synthetic CPU allocator-main regression evidence for #1921 / #2809.

These are invented probe-row fixtures, NOT production measurements or vision
export/serving qualification. The tiny BF16 safetensors are real CPU files: the
allocator must reconcile their headers through a detected profile, not a mocked
source census. No allocator/profile implementation or serving gates are changed.
"""
from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import allocator
from prismaquant.layer_config import load_assignment
from prismaquant.model_profiles.registry import detect_profile
import test_visual_allocator_roots_1921 as root_fixture
import test_visual_allocator_units_1936 as measured_fixture
from test_visual_shard_coverage import _config


@pytest.fixture(params=["qwen3_5_moe", "gemma4"])
def synthetic_cpu_inputs(tmp_path, monkeypatch, request):
    """Reuse the probe rows/lane contract, but remove the helper's census stub."""
    real_census = allocator.discover_visual_linear_stats_from_source
    if request.param == "gemma4":
        output, probe = root_fixture._gemma_inputs(tmp_path, monkeypatch)
        vision, merger = root_fixture.VISION, root_fixture.PROJECTOR
    else:
        output, probe = measured_fixture._inputs(tmp_path, monkeypatch)
        vision, merger = measured_fixture.VISION, measured_fixture.MERGER
        # Unlike the old qwen3_moe fixture, this profile declares vision roots.
        _config(tmp_path / "model", "qwen3_5_moe",
                "Qwen3_5MoeForConditionalGeneration")
    monkeypatch.setattr(allocator, "discover_visual_linear_stats_from_source", real_census)

    def argument_path(flag):
        return Path(sys.argv[sys.argv.index(flag) + 1])

    probe_path = argument_path("--probe")
    cost_path = argument_path("--costs")
    model_path = Path(probe["meta"]["model"])
    profile = detect_profile(str(model_path))
    roots = profile.visual_root_prefixes()
    assert roots, "this integration fixture must exercise declared profile roots"
    assert all(any(name.startswith(root + ".") for root in roots)
               for name in (vision, merger))

    # Include body weights too, although only the declared visual scopes enter
    # the census. Explicit device=cpu avoids an ambient default CUDA device.
    tensors = {
        name + ".weight": torch.zeros(
            (entry["out_features"], entry["in_features"]),
            dtype=torch.bfloat16, device="cpu")
        for name, entry in probe["stats"].items()
    }
    # Own the source-dtype fixture as well as the measured rows: a text-only
    # manifest omits the visual units and fails source-bpp legality before the
    # population checks under test. All of these synthetic weights are BF16.
    # Keep this separate from the REAL visual census restored above; this is
    # not evidence that production manifest discovery admits visual weights.
    # Read tensors at call time so source-only additions remain represented.
    def source_manifest(_model, _profile=None):
        assert all(weight.dtype == torch.bfloat16 for weight in tensors.values())
        return {name.removesuffix(".weight"): "bf16" for name in tensors}

    monkeypatch.setattr(allocator, "_scan_source_dtype_manifest", source_manifest)
    source = model_path / "model.safetensors"
    # Cover both source-discovery branches without duplicating the old isolated
    # census tests: Gemma main reads an index, Qwen main scans a single file.
    index = model_path / "model.safetensors.index.json" if request.param == "gemma4" else None

    def write_source():
        save_file(tensors, str(source))
        if index is not None:
            index.write_text(json.dumps({"weight_map": {
                name: source.name for name in tensors}}))

    write_source()
    source_stats = real_census(str(model_path), strict=True, profile=profile)
    assert set(source_stats) == {vision, merger}
    for name in (vision, merger):
        for field in ("n_params", "in_features", "out_features"):
            assert source_stats[name][field] == probe["stats"][name][field]

    return SimpleNamespace(
        output=output, probe=probe, probe_path=probe_path, cost_path=cost_path,
        visuals=(vision, merger),
        attribution=argument_path("--bit-attribution-json"),
    )


def _write_pickle(path, value):
    path.write_bytes(pickle.dumps(value))


def _set_cli_option(monkeypatch, flag, value):
    argv = list(sys.argv)
    argv[argv.index(flag) + 1] = value
    monkeypatch.setattr(sys, "argv", argv)


@pytest.mark.parametrize("control", ["explicit_uniform", "text_only_source_census"])
def test_synthetic_cpu_main_retains_explicit_uniform_and_text_only_controls(
        synthetic_cpu_inputs, monkeypatch, control):
    case = synthetic_cpu_inputs
    probe = copy.deepcopy(case.probe)
    cost = pickle.loads(case.cost_path.read_bytes())
    if control == "explicit_uniform":
        # An incomplete visual cost population still takes the uniform format.
        del cost["costs"][case.visuals[1]]
    else:
        # No visual probe OR cost rows: real source headers must supply both
        # roots' fixed entries, including Gemma's separate embed_vision root.
        for name in case.visuals:
            del probe["stats"][name]
            del cost["costs"][name]
    _set_cli_option(monkeypatch, "--visual-format", "BF16")
    _write_pickle(case.probe_path, probe)
    _write_pickle(case.cost_path, cost)

    allocator.main()
    assignment = load_assignment(case.output)
    assert all(assignment[name] == "BF16" for name in case.visuals)
    assert set(assignment) == set(case.probe["stats"])
    attribution = json.loads(case.attribution.read_text())
    body = {name: entry for name, entry in case.probe["stats"].items()
            if name not in case.visuals}
    assert attribution["n_body_linears"] == len(body)
    assert attribution["body_quantizable_params"] == sum(
        entry["n_params"] for entry in body.values())
