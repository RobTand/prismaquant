from __future__ import annotations

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.decision_units import discover_units
from prismaquant.model_profiles.qwen3 import Qwen3Profile


class _QwenMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(64, 128, bias=False)
        self.up_proj = nn.Linear(64, 128, bias=False)
        self.down_proj = nn.Linear(128, 64, bias=False)


class _QwenLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _QwenMlp()


class _QwenToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_QwenLayer()])


class _PackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 128, 64))
        self.down_proj = nn.Parameter(torch.zeros(2, 64, 64))


class _QwenMoeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = _PackedExperts()


class _QwenMoeToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_QwenMoeLayer()])


def _units_by_name(blocks):
    return {
        unit.name: unit
        for units in blocks.values()
        for unit in units
    }


def test_discover_units_uses_model_graph_fused_groups():
    formats = [fr.get_format("NVFP4"), fr.get_format("BF16")]
    blocks, singletons, n_params = discover_units(
        _QwenToy(),
        Qwen3Profile(),
        formats,
    )
    units = _units_by_name(blocks)

    assert singletons == []
    fused = units["model.layers.0.mlp.gate_up_proj"]
    assert fused.member_qnames == (
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    )
    assert {option.fmt for option in fused.options} == {"NVFP4", "BF16"}
    assert n_params["model.layers.0.mlp.gate_up_proj"] == 2 * 64 * 128

    down = units["model.layers.0.mlp.down_proj"]
    assert down.member_qnames == ("model.layers.0.mlp.down_proj",)


def test_discover_units_uses_configured_serving_profile_by_default():
    formats = [
        fr.get_format("MXFP4"),
        fr.get_format("NVFP4"),
        fr.get_format("BF16"),
    ]

    blocks, _singletons, _n_params = discover_units(
        _QwenToy(),
        Qwen3Profile(),
        formats,
    )
    fused = _units_by_name(blocks)["model.layers.0.mlp.gate_up_proj"]
    assert {option.fmt for option in fused.options} == {"NVFP4", "BF16"}

    research_blocks, _singletons, _n_params = discover_units(
        _QwenToy(),
        Qwen3Profile(),
        formats,
        target_profile="research",
    )
    research_fused = _units_by_name(research_blocks)[
        "model.layers.0.mlp.gate_up_proj"
    ]
    assert {option.fmt for option in research_fused.options} == {
        "MXFP4",
        "NVFP4",
        "BF16",
    }


def test_discover_units_groups_packed_experts_for_cooptimization():
    formats = [fr.get_format("NVFP4"), fr.get_format("BF16")]

    blocks, singletons, n_params = discover_units(
        _QwenMoeToy(),
        Qwen3Profile(),
        formats,
    )
    units = _units_by_name(blocks)

    assert singletons == []
    packed = units["model.layers.0.mlp.experts"]
    assert packed.member_qnames == (
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    )
    assert {option.fmt for option in packed.options} == {"NVFP4", "BF16"}
    assert n_params["model.layers.0.mlp.experts"] == (
        2 * 128 * 64 + 2 * 64 * 64
    )
