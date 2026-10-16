"""Fused-sibling NVFP4 joint globals: resident fill == streamed fill == export.

vLLM serves q/k/v (and gate/up) from one fused parameter with one
``weight_global_scale``, so every builder renders those siblings against a
joint per-tensor global, and the exporter re-derives the shipped NVFP4 codes
and FP8 block scales from the cached dequant under the joint global it
recomputes.  Principle 8 needs those to be one rendering: a builder that
derives the joint global under a different recipe from the one export uses
ships bytes the validated render never produced.

These cases drive the real algorithms on a tiny CPU fixture with the JSO
recipe (``gptq + static_act_order + joint_scale_opt`` records
``nvfp4_scale_rule=joint_mse``): the resident fill, the streamed fill, both
exporter pre-passes and the exporter's cached re-pack.  The fixture makes the
``joint_mse`` and ``static_6`` globals differ, so a builder on the wrong recipe
cannot pass by coincidence.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import prismaquant.export_native_compressed as enc
import prismaquant.native_render_codec as codec
from prismaquant.measure_quant_cost import ActivationIndex
from prismaquant.model_profiles import DefaultProfile
from prismaquant.native_render_levers import NativeRenderLevers
from prismaquant.nvfp4_activation_contract import _E2M1_POSITIVE
from prismaquant.production_weight_cache import (
    _LinearActivationCollector,
    fill_production_weight_cache,
)
from prismaquant.streaming_production_cache import run_streaming_render

HIDDEN = 32
INTER = 32
VOCAB = 64
LAYER = "model.layers.0"
JSO = {"gptq": True, "static_act_order": True, "joint_scale_opt": True}
FUSED = {
    f"{LAYER}.self_attn.{leaf}" for leaf in ("q_proj", "k_proj", "v_proj")
} | {f"{LAYER}.mlp.{leaf}" for leaf in ("gate_proj", "up_proj")}


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, h):
        return self.o_proj(self.q_proj(h) + self.k_proj(h) + self.v_proj(h))


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTER, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, HIDDEN, bias=False)

    def forward(self, h):
        return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _Mlp()

    def forward(self, h):
        # Every Linear sees its input; the residual stays the embedding so the
        # activation cache and the resident fill's own forward are identical.
        self.self_attn(h)
        self.mlp(h)
        return h


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([_Layer()])

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()

    def forward(self, input_ids, use_cache=False, **_kwargs):
        return self.model(input_ids)


def _make_model() -> _Model:
    torch.manual_seed(20261016)
    model = _Model()
    with torch.no_grad():
        # One group per fused group whose largest magnitudes sit on the
        # max-to-4 codebook levels: max-to-4 renders it exactly and wins the
        # joint_mse choice, so the joint_mse global is 1.5x the static_6 one.
        pattern = torch.tensor([1.0, 0.75, 0.5, 0.25] * 4)
        for linear in (model.model.layers[0].self_attn.q_proj,
                       model.model.layers[0].mlp.gate_proj):
            peak = 2.0 * float(linear.weight.abs().max())
            linear.weight[0, :16] = pattern * peak
    return model.to(torch.bfloat16).eval()


def _dense_qnames(model) -> list[str]:
    return sorted(
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    )


def _write_activation_cache(model, calib_ids, qnames, act_dir: Path) -> None:
    collector = _LinearActivationCollector(
        model, set(qnames), max_rows=4096, store_qnames=set(qnames),
        store_device="cpu", store_dtype=torch.float32,
    )
    collector.install()
    try:
        with torch.no_grad():
            for row in range(calib_ids.size(0)):
                model(calib_ids[row:row + 1], use_cache=False)
    finally:
        collector.remove()
    for name, inputs in collector.collected().items():
        stem = re.sub(r"[^A-Za-z0-9_-]", "__", name)
        torch.save({"inputs": inputs.detach().cpu(), "name": name},
                   act_dir / f"{stem}.pt")


def _joint_global_spy(monkeypatch):
    """Record every fused joint global the builders compute (real algorithm)."""
    seen: list[dict[str, torch.Tensor]] = []
    real = enc._compute_nvfp4_joint_global

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        seen.append({name: value.clone() for name, value in result.items()})
        return result

    monkeypatch.setattr(enc, "_compute_nvfp4_joint_global", spy)
    return seen


def _decode(packed: dict[str, torch.Tensor], global_real: torch.Tensor):
    """Dequantize an exported NVFP4 triple with the codec's own arithmetic."""
    weight_packed = packed["weight_packed"]
    rows, cols = weight_packed.shape[0], weight_packed.shape[1] * 2
    low = (weight_packed & 0xF).long()
    high = ((weight_packed >> 4) & 0xF).long()
    index = torch.stack([low, high], dim=-1).reshape(rows, cols)
    sign = -((index >> 3).to(torch.float32) * 2 - 1)
    levels = torch.tensor(_E2M1_POSITIVE, dtype=torch.float32)[index & 0x7]
    eff = (packed["weight_scale"].to(torch.float32)
           * global_real.to(torch.float32)).clamp_min(1e-12)
    eff = eff.unsqueeze(-1).expand(-1, -1, 16).reshape(rows, cols)
    return sign * levels * eff


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    model = _make_model()
    profile = DefaultProfile()
    qnames = _dense_qnames(model)
    assignment = {name: "NVFP4" for name in qnames}
    calib_ids = torch.randint(
        0, VOCAB, (2, 8), generator=torch.Generator().manual_seed(7),
    )
    act_dir = tmp_path_factory.mktemp("act")
    _write_activation_cache(model, calib_ids, qnames, act_dir)

    mp = pytest.MonkeyPatch()
    try:
        # The JSO record must choose joint_mse itself, not an ambient rule.
        mp.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
        seen = _joint_global_spy(mp)
        resident = fill_production_weight_cache(
            model, calib_ids, qnames,
            formats=["NVFP4"], render_assignment=assignment,
            levers=dict(JSO), max_act_rows=4096,
            recache_profile=profile, progress=False,
        )
        resident_globals = seen[-1]
        streamed = run_streaming_render(
            model,
            layers_prefix="model.layers.", num_layers=1,
            render_assignment=assignment,
            act_index=ActivationIndex(act_dir, []),
            formats=["NVFP4"], levers=dict(JSO),
            cache_dir_path=None, profile=profile, skip_tokens=[],
            device=torch.device("cpu"), max_act_rows=4096, progress=False,
        )
        streamed_globals = seen[-1]
    finally:
        mp.undo()
    return {
        "model": model,
        "profile": profile,
        "assignment": assignment,
        "resident": resident,
        "streamed": streamed,
        "resident_globals": resident_globals,
        "streamed_globals": streamed_globals,
    }


def _export_globals(rendered, cache, monkeypatch):
    monkeypatch.setattr(enc, "_PRODUCTION_WEIGHT_CACHE", cache)
    monkeypatch.setattr(enc, "_RENDER_LEVERS", NativeRenderLevers())
    monkeypatch.setattr(enc, "_INPUT_GLOBAL_SCALES", None)
    model, profile = rendered["model"], rendered["profile"]
    options = enc._export_render_options()
    layer = enc._compute_layer_joint_nvfp4(
        model.get_submodule(LAYER), LAYER, rendered["assignment"], profile,
        levers=options,
    )
    whole = enc._compute_nvfp4_joint_global(
        model, rendered["assignment"], profile=profile, levers=options,
    )
    return options, layer, whole


def test_fixture_separates_the_joint_mse_and_static_6_globals(rendered):
    weight = rendered["model"].get_submodule(f"{LAYER}.self_attn.q_proj").weight
    weight = weight.detach().float()
    static = codec.compute_nvfp4_global_real(
        weight, levers=NativeRenderLevers(nvfp4_scale_rule="static_6"))
    joint = codec.compute_nvfp4_global_real(
        weight, levers=NativeRenderLevers(nvfp4_scale_rule="joint_mse"))
    assert float(joint) > float(static)
    assert rendered["resident"].levers["nvfp4_scale_rule"] == "joint_mse"


@pytest.mark.parametrize("builder", ["resident", "streamed"])
def test_builders_render_fused_siblings_on_the_export_joint_global(
    rendered, builder, monkeypatch,
):
    cache = rendered[builder]
    options, layer, whole = _export_globals(rendered, cache, monkeypatch)
    assert options == NativeRenderLevers.from_record(cache.levers)
    assert set(layer) == FUSED and set(whole) == FUSED
    fill_globals = rendered[f"{builder}_globals"]
    assert set(fill_globals) == FUSED
    for name in sorted(FUSED):
        assert torch.equal(layer[name], whole[name]), name
        assert torch.equal(fill_globals[name], layer[name]), (
            f"{builder} fill rendered {name} on joint global "
            f"{float(fill_globals[name])!r}; export re-derives under "
            f"{float(layer[name])!r}"
        )


@pytest.mark.parametrize("builder", ["resident", "streamed"])
def test_export_repack_reproduces_the_cached_fused_render(
    rendered, builder, monkeypatch,
):
    cache = rendered[builder]
    _options, layer, _whole = _export_globals(rendered, cache, monkeypatch)
    for name in sorted(FUSED):
        cached = cache.get(name, "NVFP4")
        packed = enc._pack_production_cached_2d(
            name, "NVFP4", nvfp4_global_real_override=layer[name],
            device=torch.device("cpu"),
        )
        assert torch.equal(
            packed["weight_global_scale"],
            (1.0 / layer[name]).to(torch.float32).reshape(1),
        )
        shipped = _decode(packed, layer[name]).to(cached.dtype)
        differing = int((shipped != cached).sum())
        assert differing == 0, (
            f"{builder}: exported {name} differs from the cached render in "
            f"{differing}/{cached.numel()} elements (max |d|="
            f"{float((shipped.float() - cached.float()).abs().max()):.4g})"
        )


def test_streamed_and_resident_fused_renders_are_bit_identical(rendered):
    for name in sorted(FUSED):
        resident = rendered["resident"].get(name, "NVFP4")
        streamed = rendered["streamed"].get(name, "NVFP4")
        assert torch.equal(resident, streamed), name


def test_export_refuses_legacy_streamed_joint_mse_caches(rendered, monkeypatch):
    cache = rendered["streamed"]
    _options, globals_by_name, _whole = _export_globals(rendered, cache, monkeypatch)
    metadata = dict(cache.metadata)
    assert metadata["streaming"] is True
    metadata.pop("dense_nvfp4_joint_global_recipe", None)
    monkeypatch.setattr(cache, "metadata", metadata)
    name = sorted(FUSED)[0]
    with pytest.raises(RuntimeError, match="streamed NVFP4.*re-render"):
        enc._pack_production_cached_2d(
            name, "NVFP4", nvfp4_global_real_override=globals_by_name[name],
            device=torch.device("cpu"))
