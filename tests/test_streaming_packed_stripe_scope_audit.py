"""Streaming include lists own packed render scope as well as dense scope."""
from __future__ import annotations

import pytest
import torch

from prismaquant import production_weight_cache as pwc
from prismaquant.measure_quant_cost import ActivationIndex
from prismaquant.model_profiles import DefaultProfile
from prismaquant.streaming_production_cache import run_streaming_render
from test_streaming_production_cache import HIDDEN, NUM_LAYERS, ToyModel


class _RecipeAliasProfile(DefaultProfile):
    def live_to_recipe_name(self, name):
        return name.replace("model.layers.", "model.text.layers.", 1)


def _fixture(tmp_path, monkeypatch, *, alias):
    model = ToyModel().eval()
    profile = _RecipeAliasProfile() if alias else DefaultProfile()
    live_names = [
        f"model.layers.{layer}.mlp.experts.{projection}"
        for layer in range(NUM_LAYERS)
        for projection in ("gate_up_proj", "down_proj")
    ]
    assignment = {
        profile.live_to_recipe_name(name): "FP8_E4M3" for name in live_names
    }
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    for layer in range(NUM_LAYERS):
        qname = f"model.layers.{layer}.mlp.experts"
        inputs = torch.arange(4 * HIDDEN, dtype=torch.float32).reshape(4, HIDDEN) / 128
        torch.save(
            {"name": qname, "inputs": inputs},
            act_dir / (qname.replace(".", "__") + ".pt"),
        )

    rendered = []

    def render(weight, fmt, *, qname, **kwargs):
        rendered.append(qname.rsplit(".e", 1)[0])
        return weight.clone()

    monkeypatch.setattr(pwc, "render_production_weight", render)

    def build(include_qnames):
        return run_streaming_render(
            model,
            layers_prefix="model.layers.",
            num_layers=NUM_LAYERS,
            render_assignment=assignment,
            act_index=ActivationIndex(act_dir, []),
            formats=("FP8_E4M3",),
            levers={"gptq": False},
            cache_dir_path=None,
            profile=profile,
            skip_tokens=(),
            device=torch.device("cpu"),
            include_qnames=include_qnames,
            progress=False,
        )

    return profile, live_names, rendered, build


@pytest.mark.parametrize("alias", [False, True])
def test_disjoint_include_lists_render_disjoint_packed_scopes(
    tmp_path, monkeypatch, alias,
):
    profile, live_names, rendered, build = _fixture(tmp_path, monkeypatch, alias=alias)
    observed = []
    for layer in range(NUM_LAYERS):
        selected = {name for name in live_names if f".layers.{layer}." in name}
        cache = build([profile.live_to_recipe_name(name) for name in sorted(selected)])
        assert set(cache.weights) == {(name, "FP8_E4M3") for name in selected}
        assert set(rendered) == selected
        assert cache.metadata["requested_entries"] == len(selected)
        observed.append(set(cache.weights))
        rendered.clear()
    assert observed[0].isdisjoint(observed[1])
    assert set.union(*observed) == {(name, "FP8_E4M3") for name in live_names}


@pytest.mark.parametrize("alias", [False, True])
def test_include_list_can_select_one_packed_projection(tmp_path, monkeypatch, alias):
    profile, live_names, rendered, build = _fixture(tmp_path, monkeypatch, alias=alias)
    selected = live_names[0]
    cache = build([profile.live_to_recipe_name(selected)])
    assert set(cache.weights) == {(selected, "FP8_E4M3")}
    assert set(rendered) == {selected}
    assert cache.metadata["requested_entries"] == 1


@pytest.mark.parametrize("include_qnames", [[], ["model.layers.0.self_attn.o_proj"]])
def test_include_without_packed_entries_does_not_render_packed_assignments(
    tmp_path, monkeypatch, include_qnames,
):
    _profile, _live_names, rendered, build = _fixture(tmp_path, monkeypatch, alias=False)
    cache = build(include_qnames)
    assert not cache.weights
    assert not rendered
    assert cache.metadata["requested_entries"] == 0


def test_absent_include_list_preserves_complete_packed_assignment(tmp_path, monkeypatch):
    _profile, live_names, rendered, build = _fixture(tmp_path, monkeypatch, alias=False)
    cache = build(None)
    assert set(cache.weights) == {(name, "FP8_E4M3") for name in live_names}
    assert set(rendered) == set(live_names)
    assert cache.metadata["requested_entries"] == len(live_names)
