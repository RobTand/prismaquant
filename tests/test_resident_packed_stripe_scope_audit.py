"""The resident CLI must apply include lists to packed production renders."""
from __future__ import annotations

import json
import pickle

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

from prismaquant import build_production_cache as builder
from prismaquant import production_weight_cache as pwc
from prismaquant.model_profiles import DefaultProfile
from test_streaming_production_cache import NUM_LAYERS, VOCAB, ToyModel


def _fixture(tmp_path, monkeypatch, *, alias):
    model = ToyModel().eval()
    profile = DefaultProfile()
    if alias:
        monkeypatch.setattr(
            profile, "live_to_recipe_name",
            lambda name: name.replace("model.layers.", "model.text.layers.", 1),
        )
    live_names = [
        f"model.layers.{layer}.mlp.experts.{projection}"
        for layer in range(NUM_LAYERS)
        for projection in ("gate_up_proj", "down_proj")
    ]
    assignment_path = tmp_path / "layer_config.json"
    assignment_path.write_text(json.dumps({name: "FP8_E4M3" for name in live_names}))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file(model.state_dict(), str(model_dir / "model.safetensors"))
    calibration = torch.arange(8, dtype=torch.long).reshape(2, 4) % VOCAB
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(builder, "detect_profile_with_warning", lambda *args, **kwargs: profile)
    monkeypatch.setattr(builder, "require_cuda_hot_path", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(builder, "_load_cache_calibration", lambda *args: calibration)
    rendered = []

    def render(weight, fmt, *, qname, **kwargs):
        rendered.append(qname.rsplit(".e", 1)[0])
        return weight.clone()

    monkeypatch.setattr(pwc, "render_production_weight", render)
    run_number = 0

    def build(include_qnames, *, scope="assignment"):
        nonlocal run_number
        run_dir = tmp_path / f"run-{run_number}"
        run_number += 1
        run_dir.mkdir()
        output = run_dir / "production_cache.pkl"
        argv = [
            "--model", str(model_dir), "--output", str(output),
            "--render-scope", scope, "--render-layer-config", str(assignment_path),
            "--formats", "FP8_E4M3", "--enable", "none", "--dtype", "fp32",
            "--max-act-rows", "4",
        ]
        if scope == "format-menu":
            argv[argv.index("--formats") + 1] = "NVFP4"
            argv += ["--render-packed-experts", "--expert-render-mode", "per_expert"]
        if include_qnames is not None:
            include_path = run_dir / "include.txt"
            include_path.write_text("".join(name + "\n" for name in include_qnames))
            argv += ["--include-qnames-file", str(include_path)]
        assert builder.main(argv) == 0
        return pickle.loads(output.read_bytes())

    return profile, live_names, rendered, build


@pytest.mark.parametrize("alias", [False, True])
def test_resident_disjoint_include_lists_render_disjoint_packed_scopes(
    tmp_path, monkeypatch, alias,
):
    _profile, live_names, rendered, build = _fixture(tmp_path, monkeypatch, alias=alias)
    observed = []
    for layer in range(NUM_LAYERS):
        selected = {name for name in live_names if f".layers.{layer}." in name}
        cache = build(sorted(selected))
        assert set(cache.weights) == {(name, "FP8_E4M3") for name in selected}
        assert set(rendered) == selected
        assert cache.metadata["requested_entries"] == len(selected)
        observed.append(set(cache.weights))
        rendered.clear()
    assert observed[0].isdisjoint(observed[1])
    assert set.union(*observed) == {(name, "FP8_E4M3") for name in live_names}


@pytest.mark.parametrize("alias", [False, True])
def test_resident_include_list_can_select_one_packed_projection(tmp_path, monkeypatch, alias):
    _profile, live_names, rendered, build = _fixture(tmp_path, monkeypatch, alias=alias)
    selected = live_names[0]
    cache = build([selected])
    assert set(cache.weights) == {(selected, "FP8_E4M3")}
    assert set(rendered) == {selected}
    assert cache.metadata["requested_entries"] == 1


@pytest.mark.parametrize("include_qnames", [[], ["model.layers.0.self_attn.o_proj"]])
def test_resident_include_without_packed_entries_does_not_render_packed_assignments(
    tmp_path, monkeypatch, include_qnames,
):
    _profile, _live_names, rendered, build = _fixture(tmp_path, monkeypatch, alias=False)
    cache = build(include_qnames)
    assert not cache.weights
    assert not rendered
    assert cache.metadata["requested_entries"] == 0


@pytest.mark.parametrize("alias", [False, True])
def test_resident_absent_include_preserves_complete_packed_assignment(tmp_path, monkeypatch, alias):
    _profile, live_names, rendered, build = _fixture(tmp_path, monkeypatch, alias=alias)
    cache = build(None)
    assert set(cache.weights) == {(name, "FP8_E4M3") for name in live_names}
    assert set(rendered) == set(live_names)
    assert cache.metadata["requested_entries"] == len(live_names)


@pytest.mark.parametrize("alias", [False, True])
def test_resident_format_menu_include_selects_one_packed_projection(tmp_path, monkeypatch, alias):
    profile, live_names, rendered, build = _fixture(tmp_path, monkeypatch, alias=alias)
    selected = live_names[0]
    cache = build([profile.live_to_recipe_name(selected)], scope="format-menu")
    assert set(cache.weights) == {(selected, "NVFP4")}
    assert set(rendered) == {selected}
    assert cache.metadata["requested_entries"] == 1


@pytest.mark.parametrize("include_qnames", [[], ["model.layers.0.self_attn.o_proj"]])
def test_resident_format_menu_without_packed_entries_does_not_render_packed_assignments(
    tmp_path, monkeypatch, include_qnames,
):
    _profile, _live_names, rendered, build = _fixture(tmp_path, monkeypatch, alias=False)
    cache = build(include_qnames, scope="format-menu")
    assert not any(".experts." in name for name, _fmt in cache.weights)
    assert not any(".experts." in name for name in rendered)
    assert cache.metadata["requested_entries"] == len(cache.weights)


@pytest.mark.parametrize("alias", [False, True])
def test_resident_format_menu_absent_include_preserves_complete_packed_scope(
    tmp_path, monkeypatch, alias,
):
    _profile, live_names, rendered, build = _fixture(tmp_path, monkeypatch, alias=alias)
    cache = build(None, scope="format-menu")
    assert {(name, "NVFP4") for name in live_names} <= set(cache.weights)
    assert set(live_names) <= set(rendered)
    assert cache.metadata["requested_entries"] == len(cache.weights)


@pytest.mark.parametrize("scope", ["assignment", "format-menu"])
def test_packed_include_preserves_selected_render_numerics(scope):
    from test_packed_expert_hook_scope import (
        FORMAT, LEVERS, MODULE_TOKEN_BUDGET, TwoLayerMoe,
        all_tensor_qnames, digest, frozen_calib, frozen_state, run_arm,
    )

    state, calib = frozen_state(), frozen_calib()
    reference = run_arm(state, calib, force_format=FORMAT)
    selected = [name for name in all_tensor_qnames() if name.startswith("layers.1.")]
    model = TwoLayerMoe().eval()
    model.load_state_dict(state)
    cache = pwc.ProductionWeightCache(weights={}, levers=dict(LEVERS))
    render_scope = (
        {"force_format": FORMAT} if scope == "format-menu"
        else {"render_assignment": {name: FORMAT for name in all_tensor_qnames()}}
    )
    pwc.fill_packed_expert_cache_entries(
        cache, model, calib,
        **render_scope,
        include_qnames=selected,
        levers=dict(LEVERS),
        profile=None,
        module_token_budget=MODULE_TOKEN_BUDGET,
        progress=False,
    )
    assert {
        name: digest(weight) for (name, fmt), weight in cache.weights.items()
        if fmt == FORMAT
    } == {name: reference["weights"][name] for name in selected}
