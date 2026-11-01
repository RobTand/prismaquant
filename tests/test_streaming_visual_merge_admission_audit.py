"""RE-09 baseline draft. Exercises the real visual probe without a checkpoint."""
from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoProcessor

from prismaquant import layer_streaming, model_profiles, sensitivity_probe, streaming_model


class _VisualModel(nn.Module):
    def __init__(self, *, merge_fails):
        super().__init__()
        self.visual = nn.Linear(4, 4, bias=False)
        self.text = nn.Module()
        self.text.embed_tokens = nn.Embedding(8, 4)
        self.text.norm = nn.Identity()
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.config = SimpleNamespace(num_experts_per_tok=1)
        self.merge_fails = merge_fails
        with torch.no_grad():
            self.visual.weight.copy_(torch.eye(4))
            self.text.embed_tokens.weight.zero_()
            self.lm_head.weight.copy_(torch.arange(32).reshape(8, 4) * 0.01)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.visual.weight.requires_grad_(True)

    def get_image_features(self, pixel_values, *, image_grid_thw):
        return self.visual(pixel_values)

    def get_placeholder_mask(self, input_ids, *, inputs_embeds, image_features):
        if self.merge_fails:
            raise ValueError("image placeholders do not match image features")
        mask = (input_ids == 1).unsqueeze(-1).expand_as(inputs_embeds)
        return mask, None


@pytest.mark.parametrize("merge_fails", [True, False])
def test_streaming_visual_probe_requires_a_successful_image_merge(
    tmp_path, monkeypatch, merge_fails,
):
    model = _VisualModel(merge_fails=merge_fails).eval()
    context = SimpleNamespace(
        model=model, base_model=model.text, visual_module=model.visual,
        visual_prefix="visual", layers=[], num_layers=0,
        layers_prefix="model.layers.",
        layer_cache=SimpleNamespace(clear=lambda: None),
        schedule_prefetch=lambda layer: None,
        shutdown=lambda: None,
    )
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(
        streaming_model, "_build_streaming_context", lambda *a, **k: context)
    monkeypatch.setattr(model_profiles, "profile_from_model",
                        lambda model: model_profiles.DefaultProfile())
    monkeypatch.setattr(layer_streaming, "_compute_attention_mask", lambda *a: None)
    monkeypatch.setattr(layer_streaming, "_compute_position_embeddings", lambda *a: None)
    monkeypatch.setattr(sensitivity_probe, "load_multimodal_calibration", lambda *a: (
        [{"input_ids": torch.tensor([[1, 2, 3]]),
          "pixel_values": torch.ones(1, 4),
          "image_grid_thw": torch.ones(1, 3, dtype=torch.long)}],
        {"real_loaded": 1, "synthetic_loaded": 0},
    ))
    output = tmp_path / "visual.pkl"
    accepted = sensitivity_probe.run_streaming_multimodal_visual_probe_pass(
        "audit/tiny-visual", dataset_name="audit", n_samples=1,
        max_text_len=3, requested_device="cpu", dtype=torch.float32,
        linear_include=r"^visual$", linear_exclude=r"^$",
        activation_cache_dir=None, output_path=str(output),
        offload_folder=str(tmp_path / "offload"),
    )

    if merge_fails:
        assert accepted is False
        if output.exists():
            with output.open("rb") as stream:
                assert pickle.load(stream)["meta"]["nsamples"] == 0
    else:
        assert accepted is True
        with output.open("rb") as stream:
            payload = pickle.load(stream)
        assert payload["meta"]["nsamples"] == 1
        assert payload["stats"]["visual"]["h_trace_raw"] > 0.0
