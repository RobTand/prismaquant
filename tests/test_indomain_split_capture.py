"""The research quantum must consume the split its persisted receipt describes."""
from types import SimpleNamespace
import json

import pytest
import torch

from experiments import indomain_split_capture as capture


def test_matching_split_publishes_disjoint_role_coordinates(tmp_path):
    capture._toy_control_preflight(tmp_path, lambda _label: None)
    root = tmp_path / "capture"
    manifest = json.loads((root / "layers/L000/manifest.json").read_text())
    for name, roles in manifest["units"].items():
        assert roles["fit"]["count"] + roles["heldout"]["count"] == manifest["full_counts"][name]
        for role, (lo, hi) in (("fit", (0, 2)), ("heldout", (2, 4))):
            record = torch.load(root / roles[role]["file"], weights_only=True)
            ids = record["prefix_sample_ids"]
            assert torch.all((ids >= lo) & (ids < hi))
            assert len(ids) == record["inputs"].shape[0]


def test_quantum_refuses_a_different_fit_boundary_before_model_load(tmp_path, monkeypatch):
    capture._toy_control_preflight(tmp_path, lambda _label: None)
    args = SimpleNamespace(
        capture_root=str(tmp_path / "capture"), calibration_census=str(tmp_path / "census.json"),
        units=str(tmp_path / "units.json"), calibration_tokens=str(tmp_path / "tokens.safetensors"),
        corpus_text=str(tmp_path / "corpus.txt"), model=str(tmp_path / "source"),
        source_snapshot_root=None, total_samples=4, fit_stop=1, max_prefix_rows=4,
        max_act_rows=7, attention_implementation="eager", device="cpu",
        capture_layer_range="0:1", cache_dir=str(tmp_path / "unexpected-cache"),
        streaming_cache_slots=2, streaming_prefetch_workers=1, streaming_cache_headroom_gb=0.0)
    monkeypatch.setattr(capture, "build_streamed_causal_lm",
                        lambda *_args, **_kwargs: pytest.fail("mislabeled split reached model load"))
    with pytest.raises(capture.ResearchRefused, match="fit_stop"):
        capture.mode_quantum(args, lambda _label: None)
