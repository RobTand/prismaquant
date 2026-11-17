import os
import pathlib
import random

import pytest
import torch

# These tests exercise repo-root `tools/` scripts, which are not part of the
# installed package, so they can only run from a checkout. The release
# pipeline runs the suite against a non-editable install from outside the
# checkout (docs/RELEASING.md); skip there instead of failing collection.
if not (pathlib.Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip("requires a repo checkout (tools/ scripts)",
                allow_module_level=True)

from tools import smoke_graph_memory


def test_pin_rng_sets_expected_streams(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    smoke_graph_memory._pin_rng(12345)

    assert random.random() == 0.41661987254534116
    assert torch.rand(1).item() == 0.9817181825637817
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_smoke_seed_env_override(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_SMOKE_SEED", "42")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    smoke_graph_memory._set_smoke_env()
    smoke_graph_memory._pin_rng(int(os.environ["PRISMAQUANT_SMOKE_SEED"]))

    assert random.random() == 0.6394267984578837
    assert torch.rand(1).item() == 0.8822692632675171
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_smoke_linear_selection_prefers_body_and_excludes_head():
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(16, 2)
    model.fallback = torch.nn.Linear(16, 2)
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Linear(16, 2)])
    assert smoke_graph_memory._pick_linear(model) == ("model.layers.0", model.model.layers[0])
    del model.model
    assert smoke_graph_memory._pick_linear(model) == ("fallback", model.fallback)
    del model.fallback
    model.unaligned = torch.nn.Linear(15, 2)
    with pytest.raises(RuntimeError, match="no CUDA/NVFP4-compatible nn.Linear found"):
        smoke_graph_memory._pick_linear(model)
