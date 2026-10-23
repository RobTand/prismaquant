"""The shared unload owner releases selected aliases and keeps derived buffers."""
import pytest
import torch

from prismaquant.layer_streaming import _unload


@pytest.mark.parametrize("all_slots", [False, True], ids=["selected-prefixes", "whole-model"])
def test_unload_releases_every_selected_alias(all_slots):
    model = torch.nn.Module()
    model.left = torch.nn.Linear(4, 4, bias=False)
    model.right = torch.nn.Linear(4, 4, bias=False)
    model.right.weight = model.left.weight
    original = model.left.weight
    model.outside = torch.nn.Module()
    model.outside.weight = original
    persistent = torch.arange(4, dtype=torch.float32)
    model.left.register_buffer("persistent", persistent)
    model.right.register_buffer("persistent", persistent)
    derived = torch.nn.Module()
    derived.register_buffer("rotary", torch.arange(4, dtype=torch.float32), persistent=False)
    model.left.derived = derived
    model.right.derived = derived
    kept = derived.rotary

    count = _unload(model, [""] if all_slots else ["left.", "right."])

    assert model.left.weight.is_meta
    assert model.right.weight.is_meta
    assert model.left.persistent.is_meta
    assert model.right.persistent.is_meta
    assert model.left.derived.rotary is kept
    assert model.right.derived.rotary is kept
    assert not kept.is_meta
    if all_slots:
        assert model.outside.weight.is_meta
        assert count == 5
    else:
        assert model.outside.weight is original
        assert not model.outside.weight.is_meta
        assert count == 4
