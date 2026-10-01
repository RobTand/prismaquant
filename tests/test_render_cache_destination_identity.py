"""Reject ambiguous legacy render destinations before cache resume/write."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prismaquant import production_weight_cache as pwc


COLLISIONS = [
    {"layer.a": ["BF16"], "layer_a": ["BF16"]},
    {"layer/a": ["BF16"], "layer__a": ["BF16"]},
    {"layer.a": ["FP8"], "layer_a": ["FP8_DYNAMIC"]},
]


def _identity(rendered):
    return pwc.build_production_cache_render_identity(
        render_scope="format-menu",
        requested_formats=sorted({fmt for fmts in rendered.values() for fmt in fmts}),
        levers={"gptq": False},
        mechanism_plan=[],
        calib_hash="1" * 64,
        eligible_qnames=rendered,
        render_formats_by_qname=rendered,
        max_act_rows=16,
    )


@pytest.mark.parametrize("rendered", COLLISIONS)
def test_new_identity_refuses_distinct_coordinates_with_one_destination(rendered):
    with pytest.raises(ValueError, match="rendered.*destination"):
        _identity(rendered)


@pytest.mark.parametrize("rendered", COLLISIONS)
def test_resumed_identity_refuses_distinct_coordinates_with_one_destination(rendered):
    identity = _identity({"safe": ["BF16"]})
    identity["rendered_pairs"] = sorted(
        f"{qname}|{fmt}" for qname, formats in rendered.items() for fmt in formats
    )
    with pytest.raises(ValueError, match="rendered.*destination"):
        pwc.validate_production_cache_render_identity(identity, where="resumed cache")


@pytest.mark.parametrize("rendered", [
    {"layer.a": ["BF16"], "layer_a": ["FP8_E4M3"]},
    {"layer/a": ["BF16"], "other": ["BF16"]},
    {"layer": ["FP8", "FP8_DYNAMIC", "FP8_E4M3"]},
    {"layer": ["BF16", "BF16"]},
    {"layer|part": ["BF16"], "other": ["BF16"]},
])
def test_unambiguous_identity_keeps_its_existing_fields(rendered):
    identity = _identity(rendered)
    assert pwc.validate_production_cache_render_identity(identity) == identity
    assert identity["rendered_pairs"] == sorted({
        f"{qname}|{fmt}" for qname, formats in rendered.items() for fmt in formats
    })


def test_disk_fill_refuses_alias_before_sidecar_or_shard_work(tmp_path, monkeypatch):
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.a = nn.Linear(2, 2, bias=False)
    model.layer_a = nn.Linear(2, 2, bias=False)
    leaf = tmp_path / pwc._cache_weight_filename("layer.a", "FP8_E4M3")
    leaf.write_bytes(b"preexisting shard must not be consumed or changed")
    before = leaf.read_bytes()
    calls = []

    def unexpected(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("entered sidecar/shard work before refusing alias")

    monkeypatch.setattr(pwc, "_check_production_cache_render_identity", unexpected)
    with pytest.raises(ValueError, match="rendered.*destination"):
        pwc.fill_production_weight_cache(
            model,
            torch.zeros((1, 2), dtype=torch.long),
            ["layer.a", "layer_a"],
            formats=["FP8_E4M3"],
            levers={"gptq": False},
            cache_dir=tmp_path,
            recache_profile=SimpleNamespace(),
            progress=False,
        )
    assert calls == []
    assert leaf.read_bytes() == before
    assert list(tmp_path.iterdir()) == [leaf]
