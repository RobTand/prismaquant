from __future__ import annotations

import pytest
import torch

from prismaquant.cb_minchain import (
    MINCHAIN_FLAG,
    chain_identity,
    chain_identity_from_digest,
    epsilon_le,
    embed_predecessor,
    refine_one_entry,
    recipe_solution_digest,
    relative_epsilon,
    select_arm,
    solution_digest,
)
from prismaquant.nvfp4_cb_formats import (
    nvfp4_cb_fields,
    nvfp4_cb_reconstruct,
)


def _fields() -> dict:
    torch.manual_seed(7)
    weight = torch.randn(4, 256)
    return nvfp4_cb_fields(
        weight, 28, grid="fp8", mode="product", scale_sweep=False
    )


def test_embed_requires_explicit_pilot_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(MINCHAIN_FLAG, raising=False)
    with pytest.raises(RuntimeError, match=MINCHAIN_FLAG):
        embed_predecessor(_fields(), 29)


def test_embed_is_reconstruction_exact(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(MINCHAIN_FLAG, "1")
    predecessor = _fields()
    embedded = embed_predecessor(predecessor, 29)
    before = nvfp4_cb_reconstruct(
        predecessor, 28, grid="fp8", mode="product"
    )
    after = nvfp4_cb_reconstruct(
        embedded, 29, grid="fp8", mode="product"
    )
    assert torch.equal(before, after)
    assert torch.equal(embedded["scales"], predecessor["scales"])
    for old, new in zip(predecessor["codebook"], embedded["codebook"]):
        assert torch.equal(old, new[: old.shape[0]])


def test_digest_and_chain_identity_cover_arm_and_predecessor():
    fields = _fields()
    digest = solution_digest(fields)
    assert len(digest) == 64
    identity = chain_identity(
        winning_arm="embed", solution=fields, predecessor_digest=digest
    )
    assert identity["winning_arm"] == "embed"
    assert identity["predecessor_digest"] == digest
    with pytest.raises(ValueError, match="requires"):
        chain_identity(
            winning_arm="refine", solution=fields, predecessor_digest=None
        )
    recipe_digest = recipe_solution_digest({"qname": "x", "rung": 29})
    recipe_identity = chain_identity_from_digest(
        winning_arm="refine", solution_digest_value=recipe_digest,
        predecessor_digest=digest,
    )
    assert recipe_identity["digest_basis"] == "deterministic_content_gated_recipe"


def test_selection_is_deterministic_and_zero_tax():
    arm, error = select_arm({"free": 2.0, "embed": 1.0, "refine": 1.0})
    assert (arm, error) == ("embed", 1.0)
    arm, error = select_arm({"free": 0.5, "embed": 1.0, "refine": 0.75})
    assert (arm, error) == ("free", 0.5)


def test_optimized_two_arm_selection_uses_registered_epsilon():
    free = 1.0
    inside = free - 0.5e-12
    outside = free - 2.0e-12
    assert relative_epsilon(free, inside) == pytest.approx(1e-12)
    assert epsilon_le(free, inside)
    assert select_arm({"free": free, "embed": inside}) == ("free", free)
    assert select_arm({"free": free, "embed": outside}) == ("embed", outside)


def test_add_one_refine_freezes_prefix_and_scales(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(MINCHAIN_FLAG, "1")
    torch.manual_seed(11)
    weight = torch.randn(2, 256)
    col_weights = torch.rand(1, 256).add_(0.1)
    predecessor = nvfp4_cb_fields(
        weight, 28, grid="fp8", mode="product",
        col_weights=col_weights, scale_sweep=False,
    )
    refined = refine_one_entry(
        weight, predecessor, 29,
        col_weights=col_weights,
        activation_rows=torch.randn(4, 256),
        iterations=1,
    )
    assert torch.equal(refined["scales"], predecessor["scales"])
    for old, new in zip(predecessor["codebook"], refined["codebook"]):
        assert torch.equal(old, new[: old.shape[0]])
