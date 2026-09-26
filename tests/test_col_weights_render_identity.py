"""Re-vet R3: `col_weights` on the production render path.

Two contracts, both load-bearing:

1. **Strictly additive.** Passing `col_weights` must leave every non-weighted
   family's rendered bytes BIT-IDENTICAL. That is the whole risk of touching a
   shared render path, so it is pinned here rather than argued in a docstring.
2. **One render.** For the weighted families (GGUF k-quants) the cache
   render must equal the exporter's / inline cost render's weighted render
   exactly — `emu_forward_kl.weighted_quantize_dequantize` is the single
   definition all three consume.

The codebook rungs that were the other weighted family, and the cases that
pinned their serialization context, were archived with the retired codebook
lane on 2026-09-25 (#1304).
"""
import pytest
import torch

pytest.importorskip("torch")


NON_WEIGHTED_FORMATS = ["NVFP4", "FP8_E4M3", "MXFP4", "MXFP8_E4M3", "BF16"]
WEIGHTED_FORMATS = ["Q4_K"]


def _render(fmt, weight, acts, col_weights, levers=None):
    import prismaquant.production_weight_cache as pwc

    return pwc.render_production_weight(
        weight,
        fmt,
        qname="lin",
        activations={"lin": acts},
        levers={"gptq": True} if levers is None else levers,
        col_weights=col_weights,
    )


@pytest.mark.parametrize("fmt", NON_WEIGHTED_FORMATS)
def test_col_weights_is_bit_identical_for_non_weighted_formats(fmt):
    torch.manual_seed(0)
    weight = torch.randn(8, 256)
    acts = torch.randn(64, 256) * 0.3
    cw = torch.rand(256) + 0.05

    without = _render(fmt, weight, acts, None)
    with_cw = _render(fmt, weight, acts, cw)

    assert without.dtype == with_cw.dtype
    assert torch.equal(without, with_cw), (
        f"{fmt} render changed when col_weights was supplied — the argument "
        "must be inert outside the weighted-render families"
    )


@pytest.mark.parametrize("fmt", WEIGHTED_FORMATS)
def test_weighted_families_render_through_the_single_definition(fmt):
    from prismaquant import format_registry as fr
    from prismaquant.emu_forward_kl import weighted_quantize_dequantize

    torch.manual_seed(1)
    weight = torch.randn(8, 256) * 0.2
    acts = torch.randn(32, 256) * 0.3
    cw = torch.rand(256) + 0.05

    spec = fr.get_format(fmt)
    rendered = _render(fmt, weight, acts, cw)
    expected = weighted_quantize_dequantize(spec, weight, cw).to(
        device=weight.device, dtype=weight.dtype)

    assert torch.equal(rendered, expected), (
        f"{fmt} cache render diverged from the shared weighted render — "
        "cost, KL and shipped bytes would not be one render"
    )


@pytest.mark.parametrize("fmt", WEIGHTED_FORMATS)
def test_weighted_families_actually_use_the_vector(fmt):
    """A guard against the assertion above passing vacuously (i.e. against the
    vector being silently dropped on the floor by both sides)."""
    torch.manual_seed(2)
    weight = torch.randn(8, 256) * 0.2
    acts = torch.randn(32, 256) * 0.3
    # A realistic imatrix: strictly positive, wide dynamic range. (An
    # all-but-a-few-zeros vector is degenerate — the weighted metric collapses
    # and the encoders legitimately reproduce the unweighted result.)
    cw = torch.rand(256) * 10.0 + 0.01

    uniform = _render(fmt, weight, acts, torch.ones_like(cw))
    weighted = _render(fmt, weight, acts, cw)
    assert not torch.equal(uniform, weighted)


def test_only_weighted_vq_is_offered_to_the_weighted_families():
    import prismaquant.production_weight_cache as pwc

    for fmt in WEIGHTED_FORMATS:
        assert pwc._weighted_render_family(fmt) is not None
        assert pwc._format_supports_render_mechanism(fmt, "weighted_vq")
        for mech in ("gptq", "static_act_order", "joint_scale_opt",
                     "scale_sweep", "four_over_six"):
            assert not pwc._format_supports_render_mechanism(fmt, mech)
    for fmt in NON_WEIGHTED_FORMATS:
        assert pwc._weighted_render_family(fmt) is None
        assert not pwc._format_supports_render_mechanism(fmt, "weighted_vq")


def test_weighted_vq_is_a_declared_render_mechanism():
    """R26's pattern: an extension point the render path branches on must be
    declared in the mechanism registry, not just spelled in an `if`."""
    from prismaquant.render_score import (
        registered_render_mechanisms,
        resolve_render_mechanism_order,
    )

    assert "weighted_vq" in registered_render_mechanisms()
    plan = resolve_render_mechanism_order(["weighted_vq"])
    assert plan.errors == ()
    assert plan.names() == ("weighted_vq",)


def test_packed_expert_col_weight_slicing_matches_the_cost_convention():
    """`_expert_col_weights` and `measure_quant_cost._item_col_weights` are the
    same convention; a divergence would weight cost and cache differently."""
    from prismaquant.measure_quant_cost import _item_col_weights
    from prismaquant.production_weight_cache import _expert_col_weights

    stack = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    for e in range(4):
        assert torch.equal(
            _expert_col_weights(stack, e, 4), _item_col_weights(stack, e, 4))

    pooled = torch.arange(8, dtype=torch.float32)
    for e in range(4):
        assert torch.equal(_expert_col_weights(pooled, e, 4), pooled)
    assert _expert_col_weights(None, 0, 4) is None


def test_build_cache_col_weights_refuses_the_confounding_combinations():
    from prismaquant.build_production_cache import _load_col_weights

    assert _load_col_weights(None, ["NVFP4", "FP8_DYNAMIC", "BF16"]) is None
    with pytest.raises(SystemExit, match="no --col-weights"):
        _load_col_weights(None, ["Q4_K", "BF16"])
    with pytest.raises(SystemExit, match="no weighted-render family"):
        _load_col_weights("unused.pkl", ["NVFP4", "BF16"])
