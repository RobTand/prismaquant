"""Equivalence tests for the batched NVFP4 GPTQ + scale-sweep path.

These verify that the batched output matches the per-Linear output
within numerical tolerance on a small synthetic example. The batched
math is bitwise the same algorithm; differences should only come from
floating-point reduction order (the batched Cholesky / bmm may pick a
different reduction tree than the per-Linear sequential path).
"""
from __future__ import annotations

from dataclasses import replace
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

import prismaquant.export_native_compressed as _test_exporter
from prismaquant.native_render_levers import NativeRenderLevers
import prismaquant.native_render_codec as _native_render_codec


from prismaquant.native_render_codec import (
    gptq_obs_rounding_nvfp4,
    rtn_dequant_nvfp4,
    scale_sweep_nvfp4,
)
from prismaquant.export_batched_gptq import (
    gptq_obs_rounding_nvfp4_batched,
    scale_sweep_nvfp4_batched,
)


@pytest.fixture(autouse=True)
def _isolate_export_inputs():
    saved = (_test_exporter._RENDER_LEVERS, _test_exporter._CACHED_ACTIVATIONS,
             _test_exporter._INPUT_GLOBAL_SCALES)
    _test_exporter._RENDER_LEVERS = None
    _test_exporter._CACHED_ACTIVATIONS = None
    _test_exporter._INPUT_GLOBAL_SCALES = None
    yield
    (_test_exporter._RENDER_LEVERS, _test_exporter._CACHED_ACTIVATIONS,
     _test_exporter._INPUT_GLOBAL_SCALES) = saved


@pytest.fixture
def seeded():
    torch.manual_seed(0)


def _make_inputs(E: int, out_features: int, in_features: int, T: int):
    weights = torch.randn(
        E, out_features, in_features, dtype=torch.float32) * 0.1
    activations_list = [
        torch.randn(T + e, in_features, dtype=torch.float32) for e in range(E)
    ]
    return weights, activations_list


def test_gptq_batched_matches_per_linear(seeded):
    """Run GPTQ per-Linear and batched on the same inputs; outputs
    should agree to within FP32 reduction-order tolerance."""
    E, out_features, in_features = 4, 32, 64
    weights, activations_list = _make_inputs(E, out_features, in_features, T=128)

    per_outputs = []
    for e in range(E):
        per_outputs.append(gptq_obs_rounding_nvfp4(
            weights[e], activations_list[e], group_size=16,
            levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
        ))
    per_stack = torch.stack(per_outputs, dim=0)

    batch_out = gptq_obs_rounding_nvfp4_batched(
        weights, activations_list, group_size=16, expert_chunk=2,
        levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
    )
    # Reduction-order may differ — allow a tight tolerance.
    assert torch.allclose(per_stack, batch_out, atol=1e-4, rtol=1e-3), (
        f"max abs diff: {(per_stack - batch_out).abs().max().item():.3e}\n"
        f"max rel diff: {((per_stack - batch_out).abs() / per_stack.abs().clamp_min(1e-12)).max().item():.3e}")


def test_gptq_batched_empty_acts_ships_rtn_not_zeros(seeded):
    """M1 (2026-07-02 audit): a Linear with NO cached activations must
    ship as NVFP4 RTN (identity-Hessian GPTQ degenerates exactly to
    RTN, matching the per-Linear path's fallback) — NOT as all-zero
    weights. Linears with activations in the same batch are untouched."""
    E, out_features, in_features = 3, 16, 32
    weights, activations_list = _make_inputs(E, out_features, in_features, T=64)
    # Zero-numel activation tensor = "no activations available".
    activations_list[1] = torch.zeros(0, in_features, dtype=torch.float32)

    out = gptq_obs_rounding_nvfp4_batched(
        weights, activations_list, group_size=16, expert_chunk=4,
        levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
    )
    # The empty-acts Linear ships the same bytes as plain RTN — nonzero.
    expected_rtn = rtn_dequant_nvfp4(weights[1], group_size=16, levers=(_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()))
    assert out[1].abs().max() > 0.0
    torch.testing.assert_close(out[1], expected_rtn)
    # Sibling Linears with activations still match the per-Linear GPTQ.
    for e in (0, 2):
        per = gptq_obs_rounding_nvfp4(
            weights[e], activations_list[e], group_size=16, levers=(_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()))
        assert torch.allclose(per, out[e], atol=1e-4, rtol=1e-3)


def test_gptq_batched_dead_activation_column_not_zeroed(seeded):
    """§3.16: an all-zero activation CHANNEL (dead column) is detected
    pre-damping, gets an identity H diagonal, and its weights are NOT
    zeroed (serving-safe deviation from reference GPTQ)."""
    E, out_features, in_features = 2, 16, 32
    weights, activations_list = _make_inputs(E, out_features, in_features, T=64)
    dead_col = 5
    activations_list[0][:, dead_col] = 0.0

    out = gptq_obs_rounding_nvfp4_batched(
        weights, activations_list, group_size=16, expert_chunk=4,
        levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
    )
    # The dead column survives (quantized, not destroyed).
    assert out[0][:, dead_col].abs().max() > 0.0
    # And it stays consistent with the per-Linear path on the same input.
    per = gptq_obs_rounding_nvfp4(
        weights[0], activations_list[0], group_size=16, levers=(_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()))
    assert torch.allclose(per, out[0], atol=1e-4, rtol=1e-3)


def test_gptq_batched_with_global_real_overrides(seeded):
    """Per-Linear `global_real` matches between paths when explicit
    overrides are supplied (no implicit per-Linear computation)."""
    E, out_features, in_features = 4, 32, 64
    weights, activations_list = _make_inputs(E, out_features, in_features, T=128)
    overrides = torch.full((E,), 0.05, dtype=torch.float32)

    per_outputs = []
    for e in range(E):
        per_outputs.append(gptq_obs_rounding_nvfp4(
            weights[e], activations_list[e], group_size=16,
            global_real_override=overrides[e],
            levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
        ))
    per_stack = torch.stack(per_outputs, dim=0)

    batch_out = gptq_obs_rounding_nvfp4_batched(
        weights, activations_list, group_size=16,
        global_real_overrides=overrides, expert_chunk=2,
        levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
    )
    assert torch.allclose(per_stack, batch_out, atol=1e-4, rtol=1e-3)


def test_gptq_batched_cholesky_failure_falls_back_to_rtn(seeded):
    E, out_features, in_features = 3, 16, 32
    weights, activations_list = _make_inputs(E, out_features, in_features, T=64)

    with mock.patch("torch.linalg.cholesky", side_effect=RuntimeError("boom")):
        out = gptq_obs_rounding_nvfp4_batched(
            weights,
            activations_list,
            group_size=16,
            expert_chunk=4,
            levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
        )

    expected = torch.stack([
        rtn_dequant_nvfp4(weights[e], group_size=16, levers=(_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()))
        for e in range(E)
    ], dim=0)
    torch.testing.assert_close(out, expected)
    assert float((out - weights).pow(2).mean().item()) > 0.0


def test_scale_sweep_batched_matches_per_linear(seeded):
    """Same equivalence check for the scale-sweep path."""
    E, out_features, in_features = 4, 32, 64
    weights, activations_list = _make_inputs(E, out_features, in_features, T=128)
    reference_weights = weights.clone()

    per_outputs = []
    for e in range(E):
        per_outputs.append(scale_sweep_nvfp4(
            weights[e], activations_list[e],
            reference_weight=reference_weights[e],
            group_size=16,
            levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
        ))
    per_stack = torch.stack(per_outputs, dim=0)

    batch_out = scale_sweep_nvfp4_batched(
        weights, activations_list,
        reference_weights=reference_weights,
        group_size=16, expert_chunk=2,
        levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
    )
    assert torch.allclose(per_stack, batch_out, atol=1e-4, rtol=1e-3), (
        f"max abs diff: {(per_stack - batch_out).abs().max().item():.3e}")


def test_scale_sweep_batched_with_overrides(seeded):
    E, out_features, in_features = 4, 32, 64
    weights, activations_list = _make_inputs(E, out_features, in_features, T=128)
    reference_weights = weights.clone()
    overrides = torch.full((E,), 0.05, dtype=torch.float32)

    per_outputs = []
    for e in range(E):
        per_outputs.append(scale_sweep_nvfp4(
            weights[e], activations_list[e],
            reference_weight=reference_weights[e],
            group_size=16,
            global_real_override=overrides[e],
            levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
        ))
    per_stack = torch.stack(per_outputs, dim=0)

    batch_out = scale_sweep_nvfp4_batched(
        weights, activations_list,
        reference_weights=reference_weights,
        group_size=16,
        global_real_overrides=overrides,
        expert_chunk=2,
        levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
    )
    assert torch.allclose(per_stack, batch_out, atol=1e-4, rtol=1e-3)


def test_batched_paths_match_per_linear_with_four_over_six(seeded):
    E, out_features, in_features = 3, 24, 48
    weights, activations_list = _make_inputs(E, out_features, in_features, T=96)
    reference_weights = weights.clone()

    prev = (_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()).nvfp4_scale_rule
    try:
        _test_exporter._RENDER_LEVERS = replace((_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()), nvfp4_scale_rule="four_over_six_mse")
        per_gptq = torch.stack([
            gptq_obs_rounding_nvfp4(
                weights[e], activations_list[e], group_size=16,
                levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
            )
            for e in range(E)
        ], dim=0)
        batch_gptq = gptq_obs_rounding_nvfp4_batched(
            weights, activations_list, group_size=16, expert_chunk=2,
            levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
        )

        per_sweep = torch.stack([
            scale_sweep_nvfp4(
                weights[e], activations_list[e],
                reference_weight=reference_weights[e],
                group_size=16,
                levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
            )
            for e in range(E)
        ], dim=0)
        batch_sweep = scale_sweep_nvfp4_batched(
            weights, activations_list,
            reference_weights=reference_weights,
            group_size=16,
            expert_chunk=2,
            levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
        )
    finally:
        _test_exporter._RENDER_LEVERS = replace((_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()), nvfp4_scale_rule=prev)

    assert torch.allclose(per_gptq, batch_gptq, atol=1e-4, rtol=1e-3)
    assert torch.allclose(per_sweep, batch_sweep, atol=1e-4, rtol=1e-3)


def test_batched_paths_match_per_linear_with_clip_and_fisher(seeded):
    E, out_features, in_features = 3, 24, 48
    weights, activations_list = _make_inputs(E, out_features, in_features, T=96)
    reference_weights = weights.clone()
    row_weights_list = [
        torch.linspace(0.5, 2.0, activations.shape[0], dtype=torch.float32)
        for activations in activations_list
    ]
    clip_threshold = 0.75

    per_gptq = torch.stack([
        gptq_obs_rounding_nvfp4(
            weights[e],
            activations_list[e],
            group_size=16,
            clip_threshold=clip_threshold,
            fisher_row_weights=row_weights_list[e],
            levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
        )
        for e in range(E)
    ], dim=0)
    batch_gptq = gptq_obs_rounding_nvfp4_batched(
        weights,
        activations_list,
        group_size=16,
        clip_threshold=clip_threshold,
        row_weights_list=row_weights_list,
        expert_chunk=2,
        levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
    )

    per_sweep = torch.stack([
        scale_sweep_nvfp4(
            weights[e],
            activations_list[e],
            reference_weight=reference_weights[e],
            group_size=16,
            clip_threshold=clip_threshold,
            fisher_row_weights=row_weights_list[e],
            levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
        )
        for e in range(E)
    ], dim=0)
    batch_sweep = scale_sweep_nvfp4_batched(
        weights,
        activations_list,
        reference_weights=reference_weights,
        group_size=16,
        clip_threshold=clip_threshold,
        row_weights_list=row_weights_list,
        expert_chunk=2,
        levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
    )

    assert torch.allclose(per_gptq, batch_gptq, atol=1e-4, rtol=1e-3)
    assert torch.allclose(per_sweep, batch_sweep, atol=1e-4, rtol=1e-3)


def test_gptq_batched_matches_per_linear_with_lift_options(seeded):
    E, out_features, in_features = 3, 24, 48
    weights, activations_list = _make_inputs(E, out_features, in_features, T=96)

    prev = (_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()).nvfp4_scale_rule
    try:
        _test_exporter._RENDER_LEVERS = replace((_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()), nvfp4_scale_rule=_native_render_codec.NVFP4_SCALE_RULE_JOINT_MSE)
        per_gptq = torch.stack([
            gptq_obs_rounding_nvfp4(
                weights[e],
                activations_list[e],
                group_size=16,
                static_act_order=True,
                joint_scale_opt=True,
                levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
            )
            for e in range(E)
        ], dim=0)
        batch_gptq = gptq_obs_rounding_nvfp4_batched(
            weights,
            activations_list,
            group_size=16,
            expert_chunk=2,
            static_act_order=True,
            joint_scale_opt=True,
            levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
        )
    finally:
        _test_exporter._RENDER_LEVERS = replace((_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()), nvfp4_scale_rule=prev)

    assert torch.allclose(per_gptq, batch_gptq, atol=1e-5, rtol=1e-5)


def test_gptq_batched_realistic_moe_shape(seeded):
    """Production-scale equivalence: 16 experts × `[3072, 1408]` weight
    (similar to MiniMax down_proj proportions, scaled down). Verifies
    the chunked path holds bitwise equivalence at sizes representative
    of real MoE layers.

    Smaller than the live 256-expert path so the per-Linear baseline
    finishes in test time (CPU ~30 s); the chunked path's correctness
    is independent of scale."""
    E, out_features, in_features = 16, 384, 256
    weights, activations_list = _make_inputs(
        E, out_features, in_features, T=128,
    )

    per_outputs = []
    for e in range(E):
        per_outputs.append(gptq_obs_rounding_nvfp4(
            weights[e], activations_list[e], group_size=16,
            levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
        ))
    per_stack = torch.stack(per_outputs, dim=0)

    batch_out = gptq_obs_rounding_nvfp4_batched(
        weights, activations_list, group_size=16, expert_chunk=8,
        levers=_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve(),
    )
    # Tolerance scales mildly with size due to FP32 reduction-order
    # drift on bigger Cholesky / bmm; 5e-4 is comfortable.
    assert torch.allclose(per_stack, batch_out, atol=5e-4, rtol=1e-3), (
        f"max abs diff: {(per_stack - batch_out).abs().max().item():.3e}")


def test_quantize_2d_nvfp4_group_batched_integration(seeded):
    """End-to-end integration test for the batched-NVFP4 group path
    used by the export hot loop. Builds a small synthetic group of
    same-shape Linears, runs `quantize_2d_nvfp4_group_batched`, and
    verifies that each compressed dict has the expected keys + shapes
    and that values are bit-finite (no NaN/Inf).

    This does NOT compare to per-Linear `quantize_2d` because that
    function reads from the module-level `_CACHED_ACTIVATIONS` /
    `_INPUT_GLOBAL_SCALES` globals; recreating
    that side-channel state in a unit test is more invasive than the
    test itself. The math is already covered by the per-function
    equivalence tests above; this test just exercises the integration
    plumbing."""
    import torch.nn as nn
    from prismaquant.native_render_codec import quantize_2d_nvfp4_group_batched

    E, out_features, in_features = 4, 64, 128

    items = []
    cached = {}
    for e in range(E):
        mod = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            mod.weight.copy_(torch.randn_like(mod.weight) * 0.1)
        recipe_key = f"layer.expert.{e}.weight"
        cached[recipe_key] = torch.randn(64, in_features, dtype=torch.float32)
        items.append((recipe_key, recipe_key, recipe_key, mod))

    # Stub the module-level activation cache for the duration of the
    # call. Restored after.
    saved_cache = _test_exporter._CACHED_ACTIVATIONS
    saved_flags = dict({key: (_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()).record()[key] for key in ("gptq", "scale_sweep", "static_act_order", "joint_scale_opt")})
    saved_input_scales = _test_exporter._INPUT_GLOBAL_SCALES
    try:
        _test_exporter._CACHED_ACTIVATIONS = type("Idx", (), {
            "get": lambda self, name: cached.get(name)})()
        _test_exporter._RENDER_LEVERS = replace((_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()), **{"gptq": True})
        _test_exporter._RENDER_LEVERS = replace((_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()), **{"scale_sweep": False})
        _test_exporter._INPUT_GLOBAL_SCALES = {
            k: 1.0 for k, _, _, _ in items
        }

        compressed_per_linear = quantize_2d_nvfp4_group_batched(
            items, joint_globals={}, device=torch.device("cpu"),
            expert_chunk=2,
         levers=(_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()), activations=_test_exporter._CACHED_ACTIVATIONS, input_global_scales=_test_exporter._INPUT_GLOBAL_SCALES)
    finally:
        _test_exporter._CACHED_ACTIVATIONS = saved_cache
        _test_exporter._RENDER_LEVERS = NativeRenderLevers.resolve()
        _test_exporter._RENDER_LEVERS = replace((_test_exporter._RENDER_LEVERS or NativeRenderLevers.resolve()), **saved_flags)
        _test_exporter._INPUT_GLOBAL_SCALES = saved_input_scales

    assert len(compressed_per_linear) == E
    expected_keys = {
        "weight_packed", "weight_scale",
        "weight_global_scale", "input_global_scale",
    }
    for c in compressed_per_linear:
        assert set(c.keys()) >= expected_keys
        for k, v in c.items():
            # weight_scale is stored as fp8_e4m3fn; isfinite isn't
            # implemented for that dtype, but we can still check the
            # cast-to-float32 view.
            v_f32 = v.float() if v.is_floating_point() else v
            assert torch.isfinite(v_f32).all(), (
                f"non-finite values in {k}")
