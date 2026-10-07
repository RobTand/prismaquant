import torch
import torch.nn as nn

from prismaquant.build_rtn_cache import is_fused_moe_experts
from prismaquant.build_rtn_cache import iter_quantizable_tensors
from prismaquant.build_rtn_cache import _fp8_round
from prismaquant.build_rtn_cache import rtn_fp8_any_shape


class _ToyPackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 32, 32))
        self.down_proj = nn.Parameter(torch.zeros(2, 32, 32))
        self.kernel = nn.Parameter(torch.zeros(2, 32, 32))


class _ToyPackedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.experts = _ToyPackedExperts()


class _NoPackedProfile:
    def packed_expert_param_names(self) -> frozenset[str]:
        return frozenset()


class _CustomPackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.w13 = nn.Parameter(torch.zeros(2, 64, 32))
        self.w2 = nn.Parameter(torch.zeros(2, 32, 32))
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 64, 32))


class _CustomPackedProfile:
    def packed_expert_param_names(self) -> frozenset[str]:
        return frozenset({"w13", "w2"})


class _LegacyContainerProfile:
    def packed_expert_module_class_names(self) -> frozenset[str]:
        return frozenset({"LegacyPackedExperts"})


LegacyPackedExperts = type("LegacyPackedExperts", (nn.Module,), {})


def test_iter_quantizable_tensors_covers_generic_packed_experts():
    model = _ToyPackedModel()

    yielded = list(iter_quantizable_tensors(model))
    names = {name for name, _mod, _attr in yielded}
    attrs = {attr for _name, _mod, attr in yielded}

    assert names == {
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.experts.gate_up_proj",
    }
    assert attrs == {"down_proj", "gate_up_proj"}


def test_iter_quantizable_tensors_respects_profile_packed_names():
    model = nn.Module()
    model.experts = _CustomPackedExperts()

    yielded = list(iter_quantizable_tensors(model, _CustomPackedProfile()))
    names = {name for name, _mod, _attr in yielded}

    assert names == {"experts.w13", "experts.w2"}


def test_iter_quantizable_tensors_allows_profile_to_disable_packed_names():
    model = _ToyPackedModel()

    yielded = list(iter_quantizable_tensors(model, _NoPackedProfile()))

    assert yielded == []


def test_legacy_fused_expert_class_names_are_profile_owned():
    module = LegacyPackedExperts()

    assert is_fused_moe_experts(module, _LegacyContainerProfile())
    assert not is_fused_moe_experts(module)


# ---------------------------------------------------------------------------
# #2352: _fp8_round returned NaN for finite FP16 zero/tiny rows because the
# 1e-8 max-abs floor and the resulting /448 scale underflow FP16. FP16 now
# round-trips through FP32; FP32/BF16 keep the original recipe byte-for-byte.
# ---------------------------------------------------------------------------

_ORD_ROW = [0.35, -1.2, 0.5, -0.25, 2.0, -0.125, 0.0625, 1.0]


def _fp16_subnormal_row() -> torch.Tensor:
    # k * 2**-24 for k = 1..34: exactly representable FP16 subnormals.
    steps = torch.tensor([[1, 2, 3, 5, 8, 13, 21, 34]], dtype=torch.float16)
    return steps * float(2.0 ** -24)


def test_fp8_round_fp16_zero_and_tiny_rows_stay_finite_and_zero():
    zero = torch.zeros(2, 8, dtype=torch.float16)
    tiny = torch.tensor(
        [[0.0, 1e-9, -1e-9, 1e-8, -1e-8, 1e-7, -1e-7, 0.0]],
        dtype=torch.float16,
    )

    out_zero = _fp8_round(zero)
    out_tiny = _fp8_round(tiny)

    assert out_zero.dtype == torch.float16
    assert out_zero.shape == zero.shape
    assert torch.isfinite(out_zero).all()
    # Expected result 0, not a tolerance: for x = 0 the FP32 divide
    # yields exactly 0, E4M3(0) = 0, the dequantize multiply yields
    # exactly 0, and the FP16 cast preserves it.
    assert torch.equal(out_zero, torch.zeros_like(zero))

    assert out_tiny.dtype == torch.float16
    assert out_tiny.shape == tiny.shape
    assert torch.isfinite(out_tiny).all()
    # Elements that are zero in FP16 stay exactly zero; the row maximum is
    # the top of the E4M3 grid and survives the round-trip exactly.
    assert torch.equal(out_tiny[0][tiny[0] == 0], torch.zeros(6, dtype=torch.float16))
    max_pos = tiny.abs().argmax()
    assert out_tiny.flatten()[max_pos] == tiny.flatten()[max_pos]


def test_fp8_round_fp16_subnormal_row_preserves_row_max_within_fp8_grid():
    row = _fp16_subnormal_row()
    sentry = row.clone()

    out = _fp8_round(row)

    assert out.dtype == torch.float16
    assert out.shape == row.shape
    assert torch.isfinite(out).all()
    assert torch.equal(row, sentry)
    # Exact expected outputs, derived from the grids alone. Scale
    # s = (34*2**-24)/448; E4M3FN quotients x/s round to
    # [13, 26, 40, 64, 104, 176, 288, 448]; dequantizing each quotient
    # (q_hat * 34/448, in FP32) and casting to the FP16 subnormal grid
    # (steps of 2**-24) lands on [1, 2, 3, 5, 8, 13, 22, 34] * 2**-24:
    # the k=21 element rounds UP in quotient (276.7 -> 288) and its
    # dequantized 21.86 steps cast to 22.
    expected = torch.tensor(
        [[1, 2, 3, 5, 8, 13, 22, 34]], dtype=torch.float16
    ) * float(2.0 ** -24)
    assert torch.equal(out, expected)


def test_fp8_round_fp16_ordinary_row_reconstruction_contract():
    row = torch.tensor([[_ORD_ROW]], dtype=torch.float16).reshape(1, 8)
    sentry = row.clone()

    out = _fp8_round(row)

    assert out.dtype == torch.float16
    assert out.shape == row.shape
    assert torch.isfinite(out).all()
    assert torch.equal(row, sentry)
    # Exact expected outputs, derived from the grids alone. Max |x| = 2,
    # scale s = 2/448; E4M3FN quotients x/s round to
    # [80, -256, 112, -56, 448, -28, 14, 224]; dequantizing (q_hat * s,
    # in FP32) and casting to FP16 gives [0.357177734375, -1.142578125,
    # 0.5, -0.25, 2, -0.125, 0.0625, 1]: 80*s = 0.3571428... lies
    # between FP16 midpoints 1462.5*2**-12 and 1463.5*2**-12,
    # so it rounds to 1463*2**-12 = 0.357177734375. Likewise,
    # -256*s = -1.142857... rounds to -1170*2**-10 = -1.142578125.
    # The q=448 row max returns bit-exact.
    expected = torch.tensor(
        [[0.357177734375, -1.142578125, 0.5, -0.25, 2.0, -0.125, 0.0625, 1.0]],
        dtype=torch.float16,
    )
    assert torch.equal(out, expected)

    # A strided view of the same values quantizes identically.
    base = torch.tensor(
        [[0.35, 0.9, -1.2, 0.5, 0.5, -0.25, 2.0, -0.125, 0.0625, 1.0, -2.0, 0.3]],
        dtype=torch.float16,
    )
    strided = base[:, ::3]
    assert torch.equal(_fp8_round(strided), _fp8_round(strided.contiguous()))


def test_rtn_fp8_any_shape_fp16_packed_experts_stay_finite_and_row_independent():
    rows = torch.cat(
        [
            torch.zeros(1, 8, dtype=torch.float16),
            _fp16_subnormal_row(),
            torch.tensor([[_ORD_ROW]], dtype=torch.float16).reshape(1, 8),
        ],
        dim=0,
    )
    experts = torch.stack([rows, rows * 0.5 + 0.125])  # (2, 3, 8)
    sentry = experts.clone()

    out = rtn_fp8_any_shape(experts)

    assert out.dtype == torch.float16
    assert out.shape == experts.shape
    assert torch.isfinite(out).all()
    # The all-zero expert stays exactly zero and each row matches the
    # per-row result: the adapter never couples rows across experts.
    assert torch.equal(out[0, 0], torch.zeros(8, dtype=torch.float16))
    for e in range(experts.shape[0]):
        for r in range(experts.shape[1]):
            assert torch.equal(out[e, r], _fp8_round(experts[e, r : r + 1])[0])
    assert torch.equal(experts, sentry)


def test_fp8_round_preserves_fp32_bf16_recipe_bytes():
    # Expected bytes captured from the pre-#2352 owner (dl380g10,
    # torch 2.11.0+cpu, PrismaBuild action b8875e52f5e0d8656cf4105cc9d125
    # b1ff8722920650eb89aa6e58990c893659): the FP16 FP32-intermediate
    # correction must not move FP32 or BF16 outputs by one bit.
    row = torch.tensor([_ORD_ROW], dtype=torch.float32)
    out32 = _fp8_round(row)
    expected32 = torch.tensor(
        [[0.3571428656578064, -1.1428571939468384, 0.5, -0.25,
          2.0, -0.125, 0.0625, 1.0]],
        dtype=torch.float32,
    )
    assert out32.dtype == torch.float32
    assert torch.equal(out32, expected32)

    out_bf = _fp8_round(row.to(torch.bfloat16))
    expected_bf = torch.tensor(
        [[0.35546875, -1.140625, 0.5, -0.25, 2.0, -0.125, 0.0625, 1.0]],
        dtype=torch.bfloat16,
    )
    assert out_bf.dtype == torch.bfloat16
    assert torch.equal(out_bf, expected_bf)

    tiny_bf = torch.tensor(
        [[0.0, 1e-9, -1e-9, 1e-8, -1e-8, 1e-7, -1e-7, 0.0]],
        dtype=torch.bfloat16,
    )
    out_tiny_bf = _fp8_round(tiny_bf)
    expected_tiny_bf = torch.tensor(
        [[0.0, 1.0040821507573128e-09, -1.0040821507573128e-09,
          9.837094694375992e-09, -9.837094694375992e-09,
          1.0011717677116394e-07, -1.0011717677116394e-07, 0.0]],
        dtype=torch.bfloat16,
    )
    assert torch.equal(out_tiny_bf, expected_tiny_bf)
