import unittest
import types
import sys

import torch
import torch.nn.functional as F

vllm = types.ModuleType("vllm")
vllm_logger = types.ModuleType("vllm.logger")
vllm_logger.init_logger = lambda name: types.SimpleNamespace(debug=lambda *a, **k: None)
fmb = types.ModuleType("vllm.model_executor.layers.fused_moe.fused_moe_method_base")
fmb.FusedMoEMethodBase = object
cfg = types.ModuleType("vllm.model_executor.layers.fused_moe.config")
cfg.FusedMoEConfig = object
utils = types.ModuleType("vllm.model_executor.utils")
utils.set_weight_attrs = lambda param, attrs: None
sys.modules.setdefault("vllm", vllm)
sys.modules["vllm.logger"] = vllm_logger
sys.modules["vllm.model_executor.layers.fused_moe.fused_moe_method_base"] = fmb
sys.modules["vllm.model_executor.layers.fused_moe.config"] = cfg
sys.modules["vllm.model_executor.utils"] = utils

from prismquant_pkg.prismquant import prismquant_moe


class _DummyLayer:
    pass


def _make_layer(grouped_bits: bool = True):
    layer = _DummyLayer()
    layer.prismquant_num_experts = 3
    layer.prismquant_hidden_size = 4
    layer.prismquant_intermediate_size = 3
    layer.prismquant_group_size_h = 2
    layer.prismquant_group_size_i = 1
    device = "cpu"

    # Expert 0/1 share a quantization signature, expert 2 differs when
    # grouped_bits=False so we can verify bucketing behavior.
    e2_bits = 4 if grouped_bits else 5
    layer.prismquant_w13_table = [
        (0, 2, 2, 2, 4, 4, 3, 4),
        (4, 2, 6, 2, 4, 4, 3, 4),
        (8, 2, 10, 2, e2_bits, e2_bits, 3, 4),
    ]
    layer.prismquant_w2_table = [
        (0, 2, 4, 4, 3),
        (2, 2, 4, 4, 3),
        (4, 2, e2_bits, 4, 3),
    ]
    layer.w13_weight_packed = torch.nn.Parameter(
        torch.tensor([10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33], dtype=torch.uint8),
        requires_grad=False,
    )
    layer.w2_weight_packed = torch.nn.Parameter(
        torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.uint8),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones(3, 6, 2, dtype=torch.bfloat16, device=device),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.ones(3, 4, 3, dtype=torch.bfloat16, device=device),
        requires_grad=False,
    )
    prismquant_moe._build_expert_groups(layer)
    return layer


def _fake_fused_linear(x, packed, scales, n_bits, out_features, in_features, group_size):
    base = x.float().sum(dim=1, keepdim=True)
    packed_term = float(packed.reshape(-1)[0].item()) if packed.numel() else 0.0
    scale_term = float(scales.reshape(-1)[0].item()) if scales.numel() else 0.0
    factor = packed_term + scale_term + float(n_bits) + float(group_size)
    return base.repeat(1, out_features) * factor


def _legacy_apply(layer, x, topk_weights, topk_ids):
    x_f32 = x.float()
    out = torch.zeros(x.shape[0], layer.prismquant_hidden_size, dtype=x.dtype)
    for eidx in range(layer.prismquant_num_experts):
        mask = topk_ids.eq(eidx)
        expert_weights = (topk_weights * mask.to(topk_weights.dtype)).sum(dim=1)
        active = torch.nonzero(expert_weights.ne(0), as_tuple=False).flatten()
        if active.numel() == 0:
            continue
        xa = x_f32.index_select(0, active)
        ew = expert_weights.index_select(0, active).unsqueeze(1)

        g_off, g_sz, u_off, u_sz, g_bits, u_bits, out_f, in_f = layer.prismquant_w13_table[eidx]
        gate = _fake_fused_linear(
            xa,
            layer.w13_weight_packed[g_off:g_off + g_sz + 4],
            layer.w13_weight_scale[eidx, :out_f],
            g_bits,
            out_f,
            in_f,
            group_size=layer.prismquant_group_size_h,
        )
        up = _fake_fused_linear(
            xa,
            layer.w13_weight_packed[u_off:u_off + u_sz + 4],
            layer.w13_weight_scale[eidx, out_f:2 * out_f],
            u_bits,
            out_f,
            in_f,
            group_size=layer.prismquant_group_size_h,
        )
        h = F.silu(gate) * up

        d_off, d_sz, d_bits, d_out, d_in = layer.prismquant_w2_table[eidx]
        down = _fake_fused_linear(
            h.float(),
            layer.w2_weight_packed[d_off:d_off + d_sz + 4],
            layer.w2_weight_scale[eidx, :d_out],
            d_bits,
            d_out,
            d_in,
            group_size=layer.prismquant_group_size_i,
        )
        out.index_add_(0, active, ew.to(x.dtype) * down.to(x.dtype))
    return out


class TestDynaquantMoEGrouped(unittest.TestCase):
    def test_build_expert_groups_buckets_by_quant_signature(self):
        layer = _make_layer(grouped_bits=False)
        groups = layer.prismquant_expert_groups
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["experts"], (0, 1))
        self.assertEqual(groups[1]["experts"], (2,))

    def test_grouped_eager_matches_legacy(self):
        layer = _make_layer(grouped_bits=True)
        method = prismquant_moe.PrismQuantFusedMoEMethod.__new__(
            prismquant_moe.PrismQuantFusedMoEMethod
        )
        original = prismquant_moe._get_fused_linear
        prismquant_moe._get_fused_linear = lambda: _fake_fused_linear
        try:
            x = torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [0.5, 0.25, 0.75, 1.25],
                    [2.0, 1.0, 0.0, 3.0],
                ],
                dtype=torch.float32,
            )
            topk_ids = torch.tensor([[0, 2], [1, 0], [2, 1]], dtype=torch.int64)
            topk_weights = torch.tensor(
                [[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]], dtype=torch.float32
            )

            got = method._apply_grouped_eager(
                layer, x.float(), x.dtype, topk_weights, topk_ids
            )
            want = _legacy_apply(layer, x, topk_weights, topk_ids)
            self.assertTrue(torch.allclose(got, want))
        finally:
            prismquant_moe._get_fused_linear = original


if __name__ == "__main__":
    unittest.main()
