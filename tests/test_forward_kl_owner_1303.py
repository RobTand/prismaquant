"""Remaining KL consumers preserve their local reductions under one owner.

The old per-token recipes are independent oracles. Tests exercise each real
consumer, including weight restoration and replay pairing, with CPU operands.
The tiny fork runner supplies deterministic boundaries without source/GPU IO.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from prismaquant import expert_empirical_cost as ec
from prismaquant.kl_fisher import forward_kl_per_token


def _raw(value):
    return value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _old_tokens(student, teacher):
    return (teacher.exp() * (teacher - student)).sum(-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 3, 7), (2, 0, 7), (2, 3, 0)])
def test_owner_retains_raw_bits_for_strided_broadcast_and_empty_operands(dtype, shape):
    n = shape[0] * shape[1] * shape[2]
    student = ((torch.arange(n * 2) % 19 - 9) / 8).reshape(*shape[:-1], shape[-1] * 2)[..., ::2].to(dtype)
    teacher = student[:1].flip(-1)
    expected = _old_tokens(student, teacher)
    actual = forward_kl_per_token(student, teacher)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert _raw(actual) == _raw(expected)


def test_owner_retains_nonfinite_outcomes_and_native_shape_refusal():
    teacher = torch.tensor([[0., -float("inf"), float("nan")]])
    student = torch.tensor([[-1., -float("inf"), -2.]])
    assert _raw(forward_kl_per_token(student, teacher)) == _raw(_old_tokens(student, teacher))
    with pytest.raises(RuntimeError) as old:
        _old_tokens(torch.zeros(2, 3), torch.zeros(2, 4))
    with pytest.raises(RuntimeError) as new:
        forward_kl_per_token(torch.zeros(2, 3), torch.zeros(2, 4))
    assert str(new.value) == str(old.value)


def _record_owner(monkeypatch, module):
    calls = []

    def record(student, teacher):
        expected = _old_tokens(student, teacher)
        actual = forward_kl_per_token(student, teacher)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert _raw(actual) == _raw(expected)
        calls.append(actual.detach().clone())
        return actual

    monkeypatch.setattr(module, "forward_kl_per_token", record, raising=False)
    return calls


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("unpacked", [False, True])
def test_expert_consumers_route_the_metric_and_restore_weights(monkeypatch, batch_size, unpacked):
    from test_expert_empirical_cost import TinyCausal, _Dsv4PerExpertCausal
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

    monkeypatch.setenv("PRISMAQUANT_EXPERT_CALIB_BATCH", str(batch_size))
    torch.manual_seed(1303)
    model = (_Dsv4PerExpertCausal() if unpacked else TinyCausal()).eval()
    profile = DeepseekV4Profile() if unpacked else None
    calib = (torch.arange(3 * 5).reshape(3, 5) * 7) % 32
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    calls = _record_owner(monkeypatch, ec)
    _, costs, units = ec.measure_expert_unit_costs(
        model, profile, calib, ["NVFP4", "BF16"], expert_chunk=1, progress=False,
    )
    assert len(calls) == (3 + batch_size - 1) // batch_size
    expected = sum(float(tokens.sum().item()) for tokens in calls) / sum(tokens.numel() for tokens in calls)
    (unit,) = units.values()
    assert unit["NVFP4"] == expected
    assert sum(row["NVFP4"]["predicted_dloss"] for row in costs.values()) == pytest.approx(expected)
    for name, param in model.named_parameters():
        assert _raw(param) == _raw(before[name])


@pytest.mark.parametrize("unpacked", [False, True])
def test_expert_consumer_restores_weights_when_metric_refuses(monkeypatch, unpacked):
    from test_expert_empirical_cost import TinyCausal, _Dsv4PerExpertCausal
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

    model = (_Dsv4PerExpertCausal() if unpacked else TinyCausal()).eval()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    def refuse(*args):
        raise RuntimeError("metric owner refused")

    monkeypatch.setattr(ec, "forward_kl_per_token", refuse, raising=False)
    with pytest.raises(RuntimeError, match="metric owner refused"):
        ec.measure_expert_unit_costs(
            model, DeepseekV4Profile() if unpacked else None,
            torch.arange(10).reshape(2, 5), ["NVFP4"], progress=False,
        )
    for name, param in model.named_parameters():
        assert _raw(param) == _raw(before[name])


def test_forked_expert_consumer_routes_each_window_and_keeps_token_mean(monkeypatch):
    baseline = (torch.arange(3 * 4 * 7).reshape(3, 4, 7) % 17 - 8).float() / 4
    student = baseline.flip(-1) + torch.arange(7).float() / 8
    qname = "model.layers.0.mlp.experts"
    events = []
    context = SimpleNamespace(
        schedule_prefetch=lambda layer: events.append(("prefetch", layer)),
        install=lambda layer, **kwargs: events.append(("install", layer)),
        unload=lambda layer: events.append(("unload", layer)),
    )
    batch = SimpleNamespace(activations_cpu=[baseline, baseline])
    runner = SimpleNamespace(
        model=torch.nn.Module(), device=torch.device("cpu"), num_layers=1,
        prefetch_lookahead=1, require_prefetched_residency=True, context=context,
        layer_index_for_qname=lambda name: 0, capture_boundaries=lambda ids: batch,
        tail_logits=lambda batch, hidden: hidden,
    )
    profile = SimpleNamespace(new_forward_pass_state=lambda: {})
    monkeypatch.setattr(ec, "resolve_routed_expert_profile", lambda model, profile: profile)
    monkeypatch.setattr(ec, "_streamed_expert_unit_records", lambda *a, **k: (
        [("packed", qname, None)], []))

    def fork(*args, rows_meta, **kwargs):
        rows_meta[qname] = {"n_params_unit": 6, "members": [
            (qname + ".weight", 6, 3, 2, {})]}
        return student

    monkeypatch.setattr(ec, "_fork_quantized_unit", fork)
    calls = _record_owner(monkeypatch, ec)
    _, costs, units = ec.measure_expert_unit_costs_forked(
        runner, profile, torch.zeros(3, 4, dtype=torch.int64),
        ["NVFP4", "BF16"], progress=False,
    )
    assert len(calls) == 3
    expected = sum(float(tokens.sum().item()) for tokens in calls) / sum(tokens.numel() for tokens in calls)
    assert units[qname]["NVFP4"] == expected
    assert costs[qname + ".weight"]["NVFP4"]["predicted_dloss"] == expected
    assert events.count(("install", 0)) == events.count(("unload", 0)) == 1
