"""Independent residual and ownership gates for deferred joint contractions."""
import gc
import weakref

import pytest
import torch

from prismaquant import joint_aura
from test_joint_aura_projection import _assert_projection, _linear, _oracle, _spec


def _lease(modules, specs, budget=100000, candidate_budget=100000):
    return joint_aura.JointOperatorStatisticsLease(
        modules, specs, max_statistics_bytes=budget, max_candidate_bytes=candidate_budget)


def _fixture(candidates=12):
    generator = torch.Generator().manual_seed(1912)
    weight = torch.randn(4, 3, generator=generator)
    layer = _linear(weight)
    qdq = lambda x: torch.round(x * 2) / 2
    specs = {f'q{i}': _spec(f'q{i}', qdq) for i in range(candidates)}
    specs['identity'] = _spec('identity', lambda x: x, act_bits=None)
    draws = [(torch.randn(2, 3, generator=generator),
              torch.randn(2, 4, generator=generator)) for _ in range(3)]
    return layer, weight, specs, draws


def test_many_candidates_reuse_two_operator_planes_and_match_fp64_residuals():
    layer, weight, specs, draws = _fixture()
    before = layer.weight.detach().clone()
    # 12 candidate deltas would use 576 bytes. One GW plus shared GA uses 96.
    with _lease({'u': layer}, {'u': specs}, budget=96) as lease:
        assert lease.statistics_capacity_bytes == 96
        lease.begin_probe()
        for x, g in draws:
            layer(x).backward(g)
        assert lease.resident_statistics_bytes == 96
        lease.finish_observations()
        assert lease.telemetry['qdq_calls'] == len(draws)
        for index, (fmt, spec) in enumerate(reversed(list(specs.items()))):
            delta = torch.full_like(weight, (index + 1) / 128)
            reference = {key: 0. for key in ('weight', 'activation', 'mixed', 'total')}
            for x, g in draws:
                terms = _oracle(weight, delta, x,
                    spec.activation_quantize_dequantize(x), g)
                for key, value in terms.items():
                    reference[key] += value
            actual = lease.project({('u', fmt): delta})
            _assert_projection(actual[('u', fmt)], reference)
            ref = weakref.ref(delta)
            del delta
            assert ref() is None, 'candidate deltas must stay caller-owned'
        results = lease.finish_projections()
        assert len(results) == len(specs)
        assert lease.resident_statistics_bytes == 0
    torch.testing.assert_close(layer.weight, before, rtol=0, atol=0)


def test_budget_is_enforced_before_observers_or_matrix_allocation():
    layer, _, specs, _ = _fixture()
    with pytest.raises(RuntimeError, match='statistics.*budget'):
        _lease({'u': layer}, {'u': specs}, budget=95)
    assert not layer._forward_hooks


def test_observation_release_drops_source_owners_before_candidate_projection():
    layer, _, specs, draws = _fixture(2)
    lease = _lease({'u': layer}, {'u': specs})
    with lease:
        lease.begin_probe()
        x, g = draws[0]
        layer(x).backward(g)
        ref = weakref.ref(layer.weight)
        lease.finish_observations()
        assert not layer._forward_hooks and not lease.modules
        del layer
        gc.collect()
        assert ref() is None
        lease.project({('u', fmt): torch.zeros(4, 3) for fmt in specs})
        lease.finish_projections()


def test_incomplete_duplicate_unknown_and_late_projection_refuse():
    layer, _, specs, _ = _fixture(1)
    with _lease({'u': layer}, {'u': specs}) as lease:
        with pytest.raises(RuntimeError, match='ready'):
            lease.project({('u', 'q0'): torch.zeros(4, 3)})
        lease.begin_probe()
        lease.finish_observations()  # never-routed operators produce explicit zeros
        with pytest.raises(RuntimeError, match='coverage'):
            lease.finish_projections()
        with pytest.raises(ValueError, match='unknown'):
            lease.project({('u', 'bogus'): torch.zeros(4, 3)})
        lease.project({('u', 'q0'): torch.zeros(4, 3)})
        with pytest.raises(ValueError, match='duplicate'):
            lease.project({('u', 'q0'): torch.zeros(4, 3)})
        lease.project({('u', 'identity'): torch.zeros(4, 3)})
        assert all(row['total'] == 0 for row in lease.finish_projections().values())
        with pytest.raises(RuntimeError, match='ready'):
            lease.project({('u', 'identity'): torch.zeros(4, 3)})


def test_unconsumed_backward_refuses_observation_seal_and_cleans_up():
    layer, _, specs, draws = _fixture(1)
    lease = _lease({'u': layer}, {'u': specs})
    with pytest.raises(RuntimeError, match='pending backward'):
        with lease:
            lease.begin_probe()
            output = layer(draws[0][0])
            lease.finish_observations()
    assert not layer._forward_hooks and lease.resident_statistics_bytes == 0
    with pytest.raises(RuntimeError, match='active'):
        output.sum().backward()


@pytest.mark.parametrize('when', ['before_forward', 'before_seal'])
def test_changed_source_weights_are_refused(when):
    layer, _, specs, draws = _fixture(1)
    with _lease({'u': layer}, {'u': specs}) as lease:
        lease.begin_probe()
        if when == 'before_seal':
            layer(draws[0][0]).backward(draws[0][1])
        with torch.no_grad():
            layer.weight.add_(1)
        with pytest.raises(RuntimeError, match='source.*changed'):
            if when == 'before_forward':
                layer(draws[0][0])
            else:
                lease.finish_observations()


def test_cancellation_preserves_three_signed_components_and_new_identity():
    weight, delta = torch.tensor([[0., -1.]]), torch.tensor([[1., 0.]])
    layer = _linear(weight)
    spec = _spec('cancel', lambda x: x + torch.tensor([0., 1.]))
    with _lease({'u': layer}, {'u': {'cancel': spec}}) as lease:
        lease.begin_probe()
        layer(torch.tensor([[1., 0.]])).sum().backward()
        lease.finish_observations()
        lease.project({('u', 'cancel'): delta})
        assert lease.finish_projections()[('u', 'cancel')] == {
            'weight': 1., 'activation': -1., 'mixed': 0., 'total': 0.}
        identity = lease.arithmetic_identity(torch.float32)
        assert identity != joint_aura.arithmetic_identity(torch.float32)
        assert identity['operator_accumulation'] == 'sum_fp32_matrices_in_backward_invocation_order'


def test_bad_candidate_quantum_is_atomic_and_never_retained():
    layer, _, specs, draws = _fixture(1)
    with _lease({'u': layer}, {'u': specs}) as lease:
        lease.begin_probe()
        layer(draws[0][0]).backward(draws[0][1])
        lease.finish_observations()
        with pytest.raises(RuntimeError, match='shape'):
            lease.project({('u', 'q0'): torch.zeros(4, 3), ('u', 'identity'): torch.zeros(3, 4)})
        result = lease.project({('u', fmt): torch.zeros(4, 3) for fmt in specs})
        assert len(result) == 2
        lease.finish_projections()


def test_distinct_dynamic_qdq_callables_are_charged_separately():
    layer = _linear(torch.ones(4, 3))
    def quantizer(offset):
        return lambda x: x + offset
    specs = {'a': _spec('a', quantizer(1)), 'b': _spec('b', quantizer(2))}
    # Both closures have the same named identity but different dynamic owners.
    with _lease({'u': layer}, {'u': specs}, budget=144) as lease:
        assert lease.statistics_capacity_bytes == 144
        lease.begin_probe()
        layer(torch.zeros(2, 3)).sum().backward()
        lease.finish_observations()
        lease.project({('u', fmt): torch.ones(4, 3) for fmt in specs})
        result = lease.finish_projections()
        assert result[('u', 'b')]['activation'] == 2 * result[('u', 'a')]['activation']


def test_invalid_qdq_leaves_no_matrix_owner_after_context_failure():
    layer = _linear(torch.ones(4, 3))
    spec = _spec('bad', lambda x: x.double())
    lease = _lease({'u': layer}, {'u': {'bad': spec}})
    with pytest.raises(RuntimeError, match='QDQ changed'):
        with lease:
            lease.begin_probe()
            layer(torch.ones(2, 3)).sum().backward()
    assert not layer._forward_hooks and lease.resident_statistics_bytes == 0
    with pytest.raises(RuntimeError, match='reenter'):
        lease.__enter__()


def test_second_backward_on_same_observation_refuses_double_counting():
    layer = _linear(torch.ones(4, 3))
    spec = _spec('id', lambda x: x, act_bits=None)
    with _lease({'u': layer}, {'u': {'id': spec}}) as lease:
        lease.begin_probe()
        output = layer(torch.ones(2, 3))
        output.sum().backward(retain_graph=True)
        with pytest.raises(RuntimeError, match='active observation'):
            output.sum().backward()


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_candidate_projection_cannot_publish_partial_quantum(bad):
    layer, _, specs, draws = _fixture(1)
    with _lease({'u': layer}, {'u': specs}) as lease:
        lease.begin_probe()
        layer(draws[0][0]).backward(draws[0][1])
        lease.finish_observations()
        with pytest.raises(RuntimeError, match='nonfinite'):
            lease.project({('u', 'q0'): torch.zeros(4, 3), ('u', 'identity'): torch.full((4, 3), bad)})
        lease.project({('u', fmt): torch.zeros(4, 3) for fmt in specs})
        lease.finish_projections()


def test_packed_source_slices_and_unrouted_experts_match_independent_outputs():
    from test_joint_aura_packed import _fixture as packed_fixture

    model, _, _, _, _, views = packed_fixture()
    layer = model.model.layers[0]
    selected = {member.qname: member for member in views
                if member.qname.startswith('model.layers.0.')}
    spec = _spec('synthetic', lambda x: torch.round(x * 8) / 8)
    specs = {name: {'synthetic': spec} for name in selected}
    source = {name: member.weight.detach().clone() for name, member in selected.items()}
    delta = {name: torch.full_like(weight, .03125) for name, weight in source.items()}
    layer.feed_forward.experts.captures = []
    expected = {name: {key: 0. for key in ('weight', 'activation', 'mixed', 'total')}
                for name in selected}
    with _lease(selected, specs) as lease:
        lease.begin_probe()
        x = torch.randn(1, 3, 16, requires_grad=True)
        layer(x).sum().backward()
        for expert, inputs, gate_up, hidden, down in layer.feed_forward.experts.captures:
            for role, value, gradient in [('w1', inputs, gate_up.grad[:, :16]),
                                          ('w3', inputs, gate_up.grad[:, 16:]),
                                          ('w2', hidden, down.grad)]:
                name = f'model.layers.0.feed_forward.experts.{expert}.{role}'
                terms = _oracle(source[name], delta[name], value,
                    spec.activation_quantize_dequantize(value), gradient)
                for key, number in terms.items():
                    expected[name][key] += number
        lease.finish_observations()
        assert lease.resident_statistics_bytes < lease.statistics_capacity_bytes
        for name in reversed(list(selected)):
            lease.project({(name, 'synthetic'): delta[name]})
        result = lease.finish_projections()
        for name, reference in expected.items():
            _assert_projection(result[(name, 'synthetic')], reference)
            if '.experts.2.' in name:
                assert result[(name, 'synthetic')] == {key: 0. for key in reference}


def test_candidate_budget_charges_full_storage_behind_small_views():
    layer, _, specs, _ = _fixture(1)
    with _lease({'u': layer}, {'u': specs}, candidate_budget=48) as lease:
        lease.begin_probe()
        lease.finish_observations()
        pool = torch.zeros(20, 4, 3)
        with pytest.raises(RuntimeError, match='candidate storage.*budget'):
            lease.project({('u', 'q0'): pool[0]})
        with pytest.raises(RuntimeError, match='candidate storage.*budget'):
            lease.project({('u', fmt): torch.zeros(4, 3) for fmt in specs})
        for fmt in specs:
            lease.project({('u', fmt): torch.zeros(4, 3)})
        lease.finish_projections()
        assert lease.telemetry['peak_candidate_storage_bytes'] == 48


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native operator statistics needs CUDA')
@pytest.mark.parametrize('shape', [(2048, 4096), (4096, 2048)])
def test_native_glm_projection_shapes_match_independent_fp64_outputs(shape):
    # Synthetic dW and QDQ isolate contraction arithmetic at actual GLM expert
    # dimensions; they are not trained-model quality or format qualification.
    rows, columns = shape
    generator = torch.Generator().manual_seed(1907)
    weight = (torch.randn(shape, generator=generator) / columns**.5).to('cuda', torch.bfloat16)
    delta = torch.full(shape, .0009765625, device='cuda')
    module = torch.nn.Linear(columns, rows, bias=False, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(weight)
    spec = _spec('synthetic', lambda x: torch.round(x * 8) / 8)
    expected = {key: 0. for key in ('weight', 'activation', 'mixed', 'total')}
    with _lease({'u': module}, {'u': {'synthetic': spec}},
                budget=rows * columns * 8, candidate_budget=rows * columns * 4) as lease:
        lease.begin_probe()
        for count in (1, 3):
            x = torch.randn(count, columns, generator=generator).to('cuda', torch.bfloat16)
            g = torch.randn(count, rows, generator=generator).to('cuda', torch.bfloat16)
            module(x).backward(g)
            terms = _oracle(weight, delta, x, spec.activation_quantize_dequantize(x), g)
            for key, value in terms.items():
                expected[key] += value
            module.zero_grad(set_to_none=True)
        lease.finish_observations()
        actual = lease.project({('u', 'synthetic'): delta})[('u', 'synthetic')]
        _assert_projection(actual, expected)
        lease.finish_projections()
