"""Diagnostics reduce complete signed FP32 operators, without leaf gradients."""
import weakref

import pytest
import torch

from test_joint_operator_statistics import _fixture, _lease


def test_diagnostics_match_complete_operator_and_do_not_own_it():
    layer, weight, specs, draws = _fixture(2)
    layer.weight.requires_grad_(False)
    expected = torch.zeros_like(weight)
    with _lease({'u': layer}, {'u': specs}) as lease:
        lease.begin_probe()
        for x, g in draws:
            layer(x.detach().requires_grad_(True)).backward(g)
            expected.add_(g.T @ x)
        assert layer.weight.grad is None
        lease.finish_observations()
        refs = [weakref.ref(tensor) for tensor in lease._operators.values()]
        result = lease.operator_diagnostics(collect_col_energy=True)
        assert result['u']['g_trace'] == float(expected.square().sum())
        torch.testing.assert_close(result['u']['col_energy'], expected.square().sum(0), rtol=0, atol=0)
        assert result['u']['col_energy'].device.type == 'cpu'
        assert not result['u']['col_energy'].requires_grad
        lease.project({('u', fmt): torch.zeros_like(weight) for fmt in specs})
        lease.finish_projections()
        assert all(ref() is None for ref in refs)
        assert result['u']['col_energy'].shape == (3,)


def test_cancelling_invocations_are_summed_before_diagnostic_square():
    layer, _, specs, draws = _fixture(1)
    x, g = draws[0]
    with _lease({'u': layer}, {'u': specs}) as lease:
        lease.begin_probe()
        layer(x).backward(g)
        layer(x).backward(-g)
        lease.finish_observations()
        result = lease.operator_diagnostics(collect_col_energy=True)['u']
        assert result['g_trace'] == 0
        assert torch.equal(result['col_energy'], torch.zeros(3))


def test_unrouted_diagnostics_are_explicit_zero_and_columns_are_optional():
    layer, _, specs, _ = _fixture(1)
    with _lease({'u': layer}, {'u': specs}) as lease:
        with pytest.raises(RuntimeError, match='ready'):
            lease.operator_diagnostics(collect_col_energy=False)
        lease.begin_probe()
        lease.finish_observations()
        assert lease.operator_diagnostics(collect_col_energy=False) == {'u': {'g_trace': 0.0}}
        result = lease.operator_diagnostics(collect_col_energy=True)
        assert torch.equal(result['u']['col_energy'], torch.zeros(3))
    with pytest.raises(RuntimeError, match='ready'):
        lease.operator_diagnostics(collect_col_energy=False)


def test_nonfinite_operator_and_invalid_request_refuse():
    layer, _, specs, draws = _fixture(1)
    with _lease({'u': layer}, {'u': specs}) as lease:
        lease.begin_probe()
        layer(draws[0][0]).backward(draws[0][1])
        lease.finish_observations()
        with pytest.raises(ValueError, match='boolean'):
            lease.operator_diagnostics(collect_col_energy=1)
        lease._operators['u', None].fill_(float('inf'))
        with pytest.raises(RuntimeError, match='nonfinite'):
            lease.operator_diagnostics(collect_col_energy=True)
