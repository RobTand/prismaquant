"""Temporary target replay must never consume or mutate the original adjoints."""
import pytest
import torch

from prismaquant.sensitivity_probe import SharedStateCotangents


def accumulated():
    parent = SharedStateCotangents()
    state = parent.graft({'kv': {0: torch.ones(2, 3)}})
    (state['kv'][0] * 3).sum().backward()
    assert parent.harvest() == 1
    return parent


def test_replay_fork_cannot_consume_parent_producer_roots():
    parent = accumulated()
    child = parent.fork_for_replay(max_resident_bytes=24)
    assert sum(t.untyped_storage().nbytes() for t in child.resident_tensors()) == 24
    before = parent.resident_tensors()[0].clone()
    assert child.resident_tensors()[0].data_ptr() != parent.resident_tensors()[0].data_ptr()
    state = child.graft({'kv': {}})
    x = torch.ones(2, 3, requires_grad=True)
    state['kv'][0] = 2 * x
    roots, gradients = child.produced_roots()
    torch.autograd.backward(roots, gradients)
    child.harvest()
    assert torch.equal(x.grad, torch.full_like(x, 6))
    assert child.pending_keys() == []
    assert parent.pending_keys() == [('kv', 0, None)]
    assert torch.equal(parent.resident_tensors()[0], before)


def test_replay_harvest_addition_and_cleanup_do_not_change_parent():
    parent = accumulated()
    child = parent.fork_for_replay(max_resident_bytes=24)
    state = child.graft({'kv': {0: torch.ones(2, 3)}})
    (state['kv'][0] * 5).sum().backward()
    child.harvest()
    assert torch.equal(child.resident_tensors()[0], torch.full((2, 3), 8.))
    assert torch.equal(parent.resident_tensors()[0], torch.full((2, 3), 3.))
    child.release_resident_state()
    assert child.resident_tensors() == ()
    assert parent.pending_keys() == [('kv', 0, None)]
    assert parent.n_harvested == 1


@pytest.mark.parametrize('enabled', [False, True])
def test_empty_replay_fork_preserves_flags_without_diagnostic_aliases(enabled):
    parent = SharedStateCotangents(enabled=enabled)
    parent.nondifferentiable.append('existing diagnostic')
    child = parent.fork_for_replay(max_resident_bytes=0)
    assert child is not parent and child.enabled is enabled
    assert child.resident_tensors() == ()
    child.nondifferentiable.append('replay only')
    assert parent.nondifferentiable == ['existing diagnostic']


@pytest.mark.parametrize('live_field', ['_live', '_containers', '_live_ids'])
def test_replay_refuses_each_nonquiescent_owner(live_field):
    parent = accumulated()
    getattr(parent, live_field).add(1) if live_field == '_live_ids' else getattr(parent, live_field).append(None)
    with pytest.raises(RuntimeError, match='quiescent'):
        parent.fork_for_replay(max_resident_bytes=24)


def test_replay_cap_refuses_before_any_accumulator_clone(monkeypatch):
    parent = accumulated()
    original_clone = torch.Tensor.clone
    def forbidden(tensor, *args, **kwargs):
        pytest.fail('over-budget replay allocated before refusing')
    monkeypatch.setattr(torch.Tensor, 'clone', forbidden)
    with pytest.raises(RuntimeError, match='residency'):
        parent.fork_for_replay(max_resident_bytes=23)
    monkeypatch.setattr(torch.Tensor, 'clone', original_clone)
    assert torch.equal(parent.resident_tensors()[0], torch.full((2, 3), 3.))
