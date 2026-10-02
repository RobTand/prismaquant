"""CPU control tests for ordered private preparation; no CUDA execution."""
import inspect
import threading
from types import SimpleNamespace

import pytest
import torch

from prismaquant import tessera_campaign as campaign, layer_streaming


class Live:
    def __init__(self, value):
        self.value = value
        self.shape, self.dtype = value.shape, value.dtype
        self.device = torch.device('cuda:0')
    def detach(self):
        return self
    def numel(self):
        return self.value.numel()
    def element_size(self):
        return self.value.element_size()


@pytest.fixture
def cpu_transport(monkeypatch):
    """Use real CPU copies and explicit completion tokens for transport controls."""
    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '2')
    monkeypatch.setattr(layer_streaming, '_LAYER_READ_POOL', None)
    monkeypatch.setattr(layer_streaming, '_LAYER_READ_POOL_THREADS', 0)
    original_empty, original_to = torch.empty_like, torch.Tensor.to
    monkeypatch.setattr(torch, 'empty_like', lambda value, **kw: original_empty(
        value, **{k:v for k,v in kw.items() if k != 'pin_memory'}))
    def transfer(value, *args, **kwargs):
        if args and isinstance(args[0], torch.device) and args[0].type == 'cuda':
            return value.clone()
        return original_to(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', transfer)
    monkeypatch.setattr(campaign, '_device_differs', lambda live, staged:
                        torch.ne(live.value, staged).any())
    events = []
    class Event:
        def __init__(self):
            self.done = threading.Event(); self.done.set(); events.append(self)
        def record(self, stream):
            pass
        def query(self):
            return self.done.is_set()
        def synchronize(self):
            assert self.done.wait(3), 'completion token never resolved'
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device=None: object())
    synchronized = []
    def synchronize(device=None):
        synchronized.append(device)
        for event in events:
            event.done.set()
    monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
    yield SimpleNamespace(events=events, synchronized=synchronized)
    pool = layer_streaming._LAYER_READ_POOL
    if pool is not None:
        pool.shutdown(wait=True)


def run_check(monkeypatch, values, *, read=None, guard=None, max_bytes=64):
    bound = {'s': {f'u{i}':dict(source_tensor=f'w{i}', rows=2, cols=3)
                   for i in range(len(values))}}
    auth = object()
    if read is None:
        def read(name, unit, **kwargs):
            assert kwargs['source_authentication'] is auth
            return values[int(name[1:])].clone(), lambda:None
    monkeypatch.setattr(campaign, '_read_projected_unit', read)
    kwargs = dict(weights={f'u{i}':Live(v) for i,v in enumerate(values)},
        model_path='unused',source={},source_authentication=auth,
        resource_check=guard or (lambda label, **kw:None))
    # The original serial implementation is the causal RED control.
    if 'parallel_preparation' in inspect.signature(campaign._checked_projected_units).parameters:
        kwargs.update(parallel_preparation=True, preparation_max_bytes=max_bytes)
    return campaign._checked_projected_units(bound, **kwargs)


def test_preparation_reaches_second_copy_before_first_can_finish(monkeypatch, cpu_transport):
    second = threading.Event()
    original = torch.Tensor.copy_
    started = []
    def copy(target, source, *args, **kwargs):
        tag = int(source[0,0])
        started.append(tag)
        if tag == 0:
            assert second.wait(1), 'serial preparation cannot start the second copy'
        if tag == 1:
            second.set()
        return original(target, source, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'copy_', copy)
    values = [torch.full((2,3),i,dtype=torch.bfloat16) for i in range(2)]
    assert list(run_check(monkeypatch,values)) == ['u0','u1']
    assert set(started) == {0,1}
