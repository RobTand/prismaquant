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


def run_check(monkeypatch, values, *, read=None, guard=None, max_bytes=64, cancel=None):
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
        kwargs.update(parallel_preparation=True, preparation_max_bytes=max_bytes, cancel=cancel)
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


def test_reverse_preparation_completion_preserves_first_mismatch(monkeypatch, cpu_transport):
    second = threading.Event()
    values = [torch.full((2,3),i,dtype=torch.bfloat16) for i in range(2)]
    def read(name, unit, **kwargs):
        index = int(name[1:])
        if index == 0:
            assert second.wait(2)
        else:
            second.set()
        changed = values[index].clone()
        changed[0,0] += 1
        return changed, lambda:None
    with pytest.raises(RuntimeError) as caught:
        run_check(monkeypatch, values, read=read)
    message = str(caught.value)
    assert message.index('u0 (live') < message.index('u1 (live')


def _foreground(callback):
    done = threading.Event()
    result = []
    def execute():
        try:
            result.append(callback())
        except BaseException as error:
            result.append(error)
        finally:
            done.set()
    thread = threading.Thread(target=execute)
    thread.start()
    return thread, done, result


def test_four_credits_retain_pins_until_completion_before_fifth_read(monkeypatch, cpu_transport):
    import weakref
    pins, reads = [], []
    allow_new = threading.Event()
    fourth = threading.Event()
    original_empty, original_event = torch.empty_like, torch.cuda.Event
    def empty(value, **kwargs):
        tensor = original_empty(value, **kwargs)
        pins.append(weakref.ref(tensor))
        return tensor
    class DelayedEvent(original_event):
        def __init__(self):
            super().__init__()
            if not allow_new.is_set():
                self.done.clear()
        def record(self, stream):
            if len(cpu_transport.events) == 4:
                fourth.set()
    monkeypatch.setattr(torch, 'empty_like', empty)
    monkeypatch.setattr(torch.cuda, 'Event', DelayedEvent)
    values = [torch.full((2,3),i,dtype=torch.bfloat16) for i in range(8)]
    def read(name, unit, **kwargs):
        reads.append(name)
        return values[int(name[1:])].clone(), lambda:None
    reserves = []
    def guard(label, **kwargs):
        if kwargs:
            reserves.append(kwargs)
    thread, done, result = _foreground(lambda:run_check(monkeypatch,values,read=read,guard=guard))
    try:
        assert fourth.wait(3)
        assert len(reads) == 4
        assert sum(ref().numel()*ref().element_size() for ref in pins if ref() is not None) == 48
        assert not done.is_set()
        assert reserves == [dict(reserve_bytes=120)]  # 48 CPU + 72 device bytes
    finally:
        allow_new.set()
        for event in cpu_transport.events:
            event.done.set()
        thread.join(4)
    assert done.is_set()
    assert isinstance(result[0],dict), repr(result[0])
    assert len(reads) == 8


def test_partial_cuda_failure_joins_cpu_reader_before_device_fence(monkeypatch, cpu_transport):
    second_started, finish_second = threading.Event(), threading.Event()
    original_copy = torch.Tensor.copy_
    def copy(target, source, *args, **kwargs):
        if int(source[0,0]) == 1:
            second_started.set()
            assert finish_second.wait(3)
        return original_copy(target,source,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'copy_',copy)
    def fail_after_enqueue(live, staged):
        assert second_started.wait(2)
        raise RuntimeError('injected comparison failure after H2D enqueue')
    monkeypatch.setattr(campaign,'_device_differs',fail_after_enqueue)
    values=[torch.full((2,3),i,dtype=torch.bfloat16) for i in range(2)]
    thread,done,result=_foreground(lambda:run_check(monkeypatch,values))
    try:
        assert second_started.wait(2)
        assert not done.wait(.1)
        assert not cpu_transport.synchronized
    finally:
        finish_second.set();thread.join(4)
    assert done.is_set()
    assert isinstance(result[0],RuntimeError)
    assert 'after H2D enqueue' in str(result[0])
    assert cpu_transport.synchronized == [torch.device('cuda:0')]


def test_undersized_private_cap_refuses_before_source_read(monkeypatch, cpu_transport):
    def read(*args,**kwargs):
        pytest.fail('source read occurred before cap refusal')
    values=[torch.zeros((2,3),dtype=torch.bfloat16)]
    with pytest.raises(RuntimeError,match='exceeds.*byte cap'):
        run_check(monkeypatch,values,read=read,max_bytes=11)


def test_shared_pool_consumer_cannot_resize_existing_executor(monkeypatch):
    first=layer_streaming._layer_read_pool(1)
    try:
        with pytest.raises(RuntimeError,match='cannot be resized'):
            layer_streaming._layer_read_pool(2,allow_resize=False)
        assert layer_streaming._LAYER_READ_POOL is first
    finally:
        first.shutdown(wait=True)
        monkeypatch.setattr(layer_streaming,'_LAYER_READ_POOL',None)
        monkeypatch.setattr(layer_streaming,'_LAYER_READ_POOL_THREADS',0)


def test_cancellation_stops_new_reads_and_joins_started_private_copies(monkeypatch, cpu_transport):
    from concurrent.futures import CancelledError
    cancel, both, finish = threading.Event(), threading.Event(), threading.Event()
    original_copy = torch.Tensor.copy_
    started = []
    lock = threading.Lock()
    def copy(target, source, *args, **kwargs):
        with lock:
            started.append(int(source[0,0]))
            if len(started) == 2:
                both.set()
        assert finish.wait(3)
        return original_copy(target,source,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'copy_',copy)
    values=[torch.full((2,3),i,dtype=torch.bfloat16) for i in range(6)]
    thread,done,result=_foreground(lambda:run_check(monkeypatch,values,cancel=cancel))
    try:
        assert both.wait(2)
        cancel.set()
        assert not done.wait(.1)
    finally:
        finish.set();thread.join(4)
    assert done.is_set()
    assert isinstance(result[0],CancelledError),repr(result[0])
    assert set(started) == {0,1}
    assert not cpu_transport.synchronized
