"""Ordered preparation controls and real CUDA guard-residency regressions."""
import inspect
import threading
import time
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


def _default_allocator_settings():
    """Qualified PyTorch 2.11 effective snapshot, including the signed SIZE_MAX."""
    intervals = (64 * 1024**3 // 1024**2).bit_length() - 1
    return {'PYTORCH_CUDA_ALLOC_CONF': '', 'expandable_segments': False,
            'max_split_size': -1, 'garbage_collection_threshold': 0.0,
            'roundup_power2_divisions': {str(1 << i): 0 for i in range(intervals)}}


@pytest.fixture
def cpu_transport(monkeypatch):
    """Use real CPU copies and explicit completion tokens for transport controls."""
    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '2')
    monkeypatch.setattr(layer_streaming, '_LAYER_READ_POOL', None)
    monkeypatch.setattr(layer_streaming, '_LAYER_READ_POOL_THREADS', 0)
    # Controls replace CUDA transport only; residency regressions below use
    # the real allocator and the real split CaptureMemoryGuard.
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", lambda: {
        "allocator_settings": _default_allocator_settings()})
    from prismaquant.kernels import cuda_allocator_state
    monkeypatch.setattr(cuda_allocator_state, 'sizing', lambda: (20 * 1024**2, 20 * 1024**2))
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
    thread, done, result = _foreground(lambda:run_check(monkeypatch,values,read=read))
    try:
        assert fourth.wait(3)
        assert len(reads) == 4
        assert sum(ref().numel()*ref().element_size() for ref in pins if ref() is not None) == 48
        assert not done.is_set()
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


def test_reaped_credit_reaches_readers_during_head_wait(monkeypatch, cpu_transport):
    """A launch completed during a head wait frees its credit inside that wait.

    The coordinator sits in ``u2``'s staging wait while ``u1``'s completion
    event turns done. The fifth read may start only if the wait itself reaps
    the freed credit and admits; a loop top cannot, because every credit is
    held until the wait ends.
    """
    reads = []
    launched = []
    third_started = threading.Event()
    original_launch = campaign._launch_prepared_projected_check
    def launch(check, live):
        launched.append(int(live.value[0, 0]))
        return original_launch(check, live)
    monkeypatch.setattr(campaign, '_launch_prepared_projected_check', launch)
    gate = threading.Event()
    fifth = threading.Event()
    allow_new = threading.Event()
    original_event = torch.cuda.Event
    class SecondEventHeld(original_event):
        def __init__(self):
            # Decide before super().__init__ appends this event: holding the
            # FIRST launch's token strands its credit for the whole test, and
            # the mid-wait completion below must be the second launch's.
            second = len(cpu_transport.events) == 1
            super().__init__()
            if not allow_new.is_set() and second:
                self.done.clear()
    monkeypatch.setattr(torch.cuda, 'Event', SecondEventHeld)
    original_copy = torch.Tensor.copy_
    def copy(target, source, *args, **kwargs):
        if int(source[0, 0]) == 2:
            assert gate.wait(5), 'head wait never admitted the reaped credit'
        return original_copy(target, source, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'copy_', copy)
    values = [torch.full((2, 3), i, dtype=torch.bfloat16) for i in range(6)]
    def read(name, unit, **kwargs):
        # Reader starts are concurrent, unlike coordinator submissions and
        # CUDA launches. Force a legal non-FIFO start to make that distinction
        # deterministic instead of depending on executor scheduling luck.
        if name == 'u1':
            assert third_started.wait(3), 'third read never started'
        reads.append(name)
        if name == 'u2':
            third_started.set()
        if name == 'u5':
            fifth.set()
        return values[int(name[1:])].clone(), lambda:None
    monkeypatch.setattr(campaign, '_read_projected_unit', read)
    thread, done, result = _foreground(lambda: run_check(monkeypatch, values, read=read))
    try:
        deadline = time.monotonic() + 5
        while ((len(reads) < 5 or len(cpu_transport.events) < 2)
               and time.monotonic() < deadline):
            time.sleep(.005)
        assert sorted(reads[:5]) == ['u0', 'u1', 'u2', 'u3', 'u4'], reads
        assert launched == [0, 1], launched
        assert not fifth.is_set(), 'fifth read started before its credit existed'
        cpu_transport.events[1].done.set()  # u1's H2D completes mid-wait
        assert fifth.wait(2), 'no read started from the credit freed during the head wait'
    finally:
        gate.set()
        allow_new.set()
        for event in cpu_transport.events:
            event.done.set()
        thread.join(5)
    assert done.is_set()
    assert isinstance(result[0], dict), repr(result[0])
    assert sorted(result[0]) == [f'u{i}' for i in range(6)]
    assert launched == list(range(6))


@pytest.mark.parametrize('exit_kind', ['mismatch','cancel','allocation','copy','success'])
def test_preparation_releases_once_after_source_alias_drop_on_every_exit(monkeypatch, cpu_transport, exit_kind):
    import weakref
    from concurrent.futures import CancelledError
    released, read_done = [], threading.Event()
    source_ref = []
    def read(name, unit, **kwargs):
        weight = torch.ones((2,3),dtype=torch.bfloat16)
        ref = weakref.ref(weight)
        source_ref.append(ref)
        def release():
            assert ref() is None, 'source alias remains live at page release'
            released.append(name)
        read_done.set()
        return weight,release
    monkeypatch.setattr(campaign,'_read_projected_unit',read)
    if exit_kind == 'allocation':
        def bad_allocate(value,**kwargs):
            value = None
            return torch.empty((-1,))
        monkeypatch.setattr(torch,'empty_like',bad_allocate)
    elif exit_kind == 'copy':
        monkeypatch.setattr(torch,'empty_like',lambda value,**kwargs:
                            torch.empty((3,2),dtype=value.dtype))
    kwargs = dict(live_shape=(2,3),live_dtype=torch.float32 if exit_kind=='mismatch' else torch.bfloat16,
        model_path='unused',source={},release_source_pages=True,source_authentication=object(),
        cancelled=lambda:exit_kind=='cancel' and read_done.is_set())
    unit=dict(source_tensor='w',rows=2,cols=3)
    if exit_kind in ('allocation','copy','cancel'):
        with pytest.raises((RuntimeError,CancelledError)):
            campaign._prepare_device_projected_check('u',unit,**kwargs)
    else:
        check=campaign._prepare_device_projected_check('u',unit,**kwargs)
        assert check.differs is True if exit_kind=='mismatch' else check.pinned is not None
    assert released == ['u']
    assert source_ref[0]() is None


@pytest.fixture
def cuda_preparation(monkeypatch, tmp_path):
    """Real H2D, comparison, reduction, stack and reserved-memory observations."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for projected preparation guard residency')
    from prismaquant import memory_management as mm
    from test_capture_memory_guard import _cgroup

    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '2')
    monkeypatch.setattr(layer_streaming, '_LAYER_READ_POOL', None)
    monkeypatch.setattr(layer_streaming, '_LAYER_READ_POOL_THREADS', 0)
    tree = _cgroup(tmp_path, cap_bytes=8 * 1024**3, current_bytes=0)
    auth = object()
    fillers = []
    reads = []
    original_event = torch.cuda.Event

    class HeldEvent:
        # Hold all four credits deterministically, but never fake CUDA completion.
        def __init__(self):
            self.event = original_event()
        def record(self, stream):
            self.event.record(stream)
        def query(self):
            return False
        def synchronize(self):
            self.event.synchronize()

    def build(count, shape):
        values = [torch.full(shape, i % 2, dtype=torch.bfloat16) for i in range(count)]
        weights = {f'u{i:04}': value.cuda() for i, value in enumerate(values)}
        units = {name: dict(source_tensor=name, rows=shape[0], cols=shape[1])
                 for name in weights}
        def read(name, unit, **kwargs):
            assert kwargs['source_authentication'] is auth
            reads.append(name)
            return values[int(name[1:])], lambda: None
        monkeypatch.setattr(campaign, '_read_projected_unit', read)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_reserved()
        # Occupy every cached small-pool hole with real allocations. Otherwise
        # the tiny verdicts could hide in the baseline's unused 2-MiB segment,
        # and the payload-only undercharge would not be exercised.
        while True:
            holes = [block['size'] for segment in torch.cuda.memory_snapshot()
                     if segment['segment_type'] == 'small' and segment['device'] == 0
                     for block in segment['blocks'] if block['state'] == 'inactive']
            if not holes:
                break
            fillers.append(torch.empty(min(max(holes), 1024**2),
                                       dtype=torch.uint8, device='cuda'))
        return weights, {'s': units}, baseline

    def check(weights, bound, guard, cap):
        return campaign._checked_projected_units(bound, weights=weights,
            model_path='fixture', source={}, source_authentication=auth,
            resource_check=guard.check, parallel_preparation=True,
            preparation_max_bytes=cap)

    yield SimpleNamespace(build=build, check=check, reads=reads, tree=tree,
                          mm=mm, held_event=HeldEvent)
    pool = layer_streaming._LAYER_READ_POOL
    if pool is not None:
        pool.shutdown(wait=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.mark.parametrize('count,shape', [(4, (2048, 4096)), (864, (2, 3))],
                         ids=['comparison-mask', 'allocator-sized-verdicts'])
def test_cuda_guard_refuses_undercharged_window_before_any_source_read(
        monkeypatch, cuda_preparation, count, shape):
    state = cuda_preparation
    weights, bound, baseline = state.build(count, shape)
    size = weights['u0000'].numel() * weights['u0000'].element_size()
    cap = 4 * size
    # The unfixed admission grants exactly this logical-payload envelope, then
    # fails only AFTER a source read and device allocation have crossed it.
    guard = state.mm.CaptureMemoryGuard('cuda', device_bytes=baseline + cap + 2 * count,
                                        **state.tree)
    if count == 4:
        monkeypatch.setattr(torch.cuda, 'Event', state.held_event)
    with pytest.raises(RuntimeError, match='capture device memory refusal'):
        state.check(weights, bound, guard, cap)
    assert state.reads == [], 'unsafe window admitted source reads before guard refusal'
    assert guard.last['label'] == 'before_parallel_projected_preparation'


@pytest.mark.parametrize('count,shape', [(4, (2048, 4096)), (64, (2, 3))],
                         ids=['comparison-mask', 'allocator-sized-verdicts'])
def test_cuda_admitted_window_covers_actual_reserved_growth_through_settle(
        monkeypatch, cuda_preparation, count, shape):
    state = cuda_preparation
    weights, bound, baseline = state.build(count, shape)
    cap = 4 * weights['u0000'].numel() * weights['u0000'].element_size()
    guard = state.mm.CaptureMemoryGuard('cuda', device_bytes=baseline + 2 * 1024**3,
                                        **state.tree)
    growth, admitted = [], []
    original_check, original_ne, original_stack = guard.check, torch.ne, torch.stack
    def observe():
        growth.append(torch.cuda.memory_reserved() - baseline)
    def check(label, **kwargs):
        record = original_check(label, **kwargs)
        if label == 'before_parallel_projected_preparation':
            admitted.append(record['future_device_allocation_bytes'])
        observe()
        return record
    # Retain the guard's declared split-reservation contract on the wrapper.
    setattr(check, state.mm.SEPARATE_RESERVATIONS, guard.separate_reservations)
    guard.check = check
    def ne(*args, **kwargs):
        mask = original_ne(*args, **kwargs)
        observe()  # The real full-size bool mask is still live here.
        return mask
    def stack(*args, **kwargs):
        flags = original_stack(*args, **kwargs)
        observe()  # The real stack and every original device verdict coexist.
        return flags
    monkeypatch.setattr(torch, 'ne', ne)
    monkeypatch.setattr(torch, 'stack', stack)
    if count == 4:
        monkeypatch.setattr(torch.cuda, 'Event', state.held_event)
    assert state.check(weights, bound, guard, cap) == bound['s']
    assert len(state.reads) == count
    assert len(admitted) == 1
    assert max(growth) <= admitted[0], (
        f'actual CUDA reserved growth {max(growth)} exceeds admitted {admitted[0]}')
    print(f'CUDA guard: units={count}, reserved_growth={max(growth)}, '
          f'admitted_device_bytes={admitted[0]}, device={torch.cuda.get_device_name()}')




@pytest.mark.parametrize('backend,settings', [
    ('cudaMallocAsync', {'PYTORCH_CUDA_ALLOC_CONF': ''}),
    ('native', {'PYTORCH_CUDA_ALLOC_CONF': 'large_segment_size_mb:64'}),
    ('native', {}),
])
def test_unpriced_allocator_refuses_before_source_read(
        monkeypatch, cpu_transport, backend, settings):
    reads = []
    monkeypatch.setattr(torch.cuda, 'get_allocator_backend', lambda: backend)
    monkeypatch.setattr(torch.cuda.memory, '_snapshot', lambda: {'allocator_settings': settings})
    values = [torch.zeros((2, 3), dtype=torch.bfloat16)]
    def read(name, unit, **kwargs):
        reads.append(name)
        return values[0], lambda: None
    with pytest.raises(RuntimeError, match='default native CUDA allocator residency model'):
        run_check(monkeypatch, values, read=read)
    assert reads == []


@pytest.mark.parametrize('field,value', [
    ('expandable_segments', True),
    ('max_split_size', 64 * 1024**2),
    ('garbage_collection_threshold', 0.5),
    ('roundup_power2_divisions', {**_default_allocator_settings()['roundup_power2_divisions'], '1': 4}),
    ('roundup_power2_divisions', {}),
    ('roundup_power2_divisions', {'1': 0}),
    ('roundup_power2_divisions', {**_default_allocator_settings()['roundup_power2_divisions'], '1': False}),
    ('expandable_segments', None),
    ('max_split_size', None),
    ('garbage_collection_threshold', None),
    ('roundup_power2_divisions', None),
])
def test_effective_allocator_state_refuses_before_source_read(
        monkeypatch, cpu_transport, field, value):
    settings = _default_allocator_settings()
    if value is None:
        del settings[field]
    else:
        settings[field] = value
    assert settings['PYTORCH_CUDA_ALLOC_CONF'] == ''
    monkeypatch.setattr(torch.cuda.memory, '_snapshot', lambda: {'allocator_settings': settings})
    reads = []
    values = [torch.zeros((2, 3), dtype=torch.bfloat16)]
    def read(name, unit, **kwargs):
        reads.append(name)
        return values[0], lambda: None
    with pytest.raises(RuntimeError, match='default native CUDA allocator residency model'):
        run_check(monkeypatch, values, read=read)
    assert reads == []


@pytest.mark.parametrize('configuration,expandable', [
    ('expandable_segments:True,large_segment_size_mb:64', True),
    ('large_segment_size_mb:64', False),
], ids=['exposed-expandable-reset', 'hidden-large-only-reset'])
def test_sticky_allocator_reset_is_refused_in_isolated_cuda_process(configuration, expandable):
    """Real setter/reset state is process-global, so it never enters suite workers."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for isolated allocator setter/reset qualification')
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent('''
        import json, os, sys, torch
        from prismaquant import tessera_campaign as campaign
        from prismaquant.kernels import cuda_allocator_state
        torch.cuda.init()
        defaults = torch.cuda.memory._snapshot()['allocator_settings']
        assert defaults['PYTORCH_CUDA_ALLOC_CONF'] == ''
        assert defaults['expandable_segments'] is False
        assert cuda_allocator_state.sizing() == (20 * 1024**2, 20 * 1024**2)
        torch._C._accelerator_setAllocatorSettings(sys.argv[1])
        torch._C._accelerator_setAllocatorSettings('')
        sticky = torch.cuda.memory._snapshot()['allocator_settings']
        assert sticky['PYTORCH_CUDA_ALLOC_CONF'] == ''
        assert sticky['expandable_segments'] is (sys.argv[2] == '1')
        native_sizing = cuda_allocator_state.sizing()
        assert native_sizing == (64 * 1024**2, 64 * 1024**2), native_sizing
        print(json.dumps({'torch': torch.__version__, 'torch_source': torch.version.git_version,
                          'defaults': defaults, 'sticky_after_empty_reset': sticky,
                          'public_C10_sizing': native_sizing}), flush=True)
        os.environ['PRISMAQUANT_LAYER_READ_THREADS'] = '2'
        source_weight = torch.zeros((2, 3), dtype=torch.bfloat16)
        weights = {'u': source_weight.cuda()}
        unit = dict(source_tensor='w', rows=2, cols=3)
        reads = []
        def read(name, unit, **kwargs):
            reads.append(name)
            return source_weight, lambda: None
        campaign._read_projected_unit = read
        try:
            campaign._checked_projected_units({'s': {'u': unit}}, weights=weights,
                model_path='fixture', source={}, source_authentication=object(),
                resource_check=lambda label, **kwargs: None,
                parallel_preparation=True, preparation_max_bytes=48)
        except RuntimeError as error:
            assert 'default native CUDA allocator residency model' in str(error), str(error)
        else:
            raise AssertionError(f'empty config string admitted sticky allocator state; reads={reads}')
        assert reads == [], reads
        print('sticky allocator reset: refused before source reads', flush=True)
    ''')
    child = subprocess.run([sys.executable, '-c', script, configuration, '1' if expandable else '0'],
                           capture_output=True, text=True)
    assert child.returncode == 0, child.stdout + child.stderr
    assert 'sticky allocator reset: refused before source reads' in child.stdout
    print(child.stdout)



@pytest.mark.parametrize('value', [(64 * 1024**2, 64 * 1024**2),
                                 (20 * 1024**2, 64 * 1024**2), None,
                                 (True, 20 * 1024**2), RuntimeError('native accessor unavailable')])
def test_hidden_allocator_sizing_refuses_before_source_read(monkeypatch, cpu_transport, value):
    from prismaquant.kernels import cuda_allocator_state
    def sizing():
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(cuda_allocator_state, 'sizing', sizing)
    reads = []
    values = [torch.zeros((2, 3), dtype=torch.bfloat16)]
    def read(name, unit, **kwargs):
        reads.append(name)
        return values[0], lambda: None
    with pytest.raises(RuntimeError, match='default native CUDA allocator residency model'):
        run_check(monkeypatch, values, read=read)
    assert reads == []


def test_public_allocator_getters_read_default_live_sizing():
    from prismaquant.kernels import cuda_allocator_state
    assert cuda_allocator_state.sizing() == (20 * 1024**2, 20 * 1024**2)

