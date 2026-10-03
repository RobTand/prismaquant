"""Actual original memfd/alias path with CPU-spied CUDA calls, never GPU acceptance."""
from __future__ import annotations

import gc
import threading
import weakref

import pytest
import torch

from prismaquant import layer_streaming as ls, tessera_calibration_cache as cc
from test_capture_original_material import material, _owner, _forget_state  # noqa: F401
from test_streaming_source_pages import FakeCudaTensor

pytestmark = pytest.mark.own_process


class _ArithmeticCudaSpyTensor(FakeCudaTensor):
    """GPU-resident operations execute on the spy's independent CPU payload."""

    def to(self, *args, **kwargs):
        dtype = kwargs.get('dtype')
        if args and isinstance(args[0], torch.dtype):
            dtype = args[0]
        return _ArithmeticCudaSpyTensor(self.value.to(dtype=dtype or self.value.dtype))

    def __mul__(self, other):
        return self.value * (other.value if isinstance(other, FakeCudaTensor) else other)


@pytest.fixture
def original_copy_spy(material, monkeypatch):
    owner = _owner(material)
    state = dict(owner=owner, copies=[], events=[], reads=[], installed=[], fault=None,
                 cancel=None, decoded=[], local=threading.local())
    # This explicitly exposes the dormant path for CPU call-order controls.
    # The actual production gate is unchanged and no real CUDA API executes.
    monkeypatch.setattr(owner, 'require_material_device', lambda device: None)
    real_enter = cc._CaptureSourceSafeOpen.__enter__
    real_get = cc._CaptureSourceSafeOpen.get_tensor
    real_to = torch.Tensor.to

    def enter(reader):
        value = real_enter(reader)
        state['local'].native = []
        return value

    def get(reader, key):
        if state['fault'] == 'read' and state['reads']:
            raise RuntimeError('second source payload failed')
        value = real_get(reader, key)
        ref = weakref.ref(value)
        state['local'].native.append(ref)
        state['reads'].append(ref)
        return value

    def copy(value, target=None, *args, **kwargs):
        device = kwargs.get('device', target)
        if isinstance(device, torch.device) and device.type == 'cuda':
            state['copies'].append(weakref.ref(value))
            if state['cancel'] is not None:
                state['cancel'].set()
            if state['fault'] == 'copy' and len(state['copies']) == 2:
                raise RuntimeError('copy failed after enqueue')
            converted = real_to(value, dtype=kwargs.get('dtype', value.dtype))
            return _ArithmeticCudaSpyTensor(converted)
        return real_to(value, target, *args, **kwargs) if target is not None else real_to(value, **kwargs)

    streams = {}

    class Stream:
        def __init__(self):
            self.thread = threading.get_ident()
        def synchronize(self):
            assert owner.material_live_bytes == len(material['raws']['one.safetensors'])
            assert all(ref() is not None for ref in state['copies'])
            state['events'].append('stream-sync')
            if state.get('drain_fault'):
                raise RuntimeError('exact stream drain failed')

    def current_stream(*args):
        return streams.setdefault(threading.get_ident(), Stream())

    class Event:
        def record(self, stream):
            assert stream.thread == threading.get_ident()
            self.native = list(state['local'].native)
            state['events'].append('record')
            if state['fault'] == 'record':
                raise RuntimeError('event record failed')
        def synchronize(self):
            # Reader/windows have exited, so only true native storage owners
            # and the completion frame can keep whole serialized credit live.
            assert all(ref() is not None for ref in self.native)
            assert owner.material_live_bytes == len(material['raws']['one.safetensors'])
            state['events'].append('sync')
            if state['fault'] == 'sync':
                raise RuntimeError('event sync failed')

    real_open = owner.safe_open
    def opened(factory, path, **kwargs):
        state['decoded'].append(dict(kwargs))
        assert kwargs.get('device', 'cpu') == 'cpu'
        return real_open(factory, path, **kwargs)

    monkeypatch.setattr(cc._CaptureSourceSafeOpen, '__enter__', enter)
    monkeypatch.setattr(cc._CaptureSourceSafeOpen, 'get_tensor', get)
    monkeypatch.setattr(owner, 'safe_open', opened)
    monkeypatch.setattr(torch.Tensor, 'to', copy)
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'current_stream', current_stream)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: pytest.fail('device-wide fence'))
    monkeypatch.setattr(ls, '_advise_consumed_safetensors_pages', lambda *a: None)
    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '1')
    yield state
    gc.collect()
    owner.close()


def _layer(state, count=2):
    names = [f'unit.{index}' for index in range(count)]
    path = str(state['owner'].root / 'one.safetensors')
    return ls._read_layer_to_device('unit.', {n: path for n in names}, {n: 'w' for n in names},
                                    torch.bfloat16, torch.device('cuda'),
                                    source_authentication=state['owner'], cancel=state['cancel'])


def _clear_frames(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        error.__traceback__ = None
        error = error.__cause__ or error.__context__
    gc.collect()


@pytest.mark.parametrize('pages', ['0', '1'])
@pytest.mark.parametrize('direct', ['0', '1'])
def test_original_layer_fences_native_and_cast_aliases_independent_of_page_policy(
        original_copy_spy, monkeypatch, pages, direct):
    s = original_copy_spy
    monkeypatch.setenv('PRISMAQUANT_RELEASE_SOURCE_PAGES', pages)
    monkeypatch.setenv('PRISMAQUANT_DIRECT_CUDA_LOAD', direct)
    out = _layer(s)
    assert s['events'] == ['record', 'sync']
    assert all(row == {'framework': 'pt'} for row in s['decoded'])
    assert all(torch.equal(v.value, torch.arange(32).reshape(4, 8).to(torch.bfloat16))
               for v in out.values())
    assert s['owner'].material_live_bytes == 0


@pytest.mark.parametrize('fault', ['read', 'copy', 'record', 'sync'])
@pytest.mark.parametrize('pages', ['0', '1'])
def test_original_failure_fences_copies_and_retains_failed_completion_aliases(
        original_copy_spy, monkeypatch, fault, pages):
    s = original_copy_spy
    s['fault'] = fault
    monkeypatch.setenv('PRISMAQUANT_RELEASE_SOURCE_PAGES', pages)
    monkeypatch.setenv('PRISMAQUANT_DIRECT_CUDA_LOAD', '0')
    with pytest.raises(RuntimeError) as caught:
        _layer(s)
    assert 'record' in s['events']
    assert ('sync' in s['events']) is (fault != 'record')
    _clear_frames(caught.value)
    caught = None  # Discard pytest's own traceback owner too.
    gc.collect()
    if fault in ('record', 'sync'):
        assert s['events'].count('stream-sync') == 1
    assert not cc._FAILED_ORIGINAL_COPY_OWNERS
    assert not s['owner']._original_copy_completions
    assert s['owner'].material_live_bytes == 0


def test_original_cancel_during_copy_drains_before_returning(original_copy_spy, monkeypatch):
    from concurrent.futures import CancelledError
    s = original_copy_spy
    s['cancel'] = threading.Event()
    monkeypatch.setenv('PRISMAQUANT_RELEASE_SOURCE_PAGES', '0')
    with pytest.raises(CancelledError) as caught:
        _layer(s)
    assert s['events'] == ['record', 'sync']
    _clear_frames(caught.value)
    caught = None
    gc.collect()
    assert s['owner'].material_live_bytes == 0


@pytest.mark.parametrize('cancel', [False, True])
def test_original_parallel_readers_all_drain_copy_streams(original_copy_spy, monkeypatch, cancel):
    from concurrent.futures import CancelledError
    s = original_copy_spy
    monkeypatch.setenv('PRISMAQUANT_RELEASE_SOURCE_PAGES', '0')
    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '4')
    if cancel:
        s['cancel'] = threading.Event()
        with pytest.raises(CancelledError) as caught:
            _layer(s, count=20)
        _clear_frames(caught.value)
        caught = None
        gc.collect()
    else:
        assert len(_layer(s, count=20)) == 20
    assert s['events'].count('record') == s['events'].count('sync') == 4
    assert s['owner'].material_live_bytes == 0


def test_original_cancel_before_reads_acquires_no_material(original_copy_spy):
    from concurrent.futures import CancelledError
    s = original_copy_spy
    s['cancel'] = threading.Event()
    s['cancel'].set()
    with pytest.raises(CancelledError):
        _layer(s)
    assert not s['decoded'] and not s['copies'] and not s['events']
    assert s['owner'].material_live_bytes == 0


def test_original_head_copies_complete_before_install(original_copy_spy, monkeypatch):
    s = original_copy_spy
    def install(model, name, device, *, value, dtype):
        assert s['events'][-1] == 'sync'
        assert value.device.type == 'cuda'
        s['installed'].append(name)
    monkeypatch.setattr(ls, 'set_module_tensor_to_device', install)
    path = str(s['owner'].root / 'one.safetensors')
    model = torch.nn.Linear(8, 4, bias=False)
    assert ls._materialize(model, ['weight'], {'weight': path}, {'weight': 'w'},
                           torch.device('cuda'), torch.bfloat16,
                           source_authentication=s['owner']) == 1
    assert s['events'] == ['record', 'sync'] and s['installed'] == ['weight']
    assert s['owner'].material_live_bytes == 0


def test_original_failed_head_copy_installs_nothing(original_copy_spy, monkeypatch):
    s = original_copy_spy
    s['fault'] = 'copy'
    monkeypatch.setattr(ls, 'set_module_tensor_to_device',
                        lambda *a, **kw: s['installed'].append(a[1]))
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.zeros(4, 8))
    model.b = torch.nn.Parameter(torch.zeros(4, 8))
    path = str(s['owner'].root / 'one.safetensors')
    with pytest.raises(RuntimeError, match='copy failed') as caught:
        ls._materialize(model, ['a', 'b'], {'a': path, 'b': path}, {'a': 'w', 'b': 'w'},
                        torch.device('cuda'), torch.bfloat16, source_authentication=s['owner'])
    assert not s['installed'] and s['events'] == ['record', 'sync']
    _clear_frames(caught.value)
    caught = None
    gc.collect()
    assert s['owner'].material_live_bytes == 0


def test_original_direct_dequant_keeps_native_scale_through_copy_completion(original_copy_spy):
    s = original_copy_spy
    path = str(s['owner'].root / 'one.safetensors')
    scales = ls.Fp8ScaleInvMap({'weight': (path, 'w')}, block=(1, 1))
    out = {'weight': torch.ones(4, 8, dtype=torch.bfloat16)}
    assert ls._apply_fp8_dequant_inplace(out, scales, torch.device('cuda'),
                                        source_authentication=s['owner']) == 1
    assert s['events'] == ['record', 'sync']
    assert torch.equal(out['weight'], torch.arange(32).reshape(4, 8).to(torch.bfloat16))
    assert s['owner'].material_live_bytes == 0


def test_original_dequant_fallback_transfers_host_weight_beside_scale(original_copy_spy):
    s = original_copy_spy
    path = str(s['owner'].root / 'one.safetensors')
    scales = ls.Fp8ScaleInvMap({'weight': (path, 'w')}, block=(3, 3))
    out = {'weight': torch.ones(10, 22, dtype=torch.bfloat16)}
    assert ls._apply_fp8_dequant_inplace(out, scales, torch.device('cuda'),
                                        source_authentication=s['owner']) == 1
    # The original decoder always supplies CPU values; on an actual CUDA
    # run both operands must reach the execution device before arithmetic.
    assert len(s['copies']) == 2
    assert s['events'] == ['record', 'sync']
    expected = torch.arange(32).reshape(4, 8).repeat_interleave(3, 0).repeat_interleave(3, 1)[:10, :22]
    assert torch.equal(out['weight'], expected.to(torch.bfloat16))

@pytest.mark.parametrize('fault', ['record', 'sync'])
def test_double_fence_failure_roots_abandoned_owner_until_explicit_recovery(
        material, monkeypatch, fault):
    import os

    class Stream:
        fail = True
        calls = 0
        def synchronize(self):
            self.calls += 1
            if self.fail:
                raise RuntimeError('fatal exact-stream drain failure')

    stream = Stream()
    class Event:
        def record(self, observed):
            assert observed is stream
            if fault == 'record':
                raise RuntimeError('original event record failure')
        def synchronize(self):
            raise RuntimeError('original event sync failure')

    real_to = torch.Tensor.to
    def transfer(value, target=None, *args, **kwargs):
        if isinstance(target, torch.device) and target.type == 'cuda':
            return FakeCudaTensor(value.clone())
        return real_to(value, target, *args, **kwargs) if target is not None else real_to(value, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', transfer)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args: stream)
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    assert not getattr(cc, '_FAILED_ORIGINAL_COPY_OWNERS', ())
    reads = {}
    real_get = cc._CaptureSourceSafeOpen.get_tensor
    def get(reader, key):
        value = real_get(reader, key)
        reads['fd'] = reader.state['fd']
        return value
    monkeypatch.setattr(cc._CaptureSourceSafeOpen, 'get_tensor', get)
    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '1')

    def abandon():
        owner = _owner(material)
        reference = weakref.ref(owner)
        path = material['root'] / 'one.safetensors'
        # A fixture-local predicate only; the loader intake is unchanged on
        # the historical source, making loss of ownership the causal RED.
        owner.require_material_device = lambda device: None
        try:
            ls._read_layer_to_device('unit.', {'unit.w': str(path)}, {'unit.w': 'w'},
                                     torch.bfloat16, torch.device('cuda'), source_authentication=owner)
        except RuntimeError as error:
            assert 'original event' in str(error), 'secondary drain must not replace event error'
            _clear_frames(error)
        return reference, reads['fd']

    reference, held_fd = abandon()  # Only a weakref and an integer escape.
    gc.collect()
    assert reference() is not None
    assert reference().material_live_bytes == len(material['raws']['one.safetensors'])
    assert os.fstat(held_fd).st_size == len(material['raws']['one.safetensors'])
    for _ in range(2):
        with pytest.raises(RuntimeError, match='fatal exact-stream') as failed:
            cc.CaptureSourceAuthentication.close_failed_original_copies()
        _clear_frames(failed.value)
        failed = None
        gc.collect()
        assert reference() is not None and not reference()._closed
        assert reference().material_live_bytes > 0
        assert os.fstat(held_fd).st_size > 0
    stream.fail = False
    assert cc.CaptureSourceAuthentication.close_failed_original_copies() == 1
    assert stream.calls == 4
    assert not cc._FAILED_ORIGINAL_COPY_OWNERS
    gc.collect()
    assert reference() is None
    with pytest.raises(OSError):
        os.fstat(held_fd)
