"""Unrun actual CUDA controls for the dormant original owner path.

These source primitives use the actual kernel-sealed fixture owner and real
Torch transfers/events. Only the closed device-eligibility predicate is
temporarily overridden on this one fixture owner. This is qualification work,
not a complete original provider or actual GLM derivative acceptance.
"""
from __future__ import annotations

from concurrent.futures import CancelledError
import gc
import json
import os
from pathlib import Path
import threading
import weakref

import pytest
import torch

from prismaquant import layer_streaming as ls, tessera_calibration_cache as cc
from test_capture_original_material import material, _owner, _forget_state  # noqa: F401
from test_original_source_copy_completion import _clear_frames

pytestmark = [pytest.mark.skipif(
    os.environ.get('PRISMAQUANT_ORIGINAL_CUDA_QUALIFICATION_TEST') != '1'
    or not torch.cuda.is_available(),
    reason='requires explicit root-approved PB CUDA qualification; a skip proves nothing')]


@pytest.mark.parametrize('pages', ['0', '1'])
@pytest.mark.parametrize('material', [torch.float32, torch.bfloat16], indirect=True)
@pytest.mark.parametrize('case', ['layer', 'read-failure', 'copy-failure', 'cancel',
                                 'event-record-failure', 'event-sync-failure', 'head', 'dequant',
                                 'parallel-layer', 'parallel-read-failure', 'parallel-copy-failure',
                                 'parallel-cancel', 'parallel-event-record-failure',
                                 'parallel-event-sync-failure'])
def test_actual_original_copy_stream_ownership(material, monkeypatch, tmp_path, pages, case):
    parallel = case.startswith('parallel-')
    operation = case.removeprefix('parallel-')
    owner = _owner(material)
    real_enter = cc._CaptureSourceSafeOpen.__enter__
    real_get = cc._CaptureSourceSafeOpen.get_tensor
    real_exit = cc._CaptureSourceSafeOpen.__exit__
    real_to = torch.Tensor.to
    real_event = torch.cuda.Event
    real_stream_sync = torch.cuda.Stream.synchronize
    local = threading.local()
    lock = threading.Lock()
    streams = []
    reader_threads = set()
    observed = dict(copies=0, pending_events=0, completed_events=0, decoder_reads=0,
                    credit_before_sync=[], credit_after_sync=[], pending_stream_on_failure=False,
                    source_dtype=None, delayed_copies_pending_on_return=0,
                    reader_attempts=0, readers_entered=0, readers_exited=0)
    observed['stream_drains'] = 0
    cancel = threading.Event()
    primary = None
    profiler = None
    profile_path = os.environ.get('PRISMABUILD_PROFILE_TORCH_OUT')
    device = torch.device('cuda:0')
    try:
        # Keep a control showing that the unmodified production gate refuses.
        with pytest.raises(RuntimeError, match='not qualified'):
            owner.require_material_device(device)
        assert owner.material_live_bytes == 0
        monkeypatch.setattr(owner, 'require_material_device', lambda target: None)
        monkeypatch.setenv('PRISMAQUANT_RELEASE_SOURCE_PAGES', pages)
        monkeypatch.setenv('PRISMAQUANT_DIRECT_CUDA_LOAD', '1')
        monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '2' if parallel else '1')

        def enter(reader):
            value = real_enter(reader)
            local.native = []
            # A real independent copy stream exercises the same current-stream
            # API that production readers use; no fake output/Event is installed.
            stream = torch.cuda.Stream(device=device)
            torch.cuda.set_stream(stream)
            with lock:
                streams.append(stream)
                reader_threads.add(threading.get_ident())
                observed['readers_entered'] += 1
            return value

        def exit_reader(reader, *args):
            try:
                return real_exit(reader, *args)
            finally:
                with lock:
                    observed['readers_exited'] += 1

        def get(reader, key):
            with lock:
                attempt = observed['reader_attempts']
                observed['reader_attempts'] += 1
            if operation == 'read-failure' and attempt:
                raise RuntimeError('qualification second-payload failure')
            value = real_get(reader, key)
            assert value.device.type == 'cpu'
            with lock:
                assert observed['source_dtype'] in (None, str(value.dtype))
                observed['source_dtype'] = str(value.dtype)
                observed['decoder_reads'] += 1
            local.native.append(weakref.ref(value))
            return value

        def transfer(value, target=None, *args, **kwargs):
            requested = kwargs.get('device', target)
            delay_finished = None
            if isinstance(requested, torch.device) and requested.type == 'cuda' and value.device.type == 'cpu':
                # Delay BEFORE the real copy to expose pending source lifetime.
                # This is a causal interval control, not a throughput workload.
                torch.cuda._sleep(500_000_000)
                delay_finished = real_event()
                delay_finished.record(torch.cuda.current_stream(device))
                with lock:
                    observed['copies'] += 1
                    copy_number = observed['copies']
                if operation == 'cancel':
                    cancel.set()
            if target is not None:
                result = real_to(value, target, *args, **kwargs)
            else:
                result = real_to(value, **kwargs)
            if delay_finished is not None and not delay_finished.query():
                # The H2D copy follows this still-pending real event on the
                # same stream. An event pending only after a synchronous copy
                # is insufficient evidence for the source lifetime interval.
                with lock:
                    observed['delayed_copies_pending_on_return'] += 1
            if operation == 'copy-failure' and delay_finished is not None and copy_number == 2:
                raise RuntimeError('qualification failure after real copy enqueue')
            return result

        class ObservedEvent:
            def __init__(self):
                self.event = real_event()
            def record(self, stream):
                self.native = list(local.native)
                if operation == 'event-record-failure':
                    with lock:
                        observed['pending_stream_on_failure'] |= not stream.query()
                    raise RuntimeError('qualification injected event record failure')
                self.event.record(stream)
                if not self.event.query():
                    with lock:
                        observed['pending_events'] += 1
            def synchronize(self):
                assert all(ref() is not None for ref in self.native)
                held = owner.material_live_bytes
                assert held == len(material['raws']['one.safetensors'])
                with lock:
                    observed['credit_before_sync'].append(held)
                if operation == 'event-sync-failure':
                    with lock:
                        observed['pending_stream_on_failure'] |= not self.event.query()
                    raise RuntimeError('qualification injected event sync failure')
                self.event.synchronize()
                assert self.event.query()
                with lock:
                    observed['completed_events'] += 1
                    observed['credit_after_sync'].append(owner.material_live_bytes)

        def observed_stream_sync(stream):
            if stream in streams:
                assert owner.material_live_bytes > 0
                real_stream_sync(stream)
                assert stream.query()
                with lock:
                    observed['stream_drains'] += 1
                return
            return real_stream_sync(stream)

        monkeypatch.setattr(cc._CaptureSourceSafeOpen, '__enter__', enter)
        monkeypatch.setattr(cc._CaptureSourceSafeOpen, '__exit__', exit_reader)
        monkeypatch.setattr(cc._CaptureSourceSafeOpen, 'get_tensor', get)
        monkeypatch.setattr(torch.Tensor, 'to', transfer)
        monkeypatch.setattr(torch.cuda, 'Event', ObservedEvent)
        monkeypatch.setattr(torch.cuda.Stream, 'synchronize', observed_stream_sync)
        if profile_path:
            profiler = torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            profiler.__enter__()
        path = str(owner.root / 'one.safetensors')
        if operation in ('layer', 'read-failure', 'copy-failure', 'cancel',
                    'event-record-failure', 'event-sync-failure'):
            names = [f'layer.{index}' for index in range(16 if parallel else 2)]
            try:
                output = ls._read_layer_to_device(
                    'layer.', {n: path for n in names}, {n: 'w' for n in names},
                    torch.bfloat16, device, source_authentication=owner,
                    cancel=cancel if operation == 'cancel' else None)
            except (RuntimeError, CancelledError) as error:
                primary = error
                if operation == 'read-failure':
                    assert 'second-payload' in str(error)
                elif operation == 'cancel':
                    assert isinstance(error, CancelledError)
                elif operation == 'copy-failure':
                    assert 'real copy enqueue' in str(error)
                elif operation in ('event-record-failure', 'event-sync-failure'):
                    assert 'injected event' in str(error)
                    assert not owner._original_copy_completions
                else:
                    raise
            else:
                assert operation == 'layer', 'failure/cancellation returned installable output'
                expected = torch.arange(32).reshape(4, 8).to(torch.bfloat16)
                assert all(torch.equal(value.cpu(), expected) for value in output.values())
                output = None
        elif operation == 'head':
            model = torch.nn.Linear(8, 4, bias=False)
            assert ls._materialize(model, ['weight'], {'weight': path}, {'weight': 'w'},
                                   device, torch.bfloat16, source_authentication=owner) == 1
            assert torch.equal(model.weight.detach().cpu(),
                               torch.arange(32).reshape(4, 8).to(torch.bfloat16))
            model = None
        else:
            weights = {'weight': torch.ones(4, 8, dtype=torch.bfloat16, device=device)}
            torch.cuda.current_stream(device).synchronize()  # setup operand, outside source completion
            scales = ls.Fp8ScaleInvMap({'weight': (path, 'w')}, block=(1, 1))
            assert ls._apply_fp8_dequant_inplace(weights, scales, device,
                                                source_authentication=owner) == 1
            assert torch.equal(weights['weight'].cpu(),
                               torch.arange(32).reshape(4, 8).to(torch.bfloat16))
            weights = None
        assert observed['copies'] > 0
        assert observed['delayed_copies_pending_on_return'] > 0, (
            'no H2D copy returned behind a still-pending real stream dependency')
        assert observed['readers_entered'] == observed['readers_exited']
        if parallel:
            assert len(streams) == 2 and len(reader_threads) == 2
        if operation in ('event-record-failure', 'event-sync-failure'):
            assert observed['pending_stream_on_failure']
            assert observed['completed_events'] == 0
        else:
            assert observed['completed_events'] > 0
            # A synchronous/unsupported pending-copy branch does not qualify
            # the asynchronous interval merely because values match.
            assert observed['pending_events'] > 0, 'no pending copy-stream completion observed'
        if primary is not None:
            _clear_frames(primary)
            primary = None
        gc.collect()
        if operation in ('event-record-failure', 'event-sync-failure'):
            observed['completed_streams_after_traceback_disposal'] = sum(
                stream.query() for stream in streams)
            assert observed['completed_streams_after_traceback_disposal'] == len(streams)
            assert observed['stream_drains'] == len(streams)
            assert not owner._original_copy_completions
            assert owner not in cc._FAILED_ORIGINAL_COPY_OWNERS
        assert owner.material_live_bytes == 0
        # The dtype is read from the actual decoded source, not inferred from
        # a filename. The parameter is part of every exported control identity.
        source_dtype = observed['source_dtype']
        assert source_dtype in ('torch.bfloat16', 'torch.float32')
        raw = dict(schema='prismaquant.original_copy_cuda_control.v2', case=case, pages=pages,
                   source_dtype=source_dtype,
                   reader_threads=len(reader_threads), copy_streams=len(streams), parallel=parallel,
                   original_owner_type=type(owner).__name__, device=str(device),
                   source_control_override='fixture-owner device predicate only; production remains closed',
                   automatic_capture_qualified=False, actual_glm=False, observations=observed,
                   source_receipt=owner.receipt(), source_receipt_phase='after_successful_copy_or_failure_drain',
                   torch=str(torch.__version__), cuda=torch.version.cuda)
        out = os.environ.get('PRISMAQUANT_ORIGINAL_CUDA_QUALIFICATION_OUT')
        if out:
            destination = Path(out)
            destination.mkdir(parents=True, exist_ok=True)
            dtype_label = source_dtype.removeprefix('torch.')
            (destination / f'{case}-pages{pages}-{dtype_label}.json').write_text(json.dumps(raw, indent=2) + '\n')
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)
            profiler.export_chrome_trace(profile_path)
        if primary is not None:
            primary.__traceback__ = None
        gc.collect()
        owner.close()

@pytest.mark.parametrize('pages', ['0', '1'])
@pytest.mark.parametrize('material', [torch.float32, torch.bfloat16], indirect=True)
@pytest.mark.parametrize('fault', ['record', 'sync'])
def test_actual_double_fence_failure_survives_abandoned_owner(material, monkeypatch, tmp_path, pages, fault):
    from safetensors import safe_open

    device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=device)
    real_event, real_sync = torch.cuda.Event, torch.cuda.Stream.synchronize
    control = dict(fail=True, failed_drains=0, completed_drains=0, pending_copy=False,
                   native_refs=[], outputs=[], source_dtype=None)
    profiler = None
    profile_path = os.environ.get('PRISMABUILD_PROFILE_TORCH_OUT')
    monkeypatch.setenv('PRISMAQUANT_RELEASE_SOURCE_PAGES', pages)

    class Event:
        def __init__(self):
            self.event = real_event()
        def record(self, observed):
            assert observed == stream
            if fault == 'record':
                raise RuntimeError('original event record qualification failure')
            self.event.record(observed)
        def synchronize(self):
            raise RuntimeError('original event sync qualification failure')

    def synchronize(observed):
        if observed != stream:
            return real_sync(observed)
        assert all(ref() is not None for ref in control['native_refs'])
        if control['fail']:
            control['failed_drains'] += 1
            raise RuntimeError('qualification fatal stream drain failure')
        real_sync(observed)
        assert observed.query()
        control['completed_drains'] += 1

    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda.Stream, 'synchronize', synchronize)

    def abandon():
        owner = _owner(material)
        reference = weakref.ref(owner)
        with pytest.raises(RuntimeError, match='not qualified'):
            owner.require_material_device(device)
        # Direct completion primitive only; the owner's production device gate
        # remains refusing. No source/decoder/authority predicate is weakened.
        path = material['root'] / 'one.safetensors'
        held_fd = None
        try:
            with owner.material_window([path]):
                with owner.safe_open(safe_open, path, framework='pt') as reader:
                    native = reader.get_tensor('w')
                    control['source_dtype'] = str(native.dtype)
                    control['native_refs'].append(weakref.ref(native))
                    held_fd = owner._files['one.safetensors']['fd']
                    with torch.cuda.stream(stream):
                        with ls._SourceCopyCompletion(device, enabled=True, source_owner=owner) as copies:
                            copies.retain(native)
                            # The pending interval must span frame disposal and
                            # GC, not just enqueue: 500M cycles expired before
                            # the first post-GC query in the retained 04 controls.
                            torch.cuda._sleep(10_000_000_000)
                            dependency = real_event()
                            dependency.record(stream)
                            control['outputs'].append(copies.copy(native.to(torch.bfloat16), non_blocking=True))
                            control['pending_copy'] = not dependency.query()
        except RuntimeError as error:
            assert 'original event' in str(error)
            _clear_frames(error)
        return reference, held_fd

    try:
        assert not cc._FAILED_ORIGINAL_COPY_OWNERS
        if profile_path:
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA])
            profiler.__enter__()
        reference, held_fd = abandon()  # Only weakref/integer escape; hooks hold no owner.
        gc.collect()
        assert control['pending_copy'], 'no actual asynchronous H2D interval'
        assert not stream.query(), 'no outstanding DMA after every owner/error reference was dropped'
        assert reference() is not None and all(ref() is not None for ref in control['native_refs'])
        held_bytes = reference().material_live_bytes
        assert held_bytes == len(material['raws']['one.safetensors'])
        assert os.fstat(held_fd).st_size == held_bytes
        receipt = reference().receipt()
        for _ in range(2):
            with pytest.raises(RuntimeError, match='fatal stream drain') as failed:
                cc.CaptureSourceAuthentication.close_failed_original_copies()
            _clear_frames(failed.value)
            failed = None
            gc.collect()
            assert reference() is not None and not reference()._closed
            assert reference().material_live_bytes == held_bytes and os.fstat(held_fd).st_size == held_bytes
        control['fail'] = False
        assert cc.CaptureSourceAuthentication.close_failed_original_copies() == 1
        assert stream.query() and control['completed_drains'] == 1 and control['failed_drains'] == 3
        assert not cc._FAILED_ORIGINAL_COPY_OWNERS
        expected = torch.arange(32).reshape(4, 8).to(torch.bfloat16)
        assert all(torch.equal(output.cpu(), expected) for output in control['outputs'])
        gc.collect()
        assert reference() is None and all(ref() is None for ref in control['native_refs'])
        with pytest.raises(OSError):
            os.fstat(held_fd)
        out = os.environ.get('PRISMAQUANT_ORIGINAL_CUDA_QUALIFICATION_OUT')
        if out:
            destination = Path(out)
            destination.mkdir(parents=True, exist_ok=True)
            dtype = control['source_dtype']
            raw = dict(schema='prismaquant.original_copy_cuda_control.v2', case='abandoned-owner-' + fault,
                       pages=pages, source_dtype=dtype, source_receipt_before_recovery=receipt,
                       held_fd_bytes=held_bytes, pending_after_copy=True, pending_after_owner_disposal=True,
                       failed_drains=3, completed_drains=1, exact_output_parity=True,
                       final_owner_collected=True, final_fd_closed=True,
                       source_control_override='direct completion primitive; production owner device gate closed',
                       automatic_capture_qualified=False, actual_glm=False,
                       torch=str(torch.__version__), cuda=torch.version.cuda)
            (destination / f'abandoned-owner-{fault}-pages{pages}-{dtype}.json').write_text(json.dumps(raw, indent=2) + '\n')
    finally:
        control['fail'] = False
        cc.CaptureSourceAuthentication.close_failed_original_copies()
        if profiler is not None:
            profiler.__exit__(None, None, None)
            profiler.export_chrome_trace(profile_path)