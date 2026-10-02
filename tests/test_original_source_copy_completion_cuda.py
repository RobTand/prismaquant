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

pytestmark = [pytest.mark.skipif(
    os.environ.get('PRISMAQUANT_ORIGINAL_CUDA_QUALIFICATION_TEST') != '1'
    or not torch.cuda.is_available(),
    reason='requires explicit root-approved PB CUDA qualification; a skip proves nothing')]


@pytest.mark.parametrize('pages', ['0', '1'])
@pytest.mark.parametrize('material', [torch.float32, torch.bfloat16], indirect=True)
@pytest.mark.parametrize('case', ['layer', 'read-failure', 'copy-failure', 'cancel',
                                 'event-record-failure', 'event-sync-failure', 'head', 'dequant'])
def test_actual_original_copy_stream_ownership(material, monkeypatch, tmp_path, pages, case):
    owner = _owner(material)
    real_enter = cc._CaptureSourceSafeOpen.__enter__
    real_get = cc._CaptureSourceSafeOpen.get_tensor
    real_to = torch.Tensor.to
    real_event = torch.cuda.Event
    local = threading.local()
    observed = dict(copies=0, pending_events=0, completed_events=0, decoder_reads=0,
                    credit_before_sync=[], credit_after_sync=[], pending_stream_on_failure=False,
                    source_dtype=None, delayed_copies_pending_on_return=0)
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
        monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '1')

        def enter(reader):
            value = real_enter(reader)
            local.native = []
            # A real independent copy stream exercises the same current-stream
            # API that production readers use; no fake output/Event is installed.
            stream = torch.cuda.Stream(device=device)
            torch.cuda.set_stream(stream)
            return value

        def get(reader, key):
            if case == 'read-failure' and observed['decoder_reads']:
                raise RuntimeError('qualification second-payload failure')
            value = real_get(reader, key)
            assert value.device.type == 'cpu'
            assert observed['source_dtype'] in (None, str(value.dtype))
            observed['source_dtype'] = str(value.dtype)
            local.native.append(weakref.ref(value))
            observed['decoder_reads'] += 1
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
                observed['copies'] += 1
                if case == 'cancel':
                    cancel.set()
            if target is not None:
                result = real_to(value, target, *args, **kwargs)
            else:
                result = real_to(value, **kwargs)
            if delay_finished is not None and not delay_finished.query():
                # The H2D copy follows this still-pending real event on the
                # same stream. An event pending only after a synchronous copy
                # is insufficient evidence for the source lifetime interval.
                observed['delayed_copies_pending_on_return'] += 1
            if case == 'copy-failure' and observed['copies'] == 2:
                raise RuntimeError('qualification failure after real copy enqueue')
            return result

        class ObservedEvent:
            def __init__(self):
                self.event = real_event()
            def record(self, stream):
                self.native = list(local.native)
                if case == 'event-record-failure':
                    observed['pending_stream_on_failure'] = not stream.query()
                    raise RuntimeError('qualification injected event record failure')
                self.event.record(stream)
                if not self.event.query():
                    observed['pending_events'] += 1
            def synchronize(self):
                assert all(ref() is not None for ref in self.native)
                held = owner.material_live_bytes
                assert held == len(material['raws']['one.safetensors'])
                observed['credit_before_sync'].append(held)
                if case == 'event-sync-failure':
                    observed['pending_stream_on_failure'] = not self.event.query()
                    raise RuntimeError('qualification injected event sync failure')
                self.event.synchronize()
                assert self.event.query()
                observed['completed_events'] += 1
                observed['credit_after_sync'].append(owner.material_live_bytes)

        monkeypatch.setattr(cc._CaptureSourceSafeOpen, '__enter__', enter)
        monkeypatch.setattr(cc._CaptureSourceSafeOpen, 'get_tensor', get)
        monkeypatch.setattr(torch.Tensor, 'to', transfer)
        monkeypatch.setattr(torch.cuda, 'Event', ObservedEvent)
        if profile_path:
            profiler = torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            profiler.__enter__()
        path = str(owner.root / 'one.safetensors')
        if case in ('layer', 'read-failure', 'copy-failure', 'cancel',
                    'event-record-failure', 'event-sync-failure'):
            names = ['layer.a', 'layer.b']
            try:
                output = ls._read_layer_to_device(
                    'layer.', {n: path for n in names}, {n: 'w' for n in names},
                    torch.bfloat16, device, source_authentication=owner,
                    cancel=cancel if case == 'cancel' else None)
            except (RuntimeError, CancelledError) as error:
                primary = error
                if case == 'read-failure':
                    assert 'second-payload' in str(error)
                elif case == 'cancel':
                    assert isinstance(error, CancelledError)
                elif case == 'copy-failure':
                    assert 'real copy enqueue' in str(error)
                elif case in ('event-record-failure', 'event-sync-failure'):
                    assert 'injected event' in str(error)
                    assert owner.material_live_bytes > 0
                    with pytest.raises(RuntimeError, match='live readers or consumers'):
                        owner.close()
                    # An actual failed completion proof retains aliases. The
                    # qualification harness separately drains that real stream
                    # before releasing the error frame; production did not
                    # declare this failed invocation successful.
                    torch.cuda.current_stream(device).synchronize()
                    assert owner.material_live_bytes > 0
                else:
                    raise
            else:
                assert case == 'layer', 'failure/cancellation returned installable output'
                expected = torch.arange(32).reshape(4, 8).to(torch.bfloat16)
                assert all(torch.equal(value.cpu(), expected) for value in output.values())
                output = None
        elif case == 'head':
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
        if case in ('event-record-failure', 'event-sync-failure'):
            assert observed['pending_stream_on_failure']
            assert observed['completed_events'] == 0
        else:
            assert observed['completed_events'] > 0
            # A synchronous/unsupported pending-copy branch does not qualify
            # the asynchronous interval merely because values match.
            assert observed['pending_events'] > 0, 'no pending copy-stream completion observed'
        if primary is not None:
            primary.__traceback__ = None
            primary = None
        gc.collect()
        assert owner.material_live_bytes == 0
        # The dtype is read from the actual decoded source, not inferred from
        # a filename. The parameter is part of every exported control identity.
        source_dtype = observed['source_dtype']
        assert source_dtype in ('torch.bfloat16', 'torch.float32')
        raw = dict(schema='prismaquant.original_copy_cuda_control.v1', case=case, pages=pages,
                   source_dtype=source_dtype,
                   original_owner_type=type(owner).__name__, device=str(device),
                   source_control_override='fixture-owner device predicate only; production remains closed',
                   automatic_capture_qualified=False, actual_glm=False, observations=observed,
                   source_receipt=owner.receipt(), torch=str(torch.__version__), cuda=torch.version.cuda)
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
