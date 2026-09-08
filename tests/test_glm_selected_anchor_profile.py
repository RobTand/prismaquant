"""CPU state/ownership checks; native CUDA trace coverage needs the real row."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import glm_full_capture_profile as observe


class FakeProfiler:
    cuda = True
    enter_error = None

    def __enter__(self):
        if self.enter_error:
            raise self.enter_error
        return self

    def __exit__(self, *_):
        return False

    def export_chrome_trace(self, path):
        Path(path).write_text('{"traceEvents":[]}')

    def events(self):
        kind = torch.autograd.DeviceType.CUDA if self.cuda else torch.autograd.DeviceType.CPU
        return [SimpleNamespace(device_type=kind)]

    def key_averages(self):
        return SimpleNamespace(table=lambda **_: 'controlled CPU test CUDA-event marker')


@pytest.fixture
def controlled(monkeypatch):
    seen = []
    def profile(**options):
        seen.append(options)
        return FakeProfiler()
    monkeypatch.setattr(observe.torch.profiler, 'profile', profile)
    monkeypatch.setattr(observe, 'sample_netdata', lambda host: dict(host=host, metrics={}))
    return seen


def observer(path, calls=(0, 2), cap=4096):
    return observe.AnchorObserver(path, profile_calls=calls, trace_max_bytes=cap,
                                  command=['--unchanged-original-command'])


def monitor_readiness(obs, monkeypatch):
    """Signal only after the real monitor has committed its sample count."""
    ready = {kind: threading.Event() for kind in ('netdata', 'python_sampler')}
    original_wait = obs.stopped.wait
    def sampled_wait(timeout=None):
        # monitor() increments samples immediately before waiting on stopped.
        # No sample or output is fabricated, and production startup stays async.
        for kind, event in ready.items():
            if obs.result[kind].get('samples'):
                event.set()
        return original_wait(timeout)
    monkeypatch.setattr(obs.stopped, 'wait', sampled_wait)
    return ready


@pytest.mark.parametrize('delayed_start', [False, True])
def test_exact_original_calls_returns_and_finite_windows(tmp_path, controlled, monkeypatch,
                                                        delayed_start):
    obs = observer(tmp_path)
    ready = monitor_readiness(obs, monkeypatch)
    start = threading.Event()
    if delayed_start:
        original_monitor = obs.monitor
        def delayed_monitor(kind):
            assert start.wait(timeout=10), 'test did not release monitor startup'
            original_monitor(kind)
        monkeypatch.setattr(obs, 'monitor', delayed_monitor)
    token, weight, acts = object(), object(), object()
    seen = []
    def original(**kwargs):
        seen.append(kwargs)
        return token
    with obs:
        start.set()  # Force delayed monitors to start after __enter__ returns.
        for kind, event in ready.items():
            assert event.wait(timeout=10), f'{kind} did not commit its first sample'
        wrapped = obs.wrap_anchor(original)
        for i in range(5):
            assert wrapped(qname=f'unit-{i}', format_name='original-format',
                           weight=weight, activations=acts) is token
    assert len(seen) == 5
    assert all(call['weight'] is weight and call['activations'] is acts for call in seen)
    assert len(controlled) == 2
    assert all(p['activities'] == [torch.profiler.ProfilerActivity.CPU,
                                  torch.profiler.ProfilerActivity.CUDA] for p in controlled)
    assert all(not p['record_shapes'] and not p['profile_memory'] and not p['with_stack']
               for p in controlled)
    result = json.loads((obs.out/'result.json').read_text())
    assert result['status'] == 'complete' and result['native_anchor_profiled']
    assert result['anchor_calls'] == 5
    assert [r['qname'] for r in result['anchors']] == ['unit-0', 'unit-2']
    assert result['netdata']['samples'] and result['python_sampler']['samples']
    host_records = [json.loads(line) for line in (obs.out/'netdata.jsonl').read_text().splitlines()]
    assert {r['host'] for r in host_records} == {'sparky', 'sparklina'}
    assert all((obs.out/r['trace']['path']).is_file() for r in result['anchors'])


@pytest.mark.parametrize('missing', [('netdata',), ('python_sampler',),
                                    ('netdata', 'python_sampler')])
def test_delayed_monitors_refuse_missing_telemetry(tmp_path, controlled, monkeypatch, missing):
    obs = observer(tmp_path, calls=(0,))
    ready = monitor_readiness(obs, monkeypatch)
    original_monitor = obs.monitor
    def delayed_monitor(kind):
        if kind in missing:
            # Deterministically schedule this instrument after shutdown, the
            # interleaving instantaneous fake anchors could previously reach.
            assert obs.stopped.wait(timeout=10), 'observer did not reach shutdown'
        original_monitor(kind)
    monkeypatch.setattr(obs, 'monitor', delayed_monitor)
    token = object()
    journal = []
    with pytest.raises(RuntimeError, match='required profiler evidence'):
        with obs:
            for kind, event in ready.items():
                if kind not in missing:
                    assert event.wait(timeout=10), f'{kind} did not sample'
            journal.append(obs.wrap_anchor(lambda **_: token)(qname='u', format_name='f'))
    assert journal == [token]
    result = json.loads((obs.out/'result.json').read_text())
    assert result['status'] == 'failed' and result['native_anchor_profiled']
    assert result['anchors'][0]['status'] == 'complete'
    assert {row['error'] for row in result['errors']} == {
        repr(RuntimeError(f'no {kind} sample was recorded')) for kind in missing}
    for kind in ready:
        assert bool(result[kind].get('samples')) == (kind not in missing)


@pytest.mark.parametrize('failure', ['no_cuda', 'trace_cap', 'enter'])
def test_observation_failure_preserves_success_for_journaling(tmp_path, controlled, monkeypatch, failure):
    if failure == 'no_cuda':
        monkeypatch.setattr(FakeProfiler, 'cuda', False)
    if failure == 'enter':
        monkeypatch.setattr(FakeProfiler, 'enter_error', RuntimeError('profiler unavailable'))
    obs = observer(tmp_path, calls=(0,), cap=1 if failure == 'trace_cap' else 4096)
    seen, journal = [], []
    token = object()
    def original(**kwargs):
        seen.append(kwargs)
        return token
    with pytest.raises(RuntimeError, match='required profiler evidence'):
        with obs:
            journal.append(obs.wrap_anchor(original)(qname='u', format_name='f'))
    assert len(seen) == 1 and journal == [token]
    result = json.loads((obs.out/'result.json').read_text())
    assert result['status'] == 'failed' and not result['native_anchor_profiled']
    assert result['anchors'][0]['status'] == 'observation_failed'
    if failure == 'trace_cap':
        assert result['anchors'][0]['rejected_trace_bytes'] > 1
        assert not list(obs.out.glob('*.trace.json'))


def test_original_encoder_exception_is_preserved(tmp_path, controlled):
    obs = observer(tmp_path, calls=(0,))
    error = ValueError('original encoder rejection')
    calls = []
    def original(**kwargs):
        calls.append(kwargs)
        raise error
    with pytest.raises(ValueError) as caught:
        with obs:
            obs.wrap_anchor(original)(qname='u', format_name='f')
    assert caught.value is error and len(calls) == 1
    assert json.loads((obs.out/'result.json').read_text())['anchors'][0]['status'] == 'anchor_failed'


def test_retry_keeps_attempt_evidence_and_no_work_is_not_native_proof(tmp_path, controlled):
    first, second = observer(tmp_path), observer(tmp_path)
    assert first.out != second.out
    for item in (first, second):
        with item:
            pass
        result = json.loads((item.out/'result.json').read_text())
        assert result['status'] == 'complete'
        assert result['profile_status'] == 'no_new_anchors' and not result['native_anchor_profiled']


def selected():
    return ['--streaming', '--units', 'units.json', '--calibration-cache', 'capture.json',
            '--calibration-cache-sha256', 'a'*64, '--anchor-batch-size', '1']


@pytest.mark.parametrize('extra', [
    ['--anchor-batch-size', '2'], ['--capture-calibration-out', 'new'], ['--census-out', 'new']])
def test_selected_mode_refuses_changed_capture_or_non_scalar_work(extra):
    with pytest.raises(ValueError, match='selected canonical reuse'):
        observe.selected_anchor_command(selected()+extra)


def test_entrypoint_keeps_original_argv_and_restores_method(tmp_path, controlled, monkeypatch):
    from prismaquant import tessera_campaign as campaign
    original = campaign._measure_anchor
    command = selected()
    seen = []
    def main(args):
        seen.append(args)
        assert campaign._measure_anchor is not original
        return 0  # A fully journaled retry has no new anchor to observe.
    monkeypatch.setattr(campaign, 'main', main)
    monkeypatch.setattr(observe.torch.cuda, 'is_available', lambda: True)
    assert observe.main(['--evidence-out', str(tmp_path), '--selected-anchors',
        '--anchor-profile-calls', '0,31', '--anchor-trace-max-bytes', '4096', '--', *command]) == 0
    assert seen == [command] and campaign._measure_anchor is original


@pytest.mark.parametrize('calls', [(), (1,), (0, 0), (0, -1), (0, 1, 2, 3, 4)])
def test_window_configuration_is_finite_and_has_first_call(tmp_path, calls):
    with pytest.raises(ValueError, match='at most four'):
        observer(tmp_path, calls=calls)
