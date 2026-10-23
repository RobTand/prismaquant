"""The capture qualification tool settles its actual periodic sampler."""
from contextlib import nullcontext
import json
from pathlib import Path
from threading import Event

import pytest

from experiments import workspace_netdata
from prismaquant import io_spans
from tools.verify_tessera_calibration_forward import _record_netdata


@pytest.mark.parametrize('body_fails', [False, True], ids=['success', 'failure'])
@pytest.mark.parametrize('unavailable', [False, True], ids=['healthy', 'unavailable'])
def test_netdata_sampling_settles_before_status(tmp_path, monkeypatch, unavailable, body_fails):
    second_host_entered, release_second_host = Event(), Event()
    events, owners, joins = [], [], []
    real_sampler = io_spans.PeriodicSampler

    def sampler(tick, **kwargs):
        assert kwargs == dict(interval_s=1, name='capture-netdata', tick_first=True)
        owner = real_sampler(tick, **kwargs)
        owners.append(owner)
        request_stop, join = owner.request_stop, owner.join

        def stop():
            events.append('stop')
            request_stop()
            release_second_host.set()

        def settled_join(timeout=None):
            joins.append(timeout)
            return join(timeout)

        owner.request_stop = stop
        owner.join = settled_join
        return owner

    def read(host):
        events.append(host)
        if host == 'sparky' and unavailable:
            raise OSError('host unavailable')
        if host == 'sparklina':
            second_host_entered.set()
            assert release_second_host.wait(10), 'sampler stop did not release the host read'
            events.append('read-complete')
        return dict(host=host, time=1.0)

    monkeypatch.setattr(io_spans, 'PeriodicSampler', sampler)
    monkeypatch.setattr(workspace_netdata, 'sample_netdata', read)
    samples, errors = [], []
    expected_failure = pytest.raises(RuntimeError, match='capture failed') if body_fails else nullcontext()
    with expected_failure:
        with _record_netdata(tmp_path, samples, errors):
            assert second_host_entered.wait(10), 'the immediate first round never began'
            assert not (tmp_path / 'instrument-status.json').exists()
            if body_fails:
                raise RuntimeError('capture failed')

    assert events == ['sparky', 'sparklina', 'stop', 'read-complete']
    assert joins == [20]
    assert len(owners) == 1 and not owners[0].is_alive()
    expected_samples = [('sparklina', 1.0)] if unavailable else [('sparky', 1.0), ('sparklina', 1.0)]
    expected_errors = ['sparky: OSError: host unavailable'] if unavailable else []
    assert samples == expected_samples
    assert errors == expected_errors
    records = [json.loads(line) for line in (tmp_path / 'netdata-both-boxes.jsonl').read_text().splitlines()]
    assert records == [dict(host=host, time=at) for host, at in expected_samples]
    status = json.loads((tmp_path / 'instrument-status.json').read_text())
    assert status == dict(samples=[list(sample) for sample in expected_samples], errors=expected_errors)


def test_timed_out_sample_keeps_its_file_until_the_inflight_write_finishes(tmp_path, monkeypatch):
    second_host_entered, release_second_host = Event(), Event()
    owners, joins, handles = [], [], []
    real_sampler, real_open = io_spans.PeriodicSampler, Path.open

    def sampler(tick, **kwargs):
        owner = real_sampler(tick, **kwargs)
        owners.append((owner, owner.join))
        owner.join = lambda timeout=None: joins.append(timeout)
        return owner

    def opened(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        if path == tmp_path / 'netdata-both-boxes.jsonl':
            handles.append(handle)
        return handle

    def read(host):
        if host == 'sparklina':
            second_host_entered.set()
            assert release_second_host.wait(10), 'the inflight host read was not released'
        return dict(host=host, time=1.0)

    monkeypatch.setattr(io_spans, 'PeriodicSampler', sampler)
    monkeypatch.setattr(workspace_netdata, 'sample_netdata', read)
    monkeypatch.setattr(Path, 'open', opened)
    samples, errors = [], []
    try:
        with _record_netdata(tmp_path, samples, errors):
            assert second_host_entered.wait(10), 'the first paired sample never began'
        assert joins == [20]
        assert owners[0][0].is_alive()
        assert not handles[0].closed
        status = json.loads((tmp_path / 'instrument-status.json').read_text())
        assert status == dict(samples=[['sparky', 1.0]], errors=['Netdata sampler did not finish'])
    finally:
        release_second_host.set()
        if owners:
            owners[0][1](timeout=10)
    assert not owners[0][0].is_alive()
    assert handles[0].closed
    assert samples == [('sparky', 1.0), ('sparklina', 1.0)]
    assert errors == ['Netdata sampler did not finish']
