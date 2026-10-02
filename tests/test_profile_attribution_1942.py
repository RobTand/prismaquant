"""A completed wrapper is not evidence of sampled workload execution."""
import json

import pytest

from tools.pq_profile_artifact import publish_profile


def trace(tmp_path, *, process=42, samples=None):
    source = tmp_path / 'child-profile.speedscope'
    data = {
        '$schema': 'https://www.speedscope.app/file-format-schema.json',
        'shared': {'frames': [{'name': 'work', 'file': '/work.py', 'line': 1}]},
        'profiles': [{'type': 'sampled', 'name': f'Process {process} Thread {process} "work"',
                      'unit': 'seconds', 'startValue': 0, 'endValue': 1,
                      'samples': [[0]] if samples is None else samples, 'weights': [1]}],
    }
    source.write_text(json.dumps(data))
    source.with_name(source.name + '.workload-status.json').write_text(json.dumps({
        'schema': 'prismaquant.profile_workload_status.v1', 'returncode': 0,
        'workload_pid': 42, 'profiler_returncode': 0,
    }))
    return source, tmp_path / 'published.speedscope'


def test_workload_attributed_profile_is_a_real_positive(tmp_path):
    source, destination = trace(tmp_path)
    original = source.read_bytes()
    result = publish_profile(source, destination)
    assert result['bytes'] == len(original)
    assert result['workload_pid'] == 42 and result['workload_samples'] == 1
    assert destination.read_bytes() == original


def test_wrapper_only_samples_cannot_qualify_workload(tmp_path):
    source, destination = trace(tmp_path, process=17)
    with pytest.raises(RuntimeError, match='workload|attribut'):
        publish_profile(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize('samples', ['not stacks', [[8]]])
def test_malformed_sampled_profile_cannot_be_published(tmp_path, samples):
    source, destination = trace(tmp_path, samples=samples)
    with pytest.raises(RuntimeError, match='sample|frame'):
        publish_profile(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize('field,value', [
    ('type', 'evented'), ('name', 'Thread 42 "anonymous"'),
    ('unit', 'bytes'), ('startValue', True), ('endValue', float('nan')),
    ('weights', []), ('weights', [float('inf')]), ('weights', [True]),
    ('samples', [[True]]), ('samples', [[-1]]),
])
def test_invalid_profile_contract_never_publishes(tmp_path, field, value):
    source, destination = trace(tmp_path)
    data = json.loads(source.read_text())
    data['profiles'][0][field] = value
    source.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        publish_profile(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize('pid', [None, True, 0, -1, '42'])
def test_missing_or_invalid_namespace_pid_never_qualifies(tmp_path, pid):
    source, destination = trace(tmp_path)
    status = source.with_name(source.name + '.workload-status.json')
    record = json.loads(status.read_text())
    record['workload_pid'] = pid
    status.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='completion record'):
        publish_profile(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize('profiler_rc', [None, True, 1, -9, '0'])
def test_failed_or_missing_profiler_cannot_publish_successful_workload(tmp_path, profiler_rc):
    source, destination = trace(tmp_path)
    status = source.with_name(source.name + '.workload-status.json')
    record = json.loads(status.read_text())
    if profiler_rc is None:
        del record['profiler_returncode']
    else:
        record['profiler_returncode'] = profiler_rc
    status.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='profiler did not complete'):
        publish_profile(source, destination)
    assert not destination.exists()


def test_parent_and_workload_samples_are_counted_separately(tmp_path):
    source, destination = trace(tmp_path)
    data = json.loads(source.read_text())
    parent = dict(data['profiles'][0], name='Process 17 Thread 17 "parent"')
    data['profiles'].insert(0, parent)
    source.write_text(json.dumps(data))
    result = publish_profile(source, destination)
    assert result['profiles'] == 2 and result['workload_samples'] == 1
