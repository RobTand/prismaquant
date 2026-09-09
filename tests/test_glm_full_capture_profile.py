import json

import pytest
import torch

from experiments.glm_full_capture_profile import CaptureObserver
from prismaquant.tessera_campaign import _collect_activations


def test_observer_keeps_exact_cuda_capture_and_original_batch_order(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('native CUDA profiler qualification')
    torch.manual_seed(917)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3)).cuda()
    batches = [torch.randn(1, 7, 4) for _ in range(34)]
    baseline = _collect_activations(model, ['0'], batches, 11, 'cuda',
                                    want_hessian=True, forward_batch=model)
    seen = []
    def forward(batch):
        seen.append(batch.detach().cpu().clone())
        return model(batch)
    with CaptureObserver(tmp_path/'observed', profile_layers=(0,)) as observer:
        actual = observer.wrap_collector(_collect_activations)(
            model, ['0'], batches, 11, 'cuda', want_hessian=True,
            forward_batch=forward)
    assert len(seen) == len(batches)
    assert all(torch.equal(a, b) for a, b in zip(seen, batches))
    for old, new in zip(baseline[:2], actual[:2]):
        assert set(old) == set(new)
        assert all(torch.equal(old[name], new[name]) for name in old)
    assert baseline[2:] == actual[2:]
    result = json.loads((tmp_path/'observed/result.json').read_text())
    assert result['status'] == 'complete'
    assert result['netdata']['samples'] >= 1
    assert result['python_sampler']['samples'] >= 1
    assert [t['after_batches'] for t in result['collections'][0]['traces']] == [2, 33]
    assert all(t['bytes'] > 0 for t in result['collections'][0]['traces'])


def test_observer_preserves_forward_return_and_partial_failure(tmp_path):
    observer = CaptureObserver(tmp_path/'partial', profile_layers=())
    token = object()
    def forward(batch):
        return token
    def broken(*args, forward_batch):
        assert forward_batch(None) is token
        raise ValueError('original forward failure')
    with pytest.raises(ValueError, match='original forward failure'):
        with observer:
            observer.wrap_collector(broken)(forward_batch=forward)
    result = json.loads((tmp_path/'partial/result.json').read_text())
    assert result['status'] == 'failed'
    assert result['collections'][0]['batches'] == 1
    assert 'original forward failure' in result['campaign_error']

    assert json.loads((tmp_path/'partial/progress.json').read_text()) == result


def test_netdata_failure_keeps_other_host_and_later_samples(tmp_path, monkeypatch):
    from experiments import glm_full_capture_profile as module
    observer = CaptureObserver(tmp_path/'sample-gap', profile_layers=())
    calls = []
    rounds = 0

    def sample(host):
        calls.append(host)
        if len(calls) == 1:
            raise RuntimeError('Netdata required GPU power chart missing')
        return {'host': host, 'metrics': {'observed': True}}

    def wait(_seconds):
        nonlocal rounds
        rounds += 1
        if rounds == 2:
            observer.stopped.set()
        return observer.stopped.is_set()

    monkeypatch.setattr(module, 'sample_netdata', sample)
    monkeypatch.setattr(observer.stopped, 'wait', wait)
    observer.monitor('netdata')
    assert calls == ['sparky', 'sparklina', 'sparky', 'sparklina']
    records = [json.loads(line) for line in (observer.out/'netdata.jsonl').read_text().splitlines()]
    assert [row['host'] for row in records] == ['sparklina', 'sparky', 'sparklina']
    assert observer.result['netdata']['samples'] == 2
    # A lost sample remains an explicit incomplete-evidence failure, even when
    # subsequent samples were retained. Collection recovery is not gap erasure.
    with pytest.raises(RuntimeError, match='required profiler evidence'):
        observer.__exit__(None, None, None)
    result = json.loads((observer.out/'result.json').read_text())
    assert result['status'] == 'failed'
    assert result['errors'][0]['host'] == 'sparky'
    assert 'GPU power chart missing' in result['errors'][0]['error']


@pytest.mark.parametrize('kind,first_key', [('netdata', 'hosts'), ('python_sampler', 'scope')])
def test_snapshot_survives_first_monitor_round_during_dict_iteration(tmp_path, monkeypatch, kind, first_key):
    from experiments import glm_full_capture_profile as module
    observer = CaptureObserver(tmp_path/kind, profile_layers=())
    def sample(host):
        if host == 'sparky':
            raise RuntimeError('sample gap')
        return {'host': host}
    monkeypatch.setattr(module, 'sample_netdata', sample)
    monkeypatch.setattr(observer.stopped, 'wait', lambda _: observer.stopped.set())
    chunks = json.JSONEncoder(indent=2).iterencode(observer.result)
    prefix = []
    for chunk in chunks:
        prefix.append(chunk)
        if chunk == json.dumps(first_key):
            break
    else:
        pytest.fail('snapshot never entered the monitor-owned dictionary')
    # Force the real scheduling interleaving: JSON's dict iterator is alive
    # while a first monitor round publishes counters and a first host failure.
    observer.monitor(kind)
    decoded = json.loads(''.join(prefix) + ''.join(chunks))
    assert decoded[kind]['samples'] == 1
    if kind == 'netdata':
        assert decoded[kind]['sample_failures']['sparky']['failed_samples'] == 1
        # Earlier fields can precede this monitor round in a progress snapshot.
        # The final joined snapshot still carries the retained error.
        assert observer.result['errors'][0]['host'] == 'sparky'
