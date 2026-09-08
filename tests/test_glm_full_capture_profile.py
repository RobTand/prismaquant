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
