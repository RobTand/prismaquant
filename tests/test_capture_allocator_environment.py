"""A bounded capture must release completed CPU allocations before its next layer."""
import json

import pytest

from test_tessera_campaign_container import dispatch, spec


@pytest.mark.parametrize('container', [False, True])
@pytest.mark.parametrize('flags', [
    ['--streaming-capture-policy', 'shared-inputs-bounded-v1'],
    ['--streaming-capture-policy=shared-inputs-bounded-v1'],
])
def test_capture_manifest_seals_immediate_host_purge_before_python_starts(container, flags):
    data = spec()
    if not container:
        del data['container']
    original_env = dict(data['env'])
    row = dispatch._row(data, ['--streaming', '--capture-calibration-out', '/capture', *flags],
                        mem_gb=104, timeout_s=86400)
    assert row['env']['MIMALLOC_PURGE_DELAY'] == '0'
    assert row['env']['PRISMAQUANT_RELEASE_SOURCE_PAGES'] == '1'
    assert data['env'] == original_env
    if container:
        assert json.loads(row['argv'][4])['env'] == row['env']


@pytest.mark.parametrize('name,value', [('MIMALLOC_PURGE_DELAY', '10'),
                                       ('PRISMAQUANT_RELEASE_SOURCE_PAGES', '0')])
def test_explicit_incompatible_capture_environment_refuses(name, value):
    data = spec()
    data['env'][name] = value
    with pytest.raises(RuntimeError, match=name):
        dispatch._row(data, ['--streaming-capture-policy', 'shared-inputs-bounded-v1'],
                      mem_gb=104, timeout_s=86400)


def test_direct_cuda_capture_cannot_claim_missing_allocator_policy(monkeypatch):
    from prismaquant.autoscale import require_bounded_capture_environment
    with pytest.raises(RuntimeError, match='MIMALLOC_PURGE_DELAY'):
        require_bounded_capture_environment({'PRISMAQUANT_RELEASE_SOURCE_PAGES': '1'})
    require_bounded_capture_environment({'PRISMAQUANT_RELEASE_SOURCE_PAGES': '1',
                                        'MIMALLOC_PURGE_DELAY': '0'})


def test_legacy_manifest_keeps_explicit_allocator_environment():
    data = spec()
    data['env']['MIMALLOC_PURGE_DELAY'] = '10'
    row = dispatch._row(data, ['--streaming-capture-policy', 'legacy'], mem_gb=104, timeout_s=60)
    assert row['env'] == data['env']
