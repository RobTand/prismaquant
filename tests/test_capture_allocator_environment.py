"""A bounded capture must release completed CPU allocations before its next layer."""
from prismaquant import tessera_campaign_rows as campaign_rows
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
    row = campaign_rows.prismabuild_row(data, ['--streaming', '--capture-calibration-out', '/capture', *flags],
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
        campaign_rows.prismabuild_row(data, ['--streaming-capture-policy', 'shared-inputs-bounded-v1'],
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
    row = campaign_rows.prismabuild_row(data, ['--streaming-capture-policy', 'legacy'], mem_gb=104, timeout_s=60)
    assert row['env'] == data['env']


def test_a_capture_policy_flag_with_no_value_refuses_before_publication():
    """Planning keeps its row context when execution syntax refuses argv."""
    data = spec()
    data['env']['MIMALLOC_PURGE_DELAY'] = '10'
    with pytest.raises(RuntimeError, match='campaign row: argument --streaming-capture-policy: expected one argument'):
        campaign_rows.row_is_bounded(['--streaming-capture-policy'])
    with pytest.raises(RuntimeError, match="row class 'default': argument --streaming-capture-policy: expected one argument"):
        campaign_rows.prismabuild_row(data, ['--streaming-capture-policy'], mem_gb=104, timeout_s=60)
    assert data['env']['MIMALLOC_PURGE_DELAY'] == '10'


def test_invalid_shared_argv_names_the_spec(tmp_path):
    data = spec()
    data['campaign_argv'] = ['--streaming-capture-policy']
    path = tmp_path / 'campaign.json'
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError) as refused:
        campaign_rows.load_spec(path)
    assert str(refused.value).startswith(f'{path}: campaign_argv: ')
    assert '--streaming-capture-policy: expected one argument' in str(refused.value)


def test_invalid_class_argv_names_the_spec_and_class(tmp_path):
    data = spec()
    data['classes'] = {'capture': {'argv': ['--streaming-capture-policy']}}
    path = tmp_path / 'campaign.json'
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError) as refused:
        campaign_rows.load_spec(path)
    assert str(refused.value).startswith(f"{path}: row class 'capture' declares ")
    assert "declares ['argv']" in str(refused.value)


def test_invalid_row_argv_names_its_class():
    data = spec()
    data['classes'] = {'capture': {'cpus': 1}}
    with pytest.raises(RuntimeError, match="row class 'capture': argument --streaming-capture-policy: expected one argument"):
        campaign_rows.prismabuild_row(
            data, ['--streaming-capture-policy'], mem_gb=104, timeout_s=60,
            row_class_name='capture')


def test_class_container_validation_uses_shared_capture_argv(tmp_path):
    data = spec()
    data['campaign_argv'] = ['--streaming-capture-policy', 'shared-inputs-bounded-v1']
    data['classes'] = {'capture': {'env': {'MIMALLOC_PURGE_DELAY': '10'}}}
    path = tmp_path / 'campaign.json'
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='MIMALLOC_PURGE_DELAY'):
        campaign_rows.load_spec(path)
