"""Extended metadata preserves historical layer extents and complete control head."""
import copy
import pytest
from tools.prepare_extended_joint_quanta import extend_parent


def test_extension_parent_only_augments_head_and_keeps_exact_layer_body():
    parent = {'schema': 'prismaquant.prismabuild.data_manifest.v1',
        'entries': [{'path': '/old-plan', 'offset': 0, 'bytes': 10, 'sha256': None},
                    {'path': '/source', 'offset': 0, 'bytes': 30, 'sha256': None}],
        'entry_count': 2, 'total_bytes': 40, 'annotations': {'plan_sha256': 'old',
            'campaign_scope': {'full512': True}, 'layers': [0],
            'phases': [{'name': 'head', 'bytes': 10, 'cumulative_bytes': 10},
                       {'name': 'layer-0', 'bytes': 30, 'cumulative_bytes': 40}]}}
    original = copy.deepcopy(parent)
    new = {'path': '/new-plan', 'sha256': 'new'}
    added = [{'path': '/new-plan', 'offset': 0, 'bytes': 20, 'sha256': None}]
    kwargs = dict(old_plan={'sha256': 'old'}, new_plan=new,
        prepared={'formats_by_qname': {'q': ['A8', 'A4', 'BF16']}}, extension={'path': '/proof'})
    result = extend_parent(parent, added + added, **kwargs)
    assert parent == original
    assert result['entries'][-1] == original['entries'][-1]
    assert result['entry_count'] == 3 and result['total_bytes'] == 60
    assert result['annotations']['phases'] == [
        {'name': 'head', 'bytes': 30, 'cumulative_bytes': 30},
        {'name': 'layer-0', 'bytes': 30, 'cumulative_bytes': 60}]
    assert result['annotations']['campaign_scope'] == {'full512': True}
    assert result['annotations']['executable_prepared_inputs_required'] is True
    assert result['annotations']['measured_cells'] == 2
    parent['annotations']['plan_sha256'] = 'foreign'
    with pytest.raises(ValueError, match='scientific plan'):
        extend_parent(parent, added, **kwargs)
