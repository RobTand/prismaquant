"""CPU checks for instrumentation; these do not qualify native anchor bytes."""
import copy

import pytest
import torch

from experiments.joint_qualification_profile import (
    OwnershipObserver, require_parity, select_groups,
)


def test_weak_storage_observer_counts_alias_owner_after_tensor_dies():
    observer = OwnershipObserver('window', torch.device('cpu'))
    tensor = torch.ones(128)
    alias = tensor[:1]
    observer.register('H', 'unit', tensor)
    assert observer.live()['bytes'] == {'H': 512}
    del tensor
    assert observer.live()['counts'] == {'H': 1}
    del alias
    assert observer.live()['counts'] == {}
    assert observer.peak_bytes == {'H': 512}


def test_group_selection_preserves_whole_packed_group_and_dense_group():
    groups = {'u:dense': ['model.layers.0.a', 'model.layers.0.b'],
              'stack:packed': [f'model.layers.1.mlp.experts.{i}.gate_proj' for i in range(4)],
              'u:small': ['model.layers.0.small']}
    shapes = {name: [256, 256] for names in groups.values() for name in names}
    shapes['model.layers.0.b'] = [256, 512]
    shapes['model.layers.0.small'] = [32, 32]
    chosen = select_groups({'anchor_groups': groups, 'unit_shapes': shapes})
    assert chosen == [{'key': key, 'members': groups[key]} for key in ('u:dense', 'stack:packed')]
    chosen[1]['members'].pop()
    assert len(groups['stack:packed']) == 4


def test_group_selection_refuses_to_manufacture_missing_packed_scope():
    with pytest.raises(RuntimeError, match='dense and packed'):
        select_groups({'anchor_groups': {'u:dense': ['model.layers.0.a']},
                       'unit_shapes': {'model.layers.0.a': [256, 256]}})


@pytest.mark.parametrize('field', ['verified_cells', 'verified_cells_sha256',
    'candidate_roster', 'formats_by_qname', 'initialization_source_read_counts',
    'source_layer_load_counts', 'source_forward_count'])
def test_parity_requires_records_roster_and_real_source_loads(field):
    reference = dict(verified_cells=[{'unit': 'a', 'record': {'source': 'content-bound'}}],
        verified_cells_sha256='actual digest checked elsewhere', candidate_roster=[['a', 'fmt']],
        formats_by_qname={'a': ['fmt', 'BF16']}, initialization_source_read_counts={'visual': 1},
        source_layer_load_counts={'layer0': 1},
        source_forward_count=0)
    require_parity(reference, copy.deepcopy(reference))
    changed = copy.deepcopy(reference)
    changed[field] = 'changed'
    with pytest.raises(RuntimeError, match=field):
        require_parity(reference, changed)
