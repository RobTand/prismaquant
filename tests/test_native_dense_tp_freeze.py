"""CPU contract fixtures for Tessera #639; no native timing or GPU evidence."""
import copy

import pytest

from prismaquant.joint_aura import identity_sha256
from prismaquant.native_execution_binding import execution_panel_from_joint, freeze_execution_panel
from prismaquant.native_operator_panel import EXECUTION, freeze_native_panel
from test_native_operator_panel import joined  # noqa: F401 -- fixture


def tp2_inputs(joined, axis, rank):
    inputs, preflight, row = copy.deepcopy(joined)
    inputs['execution'] = {**EXECUTION, 'tensor_parallel': 2, 'tensor_parallel_cut_axis': axis}
    inputs['distributed'] = {'world_size': 2, 'rank': rank,
                             'init_method': 'tcp://fixture:29517', 'timeout_seconds': 120}
    full_shape = [8, 4] if axis == 'output' else [4, 8]
    inputs['wire']['record']['identity'] = {'source': {'shape': full_shape}}
    preflight['operator']['wire_record_sha256'] = identity_sha256(inputs['wire']['record'])
    preflight['operator']['scheme'] = {'rows': full_shape[0], 'columns': full_shape[1],
                                      'roles': [['weight', full_shape[0]]]}
    preflight['scheme_sha256'] = identity_sha256(preflight['operator']['scheme'])
    preflight['runtime'].update(execution=copy.deepcopy(inputs['execution']),
                                 distributed=copy.deepcopy(inputs['distributed']),
                                 gpu={'uuid': f'CPU-FIXTURE-RANK-{rank}'})
    preflight['runtime_sha256'] = identity_sha256(preflight['runtime'])
    return inputs, preflight, row


def freeze(parts, kind):
    inputs, preflight, row = parts
    if kind == 'joint':
        return freeze_native_panel(*parts, cost_sha256='4' * 64)
    return freeze_execution_panel(inputs, preflight,
        source_sha256=row['probe_identity']['source_model']['content_sha256'])


@pytest.mark.parametrize('axis', ['input', 'output'])
@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('kind', ['joint', 'execution'])
def test_tp2_panel_freezes_independent_world_cut_and_rank(joined, axis, rank, kind):
    parts = tp2_inputs(joined, axis, rank)
    before = copy.deepcopy(parts)
    panel = freeze(parts, kind)
    assert panel['execution'] == parts[0]['execution']
    assert panel['runtime']['distributed'] == parts[0]['distributed']
    assert panel['runtime']['gpu']['uuid'] == f'CPU-FIXTURE-RANK-{rank}'
    assert panel['shape'] == [4, 4]  # rank-local, wire retains full shape
    assert panel['wire'] == parts[0]['wire']
    assert parts == before
    assert execution_panel_from_joint(freeze(parts, 'joint')) == freeze(parts, 'execution')


@pytest.mark.parametrize('mutation', [
    lambda i, p: i.pop('execution'),
    lambda i, p: i.pop('distributed'),
    lambda i, p: i['execution'].update(tensor_parallel=True),
    lambda i, p: i['execution'].update(tensor_parallel=3),
    lambda i, p: i['execution'].update(tensor_parallel=2.0),
    lambda i, p: i['execution'].update(tensor_parallel_cut_axis='guess'),
    lambda i, p: i['execution'].update(bias=True),
    lambda i, p: i['execution'].update(unexpected='field'),
    lambda i, p: i['distributed'].update(rank=True),
    lambda i, p: i['distributed'].update(rank=2),
    lambda i, p: i['distributed'].update(world_size=1),
    lambda i, p: i['distributed'].update(timeout_seconds=0),
    lambda i, p: i['distributed'].update(init_method='file:///tmp/rendezvous'),
    lambda i, p: p['runtime']['distributed'].update(rank=1),
    lambda i, p: p['runtime']['execution'].update(tensor_parallel_cut_axis='input'),
    lambda i, p: i.update(shape=[8, 4]),
    lambda i, p: i['wire']['record']['identity']['source'].update(shape=[4, 4]),
    lambda i, p: p['operator']['scheme'].update(rows=4),
    lambda i, p: i['phases']['prefill']['reference_output'].update(shape=[1, 8]),
])
@pytest.mark.parametrize('kind', ['joint', 'execution'])
def test_tp2_freezer_refuses_bad_or_unbound_coordinates(joined, mutation, kind):
    parts = tp2_inputs(joined, 'output', 0)
    mutation(parts[0], parts[1])
    # Rehash producer facts so the semantic binding, not a stale digest, refuses.
    parts[1]['runtime_sha256'] = identity_sha256(parts[1]['runtime'])
    parts[1]['scheme_sha256'] = identity_sha256(parts[1]['operator']['scheme'])
    parts[1]['operator']['wire_record_sha256'] = identity_sha256(parts[0]['wire']['record'])
    with pytest.raises(ValueError):
        freeze(parts, kind)


def test_explicit_tp1_preserves_legacy_panel(joined):
    original = freeze(joined, 'joint')
    parts = copy.deepcopy(joined)
    parts[0]['execution'] = dict(EXECUTION)
    assert freeze(parts, 'joint') == original
    assert freeze(parts, 'execution') == execution_panel_from_joint(original)
