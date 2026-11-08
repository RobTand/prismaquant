"""Finite application documents, without native execution or runtime measurement."""
import copy
import hashlib
import json
import os

import pytest

from prismaquant import quality_prefill_contract as scientific
from prismaquant import quality_prefill_flux as prefill


def payload():
    return dict(schema=scientific.TASK_PAYLOAD_SCHEMA, serving_unit='layer.0/up+gate',
                members=['layer.0/up', 'layer.0/gate'], family='e4m3', rate=832,
                context_id='prefill-2048', currency=scientific.JOINT_CURRENCY)


def roster(size=3, *, width=2):
    # Structurally complete invented evidence. It is deliberately NOT a claim
    # about an installed interpreter; that is the native boundary's obligation.
    content = {'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
    runtime = dict(schema='prismaquant.flux_result_runtime.v5',
        python=dict(path='/runtime/python', **content, version='fixture',
                    implementation='cpython', cache_tag='fixture', flags=[1]),
        dependencies={}, stdlib={'/runtime/stdlib': ['os.py']},
        imports={'sys': dict(origin='built-in', locations=[], loader='fixture', cached=None)},
        libraries=['/runtime/lib.so'], files={name: dict(content) for name in
        ['/runtime/python', '/runtime/stdlib/os.py', '/runtime/lib.so']})
    tasks = [dict(task_id=f'task/{i}', output_id=f'out/{i}', payload=payload())
             for i in range(size)]
    return dict(schema=prefill.ROSTER_SCHEMA, phase_id='phase/A',
        plan_context_sha256='a' * 64,
        command=dict(interpreter='/runtime/python', module='experiments.producer'),
        runtime=runtime,
        inputs={'calibration': dict(path='/inputs/calibration', bytes=3,
                                   sha256=hashlib.sha256(b'abc').hexdigest())},
        tasks=tasks, children=[dict(child_id=f'child/{i}',
            task_ids=[t['task_id'] for t in tasks[i:i + width]],
            resources=dict(cores=1, memory_gib=2, gpus=0, walltime_s=60, placement='portable'),
            staged_input=None) for i in range(0, size, width)])


def test_ordered_cover_is_detached_and_preserves_scientific_meaning():
    value = roster()
    checked = prefill.validate_phase_roster(value)
    assert [c['task_ids'] for c in checked['children']] == [['task/0', 'task/1'], ['task/2']]
    assert checked['tasks'][0]['payload'] == payload()
    value['tasks'][0]['payload']['members'].clear()
    assert checked['tasks'][0]['payload']['members'] == ['layer.0/up', 'layer.0/gate']


@pytest.mark.parametrize('currency', sorted(scientific.CURRENCIES))
def test_scientific_task_retains_each_independent_currency(currency):
    task = payload()
    task['currency'] = currency
    assert scientific.validate_task_payload(task, where='task') == task


@pytest.mark.parametrize('field,bad', [
    ('schema', 'prismabuild.logical_task.v1'), ('members', []),
    ('members', ['one', 'one']), ('members', ['bad space']),
    ('rate', True), ('rate', 1.0), ('rate', 0), ('currency', 'mixed'),
    ('serving_unit', ''), ('family', 'a' * 257), ('context_id', 'non-ascii-\N{SNOWMAN}'),
])
def test_invalid_scientific_task_is_refused(field, bad):
    task = payload()
    scientific.validate_task_payload(task, where='positive')
    task[field] = bad
    with pytest.raises(scientific.QualityPrefillContractError):
        scientific.validate_task_payload(task, where='negative')


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unknown', 'reordered',
                                   'output', 'child', 'task', 'empty-child'])
def test_cover_refuses_without_silently_repairing_it(change):
    value = roster()
    prefill.validate_phase_roster(value)
    if change == 'missing': value['children'][-1]['task_ids'] = ['task/1']
    elif change == 'duplicate': value['children'][0]['task_ids'].append('task/1')
    elif change == 'unknown': value['children'][-1]['task_ids'] = ['unknown']
    elif change == 'reordered': value['children'].reverse()
    elif change == 'output': value['tasks'][1]['output_id'] = 'out/0'
    elif change == 'child': value['children'][1]['child_id'] = 'child/0'
    elif change == 'task': value['tasks'][1]['task_id'] = 'task/0'
    else: value['children'][0]['task_ids'].clear()
    with pytest.raises(ValueError): prefill.validate_phase_roster(value)


@pytest.mark.parametrize('field,bad', [('cores', 0), ('cores', 5), ('cores', True),
    ('memory_gib', 17), ('gpus', 1), ('walltime_s', 1801), ('placement', 'auto-discover')])
def test_resource_refusal_precedes_staged_file_access(monkeypatch, field, bad):
    value = roster()
    value['children'][0]['staged_input'] = {'unread': 'marker'}
    value['children'][-1]['resources'][field] = bad
    reads = []
    def forbidden(*args, **kwargs):
        reads.append(args)
        raise AssertionError('staged file access before complete admission')
    monkeypatch.setattr(prefill, 'validate_staged_binding', forbidden)
    with pytest.raises(ValueError): prefill.validate_phase_roster(value)
    assert reads == []


@pytest.mark.parametrize('kind', ['tasks', 'children', 'child-tasks', 'inputs', 'bytes'])
def test_finite_limits_and_their_positive_boundaries(monkeypatch, kind):
    if kind == 'tasks':
        value = roster(1024, width=64)
        prefill.validate_phase_roster(value)
        value['tasks'].append(dict(task_id='task/1024', output_id='out/1024', payload=payload()))
    elif kind == 'children':
        value = roster(256, width=1)
        prefill.validate_phase_roster(value)
        value['children'].append(copy.deepcopy(value['children'][-1]))
    elif kind == 'child-tasks':
        value = roster(64, width=64)
        prefill.validate_phase_roster(value)
        value['children'][0]['task_ids'].append('extra')
    elif kind == 'inputs':
        value = roster()
        value['inputs'] = {f'role/{i}': dict(path=f'/input/{i}', bytes=1, sha256='a' * 64)
                           for i in range(32)}
        prefill.validate_phase_roster(value)
        value['inputs']['extra'] = dict(path='/extra', bytes=1, sha256='a' * 64)
    else:
        value = roster()
        size = len(scientific.canonical_json_bytes(value)) + 1
        monkeypatch.setattr(prefill, 'MAX_ROSTER_BYTES', size)
        prefill.validate_phase_roster(value)
        monkeypatch.setattr(prefill, 'MAX_ROSTER_BYTES', size - 1)
    with pytest.raises(ValueError): prefill.validate_phase_roster(value)


@pytest.mark.parametrize('key', ['root', 'command', 'input', 'task', 'payload', 'child', 'resource', 'runtime'])
def test_every_document_boundary_refuses_unknown_fields(key):
    value = roster()
    bodies = dict(root=value, command=value['command'], input=value['inputs']['calibration'],
        task=value['tasks'][0], payload=value['tasks'][0]['payload'], child=value['children'][0],
        resource=value['children'][0]['resources'], runtime=value['runtime'])
    bodies[key]['undeclared'] = 1
    with pytest.raises(ValueError): prefill.validate_phase_roster(value)


@pytest.mark.parametrize('field,bad', [('path', 'relative'), ('path', '/input/../escape'),
    ('sha256', 'A' * 64), ('bytes', True), ('bytes', -1), ('bytes', 1 << 63)])
def test_input_bindings_are_exact_normalized_and_bounded(field, bad):
    value = roster()
    value['inputs']['calibration'][field] = bad
    with pytest.raises(ValueError): prefill.validate_phase_roster(value)


def test_complete_admission_delegates_staged_authentication_with_placement(tmp_path):
    from staged_native_fixtures import publish_inputs
    source = tmp_path / 'known'
    source.write_bytes(b'known staged input')
    value = roster()
    value['children'][0]['staged_input'] = publish_inputs(tmp_path, [source])
    assert prefill.validate_phase_roster(value)['children'][0]['staged_input']['schema'] == 'prismaquant.staged_launch.v1'
    value['children'][0]['staged_input']['sha256'] = '0' * 64
    with pytest.raises(ValueError): prefill.validate_phase_roster(value)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_values_never_reach_publication(tmp_path, bad):
    value = roster()
    value['tasks'][0]['payload']['rate'] = bad
    with pytest.raises(ValueError): prefill.write_phase_roster(value, tmp_path / 'never')
    assert list(tmp_path.iterdir()) == []


def test_publication_is_canonical_known_bytes_and_never_overwrites(tmp_path):
    value = roster()
    expected = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    target = tmp_path / 'roster.json'
    binding = prefill.write_phase_roster(value, path=target)
    assert target.read_bytes() == expected
    assert binding == dict(path=str(target), sha256=hashlib.sha256(expected).hexdigest(), bytes=len(expected))
    with pytest.raises(FileExistsError): prefill.write_phase_roster(value, path=target)
    assert target.read_bytes() == expected
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('kind', ['parent-symlink', 'target-symlink', 'directory', 'missing-parent'])
def test_publication_refuses_foreign_targets_without_modifying_them(tmp_path, kind):
    owned = tmp_path / 'owned'; owned.mkdir()
    foreign = tmp_path / 'foreign'; foreign.mkdir()
    sentinel = foreign / 'keep'; sentinel.write_bytes(b'unchanged')
    target = owned / 'roster.json'
    if kind == 'parent-symlink':
        owned.rmdir(); owned.symlink_to(foreign, target_is_directory=True)
    elif kind == 'target-symlink': target.symlink_to(sentinel)
    elif kind == 'directory': target.mkdir()
    else: owned.rmdir()
    with pytest.raises((ValueError, OSError)): prefill.write_phase_roster(roster(), target)
    assert sentinel.read_bytes() == b'unchanged'
    assert list(foreign.iterdir()) == [sentinel]


@pytest.mark.parametrize('when', ['before-write', 'before-read', 'after-read'])
def test_replaced_parent_never_grants_write_or_return_authority(tmp_path, monkeypatch, when):
    parent = tmp_path / 'parent'; parent.mkdir()
    moved = tmp_path / 'original'
    target = parent / 'roster.json'
    def replace():
        parent.rename(moved)
        parent.mkdir()
        (parent / 'foreign').write_bytes(b'leave')
    original_write, original_read = prefill.publish_new_bytes, prefill.read_document
    def write(*args, **kwargs):
        if when == 'before-write': replace()
        result = original_write(*args, **kwargs)
        if when == 'before-read': replace()
        return result
    def read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        if when == 'after-read': replace()
        return result
    monkeypatch.setattr(prefill, 'publish_new_bytes', write)
    monkeypatch.setattr(prefill, 'read_document', read)
    with pytest.raises((ValueError, OSError)): prefill.write_phase_roster(roster(), target)
    assert sorted(p.name for p in parent.iterdir()) == ['foreign']
    assert (parent / 'foreign').read_bytes() == b'leave'


def test_same_byte_replacement_cannot_be_adopted_as_the_writer(tmp_path, monkeypatch):
    original = prefill.publish_new_bytes
    def substitute(target, raw, **kwargs):
        result = original(target, raw, **kwargs)
        replacement = tmp_path / 'replacement'
        replacement.write_bytes(raw)
        assert replacement.stat().st_ino != target.stat().st_ino
        os.replace(replacement, target)
        return result
    monkeypatch.setattr(prefill, 'publish_new_bytes', substitute)
    with pytest.raises(ValueError, match='owned writer'):
        prefill.write_phase_roster(roster(), tmp_path / 'roster.json')


@pytest.mark.parametrize('mutation', ['missing', 'changed', 'length', 'digest', 'oversize', 'symlink'])
def test_bound_document_authenticates_known_bytes_before_json(tmp_path, mutation):
    filename = tmp_path / 'document.json'
    raw = b'{"known":42}\n'
    filename.write_bytes(raw)
    binding = dict(path=str(filename), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    assert prefill._read_bound_document(binding, len(raw)) == {'known': 42}
    if mutation == 'missing': filename.unlink()
    elif mutation == 'changed': filename.write_bytes(b'{"known":43}\n')
    elif mutation == 'length': binding['bytes'] -= 1
    elif mutation == 'digest': binding['sha256'] = '0' * 64
    elif mutation == 'oversize': binding['bytes'] += 1
    else:
        target = tmp_path / 'target'; filename.rename(target); filename.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        prefill._read_bound_document(binding, len(raw))


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}'])
def test_authenticated_document_still_refuses_invalid_json(tmp_path, raw):
    filename = tmp_path / 'invalid.json'
    filename.write_bytes(raw)
    binding = dict(path=str(filename), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    with pytest.raises(ValueError): prefill._read_bound_document(binding, len(raw))


def test_private_publisher_enforces_document_bound_before_writes(tmp_path):
    filename = tmp_path / 'never.json'
    with pytest.raises(ValueError): prefill._publish_document({'known': 42}, filename, 1)
    assert list(tmp_path.iterdir()) == []
