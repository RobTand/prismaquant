"""Retained scope binding for PQ #2446: frozen fit, held-out, and selection.

Failing-first proof: matched encode/HELD pairs from another sample
population must not remove current work. A different supplied manifest on
catalogue resume must not enter silently. Matching recovery must admit once.
The gate reads the real roster payload shape: only qname and frozen reads.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.d44_native import next_wave as wave


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


FIT_SHA = 'f' * 64
HELDOUT_SHA = 'e' * 64
SELECTION_SHA = 'c' * 64
SELECTION_FILE = 'selection-subsample-20261007.json'


def write_pair(root, kind, *, fit_sha=None, heldout_sha=None, selection=None):
    name = f'model.language_model.layers.40.mlp.experts.0.{kind}_proj'
    blob = root / f'{kind}.tessera'
    blob.write_bytes(f'retained {kind}'.encode())
    encode_path = root / f'{kind}-encode.json'
    held_path = root / f'{kind}-held.json'
    fit_sha = FIT_SHA if fit_sha is None else fit_sha
    heldout_sha = HELDOUT_SHA if heldout_sha is None else heldout_sha
    selection = SELECTION_SHA if selection is None else selection
    dimensions = [4, 8] if kind != 'down' else [8, 4]
    conditioning = {'selection_sha256': selection}
    fit = {'count': 6272, 'sha256': fit_sha}
    heldout = {'count': 1634, 'sha256': heldout_sha}
    encode_path.write_text(json.dumps({'qname': name, 'dry_run': False,
        'blob_path': str(blob), 'blob_sha256': digest(blob),
        'blob_bytes': blob.stat().st_size, 'rendered_shape': dimensions,
        'conditioning': conditioning, 'fit': fit, 'heldout': heldout}))
    held_path.write_text(json.dumps({'qname': name, 'dry_run': False,
        'actual_source_shape': dimensions, 'conditioning': conditioning,
        'fit': dict(fit), 'heldout': dict(heldout), 'replacement': {
            'blob': str(blob), 'blob_sha256': digest(blob),
            'bytes': blob.stat().st_size, 'receipt': str(encode_path),
            'receipt_sha256': digest(encode_path)}}))
    return name, {'qname': name,
                  'encode': {'path': str(encode_path), 'sha256': digest(encode_path)},
                  'held': {'path': str(held_path), 'sha256': digest(held_path)}}


def scope_task(name, task_id, *, fit_sha=None, heldout_sha=None,
               selection=None, selection_file=SELECTION_FILE,
               drop_reads=False, corrupt_digest=False,
               encode_conditioning=True, held_conditioning=True):
    leaf = name.split('.')[-1]
    selection = SELECTION_SHA if selection is None else selection
    reads = []
    if not drop_reads:
        reads = [
            {'path': f'/frozen/{leaf}.fit.pt',
             'sha256': ('short' if corrupt_digest else (FIT_SHA if fit_sha is None else fit_sha))},
            {'path': f'/frozen/{leaf}.heldout.pt',
             'sha256': HELDOUT_SHA if heldout_sha is None else heldout_sha},
            {'path': f'/frozen/{selection_file}', 'sha256': selection},
        ]
    return {'id': task_id, 'output_id': task_id,
            'payload': {'qname': name, 'reads': reads},
            '_expect': {'encode_conditioning': encode_conditioning,
                        'held_conditioning': held_conditioning}}


def strip_conditioning(root, kind, document, field='conditioning'):
    path = root / f'{kind}-{document}.json'
    doc = json.loads(path.read_text())
    doc.pop(field, None)
    path.write_text(json.dumps(doc))
    return path


class RetainedScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='d44-2446-scope-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names, self.units, self.tasks = [], [], []
        for i, kind in enumerate(('gate', 'up', 'down')):
            name, unit = write_pair(self.root, kind)
            self.names.append(name)
            self.units.append(unit)
            self.tasks.append(scope_task(name, f'unit-{i}'))
        self.tasks.append({'id': 'unit-remaining', 'output_id': 'unit-remaining',
                           'payload': {'qname': 'model.language_model.layers.40.mlp.experts.1.up_proj',
                                       'reads': []}})
        self.request = {'roster': {'tasks': self.tasks}}
        self.manifest = {'schema': 'd44.retained_units.v1', 'units': self.units}

    def test_matching_scope_admits_once_without_duplicate(self):
        request, adopted = wave.adopt_retained_units(self.request, self.manifest)
        self.assertEqual([task['id'] for task in request['roster']['tasks']], ['unit-remaining'])
        self.assertEqual({row['task_id'] for row in adopted}, {'unit-0', 'unit-1', 'unit-2'})
        self.assertEqual(len(self.request['roster']['tasks']), 4)
        catalog = {'retained_units': adopted}
        check = wave.retained_population_key
        self.assertEqual(check(catalog['retained_units']), check(adopted))
        wave.check_retained_resume(catalog, None)
        same_file = self.root / 'same-manifest.json'
        same_file.write_text(json.dumps(self.manifest))
        wave.check_retained_resume(catalog, same_file)

    def test_real_payload_shape_binds_scope_without_retained_scope_field(self):
        task = self.tasks[0]
        self.assertEqual(sorted(task['payload'].keys()), ['qname', 'reads'])
        self.assertNotIn('retained_scope', task['payload'])
        scope = wave.retained_task_scope(task)
        self.assertEqual(scope, {'fit_sha256': FIT_SHA, 'heldout_sha256': HELDOUT_SHA,
                                 'selection_sha256': SELECTION_SHA})

    def test_different_fit_rows_refuse_before_work_removal(self):
        name, unit = write_pair(self.root, 'gate', fit_sha='a' * 64)
        manifest = {'schema': 'd44.retained_units.v1', 'units': [unit]}
        before = [task['id'] for task in self.request['roster']['tasks']]
        with self.assertRaisesRegex(ValueError, 'fit.*frozen'):
            wave.adopt_retained_units(self.request, manifest)
        after = [task['id'] for task in self.request['roster']['tasks']]
        self.assertEqual(before, after)

    def test_different_heldout_rows_refuse_before_work_removal(self):
        name, unit = write_pair(self.root, 'up', heldout_sha='b' * 64)
        manifest = {'schema': 'd44.retained_units.v1', 'units': [unit]}
        with self.assertRaisesRegex(ValueError, 'heldout.*frozen'):
            wave.adopt_retained_units(self.request, manifest)

    def test_different_selection_refuses_before_work_removal(self):
        name, unit = write_pair(self.root, 'down', selection='d' * 64)
        manifest = {'schema': 'd44.retained_units.v1', 'units': [unit]}
        with self.assertRaisesRegex(ValueError, 'selection.*frozen'):
            wave.adopt_retained_units(self.request, manifest)

    def test_task_without_scope_reads_refuses(self):
        name, unit = write_pair(self.root, 'gate')
        task = scope_task(name, 'unit-x', drop_reads=True)
        request = {'roster': {'tasks': [task]}}
        manifest = {'schema': 'd44.retained_units.v1', 'units': [unit]}
        with self.assertRaisesRegex(ValueError, 'no frozen scope reads'):
            wave.adopt_retained_units(request, manifest)

    def test_malformed_scope_digest_refuses(self):
        name, unit = write_pair(self.root, 'gate')
        task = scope_task(name, 'unit-x', corrupt_digest=True)
        request = {'roster': {'tasks': [task]}}
        manifest = {'schema': 'd44.retained_units.v1', 'units': [unit]}
        with self.assertRaisesRegex(ValueError, 'no frozen scope reads'):
            wave.adopt_retained_units(request, manifest)

    def test_missing_encode_selection_refuses(self):
        name, unit = write_pair(self.root, 'gate')
        path = strip_conditioning(self.root, 'gate', 'encode')
        unit['encode']['sha256'] = digest(path)
        held_path = self.root / 'gate-held.json'
        held_doc = json.loads(held_path.read_text())
        held_doc['replacement']['receipt_sha256'] = digest(path)
        held_path.write_text(json.dumps(held_doc))
        unit['held']['sha256'] = digest(held_path)
        manifest = {'schema': 'd44.retained_units.v1', 'units': [unit]}
        request = {'roster': {'tasks': [scope_task(name, 'unit-x')]}}
        with self.assertRaisesRegex(ValueError, 'encode selection digest is missing'):
            wave.adopt_retained_units(request, manifest)

    def test_missing_held_selection_refuses(self):
        name, unit = write_pair(self.root, 'gate')
        path = strip_conditioning(self.root, 'gate', 'held')
        unit['held']['sha256'] = digest(path)
        manifest = {'schema': 'd44.retained_units.v1', 'units': [unit]}
        request = {'roster': {'tasks': [scope_task(name, 'unit-x')]}}
        with self.assertRaisesRegex(ValueError, 'HELD selection digest is missing'):
            wave.adopt_retained_units(request, manifest)

    def test_held_only_selection_still_refuses_mismatch(self):
        name, unit = write_pair(self.root, 'gate')
        encode_path = self.root / 'gate-encode.json'
        doc = json.loads(encode_path.read_text())
        doc['conditioning']['selection_sha256'] = '9' * 64
        encode_path.write_text(json.dumps(doc))
        unit['encode']['sha256'] = digest(encode_path)
        held_path = self.root / 'gate-held.json'
        held_doc = json.loads(held_path.read_text())
        held_doc['replacement']['receipt_sha256'] = digest(encode_path)
        held_path.write_text(json.dumps(held_doc))
        unit['held']['sha256'] = digest(held_path)
        manifest = {'schema': 'd44.retained_units.v1', 'units': [unit]}
        request = {'roster': {'tasks': [scope_task(name, 'unit-x')]}}
        with self.assertRaisesRegex(ValueError, 'selection.*frozen'):
            wave.adopt_retained_units(request, manifest)

    def test_different_resume_manifest_refuses_before_admission(self):
        _, adopted = wave.adopt_retained_units(self.request, self.manifest)
        catalog = {'retained_units': adopted}
        other_name, other_unit = write_pair(self.root, 'gate', selection='e' * 64)
        other_file = self.root / 'other-manifest.json'
        other_file.write_text(json.dumps(
            {'schema': 'd44.retained_units.v1', 'units': [other_unit]}))
        with self.assertRaisesRegex(ValueError, 'differs from the stored'):
            wave.check_retained_resume(catalog, other_file)
        dropped_file = self.root / 'dropped-manifest.json'
        dropped_file.write_text(json.dumps(
            {'schema': 'd44.retained_units.v1', 'units': self.units[:2]}))
        with self.assertRaisesRegex(ValueError, 'differs from the stored'):
            wave.check_retained_resume(catalog, dropped_file)


@pytest.mark.parametrize('role, keyword', [('fit', 'fit_sha'), ('heldout', 'heldout_sha'),
                                          ('selection', 'selection')])
@pytest.mark.parametrize('location', ['matching-frozen', 'encode', 'held'])
@pytest.mark.parametrize('dev_mode', ['0', '1'])
def test_nonhex_scope_digests_refuse_without_removing_work(tmp_path, monkeypatch,
                                                          role, keyword, location, dev_mode):
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', dev_mode)
    # A matching pair of length-64 strings is not necessarily a SHA-256 identity.
    invalid = 'z' * 64
    kwargs = {keyword: invalid} if location == 'matching-frozen' else {}
    name, unit = write_pair(tmp_path, 'gate', **kwargs)
    task = scope_task(name, 'unit', **kwargs)
    if location != 'matching-frozen':
        path = Path(unit[location]['path'])
        document = json.loads(path.read_text())
        if role == 'selection':
            document['conditioning']['selection_sha256'] = invalid
        else:
            document[role]['sha256'] = invalid
        path.write_text(json.dumps(document))
        unit[location]['sha256'] = digest(path)
    request = {'roster': {'tasks': [task]}}
    before = json.dumps(request)
    with pytest.raises(ValueError):
        wave.adopt_retained_units(request, {'schema': 'd44.retained_units.v1', 'units': [unit]})
    assert json.dumps(request) == before


@pytest.mark.parametrize('role', ['fit', 'heldout', 'selection'])
def test_ambiguous_frozen_scope_refuses(tmp_path, role):
    name, unit = write_pair(tmp_path, 'gate')
    task = scope_task(name, 'unit')
    read = task['payload']['reads'][['fit', 'heldout', 'selection'].index(role)]
    task['payload']['reads'].append(dict(read))
    with pytest.raises(ValueError, match='no frozen scope reads'):
        wave.adopt_retained_units({'roster': {'tasks': [task]}},
                                  {'schema': 'd44.retained_units.v1', 'units': [unit]})


def test_resume_refuses_before_controller_construction(tmp_path):
    state_dir = tmp_path / 'state'
    state_dir.mkdir()
    wave.durable_json(state_dir / 'catalog.json', {'retained_units': []})
    name, unit = write_pair(tmp_path, 'gate')
    manifest_path = tmp_path / 'retained.json'
    wave.durable_json(manifest_path, {'schema': 'd44.retained_units.v1', 'units': [unit]})
    with patch.object(wave, 'RoutedWave') as controller, patch.object(wave, 'routed_api') as api:
        with pytest.raises(ValueError, match='differs from the stored'):
            wave.routed_main(['--routed-plan', str(tmp_path / 'plan.json'),
                              '--state-dir', str(state_dir),
                              '--retained-units', str(manifest_path)])
    controller.assert_not_called()
    api.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
