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


if __name__ == '__main__':
    unittest.main(verbosity=2)
