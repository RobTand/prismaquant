"""Retained scope binding for PQ #2446: frozen fit, held-out, and selection.

Failing-first proof: matched encode/HELD pairs from another sample
population must not remove current work. A different supplied manifest on
catalogue resume must not enter silently. Matching recovery must admit once.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.d44_native import next_wave as wave


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


FIT = {'count': 10, 'sha256': 'a' * 64}
HELDOUT = {'count': 3, 'sha256': 'b' * 64}
SELECTION = 'c' * 64


def write_pair(root, kind, *, fit=None, heldout=None, selection=None):
    name = f'model.language_model.layers.40.mlp.experts.0.{kind}_proj'
    blob = root / f'{kind}.tessera'
    blob.write_bytes(f'retained {kind}'.encode())
    encode_path = root / f'{kind}-encode.json'
    held_path = root / f'{kind}-held.json'
    fit = dict(FIT if fit is None else fit)
    heldout = dict(HELDOUT if heldout is None else heldout)
    selection = SELECTION if selection is None else selection
    dimensions = [4, 8] if kind != 'down' else [8, 4]
    conditioning = {'selection_sha256': selection}
    encode_path.write_text(json.dumps({'qname': name, 'dry_run': False,
        'blob_path': str(blob), 'blob_sha256': digest(blob),
        'blob_bytes': blob.stat().st_size, 'rendered_shape': dimensions,
        'conditioning': conditioning, 'fit': fit, 'heldout': heldout}))
    held_path.write_text(json.dumps({'qname': name, 'dry_run': False,
        'actual_source_shape': dimensions, 'conditioning': conditioning,
        'fit': fit, 'heldout': heldout, 'replacement': {
            'blob': str(blob), 'blob_sha256': digest(blob),
            'bytes': blob.stat().st_size, 'receipt': str(encode_path),
            'receipt_sha256': digest(encode_path)}}))
    return name, {'qname': name,
                  'encode': {'path': str(encode_path), 'sha256': digest(encode_path)},
                  'held': {'path': str(held_path), 'sha256': digest(held_path)}}


def scope_task(name, task_id, *, fit=None, heldout=None, selection=None):
    return {'id': task_id, 'output_id': task_id,
            'payload': {'qname': name,
                        'retained_scope': {
                            'fit': dict(FIT if fit is None else fit),
                            'heldout': dict(HELDOUT if heldout is None else heldout),
                            'selection_sha256': SELECTION if selection is None else selection}}}


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
                           'payload': {'qname': 'model.language_model.layers.40.mlp.experts.1.up_proj'}})
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

    def test_different_fit_rows_refuse_before_work_removal(self):
        name, unit = write_pair(self.root, 'gate', fit={'count': 11, 'sha256': 'a' * 64})
        manifest = {'schema': 'd44.retained_units.v1', 'units': [unit]}
        before = [task['id'] for task in self.request['roster']['tasks']]
        with self.assertRaisesRegex(ValueError, 'fit.*frozen'):
            wave.adopt_retained_units(self.request, manifest)
        after = [task['id'] for task in self.request['roster']['tasks']]
        self.assertEqual(before, after)

    def test_different_heldout_rows_refuse_before_work_removal(self):
        name, unit = write_pair(self.root, 'up', heldout={'count': 4, 'sha256': 'b' * 64})
        manifest = {'schema': 'd44.retained_units.v1', 'units': [unit]}
        with self.assertRaisesRegex(ValueError, 'heldout.*frozen'):
            wave.adopt_retained_units(self.request, manifest)

    def test_different_selection_refuses_before_work_removal(self):
        name, unit = write_pair(self.root, 'down', selection='d' * 64)
        manifest = {'schema': 'd44.retained_units.v1', 'units': [unit]}
        with self.assertRaisesRegex(ValueError, 'selection.*frozen'):
            wave.adopt_retained_units(self.request, manifest)

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
