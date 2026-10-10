"""Failing-first proof for retained encode and HELD catalogue adoption."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.d44_native import next_wave as wave


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='d44-catalogue-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.units = []
        self.tasks = []
        for i, kind in enumerate(('gate', 'up', 'down')):
            name = f'model.language_model.layers.40.mlp.experts.0.{kind}_proj'
            blob = self.root / f'{kind}.tessera'
            blob.write_bytes(f'retained {kind}'.encode())
            encode = self.root / f'{kind}-encode.json'
            held = self.root / f'{kind}-held.json'
            self.tasks.append({'id': f'unit-{i}', 'output_id': f'unit-{i}', 'payload': {'qname': name}})
            dimensions = [4, 8] if kind != 'down' else [8, 4]
            roles = {'fit': {'count': 10, 'sha256': 'a' * 64},
                     'heldout': {'count': 3, 'sha256': 'b' * 64}}
            encode.write_text(json.dumps({'qname': name, 'dry_run': False,
                'blob_path': str(blob), 'blob_sha256': digest(blob), 'blob_bytes': blob.stat().st_size,
                'rendered_shape': dimensions, **roles}))
            held.write_text(json.dumps({'qname': name, 'dry_run': False,
                'actual_source_shape': dimensions, **roles, 'replacement': {
                    'blob': str(blob), 'blob_sha256': digest(blob), 'bytes': blob.stat().st_size,
                    'receipt': str(encode), 'receipt_sha256': digest(encode)}}))
            self.units.append({'qname': name, 'encode': {'path': str(encode), 'sha256': digest(encode)},
                               'held': {'path': str(held), 'sha256': digest(held)}})
        self.tasks.append({'id': 'unit-remaining', 'output_id': 'unit-remaining',
                           'payload': {'qname': 'model.language_model.layers.40.mlp.experts.1.up_proj'}})
        self.request = {'roster': {'tasks': self.tasks}}
        self.manifest = {'schema': 'd44.retained_units.v1', 'units': self.units}

    def test_three_retained_pairs_leave_only_unfinished_tasks(self):
        request, adopted = wave.adopt_retained_units(self.request, self.manifest)
        self.assertEqual([t['id'] for t in request['roster']['tasks']], ['unit-remaining'])
        self.assertEqual({row['task_id'] for row in adopted}, {'unit-0', 'unit-1', 'unit-2'})
        self.assertEqual(len(self.request['roster']['tasks']), 4, 'The frozen input roster must stay unchanged.')

    def test_corrupt_blob_refuses_adoption(self):
        encode = json.loads(Path(self.units[0]['encode']['path']).read_text())
        Path(encode['blob_path']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'digest|bytes'):
            wave.adopt_retained_units(self.request, self.manifest)

    def test_wrong_held_replacement_refuses_adoption(self):
        path = Path(self.units[0]['held']['path'])
        doc = json.loads(path.read_text())
        doc['replacement']['receipt_sha256'] = '0' * 64
        path.write_text(json.dumps(doc))
        self.units[0]['held']['sha256'] = digest(path)
        with self.assertRaisesRegex(ValueError, 'replacement'):
            wave.adopt_retained_units(self.request, self.manifest)

    def test_duplicate_or_foreign_retained_unit_refuses_adoption(self):
        self.manifest['units'].append(self.units[0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            wave.adopt_retained_units(self.request, self.manifest)
        self.manifest['units'] = [dict(self.units[0], qname='not in this roster')]
        with self.assertRaisesRegex(ValueError, 'roster'):
            wave.adopt_retained_units(self.request, self.manifest)

    def test_mismatched_roles_refuse_adoption(self):
        path = Path(self.units[0]['held']['path'])
        doc = json.loads(path.read_text())
        doc['heldout']['count'] = 4
        path.write_text(json.dumps(doc))
        self.units[0]['held']['sha256'] = digest(path)
        with self.assertRaisesRegex(ValueError, 'heldout'):
            wave.adopt_retained_units(self.request, self.manifest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
