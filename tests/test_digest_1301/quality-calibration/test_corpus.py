"""CPU-only differential consumers using frozen original function blocks."""
import argparse
import ast
import contextlib
import copy
import __future__
import importlib.util
import io
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/quality-calibration/corpus'


def load(base, name):
    spec = importlib.util.spec_from_file_location('_test_' + name, base / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CorpusTests(unittest.TestCase):
    def pair(self, name):
        new = load(ROOT / 'tools', name)
        # Only context is shared: original function bodies are frozen, never
        # extracted from the implementation under test. No package imports or
        # full original module copies are needed to execute these blocks.
        old = types.ModuleType('_original_' + name)
        old.__dict__.update(argparse=argparse, hashlib=hashlib, json=json, Path=Path)
        constants = (('MANIFEST', 'DEFAULT_TARGET_TOKENS', 'DEFAULT_SEED')
                     if name == 'build_diverse_calibration' else
                     ('WIKITEXT_DATASET', 'WIKITEXT_CONFIG', 'WIKITEXT_REVISION'))
        for key in constants:
            setattr(old, key, copy.deepcopy(getattr(new, key)))
        path = FIXTURES / (name + '.py')
        exec(compile(path.read_text(), str(path), 'exec',
                     flags=__future__.annotations.compiler_flag, dont_inherit=True), old.__dict__)
        return old, new

    def test_original_blocks(self):
        for name, expected, functions in (
            ('build_diverse_calibration', 'deeb07843de4f6ba3662c7e4164678fec9864f17d378459fd828b65f3a4490da',
             ['_sha256_text', '_json_digest', 'write_jsonl']),
            ('materialize_wikitext_corpus', '51a500cb8e17863bc8e1c5a61124b678de394ffb408aa4f57d54534f9d24e147',
             ['join_rows', 'main']),
        ):
            raw = (FIXTURES / (name + '.py')).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
            nodes = ast.parse(raw).body
            self.assertTrue(all(isinstance(n, ast.FunctionDef) for n in nodes))
            self.assertEqual([n.name for n in nodes], functions)

    def test_ledger_provenance(self):
        ledger = json.loads((FIXTURES / 'ledger.json').read_text())
        registry = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
        calls = {call['id']: call for call in registry['calls']}
        for entry in ledger['entries']:
            # Original IDs and frozen-source attribution follow the immutable
            # research registry, never the older historical census.
            self.assertIn(entry['original_id'], calls)
            self.assertEqual(entry['source_commit'], registry['source_commit'])
            self.assertEqual(entry['source_blob'], calls[entry['original_id']]['source_blob'])
            for key in ('source_commit', 'source_blob', 'owner_route', 'fixtures', 'tests'):
                self.assertTrue(entry[key], (entry['original_id'], key))
            for path in entry['fixtures']:
                self.assertTrue((ROOT / path).is_file())
            self.assertTrue(all('test_original_blobs' not in node for node in entry['tests']))
            if 'build_diverse_calibration' in entry['original_id']:
                self.assertNotEqual(entry['original_census_blob'], entry['source_blob'])
                self.assertEqual(entry['historical_census_source_commit'], 'f4436b7ca7e34bd7cb1cafa1f49e95d1365afb0e')
                self.assertIn('Historical census', entry['provenance_discrepancy'])

    def test_standalone_no_package(self):
        code = '''
import importlib.abc, importlib.util, pathlib, sys
class Refuse(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('torch', 'prismaquant'):
   raise AssertionError(fullname)
sys.meta_path.insert(0, Refuse())
for name in ('build_diverse_calibration', 'materialize_wikitext_corpus'):
 p = pathlib.Path(sys.argv[1]) / 'tools' / (name + '.py')
 s = importlib.util.spec_from_file_location(name, p)
 m = importlib.util.module_from_spec(s)
 s.loader.exec_module(m)
assert 'torch' not in sys.modules and 'prismaquant' not in sys.modules
'''
        subprocess.run([sys.executable, '-S', '-c', code, str(ROOT)], check=True,
                       cwd=tempfile.gettempdir())

    def test_diverse_digest_recipes(self):
        old, new = self.pair('build_diverse_calibration')
        for value in ({'z': '雪', 'a': [float('nan'), float('inf')]},
                      {'b': 1, 'a': 2}, ['é', '', '\n']):
            self.assertEqual(old._json_digest(value), new._json_digest(value))
        for value in ('', '雪\r\n é\x00', 'e\u0301', 'é'):
            self.assertEqual(old._sha256_text(value), new._sha256_text(value))
        for method, value, error in (('_sha256_text', '\ud800', UnicodeEncodeError),
                                      ('_json_digest', {'x': '\ud800'}, UnicodeEncodeError),
                                      ('_json_digest', {1: 1, 'x': 2}, TypeError),
                                      ('_json_digest', object(), TypeError)):
            for mod in (old, new):
                with self.assertRaises(error):
                    getattr(mod, method)(value)

    def test_diverse_jsonl_publication(self):
        old, new = self.pair('build_diverse_calibration')
        records = [{'text': ' 雪\r\n café '}, {'text': ''}, {'text': float('nan')}]
        with tempfile.TemporaryDirectory() as tmp:
            outputs = []
            for i, mod in enumerate((old, new)):
                path = Path(tmp) / str(i) / 'calib.jsonl'
                mod.write_jsonl(path, records, tokenizer_path='tokenizer/雪', seed=9)
                outputs.append(path.read_bytes())
                self.assertFalse(path.with_suffix('.jsonl.tmp').exists())
            self.assertEqual(*outputs)
            self.assertIn(b'NaN', outputs[1])
            self.assertIn('雪'.encode(), outputs[1])
            self.assertTrue(outputs[1].endswith(b'\n'))
            rows = [json.loads(line) for line in outputs[1].splitlines()]
            self.assertIn('records_sha256', rows[0]['__manifest__'])

    def test_wikitext_publication_before_hash(self):
        old, new = self.pair('materialize_wikitext_corpus')
        class Rows(list):
            _fingerprint = 'fixture-fingerprint'
        rows = Rows([{'text': ' 雪\r\n'}, {'text': ''}, {'text': None},
                     {'text': ' \t'}, {}, {'text': 'café\n'}])
        dataset = types.ModuleType('datasets')
        calls = []
        def get_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return rows
        dataset.load_dataset = get_dataset
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'corpus'
            argv = ['materialize', '--out-dir', str(out), '--cache-dir', str(Path(tmp)/'cache'),
                    '--split', 'train', 'test']
            snapshots = []
            for mod in (old, new):
                observed = []
                def check(raw):
                    split = ('train', 'test')[len(observed)]
                    self.assertEqual((out / ('wikitext-2-raw-v1.' + split + '.txt')).read_bytes(), raw)
                    observed.append(raw)
                if mod is new:
                    original = new._digests.bytes_sha256hex
                    def digest(raw):
                        check(raw)
                        return original(raw)
                    digest_patch = patch.object(new._digests, 'bytes_sha256hex', digest)
                else:
                    original = old.hashlib.sha256
                    def digest(raw):
                        check(raw)
                        return original(raw)
                    digest_patch = patch.object(old.hashlib, 'sha256', digest)
                with patch.dict(sys.modules, {'datasets': dataset}), patch.object(sys, 'argv', argv), digest_patch, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(mod.main(), 0)
                self.assertEqual(observed, [' 雪\r\n\n\ncafé\n'.encode()] * 2)
                snapshots.append({p.name: p.read_bytes() for p in out.iterdir()})
                for p in out.iterdir():
                    p.unlink()
            self.assertEqual(*snapshots)
            self.assertEqual(calls[:2], calls[2:])
            self.assertFalse(snapshots[1]['corpus_index.json'].endswith(b'\n'))


if __name__ == '__main__':
    unittest.main()
