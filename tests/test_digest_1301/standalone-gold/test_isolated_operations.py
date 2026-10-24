"""Operation goldens for stdlib-only deployment copies, not digest-owner oracles.

Every subject is copied without prismaquant and executed by python -I -S.
Literal bytes and stdlib hashlib in these tests are independent references.
No census, acceptance ledger, or production source is rewritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[3]


def stage(tmp_path, source, deployed=None):
    destination = tmp_path / (deployed or source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / source, destination)
    return destination


def isolated(tmp_path, subject, body):
    script = """
import runpy, sys
from pathlib import Path
assert sys.flags.isolated and sys.flags.no_site
assert not any(name == 'prismaquant' or name.startswith('prismaquant.') for name in sys.modules)
m = runpy.run_path(sys.argv[1], run_name='operation_subject')
""" + textwrap.dedent(body) + """
assert not any(name == 'prismaquant' or name.startswith('prismaquant.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(subject)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_sampler_append_and_error_bytes(tmp_path):
    subject = stage(tmp_path, 'tools/dspark_memavailable_sampler.py')
    isolated(tmp_path, subject, r'''
        g = m['_append_sample'].__globals__
        g['_utc_now'] = lambda: '2030-01-02T03:04:05Z'
        g['_mem_available'] = lambda: 123456
        ledger = Path('samples.jsonl')
        ledger.write_bytes(b'prior\n')
        m['_append_sample'](ledger, sequence=7, phase='measured')
        m['_append_sample'](ledger, sequence=8, phase='post')
        assert ledger.read_bytes() == (
            b'prior\n'
            b'{"mem_available_bytes":123456,"observed_at":"2030-01-02T03:04:05Z","phase":"measured","sequence":7}\n'
            b'{"mem_available_bytes":123456,"observed_at":"2030-01-02T03:04:05Z","phase":"post","sequence":8}\n'
        )
        m['_record_error'](Path('.'), ValueError('bad \u00e9'))
        expected = (b'{\n  "detail": "bad \\u00e9",\n  "error_type": "ValueError",\n'
                    b'  "observed_at": "2030-01-02T03:04:05Z",\n'
                    b'  "schema": "prismaquant.dspark_memory_sampler_error.v1"\n}\n')
        assert Path('error.json').read_bytes() == expected
        assert Path('error.json').stat().st_mode & 0o777 == 0o600
        m['_record_error'](Path('.'), RuntimeError('must not overwrite'))
        assert Path('error.json').read_bytes() == expected
    ''')


def test_sampler_refuses_unsafe_ledger_and_unarmed_start(tmp_path):
    subject = stage(tmp_path, 'tools/dspark_memavailable_sampler.py')
    isolated(tmp_path, subject, r'''
        import os
        target = Path('target')
        target.write_bytes(b'keep\n')
        Path('alias').symlink_to('target')
        for path in (Path('alias'), Path('absent')):
            try:
                m['_append_sample'](path, sequence=0, phase='startup')
            except OSError:
                pass
            else:
                raise AssertionError('unsafe ledger accepted')
        assert target.read_bytes() == b'keep\n'
        assert not Path('absent').exists()
        # /dev/null opens successfully but fstat must reject the nonregular fd.
        try:
            m['_append_sample'](Path(os.devnull), sequence=0, phase='startup')
        except m['SamplerError'] as exc:
            assert str(exc) == 'memory sample ledger is not a regular file'
        else:
            raise AssertionError('nonregular ledger accepted')
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b'0')
        os.close(write_fd)
        assert m['main'](['--state-dir', str(Path.cwd()), '--start-fd', str(read_fd),
                          '--interval-ms', '1']) == 2
        import json
        error = json.loads(Path('error.json').read_bytes())
        assert error['detail'] == 'memory sampler was not armed by its collector'
        assert not Path('samples.jsonl').exists()
        assert not Path('ready').exists()
    ''')


def test_runtime_snapshot_materializes_exact_closure_and_refuses_changes(tmp_path):
    subject = stage(tmp_path, 'tools/prismaquant_runtime_snapshot.py')
    source = tmp_path / 'source'
    source.mkdir()
    files = {
        'prismaquant/__init__.py': b'# inert package\n',
        'prismaquant/container_runtime_identity.py': b'IDENTITY = 1\n',
        'tools/prismaquant_runtime_snapshot.py': subject.read_bytes(),
        'run.sh': b'#!/bin/sh\nexit 0\n',
        'data.bin': b'\x00\xff\n',
    }
    for name, data in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o755 if name == 'run.sh' else 0o644)
    (source / 'alias').symlink_to('data.bin')

    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()

    git('init', '-q')
    git('add', '.')
    git('-c', 'user.name=Operation Test', '-c', 'user.email=test@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture')
    commit, tree = git('rev-parse', 'HEAD'), git('rev-parse', 'HEAD^{tree}')
    entries = [{'path': name, 'type': 'file', 'bytes': len(data),
                'executable': name == 'run.sh', 'sha256': hashlib.sha256(data).hexdigest()}
               for name, data in files.items()]
    entries.append({'path': 'alias', 'type': 'symlink', 'target': 'data.bin'})
    entries.sort(key=lambda entry: entry['path'])
    closure = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    expected = dict(schema='prismaquant.runtime_source_snapshot.v1', commit=commit,
                    tree=tree, closure_sha256=closure, entry_count=len(entries), entries=entries)
    (tmp_path / 'expected.json').write_bytes((json.dumps(expected, sort_keys=True, indent=2) + '\n').encode())
    # The archive must use committed bytes, not dirty or untracked worktree data.
    (source / 'data.bin').write_bytes(b'dirty')
    (source / 'untracked').write_bytes(b'not archived')
    isolated(tmp_path, subject, f'''
        import json
        expected = json.loads(Path('expected.json').read_bytes())
        result = m['materialize_snapshot'](Path('source'), Path('cache'), commit={commit!r})
        snapshot = Path(result['snapshot'])
        assert snapshot.name == {commit + '-' + tree[:12]!r}
        assert result == {{'schema': expected['schema'], 'snapshot': str(snapshot),
                          'commit': expected['commit'], 'tree': expected['tree'],
                          'closure_sha256': expected['closure_sha256'], 'entry_count': len(expected['entries'])}}
        assert (snapshot / '.prismaquant-runtime-snapshot.json').read_bytes() == Path('expected.json').read_bytes()
        assert (snapshot / 'data.bin').read_bytes() == b'\\x00\\xff\\n'
        assert m['materialize_snapshot'](Path('source'), Path('cache'), commit={commit!r}) == result
        assert not list(Path('cache').glob('*.tmp-*'))
        def verify():
            return m['verify_snapshot'](snapshot, expected_commit={commit!r},
                expected_tree={tree!r}, expected_closure_sha256={closure!r})
        assert verify() == result
        for mode in ('extra', 'content', 'executable', 'symlink'):
            if mode == 'extra':
                (snapshot / 'extra').write_bytes(b'x')
            elif mode == 'content':
                (snapshot / 'data.bin').write_bytes(b'changed')
            elif mode == 'executable':
                (snapshot / 'run.sh').chmod(0o644)
            else:
                (snapshot / 'alias').unlink()
                (snapshot / 'alias').symlink_to('run.sh')
            try:
                verify()
            except m['SnapshotError'] as exc:
                assert 'exact tracked ledger' in str(exc)
            else:
                raise AssertionError(mode + ' change accepted')
            if mode == 'extra':
                (snapshot / 'extra').unlink()
            elif mode == 'content':
                (snapshot / 'data.bin').write_bytes(b'\\x00\\xff\\n')
            elif mode == 'executable':
                (snapshot / 'run.sh').chmod(0o755)
        assert not (snapshot / 'untracked').exists()
    ''')


def test_staged_worker_hashing_recipes(tmp_path):
    subject = stage(tmp_path, 'tools/tessera_fleet/model_worker.py', 'sealed/worker.py')
    isolated(tmp_path, subject, r'''
        import hashlib
        data = b'\x00\xff\r\n' + b'x' * (8 * 1024 * 1024 + 17)
        Path('weights.bin').write_bytes(data)
        assert m['digest_file']('weights.bin') == hashlib.sha256(data).hexdigest()
        Path('empty').write_bytes(b'')
        assert m['digest_file']('empty') == hashlib.sha256(b'').hexdigest()
        value = {'z': [1, True, None], 'a': '\u00e9'}
        assert m['digest_json'](value) == hashlib.sha256(b'{"a":"\\u00e9","z":[1,true,null]}').hexdigest()
        assert m['digest_json']({'x': float('nan')}) == hashlib.sha256(b'{"x":NaN}').hexdigest()
        assert m['digest_json']({'x': float('inf')}) == hashlib.sha256(b'{"x":Infinity}').hexdigest()
        for invalid in ({1: 'number', 'x': 'string'}, {'x': object()}):
            try:
                m['digest_json'](invalid)
            except TypeError:
                pass
            else:
                raise AssertionError('native JSON refusal lost')
        cycle = []; cycle.append(cycle)
        try:
            m['digest_json'](cycle)
        except ValueError as exc:
            assert str(exc) == 'Circular reference detected'
        else:
            raise AssertionError('circular JSON accepted')
    ''')


def test_staged_worker_atomic_json_publication_and_failure(tmp_path):
    subject = stage(tmp_path, 'tools/tessera_fleet/model_worker.py', 'sealed/worker.py')
    isolated(tmp_path, subject, r'''
        g = m['atomic_json'].__globals__
        original = g['os'].replace
        path = Path('out/record.json')
        def observe(src, dst):
            assert Path(src).parent == path.parent
            assert Path(src).read_bytes() == b'{"a": "\\u00e9", "z": 2}\n'
            assert not path.exists() or path.read_bytes() == b'old'
            original(src, dst)
        g['os'].replace = observe
        m['atomic_json'](path, {'z': 2, 'a': '\u00e9'})
        assert path.read_bytes() == b'{"a": "\\u00e9", "z": 2}\n'
        path.write_bytes(b'old')
        m['atomic_json'](path, {'z': 2, 'a': '\u00e9'})
        assert path.read_bytes() == b'{"a": "\\u00e9", "z": 2}\n'
        path.write_bytes(b'old')
        def refuse(src, dst):
            assert Path(src).exists()
            raise OSError('publication refused')
        g['os'].replace = refuse
        try:
            m['atomic_json'](path, {'z': 2, 'a': '\u00e9'})
        except OSError as exc:
            assert str(exc) == 'publication refused'
        else:
            raise AssertionError('replace failure swallowed')
        assert path.read_bytes() == b'old'
        assert sorted(p.name for p in path.parent.iterdir()) == ['record.json']
        g['os'].replace = original
    ''')


@pytest.mark.parametrize('mutation', ['bytes', 'extra', 'missing', 'barrier', 'nonlocal'])
def test_staged_worker_output_verification(tmp_path, mutation):
    subject = stage(tmp_path, 'tools/tessera_fleet/model_worker.py', 'sealed/worker.py')
    isolated(tmp_path, subject, f'''
        import hashlib, json
        out = Path('part'); out.mkdir()
        (out / 'weights.bin').write_bytes(b'weights\\x00')
        (out / 'config.json').write_bytes(b'{{}}\\n')
        expected = {{'schema': 'prismabuild.tessera-model.v1', 'contract': 'fixture',
            'index': 1, 'count': 2, 'files': {{
                'weights.bin': hashlib.sha256(b'weights\\x00').hexdigest(),
                'config.json': hashlib.sha256(b'{{}}\\n').hexdigest()}}}}
        record = m['output_record'](out, {{'contract': 'fixture', 'count': 2}}, index=1)
        assert record == expected
        m['atomic_json'](out / 'pb-result.json', record)
        assert m['verify_output_record'](out, expected) == expected
        mutation = {mutation!r}
        message = 'output file population changed'
        if mutation == 'bytes':
            (out / 'weights.bin').write_bytes(b'changed')
            message = 'output changed: weights.bin'
        elif mutation == 'extra':
            (out / 'extra').write_bytes(b'x')
        elif mutation == 'missing':
            (out / 'weights.bin').unlink()
        elif mutation == 'barrier':
            expected['contract'] = 'other'
            message = 'part differs from its barrier record'
        else:
            record['files']['../escape'] = '0' * 64
            (out / 'pb-result.json').write_text(json.dumps(record))
        try:
            m['verify_output_record'](out, expected if mutation == 'barrier' else None)
        except ValueError as exc:
            assert str(exc) == message
        else:
            raise AssertionError('mutated output accepted')
    ''')


def test_duplication_inventory_reference_shingles_and_baseline_write(tmp_path):
    subject = stage(tmp_path, 'tools/duplication_inventory.py')
    isolated(tmp_path, subject, r'''
        import hashlib
        # Reference both primitive operations in this unchanged standalone tool:
        # blake2b shingle hashing and sorted, indented baseline publication.
        tokens = ['FunctionDef', 'arguments', 'Return', 'Call:hash', 'Name', 'Load', 'Constant']
        reference = frozenset(int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), 'big') for raw in (
            b'FunctionDef\x1farguments\x1fReturn\x1fCall:hash\x1fName\x1fLoad',
            b'arguments\x1fReturn\x1fCall:hash\x1fName\x1fLoad\x1fConstant',
        ))
        assert m['_shingles'](tokens) == reference
        assert m['_shingles'](tokens[:5]) == frozenset()
        Path('prismaquant').mkdir()
        Path('prismaquant/a.py').write_text('def helper():\n    return 1\n')
        Path('prismaquant/b.py').write_text('def helper():\n    return 2\n')
        # Keep the deployed scanner out of its own tiny subject tree.
        root = Path('tiny'); root.mkdir()
        import shutil
        shutil.move('prismaquant', root / 'prismaquant')
        (root / 'tools').mkdir()
        g = m['main'].__globals__
        scan = m['scan']
        g['scan'] = lambda: scan(root)
        g['BASELINE'] = root / 'tests/fixtures/baseline.json'
        sys.argv = ['duplication_inventory.py', '--write-baseline']
        assert m['main']() == 0
        assert g['BASELINE'].read_bytes() == (
            b'{\n "must_differ": [],\n "near_duplicates": [],\n "primitive_digest_sites": [],\n'
            b' "same_name_helpers": {\n  "helper": [\n   "prismaquant/a.py",\n   "prismaquant/b.py"\n  ]\n }\n}\n'
        )
    ''')
