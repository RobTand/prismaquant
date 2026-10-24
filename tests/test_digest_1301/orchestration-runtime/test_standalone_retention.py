"""Operation-level retained-call evidence, without importing PrismaQuant.

Both arms run in separate isolated interpreters outside the checkout. Frozen
excerpts come from the ledger's original git blobs, not from current functions.
The import guard stays installed throughout the operations, not just loading.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/digest_1301/orchestration-runtime/standalone"

SCRIPT = r'''
import contextlib, hashlib, importlib.abc, importlib.util, io, json, os
from pathlib import Path
import sys, types
class NoPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'prismaquant', 'torch', 'tessera', 'prismabuild'}:
            raise AssertionError('package import attempted: ' + fullname)
sys.meta_path.insert(0, NoPackage())
spec = importlib.util.spec_from_file_location('standalone_under_test', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
root = Path.cwd()
def outcome(fn):
    try:
        return {'value': fn()}
    except Exception as exc:
        return {'error': type(exc).__name__, 'message': str(exc)}
result = {}
if sys.argv[2] == 'progress':
    m.time = types.SimpleNamespace(time=lambda: 1234.5)
    m.os.getpid = lambda: 17
    os.environ.pop('PRISMAQUANT_DEV_PROGRESS_STAMP', None)
    os.environ.pop(m.PATH_ENV, None)
    os.environ.pop(m.TOKEN_ENV, None)
    result['no_channel'] = [m.report('phase', 1), m.commit(1, 'phase'),
                            m.declare_staged_wait(['x'], since_unix=2),
                            m.declare_export_wait(['x'], since_unix=2)]
    path = root / 'progress'
    os.environ[m.PATH_ENV] = str(path)
    os.environ[m.TOKEN_ENV] = 'token-é'
    for name, call, suffix in [
        ('report', lambda: m.report('雪', '7', unit='é'), ''),
        ('commit', lambda: m.commit(8, '雪'), ''),
        ('staged', lambda: m.declare_staged_wait(['雪', 'a', '雪', 3], since_unix=10), m.STAGED_WAIT_SUFFIX),
        ('export', lambda: m.declare_export_wait(['雪', 'a', '雪', 3], since_unix=10), m.EXPORT_WAIT_SUFFIX),
    ]:
        result[name] = {'returned': call(), 'bytes': Path(str(path) + suffix).read_text()}
    result['empty_waits'] = [m.declare_staged_wait([], since_unix=2), m.declare_export_wait([], since_unix=2)]
    os.environ[m.PATH_ENV] = str(root / 'missing' / 'progress')
    result['write_errors'] = [outcome(lambda: m.report('x', 1)), outcome(lambda: m.commit(1, 'x')),
                              outcome(lambda: m.declare_staged_wait(['x'], since_unix=1)),
                              outcome(lambda: m.declare_export_wait(['x'], since_unix=1))]
    # Paths differ per independent interpreter; compare the exception type only.
    result['write_errors'][1].pop('message', None)
    result['temporaries'] = sorted(p.name for p in root.glob('*.tmp'))
else:
    image = {'Os': 'linux', 'Architecture': 'arm64', 'RootFS': {'Type': 'layers', 'Layers': ['sha256:' + 'a'*64]},
             'Config': {'Env': ['NAME=雪'], 'Labels': {'é': 'value'}}, 'Id': 'ignored'}
    result['image'] = m.image_content_sha256(image)
    result['image_metadata'] = m.image_content_sha256(dict(image, Id='other', Size=999))
    bad = dict(image, Config={'nan': float('nan')})
    result['image_nan'] = outcome(lambda: m.image_content_sha256(bad))
    result['image_invalid'] = outcome(lambda: m.image_content_sha256(dict(image, RootFS={})))
    package = root / 'prismaquant'; package.mkdir()
    (package / '__init__.py').write_bytes(b'# tiny package\n')
    (package / 'data').mkdir(); (package / 'data' / 'é.txt').write_bytes(b'\x00\xff')
    (package / '__pycache__').mkdir(); (package / '__pycache__' / 'ignored').write_bytes(b'ignored')
    (package / 'ignored.pyc').write_bytes(b'ignored')
    result['source'] = m.prismaquant_source_sha256(package)
    receipt = root / 'receipt.json'; receipt.write_bytes(b'{"git_commit":"' + b'b'*40 + b'","note":"receipt"}\n')
    result['receipt'] = m._receipt_sha256(receipt, expected_commit='b'*40)
    result['receipt_wrong_commit'] = outcome(lambda: m._receipt_sha256(receipt, expected_commit='c'*40))
    duplicate = root / 'duplicate.json'; duplicate.write_text('{"git_commit":"x","git_commit":"y"}')
    result['receipt_duplicate'] = outcome(lambda: m._receipt_sha256(duplicate, expected_commit='x'))
    result['supplemental_json'] = {}
    for label, text in [('array', '[]'), ('malformed', '{'), ('duplicate', '{"a":1,"a":2}'),
                        ('nonfinite', '{"n":NaN}')]:
        extra = root / (label + '.json'); extra.write_text(text)
        item = outcome(lambda: m._load_json_object(extra, where='supplemental'))
        if label == 'malformed':
            item.pop('message', None)  # Decoder diagnostics include each arm's path.
        if label == 'nonfinite':
            item = {'nan_is_accepted': item['value']['n'] != item['value']['n']}
        result['supplemental_json'][label] = item
    checkpoint = root / 'checkpoint'; checkpoint.mkdir()
    identity = checkpoint / 'identity.json'
    kwargs = dict(identity_path=identity, checkpoint_root=checkpoint, source_root=root, target='dense',
                  image_ref='sha256:'+'d'*64, image_id='sha256:'+'e'*64,
                  git_commit='b'*40, implementation_receipt=receipt)
    result['identity'] = m.write_or_verify_identity(**kwargs)
    result['identity_bytes'] = identity.read_text()
    result['resume'] = m.write_or_verify_identity(**kwargs)
    result['resume_bytes'] = identity.read_text()
    args = ['write-or-verify', '--identity', str(identity), '--checkpoint-root', str(checkpoint),
            '--source-root', str(root), '--target', 'dense', '--image-ref', kwargs['image_ref'],
            '--image-id', kwargs['image_id'], '--git-commit', kwargs['git_commit'],
            '--implementation-receipt', str(receipt)]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result['main_status'] = m.main(args)
    result['stdout'] = out.getvalue()
    (package / '__init__.py').write_bytes(b'# changed\n')
    result['changed_source'] = outcome(lambda: m.write_or_verify_identity(**kwargs))
    result['temporaries'] = sorted(p.name for p in checkpoint.glob('.*'))
assert not any(name.split('.')[0] in {'prismaquant', 'torch', 'tessera', 'prismabuild'} for name in sys.modules)
print(json.dumps(result, sort_keys=True))
'''


def run_arm(tmp_path, filename, arm, population):
    work = tmp_path / arm
    work.mkdir()
    source = (FIXTURES if arm == 'original' else ROOT / 'prismaquant') / filename
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('PRISMAQUANT_', 'PRISMABUILD_ACTION_PROGRESS_', 'PYTHON'))}
    process = subprocess.run([sys.executable, '-I', '-S', '-c', SCRIPT, str(source), population],
                             cwd=work, env=env, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    assert process.stderr == ''
    return json.loads(process.stdout)


def test_identity_operations(tmp_path):
    old = run_arm(tmp_path, 'container_runtime_identity.py', 'original', 'identity')
    current = run_arm(tmp_path, 'container_runtime_identity.py', 'current', 'identity')
    assert current == old
    # Independent preimage oracle: neither arm supplies its own expected digest.
    content = {'schema': 'prismaquant.container_image_content.v1', 'os': 'linux',
               'architecture': 'arm64', 'variant': '',
               'rootfs': {'Type': 'layers', 'Layers': ['sha256:' + 'a' * 64]},
               'config': {'Env': ['NAME=雪'], 'Labels': {'é': 'value'}}}
    raw = json.dumps(content, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    assert current['image'] == current['image_metadata'] == hashlib.sha256(raw).hexdigest()
    digest = hashlib.sha256()
    for name, data in [('__init__.py', b'# tiny package\n'), ('data/é.txt', b'\x00\xff')]:
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(4, 'big') + encoded + len(data).to_bytes(8, 'big') + data)
    assert current['source'] == digest.hexdigest()
    assert current['receipt'] == hashlib.sha256(b'{"git_commit":"' + b'b'*40 + b'","note":"receipt"}\n').hexdigest()
    payload = current['identity']
    assert payload['prismaquant_source_sha256'] == current['source']
    assert payload['implementation_receipt_sha256'] == current['receipt']
    assert current['resume'] == payload
    assert current['identity_bytes'] == current['resume_bytes'] == json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n'
    assert current['stdout'] == json.dumps(payload, sort_keys=True, allow_nan=False) + '\n'
    assert current['main_status'] == 0
    assert current['image_nan']['error'] == 'ValueError'
    assert current['supplemental_json']['nonfinite'] == {'nan_is_accepted': True}
    for key in ('array', 'malformed', 'duplicate'):
        assert current['supplemental_json'][key]['error'] == 'RuntimeIdentityError'
    for key in ('image_invalid', 'receipt_wrong_commit', 'receipt_duplicate', 'changed_source'):
        assert current[key]['error'] == 'RuntimeIdentityError'
    assert 'duplicate JSON member' in current['receipt_duplicate']['message']
    assert 'prismaquant_source_sha256' in current['changed_source']['message']
    assert current['temporaries'] == []


def test_progress_operations(tmp_path):
    old = run_arm(tmp_path, 'prismabuild_progress.py', 'original', 'progress')
    current = run_arm(tmp_path, 'prismabuild_progress.py', 'current', 'progress')
    assert current == old
    assert current['no_channel'] == [False] * 4
    assert current['empty_waits'] == [False, False]
    base = {'schema': 'prismabuild.action_progress.v1', 'token': 'token-é',
            'phase': '雪', 'reported_unix': 1234.5}
    records = {'report': dict(base, units_completed=7, unit='é'),
               'commit': dict(base, units_completed=8, unit=None),
               'staged': {'schema': 'prismabuild.staged_wait.v1', 'token': 'token-é',
                          'since_unix': 10.0, 'movers': ['3', 'a', '雪']},
               'export': {'schema': 'prismabuild.export_wait.v1', 'token': 'token-é',
                          'since_unix': 10.0, 'exports': ['3', 'a', '雪']}}
    for name, record in records.items():
        assert current[name] == {'returned': True, 'bytes': json.dumps(record, sort_keys=True) + '\n'}
        assert '\\u' in current[name]['bytes']
    assert current['write_errors'] == [{'value': False}, {'error': 'FileNotFoundError'},
                                       {'value': False}, {'value': False}]
    assert current['temporaries'] == []


def test_retention_ledger():
    ledger = json.loads((FIXTURES / 'retention_ledger.json').read_text())
    assert len(ledger['calls']) == 10
    assert len({row['id'] for row in ledger['calls']}) == 10
    assert len(ledger['supplemental_calls']) == 1
    for row in ledger['calls'] + ledger['supplemental_calls']:
        assert len(row['source_blob']) == 40
        assert row['decision'].startswith('retain')
        assert row['witness'] and row['operation_test'] in globals()
        assert row['original_outcome'] and row['current_outcome']
        frozen = (FIXTURES / row['original_fixture']).read_bytes()
        data = (ROOT / row['current_file']).read_bytes()
        assert hashlib.sha256(frozen).hexdigest() == row['fixture_sha256']
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == row['current_blob']
        line, column = map(int, row['current_id'].rsplit('@', 1)[1].split(':'))
        tree = ast.parse(data)
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and (n.lineno, n.col_offset) == (line, column)
                    and ast.unparse(n.func) == row['callee'])
        assert ast.unparse(call.func) == row['callee']
        original = ast.parse(frozen)
        scope = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == row['scope'])
        assert any(ast.dump(call) == ast.dump(n) for n in ast.walk(scope) if isinstance(n, ast.Call))
    assert ledger['supplemental_json']
    identity_loader = (ROOT / 'tools/tessera_campaign_container.py').read_text()
    assert 'runpy.run_path' in identity_loader and 'container_runtime_identity.py' in identity_loader
    progress_contract = (ROOT / 'tests/test_tessera_campaign_container.py').read_text()
    assert 'run_path' in progress_contract and 'prismabuild_progress.py' in progress_contract
