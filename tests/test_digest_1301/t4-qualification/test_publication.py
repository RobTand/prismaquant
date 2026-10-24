"""CPU metadata publication contracts, not cell/GPU qualification.

Execute the actual publication statements, with small already-assembled records.
The independent fixture holds the original statements, not owner-derived values;
upstream catalog science and the 864-qname enumeration are outside this test.
"""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import digests

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / 'tests/fixtures/digest_1301/t4-qualification/publication_original.json'
ORIGINAL = json.loads(FIXTURE.read_text())


def functions(name, wanted, env):
    source = (ROOT / 'tools' / (name + '.py')).read_text()
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / 'tools' / name), 'exec'), env)
    return env


def live_publication(name):
    tree = ast.parse((ROOT / 'tools' / (name + '.py')).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    target = 'out' if name == 'extract_layer8_native_metadata' else 'raw'
    # Last assignment to this variable starts the real consumer publication block.
    index = max(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == target for t in n.targets))
    return compile(ast.Module(body=main.body[index:], type_ignores=[]), name, 'exec')


def run_publication(code, name, root, record):
    out = root / 'published.json'
    feeds = []
    def sha(raw):
        feeds.append(raw)
        return hashlib.sha256(raw).hexdigest()
    env = dict(json=json, os=os, Path=Path, sha=sha, result=deepcopy(record),
               catalog=deepcopy(record), out=out, ROOT=root, layer=8,
               cells=[{'format': 'F'}], formats=['F'], carried_from=[],
               rows=[{}], FORMATS=['F'], groups={'w2': {'scale': [1.0]}},
               started=2.0, time=SimpleNamespace(monotonic=lambda: 5.0),
               DIRECT_ASCII_LAX=digests.DIRECT_ASCII_LAX,
               DIRECT_ASCII_INDENT2_LAX=digests.DIRECT_ASCII_INDENT2_LAX)
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exec(code, env)
    out = env['out']
    return out.read_bytes(), stdout.getvalue(), feeds, env


def test_publication_bytes_hash_feeds_and_refusals(tmp_path):
    for name in ('build_t4_overlay_catalog', 'check_t4_overlay_prepare', 'extract_layer8_native_metadata'):
        directory = tmp_path / name
        directory.mkdir()
        record = {'z': '\u00e9\ud800', 'schema': 'metadata', 'status': 'pending',
                  'nested': {'b': float('inf'), 'a': -0.0}, 'cells': []}
        original = compile(ORIGINAL[name]['original_source'], 'independent-original', 'exec')
        old_bytes, old_stdout, old_feeds, old_env = run_publication(original, name, directory, record)
        old_env['out'].unlink()
        new_bytes, new_stdout, new_feeds, env = run_publication(live_publication(name), name, directory, record)
        assert (new_bytes, new_stdout, new_feeds) == (old_bytes, old_stdout, old_feeds)
        assert new_bytes.endswith(b'\n')
        assert len(new_feeds) == 1 and new_feeds == [new_bytes]
        assert len(hashlib.sha256(new_bytes).hexdigest()) == 64
        # Preserve original exclusive-write / existing-output refusal.
        with pytest.raises((AssertionError, FileExistsError)):
            exec(live_publication(name), env)
        assert env['out'].read_bytes() == old_bytes
        env['out'].unlink()
        for value in ({'mixed': {1: 'int', 'x': 'str'}}, {'unsupported': object()}):
            for code in (original, live_publication(name)):
                with pytest.raises(TypeError):
                    run_publication(code, name, directory, value)
                assert not env['out'].exists()


def test_assembler_pretty_document_and_publication(tmp_path):
    env = functions('assemble_t4_overlay', {'doc', 'publish'},
                    dict(json=json, os=os, sha=digests.bytes_sha256hex,
                         DIRECT_ASCII_INDENT2_LAX=digests.DIRECT_ASCII_INDENT2_LAX))
    old = {'json': json}
    exec(ORIGINAL['assemble_t4_overlay']['original_source'], old)
    for value in ({'z': '\u00e9', 'a': [float('inf'), -0.0]}, {10: 'ten', 9: 'nine'}):
        assert env['doc'](value) == old['doc'](value)
    path = tmp_path / 'overlay' / 'plan.json'
    raw = env['doc']({'status': 'metadata_only', 'render': '\u00e9'})
    binding = env['publish'](path, raw)
    assert path.read_bytes() == raw
    assert binding == {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    assert not path.with_suffix('.json.tmp').exists()
    with pytest.raises(AssertionError):
        env['publish'](path, b'changed')
    assert path.read_bytes() == raw
    for value in ({1: 2, 'x': 3}, {'object': object()}):
        for doc in (env['doc'], old['doc']):
            with pytest.raises(TypeError):
                doc(value)


def test_prepare_bound_json_authenticates_before_decode(tmp_path):
    events = []
    def loads(raw):
        events.append('decode')
        return json.loads(raw)
    def sha(raw):
        events.append(('hash', raw))
        return digests.bytes_sha256hex(raw)
    env = functions('check_t4_overlay_prepare', {'bound_json'},
                    dict(Path=Path, sha=sha, json=SimpleNamespace(loads=loads)))
    path = tmp_path / 'input.json'
    path.write_bytes(b'{"z": "\\u00e9", "a": 1}\n')
    assert env['bound_json']({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}) == {'z': '\u00e9', 'a': 1}
    assert events == [('hash', path.read_bytes()), 'decode']
    events.clear()
    path.write_bytes(b'not JSON')
    with pytest.raises(AssertionError, match='sha256'):
        env['bound_json']({'path': str(path), 'sha256': '0' * 64})
    assert events == [('hash', b'not JSON')]
    # The application exposes the settled byte owner directly, not a local wrapper.
    tree = ast.parse((ROOT / 'tools/check_t4_overlay_prepare.py').read_text())
    assert any(isinstance(n, ast.ImportFrom) and n.module == 'prismaquant.digests'
               and any(a.name == 'bytes_sha256hex' and a.asname == 'sha' for a in n.names)
               for n in tree.body)
