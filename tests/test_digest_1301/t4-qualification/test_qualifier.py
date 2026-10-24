"""CPU differential checks against verbatim pre-edit consumer functions."""
import argparse
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/t4-qualification'
spec = importlib.util.spec_from_file_location('qualifier_digest_1301', ROOT / 'tools/qualify_t4_overlay.py')
current = importlib.util.module_from_spec(spec)
spec.loader.exec_module(current)


def original():
    # Only unchanged environmental dependencies are borrowed. Serialization and
    # hashing in the oracle come solely from the original consumer and stdlib.
    ns = dict(current.__dict__)
    ns.update(hashlib=hashlib, json=json, functools=functools, argparse=argparse,
              os=os, Path=Path, digest=lambda b: hashlib.sha256(b).hexdigest())
    module = types.ModuleType('qualifier_original')
    module.__dict__.update(ns)
    exec(compile((FIXTURES / 'qualifier_original.py').read_text(),
                 'qualifier_original.py', 'exec'), module.__dict__)
    return module


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(params=['current', 'original'])
def subject(request):
    return current if request.param == 'current' else original()


def setup_main(m, tmp_path, monkeypatch, *, pb=False):
    render = tmp_path / 'render'
    wire = tmp_path / 'wire'
    render.write_bytes(b'render')
    wire.write_bytes(b'wire')
    cell = {'qname': 'unit-\u03bb', 'format': 'tiny', 'render': str(render),
            'wire': str(wire), 'render_stat': m.catalog_stat_fence(render.stat()),
            'wire_stat': m.catalog_stat_fence(wire.stat()),
            'lax': {'unicode': '\u00e9', 'nonfinite': float('inf'), 'tuple': (1, 2)}}
    # The CLI parses JSON, so compare the same parsed cell (tuples become lists).
    cell = json.loads(json.dumps(cell))
    out = tmp_path / 'result.json'
    manifest = tmp_path / 'manifest.json'
    task = {'id': 'task', 'output_id': 'output', 'payload': {'cell': cell, 'output': str(out)}}
    batch = {'schema': 'prismabuild.task_batch.v1' if pb else 'prismaquant.t4_qualification_pilot.v1',
             'tasks': [task], 'result_manifest_path': str(manifest),
             'parent_key': 'parent', 'plan_key': 'plan', 'child_ordinal': 0}
    batch_path = tmp_path / 'batch.json'
    batch_path.write_text(json.dumps(batch))
    monkeypatch.setattr(sys, 'argv', ['qualify', '--pb-task-batch' if pb else '--pilot-batch', str(batch_path)])
    monkeypatch.setenv('T4_QUALIFY_READ_WORKERS', '1')
    # A tiny synchronous CPU adapter retains the real nested read/commit logic.
    def walk(tasks, read, commit, *, workers):
        assert workers == 1
        for t in tasks:
            commit(t, read(t))
    monkeypatch.setattr(m, '_drive_ordered_walk', walk)
    from prismaquant import residency_map
    monkeypatch.setattr(residency_map, 'residency_report', lambda: {'z': '\u00e9', 'a': 1})
    return cell, out, manifest, batch, batch_path


@pytest.mark.parametrize('pb', [False, True])
def test_main_publication_and_adoption(subject, tmp_path, monkeypatch, capsys, pb):
    m = subject
    cell, out, manifest, batch, batch_path = setup_main(m, tmp_path, monkeypatch, pb=pb)
    verified = {'qualification_seconds': 0.125, 'nested': {'z': '\u00e9', 'a': float('inf')}}
    reads = []
    monkeypatch.setattr(m, 'read_cell', lambda c: reads.append(c) or ('tiny', c))
    monkeypatch.setattr(m, 'finish_cell', lambda loaded: verified)
    value = {'qname': cell['qname'], 'format': cell['format'], 'cell_sha256': sha(compact(cell)),
             'verified_cell': verified, 'verified_cell_sha256': sha(compact(verified))}
    raw = compact(value) + b'\n'
    # Observe exact raw-file hash input; oracle expected bytes do not use owner.
    hashed = []
    profile_inputs = []
    if m is current:
        owner = m.DIRECT_ASCII_LAX
        class ObservedProfile:
            def sha256(self, obj):
                profile_inputs.append(('hash', compact(obj)))
                return owner.sha256(obj)

            def encoded(self, obj):
                profile_inputs.append(('publish', compact(obj)))
                return owner.encoded(obj)
        monkeypatch.setattr(m, 'DIRECT_ASCII_LAX', ObservedProfile())
    monkeypatch.setattr(m, 'digest', lambda b: hashed.append(b) or sha(b))
    m.main()
    if m is current:
        assert profile_inputs == [('hash', compact(cell)), ('hash', compact(verified)),
                                  ('publish', compact(value))]
    assert reads == [cell]
    assert out.read_bytes() == raw
    assert raw.endswith(b'\n') and not raw.endswith(b'\n\n')
    assert b'\\u00e9' in raw and b'Infinity' in raw
    assert hashed[-1] == raw
    expected = {k: batch[k] for k in ('parent_key', 'plan_key', 'child_ordinal')} if pb else {}
    expected.update(schema='prismabuild.child_result_manifest.v1' if pb else 'prismaquant.t4_qualification_pilot_result.v1',
                    results=[{'task_id': 'task', 'output_id': 'output', 'value_sha256': sha(raw)}])
    assert manifest.read_bytes() == (json.dumps(expected, sort_keys=True) + '\n').encode()
    first_stdout = capsys.readouterr().out
    assert first_stdout.splitlines()[-1] == json.dumps({'residency_report': {'z': '\u00e9', 'a': 1}}, sort_keys=True)
    assert not out.with_suffix('.tmp').exists()
    monkeypatch.setattr(m, 'read_cell', lambda c: pytest.fail('adoption must not read/decode'))
    monkeypatch.setattr(m, 'finish_cell', lambda c: pytest.fail('adoption must not decode'))
    for pinned in (False, True):
        if pinned:
            batch['tasks'][0]['payload']['existing_result_sha256'] = sha(raw)
            batch_path.write_text(json.dumps(batch))
        m.main()
        assert out.read_bytes() == raw
        assert json.loads(manifest.read_text())['results'][0]['value_sha256'] == sha(raw)


@pytest.mark.parametrize('damage', ['json', 'cell', 'verified', 'pinned', 'missing', 'fence'])
def test_checkpoint_refusal(subject, tmp_path, monkeypatch, damage):
    m = subject
    cell, out, manifest, batch, batch_path = setup_main(m, tmp_path, monkeypatch)
    value = {'cell_sha256': sha(compact(cell)), 'verified_cell': {'qualification_seconds': 0.1}}
    value['verified_cell_sha256'] = sha(compact(value['verified_cell']))
    if damage == 'cell':
        value['cell_sha256'] = '0' * 64
    if damage == 'verified':
        value['verified_cell_sha256'] = '0' * 64
    out.write_bytes(b'{' if damage == 'json' else compact(value))
    if damage in ('pinned', 'missing'):
        batch['tasks'][0]['payload']['existing_result_sha256'] = '0' * 64
        batch_path.write_text(json.dumps(batch))
    if damage == 'missing':
        out.unlink()
    if damage == 'fence':
        Path(cell['render']).write_bytes(b'changed')
    monkeypatch.setattr(m, 'read_cell', lambda c: pytest.fail('refusal must precede payload read'))
    monkeypatch.setattr(m, 'finish_cell', lambda c: pytest.fail('refusal must precede decode'))
    before = out.read_bytes() if out.exists() else None
    with pytest.raises((AssertionError, json.JSONDecodeError, RuntimeError)):
        m.main()
    assert not manifest.exists()
    assert (out.read_bytes() if out.exists() else None) == before


def test_mixed_keys_refuse_before_publication(subject, tmp_path, monkeypatch):
    m = subject
    cell, out, manifest, _, _ = setup_main(m, tmp_path, monkeypatch)
    monkeypatch.setattr(m, 'read_cell', lambda c: c)
    monkeypatch.setattr(m, 'finish_cell', lambda c: {'qualification_seconds': 0, 'mixed': {1: 'a', '1': 'b'}})
    with pytest.raises(TypeError):
        m.main()
    assert not out.exists() and not manifest.exists()


@pytest.mark.parametrize('direct_url', [None, '{ "url": "file:///caf\u00e9", "dir_info": {} }\n', '{bad'])
def test_decoder_provenance_text_hash(subject, monkeypatch, direct_url):
    m = subject
    m.decoder_provenance.cache_clear()
    import prismaquant.tessera_campaign_wire as wire
    distribution = types.SimpleNamespace(metadata={'Name': 'tessera-quant'}, version='tiny',
                                         read_text=lambda name: direct_url)
    monkeypatch.setattr(importlib.metadata, 'distribution', lambda name: distribution)
    monkeypatch.setattr(wire, 'require_checkpoint_identity_api', lambda: types.SimpleNamespace(encoder_source_sha256=lambda: 'source'))
    monkeypatch.setattr(m, '_decoder_identity', lambda _: {'cpu': True})
    if direct_url == '{bad':
        with pytest.raises(json.JSONDecodeError):
            m.decoder_provenance()
    else:
        result = m.decoder_provenance()
        assert result['distribution']['direct_url'] == (None if direct_url is None else json.loads(direct_url))
        assert result['distribution']['direct_url_sha256'] == (None if direct_url is None else sha(direct_url.encode()))
        assert result['installed_encoder_source_sha256'] == 'source'
    m.decoder_provenance.cache_clear()
