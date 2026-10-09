"""Certified explicit metadata validation; default dev reuse covered separately."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import g3_prepared_source as p
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    owner_file = Path(os.environ.get(
        'G3_TEST_CACHE_OWNER',
        '/mnt/shared/tessera-measurements/t8-nextarm-source-preparation-20261005/'
        'sources/tessera-owner/src/tessera/source_digest_cache.py'))
    cls = p.load_cache_owner({'path': str(owner_file), 'sha256': p.sha256_file(owner_file)})
    root = tmp_path / 'model'
    root.mkdir()
    shard = root / 'model.safetensors'
    shard.write_bytes(b'actual unit-test source body')
    config = root / 'config.json'
    config.write_text('{"model_type":"fixture"}\n')
    cache_dir = tmp_path / 'source-digests'
    cache_dir.mkdir()
    cache = cls(cache_dir, source=root, quiescent_seconds=0)
    fp = cache.fingerprint(shard)
    digest = p.sha256_file(shard)
    cache.adopt(shard, digest, fingerprint=fp, writer={'kind': 'unit-test-verified-full-read'})
    identity = {'schema': 'prismaquant.source_checkpoint.identity.v1',
                'shards': [{'name': shard.name, 'size': shard.stat().st_size, 'sha256': digest}],
                'metadata': [{'name': config.name, 'size': config.stat().st_size, 'sha256': p.sha256_file(config)}]}
    identity['content_sha256'] = _canonical(identity)
    identity_sha = _canonical(identity)
    binding = tmp_path / 'upstream.json'
    binding.write_text('{}')
    binding_sha = p.sha256_file(binding)
    monkeypatch.setattr(p, 'REFERENCE_BINDING_SHA', binding_sha)
    payload = types.ModuleType('tools.full_kl_teacher_payload')
    payload.canonical_sha256 = _canonical
    teacher = types.ModuleType('experiments.build_glm_tr3_teacher')
    teacher.require_reference_binding = lambda *args: None  # upstream owner is qualified separately
    monkeypatch.setitem(sys.modules, 'tools.full_kl_teacher_payload', payload)
    monkeypatch.setitem(sys.modules, 'experiments.build_glm_tr3_teacher', teacher)
    cache_files = [{'name': f.name, 'bytes': f.stat().st_size, 'sha256': p.sha256_file(f)} for f in cache_dir.glob('*.json')]
    proof = {'schema': p.SCHEMA, 'logical_source_root': str(root),
             'source_domain': 'GLM-5.3-Flash-BF16-upstream-G3',
             'identity': identity, 'identity_sha256': identity_sha,
             'reference_binding': str(binding), 'reference_binding_sha256': binding_sha,
             'cache_directory': str(cache_dir), 'cache_files': cache_files,
             'cache_owner': {'path': str(owner_file), 'sha256': p.sha256_file(owner_file)},
             'source_bindings': [{'name': shard.name, 'key': cache.key_for_fingerprint(fp), 'fingerprint': fp, 'sha256': digest}]}
    path = tmp_path / 'prepared.json'
    path.write_text(json.dumps(proof))
    return p, root, shard, cache_dir, proof, path, identity_sha


def _run(data):
    p, root, _shard, _cache, _proof, path, identity_sha = data
    return p.validate_prepared_source(root, path, p.sha256_file(path), expected_root=str(root), expected_identity=identity_sha)


def test_prepared_identity_hit_uses_only_recorded_digest(prepared, monkeypatch):
    p, _root, _shard, _cache, proof, _path, _sha = prepared
    identity, receipt = _run(prepared)
    assert identity == proof['identity']
    assert receipt['source_body_bytes_read'] == 0
    assert receipt['legacy_cache_fingerprints_rewritten'] == 0
    assert receipt['cache_receipt']['hashed_shards'] == 0
    assert receipt['cache_receipt']['cached_shards'] == 1


@pytest.mark.parametrize('kind', ['size', 'mtime', 'inode', 'same_size_rewrite'])
def test_changed_source_stat_refuses_without_rehash(prepared, kind):
    p, _root, shard, cache_dir, _proof, _path, _sha = prepared
    entries = {f.name: f.read_bytes() for f in cache_dir.glob('*.json')}
    st = shard.stat()
    if kind == 'size':
        shard.write_bytes(shard.read_bytes() + b'x')
    elif kind == 'mtime':
        os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    elif kind == 'inode':
        replacement = shard.with_suffix('.replacement')
        replacement.write_bytes(shard.read_bytes())
        os.replace(replacement, shard)
    else:
        shard.write_bytes(b'X' * st.st_size)
        os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(ValueError, match='stat binding changed'):
        _run(prepared)
    assert {f.name: f.read_bytes() for f in cache_dir.glob('*.json')} == entries


def test_cache_entry_bytes_are_frozen(prepared):
    _p, _root, _shard, cache_dir, _proof, _path, _sha = prepared
    next(cache_dir.glob('*.json')).write_text('{}')
    with pytest.raises(ValueError, match='cache entry changed'):
        _run(prepared)


def test_missing_cache_entry_refuses(prepared):
    _p, _root, _shard, cache_dir, _proof, _path, _sha = prepared
    next(cache_dir.glob('*.json')).unlink()
    with pytest.raises(ValueError, match='inventory changed'):
        _run(prepared)


def test_source_metadata_is_always_bound(prepared):
    _p, root, _shard, _cache, _proof, _path, _sha = prepared
    (root / 'config.json').write_text('{"model_type":"changed"}\n')
    with pytest.raises(ValueError, match='source metadata changed'):
        _run(prepared)


@pytest.mark.parametrize('field,value', [('logical_source_root', '/different'),
                                        ('source_domain', 'another-domain'),
                                        ('reference_binding_sha256', '0' * 64)])
def test_foreign_prepared_proof_refuses(prepared, field, value):
    _p, _root, _shard, _cache, proof, path, _sha = prepared
    proof[field] = value
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='another'):
        _run(prepared)


def test_changed_proof_seal_refuses(prepared):
    p, root, _shard, _cache, _proof, path, identity_sha = prepared
    old_sha = p.sha256_file(path)
    path.write_text('{}')
    with pytest.raises(ValueError, match='proof bytes changed'):
        p.validate_prepared_source(root, path, old_sha, expected_root=str(root), expected_identity=identity_sha)
