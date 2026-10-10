"""Stored G3 source metadata: D32 stamps identities and never prepares again.

The own-digest of a supplied JSON stays integrity. Certified mode retains the
original explicitly requested metadata/cache validation; dev mode uses stored
teacher/preparation data without stat, adoption, quiescence or proof barriers.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time
from g3_pq_policy.digests import file_sha256hex as sha256_file
from g3_pq_policy.digests import bytes_sha256hex

SCHEMA = 'campaign.g3.prepared_source.v1'
SOURCE_MODEL = '/mnt/shared/models/GLM-5.3-Flash-BF16'
SOURCE_IDENTITY = '8bdd62ee422d82c895506a075031e3a44fb5c0023e2f00662ccb41410f8d5cde'
REFERENCE_BINDING_SHA = '23370e25f6b42e316f72d6cbc8f3f5747a65705b388c910da93545f487edc0fa'




def read_prepared_source_json(path, expected):
    raw = Path(path).read_bytes()
    if bytes_sha256hex(raw) != expected:
        raise ValueError(f'Prepared source proof bytes changed: {path}')
    return json.loads(raw)


def relative_name(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not name or str(p) != name:
        raise ValueError(f'Prepared source proof has an unsafe relative path: {name!r}')
    return name


def load_cache_owner(binding):
    path = Path(binding['path'])
    if sha256_file(path) != binding['sha256']:
        raise ValueError(f'Prepared source digest owner changed: {path}')
    name = '_g3_prepared_source_digest_owner'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load the bound source digest owner: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.SourceDigestCache


def _certified_prepared_source(model, proof_path, proof_sha256, *, expected_root=SOURCE_MODEL,
                               expected_identity=SOURCE_IDENTITY):
    """Original certified-only source validation."""
    started = time.monotonic()
    proof = read_prepared_source_json(proof_path, proof_sha256)
    if proof.get('schema') != SCHEMA:
        raise ValueError('Prepared source proof schema is not supported; return to CPU preparation')
    if proof.get('logical_source_root') != expected_root:
        raise ValueError('Prepared source proof belongs to another source root')
    if proof.get('identity_sha256') != expected_identity:
        raise ValueError('Prepared source proof belongs to another source identity')
    if proof.get('reference_binding_sha256') != REFERENCE_BINDING_SHA:
        raise ValueError('Prepared source proof has another upstream binding')
    if proof.get('source_domain') != 'GLM-5.3-Flash-BF16-upstream-G3':
        raise ValueError('Prepared source proof belongs to another source domain')
    identity = proof['identity']
    from tools.full_kl_teacher_payload import canonical_sha256
    if canonical_sha256(identity) != expected_identity:
        raise ValueError('Prepared source identity seal is invalid')
    root = Path(model).resolve(strict=True)
    cache_root = Path(proof['cache_directory']).resolve(strict=True)
    cache_files = proof['cache_files']
    if not isinstance(cache_files, list) or not cache_files:
        raise ValueError('Prepared source proof has no frozen cache entries')
    names = [relative_name(r['name']) for r in cache_files]
    if len(names) != len(set(names)):
        raise ValueError('Prepared source proof repeats a cache entry')
    if set(p.name for p in cache_root.glob('*.json')) != set(names):
        raise ValueError('Prepared source cache entry inventory changed')
    for entry in cache_files:
        p = cache_root / entry['name']
        if p.stat().st_size != entry['bytes'] or sha256_file(p) != entry['sha256']:
            raise ValueError(f'Prepared source cache entry changed: {entry["name"]}')
    cache_class = load_cache_owner(proof['cache_owner'])
    cache = cache_class(cache_root, source=root, read_only=True)
    shard_names = [relative_name(r['name']) for r in identity['shards']]
    if len(shard_names) != len(set(shard_names)):
        raise ValueError('Prepared source identity repeats a source shard')
    actual_names = {p.relative_to(root).as_posix() for p in root.rglob('*.safetensors')}
    if actual_names != set(shard_names):
        raise ValueError('Prepared source shard inventory changed; return to CPU preparation')
    bindings = proof['source_bindings']
    if len(bindings) != len(shard_names) or {r['name'] for r in bindings} != set(shard_names):
        raise ValueError('Prepared source stat bindings lack exact shard coverage')
    by_name = {r['name']: r for r in bindings}
    for shard in identity['shards']:
        p = root / shard['name']
        live = cache.fingerprint(p)
        expected = by_name[shard['name']]['key']
        # SourceDigestCache owns the reviewed portable identity: device and
        # absolute path are diagnostic, not a new key or a fingerprint rewrite.
        observed = cache.key_for_fingerprint(live)
        if observed != expected or live['size'] != shard['size']:
            raise ValueError(f'Prepared source stat binding changed: {shard["name"]}; return to CPU preparation')
        if cache.require_cached_sha256(p) != shard['sha256']:
            raise ValueError(f'Prepared source digest differs: {shard["name"]}')
    for metadata in identity['metadata']:
        p = root / relative_name(metadata['name'])
        before = cache.fingerprint(p)
        if before['size'] != metadata['size'] or sha256_file(p) != metadata['sha256']:
            raise ValueError(f'Prepared source metadata changed: {metadata["name"]}')
        if cache.fingerprint(p) != before:
            raise ValueError(f'Prepared source metadata changed while reading: {metadata["name"]}')
    from experiments.build_glm_tr3_teacher import require_reference_binding
    binding = read_prepared_source_json(proof['reference_binding'], REFERENCE_BINDING_SHA)
    require_reference_binding(binding, identity, root)
    receipt = {'schema': 'campaign.g3.prepared_source_consumer.v1',
               'source_preparation': str(proof_path), 'source_preparation_sha256': proof_sha256,
               'identity_sha256': expected_identity, 'cache_receipt': cache.receipt(),
               'source_body_bytes_read': 0, 'legacy_cache_fingerprints_rewritten': 0,
               'elapsed_seconds': time.monotonic() - started}
    return identity, receipt


def cached_checkpoint_identity(model, proof_path, proof_sha256, *, stored_identity=None):
    """Stored source identity; no new proof or recomputation in default dev mode."""
    identity, _receipt = validate_prepared_source(model, proof_path, proof_sha256, stored_identity=stored_identity)
    return identity


def validate_prepared_source(model, proof_path, proof_sha256, *, expected_root=SOURCE_MODEL,
                             expected_identity=SOURCE_IDENTITY, stored_identity=None):
    from g3_pq_policy.dev_mode import dev_mode_enabled, seal_check, NOT_COMPUTED, dev_stamp
    if not dev_mode_enabled():
        return _certified_prepared_source(model, proof_path, proof_sha256,
                                         expected_root=expected_root, expected_identity=expected_identity)
    if proof_path:
        proof = read_prepared_source_json(proof_path, proof_sha256)
        identity = proof["identity"]
        recorded = proof.get("identity_sha256", expected_identity)
    else:
        if stored_identity is None:
            raise ValueError("G3 requires stored source metadata, not a source rehash")
        identity, recorded = stored_identity, expected_identity
    seal_check("source identity", recorded, NOT_COMPUTED, where="G3 source metadata")
    receipt = {"schema": "campaign.g3.prepared_source_consumer.v1",
               "source_preparation": str(proof_path) if proof_path else None,
               "identity_sha256": recorded, "source_body_bytes_read": 0,
               "legacy_cache_fingerprints_rewritten": 0, **dev_stamp()}
    return identity, receipt
