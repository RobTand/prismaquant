"""Original expected identity uses bound descriptors, never mutable pool/cache proofs."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import cost_streaming as cs, layer_streaming as ls
from prismaquant.source_generation import original_checkpoint_description
from prismaquant import joint_cost_stage_a as stage_a
from test_original_streaming_bootstrap_2010 import original_model, _context  # noqa: F401
from test_capture_original_material import material, _owner, _forget_state  # noqa: F401

pytestmark = pytest.mark.own_process


@pytest.fixture
def original_runner(original_model):
    m = original_model
    owner = _owner(m)
    context = _context(m, owner, m['root'])
    state = {'owner': owner, 'runner': cs.StreamedCausalLM(context, ls._source_profile(m['root'], owner)),
             'model': str(m['root']), 'source_root': str(Path(m['paths']['config.json']).parent)}
    yield state
    state['runner'].shutdown()
    state.clear()
    del context
    gc.collect()
    owner.close()


def _no_pool_proof(monkeypatch):
    for name in ('_local_checkpoint_shards', '_streamed_identity_stat_fingerprint', '_hash_source_shards'):
        monkeypatch.setattr(cs, name, lambda *a, **kw: pytest.fail('mutable pool/stat/hash proof entered'))


def test_original_model_identity_never_reads_pool_or_legacy_cache(original_runner, monkeypatch):
    s = original_runner
    _no_pool_proof(monkeypatch)
    identity = cs.build_streamed_model_identity(s['runner'], s['model'])
    assert cs.validate_streamed_model_identity(identity, where='original') == identity
    assert identity['shards']
    assert identity['checkpoint_weight_map']
    assert s['owner'].receipt()['automatic_capture_qualified'] is False


def test_same_path_same_signature_pool_mutation_cannot_change_original_identity(original_runner, monkeypatch):
    s = original_runner
    _no_pool_proof(monkeypatch)
    first = cs.build_streamed_model_identity(s['runner'], s['model'])
    path = Path(s['model']) / 'model.safetensors.index.json'
    before = path.stat()
    path.write_bytes(b'x' * before.st_size)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    # Original authority is the independently authenticated descriptor, not
    # this path's timestamps or bytes. Future deliveries still verify theirs.
    assert cs.build_streamed_model_identity(s['runner'], s['model']) == first


@pytest.mark.parametrize('cache', ['identity_cache_path', 'identity_cache_bytes', 'digest_cache_path'])
def test_original_identity_does_not_adopt_stat_cache(original_runner, monkeypatch, cache):
    s = original_runner
    _no_pool_proof(monkeypatch)
    value = b'forged' if cache.endswith('bytes') else '/unopened/forged'
    with pytest.raises(RuntimeError, match='original.*cache'):
        cs.build_streamed_model_identity(s['runner'], s['model'], **{cache: value})


@pytest.mark.parametrize('broken', ['config', 'index', 'shard', 'missing-live-name',
                                    'missing-aux-proof', 'incomplete', 'forged-owner'])
def test_original_config_roster_and_proof_divergence_refuse(original_runner, monkeypatch, broken):
    s = original_runner
    _no_pool_proof(monkeypatch)
    if broken == 'config':
        s['runner'].model.config.hidden_size += 1
    elif broken == 'index':
        key = next(iter(s['runner'].context.weight_ckpt))
        s['runner'].context.weight_ckpt[key] = 'forged.checkpoint.tensor'
    elif broken == 'shard':
        key = next(iter(s['runner'].context.weight_shard))
        s['runner'].context.weight_shard[key] = '/unopened/forged.safetensors'
    elif broken == 'missing-live-name':
        key = next(iter(s['runner'].context.weight_ckpt))
        del s['runner'].context.weight_ckpt[key]
        del s['runner'].context.weight_shard[key]
    elif broken == 'missing-aux-proof':
        s['owner']._original['verified'].pop('config.json')
    elif broken == 'incomplete':
        s['runner'].context.source_snapshot_only = True
    else:
        s['runner'].context.source_authentication = SimpleNamespace(is_qualified_original_material=True)
    with pytest.raises(RuntimeError, match='original'):
        cs.build_streamed_model_identity(s['runner'], s['model'])


def test_original_source_checkpoint_identity_uses_existing_schema(original_runner, monkeypatch):
    s = original_runner
    legacy = cs.build_source_checkpoint_identity(s['source_root'])
    _no_pool_proof(monkeypatch)
    identity = cs.build_source_checkpoint_identity(s['model'], source_authentication=s['owner'])
    assert identity['schema'] == cs.SOURCE_CHECKPOINT_IDENTITY_SCHEMA
    assert {row['name'] for row in identity['metadata']} == {'config.json', 'model.safetensors.index.json'}
    assert identity['content_sha256']
    assert identity == legacy


def test_stage_a_original_identity_bypasses_legacy_proof_inputs(original_runner, monkeypatch):
    from prismaquant import tessera_joint_aura
    s = original_runner
    _no_pool_proof(monkeypatch)
    monkeypatch.setattr(tessera_joint_aura, 'source_identity_proof_kwargs',
                        lambda *a: pytest.fail('legacy stat proof entered'))
    config = {'model': s['model']}
    identity = stage_a._stage_a_source_identity(s['runner'], config, None)
    assert identity == cs.build_streamed_model_identity(s['runner'], s['model'])


@pytest.mark.parametrize('cache', ['source_identity_cache', 'source_digest_cache'])
def test_stage_a_original_identity_rejects_legacy_config_cache(original_runner, monkeypatch, cache):
    s = original_runner
    _no_pool_proof(monkeypatch)
    with pytest.raises(RuntimeError, match='original.*cache'):
        stage_a._stage_a_source_identity(s['runner'], {'model': s['model'], cache: {}}, None)


def test_public_original_checkpoint_metadata_uses_actual_owned_bytes(material):
    """Internal CPU material is real leased metadata, not capture admission."""
    with _owner(material) as owner:
        descriptor = original_checkpoint_description(owner.root, owner)
        assert descriptor["config"] == json.loads(material["raws"]["config.json"])
        assert descriptor["index"] == json.loads(material["raws"]["model.safetensors.index.json"])
        assert {row["name"]: row["size"] for row in descriptor["shards"]} == {
            name: len(raw) for name, raw in material["raws"].items() if name.endswith(".safetensors")}
        identity = cs.build_source_checkpoint_identity(owner.root, source_authentication=owner)
        assert identity["schema"] == cs.SOURCE_CHECKPOINT_IDENTITY_SCHEMA
        assert owner.receipt()["automatic_capture_qualified"] is False
        with pytest.raises(RuntimeError, match="original identity source root differs from its owner"):
            original_checkpoint_description(Path(material["paths"]["config.json"]).parent, owner)


@pytest.mark.parametrize("owner", [None, SimpleNamespace(is_qualified_original_material=True)])
def test_public_original_checkpoint_metadata_keeps_exact_class_guard(tmp_path, owner):
    with pytest.raises(RuntimeError, match="original identity requires the qualified existing original owner"):
        original_checkpoint_description(tmp_path, owner)
    with pytest.raises(RuntimeError, match="original identity requires the qualified existing original owner"):
        cs.build_source_checkpoint_identity(tmp_path, source_authentication=(owner if owner is not None else object()))


def test_public_original_checkpoint_metadata_refuses_real_recording_owner(material):
    from prismaquant.tessera_calibration_cache import CaptureSourceAuthentication

    root = Path(material["paths"]["config.json"]).parent
    with CaptureSourceAuthentication.recording(root, {}) as owner:
        assert not owner.is_qualified_original_material
        with pytest.raises(RuntimeError, match="original identity requires the qualified existing original owner"):
            original_checkpoint_description(root, owner)
