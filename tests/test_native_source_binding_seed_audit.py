"""Current-byte seed and journal/harvest boundaries, without AURA measurement.

Real checkpoint files, streamed identity, decoder, binding, journal and harvest
orchestration run here. Harvest doubles only durability-location admission,
CUDA admission, tokenizer/calibration intake and the numerical AURA call. Its
AURA double uses the real journal before recording any synthetic measurement.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pickle
import re

import pytest
import torch

from prismaquant import aura_checkpoint_journal as journal
from prismaquant import streamed_model_identity as identity
from prismaquant.cost_streaming import build_streamed_causal_lm
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.model_profiles import detect_profile
from tests.test_resident_source_decode_audit import _source


def _proof(source, tmp_path):
    runner = build_streamed_causal_lm(
        str(source), device=torch.device('cpu'), dtype=torch.bfloat16,
        offload_folder=str(tmp_path / 'identity-offload'),
        profile=detect_profile(str(source)),
        max_cache_slots=2, prefetch_workers=1, prefetch_min_available_gb=0,
        cache_headroom_gb=0, prefetch_lookahead=1,
        require_prefetched_residency=True, attn_implementation='eager')
    path = tmp_path / 'streamed-identity.json'
    try:
        result = identity.build_streamed_model_identity(
            runner, str(source), identity_cache_path=path)
        assert result['checkpoint_weight_map']
        return path
    finally:
        runner.shutdown()


def _reseal(record):
    value = record['identity']
    value['content_sha256'] = canonical_json_sha256(
        {key: value[key] for key in ('config', 'weight_map', 'shards', 'checkpoint_weight_map')},
        where='deliberately inconsistent seed fixture')


@pytest.mark.parametrize('mode', ['1', '0'], ids=['dev', 'certified'])
@pytest.mark.parametrize('change', [
    'none', 'stat-drift', 'missing-fingerprint', 'duplicate-fingerprint',
    'wrong-size', 'bool-size', 'wrong-map', 'foreign-source',
])
def test_strict_streamed_seed_requires_current_complete_correspondence(
        tmp_path, monkeypatch, mode, change):
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    source, _expected = _source(tmp_path)
    path = _proof(source, tmp_path)
    expected = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    record = json.loads(path.read_text())
    if change == 'stat-drift':
        shard = source / 'model.safetensors'
        raw = bytearray(shard.read_bytes())
        raw[-1] ^= 1
        shard.write_bytes(raw)
    elif change == 'missing-fingerprint':
        record['fingerprints'].pop()
    elif change == 'duplicate-fingerprint':
        record['fingerprints'].append(deepcopy(record['fingerprints'][0]))
    elif change == 'wrong-size':
        record['identity']['shards'][0]['size'] += 1
        _reseal(record)
    elif change == 'bool-size':
        record['identity']['shards'][0]['size'] = True
        _reseal(record)
    elif change == 'wrong-map':
        record['identity']['checkpoint_weight_map'].pop(next(iter(record['identity']['checkpoint_weight_map'])))
        _reseal(record)
    elif change == 'foreign-source':
        record['source'] = str(tmp_path / 'foreign')
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    bulk_reads = []
    actual_hash = identity._hash_source_shards

    def no_bulk_hash(work):
        bulk_reads.extend(work)
        assert not work, 'explicit seed must never fall back to a payload rehash'
        return actual_hash(work)

    monkeypatch.setattr(identity, '_hash_source_shards', no_bulk_hash)
    if change == 'none':
        assert identity.build_source_model_binding(
            source, dtype=torch.bfloat16, streamed_identity_cache_path=path) == expected
    else:
        with pytest.raises(RuntimeError, match='(?i)(source|proof|cache)'):
            identity.build_source_model_binding(
                source, dtype=torch.bfloat16, streamed_identity_cache_path=path)
    assert bulk_reads == []
    assert path.read_bytes() == before


def test_seed_reuses_shards_but_binds_current_raw_metadata(tmp_path, monkeypatch):
    source, _expected = _source(tmp_path)
    path = _proof(source, tmp_path)
    old = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    config = source / 'config.json'
    config.write_text(config.read_text() + '\n')
    (source / 'checkpoint_policy.py').write_text('SOURCE_POLICY = 2\n')
    actual = identity._hash_source_shards

    def no_bulk_hash(work):
        assert not work
        return actual(work)

    monkeypatch.setattr(identity, '_hash_source_shards', no_bulk_hash)
    current = identity.build_source_model_binding(
        source, dtype=torch.bfloat16, streamed_identity_cache_path=path)
    assert current['checkpoint']['shards'] == old['checkpoint']['shards']
    assert current['checkpoint']['metadata'] != old['checkpoint']['metadata']
    assert identity.validate_source_model_binding(current) == current
    with pytest.raises(ValueError, match='one digest cache'):
        identity.build_source_model_binding(
            source, dtype=torch.bfloat16, streamed_identity_cache_path=path,
            digest_cache_path=tmp_path / 'other.json')


@pytest.mark.parametrize('mode', ['1', '0'], ids=['dev', 'certified'])
@pytest.mark.parametrize('stored', ['same', 'missing', 'legacy', 'foreign', 'other-dtype'])
def test_native_journal_admission_precedes_any_unit_read_or_restore(
        tmp_path, monkeypatch, mode, stored):
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    source, _expected = _source(tmp_path)
    binding = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    retained = deepcopy(binding)
    if stored == 'legacy':
        retained.pop('source_decode')
        retained['schema'] = identity.SOURCE_MODEL_BINDING_SCHEMA
    elif stored == 'foreign':
        (source / 'checkpoint_policy.py').write_text('DIFFERENT_SOURCE = True\n')
        retained = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    elif stored == 'other-dtype':
        retained['effective_dtype'] = 'torch.float32'
    old = {'schema': journal.AURA_CHECKPOINT_IDENTITY_SCHEMA,
           'extra': {} if stored == 'missing' else {'source_binding': retained}}
    expected = {**old, 'extra': {'source_binding': binding}}
    root = tmp_path / 'journal'
    first = journal.AuraCheckpointJournal.open(root, resume=False, identity=old, names=['unit'])
    first.publish_state('unit', {'scalar': [1.25]})
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    reads, restored = [], {}
    actual = journal.load_unit

    def observed_load(*args, **kwargs):
        reads.append(kwargs['qname'])
        return actual(*args, **kwargs)

    class Prices:
        def restore_unit(self, name, state):
            restored[name] = state

    monkeypatch.setattr(journal, 'load_unit', observed_load)
    if stored == 'same':
        resumed = journal.AuraCheckpointJournal.open(root, resume=True, identity=expected, names=['unit'])
        assert resumed.resume_units(Prices()) == {'unit'}
        assert reads == ['unit'] and restored == {'unit': {'scalar': [1.25]}}
    else:
        with pytest.raises(RuntimeError, match='native source binding.*fresh checkpoint'):
            journal.AuraCheckpointJournal.open(root, resume=True, identity=expected, names=['unit'])
        assert reads == [] and restored == {}
    assert {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('native', [True, False], ids=['native-binding', 'bf16-unchanged'])
@pytest.mark.parametrize('legacy_resume', [False, True], ids=['fresh', 'old-journal'])
def test_real_harvest_binds_native_before_numerical_measurement(
        tmp_path, monkeypatch, native, legacy_resume):
    from prismaquant import aura_cost, calibration_data, glm53_stock_harvest as harvest, gpu_guard
    from prismaquant.perturbed_x_cache import calibration_data_hash
    from tests.native_source_contract_fixtures import _source as glm_source
    from transformers import AutoTokenizer

    source, probe, census = glm_source(tmp_path, native=native)
    calibration = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    probe['meta']['calib_hash'] = calibration_data_hash(calibration)
    probe_path, census_path = tmp_path / 'probe.json', tmp_path / 'census.json'
    probe_path.write_text(json.dumps(probe))
    census_path.write_text(json.dumps(census))
    plan = harvest.build_dense_plan(probe, census)['plan']
    activations = tmp_path / 'activations'
    activations.mkdir()
    for name in plan:
        torch.save(torch.ones(4, 128), activations / (re.sub(r'[^A-Za-z0-9_-]', '__', name) + '.pt'))
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    if legacy_resume:
        previous = journal.AuraCheckpointJournal.open(
            checkpoint / 'aura', resume=False, identity={'extra': {}}, names=sorted(plan))
        for name in plan:
            previous.publish_state(name, {'scalar': [1.25]})
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '1')
    monkeypatch.setattr(harvest, 'refuse_volatile_path', lambda _label, path, _repo: Path(path))
    monkeypatch.setattr(gpu_guard, 'require_cuda_hot_path', lambda *_args: torch.device('cpu'))
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *_args, **_kwargs: object())
    monkeypatch.setattr(calibration_data, 'load_calibration', lambda *_args, **_kwargs: calibration)
    calls = []

    def measurement(runner, tokens, **kwargs):
        assert torch.equal(tokens, calibration)
        assert runner.context is not None
        extra = kwargs['checkpoint_identity_extra']
        if native:
            binding = identity.validate_source_model_binding(extra['source_binding'])
            assert binding['schema'] == identity.NATIVE_SOURCE_MODEL_BINDING_SCHEMA
            assert binding == identity.build_source_model_binding(source, dtype=torch.bfloat16)
        else:
            assert 'source_binding' not in extra
        actual = journal.AuraCheckpointJournal.open(
            kwargs['checkpoint_dir'], resume=kwargs['resume'],
            identity={'extra': extra}, names=sorted(plan))
        calls.append(actual.identity_sha256)
        return {'costs': {}, 'provenance': {'dw_production_anchor_rows': len(plan), 'dw_rtn_fallback_rows': 0}}

    monkeypatch.setattr(aura_cost, 'run_streamed_production_anchor_aura', measurement)
    output = tmp_path / 'harvest.pkl'
    arguments = ['--model', str(source), '--probe-census', str(probe_path),
                 '--checkpoint-census', str(census_path), '--activation-cache-dir', str(activations),
                 '--dataset', 'external-calibration-double', '--n-calib-samples', '1',
                 '--calib-seqlen', '4', '--checkpoint-dir', str(checkpoint), '--output', str(output)]
    if legacy_resume:
        arguments.append('--resume')
    if native and legacy_resume:
        with pytest.raises(RuntimeError, match='native source binding.*fresh checkpoint'):
            harvest.main(arguments)
        assert calls == [] and not output.exists()
    else:
        assert harvest.main(arguments) == 0
        wrapper = pickle.loads(output.read_bytes())
        assert len(calls) == 1
        identity.validate_streamed_model_identity(wrapper['model_identity'], where='real harvest test')
        if native:
            assert wrapper['source_binding'] == identity.build_source_model_binding(source, dtype=torch.bfloat16)
        else:
            assert 'source_binding' not in wrapper


@pytest.mark.parametrize('mode', ['1', '0'], ids=['dev', 'certified'])
def test_strict_seed_preserves_existing_nfs_device_only_reuse(tmp_path, monkeypatch, mode):
    """A mount-classification double, not a cross-host/NFS execution claim."""
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    source, _expected = _source(tmp_path)
    path = _proof(source, tmp_path)
    expected = identity.build_source_model_binding(source, dtype=torch.bfloat16)
    record = json.loads(path.read_text())
    for fingerprint in record['fingerprints']:
        fingerprint['device'] += 1
    path.write_text(json.dumps(record))
    monkeypatch.setattr(identity, 'device_number_is_client_local', lambda _path: True)
    actual = identity._hash_source_shards

    def no_bulk_hash(work):
        assert not work
        return actual(work)

    monkeypatch.setattr(identity, '_hash_source_shards', no_bulk_hash)
    assert identity.build_source_model_binding(
        source, dtype=torch.bfloat16, streamed_identity_cache_path=path) == expected
