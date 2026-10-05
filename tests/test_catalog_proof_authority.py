"""Bound proof bytes must actually authorize the exact historical encoder pair."""
import copy
import json
import pickle
from pathlib import Path

import pytest
from tests.test_joint_catalog_extension import _pair, _write
from tests.test_joint_quanta_join import campaign, probe
from prismaquant.joint_catalog_extension import R13_ADDED_FORMAT, verify_catalog_pair, validated_encoder_adoption


def _adoption(inputs):
    prepared = json.loads(Path(inputs['extended_prepared']['path']).read_text())
    cache = pickle.loads(Path(prepared['production_cache']['path']).read_bytes())
    return next(c['catalog_source_adoption'] for c in cache.metadata['verified_cells'].values()
                if 'catalog_source_adoption' in c)


@pytest.mark.parametrize('mutation', ['nonproof', 'wrong_pair', 'wrong_fixture', 'changed_arm'])
def test_digest_bound_but_unproven_encoder_adoption_refuses(tmp_path, campaign, probe, mutation):
    inputs, _, _ = _pair(tmp_path, campaign, probe)
    adoption = _adoption(inputs)
    validated_encoder_adoption(adoption, fmt=R13_ADDED_FORMAT)
    if mutation == 'nonproof':
        adoption['encoder_source_proof'] = _write(tmp_path, 'nonproof.json', {'accepted': True})
    elif mutation == 'wrong_pair':
        adoption['candidate_encoding_identity']['encoder_source_sha256'] = '9'*64
    elif mutation == 'wrong_fixture':
        adoption['candidate_encoding_identity']['encoder_fixture_id'] = 'e'*64
    else:
        (tmp_path/'source-proof-arm.json').write_text('{}')
    with pytest.raises((ValueError, RuntimeError), match='encoder|proof|fixture'):
        validated_encoder_adoption(adoption, fmt=R13_ADDED_FORMAT)


def test_cached_catalog_pair_rechecks_changed_proof_arm(tmp_path, campaign, probe):
    inputs, _, _ = _pair(tmp_path, campaign, probe)
    verify_catalog_pair(inputs)
    (tmp_path/'source-proof-arm.json').write_text('{}')
    with pytest.raises((ValueError, RuntimeError), match='encoder|proof|changed'):
        verify_catalog_pair(inputs)


def test_operation_reuses_proof_without_per_cell_filesystem_checks(tmp_path, campaign, probe, monkeypatch):
    import prismaquant.joint_catalog_extension as bridge
    inputs, _, _ = _pair(tmp_path, campaign, probe)
    adoption = _adoption(inputs)
    calls = []
    original = bridge._bound_stat_fence
    def fence(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(bridge, '_bound_stat_fence', fence)
    with bridge.EncoderAdoptionValidation() as operation:
        first = operation.verify(adoption, fmt=R13_ADDED_FORMAT)
        initial = len(calls)
        for _ in range(100):
            assert operation.verify(adoption, fmt=R13_ADDED_FORMAT) is first
        assert len(calls) == initial
    assert len(calls) > initial, 'completion must recheck dependency fences'
    with pytest.raises(ValueError, match='outside'):
        operation.verify(adoption, fmt=R13_ADDED_FORMAT)


def test_operation_refuses_dependency_change_before_return(tmp_path, campaign, probe):
    from prismaquant.joint_catalog_extension import EncoderAdoptionValidation
    inputs, _, _ = _pair(tmp_path, campaign, probe)
    with pytest.raises(ValueError, match='changed during operation'):
        with EncoderAdoptionValidation() as operation:
            operation.verify(_adoption(inputs), fmt=R13_ADDED_FORMAT)
            (tmp_path/'source-proof-arm.json').write_text('{}')


# The corrective admission tests bind an actual retained proof. They create no
# passing proof artifact and qualify no model bytes by themselves.
_GENUINE_PROOF = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/identity-reseal-20260915/rollout-inputs/proof-bundle-9753a5b7c5-c92826fa4.json')
_GENUINE_SHA256 = '15b373db8429240ef29c0641818d1f7e705d18154a8c2d841381a00213fd86c9'


def _genuine_adoption():
    import hashlib
    raw = _GENUINE_PROOF.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _GENUINE_SHA256
    document = json.loads(raw)
    fixture = next(iter(document['fixture_id']['ids'].values()))
    old = document['pins']['old']['encoder_source_sha256']
    new = document['pins']['new']['encoder_source_sha256']
    identity = {'unit': 'model.language_model.layers.15.mlp.experts.56.gate_proj',
                'encoder_fixture_id': fixture, 'encoder_source_sha256': old}
    return {'schema': 'prismaquant.joint_catalog_source_adoption.v1',
            'reference_encoding_identity': identity,
            'candidate_encoding_identity': {**identity, 'encoder_source_sha256': new},
            'encoder_source_proof': {'path': str(_GENUINE_PROOF), 'sha256': _GENUINE_SHA256}}


@pytest.mark.parametrize('dev', ['0', '1'])
@pytest.mark.parametrize('fmt', ['TESSERA_E4M3_K1_R880', 'TESSERA_E4M3_K1_R912'])
@pytest.mark.parametrize('failure', ['missing', 'wrong_old_pair', 'wrong_new_pair'])
def test_actual_t8_targets_require_bound_matching_proof_in_every_mode(monkeypatch, dev, fmt, failure):
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', dev)
    adoption = _genuine_adoption()
    if failure == 'missing':
        adoption['encoder_source_proof'] = None
    elif failure == 'wrong_old_pair':
        adoption['reference_encoding_identity']['encoder_source_sha256'] = adoption['candidate_encoding_identity']['encoder_source_sha256']
    else:
        # This is the actual same-encoder candidate's pair, not a proof for
        # the retained 0833671b -> a4c92094 transition.
        adoption['candidate_encoding_identity']['encoder_source_sha256'] = adoption['reference_encoding_identity']['encoder_source_sha256']
    with pytest.raises(ValueError, match='encoder.*proof|proof.*source'):
        validated_encoder_adoption(adoption, fmt=fmt)


@pytest.mark.parametrize('dev', ['0', '1'])
def test_a_genuine_proof_never_licenses_an_uncovered_family(monkeypatch, dev):
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', dev)
    adoption = _genuine_adoption()
    document = json.loads(_GENUINE_PROOF.read_text())
    uncovered_fmt = 'TESSERA_E4M3_K2_R880'
    assert 'TESSERA_E4M3_K2' not in document['strata']['routed']
    with pytest.raises(ValueError, match='does not cover the added candidate stratum'):
        validated_encoder_adoption(adoption, fmt=uncovered_fmt)


@pytest.mark.parametrize('dev', ['0', '1'])
@pytest.mark.parametrize('fmt', ['TESSERA_E4M3_K1_R880', 'TESSERA_E4M3_K1_R912'])
def test_actual_genuine_proof_covers_only_its_real_t8_source_pair(monkeypatch, dev, fmt):
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', dev)
    result = validated_encoder_adoption(_genuine_adoption(), fmt=fmt)
    assert result['encoder_source_proof_covered'] is True
    assert result['stratum'] == ['routed', 'TESSERA_E4M3_K1']
