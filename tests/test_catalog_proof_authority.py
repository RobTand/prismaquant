"""Bound proof bytes must actually authorize the exact historical encoder pair."""
import copy
import json
import pickle
from pathlib import Path

import pytest
from tests.test_joint_catalog_extension import _pair, _write
from tests.test_joint_quanta_join import campaign, probe
from prismaquant.joint_catalog_extension import verify_catalog_pair, validated_encoder_adoption


def _adoption(inputs):
    prepared = json.loads(Path(inputs['extended_prepared']['path']).read_text())
    cache = pickle.loads(Path(prepared['production_cache']['path']).read_bytes())
    return next(c['catalog_source_adoption'] for c in cache.metadata['verified_cells'].values()
                if 'catalog_source_adoption' in c)


@pytest.mark.parametrize('mutation', ['nonproof', 'wrong_pair', 'wrong_fixture', 'changed_arm'])
def test_digest_bound_but_unproven_encoder_adoption_refuses(tmp_path, campaign, probe, mutation):
    inputs, _, _ = _pair(tmp_path, campaign, probe)
    adoption = _adoption(inputs)
    validated_encoder_adoption(adoption)
    if mutation == 'nonproof':
        adoption['encoder_source_proof'] = _write(tmp_path, 'nonproof.json', {'accepted': True})
    elif mutation == 'wrong_pair':
        adoption['candidate_encoding_identity']['encoder_source_sha256'] = '9'*64
    elif mutation == 'wrong_fixture':
        adoption['candidate_encoding_identity']['encoder_fixture_id'] = 'e'*64
    else:
        (tmp_path/'source-proof-arm.json').write_text('{}')
    with pytest.raises((ValueError, RuntimeError), match='encoder|proof|fixture'):
        validated_encoder_adoption(adoption)


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
        first = operation.verify(adoption)
        initial = len(calls)
        for _ in range(100):
            assert operation.verify(adoption) is first
        assert len(calls) == initial
    assert len(calls) > initial, 'completion must recheck dependency fences'
    with pytest.raises(ValueError, match='outside'):
        operation.verify(adoption)


def test_operation_refuses_dependency_change_before_return(tmp_path, campaign, probe):
    from prismaquant.joint_catalog_extension import EncoderAdoptionValidation
    inputs, _, _ = _pair(tmp_path, campaign, probe)
    with pytest.raises(ValueError, match='changed during operation'):
        with EncoderAdoptionValidation() as operation:
            operation.verify(_adoption(inputs))
            (tmp_path/'source-proof-arm.json').write_text('{}')
