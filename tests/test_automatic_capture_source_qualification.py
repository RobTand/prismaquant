"""Automatic recording has no qualified immutable provider (PQ #2013).

These are refusal/route controls, not positive immutable-source or capture
qualification. Existing explicit owner operations retain their weaker,
independently tested descriptor/producer/stat contract.
"""
from types import SimpleNamespace
import json

import pytest

from prismaquant import tessera_calibration_cache as cache
from prismaquant import tessera_campaign as campaign
from prismaquant import capture_layer_chain as chain


def test_no_source_claim_can_qualify_automatic_recording():
    with pytest.raises(RuntimeError, match='automatic capture.*immutable source'):
        cache.require_automatic_capture_source_recording()


@pytest.mark.parametrize('route', ['prep', 'join'])
def test_direct_bookends_refuse_before_census_tokenizer_recording_or_join(tmp_path, monkeypatch, route):
    attempted = []
    def forbidden(*args, **kwargs):
        attempted.append('original-source-or-publication')
        raise RuntimeError('forbidden source read reached before qualification')
    monkeypatch.setattr(campaign, 'load_calibration_census', forbidden)
    monkeypatch.setattr(campaign, '_calibration_tokens', forbidden)
    monkeypatch.setattr(cache, 'record_capture_source', forbidden)
    monkeypatch.setattr(chain, 'join', forbidden)
    args = SimpleNamespace(streaming=True, capture_calibration_out=str(tmp_path / 'capture'),
                           capture_chain=route)
    with pytest.raises(RuntimeError, match='automatic capture.*immutable source'):
        campaign._run_capture_chain_bookends(args, None)
    assert attempted == []
    assert not (tmp_path / 'capture').exists()


@pytest.mark.parametrize('route', ['monolith', 'prep', 'quantum', 'join'])
@pytest.mark.parametrize('policy', ['legacy', 'shared-inputs-bounded-v1'])
def test_cli_refuses_automatic_recording_before_model_reads(tmp_path, monkeypatch, route, policy):
    attempted = []
    def forbidden(*args, **kwargs):
        attempted.append('original-source-or-publication')
        raise RuntimeError('forbidden source read reached before qualification')
    from prismaquant import model_profiles, cost_streaming
    monkeypatch.setattr(model_profiles, 'detect_profile', forbidden)
    monkeypatch.setattr(cost_streaming, 'build_streamed_causal_lm', forbidden)
    monkeypatch.setattr(campaign, '_calibration_tokens', forbidden)
    monkeypatch.setattr(campaign, 'load_calibration_census', forbidden)
    monkeypatch.setattr(cache, 'record_capture_source', forbidden)
    monkeypatch.setattr(chain, 'authenticate_quantum_source', forbidden)
    monkeypatch.setattr(chain, 'join', forbidden)
    args = ['--model', str(tmp_path / 'missing-source'), '--out', str(tmp_path / 'cost.pkl'),
            '--cache-dir', str(tmp_path / 'cache'), '--streaming',
            '--capture-calibration-out', str(tmp_path / 'capture'),
            '--calibration-census', str(tmp_path / 'missing-census.json'),
            '--attention-implementation', 'eager', '--hessian', 'require',
            '--streaming-capture-policy', policy]
    if route != 'monolith':
        args += ['--capture-chain', route]
    if route == 'quantum':
        args += ['--capture-layer-range', '0:1']
    if route == 'prep':
        from prismaquant.cost_streaming import LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA
        boundary = {'schema': LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA,
                    'capture_order': 'layer_major', 'directory': str(tmp_path / 'boundaries'),
                    'max_resident_bytes': 1 << 20, 'max_auxiliary_bytes': 1 << 20,
                    'max_artifact_bytes': 1 << 24, 'prefetch_batches': 1}
        args += ['--capture-chain-ranges', '0:1',
                 '--capture-chain-boundary-storage', json.dumps(boundary)]
    with pytest.raises(RuntimeError, match='automatic capture.*immutable source'):
        campaign.main(args)
    assert attempted == []
    for output in ['cache', 'capture', 'boundaries', 'cost.pkl']:
        assert not (tmp_path / output).exists()


@pytest.mark.parametrize('streaming, output', [(False, 'capture'), (True, None), (False, None)])
def test_explicit_or_selected_paths_do_not_claim_automatic_qualification(monkeypatch, streaming, output):
    def forbidden():
        raise AssertionError('an explicit/selected path was turned into automatic recording')
    monkeypatch.setattr(cache, 'require_automatic_capture_source_recording', forbidden)
    campaign._require_automatic_capture_recording(
        SimpleNamespace(streaming=streaming, capture_calibration_out=output))
