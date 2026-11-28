"""PQ #2010: source verification must precede reusable capture completion."""
import json
import os
from pathlib import Path

import pytest
import torch

from prismaquant import capture_layer_chain as chain
from prismaquant import tessera_calibration_cache as cc
from test_capture_layer_chain import _Chain
from test_capture_single_pass_source_1896 import (
    _glm_conv_kernel_is_cuda_only, _glm_source, _mutate_preserving_mtime,
)


def _freeze_signature(monkeypatch, path, state):
    """Hide only this file's metadata drift, not its real bytes or reads."""
    path = Path(path)
    named, held = path.stat(), os.fstat(state['fd'])
    identity = held.st_dev, held.st_ino
    real_stat, real_fstat = os.stat, os.fstat

    def stat(value, *args, **kwargs):
        if not isinstance(value, int) and Path(value) == path:
            return named
        return real_stat(value, *args, **kwargs)

    def fstat(fd):
        live = real_fstat(fd)
        return held if (live.st_dev, live.st_ino) == identity else live

    monkeypatch.setattr(os, 'stat', stat)
    monkeypatch.setattr(os, 'fstat', fstat)


@pytest.mark.parametrize('mutate', (False, True))
def test_quantum_verifies_before_fragment_and_owner_completion(tmp_path, monkeypatch, mutate):
    fixture = _Chain(tmp_path)
    shard = fixture.source / 'model.safetensors'
    complete = chain.ChainQuantum.complete
    snapshots = []

    def authenticate_then_complete(quantum, **kwargs):
        owner = quantum.source_authentication
        name, state = owner._file(shard)
        owner._authenticate(name, state)
        snapshots.append(owner.receipt())
        if mutate and quantum.start == 0:
            _freeze_signature(monkeypatch, shard, state)
            _mutate_preserving_mtime(shard)
            # The old metadata gate admits: this must fail on real content.
            owner.require_unchanged()
        result = complete(quantum, **kwargs)
        assert owner._closed  # Fragment publication follows successful close.
        return result

    monkeypatch.setattr(chain.ChainQuantum, 'complete', authenticate_then_complete)
    if mutate:
        with pytest.raises(RuntimeError, match='authenticated source changed during consumption'):
            fixture.quantum((0, 1))
        assert not chain.fragment_path(fixture.root, 0, 1).exists()
        with pytest.raises(chain.CaptureChainRefused, match='not a complete owner'):
            chain.require_owner_complete(fixture.prep, 0, 1)
        with pytest.raises(chain.CaptureChainRefused, match='not a complete owner'):
            fixture.quantum((1, 2))
        with pytest.raises(chain.CaptureChainRefused, match='not a complete owner'):
            chain.join(fixture.root, census_path=fixture.census)
        assert not (fixture.root / 'capture_manifest.json').exists()
        assert not chain.join_path(fixture.root).exists()
    else:
        for layers in ((0, 1), (1, 2)):
            fixture.quantum(layers)
            chain.require_owner_complete(fixture.prep, *layers)
            fragment = chain.read_fragment(fixture.root, fixture.prep, *layers)
            assert fragment['source_authentication'] == snapshots[-1]
        document = chain.join(fixture.root, census_path=fixture.census)
        manifest = json.loads(Path(document['manifest']['path']).read_text())
        assert manifest['identity']['source_files'][shard.name] == cc.sha256(shard)


@pytest.mark.parametrize('policy', ('legacy', 'shared-inputs-bounded-v1'))
def test_monolith_refuses_same_signature_mutation_before_manifest(tmp_path, monkeypatch, policy):
    """Exercise the actual recording traversal, not automatic-provider admission."""
    if _glm_conv_kernel_is_cuda_only():
        pytest.skip('CPU GLM fixture needs the torch convolution fallback')
    from prismaquant import tessera_campaign as campaign
    monkeypatch.setattr(cc, 'require_automatic_capture_source_recording', lambda: None)
    source, _ = _glm_source(tmp_path, monkeypatch, device='cpu')
    tokens = [torch.arange(257).remainder(126).add(2).reshape(1, -1)]
    monkeypatch.setattr(campaign, '_calibration_tokens', lambda *_: (tokens, 'frozen draw'))
    census = tmp_path / 'census.json'
    common = ['--model', str(source), '--out', str(tmp_path / 'unused.pkl'),
        '--menu-mode', 'research', '--nsamples', '1', '--seqlen', '257',
        '--max-act-rows', '7', '--attention-implementation', 'eager', '--streaming',
        '--streaming-cache-headroom-gb', '0']
    assert campaign.main([*common, '--cache-dir', str(tmp_path / 'census-cache'),
                          '--census-out', str(census)]) == 0
    authenticate = cc.CaptureSourceAuthentication.authenticate_complete_source
    finish = cc.CaptureWriter.finish
    owners, published = [], []

    def mutate_after_authentication(owner):
        receipt = authenticate(owner)
        shard = source / 'model-layer-000.safetensors'
        _freeze_signature(monkeypatch, shard, owner._files[shard.name])
        _mutate_preserving_mtime(shard)
        owner.require_unchanged()
        owners.append(owner)
        return receipt

    def publish(writer, **kwargs):
        published.append(writer)
        return finish(writer, **kwargs)

    monkeypatch.setattr(cc.CaptureSourceAuthentication, 'authenticate_complete_source',
                        mutate_after_authentication)
    monkeypatch.setattr(cc.CaptureWriter, 'finish', publish)
    out, cache = tmp_path / 'capture', tmp_path / 'capture-cache'
    argv = [*common, '--calibration-census', str(census), '--cache-dir', str(cache),
        '--capture-calibration-out', str(out), '--streaming-capture-policy', policy]
    with pytest.raises(RuntimeError, match='authenticated source changed during consumption'):
        campaign.main(argv)
    assert len(owners) == 1 and owners[0]._closed
    assert not published
    assert not (out / 'capture_manifest.json').exists()
    assert not (cache / 'capture-source-authentication.json').exists()
