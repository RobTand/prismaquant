"""PQ #1497: a Stage A row adopts the campaign's source proof instead of hashing.

A selected-source row reads a few tensors from a few shards. Before this
change it hashed each of those shards whole, on its GPU reservation, before
its first encode. The row now adopts the campaign's streamed identity proof
through ``CaptureSourceAuthentication.adopt_streamed_identity_cache``, the
same check the joint pass uses (#1374), and a proof that refuses leaves every
read to hash fresh.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_selected_source_authentication import selected_source_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))


def _write_proof(source, out):
    """A real ``streamed_model.identity_cache.v1`` proof of ``source``.

    Built by the production builder, so the row validates it exactly as it
    would validate the campaign's proof. The fixture's shards are hashed here,
    before the row starts counting.
    """
    from prismaquant.cost_streaming import build_streamed_model_identity

    weight_map = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    runner = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {})),
        context=SimpleNamespace(weight_ckpt={}, weight_shard={
            name: str(source / shard) for name, shard in weight_map.items()}))
    build_streamed_model_identity(runner, str(source), identity_cache_path=out)
    return out, hashlib.sha256(out.read_bytes()).hexdigest()


def _proof_argv(argv, proof, digest):
    return [*argv, '--source-identity-cache', str(proof),
            '--source-identity-cache-sha256', digest]


def test_a_row_with_an_adoptable_proof_hashes_no_payload_shard(monkeypatch, tmp_path, capsys):
    """The regression: zero full-file payload hashes, and the same weights."""
    campaign, argv, state = selected_source_fixture(monkeypatch, tmp_path)
    proof, digest = _write_proof(state['source'], tmp_path / 'source-identity.json')
    state['hashed'].clear()
    assert campaign.main(_proof_argv(argv, proof, digest)) == campaign.EXIT_EMPTY_MENU
    assert state['hashed'] == []
    assert state['copied'] and torch.equal(state['copied'][0], state['selected'])
    assert '[source] adopted 3 full-file SHA proofs' in capsys.readouterr().out


def _restamp(path):
    """Same bytes, a new ctime: the proof no longer names this object."""
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))


def _mutate_restoring_mtime(path):
    before = path.stat()
    with path.open('r+b') as handle:
        handle.seek(-1, 2)
        old = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([old[0] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


@pytest.mark.parametrize('predicate', ['production', 'admit-anything'])
def test_a_shard_restamped_after_the_proof_refuses_adoption_and_hashes_fresh(
        monkeypatch, tmp_path, capsys, predicate):
    """The mutation test. A shard's stat fingerprint changes after the proof
    is written; the row refuses the whole proof and hashes what it reads.

    ``admit-anything`` mutates the driver, not the fixture: with the
    production reuse predicate replaced by one that admits any fingerprint,
    the same input adopts and hashes nothing. That is what shows the refusal
    comes from the fingerprint check and not from something else in the row.
    """
    from prismaquant import cost_streaming

    campaign, argv, state = selected_source_fixture(monkeypatch, tmp_path)
    proof, digest = _write_proof(state['source'], tmp_path / 'source-identity.json')
    _restamp(state['source'] / 'selected.safetensors')
    if predicate == 'admit-anything':
        monkeypatch.setattr(cost_streaming, 'stat_fingerprint_reusable', lambda *_a: True)
    state['hashed'].clear()
    assert campaign.main(_proof_argv(argv, proof, digest)) == campaign.EXIT_EMPTY_MENU
    assert torch.equal(state['copied'][0], state['selected'])
    out = capsys.readouterr().out
    if predicate == 'production':
        assert 'names another object' in out and 'hashes its shard fresh' in out
        assert sorted(name for name, _ in state['hashed']) == [
            'head.safetensors', 'selected.safetensors']
    else:
        assert state['hashed'] == []


def test_a_proof_file_that_differs_from_its_declared_digest_is_not_adopted(
        monkeypatch, tmp_path, capsys):
    campaign, argv, state = selected_source_fixture(monkeypatch, tmp_path)
    proof, _ = _write_proof(state['source'], tmp_path / 'source-identity.json')
    state['hashed'].clear()
    assert campaign.main(_proof_argv(argv, proof, '0' * 64)) == campaign.EXIT_EMPTY_MENU
    assert 'differs from its declared SHA256' in capsys.readouterr().out
    assert sorted(name for name, _ in state['hashed']) == [
        'head.safetensors', 'selected.safetensors']


def test_a_content_edit_behind_a_restored_mtime_still_refuses_the_row(monkeypatch, tmp_path):
    """Integrity is kept: the refused proof falls back to a fresh hash, and the
    fresh hash does not match the canonical capture."""
    campaign, argv, state = selected_source_fixture(monkeypatch, tmp_path)
    proof, digest = _write_proof(state['source'], tmp_path / 'source-identity.json')
    _mutate_restoring_mtime(state['source'] / 'selected.safetensors')
    with pytest.raises(RuntimeError, match='content differs from sealed capture'):
        campaign.main(_proof_argv(argv, proof, digest))


def test_the_proof_flags_go_together_and_only_on_a_selected_row(monkeypatch, tmp_path, capsys):
    campaign, argv, state = selected_source_fixture(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        campaign.main([*argv, '--source-identity-cache', str(tmp_path / 'p.json')])
    assert 'go together' in capsys.readouterr().err
    plain = list(argv)
    at = plain.index('--calibration-cache')
    assert plain[at + 2] == '--calibration-cache-sha256'
    del plain[at:at + 4]
    with pytest.raises(SystemExit):
        campaign.main(_proof_argv(plain, tmp_path / 'p.json', '0' * 64))
    assert 'requires selected streaming capture reuse' in capsys.readouterr().err


def test_the_planner_binds_the_proof_by_digest_and_owns_the_flags(tmp_path):
    import dispatch_tessera_campaign as dispatch
    model = tmp_path / 'model'
    model.mkdir()
    from safetensors.torch import save_file
    save_file({'w': torch.ones(2, 2)}, str(model / 'a.safetensors'))
    (model / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {'w': 'a.safetensors'}}))
    proof, digest = _write_proof(model, tmp_path / 'source-identity.json')
    assert dispatch._source_identity_cache_binding(proof, model) == dict(
        path=str(proof.resolve()), sha256=digest)
    assert dispatch._source_identity_cache_binding(None, model) is None
    with pytest.raises(RuntimeError, match='does not bind source'):
        dispatch._source_identity_cache_binding(proof, tmp_path / 'another-model')
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'model': str(model), 'cwd': str(tmp_path), 'python': 'python3',
        'env': {}, 'campaign_argv': ['--source-identity-cache', str(proof)]}))
    with pytest.raises(RuntimeError, match='owns per row'):
        dispatch.load_spec(spec)
