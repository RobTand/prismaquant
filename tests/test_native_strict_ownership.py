"""Retained namespace and acquire/open failure obligations without PB owners."""
import errno
import json
import os
from pathlib import Path

import pytest
import torch

from prismaquant import layer_streaming, residency_map, residency_shard_reader
from prismaquant.staged_lease import LeaseWindow, acquire_entry_window
from prismaquant.staged_manifest import ManifestRefused, load_staged_manifest, publish_staged_json
from prismaquant.staged_tier_policy import TierPolicyRefused, staged_tier_policy_test_context
from staged_native_fixtures import bind, native_staged_state, publish_inputs, sha
from test_native_strict_reader_contract import _forbid_source, _shard

pytestmark = pytest.mark.usefixtures('native_staged_state')


def test_retained_phase_cannot_adopt_an_object_outside_its_namespace(tmp_path, monkeypatch):
    source = tmp_path / 'declared'
    raw = b'known-owned-namespace-content'
    source.write_bytes(raw)
    binding = publish_inputs(tmp_path, [source])
    original = load_staged_manifest(binding['path'], binding['sha256'])
    assert Path(original.extents[0].object_path).read_bytes() == raw
    outside = tmp_path / 'outside-retained-phase'
    outside.write_bytes(raw)
    document = json.loads(Path(binding['path']).read_bytes())
    document['entries'][0]['object']['path'] = str(outside)
    replacement = tmp_path / 'outside-manifest.json'
    digest = publish_staged_json(replacement, document)
    opened = _forbid_source(monkeypatch, outside)
    with pytest.raises(ManifestRefused, match='outside retained phase'):
        load_staged_manifest(replacement, digest)
    assert opened == []
    assert sha(raw) == original.extents[0].sha256


@pytest.mark.parametrize('fault', ['missing', 'permission'])
def test_open_failure_after_acquisition_releases_only_its_owned_window(tmp_path, monkeypatch, fault):
    sources = [tmp_path / name for name in ('victim', 'retained-reader')]
    payloads = [b'victim-known-bytes', b'independent-surviving-reader-bytes']
    for source, raw in zip(sources, payloads):
        source.write_bytes(raw)
    resolver = bind(monkeypatch, publish_inputs(tmp_path, sources))
    opened = [_forbid_source(monkeypatch, source) for source in sources]
    for source in sources:
        source.unlink()
    entries = [resolver.staged_read(source) for source in sources]
    victim = Path(entries[0]['stage_path'])
    with staged_tier_policy_test_context('ssd'):
        stable, stable_key = acquire_entry_window(resolver, sources[1], entries[1])
        failed, failed_key = acquire_entry_window(resolver, sources[0], entries[0])
        with stable:
            fd, _ = stable.open(stable_key)
            assert os.pread(fd, len(payloads[1]), 0) == payloads[1]
            with failed:
                assert resolver._windows == 2
                if fault == 'missing':
                    victim.unlink()
                else:
                    original_open = residency_map.open_regular

                    def denied(filename):
                        if Path(filename) == victim:
                            raise PermissionError(errno.EACCES, 'owned object open refused', str(filename))
                        return original_open(filename)

                    monkeypatch.setattr(residency_map, 'open_regular', denied)
                with pytest.raises(FileNotFoundError if fault == 'missing' else PermissionError):
                    failed.open(failed_key)
                assert failed._fds == {}
            assert resolver._windows == 1
            assert os.pread(fd, len(payloads[1]), 0) == payloads[1]
        assert resolver._windows == 0
        with pytest.raises(OSError):
            os.fstat(fd)
    assert resolver.report()['bytes_from_pool'] == 0
    assert all(not calls for calls in opened)
    resolver.close()
    assert resolver._fds == {}


@pytest.mark.parametrize('fault', ['length', 'corrupt', 'missing'])
def test_changed_shard_refuses_without_landing_wait_or_partial_tensor(tmp_path, monkeypatch, fault):
    source, expected = _shard(tmp_path)
    resolver = bind(monkeypatch, publish_inputs(tmp_path, [source]))
    opened = _forbid_source(monkeypatch, source)
    obj = Path(resolver.manifest.extents[0].object_path)
    source.unlink()
    exposed = []
    with staged_tier_policy_test_context('ssd'):
        with layer_streaming._source_safe_open(str(source), framework='pt') as reader:
            known = reader.get_tensor('f32')
            assert torch.equal(known.view(torch.uint8), expected['f32'].view(torch.uint8))
        assert resolver._windows == 0
        before = resolver.report()['bytes_from_stage']
        raw = obj.read_bytes()
        if fault == 'missing':
            obj.unlink()
        else:
            obj.chmod(0o600)
            obj.write_bytes(raw[:-1] if fault == 'length' else raw[:-1] + bytes([raw[-1] ^ 1]))

        def forbidden_wait(*args, **kwargs):
            raise AssertionError('an integrity failure entered dynamic landing')

        monkeypatch.setattr(residency_shard_reader, 'await_staged_spans', forbidden_wait)
        with pytest.raises((TierPolicyRefused, OSError, RuntimeError, ValueError)):
            with layer_streaming._source_safe_open(str(source), framework='pt') as reader:
                exposed.append(reader.get_tensor('f32'))
        assert exposed == [] and resolver._windows == 0
        assert resolver.report()['bytes_from_stage'] == before
        assert torch.equal(known.view(torch.uint8), expected['f32'].view(torch.uint8))
    assert opened == [] and resolver.report()['bytes_from_pool'] == 0


def test_failed_shard_setup_closes_its_opened_descriptor_and_preserves_the_error(tmp_path, monkeypatch):
    source, expected = _shard(tmp_path)
    resolver = bind(monkeypatch, publish_inputs(tmp_path, [source]))
    opened = _forbid_source(monkeypatch, source)
    source.unlink()
    windows, descriptors = [], []
    original = LeaseWindow.open

    def observe(window, key):
        fd, serving = original(window, key)
        windows.append(window)
        descriptors.append(fd)
        return fd, serving

    failure = RuntimeError('owned shard setup failed after descriptor acquisition')

    def refuse_shape(path):
        assert resolver._windows == 1
        assert os.fstat(descriptors[-1]).st_size > 0
        raise failure

    with staged_tier_policy_test_context('ssd'):
        with monkeypatch.context() as failed_setup:
            failed_setup.setattr(LeaseWindow, 'open', observe)
            failed_setup.setattr(residency_shard_reader, '_read_shape', refuse_shape)
            with pytest.raises(RuntimeError) as raised:
                with layer_streaming._source_safe_open(str(source), framework='pt') as reader:
                    reader.get_tensor('f32')
            assert raised.value is failure
        assert len(windows) == len(descriptors) == 1
        assert windows[0]._fds == {} and resolver._windows == 0
        with pytest.raises(OSError):
            os.fstat(descriptors[0])
        with layer_streaming._source_safe_open(str(source), framework='pt') as reader:
            assert torch.equal(reader.get_tensor('f32').view(torch.uint8), expected['f32'].view(torch.uint8))
    assert resolver._windows == 0 and resolver.report()['bytes_from_pool'] == 0
    assert opened == []
