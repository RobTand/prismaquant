"""Scientific strict-reader obligations over actual immutable staged inputs."""
import gzip
import json
import os
from pathlib import Path
import threading

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import layer_streaming, residency_map, staged_launch
from prismaquant.residency_shard_reader import await_staged_spans
from prismaquant.staged_lease import acquire_entry_window
from prismaquant.staged_manifest import load_staged_manifest
from prismaquant.staged_tier_policy import staged_tier_policy_test_context, TierPolicyRefused
from prismaquant.staged_whole_file import read_staged_whole_file
from staged_native_fixtures import bind, native_staged_state, publish_inputs, sha


pytestmark = pytest.mark.usefixtures('native_staged_state')


def _shard(tmp_path):
    tensors = {'f32': torch.linspace(-3, 3, 256, dtype=torch.float32).reshape(8, 32),
               'bf16': torch.arange(512).reshape(16, 32).to(torch.bfloat16),
               'empty': torch.zeros((0, 4), dtype=torch.float32)}
    source = tmp_path / 'weights.safetensors'
    save_file(tensors, source, metadata={'fixture': 'strict-known-values'})
    return source, tensors


def _forbid_source(monkeypatch, source):
    """Prove the source path is never opened, including low-level syscalls."""
    real_open, real_path_open = os.open, Path.open
    seen = []

    def native(filename, *args, **kwargs):
        if not isinstance(filename, int) and os.fspath(filename) == str(source):
            seen.append('os.open')
            raise AssertionError('strict reader opened canonical source')
        return real_open(filename, *args, **kwargs)

    def path_open(filename, *args, **kwargs):
        if filename == source:
            seen.append('Path.open')
            raise AssertionError('strict reader opened canonical source')
        return real_path_open(filename, *args, **kwargs)

    monkeypatch.setattr(os, 'open', native)
    monkeypatch.setattr(Path, 'open', path_open)
    return seen


@pytest.mark.parametrize('tiers', ['ssd', 'ram,ssd'])
def test_strict_shard_preserves_dtype_slice_empty_and_owned_bytes(tmp_path, monkeypatch, tiers):
    source, tensors = _shard(tmp_path)
    binding = publish_inputs(tmp_path, [source])
    resolver = bind(monkeypatch, binding)
    source.unlink()
    opened = _forbid_source(monkeypatch, source)
    with staged_tier_policy_test_context(tiers):
        with layer_streaming._source_safe_open(str(source), framework='pt') as reader:
            assert reader._handle is None
            assert set(reader.keys()) == set(tensors)
            assert reader.metadata() == {'fixture': 'strict-known-values'}
            got = {name: reader.get_tensor(name) for name in tensors}
            proxy = reader.get_slice('f32')
            assert proxy.get_shape() == [8, 32] and proxy.get_dtype() == torch.float32
            assert torch.equal(proxy[0].view(torch.uint8), tensors['f32'][0].view(torch.uint8))
            assert resolver._windows > 0
            with pytest.raises(RuntimeError, match='live reader windows'):
                resolver.close()
        assert resolver._windows == 0
        # Owned tensors remain valid after every descriptor window is closed.
        for name, expected in tensors.items():
            assert got[name].dtype == expected.dtype and got[name].shape == expected.shape
            assert torch.equal(got[name].view(torch.uint8), expected.view(torch.uint8))
    assert opened == [] and resolver.report()['bytes_from_pool'] == 0


@pytest.mark.parametrize('fault', ['unbound', 'missing', 'length', 'corrupt', 'nonregular', 'symlink', 'disallowed'])
def test_strict_whole_input_refuses_without_pool_fallback(tmp_path, monkeypatch, fault):
    source = tmp_path / 'blob.bin'
    raw = b'known-wire-values\x00\xff0123456789'
    source.write_bytes(raw)
    binding = publish_inputs(tmp_path, [source])
    resolver = bind(monkeypatch, binding)
    manifest = load_staged_manifest(binding['path'], binding['sha256'])
    obj = Path(manifest.extents[0].object_path)
    if fault == 'unbound':
        residency_map.reset_residency_resolver_for_tests()
        monkeypatch.delenv(residency_map.ENV_VAR)
    elif fault == 'missing':
        obj.unlink()
    elif fault == 'length':
        obj.chmod(0o600)
        obj.write_bytes(raw[:-1])
    elif fault == 'corrupt':
        obj.chmod(0o600)
        obj.write_bytes(b'X' + raw[1:])
    elif fault == 'nonregular':
        obj.unlink()
        obj.mkdir()
    elif fault == 'symlink':
        renamed = obj.with_suffix('.owned-original')
        obj.rename(renamed)
        obj.symlink_to(renamed)
    source.unlink()
    opened = _forbid_source(monkeypatch, source)
    with staged_tier_policy_test_context('ram' if fault == 'disallowed' else 'ssd'):
        with pytest.raises((TierPolicyRefused, OSError, RuntimeError, ValueError)):
            read_staged_whole_file(source, sha(raw), label='strict-wire')
    assert opened == []
    assert resolver.report()['bytes_from_pool'] == 0
    assert resolver.report()['bytes_from_stage'] == 0
    assert resolver._windows == 0


@pytest.mark.parametrize('wire_gzip', [False, True])
def test_nonzero_source_ranges_and_exact_scientific_contract(tmp_path, monkeypatch, wire_gzip):
    source = tmp_path / 'range.bin'
    raw = b'prefix-EXACT-TENSOR-BYTES-tail'
    source.write_bytes(raw)
    start, end = 7, 26
    assert raw[start:end] == b'EXACT-TENSOR-BYTES-'
    document = {'schema': 'prismaquant.read_contract.v1', 'entries': [
        {'path': str(source), 'offset': start, 'bytes': end - start, 'sha256': sha(raw[start:end])}]}
    wire = (json.dumps(document) + '\n').encode()
    if wire_gzip:
        wire = gzip.compress(wire, mtime=0)
    binding = publish_inputs(tmp_path, [source], ranges={source: [(start, end)]}, read_contract=wire)
    assert Path(binding['read_contract_path']).read_bytes() == wire
    assert binding['read_contract_sha256'] == sha(wire)
    resolver = bind(monkeypatch, binding)
    source.unlink()
    with staged_tier_policy_test_context('ssd'):
        entry = resolver.staged_range(source, start, end, declared_size=len(raw))
        assert entry['offset'] == start and entry['object_offset'] == 0
        window, key = acquire_entry_window(resolver, source, entry)
        with window:
            fd, _ = window.open(key)
            assert os.pread(fd, end - start, entry['object_offset']) == b'EXACT-TENSOR-BYTES-'
        with pytest.raises(OSError):
            os.fstat(fd)
    assert await_staged_spans(resolver, [(source, start, end, len(raw))]) == 'hit'
    assert await_staged_spans(resolver, [(source, 0, start, len(raw))]) == 'undeclared'


def test_scientific_contract_must_match_the_materialized_ranges(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'known bytes')
    wrong = json.dumps({'schema': 'prismaquant.read_contract.v1', 'entries': [
        {'path': str(source), 'offset': 0, 'bytes': 5, 'sha256': sha(b'known')}]}).encode()
    with pytest.raises(ValueError, match='exact scientific coverage'):
        publish_inputs(tmp_path, [source], read_contract=wrong)


def test_independent_windows_close_only_their_own_descriptors(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    raw = b'independent-owned-descriptors'
    source.write_bytes(raw)
    binding = publish_inputs(tmp_path, [source])
    resolver = bind(monkeypatch, binding)
    source.unlink()
    with staged_tier_policy_test_context('ssd'):
        entry = resolver.staged_read(source)
        first, one = acquire_entry_window(resolver, source, entry)
        second, two = acquire_entry_window(resolver, source, entry)
        with first:
            fd1, _ = first.open(one)
            with second:
                fd2, _ = second.open(two)
                assert fd1 != fd2
                assert os.pread(fd1, len(raw), 0) == raw
            with pytest.raises(OSError):
                os.fstat(fd2)
            assert os.pread(fd1, len(raw), 0) == raw
            assert resolver._windows == 1
        with pytest.raises(OSError):
            os.fstat(fd1)
        assert resolver._windows == 0
        with pytest.raises(RuntimeError, match='re-enter'):
            first.__enter__()


def test_window_close_waits_for_owned_reader_completion(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    raw = b'completion-boundary-known-bytes'
    source.write_bytes(raw)
    resolver = bind(monkeypatch, publish_inputs(tmp_path, [source]))
    entered, release, finished = (threading.Event() for _ in range(3))
    got, errors = [], []
    with staged_tier_policy_test_context('ssd'):
        window, key = acquire_entry_window(resolver, source, resolver.staged_read(source))
        with window:
            fd, _ = window.open(key)
            def reader():
                try:
                    entered.set()
                    assert release.wait(30)
                    got.append(os.pread(fd, len(raw), 0))
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    finished.set()
            worker = threading.Thread(target=reader)
            worker.start()
            try:
                assert entered.wait(30)
                with pytest.raises(RuntimeError, match='live reader windows'):
                    resolver.close()
            finally:
                release.set()
                worker.join(30)
            assert not worker.is_alive() and finished.is_set() and not errors
            assert got == [raw]
            assert os.pread(fd, len(raw), 0) == raw
        with pytest.raises(OSError):
            os.fstat(fd)
    assert resolver._windows == 0


def test_canonical_cache_helper_uses_isolated_python(tmp_path, monkeypatch):
    binding = {'path': 'fixture', 'sha256': '1' * 64, 'read_contract_sha256': '2' * 64,
               'read_contract_path': 'fixture-contract', 'allowed_tiers': 'ssd'}
    class Manifest:
        lifetime = {'mode': 'host_local_cache', 'root': '/tmp/flux-vanilla/owned-test'}
        def members(self):
            return [{'key': '3' * 64, 'bytes': 7}]
    manifest = Manifest()
    monkeypatch.setattr(staged_launch, 'validate_binding', lambda value: manifest)
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    for name in ('PRISMAQUANT_STAGED_MANIFEST', 'PRISMAQUANT_STAGED_MANIFEST_SHA256',
                 'PRISMAQUANT_READ_CONTRACT_SHA256', 'PRISMAQUANT_READ_CONTRACT_PATH',
                 'PRISMAQUANT_STAGED_ALLOWED_TIERS'):
        monkeypatch.setenv(name, 'fixture-restored-on-exit')
    helper = tmp_path / 'cache-helper.py'
    helper.write_bytes(b'# reviewed unit fixture helper\n')
    monkeypatch.setattr(staged_launch, 'CACHE_HELPER', str(helper))
    monkeypatch.setattr(staged_launch, 'CACHE_HELPER_SHA256', sha(helper.read_bytes()))
    calls = []
    class Executed(Exception):
        pass
    def executed(executable, argv):
        calls.append((executable, argv))
        raise Executed
    monkeypatch.setattr(os, 'execv', executed)
    with pytest.raises(Executed):
        staged_launch.launch(binding, ['python3', '-s', '-m', 'owned_payload'])
    executable, argv = calls.pop()
    assert executable == '/usr/bin/python3'
    assert argv[:4] == ['/usr/bin/python3', '-I', '-c', helper.read_text()]
    assert argv[argv.index('--') + 1:] == ['python3', '-s', '-m', 'owned_payload']
    members = json.loads((tmp_path / 'staged-members.json').read_bytes())
    assert members == [{'key': '3' * 64, 'bytes': 7}]
