"""Migrated shard, whole-entry, metadata and IO lifetime regression oracles."""
import copy
import json
import os
from pathlib import Path
import sys
import threading

import pytest
import torch
from safetensors.torch import save

from prismaquant import io_engine, residency_map, residency_shard_reader, stage_b_prep_io
from prismaquant.staged_manifest import ManifestRefused, load_staged_manifest, publish_staged_json, materialize
from prismaquant.staged_tier_policy import staged_tier_policy_test_context, TierPolicyRefused
from prismaquant.staged_whole_file import read_staged_entry, read_staged_whole_file
from staged_native_fixtures import fixture, sha, bind


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    from prismaquant import staged_tier_policy
    monkeypatch.setattr(staged_tier_policy, '_ACTIVE', None)
    monkeypatch.setattr(staged_tier_policy, '_OWNER', None)
    monkeypatch.setattr(staged_tier_policy, '_STACK', [])
    residency_map.reset_residency_resolver_for_tests()
    stage_b_prep_io.reset_staged_reads_for_tests()
    yield
    residency_map.reset_residency_resolver_for_tests()
    stage_b_prep_io.reset_staged_reads_for_tests()
    assert not any(n == 'prismabuild' or n.startswith('prismabuild.') for n in sys.modules)


def forbidden(*args, **kwargs):
    raise AssertionError('canonical opener called')


def test_immutable_coverage_and_cancel_are_conclusive(tmp_path, monkeypatch):
    source, data, binding, _ = fixture(tmp_path)
    resolver = bind(monkeypatch, binding)
    length = data['entries'][0]['source']['bytes']
    wanted = [(source, 0, length, length)]
    cancel = threading.Event()
    cancel.set()
    assert residency_shard_reader.await_staged_spans(
        None, wanted,  cancel=cancel) == 'undeclared'
    assert residency_shard_reader.await_staged_spans(resolver, wanted) == 'hit'


def test_owned_whole_io_and_metadata_without_canonical_file(tmp_path, monkeypatch):
    raw = b'independently-known\x00\xff\n'
    source, data, binding, _ = fixture(tmp_path, raw)
    resolver = bind(monkeypatch, binding)
    reads = stage_b_prep_io.bind_staged_reads(manifest_sha256=binding['read_contract_sha256'], allowed_tiers='ssd')
    with staged_tier_policy_test_context('ssd'):
        assert read_staged_whole_file(source, sha(raw), label='fixture') == raw
        got, receipt, _ = io_engine.read_file(source, len(raw), staged=resolver.staged_read(source))
        assert got == raw and receipt['sha256'] == sha(raw)
        value, _ = io_engine.load_file(source, len(raw), binding=sha(raw),
                                      decode=lambda raw, receipt, staged: (raw, None))
        assert value == raw
        assert reads.whole(source, where='metadata') == raw
        sealed = read_staged_entry(resolver, source, resolver.staged_read(source), label='sealed', sealed=True)
        try:
            sealed.require_sealed()
            assert bytes(sealed) == raw
        finally:
            sealed.close()
    assert not source.exists()
    assert resolver.report()['bytes_from_pool'] == 0


def test_shard_dtypes_metadata_slices_owned_buffers_and_nonzero_offsets(tmp_path, monkeypatch):
    tensors = {'f32': torch.arange(8192, dtype=torch.float32).reshape(64, 128),
               'bf16': torch.arange(128).to(torch.bfloat16),
               'f16': torch.arange(32).to(torch.float16),
               'i8': torch.arange(-32, 32, dtype=torch.int8),
               'u8': torch.arange(64, dtype=torch.uint8),
               'flag': torch.tensor([True, False, True]), 'empty': torch.zeros((0, 4))}
    raw = save(tensors, metadata={'oracle': 'known fixture'})
    base = 8 + int.from_bytes(raw[:8], 'little')
    source, data, binding, _ = fixture(tmp_path, raw, ranges=[(0, base), (base, len(raw))])
    resolver = bind(monkeypatch, binding)
    with staged_tier_policy_test_context('ssd'):
        with residency_shard_reader.staged_shard_opener(source, forbidden)(str(source), framework='pt') as reader:
            assert set(reader.keys()) == set(tensors)
            assert reader.metadata() == {'oracle': 'known fixture'}
            retained = []
            for name, expected in tensors.items():
                got = reader.get_tensor(name)
                assert got.dtype == expected.dtype and got.shape == expected.shape
                assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
                retained.append(got)
            assert reader.get_slice('f32').get_shape() == [64, 128]
            assert torch.equal(reader.get_slice('f32')[:2], tensors['f32'][:2])
        for got, expected in zip(retained, tensors.values()):
            assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
    assert resolver.report()['bytes_from_pool'] == 0


@pytest.mark.parametrize('fault', ['missing', 'symlink', 'parent-symlink', 'length', 'digest', 'change'])
def test_whole_entry_refuses_without_canonical_read(tmp_path, monkeypatch, fault):
    source, data, binding, _ = fixture(tmp_path)
    obj = Path(data['entries'][0]['object']['path'])
    resolver = bind(monkeypatch, binding)
    with staged_tier_policy_test_context('ssd'):
        if fault == 'change':
            assert read_staged_whole_file(source, data['entries'][0]['source']['sha256'], label='before') == b'known-fixture\x00\xff'
        if fault == 'missing':
            obj.unlink()
        elif fault == 'symlink':
            dest = obj.with_name('replacement'); obj.rename(dest); obj.symlink_to(dest)
        elif fault == 'parent-symlink':
            parent = obj.parent; dest = parent.with_name('replacement'); parent.rename(dest); parent.symlink_to(dest, target_is_directory=True)
        elif fault == 'length':
            obj.write_bytes(b'short')
        else:
            blob = bytearray(obj.read_bytes()); blob[14] ^= 1; obj.write_bytes(blob)
        with pytest.raises((OSError, ValueError, RuntimeError, TierPolicyRefused)):
            read_staged_whole_file(source, data['entries'][0]['source']['sha256'], label=fault)
    assert not source.exists()


@pytest.mark.parametrize('fault', ['manifest-digest', 'null-digest', 'overlap', 'object-bounds', 'source-bounds', 'zero', 'source-conflict', 'producer-type'])
def test_manifest_refusal(tmp_path, fault):
    source, data, binding, _ = fixture(tmp_path)
    if fault == 'manifest-digest':
        with pytest.raises(ManifestRefused, match='digest mismatch'):
            load_staged_manifest(binding['path'], '0' * 64)
        return
    row = data['entries'][0]
    if fault == 'null-digest': row['object']['sha256'] = None
    elif fault == 'overlap': data['entries'].append(copy.deepcopy(row))
    elif fault == 'object-bounds': row['object_offset'] = row['object']['bytes']
    elif fault == 'source-bounds': row['source_offset'] = 1
    elif fault == 'zero': row['length'] = 0
    elif fault == 'producer-type': data['producer']['source_commit'] = int('1' * 40)
    else:
        second = copy.deepcopy(row); second['source']['sha256'] = '0' * 64; data['entries'].append(second)
    target = tmp_path / 'bad.json'
    digest = publish_staged_json(target, data)
    with pytest.raises(ManifestRefused):
        load_staged_manifest(target, digest)


def test_header_and_payload_gaps_refuse(tmp_path, monkeypatch):
    raw = save({'weight': torch.arange(16, dtype=torch.float32)})
    base = 8 + int.from_bytes(raw[:8], 'little')
    source, data, binding, _ = fixture(tmp_path, raw, ranges=[(base, len(raw))])
    bind(monkeypatch, binding)
    with staged_tier_policy_test_context('ssd'):
        with residency_shard_reader.staged_shard_opener(source, forbidden)(str(source), framework='pt') as reader:
            with pytest.raises(TierPolicyRefused): reader.keys()
    residency_map.reset_residency_resolver_for_tests()
    other = tmp_path / 'other'; other.mkdir()
    source, _, binding, _ = fixture(other, raw, ranges=[(0, base), (base + 4, len(raw))])
    bind(monkeypatch, binding)
    with staged_tier_policy_test_context('ssd'):
        with residency_shard_reader.staged_shard_opener(source, forbidden)(str(source), framework='pt') as reader:
            assert reader.keys() == ['weight']
            with pytest.raises(TierPolicyRefused): reader.get_tensor('weight')


def test_materializer_publishes_actual_hashes_and_no_partial_manifest(tmp_path):
    source = tmp_path / 'source'; source.write_bytes(b'prefix-KNOWN-extent-tail')
    dest = tmp_path / 'phase'; dest.mkdir()
    out = tmp_path / 'manifest.json'
    src = {'path': str(source), 'bytes': source.stat().st_size, 'sha256': sha(source.read_bytes())}
    kwargs = dict(destination=dest, manifest_path=out, read_contract_sha256='3' * 64, phase_id='offline',
        producer={'source_commit': '1' * 40, 'receipt_sha256': '2' * 64},
        lifetime={'mode': 'retained_shared_phase', 'namespace': str(dest), 'reader_scopes': ['owned'],
                  'retention': 'until_all_readers_stopped_or_fenced'})
    digest = materialize([{'source': src, 'source_offset': 7, 'length': 12}], **kwargs)
    manifest = load_staged_manifest(out, digest)
    obj = manifest.extents[0]
    assert Path(obj.object_path).read_bytes() == b'KNOWN-extent'
    assert obj.object_sha256 == sha(b'KNOWN-extent')
    assert obj.source_sha256 == sha(b'prefix-KNOWN-extent-tail')
    assert not list(dest.glob('.extent-*'))
    bad = tmp_path / 'bad.json'; kwargs['manifest_path'] = bad
    with pytest.raises(ManifestRefused, match='digest mismatch'):
        materialize([{'source': src, 'source_offset': 0, 'length': 6, 'sha256': '0' * 64}], **kwargs)
    assert not bad.exists() and not list(dest.glob('.extent-*'))


def test_offline_copy_hashes_known_multichunk_bytes(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'object'
    source.write_bytes(b'xx123456789yy')
    with source.open('rb') as incoming, target.open('wb') as outgoing:
        actual = io_engine.copy_extent(incoming.fileno(), outgoing.fileno(), 2, 9, chunk_bytes=3)
    assert target.read_bytes() == b'123456789'
    assert actual == '15e2b0d3c33891ebb0f1ef609ec419420c20e320ce94c65fbc8c3312448eb225'


def test_window_fork_refusal_and_closed_descriptors(tmp_path, monkeypatch):
    from prismaquant.staged_lease import acquire_entry_window
    source, _, binding, _ = fixture(tmp_path)
    resolver = bind(monkeypatch, binding)
    with staged_tier_policy_test_context('ssd'):
        window, key = acquire_entry_window(resolver, source, resolver.staged_read(source))
        with window:
            fd, _ = window.open(key)
            pid = os.fork()
            if pid == 0:
                try:
                    residency_map.residency_resolver()
                except RuntimeError:
                    pass
                else:
                    os._exit(8)
                try:
                    window.open(key)
                except RuntimeError:
                    os._exit(0)
                os._exit(9)
            assert os.waitpid(pid, 0)[1] == 0
            assert os.pread(fd, 5, 13) == b'known'
        with pytest.raises(OSError): os.fstat(fd)
    assert not resolver._closed


def test_staged_io_stream_closes_after_worker_completion(tmp_path, monkeypatch):
    from prismaquant.staged_lease import LeaseWindow
    raw = b'publication-boundary-oracle'
    source, _, binding, _ = fixture(tmp_path, raw)
    bind(monkeypatch, binding)
    entered, release, finished, closing = (threading.Event() for _ in range(4))
    real_read, real_exit = os.preadv, LeaseWindow.__exit__
    fd_seen = []

    def read(fd, buffers, offset):
        fd_seen.append(fd)
        entered.set()
        assert release.wait(30)
        got = real_read(fd, buffers, offset)
        finished.set()
        return got

    def exited(window, *args):
        assert finished.is_set()
        return real_exit(window, *args)

    monkeypatch.setattr(os, 'preadv', read)
    monkeypatch.setattr(LeaseWindow, '__exit__', exited)
    entry = io_engine.ReadEntry('owned', str(source), len(raw), len(raw), len(raw), sha(raw),
        lambda buffer, receipt, staged: (bytes(buffer), None), 'group')
    with staged_tier_policy_test_context('ssd'):
        stream = io_engine.read_stream([entry], budget=io_engine.FixedBudget(buffer_bytes=4096, headroom=4096))
        assert entered.wait(30)
        result, errors = [], []
        def consume():
            try:
                result.extend(item.value for item in stream.take('group'))
            except BaseException as exc:
                errors.append(exc)
            finally:
                closing.set()
                stream.close()
        thread = threading.Thread(target=consume)
        thread.start()
        release.set()
        thread.join(30)
        assert not thread.is_alive() and closing.is_set() and not errors
        assert result == [raw]
        for fd in fd_seen:
            with pytest.raises(OSError): os.fstat(fd)


def test_static_launch_checks_never_stat_owner_local_objects(tmp_path, monkeypatch):
    from prismaquant.staged_launch import validate_binding, validate_read_contract
    source, data, binding, contract = fixture(tmp_path)
    life = data['lifetime'] = {'mode': 'host_local_cache', 'root': '/tmp/flux-vanilla/not-local', 'owner_host': 'sparky'}
    filename = tmp_path / 'host.json'; binding['path'] = str(filename); binding['sha256'] = publish_staged_json(filename, data)
    assert validate_binding(binding, placement='host:sparky').lifetime == life
    validate_read_contract(binding, contract)
    with pytest.raises(ValueError, match='owner placement'):
        validate_binding(binding, placement='portable')
    binding['read_contract_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='read contract'):
        validate_binding(binding)


def test_legacy_scientific_manifest_exact_bytes_survive_without_sdk(tmp_path, monkeypatch):
    import gzip
    from prismaquant.staged_launch import validate_binding
    from prismaquant.staged_lease import load_sealed_manifest
    source, data, binding, contract = fixture(tmp_path)
    row = data['entries'][0]
    original = {'schema': 'prismaquant.prismabuild.data_manifest.v1',
        'produced_by': {'tool': 'independent-publisher'}, 'mount_prefix': str(tmp_path),
        'entries': [{'path': str(source), 'offset': 0, 'bytes': row['length'], 'sha256': row['sha256']}],
        'entry_count': 1, 'total_bytes': row['length'], 'annotations': {'scientific_input': 'retained-exactly'}}
    wire = gzip.compress((json.dumps(original) + '\n').encode(), mtime=0)
    contract.write_bytes(wire)
    binding['read_contract_sha256'] = data['read_contract_sha256'] = sha(wire)
    filename = tmp_path / 'legacy-staged.json'; binding['path'] = str(filename); binding['sha256'] = publish_staged_json(filename, data)
    validate_binding(binding)
    bind(monkeypatch, binding)
    assert load_sealed_manifest(sha(wire)) == original
    original['entries'][0]['sha256'] = None
    contract.write_bytes(gzip.compress(json.dumps(original).encode(), mtime=0))
    binding['read_contract_sha256'] = data['read_contract_sha256'] = sha(contract.read_bytes())
    filename = tmp_path / 'null-legacy.json'; binding['path'] = str(filename); binding['sha256'] = publish_staged_json(filename, data)
    with pytest.raises(ValueError): validate_binding(binding)


def test_close_requires_completed_windows(tmp_path, monkeypatch):
    from prismaquant.staged_lease import acquire_entry_window
    source, _, binding, _ = fixture(tmp_path)
    resolver = bind(monkeypatch, binding)
    with staged_tier_policy_test_context('ssd'):
        window, key = acquire_entry_window(resolver, source, resolver.staged_read(source))
        with window:
            fd, _ = window.open(key)
            with pytest.raises(RuntimeError, match='join readers first'): resolver.close()
            assert os.pread(fd, 5, 13) == b'known'
    resolver.close()
    assert resolver._closed and resolver._fds == {} and resolver._pins == []


def test_host_local_map_cannot_be_replaced_by_ambient_context(tmp_path, monkeypatch):
    from prismaquant.staged_launch import validate_binding
    source, data, binding, _ = fixture(tmp_path)
    monkeypatch.setattr(residency_map.socket, 'gethostname', lambda: 'dl380g10')
    data['lifetime'] = {'mode': 'host_local_cache', 'root': str(tmp_path / 'cache'), 'owner_host': 'dl380g10'}
    filename = tmp_path / 'host-manifest.json'; binding['path'] = str(filename); binding['sha256'] = publish_staged_json(filename, data)
    validate_binding(binding, placement='host:dl380g10')
    monkeypatch.delenv('FLUXCACHE_MAP_FILE', raising=False)
    monkeypatch.setenv('PRISMABUILD_ACTION_KEY', 'a' * 64)
    monkeypatch.setenv('PRISMABUILD_READER_HELPER_ROOT', str(tmp_path))
    with pytest.raises(ManifestRefused, match='inherited map/pins'):
        residency_map.ResidencyResolver(filename, manifest_sha256=binding['sha256'])


def test_materializer_reuses_only_verified_identical_publication(tmp_path):
    source = tmp_path / 'source'; source.write_bytes(b'known-known')
    dest = tmp_path / 'phase'; dest.mkdir()
    src = {'path': str(source), 'bytes': 11, 'sha256': sha(b'known-known')}
    out = tmp_path / 'manifest.json'
    digest = materialize([{'source': src, 'source_offset': 0, 'length': 5},
                          {'source': src, 'source_offset': 6, 'length': 5}],
        destination=dest, manifest_path=out, read_contract_sha256='3'*64, phase_id='dedup',
        producer={'source_commit': '1'*40, 'receipt_sha256': '2'*64},
        lifetime={'mode': 'retained_shared_phase', 'namespace': str(dest), 'reader_scopes': ['owned'],
                  'retention': 'until_all_readers_stopped_or_fenced'})
    manifest = load_staged_manifest(out, digest)
    assert len(manifest.extents) == 2 and len(manifest.members()) == 1
    assert (dest / sha(b'known')).read_bytes() == b'known'


def test_source_result_and_publication_authority_cannot_use_ambient_context(tmp_path, monkeypatch):
    from prismaquant import staged_lease
    for name in ('client_sdk', 'sdk_submodule', 'resolve_context', 'load_claimed_manifest',
                 'discover_launch_queue_root', 'set_lease_helper_root'):
        assert not hasattr(staged_lease, name)
    with pytest.raises(stage_b_prep_io.PreparationPublicationRefused,
                       match='native preparation declaration unavailable'):
        stage_b_prep_io.bind_preparation_publication(tmp_path, required=True)
    assert stage_b_prep_io.bind_preparation_publication(tmp_path, required=False) is None


def test_manifest_duplicate_fields_and_symlink_refuse(tmp_path):
    _, _, binding, _ = fixture(tmp_path)
    raw = Path(binding['path']).read_bytes()
    duplicate = raw.replace(b'"source_offset":0', b'"source_offset":0,"source_offset":0')
    assert duplicate != raw
    filename = tmp_path / 'duplicate.json'; filename.write_bytes(duplicate)
    with pytest.raises(ManifestRefused, match='duplicate'):
        load_staged_manifest(filename, sha(duplicate))
    alias = tmp_path / 'alias.json'; alias.symlink_to(binding['path'])
    with pytest.raises(OSError): load_staged_manifest(alias, binding['sha256'])


def test_manifest_object_set_cap_is_finite(tmp_path):
    _, data, binding, _ = fixture(tmp_path)
    original = data['entries'][0]
    entries = []
    for i in range(1025):
        row = copy.deepcopy(original)
        row['source']['bytes'] = 1025
        row['source_offset'] = i
        row['length'] = row['object']['bytes'] = 1
        row['object_offset'] = 0
        row['object']['sha256'] = f'{i:064x}'
        row['object']['path'] = str(Path(original['object']['path']).with_name(f'object-{i}'))
        entries.append(row)
    data['entries'] = entries[:1024]
    filename = tmp_path / 'max.json'; digest = publish_staged_json(filename, data)
    assert len(load_staged_manifest(filename, digest).members()) == 1024
    data['entries'] = entries
    filename = tmp_path / 'over.json'; digest = publish_staged_json(filename, data)
    with pytest.raises(ManifestRefused, match='1024'):
        load_staged_manifest(filename, digest)


def test_prefetch_threads_share_one_resolver_and_owned_bytes(tmp_path, monkeypatch):
    raw = b'concurrent-known-bytes'
    source, _, binding, _ = fixture(tmp_path, raw)
    bind(monkeypatch, binding)
    residency_map.reset_residency_resolver_for_tests()
    start = threading.Barrier(2)
    results, errors = [], []
    def read():
        try:
            start.wait(30)
            owner = residency_map.residency_resolver()
            results.append((owner, read_staged_whole_file(source, sha(raw), label='prefetch')))
        except BaseException as exc:
            errors.append(exc)
    with staged_tier_policy_test_context('ssd'):
        threads = [threading.Thread(target=read) for _ in range(2)]
        for thread in threads: thread.start()
        for thread in threads:
            thread.join(30)
            assert not thread.is_alive()
    assert not errors and len(results) == 2
    assert results[0][0] is results[1][0]
    assert [value for _, value in results] == [raw, raw]


def test_io_missing_binding_refuses_even_when_canonical_bytes_exist(tmp_path, monkeypatch):
    source, _, binding, _ = fixture(tmp_path)
    bind(monkeypatch, binding)
    canonical = tmp_path / 'undeclared-canonical'
    canonical.write_bytes(b'canonical-must-not-be-read')
    with staged_tier_policy_test_context('ssd'):
        with pytest.raises(TierPolicyRefused, match='staged-not-serving'):
            io_engine.load_file(canonical, 128, binding=sha(b'canonical-must-not-be-read'),
                                decode=lambda *args: forbidden())
