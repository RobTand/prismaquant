"""PQ #2010: a terminal reread must not repopulate the bounded source cache."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors import safe_open

from prismaquant import tessera_calibration_cache as cc
from test_selected_source_authentication import complete_source  # noqa: F401


@pytest.fixture(params=('recording', 'selected'))
def source_owner(request, complete_source, monkeypatch):
    monkeypatch.setenv('PRISMAQUANT_UNSAFE_DISABLE_SEAL_CHECKS', '0')
    owners = []

    def create(**kwargs):
        if request.param == 'recording':
            owner = cc.record_capture_source(complete_source.kwargs['census_path'],
                model=complete_source.root, **kwargs)
        else:
            # Bind the same selected contract without exercising its unrelated
            # bootstrap JSON reads; only the held files below are under test.
            owner = cc.CaptureSourceAuthentication(complete_source.root,
                complete_source.canonical,
                complete_source.census['expert_projection']['producer']['source'],
                manifest_sha256=complete_source.kwargs['expected_sha256'], **kwargs)
        owners.append(owner)
        return owner

    yield SimpleNamespace(source=complete_source, kind=request.param, create=create)
    for owner in owners:
        owner.close()


@pytest.fixture
def trace(source_owner, monkeypatch):
    # Scale pages/blocks down to the existing sub-page fixture. All advice is
    # intercepted: these tests check policy/order, not OS cache residency.
    monkeypatch.setattr(cc, 'SOURCE_HASH_BLOCK_BYTES', 64)
    real_sysconf = os.sysconf
    monkeypatch.setattr(cc.os, 'sysconf',
        lambda name: 64 if name == 'SC_PAGE_SIZE' else real_sysconf(name))
    observed = SimpleNamespace(hashes=[], advice=[], bad_digest=None, advice_error=False)
    real_hash = cc.sha256

    def hash_file(path, **kwargs):
        start = len(observed.advice)
        call = SimpleNamespace(name=Path(path).name, kwargs=kwargs, advice=[])
        observed.hashes.append(call)
        try:
            digest = real_hash(path, **kwargs)
            return '0' * 64 if call.name == observed.bad_digest else digest
        finally:
            call.advice = observed.advice[start:]

    def advise(fd, offset, length, advice):
        os.fstat(fd)  # Every advice must precede that descriptor's close.
        assert advice == cc.os.POSIX_FADV_DONTNEED
        observed.advice.append((fd, offset, length))
        if observed.advice_error and length == 0:
            raise OSError('advice unavailable')

    monkeypatch.setattr(cc, 'sha256', hash_file)
    monkeypatch.setattr(cc.os, 'posix_fadvise', advise, raising=False)
    monkeypatch.setattr(cc.os, 'POSIX_FADV_DONTNEED', 4, raising=False)
    return observed


def _consume(owner, source, name, tensor):
    with owner.safe_open(safe_open, source.root / name, framework='pt', device='cpu') as reader:
        reader.get_tensor(tensor)


def _read_two_shards(owner, source):
    _consume(owner, source, 'selected.safetensors', 'model.layers.0.a.weight')
    _consume(owner, source, 'head.safetensors', 'lm_head.weight')


def _held(owner):
    return {name: (state['fd'], state['before'].st_size) for name, state in owner._files.items()}


def _reset(trace):
    trace.hashes.clear()
    trace.advice.clear()


def _assert_closed(owner, held, trace):
    assert owner._closed
    # Includes the separately opened hash descriptor when it received advice.
    for fd in {fd for fd, _ in held.values()} | {row[0] for row in trace.advice}:
        with pytest.raises(OSError):
            os.fstat(fd)


def _assert_final_advice(held, trace):
    assert [(fd, offset, length) for fd, offset, length in trace.advice if length == 0] == [
        (fd, 0, 0) for fd, _ in held.values()]


def _adopt_shards(owner, source, monkeypatch):
    # Reuse the existing proof fixture pattern, leaving the real stat/roster
    # adoption predicates in place. No source payload is read or copied.
    from prismaquant import streamed_model_identity
    paths = sorted(source.root.glob('*.safetensors'))
    checkpoint = cc.json.loads((source.root / 'model.safetensors.index.json').read_text())['weight_map']
    cached = {'fingerprints': [streamed_model_identity.stat_fingerprint(str(path.resolve()), path.stat())
                               for path in paths]}
    identity = dict(checkpoint_weight_map=checkpoint,
        shards=[dict(path=str(path.resolve()), sha256=source.canonical['source_files'][path.name])
                for path in paths])
    proof = source.root.parent / 'identity-proof.json'
    proof.write_text('{}')
    monkeypatch.setattr(streamed_model_identity, '_read_streamed_model_identity_cache',
        lambda *args, **kwargs: (cached, identity))
    monkeypatch.setattr(streamed_model_identity, '_local_checkpoint_shards',
        lambda root: (checkpoint, paths))
    owner.adopt_streamed_identity_cache(proof)


@pytest.mark.parametrize('release_read_pages', (True, False))
def test_terminal_reread_releases_blocks_and_every_held_file(source_owner, trace, release_read_pages):
    source = source_owner.source
    checks = []
    owner = source_owner.create(release_read_pages=release_read_pages, resource_check=checks.append)
    _read_two_shards(owner, source)
    _consume(owner, source, 'selected.safetensors', 'model.layers.0.a.weight')
    # Header-only access must not acquire a digest just to close the owner.
    with owner.safe_open(safe_open, source.root / 'unused.safetensors', framework='pt', device='cpu') as reader:
        assert reader.keys()
    assert [call.name for call in trace.hashes] == ['selected.safetensors', 'head.safetensors']
    # Recording admission retains pages for consumers, even with bounded close.
    assert all(call.kwargs['release_read_pages'] ==
               (release_read_pages and source_owner.kind == 'selected') for call in trace.hashes)
    owner.release_retained_pages()
    if source_owner.kind == 'recording' and release_read_pages:
        assert owner._released == {'selected.safetensors', 'head.safetensors'}
    held = _held(owner)
    states = dict(owner._files)
    proofs = {name: (state['sha256'], state['sha256_source'], state['payload_reads'])
              for name, state in states.items()}
    released = set(owner._released)
    _reset(trace)
    checks.clear()

    owner.close()

    assert [call.name for call in trace.hashes] == ['selected.safetensors', 'head.safetensors']
    assert checks and all(label.startswith(('before_capture_hash:', 'after_capture_hash:'))
                          for label in checks)
    for call in trace.hashes:
        assert call.kwargs['file_descriptor'] == held[call.name][0]
        assert call.kwargs['release_read_pages'] is release_read_pages
        assert call.kwargs['resource_check'] == checks.append
        expected = [(offset, 64) for offset in range(0, held[call.name][1] // 64 * 64, 64)]
        assert [(offset, length) for _, offset, length in call.advice] == (
            expected if release_read_pages else [])
    if release_read_pages:
        _assert_final_advice(held, trace)
        assert any(size % 64 for _, size in held.values())  # Final advice covers the tail, too.
    else:
        assert not trace.advice
    # Cleanup is not a new authentication/release event.
    assert {name: (state['sha256'], state['sha256_source'], state['payload_reads'])
            for name, state in states.items()} == proofs
    assert owner._released == released
    _assert_closed(owner, held, trace)
    counts = len(trace.hashes), len(trace.advice)
    owner.close()
    assert (len(trace.hashes), len(trace.advice)) == counts


@pytest.mark.parametrize('refuse_at', (1, 2, 3))
def test_terminal_guard_refusal_releases_partial_and_remaining_files(source_owner, trace, refuse_at):
    armed = False
    calls = 0

    def guard(label):
        nonlocal calls
        assert label.startswith(('before_capture_hash:', 'after_capture_hash:'))
        if armed:
            calls += 1
            if calls == refuse_at:
                raise RuntimeError('terminal resource guard refused')

    owner = source_owner.create(release_read_pages=True, resource_check=guard)
    _read_two_shards(owner, source_owner.source)
    owner.release_retained_pages()
    held = _held(owner)
    _reset(trace)
    armed = True

    with pytest.raises(RuntimeError, match='terminal resource guard refused'):
        owner.close()

    assert calls == refuse_at
    assert len(trace.hashes) == 1
    assert trace.hashes[0].kwargs['resource_check'] is guard
    assert trace.hashes[0].kwargs['release_read_pages'] is True
    # Refusal before read, after read, or before the second block respectively.
    assert [(offset, length) for _, offset, length in trace.hashes[0].advice] == (
        [(0, 64)] if refuse_at >= 2 else [])
    _assert_final_advice(held, trace)
    _assert_closed(owner, held, trace)


@pytest.mark.parametrize('advice_error', (False, True))
def test_terminal_digest_mismatch_still_cleans_all_descriptors(source_owner, trace, advice_error):
    owner = source_owner.create(release_read_pages=True)
    _read_two_shards(owner, source_owner.source)
    held = _held(owner)
    _reset(trace)
    trace.bad_digest = 'selected.safetensors'
    trace.advice_error = advice_error

    with pytest.raises(RuntimeError, match='authenticated source changed during consumption'):
        owner.close()

    assert [call.name for call in trace.hashes] == ['selected.safetensors']
    assert trace.hashes[0].advice  # Release during reread, before digest comparison.
    _assert_final_advice(held, trace)
    _assert_closed(owner, held, trace)


@pytest.mark.parametrize('changed', (False, True))
def test_proof_rows_remain_zero_read_and_stat_fenced(source_owner, trace, monkeypatch, changed):
    source = source_owner.source
    owner = source_owner.create(release_read_pages=True)
    if source_owner.kind == 'recording':
        owner.adopt_recorded_digests(source.canonical['source_files'])
    else:
        _adopt_shards(owner, source, monkeypatch)
    _read_two_shards(owner, source)
    owner.release_retained_pages()
    assert not trace.hashes
    held = _held(owner)
    _reset(trace)
    if changed:
        path = source.root / 'selected.safetensors'
        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        with pytest.raises(RuntimeError, match='authenticated source changed during consumption'):
            owner.close()
    else:
        owner.close()

    assert not trace.hashes
    _assert_final_advice(held, trace)
    assert all(length == 0 for _, _, length in trace.advice)
    _assert_closed(owner, held, trace)


def test_fresh_stat_refusal_precedes_terminal_hash(source_owner, trace):
    source = source_owner.source
    owner = source_owner.create(release_read_pages=True)
    _read_two_shards(owner, source)
    held = _held(owner)
    _reset(trace)
    path = source.root / 'selected.safetensors'
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))

    with pytest.raises(RuntimeError, match='authenticated source changed during consumption'):
        owner.close()

    assert not trace.hashes
    _assert_final_advice(held, trace)
    _assert_closed(owner, held, trace)


def test_active_reader_defers_terminal_cleanup_and_tensor_stays_live(source_owner, trace):
    source = source_owner.source
    owner = source_owner.create(release_read_pages=True)
    with owner.safe_open(safe_open, source.root / 'selected.safetensors', framework='pt', device='cpu') as reader:
        tensor = reader.get_tensor('model.layers.0.a.weight')
        held = _held(owner)
        _reset(trace)
        with pytest.raises(RuntimeError, match='active readers'):
            owner.close()
        assert not trace.hashes and not trace.advice and not owner._closed
        assert all(os.fstat(fd).st_size == size for fd, size in held.values())
        assert float(tensor.sum()) == 6.0

    owner.close()
    assert float(tensor.sum()) == 6.0
    assert len(trace.hashes) == 1
    _assert_final_advice(held, trace)
    _assert_closed(owner, held, trace)
