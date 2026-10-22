"""Held-digest verification gates quantum completion without retaining reread pages."""
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prismaquant import capture_layer_chain as chain
from prismaquant import tessera_calibration_cache as cc
from test_capture_layer_chain import _Chain


@pytest.mark.parametrize('mutate', [False, True])
def test_quantum_verifies_before_fragment_and_owner_completion(tmp_path, monkeypatch, mutate):
    fixture = _Chain(tmp_path, prepare=False)
    shard = fixture.source / 'model.safetensors'
    save_file({'w': torch.arange(16.)}, str(shard))
    fixture.prepare()
    complete = chain.ChainQuantum.complete
    observed = []

    def consuming_complete(quantum, **kwargs):
        owner = quantum.source_authentication
        with owner.safe_open(safe_open, shard, framework='pt') as reader:
            reader.get_tensor('w')
        held = owner._files[shard.name]['fd']
        if mutate:
            # Model the ZFS metadata observability limit, not a timestamp race.
            signature = cc.file_stat_signature
            monkeypatch.setattr(cc, 'file_stat_signature',
                                lambda value: (*signature(value)[:-1], 0))
            before = shard.stat()
            with shard.open('r+b') as handle:
                handle.seek(-1, 2)
                old = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([old[0] ^ 1]))
            os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns))
            owner.require_unchanged()  # The metadata fence admits these bytes.
        try:
            path = complete(quantum, **kwargs)
            assert owner._closed
            observed.append(path)
            return path
        finally:
            if owner._closed:
                with pytest.raises(OSError):
                    os.fstat(held)

    monkeypatch.setattr(chain.ChainQuantum, 'complete', consuming_complete)
    if mutate:
        with pytest.raises(RuntimeError, match='changed during consumption'):
            fixture.quantum((0, 1))
        assert not chain.fragment_path(fixture.root, 0, 1).exists()
        # A failed quantum cannot supply a successor or a join, even though its
        # journalled unit/boundary bytes survive for a genuine retry.
        with pytest.raises(chain.CaptureChainRefused):
            chain.join(fixture.root, census_path=fixture.census)
        assert not (fixture.root / 'capture_manifest.json').exists()
        monkeypatch.setattr(chain.ChainQuantum, 'complete', complete)
        with pytest.raises(chain.CaptureChainRefused):
            fixture.quantum((1, 2))
    else:
        path = fixture.quantum((0, 1))
        assert observed == [path]
        fragment = chain.read_fragment(fixture.root, fixture.prep, 0, 1)
        assert fragment['source_authentication']['verified_files'][0]['name'] == shard.name
        fixture.quantum((1, 2))
        assert Path(chain.join(fixture.root, census_path=fixture.census)['manifest']['path']).is_file()


@pytest.mark.parametrize('recording', [False, True], ids=['selected', 'recording'])
@pytest.mark.parametrize('failure', [None, 'guard', 'digest'])
def test_close_reread_releases_pages_incrementally_and_on_failure(
    tmp_path, monkeypatch, recording, failure,
):
    page = os.sysconf('SC_PAGE_SIZE')
    shard = tmp_path / 'model.safetensors'
    save_file({'w': torch.arange(page * 3, dtype=torch.float32)}, str(shard))
    identity = {'source_files': {shard.name: cc.sha256(shard)}}
    if recording:
        owner = cc.CaptureSourceAuthentication.recording(tmp_path, {}, release_read_pages=True)
    else:
        owner = cc.CaptureSourceAuthentication(tmp_path, identity, {},
            manifest_sha256='0' * 64, release_read_pages=True)
    with owner.safe_open(safe_open, shard, framework='pt') as reader:
        reader.get_tensor('w')
    if recording:
        assert owner.release_retained_pages() == (shard.name,)
    held = owner._files[shard.name]['fd']
    advised, flags = [], []
    hash_source = cc.sha256

    def hash_spy(path, **kwargs):
        flags.append(kwargs['release_read_pages'])
        return hash_source(path, **kwargs)

    def guard(stage):
        # Fail after the first block, not before the reread does any IO.
        if stage.startswith('after_capture_hash:'):
            raise RuntimeError('close resource refusal')

    monkeypatch.setattr(cc, 'sha256', hash_spy)
    monkeypatch.setattr(cc, 'SOURCE_HASH_BLOCK_BYTES', page)
    monkeypatch.setattr(os, 'posix_fadvise', lambda *args: advised.append(args))
    if failure == 'guard':
        owner.resource_check = guard
    elif failure == 'digest':
        owner._files[shard.name]['sha256'] = '0' * 64
    if failure:
        with pytest.raises(RuntimeError, match='close resource refusal|changed during consumption'):
            owner.close()
    else:
        owner.close()
    assert flags == [True]
    assert any(offset == 0 and length == page and advice == os.POSIX_FADV_DONTNEED
               for _fd, offset, length, advice in advised)
    # Final advice covers the short tail as well as a partial/aborted hash. It
    # must happen even for a file already released by the streaming traversal.
    assert (held, 0, 0, os.POSIX_FADV_DONTNEED) in advised
    assert owner._closed
    with pytest.raises(OSError):
        os.fstat(held)
    owner.close()  # Idempotent cleanup never repeats the reread.
    assert flags == [True]
