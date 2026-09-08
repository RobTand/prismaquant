"""Verified capture reads consume each source byte once before deserialization."""
import builtins
import io
import os
from pathlib import Path

import torch

from prismaquant import tessera_calibration_cache as cc
from test_tessera_calibration_cache import capture


def policy(*, buffer=1024**2, scratch=1024**2):
    return dict(schema='prismaquant.verified_activation_load.v1',
                max_buffer_bytes=buffer, max_scratch_bytes=scratch)


def test_prefetch_reads_selected_artifact_bytes_once(capture, monkeypatch):
    root, _path, census, identity, acts, hessians, record = capture
    artifact = root / 'inputs/a.pt'
    count = {'bytes': 0, 'opens': 0}
    class Counted:
        def __init__(self, handle):
            self.handle = handle
        def __getattr__(self, name):
            return getattr(self.handle, name)
        def read(self, *args):
            result = self.handle.read(*args)
            count['bytes'] += len(result)
            return result
        def readinto(self, target):
            size = self.handle.readinto(target)
            count['bytes'] += size or 0
            return size
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
    def counting(original):
        def opened(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            actual = Path(os.readlink(f'/proc/self/fd/{path}')) if isinstance(path, int) else Path(path)
            if actual == artifact:
                count['opens'] += 1
                return Counted(result)
            return result
        return opened
    monkeypatch.setattr(builtins, 'open', counting(builtins.open))
    monkeypatch.setattr(io, 'open', counting(io.open))
    values, _ = cc.prefetch_capture(record['path'], expected_identity=identity,
        census=census, names=['a'], device='cpu', verified_load_policy=policy())
    assert torch.equal(values[0]['a'], acts['a'])
    assert torch.equal(values[1]['a'], hessians['a'])
    assert count['bytes'] == artifact.stat().st_size, count

import hashlib
import json
import weakref
import zipfile

import pytest
from prismaquant import perturbed_x_cache as px


def load(path, **kwargs):
    return px.load_verified_activation_cache_entry(path,
        expected_sha256=kwargs.pop('expected_sha256', cc.sha256(path)),
        policy=kwargs.pop('policy', policy(buffer=8*1024**2)),
        max_storage_bytes=kwargs.pop('max_storage_bytes', 4*1024**2), **kwargs)


@pytest.mark.parametrize('config', [{}, {'schema': 'wrong'}, policy(buffer=True),
    policy(scratch=1024), {**policy(), 'unknown': 1}])
def test_closed_policy_rejects_unpriced_or_ambiguous_budget(config):
    with pytest.raises(ValueError):
        px.normalize_verified_activation_load(config)


def test_cap_refuses_before_opening_or_allocating(capture, monkeypatch):
    path = capture[0]/'inputs/a.pt'
    def refuse(*args, **kwargs):
        pytest.fail('oversized input was opened')
    monkeypatch.setattr(px.os, 'open', refuse)
    with pytest.raises(RuntimeError, match='buffer budget'):
        load(path, policy=policy(buffer=1))


def test_selected_roster_preflight_refuses_before_any_load(capture, monkeypatch):
    root, _, census, identity, _, _, receipt = capture
    (root/'inputs/b.pt').write_bytes(b'x'*(1024**2+1))
    monkeypatch.setattr(px, 'load_verified_activation_cache_entry',
                        lambda *a, **k: pytest.fail('loaded before roster preflight'))
    with pytest.raises(RuntimeError, match='buffer budget'):
        cc.prefetch_capture(receipt['path'], expected_identity=identity, census=census,
            names=['a', 'b'], device='cpu', verified_load_policy=policy())


def test_large_tensor_uses_readinto_and_releases_buffer_before_return(tmp_path, monkeypatch):
    path = tmp_path/'large.pt'
    expected = torch.arange(1024**2, dtype=torch.float32)
    torch.save({'inputs': expected}, path)
    reads, refs, checks = [], [], []
    original = px._VerifiedBufferReader
    class Watched(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            refs.append(weakref.ref(self))
        def readinto(self, target):
            reads.append(len(target))
            return super().readinto(target)
    monkeypatch.setattr(px, '_VerifiedBufferReader', Watched)
    def guard(label, **kwargs):
        checks.append(label)
        if label.startswith('after_verified_capture_buffer_release'):
            assert all(ref() is None for ref in refs)
    value, receipt = load(path, resource_check=guard)
    assert torch.equal(value['inputs'], expected)
    assert max(reads) >= expected.numel()*expected.element_size()
    assert receipt['source_read_bytes'] == path.stat().st_size
    assert receipt['archive_storage_bytes'] == expected.numel()*expected.element_size()
    assert receipt['live_buffer_bytes'] == 0 and refs[0]() is None
    assert checks[-1].startswith('after_verified_capture_buffer_release')


@pytest.mark.parametrize('failure', ['checksum', 'truncate', 'replace_read', 'replace_decode', 'symlink'])
def test_mutated_or_unsealed_source_refuses(capture, monkeypatch, failure):
    path = capture[0]/'inputs/a.pt'
    digest = cc.sha256(path)
    if failure == 'checksum':
        digest = '0'*64
    if failure == 'symlink':
        real = path.with_suffix('.real')
        path.rename(real)
        path.symlink_to(real)
    fired = False
    def guard(label, **kwargs):
        nonlocal fired
        point = 'before_verified_capture_decode' if failure == 'replace_decode' else 'before_verified_capture_read'
        if fired or not label.startswith(point):
            return
        fired = True
        if failure == 'truncate':
            with path.open('r+b') as handle:
                handle.truncate(8)
        elif failure.startswith('replace'):
            replacement = path.with_suffix('.new')
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
    with pytest.raises(RuntimeError, match='checksum|changed|truncated|nonsymlink'):
        load(path, expected_sha256=digest, resource_check=guard)


def test_archive_backing_storage_refuses_before_torch_decode(tmp_path, monkeypatch):
    path = tmp_path/'view.pt'
    torch.save({'inputs': torch.zeros(4096)[:1]}, path)
    monkeypatch.setattr(px.torch, 'load', lambda *a, **k: pytest.fail('decoded overbudget storage'))
    with pytest.raises(RuntimeError, match='backing storage'):
        load(path, max_storage_bytes=4)


def test_alias_backing_storage_is_charged_once(tmp_path):
    path = tmp_path/'alias.pt'
    tensor = torch.arange(32, dtype=torch.float32)
    torch.save({'a': tensor[:4], 'b': tensor[4:8]}, path)
    value, receipt = load(path, max_storage_bytes=128)
    assert value['a'].untyped_storage()._cdata == value['b'].untyped_storage()._cdata
    assert receipt['archive_storage_bytes'] == 128


@pytest.mark.parametrize('fault', ['opaque', 'nonfinite', 'geometry', 'extra', 'stride'])
def test_capture_rejects_noncanonical_payload(capture, fault):
    root, _, census, identity, _, _, receipt = capture
    path = root/'inputs/a.pt'
    value = torch.load(path, weights_only=True)
    if fault == 'opaque': value['max_abs'] = {4.}
    if fault == 'nonfinite': value['inputs'][0, 0] = float('nan')
    if fault == 'geometry': value['inputs'] = value['inputs'].reshape(1, 4)
    if fault == 'extra': value['hidden'] = torch.zeros(1)
    if fault == 'stride': value['inputs'] = value['inputs'].T
    torch.save(value, path)
    manifest = json.loads(Path(receipt['path']).read_text())
    manifest['entries']['a']['sha256'] = cc.sha256(path)
    Path(receipt['path']).write_text(json.dumps(manifest))
    with pytest.raises((RuntimeError, TypeError), match='opaque|nonfinite|geometry|owners|contiguous|backing storage'):
        cc.prefetch_capture(receipt['path'], expected_identity=identity, census=census,
            names=['a'], device='cpu', verified_load_policy=policy())


def test_buffer_released_on_decode_failure(capture, monkeypatch):
    refs = []
    original = px._VerifiedBufferReader
    class Watched(original):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            refs.append(weakref.ref(self))
    monkeypatch.setattr(px, '_VerifiedBufferReader', Watched)
    def fail(reader, **kwargs):
        raise RuntimeError('decode failed')
    monkeypatch.setattr(px.torch, 'load', fail)
    with pytest.raises(RuntimeError, match='decode failed') as failure:
        load(capture[0]/'inputs/a.pt')
    # A retained traceback may retain the adapter, but no raw buffer view.
    assert refs[0]() is None or refs[0]()._view is None


def test_writer_resume_seal_and_prefetch_preserve_artifact_identity(capture):
    root, path, census, identity, acts, hessians, receipt = capture
    original_manifest = Path(receipt['path']).read_bytes()
    hashes = {name: cc.sha256(root/f'inputs/{name}.pt') for name in acts}
    writer = cc.CaptureWriter(root, census_path=path, identity=identity,
                             verified_load_policy=policy())
    writer.write(acts=acts, hessians=hessians, counts=census['counts'], maxima=census['max_abs'])
    result = writer.finish(model_load_contract=identity['model_load_contract'])
    assert Path(result['path']).read_bytes() == original_manifest
    assert hashes == {name: cc.sha256(root/f'inputs/{name}.pt') for name in acts}
    assert writer.load_execution['loaded_entries'] == 2
    assert writer.seal_load_execution['loaded_entries'] == 2
    execution = {}
    values, _ = cc.prefetch_capture(result['path'], expected_identity=identity,
        census=census, names=acts, device='cpu', verified_load_policy=policy(), load_execution=execution)
    assert execution['loaded_entries'] == 2 and execution['live_buffer_bytes'] == 0
    for name in acts:
        assert torch.equal(values[0][name], acts[name])
        assert torch.equal(values[1][name], hessians[name])
    other = cc._load_execution(policy(buffer=2*1024**2), identity)
    assert other['identity_sha256'] != execution['identity_sha256']


def test_regular_file_replaced_by_fifo_does_not_block(capture, monkeypatch):
    path = capture[0]/'inputs/a.pt'
    digest = cc.sha256(path)
    original = os.open
    def replaced(name, flags, *args, **kwargs):
        if Path(name) == path:
            assert flags & os.O_NONBLOCK, 'special-file replacement could block before fstat'
            path.unlink()
            os.mkfifo(path)
        return original(name, flags, *args, **kwargs)
    monkeypatch.setattr(os, 'open', replaced)
    with pytest.raises(RuntimeError, match='changed'):
        # Supply the digest directly; do not hash the FIFO in this test helper.
        px.load_verified_activation_cache_entry(path, expected_sha256=digest,
            policy=policy(), max_storage_bytes=32)


@pytest.mark.parametrize('kind', ['compressed', 'duplicate', 'metadata'])
def test_archive_layout_refuses_before_cpu_decode(tmp_path, monkeypatch, kind):
    path = tmp_path/'invalid.pt'
    with zipfile.ZipFile(path, 'w', compression=(zipfile.ZIP_DEFLATED
            if kind == 'compressed' else zipfile.ZIP_STORED)) as archive:
        archive.writestr('archive/data.pkl', b'x'*(20000 if kind == 'metadata' else 2))
        archive.writestr('archive/data/0', b'1234')
        if kind == 'duplicate':
            archive.writestr('archive/data/0', b'1234')
    monkeypatch.setattr(px.torch, 'load', lambda *a, **k: pytest.fail('decoded invalid archive'))
    with pytest.raises(RuntimeError, match='archive|metadata'):
        load(path)


def test_many_empty_zip_entries_refuse_before_zipinfo_construction(tmp_path, monkeypatch):
    path = tmp_path/'many.pt'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('archive/data.pkl', b'00')
        for index in range(1000):
            archive.writestr(f'archive/data/{index}', b'')
    # Constructing even the directory objects must wait for scratch admission.
    monkeypatch.setattr(zipfile.ZipFile, '_RealGetContents',
        lambda *a, **k: pytest.fail('ZipInfo construction preceded metadata admission'))
    with pytest.raises(RuntimeError, match='directory.*scratch'):
        load(path)


def test_forged_small_eocd_count_cannot_hide_many_entries(tmp_path, monkeypatch):
    import struct
    path = tmp_path/'lying-directory.pt'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('archive/data.pkl', b'00')
        for index in range(40):
            archive.writestr(f'archive/data/{index}', b'')
    raw = bytearray(path.read_bytes())
    location = raw.rfind(zipfile.stringEndArchive)
    struct.pack_into('<HH', raw, location+8, 1, 1)
    path.write_bytes(raw)
    monkeypatch.setattr(zipfile.ZipFile, '_RealGetContents',
        lambda *a, **k: pytest.fail('forged count reached ZipInfo allocation'))
    with pytest.raises(RuntimeError, match='directory.*scratch|directory count'):
        load(path)


def test_declared_pickle_storage_refuses_before_any_torch_allocation(tmp_path, monkeypatch):
    import pickletools
    import struct
    path = tmp_path/'declared-storage.pt'
    torch.save({'inputs': torch.ones(1)}, path)
    with zipfile.ZipFile(path) as archive:
        entries = {entry.filename: archive.read(entry) for entry in archive.infolist()}
    name = next(name for name in entries if name.endswith('/data.pkl'))
    raw = entries[name]
    operations = list(pickletools.genops(raw))
    persistent = next(i for i, (op, _, _) in enumerate(operations) if op.name == 'BINPERSID')
    numeric = max(i for i in range(persistent) if operations[i][0].name in ('BININT','BININT1','BININT2'))
    assert operations[numeric][1] == 1
    start, stop = operations[numeric][2], operations[numeric+1][2]
    entries[name] = raw[:start]+b'J'+struct.pack('<i', 1024**3)+raw[stop:]
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    monkeypatch.setattr(px.torch, 'load',
        lambda *a, **k: pytest.fail('oversized pickle declaration reached Torch allocation'))
    with pytest.raises(RuntimeError, match='declared pickle storage'):
        load(path, max_storage_bytes=4)
