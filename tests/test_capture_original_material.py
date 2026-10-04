"""Internal original-material qualification through real PB leases, CPU only.

This is not automatic campaign admission or a GLM producer transition.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prismaquant import tessera_calibration_cache as cc
from prismaquant.residency_map import bind_residency_manifest
from test_stage_b_prep_staged_reads_1092 import _stage_manifest
from test_strict_reader_tier_enforcement import MANIFEST, _forget_state  # noqa: F401

pytestmark = pytest.mark.own_process
REVISION = 'a6' * 20


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bound(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {'path': str(path), 'sha256': _sha(raw)}


@pytest.fixture
def material(tmp_path, monkeypatch, request):
    root = tmp_path / 'logical'
    root.mkdir()
    delivered = tmp_path / 'publisher-exact'
    delivered.mkdir()
    (delivered / 'config.json').write_text('{"model_type":"fixture"}')
    (delivered / 'chat_template.jinja').write_text('original template')
    source_dtype = getattr(request, 'param', torch.float32)
    assert source_dtype in (torch.float32, torch.bfloat16)
    for name, key in [('one.safetensors', 'w'), ('two.safetensors', 'v')]:
        save_file({key: torch.arange(32, dtype=source_dtype).reshape(4, 8)},
                  str(delivered / name))
    weight_map = {'w': 'one.safetensors', 'v': 'two.safetensors'}
    (delivered / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
    names = sorted(p.name for p in delivered.iterdir())
    paths = {name: str(delivered / name) for name in names}
    raws = {name: Path(paths[name]).read_bytes() for name in names}
    siblings = []
    for name in names:
        raw = raws[name]
        row = {'rfilename': name, 'size': len(raw),
               'blobId': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()}
        if name.endswith('.safetensors'):
            row['lfs'] = {'sha256': _sha(raw), 'size': len(raw)}
        siblings.append(row)
    publisher = {'id': 'fixture/source', 'sha': REVISION, 'siblings': siblings}
    entries = [{'path': paths[name], 'offset': 0, 'bytes': len(raws[name]), 'sha256': _sha(raws[name])}
               for name in names]
    readset = {'schema': 'prismaquant.prismabuild.data_manifest.v1',
               'produced_by': {'fixture': True}, 'mount_prefix': str(tmp_path),
               'entries': entries, 'entry_count': len(entries),
               'total_bytes': sum(row['bytes'] for row in entries), 'annotations': {}}
    producer = {'files': {name: _sha(raws[name]) for name in names if name.endswith('.safetensors')},
                'auxiliary_sha256': {'chat_template.jinja': _sha(raws['chat_template.jinja']),
                                     'model.safetensors.index.json': _sha(raws['model.safetensors.index.json'])},
                'config_sha256': _sha(raws['config.json']), 'tensors': weight_map}
    pub = _bound(tmp_path / 'publisher.json', publisher)
    reads = _bound(tmp_path / 'readset.json', readset)
    staged = _stage_manifest(tmp_path, monkeypatch, readset)
    bind_residency_manifest(MANIFEST)
    checks = []
    def admit(label, **kwargs):
        checks.append((label, kwargs))
    options = dict(publisher_input=pub, publisher_id='fixture/source', publisher_revision=REVISION,
                   readset_input=reads, source_paths=paths,
                   max_material_bytes=max(len(raw) for raw in raws.values()), resource_check=admit)
    return dict(root=root, paths=paths, raws=raws, publisher=publisher, readset=readset,
                producer=producer, options=options, checks=checks, staged=staged, tmp=tmp_path)


def _owner(m):
    return cc.CaptureSourceAuthentication.qualified_original_material(
        m['root'], m['producer'], **m['options'])


@pytest.mark.parametrize('broken', ['publisher-digest', 'revision', 'partial-readset', 'range',
                                   'producer-template', 'producer-weight', 'producer-index', 'native-aux-object'])
def test_bad_authority_refuses_before_model_or_payload_bootstrap(material, monkeypatch, broken):
    m = material
    if broken == 'publisher-digest':
        m['options']['publisher_input']['sha256'] = '0' * 64
    elif broken == 'revision':
        m['options']['publisher_revision'] = 'b' * 40
    elif broken in ('partial-readset', 'range'):
        readset = m['readset']
        if broken == 'partial-readset':
            readset['entries'].pop()
        else:
            readset['entries'][0]['offset'] = 1
        readset['entry_count'] = len(readset['entries'])
        readset['total_bytes'] = sum(row['bytes'] for row in readset['entries'])
        m['options']['readset_input'] = _bound(m['tmp'] / 'bad-readset.json', readset)
    elif broken == 'producer-index':
        m['producer']['tensors']['w'] = 'two.safetensors'
    elif broken.startswith('producer'):
        which = 'auxiliary_sha256' if broken == 'producer-template' else 'files'
        name = 'chat_template.jinja' if broken == 'producer-template' else 'one.safetensors'
        m['producer'][which][name] = '0' * 64
    else:
        m['publisher']['siblings'][0]['blobId'] = '0' * 40
        m['options']['publisher_input'] = _bound(m['tmp'] / 'bad-publisher.json', m['publisher'])
    decodes = []
    read_json = cc.CaptureSourceAuthentication.read_json
    def index_only(owner, path):
        name = Path(path).name
        decodes.append(name)
        if name != 'model.safetensors.index.json':
            raise AssertionError('model bootstrap reached before complete index qualification')
        return read_json(owner, path)
    monkeypatch.setattr(cc.CaptureSourceAuthentication, 'read_json', index_only)
    with pytest.raises((RuntimeError, ValueError)):
        _owner(m)
    # The independently authenticated index must be interpreted to compare
    # its tensor map. No other authority refusal permits even that step, and
    # a mismatched producer map never reaches model/config bootstrap.
    assert decodes == (['model.safetensors.index.json'] if broken == 'producer-index' else [])


def test_same_sealed_object_serves_header_json_and_tensors_despite_pool_mutation(material):
    m = material
    owner = _owner(m)
    with owner.material_window([m['root'] / 'one.safetensors', m['root'] / 'one.safetensors']):
        with owner.safe_open(safe_open, m['root'] / 'one.safetensors', framework='pt') as reader:
            path = owner.descriptor_path(m['root'] / 'one.safetensors')
            assert os.readlink(path).startswith('/memfd:')
            Path(m['paths']['one.safetensors']).write_bytes(b'x' * len(m['raws']['one.safetensors']))
            Path(m['staged'][m['paths']['one.safetensors']]).write_bytes(b'y' * len(m['raws']['one.safetensors']))
            assert reader.keys() == ['w']
            value = reader.get_tensor('w')
            torch.testing.assert_close(value, torch.arange(32, dtype=torch.float32).reshape(4, 8))
            assert owner.descriptor_path(m['root'] / 'one.safetensors') == path
        del value
    with owner.material_window([m['root'] / 'config.json']):
        Path(m['paths']['config.json']).write_text('bad mutable config')
        assert owner.read_json(m['root'] / 'config.json') == {'model_type': 'fixture'}
    assert owner.material_live_bytes == 0
    owner.close()


def test_kernel_seals_refuse_write_grow_and_shrink(material):
    import errno
    import fcntl
    from prismaquant.io_engine import _SEALS
    m = material
    with _owner(m) as owner:
        with owner.material_window([m['root'] / 'one.safetensors']):
            path = owner.descriptor_path(m['root'] / 'one.safetensors')
            fd = os.open(path, os.O_RDWR)
            try:
                assert fcntl.fcntl(fd, fcntl.F_GET_SEALS) & _SEALS == _SEALS
                for mutate in (lambda: os.pwrite(fd, b'x', 0),
                               lambda: os.ftruncate(fd, 1),
                               lambda: os.ftruncate(fd, len(m['raws']['one.safetensors']) + 1)):
                    with pytest.raises(OSError) as error:
                        mutate()
                    assert error.value.errno == errno.EPERM
            finally:
                os.close(fd)


@pytest.mark.parametrize('failure', ['wrong-digest', 'truncated', 'cancelled', 'torn-copy'])
def test_failed_material_never_reaches_header_factory_and_cleans_charge(material, monkeypatch, failure):
    from prismaquant import io_engine
    m = material
    owner = _owner(m)
    stage = Path(m['staged'][m['paths']['one.safetensors']])
    if failure in ('wrong-digest', 'truncated'):
        stage.write_bytes((b'x' * len(m['raws']['one.safetensors'])) if failure == 'wrong-digest' else b'x')
    else:
        real = io_engine.SealedBuffer.fill
        def fill(buffer, fd):
            if failure == 'cancelled':
                raise KeyboardInterrupt('fixture cancellation')
            complete = real(buffer, fd)
            # Corrupt owned delivery after copying, preserving source metadata.
            buffer.fill_bytes(b'x' * buffer.size)
            return complete
        monkeypatch.setattr(io_engine.SealedBuffer, 'fill', fill)
    with pytest.raises(BaseException):
        with owner.material_window([m['root'] / 'one.safetensors']):
            pytest.fail('unverified material became decoder-visible')
    assert owner.material_live_bytes == 0
    assert owner.receipt()['verified_files'] == []
    owner.close()


def test_native_storage_alias_outlives_reader_and_holds_charge_until_last_consumer(material):
    m = material
    owner = _owner(m)
    with owner.material_window([m['root'] / 'one.safetensors']):
        with owner.safe_open(safe_open, m['root'] / 'one.safetensors', framework='pt') as reader:
            tensor = reader.get_tensor('w')
            alias = tensor.detach()[1:]
        del tensor
    gc.collect()
    assert owner.material_live_bytes == len(m['raws']['one.safetensors'])
    with pytest.raises(RuntimeError, match='live.*consumer'):
        owner.close()
    with pytest.raises(RuntimeError, match='material.*bound'):
        with owner.material_window([m['root'] / 'two.safetensors']):
            pass
    torch.testing.assert_close(alias, torch.arange(32, dtype=torch.float32).reshape(4, 8)[1:])
    del alias
    gc.collect()
    assert owner.material_live_bytes == 0
    owner.close()


def test_slice_lifetime_and_finite_peak_reservations(material):
    m = material
    with _owner(m) as owner:
        with owner.material_window([m['root'] / 'one.safetensors']):
            with owner.safe_open(safe_open, m['root'] / 'one.safetensors', framework='pt') as reader:
                slab = reader.get_slice('w')
                assert slab.get_shape() == [4, 8]
                alias = slab[1:]
            with pytest.raises(RuntimeError, match='outside its read lease'):
                slab[0]
        assert owner.material_live_bytes > 0
        del alias
        gc.collect()
        assert owner.material_live_bytes == 0
    reserves = [kw['reserve_bytes'] for label, kw in m['checks'] if 'original_material' in label]
    assert reserves and max(reserves) <= m['options']['max_material_bytes']


@pytest.mark.parametrize('route', ['outside-window', 'range-factory', 'gpu', 'dynamic-config'])
def test_unsupported_routes_remain_closed(material, monkeypatch, route):
    m = material
    if route == 'dynamic-config':
        raw = b'{"auto_map":{"AutoConfig":"custom.Config"}}'
        Path(m['paths']['config.json']).write_bytes(raw)
        # Fully authenticate a publisher-native dynamic config, then refuse
        # its unsupported bootstrap route rather than executing custom code.
        digest = _sha(raw)
        for row in m['publisher']['siblings']:
            if row['rfilename'] == 'config.json':
                row['size'] = len(raw)
                row['blobId'] = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        for row in m['readset']['entries']:
            if row['path'] == m['paths']['config.json']:
                row.update(bytes=len(raw), sha256=digest)
        m['readset']['total_bytes'] = sum(row['bytes'] for row in m['readset']['entries'])
        m['producer']['config_sha256'] = digest
        m['options']['publisher_input'] = _bound(m['tmp'] / 'dynamic-publisher.json', m['publisher'])
        m['options']['readset_input'] = _bound(m['tmp'] / 'dynamic-readset.json', m['readset'])
        stage_root = m['tmp'] / 'dynamic-stage'
        stage_root.mkdir()
        _stage_manifest(stage_root, monkeypatch, m['readset'])
        bind_residency_manifest(MANIFEST)
        with pytest.raises(RuntimeError, match='unsupported dynamic'):
            _owner(m)
        return
    with _owner(m) as owner:
        if route == 'outside-window':
            with pytest.raises(RuntimeError, match='material window'):
                owner.read_json(m['root'] / 'config.json')
        else:
            with owner.material_window([m['root'] / 'one.safetensors']):
                with pytest.raises(RuntimeError, match='unsupported'):
                    owner.safe_open((lambda *_a, **_k: None) if route == 'range-factory' else safe_open,
                                    m['root'] / 'one.safetensors', framework='pt',
                                    **({'device': 'cuda:0'} if route == 'gpu' else {}))


def test_internal_material_does_not_admit_automatic_campaigns(material):
    with _owner(material):
        with pytest.raises(RuntimeError, match='qualified immutable source'):
            cc.require_automatic_capture_source_recording()


def test_concurrent_overlapping_windows_share_material_until_both_release(material):
    import threading
    m = material
    owner = _owner(m)
    entered = threading.Event()
    release = threading.Event()
    paths = []
    errors = []
    def borrow():
        try:
            with owner.material_window([m['root'] / 'one.safetensors']):
                paths.append(owner.descriptor_path(m['root'] / 'one.safetensors'))
                entered.set()
                assert release.wait(10)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=borrow)
    thread.start()
    try:
        assert entered.wait(10)
        with owner.material_window([m['root'] / 'one.safetensors']):
            assert owner.descriptor_path(m['root'] / 'one.safetensors') == paths[0]
        assert owner.material_live_bytes == len(m['raws']['one.safetensors'])
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and errors == []
    assert owner.material_live_bytes == 0
    owner.close()


@pytest.mark.parametrize('cancel', [False, True])
def test_accounting_failure_closes_sealed_material_without_delivery(material, monkeypatch, cancel):
    from prismaquant import io_engine
    from prismaquant.residency_map import residency_resolver

    m = material
    owner = _owner(m)
    buffers = []
    original = io_engine.SealedBuffer.__init__
    def allocated(buffer, size):
        original(buffer, size)
        buffers.append(buffer)
    monkeypatch.setattr(io_engine.SealedBuffer, '__init__', allocated)
    def accounting(*args):
        if cancel:
            raise KeyboardInterrupt('accounting cancellation')
        raise RuntimeError('accounting failure')
    monkeypatch.setattr(residency_resolver(), 'record_stage_read', accounting)
    with pytest.raises(KeyboardInterrupt if cancel else RuntimeError, match='accounting'):
        with owner.material_window([m['root'] / 'one.safetensors']):
            pytest.fail('accounting failure returned accepted material')
    assert len(buffers) == 1
    with pytest.raises(RuntimeError, match='closed'):
        _ = buffers[0].path
    assert owner.material_live_bytes == 0
    owner.close()


def test_peak_admission_callback_cannot_reap_material_the_new_window_reuses(material):
    m = material
    owner = _owner(m)
    aliases = []
    with owner.material_window([m['root'] / 'one.safetensors']):
        original_inode = owner.file_stat(m['root'] / 'one.safetensors').st_ino
        with owner.safe_open(safe_open, m['root'] / 'one.safetensors', framework='pt') as reader:
            aliases.append(reader.get_tensor('w').detach())
    observed = []
    def admit(label, **kwargs):
        # A shared guard callback may release/reap other owners while checking.
        aliases.clear()
        gc.collect()
        observed.append((owner.material_live_bytes, kwargs['reserve_bytes']))
    owner.resource_check = admit
    with owner.material_window([m['root'] / 'one.safetensors']):
        assert owner.file_stat(m['root'] / 'one.safetensors').st_ino == original_inode
        assert observed == [(len(m['raws']['one.safetensors']), 0)]
    assert owner.material_live_bytes == 0
    owner.close()


def test_control_validation_requires_generation_bound_client_before_original_material(material, monkeypatch):
    from prismaquant import staged_lease, staged_whole_file

    def unavailable():
        raise RuntimeError('generation-bound control client unavailable')
    def original_read(*args, **kwargs):
        pytest.fail('original material read before control-client qualification')
    monkeypatch.setattr(staged_lease, 'client_sdk', unavailable)
    monkeypatch.setattr(staged_whole_file, 'read_staged_sealed_file', original_read)
    with pytest.raises(RuntimeError, match='generation-bound control client'):
        _owner(material)
