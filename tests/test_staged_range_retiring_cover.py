"""A retiring overlap cannot hide another independently leased cover (#916)."""
import hashlib
import json
import errno
import os

import pytest
import torch

from prismaquant import layer_streaming
from prismaquant.residency_map import TIERS_DIR_ENV_VAR, residency_map_key
from prismaquant.staged_lease import LeaseRefused, acquire_entry_window

from test_strict_reader_tier_enforcement import (
    EPOCH, MANIFEST, _announce, _forget_state, _header_spans, _hex64,
    _launch_env, _pb, _pb_publish, _pb_publish_ram, _pb_queue, _pins_live,
    _shard, _stage_whole, _strict, _write_map,
)


@pytest.fixture
def overlap(tmp_path, monkeypatch):
    sdk, pool_mod, map_mod = _pb()
    declared, tensors = _shard(tmp_path)
    consumer, head, own = (_hex64(f'{label}-{tmp_path}')
                           for label in ('consumer', 'head', 'own'))
    queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
    root = tmp_path / 'residency'
    head_file = _stage_whole(stage, declared)
    start, end = _header_spans(declared)['f32']
    own_file = stage / 'own-range'
    own_file.write_bytes(declared.read_bytes()[start:end])
    head_key = residency_map_key(str(declared), 0)
    own_key = residency_map_key(str(declared), start)
    head_generation = _pb_publish(
        sdk, map_mod, root, stage, consumer, head, MANIFEST,
        {head_key: (declared, head_file)})
    own_generation = _pb_publish(
        sdk, map_mod, root, stage, consumer, own, MANIFEST,
        {own_key: (declared, own_file)})
    map_path = _write_map(tmp_path, {'head': (declared, head_file, None)},
                          leads=[head, own], stage_root=stage)
    body = json.loads(map_path.read_text())
    body['entries'][own_key] = {
        'stage_path': str(own_file), 'offset': start, 'bytes': end - start,
        'sha256': hashlib.sha256(own_file.read_bytes()).hexdigest()}
    map_path.write_text(json.dumps(body))
    _launch_env(monkeypatch, consumer)
    resolver = _strict(monkeypatch, map_path)
    assert resolver.staged_range(declared, start, end)['offset'] == 0
    # A real admitted reader keeps the head material alive while its phase
    # retires. The later reader must not gain a new pin on this generation.
    held, key = acquire_entry_window(resolver, declared, body['entries'][head_key])
    held.__enter__()
    held.open(key)
    sdk.write_retiring(root / 'leases', consumer_action_key=consumer,
                       mover_action_key=head, generation=head_generation)
    try:
        yield dict(sdk=sdk, queue=queue, root=root, consumer=consumer,
                   declared=declared, tensors=tensors, resolver=resolver,
                   own=own, own_generation=own_generation, own_file=own_file,
                   head=head, head_file=head_file, start=start, end=end,
                   map_path=map_path, head_key=head_key, own_key=own_key)
    finally:
        held.__exit__(None, None, None)
        assert _pins_live(tmp_path, consumer) == []


def test_retiring_head_uses_live_overlap_with_its_own_pin(overlap, tmp_path):
    ctx = overlap
    with layer_streaming._source_safe_open(str(ctx['declared']), framework='pt') as reader:
        got = reader.get_tensor('f32')
        assert torch.equal(got.view(torch.uint8), ctx['tensors']['f32'].view(torch.uint8))
        assert reader._bound[0][0] == ctx['start']
        assert len(_pins_live(tmp_path, ctx['consumer'])) == 2
        # Closing to NEW pins does not end the already admitted descriptor.
        ctx['sdk'].write_retiring(
            ctx['root'] / 'leases', consumer_action_key=ctx['consumer'],
            mover_action_key=ctx['own'], generation=ctx['own_generation'])
        assert torch.equal(reader.get_tensor('f32'), got)
        assert len(_pins_live(tmp_path, ctx['consumer'])) == 2
    assert len(_pins_live(tmp_path, ctx['consumer'])) == 1
    report = ctx['resolver'].report()
    assert report['bytes_from_pool'] == 0
    assert report['serving_tier_count'] == 1


@pytest.mark.parametrize('own_state', ['absent', 'retiring'])
def test_no_admissible_overlap_refuses_without_pool_reads(overlap, tmp_path, own_state):
    ctx = overlap
    if own_state == 'absent':
        ctx['own_file'].unlink()
    else:
        ctx['sdk'].write_retiring(
            ctx['root'] / 'leases', consumer_action_key=ctx['consumer'],
            mover_action_key=ctx['own'], generation=ctx['own_generation'])
    with layer_streaming._source_safe_open(str(ctx['declared']), framework='pt') as reader:
        with pytest.raises(LeaseRefused, match='retiring'):
            reader.get_tensor('f32')
        assert reader._bound == []
    assert len(_pins_live(tmp_path, ctx['consumer'])) == 1
    assert ctx['resolver'].report()['bytes_from_pool'] == 0
    assert ctx['resolver'].report()['serving_tier_count'] == 0


def test_uncertain_head_authority_is_not_bypassed_by_a_good_overlap(overlap):
    ctx = overlap
    marker = ctx['root'] / 'leases' / ctx['consumer'] / (ctx['head'] + '.retiring.json')
    marker.write_text('{')
    with layer_streaming._source_safe_open(str(ctx['declared']), framework='pt') as reader:
        with pytest.raises(LeaseRefused, match='ownership-uncertain'):
            reader.get_tensor('f32')
    assert ctx['resolver'].report()['bytes_from_pool'] == 0
    assert ctx['resolver'].report()['serving_tier_count'] == 0
    # Restore the mark before the existing head reader's exact release.
    marker.unlink()


def test_an_alternate_with_changed_identity_is_refused(overlap, tmp_path):
    ctx = overlap
    # Same bytes and length, different inode: pre-open shape still passes;
    # PB's real generation/identity gate must refuse this candidate.
    replacement = ctx['own_file'].with_suffix('.replacement')
    replacement.write_bytes(ctx['own_file'].read_bytes())
    replacement.replace(ctx['own_file'])
    with layer_streaming._source_safe_open(str(ctx['declared']), framework='pt') as reader:
        with pytest.raises(LeaseRefused) as refused:
            reader.get_tensor('f32')
        assert refused.value.kind == 'integrity'
        assert reader._bound == []
    assert len(_pins_live(tmp_path, ctx['consumer'])) == 1
    assert ctx['resolver'].report()['bytes_from_pool'] == 0


@pytest.mark.parametrize('damage', ['size', 'nonregular', 'permission'])
def test_an_alternate_with_hard_file_failure_stays_integrity_refused(
        overlap, tmp_path, monkeypatch, damage):
    ctx = overlap
    path = ctx['own_file']
    if damage == 'size':
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == 'nonregular':
        path.unlink()
        path.mkdir()
    else:
        real_lstat = os.lstat

        def denied(target, *args, **kwargs):
            if os.fspath(target) == str(path):
                raise OSError(errno.EACCES, os.strerror(errno.EACCES))
            return real_lstat(target, *args, **kwargs)

        monkeypatch.setattr(os, 'lstat', denied)
    with layer_streaming._source_safe_open(str(ctx['declared']), framework='pt') as reader:
        with pytest.raises(LeaseRefused) as refused:
            reader.get_tensor('f32')
        assert refused.value.kind == 'integrity'
        assert reader._bound == []
    assert len(_pins_live(tmp_path, ctx['consumer'])) == 1
    assert ctx['resolver'].report()['bytes_from_pool'] == 0


def test_live_ram_overlap_uses_its_own_material_and_pin(overlap, tmp_path, monkeypatch):
    ctx = overlap
    from prismabuild import residency_map as map_mod
    ram_root = tmp_path / 'ram'
    ram_root.mkdir()
    ram_file = ram_root / 'own-range'
    ram_file.write_bytes(ctx['own_file'].read_bytes())
    ram_mover = _hex64(f'ram-{tmp_path}')
    _pb_publish_ram(
        ctx['sdk'], map_mod, ctx['root'], ram_root, ctx['consumer'],
        ram_mover, MANIFEST, {ctx['own_key']: (ctx['declared'], ram_file)}, EPOCH)
    tiers = _announce(tmp_path)
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tiers))
    body = json.loads(ctx['map_path'].read_text())
    body['entries'][ctx['own_key']]['ram_path'] = str(ram_file)
    body.update(ram_tier_id='ram:dl380g10', ram_root=str(ram_root), ram_epoch=EPOCH)
    ctx['map_path'].write_text(json.dumps(body))
    resolver = _strict(monkeypatch, ctx['map_path'])
    with layer_streaming._source_safe_open(str(ctx['declared']), framework='pt') as reader:
        assert torch.equal(reader.get_tensor('f32'), ctx['tensors']['f32'])
        assert len(_pins_live(tmp_path, ctx['consumer'])) == 2
    assert len(_pins_live(tmp_path, ctx['consumer'])) == 1
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert report['serving_tiers'][0]['serving_tier'] == 'ram'
