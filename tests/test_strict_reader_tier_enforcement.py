"""Strict allowed-tier enforcement for GPU-consumed bulk inputs (PQ #845).

The staged-read contract (INV-03/SAFE-02) forbids bulk-input reads from the
pool/HDD tier: RAM first, SSD only when the sealed declaration permits.
These tests run the REAL reader chain — real resolver, real map/stage/ram
files, real tier records — on tiny fixtures, never giant payloads:

- the legacy test proves the hole (inactive policy serves pool bytes);
- the strict tests prove the gate (permitted tiers serve byte-equal with
  serving-tier records; forbidden opens refuse before payload bytes with
  zero pool bytes).

No PB lease stub is invented here: RNG-02/SM-03 remain PB-owned gaps, and
every test below opens through the actual PQ reader with default flags.
"""
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import layer_streaming
from prismaquant.residency_map import (
    ENV_VAR, SCHEMA, TIERS_DIR_ENV_VAR, bind_residency_manifest,
    residency_map_key, residency_resolver, reset_residency_resolver_for_tests,
)
from prismaquant.staged_tier_policy import (
    DEFAULT_ALLOWED_TIERS, TierPolicyRefused, activate_staged_tier_policy,
    active_policy, deactivate_staged_tier_policy_for_tests,
    parse_allowed_tiers, staged_tier_policy_context,
)

MANIFEST = 'e' * 64
LEAD = 'f' * 64
RAM_TIER = 'ram:dl380g10'
EPOCH = '1789771929-aba6e46e41fb03ef'
STALE_EPOCH = '1789788888-9c1d2e3f4a5b'


@pytest.fixture(autouse=True)
def _forget_state(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(TIERS_DIR_ENV_VAR, raising=False)
    reset_residency_resolver_for_tests()
    deactivate_staged_tier_policy_for_tests()
    yield
    reset_residency_resolver_for_tests()
    deactivate_staged_tier_policy_for_tests()


# -- fixtures ---------------------------------------------------------------

def _shard(tmp_path):
    pool = tmp_path / 'pool'
    pool.mkdir(parents=True, exist_ok=True)
    tensors = {
        'f32': torch.linspace(-3, 3, 256, dtype=torch.float32).reshape(8, 32),
        'bf16': (torch.arange(512, dtype=torch.int32).reshape(16, 32)
                 .to(torch.bfloat16)),
        'empty': torch.zeros((0, 4), dtype=torch.float32),
    }
    path = pool / 'model-00001-of-00002.safetensors'
    save_file(tensors, str(path))
    return path, tensors


def _header_spans(path):
    raw = path.read_bytes()
    size = int.from_bytes(raw[:8], 'little')
    body = json.loads(raw[8:8 + size])
    base = 8 + size
    return {name: (base + row['data_offsets'][0], base + row['data_offsets'][1])
            for name, row in body.items() if name != '__metadata__'}


def _stage_root(tmp_path):
    root = tmp_path / 'stage' / 'prewarm'
    root.mkdir(parents=True, exist_ok=True)
    return root


def _stage_whole(root, path):
    target = root / path.name
    target.write_bytes(path.read_bytes())
    return target


def _promote_ram(tmp_path, staged):
    root = tmp_path / 'ram' / 'prewarm'
    root.mkdir(parents=True, exist_ok=True)
    ram = {}
    for key, stage in staged.items():
        target = root / stage.name
        target.write_bytes(stage.read_bytes())
        ram[key] = target
    return root, ram


def _announce(tmp_path, epoch=EPOCH):
    directory = tmp_path / 'tiers'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{RAM_TIER}.json').write_text(json.dumps(
        {'schema': 'prismabuild.storage_tier.v1', 'tier': 'ram',
         'tier_id': RAM_TIER, 'mountpoint': '/ram/prewarm', 'epoch': epoch}))
    return directory


def _write_map(tmp_path, rows, *, name='residency.json', manifest_sha256=MANIFEST,
               ram_root=None, epoch=EPOCH):
    """``rows``: {key: (declared Path, staged Path, ram Path|None)} whole-file."""
    entries = {}
    for key, (declared, staged, ram_path) in rows.items():
        entry = {
            'stage_path': str(staged),
            'bytes': declared.stat().st_size,
            'offset': 0,
            'sha256': hashlib.sha256(declared.read_bytes()).hexdigest(),
        }
        if ram_path is not None:
            entry['ram_path'] = str(ram_path)
        entries[residency_map_key(str(declared), 0)] = entry
    body = {'schema': SCHEMA, 'tier_id': 'prismabuild-stage:dl380g10',
            'stage_root': str(staged.parent),
            'manifest_sha256': manifest_sha256, 'leads': [LEAD],
            'generation': 3, 'entries': entries}
    if ram_root is not None:
        body.update({'ram_tier_id': RAM_TIER, 'ram_root': str(ram_root),
                     'ram_epoch': epoch})
    path = tmp_path / 'residency' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


def _bind(monkeypatch, map_path, digest=MANIFEST):
    monkeypatch.setenv(ENV_VAR, str(map_path))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(digest)
    return residency_resolver()


# -- policy unit ------------------------------------------------------------

def test_parse_allowed_tiers_grammar():
    assert parse_allowed_tiers("ram,ssd") == frozenset({"ram", "ssd"})
    assert parse_allowed_tiers("ram") == frozenset({"ram"})
    assert parse_allowed_tiers(" RAM , SSD ") == frozenset({"ram", "ssd"})
    assert parse_allowed_tiers(DEFAULT_ALLOWED_TIERS) == frozenset({"ram", "ssd"})
    for bad in ("", "pool", "hdd", "ram,pool", "ssd,hdd", "nvme", 42, None):
        with pytest.raises(ValueError):
            parse_allowed_tiers(bad)


def test_policy_context_lifetime_is_explicit_and_thread_global():
    assert active_policy() is None
    seen = {}
    with staged_tier_policy_context("ram,ssd") as allowed:
        assert allowed == frozenset({"ram", "ssd"})
        worker = threading.Thread(
            target=lambda: seen.setdefault("active", active_policy()))
        worker.start()
        worker.join()
        # Prefetch threads inherit no context: the process-global cell is
        # what reaches them deterministically (never ContextVar).
        assert seen["active"] == frozenset({"ram", "ssd"})
        with pytest.raises(TierPolicyRefused):
            raise TierPolicyRefused("sentinel")
    assert active_policy() is None
    assert seen["active"] == frozenset({"ram", "ssd"})
    # Error paths restore too.
    with pytest.raises(RuntimeError, match="boom"):
        with staged_tier_policy_context("ram"):
            raise RuntimeError("boom")
    assert active_policy() is None


def test_nested_context_restores_outer_enforcement():
    """An inner context never clears or permanently weakens the outer one."""
    with staged_tier_policy_context("ram"):
        assert active_policy() == frozenset({"ram"})
        with staged_tier_policy_context("ram,ssd"):
            assert active_policy() == frozenset({"ram", "ssd"})
        assert active_policy() == frozenset({"ram"})
    assert active_policy() is None
    # An explicitly activated outer verdict survives a nested test scope.
    activate_staged_tier_policy("ram")
    try:
        with staged_tier_policy_context("ram,ssd"):
            assert active_policy() == frozenset({"ram", "ssd"})
        assert active_policy() == frozenset({"ram"})
    finally:
        deactivate_staged_tier_policy_for_tests()
    assert active_policy() is None


def test_activation_needs_an_explicit_sealed_value():
    with pytest.raises(TypeError):
        activate_staged_tier_policy()


# -- the legacy hole (RED evidence; passes before and after) ----------------

def test_legacy_unmapped_source_serves_pool_bytes(tmp_path, monkeypatch):
    """Pre-fix behavior, preserved for offline scope: no map, pool serves."""
    path, tensors = _shard(tmp_path)
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        from safetensors import safe_open
        with safe_open(str(path), framework='pt') as reference:
            for name in sorted(tensors):
                if tensors[name].numel():
                    assert torch.equal(
                        reader.get_tensor(name).view(torch.uint8),
                        reference.get_tensor(name).view(torch.uint8))


# -- strict source shard ----------------------------------------------------

def _strict(monkeypatch, map_path, tiers="ram,ssd"):
    resolver = _bind(monkeypatch, map_path)
    activate_staged_tier_policy(tiers)
    return resolver


def test_strict_source_stage_serves_with_serving_tier(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    spans = _header_spans(path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _strict(monkeypatch, _write_map(tmp_path, {'s': (path, staged, None)}))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        assert reader._handle is None  # no pool safe_open/mmap for header
        assert sorted(reader.keys()) == sorted(tensors)
        assert reader.metadata() == {}
        from safetensors import safe_open
        with safe_open(str(path), framework='pt') as reference:
            for name in sorted(tensors):
                if tensors[name].numel():
                    assert torch.equal(
                        reader.get_tensor(name).view(torch.uint8),
                        reference.get_tensor(name).view(torch.uint8))
    report = resolver.report()
    served = [n for n in tensors if spans[n][1] > spans[n][0]]
    assert report['bytes_from_pool'] == 0
    assert report['range_hits'] == len(served)
    # One bound range opened once serves every covered tensor: the serving
    # tier is recorded at open (ID-07), not per tensor.
    assert report['serving_tier_count'] == 1
    assert {row['serving_tier'] for row in report['serving_tiers']} == {'stage'}
    assert all(row['lease_id'] is None for row in report['serving_tiers'])


def test_strict_source_ram_serves_first(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    spans = _header_spans(path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    ram_root, ram = _promote_ram(tmp_path, {'s': staged})
    _announce(tmp_path)
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'s': (path, staged, ram['s'])}, ram_root=ram_root))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        from safetensors import safe_open
        with safe_open(str(path), framework='pt') as reference:
            got = reader.get_tensor('f32')
            assert torch.equal(got.view(torch.uint8),
                               reference.get_tensor('f32').view(torch.uint8))
    report = resolver.report()
    span = spans['f32'][1] - spans['f32'][0]
    assert report['bytes_from_ram'] == span
    assert report['bytes_from_stage'] == 0
    assert report['bytes_from_pool'] == 0
    assert report['serving_tiers'][-1]['serving_tier'] == 'ram'


def test_strict_source_unmapped_refuses_without_pool_bytes(tmp_path, monkeypatch):
    path, _ = _shard(tmp_path)
    other = path.with_name('model-00002-of-00002.safetensors')
    other.write_bytes(path.read_bytes())
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, other)
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'o': (other, staged, None)}))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        # Header metadata still served; payload refuses.
        assert 'f32' in reader.keys()
        with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
            reader.get_tensor('f32')
    assert resolver.report()['bytes_from_pool'] == 0


def test_strict_source_without_any_map_refuses(tmp_path, monkeypatch):
    path, _ = _shard(tmp_path)
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_residency_resolver_for_tests()
    activate_staged_tier_policy("ram,ssd")
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        assert 'f32' in reader.keys()
        with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
            reader.get_tensor('f32')


def test_strict_source_stale_ram_falls_to_allowed_stage(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    ram_root, ram = _promote_ram(tmp_path, {'s': staged})
    _announce(tmp_path, epoch=STALE_EPOCH)  # reboot: tmpfs emptied, map survived
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'s': (path, staged, ram['s'])}, ram_root=ram_root, epoch=EPOCH))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        from safetensors import safe_open
        with safe_open(str(path), framework='pt') as reference:
            assert torch.equal(
                reader.get_tensor('bf16').view(torch.uint8),
                reference.get_tensor('bf16').view(torch.uint8))
    report = resolver.report()
    assert report['bytes_from_stage'] > 0 and report['bytes_from_pool'] == 0


def test_strict_ram_only_with_dead_epoch_refuses(tmp_path, monkeypatch):
    path, _ = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    ram_root, ram = _promote_ram(tmp_path, {'s': staged})
    _announce(tmp_path, epoch=STALE_EPOCH)
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'s': (path, staged, ram['s'])}, ram_root=ram_root, epoch=EPOCH),
        tiers="ram")
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        with pytest.raises(TierPolicyRefused, match="no-permitted-tier"):
            reader.get_tensor('f32')
    assert resolver.report()['bytes_from_pool'] == 0


def test_strict_source_corrupt_range_refuses(tmp_path, monkeypatch):
    path, _ = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    staged.write_bytes(staged.read_bytes()[:-8])  # size fence fails
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'s': (path, staged, None)}))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        with pytest.raises(TierPolicyRefused):
            reader.get_tensor('f32')
    assert resolver.report()['bytes_from_pool'] == 0


def test_strict_get_slice_proxy_metadata_then_staged_payload(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _strict(monkeypatch, _write_map(tmp_path, {'s': (path, staged, None)}))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        proxy = reader.get_slice('f32')
        assert proxy.get_shape() == list(tensors['f32'].shape)
        assert proxy.get_dtype() == tensors['f32'].dtype
        from safetensors import safe_open
        with safe_open(str(path), framework='pt') as reference:
            assert torch.equal(proxy[0].view(torch.uint8),
                               reference.get_slice('f32')[0].view(torch.uint8))
    assert resolver.report()['bytes_from_pool'] == 0


def test_strict_empty_tensor_built_locally(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    _strict(monkeypatch, _write_map(tmp_path, {'s': (path, staged, None)}))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        got = reader.get_tensor('empty')
        assert got.shape == tensors['empty'].shape
        assert got.dtype == tensors['empty'].dtype
        assert got.numel() == 0


# -- strict PWC renders -----------------------------------------------------

def _pwc(tmp_path):
    from prismaquant.production_weight_cache import ProductionWeightCache
    key = ('unit0', 'TESSERA_E4M3_K1_R1024')
    tensor = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    path = tmp_path / 'pool' / 'unit0.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    cache = ProductionWeightCache(weights={key: str(path)}, levers={})
    cache.enable_lru(1 << 20)
    return cache, key, path, tensor


def test_strict_pwc_stage_serves_with_digest_binding(tmp_path, monkeypatch):
    cache, key, path, tensor = _pwc(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache.require_file_load_sha256({key: digest}, max_file_bytes=path.stat().st_size)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _strict(monkeypatch, _write_map(tmp_path, {'p': (path, staged, None)}))
    assert cache.prefetch([key], max_workers=1) == 1
    assert torch.equal(cache.get(*key), tensor)
    assert cache.file_load_receipt(key, cache.get(*key))['serving_tier'] == 'stage'
    report = resolver.report()
    assert report['bytes_from_pool'] == 0 and report['bytes_from_stage'] > 0


def test_strict_pwc_missing_digest_refuses(tmp_path, monkeypatch):
    cache, key, path, _ = _pwc(tmp_path)
    cache.enable_file_load_receipts(max_file_bytes=path.stat().st_size)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    _strict(monkeypatch, _write_map(tmp_path, {'p': (path, staged, None)}))
    with pytest.raises(TierPolicyRefused, match="missing-digest-binding"):
        cache.prefetch([key], max_workers=1)


def test_strict_pwc_unbounded_refuses_without_fallthrough(tmp_path, monkeypatch):
    cache, key, path, _ = _pwc(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    _strict(monkeypatch, _write_map(tmp_path, {'p': (path, staged, None)}))
    with pytest.raises(TierPolicyRefused, match="unbounded-read"):
        cache.prefetch([key], max_workers=1)


def test_strict_pwc_ram_serves(tmp_path, monkeypatch):
    cache, key, path, tensor = _pwc(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache.require_file_load_sha256({key: digest}, max_file_bytes=path.stat().st_size)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    ram_root, ram = _promote_ram(tmp_path, {'p': staged})
    _announce(tmp_path)
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'p': (path, staged, ram['p'])}, ram_root=ram_root))
    assert cache.prefetch([key], max_workers=1) == 1
    assert torch.equal(cache.get(*key), tensor)
    report = resolver.report()
    assert report['bytes_from_ram'] == path.stat().st_size
    assert report['bytes_from_stage'] == 0 and report['bytes_from_pool'] == 0


# -- strict wire renders ----------------------------------------------------

def _wire(tmp_path, blob=b'strict-tier-wire-0123456789'):
    pool = tmp_path / 'pool'
    pool.mkdir(parents=True, exist_ok=True)
    wire = pool / 'cell.wire'
    wire.write_bytes(blob)
    return {'wire': str(wire), 'record': {
        'blob_bytes': len(blob), 'blob_sha256': hashlib.sha256(blob).hexdigest()}}, wire, blob


def test_strict_wire_stage_serves(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob = _wire(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, wire)
    resolver = _strict(monkeypatch, _write_map(tmp_path, {'w': (wire, staged, None)}))
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob and digest == cell['record']['blob_sha256']
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert report['serving_tiers'][-1]['serving_tier'] == 'stage'


def test_strict_wire_corrupt_stage_fails_clear(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob = _wire(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, wire)
    staged.write_bytes(blob[:-1] + bytes([blob[-1] ^ 0xFF]))
    resolver = _strict(monkeypatch, _write_map(tmp_path, {'w': (wire, staged, None)}))
    with pytest.raises(TierPolicyRefused, match="content-corruption"):
        _read_verified_wire_blob(cell)
    assert resolver.report()['bytes_from_pool'] == 0


def test_strict_wire_ram_corrupt_never_adopts_stage(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob = _wire(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, wire)
    ram_root, ram = _promote_ram(tmp_path, {'w': staged})
    ram['w'].write_bytes(blob[:-1] + bytes([blob[-1] ^ 0xFF]))
    _announce(tmp_path)
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'w': (wire, staged, ram['w'])}, ram_root=ram_root))
    with pytest.raises(TierPolicyRefused, match="content-corruption"):
        _read_verified_wire_blob(cell)
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert report['bytes_from_stage'] == 0  # healthy stage NOT adopted


# -- strict activation payloads ----------------------------------------------

def _activation_policy(*, buffer=8 * 1024 ** 2, scratch=8 * 1024 ** 2):
    return dict(schema='prismaquant.verified_activation_load.v1',
                max_buffer_bytes=buffer, max_scratch_bytes=scratch)


def test_strict_verified_activation_from_stage_never_opens_pool(tmp_path, monkeypatch):
    from prismaquant.perturbed_x_cache import load_verified_activation_cache_entry
    pool = tmp_path / 'pool'
    pool.mkdir(parents=True, exist_ok=True)
    payload = {'inputs': torch.arange(64, dtype=torch.float32).reshape(8, 8)}
    path = pool / 'capture.pt'
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _strict(monkeypatch, _write_map(tmp_path, {'a': (path, staged, None)}))
    opened = []
    real_open = os.open

    def counting(target, *args, **kwargs):
        opened.append(os.fspath(target))
        return real_open(target, *args, **kwargs)

    monkeypatch.setattr(os, "open", counting)
    got, _ = load_verified_activation_cache_entry(
        path, expected_sha256=digest, policy=_activation_policy(),
        max_storage_bytes=4 * 1024 ** 2)
    assert torch.equal(got['inputs'], payload['inputs'])
    assert str(path) not in opened  # the pool file is never opened
    assert resolver.report()['serving_tiers'][-1]['serving_tier'] == 'stage'


def test_strict_verified_activation_unmapped_refuses(tmp_path, monkeypatch):
    from prismaquant.perturbed_x_cache import load_verified_activation_cache_entry
    pool = tmp_path / 'pool'
    pool.mkdir(parents=True, exist_ok=True)
    path = pool / 'capture.pt'
    torch.save({'inputs': torch.zeros(4, 4)}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    other = pool / 'other.pt'
    other.write_bytes(path.read_bytes())
    root = _stage_root(tmp_path)
    _strict(monkeypatch, _write_map(
        tmp_path, {'o': (other, _stage_whole(root, other), None)}))
    # The map is bound but names nothing for this file: staged-not-serving.
    with pytest.raises(TierPolicyRefused, match="staged-not-serving"):
        load_verified_activation_cache_entry(
            path, expected_sha256=digest, policy=_activation_policy(),
            max_storage_bytes=4 * 1024 ** 2)


def _exact_identity(name, session="strict-tier-session"):
    return {"session": session, "slot": name, "kind": "boundary",
            "coordinates": {"batch": 0, "boundary": 0, "probe": None}}


def test_strict_exact_entry_from_stage(tmp_path, monkeypatch):
    from prismaquant.perturbed_x_cache import (
        prefetch_exact_activation_cache_entries, write_exact_activation_cache_entry)
    entries = tmp_path / 'pool' / 'entries'
    entries.mkdir(parents=True, exist_ok=True)
    tensor = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    nbytes = tensor.numel() * tensor.element_size()
    ref = write_exact_activation_cache_entry(
        entries, "entry-0", tensor, identity=_exact_identity("entry-0"),
        max_tensor_bytes=nbytes, max_file_bytes=nbytes + 65536)
    path = Path(ref.path)
    root = _stage_root(tmp_path)
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'e': (path, _stage_whole(root, path), None)}))
    with prefetch_exact_activation_cache_entries(
            [ref], max_tensor_bytes=nbytes, expected_session="strict-tier-session",
            release_file_pages=False) as window:
        assert torch.equal(window._tensors[ref], tensor)
    assert resolver.report()['serving_tiers'][-1]['serving_tier'] == 'stage'


def test_strict_exact_entry_unmapped_refuses(tmp_path, monkeypatch):
    from prismaquant.perturbed_x_cache import (
        prefetch_exact_activation_cache_entries, write_exact_activation_cache_entry)
    entries = tmp_path / 'pool' / 'entries'
    entries.mkdir(parents=True, exist_ok=True)
    tensor = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    nbytes = tensor.numel() * tensor.element_size()
    ref = write_exact_activation_cache_entry(
        entries, "entry-0", tensor, identity=_exact_identity("entry-0"),
        max_tensor_bytes=nbytes, max_file_bytes=nbytes + 65536)
    root = _stage_root(tmp_path)
    other = Path(ref.path).with_name("entry-1.pt")
    other.write_bytes(Path(ref.path).read_bytes())
    _strict(monkeypatch, _write_map(
        tmp_path, {'o': (other, _stage_whole(root, other), None)}))
    with pytest.raises(TierPolicyRefused, match="staged-not-serving"):
        with prefetch_exact_activation_cache_entries(
                [ref], max_tensor_bytes=nbytes,
                expected_session="strict-tier-session",
                release_file_pages=False):
            pass


# -- sealed binding: parser, default, dispatcher payload ---------------------

def test_sealed_tier_binding_parser_default_and_dispatch(tmp_path, monkeypatch):
    from prismaquant.joint_cost_quantum import build_parser
    from prismaquant.staged_tier_policy import DEFAULT_ALLOWED_TIERS
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import dispatch_joint_quanta
    from dispatch_joint_quanta import STAGED_ALLOWED_TIERS, quantum_argv, stage_a_argv

    assert DEFAULT_ALLOWED_TIERS == "ram,ssd" == STAGED_ALLOWED_TIERS
    assert (build_parser().get_default("allowed_tiers")
            == DEFAULT_ALLOWED_TIERS == STAGED_ALLOWED_TIERS)
    parsed = build_parser().parse_args(["--allowed-tiers", "ram",
                                        "--quantum", "q", "--quantum-sha256", "0" * 64,
                                        "--plan", "p", "--plan-sha256", "0" * 64,
                                        "--prepared", "r", "--prepared-sha256", "0" * 64,
                                        "--adjoint", "a", "--adjoint-sha256", "0" * 64,
                                        "--output-root", "o"])
    assert parsed.allowed_tiers == "ram"
    with pytest.raises(ValueError, match="TIER-04"):
        parse_allowed_tiers("pool")

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "sha256:" + "0" * 64}, "env": {}}))
    monkeypatch.setattr(dispatch_joint_quanta, "SPEC_PATH", spec)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"output_root": str(tmp_path / "campaign-root")}))
    campaign = {"plan_sha256": "a" * 64, "prepared_sha256": "b" * 64,
                "manifest_sha256": "c" * 64,
                "scope": {"campaign": "strict-tier-fixture", "layers": [1]},
                "read_manifest_sha256": "d" * 64, "plan_path": str(plan_path),
                "prepared_path": "/fixture/prepare/prepared.json",
                "roster_sha256": hashlib.sha256(b"roster\n").hexdigest()}
    slices = tmp_path / "slices"
    slices.mkdir()
    manifest_path = slices / "layer-001.data-manifest.json"
    manifest_path.write_bytes(json.dumps({"slice": "layer-001"}).encode())
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    record = {
        "schema": "prismaquant.joint_layer_quanta.v1", "quantum_id": "layer-001",
        "layer": 1,
        "campaign": {
            "plan_sha256": campaign["plan_sha256"],
            "prepared_sha256": campaign["prepared_sha256"],
            "plan_path": campaign["plan_path"],
            "prepared_path": campaign["prepared_path"],
            "read_manifest_sha256": campaign["read_manifest_sha256"],
            "campaign_scope": campaign["scope"],
            "unit_roster_sha256": campaign["roster_sha256"]},
        "read_set": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "source_phase": {"name": "layer-1", "start_bytes": 0, "end_bytes": 2048}},
        "chunks": [{"name": "layer-001-chunk-000", "start_bytes": 0, "end_bytes": 2048}],
        "windows": [{"window_index": 0, "names": []}],
        "adjoint": {"checkpoint_boundary": 3, "chain_layers": [], "receipt_sha256": None},
        "output_space": {"root": "layer-quanta/layer-001"}}
    record["identity_sha256"] = canonical_json_sha256(record, where="fixture")
    record_path = tmp_path / "layer-001.json"
    record_path.write_text(json.dumps(record))
    adjoint_path = tmp_path / "adjoint-capture.json"
    adjoint_path.write_bytes(b'{"receipt": "fixture"}')
    argv = quantum_argv(record, record_path=record_path,
                        output_root=tmp_path / "campaign-root",
                        adjoint_path=adjoint_path)
    inner = argv[argv.index("--") + 1:]
    payload = inner[inner.index("prismaquant.joint_cost_quantum") - 2:]
    assert payload[:3] == ["python3", "-m", "prismaquant.joint_cost_quantum"]
    assert payload[payload.index("--allowed-tiers") + 1] == STAGED_ALLOWED_TIERS
    # The threaded payload parses at the real producer parser (no stub).
    reparsed = build_parser().parse_args(payload[3:])
    assert reparsed.allowed_tiers == STAGED_ALLOWED_TIERS

    manifest = tmp_path / "adjoint.data-manifest.json"
    manifest.write_bytes(json.dumps({
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "mount_prefix": "/mnt/shared", "entries": [], "entry_count": 0,
        "total_bytes": 0,
        "annotations": {
            "parent_manifest_sha256": campaign["read_manifest_sha256"],
            "plan_sha256": campaign["plan_sha256"],
            "prepared_sha256": campaign["prepared_sha256"],
            "phases": [{"name": "head", "bytes": 0, "cumulative_bytes": 0}]}}).encode())
    argv = stage_a_argv(manifest, campaign)
    inner = argv[argv.index("--") + 1:]
    payload = inner[inner.index("prismaquant.joint_adjoint_capture") - 2:]
    assert payload[:3] == ["python3", "-m", "prismaquant.joint_adjoint_capture"]
    assert payload[payload.index("--allowed-tiers") + 1] == STAGED_ALLOWED_TIERS
