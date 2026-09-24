"""Strict allowed-tier enforcement + lifetime-pinned reads (PQ #845, #850).

The staged-read contract forbids bulk-input reads from the pool/HDD tier;
the reader-lease contract pins every staged byte for its reader's
lifetime. These tests run the REAL chain — real resolver, real composed
map, real PB fragments/material written by the REAL PB writers, the REAL
pinned SDK (candidate integration pin, drift-refused), real claim rows in the real
format — on tiny fixtures, never giant payloads:

- the legacy test proves the hole (inactive policy serves pool bytes);
- refusal tests prove the gates (unmapped, no helper, no context,
  missing digest, corrupt content) with zero pool bytes;
- serve tests prove permitted tiers serve byte-equal through held
  descriptors with SDK serving records, and pins release exactly.

No PB behavior is stubbed or re-implemented here. Where the composed map
cannot name covers (RAM movers), the tests prove the fast refusal and the
honest SSD re-acquire — never a pretended identity.
"""
import gzip
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from fleet_sdk import require_prismabuild_sdk
import torch
from concurrent.futures import ThreadPoolExecutor
from safetensors.torch import save_file

from prismaquant import layer_streaming
from prismaquant.residency_map import (
    ENV_VAR, SCHEMA, TIERS_DIR_ENV_VAR, bind_residency_manifest,
    residency_map_key, residency_resolver, reset_residency_resolver_for_tests,
)
from prismaquant.staged_tier_policy import (
    DEFAULT_ALLOWED_TIERS, TierPolicyRefused, activate_staged_tier_policy,
    active_policy, deactivate_staged_tier_policy_for_tests,
    parse_allowed_tiers, staged_tier_policy_test_context,
)
from prismaquant.staged_lease import (
    LeaseRefused, PINNED_SDK_COMMIT, set_lease_helper_root,
)
from prismaquant.residency_shard_reader import (
    STAGED_RANGE_WAIT_ENV, STAGED_RANGE_WAIT_S, staged_range_wait_s,
)

MANIFEST = 'e' * 64
LEAD = 'f' * 64
RAM_TIER = 'ram:dl380g10'
EPOCH = '1789771929-aba6e46e41fb03ef'
STALE_EPOCH = '1789788888-9c1d2e3f4a5b'
STAGE_TIER = 'prismabuild-stage:dl380g10'

PB_PIN_NOTE = (
    "portable reviewed install: tests import the pbtest-pinned "
    "prismabuild distribution (tools/resolve_prismabuild_dev_pin.py), "
    "never a private worktree")


@pytest.fixture(autouse=True)
def _forget_state(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(TIERS_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv("PRISMABUILD_ACTION_KEY", raising=False)
    monkeypatch.delenv("PRISMABUILD_ACTION_NONCE", raising=False)
    monkeypatch.delenv("PRISMABUILD_ACTION_SCOPE", raising=False)
    monkeypatch.delenv("PRISMABUILD_READER_HELPER_ROOT", raising=False)
    reset_residency_resolver_for_tests()
    deactivate_staged_tier_policy_for_tests()
    set_lease_helper_root(None)
    from prismaquant.staged_lease import clear_injected_sdk_for_tests
    clear_injected_sdk_for_tests()
    yield
    reset_residency_resolver_for_tests()
    deactivate_staged_tier_policy_for_tests()
    set_lease_helper_root(None)
    clear_injected_sdk_for_tests()


def _hex64(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


LAUNCH_NONCE = "n" * 32
LAUNCH_SCOPE = "unit-1"


def _launch_env(monkeypatch, consumer):
    """Launch-bound identity pair, matching the claim row exactly (the new
    SDK binds pins from launch env + live claim, never a claim alone)."""
    monkeypatch.setenv("PRISMABUILD_ACTION_KEY", consumer)
    monkeypatch.setenv("PRISMABUILD_ACTION_NONCE", LAUNCH_NONCE)
    monkeypatch.setenv("PRISMABUILD_ACTION_SCOPE", LAUNCH_SCOPE)
    monkeypatch.setenv("PRISMABUILD_ACTION_NONCE", LAUNCH_NONCE)
    monkeypatch.setenv("PRISMABUILD_ACTION_SCOPE", LAUNCH_SCOPE)


# -- pinned PB SDK + queue fixtures (real writers, real formats) ------------

def _pb():
    """The reviewed installed SDK via explicit test-only injection.

    Provenance (single non-editable install at the candidate pin, no
    worktree shadow) is asserted inside the injection; the pbtest pin
    guard proves it worker-side before pytest starts. The exact API
    surface is re-checked, so a passing suite always names what it ran
    against. Never silently skips on a missing dependency.
    """
    require_prismabuild_sdk()
    from prismaquant.staged_lease import inject_installed_sdk_for_tests
    module = inject_installed_sdk_for_tests()
    import prismabuild.pool as pool_mod
    import prismabuild.residency_map as map_mod
    return module, pool_mod, map_mod


def _pb_queue(tmp_path, pool_mod, consumer):
    """A real queue layout with a real claim row (PB's own test shape).

    The claim carries the launch-bound nonce/scope pair plus the holding
    box, exactly what the strict SDK matches launch env against: no half
    without the other binds.
    """
    import socket
    queue = pool_mod.PoolQueue(tmp_path)
    queue.ensure_layout()
    stage = tmp_path / 'stage' / 'prewarm'
    stage.mkdir(parents=True, exist_ok=True)
    claimed = queue.dir(pool_mod.CLAIMED)
    claimed.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    (claimed / f"{consumer}.json").write_text(json.dumps({
        "action_key": consumer, "claimed_by": f"{host}:4242:pbtest",
        "claimed_host": host,
        "resource_scope": {"action_key": consumer, "nonce": LAUNCH_NONCE,
                           "scope_id": LAUNCH_SCOPE}}))
    return queue, stage


def _pb_publish(rl, map_mod, root, stage, consumer, mover, manifest, entries):
    """Publish one mover's window with the REAL PB writers.

    ``entries``: {map_key: (declared Path, staged Path)}. Returns the
    materialization generation. Fragments carry no ram_path (composed
    overlay only); every entry is hashed at publish like a mover would.
    """
    frag_entries, mat_entries = {}, {}
    for key, (declared, staged) in entries.items():
        blob = staged.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        frag_entries[key] = {"stage_path": str(staged), "bytes": len(blob),
                             "sha256": digest,
                             "offset": int(key.split(":", 1)[0])}
        mat_entries[key] = {"stage_path": str(staged), "bytes": len(blob),
                            "sha256": digest,
                            "file_id": rl.stat_identity(str(staged))}
    map_mod.write_fragment(root, {
        "schema": map_mod.RESIDENCY_MAP_FRAGMENT_SCHEMA_V1,
        "consumer_action_key": consumer, "mover_action_key": mover,
        "tier_id": STAGE_TIER, "stage_root": str(stage),
        "manifest_sha256": manifest, "entries": frag_entries})
    generation = rl.mint_generation()
    rl.write_material(
        root, consumer_action_key=consumer, mover_action_key=mover,
        tier_id=STAGE_TIER, stage_root=str(stage), manifest_sha256=manifest,
        generation=generation, entries=mat_entries)
    return generation


def _pb_publish_ram(rl, map_mod, root, ram_root, consumer, mover, manifest,
                    entries, epoch):
    """Publish one RAM mover's window: ram-tier fragment + sidecar at the
    announced epoch, entries naming the tmpfs copies. Returns generation."""
    frag_entries, mat_entries = {}, {}
    for key, (declared, ram_file) in entries.items():
        blob = ram_file.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        frag_entries[key] = {"stage_path": str(ram_file), "bytes": len(blob),
                             "sha256": digest,
                             "offset": int(key.split(":", 1)[0])}
        mat_entries[key] = {"stage_path": str(ram_file), "bytes": len(blob),
                            "sha256": digest,
                            "file_id": rl.stat_identity(str(ram_file))}
    map_mod.write_fragment(root, {
        "schema": map_mod.RESIDENCY_MAP_FRAGMENT_SCHEMA_V1,
        "consumer_action_key": consumer, "mover_action_key": mover,
        "tier_id": RAM_TIER, "stage_root": str(ram_root),
        "manifest_sha256": manifest, "epoch": epoch,
        "entries": frag_entries})
    generation = rl.mint_generation()
    rl.write_material(
        root, consumer_action_key=consumer, mover_action_key=mover,
        tier_id=RAM_TIER, stage_root=str(ram_root),
        manifest_sha256=manifest, generation=generation,
        entries=mat_entries, epoch=epoch)
    return generation


def _leased_fixture(tmp_path, monkeypatch, staged_files):
    """Full honest stack: SDK, queue+claim, published material, PQ map, env.

    ``staged_files``: {name: (declared Path, staged Path, ram Path|None)}.
    Returns (resolver, consumer, mover). Policy + helper activated.
    """
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover = _hex64(f"mover-{tmp_path}")
    queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
    root = tmp_path / 'residency'
    entries = {}
    for name, (declared, staged, _ram) in staged_files.items():
        key = residency_map_key(str(declared), 0)
        entries[key] = (declared, staged)
    _pb_publish(rl, map_mod, root, stage, consumer, mover, MANIFEST, entries)
    rows = {name: (declared, staged, ram)
            for name, (declared, staged, ram) in staged_files.items()}
    map_path = _write_map(tmp_path, rows, leads=[mover])
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    activate_staged_tier_policy("ram,ssd")
    return residency_resolver(), consumer, mover


def _pins_live(tmp_path, consumer):
    directory = tmp_path / 'residency' / 'leases' / consumer
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.glob("*.lease.json"))


# -- payload fixtures --------------------------------------------------------

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
    # THE PB stage root: staged files, fragment stage_root, and the
    # ownership-lock scope must agree, so every fixture stages here.
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
               ram_root=None, epoch=EPOCH, leads=None, stage_root=None):
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
        if stage_root is None:
            stage_root = staged.parent
    body = {'schema': SCHEMA, 'tier_id': STAGE_TIER,
            'stage_root': str(stage_root or (tmp_path / 'stage' / 'prewarm')),
            'manifest_sha256': manifest_sha256,
            'leads': leads if leads is not None else [LEAD],
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


def _strict(monkeypatch, map_path, tiers="ram,ssd"):
    resolver = _bind(monkeypatch, map_path)
    activate_staged_tier_policy(tiers)
    return resolver


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
    with staged_tier_policy_test_context("ram,ssd") as allowed:
        assert allowed == frozenset({"ram", "ssd"})
        worker = threading.Thread(
            target=lambda: seen.setdefault("active", active_policy()))
        worker.start()
        worker.join()
        assert seen["active"] == frozenset({"ram", "ssd"})
        with pytest.raises(TierPolicyRefused):
            raise TierPolicyRefused("sentinel")
    assert active_policy() is None
    assert seen["active"] == frozenset({"ram", "ssd"})
    with pytest.raises(RuntimeError, match="boom"):
        with staged_tier_policy_test_context("ram"):
            raise RuntimeError("boom")
    assert active_policy() is None


def test_nested_scope_narrows_never_widens_and_restores():
    with staged_tier_policy_test_context("ram,ssd"):
        assert active_policy() == frozenset({"ram", "ssd"})
        with staged_tier_policy_test_context("ram"):
            assert active_policy() == frozenset({"ram"})
        assert active_policy() == frozenset({"ram", "ssd"})
        with staged_tier_policy_test_context("ram"):
            with staged_tier_policy_test_context("ram,ssd"):
                assert active_policy() == frozenset({"ram"})
            assert active_policy() == frozenset({"ram"})
            with pytest.raises(RuntimeError, match="incompatible"):
                with staged_tier_policy_test_context("ssd"):
                    pass
            assert active_policy() == frozenset({"ram"})
        assert active_policy() == frozenset({"ram", "ssd"})
    assert active_policy() is None
    activate_staged_tier_policy("ram")
    try:
        with staged_tier_policy_test_context("ram,ssd"):
            assert active_policy() == frozenset({"ram"})
        assert active_policy() == frozenset({"ram"})
    finally:
        deactivate_staged_tier_policy_for_tests()
    assert active_policy() is None


def test_overlapping_scopes_on_threads_refuse():
    with staged_tier_policy_test_context("ram,ssd"):
        outcome = {}

        def enter_elsewhere():
            try:
                with staged_tier_policy_test_context("ram"):
                    outcome["entered"] = True
            except RuntimeError as exc:
                outcome["refused"] = str(exc)

        worker = threading.Thread(target=enter_elsewhere)
        worker.start()
        worker.join()
        assert "entered" not in outcome
        assert "unsupported" in outcome["refused"]
        assert active_policy() == frozenset({"ram", "ssd"})
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


# -- strict source shard, lifetime-pinned -----------------------------------

def test_strict_source_stage_serves_pinned_with_exact_release(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    spans = _header_spans(path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch, {'s': (path, staged, None)})
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        assert reader._handle is None
        assert sorted(reader.keys()) == sorted(tensors)
        from safetensors import safe_open
        with safe_open(str(path), framework='pt') as reference:
            for name in sorted(tensors):
                if tensors[name].numel():
                    assert torch.equal(
                        reader.get_tensor(name).view(torch.uint8),
                        reference.get_tensor(name).view(torch.uint8))
        # The window is held for the reader's life: pinned while open.
        assert len(_pins_live(tmp_path, consumer)) == 1
    report = resolver.report()
    served = [n for n in tensors if spans[n][1] > spans[n][0]]
    assert report['bytes_from_pool'] == 0
    assert report['range_hits'] == len(served)
    assert report['serving_tier_count'] == 1
    row = report['serving_tiers'][0]
    assert row['serving_tier'] == 'stage' and row['pin_id'] and row['range_ref']
    # Exact release: the last release unlinked the pin file.
    assert _pins_live(tmp_path, consumer) == []


def test_strict_source_ram_serves_first_pinned(tmp_path, monkeypatch):
    """RAM-first completion: a live tmpfs copy with real RAM-mover covers
    pins at the announced epoch and serves; the SSD copy is never opened."""
    path, tensors = _shard(tmp_path)
    spans = _header_spans(path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    ram_root, ram = _promote_ram(tmp_path, {'s': staged})
    _announce(tmp_path)
    rows = {'s': (path, staged, ram['s'])}
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover_ssd = _hex64(f"mover-ssd-{tmp_path}")
    mover_ram = _hex64(f"mover-ram-{tmp_path}")
    _pb_queue(tmp_path, pool_mod, consumer)
    root_dir = tmp_path / 'residency'
    key = residency_map_key(str(path), 0)
    _pb_publish(rl, map_mod, root_dir, root, consumer, mover_ssd, MANIFEST,
                {key: (path, staged)})
    _pb_publish_ram(rl, map_mod, root_dir, ram_root, consumer, mover_ram,
                    MANIFEST, {key: (path, ram['s'])}, EPOCH)
    map_path = _write_map(tmp_path, rows, ram_root=ram_root, leads=[mover_ssd])
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    activate_staged_tier_policy("ram,ssd")
    resolver = residency_resolver()
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        from safetensors import safe_open
        with safe_open(str(path), framework='pt') as reference:
            got = reader.get_tensor('f32')
            assert torch.equal(got.view(torch.uint8),
                               reference.get_tensor('f32').view(torch.uint8))
    report = resolver.report()
    assert report['ram_fallbacks'] == []
    span = spans['f32'][1] - spans['f32'][0]
    assert report['bytes_from_ram'] == span
    assert report['bytes_from_stage'] == 0
    assert report['bytes_from_pool'] == 0
    row = report['serving_tiers'][-1]
    assert row['serving_tier'] == 'ram' and row['pin_id'] and row['range_ref']
    assert _pins_live(tmp_path, consumer) == []


def test_strict_source_unmapped_refuses_without_pool_bytes(tmp_path, monkeypatch):
    _pb()
    path, _ = _shard(tmp_path)
    other = path.with_name('model-00002-of-00002.safetensors')
    other.write_bytes(path.read_bytes())
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, other)
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'o': (other, staged, None)}))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
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


def test_strict_source_unpublished_material_refuses(tmp_path, monkeypatch):
    """A map entry with no published material behind it refuses at
    acquisition (unpublished) instead of serving staged bytes unpinned
    or falling open to the pool. Context is live; the material is not."""
    require_prismabuild_sdk()
    _pb()
    import prismabuild.pool as pool_mod
    consumer = _hex64(f"consumer-{tmp_path}")
    _pb_queue(tmp_path, pool_mod, consumer)
    path, _ = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    map_path = _write_map(tmp_path, {'s': (path, staged, None)})
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    activate_staged_tier_policy("ram,ssd")
    resolver = residency_resolver()
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        with pytest.raises(LeaseRefused, match="unpublished"):
            reader.get_tensor('f32')
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert report['bytes_from_stage'] == 0


def test_strict_source_without_context_refuses(tmp_path, monkeypatch):
    """Helper present but no action context: identity is never guessed."""
    path, _ = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    _leased_fixture(tmp_path, monkeypatch, {'s': (path, staged, None)})
    monkeypatch.delenv("PRISMABUILD_ACTION_KEY", raising=False)
    resolver = residency_resolver()
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        with pytest.raises(LeaseRefused, match="lease-context-unavailable"):
            reader.get_tensor('f32')
    assert resolver.report()['bytes_from_pool'] == 0


def test_strict_source_stale_ram_falls_to_allowed_stage(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    ram_root, ram = _promote_ram(tmp_path, {'s': staged})
    _announce(tmp_path, epoch=STALE_EPOCH)
    rows = {'s': (path, staged, ram['s'])}
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover = _hex64(f"mover-{tmp_path}")
    _pb_queue(tmp_path, pool_mod, consumer)
    _pb_publish(rl, map_mod, tmp_path / 'residency', root, consumer, mover,
                MANIFEST, {residency_map_key(str(path), 0): (path, staged)})
    map_path = _write_map(tmp_path, rows, ram_root=ram_root, epoch=EPOCH,
                          leads=[mover])
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    activate_staged_tier_policy("ram,ssd")
    resolver = residency_resolver()
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        from safetensors import safe_open
        with safe_open(str(path), framework='pt') as reference:
            assert torch.equal(
                reader.get_tensor('bf16').view(torch.uint8),
                reference.get_tensor('bf16').view(torch.uint8))
    report = resolver.report()
    assert report['bytes_from_stage'] > 0 and report['bytes_from_pool'] == 0
    assert _pins_live(tmp_path, consumer) == []


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
        with pytest.raises(TierPolicyRefused, match="ssd-not-allowed"):
            reader.get_tensor('f32')
    assert resolver.report()['bytes_from_pool'] == 0


def test_strict_source_corrupt_range_refuses(tmp_path, monkeypatch):
    path, _ = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    staged.write_bytes(staged.read_bytes()[:-8])
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
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch, {'s': (path, staged, None)})
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        proxy = reader.get_slice('f32')
        assert proxy.get_shape() == list(tensors['f32'].shape)
        assert proxy.get_dtype() == tensors['f32'].dtype
        from safetensors import safe_open
        with safe_open(str(path), framework='pt') as reference:
            assert torch.equal(proxy[0].view(torch.uint8),
                               reference.get_slice('f32')[0].view(torch.uint8))
    assert resolver.report()['bytes_from_pool'] == 0
    assert _pins_live(tmp_path, consumer) == []


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


# -- strict PWC renders, pinned ----------------------------------------------

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


def test_strict_pwc_stage_serves_pinned_with_digest_binding(tmp_path, monkeypatch):
    cache, key, path, tensor = _pwc(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache.require_file_load_sha256({key: digest}, max_file_bytes=path.stat().st_size)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch, {'p': (path, staged, None)})
    assert cache.prefetch([key], max_workers=1) == 1
    assert torch.equal(cache.get(*key), tensor)
    receipt = cache.file_load_receipt(key, cache.get(*key))
    assert receipt['serving_tier'] == 'stage'
    report = resolver.report()
    assert report['bytes_from_pool'] == 0 and report['bytes_from_stage'] > 0
    assert report['serving_tiers'][-1]['pin_id']
    assert _pins_live(tmp_path, consumer) == []


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


# -- strict wire renders, pinned ----------------------------------------------

def _wire(tmp_path, blob=b'strict-tier-wire-0123456789'):
    pool = tmp_path / 'pool'
    pool.mkdir(parents=True, exist_ok=True)
    wire = pool / 'cell.wire'
    wire.write_bytes(blob)
    return {'wire': str(wire), 'record': {
        'blob_bytes': len(blob), 'blob_sha256': hashlib.sha256(blob).hexdigest()}}, wire, blob


def test_strict_wire_stage_serves_pinned(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob = _wire(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, wire)
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch, {'w': (wire, staged, None)})
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob and digest == cell['record']['blob_sha256']
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert report['serving_tiers'][-1]['serving_tier'] == 'stage'
    assert report['serving_tiers'][-1]['pin_id']
    assert _pins_live(tmp_path, consumer) == []


def test_strict_wire_corrupt_content_fails_clear(tmp_path, monkeypatch):
    """Same-size corruption is an integrity failure: fail clear with zero
    stage bytes served and no alternate adoption."""
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob = _wire(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, wire)
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch, {'w': (wire, staged, None)})
    staged.write_bytes(blob[:-1] + bytes([blob[-1] ^ 0xFF]))
    with pytest.raises(LeaseRefused) as excinfo:
        _read_verified_wire_blob(cell)
    assert excinfo.value.kind == "integrity"
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert report['bytes_from_stage'] == 0
    assert _pins_live(tmp_path, consumer) == []


def test_strict_ram_corrupt_fails_clear_without_stage_adoption(tmp_path, monkeypatch):
    """A corrupt tmpfs copy with published RAM material is an integrity
    failure: the RAM acquire fails clear and the healthy SSD copy is NOT
    adopted as an unchecked alternate — zero stage bytes, zero pool."""
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob = _wire(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, wire)
    ram_root, ram = _promote_ram(tmp_path, {'w': staged})
    _announce(tmp_path)
    rows = {'w': (wire, staged, ram['w'])}
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover_ssd = _hex64(f"mover-ssd-{tmp_path}")
    mover_ram = _hex64(f"mover-ram-{tmp_path}")
    _pb_queue(tmp_path, pool_mod, consumer)
    root_dir = tmp_path / 'residency'
    key = residency_map_key(str(wire), 0)
    _pb_publish(rl, map_mod, root_dir, root, consumer, mover_ssd, MANIFEST,
                {key: (wire, staged)})
    _pb_publish_ram(rl, map_mod, root_dir, ram_root, consumer, mover_ram,
                    MANIFEST, {key: (wire, ram['w'])}, EPOCH)
    ram['w'].write_bytes(blob[:-1] + bytes([blob[-1] ^ 0xFF]))
    map_path = _write_map(tmp_path, rows, ram_root=ram_root, leads=[mover_ssd])
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    activate_staged_tier_policy("ram,ssd")
    resolver = residency_resolver()
    with pytest.raises(LeaseRefused) as excinfo:
        _read_verified_wire_blob(cell)
    assert excinfo.value.kind == "integrity"
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert report['bytes_from_stage'] == 0
    assert report['bytes_from_ram'] == 0
    assert _pins_live(tmp_path, consumer) == []


def test_strict_wire_ram_corrupt_serves_checked_stage(tmp_path, monkeypatch):
    """A corrupt RAM copy never poisons the read: the RAM leg refuses for
    want of covers and the healthy stage copy serves pinned and
    digest-verified under its own material and lifetime."""
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob = _wire(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, wire)
    ram_root, ram = _promote_ram(tmp_path, {'w': staged})
    ram['w'].write_bytes(blob[:-1] + bytes([blob[-1] ^ 0xFF]))
    _announce(tmp_path)
    rows = {'w': (wire, staged, ram['w'])}
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover = _hex64(f"mover-{tmp_path}")
    _pb_queue(tmp_path, pool_mod, consumer)
    _pb_publish(rl, map_mod, tmp_path / 'residency', root, consumer, mover,
                MANIFEST, {residency_map_key(str(wire), 0): (wire, staged)})
    map_path = _write_map(tmp_path, rows, ram_root=ram_root, leads=[mover])
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    activate_staged_tier_policy("ram,ssd")
    resolver = residency_resolver()
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob and digest == cell['record']['blob_sha256']
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert report['serving_tiers'][-1]['serving_tier'] == 'stage'
    assert _pins_live(tmp_path, consumer) == []


# -- strict activation payloads, pinned ---------------------------------------

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
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch, {'a': (path, staged, None)})
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
    assert str(path) not in opened
    assert resolver.report()['serving_tiers'][-1]['pin_id']
    assert _pins_live(tmp_path, consumer) == []


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
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch, {'e': (path, _stage_whole(root, path), None)})
    with prefetch_exact_activation_cache_entries(
            [ref], max_tensor_bytes=nbytes, expected_session="strict-tier-session",
            release_file_pages=False) as window:
        assert torch.equal(window._tensors[ref], tensor)
    assert resolver.report()['serving_tiers'][-1]['pin_id']
    assert _pins_live(tmp_path, consumer) == []


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


# -- lease mechanics: duplicates, fork guard ----------------------------------

def test_duplicate_acquire_token_adopts_one_ref(tmp_path, monkeypatch):
    """Same token twice adopts one ref; two exits release exactly once."""
    require_prismabuild_sdk()
    from prismaquant.staged_lease import LeaseWindow, covers_for_leads
    _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    import prismabuild.pool as pool_mod
    import prismabuild.residency_map as map_mod
    import prismabuild.reader_lease as rl
    queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
    blob = b"duplicate-token-bytes-0123456789"
    declared = tmp_path / 'pool' / 'd.bin'
    declared.parent.mkdir(parents=True, exist_ok=True)
    declared.write_bytes(blob)
    staged = stage / 'd.bin'
    staged.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    mover = _hex64(f"mover-{tmp_path}")
    root = tmp_path / 'residency'
    key = residency_map_key(str(declared), 0)
    _pb_publish(rl, map_mod, root, stage, consumer, mover, MANIFEST,
                {key: (declared, staged)})
    _launch_env(monkeypatch, consumer)
    monkeypatch.setenv(ENV_VAR, str(tmp_path / 'residency' / 'd.map.json'))
    spec = {"tier_id": STAGE_TIER, "epoch": "",
            "covers": covers_for_leads([mover], MANIFEST),
            "expected": {key: {"bytes": len(blob), "sha256": digest}},
            "span": {"start_bytes": 0, "end_bytes": len(blob)}}
    first = LeaseWindow(spec, acquire_token="token-1")
    second = LeaseWindow(spec, acquire_token="token-1")
    with first as one:
        with second as two:
            # Same token retries idempotently onto one shared ref.
            assert one._ref_id == two._ref_id
            fd, _serving = two.open(key)
            two.close_fd(fd)
            assert len(_pins_live(tmp_path, consumer)) == 1
        # Exiting the second window drops the shared ref (duplicate
        # adoption shares it; production mints one token per window, so
        # live windows never share).
    # The first window's exit is idempotent: the pin is already gone,
    # released exactly once.
    assert _pins_live(tmp_path, consumer) == []


def test_release_failure_retains_retry_state_then_releases_exactly(tmp_path, monkeypatch):
    """SDK ``False``-return path (tainted pin file makes ``release`` return
    ``False``, not an unlink exception): exit refuses loudly and retains
    full retry state — no silent strand, no marked release. Restoring the
    pin lets the retry release exactly."""
    require_prismabuild_sdk()
    from prismaquant.staged_lease import LeaseRefused, LeaseWindow, covers_for_leads
    _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    import prismabuild.pool as pool_mod
    import prismabuild.residency_map as map_mod
    import prismabuild.reader_lease as rl
    queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
    blob = b"release-failure-retry-bytes-0123456"
    declared = tmp_path / 'pool' / 'd.bin'
    declared.parent.mkdir(parents=True, exist_ok=True)
    declared.write_bytes(blob)
    staged = stage / 'd.bin'
    staged.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    mover = _hex64(f"mover-{tmp_path}")
    root = tmp_path / 'residency'
    key = residency_map_key(str(declared), 0)
    _pb_publish(rl, map_mod, root, stage, consumer, mover, MANIFEST,
                {key: (declared, staged)})
    _launch_env(monkeypatch, consumer)
    monkeypatch.setenv(ENV_VAR, str(tmp_path / 'residency' / 'd.map.json'))
    spec = {"tier_id": STAGE_TIER, "epoch": "",
            "covers": covers_for_leads([mover], MANIFEST),
            "expected": {key: {"bytes": len(blob), "sha256": digest}},
            "span": {"start_bytes": 0, "end_bytes": len(blob)}}
    # A filesystem with no transient horizon: the failed release is not
    # retried, so the exit refuses at once.
    import prismaquant.staged_lease as staged_lease_mod
    monkeypatch.setattr(staged_lease_mod, "release_retry_horizon_s",
                        lambda path, mountinfo=None: 0.0,
                        raising=False)
    window = LeaseWindow(spec, acquire_token="token-relfail")
    pin_path = (tmp_path / 'residency' / 'leases' / consumer)
    with window:
        assert len(_pins_live(tmp_path, consumer)) == 1
        saved = json.loads(
            (pin_path / f"{window._pin_id}.lease.json").read_text())
        (pin_path / f"{window._pin_id}.lease.json").write_text("tainted{")
        with pytest.raises(LeaseRefused, match="lease-release-failed"):
            window.__exit__(None, None, None)
        assert window._released is False
        assert len(_pins_live(tmp_path, consumer)) == 1
        (pin_path / f"{window._pin_id}.lease.json").write_text(
            json.dumps(saved, sort_keys=True) + "\n")
    assert _pins_live(tmp_path, consumer) == []



def _one_published_window(tmp_path, monkeypatch, token):
    """A published stage window over one blob, and its lease window."""
    from prismaquant.staged_lease import LeaseWindow, covers_for_leads
    _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    import prismabuild.pool as pool_mod
    import prismabuild.residency_map as map_mod
    import prismabuild.reader_lease as rl
    queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
    blob = b"release-transient-retry-bytes-0123"
    declared = tmp_path / 'pool' / 'd.bin'
    declared.parent.mkdir(parents=True, exist_ok=True)
    declared.write_bytes(blob)
    staged = stage / 'd.bin'
    staged.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    mover = _hex64(f"mover-{tmp_path}")
    root = tmp_path / 'residency'
    key = residency_map_key(str(declared), 0)
    _pb_publish(rl, map_mod, root, stage, consumer, mover, MANIFEST,
                {key: (declared, staged)})
    _launch_env(monkeypatch, consumer)
    monkeypatch.setenv(ENV_VAR, str(tmp_path / 'residency' / 'd.map.json'))
    spec = {"tier_id": STAGE_TIER, "epoch": "",
            "covers": covers_for_leads([mover], MANIFEST),
            "expected": {key: {"bytes": len(blob), "sha256": digest}},
            "span": {"start_bytes": 0, "end_bytes": len(blob)}}
    return rl, consumer, LeaseWindow(spec, acquire_token=token)


def test_a_transient_release_failure_is_retried_and_releases(tmp_path, monkeypatch, capsys):
    """R13 (556d7a803098) died in forward-002 on one ``False`` from
    ``release`` after hundreds of clean ones; the pin it left read and
    validated cleanly afterwards. Inside the mount's transient horizon a
    failed release is retried (release is idempotent), recorded with what
    the pin showed, and the window exits released."""
    require_prismabuild_sdk()
    import prismaquant.staged_lease as staged_lease_mod
    rl, consumer, window = _one_published_window(
        tmp_path, monkeypatch, "token-transient")
    monkeypatch.setattr(staged_lease_mod, "release_retry_horizon_s",
                        lambda path, mountinfo=None: 10.0,
                        raising=False)
    real_release = rl.release
    calls = []

    def flaky_release(*args, **kwargs):
        calls.append(args[2])
        if len(calls) == 1:
            return False
        return real_release(*args, **kwargs)

    monkeypatch.setattr(rl, "release", flaky_release)
    with window:
        assert len(_pins_live(tmp_path, consumer)) == 1
    assert window._released is True
    assert _pins_live(tmp_path, consumer) == []
    assert len(calls) == 2 and calls[0] == calls[1]
    said = capsys.readouterr().out
    assert "attempt 1 returned False" in said
    assert "this ref held" in said
    assert "retrying for up to" in said


def test_a_release_failing_past_the_horizon_refuses_with_what_the_pin_showed(
        tmp_path, monkeypatch, capsys):
    """A release that keeps failing past the horizon refuses as before --
    retry state kept, the ref still held -- and the refusal carries the
    pin's observed state, which PB's bare ``False`` does not."""
    require_prismabuild_sdk()
    from prismaquant.staged_lease import LeaseRefused
    import prismaquant.staged_lease as staged_lease_mod
    rl, consumer, window = _one_published_window(
        tmp_path, monkeypatch, "token-persistent")
    monkeypatch.setattr(staged_lease_mod, "release_retry_horizon_s",
                        lambda path, mountinfo=None: 0.3,
                        raising=False)
    real_release = rl.release
    monkeypatch.setattr(rl, "release", lambda *args, **kwargs: False)
    with window:
        with pytest.raises(LeaseRefused, match="this ref held"):
            window.__exit__(None, None, None)
        assert window._released is False
        assert len(_pins_live(tmp_path, consumer)) == 1
        said = capsys.readouterr().out
        assert "not retrying" in said
        assert said.count("[staged-lease] release of pin") >= 2
        monkeypatch.setattr(rl, "release", real_release)
    assert _pins_live(tmp_path, consumer) == []


def _mountinfo(tmp_path, rows):
    path = tmp_path / "mountinfo"
    path.write_text("".join(row + "\n" for row in rows))
    return str(path)


def test_the_release_horizon_is_the_nfs_mounts_major_timeout(tmp_path):
    """timeo (deciseconds) x (retrans + 1), read from the mount that holds
    the path: the horizon the NFS client itself treats as transient."""
    from prismaquant.staged_lease import release_retry_horizon_s
    info = _mountinfo(tmp_path, [
        "22 1 0:21 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw",
        "90 22 0:55 / /mnt/shared rw,relatime shared:40 - nfs4 "
        "dl380g10:/export rw,vers=4.2,hard,proto=tcp,timeo=600,retrans=2,sec=sys",
        "91 22 0:56 / /mnt/shared/other rw,relatime - tmpfs tmpfs rw,size=1024k",
    ])
    assert release_retry_horizon_s(
        "/mnt/shared/prismabuild-fleet/pb-queue", info) == 180.0
    assert release_retry_horizon_s("/mnt/shared", info) == 180.0
    # The longest mount point wins, and a local filesystem is not transient.
    assert release_retry_horizon_s("/mnt/shared/other/q", info) == 0.0
    assert release_retry_horizon_s("/home/rob/q", info) == 0.0
    # A prefix that is not a path component does not match.
    assert release_retry_horizon_s("/mnt/sharedx/q", info) == 0.0


def test_the_release_horizon_is_zero_without_the_nfs_options(tmp_path):
    from prismaquant.staged_lease import release_retry_horizon_s
    info = _mountinfo(tmp_path, [
        "22 1 0:21 / / rw - ext4 /dev/sda1 rw",
        "90 22 0:55 / /mnt/shared rw - nfs4 dl380g10:/export rw,vers=4.2,hard",
    ])
    assert release_retry_horizon_s("/mnt/shared/q", info) == 0.0
    assert release_retry_horizon_s("/mnt/shared/q",
                                   str(tmp_path / "missing")) == 0.0

def test_forked_child_window_use_refused_loudly(tmp_path, monkeypatch):
    """Every window operation in a forked child raises: open (would outlive
    the parent's release), close, and exit. The pin stays intact and the
    parent's exact release still unlinks it."""
    require_prismabuild_sdk()
    from prismaquant.staged_lease import LeaseWindow, covers_for_leads
    _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    import prismabuild.pool as pool_mod
    import prismabuild.residency_map as map_mod
    import prismabuild.reader_lease as rl
    queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
    blob = b"fork-guard-bytes-0123456789abcdef"
    declared = tmp_path / 'pool' / 'd.bin'
    declared.parent.mkdir(parents=True, exist_ok=True)
    declared.write_bytes(blob)
    staged = stage / 'd.bin'
    staged.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    mover = _hex64(f"mover-{tmp_path}")
    root = tmp_path / 'residency'
    key = residency_map_key(str(declared), 0)
    _pb_publish(rl, map_mod, root, stage, consumer, mover, MANIFEST,
                {key: (declared, staged)})
    _launch_env(monkeypatch, consumer)
    monkeypatch.setenv(ENV_VAR, str(tmp_path / 'residency' / 'd.map.json'))
    spec = {"tier_id": STAGE_TIER, "epoch": "",
            "covers": covers_for_leads([mover], MANIFEST),
            "expected": {key: {"bytes": len(blob), "sha256": digest}},
            "span": {"start_bytes": 0, "end_bytes": len(blob)}}
    window = LeaseWindow(spec, acquire_token="token-2")
    with window:
        assert len(_pins_live(tmp_path, consumer)) == 1
        real_getpid = os.getpid
        monkeypatch.setattr(os, "getpid", lambda: real_getpid() + 100000)
        with pytest.raises(RuntimeError, match="forked child"):
            window.open(key)
        with pytest.raises(RuntimeError, match="forked child"):
            window.__exit__(None, None, None)
        assert len(_pins_live(tmp_path, consumer)) == 1
        monkeypatch.setattr(os, "getpid", real_getpid)
    assert _pins_live(tmp_path, consumer) == []


def test_forked_child_reader_payload_refused(tmp_path, monkeypatch):
    """A forked child cannot read through inherited bound descriptors;
    header-only metadata stays available, payload refuses loudly."""
    path, tensors = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch, {'s': (path, staged, None)})
    real_getpid = os.getpid
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        from safetensors import safe_open
        with safe_open(str(path), framework='pt') as reference:
            assert torch.equal(
                reader.get_tensor('f32').view(torch.uint8),
                reference.get_tensor('f32').view(torch.uint8))
        monkeypatch.setattr(os, "getpid", lambda: real_getpid() + 100000)
        assert sorted(reader.keys()) == sorted(tensors)
        with pytest.raises(RuntimeError, match="forked child"):
            reader.get_tensor('bf16')
        with pytest.raises(RuntimeError, match="forked child"):
            reader.get_slice('f32')
        monkeypatch.setattr(os, "getpid", real_getpid)
    assert resolver.report()['bytes_from_pool'] == 0
    assert _pins_live(tmp_path, consumer) == []


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
                                        "--adjoint-slice", "a",
                                        "--adjoint-slice-sha256", "0" * 64,
                                        "--output-root", "o"])
    assert parsed.allowed_tiers == "ram"
    with pytest.raises(ValueError, match="TIER-04"):
        parse_allowed_tiers("pool")

    from stage_a_spool_spec import with_spool
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(with_spool(
        {"container": {"image": "sha256:" + "0" * 64}, "env": {}})))
    monkeypatch.setattr(dispatch_joint_quanta, "SPEC_PATH", spec)
    plan_path = tmp_path / "plan.json"
    from stage_a_spool_spec import stage_a_plan
    plan_path.write_text(json.dumps(stage_a_plan(
        tmp_path, output_root=str(tmp_path / "campaign-root"))))
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
        "output_space": {"root": "layer-quanta/layer-001"}}
    # PQ #993: the row is bound to its own stage-A slice, written first.
    from prismaquant.joint_adjoint_slices import stage_a_slice, write_adjoint_slice
    from prismaquant.joint_layer_quanta import adjoint_binding_fields
    from test_stage_b_band_binding import synthetic_receipt
    adjoint_slice = stage_a_slice(synthetic_receipt(
        plan_sha256=campaign["plan_sha256"], prepared_sha256=campaign["prepared_sha256"],
        scope=campaign["scope"], num_layers=3, stride=1), 1)
    slice_path = tmp_path / "adjoint-slices" / "layer-001.json"
    write_adjoint_slice(slice_path, adjoint_slice, layer=1)
    record["adjoint"] = {"checkpoint_boundary": 2, "chain_layers": [],
                         **adjoint_binding_fields(adjoint_slice, slice_path=str(slice_path))}
    record["identity_sha256"] = canonical_json_sha256(record, where="fixture")
    record_path = tmp_path / "layer-001.json"
    record_path.write_text(json.dumps(record))
    argv = quantum_argv(record, record_path=record_path,
                        output_root=tmp_path / "campaign-root")
    inner = argv[argv.index("--") + 1:]
    payload = inner[inner.index("prismaquant.joint_cost_quantum") - 2:]
    assert payload[:3] == ["python3", "-m", "prismaquant.joint_cost_quantum"]
    assert payload[payload.index("--allowed-tiers") + 1] == STAGED_ALLOWED_TIERS
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


def test_lease_pin_module_reports_approved_commit():
    from prismaquant.staged_lease import (
        PINNED_SDK_COMMIT, PB_READER_LEASE_PIN_COMMIT)
    assert PINNED_SDK_COMMIT == PB_READER_LEASE_PIN_COMMIT
    assert PINNED_SDK_COMMIT == "461728e4dcc08123d5fdb410eb2f18772fdb3fe0"


# -- window enter/exit contract: single-shot, no leaks ------------------------

def _window_fixture(tmp_path, monkeypatch, blob=b"window-contract-bytes-00112233"):
    require_prismabuild_sdk()
    from prismaquant.staged_lease import LeaseWindow, covers_for_leads
    _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    import prismabuild.pool as pool_mod
    import prismabuild.residency_map as map_mod
    import prismabuild.reader_lease as rl
    queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
    declared = tmp_path / 'pool' / 'd.bin'
    declared.parent.mkdir(parents=True, exist_ok=True)
    declared.write_bytes(blob)
    staged = stage / 'd.bin'
    staged.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    mover = _hex64(f"mover-{tmp_path}")
    root = tmp_path / 'residency'
    key = residency_map_key(str(declared), 0)
    _pb_publish(rl, map_mod, root, stage, consumer, mover, MANIFEST,
                {key: (declared, staged)})
    monkeypatch.setenv("PRISMABUILD_ACTION_KEY", consumer)
    monkeypatch.setenv("PRISMABUILD_ACTION_NONCE", LAUNCH_NONCE)
    monkeypatch.setenv("PRISMABUILD_ACTION_SCOPE", LAUNCH_SCOPE)
    monkeypatch.setenv(ENV_VAR, str(tmp_path / 'residency' / 'd.map.json'))
    spec = {"tier_id": STAGE_TIER, "epoch": "",
            "covers": covers_for_leads([mover], MANIFEST),
            "expected": {key: {"bytes": len(blob), "sha256": digest}},
            "span": {"start_bytes": 0, "end_bytes": len(blob)}}
    return spec, key, consumer, staged, blob


def test_open_refusal_after_acquire_releases_exactly(tmp_path, monkeypatch):
    """Refusal between acquisition and payload setup leaks nothing: the
    acquired pin releases exactly and no pool byte is read."""
    from prismaquant.staged_lease import LeaseWindow
    spec, key, consumer, staged, blob = _window_fixture(tmp_path, monkeypatch)
    window = LeaseWindow(spec, acquire_token="token-open-fail")
    with window:
        staged.unlink()  # released between acquire and open
        with pytest.raises(LeaseRefused) as excinfo:
            window.open(key)
        assert excinfo.value.kind == "integrity"
    assert _pins_live(tmp_path, consumer) == []


def test_released_window_reuse_and_nesting_refuse(tmp_path, monkeypatch):
    """A released manager is never reused and a live one never nests:
    both refuse as programming errors instead of silent reacquisition."""
    from prismaquant.staged_lease import LeaseWindow
    spec, key, consumer, _staged, _blob = _window_fixture(tmp_path, monkeypatch)
    window = LeaseWindow(spec, acquire_token="token-reuse")
    with window:
        with pytest.raises(RuntimeError, match="nested|reentrant|already entered"):
            with window:
                pass
    assert _pins_live(tmp_path, consumer) == []
    with pytest.raises(RuntimeError, match="re-enter|released|reuse"):
        with window:
            pass
    assert _pins_live(tmp_path, consumer) == []


# -- authoritative production discovery: sealed source tree -------------------

def _build_coherent_tree(tmp_path):
    """Copy the pinned installed package into a generation-style tree.

    Build-only use of the install location: the servant under test is the
    ``gen/src`` tree via the authoritative helper root, never the install.
    """
    require_prismabuild_sdk()
    import shutil
    import prismabuild.reader_lease as installed_rl
    src_pkg = Path(installed_rl.__file__).resolve().parent
    gen = tmp_path / 'gen-root'
    tree_pkg = gen / 'src' / 'prismabuild'
    tree_pkg.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_pkg, tree_pkg,
                    ignore=shutil.ignore_patterns('__pycache__'),
                    dirs_exist_ok=True)
    return gen


def _purge_pb_modules():
    import sys
    saved_mods = {k: v for k, v in sys.modules.items()
                  if k == "prismabuild" or k.startswith("prismabuild.")}
    for k in list(saved_mods):
        del sys.modules[k]
    saved_path = list(sys.path)
    return saved_mods, saved_path


def _restore_pb_modules(saved_mods, saved_path):
    import sys
    for k in [k for k in sys.modules
              if k == "prismabuild" or k.startswith("prismabuild.")]:
        del sys.modules[k]
    sys.modules.update(saved_mods)
    sys.path[:] = saved_path


def test_production_tree_discovery_serves_and_releases(tmp_path, monkeypatch):
    """Authoritative discovery: a coherent sealed source tree serves pinned
    bytes and releases exactly, with every PB submodule under one root."""
    require_prismabuild_sdk()
    import sys
    from prismaquant.staged_lease import (
        HELPER_ROOT_ENV_VAR, LeaseWindow, _package_dir_of,
        clear_injected_sdk_for_tests, covers_for_leads, set_lease_helper_root)
    gen = _build_coherent_tree(tmp_path)
    saved_mods, saved_path = _purge_pb_modules()
    clear_injected_sdk_for_tests()
    set_lease_helper_root(None)
    monkeypatch.setenv(HELPER_ROOT_ENV_VAR, str(gen))
    sys.path.insert(0, str(gen / 'src'))
    try:
        import prismabuild.reader_lease as rl
        import prismabuild.pool as pool_mod
        import prismabuild.residency_map as map_mod
        tree_root = (gen / 'src' / 'prismabuild').resolve()
        assert Path(rl.__file__).resolve().is_relative_to(tree_root)
        assert _package_dir_of(rl).resolve() == tree_root
        consumer = _hex64(f"consumer-{tmp_path}")
        queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
        blob = b"production-tree-discovery-bytes-0011"
        declared = tmp_path / 'pool' / 'd.bin'
        declared.parent.mkdir(parents=True, exist_ok=True)
        declared.write_bytes(blob)
        staged = stage / 'd.bin'
        staged.write_bytes(blob)
        digest = hashlib.sha256(blob).hexdigest()
        mover = _hex64(f"mover-{tmp_path}")
        root = tmp_path / 'residency'
        key = residency_map_key(str(declared), 0)
        _pb_publish(rl, map_mod, root, stage, consumer, mover, MANIFEST,
                    {key: (declared, staged)})
        _launch_env(monkeypatch, consumer)
        monkeypatch.setenv(ENV_VAR, str(tmp_path / 'residency' / 'd.map.json'))
        spec = {"tier_id": STAGE_TIER, "epoch": "",
                "covers": covers_for_leads([mover], MANIFEST),
                "expected": {key: {"bytes": len(blob), "sha256": digest}},
                "span": {"start_bytes": 0, "end_bytes": len(blob)}}
        window = LeaseWindow(spec, acquire_token="token-tree-ok")
        with window:
            assert Path(
                window._pool_mod.__file__).resolve().is_relative_to(tree_root)
            fd, serving = window.open(key)
            try:
                got = os.pread(fd, len(blob), 0)
            finally:
                window.close_fd(fd)
            assert got == blob
            assert serving.get("pin_id")
        assert _pins_live(tmp_path, consumer) == []
    finally:
        _restore_pb_modules(saved_mods, saved_path)


def test_divergent_preimport_refuses_before_any_pin(tmp_path, monkeypatch):
    """A divergent ``pool``/``residency_map`` preimport refuses at enter,
    before any pin file exists — no strand, no teardown needed."""
    require_prismabuild_sdk()
    import sys
    import types
    from prismaquant.staged_lease import (
        HELPER_ROOT_ENV_VAR, LeaseRefused, LeaseWindow,
        clear_injected_sdk_for_tests, covers_for_leads, set_lease_helper_root)
    gen = _build_coherent_tree(tmp_path)
    saved_mods, saved_path = _purge_pb_modules()
    clear_injected_sdk_for_tests()
    set_lease_helper_root(None)
    monkeypatch.setenv(HELPER_ROOT_ENV_VAR, str(gen))
    sys.path.insert(0, str(gen / 'src'))
    try:
        import prismabuild.reader_lease as rl
        import prismabuild.pool as pool_mod
        import prismabuild.residency_map as map_mod
        consumer = _hex64(f"consumer-{tmp_path}")
        queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
        blob = b"divergent-preimport-refusal-bytes-22"
        declared = tmp_path / 'pool' / 'd.bin'
        declared.parent.mkdir(parents=True, exist_ok=True)
        declared.write_bytes(blob)
        staged = stage / 'd.bin'
        staged.write_bytes(blob)
        digest = hashlib.sha256(blob).hexdigest()
        mover = _hex64(f"mover-{tmp_path}")
        root = tmp_path / 'residency'
        key = residency_map_key(str(declared), 0)
        _pb_publish(rl, map_mod, root, stage, consumer, mover, MANIFEST,
                    {key: (declared, staged)})
        _launch_env(monkeypatch, consumer)
        monkeypatch.setenv(ENV_VAR, str(tmp_path / 'residency' / 'd.map.json'))
        spec = {"tier_id": STAGE_TIER, "epoch": "",
                "covers": covers_for_leads([mover], MANIFEST),
                "expected": {key: {"bytes": len(blob), "sha256": digest}},
                "span": {"start_bytes": 0, "end_bytes": len(blob)}}
        for stub_name, token in (("prismabuild.pool", "token-div-pool"),
                                 ("prismabuild.residency_map", "token-div-map")):
            stub = types.ModuleType(stub_name)
            stub.__file__ = "/elsewhere/prismabuild/" + stub_name.split(".")[-1] + ".py"
            real = sys.modules.get(stub_name)
            sys.modules[stub_name] = stub
            try:
                window = LeaseWindow(spec, acquire_token=token)
                with pytest.raises(LeaseRefused, match="divergent"):
                    with window:
                        pass
                assert not window._entered
                assert _pins_live(tmp_path, consumer) == []
            finally:
                if real is not None:
                    sys.modules[stub_name] = real
                else:
                    del sys.modules[stub_name]
    finally:
        _restore_pb_modules(saved_mods, saved_path)


# -- strict checkpoint shared-state payloads, pinned -------------------------

def _write_checkpoint(tmp_path):
    from prismaquant.joint_adjoint_checkpoints import (
        adjoint_space, write_adjoint_checkpoint)
    from prismaquant.sensitivity_probe import SharedStateCotangents
    tensor = torch.randn(3, 4)
    state = SharedStateCotangents().state_dict()
    record = write_adjoint_checkpoint(
        adjoint_space(tmp_path), boundary=5,
        session={"generation": "g" * 32, "kind": "adjoint_checkpoint"},
        cotangents={(0, 0): tensor},
        shared_adjoint={(0, 0): state},
        shared_pass={0: {"captured": None}})
    return record, tensor, state


def _states_equal(left, right):
    if set(left) != set(right):
        return False
    for key in left:
        first, second = left[key], right[key]
        if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
            if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
                return False
            if not torch.equal(first, second):
                return False
        elif first != second:
            return False
    return True


def _checkpoint_manifest_path(record):
    """Where the writer published ``checkpoint.json``: beside ``entries/``."""
    entries = Path(record["activation_entries"][0]["path"]).parent
    return str(entries.parent / "checkpoint.json")


def _stage_checkpoint_entries(tmp_path, monkeypatch, record, *,
                              stage_manifest=True):
    """Stage every file the loader reads -- ``checkpoint.json`` plus BOTH
    entry classes (activation + shared-state) -- with real PB writers
    under one mover; returns (resolver, consumer, staged_paths).
    ``stage_manifest=False`` leaves ``checkpoint.json`` unstaged."""
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover = _hex64(f"mover-{tmp_path}")
    queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
    root = tmp_path / 'residency'
    entries, rows, paths = {}, {}, []
    manifest = ([{"path": _checkpoint_manifest_path(record)}]
                if stage_manifest else [])
    for entry in manifest + record["activation_entries"] \
            + record["shared_state_entries"]:
        declared = Path(entry["path"])
        paths.append(str(declared))
        staged = stage / declared.name
        staged.write_bytes(declared.read_bytes())
        key = residency_map_key(str(declared), 0)
        entries[key] = (declared, staged)
        rows[declared.name] = (declared, staged, None)
    _pb_publish(rl, map_mod, root, stage, consumer, mover, MANIFEST, entries)
    map_path = _write_map(tmp_path, rows, leads=[mover])
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    activate_staged_tier_policy("ram,ssd")
    return residency_resolver(), consumer, paths


@pytest.mark.parametrize("local_scratch", [False, True])
def test_strict_checkpoint_roundtrip_pinned_never_opens_pool(tmp_path, monkeypatch, local_scratch):
    from prismaquant.joint_adjoint_checkpoints import (
        adjoint_space, load_adjoint_checkpoint)
    record, tensor, state = _write_checkpoint(tmp_path)
    resolver, consumer, paths = _stage_checkpoint_entries(
        tmp_path, monkeypatch, record)
    opened = []
    real_open = os.open

    def counting(target, *args, **kwargs):
        opened.append(os.fspath(target))
        return real_open(target, *args, **kwargs)

    monkeypatch.setattr(os, "open", counting)
    arena = None
    def factory(entries):
        nonlocal arena
        from prismaquant.perturbed_x_cache import ExactCotangentScratch
        arena = ExactCotangentScratch(entries, directory=tmp_path, max_bytes=1 << 20)
        return arena
    try:
        cotangents, shared_adjoint, shared_pass = load_adjoint_checkpoint(
            adjoint_space(tmp_path), record,
            cotangent_factory=factory if local_scratch else None)
        actual = cotangents[(0, 0)]
    finally:
        if arena is not None:
            arena.close()
    assert torch.equal(actual, tensor)
    assert _states_equal(shared_adjoint[(0, 0)], state)
    assert shared_pass == {0: {"captured": None}}
    assert not any(opened_path in paths for opened_path in opened)
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert any(row.get('pin_id') for row in report['serving_tiers'])
    # checkpoint.json is a declared staged read, never a pool read.
    manifest_path = _checkpoint_manifest_path(record)
    assert any(row.get('path') == manifest_path and row.get('pin_id')
               for row in report['serving_tiers']), report['serving_tiers']
    assert _pins_live(tmp_path, consumer) == []


def test_strict_checkpoint_manifest_unstaged_refuses(tmp_path, monkeypatch):
    """A readset that stages every entry but not ``checkpoint.json``
    refuses before the loader reads the manifest from the pool."""
    from prismaquant.joint_adjoint_checkpoints import (
        adjoint_space, load_adjoint_checkpoint)
    record, _tensor, _state = _write_checkpoint(tmp_path)
    resolver, _consumer, _paths = _stage_checkpoint_entries(
        tmp_path, monkeypatch, record, stage_manifest=False)
    with pytest.raises(TierPolicyRefused):
        load_adjoint_checkpoint(adjoint_space(tmp_path), record)
    assert resolver.report()['bytes_from_pool'] == 0


def test_strict_checkpoint_shared_state_altered_refuses(tmp_path, monkeypatch):
    from prismaquant.joint_adjoint_checkpoints import (
        adjoint_space, load_adjoint_checkpoint)
    record, _tensor, _state = _write_checkpoint(tmp_path)
    resolver, consumer, _paths = _stage_checkpoint_entries(
        tmp_path, monkeypatch, record)
    staged = tmp_path / 'stage' / 'prewarm' / [
        Path(e["path"]).name for e in record["shared_state_entries"]][0]
    blob = staged.read_bytes()
    staged.write_bytes(blob[:-1] + bytes([blob[-1] ^ 0xFF]))
    with pytest.raises(LeaseRefused) as excinfo:
        load_adjoint_checkpoint(adjoint_space(tmp_path), record)
    assert excinfo.value.kind == "integrity"
    assert resolver.report()['bytes_from_pool'] == 0
    # Only what was read and verified before the altered pickle counts as
    # stage bytes: checkpoint.json and the cotangent entries (PQ #1026
    # counts exact entries). None of the altered bytes do.
    manifest_bytes = Path(_checkpoint_manifest_path(record)).stat().st_size
    entry_bytes = sum(Path(entry["path"]).stat().st_size
                      for entry in record["activation_entries"])
    assert resolver.report()['bytes_from_stage'] == manifest_bytes + entry_bytes


def test_strict_checkpoint_shared_state_unmapped_refuses(tmp_path, monkeypatch):
    from prismaquant.joint_adjoint_checkpoints import (
        adjoint_space, load_adjoint_checkpoint)
    record, _tensor, _state = _write_checkpoint(tmp_path)
    _strict(monkeypatch, _write_map(tmp_path, {}))
    activate_staged_tier_policy("ram,ssd")
    with pytest.raises(TierPolicyRefused):
        load_adjoint_checkpoint(adjoint_space(tmp_path), record)


def test_lease_helper_reads_authoritative_env_automatically(tmp_path, monkeypatch):
    """Production discovery: the PB-injected PRISMABUILD_READER_HELPER_ROOT
    is read with no explicit setter and no user knob."""
    from prismaquant.staged_lease import (
        HELPER_ROOT_ENV_VAR, lease_helper_root)
    assert lease_helper_root() is None
    monkeypatch.setenv(HELPER_ROOT_ENV_VAR, str(tmp_path / 'gen-root'))
    assert lease_helper_root() == str(tmp_path / 'gen-root')


def test_lease_helper_env_without_helper_refuses(tmp_path, monkeypatch):
    """An authoritatively-named root that disagrees with the loaded SDK
    refuses as divergent: two trees must never mix. Clear refusal, zero
    pool — the installed reviewed dependency stays the only servant."""
    from prismaquant.staged_lease import HELPER_ROOT_ENV_VAR
    _pb()
    path, _ = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'s': (path, staged, None)}))
    monkeypatch.setenv(HELPER_ROOT_ENV_VAR, str(tmp_path / 'no-such-root'))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        with pytest.raises(LeaseRefused, match="lease-helper-divergent"):
            reader.get_tensor('f32')
    assert resolver.report()['bytes_from_pool'] == 0


def test_stage_epoch_convention_is_exact_absence(tmp_path, monkeypatch):
    """A stage-tier fragment carrying an epoch does NOT match epoch "":
    the empty string is absence, never a wildcard (Q4)."""
    from prismaquant.staged_lease import (
        LeaseRefused, LeaseWindow, covers_for_leads)
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover = _hex64(f"mover-{tmp_path}")
    queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
    blob = b"epoch-convention-bytes-0123456789"
    declared = tmp_path / 'pool' / 'd.bin'
    declared.parent.mkdir(parents=True, exist_ok=True)
    declared.write_bytes(blob)
    staged = stage / 'd.bin'
    staged.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    root = tmp_path / 'residency'
    key = residency_map_key(str(declared), 0)
    map_mod.write_fragment(root, {
        "schema": map_mod.RESIDENCY_MAP_FRAGMENT_SCHEMA_V1,
        "consumer_action_key": consumer, "mover_action_key": mover,
        "tier_id": STAGE_TIER, "stage_root": str(stage),
        "manifest_sha256": MANIFEST, "epoch": "stage-epoch-01",
        "entries": {key: {"stage_path": str(staged), "bytes": len(blob),
                          "sha256": digest, "offset": 0}}})
    rl.write_material(
        root, consumer_action_key=consumer, mover_action_key=mover,
        tier_id=STAGE_TIER, stage_root=str(stage), manifest_sha256=MANIFEST,
        generation=rl.mint_generation(),
        entries={key: {"stage_path": str(staged), "bytes": len(blob),
                       "sha256": digest, "file_id": rl.stat_identity(str(staged))}})
    _launch_env(monkeypatch, consumer)
    monkeypatch.setenv(ENV_VAR, str(tmp_path / 'residency' / 'd.map.json'))
    spec = {"tier_id": STAGE_TIER, "epoch": "",
            "covers": covers_for_leads([mover], MANIFEST),
            "expected": {key: {"bytes": len(blob), "sha256": digest}},
            "span": {"start_bytes": 0, "end_bytes": len(blob)}}
    window = LeaseWindow(spec, acquire_token="token-epoch")
    with pytest.raises(LeaseRefused) as excinfo:
        with window:
            pass
    assert excinfo.value.kind == "availability"
    assert "stale-epoch" in str(excinfo.value)


def test_equal_sized_files_never_serve_each_others_bytes(tmp_path, monkeypatch):
    """Known PB pin-ID collision (two equal-sized files, same covers,
    offset 0): PQ must serve byte-correct data or refuse with integrity —
    never wrong bytes, never pool. Holds before and after the PB fix."""
    from prismaquant.staged_lease import LeaseRefused, LeaseWindow, covers_for_leads
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover = _hex64(f"mover-{tmp_path}")
    queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
    pool = tmp_path / 'pool'
    pool.mkdir(parents=True, exist_ok=True)
    blobs = {name: bytes([seed]) * 32 for name, seed in (("a", 7), ("b", 9))}
    keys = {}
    staged_of = {}
    for name, blob in blobs.items():
        declared = pool / f"{name}.bin"
        declared.write_bytes(blob)
        staged = stage / f"{name}.bin"
        staged.write_bytes(blob)
        staged_of[name] = staged
        keys[name] = (residency_map_key(str(declared), 0), declared, blob)
    root = tmp_path / 'residency'
    frag_entries, mat_entries = {}, {}
    for name, (key, _declared, blob) in keys.items():
        digest = hashlib.sha256(blob).hexdigest()
        frag_entries[key] = {"stage_path": str(staged_of[name]),
                             "bytes": len(blob), "sha256": digest, "offset": 0}
        mat_entries[key] = dict(frag_entries[key],
                                file_id=rl.stat_identity(str(staged_of[name])))
    map_mod.write_fragment(root, {
        "schema": map_mod.RESIDENCY_MAP_FRAGMENT_SCHEMA_V1,
        "consumer_action_key": consumer, "mover_action_key": mover,
        "tier_id": STAGE_TIER, "stage_root": str(stage),
        "manifest_sha256": MANIFEST, "entries": frag_entries})
    rl.write_material(
        root, consumer_action_key=consumer, mover_action_key=mover,
        tier_id=STAGE_TIER, stage_root=str(stage), manifest_sha256=MANIFEST,
        generation=rl.mint_generation(), entries=mat_entries)
    _launch_env(monkeypatch, consumer)
    monkeypatch.setenv(ENV_VAR, str(tmp_path / 'residency' / 'd.map.json'))

    def read_all(name):
        key, _declared, blob = keys[name]
        digest = hashlib.sha256(blob).hexdigest()
        window = LeaseWindow(
            {"tier_id": STAGE_TIER, "epoch": "",
             "covers": covers_for_leads([mover], MANIFEST),
             "expected": {key: {"bytes": len(blob), "sha256": digest}},
             "span": {"start_bytes": 0, "end_bytes": len(blob)}},
            acquire_token=f"token-{name}")
        with window:
            fd, _serving = window.open(key)
            try:
                got = os.pread(fd, len(blob), 0)
            finally:
                window.close_fd(fd)
        return got

    first = read_all("a")
    assert first == blobs["a"]
    try:
        second = read_all("b")
    except LeaseRefused as refusal:
        # Pre-fix collision: the adopted pin lacks this key — fail closed.
        assert refusal.kind == "integrity"
    else:
        # Post-fix: distinct pins serve byte-correct data.
        assert second == blobs["b"]


# -- a declared range PrismaBuild has not moved YET (PQ #874) ----------------
#
# Measured failure these pin: PrismaBuild consumer action
# 2fd0de4dbefc50580448a883635fcba2a151c3a5dc743c1fd0b881cb61772d21, Stage A of
# the GLM-5.3-Flash joint-AURA campaign on sparklina, rc 1 after 574 s with no
# OOM, refusing in a gather worker at `residency_shard_reader.py:618`:
#
#   TierPolicyRefused: staged-tier-forbidden: readset-not-staged:
#     /mnt/shared/models/GLM-5.3-Flash-BF16/model-00087-of-00120.safetensors
#
# The refused range was declared and in flight, measured from the sealed
# artifacts (all times UTC; timeline at /home/rob/tmp/claude-main-20260920/
# pq-874-evidence/mover-timeline.md, which is a per-consumer SUBSET):
#
#   * PB's sealed data manifest for that action (43f40d18..., 36,600 entries,
#     gzipped in the CAS) declares that shard at entry indices 36447-36448,
#     in read-plan phase `forward-004`;
#   * its stage mover 6f4d3fa5097aeef7f47886d41d4578ff3a2a89167ac8642548919
#     638f27fcef4 was PUBLISHED at 20:55:57, 180 s BEFORE the refusal, and
#     claimed 7.832 s AFTER the consumer had died (root-verified, MCP451).
#
# So publication was not the lag -- the move was. The range was declared and
# unmoved, which is what `RANGE_UNCOVERED` now means, and the reader was
# flattening it into a terminal refusal on a *speculative* prefetch of layer 4
# whose failure then killed layer 0, the layer the run was working on.
#
# Two things are being pinned, and they are separate guards:
#
#   1. The resolver separates its four `None` returns, so an entry that covers
#      a span and fails a check never enters a wait -- re-asking cannot improve
#      evidence already in hand.
#   2. Declared-versus-undeclared is bound to PrismaBuild's own request
#      context: the claim row's `cas_root` plus its `residency.manifest_sha256`
#      name the sealed manifest blob, whose bytes are hashed and checked
#      against the digest this process bound before a single entry is adopted.
#      When that binding is unavailable the two are indistinguishable and
#      NOTHING waits -- not knowing is not a licence to wait.
#
# And where the waiting happens is itself part of the contract: in the thread
# that is about to submit the gather, never in a gather worker, so a ready
# current-layer read never queues behind workers sleeping on cold future
# layers.


def _seal_manifest(tmp_path, entries):
    """A real sealed data manifest in a real CAS, addressed by its digest.

    ``entries`` is ``[(declared Path, offset, bytes)]``. PrismaBuild gzips the
    manifest and addresses the blob by the digest of the COMPRESSED bytes
    (checked against the live campaign blob 43f40d18..., 1,576,940 stored,
    10,667,784 inflated, magic 1f8b), so this seals it the same way -- and it
    is a manifest ``prismabuild.core.validate_data_manifest`` accepts, v2
    ``read_plan`` and all, because that validator is what reads it back. A
    hand-shaped payload the real reader would refuse would make this whole
    fixture a fiction.

    Returns ``(cas_root, digest, size)``.
    """
    rows = [{"path": str(path), "offset": offset, "bytes": count,
             "sha256": None} for path, offset, count in entries]
    total = sum(row["bytes"] for row in rows)
    body = {
        "schema": "prismaquant.prismabuild.data_manifest.v2",
        "produced_by": {"tool": "tests/test_strict_reader_tier_enforcement.py"},
        "annotations": {},
        "mount_prefix": str(tmp_path),
        "entries": rows,
        "entry_count": len(rows),
        "total_bytes": total,
        "read_plan": {
            "phases": [{"name": "head", "entry_indices": list(range(len(rows))),
                        "bytes": total, "cumulative_bytes": total}],
            "read_bytes": total,
        },
    }
    raw = gzip.compress(json.dumps(body).encode("utf-8"), mtime=0)
    digest = hashlib.sha256(raw).hexdigest()
    cas_root = tmp_path / 'cas'
    blob = cas_root / 'blobs' / digest[:2] / digest
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(raw)
    return cas_root, digest, len(raw)


def _publish_readset_on_the_claim(tmp_path, consumer, cas_root, digest, size):
    """Add what PB publishes at publish time to the claim row this run reads.

    ``pool.py`` writes ``cas_root`` and the ``residency`` block into the pool
    item when the action is published, and the item moves ready -> claimed
    unchanged, so a live claim row carries both. ``_pb_queue`` writes only the
    identity fields the lease SDK matches; this adds the two the sealed
    readset is resolved from, leaving those fields exactly as they were.
    """
    path = tmp_path / 'claimed' / f'{consumer}.json'
    row = json.loads(path.read_text())
    row["cas_root"] = str(cas_root)
    row["residency"] = {"schema": "prismabuild.residency.v1",
                        "tier_id": STAGE_TIER,
                        "manifest_sha256": digest, "manifest_bytes": size}
    path.write_text(json.dumps(row))


def _mid_flight_fixture(tmp_path, monkeypatch, *, declared=None, seal=True):
    """The live shape: policy, lease context and claim live, the map empty.

    ``declared`` is what the SEALED MANIFEST names, ``[(path, offset, bytes)]``
    -- independently of what is staged, which is the whole point. ``seal=False``
    publishes no manifest blob at all, which is how an unbound readset is
    reached.

    Returns ``(resolver, consumer, digest, publish)``. ``publish(rows)`` does
    what a mover does when its leg lands: the REAL PB fragment and material
    writers, then the composed map replaced atomically (``os.replace``, so a
    poll never reads a torn map). ``rows`` is ``{name: (declared, staged)}``.
    """
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover = _hex64(f"mover-{tmp_path}")
    _queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
    cas_root, digest, size = _seal_manifest(tmp_path, declared or [])
    if seal:
        _publish_readset_on_the_claim(tmp_path, consumer, cas_root, digest, size)
    root = tmp_path / 'residency'
    live = root / 'residency.json'

    def publish(rows):
        _pb_publish(rl, map_mod, root, stage, consumer, mover, digest,
                    {residency_map_key(str(dec), 0): (dec, staged)
                     for dec, staged in rows.values()})
        fresh = _write_map(
            tmp_path, {name: (dec, staged, None)
                       for name, (dec, staged) in rows.items()},
            name='residency.next.json', manifest_sha256=digest,
            leads=[mover], stage_root=stage)
        os.replace(fresh, live)

    map_path = _write_map(tmp_path, {}, manifest_sha256=digest, leads=[mover],
                          stage_root=stage)
    assert map_path == live
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(digest)
    activate_staged_tier_policy("ram,ssd")
    return residency_resolver(), consumer, digest, publish


def _layer(path, names=('f32',)):
    """``_read_layer_to_device`` arguments for one shard, prefix ``layer.``."""
    model_to_shard = {f'layer.{name}': str(path) for name in names}
    model_to_ckpt = {f'layer.{name}': name for name in names}
    return model_to_shard, model_to_ckpt


def _read_layer(*shards):
    """Read one streamed layer spanning ``shards`` on the CPU."""
    model_to_shard, model_to_ckpt = {}, {}
    for index, path in enumerate(shards):
        for name in ('f32',):
            key = f'layer.{index}.{name}'
            model_to_shard[key] = str(path)
            model_to_ckpt[key] = name
    return layer_streaming._read_layer_to_device(
        'layer.', model_to_shard, model_to_ckpt, torch.float32,
        torch.device('cpu'))


def _whole_file(path):
    return [(path, 0, path.stat().st_size)]


def test_a_layer_read_waits_for_a_declared_range_staged_after_it_began(
        tmp_path, monkeypatch):
    """The campaign's shape: the read starts before the mover's leg lands."""
    path, _tensors = _shard(tmp_path)
    staged = _stage_whole(_stage_root(tmp_path), path)
    resolver, consumer, _digest, publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=_whole_file(path))
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "90")
    assert resolver.declared_readset()['state'] == 'bound'

    def mover_leg():
        time.sleep(2.0)
        publish({'s': (path, staged)})

    thread = threading.Thread(target=mover_leg, name='late-mover')
    thread.start()
    try:
        served = _read_layer(path)
    finally:
        thread.join(timeout=120)
        assert not thread.is_alive()

    from safetensors import safe_open
    with safe_open(str(path), framework='pt') as reference:
        want = reference.get_tensor('f32')
    got = served['layer.0.f32']
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))

    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert report['range_waits_served'] == 1
    assert report['range_waits_refused'] == 0
    assert report['range_wait_polls'] >= 1
    assert report['declared_readset']['state'] == 'bound'
    assert _pins_live(tmp_path, consumer) == []


def test_a_layer_read_does_not_wait_on_a_span_the_readset_never_declared(
        tmp_path, monkeypatch):
    """A span PB was never asked to stage refuses at once, whatever the bound.

    The manifest here declares only the shard's first 8 bytes -- its header
    length field -- so the tensor's own span is inside a declared FILE and
    still undeclared. Without the manifest binding this is indistinguishable
    from a mover that has not run, and would burn the whole bound.
    """
    path, _ = _shard(tmp_path)
    resolver, _consumer, _digest, _publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=[(path, 0, 8)])
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "600")

    started = time.monotonic()
    with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
        _read_layer(path)
    assert time.monotonic() - started < 30.0

    report = resolver.report()
    assert report['range_wait_polls'] == 0
    assert report['range_waits_served'] == 0
    assert report['range_waits_refused'] == 0
    assert report['bytes_from_pool'] == 0


def test_a_layer_read_does_not_wait_on_a_shard_the_readset_never_named(
        tmp_path, monkeypatch):
    """File-level: the manifest names another file entirely."""
    path, _ = _shard(tmp_path)
    other = path.with_name('model-00002-of-00002.safetensors')
    other.write_bytes(path.read_bytes())
    resolver, _consumer, _digest, _publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=_whole_file(other))
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "600")

    started = time.monotonic()
    with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
        _read_layer(path)
    assert time.monotonic() - started < 30.0
    assert resolver.report()['range_wait_polls'] == 0


def test_a_layer_read_does_not_wait_when_the_sealed_readset_is_unbound(
        tmp_path, monkeypatch):
    """No claim-row readset, no waiting -- and the report says why.

    Not knowing whether a range is declared is exactly the state in which
    waiting would be guessing. This keeps the pre-#874 behaviour and records
    the reason rather than silently doing either thing.
    """
    path, _ = _shard(tmp_path)
    resolver, _consumer, _digest, _publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=_whole_file(path), seal=False)
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "600")

    started = time.monotonic()
    with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
        _read_layer(path)
    assert time.monotonic() - started < 30.0

    state = resolver.report()['declared_readset']
    assert state['state'] == 'unbound'
    assert 'cas_root' in str(state['reason']) or 'claim row' in str(state['reason'])
    assert resolver.report()['range_wait_polls'] == 0


def test_a_layer_read_does_not_wait_on_an_entry_that_fails_a_check(
        tmp_path, monkeypatch):
    """Evidence in hand is not absence.

    A covered span whose staged copy is the wrong size refuses at once,
    however long the bound is. Without this guard every corrupt, stale or
    out-of-range entry would enter the wait -- the same conflation pointed
    the other way, since re-polling cannot improve any of them.
    """
    path, _ = _shard(tmp_path)
    staged = _stage_whole(_stage_root(tmp_path), path)
    resolver, _consumer, _digest, publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=_whole_file(path))
    publish({'s': (path, staged)})
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "600")
    staged.write_bytes(staged.read_bytes()[:-16])

    started = time.monotonic()
    with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
        _read_layer(path)
    assert time.monotonic() - started < 30.0

    report = resolver.report()
    assert report['range_wait_polls'] == 0
    assert report['range_waits_refused'] == 0
    assert report['fallback_count'] >= 1
    assert report['bytes_from_pool'] == 0


def test_a_layer_read_wait_ends_at_its_bound(tmp_path, monkeypatch):
    """A declared range nothing ever stages still refuses, and on time."""
    path, _ = _shard(tmp_path)
    resolver, _consumer, _digest, _publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=_whole_file(path))
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "3")

    started = time.monotonic()
    with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
        _read_layer(path)
    waited = time.monotonic() - started
    assert 2.0 <= waited < 60.0, f"waited {waited:.1f}s against a 3s bound"

    report = resolver.report()
    assert report['range_waits_refused'] == 1
    assert report['range_waits_served'] == 0
    assert report['range_wait_polls'] >= 1
    assert report['bytes_from_pool'] == 0


def test_a_zero_bound_refuses_on_the_first_uncovered_span(tmp_path, monkeypatch):
    """The pre-#874 behaviour stays reachable, and costs no poll."""
    path, _ = _shard(tmp_path)
    resolver, _consumer, _digest, _publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=_whole_file(path))
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "0")

    started = time.monotonic()
    with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
        _read_layer(path)
    assert time.monotonic() - started < 5.0
    assert resolver.report()['range_wait_polls'] == 0


def test_one_deadline_covers_a_layer_that_spans_several_cold_shards(
        tmp_path, monkeypatch):
    """Four cold shards wait the bound ONCE, not four times over.

    Per-span budgets would multiply the bound by the shard count and would be
    this reader choosing an order and a share -- a scheduler, which is
    PrismaBuild's job.
    """
    path, _ = _shard(tmp_path)
    shards = [path]
    for index in range(2, 5):
        extra = path.with_name(f'model-0000{index}-of-00004.safetensors')
        extra.write_bytes(path.read_bytes())
        shards.append(extra)
    declared = [row for shard in shards for row in _whole_file(shard)]
    resolver, _consumer, _digest, _publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=declared)
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "4")

    started = time.monotonic()
    with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
        _read_layer(*shards)
    waited = time.monotonic() - started
    assert 3.0 <= waited < 14.0, (
        f"{len(shards)} cold shards waited {waited:.1f}s against one 4s bound")
    assert resolver.report()['range_waits_refused'] == 1


def _wide_shard(tmp_path, name):
    """A shard with enough tensors to take the intra-layer fanout branch.

    ``_read_layer_to_device`` only uses ``_LAYER_READ_POOL`` when
    ``threads > 1 and total_tensors >= _LAYER_READ_MIN_TENSORS``. A test that
    misses either condition runs the serial path, never touches the pool, and
    would pass identically under an implementation that blocks inside a pool
    worker -- which is the property it exists to check.
    """
    pool = tmp_path / 'pool'
    pool.mkdir(parents=True, exist_ok=True)
    count = layer_streaming._LAYER_READ_MIN_TENSORS + 4
    tensors = {f't{index:02d}': torch.linspace(-1, 1, 64, dtype=torch.float32)
               .reshape(8, 8) + index for index in range(count)}
    path = pool / name
    save_file(tensors, str(path))
    return path, sorted(tensors)


def _read_wide_layer(path, names):
    model_to_shard = {f'layer.{n}': str(path) for n in names}
    model_to_ckpt = {f'layer.{n}': n for n in names}
    return layer_streaming._read_layer_to_device(
        'layer.', model_to_shard, model_to_ckpt, torch.float32,
        torch.device('cpu'))


def test_a_waiting_layer_read_holds_no_gather_worker(tmp_path, monkeypatch):
    """A cold layer read must not occupy the shared gather pool.

    ``_LAYER_READ_POOL`` is module-global, bounded and shared by every
    streamed layer read, so a worker sleeping on a cold future range is a
    worker the current layer's already-staged reads queue behind. Readiness
    is therefore decided in the submitting thread, before any slot is taken.

    Built so it can actually fail: both shards carry enough tensors to take
    the fanout branch, the pool is sized to exactly the number of chunks one
    cold layer splits into, and the executor is wrapped so the test can
    assert the submissions really happened. Under an implementation that
    waits inside a gather worker, every worker is asleep for the whole bound
    and the ready read cannot finish first.
    """
    cold, cold_names = _wide_shard(tmp_path, 'model-00001-of-00002.safetensors')
    ready, ready_names = _wide_shard(tmp_path, 'model-00002-of-00002.safetensors')
    staged_ready = _stage_whole(_stage_root(tmp_path), ready)
    resolver, _consumer, _digest, publish = _mid_flight_fixture(
        tmp_path, monkeypatch,
        declared=_whole_file(cold) + _whole_file(ready))
    publish({'r': (ready, staged_ready)})
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "30")
    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '2')

    submissions = []
    real_pool = ThreadPoolExecutor(max_workers=2,
                                   thread_name_prefix='gather-under-test')

    class _Counting:
        def submit(self, fn, *args, **kwargs):
            submissions.append(getattr(fn, '__name__', str(fn)))
            return real_pool.submit(fn, *args, **kwargs)

    monkeypatch.setattr(layer_streaming, '_layer_read_pool',
                        lambda threads: _Counting())

    results = []

    def cold_reader():
        with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
            _read_wide_layer(cold, cold_names)
        results.append(True)

    waiter = threading.Thread(target=cold_reader, name='cold-layer')
    waiter.start()
    try:
        time.sleep(2.0)
        began = time.monotonic()
        served = _read_wide_layer(ready, ready_names)
        elapsed = time.monotonic() - began
    finally:
        waiter.join(timeout=120)
        real_pool.shutdown(wait=False)

    assert results == [True]
    # The fanout branch really ran: without this a serial-path regression
    # would leave the assertion below passing for the wrong reason.
    assert len(submissions) >= 2, submissions
    assert elapsed < 10.0, (
        f"a staged layer read took {elapsed:.1f}s while a cold one waited "
        "its 30s bound: the wait is holding a gather worker")

    from safetensors import safe_open
    with safe_open(str(ready), framework='pt') as reference:
        for name in ready_names:
            assert torch.equal(served[f'layer.{name}'].view(torch.uint8),
                               reference.get_tensor(name).view(torch.uint8))
    assert resolver.report()['bytes_from_pool'] == 0


def test_the_sealed_readset_refuses_an_oversize_manifest_without_reading_it(
        tmp_path, monkeypatch):
    """A claim row may state any size; the ceiling is PrismaBuild's own.

    The row's ``manifest_bytes`` is an input. The same pool record carries
    ``detail.prewarm.manifest_bytes`` -- 1,244,988,662,830 on the live
    campaign, because it measures the payload the entries describe -- so a
    wrong field must not be able to spend a read and a hash on a terabyte.
    The blob here is tiny and is never opened: the refusal happens on the
    stated size, before any read.
    """
    require_prismabuild_sdk()
    from prismabuild.core import DATA_MANIFEST_MAX_BYTES

    path, _ = _shard(tmp_path)
    resolver, consumer, digest, _publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=_whole_file(path))
    assert resolver.declared_readset()['state'] == 'bound'

    opened = []
    real_open = open

    def watched(target, *args, **kwargs):
        opened.append(str(target))
        return real_open(target, *args, **kwargs)

    _publish_readset_on_the_claim(
        tmp_path, consumer, tmp_path / 'cas', digest,
        DATA_MANIFEST_MAX_BYTES + 1)
    bind_residency_manifest('0' * 63 + '1')      # drop the cached answer
    bind_residency_manifest(digest)
    import builtins
    builtins.open = watched                  # restored in the finally below
    try:
        state = resolver.declared_readset()
    finally:
        builtins.open = real_open

    assert state['state'] == 'unbound'
    assert str(DATA_MANIFEST_MAX_BYTES) in str(state['reason'])
    assert not [row for row in opened if digest in row], opened


def test_the_staged_range_wait_bound_must_be_finite(monkeypatch):
    """``inf`` is neither negative nor NaN and would remove the bound.

    A wait whose deadline never expires is not a longer bound, it is none,
    and the containment argument for this whole path rests on it ending.
    """
    monkeypatch.delenv(STAGED_RANGE_WAIT_ENV, raising=False)
    assert staged_range_wait_s() == STAGED_RANGE_WAIT_S
    for good, want in (("0", 0.0), ("12.5", 12.5), ("  7 ", 7.0)):
        monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, good)
        assert staged_range_wait_s() == want
    for bad in ("-1", "-0.5", "nan", "NaN", "inf", "Infinity", "-inf",
                "1e400", "abc", "5s"):
        monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, bad)
        with pytest.raises(ValueError, match=STAGED_RANGE_WAIT_ENV):
            staged_range_wait_s()


def test_the_sealed_readset_is_refused_when_it_is_not_the_bound_manifest(
        tmp_path, monkeypatch):
    """A claim row naming another manifest binds nothing.

    The blob is content-addressed and the digest is checked twice: against
    the row's own value and against the digest this run bound. A readset for
    a different submission is not this run's readset.
    """
    path, _ = _shard(tmp_path)
    resolver, consumer, digest, _publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=_whole_file(path))
    assert resolver.declared_readset()['state'] == 'bound'

    # A different, equally valid manifest in the same CAS: same paths, one
    # byte less declared, so only its digest differs.
    size = path.stat().st_size
    other_cas, other_digest, other_size = _seal_manifest(
        tmp_path, [(path, 0, size - 1)])
    assert other_digest != digest
    _publish_readset_on_the_claim(
        tmp_path, consumer, other_cas, other_digest, other_size)

    # Rebound IN PLACE, on the resolver that already answered "bound": the
    # declaration cache belongs to the binding, so it must go with it.
    # Keeping it would answer manifest A's membership for manifest B's
    # paths -- "not declared" for every one, an immediate refusal that
    # looks deliberate. That is #874 again under a new cause.
    bind_residency_manifest(other_digest)
    state = resolver.declared_readset()
    assert state['state'] == 'bound'
    bind_residency_manifest(digest)
    state = resolver.declared_readset()
    assert state['state'] == 'unbound'
    assert digest[:12] in str(state['reason'])


def test_a_non_strict_layer_read_with_a_map_still_never_waits(
        tmp_path, monkeypatch):
    """The wait is scoped by the POLICY, not by "a resolver exists".

    An inactive-policy reader serves pool bytes for an uncovered span and
    always could, so it has nothing to wait for. Entering the wait on the
    mere presence of a resolver would change a path this fix has no
    business touching.
    """
    require_prismabuild_sdk()
    _pb()
    path, _ = _shard(tmp_path)
    consumer = _hex64(f"consumer-{tmp_path}")
    import prismabuild.pool as pool_mod
    _pb_queue(tmp_path, pool_mod, consumer)
    cas_root, digest, size = _seal_manifest(tmp_path, _whole_file(path))
    _publish_readset_on_the_claim(tmp_path, consumer, cas_root, digest, size)
    _launch_env(monkeypatch, consumer)
    map_path = _write_map(tmp_path, {}, manifest_sha256=digest,
                          stage_root=_stage_root(tmp_path))
    resolver = _bind(monkeypatch, map_path, digest)      # bound, NOT active
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "600")
    assert not active_policy()

    started = time.monotonic()
    served = _read_layer(path)
    assert time.monotonic() - started < 10.0

    from safetensors import safe_open
    with safe_open(str(path), framework='pt') as reference:
        assert torch.equal(served['layer.0.f32'].view(torch.uint8),
                           reference.get_tensor('f32').view(torch.uint8))
    # bytes_from_pool stays 0 on purpose: with the policy inactive and the
    # map naming no entry for this shard, staged_shard_opener hands the
    # caller back its OWN opener, so no reader exists to count anything.
    # That untouched path is exactly what this test is protecting.
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert report['range_wait_polls'] == 0
    assert report['declared_readset']['state'] == 'unread'

# -- a stale covering row whose file is missing waits when declared (PQ #903)
#
# Split out of #902. After #902 the resolver asks every covering entry, so a
# stale entry no longer hides a staged one. One path still ended a run on bytes
# PrismaBuild was about to deliver: a neighbour phase's entry covers the span,
# its staged file has been unlinked by an eviction, its row is still in the
# composed map, and the layer's own declared range has not landed yet, so there
# is no second row. Every covering entry failed -> RANGE_REFUSED -> the reader
# stopped waiting at once. If the stale row were absent the same span would be
# RANGE_UNCOVERED and the mover would land it. The refusal was caused by the
# map being behind the file system, not by the bytes being unavailable.
#
# PQ #903: every covering staged file missing + span declared stays waitable
# for its own range (typed ENOENT, never a strerror substring match);
# wrong-size, non-regular, permission and other integrity failures still
# refuse at once. These tests rewrite the composed map mid-wait through the
# REAL PB writers with real file and lease paths.


def test_a_layer_read_waits_when_the_only_covering_entry_is_stale_missing(
        tmp_path, monkeypatch):
    """Stale missing row + declared own range not yet landed: wait, then serve."""
    path, _tensors = _shard(tmp_path)
    stage_root = _stage_root(tmp_path)
    staged_stale = _stage_whole(stage_root, path)
    resolver, consumer, _digest, publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=_whole_file(path))
    publish({'stale': (path, staged_stale)})
    assert resolver.declared_readset()['state'] == 'bound'
    # The eviction unlinked the file; the row is still in the composed map.
    staged_stale.unlink()
    # The layer's own range lands mid-wait as a distinct staged file.
    staged_own = stage_root / 'model-00001-of-00002.own.safetensors'
    staged_own.write_bytes(path.read_bytes())
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "90")

    def mover_leg():
        time.sleep(2.0)
        publish({'own': (path, staged_own)})

    thread = threading.Thread(target=mover_leg, name='own-range-mover')
    thread.start()
    try:
        served = _read_layer(path)
    finally:
        thread.join(timeout=120)
        assert not thread.is_alive()

    from safetensors import safe_open
    with safe_open(str(path), framework='pt') as reference:
        want = reference.get_tensor('f32')
    got = served['layer.0.f32']
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))

    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert report['bytes_from_stage'] > 0
    assert report['range_waits_served'] == 1
    assert report['range_waits_refused'] == 0
    assert report['range_wait_polls'] >= 1
    assert report['fallback_count'] == 0
    assert report['declared_readset']['state'] == 'bound'
    assert _pins_live(tmp_path, consumer) == []


def test_a_stale_missing_entry_with_nothing_landing_refuses_after_its_bound(
        tmp_path, monkeypatch):
    """Stale missing row + declared span nothing ever stages: wait, then refuse."""
    path, _ = _shard(tmp_path)
    staged_stale = _stage_whole(_stage_root(tmp_path), path)
    resolver, _consumer, _digest, publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=_whole_file(path))
    publish({'stale': (path, staged_stale)})
    staged_stale.unlink()
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "3")

    started = time.monotonic()
    with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
        _read_layer(path)
    waited = time.monotonic() - started
    assert 2.0 <= waited < 60.0, f"waited {waited:.1f}s against a 3s bound"

    report = resolver.report()
    assert report['range_waits_refused'] == 1
    assert report['range_waits_served'] == 0
    assert report['range_wait_polls'] >= 1
    assert report['bytes_from_pool'] == 0
    assert report['fallback_count'] == 0


def test_a_nonregular_staged_entry_refuses_without_waiting(
        tmp_path, monkeypatch):
    """A covering entry that is a directory is integrity evidence, not a miss."""
    path, _ = _shard(tmp_path)
    staged = _stage_whole(_stage_root(tmp_path), path)
    resolver, _consumer, _digest, publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=_whole_file(path))
    publish({'s': (path, staged)})
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "600")
    staged.unlink()
    staged.mkdir()

    started = time.monotonic()
    with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
        _read_layer(path)
    assert time.monotonic() - started < 30.0

    report = resolver.report()
    assert report['range_wait_polls'] == 0
    assert report['range_waits_refused'] == 0
    assert report['fallback_count'] >= 1
    assert report['bytes_from_pool'] == 0


def test_a_stale_missing_entry_with_unbound_readset_does_not_wait(
        tmp_path, monkeypatch):
    """Missing file + unbound readset: resolver says UNCOVERED, caller declines.

    PQ #903 maps all-missing to RANGE_UNCOVERED whenever `_declares` is not
    False, which includes the unbound (None) case. The wait itself stays
    gated on a bound sealed readset (`_await_layer_readset` returns early
    when `declared_readset.state != "bound"`), so an unbound run keeps the
    pre-#874 behaviour: immediate refusal, no polls, reason recorded.
    """
    path, _ = _shard(tmp_path)
    staged_stale = _stage_whole(_stage_root(tmp_path), path)
    resolver, _consumer, _digest, publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=_whole_file(path), seal=False)
    publish({'stale': (path, staged_stale)})
    staged_stale.unlink()
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, "600")

    started = time.monotonic()
    with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
        _read_layer(path)
    assert time.monotonic() - started < 30.0

    report = resolver.report()
    assert report['declared_readset']['state'] == 'unbound'
    assert report['range_wait_polls'] == 0
    assert report['range_waits_served'] == 0
    assert report['range_waits_refused'] == 0
    assert report['bytes_from_pool'] == 0


def test_clearing_the_injection_removes_only_the_modules_it_imported():
    """The teardown leaves no venv ``prismabuild`` behind (PQ #963).

    This file injects the installed SDK and imports ``prismabuild.pool``
    and ``prismabuild.residency_map`` from it. Those entries used to
    outlive the fixture, and ``_sdk_from_tree`` serves a preimported
    ``prismabuild.reader_lease`` as it is, so every later test in the
    same pytest worker that resolves the SDK out of a sealed generation
    tree refused ``lease-helper-divergent``. Whether a test was hit was
    a pure function of shard packing.
    """
    require_prismabuild_sdk()
    from prismaquant.staged_lease import (
        clear_injected_sdk_for_tests, inject_installed_sdk_for_tests)

    def loaded():
        return {name for name in sys.modules
                if name == 'prismabuild' or name.startswith('prismabuild.')}

    # Start from a known state: nothing prismabuild imported, then one
    # module this test imports itself, which the teardown must keep.
    clear_injected_sdk_for_tests()
    for name in loaded():
        del sys.modules[name]
    import prismabuild.reader_lease  # noqa: F401,PLC0415
    kept = loaded()
    assert kept, 'the pinned SDK import is the module the teardown keeps'

    inject_installed_sdk_for_tests()
    import prismabuild.pool  # noqa: F401,PLC0415
    assert 'prismabuild.pool' in loaded()

    clear_injected_sdk_for_tests()
    assert 'prismabuild.pool' not in loaded(), (
        'a leftover venv prismabuild.* refuses every later sealed-tree '
        'resolution in this pytest worker')
    assert loaded() == kept, (
        'the teardown removed a module the injection did not import')

    for name in loaded():
        del sys.modules[name]


# -- one lease per read window (PQ #997) ---------------------------------------
#
# Stage A R12 spent 36% of its main thread in the per-entry lease path: a
# cover lookup, an ownership-lock acquire and an ownership-lock release per
# 64-entry window ENTRY. These run the real chain and count the SDK calls.


def _exact_entries(tmp_path, count):
    from prismaquant.perturbed_x_cache import write_exact_activation_cache_entry
    directory = tmp_path / 'pool' / 'entries'
    directory.mkdir(parents=True, exist_ok=True)
    refs, tensors = [], []
    for index in range(count):
        tensor = torch.arange(32, dtype=torch.float32).reshape(8, 4) + 100 * index
        nbytes = tensor.numel() * tensor.element_size()
        refs.append(write_exact_activation_cache_entry(
            directory, f"entry-{index}", tensor,
            identity=_exact_identity(f"entry-{index}"),
            max_tensor_bytes=nbytes, max_file_bytes=nbytes + 65536))
        tensors.append(tensor)
    return refs, tensors


def _count_sdk_calls(monkeypatch, rl):
    calls = {"acquire_for": 0, "release": 0, "covers_for_keys": 0}
    for name in calls:
        original = getattr(rl, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(rl, name, counted)
    return calls


def _read_exact(refs):
    from prismaquant.perturbed_x_cache import prefetch_exact_activation_cache_entries
    nbytes = sum(ref.tensor_bytes for ref in refs)
    with prefetch_exact_activation_cache_entries(
            refs, max_tensor_bytes=nbytes, expected_session="strict-tier-session",
            release_file_pages=False) as window:
        return {ref: window._tensors[ref].clone() for ref in refs}


def _lease_deltas(before):
    from prismaquant.perturbed_x_cache import exact_lease_counters
    return {name: count - before[name] for name, count in exact_lease_counters().items()}


def _ram_fixture(tmp_path, monkeypatch, refs, *, ram_offered, ram_published):
    """Every entry staged on SSD. The entries in ``ram_offered`` also offer
    a tmpfs copy in the map, and the RAM mover publishes only the ones in
    ``ram_published``. Returns ``(resolver, consumer, rl)``."""
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover_ssd = _hex64(f"mover-ssd-{tmp_path}")
    mover_ram = _hex64(f"mover-ram-{tmp_path}")
    _pb_queue(tmp_path, pool_mod, consumer)
    root = _stage_root(tmp_path)
    staged = {i: _stage_whole(root, Path(ref.path)) for i, ref in enumerate(refs)}
    ram_root, ram = _promote_ram(tmp_path, {i: staged[i] for i in ram_offered})
    _announce(tmp_path)
    residency = tmp_path / 'residency'
    keys = {i: residency_map_key(str(ref.path), 0) for i, ref in enumerate(refs)}
    _pb_publish(rl, map_mod, residency, root, consumer, mover_ssd, MANIFEST,
                {keys[i]: (Path(refs[i].path), staged[i]) for i in staged})
    if ram_published:
        _pb_publish_ram(rl, map_mod, residency, ram_root, consumer, mover_ram,
                        MANIFEST, {keys[i]: (Path(refs[i].path), ram[i])
                                   for i in ram_published}, EPOCH)
    map_path = _write_map(tmp_path, {i: (Path(refs[i].path), staged[i], ram.get(i))
                                     for i in staged},
                          ram_root=ram_root, leads=[mover_ssd])
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    activate_staged_tier_policy("ram,ssd")
    return residency_resolver(), consumer, rl


def test_an_exact_window_takes_one_lease_for_all_its_entries(tmp_path, monkeypatch):
    from prismaquant.perturbed_x_cache import exact_lease_counters
    refs, tensors = _exact_entries(tmp_path, 4)
    root = _stage_root(tmp_path)
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch,
        {f'e{i}': (Path(ref.path), _stage_whole(root, Path(ref.path)), None)
         for i, ref in enumerate(refs)})
    calls = _count_sdk_calls(monkeypatch, _pb()[0])
    before = exact_lease_counters()
    batched = _read_exact(refs)
    assert calls == {"acquire_for": 1, "release": 1, "covers_for_keys": 1}, calls
    assert _lease_deltas(before) == {"windows_batched": 1, "entries_batched": 4,
                                     "entries_single": 0, "batch_fallbacks": 0}
    for ref, tensor in zip(refs, tensors):
        assert torch.equal(batched[ref].view(torch.uint8), tensor.view(torch.uint8))
    served = resolver.report()['serving_tiers'][-4:]
    assert {row['serving_tier'] for row in served} == {'stage'}
    assert len({row['pin_id'] for row in served}) == 1 and served[0]['pin_id']
    assert resolver.report()['bytes_from_pool'] == 0
    assert _pins_live(tmp_path, consumer) == []

    # The per-entry path (one entry per window) serves the same bytes with
    # one lease each: the batched window changes the lease count, never
    # the bytes.
    for key in calls:
        calls[key] = 0
    before = exact_lease_counters()
    single = {}
    for ref in refs:
        single.update(_read_exact([ref]))
    assert calls == {"acquire_for": 4, "release": 4, "covers_for_keys": 4}, calls
    assert _lease_deltas(before)["entries_single"] == 4
    for ref in refs:
        assert torch.equal(single[ref].view(torch.uint8), batched[ref].view(torch.uint8))
    assert _pins_live(tmp_path, consumer) == []


def test_ram_entries_share_a_ram_window_and_the_rest_an_ssd_window(
        tmp_path, monkeypatch):
    from prismaquant.perturbed_x_cache import exact_lease_counters
    refs, tensors = _exact_entries(tmp_path, 3)
    resolver, consumer, rl = _ram_fixture(
        tmp_path, monkeypatch, refs, ram_offered=[0, 1], ram_published=[0, 1])
    calls = _count_sdk_calls(monkeypatch, rl)
    before = exact_lease_counters()
    got = _read_exact(refs)
    assert calls["acquire_for"] == 2 and calls["release"] == 2, calls
    assert _lease_deltas(before) == {"windows_batched": 2, "entries_batched": 3,
                                     "entries_single": 0, "batch_fallbacks": 0}
    for ref, tensor in zip(refs, tensors):
        assert torch.equal(got[ref].view(torch.uint8), tensor.view(torch.uint8))
    tiers = {row['path']: row['serving_tier']
             for row in resolver.report()['serving_tiers'][-3:]}
    assert [tiers[str(ref.path)] for ref in refs] == ['ram', 'ram', 'stage']
    assert resolver.report()['ram_fallbacks'] == []
    assert _pins_live(tmp_path, consumer) == []


def test_an_unpublished_ram_tier_folds_the_window_into_one_ssd_lease(
        tmp_path, monkeypatch):
    """No RAM mover published anything at the announced epoch: every entry
    records the RAM fallback, exactly as the single-entry leg does, and the
    whole window leases once on SSD."""
    from prismaquant.perturbed_x_cache import exact_lease_counters
    refs, tensors = _exact_entries(tmp_path, 3)
    resolver, consumer, rl = _ram_fixture(
        tmp_path, monkeypatch, refs, ram_offered=[0, 1, 2], ram_published=[])
    calls = _count_sdk_calls(monkeypatch, rl)
    before = exact_lease_counters()
    got = _read_exact(refs)
    assert calls["acquire_for"] == 1 and calls["release"] == 1, calls
    assert _lease_deltas(before) == {"windows_batched": 1, "entries_batched": 3,
                                     "entries_single": 0, "batch_fallbacks": 0}
    for ref, tensor in zip(refs, tensors):
        assert torch.equal(got[ref].view(torch.uint8), tensor.view(torch.uint8))
    report = resolver.report()
    assert len(report['ram_fallbacks']) == 3
    assert {row['serving_tier'] for row in report['serving_tiers'][-3:]} == {'stage'}
    assert report['bytes_from_pool'] == 0
    assert _pins_live(tmp_path, consumer) == []


def test_a_partial_ram_cover_rereads_the_window_one_entry_at_a_time(
        tmp_path, monkeypatch):
    """The RAM mover published one of two entries. A batched lookup refuses
    the pair as a coverage gap (integrity) where each entry alone answers
    precisely, so the window falls back: the published entry serves from
    RAM and the other falls to SSD, as they did before #997."""
    from prismaquant.perturbed_x_cache import exact_lease_counters
    refs, tensors = _exact_entries(tmp_path, 2)
    resolver, consumer, _rl = _ram_fixture(
        tmp_path, monkeypatch, refs, ram_offered=[0, 1], ram_published=[0])
    before = exact_lease_counters()
    got = _read_exact(refs)
    assert _lease_deltas(before) == {"windows_batched": 0, "entries_batched": 0,
                                     "entries_single": 2, "batch_fallbacks": 1}
    for ref, tensor in zip(refs, tensors):
        assert torch.equal(got[ref].view(torch.uint8), tensor.view(torch.uint8))
    tiers = {row['path']: row['serving_tier']
             for row in resolver.report()['serving_tiers'][-2:]}
    assert [tiers[str(ref.path)] for ref in refs] == ['ram', 'stage']
    assert _pins_live(tmp_path, consumer) == []


def test_a_range_evicted_between_windows_is_refused_or_restaged_never_read_stale(
        tmp_path, monkeypatch):
    """PB #904 lets a landed range be evicted past its refill horizon. No
    proof carries from one window to the next: each window's lease
    re-verifies every key under the ownership lock, so a staged copy that
    changed after window 1 refuses in window 2, and a honest re-stage
    serves the right bytes again."""
    refs, tensors = _exact_entries(tmp_path, 3)
    root = _stage_root(tmp_path)
    staged = {i: _stage_whole(root, Path(ref.path)) for i, ref in enumerate(refs)}
    resolver, consumer, mover = _leased_fixture(
        tmp_path, monkeypatch,
        {f'e{i}': (Path(ref.path), staged[i], None) for i, ref in enumerate(refs)})
    first = _read_exact(refs)
    for ref, tensor in zip(refs, tensors):
        assert torch.equal(first[ref].view(torch.uint8), tensor.view(torch.uint8))

    # Evicted, and a different copy of the same length lands at the path.
    victim = staged[1]
    blob = victim.read_bytes()
    landed = victim.stat()
    victim.unlink()
    victim.write_bytes(bytes(reversed(blob)))
    # A later landing, stated rather than left to timestamp granularity.
    os.utime(victim, ns=(landed.st_atime_ns, landed.st_mtime_ns + 10 ** 9))
    with pytest.raises(TierPolicyRefused):
        _read_exact(refs)
    assert resolver.report()['bytes_from_pool'] == 0
    assert _pins_live(tmp_path, consumer) == []

    # Evicted outright: the map row's staged file is gone.
    victim.unlink()
    with pytest.raises(TierPolicyRefused):
        _read_exact(refs)
    assert _pins_live(tmp_path, consumer) == []

    # Re-staged honestly: a mover lands the right bytes and republishes.
    victim.write_bytes(blob)
    rl, _pool_mod, map_mod = _pb()
    _pb_publish(rl, map_mod, tmp_path / 'residency', root, consumer, mover, MANIFEST,
                {residency_map_key(str(ref.path), 0): (Path(ref.path), staged[i])
                 for i, ref in enumerate(refs)})
    again = _read_exact(refs)
    for ref, tensor in zip(refs, tensors):
        assert torch.equal(again[ref].view(torch.uint8), tensor.view(torch.uint8))
    assert resolver.report()['bytes_from_pool'] == 0
    assert _pins_live(tmp_path, consumer) == []


# -- tier byte counts are complete (PQ #1026) ---------------------------------
#
# The resolver's per-tier byte counters are how a run shows its reads rode
# the tiers. The exact-entry and verified-activation readers recorded which
# tier served each read but never counted its bytes, so an executed
# band-serial consumer reported 4,414 of the 17,332 bytes it read. Every
# test here checks that the tiers sum to the exact bytes of the files read.


def _tier_bytes(resolver):
    report = resolver.report()
    return {tier: report[f'bytes_from_{tier}'] for tier in ('ram', 'stage', 'pool')}


def _sizes(refs):
    return [Path(ref.path).stat().st_size for ref in refs]


def exact_lease_counters_snapshot():
    from prismaquant.perturbed_x_cache import exact_lease_counters
    return exact_lease_counters()


def test_single_exact_reads_count_every_byte_on_the_stage(tmp_path, monkeypatch):
    refs, _tensors = _exact_entries(tmp_path, 3)
    root = _stage_root(tmp_path)
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch,
        {f'e{i}': (Path(ref.path), _stage_whole(root, Path(ref.path)), None)
         for i, ref in enumerate(refs)})
    before = exact_lease_counters_snapshot()
    for ref in refs:
        _read_exact([ref])
    assert _lease_deltas(before)["entries_single"] == 3
    assert _tier_bytes(resolver) == {'ram': 0, 'stage': sum(_sizes(refs)), 'pool': 0}
    assert _pins_live(tmp_path, consumer) == []


def test_a_grouped_exact_window_counts_every_byte_on_the_stage(tmp_path, monkeypatch):
    refs, _tensors = _exact_entries(tmp_path, 4)
    root = _stage_root(tmp_path)
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch,
        {f'e{i}': (Path(ref.path), _stage_whole(root, Path(ref.path)), None)
         for i, ref in enumerate(refs)})
    before = exact_lease_counters_snapshot()
    _read_exact(refs)
    assert _lease_deltas(before)["entries_batched"] == 4
    assert _tier_bytes(resolver) == {'ram': 0, 'stage': sum(_sizes(refs)), 'pool': 0}
    assert _pins_live(tmp_path, consumer) == []


def test_a_grouped_window_counts_ram_and_stage_bytes_apart(tmp_path, monkeypatch):
    refs, _tensors = _exact_entries(tmp_path, 3)
    resolver, consumer, _rl = _ram_fixture(
        tmp_path, monkeypatch, refs, ram_offered=[0, 1], ram_published=[0, 1])
    before = exact_lease_counters_snapshot()
    _read_exact(refs)
    assert _lease_deltas(before)["entries_batched"] == 3
    sizes = _sizes(refs)
    assert _tier_bytes(resolver) == {'ram': sizes[0] + sizes[1], 'stage': sizes[2],
                                     'pool': 0}
    assert _pins_live(tmp_path, consumer) == []


def test_a_window_reread_one_entry_at_a_time_counts_each_tier(tmp_path, monkeypatch):
    """A partial RAM cover sends the window down the single-entry path; the
    entry RAM serves is counted as RAM bytes and the other as stage bytes."""
    refs, _tensors = _exact_entries(tmp_path, 2)
    resolver, consumer, _rl = _ram_fixture(
        tmp_path, monkeypatch, refs, ram_offered=[0, 1], ram_published=[0])
    before = exact_lease_counters_snapshot()
    _read_exact(refs)
    assert _lease_deltas(before)["entries_single"] == 2
    sizes = _sizes(refs)
    assert _tier_bytes(resolver) == {'ram': sizes[0], 'stage': sizes[1], 'pool': 0}
    assert _pins_live(tmp_path, consumer) == []


def test_a_refused_exact_read_counts_no_bytes(tmp_path, monkeypatch):
    refs, _tensors = _exact_entries(tmp_path, 2)
    root = _stage_root(tmp_path)
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'o': (Path(refs[1].path), _stage_whole(root, Path(refs[1].path)),
                         None)}))
    with pytest.raises(TierPolicyRefused, match="staged-not-serving"):
        _read_exact(refs)
    assert _tier_bytes(resolver) == {'ram': 0, 'stage': 0, 'pool': 0}


def test_a_verified_activation_load_counts_its_bytes_on_the_stage(
        tmp_path, monkeypatch):
    from prismaquant.perturbed_x_cache import load_verified_activation_cache_entry
    pool = tmp_path / 'pool'
    pool.mkdir(parents=True, exist_ok=True)
    path = pool / 'capture.pt'
    torch.save({'inputs': torch.arange(64, dtype=torch.float32).reshape(8, 8)}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    root = _stage_root(tmp_path)
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch, {'a': (path, _stage_whole(root, path), None)})
    load_verified_activation_cache_entry(
        path, expected_sha256=digest, policy=_activation_policy(),
        max_storage_bytes=4 * 1024 ** 2)
    assert _tier_bytes(resolver) == {'ram': 0, 'stage': path.stat().st_size, 'pool': 0}
    assert _pins_live(tmp_path, consumer) == []


def test_declared_reads_under_a_bound_map_count_pool_bytes(tmp_path, monkeypatch):
    """Without the strict policy the readers open the declared files; with a
    map bound, those bytes are pool bytes, so the tiers still sum to every
    byte read."""
    from prismaquant.perturbed_x_cache import load_verified_activation_cache_entry
    refs, _tensors = _exact_entries(tmp_path, 3)
    capture = tmp_path / 'pool' / 'capture.pt'
    torch.save({'inputs': torch.zeros(4, 4)}, capture)
    digest = hashlib.sha256(capture.read_bytes()).hexdigest()
    root = _stage_root(tmp_path)
    resolver = _bind(monkeypatch, _write_map(
        tmp_path, {'c': (capture, _stage_whole(root, capture), None)}))
    _read_exact(refs[:1])
    _read_exact(refs[1:])
    load_verified_activation_cache_entry(
        capture, expected_sha256=digest, policy=_activation_policy(),
        max_storage_bytes=4 * 1024 ** 2)
    total = sum(_sizes(refs)) + capture.stat().st_size
    assert _tier_bytes(resolver) == {'ram': 0, 'stage': 0, 'pool': total}


def test_declared_reads_without_a_map_count_nothing(tmp_path, monkeypatch):
    refs, tensors = _exact_entries(tmp_path, 2)
    got = _read_exact(refs)
    for ref, tensor in zip(refs, tensors):
        assert torch.equal(got[ref], tensor)
    assert residency_resolver() is None



# -- plane streams (PQ #1142) ------------------------------------------------


def _write_plane_checkpoint(tmp_path, *, probes=2, batches=2):
    from prismaquant.joint_adjoint_checkpoints import (
        adjoint_space, write_adjoint_checkpoint)
    from prismaquant.sensitivity_probe import SharedStateCotangents
    tensors = {(probe, batch): torch.arange(12, dtype=torch.float32).reshape(3, 4)
               + 100 * probe + 10 * batch
               for probe in range(probes) for batch in range(batches)}
    state = SharedStateCotangents().state_dict()
    record = write_adjoint_checkpoint(
        adjoint_space(tmp_path), boundary=5,
        session={"generation": "g" * 32, "kind": "adjoint_checkpoint"},
        cotangents=tensors,
        shared_adjoint={key: state for key in tensors},
        shared_pass={batch: {"captured": None} for batch in range(batches)})
    return record, tensors


def _load_plane(tmp_path, record, **kwargs):
    from prismaquant.joint_adjoint_checkpoints import adjoint_space, load_adjoint_checkpoint
    cotangents, _shared_adjoint, _shared_pass = load_adjoint_checkpoint(
        adjoint_space(tmp_path), record, **kwargs)
    return cotangents


def test_a_checkpoint_plane_streams_in_windows_with_one_lease_each(tmp_path, monkeypatch):
    """A budgeted plane load leases a window of entries at a time, not each entry.

    Four entries under a budget that holds two windows of two: two plane
    leases and two cover lookups for the waits, where the per-entry load
    takes four of each. The bytes are the same either way, nothing is read
    from the pool, the charges never exceed the budget and all come back,
    and no pin outlives the load.
    """
    from prismaquant.perturbed_x_cache import exact_lease_counters
    record, tensors = _write_plane_checkpoint(tmp_path)
    resolver, consumer, _paths = _stage_checkpoint_entries(tmp_path, monkeypatch, record)
    calls = _count_sdk_calls(monkeypatch, _pb()[0])
    entry_bytes = int(record["activation_entries"][0]["tensor_bytes"])

    before = exact_lease_counters()
    single = _load_plane(tmp_path, record)
    single_calls = dict(calls)
    assert _lease_deltas(before) == {"windows_batched": 0, "entries_batched": 0,
                                     "entries_single": 4, "batch_fallbacks": 0}

    for key in calls:
        calls[key] = 0
    held, peak = [0], [0]

    def charge(delta):
        held[0] += delta
        peak[0] = max(peak[0], held[0])

    budget = 4 * entry_bytes
    before = exact_lease_counters()
    streamed = _load_plane(tmp_path, record, max_resident_bytes=budget,
                           residency_check=charge)
    assert _lease_deltas(before) == {"windows_batched": 2, "entries_batched": 4,
                                     "entries_single": 0, "batch_fallbacks": 0}
    # The manifest and shared-state reads are the same in both loads, so
    # the difference is the plane's: 4 entry leases became 2 window leases,
    # and 4 + 4 cover lookups (one wait, one lease per entry) became 2 + 2.
    assert single_calls["acquire_for"] - calls["acquire_for"] == 4 - 2, (single_calls, calls)
    assert single_calls["release"] - calls["release"] == 4 - 2, (single_calls, calls)
    assert single_calls["covers_for_keys"] - calls["covers_for_keys"] == 8 - 4, (
        single_calls, calls)
    assert held == [0] and 0 < peak[0] <= budget
    for key, tensor in tensors.items():
        assert torch.equal(streamed[key], tensor)
        assert torch.equal(single[key], tensor)
    assert resolver.report()['bytes_from_pool'] == 0
    assert _pins_live(tmp_path, consumer) == []


def test_a_changed_staged_copy_in_a_window_refuses_with_the_per_entry_kind(
        tmp_path, monkeypatch):
    """A window whose third staged copy changed refuses as one entry would.

    The batched lease refuses, the window is re-read one entry at a time,
    and the changed entry's lease refuses ``integrity`` -- the refusal and
    kind the per-entry load raises. Every charge comes back and no pin is
    left.
    """
    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "4")
    record, _tensors = _write_plane_checkpoint(tmp_path)
    _resolver, consumer, _paths = _stage_checkpoint_entries(tmp_path, monkeypatch, record)
    victim = Path(record["activation_entries"][2]["path"])
    staged = tmp_path / 'stage' / 'prewarm' / victim.name
    blob = staged.read_bytes()
    staged.write_bytes(blob[:-1] + bytes([blob[-1] ^ 0xFF]))
    held = [0]

    def charge(delta):
        held[0] += delta

    entry_bytes = int(record["activation_entries"][0]["tensor_bytes"])
    with pytest.raises(LeaseRefused) as streamed:
        _load_plane(tmp_path, record, max_resident_bytes=8 * entry_bytes,
                    residency_check=charge)
    assert streamed.value.kind == "integrity"
    assert held == [0]
    assert _pins_live(tmp_path, consumer) == []
    with pytest.raises(LeaseRefused) as single:
        _load_plane(tmp_path, record)
    assert single.value.kind == "integrity"
    assert _pins_live(tmp_path, consumer) == []


def test_a_failed_read_in_a_parallel_window_exposes_nothing_and_strands_no_pin(
        tmp_path, monkeypatch):
    """A read that fails under a window's pin, on a reader thread, cleans up.

    Four reader threads read one pinned window of four, and the third
    entry's read raises. The load raises that error, gives every charge
    back, and releases the window only after the other reads returned.
    """
    import prismaquant.perturbed_x_cache as pxc
    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "4")
    record, _tensors = _write_plane_checkpoint(tmp_path)
    _resolver, consumer, _paths = _stage_checkpoint_entries(tmp_path, monkeypatch, record)
    victim = record["activation_entries"][2]["name"]
    real = pxc._read_exact_entry
    open_reads = [0]
    lock = threading.Lock()

    def failing(ref, **kwargs):
        with lock:
            open_reads[0] += 1
        try:
            if ref.name == victim:
                raise RuntimeError("exact activation entry checksum changed")
            time.sleep(0.05)
            return real(ref, **kwargs)
        finally:
            with lock:
                open_reads[0] -= 1

    released_with_reads_open = []
    real_exit = pxc._run_in_order

    def watching(pool, calls):
        try:
            return real_exit(pool, calls)
        finally:
            released_with_reads_open.append(open_reads[0])

    monkeypatch.setattr(pxc, "_read_exact_entry", failing)
    monkeypatch.setattr(pxc, "_run_in_order", watching)
    held = [0]

    def charge(delta):
        held[0] += delta

    entry_bytes = int(record["activation_entries"][0]["tensor_bytes"])
    with pytest.raises(RuntimeError, match="checksum changed"):
        _load_plane(tmp_path, record, max_resident_bytes=8 * entry_bytes,
                    residency_check=charge)
    assert released_with_reads_open and set(released_with_reads_open) == {0}
    assert held == [0]
    assert _pins_live(tmp_path, consumer) == []


def test_parallel_reads_raise_the_first_failure_in_window_order():
    """The lowest-indexed failure is raised even when a later one fails first."""
    from prismaquant.perturbed_x_cache import _exact_read_pool, _run_in_order
    pool = _exact_read_pool(4)
    started, released = [], threading.Event()

    def ok(index):
        started.append(index)
        return index

    def slow_failure():
        started.append(1)
        assert released.wait(10)
        raise ValueError("entry 1")

    def fast_failure():
        started.append(2)
        released.set()
        raise ValueError("entry 2")

    with pytest.raises(ValueError, match="entry 1"):
        _run_in_order(pool, [lambda: ok(0), slow_failure, fast_failure])
    assert _run_in_order(pool, [lambda i=i: ok(i) for i in range(6)]) == list(range(6))
    assert _run_in_order(None, [lambda: ok(7)]) == [7]


class _StartAtSubmitPool:
    """A pool that starts each call the moment it is submitted.

    The worst case for cancelling after a failure: every call has been
    picked up before the caller can see the first one fail.
    """

    def submit(self, fn, *args):
        from concurrent.futures import Future
        future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def test_calls_after_a_failure_that_have_not_started_never_start():
    from concurrent.futures import ThreadPoolExecutor
    from prismaquant.perturbed_x_cache import _run_in_order
    worker = ThreadPoolExecutor(max_workers=1)
    try:
        for pool in (_StartAtSubmitPool(), worker):
            ran = []

            def fail():
                ran.append("fail")
                raise ValueError("first")

            with pytest.raises(ValueError, match="first"):
                _run_in_order(pool, [fail] + [lambda i=i: ran.append(i) for i in range(4)])
            assert ran == ["fail"], (pool, ran)
    finally:
        worker.shutdown(wait=True)


def test_calls_before_a_later_failure_still_run_and_raise_first():
    from prismaquant.perturbed_x_cache import _run_in_order
    ran = []

    def fail(index):
        ran.append(index)
        raise ValueError(f"entry {index}")

    # Calls 0 and 2 fail; the serial loop raises entry 0 and never starts 1.
    with pytest.raises(ValueError, match="entry 0"):
        _run_in_order(_StartAtSubmitPool(), [lambda: fail(0), lambda: ran.append(1),
                                             lambda: fail(2)])
    assert ran == [0], ran


def test_concurrent_readers_borrow_distinct_buffers_and_release_frees_them():
    from prismaquant.perturbed_x_cache import EntryReadScratch
    scratch = EntryReadScratch()
    with scratch.lend() as first:
        assert first is scratch
        with scratch.lend() as second:
            assert second is not first
            assert first.buffer(16) is not second.buffer(16)
    with scratch.lend() as again:
        assert again is scratch or again is second
    assert len(scratch._extra) == 1
    scratch.release()
    assert len(scratch.buffer(0)) == 0 and scratch._extra == []


def test_exact_entry_windows_follow_the_budget():
    from prismaquant.joint_adjoint_checkpoints import exact_entry_windows
    records = [{"name": f"e{i}", "tensor_bytes": 10} for i in range(5)]
    windows, ahead = exact_entry_windows(records, max_resident_bytes=None)
    assert [len(w) for w in windows] == [1] * 5 and ahead is False
    windows, ahead = exact_entry_windows(records, max_resident_bytes=40)
    assert [len(w) for w in windows] == [2, 2, 1] and ahead is True
    windows, ahead = exact_entry_windows(records, max_resident_bytes=15)
    assert [len(w) for w in windows] == [1] * 5 and ahead is False
    with pytest.raises(RuntimeError, match="exceeds the stream's resident budget"):
        exact_entry_windows(records, max_resident_bytes=9)
    with pytest.raises(ValueError):
        exact_entry_windows(records, max_resident_bytes=0)
    assert exact_entry_windows([], max_resident_bytes=40) == ([], False)


def test_closing_a_stream_early_releases_every_charge(tmp_path):
    from contextlib import closing
    from prismaquant.joint_adjoint_checkpoints import stream_exact_entry_tensors
    record, tensors = _write_plane_checkpoint(tmp_path, probes=3, batches=2)
    entries = record["activation_entries"]
    entry_bytes = int(entries[0]["tensor_bytes"])
    held, peak = [0], [0]

    def charge(delta):
        held[0] += delta
        peak[0] = max(peak[0], held[0])

    from prismaquant.joint_adjoint_checkpoints import checkpoint_entry_session
    with closing(stream_exact_entry_tensors(
            entries, expected_session=checkpoint_entry_session(record),
            max_resident_bytes=2 * entry_bytes, residency_check=charge)) as stream:
        first, tensor = next(stream)
        assert first["name"] == entries[0]["name"]
        assert held[0] == 2 * entry_bytes  # this window and the next, read ahead
    assert held == [0] and peak[0] <= 2 * entry_bytes
    with closing(stream_exact_entry_tensors(
            entries, expected_session=checkpoint_entry_session(record),
            max_resident_bytes=2 * entry_bytes, residency_check=charge)) as stream:
        names = [entry["name"] for entry, _tensor in stream]
    assert names == [entry["name"] for entry in entries] and held == [0]
