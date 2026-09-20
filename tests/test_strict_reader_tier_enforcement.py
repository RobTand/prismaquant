"""Strict allowed-tier enforcement + lifetime-pinned reads (PQ #845, #850).

The staged-read contract forbids bulk-input reads from the pool/HDD tier;
the reader-lease contract pins every staged byte for its reader's
lifetime. These tests run the REAL chain — real resolver, real composed
map, real PB fragments/material written by the REAL PB writers, the REAL
pinned SDK (`a6e6b310a1`, drift-refused), real claim rows in the real
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
    parse_allowed_tiers, staged_tier_policy_test_context,
)
from prismaquant.staged_lease import (
    LeaseRefused, PINNED_SDK_COMMIT, set_lease_helper_root,
)

MANIFEST = 'e' * 64
LEAD = 'f' * 64
RAM_TIER = 'ram:dl380g10'
EPOCH = '1789771929-aba6e46e41fb03ef'
STALE_EPOCH = '1789788888-9c1d2e3f4a5b'
STAGE_TIER = 'prismabuild-stage:dl380g10'

PB_PIN_ROOT = Path("/home/rob/tmp/pb-reader-lease-pin2-20260920")
PB_SRC = PB_PIN_ROOT / "src"
PINNED_READER_LEASE_SHA256 = (
    "b4428c5b898a0225a0b9dca5822ff9aab72538db6e80722c0b0c9e9326050d14")


@pytest.fixture(autouse=True)
def _forget_state(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(TIERS_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv("PRISMABUILD_ACTION_KEY", raising=False)
    reset_residency_resolver_for_tests()
    deactivate_staged_tier_policy_for_tests()
    set_lease_helper_root(None)
    from prismaquant.staged_lease import _ACQUIRE_CONTEXT
    _ACQUIRE_CONTEXT.clear()
    yield
    reset_residency_resolver_for_tests()
    deactivate_staged_tier_policy_for_tests()
    set_lease_helper_root(None)
    _ACQUIRE_CONTEXT.clear()


def _hex64(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


LAUNCH_NONCE = "n" * 32
LAUNCH_SCOPE = "unit-1"


def _launch_env(monkeypatch, consumer):
    """Launch-bound identity pair, matching the claim row exactly (the new
    SDK binds pins from launch env + live claim, never a claim alone)."""
    _launch_env(monkeypatch, consumer)
    monkeypatch.setenv("PRISMABUILD_ACTION_NONCE", LAUNCH_NONCE)
    monkeypatch.setenv("PRISMABUILD_ACTION_SCOPE", LAUNCH_SCOPE)


# -- pinned PB SDK + queue fixtures (real writers, real formats) ------------

def _pb():
    """The pristine pinned SDK; refuses on any drift (declared artifact)."""
    blob = (PB_SRC / "prismabuild" / "reader_lease.py").read_bytes()
    assert hashlib.sha256(blob).hexdigest() == PINNED_READER_LEASE_SHA256, (
        "PB pin drifted: refusing instead of integrating against a "
        "different SDK")
    assert PINNED_SDK_COMMIT.startswith("d079ad33")
    if str(PB_SRC) not in sys.path:
        sys.path.insert(0, str(PB_SRC))
    import prismabuild.reader_lease as rl
    import prismabuild.pool as pool_mod
    import prismabuild.residency_map as map_mod
    assert str(Path(rl.__file__).resolve()).startswith(str(PB_SRC) + os.sep)
    return rl, pool_mod, map_mod


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
    set_lease_helper_root(PB_PIN_ROOT)
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
    set_lease_helper_root(PB_PIN_ROOT)
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
    set_lease_helper_root(PB_PIN_ROOT)
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


def test_strict_source_without_helper_refuses_lease_required(tmp_path, monkeypatch):
    """No helper, no pin: strict lifetime requires real support and
    refuses instead of serving staged bytes unpinned or falling open."""
    path, _ = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'s': (path, staged, None)}))
    assert lease_helper_root_is_unset()
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        with pytest.raises(LeaseRefused, match="lease-helper-unavailable"):
            reader.get_tensor('f32')
    assert resolver.report()['bytes_from_pool'] == 0
    assert resolver.report()['bytes_from_stage'] == 0


def lease_helper_root_is_unset():
    from prismaquant.staged_lease import lease_helper_root
    return lease_helper_root() is None


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
    set_lease_helper_root(PB_PIN_ROOT)
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
    set_lease_helper_root(PB_PIN_ROOT)
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
    set_lease_helper_root(PB_PIN_ROOT)
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
    set_lease_helper_root(PB_PIN_ROOT)
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
    set_lease_helper_root(PB_PIN_ROOT)
    with pytest.raises(TierPolicyRefused, match="missing-digest-binding"):
        cache.prefetch([key], max_workers=1)


def test_strict_pwc_unbounded_refuses_without_fallthrough(tmp_path, monkeypatch):
    cache, key, path, _ = _pwc(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    _strict(monkeypatch, _write_map(tmp_path, {'p': (path, staged, None)}))
    set_lease_helper_root(PB_PIN_ROOT)
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
    set_lease_helper_root(PB_PIN_ROOT)
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
    set_lease_helper_root(PB_PIN_ROOT)
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
    set_lease_helper_root(PB_PIN_ROOT)
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
    set_lease_helper_root(PB_PIN_ROOT)
    with pytest.raises(TierPolicyRefused, match="staged-not-serving"):
        with prefetch_exact_activation_cache_entries(
                [ref], max_tensor_bytes=nbytes,
                expected_session="strict-tier-session",
                release_file_pages=False):
            pass


# -- lease mechanics: duplicates, fork guard ----------------------------------

def test_duplicate_acquire_token_adopts_one_ref(tmp_path, monkeypatch):
    """Same token twice adopts one ref; two exits release exactly once."""
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
    set_lease_helper_root(PB_PIN_ROOT)
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


def test_forked_child_window_use_refused_loudly(tmp_path, monkeypatch):
    """Every window operation in a forked child raises: open (would outlive
    the parent's release), close, and exit. The pin stays intact and the
    parent's exact release still unlinks it."""
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
    set_lease_helper_root(PB_PIN_ROOT)
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
    from prismaquant.staged_lease import PINNED_SDK_COMMIT
    assert PINNED_SDK_COMMIT.startswith("d079ad33")


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


def _stage_checkpoint_entries(tmp_path, monkeypatch, record):
    """Stage BOTH entry classes (activation + shared-state) with real PB
    writers under one mover; returns (resolver, consumer, entry_paths)."""
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover = _hex64(f"mover-{tmp_path}")
    queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
    root = tmp_path / 'residency'
    entries, rows, paths = {}, {}, []
    for entry in record["activation_entries"] + record["shared_state_entries"]:
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
    set_lease_helper_root(PB_PIN_ROOT)
    activate_staged_tier_policy("ram,ssd")
    return residency_resolver(), consumer, paths


def test_strict_checkpoint_roundtrip_pinned_never_opens_pool(tmp_path, monkeypatch):
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
    cotangents, shared_adjoint, shared_pass = load_adjoint_checkpoint(
        adjoint_space(tmp_path), record)
    assert torch.equal(cotangents[(0, 0)], tensor)
    assert _states_equal(shared_adjoint[(0, 0)], state)
    assert shared_pass == {0: {"captured": None}}
    assert not any(opened_path in paths for opened_path in opened)
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert any(row.get('pin_id') for row in report['serving_tiers'])
    assert _pins_live(tmp_path, consumer) == []


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
    assert resolver.report()['bytes_from_stage'] == 0


def test_strict_checkpoint_shared_state_unmapped_refuses(tmp_path, monkeypatch):
    from prismaquant.joint_adjoint_checkpoints import (
        adjoint_space, load_adjoint_checkpoint)
    record, _tensor, _state = _write_checkpoint(tmp_path)
    _strict(monkeypatch, _write_map(tmp_path, {}))
    set_lease_helper_root(PB_PIN_ROOT)
    activate_staged_tier_policy("ram,ssd")
    with pytest.raises(TierPolicyRefused):
        load_adjoint_checkpoint(adjoint_space(tmp_path), record)


def test_lease_helper_reads_authoritative_env_automatically(tmp_path, monkeypatch):
    """Production discovery: the PB-injected PRISMABUILD_READER_HELPER_ROOT
    is read with no explicit setter and no user knob."""
    from prismaquant.staged_lease import (
        HELPER_ROOT_ENV_VAR, lease_helper_root)
    _pb()
    assert lease_helper_root() is None
    monkeypatch.setenv(HELPER_ROOT_ENV_VAR, str(PB_PIN_ROOT))
    assert lease_helper_root() == str(PB_PIN_ROOT)
    monkeypatch.setenv(HELPER_ROOT_ENV_VAR, "/nonexistent-root")
    assert lease_helper_root() == "/nonexistent-root"


def test_lease_helper_env_without_helper_refuses(tmp_path, monkeypatch):
    """An authoritatively-named root that names nothing usable refuses
    instead of importing whatever happens to be around — unavailable on a
    fresh interpreter, divergent when another tree is already imported
    (two trees must never mix). Either way: clear refusal, zero pool."""
    from prismaquant.staged_lease import HELPER_ROOT_ENV_VAR
    _pb()
    path, _ = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'s': (path, staged, None)}))
    monkeypatch.setenv(HELPER_ROOT_ENV_VAR, str(tmp_path / 'no-such-root'))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        with pytest.raises(LeaseRefused,
                           match="lease-helper-(unavailable|divergent)"):
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
    set_lease_helper_root(PB_PIN_ROOT)
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
    set_lease_helper_root(PB_PIN_ROOT)

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
