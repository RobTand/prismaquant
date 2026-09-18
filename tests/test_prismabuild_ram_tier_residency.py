"""The ram tier's half of the residency map, read the way the stage's is.

PrismaBuild's ram tier (RobTand/prismabuild#640) promotes staged ranges onto a
``noswap`` tmpfs and overlays the map with the tmpfs copies: an entry keeps the
stage path it already had and gains ``ram_path``, and the map's header gains
``ram_tier_id``, ``ram_root`` and ``ram_epoch``. A tmpfs empties on reboot
while the map survives on the shared mount, so the map's epoch -- checked
against the epoch the pool's tier record announces -- is the whole of the ram
half's identity in time.

Every test here mutates the driver (the map, the tier record, or the bytes on
the ram root) rather than the fixture, so a passing assertion says the
reader's own check bit.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from prismaquant.residency_map import (
    ENV_VAR, SCHEMA, TIERS_DIR_ENV_VAR, bind_residency_manifest,
    residency_map_key, residency_report, residency_resolver,
    reset_residency_resolver_for_tests,
)
from test_prismabuild_residency_map import (
    MANIFEST, LEAD, _bind, _pool_cache, _stage, _wire_cell,
)

RAM_TIER = 'ram:dl380g10'
EPOCH = '1789771929-aba6e46e41fb03ef'
LATER_EPOCH = '1789788888-9c1d2e3f4a5b'


@pytest.fixture(autouse=True)
def _forget_resolver(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(TIERS_DIR_ENV_VAR, raising=False)
    reset_residency_resolver_for_tests()
    yield
    reset_residency_resolver_for_tests()


def _promote(tmp_path, staged):
    """Lay the staged copies onto a ram root, as a promotion node does."""
    root = tmp_path / 'ram' / 'prewarm'
    root.mkdir(parents=True, exist_ok=True)
    ram = {}
    for key, stage in staged.items():
        target = root / stage.name
        target.write_bytes(stage.read_bytes())
        ram[key] = target
    return root, ram


def _announce(tmp_path, epoch=EPOCH, *, tier=RAM_TIER, name='tiers', body=None):
    """File the tier record a consumer compares the map's ram_epoch against.

    Under ``<tmp>/tiers`` by default, which is where the reader derives the
    pool's tiers directory from when the map sits under ``<tmp>/residency``.
    """
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    record = {'schema': 'prismabuild.storage_tier.v1', 'tier': 'ram',
              'tier_id': tier, 'mountpoint': '/ram/prewarm', 'epoch': epoch}
    (directory / f'{tier}.json').write_text(json.dumps(body if body is not None else record))
    return directory


def _write_ram_map(tmp_path, stage_root, paths, staged, ram, ram_root, *,
                   epoch=EPOCH, tier=RAM_TIER, announce=None,
                   ram_path_of=None, name='consumer.map.json'):
    """A map of the ram-overlay generation, filed beside a ``tiers`` sibling.

    ``announce`` names the header fields to carry: all three by default
    (``None``), none for an old-format map (``()``), or a partial set for a
    header that says less than its entries do.
    """
    announce = ('ram_tier_id', 'ram_root', 'ram_epoch') if announce is None else announce
    entries = {}
    for key in paths:
        entry = {
            'stage_path': str(staged[key]),
            'bytes': paths[key].stat().st_size,
            'offset': 0,
            'sha256': hashlib.sha256(paths[key].read_bytes()).hexdigest(),
        }
        if ram is not None:
            entry['ram_path'] = (str(ram[key]) if ram_path_of is None
                                 else ram_path_of(ram[key]))
        entries[residency_map_key(str(paths[key]), 0)] = entry
    header = {'ram_tier_id': tier, 'ram_root': str(ram_root), 'ram_epoch': epoch}
    body = {
        'schema': SCHEMA,
        'tier_id': 'prismabuild-stage:dl380g10',
        'stage_root': str(stage_root),
        'manifest_sha256': MANIFEST,
        'leads': [LEAD],
        'generation': len(paths),
        'entries': entries,
        **{field: header[field] for field in announce},
    }
    path = tmp_path / 'residency' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


def _ram_wire(tmp_path, *, corrupt_ram=(), announce=None, ram_path_of=None):
    """A wire, its stage copy, its ram copy, a map naming both, and a record.

    The map and the record share one epoch, which is the live state; a test
    that wants a stale half re-announces the record afterwards, the way a
    reboot actually would.
    """
    cell, wire, blob = _wire_cell(tmp_path)
    root, staged = _stage(tmp_path, {'w': wire})
    ram_root, ram = _promote(tmp_path, staged)
    for key in corrupt_ram:
        blob_bytes = ram[key].read_bytes()
        ram[key].write_bytes(blob_bytes[:-1] + bytes([blob_bytes[-1] ^ 0xFF]))
    _announce(tmp_path)
    map_path = _write_ram_map(tmp_path, root, {'w': wire}, staged, ram, ram_root,
                              announce=announce, ram_path_of=ram_path_of)
    return cell, wire, blob, staged, ram, map_path


# --------------------------------------------------------------------------
# the ram half is preferred when its epoch is the announced one
# --------------------------------------------------------------------------

def test_a_current_epoch_ram_copy_is_offered_beside_the_stage_copy(tmp_path, monkeypatch):
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    offered = residency_resolver().staged_read(
        wire, expected_sha256=cell['record']['blob_sha256'])
    # The stage copy is still the entry's own vouching; the ram copy is the
    # preferred servant, offered only while its epoch is the announced one.
    assert offered['stage_path'] == str(staged['w'])
    assert offered['ram_path'] == str(ram['w'])
    report = residency_report()
    assert report['ram_tier_id'] == RAM_TIER
    assert report['ram_root'] == str(ram['w'].parent)
    assert report['ram_epoch'] == EPOCH
    assert 'ram_refused' not in report


def test_the_wire_reader_reads_a_current_ram_copy_and_counts_it(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob and digest == cell['record']['blob_sha256']
    report = residency_report()
    assert report['ram_hits'] == 1
    assert report['bytes_from_ram'] == len(blob)
    assert report['bytes_from_stage'] == 0 and report['bytes_from_pool'] == 0
    assert report['ram_fallback_count'] == 0


def test_a_ram_wire_whose_bytes_differ_falls_back_to_the_stage(tmp_path, monkeypatch):
    """Same size, same published digest, different bytes: the stage serves."""
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path, corrupt_ram=('w',))
    assert ram['w'].stat().st_size == staged['w'].stat().st_size
    assert ram['w'].read_bytes() != staged['w'].read_bytes()
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob and digest == cell['record']['blob_sha256']
    report = residency_report()
    assert report['ram_hits'] == 0 and report['bytes_from_ram'] == 0
    assert report['hits'] == 1 and report['bytes_from_stage'] == len(blob)
    assert report['bytes_from_pool'] == 0
    assert report['ram_fallback_count'] == 1
    assert 'receipt digest' in report['ram_fallbacks'][0]['reason']


def test_a_ram_wire_released_between_the_stat_and_the_open_reads_the_stage(
        tmp_path, monkeypatch):
    """A vanished ram copy is a miss, not an ENOENT: the stage vouches on."""
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    original = os.open

    def vanishing(candidate, *args, **kwargs):
        if str(candidate) == str(ram['w']):
            ram['w'].unlink()
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(os, 'open', vanishing)
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob and digest == cell['record']['blob_sha256']
    report = residency_report()
    assert report['ram_hits'] == 0 and report['bytes_from_ram'] == 0
    assert report['hits'] == 1 and report['bytes_from_stage'] == len(blob)
    assert report['bytes_from_pool'] == 0
    assert 'unreadable' in report['ram_fallbacks'][0]['reason']


# --------------------------------------------------------------------------
# the reboot hole: an epoch nobody announces anymore
# --------------------------------------------------------------------------

def test_a_stale_epoch_map_reads_the_stage_not_the_dead_tmpfs(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    # The tmpfs died and was remounted: the record announces a new epoch while
    # the map, and its ram paths, survive on the shared mount from the old one.
    _announce(tmp_path, epoch=LATER_EPOCH)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    offered = residency_resolver().staged_read(
        wire, expected_sha256=cell['record']['blob_sha256'])
    assert 'ram_path' not in offered
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob and digest == cell['record']['blob_sha256']
    report = residency_report()
    assert report['ram_hits'] == 0 and report['bytes_from_ram'] == 0
    assert report['hits'] == 1 and report['bytes_from_stage'] == len(blob)
    assert 'stale' in report['ram_refused']
    assert EPOCH in report['ram_refused'] and LATER_EPOCH in report['ram_refused']


def test_an_epoch_rollover_between_lookups_switches_the_ram_half_off(tmp_path, monkeypatch):
    """The tmpfs died mid-run: the record's new epoch retires the old half."""
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    resolver = residency_resolver()
    first = resolver.staged_read(wire, expected_sha256=cell['record']['blob_sha256'])
    assert first['ram_path'] == str(ram['w'])
    record = tmp_path / 'tiers' / f'{RAM_TIER}.json'
    replaced = json.loads(record.read_text())
    replaced['epoch'] = LATER_EPOCH
    fresh = tmp_path / 'tiers' / '.replaced.json'
    fresh.write_text(json.dumps(replaced))
    os.replace(fresh, record)
    second = resolver.staged_read(wire, expected_sha256=cell['record']['blob_sha256'])
    assert 'ram_path' not in second
    assert 'stale' in residency_report()['ram_refused']


# --------------------------------------------------------------------------
# the announced record itself is missing or unreadable
# --------------------------------------------------------------------------

def test_no_tier_record_fails_closed_on_the_ram_half_only(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    (tmp_path / 'tiers' / f'{RAM_TIER}.json').unlink()
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    offered = residency_resolver().staged_read(
        wire, expected_sha256=cell['record']['blob_sha256'])
    assert 'ram_path' not in offered and offered['stage_path'] == str(staged['w'])
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob
    report = residency_report()
    assert report['hits'] == 1 and report['bytes_from_stage'] == len(blob)
    assert 'unreadable' in report['ram_refused']


def test_an_unreadable_tier_record_fails_closed_on_the_ram_half(tmp_path, monkeypatch):
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    (tmp_path / 'tiers' / f'{RAM_TIER}.json').write_text('{not json')
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    offered = residency_resolver().staged_read(
        wire, expected_sha256=cell['record']['blob_sha256'])
    assert 'ram_path' not in offered
    report = residency_report()
    assert report['bytes_from_ram'] == 0
    assert 'tier record' in report['ram_refused']


def test_a_tier_record_without_an_epoch_fails_closed_on_the_ram_half(tmp_path, monkeypatch):
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    _announce(tmp_path, body={'schema': 'prismabuild.storage_tier.v1',
                              'tier': 'ram', 'tier_id': RAM_TIER,
                              'mountpoint': '/ram/prewarm'})
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    offered = residency_resolver().staged_read(
        wire, expected_sha256=cell['record']['blob_sha256'])
    assert 'ram_path' not in offered
    assert 'epoch' in residency_report()['ram_refused']


# --------------------------------------------------------------------------
# the ram file's own identity
# --------------------------------------------------------------------------

def test_a_ram_copy_of_the_wrong_size_is_refused_and_the_stage_serves(tmp_path, monkeypatch):
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    ram['w'].write_bytes(blob + b'\0')
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    offered = residency_resolver().staged_read(
        wire, expected_sha256=cell['record']['blob_sha256'])
    assert 'ram_path' not in offered and offered['stage_path'] == str(staged['w'])
    report = residency_report()
    assert report['ram_fallback_count'] == 1
    assert 'size differs' in report['ram_fallbacks'][0]['reason']


def test_a_ram_copy_that_is_not_a_regular_file_is_refused(tmp_path, monkeypatch):
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    ram['w'].unlink()
    ram['w'].symlink_to(staged['w'])
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    offered = residency_resolver().staged_read(
        wire, expected_sha256=cell['record']['blob_sha256'])
    assert 'ram_path' not in offered
    assert 'not a regular file' in residency_report()['ram_fallbacks'][0]['reason']


def test_a_missing_ram_copy_is_refused_and_the_stage_serves(tmp_path, monkeypatch):
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    ram['w'].unlink()
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    offered = residency_resolver().staged_read(
        wire, expected_sha256=cell['record']['blob_sha256'])
    assert 'ram_path' not in offered
    assert 'unreadable' in residency_report()['ram_fallbacks'][0]['reason']


def test_a_stage_copy_that_failed_its_fence_does_not_take_the_ram_copy_down(
        tmp_path, monkeypatch):
    """The ram half stands on the announced epoch and its own identity."""
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    staged['w'].unlink()
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    offered = residency_resolver().staged_read(
        wire, expected_sha256=cell['record']['blob_sha256'])
    assert offered['ram_path'] == str(ram['w'])
    assert offered['stage_path'] == str(staged['w'])


# --------------------------------------------------------------------------
# the old format binds exactly as before
# --------------------------------------------------------------------------

def test_an_old_format_map_binds_exactly_as_before(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob = _wire_cell(tmp_path)
    root, staged = _stage(tmp_path, {'w': wire})
    map_path = _write_ram_map(tmp_path, root, {'w': wire}, staged, None,
                              tmp_path / 'ram' / 'prewarm', announce=())
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    offered = residency_resolver().staged_read(
        wire, expected_sha256=cell['record']['blob_sha256'])
    assert set(offered) == {'declared_path', 'stage_path', 'bytes', 'offset', 'sha256'}
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob and digest == cell['record']['blob_sha256']
    report = residency_report()
    assert report['hits'] == 1 and report['bytes_from_stage'] == len(blob)
    assert report['ram_hits'] == 0 and report['bytes_from_ram'] == 0
    assert report['ram_fallback_count'] == 0 and report['ram_fallbacks'] == []
    for absent in ('ram_tier_id', 'ram_root', 'ram_epoch', 'ram_refused'):
        assert absent not in report


# The upstream test beside this one exercises ``staged_range``, the ranged
# shard reader this sealed-era branch does not carry (its whole-file reader is
# ``staged_read``, covered above); the epoch retirement it proves is the same
# one ``test_an_epoch_rollover_between_lookups_switches_the_ram_half_off``
# proves on the whole-file path.


# --------------------------------------------------------------------------
# a map that does not say what the ram half requires is refused whole
# --------------------------------------------------------------------------

def test_a_ram_path_outside_the_ram_root_refuses_the_map_whole(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob, staged, ram, map_path = _ram_wire(
        tmp_path, ram_path_of=lambda copy: str(tmp_path / 'elsewhere' / 'cell.wire'))
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob and digest == cell['record']['blob_sha256']
    report = residency_report()
    assert report['entries'] == 0 and report['hits'] == 0
    assert report['bytes_from_pool'] == len(blob)
    assert 'outside' in report['refused']


def test_an_entry_naming_a_ram_path_without_the_announced_header_refuses_whole(
        tmp_path, monkeypatch):
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path, announce=())
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    # The map is read on the first lookup; a refusal there falls to the pool.
    assert residency_resolver().staged_read(
        wire, expected_sha256=cell['record']['blob_sha256']) is None
    report = residency_report()
    assert report['entries'] == 0
    assert 'ram root' in report['refused']


def test_a_header_announcing_only_part_of_the_ram_half_with_entries_refuses_whole(
        tmp_path, monkeypatch):
    cell, wire, blob, staged, ram, map_path = _ram_wire(
        tmp_path, announce=('ram_tier_id', 'ram_root'))
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    assert residency_resolver().staged_read(
        wire, expected_sha256=cell['record']['blob_sha256']) is None
    report = residency_report()
    assert report['entries'] == 0
    assert 'epoch' in report['refused']


# --------------------------------------------------------------------------
# where the announced record is read from
# --------------------------------------------------------------------------

def test_the_tiers_dir_env_var_names_where_the_record_is_read(tmp_path, monkeypatch):
    """The map's sibling ``tiers`` is the default; the variable overrides it."""
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    (tmp_path / 'tiers' / f'{RAM_TIER}.json').unlink()
    elsewhere = _announce(tmp_path, name='elsewhere-tiers')
    moved = tmp_path / 'moved'
    (moved / 'residency').mkdir(parents=True)
    os.replace(map_path, moved / 'residency' / 'consumer.map.json')
    monkeypatch.setenv(ENV_VAR, str(moved / 'residency' / 'consumer.map.json'))
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(elsewhere))
    _bind()
    offered = residency_resolver().staged_read(
        wire, expected_sha256=cell['record']['blob_sha256'])
    assert offered['ram_path'] == str(ram['w'])


def test_the_tiers_dir_parameter_of_the_resolver_is_honored(tmp_path, monkeypatch):
    from prismaquant.residency_map import ResidencyResolver
    cell, wire, blob, staged, ram, map_path = _ram_wire(tmp_path)
    (tmp_path / 'tiers' / f'{RAM_TIER}.json').unlink()
    elsewhere = _announce(tmp_path, name='parameter-tiers')
    resolver = ResidencyResolver(map_path, tiers_dir=elsewhere)
    resolver.bind_manifest_sha256(MANIFEST)
    offered = resolver.staged_read(wire, expected_sha256=cell['record']['blob_sha256'])
    assert offered['ram_path'] == str(ram['w'])
