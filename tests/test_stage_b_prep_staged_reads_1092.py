"""The Stage B preparation reads its declared inputs off the stage (PQ #1092).

``test_stage_b_prep_io_1070`` proves the preparation's data manifest names
every file the generator opens. These tests prove that, with
``--data-manifest-sha256``, the generator opens none of them at their pool
path: every declared input is opened at the stage path PrismaBuild's
residency map names, digest-checked, under a real lease.

The stack is the one ``test_strict_reader_tier_enforcement`` builds: the
pinned PrismaBuild SDK, a real queue and claim row, fragments and material
written by PrismaBuild's own writers, and a composed map naming them. The
staged copies are exactly the manifest's entries: a whole file for a whole
file, the ``[0, 8 + length)`` range for a safetensors header.

* Every file the generator opens under the pool is opened at a stage path
  the map names, and the declared pool path is never opened.
* A declared input the stage does not hold refuses, naming it, and its pool
  path is never opened.
* A staged copy rewritten after it was vouched for refuses, and bytes that
  do not hash to the entry's digest refuse.
* The flag refuses without a residency map.
* Every entry the submission adds declares its SHA-256, and the submission
  carries the flags that make the action strict.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from prismaquant import stage_b_prep_io as io  # noqa: E402
from prismaquant.residency_map import (  # noqa: E402
    ENV_VAR, SCHEMA, residency_map_key, residency_report)
from test_stage_b_prep_io_1070 import (  # noqa: E402
    _recorded_reads, generator_args, prep_fixture, write_submission)
from test_strict_reader_tier_enforcement import (  # noqa: E402,F401  (autouse)
    MANIFEST, STAGE_TIER, _forget_state, _hex64, _launch_env, _pb, _pb_publish,
    _pb_queue)


@pytest.fixture(autouse=True)
def _unbind_staged_reads():
    io.reset_staged_reads_for_tests()
    yield
    io.reset_staged_reads_for_tests()


def _stage_manifest(tmp_path, monkeypatch, manifest, *, skip=(), corrupt=()):
    """Stage every manifest entry through the real PB writers; bind the map.

    ``skip`` names declared paths left unstaged, ``corrupt`` paths whose
    staged copy is altered after the map and material are written. Returns
    ``{declared path: stage path}``.
    """
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover = _hex64(f"mover-{tmp_path}")
    _queue, stage = _pb_queue(tmp_path, pool_mod, consumer)
    staged, rows, entries = {}, {}, {}
    for number, entry in enumerate(manifest["entries"]):
        path = entry["path"]
        if path in skip:
            continue
        with open(path, "rb") as handle:
            handle.seek(entry["offset"])
            blob = handle.read(entry["bytes"])
        copy = stage / f"{number:03d}-{Path(path).name}"
        copy.write_bytes(blob)
        key = residency_map_key(path, entry["offset"])
        rows[key] = (Path(path), copy)
        entries[key] = {"stage_path": str(copy), "bytes": len(blob),
                        "offset": entry["offset"],
                        "sha256": hashlib.sha256(blob).hexdigest()}
        staged[path] = str(copy)
    _pb_publish(rl, map_mod, tmp_path / "residency", stage, consumer, mover,
                MANIFEST, rows)
    map_path = tmp_path / "residency" / "residency.json"
    map_path.write_text(json.dumps({
        "schema": SCHEMA, "tier_id": STAGE_TIER, "stage_root": str(stage),
        "manifest_sha256": MANIFEST, "leads": [mover], "generation": 3,
        "entries": entries}))
    for path in corrupt:
        copy = Path(staged[path])
        raw = bytearray(copy.read_bytes())
        raw[-1] ^= 0xFF
        # Rewritten in place: same size, so only the digest can tell.
        copy.write_bytes(bytes(raw))
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    return staged


def _campaign(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    campaign, receipt_path, _spans = prep_fixture(pool)
    metadata = pool / "metadata-generation"
    submission, manifest, _template = write_submission(
        pool, campaign, receipt_path, metadata)
    return pool, campaign, receipt_path, metadata, submission, manifest


def _strict_argv(pool, campaign, receipt_path, metadata):
    return generator_args(pool, campaign, receipt_path, metadata) + [
        "--data-manifest-sha256", MANIFEST]


def _pool_inputs(opened, pool, metadata):
    inputs = set()
    for raw in opened:
        path = Path(os.path.normpath(os.path.abspath(raw)))
        if path.is_relative_to(pool) and not path.is_relative_to(metadata) \
                and not path.is_dir():
            inputs.add(str(path))
    return inputs


def test_every_input_the_generator_opens_is_a_staged_copy(tmp_path, monkeypatch):
    import regenerate_joint_quanta as regen

    pool, campaign, receipt_path, metadata, _submission, manifest = _campaign(tmp_path)
    staged = _stage_manifest(tmp_path, monkeypatch, manifest)

    code, opened = _recorded_reads(lambda: regen.main(
        _strict_argv(pool, campaign, receipt_path, metadata)))
    assert code == 0
    assert (metadata / "records" / "records.json").is_file()
    pool_reads = _pool_inputs(opened, pool, metadata)
    assert not pool_reads, f"opened at the pool path, not the stage: {sorted(pool_reads)}"
    opened_normal = {os.path.normpath(os.path.abspath(raw)) for raw in opened}
    served = {path for path, copy in staged.items() if copy in opened_normal}
    # What the unstrict run opens at the pool (test_stage_b_prep_io_1070),
    # every one of them now at its stage copy.
    expected = {str(campaign[key]) for key in (
        "plan_path", "prepared_path", "parent_path", "derivation", "partition")}
    expected |= {str(receipt_path)}
    plan = json.loads(campaign["plan_path"].read_text())
    model = Path(plan["model"])
    expected |= {str(model / "model.safetensors.index.json")}
    expected |= {str(path) for path in model.glob("*.safetensors")}
    assert expected <= served, sorted(expected - served)
    report = residency_report()
    assert report is not None and report["fallbacks"] == []


def test_an_unstaged_input_refuses_and_never_reads_the_pool(tmp_path, monkeypatch, capsys):
    import regenerate_joint_quanta as regen

    pool, campaign, receipt_path, metadata, _submission, manifest = _campaign(tmp_path)
    prepared = json.loads(campaign["prepared_path"].read_text())
    pickle_path = os.path.normpath(prepared["production_cache"]["path"])
    _stage_manifest(tmp_path, monkeypatch, manifest, skip={pickle_path})

    code, opened = _recorded_reads(lambda: regen.main(
        _strict_argv(pool, campaign, receipt_path, metadata)))
    assert code == regen.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "production cache" in err and "not staged" in err
    assert pickle_path not in {os.path.normpath(os.path.abspath(p)) for p in opened}
    assert not _pool_inputs(opened, pool, metadata)


def test_a_staged_copy_rewritten_in_place_refuses(tmp_path, monkeypatch, capsys):
    """The lease's identity fence refuses a copy changed after it was vouched for."""
    import regenerate_joint_quanta as regen

    pool, campaign, receipt_path, metadata, _submission, manifest = _campaign(tmp_path)
    _stage_manifest(tmp_path, monkeypatch, manifest,
                    corrupt={os.path.normpath(str(receipt_path))})

    code, opened = _recorded_reads(lambda: regen.main(
        _strict_argv(pool, campaign, receipt_path, metadata)))
    assert code != 0
    assert "Stage A proof" in capsys.readouterr().err
    assert not _pool_inputs(opened, pool, metadata)


def test_bytes_that_miss_the_entry_digest_refuse(tmp_path, monkeypatch, capsys):
    """The reader hashes what it read, whatever the lease handed it."""
    import regenerate_joint_quanta as regen
    from prismaquant import staged_whole_file

    pool, campaign, receipt_path, metadata, _submission, manifest = _campaign(tmp_path)
    _stage_manifest(tmp_path, monkeypatch, manifest)
    real = staged_whole_file.read_staged_entry
    receipt = os.path.normpath(str(receipt_path))

    def altered(resolver, path, staged, *, label):
        raw = real(resolver, path, staged, label=label)
        return raw + b" " if os.path.normpath(str(path)) == receipt else raw

    monkeypatch.setattr(staged_whole_file, "read_staged_entry", altered)
    code, _opened = _recorded_reads(lambda: regen.main(
        _strict_argv(pool, campaign, receipt_path, metadata)))
    assert code == regen.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "Stage A proof" in err and "staged bytes hash to" in err


def test_the_flag_refuses_without_a_residency_map(tmp_path, monkeypatch, capsys):
    import regenerate_joint_quanta as regen

    pool, campaign, receipt_path, metadata, _submission, _manifest = _campaign(tmp_path)
    monkeypatch.delenv(ENV_VAR, raising=False)
    code = regen.main(_strict_argv(pool, campaign, receipt_path, metadata))
    assert code == regen.EXIT_REFUSED
    assert "--residency stage" in capsys.readouterr().err
    assert not metadata.exists()


def test_the_submission_declares_every_digest_and_the_strict_flags(tmp_path):
    _pool, campaign, _receipt, _metadata, submission, manifest = _campaign(tmp_path)
    head = io.head_phase_entries(json.loads(campaign["parent_path"].read_text()))
    head_keys = {(row["path"], row["offset"]) for row in head}
    for entry in manifest["entries"]:
        if (entry["path"], entry["offset"]) in head_keys:
            continue
        with open(entry["path"], "rb") as handle:
            handle.seek(entry["offset"])
            blob = handle.read(entry["bytes"])
        assert entry["sha256"] == hashlib.sha256(blob).hexdigest(), entry["path"]
    raw = Path(submission["data_manifest"]["path"]).read_bytes()
    assert gzip.decompress(raw)
    assert submission["strict_read_flags"] == [
        "--data-manifest-sha256", hashlib.sha256(raw).hexdigest(),
        "--allowed-tiers", "ram,ssd"]


def test_bind_refuses_a_second_digest_and_a_malformed_one(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "absent-map.json"))
    with pytest.raises(io.PreparationReadRefused, match="64-character"):
        io.bind_staged_reads(manifest_sha256="abc", allowed_tiers="ram,ssd")
    io.bind_staged_reads(manifest_sha256=MANIFEST, allowed_tiers="ram,ssd",
                         own_outputs=[tmp_path / "a"])
    again = io.bind_staged_reads(manifest_sha256=MANIFEST, allowed_tiers="ram,ssd",
                                 own_outputs=[tmp_path / "b"])
    assert again.owns(tmp_path / "a" / "x") and again.owns(tmp_path / "b" / "y")
    with pytest.raises(io.PreparationReadRefused, match="already bound"):
        io.bind_staged_reads(manifest_sha256="d" * 64, allowed_tiers="ram,ssd")


def test_the_bound_reader_serves_read_bound_while_bound(tmp_path, monkeypatch):
    from prismaquant import tessera_joint_allocation as allocation

    monkeypatch.setenv(ENV_VAR, str(tmp_path / "absent-map.json"))
    reads = io.bind_staged_reads(manifest_sha256=MANIFEST, allowed_tiers="ram,ssd")
    calls = []
    monkeypatch.setattr(reads, "whole", lambda path, *, sha256, where:
                        calls.append((str(path), sha256, where)) or b"{}")
    target = tmp_path / "control.json"
    target.write_bytes(b"{}")
    record = {"path": str(target), "sha256": hashlib.sha256(b"{}").hexdigest()}
    allocation._BOUND_BYTES.clear()
    assert allocation._read_bound(record, "control") == b"{}"
    assert calls == [(str(target), record["sha256"], "control")]
    io.reset_staged_reads_for_tests()
    assert allocation.BOUND_READER is None
