"""The real preparation reads immutable staged inputs, with no canonical opens.

The scientific submission still declares its original ranges and digests.
Execution uses an exact native v1 read contract over those ranges, published by
the existing fixture materializer. This does not qualify the submission's v2
scientific wire format as a native launch contract. Digest, missing-input,
unbound-launch and retained output-root controls use the real resolver.
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
from prismaquant.residency_map import ENV_VAR, residency_report  # noqa: E402
from prismaquant.staged_manifest import READ_ENV  # noqa: E402
from staged_native_fixtures import bind, native_staged_state, publish_inputs  # noqa: E402
from test_stage_b_prep_io_1070 import (  # noqa: E402
    _recorded_reads, generator_args, prep_fixture, write_submission)


@pytest.fixture(autouse=True)
def _unbind_staged_reads(native_staged_state):
    yield


def _stage_manifest(tmp_path, monkeypatch, manifest, *, skip=(), corrupt=()):
    """Publish exact ranges through the native materializer and bind its owner."""
    ranges = {}
    for entry in manifest['entries']:
        if entry['path'] not in skip:
            ranges.setdefault(Path(entry['path']), []).append(
                (entry['offset'], entry['offset'] + entry['bytes']))
    binding = publish_inputs(tmp_path, list(ranges), ranges=ranges)
    resolver = bind(monkeypatch, binding)
    staged = {extent.source_path: extent.object_path for extent in resolver.manifest.extents}
    for path in corrupt:
        copy = Path(staged[path])
        raw = bytearray(copy.read_bytes())
        raw[-1] ^= 0xFF
        copy.chmod(0o600)
        copy.write_bytes(bytes(raw))
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
        "--data-manifest-sha256", os.environ.get(READ_ENV, 'a' * 64)]


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
    """The immutable reader refuses an object changed after publication."""
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

    def altered(resolver, path, staged, *, label, max_bytes=None):
        raw = real(resolver, path, staged, label=label, max_bytes=max_bytes)
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
    source = tmp_path / 'control.json'
    source.write_bytes(b'{}')
    binding = publish_inputs(tmp_path, [source])
    bind(monkeypatch, binding)
    digest = binding['read_contract_sha256']
    with pytest.raises(io.PreparationReadRefused, match="64-character"):
        io.bind_staged_reads(manifest_sha256="abc", allowed_tiers="ram,ssd")
    io.bind_staged_reads(manifest_sha256=digest, allowed_tiers="ssd",
                         own_outputs=[tmp_path / "a"])
    again = io.bind_staged_reads(manifest_sha256=digest, allowed_tiers="ssd",
                                 own_outputs=[tmp_path / "b"])
    assert again.owns(tmp_path / "a" / "x") and again.owns(tmp_path / "b" / "y")
    with pytest.raises(io.PreparationReadRefused, match="already bound"):
        io.bind_staged_reads(manifest_sha256="d" * 64, allowed_tiers="ram,ssd")


def test_the_bound_reader_serves_read_bound_while_bound(tmp_path, monkeypatch):
    from prismaquant import stage_inputs

    target = tmp_path / "control.json"
    target.write_bytes(b"{}")
    binding = publish_inputs(tmp_path, [target])
    bind(monkeypatch, binding)
    reads = io.bind_staged_reads(manifest_sha256=binding['read_contract_sha256'],
                                 allowed_tiers="ssd")
    calls = []
    real = reads.whole
    def observed(path, *, sha256, where, max_bytes=None):
        calls.append((str(path), sha256, where))
        return real(path, sha256=sha256, where=where, max_bytes=max_bytes)
    monkeypatch.setattr(reads, "whole", observed)
    record = {"path": str(target), "sha256": hashlib.sha256(b"{}").hexdigest()}
    stage_inputs._BOUND_BYTES.clear()
    assert stage_inputs.read_bound(record, "control") == b"{}"
    assert calls == [(str(target), record["sha256"], "control")]
    io.reset_staged_reads_for_tests()
    assert stage_inputs.BOUND_READER is None
