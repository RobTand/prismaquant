"""A Stage B spill bound sealed under strict reads (PQ #1139).

#1138 made the generator's spill block read ``config.json`` (the routed
top-k) through ``StagedPreparationReads.whole`` when strict reads are bound;
``streaming_source_plan`` declares the config as a header read, so the
preparation's data manifest stages it. These tests run the generator with
``--data-manifest-sha256``, ``--replay-mode spill`` and a source model that
has a config:

* every record seals the spill bound the plain run seals, and the config is
  read only through the staged read: its pool path is never opened, by the
  spill block or by the profile detection before it;
* with the config's header read removed from the manifest, the run refuses,
  naming it, and never opens its pool path.

The strict binding here is ``StagedPreparationReads`` over a resolver double
that answers ``staged_read``/``staged_range`` with copies of exactly the
manifest's entries; the lease read is replaced by a plain read of the copy.
``test_stage_b_prep_staged_reads_1092`` drives the real residency map and
reader lease under pbtest; this file needs no PrismaBuild.
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
from test_quantum_executable_readset import (  # noqa: E402
    _exec_campaign, _exec_receipt)
from test_stage_b_prep_io_1070 import (  # noqa: E402
    _recorded_reads, generator_args, write_submission)
from test_stageb_prepared_inputs_bridge import (  # noqa: E402
    QWEN3_CONFIG, SPILL_EXECUTION)
from test_stageb_readset_source_coverage import (  # noqa: E402
    _source_model, _with_source_paths)

SPILL = ("--replay-mode", "spill", "--replay-regime", "capture_batch=2")


@pytest.fixture(autouse=True)
def _unbind_staged_reads():
    io.reset_staged_reads_for_tests()
    yield
    io.reset_staged_reads_for_tests()


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _spill_campaign(pool: Path):
    """``test_stage_b_prep_io_1070.prep_fixture`` with a source ``config.json``
    and the spill test's calibration shape, sealed before Stage A binds the
    plan digest."""
    campaign = _exec_campaign(pool)
    model, _spans = _source_model(pool)
    (model / "config.json").write_text(json.dumps(QWEN3_CONFIG))
    plan = json.loads(campaign["plan_path"].read_text())
    plan["model"] = str(model)
    plan["execution"] = {**plan["execution"], **SPILL_EXECUTION}
    campaign["plan_path"].write_text(json.dumps(plan))
    campaign["plan_sha"] = _sha(campaign["plan_path"])
    parent = _with_source_paths(json.loads(campaign["parent_path"].read_text()), model)
    head = model / "head.bin"
    head.write_bytes(b"\x01" * 100)
    parent["entries"][0]["path"] = str(head)
    campaign["parent_path"].write_text(json.dumps(parent))
    campaign["parent_sha"] = _sha(campaign["parent_path"])
    receipt, space = _exec_receipt(pool, campaign)
    receipt_path = space / "adjoint-capture.json"
    receipt_path.write_text(json.dumps(receipt))
    return campaign, receipt_path, model / "config.json"


class _StageDouble:
    """The residency resolver's two lookups, answered from stage copies."""

    def __init__(self, entries: dict):
        self.entries = entries

    def staged_read(self, path, *, expected_sha256=None):
        return self.entries.get((os.path.normpath(str(path)), 0))

    def staged_range(self, path, offset, nbytes):
        entry = self.entries.get((os.path.normpath(str(path)), offset))
        return entry if entry is not None and entry["bytes"] >= nbytes else None


def _bind_stage(tmp_path, monkeypatch, manifest, digest, *, skip=()):
    """Copy every manifest entry to the stage and bind strict reads to it.

    ``skip`` names declared paths left out, as if the manifest did not
    declare them. Returns ``{declared path: stage copy}``.
    """
    from prismaquant import staged_whole_file

    stage = tmp_path / "stage"
    stage.mkdir()
    entries, copies = {}, {}
    for number, entry in enumerate(manifest["entries"]):
        path = os.path.normpath(entry["path"])
        if path in skip:
            continue
        with open(path, "rb") as handle:
            handle.seek(entry["offset"])
            blob = handle.read(entry["bytes"])
        copy = stage / f"{number:03d}-{Path(path).name}"
        copy.write_bytes(blob)
        entries[(path, entry["offset"])] = {
            "stage_path": str(copy), "offset": entry["offset"],
            "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        copies[path] = str(copy)
    monkeypatch.setattr(staged_whole_file, "read_staged_entry",
                        lambda resolver, path, staged, *, label:
                        Path(staged["stage_path"]).read_bytes())
    reads = io.StagedPreparationReads(_StageDouble(entries), manifest_sha256=digest)
    monkeypatch.setattr(io, "_STAGED_READS", reads)
    io._install_bound_reader(reads)
    return copies


def _layout(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    campaign, receipt_path, config = _spill_campaign(pool)
    metadata = pool / "metadata-generation"
    write_submission(pool, campaign, receipt_path, metadata)
    wire = (pool / "submission" / "read-manifest.json.gz").read_bytes()
    manifest = json.loads(gzip.decompress(wire))
    argv = generator_args(pool, campaign, receipt_path, metadata) + [
        *SPILL, "--data-manifest-sha256", hashlib.sha256(wire).hexdigest()]
    return pool, campaign, receipt_path, config, manifest, argv


def _bounds(records: Path) -> dict:
    return {path.name: json.loads(path.read_text())["executable_readset"]["spill_bound"]
            for path in sorted(records.glob("layer-*.json"))}


def _opened(opened) -> set[str]:
    return {os.path.normpath(os.path.abspath(raw)) for raw in opened}


def test_a_strict_run_seals_the_spill_bound_reading_the_config_off_the_stage(
        tmp_path, monkeypatch):
    import regenerate_joint_quanta as regen
    from prismaquant.joint_replay_spill import check_spill_bound

    pool, campaign, receipt_path, config, manifest, argv = _layout(tmp_path)
    declared = {os.path.normpath(entry["path"]) for entry in manifest["entries"]}
    assert str(config) in declared, "the source plan declares the config"
    # The plain run's bounds, sealed from the config at its pool path.
    plain = pool / "plain-generation"
    assert regen.main(generator_args(pool, campaign, receipt_path, plain)
                      + list(SPILL)) == 0
    expected = _bounds(plain / "records")
    assert len(expected) == 4

    copies = _bind_stage(tmp_path, monkeypatch, manifest,
                         argv[argv.index("--data-manifest-sha256") + 1])
    code, opened = _recorded_reads(lambda: regen.main(argv))
    assert code == 0
    bounds = _bounds(pool / "metadata-generation" / "records")
    assert bounds == expected
    for bound in bounds.values():
        assert check_spill_bound(bound) == bound["reservation_bytes"]
        assert bound["capture_batch"] == 2
    opened = _opened(opened)
    assert str(config) not in opened, "config.json was read at its pool path"
    assert copies[str(config)] in opened


def test_a_strict_run_without_the_config_header_read_refuses(
        tmp_path, monkeypatch, capsys):
    import regenerate_joint_quanta as regen

    _pool, _campaign, _receipt, config, manifest, argv = _layout(tmp_path)
    _bind_stage(tmp_path, monkeypatch, manifest,
                argv[argv.index("--data-manifest-sha256") + 1], skip={str(config)})
    capsys.readouterr()
    code, opened = _recorded_reads(lambda: regen.main(argv))
    assert code == regen.EXIT_REFUSED
    err = capsys.readouterr().err
    assert str(config) in err and "not staged" in err
    assert str(config) not in _opened(opened)
