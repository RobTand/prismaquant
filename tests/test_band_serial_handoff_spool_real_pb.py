"""A producer quantum's handoff through PrismaBuild's local output spool (PQ #996).

With ``PRISMABUILD_PRODUCED_SPOOL_ROOT`` in its launch environment, a
produced-output owner writes each group to a PB-reserved directory on its
own host, and PB exports the group to its canonical paths as an ordinary
action. The handoff emitter uses the same writer as Stage A, so its groups
take that route too. ``owner-states.pkl`` and ``handoff.json`` are one
more group (PQ #1015), submitted only after every entry group's export is
acknowledged, so the record never names an entry that has not landed. The
handoff's template is write-only, so each group is committed at its origin
once PrismaBuild acknowledged its export, against the identities the export
receipt recorded (PQ #1075). The export actions run on a real private fleet
(claim, execute, finish), pinned to the PrismaBuild generation
``test_band_serial_handoff_produced`` uses.

The module is marked ``own_process`` for the reason
``test_stage_a_produced_boundary_chain`` records (PQ #1008).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import test_stage_a_produced_boundary_chain as chain
from test_band_serial_handoff_produced import (
    ORIGIN_PIN, band_campaign, check_consumer_binds, check_origin_batches,
    emit_handoff, pb_source, producer_owner)
from test_stage_a_produced_boundary_chain import _isolated_launch_context  # noqa: F401

# PQ #1008: one pinned prismabuild per process (tests/conftest.py).
pytestmark = pytest.mark.own_process


def test_the_handoff_exports_through_the_spool_before_its_record(
        tmp_path, monkeypatch):
    from fleet_sdk import require_prismabuild_sdk
    require_prismabuild_sdk()
    pin = json.loads(ORIGIN_PIN.read_text())
    root = Path(pin["bundle_root"])
    assert root.is_dir(), "the pinned PB generation must be provisioned"
    for name, digest in pin["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    _src, pb_repo = pb_source(monkeypatch)
    from prismaquant.produced_output_spool import MAX_ENV, ROOT_ENV

    producer, consumer, storage = band_campaign(tmp_path)
    spool = tmp_path / "local-output"
    capacity = {"cpu": 4, "mem_gb": 4}
    publication, q, _env = producer_owner(
        tmp_path, pb_repo, producer, storage,
        producer_environment={ROOT_ENV: str(spool), MAX_ENV: str(1 << 20)},
        claim_capacity=capacity)
    with chain._fleet(q, tmp_path, capacity=capacity) as fleet:
        emitters = []
        published, plane = emit_handoff(producer, storage, publication,
                                        emitters=emitters)
    outcomes = [json.loads(line) for line in fleet.stdout.splitlines()
                if line.startswith("{")]
    # One export per group: each probe's three batches in groups of two,
    # then the record group (owner states and handoff.json, PQ #1015).
    assert len(outcomes) == 5 and all(row["rc"] == 0 for row in outcomes), (
        outcomes, fleet.stderr[-2000:])
    assert not [path for path in spool.rglob("*")
                if path.is_file() and path.parent.name == "payload"], (
        "the spool still holds a group's payload")
    handoff = check_consumer_binds(published, plane, consumer, publication)
    check_origin_batches(publication, producer, published, handoff)
    # PQ #1225: the emitter keeps each group's export record for the row's
    # counters, and each names the export action that landed it.
    report = getattr(emitters[0], "export_report", None)
    assert report is not None, "the emitter keeps no export report"
    exports = report["local_spool"]["exports"]
    assert len(exports) == len(outcomes)
    assert all(row["export_key"] and row["landed_unix"] and row["released_unix"]
               and row["bytes"] > 0 for row in exports)
    json.dumps(report)
