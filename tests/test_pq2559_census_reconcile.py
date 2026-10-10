"""Record-only seed reconcile for the #1588 Stage B census (PQ #2559).

The census grades the 102 legacy seed rows through the adopt gates without
adopting a byte, freezes the legacy manifest by digest, and reconciles the
legacy rung ids against the pre-dispatch packet's open rungs. Pricing rows
re-run the full gate through ``--seed-checkpoint``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch

from prismaquant import schemas
from prismaquant import tessera_campaign as campaign

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = REPO_ROOT / "docs/results/pq1588_predispatch_2026-10-09.json"


def _seed_journal(root: Path, units: dict) -> Path:
    """Write a synthetic seed journal; returns the manifest path."""
    from prismaquant.cost_stage_checkpoint import (
        canonical_json_sha256,
        write_unit,
    )

    seed_inputs = {
        "currency": "output_mse_under_route_activation_contract",
        "calibration": {"draw": "fixture"},
        "input_global_scale_policy": "fixture",
        "units": {
            name: {
                "scoring_rows": {"sha256": "rows"},
                "input_global_scale": 1.0,
            }
            for name in units
        },
    }
    seed_sha = canonical_json_sha256(seed_inputs, where="seed fixture")
    parts = root / "cost.anchors.json.parts"
    parts.mkdir(parents=True)
    for name, anchors in units.items():
        write_unit(
            parts,
            stage="Tessera campaign",
            qname=name,
            identity_sha256=seed_sha,
            state={"anchors": anchors, "wire_records": {}},
        )
    manifest = root / "cost.anchors.json"
    manifest.write_text(
        json.dumps({"identity_sha256": seed_sha, "identity": seed_inputs})
    )
    return manifest


def _anchors(qname, *rungs) -> list:
    return [
        {
            "qname": qname,
            "family": family,
            "format_name": f"{family}@{rate}",
            "body_rate_q256": rate,
            "dloss": 1.0,
            "dloss_stderr": 0.0,
            "memory_bytes": 8,
            "bits_per_param": 4.0,
            "activation_contract": "c",
            "activation_quantized": False,
            "wire_bytes": 8,
            "seconds": 0.1,
        }
        for family, rate in rungs
    ]


def _expected(units=("a",), **overrides) -> dict:
    identity = {
        "currency": "output_mse_under_route_activation_contract",
        "calibration": {"draw": "fixture"},
        "input_global_scale_policy": "fixture",
        "units": {name: {"input_global_scale": 1.0} for name in units},
    }
    identity.update(overrides)
    return identity

def _refuse_adopt(name, state, where):
    raise AssertionError(f"record-only grades {name}, never adopts it")


def test_record_only_grades_without_adopting(tmp_path):
    manifest = _seed_journal(tmp_path, {"a": _anchors("a", ("FAM", 1024))})
    wire_dir = tmp_path / "run-wire"
    wire_dir.mkdir()
    record = campaign._adopt_seed_checkpoint(
        manifest,
        None,
        targets=["a"],
        wire_dir=wire_dir,
        adopt=_refuse_adopt,
        admits=lambda name, fmt: True,
        identity_sha256="census",
        expected_identity=_expected(),
        record_only=True,
        row_class_by_unit={"a": "dense"},
    )
    assert record["record_only"] is True
    assert record["units"] == ["a"]
    assert record["run_binding_complete"] is False
    (decision,) = record["decisions"]
    assert decision["qname"] == "a"
    assert decision["gate"] == "pass"
    assert decision["gates"]["contract"] == "pass"
    assert decision["gates"]["envelope"] == "pass"
    assert decision["gates"]["unit_scoring"]["scoring_rows"] == "deferred"
    assert decision["gates"]["unit_scoring"]["input_global_scale"] == "pass"
    assert decision["gates"]["scope"] == "deferred"
    assert decision["gates"]["rungs"] == "pass"
    assert decision["gates"]["rung_ids"] == [("FAM", 1024, "dense")]
    assert list(wire_dir.iterdir()) == []


def test_record_only_cites_a_contract_mismatch_without_raising(tmp_path):
    manifest = _seed_journal(tmp_path, {"a": _anchors("a", ("FAM", 1024))})
    record = campaign._adopt_seed_checkpoint(
        manifest,
        None,
        targets=["a"],
        wire_dir=tmp_path,
        adopt=_refuse_adopt,
        admits=lambda name, fmt: True,
        identity_sha256="census",
        expected_identity=_expected(currency="other"),
        record_only=True,
        row_class_by_unit={"a": "dense"},
    )
    assert record["contract"]["gate"] == "fail"
    (decision,) = record["decisions"]
    assert decision["gate"] == "fail"
    assert "currency" in decision["reason"]


def test_record_only_cites_a_broken_envelope_without_raising(tmp_path):
    from prismaquant.cost_stage_checkpoint import unit_path

    manifest = _seed_journal(
        tmp_path, {"a": _anchors("a", ("FAM", 1024)), "b": _anchors("b", ("FAM", 2048))}
    )
    shard = unit_path(manifest.with_name(manifest.name + ".parts"), "b")
    raw = bytearray(shard.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    shard.write_bytes(bytes(raw))
    record = campaign._adopt_seed_checkpoint(
        manifest,
        None,
        targets=["a", "b"],
        wire_dir=tmp_path,
        adopt=_refuse_adopt,
        admits=lambda name, fmt: True,
        identity_sha256="census",
        expected_identity=_expected(units=("a", "b")),
        record_only=True,
        row_class_by_unit={"a": "dense", "b": "dense"},
    )
    by_name = {decision["qname"]: decision for decision in record["decisions"]}
    assert by_name["a"]["gate"] == "pass"
    assert by_name["b"]["gate"] == "fail"
    assert by_name["b"]["gates"]["envelope"] == "fail"
    assert record["units"] == ["a"]


def test_record_only_still_refuses_a_missing_journal(tmp_path):
    with pytest.raises(RuntimeError, match="no unit shards"):
        campaign._adopt_seed_checkpoint(
            tmp_path / "cost.anchors.json",
            None,
            targets=["a"],
            wire_dir=tmp_path,
            adopt=_refuse_adopt,
            admits=lambda name, fmt: True,
            identity_sha256="census",
            expected_identity=_expected(),
            record_only=True,
        )


def test_freeze_binds_path_to_digest_and_refuses_drift(tmp_path):
    manifest = _seed_journal(tmp_path, {"a": _anchors("a", ("FAM", 1024))})
    freeze_path = tmp_path / "freeze.json"
    frozen = campaign.freeze_legacy_anchors(manifest, out_path=freeze_path)
    assert frozen["legacy_path"] == str(manifest)
    schemas.validate_legacy_freeze(
        json.loads(freeze_path.read_text()), path=str(freeze_path)
    )
    assert campaign.require_legacy_bytes(frozen) == manifest
    assert campaign.require_legacy_bytes(freeze_path) == manifest
    assert campaign.freeze_legacy_anchors(manifest, out_path=freeze_path) == frozen
    raw = bytearray(manifest.read_bytes())
    raw[-2] ^= 0xFF
    manifest.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="drifted"):
        campaign.require_legacy_bytes(frozen)
    with pytest.raises(RuntimeError, match="moved under a frozen digest"):
        campaign.freeze_legacy_anchors(manifest, out_path=freeze_path)


def test_freeze_shape_refusals():
    with pytest.raises(schemas.SchemaValidationError):
        schemas.validate_legacy_freeze({"schema": "wrong"}, path="freeze")
    with pytest.raises(schemas.SchemaValidationError):
        schemas.validate_legacy_freeze(
            {
                "schema": "prismaquant.tessera_legacy_freeze.v1",
                "legacy_path": "relative/path.json",
                "sha256": "0" * 64,
            },
            path="freeze",
        )


def test_open_rungs_come_from_the_packet():
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    open_ids = campaign.open_rung_ids_from_predispatch_packet(packet)
    assert len(open_ids) == 14
    for family, rate, row_class in open_ids:
        assert isinstance(family, str) and type(rate) is int
        assert row_class in ("dense", "routed")


def test_rung_reconciliation_counts_new_and_total():
    legacy = [("FAM", 1024, "dense"), ("FAM", 2048, "dense")]
    requested = [("FAM", 1024, "dense"), ("FAM", 4096, "dense")]
    report = campaign.reconcile_seed_rungs(legacy, requested)
    assert report["legacy_rung_count"] == 2
    assert report["new_rung_ids"] == [("FAM", 4096, "dense")]
    assert report["total_rung_count"] == 3


def test_reconcile_flags_travel_together():
    args = SimpleNamespace(
        census_out=None,
        streaming=False,
        reconcile_seed_checkpoint="seed",
        reconcile_open_packet=None,
        reconcile_log_out=None,
        legacy_freeze_out=None,
    )
    with pytest.raises(RuntimeError, match="missing"):
        campaign.require_reconcile_args(args)
    args.reconcile_open_packet = args.reconcile_log_out = args.legacy_freeze_out = "x"
    with pytest.raises(RuntimeError, match="census row only"):
        campaign.require_reconcile_args(args)
    args.census_out = "census.json"
    args.streaming = True
    with pytest.raises(RuntimeError, match="load-all"):
        campaign.require_reconcile_args(args)
    args.streaming = False
    campaign.require_reconcile_args(args)


def _spec(tmp_path) -> Path:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "model": str(tmp_path / "model"),
                "cwd": str(tmp_path),
                "python": "python3",
                "campaign_argv": [],
                "env": {},
            }
        )
    )
    return spec_path


def test_census_manifest_carries_the_reconcile_flags(tmp_path):
    args = SimpleNamespace(
        spec=str(_spec(tmp_path)),
        workspace=str(tmp_path),
        timeout_s=7200,
        submit=False,
        reconcile_seed_checkpoint="/mnt/shared/seed/cost.anchors.json",
        reconcile_open_packet="packet.json",
        reconcile_log_out="reconcile.json",
        legacy_freeze_out="freeze.json",
    )
    assert dispatch.cmd_census(args) == 0
    (row,) = json.loads((tmp_path / "census-manifest.json").read_text())
    argv = dispatch._inner_campaign_argv(row)
    for flag, value in (
        ("--reconcile-seed-checkpoint", "/mnt/shared/seed/cost.anchors.json"),
        ("--reconcile-open-packet", "packet.json"),
        ("--reconcile-log-out", "reconcile.json"),
        ("--legacy-freeze-out", "freeze.json"),
    ):
        assert argv[argv.index(flag) + 1] == value
    assert argv.index("--census-out") < argv.index("--reconcile-seed-checkpoint")


def test_census_manifest_without_reconcile_flags_is_unchanged(tmp_path):
    args = SimpleNamespace(
        spec=str(_spec(tmp_path)),
        workspace=str(tmp_path),
        timeout_s=7200,
        submit=False,
    )
    assert dispatch.cmd_census(args) == 0
    (row,) = json.loads((tmp_path / "census-manifest.json").read_text())
    assert not any(
        part.startswith("--reconcile-") or part == "--legacy-freeze-out"
        for part in dispatch._inner_campaign_argv(row)
    )


def test_census_refuses_partial_reconcile_flags(tmp_path):
    args = SimpleNamespace(
        spec=str(_spec(tmp_path)),
        workspace=str(tmp_path),
        timeout_s=7200,
        submit=False,
        reconcile_seed_checkpoint="seed",
    )
    with pytest.raises(RuntimeError, match="all four flags"):
        dispatch.cmd_census(args)


def test_reconcile_driver_writes_freeze_and_log_without_a_price_row(tmp_path):
    """The census reconcile entry point runs on CPU over fixtures (PQ #2559)."""
    seed = tmp_path / "seed"
    seed.mkdir()
    manifest = _seed_journal(
        seed,
        {
            "a": _anchors("a", ("FAM", 1024), ("FAM", 2048)),
            "b": _anchors("b", ("FAM", 1024)),
        },
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    log = campaign.reconcile_census_seeds(
        legacy_path=manifest,
        packet_path=PACKET_PATH,
        freeze_out=tmp_path / "freeze.json",
        log_out=tmp_path / "reconcile.json",
        census_payload={"counts": {"a": 8, "b": 8}, "model": "fixture"},
        hessian_identity={"draw": "fixture"},
        static_scales={"a": 1.0, "b": 1.0},
        static_scale_policy="fixture",
        admits=lambda name, fmt: True,
        row_class_by_unit={"a": "dense", "b": "routed_moe"},
        cache_dir=cache,
    )
    assert log["schema"] == "prismaquant.tessera_seed_reconcile.v1"
    assert log["legacy_units"] == 2
    assert log["gate_tally"] == {"pass": 2, "fail": 0}
    assert log["run_binding_complete"] is False
    rungs = log["rung_reconciliation"]
    assert rungs["legacy_rung_count"] == 3
    assert rungs["new_rung_count"] == 14
    assert rungs["total_rung_count"] == 17
    assert (tmp_path / "freeze.json").is_file()
    assert (tmp_path / "reconcile.json").is_file()
    assert list(tmp_path.rglob("cost.pkl")) == []
    assert list(cache.iterdir()) == []
