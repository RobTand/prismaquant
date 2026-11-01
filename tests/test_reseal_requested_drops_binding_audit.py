"""An assembled bundle proves the same setting removal its consumer requests."""
from __future__ import annotations

import json

import pytest

from test_reseal_campaign_identity import (
    BOUND, KNOBS, NEW, OLD, _arm_result, make_row, run, write_pins,
)


def _assembled_case(tmp_path, *, dropped=()):
    new = {**NEW, "encoder_source_sha256": OLD["encoder_source_sha256"]}
    pins = write_pins(tmp_path / "assembled-pins.json", new=new, drop_settings=dropped)
    arm = _arm_result(tmp_path / "arm.json", OLD, new, dropped=dropped, engagement=None)
    bundle = tmp_path / "bundle.json"
    assert run("proof-bundle", "--pins", pins, "--arm", arm, "--out", bundle) == 0
    assert json.loads(bundle.read_text())["ok"] is True
    row = tmp_path / "row-0001"
    make_row(row, settings=BOUND)
    return row, pins, bundle, new


def _snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("command", ["dry-run", "migrate"])
def test_public_reseal_refuses_new_requested_drops_after_bundle_assembly(tmp_path, command):
    row, _original_pins, bundle, new = _assembled_case(tmp_path)
    bundle_before = bundle.read_bytes()
    requested = write_pins(tmp_path / "requested-pins.json", new=new, drop_settings=tuple(KNOBS))
    before = _snapshot(row)

    code = run(command, "--pins", requested, "--proof", bundle, "--row", row)

    assert code == 2, "unchanged no-drop arm must not authorize additional setting removal"
    assert _snapshot(row) == before
    assert bundle.read_bytes() == bundle_before


@pytest.mark.parametrize("dropped", [(), tuple(KNOBS)])
def test_public_reseal_retains_matching_requested_drop_scope(tmp_path, dropped):
    row, pins, bundle, _new = _assembled_case(tmp_path, dropped=dropped)

    assert run("dry-run", "--pins", pins, "--proof", bundle, "--row", row) == 0
    assert run("migrate", "--pins", pins, "--proof", bundle, "--row", row) == 0
    assert run("verify", "--pins", pins, "--row", row) == 0
    manifest = json.loads((row / "cost.anchors.json").read_text())
    assert manifest["identity"]["settings"] == {
        key: value for key, value in BOUND.items() if key not in dropped
    }
    assert manifest["identity_migration"][-1]["dropped_settings"] == {
        key: BOUND[key] for key in dropped
    }
