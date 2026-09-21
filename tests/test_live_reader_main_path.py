"""Main-path binding against real SDK-produced serving/pin data (PQ lane).

Acquires a real pin through the installed SDK's real writers, opens it
through real ``open_pinned`` (the exact serving shape the probe
consumes), and runs the probe's binding helpers against that data:
pin-record read, selected-path match, exact-attempt refs, census, and
tier expectation. A tampered tier and a foreign pin fail the same
helpers. Run via published pbtest at -10 in the scoped PQ environment.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from fleet_sdk import require_prismabuild_sdk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.live_reader_qualify import (  # noqa: E402
    _read_pin_record,
    attempt_refs,
    expect_serving_tier,
)

TIER = "prismabuild-stage:dl380g10"
CONSUMER = "c" * 64
MOVER = "e" * 64
NONCE = "n1"
SCOPE = "s1"


def _fixture(tmp_path):
    require_prismabuild_sdk()
    from prismaquant.staged_lease import inject_installed_sdk_for_tests
    rl = inject_installed_sdk_for_tests()
    import prismabuild.pool as pool_mod
    import prismabuild.residency_map as map_mod
    queue = pool_mod.PoolQueue(tmp_path / "queue")
    queue.ensure_layout()
    # Default leases location, so the probe's SDK reads find the pins.
    root = queue.root / "residency"
    stage = tmp_path / "stage"
    staged = stage / "w.bin"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"\x51" * 2048)
    key = map_mod.residency_map_key("/mnt/shared/w.bin", 0)
    digest = hashlib.sha256(b"\x51" * 2048).hexdigest()
    map_mod.write_fragment(root, {
        "schema": map_mod.RESIDENCY_MAP_FRAGMENT_SCHEMA_V1,
        "consumer_action_key": CONSUMER, "mover_action_key": MOVER,
        "tier_id": TIER, "stage_root": str(stage),
        "manifest_sha256": "a" * 64,
        "entries": {key: {"stage_path": str(staged), "bytes": 2048,
                          "sha256": digest, "offset": 0}}})
    generation = rl.mint_generation()
    rl.write_material(
        root, consumer_action_key=CONSUMER, mover_action_key=MOVER,
        tier_id=TIER, stage_root=str(stage), manifest_sha256="a" * 64,
        generation=generation,
        entries={key: {"stage_path": str(staged), "bytes": 2048,
                       "sha256": digest,
                       "file_id": rl.stat_identity(str(staged))}})
    acquired = rl.acquire(
        queue, consumer_action_key=CONSUMER,
        attempt={"nonce": NONCE, "scope_id": SCOPE}, tier_id=TIER, epoch="",
        span={"start_bytes": 0, "end_bytes": 2048},
        holder={"host": "test-host", "pid": 7}, acquire_token="token-1",
        covers=[{"mover_action_key": MOVER, "manifest_sha256": "a" * 64}],
        expected={key: {"bytes": 2048, "sha256": digest}},
        residency_root=root)
    assert acquired["ok"], acquired
    return rl, queue, root, acquired, key, str(staged)


def test_binding_holds_on_real_serving_data(tmp_path) -> None:
    rl, queue, root, acquired, key, staged = _fixture(tmp_path)
    fd, serving = rl.open_pinned(
        queue, acquired["pin"], acquired["ref_id"], key)
    try:
        assert serving["pin_id"] == acquired["pin_id"]
        assert serving["range_ref"] == key
        owner, pin = _read_pin_record(rl, queue, serving["pin_id"])
        assert pin is not None
        selected = [str(e.get("stage_path", ""))
                    for e in pin.get("entries", [])
                    if isinstance(e, dict)
                    and str(e.get("key") or "") == key]
        assert selected == [staged]
        mine = attempt_refs(pin.get("refs", {}), NONCE, SCOPE)
        assert mine == [acquired["ref_id"]]
        assert attempt_refs(pin.get("refs", {}), NONCE, "other") == []
        owners, tainted = rl.live_for(queue, None)
        assert owners.get(os.path.normpath(staged)) == [acquired["pin_id"]]
        assert tainted == []
        ok, reason = expect_serving_tier(
            ram_offered=False, allowed={"ssd"}, lease_tier_id=TIER,
            serving=serving)
        assert (ok, reason) == (True, "ssd-served")
    finally:
        os.close(fd)
    assert rl.release(queue, acquired["pin_id"], acquired["ref_id"],
                      consumer_action_key=CONSUMER) is True
    owner_after, pin_after = _read_pin_record(rl, queue, acquired["pin_id"])
    assert (owner_after, pin_after) == (None, None)
    assert attempt_refs({}, NONCE, SCOPE) == []


def test_tampered_tier_and_foreign_pin_fail(tmp_path) -> None:
    rl, queue, root, acquired, key, staged = _fixture(tmp_path)
    fd, serving = rl.open_pinned(
        queue, acquired["pin"], acquired["ref_id"], key)
    try:
        assert expect_serving_tier(
            ram_offered=False, allowed={"ssd"}, lease_tier_id=TIER,
            serving={**serving, "tier_id": "arc:dl380g10"})[0] is False
        owner, pin = _read_pin_record(rl, queue, serving["pin_id"])
        assert pin is not None
        assert attempt_refs(pin.get("refs", {}), "n9", SCOPE) == []
        missing_owner, missing_pin = _read_pin_record(rl, queue, "0" * 64)
        assert (missing_owner, missing_pin) == (None, None)
    finally:
        os.close(fd)
