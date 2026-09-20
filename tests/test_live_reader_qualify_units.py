"""Unit gates for the live-reader probe verdicts (PQ live-reader lane).

Pure decision helpers only: unknown evidence never proves zero reads,
exact-attempt binding never counts another lease, RAM-first serving is
enforced rather than printed. Run via published pbtest at -10 in the
scoped PQ environment. No live cluster needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.live_reader_qualify import (  # noqa: E402
    UNQUALIFIED,
    attempt_refs,
    eval_pool_delta,
    expect_serving_tier,
    identity_presence,
)


def test_unqualified_status_is_nonzero_nonone() -> None:
    assert UNQUALIFIED == 2


def test_pool_delta_observed_only_when_complete() -> None:
    before = {"/mnt/shared": {"client_read": 100, "server_read": 50}}
    after = {"/mnt/shared": {"client_read": 100, "server_read": 50}}
    assert eval_pool_delta(before, after, "/mnt/shared") == (True, 0)
    assert eval_pool_delta(before, {"/mnt/shared": {"client_read": 140,
                                                    "server_read": 90}},
                           "/mnt/shared") == (True, 40)
    assert eval_pool_delta({}, after, "/mnt/shared") == (False, None)
    assert eval_pool_delta(before, {}, "/mnt/shared") == (False, None)
    assert eval_pool_delta(before, after, "/mnt/other") == (False, None)
    assert eval_pool_delta(before, {"/mnt/shared": {}}, "/mnt/shared") == (
        False, None)
    assert eval_pool_delta(before, None, "/mnt/shared") == (False, None)


def test_serving_tier_enforces_ram_first() -> None:
    ram = {"tier_id": "ram:dl380g10", "epoch": "e", "pin_id": "p",
           "range_ref": "k"}
    ssd = {"tier_id": "prismabuild-stage:dl380g10", "epoch": "",
           "pin_id": "p", "range_ref": "k"}
    assert expect_serving_tier(
        ram_offered=True, ram_allowed=True, ssd_allowed=True,
        serving=ram) == (True, "ram-served")
    assert expect_serving_tier(
        ram_offered=True, ram_allowed=True, ssd_allowed=True,
        serving=ssd)[0] is False
    assert expect_serving_tier(
        ram_offered=False, ram_allowed=True, ssd_allowed=True,
        serving=ssd) == (True, "ssd-served")
    assert expect_serving_tier(
        ram_offered=False, ram_allowed=False, ssd_allowed=False,
        serving=ssd)[0] is False
    assert expect_serving_tier(
        ram_offered=False, ram_allowed=True, ssd_allowed=True,
        serving={})[0] is False
    assert expect_serving_tier(
        ram_offered=None, ram_allowed=True, ssd_allowed=True,
        serving=ram)[0] is False


def test_attempt_refs_binds_exact_attempt() -> None:
    refs = {
        "r1": {"attempt": {"nonce": "n1", "scope_id": "s1"}},
        "r2": {"attempt": {"nonce": "n1", "scope_id": "s2"}},
        "r3": {"attempt": {"nonce": "n9", "scope_id": "s1"}},
        "r4": "not-a-ref",
        "r5": {"attempt": "not-an-attempt"},
    }
    assert attempt_refs(refs, "n1", "s1") == ["r1"]
    assert attempt_refs(refs, "n1", "s9") == []
    assert attempt_refs({}, "n1", "s1") == []


def test_identity_presence_reports_actual_names() -> None:
    env = {"PRISMABUILD_ACTION_KEY": "k", "PRISMABUILD_ACTION_NONCE": ""}
    assert identity_presence(env) == {
        "PRISMABUILD_ACTION_KEY": True,
        "PRISMABUILD_ACTION_NONCE": False,
        "PRISMABUILD_ACTION_SCOPE": False,
        "PRISMABUILD_READER_HELPER_ROOT": False}
