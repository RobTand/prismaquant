"""PQ #2789: a raised retirement validation must not poison its wakeup.

Reuse the acceptance lane's fake sealed SDK and virtual clock without changing
that file. Exercise the production adapter and owner, not a duplicate gate.
"""
import pytest

from prismaquant.stage_a_produced_output import BoundaryProducedPublicationFailed
from test_keyed_egress_owner_2789 import (
    assert_retained, deferred, drain, ending, lane, refill_step, release,
)


def failed_validation():
    # PB can raise acquiring stage_ownership_lock before touching egress.
    raise OSError("transient ownership lock failure")


@pytest.mark.parametrize("republished", [False, True])
def test_terminal_validation_exception_recovers_then_gates_successor(lane, republished):
    lane.owner._produced_plan["ahead_groups"] = 1
    assert release(lane, wait=False) is False
    lane.client.observe = lambda: ending("done", 1.0)
    lane.client.reply = failed_validation
    lane.clock.now = 2.0
    drain(lane)
    assert_retained(lane)
    assert lane.owner.telemetry["produced_group_release_failures"] == 1

    # A successful response is now available for the SAME terminal generation.
    # A returned keyed deferral may instead mean PB republished that key.
    lane.client.reply = lambda: deferred() if republished else {"ok": True}
    lane.clock.now = 4.0
    drain(lane)
    assert lane.client.retire_calls == [(0.0, "old"), (2.0, "old"), (4.0, "old")]
    assert lane.client.terminal_reads == [2.0, 4.0]
    for now in (6.0, 8.0):
        lane.clock.now = now
        drain(lane)
    assert len(lane.client.retire_calls) == 3
    assert not lane.clock.sleeps, "nonblocking drain must still overlap compute"
    if republished:
        assert_retained(lane)
        assert lane.client.terminal_reads == [2.0, 4.0, 6.0, 8.0]
        lane.client.observe = lambda: ending("done", 2.0)
        lane.client.reply = lambda: {"ok": True}
        lane.clock.now = 10.0
        drain(lane)
        assert lane.client.retire_calls[-1] == (10.0, "old")
    assert lane.group["retired"] and lane.group["context"] is None
    assert lane.key not in lane.owner._produced_release_pending
    assert lane.key not in lane.owner._produced_held
    assert lane.key not in lane.owner._produced_ahead


def test_terminal_validation_exceptions_keep_existing_attempt_bound(lane):
    lane.owner._produced_plan["ahead_groups"] = 1
    assert release(lane, wait=False) is False
    lane.client.observe = lambda: ending("done", 1.0)
    lane.client.reply = failed_validation
    attempts = lane.owner.PRODUCED_RELEASE_ATTEMPTS
    for i in range(1, attempts + 1):
        lane.clock.now = float(2 * i)
        drain(lane)
        assert_retained(lane)
    assert len(lane.client.retire_calls) == 1 + attempts
    assert lane.owner.telemetry["produced_group_release_failures"] == attempts
    assert lane.key not in lane.owner._produced_release_pending
    assert lane.key in lane.owner._produced_release_abandoned
    lane.clock.now += 2.0
    drain(lane)
    assert len(lane.client.retire_calls) == 1 + attempts
    assert not lane.clock.sleeps


def test_funding_recovers_raised_validation_with_original_deadline(lane):
    lane.owner._produced_plan["ahead_groups"] = 1
    lane.client.observe = lambda: ending("done", 1.0)

    def reply():
        if len(lane.client.retire_calls) == 1:
            return deferred()
        if len(lane.client.retire_calls) == 2:
            return failed_validation()
        return {"ok": True}

    lane.client.reply = reply
    lane.clock.on_sleep = lambda: assert_retained(lane)
    with lane.owner._produced_lock.held():
        # A failed candidate is retained as release debt; this funding call
        # propagates its own shortfall under the existing owner policy.
        with pytest.raises(BoundaryProducedPublicationFailed):
            lane.owner._produced_fund_for_read(
                lambda: refill_step(lane), (), deadline=1.25)
        assert lane.clock.now == 0.5
        assert_retained(lane)
        # Recovery uses the original budget, not a fresh staging timeout.
        lane.clock.now = 1.0
        out = lane.owner._produced_fund_for_read(
            lambda: refill_step(lane), (), deadline=1.25)
    assert out["ok"] and lane.group["retired"]
    assert lane.client.retire_calls == [(0.0, "old"), (0.5, "old"), (1.0, "old")]
    assert lane.client.terminal_reads == [0.5, 1.0]
    assert lane.clock.now == 1.0
    assert any(error["step"] == "reclaim-credit"
               and "transient ownership lock failure" in error["reason"]["error"]
               for error in lane.owner._produced_release_errors)
