"""PQ #2789: local owner handshake, not a replacement PB wait service.

Use the real publication and release/funding control flow. Only PB's sealed
public client and storage bookkeeping unrelated to egress are faked. No fleet,
PB source checkout, historical git objects, tensors or GPU are required.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import stage_a_produced_output as adapter
from prismaquant.streamed_boundary_artifacts import (
    BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts,
)


KEY = "egress-action"


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.on_sleep = lambda: None

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0, "owner pacing must not spin"
        self.sleeps.append(seconds)
        self.now += seconds
        self.on_sleep()


def ending(state=None, generation=None, *, ambiguous=False, unreadable=False):
    """The public PoolQueue.current_ending mapping (not a tuple/bool)."""
    record = None if state is None else {
        "action_key": KEY, "status": state, "published_unix": generation,
    }
    return {
        "state": state, "path": None if state is None else Path(state) / KEY,
        "record": record, "generation": generation, "history": [],
        "unreadable": ([{"state": "done", "path": Path("done") / KEY,
                         "reason": "unreadable"}] if unreadable else []),
        "ambiguous": ambiguous, "coexisting": [],
    }


def deferred(key=KEY, reason="own-egress-in-flight"):
    out = {"ok": False, "refusal": "egress-incomplete", "receipt": {
        "complete": False, "errors": [], "deferred_own": [reason],
    }}
    if key is not None:
        out["egress_action_key"] = key
        out["receipt"].update(egress_action_key=key, egress_state="published")
    return out


class Queue:
    def __init__(self, sdk, root):
        self.sdk, self.root = sdk, Path(root)

    def current_ending(self, action_key, *, max_bytes=None):
        assert action_key == KEY
        self.sdk.terminal_reads.append(self.sdk.clock.now)
        out = self.sdk.observe()
        if isinstance(out, Exception):
            raise out
        return out

    def terminal_outcome_covers(self, record, *, action_key=None, terminal=None):
        # Same distinct return shape as PB, with exact-generation coverage.
        observed = self.current_ending(action_key or record["action_key"])
        if observed["ambiguous"] or observed["unreadable"]:
            return None
        if observed["state"] not in ("done", "failed"):
            return None
        if record.get("published_unix") != observed["generation"]:
            return None
        return observed["state"], observed["record"]

    def withdrawal_covers(self, record, *, action_key=None, withdrawn=None):
        observed = self.current_ending(action_key or record["action_key"])
        if observed["state"] != "withdrawn" or observed["ambiguous"]:
            return None
        if record.get("published_unix") != observed["generation"]:
            return None
        return observed["record"]


class Client:
    """One selected client supplies both retirement and PoolQueue.

    The fake receipt is intentionally authoritative: a terminal read cannot
    alter credit or retire anything. Only a successful retire_batch does that.
    """
    REQUIRED_TESSERA_CONTRACT = None

    def __init__(self, clock):
        self.clock = clock
        self.retire_calls = []
        self.terminal_reads = []
        self.queue_roots = []
        self.observe = lambda: ending()
        self.reply = lambda: deferred()
        self.retired = False
        self.refill_calls = []

    def PoolQueue(self, root):
        self.queue_roots.append(Path(root))
        return Queue(self, root)

    def retire_batch(self, queue, instance, template, batch_id, **kwargs):
        self.retire_calls.append((self.clock.now, batch_id))
        out = self.reply()
        if out.get("ok"):
            self.retired = True
        return out

    def materialization_state(self, queue, **kwargs):
        return {"state": "ACTIVE", "action_key": "mover"}

    def refill_window(self, queue, instance, template, *, tier):
        self.refill_calls.append(self.clock.now)
        if not self.retired:
            return {"ok": False, "refusal": "tier-reservation-unavailable"}
        return {"ok": True}


@pytest.fixture
def lane(monkeypatch, tmp_path):
    import time
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    client = Client(clock)
    # The already-selected, sealed SDK generation; never import another PB.
    monkeypatch.setattr(adapter, "_produced_output_module", lambda: client)
    publication = adapter.BoundaryProducedPublication(
        queue=client.PoolQueue(tmp_path / "queue"),
        template={"slots": ["boundary"]},
        instance={"owner_action_key": "owner", "owner_attempt": {"nonce": "n"}},
        tier="stage", cas_root=str(tmp_path / "cas"))
    # Once bound, retirement and observation must keep using that generation.
    # A fresh SDK lookup (even through PQ's normal loader) is not permitted.
    def reject_sdk_reload():
        pytest.fail("egress observation must reuse the adapter's selected SDK")

    monkeypatch.setattr(adapter, "_produced_output_module", reject_sdk_reload)
    # Topology discovery is outside this handshake; keep the adapter's
    # actual retire method and its already-bound queue/client untouched.
    publication.fragment_root = lambda: tmp_path / "fragments"
    publication.stage_root = lambda: str(tmp_path / "stage")
    owner = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(tmp_path / "exact"),
        "max_resident_bytes": 1 << 20, "max_auxiliary_bytes": 1 << 20,
        "max_artifact_bytes": 1 << 20, "prefetch_batches": 1,
    })
    owner._produced = publication
    owner._produced_plan = {"staging_timeout_s": 30.0, "ahead_groups": 0}
    key = ("boundary", 0, -1, 0)
    group = {"batch_id": "old", "published": {"ok": True},
             "retired": False, "context": {"ownership-proof": "retained"}}
    owner._produced_groups[key] = group
    owner._produced_held.add(key)
    owner._produced_ahead.add(key)
    owner._reclaim_produced_origin_for_group = lambda *args: None
    owner._reclaim_produced_origins = lambda: None
    owner._produced_log = lambda *args: None
    owner._write_status = lambda: None
    return SimpleNamespace(clock=clock, client=client, publication=publication,
                           owner=owner, key=key, group=group)


def assert_retained(lane):
    assert not lane.group["retired"]
    assert lane.group["context"] == {"ownership-proof": "retained"}
    assert lane.key in lane.owner._produced_held
    assert lane.key in lane.owner._produced_ahead
    assert not lane.client.retired


def release(lane, **kwargs):
    with lane.owner._produced_lock.held():
        return lane.owner._release_one_produced_group(lane.key, lane.group, **kwargs)


def drain(lane):
    with lane.owner._produced_lock.held():
        lane.owner._drain_produced_releases()


def test_waited_release_observes_terminal_then_consumes_receipt(lane):
    c = lane.client
    c.observe = lambda: ending("done", 1.0) if lane.clock.now >= 1.5 else ending()
    c.reply = lambda: ({"ok": True} if lane.clock.now >= 1.5 else deferred())
    lane.clock.on_sleep = lambda: assert_retained(lane)
    assert release(lane) is True
    assert len(c.retire_calls) == 2, "live egress must not re-drive retirement"
    assert 3 <= len(c.terminal_reads) <= 10, (
        "observe terminals at bounded owner pacing, not in a busy-poll service")
    assert max(c.terminal_reads) >= 1.5
    assert all(t >= 1.5 for t, _ in c.retire_calls[1:])
    assert c.queue_roots and set(c.queue_roots) == {lane.publication.queue.root}
    assert lane.group["retired"] and lane.group["context"] is None
    assert lane.key not in lane.owner._produced_held


@pytest.mark.parametrize("unavailable", [
    "missing", "unreadable", "unreadable-done", "exception", "ambiguous",
])
def test_unavailable_terminal_times_out_without_retire_mutations(lane, unavailable):
    lane.client.observe = lambda: {
        "missing": ending(), "unreadable": ending(unreadable=True),
        "unreadable-done": ending("done", 1.0, unreadable=True),
        "exception": OSError("terminal read unavailable"),
        "ambiguous": ending("done", 1.0, ambiguous=True),
    }[unavailable]
    # Caller has already spent most of its staging budget.
    deadline = lane.clock.now + 1.25
    with pytest.raises(adapter.BoundaryProducedReleaseDeferred):
        release(lane, deadline=deadline)
    assert lane.clock.now == pytest.approx(deadline), "do not mint another budget"
    assert len(lane.client.retire_calls) == 1
    assert lane.client.terminal_reads, "use public terminal observations while waiting"
    assert_retained(lane)
    assert lane.key in lane.owner._produced_release_abandoned


def test_nonblocking_pending_drain_observes_without_redriving(lane):
    lane.owner._produced_plan["ahead_groups"] = 1
    assert release(lane, wait=False) is False
    for _ in range(3):
        lane.clock.now += lane.owner.PRODUCED_RELEASE_POLL_S
        before = lane.clock.now
        drain(lane)
        assert lane.clock.now == before, "live egress must overlap compute, not block it"
        assert_retained(lane)
    assert len(lane.client.retire_calls) == 1
    assert len(lane.client.terminal_reads) >= 3
    lane.client.observe = lambda: ending("done", 1.0)
    lane.client.reply = lambda: {"ok": True}
    lane.clock.now += lane.owner.PRODUCED_RELEASE_POLL_S
    drain(lane)
    assert len(lane.client.retire_calls) == 2
    assert lane.group["retired"]


@pytest.mark.parametrize("terminal", ["failed", "withdrawn"])
def test_terminal_failure_is_revalidated_without_refunding(lane, terminal):
    lane.owner._produced_plan["ahead_groups"] = 1
    assert release(lane, wait=False) is False
    lane.client.observe = lambda: ending(terminal, 1.0)
    refusal = {"ok": False, "refusal": "egress-withdrawn" if terminal == "withdrawn"
               else "egress-incomplete", "receipt": {
                   "complete": False, "errors": ["egress did not complete"],
                   "deferred_own": []}}
    lane.client.reply = lambda: refusal
    lane.clock.now += lane.owner.PRODUCED_RELEASE_POLL_S
    drain(lane)
    assert len(lane.client.retire_calls) == 2
    assert lane.client.terminal_reads
    assert_retained(lane)
    assert lane.owner.telemetry["produced_group_release_failures"] == 1


@pytest.mark.parametrize("terminal", ["done", "failed"])
def test_terminal_is_only_a_wakeup_and_same_generation_is_not_reused(lane, terminal):
    lane.owner._produced_plan["ahead_groups"] = 1
    assert release(lane, wait=False) is False
    # A terminal can still carry an incomplete receipt. PB re-publishes the
    # same content-addressed key. Old evidence cannot wake the successor forever.
    lane.client.observe = lambda: ending(terminal, 1.0)
    lane.clock.now += lane.owner.PRODUCED_RELEASE_POLL_S
    drain(lane)
    assert_retained(lane)
    assert len(lane.client.retire_calls) == 2
    for _ in range(3):
        lane.clock.now += lane.owner.PRODUCED_RELEASE_POLL_S
        drain(lane)
        assert_retained(lane)
    assert len(lane.client.retire_calls) == 2, "old terminal cannot cover republished egress"
    lane.client.observe = lambda: ending("done", 2.0)
    lane.client.reply = lambda: {"ok": True}
    lane.clock.now += lane.owner.PRODUCED_RELEASE_POLL_S
    drain(lane)
    assert len(lane.client.retire_calls) == 3
    assert len(lane.client.terminal_reads) >= 5
    assert lane.group["retired"]


def test_successor_egress_does_not_restart_absolute_release_deadline(lane):
    # Each terminal wakes retirement, which returns a keyed deferral again.
    # Progress across generations is not permission to extend the owner's budget.
    lane.client.observe = lambda: (
        ending("done", float(int(lane.clock.now / 0.5)))
        if lane.clock.now >= 0.5 else ending())
    lane.clock.on_sleep = lambda: assert_retained(lane)
    with pytest.raises(adapter.BoundaryProducedReleaseDeferred):
        release(lane, deadline=1.25)
    assert lane.clock.now == pytest.approx(1.25)
    assert lane.client.retire_calls == [(0.0, "old"), (0.5, "old"), (1.0, "old")]
    assert lane.client.terminal_reads, "generations must be observed, not blindly retried"
    assert_retained(lane)


def test_terminal_read_recovers_without_republishing_or_losing_ownership(lane):
    lane.owner._produced_plan["ahead_groups"] = 1
    assert release(lane, wait=False) is False
    for observation in (ending(), OSError("transient read failure")):
        lane.client.observe = lambda observation=observation: observation
        lane.clock.now += lane.owner.PRODUCED_RELEASE_POLL_S
        drain(lane)
        assert len(lane.client.retire_calls) == 1
        assert_retained(lane)
    lane.client.observe = lambda: ending("done", 3.0)
    lane.client.reply = lambda: {"ok": True}
    lane.clock.now += lane.owner.PRODUCED_RELEASE_POLL_S
    drain(lane)
    assert len(lane.client.retire_calls) == 2
    assert len(lane.client.terminal_reads) >= 3
    assert lane.group["retired"]


def test_keyless_copy_can_transition_to_keyed_egress_without_invented_wait_key(lane):
    lane.client.reply = lambda: deferred(None, "own-copy-in-flight")
    assert release(lane, wait=False) is False
    assert not lane.client.terminal_reads
    lane.client.reply = lambda: deferred()
    assert release(lane, wait=False) is False
    assert len(lane.client.retire_calls) == 2
    assert release(lane, wait=False) is False
    assert len(lane.client.retire_calls) == 2, "only keyed phase uses terminal gate"
    assert lane.client.terminal_reads
    assert_retained(lane)


@pytest.mark.parametrize("kind", ["foreign-pin", "promotion-handoff"])
def test_pin_or_handoff_does_not_take_egress_terminal_gate(lane, kind):
    # A stale key in an unrelated typed receipt is not an own-egress wait.
    receipt = {"complete": False, "errors": [], "deferred_own": [],
               "live_pins": ["foreign"] if kind == "foreign-pin" else [],
               "deferred_handoffs": ["handoff"] if kind == "promotion-handoff" else []}
    lane.client.reply = lambda: {"ok": False, "refusal": "egress-incomplete",
                                 "egress_action_key": KEY, "receipt": receipt}
    assert release(lane, wait=False) is False
    assert not lane.client.terminal_reads
    assert lane.owner._produced_release_pending[lane.key]["class"] == kind
    # Establish the real keyed phase too: this acceptance test must be RED
    # on main, not merely re-state already-green typed-pin behavior.
    lane.client.reply = lambda: deferred()
    assert release(lane, wait=False) is False
    assert release(lane, wait=False) is False
    assert len(lane.client.retire_calls) == 2
    assert lane.client.terminal_reads
    assert_retained(lane)


def test_refill_funding_uses_same_gate_and_absolute_deadline(lane):
    c = lane.client
    lane.owner._produced_plan["ahead_groups"] = 1
    c.observe = lambda: ending("done", 1.0) if lane.clock.now >= 1.5 else ending()
    c.reply = lambda: {"ok": True} if lane.clock.now >= 1.5 else deferred()
    lane.clock.on_sleep = lambda: assert_retained(lane)
    with lane.owner._produced_lock.held():
        out = lane.owner._produced_fund_for_read(
            lambda: refill_step(lane), (), deadline=3.0)
    assert out["ok"]
    assert len(c.retire_calls) == 2, "funding cannot bypass the release terminal gate"
    assert 3 <= len(c.terminal_reads) <= 10, (
        "observe terminals at bounded owner pacing, not in a busy-poll service")
    assert max(c.terminal_reads) >= 1.5
    assert max(t for t, _ in c.retire_calls) < 3.0
    assert c.refill_calls and lane.group["retired"]


def refill_step(lane):
    out = lane.publication.refill()
    if not out.get("ok"):
        raise adapter.BoundaryProducedPublicationFailed(
            batch_id="next", refusal={**out, "step": "refill"})
    return out


def test_refill_timeout_keeps_credits_and_never_retires_at_deadline(lane):
    lane.owner._produced_plan["ahead_groups"] = 1
    with lane.owner._produced_lock.held():
        with pytest.raises(adapter.BoundaryProducedPublicationFailed):
            lane.owner._produced_fund_for_read(
                lambda: refill_step(lane), (), deadline=1.25)
    assert lane.clock.now == pytest.approx(1.25)
    assert len(lane.client.retire_calls) == 1
    assert lane.client.terminal_reads
    assert_retained(lane)
