"""Fail fast on a terminal produced mover (PQ #924)."""
from __future__ import annotations
import time
import pytest
import test_stage_a_produced_boundary_chain as chain
from test_stage_a_produced_boundary_chain import (  # noqa: F401
    _isolated_launch_context)
#: How many times a stub publication may be asked before the test gives
#: up. These tests demonstrate that a terminal mover fails fast, so on a
#: tree that still waits out the budget they have to FAIL, in bounded
#: time, rather than poll a stopped clock forever (PQ #961): an unbounded
#: poll loop here OOM-killed two PrismaBuild shards on the growing call
#: list before the failure was narrowed by hand.
POLL_BOUND = 16
def _stub_publication(states: list, calls: list, *, max_polls: int = POLL_BOUND):
    from prismaquant.stage_a_produced_output import (
        BoundaryProducedPublication)
    publication = BoundaryProducedPublication.__new__(
        BoundaryProducedPublication)
    def _state(*, batch_id: str):
        calls.append(batch_id)
        if len(calls) > max_polls:
            raise AssertionError(
                f"await_materialized asked for {batch_id!r} {len(calls)} "
                f"times without failing fast on a terminal mover; the stub "
                f"bound is {max_polls} polls (PQ #961)")
        return dict(states[min(len(calls) - 1, len(states) - 1)])
    publication.materialization_state = _state  # type: ignore[method-assign]
    return publication
def _stub_clock(monkeypatch, sleeps: list, *, start: float = 0.0):
    """A monotonic clock that only the poll loop's own sleep advances.

    Sleeping moves it, so a wait that keeps polling reaches its deadline
    instead of running forever against a frozen reading, and the list of
    sleeps still records exactly what the code under test asked for.
    """
    clock = {"now": float(start)}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    def _sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += max(float(seconds), 0.001)
    monkeypatch.setattr(time, "sleep", _sleep)
    return clock
def _terminal_state(*, queue_state: str, complete=None, refusal=None):
    return {"ok": True, "batch_id": "b-0", "mover_key": "a" * 64,
            "generation": 0, "mover_receipt_complete": complete,
            "mover_refusal": refusal, "mover_queue_state": queue_state}
def test_failed_terminal_fails_fast_without_waiting(monkeypatch):
    from prismaquant.stage_a_produced_output import BoundaryStagingTimeout
    calls: list = []
    sleeps: list = []
    publication = _stub_publication(
        [_terminal_state(queue_state="failed")], calls)
    _stub_clock(monkeypatch, sleeps, start=100.0)
    with pytest.raises(BoundaryStagingTimeout) as caught:
        publication.await_materialized(batch_id="b-0", timeout_s=900.0)
    assert calls == ["b-0"], calls
    assert sleeps == [], sleeps
    text = str(caught.value)
    assert "will not stage" in text, text
    assert "b-0" in text and "failed" in text, text
    assert "generation" in text, text
def test_withdrawn_terminal_carries_refusal_and_generation(monkeypatch):
    from prismaquant.stage_a_produced_output import BoundaryStagingTimeout
    calls: list = []
    sleeps: list = []
    publication = _stub_publication(
        [_terminal_state(queue_state="withdrawn", complete=False,
                         refusal="residency_moved_nothing")], calls)
    _stub_clock(monkeypatch, sleeps, start=50.0)
    with pytest.raises(BoundaryStagingTimeout) as caught:
        publication.await_materialized(batch_id="b-1", timeout_s=900.0)
    assert calls == ["b-1"], calls
    assert sleeps == [], "a terminal mover is not waited on"
    text = str(caught.value)
    assert "will not stage" in text and "withdrawn" in text, text
    assert "residency_moved_nothing" in text, text
def test_complete_true_wins_over_terminal_queue_state(monkeypatch):
    calls: list = []
    publication = _stub_publication(
        [_terminal_state(queue_state="failed", complete=True)], calls)
    _stub_clock(monkeypatch, [])
    out = publication.await_materialized(batch_id="b-0", timeout_s=900.0)
    assert out["mover_receipt_complete"] is True
    assert calls == ["b-0"], calls
def test_ok_false_still_raises_binding_not_timeout(monkeypatch):
    from prismaquant.stage_a_produced_output import (
        BoundaryProducedBindingError)
    state = {"ok": False, "refusal": "unknown-batch",
             "mover_queue_state": "failed",
             "mover_receipt_complete": None}
    publication = _stub_publication([state], [])
    _stub_clock(monkeypatch, [])
    with pytest.raises(BoundaryProducedBindingError) as caught:
        publication.await_materialized(batch_id="b-missing", timeout_s=10.0)
    assert "unknown-batch" in str(caught.value)
def test_ready_claimed_absent_unknown_done_keep_bounded_wait(monkeypatch):
    from prismaquant.stage_a_produced_output import BoundaryStagingTimeout
    for queue_state in ("ready", "claimed", "absent", "unknown", "done"):
        calls: list = []
        sleeps: list = []
        _stub_clock(monkeypatch, sleeps)
        publication = _stub_publication(
            [_terminal_state(queue_state=queue_state)], calls)
        with pytest.raises(BoundaryStagingTimeout) as caught:
            publication.await_materialized(batch_id="b-wait", timeout_s=0.5,
                                           poll_s=0.1)
        text = str(caught.value)
        assert "was not staged within" in text, (queue_state, text)
        assert "will not stage" not in text, (queue_state, text)
        assert len(calls) >= 2, (queue_state, calls)
        assert sleeps, (queue_state, sleeps)
def _publish_one_group(storage):
    for key, group in list(storage._produced_groups.items()):
        if group["published"] is None:
            storage._produced_publish(key, group)
            return group
    raise AssertionError("no unpublished group to publish")
def test_real_spent_ready_still_waits(tmp_path):
    # R5 shape on the current runtime: one failed claim consumes funding
    # and the mover returns READY unfundable (PB848 owns the terminal
    # disposition). This lane must NOT infer failure from READY: it keeps
    # the bounded wait. Failing fast here would claim PB848's fix.
    from prismaquant.stage_a_produced_output import BoundaryStagingTimeout
    storage, publication, q, _env, _pb = chain._bound_owner(tmp_path)
    chain._write_group(storage)
    group = _publish_one_group(storage)
    mover = str(group["published"]["mover_key"])
    batch_id = str(group["batch_id"])
    claimed = q.claim(owner="w-failfast-spent",
                      tags=[chain._tier_host(q)])
    assert claimed is not None and claimed["action_key"] == mover, claimed
    q.finish(mover, status="failed")
    state = publication.materialization_state(batch_id=batch_id)
    assert state.get("ok") is True, state
    assert state.get("mover_queue_state") == "ready", state
    assert state.get("mover_receipt_complete") is not True, state
    with pytest.raises(BoundaryStagingTimeout) as caught:
        publication.await_materialized(batch_id=batch_id, timeout_s=1.0,
                                       poll_s=0.1)
    text = str(caught.value)
    assert "was not staged within" in text, text
    assert "will not stage" not in text, text
def test_real_withdrawn_mover_fails_fast(tmp_path):
    from prismaquant.stage_a_produced_output import BoundaryStagingTimeout
    storage, publication, q, _env, _pb = chain._bound_owner(tmp_path)
    chain._write_group(storage)
    group = _publish_one_group(storage)
    mover = str(group["published"]["mover_key"])
    batch_id = str(group["batch_id"])
    out = q.withdraw(mover, reason="failfast-test", by="test")
    assert out.get("status") in ("withdrawn", "withdrawn_from_ready",
                                 "withdrawn_from_claimed", "ok",
                                 "cancelled"), out
    state = publication.materialization_state(batch_id=batch_id)
    assert state.get("ok") is True, state
    assert state.get("mover_queue_state") == "withdrawn", state
    assert state.get("mover_receipt_complete") is not True, state
    started = time.monotonic()
    with pytest.raises(BoundaryStagingTimeout) as caught:
        publication.await_materialized(batch_id=batch_id, timeout_s=900.0)
    assert time.monotonic() - started < 30.0
    text = str(caught.value)
    assert "will not stage" in text and "withdrawn" in text, text
def test_real_ready_mover_keeps_bounded_wait(tmp_path):
    from prismaquant.stage_a_produced_output import BoundaryStagingTimeout
    storage, publication, _q, _env, _pb = chain._bound_owner(tmp_path)
    chain._write_group(storage)
    group = _publish_one_group(storage)
    batch_id = str(group["batch_id"])
    state = publication.materialization_state(batch_id=batch_id)
    assert state.get("mover_queue_state") == "ready", state
    started = time.monotonic()
    with pytest.raises(BoundaryStagingTimeout) as caught:
        publication.await_materialized(batch_id=batch_id, timeout_s=1.0,
                                       poll_s=0.1)
    elapsed = time.monotonic() - started
    text = str(caught.value)
    assert "was not staged within" in text, text
    assert "will not stage" not in text, text
    assert 0.5 <= elapsed < 30.0, elapsed
def test_real_successful_receipt_returns(tmp_path):
    storage, publication, q, _env, _pb = chain._bound_owner(tmp_path)
    chain._write_group(storage)
    group = _publish_one_group(storage)
    mover = str(group["published"]["mover_key"])
    batch_id = str(group["batch_id"])
    chain._execute_mover(q, mover)
    out = publication.await_materialized(batch_id=batch_id, timeout_s=30.0)
    assert out.get("mover_receipt_complete") is True, out
    assert out.get("batch_id") == batch_id, out
