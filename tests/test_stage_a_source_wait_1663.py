"""CPU contract coverage for Stage A source waits, not GPU qualification."""
from __future__ import annotations

import json

import pytest

from prismaquant import io_spans, streaming_model
from test_streamed_prefetch_scheduling import _make_ctx


class _Delivery:
    def __init__(self, value, *, ready=False, error=None):
        self.value, self.ready, self.error = value, ready, error
        self.calls = 0

    def done(self):
        return self.ready

    def result(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.value


@pytest.mark.parametrize("ready", [False, True])
def test_source_observer_times_pending_delivery_not_ready_ownership(monkeypatch, ready):
    context = _make_ctx(monkeypatch, workers=1)
    ledger = io_spans.ExposedWaitLedger()
    delivery = _Delivery({"weight": object()}, ready=ready)
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(streaming_model.time, "time", lambda: next(clock))
    try:
        with context.observe_source_waits(ledger.sink):
            loaded = context._await_prefetch(3, delivery, retry_availability=False)
        assert loaded is delivery.value
        assert delivery.calls == 1
        snapshot = ledger.snapshot()
        if ready:
            assert snapshot == {"intervals": [], "takes": []}
        else:
            assert snapshot["intervals"] == [
                {"kind": "source-prefetch", "start_unix": 10.0, "end_unix": 12.0}]
            assert snapshot["takes"][0]["layer"] == 3
            assert snapshot["takes"][0]["wait_s"] == 2.0
            assert "load_bytes_per_s" not in snapshot["takes"][0]
        assert getattr(context, "_source_wait_sink", None) is None
    finally:
        context.prefetch_pool.shutdown(wait=True)


def test_failed_source_wait_preserves_exception_and_restores_observer(monkeypatch):
    context = _make_ctx(monkeypatch, workers=1)
    ledger = io_spans.ExposedWaitLedger()
    failure = RuntimeError("fixture source failed")
    delivery = _Delivery(None, error=failure)
    clock = iter([20.0, 23.0])
    monkeypatch.setattr(streaming_model.time, "time", lambda: next(clock))
    try:
        with pytest.raises(RuntimeError, match="fixture source failed") as caught:
            with context.observe_source_waits(ledger.sink):
                context._await_prefetch(4, delivery, retry_availability=False)
        assert caught.value is failure
        assert ledger.snapshot()["takes"][0]["wait_s"] == 3.0
        assert getattr(context, "_source_wait_sink", None) is None
    finally:
        context.prefetch_pool.shutdown(wait=True)


def test_source_wait_without_observer_keeps_delivery_semantics(monkeypatch):
    context = _make_ctx(monkeypatch, workers=1)
    delivery = _Delivery({"weight": object()})
    try:
        assert context._await_prefetch(3, delivery, retry_availability=False) is delivery.value
        assert delivery.calls == 1
    finally:
        context.prefetch_pool.shutdown(wait=True)


def test_stage_a_reports_source_component_without_claiming_whole_row_instrumented(
        tmp_path, monkeypatch):
    from test_stage_a_head_skip import ONE, _campaign, _capture
    from prismaquant.joint_adjoint_checkpoints import adjoint_space

    campaign = _campaign(tmp_path, implementation=ONE)
    root = tmp_path / "run"
    result = _capture(campaign, root, monkeypatch)
    assert result["passed"] is True
    counters = json.loads((adjoint_space(root) / "counters.json").read_text())
    source = counters["source_exposed_wait"]
    assert source["schema"] == "prismaquant.exposed_wait.v1"
    assert source["coverage"] == "source-prefetch-only"
    assert source["idle_ceiling_w"] is None
    assert source["baseline"] is None
    assert source["power_samples"] == 0
    assert counters["exposed_wait"]["instrumented"] is False
    assert "boundary/checkpoint" in counters["exposed_wait"]["reason"]
