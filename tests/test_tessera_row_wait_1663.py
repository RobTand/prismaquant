"""CPU contracts for scoped Tessera row-stream wait observations."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from prismaquant import io_spans, tessera_row_stream
from test_stage_a_source_wait_1663 import _Delivery


def _consumer(delivery, sink):
    stream = tessera_row_stream.RowStream.__new__(tessera_row_stream.RowStream)
    stream._live = {}
    stream._inflight = {"unit": delivery}
    stream._first = {}
    stream._expected = None
    stream._weights = {}
    stream.wait_sink = sink
    stream.stats = dict(entries_read=0, read_seconds=0.0, rereads=0)
    return stream


@pytest.mark.parametrize("ready", [False, True])
def test_row_stream_observes_only_pending_delivery(monkeypatch, ready):
    entry = SimpleNamespace(read_seconds=1.0, weight=None, load={}, identities={},
                            count=1, max_abs=1.0)
    delivery = _Delivery(entry, ready=ready)
    ledger = io_spans.ExposedWaitLedger()
    stream = _consumer(delivery, ledger.sink)
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(tessera_row_stream.time, "time", lambda: next(clock))
    assert stream._collect("unit") is entry
    assert delivery.calls == 1
    snapshot = ledger.snapshot()
    if ready:
        assert snapshot == {"intervals": [], "takes": []}
    else:
        assert snapshot["intervals"] == [
            dict(kind="row-stream-load", start_unix=10.0, end_unix=12.0)]
        assert snapshot["takes"][0]["unit"] == "unit"
        assert snapshot["takes"][0]["wait_s"] == 2.0
        assert "load_bytes_per_s" not in snapshot["takes"][0]


def test_row_stream_observes_failed_delivery_without_admitting_it(monkeypatch):
    failure = RuntimeError("verified entry refused")
    ledger = io_spans.ExposedWaitLedger()
    stream = _consumer(_Delivery(None, error=failure), ledger.sink)
    clock = iter([20.0, 23.0])
    monkeypatch.setattr(tessera_row_stream.time, "time", lambda: next(clock))
    with pytest.raises(RuntimeError) as refused:
        stream._collect("unit")
    assert refused.value is failure
    assert stream._live == stream._first == {}
    assert ledger.snapshot()["takes"][0]["wait_s"] == 3.0


@pytest.mark.parametrize("refused", [False, True])
def test_campaign_publishes_scoped_waits_on_success_and_failure(tmp_path, monkeypatch, refused):
    from test_tessera_row_stream import stream_fixture
    campaign, argv, _state = stream_fixture(monkeypatch, tmp_path)
    original = campaign._measure_anchor

    def measure(**kwargs):
        if refused:
            raise RuntimeError("fixture encode refused")
        return original(**kwargs)

    monkeypatch.setattr(campaign, "_measure_anchor", measure)
    if refused:
        with pytest.raises(RuntimeError, match="campaign row failed"):
            campaign.main(argv)
    else:
        assert campaign.main(argv) == 0
    report = json.loads((tmp_path / "cost.waits.json").read_text())
    assert report["schema"] == "prismaquant.tessera_campaign_waits.v1"
    assert report["passed"] is (not refused)
    assert report["row_head"] == "stream"
    assert report["row_stream_exposed_wait"]["coverage"] == "row-stream-load-only"
    assert report["row_stream_exposed_wait"]["power_samples"] == 0
    assert report["row_stream_exposed_wait"]["baseline"] is None
    assert report["exposed_wait"]["instrumented"] is False
    assert "publication" in report["exposed_wait"]["reason"]


def test_failed_observer_retains_successful_entry_for_close(monkeypatch):
    entry = SimpleNamespace(read_seconds=1.0, weight=None, load={}, identities={},
                            count=1, max_abs=1.0)
    delivery = _Delivery(entry)

    def refuse(*_args):
        raise ValueError("observation refused")

    stream = _consumer(delivery, refuse)
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(tessera_row_stream.time, "time", lambda: next(clock))
    with pytest.raises(RuntimeError, match="load wait observer failed"):
        stream._collect("unit")
    assert stream._inflight["unit"] is delivery
    assert stream._live == stream._first == {}


def test_measured_idle_baseline_and_wait_bands_are_published(tmp_path, monkeypatch):
    class Power:
        interval_s = 1.0
        times = [10.0, 11.0, 21.5, 22.5]
        samples = [12.0, 13.0, 12.0, 80.0]

        def start(self):
            return self

        def stop(self):
            return {"sample_count": len(self.samples)}

    monkeypatch.setattr(io_spans, "GpuPowerSampler", Power)
    clock = iter([9.0, 20.0, 24.0])
    monkeypatch.setattr(tessera_row_stream, "time", SimpleNamespace(time=lambda: next(clock)))
    with tessera_row_stream.RowWaitTelemetry() as observed:
        observed.start(tmp_path / "cost.pkl", "cuda")
        observed.before_gpu_work()
        observed.ledger.add("row-stream-load", 21.0, 23.0)
        observed.returncode = 0
    report = json.loads((tmp_path / "cost.waits.json").read_text())
    block = report["row_stream_exposed_wait"]
    assert block["idle_ceiling_w"] == 13.0
    assert block["total"]["wait_s"] == 2.0
    assert block["total"]["idle_band_s"] == 1.0
    assert block["total"]["busy_band_s"] == 1.0
    assert block["total"]["unsampled_s"] == 0.0


def test_telemetry_write_failure_does_not_mask_workload_failure(tmp_path):
    observed = tessera_row_stream.RowWaitTelemetry()
    observed.start(tmp_path / "cost.pkl", "cpu")
    observed.path.mkdir()
    failure = RuntimeError("primary entry failure")
    observed.__exit__(type(failure), failure, None)
    assert "campaign wait telemetry failed" in failure.__notes__[0]
