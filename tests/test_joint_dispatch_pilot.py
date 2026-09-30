"""Pilot admission tests mutate published QuantumCounters receipts.

Power/rate inputs are controlled CPU doubles, not GPU performance evidence.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from test_dispatch_joint_quanta import (
    _receipt, _stamp_receipt, _write_receipt, campaign, records_dir,
)
from tools.dispatch_joint_quanta import FakeGateway, main


@pytest.fixture(autouse=True)
def _pilot_spec(tmp_path, monkeypatch):
    from stage_a_spool_spec import with_spool
    from tools import dispatch_joint_quanta as dispatch

    spec = tmp_path / "pilot-spec.json"
    spec.write_text(json.dumps(with_spool(
        {"container": {"image": "sha256:" + "0" * 64}, "env": {}})))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)


def _argv(records_dir, output_root, receipt, extra=()):
    # No implicit override: unlike legacy dispatcher behavior tests, these
    # calls exercise the production pilot gate.
    return ["--records", str(records_dir), "--output-root", str(output_root),
            "--adjoint-receipt", str(receipt), *extra]


def _ready(tmp_path, campaign, records_dir):
    receipt = tmp_path / "receipt.json"
    _write_receipt(receipt, _receipt(campaign))
    _stamp_receipt(records_dir, receipt)
    return receipt


@pytest.mark.parametrize("dry_run", [False, True])
def test_fanout_without_a_measured_pilot_refuses_before_submission(
        tmp_path, campaign, records_dir, capsys, dry_run):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _argv(records_dir, tmp_path / "out", receipt,
                 extra=["--dry-run"] if dry_run else [])
    assert main(args, _gateway=gateway) == 3
    assert not gateway.submitted
    assert "pilot" in capsys.readouterr().err


def test_one_quantum_can_produce_the_pilot(tmp_path, campaign, records_dir):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    assert main(_argv(records_dir, tmp_path / "out", receipt,
                      extra=["--quantum", "layer-000"]),
                _gateway=gateway) == 0
    assert len(gateway.submitted) == 1
    state = tmp_path / "out" / "layer-quanta" / "campaign-state.json"
    events = [json.loads(line) for line in state.read_text().splitlines()]
    assert events[-1]["pilot_admission"]["mode"] == "pilot"


class _Power:
    interval_s = 1.0

    def __init__(self, now):
        self.times = [now - 30 + i for i in range(31)]
        self.samples = [12.0] * 10 + [70.0] * 21

    def stop(self):
        return {"sample_count": len(self.samples), "gpu_joules": 100.0,
                "gpu_power_w_p50": 70.0, "gpu_power_w_p95": 70.0,
                "gpu_power_w_max": 70.0}


def _published_pilots(tmp_path, records_dir, monkeypatch, mutate=None, *,
                      disabled_kernel_profile=False):
    """Use the row's counters and publisher, then mutate their output bytes."""
    from tools import dispatch_joint_quanta as dispatch
    from prismaquant import joint_dispatch_pilot as pilot
    from prismaquant.joint_cost_quantum import (
        ChunkFrontier, QuantumCounters, publish_quantum_outputs,
    )

    monkeypatch.setattr(dispatch, "_pilot_code_sha256", lambda: "f" * 64)
    monkeypatch.setenv("PRISMABUILD_ACTION_KEY", "e" * 64)
    args = []
    for record_path in sorted(records_dir.glob("layer-*.json")):
        record = json.loads(record_path.read_text())
        now = time.time()
        windows = [{**w, "render_file_upper_bound_bytes": 300}
                   for w in record["windows"]]
        frontier = ChunkFrontier(chunks=record["chunks"], windows=windows)
        counters = QuantumCounters(
            quantum_id=record["quantum_id"], identity_sha256=record["identity_sha256"],
            chunks=record["chunks"], frontier=frontier,
            sampler=_Power(now), started=now - 30,
            pilot_binding=pilot.pilot_binding(
                record, implementation_sha256="f" * 64,
                execution_plan_sha256=record["campaign"]["plan_sha256"],
                replay_regime=None, cotangent_source="chain", emits_handoff=False))
        if disabled_kernel_profile:
            from prismaquant.joint_adjoint_checkpoints import KernelTimeProfiler
            from prismaquant.joint_cost_quantum import KERNEL_PROFILE_NOT_MEASURED
            counters.kernel_block(KernelTimeProfiler(
                not_measured=KERNEL_PROFILE_NOT_MEASURED))
        counters.gpu_work_started_unix = now - 20
        counters.phases[0]["entered_unix"] = now - 15
        counters.exposed_wait.take("window-load", now - 8, now - 7, {
            "bytes": 300, "work_before_s": 1.0,
            "load_bytes_per_s": 100.0, "first_fill": False})
        block = counters.finish(units_done=1, units_total=1)
        root = tmp_path / f"pilot-{record['layer']}"
        root.mkdir()
        outputs = {"root": str(root), "cost_payload": str(root / "cost.pkl"),
                   "counters": str(root / "counters.json"),
                   "results": str(root / "results.json")}
        publish_quantum_outputs(
            {**record, "output_space": outputs}, payload={"costs": {"unit": {}}},
            result={}, counters=block, units_total=1)
        path = root / "counters.json"
        if mutate is not None:
            document = json.loads(path.read_text())
            mutate(document)
            path.write_text(json.dumps(document))
        args += ["--pilot-counters", str(path), hashlib.sha256(path.read_bytes()).hexdigest()]
    return args


def test_matching_producer_pilots_admit_and_stamp_the_fanout(
        tmp_path, campaign, records_dir, monkeypatch):
    receipt = _ready(tmp_path, campaign, records_dir)
    pilot_args = _published_pilots(tmp_path, records_dir, monkeypatch)
    gateway = FakeGateway()
    gateway.mark_terminal("e" * 64)
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=pilot_args),
                _gateway=gateway) == 0
    assert len(gateway.submitted) == 3
    state = tmp_path / "out" / "layer-quanta" / "campaign-state.json"
    events = [json.loads(line) for line in state.read_text().splitlines()]
    for event in events:
        admission = event["pilot_admission"]
        assert admission["mode"] == "verified"
        assert admission["gpu_envelope_fraction"] == 0.5
        assert admission["action_key"] == "e" * 64
        assert len(admission["document"]["sha256"]) == 64


def test_optional_kernel_profile_does_not_invalidate_measured_wait(
        tmp_path, campaign, records_dir, monkeypatch):
    receipt = _ready(tmp_path, campaign, records_dir)
    args = _published_pilots(tmp_path, records_dir, monkeypatch,
                             disabled_kernel_profile=True)
    gateway = FakeGateway()
    gateway.mark_terminal("e" * 64)
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 0
    assert len(gateway.submitted) == 3


@pytest.mark.parametrize("field,value", [
    ("implementation_sha256", "0" * 64),
    ("replay_regime", {"capture_batch": 2, "accumulation": "per_invocation", "chunk_rows": None}),
    ("row_shape_sha256", "0" * 64),
])
def test_stale_binding_refuses_without_publishing(
        tmp_path, campaign, records_dir, monkeypatch, field, value):
    receipt = _ready(tmp_path, campaign, records_dir)
    args = _published_pilots(tmp_path, records_dir, monkeypatch,
                             mutate=lambda c: c["pilot"]["binding"].update({field: value}))
    gateway = FakeGateway()
    gateway.mark_terminal("e" * 64)
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 3
    assert not gateway.submitted


def test_before_regime_wait_refuses_with_phase_excess_and_rates(
        tmp_path, campaign, records_dir, monkeypatch, capsys):
    receipt = _ready(tmp_path, campaign, records_dir)

    def before(counters):
        take = counters["exposed_wait"]["bound"]["per_take"][0]
        take["wait_s"] = 6.0
        take["end_unix"] = take["start_unix"] + 6.0

    args = _published_pilots(tmp_path, records_dir, monkeypatch, mutate=before)
    gateway = FakeGateway()
    gateway.mark_terminal("e" * 64)
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 3
    assert not gateway.submitted
    error = capsys.readouterr().err
    assert "layer-002-chunk-000" in error
    assert "excess" in error and "4" in error
    assert "load_bytes_per_s" in error and "100" in error


@pytest.mark.parametrize("defect", ["failed", "incomplete", "power", "rates", "unsampled", "unbounded-spill"])
def test_incomplete_measurement_never_certifies_a_pilot(
        tmp_path, campaign, records_dir, monkeypatch, defect):
    receipt = _ready(tmp_path, campaign, records_dir)

    def break_receipt(c):
        if defect == "failed":
            c["outcome"]["status"] = "gapped"
        elif defect == "incomplete":
            c["units"] = [None, 1]
        elif defect == "power":
            c["gpu_sampler_samples"] = 0
        elif defect == "rates":
            c["exposed_wait"]["bound"]["per_take"][0]["load_bytes_per_s"] = None
        elif defect == "unsampled":
            c["exposed_wait"]["total"].update(unsampled_s=1.0, busy_band_s=0.0)
        else:
            total = c["exposed_wait"]["total"]
            total["by_kind"]["spill-reader"] = {"wait_s": 1.0, "count": 1}
            total.update(wait_s=2.0, busy_band_s=2.0)

    args = _published_pilots(tmp_path, records_dir, monkeypatch, mutate=break_receipt)
    gateway = FakeGateway()
    gateway.mark_terminal("e" * 64)
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 3
    assert not gateway.submitted


def test_a_nonterminal_pb_pilot_refuses(tmp_path, campaign, records_dir, monkeypatch):
    receipt = _ready(tmp_path, campaign, records_dir)
    args = _published_pilots(tmp_path, records_dir, monkeypatch)
    gateway = FakeGateway()
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 3
    assert not gateway.submitted


def test_override_is_explicit_and_persistent(tmp_path, campaign, records_dir):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    assert main(_argv(records_dir, tmp_path / "out", receipt,
                      extra=["--force-unverified-pilot"]), _gateway=gateway) == 0
    state = tmp_path / "out" / "layer-quanta" / "campaign-state.json"
    events = [json.loads(line) for line in state.read_text().splitlines()]
    assert len(events) == 3
    assert all(event["pilot_admission"]["mode"] == "override" for event in events)
