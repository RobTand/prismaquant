"""A failed Stage B quantum still writes its counters.

``counters.json`` was written only after the layer core returned, so the v4
and v5 gate failures left no per-phase record of their own: the evidence came
from outside samplers. The failure path now writes the counters it holds,
with an ``outcome`` naming the error and the spans it interrupted, before the
runner's teardown. It covers exceptions; a SIGKILL leaves nothing, and the
host sampler remains the instrument for that.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class _Stop(Exception):
    pass


class _Sampler:
    """The power sampler's surface, with no ``nvidia-smi`` child."""

    interval_s = 1.0

    def __init__(self):
        self.samples = []
        self.stopped = 0

    def start(self):
        return self

    def stop(self):
        self.stopped += 1
        return {"sample_count": len(self.samples), "gpu_joules": None}


def _record(tmp_path, **space):
    root = tmp_path / "layer-044"
    return {"layer": 44, "quantum_id": "layer-044", "identity_sha256": "a" * 64,
            "campaign": {}, "adjoint": {"slice_sha256": "b" * 64},
            "output_space": {"root": str(root), **space}}


def _counters(record, sampler):
    from prismaquant.joint_cost_quantum import ChunkFrontier, QuantumCounters

    return QuantumCounters(
        quantum_id=record["quantum_id"], identity_sha256=record["identity_sha256"],
        chunks=[], frontier=ChunkFrontier(chunks=[], windows=[]), sampler=sampler)


def test_failure_counters_close_open_spans_and_name_the_error(tmp_path, monkeypatch):
    import prismaquant.residency_map as residency
    from prismaquant.joint_cost_quantum import write_failure_counters

    monkeypatch.setattr(residency, "residency_report", lambda: None)
    record = _record(tmp_path, counters=str(tmp_path / "layer-044" / "counters.json"))
    sampler = _Sampler()
    counters = _counters(record, sampler)
    with counters.io.span("checkpoint-load"):
        pass
    counters.io.open("window", window=0)
    counters.io.open("replay", window=0, probe=2, mode="spill")
    error = TypeError("check_operator_allocation() missing 1 required "
                      "keyword-only argument: 'reserve_bytes'")
    path = write_failure_counters(record, counters=counters, io_spans=counters.io,
                                  sampler=sampler, error=error, units_total=12)
    assert path == Path(record["output_space"]["counters"])
    document = json.loads(path.read_text())
    assert document["outcome"] == {
        "status": "failed", "error_type": "TypeError", "error": f"TypeError: {error}",
        "open_spans": ["window", "replay"]}
    assert document["units"] == [None, 12]
    assert document["quantum_id"] == "layer-044"
    spans = [(span["span"], span["outcome"]) for span in document["io_spans"]]
    assert spans == [("checkpoint-load", "ok"), ("replay", "interrupted"),
                     ("window", "interrupted")]
    assert sampler.stopped == 1


def test_failure_before_the_counters_still_writes_the_spans(tmp_path):
    from prismaquant.io_spans import IoSpanLog
    from prismaquant.joint_cost_quantum import write_failure_counters

    record = _record(tmp_path)  # no counters path: the root's counters.json
    sampler = _Sampler()
    log = IoSpanLog(scope="layer-044", log=lambda line: None)
    log.open("head")
    path = write_failure_counters(record, counters=None, io_spans=log,
                                  sampler=sampler, error=_Stop("head refused"))
    assert path == Path(record["output_space"]["root"]) / "counters.json"
    document = json.loads(path.read_text())
    assert document["outcome"]["error_type"] == "_Stop"
    assert document["outcome"]["open_spans"] == ["head"]
    assert [(s["span"], s["outcome"]) for s in document["io_spans"]] == [
        ("head", "interrupted")]
    assert document["units"] == [None, 0]
    assert sampler.stopped == 1


def test_failure_counters_never_mask_the_run_error(tmp_path, capsys):
    from prismaquant.io_spans import IoSpanLog
    from prismaquant.joint_cost_quantum import write_failure_counters

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    record = _record(tmp_path, counters=str(blocker / "counters.json"))
    path = write_failure_counters(
        record, counters=None, io_spans=IoSpanLog(scope="q", log=lambda l: None),
        sampler=_Sampler(), error=RuntimeError("the run's own error"))
    assert path is None
    assert "could not write the failure counters" in capsys.readouterr().out


def test_success_counters_carry_their_outcome(tmp_path):
    from prismaquant.joint_cost_quantum import publish_quantum_outputs

    root = tmp_path / "layer-044"
    root.mkdir()
    record = _record(tmp_path, counters=str(root / "counters.json"),
                     results=str(root / "results.json"),
                     cost_payload=str(root / "cost.pkl"))
    publish_quantum_outputs(record, payload=None, result={}, counters={"io_spans": []},
                            units_total=3)
    document = json.loads((root / "counters.json").read_text())
    assert document["outcome"] == {"status": "gapped"}


@pytest.fixture
def failing_head(tmp_path, monkeypatch):
    """``run_layer_quantum`` failing in its no-slice head intake."""
    import prismaquant.gpu_guard as gpu_guard
    import prismaquant.joint_cost_quantum as quantum
    import prismaquant.joint_projection_backend as backend
    import prismaquant.residency_map as residency
    import prismaquant.tessera_joint_aura as aura

    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a, **k: None)
    monkeypatch.setattr(backend, "executing_image", lambda: "fixture-image")
    monkeypatch.setattr(backend, "prewarm_projection_backend",
                        lambda *a, **k: SimpleNamespace(identity={"name": "torch"}))
    monkeypatch.setattr(quantum, "bind_joint_served_quantizer", lambda formats: {})
    monkeypatch.setattr(aura, "_preflight_run_prepared", lambda *a, **k: None)
    monkeypatch.setattr(residency, "residency_report", lambda: None)
    sampler = _Sampler()
    monkeypatch.setattr(quantum, "GpuPowerSampler", lambda: sampler)
    monkeypatch.setenv(aura.ACTIVATION_SCALE_ENV, "0")

    def intake(inputs, **kwargs):
        raise _Stop("head intake refused")

    monkeypatch.setattr(aura, "load_measured_anchor_input", intake)
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"formats_by_qname": {}}))
    record = _record(tmp_path, counters=str(tmp_path / "layer-044" / "counters.json"))
    threads = torch.get_num_threads()
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        yield SimpleNamespace(record=record, prepared=prepared, sampler=sampler,
                              quantum=quantum)
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32


def test_a_quantum_that_fails_in_its_head_writes_its_counters(failing_head, tmp_path):
    run = failing_head
    config = {"model": "fixture", "inputs": {},
              "execution": {"production_act_scales": "0"}}
    with pytest.raises(_Stop, match="head intake refused"):
        run.quantum.run_layer_quantum(
            config, record=run.record, adjoint_slice={}, plan_sha256="c" * 64,
            prepared={"path": str(run.prepared),
                      "sha256": hashlib.sha256(run.prepared.read_bytes()).hexdigest()},
            output_root=tmp_path)
    document = json.loads(Path(run.record["output_space"]["counters"]).read_text())
    assert document["outcome"]["status"] == "failed"
    assert document["outcome"]["error_type"] == "_Stop"
    assert document["outcome"]["open_spans"] == ["head"]
    (head,) = document["io_spans"]
    assert (head["span"], head["outcome"]) == ("head", "interrupted")
    assert head["proc_io"] is not None
    assert run.sampler.stopped == 1
    # Only the counters: the success path owns status.json and cost.pkl.
    written = sorted(p.name for p in Path(run.record["output_space"]["root"]).iterdir())
    assert "status.json" not in written and "cost.pkl" not in written
