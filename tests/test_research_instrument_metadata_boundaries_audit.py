"""CLI metadata and tiny local-path controls, without GPU or NFS execution."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("idle_util", [0, 5, None], ids=["zero", "positive-low", "missing"])
def test_meter_main_preserves_observed_zero_falling_edge(tmp_path, monkeypatch, idle_util):
    from experiments import gb10_meter_step as meter

    now = [0.0]
    calls = []

    def advance(seconds):
        now[0] += seconds

    samples = [
        {"time": 0.0, "power": 10.0, "instant": 10.0, "util": idle_util},
        {"time": 6.0, "power": 80.0, "instant": 80.0, "util": 95},
        {"time": 11.0, "power": 80.0, "instant": 80.0, "util": 95},
        {"time": 18.25, "power": 10.0, "instant": 10.0, "util": idle_util},
        {"time": 26.0, "power": 20.0, "instant": 20.0, "util": 95},
        {"time": 31.0, "power": 20.0, "instant": 20.0, "util": 95},
        {"time": 36.25, "power": 10.0, "instant": 10.0, "util": idle_util},
    ]
    for row in samples:
        row["source"] = "synthetic_metadata_no_gpu_measurement"

    def sampler(stop, rows):
        calls.append("metadata-sampler")
        rows.extend(samples)

    class MetadataThread:
        def __init__(self, *, target, args, daemon):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

        def join(self):
            pass

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata replay must not launch an external process")

    monkeypatch.setattr(meter, "time", SimpleNamespace(time=lambda: now[0], sleep=advance))
    monkeypatch.setattr(meter, "torch", SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: calls.append("metadata-sync"),
        get_device_name=lambda index: "metadata-only; no GPU")))
    monkeypatch.setattr(meter, "threading", SimpleNamespace(
        Event=lambda: SimpleNamespace(set=lambda: None), Thread=MetadataThread))
    monkeypatch.setattr(meter, "subprocess", SimpleNamespace(run=forbidden))
    monkeypatch.setattr(meter, "sampler", sampler)
    monkeypatch.setattr(meter, "busy_matmul", advance)
    monkeypatch.setattr(meter, "busy_tiny", advance)
    monkeypatch.setattr(sys, "argv", ["gb10_meter_step.py", "--out", str(tmp_path)])
    meter.main()
    record = json.loads((tmp_path / "samples.json").read_text())
    assert record["samples"] == samples
    assert record["device"] == "metadata-only; no GPU"
    report = json.loads((tmp_path / "report.json").read_text())
    expected = None if idle_util is None else 0.25
    assert report["matmul"]["fall_util_le10"] == expected
    assert report["tiny"]["fall_util_le10"] == expected
    assert calls == ["metadata-sync", "metadata-sampler"]


@pytest.mark.parametrize("module_name", ["glm_manifest_read", "glm_nfs_row_read"])
@pytest.mark.parametrize("output", ["result.json", "nested/result.json"], ids=["basename", "nested"])
def test_storage_reader_main_writes_relative_output_from_tiny_local_input(
    tmp_path, monkeypatch, module_name, output,
):
    module = importlib.import_module(f"experiments.{module_name}")
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "fixture.bin"
    data = b"bounded-fixture\n"
    source.write_bytes(data)
    if module_name == "glm_manifest_read":
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"entries": [{"path": str(source), "bytes": len(data)}],
                                        "entry_count": 1, "total_bytes": len(data)}))

        snapshots = iter([1.0, 2.0])

        def sample():
            return {"unix": next(snapshots), "arc": {},
                    "disk": {name: (0, 0) for name in (*module.HDDS, *module.NVMES)},
                    "nfsd": (0, 0), "loadavg": ["0", "0", "0"]}

        monkeypatch.setattr(module, "sample", sample)
        code = module.main(["--manifest", str(manifest), "--mount-map", f"{tmp_path}={tmp_path}",
                            "--readers", "1", "--settle-s", "0",
                            "--label", "local-path-control-no-performance-claim", "--out", output])
    else:
        qname = "model.layers.0.self_attn.q_proj"
        units = tmp_path / "units.json"
        units.write_text(json.dumps({"groups": [{"members": [qname]}]}))
        capture = tmp_path / "capture.json"
        capture.write_text(json.dumps({"entries": {qname: {"path": source.name}}}))
        (tmp_path / "plan.json").write_text(json.dumps({
            "rows": [{"row_id": "fixture", "units": str(units)}],
            "calibration_cache": {"path": str(capture)},
        }))
        monkeypatch.setattr(module, "nfs_client_stats", lambda: {"normal_read": 0})
        monkeypatch.setattr(module, "meminfo", lambda: {"MemAvailable": 0})
        monkeypatch.setattr(sys, "argv", ["glm_nfs_row_read.py", "--row", "fixture",
                                        "--workspace", str(tmp_path), "--readers", "1",
                                        "--block-bytes", "16", "--label", "local-path-control-no-performance-claim",
                                        "--out", output])
        code = module.main()
    assert code == 0
    report = json.loads(Path(output).read_text())
    assert report["label"] == "local-path-control-no-performance-claim"
    if module_name == "glm_manifest_read":
        assert report["arms"][0]["bytes_read"] == len(data)
        assert report["arms"][0]["error_count"] == 0
    else:
        assert report["bytes"] == len(data)
        assert report["errors"] == []
    assert source.read_bytes() == data
