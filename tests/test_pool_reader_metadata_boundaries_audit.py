"""Actual pool-reader CLI over a tiny local file and synthetic I/O counters."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


def _run_main(tmp_path, monkeypatch, *, physical_bytes, output):
    from experiments import glm_pool_read_ceiling as pool

    monkeypatch.chdir(tmp_path)
    data = b"bounded-fixture\n"
    (tmp_path / "fixture.bin").write_bytes(data)
    spec = tmp_path / "arms.json"
    spec.write_text(json.dumps({
        "root": str(tmp_path),
        "arms": [{"arm_id": "local-metadata-control", "readers": 1,
                  "n_files": 1, "bytes": len(data), "files": ["fixture.bin"],
                  "source_row": "synthetic-local-input-no-performance-claim"}],
    }))

    class MetadataSampler:
        samples = []

        def start(self):
            pass

        def halt(self):
            pass

        def join(self, timeout):
            pass

    io = iter([{"read_bytes": 0, "rchar": 0},
               {"read_bytes": physical_bytes, "rchar": len(data)}])
    monkeypatch.setattr(pool, "read_self_io", lambda: next(io))
    monkeypatch.setattr(pool, "read_diskstats", lambda: {})
    monkeypatch.setattr(pool, "read_arcstats", lambda: {})
    monkeypatch.setattr(pool, "loadavg", lambda: [0.0, 0.0, 0.0])
    monkeypatch.setattr(pool, "concurrent_claims", lambda: 0)
    monkeypatch.setattr(pool, "zfs_params", lambda: {})
    monkeypatch.setattr(pool, "Sampler", MetadataSampler)
    monkeypatch.setattr(sys, "argv", ["glm_pool_read_ceiling.py", "--arms", str(spec),
                                    "--block-bytes", "16", "--settle-s", "0",
                                    "--out", output])
    assert pool.main() == 0
    report = json.loads(Path(output).read_text())
    assert len(report["arms"]) == 1
    arm = report["arms"][0]
    assert arm["logical_bytes"] == len(data)
    assert arm["errors"] == []
    assert arm["source_row"] == "synthetic-local-input-no-performance-claim"
    assert (tmp_path / "fixture.bin").read_bytes() == data
    return arm


@pytest.mark.parametrize("physical_bytes", [0, 16], ids=["warm-zero", "positive"])
def test_pool_reader_main_publishes_warm_and_positive_io_metadata(
    tmp_path, monkeypatch, physical_bytes,
):
    arm = _run_main(tmp_path, monkeypatch, physical_bytes=physical_bytes,
                    output="nested/result.json")
    assert arm["self_physical_read_bytes"] == physical_bytes
    assert arm["box_hdd_over_self"] == (None if physical_bytes == 0 else 0.0)
    assert arm["cold"] is (physical_bytes > 0)


@pytest.mark.parametrize("output", ["result.json", "nested/result.json"],
                         ids=["basename", "nested"])
def test_pool_reader_main_writes_relative_output(tmp_path, monkeypatch, output):
    arm = _run_main(tmp_path, monkeypatch, physical_bytes=16, output=output)
    assert arm["box_hdd_over_self"] == 0.0
