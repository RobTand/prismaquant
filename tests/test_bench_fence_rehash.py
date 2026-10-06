"""Exercise the fence benchmark's real digest path, without a speed claim."""
import hashlib
import json

from prismaquant import io_engine, joint_catalog_extension
from tools import bench_fence_rehash


def test_fence_benchmark_hashes_tiny_synthetic_wires(tmp_path, monkeypatch, capsys):
    # Register the tool's instrumentation for pytest teardown, since main also
    # wraps read_stream while collecting counters.
    monkeypatch.setattr(io_engine, "read_stream", io_engine.read_stream)
    monkeypatch.setattr(joint_catalog_extension, "FENCE_REHASHED", {})
    real_digest = hashlib.file_digest
    assert bench_fence_rehash.main([
        "--dir", str(tmp_path), "--n", "2", "--mib", "1", "--repeats", "2",
    ]) == 0
    assert hashlib.file_digest is real_digest
    lines = capsys.readouterr().out.splitlines()
    report = json.loads(next(line.removeprefix("BENCH ")
                             for line in lines if line.startswith("BENCH ")))
    assert report["mode"] == "fence"
    assert report["n"] == 2
    assert report["total_bytes"] == 2 << 20
    assert len(report["runs"]) == 2
    assert all(run["distinct_hash_threads"] >= 1 for run in report["runs"])
    assert all(run["peak_concurrent_hashes"] >= 1 for run in report["runs"])
    assert joint_catalog_extension.FENCE_REHASHED == {"bench wire fence": 2}
