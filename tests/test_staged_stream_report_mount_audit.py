"""Real benchmark main/report with tiny owned bytes, not throughput evidence.

Doubles: stage-mount existence/discovery, map admission/main resolver, the
reader's offline residency policy, read-shape lookup, cache advice and IO/NFS
counters. Tensor header/payload reads, digest/value comparison, main loop and
JSON output are real. No live stage, lease, fleet or throughput is qualified.
"""
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import io_spans, mount_facts, residency_map, residency_shard_reader
from prismaquant.mount_facts import MountFacts
from tools import staged_read_stream_ab as benchmark


@pytest.mark.parametrize("mount_present", [True, False], ids=["mount-facts", "missing-mount-control"])
def test_real_main_serializes_found_mount_and_owned_tensor_result(
        tmp_path, monkeypatch, mount_present):
    source = tmp_path / "owned.safetensors"
    save_file({"weight": torch.arange(8, dtype=torch.float32).reshape(2, 4)}, str(source))
    manifest = tmp_path / "map.json"
    manifest.write_text(json.dumps({
        "manifest_sha256": "a" * 64,
        "entries": {f"0:{source}": {
            "offset": 0, "bytes": source.stat().st_size, "stage_path": str(source),
        }},
    }))
    output = tmp_path / "result.json"
    options = {"rsize": "1048576", "nconnect": "4"}
    facts = MountFacts("/stage/prewarm", "nfs4", MappingProxyType(options))
    monkeypatch.setattr(mount_facts, "find_mount",
                        lambda *_args, **_kwargs: facts if mount_present else None)
    isdir = os.path.isdir
    monkeypatch.setattr(os.path, "isdir",
                        lambda path: True if str(path) == "/stage/prewarm" else isdir(path))
    monkeypatch.setenv("PRISMABUILD_RESIDENCY_MAP", str(manifest))
    monkeypatch.setattr(residency_map, "bind_residency_manifest", lambda _digest: None)
    monkeypatch.setattr(residency_map, "residency_resolver", lambda: object())
    # Actual source_safe_open uses its normal offline/local branch for owned bytes.
    monkeypatch.setattr(residency_shard_reader, "residency_resolver", lambda: None)
    monkeypatch.setattr(residency_shard_reader, "policy_is_active", lambda: False)
    monkeypatch.setattr(residency_shard_reader, "_read_shape", lambda _path: None)
    monkeypatch.setattr(benchmark, "drop_client_cache", lambda _paths: None)
    ticks = {"nfs": 0, "process": 0}

    def nfs_counters():
        ticks["nfs"] += 1
        count = ticks["nfs"] * (1 << 20)
        return {"/stage/prewarm": (count, count), "/mnt/shared": (0, 0)}

    def process_counters():
        ticks["process"] += 1
        return {"rchar": ticks["process"] * 32, "read_bytes": ticks["process"] * 32}

    monkeypatch.setattr(io_spans, "nfs_read_bytes", nfs_counters)
    monkeypatch.setattr(io_spans, "read_proc_io", process_counters)
    monkeypatch.setattr(sys, "argv", [
        "staged-read-audit", "--streams", "1", "--repeats", "1", "--tensors", "1",
        "--json-out", str(output),
    ])
    assert benchmark.main() == 0
    result = json.loads(output.read_text())
    assert result["mount"] == {
        "fstype": "nfs4" if mount_present else None,
        "options": options if mount_present else None,
    }
    assert result["mismatches"] == []
    assert len(result["targets"]) == len(result["records"]) == 1
    assert result["targets"][0]["bytes"] == 8 * 4
    assert len(result["records"][0]["sha256"]) == 64
