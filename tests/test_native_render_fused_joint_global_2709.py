"""The witness diagnostic gate must not depend on filesystem clock progress.

Run the original gate with colliding ctime observations, while retaining real
file I/O and the production witness decision. A metadata operation can happen
within the same timestamp tick; arranging diagnostic drift is the test's job.
"""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def release_gate():
    path = Path(__file__).with_name("test_native_render_fused_joint_global.py")
    spec = importlib.util.spec_from_file_location("joint_global_release_gate_2709", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FrozenCtime:
    def __init__(self, observed, ctime_ns):
        self._observed = observed
        self.st_ctime_ns = ctime_ns
        self.st_ctime = ctime_ns / 1_000_000_000

    def __getattr__(self, name):
        return getattr(self._observed, name)


def test_witness_diagnostic_gate_runs_real_refusals_without_clock_progress(
    release_gate, tmp_path, monkeypatch,
):
    shards = {tmp_path / cause / "entry__NVFP4.pt" for cause in ("device", "ctime")}
    first_ctimes = {}
    real_stat = Path.stat

    def frozen_stat(path, *args, **kwargs):
        observed = real_stat(path, *args, **kwargs)
        if path not in shards:
            return observed
        ctime = first_ctimes.setdefault(path, observed.st_ctime_ns)
        return _FrozenCtime(observed, ctime)

    # Path.lstat delegates to Path.stat, so both fixture recording and the
    # production witness observe the collision. All other stat fields stay real.
    monkeypatch.setattr(Path, "stat", frozen_stat)
    cache_type = release_gate.ProductionWeightCache
    real_get = cache_type.get
    real_witness = cache_type.witnessed_nvfp4_joint_global
    reads = []
    refusals = []

    def observed_get(cache, *args, **kwargs):
        result = real_get(cache, *args, **kwargs)
        assert torch.equal(result, torch.eye(2))
        reads.append(Path(cache.cache_dir).name)
        return result

    def observed_witness(cache, name, fmt, *, where):
        cause = Path(cache.cache_dir).name
        witness = release_gate.nvfp4_joint_global_witnesses(
            cache.metadata, where="test 2709")[(name, fmt)]
        shard = Path(cache.cache_dir) / witness.shard.name
        live = release_gate.file_stat_signature(shard.lstat())
        changed = [i for i, (a, b) in enumerate(zip(witness.shard.stat, live)) if a != b]
        assert changed == ([0] if cause == "device" else [4])
        try:
            return real_witness(cache, name, fmt, where=where)
        except RuntimeError as error:
            refusals.append((cause, str(error)))
            raise

    monkeypatch.setattr(cache_type, "get", observed_get)
    monkeypatch.setattr(cache_type, "witnessed_nvfp4_joint_global", observed_witness)
    for cause in ("device", "ctime"):
        scratch = tmp_path / cause
        scratch.mkdir()
        release_gate.test_export_witness_refusal_distinguishes_view_and_metadata_changes(
            cause, scratch, monkeypatch)

    assert reads == ["device", "ctime"], "both gates must read the real payload"
    assert [cause for cause, _ in refusals] == ["device", "ctime"]
    assert "device-only mismatch (" in refusals[0][1]
    assert "changed fields" not in refusals[0][1]
    assert "changed fields: ctime_ns" in refusals[1][1]
    assert "permissions, ownership" in refusals[1][1]
