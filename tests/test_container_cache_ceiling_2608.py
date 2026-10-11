"""A measured ceiling needs complete GPU workload evidence (PQ #2608)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant import container_cache_peak as peak_mod


def _growth_workload(root: Path, total: int = 600000):
    def run():
        path = root / "workload.bin"
        written = 0
        with open(path, "wb") as handle:
            while written < total:
                handle.write(b"\x5a" * 50000)
                handle.flush()
                written += 50000
                time.sleep(0.06)
        path.unlink()

    return run


def _gpu_workload() -> dict:
    return {"quantiser": "fp4_e2m1/g16", "shape": [64, 256], "seed": 2463,
            "device": "cuda:0",
            "kda_probe": {"status": "ran", "sha256": "0" * 64}}


def _complete_receipt(tmp_path: Path) -> dict:
    cache = tmp_path / "cache"
    cache.mkdir()
    receipt = peak_mod.measure_around(
        _growth_workload(cache), cache, interval_s=0.02)
    assert receipt["measurement"]["valid"], receipt["measurement"]["errors"]
    receipt["workload"] = _gpu_workload()
    receipt["device"] = "cuda"
    receipt["power"] = {"summary": {"sample_count": 3},
                        "samples": [10.0, 11.0, 10.5], "times": [1.0, 2.0, 3.0],
                        "error": None}
    return receipt


def test_ceiling_refuses_receipt_without_gpu_workload(tmp_path):
    receipt = _complete_receipt(tmp_path)
    receipt["workload"] = {}
    with pytest.raises(ValueError, match="GPU workload"):
        peak_mod.derive_cache_ceiling_from_receipt(
            receipt, headroom_bytes=4341760)


def test_ceiling_refuses_reservation_without_peaks(tmp_path):
    receipt = _complete_receipt(tmp_path)
    receipt["measurement"] = {"samples": [], "sample_count": 0,
                              "peak_allocated_bytes": 0,
                              "peak_sample_index": None,
                              "errors": [], "gaps": [],
                              "incomplete_scan": False, "valid": False,
                              "child_exitcode": 0}
    with pytest.raises(ValueError, match="peak"):
        peak_mod.derive_cache_ceiling_from_receipt(
            receipt, headroom_bytes=4341760)


def test_ceiling_refuses_partial_evidence_with_missing_peak(tmp_path):
    receipt = _complete_receipt(tmp_path)
    receipt["measurement"] = dict(receipt["measurement"])
    receipt["measurement"]["peak_allocated_bytes"] = 0
    receipt["measurement"]["peak_sample_index"] = None
    receipt["measurement"]["valid"] = False
    with pytest.raises(ValueError, match="peak"):
        peak_mod.derive_cache_ceiling_from_receipt(
            receipt, headroom_bytes=4341760)


def test_killed_child_refuses_ceiling_and_names_coverage_gap(tmp_path):
    (tmp_path / "ballast.bin").write_bytes(b"\x5a" * 100000)
    sampler = peak_mod.CachePeakSampler(tmp_path, interval_s=0.05)
    sampler.__enter__()
    sampler._process.kill()
    sampler._process.join(timeout=5)
    sampler.stop()
    result = sampler.result()
    assert not result["valid"]
    assert result["incomplete_scan"]
    receipt = {"measurement": result, "workload": _gpu_workload(),
               "device": "cuda", "failure": None}
    with pytest.raises(ValueError, match="cover the complete row"):
        peak_mod.derive_cache_ceiling_from_receipt(
            receipt, headroom_bytes=4341760)


def test_complete_evidence_matches_direct_derivation(tmp_path):
    receipt = _complete_receipt(tmp_path)
    headroom = 4341760
    assert (peak_mod.derive_cache_ceiling_from_receipt(
        receipt, headroom_bytes=headroom)
        == peak_mod.derive_cache_ceiling(
            receipt["measurement"]["peak_allocated_bytes"],
            headroom_bytes=headroom))
