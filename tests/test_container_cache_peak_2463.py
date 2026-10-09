"""The measured cache ceiling derives from the recorded peak (PQ #2463)."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant import container_cache_peak as peak_mod
from tests.fixtures import container_cache_ceiling_2463 as fixture_mod


def _growth_workload(root: Path, total: int, chunk: int = 50000,
                     interval: float = 0.06):
    def run():
        path = root / "workload.bin"
        written = 0
        with open(path, "wb") as handle:
            while written < total:
                handle.write(b"\x5a" * min(chunk, total - written))
                handle.flush()
                written += chunk
                time.sleep(interval)
        path.unlink()

    return run


def test_peak_exceeds_final_size_on_grow_then_shrink_workload(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    receipt = peak_mod.measure_around(
        _growth_workload(cache, 600000), cache, interval_s=0.02)
    measurement = receipt["measurement"]
    assert measurement["valid"], measurement["errors"]
    assert measurement["sample_count"] >= 2
    assert measurement["peak_allocated_bytes"] > measurement["final_allocated_bytes"]
    assert measurement["final_allocated_bytes"] == 0


def test_initial_inventory_is_captured_before_the_row_runs(tmp_path):
    """The receipt's initial state is the pre-row inventory, not the final one."""
    cache = tmp_path / "cache"
    cache.mkdir()
    warm = cache / "warm.bin"
    warm.write_bytes(b"\x5a" * 200000)
    before = peak_mod.describe_initial_state(cache)
    assert before["totals"]["files"] == 1
    receipt = peak_mod.measure_around(
        _growth_workload(cache, 600000), cache, interval_s=0.02)
    initial = receipt["initial_state"]["totals"]
    assert initial["allocated_bytes"] == before["totals"]["allocated_bytes"]
    assert initial["files"] == before["totals"]["files"] == 1
    # The warm file survives the row: the final inventory equals the
    # initial one, and the peak exceeds both.
    assert receipt["measurement"]["final_allocated_bytes"] == initial["allocated_bytes"]
    assert receipt["measurement"]["peak_allocated_bytes"] > initial["allocated_bytes"]


def test_failed_scan_invalidates_the_evidence(tmp_path):
    sampler = peak_mod.CachePeakSampler(tmp_path / "missing", interval_s=0.02)
    with sampler:
        time.sleep(0.08)
    result = sampler.result()
    assert not result["valid"]
    assert result["errors"]


def test_crash_marks_the_scan_incomplete(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()

    def fail():
        (cache / "partial.bin").write_bytes(b"\x5a" * 100000)
        raise RuntimeError("row crashed")

    with pytest.raises(RuntimeError, match="row crashed"):
        peak_mod.measure_around(fail, cache, interval_s=0.02)


def test_hard_links_count_once(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    target = cache / "target.bin"
    target.write_bytes(b"\x5a" * 100000)
    import os
    os.link(target, cache / "link.bin")
    totals = peak_mod.scan_cache_bytes(cache)
    assert totals["allocated_bytes"] == 512 * target.stat().st_blocks
    assert totals["files"] == 1


def test_ceiling_formula_rounds_up_to_gib():
    gib = peak_mod.GIB_BYTES
    assert peak_mod.derive_cache_ceiling(0, headroom_bytes=0)["ceiling_bytes"] == gib
    assert peak_mod.derive_cache_ceiling(1, headroom_bytes=0)["ceiling_bytes"] == gib
    assert peak_mod.derive_cache_ceiling(gib, headroom_bytes=0)["ceiling_bytes"] == gib
    derived = peak_mod.derive_cache_ceiling(gib + 1, headroom_bytes=1)
    assert derived["ceiling_bytes"] == 2 * gib


def test_ceiling_rejects_negative_inputs():
    with pytest.raises(ValueError):
        peak_mod.derive_cache_ceiling(-1, headroom_bytes=0)
    with pytest.raises(ValueError):
        peak_mod.derive_cache_ceiling(0, headroom_bytes=-1)


def test_fixture_binds_peak_inputs_ceiling_and_pb_reservation():
    fixture = dict(fixture_mod.FIXTURE)
    assert fixture["schema"] == peak_mod.CEILING_SCHEMA
    assert fixture["gib_bytes"] == peak_mod.GIB_BYTES
    derived = peak_mod.derive_cache_ceiling(
        fixture["peak_allocated_bytes"],
        headroom_bytes=fixture["headroom_bytes"])
    assert derived["formula"] == fixture["formula"]
    assert derived["ceiling_bytes"] == fixture["ceiling_bytes"]
    assert fixture["pb_cache_gib"] == math.ceil(
        fixture["ceiling_bytes"] / peak_mod.GIB_BYTES)
    assert "measurement_digest" in fixture and fixture["measurement_digest"]
    root = Path(__file__).resolve().parents[1] / "docs" / "measurements"
    for key in ("measurement_digest", "repeat_digest"):
        receipt_path = root / fixture[{"measurement_digest": "measurement_receipt",
                                       "repeat_digest": "repeat_receipt"}[key]]
        raw = receipt_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == fixture[key]
        receipt = json.loads(raw)
        peak = receipt["measurement"]["peak_allocated_bytes"]
        assert receipt["measurement"]["valid"], receipt["measurement"]["errors"]
        assert not receipt["measurement"]["gaps"]
        assert not receipt["measurement"]["incomplete_scan"]
        assert peak == fixture[{"measurement_digest": "peak_allocated_bytes",
                                "repeat_digest": "repeat_peak_allocated_bytes"}[key]]
    samples = json.loads((root / fixture["measurement_receipt"]).read_bytes())["measurement"]["samples"]
    growths = [later["allocated_bytes"] - earlier["allocated_bytes"]
               for earlier, later in zip(samples, samples[1:])
               if later["allocated_bytes"] > earlier["allocated_bytes"]]
    assert growths, "no recorded cache growth justifies headroom"
    assert fixture["headroom_bytes"] >= 4 * max(growths), fixture["headroom_basis"]


def test_both_peaks_must_fit_the_ceiling_or_it_is_rejected():
    fixture = fixture_mod.FIXTURE
    ceiling = fixture["ceiling_bytes"]
    assert fixture["peak_allocated_bytes"] <= ceiling
    assert fixture["repeat_peak_allocated_bytes"] <= ceiling
    assert max(fixture["peak_allocated_bytes"],
               fixture["repeat_peak_allocated_bytes"]) <= ceiling
    recomputed = peak_mod.derive_cache_ceiling(
        max(fixture["peak_allocated_bytes"],
            fixture["repeat_peak_allocated_bytes"]),
        headroom_bytes=fixture["headroom_bytes"])
    assert recomputed["ceiling_bytes"] == ceiling
    with pytest.raises(ValueError):
        peak_mod.derive_cache_ceiling(-1, headroom_bytes=0)


def test_pb_charges_cache_gib_plus_each_scratch_reservation():
    if importlib.util.find_spec("prismabuild") is None:
        pytest.skip("public PB SDK unavailable; pricing needs a qualified receipt")
    from prismabuild.local_scratch import scratch_terms

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    runner = importlib.import_module("tools.tessera_campaign_container")
    cache_gib = fixture_mod.FIXTURE["pb_cache_gib"]
    names = runner.CONTAINER_CACHE_SCRATCH_ENV
    root = "/pq-fixture-cache-charge-1091/compile"
    declared = {names[0]: root, names[1]: str(cache_gib * (1 << 30)),
                runner.LOCAL_SCRATCH_PAIRS_ENV: f"{names[0]}:{names[1]}"}
    assert scratch_terms(declared) == {"spool_gb": cache_gib}
