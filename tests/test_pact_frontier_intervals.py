"""PQ #1659: frontier materiality comes from real bootstrap intervals.

A sweep replay attaches the ``prismaquant.pact_selection.v1`` record, and the
record's intervals reproduce from the receipts' own repeated samples at a
fixed seed. CPU-only; the sample counts are tiny on purpose.
"""
from __future__ import annotations

import pytest
from test_prefill_frontier_replay import _replay
from test_prefill_frontier_shape_only_scope import _sweep, _v2_sweep_fixture

from prismaquant.layer_config import _pact_selection_claim
from prismaquant.measured_runtime_prices import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_SEED,
    bootstrap_sum,
)
from prismaquant.pact_selection import (
    select_pact,
    validate_pact_selection,
)
from prismaquant.prefill_frontier import (
    PrefillFrontierError,
    _select,
    sweep_selection_points,
)

#: Two priced rows, thirty repeated samples each, fixed for every test here.
SAMPLES = [[100.0 + 0.1 * i for i in range(30)],
           [50.0 + 0.05 * i for i in range(30)]]
DRAWS = 2000
TABLE = {"table_id": "interval-test-table", "sha256": "0" * 64}


def _interval_point(point_id, sha, time_ms, boot, dloss, nbytes=1000):
    return {
        "point_id": point_id,
        "assignment_sha256": sha,
        "bytes": nbytes,
        "time_ms": time_ms,
        "time_interval_ms": [boot["p2.5"], boot["p97.5"]],
        "predicted_dloss": dloss,
        "kernel_lanes": None,
        "time_samples": {"samples_per_row": list(boot["samples_per_row"]),
                         "distinct_measurements": None},
    }


def test_same_seed_same_interval():
    first = bootstrap_sum(SAMPLES, draws=DRAWS, seed=BOOTSTRAP_SEED)
    second = bootstrap_sum(SAMPLES, draws=DRAWS, seed=BOOTSTRAP_SEED)
    assert first == second
    assert first["confidence"] == BOOTSTRAP_CONFIDENCE == 0.95
    assert first["samples_per_row"] == [30, 30]
    assert first["p2.5"] < first["p50"] < first["p97.5"]


def test_repeated_samples_resolve_material_never_unresolved():
    boot = bootstrap_sum(SAMPLES, draws=DRAWS, seed=BOOTSTRAP_SEED)
    spread = boot["p97.5"] - boot["p2.5"]
    fast = _interval_point("fast", "a" * 64, boot["p50"], boot, 0.5)
    # The accurate endpoint sits ten interval-widths above the fast one.
    slow_center = boot["p50"] + 10 * spread
    slow = _interval_point(
        "best", "b" * 64, slow_center,
        {"p2.5": slow_center - spread / 2, "p97.5": slow_center + spread / 2,
         "samples_per_row": [30, 30]},
        0.1)
    record = _select([fast, slow], regime_m=2048, table_identity=TABLE,
                     frontier_scope="test_scope", draws=DRAWS, seed=BOOTSTRAP_SEED)
    assert record["materiality"]["verdict"] == "material"
    validate_pact_selection(record, require_resolved=True)


def test_repeated_samples_resolve_flat_never_unresolved():
    boot = bootstrap_sum(SAMPLES, draws=DRAWS, seed=BOOTSTRAP_SEED)
    spread = boot["p97.5"] - boot["p2.5"]
    fast = _interval_point("fast", "a" * 64, boot["p50"], boot, 0.5)
    # Same interval, a quarter-width apart: distinct endpoints, overlapping bounds.
    best = _interval_point("best", "b" * 64, boot["p50"] + spread / 4, boot, 0.1)
    record = _select([fast, best], regime_m=2048, table_identity=TABLE,
                     frontier_scope="test_scope", draws=DRAWS, seed=BOOTSTRAP_SEED)
    assert record["materiality"]["verdict"] == "flat"
    validate_pact_selection(record, require_resolved=True)


def test_record_carries_time_samples_and_interval_provenance():
    boot = bootstrap_sum(SAMPLES, draws=DRAWS, seed=BOOTSTRAP_SEED)
    points = [_interval_point("fast", "a" * 64, boot["p50"], boot, 0.5),
              _interval_point("best", "b" * 64, boot["p50"] + 1.0, boot, 0.1)]
    record = _select(points, regime_m=2048, table_identity=TABLE,
                     frontier_scope="test_scope", draws=DRAWS, seed=BOOTSTRAP_SEED)
    assert record["time_axis"]["interval"] == {
        "method": "per_shape_bootstrap_sum_v1",
        "confidence": BOOTSTRAP_CONFIDENCE, "draws": DRAWS, "seed": BOOTSTRAP_SEED}
    for entry in record["roster"]:
        assert entry["time_samples"] == {
            "samples_per_row": [30, 30], "distinct_measurements": None}


def test_sweep_point_without_bootstrap_raises():
    point = {"feasible": True, "assignment_sha256": "c" * 64,
             "payload_bytes": 1000, "attained_prefill_ms": 150.0,
             "attained_prefill_ms_bootstrap": None,
             "predicted_dloss": 0.2}
    with pytest.raises(PrefillFrontierError, match="carries no prefill bootstrap interval"):
        sweep_selection_points([point])


def test_sweep_selection_dedupes_by_digest_keeps_feasible_only():
    boot = bootstrap_sum(SAMPLES, draws=DRAWS, seed=BOOTSTRAP_SEED)
    blob = {"attained_prefill_ms_bootstrap": dict(
        boot, samples_per_row=list(boot["samples_per_row"])),
        "predicted_dloss": 0.2}
    tight = {"feasible": True, "assignment_sha256": "d" * 64,
             "payload_bytes": 1000, "attained_prefill_ms": 150.0, **blob}
    loose = dict(tight, attained_prefill_ms=999.0)
    refused = dict(tight, feasible=False, assignment_sha256="e" * 64)
    # Points arrive tightest-SLO first, so the first occurrence stands for the digest.
    selected = sweep_selection_points([tight, refused, loose])
    assert [p["assignment_sha256"] for p in selected] == ["d" * 64]
    assert selected[0]["time_ms"] == 150.0


def test_sweep_replay_attaches_record_and_passes_claim(tmp_path, monkeypatch):
    allocator_argv = _v2_sweep_fixture(tmp_path, monkeypatch)
    code, doc = _sweep(tmp_path, allocator_argv)
    assert code == 0
    assert doc["regime_m"] >= 1
    selection = doc["pact_selection"]
    validate_pact_selection(selection, require_resolved=True)
    assert selection["time_axis"]["interval"]["seed"] == BOOTSTRAP_SEED
    point = next(p for p in doc["points"] if p["feasible"])
    result = _replay(tmp_path, point, tmp_path / "replayed.json")
    from prismaquant.layer_config import LAYER_CONFIG_META_KEY
    meta = result[LAYER_CONFIG_META_KEY]
    replay = meta["prefill_frontier_replay"]
    assert replay["pact_selection_sha256"] == selection["identity_sha256"]
    assert meta["pact_selection"]["identity_sha256"] == selection["identity_sha256"]
    assert _pact_selection_claim(meta["pact_selection"], replay)["identity_sha256"] == \
        selection["identity_sha256"]


def test_select_pact_without_time_interval_stays_unresolved_capable():
    # The shared selector did not lose its old shape: no interval named, no
    # samples carried, and the record still validates on its own terms.
    points = [{"point_id": "p1", "assignment_sha256": "f" * 64, "bytes": 100,
               "time_ms": 10.0, "time_interval_ms": [9.0, 11.0],
               "predicted_dloss": 0.3, "kernel_lanes": None}]
    record = select_pact(points, regime_m=8, table_identity=TABLE,
                         frontier_scope="legacy_scope")
    assert record["time_axis"]["interval"] is None
    assert record["roster"][0]["time_samples"] is None
    validate_pact_selection(record)
