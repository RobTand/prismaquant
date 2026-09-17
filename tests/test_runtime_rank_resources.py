"""Per-rank runtime resources: the spelling, its coverage rules, its arithmetic.

Every fixture here is synthetic and says so. A passing test proves what the
consumer accepts and refuses; it establishes no box's behaviour, and the
numbers typed into these records are not measurements. The one upstream
producer of such a record is a routed MoE owner receipt, whose reader is
``runtime_provenance.routed_rank_resources`` and whose end-to-end row path is
exercised in ``tests/test_native_receipt_table_routed.py``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from prismaquant.measured_runtime_prices import (
    CONTEXT_SCHEMA, RANK_RESOURCES_SCHEMA, RANK_TIMING_RULE, RANK_TOTALS_SCHEMA, SCHEMA,
    RuntimePriceError, RuntimeRankResources, RuntimeResources, admit_rank_budgets,
    compose_rank_totals, parse_measured_runtime_table, parse_row_resources,
    parse_runtime_context, pending_rank_device_bounds, RankDeviceBounds,
)

NOW = datetime(2026, 9, 16, 12, tzinfo=timezone.utc)
COST_SHA = "c" * 64
ROUTE = "synthetic-route"


def rank_record(rank, **updates):
    record = {"rank": rank, "resident_bytes": 1000 + rank,
              "peak_scratch_bytes": 10 + rank, "activation_bytes": 20 + rank,
              "workspace_resident_bytes": 64,
              "workspace_sha256": ("a" if rank == 0 else "b") * 64,
              "bound_sha256": ("c" if rank == 0 else "d") * 64}
    record.update(updates)
    return record


def vector(*, world_size=2, ranks=None, wire_bytes=None, **updates):
    payload = {"schema": RANK_RESOURCES_SCHEMA, "world_size": world_size, "prefill_ms": 2.0,
               "decode_ms": 1.0,
               "timing_rule": RANK_TIMING_RULE,
               "wire_bytes": 5000 if wire_bytes is None else wire_bytes,
               "wire_sha256": "0" * 64,
               "rank_medians_ms": {"prefill": [2.0] * world_size, "decode": [1.0] * world_size},
               "ranks": [rank_record(rank) for rank in range(world_size)] if ranks is None else ranks}
    payload.update(updates)
    return payload


def scalar(**updates):
    return dict(RuntimeResources(prefill_ms=2.0, decode_ms=1.0, serialized_bytes=100,
                                 resident_bytes=1000, peak_scratch_bytes=10,
                                 activation_bytes=20, kv_bytes=0).as_dict(), **updates)


def context_payload(*, tensor_parallel, structure="routed_moe", unit="layer", fmt="FP8_E4M3"):
    return {"schema": CONTEXT_SCHEMA,
            "serving_context": {"platform": "sm_121", "structure": structure,
                                "residency": "resident",
                                "runtime_image": "fixture@sha256:" + "a" * 64,
                                "execution_mode": "eager"},
            "gpu_identity": "synthetic", "runtime_sha256": "a" * 64, "source_sha256": "b" * 64,
            "calibration_sha256": "d" * 64, "prompt_tokens": 16, "batch_size": 1,
            "tensor_parallel": tensor_parallel, "graph_mode": "eager",
            "operator_routes": {unit: {fmt: ROUTE}}}


def row(resources, *, unit="layer", fmt="FP8_E4M3"):
    return {"unit": unit, "format": fmt,
            "binding": {"member_formats": {unit: fmt},
                        "member_operator_identity_sha256": {unit: "f" * 64},
                        "member_shapes": {unit: [4, 4]}, "operator_route": ROUTE},
            "resources": resources,
            "prefill": {"method": "cuda_events", "samples_ms": [2.0] * 3, "warmup_iterations": 3,
                        "receipt_path": "receipt.json", "receipt_sha256": "e" * 64},
            "decode": {"method": "cuda_events", "samples_ms": [1.0] * 3, "warmup_iterations": 3,
                       "receipt_path": "receipt.json", "receipt_sha256": "e" * 64}}


def table_payload(context, rows):
    return {"schema": SCHEMA, "table_id": "synthetic-only", "status": "proposal_data",
            "composition": "sequential_operator_sum", "context": context,
            "cost_sha256": COST_SHA,
            "measured_at": (NOW - timedelta(hours=1)).isoformat(),
            "valid_until": (NOW + timedelta(days=1)).isoformat(), "fixed_assignment": {},
            "fixed_resources": RuntimeResources(prefill_ms=0.0, decode_ms=0.0,
                                                serialized_bytes=0, resident_bytes=0,
                                                peak_scratch_bytes=0, activation_bytes=0).as_dict(),
            "fixed_resources_receipt_path": "receipt.json",
            "fixed_resources_receipt_sha256": "e" * 64, "rows": rows}


def parse(payload):
    return parse_measured_runtime_table(payload, expected_context=parse_runtime_context(payload["context"]),
                                        expected_cost_sha256=COST_SHA, now=NOW)


def test_a_tensor_parallel_table_prices_every_row_per_rank():
    payload = table_payload(context_payload(tensor_parallel=2), [row(vector())])
    parsed = parse(payload)
    resources = parsed.rows[0].resources
    assert isinstance(resources, RuntimeRankResources)
    assert resources.world_size == 2
    assert [entry.rank for entry in resources.ranks] == [0, 1]
    assert resources.rank(1).resident_bytes == 1001
    assert resources.as_dict() == vector(), "a parsed vector re-emits as the row it was written from"
    assert parsed.rows[0].as_dict()["resources"] == vector()


def test_a_scalar_row_under_a_tensor_parallel_context_is_refused():
    payload = table_payload(context_payload(tensor_parallel=2), [row(scalar())])
    with pytest.raises(RuntimePriceError, match="per-rank resources"):
        parse(payload)


def test_a_rank_vector_must_cover_the_world_the_context_declares():
    payload = table_payload(context_payload(tensor_parallel=2), [row(vector(world_size=1))])
    with pytest.raises(RuntimePriceError, match="cover a world of 1 where"):
        parse(payload)


def test_a_rank_vector_is_refused_in_a_single_device_table():
    payload = table_payload(context_payload(tensor_parallel=1, structure="dense"), [row(vector())])
    with pytest.raises(RuntimePriceError, match="cover a world of 2 where"):
        parse(payload)


@pytest.mark.parametrize("ranks,diagnostic", [
    ([rank_record(0), rank_record(0)], "cover ranks 0..1 once each and in order"),
    ([rank_record(1), rank_record(0)], "cover ranks 0..1 once each and in order"),
    ([rank_record(0)], "exactly one record per rank"),
])
def test_a_rank_gap_a_duplicate_or_a_short_vector_is_refused(ranks, diagnostic):
    payload = table_payload(context_payload(tensor_parallel=2),
                            [row(vector(world_size=2, ranks=ranks))])
    with pytest.raises(RuntimePriceError, match=diagnostic):
        parse(payload)


@pytest.mark.parametrize("updates", [
    {"resident_bytes": True},
    {"resident_bytes": 1.5},
    {"peak_scratch_bytes": -1},
    {"activation_bytes": None},
    {"bound_sha256": "not-a-digest"},
])
def test_an_axis_that_is_absent_or_not_a_nonnegative_integer_is_refused(updates):
    record = rank_record(0, **updates)
    with pytest.raises(RuntimePriceError):
        parse_row_resources(vector(ranks=[record, rank_record(1)]), tensor_parallel=2)


def test_a_rank_record_without_an_axis_is_refused():
    record = rank_record(0)
    del record["workspace_resident_bytes"]
    with pytest.raises(RuntimePriceError, match="expected exactly fields"):
        parse_row_resources(vector(ranks=[record, rank_record(1)]), tensor_parallel=2)


def test_a_workspace_identity_is_required_and_spelled_as_a_digest():
    with pytest.raises(RuntimePriceError, match="workspace_sha256"):
        parse_row_resources(vector(ranks=[rank_record(0, workspace_sha256="not-a-digest"),
                                          rank_record(1)]), tensor_parallel=2)


def test_an_unknown_per_rank_schema_is_refused():
    with pytest.raises(RuntimePriceError, match="unknown per-rank resource schema"):
        parse_row_resources(vector(schema="prismaquant.runtime_rank_resources.v2"),
                            tensor_parallel=2)


def test_an_unknown_timing_rule_is_refused():
    with pytest.raises(RuntimePriceError, match="unknown per-rank timing rule"):
        parse_row_resources(vector(timing_rule="sum_of_leaf_medians"), tensor_parallel=2)


def test_a_row_that_does_not_price_the_slowest_rank_is_refused():
    with pytest.raises(RuntimePriceError, match="slowest rank's prefill median"):
        parse_row_resources(vector(prefill_ms=2.0,
                                   rank_medians_ms={"prefill": [2.0, 5.0],
                                                    "decode": [1.0, 1.0]}),
                            tensor_parallel=2)


def test_a_row_may_price_no_decode_and_then_carries_no_rank_decode_median():
    payload = vector(decode_ms=None, rank_medians_ms={"prefill": [2.0, 2.0],
                                                      "decode": [None, None]})
    parsed = parse_row_resources(payload, tensor_parallel=2)
    assert parsed.decode_ms is None and parsed.as_dict() == payload


def test_a_dense_row_keeps_the_scalar_v2_spelling_byte_for_byte():
    """No scalar field changed meaning, so an old table re-reads identically."""
    resources = scalar()
    payload = table_payload(context_payload(tensor_parallel=1, structure="dense"), [row(resources)])
    parsed = parse(payload)
    assert isinstance(parsed.rows[0].resources, RuntimeResources)
    assert parsed.rows[0].as_dict()["resources"] == resources
    assert parsed.as_dict()["rows"][0]["resources"] == resources


def test_a_scalar_row_still_refuses_kv_and_the_off_step_peak():
    for extra, diagnostic in (({"kv_bytes": 1}, "KV belongs to fixed_resources"),
                              ({"non_step_transient_peak_bytes": 1}, "off-step transient peak")):
        with pytest.raises(RuntimePriceError, match=diagnostic):
            parse_row_resources(scalar(**extra), tensor_parallel=1)


def test_composition_adds_extensive_terms_per_rank_and_takes_per_rank_peaks():
    first = RuntimeRankResources.from_dict(vector(), tensor_parallel=2)
    second = RuntimeRankResources.from_dict(
        vector(ranks=[rank_record(0, peak_scratch_bytes=99, activation_bytes=5,
                                  resident_bytes=9),
                      rank_record(1, peak_scratch_bytes=3, activation_bytes=77,
                                  resident_bytes=13)]),
        tensor_parallel=2)
    totals = compose_rank_totals([first, second])
    assert totals.world_size == 2
    assert totals.wire_bytes == 10000, "one canonical artifact charge per unit, counted once"
    assert totals.resident_bytes == (1009, 1014)
    assert totals.peak_scratch_bytes == (99, 11), "peaks are per-rank maxima, never a sum of ranks"
    assert totals.activation_bytes == (20, 77)
    assert totals.workspace_resident_bytes == (128, 128)
    assert totals.as_dict()["schema"] == RANK_TOTALS_SCHEMA
    assert totals.as_dict()["withheld_terms"], "a reader is told which terms no total prices"


def test_a_single_device_row_is_its_own_rank_record():
    walked = RuntimeResources(prefill_ms=2.0, decode_ms=1.0, serialized_bytes=5,
                              resident_bytes=7, peak_scratch_bytes=11, activation_bytes=13)
    totals = compose_rank_totals([walked])
    assert totals.world_size == 1
    assert totals.resident_bytes == (7,) and totals.wire_bytes == 5
    assert totals.peak_scratch_bytes == (11,) and totals.activation_bytes == (13,)


def test_a_scalar_row_is_refused_inside_a_multi_rank_total():
    walked = RuntimeResources(prefill_ms=2.0, decode_ms=1.0, serialized_bytes=5,
                              resident_bytes=7, peak_scratch_bytes=11, activation_bytes=13)
    with pytest.raises(RuntimePriceError, match="mixes a scalar row"):
        compose_rank_totals([RuntimeRankResources.from_dict(vector(), tensor_parallel=2), walked])


def totals_for(*, rank0, rank1):
    records = [rank_record(0, resident_bytes=rank0, peak_scratch_bytes=0, activation_bytes=0,
                           workspace_resident_bytes=0),
               rank_record(1, resident_bytes=rank1, peak_scratch_bytes=0, activation_bytes=0,
                           workspace_resident_bytes=0)]
    return compose_rank_totals([
        RuntimeRankResources.from_dict(vector(ranks=records), tensor_parallel=2)])


def test_each_rank_is_admitted_against_its_own_budget():
    totals = totals_for(rank0=40, rank1=150)
    admitted = admit_rank_budgets(totals, budgets_per_rank=[100, 200], charge_per_rank=[0, 0],
                                  charge_refusal=None)
    assert admitted == (40, 150), "each rank returns its own total, not a world-wide one"


def test_an_imbalanced_rank_is_refused_even_where_the_mean_would_pass():
    totals = totals_for(rank0=40, rank1=150)
    budgets = [100, 100]
    assert sum(totals.resident_bytes) / 2 == 95 <= 100, "the mean would have admitted this world"
    assert sum(totals.resident_bytes) == 190 <= sum(budgets), "so would the sum"
    with pytest.raises(RuntimePriceError) as refusal:
        admit_rank_budgets(totals, budgets_per_rank=budgets, charge_per_rank=[0, 0],
                           charge_refusal=None)
    text = str(refusal.value)
    assert "rank 1 needs 150 device bytes where its own budget is 100" in text
    assert "rank 0" not in text, "an admitted rank is not named as over budget"


def test_a_missing_per_rank_charge_is_refused_with_the_gates_own_text():
    totals = totals_for(rank0=1, rank1=1)
    with pytest.raises(RuntimePriceError) as refusal:
        admit_rank_budgets(totals, budgets_per_rank=[100, 100], charge_per_rank=None,
                           charge_refusal="no qualified recomputable full-engine resource partition")
    assert "no qualified recomputable full-engine resource partition" in str(refusal.value)


def test_a_withheld_workspace_term_refuses_a_device_total():
    records = [rank_record(0, workspace_resident_bytes=64), rank_record(1, workspace_resident_bytes=0)]
    totals = compose_rank_totals([RuntimeRankResources.from_dict(vector(ranks=records),
                                                                 tensor_parallel=2)])
    with pytest.raises(RuntimePriceError, match="workspace_resident_bytes"):
        admit_rank_budgets(totals, budgets_per_rank=[10 ** 9, 10 ** 9], charge_per_rank=[0, 0],
                           charge_refusal=None)


@pytest.mark.parametrize("budgets,charge,diagnostic", [
    ([100], None, "exactly one record per rank"),
    ([100, 100], [0], "exactly one record per rank"),
])
def test_a_budget_or_charge_vector_must_cover_every_rank(budgets, charge, diagnostic):
    with pytest.raises(RuntimePriceError, match=diagnostic):
        admit_rank_budgets(totals_for(rank0=1, rank1=1), budgets_per_rank=budgets,
                           charge_per_rank=charge, charge_refusal=None)


def _ranked_row(prefill_ms, residents, *, world=2):
    return RuntimeRankResources.from_dict(
        {"schema": RANK_RESOURCES_SCHEMA, "world_size": world, "timing_rule": RANK_TIMING_RULE,
         "prefill_ms": prefill_ms, "decode_ms": None,
         "rank_medians_ms": {"prefill": [prefill_ms] * world, "decode": [None] * world},
         "wire_bytes": 1000, "wire_sha256": "0" * 64,
         "ranks": [{"rank": rank, "resident_bytes": residents[rank], "peak_scratch_bytes": 0,
                    "activation_bytes": 0, "workspace_resident_bytes": 0,
                    "workspace_sha256": "a" * 64, "bound_sha256": "b" * 64}
                   for rank in range(world)]},
        tensor_parallel=world)


def _admitted_bounds(budgets, charge):
    return RankDeviceBounds(len(budgets), "recomputed_full_engine_partition", tuple(budgets),
                            tuple(charge),
                            {"full_engine_report": {"path": "report.json", "sha256": "e" * 64},
                             "per_rank_partition": True})


def test_a_rank_budget_forces_the_frontier_to_keep_the_option_that_fits():
    """The rank dimension is in the search, so pruning cannot hide the tradeoff.

    ``FAST`` is strictly better on loss and prefill and breaches rank 1;
    ``SLOW`` fits (900+1 against 800 on rank 1, where ``SLOW`` needs 100+1). If
    the rank coordinate were checked after the frontier was built -- or dropped
    from dominance -- the DP would keep a frontier that the admission gate then
    rejects, and the feasible assignment would be gone.
    """
    from prismaquant.allocator_solver import Candidate, solve_runtime_frontier

    fast = Candidate("FAST", 4.0, 1000, 1.0)
    slow = Candidate("SLOW", 4.0, 1000, 2.0)
    resources = {("unit", "FAST"): _ranked_row(1.0, [10, 900]),
                 ("unit", "SLOW"): _ranked_row(5.0, [10, 100])}
    unbudgeted = solve_runtime_frontier({"unit": [fast, slow]}, resources,
                                       max_memory_bytes=10 ** 6, max_prefill_ms=10.0)
    assert [allocation.assignment["unit"] for allocation in unbudgeted][0] == "FAST"
    budgeted = solve_runtime_frontier({"unit": [fast, slow]}, resources,
                                      max_memory_bytes=10 ** 6, max_prefill_ms=10.0,
                                      rank_devices=_admitted_bounds([1000, 800], [1, 1]))
    assert [allocation.assignment["unit"] for allocation in budgeted] == ["SLOW"]
    assert budgeted[0].device_bytes is None, "a ranked row publishes no scalar device total"


def test_a_pending_rank_charge_prices_rank_dimensions_and_admits_nothing():
    from prismaquant.allocator_solver import Candidate, solve_runtime_frontier

    fast = Candidate("FAST", 4.0, 1000, 1.0)
    slow = Candidate("SLOW", 4.0, 1000, 2.0)
    resources = {("unit", "FAST"): _ranked_row(1.0, [10, 900]),
                 ("unit", "SLOW"): _ranked_row(5.0, [10, 100])}
    pending = pending_rank_device_bounds(2, [1000, 1000])
    with pytest.raises(RuntimePriceError, match="pending measurement"):
        pending.per_rank_headroom()
    frontier = solve_runtime_frontier({"unit": [fast, slow]}, resources,
                                      max_memory_bytes=10 ** 6, max_prefill_ms=10.0,
                                      rank_devices=pending)
    assert [allocation.assignment["unit"] for allocation in frontier] == ["FAST", "SLOW"], (
        "the rank dimensions price even while the charge is pending -- a common unknown "
        "constant cannot reorder them -- so the row that only wins on rank 1 residency is "
        "kept rather than pruned by an operator-terms-only frontier")
    assert all(allocation.device_bytes is None for allocation in frontier)


@pytest.mark.parametrize("world_size,provenance,charge,evidence,diagnostic", [
    (2, "pending_measurement", (1, 1), None, "carries no value and no evidence"),
    (2, "recomputed_full_engine_partition", None, None, "carries one value per rank"),
    (2, "recomputed_full_engine_partition", (0, 5), None, "not evidence"),
    (2, "recomputed_full_engine_partition", (1, 1), None, "recomputed full-engine"),
    (2, "declared_by_the_caller", (1, 1), None, "provenance"),
    (2, "pending_measurement", None, None, "positive"),
])
def test_rank_device_bounds_refuse_an_unearned_charge(world_size, provenance, charge, evidence,
                                                      diagnostic):
    with pytest.raises(RuntimePriceError, match=diagnostic):
        RankDeviceBounds(world_size, provenance, (1, 1) if diagnostic != "positive" else (0, 1),
                         charge, evidence)


def test_pending_bounds_admit_nothing_and_an_admitted_one_reports_headroom():
    pending = pending_rank_device_bounds(2, [1000, 2000])
    assert pending.admits_ranks is False
    assert pending.as_dict()["charge_per_rank"] is None
    admitted = _admitted_bounds([1000, 2000], [100, 400])
    assert admitted.admits_ranks is True
    assert admitted.per_rank_headroom() == (900, 1600)
    assert RankDeviceBounds.from_dict(admitted.as_dict()) == admitted
