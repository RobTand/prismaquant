"""Per-rank runtime resources: the spelling, its coverage rules, its arithmetic.

Every fixture here is synthetic and says so. A passing test proves what the
consumer accepts and refuses; it establishes no box's behaviour, and the
numbers typed into these records are not measurements. The one upstream
producer of such a record is a routed MoE owner receipt, whose reader is
``runtime_provenance.routed_rank_resources`` and whose end-to-end row path is
exercised in ``tests/test_native_receipt_table_routed.py``.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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
    assert totals.workspace_resident_bytes == (64, 64), (
        "both rows carry the same frozen workspace identity per rank, and one "
        "process-global allocation is charged once per rank")
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


def test_a_shared_workspace_identity_is_charged_once_per_rank():
    """Two rows viewing one frozen allocation charge those bytes once, not twice.

    The ``vllm.WorkspaceManager`` allocation is process-global persistent state:
    a device total that charged it once per priced row would charge for
    allocations nobody made, and one that dropped it would price a box without
    the state the runtime holds.
    """
    resources = RuntimeRankResources.from_dict(vector(), tensor_parallel=2)
    totals = compose_rank_totals([resources, resources])
    assert totals.workspace_resident_bytes == (64, 64), "one identity, one allocation per rank"
    admitted = admit_rank_budgets(totals, budgets_per_rank=[10 ** 9, 10 ** 9],
                                  charge_per_rank=[1, 1], charge_refusal=None)
    assert admitted == (2000 + 20 + 10 + 64 + 1, 2002 + 21 + 11 + 64 + 1)


def test_distinct_workspace_identities_add_and_one_identity_at_two_sizes_is_refused():
    first = RuntimeRankResources.from_dict(vector(), tensor_parallel=2)
    second = RuntimeRankResources.from_dict(
        vector(ranks=[rank_record(0, workspace_sha256="e" * 64, workspace_resident_bytes=32),
                      rank_record(1, workspace_sha256="f" * 64, workspace_resident_bytes=16)]),
        tensor_parallel=2)
    totals = compose_rank_totals([first, second])
    assert totals.workspace_resident_bytes == (96, 80), (
        "distinct frozen identities are distinct allocations and add")
    conflicted = RuntimeRankResources.from_dict(
        vector(ranks=[rank_record(0, workspace_resident_bytes=8), rank_record(1)]),
        tensor_parallel=2)
    with pytest.raises(RuntimePriceError, match="one frozen allocation cannot be two"):
        compose_rank_totals([first, conflicted])


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
    """Bounds the way the recomputation produces them, not the way a payload claims them."""
    from prismaquant.runtime_provenance import RankFixedCharge

    verdict = RankFixedCharge(
        world_size=len(budgets), charge_per_rank=tuple(charge),
        per_rank_terms=tuple({"fixed_resident": value} for value in charge),
        evidence={"full_engine_report": {"path": "report.json", "sha256": "e" * 64},
                  "per_rank_partition": True},
        partition_sha256="d" * 64)
    return RankDeviceBounds.recomputed(world_size=len(budgets), budgets_per_rank=tuple(budgets),
                                       charge_per_rank=tuple(charge), recomputation=verdict)


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


def _workspace_row(prefill_ms, residents, workspace, *, world=2):
    """A ranked row with explicitly stated per-rank workspace identities."""
    return RuntimeRankResources.from_dict(
        {"schema": RANK_RESOURCES_SCHEMA, "world_size": world, "timing_rule": RANK_TIMING_RULE,
         "prefill_ms": prefill_ms, "decode_ms": None,
         "rank_medians_ms": {"prefill": [prefill_ms] * world, "decode": [None] * world},
         "wire_bytes": 1000, "wire_sha256": "0" * 64,
         "ranks": [{"rank": rank, "resident_bytes": residents[rank], "peak_scratch_bytes": 0,
                    "activation_bytes": 0, "workspace_resident_bytes": workspace[rank][1],
                    "workspace_sha256": workspace[rank][0], "bound_sha256": "b" * 64}
                   for rank in range(world)]},
        tensor_parallel=world)


def test_a_fast_option_whose_workspace_does_not_fit_loses_to_the_slower_one():
    """The workspace is composed inside the fold, so the budget sees it.

    ``FAST`` wins on loss and prefill and would dominate ``SLOW`` outright; its
    runtime-global workspace is what breaches the rank, and a search that
    composed the workspace after the frontier would have pruned ``SLOW`` before
    that number existed.
    """
    from prismaquant.allocator_solver import Candidate, solve_runtime_frontier

    fast = Candidate("FAST", 4.0, 1000, 1.0)
    slow = Candidate("SLOW", 4.0, 1000, 2.0)
    resources = {
        ("unit", "FAST"): _workspace_row(1.0, [10, 10], [("a" * 64, 990), ("a" * 64, 990)]),
        ("unit", "SLOW"): _workspace_row(5.0, [10, 10], [("b" * 64, 10), ("b" * 64, 10)])}
    unbudgeted = solve_runtime_frontier({"unit": [fast, slow]}, resources,
                                        max_memory_bytes=10 ** 6, max_prefill_ms=10.0)
    assert [allocation.assignment["unit"] for allocation in unbudgeted][0] == "FAST"
    budgeted = solve_runtime_frontier({"unit": [fast, slow]}, resources,
                                      max_memory_bytes=10 ** 6, max_prefill_ms=10.0,
                                      rank_devices=_admitted_bounds([1000, 1000], [1, 1]))
    assert [allocation.assignment["unit"] for allocation in budgeted] == ["SLOW"]
    assert budgeted[0].rank_workspace_bytes == (10, 10)


def test_dominance_is_compared_only_inside_one_workspace_identity_set():
    """A later unit can add an identity one prefix already holds.

    ``FAST`` holds ``X`` (100 B) and is numerically smaller than ``SLOW``,
    which holds ``Y`` (50 B). A second unit then adds ``Z`` (900 B) to
    whichever prefix survives. Comparing the two prefixes before that fold
    prunes ``SLOW``, whose 950 B total is the one that fits; comparing only
    inside an identical identity set keeps both until the budget is read.
    """
    from prismaquant.allocator_solver import Candidate, solve_runtime_frontier

    fast = Candidate("FAST", 4.0, 1000, 1.0)
    slow = Candidate("SLOW", 4.0, 1000, 2.0)
    tail = Candidate("TAIL", 4.0, 1000, 1.0)
    resources = {
        ("a", "FAST"): _workspace_row(1.0, [10, 10], [("1" * 64, 100), ("1" * 64, 100)]),
        ("a", "SLOW"): _workspace_row(5.0, [10, 10], [("2" * 64, 50), ("2" * 64, 50)]),
        ("b", "TAIL"): _workspace_row(1.0, [10, 10], [("3" * 64, 900), ("3" * 64, 900)])}
    frontier = solve_runtime_frontier({"a": [fast, slow], "b": [tail]}, resources,
                                      max_memory_bytes=10 ** 6, max_prefill_ms=10.0,
                                      rank_devices=_admitted_bounds([1000, 1000], [1, 1]))
    assert [allocation.assignment["a"] for allocation in frontier] == ["SLOW"], (
        "the feasible prefix survives: 10 + 50 + 900 + 1 fits, and the pruned one "
        "(10 + 100 + 900 + 1) does not")
    assert frontier[0].rank_workspace_bytes == (950, 950)


def test_one_workspace_identity_is_charged_once_inside_the_fold():
    """Two units viewing one frozen allocation charge those bytes once."""
    from prismaquant.allocator_solver import Candidate, solve_runtime_frontier

    first = Candidate("FIRST", 4.0, 1000, 1.0)
    second = Candidate("SECOND", 4.0, 1000, 1.0)
    shared = [("5" * 64, 900), ("5" * 64, 900)]
    resources = {("a", "FIRST"): _workspace_row(1.0, [10, 10], shared),
                 ("b", "SECOND"): _workspace_row(1.0, [10, 10], shared)}
    frontier = solve_runtime_frontier({"a": [first], "b": [second]}, resources,
                                      max_memory_bytes=10 ** 6, max_prefill_ms=10.0,
                                      rank_devices=_admitted_bounds([1000, 1000], [1, 1]))
    assert len(frontier) == 1
    assert frontier[0].rank_workspace_bytes == (900, 900), (
        "one identity is one allocation per rank, not one per row that views it")


def test_one_workspace_identity_at_two_sizes_refuses_inside_the_fold():
    from prismaquant.allocator_solver import Candidate, solve_runtime_frontier

    first = Candidate("FIRST", 4.0, 1000, 1.0)
    second = Candidate("SECOND", 4.0, 1000, 1.0)
    resources = {
        ("a", "FIRST"): _workspace_row(1.0, [10, 10], [("5" * 64, 900), ("5" * 64, 900)]),
        ("b", "SECOND"): _workspace_row(1.0, [10, 10], [("5" * 64, 800), ("5" * 64, 800)])}
    with pytest.raises(ValueError, match="one frozen allocation cannot be two"):
        solve_runtime_frontier({"a": [first], "b": [second]}, resources,
                               max_memory_bytes=10 ** 6, max_prefill_ms=10.0)


def test_a_duck_typed_recomputation_is_refused_by_name():
    """Carrying the attribute names a verdict has is not the verdict.

    ``getattr`` on an arbitrary object accepts a ``SimpleNamespace`` with the
    two attributes ``RankFixedCharge`` happens to have, and a charge built from
    one would reach a placement with no recomputation behind it.
    """
    impostor = SimpleNamespace(
        world_size=2, charge_per_rank=(700, 900),
        per_rank_terms=({"fixed_resident": 700}, {"fixed_resident": 900}),
        evidence={"full_engine_report": None, "per_rank_partition": True},
        partition_sha256="d" * 64)
    with pytest.raises(RuntimePriceError, match="RankFixedCharge"):
        RankDeviceBounds.recomputed(world_size=2, budgets_per_rank=[10 ** 6, 10 ** 6],
                                    charge_per_rank=(700, 900), recomputation=impostor)
    with pytest.raises(RuntimePriceError, match="RankFixedCharge"):
        RankDeviceBounds.recomputed(world_size=2, budgets_per_rank=[10 ** 6, 10 ** 6],
                                    charge_per_rank=(700, 900), recomputation=None)


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


def test_the_reported_dimensions_name_the_axes_the_search_actually_has():
    """A rank coordinate is never labelled as the scalar device term.

    The ranked byte axes start at the same index the scalar device axis does, so
    a diagnostic that sliced the combined vector labelled rank 0's residency
    ``resident_bytes`` and then appended every rank's names -- one axis too many,
    and the wrong one. Without a decode or a scalar device constraint the vector
    is memory/dloss/prefill plus three coordinates per rank, and that is what the
    report must say.
    """
    from prismaquant.allocator_solver import Candidate, solve_runtime_frontier

    diag = {}
    solve_runtime_frontier({"unit": [Candidate("FAST", 4.0, 1000, 1.0)]},
                           {("unit", "FAST"): _ranked_row(1.0, [10, 900])},
                           max_memory_bytes=10 ** 6, max_prefill_ms=10.0, diagnostics=diag)
    assert diag["dimensions"] == [
        "memory_bytes", "predicted_dloss", "prefill_ms",
        "rank0_resident_bytes", "rank0_peak_scratch_bytes", "rank0_activation_bytes",
        "rank1_resident_bytes", "rank1_peak_scratch_bytes", "rank1_activation_bytes"]


@pytest.mark.parametrize("world_size,provenance,charge,evidence,diagnostic", [
    (2, "pending_measurement", (1, 1), None, "carries no value and no evidence"),
    (2, "recomputed_full_engine_partition", (1, 1), None, "recomputed full-engine"),
    (2, "recomputed_full_engine_partition", None, None, "recomputed full-engine"),
    (2, "recomputed_full_engine_partition", (0, 5), None, "recomputed full-engine"),
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
    with pytest.raises(RuntimePriceError, match="recomputed full-engine"):
        RankDeviceBounds.from_dict(admitted.as_dict())


def _sealed_per_rank_partition(tmp_path, *, startup=(1024, 4096), world=2,
                               runtime_manifest_sha256="a" * 64, model_sha256=None,
                               workload_sha256=None, **overrides):
    """One rank-scoped capture per rank, and the partition that binds them.

    Each rank's own report carries its own startup bytes, so the two ranks'
    recomputed charges differ: a partition that merely redistributed one world
    total between them could not match both, which is the defect this fixture
    exists to expose.
    """
    from prismaquant.full_engine_resource_report import consume_full_engine_resource_report
    from prismaquant.runtime_provenance import FIXED_TERM_FIELDS
    from test_full_engine_resource_report import fixed_terms_report, written

    models = model_sha256 or ("a" * 64,) * world
    workloads = workload_sha256 or ("a" * 64,) * world
    ranks = []
    for rank in range(world):
        report = fixed_terms_report(rank=rank, world_size=world,
                                    runtime_manifest_sha256=runtime_manifest_sha256,
                                    model_sha256=models[rank],
                                    workload_sha256=workloads[rank],
                                    startup_bytes=startup[rank])
        reference = written(tmp_path, report, f"rank{rank}.report.json")
        terms = consume_full_engine_resource_report(reference, root=tmp_path).recomputed_terms
        ranks.append({"rank": rank, "world_size": world,
                      "runtime_manifest_sha256": runtime_manifest_sha256,
                      "capture_sha256": report["identity"]["capture_sha256"],
                      "report": reference,
                      "terms": {term: terms[term] for term in FIXED_TERM_FIELDS}})
    partition = {"schema": "prismaquant.full_engine_rank_partition.v1", "world_size": world,
                 "rule": ("sealed_per_rank_captures_recomputed_terms_with_optional_whole_engine"
                          "_cross_check"),
                 "full_engine_report": None, "ranks": ranks}
    partition.update(overrides)
    return written(tmp_path, partition, "partition.json"), partition


def test_a_recomputed_charge_admits_and_a_supplied_one_is_refused():
    """The bounds are built from a recomputation, and the recomputation decides.

    ``RankDeviceBounds.recomputed`` compares every value with the verdict rather
    than accepting a charge beside the claim, so a caller cannot hand in a
    number -- or a budget vector -- the recomputation did not produce.
    """
    from prismaquant.runtime_provenance import RankFixedCharge

    verdict = RankFixedCharge(
        world_size=2, charge_per_rank=(700, 900),
        per_rank_terms=({"fixed_resident": 700}, {"fixed_resident": 900}),
        evidence={"full_engine_report": {"path": "report.json", "sha256": "e" * 64},
                  "per_rank_partition": True}, partition_sha256="d" * 64)
    bounds = RankDeviceBounds.recomputed(world_size=2, budgets_per_rank=[10 ** 6, 10 ** 6],
                                         charge_per_rank=(700, 900), recomputation=verdict)
    assert bounds.admits_ranks is True
    assert bounds.per_rank_headroom() == (10 ** 6 - 700, 10 ** 6 - 900)
    with pytest.raises(RuntimePriceError, match="not the recomputed one"):
        RankDeviceBounds.recomputed(world_size=2, budgets_per_rank=[10 ** 6, 10 ** 6],
                                    charge_per_rank=(700, 901), recomputation=verdict)
    with pytest.raises(RuntimePriceError, match="where the recomputation covers 2"):
        RankDeviceBounds.recomputed(world_size=1, budgets_per_rank=[10 ** 6],
                                    charge_per_rank=(700,), recomputation=verdict)


def test_each_rank_charge_is_recomputed_from_that_ranks_own_capture(tmp_path):
    """Two rank-scoped captures, two charges, neither read from the other.

    This is the mechanism the ranked device axis waits on: every rank's four
    fixed terms come from that rank's own sealed report, whose run identity
    names the rank, the world and the runtime. The numbers here are synthetic
    and the reports say so; what the test establishes is the binding.
    """
    from prismaquant.runtime_provenance import recompute_rank_fixed_charge

    reference, partition = _sealed_per_rank_partition(tmp_path)
    charge = recompute_rank_fixed_charge(reference, root=tmp_path)
    assert charge.world_size == 2
    assert charge.charge_per_rank[0] != charge.charge_per_rank[1], (
        "each rank's charge is its own capture's recomputation, so two captures "
        "with different startup bytes cannot produce one world number")
    assert charge.per_rank_terms[0]["fixed_resident"] == 4096 + 1024
    assert charge.per_rank_terms[1]["fixed_resident"] == 4096 + 4096
    assert charge.evidence["per_rank_partition"] is True
    assert charge.evidence["full_engine_report"] is None, (
        "the whole-engine cross-check is optional and was not supplied")
    assert partition["ranks"][0]["report"] != partition["ranks"][1]["report"]


def test_a_redistributed_partition_is_refused_term_by_term(tmp_path):
    """Moving a rank's numbers to its peer keeps the sum and changes the rank.

    The old contract compared only the per-term sum against one scalar report,
    so this exact document passed: every term still summed to the world's own
    value while each rank held a number its own capture never observed.
    """
    from prismaquant.runtime_provenance import recompute_rank_fixed_charge
    from test_full_engine_resource_report import written

    reference, partition = _sealed_per_rank_partition(tmp_path)
    assert partition["ranks"][0]["terms"] != partition["ranks"][1]["terms"]
    redistributed = copy.deepcopy(partition)
    redistributed["ranks"][0]["terms"] = copy.deepcopy(partition["ranks"][1]["terms"])
    redistributed["ranks"][1]["terms"] = copy.deepcopy(partition["ranks"][0]["terms"])
    with pytest.raises(RuntimePriceError, match="may not restate it"):
        recompute_rank_fixed_charge(written(tmp_path, redistributed, "moved.json"), root=tmp_path)
    # The same numbers summed to the same world total, which is why the sum
    # alone could never have caught it.
    for term in partition["ranks"][0]["terms"]:
        assert (partition["ranks"][0]["terms"][term] + partition["ranks"][1]["terms"][term]
                == redistributed["ranks"][0]["terms"][term]
                + redistributed["ranks"][1]["terms"][term])


def test_a_partition_that_moves_a_report_its_rank_or_its_world_is_refused(tmp_path):
    from prismaquant.runtime_provenance import recompute_rank_fixed_charge
    from test_full_engine_resource_report import written

    _reference, partition = _sealed_per_rank_partition(tmp_path)
    forged = copy.deepcopy(partition)
    forged["ranks"][0]["report"] = dict(partition["ranks"][0]["report"], sha256="0" * 64)
    with pytest.raises(RuntimePriceError):
        recompute_rank_fixed_charge(written(tmp_path, forged, "forged.json"), root=tmp_path)
    wrong_rule = copy.deepcopy(partition)
    wrong_rule["rule"] = "sum_of_whatever_the_caller_says"
    with pytest.raises(RuntimePriceError, match="partition rule"):
        recompute_rank_fixed_charge(written(tmp_path, wrong_rule, "rule.json"), root=tmp_path)
    omitted = copy.deepcopy(partition)
    omitted["ranks"] = omitted["ranks"][:1]
    with pytest.raises(RuntimePriceError, match="one record per rank"):
        recompute_rank_fixed_charge(written(tmp_path, omitted, "short.json"), root=tmp_path)
    peer = copy.deepcopy(partition)
    peer["ranks"][1]["report"] = copy.deepcopy(partition["ranks"][0]["report"])
    with pytest.raises(RuntimePriceError, match="report rank"):
        recompute_rank_fixed_charge(written(tmp_path, peer, "peer.json"), root=tmp_path)
    foreign_runtime = copy.deepcopy(partition)
    foreign_runtime["ranks"][1]["runtime_manifest_sha256"] = "c" * 64
    with pytest.raises(RuntimePriceError, match="runtime manifest"):
        recompute_rank_fixed_charge(written(tmp_path, foreign_runtime, "runtime.json"), root=tmp_path)
    stale_capture = copy.deepcopy(partition)
    stale_capture["ranks"][0]["capture_sha256"] = "d" * 64
    with pytest.raises(RuntimePriceError, match="capture digest"):
        recompute_rank_fixed_charge(written(tmp_path, stale_capture, "capture.json"), root=tmp_path)


def test_a_rank_whose_own_capture_cannot_recompute_a_term_is_refused_by_name(tmp_path):
    """A capture that observed no fixed terms charges nothing, rather than zero."""
    from prismaquant.runtime_provenance import recompute_rank_fixed_charge
    from test_full_engine_resource_report import fixed_terms_report, written

    _reference, partition = _sealed_per_rank_partition(tmp_path)
    blind = copy.deepcopy(partition)
    blind["ranks"][1]["report"] = written(
        tmp_path, fixed_terms_report(rank=1, world_size=2, startup=False), "blind.json")
    with pytest.raises(RuntimePriceError, match="recomputes no fixed_"):
        recompute_rank_fixed_charge(written(tmp_path, blind, "blind-partition.json"), root=tmp_path)


@pytest.mark.parametrize("key", ["model_sha256", "workload_sha256"])
def test_two_ranks_of_one_world_cannot_be_two_models_or_workloads(tmp_path, key):
    """One shared runtime digest is not one world.

    Rank 0 could observe a small model and rank 1 a different one, or the same
    model under a different workload, and both reports would still name the
    same runtime, rank and world size. Every world coordinate is therefore
    joined across the ranks rather than checked per rank.
    """
    from prismaquant.runtime_provenance import recompute_rank_fixed_charge

    reference, _partition = _sealed_per_rank_partition(
        tmp_path, **{key: ("a" * 64, "b" * 64)})
    with pytest.raises(RuntimePriceError, match=f"rank 1 names {key}"):
        recompute_rank_fixed_charge(reference, root=tmp_path)


def test_the_charge_is_bound_to_the_bytes_the_table_was_priced_against(tmp_path):
    """The consumer's own identities decide, not the partition's own word."""
    from prismaquant.runtime_provenance import recompute_rank_fixed_charge

    reference, _partition = _sealed_per_rank_partition(tmp_path)
    charge = recompute_rank_fixed_charge(
        reference, root=tmp_path,
        expected_run_identity={"model_sha256": "a" * 64, "runtime_manifest_sha256": "a" * 64})
    assert charge.world_size == 2
    with pytest.raises(RuntimePriceError, match="report model_sha256"):
        recompute_rank_fixed_charge(reference, root=tmp_path,
                                    expected_run_identity={"model_sha256": "9" * 64})
