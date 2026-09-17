"""The routed-owner row path: per-rank receipts -> one row -> the allocator.

The panel and receipt skeleton are the synthetic routed-owner fixtures the
native MoE suite already uses; the per-rank fields added here are the ones the
producer's whole-owner receipt publishes (`resources.rank`/`world_size`/
`peers`/`self.bound_sha256`, `latency_scope`, `runtime.collective`). Nothing
here is a measurement and no GPU ran: the point is which records the consumer
accepts, which it refuses by name, and that a row it writes reaches the
allocator's own aggregate/expand path and its own verdict.

The last two tests re-price the existing dense CLI fixture's rows in the
per-rank spelling. Those units are not routed owners, deliberately: what they
exercise is the allocator pricing a ranked row and reporting per-rank terms
with no scalar device total, which is the half of the bridge this file does
not cover from the receipt side.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant import native_receipt_table as emitter
from prismaquant.measured_runtime_prices import (
    PROVENANCE_CONTEXT_SCHEMA, PROVENANCE_IDENTITY_KIND, RANK_RESOURCES_SCHEMA,
    RANK_TIMING_RULE, MeasuredRuntimeRow, OperatorMeasurement, RuntimeBinding,
    RuntimePriceError, RuntimeRankResources, identity_sha256, parse_row_resources,
    parse_runtime_context,
)
from prismaquant.native_moe_panel import FORMAT
from prismaquant.native_operator_panel import operator_route_identity
from prismaquant.production_weight_cache import _cb_cache_tensor_identity as tensor_id
from prismaquant.runtime_provenance import (
    LATENCY_SCOPE_KIND, RUNTIME_COLLECTIVE_OP, RUNTIME_COLLECTIVE_SITE, admit_native_rows,
    routed_rank_bound,
)
from test_native_moe_panel import joined, receipt_fixture  # noqa: F401 - fixtures

WHOLE_UNIT = "model.layers.2.feed_forward.experts"
WIRE_BYTES = 42          # the fixture's per-member wire bytes
MEMBERS = 96             # 32 experts x 3 roles
WHOLE_SERIALIZED = WIRE_BYTES * MEMBERS
WORKSPACE_BYTES = 64
FAST = [3.0, 1.0, 2.0]   # median 2.0
SLOW = [5.0, 4.0, 6.0]   # median 5.0
#: The producer source-tree seal this suite's synthetic relation publishes, and
#: the one every member's wire record is stamped with before the panel is
#: frozen -- the intake gate refuses an installed-package seal.
WIRE_SEAL = "7" * 64


def _collective(world_size, **updates):
    payload = {"op": RUNTIME_COLLECTIVE_OP, "site": RUNTIME_COLLECTIVE_SITE,
               "required_by_this_owner": world_size > 1,
               "runtime_declares_skip_final_all_reduce": False, "world_size": world_size}
    payload.update(updates)
    return payload


def _scope(world_size, **updates):
    """The producer's own `latency_scope`, counts included.

    `collective_calls_per_phase` is what the producer counted at the runner's
    imported symbol during each priced apply: once per phase at a world above
    one, never at a world of one. `includes_output_collective` is derived from
    those counts, not from the config's intent.
    """
    payload = {"kind": LATENCY_SCOPE_KIND, "per_rank": True,
               "includes_output_collective": world_size > 1,
               "collective_callsite": RUNTIME_COLLECTIVE_SITE,
               "collective_calls_per_phase": {phase: 1 if world_size > 1 else 0
                                               for phase in ("prefill", "decode")},
               "collective_evidence": "counted at the runtime runner's own imported symbol "
                                      "during the priced apply, and required to be 1 at a "
                                      "world above one and 0 at a world of one",
               "collective": RUNTIME_COLLECTIVE_OP,
               "never": "a sum of leaf timings, or a local matmul priced as the module"}
    payload.update(updates)
    return payload


def _rank_record(resources):
    return {"rank": resources["rank"], "world_size": resources["world_size"],
            "bound_sha256": routed_rank_bound(resources)}


def _routed_receipts(cell, *, world_size, samples=None, residents=None, scratch=None,
                     collective=None, scopes=None, receipt_mutation=None):
    """One producer-shaped receipt per rank, with the gathered bounds attached.

    ``samples`` is one sample list per rank; ``resident_bytes`` and the phase
    bounds are that rank's own, exactly as the producer observes them, and the
    peers' ``bound_sha256`` is computed from the other ranks' own records --
    the gather the producer performs before it may time anything.
    """
    panel, receipt, trace = receipt_fixture(cell, complete=True)
    if receipt_mutation is not None:
        receipt_mutation(receipt, panel)
    samples = samples or [FAST] * world_size
    residents = residents or [100 + rank for rank in range(world_size)]
    scratch = scratch or [128] * world_size
    if world_size > 1:
        panel["execution"]["tensor_parallel"] = world_size
        panel["runtime"]["execution"]["tensor_parallel"] = world_size
    # The runtime's own collective is a coordinate of the frozen panel, so it
    # is set once, before either digest is taken: a receipt may not carry a
    # different runtime than the panel it was frozen against.
    panel["runtime"]["collective"] = collective or _collective(world_size)
    receipts = []
    for rank in range(world_size):
        rank_receipt = copy.deepcopy(receipt)
        rank_receipt["runtime"] = panel["runtime"]
        rank_receipt["runtime_sha256"] = identity_sha256(panel["runtime"])
        rank_receipt["panel"] = panel
        rank_receipt["panel_sha256"] = identity_sha256(panel)
        phase_records = {}
        for phase in ("prefill", "decode"):
            bound = {"status": "complete_operator_bound",
                     "composition": "sum_of_independent_peaks_including_output",
                     "full_model_fixed_resources_complete": False,
                     "peak_scratch_bytes": scratch[rank],
                     "external_native_peak_bytes": scratch[rank] // 2,
                     "torch_peak_increment_bytes": scratch[rank] - scratch[rank] // 2}
            phase_records[phase] = {"bound": bound, "torch_observation": {"fixture": True}}
        resources = {"status": "complete_operator_bound", "scope": "torch_allocator_observation",
                     "rank": rank, "world_size": world_size, "peers": None,
                     "resident_bytes": residents[rank], "workspace_resident_bytes": WORKSPACE_BYTES,
                     "workspace_sha256": panel["workspace_sha256"],
                     "trace_sha256": identity_sha256(trace),
                     "phases": phase_records}
        rank_receipt["resources"] = resources
        rank_receipt["latency_scope"] = _scope(world_size) if scopes is None else scopes[rank]
        for phase in ("prefill", "decode"):
            rank_receipt["phases"][phase]["measurement"] = {
                "method": "cuda_events", "sample_unit": "single_apply",
                "samples_ms": list(samples[rank]), "warmup_iterations": 4}
        receipts.append(rank_receipt)
    bounds = {rank: routed_rank_bound(rank_receipt["resources"])
              for rank, rank_receipt in enumerate(receipts)}
    for rank, rank_receipt in enumerate(receipts):
        resources = rank_receipt["resources"]
        resources["self"] = {"rank": rank, "world_size": world_size, "bound_sha256": bounds[rank]}
        resources["peers"] = [{"rank": other, "world_size": world_size,
                               "bound_sha256": bounds[other]}
                              for other in range(world_size) if other != rank]
    return panel, receipts, trace


def _write(joined_cell, tmp_path, *, world_size=1, **overrides):
    """Write the panel, one file per rank's receipt, and the trace."""
    panel, receipts, trace = _routed_receipts(joined_cell, world_size=world_size, **overrides)
    panel_path = tmp_path / "routed.panel.json"
    panel_path.write_text(json.dumps(panel, sort_keys=True))
    trace_path = tmp_path / "routed.trace.json"
    trace_path.write_text(json.dumps(trace, sort_keys=True))
    spec = {"unit": panel["unit"], "format": panel["format"], "run_id": "native",
            "panel": panel_path.name, "receipt": None, "memory_trace": trace_path.name}
    for rank, receipt in enumerate(receipts):
        path = tmp_path / f"routed.receipt.{rank}.json"
        path.write_text(json.dumps(receipt, sort_keys=True))
        if rank == 0:
            spec["receipt"] = path.name
        else:
            spec.setdefault("peer_receipts", []).append(
                {"rank": rank, "receipt": path.name, "memory_trace": trace_path.name})
    _inputs, _preflight, rows = joined_cell
    cost = {"costs": {name: {FORMAT: row} for name, row in rows.items()}}
    return spec, panel, cost, receipts


@pytest.mark.parametrize("world_size", [1, 2])
def test_a_routed_owner_becomes_one_atomic_member_assignment_row(joined, tmp_path, world_size):
    spec, panel, cost, _receipts = _write(joined, tmp_path, world_size=world_size)
    item = emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                       manifest_dir=tmp_path, table_dir=tmp_path)
    row = item["row"]
    resources = row["resources"]
    assert row["unit"] == WHOLE_UNIT and row["format"] == FORMAT
    assert len(row["binding"]["member_formats"]) == MEMBERS
    assert row["binding"]["operator_route"] == operator_route_identity(
        panel["phases"]["prefill"]["expected_route"])
    assert resources["schema"] == RANK_RESOURCES_SCHEMA
    assert resources["world_size"] == world_size
    assert resources["timing_rule"] == RANK_TIMING_RULE
    assert [entry["rank"] for entry in resources["ranks"]] == list(range(world_size))
    assert [entry["resident_bytes"] for entry in resources["ranks"]] == list(range(100, 100 + world_size))
    assert resources["wire_bytes"] == WHOLE_SERIALIZED, (
        "the module's wire extent is charged once, not once per rank that views it")
    assert resources["wire_sha256"]
    assert resources["ranks"][0]["bound_sha256"] == routed_rank_bound(
        _receipts[0]["resources"])


def test_the_world_row_prices_the_slowest_ranks_median_and_keeps_every_rank(joined, tmp_path):
    spec, _panel, cost, _receipts = _write(joined, tmp_path, world_size=2,
                                           samples=[FAST, SLOW])
    item = emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                       manifest_dir=tmp_path, table_dir=tmp_path)
    resources = item["row"]["resources"]
    assert resources["rank_medians_ms"]["prefill"] == [2.0, 5.0]
    assert resources["prefill_ms"] == 5.0, "one whole-apply interval, bounded by its slowest rank"
    assert item["row"]["prefill"]["samples_ms"] == SLOW
    assert item["row"]["prefill"]["receipt_path"].endswith("routed.receipt.1.json")
    assert "peak_scratch_bytes" not in resources, "no scalar reduction is published"


def test_every_rank_own_bound_is_checked_against_the_other_ranks_view(joined, tmp_path):
    spec, _panel, cost, receipts = _write(joined, tmp_path, world_size=2)
    path = tmp_path / spec["receipt"]
    tampered = json.loads(path.read_text())
    assert tampered["resources"]["peers"][0]["bound_sha256"] == routed_rank_bound(
        receipts[1]["resources"])
    tampered["resources"]["peers"][0]["bound_sha256"] = "0" * 64
    path.write_text(json.dumps(tampered, sort_keys=True))
    with pytest.raises(RuntimePriceError, match="gathered peer bounds"):
        emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                    manifest_dir=tmp_path, table_dir=tmp_path)


def test_a_world_above_one_needs_every_ranks_receipt(joined, tmp_path):
    spec, _panel, cost, _receipts = _write(joined, tmp_path, world_size=2)
    spec.pop("peer_receipts")
    with pytest.raises(RuntimePriceError, match="every rank's own receipt"):
        emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                    manifest_dir=tmp_path, table_dir=tmp_path)


def test_a_peer_receipt_that_declares_the_wrong_rank_is_refused(joined, tmp_path):
    spec, _panel, cost, _receipts = _write(joined, tmp_path, world_size=2)
    spec["peer_receipts"][0]["rank"] = 0
    with pytest.raises(RuntimePriceError, match="declares rank 1, not 0"):
        emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                    manifest_dir=tmp_path, table_dir=tmp_path)


@pytest.mark.parametrize("scope_updates,diagnostic", [
    ({"per_rank": False}, "one whole-owner apply measured per rank"),
    ({"includes_output_collective": False}, "includes_output_collective"),
    ({"collective": "some_other_op"}, "latency scope collective"),
    ({"collective_callsite": "vllm.fused_moe.runner.moe_runner:_maybe_reduce_final_output"},
     "latency scope callsite"),
    ({"collective_calls_per_phase": {"prefill": 0, "decode": 0}},
     "priced the runtime's own output reduction 0 time"),
    ({"collective_calls_per_phase": {"prefill": 1}}, "once per priced phase"),
    ({"collective_evidence": ""}, "collective evidence"),
])
def test_a_latency_scope_that_is_not_this_worlds_apply_is_refused(joined, tmp_path,
                                                                  scope_updates, diagnostic):
    scopes = [_scope(2, **scope_updates), _scope(2, **scope_updates)]
    spec, _panel, cost, _receipts = _write(joined, tmp_path, world_size=2, scopes=scopes)
    with pytest.raises(RuntimePriceError, match=diagnostic):
        emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                    manifest_dir=tmp_path, table_dir=tmp_path)


@pytest.mark.parametrize("updates,diagnostic", [
    ({"op": "some_other_op"}, "own final"),
    ({"required_by_this_owner": False}, "own final"),
    ({"runtime_declares_skip_final_all_reduce": True}, "own final"),
])
def test_a_runtime_whose_collective_is_not_this_worlds_reduction_is_refused(
        joined, tmp_path, updates, diagnostic):
    spec, _panel, cost, _receipts = _write(joined, tmp_path, world_size=2,
                                           collective=_collective(2, **updates))
    with pytest.raises(RuntimePriceError, match=diagnostic):
        emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                    manifest_dir=tmp_path, table_dir=tmp_path)


def test_a_runtime_that_prices_another_world_is_refused(joined, tmp_path):
    """The collective's own world is a coordinate, not a copy of the panel's."""
    spec, _panel, cost, _receipts = _write(joined, tmp_path, world_size=2,
                                           collective=_collective(1))
    with pytest.raises(RuntimePriceError, match="own final"):
        emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                    manifest_dir=tmp_path, table_dir=tmp_path)


@pytest.mark.parametrize("world_size,calls,diagnostic", [
    (2, 0, "priced the runtime's own output reduction 0 time"),
    (1, 1, "priced the runtime's own output reduction 1 time"),
    (2, None, "priced the runtime's own output reduction None time"),
])
def test_the_count_of_the_runtimes_own_reduction_decides_the_claim(
        joined, tmp_path, world_size, calls, diagnostic):
    """A declaration is not a count: the sampled region must have reduced.

    The producer's own two-arm TP2 run priced its partial sum before the runner
    seam was called, which is exactly the receipt this refuses -- the timed
    region held no reduction, so its samples do not price the module.
    """
    counted = _scope(world_size, collective_calls_per_phase={"prefill": calls, "decode": calls})
    scopes = [counted] * world_size
    spec, _panel, cost, _receipts = _write(joined, tmp_path, world_size=world_size, scopes=scopes)
    with pytest.raises(RuntimePriceError, match=diagnostic):
        emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                    manifest_dir=tmp_path, table_dir=tmp_path)


@pytest.mark.parametrize("missing,diagnostic", [
    ("self", "every rank's own bound"),
    ("peers", "every rank's own bound"),
    ("world_size", "every rank's own bound"),
    ("rank", "declares no rank of its own"),
])
def test_a_receipt_that_omits_its_own_rank_identity_is_refused(joined, tmp_path, missing, diagnostic):
    spec, _panel, cost, _receipts = _write(joined, tmp_path, world_size=2)
    path = tmp_path / spec["receipt"]
    receipt = json.loads(path.read_text())
    del receipt["resources"][missing]
    path.write_text(json.dumps(receipt, sort_keys=True))
    with pytest.raises(RuntimePriceError, match=diagnostic):
        emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                    manifest_dir=tmp_path, table_dir=tmp_path)


def test_a_single_rank_receipt_still_carries_its_own_identity_and_no_peers(joined, tmp_path):
    spec, _panel, cost, receipts = _write(joined, tmp_path, world_size=1)
    assert receipts[0]["resources"]["peers"] == []
    assert receipts[0]["resources"]["self"]["bound_sha256"] == routed_rank_bound(
        receipts[0]["resources"])
    item = emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                       manifest_dir=tmp_path, table_dir=tmp_path)
    assert "peer_receipts" not in item["binding"]
    assert item["row"]["resources"]["world_size"] == 1
    assert item["row"]["resources"]["rank_medians_ms"]["prefill"] == [2.0]


def test_a_routed_member_costs_more_than_its_own_probe_identity_binds(joined, tmp_path):
    """Every member is joined to its own joint AURA row, not to a summary."""
    spec, _panel, cost, _receipts = _write(joined, tmp_path)
    members = sorted(cost["costs"])
    cost["costs"][members[0]][FORMAT] = json.loads(json.dumps(cost["costs"][members[1]][FORMAT]))
    with pytest.raises(RuntimePriceError, match="joint operator identity differs"):
        emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                    manifest_dir=tmp_path, table_dir=tmp_path)


def test_a_routed_member_keeps_its_own_rank_local_shape(joined, tmp_path):
    spec, _panel, cost, _receipts = _write(joined, tmp_path, world_size=2)
    item = emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                       manifest_dir=tmp_path, table_dir=tmp_path)
    shapes = item["row"]["binding"]["member_shapes"]
    assert shapes[f"{WHOLE_UNIT}.0.w1"] == [3, 4]
    assert shapes[f"{WHOLE_UNIT}.0.w2"] == [4, 3]


def test_the_context_of_a_routed_table_names_the_structure(joined):
    panel, _receipt, _trace = receipt_fixture(joined, complete=True)
    panel["runtime"]["gpu"] = {"uuid": "synthetic-gpu", "capability": [12, 1]}
    panel["runtime"]["native_libraries"] = {}
    panel["phases"]["decode"]["m"] = 1
    context = emitter.derive_context([panel], relation={"synthetic": True})
    assert context["serving_context"]["structure"] == "routed_moe"
    assert context["tensor_parallel"] == 1
    dense_lookalike = {"schema": emitter.DENSE_PANEL_SCHEMA, "runtime": panel["runtime"]}
    with pytest.raises(RuntimePriceError, match="more than one structure"):
        emitter.derive_context([panel, dense_lookalike], relation={"synthetic": True})


def _per_rank_table(tmp_path):
    """The dense CLI fixture's rows, re-priced in the per-rank spelling."""
    from test_allocator_measured_runtime_cli import _main_fixture

    name, argv = _main_fixture(tmp_path)
    table_path = tmp_path / "runtime.json"
    table = json.loads(table_path.read_text())
    for entry in table["rows"]:
        scalar = entry["resources"]
        entry["resources"] = {"schema": RANK_RESOURCES_SCHEMA, "world_size": 1,
                              "timing_rule": RANK_TIMING_RULE,
                              "prefill_ms": scalar["prefill_ms"],
                              "decode_ms": scalar["decode_ms"],
                              "rank_medians_ms": {"prefill": [scalar["prefill_ms"]],
                                                  "decode": [scalar["decode_ms"]]},
                              "wire_bytes": scalar["serialized_bytes"],
                              "wire_sha256": "f" * 64,
                              "ranks": [{"rank": 0,
                                         "resident_bytes": scalar["resident_bytes"],
                                         "peak_scratch_bytes": scalar["peak_scratch_bytes"],
                                         "activation_bytes": scalar["activation_bytes"],
                                         "workspace_resident_bytes": 0,
                                         "workspace_sha256": "d" * 64,
                                         "bound_sha256": "e" * 64}]}
    table_path.write_text(json.dumps(table))
    return name, argv


def test_the_allocator_prices_a_per_rank_table_and_publishes_no_device_total(tmp_path, monkeypatch):
    name, argv = _per_rank_table(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    from prismaquant import allocator
    from prismaquant.layer_config import load_assignment

    allocator.main()
    assert load_assignment(tmp_path / "layer.json")[name] == "FP8_E5M2"
    verdict = json.loads((tmp_path / "layer.json").read_text())["__prismaquant__"]["serve_constraints"]
    assert verdict["per_rank_resources"] is True
    assert verdict["device_memory_total_published"] is False
    ranked = verdict["coverage"]["memory"]["operator_rank_totals"]
    assert ranked["world_size"] == 1
    assert ranked["ranks"][0]["resident_bytes"] == 16384
    assert verdict["predicted"]["device_memory_bytes"] is None
    assert verdict["predicted"]["operator_sum_prefill_ms"] == 2
    assert verdict["certifies_p95"] is False


def test_the_allocator_refuses_a_scalar_device_budget_for_a_per_rank_table(tmp_path, monkeypatch):
    _name, argv = _per_rank_table(tmp_path)
    monkeypatch.setattr(sys, "argv", argv + ["--serve-device-budget-bytes", "1000000"])
    from prismaquant import allocator

    with pytest.raises(SystemExit, match="prices per-rank resources"):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()


def _one_token_phases(*, hidden=4, top_k=2):
    """The routed fixture's phases at the token scope the v2 intake prices.

    The routed fixture prices two-token phases; the native intake has always
    required batch size one and a one-token decode, which is what the
    producer's own request carries. Same tensors, one row.
    """
    raw_ids = torch.ones(1, top_k, dtype=torch.int64)
    raw_weights = torch.full((1, top_k), 1.0 / top_k, dtype=torch.bfloat16)
    values = {"input": torch.ones(1, hidden, dtype=torch.bfloat16), "topk_ids": raw_ids.int(),
              "topk_weights": raw_weights.float(), "source_topk_ids": raw_ids,
              "source_topk_weights": raw_weights}
    transport = {name: {"source": tensor_id(values["source_" + name]), "supplied": tensor_id(values[name]),
                        "operation": "lossless_dtype_conversion"} for name in ("topk_ids", "topk_weights")}
    fields = {key: tensor_id(value) for key, value in values.items() if not key.startswith("source_")}
    return {phase: {"m": 1, **fields, "reference_qdq": fields["input"],
                    "reference_output": fields["input"], "transport": copy.deepcopy(transport)}
            for phase in ("prefill", "decode")}


def _fake_identity(name, dims, dtype="torch.bfloat16"):
    """A tensor identity without the tensor: the contract reads the record.

    A GLM roster's rank-local widths are 1024x4096 over 864 members, and the
    contract compares published identities, so allocating those bytes would
    cost the runner tens of GiB to prove a name.
    """
    itemsize = 4 if dtype == "torch.float32" else 2
    return {"content_sha256": hashlib.sha256(name.encode()).hexdigest(), "shape": list(dims),
            "dtype": dtype, "logical_bytes": itemsize * math.prod(dims)}


def _sealed_cell(joined_cell, *, phases=None):
    """The routed cell as the native intake prices it: one token, sealed wires."""
    inputs, preflight, rows = (copy.deepcopy(part) for part in joined_cell)
    phases = _one_token_phases() if phases is None else phases
    inputs["phases"] = phases
    inputs["routing_capture"]["phases"] = phases
    inputs["routing_capture_sha256"] = identity_sha256(inputs["routing_capture"])
    preflight["operator"]["routing_capture_sha256"] = inputs["routing_capture_sha256"]
    preflight["operator"]["phases"] = {phase: {"transport": copy.deepcopy(phases[phase]["transport"])}
                                       for phase in phases}
    for member in inputs["members"]:
        member["wire"]["record"]["identity"] = {"encoder_source_sha256": WIRE_SEAL}
    sealed = {member["unit"]: member for member in inputs["members"]}
    for entry in preflight["operator"]["members"]:
        entry["wire_record_sha256"] = identity_sha256(sealed[entry["unit"]]["wire"]["record"])
    return inputs, preflight, rows


def _one_token_route(route_shape):
    """What the receipt's own phases report the runtime dispatched, at one token."""
    def mutate(receipt, _panel):
        for phase in ("prefill", "decode"):
            receipt["phases"][phase]["route"]["shape"] = route_shape
    return mutate


def _routed_gate(joined_cell, tmp_path, *, world_size=1, samples=None, tensor_parallel=None,
                 phases=None, route_shape="M1:N6:K4"):
    """An emitted routed row, plus the two inputs the loader's gate reads.

    The relation here is the minimum the intake gate consults for a native row:
    the run it prices, the producer source-tree seal its wires must carry, and
    the serving configuration whose digest the panel publishes. Whether the
    relation is a complete device account is `load_runtime_relation`'s verdict,
    reported by the loader rather than re-derived here.
    """
    cell = _sealed_cell(joined_cell, phases=phases)
    spec, panel, cost, receipts = _write(cell, tmp_path, world_size=world_size, samples=samples,
                                         receipt_mutation=_one_token_route(route_shape))
    item = emitter.bind_native_receipt(spec, cost_payload=cost, cost_sha256="4" * 64,
                                       manifest_dir=tmp_path, table_dir=tmp_path)
    context = parse_runtime_context({
        "schema": PROVENANCE_CONTEXT_SCHEMA, "runtime_identity_kind": PROVENANCE_IDENTITY_KIND,
        "serving_context": {"platform": "sm_121", "structure": "routed_moe", "residency": "resident",
                            "runtime_image": panel["runtime"]["image"], "execution_mode": "eager"},
        "gpu_identity": "synthetic-gpu", "runtime_sha256": "b" * 64,
        "source_sha256": panel["source_sha256"], "calibration_sha256": panel["calibration_sha256"],
        "prompt_tokens": 1, "batch_size": 1,
        "tensor_parallel": world_size if tensor_parallel is None else tensor_parallel,
        "graph_mode": "eager",
        "operator_routes": {panel["unit"]: {panel["format"]: item["row"]["binding"]["operator_route"]}}})
    relation = {"configuration_sha256": panel["serving_config_sha256"], "full_engine_run_id": "engine",
                "runs": {"native": {"raw": panel["runtime"],
                                    "common": {"producer_source_tree_sha256": WIRE_SEAL}}}}
    return item, context, relation, panel, receipts


def _gate_table(item, context, tmp_path, *, resources=None):
    """The row as the loader's intake gate sees it: real files, real bindings."""
    row = copy.deepcopy(item["row"])
    if resources is not None:
        row["resources"] = resources
    parsed = MeasuredRuntimeRow(
        row["unit"], row["format"], RuntimeBinding.from_dict(row["binding"]),
        parse_row_resources(row["resources"], tensor_parallel=context.tensor_parallel,
                            where="routed intake gate fixture"),
        OperatorMeasurement.from_dict(row["prefill"]), OperatorMeasurement.from_dict(row["decode"]))
    return SimpleNamespace(source_path=str(tmp_path / "routed.table.json"), cost_sha256="4" * 64,
                           context=context, rows=(parsed,),
                           native_receipt_bindings=(item["binding"],))


def _bound_path(reference, tmp_path):
    path = Path(reference["path"])
    return path if path.is_absolute() else tmp_path / path


@pytest.mark.parametrize("world_size", [1, 2])
def test_the_intake_gate_re_derives_a_routed_rows_rank_vector(joined, tmp_path, world_size):
    """The row the emitter writes is the row the loader's own gate re-derives.

    This is the gate ``allocator.main`` reaches through
    ``measured_runtime_prices.load_measured_runtime_table``: it rehashes every
    rank's receipt, requires each to carry the frozen panel, re-consumes each
    one, and recomputes the rank vector -- so the ranked row is not a number
    that travelled and was trusted.
    """
    item, context, relation, _panel, _receipts = _routed_gate(joined, tmp_path, world_size=world_size)
    table = _gate_table(item, context, tmp_path)
    admit_native_rows(table, relation)
    (row,) = table.rows
    assert isinstance(row.resources, RuntimeRankResources)
    assert row.resources.world_size == world_size
    assert row.resources.wire_bytes == WHOLE_SERIALIZED, "one canonical container, charged once"
    assert [entry["rank"] for entry in row.resources.as_dict()["ranks"]] == list(range(world_size))
    assert row.resources.prefill_ms == item["row"]["resources"]["prefill_ms"]


def test_the_intake_gate_refuses_a_rank_vector_the_receipts_do_not_support(joined, tmp_path):
    item, context, relation, _panel, _receipts = _routed_gate(joined, tmp_path, world_size=2)
    inflated = copy.deepcopy(item["row"]["resources"])
    inflated["ranks"][1]["resident_bytes"] += 1
    with pytest.raises(RuntimePriceError, match="native row per-rank resources"):
        admit_native_rows(_gate_table(item, context, tmp_path, resources=inflated), relation)


def test_the_intake_gate_refuses_a_world_the_context_does_not_price(joined, tmp_path):
    item, context, relation, _panel, _receipts = _routed_gate(joined, tmp_path, world_size=2,
                                                              tensor_parallel=1)
    with pytest.raises(RuntimePriceError, match="cover a world of 2"):
        admit_native_rows(_gate_table(item, context, tmp_path), relation)


def test_the_intake_gate_refuses_a_peer_receipt_from_another_runtime(joined, tmp_path):
    """A gathered roster is rehashed, so a peer's own panel is re-read."""
    item, context, relation, _panel, _receipts = _routed_gate(joined, tmp_path, world_size=2)
    peer = item["binding"]["peer_receipts"][0]
    path = _bound_path(peer["receipt"], tmp_path)
    receipt = json.loads(path.read_text())
    receipt["runtime"] = dict(receipt["runtime"], image="foreign/image@sha256:" + "a" * 64)
    receipt["panel"]["runtime"] = receipt["runtime"]
    path.write_text(json.dumps(receipt, sort_keys=True))
    peer["receipt"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimePriceError, match="peer receipt panel"):
        admit_native_rows(_gate_table(item, context, tmp_path), relation)


def test_the_intake_gate_refuses_a_peer_receipt_whose_bytes_moved(joined, tmp_path):
    item, context, relation, _panel, _receipts = _routed_gate(joined, tmp_path, world_size=2)
    peer = item["binding"]["peer_receipts"][0]
    path = _bound_path(peer["receipt"], tmp_path)
    receipt = json.loads(path.read_text())
    receipt["resources"]["resident_bytes"] += 1
    path.write_text(json.dumps(receipt, sort_keys=True))
    with pytest.raises(RuntimePriceError):
        admit_native_rows(_gate_table(item, context, tmp_path), relation)


# --------------------------------------------------------------------------
# The main objective's own geometry: GLM-5.3-Flash, 288 experts, top-8, TP2
# --------------------------------------------------------------------------
GLM_UNIT = "model.language_model.layers.3.mlp.experts"
GLM_EXPERTS = 288
GLM_HIDDEN = 4096
GLM_INTERMEDIATE = 2048
GLM_TP = 2
GLM_MEMBERS = GLM_EXPERTS * len(("w1", "w3", "w2"))


def _glm_shape():
    """The captured GLM-5.3-Flash routed stack, at its served TP2 cut."""
    return {"geometry_version": 1, "geometry_id": "glm53_next_routed_stack_v1",
            "source_id": "glm5_next", "n_routed_experts": GLM_EXPERTS, "top_k": 8,
            "hidden_size": GLM_HIDDEN, "intermediate_size": GLM_INTERMEDIATE, "shared_experts": 1,
            "n_group": 1, "topk_group": 1, "topk_method": "noaux_tc", "scoring_func": "sigmoid",
            "norm_topk_prob": True, "routed_scaling_factor": 2.5, "swiglu_limit": 10.0,
            "gated": True, "tensor_parallel": GLM_TP,
            "tensor_parallel_cut_axis": "intermediate"}


def _glm_routing():
    """`noaux_tc` selection with its live FP32 correction bias, and the clamp."""
    return {"activation": "silu", "scoring_func": "sigmoid", "renormalize": True,
            "routed_scaling_factor": 2.5, "apply_router_weight_on_input": False,
            "expert_map": None, "input_dtype": "torch.bfloat16",
            "topk_weights_dtype": "torch.float32", "topk_ids_dtype": "torch.int32",
            "device": "cuda:0", "weights_contract": "post_renormalization_and_routed_scaling",
            "swiglu_limit": 10.0, "n_group": 1, "topk_group": 1, "topk_method": "noaux_tc",
            "source_protocol": {"router_class": "Glm5NextTopKRouter",
                                "router_source_sha256": "a" * 64, "scoring_func": "sigmoid",
                                "topk_method": "noaux_tc", "normalization_epsilon": 1e-6,
                                "correction_bias": {"content_sha256": "b" * 64,
                                                    "dtype": "torch.float32"},
                                "expert_bias_affects": "selection_only", "norm_topk_prob": True}}


def _glm_cell():
    """One whole GLM routed owner at TP2, in the producer's own input shape."""
    from prismaquant import native_moe_panel as panel
    from prismaquant.joint_aura import arithmetic_identity, make_joint_aura_entry

    shape, routing = _glm_shape(), _glm_routing()
    width = panel.rank_local_intermediate(shape)
    activation = {"schema": "prismaquant.joint_aura.activation.v1", "quantizes_input": True,
                  "activation_max_abs": None, "input_global_scale": None, "clip_enabled": False}
    members = []
    for expert in range(GLM_EXPERTS):
        for role in panel.ROLES:
            unit = f"{GLM_UNIT}.{expert}.{role}"
            dims = [GLM_HIDDEN, width] if role == "w2" else [width, GLM_HIDDEN]
            weight = _fake_identity(unit, dims)
            members.append({"unit": unit, "expert": expert, "role": role, "format": FORMAT,
                            "shape": dims, "source_weight": weight, "rendered_weight": weight,
                            "activation": copy.deepcopy(activation),
                            "wire": {"blob_sha256": "3" * 64, "blob_bytes": 42,
                                     "record": {"unit": unit}}})
    source = {"files": {"fixture.safetensors": "8" * 64}, "config_sha256": "9" * 64,
              "auxiliary_sha256": {"config.json": "9" * 64},
              "tensors": {member["unit"] + ".weight": "fixture.safetensors" for member in members}}
    config = {"model_type": "glm5_next", "fixture": True}
    source_execution = {"schema": "prismaquant.joint_aura.source_execution.v1", "modules": {
        "": {"attention": "eager", "experts": "grouped_mm"},
        GLM_UNIT: {"attention": "eager", "experts": "grouped_mm"}}}
    value = {"config": config, "weight_map": {name: name for name in source["tensors"]},
             "checkpoint_weight_map": source["tensors"],
             "shards": [{"path": "/fixture/fixture.safetensors", "size": 1, "sha256": "8" * 64}]}
    model = {"schema": "prismaquant.streamed_model.identity.v1", "source": "/fixture",
             "resolved_commit": None, "content_sha256": identity_sha256(value), **value}
    arithmetic = arithmetic_identity(torch.bfloat16)
    probe = {"schema": "prismaquant.joint_aura.probes.v2", "source_model": model,
             "calibration_sha256": "1" * 64, "calibration_shape": [1, 2],
             "calibration_dtype": "torch.int64", "producer_source_sha256": "2" * 64,
             "n_probes": 3, "seed_base": 7, "token_scope": "causal", "distribution": "rademacher",
             "normalization": "global_kl_fisher", "temperature": 1.0, "arithmetic": arithmetic,
             "source_execution": copy.deepcopy(source_execution)}
    rows = {}
    for member in members:
        joint = {"schema": "prismaquant.joint_aura.operator.v2", "qname": member["unit"],
                 "format": FORMAT,
                 **{key: member[key] for key in ("source_weight", "rendered_weight", "activation")},
                 "arithmetic": arithmetic, "probe_identity_sha256": identity_sha256(probe)}
        rows[member["unit"]] = make_joint_aura_entry(
            operator_identity=joint, probe_identity=probe,
            signed_components=[{"weight": value, "activation": 0.0, "mixed": 0.0, "total": value}
                               for value in (.1, -.2, .3)])
    phases = _one_token_phases(hidden=GLM_HIDDEN, top_k=shape["top_k"])
    tensor_fields = {key: phases["prefill"][key] for key in ("input", "topk_ids", "topk_weights")}
    calibration = {"schema": "prismaquant.calibration_input.v1", "calibration_sha256": "1" * 64,
                   "shape": [1, 2], "dtype": "torch.int64"}
    capture = {"schema": "prismaquant.routed_boundary_capture.v1", "unit": GLM_UNIT, "shape": shape,
        "routing": copy.deepcopy(routing), "calibration_sha256": "1" * 64,
        "calibration_shape": [1, 2], "calibration_dtype": "torch.int64",
        "producer_source": source, "runtime_config": config,
        "source_execution": copy.deepcopy(source_execution), "capture_source_sha256": "c" * 64,
        "phases": phases,
        "model_load_contract": {"schema": "prismaquant.pretrained_initialization.v1",
                                "scope": "checkpoint_missing_state", "status": "completed",
                                "transformers_version": "fixture-transformers"},
        "attention_implementation": "eager",
        "capture_runtime": {"torch": "fixture-torch", "cuda": "fixture-cuda",
                            "transformers": "fixture-transformers"}}
    inputs = {"schema": panel.INPUT_SCHEMA, "unit": GLM_UNIT, "format": FORMAT, "shape": shape,
        "members": members, "profile_role_order": list(panel.ROLES),
        "routing": copy.deepcopy(routing),
        "execution": panel.owner_execution(shape, format_name=FORMAT),
        "calibration": calibration, "routing_capture": capture,
        "routing_capture_sha256": identity_sha256(capture),
        "runtime_image": "fixture/image@sha256:" + "a" * 64, "serving_config_sha256": "b" * 64,
        "numerics": {"atol": .015625, "rtol": .015625}, "phases": phases,
        "probe_request": {
            **{key: probe[key] for key in ("n_probes", "seed_base", "token_scope", "temperature",
                                           "distribution", "normalization")},
            "source_model": "/fixture", "source_shards": source["files"],
            "source_config_sha256": source["config_sha256"],
            "source_auxiliary_sha256": source["auxiliary_sha256"]}}
    native_members = [{**{key: member[key] for key in ("unit", "expert", "role", "format", "shape",
                                                       "source_weight", "rendered_weight")},
                       "wire_sha256": member["wire"]["blob_sha256"],
                       "wire_record_sha256": identity_sha256(member["wire"]["record"])}
                      for member in members]
    native = {"members": native_members, "shape": shape, "routing": copy.deepcopy(routing),
        "profile_role_order": list(panel.ROLES),
        "routing_capture_sha256": inputs["routing_capture_sha256"], "serving_config_sha256": "b" * 64,
        "native_tensors": {"fixture": tensor_fields["input"]}, "scheme": {"fixture": True},
        "config": {"fixture": "actual MoE config"},
        "phases": {phase: {"transport": copy.deepcopy(phases[phase]["transport"])}
                   for phase in phases},
        "declared_route": {"kind": "moe", "policy": "TESSERA_FP8:resident",
            "symbol": "vllm.fused_moe.modular_kernel:fixture", "decoder": "torch_materialize_stock",
            "contract": "fp8_per_token_dynamic"}}
    native["config_sha256"] = identity_sha256(native["config"])
    runtime = {"schema": "tessera.native_moe_runtime.v1", "execution": dict(inputs["execution"]),
        "image": inputs["runtime_image"], "resource_collector": {"library_sha256": "5" * 64}}
    workspace = {"schema": "tessera.native_moe_workspace.v1", "owner": "vllm.WorkspaceManager",
        "num_ubatches": 1, "num_lanes": 1, "locked": True,
        "slots": [{"index": 0, "shape": [64], "dtype": "torch.uint8", "device": "cuda:0",
                   "storage_bytes": 64, "logical_bytes": 64, "stride": [1], "storage_offset": 0}],
        "resident_bytes": 64}
    preflight = {"schema": "tessera.native_moe_preflight.v1", "status": "untimed_preparation",
        "operator": native, "runtime": runtime, "runtime_sha256": identity_sha256(runtime),
        "workspace": workspace, "workspace_sha256": identity_sha256(workspace),
        "native_tensors_sha256": identity_sha256(native["native_tensors"]),
        "scheme_sha256": identity_sha256(native["scheme"])}
    return inputs, preflight, rows


GLM_ROUTE_SHAPE = f"M1:N{2 * GLM_INTERMEDIATE}:K{GLM_HIDDEN}"


def test_a_glm_288_owner_prices_two_ranks_end_to_end(joined, tmp_path):
    """The main objective's geometry: 288 experts, top-8, TP2, one atomic row.

    Nothing here is a measurement -- the receipts are synthetic CPU fixtures --
    but the row is built and admitted exactly as a real one would be: two rank
    receipts carrying the shared original container, one row whose members are
    the rank-local shapes the served route reads (1024x4096, not 2048x4096), and
    the loader's own gate re-deriving every rank's bytes.
    """
    cell = _glm_cell()
    phases = _one_token_phases(hidden=GLM_HIDDEN, top_k=8)
    item, context, relation, panel, _receipts = _routed_gate(
        cell, tmp_path, world_size=GLM_TP, samples=[FAST, SLOW], phases=phases,
        route_shape=GLM_ROUTE_SHAPE)
    row, resources = item["row"], item["row"]["resources"]
    assert len(row["binding"]["member_formats"]) == GLM_MEMBERS
    assert row["binding"]["member_shapes"][f"{GLM_UNIT}.0.w1"] == [1024, GLM_HIDDEN], (
        "the priced member is the rank-local one, not the TP1 width")
    assert row["binding"]["member_shapes"][f"{GLM_UNIT}.0.w2"] == [GLM_HIDDEN, 1024]
    assert resources["schema"] == RANK_RESOURCES_SCHEMA and resources["world_size"] == GLM_TP
    assert resources["wire_bytes"] == WIRE_BYTES * GLM_MEMBERS, (
        "one canonical whole-module container, charged once for the owner")
    assert resources["prefill_ms"] == 5.0, "one whole-owner interval, bounded by its slowest rank"
    assert resources["rank_medians_ms"]["prefill"] == [2.0, 5.0]
    assert row["prefill"]["receipt_path"].endswith("routed.receipt.1.json")
    table = _gate_table(item, context, tmp_path)
    admit_native_rows(table, relation)
    (admitted,) = table.rows
    assert isinstance(admitted.resources, RuntimeRankResources)
    assert admitted.resources.world_size == GLM_TP and len(admitted.resources.ranks) == GLM_TP
    assert panel["execution"]["tensor_parallel"] == GLM_TP


def test_a_glm_owner_is_refused_a_single_rank_vector_for_a_two_rank_world(joined, tmp_path):
    """288 experts do not change the rule: one rank's bound is not the world's."""
    cell = _glm_cell()
    phases = _one_token_phases(hidden=GLM_HIDDEN, top_k=8)
    item, _context, _relation, _panel, receipts = _routed_gate(
        cell, tmp_path, world_size=GLM_TP, phases=phases, route_shape=GLM_ROUTE_SHAPE)
    collapsed = copy.deepcopy(item["row"]["resources"])
    collapsed["ranks"] = collapsed["ranks"][:1]
    with pytest.raises(RuntimePriceError, match="exactly one record per rank"):
        parse_row_resources(collapsed, tensor_parallel=GLM_TP, where="glm row")
    assert len(receipts) == GLM_TP
