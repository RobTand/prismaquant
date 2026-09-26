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
from datetime import datetime, timedelta, timezone
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
    LATENCY_SCOPE_KIND, RUNTIME_COLLECTIVE_OP, RUNTIME_COLLECTIVE_SITE, ArtifactReader, admit_native_rows,
    routed_rank_bound,
)
from test_native_moe_panel import joined, receipt_fixture  # noqa: F401 - fixtures
from test_allocator_measured_runtime_cli import admit_synthetic_table

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
    from test_allocator_measured_runtime_cli import _main_fixture, admit_synthetic_table

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
    admit_synthetic_table(monkeypatch)
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
        # Seal the encoder source WITHOUT discarding the rest of the wire
        # record identity: a GLM cell names its own unit there, and the
        # qualified render receipt is joined through that whole object.
        member["wire"]["record"]["identity"] = {
            **member["wire"]["record"].get("identity", {}), "encoder_source_sha256": WIRE_SEAL}
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
    the run it prices, the producer source-tree seal its wires must carry, the
    serving configuration whose digest the panel publishes, and the served
    artifact manifest that configuration names, which must carry the row's
    route family (#570 leg (b) residual). Whether the
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
        "transient_charge_boundary": "prismaquant.transient_charge_boundary.v1",
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
                                    "common": {"producer_source_tree_sha256": WIRE_SEAL}}},
                "record": {"configuration": _served_config(tmp_path)},
                "reader": ArtifactReader(tmp_path)}
    return item, context, relation, panel, receipts


def _served_config(tmp_path):
    """The served artifact manifest the intake gate's family check reads.

    The routed cell declares `TESSERA_FP8:resident`, so the manifest exercises
    that family; a test that needs another roster rewrites the manifest file,
    which carries no digest and is read, never bound.
    """
    artifact = tmp_path / "served-artifact"
    artifact.mkdir(exist_ok=True)
    (artifact / "tessera_serving_manifest.json").write_text(json.dumps(
        {"modules": {"synthetic-module": {"family": "TESSERA_FP8"}}}))
    config = {"artifact": {"path": str(artifact), "scope": "synthetic served artifact"}}
    path = tmp_path / "served-config.json"
    path.write_text(json.dumps(config, sort_keys=True))
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


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


def test_the_intake_gate_refuses_a_route_family_the_served_artifact_never_exercised(
        joined, tmp_path):
    """#570 leg (b) residual on the routed path: the cell declares
    `TESSERA_FP8:resident`, so a manifest exercising only another family
    refuses naming `TESSERA_FP8`, even though every receipt rehashes."""
    item, context, relation, _panel, _receipts = _routed_gate(joined, tmp_path)
    (tmp_path / "served-artifact" / "tessera_serving_manifest.json").write_text(json.dumps(
        {"modules": {"synthetic-module": {"family": "TESSERA_BF16"}}}))
    with pytest.raises(RuntimePriceError, match="TESSERA_FP8"):
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
            "tensor_parallel_cut_axis": "intermediate", "tensor_parallel_rank": 0}


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
                                "topk_method": "noaux_tc", "normalization_epsilon": 1e-20,
                                "correction_bias": {"content_sha256": "b" * 64,
                                                    "dtype": "torch.float32"},
                                "expert_bias_affects": "selection_only", "norm_topk_prob": True}}


def _glm_cell(tmp_path):
    """One whole GLM routed owner at TP2, in the producer's own input shape."""
    from prismaquant import native_moe_panel as panel
    from prismaquant.joint_aura import arithmetic_identity, make_joint_aura_entry

    shape, routing = _glm_shape(), _glm_routing()
    activation = {"schema": "prismaquant.joint_aura.activation.v1", "quantizes_input": True,
                  "activation_max_abs": None, "input_global_scale": None, "clip_enabled": False}
    members = []
    for expert in range(GLM_EXPERTS):
        for role in panel.ROLES:
            unit = f"{GLM_UNIT}.{expert}.{role}"
            # The MODULE's geometry, which is what the wire identity and the
            # joint quality row bind, beside this rank's own cut of it -- the
            # two fields a TP2 panel must not conflate (TS #539:
            # `member["shape"]` is the source container, the runtime binding is
            # the rank-local render).
            container = panel.container_member_shape(shape, role)
            local = panel.rank_local_member_shape(shape, role)
            members.append({"unit": unit, "expert": expert, "role": role, "format": FORMAT,
                            "shape": container,
                            "source_weight": _fake_identity(unit, container),
                            "rendered_weight": _fake_identity(unit + ".render", local),
                            "quality_rendered_weight": _fake_identity(unit + ".full-render", container),
                            "activation": copy.deepcopy(activation),
                            "wire": {"blob_sha256": "3" * 64, "blob_bytes": 42,
                                     "record": {"unit": unit, "identity": {
                                         "unit": unit, "encoder_source_sha256": WIRE_SEAL}}}})
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
        joint["rendered_weight"] = member["quality_rendered_weight"]
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
    _bind_glm_quality_fixture(tmp_path, inputs, model, calibration)
    return inputs, preflight, rows


def _glm_rank_render_proof(member, shape):
    """This rank's window of the whole container, written out rather than derived.

    The producer computes the same record through ``member_window``. Spelling
    it here as a literal is what makes the comparison in ``freeze_moe_panel``
    evidence: a fixture that asked the code under test for its own expectation
    would agree with any arithmetic, including the wrong one.
    """
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    rank = shape.get("tensor_parallel_rank", 0)
    width = GLM_INTERMEDIATE // GLM_TP
    low, high = rank * width, (rank + 1) * width
    rows, cols, axis = (([0, GLM_HIDDEN], [low, high], "column") if member["role"] == "w2"
                        else ([low, high], [0, GLM_HIDDEN], "row"))
    return {"schema": "prismaquant.native_moe_rank_render.v1", "unit": member["unit"],
            "format": member["format"], "role": member["role"], "rank": rank,
            "world_size": GLM_TP, "rows": rows, "cols": cols, "axis": axis,
            "wire_sha256": member["wire"]["blob_sha256"],
            "encoding_identity_sha256": canonical_json_sha256(
                member["wire"]["record"]["identity"], where="fixture encoding"),
            "qualified_render_sha256": identity_sha256(member["quality_rendered_weight"]),
            "rendered_weight": member["rendered_weight"]}


def _bind_glm_quality_fixture(tmp_path, inputs, model, calibration):
    """The original campaign's bound preparation: a completion and its own PWC.

    Both records are the producer's own shape -- the completion literal in
    ``tessera_joint_aura`` and the verified-cell receipt it writes per rung --
    spelled out here so the panel is checked against the artifact a campaign
    actually leaves behind and not against this test's convenience.
    """
    import pickle
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    from prismaquant.production_weight_cache import ProductionWeightCache
    from prismaquant.tessera_joint_aura import HISTORICAL_WIRE_VALIDATION, PREPARED_SCHEMA

    verified = {}
    for member in inputs["members"]:
        verified[member["unit"], member["format"]] = {
            "source_weight": member["source_weight"],
            "rendered_weight": member["quality_rendered_weight"],
            "activation": member["activation"],
            "encoding_identity_sha256": canonical_json_sha256(
                member["wire"]["record"]["identity"], where="fixture encoding"),
            "wire_sha256": member["wire"]["blob_sha256"], "render_file_sha256": "e" * 64,
            "render_origin": "encoded", "render_comparison": "independent_render_vs_wire"}
        member["rank_render_proof"] = _glm_rank_render_proof(member, inputs["shape"])
    campaign_inputs = {"fixture": "original campaign anchor inputs"}
    common = {"schema": PREPARED_SCHEMA, "source_model_identity": model,
              "plan_sha256": "d" * 64, "implementation_sha256": "f" * 64,
              "source_execution": inputs["routing_capture"]["source_execution"],
              "reader_identity": {"fixture": True}, "projection_backend": {"fixture": True}}
    cache = ProductionWeightCache(weights={}, levers={}, metadata={
        **common, "verified_cells": verified, "inputs": campaign_inputs})
    cache_path = tmp_path / "qualified-production.pkl"
    cache_path.write_bytes(pickle.dumps(cache))
    completion = {**common, "status": "complete", "calibration_input": calibration,
                  "formats_by_qname": {member["unit"]: [member["format"]]
                                       for member in inputs["members"]},
                  "production_cache": {"path": str(cache_path),
                                       "sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest()}}
    path = tmp_path / "qualified-completion.json"
    path.write_text(json.dumps(completion))
    inputs["quality_preparation"] = {
        "prepared": {"path": str(path),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "plan_sha256": common["plan_sha256"], "wire_validation": HISTORICAL_WIRE_VALIDATION,
        "inputs": campaign_inputs}


GLM_ROUTE_SHAPE = f"M1:N{2 * GLM_INTERMEDIATE}:K{GLM_HIDDEN}"


#: Each mutation and the refusal it must produce, so a fixture that broke for
#: any other reason cannot pass this test by raising something else.
_QUALITY_MUTATIONS = {
    "full_hash": "joint rendered_weight",
    "full_shape": "render/source geometry differs",
    "rank": "rank render proof",
    "axis": "rank render proof",
    "rows": "rank render proof",
    "rank_hash": "rank render proof",
    "wire": "qualified wire",
    "encoder": "qualified encoder",
    "prepared_bytes": "owned bytes",
    "cache_bytes": "owned bytes",
}


@pytest.mark.parametrize("mutation", sorted(_QUALITY_MUTATIONS))
def test_full_quality_and_rank_cut_are_independently_bound(tmp_path, mutation):
    """One forged field per run, each refused by the gate that owns it.

    The quality identity is the whole container's render and the cut proof is
    this rank's window of it. They are separate bindings: forging either, or
    the bytes of the preparation both are read from, must be refused, and the
    refusal must name the thing that was forged.
    """
    from prismaquant import native_moe_panel as native
    inputs, preflight, rows = _glm_cell(tmp_path)
    member = inputs["members"][0]
    joint = rows[member["unit"]]["joint_operator_identity"]
    if mutation == "full_hash":
        joint["rendered_weight"] = {**joint["rendered_weight"], "content_sha256": "0" * 64}
    elif mutation == "full_shape":
        # The 3dcb defect, at the panel: a joint quality row that names THIS
        # rank's cut rather than the module's render.
        joint["rendered_weight"] = dict(member["rendered_weight"])
    elif mutation in ("rank", "axis", "rows", "rank_hash"):
        proof = member["rank_render_proof"]
        if mutation == "rank_hash":
            proof["rendered_weight"] = {**proof["rendered_weight"], "content_sha256": "0" * 64}
        else:
            proof[mutation] = {"rank": 1, "axis": "column", "rows": [1024, 2048]}[mutation]
    elif mutation == "wire":
        member["wire"]["blob_sha256"] = "0" * 64
    elif mutation == "encoder":
        member["wire"]["record"]["identity"]["unit"] += ".wrong"
    else:
        path = tmp_path / ("qualified-completion.json" if mutation == "prepared_bytes"
                           else "qualified-production.pkl")
        path.write_bytes(path.read_bytes() + b" ")
    rows[member["unit"]]["joint_operator_identity_sha256"] = identity_sha256(joint)
    with pytest.raises(ValueError, match=_QUALITY_MUTATIONS[mutation]):
        native.freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)


def test_a_rank_local_panel_refuses_an_unbound_quality_preparation(tmp_path):
    """No bound historical preparation, no rank-local panel -- refused by name."""
    from prismaquant import native_moe_panel as native
    inputs, preflight, rows = _glm_cell(tmp_path)
    del inputs["quality_preparation"]
    with pytest.raises(ValueError, match="requires independently bound full-quality preparation"):
        native.freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)


def test_a_rank_local_preparation_refuses_an_unbound_quality_preparation(monkeypatch):
    """The same refusal on the producing side, before any tensor is touched.

    The activation-protocol gate runs first and is unrelated, so it is
    satisfied here rather than reordered: what this asserts is that the
    preparation refuses on its own missing binding, not on a device.
    """
    pytest.importorskip("tessera.cached_unit")
    pytest.importorskip("torch")
    monkeypatch.setenv("PRISMAQUANT_PROD_ACT_SCALES", "0")
    from prismaquant import native_moe_panel as native
    with pytest.raises(ValueError, match="requires independently bound full-quality preparation"):
        native.prepare_moe_inputs(
            None, {}, {}, unit=GLM_UNIT, members=[], shape=_glm_shape(), routing=_glm_routing(),
            calibration_receipt={}, routing_capture={}, experts_module=None, profile=None,
            wire_blobs={}, wire_records={}, encoding_identities={}, numerics={},
            max_resident_bytes=1, max_temporary_bytes=1,
            runtime_image="fixture/image@sha256:" + "a" * 64, serving_config_sha256="b" * 64,
            probe_request={})


def test_the_frozen_joint_names_the_container_render_not_the_rank_cut(tmp_path):
    """The positive half: the quality identity is the WHOLE module's render.

    The panel keeps both readings and they are different objects -- the joint
    quality row names the container, the native member roster and the runtime
    binding name this rank's cut.
    """
    from prismaquant import native_moe_panel as native
    inputs, preflight, rows = _glm_cell(tmp_path)
    expected = copy.deepcopy(inputs["quality_preparation"])
    frozen = native.freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)
    member = frozen["members"][0]
    joint = rows[member["unit"]]["joint_operator_identity"]
    assert member["role"] == "w1"
    assert joint["rendered_weight"] == member["quality_rendered_weight"]
    assert joint["rendered_weight"] != member["rendered_weight"]
    assert member["quality_rendered_weight"]["shape"] == [GLM_INTERMEDIATE, GLM_HIDDEN]
    assert member["rendered_weight"]["shape"] == [GLM_INTERMEDIATE // GLM_TP, GLM_HIDDEN]
    assert frozen["quality_preparation"] == expected
    assert "historical_prepared_identity" in frozen["quality_preparation"]["wire_validation"]
    # The rank-local reading stays rank-local everywhere it is published.
    assert "quality_rendered_weight" not in native._native_member_identity(member)
    assert frozen["runtime_binding"]["member_shapes"][member["unit"]] == [
        GLM_INTERMEDIATE // GLM_TP, GLM_HIDDEN]


@pytest.mark.parametrize("rank", [0, 1])
def test_the_rank_render_proof_moves_its_window_with_the_rank(tmp_path, rank):
    """Each rank's proof names its own window of one unchanged container."""
    from prismaquant import native_moe_panel as native
    inputs, preflight, rows = _glm_cell(tmp_path)
    inputs["shape"]["tensor_parallel_rank"] = rank
    inputs["routing_capture_sha256"] = identity_sha256(inputs["routing_capture"])
    preflight["operator"]["routing_capture_sha256"] = inputs["routing_capture_sha256"]
    for member in inputs["members"]:
        member["rank_render_proof"] = _glm_rank_render_proof(member, inputs["shape"])
    frozen = native.freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)
    member = frozen["members"][0]
    width = GLM_INTERMEDIATE // GLM_TP
    assert member["rank_render_proof"]["rows"] == [rank * width, (rank + 1) * width]
    assert member["rank_render_proof"]["cols"] == [0, GLM_HIDDEN]
    assert member["rank_render_proof"]["rank"] == rank
    assert member["quality_rendered_weight"]["shape"] == [GLM_INTERMEDIATE, GLM_HIDDEN]


def test_a_glm_288_owner_prices_two_ranks_end_to_end(joined, tmp_path):
    """The main objective's geometry: 288 experts, top-8, TP2, one atomic row.

    Nothing here is a measurement -- the receipts are synthetic CPU fixtures --
    but the row is built and admitted exactly as a real one would be: two rank
    receipts carrying the shared original container, one row whose members are
    the rank-local shapes the served route reads (1024x4096, not 2048x4096), and
    the loader's own gate re-deriving every rank's bytes.
    """
    cell = _glm_cell(tmp_path)
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
    cell = _glm_cell(tmp_path)
    phases = _one_token_phases(hidden=GLM_HIDDEN, top_k=8)
    item, _context, _relation, _panel, receipts = _routed_gate(
        cell, tmp_path, world_size=GLM_TP, phases=phases, route_shape=GLM_ROUTE_SHAPE)
    collapsed = copy.deepcopy(item["row"]["resources"])
    collapsed["ranks"] = collapsed["ranks"][:1]
    with pytest.raises(RuntimePriceError, match="exactly one record per rank"):
        parse_row_resources(collapsed, tensor_parallel=GLM_TP, where="glm row")
    assert len(receipts) == GLM_TP


# --------------------------------------------------------------------------
# The whole cost model: 864 member rows -> the allocator's own CLI
# --------------------------------------------------------------------------

GLM_CLI_BUDGETS = (10 ** 9, 10 ** 9)


def _glm_cli_fixture(tmp_path, *, budgets=GLM_CLI_BUDGETS):
    """A GLM-288 cost model, its measured owner row, and a sealed per-rank charge.

    The row is the emitter's own: one atomic whole-owner row whose members are
    the 864 rank-local expert projections, keyed by the unit its producer
    observed. What the allocator's CLI needs beyond it is the cost model those
    members are priced from (probe stats and joint-AURA rows for every member,
    because the DP reads per-member identities), a runtime context that names
    the owner's operator route, and the per-rank fixed charge two sealed
    rank-scoped captures recompute. Nothing here is a measurement.
    """
    import pickle

    from prismaquant import allocator
    from prismaquant.allocator_candidates import serialized_candidate_payload
    from prismaquant.measured_runtime_prices import CONTEXT_SCHEMA, SCHEMA
    from test_runtime_rank_resources import _sealed_per_rank_partition

    inputs, preflight, rows = _glm_cell(tmp_path)
    # The owner's canonical wire extent is what the DP's aggregated candidate
    # prices, so the fixture's wire records carry each member's real serialized
    # extent rather than a placeholder. Without that the two sides of one
    # artifact would disagree, which is a refusal the CLI would rightly make.
    format_spec = allocator.fr.get_format(FORMAT)
    for member in inputs["members"]:
        serialized, _, _ = serialized_candidate_payload(
            format_spec, tuple(member["shape"]), qname=member["unit"])
        member["wire"]["blob_bytes"] = serialized
    spec, panel, cost_payload, _receipts = _write(
        (inputs, preflight, rows), tmp_path, world_size=GLM_TP, samples=[FAST, SLOW],
        receipt_mutation=_one_token_route(GLM_ROUTE_SHAPE))
    item = emitter.bind_native_receipt(spec, cost_payload=cost_payload, cost_sha256="4" * 64,
                                       manifest_dir=tmp_path, table_dir=tmp_path)
    row = copy.deepcopy(item["row"])
    probe_path, cost_path = tmp_path / "glm-probe.pkl", tmp_path / "glm-costs.pkl"
    # A routed expert member carries its own router/expert topology, which is
    # what the explicit serving scope classifies it by; the shape and the
    # parameter count are what the DP prices it with.
    stats = {member["unit"]: {"h_trace": 1.0, "n_params": math.prod(member["shape"]),
                              "in_features": member["shape"][1], "out_features": member["shape"][0],
                              "router_path": f"{GLM_UNIT}.gate", "expert_id": member["expert"]}
             for member in inputs["members"]}
    model_dir = tmp_path / "glm-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "glm5_next"}))
    probe_path.write_bytes(pickle.dumps({"stats": stats, "meta": {"model": str(model_dir)}}))
    cost_path.write_bytes(pickle.dumps({
        "costs": cost_payload["costs"], "meta": {"formats": [FORMAT]},
        "provenance": {"cost_mode": "aura", "joint_activation": True,
                       "cost_currency": "joint_aura_predicted_dloss"}}))
    receipt = tmp_path / "synthetic-receipt.txt"
    receipt.write_text("Synthetic CPU test fixture, not GPU measurement evidence.\n")
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    context = {"schema": CONTEXT_SCHEMA,
        # The fixture declares a fixed charge inline; it composes with the
        # priced rows only under a named transient charge boundary.
        "transient_charge_boundary": "prismaquant.transient_charge_boundary.v1",
        "serving_context": {"platform": "sm_121", "structure": "routed_moe",
                            "residency": "resident",
                            "runtime_image": panel["runtime"]["image"], "execution_mode": "eager"},
        "gpu_identity": "synthetic", "runtime_sha256": "b" * 64,
        "source_sha256": panel["source_sha256"], "calibration_sha256": panel["calibration_sha256"],
        "prompt_tokens": 1, "batch_size": 1, "tensor_parallel": GLM_TP, "graph_mode": "eager",
        "operator_routes": {panel["unit"]: {panel["format"]: row["binding"]["operator_route"]}}}
    now = datetime.now(timezone.utc)
    table = {"schema": SCHEMA, "table_id": "synthetic-glm-only", "status": "proposal_data",
        "composition": "sequential_operator_sum", "context": context,
        "cost_sha256": hashlib.sha256(cost_path.read_bytes()).hexdigest(),
        "measured_at": (now - timedelta(days=1)).isoformat(),
        "valid_until": (now + timedelta(days=1)).isoformat(), "fixed_assignment": {},
        "fixed_resources": {"prefill_ms": 0.0, "decode_ms": 0.0, "serialized_bytes": 0,
                            "resident_bytes": 0, "peak_scratch_bytes": 0, "activation_bytes": 0,
                            "kv_bytes": 0},
        "fixed_resources_receipt_path": receipt.name, "fixed_resources_receipt_sha256": receipt_sha,
        "rows": [row]}
    table_path, context_path = tmp_path / "glm-runtime.json", tmp_path / "glm-context.json"
    table_path.write_text(json.dumps(table))
    context_path.write_text(json.dumps(context))
    # The rank captures must name the bytes this table was priced against: its
    # own source model and the runtime manifest its context declares. The
    # remaining world coordinates are joined across the two ranks.
    _partition_reference, partition = _sealed_per_rank_partition(
        tmp_path, startup=(1024, 4096), model_sha256=(panel["source_sha256"],) * 2,
        runtime_manifest_sha256="b" * 64)
    partition_path = tmp_path / "glm-partition.json"
    partition_path.write_text(json.dumps(partition, sort_keys=True))
    argv = ["allocator", "--probe", str(probe_path), "--costs", str(cost_path),
            "--formats", FORMAT, "--target-bits", "32", "--pareto-targets", "32",
            "--layer-config", str(tmp_path / "layer.json"),
            "--pareto-csv", str(tmp_path / "pareto.csv"),
            "--pareto-output-dir", str(tmp_path / "seeds"),
            # The rung the owner was measured in is a Tessera rung, so the run
            # declares the serving scope it was priced under. The menu mode the
            # tests set is `research`: this is a synthetic cost model that is
            # never exported, and the attested-rung gate is what refuses a real
            # export of an unattested rung.
            "--tessera-platform", "sm_121",
            "--tessera-runtime-image", panel["runtime"]["image"],
            "--tessera-execution-mode", "eager", "--tessera-residency", "resident",
            # The owner's rung is priced as research: the serving profile the
            # GLM architecture defaults to is the packed-MoE export profile,
            # whose format restrictions are about what may be exported.
            "--target-profile", "research",
            "--measured-runtime-table", str(table_path),
            "--measured-runtime-context", str(context_path),
            "--slo-prefill-p95-ttft-ms", "1000",
            "--rank-device-budget-bytes", ",".join(str(budget) for budget in budgets),
            "--measured-runtime-rank-partition", str(partition_path)]
    return argv, partition_path, panel


def test_the_glm_cost_model_reaches_the_cli_and_expands_to_its_864_members(tmp_path, monkeypatch):
    """The main objective's own geometry, end to end through ``allocator.main``.

    A whole routed owner is one DP item named by the allocator's aggregation
    (``.__packed_serving__``), while the producer's row names the unit it
    measured. The two are reconciled by the row's own member roster, and the
    assignment the CLI writes is expanded back to all 864 member Linears. No
    GPU ran: every receipt and cost row here is a synthetic CPU fixture.
    """
    admit_synthetic_table(monkeypatch)
    argv, _partition, _panel = _glm_cli_fixture(tmp_path)
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "research")
    monkeypatch.setattr(sys, "argv", argv)
    from prismaquant import allocator
    from prismaquant.layer_config import load_assignment

    allocator.main()
    assignment = load_assignment(tmp_path / "layer.json")
    members = {member["unit"] for member in _panel["members"]}
    assert GLM_MEMBERS == 864
    assert set(assignment) == members, (
        "the aggregated owner expands to its own 864 member Linears, not to a "
        "packed-serving super-item name")
    assert set(assignment.values()) == {FORMAT}
    verdict = json.loads((tmp_path / "layer.json").read_text())[
        "__prismaquant__"]["serve_constraints"]
    assert verdict["per_rank_resources"] is True
    assert verdict["device_memory_total_published"] is False
    assert verdict["predicted"]["device_memory_bytes"] is None
    ranked = verdict["coverage"]["memory"]["operator_rank_totals"]
    assert ranked["world_size"] == GLM_TP
    assert len(ranked["ranks"]) == GLM_TP


def test_the_cli_refuses_a_forged_or_changed_rank_report(tmp_path, monkeypatch):
    """The admitted budget is only as good as the per-rank evidence behind it.

    Two defects a scalar world total cannot see: the partition redistributes
    one world's numbers between its ranks, and a rank row points at its peer's
    capture. Both refuse by name, before any solve runs.
    """
    argv, partition_path, _panel = _glm_cli_fixture(tmp_path)
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "research")
    partition = json.loads(partition_path.read_text())
    redistributed = copy.deepcopy(partition)
    redistributed["ranks"][0]["terms"] = copy.deepcopy(partition["ranks"][1]["terms"])
    redistributed["ranks"][1]["terms"] = copy.deepcopy(partition["ranks"][0]["terms"])
    partition_path.write_text(json.dumps(redistributed, sort_keys=True))
    monkeypatch.setattr(sys, "argv", argv)
    from prismaquant import allocator

    with pytest.raises(SystemExit, match="may not restate it"):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()

    peer = copy.deepcopy(partition)
    peer["ranks"][1]["report"] = copy.deepcopy(partition["ranks"][0]["report"])
    partition_path.write_text(json.dumps(peer, sort_keys=True))
    with pytest.raises(SystemExit, match="report rank"):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()

    # A rank that observed a different model under the same runtime, rank and
    # world size. The partition's own reference is restated, so what refuses is
    # the world-identity join rather than the checksum reader.
    mixed_path = Path(partition["ranks"][1]["report"]["path"])
    mixed_report = json.loads(mixed_path.read_text())
    mixed_report["identity"]["run"]["model_sha256"] = "c" * 64
    mixed_report["partition"]["identity"]["model_sha256"] = "c" * 64
    mixed_path.write_text(json.dumps(mixed_report, sort_keys=True))
    mixed = copy.deepcopy(partition)
    mixed["ranks"][1]["report"] = {
        "path": str(mixed_path),
        "sha256": hashlib.sha256(mixed_path.read_bytes()).hexdigest()}
    partition_path.write_text(json.dumps(mixed, sort_keys=True))
    with pytest.raises(SystemExit, match="rank 1 names model_sha256"):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()


def _late_bound_receipt(receipt, panel):
    from prismaquant.native_moe_execution_binding import (
        RAW_RECEIPT_SCHEMA,execution_panel_from_joint,bind_execution_receipt)
    raw=copy.deepcopy(receipt);raw['schema']=RAW_RECEIPT_SCHEMA
    raw['panel']=execution_panel_from_joint(panel)
    raw['panel_sha256']=identity_sha256(raw['panel'])
    return bind_execution_receipt(raw,panel)


@pytest.mark.parametrize('world_size',[1,2])
def test_routed_emitter_reads_rank_after_verifying_late_binding(joined,tmp_path,world_size):
    spec,panel,cost,receipts=_write(joined,tmp_path,world_size=world_size)
    for rank,receipt in enumerate(receipts):
        (tmp_path/f'routed.receipt.{rank}.json').write_text(json.dumps(_late_bound_receipt(receipt,panel)))
    item=emitter.bind_native_receipt(spec,cost_payload=cost,cost_sha256='4'*64,
        manifest_dir=tmp_path,table_dir=tmp_path)
    assert item['row']['resources']['world_size']==world_size


@pytest.mark.parametrize('world_size',[1,2])
def test_native_admission_reads_rank_after_verifying_late_binding(joined,tmp_path,world_size):
    item,context,relation,panel,receipts=_routed_gate(joined,tmp_path,world_size=world_size)
    for rank,receipt in enumerate(receipts):
        path=tmp_path/f'routed.receipt.{rank}.json';path.write_text(json.dumps(_late_bound_receipt(receipt,panel)))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        ref=(item['binding']['receipt'] if rank==0 else item['binding']['peer_receipts'][rank-1]['receipt'])
        ref['sha256']=digest
        for phase in ('prefill','decode'):
            if Path(item['row'][phase]['receipt_path']).name==path.name:
                item['row'][phase]['receipt_sha256']=digest
    admit_native_rows(_gate_table(item,context,tmp_path),relation)
