"""CPU checks of the KDA route census (PQ #1214, E1): comparisons, driver, manifest.

The census runs on a GB10 inside the campaign container. Everything here runs
on the CPU: the routing comparisons against a restatement of the GLM
router's forward, the per-group driver on a toy layer whose "kernel" is a
module-global function swapped by a dispatch context, and the data manifest
against PrismaBuild's v2 contract, restated from ``src/prismabuild/core.py``
because PrismaBuild is not importable here.

The driver checks mutate the driver, not the fixture: a kernel that skips its
counters, a dispatch that leaks, a nondeterministic kernel and a layer that
never reaches its experts must each be caught by the census's own checks.
"""
from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from experiments import kda_route_census as census
from experiments.kda_route_census import (
    CensusRefused, LayerCapture, PlaneStats, build_census_manifest, changed_experts,
    compare_modes, derive_census_spec, group_checks, recompute_selection, run_group,
    sorted_routes, summarize)


# ---------------------------------------------------------------------------
# A restatement of Glm5NextTextTopkRouter.forward (transformers glm5_next)
# ---------------------------------------------------------------------------


def glm_router(hidden, weight, bias, *, n_group, topk_group, top_k, scaling, norm=True):
    router_logits = nn.functional.linear(hidden.type(torch.float32), weight.type(torch.float32))
    scores = router_logits.sigmoid()
    scores_for_choice = scores + bias
    experts = weight.shape[0]
    group_scores = (scores_for_choice.view(-1, n_group, experts // n_group)
                    .topk(2, dim=-1)[0].sum(dim=-1))
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (group_mask.unsqueeze(-1).expand(-1, n_group, experts // n_group)
                  .reshape(-1, experts))
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    return router_logits, topk_weights * scaling, topk_indices


ROUTING = {"n_group": 1, "topk_group": 1, "top_k": 8, "routed_scaling_factor": 2.5,
           "norm_topk_prob": True, "experts": 32}


def _capture_from_logits(logits, bias, routing=ROUTING, *, permute=None):
    """A capture whose routing is exactly the router's choice on ``logits``."""
    scores = logits.sigmoid()
    choice = scores + bias
    experts = logits.shape[1]
    n_group, topk_group = routing["n_group"], routing["topk_group"]
    group_scores = choice.view(-1, n_group, experts // n_group).topk(2, dim=-1)[0].sum(-1)
    group_idx = group_scores.topk(topk_group, dim=-1, sorted=False)[1]
    mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1).bool()
    mask = mask.unsqueeze(-1).expand(-1, n_group, experts // n_group).reshape(-1, experts)
    index = choice.masked_fill(~mask, float("-inf")).topk(routing["top_k"], dim=-1)[1]
    weights = scores.gather(1, index)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * routing["routed_scaling_factor"]
    if permute is not None:
        # The same selection and the same weights, listed in another order.
        index, weights = index[:, permute], weights[:, permute]
    tokens = logits.shape[0]
    moe = torch.linspace(-1, 1, tokens * 16).view(tokens, 16).to(torch.bfloat16)
    return {"attention_output": moe.clone(), "moe_input": moe, "router_logits": logits,
            "top_k_index": index, "top_k_weights": weights}


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def test_sorted_routes_orders_and_refuses_a_repeated_expert():
    assert sorted_routes(torch.tensor([[3, 1, 2]])).tolist() == [[1, 2, 3]]
    with pytest.raises(CensusRefused):
        sorted_routes(torch.tensor([[3, 1, 3]]))


def test_changed_experts_counts_set_differences_not_order():
    ref = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    other = torch.tensor([[3, 2, 1, 0], [0, 1, 2, 9], [4, 5, 6, 7]])
    assert changed_experts(ref, other).tolist() == [0, 1, 4]
    assert changed_experts(other, ref).tolist() == [0, 1, 4]


@pytest.mark.parametrize("n_group,topk_group", [(1, 1), (4, 2), (8, 3)])
def test_recompute_selection_is_the_routers_choice(n_group, topk_group):
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(64, 24, generator=generator)
    weight = torch.randn(32, 24, generator=generator)
    bias = torch.randn(32, generator=generator) * 0.1
    logits, _, index = glm_router(hidden, weight, bias, n_group=n_group,
                                  topk_group=topk_group, top_k=4, scaling=2.5)
    selected, margin, group_margin = recompute_selection(
        logits, bias, n_group=n_group, topk_group=topk_group, top_k=4)
    assert (changed_experts(index, selected) == 0).all()
    assert (margin >= 0).all()
    assert (group_margin is None) == (n_group == topk_group)


def test_identical_captures_compare_to_nothing():
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(6, 32, generator=generator)
    bias = torch.zeros(32)
    capture = _capture_from_logits(logits, bias)
    result = compare_modes(capture, capture, bias=bias, routing=ROUTING)
    arrays = result["arrays"]
    assert int(arrays["changed"].sum()) == 0
    assert float(arrays["reassigned"].sum()) == 0.0
    assert float(arrays["weight_rel"].nan_to_num(1.0).max()) == 0.0
    assert float(arrays["moe_input_rel"].max()) == 0.0
    assert result["recompute_mismatches"] == {"F1": 0, "K1": 0}


def test_a_permuted_but_equal_selection_is_not_a_flip():
    logits = torch.randn(4, 32, generator=torch.Generator().manual_seed(2))
    bias = torch.zeros(32)
    fallback = _capture_from_logits(logits, bias)
    kernel = _capture_from_logits(logits, bias, permute=torch.tensor([7, 6, 5, 4, 3, 2, 1, 0]))
    arrays = compare_modes(kernel, fallback, bias=bias, routing=ROUTING)["arrays"]
    assert int(arrays["changed"].sum()) == 0
    assert float(arrays["weight_rel"].max()) == 0.0


def test_one_boundary_swap_is_exactly_one_flip_inside_its_margin_bound():
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(5, 32, generator=generator)
    bias = torch.zeros(32)
    fallback = _capture_from_logits(logits, bias)
    choice = logits.sigmoid()
    order = choice[2].argsort(descending=True)
    eighth, ninth = int(order[7]), int(order[8])
    moved = logits.clone()
    # Swap token 2's 8th and 9th experts, and nothing else.
    moved[2, eighth], moved[2, ninth] = logits[2, ninth], logits[2, eighth]
    kernel = _capture_from_logits(moved, bias)
    result = compare_modes(kernel, fallback, bias=bias, routing=ROUTING)
    arrays = result["arrays"]
    assert arrays["changed"].tolist() == [0, 0, 1, 0, 0]
    weights = fallback["top_k_weights"][2] / 2.5
    dropped = float(weights[(fallback["top_k_index"][2] == eighth)].sum())
    assert float(arrays["reassigned"][2]) == pytest.approx(dropped)
    assert torch.isnan(arrays["weight_rel"][2])
    summary = summarize(arrays, sequence_length=5, routing=ROUTING)
    assert summary["flipped_tokens"] == 1 and summary["changed_histogram"][1] == 1
    assert summary["flips_beyond_margin_bound"] == 0
    assert summary["flips_per_sequence"] == [1]


def test_a_flip_outside_its_margin_bound_is_counted():
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn(4, 32, generator=generator)
    bias = torch.zeros(32)
    fallback = _capture_from_logits(logits, bias)
    kernel = _capture_from_logits(logits.flip(-1), bias)
    arrays = compare_modes(kernel, fallback, bias=bias, routing=ROUTING)["arrays"]
    flipped = arrays["changed"] > 0
    assert bool(flipped.any())
    # The mutation: claim the scores barely moved.
    arrays["choice_delta"] = torch.zeros_like(arrays["choice_delta"])
    summary = summarize(arrays, sequence_length=4, routing=ROUTING)
    assert summary["flips_beyond_margin_bound"] == int(flipped.sum())


def test_a_group_flip_changes_the_selection_with_grouped_routing():
    routing = {**ROUTING, "n_group": 4, "topk_group": 1, "top_k": 2}
    logits = torch.full((1, 32), -4.0)
    logits[0, 0:2] = 3.0          # group 0 leads
    logits[0, 8:10] = 2.9         # group 1 follows
    moved = logits.clone()
    moved[0, 8:10] = 3.1          # group 1 now leads
    bias = torch.zeros(32)
    fallback = _capture_from_logits(logits, bias, routing)
    kernel = _capture_from_logits(moved, bias, routing)
    result = compare_modes(kernel, fallback, bias=bias, routing=routing)
    assert result["arrays"]["changed"].tolist() == [2]
    assert result["recompute_mismatches"] == {"F1": 0, "K1": 0}
    summary = summarize(result["arrays"], sequence_length=1, routing=routing)
    assert summary["flips_beyond_margin_bound"] is None


def test_plane_stats_accumulate_and_merge():
    ref = torch.tensor([1.0, -2.0, 3.0, 4.0], dtype=torch.bfloat16)
    other = torch.tensor([1.0, -2.0, 3.5, 4.0], dtype=torch.bfloat16)
    first = PlaneStats()
    first.add(ref, other)
    merged = PlaneStats()
    merged.merge(first.partial())
    merged.merge(first.partial())
    value = merged.as_dict()
    assert value["elements"] == 8 and value["differing_elements"] == 2
    assert value["max_abs_diff"] == 0.5 and value["mean_abs_reference"] == 2.5
    assert value["rel_fro"] == pytest.approx((0.25 / 30) ** 0.5)


def test_bitwise_equal_tells_negative_zero_apart():
    assert census.bitwise_equal(torch.tensor([0.0]), torch.tensor([0.0]))
    assert not census.bitwise_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert census.differing_elements(torch.tensor([0.0, 1.0]), torch.tensor([-0.0, 1.0])) == 1


# ---------------------------------------------------------------------------
# The per-group driver on a toy layer
# ---------------------------------------------------------------------------


class Glm5NextTextTopkRouter(nn.Module):
    def __init__(self, experts=32, hidden=16):
        super().__init__()
        generator = torch.Generator().manual_seed(11)
        self.weight = nn.Parameter(torch.randn(experts, hidden, generator=generator))
        self.e_score_correction_bias = torch.zeros(experts)

    def forward(self, hidden):
        return glm_router(hidden.view(-1, hidden.shape[-1]), self.weight,
                          self.e_score_correction_bias, n_group=1, topk_group=1, top_k=8,
                          scaling=2.5)


class Glm5NextTextExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, top_k_index, top_k_weights):
        self.calls += 1
        return hidden_states


class ToyAttention(nn.Module):
    """Reads its attention function from a module global at call time, as GLM does."""

    def __init__(self, modeling, *, kda=True):
        super().__init__()
        self.modeling = modeling
        self.kda = kda

    def forward(self, x):
        if not self.kda:
            return x * 2
        return self.modeling.chunk_kimi_delta_attention(x)


class ToyLayer(nn.Module):
    def __init__(self, modeling, *, kda=True, stop_early=False):
        super().__init__()
        self.self_attn = ToyAttention(modeling, kda=kda)
        self.mlp = nn.Module()
        self.mlp.gate = Glm5NextTextTopkRouter()
        self.mlp.experts = Glm5NextTextExperts()
        self.stop_early = stop_early

    def forward(self, x):
        hidden = self.self_attn(x) + x
        _, weights, index = self.mlp.gate(hidden)
        if self.stop_early:
            return hidden
        return self.mlp.experts(hidden.view(-1, hidden.shape[-1]), index, weights)


class ToyDispatch:
    """CaptureKernelDispatch's shape: swap the module global for one block."""

    def __init__(self, modeling, kernel, *, leak=False):
        self.modeling, self.kernel, self.leak = modeling, kernel, leak
        self.fallback = modeling.chunk_kimi_delta_attention

    def __enter__(self):
        self.modeling.chunk_kimi_delta_attention = self.kernel
        return self

    def __exit__(self, *exc_info):
        if not self.leak:
            self.modeling.chunk_kimi_delta_attention = self.fallback
        return False


def _toy(*, kda=True, stop_early=False, leak=False, count=True, noise=False):
    counts = {"calls": 0, "gram_forward": 0, "gram_backward": 0}
    modeling = types.SimpleNamespace()

    def fallback(x):
        return torch.tanh(x)

    def kernel(x):
        if count:
            counts["calls"] += 1
            counts["gram_forward"] += 2
        out = torch.tanh(x.double()).to(x.dtype)
        # A different rounding: the kernel is not the fallback bitwise.
        out = out + (1e-2 * torch.sin(37 * x)).to(x.dtype)
        if noise:
            out = out + 1e-3 * torch.randn_like(out)
        return out

    modeling.chunk_kimi_delta_attention = fallback
    layer = ToyLayer(modeling, kda=kda, stop_early=stop_early)
    dispatch = ToyDispatch(modeling, kernel, leak=leak)
    return layer, dispatch, modeling, fallback, (lambda: dict(counts))


def _drive(layer, dispatch, modeling, fallback, counts, *, kda_layer=True, staged=None):
    if staged is None:
        staged = torch.randn(4, 8, 16, generator=torch.Generator().manual_seed(5))
    with LayerCapture(layer) as capture:
        return run_group(layer, capture, staged, dispatch=dispatch, counts=counts,
                         kda_layer=kda_layer,
                         fallback_active=lambda: modeling.chunk_kimi_delta_attention is fallback)


def test_the_driver_runs_four_forwards_and_stops_before_the_experts():
    layer, dispatch, modeling, fallback, counts = _toy()
    result = _drive(layer, dispatch, modeling, fallback, counts)
    assert list(result["captures"]) == list(census.RUN_ORDER)
    assert layer.mlp.experts.calls == 0
    assert result["count_deltas"]["K1"] == census.KERNEL_FORWARD
    assert result["count_deltas"]["F1"] == census.NO_KERNEL
    checks = group_checks(result["captures"])
    assert all(checks["kernel_repeat"].values()) and all(checks["fallback_repeat"].values())
    assert not checks["kernel_equals_fallback"]["attention_output"]
    assert modeling.chunk_kimi_delta_attention is fallback
    # Every capture is detached: a stopped forward keeps no graph.
    assert all(not value.requires_grad for capture in result["captures"].values()
               for value in capture.values())


def test_a_layer_without_kda_is_a_bitwise_null():
    layer, dispatch, modeling, fallback, counts = _toy(kda=False)
    result = _drive(layer, dispatch, modeling, fallback, counts, kda_layer=False)
    checks = group_checks(result["captures"])
    assert all(checks["kernel_equals_fallback"].values())
    assert result["count_deltas"]["K1"] == census.NO_KERNEL


def test_mutation_a_kernel_that_skips_its_counters_refuses():
    layer, dispatch, modeling, fallback, counts = _toy(count=False)
    with pytest.raises(CensusRefused, match="counters"):
        _drive(layer, dispatch, modeling, fallback, counts)


def test_mutation_a_kda_layer_the_census_thinks_is_not_kda_refuses():
    layer, dispatch, modeling, fallback, counts = _toy()
    with pytest.raises(CensusRefused, match="counters"):
        _drive(layer, dispatch, modeling, fallback, counts, kda_layer=False)


def test_mutation_a_leaking_dispatch_refuses_the_fallback_forward():
    layer, dispatch, modeling, fallback, counts = _toy(leak=True)
    with pytest.raises(CensusRefused, match="not the fallback"):
        _drive(layer, dispatch, modeling, fallback, counts)


def test_mutation_a_nondeterministic_kernel_fails_the_repeat_check():
    layer, dispatch, modeling, fallback, counts = _toy(noise=True)
    result = _drive(layer, dispatch, modeling, fallback, counts)
    checks = group_checks(result["captures"])
    assert not checks["kernel_repeat"]["attention_output"]
    assert all(checks["fallback_repeat"].values())


def test_mutation_a_forward_that_never_reaches_the_experts_refuses():
    layer, dispatch, modeling, fallback, counts = _toy(stop_early=True)
    with pytest.raises(CensusRefused, match="without reaching"):
        _drive(layer, dispatch, modeling, fallback, counts)


def test_layer_capture_refuses_a_foreign_moe():
    layer, *_ = _toy()
    layer.mlp.experts = nn.Identity()
    with pytest.raises(CensusRefused, match="MoE"):
        LayerCapture(layer)


def test_the_driver_output_feeds_the_comparison():
    layer, dispatch, modeling, fallback, counts = _toy()
    captures = _drive(layer, dispatch, modeling, fallback, counts)["captures"]
    bias = layer.mlp.gate.e_score_correction_bias
    arrays = compare_modes(captures["K1"], captures["F1"], bias=bias, routing=ROUTING)["arrays"]
    assert arrays["changed"].shape == (32,)
    summary = summarize(arrays, sequence_length=8, routing=ROUTING)
    assert summary["tokens"] == 32 and summary["sequences"] == 4
    assert summary["flips_beyond_margin_bound"] == 0


# ---------------------------------------------------------------------------
# The data manifest (PrismaBuild's v2 contract, restated) and the spec
# ---------------------------------------------------------------------------

#: prismabuild.core: _DATA_MANIFEST_KEYS | {"read_plan"}, and the plan's keys.
PB_V2_KEYS = {"schema", "produced_by", "mount_prefix", "entries", "entry_count",
              "total_bytes", "annotations", "read_plan"}
PB_ENTRY_KEYS = {"path", "offset", "bytes", "sha256"}
PB_READ_PLAN_KEYS = {"phases", "read_bytes"}
PB_READ_PHASE_KEYS = {"name", "entry_indices", "bytes", "cumulative_bytes"}


def _row(tmp="/mnt/shared/x"):
    model = f"{tmp}/model"
    calibration = {"path": f"{tmp}/calib.safetensors", "sha256": "c" * 64}
    entries = [
        {"path": calibration["path"], "offset": 0, "bytes": 10, "sha256": "c" * 64},
        {"path": f"{tmp}/prepared.json", "offset": 0, "bytes": 20, "sha256": "d" * 64},
        {"path": f"{model}/model-00001.safetensors", "offset": 8, "bytes": 30, "sha256": None},
        {"path": f"{model}/model-00009.safetensors", "offset": 8, "bytes": 40, "sha256": None},
        {"path": f"{model}/model-00008.safetensors", "offset": 8, "bytes": 50, "sha256": None},
    ]
    slice_entries = {}
    for layer in (3, 4):
        slice_entries[str(layer)] = []
        for batch in range(4):
            path = f"{tmp}/boundaries/boundary-{batch}-{layer}-at-{layer}.pt"
            digest = f"{layer}{batch}".ljust(64, "0")
            entries.append({"path": path, "offset": 0, "bytes": 100, "sha256": digest})
            slice_entries[str(layer)].append({"name": f"boundary-{batch}-{layer}-at-{layer}",
                                              "path": path, "file_bytes": 100,
                                              "sha256": digest})
    phases = [{"name": "head", "entry_indices": [0, 1, 2]},
              {"name": "chain-004-source", "entry_indices": [3]},
              {"name": "own-003-source", "entry_indices": [4]}]
    readset = {"schema": census.MANIFEST_SCHEMA, "mount_prefix": "/mnt/shared",
               "entries": entries, "read_plan": {"phases": phases}}
    record = {"layer": 3, "quantum_id": "layer-003", "adjoint": {"chain_layers": [4]}}
    plan = {"model": model, "calibration_input": calibration}
    return readset, {"boundary_entries": slice_entries}, record, plan


def _manifest(layer, readset=None, adjoint_slice=None, record=None, plan=None):
    row = _row()
    readset = readset or row[0]
    return build_census_manifest(
        readset, readset_path="/mnt/shared/x/readset.json.gz", readset_sha256="a" * 64,
        adjoint_slice=adjoint_slice or row[1], slice_sha256="b" * 64,
        record=record or row[2], record_sha256="e" * 64, plan=plan or row[3],
        plan_sha256="f" * 64, layer=layer)


@pytest.mark.parametrize("layer,source", [(4, 3), (3, 4)])
def test_the_manifest_is_prismabuild_v2_and_reads_what_the_census_reads(layer, source):
    manifest = _manifest(layer)
    assert set(manifest) == PB_V2_KEYS and manifest["schema"] == census.MANIFEST_SCHEMA
    assert "phases" not in manifest["annotations"]
    assert all(set(entry) == PB_ENTRY_KEYS for entry in manifest["entries"])
    keys = [(entry["path"], entry["offset"]) for entry in manifest["entries"]]
    assert len(keys) == len(set(keys))
    assert manifest["entry_count"] == len(manifest["entries"])
    assert manifest["total_bytes"] == sum(entry["bytes"] for entry in manifest["entries"])
    plan = manifest["read_plan"]
    assert set(plan) == PB_READ_PLAN_KEYS
    used, cumulative = set(), 0
    for phase in plan["phases"]:
        assert set(phase) == PB_READ_PHASE_KEYS
        assert len(phase["entry_indices"]) == len(set(phase["entry_indices"]))
        size = sum(manifest["entries"][index]["bytes"] for index in phase["entry_indices"])
        cumulative += size
        assert phase["bytes"] == size and phase["cumulative_bytes"] == cumulative
        used.update(phase["entry_indices"])
    assert used == set(range(len(manifest["entries"]))) and plan["read_bytes"] == cumulative
    names = [phase["name"] for phase in plan["phases"]]
    assert names == list(census.phase_names(layer))
    head = [manifest["entries"][index]["path"] for index in plan["phases"][0]["entry_indices"]]
    # The calibration tokens and the model's head shard; not prepared.json.
    assert head == ["/mnt/shared/x/calib.safetensors", "/mnt/shared/x/model/model-00001.safetensors"]
    source_entry = manifest["entries"][plan["phases"][1]["entry_indices"][0]]
    assert source_entry["bytes"] == {3: 40, 4: 50}[source]
    bound = [manifest["entries"][index]["path"] for index in plan["phases"][2]["entry_indices"]]
    assert bound == [f"/mnt/shared/x/boundaries/boundary-{b}-{layer}-at-{layer}.pt"
                     for b in range(4)]


def test_the_manifest_refuses_a_boundary_the_readset_reads_differently():
    readset, adjoint_slice, record, plan = _row()
    readset["entries"][-1]["sha256"] = "9" * 64
    with pytest.raises(CensusRefused, match="not the slice's"):
        _manifest(4, readset, adjoint_slice, record, plan)


def test_the_manifest_refuses_a_layer_outside_the_row():
    with pytest.raises(CensusRefused, match="neither"):
        _manifest(5)


def test_the_census_spec_drops_what_the_census_does_not_write():
    spec = {"container": {"image": "i", "mounts": [
        {"source": "/mnt/shared", "target": "/mnt/shared", "readonly": False},
        {"source": "/s/spill", "target": "/s/spill", "readonly": False},
        {"source": "/s/spool", "target": "/s/spool", "readonly": False},
        {"source": "/s/cot", "target": "/s/cot", "readonly": False}]},
        "env": {"PRISMAQUANT_STAGE_B_SPILL_ROOT": "/s/spill",
                "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES": "9",
                "PRISMABUILD_PRODUCED_SPOOL_ROOT": "/s/spool",
                "PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES": "9",
                "PRISMAQUANT_STAGE_B_REPLAY_REGIME": "capture_batch=4",
                "PRISMAQUANT_STAGE_B_COTANGENT_ROOT": "/s/cot",
                "PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES": "34359738368",
                "PRISMAQUANT_DEV_MODE": "1"},
        "cpu_memory_gb": 28}
    derived, changes = derive_census_spec(spec, cotangent_max_bytes=1 << 30)
    assert [mount["source"] for mount in derived["container"]["mounts"]] == [
        "/mnt/shared", "/s/cot"]
    assert derived["env"] == {"PRISMAQUANT_STAGE_B_COTANGENT_ROOT": "/s/cot",
                              "PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES": str(1 << 30),
                              "PRISMAQUANT_DEV_MODE": "1"}
    assert derived["cpu_memory_gb"] == 28 and spec["env"]["PRISMAQUANT_STAGE_B_SPILL_ROOT"]
    assert len(changes) == 8
