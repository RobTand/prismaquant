"""derive_per_expert_activations routes every expert at once (PQ #1931).

The per-expert loop it replaced called ``torch.where`` once per expert, a
host sync each time: 288 per GLM-5.3 MoE layer per calibration sample. The
vectorized routing must reproduce that loop tensor for tensor, including
dtype, shape, device, row order and the seeded subsample.
``_reference_derive`` below is the loop verbatim, from
``prismaquant/measure_quant_cost.py`` at f5869e0c (lines 1206-1296); only
its name changed.
"""
from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from prismaquant import measure_quant_cost as mqc
from prismaquant.measure_quant_cost import _packed_experts_router, _packed_router_topk


def _reference_derive(
    experts_mod: nn.Module,
    X: torch.Tensor,
    parent_mod: nn.Module | None,
    *,
    capture_down: bool = True,
    max_rows_per_expert: int | None = None,
    subsample_seed: int = 1234,
) -> dict:
    """Single source of truth for per-expert GPTQ activations.

    Routes the module-level expert input ``X`` ([*, hidden]) through the MoE
    block's own router and collects, per expert ``e``, exactly the tensors the
    per-expert GPTQ Hessian needs — identical to the routed forward in
    ``_packed_experts_forward_with_weights``, but COLLECTING activations instead
    of producing the output. Shared by the cost path and the export render path
    so the routing + SwiGLU derivation lives in ONE place (no duplication).

    Returns a dict of length-E lists:
      - ``gate_up``: each [n_e, hidden]  — the routed input to ``gate_up_proj``.
      - ``down``:    each [n_e, inter]   — the post-SwiGLU input to ``down_proj``
                     (``_apply_gate(gate_up)`` when present, else ``act_fn(gate)*up``).
                     Empty list when ``capture_down=False``.
      - ``gate_weights``: each [n_e]     — the router weight per routed token.
      - ``row_counts``: list[int]        — routed tokens/expert BEFORE subsample
                     (use for the fail-on-insufficient-routed-rows gate).

    Subsampling (``max_rows_per_expert``) is deterministic (fixed ``subsample_seed``)
    so the render is reproducible. ``None`` keeps every routed row. Raises (never
    silently degrades) if the router / act_fn cannot be resolved.
    """
    gate_up_w = getattr(experts_mod, "gate_up_proj", None)
    if gate_up_w is None:
        raise ValueError("packed experts module lacks gate_up_proj")
    num_experts = int(getattr(experts_mod, "num_experts", gate_up_w.size(0)))
    act_fn = getattr(experts_mod, "act_fn", None)
    apply_gate = getattr(experts_mod, "_apply_gate", None)
    router = _packed_experts_router(parent_mod)
    if router is None:
        raise ValueError("no router found for packed-experts module")
    Xf = X.reshape(-1, X.size(-1))
    dev, dt, hidden = Xf.device, Xf.dtype, Xf.size(-1)
    inter = gate_up_w.size(1) // 2
    with torch.no_grad():
        route_fn = getattr(parent_mod, "route_tokens_to_experts", None)
        if callable(route_fn):
            top_k_index, top_k_weights = route_fn(router(Xf))
        else:
            top_k_index, top_k_weights = _packed_router_topk(
                router, Xf,
                e_score_correction_bias=getattr(
                    parent_mod, "e_score_correction_bias", None),
                expert_bias=getattr(parent_mod, "expert_bias", None),
            )
        expert_mask = F.one_hot(
            top_k_index.to(torch.long), num_classes=num_experts).permute(2, 1, 0)
    gate_up_list: list[torch.Tensor] = []
    down_list: list[torch.Tensor] = []
    gw_list: list[torch.Tensor] = []
    counts: list[int] = []
    for e in range(num_experts):
        top_k_pos, token_idx = torch.where(expert_mask[e])
        n = int(token_idx.numel())
        counts.append(n)
        if n == 0:
            gate_up_list.append(torch.empty(0, hidden, device=dev, dtype=dt))
            gw_list.append(torch.empty(0, device=dev, dtype=dt))
            if capture_down:
                down_list.append(torch.empty(0, inter, device=dev, dtype=dt))
            continue
        if max_rows_per_expert is not None and n > max_rows_per_expert:
            gen = torch.Generator(device=dev).manual_seed(subsample_seed + e)
            keep = torch.randperm(n, device=dev, generator=gen)[:max_rows_per_expert]
            token_idx, top_k_pos = token_idx[keep], top_k_pos[keep]
        Xe = Xf[token_idx]
        gate_up_list.append(Xe)
        gw_list.append(top_k_weights[token_idx, top_k_pos])
        if capture_down:
            with torch.no_grad():
                gate_up = F.linear(Xe, gate_up_w[e])
                if callable(apply_gate):
                    di = apply_gate(gate_up)
                elif act_fn is not None:
                    g, u = gate_up.chunk(2, dim=-1)
                    di = act_fn(g) * u
                else:
                    raise ValueError(
                        "packed experts module exposes neither _apply_gate nor act_fn")
            down_list.append(di)
    return {"gate_up": gate_up_list, "down": down_list,
            "gate_weights": gw_list, "row_counts": counts}


HIDDEN, INTER = 12, 6


class _FixedRoute(nn.Module):
    """A parent whose routing is given, so a test can place every pair."""

    def __init__(self, top_k_index, top_k_weights):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, 4, bias=False)
        self._index = top_k_index
        self._weights = top_k_weights

    def route_tokens_to_experts(self, _logits):
        return self._index, self._weights


class _TopkRouter(nn.Module):
    """A sigmoid top-k router shaped like GLM-5.3's (logits, weights, indices)."""

    def __init__(self, num_experts, top_k):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, HIDDEN))
        self.top_k = top_k

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.float(), self.weight.float())
        scores = logits.sigmoid()
        index = torch.topk(scores, self.top_k, dim=-1, sorted=False)[1]
        weights = scores.gather(1, index)
        return logits, weights / weights.sum(-1, keepdim=True), index


class _RouterParent(nn.Module):
    def __init__(self, num_experts, top_k):
        super().__init__()
        self.gate = _TopkRouter(num_experts, top_k)


class _GateExperts(nn.Module):
    """Experts with a model-specific ``_apply_gate`` (the GLM-5.3 path)."""

    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * INTER, HIDDEN))

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate.clamp(max=0.5)) * up.clamp(-0.5, 0.5)


class _ActExperts(nn.Module):
    """Experts that expose only ``act_fn``."""

    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * INTER, HIDDEN))
        self.act_fn = nn.SiLU()


def _devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _fixed_case(kind, device, dtype):
    """A routing with duplicates, multi-slot experts and zero-row experts."""
    gen = torch.Generator().manual_seed(1931)
    num_experts, top_k = 16, 4
    if kind == "mixed":
        tokens = 13
        # Experts 12..15 are never routed to; expert 3 appears twice in token 0.
        index = torch.randint(0, 12, (tokens, top_k), generator=gen)
        index[0, :2] = 3
    elif kind == "one_expert":
        tokens = 9
        index = torch.full((tokens, top_k), 5)
    elif kind == "no_tokens":
        tokens = 0
        index = torch.empty(0, top_k, dtype=torch.long)
    else:
        raise AssertionError(kind)
    weights = torch.rand(tokens, top_k, generator=gen, dtype=torch.float32)
    X = torch.randn(tokens, HIDDEN, generator=gen).to(dtype)
    return num_experts, X.to(device), index.to(device), weights.to(device)


def _same(a, b):
    """Tensor-for-tensor equality, plus the metadata torch.equal ignores."""
    assert a.keys() == b.keys()
    assert a["row_counts"] == b["row_counts"]
    for key in ("gate_up", "down", "gate_weights"):
        assert len(a[key]) == len(b[key]), key
        for e, (x, y) in enumerate(zip(a[key], b[key])):
            assert x.dtype == y.dtype, (key, e, x.dtype, y.dtype)
            assert x.shape == y.shape, (key, e, x.shape, y.shape)
            assert x.device == y.device, (key, e)
            assert torch.equal(x, y), (key, e)


def _first_difference(a, b):
    try:
        _same(a, b)
    except AssertionError as error:
        return error
    return None


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("routing", ["mixed", "one_expert", "no_tokens"])
@pytest.mark.parametrize("experts_cls", [_GateExperts, _ActExperts])
@pytest.mark.parametrize("capture_down", [True, False])
@pytest.mark.parametrize("max_rows", [None, 3, 1000])
def test_fixed_routing_matches_the_per_expert_loop(
        device, dtype, routing, experts_cls, capture_down, max_rows):
    num_experts, X, index, weights = _fixed_case(routing, device, dtype)
    torch.manual_seed(0)
    experts = experts_cls(num_experts).to(device=device, dtype=dtype)
    parent = _FixedRoute(index, weights).to(device=device, dtype=dtype)
    kwargs = dict(capture_down=capture_down, max_rows_per_expert=max_rows,
                  subsample_seed=77)
    want = _reference_derive(experts, X, parent, **kwargs)
    got = mqc.derive_per_expert_activations(experts, X, parent, **kwargs)
    _same(got, want)
    if routing == "mixed":
        assert 0 in got["row_counts"], "the case must include zero-row experts"
    if max_rows == 3 and routing != "no_tokens":
        assert max(got["row_counts"]) > 3, "the case must subsample"
        assert max(t.shape[0] for t in got["gate_up"]) == 3


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("experts_cls", [_GateExperts, _ActExperts])
@pytest.mark.parametrize("capture_down", [True, False])
@pytest.mark.parametrize("max_rows", [None, 5])
def test_router_path_matches_the_per_expert_loop(device, experts_cls, capture_down, max_rows):
    """The ``_packed_router_topk`` path, on a [batch, seq, hidden] input."""
    torch.manual_seed(5)
    num_experts, top_k = 32, 8
    experts = experts_cls(num_experts).to(device)
    parent = _RouterParent(num_experts, top_k).to(device)
    X = torch.randn(2, 37, HIDDEN, device=device)
    kwargs = dict(capture_down=capture_down, max_rows_per_expert=max_rows)
    want = _reference_derive(experts, X, parent, **kwargs)
    got = mqc.derive_per_expert_activations(experts, X, parent, **kwargs)
    _same(got, want)
    assert sum(got["row_counts"]) == 2 * 37 * top_k


def _token_major_order(top_k_index, num_experts):
    """A plausible bug: group by expert, but in (token, slot) order."""
    tokens, top_k = top_k_index.shape
    pair_expert = top_k_index.reshape(-1)
    flat = torch.argsort(pair_expert, stable=True)
    order = (flat % top_k) * tokens + flat // top_k
    return order, torch.bincount(pair_expert, minlength=num_experts).tolist()


@pytest.mark.parametrize("max_rows", [None, 3])
def test_the_equality_check_catches_a_wrong_row_order(monkeypatch, max_rows):
    """Mutate the routing, not the fixture: the comparison must fail."""
    num_experts, X, index, weights = _fixed_case("mixed", "cpu", torch.float32)
    torch.manual_seed(0)
    experts = _GateExperts(num_experts)
    parent = _FixedRoute(index, weights)
    kwargs = dict(capture_down=True, max_rows_per_expert=max_rows)
    want = _reference_derive(experts, X, parent, **kwargs)
    assert _first_difference(
        mqc.derive_per_expert_activations(experts, X, parent, **kwargs), want) is None
    monkeypatch.setattr(mqc, "_routed_pair_order", _token_major_order)
    broken = mqc.derive_per_expert_activations(experts, X, parent, **kwargs)
    assert broken["row_counts"] == want["row_counts"], "same membership, wrong order"
    assert _first_difference(broken, want) is not None


def test_an_out_of_range_expert_index_refuses():
    num_experts, X, index, weights = _fixed_case("mixed", "cpu", torch.float32)
    index = index.clone()
    index[2, 1] = num_experts
    parent = _FixedRoute(index, weights)
    with pytest.raises(ValueError, match="outside"):
        mqc.derive_per_expert_activations(_GateExperts(num_experts), X, parent)


def test_experts_without_a_gate_or_act_fn_still_refuse():
    num_experts, X, index, weights = _fixed_case("mixed", "cpu", torch.float32)
    experts = _ActExperts(num_experts)
    del experts.act_fn
    with pytest.raises(ValueError, match="neither _apply_gate nor act_fn"):
        mqc.derive_per_expert_activations(experts, X, _FixedRoute(index, weights))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="counts CUDA host syncs")
def test_routing_syncs_the_host_once_not_once_per_expert():
    """288 experts, GLM-5.3's count: one sync to read the counts, not 288."""
    torch.manual_seed(3)
    num_experts, top_k, tokens = 288, 8, 512
    index = torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(tokens)]).cuda()
    weights = torch.rand(tokens, top_k, device="cuda")
    X = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    experts = _GateExperts(num_experts).cuda().to(torch.bfloat16)
    # The router runs in the input's dtype, as the census's bf16 layers do.
    parent = _FixedRoute(index, weights).cuda().to(torch.bfloat16)
    for derive, ceiling in ((mqc.derive_per_expert_activations, 1),
                            (_reference_derive, num_experts)):
        derive(experts, X, parent)  # warm up
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                derive(experts, X, parent)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        syncs = sum("synchroniz" in str(w.message) for w in caught)
        if derive is _reference_derive:
            assert syncs >= num_experts, syncs  # the instrument sees the old loop
        else:
            assert syncs <= ceiling, syncs
