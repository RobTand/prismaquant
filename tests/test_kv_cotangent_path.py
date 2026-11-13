"""The KV-cotangent path: shared-state Fisher, measured not modelled.

Issue #9 item 2. Gemma4 shares K/V across same-type layers: a *storing* layer
computes K/V and writes them into the per-forward-pass ``shared_kv_states``
dict; later *sharing* layers read those tensors instead of computing their own
(``modeling_gemma4.Gemma4TextAttention.forward``). PrismaQuant's phase-3
forwards each layer in ISOLATION from a phase-1 capture, and that capture is
detached — so a sharing layer's backward stopped dead at the borrowed K/V and
the storing layer's ``h_trace`` for ``k_proj``/``v_proj`` counted only its own
layer's gradient, missing the gradient flowing through EVERY layer that
consumes its K/V. ``incremental_probe.kv_shared_fisher_block_reason`` blocked
the probe over that under-count (MINOR-M33).

``SharedStateCotangents`` closes it: borrowed tensors are grafted as
grad-enabled leaves, each consumer's ``leaf.grad`` is summed per source, and
the producing layer's backward is driven with that sum alongside its own output
cotangent.

**The load-bearing test is equivalence, not difference.** A protocol that
merely produces a *bigger* number could be bigger for the wrong reason, so
these tests compute the same Fisher two ways on the same synthetic weights —
(a) one end-to-end autograd backward over the composed model, (b) the isolated
phase-3 reverse sweep with cotangent accumulation — and require them to agree,
using the shipped ``FisherAccumulator`` for both arms so only the cotangent
differs. Then they show the pre-fix protocol is strictly smaller.

No GPU, no Gemma4 weights, no checkpoint: synthetic modules whose contract
mirrors the real one (unconditional read, int keys, several consumers per
source, ``shared_kv_states`` threaded by ``_call_layer``), driven through the
REAL ``Gemma4Profile`` shared-state hooks.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.layer_streaming import _call_layer
from prismaquant.model_profiles.gemma4 import Gemma4Profile
from prismaquant.model_profiles.qwen3 import Qwen3Profile
from prismaquant.sensitivity_probe import (
    FisherAccumulator,
    SharedStateCotangents,
    kv_cotangent_path_enabled,
)

DTYPE = torch.float64  # fp64 so "equivalent" can be checked, not eyeballed


# --------------------------------------------------------------------------
# A synthetic model that mirrors Gemma4's cross-layer KV-sharing contract
# --------------------------------------------------------------------------
class _ToyAttention(nn.Module):
    """Mirrors `Gemma4TextAttention`: the sharing layer owns no k/v_proj and
    does an UNCONDITIONAL `shared_kv_states[kv_shared_layer_index]` lookup; the
    storing layer writes `shared_kv_states[layer_idx] = (k, v)` with no detach,
    gated by `store_full_length_kv` exactly as the real module is."""

    def __init__(self, layer_idx: int, hidden: int, *,
                 shares_from: int | None = None, stores: bool = False,
                 k_eq_v: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kv_shared_layer = shares_from is not None
        self.kv_shared_layer_index = shares_from
        self.store_full_length_kv = stores
        self.q_proj = nn.Linear(hidden, hidden, bias=False, dtype=DTYPE)
        if not self.is_kv_shared_layer:
            self.k_proj = nn.Linear(hidden, hidden, bias=False, dtype=DTYPE)
            # `use_alternative_attention` layers have no v_proj and reuse the
            # key tensor as the value.
            self.v_proj = (None if k_eq_v else
                           nn.Linear(hidden, hidden, bias=False, dtype=DTYPE))
        self.o_proj = nn.Linear(hidden, hidden, bias=False, dtype=DTYPE)
        self.scaling = 1.0 / math.sqrt(hidden)

    def forward(self, hidden_states, shared_kv_states):
        q = self.q_proj(hidden_states)
        if self.is_kv_shared_layer:
            k, v = shared_kv_states[self.kv_shared_layer_index]
            # "Device of past layer may be different from current one" — the
            # real module's `.to(device)`, which is what keeps the cotangent
            # flowing back to a host-resident capture.
            k = k.to(q.device)
            v = v.to(q.device)
        else:
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states) if self.v_proj is not None else k
        if self.store_full_length_kv:
            shared_kv_states[self.layer_idx] = (k, v)
        scores = torch.softmax(q @ k.transpose(-1, -2) * self.scaling, dim=-1)
        return self.o_proj(scores @ v)


class _ToyLayer(nn.Module):
    def __init__(self, layer_idx: int, hidden: int, *,
                 shares_from: int | None = None, stores: bool = False,
                 k_eq_v: bool = False):
        super().__init__()
        self.self_attn = _ToyAttention(layer_idx, hidden,
                                       shares_from=shares_from, stores=stores,
                                       k_eq_v=k_eq_v)
        self.mlp = nn.Linear(hidden, hidden, bias=False, dtype=DTYPE)

    def forward(self, *, hidden_states, shared_kv_states=None, **kw):
        # Defaulted exactly like `Gemma4TextDecoderLayer.forward`, so a profile
        # that declares no shared state can still call this layer.
        h = hidden_states + self.self_attn(hidden_states, shared_kv_states)
        return h + torch.tanh(self.mlp(h))


class _ToyModel(nn.Module):
    """`layer_specs` is a list of `(shares_from, stores)` per layer."""

    def __init__(self, layer_specs, *, hidden=6, vocab=11, seed=0,
                 k_eq_v=False):
        super().__init__()
        torch.manual_seed(seed)
        self.embed_tokens = nn.Embedding(vocab, hidden, dtype=DTYPE)
        self.layers = nn.ModuleList([
            _ToyLayer(i, hidden, shares_from=s, stores=st, k_eq_v=k_eq_v)
            for i, (s, st) in enumerate(layer_specs)
        ])
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, dtype=DTYPE)
        self.lm_head = nn.Linear(hidden, vocab, bias=False, dtype=DTYPE)


# Gemma4's real topology in miniature: a plain layer BELOW the producer (so the
# chained `grad_out` the severed edge truncated is exercised too), the storing
# layer, then TWO consumers of its K/V (so "the cotangents of several consumers
# add" is exercised, not assumed).
SHARED_SPECS = [(None, False), (None, True), (1, False), (1, False)]
PRODUCER = 1
NO_SHARING_SPECS = [(None, False), (None, False), (None, False)]


def _body_linears(model: _ToyModel) -> list[str]:
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.startswith("layers.")]


def _loss(model: _ToyModel, hidden, ids):
    """Teacher-forced CE, as phase-2 computes it."""
    logits = model.lm_head(model.norm(hidden))
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1),
        reduction="sum",
    )


def _fisher(model):
    return FisherAccumulator(model, tracked=_body_linears(model),
                             expert_info={}, hook_packed_experts=False)


def _calib(model: _ToyModel, n=2, t=5, seed=3):
    g = torch.Generator().manual_seed(seed)
    vocab = model.lm_head.out_features
    return torch.randint(0, vocab, (n, t), generator=g)


def _call(layer, hidden, pass_state):
    return _call_layer(layer, hidden, position_embeddings=None,
                       attention_mask=None, position_ids=None,
                       pass_state=pass_state)


# --------------------------------------------------------------------------
# Arm (a): one end-to-end autograd backward over the composed model
# --------------------------------------------------------------------------
def run_baseline(model: _ToyModel, ids, profile):
    """The ground truth: nothing is isolated, nothing is detached, so every
    consumer's gradient reaches the storing layer's k/v_proj by construction.
    Returns (h_trace per Linear, {source_idx: (gy_k, gy_v)})."""
    fisher = _fisher(model)
    pass_state = profile.new_forward_pass_state()
    hidden = model.embed_tokens(ids)
    for layer in model.layers:
        hidden = _call(layer, hidden, pass_state)
    # Retain grad on the shared tensors so the test can read the TOTAL
    # cotangent at the producer's k_proj/v_proj output (own layer + every
    # consumer) — the exact `gy` h_trace is built from.
    shared = pass_state.get("shared_kv_states", {})
    for kv in shared.values():
        for t in kv:
            t.retain_grad()
    _loss(model, hidden, ids).backward()
    gy = {idx: tuple(t.grad.detach().clone() for t in kv)
          for idx, kv in shared.items()}
    fisher.finalize(None, ids.numel())
    traces = {n: s["h_trace"] for n, s in fisher.stats.items()}
    fisher.remove_hooks()
    model.zero_grad(set_to_none=True)
    return traces, gy


# --------------------------------------------------------------------------
# Arm (b): the isolated phase-3 protocol, with and without the fix
# --------------------------------------------------------------------------
def run_protocol(model: _ToyModel, ids, profile, *, cotangents: bool):
    """A faithful miniature of the shipped phase-3 path: phase-1 streams the
    layers under no_grad and captures activations + shared state; phase-2 takes
    CE grad at the tail; phase-3 sweeps in REVERSE, re-forwarding each layer in
    isolation from its captured input and chaining `x_in.grad` upward.

    Mirrors `incremental_probe._run_shard_body`'s phase-3 call sequence:
    graft -> forward -> produced_roots -> backward -> harvest.
    """
    # ---- phase 1 ----
    pass_state = profile.new_forward_pass_state()
    with torch.no_grad():
        hidden = model.embed_tokens(ids)
        acts = [hidden]
        for layer in model.layers:
            hidden = _call(layer, hidden, pass_state)
            acts.append(hidden)
    captured = profile.capture_forward_pass_state(pass_state)

    # ---- phase 2 (hooks off: the body Linears are not in this backward) ----
    tail = acts[-1].detach().clone().requires_grad_(True)
    _loss(model, tail, ids).backward()
    grad_out = tail.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    # ---- phase 3 ----
    fisher = _fisher(model)
    cot = SharedStateCotangents(enabled=cotangents)
    total_gy: dict[int, tuple] = {}
    for L in reversed(range(len(model.layers))):
        x_in = acts[L].detach().clone().requires_grad_(True)
        state = cot.graft(profile.isolated_layer_pass_state(
            captured, model.layers[L]))
        out = _call(model.layers[L], x_in, state)
        roots, grads = cot.produced_roots()
        # Test-only: read the SUMMED gy at whatever this layer PRODUCED (own
        # path + any seeded consumer cotangent), deduplicated by identity so a
        # k_eq_v layer reports one tensor.
        produced: list[torch.Tensor] = []
        for t in ((state or {}).get("shared_kv_states", {}).get(L) or ()):
            if t.grad_fn is not None and all(t is not p for p in produced):
                t.retain_grad()
                produced.append(t)
        if roots:
            torch.autograd.backward([out, *roots], [grad_out, *grads])
        else:
            out.backward(grad_out)
        if produced:
            total_gy[L] = tuple(
                (t.grad.detach().clone() if t.grad is not None
                 else torch.zeros_like(t)) for t in produced)
        cot.harvest()
        grad_out = x_in.grad.detach().clone()
    fisher.finalize(None, ids.numel())
    traces = {n: s["h_trace"] for n, s in fisher.stats.items()}
    fisher.remove_hooks()
    model.zero_grad(set_to_none=True)
    return traces, total_gy, cot


def _protocol_no_fix_gy(model, ids, profile):
    """The pre-fix protocol's `gy` at the producer's K/V — measured by
    retaining grad on what the storing layer wrote, with no seeding."""
    pass_state = profile.new_forward_pass_state()
    with torch.no_grad():
        hidden = model.embed_tokens(ids)
        acts = [hidden]
        for layer in model.layers:
            hidden = _call(layer, hidden, pass_state)
            acts.append(hidden)
    captured = profile.capture_forward_pass_state(pass_state)
    tail = acts[-1].detach().clone().requires_grad_(True)
    _loss(model, tail, ids).backward()
    grad_out = tail.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    out_gy = {}
    for L in reversed(range(len(model.layers))):
        x_in = acts[L].detach().clone().requires_grad_(True)
        state = profile.isolated_layer_pass_state(captured, model.layers[L])
        out = _call(model.layers[L], x_in, state)
        produced = (state or {}).get("shared_kv_states", {}).get(L)
        if produced is not None:
            for t in produced:
                t.retain_grad()
        out.backward(grad_out)
        if produced is not None:
            out_gy[L] = tuple(t.grad.detach().clone() for t in produced)
        grad_out = x_in.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    return out_gy


# --------------------------------------------------------------------------
# THE acceptance criterion
# --------------------------------------------------------------------------
def test_cotangent_matches_an_end_to_end_backward_exactly():
    """(a) == (b) for the cotangent itself, in fp64.

    This is the assertion that makes the accounting *right* rather than merely
    *different*: the summed gradient arriving at the storing layer's
    k_proj/v_proj output during the isolated backward is the same tensor an
    end-to-end backward delivers there.
    """
    model = _ToyModel(SHARED_SPECS)
    ids = _calib(model)
    profile = Gemma4Profile()

    _, baseline_gy = run_baseline(model, ids, profile)
    _, protocol_gy, cot = run_protocol(model, ids, profile, cotangents=True)

    assert set(baseline_gy) == {PRODUCER} == set(protocol_gy)
    for pos in (0, 1):  # k, v
        base = baseline_gy[PRODUCER][pos]
        prot = protocol_gy[PRODUCER][pos]
        rel = ((prot - base).norm() / base.norm()).item()
        assert base.norm().item() > 0, "degenerate fixture: zero cotangent"
        assert rel < 1e-12, f"position {pos}: relative cotangent error {rel:.3e}"
    # Every consumer's cotangent was claimed by its producer.
    assert cot.pending_keys() == []
    assert cot.n_grafted == 4 and cot.n_seeded == 2  # 2 consumers x (k, v)


def test_h_trace_equivalence_and_the_size_of_the_undercount():
    """(a) == (b) on the shipped `h_trace`, and the pre-fix protocol is
    strictly smaller on exactly the two Linears that feed other layers."""
    model = _ToyModel(SHARED_SPECS)
    ids = _calib(model)
    profile = Gemma4Profile()

    base, _ = run_baseline(model, ids, profile)
    fixed, _, _ = run_protocol(model, ids, profile, cotangents=True)
    unfixed, _, cot_off = run_protocol(model, ids, profile, cotangents=False)

    producers = [f"layers.{PRODUCER}.self_attn.k_proj",
                 f"layers.{PRODUCER}.self_attn.v_proj"]
    assert set(base) == set(fixed) == set(unfixed)

    for name in sorted(base):
        b, f = base[name], fixed[name]
        rel = abs(f - b) / abs(b)
        # fp32 inside FisherAccumulator's squaring is the only slack.
        assert rel < 1e-6, f"{name}: baseline {b!r} vs protocol {f!r} ({rel:.2e})"

    for name in producers:
        assert unfixed[name] < base[name] * (1 - 1e-3), (
            f"{name}: pre-fix protocol {unfixed[name]!r} is not strictly "
            f"smaller than the true {base[name]!r} — fixture too weak to "
            "demonstrate the under-count")
    # Disabling the path really did sever it (no leaves, no seeds).
    assert cot_off.n_grafted == 0 and cot_off.n_seeded == 0

    # Report the numbers the fix is worth, for the record.
    for name in producers:
        print(f"[kv-cotangent] {name}: end-to-end {base[name]:.6e}  "
              f"protocol+fix {fixed[name]:.6e} (rel err "
              f"{abs(fixed[name] - base[name]) / abs(base[name]):.2e})  "
              f"protocol-no-fix {unfixed[name]:.6e}  under-count "
              f"{100 * (1 - unfixed[name] / base[name]):.1f}%")


def test_producer_backward_hook_fires_exactly_once_with_the_total():
    """The silent-halving trap. Every Fisher hook in this repo POPs its saved
    forward input (`saved_inputs.pop(name)` in `incremental_probe`'s phase-3
    hooks, `self._saved_inputs.pop` in `FisherAccumulator`), so a
    `full_backward_hook` that fired twice — once per backward root — would
    accumulate the first gradient and silently drop the second. Autograd only
    runs a node after every contribution has arrived, which is exactly why the
    two roots go into ONE `torch.autograd.backward` call; pin it."""
    model = _ToyModel(SHARED_SPECS)
    ids = _calib(model)
    profile = Gemma4Profile()

    calls: dict[str, int] = {}
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.startswith("layers."):
            calls[name] = 0

            def hook(_m, _gi, _go, _n=name):
                calls[_n] += 1

            handles.append(mod.register_full_backward_hook(hook))
    try:
        _, _, cot = run_protocol(model, ids, profile, cotangents=True)
    finally:
        for h in handles:
            h.remove()

    assert cot.n_seeded == 2, "fixture did not exercise the two-root backward"
    assert set(calls.values()) == {1}, (
        "a Linear's backward hook fired more than once under the multi-root "
        f"backward — Fisher would be silently dropped: {calls}")


def test_undercount_also_truncated_the_chained_input_gradient():
    """The severed cotangent did not only hit k/v_proj. Phase-3 chains
    `grad_out = x_in.grad` down the stack, and the producing layer's input
    gradient was missing the shared-K/V paths too — so EVERY layer below it
    inherited a truncated cotangent. Pin both halves: the layers below move,
    the consumers and the producer's own q/o/mlp do not."""
    model = _ToyModel(SHARED_SPECS)
    ids = _calib(model)
    profile = Gemma4Profile()

    base, _ = run_baseline(model, ids, profile)
    unfixed, _, _ = run_protocol(model, ids, profile, cotangents=False)

    below = [n for n in base if n.startswith("layers.0.")]
    assert below
    for name in below:
        rel = abs(unfixed[name] - base[name]) / abs(base[name])
        assert rel > 1e-3, (
            f"{name}: sits below the producer but the severed cotangent left "
            f"it unchanged ({rel:.2e}) — fixture too weak")

    # The consumers, and the producer's own paths that never crossed the
    # severed edge, are untouched: the bias is targeted, not diffuse noise.
    for name in sorted(base):
        if name.startswith(("layers.2.", "layers.3.")) or name.endswith(
                (f"layers.{PRODUCER}.self_attn.q_proj",
                 f"layers.{PRODUCER}.self_attn.o_proj",
                 f"layers.{PRODUCER}.mlp")):
            assert abs(unfixed[name] - base[name]) / abs(base[name]) < 1e-6, name


def test_pre_fix_gy_is_the_baseline_minus_the_consumers():
    """Decomposition check: the pre-fix protocol measured the producer's OWN
    contribution, and the seeded roots supply exactly the missing remainder.
    (Additivity of the cotangent is why one reverse pass is enough.)"""
    model = _ToyModel(SHARED_SPECS)
    ids = _calib(model)
    profile = Gemma4Profile()

    _, baseline_gy = run_baseline(model, ids, profile)
    own_gy = _protocol_no_fix_gy(model, ids, profile)

    # Re-run the sweep and capture the seeds handed to the producer.
    seeds: list[torch.Tensor] = []
    pass_state = profile.new_forward_pass_state()
    with torch.no_grad():
        hidden = model.embed_tokens(ids)
        acts = [hidden]
        for layer in model.layers:
            hidden = _call(layer, hidden, pass_state)
            acts.append(hidden)
    captured = profile.capture_forward_pass_state(pass_state)
    tail = acts[-1].detach().clone().requires_grad_(True)
    _loss(model, tail, ids).backward()
    grad_out = tail.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    cot = SharedStateCotangents()
    for L in reversed(range(len(model.layers))):
        x_in = acts[L].detach().clone().requires_grad_(True)
        state = cot.graft(profile.isolated_layer_pass_state(
            captured, model.layers[L]))
        out = _call(model.layers[L], x_in, state)
        roots, grads = cot.produced_roots()
        if roots:
            seeds = [g.detach().clone() for g in grads]
            torch.autograd.backward([out, *roots], [grad_out, *grads])
        else:
            out.backward(grad_out)
        cot.harvest()
        grad_out = x_in.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    assert len(seeds) == 2
    for pos in (0, 1):
        total = baseline_gy[PRODUCER][pos]
        recomposed = own_gy[PRODUCER][pos] + seeds[pos]
        rel = ((recomposed - total).norm() / total.norm()).item()
        assert rel < 1e-12, f"position {pos}: own + consumers != total ({rel:.2e})"
        # ... and the consumers' share is not noise.
        share = (seeds[pos].norm() / total.norm()).item()
        assert share > 1e-2, f"position {pos}: consumer share {share:.2e}"


# --------------------------------------------------------------------------
# Architectures without shared state must be untouched
# --------------------------------------------------------------------------
def test_no_kv_sharing_is_bit_for_bit_unaffected():
    """Same model, same sweep, accumulator on vs off: identical `h_trace` to
    the last bit, no leaves grafted, and `graft` hands back the caller's own
    object so the layer call is the one it always was."""
    model = _ToyModel(NO_SHARING_SPECS)
    ids = _calib(model)
    for profile in (Gemma4Profile(), Qwen3Profile()):
        on, _, cot_on = run_protocol(model, ids, profile, cotangents=True)
        off, _, cot_off = run_protocol(model, ids, profile, cotangents=False)
        assert on == off, f"{type(profile).__name__}: h_trace moved"
        assert cot_on.n_grafted == cot_on.n_seeded == 0
        assert cot_on.pending_keys() == [] and cot_off.pending_keys() == []

    # Qwen3 declares no shared state at all: the same dict object comes back,
    # so `_call_layer` receives byte-for-byte what it received before.
    cot = SharedStateCotangents()
    empty = Qwen3Profile().isolated_layer_pass_state(None, object())
    assert cot.graft(empty) is empty
    assert cot.graft(None) is None
    assert cot.produced_roots() == ([], [])
    assert cot.n_grafted == 0


def test_a_storing_layer_with_no_consumers_seeds_nothing():
    """A model whose layers store K/V that nobody borrows: the container gets
    written, but with no accumulated cotangent there is no root to seed and the
    backward stays the plain single-root one."""
    model = _ToyModel([(None, True), (None, True)])
    ids = _calib(model)
    base, _ = run_baseline(model, ids, Gemma4Profile())
    got, _, cot = run_protocol(model, ids, Gemma4Profile(), cotangents=True)
    assert cot.n_grafted == 0 and cot.n_seeded == 0
    for name in sorted(base):
        assert abs(got[name] - base[name]) / abs(base[name]) < 1e-6, name


# --------------------------------------------------------------------------
# No leakage: across samples, across shards, across sweeps
# --------------------------------------------------------------------------
def test_harvest_ends_the_layer_and_a_finished_sweep_carries_nothing():
    """`harvest` drops the layer's leaves and containers, and a sweep that
    reaches its producers leaves the accumulator empty — so even a reused
    instance cannot seed sample N with sample N-1's cotangent."""
    model = _ToyModel(SHARED_SPECS)
    profile = Gemma4Profile()
    ids = _calib(model)

    _, _, cot = run_protocol(model, ids, profile, cotangents=True)
    assert cot.pending_keys() == []
    assert cot._live == [] and cot._containers == []
    assert cot.produced_roots() == ([], [])


def test_two_sweeps_are_independent():
    """Sweep 2 on different calibration data must reproduce exactly what a
    fresh process would compute — no accumulation across passes."""
    model = _ToyModel(SHARED_SPECS)
    profile = Gemma4Profile()
    ids_a, ids_b = _calib(model, seed=3), _calib(model, seed=99)

    first_a, _, _ = run_protocol(model, ids_a, profile, cotangents=True)
    _, _, _ = run_protocol(model, ids_b, profile, cotangents=True)
    again_a, _, _ = run_protocol(model, ids_a, profile, cotangents=True)
    assert first_a == again_a

    base_b, _ = run_baseline(model, ids_b, profile)
    got_b, _, _ = run_protocol(model, ids_b, profile, cotangents=True)
    for name in sorted(base_b):
        assert abs(got_b[name] - base_b[name]) / abs(base_b[name]) < 1e-6, name


def test_a_borrowed_leaf_is_never_seeded_as_a_root():
    """The reverse sweep hands consumer N-1 a container keyed the same as the
    accumulator entry consumer N just filled. Seeding the borrowed LEAF there
    would inject consumer N's cotangent into consumer N-1's harvest and
    double-count it. Pin the guard directly."""
    cot = SharedStateCotangents()
    captured = {0: (torch.randn(2, 3, dtype=DTYPE),
                    torch.randn(2, 3, dtype=DTYPE))}

    class _Sharing:
        self_attn = type("A", (), {"layer_idx": 3, "is_kv_shared_layer": True,
                                   "kv_shared_layer_index": 0})()

    profile = Gemma4Profile()
    # Consumer #1: graft, fake a backward, harvest.
    state = cot.graft(profile.isolated_layer_pass_state(captured, _Sharing()))
    for leaf in state["shared_kv_states"][0]:
        leaf.grad = torch.ones_like(leaf)
    assert cot.produced_roots() == ([], []), "no producer forwarded yet"
    cot.harvest()
    assert len(cot.pending_keys()) == 2

    # Consumer #2 sees a container keyed 0 with the accumulator already loaded.
    state = cot.graft(profile.isolated_layer_pass_state(captured, _Sharing()))
    roots, grads = cot.produced_roots()
    assert roots == [] and grads == [], "borrowed leaf was seeded as a root"
    assert len(cot.pending_keys()) == 2, "accumulator was consumed by a consumer"


def test_non_differentiable_shared_state_is_reported_not_silently_dropped():
    """Integer shared state carries no cotangent — the producer's Fisher
    genuinely cannot be recovered through it. Record it rather than pretend."""
    cot = SharedStateCotangents()
    state = cot.graft({"shared_kv_states": {0: torch.zeros(2, dtype=torch.long)}})
    assert state["shared_kv_states"][0].dtype == torch.long
    assert cot.n_grafted == 0
    assert cot.nondifferentiable and "torch.int64" in cot.nondifferentiable[0]


def test_unclaimed_cotangent_is_visible():
    """A sweep that never forwards the producing layer leaves the cotangent
    unclaimed — that is a residual under-count and must be reportable."""
    cot = SharedStateCotangents()
    leaf_state = cot.graft({"shared_kv_states": {7: torch.randn(2, 2)}})
    leaf_state["shared_kv_states"][7].grad = torch.ones(2, 2)
    cot.harvest()
    assert cot.pending_keys() == [("shared_kv_states", 7, None)]
    assert "UNCLAIMED" in cot.summary()


def test_k_eq_v_layer_seeds_one_root_and_still_matches_end_to_end():
    """Gemma4's `use_alternative_attention` layers have `v_proj is None` and set
    `value_states = key_states`, so k and v in the container are the SAME
    object. It must be seeded once with the SUMMED grad (two roots would be
    legal but ambiguous) — and the result must still equal an end-to-end
    backward, which is the only check that says the sum was right."""
    model = _ToyModel([(None, True), (0, False), (0, False)], k_eq_v=True)
    ids = _calib(model)
    profile = Gemma4Profile()

    base, baseline_gy = run_baseline(model, ids, profile)
    got, protocol_gy, cot = run_protocol(model, ids, profile, cotangents=True)

    assert cot.n_grafted == 4  # 2 consumers x (k, v) leaves, distinct clones
    assert cot.n_seeded == 2   # both fold into ONE root
    assert len(protocol_gy[0]) == 1, "k_eq_v produced two roots for one tensor"
    rel = ((protocol_gy[0][0] - baseline_gy[0][0]).norm()
           / baseline_gy[0][0].norm()).item()
    assert rel < 1e-12, f"k_eq_v cotangent error {rel:.3e}"
    for name in sorted(base):
        assert abs(got[name] - base[name]) / abs(base[name]) < 1e-6, name


# --------------------------------------------------------------------------
# The guard, inverted
# --------------------------------------------------------------------------
def test_flag_default_and_parsing(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_KV_COTANGENT", raising=False)
    assert kv_cotangent_path_enabled() is True
    for off in ("0", "", "false", "False", "FALSE", "no", "NO"):
        monkeypatch.setenv("PRISMAQUANT_KV_COTANGENT", off)
        assert kv_cotangent_path_enabled() is False, off
    for on in ("1", "true", "yes"):
        monkeypatch.setenv("PRISMAQUANT_KV_COTANGENT", on)
        assert kv_cotangent_path_enabled() is True, on


def test_probe_sweeps_wire_the_accumulator_in_the_right_order():
    """AST pin on the shipped reverse sweep: the accumulator is built ONCE per
    sweep (not per layer — a per-layer instance would lose every consumer's
    cotangent), the layer's pass state goes through `graft`, and the backward is
    the two-root form guarded by `produced_roots`."""
    import ast
    import inspect
    import textwrap

    from prismaquant import incremental_probe as ip

    tree = ast.parse(textwrap.dedent(inspect.getsource(ip._run_body_streaming_shard)))
    news = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "SharedStateCotangents"]
    assert len(news) == 1, "one accumulator per sweep"
    # ... and it is gated by the flag, not hard-wired on.
    assert any(kw.arg == "enabled" for kw in news[0].keywords), \
        "accumulator not gated by PRISMAQUANT_KV_COTANGENT"
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == "L"
             and isinstance(n.iter, ast.Call)
             and isinstance(n.iter.func, ast.Name)
             and n.iter.func.id == "reversed"]
    assert len(loops) == 1, "reverse sweep not found"
    body = list(ast.walk(loops[0]))
    assert news[0] not in body, "accumulator rebuilt per layer"
    for attr in ("produced_roots", "harvest", "graft"):
        assert any(isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute)
                   and n.func.attr == attr for n in body), \
            f"reverse sweep never calls .{attr}()"
    # The seeded backward must be the multi-root form; a plain
    # `out.backward(g)` cannot carry the shared-state cotangent.
    assert any(isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute)
               and n.func.attr == "backward"
               and isinstance(n.func.value, ast.Attribute)
               and n.func.value.attr == "autograd" for n in body), \
        "no torch.autograd.backward multi-root path"


def test_the_sweep_order_claim_holds_for_gemma4():
    """The whole design rests on consumers being swept BEFORE their producer.

    Checked BEHAVIOURALLY against the installed classes, not against the text of
    upstream's source. An earlier version of this test asserted on substrings of
    `modeling_gemma4.py` and went red the moment CI installed a transformers
    whose file is generated differently — pinning someone else's source text is
    a version bomb, and the property we actually depend on is observable:
    every KV-sharing layer must borrow from a layer with a LOWER index, because
    phase-3 sweeps `reversed(range(num_layers))`.
    """
    import pytest

    pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    # A config with KV sharing switched on. `layer_types` is what the sharing
    # index is derived from, and BOTH attention kinds must appear *below* the
    # sharing point or upstream's derivation cannot resolve a source layer — so
    # spell the pattern out rather than relying on the default for a given depth
    # (with 6 layers and 2 shared, the default leaves only sliding layers below
    # the sharing point and Gemma4 raises `'full_attention' is not in list`).
    cfg = Gemma4TextConfig(
        num_hidden_layers=8,
        num_kv_shared_layers=2,
        layer_types=(["sliding_attention"] * 5 + ["full_attention"]
                     + ["sliding_attention", "full_attention"]),
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    if getattr(cfg, "num_kv_shared_layers", 0) <= 0:
        pytest.skip("installed transformers does not model KV sharing here")

    sharing = 0
    for idx in range(cfg.num_hidden_layers):
        try:
            attn = Gemma4TextAttention(cfg, idx)
        except Exception as exc:  # noqa: BLE001
            # Upstream changed how the sharing source is derived. Skip rather
            # than fail: this test guards OUR one-pass assumption, and a red
            # suite here would say nothing about whether that assumption holds.
            pytest.skip(f"Gemma4TextAttention({idx}) not constructible here: "
                        f"{type(exc).__name__}: {exc}")
        if not getattr(attn, "is_kv_shared_layer", False):
            continue
        sharing += 1
        # Read defensively: the attribute that names the source layer is
        # version-dependent (a newer transformers sets `is_kv_shared_layer`
        # but exposes no `kv_shared_layer_index`). Skip rather than fail —
        # and note the implication, because `Gemma4Profile.
        # isolated_layer_pass_state` reads the same attribute, so on such a
        # transformers our KV-sharing probe path cannot resolve a source
        # either. It fails loudly there rather than silently, but this
        # architecture is only actually supported on a build that exposes it.
        source = getattr(attn, "kv_shared_layer_index", None)
        if source is None:
            pytest.skip(
                "installed transformers marks layer "
                f"{idx} kv-shared but exposes no `kv_shared_layer_index`; the "
                "sweep-order property cannot be checked here, and "
                "Gemma4Profile.isolated_layer_pass_state cannot resolve a "
                "source layer on this build either"
            )
        # THE claim: the producer is strictly below the consumer, so a reverse
        # sweep harvests the consumer's cotangent before forwarding the producer.
        assert source < idx, (
            f"layer {idx} borrows K/V from layer {source}, which is NOT below "
            "it — a reverse sweep would forward the producer before its "
            "consumer's cotangent existed, and the one-pass accumulation in "
            "`SharedStateCotangents` would be wrong. This architecture needs a "
            "two-pass scheme."
        )
    assert sharing > 0, "config declared KV sharing but no layer reported it"


def test_guard_no_longer_blocks_kv_sharing_by_default(monkeypatch):
    from prismaquant import incremental_probe as ip

    monkeypatch.setattr(ip, "config_num_kv_shared_layers", lambda p: 5)
    monkeypatch.delenv("PRISMAQUANT_ALLOW_KV_SHARED_FISHER", raising=False)
    monkeypatch.delenv("PRISMAQUANT_KV_COTANGENT", raising=False)
    assert ip.kv_shared_fisher_block_reason("any/model") is None


def test_guard_fires_when_the_cotangent_path_is_switched_off(monkeypatch):
    from prismaquant import incremental_probe as ip

    monkeypatch.setattr(ip, "config_num_kv_shared_layers", lambda p: 5)
    monkeypatch.delenv("PRISMAQUANT_ALLOW_KV_SHARED_FISHER", raising=False)
    monkeypatch.setenv("PRISMAQUANT_KV_COTANGENT", "0")
    msg = ip.kv_shared_fisher_block_reason("any/model")
    assert msg is not None
    assert "num_kv_shared_layers=5" in msg
    assert "PRISMAQUANT_KV_COTANGENT=0" in msg
    # ... and the explicit accept-the-under-count override still wins.
    monkeypatch.setenv("PRISMAQUANT_ALLOW_KV_SHARED_FISHER", "1")
    assert ip.kv_shared_fisher_block_reason("any/model") is None
