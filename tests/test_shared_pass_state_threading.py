"""Per-forward-pass SHARED layer kwargs in PrismaQuant's streaming forward.

Issue #9 item 1: Gemma4 shares K/V across same-type layers through a
`shared_kv_states` dict that `Gemma4TextModel.forward` creates ONCE per pass
and threads through every decoder layer (storing layers write
`shared_kv_states[layer_idx] = (k, v)`; KV-sharing layers read
`shared_kv_states[kv_shared_layer_index]`). PrismaQuant's manual streaming
layer loop passed nothing, so the layer assigned into `None`:
`TypeError: 'NoneType' object does not support item assignment`.

The plumbing is `ModelProfile.new_forward_pass_state()` (per-pass, shared) +
`_call_layer(..., pass_state=...)`. These tests pin the semantics that are
easy to get wrong, with synthetic layers only — no Gemma4 weights, no GPU:

  * a layer that REQUIRES the shared kwarg is callable at all;
  * layer N's write is visible to layer N+1 in the SAME pass (the merge is
    shallow, so the container stays shared by reference);
  * pass 2 gets a FRESH container — no leakage from pass 1;
  * a profile that declares nothing adds NO kwarg (strict-signature layers
    still work);
  * `extra_layer_kwargs` / shared-kwarg key collisions fail loud;
  * `Gemma4Profile` declares the key the installed transformers expects, and
    phase-3's isolated reconstruction fails loud when the phase-1 capture is
    missing rather than raising `KeyError` from inside attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from prismaquant.layer_streaming import _call_layer, merge_pass_state_kwargs
from prismaquant.model_profiles.default import DefaultProfile
from prismaquant.model_profiles.gemma4 import Gemma4Profile
from prismaquant.model_profiles.qwen3 import Qwen3Profile


# --- fakes -----------------------------------------------------------------
class _KVSharingLayer(nn.Module):
    """Mimics a Gemma4 decoder layer: `shared_kv_states` is REQUIRED (no
    default), storing layers write into it, sharing layers read from it.

    A missing kwarg is a TypeError; a `None` kwarg reproduces the original
    crash (item assignment / subscript on None).
    """

    def __init__(self, layer_idx: int, *, shares_from: int | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.shares_from = shares_from
        self.borrowed = None

    def forward(self, *, hidden_states, shared_kv_states, **kw):
        if self.shares_from is None:
            # Storing layer: derive a "K/V" from this pass's hidden state so a
            # stale value from another pass is distinguishable.
            shared_kv_states[self.layer_idx] = hidden_states.clone()
        else:
            # KV-sharing layer: unconditional lookup, exactly as
            # Gemma4TextAttention.forward does.
            self.borrowed = shared_kv_states[self.shares_from]
        return hidden_states


class _StrictLayer(nn.Module):
    """No `**kwargs` absorption for unexpected names: any extra kwarg the
    default (no shared state) path adds would be a TypeError."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False,
                position_embeddings=None):
        self.calls += 1
        return hidden_states


def _run_pass(profile, layers, hidden):
    """Mimic the production loop (`incremental_probe._compute_global_precompute`
    phase-1): build the shared state ONCE at the outermost scope of the pass, thread it
    into every layer call. Returns the pass state so tests can inspect it."""
    pass_state = profile.new_forward_pass_state()
    for layer in layers:
        hidden = _call_layer(
            layer, hidden,
            position_embeddings=None, attention_mask=None, position_ids=None,
            pass_state=pass_state,
        )
    return pass_state, hidden


# --- the pass semantics ----------------------------------------------------
def test_shared_state_threads_through_a_pass():
    """The loop completes, and the KV-sharing layer sees what the storing
    layer wrote in the SAME pass."""
    profile = Gemma4Profile()
    store, share = _KVSharingLayer(0), _KVSharingLayer(1, shares_from=0)
    hidden = torch.randn(1, 3, 4)

    pass_state, out = _run_pass(profile, [store, share], hidden)

    assert torch.equal(share.borrowed, hidden), "layer 1 saw layer 0's write"
    assert set(pass_state["shared_kv_states"]) == {0}
    assert torch.equal(out, hidden)


def test_missing_shared_state_is_the_reported_crash():
    """Non-regression anchor: without the plumbing the layer gets nothing (or
    `None`) and dies — the exact failures issue #9 item 1 reports."""
    share = _KVSharingLayer(1, shares_from=0)
    common = dict(position_embeddings=None, attention_mask=None,
                  position_ids=None)

    try:
        _call_layer(share, torch.randn(1, 3, 4), **common)
    except TypeError as exc:
        assert "shared_kv_states" in str(exc)
    else:  # pragma: no cover - assert path keeps compatibility w/o pytest.raises
        raise AssertionError("layer accepted a call with no shared_kv_states")

    store = _KVSharingLayer(0)
    try:
        _call_layer(store, torch.randn(1, 3, 4),
                    pass_state={"shared_kv_states": None}, **common)
    except TypeError as exc:
        assert "NoneType" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("item assignment into None was accepted")


def test_second_pass_gets_a_fresh_container():
    """No leakage across passes: pass 2's sharing layer must borrow pass 2's
    K/V, and pass 1's state must not be reachable from pass 2."""
    profile = Gemma4Profile()
    store, share = _KVSharingLayer(0), _KVSharingLayer(1, shares_from=0)

    hidden_a = torch.zeros(1, 3, 4)
    state_a, _ = _run_pass(profile, [store, share], hidden_a)
    borrowed_a = share.borrowed

    hidden_b = torch.ones(1, 3, 4)
    state_b, _ = _run_pass(profile, [store, share], hidden_b)
    borrowed_b = share.borrowed

    assert state_a is not state_b
    assert state_a["shared_kv_states"] is not state_b["shared_kv_states"]
    assert torch.equal(borrowed_a, hidden_a)
    assert torch.equal(borrowed_b, hidden_b), "pass 2 borrowed pass 1's K/V"
    # Pass 1's container was not mutated by pass 2 either.
    assert torch.equal(state_a["shared_kv_states"][0], hidden_a)


def test_new_forward_pass_state_is_fresh_per_call():
    """The freshness above must come from the hook, not from luck: two calls
    return distinct containers, and mutating one cannot be seen by the other."""
    for profile in (Gemma4Profile(), DefaultProfile(), Qwen3Profile()):
        first, second = (profile.new_forward_pass_state(),
                         profile.new_forward_pass_state())
        assert first is not second
        for key, value in first.items():
            assert value is not second[key], f"{type(profile).__name__}.{key}"
        first["_probe"] = True
        assert "_probe" not in profile.new_forward_pass_state()


def test_pass_state_container_is_shared_by_reference_not_copied():
    """`_call_layer` merges the kwargs mapping shallowly; if it deep-copied,
    layer N's writes would be invisible to layer N+1."""
    shared: dict = {}
    store = _KVSharingLayer(0)
    _call_layer(store, torch.randn(1, 3, 4),
                position_embeddings=None, attention_mask=None,
                position_ids=None, pass_state={"shared_kv_states": shared})
    assert set(shared) == {0}


# --- architectures that declare nothing are untouched ----------------------
def test_default_profiles_declare_no_shared_kwargs():
    for profile in (DefaultProfile(), Qwen3Profile()):
        assert profile.new_forward_pass_state() == {}
        assert profile.isolated_layer_pass_state(None, object()) == {}
        assert profile.capture_forward_pass_state({}) is None


def test_no_shared_kwargs_adds_no_kwarg_to_the_call():
    """A layer with a strict signature (no `**kwargs`) must still be callable:
    the default `{}` may not introduce a `shared_kv_states=` kwarg."""
    profile = DefaultProfile()
    layer = _StrictLayer()
    hidden = torch.randn(1, 3, 4)

    _, out = _run_pass(profile, [layer, layer], hidden)
    assert layer.calls == 2
    assert torch.equal(out, hidden)

    # ... and the same call WITH a declared shared kwarg would have raised,
    # which is what makes the assertion above meaningful.
    try:
        _call_layer(layer, hidden, position_embeddings=None,
                    attention_mask=None, position_ids=None,
                    pass_state={"shared_kv_states": {}})
    except TypeError as exc:
        assert "shared_kv_states" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("strict layer accepted an undeclared kwarg")


def test_pass_state_key_collision_fails_loud():
    """A profile whose per-layer `extra_layer_kwargs` and per-pass shared
    kwargs use the same name would otherwise be a silent override (or a raw
    'multiple values for keyword argument')."""
    layer = _KVSharingLayer(0)
    try:
        _call_layer(layer, torch.randn(1, 3, 4), position_embeddings=None,
                    attention_mask=None, position_ids=None,
                    shared_kv_states={"per-layer": 1},
                    pass_state={"shared_kv_states": {}})
    except RuntimeError as exc:
        assert "collide" in str(exc) and "shared_kv_states" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("colliding per-pass/per-layer kwarg accepted")


def test_merge_rule_is_shared_by_every_loop():
    """`_call_layer(pass_state=...)` merges through one function, so the
    per-pass semantics are defined once."""
    shared: dict = {}
    extra = {"input_ids": "ids"}

    merged = merge_pass_state_kwargs(extra, {"shared_kv_states": shared},
                                     context="L")
    assert merged == {"input_ids": "ids", "shared_kv_states": shared}
    assert merged["shared_kv_states"] is shared  # by reference
    assert extra == {"input_ids": "ids"}  # caller's dict untouched
    assert merge_pass_state_kwargs(extra, None, context="L") is extra
    assert merge_pass_state_kwargs(extra, {}, context="L") is extra


# --- Gemma4: the declared key must match the real contract -----------------
def test_gemma4_declares_shared_kv_states():
    assert Gemma4Profile().new_forward_pass_state() == {"shared_kv_states": {}}


def test_gemma4_key_matches_installed_transformers_signature():
    """Pin the profile's key against the installed modeling code rather than
    against this test's memory of it."""
    import inspect

    import pytest

    modeling = pytest.importorskip(
        "transformers.models.gemma4.modeling_gemma4")

    declared = set(Gemma4Profile().new_forward_pass_state())
    for cls_name in ("Gemma4TextDecoderLayer", "Gemma4TextAttention"):
        params = inspect.signature(
            getattr(modeling, cls_name).forward).parameters
        assert declared <= set(params), f"{cls_name}.forward lost {declared}"

    # The decoder layer defaults it to None (hence the reported TypeError),
    # while attention requires it outright.
    layer_params = inspect.signature(
        modeling.Gemma4TextDecoderLayer.forward).parameters
    assert layer_params["shared_kv_states"].default is None
    attn_params = inspect.signature(
        modeling.Gemma4TextAttention.forward).parameters
    assert attn_params["shared_kv_states"].default is inspect.Parameter.empty


def test_gemma4_isolated_state_for_a_kv_sharing_layer():
    """Phase-3 forwards one layer in isolation: a KV-sharing layer gets its
    source layer's captured K/V; a storing layer gets a writable dict."""
    from types import SimpleNamespace

    profile = Gemma4Profile()
    kv = (torch.randn(1, 2, 3), torch.randn(1, 2, 3))
    captured = {5: kv}

    sharing = SimpleNamespace(self_attn=SimpleNamespace(
        layer_idx=9, is_kv_shared_layer=True, kv_shared_layer_index=5))
    assert profile.isolated_layer_pass_state(captured, sharing) == {
        "shared_kv_states": {5: kv}}

    storing = SimpleNamespace(self_attn=SimpleNamespace(
        layer_idx=5, is_kv_shared_layer=False, kv_shared_layer_index=None))
    state = profile.isolated_layer_pass_state(captured, storing)
    assert state == {"shared_kv_states": {}}
    state["shared_kv_states"][5] = kv  # writable, and not the captured dict
    assert set(captured) == {5}


def test_gemma4_isolated_state_without_capture_fails_loud():
    """A KV-sharing layer cannot be forwarded without the phase-1 capture —
    say so, instead of a bare KeyError from inside attention. This is the
    path a precompute cache written without `shared_pass_state` takes."""
    from types import SimpleNamespace

    profile = Gemma4Profile()
    sharing = SimpleNamespace(self_attn=SimpleNamespace(
        layer_idx=9, is_kv_shared_layer=True, kv_shared_layer_index=5))

    for captured in (None, {}, {7: (torch.zeros(1), torch.zeros(1))}):
        try:
            profile.isolated_layer_pass_state(captured, sharing)
        except RuntimeError as exc:
            assert "shared_kv_states[5]" in str(exc)
            assert "precomputed.pt" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(
                f"missing captured K/V accepted for {captured!r}")


def test_gemma4_capture_is_detached_cpu_and_layer_indexed():
    profile = Gemma4Profile()
    k = torch.randn(1, 2, 3, requires_grad=True)
    v = torch.randn(1, 2, 3, requires_grad=True)

    captured = profile.capture_forward_pass_state(
        {"shared_kv_states": {3: (k, v)}})

    assert set(captured) == {3}
    assert all(t.device.type == "cpu" and not t.requires_grad
               for t in captured[3])


# --- the probe's phase-1 -> phase-3 hand-off survives the disk cache -------
def test_precompute_cache_round_trips_shared_pass_state(tmp_path):
    """Phase-3 rebuilds KV-sharing layers' borrowed K/V from
    `GlobalPrecompute.shared_pass_state`. The shard runners routinely read the
    precompute back from disk (resume / one process per shard), so dropping
    the key on save made every cached run raise inside attention."""
    from prismaquant import incremental_probe as ip

    kv = (torch.randn(1, 2, 3), torch.randn(1, 2, 3))
    pre = ip.GlobalPrecompute(
        activations_cpu=[torch.zeros(1, 2, 3)],
        grad_at_tail=torch.zeros(1, 2, 3),
        ids=torch.zeros(1, 2, dtype=torch.long),
        resident_stats={}, resident_h_full={}, resident_g2_per_token={},
        resident_act_snaps={}, resident_act_row_indices={},
        expert_info={}, router_counts={}, router_totals={},
        router_active_counts={}, expert_route_stats={},
        shared_pass_state={5: kv},
    )
    meta = {"model": "m", "nsamples": 1}
    path = tmp_path / "work" / "precomputed.pt"

    ip._save_precompute_cache(path, pre, meta)
    loaded = ip._load_precompute_cache(path, meta, torch.device("cpu"))

    assert loaded is not None
    assert set(loaded.shared_pass_state) == {5}
    assert torch.equal(loaded.shared_pass_state[5][0], kv[0])
    assert torch.equal(loaded.shared_pass_state[5][1], kv[1])


# --- where the production loops build the shared state ---------------------
def _fn_ast(func):
    import ast
    import inspect
    import textwrap

    return ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]


def _calls(node, attr):
    import ast

    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == attr]


def _layer_loops(node):
    """`for L in range(num_layers)` / `for L in reversed(range(num_layers))`."""
    import ast

    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) \
                and n.target.id == "L":
            out.append(n)
    return out


def test_probe_phase1_builds_shared_state_outside_the_layer_loop():
    """Per PASS, not per layer: hoisted above `for L in range(num_layers)` so
    the storing layer's K/V is still there when a later layer borrows it (and
    so `_call_layer` receives the same object every iteration)."""
    import ast

    from prismaquant import incremental_probe as ip

    fn = _fn_ast(ip._compute_global_precompute)
    news = _calls(fn, "new_forward_pass_state")
    assert len(news) == 1, "phase-1 must build the shared state exactly once"
    loops = _layer_loops(fn)
    assert loops, "phase-1 layer loop not found — did the loop shape change?"
    for loop in loops:
        assert news[0] not in list(ast.walk(loop)), \
            "shared state rebuilt per layer would lose earlier layers' writes"
        threaded = [c for c in ast.walk(loop)
                    if isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Name)
                    and c.func.id == "_call_layer"
                    and any(kw.arg == "pass_state" for kw in c.keywords)]
        assert threaded, "phase-1 layer call does not thread pass_state"


def test_precompute_cache_without_shared_state_loads_as_none(tmp_path):
    """Architectures that declare nothing (and caches written before the key
    existed) load with `shared_pass_state=None`."""
    from prismaquant import incremental_probe as ip

    pre = ip.GlobalPrecompute(
        activations_cpu=[torch.zeros(1, 2, 3)],
        grad_at_tail=torch.zeros(1, 2, 3),
        ids=torch.zeros(1, 2, dtype=torch.long),
        resident_stats={}, resident_h_full={}, resident_g2_per_token={},
        resident_act_snaps={}, resident_act_row_indices={},
        expert_info={}, router_counts={}, router_totals={},
        router_active_counts={}, expert_route_stats={},
    )
    meta = {"model": "m"}
    path = tmp_path / "precomputed.pt"

    ip._save_precompute_cache(path, pre, meta)
    loaded = ip._load_precompute_cache(path, meta, torch.device("cpu"))

    assert loaded is not None and loaded.shared_pass_state is None
