"""Cross-domain do-no-harm gate for packed-MoE expert GPTQ renders.

The 2026-06-09 35B served A/B showed per-expert GPTQ overfits its calibration
DOMAIN (thin Hessians from sparse routing): a same-corpus held-out split passed
renders that lost to RTN on served cross-domain KL. These tests pin the gate's
control flow in ``fill_packed_expert_cache_entries``:

  * default (no gate corpus): unchanged in-domain holdout behavior;
  * with ``gate_calib_ids``: the GPTQ-vs-RTN decision is judged on the gate
    corpus's routed rows, GPTQ fits on ALL fit-corpus rows, and coverage
    records the gate mode.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.production_weight_cache import (
    ProductionWeightCache,
    fill_packed_expert_cache_entries,
)


class TinyRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.top_k = 1
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        logits = F.linear(hidden_states, self.weight)
        scores, indices = torch.topk(
            torch.softmax(logits.float(), dim=-1), 1, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return logits, scores.to(hidden_states.dtype), indices


class TinyPackedExperts(nn.Module):
    def __init__(self, hidden_size: int = 16, intermediate_size: int = 16,
                 num_experts: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            mask = mask.permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for ei in hit:
            ei = ei[0]
            pos, tok = torch.where(mask[ei])
            gate, up = F.linear(
                hidden_states[tok], self.gate_up_proj[ei]).chunk(2, dim=-1)
            cur = F.linear(self.act_fn(gate) * up, self.down_proj[ei])
            cur = cur * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, cur.to(final.dtype))
        return final


class TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = TinyRouter(hidden_size=16, num_experts=2)
        self.experts = TinyPackedExperts()


class TinyLM(nn.Module):
    """Minimal id->hidden->routed-experts forward so the activation
    collector's hook on ``mlp.experts`` sees real module-level X."""

    def __init__(self, vocab: int = 32, hidden: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.mlp = TinyMlp()

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        h = self.embed(input_ids).reshape(-1, self.embed.embedding_dim)
        _logits, weights, indices = self.mlp.gate(h)
        return self.mlp.experts(h, indices, weights)


ASSIGNMENT = {
    "mlp.experts.gate_up_proj": "NVFP4",
    "mlp.experts.down_proj": "NVFP4",
}


def _fill(model, calib_ids, gate_calib_ids=None, eval_rows_per_expert=8):
    cache = ProductionWeightCache(weights={}, levers={"gptq": True})
    coverage = fill_packed_expert_cache_entries(
        cache, model, calib_ids,
        render_assignment=ASSIGNMENT,
        levers={"gptq": True},
        profile=None,
        module_token_budget=4096,
        eval_rows_per_expert=eval_rows_per_expert,
        progress=False,
        gate_calib_ids=gate_calib_ids,
    )
    return cache, coverage


def test_default_path_keeps_in_domain_holdout():
    torch.manual_seed(11)
    model = TinyLM().eval()
    calib = torch.randint(0, 32, (2, 64))

    cache, cov = _fill(model, calib)

    assert set(cov) == set(ASSIGNMENT)
    for full, entry in cov.items():
        assert entry["gate_mode"] == "in-domain-holdout"
        assert entry["cross_gated_experts"] == 0
        assert (full, "NVFP4") in cache.weights
        assert cache.activation_max_abs.get(full, 0.0) > 0.0


def test_cross_domain_gate_judges_on_gate_corpus_and_fits_on_all_rows(
        monkeypatch):
    torch.manual_seed(13)
    model = TinyLM().eval()
    calib = torch.randint(0, 32, (2, 64))      # fit corpus: 128 routed tokens
    gate = torch.randint(0, 32, (1, 40))       # disjoint gate corpus

    eval_rows = 8
    scored_eval_rows: list[int] = []
    fit_row_counts: list[list[int]] = []

    import prismaquant.render_score as rs
    import prismaquant.export_batched_gptq as ebg
    real_score = rs.score_render_error
    real_batched = ebg.gptq_obs_rounding_nvfp4_batched

    def spy_score(src, rendered, acts, row_weights=None, **kw):
        scored_eval_rows.append(int(acts.shape[0]))
        return real_score(src, rendered, acts, row_weights=row_weights, **kw)

    def spy_batched(src, render_acts, **kw):
        fit_row_counts.append([int(a.shape[0]) for a in render_acts])
        return real_batched(src, render_acts, **kw)

    import prismaquant.production_weight_cache as pwc
    monkeypatch.setattr(pwc, "score_render_error", spy_score)
    monkeypatch.setattr(ebg, "gptq_obs_rounding_nvfp4_batched", spy_batched)

    cache, cov = _fill(model, calib, gate_calib_ids=gate,
                       eval_rows_per_expert=eval_rows)

    for full, entry in cov.items():
        assert entry["gate_mode"] == "cross-domain"
        # 2 experts, 128 fit tokens, top-1 routing: both experts get rows;
        # the 40-token gate corpus routes to at least one of them.
        assert entry["cross_gated_experts"] >= 1
        assert (full, "NVFP4") in cache.weights

    # The gate judged on the gate corpus's routed rows: every scored eval set
    # is capped at eval_rows (the gate-derive cap), while the fit sets kept
    # ALL in-domain routed rows (~64/expert) — no same-domain holdout was
    # carved out of them for cross-gated experts.
    assert scored_eval_rows and max(scored_eval_rows) <= eval_rows
    assert fit_row_counts
    total_fit_rows = sum(fit_row_counts[0])
    assert total_fit_rows == calib.numel(), (
        f"expected GPTQ to fit on all {calib.numel()} routed fit-corpus rows, "
        f"got {total_fit_rows} (a holdout was carved out)")


def test_cross_domain_gate_reverts_to_rtn_when_gate_prefers_it(monkeypatch):
    torch.manual_seed(17)
    model = TinyLM().eval()
    calib = torch.randint(0, 32, (2, 64))
    gate = torch.randint(0, 32, (1, 40))

    # Force the gate decision: first score call per expert pair (GPTQ) loses,
    # second (RTN) wins. This pins the control flow — a revert must replace
    # the rendered expert with the RTN dequant and count it.
    calls = {"n": 0}

    def rigged_score(src, rendered, acts, row_weights=None, **kw):
        calls["n"] += 1
        return 1.0 if calls["n"] % 2 == 1 else 0.0

    import prismaquant.production_weight_cache as pwc
    monkeypatch.setattr(pwc, "score_render_error", rigged_score)

    from prismaquant.native_render_codec import (
        rtn_dequant_nvfp4,
        compute_nvfp4_global_real,
    )
    from prismaquant.export_native_compressed import (
        _split_packed_expert_tensor,
    )
    from prismaquant.native_render_levers import NativeRenderLevers

    recipe = NativeRenderLevers.resolve({"gptq": True})
    cache, cov = _fill(model, calib, gate_calib_ids=gate)

    for full, entry in cov.items():
        assert entry["rtn_fallbacks"] == entry["n_experts"]
        assert entry["heldout_reverts"] >= entry["cross_gated_experts"] > 0

    # Every cached expert must be the RTN dequant under the same per-expert
    # joint global the render used.
    for (full, fmt), tensor in cache.weights.items():
        pn = full.split(".")[-1]
        src = dict(model.named_parameters())[full].detach().float()
        got = cache.get(full, fmt).float()
        for e in range(src.shape[0]):
            cands = [
                compute_nvfp4_global_real(
                    sp[e].float(), group_size=16, levers=recipe)
                for _, sp in _split_packed_expert_tensor(src, pn, None)
            ]
            g = torch.stack(cands).max() if len(cands) > 1 else cands[0]
            expected = rtn_dequant_nvfp4(
                src[e], group_size=16, global_real_override=g, levers=recipe)
            assert torch.allclose(
                got[e], expected.to(got.dtype), atol=2e-2, rtol=2e-2), (
                f"{full} expert {e}: cached render is not the RTN dequant "
                f"after a forced revert")
