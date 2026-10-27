"""Exercise the production body composition against end-to-end autograd."""
from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from prismaquant import incremental_probe as probe
from prismaquant.sensitivity_probe import FisherAccumulator
from prismaquant.streaming_model import _build_streaming_context


@pytest.mark.parametrize("importance_weighting", [False, True])
def test_real_streamed_body_rows_match_end_to_end_fisher(
        tmp_path, monkeypatch, importance_weighting):
    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", "1")
    monkeypatch.setenv("PRISMAQUANT_ACT_CACHE_WORKERS", "1")
    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "1")
    torch.manual_seed(123)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=2, head_dim=8, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(config).eval()
    checkpoint = tmp_path / "model"
    reference.save_pretrained(checkpoint)
    ids = torch.tensor([[1, 3, 7, 11], [2, 5, 13, 17]], dtype=torch.int64)
    tracked = [name for name, module in reference.named_modules()
               if isinstance(module, torch.nn.Linear)]
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    collector = FisherAccumulator(reference, tracked, {},
                                  hook_packed_experts=False, emit_dense_marginals=True)
    try:
        inputs = reference.get_input_embeddings()(ids).detach().requires_grad_(True)
        hidden = reference.model(inputs_embeds=inputs, use_cache=False).last_hidden_state
        logits = reference.lm_head(hidden[:, :-1, :]).float()
        log_prob = torch.nn.functional.log_softmax(logits.reshape(-1, config.vocab_size), dim=-1)
        ce = -log_prob.gather(1, ids[:, 1:].reshape(-1, 1)).squeeze(1)
        if importance_weighting:
            weights = (ce.detach() / ce.detach().mean()).clamp(.25, 4.)
            loss = (ce * weights).sum()
        else:
            loss = ce.sum()
        head_grad, = torch.autograd.grad(loss, logits, retain_graph=True)
        x = hidden[:, :-1, :].detach().reshape(-1, config.hidden_size).float()
        g = head_grad.detach().reshape(-1, config.vocab_size).float()
        h = g.square().T @ x.square()
        trace = float(h.sum())
        weighted_trace = float((h * reference.lm_head.weight.detach().float().square()).sum())
        head_expected = {
            "n_tokens_seen": x.size(0), "h_trace_raw": trace,
            "h_w2_sum_raw": weighted_trace, "h_trace": trace / ids.numel(),
            "h_w2_sum": weighted_trace / ids.numel(),
            "fisher_row": h.sum(dim=1).numpy(), "fisher_col": h.sum(dim=0).numpy(),
            "g_sq_sum": g.square().sum(dim=0).numpy(),
            "act_sq_sum": x.square().sum(dim=0).numpy(),
            "act_absmax": x.abs().amax(dim=0).numpy(),
        }
        loss.backward()
        collector.finalize(tracker=None, global_tokens=ids.numel())
    finally:
        collector.remove_hooks()

    context = _build_streaming_context(str(checkpoint), device=torch.device("cpu"),
        dtype=torch.float32, offload_folder=str(tmp_path / "offload"),
        max_cache_slots=2, prefetch_workers=1, prefetch_min_available_gb=0,
        cache_headroom_gb=0, attn_implementation="eager")
    try:
        precomputed = probe._compute_global_precompute(context, calib=ids,
            importance_weighting=importance_weighting, prefetch_lookahead=1,
            minimax_fast_moe=False, minimax_fast_moe_chunk_size=32,
            resident_include_union=".*", resident_exclude="(?!)",
            activation_cache_dir=str(tmp_path / "activations"))
        output = tmp_path / "probe.pkl"
        probe._run_body_streaming_shard(context, calib=ids, linear_include=".*",
            linear_exclude="(?!)", importance_weighting=importance_weighting,
            activation_cache_dir=str(tmp_path / "activations"), h_detail_dir=None,
            output_path=str(output), dataset_name="explicit-tokens", dtype_name="fp32",
            seqlen=ids.size(1), model_path=str(checkpoint), prefetch_lookahead=1,
            minimax_fast_moe=False, activation_rows_limit=ids.numel(),
            precomputed=precomputed)
        with output.open("rb") as handle:
            actual = pickle.load(handle)["stats"]
        assert set(actual) == set(tracked)
        for name in tracked:
            expected = head_expected if name == "lm_head" else collector.stats[name]
            assert actual[name]["n_tokens_seen"] == expected["n_tokens_seen"], name
            for field in ("h_trace_raw", "h_w2_sum_raw", "h_trace", "h_w2_sum",
                          "fisher_row", "fisher_col", "g_sq_sum", "act_sq_sum", "act_absmax"):
                np.testing.assert_allclose(actual[name][field], expected[field],
                                           rtol=2e-5, atol=1e-7, err_msg=f"{name}/{field}")
        assert all(parameter.is_meta for layer in context.layers for parameter in layer.parameters())
        assert len(precomputed.activations_cpu) == 3
    finally:
        context.shutdown()


@pytest.mark.parametrize("marginals", ["0", "1"])
def test_ordinary_mtp_retains_its_declared_legacy_marginal_payload(monkeypatch, marginals):
    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", marginals)
    torch.manual_seed(42)
    wrapper = torch.nn.Module()
    wrapper.mtp = torch.nn.Module()
    wrapper.mtp.fc = torch.nn.Linear(3, 2, bias=False)
    collector = probe._build_mtp_fisher_accumulator(wrapper, ["mtp.fc"], {}, None,
        input_rows_limit=8, detail_dir=None)
    try:
        inputs = torch.randn(1, 2, 3, requires_grad=True)
        wrapper.mtp.fc(inputs).sum().backward()
        collector.finalize(tracker=None, global_tokens=4)
        row = collector.stats["mtp.fc"]
        assert row["n_tokens_seen"] == 2
        assert row["h_trace_raw"] > 0
        assert row["h_trace"] == pytest.approx(row["h_trace_raw"] / 4)
        assert not ({"fisher_row", "fisher_col", "g_sq_sum", "act_sq_sum", "act_absmax"} & row.keys())
    finally:
        collector.remove_hooks()
