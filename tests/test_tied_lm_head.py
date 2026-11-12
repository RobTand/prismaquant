"""Tied-embedding (`tie_word_embeddings`) checkpoints.

Measured failure this pins (full text-only probe of `google/gemma-4-31b-it`,
which declares `tie_word_embeddings: True` at both the top level and in
`text_config` and ships no `lm_head.*` tensor at all):

    [incremental-cost] shard 67 (lm_head): include='^lm_head$'
      File "prismaquant/measure_quant_cost.py", line 1566, in measure_batched_gpu
        W = torch.stack([m.weight.detach().to(device=dev, dtype=dtype) ...
    NotImplementedError: Cannot copy out of meta tensor; no data!

`lm_head.weight` on such a checkpoint is an *alias* of the input embedding.
Nothing in the streaming path resolved that tie, so the head stayed on meta.

Two independent things are pinned here:

  1. Materialization — the head must be a real tensor (the probe's Phase-2
     CE backward runs through `model.lm_head(...)`), resolved through
     transformers' own embedding accessors so the VL-wrapper embedding name
     (`model.language_model.embed_tokens.weight`) works exactly like the
     plain `model.embed_tokens.weight`.

  2. Non-quantizability — a tied head shares STORAGE with the embedding,
     which the pipeline prices in the non-quantizable floor. So it gets no
     cost shard and never enters the allocator's DP budget, and that is not
     overridable by `--allow-pinned lm_head`. (`aura_cost.py` already
     refuses to cost a tied head for the same reason; this generalizes it
     to the probe/cost/allocator path.)

Everything here is synthetic and CPU-only: tiny real safetensors
checkpoints (2 layers, hidden 32, vocab 64) written to tmp_path.
"""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from prismaquant.tied_embeddings import (
    config_declares_tied_embeddings,
    lm_head_is_tied_alias,
    resolve_tied_output_embedding,
    source_has_lm_head_tensor,
)


HIDDEN = 32
VOCAB = 64
LAYERS = 2


def _config(*, tied: bool, vl_wrapper: bool) -> dict:
    text = {
        "model_type": "qwen2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "vocab_size": VOCAB,
        "max_position_embeddings": 128,
        "tie_word_embeddings": tied,
    }
    if not vl_wrapper:
        return {"architectures": ["Qwen2ForCausalLM"], **text}
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "tie_word_embeddings": tied,
        "text_config": text,
    }


def _tensors(*, embed_prefix: str, with_lm_head: bool) -> dict:
    t = {f"{embed_prefix}.embed_tokens.weight": torch.zeros(VOCAB, HIDDEN,
                                                            dtype=torch.bfloat16),
         f"{embed_prefix}.norm.weight": torch.ones(HIDDEN, dtype=torch.bfloat16)}
    for L in range(LAYERS):
        p = f"{embed_prefix}.layers.{L}"
        t[f"{p}.input_layernorm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
        t[f"{p}.post_attention_layernorm.weight"] = torch.ones(
            HIDDEN, dtype=torch.bfloat16)
        for proj, out in (("q_proj", HIDDEN), ("k_proj", 16), ("v_proj", 16),
                          ("o_proj", HIDDEN)):
            t[f"{p}.self_attn.{proj}.weight"] = torch.zeros(
                out, HIDDEN, dtype=torch.bfloat16)
            if proj != "o_proj":
                t[f"{p}.self_attn.{proj}.bias"] = torch.zeros(
                    out, dtype=torch.bfloat16)
        t[f"{p}.mlp.gate_proj.weight"] = torch.zeros(64, HIDDEN, dtype=torch.bfloat16)
        t[f"{p}.mlp.up_proj.weight"] = torch.zeros(64, HIDDEN, dtype=torch.bfloat16)
        t[f"{p}.mlp.down_proj.weight"] = torch.zeros(HIDDEN, 64, dtype=torch.bfloat16)
    if with_lm_head:
        t["lm_head.weight"] = torch.zeros(VOCAB, HIDDEN, dtype=torch.bfloat16)
    return t


def _write_ckpt(tmp_path, name, *, tied: bool, with_lm_head: bool,
                embed_prefix: str = "model", vl_wrapper: bool = False) -> str:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps(_config(tied=tied, vl_wrapper=vl_wrapper), indent=2))
    tensors = _tensors(embed_prefix=embed_prefix, with_lm_head=with_lm_head)
    save_file(tensors, str(d / "model-00001-of-00001.safetensors"))
    (d / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * 2 for t in tensors.values())},
        "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors},
    }, indent=2))
    return str(d)


# --------------------------------------------------------------------------
# Detection: config declaration AND absence of a head tensor
# --------------------------------------------------------------------------
class TestTieDetection:
    def test_declaration_read_from_top_level_and_text_config(self):
        assert config_declares_tied_embeddings({"tie_word_embeddings": True})
        assert config_declares_tied_embeddings(
            {"text_config": {"tie_word_embeddings": True}})
        assert not config_declares_tied_embeddings({"tie_word_embeddings": False})
        assert not config_declares_tied_embeddings({})

    def test_declaration_read_from_config_objects(self):
        class _Cfg:
            tie_word_embeddings = False

        class _Text:
            tie_word_embeddings = True

        cfg = _Cfg()
        assert not config_declares_tied_embeddings(cfg)
        cfg.text_config = _Text()
        assert config_declares_tied_embeddings(cfg)

    def test_tied_without_head_tensor_is_an_alias(self, tmp_path):
        p = _write_ckpt(tmp_path, "tied", tied=True, with_lm_head=False)
        assert not source_has_lm_head_tensor(p, "lm_head")
        assert lm_head_is_tied_alias(p)

    def test_untied_with_head_tensor_is_not_an_alias(self, tmp_path):
        p = _write_ckpt(tmp_path, "untied", tied=False, with_lm_head=True)
        assert source_has_lm_head_tensor(p, "lm_head")
        assert not lm_head_is_tied_alias(p)

    def test_declared_tied_but_head_tensor_present_is_not_an_alias(self, tmp_path):
        """The declaration alone is not enough: a checkpoint that ships an
        independent head has one to quantize, whatever the config says."""
        p = _write_ckpt(tmp_path, "both", tied=True, with_lm_head=True)
        assert not lm_head_is_tied_alias(p)

    def test_vl_prefixed_embedding_resolves(self, tmp_path):
        """gemma-4-31b-it's embedding key is
        `model.language_model.embed_tokens.weight`; detection must not care."""
        p = _write_ckpt(tmp_path, "vl", tied=True, with_lm_head=False,
                        embed_prefix="model.language_model", vl_wrapper=True)
        assert lm_head_is_tied_alias(p)

    def test_wrapper_prefixed_head_tensor_counts_as_present(self, tmp_path):
        p = _write_ckpt(tmp_path, "vlhead", tied=True, with_lm_head=False,
                        embed_prefix="model.language_model", vl_wrapper=True)
        idx = json.loads((tmp_path / "vlhead" /
                          "model.safetensors.index.json").read_text())
        idx["weight_map"]["model.language_model.lm_head.weight"] = \
            "model-00001-of-00001.safetensors"
        (tmp_path / "vlhead" / "model.safetensors.index.json").write_text(
            json.dumps(idx))
        assert source_has_lm_head_tensor(str(tmp_path / "vlhead"), "lm_head")
        assert not lm_head_is_tied_alias(str(tmp_path / "vlhead"))


# --------------------------------------------------------------------------
# Materialization: the alias, resolved through transformers' accessors
# --------------------------------------------------------------------------
class _Toy(nn.Module):
    """Minimal stand-in with the accessor contract every HF causal LM has.
    `embed_path` mimics either the plain or the VL-wrapper nesting."""

    def __init__(self, cfg, *, vl: bool):
        super().__init__()
        self.config = cfg
        emb = nn.Embedding(VOCAB, HIDDEN)
        emb.weight = nn.Parameter(torch.randn(VOCAB, HIDDEN))
        if vl:
            inner = nn.Module()
            inner.embed_tokens = emb
            wrapper = nn.Module()
            wrapper.language_model = inner
            self.model = wrapper
        else:
            wrapper = nn.Module()
            wrapper.embed_tokens = emb
            self.model = wrapper
        self._emb = emb
        with torch.device("meta"):
            self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def get_input_embeddings(self):
        return self._emb

    def get_output_embeddings(self):
        return self.lm_head


class _TiedCfg:
    tie_word_embeddings = True


class _UntiedCfg:
    tie_word_embeddings = False


class TestAliasMaterialization:
    @pytest.mark.parametrize("vl", [False, True])
    def test_meta_head_is_aliased_to_the_embedding(self, vl):
        m = _Toy(_TiedCfg(), vl=vl)
        assert m.lm_head.weight.is_meta
        assert resolve_tied_output_embedding(m) is True
        assert not m.lm_head.weight.is_meta
        # Same storage — the whole point of the tie.
        assert m.lm_head.weight is m.get_input_embeddings().weight
        # And the exact operation that crashed now works.
        w = torch.stack([m.lm_head.weight.detach().to(
            device="cpu", dtype=torch.float32)])
        assert w.shape == (1, VOCAB, HIDDEN)

    def test_no_meta_parameters_left_behind(self):
        m = _Toy(_TiedCfg(), vl=True)
        resolve_tied_output_embedding(m)
        assert [n for n, p in m.named_parameters() if p.is_meta] == []

    def test_untied_model_with_a_real_head_is_untouched(self):
        m = _Toy(_UntiedCfg(), vl=False)
        m.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        before = m.lm_head.weight.clone()
        assert resolve_tied_output_embedding(m) is False
        assert torch.equal(m.lm_head.weight, before)
        assert m.lm_head.weight is not m.get_input_embeddings().weight

    def test_meta_head_without_a_declared_tie_fails_loudly(self):
        """A missing head weight on an untied model is a broken checkpoint,
        not a tie. Fail here, not thousands of lines downstream."""
        m = _Toy(_UntiedCfg(), vl=False)
        with pytest.raises(RuntimeError, match="does NOT declare"):
            resolve_tied_output_embedding(m)

    def test_model_without_output_embeddings_is_a_noop(self):
        assert resolve_tied_output_embedding(nn.Linear(2, 2)) is False


# --------------------------------------------------------------------------
# The streaming context itself — the path that actually broke
# --------------------------------------------------------------------------
@pytest.mark.parametrize("explicit,environment,expected", [(None, "auto", 26), (None, "11", 11), (7, "11", 7)])
def test_cache_headroom_uses_resolved_shard_count(tmp_path, monkeypatch, explicit, environment, expected):
    from prismaquant import autoscale, streaming_model
    monkeypatch.setenv("LAYERS_PER_SHARD", "auto")
    monkeypatch.setenv("CACHE_HEADROOM_GB", environment)
    monkeypatch.setenv("PRISMAQUANT_TMPDIR", str(tmp_path / "stage"))
    monkeypatch.setattr(autoscale, "estimate_per_layer_bytes", lambda *a, **k: (0, 2 * 1024**3))
    monkeypatch.setattr(streaming_model, "mem_available_bytes", lambda: 128 * 1024**3)
    path = _write_ckpt(tmp_path, "headroom", tied=False, with_lm_head=True)
    ctx = streaming_model._build_streaming_context(path, device=torch.device("cpu"),
        dtype=torch.bfloat16, offload_folder=str(tmp_path / "offload"), layers_per_shard=3,
        cache_headroom_gb=explicit, prefetch_workers=1, prefetch_min_available_gb=0)
    try:
        assert ctx.layer_cache.max_bytes == (128 - expected) * 1024**3
        assert ctx.layer_cache._dynamic_reserve_bytes == expected * 1024**3
    finally:
        ctx.shutdown()


class TestStreamingContext:
    @pytest.mark.parametrize("tied,with_head", [(True, False), (False, True)])
    def test_head_is_never_left_on_meta(self, tmp_path, tied, with_head):
        from prismaquant.streaming_model import _build_streaming_context

        path = _write_ckpt(tmp_path, f"ck{int(tied)}", tied=tied,
                           with_lm_head=with_head)
        ctx = _build_streaming_context(
            path,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            offload_folder=str(tmp_path / f"off{int(tied)}"), layers_per_shard=1,
            log_prefix="[test]",
        )
        head = ctx.model.get_output_embeddings()
        assert head is not None and not head.weight.is_meta
        # The exact `measure_batched_gpu` stack that raised
        # "Cannot copy out of meta tensor" on gemma-4-31b-it.
        stacked = torch.stack([head.weight.detach().to(
            device="cpu", dtype=torch.float32)])
        assert stacked.shape == (1, VOCAB, HIDDEN)
        if tied:
            assert head.weight is ctx.model.get_input_embeddings().weight


# --------------------------------------------------------------------------
# Non-quantizability: no cost shard, no DP entry
# --------------------------------------------------------------------------
class TestExcludedFromCostAndAllocation:
    def _schedule(self, path):
        from prismaquant.incremental_probe import build_shard_schedule
        return build_shard_schedule(
            model_path=path,
            num_body_layers=LAYERS,
            body_layers_per_shard=1,
            body_layer_range=(0, LAYERS),
            include_mtp=False,
            include_visual=False,
            include_lm_head=True,
            unified_body_sweep=False,
        )

    def test_tied_head_gets_no_shard(self, tmp_path):
        sched = self._schedule(_write_ckpt(tmp_path, "tied", tied=True,
                                           with_lm_head=False))
        assert [e.kind for e in sched] == ["body"] * LAYERS
        assert "^lm_head$" not in sched.regexes()
        # Shard indices stay contiguous from 0.
        assert [e.shard_idx for e in sched] == list(range(LAYERS))

    def test_untied_head_still_gets_its_shard(self, tmp_path):
        sched = self._schedule(_write_ckpt(tmp_path, "untied", tied=False,
                                           with_lm_head=True))
        assert [e.kind for e in sched] == ["body"] * LAYERS + ["lm_head"]
        assert "^lm_head$" in sched.regexes()

    def test_allocator_excludes_a_tied_head_from_the_dp_budget(self, tmp_path):
        """Covers probes written before the shard-schedule fix, which still
        carry an lm_head row."""
        from prismaquant.allocator import tied_lm_head_dp_exclusions
        from prismaquant.model_profiles import DefaultProfile

        path = _write_ckpt(tmp_path, "tied", tied=True, with_lm_head=False)
        stats = {"model.layers.0.self_attn.q_proj": {}, "lm_head": {}}
        assert tied_lm_head_dp_exclusions(
            stats, {}, DefaultProfile(), path) == ["lm_head"]

    def test_allocator_leaves_an_untied_head_alone(self, tmp_path):
        from prismaquant.allocator import tied_lm_head_dp_exclusions
        from prismaquant.model_profiles import DefaultProfile

        path = _write_ckpt(tmp_path, "untied", tied=False, with_lm_head=True)
        stats = {"model.layers.0.self_attn.q_proj": {}, "lm_head": {}}
        assert tied_lm_head_dp_exclusions(
            stats, {}, DefaultProfile(), path) == []

    def test_allocator_exclusion_is_not_overridable_by_allow_pinned(self,
                                                                   tmp_path):
        """`--allow-pinned lm_head` trades an INDEPENDENT head's bytes for
        quality. A tied head has no bytes of its own to trade, and unpinning
        it would quantize the input embedding."""
        import prismaquant.allocator as alloc
        from prismaquant.model_profiles import DefaultProfile

        path = _write_ckpt(tmp_path, "tied", tied=True, with_lm_head=False)
        profile = DefaultProfile()
        stats = {"lm_head": {}}
        allow_pinned = ["lm_head"]
        allocation_excluded = []
        # Replay the allocator's own pin loop (allocator.py, `--allow-pinned`
        # branch): with lm_head whitelisted the pin no longer excludes it ...
        for name in sorted(stats):
            if profile.is_pinned_name(name):
                if any(tok in name for tok in allow_pinned):
                    continue
                allocation_excluded.append(name)
        assert profile.is_pinned_name("lm_head")
        assert allocation_excluded == []
        # ... but the structural tie exclusion still does.
        allocation_excluded.extend(alloc.tied_lm_head_dp_exclusions(
            stats, {}, profile, path, allocation_excluded))
        assert allocation_excluded == ["lm_head"]
