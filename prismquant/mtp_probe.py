#!/usr/bin/env python3
"""mtp_probe.py — Fisher-diagonal sensitivity probe for MTP (multi-token
prediction) heads on Qwen3.5/3.6 MoE.

Transformers v5 ships no MTP module for these models (the top-level
PreTrainedModel has `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`,
so MTP weights are silently dropped on load). MTP is a vLLM-only runtime
feature. To get real Fisher stats on MTP Linears we:

  1. Instantiate an HF `Qwen3_5MoeDecoderLayer` with `layer_type="full_attention"`
     to mirror the single MTP block.
  2. Wrap it in an MTP module that applies the two pre_fc_norms + `mtp.fc`
     concatenation exactly as vLLM's `Qwen3_5MultiTokenPredictor.forward`.
  3. Load `mtp.*` weights directly from safetensors into that module.
  4. Run one forward pass of the frozen body model to get final
     `hidden_states` (post model.norm) and reuse body's rotary_emb.
  5. Feed `(token_embeddings_shifted_by_1, body_hidden_states)` into MTP
     and compute MTP's aux CE loss (label = token at position t+2).
  6. Backward -> Fisher hooks accumulate for MTP Linears.

Output is a probe-compatible pickle with stats keyed by the original
checkpoint name (`mtp.layers.0.self_attn.q_proj` etc.), suitable for
merging into the body probe.pkl.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open

from .sensitivity_probe import (
    FisherAccumulator,
    install_packed_expert_hooks,
    load_calibration,
    load_probe_model_and_tokenizer,
    resolve_execution_device,
    stage_text_only,
)


# ---------------------------------------------------------------------------
# MTP module
# ---------------------------------------------------------------------------

def _build_single_layer_config(text_config):
    """Return a `Qwen3_5MoeTextConfig` (or compatible) with exactly one
    decoder layer of type 'full_attention'. This matches vLLM's MTP:
    one full-attention decoder block per MTP step.

    `copy.deepcopy` is used so the body's config is untouched and
    gradient checkpointing state on the original model doesn't leak."""
    cfg = copy.deepcopy(text_config)
    cfg.layer_types = ["full_attention"]
    cfg.num_hidden_layers = 1
    return cfg


class MtpModule(nn.Module):
    """Mirrors `vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MultiTokenPredictor`
    but built on HF primitives so Fisher hooks and autograd work normally."""

    def __init__(self, text_config):
        super().__init__()
        # Lazy import: the HF module path changes when the container is rebuilt.
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer,
            Qwen3_5MoeRMSNorm,
        )
        mtp_cfg = _build_single_layer_config(text_config)
        hidden = mtp_cfg.hidden_size
        eps = mtp_cfg.rms_norm_eps

        self.fc = nn.Linear(hidden * 2, hidden, bias=False)
        self.layers = nn.ModuleList([Qwen3_5MoeDecoderLayer(mtp_cfg, layer_idx=0)])
        self.norm = Qwen3_5MoeRMSNorm(hidden, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(hidden, eps=eps)
        self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(hidden, eps=eps)

    def forward(self,
                inputs_embeds: torch.Tensor,
                body_hidden_states: torch.Tensor,
                position_embeddings,
                causal_mask,
                position_ids):
        e = self.pre_fc_norm_embedding(inputs_embeds)
        h = self.pre_fc_norm_hidden(body_hidden_states)
        h = torch.cat([e, h], dim=-1)
        h = self.fc(h)
        h = self.layers[0](
            hidden_states=h,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        if isinstance(h, tuple):
            h = h[0]
        h = self.norm(h)
        return h


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_mtp_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Return every tensor whose key starts with `mtp.`, stripped of
    that prefix so it matches `MtpModule`'s module layout. We do not
    materialize all shards; only the ones that actually hold MTP keys."""
    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    if not idx_path.exists():
        raise RuntimeError(f"no safetensors index at {idx_path}")
    with open(idx_path) as f:
        wm = json.load(f)["weight_map"]
    mtp_files = sorted({wm[k] for k in wm if k.startswith("mtp.")})
    if not mtp_files:
        raise RuntimeError("no mtp.* weights in safetensors index")
    out: dict[str, torch.Tensor] = {}
    for fn in mtp_files:
        with safe_open(str(src / fn), framework="pt") as sf:
            for key in sf.keys():
                if not key.startswith("mtp."):
                    continue
                t = sf.get_tensor(key)
                out[key[len("mtp."):]] = t
    return out


def _load_into_mtp(mtp: MtpModule, raw: dict[str, torch.Tensor]):
    """Map checkpoint keys (mtp.* stripped) onto MtpModule's layout.

    Checkpoint layout            -> MtpModule path
      fc.weight                  -> fc.weight
      pre_fc_norm_embedding...   -> pre_fc_norm_embedding.weight
      pre_fc_norm_hidden...      -> pre_fc_norm_hidden.weight
      norm.weight                -> norm.weight
      layers.0.<rest>            -> layers.0.<rest>

    The HF `Qwen3_5MoeDecoderLayer` stores packed experts as 3D
    `mlp.experts.gate_up_proj` / `down_proj`, matching the checkpoint."""
    sd = mtp.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for k, v in raw.items():
        if k in sd:
            mapped[k] = v
        else:
            missing.append(k)
    extra = [k for k in sd if k not in mapped]
    # Load with strict=False so any HF-internal buffer (e.g. sparse
    # expert state that we don't ship) doesn't break loading.
    mtp.load_state_dict(mapped, strict=False)
    return missing, extra


# ---------------------------------------------------------------------------
# Probe driver
# ---------------------------------------------------------------------------

def _run_body_forward_for_hidden_states(model, calib_ids, exec_device):
    """Forward the body model (frozen) and return per-sample:
      - inputs_embeds   (B, T, H)
      - final hidden_states (post model.norm)  (B, T, H)
      - position_embeddings (cos, sin)         (same shapes as inside attn)
      - causal_mask    (whatever `create_causal_mask` produced)
      - text_position_ids (B, T)
    """
    from transformers.masking_utils import create_causal_mask

    base = model.model  # Qwen3_5MoeTextModel
    config = base.config
    B, T = calib_ids.shape

    # Token embeddings
    embed = base.embed_tokens(calib_ids)
    # Default position_ids: arange, broadcast for 4-axis convention.
    position_ids = (
        torch.arange(T, device=exec_device).view(1, 1, T).expand(4, B, T)
    )
    text_position_ids = position_ids[0]
    # Causal mask matches body's usage.
    causal_mask = create_causal_mask(
        config=config,
        inputs_embeds=embed,
        attention_mask=None,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    # Position embeddings (RoPE cos/sin) come from body's rotary_emb.
    # Expand position_ids to 3D (non-text) before passing in — body calls
    # `self.rotary_emb(hidden_states, position_ids)` with the 4-axis form
    # after slicing off text_position_ids.
    position_embeddings = base.rotary_emb(embed, position_ids[1:])

    # Body forward under no_grad for hidden-state capture.
    with torch.no_grad():
        out = base(
            input_ids=calib_ids,
            attention_mask=None,
            use_cache=False,
            return_dict=True,
        )
    body_hidden = out.last_hidden_state

    return embed, body_hidden, position_embeddings, causal_mask, text_position_ids


def run_mtp_probe(model_path: str,
                  dataset: str,
                  nsamples: int,
                  seqlen: int,
                  device: str,
                  dtype: torch.dtype,
                  output_path: str,
                  activation_cache_dir: str | None,
                  importance_weighting: bool):
    staged, tokenizer, body_model, exec_device, load_device_map = \
        load_probe_model_and_tokenizer(
            model_path,
            requested_device=device,
            dtype=dtype,
            device_map=device,
            gradient_checkpointing=False,
        )

    # 1. Build MTP module from body's text_config, wrapped so every
    # child is reachable as `mtp.<path>` — matching the checkpoint's
    # naming convention and what `mtp_cost.pkl` / `mtp_probe.pkl` emit.
    text_config = body_model.config
    inner_mtp = MtpModule(text_config)
    mtp_wrapper = nn.Module()
    mtp_wrapper.add_module("mtp", inner_mtp)
    mtp_wrapper.to(device=exec_device, dtype=dtype)
    mtp_wrapper.eval()

    # 2. Load mtp.* weights.
    raw = _load_mtp_state_dict(model_path)
    missing, extra = _load_into_mtp(inner_mtp, raw)
    loaded = len(raw) - len(missing)
    print(f"[mtp-probe] loaded {loaded}/{len(raw)} mtp weights into MtpModule "
          f"(missing_from_module={len(missing)}, module_params_without_weight={len(extra)})",
          flush=True)
    if missing:
        print(f"[mtp-probe] unmatched checkpoint keys: {missing[:5]}"
              + ("..." if len(missing) > 5 else ""), flush=True)

    # 3. Freeze MTP parameters (we want gradients to flow but we don't
    # need .grad buffers on leaves — the FisherAccumulator captures
    # ||grad_w||² through the forward/backward hooks instead).
    for p in mtp_wrapper.parameters():
        p.requires_grad_(False)

    # 4. Install Fisher hooks on every Linear in the wrapped MTP module,
    # excluding the MoE router so we don't treat it as quantizable.
    tracked = [n for n, m in mtp_wrapper.named_modules()
               if isinstance(m, nn.Linear) and not re.search(r"mlp\.gate$", n)]
    print(f"[mtp-probe] tracking {len(tracked)} MTP Linears", flush=True)

    # 5. Discover packed experts via the existing helper.
    from .sensitivity_probe import discover_moe_structure, read_top_k
    expert_info_all = discover_moe_structure(mtp_wrapper)
    expert_info = {k: v for k, v in expert_info_all.items() if k in tracked}
    top_k = read_top_k(mtp_wrapper, default=2)
    print(f"[mtp-probe] MoE: {len(expert_info)} expert linears, top_k={top_k}",
          flush=True)

    # Fisher accumulator. We cache MTP activations to disk so
    # `mtp_cost.py` can drive the same batched output-MSE path the body
    # cost uses — giving us parity between body and MTP cost artifacts.
    cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    acc = FisherAccumulator(mtp_wrapper, tracked, expert_info, cache_dir)

    # 6. Prepare calibration.
    calib = load_calibration(tokenizer, dataset, nsamples, seqlen)
    print(f"[mtp-probe] calibration shape: {tuple(calib.shape)}", flush=True)

    # 7. Fetch lm_head (tied / untied, via body model).
    lm_head = body_model.get_output_embeddings()
    assert isinstance(lm_head, nn.Linear), "lm_head must be a Linear for MTP CE"

    t_fwd = t_bwd = 0.0
    for i in range(calib.size(0)):
        ids = calib[i:i + 1].to(exec_device)
        t0 = time.time()

        # Body forward under no_grad to get hidden states and cached
        # position embeddings. The MTP loss is local to MTP weights;
        # we don't need grads to propagate back into the body.
        embed, body_hidden, pos_emb, causal_mask, text_pos = \
            _run_body_forward_for_hidden_states(body_model, ids, exec_device)

        # MTP input token embedding is shifted by +1 relative to the
        # hidden state input (MTP predicts token t+2 from hidden_t and
        # embed_{t+1}). Drop the last position so shapes match labels.
        # After shift: hidden_states[..., :-1, :], embed[..., 1:, :], target=ids[..., 2:]
        shifted_embed = embed[:, 1:-1, :].contiguous()      # drops [BOS] and last
        shifted_hidden = body_hidden[:, :-2, :].contiguous()  # aligns with +2 target
        target_ids = ids[:, 2:].contiguous()
        # Positional info for the shifted sequence:
        B, T2, _ = shifted_embed.shape
        trimmed_pos_ids = torch.arange(T2, device=exec_device).view(1, T2).expand(B, T2)
        # Recompute a small causal mask and rotary pos embeddings for T2.
        from transformers.masking_utils import create_causal_mask
        causal_mask_t2 = create_causal_mask(
            config=text_config,
            inputs_embeds=shifted_embed,
            attention_mask=None,
            past_key_values=None,
            position_ids=trimmed_pos_ids,
        )
        # Rotary emb requires a 4-axis position_ids with rank 3 (axes 0=text, 1/2/3=spatial).
        rot_pos = trimmed_pos_ids.view(1, B, T2).expand(3, B, T2)
        pos_emb_t2 = body_model.model.rotary_emb(shifted_embed, rot_pos)

        # Mark that hidden states need grad (for Fisher hook capture).
        shifted_hidden = shifted_hidden.detach().requires_grad_(True)
        shifted_embed = shifted_embed.detach().requires_grad_(True)

        inner_mtp.train()
        out_hidden = inner_mtp(
            inputs_embeds=shifted_embed,
            body_hidden_states=shifted_hidden,
            position_embeddings=pos_emb_t2,
            causal_mask=causal_mask_t2,
            position_ids=trimmed_pos_ids,
        )
        logits = lm_head(out_hidden)
        t_fwd += time.time() - t0

        t0 = time.time()
        lp = F.log_softmax(logits.reshape(-1, logits.size(-1)), dim=-1)
        gather = -lp.gather(1, target_ids.reshape(-1, 1)).squeeze(1)

        if importance_weighting:
            with torch.no_grad():
                mean = float(gather.mean().item())
            w = (gather.detach() / max(mean, 1e-6)).clamp(0.25, 4.0)
            loss = (gather * w).sum()
        else:
            loss = gather.sum()
        loss.backward()
        t_bwd += time.time() - t0

        n_tok = max(int(gather.numel()), 1)
        mean_loss = float(loss.detach().item()) / n_tok
        print(f"[mtp-probe] sample {i+1}/{calib.size(0)} "
              f"loss={mean_loss:.3f} fwd_avg={t_fwd/(i+1):.2f}s "
              f"bwd_avg={t_bwd/(i+1):.2f}s", flush=True)

        del out_hidden, logits, loss, gather
        acc._saved_inputs.clear()
        if exec_device.type == "cuda":
            torch.cuda.empty_cache()

    acc.finalize(tracker=None)
    acc.remove_hooks()

    # Stats / expert_info already carry the `mtp.` prefix because the
    # FisherAccumulator walked `mtp_wrapper.named_modules()`.
    renamed = dict(acc.stats)
    expert_info_renamed = dict(expert_info)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "stats": renamed,
            "router_counts": {},
            "router_totals": {},
            "expert_info": expert_info_renamed,
            "meta": {
                "model": model_path,
                "dataset": dataset,
                "nsamples": nsamples,
                "seqlen": seqlen,
                "dtype": str(dtype),
                "device": device,
                "mtp_probe": True,
                "mtp_objective": "CE(lm_head(MTP(embed_{t+1}, body_hidden_t)), ids_{t+2})",
            },
        }, f)
    print(f"[mtp-probe] wrote {output_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ultrachat_200k")
    ap.add_argument("--nsamples", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--output", required=True)
    ap.add_argument("--activation-cache-dir", default=None)
    ap.add_argument("--importance-weighting", action="store_true", default=True)
    ap.add_argument("--no-importance-weighting", action="store_false",
                    dest="importance_weighting")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    run_mtp_probe(
        model_path=args.model,
        dataset=args.dataset,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        device=args.device,
        dtype=dtype,
        output_path=args.output,
        activation_cache_dir=args.activation_cache_dir,
        importance_weighting=args.importance_weighting,
    )


if __name__ == "__main__":
    main()
