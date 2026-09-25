"""GLM-5.3-Flash's MTP layer as a priced module (PQ #1271).

The GLM checkpoint ships its multi-token-prediction (MTP) head as layer
``num_hidden_layers`` of the text tower: ``model.language_model.layers.45.*``
for GLM-5.3-Flash. Transformers does not build it (its decoder stack stops at
``num_hidden_layers``), and PrismaQuant's streamed body runner does not install
it. vLLM serves it as a drafter (``Glm5NextMTP``). This module builds that
layer so its Linears can be priced and rendered like body Linears.

The forward mirrors the served drafter, not a training recipe. From vLLM's
``glm5next/nvidia/mtp.py`` and ``model.py`` in the pinned serving image:

* The drafter's input at target position ``t`` is ``embed(token_{t+1})`` and
  the target's post-final-norm hidden state ``h_t``. The embedding is zeroed
  where the position is 0 (``fused_eh_norm``).
* ``fused_eh_norm`` computes ``cat([enorm(e), hnorm(h)])``. Each RMSNorm is
  computed in fp32, multiplied by its fp32 weight, and cast once. This module
  keeps that order.
* ``eh_proj`` is a plain ``nn.Linear``, then one decoder layer on the non-mHC
  path: MLA attention with its own DSA indexer, then MoE. The layer returns
  ``(mlp_out, residual)`` unsummed.
* ``shared_head.norm`` fuses that residual add. The logits come from
  ``shared_head.head``. The checkpoint carries no such tensor, and vLLM's
  proposer binds the target ``lm_head`` there unconditionally, so callers pass
  the target head.

Only the MoE (routed experts and the shared expert) consults the serving
quantization config. vLLM builds the MLA with ``quant_config=None``;
``eh_proj`` is not a ``LinearBase``; and the router, the norms and the head
are not quantized. So the priced population is the routed experts plus the
shared expert.

The two residual adds follow ``fused_add_rms_norm``: the sum is taken in the
activation dtype, then normalized in fp32. vLLM's GLM comment says the sum is
fp32-accumulated inside the fused kernel. The two spellings differ by one
rounding of the residual sum in bf16 and agree exactly in fp32. This module
does not claim which kernel the serving image dispatches.

**The indexer regime.** The DSA indexer selects ``index_topk // index_kpool``
key pools. When a sequence has no more pools than that, it selects every
causally visible key, and the sparse attention *is* dense causal attention.
GLM-5.3-Flash has topk 2048 and kpool 4, so the 512-token calibration has 128
pools against 512 selectable, and GLM always selects a query's own
incomplete tail pool (``index_kpool_always_select_tail``). The parity test
checks this module against a
dense-causal reference only in that regime, so :func:`require_dense_indexer_regime`
refuses a longer sequence rather than price a regime nothing has checked.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .kl_fisher import fisher_probe_scalar

#: The MTP objective's identity schema.
MTP_OBJECTIVE_SCHEMA = "prismaquant.joint_aura.mtp_objective.v1"
#: What the MTP layer's rows are priced on: the MTP head's own next-next-token
#: distribution, second-order KL against the BF16 MTP head.
MTP_OBJECTIVE = "mtp_head_self_kl"


def _glm5_modeling():
    from transformers.models.glm5_next import modeling_glm5_next

    return modeling_glm5_next


def mtp_layer_index(text_config) -> int:
    """The MTP layer's index in the checkpoint: the backbone depth.

    Refuses a config that declares anything but one MTP layer. vLLM cycles
    ``spec_step_idx % num_mtp_layers`` over several; nothing here prices more
    than one, and a silent first-layer-only answer would misprice the rest.
    """
    count = int(getattr(text_config, "num_nextn_predict_layers", 0) or 0)
    if count != 1:
        raise ValueError(
            f"GLM MTP pricing covers exactly one MTP layer; the config declares {count}")
    return int(text_config.num_hidden_layers)


def mtp_text_config(text_config):
    """A copy of ``text_config`` whose per-layer lists reach the MTP index.

    Transformers' GLM layers read ``layer_types``, ``indexer_types`` and
    ``mlp_layer_types`` at their own index, and those lists stop at the
    backbone. The values appended are the served drafter's:

    * attention: MLA with a DSA indexer (vLLM builds MLA for every MTP layer,
      ``is_kda_layer`` is false there);
    * indexer: ``"full"``, since the MTP layer carries its own indexer weights;
    * MLP: the last backbone layer's type, which is vLLM's rule for an index
      past ``mlp_layer_types``.

    The input config is not modified.
    """
    index = mtp_layer_index(text_config)
    config = copy.deepcopy(text_config)
    for field in ("layer_types", "indexer_types", "mlp_layer_types"):
        values = list(getattr(config, field))
        if len(values) != index:
            raise ValueError(
                f"{field} has {len(values)} entries for a {index}-layer backbone")
    config.layer_types = list(config.layer_types) + ["deepseek_sparse_attention"]
    config.indexer_types = list(config.indexer_types) + ["full"]
    config.mlp_layer_types = list(config.mlp_layer_types) + [config.mlp_layer_types[-1]]
    if config.mlp_layer_types[index] != "sparse":
        raise ValueError("GLM MTP pricing expects a routed-MoE MTP layer")
    return config


def require_dense_indexer_regime(config, sequence_length: int) -> None:
    """Refuse a sequence long enough for the DSA indexer to drop keys."""
    kpool = max(int(config.index_kpool or 1), 1)
    if kpool > 1 and not getattr(config, "index_kpool_always_select_tail", False):
        raise ValueError(
            "the MTP forward is checked only with index_kpool_always_select_tail; "
            "without it a query's own incomplete pool is not selected")
    pools = -(-int(sequence_length) // kpool)
    selectable = int(config.index_topk) // kpool
    if pools > selectable:
        raise ValueError(
            f"MTP sequence of {sequence_length} tokens forms {pools} key pools; the "
            f"indexer selects {selectable}, so attention would be sparse. The MTP "
            "forward is checked only where the indexer keeps every key")


def _fused_eh_norm(positions, inputs_embeds, previous_hidden, enorm_weight,
                   hnorm_weight, eps):
    """vLLM's ``fused_eh_norm`` in torch: fp32 RMS times fp32 weight, one cast."""
    def rms(x, weight):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x * weight.float()

    embeds = inputs_embeds.float().masked_fill((positions == 0)[..., None], 0.0)
    out = torch.cat([rms(embeds, enorm_weight), rms(previous_hidden, hnorm_weight)], dim=-1)
    return out.to(inputs_embeds.dtype)


class _SharedHead(nn.Module):
    """``shared_head``: only its norm is a weight; the head is the target's."""

    def __init__(self, config):
        super().__init__()
        self.norm = _glm5_modeling().Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)


class Glm5NextMtpLayer(nn.Module):
    """The served GLM MTP layer, built from Transformers' GLM modules.

    Parameter names are the checkpoint's, relative to
    ``model.language_model.layers.<N>.``. The only difference is the routed
    experts: they are packed (``mlp.experts.gate_up_proj``,
    ``mlp.experts.down_proj``) exactly as in the backbone's MoE layers, where
    the checkpoint stores one tensor per expert and projection.
    """

    def __init__(self, text_config):
        super().__init__()
        modeling = _glm5_modeling()
        config = mtp_text_config(text_config)
        self.config = config
        self.layer_idx = mtp_layer_index(text_config)
        hidden = config.hidden_size
        eps = config.rms_norm_eps
        self.enorm = modeling.Glm5NextTextRMSNorm(hidden, eps)
        self.hnorm = modeling.Glm5NextTextRMSNorm(hidden, eps)
        self.eh_proj = nn.Linear(2 * hidden, hidden, bias=False)
        self.input_layernorm = modeling.Glm5NextTextRMSNorm(hidden, eps)
        self.self_attn = modeling.Glm5NextTextAttention(config, self.layer_idx)
        self.post_attention_layernorm = modeling.Glm5NextTextRMSNorm(hidden, eps)
        self.mlp = modeling.Glm5NextTextMoE(config)
        self.shared_head = _SharedHead(config)
        if self.self_attn.indexer is None or self.self_attn.next_skip_topk:
            raise ValueError("the MTP layer must run its own indexer and share nothing onward")

    def forward(self, inputs_embeds, previous_hidden, positions):
        """The drafter's post-norm hidden state, ``[batch, rows, hidden]``.

        ``inputs_embeds[:, t]`` is ``embed(token_{t+1})``, ``previous_hidden[:, t]``
        is the target's post-final-norm hidden state at ``t``, and
        ``positions[:, t]`` is ``t``. The rows attend causally to one another,
        which is the drafter's prefill over a prompt.
        """
        if inputs_embeds.shape != previous_hidden.shape or inputs_embeds.ndim != 3:
            raise ValueError("MTP embeddings and hidden states must be matching [B, S, H]")
        if tuple(positions.shape) != tuple(inputs_embeds.shape[:2]):
            raise ValueError("MTP positions must be [B, S]")
        batch, rows = inputs_embeds.shape[:2]
        require_dense_indexer_regime(self.config, rows)
        eh_input = _fused_eh_norm(positions, inputs_embeds, previous_hidden,
                                  self.enorm.weight, self.hnorm.weight,
                                  self.enorm.variance_epsilon)
        hidden = self.eh_proj(eh_input)
        padding = torch.ones(batch, rows, dtype=torch.bool, device=hidden.device)
        attention, _, _ = self.self_attn(self.input_layernorm(hidden), attention_mask=padding)
        residual = attention + hidden
        mlp_out = self.mlp(self.post_attention_layernorm(residual))
        return self.shared_head.norm(mlp_out + residual)


def mtp_checkpoint_prefix(layer_index: int) -> str:
    """Where layer ``layer_index`` lives in the checkpoint and the census."""
    return f"model.language_model.layers.{int(layer_index)}."


class MtpCheckpointModel(nn.Module):
    """The MTP layer under its checkpoint name, for the body's collectors.

    ``named_modules()`` yields ``model.language_model.layers.<N>.*``, the
    names the census, the capture and the export use for this layer. So the
    profile declares its packed experts (``profile_declared_packed_expert_projections``)
    exactly as it declares a body MoE layer's, and the campaign's collector
    runs on it unchanged. The call is the layer's own forward.
    """

    def __init__(self, layer: Glm5NextMtpLayer):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleDict({str(layer.layer_idx): layer})
        self.config = layer.config
        self._layer_key = str(layer.layer_idx)

    @property
    def layer(self) -> Glm5NextMtpLayer:
        return self.model.language_model.layers[self._layer_key]

    def forward(self, inputs_embeds, previous_hidden, positions):
        return self.layer(inputs_embeds, previous_hidden, positions)


def mtp_priced_units(model: MtpCheckpointModel, profile) -> dict:
    """``{qname: (out_features, in_features)}`` for every priced MTP Linear.

    The routed experts come from the profile's packed-expert declaration, the
    same views the exporter splits. The shared expert's three Linears are
    plain ``nn.Linear`` modules. Nothing else in the layer consults the
    serving quantization config (module docstring).
    """
    from .routed_experts import profile_declared_packed_expert_projections

    layer = model.layer
    prefix = mtp_checkpoint_prefix(layer.layer_idx)
    units = {member.qname: tuple(int(n) for n in member.weight.shape)
             for member in profile_declared_packed_expert_projections(model, profile)}
    shared = {name: module for name, module in layer.mlp.shared_experts.named_children()
              if isinstance(module, nn.Linear)}
    if set(shared) != {"gate_proj", "up_proj", "down_proj"}:
        raise RuntimeError(f"the shared expert's Linears are {sorted(shared)}")
    for name, module in shared.items():
        units[f"{prefix}mlp.shared_experts.{name}"] = tuple(int(n) for n in module.weight.shape)
    routed = [name for name in units if ".mlp.experts." in name]
    expected = 3 * int(layer.config.n_routed_experts)
    if len(routed) != expected:
        raise RuntimeError(f"the profile declares {len(routed)} routed MTP units, not {expected}")
    return dict(sorted(units.items()))


#: Parameters Transformers keeps in fp32 whatever the load dtype
#: (``Glm5NextPreTrainedModel._keep_in_fp32_modules_strict``).
def _keeps_fp32(name: str) -> bool:
    return name.endswith("e_score_correction_bias")


def _checkpoint_index(checkpoint_dir, source_authentication):
    import json
    from pathlib import Path

    path = Path(checkpoint_dir) / "model.safetensors.index.json"
    if source_authentication is not None:
        return source_authentication.read_json(path)["weight_map"]
    return json.loads(path.read_text())["weight_map"]


def _open_shard(path, source_authentication):
    from safetensors import safe_open

    if source_authentication is not None:
        return source_authentication.safe_open(safe_open, path, framework="pt", device="cpu")
    return safe_open(str(path), framework="pt", device="cpu")


def read_checkpoint_tensor(checkpoint_dir, key, *, source_authentication=None):
    """One checkpoint tensor by key, through the source owner when given."""
    from pathlib import Path

    weight_map = _checkpoint_index(checkpoint_dir, source_authentication)
    if key not in weight_map:
        raise KeyError(f"checkpoint has no tensor {key}")
    with _open_shard(Path(checkpoint_dir) / weight_map[key], source_authentication) as handle:
        return handle.get_tensor(key), weight_map[key]


def load_mtp_layer(checkpoint_dir, text_config, *, profile, dtype=torch.bfloat16,
                   device="cpu", experts_implementation, source_authentication=None):
    """Build the MTP layer and load it from the checkpoint's own tensors.

    Reads only ``model.language_model.layers.<N>.*`` through the safetensors
    index. The checkpoint stores routed experts one tensor per expert and
    projection. They are packed by the profile's per-expert to packed bridge
    (``layer_streaming._pack_per_expert_into_packed``), the one the streamed
    body loader uses, so the gate/up order is the profile's and not restated
    here. The load is strict both ways: a missing expert refuses in the
    bridge, and a tensor the layer does not have refuses in
    ``load_state_dict``.

    Floating tensors take ``dtype``, except those Transformers keeps in fp32.
    ``experts_implementation`` sets the routed experts' dispatch, so the
    caller can bind the body's recorded selector.

    ``source_authentication`` is a capture source owner
    (``tessera_calibration_cache.CaptureSourceAuthentication``). Given one,
    the index and every shard are read through it, so each shard's bytes are
    authenticated against the sealed source roster before a tensor is used.

    Returns ``(layer, receipt)``. The receipt names the shards read, the
    tensor count and the shard of every key read.
    """
    from pathlib import Path

    from .layer_streaming import _pack_per_expert_into_packed

    if not isinstance(experts_implementation, str) or not experts_implementation:
        raise ValueError("the MTP layer's experts dispatch must be named, not left to a default")
    checkpoint_dir = Path(checkpoint_dir)
    config = copy.deepcopy(text_config)
    config._experts_implementation = experts_implementation
    with torch.device("meta"):
        layer = Glm5NextMtpLayer(config)
    prefix = mtp_checkpoint_prefix(layer.layer_idx)
    expected = layer.state_dict()

    weight_map = _checkpoint_index(checkpoint_dir, source_authentication)
    keys = sorted(key for key in weight_map if key.startswith(prefix))
    if not keys:
        raise RuntimeError(f"checkpoint has no tensors under {prefix}")
    by_shard = {}
    for key in keys:
        by_shard.setdefault(weight_map[key], []).append(key)
    state = {}
    for shard in sorted(by_shard):
        with _open_shard(checkpoint_dir / shard, source_authentication) as handle:
            for key in by_shard[shard]:
                state[key[len(prefix):]] = handle.get_tensor(key)

    def is_per_expert(name):
        head, _, projection = name.rpartition(".")
        parent, _, index = head.rpartition(".")
        return parent == "mlp.experts" and index.isdigit() and bool(projection)

    def live_shape(name):
        tensor = expected.get(name)
        return None if tensor is None else tuple(tensor.shape)

    _pack_per_expert_into_packed(
        state, is_per_expert=is_per_expert,
        parent_for_projection=profile.packed_expert_parent_for_projection,
        projection_names_for=profile.packed_expert_projection_names,
        live_param_shape=live_shape)
    for name, tensor in list(state.items()):
        target = torch.float32 if _keeps_fp32(name) else dtype
        if tensor.is_floating_point():
            state[name] = tensor.to(device=device, dtype=target)
    unexpected = sorted(set(state) - set(expected))
    missing = sorted(set(expected) - set(state))
    if unexpected or missing:
        raise RuntimeError(
            f"MTP checkpoint tensors do not match the layer: unexpected {unexpected[:4]}, "
            f"missing {missing[:4]}")
    layer.load_state_dict(state, strict=True, assign=True)
    leftover = [name for name, tensor in list(layer.named_parameters()) + list(layer.named_buffers())
                if tensor.is_meta]
    if leftover:
        raise RuntimeError(f"MTP layer tensors left unloaded: {leftover[:4]}")
    layer.eval()
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    return layer, {"prefix": prefix, "shards": sorted(by_shard), "tensors": len(keys),
                   "source_map": {key: weight_map[key] for key in keys},
                   "experts_implementation": config._experts_implementation}


def mtp_layer_initialization_contract(layer, receipt, *, input_manifest):
    """The MTP capture's source-initialization contract, from the live layer.

    ``receipt`` is :func:`load_mtp_layer`'s. ``input_manifest`` is the
    ``{schema, path, sha256}`` of the hidden states the layer runs on. Every
    tensor is hashed as it is now, so computing this again after the capture
    shows the forward left the layer as it was loaded.
    """
    import transformers

    from prismaquant.stage_a_chain_seed import tensor_payload_sha256
    from prismaquant.streaming_model import (_MTP_LAYER_INITIALIZATION_SCHEMA,
                                             _initialization_digest,
                                             validate_mtp_layer_initialization_contract)

    state = {name: {"shape": [int(n) for n in tensor.shape], "dtype": str(tensor.dtype),
                    "sha256": tensor_payload_sha256(tensor)}
             for name, tensor in sorted(layer.state_dict().items())}
    dtypes = sorted({str(t.dtype) for name, t in layer.state_dict().items()
                     if t.is_floating_point() and not _keeps_fp32(name)})
    if len(dtypes) != 1:
        raise RuntimeError(f"MTP layer holds several load dtypes: {dtypes}")
    model_class = type(layer)
    return validate_mtp_layer_initialization_contract({
        "schema": _MTP_LAYER_INITIALIZATION_SCHEMA,
        "scope": "mtp_layer_checkpoint_load", "status": "completed",
        "transformers_version": transformers.__version__,
        "model_class": f"{model_class.__module__}.{model_class.__qualname__}",
        "dtype": dtypes[0], "layer_prefix": receipt["prefix"],
        "experts_implementation": str(receipt["experts_implementation"]),
        "checkpoint_tensors": int(receipt["tensors"]),
        "state": state, "state_sha256": _initialization_digest(state),
        "source_map_sha256": _initialization_digest(receipt["source_map"]),
        "input": {key: input_manifest[key] for key in ("schema", "path", "sha256")},
    })


def mtp_rows(input_ids, final_hidden):
    """The drafter's rows over a calibration sequence.

    Returns ``(next_ids, previous_hidden, positions)`` for target positions
    ``0..T-2``: row ``t`` pairs ``h_t`` with ``token_{t+1}`` and predicts
    ``token_{t+2}``. Position ``T-1`` has no next token inside the sequence,
    so it is not a row.
    """
    if input_ids.ndim != 2 or final_hidden.ndim != 3 or final_hidden.shape[:2] != input_ids.shape:
        raise ValueError("MTP rows need [B, T] ids and [B, T, H] post-norm hidden states")
    batch, length = input_ids.shape
    if length < 2:
        raise ValueError("MTP rows need sequences of at least two tokens")
    positions = torch.arange(length - 1, device=input_ids.device).expand(batch, -1)
    return input_ids[:, 1:], final_hidden[:, :-1], positions


def mtp_logits(layer, embed_tokens, lm_head, input_ids, final_hidden):
    """Draft logits ``[B, T-1, vocab]`` for a calibration batch."""
    next_ids, previous, positions = mtp_rows(input_ids, final_hidden)
    return lm_head(layer(embed_tokens(next_ids), previous, positions))


def mtp_global_token_count(n_sequences: int, sequence_length: int) -> int:
    """The MTP seed's normalizer: every formable row of every sequence."""
    if n_sequences < 1 or sequence_length < 2:
        raise ValueError("MTP token count needs a sequence and at least two tokens")
    return int(n_sequences) * (int(sequence_length) - 1)


def mtp_probe_scalar(logits, *, seed: int, global_row_offset: int, n_sequences: int):
    """The MTP layer's adjoint seed: the body's probe on the draft logits.

    ``kl_fisher.fisher_probe_scalar`` unchanged: Rademacher noise in the
    global-row layout, every row, temperature 1. It is normalized by
    :func:`mtp_global_token_count`, so ``0.5 * mean_k <G_k, dY>^2`` estimates
    the second-order KL of the draft distribution per MTP row, which is the
    unit the body's rows use per token. The row seed hashes the row geometry,
    so these draws differ from the body's by construction.
    """
    if logits.ndim != 3:
        raise ValueError("MTP probes need [B, T-1, vocab] logits")
    rows = int(logits.shape[1])
    return fisher_probe_scalar(
        logits, seed=int(seed), token_scope="all", temperature=1.0,
        distribution="rademacher",
        token_count_override=mtp_global_token_count(n_sequences, rows + 1),
        global_row_offset=int(global_row_offset),
    )


def mtp_objective_identity(*, mtp_layer: int, sequence_length: int, n_sequences: int) -> dict:
    """What an MTP row was priced on, for its probe identity.

    A probe identity carrying this block is a different measurement from a
    body row's: ``cost_currency.probe_identity_walls_differ`` sees the extra
    key, so a join of MTP and body rows refuses. That refusal is the point,
    because a draft's KL buys throughput and a body row's buys quality.
    """
    if mtp_layer < 1:
        raise ValueError("MTP layer index must follow a backbone")
    return {
        "schema": MTP_OBJECTIVE_SCHEMA,
        "objective": MTP_OBJECTIVE,
        "reference": "bf16_mtp_head",
        "mtp_layer": int(mtp_layer),
        "rows": [0, int(sequence_length) - 2],
        "global_token_count": mtp_global_token_count(n_sequences, sequence_length),
        "previous_hidden": "target_post_final_norm",
        "position_zero_embedding": "zeroed",
        "logits_head": "target_lm_head",
    }


def with_mtp_objective(probe_identity, objective) -> dict:
    """A copy of a joint probe identity with the MTP objective attached."""
    if "objective" in probe_identity:
        raise ValueError("probe identity already names an objective")
    if not isinstance(objective, dict) or objective.get("schema") != MTP_OBJECTIVE_SCHEMA:
        raise ValueError("MTP objective identity required")
    result = copy.deepcopy(dict(probe_identity))
    result["objective"] = copy.deepcopy(objective)
    return result


__all__ = [
    "Glm5NextMtpLayer",
    "MTP_OBJECTIVE",
    "MTP_OBJECTIVE_SCHEMA",
    "MtpCheckpointModel",
    "load_mtp_layer",
    "mtp_layer_initialization_contract",
    "read_checkpoint_tensor",
    "mtp_checkpoint_prefix",
    "mtp_global_token_count",
    "mtp_priced_units",
    "mtp_layer_index",
    "mtp_logits",
    "mtp_objective_identity",
    "mtp_probe_scalar",
    "mtp_rows",
    "mtp_text_config",
    "require_dense_indexer_regime",
    "with_mtp_objective",
]
