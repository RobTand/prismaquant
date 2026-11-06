"""Checkpoint loading and priced-unit discovery for the GLM MTP skeleton.

The skeleton stays in ``glm_mtp`` so profiles can build it without loading
pipeline orchestration, and initialization contracts retain its model class.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .glm_mtp import (
    MtpCheckpointModel,
    _checkpoint_index,
    _keeps_fp32,
    _open_shard,
    mtp_checkpoint_prefix,
    mtp_layer_skeleton,
)


def mtp_priced_units(model: MtpCheckpointModel, profile) -> dict:
    """``{qname: (out_features, in_features)}`` for every priced MTP Linear.

    The routed experts come from the profile's packed-expert declaration, the
    same views the exporter splits. The shared expert's three Linears are
    plain ``nn.Linear`` modules. Nothing else in the layer consults the
    serving quantization config described in ``glm_mtp``.
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
    layer = mtp_layer_skeleton(text_config, experts_implementation=experts_implementation)
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
                   "experts_implementation": layer.config._experts_implementation}


def mtp_layer_initialization_contract(layer, receipt, *, input_manifest):
    """The MTP capture's source-initialization contract, from the live layer.

    ``receipt`` is :func:`load_mtp_layer`'s. ``input_manifest`` is the
    ``{schema, path, sha256}`` of the hidden states the layer runs on. Every
    tensor is hashed as it is now, so computing this again after the capture
    shows the forward left the layer as it was loaded.
    """
    import transformers

    from prismaquant.stage_a_chain_seed import tensor_payload_sha256
    from prismaquant.streaming_initialization import (_MTP_LAYER_INITIALIZATION_SCHEMA,
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
