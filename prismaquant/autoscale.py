"""Hardware-adaptive memory sizing for the PrismaQuant pipeline.

Two knobs the probe/cost passes care about:

  1. `layers_per_shard` — how many decoder layers get their Fisher stats
     accumulated in a single reverse sweep. Bigger shards = fewer sweeps
     through the full model = faster probe, but each shard needs more
     gradient + retained-activation memory.

  2. `cache_headroom_gb` — safety margin subtracted from available RAM
     when sizing the streaming layer cache. Lower headroom = bigger
     cache = fewer evictions = fewer `torch.cuda.empty_cache()` stalls
     on UMA hosts, but less slack for autograd spikes.

Both defaults were historically tuned for a 35B-A3B MoE on a 128 GB
Spark. Dense-27B / 122B-A10B / etc. want different values. This module
derives them from the actual checkpoint + host at runtime.

The heuristic is deliberately simple:

    per_layer_bytes(shard) ≈ weight + activations + gradients
    available = free_RAM - safety
    reserved_for_cache = num_layers * per_layer_weight   # hold all layers ⇒ no evictions
    layers_per_shard = (available - reserved_for_cache) / per_layer_bytes

and clamped to [1, num_layers]. Explicit env overrides always win.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path


DEFAULT_SAFETY_GB = 20.0     # slack above the committed estimate. NEVER rely on
                             # swap — kernel OOM-kills BEFORE swap fills on many
                             # Linux configs.
DEFAULT_ACT_MULT = 12        # multiplier in (N*T*hidden*dtype*K) per tracked
                             # layer. Captures backward transient scratch.
# Prefetch-window size for the streaming tier of pick_layers_per_shard:
# 4 concurrent prefetch reads + a completed-ahead margin of 4.
STREAMING_CACHE_WINDOW_LAYERS = 8

DEFAULT_DTYPE_BYTES = 2      # bf16
# Observed on Qwen3.6-27B dense: gradient checkpointing retains activations
# at ~sqrt(n_layers) boundaries, so the full autograd graph adds a
# per-layer-mix overhead independent of how many layers are tracked. Plus
# HF transformers wrappers, tokenizer caches, and Python heap contribute a
# roughly model-independent floor. Empirically ~35 GB at nsamples=32,
# seqlen=1024, hidden=5120. Scale by N*T*hidden so the term tracks
# calibration size.
DEFAULT_FULL_GRAPH_ACT_MULT = 48   # 64 layers × sqrt ≈ 8 × 6 (per-layer-mix overshoot)
DEFAULT_FIXED_OVERHEAD_GB = 15.0   # HF transformers + tokenizer + Python heap floor


# Bounded capture reuses one layer's physical CPU budget. Torch wheels may
# statically link mimalloc, whose delayed purge otherwise retains completed H/X
# even after every tensor owner is gone. Set this in the child environment before
# importing Torch; libc trimming and pinned-host cache release do not reach it.
BOUNDED_CAPTURE_ENV = {
    'PRISMAQUANT_RELEASE_SOURCE_PAGES': '1',
    'MIMALLOC_PURGE_DELAY': '0',
}


def require_bounded_capture_environment(environ):
    """Require the release policy underlying the bounded physical phase plan."""
    for name, expected in BOUNDED_CAPTURE_ENV.items():
        if environ.get(name) != expected:
            raise RuntimeError(f'bounded CUDA capture requires {name}={expected} before process startup')


def streamed_calibration_resources(model_path, *, unit_shapes, counts,
                                   nsamples, seqlen, max_act_rows, cache_slots,
                                   prefetch_workers, headroom_gb,
                                   capture_policy='legacy', capture_load_policy=None):
    """Bound canonical capture using the shared loader's actual source layout.

    Headers and profile mappings determine source residency. Capture owns one
    current hidden boundary per original sample, one layer's X/H, and one
    microbatch transition. The shared expert packer writes final allocations
    directly and drops consumed sources. Its physical allocator may retain
    released source blocks until reuse, so that transient is charged separately
    from the final prefetch window. The
    declared headroom is additional forward/allocator/runtime workspace.
    """
    import math
    from .artifact_completeness import read_artifact_header
    from .model_profiles import detect_profile
    if capture_policy not in ('legacy', 'shared-inputs-release-v1', 'shared-inputs-bounded-v1'):
        raise ValueError('unknown streamed capture resource policy')
    if (any(type(v) is not int or v < 1 for v in
            (nsamples, seqlen, max_act_rows, cache_slots, prefetch_workers)) or
            cache_slots < 2 or not math.isfinite(headroom_gb) or headroom_gb < 0):
        raise ValueError('invalid streamed calibration resource dimensions')
    from .perturbed_x_cache import normalize_verified_activation_load
    capture_load_policy = normalize_verified_activation_load(capture_load_policy)
    if capture_load_policy is not None and capture_policy != 'shared-inputs-bounded-v1':
        raise ValueError('verified capture load admission requires bounded capture phases')
    profile = detect_profile(str(model_path))
    cfg = json.loads((Path(model_path)/'config.json').read_text())
    text = cfg.get('text_config') or cfg
    layers = _num_layers(cfg)
    hidden = _hidden_size(cfg)
    if layers < 1 or hidden < 1:
        raise ValueError('streamed calibration needs explicit decoder geometry')
    header = read_artifact_header(model_path)
    # Price source tensors with the declared HF precision policy. GLM's
    # strict FP32 convolution is stored as three BF16 tensors, so on-disk
    # bytes alone undercount even the final resident layer.
    import torch
    from transformers import AutoConfig
    from transformers.core_model_loading import build_glob_alternation
    from .streaming_model import _resolve_declared_model_cls
    declared_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    declared_class = _resolve_declared_model_cls(declared_config, None)
    get_dtype_plan = getattr(declared_class, '_get_dtype_plan', None)
    dtype_plan = get_dtype_plan(declared_class, torch.bfloat16) if callable(get_dtype_plan) else {}
    dtype_pattern, dtype_groups, _ = build_glob_alternation(list(dtype_plan)) if dtype_plan else (None, {}, None)
    fp4_experts = declared_fp4_expert_dtype(str(model_path))
    multimodal = profile.requires_multimodal_skeleton()
    body_prefix = profile.body_layer_prefix()+'.'
    live_probe = profile.checkpoint_to_live_name(body_prefix+'0.weight', multimodal=multimodal)
    if live_probe is None or '.0.' not in live_probe:
        raise ValueError('profile cannot map the decoder prefix for resource admission')
    live_prefix = live_probe.rsplit('.0.', 1)[0]+'.'
    body, fixed, pack, concat = {}, 0, {}, {}
    raw_body, max_element_bytes = {}, 4
    packed_regex = profile.per_expert_moe_regex()
    packed_pattern = (re.compile(packed_regex.removeprefix('re:')) if packed_regex else None)
    def resident_element_bytes(name):
        match = None if dtype_pattern is None else dtype_pattern.search(name)
        return (2 if match is None else torch.empty((), dtype=
            dtype_plan[dtype_groups[match.lastgroup]]).element_size())
    for key, meta in header.items():
        name = profile.checkpoint_to_live_name(key, multimodal=multimodal)
        if name is None:
            continue
        shape = meta['shape']
        numel = math.prod(shape)
        begin, end = meta['data_offsets']
        stored = int(end)-int(begin)
        floating = _safetensors_source_float_bytes(str(meta['dtype']).upper())
        target_name = name
        for target, sources, _axis in profile.concat_merge_groups():
            for suffix in sources:
                if name.endswith(suffix):
                    target_name = name[:-len(suffix)]+target
        target_bytes = resident_element_bytes(target_name)
        size = max(stored, numel*target_bytes) if floating is not None else stored
        if (fp4_experts and str(meta['dtype']).upper() in _PACKED_BYTE_DTYPES
                and declared_expert_dtype_covers(key)):
            size = stored*4
        if not name.startswith(live_prefix):
            fixed += size
            continue
        index = name[len(live_prefix):].split('.', 1)[0]
        if not index.isdigit() or not 0 <= int(index) < layers:
            raise ValueError(f'out-of-body source tensor is still live: {name}')
        layer = int(index)
        body[layer] = body.get(layer, 0)+size
        raw_body[layer] = raw_body.get(layer, 0)+stored
        max_element_bytes = max(max_element_bytes, math.ceil(size/max(numel, 1)), floating or 0)
        leaf = name.removesuffix('.weight')
        if packed_pattern is not None and (packed_pattern.match(leaf) or
                packed_pattern.match(profile.to_vllm_internal_name(leaf))):
            owner, projection = leaf.rsplit('.', 1)
            expert_path, expert = owner.rsplit('.', 1)
            parent = profile.packed_expert_parent_for_projection(projection)
            if parent is not None and expert.isdigit():
                group = (layer, expert_path, parent)
                pack[group] = pack.get(group, 0)+size
        for target, sources, _axis in profile.concat_merge_groups():
            if any(name.endswith(suffix) for suffix in sources):
                group = (layer, target)
                concat[group] = concat.get(group, 0)+size
    if set(body) != set(range(layers)):
        raise ValueError('source headers do not cover every decoder layer')
    # A final group is preallocated and filled directly; no per-expert fused
    # slabs survive. Charge all original packed-source bytes as a conservative
    # physical allocator-cache allowance even after their references drop.
    pack_peak = [sum(size for key, size in pack.items() if key[0] == layer)
                 for layer in range(layers)]
    loader_transient = min(prefetch_workers, cache_slots) * (
        max(pack_peak, default=0)+max(concat.values(), default=0))
    h_by_layer, x_by_layer, unit_source_weight_bytes = {}, {}, {}
    total_h, total_x, widest_unit = 0, 0, 0
    for name, shape in unit_shapes.items():
        if not name.startswith(live_prefix):
            raise ValueError(f'capture unit is outside the decoder source scope: {name}')
        layer = int(name[len(live_prefix):].split('.', 1)[0])
        columns = int(shape[1])
        parameter_name = name+'.weight'
        if packed_pattern is not None and (packed_pattern.match(name) or
                packed_pattern.match(profile.to_vllm_internal_name(name))):
            owner, projection = name.rsplit('.', 1)
            expert_path, expert = owner.rsplit('.', 1)
            parent = profile.packed_expert_parent_for_projection(projection)
            if parent is not None and expert.isdigit():
                parameter_name = expert_path+'.'+parent
        unit_source_weight_bytes[name] = math.prod(shape)*resident_element_bytes(parameter_name)
        h = columns*columns*4
        x = min(int(counts[name]), max_act_rows)*columns*4
        h_by_layer[layer] = h_by_layer.get(layer, 0)+h
        x_by_layer[layer] = x_by_layer.get(layer, 0)+x
        total_h += h
        total_x += x
        widest_unit = max(widest_unit, h+x)
    hc_mult = max(1, int(text.get('hc_mult', 1)))
    boundary = nsamples*seqlen*hidden*hc_mult*2
    transition = seqlen*hidden*hc_mult*2
    # Derived metadata is ephemeral for one original B1 batch. Conservatively
    # allow a dense FP32 mask and full-hidden-width rotary pairs; original IDs
    # remain in both the original CPU draw and the device-side sample states.
    masks_positions = seqlen*seqlen*4 + seqlen*hidden*4 + nsamples*seqlen*16
    terms = dict(source_window_bytes=sum(sorted(body.values(), reverse=True)[:cache_slots]),
        nonbody_source_bytes=fixed, loader_transient_bytes=loader_transient,
        current_boundary_bytes=boundary, microbatch_transition_bytes=transition,
        masks_positions_bytes=masks_positions,
        layer_hessian_bytes=max(h_by_layer.values(), default=0),
        layer_prefix_bytes=max(x_by_layer.values(), default=0),
        entry_validation_bytes=widest_unit,
        declared_headroom_bytes=math.ceil(headroom_gb*1024**3))
    # One new entry may coexist with the old one during atomic replacement;
    # metadata/journals have an explicit per-unit serialization allowance.
    disk = total_h+total_x+widest_unit+len(unit_shapes)*16384
    result = dict(schema='prismaquant.streamed_calibration_resources.v1',
        source_header_sha256=hashlib.sha256(json.dumps(header, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest(),
        terms=terms, memory_bytes=sum(terms.values()), disk_bytes=disk,
        full_hessian_bytes=total_h, full_prefix_bytes=total_x,
        unit_source_weight_bytes=unit_source_weight_bytes,
        body_layer_bytes={str(k): v for k, v in sorted(body.items())},
        body_loader_transient_bytes={str(k): pack_peak[k]+max(
            (size for key, size in concat.items() if key[0] == k), default=0)
            for k in range(layers)},
        body_source_file_bytes={str(k): v for k, v in sorted(raw_body.items())},
        live_layer_prefix=live_prefix,
        transient_status='conservative physical allocator bound for direct final-slab packer')
    if capture_policy != 'shared-inputs-bounded-v1':
        return result

    from .routed_experts import declared_shared_capture_groups
    groups = declared_shared_capture_groups(unit_shapes, profile)
    forward_h, forward_x, drain = {}, {}, {}
    for members in groups.values():
        if any(type(counts[name]) is not int or counts[name] <= 0 for name in members):
            raise ValueError('shared capture admission requires positive census counts')
        if len({counts[name] for name in members}) != 1:
            raise ValueError('shared capture siblings disagree on census input count')
        name = members[0]
        layer = int(name[len(live_prefix):].split('.', 1)[0])
        columns = unit_shapes[name][1]
        h = columns*columns*4
        # The device buffer reserves max_act_rows even for a shorter draw.
        device_x = max_act_rows*columns*4
        output_x = min(counts[name], max_act_rows)*columns*4
        forward_h[layer] = forward_h.get(layer, 0)+h
        forward_x[layer] = forward_x.get(layer, 0)+device_x
        # Each group drains before its independent CPU siblings are cloned.
        # At every group boundary charge its larger device/output footprint;
        # one transfer may additionally coexist within the active group.
        drain[layer] = drain.get(layer, 0)+max(h+device_x, len(members)*(h+output_x))
    common = {key: value for key, value in terms.items() if key not in
              ('source_window_bytes', 'loader_transient_bytes',
               'layer_hessian_bytes', 'layer_prefix_bytes')}
    forward = dict(common, source_window_bytes=terms['source_window_bytes'],
        loader_transient_bytes=loader_transient,
        layer_hessian_bytes=max(forward_h.values(), default=0),
        layer_prefix_bytes=max(forward_x.values(), default=0))
    materialization = dict(common,
        source_window_bytes=sum(sorted(body.values(), reverse=True)[:cache_slots-1]),
        loader_transient_bytes=0,
        layer_materialization_bytes=max(drain.values(), default=0),
        finite_validation_mask_bytes=max((max(shape[1]**2,
            min(counts[name], max_act_rows)*shape[1])
            for name, shape in unit_shapes.items()), default=0))
    source_validation = dict(common, source_window_bytes=terms['source_window_bytes'],
        loader_transient_bytes=loader_transient,
        source_validation_file_bytes=max(raw_body.values(), default=0),
        source_validation_cpu_copy_bytes=max(
            (math.prod(shape)*max_element_bytes for shape in unit_shapes.values()), default=0))
    phases = dict(source_validation=source_validation, forward=forward, materialization=materialization)
    if capture_load_policy is not None:
        # Replay/validation loads occur during materialization and final seal,
        # never in the model forward. The existing entry_validation_bytes is S.
        materialization.update(
            capture_serialized_buffer_bytes=capture_load_policy['max_buffer_bytes'],
            capture_source_page_cache_bytes=capture_load_policy['max_buffer_bytes'],
            capture_load_scratch_bytes=capture_load_policy['max_scratch_bytes'])
        phases['seal'] = dict(common,
            source_window_bytes=terms['source_window_bytes'],
            loader_transient_bytes=loader_transient,
            capture_serialized_buffer_bytes=capture_load_policy['max_buffer_bytes'],
            capture_source_page_cache_bytes=capture_load_policy['max_buffer_bytes'],
            capture_load_scratch_bytes=capture_load_policy['max_scratch_bytes'],
            finite_validation_mask_bytes=materialization['finite_validation_mask_bytes'])
        result['capture_load_policy'] = capture_load_policy
    result.update(schema='prismaquant.streamed_calibration_resources.v2',
        capture_policy=capture_policy, input_groups=groups, phases=phases,
        memory_bytes=max(sum(phase.values()) for phase in phases.values()),
        transient_status='checked shared input groups; settled prefetch window and completed source release before materialization')
    # v1's additive terms are retained only in its own schema. v2 carries
    # mutually exclusive phase maps, with the maximum defining admission.
    del result['terms']
    return result


def selected_anchor_resources(model_path, *, unit_shapes, counts, max_act_rows,
                              cache_slots, prefetch_workers, headroom_gb,
                              anchor_batch_size=1, capture_load_policy=None):
    """Bound selected-source preparation separately from resident encoding.

    This extends the source loader's header/dtype accounting. No source
    forward or calibration accumulation occurs. The existing plane-keyed
    encoder memo retains at most one compatible batch's factors.
    """
    import math
    from .perturbed_x_cache import normalize_verified_activation_load
    capture_load_policy = normalize_verified_activation_load(capture_load_policy)
    if not unit_shapes or type(anchor_batch_size) is not int or anchor_batch_size < 1:
        raise ValueError('selected anchors require nonempty units and a positive batch size')
    source = streamed_calibration_resources(model_path, unit_shapes=unit_shapes,
        counts=counts, nsamples=1, seqlen=1, max_act_rows=max_act_rows,
        cache_slots=cache_slots, prefetch_workers=prefetch_workers,
        headroom_gb=headroom_gb)
    prefix = source['live_layer_prefix']
    layers = sorted({str(int(name[len(prefix):].split('.', 1)[0])) for name in unit_shapes}, key=int)
    weights = sum(source['unit_source_weight_bytes'].values())
    widest_weight = max(math.prod(shape)*4 for shape in unit_shapes.values())
    widest_h = max(shape[1]**2*4 for shape in unit_shapes.values())
    widest_x = max(min(counts[name], max_act_rows)*shape[1]*4
                   for name, shape in unit_shapes.items())
    terms = source['terms']
    common = dict(selected_source_weight_bytes=weights,
                  declared_headroom_bytes=terms['declared_headroom_bytes'])
    preparation = dict(common, nonbody_source_bytes=terms['nonbody_source_bytes'],
        source_window_bytes=sum(sorted((source['body_layer_bytes'][k] for k in layers),
                                       reverse=True)[:cache_slots]),
        loader_transient_bytes=sum(sorted((source['body_loader_transient_bytes'][k] for k in layers),
                                          reverse=True)[:min(prefetch_workers, cache_slots)]))
    encoding = dict(common, selected_hessian_bytes=source['full_hessian_bytes'],
        selected_prefix_bytes=source['full_prefix_bytes'],
        encoder_memo_bytes=anchor_batch_size*max(
            shape[1]**2*4+shape[1]*8 for shape in unit_shapes.values()),
        factorization_scratch_bytes=4*widest_h,
        compatible_batch_weight_bytes=anchor_batch_size*widest_weight*4,
        entry_validation_bytes=2*(widest_h+widest_x),
        source_validation_bytes=sum(source['body_source_file_bytes'][k] for k in layers)+widest_weight)
    export_inputs = dict(common, selected_hessian_bytes=source['full_hessian_bytes'],
        selected_prefix_bytes=source['full_prefix_bytes'],
        # The unchanged Torch serializer pauses after each tensor record;
        # verified prefix advice bounds visible file pages to one record.
        # Counts retain the whole census, not this subset. The metadata bound
        # also covers the writer's small buffered tail and central directory.
        export_input_page_window_bytes=widest_h+len(counts)*16384,
        serialization_scratch_bytes=2*widest_h)
    phases = dict(source_preparation=preparation, export_inputs=export_inputs,
                  resident_anchors=encoding)
    if capture_load_policy is not None:
        # The existing loader retains one private serialized buffer and may
        # leave its entire source file in the kernel despite page advice.
        # Decode can coexist with already resident selected X/H, but source
        # staging and encoder factors belong to different completed phases.
        phases['capture_prefetch'] = dict(common,
            selected_hessian_bytes=source['full_hessian_bytes'],
            selected_prefix_bytes=source['full_prefix_bytes'],
            capture_decode_storage_bytes=widest_h+widest_x,
            capture_serialized_buffer_bytes=capture_load_policy['max_buffer_bytes'],
            capture_source_page_cache_bytes=capture_load_policy['max_buffer_bytes'],
            capture_load_scratch_bytes=capture_load_policy['max_scratch_bytes'])
    return dict(schema='prismaquant.selected_anchor_resources.v2', phases=phases,
        **({'capture_load_policy': capture_load_policy} if capture_load_policy is not None else {}),
        memory_bytes=max(sum(phase.values()) for phase in phases.values()),
        selected_source_weight_bytes=weights, selected_layers=layers,
        source_header_sha256=source['source_header_sha256'],
        export_input_writer_policy='verified-tensor-record-prefix',
        encoder_memo_policy='compatible-anchor-batch-width',
        encoder_memo_capacity=anchor_batch_size,
        source_forward_count=0)


def _num_layers(cfg: dict) -> int:
    tc = cfg.get("text_config") or cfg
    return int(tc.get("num_hidden_layers")
               or tc.get("n_layer")
               or cfg.get("num_hidden_layers", 0))


def _hidden_size(cfg: dict) -> int:
    tc = cfg.get("text_config") or cfg
    return int(tc.get("hidden_size")
               or tc.get("n_embd")
               or cfg.get("hidden_size", 0))


def _act_width(cfg: dict) -> int:
    """Widest per-Linear activation the probe/cost retains.

    The retained-activation estimate must track the *widest* projection
    activation a layer holds, not just ``hidden_size``. A transformer MLP's
    ``down_proj`` reads an ``intermediate_size``-wide input, and the cost
    step's batched render materializes ``intermediate_size``-wide outputs
    (gate/up) in fp32 scratch. On models where ``intermediate_size`` ≫
    ``hidden_size`` (Gemma4-31B: 21504 vs 5376, 4×) sizing on ``hidden_size``
    undershoots host RAM ~4× and the watchdog aborts the shard.

    Returns ``max(hidden, ffn, moe_ffn)`` so the estimate is governed by the
    true widest activation. Collapses to ``hidden_size`` when no FFN width is
    declared (== hidden for plain models)."""
    tc = cfg.get("text_config") or cfg
    hidden = _hidden_size(cfg)
    widths = [hidden]
    for key in ("intermediate_size", "moe_intermediate_size",
                "ffn_dim", "n_inner", "shared_expert_intermediate_size"):
        v = tc.get(key) or cfg.get(key)
        if v:
            try:
                widths.append(int(v))
            except (TypeError, ValueError):
                pass
    return max(w for w in widths if w > 0) if any(w > 0 for w in widths) else hidden


# Per-expert routed-expert tensor qnames (`...experts.<id>....`). Matches
# both live (`model.layers.N.mlp.experts.7.gate_proj.weight`) and DSv4
# checkpoint (`layers.N.ffn.experts.7.w1.weight`) naming.
_EXPERT_TENSOR_RE = re.compile(r"\.experts\.\d+\.")


def declared_expert_dtype_covers(name: str) -> bool:
    """Whether the checkpoint's `expert_dtype` declaration covers `name`.

    **ROUTED experts only — verified against the real checkpoint.** The
    declaration reads like a statement about all of a layer's experts, and
    this predicate used to widen to `mlp.shared_experts.*` on that
    reasoning. The real `deepseek-ai/DeepSeek-V4-Flash` headers say
    otherwise (safetensors metadata, four shards spanning the model):

        layers.N.ffn.experts.{i}.w{1,2,3}.weight   I8        <- nibble-packed
        layers.N.ffn.experts.{i}.w{1,2,3}.scale    F8_E8M0
        layers.N.ffn.shared_experts.w{1,2,3}.weight  F8_E4M3 <- block-FP8
        layers.N.ffn.shared_experts.w{1,2,3}.scale   F8_E8M0

    i.e. the shared expert is ordinary block-FP8, 2304/2304 routed-expert
    weights are I8 and 9/9 shared-expert weights are F8_E4M3. The authors'
    own converter agrees and is the tie-breaker: `inference/convert.py`
    gates the fp4 path on ``"experts" in name and dtype == torch.int8``, so
    an F8_E4M3 shared expert never enters it.

    Widening to shared experts would therefore send a block-FP8 tensor into
    the MXFP4 decode, where `_check_mxfp4_packed_grid` refuses a non-int8
    weight — a hard DSv4 load failure. Keep this routed-only.

    Nothing here inspects a tensor's shape or dtype: the trigger stays the
    config declaration (`declared_fp4_expert_dtype`) and the packed layout
    stays a hard assertion after the fact
    (`layer_streaming._check_mxfp4_packed_grid`). Non-expert tensors
    (attention projections, the router gate, norms) and the shared expert
    keep the block-FP8 dequant path and its `_check_fp8_scale_grid`
    assertion.
    """
    return bool(_EXPERT_TENSOR_RE.search(name))


# safetensors dtype names for a 1-byte integer plane — what a nibble-packed
# MXFP4 expert weight ships as. Both spellings must be priced the same way
# the decode treats them: `layer_streaming._check_mxfp4_packed_grid` accepts
# int8 *and* uint8 nibble-packs, so sizing only "I8" would leave a U8
# checkpoint undercounted 4x.
_PACKED_BYTE_DTYPES = frozenset({"I8", "U8"})


def declared_fp4_expert_dtype(model_path: str) -> bool:
    """True when the checkpoint config *explicitly* declares packed-FP4
    routed experts (DSv4-Flash: top-level `expert_dtype: "fp4"` alongside a
    block-FP8 `quantization_config`; the MXFP4 scale siblings are E8M0).

    This declaration — never a tensor-shape heuristic — is what gates the
    streaming loader's MXFP4 decode (`layer_streaming` step 3b) and what
    the resident-size estimators key on: a nibble-packed I8 expert byte
    dequants to 2 logical elements of the execution dtype."""
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            return False
        tc = cfg.get("text_config")
        tc = tc if isinstance(tc, dict) else {}
        val = cfg.get("expert_dtype") or tc.get("expert_dtype") or ""
    except Exception:
        # Absent, unreadable, or unexpectedly-shaped config: not declared.
        # Sizing wrongly here silently mis-budgets the streaming cache, so
        # every failure mode resolves to "verbatim" rather than to a guess.
        return False
    return str(val).lower() in {"fp4", "mxfp4", "mx_fp4"}


def _safetensors_source_float_bytes(dtype_name: str) -> int | None:
    """On-disk bytes/element for a safetensors *floating* dtype name;
    None for non-float dtypes (kept verbatim by the streaming loader)."""
    dt = str(dtype_name).upper()
    if dt.startswith("F8"):
        return 1
    if dt in ("F16", "BF16"):
        return 2
    if dt == "F32":
        return 4
    if dt == "F64":
        return 8
    return None


def _shard_resident_bytes(path: Path, dtype_bytes: int,
                          fp4_experts: bool = False) -> int:
    """Resident bytes for one safetensors shard after streaming load.

    `_read_layer_to_device` casts every floating tensor to the execution
    dtype (native-FP8 weights are block-dequanted to bf16), so resident
    bytes per float element = ``dtype_bytes`` regardless of on-disk
    element size — the same rule `streaming_model._estimate_layer_cache_bytes`
    applies per tensor. fp8-native checkpoints (1 byte/elem on disk)
    therefore occupy 2x their disk size in the layer cache; sizing from
    raw file size undercounts them 2x and blows the memory budget.

    ``fp4_experts`` is the checkpoint's explicit packed-FP4 expert
    declaration (`declared_fp4_expert_dtype`): expert I8/U8 tensors (routed
    and shared alike, see `declared_expert_dtype_covers`) are then MXFP4
    nibble-packs that dequant to TWO logical elements of the execution
    dtype per on-disk byte (a 4x undercount at bf16 if sized verbatim).
    Other non-float dtypes stay verbatim.

    Parses the safetensors JSON header directly (stdlib-only; no tensor
    data is read). Raises on malformed files; the caller falls back to
    the raw file size."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        if header_len <= 0 or header_len > 512 * 1024 ** 2:
            raise ValueError(
                f"implausible safetensors header length {header_len} in {path}"
            )
        header = json.loads(f.read(header_len))
    total = 0
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        off = meta["data_offsets"]
        nbytes = int(off[1]) - int(off[0])
        dtype_name = str(meta.get("dtype", "")).upper()
        if (fp4_experts and dtype_name in _PACKED_BYTE_DTYPES
                and declared_expert_dtype_covers(key)):
            total += nbytes * 2 * int(dtype_bytes)
            continue
        src_bytes = _safetensors_source_float_bytes(dtype_name)
        if src_bytes is None:
            total += nbytes
        else:
            total += (nbytes // src_bytes) * int(dtype_bytes)
    return total


def _model_resident_weight_bytes(model_path: str, dtype_bytes: int) -> int:
    """Sum of resident (post-cast/dequant) bytes across all *.safetensors
    blobs — dtype-aware, see `_shard_resident_bytes`. Falls back to the
    raw blob size per shard when a header can't be parsed, and to 0 if
    the dir doesn't exist yet."""
    p = Path(model_path)
    if not p.exists():
        return 0
    fp4_experts = declared_fp4_expert_dtype(model_path)
    total = 0
    for f in p.glob("*.safetensors"):
        try:
            total += _shard_resident_bytes(f, dtype_bytes, fp4_experts)
        except Exception:
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _available_ram_bytes() -> int:
    """Free RAM right now. On UMA (Grace-Blackwell) this is the shared
    LPDDR5X pool that both CPU and GPU draw from — same number matters
    for CUDA and host work."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 64 * 1024 ** 3  # conservative fallback


def estimate_per_layer_bytes(
    model_path: str,
    num_layers: int,
    hidden_size: int,
    nsamples: int,
    seqlen: int,
    dtype_bytes: int = DEFAULT_DTYPE_BYTES,
    act_mult: int = DEFAULT_ACT_MULT,
    act_width: int | None = None,
) -> tuple[int, int]:
    """Return `(per_layer_weight_bytes, per_layer_active_shard_bytes)`.

    - weight bytes: *resident* size / num_layers, minus head/embed
      approximation. Resident is dtype-aware: floating checkpoint tensors
      cast to the execution dtype at load, so fp8-native sources dequant
      to bf16 in the layer cache (2 bytes/elem resident, not the 1
      byte/elem on disk — sizing from disk undercounted those 2x).
    - active_shard bytes: gradients (~weight) + retained activations
      (N·T·act_width·dtype·act_mult)

    `act_width` is the widest per-Linear activation the layer retains —
    `max(hidden, intermediate)` (see `_act_width`). Defaults to `hidden_size`
    for back-compat; pass the FFN-aware width so large-MLP models size right.
    """
    total_resident = _model_resident_weight_bytes(model_path, dtype_bytes)
    if total_resident > 0 and num_layers > 0:
        # subtract a conservative 10% for non-layer weights (embed, lm_head, norms)
        body_bytes = int(total_resident * 0.90)
        per_layer_weight = body_bytes // num_layers
    else:
        per_layer_weight = 1 * 1024 ** 3  # 1 GB fallback

    grad_bytes = per_layer_weight  # same shape, same dtype
    width = act_width if act_width else hidden_size
    act_bytes = nsamples * seqlen * width * dtype_bytes * act_mult
    per_layer_active = grad_bytes + act_bytes
    return per_layer_weight, per_layer_active


def pick_layers_per_shard(
    model_path: str,
    *,
    nsamples: int,
    seqlen: int,
    dtype_bytes: int = DEFAULT_DTYPE_BYTES,
    act_mult: int = DEFAULT_ACT_MULT,
    safety_gb: float = DEFAULT_SAFETY_GB,
    full_graph_act_mult: int = DEFAULT_FULL_GRAPH_ACT_MULT,
    fixed_overhead_gb: float = DEFAULT_FIXED_OVERHEAD_GB,
    available_ram_bytes: int | None = None,
    hold_all_layers_in_cache: bool = True,
    default: int = 2,
) -> tuple[int, dict]:
    """Pick LAYERS_PER_SHARD from host memory + model size.

    Returns `(lps, diagnostics)` so callers can log the derivation.

    `hold_all_layers_in_cache=True` reserves enough RAM for the layer
    cache to fit every decoder layer (zero evictions → no empty_cache
    stalls). Falls back to holding half the layers if that leaves
    too little for shard work.
    """
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return default, {"reason": "no config.json", "lps": default}
    with open(cfg_path) as f:
        cfg = json.load(f)
    n_layers = _num_layers(cfg)
    hidden = _hidden_size(cfg)
    if n_layers <= 0 or hidden <= 0:
        return default, {"reason": "missing layer/hidden in config", "lps": default}

    per_layer_weight, per_layer_active = estimate_per_layer_bytes(
        model_path, n_layers, hidden, nsamples, seqlen,
        dtype_bytes=dtype_bytes, act_mult=act_mult, act_width=_act_width(cfg),
    )
    avail = available_ram_bytes if available_ram_bytes is not None else _available_ram_bytes()
    safety = int(safety_gb * 1024 ** 3)

    if hold_all_layers_in_cache:
        cache_reserve = n_layers * per_layer_weight
    else:
        # Streaming tier. LRU under a cyclic layer sweep yields ZERO
        # reuse whenever the cache cannot hold the full cycle — a
        # half-model reserve buys ~nothing (measured 9-11% hit rate on
        # Laguna-117B) while starving shard width down to lps=1, which
        # multiplies the number of full-model sweeps. Reserve only a
        # prefetch window deep enough to overlap reads with compute
        # (workers in flight + completed-ahead margin) and spend the
        # rest of RAM on layers-per-shard: each extra layer per shard
        # removes an entire model sweep from the phase-3 schedule.
        cache_reserve = (
            min(n_layers, STREAMING_CACHE_WINDOW_LAYERS) * per_layer_weight)

    # Full-graph checkpointed activations: autograd retains activations
    # at ~sqrt(n_layers) boundaries across ALL layers, not just tracked
    # ones — this is fixed memory that any shard incurs. Plus HF /
    # tokenizer / Python overhead floor.
    full_graph_act = nsamples * seqlen * hidden * dtype_bytes * full_graph_act_mult
    overhead = int(fixed_overhead_gb * 1024 ** 3)

    shard_budget = avail - safety - cache_reserve - full_graph_act - overhead
    # If reserving the full cache leaves too little, fall back to half-cache
    if shard_budget < per_layer_active and hold_all_layers_in_cache:
        return pick_layers_per_shard(
            model_path, nsamples=nsamples, seqlen=seqlen,
            dtype_bytes=dtype_bytes, act_mult=act_mult,
            safety_gb=safety_gb,
            full_graph_act_mult=full_graph_act_mult,
            fixed_overhead_gb=fixed_overhead_gb,
            available_ram_bytes=avail,
            hold_all_layers_in_cache=False, default=default,
        )
    shard_budget = max(shard_budget, per_layer_active)  # never below 1 layer

    lps = max(1, int(shard_budget // per_layer_active))
    lps = min(lps, n_layers)

    return lps, {
        "lps": lps,
        "n_layers": n_layers,
        "hidden": hidden,
        "per_layer_weight_gb": per_layer_weight / 1024 ** 3,
        "per_layer_active_gb": per_layer_active / 1024 ** 3,
        "full_graph_act_gb": full_graph_act / 1024 ** 3,
        "fixed_overhead_gb": fixed_overhead_gb,
        "available_gb": avail / 1024 ** 3,
        "safety_gb": safety_gb,
        "cache_reserve_gb": cache_reserve / 1024 ** 3,
        "shard_budget_gb": shard_budget / 1024 ** 3,
        "hold_all_layers": hold_all_layers_in_cache,
    }


def pick_cache_headroom_gb(
    model_path: str,
    *,
    safety_gb: float = DEFAULT_SAFETY_GB,
    layers_per_shard: int = 1,
    nsamples: int = 32,
    seqlen: int = 1024,
    default: float = 75.0,
) -> tuple[float, dict]:
    """Pick `cache_headroom_gb` so the layer cache gets (available - headroom)
    bytes for fitting decoder layers. Returns `(headroom_gb, diagnostics)`.

    The probe's active working set dominates the headroom: safety margin
    + gradients/activations for `layers_per_shard` layers. Anything
    leftover goes to the streaming cache.
    """
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return default, {"reason": "no config.json", "headroom_gb": default}
    with open(cfg_path) as f:
        cfg = json.load(f)
    n_layers = _num_layers(cfg)
    hidden = _hidden_size(cfg)
    if n_layers <= 0 or hidden <= 0:
        return default, {"reason": "missing layer/hidden", "headroom_gb": default}

    _, per_layer_active = estimate_per_layer_bytes(
        model_path, n_layers, hidden, nsamples, seqlen, act_width=_act_width(cfg),
    )
    shard_working_bytes = layers_per_shard * per_layer_active
    headroom_bytes = shard_working_bytes + int(safety_gb * 1024 ** 3)
    headroom_gb = headroom_bytes / 1024 ** 3
    return headroom_gb, {
        "headroom_gb": headroom_gb,
        "shard_working_gb": shard_working_bytes / 1024 ** 3,
        "safety_gb": safety_gb,
        "layers_per_shard": layers_per_shard,
    }


def autoscale(
    model_path: str,
    *,
    nsamples: int,
    seqlen: int,
    layers_per_shard_env: str | int | None = None,
    cache_headroom_gb_env: str | float | None = None,
    safety_gb: float = DEFAULT_SAFETY_GB,
) -> tuple[int, float, dict]:
    """Compute `(layers_per_shard, cache_headroom_gb)` from model + host.

    Explicit env overrides win:
      - `layers_per_shard_env` (int or stringified int) skips LPS autoscale
      - `cache_headroom_gb_env` (float or stringified float) skips headroom autoscale

    Use `"auto"` or `None` to request autoscale.
    """
    diag: dict = {}

    # Parse LPS override
    lps: int
    if layers_per_shard_env in (None, "", "auto", "AUTO"):
        lps, lps_diag = pick_layers_per_shard(
            model_path, nsamples=nsamples, seqlen=seqlen, safety_gb=safety_gb,
        )
        diag["lps_autoscaled"] = lps_diag
    else:
        lps = int(layers_per_shard_env)
        diag["lps_source"] = f"explicit={lps}"

    # Parse headroom override
    headroom: float
    if cache_headroom_gb_env in (None, "", "auto", "AUTO"):
        headroom, hr_diag = pick_cache_headroom_gb(
            model_path, safety_gb=safety_gb,
            layers_per_shard=lps, nsamples=nsamples, seqlen=seqlen,
        )
        diag["headroom_autoscaled"] = hr_diag
    else:
        headroom = float(cache_headroom_gb_env)
        diag["headroom_source"] = f"explicit={headroom}"

    return lps, headroom, diag


if __name__ == "__main__":
    # CLI usage: python -m prismaquant.autoscale <model_path> [--nsamples N --seqlen T]
    import argparse
    ap = argparse.ArgumentParser(description="Print autoscaled memory knobs.")
    ap.add_argument("model_path")
    ap.add_argument("--nsamples", type=int, default=int(os.environ.get("NSAMPLES", 32)))
    ap.add_argument("--seqlen", type=int, default=int(os.environ.get("SEQLEN", 1024)))
    ap.add_argument("--safety-gb", type=float, default=DEFAULT_SAFETY_GB)
    ap.add_argument("--lps", default=os.environ.get("LAYERS_PER_SHARD"))
    ap.add_argument("--headroom", default=os.environ.get("CACHE_HEADROOM_GB"))
    args = ap.parse_args()

    lps, hr, diag = autoscale(
        args.model_path,
        nsamples=args.nsamples, seqlen=args.seqlen,
        layers_per_shard_env=args.lps,
        cache_headroom_gb_env=args.headroom,
        safety_gb=args.safety_gb,
    )
    print(f"LAYERS_PER_SHARD={lps}")
    print(f"CACHE_HEADROOM_GB={hr:.1f}")
    print(json.dumps(diag, indent=2))
