"""Capture original GLM MoE arguments in one authenticated BF16 layer replay.

The caller supplies an owned completed boundary or a fresh BF16 source prefix.
This module records actual router arguments before executing the expert owner.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

import torch


def require_cached_prefix_source_identity(source_model, cache_path, expected_identity):
    """Bounded capture reuses an authenticated cache or refuses before payload IO."""
    from .dev_mode import dev_mode_enabled
    from .cost_streaming import validate_cached_streamed_model_identity
    if not dev_mode_enabled():
        raise RuntimeError("bounded routing prefix requires explicit DEV_MODE cache reuse; automatic source rehash is forbidden")
    identity = validate_cached_streamed_model_identity(
        source_model, cache_path, require_complete_checkpoint=True)
    if identity != expected_identity:
        raise RuntimeError("routing source cache identity differs from original preparation")
    return identity


def select_original_routes(module, args, kwargs, *, sequence_length):
    """Select the complete first sequence without casting any routed tensor."""
    bound = inspect.signature(type(module).forward).bind(module, *args, **kwargs)
    tensors = [bound.arguments[k] for k in ("hidden_states", "top_k_index", "top_k_weights")]
    x, ids, weights = tensors
    if (any(not isinstance(t, torch.Tensor) or t.ndim != 2 for t in tensors)
            or x.shape[0] != sequence_length or ids.shape != weights.shape
            or ids.shape[0] != sequence_length):
        raise ValueError("routing replay must contain exactly the original first sequence")
    if ids.dtype not in (torch.int32, torch.int64) or weights.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("routing replay refuses a cast or unsupported original route dtype")
    coordinates = torch.stack((torch.zeros(sequence_length, dtype=torch.int64),
                               torch.arange(sequence_length, dtype=torch.int64)), dim=1)
    return {"inputs": x.detach().cpu().contiguous(), "top_k_index": ids.detach().cpu().contiguous(),
            "top_k_weights": weights.detach().cpu().contiguous(), "coordinates": coordinates}


def router_normalization_epsilon(router):
    """Read the original router's denominator constant, rather than invent it."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(type(router).forward)))
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "denominator"
                                               for t in node.targets):
            value = node.value
            if (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
                    and isinstance(value.right, ast.Constant)
                    and isinstance(value.right.value, (int, float))):
                values.append(float(value.right.value))
    if len(values) != 1 or values[0] != 1e-20:
        raise ValueError("GLM source router denominator differs from its inspected noaux_tc implementation")
    return values[0]


def capture_replayed_glm_routes(runner, hidden, calibration_ids, *, layer, calibration,
                               producer_source, parent_boundary):
    """Run a source layer only until the real packed-expert call is reached."""
    from . import pretrained_initialization_contract
    from .cost_streaming import StreamedForwardBoundaries
    from .joint_aura import source_execution_identity
    from .production_weight_cache import _cb_cache_tensor_identity
    from .tessera_expert_projection import _require_source_identity

    if runner.profile.name != "glm5_next" or not (3 <= layer < runner.num_layers):
        raise ValueError("routing replay requires a body GLM MoE layer")
    if (calibration_ids.shape != (1, 512) or hidden.shape[0] != 1 or hidden.shape[1] != 512
            or calibration.get("shape") != [512, 512] or hidden.dtype != torch.bfloat16):
        raise ValueError("routing replay requires sample zero of the exact 512x512 calibration")
    attention = getattr(runner.base_model.config, '_attn_implementation', None)
    if attention != 'eager':
        raise ValueError('routing replay requires the actual BF16 source eager attention selector')
    source = _require_source_identity(producer_source)
    unit = f"model.language_model.layers.{layer}.mlp.experts"
    module = runner.model.get_submodule(unit)
    parent = runner.model.get_submodule(unit.rsplit(".", 1)[0])
    router = parent.gate
    if type(module).__name__ != "Glm5NextTextExperts" or type(router).__name__ != "Glm5NextTextTopkRouter":
        raise ValueError("routing replay target is not the actual GLM packed expert/router pair")
    epsilon = router_normalization_epsilon(router)
    pass_state = runner.profile.isolated_layer_pass_state(None, runner.layers[layer])
    if pass_state:
        raise ValueError("routing replay requires a separately bound shared forward state")
    ids, positions, unused, embeddings, mask = runner._prepare(calibration_ids)
    del unused
    batch = StreamedForwardBoundaries(ids, positions, embeddings, mask, [], None)
    captured = {}
    captured_device = None

    class Captured(BaseException):
        pass

    def hook(actual, args, kwargs):
        nonlocal captured_device
        if actual is not module or captured:
            raise RuntimeError("unexpected repeated routed boundary")
        bound = inspect.signature(type(actual).forward).bind(actual, *args, **kwargs)
        devices = {str(bound.arguments[k].device) for k in ('hidden_states', 'top_k_index', 'top_k_weights')}
        if len(devices) != 1 or not next(iter(devices)).startswith('cuda:'):
            raise ValueError('original routed tensors must share their actual CUDA device')
        captured_device = next(iter(devices))
        captured.update(select_original_routes(actual, args, kwargs, sequence_length=512))
        raise Captured()

    handle = module.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            runner.isolated_layer(batch, layer, hidden.to(runner.device), pass_state=pass_state)
    except Captured:
        pass
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("source layer did not reach its packed MoE boundary")
    bias = router.e_score_correction_bias
    if bias.dtype != torch.float32 or list(bias.shape) != [module.num_experts] or not torch.isfinite(bias).all():
        raise ValueError("GLM routing capture requires the original finite FP32 correction bias")
    captured["expert_bias"] = bias.detach().cpu().contiguous()
    tensor_identity = {k: _cb_cache_tensor_identity(v) for k, v in captured.items()}
    routing = dict(activation="silu", scoring_func="sigmoid", renormalize=router.norm_topk_prob,
        routed_scaling_factor=router.routed_scaling_factor, apply_router_weight_on_input=False,
        expert_map=None, input_dtype=str(captured["inputs"].dtype),
        topk_weights_dtype=str(captured["top_k_weights"].dtype),
        topk_ids_dtype=str(captured["top_k_index"].dtype), device=captured_device,
        weights_contract="post_renormalization_and_routed_scaling", topk_method="noaux_tc",
        n_group=router.num_group, topk_group=router.topk_group, swiglu_limit=module.swiglu_limit,
        source_protocol=dict(router_class=f"{type(router).__module__}.{type(router).__qualname__}",
            router_source_sha256=hashlib.sha256(inspect.getsource(type(router)).encode()).hexdigest(),
            scoring_func="sigmoid", topk_method="noaux_tc", normalization_epsilon=epsilon,
            correction_bias={k: tensor_identity["expert_bias"][k] for k in ("content_sha256", "dtype")},
            expert_bias_affects="selection_only", norm_topk_prob=router.norm_topk_prob))
    from .native_moe_panel import GEOMETRY_VERSION, validate_geometry
    shape = {"geometry_version":GEOMETRY_VERSION,"geometry_id":"glm53_next_routed_stack_v1",
        "source_id":"glm5_next","n_routed_experts":module.num_experts,"top_k":router.top_k,
        "hidden_size":module.hidden_dim,"intermediate_size":module.intermediate_dim,
        "shared_experts":runner.base_model.config.n_shared_experts,"n_group":router.num_group,
        "topk_group":router.topk_group,"topk_method":routing["topk_method"],
        "scoring_func":routing["scoring_func"],"norm_topk_prob":router.norm_topk_prob,
        "routed_scaling_factor":router.routed_scaling_factor,"swiglu_limit":module.swiglu_limit,
        "gated":True,"tensor_parallel":1,"tensor_parallel_cut_axis":"intermediate"}
    # The actual module stores a fused gate/up tensor; refusal precedes any geometry claim.
    if tuple(module.gate_up_proj.shape) != (module.num_experts,2*module.intermediate_dim,module.hidden_dim):
        raise ValueError("actual GLM expert gate/up geometry differs")
    validate_geometry(shape)
    return {"tensors": captured, "metadata": {
        "schema": "prismaquant.native_moe_raw_boundary.v1", "unit": unit,
        "shape": shape,
        "routing": routing, "profile_role_order": ["w1", "w3", "w2"],
        "scope": "first calibration sequence; decode uses its first row, not autoregressive generation",
        "calibration_sha256": calibration["calibration_sha256"],
        "calibration_shape": calibration["shape"], "calibration_dtype": calibration["dtype"],
        "producer_source": source, "runtime_config": runner.model.config.to_dict(),
        "source_execution": source_execution_identity(runner.model),
        "model_load_contract": pretrained_initialization_contract(runner.model),
        "attention_implementation": attention,
        "capture_runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                            "transformers": __import__("transformers").__version__},
        "capture_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "replay": {"schema": "prismaquant.glm_routing_boundary_replay.v1", "layer": layer,
                   "parent_boundary": parent_boundary, "source": "fresh_isolated_bf16_layer_replay",
                   "sample": 0, "stop": "before_original_packed_experts_forward"},
        "tensors": tensor_identity}}
