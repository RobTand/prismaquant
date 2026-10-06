"""One source traversal for every GLM routed owner's native input.

This is a first-sequence native panel, not a new calibration cache or a quality
probe. The complete 512x512 draw is bound, but only its original sample zero is
executed, as required by ``routed_boundary_inputs``. Source residency and
first-use authentication remain with StreamingContext and its capture owner.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from .glm_routing_replay import (
    glm_route_record,
    router_normalization_epsilon,
    select_original_routes,
)
from .digests import bytes_sha256hex


def visit_routed_boundaries(runner, tokens, consume, *, first_layer=3):
    """Visit original expert arguments immediately, with one hidden batch held.

    No tensor list, second source loader, or read-ahead pool is created here.
    ``visit_layer_batches`` installs prefetched layers and unloads them after
    the consumer. A consumer failure removes the hook and unwinds residency.
    """
    if not runner.require_prefetched_residency:
        raise ValueError("routed capture requires prefetched source residency")
    if any(module.training for module in runner.model.modules()):
        raise ValueError("routed capture requires evaluation mode")
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("routed capture executes exactly the original first sequence")
    observed = []

    def visit(layer, forward):
        if layer < first_layer:
            forward(tokens)
            return
        module = runner.model.get_submodule(
            f"model.language_model.layers.{layer}.mlp.experts")
        calls = 0

        def capture(actual, args, kwargs):
            nonlocal calls
            if actual is not module or calls:
                raise RuntimeError("routed source repeated its expert boundary")
            calls += 1
            consume(layer, actual, args, kwargs)

        handle = module.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            forward(tokens)
            if calls != 1:
                raise RuntimeError("routed source omitted its expert boundary")
            observed.append(layer)
        finally:
            handle.remove()

    runner.visit_layer_batches([tokens], visit)
    if observed != list(range(first_layer, runner.num_layers)):
        raise RuntimeError("routed capture omitted a source layer")
    return observed


def glm_capture_geometry(runner, module, router):
    """Derive source facts from the installed owner, not from the target menu."""
    from .native_moe_panel import validate_geometry

    config = runner.model.config.text_config
    shape = {
        "geometry_version": 1, "geometry_id": "glm53_next_routed_stack_v1",
        "source_id": "glm5_next", "n_routed_experts": module.num_experts,
        "hidden_size": module.hidden_dim, "intermediate_size": module.intermediate_dim,
        "top_k": router.top_k, "shared_experts": config.n_shared_experts,
        "n_group": router.num_group, "topk_group": router.topk_group,
        "scoring_func": config.scoring_func, "topk_method": config.topk_method,
        "norm_topk_prob": router.norm_topk_prob,
        "routed_scaling_factor": router.routed_scaling_factor,
        "swiglu_limit": module.swiglu_limit, "gated": config.hidden_act == "silu",
        "tensor_parallel": 1, "tensor_parallel_cut_axis": "intermediate",
    }
    return validate_geometry(shape)


def capture_streamed_glm_routes(runner, calibration_ids, *, calibration,
                                producer_source, source_acquisition=None, consume,
                                original_authority_input=None, original_plan_input=None,
                                admitted_execution=None, artifact_owner=None):
    """Capture each routed layer through the existing visitor and source owner."""
    from .native_moe_panel import _validate_prefix_capture, original_capture_entry, ORIGINAL_ACQUISITION_SCHEMA
    from .stage_inputs import require_source_identity

    original = any(value is not None for value in
        (original_authority_input, original_plan_input, admitted_execution, artifact_owner))
    authority = owner = source_identity = None
    if original:
        from .calibration_data import _validate_calibration_draw
        from .cost_streaming import StreamedBoundaryArtifacts, build_streamed_model_identity
        from .joint_aura import source_execution_identity
        from .source_generation import _control, original_source_runtime, validate_original_source_runtime
        from .tessera_calibration_cache import require_original_source_authority

        if source_acquisition is not None:
            raise ValueError("original and legacy DEV/cache acquisition are mutually exclusive")
        if any(value is None for value in
               (original_authority_input, original_plan_input, admitted_execution, artifact_owner)):
            raise ValueError("original routed capture requires independent authority/plan/execution and owning session")
        owner = getattr(runner.context, "source_authentication", None)
        authority = require_original_source_authority(owner, original_authority_input,
                                                       original_plan_input, admitted_execution)
        if owner is None:
            raise ValueError("original routed capture requires its existing material owner")
        owner.require_material_device(runner.device)
        if (not isinstance(artifact_owner, StreamedBoundaryArtifacts)
                or artifact_owner.session != authority["session"]
                or artifact_owner._readonly or artifact_owner._status != "running"
                or not artifact_owner._published):
            raise ValueError("original routed capture requires its current owning artifact generation")
        source_identity = build_streamed_model_identity(runner, str(owner.root))
        if source_identity != authority["source_model_identity"]:
            raise ValueError("original routed capture differs from its independently expected complete source")
        _, expected_producer = _control(authority["producer"], "original routed producer")
        if require_source_identity(producer_source) != require_source_identity(expected_producer):
            raise ValueError("original routed capture differs from its bound producer")
        _, actual_calibration = _validate_calibration_draw(calibration_ids.cpu(), calibration["provenance"],
            artifact_sha256=calibration["artifact_sha256"], n_samples=512, seqlen=512)
        if actual_calibration != calibration or calibration != authority["calibration"]:
            raise ValueError("original routed capture differs from the actual full calibration draw")
        if source_execution_identity(runner.model) != authority["source_execution"]:
            raise ValueError("original routed capture source execution differs")
        validate_original_source_runtime(original_source_runtime(runner, owner), authority["runtime"])

    if (runner.profile.name != "glm5_next" or runner.num_layers != 45
            or runner.dtype != torch.bfloat16
            or calibration_ids.shape != (512, 512)
            or calibration_ids.dtype != torch.int64
            or calibration.get("shape") != [512, 512]
            or calibration.get("dtype") != "torch.int64"):
        raise ValueError("routed capture requires the GLM body and exact 512x512 draw")
    if getattr(runner.model.config, "_attn_implementation", None) != "eager":
        raise ValueError("routed capture requires actual eager source attention")
    source = require_source_identity(producer_source)
    runner.context.begin_source_initialization_audit()

    def consume_arguments(layer, module, args, kwargs):
        parent = runner.model.get_submodule(
            f"model.language_model.layers.{layer}.mlp")
        router = parent.gate
        if (type(module).__name__ != "Glm5NextTextExperts"
                or type(router).__name__ != "Glm5NextTextTopkRouter"):
            raise ValueError("routed capture requires the original GLM expert/router pair")
        if original:
            bias = router.e_score_correction_bias
            bias_device = str(bias.device)
            captured = select_original_routes(module, args, kwargs, sequence_length=512, expert_bias=bias)
            tensor_devices = {name: captured["observed_device"] for name in
                              ("inputs", "top_k_index", "top_k_weights")}
            tensor_devices.update(expert_bias=bias_device, coordinates=str(captured["coordinates"].device))
        else:
            captured = select_original_routes(module, args, kwargs, sequence_length=512)
        contract = (runner.context.source_initialization_contract() if layer == 44 else
                    runner.context.source_prefix_initialization_contract(layer))
        result = glm_route_record(runner, module, router, captured, layer=layer,
            calibration=calibration, producer_source=source,
            epsilon=router_normalization_epsilon(router), model_load_contract=contract,
            replay_source="fresh_streamed_bf16_source_pass",
            shape=glm_capture_geometry(runner, module, router))
        metadata = result["metadata"]
        metadata["capture_source_sha256"] = bytes_sha256hex(Path(__file__).read_bytes())
        if original:
            runtime = original_source_runtime(runner, owner)
            validate_original_source_runtime(runtime, authority["runtime"])
            metadata["source_acquisition"] = {
                "schema": ORIGINAL_ACQUISITION_SCHEMA,
                "authority": copy.deepcopy(original_authority_input),
                "session": copy.deepcopy(artifact_owner.session),
                "source_material": owner.receipt(),
                "source_initialization": copy.deepcopy(contract),
                "source_execution": copy.deepcopy(metadata["source_execution"]),
                "runtime": {
                    "schema": "prismaquant.original_routed_capture_runtime.v1", "source_runtime": runtime,
                    "device": metadata["routing"]["device"],
                    "source_tensor_dtypes": {name: record["dtype"] for name, record in metadata["tensors"].items()},
                    "source_tensor_devices": tensor_devices,
                    "expert_class": f"{type(module).__module__}.{type(module).__qualname__}",
                    "router_class": metadata["routing"]["source_protocol"]["router_class"],
                    "router_source_sha256": metadata["routing"]["source_protocol"]["router_source_sha256"],
                },
                "entry": original_capture_entry(metadata),
            }
            _validate_prefix_capture(metadata, source_identity,
                expected_original_authority=original_authority_input,
                expected_original_session=artifact_owner.session, original_source_authority=authority,
                calibration_receipt=calibration)
        else:
            if set(source_acquisition) != {"dev_uncertified", "dev_mode", "source_cache_reuse"}:
                raise ValueError("source authority cannot replace observed capture fields")
            metadata.update(copy.deepcopy(source_acquisition))
            _validate_prefix_capture(metadata)
        consume(layer, result)

    return visit_routed_boundaries(runner, calibration_ids[:1], consume_arguments)
