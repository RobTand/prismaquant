"""Temporary, model-local perturbations for G3 and parent band replay.

quantized_rows_fp32 retains unrounded dequantized values for output-energy sums.
Activation hooks return the input dtype: G3 therefore has one additional BF16
rounding versus the native FP32 epilogue. CPU oracles do not attest GPU equality.
No operator, class, router, cache or process-global forward is replaced.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
import threading
import weakref

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

T4 = "e2m1_group16_ue4m3_static"
T8 = "fp8_per_token_dynamic"
BF16 = "bf16_unquantized"
CONTRACT_FAMILY = {T4: "T4", T8: "T8", BF16: "BF16", "source_bf16": "BF16"}
ROLES = frozenset(("gate_proj", "up_proj", "down_proj"))
KINDS = frozenset(("routed", "shared", "dense", "attention", "lm_head"))
_LOCK = threading.Lock()
_LEASES = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class StaticScale:
    """Actual source scalar plus the explicitly resolved executed scalar.

    source names the exact scale tensor/metadata entry, never a guessed maximum.
    group and members record a serving reduction when effective differs from raw.
    """
    raw: float
    effective: float
    source: str
    group: str
    members: tuple[str, ...] = ()

    def __post_init__(self):
        for value in (self.raw, self.effective):
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError("Static activation scale must be finite and positive")
        if not self.source or not self.group:
            raise ValueError("Static activation scale requires its source entry and group")


@dataclass(frozen=True)
class UnitSpec:
    qname: str
    kind: str
    role: str
    expert: int | None
    family: str
    contract: str
    scale: StaticScale | None = None
    module: torch.nn.Module | None = None
    tp_splits: int | None = None

    def __post_init__(self):
        if self.kind not in KINDS or not self.role or not self.qname or (self.kind in {"routed", "shared", "dense"} and self.role not in ROLES):
            raise ValueError("Unknown unit kind, role or empty name")
        if (self.kind == "routed") != (type(self.expert) is int and self.expert >= 0):
            raise ValueError("Only routed units require a nonnegative expert index")
        if self.kind != "routed" and self.expert is not None:
            raise ValueError("Nonrouted units cannot name an expert")
        if CONTRACT_FAMILY.get(self.contract) != self.family:
            raise ValueError("Family does not select its actual activation contract")
        if self.contract == T4 and self.scale is None:
            raise ValueError(self.qname + ": static A4 requires the actual bound scale")
        if self.tp_splits is not None and self.tp_splits not in (1,2):
            raise ValueError("The explicit activation TP contract is invalid")
        if self.contract != T4 and self.scale is not None:
            raise ValueError(self.qname + ": nonstatic contract cannot carry a static scale")

    @property
    def key(self):
        return self.kind, self.expert, self.role


def quantized_rows_fp32(x, contract, *, input_global_scale=None, tp_splits=1):
    """Real served QDQ residual source, [rows,K] -> FP32, no epilogue rounding.

    T8 quantizes contiguous rank-local down halves separately. T4 groups are
    sixteen columns inside each rank's half; static G is explicit per invocation.
    GPU branches call the public pinned native operations; they need GPU proof.
    """
    if x.ndim != 2 or type(tp_splits) is not int or tp_splits < 1 or x.shape[1] % tp_splits:
        raise ValueError("Activation requires two-dimensional integral TP slices")
    if contract not in CONTRACT_FAMILY:
        raise ValueError("Unknown activation contract: " + str(contract))
    width = x.shape[1] // tp_splits
    if contract == T4:
        if width % 16 or input_global_scale is None or isinstance(input_global_scale, bool):
            raise ValueError("Static A4 needs actual G and group-sixteen rank geometry")
        g = float(input_global_scale)
        if not math.isfinite(g) or g <= 0:
            raise ValueError("Static A4 G must be finite and positive")
    if contract in (BF16, "source_bf16") or not x.shape[0]:
        return x.float()
    rows = x.reshape(-1, tp_splits, width).reshape(-1, width)
    if contract == T8:
        if x.device.type == "cuda":
            from tessera.serving.native_ops import native_fp8_quant, require_native_fp8_quant
            require_native_fp8_quant("G3 mixed activation diagnostic")
            codes, scales = native_fp8_quant(rows.to(torch.bfloat16).contiguous())
        else:
            from accepted.g3_lib import fp8_per_token_dynamic
            codes, scales = fp8_per_token_dynamic(rows)
        return (codes.float() * scales.float()).reshape(x.shape)
    if x.device.type == "cuda":
        from tessera.kernel_a4 import a4_quantize_activation
        from prismaquant.mxfp4_widen import e2m1_to_e4m3_table
        global_scale = torch.tensor(g, dtype=torch.float32, device=x.device)
        packed, scales = a4_quantize_activation(rows.to(torch.bfloat16).contiguous(), global_scale)
        # The public widening table addresses one packed byte and emits two
        # E2M1 values in low-nibble-even-column order. Keep the product FP32.
        values = e2m1_to_e4m3_table(x.device).float()[packed.long()].reshape(rows.shape)
        output = values * (scales.float() / global_scale).repeat_interleave(16, dim=-1)
    else:
        from prismaquant.nvfp4_activation_contract import nvfp4_activation_qdq_served
        output = nvfp4_activation_qdq_served(rows.float(), g)
    return output.reshape(x.shape)


def _amplitude_value(amplitude):
    if isinstance(amplitude, bool) or not isinstance(amplitude, (int, float)):
        raise ValueError("Perturbation amplitude must be a positive finite number")
    value = float(amplitude)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Perturbation amplitude must be a positive finite number")
    return value


def activation_residual(x, spec, amplitude, *, tp=2, round_to_input=True):
    """x + amplitude * (Q(x)-x); never Q(amplitude*x)."""
    amplitude = _amplitude_value(amplitude)
    if spec.family == "BF16":
        return x if round_to_input else x.float()
    width = x.shape[-1]
    q = quantized_rows_fp32(x.reshape(-1, width), spec.contract,
        input_global_scale=spec.scale.effective if spec.scale else None,
        tp_splits=spec.tp_splits if spec.tp_splits is not None else tp if spec.role == "down_proj" else 1).reshape(x.shape)
    residual = q if amplitude == 1 else x.float() + amplitude * (q - x.float())
    return residual.to(x.dtype) if round_to_input else residual


def unit_view(layer, spec):
    """Use the accepted G3 mapping of logical roles to executed packed slices."""
    if spec.kind in {"attention", "lm_head"}:
        if not isinstance(spec.module, torch.nn.Linear):
            raise ValueError("A new unit needs its actual executed Linear")
        return spec.module.weight.detach()
    from accepted.g3_lib import unit_view as accepted_unit_view
    return accepted_unit_view(layer, {"qname": spec.qname, "kind": spec.kind,
                                     "role": spec.role, "expert": spec.expert})


def _slice_index(weight, parameter):
    # Same shape/stride/storage/offset recognizer as the existing packed expert
    # activation machinery (sensitivity_probe._packed_expert_slice_index).
    # Public Tensor metadata avoids importing that owner's private tap or using
    # a process-global F.linear replacement.
    if weight.ndim != 2 or parameter.ndim != 3 or weight.device.type == "meta":
        return None
    if weight.shape != parameter.shape[1:] or weight.stride() != parameter.stride()[1:]:
        return None
    if weight.untyped_storage().data_ptr() != parameter.untyped_storage().data_ptr():
        return None
    stride = parameter.stride(0)
    offset = weight.storage_offset() - parameter.storage_offset()
    if stride <= 0 or offset < 0 or offset % stride or offset // stride >= parameter.shape[0]:
        return None
    return offset // stride


def _same_q(a, b):
    return (a is None and b is None) or (a is not None and b is not None and
        a.contract == b.contract and
        (a.scale.effective if a.scale else None) == (b.scale.effective if b.scale else None))


class _PackedActivationMode(TorchFunctionMode):
    """Thread-local interception inside the existing packed expert forward.

    The original expert loop, router, _apply_gate and output reduction run as-is.
    Linear and grouped_mm use the existing exact packed slice/transpose contract.
    A partial gate/up class perturbation computes its two output halves separately;
    a served fused pair with equal contracts executes one original fused GEMM.
    """
    def __init__(self, module, specs, amplitude, tp):
        self.module, self.specs, self.amplitude, self.tp = module, specs, amplitude, tp
        self.parameters = {"gate_up_proj": module.gate_up_proj, "down_proj": module.down_proj}
        self.calls = 0

    def q(self, inputs, expert, role):
        spec = self.specs.get(("routed", expert, role))
        return inputs if spec is None else activation_residual(inputs, spec, self.amplitude, tp=self.tp)

    def _linear(self, func, inputs, weight, bias):
        for name, parameter in self.parameters.items():
            expert = _slice_index(weight, parameter)
            if expert is None:
                if weight.untyped_storage().data_ptr() == parameter.untyped_storage().data_ptr():
                    raise ValueError("Packed activation weight escaped its declared expert slice")
                continue
            self.calls += 1
            if name == "down_proj":
                return func(self.q(inputs, expert, "down_proj"), weight, bias)
            gate = self.specs.get(("routed", expert, "gate_proj"))
            up = self.specs.get(("routed", expert, "up_proj"))
            if _same_q(gate, up):
                return func(self.q(inputs, expert, "gate_proj"), weight, bias)
            middle = weight.shape[0] // 2
            return torch.cat((
                func(self.q(inputs, expert, "gate_proj"), weight[:middle], None if bias is None else bias[:middle]),
                func(self.q(inputs, expert, "up_proj"), weight[middle:], None if bias is None else bias[middle:])), dim=-1)
        return func(inputs, weight, bias)

    def _grouped(self, func, inputs, weight, kwargs):
        for name, parameter in self.parameters.items():
            if weight.untyped_storage().data_ptr() != parameter.untyped_storage().data_ptr():
                continue
            expected = parameter.transpose(-2, -1)
            ends = kwargs.get("offs")
            if (weight.shape != expected.shape or weight.stride() != expected.stride()
                or weight.storage_offset() != expected.storage_offset() or inputs.ndim != 2
                or inputs.shape[1] != parameter.shape[2] or not isinstance(ends, torch.Tensor)
                or ends.ndim != 1 or len(ends) != parameter.shape[0]
                or ends.dtype not in (torch.int32, torch.int64)):
                raise ValueError("Packed grouped activation geometry differs")
            offsets = ends.tolist()
            if any(b < a for a, b in zip([0, *offsets[:-1]], offsets)) or offsets[-1] != len(inputs):
                raise ValueError("Grouped offsets do not cover the routed input rows")
            def grouped_input(role):
                result = inputs.clone()
                start = 0
                for expert, end in enumerate(offsets):
                    if end > start:
                        result[start:end] = self.q(inputs[start:end], expert, role)
                    start = end
                return result
            self.calls += 1
            if name == "down_proj":
                return func(grouped_input("down_proj"), weight, **kwargs)
            equal = all(_same_q(self.specs.get(("routed", e, "gate_proj")),
                               self.specs.get(("routed", e, "up_proj"))) for e in range(parameter.shape[0]))
            if equal:
                return func(grouped_input("gate_proj"), weight, **kwargs)
            middle = weight.shape[-1] // 2
            return torch.cat((func(grouped_input("gate_proj"), weight[..., :middle], **kwargs),
                              func(grouped_input("up_proj"), weight[..., middle:], **kwargs)), dim=-1)
        return func(inputs, weight, **kwargs)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.linear:
            inputs = args[0] if args else kwargs["input"]
            weight = args[1] if len(args) > 1 else kwargs["weight"]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            return self._linear(func, inputs, weight, bias)
        if func is getattr(F, "grouped_mm", None):
            return self._grouped(func, args[0], args[1], kwargs)
        return func(*args, **kwargs)


EAGER_EXPERTS = (None, "eager")


@contextmanager
def interceptable_packed_experts(model):
    """Pin the eager per-expert MoE backend for one whole replay, then restore it.

    transformers 5.x can dispatch packed experts through batched_mm or grouped_mm
    kernels with no per-expert F.linear boundary, so the packed activation taps
    see no call. The existing owner (prismaquant sensitivity_probe) pins "eager"
    for the same reason: the same math through the interceptable kernel. Pin it
    for every stream of a replay, so null and perturbed streams share one backend.
    Yields the sorted previous backend names.
    """
    configs = {}
    root = getattr(model, "config", None)
    candidates = [root] if root is not None else []
    if root is not None and hasattr(root, "get_text_config"):
        candidates.append(root.get_text_config())
    candidates.extend(getattr(module, "config", None) for module in model.modules())
    for config in candidates:
        if config is not None and hasattr(config, "_experts_implementation"):
            configs[id(config)] = config
    previous = {key: config._experts_implementation for key, config in configs.items()}
    try:
        for config in configs.values():
            config._experts_implementation = "eager"
        yield sorted({str(value) for value in previous.values()})
    finally:
        for key, config in configs.items():
            config._experts_implementation = previous[key]


@contextmanager
def inject_layer(layer_module, unit_specs, decoded_weights=None, *, kinds=KINDS,
                 roles=None, families=frozenset(("T4", "T8", "BF16")), amplitude=1,
                 weight=False, activation=True, tp=2, allow_empty=False):
    """Shared parent interface; all changes restored on success or exception.

    unit_specs: UnitSpec list for this layer, with actual role/expert scales.
    decoded_weights: qname -> actual FP32 or BF16 decoded [out,in] tensor.
    Select any family/kind/role class. W-only uses activation=False (A16),
    A-only uses weight=False on the caller's clean BF16 source weights.
    Joint uses both. Positive finite amplitudes multiply the same residual,
    including approved two/four fallback; the quantizer input never changes.
    SOURCE specifications always remain untouched. allow_empty=True explicitly
    permits null replay, reporting unavailable weight gain rather than zero price.
    This function does not load, cache, select or score. The caller must not
    concurrently execute the same mutated model instance.
    """
    amplitude = _amplitude_value(amplitude)
    if type(tp) is not int or tp != 2:
        raise ValueError("Only the accepted TP2 geometry is supported")
    if type(allow_empty) is not bool:
        raise ValueError("Explicit null replay selection must be Boolean")
    kinds, families = frozenset(kinds), frozenset(families)
    specs = list(unit_specs)
    roles = frozenset(s.role for s in specs) if roles is None else frozenset(roles)
    if not kinds <= KINDS or any(not isinstance(role, str) or not role for role in roles) or not families <= {"T4", "T8", "BF16"}:
        raise ValueError("Unknown selected kind, role or family")
    if len({s.key for s in specs}) != len(specs) or len({s.qname for s in specs}) != len(specs):
        raise ValueError("Unit specification duplicates an executed role or name")
    class_specs = [s for s in specs if s.kind in kinds and s.role in roles and s.family in families]
    source_specs = [s for s in specs if s.kind in kinds and s.role in roles and s.contract == "source_bf16"]
    selected = [s for s in class_specs if s.contract != "source_bf16"]
    if not selected and not allow_empty:
        raise ValueError("Selected perturbation class has no declared rendered units; explicit null replay is required")
    decoded_weights = decoded_weights or {}
    thread = threading.get_ident()
    with _LOCK:
        if layer_module in _LEASES:
            raise RuntimeError("Layer already has a perturbation lease")
        _LEASES[layer_module] = thread
    handles, saved, forwards = [], [], []
    telemetry = {"selected_units": len(selected), "weight_units": 0,
                 "source_passthrough_units": len(source_specs), "explicit_null_replay": not selected,
                 "missing_weight_render": bool(weight and not selected),
                 "weight_gain_status": "not measured by injection API" if selected else "unavailable: no selected rendered weight",
                 "amplitude": amplitude, "activation_rounding": "input dtype after FP32 residual",
                 "native_gpu_equivalence": "not established by CPU proof"}
    try:
        # Validate every decoded geometry before any mutation.
        from canonical_weight import CanonicalWeight
        jobs, canonical_jobs = [], []
        for spec in selected:
            view = unit_view(layer_module, spec)
            if view.ndim != 2 or view.shape[1] % (tp if spec.role == "down_proj" else 1):
                raise ValueError(spec.qname + ": invalid executed Linear geometry")
            if activation and spec.family == "T4" and view.shape[1] % (16 * (tp if spec.role == "down_proj" else 1)):
                raise ValueError(spec.qname + ": invalid static group/TP geometry")
            if weight:
                if spec.qname not in decoded_weights:
                    raise ValueError(spec.qname + ": actual decoded weight is missing")
                decoded = decoded_weights[spec.qname]
                if isinstance(decoded, CanonicalWeight):
                    if spec.family != "BF16" or spec.module is None or tuple(decoded.shape) != tuple(view.shape) or decoded.device != view.device:
                        raise ValueError(spec.qname + ": the canonical epilogue lacks its actual Linear")
                    if not bool(torch.isfinite(decoded.values).all() and torch.isfinite(decoded.row_scales).all()):
                        raise ValueError(spec.qname + ": canonical planes are not finite")
                    canonical_jobs.append((spec.module, view, decoded))
                    continue
                if decoded.shape != view.shape or decoded.device != view.device or not decoded.is_floating_point():
                    raise ValueError(spec.qname + ": decoded shape, device or dtype differs")
                if not bool(torch.isfinite(decoded).all()):
                    raise ValueError(spec.qname + ": decoded weight is nonfinite")
                jobs.append((view, decoded))
        with torch.no_grad():
            for view, decoded in jobs:
                source = view.clone()
                saved.append((view, source))
                view.copy_((source.float() + amplitude * (decoded.float() - source.float())).to(view.dtype))
        for module, source, decoded in canonical_jobs:
            def epilogue(_module,args,output,source=source,decoded=decoded):
                if threading.get_ident() != thread or not args:
                    raise RuntimeError("The canonical epilogue lost its forward lease")
                x = args[0]
                width = x.shape[-1]
                rows = x.reshape(-1,width).float()
                delta = decoded.project(rows) - F.linear(rows,source.float())
                return (output.float() + amplitude * delta.reshape(output.shape)).to(output.dtype)
            handles.append(module.register_forward_hook(epilogue))
        telemetry["canonical_epilogue_units"] = len(canonical_jobs)
        telemetry["weight_units"] = len(saved) + len(canonical_jobs)
        if activation:
            active = {s.key: s for s in selected if s.family != "BF16"}
            def pre_hook(spec):
                def hook(_module, args, kwargs):
                    if threading.get_ident() != thread:
                        raise RuntimeError("Concurrent forward on a leased layer is forbidden")
                    if args:
                        return (activation_residual(args[0], spec, amplitude, tp=tp), *args[1:]), kwargs
                    # Linear's public keyword input has the same last-dimension contract.
                    if "input" not in kwargs:
                        raise ValueError(spec.qname + ": no Linear activation input")
                    return args, dict(kwargs, input=activation_residual(kwargs["input"], spec, amplitude, tp=tp))
                return hook
            for spec in active.values():
                if spec.kind == "routed":
                    continue
                if spec.kind in {"attention", "lm_head"}:
                    module = spec.module
                else:
                    parent = layer_module.mlp.shared_experts if spec.kind == "shared" else layer_module.mlp
                    module = getattr(parent, spec.role)
                handles.append(module.register_forward_pre_hook(pre_hook(spec), with_kwargs=True))
            if any(s.kind == "routed" for s in active.values()):
                experts = layer_module.mlp.experts
                backend = getattr(getattr(experts, "config", None), "_experts_implementation", None)
                if backend not in EAGER_EXPERTS:
                    raise RuntimeError("Packed expert taps need the eager experts backend, not %r; "
                                       "pin it for the whole replay with interceptable_packed_experts" % (backend,))
                had_instance_forward = "forward" in experts.__dict__
                previous = experts.__dict__.get("forward")
                original = experts.forward
                mode = _PackedActivationMode(experts, active, amplitude, tp)
                def forward(*args, **kwargs):
                    if threading.get_ident() != thread:
                        raise RuntimeError("Concurrent packed forward on a leased layer is forbidden")
                    before = mode.calls
                    with mode:
                        result = original(*args, **kwargs)
                    if mode.calls == before:
                        raise RuntimeError("Packed expert backend bypassed Linear/grouped activation taps")
                    return result
                forwards.append((experts, had_instance_forward, previous))
                experts.forward = forward
        yield telemetry
    finally:
        for handle in reversed(handles):
            handle.remove()
        for module, had_instance, previous in reversed(forwards):
            if had_instance:
                module.forward = previous
            else:
                delattr(module, "forward")
        with torch.no_grad():
            for view, source in reversed(saved):
                view.copy_(source)
        with _LOCK:
            del _LEASES[layer_module]