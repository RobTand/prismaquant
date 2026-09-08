"""Research-only joint activation/weight AURA projections and row contract.

This owns no cache. A lease observes the existing streamed layer's baseline
activations and cotangents and consumes its resident production weight deltas.
QDQ is the same owner used by PerturbedActivationCache and assignment KL.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.perturbed_x_cache import (
    _activation_max_abs_lookup, _activation_qdq, _first_tensor_location,
    _served_nvfp4_act_qdq_enabled,
)
from prismaquant.memory_management import env_truthy
from prismaquant.routed_experts import PackedExpertProjection


JOINT_CURRENCY = "joint_aura_predicted_dloss"
JOINT_AURA_COST_CURRENCY = JOINT_CURRENCY
JOINT_AURA_COST_SOURCE = "joint_aura"
PROBE_UNCERTAINTY_SCOPE = "probe_sampling_conditional_on_fixed_calibration"
ASSIGNMENT_OBJECTIVES = ("additive", "joint_quadratic")


@dataclass(frozen=True, slots=True, init=False)
class _ValidatedProbeIdentity(Mapping):
    """Owned immutable JSON fields; mutable values are returned as fresh copies.

    Only this constructor can issue the fast-path type, after checking the
    source model. Serialization deliberately restores an ordinary dict so a
    validation result never crosses a persisted artifact boundary.
    """

    _fields: tuple
    _sha256: str

    def __init__(self, probe):
        encoded = json.dumps(probe, sort_keys=True, separators=(",", ":"), allow_nan=False)
        snapshot = json.loads(encoded)
        from prismaquant.cost_streaming import validate_streamed_model_identity
        validate_streamed_model_identity(probe["source_model"], where="joint AURA row")
        object.__setattr__(self, "_fields", tuple(
            (key, json.dumps(value, separators=(",", ":"), allow_nan=False))
            for key, value in snapshot.items()))
        object.__setattr__(self, "_sha256", hashlib.sha256(encoded.encode()).hexdigest())

    def __getitem__(self, key):
        for name, encoded in self._fields:
            if name == key:
                return json.loads(encoded)
        raise KeyError(key)

    def __iter__(self):
        return (name for name, _ in self._fields)

    def __len__(self):
        return len(self._fields)

    def __reduce__(self):
        return dict, (dict(self),)


def prepare_joint_aura_identities(cost_data):
    """Own shared probe identities for one allocator table, without trusting rows.

    Pickle preserves shared object references. Deduplicate by that reference
    only while retaining it here, then discard the temporary index. No global
    cache, digest-only trust, or caller promise of immutability is involved.
    Row/operator/digest/sample checks still run at every consumption boundary.
    """
    prepared = {}
    for per_unit in cost_data.get("costs", {}).values():
        if not isinstance(per_unit, Mapping):
            continue
        for row in per_unit.values():
            if not isinstance(row, dict):
                continue
            probe = row.get("probe_identity")
            if not isinstance(probe, dict) or probe.get("schema") != "prismaquant.joint_aura.probes.v2":
                continue
            key = id(probe)
            if key not in prepared:
                prepared[key] = (probe, _ValidatedProbeIdentity(probe))
            row["probe_identity"] = prepared[key][1]


def identity_sha256(value) -> str:
    if type(value) is _ValidatedProbeIdentity:
        return value._sha256
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def activation_identity(spec, activation_max_abs: Mapping, qname: str) -> dict:
    """Bind the resolved QDQ policy, including its calibrated static scale."""
    changes = spec.act_quant_changes_input
    maximum = _activation_max_abs_lookup(activation_max_abs, qname) if changes else None
    if maximum is not None and not math.isfinite(float(maximum)):
        raise ValueError(f"joint AURA activation maximum is nonfinite for {qname}")
    contract = spec.static_activation_contract if changes else None
    served = _served_nvfp4_act_qdq_enabled()
    static_scale = None
    if contract is not None and (contract.measured_as_served or served):
        static_scale = contract.require_input_global_scale(
            maximum, qname=qname, consumer="joint AURA",
        )
    quantizer = (contract.quantize_dequantize if static_scale is not None
                 else spec.activation_quantize_dequantize)
    return {
        "schema": "prismaquant.joint_aura.activation.v1",
        "quantizes_input": bool(changes),
        "act_bits": spec.act_bits,
        "act_dtype_name": spec.act_dtype_name,
        "act_group_size": spec.act_group_size,
        "quantizer": (f"{quantizer.__module__}.{quantizer.__qualname__}"
                      if changes else "identity"),
        "static_contract": ({
            "execution": contract.execution, "group_size": contract.group_size,
            "measured_as_served": contract.measured_as_served,
        } if contract is not None else None),
        "activation_max_abs": float(maximum) if maximum is not None else None,
        "input_global_scale": static_scale,
        "clip_enabled": bool(changes and static_scale is None and env_truthy("PRISMAQUANT_PROD_ACT_SCALES", default=True)),
        "served_scales_enabled": served,
    }


def source_execution_identity(model) -> dict:
    """Bind resolved dispatch selectors omitted by Transformers config dumps.

    Include module-local configs because a model may override a single layer's
    backend without changing the root config or any checkpoint tensor.
    """
    modules = {}
    for name, module in model.named_modules():
        config = getattr(module, "config", None)
        selectors = {}
        for label, field in (("attention", "_attn_implementation"),
                             ("experts", "_experts_implementation")):
            if config is not None and hasattr(config, field):
                # Take an independent JSON value, so later config mutation
                # cannot mutate the sealed identity through a shared dict.
                selectors[label] = json.loads(json.dumps(
                    getattr(config, field), sort_keys=True, allow_nan=False))
        if selectors:
            modules[name] = selectors
    return {"schema": "prismaquant.joint_aura.source_execution.v1", "modules": modules}


def arithmetic_identity(measurement_dtype, projection_backend=None) -> dict:
    from .joint_projection_backend import REFERENCE_IDENTITY, validate_projection_backend_identity

    backend_identity = (dict(REFERENCE_IDENTITY) if projection_backend is None
                        else projection_backend.identity)
    validate_projection_backend_identity(backend_identity)
    return {
        "projection_backend": backend_identity,
        "projection_dtype": "torch.float32", "delta_dtype": "torch.float32",
        "measurement_dtype": str(measurement_dtype),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "weight_projection": "output_cotangent_fp32_gemm",
        "residual": "X_dW_T+dX_W_T+dX_dW_T",
        "aggregation": "sum_signed_invocations_then_square",
    }


def prefetch_joint_cache(cache, names, formats_by_qname, *, max_resident_bytes, max_workers=4):
    """Use the production cache's key resolver and prefetch; own no tensors."""
    keys = set()
    for name in names:
        for fmt in formats_by_qname[name]:
            resolved, missing = cache.assignment_keys({name: fmt})
            if missing:
                raise RuntimeError(f"joint AURA production cache missing {name}@{fmt}")
            keys.update(resolved)
    nbytes = cache.estimate_nbytes(list(keys))
    if nbytes > max_resident_bytes:
        raise RuntimeError("joint AURA production cache prefetch exceeds resident budget")
    loaded = cache.prefetch(list(keys), max_workers=max_workers)
    return {"entries": len(keys), "resident_bytes": nbytes, "loaded": loaded, "misses": 0}


class SignedJointProjectionLease:
    """Observe a resident layer and retain only signed scalar probe terms.

    G.T@X gives the weight direction in FP32, including when parameter grads
    would otherwise have been rounded to BF16. G.T@dX is shared by formats
    with the same activation receipt AND the same dynamic QDQ callable.
    """

    def __init__(self, modules, specs_by_qname, delta_weights, *, activation_max_abs=None,
                 projection_backend=None):
        from .joint_projection_backend import require_prewarmed_projection

        self.modules = dict(modules)
        device = next((module.weight.device for module in self.modules.values()
                       if isinstance(module, (nn.Linear, PackedExpertProjection))), torch.device("cpu"))
        self.projection_backend = require_prewarmed_projection(projection_backend, device=device)
        self._projection_product_sum = self.projection_backend.product_sum
        self.specs = specs_by_qname
        self.deltas = delta_weights
        self.activation_max_abs = dict(activation_max_abs or {})
        self.handles = []
        self.forward_originals = []
        self.active = False
        self.terms = {}
        self.groups = {}
        self.telemetry = {"qdq_calls": 0, "operator_gemms": 0, "persistent_cache_entries": 0}
        dense_modules = [mod for mod in self.modules.values() if isinstance(mod, nn.Linear)]
        if len({id(mod) for mod in dense_modules}) != len(dense_modules):
            raise ValueError("joint AURA refuses aliased Linear modules")
        packed_aliases = set()
        for name, module in self.modules.items():
            if isinstance(module, PackedExpertProjection):
                if module.qname != name:
                    raise ValueError(f"joint AURA packed projection name differs for {name}")
                rows = module.output_slice
                alias = (id(module.parameter), module.expert_id, rows.start, rows.stop)
                if alias in packed_aliases:
                    raise ValueError("joint AURA refuses aliased packed projection views")
                packed_aliases.add(alias)
            elif not isinstance(module, nn.Linear):
                raise TypeError(f"joint AURA target {name} is not Linear")
            self.projection_backend.require_device(module.weight.device)
            self._validate_delta_coverage(name, module)
            grouped = {}
            for fmt, spec in self.specs[name].items():
                receipt = activation_identity(spec, self.activation_max_abs, name)
                if receipt["input_global_scale"] is not None:
                    if module.weight.shape[1] % spec.static_activation_contract.group_size:
                        raise ValueError(f"joint AURA static activation group geometry differs for {name}")
                    # A static served contract owns QDQ; different dynamic
                    # registry lambdas are never executed on this path.
                    callable_key = 0
                else:
                    callable_key = id(spec.activation_quantize_dequantize)
                group = (identity_sha256(receipt), callable_key)
                grouped.setdefault(group, (spec, []))[1].append(fmt)
            self.groups[name] = tuple(grouped.values())

    def _validate_delta_coverage(self, name, module):
        expected = {fmt for qname, fmt in self.deltas if qname == name}
        if set(self.specs[name]) != expected:
            raise ValueError(f"joint AURA render/spec coverage mismatch for {name}")
        for fmt in expected:
            delta = self.deltas[(name, fmt)]
            if delta.device != module.weight.device or delta.shape != module.weight.shape:
                raise RuntimeError(f"joint AURA dW residency/shape differs for {name}@{fmt}")

    def __enter__(self):
        if self.handles or self.forward_originals:
            raise RuntimeError("joint AURA lease already entered")
        packed = {}
        try:
            for name, module in self.modules.items():
                if isinstance(module, PackedExpertProjection):
                    packed.setdefault(id(module.module), []).append(module)
                else:
                    self.handles.append(module.register_forward_hook(self._hook(name), with_kwargs=True))
            for members in packed.values():
                self._install_packed_observer(members)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_args):
        self._remove_observers()
        self.terms.clear()
        self.active = False

    def _remove_observers(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module, original in reversed(self.forward_originals):
            module.forward = original
        self.forward_originals.clear()

    def _install_packed_observer(self, members):
        """Observe the existing per-expert Linear slices without changing them.

        F.linear uses the packed Fisher/activation tap's exact dim-0 slice
        recognizer. F.grouped_mm uses the exact packed transpose and cumulative
        expert row offsets. Both execute the original operator unchanged; the
        baseline forward, expert backend and routing arithmetic are preserved.
        """
        from prismaquant.sensitivity_probe import _packed_expert_slice_index

        module = members[0].module
        original = module.forward
        parameters = {id(member.parameter): member.parameter for member in members}
        by_expert = {}
        for member in members:
            by_expert.setdefault((id(member.parameter), member.expert_id), []).append(member)

        def forward(*args, **kwargs):
            if not self.active:
                raise RuntimeError("joint AURA packed forward outside active probe")
            seen = set()
            original_linear = F.linear
            original_grouped = getattr(F, "grouped_mm", None)

            def linear(inputs, weight, bias=None):
                base = weight._base if weight._is_view() else weight
                parameter = parameters.get(id(base))
                if parameter is None:
                    return original_linear(inputs, weight, bias)
                expert = _packed_expert_slice_index(weight, parameter)
                if expert is None:
                    raise RuntimeError("joint AURA packed weight escaped its declared Linear slice")
                seen.add(id(parameter))
                output = original_linear(inputs, weight, bias)
                for member in by_expert.get((id(parameter), expert), ()):
                    self._observe(member.qname, member.weight, inputs, output, member.output_slice)
                return output

            def grouped(inputs, weight, *, offs=None, **kwargs):
                base = weight._base if weight._is_view() else weight
                parameter = parameters.get(id(base))
                if parameter is None:
                    return original_grouped(inputs, weight, offs=offs, **kwargs)
                expected = parameter.transpose(-2, -1)
                if (weight.shape != expected.shape or weight.stride() != expected.stride()
                        or weight.storage_offset() != expected.storage_offset()
                        or weight.untyped_storage().data_ptr() != parameter.untyped_storage().data_ptr()
                        or inputs.ndim != 2 or inputs.shape[1] != parameter.shape[2]
                        or not isinstance(offs, torch.Tensor) or offs.ndim != 1
                        or len(offs) != parameter.shape[0] or offs.dtype not in (torch.int32, torch.int64)):
                    raise RuntimeError("joint AURA grouped weight/offset geometry differs")
                ends = offs.tolist()
                if any(end < start for start, end in zip([0, *ends[:-1]], ends)) or ends[-1] > inputs.shape[0]:
                    raise RuntimeError("joint AURA grouped offsets are outside input rows")
                output = original_grouped(inputs, weight, offs=offs, **kwargs)
                seen.add(id(parameter))
                start = 0
                for expert, end in enumerate(ends):
                    if end > start:
                        for member in by_expert.get((id(parameter), expert), ()):
                            self._observe(member.qname, member.weight, inputs[start:end], output,
                                          member.output_slice, slice(start, end))
                    start = end
                return output

            F.linear = linear
            if original_grouped is not None:
                F.grouped_mm = grouped
            try:
                result = original(*args, **kwargs)
            finally:
                F.linear = original_linear
                if original_grouped is not None:
                    F.grouped_mm = original_grouped
            if seen != set(parameters):
                raise RuntimeError("joint AURA packed source did not execute declared F.linear/grouped_mm boundaries")
            return result

        self.forward_originals.append((module, original))
        module.forward = forward

    def begin_probe(self):
        if self.active:
            raise RuntimeError("joint AURA probe already active")
        self.terms.clear()
        self.active = True

    def _hook(self, name):
        def observe(module, args, kwargs, output):
            _, _, x = _first_tensor_location(args, kwargs)
            self._observe(name, module.weight, x, output)
        return observe

    def _observe(self, name, source_weight, x, output, output_slice=None, row_slice=None):
        if not self.active:
            raise RuntimeError("joint AURA forward outside active probe")
        if not isinstance(x, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise TypeError(f"joint AURA Linear {name} needs Tensor input/output")
        # Detach, without copying the baseline activation or retaining its
        # upstream graph. The existing reverse window owns its lifetime.
        x = x.detach()

        @torch.no_grad()
        def project(gradient):
            selected = gradient if row_slice is None else gradient[row_slice]
            selected = selected if output_slice is None else selected[..., output_slice]
            if x.device != selected.device or x.device != source_weight.device:
                raise RuntimeError(f"joint AURA residency mismatch for {name}")
            if x.shape[:-1] != selected.shape[:-1] or x.shape[-1] != source_weight.shape[1] or selected.shape[-1] != source_weight.shape[0]:
                raise RuntimeError(f"joint AURA Linear geometry/shape mismatch for {name}")
            x2 = x.reshape(-1, x.shape[-1]).float()
            g2 = selected.reshape(-1, selected.shape[-1]).float()
            gw = g2.T @ x2
            for spec, formats in self.groups[name]:
                d_operator = None
                activation = torch.zeros((), device=x.device)
                if spec.act_quant_changes_input:
                    quantized = _activation_qdq(x, spec, self.activation_max_abs, name)
                    if not isinstance(quantized, torch.Tensor) or quantized.shape != x.shape or quantized.device != x.device or quantized.dtype != x.dtype:
                        raise RuntimeError(f"joint AURA QDQ changed residency/dtype/shape for {name}")
                    dx = quantized.reshape_as(x2).float() - x2
                    d_operator = g2.T @ dx
                    activation = self._projection_product_sum(d_operator, source_weight.float())
                    self.telemetry["qdq_calls"] += 1
                    self.telemetry["operator_gemms"] += 1
                for fmt in formats:
                    delta = self.deltas[(name, fmt)].float()
                    weight = self._projection_product_sum(gw, delta)
                    mixed = (self._projection_product_sum(d_operator, delta) if d_operator is not None
                             else torch.zeros((), device=x.device))
                    components = torch.stack((weight, activation, mixed))
                    key = (name, fmt)
                    self.terms[key] = self.terms.get(key, 0) + components
            return gradient

        if output.requires_grad:
            output.register_hook(project)

    def finish_probe(self):
        if not self.active:
            raise RuntimeError("joint AURA probe is not active")
        self.active = False
        result = {}
        for key in self.deltas:
            values = self.terms.get(key)
            weight, activation, mixed = ([float(x) for x in values.tolist()]
                                         if values is not None else [0.0, 0.0, 0.0])
            total = weight + activation + mixed
            if not all(math.isfinite(x) for x in (weight, activation, mixed, total)):
                raise RuntimeError(f"joint AURA nonfinite signed projection for {key}")
            result[key] = {"weight": weight, "activation": activation, "mixed": mixed, "total": total}
        self.terms.clear()
        return result


class JointOperatorStatisticsLease(SignedJointProjectionLease):
    """Opt-in, one-probe deferred contraction using the shared source observers.

    Retain sum(G.T @ X) and sum(G.T @ dX) per activation group, independent
    of candidate count. After baseline backwards finish, seal observation,
    release this lease's source references, and project caller-owned resident
    candidate dW quanta. The caller's ProductionWeightCache owns candidate
    prefetch and its physical budget. No candidate tensor is retained here.

    ``max_statistics_bytes`` covers FP32 operator matrices only. Baseline
    source/activations/cotangents, GEMM inputs and temporary matrices, source
    conversion, projection workspaces, CUDA reservations and allocator overhead
    still need separate phase admission. This primitive is not a full fit gate.
    ``max_candidate_bytes`` charges complete backing storages, including any
    unselected data kept alive by a caller's view, for each projection quantum.
    Matrix accumulation changes rounding and has a distinct arithmetic identity.
    """

    def __init__(self, modules, specs_by_qname, *, max_statistics_bytes, max_candidate_bytes,
                 activation_max_abs=None, projection_backend=None):
        if type(max_statistics_bytes) is not int or max_statistics_bytes < 0:
            raise ValueError("joint statistics budget must be a nonnegative integer")
        if type(max_candidate_bytes) is not int or max_candidate_bytes <= 0:
            raise ValueError("joint candidate budget must be a positive integer")
        self.max_candidate_bytes = max_candidate_bytes
        if not modules or set(modules) != set(specs_by_qname) or any(not specs for specs in specs_by_qname.values()):
            raise ValueError("joint statistics module/spec coverage differs")
        # FormatSpec is mutable; freeze its resolved fields for this lease.
        specs = {name: {fmt: copy(spec) for fmt, spec in choices.items()}
                 for name, choices in specs_by_qname.items()}
        super().__init__(modules, specs, {}, activation_max_abs=activation_max_abs,
                         projection_backend=projection_backend)
        self._geometry = {name: (tuple(module.weight.shape), module.weight.device)
                          for name, module in self.modules.items()}
        self._sources = {name: self._source_fingerprint(module.weight)
                         for name, module in self.modules.items()}
        self._format_groups = {(name, fmt): index
                               for name, groups in self.groups.items()
                               for index, (_, formats) in enumerate(groups) for fmt in formats}
        self.statistics_capacity_bytes = sum(
            module.weight.numel() * 4 * (1 + sum(spec.act_quant_changes_input
                                                 for spec, _ in self.groups[name]))
            for name, module in self.modules.items())
        if self.statistics_capacity_bytes > max_statistics_bytes:
            raise RuntimeError("joint statistics matrices exceed statistics budget")
        self._operators = {}
        self._activation_terms = {}
        self._results = {}
        self._pending_backwards = 0
        self._observation_inputs = {}
        self._phase = 'new'
        self.telemetry.update(statistics_capacity_bytes=self.statistics_capacity_bytes,
                              peak_statistics_bytes=0, projected_candidates=0,
                              peak_candidate_storage_bytes=0)

    def _validate_delta_coverage(self, name, module):
        # The exact candidate roster is sealed now; tensors arrive only after
        # source observation, through project(), which checks each quantum.
        if name not in self.specs or not self.specs[name]:
            raise ValueError(f"joint statistics missing spec coverage for {name}")

    @staticmethod
    def _source_fingerprint(weight):
        return (weight.data_ptr(), weight._version, tuple(weight.shape),
                tuple(weight.stride()), weight.storage_offset(), weight.dtype, weight.device)

    def _require_source(self, name, weight):
        if self._source_fingerprint(weight) != self._sources[name]:
            raise RuntimeError(f"joint statistics source weight changed for {name}")

    @property
    def resident_statistics_bytes(self):
        return sum(value.numel() * value.element_size() for value in self._operators.values())

    def arithmetic_identity(self, measurement_dtype):
        identity = arithmetic_identity(measurement_dtype, self.projection_backend)
        identity.update(
            weight_projection='summed_output_operator_fp32_gemm',
            operator_accumulation='sum_fp32_matrices_in_backward_invocation_order',
            contraction_order='sum_operators_then_project_each_signed_component')
        return identity

    def begin_probe(self):
        if self._phase != 'new' or not (self.handles or self.forward_originals):
            raise RuntimeError("joint statistics requires one new entered lease per probe")
        for name, module in self.modules.items():
            self._require_source(name, module.weight)
        self._phase, self.active = 'observing', True

    def __enter__(self):
        if self._phase != 'new':
            raise RuntimeError("joint statistics cannot reenter a consumed lease")
        return super().__enter__()

    def _accumulate(self, key, matrix):
        if key in self._operators:
            self._operators[key].add_(matrix)
        else:
            self._operators[key] = matrix
        self.telemetry['peak_statistics_bytes'] = max(
            self.telemetry['peak_statistics_bytes'], self.resident_statistics_bytes)

    def _release_observation_inputs(self):
        for inputs in self._observation_inputs.values():
            inputs.clear()
        self._observation_inputs.clear()

    def _observe(self, name, source_weight, x, output, output_slice=None, row_slice=None):
        if self._phase != 'observing' or not self.active:
            raise RuntimeError("joint statistics forward outside active observation")
        self._require_source(name, source_weight)
        if not isinstance(x, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise TypeError(f"joint statistics Linear {name} needs Tensor input/output")
        inputs = [x.detach(), source_weight]
        consumed = False

        @torch.no_grad()
        def collect(gradient):
            nonlocal consumed
            if self._phase != 'observing' or not self.active or consumed:
                raise RuntimeError("joint statistics backward outside active observation")
            try:
                x, source_weight = inputs
                self._require_source(name, source_weight)
                selected = gradient if row_slice is None else gradient[row_slice]
                selected = selected if output_slice is None else selected[..., output_slice]
                if x.device != selected.device or x.device != source_weight.device:
                    raise RuntimeError(f"joint statistics residency mismatch for {name}")
                if (x.shape[:-1] != selected.shape[:-1] or x.shape[-1] != source_weight.shape[1]
                        or selected.shape[-1] != source_weight.shape[0]):
                    raise RuntimeError(f"joint statistics Linear geometry/shape mismatch for {name}")
                x2 = x.reshape(-1, x.shape[-1]).float()
                g2 = selected.reshape(-1, selected.shape[-1]).float()
                self._accumulate((name, None), g2.T @ x2)
                self.telemetry['operator_gemms'] += 1
                for index, (spec, _) in enumerate(self.groups[name]):
                    if not spec.act_quant_changes_input:
                        continue
                    quantized = _activation_qdq(x, spec, self.activation_max_abs, name)
                    if (not isinstance(quantized, torch.Tensor) or quantized.shape != x.shape
                            or quantized.device != x.device or quantized.dtype != x.dtype):
                        raise RuntimeError(f"joint statistics QDQ changed residency/dtype/shape for {name}")
                    dx = quantized.reshape_as(x2).float() - x2
                    self._accumulate((name, index), g2.T @ dx)
                    self.telemetry['qdq_calls'] += 1
                    self.telemetry['operator_gemms'] += 1
                consumed = True
                self._pending_backwards -= 1
            except BaseException:
                # A QDQ/GEMM may fail after an earlier operator committed.
                # Never allow a caught backward failure to become a retry.
                self._phase, self.active = 'failed', False
                self._remove_observers()
                self.modules.clear()
                self._operators.clear()
                self._release_observation_inputs()
                raise
            finally:
                inputs.clear()
                self._observation_inputs.pop(id(inputs), None)
            return gradient

        if output.requires_grad:
            self._pending_backwards += 1
            self._observation_inputs[id(inputs)] = inputs
            output.register_hook(collect)

    @torch.no_grad()
    def finish_observations(self):
        if self._phase != 'observing':
            raise RuntimeError("joint statistics observations are not active")
        if self._pending_backwards:
            raise RuntimeError("joint statistics has pending backward observations")
        for name, module in self.modules.items():
            self._require_source(name, module.weight)
            for index, (spec, _) in enumerate(self.groups[name]):
                operator = self._operators.get((name, index))
                value = (float(self._projection_product_sum(operator, module.weight.float()))
                         if operator is not None else 0.)
                if not math.isfinite(value):
                    raise RuntimeError(f"joint statistics nonfinite activation projection for {name}")
                self._activation_terms[(name, index)] = value
        self._remove_observers()
        self.modules.clear()
        self._phase, self.active = 'ready', False

    @torch.no_grad()
    def project(self, delta_weights):
        if self._phase != 'ready':
            raise RuntimeError("joint statistics is not ready for candidate projection")
        if not isinstance(delta_weights, Mapping) or not delta_weights:
            raise ValueError("joint statistics requires a nonempty candidate quantum")
        storages = {}
        for key, delta in delta_weights.items():
            if key not in self._format_groups:
                raise ValueError(f"joint statistics unknown candidate {key}")
            if key in self._results:
                raise ValueError(f"joint statistics duplicate candidate {key}")
            shape, device = self._geometry[key[0]]
            if (not isinstance(delta, torch.Tensor) or tuple(delta.shape) != shape
                    or delta.device != device or not delta.is_floating_point()):
                raise RuntimeError(f"joint statistics candidate residency/dtype/shape differs for {key}")
            storage = delta.untyped_storage()
            storages[(delta.device, storage.data_ptr())] = storage.nbytes()
        candidate_bytes = sum(storages.values())
        if candidate_bytes > self.max_candidate_bytes:
            raise RuntimeError("joint statistics candidate storage exceeds candidate budget")
        # Commit scalar results only after the whole quantum succeeds.
        results = {}
        for key, delta in delta_weights.items():
            name, fmt = key
            group = self._format_groups[key]
            dw = delta.float()
            gw = self._operators.get((name, None))
            ga = self._operators.get((name, group))
            if gw is None and not bool(torch.isfinite(dw).all()):
                raise RuntimeError(f"joint statistics nonfinite unrouted candidate for {key}")
            weight = float(self._projection_product_sum(gw, dw)) if gw is not None else 0.
            activation = self._activation_terms[(name, group)]
            mixed = float(self._projection_product_sum(ga, dw)) if ga is not None else 0.
            total = weight + activation + mixed
            if not all(math.isfinite(value) for value in (weight, activation, mixed, total)):
                raise RuntimeError(f"joint statistics nonfinite signed projection for {key}")
            results[key] = dict(weight=weight, activation=activation, mixed=mixed, total=total)
        self._results.update(results)
        self.telemetry['projected_candidates'] += len(results)
        self.telemetry['peak_candidate_storage_bytes'] = max(
            self.telemetry['peak_candidate_storage_bytes'], candidate_bytes)
        return {key: dict(value) for key, value in results.items()}

    def finish_projections(self):
        if self._phase != 'ready':
            raise RuntimeError("joint statistics is not ready to seal projections")
        if set(self._results) != set(self._format_groups):
            raise RuntimeError("joint statistics candidate coverage is incomplete")
        result = {key: dict(value) for key, value in self._results.items()}
        self._operators.clear()
        self._activation_terms.clear()
        self._results.clear()
        self._phase = 'complete'
        return result

    def finish_probe(self):
        raise RuntimeError("joint statistics requires observation and candidate projection seals")

    def __exit__(self, *_args):
        super().__exit__(*_args)
        self.modules.clear()
        self._release_observation_inputs()
        self._operators.clear()
        self._activation_terms.clear()
        self._results.clear()
        self._phase = 'closed'


def validate_joint_aura_entry(entry: Mapping) -> bool:
    """Recognize joint claims and fail closed before any scalar cost branch."""
    claims = (entry.get("cost_source") == "joint_aura" or
              entry.get("cost_currency") == JOINT_CURRENCY or
              "joint_operator_identity" in entry or "joint_operator_identity_sha256" in entry)
    if not claims:
        return False
    # A single draw provides no estimate of sampling variance. Refuse it
    # instead of publishing a zero standard error that looks like certainty.
    if (not isinstance(entry.get("probe_identity"), Mapping)
            or type(entry["probe_identity"].get("n_probes")) is not int
            or entry["probe_identity"]["n_probes"] < 2):
        raise ValueError("joint AURA requires at least two probes for sampling uncertainty")
    expected = {"cost_source": "joint_aura", "cost_currency": JOINT_CURRENCY,
                "fisher_application_count": 1, "activation_quantization_included": True,
                "output_mse_measured": False}
    for key, value in expected.items():
        if entry.get(key) != value or type(entry.get(key)) is not type(value):
            raise ValueError(f"joint AURA invalid {key}")
    if any(key in entry for key in ("act_dloss", "act_dloss_applied", "aqua_activation_dloss", "activation_pricing_applied", "output_mse")):
        raise ValueError("joint AURA refuses a second activation/Fisher application")
    operator = entry.get("joint_operator_identity")
    if not isinstance(operator, Mapping) or operator.get("schema") != "prismaquant.joint_aura.operator.v2":
        raise ValueError("joint AURA requires v2 operator identity; legacy artifacts require fresh prepare and recompute")
    if identity_sha256(operator) != entry.get("joint_operator_identity_sha256"):
        raise ValueError("joint AURA operator identity digest mismatch")
    probe = entry.get("probe_identity")
    digest = entry.get("probe_identity_sha256")
    if not isinstance(probe, Mapping) or probe.get("schema") != "prismaquant.joint_aura.probes.v2" or identity_sha256(probe) != digest or operator.get("probe_identity_sha256") != digest:
        raise ValueError("joint AURA probe identity mismatch")
    from prismaquant.cost_streaming import validate_streamed_model_identity
    try:
        if type(probe) is not _ValidatedProbeIdentity:
            validate_streamed_model_identity(probe["source_model"], where="joint AURA row")
        for field in ("calibration_sha256", "producer_source_sha256"):
            if re.fullmatch(r"[a-f0-9]{64}", probe[field]) is None:
                raise ValueError(f"invalid {field}")
        if type(probe["n_probes"]) is not int or probe["n_probes"] < 1 or type(probe["seed_base"]) is not int:
            raise ValueError("invalid probe indices")
        if probe["distribution"] != "rademacher" or probe["normalization"] != "global_kl_fisher":
            raise ValueError("invalid probe distribution/normalization")
        if not math.isfinite(float(probe["temperature"])) or probe["temperature"] <= 0:
            raise ValueError("invalid probe temperature")
        if not isinstance(operator["qname"], str) or not operator["qname"] or not isinstance(operator["format"], str) or not operator["format"]:
            raise ValueError("invalid operator coordinate")
        for field in ("source_weight", "rendered_weight"):
            tensor = operator[field]
            if len(tensor["shape"]) != 2 or any(type(dim) is not int or dim <= 0 for dim in tensor["shape"]):
                raise ValueError("invalid tensor geometry")
            if re.fullmatch(r"[a-f0-9]{64}", tensor["content_sha256"]) is None:
                raise ValueError("invalid tensor content hash")
            if tensor["dtype"] not in ("torch.float32", "torch.bfloat16", "torch.float16", "torch.float64") or type(tensor["logical_bytes"]) is not int or tensor["logical_bytes"] <= 0:
                raise ValueError("invalid tensor storage identity")
        if operator["source_weight"]["shape"] != operator["rendered_weight"]["shape"]:
            raise ValueError("render/source geometry differs")
        activation = operator["activation"]
        if activation["schema"] != "prismaquant.joint_aura.activation.v1" or type(activation["quantizes_input"]) is not bool:
            raise ValueError("invalid activation identity")
        for field in ("activation_max_abs", "input_global_scale"):
            if activation[field] is not None and not math.isfinite(float(activation[field])):
                raise ValueError("nonfinite activation scale")
        arithmetic = operator["arithmetic"]
        from .joint_projection_backend import validate_projection_backend_identity
        validate_projection_backend_identity(arithmetic["projection_backend"])
        if arithmetic != probe["arithmetic"] or arithmetic["projection_dtype"] != "torch.float32" or arithmetic["delta_dtype"] != "torch.float32" or arithmetic["aggregation"] != "sum_signed_invocations_then_square":
            raise ValueError("invalid projection arithmetic")
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError(f"joint AURA incomplete identity: {exc}") from exc
    ids = entry.get("probe_ids")
    expected_ids = [int(probe["seed_base"]) + k for k in range(int(probe["n_probes"]))]
    signed, squared = entry.get("signed_per_probe"), entry.get("x2_per_probe")
    if not expected_ids or ids != expected_ids or not isinstance(signed, list) or not isinstance(squared, list) or len(signed) != len(ids) or len(squared) != len(ids):
        raise ValueError("joint AURA probe alignment mismatch")
    if any(not math.isfinite(float(a)) or not math.isfinite(float(b)) or not math.isclose(float(b), float(a)**2, rel_tol=1e-12, abs_tol=1e-30) for a, b in zip(signed, squared)):
        raise ValueError("joint AURA signed/squared sample mismatch")
    mean = 0.5 * sum(squared) / len(squared)
    if not math.isclose(float(entry["predicted_dloss"]), mean, rel_tol=1e-12, abs_tol=1e-30):
        raise ValueError("joint AURA predicted loss differs from aligned samples")
    stderr = float(entry.get("predicted_dloss_stderr", float("nan")))
    sample_mean = 2 * mean
    variance = sum((x - sample_mean)**2 for x in squared) / (len(squared) - 1) if len(squared) > 1 else 0.0
    expected_stderr = 0.5 * math.sqrt(variance / len(squared))
    if not math.isfinite(stderr) or stderr < 0 or not math.isclose(stderr, expected_stderr, rel_tol=1e-12, abs_tol=1e-30):
        raise ValueError("joint AURA invalid probe standard error")
    components = entry.get("signed_components_per_probe")
    if not isinstance(components, list) or len(components) != len(signed):
        raise ValueError("joint AURA incomplete signed components")
    for value, total in zip(components, signed):
        if not isinstance(value, Mapping) or set(value) != {"weight", "activation", "mixed", "total"}:
            raise ValueError("joint AURA invalid signed components")
        if not all(math.isfinite(float(x)) for x in value.values()) or value["total"] != total or not math.isclose(value["weight"] + value["activation"] + value["mixed"], total, rel_tol=1e-12, abs_tol=1e-30):
            raise ValueError("joint AURA component/signed sample mismatch")
    return True


def squared_signed(total) -> float:
    """The one squaring of a signed joint projection.

    ``x ** 2`` goes through C ``pow`` and ``x * x`` is a correctly rounded
    multiply; on glibc they differ by one ulp for some inputs. The streamed
    accumulator and this row must square identically, because checkpoint
    reload compares the two sample lists exactly (PQ #261, run-02: 9 of 128
    samples of ``model.layers.0.mlp.down_proj@TESSERA_E4M3_K1_R896`` differed).
    """
    total = float(total)
    return total * total


def make_joint_aura_entry(*, operator_identity, probe_identity, signed_components) -> dict:
    """Publish one complete, aligned cost row, also used by checkpoint replay."""
    signed = [float(value["total"]) for value in signed_components]
    squared = [squared_signed(value) for value in signed]
    n = len(squared)
    if not n:
        raise ValueError("joint AURA needs at least one signed probe")
    mean = sum(squared) / n
    variance = sum((value - mean)**2 for value in squared) / (n - 1) if n > 1 else 0.0
    row = {
        "cost_currency": JOINT_CURRENCY, "cost_source": "joint_aura",
        "fisher_application_count": 1, "activation_quantization_included": True,
        "output_mse_measured": False, "predicted_dloss": 0.5 * mean,
        "predicted_dloss_stderr": 0.5 * math.sqrt(variance / n),
        "signed_per_probe": signed, "signed_components_per_probe": signed_components,
        "x2_per_probe": squared,
        "probe_ids": [probe_identity["seed_base"] + k for k in range(n)],
        "probe_identity": probe_identity,
        "probe_identity_sha256": identity_sha256(probe_identity),
        "joint_operator_identity": operator_identity,
        "joint_operator_identity_sha256": identity_sha256(operator_identity),
        "measurement_status": "research",
        "uncertainty_scope": "probe_sampling_conditional_on_fixed_calibration",
    }
    validate_joint_aura_entry(row)
    return row


def paired_candidate_difference(entry_a: Mapping, entry_b: Mapping) -> dict:
    """A minus B with common-probe covariance retained, conditional on calibration."""
    if not validate_joint_aura_entry(entry_a) or not validate_joint_aura_entry(entry_b):
        raise ValueError("paired joint AURA requires joint rows")
    if not _same_probe_identity(entry_a, entry_b):
        raise ValueError("paired joint AURA probe alignment mismatch")
    values = [0.5 * (a - b) for a, b in zip(entry_a["x2_per_probe"], entry_b["x2_per_probe"])]
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean)**2 for value in values) / (n - 1) if n > 1 else 0.0
    return {"mean_difference": mean, "paired_standard_error": math.sqrt(variance / n),
            "difference_per_probe": values, "probe_ids": list(entry_a["probe_ids"]),
            "probe_identity_sha256": entry_a["probe_identity_sha256"],
            "uncertainty_scope": "probe_sampling_conditional_on_fixed_calibration"}


def _same_probe_identity(left: Mapping, right: Mapping) -> bool:
    # Hashes have already been checked against canonical JSON by the row
    # validator. Python equality would conflate distinct JSON true/1/1.0.
    return (left["probe_identity_sha256"] == right["probe_identity_sha256"]
            and identity_sha256(left["probe_ids"]) == identity_sha256(right["probe_ids"]))


def _validated_assignment(rows: Mapping, objective: str) -> dict:
    if objective not in ASSIGNMENT_OBJECTIVES:
        raise ValueError(f"unsupported joint AURA assignment objective: {objective!r}")
    if not isinstance(rows, Mapping) or not rows:
        raise ValueError("joint AURA assignment requires a nonempty unit roster")
    if any(not isinstance(name, str) or not name for name in rows):
        raise ValueError("joint AURA assignment requires named units")
    ordered = {name: rows[name] for name in sorted(rows)}
    reference = None
    for name, row in ordered.items():
        if not isinstance(row, Mapping) or not validate_joint_aura_entry(row):
            raise ValueError("joint AURA assignment requires complete joint rows")
        if row["joint_operator_identity"]["qname"] != name:
            raise ValueError(f"joint AURA assignment operator coordinate mismatch: {name}")
        if reference is None:
            reference = row
        elif not _same_probe_identity(row, reference):
            raise ValueError("joint AURA assignment probe alignment mismatch")
    return ordered


def _probe_moments(values: list[float]) -> tuple[float, float]:
    """Empirical sample SE; the measured calibration remains fixed."""
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("joint AURA diagnostic requires finite aligned samples")
    mean = math.fsum(value / len(values) for value in values)
    # hypot avoids overflow in the sum of squared deviations.
    stderr = math.hypot(*(value - mean for value in values)) / math.sqrt(
        len(values) * (len(values) - 1))
    if not math.isfinite(mean) or not math.isfinite(stderr):
        raise ValueError("joint AURA diagnostic sample moments overflow")
    return mean, stderr


def _assignment_metadata(rows: Mapping, objective: str) -> dict:
    reference = next(iter(rows.values()))
    identities = {name: row["joint_operator_identity_sha256"]
                  for name, row in rows.items()}
    return {
        "objective": objective, "cost_currency": JOINT_CURRENCY,
        "probe_ids": list(reference["probe_ids"]),
        "probe_identity_sha256": reference["probe_identity_sha256"],
        "operator_identity_sha256_by_unit": identities,
        "assignment_identity_sha256": identity_sha256(identities),
        "uncertainty_scope": PROBE_UNCERTAINTY_SCOPE,
        "measurement_status": "research",
    }


def assignment_probe_summary(rows: Mapping, *, objective: str = "additive") -> dict:
    """Summarize one complete assignment on validated, common signed probes.

    ``additive`` is the allocator's sum of local quadratic prices:
    0.5 sum_i(a_i[k]**2). ``joint_quadratic`` is 0.5 (sum_i a_i[k])**2,
    retaining cross-unit terms in the baseline's local linearization. Neither
    updates the background model nor measures held-out assignment quality.
    """
    rows = _validated_assignment(rows, objective)
    samples = zip(*(row["signed_per_probe"] for row in rows.values()))
    values = [0.5 * (math.fsum(x * x for x in sample) if objective == "additive"
                     else math.fsum(sample) ** 2) for sample in samples]
    mean, stderr = _probe_moments(values)
    return {"schema": "prismaquant.joint_aura.assignment_summary.v1",
            **_assignment_metadata(rows, objective),
            "mean": mean, "standard_error": stderr, "per_probe": values}


def paired_assignment_difference(
    rows_a: Mapping, rows_b: Mapping, *, objective: str = "additive",
) -> dict:
    """A minus B, retaining common-probe covariance conditional on calibration.

    Both arms must name the complete same unit roster, including unchanged
    units (which still contribute cross terms to ``joint_quadratic``). Each
    candidate binds its own actual render/activation operator. Different
    formats may differ there, but the same candidate cannot silently change
    operator identity, and each unit must retain the same source weight.
    """
    a, b = _validated_assignment(rows_a, objective), _validated_assignment(rows_b, objective)
    if a.keys() != b.keys():
        raise ValueError("paired joint AURA requires the same complete unit roster")
    pairs = []
    for name in a:
        left, right = a[name], b[name]
        operator_a, operator_b = left["joint_operator_identity"], right["joint_operator_identity"]
        if not _same_probe_identity(left, right):
            raise ValueError("paired joint AURA assignment probe alignment mismatch")
        if identity_sha256(operator_a["source_weight"]) != identity_sha256(operator_b["source_weight"]):
            raise ValueError(f"paired joint AURA source weight identity mismatch: {name}")
        if (operator_a["format"] == operator_b["format"]
                and left["joint_operator_identity_sha256"] != right["joint_operator_identity_sha256"]):
            raise ValueError(f"paired joint AURA changed operator identity for the same candidate: {name}")
        pairs.append((left, right))
    if objective == "additive":
        # The candidate-difference algebra, with every signed squared term
        # retained until fsum: neither rounded assignment totals nor rounded
        # per-unit differences may erase a small residual across unit changes.
        values = [math.fsum(sign * 0.5 * row["x2_per_probe"][k]
                            for pair in pairs for sign, row in zip((1, -1), pair))
                  for k in range(len(pairs[0][0]["probe_ids"]))]
    else:
        values = []
        for k in range(len(pairs[0][0]["probe_ids"])):
            # Difference of squares, factored before summing the background.
            delta = math.fsum(sign * row["signed_per_probe"][k]
                              for pair in pairs for sign, row in zip((1, -1), pair))
            total = math.fsum(row["signed_per_probe"][k]
                              for pair in pairs for row in pair)
            values.append(0.5 * delta * total)
    mean, stderr = _probe_moments(values)
    metadata_a, metadata_b = _assignment_metadata(a, objective), _assignment_metadata(b, objective)
    return {
        "schema": "prismaquant.joint_aura.paired_assignment_difference.v1",
        "objective": objective, "cost_currency": JOINT_CURRENCY,
        "mean_difference": mean, "paired_standard_error": stderr,
        "difference_per_probe": values, "probe_ids": metadata_a["probe_ids"],
        "probe_identity_sha256": metadata_a["probe_identity_sha256"],
        "assignment_a": metadata_a, "assignment_b": metadata_b,
        "uncertainty_scope": PROBE_UNCERTAINTY_SCOPE, "measurement_status": "research",
    }
