"""Native panel identities shared by receipt emission and admission.

Each published schema retains its own producer consumer and execution projection.
The registry owns row bindings, scope and immutable late-binding envelopes.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from prismaquant.joint_aura_rows import joint_identity_sha256
from .measured_runtime_prices import RuntimePriceError

DENSE_PANEL_SCHEMA = "tessera.native_dense_panel.v1"
MOE_PANEL_SCHEMA = "tessera.native_moe_panel.v1"
DENSE_EXECUTION_PANEL_SCHEMA = "tessera.native_dense_execution_panel.v1"
MOE_EXECUTION_PANEL_SCHEMA = "tessera.native_moe_execution_panel.v1"
DENSE_EXECUTION_RECEIPT_SCHEMA = "tessera.native_dense_execution_receipt.v1"
MOE_EXECUTION_RECEIPT_SCHEMA = "tessera.native_moe_execution_receipt.v1"
DENSE_BINDING_SCHEMA = "prismaquant.native_dense_late_binding.v1"
MOE_BINDING_SCHEMA = "prismaquant.native_moe_late_binding.v1"


def member_execution_identity(member):
    return {"qname": member["unit"], **{key: member[key] for key in
            ("format", "source_weight", "rendered_weight", "activation")}}


def _dense_execution_panel(panel):
    value = copy.deepcopy(panel)
    joint = value.pop("joint_operator_identity")
    if joint_identity_sha256(joint) != value.pop("joint_operator_identity_sha256"):
        raise ValueError("final joint operator digest differs")
    value.pop("cost_sha256")
    value.pop("probe_identity_sha256")
    value["schema"] = DENSE_EXECUTION_PANEL_SCHEMA
    value["operator_identity"] = {key: joint[key] for key in
        ("qname", "format", "source_weight", "rendered_weight", "activation")}
    value["operator_identity_sha256"] = joint_identity_sha256(value["operator_identity"])
    return value


def _moe_execution_panel(panel):
    value = copy.deepcopy(panel)
    value["schema"] = MOE_EXECUTION_PANEL_SCHEMA
    value.pop("cost_sha256")
    value.pop("probe_identity_sha256")
    value["runtime_binding"].pop("member_operator_identity_sha256")
    value["runtime_binding"]["member_execution_identity_sha256"] = {
        member["unit"]: joint_identity_sha256(member_execution_identity(member))
        for member in value["members"]}
    return value


def _dense_row_binding(panel):
    from .native_operator_panel import operator_route_identity
    unit = panel["unit"]
    return {"member_formats": {unit: panel["format"]},
            "member_operator_identity_sha256": {unit: panel["joint_operator_identity_sha256"]},
            "member_shapes": {unit: list(panel["shape"])},
            "operator_route": operator_route_identity(panel["phases"]["prefill"]["expected_route"])}


def _moe_row_binding(panel):
    return copy.deepcopy(panel["runtime_binding"])


def _consume_dense(*args, **kwargs):
    from .native_operator_panel import consume_native_receipt
    return consume_native_receipt(*args, **kwargs)


def _consume_moe(*args, **kwargs):
    from .native_moe_panel import consume_moe_receipt
    return consume_moe_receipt(*args, **kwargs)


@dataclass(frozen=True)
class NativePanelKind:
    schema: str
    structure: str
    runtime_schema: str
    raw_receipt_schema: str
    bound_schema: str
    operator_receipt_schema: str
    execution_projection: Callable
    row_binding: Callable
    consume: Callable
    routed: bool = False

    def world_size(self, panel):
        runtime = panel["runtime"]
        execution = runtime.get("execution") if isinstance(runtime, Mapping) else None
        if not isinstance(execution, Mapping) or "tensor_parallel" not in execution:
            raise RuntimePriceError("native runtime record names no execution world size")
        world = execution["tensor_parallel"]
        owner = panel.get("execution")
        if isinstance(owner, Mapping) and "tensor_parallel" in owner:
            if joint_identity_sha256(owner["tensor_parallel"]) != joint_identity_sha256(world):
                raise RuntimePriceError("native owner/runtime world size differs")
        return world

    def wire_records(self, panel):
        if self.routed:
            return [member["wire"]["record"] for member in panel["members"]]
        return [panel["wire"]["record"]]


PANEL_KINDS = MappingProxyType({
    DENSE_PANEL_SCHEMA: NativePanelKind(
        DENSE_PANEL_SCHEMA, "dense", "tessera.native_dense_runtime.v1",
        DENSE_EXECUTION_RECEIPT_SCHEMA, DENSE_BINDING_SCHEMA,
        "tessera.native_dense_operator_receipt.v1", _dense_execution_panel,
        _dense_row_binding, _consume_dense),
    MOE_PANEL_SCHEMA: NativePanelKind(
        MOE_PANEL_SCHEMA, "routed_moe", "tessera.native_moe_runtime.v1",
        MOE_EXECUTION_RECEIPT_SCHEMA, MOE_BINDING_SCHEMA,
        "tessera.native_moe_operator_receipt.v1", _moe_execution_panel,
        _moe_row_binding, _consume_moe, routed=True),
})


def panel_kind(panel) -> NativePanelKind:
    if not isinstance(panel, Mapping) or panel.get("schema") not in PANEL_KINDS:
        raise RuntimePriceError("unsupported native producer panel")
    return PANEL_KINDS[panel["schema"]]


def execution_panel_from_joint(panel):
    return panel_kind(panel).execution_projection(panel)


def bind_execution_receipt(raw_receipt, final_panel):
    """Join final cost coordinates without changing measured execution."""
    kind = panel_kind(final_panel)
    if (raw_receipt.get("schema") != kind.raw_receipt_schema
            or raw_receipt.get("status") != "timing_admissible"):
        raise ValueError("late binding requires numerically admitted execution receipt")
    if not kind.routed:
        from .native_execution_binding import require_reference_quantizer
        require_reference_quantizer(final_panel, final_panel["joint_operator_identity"]["activation"])
    panel = kind.execution_projection(final_panel)
    equal = (joint_identity_sha256(raw_receipt["panel"]) == joint_identity_sha256(panel)
             if kind.routed else raw_receipt["panel"] == panel)
    if not equal or raw_receipt["panel_sha256"] != joint_identity_sha256(panel):
        raise ValueError("late joint binding changes measured execution identity")
    return copy.deepcopy({"schema": kind.bound_schema,
        "raw_receipt": raw_receipt, "raw_receipt_sha256": joint_identity_sha256(raw_receipt),
        "panel": final_panel, "panel_sha256": joint_identity_sha256(final_panel)})


def resolve_execution_binding(binding, final_panel):
    kind = panel_kind(final_panel)
    if not kind.routed and binding.get("schema") != kind.bound_schema:
        raise ValueError("unsupported execution binding")
    raw = binding["raw_receipt"]
    if joint_identity_sha256(raw) != binding["raw_receipt_sha256"]:
        raise ValueError("raw receipt digest differs")
    if binding != bind_execution_receipt(raw, final_panel):
        raise ValueError("late execution binding differs from final panel")
    view = copy.deepcopy(raw)
    view.update(schema=kind.operator_receipt_schema,
                panel=copy.deepcopy(final_panel), panel_sha256=joint_identity_sha256(final_panel))
    return view


def resolve_native_receipt_view(receipt, panel):
    if receipt.get("schema") in {kind.bound_schema for kind in PANEL_KINDS.values()}:
        return resolve_execution_binding(receipt, panel)
    return receipt
