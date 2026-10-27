"""Identity contracts for each native panel kind, without timing claims."""
import copy

import pytest

from prismaquant.joint_aura_rows import joint_identity_sha256
from prismaquant.measured_runtime_prices import RuntimePriceError
from prismaquant.native_panel_kind import (
    DENSE_PANEL_SCHEMA, MOE_PANEL_SCHEMA, bind_execution_receipt,
    execution_panel_from_joint, panel_kind, resolve_native_receipt_view,
)


@pytest.fixture(params=[DENSE_PANEL_SCHEMA, MOE_PANEL_SCHEMA])
def final_panel(request):
    member = {"unit": "q", "format": "f", "source_weight": {"sha": "source"},
              "rendered_weight": {"sha": "render"}, "activation": {"scale": None},
              "wire": {"record": {"sha": "wire"}}}
    route = {"symbol": "gemm", "policy": "f:resident"}
    panel = {"schema": request.param, "unit": "q", "format": "f", "shape": [2, 4],
             "cost_sha256": "cost", "probe_identity_sha256": "probe",
             "runtime": {"execution": {"tensor_parallel": 1}},
             "phases": {"prefill": {"expected_route": route}}}
    if request.param == DENSE_PANEL_SCHEMA:
        joint = {"qname": "q", **{key: member[key] for key in
                 ("format", "source_weight", "rendered_weight", "activation")}}
        panel.update(joint_operator_identity=joint,
                     joint_operator_identity_sha256=joint_identity_sha256(joint), wire=member["wire"])
    else:
        panel.update(members=[member], runtime_binding={
            "member_formats": {"q": "f"},
            "member_operator_identity_sha256": {"q": "joint"},
            "member_shapes": {"q": [2, 4]}, "operator_route": "route"})
    return panel


def test_each_kind_preserves_the_raw_receipt_and_rejects_measured_changes(final_panel):
    kind = panel_kind(final_panel)
    measured = execution_panel_from_joint(final_panel)
    raw = {"schema": kind.raw_receipt_schema, "status": "timing_admissible",
           "panel": measured, "panel_sha256": joint_identity_sha256(measured),
           "samples": [1.0, 2.0]}
    original = copy.deepcopy(raw)
    binding = bind_execution_receipt(raw, final_panel)
    view = resolve_native_receipt_view(binding, final_panel)
    assert raw == original == binding["raw_receipt"]
    assert view["schema"] == kind.operator_receipt_schema
    assert view["panel"] == final_panel
    assert view["samples"] == raw["samples"]
    changed = copy.deepcopy(final_panel)
    changed["phases"]["prefill"]["expected_route"]["symbol"] = "another_gemm"
    with pytest.raises(ValueError, match="changes measured"):
        bind_execution_receipt(raw, changed)


def test_each_kind_returns_independent_row_binding_and_validates_the_world(final_panel):
    kind = panel_kind(final_panel)
    before = copy.deepcopy(final_panel)
    binding = kind.row_binding(final_panel)
    assert binding["member_formats"] == {"q": "f"}
    binding["member_formats"]["q"] = "changed"
    assert final_panel == before
    assert kind.world_size(final_panel) == 1
    final_panel["execution"] = {"tensor_parallel": 2}
    with pytest.raises(RuntimePriceError, match="world size"):
        kind.world_size(final_panel)


def test_unknown_schema_cannot_select_a_consumer():
    with pytest.raises(RuntimePriceError, match="unsupported native producer panel"):
        panel_kind({"schema": "tessera.native_future_panel.v1"})
