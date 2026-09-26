"""A GLM allocation keyed by source units reaches the producer's plan (#1388).

GLM-5.3-Flash roots its decoder under ``model.language_model.``, and the
glm5_next profile's recipe namespace folds that prefix to ``model.``.  The
allocation, Tessera's translator and its exporter all join on the SOURCE
names.  Three joins broke on the real allocation, one preflight at a time:

* the scope gate looked source-named units up in a recipe-keyed shape map and
  found no shape (``found []``);
* the plan view carried the allocator's BF16 spelling
  (``{"bits": 16, "data_type": "float"}``), which the translator refuses as a
  quantised non-Tessera choice;
* the plan view carried the vision tower's BF16 units, which the translator
  refuses as absent from its body projection.

This drives one layer-43-shaped glm5_next checkpoint through the lane CLI and
then through the pinned translator's own ``main``, so a fourth join of the
same kind fails here rather than on the fleet.
"""
from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest

from prismaquant import tessera_export_lane as export
from test_tessera_campaign_packed import _pinned_producer_checkout
from test_tessera_export_projection import _hessian_block, _isolate_other_gates
from test_tessera_pin_v38_scope import NEW_IMAGE, _Target

LAYER = "model.language_model.layers.43"
EXPERT = f"{LAYER}.mlp.experts.0"
#: Source shapes of GLM-5.3-Flash's layer-43 routed expert 0.
ROUTED = {f"{EXPERT}.gate_proj": (2048, 4096), f"{EXPERT}.up_proj": (2048, 4096),
          f"{EXPERT}.down_proj": (4096, 2048)}
#: A body Linear the allocator kept BF16, in the allocator's own spelling.
BODY_BF16 = f"{LAYER}.self_attn.o_proj"
#: A vision-tower Linear: outside the text graph the profile declares.
VISUAL = "model.visual.blocks.0.attn.proj"
PICK = "TESSERA_E4M3_K1_R896"
ALLOCATOR_BF16 = {"bits": 16, "data_type": "float", "group_size": None}


def _glm_case(tmp_path, *, visual_entry=ALLOCATOR_BF16):
    import torch
    from safetensors.torch import save_file

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"],
        # The one-expert population this fixture writes, at GLM's own widths.
        "text_config": {"n_routed_experts": 1, "hidden_size": 4096,
                        "moe_intermediate_size": 2048},
    }))
    tensors = {name + ".weight": torch.zeros(shape, dtype=torch.bfloat16)
               for name, shape in {**ROUTED, BODY_BF16: (256, 256), VISUAL: (64, 64)}.items()}
    save_file(tensors, str(model / "model.safetensors"))
    target = _Target().as_dict()
    payload = {name: {"data_type": "tessera", "bits": 4, "tessera_format": PICK}
               for name in ROUTED}
    payload[BODY_BF16] = dict(ALLOCATOR_BF16)
    payload[VISUAL] = visual_entry
    payload["__prismaquant__"] = {
        "tessera_serving_scope": {
            "target": target,
            "by_unit": {name: {**target, "structure": "routed_moe"} for name in ROUTED},
        },
        "tessera_hessian": _hessian_block(),
    }
    assignment = tmp_path / "layer_config.json"
    assignment.write_text(json.dumps(payload))
    return model, assignment


def _preflight(model, assignment, tmp_path):
    return export.main([
        "--model", str(model), "--assignment", str(assignment),
        "--write-build-json", str(tmp_path / "build.json"),
        "--tessera-platform", "sm_121", "--tessera-runtime-image", NEW_IMAGE,
        "--tessera-execution-mode", "eager", "--tessera-residency", "resident",
    ])


def test_glm_source_units_pass_preflight_and_the_pinned_translator(tmp_path, monkeypatch):
    checkout = _pinned_producer_checkout()
    if checkout is None:
        pytest.skip("the pinned Tessera checkout is not on this host")
    pytest.importorskip("tessera", reason="the translator imports the producer package")
    _isolate_other_gates(monkeypatch)
    model, assignment = _glm_case(tmp_path)
    original = assignment.read_bytes()

    assert _preflight(model, assignment, tmp_path) == 0
    build = json.loads((tmp_path / "build.json").read_text())
    view_path = Path(build["plan_assignment"])
    view = json.loads(view_path.read_text())
    assert assignment.read_bytes() == original
    assert view[BODY_BF16] == "BF16"
    assert VISUAL not in view
    assert view["__prismaquant__"]["tessera_export_assignment"][
        "source_precision_outside_graph"] == [VISUAL]

    translator = runpy.run_path(str(checkout / "experiments" / "plan_from_layer_config.py"))
    out = tmp_path / "plan.json"
    translator["main"]([str(view_path), str(model), str(out),
                        "--cover", "as-allocated", "--no-uniform-control"])
    plan = json.loads(out.read_text())
    # The routed expert units come back as their producer stack, at the pick.
    assert plan[f"{LAYER}.mlp.experts"] == {
        "grid": "E4M3", "q256": 896, "source_layout": "unpacked_per_expert"}, plan
    assert plan[BODY_BF16 + ".weight"] == "BF16"
    assert VISUAL + ".weight" not in plan


def test_a_quantised_choice_outside_the_text_graph_is_refused(tmp_path, monkeypatch):
    """The exporter writes the vision tower at source precision; a priced
    non-BF16 choice there would not be the written one."""
    _isolate_other_gates(monkeypatch)
    model, assignment = _glm_case(tmp_path, visual_entry={"bits": 8, "data_type": "fp8_e4m3",
                                                          "group_size": 0})
    with pytest.raises(export.TesseraExportLaneError, match="outside the text graph"):
        export.preflight(model, target=_Target(), assignment_path=assignment)
