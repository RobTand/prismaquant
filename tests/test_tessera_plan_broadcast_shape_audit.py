from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import tessera_plan_writer as writer


@pytest.mark.parametrize("prefix", ["model.layers", "model.language_model.layers"])
@pytest.mark.parametrize("target_columns", [64, 96], ids=["equal_shape", "unequal_shape"])
def test_broadcast_preserves_actual_priced_source_shape(tmp_path, prefix, target_columns):
    model = tmp_path / "model"
    model.mkdir()
    source = f"{prefix}.0.mlp.down_proj"
    target = f"{prefix}.1.mlp.down_proj"
    save_file({
        source + ".weight": torch.zeros((32, 64), dtype=torch.bfloat16),
        target + ".weight": torch.zeros((32, target_columns), dtype=torch.bfloat16),
    }, str(model / "model.safetensors"))
    (model / "config.json").write_text(json.dumps({
        "architectures": ["Glm5NextForConditionalGeneration"],
        "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
    }))
    assignment = tmp_path / "layer_config.json"
    assignment.write_text(json.dumps({source: {"tessera_format": "TESSERA_E4M3_K1_R1024"}}))
    output = tmp_path / "plan.json"
    argv = [str(assignment), str(model), str(output),
            "--cover", "broadcast-by-role", "--no-uniform-control"]

    if target_columns != 64:
        with pytest.raises(writer.PlanError, match="priced.*different rate"):
            writer.main(argv)
        assert not output.exists()
        return

    assert writer.main(argv) == 0
    plan = json.loads(output.read_text())
    assert plan == {
        "schema": "tessera.serving_plan.v1",
        source + ".weight": {"grid": "E4M3", "q256": 1024},
        target + ".weight": {"grid": "E4M3", "q256": 1024},
    }
    provenance = json.loads(output.with_suffix(".json.provenance.json").read_text())
    assert provenance["coverage"]["extrapolated"] is True
    assert provenance["coverage"]["broadcast_from_layer"] == 0
    assert provenance["coverage"]["unplanned_body_linears"] == 0
    assert {row["tensor"]: (row["rows"], row["columns"]) for row in provenance["units"]} == {
        source + ".weight": (32, 64), target + ".weight": (32, 64),
    }
