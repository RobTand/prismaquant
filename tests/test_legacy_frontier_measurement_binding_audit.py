"""Baseline draft for RE-07; authored only, never executed by this worker."""
from __future__ import annotations

import json

import pytest

from prismaquant.footprint import assignment_serialization_sha256
from prismaquant.select_validated_frontier import main


@pytest.mark.parametrize("replacement", ["NVFP4", "BF16"])
def test_legacy_frontier_binds_output_to_measured_assignment(tmp_path, replacement):
    qname = "model.layers.0.self_attn.q_proj"
    measured_assignment = {qname: "NVFP4"}
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"assignment": {qname: replacement}}))
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({
        "results": [{
            "label": "legacy",
            "path": str(candidate),
            "bpp": 4.0,
            "last_token_kl": 0.01,
            "assignment_sha256": assignment_serialization_sha256(
                measured_assignment),
        }],
    }))
    output_assignment = tmp_path / "selected_assignment.json"
    argv = [
        "--validation-json", str(validation),
        "--mode", "best-kl",
        "--output-layer-config", str(tmp_path / "layer_config.json"),
        "--output-assignment", str(output_assignment),
        "--output-summary", str(tmp_path / "summary.json"),
    ]

    if replacement != "NVFP4":
        with pytest.raises(ValueError, match="assignment"):
            main(argv)
        assert not output_assignment.exists()
    else:
        assert main(argv) == 0
        emitted = json.loads(output_assignment.read_text())
        assert emitted["assignment"] == measured_assignment
