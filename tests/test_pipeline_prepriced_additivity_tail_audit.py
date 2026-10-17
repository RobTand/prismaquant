"""External prepriced bytes stay immutable through the real report tail.

The shipped path-selection and AURA report sections execute unchanged with
the real prepriced validator and additivity CLI. Synthetic zero-KL input
records avoid model/GPU work; they establish no measured quality claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from test_run_pipeline_prepriced_override import SCRIPT, _input, _sections


def _run_tail(tmp_path, *, external, gate="measure"):
    supplied, model = _input(tmp_path, "aura")
    before = supplied.read_bytes()
    work = tmp_path / "work"
    artifacts = work / "artifacts"
    artifacts.mkdir(parents=True)
    (work / "logs").mkdir()
    assignment = {"model.layers.0.self_attn.o_proj": "BF16"}
    (artifacts / "layer_config.json").write_text(json.dumps(assignment))
    (artifacts / "validated_frontier_kl.json").write_text(json.dumps({
        "results": [{"label": "selected", "kl_mean": 0.0, "kl_stderr": 0.0}],
    }))
    (artifacts / "validated_frontier_selection.json").write_text(json.dumps({
        "selected": {"label": "selected"},
    }))
    owned = artifacts / "cost.pkl"
    if not external:
        shutil.copyfile(supplied, owned)
    paths, _costs, _allocator = _sections()
    text = SCRIPT.read_text()
    start = text.index("# [3c] AURA additivity report")
    tail = text[start:text.index("# [3d] Discovery-walker export gate", start)]
    python = shlex.quote(sys.executable)
    preamble = f"""set -euo pipefail
python3() {{ command {python} "$@"; }}
"""
    env = os.environ.copy()
    env.update({
        "WORK_DIR": str(work), "MODEL_PATH": str(model), "COST_MODE": "aura",
        "COST_PATH_OVERRIDE": str(supplied) if external else "",
        "SELECTION_MODE": "surrogate", "AURA_ADDITIVITY_GATE": gate,
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
    })
    result = subprocess.run(
        ["bash", "-c", preamble + paths + tail], cwd=SCRIPT.parents[1],
        env=env, text=True, capture_output=True, timeout=180, check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads((artifacts / "layer_config.json").read_text()) == assignment
    return supplied, before, owned, artifacts


@pytest.mark.parametrize("gate", ["measure", "auto"])
def test_external_prepriced_input_is_not_rewritten_by_later_report(tmp_path, gate):
    supplied, before, _owned, artifacts = _run_tail(tmp_path, external=True, gate=gate)
    report = json.loads((artifacts / "aura_additivity.json").read_text())
    assert report["schema"] == "prismaquant.aura_additivity_gate.v1"
    assert report["n_zero_cost"] == 1
    assert report["measured_kl"] == report["predicted_sum"] == report["residual"] == 0.0
    receipt = json.loads((artifacts / "prepriced_cost_input.json").read_text())
    assert receipt["input_path"] == str(supplied)

    assert supplied.read_bytes() == before, "AURA report tail rewrote external caller cost bytes"


def test_run_owned_cost_keeps_existing_additivity_provenance_stamp(tmp_path):
    supplied, before, owned, artifacts = _run_tail(tmp_path, external=False)
    import pickle

    payload = pickle.loads(owned.read_bytes())
    assert owned.read_bytes() != before
    assert payload["provenance"]["additivity"]["measured_kl"] == 0.0
    assert payload["provenance"]["additivity"]["measured_kl_source"] == "validated_frontier_kl.json"
    assert (artifacts / "aura_additivity.json").is_file()
    assert supplied.read_bytes() == before


def test_disabled_additivity_keeps_external_input_and_writes_no_report(tmp_path):
    supplied, before, _owned, artifacts = _run_tail(tmp_path, external=True, gate="0")

    assert supplied.read_bytes() == before
    assert not (artifacts / "aura_additivity.json").exists()
