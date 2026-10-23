"""A fresh pin check cannot qualify cells read from another document."""
from prismaquant.tessera_runtime_contract import installed_runtime_snapshot

import json
import subprocess
import sys

import pytest
from pathlib import Path

from prismaquant import tessera_runtime_contract as runtime


@pytest.mark.parametrize("replace_during_capture", [False, True])
def test_restoring_pinned_bytes_does_not_admit_cached_unreviewed_cells(
        tmp_path, replace_during_capture):
    payload = json.loads(runtime.contract_path().read_bytes())
    reference = next(cell for cell in payload["lane_eligibility"]["cells"]
                     if cell["family"] == "TESSERA_E2M1_K2"
                     and cell["structure"] == "dense"
                     and cell["regime"] == "decode")
    original_image = reference["runtime"]["image"]
    unreviewed_image = "example/unreviewed@sha256:" + "a" * 64
    for cell in payload["lane_eligibility"]["cells"]:
        if cell["runtime"]["image"] == original_image:
            cell["runtime"]["image"] = unreviewed_image
    altered = tmp_path / "runtime_contract.json"
    altered.write_text(json.dumps(payload))
    name = reference["family"] + "_R" + str(reference["rungs_q256"][0])
    context = {
        "platform": reference["platform"], "structure": reference["structure"],
        "residency": "resident", "runtime_image": unreviewed_image,
        "execution_mode": "eager",
    }
    script = """
import json, sys
from pathlib import Path
from prismaquant import tessera_render as render, tessera_runtime_contract as runtime
from prismaquant.lane_eligibility import ServingContext
from prismaquant.tessera_serving_runtime_pin import require_pinned_tessera_runtime
canonical = runtime.contract_path()
runtime.contract_path = lambda: Path(sys.argv[1])
context = ServingContext(**json.loads(sys.argv[3]))
if sys.argv[4] == 'True':
    read_bytes = Path.read_bytes

    def replace_after_capture(path):
        raw = read_bytes(path)
        if path == Path(sys.argv[1]):
            runtime.contract_path = lambda: canonical
        return raw
    Path.read_bytes = replace_after_capture
    warmed = None
else:
    warmed = render.tessera_attesting_cells(sys.argv[2], serving_context=context)
    runtime.contract_path = lambda: canonical
attested = render.tessera_lane_admission(sys.argv[2], serving_context=context).admitted
require_pinned_tessera_runtime()
print(json.dumps({'warmed_cells': len(warmed) if warmed is not None else None, 'attested': attested}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(altered), name, json.dumps(context),
         str(replace_during_capture)],
        cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    observed = json.loads(result.stdout)
    if not replace_during_capture:
        assert observed["warmed_cells"] > 0
    assert observed["attested"] is False
