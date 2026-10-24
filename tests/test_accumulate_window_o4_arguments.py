"""CPU-only CLI-shape coverage for the O4 recipe, without running capture phases."""
from __future__ import annotations

import os
from pathlib import Path
import shlex


def test_o4_documented_mtp_phase_arguments(monkeypatch, tmp_path):
    from tools.glm_mtp_capture import _capture_arguments

    bindings = {
        "BODY_PLAN": str(tmp_path / "body plan.json"),
        "BODY_PLAN_SHA": "a" * 64,
        "BODY_PREPARED": str(tmp_path / "body prepared.json"),
        "BODY_PREPARED_SHA": "b" * 64,
        "BOUNDARIES": str(tmp_path / "boundaries.json"),
        "BOUNDARIES_SHA": "c" * 64,
        "MTP_OFFLOAD_FOLDER": str(tmp_path / "offload scratch"),
        "O4_ROOT": str(tmp_path / "o4 output"),
    }
    for name, value in bindings.items():
        monkeypatch.setenv(name, value)

    recipe = (Path(__file__).resolve().parents[1] / "docs" / "experiments"
              / "accumulate_window_o4_2896.md").read_text(encoding="utf-8")
    prefix = "python -m tools.glm_mtp_capture "
    phases = {}
    for line in recipe.replace("\\\n", " ").splitlines():
        if not line.startswith(prefix):
            continue
        # Tokenize the actual documented command; never execute shell hash
        # substitutions, read artifacts, or call a model/GPU phase handler.
        argv = [os.path.expandvars(arg) for arg in shlex.split(line[len(prefix):])]
        args = _capture_arguments(argv)
        assert args.phase not in phases
        phases[args.phase] = args
        assert args.plan == bindings["BODY_PLAN"]
        assert args.plan_sha256 == bindings["BODY_PLAN_SHA"]

    assert set(phases) == {"projection", "final-hidden", "capture"}
    root = Path(bindings["O4_ROOT"])
    assert phases["projection"].out == str(root / "mtp-projection")
    hidden = phases["final-hidden"]
    assert hidden.out == str(root / "final-hidden")
    assert hidden.boundaries == bindings["BOUNDARIES"]
    assert hidden.boundaries_sha256 == bindings["BOUNDARIES_SHA"]
    assert hidden.offload_folder == bindings["MTP_OFFLOAD_FOLDER"]
    capture = phases["capture"]
    assert capture.out == str(root / "mtp-capture")
    assert capture.prepared == bindings["BODY_PREPARED"]
    assert capture.prepared_sha256 == bindings["BODY_PREPARED_SHA"]
    assert capture.final_hidden == str(root / "final-hidden" / "manifest.json")
    assert capture.expert_projection == str(root / "mtp-projection" / "mtp-projection.json")
    assert capture.census_out == str(root / "mtp-census.json")
