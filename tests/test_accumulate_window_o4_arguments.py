"""CPU-only CLI-shape coverage for the O4 recipe, without running capture phases."""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import shlex

import pytest

from test_tessera_calibration_cache import capture as capture


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


@pytest.mark.parametrize("consumer", ["dispatch", "final-plan"])
def test_o4_documented_capture_bindings(consumer, capture, monkeypatch, tmp_path):
    from prismaquant import tessera_calibration_cache
    from tools import dispatch_tessera_campaign

    _, census_path, census, identity, acts, hessians, _ = capture
    receipt = tessera_calibration_cache.publish_capture(
        tmp_path / "mtp-capture", census_path=census_path, identity=identity,
        acts=acts, hessians=hessians, counts=census["counts"], maxima=census["max_abs"],
    )
    monkeypatch.setenv("O4_ROOT", str(tmp_path))
    monkeypatch.setenv("WORKSPACE", str(tmp_path / "anchor campaign"))
    monkeypatch.setenv("CAMPAIGN_SPEC", str(tmp_path / "campaign spec.json"))
    recipe = (Path(__file__).resolve().parents[1] / "docs" / "experiments"
              / "accumulate_window_o4_2896.md").read_text(encoding="utf-8")

    if consumer == "dispatch":
        prefix = "python -m tools.dispatch_tessera_campaign "
        commands = [
            line[len(prefix):] for line in recipe.replace("\\\n", " ").splitlines()
            if line.startswith(prefix + "plan ")
        ]
        assert len(commands) == 1
        parsed = []
        monkeypatch.setattr(dispatch_tessera_campaign, "cmd_plan", parsed.append)
        dispatch_tessera_campaign.main([
            os.path.expandvars(arg) for arg in shlex.split(commands[0])
        ])
        assert len(parsed) == 1
        binding = Path(parsed[0].calibration_cache)
        assert binding == Path(receipt["path"])
        assert hashlib.sha256(binding.read_bytes()).hexdigest() == receipt["sha256"]
    else:
        snippets = recipe.split("python - <<'PY'\n")[1:]
        constructions = [snippet.split("\nPY", 1)[0] for snippet in snippets
                         if "p['canonical_capture'] =" in snippet.split("\nPY", 1)[0]]
        assert len(constructions) == 1
        construction = ast.parse(constructions[0])
        nodes = [
            node for node in construction.body
            if (isinstance(node, ast.FunctionDef) and node.name == "bound")
            or (isinstance(node, ast.Assign)
                and ast.unparse(node.targets[0]) == "p['canonical_capture']")
        ]
        assert len(nodes) == 2
        plan = {}
        namespace = {"Path": Path, "hashlib": hashlib, "root": tmp_path, "p": plan}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<O4 final-plan>", "exec"),
             namespace)
        assert plan["canonical_capture"] == receipt
