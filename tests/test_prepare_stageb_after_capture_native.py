"""The Stage B preparation wrapper submits one native CPU job and refuses an unbound staged input (the retired wrapper's progress-phase test is gone with progress phases)."""
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prepare_stageb_after_capture as wrapper  # noqa: E402

DECLARATION = ["--produced-declaration", "/mnt/shared/m.submission/template.json", "--produced-declaration-sha256", "d" * 64]
STRICT = ["--data-manifest-sha256", "a" * 64, "--allowed-tiers", "ssd"]


def _args(tmp_path, binding_path):
    return SimpleNamespace(pair_json="pair.json", pair_sha="b" * 64, capture_json="capture.json", capture_sha="c" * 64,
                           metadata_root=tmp_path / "metadata", staged_binding=binding_path)


def _fake_run(tmp_path, calls):
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        submission = tmp_path / "metadata.submission"
        submission.mkdir(exist_ok=True)
        (submission / "submission.json").write_text(json.dumps({"strict_read_flags": STRICT, "native_argv": DECLARATION}))
        return subprocess.CompletedProcess(argv, 0)
    return run


def _environ(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir(exist_ok=True)
    return {"PQ_CHECKOUT": str(checkout), "PQ_PYTHON": "/venv/bin/python", "STAGEB_SPEC": "/mnt/shared/spec.json",
            "STAGEB_SPEC_SHA256": "e" * 64}


def test_the_job_argv_carries_the_preparation_strict_flags_and_declaration(tmp_path):
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps({"read_contract_path": str(tmp_path / "metadata.submission" / "read-manifest.json.gz")}))
    calls = []
    plan = wrapper.plan_job(_args(tmp_path, binding), _environ(tmp_path), run=_fake_run(tmp_path, calls))
    prepare_call = calls[0][0]
    assert prepare_call[:4] == ["/venv/bin/python", "-m", "tools.stage_b_preparation_submission", "prepare"]
    assert "--tier" not in prepare_call
    argv = plan["argv"]
    assert argv[:3] == ["/venv/bin/python", "-m", "tools.prepare_extended_joint_quanta"]
    assert argv[-len(STRICT):] == STRICT
    assert argv[argv.index("--produced-output"):][1:5] == DECLARATION
    assert "--spec" in argv and argv[argv.index("--spec") + 1] == "/mnt/shared/spec.json"


def test_a_staged_binding_for_another_read_manifest_refuses(tmp_path):
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps({"read_contract_path": "/mnt/shared/other/read-manifest.json.gz"}))
    with pytest.raises(ValueError, match="read manifest"):
        wrapper.plan_job(_args(tmp_path, binding), _environ(tmp_path), run=_fake_run(tmp_path, []))


def test_the_resources_are_a_native_cpu_profile():
    assert (wrapper.RESOURCES.gpus, wrapper.RESOURCES.cores, wrapper.RESOURCES.walltime_s) == (0, 4, 1800)
    assert wrapper.RESOURCES.memory_gib <= 16


def test_no_retired_submitter_or_progress_phase_remains():
    source = (ROOT / "tools" / "prepare_stageb_after_capture.py").read_text()
    assert re.search(r'--progress-(?:phase|grace-s)\b', source) is None
    assert '--tag' not in source
