"""The #753 pre-submit guard pins pbtest's provenance requirement, locally.

Hermetic: every interpreter-facing path runs through a fake ``python`` (a
shell script printing canned probe JSON), so no test depends on which
interpreters a box happens to provision.
"""
from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "check_tessera_dev_install.py"
RESOLVER = ROOT / "tools" / "resolve_tessera_dev_pin.py"


def _load():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import check_tessera_dev_install as mod
    finally:
        sys.path.pop(0)
    return mod


def test_the_guard_reads_the_commit_pbtest_will_demand():
    """A copied SHA becomes a second pin; the guard runs the resolver."""
    source = TOOL.read_text(encoding="utf-8")
    assert "tools/resolve_tessera_dev_pin.py" in source or "PIN_RESOLVER" in source
    mod = _load()
    printed = subprocess.run(
        [sys.executable, str(RESOLVER)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert mod.reviewed_commit() == printed


def test_the_guard_needs_neither_prismaquant_nor_tessera():
    """It runs on interpreters that have neither; that is the point of it."""
    source = TOOL.read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert not stripped.startswith(("import prismaquant", "from prismaquant"))
            assert not stripped.startswith(("import tessera", "from tessera"))


def _fake_python(tmp_path, payload):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\nprintf '%s' '" + payload.replace("'", "'\\''") + "'\n",
                    encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(fake)


def _cases(expected):
    return {
        "local_dir": ({}, "local-directory install"),
        "malformed": (["not", "an", "object"], "is not an object"),
        "wrong_commit": ({"vcs_info": {"vcs": "git", "commit_id": "0" * 40}},
                         "expected " + expected),
        "editable": ({"vcs_info": {"vcs": "git", "commit_id": expected},
                      "dir_info": {"editable": True}}, "editable install"),
    }


@pytest.mark.parametrize("case", ["local_dir", "malformed", "wrong_commit", "editable"])
def test_non_git_provenance_refuses_with_the_idle_window_repair(tmp_path, case):
    """#753: every provenance the gate refuses, refused here first, with the
    repair the filer may not run unilaterally (idle pool, fleet owner)."""
    mod = _load()
    direct_url, detail = _cases(mod.reviewed_commit())[case]
    payload = json.dumps({"distribution": "tessera-quant", "direct_url": direct_url})
    fake = _fake_python(tmp_path, payload)
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--python", fake],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 1, completed.stdout
    assert detail in completed.stdout
    assert "tools/provision_tessera_pin.py --python" in completed.stdout
    assert "idle" in completed.stdout
    assert "RobTand/prismabuild#658" in completed.stdout
    assert "RobTand/prismaquant#753" in completed.stdout


def test_the_reviewed_commit_passes(tmp_path):
    mod = _load()
    payload = json.dumps({"distribution": "tessera-quant", "direct_url": {
        "vcs_info": {"vcs": "git", "commit_id": mod.reviewed_commit()}}})
    fake = _fake_python(tmp_path, payload)
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--python", fake],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert "non-editable Git install at the reviewed commit" in completed.stdout


def test_a_missing_interpreter_is_a_refusal_not_a_tool_failure(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--python", str(tmp_path / "no-such-python")],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 1
    assert "tools/provision_tessera_pin.py --python" in completed.stdout
