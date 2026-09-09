"""The fleet provisioner reads the same pin CI does, and checks bytes not versions."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "provision_tessera_pin.py"
RESOLVER = ROOT / "tools" / "resolve_tessera_dev_pin.py"


def _load():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import provision_tessera_pin as mod
    finally:
        sys.path.pop(0)
    return mod


def test_the_provisioner_reads_the_commit_the_ci_resolver_prints():
    """Two readers of one pin is how the fleet came to sit behind CI."""
    mod = _load()
    commit, sha = mod.reviewed_pin()
    printed = subprocess.run(
        [sys.executable, str(RESOLVER)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert commit == printed
    assert len(sha) == 64


def test_the_provisioner_needs_neither_prismaquant_nor_tessera():
    """It runs on an interpreter that has neither; that is the point of it.

    ``prismaquant`` pulls ``compressed_tensors`` and ``tessera`` is the thing
    being installed, so importing either here would make the tool unusable on
    exactly the interpreter it exists to repair.
    """
    source = TOOL.read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert not stripped.startswith(("import prismaquant", "from prismaquant"))
            assert not stripped.startswith(("import tessera", "from tessera"))


def test_a_missing_interpreter_reports_no_installed_contract():
    mod = _load()
    sha, absent = mod.installed_contract(str(ROOT / "no-such-python"))
    assert sha is None
    # And it says which of the three absences this is.  A bare ``None`` reads
    # as "no venv", "no Tessera" and "no packaged contract" at once, and those
    # want different repairs.
    assert "FileNotFoundError" in absent


def test_an_interpreter_without_tessera_names_the_import_that_failed(tmp_path):
    """The GB10 venv's first audit returned an empty contract and no reason.

    It was not a missing venv and not a Tessera built from the wrong commit:
    Tessera is not installed in it at all, and the report said only ``null``.
    The reason travels now, so the next audit reads a repair instead of a gap.

    The interpreter here is a real one that really lacks Tessera, built by
    ``venv`` in the test's own directory, rather than the running interpreter
    under a skip.  A test that skips wherever the suite actually runs proves
    nothing about the branch it names.
    """
    import venv

    env = tmp_path / "bare"
    venv.create(env, with_pip=False)
    python = env / "bin" / "python"
    assert python.exists()

    mod = _load()
    sha, absent = mod.installed_contract(str(python))
    assert sha is None
    assert "ModuleNotFoundError" in absent
    assert "tessera" in absent


def test_check_only_refuses_an_interpreter_that_is_not_on_the_pin(tmp_path):
    """Refusing is the useful answer: a green suite on stale bytes is the bug.

    The stale install carried the pin's own static version ``0.1.0``, so a
    version comparison would have called it correct.  Only the contract bytes
    tell the two apart, and this exit status is what a provisioning check in
    CI or a fleet audit would read.
    """
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--python", sys.executable, "--check-only"],
        capture_output=True, text=True, check=False,
    )
    report = json.loads(completed.stdout)
    mod = _load()
    commit, sha = mod.reviewed_pin()
    assert report["reviewed_commit"] == commit
    assert report["reviewed_contract_sha256"] == sha
    if report["installed_contract_sha256_before"] == sha:
        assert completed.returncode == 0
        assert report["action"] == "none, already the reviewed bytes"
    else:
        assert completed.returncode == 1
        assert report["action"] == "none, --check-only"


def test_materialise_reuses_a_tree_that_is_already_laid_down(tmp_path):
    """A directory named for a commit either holds it or does not exist.

    So a second run must not re-archive, and must not need a clone at all --
    which is what makes the shared path usable from a worker with no Tessera
    checkout of its own.
    """
    mod = _load()
    commit = "0" * 40
    target = tmp_path / commit
    marker = target / mod.CONTRACT_IN_TREE
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    assert mod.materialise(commit, tmp_path / "absent-clone", tmp_path) == target


def test_materialise_refuses_when_it_has_no_clone_to_archive_from(tmp_path):
    mod = _load()
    with pytest.raises(SystemExit) as excinfo:
        mod.materialise("0" * 40, tmp_path / "absent-clone", tmp_path)
    assert "not a Tessera clone" in str(excinfo.value)
