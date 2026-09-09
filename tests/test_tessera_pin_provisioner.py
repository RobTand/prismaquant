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


def test_a_missing_interpreter_reports_no_installed_tessera():
    mod = _load()
    found, absent = mod.installed_identity(str(ROOT / "no-such-python"))
    assert found is None
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
    found, absent = mod.installed_identity(str(python))
    assert found is None
    assert "ModuleNotFoundError" in absent
    assert "tessera" in absent


def test_check_only_refuses_an_interpreter_that_is_not_on_the_pin(tmp_path):
    """Refusing is the useful answer: a green suite on stale bytes is the bug.

    Run as a subprocess against a synthetic pin rather than the repository's
    live one, because this asserts the RULE and the exit status a fleet audit
    reads, and asserting against today's pin would make the test move whenever
    the pin does.  The stale install this was written for carried the pin's own
    static version ``0.1.0``, so a version comparison would have called it
    correct.
    """
    import hashlib

    clone, old, new = _git_clone_with_two_commits(tmp_path)
    pin = _pin_module(tmp_path, new, hashlib.sha256(CONTRACT).hexdigest())
    python = _venv_with_tessera(tmp_path, "venv-stale", "VERSION = 1\n")

    completed = subprocess.run(
        [sys.executable, str(TOOL), "--python", str(python),
         "--pin-source", str(pin), "--clone", str(clone),
         "--pins-root", str(tmp_path / "pins"), "--check-only"],
        capture_output=True, text=True, check=False,
    )
    report = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert report["reviewed_commit"] == new
    assert report["action"] == "none, --check-only"
    # And it names the field that drifted, so the report is a repair order.
    assert report["drift"] == ["package_sha256"]
    assert report["installed_before"]["contract_sha256"] == \
        report["expected"]["contract_sha256"]


def test_the_provisioner_reads_the_live_pin_it_ships_with():
    """The synthetic pin above proves the rule; this proves the wiring.

    A tool that only ever ran against a test's own pin module could have the
    literal names wrong and nothing would say so.
    """
    mod = _load()
    commit, sha = mod.reviewed_pin()
    assert len(commit) == 40 and len(sha) == 64


def test_a_cached_tree_without_a_manifest_is_not_trusted(tmp_path):
    """This test used to assert the opposite, and the opposite was the bug.

    It said a directory named for a commit either holds that commit's tree or
    does not exist, so a marker file was enough to reuse it.  That is true of
    a content-addressed store and false of a directory: an interrupted ``tar``
    leaves a directory with the right name and some of the right files.  A
    cached tree is now reused only when it still digests to what its own
    manifest records, and one without a manifest is re-materialised.
    """
    mod = _load()
    commit = "0" * 40
    target = tmp_path / commit
    marker = target / mod.CONTRACT_IN_TREE
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        mod.materialise(commit, tmp_path / "absent-clone", tmp_path)
    assert "not a Tessera clone" in str(excinfo.value)


def test_a_verified_cached_tree_is_reused_without_a_clone(tmp_path):
    """And a tree that does verify is still reused, which is the point of it.

    A worker with no Tessera checkout has to be able to install from the
    shared path; requiring a clone every time would move the cost this cache
    exists to remove.
    """
    mod = _load()
    clone, old, new = _git_clone_with_two_commits(tmp_path)
    pins = tmp_path / "pins"
    first = mod.materialise(new, clone, pins)
    assert (first / mod.MANIFEST).exists()

    again = mod.materialise(new, tmp_path / "absent-clone", pins)
    assert again == first
    assert (again / "src" / "tessera" / "encode.py").read_text() == "VERSION = 2\n"


def test_materialise_refuses_when_it_has_no_clone_to_archive_from(tmp_path):
    mod = _load()
    with pytest.raises(SystemExit) as excinfo:
        mod.materialise("0" * 40, tmp_path / "absent-clone", tmp_path)
    assert "not a Tessera clone" in str(excinfo.value)


# --- the same-contract, different-source regression (root, PR461 review) -----
#
# The pin names a COMMIT.  ``runtime_contract.json`` is one file in that
# commit's tree, and two commits can publish identical contract bytes while
# differing everywhere else -- tessera#437 is exactly that: it rewrites the
# window encoder and leaves the contract alone.  A gate that compares only the
# contract therefore reports "already the reviewed bytes" for a re-pin whose
# whole point is new encoder code, and leaves the old encoder installed.  These
# tests were written red against that gate.


def _pin_module(tmp_path: Path, commit: str, contract_sha: str) -> Path:
    """A stand-in for ``prismaquant/tessera_runtime_contract.py``.

    The real one is the repository's live pin and moves with re-pins; a test
    that asserted against it would be asserting today's pin, not the rule.
    """
    p = tmp_path / "pin_module.py"
    p.write_text(
        f'TESSERA_DEV_PIN_COMMIT = "{commit}"\n'
        f'TESSERA_DEV_PIN_CONTRACT_SHA256 = "{contract_sha}"\n',
        encoding="utf-8")
    return p


CONTRACT = b'{"schema": "tessera.runtime_contract.v1", "executes": []}\n'


def _tessera_tree(root: Path, encoder_body: str) -> None:
    """A minimal tree shaped like Tessera's: contract under the package."""
    pkg = root / "src" / "tessera" / "serving"
    pkg.mkdir(parents=True, exist_ok=True)
    (root / "src" / "tessera" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "runtime_contract.json").write_bytes(CONTRACT)
    (root / "src" / "tessera" / "encode.py").write_text(encoder_body,
                                                        encoding="utf-8")


def _git_clone_with_two_commits(tmp_path: Path) -> tuple[Path, str, str]:
    """A clone whose second commit changes the encoder and not the contract."""
    clone = tmp_path / "tessera-clone"
    clone.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(clone), *a], check=True,
                                    capture_output=True)
    subprocess.run(["git", "init", "-q", str(clone)], check=True,
                   capture_output=True)
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "t")
    _tessera_tree(clone, "VERSION = 1\n")
    run("add", "-A")
    run("commit", "-q", "-m", "old encoder")
    old = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"],
                         check=True, capture_output=True,
                         text=True).stdout.strip()
    _tessera_tree(clone, "VERSION = 2\n")          # contract byte-identical
    run("add", "-A")
    run("commit", "-q", "-m", "new encoder, same contract")
    new = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"],
                         check=True, capture_output=True,
                         text=True).stdout.strip()
    return clone, old, new


def _venv_with_tessera(tmp_path: Path, name: str, encoder_body: str) -> Path:
    """A real interpreter carrying a Tessera-shaped package in site-packages.

    Placed rather than pip-installed: the regression is about what the tool
    DECIDES, and a build backend in the loop would add a network dependency
    and a second failure mode without adding evidence.
    """
    import sysconfig
    import venv

    env = tmp_path / name
    venv.create(env, with_pip=False)
    site = env / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if not site.exists():                       # non-posix_prefix layouts
        site = Path(sysconfig.get_paths()["purelib"].replace(
            sys.prefix, str(env)))
        site.mkdir(parents=True, exist_ok=True)
    _tessera_tree(env / "staging", encoder_body)
    import shutil
    shutil.copytree(env / "staging" / "src" / "tessera", site / "tessera")
    return env / "bin" / "python"


def test_same_contract_with_older_source_is_not_already_the_reviewed_bytes(tmp_path):
    """The regression root found: a re-pin that only moves code is a no-op.

    The interpreter carries the OLD encoder under the SAME contract bytes the
    new pin reviews.  A contract-only gate says there is nothing to do; there
    is, and it is the entire content of the re-pin.
    """
    mod = _load()
    clone, old, new = _git_clone_with_two_commits(tmp_path)
    import hashlib
    contract_sha = hashlib.sha256(CONTRACT).hexdigest()
    pin = _pin_module(tmp_path, new, contract_sha)
    python = _venv_with_tessera(tmp_path, "venv-old", "VERSION = 1\n")

    rc = mod.main(["--python", str(python), "--pin-source", str(pin),
                   "--clone", str(clone), "--pins-root", str(tmp_path / "pins"),
                   "--check-only"])
    assert rc != 0, "an interpreter on the wrong source must not report clean"


def test_the_matching_source_is_reported_clean(tmp_path):
    """And the other direction, so the check above is not vacuously red."""
    mod = _load()
    clone, old, new = _git_clone_with_two_commits(tmp_path)
    import hashlib
    pin = _pin_module(tmp_path, new, hashlib.sha256(CONTRACT).hexdigest())
    python = _venv_with_tessera(tmp_path, "venv-new", "VERSION = 2\n")

    rc = mod.main(["--python", str(python), "--pin-source", str(pin),
                   "--clone", str(clone), "--pins-root", str(tmp_path / "pins"),
                   "--check-only"])
    assert rc == 0


def test_a_truncated_cached_tree_is_not_reused(tmp_path):
    """``materialise`` trusted a directory name and one marker file.

    A tree left half-extracted by an interrupted tar, or edited afterwards,
    still has ``runtime_contract.json`` and still has the commit's name on it.
    Reusing it installs something that is not the commit, under the commit's
    label, which is the failure the pin exists to prevent.
    """
    mod = _load()
    clone, old, new = _git_clone_with_two_commits(tmp_path)
    pins = tmp_path / "pins"

    first = mod.materialise(new, clone, pins)
    assert (first / "src" / "tessera" / "encode.py").read_text() == "VERSION = 2\n"

    (first / "src" / "tessera" / "encode.py").unlink()          # truncated
    second = mod.materialise(new, clone, pins)
    assert (second / "src" / "tessera" / "encode.py").read_text() == "VERSION = 2\n", \
        "a cached tree missing a file must be repaired, not reused"


def test_a_tampered_cached_tree_is_not_reused(tmp_path):
    """Same rule for a tree whose bytes were changed rather than removed."""
    mod = _load()
    clone, old, new = _git_clone_with_two_commits(tmp_path)
    pins = tmp_path / "pins"

    first = mod.materialise(new, clone, pins)
    (first / "src" / "tessera" / "encode.py").write_text("VERSION = 99\n")
    second = mod.materialise(new, clone, pins)
    assert (second / "src" / "tessera" / "encode.py").read_text() == "VERSION = 2\n", \
        "a cached tree whose bytes moved must be repaired, not reused"
