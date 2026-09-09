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
    # --check-only reads a source that already verifies and writes none, so
    # the source is published first, the way a provisioning run leaves it.
    _load().materialise(new, clone, tmp_path / "pins")

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


#: The real tree's build config excludes its own tooling subpackage from what
#: a consumer installs, so a source tree has files an install will never have.
#: The fixture carries that shape because the first version did not, and a
#: fixture that copies everything into site-packages cannot see a digest which
#: compares a source tree against an install.
PYPROJECT = """\
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "tessera-quant"
version = "0.1.0"

[tool.setuptools.packages.find]
where = ["src"]
exclude = ["tessera._dev*"]

[tool.setuptools.package-data]
"tessera.serving" = ["runtime_contract.json"]
"""

#: What a wheel leaves out of ``src/tessera``, as relative paths.
NOT_INSTALLED = ("_dev",)


def _tessera_tree(root: Path, encoder_body: str) -> None:
    """A minimal tree shaped like Tessera's: contract under the package.

    Including the halves that make source and install differ: a build config
    that excludes ``tessera._dev``, and a ``_dev`` subpackage for it to
    exclude.
    """
    pkg = root / "src" / "tessera" / "serving"
    pkg.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (root / "src" / "tessera" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "runtime_contract.json").write_bytes(CONTRACT)
    (root / "src" / "tessera" / "encode.py").write_text(encoder_body,
                                                        encoding="utf-8")
    dev = root / "src" / "tessera" / "_dev"
    dev.mkdir(parents=True, exist_ok=True)
    (dev / "__init__.py").write_text("", encoding="utf-8")
    (dev / "tooling.py").write_text("# repository tooling, not shipped\n",
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
    # A wheel does not carry what the build config excludes.  Copying the
    # whole source directory is the shape that hid the defect below.
    for rel in NOT_INSTALLED:
        shutil.rmtree(site / "tessera" / rel)
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
    mod.materialise(new, clone, tmp_path / "pins")   # --check-only only reads

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
    mod.materialise(new, clone, tmp_path / "pins")

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


def test_a_subpackage_the_build_excludes_is_not_expected_in_the_install(tmp_path):
    """A digest over the whole source tree can never match a real install.

    The pinned tree's own build config excludes ``tessera._dev`` from what a
    consumer installs, so the source has files a correct install will never
    have.  Comparing the two directly reports drift on a correctly provisioned
    interpreter, forever: install, re-probe, still different, exit 1.  The
    earlier fixture copied the whole source directory into site-packages and
    so could not see this at all.

    What the digest has to compare is the source MINUS what the build config
    says it does not ship, which is read from the pinned tree rather than
    known here.
    """
    mod = _load()
    clone, old, new = _git_clone_with_two_commits(tmp_path)
    import hashlib
    pin = _pin_module(tmp_path, new, hashlib.sha256(CONTRACT).hexdigest())
    python = _venv_with_tessera(tmp_path, "venv-wheel-shaped", "VERSION = 2\n")
    mod.materialise(new, clone, tmp_path / "pins")

    rc = mod.main(["--python", str(python), "--pin-source", str(pin),
                   "--clone", str(clone), "--pins-root", str(tmp_path / "pins"),
                   "--check-only"])
    assert rc == 0, (
        "an interpreter carrying exactly what a wheel of the pin installs is "
        "on the pin; a digest that counts unshipped source files says it never is"
    )


def test_check_only_does_not_write_into_the_pins_root(tmp_path):
    """``--check-only`` reports; it does not repair.

    The pins root is shared across the fleet and the flag's whole contract is
    that it changes nothing.  Materialising to learn the expected digest is a
    write, and an operator who ran a read-only check to see where a box stood
    would find a tree laid down by it.
    """
    mod = _load()
    clone, old, new = _git_clone_with_two_commits(tmp_path)
    import hashlib
    pin = _pin_module(tmp_path, new, hashlib.sha256(CONTRACT).hexdigest())
    python = _venv_with_tessera(tmp_path, "venv-ro", "VERSION = 2\n")
    pins = tmp_path / "pins"
    pins.mkdir()

    rc = mod.main(["--python", str(python), "--pin-source", str(pin),
                   "--clone", str(clone), "--pins-root", str(pins),
                   "--check-only"])

    assert list(pins.iterdir()) == [], (
        "--check-only materialised the pinned tree into the shared pins root")
    assert rc != 0, (
        "with no verifiable source to compare against, a read-only check "
        "cannot report clean")


def test_a_cached_tree_that_fails_its_manifest_is_kept_not_deleted(tmp_path):
    """A tree that does not verify is evidence, and evidence is not deleted.

    Whatever edited or truncated it is unexplained, and the bytes are the only
    record of what happened.  Replacing the tree is right; removing the thing
    that would say why is not, and on a shared pins root it removes it for
    everyone.
    """
    mod = _load()
    clone, old, new = _git_clone_with_two_commits(tmp_path)
    pins = tmp_path / "pins"

    first = mod.materialise(new, clone, pins)
    (first / "src" / "tessera" / "encode.py").write_text("VERSION = 99\n")

    second = mod.materialise(new, clone, pins)
    assert (second / "src" / "tessera" / "encode.py").read_text() == "VERSION = 2\n"

    kept = [p for p in pins.rglob("encode.py")
            if p.read_text() == "VERSION = 99\n"]
    assert kept, (
        "the tree that failed its manifest was deleted; it is the only record "
        "of what wrote to a shared pins root")


def test_an_unmanifested_tree_matching_the_commit_is_attested_not_replaced(tmp_path):
    """Bytes that already are the commit do not need to be written again.

    ``/mnt/shared/tessera-pins/07ad344c...`` predates the manifest and is
    correct.  Rewriting it churns a shared directory other boxes may be
    reading, for a tree that is already right; verifying it against a fresh
    ``git archive`` establishes the same thing without touching it.
    """
    mod = _load()
    clone, old, new = _git_clone_with_two_commits(tmp_path)
    pins = tmp_path / "pins"

    laid = mod.materialise(new, clone, pins)
    encode = laid / "src" / "tessera" / "encode.py"
    before = encode.stat().st_ino
    (laid / mod.MANIFEST).unlink()               # a pre-manifest tree

    again = mod.materialise(new, clone, pins)
    assert again == laid
    assert encode.read_text() == "VERSION = 2\n"
    assert encode.stat().st_ino == before, (
        "a tree that already holds the commit's bytes was rewritten rather "
        "than verified and attested")
    assert (laid / mod.MANIFEST).exists(), "and the manifest was not written"


def test_the_install_does_not_run_inside_the_frozen_tree(tmp_path, monkeypatch):
    """``pip install <dir>`` writes ``*.egg-info`` into the directory it builds.

    Pointed at the pinned tree that lands inside the tree, so the manifest
    that tree was published with stops matching it, and the next run reports
    the pin's own source corrupt.  The build gets a copy; the frozen tree is
    an input.
    """
    mod = _load()
    clone, old, new = _git_clone_with_two_commits(tmp_path)
    import hashlib
    pin = _pin_module(tmp_path, new, hashlib.sha256(CONTRACT).hexdigest())
    python = _venv_with_tessera(tmp_path, "venv-stale", "VERSION = 1\n")
    pins = tmp_path / "pins"

    seen: list[list[str]] = []
    real_run = subprocess.run

    def spy(argv, *a, **kw):
        if "pip" in argv:
            seen.append([str(x) for x in argv])
            # A build backend really does write here; act like one.
            built = Path(argv[-1])
            (built / "tessera_quant.egg-info").mkdir(exist_ok=True)
            (built / "tessera_quant.egg-info" / "PKG-INFO").write_text("x")
            # capture_output=True with text=True always yields strings, and a
            # fake that returns None for them makes the caller's own logging
            # raise.  Answer in the shape the real call answers in.
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return real_run(argv, *a, **kw)

    monkeypatch.setattr(mod.subprocess, "run", spy)
    mod.main(["--python", str(python), "--pin-source", str(pin),
              "--clone", str(clone), "--pins-root", str(pins),
              "--staging-root", str(tmp_path / "staging")])

    assert seen, "no install was attempted"
    built = Path(seen[0][-1]).resolve()
    assert pins.resolve() not in built.parents and built != (pins / new), (
        f"the build ran in the frozen tree at {built}")

    tree = pins / new
    held = json.loads((tree / mod.MANIFEST).read_text())
    digest, count = mod.tree_digest(tree)
    assert (digest, count) == (held["tree_sha256"], held["files"]), (
        "the install left the pinned tree failing its own manifest")


def test_a_failure_attesting_the_replacement_leaves_the_old_tree_in_place(
        tmp_path, monkeypatch):
    """The repair opens a window where the path is absent; it opens it late.

    Replacing a wrong tree is two renames, and between them the pinned path
    does not exist.  Nothing can make that atomic, so the ordering is what
    there is to get right: the staged tree is hashed and attested BEFORE the
    old name is vacated.  A failure in that step then costs nothing a reader
    can see.  Written the other way round the same failure leaves the pinned
    path gone with a repair that never finished.
    """
    mod = _load()
    clone, old, new = _git_clone_with_two_commits(tmp_path)
    pins = tmp_path / "pins"

    target = mod.materialise(new, clone, pins)
    (target / "src" / "tessera" / "encode.py").write_text("VERSION = 99\n")

    real = mod._write_manifest
    calls: list[Path] = []

    def failing(root, commit):
        calls.append(Path(root))
        raise OSError("disk full while attesting the replacement")

    monkeypatch.setattr(mod, "_write_manifest", failing)
    with pytest.raises(OSError):
        mod.materialise(new, clone, pins)

    assert calls and calls[0] != target, (
        "the first attestation was of the published path, so the old tree had "
        "already been vacated when it failed")
    assert (target / "src" / "tessera" / "encode.py").read_text() == "VERSION = 99\n", (
        "the pinned path is missing or half-repaired after a failure that "
        "happened before anything needed to move")

    monkeypatch.setattr(mod, "_write_manifest", real)
    repaired = mod.materialise(new, clone, pins)
    assert (repaired / "src" / "tessera" / "encode.py").read_text() == "VERSION = 2\n"
