"""The campaign container imports the pinned tree, not the sealed checkout.

``tools/tessera_campaign_container.py`` launched the campaign as ``python3 -u
-m prismaquant.tessera_campaign`` with the working directory set to the PB
sealed checkout, which carries its own ``prismaquant`` package. ``python -m``
puts the working directory at ``sys.path[0]``, ahead of every ``PYTHONPATH``
entry, so the pinned mount named first in ``PYTHONPATH`` never won: 111
completed ``extension-r1024-02`` rows stamped the sealed checkout's package
digest, and the 16/32-reader prefetch path that exists only in the pinned tree
never ran (#519).

These tests build the environment and working directory from the launcher's
own argv builder, then let a real interpreter resolve the import. They prove
the property on this runner's Python, not on the container's.
"""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ORIGIN = ("import importlib.util;"
          "print(importlib.util.find_spec('prismaquant').origin)")


def runner():
    return importlib.import_module("tools.tessera_campaign_container")


def trees(tmp_path):
    """One pinned mount and one sealed checkout, each with a package."""

    pinned, checkout = tmp_path / "pinned", tmp_path / "checkout"
    for root, marker in ((pinned, "pinned"), (checkout, "sealed")):
        package = root / "prismaquant"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"TREE = {marker!r}\n")
    return pinned, checkout


def spec(pinned, *, pythonpath=None):
    return {"container": {"image": "qualified:fixed", "mounts": [
                {"source": str(pinned), "target": str(pinned), "readonly": True}]},
            "env": {"PYTHONPATH": pythonpath or str(pinned),
                    "OMP_NUM_THREADS": "4"}}


def launched_environment(argv):
    """The environment the launcher hands Docker, as a mapping."""

    pairs = [argv[index + 1] for index, value in enumerate(argv) if value == "--env"]
    return dict(pair.split("=", 1) for pair in pairs)


def resolved_origin(environment, cwd):
    result = subprocess.run([sys.executable, "-c", ORIGIN], cwd=str(cwd), text=True,
                            capture_output=True, check=True,
                            env={"PATH": os.environ.get("PATH", ""), **environment})
    return result.stdout.strip()


# -- the property ----------------------------------------------------------

def test_the_pinned_tree_is_what_the_launched_environment_imports(tmp_path):
    pinned, checkout = trees(tmp_path)
    argv = runner().docker_command(spec(pinned), ["python3", "-u", "-m",
                                                  "prismaquant.tessera_campaign"],
                                   cwd=str(checkout), uid=1000, gid=1000,
                                   image_id="sha256:" + "a" * 64)
    environment = launched_environment(argv)
    assert resolved_origin(environment, checkout).startswith(str(pinned))


def test_without_the_guard_the_sealed_checkout_wins(tmp_path):
    """The defect itself, so the guard above is load-bearing and not the fixture."""

    pinned, checkout = trees(tmp_path)
    argv = runner().docker_command(spec(pinned), ["python3", "-u", "-m",
                                                  "prismaquant.tessera_campaign"],
                                   cwd=str(checkout), uid=1000, gid=1000,
                                   image_id="sha256:" + "a" * 64)
    environment = launched_environment(argv)
    environment.pop("PYTHONSAFEPATH")
    assert resolved_origin(environment, checkout).startswith(str(checkout))


def test_the_launched_argv_carries_the_guard(tmp_path):
    pinned, checkout = trees(tmp_path)
    argv = runner().docker_command(spec(pinned), ["python3"], cwd=str(checkout),
                                   uid=1000, gid=1000, image_id="sha256:" + "a" * 64)
    assert "PYTHONSAFEPATH=1" in argv
    assert argv[argv.index("--workdir") + 1] == "/workspace"


def test_a_spec_cannot_supply_its_own_guard(tmp_path):
    pinned, _ = trees(tmp_path)
    data = spec(pinned)
    data["env"]["PYTHONSAFEPATH"] = "0"
    with pytest.raises(RuntimeError, match="supplied by the launcher"):
        runner().validate_container(data)


# -- the replay the gate reasons on ---------------------------------------

@pytest.mark.parametrize("safe_path", [True, False])
def test_the_replay_agrees_with_a_real_interpreter(tmp_path, safe_path):
    """The gate predicts resolution; an interpreter decides it. They must match."""

    module = runner()
    pinned, checkout = trees(tmp_path)
    data = spec(pinned)
    roots = module.import_search_roots(data, cwd=str(checkout), safe_path=safe_path)
    predicted = module._package_root(roots)[1]
    environment = launched_environment(
        module.docker_command(data, ["python3"], cwd=str(checkout), uid=1, gid=1,
                              image_id="sha256:" + "a" * 64))
    if not safe_path:
        environment.pop("PYTHONSAFEPATH")
    observed = Path(resolved_origin(environment, checkout)).parent.parent
    assert observed == predicted


def test_a_mount_target_deeper_than_its_source_still_maps(tmp_path):
    """A ``/producer/src`` entry resolves through a ``/producer`` mount."""

    module = runner()
    pinned, checkout = trees(tmp_path)
    data = {"container": {"image": "qualified:fixed", "mounts": [
                {"source": str(pinned.parent), "target": "/producer", "readonly": True}]},
            "env": {"PYTHONPATH": "/producer/pinned"}}
    assert module.pinned_source_root(data, cwd=str(checkout))[1] == pinned
    assert module.pinned_source_root(data, cwd=str(checkout))[2] is False
    assert module.host_path("/producer/pinned", cwd=str(checkout),
                            mounts=data["container"]["mounts"]) == pinned


# -- the gate --------------------------------------------------------------

def test_the_receipt_records_the_pinned_and_resolved_digests(tmp_path):
    module = runner()
    identity = importlib.import_module("tools.container_runtime_identity")
    pinned, checkout = trees(tmp_path)
    receipt = module.verify_pinned_import(spec(pinned), cwd=str(checkout))
    expected = identity.prismaquant_source_sha256(pinned / "prismaquant")
    assert receipt["pinned_source_sha256"] == expected
    assert receipt["import_resolution_source_sha256"] == expected
    assert receipt["import_resolution_root"] == str(pinned)
    assert receipt["working_directory_source_sha256"] == (
        identity.prismaquant_source_sha256(checkout / "prismaquant"))
    assert receipt["safe_path_guard_is_load_bearing"] is True


def test_an_entry_ahead_of_the_pinned_mount_is_refused(tmp_path):
    """Safe-path mode drops the implicit entry; it does not drop a written ``.``."""

    module = runner()
    pinned, checkout = trees(tmp_path)
    data = spec(pinned, pythonpath="." + os.pathsep + str(pinned))
    with pytest.raises(RuntimeError, match="not from the pinned mount"):
        module.verify_pinned_import(data, cwd=str(checkout))


def test_a_mounted_tree_no_entry_reaches_stays_visible_in_the_receipt(tmp_path):
    """The mistyped-path case: the tree is mounted, and nothing names it.

    This spec mounts the pinned tree and then writes only ``.`` into
    ``PYTHONPATH``, so no entry reaches the mount. The launch proceeds,
    because nothing is shadowing anything the operator declared, but it is not
    silent: the receipt names the sealed checkout as the root and sets
    ``pinned_by_default``, which is the member that tells a reader the root
    was defaulted rather than declared.
    """

    module = runner()
    identity = importlib.import_module("tools.container_runtime_identity")
    pinned, checkout = trees(tmp_path)
    data = spec(pinned, pythonpath=".")
    receipt = module.verify_pinned_import(data, cwd=str(checkout))
    assert receipt["pinned_by_default"] is True
    assert receipt["pinned_source_entry"] is None
    assert receipt["pinned_source_root"] == str(checkout)
    assert receipt["pinned_source_sha256"] == identity.prismaquant_source_sha256(
        checkout / "prismaquant")
    assert receipt["import_resolution_root"] == str(checkout)
    assert receipt["safe_path_guard_is_load_bearing"] is False


def test_a_launch_that_imports_no_prismaquant_is_not_this_question(tmp_path):
    """A generic payload is not a campaign row.

    ``main`` launches whatever argv it is handed, and a spec that names no
    ``PYTHONPATH`` pins no tree. Under the guard the working directory is off
    the search too, so nothing can shadow anything and there is nothing to
    compare; refusing here would stop launches the defect never touched. The
    receipt says nothing was pinned, and still records the sealed checkout.
    """

    module = runner()
    _, checkout = trees(tmp_path)
    receipt = module.verify_pinned_import({"container": {"image": "qualified:fixed"}},
                                          cwd=str(checkout))
    assert receipt["pinned_source_entry"] is None
    assert receipt["pinned_source_sha256"] is None
    assert receipt["pinned_by_default"] is False
    assert receipt["import_resolution_root"] is None
    assert receipt["working_directory_source_sha256"] is not None
    assert receipt["safe_path_guard_is_load_bearing"] is True


def test_the_recorded_census_shape_launches_and_records_a_defaulted_root(tmp_path):
    """The shape every recorded campaign invocation uses.

    ``/workspace`` first, then source trees that hold no ``prismaquant``
    package -- the 2026-09-08 census invocations are Tessera checkouts. The
    launch runs the sealed checkout, which is what it asked for, and the
    receipt marks the root as defaulted.
    """

    module = runner()
    _, checkout = trees(tmp_path)
    tessera = tmp_path / "tessera" / "src"
    (tessera / "tessera").mkdir(parents=True)
    (tessera / "tessera" / "__init__.py").write_text("")
    data = {"container": {"image": "qualified:fixed", "mounts": [
                {"source": str(tessera), "target": "/tessera/src", "readonly": True}]},
            "env": {"PYTHONPATH": "/workspace" + os.pathsep + "/tessera/src"}}
    receipt = module.verify_pinned_import(data, cwd=str(checkout))
    assert receipt["pinned_by_default"] is True
    assert receipt["pinned_source_entry"] is None
    assert receipt["pinned_source_root"] == str(checkout)
    assert receipt["import_resolution_root"] == str(checkout)


def test_main_refuses_a_shadowed_launch_before_it_execs(tmp_path, monkeypatch):
    """The wiring, not only the decision: nothing is exec'd on a refusal."""

    module = runner()
    pinned, checkout = trees(tmp_path)
    launched = {}
    monkeypatch.setattr(module, "inspect_or_load", lambda container: [
        {"Id": "sha256:" + "c" * 64, "RepoDigests": [], "RootFS": {"Layers": []}}])
    monkeypatch.setattr(module, "image_content_sha256", lambda inspected: "d" * 64)
    monkeypatch.setattr(module.os, "execvp",
                        lambda file, argv: launched.setdefault("argv", argv))
    monkeypatch.chdir(checkout)
    data = spec(pinned, pythonpath="." + os.pathsep + str(pinned))
    with pytest.raises(RuntimeError, match="not from the pinned mount"):
        module.main(["--spec", json.dumps(data), "--", "python3", "-c", "pass"])
    assert launched == {}
