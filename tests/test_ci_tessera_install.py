"""CI must install the exact Tessera checkout PrismaQuant was reviewed against."""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PIN_SOURCE = ROOT / "prismaquant" / "tessera_runtime_contract.py"
PIN_RESOLVER = ROOT / "tools" / "resolve_tessera_dev_pin.py"


def _dev_pin_literal() -> str:
    tree = ast.parse(PIN_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name)
            and target.id == "TESSERA_DEV_PIN_COMMIT"
            for target in targets
        ):
            continue
        value = ast.literal_eval(node.value)
        assert isinstance(value, str)
        return value
    raise AssertionError("TESSERA_DEV_PIN_COMMIT is not a literal assignment")


HOSTED_JOBS = [("ci.yml", "imports"), ("bazel-hosted.yml", "tests")]


def _job(filename: str, name: str) -> dict:
    workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
    return workflow["jobs"][name]


@pytest.mark.parametrize("filename,job_name", HOSTED_JOBS)
@pytest.mark.parametrize("valid_pin", [False, True])
def test_ci_pin_step_propagates_resolver_status(tmp_path, filename, job_name, valid_pin):
    steps = _job(filename, job_name)["steps"]
    step = next(step for step in steps if step.get("id") == "tessera-pin")
    command = step["run"]

    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / PIN_RESOLVER.name).write_text(
        PIN_RESOLVER.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "prismaquant").mkdir()
    pin = _dev_pin_literal() if valid_pin else "not-a-commit"
    (tmp_path / "prismaquant" / PIN_SOURCE.name).write_text(
        f"TESSERA_DEV_PIN_COMMIT = {pin!r}\n", encoding="utf-8"
    )
    output = tmp_path / "github-output"
    # setup-python supplies this name on CI. Reproduce that contract even
    # when the local interpreter is only installed as python3.
    python_bin = tmp_path / "bin"
    python_bin.mkdir()
    (python_bin / "python").symlink_to(sys.executable)
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", command],
        cwd=tmp_path,
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(output),
            "PATH": f"{python_bin}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if valid_pin:
        assert completed.returncode == 0, completed.stderr
        assert output.read_text(encoding="utf-8") == f"commit={pin}\n"
    else:
        assert completed.returncode != 0, "CI masked a failed pin resolver"
        assert "must be a full lowercase Git SHA" in completed.stderr
        assert not output.exists() or not output.read_text(encoding="utf-8")


def test_stdlib_resolver_reads_the_authoritative_pin_without_package_imports():
    completed = subprocess.run(
        [sys.executable, str(PIN_RESOLVER)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == _dev_pin_literal()
    resolver = PIN_RESOLVER.read_text(encoding="utf-8")
    assert "import prismaquant" not in resolver
    assert _dev_pin_literal() not in resolver


@pytest.mark.parametrize("filename,name", HOSTED_JOBS)
def test_ci_installs_the_authoritative_tessera_pin_without_private_credentials(filename, name):
    workflow = (ROOT / ".github/workflows" / filename).read_text()
    assert _dev_pin_literal() not in workflow
    job = _job(filename, name)
    assert "TESSERA_REPO_TOKEN" not in yaml.safe_dump(job)
    steps = job["steps"]
    resolve_index = next(index for index, step in enumerate(steps)
                         if step.get("id") == "tessera-pin")
    assert "python tools/resolve_tessera_dev_pin.py" in steps[resolve_index]["run"]
    checkout_index = next(index for index, step in enumerate(steps)
                          if step.get("with", {}).get("repository") == "RobTand/tessera")
    checkout = steps[checkout_index]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"] == {
        "repository": "RobTand/tessera",
        "ref": "${{ steps.tessera-pin.outputs.commit }}",
        "persist-credentials": False,
        "path": ".ci/tessera",
    }
    commands = [(index, line.strip()) for index, step in enumerate(steps)
                for line in step.get("run", "").splitlines()]
    install = next(position for position, (_, command) in enumerate(commands)
                   if command == "python -m pip install --no-deps .ci/tessera")
    prismaquant = next(position for position, (_, command) in enumerate(commands)
                       if command.startswith("python -m pip install ") and "-e " in command)
    assert resolve_index < checkout_index < commands[install][0]
    assert install < prismaquant


@pytest.mark.parametrize("export_path", [
    ROOT / "tools/bazel/requirements_linux_x86_64.txt",
    ROOT / "tools/bazel/requirements_linux_aarch64.txt",
])
def test_remote_tests_install_the_same_hashed_tessera_pin(export_path):
    export = export_path.read_text()
    blocks = [block for block in export.split("\n\n")
              if re.match(r"tessera[-_.]quant(?:\s|[=@\[])", block, re.I)]
    assert len(blocks) == 1
    words = blocks[0].replace("\\\n", " ").split()
    assert words[:3] == [
        "tessera-quant", "@",
        "https://github.com/RobTand/tessera/archive/" + _dev_pin_literal() + ".tar.gz",
    ]
    assert words[3:] and all(re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", word)
                             for word in words[3:])
