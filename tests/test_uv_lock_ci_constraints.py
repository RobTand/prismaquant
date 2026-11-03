"""Keep the Bazel lock aligned with the hosted test environment (PQ #3216)."""
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = [
    Requirement(line)
    for raw in (ROOT / ".github/ci-constraints.txt").read_text().splitlines()
    if (line := raw.split("#", 1)[0].strip())
]


@pytest.mark.parametrize("constraint", CONSTRAINTS, ids=lambda item: item.name)
def test_uv_lock_matches_every_hosted_pin(constraint):
    pins = list(constraint.specifier)
    assert len(pins) == 1 and pins[0].operator == "==" and "*" not in pins[0].version
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions = {
        package["version"]
        for package in lock["package"]
        if canonicalize_name(package["name"]) == canonicalize_name(constraint.name)
    }
    assert versions == {pins[0].version}, (
        f"uv.lock resolves {constraint.name} to {sorted(versions)}, "
        f"but the hosted gate requires {constraint}; run uv lock --upgrade"
    )


def test_cpu_torch_index_is_explicit():
    settings = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]
    assert settings["sources"]["torch"]["index"] == "pytorch-cpu"
    index = next(index for index in settings["index"] if index["name"] == "pytorch-cpu")
    assert index["url"] == "https://download.pytorch.org/whl/cpu"
    assert index["explicit"] is True


def test_cpu_torch_source_is_limited_to_test_extra():
    settings = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]
    assert settings["sources"]["torch"].get("extra") == "test", (
        "the CPU index must not replace torch for ordinary uv sync or serving"
    )


def test_non_test_torch_keeps_its_pypi_source():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    project = next(package for package in lock["package"] if package["name"] == "prismaquant")
    assert any(
        edge["name"] == "torch"
        and edge.get("source") == {"registry": "https://pypi.org/simple"}
        for edge in project["dependencies"]
    ), "the non-test project must retain a resolved PyPI torch dependency"
