"""Keep the editable project's uv lock metadata aligned with its declarations."""
from pathlib import Path
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
LOCK = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
ROOT_PACKAGE = next(
    package for package in LOCK["package"]
    if package["name"] == canonicalize_name(PROJECT["name"])
    and package.get("source") == {"editable": "."}
)
DECLARATIONS = [(None, text) for text in PROJECT["dependencies"]] + [
    (extra, text)
    for extra, requirements in PROJECT["optional-dependencies"].items()
    for text in requirements
]


def test_lock_python_floor_and_project_version():
    assert LOCK["requires-python"] == PROJECT["requires-python"]
    assert ROOT_PACKAGE["version"] == PROJECT["version"]


@pytest.mark.parametrize("extra,text", DECLARATIONS)
def test_each_declared_dependency_is_locked(extra, text):
    requirement = Requirement(text)
    name = canonicalize_name(requirement.name)
    references = (
        ROOT_PACKAGE["dependencies"] if extra is None
        else ROOT_PACKAGE["optional-dependencies"][extra]
    )
    matches = [reference for reference in references if reference["name"] == name]
    assert matches, f"missing {extra or 'base'} dependency: {name}"
    packages = [package for package in LOCK["package"] if package["name"] == name]
    assert packages, f"missing resolved package: {name}"
    for package in packages:
        assert requirement.specifier.contains(package["version"], prereleases=True)
    for reference in matches:
        if "version" in reference:
            assert any(package["version"] == reference["version"] for package in packages)


@pytest.mark.parametrize("extra,text", DECLARATIONS)
def test_each_declared_requirement_has_fresh_root_metadata(extra, text):
    requirement = Requirement(text)
    marker = str(requirement.marker) if requirement.marker is not None else None
    if extra is not None:
        assert marker is None, "extend this check before adding conditional extras"
        marker = f'extra == "{extra}"'
    matches = [
        record for record in ROOT_PACKAGE["metadata"]["requires-dist"]
        if record["name"] == canonicalize_name(requirement.name)
        and (
            str(Requirement(f"x; {record['marker']}").marker)
            if record.get("marker") else None
        ) == marker
    ]
    assert len(matches) == 1, f"missing/duplicate metadata for {text} ({extra})"
    assert str(Requirement("x" + matches[0].get("specifier", "")).specifier) == str(
        requirement.specifier
    )


def test_test_extra_resolves_xdist_and_its_execnet_dependency():
    xdist = [package for package in LOCK["package"] if package["name"] == "pytest-xdist"]
    assert xdist, "pytest-xdist must be resolved, not just named in root metadata"
    for package in xdist:
        assert "execnet" in {reference["name"] for reference in package["dependencies"]}
    assert any(package["name"] == "execnet" for package in LOCK["package"])
