"""The hosted test environment installs the reviewed PrismaBuild SDK (PQ #3216).

Rob, 2026-11-02: "The library should be installed for tests if it's
needed." Before this, no hosted job installed ``prismabuild``, so 779 tests
skipped on every hosted run with ``fleet_sdk.SKIP_REASON`` and only the
fleet's pbtest shards ran them.

The reviewed commit lives in one place,
``staged_lease.PB_READER_LEASE_PIN_COMMIT``, which pbtest's pin guard and
``staged_lease.reviewed_installed_distribution`` already enforce. The
``pbsdk`` dependency group in ``pyproject.toml`` and its ``uv.lock`` entry
must name that same commit. The hosted exception suite installs the group;
Remote Bazel suites install the same reviewed archive with its authoritative
hash. A dependency group, not the ``test`` extra: an extra is published package
metadata, and PyPI refuses a direct Git reference in it.
"""
from __future__ import annotations

from pathlib import Path
import json
import tomllib
from urllib.parse import parse_qs, urlsplit

from packaging.requirements import Requirement
import pytest
import yaml

from prismaquant.staged_lease import PB_READER_LEASE_PIN_COMMIT
from tools.bazel import requirements

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/RobTand/prismabuild"
GROUP = "pbsdk"
#: The one install line every suite-running hosted job uses.
INSTALL = "python -m pip install -c .github/ci-constraints.txt -e '.[test]' --group pbsdk"


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _lock() -> dict:
    return tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))


def _workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def _runs(job: dict) -> list[str]:
    return [line.strip() for step in job["steps"] for line in step.get("run", "").splitlines()]


def test_the_pbsdk_group_pins_the_reviewed_commit():
    group = _pyproject().get("dependency-groups", {}).get(GROUP)
    assert group is not None, f"pyproject.toml declares no {GROUP!r} dependency group"
    requirements = [Requirement(text) for text in group]
    assert [requirement.name for requirement in requirements] == ["prismabuild"]
    assert requirements[0].url == f"git+{REPOSITORY}@{PB_READER_LEASE_PIN_COMMIT}"


def test_published_metadata_never_names_prismabuild():
    project = _pyproject()["project"]
    declared = list(project["dependencies"]) + [
        text for texts in project["optional-dependencies"].values() for text in texts]
    assert "prismabuild" not in {Requirement(text).name for text in declared}


def test_the_lock_resolves_prismabuild_at_the_reviewed_commit():
    lock = _lock()
    packages = [package for package in lock["package"] if package["name"] == "prismabuild"]
    assert len(packages) == 1, "uv.lock must resolve exactly one prismabuild"
    source = urlsplit(packages[0]["source"]["git"])
    assert f"{source.scheme}://{source.netloc}{source.path}" == REPOSITORY
    assert parse_qs(source.query) == {"rev": [PB_READER_LEASE_PIN_COMMIT]}
    assert source.fragment == PB_READER_LEASE_PIN_COMMIT
    root = next(package for package in lock["package"]
                if package["name"] == "prismaquant" and package["source"] == {"editable": "."})
    assert root["dev-dependencies"][GROUP] == [{"name": "prismabuild"}]
    declared = root["metadata"]["requires-dev"][GROUP]
    assert declared == [{"name": "prismabuild",
                         "git": f"{REPOSITORY}?rev={PB_READER_LEASE_PIN_COMMIT}"}]


def test_the_hosted_exception_suite_installs_the_pbsdk_group():
    assert INSTALL in _runs(_workflow("bazel-hosted.yml")["jobs"]["tests"])


@pytest.mark.parametrize("platform", requirements.PLATFORMS)
def test_remote_suites_install_the_hash_verified_reviewed_sdk(platform):
    source = json.loads((ROOT / requirements.SDK_PROVENANCE).read_text())
    assert source == requirements.sdk_source(ROOT)
    assert source["commit"] == PB_READER_LEASE_PIN_COMMIT
    export = (ROOT / requirements.EXPORT.format(platform)).read_text()
    assert requirements.pins(export)["prismabuild"] == source["url"]
    sdk = next(block for block in requirements.export_blocks(export)
               if block[0].startswith("prismabuild @"))
    assert sdk[1:] == ["--hash=sha256:" + source["sha256"]]
