"""Acceptance of retained #2948 evidence, not a new provenance issuer.

The existing artifact owner supplies docs/measurements/issue-2948/handoff.json
and the original referenced bytes. This small *test index* accommodates the
owner's original JSON layouts without rewriting their records. References use
repo-relative path/bytes/sha256; *_at values are nonempty lists of JSON keys
(or integer array indices) in the original selection/capture document.

Authenticity, original-capture completeness, the historical 29-file roster,
and the independent exact-head verdict still require review of the existing
owner/pipeline evidence. Passing this test alone is NOT image authentication.
Do not satisfy it with a synthetic capture or today's installed environment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "docs/measurements/issue-2948/handoff.json"


def _sha(value):
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), value
    return value


def _artifact(reference):
    """Hash retained bytes, never a URL, host path, or newly installed package."""
    relative = PurePosixPath(reference["path"])
    assert not relative.is_absolute() and ".." not in relative.parts
    path = ROOT / str(relative)
    assert path.resolve().is_relative_to(ROOT)
    assert path.is_file(), f"Original retained artifact is not readable: {relative}"
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    assert type(reference["bytes"]) is int and reference["bytes"] >= 0
    assert size == reference["bytes"], relative
    assert digest.hexdigest() == _sha(reference["sha256"]), relative
    return path


def _at(document, keys):
    assert isinstance(keys, list) and keys, "Supply a location in the original record"
    for key in keys:
        assert type(key) in (str, int)
        document = document[key]
    return document


def _records():
    assert HANDOFF.is_file(), (
        "#2948 remains unmet: existing owner/pipeline must deliver the original "
        "selected-image installed-source/dependency capture and selection evidence; "
        "archive integrity or a matching runner is not a substitute."
    )
    index = json.loads(HANDOFF.read_bytes())
    selection = json.loads(_artifact(index["selection"]["artifact"]).read_bytes())
    capture = json.loads(_artifact(index["capture"]["artifact"]).read_bytes())
    return index, selection, capture


def test_original_capture_binds_the_immutable_selected_image():
    index, selection, capture = _records()
    selected = _at(selection, index["selection"]["image_identity_at"])
    observed = _at(capture, index["capture"]["image_identity_at"])
    assert isinstance(selected, str)
    # Permit content SHA-256, sha256:<digest>, or an OCI repo@sha256:<digest>.
    assert re.fullmatch(r"(?:[^\s]+@sha256:|sha256:)?[0-9a-f]{64}", selected)
    assert observed == selected, "Original capture must name the actual selected image"


def test_original_installed_paths_source_bytes_and_dependency_observations_are_retained():
    index, _, capture = _records()
    sources = index["installed_sources"]
    assert isinstance(sources, list) and sources, "Original installed-source roster is required"
    paths = set()
    for source in sources:
        installed = _at(capture, source["installed_path_at"])
        assert isinstance(installed, str) and PurePosixPath(installed).is_absolute()
        assert ".." not in PurePosixPath(installed).parts
        assert installed not in paths, f"Repeated installed path: {installed}"
        paths.add(installed)
        _artifact(source["artifact"])
        assert _sha(_at(capture, source["source_sha256_at"])) == source["artifact"]["sha256"]
    observations = _at(capture, index["dependency_observations_at"])
    assert isinstance(observations, (dict, list)) and observations, (
        "The original image-bound dependency observations must be retained"
    )
    dependencies = index["dependency_artifacts"]
    assert isinstance(dependencies, list) and dependencies, "Retain the unchanged dependency bytes"
    for dependency in dependencies:
        _artifact(dependency["artifact"])
        assert _sha(_at(capture, dependency["source_sha256_at"])) == dependency["artifact"]["sha256"]
