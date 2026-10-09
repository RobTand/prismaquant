"""Check the byte links between CLI observations and source inventories."""

import hashlib
import json
from pathlib import Path

REPLAY_ROOT = Path(__file__).resolve().parents[1] / "prismaquant" / "pact_replay"
MANIFEST = REPLAY_ROOT / "DEPENDENCY_MANIFEST.json"
PROVENANCE = REPLAY_ROOT / "IMPORT_PROVENANCE.json"


def test_observed_imports_match_cli_capture():
    manifest = json.loads(MANIFEST.read_bytes())
    binding = manifest["cli_capture"]
    raw = (REPLAY_ROOT / binding["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == binding["sha256"]
    provenance = json.loads(PROVENANCE.read_bytes())
    capture = json.loads(raw)
    assert capture["action_key"] == manifest["d38_evidence"]["action"]
    assert capture["source_head"] == manifest["d38_evidence"]["snapshot_parent"]
    assert capture["returncode"] == 0
    assert capture["entry"]["sha256"] == provenance["correction"]["corrected_sha256"]
    for name, observation in capture["modules"].items():
        if observation["source"] == "pact-replay":
            expected = provenance["imported_files"]["prismaquant/pact_replay/" + observation["path"]]
        else:
            source = manifest["sources"][observation["source"]]
            expected = source["files"][observation["path"]]["sha256"]
        assert observation["sha256"] == expected, name


def test_provenance_references_manifest_digest():
    provenance = json.loads(PROVENANCE.read_bytes())
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert provenance["dependency_manifest_sha256"] == digest


def test_imported_source_bytes_match_provenance():
    provenance = json.loads(PROVENANCE.read_bytes())
    root = REPLAY_ROOT.parents[1]
    for path, expected in provenance["imported_files"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path
