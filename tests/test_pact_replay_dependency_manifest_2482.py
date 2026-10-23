"""Check the byte links between CLI observations and source inventories."""

import hashlib
import json
from pathlib import Path

REPLAY_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "pact_replay"
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
    for name, (tag, path, digest) in capture["modules"].items():
        if tag == "pact-replay":
            expected = provenance["imported_files"]["experiments/pact_replay/" + path]
        else:
            expected = manifest["sources"][tag]["files"][path]["sha256"]
        assert digest == expected, name


def test_provenance_references_manifest_digest():
    provenance = json.loads(PROVENANCE.read_bytes())
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert provenance["dependency_manifest_sha256"] == digest


def test_imported_baseline_bytes_match_provenance():
    provenance = json.loads(PROVENANCE.read_bytes())
    root = REPLAY_ROOT.parents[1]
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "pact_replay_2520_baseline_edits.json").read_bytes())
    assert fixture["baseline_commit"] == "fc518e4897f88982b6ff49bf9293eea5d76ecd46"
    assert set(fixture["files"]) == {
        provenance["entry"], "experiments/pact_replay/bounded_profile.py",
    }
    for path, expected in provenance["imported_files"].items():
        raw = (root / path).read_bytes()
        # Retain only the small reviewed reverse edits, not a copy of the
        # imported closure. Exact anchors reject any unrecorded source drift.
        for edit in fixture["files"].get(path, []):
            current, baseline = edit["current"].encode(), edit["baseline"].encode()
            assert current and raw.count(current) == 1, path
            raw = raw.replace(current, baseline, 1)
        assert hashlib.sha256(raw).hexdigest() == expected, path
