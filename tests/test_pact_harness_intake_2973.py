"""Portable source-intake contract, not the #2974/#2975 harness migration.

The implementation supplies tests/fixtures/pact_harness/manifest.json. Paths in
this small fixture index are relative to that directory. ``baselines`` maps
names to {revision, root, files}; files maps source-relative paths to SHA-256.
``evidence`` maps names to {path, sha256}. Original campaign bytes stay separate
from oracle.py (the portable test driver) and from any future migrated code.

``cases`` contains {id, operation, input, expected}, where input and expected
are {path, sha256} descriptors. oracle.py accepts --case FILE --output-dir DIR,
executes the pinned source using only bound inputs, and writes result.json.
A case input contains its own portable source/input bindings, never the golden
output. Stable result projections are JSON objects; volatile paths/times may
be excluded, but science, refusals, and source identity may not be normalized
away. A refusal is a successful oracle observation, not a crashed driver.

No test here fetches historical git objects or campaign data. Missing fixtures
are the reported delivery gap, rather than an import/discovery/setup failure.
The authenticity/completeness of Campaign's declaration still needs review:
self-consistent hashes alone cannot establish source acceptance authority.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pact_harness"
C926 = "c9260081766ec8a47e7a0f357432833a4d3b1cb6"
AB6 = "ab6bfb843b85ac61b96693d13bc7e9463f004046"
OLD_ENTRY = "6e6f2101ed991a468f5aaaf1be7258a463c6818a52b767c6badf6108f2eb25ec"
CORRECTED_ENTRY = "c661efc8228545b91e97b6c5d50bf5d35f922a1ca47e3b0e00ff081c142a1eaf"
STALL_ENTRY = "2a4e67430a25e76c3d34b649f71239d5763f128f682347bdbf5ec24ea77855b4"
STREAMS = sorted(["null", "W_T8-a1", "W_T8-a2", "W_T8-a4", "A8-a1", "A8-a2", "A8-a4"])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(root, name):
    path = PurePosixPath(name)
    assert name and not path.is_absolute() and ".." not in path.parts, name
    result = root.joinpath(*path.parts)
    assert result.is_relative_to(root)
    assert not result.is_symlink(), f"fixture must carry bytes, not a fleet link: {name}"
    assert result.resolve().is_relative_to(root.resolve()), name
    return result


def checked(root, descriptor):
    path = local(root, descriptor["path"])
    assert path.is_file(), f"Missing portable PACT artifact: {descriptor['path']}"
    assert digest(path) == descriptor["sha256"], str(path)
    return path


def manifest():
    path = FIXTURES / "manifest.json"
    assert path.is_file(), (
        "#2973 has no committed canonical PACT source/caller/input inventory; "
        "do not substitute the old pq2482-canonical-a8 snapshot for the approved handoff"
    )
    data = json.loads(path.read_text())
    assert data["schema"] == "pact.harness_intake.v1"
    return data


def test_original_source_generations_are_reconciled_and_byte_pinned():
    data = manifest()
    expected = {
        "deadline": (C926, CORRECTED_ENTRY, 30),
        "stall": (AB6, STALL_ENTRY, 31),
    }
    for name, (revision, entry_hash, python_count) in expected.items():
        source = data["baselines"][name]
        assert source["revision"] == revision
        root = local(FIXTURES, source["root"])
        files = source["files"]
        assert files["multi_stream_replay.py"] == entry_hash
        assert sum(path.endswith(".py") for path in files) == python_count
        for path, sha256 in files.items():
            checked(root, {"path": path, "sha256": sha256})
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*.py")}
        assert actual == {p for p in files if p.endswith(".py")}
        assert files["API_CONTRACT.json"] == "93465a55cca0936cc7a6c6a3d461c1574ef0a9c16541ff33860b3626d6fddc05"

    # Retain the contradictory source as evidence, not as the corrected oracle.
    candidate = checked(FIXTURES, data["evidence"]["old_snapshot_entry"])
    assert digest(candidate) == OLD_ENTRY
    old_imports = checked(FIXTURES, data["evidence"]["old_import_manifest"])
    assert digest(old_imports) == "48d1432271666974e28445ccc6f7648b74a2cb81d78fd5806e6360c0b66c9729"
    report = checked(FIXTURES, data["evidence"]["equivalence_a8"])
    assert digest(report) == "76bb6c75f1d79c8cef81b848d7493b4f5fd35e18f5f8e336ab9e5852b5f294c2"
    provenance = checked(FIXTURES, data["evidence"]["deadline_import_provenance"])
    assert digest(provenance) == "3e50ec0afe04f04ea6d1db90a035aa80ab42d8aff824740665446a839ce76da2"
    imported = json.loads(provenance.read_text())["imported_files"]
    assert len(imported) == 31  # 30 Python files plus API_CONTRACT.json.
    deadline = data["baselines"]["deadline"]
    for path, sha256 in imported.items():
        relative = path.removeprefix("experiments/pact_replay/")
        assert deadline["files"][relative] == sha256
    reconciliation = data["reconciliation"]
    for field in ("entry_hash_conflict", "file_count", "observed_vs_declared", "deadline_vs_stall"):
        assert reconciliation[field].strip(), field


def test_campaign_declared_callers_dependencies_and_frozen_copies_are_inventoried():
    data = manifest()
    # This is a retained declaration, not a list inferred solely from imports.
    declaration = json.loads(checked(FIXTURES, data["evidence"]["campaign_handoff"]).read_text())
    assert declaration["authority"] and declaration["source_evidence"]
    for evidence in declaration["source_evidence"]:
        checked(FIXTURES, evidence)
    canonical = data["baselines"][data["canonical_baseline"]]
    assert declaration["canonical_revision"] == canonical["revision"]
    declared = declaration["declared_callers"]
    callers = data["callers"]
    assert declared and len(declared) == len(set(declared))
    assert sorted(declared) == sorted(row["id"] for row in callers)
    for row in callers:
        checked(FIXTURES, row["source"])
        assert row["entry_point"] and row["source_revision"]
        assert row["disposition"] in {"migrate", "frozen", "historical", "owned_elsewhere"}
        destination = PurePosixPath(row["planned_destination"])
        assert not destination.is_absolute() and ".." not in destination.parts
        assert len(destination.parts) > 1
        if row["disposition"] == "owned_elsewhere":
            assert row["owner_issue"] in {"prismaquant#2960", "prismaquant#2961"}
    dependencies = data["dependencies"]
    assert dependencies
    assert {row["id"] for row in dependencies} >= {"g3-source", "pq-owner", "tessera-pin"}
    for row in dependencies:
        assert row["revision"] and row["files"]
        checked(FIXTURES, row["identity_evidence"])
        assert all(len(value) == 64 for value in row["files"].values())
    frozen = data["frozen_execution_copies"]
    assert frozen
    assert sorted(row["id"] for row in frozen) == sorted(declaration["frozen_execution_copies"])
    for row in frozen:
        # Original absolute paths are provenance only, never read by the check.
        assert row["original_path"] and row["source_revision"] and row["files"]
        assert row["cutover_condition"] and row["leave_untouched"] is True
        assert all(len(value) == 64 for value in row["files"].values())
    assert data["external_ownership"]["price_adapter"] == ["prismaquant#2960", "prismaquant#2961"]


def test_approved_deadline_and_later_stall_policy_are_distinguished():
    data = manifest()
    report = json.loads(checked(FIXTURES, data["evidence"]["equivalence_a8"]).read_text())
    assert report["cli_source_head"] == C926
    assert report["approved"]["args"]["deadline_seconds"] == 3500
    assert report["legacy"][0]["resolved"]["args"]["deadline_seconds"] == 1700
    packet_path = checked(FIXTURES, data["evidence"]["stall_approval_packet"])
    assert digest(packet_path) == "abcf93676a10b7a95d393168b46cc5f9ae270fe636e77adefd970e460e784faa"
    packet = json.loads(packet_path.read_text())
    assert packet["source_head"] == AB6
    assert packet["execution"]["app_stall_s"] == 1800
    assert packet["execution"]["payload_backstop_s"] == 7200
    assert packet["execution"]["pb_phase_grace_s"] == 1560
    policy = data["approved_behavior"]
    assert policy["deadline"] == {"default_seconds": 1700, "minimum_exclusive": 0, "maximum_inclusive": 3500}
    assert policy["stall"]["default_seconds"] == 1800
    assert policy["stall"]["fixed_application_duration_cap"] is False
    assert policy["stall"]["resets_on"] == "completed_work"
    assert policy["selected_baseline"] == data["canonical_baseline"]
    assert policy["selection_evidence"]  # Authority of this selection is reviewed.
    checked(FIXTURES, policy["selection_evidence"])


def test_canonical_oracle_replays_bound_inputs_without_history_or_fleet(tmp_path):
    data = manifest()
    cases = {row["id"]: row for row in data["cases"]}
    assert len(cases) == len(data["cases"])
    required = {
        "glm-cpu-dry-run", "deadline-default", "deadline-3500",
        "deadline-zero", "deadline-negative", "deadline-over-3500", "deadline-nan",
        "wrong-band", "raw-cohort", "off-grid-window", "wrong-weight-source",
    }
    assert required <= cases.keys()
    # Copy ONLY the fixture package, not the checkout or .git. Goldens are then
    # removed before running: the oracle cannot pass by reading expected JSON.
    bundle = tmp_path / "relocated"
    shutil.copytree(FIXTURES, bundle)
    goldens = {}
    for row in cases.values():
        case = json.loads(checked(FIXTURES, row["input"]).read_text())
        assert case["bindings"], f"{row['id']}: every source/input binding needs a digest"
        for binding in case["bindings"]:
            checked(FIXTURES, binding)
        goldens[row["id"]] = json.loads(checked(FIXTURES, row["expected"]).read_text())
    for row in cases.values():
        local(bundle, row["expected"]["path"]).unlink(missing_ok=True)
    (bundle / "manifest.json").unlink()
    driver = checked(bundle, data["oracle"])
    before = {p.relative_to(bundle).as_posix(): digest(p) for p in bundle.rglob("*") if p.is_file()}
    for name, row in cases.items():
        output = tmp_path / "outputs" / name
        env = {
            "PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path),
            "TMPDIR": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        }
        result = subprocess.run(
            [sys.executable, "-B", str(driver), "--case", str(local(bundle, row["input"]["path"])),
             "--output-dir", str(output)], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=90,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}\n{result.stderr}"
        result_file = output / "result.json"
        assert result_file.is_file(), f"{name}: oracle must emit machine-readable result.json"
        observed = json.loads(result_file.read_text())
        assert observed == goldens[name], name
        assert observed["operation"] == row["operation"]
        baseline = data["baselines"][observed["baseline"]]
        assert observed["source_revision"] == baseline["revision"]
        assert observed["source_entry_sha256"] == baseline["files"]["multi_stream_replay.py"]
        if name in {"deadline-default", "deadline-3500"}:
            assert observed["baseline"] == "deadline"
            assert observed["status"] == "accepted"
            assert observed["args"]["deadline_seconds"] == (1700 if name == "deadline-default" else 3500)
            run = observed["run"]
            assert [run["band_start"], run["band_stop"]] == [39, 45]
            assert run["replay_stop"] == 45
            assert run["plan"] == [{"window_start": 39, "window_stop": 45}]
            assert sorted(s["stream_id"] for s in run["manifest_streams"]) == STREAMS
        elif name == "glm-cpu-dry-run":
            assert row["operation"] == "dry-run-cpu"
            assert observed["status"] == "accepted"
            assert observed["proof"]["actual_consumer_passed"] is True
            assert observed["proof"]["full_heldout_measured"] is False
            assert sorted(observed["proof"]["streams"]) == STREAMS
        elif name in required:
            assert observed["status"] == "refused"
            assert observed["error_type"] in {"ValueError", "SystemExit"}
            assert observed["message"]
    after = {p.relative_to(bundle).as_posix(): digest(p) for p in bundle.rglob("*") if p.is_file()}
    assert before == after, "dry-run oracle must not modify canonical comparison bytes or bound inputs"
