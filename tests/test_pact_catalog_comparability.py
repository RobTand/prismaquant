"""Portable retained-producer qualification boundary (PQ #2949).

Only synthetic source snapshots are executed here. Historical tests and packets
under fixtures/ are inert, digest-bound evidence of their actual producer, not
assertions about the development-mode policy of the running PrismaQuant package.
No historical Git objects, fleet paths, proof opt-in, imports of producer code,
network, numerical measurements or GPU are needed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/pact_catalog_comparability.py"
RETAINED = ROOT / "tests/fixtures/pact_catalog_comparability/retained"
MANIFEST = ROOT / "tools/manifests/pact_catalog_bridge_8967a796.json"
OWNERS = ["joint_catalog_extension.py", "joint_served_activation.py", "joint_stageb_resources.py"]
NUMERICAL = ["production_weight_cache.py", "joint_loss.py", "joint_probe.py",
             "calibration_loader.py", "model_profiles/fixture.json", "data/quantizer.bin"]


def _json(path):
    return json.loads(path.read_text())


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _binding(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _files(root):
    return {p.relative_to(root).as_posix(): _binding(p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()
            and ".git" not in p.relative_to(root).parts
            and "__pycache__" not in p.relative_to(root).parts
            and p.suffix not in {".pyc", ".pyo"}}


def _package_hash(root):
    # The retained ProductionWeightCache framing, independently implemented as
    # a test oracle; neither source snapshot is importable/executed.
    h = hashlib.sha256()
    for name in _files(root):
        raw = (root / name).read_bytes()
        encoded = name.encode()
        h.update(len(encoded).to_bytes(4, "big"))
        h.update(encoded)
        h.update(len(raw).to_bytes(8, "big"))
        h.update(raw)
    return h.hexdigest()


def _git(root, *args):
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_AUTHOR_NAME="fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
               GIT_COMMITTER_NAME="fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid",
               GIT_AUTHOR_DATE="2000-01-01T00:00:00+0000",
               GIT_COMMITTER_DATE="2000-01-01T00:00:00+0000")
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, env=env).stdout


@pytest.fixture
def artifact(tmp_path):
    """Tiny real local Git lineage; never retrieves a historical commit."""
    original, bridge, evidence = (tmp_path / name for name in ("original", "bridge", "evidence"))
    original.mkdir()
    evidence.mkdir()
    files = {"prismaquant/__init__.py": "raise AssertionError('do not import producer')\n",
             "docs/ARCHITECTURE.md": "original architecture\n",
             "tools/prepare.py": "raise AssertionError('do not run producer')\n"}
    files.update({"prismaquant/" + name: "original " + name + "\n" for name in OWNERS + NUMERICAL})
    for name, text in files.items():
        p = original / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    _git(original, "init", "-q")
    _git(original, "add", ".")
    _git(original, "commit", "-qm", "synthetic original")
    base = _git(original, "rev-parse", "HEAD").decode().strip()
    shutil.copytree(original, bridge)
    for name in OWNERS:
        (bridge / "prismaquant" / name).write_text("bridge metadata/proof wall " + name + "\n")
    (bridge / "tools/prepare.py").write_text("bridge preparation metadata\n")
    (bridge / "docs/ARCHITECTURE.md").write_text("bridge architecture\n")
    _git(bridge, "add", ".")
    _git(bridge, "commit", "-qm", "synthetic bridge")
    head = _git(bridge, "rev-parse", "HEAD").decode().strip()
    source_diff = _git(bridge, "diff", "--binary", "--full-index", "--no-renames", base, head)
    (evidence / "source.diff").write_bytes(source_diff)
    old_files, new_files = _files(original), _files(bridge)
    changed = sorted(k for k in old_files.keys() | new_files.keys() if old_files.get(k) != new_files.get(k))
    source = {"original_source": {"commit": base, "package_sha256": _package_hash(original / "prismaquant")},
              "bridge_source": {"commit": head, "package_sha256": _package_hash(bridge / "prismaquant")}}
    accepted = _binding((bridge / "prismaquant/joint_catalog_extension.py").read_bytes())
    identities = {k.removeprefix("prismaquant/"): {"original": old_files[k], "bridge": new_files[k]}
                  for k in old_files if k.startswith("prismaquant/")}
    closure = {"schema": "pact.catalog_source_comparability.v2", **source,
               "package_files": len(identities), "changed_package_paths": OWNERS,
               "file_identities": identities, "accepted_b522_catalog_owner": accepted,
               "accepted_b522_catalog_owner_byte_identical": True,
               "non_admission_package_files_byte_identical": True,
               "bridge_same_encoder_control": None, "canary_comparability_proved": False,
               "accepted_as_comparable": False, "new_joint_prices_admitted": 0,
               "retained_rows_modified": False, "source_hash_restamp": False, "review_required": True}
    _write_json(evidence / "closure.json", closure)
    closure_binding = _binding((evidence / "closure.json").read_bytes())
    _write_json(evidence / "attestation.json", {
        "schema": "pact.test_only_head_package_attestation.v1", "head": head,
        "actual_package_sha256": source["bridge_source"]["package_sha256"], "package_equal_companion": True})
    _write_json(evidence / "qualification.json", {
        "schema": "pact.fixture_historical_regressions.v1", "head": head,
        "scope": "synthetic producer only; not measured or independently accepted"})
    qualification_binding = _binding((evidence / "qualification.json").read_bytes())
    # Deliberately obsolete absolute paths: resolution uses explicit evidence
    # locations by role/digest, never these provenance-only path strings.
    _write_json(evidence / "readiness.json", {
        "schema": "pact.companion_source_default_control_readiness.v1", "head_sha": head,
        "actual_closure": {"path": "/historical/unavailable/closure.json", **closure_binding,
                           "package_sha256": source["bridge_source"]["package_sha256"],
                           "changed_package_paths": OWNERS, "package_files": len(identities),
                           "accepted_b522_catalog_owner_sha256": accepted["sha256"]}})
    _write_json(evidence / "handoff.json", {
        "schema": "pact.exact_review_handoff.v1", "head": head,
        "source_artifact": "historical isolated branch only, not main",
        "qualification_packet": {"path": "/historical/unavailable/qualification.json", **qualification_binding},
        "table_joint_pin_default_source_promotion": False})
    roles = {role: {"path": name, **_binding((evidence / name).read_bytes())} for role, name in {
        "source_closure": "closure.json", "package_attestation": "attestation.json",
        "source_default_control_readiness": "readiness.json", "exact_review_handoff": "handoff.json",
        "qualification_packet": "qualification.json"}.items()}
    shared = {k: v for k, v in old_files.items() if new_files.get(k) == v}
    manifest = {"schema": "pact.catalog_source_manifest.v1", **source,
                "source_inventory": {"shared": shared,
                                     "original": {k: v for k, v in old_files.items() if k not in shared},
                                     "bridge": {k: v for k, v in new_files.items() if k not in shared}},
                "allowed_package_changes": OWNERS,
                "accepted_b522_catalog_owner_sha256": accepted["sha256"],
                "source_diff": {"path": "source.diff", **_binding(source_diff), "paths": changed},
                "evidence": roles}
    path = tmp_path / "manifest.json"
    _write_json(path, manifest)
    return {"original": original, "bridge": bridge, "evidence": evidence,
            "manifest_path": path, "manifest": manifest, "output": tmp_path / "report.json",
            "old_files": old_files, "new_files": new_files}


def _invoke(a, *, original=None, bridge=None, output=None):
    assert TOOL.is_file(), "#2949: portable retained-artifact verifier has not been delivered"
    env = dict(os.environ)
    env.pop("PQ_TEST_GENUINE_ENCODER_PROOF", None)
    return subprocess.run([sys.executable, str(TOOL), "--identity-only",
                           "--original-source", str(original or a["original"]),
                           "--bridge-source", str(bridge or a["bridge"]),
                           "--manifest", str(a["manifest_path"]),
                           "--evidence-root", str(a["evidence"]),
                           "--output", str(output or a["output"])],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)


def _good(a, **kwargs):
    result = _invoke(a, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return _json(kwargs.get("output", a["output"]))


def _refused(a, reason, **kwargs):
    result = _invoke(a, **kwargs)
    message = (result.stdout + result.stderr).lower()
    assert result.returncode != 0, "tampered identity was accepted"
    assert "unrecognized arguments" not in message and "modulenotfounderror" not in message, message
    assert reason.lower() in message, message
    assert not a["output"].exists(), "a refusal must not publish a successful closure report"


@pytest.mark.parametrize("exported", [False, True], ids=["git-roots", "exported-roots"])
def test_relocated_explicit_sources_and_evidence_are_bound_without_qualification(artifact, exported):
    a = artifact
    destination = a["output"].parent / "relocated with spaces"
    destination.mkdir()
    for key in ("original", "bridge", "evidence"):
        new = destination / key
        shutil.move(str(a[key]), new)
        a[key] = new
    if exported:
        for key in ("original", "bridge"):
            shutil.rmtree(a[key] / ".git")
    before = {key: _files(a[key]) for key in ("original", "bridge", "evidence")}
    report = _good(a)
    m = a["manifest"]
    assert report["schema"] == "pact.catalog_source_comparability.v3"
    for key in ("original_source", "bridge_source"):
        assert report[key]["commit"] == m[key]["commit"]
        assert report[key]["package_sha256"] == m[key]["package_sha256"]
    assert report["manifest"]["sha256"] == _binding(a["manifest_path"].read_bytes())["sha256"]
    assert report["changed_package_paths"] == OWNERS
    assert report["source_diff"]["paths"] == m["source_diff"]["paths"]
    for key in ("sha256", "bytes"):
        assert report["source_diff"][key] == m["source_diff"][key]
    assert report["source_diff"]["text"] == (a["evidence"] / "source.diff").read_text()
    expected = _json(a["evidence"] / "closure.json")["file_identities"]
    assert report["file_identities"] == expected
    assert report["package_files"] == len(expected)
    assert report["non_admission_package_files_byte_identical"] is True
    assert report["accepted_b522_catalog_owner_byte_identical"] is True
    assert report["accepted_b522_catalog_owner"]["sha256"] == m["accepted_b522_catalog_owner_sha256"]
    for role, bound in m["evidence"].items():
        assert report["evidence"][role]["sha256"] == bound["sha256"]
        assert report["evidence"][role]["bytes"] == bound["bytes"]
    assert report["bridge_same_encoder_control"] is None
    assert report["canary_comparability_proved"] is False
    assert report["accepted_as_comparable"] is False
    assert report["new_joint_prices_admitted"] == 0
    assert report["retained_rows_modified"] is False
    assert report["source_hash_restamp"] is False
    assert report["review_required"] is True
    assert before == {key: _files(a[key]) for key in before}


@pytest.mark.parametrize("side", ["original", "bridge"])
def test_wrong_git_source_pin_is_not_hidden_by_matching_files(artifact, side):
    a = artifact
    _good(a, output=a["output"].with_name("baseline.json"))
    # Same tree, different commit; checking only package bytes misses this.
    _git(a[side], "commit", "--allow-empty", "-qm", "not the pinned producer")
    _refused(a, "commit")


@pytest.mark.parametrize("side", ["original", "bridge"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "changed"])
def test_complete_inventory_refuses_missing_extra_changed_package_files(artifact, side, mutation):
    a = artifact
    _good(a, output=a["output"].with_name("baseline.json"))
    p = a[side] / "prismaquant/data/quantizer.bin"
    if mutation == "missing":
        p.unlink()
    elif mutation == "extra":
        p = a[side] / "prismaquant/data/extra.bin"
        p.write_bytes(b"unauthorized")
    else:
        p.write_bytes(b"altered durable input")
    _refused(a, p.name)


@pytest.mark.parametrize("name", NUMERICAL)
def test_unauthorized_fourth_package_owner_including_numerical_closure(artifact, name):
    a = artifact
    _good(a, output=a["output"].with_name("baseline.json"))
    p = a["bridge"] / "prismaquant" / name
    p.write_bytes(p.read_bytes() + b"changed numerical owner\n")
    _refused(a, Path(name).name)


def test_altered_catalog_proof_owner_is_refused_even_inside_three_owner_allowlist(artifact):
    a = artifact
    _good(a, output=a["output"].with_name("baseline.json"))
    p = a["bridge"] / "prismaquant/joint_catalog_extension.py"
    p.write_bytes(p.read_bytes() + b"waive proof wall\n")
    _refused(a, "joint_catalog_extension.py")


def test_accepted_catalog_digest_is_an_independent_pin(artifact):
    a = artifact
    _good(a, output=a["output"].with_name("baseline.json"))
    # Matching source/package inventories and an allowed changed-path set do
    # not replace the independently accepted catalog/proof-owner identity.
    a["manifest"]["accepted_b522_catalog_owner_sha256"] = "0" * 64
    _write_json(a["manifest_path"], a["manifest"])
    _refused(a, "catalog")


def test_manifest_cannot_authorize_a_fourth_package_owner(artifact):
    a = artifact
    _good(a, output=a["output"].with_name("baseline.json"))
    a["manifest"]["allowed_package_changes"] = OWNERS + ["production_weight_cache.py"]
    _write_json(a["manifest_path"], a["manifest"])
    _refused(a, "allow")


@pytest.mark.parametrize("side", ["original", "bridge"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "changed"])
def test_full_source_inventory_extends_beyond_package(artifact, side, mutation):
    a = artifact
    _good(a, output=a["output"].with_name("baseline.json"))
    p = a[side] / "tools/prepare.py"
    if mutation == "missing":
        p.unlink()
    elif mutation == "extra":
        p = a[side] / "tools/extra.py"
        p.write_text("unapproved repository delta\n")
    else:
        p.write_text("changed preparation invocation\n")
    _refused(a, p.name)


@pytest.mark.parametrize("role", ["source_closure", "package_attestation", "source_default_control_readiness", "exact_review_handoff", "qualification_packet"])
def test_tampered_packet_bytes_are_not_trusted(artifact, role):
    a = artifact
    _good(a, output=a["output"].with_name("baseline.json"))
    name = a["manifest"]["evidence"][role]["path"]
    p = a["evidence"] / name
    p.write_bytes(p.read_bytes() + b" ")
    _refused(a, name)


@pytest.mark.parametrize("mutation", ["head", "package", "closure-link", "proof-link", "inventory"])
def test_semantic_packet_cross_bindings_are_checked_not_just_outer_digests(artifact, mutation):
    a = artifact
    _good(a, output=a["output"].with_name("baseline.json"))
    role, field, value, diagnostic = {
        "head": ("package_attestation", "head", "f" * 40, "head"),
        "package": ("package_attestation", "actual_package_sha256", "0" * 64, "package"),
        "closure-link": ("source_default_control_readiness", "actual_closure", {"sha256": "0" * 64, "path": "/stale/closure.json"}, "closure"),
        "proof-link": ("exact_review_handoff", "qualification_packet", {"sha256": "0" * 64, "path": "/stale/proof.json"}, "qualification"),
        "inventory": ("source_closure", "file_identities", {}, "file"),
    }[mutation]
    bound = a["manifest"]["evidence"][role]
    p = a["evidence"] / bound["path"]
    packet = _json(p)
    packet[field] = value
    _write_json(p, packet)
    # Even when a caller supplies a rehashed packet location, contradictory
    # internal source/closure links must not be reported as verified.
    bound.update(_binding(p.read_bytes()))
    _write_json(a["manifest_path"], a["manifest"])
    _refused(a, diagnostic)


def test_source_diff_bytes_are_bound(artifact):
    a = artifact
    _good(a, output=a["output"].with_name("baseline.json"))
    p = a["evidence"] / "source.diff"
    p.write_bytes(p.read_bytes() + b"tampered patch\n")
    _refused(a, "diff")


def test_output_is_fresh_and_no_clobber(artifact):
    a = artifact
    _good(a)
    published = a["output"].read_bytes()
    result = _invoke(a)
    assert result.returncode != 0, "existing report must not be overwritten"
    assert a["output"].read_bytes() == published
    assert "exist" in (result.stdout + result.stderr).lower()


def test_omitted_bridge_source_never_falls_back_to_delivery_checkout(artifact):
    a = artifact
    _good(a, output=a["output"].with_name("baseline.json"))
    result = subprocess.run([sys.executable, str(TOOL), "--identity-only",
                             "--original-source", str(a["original"]),
                             "--manifest", str(a["manifest_path"]),
                             "--evidence-root", str(a["evidence"]), "--output", str(a["output"])],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "bridge" in (result.stdout + result.stderr).lower()
    assert not a["output"].exists()


def test_versioned_manifest_pins_the_real_retained_lineage_and_regression_evidence():
    assert MANIFEST.is_file(), "#2949: historical immutable source manifest is missing"
    m = _json(MANIFEST)
    expected = _json(RETAINED / "identities.json")
    assert m["schema"] == "pact.catalog_source_manifest.v1"
    assert m["original_source"]["commit"] == expected["commits"]["base"]
    assert m["bridge_source"]["commit"] == expected["commits"]["head"]
    assert m["original_source"]["package_sha256"] == "6b558df3a2746f2a79637fc1d6dc488dc69351c20388b2d8d7212bd3087837c5"
    assert m["bridge_source"]["package_sha256"] == "6bce2eb8c6324b56f09bf9c9e89ac878d21c780bc79b1207d66190984c330d42"
    assert m["accepted_b522_catalog_owner_sha256"] == expected["accepted_owner"]["sha256"]
    assert m["allowed_package_changes"] == OWNERS
    assert m["source_diff"]["paths"] == expected["changed_source_paths"]
    for k in ("sha256", "bytes"):
        assert m["source_diff"][k] == _binding((RETAINED / "source.diff.txt").read_bytes())[k]
    actual = _json(RETAINED / "source-inventories.json")
    for side in ("original", "bridge"):
        expanded = {**m["source_inventory"]["shared"], **m["source_inventory"][side]}
        full = {**actual["shared"], **actual[side]}
        assert {k: {f: v[f] for f in ("sha256", "bytes")} for k, v in expanded.items()} == {
            k: {f: v[f] for f in ("sha256", "bytes")} for k, v in full.items()}
    # Evidence remains tied to its producer and retains both proof populations;
    # do not silently substitute a passing current-main catalog test result.
    for role, filename in {
        "source_closure": "source-closure-7daab5d3.json",
        "package_attestation": "package-attestation-8967a796.json",
        "source_default_control_readiness": "source-default-control-readiness-7daab5d3.json",
        "exact_review_handoff": "exact-review-handoff-8967a796.json",
    }.items():
        bound = _binding((RETAINED / filename).read_bytes())
        for k in ("sha256", "bytes"):
            assert m["evidence"][role][k] == bound[k]
    handoff = _json(RETAINED / "exact-review-handoff-8967a796.json")
    populations = handoff["proof_fixture_populations"]
    assert populations["configured"]["passed"] == 28
    assert populations["configured"]["skipped"] == 0
    assert populations["unconfigured"]["does_not_exercise_real_integration"] is True
    assert populations["unconfigured"]["skipped"] == 18
    regressions = m["historical_regressions"]
    for filename in ("test_catalog_proof_authority.py", "test_joint_catalog_extension.py",
                     "test_joint_stageb_resources.py", "test_joint_quanta_join.py"):
        entry = regressions[filename]
        assert entry["producer_commit"] == expected["commits"]["head"]
        assert entry["sha256"] == _binding((RETAINED / (filename + ".txt")).read_bytes())["sha256"]
    assert m["historical_regressions_are_fresh_acceptance"] is False
