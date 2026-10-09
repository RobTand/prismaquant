"""Qualification package for pq1317 GPU tests is complete."""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "docs" / "measurements" / "pq1317-gpu-tests"


def _load(name):
    return json.loads((PKG / name).read_text(encoding="utf-8"))


def test_candidate_binds_one_exact_identity():
    candidate = _load("candidate.json")
    assert candidate["tessera_commit"] == "fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb"
    assert candidate["contract_sha256"] == "ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e"
    assert candidate["image_digest"].startswith("localhost/prismaquant/spark-vllm-nccl230@sha256:")
    assert len(candidate["serving_native_extensions"]) == 4


def test_corrections_map_to_required_nodes():
    corrections = _load("corrections.json")
    by_issue = {c["issue"]: c for c in corrections["corrections"]}
    assert set(by_issue) == {610, 611}
    for issue in (610, 611):
        assert by_issue[issue]["state"] == "CLOSED"
    node_map = corrections.get("correction_node_map", {})
    assert set(map(str, node_map.keys())) == {"610", "611"}
    for issue in ("610", "611"):
        assert len(node_map[issue]) > 0


def test_roster_lists_exact_required_nodes():
    roster = _load("roster.json")
    nodes = roster.get("required_nodes", [])
    assert len(nodes) > 0
    assert roster.get("owner_verified_required_roster", "MISSING") != "MISSING"
    for node in nodes:
        assert "::" in node


def test_gpu_results_cover_every_required_node():
    roster = _load("roster.json")
    required = set(roster.get("required_nodes", []))
    assert required
    results_dir = PKG / "results"
    assert results_dir.is_dir()
    outcomes = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name in ("summary.json", "candidate-manifest.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for node, record in payload.get("nodes", {}).items():
            outcomes[node] = record
    missing = required - set(outcomes)
    assert not missing, f"missing outcomes for {sorted(missing)[:5]}"
    bad = [n for n, r in outcomes.items() if r.get("outcome") != "passed" and n in required]
    assert not bad, f"required nodes lack pass: {bad[:5]}"
    for node in required:
        record = outcomes[node]
        assert record.get("device_capability") == "sm_121"
        assert "native" in record.get("native_path_proof", "")


def test_package_declares_complete_verdict():
    readme = (PKG / "README.md").read_text(encoding="utf-8")
    assert "Status: complete" in readme
    assert "fca4c6ce0" in readme
