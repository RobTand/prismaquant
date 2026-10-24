"""Exercise the actual retained packet graph without running either producer."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
RETAINED = ROOT / "tests/fixtures/pact_catalog_comparability/retained"
SPEC = importlib.util.spec_from_file_location("catalog_identity_tool", ROOT / "tools/pact_catalog_comparability.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


@pytest.fixture
def retained_graph():
    manifest = json.loads((ROOT / "tools/manifests/pact_catalog_bridge_8967a796.json").read_text())
    sources = {side: manifest[side + "_source"] for side in ("original", "bridge")}
    inventories = {side: {**manifest["source_inventory"]["shared"], **manifest["source_inventory"][side]}
                   for side in sources}
    packages = {side: {k.removeprefix("prismaquant/"): v for k, v in inv.items()
                       if k.startswith("prismaquant/")} for side, inv in inventories.items()}
    files = {k: {side: packages[side][k] for side in sources} for k in packages["original"]}
    catalog = packages["bridge"]["joint_catalog_extension.py"]
    return manifest, sources, packages, files, catalog


def test_actual_retained_graph_preserves_companion_and_final_producers(retained_graph):
    m, sources, packages, files, catalog = retained_graph
    bindings, dependencies = TOOL.verify_packets(m, RETAINED, sources, packages, files, catalog)
    assert bindings["source_closure"]["producer_commit"] == bindings["source_default_control_readiness"]["producer_commit"]
    assert bindings["source_closure"]["producer_commit"] != sources["bridge"]["commit"]
    assert bindings["package_attestation"]["producer_commit"] == sources["bridge"]["commit"]
    assert bindings["exact_review_handoff"]["producer_commit"] == sources["bridge"]["commit"]
    assert dependencies["qualification_packet"]["verification"] == "retained_reference_only_not_observed"
    # All eight inert snapshots bind final producer source, not current main's
    # policy or fresh regression counts.
    assert len(m["historical_regressions"]) == 8
    for name, entry in m["historical_regressions"].items():
        path = TOOL.evidence_file(RETAINED, entry, name)
        assert TOOL.identity(path) == m["source_inventory"]["bridge"]["tests/" + name]


@pytest.mark.parametrize("claim", ["closure-head", "readiness-head", "accepted-head", "external-qualification-link"])
def test_rehashed_retained_claim_cannot_conflate_producers_or_external_links(retained_graph, tmp_path, claim):
    m, sources, packages, files, catalog = retained_graph
    for entry in m["evidence"].values():
        shutil.copyfile(RETAINED / entry["path"], tmp_path / entry["path"])
    role = {"closure-head": "source_closure", "readiness-head": "source_default_control_readiness",
            "accepted-head": "source_default_control_readiness",
            "external-qualification-link": "exact_review_handoff"}[claim]
    entry = m["evidence"][role]
    path = tmp_path / entry["path"]
    packet = json.loads(path.read_text())
    if claim == "closure-head":
        packet["bridge_source"]["commit"] = sources["bridge"]["commit"]
    elif claim == "readiness-head":
        packet["head_sha"] = sources["bridge"]["commit"]
    elif claim == "accepted-head":
        packet["accepted_b522"]["head"] = sources["bridge"]["commit"]
    else:
        packet["qualification_packet"]["sha256"] = "0" * 64
    path.write_text(json.dumps(packet))
    entry.update(TOOL.identity(path))
    with pytest.raises(ValueError, match="commit|head|qualification"):
        TOOL.verify_packets(m, tmp_path, sources, packages, files, catalog)
