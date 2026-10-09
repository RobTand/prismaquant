"""The replay dependency manifest binds every external runtime import."""

import hashlib
import json
import sys
from pathlib import Path

REPLAY_ROOT = Path(__file__).resolve().parents[1] / "prismaquant" / "pact_replay"
MANIFEST = REPLAY_ROOT / "DEPENDENCY_MANIFEST.json"
PROVENANCE = REPLAY_ROOT / "IMPORT_PROVENANCE.json"


def _manifest():
    return json.loads(MANIFEST.read_bytes())


def test_manifest_is_valid_json_with_schema():
    manifest = _manifest()
    assert manifest["schema"] == "prismaquant.pact_replay_dependency_manifest.v1"
    assert manifest["issue"] == "prismaquant#2482"


def test_manifest_names_all_d38_external_modules():
    manifest = _manifest()
    origins = manifest["resolved_origins"]
    assert len(origins) == 92
    tags = {entry["origin_tag"] for entry in origins.values()}
    assert tags == {"g3-source", "pq-owner", "tessera-pin"}
    assert origins["g3_residency"]["origin_tag"] == "g3-source"
    assert origins["g3_pq_policy.dev_mode"]["origin_tag"] == "g3-source"
    assert origins["g3_readset"]["origin_tag"] == "g3-source"
    assert origins["g3_lib"]["origin_tag"] == "g3-source"
    assert origins["exl3_torch"]["origin_tag"] == "g3-source"
    assert origins["prismaquant.cost_streaming"]["origin_tag"] == "pq-owner"
    assert origins["tessera.unit_artifact"]["origin_tag"] == "tessera-pin"


def test_manifest_marks_evidence_classes():
    manifest = _manifest()
    classes = [entry["evidence_class"] for entry in manifest["resolved_origins"].values()]
    assert classes.count("executed") == 75
    assert classes.count("latent") == 17
    for name, entry in manifest["resolved_origins"].items():
        if entry["evidence_class"] == "latent":
            assert entry["latent_reason"], name


def test_manifest_carries_exact_source_identities():
    manifest = _manifest()
    g3 = manifest["sources"]["g3-source"]
    assert g3["head"] == "a23165b979a90536e060a2c99ea26930d5445a59"
    assert g3["origin_bundle_sha256"] == (
        "8a2468c67ea17cc10e9a7a15d3dcacd5afd86e465e79025aad28357b0b1611ec"
    )
    assert g3["files"]["g3_residency.py"]["git_blob"] == (
        "baebfcfdca38b5d3ef38e817be2783b37d9637e6"
    )
    pq = manifest["sources"]["pq-owner"]
    assert pq["head"] == "931085dbdf9cc321e3e586c6a47cc62f970be978"
    assert pq["files"]["prismaquant/cost_streaming.py"]["git_blob"] == (
        "6faff0ffa117e282b2f4226db3a636134fd9b0b0"
    )
    tess = manifest["sources"]["tessera-pin"]
    assert tess["commit"] == "b40c93cb73745097e57a1ba4cf5b9eee166c759a"
    assert tess["tree_sha256"] == (
        "1e93c8eecff2f2389e2d07647ade6de681d5bd7a9f78292d396c5a986dbc7462"
    )


def test_manifest_binds_d38_evidence():
    manifest = _manifest()
    d38 = manifest["d38_evidence"]
    assert d38["action"] == (
        "25d65391d7ce29c27f4c265396d8810d0ad95d76782396117b7ed3aa2e81cd6d"
    )
    assert d38["snapshot_parent"] == "7672bace107fe09b9907747a9a70ba758af59b32"
    assert d38["proof_schema"] == "pact.same_entry_real_cpu_proof.v2"


def test_provenance_references_manifest_digest():
    provenance = json.loads(PROVENANCE.read_bytes())
    assert provenance["dependency_manifest"] == (
        "prismaquant/pact_replay/DEPENDENCY_MANIFEST.json"
    )
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert provenance["dependency_manifest_sha256"] == digest
    assert provenance["external_pins"]["g3_source_head"] == (
        "a23165b979a90536e060a2c99ea26930d5445a59"
    )
    assert provenance["external_pins"]["tessera_pin_commit"] == (
        "b40c93cb73745097e57a1ba4cf5b9eee166c759a"
    )
    assert provenance["d38_corrected"]["action"] == (
        "25d65391d7ce29c27f4c265396d8810d0ad95d76782396117b7ed3aa2e81cd6d"
    )
