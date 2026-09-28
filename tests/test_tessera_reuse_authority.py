"""PrismaQuant's reuse authority: its schemas, and Tessera's reader running it.

Tessera names no PrismaQuant record (tessera#599). These tests moved here
from Tessera's ``tests/test_rooted_cached_bundle.py`` and
``tests/test_cached_unit_reader_gamut.py`` with the schema roster they pinned,
and they run the real ``tessera.cached_unit.CachedUnitBundle`` with
``prismaquant.tessera_reuse_authority.PRODUCER_AUTHORITY``.
"""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from prismaquant import joint_catalog_extension as extension
from prismaquant import joint_served_activation as served_policy
from prismaquant import tessera_calibration_cache as calibration_cache
from prismaquant import tessera_reuse_authority as authority_module
from prismaquant.tessera_reuse_authority import CANONICAL_CAPTURE, PRODUCER_AUTHORITY

cached_unit = pytest.importorskip("tessera.cached_unit")
ROOT = Path(__file__).resolve().parents[1]


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- the schemas are the writers' ------------------------------------------------

def test_each_schema_is_the_one_its_writer_writes():
    assert authority_module.CATALOG_EXTENSION_SCHEMAS == {
        extension.SCHEMA_V1, extension.SCHEMA, extension.SCHEMA_V3}
    # The T4 overlay reader was v1 only in Tessera; it moved as is.
    assert authority_module.CANDIDATE_OVERLAY_SCHEMAS == {extension.CATALOG_SCHEMA_V1}
    assert authority_module.ADOPTION_SCHEMA == extension.ADOPTION_SCHEMA
    assert authority_module.RESEAL_PROOF_SCHEMA == _tool("reseal_campaign_identity").BUNDLE_SCHEMA
    assert authority_module.SERVED_POLICY_SCHEMA_V1 == served_policy.SCHEMA
    assert authority_module.SERVED_POLICY_SCHEMA_V2 == served_policy.SCHEMA_V2
    assert CANONICAL_CAPTURE == (calibration_cache.SCHEMA, calibration_cache.SOURCE)


def test_the_authority_file_stands_alone_for_the_exporter():
    """Tessera's exporter loads it by path, where ``prismaquant`` may not import."""
    path = authority_module.AUTHORITY_PATH
    assert path == ROOT / "prismaquant" / "tessera_reuse_authority.py"
    imported = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import cannot load by path"
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "pathlib", "re"}
    spec = importlib.util.spec_from_file_location("standalone_reuse_authority", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module.PRODUCER_AUTHORITY, cached_unit.ReuseAuthority)
    assert module.PRODUCER_AUTHORITY.canonical_hessian_capture == CANONICAL_CAPTURE


def test_run_pipeline_hands_the_exporter_this_authority():
    # Gated on the pinned checkout's contract; the gate itself is held by
    # tests/test_tessera_export_authority_gate.py.
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    call = script[:script.index('python3 "${TESSERA_REPO%/}/experiments/export_tessera_serving.py"')]
    call = call[call.rindex("TESSERA_AUTHORITY_LINES=$("):]
    assert '"${PIPELINE_SCRIPT_DIR}/tessera_reuse_authority.py"' in call


# -- a rooted bundle of PrismaQuant's records ------------------------------------

def _bound(tmp_path, name, document):
    path = tmp_path / (name + ".json")
    path.write_text(json.dumps(document))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def rooted(tmp_path, extension_schema=extension.SCHEMA_V1):
    roots = {key: tmp_path / key for key in ("old", "added")}
    for path in roots.values():
        path.mkdir()
        (path / "unit.tessera").write_bytes(b"unchanged wire")
    bound = {"catalog_extension": _bound(tmp_path, "catalog_extension", {"schema": extension_schema}),
             "candidate_overlay": _bound(tmp_path, "candidate_overlay",
                                         {"schema": extension.CATALOG_SCHEMA_V1})}
    units = {name: {"file": "unit.tessera", "blob_bytes": 14,
                    "blob_sha256": hashlib.sha256(b"unchanged wire").hexdigest(),
                    "identity": {"unit": name, "encoder_source_sha256": seal,
                                 "source": {"sha256": "c" * 64}, "calibration": None,
                                 "encoder_fixture_id": "f" * 64}}
             for name, seal in [("dense", "a" * 64), ("expert", "b" * 64)]}
    manifest = {"schema": "tessera.cached_units.v2", "source": {"sha256": "source"},
                "units": units, "wire_roots": {k: str(v) for k, v in roots.items()},
                "unit_roots": {"dense": "old", "expert": "added"},
                "producer_packages": {seal: {"path": str(tmp_path / ("producer-" + seal[0])),
                                             "sha256": seal} for seal in ("a" * 64, "b" * 64)},
                "reuse_authority": {**bound, "checkpoint_encoder_source_sha256": "a" * 64,
                                    "encoder_source_proofs": []},
                "encoder_adoptions": {}, "served_activation_policy": None, "served_activations": {}}
    proof_bound = _bound(tmp_path, "encoder-proof", {
        "schema": authority_module.RESEAL_PROOF_SCHEMA, "ok": True, "encoder_fixture_id_equal": True,
        "pins": {"old": {"encoder_source_sha256": "a" * 64},
                 "new": {"encoder_source_sha256": "b" * 64}},
        "fixture_id": {"ids": {"old": "f" * 64, "new": "f" * 64}}})
    manifest["reuse_authority"]["encoder_source_proofs"] = [proof_bound]
    candidate = units["expert"]["identity"]
    manifest["encoder_adoptions"]["expert"] = {
        "schema": extension.ADOPTION_SCHEMA, "reference_pair": ["expert", "old-format"],
        "reference_encoding_identity": {**candidate, "encoder_source_sha256": "a" * 64},
        "candidate_encoding_identity": candidate, "encoder_source_proof": proof_bound}
    return manifest


def load(manifest, tmp_path, mode="strict", units=("dense", "expert"), **kwargs):
    return cached_unit.CachedUnitBundle(manifest, tmp_path, set(units), manifest["source"],
                                        encoder_source_proof_mode=mode,
                                        authority=PRODUCER_AUTHORITY, **kwargs)


@pytest.mark.parametrize("mode", ["strict", "permissive"])
@pytest.mark.parametrize("schema", sorted(authority_module.CATALOG_EXTENSION_SCHEMAS))
def test_each_catalog_extension_version_is_read(tmp_path, schema, mode):
    manifest = rooted(tmp_path, extension_schema=schema)
    bundle = load(manifest, tmp_path, mode)
    assert bundle.warnings == []
    for name in manifest["units"]:
        assert bundle.read(name)[1] == manifest["units"][name]


@pytest.mark.parametrize("schema", ["prismaquant.joint_catalog_extension.v4",
                                    extension.CATALOG_SCHEMA_V1, None, [extension.SCHEMA]])
def test_any_other_catalog_extension_refuses(tmp_path, schema):
    with pytest.raises(ValueError, match="authority schema differs"):
        load(rooted(tmp_path, extension_schema=schema), tmp_path)


def test_without_the_authority_tessera_refuses_by_name(tmp_path):
    manifest = rooted(tmp_path)
    with pytest.raises(ValueError, match="need a producer reuse authority"):
        cached_unit.CachedUnitBundle(manifest, tmp_path, {"dense", "expert"}, manifest["source"])


@pytest.mark.parametrize("field", ["unit", "source", "calibration", "encoder_fixture_id", "projection"])
def test_an_adoption_that_changes_identity_refuses_in_either_mode(tmp_path, field):
    manifest = rooted(tmp_path)
    manifest["encoder_adoptions"]["expert"]["reference_encoding_identity"][field] = {"changed": True}
    for mode in ("strict", "permissive"):
        with pytest.raises(ValueError, match="identities differ|changed " + field):
            load(manifest, tmp_path, mode)


def test_an_adoption_of_another_schema_refuses(tmp_path):
    manifest = rooted(tmp_path)
    manifest["encoder_adoptions"]["expert"]["schema"] = "prismaquant.joint_catalog_source_adoption.v2"
    with pytest.raises(ValueError, match="adoption schema differs"):
        load(manifest, tmp_path)


@pytest.mark.parametrize("proof_state", ["absent", "unlisted", "wrong_pins", "failed", "other_schema"])
def test_a_proof_that_does_not_authorize_warns_only_when_permissive(tmp_path, proof_state):
    manifest = rooted(tmp_path)
    adoption = manifest["encoder_adoptions"]["expert"]
    if proof_state in ("absent", "unlisted"):
        manifest["reuse_authority"]["encoder_source_proofs"] = []
        if proof_state == "absent":
            adoption["encoder_source_proof"] = None
    else:
        document = json.loads(Path(adoption["encoder_source_proof"]["path"]).read_text())
        if proof_state == "wrong_pins":
            document["pins"]["new"]["encoder_source_sha256"] = "0" * 64
        elif proof_state == "failed":
            document["ok"] = False
        else:
            document["schema"] = "prismaquant.reseal_proof_bundle.v2"
        proof = _bound(tmp_path, "altered-proof", document)
        manifest["reuse_authority"]["encoder_source_proofs"] = [proof]
        adoption["encoder_source_proof"] = proof
    bundle = load(manifest, tmp_path, "permissive")
    assert [warning["proof_status"] for warning in bundle.warnings] == [
        proof_state if proof_state in ("absent", "unlisted") else "not_authorizing"]
    with pytest.raises(ValueError, match="proof does not authorize"):
        load(manifest, tmp_path, "strict")


# -- the served activation policy ------------------------------------------------

def policy_fixture(tmp_path, version):
    manifest = rooted(tmp_path)
    rates = [640, 768, 896, 1152]
    template = copy.deepcopy(manifest["units"]["expert"])
    adoption = copy.deepcopy(manifest["encoder_adoptions"]["expert"])
    for key in ("units", "unit_roots", "encoder_adoptions"):
        manifest[key].pop("expert")
    for rate in rates:
        name = f"expert{rate}"
        record = copy.deepcopy(template)
        record["file"] = name + ".tessera"
        record["identity"].update(unit=name, recipe={"grid": "E2M1x2", "q256": rate})
        manifest["units"][name] = record
        manifest["unit_roots"][name] = "added"
        entry = copy.deepcopy(adoption)
        entry["reference_pair"][0] = name
        entry["reference_encoding_identity"]["unit"] = name
        entry["candidate_encoding_identity"] = record["identity"]
        manifest["encoder_adoptions"][name] = entry
    names = set(manifest["encoder_adoptions"])
    formats = [f"TESSERA_E2M1_K2_R{rate}" for rate in (640, 768, 1152)]
    policy = {"schema": (served_policy.SCHEMA if version == 1 else served_policy.SCHEMA_V2),
              **({"format": "TESSERA_E2M1_K2_R896"} if version == 1 else {"formats": sorted(formats)}),
              "executed_grouping": {"groups": {"group": {"members": sorted(names), "input_global_scale": 0.5}}}}
    chosen = {896} if version == 1 else {640, 768, 1152}
    manifest["served_activation_policy"] = _bound(tmp_path, "policy", policy)
    manifest["served_activations"] = {f"expert{rate}": {"group": "group", "input_global_scale": 0.5}
                                      for rate in chosen}
    return manifest, policy


@pytest.mark.parametrize("version", [1, 2])
def test_policy_v1_keeps_its_single_rung_and_v2_reads_its_formats(tmp_path, version):
    manifest, _ = policy_fixture(tmp_path, version)
    bundle = load(manifest, tmp_path, units=manifest["units"])
    assert bundle.warnings == []
    assert bundle.served_activations == manifest["served_activations"]


@pytest.mark.parametrize("damage", ["missing", "extra", "member"])
def test_policy_coverage_stays_exact(tmp_path, damage):
    manifest, policy = policy_fixture(tmp_path, 2)
    if damage == "missing":
        del manifest["served_activations"]["expert640"]
    elif damage == "extra":
        manifest["served_activations"]["expert896"] = {"group": "group", "input_global_scale": 0.5}
    else:
        policy["executed_grouping"]["groups"]["group"]["members"].remove("expert640")
        manifest["served_activation_policy"] = _bound(tmp_path, "policy", policy)
    with pytest.raises(ValueError, match="served activations differ|absent from served activation policy"):
        load(manifest, tmp_path, "permissive", units=manifest["units"])


@pytest.mark.parametrize("formats", [None, [], ["TESSERA_E2M1_K2_R768"] * 2,
                                     ["TESSERA_E2M1_K2_R896", "TESSERA_E2M1_K2_R640"],
                                     [None], ["TESSERA_E4M3_K1_R1024"], ["TESSERA_E2M1_K2_R0768"]])
def test_policy_v2_refuses_a_malformed_format_scope(tmp_path, formats):
    manifest, policy = policy_fixture(tmp_path, 2)
    policy["formats"] = formats
    manifest["served_activation_policy"] = _bound(tmp_path, "policy", policy)
    with pytest.raises(ValueError, match="policy schema differs"):
        load(manifest, tmp_path, units=manifest["units"])


def test_policy_v1_of_another_rung_refuses(tmp_path):
    manifest, policy = policy_fixture(tmp_path, 1)
    policy["format"] = "TESSERA_E2M1_K2_R768"
    manifest["served_activation_policy"] = _bound(tmp_path, "policy", policy)
    with pytest.raises(ValueError, match="policy schema differs"):
        load(manifest, tmp_path, units=manifest["units"])


# -- the canonical calibration cache a Hessian reference binds -------------------

def test_a_reference_to_our_capture_opens_only_with_our_canonical_capture(tmp_path):
    torch = pytest.importorskip("torch")
    from tessera.errors import GrammarError
    from tessera.export import ActivationSource
    from tessera.hessian_capture import ReferenceHessians
    from tessera.cached_unit import tensor_identity

    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    H = {"a": torch.eye(4) * 3}
    provenance = {"text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64, "fit_tokens": 8,
                  "model": "fixture", "seqlen": 8, "source": "fixture", "hessian_role": "fit"}
    census = tmp_path / "census.json"
    census_sha = write(census, {"unit_shapes": {"a": [4, 4]}, "counts": {"a": 8}, "max_abs": {"a": 1.0}})
    root = tmp_path / "capture"
    (root / "inputs").mkdir(parents=True)
    torch.save({"inputs": torch.ones(2, 4), "hessian": H["a"], "name": "a",
                "source": calibration_cache.SOURCE, "count": 8, "max_abs": 1.0}, root / "inputs" / "a.pt")
    canonical = root / "capture_manifest.json"
    canonical_sha = write(canonical, {
        "schema": calibration_cache.SCHEMA, "status": "complete",
        "identity": {"schema": calibration_cache.SCHEMA, "census_sha256": census_sha,
                     "units": {"a": [4, 4]}, "storage_source": calibration_cache.SOURCE,
                     "max_act_rows": 2,
                     "calibration": {k: v for k, v in provenance.items() if k != "hessian_role"}},
        "entries": {"a": {"path": "inputs/a.pt", "sha256": hashlib.sha256(
            (root / "inputs" / "a.pt").read_bytes()).hexdigest()}}})
    digest = ActivationSource(H, provenance).capture_sha256()
    handoff = tmp_path / "hessian_capture.references.json"
    write(handoff, {"schema": "tessera.hessian_capture.references.v1",
                    "canonical_capture": {"path": str(canonical), "sha256": canonical_sha},
                    "census": {"path": str(census), "sha256": census_sha},
                    "provenance": provenance, "counts": {"a": 8},
                    "hessians": {"a": tensor_identity(H["a"])}, "capture_sha256": digest,
                    "rows": [{"units": ["a"], "capture_sha256": digest}],
                    "load_policy": {"schema": "tessera.hessian_reference_load.v1",
                                    "max_metadata_bytes": 1024 ** 2, "max_file_bytes": 1024 ** 2,
                                    "max_hessian_bytes": 1024 ** 2}})
    with ReferenceHessians(handoff, canonical_capture=CANONICAL_CAPTURE) as owner:
        assert torch.equal(owner["a"], H["a"])
    with pytest.raises(GrammarError, match="need the producer canonical capture"):
        ReferenceHessians(handoff)
