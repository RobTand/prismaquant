"""PQ #884: the control-metadata generation seam.

Root decision (2026-09-21, Astra brief): the pending stage-A recovery
generation is pre-#852 and cannot pass current-code Gate 1a exact
reproduction; the sanctioned path is a NEW immutable control-metadata
generation derived from the same primary sealed inputs under an explicit
metadata namespace, while every reference to actual run/stage-A artifacts
is retained. These tests prove the seam on tiny CPU fixtures built by the
REAL producer, the REAL regen CLI and the REAL receipt writers:

* the behavioral regression: default-layout publication beside an
  immutable prior generation with drifted control bytes refuses
  (first-writer), which is why the seam exists;
* the seam itself: control placement (slice manifests, record paths, bound
  readsets) moves to the metadata namespace; data fields (output_space,
  adjoint.boundary_artifacts, campaign identity, membership, extents) never
  move; the default layout is byte-identical when the flag is absent;
* the relocation gate over an EXACTLY reproduced source generation, and the
  prior-generation scientific comparison that certifies only the known
  control drift (the one zero-byte head phase row, the exact record-path
  relocation) and refuses everything else;
* receipt/readset binding through the real validators under the metadata
  namespace, with the complete static prepared-input contract accepted by
  the dispatcher. Sequencing-only rows remain unsupported.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import regenerate_joint_quanta as regen  # noqa: E402
from prismaquant.joint_layer_quanta import (  # noqa: E402
    bind_quantum_boundary_readset,
    canonical_sha256,
    layer_quanta,
    seal_manifest_bytes,
)


def _reseal_record_identity(record: dict) -> None:
    """Reseal a tampered prior record's identity so the prior set stays
    internally consistent -- the honest shape of a real drifted generation,
    and the only shape that exercises the comparison's own drift gates
    rather than its identity gate."""
    body = {key: value for key, value in record.items()
            if key != "identity_sha256"}
    record["identity_sha256"] = canonical_sha256(
        body, where=f"quantum record {record.get('quantum_id')}")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tiny_campaign(tmp_path: Path) -> dict:
    """Two-layer plan + prepared + parent files the real producer binds."""
    from test_stageb_prepared_inputs_bridge import (
        _execution, _render_files, _production_pkl, _qnames, FMT)
    files = _render_files(tmp_path)
    pkl_path = _production_pkl(tmp_path, files)
    root = tmp_path / "campaign"
    calib = tmp_path / "calib.bin"
    calib.write_bytes(b"calibration-bytes")
    plan = {"output_root": str(root), "model": "/fixture/model",
            "distributed_campaign": {},
            "execution": _execution(),
            "calibration_input": {"path": str(calib),
                                  "sha256": "a" * 64}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    prepared = {"formats_by_qname": {
        name: [FMT] for name in _qnames(0) + _qnames(1)[:1]},
        "production_cache": {"path": str(pkl_path),
                             "sha256": hashlib.sha256(pkl_path.read_bytes()).hexdigest()}}
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps(prepared, sort_keys=True))
    prepared_sha = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
    entries = [
        {"path": "/fixture/model/shard-h0.pt", "offset": 0, "bytes": 100,
         "sha256": None},
        {"path": "/fixture/model/shard-h1.pt", "offset": 0, "bytes": 100,
         "sha256": None},
        {"path": "/fixture/model/shard-l0a.pt", "offset": 0, "bytes": 300,
         "sha256": None},
        {"path": "/fixture/model/shard-l0b.pt", "offset": 0, "bytes": 300,
         "sha256": None},
        {"path": "/fixture/model/shard-l0c.pt", "offset": 0, "bytes": 300,
         "sha256": None},
        {"path": "/fixture/model/shard-l1.pt", "offset": 0, "bytes": 300,
         "sha256": None},
    ]
    parent = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "produced_by": {"plan": str(plan_path)},
        "mount_prefix": "/mnt/shared",
        "entries": entries, "entry_count": 6, "total_bytes": 1400,
        "annotations": {
            "campaign_scope": {"campaign": "metadata-seam-884"},
            "argv": ["python3", "-m", "prismaquant.joint_adjoint_capture",
                     "--prepared", str(prepared_path),
                     "--prepared-sha256", prepared_sha],
            "layers": [0, 1],
            "phases": [
                {"name": "head", "bytes": 200, "cumulative_bytes": 200},
                {"name": "layer-0", "bytes": 900, "cumulative_bytes": 1100},
                {"name": "layer-1", "bytes": 300, "cumulative_bytes": 1400},
            ],
        },
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent, sort_keys=True))
    return {"root": root, "plan_path": plan_path, "plan_sha": _sha(plan_path),
            "prepared_path": prepared_path, "prepared_sha": prepared_sha,
            "parent_path": parent_path, "parent_sha": _sha(parent_path)}


def _regen_argv(tmp_path: Path, campaign: dict) -> list[str]:
    derivation = tmp_path / "derivation.json"
    derivation.write_text(json.dumps(
        {"chunk_target_bytes": 400, "stride": 1, "ram_window_gib": 160,
         "max_resident_consumers": 2}, sort_keys=True))
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps(
        {"windows_by_layer": {"0": 2, "1": 1}}, sort_keys=True))
    return ["--plan", str(campaign["plan_path"]),
            "--plan-sha256", campaign["plan_sha"],
            "--prepared", str(campaign["prepared_path"]),
            "--prepared-sha256", campaign["prepared_sha"],
            "--parent-manifest", str(campaign["parent_path"]),
            "--parent-manifest-sha256", campaign["parent_sha"],
            "--derivation", str(derivation),
            "--partition", str(partition)]


def _produce(campaign: dict, *, receipt=None, metadata_root=None) -> dict:
    return layer_quanta(
        {"output_root": str(campaign["root"]), "model": "/fixture/model",
         "distributed_campaign": {}},
        json.loads(campaign["prepared_path"].read_text()),
        json.loads(campaign["parent_path"].read_text()),
        chunk_target_bytes=400, stride=1,
        output_root=str(campaign["root"]),
        metadata_root=metadata_root,
        plan_path=str(campaign["plan_path"]), plan_sha256=campaign["plan_sha"],
        prepared_path=str(campaign["prepared_path"]),
        prepared_sha256=campaign["prepared_sha"],
        parent_manifest_sha256=campaign["parent_sha"],
        window_partition={"windows_by_layer": {"0": 2, "1": 1}},
        adjoint_receipt=receipt)


def _publish_default_generation(campaign: dict, records_out: Path) -> None:
    """Publish a prior generation at the DEFAULT layout through the CLI."""
    assert regen.main(
        _regen_argv(Path(campaign["plan_path"]).parent, campaign)
        + ["--output-root", str(campaign["root"]),
           "--records-out", str(records_out)]) == 0


def _snapshot(tree: Path) -> dict[str, bytes]:
    return {str(path.relative_to(tree)): path.read_bytes()
            for path in sorted(tree.rglob("*")) if path.is_file()}


def _reseal_manifest(manifest: dict) -> bytes:
    return seal_manifest_bytes(manifest)


def _age_prior_generation(campaign: dict, records_dir: Path,
                          *, mutate=None) -> None:
    """Rewrite the prior generation's sealed slice manifests to the pre-#852
    control shape (one zero-byte head phase row) -- and optionally apply a
    caller mutation -- updating each prior record's sealed digest so the
    prior set stays internally consistent but no longer reproduces under
    the current producer. This is the campaign's immutable prior
    generation, miniaturized."""
    for record_path in sorted(records_dir.glob("layer-*.json")):
        record = json.loads(record_path.read_text())
        slice_path = Path(record["read_set"]["manifest_path"])
        manifest = json.loads(
            gzip.decompress(slice_path.read_bytes()).decode("utf-8"))
        phases = manifest["annotations"]["phases"]
        manifest["annotations"]["phases"] = [
            {"name": "head", "bytes": 0, "cumulative_bytes": 0},
            *[{"name": phase["name"], "bytes": phase["bytes"],
               "cumulative_bytes": phase["cumulative_bytes"]}
              for phase in phases]]
        if mutate is not None:
            mutate(manifest, record)
        wire = _reseal_manifest(manifest)
        slice_path.write_bytes(wire)
        record["read_set"]["manifest_sha256"] = \
            hashlib.sha256(wire).hexdigest()
        _reseal_record_identity(record)
        record_path.write_text(json.dumps(record, indent=1, sort_keys=True)
                               + "\n")


# ---------------------------------------------------------------------------
# The behavioral regression: why the seam exists
# ---------------------------------------------------------------------------

def test_default_layout_publication_refuses_beside_immutable_prior(tmp_path,
                                                                   capsys):
    """At the SAME data output root, a new default-layout generation would
    have to re-seal the slice manifests at the existing producer-named
    paths; beside an immutable prior generation whose control bytes differ
    (any prior producer revision), first-writer publication refuses. This
    is the current producer's inability to publish new control metadata
    beside a prior generation while retaining the original output/adjoint
    roots -- a behavioral refusal, not a missing import."""
    campaign = _tiny_campaign(tmp_path)
    prior = tmp_path / "prior-records"
    _publish_default_generation(campaign, prior)
    _age_prior_generation(campaign, prior)
    before = _snapshot(campaign["root"])
    out = tmp_path / "new-records"
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--output-root", str(campaign["root"]),
           "--records-out", str(out)]) == 3
    assert "refusing to overwrite differing bytes" in capsys.readouterr().err
    # The refusal published nothing and touched nothing.
    assert not out.exists()
    assert _snapshot(campaign["root"]) == before


# ---------------------------------------------------------------------------
# The producer seam
# ---------------------------------------------------------------------------

def test_producer_metadata_root_moves_control_only(tmp_path):
    campaign = _tiny_campaign(tmp_path)
    meta = tmp_path / "meta-gen" / "layer-quanta"
    default = _produce(campaign)
    moved = _produce(campaign, metadata_root=str(meta))
    for old, new in zip(default["records"], moved["records"]):
        qid = new["quantum_id"]
        # Data fields never move.
        assert new["output_space"] == old["output_space"]
        assert new["adjoint"]["boundary_artifacts"] == \
            old["adjoint"]["boundary_artifacts"]
        assert new["output_space"]["root"].startswith(str(campaign["root"]))
        # Scientific fields are identical.
        assert new["campaign"] == old["campaign"]
        assert new["chunks"] == old["chunks"]
        assert new["windows"] == old["windows"]
        old_read = dict(old["read_set"])
        new_read = dict(new["read_set"])
        del old_read["manifest_path"], new_read["manifest_path"]
        del old_read["manifest_sha256"], new_read["manifest_sha256"]
        assert new_read == old_read
        # Control placement moves, and the identity moves with it.
        assert new["read_set"]["manifest_path"] == \
            f"{meta}/manifests/{qid}.data-manifest.json.gz"
        assert new["identity_sha256"] != old["identity_sha256"]
        # The slice argv relocates exactly the record path; --output-root
        # still names the data root.
        argv = moved["slice_manifests"][qid]["annotations"]["argv"]
        assert argv[argv.index("--quantum") + 1] == \
            f"{meta}/records/{qid}.json"
        assert argv[argv.index("--output-root") + 1] == str(campaign["root"])
    # The slice wire bytes change only through the argv relocation.
    for qid in moved["slice_manifests"]:
        old_manifest = default["slice_manifests"][qid]
        new_manifest = moved["slice_manifests"][qid]
        old_manifest = json.loads(json.dumps(old_manifest))
        new_manifest = json.loads(json.dumps(new_manifest))
        old_argv = old_manifest["annotations"]["argv"]
        new_argv = new_manifest["annotations"]["argv"]
        del old_manifest["annotations"]["argv"]
        del new_manifest["annotations"]["argv"]
        assert old_manifest == new_manifest
        assert len(old_argv) == len(new_argv)
        for index, (a, b) in enumerate(zip(old_argv, new_argv)):
            if a != b:
                assert old_argv[index - 1] == "--quantum"
    # Derivation provenance: present only for the explicit namespace.
    assert "control_metadata_root" not in default["derivation"]
    assert moved["derivation"]["control_metadata_root"] == str(meta)


def test_producer_metadata_root_equal_to_default_is_identity(tmp_path):
    """metadata_root == {output_root}/layer-quanta reproduces the default
    layout exactly -- records, derivation and slice wire bytes. The control
    layout is one derivation, not two spellings."""
    campaign = _tiny_campaign(tmp_path)
    default = _produce(campaign)
    same = _produce(campaign,
                    metadata_root=f"{campaign['root']}/layer-quanta")
    assert json.dumps(default["records"], sort_keys=True) == \
        json.dumps(same["records"], sort_keys=True)
    assert json.dumps(default["derivation"], sort_keys=True) != \
        json.dumps(same["derivation"], sort_keys=True)  # provenance records it
    for qid in default["slice_manifests"]:
        assert seal_manifest_bytes(default["slice_manifests"][qid]) == \
            seal_manifest_bytes(same["slice_manifests"][qid])


def test_producer_refuses_noncanonical_metadata_root(tmp_path):
    campaign = _tiny_campaign(tmp_path)
    for bad in ("relative/root", "/a/../b", "/a//b", "/"):
        with pytest.raises(ValueError, match="metadata_root|canonical"):
            _produce(campaign, metadata_root=bad)


# ---------------------------------------------------------------------------
# The CLI seam
# ---------------------------------------------------------------------------

def test_cli_metadata_generation_roundtrip_replay_and_refusals(
        tmp_path, capsys):
    campaign = _tiny_campaign(tmp_path)
    prior = tmp_path / "prior-records"
    _publish_default_generation(campaign, prior)
    prior_tree = _snapshot(campaign["root"])
    meta = tmp_path / "meta-gen" / "layer-quanta"
    base = _regen_argv(tmp_path, campaign) + ["--metadata-root", str(meta)]
    # records-out defaults to {metadata_root}/records.
    assert regen.main(base) == 0
    records = sorted((meta / "records").glob("layer-*.json"))
    assert len(records) == 2
    assert (meta / "records" / "records.json").is_file()
    assert (meta / "records" / "derivation.json").is_file()
    for record_path in records:
        record = json.loads(record_path.read_text())
        assert record["output_space"]["root"].startswith(str(campaign["root"]))
        slice_path = Path(record["read_set"]["manifest_path"])
        assert slice_path == (meta / "manifests" /
                              f"{record['quantum_id']}"
                              ".data-manifest.json.gz")
        assert slice_path.is_file()
        assert _sha(slice_path) == record["read_set"]["manifest_sha256"]
    # The prior generation is untouched, byte for byte.
    assert _snapshot(campaign["root"]) == prior_tree
    # Same-bytes replay is idempotent.
    first = _snapshot(meta)
    assert regen.main(base) == 0
    assert _snapshot(meta) == first
    assert _snapshot(campaign["root"]) == prior_tree
    # Differing bytes at a metadata path refuse.
    (meta / "records" / "layer-000.json").write_text("{}")
    assert regen.main(base) == 3
    assert "refusing to overwrite differing bytes" in capsys.readouterr().err
    # A noncanonical metadata root refuses by name.
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--metadata-root", "relative/root"]) == 3
    # Default layout is unchanged when the flag is absent.
    (tmp_path / "fresh").mkdir()
    fresh = _tiny_campaign(tmp_path / "fresh")
    out = tmp_path / "fresh-records"
    assert regen.main(_regen_argv(tmp_path / "fresh", fresh)
                      + ["--output-root", str(fresh["root"]),
                         "--records-out", str(out)]) == 0
    record = json.loads((out / "layer-000.json").read_text())
    assert record["read_set"]["manifest_path"].startswith(
        str(fresh["root"] / "layer-quanta" / "manifests"))


def test_cli_check_only_writes_nothing(tmp_path):
    campaign = _tiny_campaign(tmp_path)
    meta = tmp_path / "meta-gen" / "layer-quanta"
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--metadata-root", str(meta), "--check-only"]) == 0
    assert not meta.exists()
    assert not (campaign["root"]).exists()


def test_cli_relocation_over_exactly_reproduced_source(tmp_path, capsys):
    campaign = _tiny_campaign(tmp_path)
    prior = tmp_path / "prior-records"
    _publish_default_generation(campaign, prior)
    meta = tmp_path / "meta-gen" / "layer-quanta"
    base = (_regen_argv(tmp_path, campaign)
            + ["--metadata-root", str(meta),
               "--expect-existing", str(prior),
               "--original-root", str(campaign["root"])])
    # Dry run: reproduce exactly, then validate the relocation; no writes.
    assert regen.main(base + ["--check-only"]) == 0
    assert not meta.exists()
    # Write mode publishes the relocated control metadata.
    assert regen.main(base) == 0
    moved = json.loads((meta / "records" / "layer-000.json").read_text())
    assert moved["read_set"]["manifest_path"].startswith(str(meta))
    assert moved["output_space"]["root"].startswith(str(campaign["root"]))
    # The prior generation is untouched.
    assert json.loads((prior / "layer-000.json").read_text())[
        "read_set"]["manifest_path"].startswith(str(campaign["root"]))
    # A data-root move combined with the relocation refuses by name.
    assert regen.main(base + ["--output-root",
                              str(tmp_path / "other-root")]) == 3
    assert "relocation over an exactly reproduced source" in \
        capsys.readouterr().err
    # An aged (non-reproducible) prior generation still fails Gate 1a:
    # the relocation never weakens exact reproduction.
    _age_prior_generation(campaign, prior)
    meta2 = tmp_path / "meta-gen-2" / "layer-quanta"
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--metadata-root", str(meta2),
           "--expect-existing", str(prior),
           "--original-root", str(campaign["root"]),
           "--check-only"]) == 3
    assert not meta2.exists()


# ---------------------------------------------------------------------------
# The prior-generation scientific comparison
# ---------------------------------------------------------------------------

def _aged_campaign(tmp_path):
    campaign = _tiny_campaign(tmp_path)
    prior = tmp_path / "prior-records"
    _publish_default_generation(campaign, prior)
    _age_prior_generation(campaign, prior)
    return campaign, prior


def test_compare_existing_certifies_only_known_drift(tmp_path, capsys):
    campaign, prior = _aged_campaign(tmp_path)
    meta = tmp_path / "meta-gen" / "layer-quanta"
    argv = (_regen_argv(tmp_path, campaign)
            + ["--metadata-root", str(meta),
               "--compare-existing", str(prior), "--check-only"])
    assert regen.main(argv) == 0
    out = capsys.readouterr().out
    assert "Compare-existing: 2/2 quanta scientifically identical" in out
    assert "windows_total=3" in out
    assert "manifest placement moved for 2" in out
    assert "record path relocated for 2" in out
    assert "zero-byte head rows dropped 2" in out
    assert "receipts newly bound 0" in out
    assert "wrote nothing" in out
    assert not meta.exists()


def test_compare_existing_default_layout_reports_digest_drift(
        tmp_path, capsys):
    campaign, prior = _aged_campaign(tmp_path)
    argv = (_regen_argv(tmp_path, campaign)
            + ["--compare-existing", str(prior), "--check-only"])
    assert regen.main(argv) == 0
    out = capsys.readouterr().out
    assert "manifest placement moved for 2" in out
    assert "record path relocated for 0" in out
    assert "zero-byte head rows dropped 2" in out


def test_compare_existing_refuses_gate_combination(tmp_path, capsys):
    campaign, prior = _aged_campaign(tmp_path)
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--compare-existing", str(prior),
           "--expect-existing", str(prior),
           "--original-root", str(campaign["root"]),
           "--check-only"]) == 3
    assert "separate gates" in capsys.readouterr().err


def _tamper(prior: Path, mutate_manifest=None, mutate_record=None,
            reseal_digest=True):
    """Apply a tampering to the prior generation, keeping it internally
    consistent unless the caller wants inconsistent bytes."""
    for record_path in sorted(prior.glob("layer-*.json")):
        record = json.loads(record_path.read_text())
        slice_path = Path(record["read_set"]["manifest_path"])
        manifest = json.loads(gzip.decompress(
            slice_path.read_bytes()).decode("utf-8"))
        if mutate_manifest is not None:
            mutate_manifest(manifest)
        if mutate_record is not None:
            mutate_record(record)
        wire = seal_manifest_bytes(manifest)
        slice_path.write_bytes(wire)
        if reseal_digest:
            record["read_set"]["manifest_sha256"] = \
                hashlib.sha256(wire).hexdigest()
        _reseal_record_identity(record)
        record_path.write_text(json.dumps(record, indent=1, sort_keys=True)
                               + "\n")


def test_compare_existing_refuses_nonzero_phase_tampering(tmp_path, capsys):
    swapped = []

    def swap_phases(manifest):
        phases = manifest["annotations"]["phases"]
        start = 1 if (len(phases) >= 2 and phases[0].get("name") == "head"
                      and phases[0].get("bytes") == 0) else 0
        if len(phases) - start >= 2:
            phases[start], phases[start + 1] = phases[start + 1], phases[start]
            swapped.append(manifest["annotations"]["quantum_id"])

    campaign, prior = _aged_campaign(tmp_path)
    _tamper(prior, mutate_manifest=swap_phases)
    assert swapped, "the fixture must hold a multi-chunk layer to swap"
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--compare-existing", str(prior), "--check-only"]) == 3
    assert "phase tables differ beyond" in capsys.readouterr().err


def test_compare_existing_refuses_nonzero_phase_bounds_tampering(
        tmp_path, capsys):
    def edit_phase(manifest):
        manifest["annotations"]["phases"][1]["bytes"] += 1
        manifest["annotations"]["phases"][1]["cumulative_bytes"] += 1

    campaign, prior = _aged_campaign(tmp_path)
    _tamper(prior, mutate_manifest=edit_phase)
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--compare-existing", str(prior), "--check-only"]) == 3
    assert "phase tables differ beyond" in capsys.readouterr().err


def test_compare_existing_refuses_entrypoint_tampering(tmp_path, capsys):
    def edit_entrypoint(manifest):
        argv = manifest["annotations"]["argv"]
        argv[argv.index("-m") + 1] = "prismaquant.foreign_entry"

    campaign, prior = _aged_campaign(tmp_path)
    _tamper(prior, mutate_manifest=edit_entrypoint)
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--compare-existing", str(prior), "--check-only"]) == 3
    assert "argv differs beyond" in capsys.readouterr().err


def test_compare_existing_refuses_output_root_binding_tampering(
        tmp_path, capsys):
    def edit_output_root(manifest):
        argv = manifest["annotations"]["argv"]
        argv[argv.index("--output-root") + 1] = "/fixture/other-root"

    campaign, prior = _aged_campaign(tmp_path)
    _tamper(prior, mutate_manifest=edit_output_root)
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--compare-existing", str(prior), "--check-only"]) == 3
    assert "argv differs beyond" in capsys.readouterr().err


def test_compare_existing_refuses_unverified_prior_bytes(tmp_path, capsys):
    campaign, prior = _aged_campaign(tmp_path)
    record_path = prior / "layer-000.json"
    record = json.loads(record_path.read_text())
    slice_path = Path(record["read_set"]["manifest_path"])
    slice_path.write_bytes(slice_path.read_bytes() + b"drift")
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--compare-existing", str(prior), "--check-only"]) == 3
    assert "do not hash to the digest" in capsys.readouterr().err


def test_compare_existing_refuses_missing_prior_manifest(tmp_path, capsys):
    campaign, prior = _aged_campaign(tmp_path)
    record = json.loads((prior / "layer-000.json").read_text())
    Path(record["read_set"]["manifest_path"]).unlink()
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--compare-existing", str(prior), "--check-only"]) == 3
    assert "unreadable" in capsys.readouterr().err


@pytest.mark.parametrize("field,value", [
    ("windows", [{"window_index": 0}, {"window_index": 1},
                 {"window_index": 2}]),
    ("output_space", {"root": "/fixture/other/space"}),
    ("chunks", []),
])
def test_compare_existing_refuses_record_drift(tmp_path, capsys, field,
                                               value):
    campaign, prior = _aged_campaign(tmp_path)

    def mutate_record(record):
        record[field] = value

    _tamper(prior, mutate_record=mutate_record)
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--compare-existing", str(prior), "--check-only"]) == 3
    assert "scientific fields differ" in capsys.readouterr().err


def test_compare_existing_refuses_membership_drift(tmp_path, capsys):
    campaign, prior = _aged_campaign(tmp_path)

    def mutate_manifest(manifest):
        manifest["entries"] = manifest["entries"][:-1]
        manifest["entry_count"] = len(manifest["entries"])

    _tamper(prior, mutate_manifest=mutate_manifest)
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--compare-existing", str(prior), "--check-only"]) == 3
    assert "membership, accounting or schema drift" in \
        capsys.readouterr().err


def test_compare_existing_refuses_stale_prior_identity(tmp_path, capsys):
    """An edited prior record body with a stale identity refuses through
    the existing canonical validator BEFORE any comparison certifies it."""
    campaign, prior = _aged_campaign(tmp_path)
    record_path = prior / "layer-000.json"
    record = json.loads(record_path.read_text())
    record["layer"] = 99  # edited body, identity left stale
    record_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--compare-existing", str(prior), "--check-only"]) == 3
    assert "fails its own identity check" in capsys.readouterr().err


def test_compare_existing_refuses_duplicate_prior_ids(tmp_path, capsys):
    campaign, prior = _aged_campaign(tmp_path)
    record = json.loads((prior / "layer-000.json").read_text())
    (prior / "layer-copy.json").write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n")
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--compare-existing", str(prior), "--check-only"]) == 3
    assert "duplicates quantum" in capsys.readouterr().err


def test_compare_existing_refuses_prebound_prior(tmp_path, capsys):
    """A prior generation carrying a boundary/executable readset binding
    binds actual reads; the comparison refuses it outright instead of
    silently ignoring the changed binding. The tampered record reseals its
    identity so the refusal is the pre-bound one, not the identity one."""
    campaign, prior = _aged_campaign(tmp_path)
    record_path = prior / "layer-000.json"
    record = json.loads(record_path.read_text())
    record["boundary_readset"] = {
        "manifest_path": "/fixture/elsewhere/layer-000"
                         ".boundary-readset.json.gz",
        "manifest_sha256": "c" * 64, "entry_count": 1, "total_bytes": 1,
        "read_bytes": 1, "phases": ["checkpoint"], "receipt_sha256": "d" * 64}
    _reseal_record_identity(record)
    record_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--compare-existing", str(prior), "--check-only"]) == 3
    assert "pre-bound" in capsys.readouterr().err


def test_compare_existing_refuses_unknown_manifest_field(tmp_path, capsys):
    """An unknown top-level field in the prior slice manifest cannot
    silently vanish: every top-level field except the transformed
    annotations compares exactly."""
    campaign, prior = _aged_campaign(tmp_path)

    def add_field(manifest):
        manifest["mystery_science_block"] = {"kl": 0.5}

    _tamper(prior, mutate_manifest=add_field)
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--compare-existing", str(prior), "--check-only"]) == 3
    assert "beyond the control annotations" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Receipt and readset binding under the metadata namespace
# ---------------------------------------------------------------------------

def _tiny_receipt(campaign: dict, space: Path) -> Path:
    """A completed capture at tiny scale, sealed by the real writers into
    the DATA adjoint space (the original artifact paths)."""
    torch = pytest.importorskip("torch")
    from prismaquant.joint_adjoint_checkpoints import (
        exact_entry_record,
        write_adjoint_checkpoint,
        write_adjoint_receipt,
    )
    from prismaquant.perturbed_x_cache import (
        write_exact_activation_cache_entry,
    )
    session = {"generation": "fixture-gen-884",
               "run_identity_sha256": "ef" * 32}
    entries_dir = space / "entries"
    entries_dir.mkdir(parents=True, exist_ok=True)
    boundary_entries = {}
    for boundary in (0, 1):
        rows = []
        for batch in range(3):
            ref = write_exact_activation_cache_entry(
                entries_dir, f"boundary-{batch}-{boundary}-at-{boundary}",
                torch.zeros(2, 4),
                identity={"session": dict(session),
                          "slot": f"boundary-{batch}-{boundary}",
                          "kind": "boundary",
                          "coordinates": {"batch": batch,
                                          "boundary": boundary,
                                          "probe": None}},
                max_tensor_bytes=1 << 20, max_file_bytes=1 << 20)
            rows.append(exact_entry_record(ref))
        boundary_entries[str(boundary)] = rows
    checkpoints = []
    for boundary in (1, 2):
        checkpoints.append(write_adjoint_checkpoint(
            space, boundary=boundary, session=dict(
                session, kind="adjoint_checkpoint"),
            cotangents={(p, b): torch.zeros(2, 4) for p in range(2)
                        for b in range(3)},
            shared_adjoint={(p, b): {"scale": 1.0} for p in range(2)
                            for b in range(3)},
            shared_pass={b: {"mask": [0, 1]} for b in range(3)}))
    receipt = {
        "schema": "prismaquant.joint_adjoint_capture.v1",
        "run_identity": {
            "plan_sha256": campaign["plan_sha"],
            "prepared_sha256": campaign["prepared_sha"],
            "campaign_scope": {"campaign": "metadata-seam-884"}},
        "stride": {"value": 1, "source": None, "boundaries": [2, 1],
                   "max_chain_layers": 0},
        "boundary_storage": {
            "session": dict(session),
            "policy": {"prefetch_batches": 2},
            "directory": str(entries_dir)},
        "boundary_entries": boundary_entries,
        "checkpoints": checkpoints,
        "status": "complete",
    }
    write_adjoint_receipt(space, receipt)
    return space / "adjoint-capture.json"


def test_receipt_and_readsets_bind_under_metadata_namespace(tmp_path):
    pytest.importorskip("torch")
    from prismaquant import joint_cost_quantum as quantum
    from prismaquant.joint_adjoint_slices import stage_a_slice
    campaign = _tiny_campaign(tmp_path)
    prior = tmp_path / "prior-records"
    _publish_default_generation(campaign, prior)
    prior_tree = _snapshot(campaign["root"])
    # The receipt lives in the DATA adjoint space -- original artifact
    # paths preserved.
    adjoint_space = campaign["root"] / "layer-quanta" / "adjoint"
    receipt_path = _tiny_receipt(campaign, adjoint_space)
    meta = tmp_path / "meta-gen" / "layer-quanta"
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--metadata-root", str(meta),
           "--adjoint-receipt", str(receipt_path),
           "--boundary-readsets",
           "--executable-readsets"]) == 0
    records = [json.loads(path.read_text())
               for path in sorted((meta / "records").glob("layer-*.json"))]
    assert len(records) == 2
    for record in records:
        # Input identities and source artifact paths retained.
        assert record["campaign"]["plan_sha256"] == campaign["plan_sha"]
        assert record["campaign"]["prepared_sha256"] == \
            campaign["prepared_sha"]
        assert record["adjoint"]["boundary_artifacts"] == str(adjoint_space)
        assert record["output_space"]["root"].startswith(str(campaign["root"]))
        # Bound manifests live in the metadata namespace and hash to their
        # sealed bytes.
        for block, suffix in (("boundary_readset", ".boundary-readset"),
                              ("executable_readset", ".executable")):
            bound = record[block]
            path = Path(bound["manifest_path"])
            assert path == (meta / "adjoint" / "bound-readsets"
                            / f"{record['quantum_id']}{suffix}.json.gz")
            assert path.is_file()
            assert _sha(path) == bound["manifest_sha256"]
        # Static prepared inputs use ordinary immutable input staging;
        # they do not claim a dynamic produced-output binding.
        manifest = json.loads(gzip.decompress(
            Path(record["executable_readset"]["manifest_path"])
            .read_bytes()).decode("utf-8"))
        assert manifest["annotations"]["render_prerequisite"][
            "binding"] is None
        # The record's stage-A slice is control metadata too: it lives in
        # the metadata namespace, and its bytes hash to the bound digest.
        slice_path = Path(record["adjoint"]["slice_path"])
        assert slice_path == meta / "adjoint-slices" / f"{record['quantum_id']}.json"
        assert _sha(slice_path) == record["adjoint"]["slice_sha256"]
        # The consumer's identity gate passes with the DATA output root.
        record_path = meta / "records" / f"{record['quantum_id']}.json"
        found, loaded = quantum.verify_quantum_identity(
            quantum_path=record_path,
            quantum_sha256=_sha(record_path),
            plan_path=campaign["plan_path"], plan_sha256=campaign["plan_sha"],
            prepared_path=campaign["prepared_path"],
            prepared_sha256=campaign["prepared_sha"],
            adjoint_path=slice_path, adjoint_sha256=_sha(slice_path),
            output_root=Path(str(campaign["root"])))
        assert found["quantum_id"] == record["quantum_id"]
        assert loaded == stage_a_slice(json.loads(receipt_path.read_text()),
                                       record["layer"])
    # The prior generation is untouched and no control metadata landed in
    # the data tree: every prior file is byte-identical, and the only new
    # files under the data root are the receipt's own stage-A artifacts
    # under layer-quanta/adjoint/.
    after = _snapshot(campaign["root"])
    for name, data in prior_tree.items():
        assert after[name] == data
    new_under_data = set(after) - set(prior_tree)
    assert new_under_data, "the receipt must have sealed real artifacts"
    assert all(name.startswith("layer-quanta/adjoint/")
               for name in new_under_data)
    assert not (adjoint_space / "bound-readsets").exists()
    # Replay is byte-idempotent.
    first = _snapshot(meta)
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--metadata-root", str(meta),
           "--adjoint-receipt", str(receipt_path),
           "--boundary-readsets",
           "--executable-readsets"]) == 0
    assert _snapshot(meta) == first


def test_dispatcher_accepts_prepared_executable_rows(tmp_path, monkeypatch):
    """Metadata placement retains the complete prepared-input binding."""
    pytest.importorskip("torch")
    import dispatch_joint_quanta as dispatch
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "sha256:" + "0" * 64},
                                "env": {}}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)
    campaign = _tiny_campaign(tmp_path)
    adjoint_space = campaign["root"] / "layer-quanta" / "adjoint"
    receipt_path = _tiny_receipt(campaign, adjoint_space)
    meta = tmp_path / "meta-gen" / "layer-quanta"
    assert regen.main(
        _regen_argv(tmp_path, campaign)
        + ["--metadata-root", str(meta),
           "--adjoint-receipt", str(receipt_path),
           "--executable-readsets"]) == 0
    record_path = meta / "records" / "layer-000.json"
    record = json.loads(record_path.read_text())
    argv = dispatch.quantum_argv(
        record, record_path=record_path,
        output_root=Path(str(campaign["root"])))
    assert "--data-manifest" in argv
    assert "prepared_input" in record["executable_readset"]


def test_binder_requires_exact_metadata_path(tmp_path):
    """The bound-readset binder accepts exactly the producer-named metadata
    path -- the default-layout path under the data root refuses."""
    pytest.importorskip("torch")
    from prismaquant.joint_layer_quanta import (
        emit_quantum_boundary_readsets,
    )
    campaign = _tiny_campaign(tmp_path)
    adjoint_space = campaign["root"] / "layer-quanta" / "adjoint"
    receipt_path = _tiny_receipt(campaign, adjoint_space)
    receipt = json.loads(receipt_path.read_text())
    produced = _produce(campaign, receipt=receipt)
    meta = str(tmp_path / "meta-gen" / "layer-quanta")
    emitted = emit_quantum_boundary_readsets(
        receipt, produced["records"],
        strided_boundaries=produced["derivation"]["checkpoints"], n_probes=2,
        output_root=str(campaign["root"]), metadata_root=meta)
    assert emitted[0]["manifest_path"] == (
        f"{meta}/adjoint/bound-readsets/layer-000.boundary-readset.json.gz")
    with pytest.raises(ValueError, match="must be exactly"):
        bind_quantum_boundary_readset(
            emitted[0]["record"],
            receipt,
            manifest=emitted[0]["manifest"],
            manifest_path=(f"{campaign['root']}/layer-quanta/adjoint/"
                           "bound-readsets/layer-000"
                           ".boundary-readset.json.gz"),
            manifest_sha256=emitted[0]["manifest_sha256"],
            output_root=str(campaign["root"]),
            strided_boundaries=produced["derivation"]["checkpoints"],
            n_probes=2,
            metadata_root=meta)
