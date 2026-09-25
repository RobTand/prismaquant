"""Joiner gates for the distributed joint-AURA cost campaign (§7 of
``docs/design/distributed_campaign_2026-09-19.md``).

The fixtures seal the wire the producer (§3, ``joint_layer_quanta``) and the
runtime (§6, ``joint_cost_quantum``) actually write -- the shapes verified
read-only against the sealed takeover panel on 2026-09-19 and pinned through
issue #787: phase rows ``{name, bytes, cumulative_bytes}`` with unpadded
``layer-N`` names and no ``quantum_id`` field, index-only record windows
(D2), the #768 roster digest (sorted, one per line, no trailing newline),
and payload provenance as ``campaign_binding`` + ``distributed_quantum`` +
``adjoint_slice_sha256``. Every record binds its layer's stage-A slice and
the join recomputes it from one stage-A proof: the completed receipt (mode
a) or the checkpoint bands (mode b, PQ #993). The final test bridges the
real producer into the joiner end to end with synthetic runtime outputs.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import pickle

import pytest
import torch

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.joint_adjoint_slices import adjoint_slice_sha256, stage_a_slice
from prismaquant.joint_aura import (
    arithmetic_identity,
    identity_sha256,
    make_joint_aura_entry,
)
from prismaquant.joint_quanta_join import (
    JOINED_RESULTS_SCHEMA,
    JoinRefused,
    join_joint_quanta,
    load_joint_cost_for_allocation,
    main,
)
from prismaquant.production_weight_cache import _cb_cache_tensor_identity
from tests.test_stage_b_band_binding import band_from_receipt, synthetic_receipt
from tests.test_streamed_cost_checkpoints import _model_identity

RECORD_SCHEMA = "prismaquant.joint_layer_quanta.v1"
STATUS_SCHEMA = "prismaquant.joint_layer_quantum.status.v1"
N_LAYERS = 3
UNITS_PER_LAYER = 2
FORMATS = ["TESSERA_BF16_K1_R1792", "TESSERA_E4M3_K1_R896"]


def _stage_a_receipt(campaign, **overrides):
    """The completed stage-A receipt of the campaign's current digests.

    Stride 1: layer L reads checkpoint L + 1 and chains nothing, the shape
    the records below seal. Deterministic, so recomputing it after the
    digests are sealed gives the same bytes every record binds.
    """
    return synthetic_receipt(
        plan_sha256=campaign["plan_sha256"], prepared_sha256=campaign["prepared_sha256"],
        scope=campaign["scope"], num_layers=N_LAYERS, stride=1, **overrides)


def _bind_stage_a(campaign):
    """Hand the joiner the completed receipt (mode a) for the current digests."""
    campaign.pop("adjoint_bands", None)
    campaign["adjoint_receipt"] = _stage_a_receipt(campaign)


def _slice_sha(campaign, layer):
    return adjoint_slice_sha256(stage_a_slice(_stage_a_receipt(campaign), layer))


def _row(qname, fmt, probe, sign=0.1):
    weight = _cb_cache_tensor_identity(
        torch.arange(16, dtype=torch.bfloat16).reshape(4, 4))
    activation = {"schema": "prismaquant.joint_aura.activation.v1",
                  "quantizes_input": False, "activation_max_abs": None,
                  "input_global_scale": None, "clip_enabled": False}
    operator = {"schema": "prismaquant.joint_aura.operator.v2", "qname": qname,
                "format": fmt, "source_weight": weight,
                "rendered_weight": weight, "activation": activation,
                "arithmetic": probe["arithmetic"],
                "probe_identity_sha256": identity_sha256(probe)}
    steps = [sign * (i + 1) * (1 if i % 2 == 0 else -1) for i in range(3)]
    return make_joint_aura_entry(
        operator_identity=operator, probe_identity=probe,
        signed_components=[{"weight": v, "activation": 0., "mixed": 0.,
                            "total": v} for v in steps])


def _units_of_layer(campaign, layer):
    return sorted(q for q in campaign["roster"] if f".layers.{layer}." in q)


def _roster_digest_of(campaign):
    """§3.1's roster digest: sha256 of the sorted roster, one per line, no
    trailing newline (#768's construction, what every sealed record carries)."""
    return hashlib.sha256(
        "\n".join(sorted(campaign["roster"])).encode()).hexdigest()


@pytest.fixture
def probe():
    arithmetic = arithmetic_identity(torch.bfloat16)
    return {"schema": "prismaquant.joint_aura.probes.v2",
            "source_model": _model_identity("join-fixture"),
            "calibration_sha256": "1" * 64, "calibration_shape": [1, 4],
            "calibration_dtype": "torch.int64",
            "producer_source_sha256": "2" * 64, "n_probes": 3,
            "seed_base": 7000, "token_scope": "causal",
            "distribution": "rademacher",
            "normalization": "global_kl_fisher", "temperature": 1.0,
            "arithmetic": arithmetic}


@pytest.fixture
def campaign(probe):
    """The campaign binding, sealed exactly as the takeover panel shapes it:
    the parent run manifest's phase table carries ``head`` plus unpadded
    ``layer-N`` rows keyed ``{name, bytes, cumulative_bytes}`` -- no
    ``quantum_id``, no start/end offsets (#787 B2)."""
    roster, formats = [], {}
    for layer in range(N_LAYERS):
        for unit in range(UNITS_PER_LAYER):
            qname = f"model.language_model.layers.{layer}.mlp.expert.{unit}"
            roster.append(qname)
            formats[qname] = list(FORMATS)
    roster.sort()
    scope = {"campaign": "join-fixture", "layers": list(range(N_LAYERS))}
    entries = [{"path": "/mnt/shared/head.bin", "offset": 0,
                "bytes": 512, "sha256": None}]
    phases = [{"name": "head", "bytes": 512, "cumulative_bytes": 512}]
    for layer in range(N_LAYERS):
        size = 1024 * (layer + 1)
        halves = (size // 2, size - size // 2)
        # The first entry of each layer phase is a source extent under the
        # plan's model dir (the stage-A adjoint manifest requires one per
        # layer); the rest are render-cache paths outside it.
        entries.append({"path": f"/mnt/shared/models/TEST/shard-{layer}.safetensors",
                        "offset": 0, "bytes": halves[0], "sha256": None})
        entries.append({"path": f"/mnt/shared/rows/row-{layer}/cache/w0.pt",
                        "offset": 0, "bytes": halves[1], "sha256": None})
        phases.append({"name": f"layer-{layer}", "bytes": size,
                       "cumulative_bytes": phases[-1]["cumulative_bytes"] + size})
    parent_manifest = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "produced_by": {"tool": "fixture"},
        "mount_prefix": "/mnt/shared",
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": phases[-1]["cumulative_bytes"],
        "annotations": {
            "entry_point": "prismaquant.tessera_joint_aura:run",
            "plan_sha256": "a" * 64,
            "layers": list(range(N_LAYERS)),
            "campaign_scope": scope,
            "phases": phases,
        },
    }
    bound = {"plan_sha256": "a" * 64, "prepared_sha256": "b" * 64,
             "manifest_sha256": "c" * 64, "scope": scope,
             "roster": roster, "formats_by_qname": formats,
             "parent_manifest": parent_manifest}
    _bind_stage_a(bound)
    return bound


def _phase_row(campaign, layer):
    return next(row for row in
                campaign["parent_manifest"]["annotations"]["phases"]
                if row["name"] == f"layer-{layer}")


def _record(campaign, layer, root, *, roster_digest=None):
    """One layer-quantum record in the producer's sealed shape (§3.1):
    index-only windows (D2), the source phase citing the parent manifest's
    layer-N row, and the #768 roster digest."""
    quantum_id = f"layer-{layer:03d}"
    phase = _phase_row(campaign, layer)
    start = phase["cumulative_bytes"] - phase["bytes"]
    space = root / "layer-quanta" / quantum_id
    record = {
        "schema": RECORD_SCHEMA, "quantum_id": quantum_id, "layer": layer,
        "campaign": {
            "plan_path": "/mnt/shared/plan.json",
            "plan_sha256": campaign["plan_sha256"],
            "prepared_path": "/mnt/shared/prepared.json",
            "prepared_sha256": campaign["prepared_sha256"],
            "read_manifest_sha256": campaign["manifest_sha256"],
            "campaign_scope": campaign["scope"],
            "unit_roster_sha256": roster_digest or _roster_digest_of(campaign),
        },
        "read_set": {
            "manifest_path": f"manifests/{quantum_id}.data-manifest.json.gz",
            "manifest_sha256": "0" * 64,
            "entry_count": 2, "total_bytes": phase["bytes"],
            "source_phase": {"name": f"layer-{layer}",
                             "start_bytes": start,
                             "end_bytes": phase["cumulative_bytes"]},
        },
        "chunks": [{"name": f"{quantum_id}-chunk-000", "start_bytes": 0,
                    "end_bytes": phase["bytes"]}],
        "windows": [{"window_index": 0}, {"window_index": 1}],
        "adjoint": {"checkpoint_boundary": layer + 1, "chain_layers": [],
                    "boundary_artifacts": "/mnt/shared/adjoint",
                    "slice_sha256": _slice_sha(campaign, layer),
                    "slice_path": f"/mnt/shared/adjoint-slices/{quantum_id}.json"},
        "output_space": {"root": str(space),
                         "cost_payload": str(space / "cost.pkl"),
                         "results": str(space / "results.json"),
                         "counters": str(space / "counters.json"),
                         "checkpoint_dir": str(space / "checkpoints")},
    }
    record["identity_sha256"] = canonical_json_sha256(
        record, where="fixture layer-quantum record")
    return record


def _payload_provenance(campaign, record):
    """The §6.4 provenance grammar the runtime seals (#787 B4)."""
    return {
        "campaign_binding": {
            "plan_sha256": campaign["plan_sha256"],
            "prepared_sha256": campaign["prepared_sha256"],
            "read_manifest_sha256": campaign["manifest_sha256"],
            "campaign_scope": campaign["scope"],
            "unit_roster_sha256": _roster_digest_of(campaign),
        },
        "distributed_quantum": {
            "quantum_id": record["quantum_id"],
            "identity_sha256": record["identity_sha256"],
            "adjoint_slice_sha256": record["adjoint"].get("slice_sha256"),
            "checkpoint_boundary": record["adjoint"]["checkpoint_boundary"],
            "chain_layers": list(record["adjoint"]["chain_layers"]),
            "windows": len(record["windows"]),
            "chunks": [chunk["name"] for chunk in record["chunks"]],
        },
        "adjoint_slice_sha256": record["adjoint"].get("slice_sha256"),
    }


def _write_quantum(root, campaign, probe, layer, *, status="complete",
                   costs=None, provenance=None, roster_digest=None,
                   record=None):
    quantum_id = f"layer-{layer:03d}"
    record = record or _record(campaign, layer, root,
                               roster_digest=roster_digest)
    records = root / "layer-quanta" / "records"
    records.mkdir(parents=True, exist_ok=True)
    (records / f"{quantum_id}.json").write_text(json.dumps(record))
    space = root / "layer-quanta" / quantum_id
    space.mkdir(parents=True, exist_ok=True)
    if costs is None:
        costs = {qname: {fmt: _row(qname, fmt, probe, sign=0.1 * (layer + 1))
                         for fmt in FORMATS}
                 for qname in _units_of_layer(campaign, layer)}
    payload = {"costs": costs,
               "provenance": provenance or _payload_provenance(campaign, record)}
    cost_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    (space / "cost.pkl").write_bytes(cost_bytes)
    units = [len(costs), len(costs)]
    status_doc = {"schema": STATUS_SCHEMA, "quantum_id": quantum_id,
                  "identity_sha256": record["identity_sha256"],
                  "status": status, "units": units, "unix": 1750000000}
    (space / "status.json").write_text(json.dumps(status_doc))
    (space / "results.json").write_text(json.dumps({"quantum_id": quantum_id}))
    return record, hashlib.sha256(cost_bytes).hexdigest()


def _seal_inputs(root, campaign, *, gzip_manifest=False):
    """Write the CLI input files and bind their real digests into the
    campaign, so records and argv digest-check against actual bytes."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(json.dumps({"plan": "fixture"}))
    (root / "prepared.json").write_text(json.dumps({"prepared": "fixture"}))
    manifest_bytes = json.dumps(
        campaign["parent_manifest"]).encode() + b"\n"
    if gzip_manifest:
        (root / "manifest.json.gz").write_bytes(
            gzip.compress(manifest_bytes, mtime=0))
        manifest_name = "manifest.json.gz"
    else:
        (root / "manifest.json").write_bytes(manifest_bytes)
        manifest_name = "manifest.json"
    (root / "scope.json").write_text(json.dumps(campaign["scope"]))
    (root / "roster.txt").write_text("\n".join(campaign["roster"]) + "\n")
    (root / "formats.json").write_text(
        json.dumps(campaign["formats_by_qname"]))
    campaign["manifest_name"] = manifest_name
    for key, name in (("plan_sha256", "plan.json"),
                      ("prepared_sha256", "prepared.json"),
                      ("manifest_sha256", manifest_name)):
        campaign[key] = hashlib.sha256(
            (root / name).read_bytes()).hexdigest()
    _bind_stage_a(campaign)


def _proof_file(root, name, document):
    path = root / name
    path.write_text(json.dumps(document, sort_keys=True))
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _argv(root, out, campaign):
    return ["--input-root", str(root), "--output-dir", str(out),
            "--plan", str(root / "plan.json"),
            "--plan-sha256", campaign["plan_sha256"],
            "--prepared", str(root / "prepared.json"),
            "--prepared-sha256", campaign["prepared_sha256"],
            "--manifest", str(root / campaign["manifest_name"]),
            "--manifest-sha256", campaign["manifest_sha256"],
            "--scope", str(root / "scope.json"),
            "--roster", str(root / "roster.txt"),
            "--formats-by-qname", str(root / "formats.json"),
            *_proof_argv(root, campaign)]


def _proof_argv(root, campaign):
    """The stage-A proof files the CLI reads, written from the campaign."""
    if "adjoint_bands" in campaign:
        argv = []
        for band in campaign["adjoint_bands"]:
            path, sha = _proof_file(root, f"band-{band['band']['boundary']:03d}.json", band)
            argv += ["--adjoint-band", path, "--adjoint-band-sha256", sha]
        return argv
    path, sha = _proof_file(root, "adjoint-capture.json", campaign["adjoint_receipt"])
    return ["--adjoint-receipt", path, "--adjoint-receipt-sha256", sha]


def _run_cli(root, out, campaign):
    assert main(_argv(root, out, campaign)) == 0
    return out / "joint-cost.pkl", out / "results.json"


def _write_all(root, campaign, probe, **kwargs):
    for layer in range(N_LAYERS):
        _write_quantum(root, campaign, probe, layer, **kwargs)


# -- canonical bytes under receipt permutation -------------------------------


def test_permuted_arrival_yields_canonical_bytes(tmp_path, campaign, probe):
    """The join is a disjoint union: arrival order cannot change the bytes."""
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    _write_all(root, campaign, probe)
    first = tmp_path / "joined-first"
    cost_path, results_path = _run_cli(root, first, campaign)
    first_cost, first_results = cost_path.read_bytes(), results_path.read_bytes()

    second = tmp_path / "joined-second"
    assert main(_argv(root, second, campaign)) == 0
    assert (second / "joint-cost.pkl").read_bytes() == first_cost
    first_doc = json.loads(first_results.decode())
    second_doc = json.loads((second / "results.json").read_bytes().decode())
    # The wall-clock seal moves; everything else is canonical.
    assert isinstance(first_doc.pop("joined_unix"), float)
    assert isinstance(second_doc.pop("joined_unix"), float)
    assert first_doc.pop("distributed").pop("joined_unix") is not None
    assert second_doc.pop("distributed").pop("joined_unix") is not None
    assert second_doc == first_doc

    payload = pickle.loads(first_cost)
    assert sorted(payload["costs"]) == campaign["roster"]
    assert payload["provenance"]["coverage"]["gaps"] == []
    results = json.loads(first_results.decode())
    assert results["schema"] == JOINED_RESULTS_SCHEMA
    assert results["status"] == "complete"


def test_join_reads_the_gzip_sealed_parent_manifest(tmp_path, campaign, probe):
    """The parent run manifest is sealed as one gzip member (§4.3); the CLI
    reads the member and digests the sealed file bytes."""
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign, gzip_manifest=True)
    _write_all(root, campaign, probe)
    out = tmp_path / "joined"
    assert main(_argv(root, out, campaign)) == 0
    results = json.loads((out / "results.json").read_text())
    assert results["status"] == "complete"


# -- gapped quanta: exit 0, gaps named, consumption refuses -------------------


def test_gapped_quantum_exits_zero_and_names_the_gap(tmp_path, campaign, probe):
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    for layer in range(N_LAYERS):
        _write_quantum(root, campaign, probe, layer,
                       status="complete" if layer else "gapped")
    out = tmp_path / "joined"
    assert main(_argv(root, out, campaign)) == 0
    payload = pickle.loads((out / "joint-cost.pkl").read_bytes())
    gaps = payload["provenance"]["coverage"]["gaps"]
    assert [g["quantum_id"] for g in gaps] == ["layer-000"]
    # B3: the record seals index-only windows, so the gap's units are named
    # from the roster by layer -- the count is the layer's true unit count.
    assert gaps[0]["unit_count"] == UNITS_PER_LAYER
    assert gaps[0]["units"] == _units_of_layer(campaign, 0)
    results = json.loads((out / "results.json").read_text())
    assert results["status"] == "gapped"
    assert [g["quantum_id"] for g in results["distributed"]["gaps"]] == ["layer-000"]
    present = [q for q in campaign["roster"] if ".layers.0." not in q]
    assert sorted(payload["costs"]) == sorted(present)


def test_missing_status_is_a_gap_not_an_error(tmp_path, campaign, probe):
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    _write_all(root, campaign, probe)
    (root / "layer-quanta" / "layer-001" / "status.json").unlink()
    out = tmp_path / "joined"
    assert main(_argv(root, out, campaign)) == 0
    results = json.loads((out / "results.json").read_text())
    assert results["status"] == "gapped"
    assert [g["quantum_id"] for g in results["distributed"]["gaps"]] == ["layer-001"]


def test_missing_record_is_a_named_gap_never_a_shrunk_set(tmp_path, campaign,
                                                          probe):
    """B2: the parent manifest's phases carry unpadded names and no
    ``quantum_id``; the expected set is still the padded quantum ids, a lost
    record is a named gap with its roster units, and ``head`` -- a phase, not
    a quantum -- is never a gap."""
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    _write_all(root, campaign, probe)
    lost = root / "layer-quanta" / "records" / "layer-001.json"
    lost.unlink()
    out = tmp_path / "joined"
    assert main(_argv(root, out, campaign)) == 0
    results = json.loads((out / "results.json").read_text())
    assert results["status"] == "gapped"
    gaps = results["distributed"]["gaps"]
    assert [g["quantum_id"] for g in gaps] == ["layer-001"]
    assert gaps[0]["record_absent"] is True
    assert gaps[0]["unit_count"] == UNITS_PER_LAYER
    assert gaps[0]["units"] == _units_of_layer(campaign, 1)
    assert all(g["quantum_id"] != "head" for g in gaps)


def test_allocation_reader_refuses_a_gapped_payload(tmp_path, campaign, probe):
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    for layer in range(N_LAYERS):
        _write_quantum(root, campaign, probe, layer,
                       status="complete" if layer else "gapped")
    cost_path, _ = _run_cli(root, tmp_path / "joined", campaign)
    with pytest.raises(Exception, match="gapped"):
        load_joint_cost_for_allocation(cost_path)


def test_allocation_reader_accepts_a_complete_payload(tmp_path, campaign, probe):
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    _write_all(root, campaign, probe)
    cost_path, _ = _run_cli(root, tmp_path / "joined", campaign)
    payload = load_joint_cost_for_allocation(cost_path)
    assert sorted(payload["costs"]) == campaign["roster"]


# -- custody: the four wire pins (#787) --------------------------------------


def test_roster_digest_is_the_768_construction(tmp_path, campaign, probe):
    """B1: a record sealed with any other roster-digest spelling -- the
    trailing-newline preimage the joiner once recomputed -- refuses custody."""
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    wrong = hashlib.sha256(
        ("\n".join(campaign["roster"]) + "\n").encode()).hexdigest()
    for layer in range(N_LAYERS):
        _write_quantum(root, campaign, probe, layer, roster_digest=wrong)
    with pytest.raises(JoinRefused, match="roster digest"):
        join_joint_quanta(receipts=None, campaign=campaign,
                          output_dir=tmp_path / "joined", input_root=root)


def test_retargeted_receipt_refuses(tmp_path, campaign, probe):
    """A cost payload answering for another quantum's identity fails closed."""
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    _write_all(root, campaign, probe)
    (root / "layer-quanta" / "layer-001" / "cost.pkl").write_bytes(
        (root / "layer-quanta" / "layer-002" / "cost.pkl").read_bytes())
    with pytest.raises(JoinRefused, match="custody"):
        join_joint_quanta(
            receipts=None, campaign=campaign, output_dir=tmp_path / "joined",
            input_root=root)


def test_foreign_provenance_refuses(tmp_path, campaign, probe):
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    _write_all(root, campaign, probe)
    other = copy.deepcopy(campaign)
    other["plan_sha256"] = "f" * 64
    with pytest.raises(JoinRefused, match="custody|provenance|campaign"):
        join_joint_quanta(receipts=None, campaign=other,
                          output_dir=tmp_path / "joined", input_root=root)


def test_adjoint_slice_digest_is_checked(tmp_path, campaign, probe):
    """B4 after PQ #993: the §6.4 adjoint digest is the slice digest, pinned
    and checked every way -- the payload must answer for the record's bound
    slice, an unbound (pre-A) record never joins, a record binding a whole
    receipt never joins, and every record must bind exactly the slice the
    stage-A proof gives its layer."""
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    # (a) the payload claims a slice the record does not bind.
    record = _record(campaign, 0, root)
    lying = _payload_provenance(campaign, record)
    lying["adjoint_slice_sha256"] = "f" * 64
    lying["distributed_quantum"]["adjoint_slice_sha256"] = "f" * 64
    _write_quantum(root, campaign, probe, 0, provenance=lying)
    _write_quantum(root, campaign, probe, 1)
    with pytest.raises(JoinRefused, match="adjoint"):
        join_joint_quanta(receipts=None, campaign=campaign,
                          output_dir=tmp_path / "refused-a", input_root=root)

    # (b) an unbound (pre-A) record refuses even with a matching payload.
    # (d) so does a record that binds a whole receipt instead of its slice.
    for label, binding, match in (("pre-a", {"receipt_sha256": None}, "unbound"),
                                  ("whole", {"receipt_sha256": "e" * 64},
                                   "whole stage-A receipt")):
        root_b = tmp_path / label
        _seal_inputs(root_b, campaign)
        unbound = _record(campaign, 0, root_b)
        unbound["adjoint"] = {**{k: v for k, v in unbound["adjoint"].items()
                                 if k not in ("slice_sha256", "slice_path")}, **binding}
        unbound["identity_sha256"] = canonical_json_sha256(
            {k: v for k, v in unbound.items() if k != "identity_sha256"},
            where="unbound fixture record")
        provenance = _payload_provenance(campaign, unbound)
        _write_quantum(root_b, campaign, probe, 0, provenance=provenance,
                       record=unbound)
        _write_quantum(root_b, campaign, probe, 1)
        with pytest.raises(JoinRefused, match=match):
            join_joint_quanta(receipts=None, campaign=campaign,
                              output_dir=tmp_path / f"refused-{label}", input_root=root_b)

    # (c) mode (a): the join recomputes every record's slice from the
    # completed receipt; a receipt of another Stage A generation gives other
    # slices, so each record fails closed.
    root_c = tmp_path / "pinned"
    _seal_inputs(root_c, campaign)
    _write_all(root_c, campaign, probe)
    pinned = copy.deepcopy(campaign)
    pinned["adjoint_receipt"] = _stage_a_receipt(campaign, generation="another-generation")
    with pytest.raises(JoinRefused, match="another stage-A slice than the completed receipt"):
        join_joint_quanta(receipts=None, campaign=pinned,
                          output_dir=tmp_path / "refused-c", input_root=root_c)
    # A receipt answering for another plan refuses at the header check.
    foreign = copy.deepcopy(campaign)
    foreign["adjoint_receipt"] = synthetic_receipt(
        plan_sha256="f" * 64, prepared_sha256=campaign["prepared_sha256"],
        scope=campaign["scope"], num_layers=N_LAYERS, stride=1)
    with pytest.raises(JoinRefused):
        join_joint_quanta(receipts=None, campaign=foreign,
                          output_dir=tmp_path / "refused-foreign", input_root=root_c)


# -- mode (b): checkpoint bands ------------------------------------------------


def _band_campaign(campaign, boundaries=None, **overrides):
    receipt = _stage_a_receipt(campaign, **overrides)
    boundaries = boundaries or receipt["stride"]["boundaries"]
    bound = copy.deepcopy(campaign)
    bound.pop("adjoint_receipt")
    bound["adjoint_bands"] = [band_from_receipt(receipt, b) for b in boundaries]
    return bound


def test_bands_join_like_the_completed_receipt(tmp_path, campaign, probe):
    """Mode (b): one header, every stride checkpoint has a band, every layer
    has a matching record -- the join is byte-identical to mode (a)'s."""
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    _write_all(root, campaign, probe)
    receipt_out = tmp_path / "joined-receipt"
    assert main(_argv(root, receipt_out, campaign)) == 0
    bands = _band_campaign(campaign)
    bands_out = tmp_path / "joined-bands"
    assert main(_argv(root, bands_out, bands)) == 0
    assert (bands_out / "joint-cost.pkl").read_bytes() == \
        (receipt_out / "joint-cost.pkl").read_bytes()
    by_bands = json.loads((bands_out / "results.json").read_text())
    by_receipt = json.loads((receipt_out / "results.json").read_text())
    assert by_bands["status"] == "complete"
    assert by_bands["stage_a"]["mode"] == "checkpoint bands"
    assert by_receipt["stage_a"]["mode"] == "completed receipt"
    assert by_bands["stage_a"]["run_header_sha256"] == by_receipt["stage_a"]["run_header_sha256"]
    # A layer without a record is a named gap, as in mode (a).
    (root / "layer-quanta" / "records" / "layer-001.json").unlink()
    gapped = join_joint_quanta(receipts=None, campaign=bands,
                               output_dir=tmp_path / "joined-gapped", input_root=root)
    assert gapped["status"] == "gapped"
    assert [gap["quantum_id"] for gap in gapped["gaps"]] == ["layer-001"]


def test_band_join_refusals(tmp_path, campaign, probe):
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    _write_all(root, campaign, probe)
    # A missing stride checkpoint: bands 3 and 2 but not 1.
    with pytest.raises(JoinRefused, match=r"stride checkpoints \[1\] have no band"):
        join_joint_quanta(receipts=None, campaign=_band_campaign(campaign, [3, 2]),
                          output_dir=tmp_path / "refused-missing", input_root=root)
    # Bands with a different run header: another generation sealed one band.
    mixed = _band_campaign(campaign, [3, 2])
    mixed["adjoint_bands"].append(band_from_receipt(
        _stage_a_receipt(campaign, generation="another-generation"), 1))
    with pytest.raises(JoinRefused, match="run header"):
        join_joint_quanta(receipts=None, campaign=mixed,
                          output_dir=tmp_path / "refused-mixed", input_root=root)
    # A slice mismatch at the join: every band of another generation gives
    # every layer another slice than the one its record binds.
    other = _band_campaign(campaign, generation="another-generation")
    with pytest.raises(JoinRefused, match="another stage-A slice than the checkpoint bands"):
        join_joint_quanta(receipts=None, campaign=other,
                          output_dir=tmp_path / "refused-slice", input_root=root)
    # A band for the wrong boundary never enters the set.
    wrong = _band_campaign(campaign)
    wrong["adjoint_bands"][0]["band"] = {"boundary": 2, "layers": [1]}
    with pytest.raises(JoinRefused):
        join_joint_quanta(receipts=None, campaign=wrong,
                          output_dir=tmp_path / "refused-wrong", input_root=root)
    # Exactly one proof: both or neither refuses.
    both = _band_campaign(campaign)
    both["adjoint_receipt"] = campaign["adjoint_receipt"]
    neither = copy.deepcopy(campaign)
    neither.pop("adjoint_receipt")
    for bad in (both, neither):
        with pytest.raises(JoinRefused, match="exactly one stage-A proof"):
            join_joint_quanta(receipts=None, campaign=bad,
                              output_dir=tmp_path / "refused-proof", input_root=root)


# -- rows: a defective row fails the join and names the qname ------------------


def test_defective_row_fails_the_join_and_names_the_qname(tmp_path, campaign,
                                                          probe):
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    _write_all(root, campaign, probe)
    victim = next(q for q in campaign["roster"] if ".layers.1." in q)
    space = root / "layer-quanta" / "layer-001"
    payload = pickle.loads((space / "cost.pkl").read_bytes())
    payload["costs"][victim]["TESSERA_BF16_K1_R1792"] = {
        "cost_source": "joint_aura", "predicted_dloss": float("nan")}
    (space / "cost.pkl").write_bytes(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    with pytest.raises(JoinRefused, match=victim):
        join_joint_quanta(receipts=None, campaign=campaign,
                          output_dir=tmp_path / "joined", input_root=root)
    out = tmp_path / "joined-cli"
    assert main(_argv(root, out, campaign)) == 1
    assert not (out / "joint-cost.pkl").exists()


def test_complete_quantum_dropping_rows_refuses(tmp_path, campaign, probe):
    """B3's defect side: gaps account their own layers' roster units, so a
    COMPLETE quantum that drops rows cannot hide behind a sibling's gap."""
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    _write_quantum(root, campaign, probe, 0, status="gapped")
    dropped = _units_of_layer(campaign, 1)[0]
    costs = {qname: {fmt: _row(qname, fmt, probe, sign=0.2) for fmt in FORMATS}
             for qname in _units_of_layer(campaign, 1) if qname != dropped}
    _write_quantum(root, campaign, probe, 1, costs=costs)
    _write_quantum(root, campaign, probe, 2)
    with pytest.raises(JoinRefused, match="outside the named gaps"):
        join_joint_quanta(receipts=None, campaign=campaign,
                          output_dir=tmp_path / "joined", input_root=root)


# -- the bridge: real producer records join synthetic runtime outputs ---------


def test_producer_records_join_synthetic_runtime_outputs(tmp_path, campaign,
                                                         probe):
    """End to end with the real §3 producer: ``layer_quanta`` cuts a small
    campaign, synthetic §6.4 runtime outputs land in each record's sealed
    output space, and the §7 joiner merges them complete. Nothing here
    fabricates campaign data -- the records are the producer's own bytes and
    the payloads are marked synthetic only in their measured rows."""
    from prismaquant.joint_layer_quanta import (
        derive_stride,
        layer_quanta,
    )

    root = tmp_path / "produced"
    root.mkdir(parents=True)
    parent = copy.deepcopy(campaign["parent_manifest"])
    # Two layers of the fixture campaign: the producer cuts real slices.
    parent["annotations"]["layers"] = [0, 1]
    keep = {"head", "layer-0", "layer-1"}
    parent["annotations"]["phases"] = [
        row for row in parent["annotations"]["phases"]
        if row["name"] in keep]
    parent["entries"] = [
        entry for entry in parent["entries"]][:1 + 2 * 2]
    # The kept entries must tile exactly the kept phases (head + two layers).
    assert sum(entry["bytes"] for entry in parent["entries"]) == \
        parent["annotations"]["phases"][-1]["cumulative_bytes"]
    parent["total_bytes"] = \
        parent["annotations"]["phases"][-1]["cumulative_bytes"]
    parent["entry_count"] = len(parent["entries"])
    plan = {"output_root": str(root),
            "model": "/mnt/shared/models/TEST",
            "retained_window_budget_derivation": {
                "windows_by_layer": {"0": 1, "1": 2}}}
    roster = sorted(q for q in campaign["roster"] if ".layers.2." not in q)
    formats = {q: list(campaign["formats_by_qname"][q]) for q in roster}
    prepared = {"formats_by_qname": {q: dict.fromkeys(f) for q, f in formats.items()}}
    (root / "manifest.json").write_text(json.dumps(parent))
    manifest_sha256 = hashlib.sha256(
        (root / "manifest.json").read_bytes()).hexdigest()
    stride = derive_stride(2, 8)
    receipt = synthetic_receipt(
        plan_sha256=campaign["plan_sha256"], prepared_sha256=campaign["prepared_sha256"],
        scope=campaign["scope"], num_layers=2, stride=8)
    assert receipt["stride"]["boundaries"] == list(stride["checkpoints"])
    built = layer_quanta(
        plan, prepared, parent, parent_manifest_sha256=manifest_sha256,
        output_root=str(root), plan_path="/mnt/shared/plan.json",
        plan_sha256=campaign["plan_sha256"],
        prepared_path="/mnt/shared/prepared.json",
        prepared_sha256=campaign["prepared_sha256"],
        adjoint_receipt=receipt)
    records = built["records"]
    assert [record["quantum_id"] for record in records] == [
        "layer-000", "layer-001"]

    # Persist the records where the producer's own layout says they live and
    # land synthetic runtime outputs in each record's sealed output space.
    for record in records:
        record_path = root / "layer-quanta" / "records" / (
            record["quantum_id"] + ".json")
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record))
        space = root / "layer-quanta" / record["quantum_id"]
        space.mkdir(parents=True, exist_ok=True)
        layer = record["layer"]
        costs = {}
        for qname in sorted(q for q in roster if f".layers.{layer}." in q):
            costs[qname] = {fmt: _row(qname, fmt, probe, sign=0.1 * (layer + 1))
                            for fmt in formats[qname]}
        payload = {
            "costs": costs,
            "provenance": {
                "campaign_binding": {
                    "plan_sha256": record["campaign"]["plan_sha256"],
                    "prepared_sha256": record["campaign"]["prepared_sha256"],
                    "read_manifest_sha256":
                        record["campaign"]["read_manifest_sha256"],
                    "campaign_scope": record["campaign"]["campaign_scope"],
                    "unit_roster_sha256":
                        record["campaign"]["unit_roster_sha256"],
                },
                "distributed_quantum": {
                    "quantum_id": record["quantum_id"],
                    "identity_sha256": record["identity_sha256"],
                    "adjoint_slice_sha256":
                        record["adjoint"]["slice_sha256"],
                    "checkpoint_boundary":
                        record["adjoint"]["checkpoint_boundary"],
                    "chain_layers": list(record["adjoint"]["chain_layers"]),
                    "windows": len(record["windows"]),
                    "chunks": [chunk["name"] for chunk in record["chunks"]],
                },
                "adjoint_slice_sha256": record["adjoint"]["slice_sha256"],
            },
        }
        (space / "cost.pkl").write_bytes(
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        (space / "status.json").write_text(json.dumps({
            "schema": STATUS_SCHEMA, "quantum_id": record["quantum_id"],
            "identity_sha256": record["identity_sha256"],
            "status": "complete",
            "units": [len(costs), len(costs)], "unix": 1750000000}))

    join_campaign = {
        "plan_sha256": campaign["plan_sha256"],
        "prepared_sha256": campaign["prepared_sha256"],
        "manifest_sha256": manifest_sha256,
        "scope": campaign["scope"],
        "roster": roster,
        "formats_by_qname": formats,
        "parent_manifest": parent,
        "adjoint_receipt": receipt,
    }
    out = tmp_path / "joined"
    result = join_joint_quanta(receipts=None, campaign=join_campaign,
                               output_dir=out, input_root=root, now=1750000009)
    assert result["status"] == "complete"
    assert result["gaps"] == []
    joined = pickle.loads((out / "joint-cost.pkl").read_bytes())
    assert sorted(joined["costs"]) == roster
    assert joined["provenance"]["coverage"]["status"] == "complete"
    assert load_joint_cost_for_allocation(out / "joint-cost.pkl")["costs"] \
        == joined["costs"]


# -- one probe-identity validation per payload (PQ #1256) --------------------


def test_join_validates_each_payloads_probe_identity_once(
        tmp_path, campaign, probe, monkeypatch):
    """Every row of a payload shares one probe identity object, and on
    GLM-5.3 that identity is 8.9 MB of model weight map. The join validated
    it once per ROW (119 ms each, about 14 h for 45 layers); it validates it
    once per PAYLOAD, and the joined bytes still carry the plain probe."""
    from prismaquant import cost_streaming
    calls = []
    validate = cost_streaming.validate_streamed_model_identity

    def counted(*args, **kwargs):
        calls.append(kwargs.get("where"))
        return validate(*args, **kwargs)

    monkeypatch.setattr(cost_streaming, "validate_streamed_model_identity",
                        counted)
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    _write_all(root, campaign, probe)
    rows = N_LAYERS * UNITS_PER_LAYER * len(FORMATS)
    calls.clear()  # building the fixture rows validates each one
    cost_path, _ = _run_cli(root, tmp_path / "joined", campaign)
    row_calls = [where for where in calls if where == "joint AURA row"]
    assert len(row_calls) == N_LAYERS < rows

    joined = pickle.loads(cost_path.read_bytes())
    for per_unit in joined["costs"].values():
        for row in per_unit.values():
            assert type(row["probe_identity"]) is dict
            assert row["probe_identity"] == probe
