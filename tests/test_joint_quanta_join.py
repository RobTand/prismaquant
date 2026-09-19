"""Joiner gates for the distributed joint-AURA cost campaign (§7 of
``docs/design/distributed_campaign_2026-09-19.md``).

RED-first: these tests fixture the producer (§3) and runtime (§6) record
shapes per the contract's schemas and pin the joiner's behavior before
``prismaquant/joint_quanta_join.py`` exists. The producer
(``joint_layer_quanta.py``) and the runtime (``joint_cost_quantum``) are
built in parallel; nothing here imports them.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle

import pytest
import torch

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
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
from tests.test_streamed_cost_checkpoints import _model_identity

RECORD_SCHEMA = "prismaquant.joint_layer_quanta.v1"
STATUS_SCHEMA = "prismaquant.joint_layer_quantum.status.v1"
N_LAYERS = 3
UNITS_PER_LAYER = 2
FORMATS = ["TESSERA_BF16_K1_R1792", "TESSERA_E4M3_K1_R896"]


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
    roster, formats = [], {}
    for layer in range(N_LAYERS):
        for unit in range(UNITS_PER_LAYER):
            qname = f"model.language_model.layers.{layer}.mlp.expert.{unit}"
            roster.append(qname)
            formats[qname] = list(FORMATS)
    roster.sort()
    scope = {"campaign": "join-fixture", "layers": list(range(N_LAYERS))}
    phases = []
    cursor = 0
    for layer in range(N_LAYERS):
        size = 1024 * (layer + 1)
        phases.append({"name": f"layer-{layer}", "quantum_id": f"layer-{layer:03d}",
                       "start_bytes": cursor, "end_bytes": cursor + size})
        cursor += size
    return {"plan_sha256": "a" * 64, "prepared_sha256": "b" * 64,
            "manifest_sha256": "c" * 64, "scope": scope,
            "implementation_digest": "d" * 64, "roster": roster,
            "formats_by_qname": formats,
            "parent_manifest": {"phases": phases, "total_bytes": cursor}}


def _record(campaign, layer, slice_path="slice.json.gz"):
    quantum_id = f"layer-{layer:03d}"
    qnames = sorted(q for q in campaign["roster"]
                    if f".layers.{layer}." in q)
    record = {
        "schema": RECORD_SCHEMA, "quantum_id": quantum_id, "layer": layer,
        "campaign": {
            "plan_sha256": campaign["plan_sha256"],
            "prepared_sha256": campaign["prepared_sha256"],
            "read_manifest_sha256": campaign["manifest_sha256"],
            "campaign_scope": campaign["scope"],
            "unit_roster_sha256": hashlib.sha256(
                ("\n".join(campaign["roster"]) + "\n").encode()).hexdigest(),
        },
        "read_set": {
            "manifest_path": f"manifests/{quantum_id}.data-manifest.json.gz",
            "manifest_sha256": hashlib.sha256(slice_path.encode()).hexdigest(),
            "source_phase": next(p for p in
                                 campaign["parent_manifest"]["phases"]
                                 if p["quantum_id"] == quantum_id),
        },
        "chunks": [{"name": f"{quantum_id}-chunk-000", "start_bytes": 0,
                    "end_bytes": 1024 * (layer + 1)}],
        "windows": [{"window_index": 0, "names": qnames}],
        "adjoint": {"checkpoint_boundary": N_LAYERS, "chain_layers": [],
                    "receipt_sha256": "e" * 64},
        "output_space": {"root": f"layer-quanta/{quantum_id}"},
    }
    record["identity_sha256"] = canonical_json_sha256(
        record, where="fixture layer-quantum record")
    return record


def _write_quantum(root, campaign, probe, layer, *, status="complete"):
    quantum_id = f"layer-{layer:03d}"
    record = _record(campaign, layer)
    records = root / "layer-quanta" / "records"
    records.mkdir(parents=True, exist_ok=True)
    (records / f"{quantum_id}.json").write_text(json.dumps(record))
    space = root / "layer-quanta" / quantum_id
    space.mkdir(parents=True, exist_ok=True)
    costs = {}
    for qname in record["windows"][0]["names"]:
        costs[qname] = {fmt: _row(qname, fmt, probe, sign=0.1 * (layer + 1))
                        for fmt in FORMATS}
    payload = {"costs": costs,
               "provenance": {"plan_sha256": campaign["plan_sha256"],
                              "prepared_sha256": campaign["prepared_sha256"],
                              "campaign_scope": campaign["scope"],
                              "implementation_digest":
                                  campaign["implementation_digest"],
                              "quantum_id": quantum_id,
                              "identity_sha256": record["identity_sha256"]}}
    cost_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    (space / "cost.pkl").write_bytes(cost_bytes)
    units = [len(costs), len(costs)]
    status_doc = {"schema": STATUS_SCHEMA, "quantum_id": quantum_id,
                  "identity_sha256": record["identity_sha256"],
                  "status": status, "units": units, "unix": 1750000000}
    (space / "status.json").write_text(json.dumps(status_doc))
    (space / "results.json").write_text(json.dumps({"quantum_id": quantum_id}))
    return record, hashlib.sha256(cost_bytes).hexdigest()


def _seal_inputs(root, campaign):
    """Write the CLI input files and bind their real digests into the
    campaign, so records and argv digest-check against actual bytes."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(json.dumps({"plan": "fixture"}))
    (root / "prepared.json").write_text(json.dumps({"prepared": "fixture"}))
    (root / "manifest.json").write_text(
        json.dumps(campaign["parent_manifest"]))
    (root / "scope.json").write_text(json.dumps(campaign["scope"]))
    (root / "roster.txt").write_text("\n".join(campaign["roster"]) + "\n")
    (root / "formats.json").write_text(
        json.dumps(campaign["formats_by_qname"]))
    for key, name in (("plan_sha256", "plan.json"),
                      ("prepared_sha256", "prepared.json"),
                      ("manifest_sha256", "manifest.json")):
        campaign[key] = hashlib.sha256(
            (root / name).read_bytes()).hexdigest()


def _argv(root, out, campaign):
    return ["--input-root", str(root), "--output-dir", str(out),
            "--plan", str(root / "plan.json"),
            "--plan-sha256", campaign["plan_sha256"],
            "--prepared", str(root / "prepared.json"),
            "--prepared-sha256", campaign["prepared_sha256"],
            "--manifest", str(root / "manifest.json"),
            "--manifest-sha256", campaign["manifest_sha256"],
            "--scope", str(root / "scope.json"),
            "--roster", str(root / "roster.txt"),
            "--formats-by-qname", str(root / "formats.json"),
            "--implementation-digest", campaign["implementation_digest"]]


def _run_cli(root, out, campaign):
    assert main(_argv(root, out, campaign)) == 0
    return out / "joint-cost.pkl", out / "results.json"


# -- canonical bytes under receipt permutation -------------------------------


def test_permuted_arrival_yields_canonical_bytes(tmp_path, campaign, probe):
    """The join is a disjoint union: arrival order cannot change the bytes."""
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    for layer in range(N_LAYERS):
        _write_quantum(root, campaign, probe, layer)
    first = tmp_path / "joined-first"
    cost_path, results_path = _run_cli(root, first, campaign)
    first_cost, first_results = cost_path.read_bytes(), results_path.read_bytes()

    second = tmp_path / "joined-second"
    assert main(_argv(root, second, campaign)) == 0
    assert (second / "joint-cost.pkl").read_bytes() == first_cost
    assert (second / "results.json").read_bytes() == first_results

    payload = pickle.loads(first_cost)
    assert sorted(payload["costs"]) == campaign["roster"]
    assert payload["provenance"]["coverage"]["gaps"] == []
    results = json.loads(first_results.read_text())
    assert results["schema"] == JOINED_RESULTS_SCHEMA
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
    assert gaps[0]["unit_count"] == UNITS_PER_LAYER
    results = json.loads((out / "results.json").read_text())
    assert results["status"] == "gapped"
    assert [g["quantum_id"] for g in results["distributed"]["gaps"]] == ["layer-000"]
    present = [q for q in campaign["roster"] if ".layers.0." not in q]
    assert sorted(payload["costs"]) == sorted(present)


def test_missing_status_is_a_gap_not_an_error(tmp_path, campaign, probe):
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    for layer in range(N_LAYERS):
        _write_quantum(root, campaign, probe, layer)
    (root / "layer-quanta" / "layer-001" / "status.json").unlink()
    out = tmp_path / "joined"
    assert main(_argv(root, out, campaign)) == 0
    results = json.loads((out / "results.json").read_text())
    assert results["status"] == "gapped"
    assert [g["quantum_id"] for g in results["distributed"]["gaps"]] == ["layer-001"]


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
    for layer in range(N_LAYERS):
        _write_quantum(root, campaign, probe, layer)
    cost_path, _ = _run_cli(root, tmp_path / "joined", campaign)
    payload = load_joint_cost_for_allocation(cost_path)
    assert sorted(payload["costs"]) == campaign["roster"]


# -- custody: retargeted receipts and foreign provenance refuse ----------------


def test_retargeted_receipt_refuses(tmp_path, campaign, probe):
    """A cost payload answering for another quantum's identity fails closed."""
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    records = {}
    for layer in range(N_LAYERS):
        record, _ = _write_quantum(root, campaign, probe, layer)
        records[layer] = record
    (root / "layer-quanta" / "layer-001" / "cost.pkl").write_bytes(
        (root / "layer-quanta" / "layer-002" / "cost.pkl").read_bytes())
    with pytest.raises(JoinRefused, match="custody"):
        join_joint_quanta(
            receipts=None, campaign=campaign, output_dir=tmp_path / "joined",
            input_root=root)


def test_foreign_provenance_refuses(tmp_path, campaign, probe):
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    for layer in range(N_LAYERS):
        _write_quantum(root, campaign, probe, layer)
    other = copy.deepcopy(campaign)
    other["plan_sha256"] = "f" * 64
    with pytest.raises(JoinRefused, match="custody|provenance|campaign"):
        join_joint_quanta(receipts=None, campaign=other,
                          output_dir=tmp_path / "joined", input_root=root)


# -- rows: a defective row fails the join and names the qname ------------------


def test_defective_row_fails_the_join_and_names_the_qname(tmp_path, campaign, probe):
    root = tmp_path / "campaign"
    _seal_inputs(root, campaign)
    for layer in range(N_LAYERS):
        _write_quantum(root, campaign, probe, layer)
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
