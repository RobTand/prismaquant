"""Real producer -> join -> allocator, with explicitly synthetic measurements."""
from __future__ import annotations

import copy
import hashlib
import json
import pickle

import pytest

from prismaquant import allocator
from prismaquant.joint_layer_quanta import derive_stride, layer_quanta
from prismaquant.joint_quanta_join import (
    EXIT_REFUSED, JoinRefused, join_joint_quanta, main as join_main)
from prismaquant.layer_config import load_assignment
from tests.test_joint_quanta_join import campaign, probe, STATUS_SCHEMA
from tests.test_stage_b_band_binding import band_from_receipt, synthetic_receipt
from tests.test_allocator_measured_runtime_cli import _main_fixture, admit_synthetic_table


def _generated_outputs(tmp_path, campaign):
    names = [f"model.language_model.layers.{layer}.self_attn.o_proj" for layer in range(3)]
    _, argv = _main_fixture(tmp_path, units=names)
    measured = pickle.loads((tmp_path / "costs.pkl").read_bytes())
    stats = pickle.loads((tmp_path / "probe.pkl").read_bytes())["stats"]
    formats = {name: list(measured["costs"][name]) for name in names}
    campaign = copy.deepcopy(campaign)
    campaign.update(roster=names, formats_by_qname=formats)
    root = tmp_path / "generated"
    root.mkdir()
    plan = {"output_root": str(root), "model": "/mnt/shared/models/TEST",
            "retained_window_budget_derivation": {
                "windows_by_layer": {str(layer): 1 for layer in range(3)}}}
    prepared = {"formats_by_qname": formats}
    def write_json(name, value):
        path = root / name
        path.write_text(json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    campaign['plan_sha256'] = write_json('plan.json', plan)
    campaign['prepared_sha256'] = write_json('prepared.json', prepared)
    campaign['parent_manifest']['annotations']['plan_sha256'] = campaign['plan_sha256']
    campaign['manifest_sha256'] = write_json('manifest.json', campaign['parent_manifest'])
    write_json('scope.json', campaign['scope'])
    write_json('formats.json', formats)
    (root / 'roster.txt').write_text('\n'.join(names) + '\n')
    stride = derive_stride(3, 8)
    receipt = synthetic_receipt(
        plan_sha256=campaign["plan_sha256"], prepared_sha256=campaign["prepared_sha256"],
        scope=campaign["scope"], num_layers=3, stride=8)
    assert receipt["stride"]["boundaries"] == list(stride["checkpoints"])
    campaign.pop("adjoint_bands", None)
    campaign["adjoint_receipt"] = receipt
    (root / "adjoint-capture.json").write_text(json.dumps(receipt))
    # The stage-A tail band (the whole run at stride 8) for the mode (b) join.
    (root / "band-003.json").write_text(json.dumps(band_from_receipt(receipt, 3)))
    produced = layer_quanta(plan, prepared, campaign["parent_manifest"],
        parent_manifest_sha256=campaign["manifest_sha256"], output_root=str(root),
        plan_path=str(root / 'plan.json'), plan_sha256=campaign["plan_sha256"],
        prepared_path=str(root / 'prepared.json'), prepared_sha256=campaign["prepared_sha256"],
        adjoint_receipt=receipt)
    for record in produced["records"]:
        qid = record["quantum_id"]
        record_path = root / "layer-quanta" / "records" / (qid + ".json")
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record))
        unit = names[record["layer"]]
        payload = {"schema": "prismaquant.aura_cost.v1", "n_probes": 3,
            "token_scope": "causal", "formats": formats[unit],
            "stats": {unit: stats[unit]}, "costs": {unit: measured["costs"][unit]},
            "provenance": copy.deepcopy(measured["provenance"])}
        payload["provenance"].update({
            "campaign_binding": {key: record["campaign"][key] for key in
                ("plan_sha256", "prepared_sha256", "read_manifest_sha256",
                 "campaign_scope", "unit_roster_sha256")},
            "distributed_quantum": {
                "quantum_id": qid, "identity_sha256": record["identity_sha256"],
                "adjoint_slice_sha256": record["adjoint"]["slice_sha256"],
                "checkpoint_boundary": record["adjoint"]["checkpoint_boundary"],
                "chain_layers": record["adjoint"]["chain_layers"],
                "windows": len(record["windows"]),
                "chunks": [chunk["name"] for chunk in record["chunks"]]},
            "adjoint_slice_sha256": record["adjoint"]["slice_sha256"]})
        space = root / "layer-quanta" / qid
        space.mkdir(parents=True)
        (space / "cost.pkl").write_bytes(pickle.dumps(payload))
        (space / "status.json").write_text(json.dumps({
            "schema": STATUS_SCHEMA, "quantum_id": qid,
            "identity_sha256": record["identity_sha256"], "status": "complete",
            "units": [1, 1], "unix": 1750000000}))
    return root, campaign, argv, measured, stats


def _proof_argv(root, runtime):
    """Mode (a) the completed receipt, or mode (b) its bands (PQ #993)."""
    name, flag = ("band-003.json", "--adjoint-band") if runtime else (
        "adjoint-capture.json", "--adjoint-receipt")
    path = root / name
    return [flag, str(path), flag + "-sha256", hashlib.sha256(path.read_bytes()).hexdigest()]


@pytest.mark.parametrize("runtime", [False, True])
def test_generated_quanta_reach_real_allocator_with_measured_stats(tmp_path, campaign, monkeypatch, runtime):
    root, campaign, argv, measured, stats = _generated_outputs(tmp_path, campaign)
    joined_dir = tmp_path / "joined"
    assert join_main([*_proof_argv(root, runtime),
        '--records', str(root / 'layer-quanta' / 'records'), '--output-dir', str(joined_dir),
        '--plan', str(root / 'plan.json'), '--plan-sha256', campaign['plan_sha256'],
        '--prepared', str(root / 'prepared.json'), '--prepared-sha256', campaign['prepared_sha256'],
        '--manifest', str(root / 'manifest.json'), '--manifest-sha256', campaign['manifest_sha256'],
        '--scope', str(root / 'scope.json'), '--roster', str(root / 'roster.txt'),
        '--formats-by-qname', str(root / 'formats.json')]) == 0
    path = joined_dir / "joint-cost.pkl"
    joined = pickle.loads(path.read_bytes())
    assert joined["stats"] == stats
    assert joined["costs"] == measured["costs"]
    assert len(joined["provenance"]["quantum_provenance"]) == 3
    argv[argv.index("--probe") + 1] = str(path)
    argv[argv.index("--costs") + 1] = str(path)
    if runtime:
        # Synthetic attestation only; the parser, row identities, timing
        # arithmetic, search and output writer are the real implementations.
        admit_synthetic_table(monkeypatch)
        table_path = tmp_path / "runtime.json"
        table = json.loads(table_path.read_text())
        table["cost_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        table_path.write_text(json.dumps(table))
        argv[argv.index("--slo-prefill-p95-ttft-ms") + 1] = "6"
    else:
        argv = argv[:argv.index("--measured-runtime-table")]
    allocator.main(argv[1:])
    assignment = load_assignment(tmp_path / "layer.json")
    assert assignment == dict.fromkeys(campaign["roster"], "FP8_E5M2" if runtime else "FP8_E4M3")


def test_missing_quantum_statistics_cannot_be_filled_from_another_unit(tmp_path, campaign):
    root, campaign, _, _, _ = _generated_outputs(tmp_path, campaign)
    path = root / "layer-quanta" / "layer-001" / "cost.pkl"
    payload = pickle.loads(path.read_bytes())
    payload["stats"] = {}
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(JoinRefused, match="statistics do not cover"):
        join_joint_quanta(receipts=None, campaign=campaign, input_root=root,
                          output_dir=tmp_path / "refused")


def test_a_currency_mismatch_is_refused_by_the_join(tmp_path, campaign, capsys):
    """PQ #1031: a quantum whose cost currency cannot be ranked is the join's refusal.

    ``require_run_currency`` raises ``CostCurrencyError``, a ``RuntimeError``,
    so the join must name it as a refusal rather than let it escape.
    """
    root, campaign, _, _, _ = _generated_outputs(tmp_path, campaign)
    path = root / "layer-quanta" / "layer-001" / "cost.pkl"
    payload = pickle.loads(path.read_bytes())
    payload["provenance"]["cost_currency"] = "aura_predicted_dloss"
    path.write_bytes(pickle.dumps(payload))
    joined_dir = tmp_path / "joined"
    code = join_main([*_proof_argv(root, False),
        '--records', str(root / 'layer-quanta' / 'records'), '--output-dir', str(joined_dir),
        '--plan', str(root / 'plan.json'), '--plan-sha256', campaign['plan_sha256'],
        '--prepared', str(root / 'prepared.json'), '--prepared-sha256', campaign['prepared_sha256'],
        '--manifest', str(root / 'manifest.json'), '--manifest-sha256', campaign['manifest_sha256'],
        '--scope', str(root / 'scope.json'), '--roster', str(root / 'roster.txt'),
        '--formats-by-qname', str(root / 'formats.json')])
    assert code == EXIT_REFUSED
    assert capsys.readouterr().err == (
        "joint_quanta_join: refused: allocation layer-001: "
        "joint AURA requires matching aura/joint provenance\n")
    assert not (joined_dir / "joint-cost.pkl").exists()
