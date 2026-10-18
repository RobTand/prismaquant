"""Actual retained-cost intake, with explicit synthetic measurement metadata.

Source headers/bytes and the qualified control's resident source decode are
real. Cost numbers are synthetic; no quality, timing or artifact claim follows.
Stock admission is covered separately with actual GLM source/identity fixtures.
No reader/model/dequantizer, admission checker or final writer is mocked.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest
import torch

from prismaquant import prepriced_cost
from prismaquant.streamed_model_identity import build_source_model_binding
from tests.test_resident_source_decode_audit import (
    _check_values, _cpu_guard, _load, _source,
)


ROOT = Path(__file__).resolve().parents[1]
V1 = "prismaquant.source_model_binding.v1"
V2 = "prismaquant.source_model_binding.v2"
DECODE = "prismaquant.native_fp8_source_decode.v1"
CASES = ["native-unmarked", "native-v1", "native-stream-label-only",
         "native-qualified", "bf16-unmarked"]


def _qualified_binding(root, expected, tmp_path, monkeypatch):
    # Baseline-compatible: v1 builders lack the new schema, but the resident
    # owner already delivers this representation. Only this fresh control
    # constructs the proposed marker, after its actual value/witness checks.
    model = _load(root, tmp_path, _cpu_guard(tmp_path, monkeypatch))
    _check_values(model, expected, torch.bfloat16, False)
    binding = build_source_model_binding(root, dtype=torch.bfloat16)
    return {**binding, "schema": V2, "source_decode": DECODE}


def _decorate(provenance, case, root, expected, tmp_path, monkeypatch):
    provenance.update(model=str(root), cost_mode="aura")
    if case == "native-v1":
        binding = build_source_model_binding(root, dtype=torch.bfloat16)
        provenance["source_binding"] = {
            "schema": V1, "checkpoint": deepcopy(binding["checkpoint"]),
            "effective_dtype": binding["effective_dtype"],
        }
    elif case in {"native-qualified", "native-mixed-unbound-backfill"}:
        provenance["source_binding"] = _qualified_binding(
            root, expected, tmp_path, monkeypatch)
    elif case == "native-stream-label-only":
        provenance.update(streaming=True, eval_driver="forked-stream/1")


def _cost_payload(root, expected, case, tmp_path, monkeypatch):
    names = sorted(name.removesuffix(".weight") for name in expected
                   if name.endswith(".self_attn.o_proj.weight"))
    assert len(names) == 2
    names = names if case == "native-mixed-unbound-backfill" else names[:1]
    payload = {
        "schema": "prismaquant.expert_empirical_cost.v1",
        "formats": ["NVFP4"],
        "costs": {name: {"NVFP4": {
            "predicted_dloss": 0.125, "cost_source": "empirical_unit_kl",
        }} for name in names},
        "provenance": {},
    }
    _decorate(payload["provenance"], case, root, expected, tmp_path, monkeypatch)
    if case in {"native-qualified", "native-v1", "native-mixed-unbound-backfill"}:
        for name in names[:1]:
            for entry in payload["costs"][name].values():
                entry["source_binding"] = payload["provenance"]["source_binding"]
    if case == "native-mixed-unbound-backfill":
        # This matches the existing mixed product's metadata shape: the
        # marker qualifies fresh measured rows, never copied legacy rows.
        payload["provenance"]["backfilled_from_base"] = names[1:]
        payload["provenance"]["backfill_base"] = "unbound-old-cost.pkl"
        payload["provenance"]["backfill_provenance"] = {
            "model": str(root), "cost_mode": "local",
        }
    return payload


def _pipeline_reusable(path, root):
    # Same actual function extraction seam as the existing provenance test.
    # No outer campaign, workers, measurements or fake Python executable.
    driver = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    block = "cost_table_cost_mode() {" + driver.split(
        "cost_table_cost_mode() {", 1)[1].split(
        "\nrequire_stage_settings() {", 1)[0]
    environment = dict(os.environ, COST_MODE="aura", MODEL_PATH=str(root))
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    return subprocess.run(
        ["bash", "-c", block + '\ncost_table_reusable "$1"\n', "bash", str(path)],
        cwd=ROOT, env=environment, text=True, capture_output=True, timeout=120,
    )


@pytest.mark.parametrize("site", ["pipeline", "prepriced"])
@pytest.mark.parametrize("case", CASES + ["native-mixed-unbound-backfill"])
def test_actual_retained_cost_intake_requires_qualified_native_source(
        tmp_path, monkeypatch, site, case):
    native = case != "bf16-unmarked"
    root, expected = _source(tmp_path, native=native)
    payload = _cost_payload(root, expected, case, tmp_path, monkeypatch)
    path = tmp_path / "retained cost.pkl"
    path.write_bytes(pickle.dumps(payload))
    before = path.read_bytes()
    refuses = case not in {"native-qualified", "bf16-unmarked"}
    if site == "pipeline":
        result = _pipeline_reusable(path, root)
        if refuses:
            assert result.returncode != 0, (
                "actual pipeline reuse predicate admitted unqualified native "
                f"product ({case}): stdout={result.stdout!r} stderr={result.stderr!r}")
            assert any(word in (result.stdout + result.stderr).lower()
                       for word in ("native", "decode", "source binding"))
        else:
            assert result.returncode == 0, result.stdout + result.stderr
    elif refuses:
        with pytest.raises(ValueError, match="(?i)(native|decode|source.*binding)"):
            prepriced_cost.validate_prepriced_cost(path, cost_mode="aura", model=str(root))
    else:
        receipt = prepriced_cost.validate_prepriced_cost(
            path, cost_mode="aura", model=str(root))
        assert receipt["units"] == len(payload["costs"])
    assert path.read_bytes() == before, "intake must not repair/restamp old costs"



@pytest.mark.parametrize('change', [
    'valid', 'forged-digest', 'opaque-label', 'unplanned-cell',
    'forged-source', 'incomplete-records', 'invalid-shape',
])
def test_actual_sparse_anchor_contract_keeps_its_independent_source_regime(
        tmp_path, monkeypatch, change):
    """Real render/identity owners, synthetic scalar, no AURA quality claim."""
    from prismaquant.model_profiles import profile_from_model
    from prismaquant.streaming_production_cache import StreamedProductionAnchorRenderer
    from tests.test_native_source_binding_seed_audit import _proof

    root, expected = _source(tmp_path)
    model = _load(root, tmp_path, _cpu_guard(tmp_path, monkeypatch))
    qname = next(name.removesuffix('.weight') for name in sorted(expected)
                 if name.endswith('.self_attn.o_proj.weight'))
    proof = _proof(root, tmp_path)
    source_identity = json.loads(proof.read_text())['identity']
    fmt = 'FP8_E4M3'
    module = model.get_submodule(qname)

    class Activations:
        def __contains__(self, name):
            return name == qname

        def load_with_row_indices(self, name):
            assert name == qname
            return torch.ones(4, module.weight.shape[1]), None

    renderer = StreamedProductionAnchorRenderer(
        model, act_index=Activations(), formats_by_qname={qname: [fmt]},
        levers={'gptq': False, 'scale_sweep': False, 'joint_scale_opt': False},
        profile=profile_from_model(model), device='cpu', col_weights={},
        calibration_hash='synthetic-four-row-control',
        arm_identity={'arm': 'actual-render-synthetic-cost-control'},
        model_identity=source_identity, max_act_rows=4)
    rendered = renderer.render_layer(
        layer=0, modules={qname: module}, formats_by_qname={qname: [fmt]})
    assert set(rendered) == {(qname, fmt)}
    assert torch.isfinite(rendered[qname, fmt]).all()
    anchor = renderer.bind_completed_source_weight_identities({
        qname: renderer.source_weight_identity_for(qname)})
    assert renderer.render_count == 1
    payload = {'costs': {qname: {fmt: {
        'predicted_dloss': 0.125, 'cost_source': 'aura',
        'dw_source': 'production_render', 'production_anchor_measured': True,
    }}}, 'provenance': {'production_anchor_renderer': anchor}}
    if change == 'forged-digest':
        anchor['source_weights']['identity_sha256'] = '0' * 64
    elif change == 'opaque-label':
        payload['provenance'] = {'streaming': True, 'production_anchor_renderer': {'schema': anchor['schema']}}
    elif change == 'unplanned-cell':
        payload['costs']['unplanned'] = payload['costs'].pop(qname)
    elif change == 'forged-source':
        anchor['source_model']['shards'][0]['sha256'] = 'f' * 64
    elif change == 'incomplete-records':
        anchor['source_weights']['records'].clear()
    elif change == 'invalid-shape':
        anchor['source_weights']['records'][qname]['shape'] = [True, 4]
    if change == 'valid':
        prepriced_cost.validate_native_cost_source_for_model(payload, model=str(root), where='actual anchor')
    else:
        with pytest.raises((ValueError, RuntimeError)):
            prepriced_cost.validate_native_cost_source_for_model(payload, model=str(root), where='actual anchor')


def test_native_intake_retains_explicit_fp32_priced_representation(tmp_path, monkeypatch):
    root, expected = _source(tmp_path)
    model = _load(root, tmp_path, _cpu_guard(tmp_path, monkeypatch), dtype=torch.float32)
    _check_values(model, expected, torch.float32, False)
    payload = _cost_payload(root, expected, 'native-unmarked', tmp_path, monkeypatch)
    binding = build_source_model_binding(root, dtype=torch.float32)
    for rows in payload['costs'].values():
        for row in rows.values():
            row['source_binding'] = binding
    prepriced_cost.validate_native_cost_source_for_model(payload, model=str(root), where='actual FP32-priced source')
