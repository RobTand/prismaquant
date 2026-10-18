"""Offline stock source admission with actual GLM checkpoint/identity facts.

The GLM source, source-kind census, model loading, streamed identity and stock
CLI are real. Prices and calibration-hash label are synthetic. No AURA/KL,
full harvest, artifact or serving qualification follows. Model sources are
removed before pricing to expose any accidental live-checkpoint reread.
"""
from __future__ import annotations

from copy import deepcopy
import json
import pickle
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant import glm53_stock_reprice as stock
from prismaquant.streamed_model_identity import build_source_model_binding
from tests.native_source_contract_fixtures import (
    _source, _actual_streamed_identity, _expert_payload, _harvest,
)
from tests.test_resident_source_decode_audit import _cpu_guard, _load


CASES = ["unmarked", "legacy-v1", "same-source-v2", "foreign-source-v2",
         "config-only-v2", "root-code-metadata-v2", "legacy-native-harvest",
         "missing-harvest-identity", "corrupt-harvest-identity"]


def _actual_resident_binding(source, tmp_path, monkeypatch):
    model = _load(source, tmp_path, _cpu_guard(tmp_path, monkeypatch))
    witness = model._prismaquant_resident_source_initialization
    assert witness["status"] == "completed"
    weights = dict(model.named_parameters())
    for name in (
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.1.mlp.experts.gate_up_proj"):
        value = weights[name]
        assert value.dtype == torch.bfloat16 and not value.is_meta
        # Code 0.5 times the declared scale's BF16 rounding 2.0 is exactly1.
        assert torch.equal(value, torch.ones_like(value)), name
    return build_source_model_binding(source, dtype=torch.bfloat16)


def _write_inputs(tmp_path, probe, census, harvest, expert):
    paths = {name: tmp_path / name for name in (
        "probe.json", "census.json", "harvest.pkl", "expert.pkl")}
    paths["probe.json"].write_text(json.dumps(probe))
    paths["census.json"].write_text(json.dumps(census))
    paths["harvest.pkl"].write_bytes(pickle.dumps(harvest))
    paths["expert.pkl"].write_bytes(pickle.dumps(expert))
    output = tmp_path / "output.pkl"
    arguments = ["campaign", "--probe-census", str(paths["probe.json"]),
                 "--checkpoint-census", str(paths["census.json"]),
                 "--harvest", str(paths["harvest.pkl"]),
                 "--expert-empirical", str(paths["expert.pkl"]),
                 "--output", str(output)]
    return paths, arguments, output


@pytest.mark.parametrize("case", CASES)
def test_real_offline_stock_source_identity_admission(tmp_path, monkeypatch, case):
    source, probe, census = _source(tmp_path)
    harvest = _harvest(probe, census)
    harvest["model_identity"] = _actual_streamed_identity(source, tmp_path)
    expert = _expert_payload(probe)
    expert["provenance"]["model"] = str(source)
    source_roots = [source]
    binding = _actual_resident_binding(source, tmp_path, monkeypatch)
    # Proposed producer evidence, only after the real streamed and resident
    # source delivery controls. Never manufacture this on an old harvest.
    harvest["source_binding"] = deepcopy(binding)
    if case in {"foreign-source-v2", "config-only-v2", "root-code-metadata-v2"}:
        other = tmp_path / "other"
        other.mkdir()
        foreign = other / "glm-source"
        shutil.copytree(source, foreign)
        if case == "foreign-source-v2":
            raw = {name: value.clone() for name, value in
                   load_file(str(foreign / "model.safetensors")).items()}
            raw["lm_head.weight"].add_(0.25)
            fresh = foreign / "replacement.safetensors"
            save_file(raw, str(fresh))
            fresh.replace(foreign / "model.safetensors")
        elif case == "config-only-v2":
            document = json.loads((foreign / "config.json").read_text())
            document["text_config"]["rms_norm_eps"] *= 2
            (foreign / "config.json").write_text(json.dumps(document))
        else:
            # v2 deliberately binds every root Python file, including one
            # not imported. This checks that exact metadata contract, not a
            # claim that the toy executes custom remote code.
            (foreign / "checkpoint_policy.py").write_text("SOURCE_POLICY = 2\n")
        foreign_binding = _actual_resident_binding(foreign, other, monkeypatch)
        assert foreign_binding["checkpoint"] != binding["checkpoint"]
        if case != "foreign-source-v2":
            assert foreign_binding["checkpoint"]["shards"] == binding["checkpoint"]["shards"]
            assert foreign_binding["checkpoint"]["metadata"] != binding["checkpoint"]["metadata"]
        binding = foreign_binding
        source_roots.append(foreign)
    elif case == "legacy-v1":
        binding = {key: deepcopy(value) for key, value in binding.items()
                   if key != "source_decode"}
        binding["schema"] = "prismaquant.source_model_binding.v1"
    elif case == "legacy-native-harvest":
        del harvest["source_binding"]
    elif case == "missing-harvest-identity":
        del harvest["model_identity"]
    elif case == "corrupt-harvest-identity":
        harvest["model_identity"]["shards"][0]["sha256"] = "f" * 64
    if case != "unmarked":
        for rows in expert["costs"].values():
            for entry in rows.values():
                entry["source_binding"] = binding
    paths, arguments, output = _write_inputs(tmp_path, probe, census, harvest, expert)
    before = {name: path.read_bytes() for name, path in paths.items()}
    for root in source_roots:
        shutil.rmtree(root)
        assert not root.exists()
    if case == "same-source-v2":
        assert stock.main(arguments) == 0
        actual = pickle.loads(output.read_bytes())
        from prismaquant.prepriced_cost import require_native_cost_source_binding

        require_native_cost_source_binding(
            actual, harvest["source_binding"], where="offline generated stock output")
        for name, rows in expert["costs"].items():
            assert actual["costs"][name] == rows
    else:
        with pytest.raises((ValueError, RuntimeError, stock.Glm53StockError),
                           match="(?i)(native|decode|source|model.identity)"):
            stock.main(arguments)
        assert not output.exists()
    assert {name: path.read_bytes() for name, path in paths.items()} == before


def test_real_bf16_packed_stock_scope_keeps_existing_ladder_refusal(tmp_path):
    source, probe, census = _source(tmp_path, native=False)
    assert all(row["source_kind"] == "bf16" for row in census["units"].values())
    harvest = _harvest(probe, census)
    harvest["model_identity"] = _actual_streamed_identity(source, tmp_path)
    paths, arguments, output = _write_inputs(
        tmp_path, probe, census, harvest, _expert_payload(probe))
    before = {name: path.read_bytes() for name, path in paths.items()}
    with pytest.raises(stock.Glm53StockError, match="non-dense declarations survived"):
        stock.main(arguments)
    assert not output.exists()
    assert {name: path.read_bytes() for name, path in paths.items()} == before
