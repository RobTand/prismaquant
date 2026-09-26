"""AURA stats rows carry the per-unit topology a scoped allocation reads (PQ #1278).

``tessera_serving_scope.unit_structure_from_stats`` classifies each unit from
facts its producer recorded. The incremental probe writes them; the AURA cost
payload did not, so a scoped Tessera allocation over any AURA table stopped at
``context_by_unit_from_stats`` with "missing per-unit router_path/expert_id
topology". These tests hold both halves of the fix: the producer writes the
probe's facts, and a table built before that can be re-stamped from the model
profile's declared grammar with the source recorded on every row it stamps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle

import pytest
import torch
import torch.nn as nn


IMAGE = "example/runtime@sha256:" + "a" * 64
DENSE = "model.layers.0.self_attn.o_proj"
EXPERT = "model.layers.0.mlp.experts.0.down_proj"
SHARED = "model.layers.0.mlp.shared_expert.down_proj"
ROUTER = "model.layers.0.mlp.gate"
PACKED_MODULE = "model.layers.1.mlp.experts"
PACKED_VIEWS = tuple(f"{PACKED_MODULE}.{expert}.down_proj" for expert in range(2))
AURA_ROW_KEYS = {"h_trace", "n_params", "in_features", "out_features", "n_probes"}


def _scope():
    from prismaquant import tessera_serving_scope
    return tessera_serving_scope


def _target():
    return _scope().serving_target_from_args(argparse.Namespace(
        tessera_platform="sm_121", tessera_runtime_image=IMAGE,
        tessera_execution_mode="eager", tessera_residency="resident"))


def _profile():
    from prismaquant.model_profiles import Qwen3Profile
    return Qwen3Profile()


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(8, 8, bias=False)


class _Moe(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 2, bias=False)
        self.experts = nn.ModuleList([_Expert(), _Expert()])
        self.shared_expert = _Expert()


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(8, 8, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _Moe()


class _Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Body()


def _packed_members():
    from prismaquant.routed_experts import PackedExpertProjection
    holder = nn.Module()
    return [PackedExpertProjection(
        qname=name, packed_qname=f"{PACKED_MODULE}.down_proj", module_qname=PACKED_MODULE,
        module=holder, param_name="down_proj", expert_id=expert,
        projection_name="down_proj", weight=torch.zeros(8, 8))
        for expert, name in enumerate(PACKED_VIEWS)]


def _aura_payload():
    """Stats rows exactly as the streamed AURA producer assembles them."""
    from prismaquant import aura_cost
    model = _Model()
    modules = dict(model.named_modules())
    members = _packed_members()
    linears = {name: modules[name] for name in (DENSE, EXPERT, SHARED)}
    linears.update({member.qname: member for member in members})
    topology = aura_cost.aura_unit_topology(
        model, linears, profile=_profile(), packed_members=members)
    return aura_cost._assemble_streamed_aura_payload(
        linears=linears, names=list(linears), formats=["BF16"],
        formats_by_qname={name: ("BF16",) for name in linears},
        unmeasured_formats_by_qname={}, n_probes=2, token_scope="all",
        seed_base=0, temperature=1.0, dw_dtype="float32",
        measurement_dtype=torch.float32, n_linear_chunks=1,
        calib_ids=torch.zeros(1, 4, dtype=torch.long),
        omitted_packed_experts=[],
        checkpoint_git_commit="test", collect_col_energy=False,
        s2={}, s4={}, x2_probe={}, dw_src={},
        g_trace={name: 1.0 for name in linears}, col_energy={},
        weight_mse_diagnostic={}, unit_topology=topology)


def test_aura_stats_rows_classify_dense_and_routed_units_under_a_serving_target():
    stats = _aura_payload()["stats"]
    contexts = _scope().context_by_unit_from_stats(_target(), stats, _profile())
    assert {name: context.structure for name, context in contexts.items()} == {
        DENSE: "dense", SHARED: "dense", EXPERT: "routed_moe",
        PACKED_VIEWS[0]: "routed_moe", PACKED_VIEWS[1]: "routed_moe"}
    # The facts are the probe's own spelling, from the probe's own sources.
    assert stats[DENSE]["router_path"] is None and stats[DENSE]["expert_id"] is None
    assert (stats[EXPERT]["router_path"], stats[EXPERT]["expert_id"]) == (ROUTER, "0")
    assert stats[PACKED_VIEWS[1]]["_packed_experts_module"] == PACKED_MODULE
    assert stats[PACKED_VIEWS[1]]["num_experts"] == 2
    assert all("unit_topology_source" not in row for row in stats.values())


def test_undiscoverable_routed_linear_carries_no_false_dense_fact():
    """A profile-declared expert the router walk cannot place stays unknown."""
    from prismaquant import aura_cost
    model = _Model()
    del model.model.layers[0].mlp.gate  # no router sibling to discover
    linears = {EXPERT: dict(model.named_modules())[EXPERT]}
    topology = aura_cost.aura_unit_topology(model, linears, profile=_profile())
    assert topology == {EXPERT: {}}
    with pytest.raises(ValueError, match="missing per-unit router_path/expert_id"):
        _scope().context_by_unit_from_stats(
            _target(), {EXPERT: {"h_trace": 1.0, **topology[EXPERT]}}, _profile())


def _legacy_rows():
    """An AURA table built before the fix: five keys and no topology."""
    return {name: {"h_trace": 1.0, "n_params": 64, "in_features": 8,
                   "out_features": 8, "n_probes": 4}
            for name in (DENSE, EXPERT, SHARED, *PACKED_VIEWS)}


def test_legacy_aura_rows_still_refuse_without_a_recorded_source():
    with pytest.raises(ValueError, match="missing per-unit router_path/expert_id topology"):
        _scope().context_by_unit_from_stats(_target(), _legacy_rows(), _profile())


def test_restamp_classifies_legacy_rows_from_the_profile_grammar_and_labels_them():
    scope = _scope()
    payload = {"stats": _legacy_rows(), "costs": {}, "provenance": {"model": "m"}}
    stamped, summary = scope.restamp_unit_topology(payload, _profile())
    contexts = scope.context_by_unit_from_stats(_target(), stamped["stats"], _profile())
    assert {name: context.structure for name, context in contexts.items()} == {
        DENSE: "dense", SHARED: "dense", EXPERT: "routed_moe",
        PACKED_VIEWS[0]: "routed_moe", PACKED_VIEWS[1]: "routed_moe"}
    assert all(row["unit_topology_source"] == "profile_grammar"
               for row in stamped["stats"].values())
    # No probe fact is manufactured: the stamp is its own labelled form.
    assert all(not ({"router_path", "expert_id", "_packed_experts_module", "num_experts"}
                    & set(row)) for row in stamped["stats"].values())
    assert summary["sources"] == {"producer": 0, "profile_grammar": 5}
    assert summary["structures"] == {"dense": 2, "routed_moe": 3}
    assert stamped["provenance"]["unit_topology_restamp"] == summary
    # The input table is not modified.
    assert all(set(row) == AURA_ROW_KEYS for row in payload["stats"].values())
    assert "unit_topology_restamp" not in payload["provenance"]


def test_restamp_keeps_producer_facts_and_counts_both_sources():
    scope = _scope()
    rows = _legacy_rows()
    rows[EXPERT].update(router_path=ROUTER, expert_id="0")
    stamped, summary = scope.restamp_unit_topology({"stats": rows, "provenance": {}}, _profile())
    assert stamped["stats"][EXPERT] == rows[EXPERT]
    assert summary["sources"] == {"producer": 1, "profile_grammar": 4}


@pytest.mark.parametrize("row,match", [
    ({"unit_structure": "dense"}, "unrecognised unit topology stamp"),
    ({"unit_structure": "dense", "unit_topology_source": "guess"},
     "unrecognised unit topology stamp"),
    ({"unit_structure": "dense", "unit_topology_source": "profile_grammar"},
     "differs from the live profile"),
    ({"unit_structure": "routed_moe", "unit_topology_source": "profile_grammar",
      "router_path": None, "expert_id": None}, "both producer topology and a profile-grammar stamp"),
])
def test_a_stamp_is_checked_against_the_live_profile(row, match):
    with pytest.raises(ValueError, match=match):
        _scope().context_by_unit_from_stats(_target(), {EXPERT: row}, _profile())


def test_restamp_cli_writes_a_new_table_and_receipt(tmp_path, monkeypatch):
    from prismaquant import model_profiles, unit_topology_restamp as tool
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _: _profile())
    source = tmp_path / "joint-allocation.pkl"
    raw = pickle.dumps({"stats": _legacy_rows(), "costs": {},
                        "provenance": {"model": "/models/example"}})
    source.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    output = tmp_path / "joint-allocation.topology.pkl"
    with pytest.raises(ValueError, match="sha256"):
        tool.main(["--table", str(source), "--table-sha256", "0" * 64, "--output", str(output)])
    assert not output.exists()
    assert tool.main(["--table", str(source), "--table-sha256", digest,
                      "--output", str(output)]) == 0
    assert source.read_bytes() == raw
    written = output.read_bytes()
    receipt = json.loads((tmp_path / "joint-allocation.topology.pkl.receipt.json").read_text())
    assert receipt["input"] == {"path": str(source.resolve()), "sha256": digest}
    assert receipt["output"] == {"path": str(output.resolve()),
                                 "sha256": hashlib.sha256(written).hexdigest()}
    assert receipt["model"] == "/models/example"
    assert receipt["summary"]["sources"] == {"producer": 0, "profile_grammar": 5}
    table = pickle.loads(written)
    assert table["provenance"]["unit_topology_restamp"]["input_sha256"] == digest
    with pytest.raises(Exception, match="exists"):
        tool.main(["--table", str(source), "--table-sha256", digest, "--output", str(output)])
