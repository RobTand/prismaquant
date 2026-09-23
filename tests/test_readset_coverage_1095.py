"""The readset and the loader enumerate one set of source reads (PQ #1095).

The first strictly staged Stage B quantum refused its first head tensor: the
executable readset never declared the resident head the streaming loader
materializes. These tests hold the fix:

* the loader's selection and the readset's are one function
  (``source_read_plan``), and a sealed head the loader does not select
  refuses before a byte is read;
* the head phase stages the resident head, and the bound record carries it;
* a quantum installs and prefetches only its own walk;
* the pre-submission check lists every undeclared read in one pass, and the
  dispatcher refuses a package it finds gaps in, dry run included.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from prismaquant import joint_layer_quanta as jl
from prismaquant import readset_coverage as rc
from prismaquant import source_read_plan as srp
from prismaquant.layer_streaming import streaming_source_plan
from test_stageb_readset_source_coverage import (
    HEAD_TENSORS, PREFIX, _build, _source_model, _with_source_paths,
)
from test_quantum_executable_readset import _bound_inputs


# -- one enumeration -----------------------------------------------------------

def test_head_prefixes_follow_the_layers_prefix():
    assert srp.base_prefix_of_layers("model.language_model.layers.") == \
        "model.language_model"
    assert srp.base_prefix_of_layers("layers.") == ""
    with pytest.raises(ValueError):
        srp.base_prefix_of_layers("model.blocks.")
    assert srp.resident_head_prefixes("model.language_model", ["hc_head.", "lm_head."]) == [
        "model.language_model.embed_tokens.", "model.language_model.norm.",
        "lm_head.", "model.language_model.rotary_emb.", "hc_head."]
    assert srp.resident_head_prefixes("")[0] == "embed_tokens."


def test_roster_names_one_live_layers_prefix():
    assert srp.roster_layers_prefix([
        "model.language_model.layers.3.mlp.experts.7.down_proj",
        "model.language_model.layers.10.self_attn.q_proj", "lm_head"]) == \
        "model.language_model.layers."
    assert srp.roster_layers_prefix(["model.layers.0.mlp.layers.1.x"]) == "model.layers."
    with pytest.raises(ValueError, match="not one"):
        srp.roster_layers_prefix(["model.layers.0.a", "model.language_model.layers.1.b"])


def test_the_plan_selects_what_the_loader_selects(tmp_path):
    model, spans = _source_model(tmp_path)
    plan = streaming_source_plan(str(model), layers_prefix=PREFIX, layers=range(4))
    assert plan["multimodal"] is False
    assert plan["head_tensors"] == HEAD_TENSORS
    assert {plan["span_tensors"][span] for span in plan["head_spans"]} == {
        name for _shard, name in HEAD_TENSORS}
    assert plan["layer_spans"] == spans
    # The index whole, then one header read per shard the loader opens.
    assert [read[0].rsplit("/", 1)[1] for read in plan["header_reads"]] == [
        "model.safetensors.index.json"] + [f"layer-{n}.safetensors" for n in range(4)]


def test_a_sealed_head_the_loader_does_not_select_refuses_before_reading():
    selection = {"/m/a.safetensors": [("model.embed_tokens.weight",
                                       "model.language_model.embed_tokens.weight")]}
    sealed = [["a.safetensors", "model.language_model.embed_tokens.weight"]]
    srp.check_sealed_selection(selection, sealed, what="resident head")
    with pytest.raises(RuntimeError, match="refusing before the first read"):
        srp.check_sealed_selection(selection, sealed + [["b.safetensors", "lm_head.weight"]],
                                   what="resident head")


# -- the head phase ------------------------------------------------------------

def _head_source(plan):
    return {"layers_prefix": plan["layers_prefix"], "tensors": plan["head_tensors"],
            "spans": plan["head_spans"]}


def _phase(manifest, name):
    return rc.phase_entries(manifest)[name]


def test_the_head_phase_stages_the_resident_head_and_the_record_carries_it(tmp_path):
    record, receipt, parent, kwargs = _bound_inputs(tmp_path)
    model, spans = _source_model(tmp_path)
    parent = _with_source_paths(parent, model)
    plan = streaming_source_plan(str(model), layers_prefix=PREFIX, layers=range(4))
    legacy = _build(record, receipt, parent, layer_source_spans=spans)
    assert srp.uncovered_spans(_phase(legacy, "head"), plan["head_spans"])
    manifest = _build(record, receipt, parent, layer_source_spans=spans,
                      head_source=_head_source(plan))
    assert not srp.uncovered_spans(_phase(manifest, "head"), plan["head_spans"])
    note = manifest["annotations"]["head_source"]
    assert note["tensors"] == HEAD_TENSORS and note["layers_prefix"] == PREFIX
    keys = [(entry["path"], entry["offset"]) for entry in manifest["entries"]]
    assert len(set(keys)) == len(keys)
    kwargs.update(manifest=manifest, layer_source_spans=spans,
                  head_source=_head_source(plan),
                  manifest_sha256=hashlib.sha256(
                      jl.seal_manifest_bytes(manifest)).hexdigest())
    bound = jl.bind_quantum_executable(record, receipt, parent, **kwargs)
    assert bound["executable_readset"]["head_source"] == {
        "schema": jl.HEAD_SOURCE_SCHEMA, "layers_prefix": PREFIX,
        "tensors": HEAD_TENSORS}
    # A readset built without the head does not derive from these inputs.
    kwargs.update(manifest=legacy, manifest_sha256=hashlib.sha256(
        jl.seal_manifest_bytes(legacy)).hexdigest())
    with pytest.raises(ValueError, match="do not originate"):
        jl.bind_quantum_executable(record, receipt, parent, **kwargs)


@pytest.mark.parametrize("defect", ["no-spans", "unsorted", "relative", "prefix"])
def test_a_malformed_head_source_refuses(tmp_path, defect):
    record, receipt, parent, _ = _bound_inputs(tmp_path)
    head = {"layers_prefix": PREFIX, "tensors": [["a", "x"], ["b", "y"]],
            "spans": [("/m/a", 10, 20)]}
    if defect == "no-spans":
        head["spans"] = []
    elif defect == "unsorted":
        head["tensors"] = [["b", "y"], ["a", "x"]]
    elif defect == "relative":
        head["spans"] = [("m/a", 10, 20)]
    else:
        head["layers_prefix"] = "model.blocks."
    with pytest.raises(ValueError, match="refusing"):
        _build(record, receipt, parent, head_source=head)


# -- the walk ------------------------------------------------------------------

def _walk(order, lookahead, *, operator_windows=None):
    from prismaquant.joint_cost_quantum import _install_with_settlement
    events = []
    context = SimpleNamespace(
        schedule_prefetch=lambda layer: events.append(("prefetch", layer)),
        install=lambda layer, **kw: events.append(("install", layer, kw["prefetch_following"])),
        settle_prefetched_layers=lambda layers: events.append(("settle", tuple(layers))))
    runner = SimpleNamespace(context=context, prefetch_lookahead=lookahead,
                             require_prefetched_residency=True, device="cpu")
    for layer in order:
        _install_with_settlement(runner, layer, operator_windows=operator_windows,
                                 order=order)
    return events


def test_a_quantum_asks_for_its_first_layer_and_reads_nothing_past_its_walk():
    assert _walk((44,), 1) == [("prefetch", 44), ("install", 44, False)]
    events = _walk((46, 45, 44), 2, operator_windows=object())
    assert {event[1] for event in events if event[0] == "prefetch"} == {46, 45, 44}
    assert events[:2] == [("prefetch", 46), ("install", 46, False)]
    assert ("settle", ()) == events[-1]
    assert rc.quantum_prefetch_targets((46, 45, 44), 3) == {46, 45, 44}
    assert jl.quantum_source_layer_order([46, 45], 44) == (46, 45, 44)


def test_the_stage_a_schedule_stops_at_its_walk():
    """PQ #1100: the Stage A chain reads nothing below its last layer."""
    order = list(range(44, 39, -1))
    assert rc.stage_a_prefetch_targets(order, 2) == set(order)
    assert rc.quantum_prefetch_targets(order, 2) == set(order)
    assert srp.chain_opening_window(order, 2) == (44, 43)


# -- the check -------------------------------------------------------------------

def _bound_manifest(tmp_path, *, with_head):
    record, receipt, parent, kwargs = _bound_inputs(tmp_path)
    model, spans = _source_model(tmp_path)
    parent = _with_source_paths(parent, model)
    plan = streaming_source_plan(str(model), layers_prefix=PREFIX, layers=range(4))
    head = _head_source(plan) if with_head else None
    manifest = _build(record, receipt, parent, layer_source_spans=spans,
                      head_source=head)
    kwargs.update(manifest=manifest, layer_source_spans=spans, head_source=head,
                  manifest_sha256=hashlib.sha256(
                      jl.seal_manifest_bytes(manifest)).hexdigest())
    return jl.bind_quantum_executable(record, receipt, parent, **kwargs), manifest, plan


def test_red_names_every_resident_head_tensor_and_the_unsealed_selection(tmp_path):
    record, manifest, plan = _bound_manifest(tmp_path, with_head=False)
    gaps = rc.quantum_record_gaps(record, manifest, plan, lookahead=1)
    assert {gap["kind"] for gap in gaps} == {"resident-head", "head-selection-unsealed"}
    assert sorted(gap["tensor"] for gap in gaps if gap["kind"] == "resident-head") == \
        sorted(name for _shard, name in HEAD_TENSORS)
    assert all(gap["quantum_id"] == "layer-002" for gap in gaps)


def test_green_is_empty_and_a_dropped_layer_phase_is_named(tmp_path):
    record, manifest, plan = _bound_manifest(tmp_path, with_head=True)
    assert rc.quantum_record_gaps(record, manifest, plan, lookahead=1) == []
    for phase in manifest["read_plan"]["phases"]:
        if phase["name"] == "own-002-source":
            phase["entry_indices"] = []
    gaps = rc.quantum_record_gaps(record, manifest, plan, lookahead=1)
    assert gaps and {(gap["kind"], gap["layer"]) for gap in gaps} == {("layer-source", 2)}


def test_a_prefetch_outside_the_walk_is_a_gap(tmp_path):
    _record, manifest, plan = _bound_manifest(tmp_path, with_head=True)
    gaps = rc.source_read_gaps(
        manifest, plan, order=[3, 2], prefetch_targets={3, 2, 1},
        phase_of=lambda layer: {3: "chain-003-source", 2: "own-002-source"}[layer])
    assert [(gap["kind"], gap["layer"]) for gap in gaps] == [("prefetch-outside-walk", 1)]


def test_the_dispatcher_refuses_a_gap_in_one_pass(capsys):
    import dispatch_joint_quanta as dispatch
    seen = []

    def coverage(rows):
        seen.append(len(rows))
        return [{"kind": "resident-head", "phase": "head", "layer": None,
                 "quantum_id": "layer-044", "tensor": "lm_head.weight"},
                {"kind": "head-selection-unsealed", "phase": "head", "layer": None,
                 "quantum_id": "layer-044"}]
    with pytest.raises(dispatch.DispatchRefused, match="2 source read"):
        dispatch.check_source_coverage([{"record": {}}], coverage=coverage)
    report = json.loads(capsys.readouterr().err)
    assert report["gap_count"] == 2 and not report["covered"]
    assert seen == [1]
    dispatch.check_source_coverage([], coverage=coverage)
    assert seen == [1]
