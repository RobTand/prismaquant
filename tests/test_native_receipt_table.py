"""The v2 table emitter binds real receipts or refuses by name (CPU; synthetic evidence).

Everything the emitter writes is re-read through the production loader; the
tests assert the loader's verdict, never the emitter's own summary. The
fixtures are the synthetic relation and panel contracts the admission suites
already use, so a table this emitter writes reaches ``admit_native_rows`` and
is admitted there, and reaches ``admit_fixed_resources`` and is refused there.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
import pickle
from types import SimpleNamespace

import pytest

from prismaquant import native_receipt_table as emitter
from prismaquant.joint_aura import make_joint_aura_entry
from prismaquant.measured_runtime_prices import (
    RuntimePriceError, identity_sha256, parse_measured_runtime_table, parse_runtime_context,
)
from prismaquant.native_operator_panel import freeze_native_panel
from prismaquant.runtime_provenance import admit_native_rows
from test_native_operator_panel import joined, receipt_fixture
from test_runtime_fixed_resource_admission import agreeing_report
from test_runtime_provenance import relation_fixture, relation_load
from test_full_engine_resource_report import written

NOW = datetime(2026, 9, 13, 12, tzinfo=timezone.utc)
FIXED_PREFIX = "no qualified recomputable full-engine resource partition: "


def _cell(joined_cell, unit=None):
    """A copy of the fixture cell, optionally renamed to a second unit."""
    inputs, preflight, joint = (copy.deepcopy(part) for part in joined_cell)
    if unit is not None:
        inputs["unit"] = unit
        operator = dict(joint["joint_operator_identity"], qname=unit)
        joint = make_joint_aura_entry(operator_identity=operator, probe_identity=joint["probe_identity"],
                                      signed_components=joint["signed_components_per_probe"])
    return inputs, preflight, joint


def _bind_to_relation(cell, relation_fixture):
    """Point the cell's runtime and wire seal at the synthetic relation's native run."""
    evidence, relation, _ = relation_fixture
    inputs, preflight, joint = cell
    raw = evidence.get(relation["runs"]["native"]["runtime"])
    inputs["runtime_image"] = raw["image"]
    source_tree_sha = relation_load(relation_fixture)["runs"]["native"]["common"]["producer_source_tree_sha256"]
    inputs["wire"]["record"]["identity"] = {"encoder_source_sha256": source_tree_sha}
    preflight["operator"]["wire_record_sha256"] = identity_sha256(inputs["wire"]["record"])
    raw["execution"] = copy.deepcopy(preflight["runtime"]["execution"])
    evidence.replace(relation["runs"]["native"]["runtime"], raw)
    preflight["runtime"] = raw
    preflight["runtime_sha256"] = identity_sha256(raw)
    return inputs, preflight, joint, raw


def _write_cell(root, name, cell, raw, cost_sha256):
    inputs, preflight, joint = cell
    panel, receipt, trace = receipt_fixture((inputs, preflight, joint), complete=True, cost_sha256=cost_sha256)
    trace["capture"]["collector_library_sha256"] = raw["resource_collector"]["library_sha256"]
    receipt["resources"]["trace_sha256"] = identity_sha256(trace)
    paths = {}
    for kind, value in (("panel", panel), ("receipt", receipt), ("memory_trace", trace)):
        path = root / f"{name}.{kind}.json"
        path.write_text(json.dumps(value, sort_keys=True))
        paths[kind] = str(path)
    return {"unit": inputs["unit"], "format": inputs["format"], "run_id": "native", **paths}, panel


@pytest.fixture
def emitted(tmp_path, relation_fixture, joined):
    """Everything one emission needs, bound to the synthetic relation and cost bytes."""
    evidence, relation, _ = relation_fixture
    root = tmp_path / "emit"
    root.mkdir()
    inputs, preflight, joint, raw = _bind_to_relation(_cell(joined), relation_fixture)
    cost = {"costs": {inputs["unit"]: {inputs["format"]: joint}}, "stats": {}, "provenance": {}}
    cost_path = root / "cost.pkl"
    cost_path.write_bytes(pickle.dumps(cost))
    cost_sha256 = hashlib.sha256(cost_path.read_bytes()).hexdigest()
    binding, panel = _write_cell(root, "cell", (inputs, preflight, joint), raw, cost_sha256)
    report = written(root, agreeing_report(), "report.json")
    relation_load(relation_fixture)          # writes relation.json with the rebound native run
    manifest = root / "receipts.json"
    manifest.write_text(json.dumps([binding]))
    return SimpleNamespace(root=root, evidence=evidence, relation=relation, relation_fixture=relation_fixture,
                           cost_path=cost_path, cost=cost, cost_sha256=cost_sha256, binding=binding, panel=panel,
                           raw=raw, report=report, manifest=manifest, cell=(inputs, preflight, joint),
                           out=root / "table.json")


def _argv(state, **overrides):
    values = {"--out": state.out, "--table-id": "synthetic-emission", "--costs": state.cost_path,
              "--relation": state.evidence.root / "relation.json", "--full-engine-report": state.report["path"],
              "--receipts": state.manifest, "--now": NOW.isoformat()}
    values.update(overrides)
    return [token for flag, value in values.items() for token in (flag, str(value))]


def test_emitted_table_is_admitted_by_admit_native_rows_and_refused_only_at_fixed_resources(emitted, capsys):
    code = emitter.main(_argv(emitted))
    printed = capsys.readouterr().out
    emission = json.loads((emitted.root / "table.emission.json").read_text())
    admission = emission["admission"]
    assert json.loads(printed)["admission"] == admission, "the CLI prints the report's verdict verbatim"
    assert admission["native_rows"] == {"status": "admitted", "refusal": None}
    assert admission["fixed_resources"]["status"] == "refused"
    assert admission["fixed_resources"]["refusal"].startswith(FIXED_PREFIX), admission["fixed_resources"]
    assert admission["status"] == "native_rows_only"
    assert admission["refusal"] == admission["fixed_resources"]["refusal"]
    assert code == emitter.EXIT_NATIVE_ROWS_ONLY, printed
    # The loader got past the relation and the native rows: the only refusal is the fixed charge's.
    payload = json.loads(emitted.out.read_text())
    table = parse_measured_runtime_table(payload, expected_context=parse_runtime_context(payload["context"]),
                                         expected_cost_sha256=emitted.cost_sha256, now=NOW,
                                         source_path=str(emitted.out))
    admit_native_rows(table, relation_load(emitted.relation_fixture))
    (row,) = table.rows
    assert row.resources.prefill_ms == 2. and row.resources.decode_ms == 2.
    assert row.resources.peak_scratch_bytes == 128 and row.resources.serialized_bytes == 42
    assert row.resources.resident_bytes == 64 and row.resources.activation_bytes == 8
    assert row.binding.operator_route == "torch.mm"
    assert dict(row.binding.member_operator_identity_sha256) == {row.unit: emitted.panel["joint_operator_identity_sha256"]}
    assert payload["context"]["runtime_sha256"] == identity_sha256(json.loads((emitted.evidence.root / "relation.json").read_text()))
    assert payload["context"]["prompt_tokens"] == 1 and payload["context"]["gpu_identity"] == "synthetic-gpu"
    assert emission["rows"][0]["unknown"] == ["fixed_and_full_model_resources"]
    evidence = emission["fixed_resources"]["evidence"]
    assert evidence["peak_scratch_bytes"] == "recomputed fixed_scratch"
    assert all("without evidence" in note for field, note in evidence.items() if field != "peak_scratch_bytes")


def test_refuses_a_panel_frozen_against_other_cost_bytes(emitted):
    emitted.cost_path.write_bytes(pickle.dumps(dict(emitted.cost, provenance={"other": True})))
    with pytest.raises(RuntimePriceError, match="panel cost_sha256 is not the digest of the supplied cost payload"):
        emitter.main(_argv(emitted))
    assert not emitted.out.exists()


def test_refuses_a_legacy_scalar_cost_row(emitted):
    inputs, _, _ = emitted.cell
    legacy = {"costs": {inputs["unit"]: {inputs["format"]: {"output_mse": 1.0, "predicted_dloss": 1.0}}}}
    emitted.cost_path.write_bytes(pickle.dumps(legacy))
    with pytest.raises(RuntimePriceError, match="cost payload row is not joint AURA currency"):
        emitter.main(_argv(emitted))
    assert not emitted.out.exists()


def test_refuses_a_cost_payload_with_no_row_for_the_panel(emitted):
    emitted.cost_path.write_bytes(pickle.dumps({"costs": {}}))
    with pytest.raises(RuntimePriceError, match="cost payload has no row for"):
        emitter.main(_argv(emitted))


def test_refuses_without_a_runtime_provenance_relation(emitted):
    with pytest.raises(RuntimePriceError, match="runtime provenance relation is missing"):
        emitter.main(_argv(emitted, **{"--relation": emitted.root / "absent.json"}))
    assert not emitted.out.exists()


def test_refuses_a_relation_of_another_schema(emitted):
    other = emitted.root / "other.json"
    other.write_text(json.dumps({"schema": "prismaquant.something_else.v1"}))
    with pytest.raises(RuntimePriceError, match="runtime provenance relation schema is not"):
        emitter.main(_argv(emitted, **{"--relation": other}))


@pytest.mark.parametrize("mutation", ["tolerance", "scratch_unproved", "status"])
def test_refuses_a_receipt_the_producer_consumer_refuses(emitted, mutation):
    receipt = json.loads(open(emitted.binding["receipt"]).read())
    if mutation == "tolerance":
        receipt["phases"]["decode"]["numerics"]["atol"] *= 2
    elif mutation == "scratch_unproved":
        receipt["resources"]["phases"]["decode"]["bound"]["external_native_peak_bytes"] = None
    else:
        receipt["status"] = "resources_observed"
    open(emitted.binding["receipt"], "w").write(json.dumps(receipt, sort_keys=True))
    with pytest.raises(RuntimePriceError, match="native producer admission refused"):
        emitter.main(_argv(emitted))
    assert not emitted.out.exists()


def test_refuses_an_incomplete_resource_ledger(emitted):
    receipt = json.loads(open(emitted.binding["receipt"]).read())
    receipt["resources"]["status"] = "incomplete"
    receipt["resources"]["phases"] = {phase: {} for phase in ("prefill", "decode")}
    open(emitted.binding["receipt"], "w").write(json.dumps(receipt, sort_keys=True))
    with pytest.raises(RuntimePriceError, match="native row has an incomplete resource ledger"):
        emitter.main(_argv(emitted))


def test_refuses_a_missing_artifact_by_name(emitted):
    binding = dict(emitted.binding, memory_trace=str(emitted.root / "gone.json"))
    emitted.manifest.write_text(json.dumps([binding]))
    with pytest.raises(RuntimePriceError, match="native memory trace for fixture.dense@TESSERA_BF16_K1_R1792 is missing"):
        emitter.main(_argv(emitted))


def test_refuses_receipts_from_more_than_one_runtime(emitted, joined):
    inputs, preflight, joint = _cell(joined, unit="fixture.dense2")
    inputs["runtime_image"] = emitted.raw["image"]
    inputs["wire"]["record"]["identity"] = dict(emitted.cell[0]["wire"]["record"]["identity"])
    preflight["operator"]["wire_record_sha256"] = identity_sha256(inputs["wire"]["record"])
    other_raw = copy.deepcopy(emitted.raw)
    other_raw["gpu"] = dict(other_raw["gpu"], uuid="another-gpu")
    preflight["runtime"], preflight["runtime_sha256"] = other_raw, identity_sha256(other_raw)
    emitted.cost["costs"][inputs["unit"]] = {inputs["format"]: joint}
    emitted.cost_path.write_bytes(pickle.dumps(emitted.cost))
    cost_sha256 = hashlib.sha256(emitted.cost_path.read_bytes()).hexdigest()
    first, _ = _write_cell(emitted.root, "cell", emitted.cell, emitted.raw, cost_sha256)
    second, _ = _write_cell(emitted.root, "cell2", (inputs, preflight, joint), other_raw, cost_sha256)
    emitted.manifest.write_text(json.dumps([first, second]))
    with pytest.raises(RuntimePriceError, match="native receipts were produced on more than one runtime"):
        emitter.main(_argv(emitted))


def test_refuses_a_duplicate_binding(emitted):
    emitted.manifest.write_text(json.dumps([emitted.binding, emitted.binding]))
    with pytest.raises(RuntimePriceError, match="duplicate native receipt for"):
        emitter.main(_argv(emitted))


def test_refuses_an_empty_manifest(emitted):
    emitted.manifest.write_text("[]")
    with pytest.raises(RuntimePriceError, match="at least one native receipt binding"):
        emitter.main(_argv(emitted))


def test_the_emitter_never_reports_an_admission_it_does_not_have(emitted, capsys):
    """A null refusal and exit 0 mean both gates passed, and nothing else does.

    `admit_runtime_provenance` raises on the native-row gate and *returns* the
    fixed-resource refusal. A caller that reads only the exception therefore
    sees a table it was never told about: this one, whose fixed charge no v2
    table can have admitted while D37 stands.
    """
    code = emitter.main(_argv(emitted))
    admission = json.loads(capsys.readouterr().out)["admission"]
    assert admission["status"] != "admitted", "no v2 table's fixed charge is admitted while D37 stands"
    assert admission["refusal"], "and the report says why, verbatim from the gate"
    assert (admission["refusal"] is None) == (admission["status"] == "admitted")
    assert (code == emitter.EXIT_ADMITTED) == (admission["status"] == "admitted")


def test_a_table_the_loader_refuses_outright_prices_nothing(emitted, capsys):
    """Exit 2 is the other answer: no row is admitted, so the fixed gate is unreached."""
    code = emitter.main(_argv(emitted, **{"--valid-hours": 0}))
    admission = json.loads(capsys.readouterr().out)["admission"]
    assert admission["status"] == "refused"
    assert admission["native_rows"]["status"] == "refused"
    assert "measurement window" in admission["refusal"]
    assert admission["fixed_resources"] == {"status": "unreached", "refusal": None}
    assert code == emitter.EXIT_REFUSED
