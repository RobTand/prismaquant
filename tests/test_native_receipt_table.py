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
from prismaquant.native_operator_panel import freeze_native_panel, operator_route_identity
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


def _write_cell(root, name, cell, raw, cost_sha256, run_id="native"):
    inputs, preflight, joint = cell
    panel, receipt, trace = receipt_fixture((inputs, preflight, joint), complete=True, cost_sha256=cost_sha256)
    trace["capture"]["collector_library_sha256"] = raw["resource_collector"]["library_sha256"]
    receipt["resources"]["trace_sha256"] = identity_sha256(trace)
    paths = {}
    for kind, value in (("panel", panel), ("receipt", receipt), ("memory_trace", trace)):
        path = root / f"{name}.{kind}.json"
        path.write_text(json.dumps(value, sort_keys=True))
        paths[kind] = str(path)
    return {"unit": inputs["unit"], "format": inputs["format"], "run_id": run_id, **paths}, panel


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
    assert row.binding.operator_route == operator_route_identity(
        emitted.panel["phases"]["prefill"]["expected_route"])
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


def _pair(emitted, joined, other_raw, run_id="native"):
    """Write a second cell, on a second unit, measured under ``other_raw``.

    The caller decides what ``other_raw`` is: a comparable runtime that loaded
    one more kernel, or a different box. Both cells are rewritten because both
    panels freeze against the cost bytes, which gain the second unit's row.
    """
    inputs, preflight, joint = _cell(joined, unit="fixture.dense2")
    inputs["runtime_image"] = other_raw["image"]
    inputs["wire"]["record"]["identity"] = dict(emitted.cell[0]["wire"]["record"]["identity"])
    preflight["operator"]["wire_record_sha256"] = identity_sha256(inputs["wire"]["record"])
    preflight["runtime"], preflight["runtime_sha256"] = other_raw, identity_sha256(other_raw)
    emitted.cost["costs"][inputs["unit"]] = {inputs["format"]: joint}
    emitted.cost_path.write_bytes(pickle.dumps(emitted.cost))
    cost_sha256 = hashlib.sha256(emitted.cost_path.read_bytes()).hexdigest()
    first, _ = _write_cell(emitted.root, "cell", emitted.cell, emitted.raw, cost_sha256)
    second, _ = _write_cell(emitted.root, "cell2", (inputs, preflight, joint), other_raw, cost_sha256, run_id=run_id)
    emitted.manifest.write_text(json.dumps([first, second]))
    return cost_sha256


#: Every mutation is applied to the *second* panel's runtime record, which is
#: otherwise a byte-for-byte copy of the first: the driver is mutated, never the
#: fixture, so each case proves this particular check bites and not some other.
#: ``tf32_zero`` is the one that is not about a box at all -- it replaces
#: ``false`` with ``0``, which ``!=`` considers equal in Python. The whole-record
#: SHA-256 this check replaced told them apart, and so must its partition.
RUNTIME_MUTATIONS = {
    "gpu_uuid": lambda raw: raw.update(gpu=dict(raw["gpu"], uuid="another-gpu")),
    "gpu_capability": lambda raw: raw.update(gpu=dict(raw["gpu"], capability=[10, 0])),
    "torch_build": lambda raw: raw.update(versions=dict(raw["versions"], torch="synthetic-other")),
    "arithmetic": lambda raw: raw.update(arithmetic=dict(raw["arithmetic"], tf32=True)),
    "tf32_zero": lambda raw: raw.update(arithmetic=dict(raw["arithmetic"], tf32=0)),
    "image": lambda raw: raw.update(image=raw["image"][:-1] + ("0" if raw["image"][-1] != "0" else "1")),
    "package_source": lambda raw: raw.update(source=dict(raw["source"], tessera_package_sha256="9" * 64)),
    "collector": lambda raw: raw.update(resource_collector=dict(raw["resource_collector"], library_sha256="9" * 64)),
    "shared_library_bytes": lambda raw: raw["native_libraries"].update({"/usr/lib/libtorch.so": "9" * 64}),
    "unknown_field": lambda raw: raw.update(unclassified_producer_field="present"),
    "missing_field": lambda raw: raw.pop("arithmetic"),
}


@pytest.mark.parametrize("mutation", sorted(RUNTIME_MUTATIONS))
def test_refuses_receipts_from_more_than_one_runtime(emitted, joined, mutation):
    other_raw = copy.deepcopy(emitted.raw)
    RUNTIME_MUTATIONS[mutation](other_raw)
    # Not `!=`: on ``tf32_zero`` Python considers the two records equal, which
    # is the whole reason that case exists. The digest is what must differ.
    assert identity_sha256(other_raw) != identity_sha256(emitted.raw)
    _pair(emitted, joined, other_raw)
    with pytest.raises(RuntimePriceError, match="native receipts were produced on more than one runtime"):
        emitter.main(_argv(emitted))


def test_a_route_that_loads_its_own_kernel_extension_still_needs_its_own_run(emitted, joined):
    """The relaxation moves the refusal onto the binding; it does not remove one.

    A second panel that loaded one more library is now comparable, so the table
    is emitted -- and ``admit_native_rows`` then refuses it, because the row
    names a run whose attested runtime record is not the one it was measured
    under. The loosened check was the one that named nothing.
    """
    other_raw = copy.deepcopy(emitted.raw)
    other_raw["native_libraries"]["/out/cache/extensions/route-kernel.so"] = "7" * 64
    _pair(emitted, joined, other_raw)
    emitter.main(_argv(emitted))
    emission = json.loads((emitted.root / "table.emission.json").read_text())
    assert "original native panel runtime" in emission["admission"]["refusal"]


def test_refuses_a_duplicate_binding(emitted):
    emitted.manifest.write_text(json.dumps([emitted.binding, emitted.binding]))
    with pytest.raises(RuntimePriceError, match="duplicate native receipt for"):
        emitter.main(_argv(emitted))


def test_refuses_an_empty_manifest(emitted):
    emitted.manifest.write_text("[]")
    with pytest.raises(RuntimePriceError, match="at least one native receipt binding"):
        emitter.main(_argv(emitted))


def _second_native_run(emitted, extension, digest):
    """Clone the relation's native run into one that loaded a second kernel.

    The full-engine run is given the same bytes at the same path and the
    dependency relation names them, because that is what the relation already
    demands of every production library a native run loads. This is exactly the
    relation shape the fp4 receipts do **not** have: their full-engine run
    served an fp8 artifact and never loaded the fp4 extension (PQ #570).
    """
    evidence, relation, _ = emitted.relation_fixture
    raw = copy.deepcopy(emitted.raw)
    raw["native_libraries"][extension] = digest
    run = copy.deepcopy(relation["runs"]["native"])
    run["runtime"] = evidence.put("native2-runtime.json", raw)
    relation["runs"]["native2"] = run
    engine = evidence.get(relation["runs"]["engine"]["runtime"])
    engine["base"]["native_libraries"][extension] = digest
    evidence.replace(relation["runs"]["engine"]["runtime"], engine)
    relation["production_dependencies"] += [
        {"native_run_id": "native2", "native_path": path, "full_engine_path": path, "sha256": sha}
        for path, sha in (("/usr/lib/libtorch.so", "5" * 64), (extension, digest))]
    return raw


def test_two_routes_that_load_different_kernels_admit_into_one_table(emitted, joined):
    """The mixed-route table #559 could not emit at all, emitted and admitted.

    Two rows, one box, two loaded-library sets: the second row's route loaded a
    kernel extension the first row's did not. Nothing about comparability is
    waived -- every other runtime field is equal, the shared libraries carry the
    same bytes, and each row still binds to the run it was measured on.
    """
    extension, digest = "/out/cache/extensions/route-kernel.so", "7" * 64
    other_raw = _second_native_run(emitted, extension, digest)
    cost_sha256 = _pair(emitted, joined, other_raw, run_id="native2")
    relation_load(emitted.relation_fixture)      # rewrite relation.json with both native runs
    emitter.main(_argv(emitted))
    payload = json.loads(emitted.out.read_text())
    table = parse_measured_runtime_table(payload, expected_context=parse_runtime_context(payload["context"]),
                                         expected_cost_sha256=cost_sha256, now=NOW, source_path=str(emitted.out))
    admit_native_rows(table, relation_load(emitted.relation_fixture))
    assert [row.unit for row in table.rows] == ["fixture.dense", "fixture.dense2"]
    # Past the relation and past the native rows: the refusal left is the fixed
    # charge's (the owed observations no v1 fixture carries), never a row's.
    emission = json.loads((emitted.root / "table.emission.json").read_text())
    assert emission["admission"]["refusal"].startswith(FIXED_PREFIX), emission["admission"]["refusal"]
    # The kernel that varies is named per row, read off each panel's own record.
    assert {row["unit"]: row["unshared_native_libraries"] for row in emission["rows"]} == {
        "fixture.dense": {}, "fixture.dense2": {extension: digest}}
    assert len({row["runtime_sha256"] for row in emission["rows"]}) == 2


def test_a_second_native_run_on_another_box_is_still_refused(emitted, joined):
    """The relation has two native runs and they are still not comparable.

    Same construction as the admitted case, one field changed: the second run
    reports another GPU. A per-row binding to a real run does not make two
    boxes one clock, and the emitter must refuse before a table exists.
    """
    extension, digest = "/out/cache/extensions/route-kernel.so", "7" * 64
    other_raw = _second_native_run(emitted, extension, digest)
    other_raw["gpu"] = dict(other_raw["gpu"], uuid="another-gpu")
    emitted.evidence.replace(emitted.relation_fixture[1]["runs"]["native2"]["runtime"], other_raw)
    _pair(emitted, joined, other_raw, run_id="native2")
    with pytest.raises(RuntimePriceError, match="native receipts were produced on more than one runtime: gpu differs"):
        emitter.main(_argv(emitted))
    assert not emitted.out.exists()
def test_the_emitter_never_reports_an_admission_it_does_not_have(emitted, capsys):
    """A null refusal and exit 0 mean both gates passed, and nothing else does.

    `admit_runtime_provenance` raises on the native-row gate and *returns* the
    fixed-resource refusal. A caller that reads only the exception therefore
    sees a table it was never told about: this one, whose fixed charge no v1
    report can have admitted while the four observations v1 owes stay null
    (the transient charge boundary is versioned now and is not what blocks it).
    """
    code = emitter.main(_argv(emitted))
    admission = json.loads(capsys.readouterr().out)["admission"]
    assert admission["status"] != "admitted", "no v1 report's fixed charge is admitted while its owed observations are null"
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


def _emit_with_route(state, *, policy, contract, symbol="torch._scaled_mm"):
    """Re-emit the one cell under a declared route, and read back its binding."""
    inputs, preflight, joint = state.cell
    preflight["operator"]["declared_route"] = {"kind": "dense", "policy": policy, "symbol": symbol,
                                               "decoder": "fixture-window", "contract": contract}
    preflight["operator"]["activation_contract"] = contract
    _write_cell(state.root, "cell", (inputs, preflight, joint), state.raw, state.cost_sha256)
    emitter.main(_argv(state))
    return json.loads(state.out.read_text())["rows"][0]["binding"]["operator_route"]


def test_two_route_classes_that_share_one_gemm_symbol_are_two_bindings(emitted):
    """TESSERA_FP8 and TESSERA_NVFP4 both execute `torch._scaled_mm`.

    They do it on differently packed operands under different activation
    contracts, so a binding carrying the GEMM symbol alone makes a downstream
    consumer unable to tell the two apart in the one field it compares.
    """
    fp8 = _emit_with_route(emitted, policy="TESSERA_FP8:resident", contract="fp8_per_token_dynamic")
    fp4 = _emit_with_route(emitted, policy="TESSERA_NVFP4:resident", contract="nvfp4_per_block_static")
    assert fp8 != fp4, "the binding cannot tell fp8 from fp4"
    assert "fp8_per_token_dynamic" in fp8 and "TESSERA_FP8:resident" in fp8
    assert "nvfp4_per_block_static" in fp4 and "TESSERA_NVFP4:resident" in fp4
