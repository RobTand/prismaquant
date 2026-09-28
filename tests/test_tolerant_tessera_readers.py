"""Tessera record readers accept additive fields and refuse only on what they read (#1548).

Two reader breakages set the shape of these tests. #926: contract v33 moved the
activation-quantizer table to a per-image list, and a reader that refused any
member it did not know turned every later contract into a pin bump. #958:
contract v34 minted cells the lane reader over-read. The report consumer broke
the same way on the full-engine report (#927): a producer observation it had
never heard of refused the whole report.

The rule these tests pin is one rule, applied through one helper
(:func:`prismaquant.record_fields.admit_fields`) by all six readers:

* a field or block the reader does not know is accepted, and every value the
  reader does use is unchanged;
* a consumed field that is missing, or carries the wrong type, still refuses;
* a producer that adds a field an old reader must not skip names it in the
  object's ``must_understand`` list, and a reader that does not know that name
  refuses the record.

Every fixture is a real published record (the installed Tessera contract, the
quoted v34 activation table, the synthetic full-engine report and the native
panel the admission suites already use), mutated in exactly the way the named
breakage mutated it.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

from prismaquant import lane_eligibility as lane
from prismaquant import native_receipt_table as emitter
from prismaquant import tessera_runtime_contract as trc
from prismaquant.measured_runtime_prices import RuntimePriceError

from conftest import _installed_contract
from test_full_engine_resource_report import _v2_with_admission, consume, supplied
from test_native_operator_panel import joined  # noqa: F401  (fixture for `emitted`)
from test_native_receipt_table import emitted  # noqa: F401  (fixture)
from test_runtime_provenance import relation_fixture  # noqa: F401  (fixture for `emitted`)

MU = "must_understand"
PLATFORM = "sm_121"
V34 = Path(__file__).parent / "fixtures" / "tessera_activation_quantizers_v34.json"


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------
def test_the_helper_accepts_an_additive_field_and_returns_the_record():
    from prismaquant.record_fields import admit_fields

    record = {"a": 1, "b": 2, "added": {"x": 1}}
    assert admit_fields(record, "r", required=("a",), optional=("b",),
                        error=ValueError) is record


@pytest.mark.parametrize("record,message", [
    ({"b": 2}, "missing field"),
    ([1, 2], "JSON object"),
    ({"a": 1, "new": 1, MU: ["new"]}, "must-understand"),
    ({"a": 1, MU: "new"}, MU),
    ({"a": 1, MU: ["b", "b"]}, MU),
    ({"a": 1, MU: [""]}, MU),
])
def test_the_helper_refuses_on_what_the_reader_needs(record, message):
    from prismaquant.record_fields import admit_fields

    class Refused(Exception):
        pass

    with pytest.raises(Refused, match=message):
        admit_fields(record, "r", required=("a",), optional=("b",), error=Refused)


def test_a_must_understand_field_the_reader_knows_is_read():
    from prismaquant.record_fields import admit_fields

    admit_fields({"a": 1, "b": 2, MU: ["a", "b"]}, "r", required=("a",),
                 optional=("b",), error=ValueError)


# ---------------------------------------------------------------------------
# tessera_runtime_contract: the activation-quantizer table (#926's block)
# ---------------------------------------------------------------------------
def _v34():
    return copy.deepcopy(json.loads(V34.read_text()))


def _aq(payload):
    return trc._parse_activation_quantizers(payload, "<v34>")


def _entries(payload):
    return payload["activation_quantizers"]["platforms"][PLATFORM]


def _first_contract(entry):
    return next(iter(entry["contracts"].values()))


def _add_members(payload):
    """What a producer adds beside the v2 members without changing them."""
    payload["activation_quantizers"]["generator_revision"] = "abc"
    for entry in _entries(payload):
        entry["receipt"] = {"path": "docs/measurements/x.md", "sha256": "0" * 64}
        entry["generated"]["cuda"] = "13.0"
        row = _first_contract(entry)
        row["detail"] = "the kernel rounds to nearest even"
        row["vectors"][0]["note"] = "midpoint"
    return payload


def test_an_activation_table_with_added_members_reads_to_the_same_values():
    assert _aq(_add_members(_v34())) == _aq(_v34())


def test_an_activation_table_missing_a_consumed_member_is_refused():
    payload = _v34()
    _entries(payload)[0]["generated"].pop("driver")
    with pytest.raises(trc.TesseraContractError, match="driver"):
        _aq(payload)


def test_an_activation_table_with_a_retyped_consumed_member_is_refused():
    payload = _v34()
    _first_contract(_entries(payload)[0])["unit_length"] = "16"
    with pytest.raises(trc.TesseraContractError, match="positive integer"):
        _aq(payload)
    payload = _v34()
    _entries(payload)[0]["generated"]["vllm"] = 28
    with pytest.raises(trc.TesseraContractError, match="non-empty string"):
        _aq(payload)


@pytest.mark.parametrize("where", ["entry", "generated", "contract", "vector"])
def test_an_activation_member_marked_must_understand_is_refused(where):
    payload = _add_members(_v34())
    entry = _entries(payload)[0]
    target, name = {
        "entry": (entry, "receipt"),
        "generated": (entry["generated"], "cuda"),
        "contract": (_first_contract(entry), "detail"),
        "vector": (_first_contract(entry)["vectors"][0], "note"),
    }[where]
    target[MU] = [name]
    with pytest.raises(trc.TesseraContractError, match="must-understand"):
        _aq(payload)


def test_a_must_understand_list_naming_only_read_members_is_accepted():
    payload = _v34()
    _entries(payload)[0][MU] = ["contracts", "generated"]
    assert _aq(payload) == _aq(_v34())


# ---------------------------------------------------------------------------
# The whole contract: tessera_runtime_contract + lane_eligibility (#958's cells)
# ---------------------------------------------------------------------------
def _parse(payload):
    return trc._parse(payload, commit="t", sha="t", path="<contract>")


def _launching_cell(payload):
    for cell in payload["lane_eligibility"]["cells"]:
        if cell.get("executes"):
            return cell
    raise AssertionError("the installed contract publishes no cell with a launch")


def _additive_contract():
    """#958's shape: new cells and launches carrying words the reader never read."""
    payload = _installed_contract()
    payload["release_notes"] = {"v41": "prose"}
    cell = _launching_cell(payload)
    cell["executes"][0]["lane"] = None
    cell["census_receipt"] = "docs/measurements/census.md"
    cell["runtime"]["cuda"] = "13.0"
    cell["evidence"]["reviewer"] = "tessera"
    cell["evidence"]["smoke"]["observed_by"] = "census"
    payload["lane_eligibility"]["generated_by"] = "experiments/mint.py"
    for platform in payload["lane_eligibility"]["platforms"].values():
        if isinstance(platform, dict):
            platform["sm_count"] = 48
    row = payload["native_extensions"][0]
    row["build_flags"] = ["-O3"]
    row["lane"]["kernel"] = "gemv"
    next(iter(row["when_unavailable"].values()))["warning"] = "slow"
    payload["formats"][0]["encoder_note"] = "prose"
    payload["fused_module"]["writer"] = "none"
    unit = payload["tensor_parallel"]["units"][0]
    unit["measured_on"] = "gb10"
    payload["tensor_parallel"]["world_size_receipts"][0]["duration_s"] = 12.5
    return payload


def test_a_contract_with_additive_fields_answers_exactly_as_before():
    assert trc.contract_answer(_parse(_additive_contract())) == trc.contract_answer(
        _parse(_installed_contract()))


def test_a_launch_that_drops_a_consumed_field_is_still_refused():
    payload = _installed_contract()
    del _launching_cell(payload)["executes"][0]["decoder"]
    with pytest.raises(trc.TesseraContractError, match="decoder"):
        _parse(payload)


def test_a_launch_whose_symbol_is_retyped_is_still_refused():
    payload = _installed_contract()
    _launching_cell(payload)["executes"][0]["symbol"] = 7
    with pytest.raises(trc.TesseraContractError, match="non-empty strings"):
        _parse(payload)


def _mark(payload, where):
    """Mark one added field must-understand, on the object that carries it."""
    cell = _launching_cell(payload)
    row = payload["native_extensions"][0]
    target, name = {
        "contract": (payload, "release_notes"),
        "lane_table": (payload["lane_eligibility"], "generated_by"),
        "cell": (cell, "census_receipt"),
        "launch": (cell["executes"][0], "lane"),
        "runtime": (cell["runtime"], "cuda"),
        "evidence": (cell["evidence"], "reviewer"),
        "smoke": (cell["evidence"]["smoke"], "observed_by"),
        "extension": (row, "build_flags"),
        "extension_lane": (row["lane"], "kernel"),
        "format": (payload["formats"][0], "encoder_note"),
        "fused_module": (payload["fused_module"], "writer"),
        "tp_unit": (payload["tensor_parallel"]["units"][0], "measured_on"),
        "tp_receipt": (payload["tensor_parallel"]["world_size_receipts"][0], "duration_s"),
    }[where]
    target[MU] = [name]
    return payload


@pytest.mark.parametrize("where", [
    "contract", "lane_table", "cell", "launch", "runtime", "evidence", "smoke",
    "extension", "extension_lane", "format", "fused_module", "tp_unit", "tp_receipt",
])
def test_a_contract_field_marked_must_understand_is_refused(where):
    with pytest.raises(trc.TesseraContractError, match="must-understand"):
        _parse(_mark(_additive_contract(), where))


def test_an_unknown_lane_predicate_is_still_refused_without_a_mark():
    """``requires`` is a predicate: every key is a condition, so none is skippable."""
    payload = _installed_contract()
    payload["native_extensions"][0]["lane"]["requires"]["span"] = [2]
    with pytest.raises(trc.TesseraContractError, match="cannot decide"):
        _parse(payload)


# ---------------------------------------------------------------------------
# full_engine_resource_report (#927's observations and derived blocks)
# ---------------------------------------------------------------------------
def _v2():
    return _v2_with_admission(supplied())


def _additive_report():
    report = _v2()
    report["producer_note"] = "prose"
    report["identity"]["producer"] = {"commit": "abc"}
    report["identity"]["run"]["host_hint"] = "gb10"
    report["execution"]["cuda_graphs"] = "off"
    report["reference"]["catalog"] = "none"
    report["workload"]["seed"] = 7
    report["observations"]["next_observation"] = {"schema": "tessera.next.v1"}
    report["observations"]["torch_allocations"][0]["stream"] = 7
    report["derived"]["next_verdict"] = {}
    report["partition"]["scope"]["observer_note"] = "prose"
    report["partition"]["membership"][0]["site"] = {"frame": "f"}
    return report


def test_a_report_with_additive_fields_recomputes_the_same_verdict(tmp_path):
    assert _v2()["partition"]["membership"], "the fixture must carry a membership row"
    assert consume(tmp_path, _additive_report()) == consume(tmp_path, _v2())


def test_a_report_missing_a_consumed_field_is_still_refused(tmp_path):
    report = _v2()
    del report["partition"]["membership"][0]["bytes"]
    with pytest.raises(RuntimePriceError, match="bytes"):
        consume(tmp_path, report)


def test_a_report_with_a_retyped_consumed_field_is_still_refused(tmp_path):
    report = _v2()
    report["partition"]["membership"][0]["bytes"] = "12"
    with pytest.raises(RuntimePriceError, match="expected integer"):
        consume(tmp_path, report)


def test_a_rank_without_its_world_is_still_refused(tmp_path):
    report = _v2()
    report["identity"]["run"]["rank"] = 1
    with pytest.raises(RuntimePriceError, match="world_size"):
        consume(tmp_path, report)


@pytest.mark.parametrize("where", [
    "report", "observations", "derived", "membership", "allocation", "run",
])
def test_a_report_field_marked_must_understand_is_refused(tmp_path, where):
    report = _additive_report()
    target, name = {
        "report": (report, "producer_note"),
        "observations": (report["observations"], "next_observation"),
        "derived": (report["derived"], "next_verdict"),
        "membership": (report["partition"]["membership"][0], "site"),
        "allocation": (report["observations"]["torch_allocations"][0], "stream"),
        "run": (report["identity"]["run"], "host_hint"),
    }[where]
    target[MU] = [name]
    with pytest.raises(RuntimePriceError, match="must-understand"):
        consume(tmp_path, report)


def test_a_keyed_domain_table_stays_exact(tmp_path):
    """Domain names are values the consumer recomputes, not fields it may skip."""
    report = _v2()
    report["partition"]["domains"]["next_domain"] = copy.deepcopy(
        report["partition"]["domains"]["history_join"])
    with pytest.raises(RuntimePriceError, match="expected exactly fields"):
        consume(tmp_path, report)


# ---------------------------------------------------------------------------
# native_receipt_table: the native runtime record a panel carries
# ---------------------------------------------------------------------------
def _context(emitted, mutate=None):
    panel = copy.deepcopy(emitted.panel)
    if mutate is not None:
        mutate(panel["runtime"])
    return emitter.derive_context([panel], relation=emitted.relation)


def test_a_runtime_record_with_additive_fields_derives_the_same_context(emitted):
    def add(runtime):
        runtime["collector_build"] = "abc"
        runtime["execution"]["cudagraph_capture_sizes"] = [1]
        runtime["gpu"]["clock_mhz"] = 2418
    assert _context(emitted, add) == _context(emitted)


@pytest.mark.parametrize("drop", [("image",), ("execution", "mode"), ("gpu", "uuid")])
def test_a_runtime_record_missing_a_consumed_field_is_refused_by_name(emitted, drop):
    def remove(runtime):
        node = runtime
        for key in drop[:-1]:
            node = node[key]
        del node[drop[-1]]
    with pytest.raises(RuntimePriceError, match=drop[-1]):
        _context(emitted, remove)


@pytest.mark.parametrize("where", ["runtime", "execution", "gpu"])
def test_a_runtime_field_marked_must_understand_is_refused(emitted, where):
    def mark(runtime):
        target = runtime if where == "runtime" else runtime[where]
        target["clock_policy"] = "locked"
        target[MU] = ["clock_policy"]
    with pytest.raises(RuntimePriceError, match="must-understand"):
        _context(emitted, mark)


# ---------------------------------------------------------------------------
# One helper, six readers
# ---------------------------------------------------------------------------
def test_all_six_readers_admit_fields_through_the_one_helper(monkeypatch, tmp_path, emitted, joined):
    from prismaquant import record_fields

    callers = set()
    real = record_fields.admit_fields

    def spy(*args, **kwargs):
        callers.add(sys._getframe(1).f_globals["__name__"])
        return real(*args, **kwargs)

    monkeypatch.setattr(record_fields, "admit_fields", spy)
    _parse(_installed_contract())
    consume(tmp_path, _v2())
    _context(emitted)
    # The two native panel readers (#1565): the dense preflight through a
    # freeze, and the routed-MoE workspace block through its own reader.
    from prismaquant import native_moe_panel
    from prismaquant.native_operator_panel import freeze_native_panel
    freeze_native_panel(*joined, cost_sha256="4" * 64)
    native_moe_panel._workspace_identity({
        "schema": "tessera.native_moe_workspace.v1", "owner": "vllm.WorkspaceManager",
        "num_ubatches": 1, "num_lanes": 1, "locked": True, "resident_bytes": 0,
        "slots": [{"index": 0, "allocation": None}]})
    assert {"prismaquant.lane_eligibility", "prismaquant.tessera_runtime_contract",
            "prismaquant.full_engine_resource_report",
            "prismaquant.native_receipt_table", "prismaquant.native_operator_panel",
            "prismaquant.native_moe_panel"} <= callers, sorted(callers)
