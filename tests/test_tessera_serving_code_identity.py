"""PrismaQuant reads Tessera's serving code identity (#1561, the #1549 follow-up).

Tessera #678 (contract v41) lets a lane cell's ``runtime`` block name the
Tessera code its evidence was taken on -- ``tessera_commit`` for a person and
``serving_source_sha256`` for a program -- and stamps the same digest into the
route-trace header of every serve. A cell is an attestation about the code
that ran, so under principle 14 its scope has to be compared with the code
that serves, not with a commit somebody typed.

The PrismaQuant side is dormant under the tracked v2 pin, which carries no
serving digest. Under a v3 pin:

- the one cell matcher every admission path shares requires the cell's digest
  to equal the pin's, and reads a cell that names no code as a mismatch;
- a unit whose only cells name other code is unattested, and the regime route
  names the cell and both digests;
- the route-trace gate requires every rank's header digest to equal the pin's,
  refuses a different one naming both, and reads an absent one as NOT VERIFIED.

Fail-before on ``a796448b5e8`` (origin/main): the matcher, the trace gate and
the unit resolver take no digest, the pin parser knows no v3 schema, and a
cell that names its code is refused by the runtime-scope parser, so none of
the tests below except the v2 snapshot can pass there.
"""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_route_trace_gate as gate
from prismaquant import tessera_serving_runtime_pin as pin_module
from prismaquant.lane_spec import load_lane_spec

from tests import tessera_serving_identity_snapshot as snapshot_module

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "fixtures" / "tessera_serving_identity_v2_snapshot.json"

#: The digest a v3 pin names in these tests, and one no pinned tree hashes to.
PINNED = "1f" * 32
OTHER = "e0" * 32
SERVING_COMMIT = "c" * 40


# ---------------------------------------------------------------------------
# Fixtures: the packaged contract, with its cells naming their code
# ---------------------------------------------------------------------------
def _packaged_contract() -> dict:
    from importlib.resources import as_file

    from prismaquant.tessera_render import tessera_serving_contract_path

    with as_file(tessera_serving_contract_path()) as path:
        return json.loads(path.read_text(encoding="utf-8"))


def _stamped_table(tmp_path, digest=PINNED, *, only=None, fields=None):
    """The packaged table with every cell's runtime naming ``digest``.

    ``only`` limits the stamp to one cell id; ``fields`` replaces the two
    stamped members outright, for the malformed cases.
    """
    contract = _packaged_contract()
    for cell in contract["lane_eligibility"]["cells"]:
        if only is not None and cell["id"] != only:
            continue
        if fields is None:
            cell["runtime"]["tessera_commit"] = SERVING_COMMIT
            cell["runtime"]["serving_source_sha256"] = digest
        else:
            cell["runtime"].update(fields)
    path = tmp_path / "runtime_contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    return lane.load_eligibility_table("0.1.0", contract_path=path)


def _context(cell, residency=None, mode=None):
    return lane.ServingContext(
        platform=cell.platform, structure=cell.structure,
        residency=residency or cell.residency_modes[0],
        runtime_image=cell.runtime_image,
        execution_mode=mode or cell.execution_modes[0])


def _facts(cell):
    rung = cell.rungs_q256[0]
    return lane.UnitStructuralFacts(
        qname="model.layers.0.mlp.down_proj", format_name=f"{cell.family}@{rung}",
        payload_family=cell.family, k=None, n_sub=None, structure=cell.structure,
        role_split=False, in_features=4096, out_features=4096, rate_q256=rung)


def _route(table, cell, **kwargs):
    return lane.resolve_unit_route(
        _facts(cell), table, platform=cell.platform,
        residency=cell.residency_modes[0], runtime_image=cell.runtime_image,
        execution_mode=cell.execution_modes[0], **kwargs)


# ---------------------------------------------------------------------------
# The cell grammar: both fields or neither, exact shapes
# ---------------------------------------------------------------------------
def test_a_cell_that_names_its_code_parses_and_keeps_both_fields(tmp_path):
    table = _stamped_table(tmp_path)
    assert table.cells
    for cell in table.cells:
        assert cell.runtime_tessera_commit == SERVING_COMMIT
        assert cell.runtime_serving_source_sha256 == PINNED
        runtime = cell.as_dict()["runtime"]
        assert runtime["tessera_commit"] == SERVING_COMMIT
        assert runtime["serving_source_sha256"] == PINNED


def test_a_cell_that_names_no_code_serializes_as_it_always_did(tmp_path):
    contract = _packaged_contract()
    path = tmp_path / "runtime_contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    for cell in lane.load_eligibility_table("0.1.0", contract_path=path).cells:
        assert cell.runtime_serving_source_sha256 == ""
        assert set(cell.as_dict()["runtime"]) == {"image", "execution_modes"}


@pytest.mark.parametrize("fields", [
    {"tessera_commit": SERVING_COMMIT},
    {"serving_source_sha256": PINNED},
    {"tessera_commit": SERVING_COMMIT, "serving_source_sha256": "1F" * 32},
    {"tessera_commit": SERVING_COMMIT, "serving_source_sha256": PINNED[:-2]},
    {"tessera_commit": "c" * 39, "serving_source_sha256": PINNED},
    {"tessera_commit": SERVING_COMMIT, "serving_source_sha256": None},
])
def test_a_half_or_malformed_code_scope_is_refused(tmp_path, fields):
    with pytest.raises(lane.LaneEligibilityError, match="runtime"):
        _stamped_table(tmp_path, fields=fields)


# ---------------------------------------------------------------------------
# The matcher every admission path shares
# ---------------------------------------------------------------------------
def test_a_matching_digest_is_admitted(tmp_path):
    table = _stamped_table(tmp_path)
    for cell in table.cells:
        assert lane.cell_matches_serving_context(
            cell, _context(cell), serving_source_sha256=PINNED)
        assert lane.cell_serving_code_admits(cell, PINNED) == (True, "")


def test_a_different_digest_is_refused_naming_both(tmp_path):
    table = _stamped_table(tmp_path)
    for cell in table.cells:
        assert not lane.cell_matches_serving_context(
            cell, _context(cell), serving_source_sha256=OTHER)
        admits, why = lane.cell_serving_code_admits(cell, OTHER)
        assert admits is False
        assert PINNED in why and OTHER in why, why


def test_a_cell_that_names_no_code_is_not_admitted_by_a_v3_pin(tmp_path):
    table = lane.load_eligibility_table(
        "0.1.0", contract_path=_write(tmp_path, _packaged_contract()))
    for cell in table.cells:
        assert not lane.cell_matches_serving_context(
            cell, _context(cell), serving_source_sha256=PINNED)
        admits, why = lane.cell_serving_code_admits(cell, PINNED)
        assert admits is False
        assert "names no serving code" in why and PINNED in why, why


def test_a_v2_pin_skips_the_code_check(tmp_path):
    """``None`` is the v2 answer: the pin names no code, so nothing compares."""
    stamped = _stamped_table(tmp_path, OTHER)
    plain = lane.load_eligibility_table(
        "0.1.0", contract_path=_write(tmp_path / "plain", _packaged_contract()))
    for cell in stamped.cells + plain.cells:
        assert lane.cell_matches_serving_context(
            cell, _context(cell), serving_source_sha256=None)
        assert lane.cell_serving_code_admits(cell, None) == (True, "")


def test_the_default_reads_the_tracked_pin_so_no_caller_can_skip_it(
        tmp_path, monkeypatch):
    """A caller that passes nothing gets the tracked pin's digest, not a skip.

    Five admission paths call the matcher; a ``None`` default would let any of
    them skip the check under a v3 pin by omission. Standing a v3 digest in
    for the tracked pin flips every omitted-keyword call.
    """
    table = _stamped_table(tmp_path, OTHER)
    cell = table.cells[0]
    assert pin_module.pinned_serving_source_sha256() is None
    assert lane.cell_matches_serving_context(cell, _context(cell))
    monkeypatch.setattr(pin_module, "pinned_serving_source_sha256", lambda: PINNED)
    assert not lane.cell_matches_serving_context(cell, _context(cell))
    assert _route(table, cell).route_status == lane.ROUTE_STATUS_UNATTESTED


# ---------------------------------------------------------------------------
# A unit's route: the refusal names the cell and both digests
# ---------------------------------------------------------------------------
def test_a_unit_route_is_backed_when_the_digest_matches(tmp_path):
    table = _stamped_table(tmp_path)
    cell = table.cells[0]
    route = _route(table, cell, serving_source_sha256=PINNED)
    assert route.route_status in (lane.ROUTE_STATUS_BACKED,
                                  lane.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG)
    assert route.as_dict() == _route(table, cell, serving_source_sha256=None).as_dict()


def test_a_unit_route_whose_cells_name_other_code_is_unattested_and_says_why(tmp_path):
    table = _stamped_table(tmp_path)
    cell = table.cells[0]
    route = _route(table, cell, serving_source_sha256=OTHER)
    assert route.route_status == lane.ROUTE_STATUS_UNATTESTED
    assert route.regimes
    for regime in route.regimes:
        assert regime.route_status == lane.ROUTE_STATUS_UNATTESTED
        assert regime.cell_id, "the refusal must name the cell that named this unit"
        assert PINNED in regime.detail and OTHER in regime.detail, regime.detail


def test_a_v3_pin_admits_no_cell_that_names_no_code(tmp_path):
    """The packaged cells (v42) name no code, so a v3 pin attests none of them."""
    table = lane.load_eligibility_table(
        "0.1.0", contract_path=_write(tmp_path, _packaged_contract()))
    for cell in table.cells:
        route = _route(table, cell, serving_source_sha256=PINNED)
        assert route.route_status == lane.ROUTE_STATUS_UNATTESTED
        assert all("names no serving code" in r.detail for r in route.regimes)


# ---------------------------------------------------------------------------
# Pin file v3
# ---------------------------------------------------------------------------
def _v2_payload() -> dict:
    return json.loads(pin_module.tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))


def _v3_payload(**overrides) -> dict:
    payload = _v2_payload()
    commit = payload.pop("commit")
    payload["schema"] = pin_module.TESSERA_SERVING_RUNTIME_PIN_SCHEMA_V3
    payload["producer_commit"] = commit
    payload["serving_commit"] = SERVING_COMMIT
    payload["serving_source_sha256"] = PINNED
    payload.update(overrides)
    return payload


def _write(directory, payload) -> pathlib.Path:
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "runtime_contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_v2_pin_reads_as_one_commit_and_no_code_digest():
    pin = pin_module.load_tessera_serving_runtime_pin()
    assert pin.schema == "prismaquant.tessera_serving_runtime_pin.v2"
    assert pin.producer_commit == pin.serving_commit == pin.commit
    assert pin.serving_source_sha256 is None
    assert pin_module.pinned_serving_source_sha256() is None


def test_a_v3_pin_splits_the_producer_the_serve_and_the_code(tmp_path):
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(_v3_payload()), encoding="utf-8")
    pin = pin_module.load_tessera_serving_runtime_pin(path)
    assert pin.producer_commit == _v2_payload()["commit"]
    assert pin.serving_commit == pin.commit == SERVING_COMMIT
    assert pin.serving_source_sha256 == PINNED
    assert pin.contract_sha256 == _v2_payload()["contract_sha256"]


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda p: p.pop("serving_source_sha256"), id="no-digest"),
    pytest.param(lambda p: p.pop("producer_commit"), id="no-producer"),
    pytest.param(lambda p: p.update(commit=SERVING_COMMIT), id="v2-commit-member"),
    pytest.param(lambda p: p.update(serving_source_sha256="xyz"), id="digest-not-hex"),
    pytest.param(lambda p: p.update(serving_source_sha256=PINNED.upper()), id="digest-upper"),
    pytest.param(lambda p: p.update(serving_source_sha256=None), id="digest-null"),
    pytest.param(lambda p: p.update(serving_commit="c" * 39), id="short-serving-commit"),
    pytest.param(lambda p: p.update(
        producer_commit=pin_module.TESSERA_SERVING_RUNTIME_COMMIT_PENDING), id="pending-producer"),
    pytest.param(lambda p: p.update(
        contract_sha256=pin_module.TESSERA_SERVING_RUNTIME_CONTRACT_SHA256_PENDING),
        id="pending-contract"),
    pytest.param(lambda p: p.update(schema="prismaquant.tessera_serving_runtime_pin.v4"),
                 id="unknown-schema"),
])
def test_a_malformed_v3_pin_is_refused(mutate):
    payload = _v3_payload()
    mutate(payload)
    with pytest.raises(pin_module.TesseraServingRuntimePinError):
        pin_module.parse_tessera_serving_runtime_pin(payload, where="fixture")


def _v3_constants(monkeypatch, pin):
    for name, value in (
            ("TESSERA_SERVING_RUNTIME_PINNED_COMMIT", pin.serving_commit),
            ("TESSERA_SERVING_RUNTIME_PINNED_PRODUCER_COMMIT", pin.producer_commit),
            ("TESSERA_SERVING_RUNTIME_PINNED_SERVING_SOURCE_SHA256",
             pin.serving_source_sha256)):
        monkeypatch.setattr(pin_module, name, value)


def test_a_v3_pin_is_one_reviewed_change_with_its_constants(monkeypatch):
    pin = pin_module.parse_tessera_serving_runtime_pin(_v3_payload(), where="fixture")
    installed = pin.contract_sha256
    # The tracked constants still name the v2 pin: a v3 JSON edit alone
    # admits nothing.
    with pytest.raises(pin_module.TesseraServingRuntimePinError,
                       match="ONE reviewed change"):
        pin_module.require_exact_tessera_runtime_pin(
            pin, installed_contract_sha256=installed)
    _v3_constants(monkeypatch, pin)
    pin_module.require_exact_tessera_runtime_pin(pin, installed_contract_sha256=installed)
    for member, value in (("serving_source_sha256", OTHER),
                          ("producer_commit", "d" * 40),
                          ("serving_commit", "d" * 40)):
        moved = pin_module.parse_tessera_serving_runtime_pin(
            _v3_payload(**{member: value}), where="fixture")
        with pytest.raises(pin_module.TesseraServingRuntimePinError,
                           match="ONE reviewed change"):
            pin_module.require_exact_tessera_runtime_pin(
                moved, installed_contract_sha256=installed)


def test_the_producer_commit_names_the_venv():
    """``producer_commit`` is the only pin field that names a venv."""
    from prismaquant.tessera_runtime_contract import TESSERA_DEV_PIN_COMMIT

    pin = pin_module.load_tessera_serving_runtime_pin()
    assert pin.producer_commit == TESSERA_DEV_PIN_COMMIT
    assert pin_module.TESSERA_SERVING_RUNTIME_PINNED_PRODUCER_COMMIT == TESSERA_DEV_PIN_COMMIT


def test_the_producer_repo_gate_names_the_producer_commit(tmp_path, monkeypatch):
    from prismaquant import tessera_export_lane as tel

    pin = pin_module.parse_tessera_serving_runtime_pin(_v3_payload(), where="fixture")
    monkeypatch.setattr(pin_module, "load_tessera_serving_runtime_pin", lambda *a: pin)
    tool = load_lane_spec("tessera").producer_tools[0]
    with pytest.raises(tel.TesseraExportLaneError) as excinfo:
        tel.require_producer_repo_is_pinned({tool.repo_env: str(tmp_path)})
    message = str(excinfo.value)
    assert pin.producer_commit in message
    assert SERVING_COMMIT not in message


# ---------------------------------------------------------------------------
# The route-trace gate
# ---------------------------------------------------------------------------
def _trace_contract():
    spec = load_lane_spec("tessera")
    executes = {platform: dict(entry) for platform, entry in
                spec.served_activation_quantization.executes_by_platform.items()}
    formats = {family: {"family": family, "grid": grid}
               for family, grid in snapshot_module._GRIDS.items()}
    return executes, formats


def _traces(*digests):
    traces = snapshot_module._read(snapshot_module.TRACE_509,
                                   snapshot_module.TRACE_509_FILES)
    out = []
    for (label, payload), digest in zip(traces, digests):
        payload = copy.deepcopy(payload)
        if digest is not _ABSENT:
            payload["serving_source_sha256"] = digest
        out.append((label, payload))
    return out


_ABSENT = object()


def _compare(traces, **kwargs):
    executes, formats = _trace_contract()
    config = json.loads((snapshot_module.TRACE_M44E1 / "config.json").read_text())
    return gate.compare_route_traces(
        traces, expected_ranks=2, config=config, platform="sm_121",
        executes_by_platform=executes, formats=formats, **kwargs)


def test_every_rank_serving_the_pinned_code_agrees():
    verdict = _compare(_traces(PINNED, PINNED), serving_source_sha256=PINNED)
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    assert verdict["serving_source_sha256"] == {
        "pinned": PINNED, "served": {"rank0": PINNED, "rank1": PINNED}}


def test_a_rank_serving_other_code_is_refused_naming_both():
    verdict = _compare(_traces(PINNED, OTHER), serving_source_sha256=PINNED)
    assert verdict["status"] == gate.REFUSED
    assert PINNED in verdict["detail"] and OTHER in verdict["detail"]
    assert "rank1" in verdict["detail"]


@pytest.mark.parametrize("digests", [
    (_ABSENT, _ABSENT), (PINNED, _ABSENT), (None, PINNED), (None, None)])
def test_an_absent_digest_is_not_verified(digests):
    verdict = _compare(_traces(*digests), serving_source_sha256=PINNED)
    assert verdict["status"] == gate.NOT_VERIFIED, verdict["detail"]
    assert "serving_source_sha256" in verdict["detail"]


@pytest.mark.parametrize("bad", ["xyz", PINNED.upper(), 7, PINNED[:-1]])
def test_a_malformed_digest_is_refused(bad):
    verdict = _compare(_traces(PINNED, bad), serving_source_sha256=PINNED)
    assert verdict["status"] == gate.REFUSED, verdict["detail"]


def test_a_v2_pin_reads_no_digest_and_writes_no_verdict_key():
    for traces in (_traces(_ABSENT, _ABSENT), _traces(OTHER, "xyz")):
        verdict = _compare(traces, serving_source_sha256=None)
        assert "serving_source_sha256" not in verdict
        assert verdict == _compare(traces)
        assert verdict["status"] == gate.AGREE


def test_the_trace_gate_default_reads_the_tracked_pin(monkeypatch):
    monkeypatch.setattr(pin_module, "pinned_serving_source_sha256", lambda: PINNED)
    assert _compare(_traces(_ABSENT, _ABSENT))["status"] == gate.NOT_VERIFIED
    assert _compare(_traces(OTHER, OTHER))["status"] == gate.REFUSED
    assert _compare(_traces(PINNED, PINNED))["status"] == gate.AGREE


# ---------------------------------------------------------------------------
# A v2 pin behaves exactly as before: a before/after comparison
# ---------------------------------------------------------------------------
def test_every_gate_answers_exactly_as_it_did_before_under_the_v2_pin():
    """Byte equality with the answers taken on origin/main before this change.

    ``GOLDEN`` was produced by ``tests.tessera_serving_identity_snapshot`` on
    ``a796448b5e8`` plus only that helper, through PrismaBuild. It covers the
    tracked pin and the live pin gate, all 14 packaged cells, the 56 unit routes
    their scopes resolve to, the development contract's reviewed answer and
    admitted cells, and eight route-trace verdicts, four of them on traces
    whose header carries a digest a v2 pin must not read.

    Regenerated once, through PrismaBuild, when #1490 merged onto this gate:
    the route-trace verdicts gained the glm5_next served-namespace fields
    (``served_namespace``, renamed priced targets, one sentence in
    ``detail``), and no verdict changed.
    """
    now = snapshot_module.canonical(snapshot_module.snapshot())
    before = GOLDEN.read_text(encoding="utf-8")
    assert now == before
