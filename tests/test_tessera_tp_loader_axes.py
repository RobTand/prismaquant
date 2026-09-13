"""The loader-axis leg of tensor-parallel legality (issue #536).

Three legs decide whether a Tessera rung is servable at ``tp > 1`` and they
answer different questions:

* **attested world** -- ``tensor_parallel.units[].max_world_size``: has a
  served receipt covered this world size? (``tessera_tp_world_attested``)
* **shard geometry** -- ``tessera.layout.can_shard``: does the rung's own
  period divide the shard a rank holds?
* **loader axis** -- ``tensor_parallel.units[].loader_axes``: does this
  build's loader accept a cut on this axis at all?

The third is what this file pins. It is a published STATUS, not prose and not
a bound: the pinned contract refuses ``TESSERA_E2M1_K2`` on the ``row`` axis
because a row shard begins mid-column and the span-2 decoders supply
``state_{-1}`` themselves, and it refuses it on every rank, so no world size
makes it legal. A column-parallel Linear splits vLLM's output features, which
are this unit's rows, so ``PARALLEL_COLUMN`` is the cut that asks about
``row`` -- the same mapping the geometry leg uses, and inverting it would gate
exactly the wrong half of a model.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import prismaquant.tessera_menu as tm
import prismaquant.tessera_runtime_contract as trc

E2M1 = "TESSERA_E2M1_K2"
#: The contract's unit key for what Tessera's sharding table calls
#: ``TESSERA_FP8``: both axes shard.
FP8 = "TESSERA_E4M3_K1"
BF16 = "TESSERA_BF16_K1"

#: Divides cleanly by 2 on both axes, so nothing here refuses on geometry.
SHAPE = (2048, 1024)

_BOTH_SHARDED = {"row": "sharded", "column": "sharded"}
#: What the pinned contract publishes for the K2 family.
_COLUMN_CUT_REFUSED = {"row": "refused", "column": "sharded"}


def _axis_contract(monkeypatch, axes, *, max_world_size=8):
    """A stand-in contract that publishes ``axes`` and attests nothing.

    It governs no family on purpose: the axis leg is a LOADER fact and must
    answer without a route claim, so a table that attests nothing is the
    honest witness for it.

    ``max_world_size`` is deliberately generous. The attestation leg is a
    different question and it must not be the leg that answers here: a table
    that attested nothing would refuse first and hide whether the axis leg
    works at all.
    """
    contract = SimpleNamespace(
        commit="synthetic",
        requires_serving_context=False,
        lane_schema="legacy",
        max_world_size={family: max_world_size for family in axes},
        reader_rate_range={},
        attested_rungs={},
        loader_axes=axes,
        governs=lambda name: False,
        native_cells=lambda name, rate, **_kw: (),
    )
    monkeypatch.setattr(tm, "tessera_runtime_contract", lambda: contract)
    return contract


def _rungs(family, *, tp_degree, parallel_kind):
    return [
        rung
        for rung in tm.expand_tessera_menu(
            SHAPE, mode=tm.MENU_RESEARCH,
            tp_degree=tp_degree, parallel_kind=parallel_kind)
        if rung.family == family
    ]


def _menu(family, *, tp_degree, parallel_kind):
    return {rung.format_name for rung in
            _rungs(family, tp_degree=tp_degree, parallel_kind=parallel_kind)}


def _first_rung(family):
    """A rung this family really carries on ``SHAPE``, from the menu itself.

    Hard-coding one would make a geometry refusal read as an axis refusal.
    """
    rungs = _rungs(family, tp_degree=1, parallel_kind=tm.PARALLEL_NONE)
    assert rungs, f"{family} carries no rung on {SHAPE}"
    return rungs[0].body_rate_q256


def _axes_payload(units):
    return {"tensor_parallel": {"axis": "vllm_tensor_parallel_world_size",
                                "semantics": "closed_world",
                                "units": list(units)}}


def _unit(name, axes, *, max_world_size=1):
    unit = {"unit": name, "kind": "tessera_wire_family",
            "max_world_size": max_world_size}
    if axes is not None:
        unit["loader_axes"] = {
            axis: {"status": status, "reason": None}
            for axis, status in axes.items()
        }
    return unit


def _written(tmp_path, payload, name="runtime_contract.json"):
    path = tmp_path / name
    raw = json.dumps(payload, indent=1).encode("utf-8")
    path.write_bytes(raw)
    return str(path), hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# The contract reader: a published vocabulary, or a refusal
# ---------------------------------------------------------------------------

def _installed_contract():
    """The contract bytes the importable Tessera actually packages."""
    from importlib.resources import as_file

    with as_file(trc.contract_path()) as path:
        raw = path.read_bytes()
        return str(path), hashlib.sha256(raw).hexdigest(), json.loads(raw)


def test_the_real_pinned_contract_round_trips_every_declared_loader_axis():
    """The real table, read by the real reader, for all three families.

    Read off the packaged file rather than through ``load_tessera_contract``
    so this says what the TABLE publishes and not what the answer pin
    accepts: the two are separate refusals and the answer pin has its own
    test.
    """
    path, sha, payload = _installed_contract()
    assert trc.published_tensor_parallel_axes(path, sha) == {
        E2M1: {"column": "sharded", "row": "refused"},
        FP8: {"column": "sharded", "row": "sharded"},
        BF16: {"column": "sharded", "row": "sharded"},
    }
    assert (set(trc.published_tensor_parallel_axes(path, sha))
            == set(trc.published_tensor_parallel_limits(path, sha))), (
        "both facts come off the same unit row; a family with a ceiling and "
        "no axis claim would be half-read"
    )
    for unit in payload["tensor_parallel"]["units"]:
        assert "reason" in unit["loader_axes"]["row"], (
            "the publisher's prose is there to be ignored, not absent"
        )


def test_the_published_axes_accessor_reads_the_same_table(tmp_path):
    """The file-reading path and the parsed-contract path are one reader."""
    path, sha = _written(tmp_path, _axes_payload([
        _unit(E2M1, _COLUMN_CUT_REFUSED), _unit(FP8, _BOTH_SHARDED)]))
    assert trc.published_tensor_parallel_axes(path, sha) == {
        E2M1: {"column": "sharded", "row": "refused"},
        FP8: {"column": "sharded", "row": "sharded"},
    }


def test_a_unit_with_no_loader_axes_block_is_refused(tmp_path):
    """Absence is not "both axes shard"; it is a table this reader cannot use."""
    path, sha = _written(tmp_path, _axes_payload([_unit(FP8, None)]))
    with pytest.raises(trc.TesseraContractError) as excinfo:
        trc.published_tensor_parallel_axes(path, sha)
    assert "loader_axes" in str(excinfo.value)


@pytest.mark.parametrize("axes", [
    pytest.param({"row": "sharded"}, id="missing-an-axis"),
    pytest.param({"row": "sharded", "column": "sharded", "diagonal": "sharded"},
                 id="an-axis-this-reader-does-not-know"),
])
def test_an_axis_vocabulary_this_reader_does_not_share_is_refused(tmp_path, axes):
    path, sha = _written(tmp_path, _axes_payload([_unit(FP8, axes)]))
    with pytest.raises(trc.TesseraContractError):
        trc.published_tensor_parallel_axes(path, sha)


def test_a_status_this_reader_does_not_know_is_refused(tmp_path):
    """``maybe`` is neither ``sharded`` nor ``refused``, so it is neither."""
    path, sha = _written(tmp_path, _axes_payload([
        _unit(FP8, {"row": "maybe", "column": "sharded"})]))
    with pytest.raises(trc.TesseraContractError) as excinfo:
        trc.published_tensor_parallel_axes(path, sha)
    assert "maybe" in str(excinfo.value)


def test_a_status_read_from_prose_is_refused(tmp_path):
    """The status field is the value; the reason beside it is never read."""
    payload = _axes_payload([_unit(FP8, _BOTH_SHARDED)])
    unit = payload["tensor_parallel"]["units"][0]
    unit["loader_axes"]["row"] = {"reason": "sharded"}
    path, sha = _written(tmp_path, payload)
    with pytest.raises(trc.TesseraContractError) as excinfo:
        trc.published_tensor_parallel_axes(path, sha)
    assert "status" in str(excinfo.value)


def test_the_reviewed_answer_carries_the_axis_statuses():
    """A value a gate reads is answer; widening the projection is the review.

    The literal is what refuses, so this asserts the shape of what was
    widened: statuses only, per family, beside the ceiling they are
    deliberately not a spelling of. The publisher's reason is prose and
    stays out.
    """
    answer = trc.TESSERA_DEV_PIN_ANSWER
    for family in (E2M1, FP8, BF16):
        entry = answer["families"][family]
        assert set(entry) == {
            "reader_rate_range_q256", "attested_rungs_q256", "max_world_size",
            "loader_axes",
        }
        assert set(entry["loader_axes"]) == {"row", "column"}
        assert set(entry["loader_axes"].values()) <= {"sharded", "refused"}
    assert answer["families"][E2M1]["loader_axes"]["row"] == "refused"
    assert "state_{-1}" not in repr(answer), (
        "the publisher's reason is prose and no gate reads it"
    )


def test_the_projection_emits_what_the_reviewed_answer_declares():
    """``contract_answer`` really reads the field the literal was widened for.

    Against the installed contract, whichever pin it carries: every build
    that publishes this block publishes the same statuses, and a literal that
    named a field the projection never emitted would drift on every run.
    """
    path, sha, payload = _installed_contract()
    contract = trc._parse(payload, commit="installed", sha=sha, path=path)
    answer = trc.contract_answer(contract)
    for family in (E2M1, FP8, BF16):
        assert (answer["families"][family]["loader_axes"]
                == dict(sorted(contract.loader_axes[family].items())))
    assert answer["families"][E2M1]["loader_axes"]["row"] == "refused"


# ---------------------------------------------------------------------------
# The legality leg
# ---------------------------------------------------------------------------

def test_a_refused_cut_axis_refuses_the_rung_at_tp2(monkeypatch):
    """The regression: one family refused on the axis a column cut asks for."""
    _axis_contract(monkeypatch, {E2M1: _COLUMN_CUT_REFUSED, FP8: _BOTH_SHARDED})
    at_one = _rungs(E2M1, tp_degree=1, parallel_kind=tm.PARALLEL_COLUMN)
    assert at_one, "the synthetic table must leave K2 rungs to remove"
    body_rate_q256 = at_one[0].body_rate_q256

    legal, reason = tm.tessera_tp_legal(
        E2M1, body_rate_q256, SHAPE,
        tp_degree=1, parallel_kind=tm.PARALLEL_COLUMN)
    assert legal, reason

    legal, reason = tm.tessera_tp_legal(
        E2M1, body_rate_q256, SHAPE,
        tp_degree=2, parallel_kind=tm.PARALLEL_COLUMN)
    assert not legal
    assert reason == f"tp_axis_refused:{E2M1}:{E2M1}:row", reason


def test_the_axis_leg_does_not_wait_for_the_attestation_leg(monkeypatch):
    """Two legs, and this one answers on its own.

    ``require_attested_world`` turns the world-size question on. The loader
    fact is a different question -- can the loader cut it at all -- and a
    research menu that prices unattested rungs on purpose still must not
    price a cut the loader refuses on every rank. The table here attests the
    world size, so the attestation leg passes in both modes and the axis leg
    is the one that answers.
    """
    _axis_contract(monkeypatch, {E2M1: _COLUMN_CUT_REFUSED})
    attested, why = tm.tessera_tp_world_attested(E2M1, 2)
    assert attested, why
    for require in (False, True):
        legal, reason = tm.tessera_tp_legal(
            E2M1, _first_rung(E2M1), SHAPE, tp_degree=2,
            parallel_kind=tm.PARALLEL_COLUMN,
            require_attested_world=require)
        assert not legal
        assert reason.startswith("tp_axis_refused:"), (require, reason)


def test_the_refusal_names_the_linear_when_the_caller_knows_it(monkeypatch):
    """``tp_axis_refused:<family>:<unit>:<axis>`` -- unit is the caller's."""
    _axis_contract(monkeypatch, {E2M1: _COLUMN_CUT_REFUSED})
    qname = "model.layers.0.self_attn.q_proj"
    legal, reason = tm.tessera_tp_legal(
        E2M1, _first_rung(E2M1), SHAPE, tp_degree=2,
        parallel_kind=tm.PARALLEL_COLUMN, unit=qname)
    assert not legal
    assert reason == f"tp_axis_refused:{E2M1}:{qname}:row"


def test_the_other_cut_direction_is_not_refused(monkeypatch):
    """Only the refused axis is refused: a row-parallel cut asks about columns."""
    _axis_contract(monkeypatch, {E2M1: _COLUMN_CUT_REFUSED})
    _, reason = tm.tessera_tp_legal(
        E2M1, _first_rung(E2M1), SHAPE, tp_degree=2,
        parallel_kind=tm.PARALLEL_ROW)
    assert "tp_axis_refused" not in reason


def test_a_family_that_shards_both_axes_passes_at_both_degrees(monkeypatch):
    """The control arm: same table, same shape, no refusal at either degree."""
    _axis_contract(monkeypatch, {E2M1: _COLUMN_CUT_REFUSED, FP8: _BOTH_SHARDED})
    at_one = _menu(FP8, tp_degree=1, parallel_kind=tm.PARALLEL_COLUMN)
    at_two = _menu(FP8, tp_degree=2, parallel_kind=tm.PARALLEL_COLUMN)
    assert at_one, "the control family must have rungs on this shape"
    assert at_two == at_one, sorted(at_one ^ at_two)

    assert _menu(E2M1, tp_degree=2, parallel_kind=tm.PARALLEL_COLUMN) == set(), (
        "the refused family leaves the menu at the degree it is refused at"
    )


def test_a_whole_unit_is_never_refused_on_an_axis_it_does_not_cut(monkeypatch):
    """``tp=1`` and ``PARALLEL_NONE`` shard nothing, so no axis is asked."""
    _axis_contract(monkeypatch, {E2M1: _COLUMN_CUT_REFUSED})
    rung = _first_rung(E2M1)
    for tp, kind in ((1, tm.PARALLEL_COLUMN), (2, tm.PARALLEL_NONE)):
        legal, reason = tm.tessera_tp_legal(
            E2M1, rung, SHAPE, tp_degree=tp, parallel_kind=kind)
        assert legal, (tp, kind, reason)


def test_with_no_contract_pinned_the_axis_leg_adds_no_refusal(monkeypatch):
    """This leg subtracts on a published fact; it never invents one.

    With no pinned table there is no loader fact to read, and refusing here
    would make the TP gate a second route gate that turns Tessera off on the
    research menu -- whose whole purpose is to price what nothing attests.
    """
    monkeypatch.setattr(tm, "tessera_runtime_contract", lambda: None)
    legal, reason = tm.tessera_tp_legal(
        E2M1, _first_rung(E2M1), SHAPE, tp_degree=2,
        parallel_kind=tm.PARALLEL_COLUMN)
    assert legal, reason


def test_a_family_the_table_does_not_list_gets_no_axis_verdict(monkeypatch):
    """Silence about a family is silence, not a refusal.

    The closed world lives on ``max_world_size`` and the attestation leg
    reads it. This leg reports a published status and nothing else.
    """
    _axis_contract(monkeypatch, {FP8: _BOTH_SHARDED})
    legal, reason = tm.tessera_tp_legal(
        E2M1, _first_rung(E2M1), SHAPE, tp_degree=2,
        parallel_kind=tm.PARALLEL_COLUMN)
    assert legal, reason
