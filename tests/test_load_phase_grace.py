"""The derived stall allowance of Stage B's load phases (PB #480).

checkpoint-load and handoff-load commit no progress units, so their grace is
the phase's whole time budget. The dispatcher derives it per row as
W + ceil(bytes / floor): W is the spec's staged-range wait, which the reader
spends at most once per phase, and the floor is a read rate measured for one
reader on the dl380g10 link. A floor applies only at the link-reader count it
was measured at; any other count takes the blanket grace, and the stamp says
which. The count is this dispatch's own rows (``--quantum`` narrows it).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import dispatch_joint_quanta as dispatch  # noqa: E402
from dispatch_joint_quanta import (  # noqa: E402
    EXIT_PRECONDITION_REFUSED,
    HEAD_PROGRESS_GRACE_S,
    LINK_READERS_SCOPE,
    LOAD_PHASE_FLOOR_ONE_READER,
    LOAD_PHASE_FLOOR_SCHEMA,
    PROGRESS_GRACE_FLAG,
    DispatchRefused,
    link_readers,
    load_phase_grace,
    require_staged_wait_below_grace,
)
from tests.test_band_serial_dispatch import (  # noqa: E402,F401
    _by_id,
    _dispatch_layout,
    _inner,
    _main,
    _offline_tier_policy,
)

#: The R13 layer-044 gate's checkpoint-load phase (its sealed read plan).
LAYER_044_CHECKPOINT_BYTES = 34_367_176_983
FLOOR = LOAD_PHASE_FLOOR_ONE_READER["floor_bytes_per_s"]
V4_KEY = "70e7baeb6e96564ed5fd05b8eb90f1f709a550dcc213298a3dff5db5dd5dea8b"
V5_KEY = "2dc145299e0eebfef1b3fbc159307e07c4a5a2ca0cf394c187018a1e9d907e86"


def _floor_document(readers, floor=20_000_000):
    return {"schema": LOAD_PHASE_FLOOR_SCHEMA, "floor_bytes_per_s": floor,
            "readers": readers, "scope": f"{readers} readers on the dl380g10 link",
            "method": "slowest 30 s window, fixture",
            "sources": [{"action_key": "c" * 64}]}


def test_the_layer_044_gate_derives_846_seconds_from_the_one_reader_floor():
    link = link_readers(rows=1)
    stamp = load_phase_grace("checkpoint-load",
                             phase_bytes=LAYER_044_CHECKPOINT_BYTES,
                             staged_wait_s=300.0, link=link)
    assert stamp["mode"] == "derived"
    assert stamp["grace_s"] == 846
    assert stamp["transfer_s"] == 546
    assert stamp["phase_bytes"] == LAYER_044_CHECKPOINT_BYTES
    assert stamp["staged_wait_s"] == 300.0
    assert stamp["floor_bytes_per_s"] == FLOOR == 62_954_973
    assert stamp["floor_readers"] == stamp["readers"] == 1
    assert stamp["readers_source"] == "the quantum rows this dispatch publishes"
    assert stamp["floor_scope"] == "one reader on the dl380g10 link"
    assert stamp["readers_scope"] == LINK_READERS_SCOPE
    assert "other workloads" in stamp["readers_scope"]
    assert [s["action_key"] for s in stamp["floor_sources"]] == [V4_KEY, V5_KEY]
    # The floor is the slower of the two measured gates.
    assert FLOOR == min(s["bytes_per_s"] for s in stamp["floor_sources"])
    assert stamp["floor_document"] is None
    # The derived grace clears the reader's own deadline, so the staged-wait
    # rule holds with the gate's 300 s wait.
    require_staged_wait_below_grace(
        {"env": {"PRISMAQUANT_STAGED_RANGE_WAIT_S": "300"}},
        [("head", HEAD_PROGRESS_GRACE_S), ("checkpoint-load", 846),
         ("chain-043", 900)])


def test_the_bound_cites_the_reader_deadline_lines():
    """The stamp's file:line references are the reader's one deadline."""
    for ref in ("prismaquant/joint_adjoint_checkpoints.py:1846",
                "prismaquant/joint_quantum_handoff.py:525"):
        assert ref in dispatch.LOAD_PHASE_BOUND
        path, line = ref.rsplit(":", 1)
        text = (ROOT / path).read_text().splitlines()[int(line) - 1]
        assert text.strip() == (
            "deadline = time.monotonic() + staged_range_wait_s()"), ref


@pytest.mark.parametrize("rows,declared", [(5, None), (1, 3)])
def test_more_than_one_reader_without_a_floor_takes_the_blanket_grace(rows, declared):
    link = link_readers(rows=rows, declared=declared)
    assert link["floor"] is None
    stamp = load_phase_grace("checkpoint-load",
                             phase_bytes=LAYER_044_CHECKPOINT_BYTES,
                             staged_wait_s=300.0, link=link)
    assert stamp["mode"] == "blanket"
    assert stamp["grace_s"] == HEAD_PROGRESS_GRACE_S
    readers = rows if declared is None else declared
    assert stamp["readers"] == readers
    assert f"no floor measured at {readers} link readers" in stamp["reason"]
    assert stamp["readers_source"] == (
        "--link-readers" if declared else "the quantum rows this dispatch publishes")
    # No concurrency discount: the one-reader floor is nowhere in the stamp.
    assert "floor_bytes_per_s" not in stamp


def test_a_floor_measured_at_another_concurrency_refuses():
    with pytest.raises(DispatchRefused, match="measured at 2 link reader"):
        link_readers(rows=3, floor_document=_floor_document(2))


def test_a_floor_measured_at_the_row_count_derives_and_names_its_document():
    document = {**_floor_document(3), "document": {"path": "/x/floor.json",
                                                   "sha256": "d" * 64}}
    link = link_readers(rows=3, floor_document=document)
    stamp = load_phase_grace("checkpoint-load", phase_bytes=40_000_001,
                             staged_wait_s=299.5, link=link)
    assert stamp["mode"] == "derived"
    assert stamp["grace_s"] == 300 + 3
    assert stamp["floor_readers"] == 3
    assert stamp["floor_document"] == {"path": "/x/floor.json", "sha256": "d" * 64}
    assert [s["action_key"] for s in stamp["floor_sources"]] == ["c" * 64]


@pytest.mark.parametrize("phase_bytes,why", [(None, "no byte count"),
                                             (0, "0 bytes")])
def test_a_phase_without_bytes_to_bound_takes_the_blanket_grace(phase_bytes, why):
    stamp = load_phase_grace("handoff-load", phase_bytes=phase_bytes,
                             staged_wait_s=300.0, link=link_readers(rows=1))
    assert stamp["mode"] == "blanket"
    assert stamp["grace_s"] == HEAD_PROGRESS_GRACE_S
    assert why in stamp["reason"]


def test_a_row_built_without_a_link_count_says_so():
    stamp = load_phase_grace("checkpoint-load", phase_bytes=1,
                             staged_wait_s=300.0, link=None)
    assert stamp["mode"] == "blanket"
    assert stamp["readers"] is None
    assert "without a link-reader count" in stamp["reason"]


@pytest.mark.parametrize("change", [
    {"schema": "other"}, {"floor_bytes_per_s": 0}, {"floor_bytes_per_s": 1.5},
    {"readers": 0}, {"scope": ""}, {"sources": []},
    {"sources": [{"action_key": "short"}]}])
def test_malformed_floor_documents_refuse(change):
    with pytest.raises(DispatchRefused, match="load-phase floor document"):
        link_readers(rows=1, floor_document={**_floor_document(1), **change})


def test_a_declared_reader_count_below_one_refuses():
    with pytest.raises(DispatchRefused, match="at least 1"):
        link_readers(rows=1, declared=0)


# The dispatch edge, on the band-serial test's four bound executable rows.


def _graces(argv):
    return {name: int(value) for name, value in (
        argv[i + 1].split("=", 1) for i, word in enumerate(argv)
        if word == "--progress-phase")}


def _stamps(argv):
    inner = _inner(argv)
    return json.loads(inner[inner.index(PROGRESS_GRACE_FLAG) + 1])


def _checkpoint_bytes(record):
    wire = Path(record["executable_readset"]["manifest_path"]).read_bytes()
    phases = json.loads(gzip.decompress(wire))["read_plan"]["phases"]
    return {p["name"]: p["bytes"] for p in phases}["checkpoint-load"]


def _events(out):
    path = out / "layer-quanta" / dispatch.STATE_FILENAME
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_filtered_dispatch_counts_only_its_own_rows(tmp_path, monkeypatch):
    _, bound, records, receipt_path, out = _dispatch_layout(tmp_path, monkeypatch)
    gateway = dispatch.FakeGateway()
    assert _main(dispatch, gateway, records, receipt_path, out,
                 "--quantum", "layer-003") == 0
    assert sorted(_by_id(gateway)) == ["layer-003"]
    argv = _by_id(gateway)["layer-003"]["argv"]
    phase_bytes = _checkpoint_bytes(bound[3])
    assert phase_bytes > 0
    expected = 300 + -(-phase_bytes // FLOOR)
    assert _graces(argv)["checkpoint-load"] == expected
    (stamp,) = _stamps(argv)
    assert (stamp["mode"], stamp["readers"], stamp["grace_s"]) == (
        "derived", 1, expected)
    assert stamp["readers_source"] == "the quantum rows this dispatch publishes"
    (event,) = [e for e in _events(out) if e.get("event") == "quantum-submitted"]
    assert event["quantum_id"] == "layer-003"
    assert event["link"]["readers"] == 1
    assert event["progress_grace"] == [stamp]


def test_an_unfiltered_dispatch_counts_every_row_it_publishes(tmp_path, monkeypatch):
    _, bound, records, receipt_path, out = _dispatch_layout(tmp_path, monkeypatch)
    gateway = dispatch.FakeGateway()
    assert _main(dispatch, gateway, records, receipt_path, out) == 0
    rows = _by_id(gateway)
    assert len(rows) == len(bound) > 1
    loads = 0
    for row in rows.values():
        graces = _graces(row["argv"])
        if "checkpoint-load" not in graces:
            continue
        loads += 1
        assert graces["checkpoint-load"] == HEAD_PROGRESS_GRACE_S
        (stamp,) = _stamps(row["argv"])
        assert (stamp["mode"], stamp["readers"]) == ("blanket", len(bound))
    assert loads > 0


def test_an_unknown_quantum_id_refuses(tmp_path, monkeypatch, capsys):
    _, _bound, records, receipt_path, out = _dispatch_layout(tmp_path, monkeypatch)
    gateway = dispatch.FakeGateway()
    assert _main(dispatch, gateway, records, receipt_path, out,
                 "--quantum", "layer-003", "--quantum", "layer-999") == (
        EXIT_PRECONDITION_REFUSED)
    assert "layer-999" in capsys.readouterr().err
    assert gateway.submitted == []


def test_a_resubmitted_row_keeps_the_link_it_was_first_submitted_with(
        tmp_path, monkeypatch):
    """The same row stays the same sealed action whatever else a run holds."""
    _, _bound, records, receipt_path, out = _dispatch_layout(tmp_path, monkeypatch)
    gateway = dispatch.FakeGateway()
    assert _main(dispatch, gateway, records, receipt_path, out) == 0
    first = _by_id(gateway)["layer-003"]["argv"]
    gateway.submitted = []
    assert _main(dispatch, gateway, records, receipt_path, out,
                 "--quantum", "layer-003") == 0
    assert _by_id(gateway)["layer-003"]["argv"] == first
    assert _stamps(first)[0]["readers"] == 4


def test_a_floor_document_at_the_dispatch_count_derives_every_row(
        tmp_path, monkeypatch):
    _, bound, records, receipt_path, out = _dispatch_layout(tmp_path, monkeypatch)
    floor = tmp_path / "floor.json"
    floor.write_text(json.dumps(_floor_document(len(bound), floor=1000)))
    gateway = dispatch.FakeGateway()
    assert _main(dispatch, gateway, records, receipt_path, out,
                 "--checkpoint-load-floor", str(floor)) == 0
    stamps = [stamp for row in _by_id(gateway).values()
              for stamp in _stamps(row["argv"])]
    assert stamps and {s["mode"] for s in stamps} == {"derived"}
    assert {s["floor_document"]["path"] for s in stamps} == {str(floor)}


def test_a_floor_document_at_another_count_refuses_the_dispatch(
        tmp_path, monkeypatch, capsys):
    _, _bound, records, receipt_path, out = _dispatch_layout(tmp_path, monkeypatch)
    floor = tmp_path / "floor.json"
    floor.write_text(json.dumps(_floor_document(2)))
    gateway = dispatch.FakeGateway()
    assert _main(dispatch, gateway, records, receipt_path, out,
                 "--checkpoint-load-floor", str(floor)) == EXIT_PRECONDITION_REFUSED
    assert "measured at 2 link reader" in capsys.readouterr().err
    assert gateway.submitted == []


def test_the_quantum_parses_the_stamps_and_keeps_a_malformed_value(capsys):
    from prismaquant.joint_cost_quantum import build_parser, progress_grace_stamps

    stamps = [{"schema": "prismaquant.load_phase_grace.v1", "grace_s": 846}]
    raw = json.dumps(stamps)
    assert progress_grace_stamps(None) is None
    assert progress_grace_stamps(raw) == stamps
    assert progress_grace_stamps("{not json") == [{"unparsed": "{not json"}]
    assert "recording it unparsed" in capsys.readouterr().out
    args = build_parser().parse_args([
        "--quantum", "q", "--quantum-sha256", "a", "--plan", "p",
        "--plan-sha256", "b", "--prepared", "r", "--prepared-sha256", "c",
        "--adjoint-slice", "s", "--adjoint-slice-sha256", "d",
        "--output-root", "o", PROGRESS_GRACE_FLAG, raw])
    assert args.progress_grace_derivation == raw
