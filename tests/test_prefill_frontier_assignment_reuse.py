"""The digest-named assignment a frontier point publishes must BE that assignment.

PQ-R3.  ``prefill_frontier._point_record`` wrote ``<digest>.json`` only when the
path was absent and never read a file that was already there, so a leftover at
that name -- an interrupted write, a hand edit, another table's sweep -- was
reused and the point published the CURRENT solve's dloss, attained prefill and
digest beside whatever bytes happened to be on disk.  The review's reproduction
replaced a BF16 assignment with an NVFP4 one and watched the second call return
the BF16 digest pointing at the NVFP4 file.

These tests import the module for real (``prismaquant.prefill_frontier``, whose
import chain reaches ``compressed_tensors`` through ``format_registry``).  An
AST-extracted copy of the function does not run this file, which is the point:
the review's first PB attempt failed on the import and could not exercise it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant import prefill_frontier
from prismaquant.layer_config import LAYER_CONFIG_META_KEY
from prismaquant.measured_runtime_prices import identity_sha256

ASSIGNMENT = {"model.layers.0.self_attn.o_proj": "NVFP4",
              "model.layers.1.self_attn.o_proj": "FP8_E4M3"}
OTHER_ASSIGNMENT = {"model.layers.0.self_attn.o_proj": "FP8_E4M3",
                    "model.layers.1.self_attn.o_proj": "FP8_E4M3"}
#: The provenance this module stamps beside an assignment, as ``run_sweep``
#: builds it: the table the solve priced against.
PROVENANCE = {"table_id": "synthetic-only", "table_sha256": "a" * 64}
OTHER_TABLE = {"table_id": "synthetic-only", "table_sha256": "b" * 64}


def _record(assignment=None) -> dict:
    return {
        "slo_ms": 12.5,
        "target_bits": 4.0,
        "feasible": True,
        "assignment": dict(ASSIGNMENT if assignment is None else assignment),
        "predicted_dloss": 1.25,
        "achieved_bits": 3.75,
        "payload_bytes": 4096,
        "serve_constraints": {"predicted": {"operator_sum_prefill_ms": 11.0,
                                            "operator_sum_decode_ms": 4.0,
                                            "device_memory_bytes": 1 << 20}},
        "diagnostics": {"frontier_size": 3},
    }


def _digest(assignment) -> str:
    return identity_sha256(assignment)


def _encoded(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _payload(assignment, *, provenance=PROVENANCE, digest=None) -> dict:
    """The file this module writes, from an assignment that need not be current."""
    meta = {
        "schema": prefill_frontier.ASSIGNMENT_SCHEMA,
        "assignment_sha256": digest or _digest(assignment),
        "research_only": True,
        "note": "fixture written by tests/test_prefill_frontier_assignment_reuse.py",
        **provenance,
    }
    return {**assignment, LAYER_CONFIG_META_KEY: meta}


def _point(assignments_dir, *, provenance=PROVENANCE, record=None) -> dict:
    return prefill_frontier._point_record(
        record or _record(), assignments_dir, provenance_stub=dict(provenance))


def test_a_leftover_file_at_the_digest_path_is_read_before_it_is_reused(tmp_path):
    """The review's reproduction: the file is a DIFFERENT assignment.

    Nothing in the name proves the bytes; the point must refuse rather than
    return the current solve's metrics and digest beside them.
    """
    digest = _digest(ASSIGNMENT)
    path = tmp_path / f"{digest}.json"
    path.write_bytes(_encoded(_payload(OTHER_ASSIGNMENT)))
    before = path.read_bytes()

    with pytest.raises(prefill_frontier.PrefillFrontierError) as caught:
        _point(tmp_path)
    message = str(caught.value)
    assert digest in message and str(path) in message, message
    assert path.read_bytes() == before, "a refused reuse must not rewrite the file"


def test_a_truncated_assignment_file_is_refused_rather_than_reused(tmp_path):
    """An interrupted write leaves unparseable bytes at an expected name."""
    digest = _digest(ASSIGNMENT)
    path = tmp_path / f"{digest}.json"
    path.write_bytes(_encoded(_payload(ASSIGNMENT))[:40])

    with pytest.raises(prefill_frontier.PrefillFrontierError) as caught:
        _point(tmp_path)
    assert "JSON" in str(caught.value), str(caught.value)


def test_reuse_across_tables_is_recorded_rather_than_adopted(tmp_path):
    """The explicit changed-provenance contract for an identical assignment.

    The filename is the assignment's digest, so two tables that agree on an
    assignment resolve to one path -- re-measuring a row without moving its
    median does it, which is what ``test_prefill_frontier_dispersion`` runs.  The
    assignment is reusable, because it IS the assignment; what must not happen is
    this run adopting the file's provenance as its own, so the point reports the
    provenance the file records beside its own numbers.
    """
    digest = _digest(ASSIGNMENT)
    path = tmp_path / f"{digest}.json"
    written = _encoded(_payload(ASSIGNMENT, provenance=OTHER_TABLE))
    path.write_bytes(written)

    point = _point(tmp_path, provenance=PROVENANCE)
    assert point["assignment_sha256"] == digest
    assert point["assignment_file_provenance"] == OTHER_TABLE
    assert point["assignment_file_provenance"] != PROVENANCE
    assert path.read_bytes() == written, "the first publisher's file is left alone"


def test_an_assignment_file_with_no_readable_provenance_is_refused(tmp_path):
    """``cannot tell which table published it`` is not ``reuse it anyway``."""
    digest = _digest(ASSIGNMENT)
    path = tmp_path / f"{digest}.json"
    payload = _payload(ASSIGNMENT)
    del payload[LAYER_CONFIG_META_KEY]["table_sha256"]
    path.write_bytes(_encoded(payload))

    with pytest.raises(prefill_frontier.PrefillFrontierError) as caught:
        _point(tmp_path)
    assert "table_sha256" in str(caught.value), str(caught.value)


def test_an_existing_verified_assignment_is_reused_without_being_rewritten(tmp_path):
    """Identical valid reuse: the binding is established, the bytes are untouched."""
    digest = _digest(ASSIGNMENT)
    path = tmp_path / f"{digest}.json"
    path.write_bytes(_encoded(_payload(ASSIGNMENT)))
    before = path.read_bytes()
    stamp = path.stat().st_mtime_ns

    point = _point(tmp_path)
    assert point["assignment_sha256"] == digest
    assert Path(point["assignment_path"]) == path
    assert point["assignment_file_provenance"] == PROVENANCE
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == stamp, "a verified reuse does not rewrite the file"


def test_a_fresh_sweep_publishes_a_bound_research_only_assignment(tmp_path):
    """The published file is the solve's own, and still research-only."""
    digest = _digest(ASSIGNMENT)
    point = _point(tmp_path)
    assert point["assignment_sha256"] == digest
    assert point["assignment_file_provenance"] == PROVENANCE
    path = Path(point["assignment_path"])
    assert path == tmp_path / f"{digest}.json"

    payload = json.loads(path.read_text())
    meta = payload.pop(LAYER_CONFIG_META_KEY)
    assert payload == ASSIGNMENT
    assert identity_sha256(payload) == digest
    assert meta["schema"] == prefill_frontier.ASSIGNMENT_SCHEMA
    assert meta["assignment_sha256"] == digest
    assert meta["research_only"] is True
    assert meta["table_id"] == PROVENANCE["table_id"]
    assert meta["table_sha256"] == PROVENANCE["table_sha256"]

    # And it re-reads as itself: the point beside it is what a reader gets.
    again = _point(tmp_path)
    assert again["assignment_sha256"] == digest
    assert again["predicted_dloss"] == point["predicted_dloss"]


def test_a_link_race_loser_adopts_a_winners_identical_bytes(tmp_path, monkeypatch):
    """No-clobber publication: the loser validates what the winner wrote."""
    digest = _digest(ASSIGNMENT)
    path = tmp_path / f"{digest}.json"
    written: dict = {}

    def winner(publish_path, encoded):
        written["bytes"] = encoded
        publish_path.write_bytes(encoded)
        return False                      # "somebody else created the path"

    monkeypatch.setattr(prefill_frontier, "publish_new_bytes", winner)
    point = _point(tmp_path)
    assert point["assignment_sha256"] == digest
    assert written["bytes"], "the losing publish must have carried the payload"
    assert path.read_bytes() == written["bytes"]
    assert json.loads(path.read_text())[LAYER_CONFIG_META_KEY]["table_sha256"] \
        == PROVENANCE["table_sha256"]


def test_a_link_race_loser_refuses_a_winners_different_bytes(tmp_path, monkeypatch):
    """The race-winner rule is verify-or-refuse, never replace-or-trust."""
    digest = _digest(ASSIGNMENT)
    path = tmp_path / f"{digest}.json"
    winner_bytes = _encoded(_payload(OTHER_ASSIGNMENT))

    def winner(publish_path, encoded):
        publish_path.write_bytes(winner_bytes)
        return False

    monkeypatch.setattr(prefill_frontier, "publish_new_bytes", winner)
    with pytest.raises(prefill_frontier.PrefillFrontierError):
        _point(tmp_path)
    assert path.read_bytes() == winner_bytes, "the loser must not replace the winner"


def test_publish_new_bytes_creates_once_and_never_clobbers(tmp_path):
    """The shared primitive: create-or-report, with the existing bytes left alone."""
    path = tmp_path / "artifact.json"
    assert prefill_frontier.publish_new_bytes(path, b"first\n") is True
    assert path.read_bytes() == b"first\n"
    assert prefill_frontier.publish_new_bytes(path, b"second\n") is False
    assert path.read_bytes() == b"first\n"
