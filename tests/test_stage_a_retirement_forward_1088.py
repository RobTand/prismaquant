"""Actual Stage A forward records and PB origin accounting (PQ #1088).

The controlled producer's ordinary close may remove forward files. Preserve
its actual serialized bytes before that close, then restore them at the same
recorded paths to model an acknowledged PB batch whose origin remains held.
Only that committed batch is a new retirement target; uncommitted copies are
controls. No production queue or cleanup is used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from prismaquant import stage_a_chain_resume as resume_mod
from prismaquant import stage_a_retirement as retire
from prismaquant.cost_streaming import StreamedBoundaryArtifacts
from prismaquant.stage_a_chain_resume import chain_state_path
from test_stage_a_retirement_1073 import (
    argv, completed, last_json, retire_main, successor,
)
from test_stage_a_retirement_pb_1073 import (
    _commit, _isolated_launch_context, _name_producer, _offline_tier_policy,
    _owner, _pb, _template,
)

pytestmark = pytest.mark.own_process


def forward_case(tmp_path, monkeypatch, *, mixed=False):
    pb_repo = _pb(monkeypatch)
    root = tmp_path / "run"
    saved = {}
    original = StreamedBoundaryArtifacts.write

    def preserve_forward(self, tensor, **kwargs):
        reference = original(self, tensor, **kwargs)
        if kwargs.get("probe_index") is None:
            saved[reference.path] = Path(reference.path).read_bytes()
        return reference

    with monkeypatch.context() as patch:
        patch.setattr(StreamedBoundaryArtifacts, "write", preserve_forward)
        space = completed(root, monkeypatch)
    state = json.loads(chain_state_path(space).read_text())
    records = [record for rows in state["boundary_entries"].values()
               for record in rows]
    assert records and saved
    assert all(record["metadata"]["identity"]["kind"] == "boundary"
               for record in records)
    sealed_paths = {record["path"] for record in records}
    # Tail capture also writes a boundary that is not a persisted chain
    # boundary. Preserve it as an unrelated, uncommitted control.
    assert sealed_paths <= set(saved)
    for path, payload in saved.items():
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(payload)
    selected = sorted(sealed_paths)[:2]
    if mixed:
        selected[1] = sorted(set(saved) - sealed_paths)[0]
    untouched = sorted(set(saved) - set(selected))
    assert len(selected) == 2 and untouched
    _queue, publication = _owner(tmp_path, pb_repo, _template(Path(selected[0]).parent))
    charge = _commit(publication, "forward-retained", selected)
    _name_producer(space, resume_mod.producer_binding(publication))
    monkeypatch.setattr(resume_mod, "require_producer_contained", lambda producer: None)
    bindings = tmp_path / "bindings"
    bindings.mkdir()
    return root, space, successor(tmp_path, monkeypatch), bindings, publication, selected, untouched, charge


def test_unbound_forward_origin_batch_is_reclaimed_exactly(tmp_path, monkeypatch, capsys):
    root, space, succ, bindings, publication, selected, untouched, charge = forward_case(
        tmp_path, monkeypatch)
    assert publication.durable_charge()["payload"] == charge
    assert retire_main(argv(root, succ, [bindings])) == 0
    report = last_json(capsys)
    assert report["batches"] == {"forward-retained": "reclaimed"}
    assert report["durable_charge_before"] - report["durable_charge_after"] == charge
    assert publication.durable_charge()["payload"] == 0
    assert not any(os.path.exists(path) for path in selected)
    assert all(os.path.exists(path) for path in untouched)
    assert retire.retirement_record_path(space).is_file()


@pytest.mark.parametrize("name", ["seed.json", "forward-capsule.json"])
def test_forward_path_binding_refuses_before_any_unlink(tmp_path, monkeypatch, capsys, name):
    root, space, succ, bindings, publication, selected, _untouched, charge = forward_case(
        tmp_path, monkeypatch)
    # Retirement checks every declared JSON path, not the validity of the
    # consumer's unrelated schema. This fixture asserts the no-live-binding
    # proof only; it does not claim seed/capsule execution qualification.
    (bindings / name).write_text(json.dumps({"forward_inputs": [{"path": selected[0]}]}))
    assert retire_main(argv(root, succ, [bindings])) == 3
    error = capsys.readouterr().err
    assert name in error and "a live binding holds the run" in error
    assert not retire.retirement_record_path(space).exists()
    assert all(os.path.exists(path) for path in selected)
    assert publication.durable_charge()["payload"] == charge


def rewrite_forward_records(space, mutate):
    path = chain_state_path(space)
    document = json.loads(path.read_text())
    for entries in document["boundary_entries"].values():
        for entry in entries:
            mutate(entry)
    path.write_text(json.dumps(resume_mod._seal(document), sort_keys=True) + "\n")


def test_foreign_forward_records_do_not_admit_unlink(tmp_path, monkeypatch, capsys):
    root, space, succ, bindings, publication, selected, _untouched, charge = forward_case(
        tmp_path, monkeypatch)

    def foreign(entry):
        entry["metadata"]["identity"]["session"] = {
            **entry["metadata"]["identity"]["session"], "generation": "foreign-generation"}

    # Mutate the sealed record to exercise ownership policy only, not a
    # claim that an imported execution has been qualified.
    rewrite_forward_records(space, foreign)
    assert retire_main(argv(root, succ, [bindings])) == 0
    assert last_json(capsys)["batches"] == {}
    assert publication.durable_charge()["payload"] == charge
    assert all(os.path.exists(path) for path in selected)


def test_preexisting_retirement_record_never_adds_forward_targets(tmp_path, monkeypatch, capsys):
    root, space, succ, bindings, publication, selected, _untouched, charge = forward_case(
        tmp_path, monkeypatch)
    # Produce the old target set through the same real retirement writer,
    # rather than manually constructing an alternate sealed-record recipe.
    with monkeypatch.context() as patch:
        patch.setattr(retire, "_owned_forward_paths", lambda document, **kwargs: set())
        assert retire_main(argv(root, succ, [bindings])) == 0
    capsys.readouterr()
    before = retire.retirement_record_path(space).read_bytes()
    assert not set(selected).intersection(json.loads(before)["entry_paths"])
    assert retire_main(argv(root, succ, [bindings])) == 0
    assert last_json(capsys)["batches"] == {}
    assert retire.retirement_record_path(space).read_bytes() == before
    assert publication.durable_charge()["payload"] == charge
    assert all(os.path.exists(path) for path in selected)


@pytest.mark.parametrize("field", ["escaping-path", "wrong-kind", "missing-session"])
def test_malformed_owned_forward_refuses_before_unlink(tmp_path, monkeypatch, capsys, field):
    root, space, succ, bindings, publication, selected, _untouched, charge = forward_case(
        tmp_path, monkeypatch)

    def malformed(entry):
        if field == "escaping-path":
            entry["path"] = str(tmp_path / "unrelated.pt")
        elif field == "wrong-kind":
            entry["metadata"]["identity"]["kind"] = "cotangent"
        else:
            entry["metadata"]["identity"]["session"] = None

    rewrite_forward_records(space, malformed)
    assert retire_main(argv(root, succ, [bindings])) == 3
    assert "forward entry" in capsys.readouterr().err
    assert not retire.retirement_record_path(space).exists()
    assert publication.durable_charge()["payload"] == charge
    assert all(os.path.exists(path) for path in selected)


def test_changed_forward_origin_refuses_before_unlink(tmp_path, monkeypatch, capsys):
    root, space, succ, bindings, publication, selected, _untouched, charge = forward_case(
        tmp_path, monkeypatch)
    first = Path(selected[0])
    first.write_bytes(first.read_bytes() + b"changed")
    assert retire_main(argv(root, succ, [bindings])) == 3
    assert "is not the file its commit recorded" in capsys.readouterr().err
    assert not retire.retirement_record_path(space).exists()
    assert publication.durable_charge()["payload"] == charge
    assert all(os.path.exists(path) for path in selected)


def test_mixed_forward_and_unsealed_tail_batch_refuses(tmp_path, monkeypatch, capsys):
    root, space, succ, bindings, publication, selected, _untouched, charge = forward_case(
        tmp_path, monkeypatch, mixed=True)
    assert retire_main(argv(root, succ, [bindings])) == 3
    assert "batch forward-retained mixes checkpoint entries" in capsys.readouterr().err
    assert not retire.retirement_record_path(space).exists()
    assert publication.durable_charge()["payload"] == charge
    assert all(os.path.exists(path) for path in selected)


def test_interrupted_forward_unlink_reuses_sealed_targets(tmp_path, monkeypatch, capsys):
    root, space, succ, bindings, publication, selected, untouched, charge = forward_case(
        tmp_path, monkeypatch)
    original = os.unlink
    calls = []

    def interrupted(path, *args, **kwargs):
        if str(path) in selected:
            calls.append(str(path))
            if len(calls) == 2:
                raise OSError("controlled forward retirement interruption")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(retire.os, "unlink", interrupted)
        assert retire_main(argv(root, succ, [bindings])) == 3
    capsys.readouterr()
    assert retire.retirement_record_path(space).is_file()
    assert sum(os.path.exists(path) for path in selected) == 1
    assert retire_main(argv(root, succ, [bindings])) == 0
    report = last_json(capsys)
    assert report["batches"] == {"forward-retained": "reclaimed"}
    assert publication.durable_charge()["payload"] == 0
    assert not any(os.path.exists(path) for path in selected)
    assert all(os.path.exists(path) for path in untouched)
