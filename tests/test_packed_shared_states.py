"""Packed shared states (PQ #1037).

A v3 checkpoint writes its shared-adjoint and shared-pass states as one
sealed pack file instead of one pickle per state. These tests write the same
tensors and states as v2 (one pickle each) and v3 (one pack) with the real
owner and writer, and check that:

- both load identically, through every loader;
- the records differ only in ``schema``, ``shared_state_entries`` and the
  seal ``cotangent_sha256``;
- the pack's structure is checked member by member, with the whole-file
  digest made valid so each check is the one that refuses;
- the band reader and the band-serial handoff read the pack.
"""
from __future__ import annotations

import copy
import hashlib
import json
import struct
import uuid
from pathlib import Path

import pytest
import torch

from prismaquant.joint_adjoint_band import read_sealed_checkpoint
from prismaquant.joint_adjoint_checkpoints import (
    ADJOINT_CHECKPOINT_PACKED_SCHEMA,
    ADJOINT_CHECKPOINT_REFERENCED_SCHEMA,
    SHARED_STATE_PACK_SCHEMA,
    adjoint_space,
    checkpoint_is_packed,
    checkpoint_is_referenced,
    checkpoint_manifest_entry,
    load_adjoint_checkpoint,
    load_checkpoint_shared_states,
    open_adjoint_checkpoint,
    unpack_shared_states,
    write_adjoint_checkpoint,
)
from prismaquant.joint_quantum_handoff import _shared_pass_entries, handoff_read_entries
from test_referenced_adjoint_checkpoint import BOUNDARY, _owner, _session, _tensors


def _shared():
    shared_adjoint = {(p, b): {"w": torch.full((3,), float(10 * p + b)), "n": p + b}
                      for p in range(2) for b in range(12)}
    shared_pass = {b: ({"mask": [b, b + 1]} if b % 2 else None) for b in range(12)}
    return shared_adjoint, shared_pass


def _write(root, monkeypatch, *, packed):
    """The same plane and states, sealed under a fixed generation."""
    space = adjoint_space(root / "out")
    with monkeypatch.context() as patch:
        patch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=0x1037))
        owner = _owner(space)
    references = {key: owner.write(tensor, batch_index=key[1], boundary_index=BOUNDARY,
                                   probe_index=key[0])
                  for key, tensor in _tensors().items()}
    shared_adjoint, shared_pass = _shared()
    record = write_adjoint_checkpoint(
        space, boundary=BOUNDARY, session=_session(owner), cotangents=references,
        shared_adjoint=shared_adjoint, shared_pass=shared_pass, owner=owner,
        referenced=True, packed=packed)
    return space, owner, record


def _equal_states(left, right):
    assert set(left) == set(right)
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, dict) and "w" in a:
            assert torch.equal(a["w"], b["w"]) and a["n"] == b["n"], key
        else:
            assert a == b, key


def test_packed_and_unpacked_checkpoints_load_identically(tmp_path, monkeypatch):
    v2_space, _, v2 = _write(tmp_path / "v2", monkeypatch, packed=False)
    v3_space, _, v3 = _write(tmp_path / "v3", monkeypatch, packed=None)
    assert v2["schema"] == ADJOINT_CHECKPOINT_REFERENCED_SCHEMA
    assert v3["schema"] == ADJOINT_CHECKPOINT_PACKED_SCHEMA
    assert checkpoint_is_referenced(v3) and checkpoint_is_packed(v3)
    assert not checkpoint_is_packed(v2)
    assert len(v2["shared_state_entries"]) == 36
    (pack,) = v3["shared_state_entries"]
    assert pack["name"] == "shared-states"
    assert Path(pack["path"]).name == "shared-states.pack"
    entries = Path(pack["path"]).parent
    assert sorted(path.name for path in entries.iterdir()) == ["shared-states.pack"]

    loaded_v2 = load_adjoint_checkpoint(v2_space, v2)
    loaded_v3 = load_adjoint_checkpoint(v3_space, v3)
    for key in _tensors():
        assert torch.equal(loaded_v2[0][key], loaded_v3[0][key])
    _equal_states(loaded_v2[1], loaded_v3[1])
    _equal_states(loaded_v2[2], loaded_v3[2])
    _equal_states(loaded_v2[1], _shared()[0])
    _equal_states(loaded_v2[2], _shared()[1])
    borrowed_v2 = load_checkpoint_shared_states(v2_space, v2)
    borrowed_v3 = load_checkpoint_shared_states(v3_space, v3, shared_state_max_bytes=1 << 20)
    _equal_states(borrowed_v2[0], borrowed_v3[0])
    _equal_states(borrowed_v2[1], borrowed_v3[1])

    # One member per v2 pickle. A tensor's pickle carries its storage's
    # address, so only the tensor-free shared-pass members are byte-equal.
    members = dict(unpack_shared_states(Path(pack["path"]).read_bytes()))
    assert sorted(members) == sorted(row["name"] for row in v2["shared_state_entries"])
    for row in v2["shared_state_entries"]:
        if row["name"].startswith("shared-pass-"):
            assert bytes(members[row["name"]]) == Path(row["path"]).read_bytes()


def test_the_record_changes_only_in_named_fields(tmp_path, monkeypatch):
    """The band slice carries this record verbatim; nothing else moves."""
    _, _, v2 = _write(tmp_path / "v2", monkeypatch, packed=False)
    _, _, v3 = _write(tmp_path / "v3", monkeypatch, packed=None)
    normal_v2 = json.loads(json.dumps(v2).replace(str(tmp_path / "v2"), "ROOT"))
    normal_v3 = json.loads(json.dumps(v3).replace(str(tmp_path / "v3"), "ROOT"))
    assert set(normal_v2) == set(normal_v3)
    changed = {key for key in normal_v2 if normal_v2[key] != normal_v3[key]}
    assert changed == {"schema", "shared_state_entries", "cotangent_sha256"}
    assert normal_v2["activation_entries"] == normal_v3["activation_entries"]


def test_the_owner_counts_the_pack_as_one_checkpoint_file(tmp_path, monkeypatch):
    space, owner, record = _write(tmp_path, monkeypatch, packed=None)
    (pack,) = record["shared_state_entries"]
    manifest = Path(checkpoint_manifest_entry(record)["path"])
    assert pack["file_bytes"] == Path(pack["path"]).stat().st_size
    assert owner.telemetry["live_checkpoint_bytes"] == (
        pack["file_bytes"] + manifest.stat().st_size)
    sealed, binding = read_sealed_checkpoint(space, BOUNDARY)
    assert sealed == record
    assert binding["path"] == str(manifest)


def test_the_stage_a_tee_seals_a_pack(tmp_path):
    """Stage A's path: reserve before the pass, admit the pack after it."""
    space = adjoint_space(tmp_path / "out")
    owner = _owner(space)
    tensors = _tensors()
    shared_adjoint, shared_pass = _shared()
    attempt = open_adjoint_checkpoint(
        space, boundary=BOUNDARY, session=_session(owner),
        specs={key: (t.numel() * t.element_size(), list(t.shape), str(t.dtype))
               for key, t in tensors.items()},
        shared_adjoint_keys=list(shared_adjoint), shared_pass_keys=list(shared_pass),
        owner=owner, referenced=True)
    for key in sorted(tensors):
        attempt.reference_activation(*key, owner.write(
            tensors[key], batch_index=key[1], boundary_index=BOUNDARY, probe_index=key[0]))
    with pytest.raises(RuntimeError, match="differ from the set it planned"):
        attempt.write_shared_states({(0, 0): {}}, shared_pass)
    attempt.write_shared_states(shared_adjoint, shared_pass)
    reservation = owner._checkpoint_reservations[attempt.reservation]
    shared = [row for row in reservation["files"] if row["name"].startswith("shared-")]
    assert [row["name"] for row in shared] == ["shared-states"]
    record = attempt.seal()
    assert record["schema"] == ADJOINT_CHECKPOINT_PACKED_SCHEMA
    (pack,) = record["shared_state_entries"]
    assert pack["file_bytes"] <= shared[0]["envelope_bytes"]
    _, loaded_adjoint, loaded_pass = load_adjoint_checkpoint(space, record)
    _equal_states(loaded_adjoint, shared_adjoint)
    _equal_states(loaded_pass, shared_pass)


def test_an_empty_shared_set_still_seals_one_pack(tmp_path):
    space = adjoint_space(tmp_path / "out")
    owner = _owner(space)
    references = {key: owner.write(tensor, batch_index=key[1], boundary_index=BOUNDARY,
                                   probe_index=key[0])
                  for key, tensor in _tensors().items()}
    record = write_adjoint_checkpoint(
        space, boundary=BOUNDARY, session=_session(owner), cotangents=references,
        shared_adjoint={}, shared_pass={}, owner=owner, referenced=True)
    (pack,) = record["shared_state_entries"]
    assert unpack_shared_states(Path(pack["path"]).read_bytes()) == []
    assert load_adjoint_checkpoint(space, record)[1:] == ({}, {})


def test_a_packed_copied_checkpoint_refuses(tmp_path):
    space = adjoint_space(tmp_path / "out")
    owner = _owner(space)
    with pytest.raises(RuntimeError, match="is a referenced checkpoint"):
        write_adjoint_checkpoint(
            space, boundary=BOUNDARY, session=_session(owner), cotangents=_tensors(),
            shared_adjoint={}, shared_pass={}, owner=owner, packed=True)


# --------------------------------------------------------------------------
# Pack structure: each forged pack is internally consistent except for the
# one fault, and re-sealed so its whole-file digest matches its row. The
# whole-file check therefore passes and the structural check under test is
# the one that refuses.
# --------------------------------------------------------------------------

_MAGIC = b"PQSSPK01"


def _parts(payload):
    size = len(payload)
    (index_bytes,) = struct.unpack("<Q", payload[size - 16:size - 8])
    body_end = size - 16 - index_bytes
    return payload[:body_end], json.loads(payload[body_end:size - 16])


def _pack(body, index, *, canonical=True, magic=_MAGIC):
    raw = (json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
           if canonical else json.dumps(index, indent=1).encode())
    return body + raw + struct.pack("<Q", len(raw)) + magic


def _forgeries(payload):
    body, index = _parts(payload)
    members = index["members"]
    first, second = members[0], members[1]

    def edited(change):
        forged = copy.deepcopy(index)
        change(forged["members"])
        return _pack(body, forged)

    def swap(rows):
        rows[0], rows[1] = rows[1], rows[0]

    def gap(rows):
        rows[1]["offset"] += 1
        rows[1]["bytes"] -= 1

    def overlap(rows):
        rows[1]["offset"] -= 1
        rows[1]["bytes"] += 1
        rows[1]["sha256"] = hashlib.sha256(
            body[rows[1]["offset"]:rows[1]["offset"] + rows[1]["bytes"]]).hexdigest()

    def wrong_digest(rows):
        rows[0]["sha256"] = "0" * 64

    def bad_name(rows):
        rows[0]["name"] = "shared-pass-01"

    def short(rows):
        rows.pop()

    def extra_field(rows):
        rows[0]["note"] = 1

    def bool_offset(rows):
        rows[0]["offset"] = False

    assert first["offset"] == 0 and second["offset"] == first["bytes"]
    return {
        "unsorted": (edited(swap), "does not start|strictly ascending"),
        "gap": (edited(gap), "does not start|changed"),
        "overlap": (edited(overlap), "does not start"),
        "member_digest": (edited(wrong_digest), "member changed"),
        "member_name": (edited(bad_name), "not a checkpoint shared-state name"),
        "short_index": (edited(short), "do not reach its index"),
        "extra_field": (edited(extra_field), "malformed"),
        "bool_offset": (edited(bool_offset), "malformed"),
        "wrong_schema": (_pack(body, {**index, "schema": "other"}), "is not a"),
        "not_canonical": (_pack(body, index, canonical=False), "not canonical"),
        "no_magic": (_pack(body, index, magic=b"XXXXXXXX"), "no trailer"),
        "index_overrun": (payload[:-16] + struct.pack("<Q", len(payload)) + _MAGIC,
                          "overruns the file"),
        "truncated": (payload[-8:], "no trailer"),
        "trailing_member_bytes": (_pack(body + b"x", index), "do not reach its index"),
    }


def test_every_pack_structure_fault_refuses(tmp_path, monkeypatch):
    space, _, record = _write(tmp_path, monkeypatch, packed=None)
    (row,) = record["shared_state_entries"]
    payload = Path(row["path"]).read_bytes()
    assert [name for name, _ in unpack_shared_states(payload)] == sorted(
        [f"shared-adjoint-{p}-{b}" for p in range(2) for b in range(12)]
        + [f"shared-pass-{b}" for b in range(12)])
    _, index = _parts(payload)
    assert index["schema"] == SHARED_STATE_PACK_SCHEMA
    for fault, (forged, match) in _forgeries(payload).items():
        with pytest.raises(RuntimeError, match=match):
            unpack_shared_states(forged)
            pytest.fail(f"{fault} was accepted")


def test_a_forged_pack_with_a_valid_row_refuses_at_load(tmp_path, monkeypatch):
    """End to end: the manifest, row and file agree; the pack's index does not."""
    space, _, record = _write(tmp_path, monkeypatch, packed=None)
    (row,) = record["shared_state_entries"]
    path = Path(row["path"])
    forged, _ = _forgeries(path.read_bytes())["member_digest"]
    path.write_bytes(forged)
    resealed = copy.deepcopy(record)
    resealed["shared_state_entries"][0].update(
        sha256=hashlib.sha256(forged).hexdigest(), file_bytes=len(forged))
    from prismaquant.joint_adjoint_checkpoints import (
        checkpoint_manifest_bytes, checkpoint_seal_sha256)
    resealed["cotangent_sha256"] = checkpoint_seal_sha256(resealed)
    manifest = Path(checkpoint_manifest_entry(resealed)["path"])
    manifest.write_bytes(checkpoint_manifest_bytes(resealed))
    with pytest.raises(RuntimeError, match="member changed"):
        load_checkpoint_shared_states(space, resealed)
    # The unforged record no longer matches its bytes at all.
    with pytest.raises(RuntimeError, match="differs from its receipt"):
        load_checkpoint_shared_states(space, record)


def test_a_packed_record_names_exactly_one_pack(tmp_path, monkeypatch):
    _, _, record = _write(tmp_path, monkeypatch, packed=None)
    (row,) = record["shared_state_entries"]
    for rows in ([], [row, dict(row, name="shared-states-2")],
                 [dict(row, name="shared-pass-0")],
                 [dict(row, path=str(Path(row["path"]).with_name("other.pack")))]):
        mutated = dict(record, shared_state_entries=rows)
        with pytest.raises(ValueError, match="exactly one shared-state pack"):
            checkpoint_manifest_entry(mutated)


def test_band_serial_stages_the_pack_whole(tmp_path, monkeypatch):
    """The handoff-load phase reads the pack row; v2 still reads per file."""
    _, _, v2 = _write(tmp_path / "v2", monkeypatch, packed=False)
    _, _, v3 = _write(tmp_path / "v3", monkeypatch, packed=None)
    assert _shared_pass_entries(v3) == v3["shared_state_entries"]
    assert [row["name"] for row in _shared_pass_entries(v2)] == sorted(
        f"shared-pass-{b}" for b in range(12))
    handoff = {"activation_entries": [], "owner_states": {
        "path": "/owner-states.pkl", "file_bytes": 1, "sha256": "0" * 64}}
    (pack,) = v3["shared_state_entries"]
    assert handoff_read_entries(handoff, v3)[-1] == {
        "path": pack["path"], "offset": 0, "bytes": pack["file_bytes"],
        "sha256": pack["sha256"]}
