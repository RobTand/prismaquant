"""Referenced adjoint checkpoints (PQ #1036).

A referenced (v2, and since PQ #1037 packed v3) checkpoint's cotangent rows
are the Stage A owner's own committed entries, not copies. The checkpoint directory holds only the shared states
and the manifest. These tests use the real owner and writer on tiny CPU
tensors. They check that the writer copies nothing, that a pinned entry
outlives its rollover and the owner's close, that each identity refusal
fires, and that the loaded plane matches the one a copied (v1) checkpoint
of the same tensors loads.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from prismaquant.cost_streaming import StreamedBoundaryArtifacts
from prismaquant.joint_adjoint_checkpoints import (
    ADJOINT_CHECKPOINT_PACKED_SCHEMA,
    ADJOINT_CHECKPOINT_SCHEMA,
    adjoint_space,
    checkpoint_cotangent_plane,
    checkpoint_is_referenced,
    load_adjoint_checkpoint,
    write_adjoint_checkpoint,
)
from test_streamed_boundary_artifacts import _policy

BOUNDARY = 5


def _owner(space, *, published=False):
    owner = StreamedBoundaryArtifacts(_policy(
        Path(space) / "exact-boundaries", cap=1 << 22, aux=1 << 20, disk=1 << 24))
    owner.bind({"fixture": "referenced-checkpoint"}, n_probes=2, published=published)
    return owner


def _session(owner):
    return {"generation": owner.session["generation"], "kind": "adjoint_checkpoint",
            "run_identity_sha256": owner.session["run_identity_sha256"]}


def _tensors():
    return {(0, 0): torch.arange(16, dtype=torch.float32).reshape(4, 4),
            (1, 0): torch.full((2, 2), 3.0)}


def _shared():
    return ({(0, 0): {"w": torch.zeros(2)}, (1, 0): {"w": torch.ones(3)}},
            {0: {"tag": "a"}})


def _referenced(tmp_path, *, published=False):
    space = adjoint_space(tmp_path / "out")
    owner = _owner(space, published=published)
    references = {key: owner.write(tensor, batch_index=key[1], boundary_index=BOUNDARY,
                                   probe_index=key[0])
                  for key, tensor in _tensors().items()}
    shared_adjoint, shared_pass = _shared()
    record = write_adjoint_checkpoint(
        space, boundary=BOUNDARY, session=_session(owner), cotangents=references,
        shared_adjoint=shared_adjoint, shared_pass=shared_pass, owner=owner,
        referenced=True)
    return space, owner, references, record


def test_referenced_checkpoint_copies_no_activation(tmp_path):
    space, owner, references, record = _referenced(tmp_path)
    assert record["schema"] == ADJOINT_CHECKPOINT_PACKED_SCHEMA
    assert checkpoint_is_referenced(record)
    entries = Path(space) / "checkpoints" / f"boundary-{BOUNDARY:03d}" / "entries"
    assert sorted(path.name.split(".")[0] for path in entries.iterdir()) == sorted(
        row["name"] for row in record["shared_state_entries"])
    plane = checkpoint_cotangent_plane(record)
    assert {key: row["path"] for key, row in plane.items()} == {
        key: reference.path for key, reference in references.items()}
    # Only shared states and the manifest are checkpoint bytes.
    manifest = entries.parent / "checkpoint.json"
    assert owner.telemetry["live_checkpoint_bytes"] == (
        sum(row["file_bytes"] for row in record["shared_state_entries"])
        + manifest.stat().st_size)
    cotangents, shared_adjoint, shared_pass = load_adjoint_checkpoint(space, record)
    for key, tensor in _tensors().items():
        assert torch.equal(cotangents[key], tensor)
    assert set(shared_adjoint) == {(0, 0), (1, 0)} and shared_pass[0] == {"tag": "a"}


def test_pinned_entry_outlives_rollover_and_close(tmp_path):
    space, owner, references, record = _referenced(tmp_path)
    before = owner.telemetry["live_checkpoint_bytes"]
    for key, reference in references.items():
        owner.write(torch.zeros(1), batch_index=key[1], boundary_index=BOUNDARY - 1,
                    probe_index=key[0], previous=reference)
    for reference in references.values():
        assert Path(reference.path).is_file()
    assert owner.telemetry["pinned_checkpoint_entries_retired"] == len(references)
    assert owner.telemetry["live_checkpoint_bytes"] == before + sum(
        reference.file_bytes for reference in references.values())
    # An unpublished close retires every live entry, and a pinned one stays.
    owner.__exit__(None, None, None)
    for reference in references.values():
        assert Path(reference.path).is_file()
    cotangents, _, _ = load_adjoint_checkpoint(space, record)
    for key, tensor in _tensors().items():
        assert torch.equal(cotangents[key], tensor)


def test_unpinned_rollover_still_unlinks(tmp_path):
    """Control: without a checkpoint, a rollover disposes its previous entry."""
    owner = _owner(adjoint_space(tmp_path / "out"))
    reference = owner.write(torch.ones(4), batch_index=0, boundary_index=BOUNDARY,
                            probe_index=0)
    owner.write(torch.zeros(4), batch_index=0, boundary_index=BOUNDARY - 1,
                probe_index=0, previous=reference)
    assert not Path(reference.path).exists()
    assert owner.telemetry.get("pinned_checkpoint_entries_retired", 0) == 0


def test_reference_from_another_boundary_refuses(tmp_path):
    space = adjoint_space(tmp_path / "out")
    owner = _owner(space)
    references = {key: owner.write(tensor, batch_index=key[1],
                                   boundary_index=BOUNDARY + 1, probe_index=key[0])
                  for key, tensor in _tensors().items()}
    shared_adjoint, shared_pass = _shared()
    with pytest.raises(RuntimeError):
        write_adjoint_checkpoint(
            space, boundary=BOUNDARY, session=_session(owner), cotangents=references,
            shared_adjoint=shared_adjoint, shared_pass=shared_pass, owner=owner,
            referenced=True)
    assert not (Path(space) / "checkpoints" / f"boundary-{BOUNDARY:03d}"
                / "checkpoint.json").exists()


def test_referenced_checkpoint_needs_its_own_generation(tmp_path):
    space = adjoint_space(tmp_path / "out")
    owner = _owner(space)
    references = {key: owner.write(tensor, batch_index=key[1], boundary_index=BOUNDARY,
                                   probe_index=key[0])
                  for key, tensor in _tensors().items()}
    shared_adjoint, shared_pass = _shared()
    foreign = {**_session(owner), "generation": "ef" * 16}
    with pytest.raises(RuntimeError, match="own owner's generation"):
        write_adjoint_checkpoint(
            space, boundary=BOUNDARY, session=foreign, cotangents=references,
            shared_adjoint=shared_adjoint, shared_pass=shared_pass, owner=owner,
            referenced=True)


def _mutations(record):
    key = sorted(checkpoint_cotangent_plane(record))[0]
    index = next(i for i, row in enumerate(record["activation_entries"])
                 if row["path"] == checkpoint_cotangent_plane(record)[key]["path"])

    def edit(change):
        mutated = copy.deepcopy(record)
        change(mutated["activation_entries"][index], mutated)
        return mutated

    def other_boundary_name(row, _):
        row["name"] = row["name"].replace(f"-at-{BOUNDARY}", f"-at-{BOUNDARY - 1}")

    def other_directory(row, _):
        path = Path(row["path"])
        row["path"] = str(path.parents[2] / ("ef" * 16) / "entries" / path.name)

    def other_generation(row, _):
        row["metadata"]["identity"]["session"]["generation"] = "ef" * 16

    def other_coordinate(row, _):
        row["metadata"]["identity"]["coordinates"]["boundary"] = BOUNDARY - 1

    def copied_kind(row, _):
        row["metadata"]["identity"]["kind"] = "adjoint_checkpoint_cotangent"

    def duplicate(_row, mutated):
        mutated["activation_entries"].append(copy.deepcopy(mutated["activation_entries"][0]))

    return {name: edit(change) for name, change in (
        ("other_boundary_name", other_boundary_name),
        ("other_directory", other_directory),
        ("other_generation", other_generation),
        ("other_coordinate", other_coordinate),
        ("copied_kind", copied_kind),
        ("duplicate", duplicate))}


def test_referenced_plane_refuses_each_identity_drift(tmp_path):
    _, _, _, record = _referenced(tmp_path)
    checkpoint_cotangent_plane(record)
    for name, mutated in _mutations(record).items():
        with pytest.raises(ValueError):
            checkpoint_cotangent_plane(mutated)
            pytest.fail(f"{name} was accepted")


def test_referenced_and_copied_planes_hold_the_same_tensors(tmp_path):
    """The band-slice difference is identity and place, never the payload."""
    space, owner, references, referenced = _referenced(tmp_path / "v2")
    copied_space = adjoint_space(tmp_path / "v1" / "out")
    copied_owner = _owner(copied_space)
    shared_adjoint, shared_pass = _shared()
    copied = write_adjoint_checkpoint(
        copied_space, boundary=BOUNDARY, session=_session(copied_owner),
        cotangents=_tensors(), shared_adjoint=shared_adjoint, shared_pass=shared_pass,
        owner=copied_owner)
    assert copied["schema"] == ADJOINT_CHECKPOINT_SCHEMA
    v1, v2 = checkpoint_cotangent_plane(copied), checkpoint_cotangent_plane(referenced)
    assert set(v1) == set(v2)
    for key in v1:
        for field in ("shape", "dtype", "tensor_bytes"):
            assert v1[key][field] == v2[key][field], (key, field)
        assert v1[key]["name"] == f"cotangent-{key[0]}-{key[1]}"
        assert v2[key]["name"] == f"cotangent-{key[0]}-{key[1]}-at-{BOUNDARY}"
        assert v1[key]["metadata"]["identity"]["kind"] == "adjoint_checkpoint_cotangent"
        assert v2[key]["metadata"]["identity"]["kind"] == "cotangent"
    loaded_v1, _, _ = load_adjoint_checkpoint(copied_space, copied)
    loaded_v2, _, _ = load_adjoint_checkpoint(space, referenced)
    for key in v1:
        assert torch.equal(loaded_v1[key], loaded_v2[key])


def test_a_referenced_tee_names_each_entry_as_the_roll_hands_it_over(tmp_path):
    """The Stage A path: open before the pass, reference in roll order, seal."""
    from prismaquant.joint_adjoint_checkpoints import open_adjoint_checkpoint
    from prismaquant.joint_adjoint_checkpoints import exact_entry_record

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
    references = {}
    for key in sorted(tensors, reverse=True):
        references[key] = owner.write(tensors[key], batch_index=key[1],
                                      boundary_index=BOUNDARY, probe_index=key[0])
        with pytest.raises(RuntimeError, match="writes no cotangent copy"):
            attempt.write_activation(*key, tensors[key])
        attempt.reference_activation(*key, references[key])
        with pytest.raises(RuntimeError, match="already names"):
            attempt.reference_activation(*key, references[key])
    other = owner.write(torch.zeros(1), batch_index=1, boundary_index=BOUNDARY,
                        probe_index=0)
    with pytest.raises(RuntimeError, match="planned no cotangent at probe 0, batch 1"):
        attempt.reference_activation(0, 1, other)
    attempt.write_shared_states(shared_adjoint, shared_pass)
    record = attempt.seal()
    assert record["schema"] == ADJOINT_CHECKPOINT_PACKED_SCHEMA
    assert sorted(record["activation_entries"], key=lambda row: row["name"]) == sorted(
        (exact_entry_record(reference) for reference in references.values()),
        key=lambda row: row["name"])
    # Once committed, the entries are pinned: the roll past them keeps the files.
    for key, reference in references.items():
        owner.write(torch.zeros(1), batch_index=key[1], boundary_index=BOUNDARY - 1,
                    probe_index=key[0], previous=reference)
        assert Path(reference.path).is_file()
    loaded, _, _ = load_adjoint_checkpoint(space, record)
    for key, tensor in tensors.items():
        assert torch.equal(loaded[key], tensor)


def test_a_catalog_extension_reads_a_referenced_checkpoint_in_place(tmp_path):
    """Stage B's original-namespace check accepts the owner's plane, nothing else."""
    from prismaquant.joint_cost_quantum import quantum_adjoint_space

    space, _owner_, _references, record = _referenced(tmp_path)
    adjoint_slice = {"boundary_storage": {"directory": str(Path(space) / "exact-boundaries")},
                     "checkpoint": copy.deepcopy(record)}
    extension = {"catalog_extension": {"path": "/proof"}}
    assert quantum_adjoint_space(extension, adjoint_slice, "/new") == Path(space)
    row = adjoint_slice["checkpoint"]["activation_entries"][0]
    row["path"] = str(Path(space) / "checkpoints" / f"boundary-{BOUNDARY:03d}" / "entries"
                      / Path(row["path"]).name)
    with pytest.raises(RuntimeError, match="refused"):
        quantum_adjoint_space(extension, adjoint_slice, "/new")
