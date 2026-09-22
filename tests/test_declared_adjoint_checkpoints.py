"""Declared adjoint checkpoints: cotangents referenced once, files declared.

``write_adjoint_checkpoint(declared=True)`` (Stage A's
``PRISMAQUANT_STAGE_A_DECLARED_CHECKPOINTS=1``) names the chain's own
cotangent entries in the checkpoint record instead of writing a second copy,
pins them so the rolling chain never unlinks them, and writes the
checkpoint's own files through the produced-output spool as one
``checkpoint``-class group. These tests run the actual writer, owner, loader
and spool adapter on tiny CPU tensors; PrismaBuild's export is the
controlled transport double of ``test_produced_output_spool`` or, in the last
test, PrismaBuild's own export action on a private queue.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pytest
import torch

from prismaquant import joint_adjoint_checkpoints as checkpoints
from prismaquant import perturbed_x_cache
from prismaquant.joint_adjoint_checkpoints import (
    ADJOINT_CHECKPOINT_SCHEMA,
    DECLARED_ADJOINT_CHECKPOINT_SCHEMA,
    adjoint_space,
    checkpoint_directory,
    exact_entry_record,
    load_adjoint_checkpoint,
    write_adjoint_checkpoint,
)
from test_checkpoint_artifact_budget import _owner
import test_stage_a_produced_boundary_chain as chain
from test_stage_a_produced_boundary_chain import _isolated_launch_context  # noqa: F401

BOUNDARY = 5


def _session(owner):
    return {"generation": owner.session["generation"], "kind": "adjoint_checkpoint",
            "run_identity_sha256": owner.session["run_identity_sha256"]}


def _tensor(probe, batch, *, shift=0.0):
    return torch.arange(64, dtype=torch.float32).reshape(8, 8) * (probe + 1) + batch + shift


def _shared_adjoint(n_probes, n_batches):
    return {(probe, batch): {"w": torch.arange(4, dtype=torch.float32) + probe + batch}
            for probe in range(n_probes) for batch in range(n_batches)}


def _shared_pass(n_batches):
    return {batch: {"tag": f"batch-{batch}", "t": torch.ones(2) * batch}
            for batch in range(n_batches)}


def _plane(owner, *, n_probes=2, n_batches=3, boundary=BOUNDARY):
    return {(probe, batch): owner.write(_tensor(probe, batch), probe_index=probe,
                                        batch_index=batch, boundary_index=boundary)
            for probe in range(n_probes) for batch in range(n_batches)}


def _same(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.contiguous().view(torch.uint8),
                           right.contiguous().view(torch.uint8))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    else:
        assert left == right


def _count_exact_writes(monkeypatch):
    written = []
    original = perturbed_x_cache.write_exact_activation_cache_entry

    def counted(directory, name, tensor, **kwargs):
        written.append(tensor.numel() * tensor.element_size())
        return original(directory, name, tensor, **kwargs)

    monkeypatch.setattr(perturbed_x_cache, "write_exact_activation_cache_entry", counted)
    return written


# -- 3(a): zero cotangent tensor bytes, record rows are the entries' records --

def test_declared_checkpoint_references_entries_and_writes_no_tensor_bytes(
        tmp_path, monkeypatch):
    owner = _owner(tmp_path / "entries", n_probes=2)
    plane = _plane(owner)
    written = _count_exact_writes(monkeypatch)
    monkeypatch.setattr(checkpoints, "write_checkpoint_cotangent_entry",
                        lambda *a, **k: pytest.fail("declared checkpoint copied a cotangent"))
    space = adjoint_space(tmp_path / "out")
    record = write_adjoint_checkpoint(
        space, boundary=BOUNDARY, session=_session(owner), cotangents=plane,
        shared_adjoint=_shared_adjoint(2, 3), shared_pass=_shared_pass(3),
        owner=owner, declared=True)
    assert written == []
    assert record["schema"] == DECLARED_ADJOINT_CHECKPOINT_SCHEMA
    assert record["activation_entries"] == sorted(
        (exact_entry_record(reference) for reference in plane.values()),
        key=lambda row: row["name"])
    directory = checkpoint_directory(space, BOUNDARY)
    assert not list(directory.rglob("*.pt"))
    assert sorted(path.name for path in (directory / "entries").iterdir()) == sorted(
        [f"shared-adjoint-{p}-{b}.pkl" for p in range(2) for b in range(3)]
        + [f"shared-pass-{b}.pkl" for b in range(3)])
    commitment = owner.checkpoint_commitment(record["cotangent_sha256"])
    assert commitment["referenced_entries"] == 6
    assert commitment["referenced_bytes"] == sum(r.file_bytes for r in plane.values())
    assert all(owner.checkpoint_pinned(reference) for reference in plane.values())


# -- 3(b) and 3(d): same tensors as the copy, and the entries outlive the chain --

def test_declared_load_matches_default_copy_and_outlives_the_roll(tmp_path):
    owner = _owner(tmp_path / "entries", n_probes=2, published=False)
    plane = _plane(owner)
    shared_adjoint, shared_pass = _shared_adjoint(2, 3), _shared_pass(3)
    copied = write_adjoint_checkpoint(
        adjoint_space(tmp_path / "copy"), boundary=BOUNDARY, session=_session(owner),
        cotangents=plane, shared_adjoint=shared_adjoint, shared_pass=shared_pass,
        owner=owner)
    assert copied["schema"] == ADJOINT_CHECKPOINT_SCHEMA
    ledgers_before = (owner.telemetry["live_artifact_bytes"]
                      + owner.telemetry["live_checkpoint_bytes"])
    declared_space = adjoint_space(tmp_path / "declared")
    declared = write_adjoint_checkpoint(
        declared_space, boundary=BOUNDARY, session=_session(owner),
        cotangents=plane, shared_adjoint=shared_adjoint, shared_pass=shared_pass,
        owner=owner, declared=True)
    referenced_bytes = sum(reference.file_bytes for reference in plane.values())
    commitment = owner.checkpoint_commitment(declared["cotangent_sha256"])
    # The referenced bytes change ledgers; only the checkpoint's own files add.
    assert (owner.telemetry["live_artifact_bytes"]
            + owner.telemetry["live_checkpoint_bytes"]) == (
        ledgers_before + commitment["actual_bytes"])
    expected = load_adjoint_checkpoint(adjoint_space(tmp_path / "copy"), copied)
    _same(load_adjoint_checkpoint(declared_space, declared), expected)

    # Roll the chain one layer exactly as Stage A does, then retire the
    # new plane (the walk's end) and close an UNPUBLISHED owner, which
    # retires every live entry: the referenced files must all survive.
    rolled = {key: owner.write(_tensor(*key, shift=1.0), probe_index=key[0],
                               batch_index=key[1], boundary_index=BOUNDARY - 1,
                               previous=reference)
              for key, reference in plane.items()}
    for reference in plane.values():
        assert Path(reference.path).is_file()
    for reference in rolled.values():
        owner.retire(reference)
        assert not Path(reference.path).exists()
    owner.__exit__(None, None, None)
    assert all(Path(reference.path).is_file() for reference in plane.values())
    assert owner.telemetry["live_checkpoint_bytes"] >= referenced_bytes
    _same(load_adjoint_checkpoint(declared_space, declared), expected)


def test_without_a_declared_reference_the_roll_still_unlinks(tmp_path):
    """Control: the pin is what keeps the file, not something else."""
    owner = _owner(tmp_path / "entries", n_probes=1)
    reference = owner.write(_tensor(0, 0), probe_index=0, batch_index=0,
                            boundary_index=BOUNDARY)
    write_adjoint_checkpoint(
        adjoint_space(tmp_path / "copy"), boundary=BOUNDARY, session=_session(owner),
        cotangents={(0, 0): reference}, shared_adjoint={}, shared_pass={0: {}},
        owner=owner)
    owner.write(_tensor(0, 0, shift=1.0), probe_index=0, batch_index=0,
                boundary_index=BOUNDARY - 1, previous=reference)
    assert not Path(reference.path).exists()


def test_failed_declared_attempt_pins_nothing(tmp_path, monkeypatch):
    owner = _owner(tmp_path / "entries", n_probes=1)
    reference = owner.write(_tensor(0, 0), probe_index=0, batch_index=0,
                            boundary_index=BOUNDARY)

    def fail(*args, **kwargs):
        raise RuntimeError("serializer failed")

    monkeypatch.setattr(checkpoints, "_write_shared_state_streaming", fail)
    with pytest.raises(RuntimeError, match="serializer failed"):
        write_adjoint_checkpoint(
            adjoint_space(tmp_path / "out"), boundary=BOUNDARY, session=_session(owner),
            cotangents={(0, 0): reference}, shared_adjoint={}, shared_pass={0: {}},
            owner=owner, declared=True)
    assert not owner.checkpoint_pinned(reference)
    assert any(row["state"] == "retained"
               for row in owner._checkpoint_reservations.values())
    owner.write(_tensor(0, 0, shift=1.0), probe_index=0, batch_index=0,
                boundary_index=BOUNDARY - 1, previous=reference)
    assert not Path(reference.path).exists()


# -- loading stays digest-checked and bound to the checkpoint --

def _declared_single(tmp_path):
    owner = _owner(tmp_path / "entries", n_probes=1)
    reference = owner.write(_tensor(0, 0), probe_index=0, batch_index=0,
                            boundary_index=BOUNDARY)
    space = adjoint_space(tmp_path / "out")
    record = write_adjoint_checkpoint(
        space, boundary=BOUNDARY, session=_session(owner),
        cotangents={(0, 0): reference}, shared_adjoint={}, shared_pass={0: {}},
        owner=owner, declared=True)
    return owner, reference, space, record


def test_loader_refuses_changed_referenced_bytes(tmp_path):
    _owner_, reference, space, record = _declared_single(tmp_path)
    data = bytearray(Path(reference.path).read_bytes())
    data[-1] ^= 0xFF
    Path(reference.path).write_bytes(bytes(data))
    with pytest.raises(RuntimeError):
        load_adjoint_checkpoint(space, record)


def _tamper(space, record, edit):
    """Apply ``edit`` to the stored manifest and the record alike, so the
    loader's record-vs-manifest check passes and the binding check bites."""
    manifest = checkpoint_directory(space, BOUNDARY) / "checkpoint.json"
    stored = json.loads(manifest.read_text())
    edit(stored)
    edit(record)
    manifest.write_text(json.dumps(stored, sort_keys=True, indent=2) + "\n")


def test_writer_refuses_a_reference_from_another_boundary(tmp_path):
    owner = _owner(tmp_path / "entries", n_probes=1)
    other = owner.write(_tensor(0, 0), probe_index=0, batch_index=0,
                        boundary_index=BOUNDARY + 1)
    with pytest.raises(RuntimeError, match="only its owner's"):
        write_adjoint_checkpoint(
            adjoint_space(tmp_path / "out"), boundary=BOUNDARY, session=_session(owner),
            cotangents={(0, 0): other}, shared_adjoint={}, shared_pass={0: {}},
            owner=owner, declared=True)
    assert not owner.checkpoint_pinned(other)
    assert not checkpoint_directory(adjoint_space(tmp_path / "out"), BOUNDARY).exists()


def test_loader_refuses_a_reference_bound_to_another_boundary(tmp_path):
    _o, _ref, space, record = _declared_single(tmp_path)

    def edit(document):
        row = document["activation_entries"][0]
        row["metadata"]["identity"]["coordinates"]["boundary"] = BOUNDARY + 1
    _tamper(space, record, edit)
    with pytest.raises(RuntimeError, match="is not the cotangent at boundary"):
        load_adjoint_checkpoint(space, record)


def test_loader_refuses_a_foreign_session(tmp_path):
    _o, _ref, space, record = _declared_single(tmp_path)

    def edit(document):
        document["session"]["generation"] = "ef" * 16
    _tamper(space, record, edit)
    with pytest.raises(RuntimeError):
        load_adjoint_checkpoint(space, record)


def test_quantum_scratch_factory_loads_a_declared_checkpoint(tmp_path, monkeypatch):
    owner = _owner(tmp_path / "entries", n_probes=2)
    plane = _plane(owner)
    space = adjoint_space(tmp_path / "out")
    record = write_adjoint_checkpoint(
        space, boundary=BOUNDARY, session=_session(owner), cotangents=plane,
        shared_adjoint=_shared_adjoint(2, 3), shared_pass=_shared_pass(3),
        owner=owner, declared=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("PRISMAQUANT_STAGE_B_COTANGENT_ROOT", str(scratch))
    monkeypatch.setenv("PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES", str(1 << 20))
    reader = _owner(tmp_path / "reader", n_probes=2)
    loaded, _shared, _pass = load_adjoint_checkpoint(
        space, record, cotangent_factory=reader.checkpoint_cotangent_sink)
    assert sorted(loaded) == sorted(plane)
    for key in plane:
        _same(loaded[key], _tensor(*key))


def test_declared_refuses_without_an_owner(tmp_path):
    with pytest.raises(RuntimeError, match="needs that owner"):
        write_adjoint_checkpoint(
            adjoint_space(tmp_path / "out"), boundary=BOUNDARY,
            session={"generation": "ab" * 16, "run_identity_sha256": "cd" * 32},
            cotangents={(0, 0): _tensor(0, 0)}, shared_adjoint={},
            shared_pass={0: {}}, declared=True)
    assert not (tmp_path / "out").exists()


def test_stage_a_flag_grammar(monkeypatch):
    from prismaquant.joint_cost_stage_a import (
        AdjointIdentityRefused, resolve_declared_checkpoints)
    assert resolve_declared_checkpoints({}) is False
    assert resolve_declared_checkpoints({"PRISMAQUANT_STAGE_A_DECLARED_CHECKPOINTS": "0"}) is False
    assert resolve_declared_checkpoints({"PRISMAQUANT_STAGE_A_DECLARED_CHECKPOINTS": "1"}) is True
    with pytest.raises(AdjointIdentityRefused):
        resolve_declared_checkpoints({"PRISMAQUANT_STAGE_A_DECLARED_CHECKPOINTS": "yes"})


# -- 3(c): the checkpoint's own files are a declared checkpoint-class group --

def _spool_owner(tmp_path):
    from prismaquant.produced_output_spool import ProducedOutputSpool
    from test_produced_output_spool import CapacityDeferred, ControlledExport

    owner, publication, queue, _env, _repo = chain._bound_owner(
        tmp_path, n_batches=chain.GROUP_SIZE, staging_timeout_s=2,
        checkpoint_max_bytes=1 << 20)
    owner._published = True
    backend = ControlledExport(tmp_path / "local")
    owner._local_output_spool = ProducedOutputSpool(
        backend, capacity_deferred=CapacityDeferred, timeout_s=2)
    return owner, publication, queue, backend


def test_spooled_declared_checkpoint_is_a_checkpoint_class_group(tmp_path):
    owner, publication, queue, backend = _spool_owner(tmp_path)
    plane = {(0, batch): owner.write(_tensor(0, batch), probe_index=0,
                                     batch_index=batch, boundary_index=BOUNDARY)
             for batch in range(chain.GROUP_SIZE)}
    space = adjoint_space(tmp_path / "outputs")
    shared_adjoint, shared_pass = _shared_adjoint(1, chain.GROUP_SIZE), _shared_pass(chain.GROUP_SIZE)
    record = write_adjoint_checkpoint(
        space, boundary=BOUNDARY, session=_session(owner), cotangents=plane,
        shared_adjoint=shared_adjoint, shared_pass=shared_pass, owner=owner,
        declared=True)
    directory = checkpoint_directory(space, BOUNDARY)
    # Nothing in the shared root until PrismaBuild's export lands.
    assert not directory.exists()
    batch_id = publication.batch_id_for(kind="checkpoint", boundary_index=BOUNDARY,
                                        group_index=0)
    group = backend.groups[batch_id]
    entries = group["entries"]
    expected = {row["path"] for row in record["shared_state_entries"]} | {
        str(directory / "checkpoint.json")}
    assert {row["destination_path"] for row in entries} == expected
    assert all(row["artifact_class"] == "checkpoint" for row in entries)
    assert all(Path(row["source_path"]).is_relative_to(group["directory"]) for row in entries)
    for row in entries:
        assert hashlib.sha256(Path(row["source_path"]).read_bytes()).hexdigest() == row["sha256"]
    # The prewrite charges the checkpoint class, not payload.
    [prewrite] = [json.loads(path.read_text())
                  for path in queue.root.rglob(f"{batch_id}.prewrite.json")]
    assert prewrite["class_bytes"]["payload"] == 0
    assert prewrite["class_bytes"]["checkpoint"] > 0
    assert {path for path in prewrite["paths"] if not path.endswith(".tmp")} == expected
    # The referenced cotangent entries are the payload group's, unchanged.
    assert record["activation_entries"] == sorted(
        (exact_entry_record(reference) for reference in plane.values()),
        key=lambda row: row["name"])
    for key in list(backend.groups):
        backend.acknowledge(key)
    owner.settle_local_output()
    assert backend.groups[batch_id]["released"]
    assert (directory / "checkpoint.json").is_file()
    loaded, loaded_shared, loaded_pass = load_adjoint_checkpoint(space, record)
    for key in plane:
        _same(loaded[key], _tensor(*key))
    _same(loaded_shared, shared_adjoint)
    _same(loaded_pass, shared_pass)
    assert owner.produced_output_report()["local_spool"]["pending_groups"] == 0


def test_bound_owner_without_a_spool_refuses_declared_checkpoints(tmp_path):
    owner, _publication, _queue, _env, _repo = chain._bound_owner(
        tmp_path, n_batches=chain.GROUP_SIZE, staging_timeout_s=2,
        checkpoint_max_bytes=1 << 20)
    reference = owner.write(_tensor(0, 0), probe_index=0, batch_index=0,
                            boundary_index=BOUNDARY)
    with pytest.raises(RuntimeError, match="need the local output spool"):
        write_adjoint_checkpoint(
            adjoint_space(tmp_path / "outputs"), boundary=BOUNDARY,
            session=_session(owner), cotangents={(0, 0): reference},
            shared_adjoint={}, shared_pass={0: {}}, owner=owner, declared=True)
    assert not (tmp_path / "outputs" / "layer-quanta").exists()


def test_spooled_checkpoint_outside_the_prefix_refuses_before_any_byte(tmp_path):
    owner, _publication, _queue, backend = _spool_owner(tmp_path)
    plane = {(0, batch): owner.write(_tensor(0, batch), probe_index=0,
                                     batch_index=batch, boundary_index=BOUNDARY)
             for batch in range(chain.GROUP_SIZE)}
    groups = set(backend.groups)
    with pytest.raises(RuntimeError, match="outside the bound output prefix"):
        write_adjoint_checkpoint(
            adjoint_space(tmp_path / "elsewhere"), boundary=BOUNDARY,
            session=_session(owner), cotangents=plane, shared_adjoint={},
            shared_pass={0: {}}, owner=owner, declared=True)
    assert set(backend.groups) == groups
    assert owner._checkpoint_active is None
    assert not (tmp_path / "elsewhere").exists()


def test_real_prismabuild_exports_the_checkpoint_group(tmp_path, monkeypatch):
    """PrismaBuild's own export action copies the checkpoint-class group."""
    from fleet_sdk import require_prismabuild_sdk
    require_prismabuild_sdk()
    pin_path = Path(__file__).with_name("stagea_local_spool_pb_pin.json")
    pin = json.loads(pin_path.read_text())
    root = Path(pin["bundle_root"])
    assert root.is_dir(), "the qualified PB source bundle must be provisioned"
    for name, digest in pin["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    monkeypatch.setattr(chain, "PIN_PATH", pin_path)
    from prismaquant.produced_output_spool import MAX_ENV, ROOT_ENV
    owner, publication, queue, _env, _repo = chain._bound_owner(
        tmp_path, staging_timeout_s=90, published=True,
        claim_capacity={"cpu": 4, "mem_gb": 4}, checkpoint_max_bytes=1 << 20,
        producer_environment={ROOT_ENV: str(tmp_path / "local-output"),
                              MAX_ENV: str(1 << 20)})
    spool = owner._local_output_spool
    assert spool is not None
    plane = {(0, batch): owner.write(_tensor(0, batch), probe_index=0,
                                     batch_index=batch, boundary_index=BOUNDARY)
             for batch in range(chain.GROUP_SIZE)}
    space = adjoint_space(tmp_path / "outputs")
    shared_adjoint, shared_pass = (_shared_adjoint(1, chain.GROUP_SIZE),
                                   _shared_pass(chain.GROUP_SIZE))
    record = write_adjoint_checkpoint(
        space, boundary=BOUNDARY, session=_session(owner), cotangents=plane,
        shared_adjoint=shared_adjoint, shared_pass=shared_pass, owner=owner,
        declared=True)
    directory = checkpoint_directory(space, BOUNDARY)
    assert not directory.exists()
    batch_id = publication.batch_id_for(kind="checkpoint", boundary_index=BOUNDARY,
                                        group_index=0)
    rows = spool._groups[batch_id]["references"]
    assert rows and {artifact_class for _l, _c, artifact_class in rows} == {"checkpoint"}
    sources = [Path(local.path) for local, _c, _a in rows]
    assert all(path.is_file() for path in sources)
    with chain._fleet(queue, tmp_path, capacity={"cpu": 4, "mem_gb": 4}) as fleet:
        owner.settle_local_output()
    outcomes = [json.loads(line) for line in fleet.stdout.splitlines()
                if line.startswith("{")]
    assert len(outcomes) >= 2 and all(row["rc"] == 0 for row in outcomes), outcomes
    for local, canonical, _a in rows:
        data = Path(canonical.path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == local.sha256
    assert not any(path.exists() for path in sources)
    assert spool.report()["pending_groups"] == 0
    loaded, loaded_shared, loaded_pass = load_adjoint_checkpoint(space, record)
    for key in plane:
        _same(loaded[key], _tensor(*key))
    _same(loaded_shared, shared_adjoint)
    _same(loaded_pass, shared_pass)


# -- Stage A end to end: the quantum still matches the single run bitwise --

def test_stage_a_declared_checkpoints_match_the_single_run_bitwise(tmp_path, monkeypatch):
    from prismaquant import aura_cost as aura
    from prismaquant.joint_cost_stage_a import run_adjoint_capture_core
    from test_joint_cost_quantum_runtime import (
        _execution, _hex, _run_quantum, _single_run, _stage_a)
    from test_layer_major_boundary_capture import draw
    from test_streamed_cost_checkpoints import _model_identity
    from tools.compare_joint_layer_gate import compare_layer

    single_root = tmp_path / "single"
    single = _single_run(single_root, monkeypatch, checkpoint=single_root / "checkpoints")
    payload_single = single[0]

    captures = {}
    for declared in (False, True):
        runner, _ = _stage_a(tmp_path, monkeypatch)
        runner.context.settle_prefetch_layers = lambda layers: None
        root = tmp_path / ("declared" if declared else "default")
        captures[declared] = (root, run_adjoint_capture_core(
            runner, draw(), execution=_execution(tmp_path), output_root=root,
            stride=2, source_model_identity=_model_identity("joint-source"),
            unit_roster_sha256=_hex("a"), plan_sha256=_hex("d"),
            prepared_sha256=_hex("e"), read_manifest_sha256=_hex("f"),
            implementation_sha256=aura._aura_source_sha256(),
            declared_checkpoints=declared))
    default_root, default_receipt = captures[False]
    output_root, receipt = captures[True]
    assert "declared_checkpoints" not in default_receipt
    assert [c["schema"] for c in default_receipt["checkpoints"]] == [ADJOINT_CHECKPOINT_SCHEMA]
    assert [c["schema"] for c in receipt["checkpoints"]] == [DECLARED_ADJOINT_CHECKPOINT_SCHEMA]
    block = receipt["declared_checkpoints"]
    assert block["checkpoints"] == 1 and block["referenced_entries"] > 0
    assert block["through_spool"] is False
    # The chain rolled past the referenced tail plane and retired layer 0's:
    # the referenced files are still there, and load the same tensors.
    for row in receipt["checkpoints"][0]["activation_entries"]:
        assert Path(row["path"]).is_file()
    _same(load_adjoint_checkpoint(adjoint_space(output_root), receipt["checkpoints"][0])[0],
          load_adjoint_checkpoint(adjoint_space(default_root),
                                  default_receipt["checkpoints"][0])[0])
    for layer in (1, 0):
        payload, record, _counters = _run_quantum(
            tmp_path, monkeypatch, single=single, layer=layer, receipt=receipt,
            output_root=output_root, plan_sha=_hex("d"), prepared_sha=_hex("e"),
            adjoint_sha=_hex("f"))
        for name, rows in payload["costs"].items():
            for fmt, row in rows.items():
                single_row = payload_single["costs"][name][fmt]
                assert row["signed_components_per_probe"] == \
                    single_row["signed_components_per_probe"], (layer, name, fmt)
                assert row["x2_per_probe"] == single_row["x2_per_probe"]
        verdict = compare_layer(single_root / "checkpoints",
                                Path(record["output_space"]["checkpoint_dir"]),
                                layer=layer, qname_filter=None)
        assert verdict["verdict"] == "match", verdict


def test_declared_pickled_state_is_the_default_bytes(tmp_path):
    """The shared-state pickles are the default owner path's bytes."""
    owner = _owner(tmp_path / "entries", n_probes=2)
    plane = _plane(owner)
    shared_adjoint, shared_pass = _shared_adjoint(2, 3), _shared_pass(3)
    copied = write_adjoint_checkpoint(
        adjoint_space(tmp_path / "copy"), boundary=BOUNDARY, session=_session(owner),
        cotangents=plane, shared_adjoint=shared_adjoint, shared_pass=shared_pass,
        owner=owner)
    declared = write_adjoint_checkpoint(
        adjoint_space(tmp_path / "declared"), boundary=BOUNDARY, session=_session(owner),
        cotangents=plane, shared_adjoint=shared_adjoint, shared_pass=shared_pass,
        owner=owner, declared=True)
    assert [(row["name"], row["sha256"], row["file_bytes"])
            for row in copied["shared_state_entries"]] == [
        (row["name"], row["sha256"], row["file_bytes"])
        for row in declared["shared_state_entries"]]
    assert pickle.loads(Path(declared["shared_state_entries"][0]["path"]).read_bytes()) is not None
