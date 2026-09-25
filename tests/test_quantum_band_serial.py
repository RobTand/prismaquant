"""Band-serial Stage B (PQ #996): quantum L hands its cotangent to L - 1.

A real Stage A run on a five-layer fixture at stride 3 seals checkpoints 5
and 3: band 5 serves layers 4 and 3, band 3 serves layers 2, 1 and 0. In
band 3, quantum 1 both consumes quantum 2's handoff and emits its own for
quantum 0, so a handoff built from a handoff is covered too.

The acceptance, checked byte for byte against the chain-rebuild path, with
Stage A's probe fusion off and on (PQ #997; a batch size of one either way):

* every band-serial cost payload pickles to the chain-mode bytes, and the
  unit journals match;
* every handoff plane is sha256-equal, entry by entry, to the plane the
  consumer's chain rebuild ends on, and the owner states are equal too.

This fixture shares no KV state, so its owner states are all empty. A
separate round trip gives each ``(probe, sample)`` its own accumulator and
checks that the consumer reads each one back at its own coordinate.
"""
from pathlib import Path
import hashlib
import json
import pickle
import shutil

import pytest
import torch

import prismaquant.aura_cost as aura
# The core imports the roll from its owner at call time: patch it there.
import prismaquant.joint_adjoint_checkpoints as quantum_module
from prismaquant.cost_stage_checkpoint import canonical_json_bytes
from prismaquant.joint_adjoint_checkpoints import read_exact_entry_tensors
from prismaquant.joint_adjoint_slices import chain_regime_identity, stage_a_slice
from prismaquant.joint_cost_stage_a import run_adjoint_capture_core
from prismaquant.joint_quantum_handoff import (
    HANDOFF_RECORD_NAME,
    HandoffEmitter,
    QuantumHandoffRefused,
    handoff_read_entries,
    handoff_record_bytes,
    handoff_seal_sha256,
    load_handoff_inputs,
    load_quantum_handoff,
)
from prismaquant.sensitivity_probe import SharedStateCotangents

from test_joint_cost_quantum_runtime import (  # noqa: F401 (autouse fixture)
    _execution,
    _hex,
    _offline_tier_policy,
    _run_quantum,
    _single_run,
    _stage_a,
)
from test_streamed_cost_checkpoints import _model_identity

LAYERS, STRIDE = 5, 3
#: Band boundary -> its layers, top first (the order a band runs serially).
BANDS = {5: [4, 3], 3: [2, 1, 0]}


def _tensor_digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return (str(tensor.dtype), tuple(tensor.shape),
            hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest())


def _state_digest(state):
    """A shared-cotangent owner state, compared by value, tensors by bytes."""
    return (state["enabled"], sorted(state["counters"].items()),
            list(state["nondifferentiable"]),
            [(tuple(row["slot"]), _tensor_digest(row["tensor"]))
             for row in state["accumulators"]])


def _campaign(tmp_path, monkeypatch, *, fusion=False):
    single_root = tmp_path / "single"
    single = _single_run(single_root, monkeypatch,
                         checkpoint=single_root / "checkpoints", layers=LAYERS)
    output_root = tmp_path / "campaign"
    runner_a, _ = _stage_a(tmp_path, monkeypatch, layers=LAYERS)
    runner_a.context.settle_prefetch_layers = lambda layers: None
    execution = _execution(tmp_path)
    if fusion:
        # A fused window holds each batch's boundary and all four probes'
        # entries, beside the one rolled entry being written (#997's own
        # sizing, tests/test_stage_a_chain_regime.py).
        policy = execution["boundary_storage"]
        policy["max_resident_bytes"] = 2 * (1 + execution["n_probes"]) * 256 + 256
    receipt = run_adjoint_capture_core(
        runner_a, torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [2, 3, 4, 1],
                                [3, 4, 1, 2], [1, 3, 2, 4]]),
        execution=execution,
        output_root=output_root, stride=STRIDE,
        source_model_identity=_model_identity("joint-source"),
        unit_roster_sha256=_hex("a"), plan_sha256=_hex("d"),
        prepared_sha256=_hex("e"), read_manifest_sha256=_hex("f"),
        implementation_sha256=aura._aura_source_sha256(),
        chain_probe_fusion=fusion)
    receipt = json.loads(json.dumps(receipt))
    assert sorted(c["boundary"] for c in receipt["checkpoints"]) == [3, 5]
    return single, receipt, output_root


def _emitter(record, adjoint_slice, execution):
    # This fixture runs the default replay regime: a capture batch of one.
    return HandoffEmitter(record=record, adjoint_slice=adjoint_slice,
                          boundary_storage=execution["boundary_storage"],
                          capture_batch=1)


def _consumer(handoff):
    def bind(record, adjoint_slice):
        return load_quantum_handoff(handoff["path"], handoff["sha256"],
                                    record=record, adjoint_slice=adjoint_slice,
                                    kda_capture_kernel=None)
    return bind


def _quantum(tmp_path, monkeypatch, *, single, receipt, output_root, layer, **kw):
    return _run_quantum(tmp_path, monkeypatch, single=single, layer=layer,
                        receipt=receipt, output_root=output_root,
                        plan_sha=_hex("d"), prepared_sha=_hex("e"), **kw)


def _handoff_plane(handoff):
    """Read a published handoff back through the verified exact reader."""
    document = json.loads(Path(handoff["path"]).read_bytes())
    plane = {}
    for entry in document["activation_entries"]:
        coordinates = entry["metadata"]["identity"]["coordinates"]
        tensors = read_exact_entry_tensors([entry],
                                           expected_session=document["session"])
        plane[(coordinates["probe"], coordinates["batch"])] = tensors[entry["name"]]
    states = pickle.loads(Path(document["owner_states"]["path"]).read_bytes())
    return document, plane, dict(states["states"])


@pytest.mark.parametrize("fusion", [False, True], ids=["unfused", "fused"])
def test_band_serial_payloads_and_planes_equal_the_chain_rebuild(
        tmp_path, monkeypatch, fusion):
    single, receipt, output_root = _campaign(tmp_path, monkeypatch, fusion=fusion)
    assert ("chain_regime" in receipt["run_identity"]) is fusion

    # ---- chain mode: every quantum rebuilds its incoming plane -----------
    rolled = {}
    regimes = set()
    original = quantum_module.render_free_layer_roll

    def spy(runner, *, layer, roll, cotangents, **kwargs):
        regimes.add((kwargs.get("batch_size"), kwargs.get("probe_fusion")))
        plane = {}

        def keep(tensor, batch, probe):
            plane[(probe, batch)] = tensor.clone()
            roll(tensor, batch, probe)

        backwards = original(runner, layer=layer, roll=keep,
                             cotangents=cotangents, **kwargs)
        rolled.setdefault(current["layer"], {})[layer] = (
            plane, [[owner.state_dict() for owner in row] for row in cotangents])
        return backwards

    current = {}
    chain = {}
    kept = tmp_path / "chain-spaces"
    kept.mkdir()
    with monkeypatch.context() as patch:
        patch.setattr(quantum_module, "render_free_layer_roll", spy)
        for layers in BANDS.values():
            for layer in layers:
                current["layer"] = layer
                payload, record, counters = _quantum(
                    tmp_path, monkeypatch, single=single, receipt=receipt,
                    output_root=output_root, layer=layer)
                assert payload["costs"], layer
                assert counters["chain"]["layers"] == len(record["adjoint"]["chain_layers"])
                chain[layer] = (payload, record)
                space = Path(record["output_space"]["root"])
                shutil.move(str(space), str(kept / space.name))

    # The chains ran in Stage A's regime, so the fused case is not vacuous.
    assert regimes == {(1, fusion)}

    # ---- band-serial: each band walks top down, handing off --------------
    handoffs = {}
    with monkeypatch.context() as patch:
        def no_chain(*args, **kwargs):
            raise AssertionError("a band-serial quantum walked the chain")
        patch.setattr(quantum_module, "render_free_layer_roll", no_chain)
        for boundary, layers in BANDS.items():
            for position, layer in enumerate(layers):
                emits = position + 1 < len(layers)
                consumes = position > 0
                payload, record, counters = _quantum(
                    tmp_path, monkeypatch, single=single, receipt=receipt,
                    output_root=output_root, layer=layer,
                    adjoint_handoff=(_consumer(handoffs[layer + 1])
                                     if consumes else None),
                    handoff_emitter=_emitter if emits else None)
                chain_payload, chain_record = chain[layer]
                # The record, the payload and the journal are the chain bytes.
                assert canonical_json_bytes(record, where="record") == \
                    canonical_json_bytes(chain_record, where="record")
                assert pickle.dumps(payload) == pickle.dumps(chain_payload), layer
                assert counters["chain"]["layers"] == 0
                from tools.compare_joint_layer_gate import compare_layer
                verdict = compare_layer(
                    kept / f"layer-{layer:03d}" / "checkpoints",
                    Path(record["output_space"]["checkpoint_dir"]),
                    layer=layer, qname_filter=None)
                assert verdict["verdict"] == "match", verdict
                if emits:
                    handoff = next((Path(record["output_space"]["root"])
                                    / "handoff").glob(f"*/{HANDOFF_RECORD_NAME}"))
                    raw = handoff.read_bytes()
                    handoffs[layer] = {"path": str(handoff),
                                       "sha256": hashlib.sha256(raw).hexdigest()}

    # ---- each handoff plane is the plane the consumer's chain ends on -----
    assert sorted(handoffs) == [1, 2, 4]
    for producer, handoff in handoffs.items():
        consumer = producer - 1
        document, plane, states = _handoff_plane(handoff)
        assert document["boundary"] == producer
        chain_plane, chain_states = rolled[consumer][producer]
        assert set(plane) == set(chain_plane)
        for key in sorted(plane):
            assert _tensor_digest(plane[key]) == _tensor_digest(chain_plane[key]), \
                (producer, key)
        assert {key: _state_digest(state) for key, state in states.items()} == {
            (probe, batch): _state_digest(chain_states[probe][batch])
            for probe in range(len(chain_states))
            for batch in range(len(chain_states[probe]))}


def _producer(tmp_path, monkeypatch, *, layer):
    """A campaign plus one emitted handoff from ``layer`` (band-serial)."""
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    _payload, record, _counters = _quantum(
        tmp_path, monkeypatch, single=single, receipt=receipt,
        output_root=output_root, layer=layer, handoff_emitter=_emitter)
    path = next((Path(record["output_space"]["root"]) / "handoff").glob(
        f"*/{HANDOFF_RECORD_NAME}"))
    return single, receipt, output_root, record, path


def _consumer_record(producer_record, *, layer, receipt):
    """The chain-mode record the harness would seal for ``layer``."""
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    from prismaquant.joint_adjoint_slices import adjoint_slice_sha256, chain_layers_for

    adjoint_slice = stage_a_slice(receipt, layer)
    record = json.loads(json.dumps(producer_record))
    root = Path(record["output_space"]["root"]).parent / f"layer-{layer:03d}"
    record.update(quantum_id=f"layer-{layer:03d}", layer=layer)
    record["output_space"] = {key: str(root / Path(value).name)
                              for key, value in record["output_space"].items()}
    record["output_space"]["root"] = str(root)
    boundary = adjoint_slice["checkpoint"]["boundary"]
    record["adjoint"].update(
        checkpoint_boundary=boundary,
        chain_layers=list(chain_layers_for(boundary, layer)),
        slice_sha256=adjoint_slice_sha256(adjoint_slice))
    record.pop("identity_sha256")
    record["identity_sha256"] = canonical_json_sha256(record, where="record")
    return record, adjoint_slice


def test_a_handoff_binds_its_consumer_and_refuses_everything_else(tmp_path, monkeypatch):
    _single, receipt, _root, producer_record, path = _producer(
        tmp_path, monkeypatch, layer=2)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    record, adjoint_slice = _consumer_record(producer_record, layer=1, receipt=receipt)
    handoff = load_quantum_handoff(path, sha, record=record, adjoint_slice=adjoint_slice,
                                   kda_capture_kernel=None)
    assert handoff["boundary"] == 2 and handoff["producer"]["layer"] == 2
    # The head's staged reads: the plane, the owner states and the slice
    # checkpoint's shared-pass states -- never the checkpoint plane. A packed
    # (v3) checkpoint holds those in its one pack (PQ #1037).
    rows = handoff_read_entries(handoff, adjoint_slice["checkpoint"])
    shared_pass = [entry for entry in adjoint_slice["checkpoint"]["shared_state_entries"]
                   if entry["name"].startswith("shared-pass-")
                   or entry["name"] == "shared-states"]
    assert shared_pass
    assert [row["path"] for row in rows] == [
        *(entry["path"] for entry in handoff["activation_entries"]),
        handoff["owner_states"]["path"], *(entry["path"] for entry in shared_pass)]
    assert all(Path(entry["path"]).parent == path.parent / "entries"
               for entry in handoff["activation_entries"])

    def refused(match, *, record=record, adjoint_slice=adjoint_slice, path=path, sha=sha):
        with pytest.raises(QuantumHandoffRefused, match=match):
            load_quantum_handoff(path, sha, record=record, adjoint_slice=adjoint_slice,
                                 kda_capture_kernel=None)

    # The bound digest is the file's.
    refused("does not hash", sha=_hex("0"))
    # Layer 0 consumes boundary 1, not the layer-2 handoff.
    zero, zero_slice = _consumer_record(producer_record, layer=0, receipt=receipt)
    refused("consumes boundary 1", record=zero, adjoint_slice=zero_slice)
    # A Stage A run whose chain batch size is not one refuses (PQ #997), and
    # so does a malformed stamp.
    stamped = json.loads(json.dumps(adjoint_slice))
    stamped["run_identity"]["chain_regime"] = chain_regime_identity(
        {"batch_size": 2, "probe_fusion": False})
    refused("batch size 2", adjoint_slice=stamped)
    stamped["run_identity"]["chain_regime"] = {"batch_size": 2}
    refused("chain regime is malformed", adjoint_slice=stamped)
    # A re-sealed record naming another checkpoint plane refuses.
    forged = json.loads(raw)
    forged["source"]["checkpoint_cotangent_sha256"] = _hex("0")
    forged["handoff_sha256"] = handoff_seal_sha256(forged)
    forged_bytes = handoff_record_bytes(forged)
    moved = path.parent.parent / "forged-generation" / HANDOFF_RECORD_NAME
    moved.parent.mkdir()
    moved.write_bytes(forged_bytes)
    refused("another campaign", path=moved,
            sha=hashlib.sha256(forged_bytes).hexdigest())
    # A handoff outside the producer's output space refuses.
    elsewhere = tmp_path / "elsewhere" / path.parent.name / HANDOFF_RECORD_NAME
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_bytes(raw)
    refused("is not inside", path=elsewhere)
    # A seal that does not cover the bytes refuses.
    tampered = json.loads(raw)
    tampered["n_batches"] += 1
    tampered_bytes = handoff_record_bytes(tampered)
    path.write_bytes(tampered_bytes)
    refused("does not match its seal", sha=hashlib.sha256(tampered_bytes).hexdigest())
    path.write_bytes(raw)


def test_owner_states_travel_per_probe_and_sample(tmp_path, monkeypatch, capsys):
    """Distinct shared-state owners survive the handoff, coordinate by coordinate.

    The end-to-end fixtures share no KV state, so each of their owner states
    is empty and equal to every other: that comparison cannot tell one
    ``(probe, sample)`` from another. Here every owner holds its own
    accumulator, and the consumer's read must hand each one back at its own
    coordinate. Only the Gemma4 profile carries shared KV state today.
    """
    _single, receipt, _root, producer_record, path = _producer(
        tmp_path, monkeypatch, layer=2)
    document, plane, _states = _handoff_plane({"path": str(path)})
    n_probes, n_batches = document["n_probes"], document["n_batches"]
    slot = ("shared_kv_states", 0, 0)
    owners = []
    for probe in range(n_probes):
        row = []
        for batch in range(n_batches):
            owner = SharedStateCotangents()
            grafted = owner.graft({"shared_kv_states": {0: (torch.ones(2, 2),)}})
            grafted["shared_kv_states"][0][0].grad = torch.full(
                (2, 2), float(1 + 10 * probe + batch))
            owner.harvest()
            assert slot in owner._acc
            row.append(owner)
        owners.append(row)
    published = _emitter(producer_record, stage_a_slice(receipt, 2),
                         _execution(tmp_path)).emit(
        grad_plane=plane, cotangent_owners=owners, n_probes=n_probes,
        n_batches=n_batches, kda_capture_kernel=None)

    record, adjoint_slice = _consumer_record(producer_record, layer=1, receipt=receipt)
    bound = load_quantum_handoff(published["path"], published["sha256"],
                                 record=record, adjoint_slice=adjoint_slice,
                                 kda_capture_kernel=None)
    capsys.readouterr()
    read_plane, shared_adjoint, _shared_pass = load_handoff_inputs(
        bound, adjoint_slice["checkpoint"], n_probes=n_probes, n_batches=n_batches)
    assert {key: _tensor_digest(value) for key, value in read_plane.items()} == \
        {key: _tensor_digest(value) for key, value in plane.items()}
    # The handoff read prints its rate lines, ending with a final one.
    from prismaquant.io_spans import READ_RATE_MARKER
    rates = [json.loads(line[len(READ_RATE_MARKER) + 1:])
             for line in capsys.readouterr().out.splitlines()
             if line.startswith(READ_RATE_MARKER + " {")]
    assert [(r["label"], r["final"]) for r in rates][-1] == ("handoff-load", True)
    assert rates[-1]["entries"] == rates[-1]["entries_total"] == n_probes * n_batches
    expected = {(probe, batch): _state_digest(owners[probe][batch].state_dict())
                for probe in range(n_probes) for batch in range(n_batches)}
    assert len(set(map(repr, expected.values()))) == n_probes * n_batches
    assert {key: _state_digest(state) for key, state in shared_adjoint.items()} \
        == expected
    # The quantum restores each state into a fresh owner (joint_cost_quantum).
    for (probe, batch), state in shared_adjoint.items():
        restored = SharedStateCotangents()
        restored.load_state_dict(state)
        assert torch.equal(restored._acc[slot], owners[probe][batch]._acc[slot])


def test_the_band_top_takes_no_handoff(tmp_path, monkeypatch):
    """Layer 3 is band 5's bottom; its handoff cannot serve band 3's top."""
    _single, receipt, _root, producer_record, path = _producer(
        tmp_path, monkeypatch, layer=3)
    record, adjoint_slice = _consumer_record(producer_record, layer=2, receipt=receipt)
    with pytest.raises(QuantumHandoffRefused, match="tops band 3"):
        load_quantum_handoff(path, hashlib.sha256(path.read_bytes()).hexdigest(),
                             record=record, adjoint_slice=adjoint_slice,
                             kda_capture_kernel=None)


def test_the_emitter_refuses_layer_zero_and_a_capture_batch_off_the_chains(
        tmp_path, monkeypatch):
    """A producer captures at its slice's chain batch size, and only there
    (PQ #994, #997): the plane it hands off is then the chain's plane."""
    receipt_record = {"layer": 0}
    with pytest.raises(QuantumHandoffRefused, match="layer 0"):
        HandoffEmitter(record=receipt_record, adjoint_slice={"run_identity": {}},
                       boundary_storage={}, capture_batch=1)
    batched = chain_regime_identity({"batch_size": 8, "probe_fusion": True})
    with pytest.raises(QuantumHandoffRefused, match="batch size 8.*captured at batch 1"):
        HandoffEmitter(record={"layer": 2},
                       adjoint_slice={"run_identity": {"chain_regime": batched}},
                       boundary_storage={}, capture_batch=1)
    # The default chain regime rolls per sample: a batched capture refuses.
    with pytest.raises(QuantumHandoffRefused, match="batch size 1.*captured at batch 4"):
        HandoffEmitter(record={"layer": 2}, adjoint_slice={"run_identity": {}},
                       boundary_storage={}, capture_batch=4)
    with pytest.raises(QuantumHandoffRefused, match="chain regime is malformed"):
        HandoffEmitter(record={"layer": 2},
                       adjoint_slice={"run_identity": {"chain_regime": {"batch_size": 1}}},
                       boundary_storage={}, capture_batch=1)
    with pytest.raises(QuantumHandoffRefused, match="positive integer"):
        HandoffEmitter(record={"layer": 2}, adjoint_slice={"run_identity": {}},
                       boundary_storage={}, capture_batch=0)
    with pytest.raises(TypeError):
        HandoffEmitter(record={"layer": 2}, adjoint_slice={"run_identity": {}},
                       boundary_storage={})  # the capture batch is never implied
    # Probe fusion at a batch size of one is the per-sample arithmetic.
    fused = chain_regime_identity({"batch_size": 1, "probe_fusion": True})
    HandoffEmitter(record={"layer": 2},
                   adjoint_slice={"run_identity": {"chain_regime": fused}},
                   boundary_storage={}, capture_batch=1)
    # A batched chain regime admits the same batch, fusion on or off.
    for fusion in (False, True):
        regime = chain_regime_identity({"batch_size": 8, "probe_fusion": fusion})
        emitter = HandoffEmitter(
            record={"layer": 2},
            adjoint_slice={"run_identity": {"chain_regime": regime}},
            boundary_storage={}, capture_batch=8)
        assert emitter.capture_batch == 8


def test_a_failed_emission_publishes_no_record(tmp_path, monkeypatch):
    """The record is written last: a write that fails leaves no handoff.json.

    The quantum streams its handoff (PQ #1251): the writer thread's read
    fails, and the quantum raises that error at its next store or finish.
    """
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)

    class Broken(HandoffEmitter):
        def stream(self, *, grad_plane, **kwargs):
            class Plane(dict):
                def __getitem__(self, key):
                    if key == (1, 2):
                        raise RuntimeError("fixture: plane read failed")
                    return grad_plane[key]
            return super().stream(grad_plane=Plane(), **kwargs)

    def broken(record, adjoint_slice, execution):
        return Broken(record=record, adjoint_slice=adjoint_slice,
                      boundary_storage=execution["boundary_storage"],
                      capture_batch=1)

    with pytest.raises(RuntimeError, match="plane read failed"):
        _quantum(tmp_path, monkeypatch, single=single, receipt=receipt,
                 output_root=output_root, layer=2, handoff_emitter=broken)
    root = output_root / "layer-quanta" / "layer-002" / "handoff"
    assert not list(root.glob(f"*/{HANDOFF_RECORD_NAME}"))
    statuses = [json.loads(path.read_text())["status"]
                for path in root.glob("*/generation.json")]
    assert statuses == ["failed"]
