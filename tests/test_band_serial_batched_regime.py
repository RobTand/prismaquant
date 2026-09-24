"""Band-serial Stage B under the campaign replay regime (PQ #996, #994, #997).

The GLM campaign captures its one-pass spill at ``capture_batch=4``
(``accumulation=operator_gemm``, ``chunk_rows=65536``) and its Stage A run
rolled the chain at batch size 4 with probe fusion. A band-serial producer
hands off the boundary plane its capture pass wrote; a consumer's chain
rebuild runs the slice's chain regime. Both refusals in the tree compare
their side with the constant 1 (``joint_replay_regime.handoff_regime_refusal``
and ``joint_quantum_handoff.handoff_chain_regime_refusal``), so the campaign
cannot run band-serial today.

The code says the two planes are the same arithmetic when the capture batch
equals the chain batch size: ``capture_group``
(``prismaquant/joint_cost_quantum.py``) and ``_roll_group`` /
``_render_free_fused_passes`` (``prismaquant/joint_adjoint_checkpoints.py``)
build the same ``_chain_group_batch``, run the same ``isolated_layer`` on the
same stacked boundary and the same stacked incoming cotangent, and split
``x_in.grad`` back per stored batch the same way. This module is the GPU
witness for that claim on the spill suite's bf16 packed-expert fixture,
grown to eight samples in read windows of four so a batch of four is legal:

* **Witness.** With both refusals stood down, a producer at
  ``capture_batch=4`` emits a plane that is sha256-equal, entry by entry, to
  the plane the consumer's batch-4 fused chain rebuild ends on; the
  band-serial consumer then pickles to the chain-mode payload.
* **Control.** The same at ``capture_batch=1`` under a batch-1 fused chain,
  the arm every earlier band-serial test ran.
* **The regime refinement.** Unpatched, a producer at ``capture_batch=4``
  under a batch-4 chain regime emits, and its consumer runs band-serial.
  On a tree that still refuses a capture batch above 1 this fails on its own
  assertion, with the refusal as the message.

Every quantum here runs under the spill (a non-default regime replays only
from it) with a 1 GiB ceiling; the fixture is a few kilobytes.
"""
from pathlib import Path
import hashlib
import json
import pickle
import shutil
from types import SimpleNamespace

import pytest
import torch

import prismaquant.aura_cost as aura
# The core imports the roll from its owner at call time: patch it there.
import prismaquant.joint_adjoint_checkpoints as roll_owner
import prismaquant.joint_cost_quantum as core
import prismaquant.joint_quantum_handoff as handoff_module
import prismaquant.joint_replay_regime as regime_module
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.joint_adjoint_checkpoints import chain_layers_for
from prismaquant.joint_adjoint_slices import adjoint_slice_sha256, stage_a_slice
from prismaquant.joint_quantum_handoff import HandoffEmitter, load_quantum_handoff

import test_joint_cost_quantum_runtime as rt
import test_stageb_one_pass_spill as spill_tests
from test_quantum_band_serial import _handoff_plane, _state_digest, _tensor_digest
from test_stageb_one_pass_spill import (
    FORMATS,
    N_PROBES,
    RENDER_FORMATS,
    VOCAB,
    _MoELM,
    _chain,
    _checkpoint_dir,
    _clear_output,
    _evidence,
    _policy_budget,
    _prepared,
    _report,
    _runner,
    _spill_root,
    _targets,
)
from test_streamed_cost_checkpoints import _model_identity

#: The spill suite's helpers, bound before ``_run`` patches their names.
_SPILL_EXECUTION = spill_tests._execution

#: The campaign's launch regime, verbatim
#: (``stage-b-spec.v7-gpu68.json`` ``PRISMAQUANT_STAGE_B_REPLAY_REGIME``).
CAMPAIGN_REGIME = "capture_batch=4,accumulation=operator_gemm,chunk_rows=65536"
SAMPLES, WINDOW, TOKENS = 8, 4, 8
#: One sample's boundary or cotangent entry on this fixture: 8 tokens of
#: width 16 in bf16.
ENTRY_BYTES = TOKENS * spill_tests.WIDTH * 2


def _calibration():
    generator = torch.Generator().manual_seed(996)
    return torch.randint(0, VOCAB, (SAMPLES, TOKENS), generator=generator)


def _execution(root):
    """The spill suite's execution with read windows of four.

    A fused roll at batch 4 holds, per batch, the boundary and every probe's
    incoming entry beside the one rolled entry being written
    (``cost_streaming.fused_window_size``), so the resident cap is sized for
    one window of four such batches.
    """
    execution = _SPILL_EXECUTION(root)
    policy = rt._boundary_policy(root / "boundaries", window=WINDOW)
    policy["max_resident_bytes"] = WINDOW * (1 + N_PROBES) * ENTRY_BYTES + ENTRY_BYTES
    execution["boundary_storage"] = policy
    return execution


def _build(tmp_path_factory, *, name, chain_batch, fusion):
    """One Stage A capture at ``chain_batch``/``fusion`` plus sealed records.

    The spill suite's ``campaign`` fixture, on eight samples in windows of
    four, with the chain regime chosen by the caller.
    """
    from prismaquant.joint_cost_stage_a import run_adjoint_capture_core
    from prismaquant.joint_statistics_replay import preflight_joint_operator_admission

    root = tmp_path_factory.mktemp(name)
    device = spill_tests._device()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
        patch.setenv("PRISMAQUANT_DEV_MODE", "1")
        torch.manual_seed(715)
        state = {key: tensor.detach().clone()
                 for key, tensor in _MoELM().state_dict().items()}
        model, context, runner = _runner(state, device)
        linears = _targets(model, runner.profile)
        weights = {(qname, fmt): module.weight.detach().to("cpu").clone() + 0.03125
                   for qname, module in linears.items() for fmt in RENDER_FORMATS}
        cache, linears = _prepared(model, context, runner, weights, root / "shared")
        formats_by_qname = {qname: list(FORMATS) for qname in linears}
        policy, budget, _retained = _policy_budget()
        names_by_layer = {layer: sorted(n for n in linears
                                        if runner.layer_index_for_qname(n) == layer)
                          for layer in (0, 1)}
        preflight = preflight_joint_operator_admission(
            names_by_layer, linears, {qname: list(RENDER_FORMATS) for qname in linears},
            cache, policy=policy, retained_budget=budget, source_bytes=1 << 20)
        output_root = root / "campaign"
        _model_a, _context_a, runner_a = _runner(state, device)
        receipt = run_adjoint_capture_core(
            runner_a, _calibration(), execution=_execution(root / "exec"),
            output_root=output_root, stride=2,
            source_model_identity=_model_identity("joint-source"),
            unit_roster_sha256=rt._hex("a"), plan_sha256=rt._hex("d"),
            prepared_sha256=rt._hex("e"), read_manifest_sha256=rt._hex("f"),
            implementation_sha256=aura._aura_source_sha256(),
            chain_batch_size=chain_batch, chain_probe_fusion=fusion)
    assert [c["boundary"] for c in receipt["checkpoints"]] == [2]
    records, slices = {}, {}
    for layer in (0, 1):
        slices[layer] = stage_a_slice(json.loads(json.dumps(receipt)), layer)
        windows = rt._windows_records(preflight[layer])
        record = rt._quantum_record(
            output_root=output_root, layer=layer,
            checkpoint_boundary=slices[layer]["checkpoint"]["boundary"], chain=[],
            windows=[{"window_index": index} for index in range(len(windows))],
            total_bytes=sum(w["render_file_upper_bound_bytes"] for w in windows),
            plan_sha=rt._hex("d"), prepared_sha=rt._hex("e"),
            adjoint_sha=adjoint_slice_sha256(slices[layer]))
        record["adjoint"]["chain_layers"] = list(chain_layers_for(2, layer))
        record["identity_sha256"] = canonical_json_sha256(
            {k: v for k, v in record.items() if k != "identity_sha256"}, where="record")
        records[layer] = record
    return SimpleNamespace(root=root, state=state, weights=weights, receipt=receipt,
                           records=records, slices=slices, output_root=output_root,
                           device=device, formats_by_qname=formats_by_qname,
                           preflight=preflight, chain_batch=chain_batch, fusion=fusion)


@pytest.fixture(scope="module")
def campaign4(tmp_path_factory):
    """R13's chain regime: batch 4, probe fusion on."""
    return _build(tmp_path_factory, name="band-serial-b4", chain_batch=4, fusion=True)


@pytest.fixture(scope="module")
def campaign1(tmp_path_factory):
    """The control: batch 1, probe fusion on (the per-sample arithmetic)."""
    return _build(tmp_path_factory, name="band-serial-b1", chain_batch=1, fusion=True)


def _emitter(record, adjoint_slice, execution, regime):
    """A producer's emitter, built as ``run_layer_quantum`` builds it."""
    from prismaquant.joint_replay_regime import normalize_replay_regime

    return HandoffEmitter(record=record, adjoint_slice=adjoint_slice,
                          boundary_storage=execution["boundary_storage"],
                          capture_batch=normalize_replay_regime(regime)["capture_batch"])


def _run(campaign, monkeypatch, *, layer, regime, spill_root, handoff=None,
         emit=False, stand_down=False):
    """The spill suite's quantum harness on this module's fixture.

    ``stand_down`` replaces both regime refusals with ``None`` so the
    arithmetic is compared on its own; the unpatched run is the module's
    last test. The harness imports ``run_layer_quantum_core`` at call time,
    so a wrapper patched onto the module adds the consumer's bound handoff
    and the producer's emitter, both built as ``main`` builds them.
    """
    original = core.run_layer_quantum_core
    published = {}

    def band_serial(*args, record, adjoint_slice, execution, **kwargs):
        emitter = (_emitter(record, adjoint_slice, execution, regime)
                   if emit else None)
        bound = (None if handoff is None else load_quantum_handoff(
            handoff["path"], handoff["sha256"], record=record,
            adjoint_slice=adjoint_slice))
        payload = original(*args, record=record, adjoint_slice=adjoint_slice,
                           execution=execution, adjoint_handoff=bound,
                           handoff_emitter=emitter, **kwargs)
        if emitter is not None:
            published.update(emitter.published)
        return payload

    _clear_output(campaign, layer)
    with monkeypatch.context() as patch:
        patch.setattr(spill_tests, "_calibration", _calibration)
        patch.setattr(spill_tests, "_execution", _execution)
        patch.setattr(core, "run_layer_quantum_core", band_serial)
        if stand_down:
            patch.setattr(regime_module, "handoff_regime_refusal",
                          lambda *args, **kwargs: None)
            patch.setattr(handoff_module, "handoff_chain_regime_refusal",
                          lambda *args, **kwargs: None)
        payload, state = spill_tests._quantum(
            campaign, monkeypatch, layer=layer, spill_root=spill_root,
            ceiling=1 << 30, regime=regime)
    return payload, state, (published or None)


def _chain_consumer(campaign, monkeypatch, *, regime, spill_root, tmp_path):
    """Quantum 0 in chain mode; keeps the plane its chain ends on."""
    rolled = {}
    original_roll = roll_owner.render_free_layer_roll

    def spy(runner, *, layer, roll, cotangents, **kwargs):
        plane = {}

        def keep(tensor, batch, probe):
            plane[(probe, batch)] = tensor.clone()
            roll(tensor, batch, probe)

        backwards = original_roll(runner, layer=layer, roll=keep,
                                  cotangents=cotangents, **kwargs)
        rolled[layer] = (plane, [[owner.state_dict() for owner in row]
                                 for row in cotangents],
                         (kwargs.get("batch_size"), kwargs.get("probe_fusion")),
                         backwards)
        return backwards

    with monkeypatch.context() as patch:
        patch.setattr(roll_owner, "render_free_layer_roll", spy)
        payload, state, _ = _run(campaign, monkeypatch, layer=0, regime=regime,
                                 spill_root=spill_root)
    assert payload is not None, _chain(state.error)
    assert sorted(rolled) == [1]
    plane, states, chain_regime, backwards = rolled[1]
    assert chain_regime == (campaign.chain_batch, campaign.fusion)
    assert backwards == N_PROBES * SAMPLES
    assert set(plane) == {(probe, batch) for probe in range(N_PROBES)
                          for batch in range(SAMPLES)}
    kept = tmp_path / "chain-space"
    shutil.copytree(_checkpoint_dir(campaign, 0), kept)
    return SimpleNamespace(payload=payload, evidence=_evidence(campaign, 0, payload),
                           plane=plane, states=states, kept=kept,
                           counters=state.counters_block)


def _plane_digests(plane):
    return {"/".join(map(str, key)) if isinstance(key, tuple) else str(key):
            _tensor_digest(plane[key])[2] for key in sorted(plane)}


def _assert_is_the_chain_plane(handoff, chain):
    document, plane, states = _handoff_plane(handoff)
    assert document["boundary"] == 1
    assert set(plane) == set(chain.plane)
    for key in sorted(plane):
        assert _tensor_digest(plane[key]) == _tensor_digest(chain.plane[key]), key
    assert {key: _state_digest(state) for key, state in states.items()} == {
        (probe, batch): _state_digest(chain.states[probe][batch])
        for probe in range(len(chain.states))
        for batch in range(len(chain.states[probe]))}


def _band_serial_consumer(campaign, monkeypatch, *, regime, spill_root, handoff,
                          chain, stand_down):
    """Quantum 0 on the handoff: the chain-mode bytes, no chain walked."""
    from tools.compare_joint_layer_gate import compare_layer

    with monkeypatch.context() as patch:
        def no_chain(*args, **kwargs):
            raise AssertionError("a band-serial quantum walked the chain")
        patch.setattr(roll_owner, "render_free_layer_roll", no_chain)
        payload, state, _ = _run(campaign, monkeypatch, layer=0, regime=regime,
                                 spill_root=spill_root, handoff=handoff,
                                 stand_down=stand_down)
    assert payload is not None, _chain(state.error)
    assert state.counters_block["chain"]["layers"] == 0
    assert pickle.dumps(payload) == pickle.dumps(chain.payload)
    assert _evidence(campaign, 0, payload) == chain.evidence
    verdict = compare_layer(chain.kept, _checkpoint_dir(campaign, 0),
                            layer=0, qname_filter=None)
    assert verdict["verdict"] == "match", verdict


def _handoff_of(published):
    path = Path(published["path"])
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _witness(campaign, monkeypatch, tmp_path, capsys, *, regime, stand_down, label):
    spill_root = _spill_root(tmp_path)
    assert campaign.records[1]["adjoint"]["chain_layers"] == []
    assert campaign.records[0]["adjoint"]["chain_layers"] == [1]
    chain = _chain_consumer(campaign, monkeypatch, regime=regime,
                            spill_root=spill_root, tmp_path=tmp_path)
    capture_batch = chain.counters["replay"]["regime"]["capture_batch"] \
        if chain.counters["replay"].get("regime") else 1
    assert capture_batch == campaign.chain_batch

    # The producer's payload and journal do not depend on emitting.
    _silent, silent_state, nothing = _run(campaign, monkeypatch, layer=1,
                                          regime=regime, spill_root=spill_root)
    assert _silent is not None, _chain(silent_state.error)
    assert nothing is None
    silent_evidence = _evidence(campaign, 1, _silent)
    payload, state, published = _run(campaign, monkeypatch, layer=1, regime=regime,
                                     spill_root=spill_root, emit=True,
                                     stand_down=stand_down)
    assert payload is not None, _chain(state.error)
    assert published is not None and Path(published["path"]).is_file()
    assert _evidence(campaign, 1, payload) == silent_evidence
    handoff = _handoff_of(published)
    _assert_is_the_chain_plane(handoff, chain)
    _band_serial_consumer(campaign, monkeypatch, regime=regime, spill_root=spill_root,
                          handoff=handoff, chain=chain, stand_down=stand_down)
    document, plane, _states = _handoff_plane(handoff)
    with capsys.disabled():
        # Past pytest's capture, so the shard log carries the digests of a
        # passing witness too (PrismaBuild shards run ``-q``).
        _report(f"band-serial-{label}-{campaign.device.type}", {
            "regime": regime,
            "chain_regime": {"batch_size": campaign.chain_batch,
                             "probe_fusion": campaign.fusion},
            "samples": SAMPLES, "window": WINDOW, "n_probes": N_PROBES,
            "capture_groups": chain.counters["replay"].get("capture_groups"),
            "handoff_sha256": document["handoff_sha256"],
            "chain_plane": _plane_digests(chain.plane),
            "handoff_plane": _plane_digests(plane),
            "consumer_identity": chain.evidence["identity"]})
    return handoff, chain


def test_the_batched_capture_plane_is_the_batched_chain_plane(campaign4, monkeypatch,
                                                              tmp_path, capsys):
    """The witness: capture batch 4 against a batch-4 fused chain rebuild.

    Both regime refusals are stood down so the arithmetic is compared on
    its own. The producer's capture pass and the consumer's chain roll are
    the same grouping of the same samples through the same kernels, so the
    planes must be sha256-equal entry by entry.
    """
    _witness(campaign4, monkeypatch, tmp_path, capsys, regime=CAMPAIGN_REGIME,
             stand_down=True, label="witness-b4")


def test_the_control_at_batch_one_still_holds(campaign1, monkeypatch, tmp_path, capsys):
    """The control: capture batch 1 under a batch-1 fused chain, unpatched."""
    _witness(campaign1, monkeypatch, tmp_path, capsys, regime=None,
             stand_down=False, label="control-b1")


def test_a_band_serial_producer_runs_the_campaign_regime(campaign4, monkeypatch,
                                                         tmp_path, capsys):
    """Unpatched: a batch-4 capture under a batch-4 chain emits and is consumed.

    On a tree whose refusals compare the capture batch with 1 instead of
    with the slice's chain batch size, the producer refuses before any GPU
    work and this fails on its first assertion, with the refusal as the
    message.
    """
    _witness(campaign4, monkeypatch, tmp_path, capsys, regime=CAMPAIGN_REGIME,
             stand_down=False, label="regime-b4")


# -- the refusals, once the predicate compares the two batch sizes ------------


def _tampered(handoff, mutate, label):
    """A copy of ``handoff`` with ``mutate(document)`` applied and re-sealed."""
    from prismaquant.joint_quantum_handoff import (
        HANDOFF_RECORD_NAME, handoff_record_bytes, handoff_seal_sha256)

    path = Path(handoff["path"])
    document = json.loads(path.read_bytes())
    mutate(document)
    document["handoff_sha256"] = handoff_seal_sha256(document)
    payload = handoff_record_bytes(document)
    twin = path.parent.parent / f"{path.parent.name}-{label}" / HANDOFF_RECORD_NAME
    twin.parent.mkdir()
    twin.write_bytes(payload)
    document["session"]["generation"] = twin.parent.name
    document["handoff_sha256"] = handoff_seal_sha256(document)
    payload = handoff_record_bytes(document)
    twin.write_bytes(payload)
    return {"path": str(twin), "sha256": hashlib.sha256(payload).hexdigest()}


def test_a_capture_off_the_chains_batch_size_refuses_before_any_gpu_work(
        campaign4, monkeypatch, tmp_path):
    """Under a batch-4 chain regime a batch-1 producer refuses at its head."""
    from prismaquant.joint_quantum_handoff import QuantumHandoffRefused

    with pytest.raises(QuantumHandoffRefused, match="batch size 4.*captured at batch 1"):
        _emitter(campaign4.records[1], campaign4.slices[1],
                 _execution(campaign4.root / "exec"), None)
    with pytest.raises(QuantumHandoffRefused, match="batch size 4.*captured at batch 2"):
        _emitter(campaign4.records[1], campaign4.slices[1],
                 _execution(campaign4.root / "exec"), "capture_batch=2")
    # The core's own check, for an emitter that slipped past the head.
    emitter = _emitter(campaign4.records[1], campaign4.slices[1],
                       _execution(campaign4.root / "exec"), CAMPAIGN_REGIME)
    with monkeypatch.context() as patch:
        patch.setattr(handoff_module, "handoff_chain_regime_refusal",
                      lambda *args, **kwargs: None)
        patch.setattr(spill_tests, "_calibration", _calibration)
        patch.setattr(spill_tests, "_execution", _execution)
        original = core.run_layer_quantum_core

        def with_emitter(*args, **kwargs):
            return original(*args, handoff_emitter=emitter, **kwargs)

        patch.setattr(core, "run_layer_quantum_core", with_emitter)
        _clear_output(campaign4, 1)
        payload, state = spill_tests._quantum(
            campaign4, monkeypatch, layer=1, spill_root=_spill_root(tmp_path),
            ceiling=1 << 30, regime="capture_batch=2")
    assert payload is None
    assert "capture_batch=2" in _chain(state.error)
    assert "batch size 4" in _chain(state.error)
    assert state.context.install_calls == 0


def test_a_consumer_refuses_a_handoff_captured_at_another_batch(
        campaign4, monkeypatch, tmp_path, capsys):
    """The stamped capture batch is checked against the consumer's slice."""
    from prismaquant.joint_quantum_handoff import QuantumHandoffRefused

    handoff, _chain_state = _witness(campaign4, monkeypatch, tmp_path, capsys,
                                     regime=CAMPAIGN_REGIME, stand_down=False,
                                     label="consumer-refusals-b4")
    document = json.loads(Path(handoff["path"]).read_bytes())
    assert document["producer"]["capture_batch"] == 4
    record, adjoint_slice = campaign4.records[0], campaign4.slices[0]
    load_quantum_handoff(handoff["path"], handoff["sha256"], record=record,
                         adjoint_slice=adjoint_slice)

    def at_batch_one(document):
        document["producer"]["capture_batch"] = 1

    def unstamped(document):
        del document["producer"]["capture_batch"]

    with pytest.raises(QuantumHandoffRefused, match="batch size 4.*captured at batch 1"):
        twin = _tampered(handoff, at_batch_one, "batch-one")
        load_quantum_handoff(twin["path"], twin["sha256"], record=record,
                             adjoint_slice=adjoint_slice)
    with pytest.raises(QuantumHandoffRefused, match="records no capture batch"):
        twin = _tampered(handoff, unstamped, "unstamped")
        load_quantum_handoff(twin["path"], twin["sha256"], record=record,
                             adjoint_slice=adjoint_slice)
