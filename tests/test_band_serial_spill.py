"""Band-serial Stage B (PQ #996) under the one-pass replay spill (PQ #994).

The spill runs each probe's target layer once, as that probe's final pass,
and feeds every retained window from a local spill. That final pass is the
one that writes the plane a band-serial producer hands off, so the handoff
cannot depend on which replay mode ran. Replay mode is declared by the
environment and sits outside the quantum identity, so a producer and its
consumer may run under different modes.

This test uses the spill suite's bf16 packed-expert fixture: one Stage A
checkpoint at boundary 2, which makes one band in which quantum 1 hands off
to quantum 0. It checks the following:

* Quantum 1 emits the same handoff plane and owner states under the spill as
  under the windowed replay, and both are sha256-equal to the plane that
  quantum 0's chain rebuild ends on.
* Quantum 1's payload and unit journals are the same whether or not it emits.
* Quantum 0 run band-serial, under the spill and under the windowed replay,
  pickles to its chain-mode payload, with the same unit journals.
"""
from pathlib import Path
import pickle
import shutil

import pytest

# The core imports the roll from its owner at call time: patch it there.
import prismaquant.joint_adjoint_checkpoints as roll_owner
import prismaquant.joint_cost_quantum as core
from prismaquant.joint_quantum_handoff import HandoffEmitter, load_quantum_handoff

from test_quantum_band_serial import _handoff_plane, _state_digest, _tensor_digest
from test_stageb_one_pass_spill import (  # noqa: F401 (module fixture)
    _chain,
    _checkpoint_dir,
    _clear_output,
    _evidence,
    _quantum as _spill_harness,
    _spill_root,
    campaign,
)

SPILL, WINDOWED = "one_pass_spill", "windowed"


def _run(campaign, monkeypatch, *, layer, spill_root=None, handoff=None, emit=False):
    """The spill suite's quantum harness, run band-serial when asked.

    The harness imports ``run_layer_quantum_core`` at call time, so a wrapper
    patched onto the module adds the consumer's bound handoff and the
    producer's emitter, both built as ``main`` builds them.
    """
    original = core.run_layer_quantum_core
    published = {}

    def band_serial(*args, record, adjoint_slice, execution, **kwargs):
        # This suite's producer captures at batch 1 (the default regime).
        emitter = (HandoffEmitter(record=record, adjoint_slice=adjoint_slice,
                                  boundary_storage=execution["boundary_storage"],
                                  capture_batch=1)
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
        patch.setattr(core, "run_layer_quantum_core", band_serial)
        payload, state = _spill_harness(
            campaign, monkeypatch, layer=layer, spill_root=spill_root,
            ceiling=None if spill_root is None else 1 << 30)
    assert payload is not None, _chain(state.error)
    mode = state.counters_block["replay"]["mode"]
    assert mode == (WINDOWED if spill_root is None else SPILL), mode
    return payload, _evidence(campaign, layer, payload), (published or None)


def test_band_serial_under_the_spill_equals_the_chain_rebuild(campaign, monkeypatch,
                                                             tmp_path):
    from tools.compare_joint_layer_gate import compare_layer

    spill_root = _spill_root(tmp_path)
    assert campaign.records[1]["adjoint"]["chain_layers"] == []
    assert campaign.records[0]["adjoint"]["chain_layers"] == [1]

    # ---- quantum 0, chain mode: keep the plane its chain ends on ----------
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
                                 for row in cotangents])
        return backwards

    chain = {}
    with monkeypatch.context() as patch:
        patch.setattr(roll_owner, "render_free_layer_roll", spy)
        for mode, root in ((SPILL, spill_root), (WINDOWED, None)):
            payload, evidence, _ = _run(campaign, monkeypatch, layer=0,
                                        spill_root=root)
            kept = tmp_path / f"chain-{mode}"
            shutil.copytree(_checkpoint_dir(campaign, 0), kept)
            chain[mode] = (payload, evidence, kept)
    assert sorted(rolled) == [1]
    chain_plane, chain_states = rolled[1]
    # The two replay modes price quantum 0 identically (#994's own claim,
    # here on the chain side of this test).
    assert chain[SPILL][1] == chain[WINDOWED][1]

    def assert_is_the_chain_plane(handoff):
        document, plane, states = _handoff_plane(handoff)
        assert document["boundary"] == 1
        assert set(plane) == set(chain_plane)
        for key in sorted(plane):
            assert _tensor_digest(plane[key]) == _tensor_digest(chain_plane[key]), key
        assert {key: _state_digest(state) for key, state in states.items()} == {
            (probe, batch): _state_digest(chain_states[probe][batch])
            for probe in range(len(chain_states))
            for batch in range(len(chain_states[probe]))}

    # ---- quantum 1, the producer, under both replay modes -----------------
    _silent, silent_evidence, nothing = _run(campaign, monkeypatch, layer=1,
                                             spill_root=spill_root)
    assert nothing is None
    _windowed, windowed_evidence, windowed_handoff = _run(
        campaign, monkeypatch, layer=1, emit=True)
    assert_is_the_chain_plane(windowed_handoff)
    # The windowed producer's output space, handoff included, goes next.
    _spilled, spilled_evidence, handoff = _run(campaign, monkeypatch, layer=1,
                                               spill_root=spill_root, emit=True)
    assert_is_the_chain_plane(handoff)
    assert spilled_evidence == windowed_evidence == silent_evidence
    assert Path(handoff["path"]).is_file()

    # ---- quantum 0, band-serial, consumes the spill producer's handoff ----
    with monkeypatch.context() as patch:
        def no_chain(*args, **kwargs):
            raise AssertionError("a band-serial quantum walked the chain")
        patch.setattr(roll_owner, "render_free_layer_roll", no_chain)
        for mode, root in ((SPILL, spill_root), (WINDOWED, None)):
            payload, evidence, _ = _run(campaign, monkeypatch, layer=0,
                                        spill_root=root, handoff=handoff)
            chain_payload, chain_evidence, kept = chain[mode]
            assert pickle.dumps(payload) == pickle.dumps(chain_payload), mode
            assert evidence == chain_evidence, mode
            verdict = compare_layer(kept, _checkpoint_dir(campaign, 0),
                                    layer=0, qname_filter=None)
            assert verdict["verdict"] == "match", (mode, verdict)
