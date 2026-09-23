"""Stage A's reverse chain prefetches its own first layers (RobTand/prismaquant#997).

The chain installs every layer with ``require_prefetched``: a layer that is
neither resident nor in flight refuses rather than read its source
synchronously. Each roll asks for the layer ``lookahead`` below it
(``schedule_reverse_prefetch``), so nothing asks for the chain's first
``lookahead`` layers except the chain's own start. The claims these tests
hold it to, on every entry path:

* A fresh walk, whose forward pass leaves its top layers resident: the chain
  starts on them without reading any of them a second time, and asks for
  whatever the source cache no longer holds.
* A seed, which runs no forward pass, so nothing is resident when its chain
  starts, and whose chain ends above layer 0: it reads nothing below its
  last layer (PQ #1100).
* A resume, likewise.

On every path the chain holds no more source reads in flight than its
lookahead, which is what the forward pass holds too. Each path runs with the
plan's operator windows, which ask for a layer's successors after its
install, and without them, where each roll asks for the layer ``lookahead``
below it.

The context is a bounded LRU source cache that refuses an install the way
``StreamingContext.ensure_loaded`` does. The fixture's own context accepts
every install, which is why a seed and a resume that ask for nothing passed
their tests and refused on the real runtime: R12's checkpoint 045 seeded on
lina refused layer 44 with "streamed layer 44 is not resident after its
required prefetch".
"""
from __future__ import annotations

import pytest
import torch

from prismaquant.cost_streaming import StreamedCausalLM
from prismaquant.model_profiles.default import DefaultProfile

from test_joint_cost_quantum_runtime import _boundary_policy, _execution
from test_stage_a_chain_resume import (
    CAP,
    _DeepTinyLM,
    _at,
    _interrupted,
    _resume,
    _run,
    _streamed,
)
from test_stage_a_chain_seed import _seed, _source, _spec
from test_streamed_cost_checkpoints import _FakeStreamingContext

#: The fixture model's depth (``_DeepTinyLM``).
LAYERS = 5


@pytest.fixture(autouse=True)
def _offline_tier_policy():
    """Run with no staged-tier policy, as the runtime tests do (PQ #845)."""
    from prismaquant.staged_tier_policy import deactivate_staged_tier_policy_for_tests
    deactivate_staged_tier_policy_for_tests()
    yield
    deactivate_staged_tier_policy_for_tests()


class _BoundedResidency(_FakeStreamingContext):
    """A source cache of ``slots`` layers that refuses a cold install.

    ``schedule_prefetch`` mirrors ``StreamingContext.schedule_prefetch``:
    ``None`` for a resident layer, the held read for one in flight, and a
    fresh read otherwise. ``install`` mirrors ``ensure_loaded``: a resident
    layer is hot, an in-flight read is claimed, and anything else refuses
    under ``require_prefetched``. An unloaded layer stays cached until the
    least recently used entry is evicted to make room, the way ``LayerCache``
    keeps value-aware retention.
    """

    def __init__(self, model, *, slots):
        super().__init__(model)
        self.slots = slots
        self.cache: list[int] = []
        self.in_flight: dict[int, object] = {}
        self.reads: list[int] = []
        self.max_in_flight = 0

    def _retain(self, layer):
        if layer in self.cache:
            self.cache.remove(layer)
        self.cache.append(layer)
        for resident in list(self.cache):
            if len(self.cache) <= self.slots:
                break
            if resident not in self.active and resident != layer:
                self.cache.remove(resident)

    def schedule_prefetch(self, layer):
        super().schedule_prefetch(layer)
        layer = int(layer)
        if not 0 <= layer < self.num_layers or layer in self.cache:
            return None
        if layer in self.in_flight:
            return self.in_flight[layer]
        read = self.in_flight[layer] = object()
        self.reads.append(layer)
        self.max_in_flight = max(self.max_in_flight, len(self.in_flight))
        return read

    def install(self, layer, *, require_prefetched=False, prefetch_following=True):
        layer = int(layer)
        if layer not in self.cache and layer not in self.in_flight:
            if require_prefetched:
                raise RuntimeError(
                    f"streamed layer {layer} is not resident after its required "
                    "prefetch; refusing synchronous cold source read")
            self.reads.append(layer)
        self.in_flight.pop(layer, None)
        self._retain(layer)
        return super().install(layer, require_prefetched=require_prefetched)


def _factory(contexts, *, slots, lookahead):
    """The dense fixture runner over a bounded, refusing source cache."""
    def build():
        torch.manual_seed(85)
        state = _DeepTinyLM().eval().state_dict()
        model = _DeepTinyLM(state).eval()
        for layer in model.model.layers:
            layer._fixture_requires_stream_residency = True
        context = _BoundedResidency(model, slots=slots)
        contexts.append(context)
        runner = _streamed(StreamedCausalLM(context, DefaultProfile()), context)
        runner.prefetch_lookahead = lookahead
        return runner
    return build


def _execution_for(root, windows):
    """``_run``'s own execution, with or without the plan's operator windows."""
    execution = _execution(root)
    policy = _boundary_policy(root / "boundaries")
    policy["max_resident_bytes"] = CAP
    execution["boundary_storage"] = policy
    if windows == "none":
        del execution["operator_windows"]
    return execution


LOOKAHEAD = pytest.mark.parametrize("lookahead", [1, 2], ids=["lookahead1", "lookahead2"])
SLOTS = pytest.mark.parametrize("slots", [1, 2], ids=["slots1", "slots2"])
WINDOWS = pytest.mark.parametrize("windows", ["operator", "none"],
                                  ids=["operator-windows", "no-operator-windows"])


@WINDOWS
@LOOKAHEAD
@SLOTS
def test_a_fresh_walk_starts_its_chain_on_what_the_forward_pass_left(
        tmp_path, monkeypatch, lookahead, slots, windows):
    contexts = []
    root = tmp_path / "run"
    receipt = _run(root, monkeypatch, execution=_execution_for(root, windows),
                   runner_factory=_factory(contexts, slots=slots, lookahead=lookahead))
    [context] = contexts
    assert [c["boundary"] for c in receipt["checkpoints"]] == [5, 4, 2]

    # The forward pass read every layer once, in order, and left the top
    # ``slots`` layers cached. The chain read none of those again.
    forward = context.reads[:LAYERS]
    assert forward == list(range(LAYERS))
    left_resident = set(range(LAYERS - slots, LAYERS))
    chain_reads = context.reads[LAYERS:]
    assert not left_resident & set(chain_reads)
    assert sorted(chain_reads, reverse=True) == chain_reads
    assert set(chain_reads) == set(range(LAYERS - slots))

    # The forward pass installed every layer bottom up and the chain top
    # down, all under the refusal.
    assert [layer for kind, layer in context.events if kind == "install"] == (
        list(range(LAYERS)) + list(reversed(range(LAYERS))))
    assert all(context.install_require_prefetched)
    assert context.max_in_flight <= lookahead


@WINDOWS
@LOOKAHEAD
def test_a_seed_starts_its_chain_with_nothing_resident(
        tmp_path, monkeypatch, lookahead, windows):
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    contexts = []
    scratch = tmp_path / "seed"
    receipt = _seed(scratch, monkeypatch, _spec(source),
                    execution=_execution_for(scratch, windows),
                    runner_factory=_factory(contexts, slots=1, lookahead=lookahead))
    [context] = contexts

    # No forward pass: the chain from checkpoint 4 through 2 read layers 3
    # and 2, once each and nothing below them, and installed them top
    # down. Its manifest declares no layer below 2, so a read of layer 1
    # or 0 is one the strict reader refuses (PQ #1100)...
    assert [layer for kind, layer in context.events if kind == "install"] == [3, 2]
    assert all(context.install_require_prefetched)
    assert context.reads == [3, 2]
    assert context.max_in_flight <= lookahead
    # ...and continued the source run bitwise.
    assert receipt["plane_comparison"]["bitwise_equal"] is True


@WINDOWS
@LOOKAHEAD
def test_a_resume_starts_its_chain_with_nothing_resident(
        tmp_path, monkeypatch, lookahead, windows):
    root = tmp_path / "run"
    execution = _execution_for(root, windows)
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2), execution=execution)
    contexts = []
    receipt = _run(root, monkeypatch, chain_resume=_resume(root, resume_from=4),
                   execution=execution,
                   runner_factory=_factory(contexts, slots=1, lookahead=lookahead))
    [context] = contexts
    assert [c["boundary"] for c in receipt["checkpoints"]] == [5, 4, 2]

    # No forward pass: the chain below checkpoint 4 read each of its layers
    # once, top down, under the refusal.
    assert [layer for kind, layer in context.events if kind == "install"] == [3, 2, 1, 0]
    assert all(context.install_require_prefetched)
    assert context.reads == [3, 2, 1, 0]
    assert context.max_in_flight <= lookahead
