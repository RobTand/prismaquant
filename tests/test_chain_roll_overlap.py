"""The chain roll's host side overlaps the GPU (RobTand/prismaquant#1162).

Each backward's input cotangent is copied out without a host wait and handed
to ``roll`` one step late, while the next backward runs. The claims:

* Every step but the last of a read window rolls after the next backward
  was launched; the last rolls before the window closes. At most one step is
  ever waiting (the in-flight bound).
* No read sees a stale or unpublished entry: the next window, and the next
  layer's first window that the last window stages, name only rows that were
  already rolled.
* A roll that raises fails the layer roll where it is raised, and nothing
  rolls after it. A backward that raises is not masked by the step waiting
  behind it, and that step is never rolled.
* A compact host tensor is serialized without a second copy, into the same
  bytes; a view of a wider storage is still copied.
* On CUDA, a roll that keeps nothing receives each row in its own pinned
  buffer; the default roll receives pageable rows it may keep. Both are the
  gradient's bytes, and the rolled planes equal the pre-#997 roll's.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from prismaquant import perturbed_x_cache as cache_module
from prismaquant.cost_streaming import prefetched_boundary_batches, prefetched_fused_boundary_windows
from prismaquant.joint_adjoint_checkpoints import render_free_layer_roll
from prismaquant.perturbed_x_cache import write_exact_activation_cache_entry
from prismaquant.sensitivity_probe import SharedStateCotangents

from test_stage_a_chain_regime import _toy_batches, _toy_runner

REGIMES = pytest.mark.parametrize(
    "batch_size, fusion", [(1, False), (2, False), (1, True), (2, True)],
    ids=["b1", "b2", "b1-fused", "b2-fused"])
IDS = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [2, 2, 5, 7], [9, 1, 1, 3]])
N_PROBES = 3


def _incoming(n_batches, n_probes, *, device="cpu", seed=11):
    generator = torch.Generator().manual_seed(seed)
    return [[torch.randn(1, 4, 16, generator=generator).to(device)
             for _ in range(n_batches)] for _ in range(n_probes)]


def _owners(n_batches, n_probes):
    return [[SharedStateCotangents() for _ in range(n_batches)]
            for _ in range(n_probes)]


@pytest.fixture
def launches(monkeypatch):
    """Count every backward the roll launches."""
    count = [0]
    original = torch.autograd.backward

    def counted(*args, **kwargs):
        count[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "backward", counted)
    return count


def _steps(calls, fusion, batch_size):
    """Group the roll's calls into backward steps, in call order."""
    steps, current = [], None
    for probe, batch, launched in calls:
        key = (probe, batch // batch_size) if batch_size > 1 else (probe, batch)
        if current is None or current[0] != key:
            current = [key, launched, 0]
            steps.append(current)
        current[2] += 1
        assert current[1] == launched, "one step's rows roll together"
    return [(key, launched) for key, launched, _rows in steps]


# -- the overlap and its bound -------------------------------------------------

@REGIMES
def test_each_step_rolls_while_the_next_backward_runs(launches, batch_size, fusion):
    _model, runner = _toy_runner()
    batches = _toy_batches(runner, IDS)
    incoming = _incoming(len(batches), N_PROBES)
    calls = []
    render_free_layer_roll(
        runner, storage=None, batches=batches, layer=0,
        cotangents=_owners(len(batches), N_PROBES), n_probes=N_PROBES,
        incoming_entries=None,
        incoming_tensor=lambda probe, batch: incoming[probe][batch],
        roll=lambda tensor, batch, probe: calls.append((probe, batch, launches[0])),
        batch_size=batch_size, probe_fusion=fusion)
    steps = _steps(calls, fusion, batch_size)
    assert len(steps) == launches[0]
    # Storage None reads one window per pass: one pass when fused, one per
    # probe otherwise. The last step of each window rolls at its close.
    per_window = len(steps) if fusion else len(steps) // N_PROBES
    for position, (_key, launched) in enumerate(steps):
        last_of_window = (position + 1) % per_window == 0
        assert launched == position + (1 if last_of_window else 2), (
            f"step {position} rolled with {launched} backwards launched")


def test_the_last_step_of_every_window_rolls_before_the_window_closes():
    """The window-end hook runs inside each window, after its last item."""
    events = []

    class Storage:
        config = {"prefetch_batches": 2, "max_resident_bytes": 1 << 30}

        @contextmanager
        def prefetch(self, references):
            events.append(("open", len(references)))
            try:
                yield object()
            finally:
                events.append(("close",))

        def get(self, window, reference):
            return reference.tensor

    class Ref:
        def __init__(self, tensor):
            self.tensor = tensor
            self.tensor_bytes = tensor.numel() * tensor.element_size()

    class Batch:
        def __init__(self):
            self.activations_cpu = [Ref(torch.zeros(1, 4, 16))]

    batches = [Batch() for _ in range(4)]
    with prefetched_fused_boundary_windows(
            Storage(), batches, 0, window_batches=2,
            window_end=lambda: events.append(("end",))) as windows:
        for _indices, _boundary, _incoming in windows:
            events.append(("item",))
    assert events == [("open", 2), ("item",), ("end",), ("close",)] * 2
    events.clear()
    with prefetched_boundary_batches(
            Storage(), batches, 0,
            window_end=lambda: events.append(("end",))) as items:
        for _item in items:
            events.append(("item",))
    assert events == [("open", 2), ("item",), ("item",), ("end",), ("close",)] * 2


class _Ref:
    def __init__(self, name, tensor):
        self.name = name
        self.tensor = tensor
        self.tensor_bytes = tensor.numel() * tensor.element_size()

    def __repr__(self):
        return f"<{self.name}>"


class _PublishingStorage:
    """A read-window owner whose reads refuse a retired or unpublished entry.

    ``write`` publishes a rolled entry and retires the one it replaces, as
    the exact boundary owner's ``previous=`` rollover does. Staging the next
    layer's first window must name this roll's rows, never the entries they
    replace: a row still waiting to roll there would be staged stale.
    """

    def __init__(self, prefetch_batches):
        self.config = {"prefetch_batches": prefetch_batches,
                       "max_resident_bytes": 1 << 30}
        self.published = set()
        self.staged = []
        self._open = None

    def publish(self, reference):
        self.published.add(reference)
        return reference

    def write(self, tensor, name, previous):
        self.published.discard(previous)
        return self.publish(_Ref(name, tensor.detach().clone()))

    def _require(self, references, what):
        stale = [ref for ref in references if ref not in self.published]
        assert not stale, f"{what} names entries not published: {stale}"

    @contextmanager
    def prefetch(self, references):
        self._require(references, "a window read")
        self._open = window = object()
        try:
            yield window
        finally:
            self._open = None

    def get(self, window, reference):
        assert window is self._open and reference in self.published
        return reference.tensor

    def stage_produced_reads_ahead(self, following):
        self._require(following, "staging ahead")
        names = [ref.name for ref in following]
        if any(name.startswith("boundary-") and name.endswith("-0") for name in names):
            # The next layer's first window: its cotangents are this roll's.
            stale = [name for name in names
                     if name.startswith("cotangent-") and not name.endswith("-at-1")]
            assert not stale, f"staging the next layer names unrolled rows: {stale}"
        self.staged.append(names)


@REGIMES
def test_no_read_or_staging_names_a_row_the_roll_has_not_delivered(batch_size, fusion):
    """Stage A's shape: the roll replaces the entries it reads (``grad_outs``)."""
    _model, runner = _toy_runner()
    toy = _toy_batches(runner, IDS)
    storage = _PublishingStorage(prefetch_batches=2)
    for index, batch in enumerate(toy):
        hidden = batch.activations_cpu[0]
        batch.activations_cpu = [
            storage.publish(_Ref(f"boundary-{index}-{layer}", hidden.clone()))
            for layer in range(2)]
    grad_outs = [[storage.publish(_Ref(f"cotangent-{probe}-{index}-at-2", tensor))
                  for index, tensor in enumerate(row)]
                 for probe, row in enumerate(_incoming(len(toy), N_PROBES))]

    def roll(tensor, batch, probe):
        grad_outs[probe][batch] = storage.write(
            tensor, f"cotangent-{probe}-{batch}-at-1", grad_outs[probe][batch])

    render_free_layer_roll(
        runner, storage=storage, batches=toy, layer=1,
        cotangents=_owners(len(toy), N_PROBES), n_probes=N_PROBES,
        incoming_entries=grad_outs, incoming_tensor=None, roll=roll,
        then=(0, grad_outs if fusion else grad_outs[0]),
        batch_size=batch_size, probe_fusion=fusion)
    assert any(name.startswith("boundary-") and name.endswith("-0")
               for names in storage.staged for name in names), (
        "the last window staged the next layer's first")
    rolled = {ref.name for row in grad_outs for ref in row}
    assert rolled == {f"cotangent-{p}-{b}-at-1"
                      for p in range(N_PROBES) for b in range(len(toy))}


# -- failures ------------------------------------------------------------------

@REGIMES
def test_a_roll_that_raises_fails_the_layer_roll_and_nothing_rolls_after_it(
        batch_size, fusion):
    _model, runner = _toy_runner()
    batches = _toy_batches(runner, IDS)
    incoming = _incoming(len(batches), N_PROBES)
    calls = []

    def roll(tensor, batch, probe):
        calls.append((probe, batch))
        if len(calls) == 3:
            raise OSError("cotangent write failed")

    with pytest.raises(OSError, match="cotangent write failed"):
        render_free_layer_roll(
            runner, storage=None, batches=batches, layer=0,
            cotangents=_owners(len(batches), N_PROBES), n_probes=N_PROBES,
            incoming_entries=None,
            incoming_tensor=lambda probe, batch: incoming[probe][batch],
            roll=roll, batch_size=batch_size, probe_fusion=fusion)
    assert len(calls) == 3


@REGIMES
def test_the_last_steps_roll_error_is_raised_by_the_layer_roll(batch_size, fusion):
    """The final step rolls at the window's close, and its failure still lands."""
    _model, runner = _toy_runner()
    batches = _toy_batches(runner, IDS)
    incoming = _incoming(len(batches), N_PROBES)
    total = len(batches) * N_PROBES
    calls = []

    def roll(tensor, batch, probe):
        calls.append((probe, batch))
        if len(calls) == total:
            raise OSError("last cotangent write failed")

    with pytest.raises(OSError, match="last cotangent write failed"):
        render_free_layer_roll(
            runner, storage=None, batches=batches, layer=0,
            cotangents=_owners(len(batches), N_PROBES), n_probes=N_PROBES,
            incoming_entries=None,
            incoming_tensor=lambda probe, batch: incoming[probe][batch],
            roll=roll, batch_size=batch_size, probe_fusion=fusion)


@REGIMES
def test_a_failed_backward_is_raised_and_the_waiting_step_never_rolls(
        launches, monkeypatch, batch_size, fusion):
    _model, runner = _toy_runner()
    batches = _toy_batches(runner, IDS)
    incoming = _incoming(len(batches), N_PROBES)
    counted = torch.autograd.backward

    def failing(*args, **kwargs):
        if launches[0] == 1:
            launches[0] += 1
            raise RuntimeError("backward failed")
        return counted(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "backward", failing)
    calls = []
    with pytest.raises(RuntimeError, match="backward failed"):
        render_free_layer_roll(
            runner, storage=None, batches=batches, layer=0,
            cotangents=_owners(len(batches), N_PROBES), n_probes=N_PROBES,
            incoming_entries=None,
            incoming_tensor=lambda probe, batch: incoming[probe][batch],
            roll=lambda tensor, batch, probe: calls.append((probe, batch)),
            batch_size=batch_size, probe_fusion=fusion)
    # Step 0's rows waited behind the second backward, which failed; in no
    # regime is step 0 the last of its window, so nothing rolled.
    assert calls == []


# -- the exact writer's one host copy ------------------------------------------

def _write(root, name, tensor, monkeypatch):
    seen = {}
    original = cache_module.write_activation_cache_entry

    def spy(cache_dir, entry, inputs, **kwargs):
        seen["inputs"] = inputs
        return original(cache_dir, entry, inputs, **kwargs)

    monkeypatch.setattr(cache_module, "write_activation_cache_entry", spy)
    nbytes = tensor.numel() * tensor.element_size()
    reference = write_exact_activation_cache_entry(
        root, name, tensor, identity={"slot": name},
        max_tensor_bytes=nbytes, max_file_bytes=nbytes + 65536)
    return reference, seen["inputs"]


def test_a_compact_host_tensor_is_serialized_without_a_second_copy(tmp_path, monkeypatch):
    compact = torch.randn(1, 4, 16)
    reference, serialized = _write(tmp_path / "a", "entry", compact, monkeypatch)
    assert serialized.untyped_storage().data_ptr() == compact.untyped_storage().data_ptr()
    wide = torch.cat([torch.randn(1, 4, 16), compact.clone(), torch.randn(1, 4, 16)])
    for view in (wide[1:2], wide[0:1]):
        copied, serialized = _write(tmp_path / f"v{view.storage_offset()}", "entry",
                                    view, monkeypatch)
        assert serialized.untyped_storage().nbytes() == view.numel() * view.element_size()
        assert serialized.untyped_storage().data_ptr() != view.untyped_storage().data_ptr()
    same, _ = _write(tmp_path / "b", "entry", wide[1:2], monkeypatch)
    assert same.sha256 == reference.sha256
    assert (Path(same.path).read_bytes() == Path(reference.path).read_bytes())


def test_a_size_one_dimension_with_an_odd_stride_is_copied_to_the_same_bytes(
        tmp_path, monkeypatch):
    base = torch.randn(4, 16)
    odd = base.as_strided((1, 4, 16), (999, 16, 1))
    assert odd.is_contiguous() and odd.stride() != (64, 16, 1)
    written, serialized = _write(tmp_path / "odd", "entry", odd, monkeypatch)
    assert serialized.stride() == (64, 16, 1)
    plain, _ = _write(tmp_path / "plain", "entry", base.reshape(1, 4, 16).clone(),
                      monkeypatch)
    assert Path(written.path).read_bytes() == Path(plain.path).read_bytes()


# -- CUDA: pinned rows, same bytes ---------------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_a_pinned_row_serializes_to_the_pageable_rows_bytes(tmp_path, monkeypatch):
    pageable = torch.randn(1, 4, 16)
    pinned = torch.empty(1, 4, 16, pin_memory=True)
    pinned.copy_(pageable)
    one, _ = _write(tmp_path / "pageable", "entry", pageable, monkeypatch)
    two, serialized = _write(tmp_path / "pinned", "entry", pinned, monkeypatch)
    assert serialized.untyped_storage().data_ptr() == pinned.untyped_storage().data_ptr()
    assert Path(one.path).read_bytes() == Path(two.path).read_bytes()


@cuda
@REGIMES
def test_rows_are_pinned_only_for_a_roll_that_keeps_nothing(batch_size, fusion):
    _model, runner = _toy_runner(device="cuda")
    batches = _toy_batches(runner, IDS)
    incoming = _incoming(len(batches), N_PROBES, device="cpu")
    planes = {}
    for keeps in (True, False):
        rows = {}

        def roll(tensor, batch, probe):
            assert tensor.device.type == "cpu"
            assert tensor.is_pinned() is (not keeps)
            assert tensor.storage_offset() == 0
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
            rows[(probe, batch)] = tensor.clone()

        render_free_layer_roll(
            runner, storage=None, batches=batches, layer=0,
            cotangents=_owners(len(batches), N_PROBES), n_probes=N_PROBES,
            incoming_entries=None,
            incoming_tensor=lambda probe, batch: incoming[probe][batch],
            roll=roll, batch_size=batch_size, probe_fusion=fusion,
            roll_may_keep=keeps)
        planes[keeps] = rows
    assert planes[True].keys() == planes[False].keys()
    for key in planes[True]:
        assert torch.equal(planes[True][key].view(torch.uint8),
                           planes[False][key].view(torch.uint8))


@cuda
def test_the_cuda_default_roll_equals_the_pre_997_roll_bitwise():
    from test_stage_a_chain_regime import _legacy_render_free_layer_roll

    planes = []
    for roller in (render_free_layer_roll, _legacy_render_free_layer_roll):
        _model, runner = _toy_runner(device="cuda", dtype=torch.bfloat16)
        batches = _toy_batches(runner, IDS)
        incoming = [[tensor.to(torch.bfloat16) for tensor in row]
                    for row in _incoming(len(batches), N_PROBES)]
        rows = {}
        roller(runner, storage=None, batches=batches, layer=0,
               cotangents=_owners(len(batches), N_PROBES), n_probes=N_PROBES,
               incoming_entries=None,
               incoming_tensor=lambda probe, batch: incoming[probe][batch],
               roll=lambda tensor, batch, probe: rows.__setitem__(
                   (probe, batch), tensor.clone()),
               min_free_gib=0.0)
        planes.append(rows)
    assert planes[0].keys() == planes[1].keys()
    for key in planes[0]:
        assert torch.equal(planes[0][key].view(torch.uint8),
                           planes[1][key].view(torch.uint8))

