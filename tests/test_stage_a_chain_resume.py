"""Stage A chain resume: a relaunch of the same run (RobTand/prismaquant#1001).

A Stage A run that dies partway down its reverse chain is relaunched from
its lowest sealed checkpoint. The claims these tests hold it to:

* An interrupted run resumed under the same implementation writes what the
  uninterrupted run writes: every rolled cotangent, every checkpoint and
  entry file, the receipt, and every band and slice. Bands sealed before the
  resume and after it form one set.
* The interrupted run keeps its forward boundary entries and its sealed
  checkpoints when it fails, and a resume removes only the interrupted
  attempt's rolling entries and sets partial checkpoint directories aside.
* A resume under a different implementation needs dev mode and an explicit
  declaration. The run header keeps the original implementation; the
  declaration is in the receipt and in every band sealed below the switch.
* Every difference in what decides the chain's bytes refuses, before
  anything is removed or renamed.

The fixture is a five-layer dense model at stride 2, so the run seals the
tail checkpoint 5 and checkpoints 4 and 2, and a four-layer model whose
layers share K/V state, so a checkpoint carries a non-empty shared adjoint.
The equality test also runs the dense model at stride 1, resuming from
checkpoint 1, and bound to a forward-recovery capsule, the way R13 launches.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace
import uuid

import pytest
import torch

from prismaquant import joint_cost_stage_a as stage_a
from prismaquant import stage_a_chain_resume as resume_mod
from prismaquant.cost_stage_checkpoint import canonical_json_bytes
from prismaquant.cost_streaming import StreamedBoundaryArtifacts, StreamedCausalLM
from prismaquant.joint_adjoint_band import build_band_receipt
from prismaquant.joint_adjoint_checkpoints import (
    adjoint_receipt_path,
    adjoint_slice_sha256,
    adjoint_space,
    band_set,
    checkpoint_directory,
    load_checkpoint_shared_states,
    stage_a_run_header,
    stage_a_slice,
    unpack_shared_states,
    write_adjoint_receipt,
)
from prismaquant.joint_cost_stage_a import AdjointIdentityRefused
from prismaquant.model_profiles.default import DefaultProfile
from prismaquant.stage_a_chain_resume import (
    RESUME_COMPATIBILITY_KEY,
    ChainResumeRefused,
    chain_state_path,
)

from test_joint_cost_quantum_runtime import _boundary_policy, _execution
from test_layer_major_boundary_capture import draw
from test_streamed_cost_checkpoints import (
    _DenseLayer,
    _DenseTinyLM,
    _FakeStreamingContext,
    _model_identity,
)

ONE, TWO, THREE = "1" * 64, "2" * 64, "3" * 64
CAMPAIGN = {"unit_roster_sha256": "a" * 64, "plan_sha256": "b" * 64,
            "prepared_sha256": "c" * 64, "read_manifest_sha256": "d" * 64}
DIGESTS = {"plan_sha256": "b" * 64, "prepared_sha256": "c" * 64,
           "read_manifest_sha256": "d" * 64, "stride_source": None,
           "unit_roster_sha256": "a" * 64}
N_PROBES = 4
#: A resident cap wide enough for a fused window, so the probe-fusion
#: mismatch refuses on identity rather than on fit.
CAP = 2 * (1 + N_PROBES) * 256 + 256


@pytest.fixture(autouse=True)
def _offline_tier_policy():
    """Run with no staged-tier policy, as the runtime tests do (PQ #845)."""
    from prismaquant.staged_tier_policy import deactivate_staged_tier_policy_for_tests
    deactivate_staged_tier_policy_for_tests()
    yield
    deactivate_staged_tier_policy_for_tests()


class _Interrupted(Exception):
    """The fixture's stand-in for a killed or failed Stage A action."""


class _DeepTinyLM(_DenseTinyLM):
    def __init__(self, state=None, *, layers=5):
        super().__init__()
        self.model.layers = torch.nn.ModuleList([_DenseLayer(16) for _ in range(layers)])
        if state is not None:
            self.load_state_dict(state)


def _dense_runner():
    torch.manual_seed(85)
    state = _DeepTinyLM().eval().state_dict()
    model = _DeepTinyLM(state).eval()
    for layer in model.model.layers:
        layer._fixture_requires_stream_residency = True
    context = _FakeStreamingContext(model)
    runner = StreamedCausalLM(context, DefaultProfile())
    return _streamed(runner, context)


def _shared_runner():
    from prismaquant.model_profiles.gemma4 import Gemma4Profile
    from test_kv_cotangent_path import SHARED_SPECS, _ToyModel

    toy = _ToyModel(SHARED_SPECS, hidden=16, seed=91)
    toy.config = SimpleNamespace(layer_types=())
    model = torch.nn.Module()
    model.model = toy
    model.lm_head = toy.lm_head
    model.eval()
    context = _FakeStreamingContext(model)
    return _streamed(StreamedCausalLM(context, Gemma4Profile()), context)


def _streamed(runner, context):
    install = context.install
    context.install = lambda layer, *, require_prefetched=False, prefetch_following=True: \
        install(layer, require_prefetched=require_prefetched)
    context.settle_prefetched_layers = lambda *a, **kw: None
    runner.require_prefetched_residency = True
    runner.prefetch_lookahead = 1
    return runner


def _at(boundary, probe, batch):
    """Interrupt right after the rolled cotangent at these coordinates is written."""
    return lambda kw: (kw.get("probe_index"), kw["boundary_index"], kw["batch_index"]) == (
        probe, boundary, batch)


def _run(root, monkeypatch, *, model="dense", stride=2, implementation=ONE,
         chain_resume=None, interrupt=None, partial_at=None, writes=None,
         identities=None, campaign=None, calib=None, execution=None,
         generation=1001, forward_refs=None, runner_factory=None, **core):
    """One fixture Stage A invocation into ``root``; returns the receipt.

    ``interrupt`` raises after the write it matches; ``partial_at`` leaves a
    checkpoint directory half written at that boundary and raises. Every
    rolled cotangent the invocation writes is appended to ``writes`` as
    ``((boundary, probe, batch), sha256)``; every forward boundary reference
    lands in ``forward_refs`` by ``(boundary, batch)``. The generation id is
    pinned, so two runs into one path write equal bytes. ``runner_factory``
    builds the runner in place of the fixture's own.
    """
    if runner_factory is not None:
        runner = runner_factory()
    else:
        runner = _dense_runner() if model == "dense" else _shared_runner()
    if execution is None:
        execution = _execution(root)
        policy = _boundary_policy(root / "boundaries")
        policy["max_resident_bytes"] = CAP if model == "dense" else 1 << 16
        execution["boundary_storage"] = policy
    write = StreamedBoundaryArtifacts.write
    bind = StreamedBoundaryArtifacts.bind
    checkpoint = stage_a.write_checkpoint_with_snapshot

    def recorded(self, tensor, **kw):
        reference = write(self, tensor, **kw)
        if kw.get("probe_index") is None and forward_refs is not None:
            forward_refs[kw["boundary_index"], kw["batch_index"]] = reference
        if kw.get("probe_index") is not None and writes is not None:
            data = tensor.detach().to("cpu").contiguous()
            writes.append(((kw["boundary_index"], kw["probe_index"], kw["batch_index"]),
                           hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()))
        if interrupt is not None and interrupt(kw):
            raise _Interrupted(kw)
        return reference

    def bound(self, identity, **kw):
        if identities is not None:
            identities.append(json.loads(json.dumps(identity)))
        return bind(self, identity, **kw)

    at_death = []

    def partial(storage, space, *, boundary, **kw):
        if boundary == partial_at:
            # The roll has written the cotangents into the open checkpoint
            # (PQ #1002); the action is killed sealing it, with one file half
            # written. A kill runs no close, and the owner's close is what
            # disposes a failed attempt's directory, so the directory as it
            # is now is what a relaunch finds: kept aside and put back below.
            directory = checkpoint_directory(space, boundary)
            (directory / "entries").mkdir(parents=True, exist_ok=True)
            (directory / "entries" / "cotangent-0-0.pt").write_bytes(b"half written")
            kept = root.parent / f"{root.name}-boundary-{boundary}-at-death"
            shutil.copytree(directory, kept)
            at_death.append((kept, directory))
            raise _Interrupted(f"died writing checkpoint {boundary}")
        return checkpoint(storage, space, boundary=boundary, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(StreamedBoundaryArtifacts, "write", recorded)
        patch.setattr(StreamedBoundaryArtifacts, "bind", bound)
        patch.setattr(stage_a, "write_checkpoint_with_snapshot", partial)
        patch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=generation))
        try:
            receipt = stage_a.run_adjoint_capture_core(
                runner, draw() if calib is None else calib, execution=execution,
                output_root=root, stride=stride,
                source_model_identity=_model_identity("joint-source"),
                implementation_sha256=implementation, chain_resume=chain_resume,
                **{**CAMPAIGN, **(campaign or {})}, **core)
        finally:
            for kept, directory in at_death:
                assert not directory.exists(), "the owner's close kept a failed attempt"
                kept.rename(directory)
    return json.loads(json.dumps(receipt))


def _interrupted(root, monkeypatch, **kw):
    with pytest.raises(_Interrupted):
        _run(root, monkeypatch, **kw)


def _state_sha256(root):
    return hashlib.sha256(chain_state_path(adjoint_space(root)).read_bytes()).hexdigest()


def _resume(root, *, declaration=None, resume_from=None, sha256=None):
    return {"chain_state_sha256": sha256 or _state_sha256(root),
            "declaration": declaration, "resume_from": resume_from}


def _tree(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def _names(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def _without(document, *paths):
    document = json.loads(json.dumps(document))
    for path in paths:
        *parents, leaf = path.split(".")
        node = document
        for key in parents:
            node = node[key]
        node.pop(leaf, None)
    return document


VARIABLE = ("telemetry", "dev_mode", "retention.telemetry")


def _generation(root):
    [path] = sorted((adjoint_space(root) / "exact-boundaries").glob("*/generation.json"))
    return path


def _merged(writes):
    """Every rolled plane, checking that a rewritten entry has equal bytes."""
    merged = {}
    for key, digest in writes:
        assert merged.setdefault(key, digest) == digest, f"{key} rewritten differently"
    return merged


def _band(root, boundary, sources, stride=2):
    """``sources`` is ``{"bind_identity": ...}`` or ``{"forward_recovery": ...}``."""
    return json.loads(json.dumps(build_band_receipt(
        output_root=root, boundary=boundary, stride_value=stride, **sources, **DIGESTS)))


def _capsule(tmp_path, monkeypatch):
    """R9's forward pass, contained after boundary 1, frozen into a capsule.

    The same shape as ``test_stage_a_bands``: the capsule is what R13's own
    launch binds, so a chain resume must continue a capsule-bound run too.
    """
    from prismaquant import joint_forward_resume as recovery_mod
    from test_joint_forward_resume import _bound, _chain_sdk, _owner_spool

    monkeypatch.setattr(recovery_mod, "_sdk", _chain_sdk)
    refs, identities = {}, []
    last = len(draw()) - 1
    _interrupted(tmp_path / "r9", monkeypatch, generation=9, forward_refs=refs,
                 identities=identities,
                 interrupt=lambda kw: kw.get("probe_index") is None and (
                     kw["boundary_index"], kw["batch_index"]) == (1, last))
    spool, instance, template = _owner_spool(tmp_path, "e" * 64, refs)
    ref = refs[0, 0]
    capsule = tmp_path / "r9-to-r10.json"
    recovery_mod.build_forward_recovery(specification={
        "schema": recovery_mod.SCHEMA, "queue_root": str(tmp_path / "queue"),
        "instance": instance, "template": template,
        "session": json.loads(ref.metadata_json)["identity"]["session"],
        "original_bind_identity": identities[-1],
        "campaign_identity": {**CAMPAIGN, "campaign_scope": None},
        "implementation_compatibility": {"original": ONE, "recovery": TWO,
                                         "scope": "forward-identical-memory-only"},
        "n_batches": len(draw()), "entry_shape": list(ref.shape), "entry_dtype": ref.dtype},
        spool_directory=spool, output=capsule, frontier=1)
    return _bound(capsule)


def _slices(band):
    return {layer: canonical_json_bytes(stage_a_slice(band, layer), where="slice")
            for layer in band["band"]["layers"]}


# -- same implementation: a bitwise continuation ------------------------------

SHAPES = {
    # Killed mid-roll of layer 3: checkpoints 5 and 4 are sealed.
    "mid-chain": dict(interrupt=_at(3, 1, 2), resumes_from=4),
    # Killed in the last roll, after the last checkpoint: 5, 4 and 2 sealed.
    "after-last-checkpoint": dict(interrupt=_at(0, 2, 1), resumes_from=2),
    # Killed while writing checkpoint 2: its directory is left half written.
    "partial-checkpoint": dict(partial_at=2, resumes_from=4),
    # Failed mid-roll of layer 2, whose checkpoint is open and holds the
    # cotangents rolled so far (PQ #1002): the attempt is retained, and the
    # owner's close disposes its directory, so nothing is left to set aside.
    "mid-checkpoint-roll": dict(interrupt=_at(2, 1, 2), resumes_from=4),
    # SIGKILL: no exception handler ran, the generation still says running.
    "killed": dict(interrupt=_at(1, 0, 4), resumes_from=2, killed=True),
    # Stride 1: the resume point is checkpoint 1, so the only roll left is
    # layer 0's, which writes without reading back.
    "last-layer-only": dict(interrupt=_at(0, 2, 1), resumes_from=1, stride=1),
    # R13's shape: a run bound to a forward-recovery capsule (boundaries 0
    # and 1 are the capsule's), killed mid-chain.
    "capsule-bound": dict(interrupt=_at(3, 1, 2), resumes_from=4, capsule=True),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_resumed_run_writes_what_the_uninterrupted_run_writes(tmp_path, monkeypatch, shape):
    """Planes, files, receipt, bands and slices all equal the uninterrupted run's.

    Both runs write into one path, one after the other (the first moved
    aside), under one pinned generation id. The resumed run's only extra
    files are its resume record and any partial checkpoint it set aside.
    """
    spec = dict(SHAPES[shape])
    resumes_from = spec.pop("resumes_from")
    killed = spec.pop("killed", False)
    stride = spec.pop("stride", 2)
    run = {"stride": stride}
    if spec.pop("capsule", False):
        run.update(forward_recovery=_capsule(tmp_path, monkeypatch), implementation=TWO)
    root = tmp_path / "run"
    identities, baseline_writes = [], []
    baseline = _run(root, monkeypatch, writes=baseline_writes, identities=identities, **run)
    sources = ({"forward_recovery": run["forward_recovery"]} if "forward_recovery" in run
               else {"bind_identity": identities[-1]})
    marks = {1: [5, 4, 3, 2, 1], 2: [5, 4, 2]}[stride]
    assert baseline["stride"]["boundaries"] == marks
    assert [c["boundary"] for c in baseline["checkpoints"]] == marks
    baseline_bands = {b: _band(root, b, sources, stride) for b in marks}
    aside = tmp_path / "baseline"
    root.rename(aside)

    writes = []
    _interrupted(root, monkeypatch, writes=writes, **run, **spec)
    space = adjoint_space(root)
    generation = _generation(root)
    status = json.loads(generation.read_text())
    # The failed attempt keeps its forward entries, its sealed checkpoints
    # and the chain state; only the receipt is missing.
    assert status["status"] == "failed"
    assert not adjoint_receipt_path(space).exists()
    assert chain_state_path(space).is_file()
    entries = generation.parent / "entries"
    own = range(2 if "forward_recovery" in run else 0, 5)
    assert {f"boundary-{b}-{k}-at-{k}.pt" for b in range(5) for k in own} <= {
        path.name for path in entries.iterdir()}
    sealed = sorted(int(path.parent.name.removeprefix("boundary-"))
                    for path in (space / "checkpoints").glob("boundary-*/checkpoint.json"))
    assert min(sealed) == resumes_from
    # A sealed (v2, PQ #1036) checkpoint names the owner's own cotangent
    # entries: those are the checkpoint's, not rolling leftovers.
    pinned = {Path(row["path"]).name
              for boundary in sealed
              for row in json.loads((space / "checkpoints" / f"boundary-{boundary:03d}"
                                     / "checkpoint.json").read_text())["activation_entries"]}
    assert pinned and all((entries / name).is_file() for name in pinned)
    leftovers = sorted(path.name for path in entries.iterdir()
                       if "cotangent" in path.name and path.name not in pinned)
    assert leftovers, "the interrupted attempt left rolling entries behind"
    if killed:
        status["status"] = "running"
        generation.write_text(json.dumps(status))
    early_bands = {b: _band(root, b, sources, stride) for b in sealed}

    resumed = _run(root, monkeypatch, writes=writes, **run,
                   chain_resume=_resume(root, resume_from=resumes_from))

    assert _merged(writes) == _merged(baseline_writes)
    record = json.loads((space / "resumes" / "resume-001.json").read_text())
    assert record["index"] == 1 and record["switch_checkpoint"] == resumes_from
    assert record["compatibility"] is None
    assert record["implementation_sha256"] == run.get("implementation", ONE)
    assert record["removed_rolling_entries"] == len(leftovers)
    assert all((entries / name).is_file() for name in pinned)
    set_aside = ["boundary-002.partial-resume-001"] if shape == "partial-checkpoint" else []
    assert record["partial_checkpoints_set_aside"] == set_aside
    assert resumed["telemetry"]["chain_resume"]["switch_checkpoint"] == resumes_from

    ours, theirs = _tree(root), _tree(aside)
    partial = {key for key in ours for name in set_aside
               if key.startswith(f"layer-quanta/adjoint/checkpoints/{name}/")}
    assert bool(partial) == bool(set_aside)
    assert not any(key.endswith("/checkpoint.json") for key in partial)
    extra = {"layer-quanta/adjoint/resumes/resume-001.json"} | partial
    names = {str(generation.relative_to(root))}
    assert set(ours) - set(theirs) == extra and set(theirs) <= set(ours)
    assert {k: v for k, v in ours.items() if k not in extra | names} == \
        {k: v for k, v in theirs.items() if k not in names}
    for name in names:
        assert _without(json.loads((root / name).read_text()), "telemetry") == \
            _without(json.loads((aside / name).read_text()), "telemetry")
    assert RESUME_COMPATIBILITY_KEY not in resumed
    assert _without(resumed, *VARIABLE) == _without(baseline, *VARIABLE)

    # Bands sealed before the resume and after it form one set, and every
    # band and slice is the uninterrupted run's.
    bands = {**{b: _band(root, b, sources, stride) for b in marks if b not in early_bands},
             **early_bands}
    assert sorted(band_set(bands.values())) == sorted(marks)
    for boundary, band in bands.items():
        assert canonical_json_bytes(band, where="band") == \
            canonical_json_bytes(baseline_bands[boundary], where="band")
        assert stage_a_run_header(band) == stage_a_run_header(resumed)
        for layer, data in _slices(band).items():
            assert data == canonical_json_bytes(stage_a_slice(resumed, layer), where="slice")
            assert adjoint_slice_sha256(stage_a_slice(band, layer)) == \
                adjoint_slice_sha256(stage_a_slice(baseline, layer))


def _content(value):
    """A shared state's content, independent of how pickle spelled it."""
    if isinstance(value, torch.Tensor):
        data = value.detach().to("cpu").contiguous().reshape(-1)
        return ["tensor", str(value.dtype), list(value.shape), bool(value.requires_grad),
                hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()]
    if isinstance(value, dict):
        return ["dict", [[repr(key), _content(item)] for key, item in value.items()]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_content(item) for item in value]]
    return ["value", repr(value)]


def _shared_contents(receipt, root, where):
    """The receipt with each shared-state pickle replaced by its content.

    A pickled tensor's storage key is its storage's address (Torch's legacy
    ``Storage.__reduce__``), so two uninterrupted runs of a model with shared
    state already write different shared-state bytes for equal tensors. The
    run's own cotangent entries carry no such key and stay byte-compared.
    """
    receipt = json.loads(json.dumps(receipt))
    for record in receipt["checkpoints"]:
        record.pop("cotangent_sha256")
        for entry in record["shared_state_entries"]:
            path = Path(entry.pop("path").replace(str(root), str(where)))
            entry.pop("sha256"), entry.pop("file_bytes")
            if entry["name"] == "shared-states":
                # PQ #1037: a packed checkpoint's one file holds every state.
                entry["content"] = [
                    (name, _content(pickle.loads(member)))
                    for name, member in unpack_shared_states(path.read_bytes())]
            else:
                entry["content"] = _content(pickle.loads(path.read_bytes()))
    return receipt


def test_a_shared_adjoint_checkpoint_resumes_bitwise(tmp_path, monkeypatch):
    """A checkpoint with a non-empty shared adjoint restores it exactly.

    Layers 2 and 3 read layer 1's K/V, so checkpoint 2 carries the
    cotangent those two consumers accumulated for layer 1. A resume from 2
    must hand it back before layer 1's roll: every rolled plane, every
    cotangent entry and every shared state equal the uninterrupted run's.
    """
    root = tmp_path / "run"
    baseline_writes = []
    baseline = _run(root, monkeypatch, model="shared", stride=1, writes=baseline_writes)
    assert baseline["stride"]["boundaries"] == [4, 3, 2, 1]
    aside = tmp_path / "baseline"
    root.rename(aside)

    writes = []
    _interrupted(root, monkeypatch, model="shared", stride=1, writes=writes,
                 interrupt=_at(1, 1, 0))
    space = adjoint_space(root)
    record = json.loads((checkpoint_directory(space, 2) / "checkpoint.json").read_text())
    shared_adjoint, _pass = load_checkpoint_shared_states(space, record)
    assert any(state["accumulators"] for state in shared_adjoint.values()), \
        "checkpoint 2 carries no shared adjoint: the fixture does not test the restore"

    resumed = _run(root, monkeypatch, model="shared", stride=1, writes=writes,
                   chain_resume=_resume(root, resume_from=2))
    assert _merged(writes) == _merged(baseline_writes)
    ours = _without(_shared_contents(resumed, root, root), *VARIABLE)
    theirs = _without(_shared_contents(baseline, root, aside), *VARIABLE)
    assert ours == theirs
    ours, theirs = _tree(root), _tree(aside)
    cotangents = {name for name in theirs if name.endswith(".pt")}
    assert cotangents and {k: ours[k] for k in cotangents} == {k: theirs[k] for k in cotangents}


# -- a different implementation: dev mode and an explicit declaration ---------

def test_a_declared_implementation_switch_is_recorded_outside_the_header(
        tmp_path, monkeypatch):
    """The receipt and every band below the switch carry the declaration.

    The run is interrupted under implementation 1, resumed under 2 with a
    declaration and interrupted again, then resumed under 2 once more, which
    is a plain continuation: the last resume sealed its checkpoints.
    """
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    baseline_writes = []
    _run(tmp_path / "baseline", monkeypatch, writes=baseline_writes)
    root = tmp_path / "run"
    identities, writes = [], []
    _interrupted(root, monkeypatch, writes=writes, identities=identities,
                 interrupt=_at(3, 0, 0))
    identity = identities[-1]
    switch = {"from": ONE, "to": TWO}
    _interrupted(root, monkeypatch, implementation=TWO, writes=writes,
                 chain_resume=_resume(root, declaration=switch, resume_from=4),
                 interrupt=_at(1, 3, 3))
    # Checkpoint 2 was sealed by implementation 2: a relaunch under 1 is a
    # switch back, which certified mode refuses without its own declaration
    # (dev mode resumes it undeclared since PQ #1147).
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(AdjointIdentityRefused, match="explicit"):
        _run(root, monkeypatch, chain_resume=_resume(root))
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    resumed = _run(root, monkeypatch, implementation=TWO, writes=writes,
                   chain_resume=_resume(root, resume_from=2))

    assert _merged(writes) == _merged(baseline_writes)
    declared = {"schema": resume_mod.RESUME_COMPATIBILITY_SCHEMA,
                "scope": "stage-a-chain-continuation",
                "from_implementation_sha256": ONE, "to_implementation_sha256": TWO,
                "switch_checkpoint": 4}
    assert resumed["run_identity"]["implementation_sha256"] == ONE
    assert resumed[RESUME_COMPATIBILITY_KEY] == [declared]
    space = adjoint_space(root)
    records = [json.loads((space / "resumes" / f"resume-{i:03d}.json").read_text())
               for i in (1, 2)]
    assert [r["compatibility"] for r in records] == [declared, None]
    assert [r["implementation_sha256"] for r in records] == [TWO, TWO]
    assert [r["switch_checkpoint"] for r in records] == [4, 2]

    bands = {b: _band(root, b, {"bind_identity": identity}) for b in (5, 4, 2)}
    assert [RESUME_COMPATIBILITY_KEY in bands[b] for b in (5, 4)] == [False, False]
    assert bands[2][RESUME_COMPATIBILITY_KEY] == [declared]
    assert sorted(band_set(bands.values())) == [2, 4, 5]
    for band in bands.values():
        assert stage_a_run_header(band) == stage_a_run_header(resumed)
        for layer, data in _slices(band).items():
            assert data == canonical_json_bytes(stage_a_slice(resumed, layer), where="slice")


# -- refusals: before anything is removed or renamed ---------------------------

def _refusal_kwargs(root, name):
    """The relaunch ``name`` makes of the interrupted run at ``root``."""
    resume = _resume(root)
    execution = _execution(root)
    policy = _boundary_policy(root / "boundaries")
    policy["max_resident_bytes"] = CAP
    execution["boundary_storage"] = policy
    other_draw = draw().clone()
    other_draw[0, 0] = 5
    cases = {
        "batch-size": dict(chain_batch_size=2),
        "probe-fusion": dict(chain_probe_fusion=True),
        "plan": dict(campaign={"plan_sha256": "e" * 64}),
        "prepared": dict(campaign={"prepared_sha256": "e" * 64}),
        "read-manifest": dict(campaign={"read_manifest_sha256": "e" * 64}),
        "unit-roster": dict(campaign={"unit_roster_sha256": "e" * 64}),
        "stride": dict(stride=1),
        "artifact-budget": dict(boundary_artifact_bytes=(1 << 24) - 256),
        "read-window": dict(execution={**execution, "boundary_storage": {
            **policy, "prefetch_batches": 1}}),
        "calibration": dict(calib=other_draw),
        "probes": dict(execution={**execution, "n_probes": 3}),
        "seed": dict(execution={**execution, "seed_base": 7001}),
        "arithmetic": dict(arithmetic_extra={"container_content_sha256": "f" * 64}),
        "chain-state-digest": dict(chain_resume={**resume, "chain_state_sha256": "0" * 64}),
        "not-the-lowest-checkpoint": dict(chain_resume={**resume, "resume_from": 5}),
        "implementation-undeclared": dict(implementation=TWO),
        "declaration-without-switch": dict(
            chain_resume={**resume, "declaration": {"from": ONE, "to": TWO}}),
        "declaration-of-another-switch": dict(
            implementation=TWO, dev_mode=True,
            chain_resume={**resume, "declaration": {"from": THREE, "to": TWO}}),
        "certified-mode": dict(
            implementation=TWO, dev_mode=False,
            chain_resume={**resume, "declaration": {"from": ONE, "to": TWO}}),
    }
    return cases[name]


#: Each relaunch and the reason it must refuse for: a refusal for another
#: reason would pass a bare ``pytest.raises`` and prove nothing.
REFUSALS = {
    "batch-size": "differs in run_identity$",
    "probe-fusion": "differs in run_identity$",
    "plan": "differs in run_identity$",
    "prepared": "differs in run_identity$",
    "read-manifest": "differs in run_identity$",
    "unit-roster": "differs in run_identity$",
    "stride": "differs in stride$",
    "artifact-budget": "another boundary storage policy",
    "read-window": "another boundary storage policy",
    "calibration": "differs in run_identity, bind_identity$",
    "probes": "differs in run_identity, bind_identity$",
    "seed": "differs in run_identity, bind_identity$",
    "arithmetic": "differs in arithmetic$",
    "chain-state-digest": "does not have the pinned digest",
    "not-the-lowest-checkpoint": "the lowest sealed checkpoint of the run is 4",
    "implementation-undeclared": "needs an explicit --resume-implementation-compatibility",
    "declaration-without-switch": "names a switch that is not happening",
    "declaration-of-another-switch": "is not the switch this relaunch makes",
    "certified-mode": "requires PRISMAQUANT_DEV_MODE=1",
}


def _refused_untouched(root, monkeypatch, match, **kw):
    """Refuse the relaunch for ``match``; nothing but the generation status changes."""
    before_names, before = _names(root), _tree(root)
    generation = str(_generation(root).relative_to(root))
    with pytest.raises(AdjointIdentityRefused, match=match):
        _run(root, monkeypatch, **kw)
    assert _names(root) == before_names, "a refused resume removed or renamed a path"
    after = _tree(root)
    assert {k: v for k, v in after.items() if k != generation} == \
        {k: v for k, v in before.items() if k != generation}
    return json.loads((root / generation).read_text())["status"]


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_every_regime_difference_refuses_before_anything_is_removed(
        tmp_path, monkeypatch, name):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    kw = _refusal_kwargs(root, name)
    kw.setdefault("chain_resume", _resume(root))
    dev_mode = kw.pop("dev_mode", None)
    if dev_mode is True:
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    elif dev_mode is False:
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    assert _refused_untouched(root, monkeypatch, REFUSALS[name], **kw) == "failed"


def test_a_matmul_precision_difference_refuses(tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    precision = torch.get_float32_matmul_precision()
    other = "medium" if precision != "medium" else "high"
    torch.set_float32_matmul_precision(other)
    try:
        _refused_untouched(root, monkeypatch, "differs in arithmetic$",
                           chain_resume=_resume(root))
    finally:
        torch.set_float32_matmul_precision(precision)


def test_a_bf16_reduction_difference_refuses(tmp_path, monkeypatch):
    """The flag joins the run identity and the chain arithmetic (PQ #1028)."""
    root = tmp_path / "run"
    matmul = torch.backends.cuda.matmul
    monkeypatch.setattr(matmul, "allow_bf16_reduced_precision_reduction", True)
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    monkeypatch.setattr(matmul, "allow_bf16_reduced_precision_reduction", False)
    _refused_untouched(root, monkeypatch, "differs in run_identity, arithmetic$",
                       chain_resume=_resume(root))


def test_the_run_identity_carries_the_bf16_flag_only_when_it_is_off(tmp_path, monkeypatch):
    from prismaquant.matmul_arithmetic import BF16_REDUCTION_FIELD

    matmul = torch.backends.cuda.matmul
    monkeypatch.setattr(matmul, "allow_bf16_reduced_precision_reduction", True)
    default = _run(tmp_path / "default", monkeypatch)["run_identity"]
    monkeypatch.setattr(matmul, "allow_bf16_reduced_precision_reduction", False)
    off = _run(tmp_path / "off", monkeypatch)["run_identity"]
    assert BF16_REDUCTION_FIELD not in default
    assert off[BF16_REDUCTION_FIELD] is False
    assert {key: value for key, value in off.items() if key != BF16_REDUCTION_FIELD} == default


def test_another_capsule_refuses_and_leaves_the_run_resumable(tmp_path, monkeypatch):
    """The capsule check runs after the rebind; the run stays resumable."""
    from prismaquant import joint_forward_resume

    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    with monkeypatch.context() as patch:
        patch.setattr(joint_forward_resume, "load_forward_recovery",
                      lambda *a, **kw: SimpleNamespace(
                          receipt_binding={"capsule": "another"}, frontier=0,
                          n_batches=len(draw()), records={}))
        assert _refused_untouched(root, monkeypatch, "another forward-recovery capsule",
                                  chain_resume=_resume(root)) == "failed"
    resumed = _run(root, monkeypatch, chain_resume=_resume(root, resume_from=4))
    assert [c["boundary"] for c in resumed["checkpoints"]] == [5, 4, 2]


def test_a_checkpoint_gap_refuses(tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(0, 2, 1))
    four = checkpoint_directory(adjoint_space(root), 4)
    four.rename(four.with_name(four.name + ".gone"))
    _refused_untouched(root, monkeypatch, "a gap cannot be resumed across",
                       chain_resume=_resume(root))


def test_a_completed_run_and_a_completed_generation_refuse(tmp_path, monkeypatch):
    root = tmp_path / "run"
    write_adjoint_receipt(adjoint_space(root), _run(root, monkeypatch))
    _refused_untouched(root, monkeypatch, "the run completed",
                       chain_resume=_resume(root))

    root = tmp_path / "complete-generation"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    generation = _generation(root)
    status = json.loads(generation.read_text())
    status["status"] = "complete"
    generation.write_text(json.dumps(status))
    assert _refused_untouched(root, monkeypatch, "only an interrupted run resumes",
                              chain_resume=_resume(root)) == "complete"


def test_a_run_interrupted_before_its_tail_checkpoint_has_no_chain_to_resume(
        tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(5, 0, 0))
    assert not chain_state_path(adjoint_space(root)).exists()
    before = _names(root)
    with pytest.raises(AdjointIdentityRefused, match="no chain state"):
        _run(root, monkeypatch, chain_resume=_resume(root, sha256="0" * 64))
    assert _names(root) == before


def test_a_fresh_run_refuses_another_runs_chain_state_before_its_forward_pass(
        tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    before = _names(root)
    with pytest.raises(AdjointIdentityRefused, match="another run's chain state"):
        _run(root, monkeypatch, interrupt=lambda kw: pytest.fail("the forward pass ran"))
    assert _names(root) == before


def test_the_interrupted_owner_must_be_contained(tmp_path, monkeypatch):
    """A resume asks PrismaBuild that the owner that wrote the attempt is gone."""
    producer = {"owner_action_key": "e" * 64, "owner_attempt": 1,
                "queue_root": str(tmp_path / "queue")}
    monkeypatch.setattr(resume_mod, "producer_binding",
                        lambda publication: dict(producer))
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    state = json.loads(chain_state_path(adjoint_space(root)).read_text())
    assert state["producer"] == producer

    def live(owner):
        raise ChainResumeRefused("the owner is still running")

    monkeypatch.setattr(resume_mod, "require_producer_contained", live)
    _refused_untouched(root, monkeypatch, "the owner is still running",
                       chain_resume=_resume(root))
    asked = []
    monkeypatch.setattr(resume_mod, "require_producer_contained", asked.append)
    _run(root, monkeypatch, chain_resume=_resume(root))
    assert asked == [producer]


def test_containment_is_the_forward_recovery_check(monkeypatch):
    from prismaquant import joint_forward_resume

    calls = []
    sdk = {"pool": SimpleNamespace(PoolQueue=lambda root: ("queue", root))}
    monkeypatch.setattr(joint_forward_resume, "_sdk", lambda: sdk)

    def contained(queue, instance, given):
        calls.append((queue, instance, given))
        raise joint_forward_resume.ForwardRecoveryRefused("owner still running")

    monkeypatch.setattr(joint_forward_resume, "require_contained", contained)
    producer = {"owner_action_key": "e" * 64, "owner_attempt": 2, "queue_root": "/q"}
    with pytest.raises(ChainResumeRefused, match="not contained"):
        resume_mod.require_producer_contained(producer)
    assert calls == [(("queue", "/q"), {"owner_action_key": "e" * 64,
                                        "owner_attempt": 2}, sdk)]


# -- the command line -----------------------------------------------------------

def test_the_resume_flags_need_the_chain_state_digest(capsys):
    with pytest.raises(SystemExit) as exit_info:
        stage_a.main(["--plan", "/p", "--plan-sha256", "b" * 64, "--prepared", "/q",
                      "--prepared-sha256", "c" * 64, "--output-root", "/o",
                      "--resume-from-checkpoint", "4"])
    assert exit_info.value.code == 2
    assert "need --resume-chain-state-sha256" in capsys.readouterr().err
    args = SimpleNamespace(resume_chain_state_sha256="f" * 64, resume_from_checkpoint=4,
                           resume_implementation_compatibility=f"{ONE}:{TWO}")
    assert stage_a._chain_resume_argument(args) == {
        "chain_state_sha256": "f" * 64, "resume_from": 4,
        "declaration": {"from": ONE, "to": TWO}}
    for malformed in (ONE, f"{ONE}:{TWO}:{THREE}", f"{ONE}:xyz", f"{'A' * 64}:{TWO}"):
        args.resume_implementation_compatibility = malformed
        with pytest.raises(AdjointIdentityRefused, match="FROM:TO"):
            stage_a._chain_resume_argument(args)
    args.resume_chain_state_sha256 = None
    assert stage_a._chain_resume_argument(args) is None


# -- dev mode, the default since PQ #1147: run seals stamp and continue ---------

#: Relaunches that differ from the interrupted run only in a run seal. With
#: ``PRISMAQUANT_DEV_MODE`` unset each prints ``[DEV-MODE]`` and resumes.
DEV_SEALS = ("plan", "prepared", "read-manifest", "artifact-budget", "read-window",
             "arithmetic", "implementation-undeclared")
#: Relaunches that differ in the chain's layout or data: walls in dev mode too.
DEV_WALLS = ("batch-size", "probe-fusion", "unit-roster", "stride", "calibration",
             "probes", "seed")


def _dev_resume(root, monkeypatch, capsys, **kw):
    """Resume the run at ``root`` with the switch unset; ``(receipt, stdout)``."""
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    capsys.readouterr()
    receipt = _run(root, monkeypatch, **kw)
    return receipt, capsys.readouterr().out


@pytest.mark.parametrize("name", DEV_SEALS)
def test_dev_mode_resumes_through_a_run_seal(tmp_path, monkeypatch, capsys, name):
    # The baseline runs at the same root under the same generation id, then
    # moves aside: a checkpoint's session names its root.
    root = tmp_path / "run"
    baseline_writes = []
    baseline = _run(root, monkeypatch, writes=baseline_writes)
    root.rename(tmp_path / "baseline")
    writes = []
    _interrupted(root, monkeypatch, writes=writes, interrupt=_at(3, 1, 2))
    stored = json.loads(chain_state_path(adjoint_space(root)).read_text())
    kw = _refusal_kwargs(root, name)
    kw.setdefault("chain_resume", _resume(root))
    resumed, out = _dev_resume(root, monkeypatch, capsys, writes=writes, **kw)
    assert "[DEV-MODE] seal " in out
    assert [c["boundary"] for c in resumed["checkpoints"]] == [5, 4, 2]
    # The resume continues under the stored header, and the rolled planes
    # are the uninterrupted run's.
    assert resumed["run_identity"] == stored["run_identity"]
    assert _merged(writes) == _merged(baseline_writes)
    assert [c["cotangent_sha256"] for c in resumed["checkpoints"]] == [
        c["cotangent_sha256"] for c in baseline["checkpoints"]]
    if name == "implementation-undeclared":
        [switch] = resumed[RESUME_COMPATIBILITY_KEY]
        assert (switch["from_implementation_sha256"], switch["to_implementation_sha256"],
                switch["declared"]) == (ONE, TWO, False)


@pytest.mark.parametrize("name", DEV_WALLS)
def test_dev_mode_still_refuses_a_layout_or_data_difference(tmp_path, monkeypatch, name):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    kw = _refusal_kwargs(root, name)
    kw.setdefault("chain_resume", _resume(root))
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    assert _refused_untouched(root, monkeypatch, REFUSALS[name], **kw) == "failed"


def test_dev_mode_resume_does_not_compute_the_source_identity(
        tmp_path, monkeypatch, capsys):
    """A dev resume may pass NOT_COMPUTED for the source it would only compare."""
    import sys

    from prismaquant.dev_mode import NOT_COMPUTED

    module = sys.modules[__name__]
    root = tmp_path / "run"
    certified = tmp_path / "certified"
    for interrupted in (root, certified):
        _interrupted(interrupted, monkeypatch, interrupt=_at(3, 1, 2))
    stored = json.loads(chain_state_path(adjoint_space(root)).read_text())
    monkeypatch.setattr(module, "_model_identity", lambda _name: NOT_COMPUTED)
    resumed, out = _dev_resume(root, monkeypatch, capsys, chain_resume=_resume(root))
    assert "[DEV-MODE] seal chain state bind_identity at source_model not computed" in out
    assert [c["boundary"] for c in resumed["checkpoints"]] == [5, 4, 2]
    assert resumed["run_identity"] == stored["run_identity"]
    # Certified mode never skips it: NOT_COMPUTED differs from every source.
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(AdjointIdentityRefused, match="differs in bind_identity$"):
        _run(certified, monkeypatch, chain_resume=_resume(certified))


def test_dev_mode_reopens_a_generation_whose_status_is_not_interrupted(
        tmp_path, monkeypatch, capsys):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    generation = _generation(root)
    status = json.loads(generation.read_text())
    status["status"] = "complete"
    generation.write_text(json.dumps(status))
    resumed, out = _dev_resume(root, monkeypatch, capsys, chain_resume=_resume(root))
    assert "[DEV-MODE] seal generation status differs" in out
    assert [c["boundary"] for c in resumed["checkpoints"]] == [5, 4, 2]


def test_dev_mode_resumes_under_another_capsule_binding(tmp_path, monkeypatch, capsys):
    from prismaquant import joint_forward_resume

    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    with monkeypatch.context() as patch:
        patch.setattr(joint_forward_resume, "load_forward_recovery",
                      lambda *a, **kw: SimpleNamespace(
                          receipt_binding={"capsule": "another"}, frontier=0,
                          n_batches=len(draw()), records={}))
        monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
        capsys.readouterr()
        try:
            _run(root, monkeypatch, chain_resume=_resume(root))
        except AdjointIdentityRefused as exc:
            assert "another forward-recovery capsule" not in str(exc)
    out = capsys.readouterr().out
    assert "[DEV-MODE] seal forward-recovery capsule differs" in out
    assert "another forward-recovery capsule" not in out
