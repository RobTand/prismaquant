"""The Stage A chain regime: batching and probe fusion (RobTand/prismaquant#997).

``render_free_layer_roll`` may carry B calibration samples through one
layer forward and backward (``batch_size``), and may run one forward per
sample group with one backward per probe on the retained graph
(``probe_fusion``). The claims these tests hold it to:

* The default regime is the roll that ran before #997, byte for byte:
  every rolled plane, every file the run writes and the receipt.
* At a fixed batch size, fusion is bitwise-neutral.
* A batch size above 1 changes GEMM shapes, so it changes rounding. Its
  rolled planes are as close to a float64 reference as the batch-one
  planes are, and a rerun at the same batch size is bitwise equal.
* A regime is part of the run's identity: stamped in the receipt, carried
  by every band and slice, read back by the quantum's chain, and refused
  when it is malformed, does not fit the sealed window, or meets a pass
  state that is not per-sample.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
import uuid
import warnings

import pytest
import torch
from torch import nn

from prismaquant import joint_cost_stage_a as stage_a
from prismaquant.cost_stage_checkpoint import canonical_json_bytes
from prismaquant.cost_streaming import (
    StreamedBoundaryArtifacts,
    StreamedCausalLM,
    StreamedForwardBoundaries,
    fused_window_size,
)
from prismaquant.joint_adjoint_band import (
    BandRefused,
    build_band_receipt,
    request_chain_regime,
    stage_a_argv,
)
from prismaquant.joint_adjoint_checkpoints import (
    CHAIN_REGIME_KEY,
    ChainRegimeRefused,
    adjoint_slice_sha256,
    chain_regime_identity,
    chain_regime_of,
    normalize_chain_regime,
    render_free_layer_roll,
    require_chain_regime,
    stage_a_run_header,
    stage_a_slice,
)
from prismaquant.model_profiles.default import DefaultProfile
from prismaquant.sensitivity_probe import SharedStateCotangents

from test_joint_cost_quantum_runtime import _boundary_policy, _execution, _stage_a
from test_layer_major_boundary_capture import draw
from test_streamed_cost_checkpoints import _FakeStreamingContext, _model_identity

CAMPAIGN = {"unit_roster_sha256": "a" * 64, "plan_sha256": "b" * 64,
            "prepared_sha256": "c" * 64, "read_manifest_sha256": "d" * 64,
            "implementation_sha256": "1" * 64}
#: The fixture's boundary and cotangent entries: [1, 4, 16] float32.
ENTRY_BYTES = 4 * 16 * 4
N_PROBES = 4


def _digest(tensor) -> str:
    data = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()


# -- the oracle: origin/main 0b16becc6e8's roll, verbatim but for imports ----

def _legacy_render_free_layer_roll(
    runner, *, storage, batches, layer, cotangents, n_probes,
    incoming_entries, incoming_tensor, roll, min_free_gib: float = 0.0,
    then=None,
) -> int:
    from contextlib import nullcontext

    retain = getattr(storage, "retain_produced_boundary", None)
    stage_ahead = getattr(storage, "stage_produced_boundary_ahead", None)
    if stage_ahead is not None and int(layer) > 0:
        stage_ahead(int(layer) - 1)
    with (retain(int(layer)) if retain is not None else nullcontext()):
        backwards = _legacy_render_free_probe_passes(
            runner, storage=storage, batches=batches, layer=layer,
            cotangents=cotangents, n_probes=n_probes,
            incoming_entries=incoming_entries,
            incoming_tensor=incoming_tensor, roll=roll,
            min_free_gib=min_free_gib, then=then)
    return backwards


def _legacy_render_free_probe_passes(
    runner, *, storage, batches, layer, cotangents, n_probes,
    incoming_entries, incoming_tensor, roll, min_free_gib, then=None,
) -> int:
    from prismaquant.aura_cost import _free_gib
    from prismaquant.cost_streaming import prefetched_boundary_batches

    profile = runner.profile
    device, dtype = runner.device, runner.dtype
    backwards = 0
    for probe_index in range(int(n_probes)):
        entries = (None if incoming_entries is None
                   else incoming_entries[probe_index])
        following = then
        if probe_index + 1 < int(n_probes):
            following = (int(layer), None if incoming_entries is None
                         else incoming_entries[probe_index + 1])
        with prefetched_boundary_batches(
                storage, batches, int(layer), incoming=entries,
                then=following) as windows:
            for batch_index, batch, boundary_cpu, incoming_cpu in windows:
                owner = cotangents[probe_index][batch_index]
                try:
                    if _free_gib() < min_free_gib:
                        raise RuntimeError(
                            f"free UMA {_free_gib():.1f} < floor {min_free_gib:.1f}; "
                            f"render-free chain layer {layer} probe {probe_index}")
                    cpu_rng = torch.get_rng_state()
                    cuda_rng = (torch.cuda.get_rng_state(device)
                                if torch.device(device).type == "cuda" else None)
                    if entries is None:
                        incoming_cpu = incoming_tensor(probe_index, batch_index)
                    incoming_grad = incoming_cpu.to(device)
                    x_in = boundary_cpu.to(
                        device=device, dtype=dtype).detach().requires_grad_(True)
                    isolated = profile.isolated_layer_pass_state(
                        batch.shared_pass_state, runner.layers[layer])
                    isolated = owner.graft(isolated)
                    out = runner.isolated_layer(batch, layer, x_in, pass_state=isolated)
                    roots, root_grads = owner.produced_roots()
                    torch.autograd.backward([out, *roots], [incoming_grad, *root_grads])
                    owner.harvest()
                    if not torch.equal(cpu_rng, torch.get_rng_state()) or (
                            cuda_rng is not None
                            and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device))):
                        raise RuntimeError(
                            "render-free chain source consumed Torch RNG")
                    if x_in.grad is None:
                        raise RuntimeError(
                            f"render-free chain layer {layer} produced no input cotangent")
                    roll(x_in.grad.detach().to("cpu"), batch_index, probe_index)
                    backwards += 1
                finally:
                    boundary_cpu = incoming_cpu = None
                    out = x_in = incoming_grad = isolated = roots = root_grads = None
    return backwards


def _legacy_roll_in_the_core(runner, *, batch_size, probe_fusion, roll_may_keep=True,
                             **kwargs):
    """The oracle as the current core calls it: the default regime only.

    ``roll_may_keep`` (PQ #1162) picks pinned or pageable rows on CUDA. This
    fixture runs on the CPU, where the rows are the gradient's either way.
    """
    assert (batch_size, probe_fusion) == (1, False)
    return _legacy_render_free_layer_roll(runner, **kwargs)


# -- the fixture Stage A run -------------------------------------------------

def _capture(root, monkeypatch, *, batch_size=1, fusion=False, window=2, cap=None,
             stride=1, roll=None, identities=None, values=None, campaign=None):
    """One fixture Stage A run into ``root``.

    Returns ``(receipt, planes, forwards)``. ``planes`` maps
    ``(boundary, probe, batch)`` to the sha256 of every cotangent the run
    wrote -- the tail's and every rolled one, including boundary 0, which no
    checkpoint keeps; ``values``, when given, receives the tensors.
    ``forwards`` lists the chain's isolated layer forwards as
    ``(layer, rows)``. The generation id is pinned so two runs into one path
    write equal bytes.
    """
    planes = {}
    forwards = []
    write = StreamedBoundaryArtifacts.write
    bind = StreamedBoundaryArtifacts.bind

    def recorded(self, tensor, **kw):
        if kw.get("probe_index") is not None:
            key = (kw["boundary_index"], kw["probe_index"], kw["batch_index"])
            assert key not in planes, key
            planes[key] = _digest(tensor)
            if values is not None:
                values[key] = tensor.detach().to("cpu").clone()
        return write(self, tensor, **kw)

    def bound(self, identity, **kw):
        if identities is not None:
            identities.append(json.loads(json.dumps(identity)))
        return bind(self, identity, **kw)

    runner, _ = _stage_a(root, monkeypatch)
    runner.context.settle_prefetched_layers = lambda *a, **kw: None
    isolated = runner.isolated_layer

    def counted(batch, layer, hidden, *, pass_state):
        forwards.append((layer, int(hidden.shape[0])))
        return isolated(batch, layer, hidden, pass_state=pass_state)

    runner.isolated_layer = counted
    execution = _execution(root)
    policy = _boundary_policy(root / "boundaries", window=window)
    if cap is not None:
        policy["max_resident_bytes"] = cap
    execution["boundary_storage"] = policy
    with monkeypatch.context() as patch:
        patch.setattr(StreamedBoundaryArtifacts, "write", recorded)
        patch.setattr(StreamedBoundaryArtifacts, "bind", bound)
        patch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=997))
        if roll is not None:
            patch.setattr(stage_a, "render_free_layer_roll", roll)
        receipt = stage_a.run_adjoint_capture_core(
            runner, draw(), execution=execution, output_root=root, stride=stride,
            source_model_identity=_model_identity("joint-source"),
            chain_batch_size=batch_size, chain_probe_fusion=fusion,
            **{**CAMPAIGN, **(campaign or {})})
    return json.loads(json.dumps(receipt)), planes, forwards


def _tree(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def _without(document, *paths):
    """A deep copy of ``document`` with each dotted path removed."""
    document = json.loads(json.dumps(document))
    for path in paths:
        *parents, leaf = path.split(".")
        node = document
        for key in parents:
            node = node[key]
        node.pop(leaf, None)
    return document


def _generation_files(tree):
    return {name for name in tree if name.endswith("generation.json")}


# -- default regime: the pre-#997 roll, byte for byte -------------------------

def test_the_default_regime_is_the_pre_997_roll_byte_for_byte(tmp_path, monkeypatch):
    """Planes, every written file and the receipt equal the oracle's.

    The oracle is origin/main's roll, kept verbatim above and substituted
    into the same Stage A core. Both runs write into one path, one after the
    other (the first moved aside), under one pinned generation id, so equal
    bytes are the claim and nothing needs normalizing but the receipt's
    wall-clock telemetry.
    """
    root = tmp_path / "run"
    legacy, legacy_planes, legacy_forwards = _capture(
        root, monkeypatch, roll=_legacy_roll_in_the_core)
    aside = tmp_path / "legacy"
    root.rename(aside)
    current, planes, forwards = _capture(root, monkeypatch)

    assert planes == legacy_planes
    assert len(planes) == N_PROBES * len(draw()) * 3, "tail plus two rolled boundaries"
    assert forwards == legacy_forwards
    assert _tree(root) == _tree(aside)
    assert CHAIN_REGIME_KEY not in current["run_identity"]
    assert _without(current, "telemetry") == _without(legacy, "telemetry")
    assert current["telemetry"]["chain_backwards"] == \
        legacy["telemetry"]["chain_backwards"] == N_PROBES * len(draw()) * 2


# -- fusion is bitwise-neutral at a fixed batch size --------------------------

@pytest.mark.parametrize("batch_size", [1, 2])
def test_fusion_is_bitwise_neutral_at_a_fixed_batch_size(
        tmp_path, monkeypatch, batch_size):
    """Fused and unfused runs write equal planes, entries and checkpoints.

    Both runs seal one policy, wide enough for a fused window of two
    batches: 2 x (1 boundary + 4 probes) x 256 bytes, plus the one rolled
    entry being written. What may differ is only what the regime is: its
    stamp, and the resident-byte telemetry of windows that read every probe
    at once.
    """
    cap = 2 * (1 + N_PROBES) * ENTRY_BYTES + ENTRY_BYTES
    root = tmp_path / "run"
    unfused, unfused_planes, unfused_forwards = _capture(
        root, monkeypatch, batch_size=batch_size, cap=cap)
    aside = tmp_path / "unfused"
    root.rename(aside)
    fused, fused_planes, fused_forwards = _capture(
        root, monkeypatch, batch_size=batch_size, fusion=True, cap=cap)

    assert fused_planes == unfused_planes
    a, b = _tree(aside), _tree(root)
    generation = _generation_files(a)
    assert generation and generation == _generation_files(b)
    # The chain state (PQ #1001) seals the run identity, regime stamp included.
    state = "layer-quanta/adjoint/chain-state.json"
    assert state in a and state in b
    assert {k: v for k, v in a.items() if k not in generation | {state}} == \
        {k: v for k, v in b.items() if k not in generation | {state}}
    for name in generation:
        assert _without(json.loads((aside / name).read_text()), "telemetry") == \
            _without(json.loads((root / name).read_text()), "telemetry")
    stamp = f"run_identity.{CHAIN_REGIME_KEY}"
    assert _without(json.loads((aside / state).read_text()), stamp, "chain_state_sha256") == \
        _without(json.loads((root / state).read_text()), stamp, "chain_state_sha256")

    variable = ("telemetry", f"run_identity.{CHAIN_REGIME_KEY}", "retention.telemetry")
    assert _without(fused, *variable) == _without(unfused, *variable)
    assert chain_regime_of(fused["run_identity"]) == {
        "batch_size": batch_size, "probe_fusion": True}
    assert chain_regime_of(unfused["run_identity"]) == {
        "batch_size": batch_size, "probe_fusion": False}
    assert (CHAIN_REGIME_KEY in unfused["run_identity"]) is (batch_size != 1)

    # One forward per batch group per layer instead of one per probe. Five
    # samples in windows of two: groups {0}..{4} at B = 1, {0,1} {2,3} {4}
    # at B = 2.
    groups = {1: 5, 2: 3}[batch_size]
    assert len(unfused_forwards) == 2 * N_PROBES * groups
    assert len(fused_forwards) == 2 * groups
    assert sorted(rows for _layer, rows in fused_forwards) == \
        sorted(rows for _layer, rows in unfused_forwards)[::N_PROBES]
    assert fused["telemetry"]["chain_backwards"] == \
        unfused["telemetry"]["chain_backwards"] == N_PROBES * len(draw()) * 2


def test_batch_two_planes_agree_with_batch_one_to_fp32_rounding(tmp_path, monkeypatch):
    """B = 2 on the float32 fixture: equal to B = 1 within fp32 rounding."""
    one_values, two_values = {}, {}
    one, one_planes, _ = _capture(tmp_path / "one", monkeypatch, values=one_values)
    two, two_planes, forwards = _capture(tmp_path / "two", monkeypatch, batch_size=2,
                                         values=two_values)
    assert set(two_planes) == set(one_planes)
    assert chain_regime_of(two["run_identity"])["batch_size"] == 2
    assert sorted({rows for _layer, rows in forwards}) == [1, 2]
    # The tail cotangents come before the chain: bitwise equal.
    tail = {key for key in one_planes if key[0] == 2}
    assert {k: two_planes[k] for k in tail} == {k: one_planes[k] for k in tail}
    eps = torch.finfo(torch.float32).eps
    for key, tensor in one_values.items():
        scale = float(tensor.abs().max()) or 1.0
        assert float((two_values[key] - tensor).abs().max()) <= 64 * eps * scale, key


# -- refusals before the capture ----------------------------------------------

def test_a_batch_size_that_does_not_divide_the_window_refuses_before_capture(
        tmp_path, monkeypatch):
    with pytest.raises(ChainRegimeRefused, match="does not divide the sealed read window"):
        _capture(tmp_path / "run", monkeypatch, batch_size=3)
    assert not list((tmp_path / "run").rglob("*.pt")), "nothing was captured"


def test_a_fused_window_that_does_not_fit_refuses_before_capture(tmp_path, monkeypatch):
    # The default fixture window holds 5 entries: one batch fused, not two.
    with pytest.raises(ChainRegimeRefused, match="fused window at batch size 2"):
        _capture(tmp_path / "run", monkeypatch, batch_size=2, fusion=True)
    assert not list((tmp_path / "run").rglob("*.pt")), "nothing was captured"


def test_fused_window_size_is_the_largest_fitting_divisor():
    size = lambda **kw: fused_window_size(
        prefetch_batches=64, per_batch_bytes=5, **kw)
    assert size(max_resident_bytes=5 * 64, batch_size=1) == 64
    assert size(max_resident_bytes=5 * 20, batch_size=1) == 16
    assert size(max_resident_bytes=5 * 20, batch_size=8) == 16
    assert size(max_resident_bytes=5 * 15, batch_size=8) == 8
    # The rolled entry being written shares the ceiling.
    assert size(max_resident_bytes=5 * 16, batch_size=8) == 16
    assert size(max_resident_bytes=5 * 16, batch_size=8, write_bytes=1) == 8
    with pytest.raises(ChainRegimeRefused):
        size(max_resident_bytes=5 * 7, batch_size=8)
    with pytest.raises(ChainRegimeRefused):
        size(max_resident_bytes=5 * 64, batch_size=3)


# -- the regime identity -------------------------------------------------------

def test_the_regime_stamp_grammar():
    assert chain_regime_identity({"batch_size": 1, "probe_fusion": False}) is None
    stamp = chain_regime_identity({"batch_size": 8, "probe_fusion": True})
    assert stamp == {"schema": "prismaquant.stage_a.chain_regime.v1",
                     "batch_size": 8, "probe_fusion": True}
    assert chain_regime_of({}) == {"batch_size": 1, "probe_fusion": False}
    assert chain_regime_of({CHAIN_REGIME_KEY: stamp}) == {
        "batch_size": 8, "probe_fusion": True}
    for bad in (None, {**stamp, "schema": "v0"}, {**stamp, "extra": 1},
                {**stamp, "batch_size": 0}, {**stamp, "batch_size": True},
                {**stamp, "probe_fusion": "on"},
                chain_regime_identity({"batch_size": 2, "probe_fusion": False})
                | {"batch_size": 1}):
        with pytest.raises(ChainRegimeRefused):
            chain_regime_of({CHAIN_REGIME_KEY: bad})
    for bad in ((0, False), (1.0, False), (True, False), (2, 1)):
        with pytest.raises(ChainRegimeRefused):
            normalize_chain_regime(*bad)
    with pytest.raises(ChainRegimeRefused, match="batch size 8, probe fusion on"):
        require_chain_regime({CHAIN_REGIME_KEY: stamp},
                             {"batch_size": 8, "probe_fusion": False}, where="resume")
    with pytest.raises(ChainRegimeRefused, match="batch size 1, probe fusion off"):
        require_chain_regime({}, {"batch_size": 2, "probe_fusion": False}, where="resume")
    assert require_chain_regime({CHAIN_REGIME_KEY: stamp},
                                {"batch_size": 8, "probe_fusion": True},
                                where="resume")["batch_size"] == 8


def test_the_band_tool_reads_the_regime_from_the_sealed_request():
    command = ["python3", "-m", "prismaquant.joint_adjoint_capture",
               "--plan", "/p.json", "--plan-sha256", "b" * 64,
               "--prepared", "/q.json", "--prepared-sha256", "c" * 64,
               "--output-root", "/o"]
    assert request_chain_regime(stage_a_argv(command)) == {
        "batch_size": 1, "probe_fusion": False}
    assert request_chain_regime(stage_a_argv(
        command + ["--chain-batch-size", "8", "--chain-probe-fusion", "on"])) == {
        "batch_size": 8, "probe_fusion": True}
    for flags in (["--chain-batch-size", "x"], ["--chain-batch-size", "0"],
                  ["--chain-probe-fusion", "yes"]):
        with pytest.raises(BandRefused):
            request_chain_regime(stage_a_argv(command + flags))
    with pytest.raises(BandRefused, match="repeats"):
        stage_a_argv(command + ["--chain-batch-size", "2", "--chain-batch-size", "2"])


def test_a_band_answers_for_the_regime_its_run_sealed(tmp_path, monkeypatch):
    """A B = 2 run's bands equal its receipt's slices only under its regime."""
    identities = []
    root = tmp_path / "run"
    receipt, _planes, _forwards = _capture(
        root, monkeypatch, batch_size=2, identities=identities)
    digests = {"plan_sha256": "b" * 64, "prepared_sha256": "c" * 64,
               "read_manifest_sha256": "d" * 64, "stride_value": 1,
               "stride_source": None, "unit_roster_sha256": "a" * 64,
               "bind_identity": identities[-1]}
    regime = {"batch_size": 2, "probe_fusion": False}
    for boundary in receipt["stride"]["boundaries"]:
        band = json.loads(json.dumps(build_band_receipt(
            output_root=root, boundary=boundary, chain_regime=regime, **digests)))
        assert stage_a_run_header(band) == stage_a_run_header(receipt)
        for layer in band["band"]["layers"]:
            from_band, from_receipt = stage_a_slice(band, layer), stage_a_slice(receipt, layer)
            assert canonical_json_bytes(from_band, where="band") == \
                canonical_json_bytes(from_receipt, where="receipt")
            assert chain_regime_of(from_band["run_identity"]) == regime
        unstamped = json.loads(json.dumps(build_band_receipt(
            output_root=root, boundary=boundary, **digests)))
        assert stage_a_run_header(unstamped) != stage_a_run_header(receipt), (
            "a band built without the run's regime answers for another run")
    with pytest.raises(BandRefused):
        build_band_receipt(output_root=root, boundary=1,
                           chain_regime={"batch_size": 0, "probe_fusion": False},
                           **digests)


# -- the quantum's chain reads the slice's regime -----------------------------

def test_the_quantum_chain_runs_the_slice_regime_bitwise(tmp_path, monkeypatch):
    """A layer-0 quantum rebuilds Stage A's B = 2 chain through layer 1, bitwise.

    Stage A at stride 2 checkpoints only the tail, so the layer-0 quantum
    chains layer 1 itself. Its rolled cotangents must be Stage A's own
    boundary-1 plane, which needs Stage A's batch size.
    """
    from prismaquant import joint_adjoint_checkpoints as checkpoints_mod
    from prismaquant.joint_cost_quantum import QuantumIdentityRefused
    from test_joint_cost_quantum_runtime import _run_quantum, _single_run

    import prismaquant.aura_cost as aura

    single_root = tmp_path / "single"
    single = _single_run(single_root, monkeypatch, checkpoint=single_root / "checkpoints")
    output_root = tmp_path / "campaign"
    # The campaign the runtime fixture's quantum record binds.
    campaign = {"plan_sha256": "d" * 64, "prepared_sha256": "e" * 64,
                "read_manifest_sha256": "f" * 64,
                "implementation_sha256": aura._aura_source_sha256()}
    receipt, planes, _ = _capture(output_root, monkeypatch, batch_size=2, stride=2,
                                  campaign=campaign)
    assert [c["boundary"] for c in receipt["checkpoints"]] == [2]

    chained = {}
    calls = []
    real = checkpoints_mod.render_free_layer_roll

    def watched(runner, **kw):
        calls.append((kw["layer"], kw.get("batch_size"), kw.get("probe_fusion")))
        roll = kw["roll"]

        def recorded(tensor, batch, probe):
            chained[(kw["layer"], probe, batch)] = _digest(tensor)
            return roll(tensor, batch, probe)

        return real(runner, **{**kw, "roll": recorded})

    monkeypatch.setattr(checkpoints_mod, "render_free_layer_roll", watched)
    _run_quantum(tmp_path, monkeypatch, single=single, layer=0, receipt=receipt,
                 output_root=output_root, plan_sha="d" * 64, prepared_sha="e" * 64)
    assert calls == [(1, 2, False)]
    assert chained == {key: digest for key, digest in planes.items() if key[0] == 1}

    adjoint_slice = stage_a_slice(receipt, 0)
    adjoint_slice["run_identity"][CHAIN_REGIME_KEY]["batch_size"] = 0
    with pytest.raises(QuantumIdentityRefused, match="chain regime"):
        _run_quantum(tmp_path, monkeypatch, single=single, layer=0, receipt=receipt,
                     output_root=tmp_path / "refused", plan_sha="d" * 64,
                     prepared_sha="e" * 64, adjoint_slice=adjoint_slice)


# -- a pass state that is not per-sample --------------------------------------

class _SharingProfile(DefaultProfile):
    """A profile whose layers borrow shared forward state, as Gemma4's do."""

    def isolated_layer_pass_state(self, captured, layer):
        return {"shared_kv_states": {0: torch.zeros(1)}}


def _toy_runner(*, profile=None, dtype=torch.float32, device="cpu", width=16,
                routed=False, seed=5):
    torch.manual_seed(seed)
    model = _RoutedTinyLM(width=width) if routed else _PlainTinyLM(width=width)
    model = model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    context = _FakeStreamingContext(model)
    context.device = torch.device(device)
    return model, StreamedCausalLM(context, profile or DefaultProfile())


def _toy_batches(runner, ids):
    batches = []
    with torch.no_grad():
        for row in ids:
            prepared, position_ids, hidden, embeddings, mask = runner._prepare(row[None])
            batches.append(StreamedForwardBoundaries(
                prepared, position_ids, embeddings, mask,
                [hidden.detach().to("cpu")], None))
    return batches


def _toy_roll(runner, batches, incoming, *, batch_size, fusion, n_probes,
              cotangents=None):
    rolled = {}
    render_free_layer_roll(
        runner, storage=None, batches=batches, layer=0,
        cotangents=cotangents or [[SharedStateCotangents() for _ in batches]
                                  for _ in range(n_probes)],
        n_probes=n_probes, incoming_entries=None,
        incoming_tensor=lambda probe, batch: incoming[probe][batch],
        roll=lambda tensor, batch, probe: rolled.__setitem__((probe, batch), tensor),
        batch_size=batch_size, probe_fusion=fusion)
    return rolled


class _PlainLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states, **_kwargs):
        return torch.tanh(self.proj(hidden_states))


class _PlainTinyLM(nn.Module):
    def __init__(self, *, width=16, vocab=23, layer=_PlainLayer):
        super().__init__()
        self.model = nn.Module()
        self.model.config = SimpleNamespace(layer_types=())
        self.model.embed_tokens = nn.Embedding(vocab, width)
        self.model.layers = nn.ModuleList([layer(width), layer(width)])
        self.model.norm = nn.Identity()
        self.lm_head = nn.Linear(width, vocab, bias=False)


@pytest.mark.parametrize("batch_size, fusion", [(2, False), (1, True), (2, True)])
def test_a_shared_pass_state_refuses_a_grouped_roll(batch_size, fusion):
    _model, runner = _toy_runner(profile=_SharingProfile())
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    batches = _toy_batches(runner, ids)
    incoming = [[torch.ones(1, 4, 16) for _ in batches]]
    with pytest.raises(ChainRegimeRefused, match="per-sample pass state"):
        _toy_roll(runner, batches, incoming, batch_size=batch_size, fusion=fusion,
                  n_probes=1)


@pytest.mark.parametrize("batch_size, fusion", [(2, False), (1, True)])
def test_a_shared_state_cotangent_refuses_a_grouped_roll(batch_size, fusion):
    _model, runner = _toy_runner()
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    batches = _toy_batches(runner, ids)
    incoming = [[torch.ones(1, 4, 16) for _ in batches]]
    owners = [[SharedStateCotangents() for _ in batches]]
    owners[0][1]._acc[("shared_kv_states", 0, None)] = torch.ones(1)
    with pytest.raises(ChainRegimeRefused, match="shared-state cotangent"):
        _toy_roll(runner, batches, incoming, batch_size=batch_size, fusion=fusion,
                  n_probes=1, cotangents=owners)


def test_the_default_roll_still_serves_a_shared_pass_state():
    """B = 1 unfused keeps the per-(probe, batch) graft the roll always had."""
    _model, runner = _toy_runner(profile=_SharingProfile())
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    batches = _toy_batches(runner, ids)
    incoming = [[torch.ones(1, 4, 16) for _ in batches]]
    rolled = _toy_roll(runner, batches, incoming, batch_size=1, fusion=False, n_probes=1)
    assert sorted(rolled) == [(0, 0), (0, 1)]


# -- B = 8 against B = 1: precision, route flips, reruns ----------------------

class _RoutedLayer(nn.Module):
    """A top-k routed MLP: every expert computed, gated by the router's top k.

    ``routes`` records each forward's top-k expert indices, so a test can
    count the tokens whose route a batch size or a dtype flipped.
    """

    def __init__(self, width, experts=8, top_k=2, hidden=None):
        super().__init__()
        hidden = hidden or 2 * width
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(width, hidden, bias=False), nn.SiLU(),
                          nn.Linear(hidden, width, bias=False))
            for _ in range(experts)])
        self.top_k = top_k
        self.routes = []

    def forward(self, hidden_states, **_kwargs):
        logits = self.router(hidden_states)
        weights, index = logits.float().softmax(-1).topk(self.top_k, dim=-1)
        self.routes.append(index.detach().sort(dim=-1).values.to("cpu"))
        gates = torch.zeros_like(logits, dtype=torch.float32).scatter(
            -1, index, weights).to(hidden_states.dtype)
        out = sum(gates[..., e:e + 1] * expert(hidden_states)
                  for e, expert in enumerate(self.experts))
        return hidden_states + out


class _RoutedTinyLM(_PlainTinyLM):
    def __init__(self, *, width=16, vocab=97):
        super().__init__(width=width, vocab=vocab, layer=_RoutedLayer)


def _routes_by_sample(layer, n_samples, *, first_forwards):
    """Each sample's routes from the first probe's forwards, sample order."""
    rows = torch.cat(layer.routes[:first_forwards], dim=0)
    assert rows.shape[0] == n_samples
    return rows


def test_batch_eight_is_as_close_to_float64_as_batch_one():
    """B = 8 vs B = 1 on a bf16 routed layer, against a float64 reference.

    Per-sample relative L2 of every rolled cotangent, over the tokens whose
    route all three runs agree on (a flipped route is a different function,
    counted separately). B = 8's mean and max may exceed B = 1's by at most
    one bfloat16 unit roundoff, relative: e8 <= e1 * (1 + 2**-8). A rerun at
    each batch size is bitwise
    equal, and so is the fused roll at B = 8. Runs on the GPU when there is
    one, where batching changes the GEMM shapes the kernels see.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    width, samples, tokens, probes = 256, 16, 32, 2
    generator = torch.Generator().manual_seed(997)
    ids = torch.randint(0, 97, (samples, tokens), generator=generator)
    incoming64 = [[torch.randn(1, tokens, width, generator=generator, dtype=torch.float64)
                   for _ in range(samples)] for _ in range(probes)]

    def run(dtype, batch_size, fusion=False):
        model, runner = _toy_runner(dtype=dtype, device=device, width=width, routed=True)
        batches = _toy_batches(runner, ids)
        incoming = [[tensor.to(dtype) for tensor in plane] for plane in incoming64]
        layer = model.model.layers[0]
        layer.routes.clear()
        rolled = _toy_roll(runner, batches, incoming, batch_size=batch_size,
                           fusion=fusion, n_probes=probes)
        forwards = -(-samples // batch_size)
        return rolled, _routes_by_sample(layer, samples, first_forwards=forwards)

    reference, routes64 = run(torch.float64, 1)
    one, routes1 = run(torch.bfloat16, 1)
    eight, routes8 = run(torch.bfloat16, 8)
    one_again, _ = run(torch.bfloat16, 1)
    eight_again, _ = run(torch.bfloat16, 8)
    eight_fused, routes8f = run(torch.bfloat16, 8, fusion=True)

    digests = lambda rolled: {key: _digest(tensor) for key, tensor in rolled.items()}
    assert digests(one_again) == digests(one), "B = 1 reruns are bitwise equal"
    assert digests(eight_again) == digests(eight), "B = 8 reruns are bitwise equal"
    assert digests(eight_fused) == digests(eight), "fusion is bitwise-neutral at B = 8"
    assert torch.equal(routes8f, routes8)

    flipped = lambda a, b: (a != b).any(dim=-1)
    flips = {"b8_vs_b1": int(flipped(routes8, routes1).sum()),
             "b1_vs_f64": int(flipped(routes1, routes64).sum()),
             "b8_vs_f64": int(flipped(routes8, routes64).sum())}
    agree = ~(flipped(routes8, routes1) | flipped(routes1, routes64)
              | flipped(routes8, routes64))

    def relative(rolled):
        errors = []
        for (probe, sample), tensor in sorted(rolled.items()):
            want = reference[(probe, sample)][0][agree[sample]]
            got = tensor.to(torch.float64)[0][agree[sample]]
            errors.append(float((got - want).norm() / want.norm()))
        return torch.tensor(errors, dtype=torch.float64)

    e1, e8 = relative(one), relative(eight)
    report = {"device": device, "tokens": samples * tokens, "route_flips": flips,
              "agreeing_tokens": int(agree.sum()),
              "b1": {"mean": float(e1.mean()), "max": float(e1.max())},
              "b8": {"mean": float(e8.mean()), "max": float(e8.max())},
              "b8_bitwise_equal_b1": digests(eight) == digests(one)}
    # A warning, not a print: the pool's pytest runs without -rP, and the
    # warnings summary is what reaches the shard log.
    warnings.warn("chain-regime precision: " + json.dumps(report, sort_keys=True),
                  UserWarning)
    assert int(agree.sum()) > samples * tokens // 2, report
    slack = 1.0 + 2.0 ** -8
    assert float(e8.mean()) <= float(e1.mean()) * slack, report
    assert float(e8.max()) <= float(e1.max()) * slack, report
