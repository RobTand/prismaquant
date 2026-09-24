"""Stage B one-pass spill replay (PQ #994) against the windowed replay.

The windowed replay runs the target layer once per (retained window, probe);
the spill replay runs it once per probe and feeds every window from a local
spill. These tests drive the real layer quantum (stage A capture, sealed
windows, ``run_layer_quantum_core``) over a bf16 model whose routed experts
are packed and dispatched through ``F.grouped_mm``, and require the two
paths' unit envelopes and cost rows to be byte-identical.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import prismaquant.aura_cost as aura
import prismaquant.joint_replay_spill as spill_mod
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.cost_streaming import StreamedCausalLM
from prismaquant.joint_adjoint_checkpoints import chain_layers_for
from prismaquant.joint_adjoint_slices import adjoint_slice_sha256, stage_a_slice
from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.routed_experts import profile_declared_packed_expert_projections

import test_joint_cost_quantum_runtime as rt
from test_joint_operator_windows import policy as operator_policy
from test_streamed_cost_checkpoints import _FakeStreamingContext, _model_identity

VOCAB, WIDTH, INTER, EXPERTS, TOP_K = 23, 16, 32, 4, 2
N_PROBES = 3
RENDER_FORMATS = ("FP8_E4M3", "NVFP4A16")
FORMATS = (*RENDER_FORMATS, "BF16")
DTYPE = torch.bfloat16


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _RoutedExperts(nn.Module):
    """Transformers-style packed experts: sorted rows, cumulative offsets."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(EXPERTS, 2 * INTER, WIDTH) / 4)
        self.down_proj = nn.Parameter(torch.randn(EXPERTS, WIDTH, INTER) / 4)

    def forward(self, x, experts, weights):
        tokens, width = x.shape
        flat = experts.reshape(-1)
        order = torch.argsort(flat, stable=True)
        inputs = x.unsqueeze(1).expand(tokens, TOP_K, width).reshape(-1, width)[order]
        offsets = torch.cumsum(torch.bincount(flat, minlength=EXPERTS), 0).to(torch.int32)
        gate, up = F.grouped_mm(inputs, self.gate_up_proj.transpose(-2, -1),
                                offs=offsets).chunk(2, dim=-1)
        down = F.grouped_mm(F.silu(gate) * up, self.down_proj.transpose(-2, -1),
                            offs=offsets)
        # A permutation scatter, never an atomic accumulate: every row lands
        # in exactly one slot, so the forward and its gradient are exact.
        routed = torch.zeros_like(down)
        routed[order] = down
        return (routed.reshape(tokens, TOP_K, width) * weights.unsqueeze(-1)).sum(dim=1)


class _SharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(WIDTH, INTER, bias=False)
        self.up_proj = nn.Linear(WIDTH, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, WIDTH, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _MoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward = nn.Module()
        self.feed_forward.gate = nn.Linear(WIDTH, EXPERTS, bias=False)
        self.feed_forward.experts = _RoutedExperts()
        self.feed_forward.shared_expert = _SharedExpert()

    def forward(self, hidden_states, **_kwargs):
        block = self.feed_forward
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        scores = block.gate(x).float().softmax(dim=-1)
        weights, experts = torch.topk(scores, TOP_K, dim=-1)
        mixed = block.experts(x, experts, weights.to(x.dtype)) + block.shared_expert(x)
        return torch.tanh(hidden_states + mixed.reshape(shape))


class _MoELM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_experts_per_tok=TOP_K)
        self.model = nn.Module()
        self.model.config = SimpleNamespace(layer_types=())
        self.model.embed_tokens = nn.Embedding(VOCAB, WIDTH)
        self.model.layers = nn.ModuleList([_MoELayer(), _MoELayer()])
        self.model.norm = nn.Identity()
        self.lm_head = nn.Linear(WIDTH, VOCAB, bias=False)

    def forward(self, input_ids):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden, input_ids=input_ids)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(hidden)))


def _calibration():
    generator = torch.Generator().manual_seed(994)
    return torch.randint(0, VOCAB, (4, 8), generator=generator)


def _runner(state, device):
    model = _MoELM()
    model.load_state_dict(state)
    model = model.to(device=device, dtype=DTYPE).eval()
    context = _FakeStreamingContext(model)
    context.device = torch.device(device)
    install = context.install
    context.install = (lambda layer, *, require_prefetched=False, prefetch_following=True:
                       install(layer, require_prefetched=require_prefetched))
    context.settle_prefetched_layers = lambda layers, *, retry_availability=False: None
    context.settle_prefetch_layers = lambda layers: None
    context.source_residency_snapshot = lambda layers, include_head=False: {
        "owners": [], "unique_storage_bytes": sum(
            p.numel() * p.element_size() for p in model.parameters())}
    runner = StreamedCausalLM(context, Lfm2MoeProfile())
    runner.require_prefetched_residency = True
    runner.prefetch_lookahead = 1
    return model, context, runner


def _targets(model, profile):
    linears = aura._target_linears(model, include_lm_head=False,
                                   include_routed_experts=True, profile=profile)
    linears.update({member.qname: member for member in
                    profile_declared_packed_expert_projections(model, profile)})
    return linears


def _policy_budget():
    from prismaquant.joint_retained_window_plan import EXECUTION_SCHEMA

    single = INTER * WIDTH * 4 * 2  # FP32 G.T@X plus one activation group
    policy = operator_policy(max_statistics_bytes=1 << 20, max_candidate_bytes=1 << 20)
    # Three targets per window: five windows over a layer's fifteen targets.
    budget = dataclasses.replace(rt._budget(), statistics_cap_bytes=3 * single,
                                 candidate_delta_bytes=INTER * WIDTH * 4,
                                 max_windows_per_layer=16)
    retained = {"schema": EXECUTION_SCHEMA, "budget": budget.as_dict(),
                "source_reserve_bytes": 1 << 20,
                "source_loading_reserve_bytes": 2 << 20}
    return policy, budget, retained


def _execution(root):
    policy, _budget, retained = _policy_budget()
    return {"n_probes": N_PROBES, "seed_base": 7000, "probe_microbatch": 1,
            "token_scope": "all", "temperature": 1.0, "production_act_scales": "0",
            "boundary_storage": rt._boundary_policy(root / "boundaries"),
            "operator_windows": dict(policy), "retained_operator_windows": retained,
            "min_free_gib": 0, "device_envelope_bytes": None}


def _prepared(model, context, runner, weights, root):
    linears = _targets(model, runner.profile)
    cache = ProductionWeightCache(
        weights={key: tensor.clone() for key, tensor in weights.items()}, levers={},
        activation_max_abs={name: 1.0 for name in linears})
    cache, _proofs = rt._prepared_cache(model, context, runner, cache, root)
    return cache, linears


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """One stage A capture and the sealed records both replays consume."""
    from prismaquant.joint_cost_stage_a import run_adjoint_capture_core
    from prismaquant.joint_statistics_replay import preflight_joint_operator_admission

    root = tmp_path_factory.mktemp("spill-campaign")
    device = _device()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
        patch.setenv("PRISMAQUANT_DEV_MODE", "1")
        torch.manual_seed(715)
        state = {name: tensor.detach().clone()
                 for name, tensor in _MoELM().state_dict().items()}
        model, context, runner = _runner(state, device)
        linears = _targets(model, runner.profile)
        weights = {(name, fmt): module.weight.detach().to("cpu").clone() + 0.03125
                   for name, module in linears.items() for fmt in RENDER_FORMATS}
        cache, linears = _prepared(model, context, runner, weights, root / "shared")
        formats_by_qname = {name: list(FORMATS) for name in linears}
        policy, budget, _retained = _policy_budget()
        names_by_layer = {layer: sorted(n for n in linears
                                        if runner.layer_index_for_qname(n) == layer)
                          for layer in (0, 1)}
        preflight = preflight_joint_operator_admission(
            names_by_layer, linears, {name: list(RENDER_FORMATS) for name in linears},
            cache, policy=policy, retained_budget=budget, source_bytes=1 << 20)
        output_root = root / "campaign"
        _model_a, _context_a, runner_a = _runner(state, device)
        receipt = run_adjoint_capture_core(
            runner_a, _calibration(), execution=_execution(root / "exec"),
            output_root=output_root, stride=2,
            source_model_identity=_model_identity("joint-source"),
            unit_roster_sha256=rt._hex("a"), plan_sha256=rt._hex("d"),
            prepared_sha256=rt._hex("e"), read_manifest_sha256=rt._hex("f"),
            implementation_sha256=aura._aura_source_sha256())
    assert [c["boundary"] for c in receipt["checkpoints"]] == [2]
    records, slices = {}, {}
    for layer in (0, 1):
        # Each quantum reads its own stage-A slice (PQ #993), as the
        # runtime fixture's ``_run_quantum`` builds it.
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
                           records=records, slices=slices, output_root=output_root, device=device,
                           formats_by_qname=formats_by_qname, preflight=preflight)


def _quantum(campaign, monkeypatch, *, layer, spill_root=None, ceiling=None,
             resume=False, label=None, regime=None, emit_handoff=False,
             guard=None):
    from prismaquant.joint_cost_quantum import (
        ChunkFrontier, QuantumCounters, QuantumProgress, quantum_layer_roster,
        quantum_retained_state, resolve_quantum_windows, run_layer_quantum_core)

    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    for name in spill_mod.SPILL_ENV:
        monkeypatch.delenv(name, raising=False)
    if spill_root is not None:
        monkeypatch.setenv(spill_mod.SPILL_ENV[0], str(spill_root))
    if ceiling is not None:
        monkeypatch.setenv(spill_mod.SPILL_ENV[1], str(ceiling))
    if guard is not None:
        # A caller-supplied guard stands in for the capture guard on any
        # device, so every phase admission the quantum makes reaches it.
        import prismaquant.joint_statistics_replay as replay
        monkeypatch.setattr(replay, "operator_window_guard", lambda *a, **k: guard)
    elif campaign.device.type == "cuda":
        # The fixture's budget is far below a real capture guard's physical
        # floor; the guard charges admissions and never touches arithmetic.
        # test_layer_quantum_charges_each_phase_to_its_guard drives the
        # guarded path with a recording guard instead.
        import prismaquant.joint_statistics_replay as replay
        monkeypatch.setattr(replay, "operator_window_guard", lambda *a, **k: None)
    model, context, runner = _runner(campaign.state, campaign.device)
    cache, _linears = _prepared(model, context, runner, campaign.weights,
                                campaign.root / "shared")
    record = campaign.records[layer]
    execution = _execution(campaign.root / "exec")
    if regime is not None:
        execution["replay_regime"] = regime
    retained = quantum_retained_state(execution)
    roster = quantum_layer_roster(runner, campaign.formats_by_qname, layer)
    resolved = resolve_quantum_windows(
        record, layer=layer, names=roster.names, linears=roster.linears,
        render_formats=roster.render_formats, production_cache=cache,
        operator_windows=retained.operator_windows,
        retained_budget=retained.retained_budget, source_bytes=retained.source_bytes)
    frontier = ChunkFrontier(chunks=record["chunks"], windows=resolved)
    counters = QuantumCounters(quantum_id=record["quantum_id"],
                               identity_sha256=record["identity_sha256"],
                               chunks=record["chunks"], frontier=frontier)
    progress = QuantumProgress(frontier=frontier, base_units=0)
    state = SimpleNamespace(context=context, counters=counters, resolved=resolved,
                            handoff=None)
    # A band-serial producer (PQ #996), built as ``main`` builds it. The
    # keyword is passed only when asked, so a wrapper that adds its own
    # emitter (tests/test_band_serial_spill.py) sees the call unchanged.
    band = {}
    try:
        if emit_handoff:
            # The emitter refuses a capture batch off the slice's chain batch
            # size at construction, before the core: that refusal lands in
            # ``state.error`` as ``run_layer_quantum``'s own would.
            from prismaquant.joint_quantum_handoff import HandoffEmitter
            from prismaquant.joint_replay_regime import normalize_replay_regime
            band["handoff_emitter"] = HandoffEmitter(
                record=record, adjoint_slice=campaign.slices[layer],
                boundary_storage=execution["boundary_storage"],
                capture_batch=normalize_replay_regime(regime)["capture_batch"])
        payload = run_layer_quantum_core(
            runner, cache, _calibration(), campaign.formats_by_qname,
            record=record, adjoint_slice=campaign.slices[layer], execution=execution,
            output_root=campaign.output_root, projection_backend=None, resume=resume,
            resolved_windows=resolved, counters=counters, progress=progress, **band)
    except BaseException as exc:
        state.error = exc
        return None, state
    state.counters_block = counters.finish(
        units_done=len(payload["costs"]),
        units_total=sum(len(w["names"]) for w in resolved))
    if emit_handoff:
        state.handoff = band["handoff_emitter"].published
    return payload, state


class _RecordingGuard:
    """The capture guard's admission contract, admitting everything.

    ``check`` takes the same keyword-only arguments as
    ``CaptureMemoryGuard.check`` and records each admission. The physical
    cap and margin admit any fixture plan, so the quantum runs its guarded
    path end to end on either device.
    """

    def __init__(self, device):
        self.device = device
        self.physical_cap_bytes = 1 << 62
        self.margin_bytes = 0
        self.admissions = []

    def check(self, label, *, reserve_bytes=0, reserve_device_bytes=0):
        self.admissions.append((label, reserve_bytes))
        return {"conservative_cgroup_plus_cuda_reserved_bytes": 0,
                "committed_cgroup_plus_cuda_reserved_bytes": 0}


def test_layer_quantum_charges_each_phase_to_its_guard(campaign, monkeypatch):
    """Every guarded admission of a layer quantum reaches its guard.

    Every CUDA quantum builds a capture guard, and the other tests here
    replace it with None, so this test is the only one that runs the
    guarded path.
    """
    layer = 1
    guard = _RecordingGuard(campaign.device)
    _clear_output(campaign, layer)
    payload, state = _quantum(campaign, monkeypatch, layer=layer, guard=guard)
    assert payload is not None, _chain(state.error)
    labels = [label for label, _reserve in guard.admissions]
    assert labels[0] == "before_layer_quantum_replay"
    assert guard.admissions[0][1] == 0
    for label in (f"before_quantum_source_loading:{layer}",
                  f"admit_quantum_source_loading:{layer}",
                  f"before_quantum_reverse:{layer}",
                  f"admit_quantum_reverse:{layer}",
                  "before_joint_retained_candidate_load",
                  "before_joint_retained_statistics_probe"):
        assert label in labels


class _DeviceRecordingGuard(_RecordingGuard):
    """A recording guard with a declared device envelope, as Stage B's has.

    ``check_operator_allocation`` routes a device reservation to
    ``reserve_device_bytes`` only for a guard with a device envelope; this one
    records both sides of every admission.
    """

    device_bytes = 1 << 61

    def check(self, label, *, reserve_bytes=0, reserve_device_bytes=0):
        self.admissions.append((label, reserve_bytes, reserve_device_bytes))
        return {"conservative_cgroup_plus_cuda_reserved_bytes": 0,
                "committed_cgroup_plus_cuda_reserved_bytes": 0}


def _record_capture_sides(monkeypatch):
    """Record every read of the spill's host and device capture reserves.

    Returns ``{"capture_reserve_host_bytes": [...],
    "capture_reserve_device_bytes": [...]}``, filled in read order.
    """
    sides = {"capture_reserve_host_bytes": [], "capture_reserve_device_bytes": []}

    def recording(name, original):
        def read(self):
            sides[name].append(original.fget(self))
            return sides[name][-1]
        return property(read)

    for name in sides:
        monkeypatch.setattr(spill_mod.StageBReplaySpill, name,
                            recording(name, getattr(spill_mod.StageBReplaySpill, name)))
    return sides


def test_capture_pass_charges_its_cuda_allocations_to_the_device_side(campaign, monkeypatch,
                                                                     tmp_path):
    """The capture's workspace and held inputs are device charges (PQ #1157).

    Every stored batch's backward workspace and every target input the spill
    holds until its backward are CUDA allocations, so the capture admission
    charges them to the device side, where the guard also holds them against
    the device envelope. Only the spill's pinned host arenas stay on the host
    side. Before #1157 the whole sum rode ``reserve_bytes``.
    """
    regime, batch = "capture_batch=2", 2
    layer = 0
    _policy, budget, _retained = _policy_budget()
    sides = _record_capture_sides(monkeypatch)
    guard = _DeviceRecordingGuard(campaign.device)
    _clear_output(campaign, layer)
    payload, state = _quantum(campaign, monkeypatch, layer=layer, guard=guard,
                              spill_root=_spill_root(tmp_path), ceiling=1 << 30,
                              regime=regime)
    assert payload is not None, _chain(state.error)
    captures = [(host, device) for label, host, device in guard.admissions
                if label == "before_joint_window_backward"]
    hosts = sides["capture_reserve_host_bytes"]
    devices = sides["capture_reserve_device_bytes"]
    assert len(captures) == N_PROBES == len(hosts) == len(devices)
    assert captures == [(host, budget.capture_workspace_bytes(batch) + device)
                        for host, device in zip(hosts, devices)]
    # ``spill.capture`` allocates the pinned arenas as it opens, before this
    # admission, so their host charge is already in the guard's reading and
    # the host side reserves nothing more here.
    assert hosts == [0] * N_PROBES
    assert all(device > 0 for device in devices)


@pytest.mark.parametrize("spilled", [False, True], ids=["windowed", "spill"])
def test_layer_quantum_opens_one_io_span_per_phase(campaign, monkeypatch, tmp_path,
                                                   capsys, spilled):
    """Every phase of a quantum has its ``/proc/self/io`` span, in counters.json.

    The v6 IO baseline reads these: checkpoint-load, each chain layer, the
    own-source install, each window, each (window, probe) replay and each
    probe's spill capture. The checkpoint loader also prints rate lines.
    """
    from prismaquant.io_spans import PROC_IO_FIELDS, READ_RATE_MARKER

    layer = 1
    _clear_output(campaign, layer)
    kwargs = ({"spill_root": _spill_root(tmp_path), "ceiling": 1 << 30}
              if spilled else {})
    payload, state = _quantum(campaign, monkeypatch, layer=layer, **kwargs)
    assert payload is not None, _chain(state.error)
    spans = state.counters_block["io_spans"]
    windows = len(campaign.preflight[layer])
    chain = [int(c) for c in campaign.records[layer]["adjoint"]["chain_layers"]]
    assert [s["span"] for s in spans].count("checkpoint-load") == 1
    assert [s["layer"] for s in spans if s["span"] == "chain-layer"] == chain
    assert [s["layer"] for s in spans if s["span"] == "own-source"] == [layer]
    assert [s["window"] for s in spans if s["span"] == "window"] == list(range(windows))
    replays = [s for s in spans if s["span"] == "replay"]
    assert sorted((s["window"], s["probe"]) for s in replays) == [
        (w, p) for w in range(windows) for p in range(N_PROBES)]
    assert {s["mode"] for s in replays} == {"spill" if spilled else "window"}
    assert {s["parent"] for s in replays} == {"window"}
    captures = [s["probe"] for s in spans if s["span"] == "spill-capture"]
    assert captures == (list(range(N_PROBES)) if spilled else [])
    assert {s["outcome"] for s in spans} == {"ok"}
    assert all(set(PROC_IO_FIELDS) <= set(s["proc_io"]) for s in spans)
    assert {s["scope"] for s in spans} == {campaign.records[layer]["quantum_id"]}
    rates = [json.loads(line[len(READ_RATE_MARKER) + 1:])
             for line in capsys.readouterr().out.splitlines()
             if line.startswith(READ_RATE_MARKER + " {")]
    final = [r for r in rates if r["final"]]
    assert [r["label"] for r in final] == ["checkpoint-load"]
    assert final[0]["entries"] == final[0]["entries_total"] > 0
    assert final[0]["bytes"] == final[0]["bytes_total"] > 0


def _checkpoint_dir(campaign, layer):
    return Path(campaign.records[layer]["output_space"]["checkpoint_dir"])


def _evidence(campaign, layer, payload):
    """Per-unit envelope digests, unit file digests and cost-row digests.

    ``envelope`` is ``tools/compare_joint_layer_gate.envelope_sha256`` (the
    cutover gate's canonical unit envelope); ``file`` is the SHA-256 of the
    unit's journal file bytes; ``rows`` is the canonical-JSON SHA-256 of each
    cost row of the returned payload.
    """
    from prismaquant.aura_cost import _aura_unit_checkpoint_path
    from tools.compare_joint_layer_gate import (
        _journal_identity, _load_unit, envelope_sha256)

    directory = _checkpoint_dir(campaign, layer)
    journal = _journal_identity(directory)
    units = {}
    for name in sorted(journal["units"]):
        state = _load_unit(directory, name, journal["identity_sha256"])
        units[name] = {
            "envelope": envelope_sha256(state),
            "file": hashlib.sha256(
                Path(_aura_unit_checkpoint_path(directory, name)).read_bytes()).hexdigest(),
        }
    rows = {f"{name}@{fmt}": canonical_json_sha256(row, where="cost row")
            for name, formats in sorted(payload["costs"].items())
            for fmt, row in sorted(formats.items())}
    identity = {key: payload["provenance"][key] for key in
                ("joint_aura_identity_sha256", "probe_identity_sha256",
                 "checkpoint_identity_sha256")}
    identity["journal_identity_sha256"] = journal["identity_sha256"]
    return {"units": units, "rows": rows, "identity": identity}


def _clear_output(campaign, layer):
    shutil.rmtree(Path(campaign.records[layer]["output_space"]["root"]),
                  ignore_errors=True)


def _open_under(root):
    root = str(Path(root).resolve())
    found = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith(root):
            found.append(target)
    return found


def _chain(error):
    messages, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        messages.append(str(error))
        error = error.__cause__ or error.__context__
    return " | ".join(messages)


def _report(name, value):
    """Print the evidence (``-rA`` keeps it in the PB log); optionally save it."""
    text = json.dumps(value, sort_keys=True, default=str)
    print(f"SPILL-EVIDENCE {name} {text}")
    path = os.environ.get("PQ_SPILL_EVIDENCE_DIR")
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / f"{name}.json").write_text(text)


def _spill_root(tmp_path):
    """The declared local root: the environment's, else pytest's tmp_path."""
    from prismaquant.perturbed_x_cache import StageBSpillScratch

    base = Path(os.environ.get("PQ_STAGE_B_SPILL_TEST_ROOT") or tmp_path)
    root = base / f"spill-{os.getpid()}-{tmp_path.name}"
    root.mkdir(parents=True, exist_ok=True)
    StageBSpillScratch.require_local_root(root)  # fail loudly, never skip
    return root


@pytest.mark.parametrize("threads,small", [(True, True), (False, False)],
                         ids=["threaded-small-buffers", "inline-default-buffers"])
def test_spill_replay_is_bitwise_the_windowed_replay(campaign, monkeypatch, tmp_path,
                                                     threads, small):
    from tools.compare_joint_layer_gate import compare_layer

    import prismaquant.format_registry as fr
    import prismaquant.joint_aura as joint

    # The act-quant leg is live: FP8_E4M3 quantizes the input, and both
    # replays call the activation QDQ the same number of times.
    assert fr.get_format("FP8_E4M3").act_quant_changes_input
    qdq_calls = {"count": 0}
    original_qdq = joint._activation_qdq

    def counting_qdq(*args, **kwargs):
        qdq_calls["count"] += 1
        return original_qdq(*args, **kwargs)

    monkeypatch.setattr(joint, "_activation_qdq", counting_qdq)
    monkeypatch.setattr(spill_mod, "DEFAULT_THREADS", threads)
    if small:
        # Arenas and read buffers of one tensor pair: many flushes, many
        # chunks, and inputs shared across chunk boundaries.
        monkeypatch.setattr(spill_mod, "ARENA_BYTES", 1)
        monkeypatch.setattr(spill_mod, "READ_BYTES", 1)
    spill_root = _spill_root(tmp_path)
    evidence = {}
    shared_inputs = 0
    for layer in (1, 0):
        windows = len(campaign.preflight[layer])
        assert windows >= 3
        _clear_output(campaign, layer)
        qdq_calls["count"] = 0
        payload, state = _quantum(campaign, monkeypatch, layer=layer)
        windowed_qdq = qdq_calls["count"]
        assert windowed_qdq > 0
        assert payload is not None, _chain(state.error)
        windowed = _evidence(campaign, layer, payload)
        windowed_replay = state.counters_block["replay"]
        assert windowed_replay["mode"] == "windowed"
        assert windowed_replay["layer_passes"] == windows * N_PROBES
        kept = tmp_path / f"windowed-{layer}"
        shutil.copytree(_checkpoint_dir(campaign, layer), kept)
        _clear_output(campaign, layer)

        qdq_calls["count"] = 0
        payload, state = _quantum(campaign, monkeypatch, layer=layer,
                                  spill_root=spill_root, ceiling=1 << 30)
        assert payload is not None, _chain(state.error)
        assert qdq_calls["count"] == windowed_qdq
        spilled = _evidence(campaign, layer, payload)
        replay = state.counters_block["replay"]
        assert replay["mode"] == "one_pass_spill"
        assert replay["layer_passes"] == N_PROBES
        assert replay["noncontiguous_cotangent_seeds"] == 0
        telemetry = replay["spill"]
        assert telemetry["threads"] is threads
        assert telemetry["x_digest_checks"] > 0
        assert len(set(telemetry["records_per_probe"])) == 1
        assert telemetry["records_per_probe"][0] > 0
        assert 0 < telemetry["x_bytes_written"]
        assert 0 < telemetry["g_bytes_written"]
        assert (telemetry["x_bytes_written"] + telemetry["g_bytes_written"]
                <= telemetry["bound_bytes"])
        shared_inputs += telemetry["shared_input_linears"]
        assert spilled == windowed
        verdict = compare_layer(kept, _checkpoint_dir(campaign, layer),
                                layer=layer, qname_filter=None)
        assert verdict["verdict"] == "match", verdict
        assert verdict["matched"] == verdict["units_shared"] == len(windowed["units"])
        assert os.listdir(spill_root) == [] and not _open_under(spill_root)
        evidence[layer] = {
            "windows": windows, "n_probes": N_PROBES, "device": str(campaign.device),
            "windowed": windowed, "spill": spilled, "compare_layer": verdict,
            "windowed_layer_passes": windowed_replay["layer_passes"],
            "spill_layer_passes": replay["layer_passes"],
            "spill_telemetry": telemetry,
            "spill_geometry": replay["spill_geometry"], "qdq_calls": windowed_qdq}
    # Some Linear read another Linear's input stream (gate and up of one
    # expert, or of the shared expert, in one window).
    assert shared_inputs > 0
    _report(f"bitwise-{campaign.device.type}-{'threaded' if threads else 'inline'}",
            evidence)


def test_spill_resume_after_partial_completion_is_bitwise(campaign, monkeypatch,
                                                         tmp_path):
    layer = 1
    spill_root = _spill_root(tmp_path)
    _clear_output(campaign, layer)
    payload, state = _quantum(campaign, monkeypatch, layer=layer)
    assert payload is not None, _chain(state.error)
    windowed = _evidence(campaign, layer, payload)
    _clear_output(campaign, layer)

    # Fail the second window's first replay: the first window's units are
    # already committed to the journal.
    original = spill_mod.StageBReplaySpill.replay
    failed = {"at": None}

    def replay(self, window_index, probe_index, lease):
        if window_index == 1 and probe_index == 0 and failed["at"] is None:
            failed["at"] = (window_index, probe_index)
            raise RuntimeError("injected failure after the first window")
        return original(self, window_index, probe_index, lease)

    monkeypatch.setattr(spill_mod.StageBReplaySpill, "replay", replay)
    payload, state = _quantum(campaign, monkeypatch, layer=layer, spill_root=spill_root,
                              ceiling=1 << 30)
    assert payload is None and "injected failure" in _chain(state.error)
    # The scratch is gone after the failure: no name, no open descriptor.
    assert os.listdir(spill_root) == [] and not _open_under(spill_root)
    from prismaquant.aura_cost import _aura_unit_checkpoint_path

    layer_names = [name for window in campaign.preflight[layer]
                   for name in window.original_full_target_names]
    committed = sorted(name for name in layer_names if _aura_unit_checkpoint_path(
        _checkpoint_dir(campaign, layer), name).is_file())
    first_window = sorted(campaign.preflight[layer][0].original_full_target_names)
    assert committed == first_window
    monkeypatch.setattr(spill_mod.StageBReplaySpill, "replay", original)

    payload, state = _quantum(campaign, monkeypatch, layer=layer, spill_root=spill_root,
                              ceiling=1 << 30, resume=True)
    assert payload is not None, _chain(state.error)
    replay_block = state.counters_block["replay"]
    assert replay_block["mode"] == "one_pass_spill"
    assert replay_block["layer_passes"] == N_PROBES
    resumed = _evidence(campaign, layer, payload)
    assert resumed == windowed
    assert os.listdir(spill_root) == [] and not _open_under(spill_root)
    _report(f"resume-{campaign.device.type}",
            {"windowed": windowed, "resumed": resumed,
             "committed_before_failure": committed,
             "spill_telemetry": replay_block["spill"],
             "spill_geometry": replay_block["spill_geometry"]})


def _restamp_unit_probe(path, field, value):
    """Rewrite a unit checkpoint's joint rows with ``probe_identity[field] = value``.

    The row digests and the envelope's payload digest are recomputed, so the
    unit's bytes stay intact: only the identity the rows record differs.
    """
    import pickle

    from prismaquant.joint_aura import identity_sha256

    envelope = pickle.loads(path.read_bytes())
    state = pickle.loads(envelope["payload"])
    for entry in state["joint_aura_rows"].values():
        entry["probe_identity"][field] = value
        operator = entry["joint_operator_identity"]
        digest = identity_sha256(entry["probe_identity"])
        entry["probe_identity_sha256"] = operator["probe_identity_sha256"] = digest
        entry["joint_operator_identity_sha256"] = identity_sha256(operator)
    envelope["payload"] = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    envelope["payload_sha256"] = hashlib.sha256(envelope["payload"]).hexdigest()
    path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
    return len(state["joint_aura_rows"])


@pytest.mark.parametrize("field,wall", [("calibration_sha256", True),
                                        ("producer_source_sha256", False)])
def test_resume_splits_a_restored_row_probe_identity(campaign, monkeypatch, capsys,
                                                     field, wall):
    """PQ #1147: a restored row's calibration draw is a wall, its producer source a seal.

    One committed unit's rows are rewritten to record another value of
    ``field``. Another draw refuses the resume in dev mode too. Another
    producer source prints a ``[DEV-MODE]`` line and the row is reused;
    certified mode refuses it as main does.
    """
    from prismaquant.aura_cost import _aura_unit_checkpoint_path

    layer = 1
    _clear_output(campaign, layer)
    payload, state = _quantum(campaign, monkeypatch, layer=layer)
    assert payload is not None, _chain(state.error)
    directory = _checkpoint_dir(campaign, layer)
    for window in campaign.preflight[layer][1:]:
        for name in window.original_full_target_names:
            _aura_unit_checkpoint_path(directory, name).unlink()
    restored = sorted(campaign.preflight[layer][0].original_full_target_names)[0]
    assert _restamp_unit_probe(_aura_unit_checkpoint_path(directory, restored),
                               field, "9" * 64) > 0

    capsys.readouterr()
    payload, state = _quantum(campaign, monkeypatch, layer=layer, resume=True)
    out = capsys.readouterr().out
    if wall:
        assert payload is None
        assert f"joint AURA checkpoint identity mismatch for {restored}@" in _chain(state.error)
        assert "probe/operator alignment mismatch" in _chain(state.error)
        assert "[DEV-MODE] seal joint probe identity" not in out
        return
    assert payload is not None, _chain(state.error)
    assert f"[DEV-MODE] seal joint probe identity differs at {field}" in out

    # Certified at this site. The fixture was captured in dev mode, so only
    # the quantum module's seal_check runs certified here.
    import functools

    import prismaquant.joint_cost_quantum as quantum_mod
    from prismaquant.dev_mode import seal_check

    for window in campaign.preflight[layer][1:]:
        for name in window.original_full_target_names:
            _aura_unit_checkpoint_path(directory, name).unlink()
    monkeypatch.setattr(quantum_mod, "seal_check", functools.partial(
        seal_check, environ={"PRISMAQUANT_DEV_MODE": "0"}))
    payload, state = _quantum(campaign, monkeypatch, layer=layer, resume=True)
    assert payload is None
    assert f"joint AURA checkpoint identity mismatch for {restored}@" in _chain(state.error)
    assert "probe/operator alignment mismatch" in _chain(state.error)


@pytest.mark.parametrize("probe", [1, N_PROBES - 1])
def test_spill_refuses_an_input_that_differs_at_a_later_probe(campaign, monkeypatch,
                                                              tmp_path, probe):
    """One flipped bit in one input of a later probe fails the capture (#1030).

    The input is digested on its device when the hook fires; the flip is made
    there, on a copy of the operand the hook read, so the real digest and the
    real comparison see a real difference.
    """
    layer = 1
    spill_root = _spill_root(tmp_path)
    original = spill_mod.StageBReplaySpill._input_digest
    injected = {"done": False}

    def digest(self, x):
        if self._probe == probe and not injected["done"]:
            injected["done"] = True
            # Same shape and strides, so the storage order is the operand's.
            flipped = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype,
                                          device=x.device)
            flipped.copy_(x.detach())
            spill_mod._storage_order(flipped).view(torch.int16)[-1] ^= 1
            x = flipped
        return original(self, x)

    monkeypatch.setattr(spill_mod.StageBReplaySpill, "_input_digest", digest)
    _clear_output(campaign, layer)
    payload, state = _quantum(campaign, monkeypatch, layer=layer, spill_root=spill_root,
                              ceiling=1 << 30)
    assert payload is None and injected["done"]
    assert f"probe {probe} input differs from probe 0" in _chain(state.error)
    assert os.listdir(spill_root) == [] and not _open_under(spill_root)


def test_only_probe_zero_inputs_reach_the_host(campaign, monkeypatch, tmp_path):
    """A later probe's input is digested on its device and never staged (#1030)."""
    layer = 1
    staged: dict[int, int] = {}
    original = spill_mod.StageBReplaySpill._stage

    def stage(self, window_index, stream, *args):
        if stream[0] == "x":
            staged[self._probe] = staged.get(self._probe, 0) + 1
        return original(self, window_index, stream, *args)

    monkeypatch.setattr(spill_mod.StageBReplaySpill, "_stage", stage)
    _clear_output(campaign, layer)
    payload, state = _quantum(campaign, monkeypatch, layer=layer,
                              spill_root=_spill_root(tmp_path), ceiling=1 << 30)
    assert payload is not None, _chain(state.error)
    spill = state.counters_block["replay"]["spill"]
    assert set(staged) == {0} and staged[0] == spill["x_entries"]
    assert spill["x_digest_checks"] == (N_PROBES - 1) * spill["x_entries"]


def _bf16_expert_fixture():
    """``test_quantum_executable_readset``'s expert runner, in bf16."""
    import test_streamed_cost_checkpoints as tsc
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

    torch.manual_seed(109)
    seed = tsc._ExpertTinyLM().eval()
    state = {name: tensor.detach().clone() for name, tensor in seed.state_dict().items()}
    model = tsc._ExpertTinyLM(state).eval().to(DTYPE)
    for layer in model.model.layers:
        layer._fixture_requires_stream_residency = True
    context = tsc._FakeStreamingContext(model)
    runner = StreamedCausalLM(context, DeepseekV4Profile())
    install = context.install
    context.install = (lambda layer, *, require_prefetched=False, prefetch_following=True:
                       install(layer, require_prefetched=require_prefetched))
    runner.require_prefetched_residency = True
    runner.prefetch_lookahead = 1
    return model, context, runner


def test_spill_keeps_the_executable_readset_phase_order(tmp_path, monkeypatch):
    """Spill mode on a bound executable readset: phases, reads and records.

    The executable-readset acceptance harness drives the real quantum with
    its reporter and read seams instrumented. In spill mode the plan is
    sealed for the spill (PQ #1011): the phases reported equal the sealed
    plan in order, every target boundary read happens under an
    already-reported ``spill-p{probe}`` phase with a retained window open,
    the target layer's boundaries are read once per probe instead of once
    per (window, probe), the read plan counts them once per probe, and the
    cost rows equal the windowed run's. A plan sealed for one mode refuses
    a launch in the other before any GPU work. A resume that has committed
    window zero captures under the same phases with no retained window
    open, before the first active window opens (PQ #1172).
    """
    import test_quantum_executable_readset as phases

    monkeypatch.setattr(phases, "_expert_fixture", _bf16_expert_fixture)
    for name in spill_mod.SPILL_ENV:
        monkeypatch.delenv(name, raising=False)
    setup = phases._acceptance_setup_expert(tmp_path, monkeypatch)
    record0 = setup["records"]["layer-000"]
    n_windows = len(record0["windows"])
    assert n_windows >= 2
    constructed = []
    original_init = spill_mod.StageBReplaySpill.__init__

    def init(self, **kwargs):
        constructed.append(kwargs["geometry"].as_dict())
        original_init(self, **kwargs)

    monkeypatch.setattr(spill_mod.StageBReplaySpill, "__init__", init)

    def target_reads(events):
        return sum(1 for event in events if event[0] == "boundary-open" and event[1] == 0)

    def rows(payload):
        return {f"{name}@{fmt}": canonical_json_sha256(row, where="cost row")
                for name, formats in sorted(payload["costs"].items())
                for fmt, row in sorted(formats.items())}

    events_w, _manifest_w, payload_w, resolved = phases._drive_quantum(
        tmp_path, monkeypatch, setup, layer=0, resume=False)
    assert not constructed
    shutil.rmtree(record0["output_space"]["root"])

    from prismaquant.joint_cost_quantum import QuantumIdentityRefused

    # A windowed launch of a spill-sealed plan refuses before any read.
    with pytest.raises(QuantumIdentityRefused, match="sealed for the spill"):
        phases._drive_quantum(tmp_path, monkeypatch, setup, layer=0,
                              resume=False, replay_mode="spill")
    assert not constructed
    shutil.rmtree(record0["output_space"]["root"], ignore_errors=True)

    spill_root = _spill_root(tmp_path)
    monkeypatch.setenv(spill_mod.SPILL_ENV[0], str(spill_root))
    monkeypatch.setenv(spill_mod.SPILL_ENV[1], str(1 << 30))
    # ... and a spill launch of a windowed plan.
    with pytest.raises(QuantumIdentityRefused, match="sealed for the windowed"):
        phases._drive_quantum(tmp_path, monkeypatch, setup, layer=0,
                              resume=False)
    assert not constructed
    shutil.rmtree(record0["output_space"]["root"], ignore_errors=True)

    # The sealed spill bound: a geometry the live modules do not reproduce,
    # and a ceiling other than the sealed reservation, refuse before the
    # scratch exists. Both are run seals (PQ #1147): certified mode refuses
    # them; dev mode stamps them (tests/test_stage_b_spill_ceiling_sealed.py).
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")

    def other_geometry(bound):
        bound = json.loads(json.dumps(bound))
        bound["geometry"]["x_bytes"] += 2
        return bound

    with pytest.raises(QuantumIdentityRefused, match="seals spill geometry"):
        phases._drive_quantum(tmp_path, monkeypatch, setup, layer=0, resume=False,
                              replay_mode="spill", spill_bound_edit=other_geometry)
    shutil.rmtree(record0["output_space"]["root"], ignore_errors=True)
    with pytest.raises(QuantumIdentityRefused, match="the spill ceiling is 1073741824"):
        phases._drive_quantum(tmp_path, monkeypatch, setup, layer=0, resume=False,
                              replay_mode="spill", spill_ceiling=1 << 30)
    assert not constructed
    assert os.listdir(spill_root) == [] and not _open_under(spill_root)
    shutil.rmtree(record0["output_space"]["root"], ignore_errors=True)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    events_s, manifest_s, payload_s, _ = phases._drive_quantum(
        tmp_path, monkeypatch, setup, layer=0, resume=False,
        replay_mode="spill")
    assert len(constructed) == 1
    phases._assert_acceptance_run(events_s, manifest_s, record0, tmp_path,
                                  expect_replay_windows=set())
    sealed = [phase["name"] for phase in manifest_s["read_plan"]["phases"]]
    reported = phases._reported(events_s)
    assert [name for i, name in enumerate(reported)
            if i == 0 or name != reported[i - 1]] == sealed
    assert sealed[-4:] == [f"spill-p{probe}" for probe in range(4)]
    assert manifest_s["annotations"]["replay_mode"] == "spill"
    by_phase = {phase["name"]: phase for phase in manifest_s["read_plan"]["phases"]}
    windowed_manifest = _manifest_w
    own_run = sum(phase["bytes"] for phase in windowed_manifest["read_plan"]["phases"]
                  if phase["name"].startswith("replay-00-"))
    assert sum(by_phase[f"spill-p{probe}"]["bytes"] for probe in range(4)) == own_run
    assert (windowed_manifest["read_plan"]["read_bytes"]
            - manifest_s["read_plan"]["read_bytes"]) == (n_windows - 1) * own_run
    assert target_reads(events_w) == n_windows * target_reads(events_s) > 0
    assert rows(payload_s) == rows(payload_w)
    assert os.listdir(spill_root) == [] and not _open_under(spill_root)

    # Partial resume in spill mode: window 0 is committed, so every probe is
    # captured in window zero's slot with no retained window open, and
    # windows 1+ then open and replay from the spill (PQ #1172).
    from prismaquant.aura_cost import _aura_unit_checkpoint_path
    checkpoint_dir = Path(record0["output_space"]["checkpoint_dir"])
    for name in (n for window in resolved[1:] for n in window["names"]):
        _aura_unit_checkpoint_path(checkpoint_dir, name).unlink()
    events_p, manifest_p, payload_p, _ = phases._drive_quantum(
        tmp_path, monkeypatch, setup, layer=0, resume=True,
        replay_mode="spill")
    assert len(constructed) == 2
    assert constructed[1]["x_bytes"] < constructed[0]["x_bytes"]
    phases._assert_acceptance_run(events_p, manifest_p, record0, tmp_path,
                                  expect_replay_windows=set(),
                                  require_window=False)
    opens = [i for i, event in enumerate(events_p) if event[0] == "window-open"]
    captures = [i for i, event in enumerate(events_p)
                if event[0] == "boundary-open" and event[1] == 0]
    assert len(opens) == n_windows - 1
    assert captures and max(captures) < min(opens), (captures, opens)
    assert target_reads(events_p) == target_reads(events_s)
    assert rows(payload_p) == rows(payload_w)
    assert os.listdir(spill_root) == [] and not _open_under(spill_root)
    _report("executable-readset-spill", {
        "windows": n_windows, "windowed_target_reads": target_reads(events_w),
        "spill_target_reads": target_reads(events_s),
        "rows": rows(payload_s), "geometry": constructed})


@pytest.mark.parametrize("ceiling,message", [
    (None, "positive PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES"),
    ("1", "needs"),
])
def test_spill_ceiling_refuses_before_any_gpu_work(campaign, monkeypatch, tmp_path,
                                                   ceiling, message):
    layer = 0
    spill_root = _spill_root(tmp_path)
    _clear_output(campaign, layer)
    payload, state = _quantum(campaign, monkeypatch, layer=layer, spill_root=spill_root,
                              ceiling=ceiling)
    assert payload is None and message in _chain(state.error)
    # Refused before the chain: no layer was installed, no checkpoint read.
    assert state.context.install_calls == 0
    assert os.listdir(spill_root) == [] and not _open_under(spill_root)


def _observations(monkeypatch):
    """Record every window lease's per-Linear (observed tokens, calls)."""
    import prismaquant.joint_aura as joint

    seen = []
    original = joint.JointOperatorStatisticsLease.operator_diagnostics

    def recording(self, *, collect_col_energy):
        result = original(self, collect_col_energy=collect_col_energy)
        seen.append({name: (row["observed_tokens"], row["observed_calls"])
                     for name, row in result.items()})
        return result

    monkeypatch.setattr(joint.JointOperatorStatisticsLease, "operator_diagnostics",
                        recording)
    return seen


def _token_totals(seen):
    totals = {}
    for window in seen:
        for name, (tokens, _calls) in window.items():
            totals[name] = totals.get(name, 0) + tokens
    return totals


def test_batched_spill_capture_is_stamped_and_reruns_bitwise(campaign, monkeypatch,
                                                             tmp_path):
    """capture_batch=2: two stored batches per layer pass, stamped rows.

    A rerun at the same regime is bitwise (the rows and unit files hash
    equal). Against the batch-1 spill, every dense Linear observes the same
    tokens, and a packed projection's experts observe the same total, since
    every token still routes to top-k experts; a route flip only moves rows
    between experts.
    """
    from prismaquant.joint_replay_regime import normalize_replay_regime, replay_regime_of

    regime = "capture_batch=2"
    layer = 0
    seen = _observations(monkeypatch)
    _clear_output(campaign, layer)
    payload, state = _quantum(campaign, monkeypatch, layer=layer,
                              spill_root=_spill_root(tmp_path / "b1"), ceiling=1 << 30)
    assert payload is not None, _chain(state.error)
    single = _token_totals(seen)
    single_evidence = _evidence(campaign, layer, payload)
    runs = []
    for attempt in range(2):
        seen.clear()
        _clear_output(campaign, layer)
        spill_root = _spill_root(tmp_path / f"b2-{attempt}")
        payload, state = _quantum(campaign, monkeypatch, layer=layer,
                                  spill_root=spill_root, ceiling=1 << 30, regime=regime)
        assert payload is not None, _chain(state.error)
        for rows in payload["costs"].values():
            for row in rows.values():
                assert replay_regime_of(row["probe_identity"]["arithmetic"]) == (
                    normalize_replay_regime(regime))
        replay = state.counters_block["replay"]
        assert replay["mode"] == "one_pass_spill"
        assert replay["layer_passes"] == N_PROBES
        assert replay["capture_groups"] == 2
        assert replay["regime"]["capture_batch"] == 2
        assert replay["row_local_qdq"] == [["FP8_E4M3", WIDTH], ["FP8_E4M3", INTER]]
        assert len(set(replay["spill"]["records_per_probe"])) == 1
        assert os.listdir(spill_root) == [] and not _open_under(spill_root)
        runs.append((_evidence(campaign, layer, payload), _token_totals(seen)))
    assert runs[0][0] == runs[1][0]
    assert runs[0][0]["identity"] != single_evidence["identity"]
    batched = runs[0][1]
    assert set(batched) == set(single)
    model, _context, runner = _runner(campaign.state, campaign.device)
    linears = _targets(model, runner.profile)
    packed = {}
    for name, tokens in single.items():
        module = linears[name]
        if isinstance(module, spill_mod.PackedExpertProjection):
            key = (module.module_qname, module.projection_name)
            totals = packed.setdefault(key, [0, 0])
            totals[0] += tokens
            totals[1] += batched[name]
        else:
            assert batched[name] == tokens, name
    assert packed and all(left == right for left, right in packed.values())
    _report(f"batched-rerun-{campaign.device.type}", {
        "regime": regime, "evidence": runs[0][0],
        "route_moves": sum(abs(batched[name] - single[name]) for name in single)})


def test_capture_pass_charges_the_planned_workspace_per_stored_batch(campaign, monkeypatch,
                                                                     tmp_path):
    """The guard charges the capture the quantity the plan priced (PQ #1151).

    The operator windows here declare a workspace five times the retained
    budget's. The retained budget is what the derivation plans the capture
    pass with (``RetainedWindowBudget.capture_peak_bytes``), so every capture
    admission must be the budget's reserve times the capture batch, plus the
    spill's own capture reserve on its two sides. Before #1151 the guard
    multiplied the operator windows' reserve instead, a quantity no plan
    priced.
    """
    regime, batch = "capture_batch=2", 2
    layer = 0
    policy, budget, retained = _policy_budget()
    wide = dict(policy, workspace_reserve_bytes=5 * budget.workspace_reserve_bytes)
    monkeypatch.setitem(globals(), "_policy_budget", lambda: (wide, budget, retained))
    sides = _record_capture_sides(monkeypatch)
    guard = _DeviceRecordingGuard(campaign.device)
    _clear_output(campaign, layer)
    payload, state = _quantum(campaign, monkeypatch, layer=layer, guard=guard,
                              spill_root=_spill_root(tmp_path), ceiling=1 << 30,
                              regime=regime)
    assert payload is not None, _chain(state.error)
    captures = [(host, device) for label, host, device in guard.admissions
                if label == "before_joint_window_backward"]
    hosts = sides["capture_reserve_host_bytes"]
    devices = sides["capture_reserve_device_bytes"]
    assert len(captures) == N_PROBES == len(hosts) == len(devices)
    # The workspace rides the device side with the spill's held inputs
    # (PQ #1157); the spill's pinned arenas are the only host charge.
    assert captures == [(host, batch * budget.workspace_reserve_bytes + device)
                        for host, device in zip(hosts, devices)]
    assert budget.capture_workspace_bytes(batch) == batch * budget.workspace_reserve_bytes


def _operator_snapshots(monkeypatch):
    """Record every window lease's FP32 statistics matrices, in order."""
    import prismaquant.joint_aura as joint

    seen = []
    original = joint.JointOperatorStatisticsLease.finish_observations

    def recording(self):
        seen.append({key: value.detach().to("cpu", copy=True)
                     for key, value in self._operators.items()})
        return original(self)

    monkeypatch.setattr(joint.JointOperatorStatisticsLease, "finish_observations",
                        recording)
    return seen


def _relative_frobenius(left, right):
    """max over (window, probe, key) of ||left - right|| / ||right||."""
    assert len(left) == len(right)
    worst = 0.0
    for mine, reference in zip(left, right):
        assert set(mine) == set(reference)
        for key, matrix in reference.items():
            norm = float(torch.linalg.matrix_norm(matrix.double()))
            delta = float(torch.linalg.matrix_norm((mine[key] - matrix).double()))
            worst = max(worst, delta / norm if norm else delta)
    return worst


@pytest.mark.parametrize("regime", [
    "accumulation=operator_gemm,chunk_rows=5",
    "accumulation=operator_gemm,chunk_rows=65536",
    "capture_batch=2,accumulation=operator_gemm,chunk_rows=7",
])
def test_operator_gemm_replay_matches_the_invocation_counts_and_reruns_bitwise(
        campaign, monkeypatch, tmp_path, regime):
    """One GEMM per operator over a Linear's spilled rows, chunked.

    At capture batch 1 it observes exactly the bitwise replay's tokens and
    calls per Linear, and its statistics matrices differ from the bitwise
    ones only by FP32 summation order. A rerun at the same regime is
    bitwise. A chunk of 5 or 7 rows cuts across invocations; 65536 is one
    GEMM per operator per window and probe.
    """
    from prismaquant.joint_replay_regime import normalize_replay_regime, replay_regime_of

    expected = normalize_replay_regime(regime)
    layer = 0
    seen = _observations(monkeypatch)
    matrices = _operator_snapshots(monkeypatch)
    _clear_output(campaign, layer)
    payload, state = _quantum(campaign, monkeypatch, layer=layer,
                              spill_root=_spill_root(tmp_path / "bitwise"), ceiling=1 << 30)
    assert payload is not None, _chain(state.error)
    bitwise_seen, bitwise_matrices = list(seen), list(matrices)
    runs = []
    for attempt in range(2):
        seen.clear()
        matrices.clear()
        _clear_output(campaign, layer)
        spill_root = _spill_root(tmp_path / f"gemm-{attempt}")
        payload, state = _quantum(campaign, monkeypatch, layer=layer, spill_root=spill_root,
                                  ceiling=1 << 30, regime=regime)
        assert payload is not None, _chain(state.error)
        for rows in payload["costs"].values():
            for row in rows.values():
                arithmetic = row["probe_identity"]["arithmetic"]
                assert replay_regime_of(arithmetic) == expected
                assert arithmetic["operator_accumulation"] == (
                    "fp32_gemm_over_spilled_rows_in_capture_order_by_row_chunk")
        telemetry = state.counters_block["replay"]["spill"]
        assert telemetry["accumulation"] == "operator_gemm"
        assert telemetry["chunk_rows"] == expected["chunk_rows"]
        assert telemetry["row_chunks"] > 0
        assert os.listdir(spill_root) == [] and not _open_under(spill_root)
        runs.append((_evidence(campaign, layer, payload), list(seen), list(matrices)))
    assert runs[0][0] == runs[1][0]
    assert all(torch.equal(a[key], b[key]) for a, b in zip(runs[0][2], runs[1][2])
               for key in b)
    if expected["capture_batch"] == 1:
        assert runs[0][1] == bitwise_seen
        worst = _relative_frobenius(runs[0][2], bitwise_matrices)
        assert worst < 1e-5, worst
    else:
        assert _token_totals(runs[0][1]).keys() == _token_totals(bitwise_seen).keys()
        worst = _relative_frobenius(runs[0][2], bitwise_matrices)
    _report(f"operator-gemm-{campaign.device.type}-{regime}", {
        "regime": regime, "worst_relative_frobenius": worst,
        "row_chunks": telemetry["row_chunks"]})


def test_capture_batch_that_splits_a_read_window_refuses_before_any_gpu_work(
        campaign, monkeypatch, tmp_path):
    _clear_output(campaign, 0)
    payload, state = _quantum(campaign, monkeypatch, layer=0,
                              spill_root=_spill_root(tmp_path), ceiling=1 << 30,
                              regime="capture_batch=4")
    assert payload is None
    assert "does not divide the sealed read window of 2" in _chain(state.error)
    assert state.context.install_calls == 0


def test_batched_capture_refuses_shared_pass_state_before_the_chain(
        campaign, monkeypatch, tmp_path):
    monkeypatch.setattr(Lfm2MoeProfile, "isolated_layer_pass_state",
                        lambda self, captured, layer: {"shared_kv": object()})
    _clear_output(campaign, 0)
    spill_root = _spill_root(tmp_path)
    payload, state = _quantum(campaign, monkeypatch, layer=0, spill_root=spill_root,
                              ceiling=1 << 30, regime="capture_batch=2")
    assert payload is None
    assert "carries shared pass state" in _chain(state.error)
    assert state.context.install_calls == 0
    assert os.listdir(spill_root) == [] and not _open_under(spill_root)


def _published_plane(handoff):
    from test_quantum_band_serial import _handoff_plane, _state_digest, _tensor_digest

    document, plane, states = _handoff_plane(handoff)
    return (document["boundary"],
            {key: _tensor_digest(tensor) for key, tensor in sorted(plane.items())},
            {key: _state_digest(state) for key, state in sorted(states.items())})


@pytest.mark.parametrize("regime", [
    "capture_batch=2",
    "capture_batch=2,accumulation=operator_gemm,chunk_rows=7",
    "accumulation=operator_gemm,chunk_rows=5",
])
def test_a_band_serial_producer_captures_at_the_chains_batch_size(campaign, monkeypatch,
                                                                tmp_path, regime):
    """A band-serial producer (PQ #996) hands off the plane its capture wrote.

    The handoff must equal the plane the consumer's chain rebuild ends on,
    which rolls at the Stage A slice's chain batch size (PQ #997): batch 1 on
    this fixture, so a capture batch above 1 refuses before any GPU work
    (``tests/test_band_serial_batched_regime.py`` runs the batch-4 case).
    One GEMM per operator changes only the statistics: its producer hands off
    the default spill's plane, sha256-equal entry by entry.
    """
    from prismaquant.joint_replay_regime import normalize_replay_regime

    spill_root = _spill_root(tmp_path)
    if normalize_replay_regime(regime)["capture_batch"] > 1:
        _clear_output(campaign, 1)
        payload, state = _quantum(campaign, monkeypatch, layer=1, spill_root=spill_root,
                                  ceiling=1 << 30, regime=regime, emit_handoff=True)
        assert payload is None
        assert ("chain regime has batch size 1, and the handoff plane is captured "
                "at batch 2") in _chain(state.error)
        assert state.context.install_calls == 0
        assert state.counters.replay["layer_passes"] == 0
        return
    planes = []
    for label, run_regime in (("default", None), ("gemm", regime)):
        _clear_output(campaign, 1)
        payload, state = _quantum(campaign, monkeypatch, layer=1,
                                  spill_root=_spill_root(tmp_path / label), ceiling=1 << 30,
                                  regime=run_regime, emit_handoff=True)
        assert payload is not None, _chain(state.error)
        assert state.handoff is not None
        planes.append(_published_plane(state.handoff))
    assert planes[0][0] == 1 and planes[0][1]
    assert planes[0] == planes[1]


def test_row_local_qdq_admission_refuses_a_tensor_wide_scale(monkeypatch):
    import prismaquant.format_registry as fr
    import prismaquant.perturbed_x_cache as pxc

    linear = nn.Linear(WIDTH, 3, bias=False).to(DTYPE)
    specs = {"u": {"FP8_E4M3": fr.get_format("FP8_E4M3"),
                   "NVFP4A16": fr.get_format("NVFP4A16")}}
    assert spill_mod.require_row_local_activation_qdq(
        {"u": linear}, specs, {}, device="cpu", dtype=DTYPE) == [("FP8_E4M3", WIDTH)]

    def tensor_wide(x, spec, maxima, name, *args, **kwargs):
        scale = x.float().abs().amax() / 448.0
        return ((x.float() / scale).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)

    monkeypatch.setattr(pxc, "_activation_qdq", tensor_wide)
    with pytest.raises(spill_mod.ReplayRegimeInadmissible, match="not row-local"):
        spill_mod.require_row_local_activation_qdq(
            {"u": linear}, specs, {}, device="cpu", dtype=DTYPE)


@pytest.mark.parametrize("served", [False, True], ids=["dynamic-rtn", "served-static-scale"])
def test_row_local_qdq_admission_admits_the_nvfp4_activation_paths(monkeypatch, served):
    """W4A4 rows reach the QDQ through NVFP4's static activation contract.

    By default that is the dynamic per-16-group RTN screen; with
    ``PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES=1`` and a calibrated
    maximum it is the contract's static-scale oracle, the same
    ``quantize_dequantize`` a Tessera ``measured_as_served`` row calls. Both
    are row-local on this device, so the admission passes them.
    """
    import prismaquant.format_registry as fr
    import prismaquant.perturbed_x_cache as pxc

    if served:
        monkeypatch.setenv("PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES", "1")
    else:
        monkeypatch.delenv("PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES", raising=False)
    assert pxc._served_nvfp4_act_qdq_enabled() is served
    spec = fr.get_format("NVFP4")
    assert spec.act_quant_changes_input and spec.static_activation_contract is not None
    device = _device()
    linear = nn.Linear(64, 3, bias=False).to(device=device, dtype=DTYPE)
    assert spill_mod.require_row_local_activation_qdq(
        {"u": linear}, {"u": {"NVFP4": spec}}, {"u": 4.0},
        device=device, dtype=DTYPE) == [("NVFP4", 64)]


def test_replay_regime_without_the_spill_refuses_before_any_gpu_work(
        campaign, monkeypatch):
    _clear_output(campaign, 0)
    payload, state = _quantum(campaign, monkeypatch, layer=0, regime="capture_batch=2")
    assert payload is None
    assert "replays from the spill" in _chain(state.error)
    assert state.context.install_calls == 0
    assert state.counters.replay["layer_passes"] == 0


@pytest.mark.parametrize("root,message", [
    ("/mnt/shared/stage-b-spill", "must not be under /mnt/shared"),
    ("/dev/shm", "local ext4/xfs/btrfs"),
    ("relative/spill", "absolute"),
])
def test_spill_root_refuses_shared_and_memory_filesystems(root, message):
    from prismaquant.perturbed_x_cache import StageBSpillScratch

    if root == "/dev/shm" and not Path(root).is_dir():
        pytest.skip("this host has no /dev/shm tmpfs")
    with pytest.raises(ValueError, match=message):
        StageBSpillScratch(directory=root, max_bytes=1 << 20, nbytes=1024)


def test_spill_scratch_is_unnamed_and_bounded(tmp_path):
    from prismaquant.perturbed_x_cache import StageBSpillScratch

    root = _spill_root(tmp_path)
    scratch = StageBSpillScratch(directory=root, max_bytes=1 << 20, nbytes=3000,
                                 parts=2, part_padding=512, alignment=512)
    try:
        block = scratch.block
        assert block >= 512 and not block & (block - 1)
        # The payload plus every slot's worst-case padding, on the grid.
        reserve = 3000 + 2 * (512 + block)
        assert scratch.capacity == reserve + (-reserve) % block
        assert os.listdir(root) == []
        assert _open_under(root)
        source = spill_mod._aligned_buffer(2 * block, block, False)
        assert source.data_ptr() % block == 0
        source.copy_(torch.arange(2 * block).remainder(251).to(torch.uint8))
        view = memoryview(source.numpy())
        offset = scratch.allocate(2 * block)
        assert offset == 0
        # One call per block at a one-block call size; one call for both.
        assert scratch.write(offset, [view[:block], view[block:]], call_bytes=block) == 2
        assert scratch.write(offset, [view[:block], view[block:]], call_bytes=4 * block) == 1
        target = spill_mod._aligned_buffer(2 * block, block, False)
        assert scratch.read_into(offset, [memoryview(target.numpy())]) == 2 * block
        assert torch.equal(target, source)
        # Split reads land in their own views.
        target.zero_()
        tail = memoryview(target.numpy())
        assert scratch.read_into(offset + block, [tail[:block]]) == block
        assert torch.equal(target[:block], source[block:])
        with pytest.raises(ValueError, match="direct-I/O block"):
            scratch.allocate(block + 1)
        with pytest.raises(RuntimeError, match="direct-I/O grid"):
            scratch.write(offset, [view[:block // 2]], call_bytes=block)
        with pytest.raises(RuntimeError, match="outside its allocation"):
            scratch.read_into(offset + block, [memoryview(target.numpy())])
        with pytest.raises(RuntimeError, match="geometry bound"):
            scratch.allocate(scratch.capacity)
    finally:
        scratch.close()
    assert not _open_under(root) and os.listdir(root) == []
    # The payload alone over the ceiling refuses before a file exists.
    with pytest.raises(RuntimeError, match="ceiling is 100"):
        StageBSpillScratch(directory=root, max_bytes=100, nbytes=101)
    # So does the slot padding on top of a payload that fits.
    with pytest.raises(RuntimeError, match="ceiling is 4096"):
        StageBSpillScratch(directory=root, max_bytes=4096, nbytes=4096,
                           parts=1, part_padding=512)
    with pytest.raises(ValueError, match="power of two"):
        StageBSpillScratch(directory=root, max_bytes=1 << 20, nbytes=4096,
                           alignment=768)
    assert not _open_under(root) and os.listdir(root) == []


def test_spill_slot_keeps_the_replay_residue_on_the_grid():
    block = spill_mod.ADDRESS_ALIGNMENT
    assert spill_mod._slot(0, 0, 0, block) == (0, 0, 0)
    assert spill_mod._slot(1, 7, 0, block) == (block, block, block)
    assert spill_mod._slot(1, 100, 1000, block) == (block, block + 100, 4 * block)
    assert spill_mod._slot(2 * block, 0, block, block) == (2 * block, 2 * block, 3 * block)
    for grid in (block, 8 * block):
        for cursor in range(0, 3 * grid, 97):
            for residue in (0, 2, 254, block - 2):
                for nbytes in (2, 510, 512, 514, 4096, 5000):
                    start, offset, end = spill_mod._slot(cursor, residue, nbytes, grid)
                    assert start % grid == 0 and 0 <= start - cursor < grid
                    assert offset % block == residue
                    assert end % grid == 0 and 0 <= end - (offset + nbytes) < grid
                    # The padding the scratch reserves per part bounds the slot.
                    assert end - start < nbytes + block + grid


def test_spill_refuses_a_non_dense_input(tmp_path):
    geometry = spill_mod.spill_geometry(
        {"a": nn.Linear(8, 8, bias=False)}, [("a",)], pending={"a"},
        batch_tokens=[4], n_probes=1, element_size=2, experts_per_token=None)
    session = spill_mod.StageBReplaySpill(
        root=_spill_root(tmp_path), max_bytes=1 << 20, geometry=geometry,
        window_names=[("a",)], n_probes=1, dtype=DTYPE, device="cpu", threads=False)
    try:
        session._probe = 0
        session._start_arenas()
        x = torch.zeros(4, 16, dtype=DTYPE)[:, ::2]
        with pytest.raises(RuntimeError, match="non-dense input"):
            session._record("a", x, torch.zeros(4, 8, dtype=DTYPE))
    finally:
        session.close()


def test_spill_refuses_fp32_measurement(tmp_path):
    geometry = spill_mod.spill_geometry(
        {"a": nn.Linear(8, 8, bias=False)}, [("a",)], pending={"a"},
        batch_tokens=[4], n_probes=1, element_size=4, experts_per_token=None)
    with pytest.raises(RuntimeError, match="16-bit"):
        spill_mod.StageBReplaySpill(
            root=tmp_path, max_bytes=1 << 20, geometry=geometry,
            window_names=[("a",)], n_probes=1, dtype=torch.float32, device="cpu")
    assert os.listdir(tmp_path) == []


def test_spill_geometry_counts_shared_expert_rows_once_per_window():
    torch.manual_seed(1)
    model = _MoELM()
    linears = _targets(model, Lfm2MoeProfile())
    layer = sorted(n for n in linears if ".layers.0." in n)
    packed = [n for n in layer if ".experts." in n and ".shared_expert." not in n]
    dense = [n for n in layer if n not in packed]
    assert len(packed) == 3 * EXPERTS and len(dense) == 3
    tokens = [8, 8]
    geometry = spill_mod.spill_geometry(
        linears, [tuple(layer)], pending=set(layer), batch_tokens=tokens,
        n_probes=2, element_size=2, experts_per_token=TOP_K)
    total = sum(tokens)
    # Dense: gate, up and down each spill their input; x is not deduplicated
    # in the bound. Packed: one x per parameter over min(experts, top_k) rows.
    x = (WIDTH + WIDTH + INTER) * total + (WIDTH + INTER) * TOP_K * total
    g = (INTER + INTER + WIDTH) * total + (INTER + INTER + WIDTH) * TOP_K * total
    assert geometry.x_bytes == 2 * x and geometry.g_bytes_per_probe == 2 * g
    assert geometry.total_bytes == 2 * x + 2 * 2 * g
    # Per sample: an input and a gradient per probe for every target.
    assert geometry.max_parts == (2 + 1) * len(layer) * len(tokens)
    unbounded = spill_mod.spill_geometry(
        linears, [tuple(layer)], pending=set(layer), batch_tokens=tokens,
        n_probes=2, element_size=2, experts_per_token=None)
    assert unbounded.x_bytes > geometry.x_bytes

    # One window per expert: a token still visits TOP_K experts in all, so
    # the layer bound is unchanged, while the per-window bounds add up to
    # TOP_K rows per window.
    def expert_of(name):
        return linears[name].expert_id

    per_expert = [tuple(dense)] + [
        tuple(n for n in packed if expert_of(n) == expert) for expert in range(EXPERTS)]
    split = spill_mod.spill_geometry(
        linears, per_expert, pending=set(layer), batch_tokens=tokens,
        n_probes=2, element_size=2, experts_per_token=TOP_K)
    assert split.total_bytes == geometry.total_bytes
    assert sum(split.window_x_bytes) > split.x_bytes
    assert sum(split.window_g_bytes) > split.g_bytes_per_probe

    # Gate and up of one expert in two windows: that expert's input is
    # written twice, and a token may visit it, so the bound grows by one
    # expert's input rows per token, not by TOP_K of them.
    def param_of(name):
        return (linears[name].module_qname, linears[name].param_name)

    gate = next(n for n in packed if expert_of(n) == 0 and sum(
        param_of(m) == param_of(n) and expert_of(m) == 0 for m in packed) == 2)
    apart = [tuple(n for n in layer if n != gate), (gate,)]
    twice = spill_mod.spill_geometry(
        linears, apart, pending=set(layer), batch_tokens=tokens,
        n_probes=2, element_size=2, experts_per_token=TOP_K)
    assert twice.x_bytes == geometry.x_bytes + 2 * linears[gate].weight.shape[1] * total
    assert twice.g_bytes_per_probe == geometry.g_bytes_per_probe


def test_container_forwards_the_spill_through_an_identity_bind():
    from tools.tessera_campaign_container import (
        STAGE_B_SPILL_ENV, stage_b_spill_environment)

    assert STAGE_B_SPILL_ENV == spill_mod.SPILL_ENV
    spec = {"container": {"mounts": [
        {"source": "/nvme/scratch", "target": "/nvme/scratch"}]}, "env": {}}
    environ = {STAGE_B_SPILL_ENV[0]: "/nvme/scratch/spill",
               STAGE_B_SPILL_ENV[1]: str(200 << 30)}
    assert stage_b_spill_environment(spec, environ) == environ
    assert stage_b_spill_environment(spec, {}) == {}
    shared = {"container": {"mounts": [
        {"source": "/mnt/shared/x", "target": "/mnt/shared/x"}]}, "env": {}}
    with pytest.raises(RuntimeError, match="/mnt/shared"):
        stage_b_spill_environment(shared, {STAGE_B_SPILL_ENV[0]: "/mnt/shared/x/spill",
                                           STAGE_B_SPILL_ENV[1]: "100"})
    with pytest.raises(RuntimeError, match="identity bind"):
        stage_b_spill_environment({"container": {"mounts": []}, "env": {}}, environ)
    with pytest.raises(RuntimeError, match="positive byte ceiling"):
        stage_b_spill_environment(spec, {STAGE_B_SPILL_ENV[0]: "/nvme/scratch/spill"})
