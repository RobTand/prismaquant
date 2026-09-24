"""The distributed campaign's per-quantum runtime gates (#778, §11).

Stage A (adjoint capture) and stage B (the layer quantum) against the
contract in ``docs/design/distributed_campaign_2026-09-19.md``: identity
refusals (exit 3, nothing written), the strided-checkpoint chain's equality
with the single-run path (bitwise unit envelopes -- the §9.3 gate's dry
run), write-space isolation, chunk-granular progress cadence, §8.1 counter
shapes, and the comparison tool on a synthetic pair.
"""
from pathlib import Path
import hashlib
import json
import os

import pytest
import torch

import prismaquant.aura_cost as aura
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.cost_streaming import StreamedBoundaryArtifacts
from prismaquant.joint_adjoint_checkpoints import (
    adjoint_space,
    adjoint_receipt_path,
    chain_layers_for,
    derive_checkpoint_boundaries,
    load_adjoint_checkpoint,
    write_adjoint_receipt,
)
from prismaquant.joint_adjoint_slices import (
    adjoint_slice_sha256,
    stage_a_slice,
    write_adjoint_slice,
)
from prismaquant.joint_cost_quantum import (
    EXIT_IDENTITY_REFUSED,
    IDENTITY_REFUSED_MARKER,
    QuantumCounters,
    QuantumProgress,
    ChunkFrontier,
    QuantumIdentityRefused,
    quantum_layer_roster,
    quantum_retained_state,
    record_window_indices,
    resolve_quantum_windows,
    run_layer_quantum_core,
    verify_quantum_identity,
)
from prismaquant.joint_cost_stage_a import (
    resolve_stride,
    run_adjoint_capture_core,
)
from prismaquant.sensitivity_probe import SharedStateCotangents

from test_joint_operator_windows import policy as operator_policy
from test_layer_major_boundary_capture import fixture, draw
from test_streamed_cost_checkpoints import _model_identity

QUANTUM_SCHEMA = "prismaquant.joint_layer_quanta.v1"
HEX = "0123456789abcdef"


@pytest.fixture(autouse=True)
def _offline_tier_policy():
    """Each offline test runs outside campaign scope with no staged-tier
    policy: the entrypoints activate the process-global policy from sealed
    args when driven in-process, and without this reset a passing main()
    would poison every later offline bulk read in the session (PQ #845)."""
    from prismaquant.staged_tier_policy import deactivate_staged_tier_policy_for_tests
    deactivate_staged_tier_policy_for_tests()
    yield
    deactivate_staged_tier_policy_for_tests()


def _hex(char: str) -> str:
    return char * 64


def _boundary_policy(path, *, window=2):
    """The retained lane's tiny layer-major policy (the budget undercharges
    any larger resident cap, exactly as the production plan pins it)."""
    from test_streamed_boundary_artifacts import _policy

    return {**_policy(path, window=window, cap=(2 * window + 1) * 256),
            "schema": "prismaquant.aura.boundary_storage.v2",
            "capture_order": "layer_major"}


def _budget():
    from prismaquant.joint_retained_window_plan import RetainedWindowBudget

    return RetainedWindowBudget(
        50 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20,
        1 << 20, 1 << 20, 1 << 20, 1 << 20, 1024,
        2048, 4 << 20, 4)


def _retained_execution():
    from prismaquant.joint_retained_window_plan import EXECUTION_SCHEMA

    return {"schema": EXECUTION_SCHEMA, "budget": _budget().as_dict(),
            "source_reserve_bytes": 1 << 20,
            "source_loading_reserve_bytes": 2 << 20}


def _prepared_cache(model, context, runner, cache, tmp_path):
    """Disk-backed PWC with prepared render identities (the retained shape)."""
    files, proofs, file_shas = {}, {}, {}
    assets = tmp_path / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for index, (key, tensor) in enumerate(cache.weights.items()):
        path = assets / f"{index}.pt"
        if not path.exists():
            torch.save(tensor, path)
        files[key] = str(path)
        proofs[key] = _tensor_identity(tensor)
        file_shas[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    cache.weights = files
    cache.enable_lru(1 << 20)
    cache.metadata = {
        "source_model_identity": _model_identity("joint-source"),
        "calib_hash": "fixture-calibration",
        "verified_cells": {key: {"rendered_weight": value,
                                 "render_file_sha256": file_shas[key]}
                           for key, value in proofs.items()},
    }
    cache.require_file_load_sha256(file_shas, max_file_bytes=1 << 20)
    return cache, proofs


def _tensor_identity(tensor):
    from prismaquant.production_weight_cache import _cb_cache_tensor_identity

    return _cb_cache_tensor_identity(tensor)


FORMATS = ("FP8_E4M3", "NVFP4A16", "BF16")


def _execution(tmp_path, *, n_probes=4, seed_base=7000):
    return {
        "n_probes": n_probes,
        "seed_base": seed_base,
        "probe_microbatch": 1,
        "token_scope": "all",
        "temperature": 1.0,
        "production_act_scales": "0",
        "boundary_storage": _boundary_policy(tmp_path / "boundaries"),
        "operator_windows": operator_policy(),
        "retained_operator_windows": _retained_execution(),
        "min_free_gib": 0,
        "device_envelope_bytes": None,
    }


def _single_run(tmp_path, monkeypatch, *, checkpoint, layers=2):
    """The campaign of record: one consumer, retained windows, exact boundaries."""
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    model, context, runner, cache = fixture(layers)
    context.settle_prefetched_layers = lambda layers, *, retry_availability=False: None
    context.source_residency_snapshot = lambda layers, include_head=False: {
        "owners": [], "unique_storage_bytes": sum(
            p.numel() * p.element_size() for p in model.parameters())}
    cache, proofs = _prepared_cache(model, context, runner, cache, tmp_path / "shared")
    result = aura.compute_aura_cost_streamed(
        runner, draw(), list(FORMATS), n_probes=4, probe_microbatch=1,
        min_free_gib=0, production_cache=cache, joint_activation=True,
        prepared_render_identities=proofs,
        model_identity=_model_identity("joint-source"),
        operator_windows=operator_policy(),
        boundary_storage=_boundary_policy(tmp_path / "single-boundaries"),
        formats_by_qname={name: list(FORMATS) for name in
                          [f"model.layers.{i}.proj" for i in range(runner.num_layers)]},
        checkpoint_dir=checkpoint, resume=False,
        retained_operator_windows=_retained_execution(),
        include_routed_experts=True, dw_dtype="float32")
    return result, runner, cache, proofs


def _stage_a(tmp_path, monkeypatch, runner_seed=85, layers=2):
    torch.manual_seed(runner_seed)
    model, context, runner, cache = fixture(layers)
    context.settle_prefetch_layers = lambda layers: None
    return runner, cache


def _quantum_record(*, output_root, layer, checkpoint_boundary, chain, windows,
                    total_bytes, plan_sha, prepared_sha, adjoint_sha, slice_path=None):
    """A producer-shaped record: ``windows`` seals ordered indices only (D2).

    Pass ``windows`` as a list of ``{"window_index": i}`` dicts (what the
    producer seals); advisory ``names`` may be attached per window to
    exercise the runtime's cross-check, but the replay never trusts them.
    ``adjoint_sha`` is the stage-A slice digest the record binds (PQ #993);
    ``None`` seals an unbound (pre-A) record.
    """
    root = Path(output_root) / "layer-quanta" / f"layer-{layer:03d}"
    if slice_path is None:
        slice_path = Path(output_root) / "layer-quanta" / "adjoint-slices" / f"layer-{layer:03d}.json"
    binding = ({"receipt_sha256": None} if adjoint_sha is None else
               {"slice_sha256": adjoint_sha, "slice_path": str(slice_path)})
    record = {
        "schema": QUANTUM_SCHEMA,
        "quantum_id": f"layer-{layer:03d}",
        "layer": layer,
        "campaign": {
            "plan_path": "plan.json", "plan_sha256": plan_sha,
            "prepared_path": "prepared.json", "prepared_sha256": prepared_sha,
            "read_manifest_sha256": _hex("f"),
            "campaign_scope": {"fixture": True},
            "unit_roster_sha256": _hex("a"),
        },
        "read_set": {
            "manifest_path": "slice.json.gz", "manifest_sha256": _hex("b"),
            "entry_count": len(windows), "total_bytes": total_bytes,
            "source_phase": {"name": f"layer-{layer:03d}",
                             "start_bytes": 0, "end_bytes": total_bytes},
        },
        "chunks": [{"name": f"layer-{layer:03d}-chunk-000",
                    "start_bytes": 0, "end_bytes": total_bytes}],
        "windows": windows,
        "adjoint": {"checkpoint_boundary": checkpoint_boundary,
                    "chain_layers": chain,
                    "boundary_artifacts": "…/layer-quanta/adjoint",
                    **binding},
        "output_space": {
            "root": str(root),
            "cost_payload": str(root / "cost.pkl"),
            "results": str(root / "results.json"),
            "counters": str(root / "counters.json"),
            "checkpoint_dir": str(root / "checkpoints"),
        },
    }
    record["identity_sha256"] = canonical_json_sha256(record, where="record")
    return record


def _write(path, payload: bytes) -> tuple[Path, str]:
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


@pytest.fixture
def identity_files(tmp_path):
    """Plan, prepared and the layer-1 stage-A slice file a quantum reads."""
    from test_stage_b_band_binding import synthetic_receipt
    plan = _write(tmp_path / "plan.json", b"plan-fixture")
    prepared = _write(tmp_path / "prepared.json", b"prepared-fixture")
    receipt = synthetic_receipt(
        plan_sha256=plan[1], prepared_sha256=prepared[1], scope={"fixture": True},
        num_layers=2, stride=1)
    path = tmp_path / "adjoint-slice-layer-001.json"
    adjoint = (path, write_adjoint_slice(path, stage_a_slice(receipt, 1), layer=1))
    return {"plan": plan, "prepared": prepared, "adjoint": adjoint,
            "output_root": tmp_path / "campaign"}


def _valid_record(identity_files, *, layer=1, checkpoint_boundary=2, chain=(),
                  windows=None, total_bytes=100):
    windows = windows or [{"window_index": 0}]
    return _quantum_record(
        output_root=identity_files["output_root"], layer=layer,
        checkpoint_boundary=checkpoint_boundary, chain=chain, windows=windows,
        total_bytes=total_bytes, plan_sha=identity_files["plan"][1],
        prepared_sha=identity_files["prepared"][1],
        adjoint_sha=identity_files["adjoint"][1],
        slice_path=identity_files["adjoint"][0])


@pytest.mark.parametrize("tamper", [
    "quantum_sha", "plan_sha", "prepared_sha", "adjoint_sha",
    "record_identity", "output_root",
])
def test_identity_refusals_exit_3_nothing_written(identity_files, tamper, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    record = _valid_record(identity_files)
    record_path = identity_files["output_root"].parent / "record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    kwargs = dict(
        quantum_path=record_path,
        quantum_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
        plan_path=identity_files["plan"][0], plan_sha256=identity_files["plan"][1],
        prepared_path=identity_files["prepared"][0],
        prepared_sha256=identity_files["prepared"][1],
        adjoint_path=identity_files["adjoint"][0],
        adjoint_sha256=identity_files["adjoint"][1],
        output_root=identity_files["output_root"])
    if tamper == "quantum_sha":
        kwargs["quantum_sha256"] = _hex("0")
    elif tamper == "plan_sha":
        kwargs["plan_sha256"] = _hex("0")
    elif tamper == "prepared_sha":
        kwargs["prepared_sha256"] = _hex("0")
    elif tamper == "adjoint_sha":
        kwargs["adjoint_sha256"] = _hex("0")
    elif tamper == "record_identity":
        record["identity_sha256"] = _hex("9")
        record_path.write_text(json.dumps(record))
        kwargs["quantum_path"] = record_path
        kwargs["quantum_sha256"] = hashlib.sha256(
            record_path.read_bytes()).hexdigest()
    elif tamper == "output_root":
        kwargs["output_root"] = identity_files["output_root"].parent / "elsewhere"
    from prismaquant.joint_cost_quantum import QuantumIdentityRefused

    with pytest.raises(QuantumIdentityRefused) as info:
        verify_quantum_identity(**kwargs)
    assert info.value.args
    assert not (identity_files["output_root"]).exists()



def test_dev_mode_runs_a_record_under_a_re_declared_plan(identity_files, monkeypatch, capsys):
    """PQ #1147: a plan re-declared after the record was sealed stamps by default.

    A re-declared ``workspace_reserve_bytes`` moves the plan digest the record
    and the stage-A header bind. The supplied plan's own bytes are still
    checked against the supplied digest.
    """
    from prismaquant.joint_cost_quantum import QuantumIdentityRefused

    record = _valid_record(identity_files)
    record_path = identity_files["output_root"].parent / "record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    plan_path, plan_sha = _write(identity_files["plan"][0].parent / "plan-measured.json",
                                 b"plan-fixture, workspace_reserve_bytes measured")
    kwargs = dict(
        quantum_path=record_path,
        quantum_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
        plan_path=plan_path, plan_sha256=plan_sha,
        prepared_path=identity_files["prepared"][0],
        prepared_sha256=identity_files["prepared"][1],
        adjoint_path=identity_files["adjoint"][0],
        adjoint_sha256=identity_files["adjoint"][1],
        output_root=identity_files["output_root"])
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    capsys.readouterr()
    verified, _slice = verify_quantum_identity(**kwargs)
    assert verified == json.loads(json.dumps(record))
    assert "[DEV-MODE] seal quantum record plan differs" in capsys.readouterr().out
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(QuantumIdentityRefused, match="binds another plan"):
        verify_quantum_identity(**kwargs)
    # The supplied file's bytes stay integrity in both modes.
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    with pytest.raises(QuantumIdentityRefused, match="plan digest mismatch"):
        verify_quantum_identity(**{**kwargs, "plan_sha256": identity_files["plan"][1]})

def test_cli_identity_refusal_is_exit_3(identity_files, monkeypatch, capsys):
    from prismaquant.joint_cost_quantum import main

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    record = _valid_record(identity_files)
    record_path = identity_files["output_root"].parent / "record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    code = main([
        "--quantum", str(record_path),
        "--quantum-sha256", _hex("0"),
        "--plan", str(identity_files["plan"][0]),
        "--plan-sha256", identity_files["plan"][1],
        "--prepared", str(identity_files["prepared"][0]),
        "--prepared-sha256", identity_files["prepared"][1],
        "--adjoint-slice", str(identity_files["adjoint"][0]),
        "--adjoint-slice-sha256", identity_files["adjoint"][1],
        "--output-root", str(identity_files["output_root"]),
    ])
    assert code == EXIT_IDENTITY_REFUSED
    assert IDENTITY_REFUSED_MARKER in capsys.readouterr().out
    assert not identity_files["output_root"].exists()


def test_stage_a_threads_the_plan_derivative_and_prefetch(tmp_path, monkeypatch):
    """The model build receives the plan's derivative binding and prefetch budget.

    Stage A v8 (d4578e5e6af4) refused at 363 s: the corrected GLM runtime on
    disk requires an explicit ``source_derivative`` binding, the plan's
    execution block carries one, and the stage-A caller passed neither it nor
    the plan's ``source_prefetch`` budget (the single-run threads both). This
    pins the seam: whatever the plan carries reaches the model build, and a
    plan without them reaches it as absent.
    """
    import prismaquant.joint_cost_stage_a as stage_a
    import prismaquant.tessera_joint_aura as aura
    import prismaquant.cost_streaming as streaming
    import prismaquant.tessera_reader as reader_mod
    import prismaquant.gpu_guard as guard
    import prismaquant.joint_projection_backend as backend
    import prismaquant.aura_cost as aura_cost
    import prismaquant.residency_map as residency

    class _Done(Exception):
        pass

    captured = {}

    def fake_build(model, **kwargs):
        captured.update(kwargs)
        raise _Done

    monkeypatch.setattr(guard, "require_cuda_hot_path", lambda *a, **k: None)
    monkeypatch.setattr(backend, "prewarm_projection_backend",
                        lambda *a, **k: type("B", (), {"identity": None})())
    monkeypatch.setattr(reader_mod, "load_declared_reader", lambda *a, **k: None)
    monkeypatch.setattr(aura_cost, "_aura_source_sha256", lambda: _hex("b"))
    # Stage A takes its head from the completion the preflight checked
    # (PQ #1051); the stub returns it as the real preflight does.
    monkeypatch.setattr(aura, "_preflight_run_prepared",
                        lambda prepared, **k: json.loads(Path(prepared["path"]).read_text()))
    draw = {"fit_ids_sha256": _hex("1"), "text_sha256": _hex("2"),
            "nsamples": 4, "seqlen": 512, "seed": 7}
    import prismaquant.calibration_data as calib
    monkeypatch.setattr(calib, "load_calibration_input",
                        lambda *a, **k: ([], {"provenance": dict(draw)}))
    monkeypatch.setattr(aura, "_seed_source_identity_cache",
                        lambda *a, **k: None)
    import prismaquant.model_profiles as profiles
    monkeypatch.setattr(profiles, "detect_profile", lambda *a, **k: None)
    monkeypatch.setattr(streaming, "build_streamed_causal_lm", fake_build)
    monkeypatch.setattr(residency, "bind_residency_manifest", lambda *a, **k: None)
    monkeypatch.setattr(stage_a, "GpuPowerSampler",
                        lambda: type("S", (), {"start": lambda s: s,
                                               "stop": lambda s: {}})())
    monkeypatch.setattr(stage_a, "KernelTimeProfiler",
                        lambda **_scope: type("K", (), {
                            "__enter__": lambda s: s,
                            "__exit__": lambda s, *a: None,
                            "kernel_active_s": 0.0,
                            "block": lambda s: {"kernel_active_s": 0.0},
                            "error": None})())
    monkeypatch.setattr(residency, "residency_report", lambda: None)

    prepared_path = tmp_path / "prepared.json"
    derivative = {"schema": "prismaquant.glm_source_derivative.v1"}
    prefetch = {"max_cache_slots": 4, "prefetch_workers": 3,
                "prefetch_lookahead": 2, "cache_headroom_gb": 8,
                "prefetch_min_available_gb": 16,
                "require_prefetched_residency": True}
    prepared_path.write_text(json.dumps(
        {"plan_sha256": _hex("d"), "source_model_identity": None,
         "calibration_input": {"provenance": draw},
         "formats_by_qname": {"model.layers.0.mlp.down_proj": ["NVFP4"]},
         "measured_cells": 1}))
    prepared = {"path": str(prepared_path),
                "sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest()}
    config = {"execution": {"production_act_scales": "scales",
                            "source_derivative": derivative,
                            "n_calib_samples": 4, "calib_seqlen": 512,
                            "boundary_storage": _boundary_policy(
                                tmp_path / "deriv-boundaries")},
              "inputs": {}, "output_root": str(tmp_path),
              "model": "/models/x",
              "calibration_input": {"path": str(tmp_path / "cal.json"),
                                    "sha256": _hex("3")},
              "source_prefetch": prefetch}

    with pytest.raises(_Done):
        stage_a.run_adjoint_capture(
            config, plan_sha256=_hex("d"), prepared=prepared,
            output_root=str(tmp_path), stride=2)
    assert captured["source_derivative"] == derivative
    for key, value in prefetch.items():
        assert captured[key] == value

    # A plan without source_prefetch is refused fail-closed (the block is
    # mandatory -- the same contract the single-run's _source_prefetch
    # enforces), and a plan without source_derivative threads absence.
    config["execution"].pop("source_derivative")
    config.pop("source_prefetch")
    with pytest.raises(ValueError, match="complete source_prefetch"):
        stage_a.run_adjoint_capture(
            config, plan_sha256=_hex("d"), prepared=prepared,
            output_root=str(tmp_path), stride=2)
    config["source_prefetch"] = prefetch
    captured.clear()
    with pytest.raises(_Done):
        stage_a.run_adjoint_capture(
            config, plan_sha256=_hex("d"), prepared=prepared,
            output_root=str(tmp_path), stride=2)
    assert captured.get("source_derivative") is None


def _prefetch_budget(**overrides):
    budget = {"max_cache_slots": 2, "prefetch_workers": 1,
              "prefetch_lookahead": 1, "cache_headroom_gb": 8,
              "prefetch_min_available_gb": 16,
              "require_prefetched_residency": True}
    budget.update(overrides)
    return budget


def _override_document(tmp_path, budget, *, reason="v10 IO widening (#819)",
                       schema=None, name="prefetch-override.json"):
    from prismaquant.joint_cost_stage_a import PREFETCH_OVERRIDE_INPUT_SCHEMA

    document = {"schema": schema or PREFETCH_OVERRIDE_INPUT_SCHEMA,
                "reason": reason, "source_prefetch": budget}
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


def _stage_a_run_stub(tmp_path, monkeypatch, out_root):
    """The full fake stage-A environment: every loader/build seam stubbed so
    ``run_adjoint_capture`` runs to its provenance writes (results.json,
    counters.json) on CPU. Returns the model-build kwargs capture."""
    import prismaquant.joint_cost_stage_a as stage_a
    import prismaquant.tessera_joint_aura as aura
    import prismaquant.cost_streaming as streaming
    import prismaquant.tessera_reader as reader_mod
    import prismaquant.gpu_guard as guard
    import prismaquant.joint_projection_backend as backend
    import prismaquant.aura_cost as aura_cost
    import prismaquant.residency_map as residency
    import prismaquant.glm_capture_compatibility as compatibility

    # This is an OFFLINE stage-A environment, and the launch context is the
    # one seam a stub cannot fake: a capture that carries
    # PRISMABUILD_ACTION_KEY is an admitted produced-output owner and now
    # binds one unconditionally, refusing if it cannot. Inside pbtest this
    # process IS admitted, and inherits a key with no residency map, so the
    # stub would be asked to bind a queue that does not exist. Clearing the
    # context is what makes this run offline; it does not relax the
    # production check, which still refuses a half-present launch context.
    for name in ("PRISMABUILD_ACTION_KEY", "PRISMABUILD_ACTION_NONCE",
                 "PRISMABUILD_ACTION_SCOPE", "PRISMABUILD_RESIDENCY_MAP"):
        monkeypatch.delenv(name, raising=False)

    captured = {}

    class _Runner:
        num_layers = 2
        device = "cpu"
        model = object()

        def shutdown(self):
            pass

    class _Ids:
        shape = (4, 512)

        def to(self, device):
            return self

    monkeypatch.setattr(guard, "require_cuda_hot_path", lambda *a, **k: None)
    monkeypatch.setattr(backend, "prewarm_projection_backend",
                        lambda *a, **k: type("B", (), {"identity": None})())
    monkeypatch.setattr(backend, "executing_image", lambda: None)
    monkeypatch.setattr(reader_mod, "load_declared_reader", lambda *a, **k: None)
    monkeypatch.setattr(aura_cost, "_aura_source_sha256", lambda: _hex("b"))
    # Stage A takes its head from the completion the preflight checked
    # (PQ #1051); the stub returns it as the real preflight does.
    monkeypatch.setattr(aura, "_preflight_run_prepared",
                        lambda prepared, **k: json.loads(Path(prepared["path"]).read_text()))
    monkeypatch.setattr(compatibility, "require_capture_compatibility",
                        lambda *a, **k: None)
    draw = {"fit_ids_sha256": _hex("1"), "text_sha256": _hex("2"),
            "nsamples": 4, "seqlen": 512, "seed": 7}
    import prismaquant.calibration_data as calib
    monkeypatch.setattr(calib, "load_calibration_input",
                        lambda *a, **k: (_Ids(), {"provenance": dict(draw)}))
    monkeypatch.setattr(aura, "_seed_source_identity_cache",
                        lambda *a, **k: None)
    import prismaquant.model_profiles as profiles
    monkeypatch.setattr(profiles, "detect_profile", lambda *a, **k: None)
    monkeypatch.setattr(streaming, "build_streamed_causal_lm",
                        lambda model, **kwargs: (
                            captured.update(kwargs) or _Runner()))
    monkeypatch.setattr(streaming, "build_streamed_model_identity",
                        lambda *a, **k: {"identity": "stub"})
    monkeypatch.setattr(residency, "bind_residency_manifest", lambda *a, **k: None)
    monkeypatch.setattr(stage_a, "GpuPowerSampler",
                        lambda: type("S", (), {"start": lambda s: s,
                                               "stop": lambda s: {}})())
    monkeypatch.setattr(stage_a, "KernelTimeProfiler",
                        lambda **_scope: type("K", (), {
                            "__enter__": lambda s: s,
                            "__exit__": lambda s, *a: None,
                            "kernel_active_s": 0.0,
                            "block": lambda s: {"kernel_active_s": 0.0},
                            "error": None})())
    monkeypatch.setattr(residency, "residency_report", lambda: None)
    monkeypatch.setattr(stage_a, "run_adjoint_capture_core",
                        lambda *a, **k: {
                            "stride": {"value": 2, "source": None,
                                       "boundaries": [], "max_chain_layers": 1},
                            "checkpoints": []})
    # The #882 durable-budget seam preflights from live geometry before the
    # (mocked) core; these fakes carry no model geometry, so the preflight
    # is stubbed here -- the real preflight is covered by
    # tests/test_stage_a_artifact_budget.py on the actual writer path.
    monkeypatch.setattr(stage_a, "_run_artifact_preflight",
                        lambda *a, **k: {
                            "declared_bytes": 1,
                            "required_floor_bytes": 1,
                            "planning_estimate_bytes": 1,
                            "demand": {}})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)

    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps(
        {"plan_sha256": _hex("d"), "source_model_identity": {"identity": "stub"},
         "calibration_input": {"provenance": draw},
         "formats_by_qname": {"model.layers.0.mlp.down_proj": ["NVFP4"]},
         "measured_cells": 1}))
    prepared = {"path": str(prepared_path),
                "sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest()}
    config = {"execution": {"production_act_scales": "0",
                            "n_calib_samples": 4, "calib_seqlen": 512,
                            "n_probes": 1, "probe_microbatch": 0,
                            "boundary_storage": _boundary_policy(
                                tmp_path / "stub-boundaries")},
              "inputs": {}, "output_root": str(out_root),
              "model": "/models/x",
              "canonical_capture": None,
              "calibration_input": {"path": str(tmp_path / "cal.json"),
                                    "sha256": _hex("3")},
              "max_gpu_bytes": 1 << 50}

    def run(**kwargs):
        captured.clear()
        result = stage_a.run_adjoint_capture(
            dict(config, source_prefetch=kwargs.pop("plan_budget")),
            plan_sha256=_hex("d"), prepared=prepared,
            output_root=str(out_root), stride=2, **kwargs)
        space = out_root / "layer-quanta" / "adjoint"
        return result, json.loads((space / "results.json").read_text()), \
            json.loads((space / "counters.json").read_text())

    return captured, run


def test_stage_a_prefetch_override_replaces_plan_budget_and_stamps(
        tmp_path, monkeypatch):
    """The #819 seam: an explicit override replaces the plan's sealed budget
    for the run only, and the deviation is stamped -- plan sealed X, run
    used Y, reason -- into results.json and counters.json, never silently.
    Without an override the plan's block threads verbatim and the stamp is
    null."""
    plan_budget = _prefetch_budget()  # the frozen v9 pin: 1 worker, 2 slots
    override_budget = _prefetch_budget(max_cache_slots=7, prefetch_workers=8,
                                       prefetch_lookahead=4)
    assert override_budget != plan_budget
    override_path = _override_document(tmp_path, override_budget)

    captured, run = _stage_a_run_stub(
        tmp_path, monkeypatch, tmp_path / "out-override")
    result, results_json, counters_json = run(
        plan_budget=plan_budget, prefetch_override=override_path)

    # The model build threads the override's budget, not the plan's.
    assert {key: captured[key] for key in override_budget} == override_budget
    stamp = result["prefetch_override"]
    assert stamp["schema"] == "prismaquant.joint_adjoint_capture.prefetch_override.v1"
    assert stamp["plan_sealed"] == plan_budget
    assert stamp["run_used"] == override_budget
    assert stamp["reason"] == "v10 IO widening (#819)"
    assert stamp["source"] == "cli"
    assert stamp["path"] == str(override_path)
    assert stamp["sha256"] == hashlib.sha256(override_path.read_bytes()).hexdigest()
    # Both provenance files carry the same honest block, verbatim.
    assert results_json["prefetch_override"] == stamp
    assert counters_json["prefetch_override"] == stamp

    captured, run = _stage_a_run_stub(
        tmp_path, monkeypatch, tmp_path / "out-plain")
    result, results_json, counters_json = run(plan_budget=plan_budget)
    assert {key: captured[key] for key in plan_budget} == plan_budget
    assert result["prefetch_override"] is None
    assert results_json["prefetch_override"] is None
    assert counters_json["prefetch_override"] is None


def test_prefetch_override_document_grammar_refuses(tmp_path):
    """The override passes the plan's own field grammar: the same six
    fields, the same rules, residency still required, plus a reason. A
    document that fails any of it refuses before anything runs."""
    from prismaquant.joint_cost_stage_a import (
        PREFETCH_OVERRIDE_ENV, resolve_prefetch_override)

    config = {"source_prefetch": _prefetch_budget()}
    path = _override_document(tmp_path, _prefetch_budget(
        max_cache_slots=7, prefetch_workers=8, prefetch_lookahead=4))
    resolved = resolve_prefetch_override(
        config, None, environ={PREFETCH_OVERRIDE_ENV: str(path)})
    assert resolved["override"]["source"] == "env"
    assert resolved["run_used"]["prefetch_workers"] == 8

    defects = {
        "missing field": _override_document(
            tmp_path / "d0", {k: v for k, v in _prefetch_budget().items()
                              if k != "max_cache_slots"}),
        "lookahead beyond slots": _override_document(
            tmp_path / "d1", _prefetch_budget(max_cache_slots=2,
                                              prefetch_lookahead=2)),
        "residency not required": _override_document(
            tmp_path / "d2", _prefetch_budget(require_prefetched_residency=False)),
        "no reason": _override_document(tmp_path / "d3", _prefetch_budget(),
                                        reason="  "),
        "wrong schema": _override_document(
            tmp_path / "d4", _prefetch_budget(), schema="prismaquant.other.v1"),
    }
    for label, document in defects.items():
        # Either grammar's own refusal message: the document-level checks
        # name the override file; the budget-level ones are _source_prefetch's
        # unchanged words.
        with pytest.raises(ValueError, match="prefetch override|source_prefetch"):
            resolve_prefetch_override(config, document, environ={})
    unwritable = tmp_path / "absent.json"
    with pytest.raises(ValueError, match="unreadable JSON"):
        resolve_prefetch_override(config, unwritable, environ={})


def test_prefetch_override_cli_and_env_disagreement_refuses(tmp_path):
    """Two explicit sources that disagree refuse (#809); one source alone
    (either channel) resolves, and neither source is the plan verbatim."""
    from prismaquant.joint_cost_stage_a import (
        PREFETCH_OVERRIDE_ENV, AdjointIdentityRefused, resolve_prefetch_override)

    config = {"source_prefetch": _prefetch_budget()}
    first = _override_document(tmp_path, _prefetch_budget(prefetch_workers=8))
    second = _override_document(
        tmp_path / "alt", _prefetch_budget(prefetch_workers=4))
    with pytest.raises(AdjointIdentityRefused, match="disagrees"):
        resolve_prefetch_override(
            config, first, environ={PREFETCH_OVERRIDE_ENV: str(second)})
    same = resolve_prefetch_override(
        config, first, environ={PREFETCH_OVERRIDE_ENV: str(first)})
    assert same["override"]["source"] == "cli"


def test_stage_a_cli_threads_the_override_flag(tmp_path, monkeypatch):
    """main() hands --prefetch-override to the capture call. Dropping the
    kwarg would revert the run to the plan's budget with no stamp -- the
    silent deviation this seam exists to make impossible."""
    import prismaquant.joint_cost_stage_a as stage_a
    import prismaquant.tessera_joint_aura as aura

    captured = {}

    def fake_capture(config, **kwargs):
        captured.update(kwargs)
        return {"command": "adjoint-capture", "passed": True,
                "stride": {"value": 8, "source": "plan"}, "checkpoints": []}

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    monkeypatch.setattr(stage_a, "require_dev_mode", lambda *a, **k: None)
    monkeypatch.setattr(aura, "_load_plan", lambda *a, **k: {})
    monkeypatch.setattr(stage_a, "run_adjoint_capture", fake_capture)

    override = _override_document(tmp_path, _prefetch_budget(prefetch_workers=8))
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    code = stage_a.main([
        "--plan", str(plan), "--plan-sha256", _hex("a"),
        "--prepared", str(tmp_path / "prepared.json"),
        "--prepared-sha256", _hex("b"),
        "--output-root", str(tmp_path),
        "--prefetch-override", str(override)])
    assert code == stage_a.EXIT_OK
    assert captured["prefetch_override"] == override


def test_stage_a_threads_the_plan_historical_encoder_reuse(tmp_path, monkeypatch):
    """The plan's allowlist must reach the anchor loader (the v6 lesson).

    Stage A v6 (89d3a0eb012a) refused at 85 s: the anchor checkpoint's
    recorded encoder source was historically allowlisted in the plan, but the
    stage-A caller never passed ``historical_encoder_reuse`` through, so the
    loader saw no policy and refused a reuse the plan explicitly names. The
    single-run path threads it (tessera_joint_aura.py:2777); stage A must
    too. This pins the seam: whatever the plan carries reaches the loader's
    kwarg verbatim, and a plan without the block reaches it as None. Since
    PQ #1051 Stage A walks only in its verification arm (``head_walk``), so
    that arm is the caller pinned here.
    """
    import prismaquant.joint_cost_stage_a as stage_a
    import prismaquant.tessera_joint_aura as aura
    import prismaquant.tessera_reader as reader_mod
    import prismaquant.gpu_guard as guard
    import prismaquant.joint_projection_backend as backend
    import prismaquant.aura_cost as aura_cost
    import prismaquant.residency_map as residency

    class _Done(Exception):
        pass

    captured = {}

    def fake_loader(inputs, **kwargs):
        captured.update(kwargs)
        raise _Done

    monkeypatch.setattr(guard, "require_cuda_hot_path", lambda *a, **k: None)
    monkeypatch.setattr(backend, "prewarm_projection_backend",
                        lambda *a, **k: type("B", (), {"identity": None})())
    monkeypatch.setattr(reader_mod, "load_declared_reader", lambda *a, **k: None)
    monkeypatch.setattr(aura_cost, "_aura_source_sha256", lambda: _hex("b"))
    monkeypatch.setattr(aura, "_preflight_run_prepared",
                        lambda *a, **k: {"plan_sha256": _hex("d")})
    monkeypatch.setattr(aura, "load_measured_anchor_input", fake_loader)
    monkeypatch.setattr(residency, "bind_residency_manifest", lambda *a, **k: None)
    monkeypatch.setattr(stage_a, "GpuPowerSampler",
                        lambda: type("S", (), {"start": lambda s: s,
                                               "stop": lambda s: {}})())
    monkeypatch.setattr(stage_a, "KernelTimeProfiler",
                        lambda **_scope: type("K", (), {
                            "__enter__": lambda s: s,
                            "__exit__": lambda s, *a: None,
                            "kernel_active_s": 0.0,
                            "block": lambda s: {"kernel_active_s": 0.0},
                            "error": None})())
    monkeypatch.setattr(residency, "residency_report", lambda: None)

    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("{}")
    prepared = {"path": str(prepared_path),
                "sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest()}
    entry = {"encoder_source_sha256": _hex("1"), "reason": "pin test",
             "evidence": "/pin/test.md", "recorded_by": "test",
             "recorded_unix": 1789625518.0}
    reuse_block = {"schema": aura.HISTORICAL_ENCODER_REUSE_SCHEMA,
                   "allowlist": [entry]}
    # Every sealed plan carries a source_prefetch block (mandatory since the
    # #816 seam; _load_plan refuses without one), and the capture validates
    # it up front now that the #819 override resolves beside it -- so the
    # fixture config carries one like any real plan.
    config = {"execution": {"production_act_scales": "scales",
                            "boundary_storage": _boundary_policy(
                                tmp_path / "reuse-boundaries")},
              "inputs": {}, "output_root": str(tmp_path),
              "source_prefetch": _prefetch_budget(),
              "historical_encoder_reuse": reuse_block}

    with pytest.raises(_Done):
        stage_a.run_adjoint_capture(
            config, plan_sha256=_hex("d"), prepared=prepared,
            output_root=str(tmp_path), stride=2, head_walk=True)
    assert captured["historical_encoder_reuse"] == reuse_block

    captured.clear()
    config.pop("historical_encoder_reuse")
    with pytest.raises(_Done):
        stage_a.run_adjoint_capture(
            config, plan_sha256=_hex("d"), prepared=prepared,
            output_root=str(tmp_path), stride=2, head_walk=True)
    assert captured["historical_encoder_reuse"] is None


def test_stride_derivation_is_pinned():
    # The producer's set (PR #785, verified numbers): multiples of S below
    # the tail plus the tail itself; stage A must publish exactly this set
    # or bind_adjoint_receipt refuses. Order here is tail-first (stage A
    # serializes the tail checkpoint first); the binding compares sorted sets.
    assert derive_checkpoint_boundaries(45, 8) == (45, 40, 32, 24, 16, 8)
    assert sorted(derive_checkpoint_boundaries(45, 8)) == [8, 16, 24, 32, 40, 45]
    assert len(derive_checkpoint_boundaries(45, 8)) == 6
    assert max(len(chain_layers_for(b, b - 8))
               for b in derive_checkpoint_boundaries(45, 8)[1:]) == 7
    assert chain_layers_for(16, 13) == (15, 14)
    assert chain_layers_for(45, 44) == ()
    try:
        from prismaquant.joint_layer_quanta import derive_stride
    except ImportError:
        pass
    else:  # the producer's derivation, when it has landed, seals this set
        assert sorted(derive_stride(45, 8)["checkpoints"]) == sorted(
            derive_checkpoint_boundaries(45, 8))


def test_stride_resolution_refuses_disagreeing_cli():
    config = {"distributed_campaign": {"cotangent_checkpoint_stride": 4}}
    assert resolve_stride(config, None) == (4, "plan")
    assert resolve_stride({}, 3) == (3, "cli")
    assert resolve_stride({}, None) == (8, "cli")
    from prismaquant.joint_cost_stage_a import AdjointIdentityRefused

    with pytest.raises(AdjointIdentityRefused):
        resolve_stride(config, 8)


def _preflight_windows(runner, cache, proofs):
    from prismaquant.joint_statistics_replay import preflight_joint_operator_admission

    linears = {f"model.layers.{i}.proj": runner.layers[i].proj
               for i in range(runner.num_layers)}
    render_formats = {name: ["FP8_E4M3", "NVFP4A16"] for name in linears}
    return preflight_joint_operator_admission(
        {i: [f"model.layers.{i}.proj"] for i in range(runner.num_layers)},
        linears, render_formats, cache, policy=operator_policy(),
        retained_budget=_budget(), source_bytes=1 << 20)


def _windows_records(windows):
    return [{"window_index": index,
             "names": list(window.original_full_target_names),
             "statistics_bytes": window.statistics_bytes,
             "render_file_upper_bound_bytes": window.render_file_upper_bound_bytes,
             "candidate_count": window.candidate_count}
            for index, window in enumerate(windows)]


def _run_quantum(tmp_path, monkeypatch, *, single, layer, receipt, output_root,
                 progress_env=None, plan_sha="p", prepared_sha="r", adjoint_slice=None,
                 adjoint_handoff=None, handoff_emitter=None):
    """One layer quantum on its stage-A slice (the receipt's, unless given).

    ``adjoint_handoff``/``handoff_emitter`` run it band-serial (PQ #996); an
    emitter is a callable ``(record, adjoint_slice, execution) -> emitter``
    because the record is built here.
    """
    payload_single, runner_single, cache, proofs = single
    if adjoint_slice is None:
        # The slice a quantum reads is sliced from the sealed JSON receipt.
        adjoint_slice = stage_a_slice(json.loads(json.dumps(receipt)), layer)
    del payload_single
    layers = runner_single.num_layers
    torch.manual_seed(85)
    model, context, runner, _ = fixture(layers)
    context.settle_prefetched_layers = lambda layers, *, retry_availability=False: None
    context.source_residency_snapshot = lambda layers, include_head=False: {
        "owners": [], "unique_storage_bytes": sum(
            p.numel() * p.element_size() for p in model.parameters())}
    fresh = {}
    for key, path in cache.weights.items():
        fresh[key] = torch.load(path)
    from prismaquant.production_weight_cache import ProductionWeightCache

    # The same prepared cache scales the single run read: activation
    # identities (and their sha) are inputs both sides bind, so the quantum
    # reuses them instead of inventing its own.
    fresh_cache = ProductionWeightCache(
        weights=fresh, levers={},
        activation_max_abs=dict(cache.activation_max_abs or {}))
    fresh_cache, _ = _prepared_cache(model, context, runner, fresh_cache,
                                     tmp_path / "shared")

    windows_by_layer = _preflight_windows(runner, fresh_cache, None)
    windows = _windows_records(windows_by_layer[layer])
    total = sum(w["render_file_upper_bound_bytes"] for w in windows)
    record = _quantum_record(
        output_root=output_root, layer=layer,
        checkpoint_boundary=adjoint_slice["checkpoint"]["boundary"],
        chain=[],
        # Producer-shaped: indices only (D2). Membership and footprints are
        # recomputed below, never read from the record.
        windows=[{"window_index": index} for index in range(len(windows))],
        total_bytes=total, plan_sha=plan_sha,
        prepared_sha=prepared_sha, adjoint_sha=adjoint_slice_sha256(adjoint_slice))
    # chain_layers must match the record's checkpoint boundary exactly
    boundary = record["adjoint"]["checkpoint_boundary"]
    record["adjoint"]["chain_layers"] = list(chain_layers_for(boundary, layer))
    record["identity_sha256"] = canonical_json_sha256(
        {k: v for k, v in record.items() if k != "identity_sha256"},
        where="record")

    execution = _execution(tmp_path)
    retained = quantum_retained_state(execution)
    roster = quantum_layer_roster(
        runner, {name: list(FORMATS) for name in
                 [f"model.layers.{i}.proj" for i in range(runner.num_layers)]},
        layer)
    resolved = resolve_quantum_windows(
        record, layer=layer, names=roster.names, linears=roster.linears,
        render_formats=roster.render_formats, production_cache=fresh_cache,
        operator_windows=retained.operator_windows,
        retained_budget=retained.retained_budget,
        source_bytes=retained.source_bytes)
    # The handshake recomputes what the fixture preflight admitted.
    assert [w["names"] for w in resolved] == [w["names"] for w in windows]
    assert [w["render_file_upper_bound_bytes"] for w in resolved] == [
        w["render_file_upper_bound_bytes"] for w in windows]

    frontier = ChunkFrontier(chunks=record["chunks"], windows=resolved)
    counters = QuantumCounters(quantum_id=record["quantum_id"],
                               identity_sha256=record["identity_sha256"],
                               chunks=record["chunks"], frontier=frontier)
    progress = QuantumProgress(frontier=frontier, base_units=0)
    if callable(adjoint_handoff):
        adjoint_handoff = adjoint_handoff(record, adjoint_slice)
    if handoff_emitter is not None:
        handoff_emitter = handoff_emitter(record, adjoint_slice, execution)
    payload = run_layer_quantum_core(
        runner, fresh_cache, draw(),
        {name: list(FORMATS) for name in
         [f"model.layers.{i}.proj" for i in range(runner.num_layers)]},
        record=record, adjoint_slice=adjoint_slice, execution=execution,
        output_root=output_root, projection_backend=None, resume=False,
        resolved_windows=resolved,
        counters=counters, progress=progress,
        adjoint_handoff=adjoint_handoff, handoff_emitter=handoff_emitter)
    resolved_names = [name for window in resolved for name in window["names"]]
    return payload, record, counters.finish(
        units_done=len(payload["costs"]), units_total=len(resolved_names))


def test_quantum_matches_single_run_bitwise(tmp_path, monkeypatch):
    """The §9.3 gate's dry run: stage A + one quantum vs the single consumer.

    The single run commits each unit's envelope to its journal; the quantum
    chains from stage A's strided checkpoint and replays its layer's sealed
    windows. The unit envelopes must hash identically -- bitwise, no
    tolerance -- because both sides run the same kernels in the same order
    from the same digest-checked bytes.
    """
    single_root = tmp_path / "single"
    single = _single_run(single_root, monkeypatch,
                         checkpoint=single_root / "checkpoints")
    payload_single, runner_single, _cache, _proofs = single

    output_root = tmp_path / "campaign"
    runner_a, _ = _stage_a(tmp_path, monkeypatch)
    context_a = runner_a.context
    context_a.settle_prefetch_layers = lambda layers: None
    receipt = run_adjoint_capture_core(
        runner_a, draw(), execution=_execution(tmp_path),
        output_root=output_root, stride=2,
        source_model_identity=_model_identity("joint-source"),
        unit_roster_sha256=_hex("a"), plan_sha256=_hex("d"),
        prepared_sha256=_hex("e"), read_manifest_sha256=_hex("f"),
        implementation_sha256=aura._aura_source_sha256())
    # The tail checkpoint exists, boundary entries all survived, receipt not
    # yet sealed on disk by the core (the CLI seals it).
    assert [c["boundary"] for c in receipt["checkpoints"]] == [2]
    assert set(receipt["boundary_entries"]) == {"0", "1"}

    from tools.compare_joint_layer_gate import compare_layer

    for layer in (1, 0):
        payload, record, _counters = _run_quantum(
            tmp_path, monkeypatch, single=single, layer=layer, receipt=receipt,
            output_root=output_root, plan_sha=_hex("d"), prepared_sha=_hex("e"))
        for name, rows in payload["costs"].items():
            for fmt, row in rows.items():
                single_row = payload_single["costs"][name][fmt]
                assert row["signed_components_per_probe"] == \
                    single_row["signed_components_per_probe"], (layer, name, fmt)
                assert row["x2_per_probe"] == single_row["x2_per_probe"]
        # The journals hold identical measurement envelopes for this layer.
        verdict = compare_layer(single_root / "checkpoints",
                                Path(record["output_space"]["checkpoint_dir"]),
                                layer=layer, qname_filter=None)
        assert verdict["verdict"] == "match", verdict
        assert verdict["matched"] == verdict["units_shared"] == 1


def test_the_core_refuses_a_live_bf16_flag_the_slice_does_not_record(
        tmp_path, monkeypatch):
    """PQ #1065: the core compares the live flag with the verified slice.

    Stage A ran at PyTorch's default, so its slice carries no bf16 stamp; a
    quantum core running with the flag off would rebuild another chain. It
    refuses before the checkpoint is read, naming both settings.
    """
    from prismaquant.joint_cost_quantum import QuantumIdentityRefused
    from prismaquant.matmul_arithmetic import BF16_REDUCTION_ENV, BF16_REDUCTION_FIELD

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    single_root = tmp_path / "single"
    single = _single_run(single_root, monkeypatch,
                         checkpoint=single_root / "checkpoints")
    output_root = tmp_path / "campaign"
    runner_a, _ = _stage_a(tmp_path, monkeypatch)
    runner_a.context.settle_prefetch_layers = lambda layers: None
    matmul = torch.backends.cuda.matmul
    saved = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_bf16_reduced_precision_reduction = True
    try:
        receipt = run_adjoint_capture_core(
            runner_a, draw(), execution=_execution(tmp_path),
            output_root=output_root, stride=2,
            source_model_identity=_model_identity("joint-source"),
            unit_roster_sha256=_hex("a"), plan_sha256=_hex("d"),
            prepared_sha256=_hex("e"), read_manifest_sha256=_hex("f"),
            implementation_sha256=aura._aura_source_sha256())
        assert BF16_REDUCTION_FIELD not in receipt["run_identity"]
        matmul.allow_bf16_reduced_precision_reduction = False
        with pytest.raises(QuantumIdentityRefused) as refused:
            _run_quantum(tmp_path, monkeypatch, single=single, layer=1,
                         receipt=receipt, output_root=output_root,
                         plan_sha=_hex("d"), prepared_sha=_hex("e"))
    finally:
        matmul.allow_bf16_reduced_precision_reduction = saved
    message = str(refused.value)
    assert f"{BF16_REDUCTION_FIELD}=True ({BF16_REDUCTION_ENV} unset)" in message
    assert f"{BF16_REDUCTION_FIELD}=False ({BF16_REDUCTION_ENV}=off)" in message
    assert not (output_root / "layer-quanta" / "layer-001" / "checkpoints").exists()


def _skeleton_until_install(real_fixture, *, layer, skeleton_dtype=None,
                            installed_dtype=None):
    """``fixture`` with one layer's weight as the streaming skeleton holds it.

    Until the layer installs, its ``proj.weight`` is a meta parameter in
    ``skeleton_dtype``: GLM's skeleton is torch's default float32 against a
    bf16 checkpoint.  Install replaces it with the checkpoint tensor, as
    ``_fast_install`` does, cast to ``installed_dtype`` when one is given.
    """

    def build(layers):
        model, context, runner, cache = real_fixture(layers)
        proj = model.model.layers[layer].proj
        real = proj.weight
        if installed_dtype is not None:
            real = torch.nn.Parameter(real.detach().to(installed_dtype),
                                      requires_grad=real.requires_grad)
        if skeleton_dtype is not None:
            proj.weight = torch.nn.Parameter(
                torch.empty(real.shape, dtype=skeleton_dtype, device="meta"),
                requires_grad=real.requires_grad)
        install = context.install

        def install_checkpoint(index, **kwargs):
            if int(index) == layer and proj.weight is not real:
                proj.weight = real
            return install(index, **kwargs)

        context.install = install_checkpoint
        return model, context, runner, cache

    return build


def _campaign_receipt(tmp_path, monkeypatch, output_root):
    runner_a, _ = _stage_a(tmp_path, monkeypatch)
    runner_a.context.settle_prefetch_layers = lambda layers: None
    return run_adjoint_capture_core(
        runner_a, draw(), execution=_execution(tmp_path),
        output_root=output_root, stride=2,
        source_model_identity=_model_identity("joint-source"),
        unit_roster_sha256=_hex("a"), plan_sha256=_hex("d"),
        prepared_sha256=_hex("e"), read_manifest_sha256=_hex("f"),
        implementation_sha256=aura._aura_source_sha256())


def test_the_proof_reads_only_the_shape_off_a_meta_skeleton(tmp_path, monkeypatch):
    """PQ #1102: a skeleton's dtype is torch's default, not the checkpoint's.

    The layer-44 quantum refused its own prepare on the first routed expert,
    comparing the prepared bf16 bytes with GLM's float32 meta skeleton.  Before
    install the proof may compare only the shape; the installed tensor proves
    the dtype and bytes, and the replay is unchanged.
    """
    single_root = tmp_path / "single"
    single = _single_run(single_root, monkeypatch,
                         checkpoint=single_root / "checkpoints")
    output_root = tmp_path / "campaign"
    receipt = _campaign_receipt(tmp_path, monkeypatch, output_root)
    monkeypatch.setitem(globals(), "fixture", _skeleton_until_install(
        fixture, layer=1, skeleton_dtype=torch.float64))
    payload, _record, _counters = _run_quantum(
        tmp_path, monkeypatch, single=single, layer=1, receipt=receipt,
        output_root=output_root, plan_sha=_hex("d"), prepared_sha=_hex("e"))
    assert payload["costs"]
    for name, rows in payload["costs"].items():
        for fmt, row in rows.items():
            single_row = single[0]["costs"][name][fmt]
            assert row["signed_components_per_probe"] == \
                single_row["signed_components_per_probe"], (name, fmt)
            assert row["x2_per_probe"] == single_row["x2_per_probe"]


def test_the_installed_source_still_proves_dtype_and_bytes(tmp_path, monkeypatch):
    """PQ #1102: the skeleton check moved, it was not dropped.

    A layer that installs in a dtype the prepare never proved is refused at
    install, before its first render is consumed, and the refusal names both
    sides.
    """
    single_root = tmp_path / "single"
    single = _single_run(single_root, monkeypatch,
                         checkpoint=single_root / "checkpoints")
    output_root = tmp_path / "campaign"
    receipt = _campaign_receipt(tmp_path, monkeypatch, output_root)
    monkeypatch.setitem(globals(), "fixture", _skeleton_until_install(
        fixture, layer=1, installed_dtype=torch.float64))
    with pytest.raises(RuntimeError) as refused:
        _run_quantum(tmp_path, monkeypatch, single=single, layer=1,
                     receipt=receipt, output_root=output_root,
                     plan_sha=_hex("d"), prepared_sha=_hex("e"))
    message = str(refused.value)
    assert "differs from the installed source for model.layers.1.proj@" in message
    assert "prepared shape [16, 16] dtype torch.float32 bytes 1024" in message
    assert "installed shape [16, 16] dtype torch.float64 bytes 2048" in message


def test_quantum_writes_only_inside_its_output_space(tmp_path, monkeypatch):
    single_root = tmp_path / "single"
    single = _single_run(single_root, monkeypatch,
                         checkpoint=single_root / "checkpoints")
    output_root = tmp_path / "campaign"
    runner_a, _ = _stage_a(tmp_path, monkeypatch)
    runner_a.context.settle_prefetch_layers = lambda layers: None
    receipt = run_adjoint_capture_core(
        runner_a, draw(), execution=_execution(tmp_path),
        output_root=output_root, stride=2,
        source_model_identity=_model_identity("joint-source"),
        unit_roster_sha256=_hex("a"), plan_sha256=_hex("d"),
        prepared_sha256=_hex("e"), read_manifest_sha256=_hex("f"),
        implementation_sha256=aura._aura_source_sha256())

    def tree(root):
        return {str(p.relative_to(root)) for p in Path(root).rglob("*")}

    before = tree(output_root)
    _payload, record, _counters = _run_quantum(
        tmp_path, monkeypatch, single=single, layer=0, receipt=receipt,
        output_root=output_root, plan_sha=_hex("d"), prepared_sha=_hex("e"))
    after = tree(output_root)
    allowed = str(Path("layer-quanta") / "layer-000")
    outside = {path for path in after - before if not path.startswith(allowed)}
    assert not outside, outside
    space = Path(record["output_space"]["root"])
    assert (space / "checkpoints" / "manifest.json").is_file()


def test_band_and_receipt_slices_give_byte_identical_payloads(tmp_path, monkeypatch):
    """Stream vs batch (PQ #993): the band's slice prices like the receipt's.

    A real Stage A run seals checkpoint 2 and its receipt; the band tool
    rebuilds band 2 from the sealed sources alone. For every layer the band
    serves, compared byte for byte: the slice (canonical JSON: run
    identity, stride, boundary storage, checkpoint record, boundary
    entries), the bound record (canonical JSON, so its ``identity_sha256``
    and ``adjoint.slice_sha256``), and the whole pickled cost payload --
    ``costs`` rows, ``stats``, the journal identity and the
    ``distributed_quantum`` block naming the slice digest.
    """
    import copy
    import pickle
    import shutil
    from prismaquant.cost_stage_checkpoint import canonical_json_bytes
    from prismaquant.joint_adjoint_band import build_band_receipt
    from prismaquant.joint_adjoint_slices import validate_band_receipt

    single_root = tmp_path / "single"
    single = _single_run(single_root, monkeypatch,
                         checkpoint=single_root / "checkpoints")
    output_root = tmp_path / "campaign"
    runner_a, _ = _stage_a(tmp_path, monkeypatch)
    runner_a.context.settle_prefetch_layers = lambda layers: None
    identities = []
    bind = StreamedBoundaryArtifacts.bind

    def recorded(self, identity, **kw):
        identities.append(copy.deepcopy(identity))
        return bind(self, identity, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(StreamedBoundaryArtifacts, "bind", recorded)
        receipt = run_adjoint_capture_core(
            runner_a, draw(), execution=_execution(tmp_path),
            output_root=output_root, stride=2,
            source_model_identity=_model_identity("joint-source"),
            unit_roster_sha256=_hex("a"), plan_sha256=_hex("d"),
            prepared_sha256=_hex("e"), read_manifest_sha256=_hex("f"),
            implementation_sha256=aura._aura_source_sha256())
    receipt = json.loads(json.dumps(receipt))
    band = build_band_receipt(
        output_root=output_root, boundary=2, plan_sha256=_hex("d"),
        prepared_sha256=_hex("e"), read_manifest_sha256=_hex("f"),
        stride_value=2, stride_source=None, unit_roster_sha256=_hex("a"),
        bind_identity=identities[-1])
    validate_band_receipt(band)
    assert band["band"]["layers"] == [1, 0]
    for layer in band["band"]["layers"]:
        from_receipt = stage_a_slice(receipt, layer)
        from_band = stage_a_slice(band, layer)
        assert canonical_json_bytes(from_band, where="band slice") == \
            canonical_json_bytes(from_receipt, where="receipt slice")
        runs = {}
        for name, adjoint_slice in (("receipt", from_receipt), ("band", from_band)):
            payload, record, _counters = _run_quantum(
                tmp_path, monkeypatch, single=single, layer=layer, receipt=None,
                output_root=output_root, plan_sha=_hex("d"), prepared_sha=_hex("e"),
                adjoint_slice=adjoint_slice)
            runs[name] = (payload, record)
            # The next run starts from an empty output space, not a resume.
            shutil.rmtree(record["output_space"]["root"])
        (payload_r, record_r), (payload_b, record_b) = runs["receipt"], runs["band"]
        assert payload_r["costs"], "the quantum priced nothing"
        assert canonical_json_bytes(record_b, where="record") == \
            canonical_json_bytes(record_r, where="record")
        assert record_b["adjoint"]["slice_sha256"] == adjoint_slice_sha256(from_band)
        assert pickle.dumps(payload_b) == pickle.dumps(payload_r), layer
        assert adjoint_slice_sha256(from_band).encode() in pickle.dumps(payload_b)


def test_progress_cadence_at_chunk_granularity(tmp_path, monkeypatch):
    from prismaquant.joint_cost_quantum import QuantumProgress

    chunks = [{"name": "layer-001-chunk-000", "start_bytes": 0, "end_bytes": 50},
              {"name": "layer-001-chunk-001", "start_bytes": 50, "end_bytes": 100}]
    windows = [{"window_index": 0, "names": ["a"], "statistics_bytes": 1,
                "render_file_upper_bound_bytes": 40, "candidate_count": 1},
               {"window_index": 1, "names": ["b"], "statistics_bytes": 1,
                "render_file_upper_bound_bytes": 60, "candidate_count": 1}]
    progress_file = tmp_path / "progress.json"
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_PATH", str(progress_file))
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_TOKEN", "token")
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_PHASES", json.dumps(
        ["head", "layer-001-chunk-000", "layer-001-chunk-001"]))
    frontier = ChunkFrontier(chunks=chunks, windows=windows)
    progress = QuantumProgress(frontier=frontier, base_units=3)
    progress.enter_head(0)
    assert progress_file.is_file()
    first = json.loads(progress_file.read_text())
    assert first["phase"] == "head" and first["units_completed"] == 3
    progress.priced(4)
    progress.window_done(windows[0])
    progress.commit()
    second = json.loads(progress_file.read_text())
    assert second["phase"] == "layer-001-chunk-000"
    # Cumulative across phases, continuing from the head-committed base:
    # 3 head units + 4 quantum-local units.
    assert second["units_completed"] == 7
    progress.priced(5)
    progress.window_done(windows[1])
    progress.commit()
    third = json.loads(progress_file.read_text())
    assert third["phase"] == "layer-001-chunk-001"
    assert third["units_completed"] == 8


def test_counters_shapes_per_contract(tmp_path):
    chunks = [{"name": "layer-001-chunk-000", "start_bytes": 0, "end_bytes": 100}]
    windows = [{"window_index": 0, "names": ["a"], "statistics_bytes": 1,
                "render_file_upper_bound_bytes": 100, "candidate_count": 1}]
    frontier = ChunkFrontier(chunks=chunks, windows=windows)
    counters = QuantumCounters(quantum_id="layer-001", identity_sha256=_hex("1"),
                               chunks=chunks, frontier=frontier)
    counters.open()
    counters.enter_phase()
    counters.open_window(0, windows[0])
    counters.close_window(0, kernel_active_s=0.5, wall_s=1.0)
    counters.chain_step(layers=7, backwards=7 * 4 * 2, wall_s=10.0,
                        kernel_active_s=4.0)
    block = counters.finish(units_done=1, units_total=1)
    assert block["schema"] == "prismaquant.joint_layer_quantum.counters.v1"
    assert {"bytes_from_ram", "bytes_from_stage", "bytes_from_pool"} <= set(block)
    for phase in block["phases"]:
        assert {"name", "start_bytes", "end_bytes", "entered_unix",
                "bytes_from_ram", "bytes_from_stage", "bytes_from_pool"} <= set(phase)
    for window in block["windows"]:
        assert {"window_index", "render_bytes_upper_bound", "bytes_from_ram",
                "bytes_from_stage", "bytes_from_pool", "kernel_active_s",
                "wall_s"} <= set(window)
    assert {"gpu_joules", "gpu_power_w_p50", "gpu_power_w_p95",
            "gpu_power_w_max", "kernel_active_s", "wall_s",
            "kernel_active_ratio"} <= set(block)
    assert block["chain"]["layers"] == 7 and block["chain"]["backwards"] == 56
    text = json.dumps(block)
    assert "gpu_utilization" not in text and "utilization_percent" not in text


def test_stage_a_receipt_first_writer_wins(tmp_path):
    space = adjoint_space(tmp_path)
    space.mkdir(parents=True)
    receipt = {"schema": "prismaquant.joint_adjoint_capture.v1",
               "status": "complete"}
    assert write_adjoint_receipt(space, receipt)
    with pytest.raises(RuntimeError, match="already exists"):
        write_adjoint_receipt(space, {"schema": "x", "status": "complete"})


def test_checkpoint_roundtrip_is_digest_checked(tmp_path):
    tensor_a = torch.randn(3, 4)
    tensor_b = torch.randn(2, 2)
    from prismaquant.joint_adjoint_checkpoints import write_adjoint_checkpoint

    record = write_adjoint_checkpoint(
        adjoint_space(tmp_path), boundary=5,
        session={"generation": "g" * 32, "kind": "adjoint_checkpoint"},
        cotangents={(0, 0): tensor_a, (1, 1): tensor_b},
        shared_adjoint={(0, 0): SharedStateCotangents().state_dict()},
        shared_pass={0: {"captured": None}})
    cotangents, shared_adjoint, shared_pass = load_adjoint_checkpoint(
        adjoint_space(tmp_path), record)
    assert torch.equal(cotangents[(0, 0)], tensor_a)
    assert torch.equal(cotangents[(1, 1)], tensor_b)
    assert list(shared_pass) == [0]
    tampered = json.loads(json.dumps(record))
    tampered["activation_entries"][0]["sha256"] = _hex("0")
    with pytest.raises(RuntimeError):
        load_adjoint_checkpoint(adjoint_space(tmp_path), tampered)


def test_attached_storage_is_read_only(tmp_path):
    producer_policy = _boundary_policy(tmp_path / "gen")
    with StreamedBoundaryArtifacts(producer_policy) as producer:
        producer.bind({"fixture": "producer"}, n_probes=2, published=True)
        producer.write(torch.ones(2, 2), batch_index=0, boundary_index=0)
        reference = next(iter(producer._references.values()))
        generation = producer.session["generation"]
    reader_policy = _boundary_policy(tmp_path / "gen")
    with StreamedBoundaryArtifacts(reader_policy) as reader:
        reader.attach(producer.session, n_probes=2)
        with reader.prefetch([reference]) as window:
            assert torch.equal(window.get(reference), torch.ones(2, 2))
        with pytest.raises(RuntimeError):
            reader.write(torch.zeros(2, 2), batch_index=0, boundary_index=1)
        with pytest.raises(RuntimeError):
            reader.retire(reference)
    assert (Path(reader_policy["directory"]) / generation / "generation.json").is_file()
    status = json.loads((Path(reader_policy["directory"]) / generation /
                         "generation.json").read_text())
    # A published generation's entries survive its owner; the foreign status
    # file was never rewritten by the attached reader.
    assert status["status"] == "complete"
    assert (Path(reader_policy["directory"]) / generation / "entries").is_dir()


def test_unpublished_generation_still_retires_every_entry(tmp_path):
    producer_policy = _boundary_policy(tmp_path / "work")
    with StreamedBoundaryArtifacts(producer_policy) as producer:
        producer.bind({"fixture": "working"}, n_probes=2)
        producer.write(torch.ones(2, 2), batch_index=0, boundary_index=0)
        generation = producer.session["generation"]
        entries = Path(producer_policy["directory"]) / generation / "entries"
        assert list(entries.glob("*.pt"))
    assert not list(entries.glob("*.pt"))  # the single run's disposable generation


def test_shared_state_cotangents_state_dict_roundtrip():
    owner = SharedStateCotangents()
    pass_state = {"shared_kv_states": {3: (torch.ones(2, 2),)}}
    grafted = owner.graft(pass_state)
    leaf = grafted["shared_kv_states"][3][0]
    leaf.grad = torch.full((2, 2), 0.5)
    owner.harvest()
    state = owner.state_dict()
    restored = SharedStateCotangents()
    restored.load_state_dict(state)
    slot = ("shared_kv_states", 3, 0)
    assert torch.equal(restored._acc[slot], owner._acc[slot])
    assert restored.n_harvested == owner.n_harvested


def _journal(tmp_path, rows, identity_extra):
    """A synthetic AURA journal: manifest + per-unit envelopes."""
    from prismaquant.aura_cost import (
        _aura_unit_checkpoint_path, _write_aura_checkpoint_manifest,
        _write_aura_unit_checkpoint,
    )

    root = tmp_path
    identity = {"schema": "prismaquant.aura.checkpoint.identity.v1",
                "git_commit": "1" * 40, **identity_extra}
    identity_sha256 = _write_aura_checkpoint_manifest(root, identity, sorted(rows))
    for name, state in rows.items():
        _write_aura_unit_checkpoint(root, qname=name,
                                    identity_sha256=identity_sha256, state=state)
    return root


def _unit_state(value):
    return {"g_trace": 0.25, "rows": {"FP8_E4M3": {
        "s2": value, "s4": value * value, "x2_probe": [value], "dw_src": "rendered"}},
        "col_energy": None,
        "joint_aura_rows": {"FP8_E4M3": {"signed_per_probe": [value]}}}


def test_comparison_tool_on_a_synthetic_pair(tmp_path):
    import subprocess
    import sys

    rows = {"model.layers.13.proj": _unit_state(0.5),
            "model.layers.13.mlp.gate_proj": _unit_state(0.25)}
    left = _journal(tmp_path / "left", dict(rows), {"side": "single"})
    right = _journal(tmp_path / "right", dict(rows), {"side": "quantum"})
    tool = Path(__file__).resolve().parents[1] / "tools" / "compare_joint_layer_gate.py"
    verdict_path = tmp_path / "verdict.json"
    code = subprocess.run(
        [sys.executable, str(tool), "--single-run", str(left),
         "--quantum", str(right), "--layers", "13", "--json", str(verdict_path)],
        capture_output=True, text=True, check=False).returncode
    assert code == 0
    verdict = json.loads(verdict_path.read_text())
    assert verdict["all_match"] is True
    assert verdict["layers"][0]["verdict"] == "match"

    tampered = dict(rows)
    tampered["model.layers.13.proj"] = _unit_state(0.75)
    wrong = _journal(tmp_path / "wrong", tampered, {"side": "quantum"})
    code = subprocess.run(
        [sys.executable, str(tool), "--single-run", str(left),
         "--quantum", str(wrong), "--layers", "13"],
        capture_output=True, text=True, check=False).returncode
    assert code == 1


def test_quantum_requires_dev_mode(identity_files, monkeypatch):
    from prismaquant.joint_cost_quantum import main

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    record = _valid_record(identity_files)
    record_path = identity_files["output_root"].parent / "record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="PRISMAQUANT_DEV_MODE"):
        main(["--quantum", str(record_path),
              "--quantum-sha256", hashlib.sha256(record_path.read_bytes()).hexdigest(),
              "--plan", str(identity_files["plan"][0]),
              "--plan-sha256", identity_files["plan"][1],
              "--prepared", str(identity_files["prepared"][0]),
              "--prepared-sha256", identity_files["prepared"][1],
              "--adjoint-slice", str(identity_files["adjoint"][0]),
              "--adjoint-slice-sha256", identity_files["adjoint"][1],
              "--output-root", str(identity_files["output_root"])])


def test_publish_quantum_outputs_writes_exactly_the_contract_shapes(tmp_path):
    from prismaquant.joint_cost_quantum import publish_quantum_outputs

    output_root = tmp_path / "campaign"
    # Producer-shaped: indices only (D2) -- the unit count arrives with the
    # resolved windows' names, not the record.
    record = _quantum_record(
        output_root=output_root, layer=0, checkpoint_boundary=2, chain=[1],
        windows=[{"window_index": 0}],
        total_bytes=100, plan_sha=_hex("d"), prepared_sha=_hex("e"),
        adjoint_sha=_hex("f"))

    def tree(root):
        return {str(p.relative_to(root)) for p in Path(root).rglob("*")}

    space = Path(record["output_space"]["root"])
    space.mkdir(parents=True)
    before = tree(output_root)
    status = publish_quantum_outputs(
        record, payload={"costs": {"model.layers.0.proj": {}}},
        result={"schema": "prismaquant.joint_cost_quantum.execution.v1"},
        counters={"schema": "prismaquant.joint_layer_quantum.counters.v1"},
        units_total=1)
    written = tree(output_root) - before
    assert written == {"layer-quanta/layer-000/cost.pkl",
                       "layer-quanta/layer-000/results.json",
                       "layer-quanta/layer-000/counters.json",
                       "layer-quanta/layer-000/status.json"}
    assert status["status"] == "complete"
    assert status["units"] == [1, 1]
    assert status["schema"] == "prismaquant.joint_layer_quantum.status.v1"
    # The fallback counts the payload when the caller omits the total.
    fallback = publish_quantum_outputs(
        record, payload={"costs": {"a": {}, "b": {}}},
        result={"schema": "prismaquant.joint_cost_quantum.execution.v1"},
        counters={"schema": "prismaquant.joint_layer_quantum.counters.v1"})
    assert fallback["units"] == [2, 2]
    gapped = publish_quantum_outputs(
        record, payload=None,
        result={"schema": "prismaquant.joint_cost_quantum.execution.v1"},
        counters={"schema": "prismaquant.joint_layer_quantum.counters.v1"},
        units_total=1)
    assert gapped["status"] == "gapped"
    assert gapped["units"] == [0, 1]


def test_window_index_order_is_the_handshake(identity_files):
    """D2: the record seals ordered indices; anything else refuses (exit 3)."""
    record = _valid_record(
        identity_files, windows=[{"window_index": 0}, {"window_index": 1}])
    assert record_window_indices(record) == [0, 1]
    for bad in ([{"window_index": 1}, {"window_index": 0}],
                [{"window_index": 0}, {"window_index": 0}],
                [{"window_index": 0}, {"window_index": 2}],
                [{"names": ["model.layers.1.proj"]}]):
        broken = dict(record)
        broken["windows"] = bad
        with pytest.raises(QuantumIdentityRefused):
            record_window_indices(broken)


@pytest.mark.parametrize("binding, match", [
    # D3: a pre-stage-A record (receipt_sha256 None) refuses with the fix.
    ({"receipt_sha256": None}, "bind_adjoint_slice"),
    # PQ #993: a record binding a whole receipt refuses; it reads a slice.
    ({"receipt_sha256": "a" * 64}, "whole stage-A receipt"),
])
def test_unbound_record_names_the_reseal(identity_files, binding, match):
    record = _valid_record(identity_files)
    record["adjoint"] = {**{k: v for k, v in record["adjoint"].items()
                            if k not in ("slice_sha256", "slice_path")}, **binding}
    record["identity_sha256"] = canonical_json_sha256(
        {k: v for k, v in record.items() if k != "identity_sha256"},
        where="record")
    record_path = identity_files["output_root"].parent / "unbound.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    with pytest.raises(QuantumIdentityRefused, match=match):
        verify_quantum_identity(
            quantum_path=record_path,
            quantum_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
            plan_path=identity_files["plan"][0],
            plan_sha256=identity_files["plan"][1],
            prepared_path=identity_files["prepared"][0],
            prepared_sha256=identity_files["prepared"][1],
            adjoint_path=identity_files["adjoint"][0],
            adjoint_sha256=identity_files["adjoint"][1],
            output_root=identity_files["output_root"])


def test_quantum_refuses_a_slice_of_another_campaign(identity_files, tmp_path):
    """The slice's run header must answer for the argv plan (PQ #993)."""
    from test_stage_b_band_binding import synthetic_receipt
    foreign = synthetic_receipt(
        plan_sha256=_hex("0"), prepared_sha256=identity_files["prepared"][1],
        scope={"fixture": True}, num_layers=2, stride=1)
    path = tmp_path / "foreign-slice.json"
    digest = write_adjoint_slice(path, stage_a_slice(foreign, 1), layer=1)
    record = _quantum_record(
        output_root=identity_files["output_root"], layer=1, checkpoint_boundary=2,
        chain=[], windows=[{"window_index": 0}], total_bytes=100,
        plan_sha=identity_files["plan"][1], prepared_sha=identity_files["prepared"][1],
        adjoint_sha=digest, slice_path=path)
    record_path = identity_files["output_root"].parent / "foreign.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    with pytest.raises(QuantumIdentityRefused, match="another plan_sha256"):
        verify_quantum_identity(
            quantum_path=record_path,
            quantum_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
            plan_path=identity_files["plan"][0],
            plan_sha256=identity_files["plan"][1],
            prepared_path=identity_files["prepared"][0],
            prepared_sha256=identity_files["prepared"][1],
            adjoint_path=path, adjoint_sha256=digest,
            output_root=identity_files["output_root"])


def test_resolve_handshake_refuses_stale_records(tmp_path, monkeypatch):
    """The D2 handshake: wrong count or advisory-names mismatch refuses."""
    torch.manual_seed(85)
    model, context, runner, cache = fixture()
    context.settle_prefetched_layers = lambda layers, *, retry_availability=False: None
    context.source_residency_snapshot = lambda layers, include_head=False: {
        "owners": [], "unique_storage_bytes": sum(
            p.numel() * p.element_size() for p in model.parameters())}
    cache, _proofs = _prepared_cache(model, context, runner, cache,
                                     tmp_path / "shared")
    execution = _execution(tmp_path)
    retained = quantum_retained_state(execution)
    roster = quantum_layer_roster(
        runner, {f"model.layers.{i}.proj": list(FORMATS)
                 for i in range(runner.num_layers)}, 1)
    output_root = tmp_path / "campaign"
    admitted = _preflight_windows(runner, cache, None)[1]
    assert len(admitted) >= 1
    good = _quantum_record(
        output_root=output_root, layer=1, checkpoint_boundary=2, chain=[],
        windows=[{"window_index": index} for index in range(len(admitted))],
        total_bytes=100, plan_sha=_hex("d"), prepared_sha=_hex("e"),
        adjoint_sha=_hex("f"))
    resolved = resolve_quantum_windows(
        good, layer=1, names=roster.names, linears=roster.linears,
        render_formats=roster.render_formats, production_cache=cache,
        operator_windows=retained.operator_windows,
        retained_budget=retained.retained_budget,
        source_bytes=retained.source_bytes)
    assert [w["names"] for w in resolved] == [
        list(w.original_full_target_names) for w in admitted]
    # One window too many: the record is stale, re-seal it.
    stale = dict(good)
    stale["windows"] = list(good["windows"]) + [
        {"window_index": len(admitted)}]
    with pytest.raises(QuantumIdentityRefused, match="re-seal"):
        resolve_quantum_windows(
            stale, layer=1, names=roster.names, linears=roster.linears,
            render_formats=roster.render_formats, production_cache=cache,
            operator_windows=retained.operator_windows,
            retained_budget=retained.retained_budget,
            source_bytes=retained.source_bytes)
    # Advisory names that disagree with the sealed budget: refused, not trusted.
    lying = dict(good)
    lying["windows"] = [dict(good["windows"][0], names=["model.layers.0.proj"])] + \
        list(good["windows"][1:])
    with pytest.raises(QuantumIdentityRefused, match="disagree"):
        resolve_quantum_windows(
            lying, layer=1, names=roster.names, linears=roster.linears,
            render_formats=roster.render_formats, production_cache=cache,
            operator_windows=retained.operator_windows,
            retained_budget=retained.retained_budget,
            source_bytes=retained.source_bytes)
