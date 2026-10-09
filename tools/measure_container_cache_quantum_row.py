"""One complete Stage B layer quantum under the cache-peak sampler (PQ #2463).

Part of PQ #1091. The probe tool beside this one
(``tools.measure_container_cache_row``) exercises the served compile path
alone. This tool runs a complete Stage B row instead: the Stage A adjoint
capture core (``joint_cost_stage_a.run_adjoint_capture_core``) on a tiny
fixture model, then one layer quantum core
(``joint_cost_quantum.run_layer_quantum_core``) for that capture's slice,
under the container-cache sampler. The quantum replays its layer through
the served activation quantiser path, so a CUDA run compiles and writes
the inductor and Triton caches the ceiling must cover.

Workload scope: two decoder layers, width 16, calibration draw of five
4-token rows, formats ``FP8_E4M3``/``NVFP4A16``/``BF16``. Fixed seed 85,
fixed probe count, fixed budgets. The receipt binds the command, the
runtime versions, the initial cache state (captured BEFORE the sampler
starts), the samples, the in-process profile and the PB action identity.

``--cache-root`` holds the four compilation caches (hf, triton, inductor,
xdg) as the launcher routes them. ``--workspace`` is the separate charged
temporary workspace (``PRISMAQUANT_TMPDIR``): never the cache root. ``--out``
is the receipt path. ``--device`` selects ``cuda`` (default) or ``cpu``;
``cpu`` is the D38 preflight only and carries no ceiling claim.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import pstats
import socket
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant import container_cache_peak as peak_mod  # noqa: E402
from prismaquant.io_spans import (  # noqa: E402
    GpuPowerSampler,
    read_proc_io,
    read_proc_status,
    stage_span_log,
)

SCHEMA = "prismaquant.container_cache_quantum_row.v1"
#: Fixed workload identity: seed, probes, formats, every run.
WORKLOAD_SEED = 85
N_PROBES = 2
SEED_BASE = 7000
FORMATS = ("FP8_E4M3", "NVFP4A16", "BF16")
HEX = "0123456789abcdef"


def _hex(char: str) -> str:
    return char * 64


def _runtime_versions() -> dict:
    versions = {"schema": SCHEMA}
    try:
        import torch

        versions["torch"] = str(torch.__version__)
        versions["torch_cuda"] = str(torch.version.cuda)
        versions["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["device_name"] = torch.cuda.get_device_name(0)
            versions["device_capability"] = list(
                torch.cuda.get_device_capability(0))
    except ImportError:
        versions["torch"] = None
    try:
        import triton

        versions["triton"] = str(triton.__version__)
    except ImportError:
        versions["triton"] = None
    try:
        import transformers

        versions["transformers"] = str(transformers.__version__)
    except ImportError:
        versions["transformers"] = None
    return versions


def _operator_policy() -> dict:
    """The tiny-fixture operator window policy (mirrors the runtime tests)."""
    from prismaquant.joint_statistics_replay import SCHEMA

    try:
        workers = max(1, len(os.sched_affinity(0)))
    except AttributeError:
        workers = 1
    return dict(schema=SCHEMA, max_statistics_bytes=2048,
                max_candidate_bytes=1024,
                max_render_resident_bytes=1024 * 1024,
                max_load_buffer_bytes=1024 * 1024,
                workspace_reserve_bytes=1024 * 1024,
                max_replay_cotangent_bytes=1024 * 1024,
                prefetch_workers=min(2, workers))


class _DenseLayer:
    """One width-16 dense layer (mirrors the streamed cost tests)."""

    def __init__(self, module):
        self._module = module


def _build_tiny_fixture():
    """Two decoder layers, width 16, on a fake streaming context.

    Inlined from ``tests/test_streamed_cost_checkpoints.py`` so the GPU
    container needs no pytest install: the campaign image ships no test
    tooling. The shapes, seeds and identities match the runtime tests.
    """
    import torch
    import torch.nn as nn

    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    from prismaquant.cost_streaming import (
        STREAMED_MODEL_IDENTITY_SCHEMA,
        StreamedCausalLM,
    )
    from prismaquant.model_profiles.default import DefaultProfile

    class DenseLayer(nn.Module):
        def __init__(self, width=16):
            super().__init__()
            self.proj = nn.Linear(width, width, bias=False)

        def forward(self, hidden_states, **_kwargs):
            if getattr(self, "_fixture_requires_stream_residency", False):
                assert getattr(self, "_fixture_stream_resident", False)
            return torch.tanh(self.proj(hidden_states))

    class TinyLM(nn.Module):
        def __init__(self, state=None, vocab=23, width=16):
            super().__init__()
            self.model = nn.Module()
            self.model.config = SimpleNamespace(layer_types=())
            self.model.embed_tokens = nn.Embedding(vocab, width)
            self.model.layers = nn.ModuleList(
                [DenseLayer(width) for _ in range(2)])
            self.model.norm = nn.Identity()
            self.lm_head = nn.Linear(width, vocab, bias=False)
            if state is not None:
                self.load_state_dict(state)

        def forward(self, input_ids):
            hidden = self.model.embed_tokens(input_ids)
            for layer in self.model.layers:
                hidden = layer(hidden)
            return SimpleNamespace(
                logits=self.lm_head(self.model.norm(hidden)))

    class FakeContext:
        def __init__(self, model, device):
            self.model = model
            self.base_model = model.model
            self.layers = model.model.layers
            self.layers_prefix = "model.layers."
            self.num_layers = len(self.layers)
            self.device = torch.device(device)
            self.dtype = next(model.parameters()).dtype
            self.active = set()
            self.max_active = 0
            self.install_calls = 0

        def install(self, layer, *, require_prefetched=False,
                    prefetch_following=True):
            self.install_calls += 1
            self.active.add(int(layer))
            self.layers[int(layer)]._fixture_stream_resident = True
            self.max_active = max(self.max_active, len(self.active))
            return "fixture"

        def unload(self, layer):
            self.active.discard(int(layer))
            self.layers[int(layer)]._fixture_stream_resident = False
            return 0

        def schedule_prefetch(self, layer):
            return None

        def observe_source_waits(self, sink):
            assert callable(sink)
            return nullcontext()

        def shutdown(self):
            self.active.clear()

    def model_identity(label: str):
        shard_digest = hashlib.sha256(label.encode()).hexdigest()
        value = {
            "config": {"fixture": True},
            "weight_map": {"fixture.weight": "fixture.weight"},
            "shards": [{
                "path": f"/fixture/{label}.safetensors",
                "size": 1,
                "sha256": shard_digest,
            }],
        }
        return {
            "schema": STREAMED_MODEL_IDENTITY_SCHEMA,
            "source": label,
            "resolved_commit": None,
            "content_sha256": canonical_json_sha256(
                value, where="fixture streamed model identity"),
            **value,
        }

    return TinyLM, FakeContext, model_identity, StreamedCausalLM, DefaultProfile


TENSOR_SHAPE = (64, 256)
TENSOR_SEED = 2463


def _run_served_compile_probe() -> dict:
    """Compile and run the served quantiser; probe the KDA kernel on CUDA."""
    import torch

    from prismaquant.format_registry import _make_rtn

    torch.manual_seed(TENSOR_SEED)
    quantise = _make_rtn("fp4_e2m1", 16)
    tensor = torch.randn(*TENSOR_SHAPE)
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")
    started = time.time()
    result = quantise(tensor)
    quantise_s = time.time() - started
    workload = {"quantiser": "fp4_e2m1/g16", "shape": list(TENSOR_SHAPE),
                "seed": TENSOR_SEED, "quantise_s": round(quantise_s, 3),
                "output_mean": float(result.float().mean()),
                "device": str(result.device)}
    try:
        from prismaquant.kernels import kda_chunk
    except ImportError as exc:
        workload["kda_probe"] = {"status": "missing", "error": str(exc)}
        return workload
    if not torch.cuda.is_available():
        workload["kda_probe"] = {"status": "skipped_no_cuda"}
        return workload
    started = time.time()
    digest = kda_chunk.probe_digest("cuda")
    workload["kda_probe"] = {"status": "ran", "sha256": digest["sha256"],
                             "shape": digest["shape"],
                             "probe_s": round(time.time() - started, 3),
                             "compiled": sorted(kda_chunk.compiled_kernels())}
    return workload


def _preview_reserves(device: str) -> tuple:
    """The retained budget's bound, margin and runtime reserve for the row.

    A real dispatch prices the plan for its box. This preview reads the
    live guard the quantum will hold: its physical cap is the plan's
    bound, and the runtime reserve covers the live committed baseline
    (torch/CUDA runtime) plus headroom for Stage A and later imports.
    Runs before the sampler starts so its allocations never starve it.
    """
    import torch

    if torch.device(device).type != "cuda":
        return 50 << 20, 1 << 20, 1 << 20
    from prismaquant.joint_retained_window_plan import (
        OBSERVED_BASELINE_KEY,
    )
    from prismaquant.joint_statistics_replay import (
        check_operator_allocation,
        operator_window_guard,
    )

    preview = operator_window_guard(device)
    physical_limit = int(preview.physical_cap_bytes)
    safety_margin = int(preview.margin_bytes)
    observed = check_operator_allocation(
        preview, "quantum_row_baseline", reserve_bytes=0)
    baseline = int(observed[OBSERVED_BASELINE_KEY])
    del preview
    runtime_reserve = ((baseline + (1 << 31) - 1) // (1 << 30)) * (1 << 30) + (1 << 30)
    return physical_limit, safety_margin, runtime_reserve


def _run_complete_row(*, output_root: Path, device: str, reserves) -> dict:
    """Stage A capture then one layer quantum on the tiny fixture."""
    import torch

    import prismaquant.aura_cost as aura
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    from prismaquant.cost_streaming import StreamedCausalLM
    from prismaquant.joint_adjoint_checkpoints import (
        chain_layers_for,
        derive_checkpoint_boundaries,
    )
    from prismaquant.joint_adjoint_slices import (
        adjoint_slice_sha256,
        stage_a_slice,
        write_adjoint_slice,
    )
    from prismaquant.joint_cost_quantum import (
        ChunkFrontier,
        QuantumCounters,
        QuantumProgress,
        quantum_layer_roster,
        quantum_retained_state,
        resolve_quantum_windows,
        run_layer_quantum_core,
    )
    from prismaquant.joint_cost_stage_a import run_adjoint_capture_core
    from prismaquant.joint_retained_window_plan import RetainedWindowBudget
    from prismaquant.model_profiles.default import DefaultProfile
    from prismaquant.production_weight_cache import ProductionWeightCache

    TinyLM, FakeContext, model_identity, StreamedCausalLM, DefaultProfile = (
        _build_tiny_fixture())

    torch.manual_seed(WORKLOAD_SEED)
    state = TinyLM().eval().state_dict()
    # The spill replay measures 16-bit arithmetic only: the fixture runs
    # bf16 like a production row, on both devices.
    model = TinyLM(state).eval().to(torch.bfloat16).to(device)
    context = FakeContext(model, device)
    context.settle_prefetch_layers = lambda layers: None
    context.settle_prefetched_layers = lambda layers, *, retry_availability=False: None
    context.source_residency_snapshot = lambda layers, include_head=False: {
        "owners": [], "unique_storage_bytes": sum(
            p.numel() * p.element_size() for p in model.parameters())}
    runner = StreamedCausalLM(context, DefaultProfile(),
                              require_prefetched_residency=True,
                              prefetch_lookahead=1)
    weights = {
        (name, fmt): module.weight.detach().clone() + 0.03125
        for name, module in model.named_modules() if name.endswith(".proj")
        for fmt in ("FP8_E4M3", "NVFP4A16")
    }
    cache = ProductionWeightCache(
        weights=dict(weights),
        levers={},
        activation_max_abs={name: 1.0 for name, _ in weights},
    )
    calib = torch.tensor(
        [[1, 2, 3, 4], [4, 3, 2, 1], [2, 3, 4, 1], [3, 4, 1, 2], [1, 3, 2, 4]])
    files, proofs, file_shas = {}, {}, {}
    assets = output_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for index, (key, tensor) in enumerate(cache.weights.items()):
        path = assets / f"{index}.pt"
        if not path.exists():
            torch.save(tensor, path)
        files[key] = str(path)
        from prismaquant.production_weight_cache import (
            _cb_cache_tensor_identity,
        )

        proofs[key] = _cb_cache_tensor_identity(tensor)
        file_shas[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    cache.weights = files
    cache.enable_lru(1 << 20)
    cache.metadata = {
        "source_model_identity": model_identity("joint-source"),
        "calib_hash": "fixture-calibration",
        "verified_cells": {
            key: {"rendered_weight": value,
                  "render_file_sha256": file_shas[key]}
            for key, value in proofs.items()},
    }
    cache.require_file_load_sha256(file_shas, max_file_bytes=1 << 20)

    boundary_storage = {
        "schema": "prismaquant.aura.boundary_storage.v2",
        "directory": str(output_root / "boundaries"),
        "max_resident_bytes": 5 * 256,
        "max_auxiliary_bytes": 1 << 20,
        "max_artifact_bytes": 1 << 24,
        "prefetch_batches": 2,
        "capture_order": "layer_major",
    }
    # The retained budget's physical bound must match the row's live guard:
    # a 50 MiB fixture plan cannot run under a 34 GiB container guard.
    # A real dispatch prices the plan for its box; the preview reads the
    # guard the quantum will hold and states that bound. The runtime
    # reserve covers the live committed baseline (torch/CUDA runtime);
    # the other reserves stay the tiny fixture's sealed numbers.
    physical_limit, safety_margin, runtime_reserve = reserves
    execution = {
        "n_probes": N_PROBES,
        "seed_base": SEED_BASE,
        "probe_microbatch": 1,
        "token_scope": "all",
        "temperature": 1.0,
        "production_act_scales": "0",
        "boundary_storage": boundary_storage,
        "operator_windows": _operator_policy(),
        "retained_operator_windows": {
            "schema": "prismaquant.joint_retained_execution.v1",
            "budget": RetainedWindowBudget(
                physical_limit, safety_margin, 1 << 20, runtime_reserve,
                1 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20, 1024,
                2048, 4 << 20, 4).as_dict(),
            "source_reserve_bytes": 1 << 20,
            "source_loading_reserve_bytes": 2 << 20,
        },
        "min_free_gib": 0,
        "device_envelope_bytes": None,
    }
    stage_root = output_root / "campaign"
    receipt = run_adjoint_capture_core(
        runner, calib, execution=execution,
        output_root=stage_root, stride=2,
        source_model_identity=model_identity("joint-source"),
        unit_roster_sha256=_hex("a"), plan_sha256=_hex("d"),
        prepared_sha256=_hex("e"), read_manifest_sha256=_hex("f"),
        implementation_sha256=aura._aura_source_sha256())
    layer = 1
    adjoint_slice = stage_a_slice(json.loads(json.dumps(receipt)), layer)
    layer_output = stage_root / "layer-quanta" / f"layer-{layer:03d}"
    slice_path = (stage_root / "layer-quanta" / "adjoint-slices"
                  / f"layer-{layer:03d}.json")
    slice_path.parent.mkdir(parents=True, exist_ok=True)
    write_adjoint_slice(slice_path, adjoint_slice, layer=layer)
    record = {
        "schema": "prismaquant.joint_layer_quanta.v1",
        "quantum_id": f"layer-{layer:03d}",
        "layer": layer,
        "campaign": {
            "plan_path": "plan.json", "plan_sha256": _hex("d"),
            "prepared_path": "prepared.json", "prepared_sha256": _hex("e"),
            "read_manifest_sha256": _hex("f"),
            "campaign_scope": {"fixture": True},
            "unit_roster_sha256": _hex("a"),
        },
        "read_set": {
            "manifest_path": "slice.json.gz", "manifest_sha256": _hex("b"),
            "entry_count": 1, "total_bytes": 100,
            "source_phase": {"name": f"layer-{layer:03d}",
                             "start_bytes": 0, "end_bytes": 100},
        },
        "chunks": [{"name": f"layer-{layer:03d}-chunk-000",
                    "start_bytes": 0, "end_bytes": 100}],
        "windows": [{"window_index": 0}],
        "adjoint": {"checkpoint_boundary": receipt["checkpoints"][-1]["boundary"],
                    "chain_layers": list(chain_layers_for(
                        receipt["checkpoints"][-1]["boundary"], layer)),
                    "boundary_artifacts": ".../layer-quanta/adjoint",
                    "slice_sha256": adjoint_slice_sha256(adjoint_slice),
                    "slice_path": str(slice_path)},
        "output_space": {
            "root": str(layer_output),
            "cost_payload": str(layer_output / "cost.pkl"),
            "results": str(layer_output / "results.json"),
            "counters": str(layer_output / "counters.json"),
            "checkpoint_dir": str(layer_output / "checkpoints"),
        },
    }
    record["identity_sha256"] = canonical_json_sha256(
        record, where="record")
    retained = quantum_retained_state(execution)
    roster = quantum_layer_roster(
        runner, {f"model.layers.{i}.proj": list(FORMATS)
                 for i in range(runner.num_layers)}, layer)
    resolved = resolve_quantum_windows(
        record, layer=layer, names=roster.names, linears=roster.linears,
        render_formats=roster.render_formats, production_cache=cache,
        operator_windows=retained.operator_windows,
        retained_budget=retained.retained_budget,
        source_bytes=retained.source_bytes)
    frontier = ChunkFrontier(chunks=record["chunks"], windows=resolved)
    counters = QuantumCounters(quantum_id=record["quantum_id"],
                               identity_sha256=record["identity_sha256"],
                               chunks=record["chunks"], frontier=frontier)
    progress = QuantumProgress(frontier=frontier, base_units=0)
    payload = run_layer_quantum_core(
        runner, cache, calib,
        {name: list(FORMATS) for name in
         [f"model.layers.{i}.proj" for i in range(runner.num_layers)]},
        record=record, adjoint_slice=adjoint_slice, execution=execution,
        output_root=stage_root, projection_backend=None, resume=False,
        resolved_windows=resolved,
        counters=counters, progress=progress)
    derive_checkpoint_boundaries  # bound by the capture above; kept explicit
    # The fixture quantum never quantizes an input, so the served compile
    # path a production NVFP4A16 row always exercises would stay cold and
    # the sampler would record an empty cache. Run the served activation
    # quantiser compile and the KDA capture-kernel probe exactly as the
    # probe tool does, after the quantum, under the same sampler.
    compile_workload = _run_served_compile_probe()
    return {
        "row": "stage_a_capture_plus_stage_b_quantum",
        "layer": layer,
        "units": sorted(payload["costs"]),
        "formats": list(FORMATS),
        "n_probes": N_PROBES,
        "seed_base": SEED_BASE,
        "checkpoints": [c["boundary"] for c in receipt["checkpoints"]],
        "windows": len(resolved),
        "served_compile": compile_workload,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile-out", type=Path, default=None)
    parser.add_argument("--device", default="cuda",
                        help="torch device for the row (cuda measures; cpu preflights)")
    parser.add_argument("--interval-s", type=float,
                        default=peak_mod.SAMPLE_INTERVAL_S)
    args = parser.parse_args(argv)
    cache_root = args.cache_root
    workspace = args.workspace
    if workspace == cache_root or cache_root in workspace.parents:
        parser.error("--workspace must be separate from --cache-root")
    for name in ("HF_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                 "XDG_CACHE_HOME"):
        value = os.environ.get(name)
        if value is None:
            parser.error(f"{name} is not set; launch under the charged cache root")
    workspace.mkdir(parents=True, exist_ok=True)
    if args.profile_out is not None:
        args.profile_out.parent.mkdir(parents=True, exist_ok=True)
    # A retry lands on the same host scratch: clear the cache root so the
    # row measures a cold compile. The root is this row's private scratch;
    # no concurrent attempt shares it.
    import shutil

    cache_root.mkdir(parents=True, exist_ok=True)
    for entry in sorted(cache_root.iterdir()):
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
        else:
            shutil.rmtree(entry)
    # Fork the sampler BEFORE the preview: fork carries only this thread,
    # and the preview initializes CUDA. The child only reads the
    # filesystem, but it must be forked first regardless.
    sampler = peak_mod.CachePeakSampler(cache_root, interval_s=args.interval_s)
    sampler.__enter__()
    try:
        reserves = _preview_reserves(args.device)
    except BaseException:
        sampler.stop()
        raise
    # The initial inventory is captured BEFORE the row runs, on the
    # cleared root, with the sampler already watching.
    initial_state = peak_mod.describe_initial_state(cache_root)
    # A retry lands on the same host workspace: a stale row-output from an
    # earlier attempt would read as another run's chain state and refuse.
    # Each attempt runs under its own nonce.
    nonce = os.environ.get("PRISMABUILD_ACTION_NONCE", "local")
    attempt_root = workspace / f"row-output-{nonce}"
    command = [sys.executable, "-m", "tools.measure_container_cache_quantum_row",
               "--cache-root", str(cache_root), "--workspace", str(workspace),
               "--out", str(args.out), "--device", str(args.device)]
    power = GpuPowerSampler().start()
    spans = stage_span_log("container-cache-quantum-row", power_sampler=power)
    io_before = read_proc_io()
    wall_before = time.time()
    profile = cProfile.Profile()
    failure = None
    workload = {}
    with spans.span("row"):
        try:
            profile.enable()
            workload = _run_complete_row(
                output_root=attempt_root, device=args.device,
                reserves=reserves)
            profile.disable()
        except BaseException as exc:  # noqa: BLE001 - receipt records it
            failure = f"{type(exc).__name__}: {exc}"
            try:
                profile.disable()
            except RuntimeError:
                pass
            raise
        finally:
            sampler.stop()
    power_summary = power.stop()
    measurement = sampler.result(incomplete_scan=failure is not None)
    profile_io = io.StringIO()
    pstats.Stats(profile, stream=profile_io).strip_dirs().sort_stats(
        "tottime").print_stats(30)
    profile_text = profile_io.getvalue()
    if args.profile_out is not None:
        args.profile_out.write_text(profile_text)
    receipt = {
        "schema": SCHEMA,
        "command": command,
        "host": socket.gethostname().split(".")[0],
        "runtime": _runtime_versions(),
        "device": args.device,
        "action_key": os.environ.get("PRISMABUILD_ACTION_KEY"),
        "action_nonce": os.environ.get("PRISMABUILD_ACTION_NONCE"),
        "row_output": str(attempt_root),
        "cache_env": {name: os.environ.get(name) for name in (
            "HF_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
            "XDG_CACHE_HOME", "PRISMAQUANT_TMPDIR",
            "PRISMAQUANT_CONTAINER_CACHE_ROOT",
            "PRISMAQUANT_CONTAINER_CACHE_MAX_BYTES")},
        "initial_state": initial_state,
        "measurement": measurement,
        "workload": workload if failure is None else {"failure": failure},
        "profile_top": profile_text.splitlines()[:40],
        "proc_io_delta": _delta_io(io_before, read_proc_io()),
        "peak_rss_kib": _peak_rss_kib(),
        "wall_s": round(time.time() - wall_before, 3),
        "power": {"summary": power_summary,
                  "samples": list(power.samples), "times": list(power.times),
                  "error": power.error},
        "byte_convention": ("allocated bytes count 512 * st_blocks for each "
                            "unique device/inode, directories included; "
                            "apparent bytes sum st_size"),
        "failure": failure,
    }
    digest = peak_mod.write_receipt(args.out, receipt)
    print(json.dumps({"receipt": str(args.out), "sha256": digest,
                      "peak_allocated_bytes": measurement["peak_allocated_bytes"],
                      "valid": measurement["valid"]}, sort_keys=True))
    # The receipt file lives on the worker's local disk. Print the full
    # canonical bytes to stdout so the PB log carries the evidence.
    print(args.out.read_bytes().decode())
    return 0 if measurement["valid"] else 1


def _delta_io(before: dict, after: dict) -> dict:
    return {key: (after.get(key) - before.get(key)
                  if isinstance(after.get(key), int)
                  and isinstance(before.get(key), int) else None)
            for key in set(before) | set(after)}


def _peak_rss_kib() -> int | None:
    status = read_proc_status()
    peak = status.get("VmHWM")
    return None if peak is None else peak // 1024


if __name__ == "__main__":
    raise SystemExit(main())
