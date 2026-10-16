"""Inline packed-expert GPTQ render at export == prebuilt-cache render.

Covers ``PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ``: for a 295B-class per-expert-
source MoE the ~588 GB dequant weight cache cannot coexist with the source on
disk, so packed experts are rendered ON THE FLY during export — one layer's
stack at a time — through the SAME ``fill_packed_expert_cache_entries`` batched
GPTQ path the cache builder uses, sourcing the experts-module input snapshot
from the probe activation cache.

This test exports a tiny per-expert-source MoE (reusing the toy fixture from
``test_streaming_production_cache``) twice at an NVFP4-experts assignment:

  (a) with a prebuilt ``ProductionWeightCache`` (``run_streaming_render``), and
  (b) with the inline env gate and NO cache,

and asserts the packed-expert tensor BYTES in the two exported checkpoints are
identical. Both arms source the experts-module input snapshot from the SAME
activation cache, so — under a pinned cuBLAS workspace + strict deterministic
algorithms — the render (including the GPTQ-vs-RTN do-no-harm gate, whose
score matmul is otherwise not GB10-reproducible across allocation context) is
bit-reproducible and the re-derived NVFP4 codes match byte-for-byte.

The activation cache is captured with a SYNCHRONOUS pre-hook rather than the
production reservoir collector: that collector's ``non_blocking=True`` D2H copy
to pageable memory deterministically corrupts the toy snapshot to NaN when the
GPU is under sustained contention (a real box property, orthogonal to the
inline-render logic under test). A synchronous capture gives finite rows the
render can actually key on.

Runs in a FRESH subprocess with the cuBLAS workspace pinned from process start.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

import prismaquant.export_native_compressed as _test_exporter
import prismaquant.native_render_codec as _native_render_codec
from prismaquant.native_render_levers import NativeRenderLevers

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="inline packed-expert render is GPU-or-bust",
)

_SCRATCH_ROOT = "/home/rob/dq-runs"
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_STREAM_TEST = str(Path(__file__).resolve().parent
                   / "test_streaming_production_cache.py")


@pytest.fixture(autouse=True)
def _isolate_export_inputs():
    saved = (_test_exporter._RENDER_LEVERS, _test_exporter._CACHED_ACTIVATIONS,
             _test_exporter._INPUT_GLOBAL_SCALES)
    _test_exporter._RENDER_LEVERS = None
    _test_exporter._CACHED_ACTIVATIONS = None
    _test_exporter._INPUT_GLOBAL_SCALES = None
    yield
    (_test_exporter._RENDER_LEVERS, _test_exporter._CACHED_ACTIVATIONS,
     _test_exporter._INPUT_GLOBAL_SCALES) = saved


def _load_toy_fixtures():
    """Import the toy per-expert-source MoE (ToyModel, _dense_qnames)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pq_toy_moe", _STREAM_TEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_clean_act_cache(model, calib_ids, dense_qnames, experts_qnames,
                           cache_dir, device):
    """Capture probe-shaped activations with SYNCHRONOUS pre-hooks.

    Same on-disk shape as ``incremental_probe`` (``{"inputs": X}`` keyed by the
    canonical Linear / experts-module name), but the copy to CPU is synchronous
    so the rows are always finite even under GPU contention."""
    import torch.nn as nn

    caps: dict[str, list[torch.Tensor]] = {}
    handles = []

    def _pre_hook(name):
        def hook(_module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                return
            x = args[0].detach().reshape(-1, args[0].shape[-1])
            caps.setdefault(name, []).append(x.to("cpu", torch.float32))
        return hook

    name_by_module = {}
    for qname in dense_qnames:
        name_by_module[model.get_submodule(qname)] = qname
    for eq in experts_qnames:
        name_by_module[model.get_submodule(eq)] = eq
    for mod, name in name_by_module.items():
        handles.append(mod.register_forward_pre_hook(_pre_hook(name)))
    try:
        with torch.no_grad():
            for i in range(calib_ids.size(0)):
                model(calib_ids[i:i + 1].to(device), use_cache=False)
        torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()

    for name, parts in caps.items():
        X = torch.cat(parts, dim=0)
        assert torch.isfinite(X).all(), f"{name}: captured non-finite activations"
        fname = re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
        torch.save({"inputs": X, "name": name}, cache_dir / fname)


def _set_export_act_globals(enc, act_dir):
    """Populate the export module's activation globals from the act cache."""
    from prismaquant.measure_quant_cost import ActivationIndex

    levers = NativeRenderLevers.resolve({
        "gptq": True, "scale_sweep": False,
        "static_act_order": True, "joint_scale_opt": True})
    enc._RENDER_LEVERS = levers
    idx = ActivationIndex(act_dir, [])
    enc._CACHED_ACTIVATIONS = enc._LazyActivationCache(idx)
    scales: dict[str, float] = {}
    for name in idx.names():
        try:
            scales[name] = _native_render_codec.compute_nvfp4_input_global_scale(
                idx.load(name), levers=levers)
        except Exception:
            pass
    enc._INPUT_GLOBAL_SCALES = scales


def _packed_tensor_bytes(tensors):
    return {k: v for k, v in tensors.items() if ".experts." in k}


def _inline_export_body(workdir: str) -> None:
    tmp_path = Path(workdir)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260710)
    device = torch.device("cuda")

    toy = _load_toy_fixtures()
    from prismaquant.model_profiles import profile_from_model
    from prismaquant.probe_inventory import (
        is_packed_experts_module as _is_packed_experts_module,
    )

    model = toy.ToyModel().to(device=device, dtype=torch.bfloat16).eval()
    profile = profile_from_model(model)
    calib_ids = torch.randint(0, toy.VOCAB, (2, 8))
    dense_qnames = toy._dense_qnames(model, profile)
    experts_qnames = [
        n for n, m in model.named_modules()
        if _is_packed_experts_module(m, profile)
    ]
    assert experts_qnames, "toy must expose packed experts modules"

    # NVFP4 everywhere (dense + both packed expert params). Only the packed
    # tensors are compared — dense just keeps the model exportable.
    assignment: dict[str, str] = {q: "NVFP4" for q in dense_qnames}
    for eq in experts_qnames:
        assignment[f"{eq}.gate_up_proj"] = "NVFP4"
        assignment[f"{eq}.down_proj"] = "NVFP4"
    levers = {"gptq": True, "static_act_order": True, "joint_scale_opt": True}

    act_dir = tmp_path / "act"
    act_dir.mkdir()
    _write_clean_act_cache(model, calib_ids, dense_qnames, experts_qnames,
                           act_dir, device)

    import prismaquant.export_native_compressed as enc
    from prismaquant.measure_quant_cost import ActivationIndex
    from prismaquant.streaming_production_cache import run_streaming_render

    # ---- Arm (a): prebuilt production cache via the streaming render path.
    cache_dir_a = tmp_path / "cache_a"
    cache_dir_a.mkdir()
    cache = run_streaming_render(
        model,
        layers_prefix="model.layers.",
        num_layers=toy.NUM_LAYERS,
        render_assignment=assignment,
        act_index=ActivationIndex(act_dir, []),
        formats=["NVFP4"],
        levers=dict(levers),
        cache_dir_path=cache_dir_a,
        profile=profile,
        skip_tokens=[],
        device=device,
        expert_render_mode="batched",
        progress=False,
    )

    saved = (enc._PRODUCTION_WEIGHT_CACHE, enc._INLINE_EXPERT_GPTQ,
             enc._CACHED_ACTIVATIONS, enc._INPUT_GLOBAL_SCALES, enc._RENDER_LEVERS)
    try:
        enc._PRODUCTION_WEIGHT_CACHE = cache
        enc._INLINE_EXPERT_GPTQ = False
        _set_export_act_globals(enc, act_dir)
        tensors_a, hist_a = enc._materialize_tensors_inmemory(
            model, assignment, bf16_passthrough=set(), profile=profile)

        # ---- Arm (b): NO cache, inline expert GPTQ gate.
        enc._PRODUCTION_WEIGHT_CACHE = None
        enc._INLINE_EXPERT_GPTQ = True
        _set_export_act_globals(enc, act_dir)
        tensors_b, hist_b = enc._materialize_tensors_inmemory(
            model, assignment, bf16_passthrough=set(), profile=profile)
    finally:
        (enc._PRODUCTION_WEIGHT_CACHE, enc._INLINE_EXPERT_GPTQ,
         enc._CACHED_ACTIVATIONS, enc._INPUT_GLOBAL_SCALES, enc._RENDER_LEVERS) = saved

    packed_a = _packed_tensor_bytes(tensors_a)
    packed_b = _packed_tensor_bytes(tensors_b)
    assert packed_a, "no packed expert tensors emitted (arm a)"
    assert set(packed_a) == set(packed_b), (
        f"packed key mismatch: a-only={set(packed_a) - set(packed_b)} "
        f"b-only={set(packed_b) - set(packed_a)}"
    )
    # The inline arm must actually render (not RTN-by-omission).
    assert hist_b.get(("packed_moe_per_expert", "NVFP4+cached")), (
        f"inline arm did not render packed experts through the cache path: "
        f"{dict(hist_b)}"
    )
    assert not any(lbl.endswith("+rtn") for (_, lbl) in hist_b), (
        f"inline arm fell back to RTN: {dict(hist_b)}"
    )

    for k in sorted(packed_a):
        a, b = packed_a[k], packed_b[k]
        assert a.shape == b.shape and a.dtype == b.dtype, (
            f"{k}: ({a.dtype},{tuple(a.shape)}) != ({b.dtype},{tuple(b.shape)})")
        assert torch.equal(a, b), (
            f"{k}: inline packed-expert export bytes differ from the prebuilt "
            f"cache export (byte-equal fraction "
            f"{float((a == b).float().mean().item()):.4f})"
        )
    print("INLINE_EXPERT_EXPORT_OK", flush=True)


def test_inline_expert_export_matches_prebuilt_cache():
    workdir = tempfile.mkdtemp(prefix="pq_inline_expert_", dir=_SCRATCH_ROOT)
    env = dict(os.environ)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PRISMAQUANT_DETERMINISTIC"] = "1"
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    this_file = str(Path(__file__).resolve())
    code = (
        "import importlib.util;"
        f"spec=importlib.util.spec_from_file_location('pq_inline_eq', r'{this_file}');"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        f"m._inline_export_body(r'{workdir}')"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env, cwd=_REPO_ROOT,
            capture_output=True, text=True, timeout=600,
        )
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
    assert proc.returncode == 0 and "INLINE_EXPERT_EXPORT_OK" in proc.stdout, (
        f"subprocess failed (rc={proc.returncode})\n"
        f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
    )
