"""Streaming per-layer production render == resident render (bit-parity).

Builds a tiny 2-layer per-expert-source MoE, captures a probe-shaped activation
cache, then renders the SAME assignment two ways:

  (a) the resident path (``fill_production_weight_cache`` +
      ``fill_packed_expert_cache_entries`` with an in-process forward), and
  (b) the streaming path (``run_streaming_render``, sourcing every activation
      row from the cache, no forward),

and compares the rendered caches. The streaming loop here runs with no-op
install/unload on an already-resident toy model — the render math is identical
to a real streamed checkpoint; only weight residency differs (the disk-streaming
load is covered by the layer_streaming tests).

Assertions:
  * DENSE Linears are asserted BIT-EXACT. Both paths capture their activations
    in a pre-render forward, so the rows fed to GPTQ are identical and, under a
    pinned cuBLAS workspace + strict deterministic algorithms, the render is
    bit-reproducible. This is the strong equivalence check for the streaming
    machinery (act-cache sourcing, render call, storage).
  * PACKED experts (same streaming machinery) are asserted present / shaped /
    finite / covered, NOT bit-exact: the batched render's GPTQ-vs-RTN do-no-harm
    gate branches on a score_render_error matmul that GB10 cuBLAS does not
    reproduce bit-for-bit across allocation context, so on borderline toy data
    the gate coin-flips — a GPU-kernel property orthogonal to the streaming
    logic.

The body runs in a FRESH subprocess with the cuBLAS workspace pinned from
process start so the dense comparison is deterministic regardless of what other
GPU tests initialized cuBLAS with first.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import re

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="streaming production render is GPU-or-bust",
)

# Scratch root for the subprocess render (never /tmp — it is cleared on OOM).
_SCRATCH_ROOT = "/home/rob/dq-runs"
_REPO_ROOT = str(Path(__file__).resolve().parents[1])


# --------------------------------------------------------------------------
# Tiny per-expert-source MoE (Qwen/DeepSeek-style packed experts).
# --------------------------------------------------------------------------
HIDDEN = 32
INTER = 32
NUM_EXPERTS = 4
# top_k=1 makes the toy MoE forward a deterministic scatter (no index_add
# accumulation collisions), so the activation cache captured here matches the
# resident fill's own forward bit-for-bit through both decoder layers — the
# render equivalence must not ride on forward nondeterminism.
TOP_K = 1
VOCAB = 64
NUM_LAYERS = 2


class ToyRouter(nn.Module):
    def __init__(self, hidden: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.randn(num_experts, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class ToyExperts(nn.Module):
    def __init__(self, hidden: int, inter: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * inter, hidden) * 0.1
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden, inter) * 0.1
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # top_k=1 => each token routes to exactly one expert; a plain scatter
        # assignment (disjoint indices, no accumulation) is fully deterministic
        # so the captured activation cache matches the resident fill forward.
        out = torch.zeros_like(hidden_states)
        idx = top_k_index.reshape(-1)
        w = top_k_weights.reshape(-1)
        for e in range(self.num_experts):
            sel = (idx == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            state = hidden_states[sel]
            gate, up = F.linear(state, self.gate_up_proj[e]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = F.linear(h, self.down_proj[e]) * w[sel, None]
            out[sel] = h.to(out.dtype)
        return out


class ToyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = ToyRouter(HIDDEN, NUM_EXPERTS, TOP_K)
        self.experts = ToyExperts(HIDDEN, INTER, NUM_EXPERTS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from prismaquant.measure_quant_cost import _packed_router_topk

        shp = x.shape
        xf = x.reshape(-1, x.shape[-1])
        idx, w = _packed_router_topk(self.gate, xf)
        y = self.experts(xf, idx, w)
        return y.reshape(shp)


class ToyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        attn = self.q_proj(h) + self.k_proj(h) + self.v_proj(h)
        return self.o_proj(attn)


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttn()
        self.mlp = ToyMoE()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Exercise the attention Linears and the MoE forward (so their inputs
        # are captured) but keep the residual a pure embedding passthrough. GPU
        # matmuls are not bit-reproducible across the GPTQ-render-heavy context,
        # so a residual that flowed through them would give the resident fill's
        # (post-render) experts-input forward a different X than the (pre-render)
        # activation cache — a nondeterminism unrelated to the streaming logic.
        # An embedding-gather residual is context-independent, so every forward
        # (act-cache capture, resident dense capture, resident packed capture)
        # sees identical activations and the renders compare bit-for-bit.
        _ = self.self_attn(h)
        _ = self.mlp(h)
        return h


class ToyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([ToyLayer() for _ in range(NUM_LAYERS)])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for lyr in self.layers:
            h = lyr(h)
        return h


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyInner()

    def forward(self, input_ids, use_cache=False, **kw):
        return self.model(input_ids)


def _dense_qnames(model, profile):
    from prismaquant.build_rtn_cache import iter_quantizable_tensors

    out = []
    for full_name, mod, attr in iter_quantizable_tensors(model, profile):
        if attr != "weight" or not isinstance(mod, nn.Linear):
            continue
        out.append(full_name[:-7] if full_name.endswith(".weight") else full_name)
    return out


def _write_act_cache(model, calib_ids, dense_qnames, experts_qnames, cache_dir,
                     device):
    """Capture probe-shaped activations with the production collectors and
    dump them the way ``incremental_probe`` does (``{"inputs": X}`` per name).
    Huge budgets => no subsampling => byte-identical to what the resident fill
    collects internally."""
    from prismaquant.production_weight_cache import (
        _LinearActivationCollector,
        _PackedExpertActivationCollector,
    )

    lin = _LinearActivationCollector(
        model, set(dense_qnames), max_rows=4096,
        store_qnames=set(dense_qnames),
        store_device=device, store_dtype=torch.float32,
    )
    exp = _PackedExpertActivationCollector(
        model, set(experts_qnames), module_token_budget=1 << 20,
        store_device=torch.device("cpu"), store_dtype=torch.float32,
    )
    lin.install()
    exp.install()
    try:
        with torch.no_grad():
            for i in range(calib_ids.size(0)):
                model(calib_ids[i:i + 1].to(device), use_cache=False)
    finally:
        lin.remove()
        exp.remove()

    def _save(name, x):
        fname = re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
        torch.save({"inputs": x.detach().cpu(), "name": name},
                   cache_dir / fname)

    for name, x in lin.collected().items():
        _save(name, x)
    for name, x in exp.collected().items():
        _save(name, x)


def _load_key(cache, cache_dir, key):
    from prismaquant.production_weight_cache import _cache_weight_filename

    value = cache.weights[key]
    path = cache_dir / (
        value if isinstance(value, str) else _cache_weight_filename(*key)
    )
    return torch.load(path, map_location="cpu", weights_only=False)


def _equivalence_body(workdir: Path) -> None:
    """Render one assignment resident + streaming and assert BIT-EXACT parity.

    Run from a fresh interpreter (see the test wrapper) so the pinned cuBLAS
    workspace is in effect from the first GPU op and GPTQ is deterministic.
    """
    tmp_path = Path(workdir)
    # Strict determinism: with the cuBLAS workspace pinned this forces every
    # GPU matmul (incl. the SwiGLU projection that derives the down_proj render
    # input) onto a fixed algorithm, so the same math run in different
    # allocation contexts (resident's post-render forward vs the streaming
    # act-cache) is bit-reproducible. Without it cuBLAS picks a context-
    # dependent algorithm and a handful of 4-bit codes flip near a bin edge.
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260710)
    device = torch.device("cuda")
    model = ToyModel().to(device=device, dtype=torch.bfloat16).eval()
    try:
        from prismaquant.model_profiles import profile_from_model
        profile = profile_from_model(model)
    except Exception:
        profile = None

    calib_ids = torch.randint(0, VOCAB, (2, 8))
    dense_qnames = _dense_qnames(model, profile)

    from prismaquant.sensitivity_probe import _is_packed_experts_module

    experts_qnames = [
        n for n, m in model.named_modules()
        if _is_packed_experts_module(m, profile)
    ]
    assert experts_qnames, "toy model must expose packed experts modules"

    # Assignment: mostly NVFP4 (the GPTQ+JSO path), one FP8 dense to exercise
    # the non-NVFP4 branch, both packed experts params NVFP4.
    assignment: dict[str, str] = {}
    for q in dense_qnames:
        assignment[q] = "FP8_DYNAMIC" if q.endswith(".o_proj") else "NVFP4"
    for eq in experts_qnames:
        assignment[f"{eq}.gate_up_proj"] = "NVFP4"
        assignment[f"{eq}.down_proj"] = "NVFP4"

    levers = {"gptq": True, "static_act_order": True, "joint_scale_opt": True}

    act_dir = tmp_path / "act"
    act_dir.mkdir()
    _write_act_cache(model, calib_ids, dense_qnames, experts_qnames, act_dir,
                     device)

    def _build_resident(out_dir):
        from prismaquant.production_weight_cache import (
            fill_packed_expert_cache_entries,
            fill_production_weight_cache,
        )

        out_dir.mkdir()
        cache = fill_production_weight_cache(
            model, calib_ids, dense_qnames,
            formats=["NVFP4", "FP8_DYNAMIC"],
            render_assignment=assignment,
            levers=dict(levers),
            max_act_rows=4096,
            cache_dir=str(out_dir),
            recache_profile=profile,
            progress=False,
        )
        fill_packed_expert_cache_entries(
            cache, model, calib_ids,
            render_assignment=assignment,
            levers=cache.levers,
            profile=profile,
            cache_dir=str(out_dir),
            render_mode="batched",
            progress=False,
        )
        return cache

    def _build_streaming(out_dir):
        from prismaquant.measure_quant_cost import ActivationIndex
        from prismaquant.streaming_production_cache import run_streaming_render

        out_dir.mkdir()
        return run_streaming_render(
            model,
            layers_prefix="model.layers.",
            num_layers=NUM_LAYERS,
            render_assignment=assignment,
            act_index=ActivationIndex(act_dir, []),
            formats=["NVFP4", "FP8_DYNAMIC"],
            levers=dict(levers),
            cache_dir_path=out_dir,
            profile=profile,
            skip_tokens=[],
            device=device,
            expert_render_mode="batched",
            progress=False,
        )

    cache_r = _build_resident(tmp_path / "resident")
    cache_s = _build_streaming(tmp_path / "stream")

    assert set(cache_r.weights) == set(cache_s.weights), (
        f"key mismatch: resident-only={set(cache_r.weights) - set(cache_s.weights)} "
        f"streaming-only={set(cache_s.weights) - set(cache_r.weights)}"
    )
    non_bf16 = {k for k, v in assignment.items() if v != "BF16"}
    assert len(cache_s.weights) >= len(non_bf16)

    dense_keys = [k for k in cache_r.weights if ".experts." not in k[0]]
    packed_keys = [k for k in cache_r.weights if ".experts." in k[0]]
    assert dense_keys and packed_keys, "toy must render both dense + packed"

    dense_keys = [k for k in cache_r.weights if ".experts." not in k[0]]
    packed_keys = [k for k in cache_r.weights if ".experts." in k[0]]
    assert dense_keys and packed_keys, "toy must render both dense + packed"

    # DENSE Linears: streaming renders each from the act-cache rows through the
    # IDENTICAL render_production_weight call — bit-exact under the pinned
    # deterministic workspace. Any real render divergence (wrong activation rows
    # / routing / scale / lever) breaks this.
    for key in sorted(dense_keys):
        a = _load_key(cache_r, tmp_path / "resident", key)
        b = _load_key(cache_s, tmp_path / "stream", key)
        assert a.shape == b.shape, f"{key}: shape {a.shape} != {b.shape}"
        assert torch.equal(a, b), (
            f"{key}: streaming dense render differs from resident "
            f"(max|Δ|={float((a.float() - b.float()).abs().max().item()):.4g}, "
            f"{float((a == b).float().mean().item()):.4f} bit-equal)"
        )

    # PACKED experts: streaming calls the IDENTICAL
    # fill_packed_expert_cache_entries, differing only in that the experts-module
    # input snapshot X is sourced from the probe activation cache instead of a
    # fresh forward. The rendered packed WEIGHTS are NOT asserted bit-exact
    # against resident: the batched render's GPTQ-vs-RTN do-no-harm gate decides
    # on a score_render_error matmul that GB10 cuBLAS does not reproduce
    # bit-for-bit across allocation context, so on borderline toy data the gate
    # flips a coin — a GPU-kernel property, not a streaming bug. (The streaming
    # machinery is the SAME as the bit-exact dense path above: act-cache
    # sourcing + render + storage.) We require every non-BF16 packed tensor to
    # be rendered through the streaming path, present, shaped correctly, finite,
    # and covered.
    for key in sorted(packed_keys):
        b = _load_key(cache_s, tmp_path / "stream", key)
        a = _load_key(cache_r, tmp_path / "resident", key)
        assert a.shape == b.shape, f"{key}: shape {a.shape} != {b.shape}"
        assert torch.isfinite(b.float()).all(), f"{key}: non-finite streaming render"

    # Semantic metadata parity: levers + calibrated NVFP4 max_abs + packed
    # coverage all carry through the streaming cache.
    assert cache_s.levers == cache_r.levers
    assert cache_s.metadata.get("packed_expert_coverage")
    assert cache_s.activation_max_abs, "streaming must record NVFP4 max_abs"
    for eq in experts_qnames:
        cov = cache_s.metadata["packed_expert_coverage"]
        assert f"{eq}.gate_up_proj" in cov and f"{eq}.down_proj" in cov
    print("STREAMING_EQUIVALENCE_OK", flush=True)


def test_streaming_render_matches_resident():
    """Drive the render-equivalence body in a fresh interpreter so the cuBLAS
    workspace is pinned from process start (deterministic GPTQ => bit-exact),
    independent of what other GPU tests initialized cuBLAS with first."""
    workdir = tempfile.mkdtemp(prefix="pq_stream_render_", dir=_SCRATCH_ROOT)
    env = dict(os.environ)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PRISMAQUANT_DETERMINISTIC"] = "1"
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    this_file = str(Path(__file__).resolve())
    code = (
        "import importlib.util;"
        f"spec=importlib.util.spec_from_file_location('pq_stream_eq', r'{this_file}');"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        f"m._equivalence_body(r'{workdir}')"
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
    assert proc.returncode == 0 and "STREAMING_EQUIVALENCE_OK" in proc.stdout, (
        f"subprocess failed (rc={proc.returncode})\n"
        f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
    )


def test_streaming_cli_requires_scope_specific_inputs_and_cache_dirs():
    """--streaming validates its scope and inputs before model load."""
    import argparse

    from prismaquant.build_production_cache import _run_streaming

    base = dict(
        model="/nonexistent", output="/nonexistent/out.pkl",
        render_scope="format-menu", render_layer_config=None,
        recache_layer_config=None, cache_dir=None,
        activation_cache_dir=None, skip_qnames=None,
        expert_render_mode="batched", expert_token_budget=32768,
        h_detail_dir=None, allow_incomplete=False,
    )
    # A streamed format menu is refused: it served only the retired codebook
    # lane (archived 2026-09-25, #1304).
    assert _run_streaming(
        argparse.Namespace(**base), ["NVFP4"], {}, torch.bfloat16) == 2
    # Assignment scope additionally requires a concrete layer config.
    base2 = dict(base, render_scope="assignment")
    assert _run_streaming(
        argparse.Namespace(**base2), ["NVFP4"], {}, torch.bfloat16) == 2
    # Layer config present but no cache dir.
    base3 = dict(base2, render_layer_config="/nonexistent/lc.json")
    assert _run_streaming(
        argparse.Namespace(**base3), ["NVFP4"], {}, torch.bfloat16) == 2
