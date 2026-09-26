"""M4 frontier packed-expert selection: pin the paths that let the validated
frontier SELECT each expert's format by real KL and ship the same bytes.

Covers the 2026-07-01 M4 batch:
  * ``fill_packed_expert_cache_entries`` force-format mode (eager frontier
    render) and its mutual-exclusion contract with ``render_assignment``;
  * ``render_format_menu_packed_experts`` (eager NVFP4 rung only; loud
    warning when the menu carries no NVFP4);
  * lazy gap-fill persistence (``_persist_lazy_expert_renders``): shards
    rendered during validation must survive into the pickled cache or a
    selected FP8-expert point is unshippable (principle #8);
  * ``apply_activation_max_abs_to_cache`` preserving packed-expert scales —
    the recache replay captures max|module input| under EVERY plan param
    name, which is the wrong tensor for ``down_proj`` (routed post-SwiGLU);
  * ``_module_input_member_name`` selecting the module-input projection's
    clip scale structurally instead of by assignment-dict order.
"""
from __future__ import annotations

import pickle

import pytest
import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.build_production_cache import (
    render_format_menu_packed_experts,
)
from prismaquant.perturbed_x_cache import (
    _ModulePlan,
    _ParamPlan,
    _module_input_member_name,
)
from prismaquant.production_recache import apply_activation_max_abs_to_cache
from prismaquant.production_weight_cache import (
    ProductionWeightCache,
    fill_packed_expert_cache_entries,
)
from prismaquant.validate_assignments_kl import _persist_lazy_expert_renders

from test_packed_expert_cross_domain_gate import ASSIGNMENT, TinyLM


EXPERT_NAMES = sorted(ASSIGNMENT)


def _calib(seed: int = 11) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, 32, (2, 64))


def _model(seed: int = 11) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM().eval()


def _empty_cache(levers=None) -> ProductionWeightCache:
    return ProductionWeightCache(weights={}, levers=dict(levers or {}))


# ---------------------------------------------------------------------------
# force_format contract + eager render
# ---------------------------------------------------------------------------

def test_fill_packed_experts_requires_exactly_one_mode():
    model = _model()
    calib = _calib()
    cache = _empty_cache()
    with pytest.raises(ValueError, match="render_assignment or"):
        fill_packed_expert_cache_entries(
            cache, model, calib, levers={}, profile=None, progress=False)
    with pytest.raises(ValueError, match="not both"):
        fill_packed_expert_cache_entries(
            cache, model, calib,
            render_assignment=ASSIGNMENT, force_format="NVFP4",
            levers={}, profile=None, progress=False)


def test_force_format_renders_all_packed_experts_without_assignment():
    model = _model()
    cache = _empty_cache({"gptq": True})
    coverage = fill_packed_expert_cache_entries(
        cache, model, _calib(),
        force_format="NVFP4",
        levers={"gptq": True},
        profile=None,
        module_token_budget=4096,
        eval_rows_per_expert=8,
        progress=False,
    )
    assert set(coverage) == set(EXPERT_NAMES)
    for full in EXPERT_NAMES:
        assert coverage[full]["fmt"] == "NVFP4"
        assert (full, "NVFP4") in cache.weights
        assert cache.activation_max_abs.get(full, 0.0) > 0.0


def test_packed_render_refuses_a_retired_codebook_rung(tmp_path):
    # A stale assignment naming a retired codebook rung (archived 2026-09-25,
    # #1304) must refuse; it must not skip the expert or render it as BF16.
    model = _model()
    full = EXPERT_NAMES[0]
    with pytest.raises(fr.RetiredFormatError, match="gridbook_lane"):
        fill_packed_expert_cache_entries(
            _empty_cache(),
            model,
            _calib(),
            render_assignment={full: "NVFP4_CB_K16"},
            levers={},
            profile=None,
            cache_dir=tmp_path,
            progress=False,
        )


def test_format_menu_eager_renders_nvfp4_rung_only():
    model = _model()
    cache = _empty_cache({"gptq": True})
    merged = render_format_menu_packed_experts(
        cache, model, _calib(), ["NVFP4", "FP8_DYNAMIC", "BF16"],
        profile=None, module_token_budget=4096,
    )
    assert set(merged) == set(EXPERT_NAMES)
    fp8_canon = fr.canonical_format_name("FP8_DYNAMIC")
    for full in EXPERT_NAMES:
        assert (full, "NVFP4") in cache.weights
        assert (full, fp8_canon) not in cache.weights
    assert cache.metadata["packed_expert_coverage"] == merged


def test_format_menu_eager_warns_without_nvfp4(capsys):
    model = _model()
    cache = _empty_cache()
    merged = render_format_menu_packed_experts(
        cache, model, _calib(), ["FP8_DYNAMIC", "BF16"], profile=None)
    assert merged == {}
    assert cache.weights == {}
    out = capsys.readouterr().out
    assert "no NVFP4 in the format menu" in out


# ---------------------------------------------------------------------------
# lazy gap-fill persistence (Fix A)
# ---------------------------------------------------------------------------

def test_lazy_gap_fill_persists_to_cache_pkl(tmp_path):
    model = _model()
    calib = _calib()
    cache_dir = tmp_path / "shards"
    pkl_path = tmp_path / "frontier_cache.pkl"

    # Frontier build: eager NVFP4 render, streamed to shards, pickled.
    cache = _empty_cache({"gptq": True})
    render_format_menu_packed_experts(
        cache, model, calib, ["NVFP4", "FP8_DYNAMIC", "BF16"],
        profile=None, cache_dir=cache_dir, module_token_budget=4096,
    )
    cache.compact_for_pickle()
    with pkl_path.open("wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # Validator: reload, gap-fill an FP8-expert Pareto point, persist.
    with pkl_path.open("rb") as fh:
        loaded = pickle.load(fh)
    fp8_canon = fr.canonical_format_name("FP8_DYNAMIC")
    fp8_point = {name: fp8_canon for name in EXPERT_NAMES}
    _keys, missing = loaded.assignment_keys(
        fp8_point, include_packed_experts=True)
    assert {n for n, _f in missing} == set(EXPERT_NAMES)
    fill_packed_expert_cache_entries(
        loaded, model, calib,
        render_assignment=fp8_point,
        levers={}, profile=None,
        cache_dir=loaded.cache_dir,
        module_token_budget=4096,
        progress=False,
    )
    _persist_lazy_expert_renders(loaded, str(pkl_path))

    # Downstream consumer (recache/export) reloads the pkl: the lazy FP8
    # keys must resolve — orphaned shards were the M4 ship-blocker.
    with pkl_path.open("rb") as fh:
        reloaded = pickle.load(fh)
    _keys2, missing2 = reloaded.assignment_keys(
        fp8_point, include_packed_experts=True)
    assert missing2 == []
    for name in EXPERT_NAMES:
        assert (name, fp8_canon) in reloaded.weights
        assert (name, "NVFP4") in reloaded.weights  # eager rung intact
    # No stray tmp file from the atomic replace.
    assert list(tmp_path.glob("*.tmp")) == []


def test_source_dtype_manifest_classifies_packed_expert_params(tmp_path):
    # Packed expert checkpoint keys carry NO .weight suffix; without direct
    # classification the allocator drops the BF16 passthrough for every
    # expert on a BF16 source (menu-completeness bug found in the LFM2.5
    # smoke: 44 tensors 'source_dtype_mismatch').
    from safetensors.torch import save_file
    from prismaquant.allocator_candidates import _scan_source_dtype_manifest

    save_file(
        {
            "model.layers.0.feed_forward.experts.gate_up_proj":
                torch.randn(2, 8, 4, dtype=torch.bfloat16),
            "model.layers.0.feed_forward.experts.down_proj":
                torch.randn(2, 4, 4, dtype=torch.bfloat16),
            "model.layers.0.self_attn.q_proj.weight":
                torch.randn(4, 4, dtype=torch.bfloat16),
            "model.layers.0.self_attn.k_proj.weight":
                torch.randn(4, 4, dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    manifest = _scan_source_dtype_manifest(str(tmp_path), profile=None)
    assert manifest[
        "model.layers.0.feed_forward.experts.gate_up_proj"] == "bf16"
    assert manifest[
        "model.layers.0.feed_forward.experts.down_proj"] == "bf16"
    assert manifest["model.layers.0.self_attn.q_proj"] == "bf16"
    assert manifest["model.layers.0.self_attn.k_proj"] == "f32"


def test_source_dtype_manifest_classifies_mtp_tensors(tmp_path):
    # MTP tensors are real source tensors stored under the recipe namespace
    # itself; the historical mtp. skip left them without a source kind, so
    # BF16 passthrough was dropped and --mtp-format=BF16 hard-failed once
    # MTP rows carried costs (35B frontier, 2026-07-02).
    from safetensors.torch import save_file
    from prismaquant.allocator_candidates import _scan_source_dtype_manifest

    save_file(
        {
            "mtp.fc.weight": torch.randn(4, 8, dtype=torch.bfloat16),
            "mtp.layers.0.self_attn.q_proj.weight":
                torch.randn(4, 4, dtype=torch.bfloat16),
            "mtp.layers.0.mlp.experts.gate_up_proj":
                torch.randn(2, 8, 4, dtype=torch.bfloat16),
        },
        str(tmp_path / "model.safetensors"),
    )
    manifest = _scan_source_dtype_manifest(str(tmp_path), profile=None)
    assert manifest["mtp.fc"] == "bf16"
    assert manifest["mtp.layers.0.self_attn.q_proj"] == "bf16"
    assert manifest["mtp.layers.0.mlp.experts.gate_up_proj"] == "bf16"


# ---------------------------------------------------------------------------
# recache preserves packed-expert scales (Fix B)
# ---------------------------------------------------------------------------

def test_recache_preserves_packed_expert_scales():
    cache = _empty_cache()
    cache.activation_max_abs = {
        "model.layers.0.mlp.experts.gate_up_proj": 2.0,
        "model.layers.0.mlp.experts.down_proj": 3.5,
        "model.layers.0.self_attn.q_proj": 1.0,
    }
    apply_activation_max_abs_to_cache(
        cache,
        {
            # Replay records max|module input| under BOTH expert param
            # names — hidden-state max, the wrong tensor for down_proj.
            "model.layers.0.mlp.experts.gate_up_proj": 99.0,
            "model.layers.0.mlp.experts.down_proj": 99.0,
            "model.layers.0.self_attn.q_proj": 4.0,
        },
    )
    scales = cache.activation_max_abs
    assert scales["model.layers.0.self_attn.q_proj"] == 4.0  # re-fitted
    assert scales["model.layers.0.mlp.experts.gate_up_proj"] == 2.0
    assert scales["model.layers.0.mlp.experts.down_proj"] == 3.5
    meta = cache.metadata["activation_recache"]
    assert meta["n_packed_expert_scales_preserved"] == 2


def test_recache_drops_replay_expert_scales_without_render():
    # A BF16-expert assignment never rendered expert entries; the replay's
    # module-input captures under expert names must not leak into the cache.
    cache = _empty_cache()
    cache.activation_max_abs = {"model.layers.0.self_attn.q_proj": 1.0}
    apply_activation_max_abs_to_cache(
        cache,
        {
            "model.layers.0.mlp.experts.down_proj": 99.0,
            "model.layers.0.self_attn.q_proj": 4.0,
        },
    )
    assert "model.layers.0.mlp.experts.down_proj" not in cache.activation_max_abs
    assert cache.activation_max_abs["model.layers.0.self_attn.q_proj"] == 4.0


def test_recache_refits_indexed_per_expert_linear_scales():
    # Per-expert-INDEXED names (DSv4/MiniMax layouts) are plain nn.Linear
    # modules: the replay capture IS the correct per-projection tensor, so
    # their re-fit must be applied, not preserved (review: the packed-only
    # preservation must not over-match indexed names).
    cache = _empty_cache()
    cache.activation_max_abs = {
        "model.layers.0.mlp.experts.5.down_proj": 3.5,
        "model.layers.0.block_sparse_moe.experts.2.w1": 2.0,
    }
    apply_activation_max_abs_to_cache(
        cache,
        {
            "model.layers.0.mlp.experts.5.down_proj": 7.0,
            "model.layers.0.block_sparse_moe.experts.2.w1": 8.0,
        },
    )
    scales = cache.activation_max_abs
    assert scales["model.layers.0.mlp.experts.5.down_proj"] == 7.0
    assert scales["model.layers.0.block_sparse_moe.experts.2.w1"] == 8.0
    meta = cache.metadata["activation_recache"]
    assert "n_packed_expert_scales_preserved" not in meta


def test_lazy_fp8_fill_does_not_clobber_eager_scales_or_sidecar(tmp_path):
    # F-review majors: (1) a later FP8 render of the same qname must not
    # overwrite the activation scale the eager NVFP4 rung calibrated and
    # ships; (2) a SUBSET gap-fill must merge into packed_expert_max_abs.json,
    # not truncate it to the subset.
    import json
    model = _model()
    calib = _calib()
    cache_dir = tmp_path / "shards"
    cache = _empty_cache({"gptq": True})
    render_format_menu_packed_experts(
        cache, model, calib, ["NVFP4", "FP8_DYNAMIC", "BF16"],
        profile=None, cache_dir=cache_dir, module_token_budget=4096,
    )
    eager_scales = dict(cache.activation_max_abs)
    sidecar = cache_dir / "packed_expert_max_abs.json"
    assert set(json.loads(sidecar.read_text())) == set(EXPERT_NAMES)

    fp8_canon = fr.canonical_format_name("FP8_DYNAMIC")
    subset = {EXPERT_NAMES[0]: fp8_canon}  # one-tensor gap-fill
    torch.manual_seed(99)  # different draw: a clobber would change the scale
    fill_packed_expert_cache_entries(
        cache, model, torch.randint(0, 32, (1, 40)),
        render_assignment=subset,
        levers={}, profile=None,
        cache_dir=cache_dir,
        module_token_budget=4096,
        progress=False,
    )
    for name in EXPERT_NAMES:
        assert cache.activation_max_abs[name] == eager_scales[name]
    assert set(json.loads(sidecar.read_text())) == set(EXPERT_NAMES)


def test_persist_lazy_renders_strips_session_state(tmp_path):
    model = _model()
    cache_dir = tmp_path / "shards"
    pkl_path = tmp_path / "frontier_cache.pkl"
    cache = _empty_cache({"gptq": True})
    render_format_menu_packed_experts(
        cache, model, _calib(), ["NVFP4"],
        profile=None, cache_dir=cache_dir, module_token_budget=4096,
    )
    # Simulate the validator session: dir override + LRU enabled.
    pristine_dir = cache.cache_dir
    cache.relocate(str(tmp_path / "relocated"))
    cache.enable_lru(4 * 1024**3)
    _persist_lazy_expert_renders(
        cache, str(pkl_path), pristine_cache_dir=pristine_dir)
    # Session object keeps its state ...
    assert cache.cache_dir == str(tmp_path / "relocated")
    assert cache._lru_max_bytes == 4 * 1024**3
    # ... but the shared artifact does not inherit it.
    with pkl_path.open("rb") as fh:
        reloaded = pickle.load(fh)
    assert reloaded.cache_dir == pristine_dir
    assert getattr(reloaded, "_lru_max_bytes", 0) == 0
    assert getattr(reloaded, "_lru_order", None) is None


# ---------------------------------------------------------------------------
# module-input clip member selection (Fix C)
# ---------------------------------------------------------------------------

class _PackedOnly(nn.Module):
    def __init__(self, hidden: int = 8, inter: int = 4, n_exp: int = 2):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(n_exp, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.randn(n_exp, hidden, inter))


def test_module_input_member_name_is_order_independent():
    mod = _PackedOnly()
    spec = fr.get_format("NVFP4")
    x = torch.randn(3, 8)  # feature dim == hidden == gate_up in-dim
    for order in (("down_proj", "gate_up_proj"), ("gate_up_proj", "down_proj")):
        plan = _ModulePlan(module=mod, params=[
            _ParamPlan(name=f"mlp.experts.{attr}", attr=attr, spec=spec)
            for attr in order
        ])
        assert _module_input_member_name(plan, x) == "mlp.experts.gate_up_proj"


def test_module_input_member_name_square_tie_breaks_by_role():
    # Degenerate square case intermediate == hidden: both projections match
    # x's feature dim; the tie must break to the non-down projection, not
    # assignment-dict order.
    mod = _PackedOnly(hidden=8, inter=8)
    spec = fr.get_format("NVFP4")
    x = torch.randn(3, 8)
    plan = _ModulePlan(module=mod, params=[
        _ParamPlan(name="mlp.experts.down_proj", attr="down_proj", spec=spec),
        _ParamPlan(name="mlp.experts.gate_up_proj", attr="gate_up_proj",
                   spec=spec),
    ])
    assert _module_input_member_name(plan, x) == "mlp.experts.gate_up_proj"


def test_module_input_member_name_dense_weight_unchanged():
    lin = nn.Linear(8, 4)
    spec = fr.get_format("NVFP4")
    plan = _ModulePlan(module=lin, params=[
        _ParamPlan(name="blk.q_proj", attr="weight", spec=spec)])
    assert _module_input_member_name(plan, torch.randn(3, 8)) == "blk.q_proj"


def test_pre_hook_call_site_uses_module_input_member(tmp_path, monkeypatch):
    # Pins the CALL SITE, not just the helper: the replay/KL pre-hook must
    # derive its act-clip scale via _module_input_member_name (reverting to
    # the old params[0] logic must fail this test). Weight-quant install is
    # stubbed out — only the clip member selection is under test.
    import prismaquant.perturbed_x_cache as pxc

    model = _model().eval()
    # down_proj FIRST in the assignment dict = the order that broke params[0].
    assignment = {
        "mlp.experts.down_proj": "NVFP4",
        "mlp.experts.gate_up_proj": "NVFP4",
    }
    seen: list[str] = []
    real_clip = pxc._maybe_clip_activations

    def recording_clip(x, scales, member_name):
        seen.append(member_name)
        return real_clip(x, scales, member_name)

    monkeypatch.setattr(pxc, "_maybe_clip_activations", recording_clip)
    monkeypatch.setattr(
        pxc.PerturbedActivationCache, "_apply_weight_quant",
        lambda self, plan: None)
    monkeypatch.setattr(
        pxc.PerturbedActivationCache, "_try_install_nvfp4_fused_forward",
        lambda self, plan: False)

    cache = pxc.PerturbedActivationCache(
        model, assignment, tmp_path, cal_hash="m4-test", profile=None)
    cache._activation_scales = {
        "mlp.experts.gate_up_proj": 2.0,
        "mlp.experts.down_proj": 3.5,
    }
    cache.install()
    try:
        with torch.no_grad():
            model(torch.randint(0, 32, (1, 16)))
    finally:
        cache.remove() if hasattr(cache, "remove") else None
    expert_calls = [n for n in seen if n and "experts" in n]
    assert expert_calls, "pre-hook never reached the experts module clip"
    assert all(n == "mlp.experts.gate_up_proj" for n in expert_calls), seen
