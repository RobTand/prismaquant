"""A Tessera census over a scoped roster of profile-pinned Linears (PQ #1843).

GLM-5.3 attention is pinned in ``glm5_next``, and a global unpin would change the roster of
every GLM re-run from main. ``--allow-pinned`` lifts named pins for one campaign (the
allocator's grammar), ``--pinned-roster-only`` keeps only what it lifts, and the census
records the lift so a run under another lift refuses it. Unset, nothing changes.
"""
import argparse
import json

import pytest

from prismaquant import tessera_campaign as tc
from prismaquant.model_profiles.glm5_next import Glm5NextProfile

P = "model.language_model.layers."
KDA = ("q_proj", "k_proj", "v_proj", "b_proj", "forget_gate.f_a_proj", "forget_gate.f_b_proj",
       "g_a_proj", "g_b_proj", "o_proj")
MLA = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
INDEXER = ("indexer.wq_b", "indexer.wk", "indexer.weights_proj")
#: every attention Linear except the DSA indexer (no AURA cotangent: PQ #1842)
ATTENTION = ("self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.b_proj,f_a_proj,"
             "f_b_proj,self_attn.g_a_proj,self_attn.g_b_proj,self_attn.o_proj,self_attn.q_a_proj,"
             "self_attn.q_b_proj,self_attn.kv_a_proj_with_mqa,self_attn.kv_b_proj")


def _names():
    """A GLM-shaped Linear roster: dense KDA layer 0, MLA layer 3 with the indexer, KDA layer 4."""
    kda0 = [f"{P}0.self_attn.{leaf}" for leaf in KDA]
    dense0 = [f"{P}0.mlp.{r}" for r in ("gate_proj", "up_proj", "down_proj")]
    mla3 = [f"{P}3.self_attn.{leaf}" for leaf in MLA]
    idx3 = [f"{P}3.self_attn.{leaf}" for leaf in INDEXER]
    moe3 = [f"{P}3.mlp.gate", *(f"{P}3.mlp.shared_experts.{r}" for r in ("gate_proj", "up_proj", "down_proj"))]
    kda4 = [f"{P}4.self_attn.{leaf}" for leaf in KDA]
    head = ["lm_head", "model.language_model.embed_tokens"]
    return {"kda": kda0 + kda4, "mla": mla3, "indexer": idx3, "dense": dense0,
            "shared": moe3[1:], "router": moe3[:1], "head": head,
            "all": kda0 + dense0 + mla3 + idx3 + moe3 + kda4 + head}


@pytest.fixture(scope="module")
def profile():
    result = Glm5NextProfile()
    result._declare_config_document({"text_config": {"layer_types": ["linear_attention"] * 3
                                                   + ["deepseek_sparse_attention", "linear_attention"]}})
    return result


def test_unset_roster_is_the_body_campaign_roster(profile):
    n = _names()
    roster = tc.campaign_roster(n["all"], profile)
    assert set(roster.dense) == set(n["dense"] + n["shared"])
    assert set(roster.pinned) == set(n["kda"] + n["mla"] + n["indexer"] + n["router"])
    assert roster.lifted == ()


def test_pinned_roster_only_prices_attention_and_keeps_the_indexer_pinned(profile):
    n = _names()
    roster = tc.campaign_roster(n["all"], profile, allow_pinned=ATTENTION, pinned_roster_only=True)
    assert set(roster.dense) == set(roster.lifted) == set(n["kda"] + n["mla"])
    assert set(roster.pinned) == set(n["indexer"] + n["router"])
    # every member of a runtime-fused module is lifted with its siblings
    for layer, group in ((0, "in_proj_qkvbfg_a"), (3, "fused_qkv_a_proj")):
        members = [name for name in n["all"]
                   if profile.fused_sibling_group(name) == f"{P}{layer}.self_attn.{group}"]
        assert members and set(members) <= set(roster.lifted)


def test_allow_pinned_without_only_adds_attention_to_the_body(profile):
    n = _names()
    roster = tc.campaign_roster(n["all"], profile, allow_pinned=ATTENTION)
    assert set(roster.dense) == set(n["dense"] + n["shared"] + n["kda"] + n["mla"])
    assert set(roster.lifted) == set(n["kda"] + n["mla"])


def test_a_token_that_lifts_nothing_refuses(profile):
    with pytest.raises(ValueError, match="lift no profile-pinned Linear.*self_attn.qq_proj"):
        tc.campaign_roster(_names()["all"], profile, allow_pinned=ATTENTION + ",self_attn.qq_proj",
                           pinned_roster_only=True)
    # an unpinned name is not a lift either: the token must name a pin
    with pytest.raises(ValueError, match="shared_experts.down_proj"):
        tc.campaign_roster(_names()["all"], profile, allow_pinned="shared_experts.down_proj")


def test_pinned_roster_only_requires_a_lift(profile, tmp_path, capsys):
    with pytest.raises(ValueError, match="requires --allow-pinned"):
        tc.campaign_roster(_names()["all"], profile, pinned_roster_only=True)
    with pytest.raises(SystemExit) as exit_info:
        tc.main(["--pinned-roster-only", "--model", "m", "--out", str(tmp_path / "o.pkl"),
                 "--cache-dir", str(tmp_path)])
    assert exit_info.value.code == 2
    assert "--pinned-roster-only requires --allow-pinned" in capsys.readouterr().err


def _census(tmp_path, args, roster, name="census.json"):
    payload = tc.calibration_census(
        {"u": 3}, {"u": 1.0}, args=args, groups={"u": ["u"]}, dense_targets=["u"],
        expert_targets=[], shapes={"u": (2, 2)}, identity={"text_sha256": "t", "fit_ids_sha256": "f"},
        pinned_roster=tc.pinned_roster_block(args, roster))
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path, payload


def _args(**kw):
    return argparse.Namespace(model="m", nsamples=4, seqlen=8, seed=0, layer_stride=1, **kw)


def test_census_records_the_lift_and_refuses_another(tmp_path, profile):
    roster = tc.campaign_roster(_names()["all"], profile, allow_pinned=ATTENTION, pinned_roster_only=True)
    lifted = _args(allow_pinned=ATTENTION, pinned_roster_only=True)
    path, payload = _census(tmp_path, lifted, roster)
    block = payload["pinned_roster"]
    assert block["schema"] == tc.PINNED_ROSTER_SCHEMA
    assert block["pinned_roster_only"] is True
    assert block["lifted"] == sorted(roster.lifted)
    assert tc.load_calibration_census(path, args=lifted)["counts"] == {"u": 3}
    for other in (_args(), _args(allow_pinned=ATTENTION, pinned_roster_only=False),
                  _args(allow_pinned="self_attn.o_proj", pinned_roster_only=True)):
        with pytest.raises(RuntimeError, match="must be the same scope"):
            tc.load_calibration_census(path, args=other)


def test_a_body_census_carries_no_block_and_refuses_a_lifted_run(tmp_path, profile):
    roster = tc.campaign_roster(_names()["all"], profile)
    path, payload = _census(tmp_path, _args(), roster)
    assert "pinned_roster" not in payload
    tc.load_calibration_census(path, args=_args())
    tc.load_calibration_census(path, args=argparse.Namespace(  # a caller with no such flags
        model="m", nsamples=4, seqlen=8, seed=0, layer_stride=1))
    with pytest.raises(RuntimeError, match="must be the same scope"):
        tc.load_calibration_census(path, args=_args(allow_pinned=ATTENTION, pinned_roster_only=True))


def test_unset_flags_leave_the_checkpoint_identity_unchanged(monkeypatch):
    class Api:
        @staticmethod
        def encoder_source_sha256():
            return "encoder"

        @staticmethod
        def tensor_identity(tensor):
            return {"id": tensor}

    monkeypatch.setattr(tc, "_checkpoint_identity_api", lambda: Api)
    monkeypatch.setattr(tc.th, "encoder_recipe", lambda: {"recipe": 1})
    monkeypatch.setattr("prismaquant.production_weight_cache._production_cache_source_sha256",
                        lambda: "package")
    base = dict(model="m", layer_stride=1, out="o", cache_dir="c", checkpoint="k",
                deadline_seconds=0.0, units=None, calibration_census=None, census_out=None,
                seed_checkpoint=None, seed_wire_dir=None)

    def identity(**extra):
        return tc._campaign_checkpoint_identity(
            weights={"a": "wa"}, acts={"a": None}, hessians={"a": None}, menus={"a": []},
            args=argparse.Namespace(**base, **extra), calibration_identity={"text_sha256": "t"},
            serving_scope=None, static_scales={}, static_scale_policy="policy")

    before = identity()
    assert identity(allow_pinned=None, pinned_roster_only=False) == before
    lifted = identity(allow_pinned=ATTENTION, pinned_roster_only=True)
    assert lifted["settings"]["allow_pinned"] == ATTENTION
    assert lifted["settings"]["pinned_roster_only"] is True
    assert lifted != before


def test_sparse_query_does_not_share_a_kda_menu():
    profile = Glm5NextProfile()
    profile._declare_config_document({"text_config": {"layer_types": ["deepseek_sparse_attention"],
                                                    "q_lora_rank": None}})
    query = P + "0.self_attn.q_proj"
    assert profile.fused_sibling_group(query) is None
    assert tc.resolve_anchor_groups([query], profile=profile, expert_members={}) == {
        "u:" + query: [query]}


def test_explicit_roster_covers_every_commissioned_projection(profile):
    names = _names()
    visual = ["model.visual.blocks.0.attn.qkv", "model.visual.blocks.0.attn.proj",
              "model.visual.blocks.0.mlp.gate_proj", "model.visual.blocks.0.mlp.up_proj",
              "model.visual.blocks.0.mlp.down_proj", "model.visual.merger.proj",
              "model.visual.merger.gate_proj", "model.visual.merger.up_proj",
              "model.visual.merger.down_proj"]
    selected = names["kda"] + names["mla"] + names["indexer"] + names["router"] + visual
    roster = tc.campaign_roster(names["all"] + visual, profile,
                               allow_pinned=','.join(selected), pinned_roster_only=True)
    assert set(roster.dense) == set(roster.lifted) == set(selected)


def test_real_router_uses_the_existing_dense_capture_without_a_model_change():
    from types import SimpleNamespace
    import torch
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter
    router = Glm5NextTextTopkRouter(SimpleNamespace(num_experts_per_tok=2, num_local_experts=4,
        hidden_size=4, routed_scaling_factor=1.0, n_group=1, topk_group=1, norm_topk_prob=True))
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.mlp = torch.nn.Module()
    model.model.mlp.gate = router
    name = "model.mlp.gate"
    profile = Glm5NextProfile()
    assert profile.campaign_dense_unit_names(model) == []
    assert profile.campaign_dense_unit_names(model, allow_pinned=name) == [name]
    from prismaquant.cost_streaming import StreamedCausalLM
    source = object.__new__(StreamedCausalLM)
    source.model, source.profile, source.dtype = model, profile, router.weight.dtype
    source.context = SimpleNamespace(buffer_dtypes={name + ".weight": router.weight.dtype})
    assert source.selected_weight_specs([name]) == {
        name: ((4, 4), router.weight.dtype, router.weight.numel() * router.weight.element_size())}
    x = torch.arange(8).reshape(2, 4).bfloat16()
    original_weight = router.weight.detach().clone()
    before = router(x)
    rows, hessian, counts, maxima = tc._collect_activations(
        model, [name], [x], 2, "cpu", want_hessian=True, forward_batch=router)
    assert torch.equal(rows[name], x.float())
    assert torch.equal(hessian[name], x.float().t() @ x.float())
    assert counts[name] == 2 and maxima[name] == 7.0
    after = router(x)
    assert before[0].dtype == after[0].dtype == torch.float32
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert torch.equal(router.weight, original_weight)


def test_direct_consumer_prices_charge_the_final_buffer_once(profile):
    from prismaquant.tessera_formats import get_tessera_family
    family = get_tessera_family("TESSERA_E4M3_K1")
    prefix = P + "3.self_attn."
    assert tc._direct_consumer_memory_bytes(prefix + "indexer.weights_proj", family, (32, 64), profile=profile) == 32 * 64 * 4
    assert tc._direct_consumer_memory_bytes(prefix + "indexer.wk", family, (64, 64), profile=profile) == 0
    assert tc._direct_consumer_memory_bytes(prefix + "kv_b_proj", family, (128, 64), profile=profile) == 128 * 64 * 2


def test_t8_head_screen_does_not_quantize_its_fp32_input(profile):
    import torch
    from prismaquant.production_weight_cache import _local_forward_render_score
    prefix = P + "3.self_attn.indexer."
    head = tc._prepare_anchor(qname=prefix + "weights_proj", format_name="TESSERA_E4M3_K1_R1024",
        activation_kwargs_for=None, hessian_required=False, static_input_scale=None, structure="dense",
        profile=profile)
    key = tc._prepare_anchor(qname=prefix + "wk", format_name="TESSERA_E4M3_K1_R1024",
        activation_kwargs_for=None, hessian_required=False, static_input_scale=None, structure="dense",
        profile=profile)
    x = torch.linspace(0.013, 1.073, 32).reshape(1, 32)
    weight = torch.eye(32)
    head_score = _local_forward_render_score(reference_weight=weight, rendered_weight=weight,
        activations=x, activation_quantize=head["activation_qdq"], activation_max_abs=None)
    key_score = _local_forward_render_score(reference_weight=weight, rendered_weight=weight,
        activations=x, activation_quantize=key["activation_qdq"], activation_max_abs=None)
    assert head_score == (0.0, "output_mse", False, False)
    assert key_score[0] > 0 and key_score[2] is True


def test_explicit_fp32_cache_keeps_the_direct_head_values():
    import torch
    from prismaquant.production_weight_cache import _store_rendered_weight_entry
    value = torch.tensor([[0.1234567, 1.0034567]], dtype=torch.float32)
    weights = {}
    kwargs = dict(weights=weights, cache_dir_path=None, qname=P + "3.self_attn.indexer.weights_proj",
                  fmt="TESSERA_E4M3_K1_R1024", tensor=value, weight_dtype=torch.float32)
    _store_rendered_weight_entry(**kwargs, preserve_fp32=True)
    retained = next(iter(weights.values()))
    assert retained.dtype == torch.float32 and torch.equal(retained, value)
    _store_rendered_weight_entry(**kwargs)
    retained = next(iter(weights.values()))
    assert retained.dtype == torch.bfloat16 and torch.equal(retained, value.bfloat16())


@pytest.mark.parametrize("format_name, route_family", [
    ("TESSERA_E4M3_K1_R1024", "TESSERA_FP8"),
    ("TESSERA_BF16_K1_R1792", "TESSERA_BF16")])
def test_head_wire_decoder_cache_and_screen_agree(tmp_path, profile, format_name, route_family):
    import torch
    from prismaquant.production_weight_cache import ProductionWeightCache, _local_forward_render_score
    from tessera.serving.projection_routes import direct_consumer_weight
    name = P + "3.self_attn.indexer.weights_proj"
    source = torch.randn(32, 32, generator=torch.Generator().manual_seed(17)).bfloat16()
    original = source.clone()
    inputs = torch.randn(2, 32, generator=torch.Generator().manual_seed(19)).bfloat16()
    cache_dir, wire_dir = tmp_path / "cache", tmp_path / "wire"
    cache_dir.mkdir()
    wire_dir.mkdir()
    cache = ProductionWeightCache(weights={}, levers={}, cache_dir=str(cache_dir), metadata={})
    anchor = tc._measure_anchor(qname=name, weight=source, activations=inputs, format_name=format_name,
        cache=cache, wire_dir=wire_dir, hessian_required=False, structure="dense", profile=profile)
    blob = next(wire_dir.glob("*.tessera")).read_bytes()
    decoded = direct_consumer_weight(blob, P + "3.self_attn.indexer.wk_weights_proj",
                                    "weights_proj", route_family)
    retained = torch.load(cache_dir / cache.weights[(name, format_name)], weights_only=True)
    assert decoded.dtype == retained.dtype == torch.float32
    assert torch.equal(decoded, retained)
    score = _local_forward_render_score(reference_weight=source, rendered_weight=retained,
        activations=inputs, activation_quantize=lambda x: x, activation_max_abs=None)
    assert anchor.dloss == score[0]
    assert anchor.activation_contract == "a32" and anchor.activation_quantized is False
    assert anchor.wire_bytes == len(blob)
    assert torch.equal(source, original)


def test_direct_consumer_contract_refuses_a_missing_profile():
    for leaf in ("kv_b_proj", "indexer.weights_proj"):
        name = P + "3.self_attn." + leaf
        with pytest.raises(tc.ActivationScaleContractError, match="profile"):
            tc._direct_consumer_activation_contract(name, profile=None)


def test_unrelated_names_keep_a_missing_profile_meaning_no_contract():
    assert tc._direct_consumer_activation_contract(P + "3.self_attn.q_proj", profile=None) is None
    assert tc._direct_consumer_activation_contract(P + "3.self_attn.indexer.wk", profile=None) is None
    assert tc._direct_consumer_activation_contract("lm_head", profile=None) is None


def test_measure_anchor_direct_leaf_without_profile_refuses_before_encode(tmp_path):
    import torch
    from prismaquant.production_weight_cache import ProductionWeightCache
    name = P + "3.self_attn.indexer.weights_proj"
    weight = torch.randn(32, 32, dtype=torch.bfloat16)
    inputs = torch.randn(2, 32, dtype=torch.bfloat16)
    cache_dir, wire_dir = tmp_path / "cache", tmp_path / "wire"
    cache_dir.mkdir()
    wire_dir.mkdir()
    cache = ProductionWeightCache(weights={}, levers={}, cache_dir=str(cache_dir), metadata={})
    with pytest.raises(tc.ActivationScaleContractError, match="profile"):
        tc._measure_anchor(qname=name, weight=weight, activations=inputs,
            format_name="TESSERA_E4M3_K1_R1024", cache=cache, wire_dir=wire_dir,
            hessian_required=False, structure="dense")
    assert list(wire_dir.glob("*.tessera")) == []
    assert cache.weights == {}


def _explicit_select_model():
    """A GLM-shaped tree: quantizable body plus every commissioned family."""
    import torch
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter
    from types import SimpleNamespace
    model = torch.nn.Module()

    def attach(dotted, module):
        target = model
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = getattr(target, part, None)
            if child is None:
                child = torch.nn.Module()
                setattr(target, part, child)
            target = child
        setattr(target, parts[-1], module)
        return dotted

    linear = lambda rows=8, cols=8: torch.nn.Linear(cols, rows, bias=False)
    dense0 = [attach(f"{P}0.mlp.{leaf}", linear())
              for leaf in ("gate_proj", "up_proj", "down_proj")]
    kda0 = [attach(f"{P}0.self_attn.{leaf}", linear()) for leaf in KDA]
    kda0.append(attach(f"{P}0.self_attn.f_a_proj", linear()))
    mla3 = [attach(f"{P}3.self_attn.{leaf}", linear()) for leaf in MLA]
    idx3 = [attach(f"{P}3.self_attn.{leaf}", linear()) for leaf in INDEXER]
    shared3 = [attach(f"{P}3.mlp.shared_experts.{leaf}", linear())
               for leaf in ("gate_proj", "up_proj", "down_proj")]
    router = Glm5NextTextTopkRouter(SimpleNamespace(num_experts_per_tok=2, num_local_experts=4,
        hidden_size=4, routed_scaling_factor=1.0, n_group=1, topk_group=1, norm_topk_prob=True))
    router_name = attach(f"{P}3.mlp.gate", router)
    kda4 = [attach(f"{P}4.self_attn.{leaf}", linear()) for leaf in ("q_proj", "o_proj")]
    visual = [attach(name, linear()) for name in (
        "model.visual.blocks.0.attn.proj", "model.visual.blocks.0.mlp.down_proj",
        "model.visual.merger.proj")]
    head = attach("lm_head", linear())
    attach(f"{P}0.self_attn.conv1d", torch.nn.Conv1d(4, 8, 4))
    head_mtp = attach(f"{P}45.eh_proj", linear())
    commissioned = kda0 + mla3 + idx3 + [router_name] + kda4 + visual + [head_mtp]
    quantizable = dense0 + shared3
    return model, quantizable, commissioned, head


def test_default_enumeration_holds_every_commissioned_unit_for_a_token():
    model, quantizable, commissioned, head = _explicit_select_model()
    profile = Glm5NextProfile()
    assert set(profile.campaign_dense_unit_names(model)) == set(quantizable + [head])


def test_explicit_token_enumerates_each_commissioned_family():
    model, quantizable, commissioned, head = _explicit_select_model()
    profile = Glm5NextProfile()
    tokens = ",".join(commissioned)
    assert set(profile.campaign_dense_unit_names(model, allow_pinned=tokens)) == set(
        quantizable + commissioned + [head])
    partial = profile.campaign_dense_unit_names(model, allow_pinned="indexer.weights_proj")
    linears = [name for name in commissioned if not name.endswith(".mlp.gate")]
    assert set(partial) == set(quantizable + linears + [head])


def test_default_roster_from_default_enumeration_keeps_the_body_set():
    model, quantizable, commissioned, head = _explicit_select_model()
    profile = Glm5NextProfile()
    roster = tc.campaign_roster(profile.campaign_dense_unit_names(model), profile)
    assert set(roster.dense) == set(quantizable)
    assert roster.lifted == ()
    assert set(profile.campaign_dense_unit_names(model)) == set(
        profile.campaign_dense_unit_names(model))
