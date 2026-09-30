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
    return Glm5NextProfile()


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
