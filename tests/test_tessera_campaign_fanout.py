"""Fanning the anchor campaign out, and putting it back together.

The campaign is sharded per **fused anchor group**, because that is the
smallest scope whose measured values do not depend on what else the run priced.
Three properties make the split honest, and these tests pin them:

* a selection narrows the checkpoint identity to exactly the units it names, so
  two invocations over disjoint selections can never contend for one journal;
* the scope-wide answers a shard may not re-derive -- the calibration row
  counts, the activation maxima, the draw itself -- are read from a census and
  **checked** against what the shard saw, never adopted on trust;
* the merge reproduces a whole-scope table, and refuses a set of rows that does
  not describe one campaign: a mixed Hessian identity, a gap in the coverage,
  or a unit priced twice.
"""

import copy
import json
import pathlib
import sys

import pytest

from prismaquant import tessera_campaign_rows as campaign_rows
from prismaquant import tessera_campaign_wire as campaign_wire
from prismaquant import tessera_campaign_table as campaign_table

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))


# ---------------------------------------------------------------------------
# The selection
# ---------------------------------------------------------------------------

def _selection(*groups):
    return {
        "schema": "prismaquant.tessera_campaign_units.v1",
        "groups": [{"key": key, "members": list(members)} for key, members in groups],
    }


def test_selection_is_checked_against_the_run_s_own_grouping():
    from prismaquant.tessera_campaign_table import select_anchor_groups

    resolved = {"g:attn": ["q", "k", "v"], "u:down": ["down"]}
    assert select_anchor_groups(
        _selection(("g:attn", ["k", "q", "v"])), resolved, where="w") == ["g:attn"]

    with pytest.raises(RuntimeError, match="does not contain"):
        select_anchor_groups(_selection(("g:mlp", ["w1"])), resolved, where="w")
    # A group whose membership differs here is not the group that was planned:
    # a shard that quietly measured the smaller set would leave the merged
    # table short of rows nothing reported.
    with pytest.raises(RuntimeError, match="has members"):
        select_anchor_groups(_selection(("g:attn", ["q", "k"])), resolved, where="w")
    with pytest.raises(RuntimeError, match="selected twice"):
        select_anchor_groups(
            _selection(("u:down", ["down"]), ("u:down", ["down"])), resolved, where="w")


def test_group_key_puts_fused_siblings_and_a_whole_stack_together():
    from prismaquant.tessera_campaign import anchor_group_key, resolve_anchor_groups

    class Profile:
        def fused_sibling_group(self, name):
            return "layer0.qkv" if name.endswith(("q_proj", "k_proj")) else None

    class Member:
        module_qname = "layers.0.experts"

    members = {"layers.0.experts.3.w1": Member()}
    assert anchor_group_key("layers.0.self_attn.q_proj", profile=Profile(),
                            expert_members={}) == "g:layer0.qkv"
    assert anchor_group_key("layers.0.mlp.down_proj", profile=Profile(),
                            expert_members={}) == "u:layers.0.mlp.down_proj"
    assert anchor_group_key("layers.0.experts.3.w1", profile=Profile(),
                            expert_members=members) == "s:layers.0.experts"

    groups = resolve_anchor_groups(
        ["layers.0.self_attn.k_proj", "layers.0.self_attn.q_proj",
         "layers.0.experts.3.w1"],
        profile=Profile(), expert_members=members)
    assert groups == {
        "g:layer0.qkv": ["layers.0.self_attn.k_proj", "layers.0.self_attn.q_proj"],
        "s:layers.0.experts": ["layers.0.experts.3.w1"],
    }


def test_the_checkpoint_identity_covers_only_the_selected_units(monkeypatch):
    """Disjoint selections are disjoint identities, so they cannot contend."""

    from prismaquant import tessera_campaign as tc

    class Api:
        @staticmethod
        def encoder_source_sha256():
            return "encoder"

        @staticmethod
        def tensor_identity(tensor):
            return {"id": tensor}

    monkeypatch.setattr(campaign_wire, 'require_checkpoint_identity_api', lambda: Api)
    monkeypatch.setattr(tc.th, "encoder_recipe", lambda: {"recipe": 1})
    monkeypatch.setattr(
        "prismaquant.production_weight_cache._production_cache_source_sha256",
        lambda: "package")

    weights = {"a": "wa", "b": "wb", "c": "wc"}
    common = dict(
        acts={n: None for n in weights}, hessians={n: None for n in weights},
        menus={n: [] for n in weights},
        settings=dict(model="m", layer_stride=1, out="o", cache_dir="c",
                                checkpoint="k", deadline_seconds=0.0, units="path",
                                calibration_census=None, census_out=None,
                                seed_checkpoint=None, seed_wire_dir=None),
        calibration_identity={"text_sha256": "t"}, serving_scope=None,
        static_scales={}, static_scale_policy="policy")

    whole = tc.campaign_checkpoint_identity(weights=weights, **common)
    left = tc.campaign_checkpoint_identity(
        weights={"a": "wa"}, **{**common, "acts": {"a": None},
                                "hessians": {"a": None}, "menus": {"a": []}})
    right = tc.campaign_checkpoint_identity(
        weights={"b": "wb", "c": "wc"},
        **{**common, "acts": {"b": None, "c": None},
           "hessians": {"b": None, "c": None}, "menus": {"b": [], "c": []}})

    assert set(whole["units"]) == {"a", "b", "c"}
    assert set(left["units"]) == {"a"}
    assert set(right["units"]) == {"b", "c"}
    assert not set(left["units"]) & set(right["units"])
    # The selection file's PATH is not bound: it is a location, and its content
    # is already bound by the units map above.
    assert "units" not in left["settings"]
    assert left["settings"] == right["settings"] == whole["settings"]


def test_streaming_residency_knobs_are_scheduling_not_identity(monkeypatch):
    """Cache slots, prefetch workers and the free-memory floor change no receipt;
    the capture policy and the streaming switch do."""
    import argparse
    from prismaquant import tessera_campaign as tc

    class Api:
        @staticmethod
        def encoder_source_sha256():
            return "encoder"

        @staticmethod
        def tensor_identity(tensor):
            return {"id": tensor}

    monkeypatch.setattr(campaign_wire, 'require_checkpoint_identity_api', lambda: Api)
    monkeypatch.setattr(tc.th, "encoder_recipe", lambda: {"recipe": 1})
    monkeypatch.setattr(
        "prismaquant.production_weight_cache._production_cache_source_sha256",
        lambda: "package")

    def identity(**overrides):
        args = argparse.Namespace(
            model="m", layer_stride=1, out="o", cache_dir="c", checkpoint="k",
            deadline_seconds=0.0, units="path", calibration_census=None,
            census_out=None, seed_checkpoint=None, seed_wire_dir=None,
            streaming=True, streaming_capture_policy="legacy",
            streaming_cache_headroom_gb=24.0, streaming_cache_slots=2,
            streaming_prefetch_workers=1, anchor_batch_size=8)
        for name, value in overrides.items():
            setattr(args, name, value)
        return tc.campaign_checkpoint_identity(
            weights={"a": "wa"}, acts={"a": None}, hessians={"a": None},
            menus={"a": []}, settings=vars(args), calibration_identity={"text_sha256": "t"},
            serving_scope=None, static_scales={}, static_scale_policy="policy")

    base = identity()
    for name in ("streaming_cache_headroom_gb", "streaming_cache_slots",
                 "streaming_prefetch_workers", "anchor_batch_size"):
        assert name not in base["settings"]
    assert identity(streaming_cache_headroom_gb=12.0, streaming_cache_slots=6,
                    streaming_prefetch_workers=4, anchor_batch_size=32) == base
    assert identity(streaming_capture_policy="strict") != base
    assert identity(streaming=False) != base


def test_source_identity_cache_flags_are_not_checkpoint_identity(monkeypatch):
    """PQ #1532: ``--source-identity-cache`` and its sha256 name a proof file's
    location and digest, never a measurement input.  The identity is
    byte-identical with and without them, so a tree containing #1520 resumes a
    journal an older tree wrote."""
    import argparse
    import json
    from prismaquant import tessera_campaign as tc

    class Api:
        @staticmethod
        def encoder_source_sha256():
            return "encoder"

        @staticmethod
        def tensor_identity(tensor):
            return {"id": tensor}

    monkeypatch.setattr(campaign_wire, 'require_checkpoint_identity_api', lambda: Api)
    monkeypatch.setattr(tc.th, "encoder_recipe", lambda: {"recipe": 1})
    monkeypatch.setattr(
        "prismaquant.production_weight_cache._production_cache_source_sha256",
        lambda: "package")

    def identity(**overrides):
        args = argparse.Namespace(
            model="m", layer_stride=1, out="o", cache_dir="c", checkpoint="k",
            deadline_seconds=0.0, units="path", calibration_census=None,
            census_out=None, seed_checkpoint=None, seed_wire_dir=None,
            streaming=True, streaming_capture_policy="legacy")
        for name, value in overrides.items():
            setattr(args, name, value)
        return tc.campaign_checkpoint_identity(
            weights={"a": "wa"}, acts={"a": None}, hessians={"a": None},
            menus={"a": []}, settings=vars(args), calibration_identity={"text_sha256": "t"},
            serving_scope=None, static_scales={}, static_scale_policy="policy")

    without = json.dumps(identity(), sort_keys=True)
    with_flags = json.dumps(identity(source_identity_cache="/pool/source-identity.json",
                                     source_identity_cache_sha256="ab" * 32),
                            sort_keys=True)
    assert with_flags == without
    assert "source_identity_cache" not in with_flags


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------

def _census(counts, maxima=None):
    maxima = maxima or {name: 1.0 for name in counts}
    return {"counts": dict(counts), "max_abs": dict(maxima)}


def test_census_supplies_the_scope_s_counts_and_verifies_the_run_s_own():
    from prismaquant.tessera_campaign import census_max_abs, census_token_counts

    census = _census({"a": 16384, "b": 16384, "e": 512})
    # Without a census a run reports what it saw; with one it reports the
    # scope's, which is what makes a shard's fit_tokens the monolith's.
    assert census_token_counts(None, {"a": 16384, "b": 16384}) == (16384, 16384)
    assert census_token_counts(census, {"a": 16384}) == (16384, 512)
    assert census_max_abs(census, {"a": 1.0}) == {"a": 1.0, "b": 1.0, "e": 1.0}

    with pytest.raises(RuntimeError, match="does not cover"):
        census_token_counts(census, {"z": 1})
    with pytest.raises(RuntimeError, match="disagrees"):
        census_token_counts(census, {"a": 8192})
    with pytest.raises(RuntimeError, match="disagrees"):
        census_max_abs(census, {"a": 2.0})


def test_a_derived_census_names_its_base_census_s_draw(tmp_path):
    """The GLM MTP census covers units the body census never ran, over the
    body's draw. Its capture identity carries the body's (max, min) rows, so
    the pair is read off the hash-bound base census it names, and the run's
    own rows are still checked against the derived census's counts."""
    import hashlib
    from prismaquant.glm_mtp_capture import CENSUS_EXTENSION_SCHEMA
    from prismaquant.tessera_campaign import census_token_counts

    draw = {"text_sha256": "corpus", "fit_ids_sha256": "ids",
            "nsamples": 512, "seqlen": 512, "seed": 0}
    base = dict(_census({"a": 262144, "b": 13}), **draw)
    base_path = tmp_path / "census.json"
    base_path.write_text(json.dumps(base))
    ref = {"path": str(base_path), "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest()}
    derived = dict(_census({"m": 261632, "n": 722}), **draw,
                   mtp_extension={"schema": CENSUS_EXTENSION_SCHEMA, "layer": 45,
                                  "base_census": ref})
    assert census_token_counts(derived, {}) == (262144, 13)
    assert census_token_counts(derived, {"m": 261632}) == (262144, 13)
    with pytest.raises(RuntimeError, match="disagrees"):
        census_token_counts(derived, {"m": 262144})
    with pytest.raises(RuntimeError, match="does not cover"):
        census_token_counts(derived, {"a": 262144})

    # The base is bound by its bytes, and must be the same draw.
    with pytest.raises(RuntimeError, match="sha256"):
        census_token_counts(dict(derived, mtp_extension=dict(
            derived["mtp_extension"], base_census=dict(ref, sha256="0" * 64))), {})
    other = dict(base, fit_ids_sha256="other")
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps(other))
    with pytest.raises(RuntimeError, match="another draw"):
        census_token_counts(dict(derived, mtp_extension=dict(
            derived["mtp_extension"], base_census={
                "path": str(other_path),
                "sha256": hashlib.sha256(other_path.read_bytes()).hexdigest()})), {})
    with pytest.raises(RuntimeError, match="extension"):
        census_token_counts(dict(derived, mtp_extension=dict(
            derived["mtp_extension"], schema="another")), {})


def test_a_census_of_another_draw_is_refused():
    from prismaquant.tessera_campaign import require_census_draw

    identity = {"text_sha256": "corpus", "fit_ids_sha256": "ids"}
    require_census_draw({"text_sha256": "corpus", "fit_ids_sha256": "ids"},
                        identity, where="w")
    with pytest.raises(RuntimeError, match="different draws"):
        require_census_draw({"text_sha256": "corpus", "fit_ids_sha256": "other"},
                            identity, where="w")


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------

HESSIAN = {
    "supplied": True, "text_sha": "ids", "token_count": 16384,
    "text_sha256": "corpus", "fit_ids_sha256": "ids", "fit_tokens": 16384,
    "capture_sha256": "shard-digest", "kwarg": ["hessian"], "applied": True,
}

SCOPE = {
    "dense_targets": ["a", "b"], "expert_targets": [], "dense_all": ["a", "b"],
    "pinned": [], "declared_stacks": {}, "packed_in_scope": {},
    "packed_outside_layer_stride": {},
    "anchor_groups": {"u:a": ["a"], "u:b": ["b"]},
    "calibration_census": {"counts": {"a": 16384, "b": 16384},
                           "token_count": 16384, "token_count_min": 16384},
}


def _shard(row_key, unit):
    return {
        "schema": "prismaquant.tessera_campaign_cost.v1",
        "currency": "output_mse_under_route_activation_contract",
        "costs": {unit: {"E4M3_R512": {"output_mse": 1.5, "output_mse_measured": True,
                                       "hessian_identity": dict(HESSIAN)}}},
        "formats": ["E4M3_R512"],
        "leave_one_anchor_out": {unit: {"e4m3": {"max_abs_log2_error": 0.1}}},
        "non_interpolable": [],
        "menu_sizes": {unit: 4},
        "anchor_counts": {unit: {"e4m3": 3}},
        "provenance": {
            "menu_mode": "research", "tp_degree": 1, "model": "m", "nsamples": 32,
            "seqlen": 512, "max_act_rows": 512, "layer_stride": 1000,
            "anchors_round_one": 3, "max_rounds": 0, "anchor_budget": 12,
            "loo_gate": 0.25, "max_artifact_bpp": 8.0, "cost_mode": "render-score",
            "rounds_run": 2, "stopped_early": False, "wall_seconds": 10.0,
            "surfaces": {unit: {"e4m3": {"anchors": 3}}},
            "anchor_groups": {row_key: [unit]},
            "seed_checkpoint": None,
            "unit_selection": {"schema": "prismaquant.tessera_campaign_units.v1",
                               "selected": True,
                               "groups": [{"key": row_key, "members": [unit]}]},
            "campaign_scope": copy.deepcopy(SCOPE),
            "activation_static_scales": {"policy": "p", "path": None,
                                         "units": {"a": 1.0, "b": 2.0}},
            "hessian": {"supplied": True, "calibration_identity": {"text_sha256": "corpus"},
                        "capture_sha256": "shard-digest"},
        },
    }


def _payloads():
    return {"row-0000": _shard("u:a", "a"), "row-0001": _shard("u:b", "b")}


def test_merge_reproduces_a_whole_scope_table():
    merged = campaign_table.merge_payloads(
        _payloads(), census={"counts": {"a": 16384, "b": 16384}},
        capture_sha256="merged-digest")

    assert sorted(merged["costs"]) == ["a", "b"]
    assert merged["formats"] == ["E4M3_R512"]
    assert sorted(merged["leave_one_anchor_out"]) == ["a", "b"]
    assert sorted(merged["provenance"]["surfaces"]) == ["a", "b"]
    # One Hessian identity, and it is the merged capture's -- the digest of the
    # union, not either shard's digest of its own half.
    digests = {row["hessian_identity"]["capture_sha256"]
               for rows in merged["costs"].values() for row in rows.values()}
    assert digests == {"merged-digest"}
    assert merged["provenance"]["hessian"]["capture_sha256"] == "merged-digest"
    # The merged table claims the whole scope, not a shard's selection.
    assert merged["provenance"]["unit_selection"]["selected"] is False
    # Under the key the ALLOCATION reads.  The campaign and the allocation
    # share one constant for it; a literal here would let the merge write a
    # population block that nothing consumes while the reference row's own
    # narrow block stayed in place under the real key.
    from prismaquant.tessera_expert_projection import POPULATION_KEY
    assert merged["provenance"][POPULATION_KEY]["priced"]["dense"] == ["a", "b"]
    assert "tessera_population" not in merged["provenance"]
    assert merged["provenance"]["campaign_fanout"]["rows"] == {
        "row-0000": ["u:a"], "row-0001": ["u:b"]}


def test_merge_refuses_a_mixed_hessian_identity():
    payloads = _payloads()
    row = payloads["row-0001"]["costs"]["b"]["E4M3_R512"]
    row["hessian_identity"] = {**row["hessian_identity"], "fit_ids_sha256": "other"}
    with pytest.raises(campaign_table.MergeRefused, match="fit_ids_sha256"):
        campaign_table.merge_payloads(payloads, census={"counts": {}},
                                capture_sha256="merged-digest")


def test_merge_refuses_a_gap_and_a_double_price():
    payloads = _payloads()
    del payloads["row-0001"]
    with pytest.raises(campaign_table.MergeRefused, match="do not cover"):
        campaign_table.merge_payloads(payloads, census={"counts": {}},
                                capture_sha256="merged-digest")

    payloads = _payloads()
    payloads["row-0001"]["provenance"]["unit_selection"]["groups"] = [
        {"key": "u:a", "members": ["a"]}]
    with pytest.raises(campaign_table.MergeRefused, match="priced by both"):
        campaign_table.merge_payloads(payloads, census={"counts": {}},
                                capture_sha256="merged-digest")


def test_merge_refuses_rows_from_two_calibrations():
    payloads = _payloads()
    payloads["row-0001"]["provenance"]["nsamples"] = 64
    with pytest.raises(campaign_table.MergeRefused, match="provenance.nsamples"):
        campaign_table.merge_payloads(payloads, census={"counts": {}},
                                capture_sha256="merged-digest")


def test_a_row_may_not_carry_the_round_loop_s_own_deadline(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "model": "m", "cwd": "/tmp", "python": "py", "env": {},
        "campaign_argv": ["--menu-mode", "research", "--deadline-seconds", "100"]}))
    with pytest.raises(RuntimeError, match="deadline"):
        campaign_rows.load_spec(spec)


def test_merge_refuses_a_journal_that_names_a_shard_it_does_not_have(tmp_path):
    """A row whose anchors would silently vanish from the merged journal.

    ``merge_checkpoint`` reads one unit envelope per journal entry.  Skipping
    an entry whose file is absent loses exactly the rows a seeded run adopts
    and never re-encodes, and loses them without a word.
    """

    row = tmp_path / "row-0000"
    (row / "cost.anchors.json.parts").mkdir(parents=True)
    (row / "cost.anchors.json").write_text(json.dumps({
        "schema": "prismaquant.cost_stage_checkpoint.manifest.v1",
        "stage": "Tessera campaign", "identity_sha256": "x",
        "identity": {"units": {"a": {}}},
        "units": [{"qname": "a", "file": "a.pkl"}],
    }))
    with pytest.raises(campaign_table.MergeRefused, match="is not there"):
        campaign_table.merge_checkpoint({"row-0000": str(row)},
                                  tmp_path / "cost.anchors.json")


def test_receipts_accept_a_memoized_row_and_refuse_a_failed_campaign(tmp_path):
    """A re-submitted row reports no launcher status of its own.

    Re-running the manifest IS the resume, so the second submit's rows come
    back memoized with ``rc`` unset.  Those are done.  What is not done is a
    campaign whose own verdict was non-zero.
    """

    (tmp_path / "receipts.json").write_text(json.dumps({
        "returncode": 0,
        "rows": [{"key": "a", "status": "cache_hit", "host": "-", "rc": "-"},
                 {"key": "b", "status": "executed", "host": "sparky", "rc": "0"}],
    }))
    campaign_table.require_campaign_receipts(tmp_path, 2)

    (tmp_path / "receipts.json").write_text(json.dumps({
        "returncode": 75,
        "rows": [{"key": "a", "status": "waiting", "host": "-", "rc": "-"},
                 {"key": "b", "status": "executed", "host": "sparky", "rc": "0"}],
    }))
    with pytest.raises(campaign_table.MergeRefused, match="not every row is done"):
        campaign_table.require_campaign_receipts(tmp_path, 2)


# ---------------------------------------------------------------------------
# Rows the pinned reader cannot decode: kept as evidence, never as a price.


def _unservable_row(fmt="TESSERA_E2M1_K1_R512"):
    return {"anchor": {"qname": "a", "format_name": fmt, "family": "e2m1_k1",
                       "body_rate_q256": 512, "dloss": 3.0},
            "wire_record": {"file": "a." + fmt + ".wire"},
            "adopted_from": "seed checkpoint /x/cost.anchors.json"}


def test_merge_unions_the_unservable_evidence_rather_than_copying_one_row_s():
    payloads = _payloads()
    payloads["row-0000"]["provenance"]["unservable"] = {
        "a": {"TESSERA_E2M1_K1_R512": _unservable_row()}}
    other = _unservable_row("TESSERA_E2M1_K2_R128")
    other["anchor"]["qname"] = "b"
    payloads["row-0001"]["provenance"]["unservable"] = {
        "b": {"TESSERA_E2M1_K2_R128": other}}

    merged = campaign_table.merge_payloads(
        payloads, census={"counts": {"a": 16384, "b": 16384}},
        capture_sha256="merged-digest")

    # Both rows' evidence survives.  The reference row's block alone would be
    # the same shape and half the content, which is why this is a union and
    # not the copy the rest of the reference provenance is.
    assert sorted(merged["provenance"]["unservable"]) == ["a", "b"]
    # And none of it reached a price.
    assert sorted(merged["costs"]) == ["a", "b"]
    assert merged["formats"] == ["E4M3_R512"]
    assert all("E2M1" not in fmt
               for rows in merged["costs"].values() for fmt in rows)


def test_merge_refuses_two_rows_that_disagree_about_one_unservable_row():
    payloads = _payloads()
    mine = _unservable_row()
    theirs = copy.deepcopy(mine)
    theirs["anchor"]["dloss"] = 9.0
    payloads["row-0000"]["provenance"]["unservable"] = {
        "a": {"TESSERA_E2M1_K1_R512": mine}}
    payloads["row-0001"]["provenance"]["unservable"] = {
        "a": {"TESSERA_E2M1_K1_R512": theirs}}

    with pytest.raises(campaign_table.MergeRefused, match="unservable evidence"):
        campaign_table.merge_payloads(
            payloads, census={"counts": {"a": 16384, "b": 16384}},
            capture_sha256="merged-digest")


@pytest.mark.parametrize("malformed", [False, True])
def test_a_seed_links_only_the_wire_bytes_its_adopter_will_price(tmp_path, malformed):
    """The export intake reads the wire directory, so evidence stays out of it.

    An adopted row outside this run's menu is a measurement, not a price. Its
    blob must not appear where the export leg looks for priced bytes, and the
    only place that decision can be made is at the link, before the adopter
    ever sees the state.
    """
    from prismaquant.cost_stage_checkpoint import write_unit
    from prismaquant import tessera_campaign as campaign

    seed = tmp_path / "seed"
    (seed / "cache" / "wire").mkdir(parents=True)
    priced_filename = campaign_wire.wire_path(seed / "cache" / "wire", "a", "ON").name
    receipt_filename = "other-coordinate.tessera" if malformed else priced_filename
    for name in (receipt_filename, "evidence.wire"):
        (seed / "cache" / "wire" / name).write_bytes(b"x")
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    seed_inputs = {'currency': 'output_mse', 'calibration': {},
        'input_global_scale_policy': 'fixture',
        'units': {'a': {'scoring_rows': {'sha256': 'rows'}, 'input_global_scale': 1.0}}}
    seed_sha = canonical_json_sha256(seed_inputs, where='seed fixture')
    parts = seed / "cost.anchors.json.parts"
    parts.mkdir()
    state = {
        "anchors": [
            {"qname": "a", "format_name": "ON", "family": "f", "dloss": 1.0},
            {"qname": "a", "format_name": "OFF", "family": "g", "dloss": 2.0},
        ],
        "wire_records": {"ON": {"file": receipt_filename},
                         "OFF": {"file": "evidence.wire"}},
    }
    write_unit(parts, stage="Tessera campaign", qname="a",
               identity_sha256=seed_sha, state=state)
    (seed / "cost.anchors.json").write_text(json.dumps({"identity_sha256": seed_sha, "identity": seed_inputs}))

    wire_dir = tmp_path / "run-wire"
    wire_dir.mkdir()
    seen = {}
    def adopt_seed():
        return campaign._adopt_seed_checkpoint(
            seed / "cost.anchors.json", None, targets=["a"], wire_dir=wire_dir,
            adopt=lambda name, state, where: seen.update({name: state}),
            admits=lambda name, fmt: fmt == "ON",
            identity_sha256="run-identity", expected_identity=seed_inputs)

    if malformed:
        with pytest.raises(RuntimeError, match="wire filename differs"):
            adopt_seed()
        assert list(wire_dir.iterdir()) == []
        assert seen == {}
        assert (seed / "cache" / "wire" / receipt_filename).read_bytes() == b"x"
        return
    adopt_seed()

    assert sorted(p.name for p in wire_dir.iterdir()) == [priced_filename]
    # The adopter still gets the whole state: deciding what to do with the
    # row it will not price is the adopter's job, not the linker's.
    assert sorted(record["format_name"] for record in seen["a"]["anchors"]) == \
        ["OFF", "ON"]


# ---------------------------------------------------------------------------
# The admissible partition
#
# One row wider than the worker is a demand to report, not a reason to refuse
# the rows that fit: on the GLM plan the routed stacks derive 124.783 GiB
# against a 104 GiB box and used to block the 90 rows that fit with them
# (RobTand/prismaquant#391).
# ---------------------------------------------------------------------------

#: Column counts whose derived demand is exact: ``in**2 * 4`` bytes of fp32
#: Hessian is 1 GiB at 16384 and 64 GiB at 131072, and the retained scoring
#: rows add the fraction that makes the ceiling 2 GB and 65 GB.
_NARROW_COLUMNS = 16384
_WIDE_COLUMNS = 131072


def _plan_args(spec, workspace, **overrides):
    import types

    args = dict(spec=spec, workspace=workspace, calibration_cache=None,
                groups_per_row=1, rows_per_box=2, timeout_s=300,
                stack_sample=None, stack_sample_seed=0, audit_rate=10,
                probe=None, seed_checkpoint=None, seed_wire_dir=None)
    args.update(overrides)
    return types.SimpleNamespace(**args)


# ---------------------------------------------------------------------------
# The expert sample
#
# A routed stack is priced from a subset of its experts because pricing all of
# them is ~250 box-days at GLM scale.  What makes a subset an estimate rather
# than a guess is that the draw is proportional to a per-expert importance the
# probe already knows, and that the inclusion probability it was drawn under
# travels with the prices.  These pin the draw's arithmetic, not its taste.
# ---------------------------------------------------------------------------

def _sizes(*values):
    return {f"e{index}": float(value) for index, value in enumerate(values)}


def test_the_draw_is_fixed_size_and_deterministic_from_its_seed():
    from prismaquant.tessera_campaign_table import draw_stack_sample

    sizes = _sizes(*[1.0 + index for index in range(32)])
    first = draw_stack_sample(sizes, 8, seed=0, stack="s:x|w1")
    again = draw_stack_sample(sizes, 8, seed=0, stack="s:x|w1")
    other = draw_stack_sample(sizes, 8, seed=1, stack="s:x|w1")
    assert len(first["units"]) == 8
    assert first["units"] == again["units"]
    # A different seed is allowed to draw the same set, but not by
    # construction: over a 32-unit frame these two must differ.
    assert first["units"] != other["units"]
    # And a different stack draws independently under one seed.
    assert first["units"] != draw_stack_sample(
        sizes, 8, seed=0, stack="s:x|w2")["units"]


def test_inclusion_probabilities_sum_to_the_sample_size():
    from prismaquant.tessera_campaign_table import draw_stack_sample

    # No unit dominates, so nothing lands in the certainty stratum and every
    # pi is strictly interior.  sum(pi) == n is the identity that makes the
    # Horvitz-Thompson estimator unbiased; if it does not hold, nothing
    # downstream is an estimate of anything.
    sizes = _sizes(*[1.0 + 0.1 * index for index in range(32)])
    draw = draw_stack_sample(sizes, 8, seed=7, stack="s:y|w1")
    total = sum(draw["inclusion_probability"].values())
    assert abs(total - 8.0) < 1e-9
    assert not draw["certainty"]
    assert all(0.0 < value < 1.0
               for value in draw["inclusion_probability"].values())


def test_a_dominant_unit_is_taken_with_certainty_not_clipped():
    from prismaquant.tessera_campaign_table import draw_stack_sample

    # One expert holding most of the stack's h_trace would want pi > 1 under a
    # plain proportional rule.  Clipping it to 1 and leaving the others alone
    # would make sum(pi) < n and bias the estimate low; the certainty stratum
    # is the correction, and this is the case that tells them apart.
    sizes = _sizes(1000.0, *[1.0] * 31)
    draw = draw_stack_sample(sizes, 4, seed=0, stack="s:z|w1")
    assert draw["certainty"] == ["e0"]
    assert draw["inclusion_probability"]["e0"] == 1.0
    assert "e0" in draw["units"]
    assert len(draw["units"]) == 4
    assert abs(sum(draw["inclusion_probability"].values()) - 4.0) < 1e-9


def test_asking_for_the_whole_frame_is_a_census_not_a_sample():
    from prismaquant.tessera_campaign_table import draw_stack_sample

    sizes = _sizes(1.0, 2.0, 3.0)
    draw = draw_stack_sample(sizes, 8, seed=0, stack="s:z|w1")
    assert draw["method"] == "census"
    assert draw["units"] == ["e0", "e1", "e2"]
    assert set(draw["inclusion_probability"].values()) == {1.0}


def test_a_zero_importance_expert_gets_probability_zero_and_is_never_drawn():
    from prismaquant.tessera_campaign_table import draw_stack_sample

    # Zero h_trace is exact, not small: under the campaign's Fisher convention
    # a token never routed to an expert contributes zero gradient, so that
    # expert's term in the stack total is zero and encoding it would buy
    # nothing. A negative weight is not a size at all and refuses.
    draw = draw_stack_sample(_sizes(1.0, 0.0, 1.1, 1.2, 1.3, 1.4), 3,
                             seed=0, stack="s:z|w1")
    assert draw["inclusion_probability"]["e1"] == 0.0
    assert "e1" not in draw["units"] and draw["zero_size"] == ["e1"]
    assert abs(sum(draw["inclusion_probability"].values()) - 3.0) < 1e-9
    with pytest.raises(RuntimeError, match="negative"):
        draw_stack_sample(_sizes(1.0, -1.0), 1, seed=0, stack="s:z|w1")


def test_a_single_random_draw_is_refused_because_it_has_no_error_bar():
    from prismaquant.tessera_campaign_table import draw_stack_sample

    # One random draw admits no variance estimate. Stamping 0.0 for it would
    # read downstream as "measured exactly", so the refusal happens here --
    # before any GPU second is spent on a sample nothing can bound.
    with pytest.raises(RuntimeError, match="one randomly drawn expert"):
        draw_stack_sample(_sizes(*[1.0 + index for index in range(32)]), 1,
                          seed=0, stack="s:z|w1")


def test_the_draw_stamps_what_a_variance_estimate_will_need():
    from prismaquant.tessera_campaign_table import draw_stack_sample

    # The permutation and the systematic start re-derive the sample
    # arithmetically, so a reader checks the draw rather than trusting a PRNG
    # to have been stable. The randomized order is not cosmetic: the practical
    # variance estimator for this design is only justified under it.
    draw = draw_stack_sample(_sizes(*[1.0 + index for index in range(32)]), 8,
                             seed=3, stack="s:z|w1")
    assert draw["method"] == "randomized_systematic_pps_with_take_all_v1"
    assert 0.0 <= draw["start"] < 1.0
    assert sorted(draw["permutation"]) == sorted(
        name for name, value in draw["inclusion_probability"].items()
        if 0.0 < value < 1.0)
    assert draw["permutation"] != sorted(draw["permutation"])
    assert draw["random_draws"] + len(draw["certainty"]) == 8
    assert len(draw["size_sha256"]) == 64


def test_the_audit_subsample_is_a_subset_that_cannot_move_the_estimate():
    from prismaquant.tessera_campaign_table import draw_stack_sample
    from prismaquant.tessera_campaign_table import audit_subsample

    sizes = _sizes(*[1.0 + index for index in range(32)])
    draw = draw_stack_sample(sizes, 20, seed=0, stack="s:a|w1")
    audit = audit_subsample(draw["units"], rate=10, seed=0, stack="s:a|w1")
    assert len(audit) == 2 and set(audit) <= set(draw["units"])
    assert audit == audit_subsample(draw["units"], rate=10, seed=0,
                                    stack="s:a|w1")
    # Changing the audit fraction must not disturb the draw the prices are
    # built from -- it runs on a stream of its own.
    wider = audit_subsample(draw["units"], rate=4, seed=0, stack="s:a|w1")
    assert len(wider) == 5
    assert draw_stack_sample(sizes, 20, seed=0, stack="s:a|w1")["units"] \
        == draw["units"]
    assert audit_subsample(["e00"], rate=10, seed=0, stack="s") == ["e00"]
    assert audit_subsample([], rate=10, seed=0, stack="s") == []


def test_a_v1_selection_that_samples_is_refused_by_its_own_schema(tmp_path):
    from prismaquant.tessera_campaign_selection import load_unit_selection

    path = tmp_path / "units.json"
    path.write_text(json.dumps({
        "schema": "prismaquant.tessera_campaign_units.v1",
        "model": "m", "layer_stride": 1,
        "groups": [{"key": "s:a", "members": ["a", "b"], "sampled": ["a"]}],
    }))
    with pytest.raises(RuntimeError, match="must say so in its schema"):
        load_unit_selection(path)


def test_a_sample_without_inclusion_probabilities_is_refused(tmp_path):
    from prismaquant.tessera_campaign_selection import load_unit_selection

    path = tmp_path / "units.json"
    path.write_text(json.dumps({
        "schema": "prismaquant.tessera_campaign_units.v2",
        "model": "m", "layer_stride": 1,
        "groups": [{"key": "s:a", "members": ["a", "b"], "sampled": ["a"]}],
    }))
    with pytest.raises(RuntimeError, match="inclusion probability"):
        load_unit_selection(path)


def test_the_sample_narrows_what_is_priced_but_not_what_the_group_is(tmp_path):
    from prismaquant.tessera_campaign_table import select_anchor_groups
    from prismaquant.tessera_campaign_selection import load_unit_selection, selection_priced_units

    path = tmp_path / "units.json"
    path.write_text(json.dumps({
        "schema": "prismaquant.tessera_campaign_units.v2",
        "model": "m", "layer_stride": 1,
        "groups": [{"key": "s:a", "members": ["a", "b", "c"],
                    "sampled": ["a", "c"], "audit": ["c"],
                    "inclusion_probability": {"a": 0.5, "c": 0.5}}],
    }))
    selection = load_unit_selection(path)
    # The membership check still sees the WHOLE stack: a sampled run must not
    # be able to pass a check a full run would fail.
    assert select_anchor_groups(selection, {"s:a": ["a", "b", "c"]},
                                where="t") == ["s:a"]
    priced, audit, pi = selection_priced_units(selection)
    assert priced == {"a", "c"} and audit == {"c"} and pi == {"a": 0.5, "c": 0.5}


# ---------------------------------------------------------------------------
# The rate band
# ---------------------------------------------------------------------------

def test_the_band_places_two_anchors_at_its_ends_not_across_the_range():
    from prismaquant.tessera_campaign import round_one_rates

    allowed = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]

    def snap(rate, allowed):
        return min(allowed, key=lambda r: (abs(int(r) - int(rate)), int(r)))

    # Unset: the historical schedule over the whole realisable range.
    assert round_one_rates(allowed, band=None, anchors=3,
                           snap=snap) == [256, 1024, 2048]
    # Set: the ends of the band, and nothing outside it.
    assert round_one_rates(allowed, band=(700, 1300), anchors=3,
                           snap=snap) == [768, 1280]
    # A family whose rungs do not reach the band says so by measuring nothing
    # there rather than by being priced at a rate outside it.
    assert round_one_rates(allowed, band=(4000, 5000), anchors=3,
                           snap=snap) == []


def test_an_explicit_exhaustive_band_uses_each_legal_rung_only_in_that_band():
    from prismaquant.tessera_campaign import round_one_rates

    allowed = [832, 833, 835, 896, 1024]

    def snap(rate, allowed):
        return min(allowed, key=lambda rung: (abs(int(rung) - int(rate)), int(rung)))

    assert round_one_rates(allowed, band=(832, 896), anchors=2, snap=snap,
                           exhaustive_band=True) == [832, 833, 835, 896]
    with pytest.raises(RuntimeError, match="requires --rate-band"):
        round_one_rates(allowed, band=None, anchors=2, snap=snap,
                        exhaustive_band=True)


def test_exhaustive_grid_requires_the_research_menu():
    from prismaquant.tessera_campaign import require_exhaustive_rate_research_mode
    from prismaquant.tessera_menu import MENU_ATTESTED, MENU_READABLE, MENU_RESEARCH

    for mode in (MENU_ATTESTED, MENU_READABLE):
        with pytest.raises(RuntimeError, match="requires --menu-mode research"):
            require_exhaustive_rate_research_mode(True, mode)
        require_exhaustive_rate_research_mode(False, mode)
    require_exhaustive_rate_research_mode(True, MENU_RESEARCH)


def test_the_audit_anchor_lands_inside_the_bracket_or_nowhere():
    from prismaquant.tessera_campaign import audit_extra_rate

    allowed = [256, 512, 768, 1024, 1280]

    def snap(rate, allowed):
        return min(allowed, key=lambda r: (abs(int(r) - int(rate)), int(r)))

    inner = audit_extra_rate(allowed, [512, 1280], snap=snap)
    assert inner is not None and 512 < inner < 1280
    # Adjacent rungs have no interior, so there is no third anchor to place.
    assert audit_extra_rate(allowed, [512, 768], snap=snap) is None
    assert audit_extra_rate(allowed, [512], snap=snap) is None


def test_a_malformed_band_is_refused():
    from prismaquant.tessera_campaign_table import parse_rate_band

    assert parse_rate_band(None) is None and parse_rate_band("") is None
    assert parse_rate_band("256,896") == (256, 896)
    for bad in ("256", "a,b", "896,256", "0,896", "1,2,3"):
        with pytest.raises(RuntimeError):
            parse_rate_band(bad)


# ---------------------------------------------------------------------------
# The empty menu
# ---------------------------------------------------------------------------

def test_units_with_no_admitted_rung_are_named_not_dropped(capsys):
    from prismaquant.tessera_campaign import report_empty_menus

    empty = report_empty_menus({"a": [object()], "b": [], "c": []},
                               mode="readable")
    assert empty == ["b", "c"]
    printed = capsys.readouterr().out
    assert "menu_sizes[b] = 0" in printed and "mode=readable" in printed
    assert report_empty_menus({"a": [object()]}, mode="research") == []
