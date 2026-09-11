"""``--stack-sample-sizes``: which per-expert vector the PPS draw follows.

RobTand/prismaquant#495 part 1.  ``sample_stack_groups`` drew proportional to
the probe's ``h_trace_per_expert``, and GLM-5.3-Flash has no probe carrying
one -- so the sampling path was unreachable on the model it was written for
and the campaign paid for a full census instead
(``docs/results/glm_tessera_probe_reduction_regret_2026-09-10.md`` section 2,
and section 3.6's bullet on sizes).  The census's per-expert routed-row counts
are the only other per-expert size that exists.

Two things are pinned:

* the default is unchanged, and unchanged is demonstrated rather than
  asserted: the test rebuilds the pre-change record from the campaign's own
  primitives and compares the two serialisations byte for byte;
* a ``counts`` draw says so.  ``counts`` is a routed-token proxy for
  ``h_trace``, not ``h_trace``, so the record names the source, carries the
  vector and its digest, and suffixes the design -- and the campaign's
  rehydration refuses a record whose declared digest is not the digest of the
  draw it carries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

MODULE = "model.layers.18.feed_forward.experts"
PACKED = f"{MODULE}.gate_up_proj"
EXPERTS = 8
ROLES = ("w1", "w3")


def _dispatch():
    return pytest.importorskip("dispatch_tessera_campaign")


def _campaign():
    return pytest.importorskip("prismaquant.tessera_campaign")


def _profile():
    from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile
    return Lfm2MoeProfile()


def _members():
    return sorted(f"{MODULE}.{expert}.{role}"
                  for expert in range(EXPERTS) for role in ROLES)


def _probe_rows(h):
    return {PACKED: {
        "h_trace": float(sum(h)),
        "h_trace_per_expert": [float(v) for v in h],
        "num_experts": EXPERTS,
        "_packed_experts_module": MODULE,
        "_packed_param": "gate_up_proj",
        "out_features": 8, "in_features": 4,
        "n_params": EXPERTS * 32,
        "router_path": None, "expert_id": None,
    }}


#: Deliberately anti-correlated with the counts below, so a draw that followed
#: the wrong vector cannot accidentally agree with one that followed the right
#: one.
H = [1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0]
COUNTS_PER_EXPERT = [900, 100, 800, 200, 700, 300, 600, 400]


def _census():
    counts = {}
    for expert in range(EXPERTS):
        for role in ROLES:
            # Split the expert's count across its roles; the size is the sum.
            counts[f"{MODULE}.{expert}.{role}"] = COUNTS_PER_EXPERT[expert] // 2
    return {"counts": counts}


def _expected_probe_record(campaign, *, n, seed, audit_rate):
    """The record the pre-#495 code wrote, rebuilt from the same primitives.

    This is the oracle for "the default is unchanged": it is the old function
    body, minus the bookkeeping that surrounds it, so a byte comparison against
    it is a regression check rather than a restatement of the new code.
    """
    frame = campaign.stack_sample_from_probe(
        PACKED, _probe_rows(H)[PACKED], _profile(),
        sampled_experts=range(EXPERTS),
        inclusion_prob={e: 1.0 for e in range(EXPERTS)}, seed=seed,
        design="census")
    draw = campaign.draw_stack_sample(
        {str(e): h for e, h in enumerate(frame.h_trace_per_expert)},
        n, seed=seed, stack=PACKED)
    audit = campaign.audit_subsample(draw["units"], rate=audit_rate, seed=seed,
                                     stack=PACKED)
    return {
        "probe_row": {
            "_packed_experts_module": frame.packed_experts_module,
            "_packed_param": frame.packed_param,
            "num_experts": frame.num_experts,
            "h_trace": frame.stack_h_trace,
            "h_trace_per_expert": list(frame.h_trace_per_expert),
        },
        "sampled_experts": sorted(int(e) for e in draw["units"]),
        "inclusion_prob": dict(draw["inclusion_probability"]),
        "seed": seed,
        "design": draw["method"],
        "draw": draw,
        "audit_experts": sorted(int(e) for e in audit),
    }


def _sample(dispatch, **over):
    kwargs = dict(profile=_profile(), stack_sample=3, seed=5, audit_rate=10)
    kwargs.update(over)
    return dispatch.sample_stack_groups(
        {f"s:{MODULE}": _members()}, _probe_rows(H), **kwargs)


def test_the_default_draw_is_byte_for_byte_what_it_was():
    """No ``sizes`` argument: the same record, serialised identically."""
    dispatch, campaign = _dispatch(), _campaign()
    entry = _sample(dispatch)[f"s:{MODULE}"]
    record = entry["stack_samples"][PACKED]
    expected = _expected_probe_record(campaign, n=3, seed=5, audit_rate=10)
    assert json.dumps(record, sort_keys=True) == json.dumps(
        expected, sort_keys=True)
    # And spelling ``probe`` explicitly is the same run, not a second one.
    explicit = _sample(dispatch, sizes="probe")[f"s:{MODULE}"]
    assert json.dumps(explicit, sort_keys=True) == json.dumps(
        entry, sort_keys=True)
    # The absent-flag record carries no size declaration at all, which is what
    # keeps a plan written today identical to the plans already on disk.
    assert "sizes" not in record


def test_a_counts_draw_follows_the_counts_and_declares_them():
    """The vector that drove the draw is named, carried and digested."""
    dispatch, campaign = _dispatch(), _campaign()
    entry = _sample(dispatch, sizes="counts", census=_census())[f"s:{MODULE}"]
    record = entry["stack_samples"][PACKED]
    assert record["design"] == (record["draw"]["method"]
                                + campaign.STACK_SAMPLE_COUNTS_SUFFIX)
    assert record["design"].endswith("_counts")
    sizes = record["sizes"]
    assert sizes["source"] == "counts"
    assert sizes["sha256"] == record["draw"]["size_sha256"]
    assert sizes["values"] == {str(e): float(COUNTS_PER_EXPERT[e])
                               for e in range(EXPERTS)}
    # The draw followed the counts, not the Fisher vector: with H and
    # COUNTS_PER_EXPERT anti-correlated, the two draws cannot coincide.
    probe_draw = _sample(dispatch)[f"s:{MODULE}"]["stack_samples"][PACKED]
    assert record["sampled_experts"] != probe_draw["sampled_experts"]
    # The Fisher currency is untouched: the stack row still multiplies by the
    # probe's own h_trace, whatever the draw was proportional to.
    assert record["probe_row"]["h_trace_per_expert"] == [float(v) for v in H]


def test_a_counts_draw_survives_the_campaign_rehydration():
    """``selection_stack_samples`` replays it on the declared sizes."""
    dispatch, campaign = _dispatch(), _campaign()
    entry = _sample(dispatch, sizes="counts", census=_census())[f"s:{MODULE}"]
    selection = {"groups": [{"key": f"s:{MODULE}", "members": _members(), **entry}]}
    samples = campaign.selection_stack_samples(selection, _profile())
    sample = samples[PACKED]
    assert sample.design.endswith(campaign.STACK_SAMPLE_COUNTS_SUFFIX)
    assert list(sample.sampled_experts) == entry["stack_samples"][PACKED][
        "sampled_experts"]


def test_a_record_whose_declared_digest_is_not_the_draws_is_refused():
    """The digest is what binds the named vector to the one actually used."""
    dispatch, campaign = _dispatch(), _campaign()
    entry = _sample(dispatch, sizes="counts", census=_census())[f"s:{MODULE}"]
    record = dict(entry["stack_samples"][PACKED])
    record["sizes"] = {**record["sizes"], "sha256": "0" * 64}
    selection = {"groups": [{"key": f"s:{MODULE}", "members": _members(),
                             **{**entry, "stack_samples": {PACKED: record}}}]}
    with pytest.raises(campaign.StackSampleError, match="size digest"):
        campaign.selection_stack_samples(selection, _profile())


def test_counts_without_a_census_is_refused():
    """A size source that names the census needs the census."""
    dispatch = _dispatch()
    with pytest.raises(RuntimeError, match="needs the census"):
        _sample(dispatch, sizes="counts")


def test_a_missing_expert_count_is_refused_rather_than_drawn_at_zero():
    """An absent count would give an expert inclusion probability zero."""
    dispatch = _dispatch()
    census = _census()
    census["counts"].pop(f"{MODULE}.3.w1")
    with pytest.raises(RuntimeError, match="no row count"):
        _sample(dispatch, sizes="counts", census=census)


def test_an_unknown_size_source_is_refused():
    dispatch = _dispatch()
    with pytest.raises(RuntimeError, match="not one of"):
        _sample(dispatch, sizes="h_trace", census=_census())
