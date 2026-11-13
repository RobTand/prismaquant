"""A stack draw follows the census's per-expert routed-row counts.

RobTand/prismaquant#495 part 1 and #3457. The probe no longer writes the
per-expert Fisher vector, so the census counts are the only draw size. A
draw says so: the record names the source, carries the vector and its digest,
and suffixes the design. The campaign's rehydration refuses a record whose
declared digest is not the digest of its draw, and a record with no counts.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

from prismaquant import tessera_campaign_table as campaign

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

MODULE = "model.layers.18.feed_forward.experts"
PACKED = f"{MODULE}.gate_up_proj"
EXPERTS = 8
ROLES = ("w1", "w3")


def _dispatch():
    return pytest.importorskip("dispatch_tessera_campaign")


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


#: These opposing vectors produce different draws for the fixture's fixed seed.
H = [1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0]
COUNTS_PER_EXPERT = [900, 100, 800, 200, 700, 300, 600, 400]


def _census():
    counts = {}
    for expert in range(EXPERTS):
        for role in ROLES:
            # Split the expert's count across its roles; the size is the sum.
            counts[f"{MODULE}.{expert}.{role}"] = COUNTS_PER_EXPERT[expert] // 2
    return {"counts": counts}


def _sample(dispatch, census=None):
    return dispatch.sample_stack_groups(
        {f"s:{MODULE}": _members()}, _probe_rows(H),
        _census() if census is None else census,
        profile=_profile(), stack_sample=3, seed=5, audit_rate=10)


def test_a_counts_draw_follows_the_counts_and_declares_them():
    """The vector that drove the draw is named, carried and digested."""
    dispatch = _dispatch()
    entry = _sample(dispatch)[f"s:{MODULE}"]
    record = entry["stack_samples"][PACKED]
    assert record["design"] == (record["draw"]["method"]
                                + campaign.STACK_SAMPLE_COUNTS_SUFFIX)
    assert record["design"].endswith("_counts")
    sizes = record["sizes"]
    assert sizes["source"] == "counts"
    assert sizes["sha256"] == record["draw"]["size_sha256"]
    assert sizes["values"] == {str(e): float(COUNTS_PER_EXPERT[e])
                               for e in range(EXPERTS)}
    # This fixture and seed distinguish a counts draw from a Fisher draw.
    fisher = campaign.draw_stack_sample(
        {str(e): h for e, h in enumerate(H)}, 3, seed=5, stack=PACKED)
    assert record["sampled_experts"] != sorted(int(e) for e in fisher["units"])
    # The Fisher currency is untouched: the stack row still multiplies by the
    # probe's own h_trace.
    assert record["probe_row"]["h_trace_per_expert"] == [float(v) for v in H]


def test_a_counts_draw_survives_the_campaign_rehydration():
    """``selection_stack_samples`` replays it on the declared sizes."""
    dispatch = _dispatch()
    entry = _sample(dispatch)[f"s:{MODULE}"]
    selection = {"groups": [{"key": f"s:{MODULE}", "members": _members(), **entry}]}
    samples = campaign.selection_stack_samples(selection, _profile())
    sample = samples[PACKED]
    assert sample.design.endswith(campaign.STACK_SAMPLE_COUNTS_SUFFIX)
    assert list(sample.sampled_experts) == entry["stack_samples"][PACKED][
        "sampled_experts"]


def test_a_record_whose_declared_digest_is_not_the_draws_is_refused():
    """The digest is what binds the named vector to the one actually used."""
    dispatch = _dispatch()
    entry = _sample(dispatch)[f"s:{MODULE}"]
    record = dict(entry["stack_samples"][PACKED])
    record["sizes"] = {**record["sizes"], "sha256": "0" * 64}
    selection = {"groups": [{"key": f"s:{MODULE}", "members": _members(),
                             **{**entry, "stack_samples": {PACKED: record}}}]}
    with pytest.raises(campaign.StackSampleError, match="size digest"):
        campaign.selection_stack_samples(selection, _profile())


def test_a_missing_expert_count_is_refused_rather_than_drawn_at_zero():
    """An absent count would give an expert inclusion probability zero."""
    dispatch = _dispatch()
    census = _census()
    census["counts"].pop(f"{MODULE}.3.w1")
    with pytest.raises(RuntimeError, match="no row count"):
        _sample(dispatch, census)


def test_a_record_without_census_counts_is_refused():
    """A draw proportional to the probe's h_trace is retired."""
    dispatch = _dispatch()
    entry = _sample(dispatch)[f"s:{MODULE}"]
    record = dict(entry["stack_samples"][PACKED])
    del record["sizes"]
    selection = {"groups": [{"key": f"s:{MODULE}", "members": _members(),
                             **{**entry, "stack_samples": {PACKED: record}}}]}
    with pytest.raises(campaign.StackSampleError, match="no census counts"):
        campaign.selection_stack_samples(selection, _profile())
