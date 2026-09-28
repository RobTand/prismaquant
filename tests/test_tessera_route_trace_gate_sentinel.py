"""The route.trace gate's long-list refusals name real modules (RobTand/prismaquant#1620).

When more than eight modules differ, the gate shows the first eight and
summarises the rest with a count, ``"(N further module(s))"``. The two places
that print such a list (``_module_difference`` and ``exact_served_modules``)
looked every shown item up as a module name, including the count line, so a
receipt read::

    (257 further module(s)): served None but the price names no such module

A reader took that for 257 served modules that carry no contract, and the
campaign chased a naming bug that #1490 had already fixed. The count line is
the gate's own text, never a module. It now prints as a count.

Fixtures (``tests/fixtures/tessera_route_trace_1620``, see ``SOURCE.md``): a
cut of the two TR3 rank traces the U4 A8 arm wrote (uniform Tessera-8 on
Tessera v39, TP2) and the A8 export's ``config.json`` trimmed to the same
modules. Every key, module name and contract string is the serve's. The cut
holds 18 named modules per token count: 14 dense and 4 routed-MoE, all
``TESSERA_FP8``.

The gate is fail-closed and this change does not loosen it. The still-bites
tests below refuse for the same reasons before and after:

* a served module rides a different contract than its price
  (``test_a_served_module_on_another_contract_still_refuses``);
* a priced module is not served (``test_a_priced_module_no_serve_dispatched_still_refuses``);
* a served Tessera module has no price
  (``test_a_served_module_with_no_price_still_refuses``).

Fail-before on ``58ae59b4``: the two sentinel tests below fail, because the
count line is looked up as a module and prints as ``served None``.
"""
from __future__ import annotations

import copy
import json
import pathlib

from prismaquant import tessera_route_trace_gate as gate

from test_tessera_route_trace_gate import _contract

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "tessera_route_trace_1620"

FP8 = "fp8_per_token_dynamic"
FP8_DENSE = f"TESSERA_FP8/dense/{FP8}"
FP8_MOE = f"TESSERA_FP8/moe/{FP8}"
BF16 = "bf16_unquantized"


def _config():
    return json.loads((FIXTURE / "config.json").read_text())


def _traces():
    return [(f"rank{rank}",
             json.loads((FIXTURE / f"tr3-rank{rank}.json").read_text()))
            for rank in range(2)]


def _compare(traces, config=None):
    executes, formats = _contract()
    return gate.compare_route_traces(
        traces, expected_ranks=2, config=_config() if config is None else config,
        platform="sm_121", executes_by_platform=executes, formats=formats)


def _entry_m(entry):
    return int(entry["shape"].split(":")[0][1:])


def _drop_entries(traces, *, when):
    """The traces with every entry ``when(entry)`` selects removed."""
    out = []
    for label, trace in traces:
        trace = copy.deepcopy(trace)
        trace["entries"] = [e for e in trace["entries"] if not when(e)]
        out.append((label, trace))
    return out


def _four_module(entry):
    """The entries that name four modules: 12 of the cut's 18 per token count.

    Dropping them leaves 12 priced modules unserved, more than the eight a
    receipt shows, so the count line is exercised.
    """
    return entry["modules"] == 4


def _rewrite_names(traces, change):
    out = []
    for label, trace in traces:
        trace = copy.deepcopy(trace)
        for entry in trace["entries"]:
            names = change(list(entry["module_names"]))
            entry["modules"] -= len(entry["module_names"]) - len(names)
            entry["launches"] = entry["modules"] * (
                entry["launches"] // max(len(entry["module_names"]), 1))
            entry["module_names"] = sorted(names)
        out.append((label, trace))
    return out


def _assert_no_module_lookup_of_the_count(detail):
    """The count line is text, not a module the gate looked up."""
    assert "served None" not in detail, detail
    assert "None but the price" not in detail, detail
    assert "further module(s)): " not in detail, detail


def test_the_unmodified_cut_agrees_on_every_named_module():
    verdict = _compare(_traces())
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    assert verdict["exact_module_qualified"] is True
    assert verdict["served"] == {FP8_DENSE: 14, FP8_MOE: 4}


def test_a_long_price_difference_counts_the_rest_and_looks_up_no_module():
    """12 priced modules with no served dispatch: 8 named, then a count."""
    traces = _drop_entries(_traces(), when=_four_module)
    verdict = _compare(traces)
    assert verdict["status"] == gate.REFUSED, verdict["detail"]
    detail = verdict["detail"]
    assert "the priced and served activation contracts differ per module" in detail
    # The first eight are real modules, named with what they were priced as.
    assert detail.count("served by no module") == 8, detail
    assert "priced TESSERA_FP8/moe/" in detail, detail
    # The remaining four are a count, in the gate's own words.
    assert "(4 further module(s) not shown)" in detail, detail
    _assert_no_module_lookup_of_the_count(detail)


def test_a_long_token_count_difference_counts_the_rest_and_looks_up_no_module():
    """24 module names differ between M=1 and M=2049: 8 shown, then a count.

    The 12 modules of the four-module entries answer to a different name at
    M=2049 only. Every M still dispatches 18 modules on the same contracts, so
    the histogram check passes and the per-name check refuses.
    """
    traces = []
    for label, trace in _traces():
        trace = copy.deepcopy(trace)
        for entry in trace["entries"]:
            if _four_module(entry) and _entry_m(entry) == 2049:
                entry["module_names"] = sorted(
                    n.replace(".model.layers.", ".model.renamed.")
                    for n in entry["module_names"])
        traces.append((label, trace))
    verdict = _compare(traces)
    assert verdict["status"] == gate.REFUSED, verdict["detail"]
    detail = verdict["detail"]
    assert "the served modules differ between token counts" in detail, detail
    # Sorted, the 12 original names come first: each was served at M=1 and
    # by nothing at M=2049. Eight are shown.
    assert detail.count(", M=2049 None") == 8, detail
    assert "(16 further module(s) not shown)" in detail, detail
    _assert_no_module_lookup_of_the_count(detail)


def test_a_short_difference_is_listed_whole_without_a_count():
    """Eight or fewer differing modules need no summary line."""
    def keep_one_short(names):
        return names[:-1] if len(names) == 4 else names

    verdict = _compare(_rewrite_names(_traces(), keep_one_short))
    assert verdict["status"] == gate.REFUSED, verdict["detail"]
    assert "further module(s)" not in verdict["detail"], verdict["detail"]
    assert verdict["detail"].count("served by no module") == 3, verdict["detail"]


def test_a_served_module_on_another_contract_still_refuses():
    """Still bites: one served module rides another contract than its price."""
    traces = _traces()
    target = "language_model.model.layers.0.mlp.down_proj"
    changed = []
    for label, trace in traces:
        trace = copy.deepcopy(trace)
        for entry in trace["entries"]:
            if target in entry["module_names"]:
                assert entry["contract"] == FP8, entry
                entry["policy"] = "TESSERA_BF16:resident"
                entry["contract"] = BF16
        changed.append((label, trace))
    verdict = _compare(changed)
    assert verdict["status"] == gate.REFUSED, verdict["detail"]
    assert target in verdict["detail"], verdict["detail"]
    assert "priced TESSERA_FP8/dense/" in verdict["detail"], verdict["detail"]
    assert "but served TESSERA_BF16/dense/" in verdict["detail"], verdict["detail"]
    assert verdict["exact_module_qualified"] is False


def test_a_priced_module_no_serve_dispatched_still_refuses():
    """Still bites: a priced module is missing from the served trace."""
    target = "language_model.model.layers.0.mlp.gate_up_proj"
    verdict = _compare(_rewrite_names(
        _traces(), lambda names: [n for n in names if n != target]))
    assert verdict["status"] == gate.REFUSED, verdict["detail"]
    assert f"{target} (priced as " in verdict["detail"], verdict["detail"]
    assert "served by no module" in verdict["detail"], verdict["detail"]
    assert verdict["exact_module_qualified"] is False


def test_a_served_module_with_no_price_still_refuses():
    """Still bites: a served Tessera module the artifact never priced."""
    config = _config()
    groups = config["quantization_config"]["config_groups"]
    victim = "model.language_model.layers.0.mlp.down_proj"
    for name, group in list(groups.items()):
        if victim in group["targets"]:
            group["targets"] = [t for t in group["targets"] if t != victim]
            if not group["targets"]:
                del groups[name]
    verdict = _compare(_traces(), config=config)
    assert verdict["status"] == gate.REFUSED, verdict["detail"]
    assert ("language_model.model.layers.0.mlp.down_proj: served "
            in verdict["detail"]), verdict["detail"]
    assert "but the price names no such module" in verdict["detail"], verdict["detail"]
    assert verdict["exact_module_qualified"] is False
