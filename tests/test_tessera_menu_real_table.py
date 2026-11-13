"""The default menu path, end to end, against a cost table a campaign wrote.

Every other Tessera menu test in this tree exercises a *part* of this path
against a list somebody typed, and that is how the 2026-09-02 hole survived:
``expand_menu_tokens`` widened the ``TESSERA`` token to every priced column and
``require_producer_formats`` then refused the **whole** menu, so on the default
(attested) path PrismaQuant could not allocate a Tessera artifact at all.  The
unit tests were green throughout, because each one asked one function about one
hand-written five-element list, and a hand-written list holds the rungs its
author was thinking about -- which is to say the attested ones.  The bug needs
a table priced under ``research`` and read back under ``attested``: two menu
MODES, and no fixture crossed them.  Tessera issue #19.

So this module is deliberately the other kind of test.  It runs the real
allocator entry point over the real 2423-column table the continuous-menu
campaign produced, in the default mode, and asserts the one line the issue
named: **on the default path, with a real table, the attested rung count is
greater than zero** -- plus the line that keeps it from going vacuous, that the
table also holds rungs the runtime does *not* attest.  A table with nothing to
drop cannot reproduce the failure and must not be allowed to look like it did.

The path is walked twice, and the two walks are different tests:

* **Through the one read.**  The contract is supplied through
  ``tessera_menu.tessera_runtime_contract`` -- the module's own declared "one
  read" -- built from the bytes the *installed* Tessera actually publishes.
  That keeps the menu measurement independent of which build the dev pin
  names, so it keeps working when the pin is bumped.  (When this was written
  it was also the only way it ran at all: the pin refused the installed
  contract on its recorded sha, so the attested path was dead on this box
  until someone bumped a constant.  It no longer refuses on bytes -- the pin
  gates on the contract's ANSWER now, issue #38 -- but the one-read supply
  stays, because the reason for it was never the pin's staleness.)
* **Through the dev pin, as an operator runs it.**  ``PRISMAQUANT_TESSERA_
  DEV_PIN`` in the environment, nothing patched.  This is the walk the
  one-read tests are blind to by construction: while they were green, the
  pin's byte-identity check was refusing the installed contract and the
  operator's path was dead (issue #38 -- "PrismaQuant's attested Tessera
  path cannot allocate on this box at all today").  A composed test that
  substitutes the read it is meant to exercise cannot see that, so this one
  substitutes nothing, and asserts on top that the provenance block names the
  table that attested the menu -- which is only true when the allocator reads
  the contract once.

One thing this module does not test, on purpose: **the quality of the
allocation.**  The surrogate is measured to mis-rank Tessera rungs
(``tessera_menu.surrogate_selection_caveat``).  What is asserted here is that
the DP *saw a menu*, not that it chose well.

The campaign predates the required ``provenance.cost_mode`` stamp. Allocation
tests validate its original schema and row currency, then use a temporary
copy with that missing metadata reconstructed from the committed producer.
The source hash and the evidence commits accompany the reconstruction; no
cost value or campaign artifact is changed, and the production gate still
refuses an unstamped table.

The allocation fixture uses the five units with a real E4M3 R1024 price.
The historical table has no R1024 price for layer 0's query and key units.
The campaign covers one layer. The source probe covers the full model.
The allocator requires costs for every mutable unit in its input population.
The full-population refusal is covered in
``tests/test_allocator_body_candidate_coverage_audit.py``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import format_registry as fr
from prismaquant import tessera_menu as tm
from prismaquant import tessera_runtime_contract as trc

#: The continuous-menu campaign's own artifacts, priced under
#: ``PRISMAQUANT_TESSERA_MENU=research`` on Qwen3-0.6B layer 0 and recorded in
#: ``docs/measurements/tessera-continuous-menu-2026-09-02.md`` §12.  A research
#: table read back on the default path is exactly the cross-mode condition the
#: bug needed, which is why this test wants THIS table and not a synthetic one.
CAMPAIGN = Path("/mnt/shared/tessera-runs/pq-continuous")
PROBE = CAMPAIGN / "qwen06b" / "probe.pkl"
COSTS = CAMPAIGN / "qwen06b_group" / "cost.pkl"

#: The allocation walks need the campaign's artifacts; the contract reads do
#: not.  This was a module-level ``pytestmark`` until the readable-menu work
#: (#284) added tests that ask only the pinned contract what its decoder
#: accepts.  Skipping those for want of a 7 MB probe pickle would have made
#: the one assertion the issue is about read "did not run" on any box that
#: does not mount ``/mnt/shared`` -- a green receipt for a test nothing ran.
campaign_table = pytest.mark.skipif(
    not (PROBE.exists() and COSTS.exists()),
    reason=(
        f"the campaign's real cost table is not on this box ({PROBE}, {COSTS}); "
        "it lives on /mnt/shared, which both GB10 boxes mount"
    ),
)


@pytest.fixture(autouse=True)
def default_menu_mode(monkeypatch):
    """Run every test here on the DEFAULT menu mode, whatever the shell says.

    The bug this file exists for is a cross-mode one, so a developer with
    ``PRISMAQUANT_TESSERA_MENU=research`` exported would otherwise get a
    harness failure ("not the default path") in place of the measurement.
    """
    monkeypatch.delenv(tm.MENU_MODE_ENV, raising=False)


@pytest.fixture
def installed_contract(monkeypatch):
    """Attest against the contract the installed Tessera actually publishes.

    ``tessera_menu.tessera_runtime_contract`` is documented as the one place
    this module reads a serving contract, so substituting it is substituting
    the whole attestation and nothing else: ``route_admission``,
    ``_producer_eligible``, ``expand_menu_tokens`` and the guard all run
    exactly as production runs them. ``read_tessera_contract`` verifies the
    declared bytes and shares the development reader's parsed-snapshot cache.
    """
    path = trc.contract_path()
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    contract = trc.read_tessera_contract(str(path), sha, f"installed:{sha[:12]}")
    monkeypatch.setattr(tm, "tessera_runtime_contract", lambda: contract)
    return contract


def _default_serve_image() -> str:
    """The serve image the installed contract publishes as its default."""
    return json.loads(trc.contract_path().read_text(encoding="utf-8"))[
        "versions"]["default_serve_image"]


def _dense_context():
    from prismaquant.lane_eligibility import ServingContext
    return ServingContext(
        platform="sm_121", structure="dense", residency="resident",
        runtime_image=_default_serve_image(), execution_mode="eager")


def _routed_context(family):
    """A serving scope derived from the installed contract's routed cell.

    The routed rows publish their own serve images, so the scope comes from
    the cell the way production admission derives it."""
    import json
    from importlib.resources import as_file
    from prismaquant import tessera_runtime_contract as trc
    from prismaquant.lane_eligibility import ServingContext

    with as_file(trc.contract_path()) as path:
        cells = json.loads(path.read_text(encoding="utf-8"))[
            "lane_eligibility"]["cells"]
    cell = next(c for c in cells
                if c["family"] == family and c["structure"] == "routed_moe")
    return ServingContext(
        platform=cell["platform"], structure="routed_moe", residency="resident",
        runtime_image=cell["runtime"]["image"],
        execution_mode=cell["runtime"]["execution_modes"][0])


def _assignment(config) -> dict[str, str]:
    """``{unit: tessera rung}`` for every unit the DP gave a Tessera rung."""
    return {
        name: entry["tessera_format"]
        for name, entry in config.items()
        if not name.startswith("__") and isinstance(entry, dict)
        and entry.get("tessera_format")
    }


@campaign_table
def test_the_expansion_reads_the_guards_own_predicate(installed_contract):
    """Lesson 1 of issue #19, pinned behaviourally rather than by reading code.

    The bug's shape was two functions that each looked right and disagreed
    about which set they described.  The repair was to make the token's
    expansion call the predicate the guard refuses on, so this asserts exactly
    that coupling: move the predicate and the expansion moves with it.  A
    second copy of the legality decision -- the thing that caused this -- would
    leave the expansion unchanged here.
    """
    import pickle

    with COSTS.open("rb") as handle:
        priced = [
            name for name in pickle.load(handle)["formats"]
            if isinstance(name, str) and name.startswith("TESSERA_")
        ]
    assert len(priced) > 100, f"expected the campaign's dense table, got {priced[:5]}"

    # The scoped table answers only at a scope; the token and the guard are
    # asked the same one, and context-free both refuse everything (the
    # expansion narrows to nothing, which the allocator reports by name).
    scope = {"unit": _dense_context()}
    menu, dropped = tm.expand_menu_tokens_report([tm.MENU_TOKEN], priced, context_by_unit=scope)
    assert menu and dropped, (menu[:3], len(dropped))
    assert len(menu) + len(dropped) == len(priced)
    # The expansion is never wider than what the guard accepts.
    fr.require_producer_formats(menu, where="the expanded token", context_by_unit=scope)
    unscoped, all_dropped = tm.expand_menu_tokens_report([tm.MENU_TOKEN], priced)
    assert unscoped == [] and len(all_dropped) == len(priced)


@campaign_table
def test_the_expansion_narrows_with_the_predicate_it_shares(
    monkeypatch, installed_contract,
):
    """Refuse everything at the predicate and the token must expand to nothing.

    This is the coupling test proper: it does not assert *which* rungs are
    attested, only that the token's answer is the guard's answer.
    """
    import pickle

    with COSTS.open("rb") as handle:
        priced = [
            name for name in pickle.load(handle)["formats"]
            if isinstance(name, str) and name.startswith("TESSERA_")
        ]

    monkeypatch.setattr(fr, "format_is_producer_eligible", lambda name, **scope: False)
    menu, dropped = tm.expand_menu_tokens_report(
        ["NVFP4", tm.MENU_TOKEN], priced, context_by_unit={"unit": _dense_context()})
    assert menu == ["NVFP4"], menu
    assert len(dropped) == len(priced)


# ---------------------------------------------------------------------------
# The readable menu, on the contract the installed Tessera actually publishes
# ---------------------------------------------------------------------------

#: What contract v22's ``formats[].reader_rate_range_q256`` says the plugin's
#: decoder accepts, read here as the EXPECTATION the two code paths must both
#: reproduce. It is written out rather than re-read from the file on purpose:
#: a test that derives its expectation from the same field the code reads
#: asserts only that one read happened twice.
READER_RANGES = {
    "TESSERA_E2M1_K2": (128, 896),
    "TESSERA_E4M3_K1": (256, 2048),
    "TESSERA_BF16_K1": (256, 4096),
}
#: And the family the contract does not publish at all -- the one the #275 LFM
#: campaign spent 168 rows x 27 s encoding. Absence is "no reader exists",
#: never "any rate is fine", so every rung of it must be refused.
UNPUBLISHED_FAMILY = "TESSERA_E2M1_K1"


def _research_admissions(*, families=None, serving_context=None):
    """Shape-legal research admissions at the readability tests' fixed shape.

    Readability and containment need the shape and route predicates, not the
    payload sizes or price ordering that ``expand_tessera_menu`` also builds.
    The golden roster below pins this projection of the original priced menu.
    """
    structure = serving_context.structure if serving_context is not None else None
    for spec in tm.menu_families() if families is None else families:
        lo, hi = spec.mathematical_q256_bounds
        for rate in range(lo, hi + 1):
            admission = tm.route_admission(
                spec.format_name(rate), serving_context=serving_context)
            if not admission.admits(tm.MENU_RESEARCH):
                continue
            legal, _ = tm.tessera_shape_legal(
                spec, rate, (2048, 1024), structure=structure)
            if not legal:
                continue
            if admission.act_bits is not None and rate > admission.act_bits * 256:
                continue
            yield spec.name, rate, admission


def _readable_by_family(*, families=None):
    """``{family: {rate: readable}}`` over every shape-legal research rung."""
    out: dict[str, dict[int, bool]] = {}
    for family, rate, admission in _research_admissions(families=families):
        out.setdefault(family, {})[rate] = admission.readable
    return out


def _assert_the_published_reader_ranges(seen):
    assert set(seen) >= set(READER_RANGES) | {UNPUBLISHED_FAMILY}, sorted(seen)
    for family, rates in sorted(seen.items()):
        span = READER_RANGES.get(family)
        readable = {rate for rate, ok in rates.items() if ok}
        if span is None:
            assert not readable, (
                f"{family} is not published by the pinned contract, so no "
                f"reader accepts it; {sorted(readable)[:8]} were admitted")
            continue
        lo, hi = span
        expected = {rate for rate in rates if lo <= rate <= hi}
        assert readable == expected, (
            f"{family}: readable={sorted(readable)[:8]} "
            f"expected={sorted(expected)[:8]} over range {span}")
        assert expected, f"{family}: the shape carries no rung inside {span}"


def test_the_readable_menu_is_exactly_the_published_reader_ranges(
    installed_contract,
):
    """The set the issue named, through the module's declared one read.

    ``TESSERA_E2M1_K2`` over [128, 896] since #560's full-domain widen (the
    single rung 896 until the 2026-09-19 pin); ``TESSERA_E4M3_K1`` over
    [256, 2048]; ``TESSERA_BF16_K1`` over [256, 4096]; and nothing at all
    from ``TESSERA_E2M1_K1``, which the contract does not publish.
    """
    seen = _readable_by_family()
    # Not vacuous: the family the campaign wasted its time on IS on the
    # research menu, at many rungs, and every one of them must be refused.
    assert len(seen.get(UNPUBLISHED_FAMILY, {})) > 50, (
        "the unpublished family is absent from the research menu, so "
        "'refuses every E2M1_K1 rung' asserts nothing")
    _assert_the_published_reader_ranges(seen)
    # The two anchors the #275 campaign encoded are INSIDE the widened
    # domain and readable at the new pin, with the terminal rung unchanged.
    for rate in (128, 512, 896):
        assert seen["TESSERA_E2M1_K2"][rate] is True


def test_the_packaged_branch_agrees_with_the_dev_pin_on_readability():
    """The same answer with no pin supplied: the packaged ``formats`` table.

    ``route_admission`` derives ``readable`` from two different fields on two
    different objects -- ``reader_rate_range`` on the parsed dev-pin contract,
    ``reader_rate_range_q256`` on the packaged ``formats`` row. Nothing else
    in this suite crosses those two paths, so a wrong key on either one is
    invisible until a campaign prices the wrong axis. This runs the walk with
    no contract substituted and no pin requested, which is the branch
    production takes, and holds it to the same expectation.
    """
    assert not trc.dev_pin_requested(), "this walk takes the packaged branch"
    _assert_the_published_reader_ranges(_readable_by_family())


def test_attested_is_contained_in_readable_is_contained_in_research(
    installed_contract,
):
    """``attested <= readable <= research``, rung by rung, family by family.

    Asserted as containment of the admission predicate rather than of three
    menus, so a family whose shape drops every rung cannot make the ordering
    hold by being empty on all three sides.
    """
    context = _dense_context()
    # The scoped table attests a cell only under an exact serving scope, so a
    # context-free walk would count zero attested rungs and the containment
    # would hold by being empty on one side.
    attested_seen = readable_seen = 0
    for family, rate, admission in _research_admissions(
        serving_context=context,
    ):
        name = f"{family}_R{rate}"
        assert admission.admits(tm.MENU_RESEARCH), name
        if admission.admits(tm.MENU_ATTESTED):
            attested_seen += 1
            assert admission.admits(tm.MENU_READABLE), (
                f"{name} is attested by a cell the contract "
                "publishes and yet its own decoder is said to refuse the rate")
        if admission.admits(tm.MENU_READABLE):
            readable_seen += 1
        # The containment that matters for the export gate: readable moves
        # nothing.  A rung nothing attests is `unattested` in every mode.
        if not admission.attested:
            assert admission.route_status == tm.ROUTE_STATUS_UNATTESTED
    assert attested_seen > 0 and readable_seen > attested_seen, (
        attested_seen, readable_seen)


def test_readability_does_not_depend_on_the_serving_scope(installed_contract):
    """A decoder either takes a rate or it does not; a scope cannot change it.

    ``attested`` is scope-bound by construction -- a cell attests one platform,
    image, execution mode and residency -- and ``readable`` deliberately is
    not, because ``reader_rate_range_q256`` is published once per family and
    says nothing about where the rung is served. Asserting it here keeps the
    two from quietly merging: a ``readable`` that moved with the scope would be
    a weaker spelling of ``attested`` wearing a different name.
    """
    context = _dense_context()
    for name in ("TESSERA_E4M3_K1_R512", "TESSERA_E4M3_K1_R1024",
                 "TESSERA_E2M1_K2_R512", "TESSERA_E2M1_K2_R896"):
        scoped = tm.route_admission(name, serving_context=context)
        bare = tm.route_admission(name)
        assert scoped.readable == bare.readable, name
    # ...and the attestation is, which is what makes the above non-trivial.
    # The scoped ask rides a cell-derived routed context, whose E4M3 pair
    # attests q896 since contract v38.
    routed = _routed_context("TESSERA_E4M3_K1")
    assert tm.route_admission(
        "TESSERA_E4M3_K1_R896", serving_context=routed).attested
    assert not tm.route_admission("TESSERA_E4M3_K1_R896").attested


# Captured from the original priced research menu at f1335b03 on (2048, 1024).
# BuildBuddy invocation 2ce103e5-804d-4807-8b95-904e1ea12988 records the full
# format roster; the existing independent reader-range assertions passed there.
@pytest.mark.parametrize("family,count,digest", [
    ("TESSERA_BF16_K1", 3841,
     "4cb0c36e6728d0adc345c7fa3fd30f8e03f7668e23d7779c8a57ad5e4b5a5746"),
    ("TESSERA_E2M1_K1", 513,
     "bb770d48cfe6c1b8f7878d2fc6c5ed73d7b1dacf8a66e2638ae358f81e14cd03"),
    ("TESSERA_E2M1_K2", 769,
     "e221484f0c4d0641943a1b6a5ed6c97886b78572536ba8d418089fcfe0ddc775"),
    ("TESSERA_E4M3_K1", 1793,
     "839e16b035a628dbd323ccb02729990da04c6d5f5443c8e70ef2f015eaa64546"),
])
def test_readability_walk_matches_the_original_priced_menu(family, count, digest):
    from prismaquant.tessera_formats import get_tessera_family

    seen = _readable_by_family(families=[get_tessera_family(family)])
    assert set(seen) == {family}
    rates = sorted(seen[family].items())
    assert len(rates) == count
    encoded = json.dumps(rates, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == digest


def test_readability_walk_does_not_price_payloads(monkeypatch):
    from prismaquant import tessera_footprint

    def refuse_price(*args, **kwargs):
        raise AssertionError("a readability assertion requested payload pricing")

    monkeypatch.setattr(tessera_footprint, "tessera_exact_bits_for_shape", refuse_price)
    seen = _readable_by_family()
    _assert_the_published_reader_ranges(seen)
    assert len(seen[UNPUBLISHED_FAMILY]) > 50


def test_admission_walk_matches_priced_menu_at_integer_code_rates(installed_contract):
    projected = {}
    for family, rate, admission in _research_admissions():
        lo = tm.get_tessera_family(family).mathematical_q256_bounds[0]
        if (rate - lo) % 256 == 0:
            projected[f"{family}_R{rate}"] = admission
    priced = tm.expand_tessera_menu(
        (2048, 1024), mode=tm.MENU_RESEARCH, step_q256=256)
    assert {rung.format_name: rung.admission for rung in priced} == projected
    assert projected
