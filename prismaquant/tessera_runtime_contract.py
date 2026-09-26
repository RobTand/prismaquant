"""Read the Tessera serving plugin's own packaged runtime contract.

Principle 14 in one file: every statement PrismaQuant makes about what a
Tessera artifact *serves* -- which family, at which rate, on which route,
behind which serve flags, at which tensor-parallel world size -- is read from
``tessera/serving/runtime_contract.json``, the table the plugin packages and
publishes, or it is refused.  Nothing here decides anything; it parses.

Since Tessera contract v7 that table also publishes what the plugin *LOADS*
(``native_extensions``: which CUDA library, under which filename pattern,
matched by which named rule, and what runs instead when it is absent), which
is the other §7.4 fact PrismaQuant used to maintain on this side by hand.
:func:`require_pin_native_extensions_match_contract` is the refusal that keeps
the serving pin a transcription of it.

The cell grammar has one home in ``lane_eligibility``. This reader adds the
development answer pin, format ranges, native-extension identity and fused
module licence; export uses the same parsed cells under the separate serving
pin. Since lane schema v4, residency scopes each cell and ``executes`` names
its launches. Both readers retain those fields and reject malformed or
overlapping claims through the shared parser. Neither accepts a Gridbook
schema; that serving lane was retired on 2026-09-02.

**The dev pin.** This reader provides an explicit development answer pin.
Serving and export separately require the exact commit and packaged contract
digest in ``tessera_serving_runtime_pin``; that gate does not require a release
tag. The development override is explicit:

* ``PRISMAQUANT_TESSERA_DEV_PIN=<anything non-empty>`` opts in.  The value is
  recorded verbatim in provenance as what the operator asked for; it is not
  the gate.
* the installed contract's **answer** -- every value the ADMISSION gate
  reads, in the vocabulary of :func:`contract_answer` -- must equal
  :data:`TESSERA_DEV_PIN_ANSWER` or the read raises, with a field-level diff
  naming what moved.
* unset disables this development reader: it returns ``None`` without reading
  a contract. This does not replace the separate serving/export pin checks.

There is no third state.  A mismatch never degrades to "unattested" -- that
would turn a stale pin into a silently empty menu, which is the failure mode
this whole file exists to prevent.

**Why the answer and not the bytes** (issue #38).  This pin used to compare
the environment's value against an exact commit and the file's sha256 against
a recorded one.  Both legs fired on *identity*, and the thing they name is an
editable checkout on the same box, so every Tessera commit that touched the
contract -- a prose ``detail``, a changelog paragraph, a ``contract_version``
bump -- turned PrismaQuant's attested path off and a fistful of tests red in a
repo nobody was editing, while the two rungs' meaning had not moved at all.
That is principle 14 read backwards: prose fields explain, they are never a
value a gate reads, so a prose edit is not a thing to re-review.  The gate now
reads the answer.  A commit that moves no answer passes silently; a commit
that moves one refuses and says which field, which is a review prompt rather
than a corruption warning.  :data:`TESSERA_DEV_PIN_COMMIT` and
:data:`TESSERA_DEV_PIN_CONTRACT_SHA256` survive as the *record of the review*
-- the build and the bytes a human read when the answer was accepted -- and
travel into provenance alongside the bytes this run actually read, so
prose-only drift is visible without being fatal.

This is deliberately weaker than an exact contract-byte pin: it admits any
Tessera whose table answers identically, which is exactly the claim
this development reader makes about it. The serving pin independently requires
the reviewed contract bytes, whether or not the pinned version is a release.

The contract's own identity (commit, sha, path, schema, contract_version)
travels into every allocation's provenance as ``tessera_dev_pin`` so a
shipcard records which table admitted its units.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .lane_eligibility import (
    CellEvidence,
    _DIGEST_IMAGE,
    SCOPED_LANE_SCHEMAS,
    cell_evidence_admits,
    cell_lane_admits,
    LANE_ELIGIBILITY_SCHEMA_TESSERA,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V4,
    LaneClaim,
    LaneEligibilityError,
    ServingContext,
    QUALIFICATION_DEVICE_QUALIFIED,
    ROUTE_STATUS_BACKED,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
    _parse_table,
    cell_matches_serving_context,
    parse_lane_claim,
)
from .tessera_serving_runtime_pin import native_extension_contract_row

__all__ = [
    "FUSED_MODULE_FIELD_LICENCES",
    "FUSED_MODULE_SCHEMA",
    "FusedModuleLicence",
    "MATCH_BASENAME_FNMATCH",
    "TESSERA_CONTRACT_SCHEMA",
    "TESSERA_DEV_PIN_COMMIT",
    "TESSERA_DEV_PIN_ANSWER",
    "TESSERA_DEV_PIN_CONTRACT_SHA256",
    "TESSERA_DEV_PIN_ENV",
    "TESSERA_LANE_SCHEMA",
    "TesseraContract",
    "TesseraContractError",
    "TesseraNativeExtension",
    "TesseraRouteCell",
    "cell_activation_projection",
    "ActivationQuantizerAttestation",
    "ActivationQuantizerGeneration",
    "ActivationQuantizerVector",
    "ACTIVATION_QUANTIZER_SCHEMA",
    "ACTIVATION_QUANTIZER_SCHEMAS",
    "ACTIVATION_QUANTIZER_SCHEMA_V1",
    "ACTIVATION_QUANTIZER_SCHEMA_V2",
    "contract_answer",
    "packaged_activation_quantizers",
    "require_activation_quantizer_attested",
    "describe_dev_pin",
    "dev_pin_requested",
    "load_tessera_contract",
    "require_pin_native_extensions_match_contract",
]


class TesseraContractError(RuntimeError):
    """The Tessera runtime contract is absent, malformed, or off its pin."""


#: The schema ids this reader accepts.  Both are checked before any key is
#: read: an older table is not a subset of this one, and "missing field" is the
#: wrong error to hand someone whose contract predates the field.
#: ``TESSERA_LANE_SCHEMA`` is the CURRENT lane grammar and follows
#: ``lane_eligibility.LANE_ELIGIBILITY_SCHEMA_TESSERA`` (v10 since
#: 2026-09-12) so that the two readers cannot disagree about which schema
#: is newest.
TESSERA_CONTRACT_SCHEMA = "tessera.runtime-contract.v1"
TESSERA_LANE_SCHEMA = LANE_ELIGIBILITY_SCHEMA_TESSERA
TESSERA_LANE_SCHEMAS = frozenset(
    {LANE_ELIGIBILITY_SCHEMA_TESSERA_V4} | SCOPED_LANE_SCHEMAS)
#: The ``fused_module`` block's own schema id, checked the same way.
FUSED_MODULE_SCHEMA = "tessera.fused-module.v1"

#: The only two licences a ``fused_module.fields`` entry may carry.  An
#: unknown token is REFUSED rather than mapped onto either: "shared" and
#: "per_member" are the two answers a group allocator can act on, and a third
#: word this reader does not know is a constraint it would silently drop.
FUSED_MODULE_FIELD_LICENCES = frozenset({"shared", "per_member"})

#: The development override.  See the module docstring; there is no default.
TESSERA_DEV_PIN_ENV = "PRISMAQUANT_TESSERA_DEV_PIN"

#: The Tessera commit this pin's answer was reviewed against.  Declared and
#: recorded; NOT compared to anything.  A moving ``master`` is not a review
#: event -- :data:`TESSERA_DEV_PIN_ANSWER` is what refuses. Re-pinned
#: 2026-09-09 to b1eb1dccc (Tessera #437/#438) for the opt-in best-form
#: window encoder and explicit tile controls. Actual GLM endpoint runs retain
#: exact wire bytes and scores; no encoder default or serving cell changes.
#: The pin also includes the reviewed cached-export and rank-local TP2 intake
#: fixes (#434/#436), whose bounded controls do not qualify a full GLM serve.
#: Previously re-pinned
#: 2026-09-08 to 07ad344c3 (Tessera #431) for explicit packed checkpoint
#: execution and strict typed export carriers, retaining the exact contract
#: bytes and admission answer. Previously re-pinned
#: 2026-09-08 to 9d2314819 (Tessera #429) for bounded canonical Hessian
#: references and their nonblocking file refusal. The allocator/serving-cell
#: answer remains unchanged; the raw contract additionally records LFM
#: construction output sizes. The admission answer was last changed at
#: contract v22 / lane schema v9 (Tessera master 8ed1d9a, the merge of its
#: #332, which answers its #327; v21 landed at b8b1cb38 in its #313 and the
#: release e78959ed carried v20). At that review both pins were aligned on
#: b1eb1dccc: they name ONE object, and
#: letting them drift is how two of this repository's own spec files came to
#: disagree about one runtime.  Between the v17 review and this one the
#: answer moved in exactly four places and nowhere else -- no family, rung,
#: route status, launch, image or version moved, and the ten cell ids are the
#: same ten in the same order: the lane schema (v6 -> v8); every cell's
#: evidence gained ``smoke.attribution`` / ``smoke.control`` (v7) and
#: ``artifact`` (v8); every native extension gained its ``lane`` predicate
#: (v20); and the two ``routed_moe`` cells' ``smoke.status`` moved from
#: ``repetitive`` to ``recorded`` (v21, Tessera #313, receipt
#: ``docs/measurements/moe-smoke-recorded-2026-09-05.md``), with the v18
#: control retired from them (``attribution: unattributed``, ``control:
#: null`` -- the shape the BF16 cells already used).
#:
#: The answer below TRANSCRIBES what the runtime publishes, including its two
#: ``routed_moe`` cells.  Transcribing them does not admit them: admission is
#: a separate predicate (``lane_eligibility.cell_evidence_admits``) that
#: refuses on ``smoke.status`` alone.  Under v17 and v20 it refused both
#: routed-MoE cells on the degenerate greedy smoke the runtime recorded
#: (``repetitive``); under v21 the runtime records a smoke through the
#: checkpoint's own chat template that does not degenerate on either arm, the
#: status reads ``recorded``, and the SAME status-only rule admits both cells
#: with no predicate change here (prismaquant #198, option C).  That is the
#: mechanism working as designed: the evidence block is part of the answer,
#: so the clean smoke re-staled the pin and this literal's diff is the review
#: a human reads before anything is admitted.  Whether routed-MoE Tessera is
#: PROMOTED past the menu still requires the independent artifact and serving
#: gates; #198 records the resolved producer remeasurement decision.
#:
#: What moved v21 -> v22 (Tessera #332, answering its #327), and nothing
#: else -- same ten cell ids in the same order, no family, rung, route
#: status, launch, image or version:
#:   1. lane_schema v8 -> v9;
#:   2. every cell's ``evidence.smoke`` gained ``record`` -- ``null`` on the
#:      eight dense cells, and on the two ``routed_moe`` cells the
#:      instrument, the rule, the reference and the rows the status was
#:      derived over;
#:   3. those two cells' ``smoke.attribution`` moved ``unattributed`` ->
#:      ``shared_with_reference``, DERIVED from the record by Tessera's own
#:      ``derive_smoke_attribution`` rather than asserted.
#: Why that is a re-review and not a bump: at v21 the ``recorded`` status
#: rested on a repetition rule that lived only in a dated measurements file,
#: was checked by nothing, and was satisfiable by an empty completion
#: (Tessera #327, P1).  v22 puts the rule and its observations in the
#: contract, so the status this pin admits on is one a reader can re-derive
#: -- which ``lane_eligibility.parse_cell_evidence`` does, through Tessera's
#: ``derive_smoke_status``, refusing a published status the record does not
#: derive.  The admission itself is unchanged: ``cell_evidence_admits`` is
#: still status-only and still ``{repetitive}``.
#:
#: What moved v22 -> v23 (Tessera #456, merged as #464), and nothing else --
#: same ten cell ids in the same order, byte for byte, and no family, rung,
#: route status, launch, activation contract, image or version:
#:   1. lane_schema v9 -> v10;
#:   2. ``lane_eligibility.platforms`` entries stopped being bare keys. Each
#:      now carries ``backend`` (``cuda | hip``), exactly one of
#:      ``compute_capability`` / ``gcn_arch``, a ``serve_image`` that is a
#:      digest iff the platform has at least one cell and ``null`` otherwise,
#:      and ``executes`` -- a map over every family in ``formats[]`` whose
#:      value is that family's own route contract or ``null``;
#:   3. two AMD platforms arrived, ``gfx1151`` (Strix Halo, RDNA3.5) and
#:      ``gfx1201`` (RDNA4), with ``serve_image: null``, NO cells,
#:      ``TESSERA_BF16_K1`` backed and ``TESSERA_E4M3_K1`` /
#:      ``TESSERA_E2M1_K2`` ``null``.
#: Why the answer below moves by exactly ONE entry: the answer is the
#: projection an ADMISSION gate reads, and at this pin nothing here decides on
#: a platform's ``executes``.  The platform axis is published and admitted as
#: grammar; the first gate that reads it (the platform-aware serving route,
#: PrismaQuant #528) widens this projection, and widening a projection is its
#: own re-review by the rule stated above.  ``contract_version`` itself is not
#: in the answer (22 -> 23 alone would not have re-staled it); the lane schema
#: is, and a v10 document read by a v9-closed reader is refused by name, which
#: is the designed fail-closed rather than a compatibility break.
#:
#: What moved v23 -> v24 (Tessera #474, merged as 27be1a602, with its #475
#: on top), and nothing else:
#:   1. two NEW cells, ``tessera_bf16_k1_dense_gfx1201_decode`` and
#:      ``..._batch`` -- ``TESSERA_BF16_K1`` dense on ``gfx1201`` (RDNA4,
#:      RX 9070 XT) at rung q256 = 1792, ``route_status``
#:      ``backed_with_serve_flag``, ``qualification`` ``device_qualified``,
#:      executing ``[{torch.mm, torch_window}]`` on a ROCm vLLM image;
#:   2. ``lane_eligibility.platforms.gfx1201.serve_image`` stops being
#:      ``null`` -- v10 requires it to be an image one of that platform's OWN
#:      cells attests, and at v23 it had none.
#: The lane schema does NOT move: v24 is additive for a v10 reader, every
#: field these cells carry is a field v10 already defines, and the ten
#: ``sm_121`` cells are byte-identical -- the answer's drift is exactly the
#: two NEW lines and no reviewed line changed.
#:
#: What moved v24 -> v29 (Tessera master 4c384e6049, the merge of its #517,
#: 88 commits after 7dbbacbd), and nothing else -- no rate range, attested
#: rung, quant method, fused module, native extension or lane schema, and the
#: twelve v24 cells are byte-identical in the answer:
#:   1. ``activation_quantizers`` (v25, Tessera #484/#485) stops being empty:
#:      the ``sm_121`` / ``e2m1_group16_ue4m3_static`` table, eleven probe
#:      groups run on ``torch.ops._C.scaled_fp4_quant``.  This FLIPS
#:      ``require_activation_quantizer_attested`` for that contract from
#:      "publishes no quantiser" to recomputing every group with PrismaQuant's
#:      own oracle -- the review of a rounding rule #567/#574 asked for;
#:   2. ``TESSERA_E2M1_K2``'s ``row`` loader axis ``refused`` -> ``sharded``
#:      (v26), so a column-parallel K2 unit stops being unloadable at TP > 1;
#:   3. two NEW cells (v28): ``tessera_e2m1_k2_routed_moe_sm121_decode_resident``
#:      and ``..._batch_resident`` -- ``TESSERA_E2M1_K2`` routed MoE at q896,
#:      ``backed_with_serve_flag``, ``device_qualified``, executing
#:      ``vllm.fused_moe.modular_kernel`` through ``torch_materialize_stock``,
#:      eager only, evidence grade ``route_only`` with ``smoke.status:
#:      not_recorded``;
#:   4. every unit's ``max_world_size`` 1 -> 2 (v29), each citing the served
#:      TP2 receipt ``glm53_a4_stub_tp2_sm121`` that
#:      :func:`_require_world_size_receipt` now requires above 1.
#: v27's per-format ``structures`` lists are grammar no gate here reads, so
#: they are not in the answer.
#:
#: ITEM 3 IS AN ADMISSION, AND A HUMAN'S CALL.  ``cell_evidence_admits``
#: refuses only ``repetitive``; ``not_recorded`` is not a refusal, so
#: accepting this answer admits routed-MoE ``TESSERA_E2M1_K2_R896`` on
#: ``sm_121`` at those two scopes on a ``route_only`` grade with no recorded
#: smoke.  That is a routed-MoE promotion under principle 9 and it is flagged
#: for review in the pull request rather than decided by this literal
#: (prismaquant #198).
#:
#: THIS ONE IS NOT A RE-TRANSCRIPTION.  The two bumps before it moved grammar
#: or prose and admitted what the previous pin admitted; this one admits a
#: route that did not exist on this side.  ``cell_evidence_admits`` is
#: status-only and both cells publish ``smoke.status: recorded``, so accepting
#: this answer flips ``ServingLaneSpec.route_status_for`` and
#: ``tessera_render.tessera_attesting_cells`` for ``TESSERA_BF16_K1_R1792``
#: on ``gfx1201`` from ``unattested``/``:no_cell`` to
#: ``backed_with_serve_flag``.  That is the whole review: the Tessera-16
#: W16A16 lane is now attested on one AMD device.  ``gfx1151`` still ships no
#: cell and still answers ``:no_cell`` for every family -- backing is
#: permission to price, a cell is permission to ship, and only one of the two
#: AMD platforms crossed that line.
#:
#: Two scope facts the receipts carry, recorded because a claim inherits the
#: scope of the artifact it was measured on (principle 14's corollary): the
#: grade is ``kl_lower_bound`` (a top-1024 teacher-student intersection bound,
#: KL >= 0.004906 over 4088 prefill positions for batch and >= 0.004804 over
#: 256 M=1 positions for decode), NOT ``kl_full_vocab`` -- no instrument in
#: either tree produces a full-vocab KL, so a producer-side gate that expected
#: one would refuse an artifact for a measurement that does not exist.  And
#: the receipt's own scope line says gfx1201 under WSL2 proves the HIP code
#: path; it says nothing about gfx1151 numerics and nothing about performance.
#:
#: The resident-H integration uses the reviewed runtime tree from Tessera
#: PR #441, which this pin supersedes without changing any priced byte: the
#: producer source identity changes and existing priced bytes keep their seal.
#: No release tag names this development commit.
#: Re-pinned 2026-09-19 to cc739a55c -- Tessera master's merge of #563,
#: the head at which the #562 (D2/D2b, stacked on #560's
#: flash/506-routed-full-domain-rates-20260918) union with #563's docs
#: fix-forward is complete -- for the coordinated post-#560-lineage bump
#: (PrismaQuant #760).  Contract v32, lane schema still v10.  The digest is
#: bound by the one command against the canonical remote, never an
#: installed copy.  THE GATED LANDING IS EXECUTED: the union digest
#: predicted while the gate was written (`712a15e4…`) went stale because
#: #563's rework resolved two master conflicts on its branch; the measured
#: bytes at the union head are what this pin names.  Tessera master has
#: since advanced past contract v33 (#568), whose activation-quantizer
#: schema v2 this reader now reads (RobTand/prismaquant#926): the READER no
#: longer blocks a bump, and which commit to pin stays a separate review
#: with its own answer diff.
#: Re-pinned 2026-09-22 to acf9eafa6 -- Tessera master after #588, #590 and
#: #592 -- so PrismaQuant #940 (the rooted ``tessera.cached_units.v2``
#: reader) and #945 (fenced source-digest adoption) are tested against the
#: pin rather than a vendored tree.  Contract v32 -> v34, lane schema v10.
#: The answer diff below is the review: see the v34 note inside the literal.
#: Re-pinned 2026-09-22 to 07bfcc0e9 -- Tessera master after #580, #582, #583,
#: #585 and #596 (``SourceDigestCache.adopt``).  The packaged contract is
#: byte-identical to ``acf9eafa6``'s, so the answer below does not move.
TESSERA_DEV_PIN_COMMIT = "07bfcc0e9b7da13276938cb722bc7dcd893e6c63"

#: sha256 of ``tessera/serving/runtime_contract.json`` at that commit -- the
#: bytes a human read when the answer below was accepted.  Recorded, and
#: compared into provenance against the bytes this run read, so prose-only
#: drift is visible; it is not the refusal.
TESSERA_DEV_PIN_CONTRACT_SHA256 = (
    "d37c9448a751feb3e65db1807a7dff1fbacc767a2ce419dfee70f458dbf03472"
)

#: The ANSWER this pin was reviewed against -- every value the ADMISSION
#: gate reads, in the vocabulary of :func:`contract_answer`.  This literal, not the file's
#: bytes, is what refuses: a Tessera commit that rewrites prose, reorders keys
#: or bumps ``contract_version`` publishes the same answer and does not
#: re-stale the pin, while any move in a family, a rate range, an attested
#: rung, a world-size ceiling, a cell, the canonical ``quant_method``, or a
#: published native extension (its prefix, its glob, the ``match`` rule, the
#: routes that need it, or what runs when it is absent) does -- with a
#: field-level diff naming it.  The git diff of this literal is the review.
#:
#: It also re-stales when the PROJECTION widens, which is the other half of
#: the same property -- and it just did, at this same pin and without a
#: Tessera byte moving: reading activation-quantizer schema v2 (#926) made
#: ``generated.image`` the value that SELECTS which quantiser table covers a
#: measurement, so it joined ``ActivationQuantizerAttestation.answer()`` as
#: its third column.  The row below carries the image the v32 table was
#: generated in; nothing else about it moved.  Earlier: reading lane schema
#: v9 gave ``CellEvidence.answer()`` a
#: seventh member for the smoke's record, and a reader that widened what it
#: looks at without re-reviewing would be admitting a field nobody had read.
#:
#: Against the v22 contract that seventh member is where the review is.  The
#: eight dense cells publish ``record: null`` and project a trailing ``None``,
#: so they did not move at all; the drift from the v21 answer was exactly
#: three entries -- ``lane_schema`` (v8 -> v9) and the two ``routed_moe``
#: cells, which gained the record and whose ``attribution`` moved
#: ``unattributed`` -> ``shared_with_reference`` because v9 DERIVES it from
#: that record.  Read those three and nothing else changed.
#:
#: Against the v24 contract the review is the two NEW ``gfx1201`` rows and
#: nothing else: the ten ``sm_121`` rows below are byte-identical to the ones
#: the v23 pin carried, in the same order, and no other key in this literal
#: moved.  See :data:`TESSERA_DEV_PIN_COMMIT` for what accepting them admits.
#:
#: Against the v29 contract the review is seven entries: the
#: ``activation_quantizers`` table, three ``max_world_size`` 1 -> 2, the K2
#: ``row`` axis, and the two routed-MoE K2 rows appended to ``cells``.  The
#: twelve rows above them did not move.  The literal is
#: ``pprint.pformat(contract_answer(c), width=79, sort_dicts=False)`` with
#: each dict in the key order the previous literal used.
#:
#: Against the v32 contract the review is the four moves the comment inside
#: the literal names -- eight dense cells WITHDRAWN (admission shrinks: the
#: four ``recorded`` TESSERA_BF16_K1 rows among them), the routed E2M1_K2
#: reader domain and cells widened to [128..896] on the unchanged
#: ``not_recorded``/``route_only`` pair, the D2b ``q256`` scoping of KL
#: receipts (this reader's widened ``@q`` projection), and the retired
#: span-2 CUDA decoder leaving one native-extension row.  The routed-MoE
#: ``recorded`` pair is byte-identical, so the status-only evidence gate
#: admits the same scope it did at v29; the withdrawal and the widen are
#: reviewed facts, not gate changes.
#:
#: **The ``cells`` rows are POSITIONAL tuples, and this is the column order.**
#: They stay positional -- a per-row dict would triple the diff a reviewer
#: reads for no fact -- so the order lives here, beside the values, and it is
#: the order :func:`contract_answer` builds.  ``platform`` has been column 1
#: since the v10 re-review, which is what keeps two rows that differ only in
#: their device from publishing as rows a reader cannot tell apart; v24 is the
#: first contract where that actually happens.  The columns are::
#:
#:      0  id                     10  requires_serve_flags (sorted)
#:      1  platform               11  executes [[symbol, decoder], ...]
#:      2  family                 12  residency_modes (sorted)
#:      3  structure              13  runtime {image, execution_modes}
#:      4  regime                 14  runtime vllm version
#:      5  rungs_q256 (sorted)    15  runtime torch version
#:      6  activation_contract    16  evidence (see below)
#:      7  route_status
#:      8  qualification
#:      9  requires_plugin
#:
#: Columns 13-15 are present only while the contract requires a serving
#: context, and column 16 only while the cell publishes evidence -- both are
#: conditional in :func:`contract_answer` and both hold at this pin.  Column
#: 16 is itself positional, ``CellEvidence.answer()``::
#:
#:      0  grade                  4  smoke.control
#:      1  smoke.status           5  smoke.artifact
#:      2  kl kinds (sorted)      6  smoke.record
#:      3  smoke.attribution
#:
#: A column added on either tuple is a WIDENED projection, and the rule above
#: applies to it: it re-stales this pin even when no published value moved.
TESSERA_DEV_PIN_ANSWER = {'schema': 'tessera.runtime-contract.v1',
 # Against the v34 contract the review is five additions and no removals.
 # (1) v33/#568 publishes the sm_121 fp4 activation-quantiser table as a list
 # of per-image attestations; the stock-image row below is byte-identical to
 # v32's, and a second row for the glm53-nope-sm121 serving image arrives
 # with the same rounding vectors.  (2)-(5) v34/#579 attests the fused window
 # GEMM and mints four dense sm_121 cells on tessera::window_gemm_dense:
 # TESSERA_BF16_K1 q1792 and TESSERA_E4M3_K1 q1024, batch and decode, graded
 # route_only with smoke.status not_recorded, which the status-only evidence
 # gate does not refuse.  Accepting this answer therefore ADMITS those two
 # dense rungs on sm_121 (backed_with_serve_flag, TESSERA_SERVE_MODE) where
 # the v32 pin answered unattested/no_cell.  Every other row is unchanged.
 # The v32 review follows, as history; its 'only cells' count is superseded.
 # Against the v32 contract the review is exactly four moves, and the
 # WITHDRAWALS are the headline, not the widenings.  Tessera master (its #538
 # and the A4 retirement) withdrew all eight dense cells that were not
 # E2M1_K2: the four TESSERA_BF16_K1 rows -- which carried ``recorded``
 # evidence and were the only cells on gfx1201 -- and the four TESSERA_E4M3_K1
 # dense rows, resident and streamed.  Accepting this answer therefore admits
 # STRICTLY LESS than the v29 pin did on dense routes: TESSERA_BF16_K1_R1792
 # answers ``unattested``/``no_cell`` on every platform again, as it did
 # before 2026-09-13, and the lane's only cells are the six sm_121 rows below.
 # The routed-MoE ``recorded`` pair (E4M3_K1, rung 1024) is byte-identical,
 # so the status-only evidence gate admits the same scope it did at v29 --
 # the routed E2M1_K2 widen ([896] -> the full trellis domain [128..896]
 # step 128, receipted by the seven-rung green load) rides the two
 # ``not_recorded``/``route_only`` cells as before and is Rob's #198 call,
 # flagged here rather than decided by this literal.  D2b scopes the dense
 # batch cell's KL receipts to the rung they measured (``@q896`` in the kl
 # token -- the reader's widened projection, first exercised by this pin),
 # and the retired span-2 CUDA decoder leaves ``native_extensions`` with the
 # one window-GEMV row.  Nothing else moved: lane schema stays v10, the TP
 # ceiling stays 2 on the same receipt, and the quantiser table is
 # byte-identical.
 'lane_schema': 'tessera.lane-eligibility.v10',
 'required_regimes': ['batch', 'decode'],
 'quant_method': 'tessera',
 'fused_module': {'schema': 'tessera.fused-module.v1',
                  'fields': {'body': 'shared',
                             'columns': 'shared',
                             'family': 'shared',
                             'grid': 'shared',
                             'plane': 'shared',
                             'q256': 'per_member',
                             'rows': 'per_member',
                             'structure': 'shared'},
                  'sidecar_q256': 'int_or_per_role_list',
                  'mixed_rung_receipt': False},
 'native_extensions': [{'module_name_prefix': 'tessera_window_gemv',
                        'filename_glob': 'tessera_window_gemv*.so',
                        'match': 'basename_fnmatch',
                        'routes': ['TESSERA_BF16', 'TESSERA_FP8'],
                        'when_unavailable': {'resident': {'status': 'substituted',
                                                          'decoder': 'torch_window'},
                                             'streamed': {'status': 'substituted',
                                                          'decoder': 'torch_window'}},
                        'lane': {'decoder': 'window_gemv',
                                 'requires': {'column_rates': [1, 2, 4],
                                              'window_bits': [14],
                                              'body': 'window',
                                              'plane': 'channel',
                                              'release_overrides': False,
                                              'diagonals': False,
                                              'rotation': ['none'],
                                              'start_state': False,
                                              'grid_arities': [1]}}}],
 'activation_quantizers': [['sm_121',
                            'e2m1_group16_ue4m3_static',
                            'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc'
                            '4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
                            'torch.ops._C.scaled_fp4_quant',
                            'group',
                            16,
                            'E2M1',
                            'UE4M3',
                            'static_per_module',
                            [['midpoint_dyadic',
                              'e2m1_midpoint_dyadic',
                              1065353216,
                              [16576,
                               16000,
                               16192,
                               16288,
                               16352,
                               16416,
                               16480,
                               16544,
                               48768,
                               48960,
                               49056,
                               49120,
                               49184,
                               49248,
                               49312,
                               0],
                              56,
                              [7,
                               0,
                               2,
                               2,
                               4,
                               4,
                               6,
                               6,
                               8,
                               10,
                               10,
                               12,
                               12,
                               14,
                               14,
                               0]],
                             ['code_identity',
                              'e2m1_code_identity',
                              1065353216,
                              [0,
                               32768,
                               16128,
                               16256,
                               16320,
                               16384,
                               16448,
                               16512,
                               16576,
                               48896,
                               49024,
                               49088,
                               49152,
                               49216,
                               49280,
                               49344],
                              56,
                              [0,
                               8,
                               1,
                               2,
                               3,
                               4,
                               5,
                               6,
                               7,
                               9,
                               10,
                               11,
                               12,
                               13,
                               14,
                               15]],
                             ['midpoint_reciprocal',
                              'e2m1_midpoint_reciprocal',
                              1065353216,
                              [16656,
                               16064,
                               16272,
                               16368,
                               16424,
                               16496,
                               16552,
                               16624,
                               48832,
                               49040,
                               49136,
                               49192,
                               49264,
                               49320,
                               49392,
                               0],
                              60,
                              [7,
                               0,
                               2,
                               2,
                               4,
                               4,
                               6,
                               6,
                               8,
                               10,
                               10,
                               12,
                               12,
                               14,
                               14,
                               0]],
                             ['midpoint_global_scale',
                              'e2m1_midpoint_global_scale',
                              1069547520,
                              [16656,
                               16064,
                               16272,
                               16368,
                               16424,
                               16496,
                               16552,
                               16624,
                               48832,
                               49040,
                               49136,
                               49192,
                               49264,
                               49320,
                               49392,
                               0],
                              65,
                              [7,
                               0,
                               2,
                               2,
                               4,
                               4,
                               6,
                               6,
                               8,
                               10,
                               10,
                               12,
                               12,
                               14,
                               14,
                               0]],
                             ['midpoint_seven_fourths',
                              'e2m1_midpoint_reciprocal',
                              1065353216,
                              [16680,
                               16096,
                               16296,
                               16396,
                               16452,
                               16524,
                               16580,
                               16652,
                               48864,
                               49064,
                               49164,
                               49220,
                               49292,
                               49348,
                               49420,
                               0],
                              62,
                              [7,
                               0,
                               2,
                               2,
                               4,
                               4,
                               6,
                               6,
                               8,
                               10,
                               10,
                               12,
                               12,
                               14,
                               14,
                               0]],
                             ['midpoint_reciprocal_ulp_below',
                              'e2m1_midpoint_ulp_below',
                              1065353216,
                              [16656,
                               16063,
                               16271,
                               16367,
                               16423,
                               16495,
                               16551,
                               16623,
                               48831,
                               49039,
                               49135,
                               49191,
                               49263,
                               49319,
                               49391,
                               0],
                              60,
                              [7,
                               0,
                               1,
                               2,
                               3,
                               4,
                               5,
                               6,
                               8,
                               9,
                               10,
                               11,
                               12,
                               13,
                               14,
                               0]],
                             ['midpoint_reciprocal_ulp_above',
                              'e2m1_midpoint_ulp_above',
                              1065353216,
                              [16656,
                               16065,
                               16273,
                               16369,
                               16425,
                               16497,
                               16553,
                               16625,
                               48833,
                               49041,
                               49137,
                               49193,
                               49265,
                               49321,
                               49393,
                               0],
                              60,
                              [7,
                               1,
                               2,
                               3,
                               4,
                               5,
                               6,
                               7,
                               9,
                               10,
                               11,
                               12,
                               13,
                               14,
                               15,
                               0]],
                             ['element_saturation',
                              'e2m1_saturation',
                              1065353216,
                              [16584,
                               49352,
                               16580,
                               49348,
                               16576,
                               49344,
                               16574,
                               49342,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0],
                              56,
                              [7,
                               15,
                               7,
                               15,
                               7,
                               15,
                               7,
                               15,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0]],
                             ['block_scale_underflow_tie',
                              'block_scale_underflow_tie',
                              1065353216,
                              [15296,
                               48064,
                               15168,
                               47936,
                               15040,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0],
                              0,
                              [0,
                               8,
                               0,
                               8,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0]],
                             ['block_scale_underflow_above',
                              'block_scale_underflow_above',
                              1065353216,
                              [15297,
                               48065,
                               15168,
                               47936,
                               15040,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0],
                              1,
                              [5,
                               13,
                               3,
                               11,
                               2,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0]],
                             ['block_scale_overflow',
                              'block_scale_overflow',
                              1065353216,
                              [17728,
                               50496,
                               17600,
                               50368,
                               17472,
                               17344,
                               16128,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0],
                              126,
                              [7,
                               15,
                               5,
                               13,
                               3,
                               2,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0]]]],
                           ['sm_121',
                            'e2m1_group16_ue4m3_static',
                            '192.168.1.107/prismaquant/glm53-nope-sm121@sha256:'
                            '6941847351647ca714bbe7115ce6f627131bf78fc7eff86ddb98b11e6d25b46e',
                            'torch.ops._C.scaled_fp4_quant',
                            'group',
                            16,
                            'E2M1',
                            'UE4M3',
                            'static_per_module',
                            [['midpoint_dyadic',
                              'e2m1_midpoint_dyadic',
                              1065353216,
                              [16576,
                               16000,
                               16192,
                               16288,
                               16352,
                               16416,
                               16480,
                               16544,
                               48768,
                               48960,
                               49056,
                               49120,
                               49184,
                               49248,
                               49312,
                               0],
                              56,
                              [7,
                               0,
                               2,
                               2,
                               4,
                               4,
                               6,
                               6,
                               8,
                               10,
                               10,
                               12,
                               12,
                               14,
                               14,
                               0]],
                             ['code_identity',
                              'e2m1_code_identity',
                              1065353216,
                              [0,
                               32768,
                               16128,
                               16256,
                               16320,
                               16384,
                               16448,
                               16512,
                               16576,
                               48896,
                               49024,
                               49088,
                               49152,
                               49216,
                               49280,
                               49344],
                              56,
                              [0,
                               8,
                               1,
                               2,
                               3,
                               4,
                               5,
                               6,
                               7,
                               9,
                               10,
                               11,
                               12,
                               13,
                               14,
                               15]],
                             ['midpoint_reciprocal',
                              'e2m1_midpoint_reciprocal',
                              1065353216,
                              [16656,
                               16064,
                               16272,
                               16368,
                               16424,
                               16496,
                               16552,
                               16624,
                               48832,
                               49040,
                               49136,
                               49192,
                               49264,
                               49320,
                               49392,
                               0],
                              60,
                              [7,
                               0,
                               2,
                               2,
                               4,
                               4,
                               6,
                               6,
                               8,
                               10,
                               10,
                               12,
                               12,
                               14,
                               14,
                               0]],
                             ['midpoint_global_scale',
                              'e2m1_midpoint_global_scale',
                              1069547520,
                              [16656,
                               16064,
                               16272,
                               16368,
                               16424,
                               16496,
                               16552,
                               16624,
                               48832,
                               49040,
                               49136,
                               49192,
                               49264,
                               49320,
                               49392,
                               0],
                              65,
                              [7,
                               0,
                               2,
                               2,
                               4,
                               4,
                               6,
                               6,
                               8,
                               10,
                               10,
                               12,
                               12,
                               14,
                               14,
                               0]],
                             ['midpoint_seven_fourths',
                              'e2m1_midpoint_reciprocal',
                              1065353216,
                              [16680,
                               16096,
                               16296,
                               16396,
                               16452,
                               16524,
                               16580,
                               16652,
                               48864,
                               49064,
                               49164,
                               49220,
                               49292,
                               49348,
                               49420,
                               0],
                              62,
                              [7,
                               0,
                               2,
                               2,
                               4,
                               4,
                               6,
                               6,
                               8,
                               10,
                               10,
                               12,
                               12,
                               14,
                               14,
                               0]],
                             ['midpoint_reciprocal_ulp_below',
                              'e2m1_midpoint_ulp_below',
                              1065353216,
                              [16656,
                               16063,
                               16271,
                               16367,
                               16423,
                               16495,
                               16551,
                               16623,
                               48831,
                               49039,
                               49135,
                               49191,
                               49263,
                               49319,
                               49391,
                               0],
                              60,
                              [7,
                               0,
                               1,
                               2,
                               3,
                               4,
                               5,
                               6,
                               8,
                               9,
                               10,
                               11,
                               12,
                               13,
                               14,
                               0]],
                             ['midpoint_reciprocal_ulp_above',
                              'e2m1_midpoint_ulp_above',
                              1065353216,
                              [16656,
                               16065,
                               16273,
                               16369,
                               16425,
                               16497,
                               16553,
                               16625,
                               48833,
                               49041,
                               49137,
                               49193,
                               49265,
                               49321,
                               49393,
                               0],
                              60,
                              [7,
                               1,
                               2,
                               3,
                               4,
                               5,
                               6,
                               7,
                               9,
                               10,
                               11,
                               12,
                               13,
                               14,
                               15,
                               0]],
                             ['element_saturation',
                              'e2m1_saturation',
                              1065353216,
                              [16584,
                               49352,
                               16580,
                               49348,
                               16576,
                               49344,
                               16574,
                               49342,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0],
                              56,
                              [7,
                               15,
                               7,
                               15,
                               7,
                               15,
                               7,
                               15,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0]],
                             ['block_scale_underflow_tie',
                              'block_scale_underflow_tie',
                              1065353216,
                              [15296,
                               48064,
                               15168,
                               47936,
                               15040,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0],
                              0,
                              [0,
                               8,
                               0,
                               8,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0]],
                             ['block_scale_underflow_above',
                              'block_scale_underflow_above',
                              1065353216,
                              [15297,
                               48065,
                               15168,
                               47936,
                               15040,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0],
                              1,
                              [5,
                               13,
                               3,
                               11,
                               2,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0]],
                             ['block_scale_overflow',
                              'block_scale_overflow',
                              1065353216,
                              [17728,
                               50496,
                               17600,
                               50368,
                               17472,
                               17344,
                               16128,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0],
                              126,
                              [7,
                               15,
                               5,
                               13,
                               3,
                               2,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0]]]]],
 'families': {'TESSERA_BF16_K1': {'reader_rate_range_q256': [256, 4096],
                                  'attested_rungs_q256': [1792],
                                  'max_world_size': 2,
                                  'loader_axes': {'column': 'sharded',
                                                  'row': 'sharded'}},
              'TESSERA_E2M1_K2': {'reader_rate_range_q256': [128, 896],
                                  'attested_rungs_q256': [128,
                                                          256,
                                                          384,
                                                          512,
                                                          640,
                                                          768,
                                                          896],
                                  'max_world_size': 2,
                                  'loader_axes': {'column': 'sharded',
                                                  'row': 'sharded'}},
              'TESSERA_E4M3_K1': {'reader_rate_range_q256': [256, 2048],
                                  'attested_rungs_q256': [1024],
                                  'max_world_size': 2,
                                  'loader_axes': {'column': 'sharded',
                                                  'row': 'sharded'}}},
 'cells': [['tessera_bf16_k1_dense_sm121_batch',
            'sm_121',
            'TESSERA_BF16_K1',
            'dense',
            'batch',
            [1792],
            'bf16_unquantized',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident|streamed'],
            [['tessera::window_gemm_dense', 'native_window_gemm']],
            ['resident', 'streamed'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['route_only',
             'not_recorded',
             [],
             'unattributed',
             None,
             None,
             None]],
           ['tessera_bf16_k1_dense_sm121_decode',
            'sm_121',
            'TESSERA_BF16_K1',
            'dense',
            'decode',
            [1792],
            'bf16_unquantized',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident|streamed'],
            [['tessera::window_gemm_dense', 'native_window_gemm']],
            ['resident', 'streamed'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['route_only',
             'not_recorded',
             [],
             'unattributed',
             None,
             None,
             None]],
           ['tessera_e2m1_k2_dense_sm121_batch',
            'sm_121',
            'TESSERA_E2M1_K2',
            'dense',
            'batch',
            [896],
            'e2m1_group16_ue4m3_static',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident|streamed'],
            [['torch._scaled_mm', 'native_span2']],
            ['resident', 'streamed'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['compiled', 'eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['kl_lower_bound',
             'not_recorded',
             ['topk_intersection_lower_bound@1024@q896'],
             'unattributed',
             None,
             None,
             None]],
           ['tessera_e2m1_k2_dense_sm121_decode',
            'sm_121',
            'TESSERA_E2M1_K2',
            'dense',
            'decode',
            [896],
            'e2m1_group16_ue4m3_static',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident|streamed'],
            [['torch._scaled_mm', 'native_span2']],
            ['resident', 'streamed'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['compiled', 'eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['route_only',
             'not_recorded',
             [],
             'unattributed',
             None,
             None,
             None]],
           ['tessera_e2m1_k2_routed_moe_sm121_batch_resident',
            'sm_121',
            'TESSERA_E2M1_K2',
            'routed_moe',
            'batch',
            [128, 256, 384, 512, 640, 768, 896],
            'e2m1_group16_ue4m3_static',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident'],
            [['vllm.fused_moe.modular_kernel', 'torch_materialize_stock']],
            ['resident'],
            {'image': 'localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb',
             'execution_modes': ['eager']},
            '0.28.1rc1.dev397+gfd4a15126.d20260904',
            '2.13.0+cu130',
            ['route_only',
             'not_recorded',
             [],
             'unattributed',
             None,
             None,
             None]],
           ['tessera_e2m1_k2_routed_moe_sm121_decode_resident',
            'sm_121',
            'TESSERA_E2M1_K2',
            'routed_moe',
            'decode',
            [128, 256, 384, 512, 640, 768, 896],
            'e2m1_group16_ue4m3_static',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident'],
            [['vllm.fused_moe.modular_kernel', 'torch_materialize_stock']],
            ['resident'],
            {'image': 'localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb',
             'execution_modes': ['eager']},
            '0.28.1rc1.dev397+gfd4a15126.d20260904',
            '2.13.0+cu130',
            ['route_only',
             'not_recorded',
             [],
             'unattributed',
             None,
             None,
             None]],
           ['tessera_e4m3_k1_dense_sm121_batch',
            'sm_121',
            'TESSERA_E4M3_K1',
            'dense',
            'batch',
            [1024],
            'fp8_per_token_dynamic',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident|streamed'],
            [['tessera::window_gemm_dense', 'native_window_gemm']],
            ['resident', 'streamed'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['route_only',
             'not_recorded',
             [],
             'unattributed',
             None,
             None,
             None]],
           ['tessera_e4m3_k1_dense_sm121_decode',
            'sm_121',
            'TESSERA_E4M3_K1',
            'dense',
            'decode',
            [1024],
            'fp8_per_token_dynamic',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident|streamed'],
            [['tessera::window_gemm_dense', 'native_window_gemm']],
            ['resident', 'streamed'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['route_only',
             'not_recorded',
             [],
             'unattributed',
             None,
             None,
             None]],
           ['tessera_e4m3_k1_routed_moe_sm121_batch_resident',
            'sm_121',
            'TESSERA_E4M3_K1',
            'routed_moe',
            'batch',
            [1024],
            'fp8_per_token_dynamic',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident'],
            [['vllm.fused_moe.modular_kernel', 'torch_materialize_stock']],
            ['resident'],
            {'image': 'eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c',
             'execution_modes': ['eager']},
            '0.28.1rc1.dev397+gfd4a15126.d20260904',
            '2.13.0+cu130',
            ['kl_lower_bound',
             'recorded',
             ['topk_intersection_lower_bound@1024'],
             'shared_with_reference',
             None,
             None,
             ['experiments/moe_greedy_smoke.py',
              'repetitive iff the completion ends in a cycle: some period p '
              'with 2p <= L has a p-periodic suffix holding >= 2 full periods '
              '(s >= 2p), whatever its share of the completion; not_recorded '
              'iff the completion is empty (L = 0), which is no completion '
              'for a verdict to be true of; recorded otherwise; tokens are '
              "the artifact tokenizer's canonical encoding of the returned "
              'text',
              'bf16_source',
              [['P0',
                'campaign',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P1',
                'campaign',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P2', 'campaign', 'raw_completion', 'recorded', 'repetitive'],
               ['P3',
                'campaign',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P4', 'campaign', 'chat_template', 'recorded', 'recorded'],
               ['P5', 'campaign', 'chat_template', 'recorded', 'recorded'],
               ['P6', 'campaign', 'chat_template', 'recorded', 'recorded'],
               ['P0',
                'pure_greedy',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P1',
                'pure_greedy',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P2',
                'pure_greedy',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P3',
                'pure_greedy',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P4', 'pure_greedy', 'chat_template', 'recorded', 'recorded'],
               ['P5', 'pure_greedy', 'chat_template', 'recorded', 'recorded'],
               ['P6',
                'pure_greedy',
                'chat_template',
                'recorded',
                'recorded']]]]],
           ['tessera_e4m3_k1_routed_moe_sm121_decode_resident',
            'sm_121',
            'TESSERA_E4M3_K1',
            'routed_moe',
            'decode',
            [1024],
            'fp8_per_token_dynamic',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident'],
            [['vllm.fused_moe.modular_kernel', 'torch_materialize_stock']],
            ['resident'],
            {'image': 'eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c',
             'execution_modes': ['eager']},
            '0.28.1rc1.dev397+gfd4a15126.d20260904',
            '2.13.0+cu130',
            ['route_only',
             'recorded',
             [],
             'shared_with_reference',
             None,
             None,
             ['experiments/moe_greedy_smoke.py',
              'repetitive iff the completion ends in a cycle: some period p '
              'with 2p <= L has a p-periodic suffix holding >= 2 full periods '
              '(s >= 2p), whatever its share of the completion; not_recorded '
              'iff the completion is empty (L = 0), which is no completion '
              'for a verdict to be true of; recorded otherwise; tokens are '
              "the artifact tokenizer's canonical encoding of the returned "
              'text',
              'bf16_source',
              [['P0',
                'campaign',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P1',
                'campaign',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P2', 'campaign', 'raw_completion', 'recorded', 'repetitive'],
               ['P3',
                'campaign',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P4', 'campaign', 'chat_template', 'recorded', 'recorded'],
               ['P5', 'campaign', 'chat_template', 'recorded', 'recorded'],
               ['P6', 'campaign', 'chat_template', 'recorded', 'recorded'],
               ['P0',
                'pure_greedy',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P1',
                'pure_greedy',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P2',
                'pure_greedy',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P3',
                'pure_greedy',
                'raw_completion',
                'repetitive',
                'repetitive'],
               ['P4', 'pure_greedy', 'chat_template', 'recorded', 'recorded'],
               ['P5', 'pure_greedy', 'chat_template', 'recorded', 'recorded'],
               ['P6',
                'pure_greedy',
                'chat_template',
                'recorded',
                'recorded']]]]]]}

#: Route statuses under which a cell says a native route EXECUTES.
_NATIVE_ROUTE_STATUSES = frozenset(
    {ROUTE_STATUS_BACKED, ROUTE_STATUS_BACKED_WITH_SERVE_FLAG}
)


#: What a published ``activation_contract`` string executes as, in the
#: vocabulary the allocator prices in: ``(act_bits, act_group_size)``.
#: A transcription of the runtime's vocabulary, in exactly one place, and
#: fail-closed: a contract string not listed here raises rather than
#: guessing, because guessing an A side is the currency error this table
#: exists to prevent (2026-08-17, 87 GB on NVFP4_CB).  Each leg is pinned
#: against the producer route for the family that publishes it -- the
#: ``tessera_serving_route`` contract quoted -- so the two halves of one
#: A side cannot drift apart without renaming one of them:
#:
#: * ``e2m1_group16_ue4m3_static``: 4-bit E2M1 activations in groups of 16
#:   with static scales -- ``w4a4-nvfp4-e2m1-group16-ue4m3``,
#:   ``(act_bits, act_group_size) == (4, 16)``.
#: * ``fp8_per_token_dynamic``: 8-bit FP8 activations with per-token dynamic
#:   scales -- ``w8a8-dynamic-e4m3-channel``, ``(8, 0)``: group 0 is the
#:   registry's spelling for ungrouped (the same one the ``FP8_E4M3`` row
#:   carries), and the dynamic half lives in the activation quantiser the
#:   route borrows by reference, not in the group size.
#: * ``bf16_unquantized``: 16-bit activations, no quantisation --
#:   ``w16a16-bf16-channel``, ``(16, 0)``.
_CELL_ACTIVATION_PROJECTION = {
    "e2m1_group16_ue4m3_static": (4, 16),
    "fp8_per_token_dynamic": (8, 0),
    "bf16_unquantized": (16, 0),
}


def cell_activation_projection(cell_contract: str) -> tuple[int, int]:
    """Project a cell's ``activation_contract`` onto ``(act_bits, act_group_size)``.

    The comparison :func:`prismaquant.tessera_menu.route_admission` runs:
    the producer prices ``tessera_serving_route``'s ``(act_bits,
    act_group_size)`` while the runtime executes the cell's string, and the
    two vocabularies do not match character for character.  Comparing the
    projection refuses a family whose priced A side is not the executed one.
    An unpublished string raises :class:`TesseraContractError` -- a new
    runtime vocabulary is a thing to transcribe in
    :data:`_CELL_ACTIVATION_PROJECTION`, never a thing to admit blind.
    """
    try:
        return _CELL_ACTIVATION_PROJECTION[str(cell_contract)]
    except KeyError:
        raise TesseraContractError(
            f"cell activation contract {cell_contract!r} is not a published "
            f"vocabulary this reader transcribes "
            f"({sorted(_CELL_ACTIVATION_PROJECTION)}); transcribing a new "
            "runtime vocabulary is a review, and admitting it blind would "
            "price an A side nothing executed"
        ) from None


@dataclass(frozen=True, slots=True)
class TesseraNativeExtension:
    """One ``native_extensions[]`` row, verbatim in the fields we read.

    The contract says what the plugin EXECUTES in ``formats`` /
    ``lane_eligibility``; this table says what it LOADS.  PrismaQuant needs it
    because §7.4 keys reproducibility on extension residency: a KL is
    bit-identical inside one container session and drifts 4-8x across them,
    keyed purely on whether a lane's ``.so`` was resident, so an A/B is
    comparable only across serves whose native-extension residency matches.
    A lane whose library no fingerprint pattern matches reports "nothing
    resident", and two serves the rule cannot tell apart get compared.

    ``match`` is the field that makes this principle 14 rather than a guess:
    the runtime names the RULE a consumer applies, so a consumer does not have
    to decide whether the published string is a stem, a prefix or a pattern.
    """

    #: The constant the runtime's JIT load path passes to
    #: ``cpp_extension.load``, trailing separator included.
    module_name_prefix: str
    #: The library name that produces.  A glob, because the module name
    #: carries a build-identity hash and no exact basename exists.
    filename_glob: str
    #: The rule that turns ``filename_glob`` into a decision.
    match: str
    #: Where the sources live in the runtime's tree -- identity, not answer.
    source: str
    #: The runtime module that loads it -- identity, not answer.
    loaded_by: str
    #: The routes that need it.
    routes: tuple[str, ...]
    #: Per residency mode, what a serve does when the library is absent:
    #: ``{"resident": {"status": "substituted", "decoder": "..."},
    #: "streamed": {"status": "refused", "decoder": None}}``.  This is what
    #: makes an absent ``.so`` readable: in one mode the serve keeps running on
    #: a NAMED substitute decoder and is a different numeric object, in the
    #: other there is no serve at all.
    when_unavailable: Mapping[str, Mapping[str, Any]]
    #: What the extension's kernel READS (contract v20): the decoder name it
    #: stamps and, when it publishes one, the predicate a unit's wire must
    #: satisfy for the lane to take it (``lane.requires``).  Parsed by
    #: ``lane_eligibility.parse_lane_claim`` -- the same reader the serving
    #: pin's table uses -- and decided by ``cell_lane_admits`` in
    #: :meth:`TesseraContract.native_cells`, so the development contract
    #: admits a rung on exactly the terms the pinned one does.
    lane: LaneClaim

    #: The four fields a residency predicate -- and the reading of an absent
    #: library -- is made of, as published. One spelling with the pin's row.
    as_contract_row = native_extension_contract_row


@dataclass(frozen=True, slots=True)
class TesseraRouteCell:
    """One ``lane_eligibility.cells[]`` row, verbatim in the fields we read."""

    cell_id: str
    platform: str
    family: str
    structure: str
    regime: str
    rungs_q256: frozenset[int]
    activation_contract: str
    route_status: str
    qualification: str
    requires_plugin: str
    requires_serve_flags: tuple[str, ...]
    executes: tuple[tuple[str, str], ...]
    residency_modes: tuple[str, ...]
    runtime_image: str
    execution_modes: tuple[str, ...]
    #: v6's per-cell runtime versions and evidence block; empty/``None`` under
    #: the pre-v6 grammars, which published neither.
    runtime_vllm: str = ""
    runtime_torch: str = ""
    evidence: "CellEvidence | None" = None

    @property
    def native(self) -> bool:
        """Does this cell attest a route the runtime executes natively?"""
        return (
            self.qualification == QUALIFICATION_DEVICE_QUALIFIED
            and self.route_status in _NATIVE_ROUTE_STATUSES
        )


@dataclass(frozen=True, slots=True)
class FusedModuleLicence:
    """``fused_module``: what one vLLM-fused module's roles must SHARE.

    The value a producer's group allocator reads instead of guessing.  vLLM
    merges q/k/v and gate/up into one Linear and builds ONE quant method per
    module, so everything that selects a method or a tile is a module fact;
    the RATE is not, because every decoder in the plugin is fed from each
    member's own parsed manifest.  Which of the two a field is is exactly what
    :attr:`fields` says, and it is checked on the Tessera side against
    ``tessera.serving.scheme.FUSED_MODULE_FIELDS`` -- the dict the loader
    itself gates on -- so the table cannot drift from the code.

    Read, never inferred.  ``prismaquant.allocator_candidates`` folds a fused
    group's menu over the fields this block marks ``per_member`` and holds
    every ``shared`` one fixed; a contract that re-tightens ``q256`` to
    ``shared`` therefore stops the fold rather than leaving it enumerating
    rungs the exporter refuses (prismaquant #132, RobTand/tessera#37).

    Every attribute here is a value a gate on this side decides on, which is
    what makes :meth:`answer` the block's whole projection into
    :func:`contract_answer`.  The block's ``container`` is deliberately not
    among them: nothing here reads the sidecar's magic, because there is no
    Tessera export leg to write one with.
    """

    #: The block's own schema id.  A gate: the reader refuses a block carrying
    #: another id rather than reading it as a subset of this one.
    schema: str
    #: ``field -> "shared" | "per_member"``, verbatim.
    fields: Mapping[str, str]
    #: How a mixed-rung module is SPELLED in the checkpoint's scheme.
    #: Recorded and carried into the fold's receipt; there is no writer yet.
    sidecar_q256: str
    #: Whether a container receipt covers a SERVED mixed-rung module.  False
    #: today: the relaxation is proven by a decode identity, not by a serve.
    #: A shipcard for an artifact that ships one has to say so.
    mixed_rung_receipt: bool

    def licence_for(self, field: str) -> str:
        """``"shared"``/``"per_member"`` for ``field``; an unpublished field raises.

        Absence is not permission.  A field the contract does not name is one
        this runtime has published nothing about, and guessing either way is
        the assertion principle 14 refuses.
        """
        try:
            return self.fields[str(field)]
        except KeyError:
            raise TesseraContractError(
                f"the Tessera contract's fused_module block publishes no "
                f"licence for {field!r}; it names "
                f"{sorted(self.fields)}. A field it does not name is not "
                "'probably per-member' -- ask Tessera to publish it."
            ) from None

    def is_per_member(self, field: str) -> bool:
        """Is ``field`` free to differ between one module's roles?"""
        return self.licence_for(field) == "per_member"

    def shared_fields(self) -> frozenset[str]:
        """Every field the module's roles must agree on."""
        return frozenset(
            name for name, licence in self.fields.items() if licence == "shared"
        )

    def per_member_fields(self) -> frozenset[str]:
        """Every field each role may hold on its own."""
        return frozenset(
            name for name, licence in self.fields.items()
            if licence == "per_member"
        )

    def answer(self) -> dict:
        """The gate-read projection, for :func:`contract_answer`.

        Derived from this object rather than from the JSON block, so the
        answer carries exactly what the reader kept and a field the reader
        does not parse cannot reach the pin.
        """
        return {
            "schema": str(self.schema),
            "fields": {str(k): str(v) for k, v in sorted(self.fields.items())},
            "sidecar_q256": str(self.sidecar_q256),
            "mixed_rung_receipt": bool(self.mixed_rung_receipt),
        }


@dataclass(frozen=True, slots=True)
class TesseraContract:
    """The plugin's packaged contract, parsed.  Every field is read, not typed."""

    #: ``family -> (lo, hi)`` inclusive q256 rate range the reader accepts.
    reader_rate_range: Mapping[str, tuple[int, int]]
    #: ``family -> the rungs a ``lane_eligibility`` cell attests``.
    attested_rungs: Mapping[str, frozenset[int]]
    cells: tuple[TesseraRouteCell, ...]
    #: The libraries the plugin loads into a serving process, in the table's
    #: order.  Non-empty by construction: :func:`_parse` refuses a contract
    #: that publishes no table rather than reading one as "loads nothing".
    native_extensions: tuple[TesseraNativeExtension, ...]
    #: ``family -> max tensor-parallel world size``, closed world.
    max_world_size: Mapping[str, int]
    #: ``family -> axis -> status`` from the same unit rows: what this build's
    #: LOADER does with a shard on each axis.  A different question from the
    #: ceiling beside it -- ``max_world_size`` says what a served receipt
    #: covers, this says whether the cut loads at all -- and a family the
    #: block does not list publishes neither.
    loader_axes: Mapping[str, Mapping[str, str]]
    #: What one vLLM-fused module's roles must share, and what is free.
    fused_module: FusedModuleLicence
    quant_method: str
    contract_version: int
    plugin_version: str
    #: ``versions.default_serve_image``: the ONE serve-image pin every harness
    #: reads. It replaced ``versions.attested_on`` at contract v17, which had
    #: made a single global claim about the runtime every cell was measured on
    #: -- false the moment one cell was measured on a dev wheel. The per-cell
    #: truth now lives on the cells (``runtime_image``/``runtime_vllm``/
    #: ``runtime_torch``); this is a default, and identity, not a gate input.
    default_serve_image: str
    #: Identity -- what travels into provenance.
    commit: str
    sha256: str
    path: str
    lane_schema: str
    regimes: tuple[str, ...]
    #: ``activation_contract -> the runtime's own quantiser table``, empty when
    #: the contract publishes none.  Empty is not "fine": it is what
    #: :func:`require_activation_quantizer_attested` REFUSES on, which is the
    #: only reading principle 14 allows for an unattested rounding rule.  The
    #: PARSER stays permissive so a contract published before the block existed
    #: still parses and every other gate on it keeps working; the PREFLIGHT is
    #: where absence bites, and it bites only the lane that is about to price
    #: an activation residual with PrismaQuant's own re-implementation.
    #:
    #: Three levels since Tessera contract v33 (activation-quantizer schema
    #: v2): ``[platform][image][activation_contract]``.  The middle level is
    #: the IMAGE the table was generated in -- its content digest when the
    #: contract names one, the reference verbatim when it does not -- because
    #: a platform may publish one table per serving image and the consumer
    #: admits a cell only under the attestation whose image is the executing
    #: one.  A v1 contract parses into the same shape with exactly one image
    #: level, so nothing below reads two schemas.
    activation_quantizers: Mapping[str, Mapping[str, Mapping[
        str, "ActivationQuantizerAttestation"]]] = field(default_factory=dict)

    @property
    def requires_serving_context(self) -> bool:
        return self.lane_schema in SCOPED_LANE_SCHEMAS

    def governs(self, family: str) -> bool:
        """Does the contract publish this payload family at all?"""
        return str(family) in self.reader_rate_range

    def native_cells(self, family: str, rate_q256: int, *,
                     serving_context: ServingContext | None = None,
                     ) -> tuple[TesseraRouteCell, ...]:
        """V5 admits every required regime only under one explicit context.

        Context-free v4 keeps its historical family/rate projection. An
        explicit runtime query cannot borrow that unscoped claim. V5 never uses
        the first matching cell to infer an image, execution mode or structure.
        """
        if self.requires_serving_context and serving_context is None:
            return ()
        if not self.requires_serving_context and serving_context is not None:
            return ()
        selected = tuple(
            cell for cell in self.cells
            if cell.family == str(family)
            and int(rate_q256) in cell.rungs_q256
            and cell.native
            # The development menu reads the SAME evidence predicate the
            # export gate reads (``lane_eligibility.cell_evidence_admits``).
            # A rung the dev menu offers and the export refuses is the
            # split-brain principle 8 exists to stop.
            and cell_evidence_admits(cell)[0]
            # And the SAME lane predicate: the lane a cell launches through
            # must read what this producer plans at the rung
            # (``lane_eligibility.cell_lane_admits``, deciding
            # ``native_extensions[].lane.requires`` over
            # ``tessera_render.planned_wire_facts``).
            and cell_lane_admits(cell, int(rate_q256), self.lanes)[0]
            and (not self.requires_serving_context
                 or cell_matches_serving_context(cell, serving_context))
        )
        if self.requires_serving_context and {cell.regime for cell in selected} != set(self.regimes):
            return ()
        return selected

    @property
    def lanes(self) -> tuple[LaneClaim, ...]:
        """Every extension's lane claim, in table order, for the lane gate."""
        return tuple(ext.lane for ext in self.native_extensions)

    def identity(self) -> dict:
        """The ``tessera_dev_pin`` provenance block.

        Records the review *and* the read: which Tessera build and bytes a
        human accepted this answer against, which bytes this run actually
        consumed, and whether the two are the same file.  They can differ
        legitimately -- the gate is the answer, not the bytes -- and a
        shipcard that could not tell the two apart would be asserting a
        review it did not get.
        """
        return {
            "requested": dev_pin_requested(),
            "commit": self.commit,
            "contract_sha256": self.sha256,
            "reviewed_contract_sha256": TESSERA_DEV_PIN_CONTRACT_SHA256,
            "bytes_are_the_reviewed_bytes":
                self.sha256 == TESSERA_DEV_PIN_CONTRACT_SHA256,
            "contract_path": self.path,
            "schema": TESSERA_CONTRACT_SCHEMA,
            "contract_version": self.contract_version,
            "plugin_version": self.plugin_version,
            "quant_method": self.quant_method,
            "default_serve_image": self.default_serve_image,
            "native_extensions": [
                {
                    "module_name_prefix": ext.module_name_prefix,
                    "filename_glob": ext.filename_glob,
                    "match": ext.match,
                    "source": ext.source,
                    "loaded_by": ext.loaded_by,
                    "routes": list(ext.routes),
                    "when_unavailable": {
                        mode: dict(behaviour)
                        for mode, behaviour in sorted(
                            ext.when_unavailable.items())
                    },
                }
                for ext in self.native_extensions
            ],
            "note": (
                "development override: this allocation's Tessera routes "
                "were admitted by the packaged "
                "contract, whose ANSWER (every value the admission gate "
                "reads) equals the one reviewed at the named commit. Its "
                "scope is admission: the export lane's structures/platforms/"
                "regimes require the separate exact serving pin. This development "
                "contract need not have the reviewed bytes; "
                "bytes_are_the_reviewed_bytes says which"
            ),
        }


def describe_dev_pin(identity: "Mapping[str, Any]") -> str:
    """One human-readable line for a :meth:`TesseraContract.identity` block.

    Since the pin became an *answer* pin, bytes that differ from the reviewed
    ones are legal: a Tessera commit adding a block no gate reads moves the
    file and not the answer, and the answer is what admitted the routes.  But
    that makes ``bytes_are_the_reviewed_bytes`` the one field a reader needs
    and the one the old log line left out -- it printed a sha with nothing to
    compare it against, so "these are not the bytes a human reviewed" existed
    only inside a provenance blob nobody opens.  Name it here.
    """

    line = (f"commit={identity['commit'][:12]} "
            f"contract_sha={identity['contract_sha256'][:12]}")
    if not identity["bytes_are_the_reviewed_bytes"]:
        line += (f" (reviewed {identity['reviewed_contract_sha256'][:12]}, "
                 "answer equal)")
    return (f"{line} plugin={identity['plugin_version']} "
            f"contract_v{identity['contract_version']}")


def contract_answer(contract: "TesseraContract") -> dict:
    """Exactly the values the gates on THIS pin decide on, canonicalised, and no more.

    Two gates read the dev pin: admission -- which ``(family, rate)`` the
    allocator may put on the menu and under which route status -- and the
    serve fingerprint's native-extension residency (prismaquant #133).  The
    answer is their union and nothing else.  It is deliberately NOT every
    value any PrismaQuant gate reads out of this file: ``lane_eligibility.structures``,
    ``platforms`` and ``regimes`` are read by the EXPORT lane's own reader
    (``tessera_export_lane.require_declared_structure`` through
    ``lane_eligibility.load_eligibility_table``), which is gated by the
    RELEASE pin -- an exact reviewed commit and contract SHA, refusing
    unresolved sentinels or mismatched installed contract bytes.  Pulling them in here would make an export-lane edit re-stale
    the allocator's menu, which is issue #38's own failure mode wearing a
    different hat.  Each pin covers the values its own gates read.

    Principle 14's line, made mechanical.  ``detail``, ``rationale``, the
    changelog and every other prose field explains; none of them is a value a
    gate reads, so none of them appears here.  Neither do ``contract_version``,
    ``plugin_version`` or ``default_serve_image``: those are the table's *identity*,
    which travels into provenance, and a version bump that moved no answer is
    not a thing to re-review.

    What is here is what a GATE decides on.  Most of it is the admission
    decision -- which families exist, the rate range the decoder accepts for
    each, the rungs a cell attests, the tensor-parallel ceiling, the canonical
    ``quant_method`` this producer writes into the checkpoint, and every cell
    field the route gate reads.  Two contracts with the same answer admit the
    same units.

    Schema v4's executed launches and residency scope are retained per cell.
    They identify which serving path the route claim covers; a changed
    decoder or narrowed residency therefore requires the same review as a
    changed rung, even when the cell's id and route status stay unchanged.

    ``native_extensions`` is here for a second gate, not the admission one:
    §7.4 says an A/B's arms must have identical native-extension residency,
    ``tools/serve_fingerprint.py`` decides residency by matching mapped
    libraries, and the prefix, the glob and the ``match`` rule are the values
    that decision is made of -- with ``when_unavailable`` the value that says
    what an ABSENT library means (a named substitute decoder, or no serve at
    all).  Those move the fingerprint's behaviour, so they are answer.
    ``source`` and ``loaded_by`` name files and modules in the runtime's own
    tree and move nothing on this side, so they are identity and stay out,
    exactly like ``plugin_version``.  ``lane`` (contract v20) IS answer: its
    ``decoder`` is what binds a cell's launch to the extension, and its
    ``requires`` is the predicate ``native_cells`` and the export gate
    decide this producer's plan against -- a lane that widens or narrows
    what it reads changes which rungs are admitted, and is a re-review.

    ``fused_module`` is here for a third (prismaquant #132): the group
    knapsack's fold reads it, so a contract that re-tightened ``q256`` to
    ``shared`` would change what this producer may allocate.  It carries
    exactly the block's values this reader parses -- see
    :class:`FusedModuleLicence` -- which is why ``container`` is NOT here.
    Nothing on this side reads the sidecar's magic (there is no Tessera export
    leg to write one with), and a value in the answer that no gate reads makes
    a re-review out of a field nobody uses.  The block's three ``*_note`` keys
    are prose and stay out for the same reason ``rationale`` does.
    """
    return {
        "schema": TESSERA_CONTRACT_SCHEMA,
        "lane_schema": contract.lane_schema,
        **({"required_regimes": sorted(contract.regimes)}
           if contract.requires_serving_context else {}),
        "quant_method": contract.quant_method,
        "fused_module": contract.fused_module.answer(),
        "native_extensions": [
            {
                "module_name_prefix": ext.module_name_prefix,
                "filename_glob": ext.filename_glob,
                "match": ext.match,
                "routes": sorted(ext.routes),
                "when_unavailable": {
                    mode: {"status": behaviour["status"],
                           "decoder": behaviour["decoder"]}
                    for mode, behaviour in sorted(ext.when_unavailable.items())
                },
                "lane": ext.lane.answer(),
            }
            for ext in sorted(contract.native_extensions,
                              key=lambda e: e.module_name_prefix)
        ],
        "activation_quantizers": [
            contract.activation_quantizers[platform][image][name].answer()
            for platform in sorted(contract.activation_quantizers)
            for image in sorted(contract.activation_quantizers[platform])
            for name in sorted(contract.activation_quantizers[platform][image])
        ],
        "families": {
            family: {
                "reader_rate_range_q256": [int(rng[0]), int(rng[1])],
                "attested_rungs_q256": sorted(
                    int(r) for r in contract.attested_rungs.get(family, ())),
                "max_world_size": int(contract.max_world_size.get(family, 0)),
                "loader_axes": {
                    str(axis): str(status) for axis, status in
                    sorted(contract.loader_axes.get(family, {}).items())
                },
            }
            for family, rng in sorted(contract.reader_rate_range.items())
        },
        "cells": sorted(
            [
                cell.cell_id,
                cell.platform,
                cell.family,
                cell.structure,
                cell.regime,
                sorted(int(r) for r in cell.rungs_q256),
                cell.activation_contract,
                cell.route_status,
                cell.qualification,
                cell.requires_plugin,
                sorted(cell.requires_serve_flags),
                [list(launch) for launch in sorted(cell.executes)],
                sorted(cell.residency_modes),
            ] + ([{"image": cell.runtime_image,
                   "execution_modes": sorted(cell.execution_modes)}]
                 if contract.requires_serving_context else [])
            # v6's per-cell runtime versions and evidence. Both are ANSWER,
            # not identity: ``cell_evidence_admits`` decides on the evidence
            # block, so a flipped smoke status or a new KL kind changes which
            # units this producer may put on the menu -- and a re-review is
            # exactly what should stand between Tessera recording a smoke and
            # PrismaQuant admitting the route it names. The versions are here
            # for the same reason the image is: they scope the claim.
            + ([cell.runtime_vllm, cell.runtime_torch, cell.evidence.answer()]
               if cell.evidence is not None else [])
            for cell in contract.cells
        ),
    }


def _answer_drift(reviewed: Mapping[str, Any], installed: Mapping[str, Any]
                  ) -> list[str]:
    """Field-level lines naming what moved, so the refusal is reviewable."""
    lines: list[str] = []
    for key in ("schema", "lane_schema", "quant_method", "required_regimes"):
        if reviewed.get(key) != installed.get(key):
            lines.append(
                f"  {key}: reviewed {reviewed.get(key)!r}, installed "
                f"{installed.get(key)!r}")
    r_fused = dict(reviewed.get("fused_module", {}))
    i_fused = dict(installed.get("fused_module", {}))
    r_lic = dict(r_fused.pop("fields", {}) or {})
    i_lic = dict(i_fused.pop("fields", {}) or {})
    for key in sorted(set(r_fused) | set(i_fused)):
        if r_fused.get(key) != i_fused.get(key):
            lines.append(
                f"  fused_module.{key}: reviewed {r_fused.get(key)!r}, "
                f"installed {i_fused.get(key)!r}")
    for field in sorted(set(r_lic) | set(i_lic)):
        if field not in r_lic:
            lines.append(
                f"  fused_module.fields[{field}]: NEW ({i_lic[field]!r}), not "
                "in the reviewed answer")
        elif field not in i_lic:
            lines.append(
                f"  fused_module.fields[{field}]: GONE from the installed "
                f"contract (reviewed {r_lic[field]!r})")
        elif r_lic[field] != i_lic[field]:
            lines.append(
                f"  fused_module.fields[{field}]: reviewed "
                f"{r_lic[field]!r}, installed {i_lic[field]!r}")
    r_fam, i_fam = reviewed.get("families", {}), installed.get("families", {})
    for family in sorted(set(r_fam) | set(i_fam)):
        if family not in r_fam:
            lines.append(f"  families[{family}]: NEW, not in the reviewed answer")
        elif family not in i_fam:
            lines.append(f"  families[{family}]: GONE from the installed contract")
        elif r_fam[family] != i_fam[family]:
            for k in sorted(set(r_fam[family]) | set(i_fam[family])):
                if r_fam[family].get(k) != i_fam[family].get(k):
                    lines.append(
                        f"  families[{family}].{k}: reviewed "
                        f"{r_fam[family].get(k)!r}, installed "
                        f"{i_fam[family].get(k)!r}")
    r_ext = {row["module_name_prefix"]: row
             for row in reviewed.get("native_extensions", ())}
    i_ext = {row["module_name_prefix"]: row
             for row in installed.get("native_extensions", ())}
    for prefix in sorted(set(r_ext) | set(i_ext)):
        if prefix not in r_ext:
            lines.append(
                f"  native_extensions[{prefix}]: NEW, not in the reviewed "
                "answer -- a library a serve can map that no fingerprint on "
                "this side was reviewed against")
        elif prefix not in i_ext:
            lines.append(
                f"  native_extensions[{prefix}]: GONE from the installed "
                "contract")
        elif r_ext[prefix] != i_ext[prefix]:
            for key in sorted(set(r_ext[prefix]) | set(i_ext[prefix])):
                if r_ext[prefix].get(key) != i_ext[prefix].get(key):
                    lines.append(
                        f"  native_extensions[{prefix}].{key}: reviewed "
                        f"{r_ext[prefix].get(key)!r}, installed "
                        f"{i_ext[prefix].get(key)!r}")
    # Keyed by (platform, contract name, generated image): a platform may
    # publish one table per serving image since Tessera contract v33, and a
    # two-field key would fold them onto each other -- the second image's
    # table would then drift with nothing to compare it against.
    r_quant = {(row[0], row[1], row[2]): row
               for row in reviewed.get("activation_quantizers", ())}
    i_quant = {(row[0], row[1], row[2]): row
               for row in installed.get("activation_quantizers", ())}
    for name in sorted(set(r_quant) | set(i_quant)):
        if name not in r_quant:
            lines.append(
                f"  activation_quantizers[{name}]: NEW, not in the reviewed "
                "answer -- a rounding rule this producer has not read")
        elif name not in i_quant:
            lines.append(
                f"  activation_quantizers[{name}]: GONE from the installed "
                "contract -- a quantiser that was attested is now asserted")
        elif list(r_quant[name]) != list(i_quant[name]):
            lines.append(
                f"  activation_quantizers[{name}]: reviewed {r_quant[name]!r}, "
                f"installed {i_quant[name]!r}")
    r_cells = {tuple(c[:1])[0]: c for c in reviewed.get("cells", ())}
    i_cells = {tuple(c[:1])[0]: c for c in installed.get("cells", ())}
    for cell_id in sorted(set(r_cells) | set(i_cells)):
        if cell_id not in r_cells:
            lines.append(f"  cells[{cell_id}]: NEW, not in the reviewed answer")
        elif cell_id not in i_cells:
            lines.append(f"  cells[{cell_id}]: GONE from the installed contract")
        elif list(r_cells[cell_id]) != list(i_cells[cell_id]):
            lines.append(
                f"  cells[{cell_id}]: reviewed {r_cells[cell_id]!r}, installed "
                f"{i_cells[cell_id]!r}")
    return lines


def dev_pin_requested() -> str:
    """The commit the environment asks for, or ``""``."""
    return str(os.environ.get(TESSERA_DEV_PIN_ENV, "")).strip()


def contract_path():
    """Where the packaged contract lives: the actually-importable package's table.

    ``importlib.resources.files("tessera.serving")`` rather than path
    arithmetic on ``tessera.__file__``, so a wheel install, an editable
    install and an in-repo checkout all resolve identically -- and so this
    reads the table of the ``tessera.serving`` package that is actually
    importable, never a copy.  This is the one resolver both producer readers
    share: ``tessera_render.tessera_serving_contract_path`` delegates here
    rather than carrying a second policy for the same file.

    That call does import the ``tessera.serving`` package, but the import is
    cheap by the runtime's own design: its ``__init__`` defines ``register()``
    (which imports vLLM and registers the config) and calls nothing at module
    scope, importing neither torch nor vLLM, so locating the contract
    registers nothing and needs no GPU.
    ``tests/test_tessera_serving_contract_path.py`` pins that property in a
    subprocess rather than in prose, so drift in the runtime's import
    behaviour fails a test instead of silently aging this docstring.  What a
    producer must not need is the serving-side *code*:
    ``tessera.serving.contract``'s validator imports the plugin's dispatch
    tables, which is a serving-side import a producer must not need on a
    machine with no GPU -- so the JSON is read directly rather than through
    that module.
    """
    from importlib import resources

    return resources.files("tessera.serving").joinpath("runtime_contract.json")


def _require(block: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in block:
        raise TesseraContractError(f"{where} publishes no {key!r}")
    return block[key]


#: The one ``native_extensions[].match`` rule this reader implements.  A
#: contract naming another rule is REFUSED rather than read with this one:
#: the whole reason ``match`` is a value is that the predicate is not
#: guessable from the glob.
MATCH_BASENAME_FNMATCH = "basename_fnmatch"

_NATIVE_EXTENSION_MEMBERS = (
    "module_name_prefix", "filename_glob", "match", "source", "loaded_by",
    "routes", "when_unavailable", "lane",
)


def _parse_native_extensions(
    entries: Any, *, where: str
) -> tuple[TesseraNativeExtension, ...]:
    """Read ``native_extensions``, refusing anything a fingerprint can't use.

    Published since Tessera contract v7.  An older table does not carry it,
    and the honest answer there is a refusal: "this contract does not say what
    the plugin loads" is not the same statement as "the plugin loads nothing",
    and reading the second from the first is how a Tessera serve came to
    fingerprint as a stock serve in the first place.
    """
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TesseraContractError(f"{where} must be a JSON array")
    if not entries:
        raise TesseraContractError(
            f"{where} is empty. A contract that publishes no loadable library "
            "makes every serve fingerprint identical on the one axis §7.4 "
            "keys reproducibility on, so an empty table is refused rather "
            "than read as 'loads nothing'."
        )
    parsed: list[TesseraNativeExtension] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        at = f"{where}[{i}]"
        if not isinstance(entry, Mapping):
            raise TesseraContractError(f"{at} must be a JSON object")
        for member in _NATIVE_EXTENSION_MEMBERS:
            _require(entry, member, at)
        prefix = str(entry["module_name_prefix"])
        if not prefix:
            raise TesseraContractError(
                f"{at}.module_name_prefix must be a non-empty string")
        if prefix in seen:
            raise TesseraContractError(
                f"{at}.module_name_prefix {prefix!r} is declared twice")
        seen.add(prefix)
        rule = str(entry["match"])
        if rule != MATCH_BASENAME_FNMATCH:
            raise TesseraContractError(
                f"{at}.match is {rule!r}; this reader implements only "
                f"{MATCH_BASENAME_FNMATCH!r} (fnmatch the glob against the "
                "BASENAME of a mapped .so) and will not apply that predicate "
                "to a rule it does not know. The rule is a published value "
                "precisely because it is not guessable from the glob."
            )
        glob = str(entry["filename_glob"])
        # By MEANING, not by spelling, the same way the runtime's own
        # validator checks it: a library name this load path can produce must
        # match. A glob that matches nothing a serve maps is a fingerprint
        # that reports every serve identical.
        if not fnmatch.fnmatch(f"{prefix}0123456789abcdef.so", glob):
            raise TesseraContractError(
                f"{at}.filename_glob {glob!r} matches no library name the "
                f"load path can produce ({prefix}<build identity>.so)"
            )
        when = entry["when_unavailable"]
        if not isinstance(when, Mapping) or not when:
            raise TesseraContractError(
                f"{at}.when_unavailable must be a non-empty object keyed by "
                "residency mode"
            )
        behaviours: dict[str, Mapping[str, Any]] = {}
        for mode, behaviour in when.items():
            if not isinstance(behaviour, Mapping):
                raise TesseraContractError(
                    f"{at}.when_unavailable[{mode!r}] must be an object")
            _require(behaviour, "status", f"{at}.when_unavailable[{mode!r}]")
            _require(behaviour, "decoder", f"{at}.when_unavailable[{mode!r}]")
            behaviours[str(mode)] = {
                "status": str(behaviour["status"]),
                "decoder": (None if behaviour["decoder"] is None
                            else str(behaviour["decoder"])),
            }
        routes = entry["routes"]
        if (not isinstance(routes, Sequence)
                or isinstance(routes, (str, bytes)) or not routes):
            raise TesseraContractError(
                f"{at}.routes must name at least one route that needs it")
        # The lane block is read by the serving pin's reader, so the two
        # contracts a producer holds -- the pinned one and this development
        # one -- refuse the same block for the same reason, once.
        try:
            lane = parse_lane_claim(entry["lane"], f"{at}.lane", extension=prefix)
        except LaneEligibilityError as exc:
            raise TesseraContractError(str(exc)) from exc
        parsed.append(TesseraNativeExtension(
            module_name_prefix=prefix,
            filename_glob=glob,
            match=rule,
            source=str(entry["source"]),
            loaded_by=str(entry["loaded_by"]),
            routes=tuple(str(r) for r in routes),
            when_unavailable=behaviours,
            lane=lane,
        ))
    return tuple(parsed)


def require_pin_native_extensions_match_contract(
    contract: "TesseraContract",
    pin: "Any | None" = None,
) -> None:
    """Refuse a pin whose extension table is not the pinned contract's.

    The middle link of the §7.4 chain.  ``tools/serve_fingerprint.py`` runs
    inside a serving container from a bootstrap with no installed package, so
    it can read neither this contract nor the pin's reader module: it reads
    the transported pin JSON beside itself -- a member of its gold-producer
    source closure -- and ``tests/test_tessera_serve_fingerprint.py`` refuses
    a tool that does not read the pin.  That test was the ONLY refusal in the
    chain, and the link it checked was the one that was already sound -- the
    pin itself was a hand-written claim about another runtime, maintained one
    repository over, where nothing here could refuse it on drift.  Rename the
    extension in the plugin and a Tessera serve fingerprinted as "nothing
    resident", which is the hole that existed before 2026-09-03, when the
    pattern named no Tessera library at all.

    Compared over the fields a residency predicate -- and the reading of an
    absent library -- is MADE of -- prefix, glob, the match rule, and what
    runs when the library is absent (``when_unavailable``) -- keyed by prefix,
    in both directions: a library the contract publishes and the pin omits
    makes the fingerprint go quietly short, and a library the pin invents is
    a claim about a runtime that does not load it.  A substitute decoder the
    pin mistranscribes names the wrong fallback in the §7.4 refusal, so the
    block is compared value for value, not merely for presence.
    """
    from .tessera_serving_runtime_pin import (
        TesseraServingRuntimePinError,
        load_tessera_serving_runtime_pin,
    )

    if pin is None:
        pin = load_tessera_serving_runtime_pin()
    published = {row["module_name_prefix"]: row
                 for row in (ext.as_contract_row()
                             for ext in contract.native_extensions)}
    pinned = {row["module_name_prefix"]: row
              for row in pin.native_extension_rows()}
    lines: list[str] = []
    for prefix in sorted(set(published) | set(pinned)):
        if prefix not in pinned:
            lines.append(
                f"  {prefix!r}: the contract publishes it and the pin omits "
                "it -- a serve loading it would fingerprint as nothing "
                "resident")
        elif prefix not in published:
            lines.append(
                f"  {prefix!r}: the pin declares it and the contract "
                "publishes no such extension")
        elif published[prefix] != pinned[prefix]:
            for key in sorted(set(published[prefix]) | set(pinned[prefix])):
                if published[prefix].get(key) != pinned[prefix].get(key):
                    lines.append(
                        f"  {prefix!r}.{key}: contract "
                        f"{published[prefix].get(key)!r}, pin "
                        f"{pinned[prefix].get(key)!r}")
    if lines:
        raise TesseraServingRuntimePinError(
            "The Tessera serving pin's serving_native_extensions is not the "
            f"pinned contract's native_extensions table ({contract.path}):\n"
            + "\n".join(lines) + "\n"
            "The pin is a TRANSCRIPTION of that table (principle 14: a claim "
            "about another runtime is attested, never asserted), and "
            "tools/serve_fingerprint.py carries the same rows because it "
            "cannot read either file from inside a serving container. Fix the "
            "pin, and the tool's TESSERA_NATIVE_EXTENSIONS with it, in one "
            "commit."
        )

#: The ``activation_quantizers`` grammars this reader implements.  A block
#: naming any other schema is REFUSED rather than read with one of these: the
#: whole point of the block is that nothing here guesses at a runtime's
#: arithmetic, and guessing at the shape it published it in is the same
#: mistake one level up.
#:
#: v1 (Tessera contract v25) publishes ``platforms[p]`` as one attestation
#: object.  v2 (Tessera contract v33, tessera#555) publishes a LIST of them,
#: one per serving image, because the fp4 rounding decision belongs to the
#: runtime's compiled operator and two builds of one operator are two objects.
#: Both are read here, into the same three-level table: a v1 block is a v2
#: list of one.  What the list costs is that "which attestation" stops being
#: obvious, and the answer is never "the first one" -- see
#: :func:`require_activation_quantizer_attested`.
ACTIVATION_QUANTIZER_SCHEMA_V1 = "tessera.activation-quantizer.v1"
ACTIVATION_QUANTIZER_SCHEMA_V2 = "tessera.activation-quantizer.v2"
#: Kept as the v1 spelling for callers that import it by its old name.
ACTIVATION_QUANTIZER_SCHEMA = ACTIVATION_QUANTIZER_SCHEMA_V1
ACTIVATION_QUANTIZER_SCHEMAS = (ACTIVATION_QUANTIZER_SCHEMA_V1,
                                ACTIVATION_QUANTIZER_SCHEMA_V2)

#: The vocabulary of one ``contracts[]`` entry this reader transcribes.  Each
#: is a fact about what the published vectors MEAN, and a different value is a
#: different quantiser -- so each is compared, never defaulted.
_ATTESTED_UNIT = "group"
_ATTESTED_GRID = "E2M1"
_ATTESTED_BLOCK_SCALE = "UE4M3"
_ATTESTED_OP = "torch.ops._C.scaled_fp4_quant"
_ATTESTED_GLOBAL_SCALE = "static_per_module"

_ACTIVATION_CONTRACT_MEMBERS = (
    "op", "unit", "unit_length", "grid", "block_scale", "global_scale",
    "vectors",
)
_ACTIVATION_VECTOR_MEMBERS = (
    "id", "boundary", "global_scale", "input", "stored_scale", "codes",
)
#: The vocabulary of ``platforms[p]``.  ``generated`` is the scope the table
#: was taken under and is read here (RobTand/prismaquant#715); a third member
#: is a review, for the same reason a contract entry's is.
_ACTIVATION_PLATFORM_MEMBERS = ("contracts", "generated")
#: Every field of ``platforms[p].generated``.  All of them, exactly: a table
#: that names its box but not its build, or its build but not its image, does
#: not say what a consumer has to compare against.
_ACTIVATION_GENERATED_MEMBERS = (
    "image", "vllm", "torch", "device", "compute_capability", "driver",
    "generator_sha256",
)


@dataclass(frozen=True, slots=True)
class ActivationQuantizerVector:
    """One probe group the runtime ran its kernel on, verbatim.

    Bit patterns rather than decimal literals, because the disagreement this
    table settles lives at ties: a decimal that re-parses one ULP off the
    midpoint tests the arithmetic instead of the rounding RULE, and would make
    the attestation pass or fail on the JSON writer's formatting.

    A whole group of ``unit_length`` values, not one element, which is what
    lets this attest BOTH halves of the quantiser -- the ``amax -> UE4M3``
    scale the kernel stored, and the code each element became under it.
    """

    #: The generator's name for this probe, and the boundary it covers.
    #: Identity: the coverage check below is numeric, so a mislabelled probe
    #: cannot buy coverage it does not reach.
    id: str
    boundary: str
    #: IEEE-754 binary32 bits of the module's ``trellis_input_global_scale``.
    global_scale_bits: int
    #: bfloat16 bit patterns, one per element of the group.
    input_bits: tuple[int, ...]
    #: The ``float8_e4m3fn`` byte the kernel STORED for this group.
    stored_scale_byte: int
    #: The codes the kernel emitted, read back from its packed output: bit 3
    #: the sign, bits 0..2 the index into the positive E2M1 magnitudes.
    codes: tuple[int, ...]

    def as_row(self) -> list:
        return [self.id, self.boundary, self.global_scale_bits,
                list(self.input_bits), self.stored_scale_byte, list(self.codes)]


@dataclass(frozen=True, slots=True)
class ActivationQuantizerGeneration:
    """``activation_quantizers.platforms[p].generated``: the table's scope.

    The vectors say what the kernel emitted.  This says WHICH kernel: the
    image the runtime ran to emit them, and the build inside it.  The two are
    one claim -- principle 14's corollary is that a capability claim inherits
    the scope of the artifact it was measured on -- and until #715 this
    producer read the vectors and dropped the scope, so a cell measured in
    another build of the same operator carried an attestation that did not
    cover it.

    Published at the PLATFORM level, not per contract: #715's text quotes it
    under ``contracts[...]``, but the pinned bytes (``db9ca4c0…``, Tessera
    ``4c384e6049``) and master both publish it beside ``contracts``, one
    generation for every contract the platform attests.  Each row for the
    platform carries it, which is the same fact addressed the way a consumer
    reads it.
    """

    image: str
    vllm: str
    torch: str
    device: str
    compute_capability: str
    driver: str
    generator_sha256: str

    def as_stamp(self) -> dict:
        """The scope, as a producer freezes it beside the attestation."""
        return {field: getattr(self, field)
                for field in _ACTIVATION_GENERATED_MEMBERS}


@dataclass(frozen=True, slots=True)
class ActivationQuantizerAttestation:
    """One ``activation_quantizers.platforms[p].contracts[c]`` entry.

    What the name-only ``activation_contract`` field never said.  The string
    ``"e2m1_group16_ue4m3_static"`` names a quantiser; it does not say what
    that quantiser does at a tie, at the top of the lattice, or at the block
    scale's underflow, and PrismaQuant's priced activation residual is the
    output of its own re-implementation of exactly those decisions
    (``nvfp4_activation_contract.nvfp4_activation_qdq_served``).  Two
    implementations of one name are two objects, and on 2026-09-13 they were
    measured to be: single E2M1 code flips on ~3.4% of one tensor, 0.0957 and
    0.1436 of divergence on the activation representation, while the GEMM gate
    passed on both.  This is the table that makes the rule attested rather than
    asserted.
    """

    platform: str
    activation_contract: str
    op: str
    unit: str
    unit_length: int
    grid: str
    block_scale: str
    #: How the global scale is supplied, as the runtime names it.
    global_scale: str
    vectors: tuple[ActivationQuantizerVector, ...]
    #: The platform's ``generated`` block, or ``None`` when the table
    #: publishes none.  ``None`` is not a pass: it travels into the stamp as
    #: an explicit absence and the consumer refuses an unscoped attestation
    #: (RobTand/prismaquant#715).
    generated: "ActivationQuantizerGeneration | None" = None

    def answer(self) -> list:
        """The gate-read projection, for :func:`contract_answer`.

        Every vector is in it.  A table that silently dropped the 1.75 midpoint
        would admit the same families while attesting strictly less, so the
        vectors ARE answer, not identity.  ``op`` is in it too: the same table
        produced by a different symbol is a different claim about what a serve
        runs.

        ``generated.image`` joined it at Tessera contract v33 (schema v2),
        when a platform started publishing one table per serving image: the
        image is what SELECTS which table covers a measurement
        (:func:`require_activation_quantizer_attested`), so it is a value an
        admission decision is made of rather than provenance -- and without it
        two byte-identical tables project two identical rows, which the
        answer's drift key would then read as one.  The rest of ``generated``
        stays out: ``vllm`` and ``torch`` are inside the image the digest
        already pins, ``device``, ``compute_capability`` and ``driver`` are
        host facts nothing here compares, and ``generator_sha256`` names the
        script.  ``generator`` stays out for the same reason.
        """
        return [self.platform, self.activation_contract,
                self.generated.image if self.generated else None,
                self.op, self.unit,
                int(self.unit_length), self.grid, self.block_scale,
                self.global_scale, [v.as_row() for v in self.vectors]]


def _parse_activation_quantizers(payload: Mapping[str, Any], path: str
                                 ) -> dict[str, dict[str, ActivationQuantizerAttestation]]:
    """Read ``activation_quantizers``, or read that there is none.

    Absence is permitted HERE and refused in
    :func:`require_activation_quantizer_attested`.  The split is deliberate:
    every other gate on this contract -- admission, the fingerprint, the fused
    licence -- was sound before this block existed, and making the parser
    refuse would take them all down to fix an activation-pricing defect they
    have nothing to do with.  What must never happen is the other reading: a
    missing block silently satisfying a check.
    """
    block = payload.get("activation_quantizers")
    if block is None:
        return {}
    where = f"{path}.activation_quantizers"
    if not isinstance(block, Mapping):
        raise TesseraContractError(f"{where} must be a JSON object")
    schema = block.get("schema")
    if schema not in ACTIVATION_QUANTIZER_SCHEMAS:
        raise TesseraContractError(
            f"{where}.schema is {schema!r}; this reader implements only "
            f"{list(ACTIVATION_QUANTIZER_SCHEMAS)}. A quantiser table in a "
            "grammar this reader has not been taught is a review, not a thing "
            "to read with the grammar it happens to have.")
    platforms = _require(block, "platforms", where)
    if not isinstance(platforms, Mapping):
        raise TesseraContractError(f"{where}.platforms must be a JSON object")
    table: dict[str, dict[str, dict[str, ActivationQuantizerAttestation]]] = {}
    for platform, published in platforms.items():
        spot = f"{where}.platforms[{platform}]"
        if schema == ACTIVATION_QUANTIZER_SCHEMA_V2:
            if not isinstance(published, Sequence) or isinstance(
                    published, (str, bytes, Mapping)):
                raise TesseraContractError(
                    f"{spot} must be a JSON array under "
                    f"{ACTIVATION_QUANTIZER_SCHEMA_V2!r}: v2 publishes one "
                    "attestation per serving image, and a reader that also "
                    "accepted the v1 object here would be guessing which "
                    "grammar it was handed.")
            if not published:
                raise TesseraContractError(
                    f"{spot} is an empty array. A platform that publishes no "
                    "attestation at all says nothing this reader can price "
                    "against; publish one or publish no platform entry.")
            entries = [(f"{spot}[{i}]", item)
                       for i, item in enumerate(published)]
        else:
            if not isinstance(published, Mapping):
                raise TesseraContractError(f"{spot} must be a JSON object")
            entries = [(spot, published)]
        by_image: dict[str, dict[str, ActivationQuantizerAttestation]] = {}
        for at, entry in entries:
            key, rows = _parse_activation_platform_entry(
                entry, platform=str(platform), where=at)
            if key in by_image:
                raise TesseraContractError(
                    f"{at} publishes a second attestation for image {key!r}, "
                    "which the reader already read on this platform. Two "
                    "tables for one image are two answers to one question, "
                    "and picking either is a guess; publish one attestation "
                    "per image.")
            by_image[key] = rows
        if len(by_image) > 1 and "" in by_image:
            raise TesseraContractError(
                f"{spot} publishes {len(by_image)} attestations and one of "
                "them names no `generated` block, so nothing says which image "
                "it covers. With more than one table on a platform the image "
                "is what selects, and an unscoped table cannot be selected "
                "(RobTand/prismaquant#715).")
        table[str(platform)] = by_image
    return table


def _parse_activation_platform_entry(entry: Any, *, platform: str, where: str
                                     ) -> "tuple[str, dict]":
    """One ``platforms[p]`` attestation: its image key and its contract rows.

    The same object under both grammars -- v1 publishes exactly one of these
    and v2 a list of them -- so the member vocabulary, the scope block and the
    contract rows are read here once and the caller only decides how many to
    expect.

    The KEY is the image the table was generated in: its content digest when
    the contract names a digest reference, and the reference verbatim when it
    does not.  A digest is what the consumer compares (a tag moves, and one
    digest can sit behind two repository names), so keying on it makes two
    tables for one build a refusal at read time rather than an ambiguity at
    selection time.  A table with no ``generated`` block keys on ``""``: it
    names no image, which is permitted for a lone v1 table -- the consumer
    refuses it as unscoped -- and refused beside a second table above.
    """
    if not isinstance(entry, Mapping):
        raise TesseraContractError(f"{where} must be a JSON object")
    unknown = sorted(set(entry) - set(_ACTIVATION_PLATFORM_MEMBERS))
    if unknown:
        raise TesseraContractError(
            f"{where} publishes {unknown} which this reader does not know. "
            "A field beside a platform's quantiser tables that nothing "
            "here reads is either a value a gate should decide on or "
            "prose that does not belong; either way it is a review.")
    contracts = _require(entry, "contracts", where)
    if not isinstance(contracts, Mapping):
        raise TesseraContractError(f"{where}.contracts must be a JSON object")
    generated = _parse_activation_generated(entry.get("generated"),
                                            f"{where}.generated")
    rows: dict[str, ActivationQuantizerAttestation] = {}
    for name, row in contracts.items():
        rows[str(name)] = _parse_activation_contract(
            row, platform=platform, name=str(name),
            where=f"{where}.contracts[{name}]", generated=generated)
    return (activation_image_key(generated.image) if generated else "", rows)


def activation_image_key(reference: Any) -> str:
    """The key one image reference is read under, digest first.

    ``repository@sha256:<64 hex>`` keys on the 64 hex digits alone, because
    that is the only part that names bytes: the same build reaches a local
    registry under another repository name, and the GLM-5.3 campaign image is
    exactly that case.  Anything else -- a tag, a bare name -- keys on the
    reference verbatim, so it parses and can never be selected by a digest
    comparison.
    """
    text = str(reference)
    return text.split("@sha256:", 1)[1] if _DIGEST_IMAGE.fullmatch(text) else text


def _parse_activation_generated(entry: Any, where: str
                                ) -> "ActivationQuantizerGeneration | None":
    """Read the table's scope, or read that it publishes none.

    Absence returns ``None`` rather than refusing, for the same reason
    :func:`_parse_activation_quantizers` permits a missing block: every other
    gate on this contract was sound before the scope existed.  The refusal
    lives where the claim is consumed -- an attested stamp with no scope is
    ``not verified`` at
    :func:`native_operator_panel.require_panel_execution_scope` -- so a
    missing scope can never become a silent pass.

    A PRESENT block is read strictly: exactly the published vocabulary, every
    value a non-empty string.  A half-written scope is worse than none,
    because it looks like an answer.
    """
    if entry is None:
        return None
    if not isinstance(entry, Mapping):
        raise TesseraContractError(f"{where} must be a JSON object")
    if set(entry) != set(_ACTIVATION_GENERATED_MEMBERS):
        raise TesseraContractError(
            f"{where} must publish exactly "
            f"{sorted(_ACTIVATION_GENERATED_MEMBERS)}, got {sorted(entry)}")
    for field, value in entry.items():
        if not isinstance(value, str) or not value.strip():
            raise TesseraContractError(
                f"{where}.{field} must be a non-empty string, got {value!r}")
    return ActivationQuantizerGeneration(
        **{field: str(entry[field]) for field in _ACTIVATION_GENERATED_MEMBERS})


def _parse_activation_contract(entry: Any, *, platform: str, name: str,
                               where: str,
                               generated: "ActivationQuantizerGeneration | None" = None,
                               ) -> ActivationQuantizerAttestation:
    if not isinstance(entry, Mapping):
        raise TesseraContractError(f"{where} must be a JSON object")
    unknown = sorted(set(entry) - set(_ACTIVATION_CONTRACT_MEMBERS))
    if unknown:
        raise TesseraContractError(
            f"{where} publishes {unknown} which this reader does not know. A "
            "field in a quantiser attestation that nothing here reads is "
            "either a value a gate should decide on or prose that does not "
            "belong; either way it is a review, not a thing to skip.")
    length = _require(entry, "unit_length", where)
    if type(length) is not int or length < 1:
        raise TesseraContractError(
            f"{where}.unit_length must be a positive integer, got {length!r}")
    raw = _require(entry, "vectors", where)
    if (not isinstance(raw, Sequence) or isinstance(raw, (str, bytes))
            or not raw):
        raise TesseraContractError(f"{where}.vectors must be a non-empty array")
    vectors = []
    for i, vector in enumerate(raw):
        spot = f"{where}.vectors[{i}]"
        if not isinstance(vector, Mapping):
            raise TesseraContractError(f"{spot} must be a JSON object")
        if set(vector) != set(_ACTIVATION_VECTOR_MEMBERS):
            raise TesseraContractError(
                f"{spot} must publish exactly "
                f"{sorted(_ACTIVATION_VECTOR_MEMBERS)}, got {sorted(vector)}")
        inputs = vector["input"]
        codes = vector["codes"]
        for field, value in (("input", inputs), ("codes", codes)):
            if (not isinstance(value, Sequence)
                    or isinstance(value, (str, bytes))
                    or len(value) != length):
                raise TesseraContractError(
                    f"{spot}.{field} must carry exactly unit_length "
                    f"({length}) entries, got "
                    f"{len(value) if hasattr(value, '__len__') else value!r}")
        vectors.append(ActivationQuantizerVector(
            id=str(vector["id"]), boundary=str(vector["boundary"]),
            global_scale_bits=_parse_hex(vector["global_scale"], 8, spot,
                                         "global_scale"),
            input_bits=tuple(_parse_hex(bits, 4, spot, "input")
                             for bits in inputs),
            stored_scale_byte=_parse_byte(vector["stored_scale"], spot),
            codes=tuple(_parse_code(code, spot) for code in codes),
        ))
    return ActivationQuantizerAttestation(
        platform=platform, activation_contract=name,
        op=str(_require(entry, "op", where)),
        unit=str(_require(entry, "unit", where)),
        unit_length=length,
        grid=str(_require(entry, "grid", where)),
        block_scale=str(_require(entry, "block_scale", where)),
        global_scale=str(_require(entry, "global_scale", where)),
        vectors=tuple(vectors),
        generated=generated,
    )


def _parse_hex(value: Any, digits: int, where: str, key: str) -> int:
    """``"0x3f800000"`` -> ``int``, or refuse.

    Lowercase, ``0x``-prefixed, exactly ``digits`` wide.  Strict because a
    short form is ambiguous about width and a decimal integer here would be a
    different field.
    """
    text = str(value)
    if (len(text) != digits + 2 or not text.startswith("0x")
            or any(c not in "0123456789abcdef" for c in text[2:])):
        raise TesseraContractError(
            f"{where}.{key} must be '0x' plus {digits} lowercase hex digits, "
            f"got {value!r}")
    return int(text, 16)


def _parse_byte(value: Any, where: str) -> int:
    if type(value) is not int or not 0 <= value <= 255:
        raise TesseraContractError(
            f"{where}.stored_scale must be an integer byte 0..255, got "
            f"{value!r}")
    return value


def _parse_code(value: Any, where: str) -> int:
    if type(value) is not int or not 0 <= value <= 15:
        raise TesseraContractError(
            f"{where}.codes entries must be integers 0..15, got {value!r}")
    return value


def packaged_activation_quantizers() -> tuple[str, dict]:
    """The installed runtime's quantiser tables, with the bytes' digest.

    Read straight from the packaged JSON through :func:`contract_path`, NOT
    through :func:`load_tessera_contract`: that one is the development pin and
    returns ``None`` when the pin is not requested, and "the pin is not
    requested" must not be a way for an unattested quantiser to be priced.
    The digest travels with the table so a producer can stamp WHICH bytes
    attested it. It must also match the independently reviewed serving pin:
    otherwise an installed v25 table would admit fp4 pricing while the pin
    still declares v24, bypassing the review that moves the pin.
    """
    from importlib.resources import as_file

    with as_file(contract_path()) as path:
        raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    from .tessera_serving_runtime_pin import (
        load_tessera_serving_runtime_pin,
        require_exact_tessera_runtime_pin,
    )

    require_exact_tessera_runtime_pin(
        load_tessera_serving_runtime_pin(), installed_contract_sha256=sha)
    payload = json.loads(raw.decode("utf-8"))
    return sha, _parse_activation_quantizers(payload, str(contract_path()))


def _select_activation_attestation_set(
    table: Mapping[str, Any], *, platform: str, executing_image: str,
) -> "Mapping[str, ActivationQuantizerAttestation]":
    """The one attestation this run may price against, or a refusal.

    Since Tessera contract v33 a platform may publish several tables, one per
    serving image, and the rule is the one the consumer already enforces on a
    frozen panel: a cell is covered only by the attestation whose
    ``generated.image`` is the image it EXECUTES in (#715).  Reading that rule
    forward to the point the stamp is produced is what makes the list safe:

    * one table, no image named by the caller -- read it, exactly as a v1
      contract has always been read.  The image still travels into the stamp,
      and ``native_operator_panel.require_panel_execution_scope`` still
      refuses the panel when the executing image is another one, so this is
      the same verdict taken later rather than a gap.
    * an image named -- select by CONTENT DIGEST.  Exactly one table may
      match; none is a refusal that names both sides, and a reference with no
      digest is a refusal rather than a comparison made on a name.
    * several tables and no image named -- REFUSED.  There is no first entry
      here: picking one would be the producer asserting which build it ran,
      which is the assertion principle 14 exists to refuse.
    """
    by_image = table.get(platform) or {}
    if not by_image:
        return {}
    if executing_image:
        if not _DIGEST_IMAGE.fullmatch(executing_image):
            raise TesseraContractError(
                f"the executing image {executing_image!r} is not a digest "
                "reference (repository@sha256:<64 lowercase hex>), so there "
                "is nothing here to compare against the attested image: a tag "
                "names a moving target. Name the digest the serve resolved "
                "(RobTand/prismaquant#715).")
        key = activation_image_key(executing_image)
        if key in by_image:
            return by_image[key]
        raise TesseraContractError(
            f"the pinned Tessera contract publishes no quantiser attestation "
            f"generated in the executing image on {platform!r}.\n"
            f"  executing image: {executing_image}\n"
            f"  attested images: {sorted(by_image) or 'none'}\n"
            "Two builds of one operator are two objects (#567), so a table "
            "taken in another image does not cover this run. Re-measure in an "
            "attested image, or publish a table generated in this one; there "
            "is no tolerance and no allow-list here.")
    if len(by_image) > 1:
        raise TesseraContractError(
            f"the pinned Tessera contract publishes {len(by_image)} quantiser "
            f"attestations on {platform!r}, one per serving image "
            f"({sorted(by_image)}), and this call named none. Which rounding "
            "rule covers a measurement is decided by the image it executes "
            "in, never by the order the contract lists them in: pass "
            "executing_image=<repository@sha256:...> "
            "(RobTand/prismaquant#715, #926).")
    return next(iter(by_image.values()))


def require_activation_quantizer_attested(
    activation_contract: str,
    *,
    platform: str,
    table: "Mapping[str, Mapping[str, Mapping[str, ActivationQuantizerAttestation]]] | None" = None,
    contract_sha256: str = "",
    executing_image: str = "",
) -> dict:
    """Refuse to price an activation residual the runtime never attested.

    The same shape as :func:`require_pin_native_extensions_match_contract`, for
    the quantiser's ARITHMETIC instead of the extension table: the runtime
    publishes a machine-readable table generated by running its own kernel,
    this producer recomputes it with the code it actually prices with, and a
    disagreement REFUSES.

    Both halves of the quantiser are checked, because the table publishes a
    whole group: the ``amax -> UE4M3`` byte the kernel stored, and the code
    each element became under it.  Four refusals, and each of them is a state
    that used to read as "fine":

    * no table, no platform, or no row for this contract string --
      **unattested**.  The name ``"e2m1_group16_ue4m3_static"`` says which
      quantiser; it never said what that quantiser does at a tie.
    * a platform that publishes several tables -- one per serving image since
      Tessera contract v33 -- and no ``executing_image`` to select with, or
      one that matches none of them.  Which build ran is a fact about this
      run, and there is no first entry here
      (:func:`_select_activation_attestation_set`).
    * a vocabulary this reader does not transcribe -- another unit, grid or
      block-scale format is a different quantiser, not a thing to read with
      this one's arithmetic.
    * a coverage gap -- a table that omits the ties attests nothing about the
      disagreement it is here to settle, so all seven E2M1 midpoints, both
      signs, element saturation and the block-scale underflow boundary are
      required.  The check is NUMERIC, computed from the published bits, so a
      mislabelled ``boundary`` cannot buy coverage it does not reach.
    * a stored-scale or code disagreement -- named vector by vector.

    Returns the stamp a producer freezes beside the reference it just
    computed, so an admitted receipt carries WHICH bytes attested its oracle.

    What this does NOT attest, recorded rather than implied: every probe is a
    value both sides represent exactly, so the table says nothing about a
    NON-DYADIC used scale -- and the cells that diverge in production run at
    one.  That is where the remaining suspicion belongs, and no contract row
    can settle it (RobTand/prismaquant#567).
    """
    import torch

    from .nvfp4_activation_contract import (
        E2M1_MIDPOINTS,
        FP4_E2M1_MAX,
        FP4_GROUP_SIZE,
        nvfp4_e2m1_code,
        nvfp4_e2m1_normalize,
        nvfp4_group_stored_scale,
    )

    sha = contract_sha256
    if table is None:
        sha, table = packaged_activation_quantizers()
    published = _select_activation_attestation_set(
        table, platform=str(platform), executing_image=str(executing_image or ""))
    row = published.get(str(activation_contract))
    if row is None:
        raise TesseraContractError(
            "the pinned Tessera contract publishes no quantiser attestation "
            f"for {activation_contract!r} on {platform!r} (it publishes "
            f"{ {p: {i: sorted(c) for i, c in by_image.items()} for p, by_image in table.items()} or 'nothing at all'}"
            ").\n"
            "PrismaQuant prices this contract's activation residual with its "
            "OWN re-implementation of the rounding rule "
            "(nvfp4_activation_contract.nvfp4_activation_qdq_served), and "
            "principle 14 reads an unattested claim about another runtime as "
            "REFUSED, not as fine. Publish activation_quantizers beside "
            "activation_contract -- generated by running the kernel -- and "
            "this passes or names what disagrees (RobTand/prismaquant#567).")
    for field, value, expected in (
            ("op", row.op, _ATTESTED_OP),
            ("unit", row.unit, _ATTESTED_UNIT),
            ("grid", row.grid, _ATTESTED_GRID),
            ("block_scale", row.block_scale, _ATTESTED_BLOCK_SCALE),
            ("global_scale", row.global_scale, _ATTESTED_GLOBAL_SCALE),
            ("unit_length", row.unit_length, FP4_GROUP_SIZE)):
        if value != expected:
            raise TesseraContractError(
                f"the quantiser attestation for {activation_contract!r} on "
                f"{platform!r} publishes {field}={value!r}; PrismaQuant's "
                f"oracle implements {expected!r}, and a table taken under "
                "another vocabulary attests a different quantiser. "
                "Transcribing a new one is a review, never a thing to admit "
                "blind.")

    scale_lines, code_lines = [], []
    reached, signed, saturating = set(), False, False
    for vector in row.vectors:
        g = _bits_to_f32([vector.global_scale_bits])
        grouped = _bits_to_bf16(vector.input_bits).reshape(1, 1, -1).float()
        stored = nvfp4_group_stored_scale(grouped, g)
        byte = int(stored.view(torch.uint8).reshape(-1)[0])
        if byte != vector.stored_scale_byte:
            scale_lines.append(
                f"  {vector.id} ({vector.boundary}): the runtime stored "
                f"{vector.stored_scale_byte:#04x}, PrismaQuant derives "
                f"{byte:#04x}")
            continue
        used = stored.float() / g
        underflow = used == 0
        normalized = nvfp4_e2m1_normalize(
            grouped, torch.where(underflow, torch.ones_like(used), used))
        ours = nvfp4_e2m1_code(
            torch.where(underflow, torch.zeros_like(normalized), normalized)
        ).reshape(-1).tolist()
        for i, (mine, theirs) in enumerate(zip(ours, vector.codes)):
            if not _e2m1_codes_agree(int(mine), theirs):
                code_lines.append(
                    f"  {vector.id} ({vector.boundary}) element {i}: input "
                    f"{vector.input_bits[i]:#06x} at stored_scale "
                    f"{vector.stored_scale_byte:#04x} -- the runtime emitted "
                    f"code {theirs}, PrismaQuant's oracle emits {int(mine)}")
        if not bool(underflow.reshape(-1)[0]):
            scale = float(used.reshape(-1)[0])
            for value in _bits_to_bf16(vector.input_bits).float().tolist():
                if value < 0.0:
                    signed = True
                magnitude = abs(value) / scale
                if magnitude > FP4_E2M1_MAX:
                    saturating = True
                for midpoint in E2M1_MIDPOINTS:
                    if magnitude == midpoint:
                        reached.add(midpoint)
        elif any(bits >> 15 for bits in vector.input_bits):
            signed = True

    gaps = [f"midpoint t={m} is not reached by any vector"
            for m in E2M1_MIDPOINTS if m not in reached]
    if not signed:
        gaps.append("no vector has a negative input; the sign convention and "
                    "the negative-zero code are unattested")
    if not saturating:
        gaps.append("no vector has an element above the top code; the "
                    "saturation convention is unattested")
    for byte, why in ((0x00, "the block the runtime zeroes"),
                      (0x01, "the smallest nonzero stored scale")):
        if not any(v.stored_scale_byte == byte for v in row.vectors):
            gaps.append(f"no vector carries stored_scale {byte:#04x} ({why}); "
                        "the block-scale underflow boundary is unattested")
    if gaps and not scale_lines:
        raise TesseraContractError(
            f"the quantiser attestation for {activation_contract!r} on "
            f"{platform!r} does not cover the points where the rounding RULE "
            "decides:\n" + "\n".join(f"  {gap}" for gap in gaps) + "\n"
            "A table that omits the ties attests nothing about the "
            "disagreement it is here to settle.")
    if scale_lines or code_lines:
        raise TesseraContractError(
            "PrismaQuant's activation quantiser is not the one the pinned "
            f"Tessera runtime executes for {activation_contract!r} on "
            f"{platform!r} ({row.op}):\n"
            + "\n".join(scale_lines + code_lines) + "\n"
            "The measurement is right and the oracle is wrong (principle 1): "
            "fix nvfp4_group_stored_scale / nvfp4_e2m1_normalize / "
            "nvfp4_e2m1_magnitude_index until every vector agrees. Do NOT "
            "widen a tolerance -- with a bit-identical stored scale the honest "
            "per-element bound is zero, and a code flip is a wrong activation, "
            "not a rounding epsilon.")
    return {
        "schema": "prismaquant.activation_quantizer_attestation.v1",
        "activation_contract": row.activation_contract,
        "platform": row.platform,
        "op": row.op,
        "vectors": len(row.vectors),
        "elements": sum(len(v.codes) for v in row.vectors),
        "boundaries": sorted({v.boundary for v in row.vectors}),
        "contract_sha256": sha,
        # WHICH kernel emitted the vectors this stamp reproduces: read from
        # the contract bytes, never from the driver or the environment.  A
        # consumer compares the executing image against this one and refuses
        # a cell measured on another build of the same operator (#715).
        "generated": row.generated.as_stamp() if row.generated else None,
        "generated_absent_because": None if row.generated else (
            "the pinned contract publishes no activation_quantizers"
            ".platforms[<platform>].generated block, so the table states no "
            "image or build it was taken under; an attestation with no scope "
            "is not verified (RobTand/prismaquant#715)"),
        "oracle": "prismaquant.nvfp4_activation_contract",
        "attests": ["amax_to_ue4m3_stored_scale", "value_to_code_rounding"],
        "does_not_attest": ["non_dyadic_used_scale"],
    }


def _bits_to_f32(bits: "list[int]"):
    """Unsigned binary32 patterns as the floats they are.

    Through ``int32`` because that is what ``view(torch.float32)`` needs, and a
    negative float's pattern has the top bit set, which is out of range as an
    unsigned Python int.  The wrap is exact: it is the same 32 bits either way.
    """
    import torch

    return torch.tensor(
        [b - (1 << 32) if b >= (1 << 31) else b for b in bits],
        dtype=torch.int32).view(torch.float32)


def _bits_to_bf16(bits: "tuple[int, ...]"):
    """Unsigned bfloat16 patterns as the values the kernel was handed."""
    import torch

    return torch.tensor(
        [b - (1 << 16) if b >= (1 << 15) else b for b in bits],
        dtype=torch.int16).view(torch.bfloat16)


def _e2m1_codes_agree(ours: int, theirs: int) -> bool:
    """Code equality, with the one equivalence the grid actually has.

    ``0`` and ``8`` are ``+0`` and ``-0``: they dequantise to the same number,
    so no GEMM and no measurement downstream can tell them apart, and treating
    them as a disagreement would refuse on a difference that cannot exist in
    any observable. Every other pair of codes is a different magnitude.
    """
    return ours == theirs or {ours, theirs} <= {0, 8}


def _parse_fused_module(payload: Mapping[str, Any], path: str
                        ) -> FusedModuleLicence:
    """Read ``fused_module``, or refuse.

    Required, not defaulted.  A contract without this block has published
    nothing about what a fused module's roles may disagree on, and the two
    ways to default it are both assertions: "everything shared" would invent a
    constraint this runtime does not state, and "the rate is free" is the exact
    prose claim reading the block exists to replace.

    Only the values a gate on this side decides on are kept, which is what
    makes :meth:`FusedModuleLicence.answer` the whole block's projection.
    ``container`` is read past deliberately: there is no Tessera export leg to
    write a sidecar with, so nothing here consumes the magic, and parsing it
    would put a field nobody uses into the pin's answer and make a re-review
    out of it.  The day a writer reads it, it joins the dataclass and the
    answer in one commit.
    """
    where = f"{path}.fused_module"
    block = _require(payload, "fused_module", path)
    if not isinstance(block, Mapping):
        raise TesseraContractError(f"{where} must be a JSON object")
    schema = block.get("schema")
    if schema != FUSED_MODULE_SCHEMA:
        raise TesseraContractError(
            f"{where}.schema must be {FUSED_MODULE_SCHEMA!r}, got {schema!r}. "
            "A block with another id is not a subset of this one, so it is "
            "refused rather than partially read."
        )
    fields = _require(block, "fields", where)
    if not isinstance(fields, Mapping) or not fields:
        raise TesseraContractError(
            f"{where}.fields must be a non-empty object mapping a field name "
            "to its licence"
        )
    parsed: dict[str, str] = {}
    for name, licence in fields.items():
        if licence not in FUSED_MODULE_FIELD_LICENCES:
            raise TesseraContractError(
                f"{where}.fields[{name!r}] is {licence!r}; this reader knows "
                f"only {sorted(FUSED_MODULE_FIELD_LICENCES)} and will not "
                "guess at a third licence's meaning -- a token it mapped onto "
                "'per_member' would widen a group allocator's menu on a word "
                "it did not understand"
            )
        parsed[str(name)] = str(licence)
    receipt = _require(block, "mixed_rung_receipt", where)
    if not isinstance(receipt, bool):
        raise TesseraContractError(
            f"{where}.mixed_rung_receipt must be a JSON boolean, got "
            f"{receipt!r}. It is the difference between "
            "\"a serve has covered a mixed-rung module\" and \"a decode "
            "identity has\", and a truthy string answers neither question."
        )
    return FusedModuleLicence(
        schema=str(schema),
        fields=parsed,
        sidecar_q256=str(_require(block, "sidecar_q256", where)),
        mixed_rung_receipt=receipt,
    )


#: The shard axes ``tensor_parallel.units[].loader_axes`` may name, in
#: Tessera's own vocabulary (``tessera.serving.sharding.AXES``).  A unit that
#: names another axis, or omits one of these, is refused rather than read with
#: the axes this reader happens to know: the point of publishing a status per
#: axis is that the answer is not derivable from the axis's name.
TP_LOADER_AXES = ("column", "row")

#: What a published axis status may say.  ``sharded`` is "this build's loader
#: accepts a shard on this axis", ``refused`` is "it does not, on every rank".
#: Neither is an attestation: ``max_world_size`` in the same unit row is the
#: attestation, and the two are separate questions (a loader that would cut an
#: axis it has never been measured cutting is still unattested).
TP_LOADER_AXIS_SHARDED = "sharded"
TP_LOADER_AXIS_REFUSED = "refused"
TP_LOADER_AXIS_STATUSES = (TP_LOADER_AXIS_SHARDED, TP_LOADER_AXIS_REFUSED)


def _parse_loader_axes(block: Any, where: str) -> dict[str, str]:
    """Read one unit's per-axis loader statuses, or refuse the table.

    There is no default here on purpose.  A missing block, an axis this
    reader does not share, a status it does not know, and a ``status`` read
    off the prose beside it are all the same failure: a claim about what
    another runtime's loader does, read as something other than what was
    published (principle 14).  Reading absence as ``sharded`` would price a
    cut the loader refuses on every rank.
    """
    if not isinstance(block, Mapping):
        raise TesseraContractError(
            f"{where} publishes no usable 'loader_axes' mapping (got "
            f"{type(block).__name__}). This reader will not assume an axis "
            "shards because the table did not say it does not."
        )
    if set(block) != set(TP_LOADER_AXES):
        raise TesseraContractError(
            f"{where}.loader_axes names axes {sorted(block)}; this reader "
            f"implements exactly {sorted(TP_LOADER_AXES)} and refuses a "
            "vocabulary it does not share rather than reading the axes it "
            "recognises and dropping the rest"
        )
    axes: dict[str, str] = {}
    for axis in sorted(TP_LOADER_AXES):
        entry = block[axis]
        if not isinstance(entry, Mapping):
            raise TesseraContractError(
                f"{where}.loader_axes.{axis} must be an object carrying a "
                f"'status', got {entry!r}"
            )
        status = str(_require(entry, "status", f"{where}.loader_axes.{axis}"))
        if status not in TP_LOADER_AXIS_STATUSES:
            raise TesseraContractError(
                f"{where}.loader_axes.{axis}.status is {status!r}; this "
                f"reader knows {sorted(TP_LOADER_AXIS_STATUSES)}. The "
                "``reason`` beside it is prose and is never the value a gate "
                "reads."
            )
        axes[axis] = status
    return axes


def _parse_tensor_parallel(
    payload: Mapping[str, Any], path: str,
) -> tuple[dict[str, int], dict[str, dict[str, str]]]:
    """Read both facts the ``tensor_parallel`` unit rows publish.

    ``max_world_size`` is the ATTESTATION bound -- the largest world size a
    served receipt covers -- and ``loader_axes`` is what this build's loader
    does with each shard axis.  They answer different questions and they are
    read together here so a unit row cannot be half-read: a family with a
    ceiling and no axis claim is a table this reader refuses.
    """
    tp = _require(payload, "tensor_parallel", path)
    if str(tp.get("semantics")) != "closed_world":
        raise TesseraContractError(
            f"{path}.tensor_parallel.semantics is "
            f"{tp.get('semantics')!r}; this reader treats the block as a closed "
            "world (a family absent from it is not attested at any degree) and "
            "will not read an open-world table under that assumption"
        )
    receipts = _parse_world_size_receipts(tp, path)
    world: dict[str, int] = {}
    axes: dict[str, dict[str, str]] = {}
    for i, unit in enumerate(tp.get("units", ())):
        where = f"{path}.tensor_parallel.units[{i}]"
        name = str(_require(unit, "unit", where))
        declared = int(_require(unit, "max_world_size", where))
        if declared > 1:
            _require_world_size_receipt(unit, name, declared, receipts, where)
        world[name] = declared
        axes[name] = _parse_loader_axes(unit.get("loader_axes"), where)

    return world, axes


def _parse_world_size_receipts(
    tp: Mapping[str, Any], path: str,
) -> dict[str, tuple[int, frozenset[str]]]:
    """``receipt id -> (world_size, executed_units)``, the two members a gate needs.

    Contract v29 (Tessera #517) raised ``max_world_size`` above 1 and, in the
    same bump, published the served run that covers it.  This reads only the
    numbers the ceiling rests on: which world the run served and which units it
    executed.  The images, flags, route traces and the single-rank KL table
    beside them are the publisher's evidence for a human, not values this side
    decides on, so they are neither read nor projected into the answer.
    """
    rows = tp.get("world_size_receipts", ())
    where_block = f"{path}.tensor_parallel.world_size_receipts"
    if not isinstance(rows, (list, tuple)):
        raise TesseraContractError(f"{where_block} must be a list")
    receipts: dict[str, tuple[int, frozenset[str]]] = {}
    for i, row in enumerate(rows):
        where = f"{where_block}[{i}]"
        receipt_id = str(_require(row, "id", where))
        if receipt_id in receipts:
            raise TesseraContractError(
                f"{where}: receipt id {receipt_id!r} is published twice")
        world_size = _require(row, "world_size", where)
        if (isinstance(world_size, bool) or not isinstance(world_size, int)
                or world_size < 1):
            raise TesseraContractError(
                f"{where}.world_size must be a positive integer, got "
                f"{world_size!r}")
        executed = _require(row, "executed_units", where)
        if (not isinstance(executed, (list, tuple))
                or not all(isinstance(u, str) for u in executed)):
            raise TesseraContractError(
                f"{where}.executed_units must be a list of unit names")
        receipts[receipt_id] = (world_size, frozenset(executed))
    return receipts


def _require_world_size_receipt(
    unit: Mapping[str, Any],
    name: str,
    declared: int,
    receipts: Mapping[str, tuple[int, frozenset[str]]],
    where: str,
) -> None:
    """Refuse a ceiling above 1 that no published served run covers.

    A world size of 1 is what a unit serves with no collective, and every
    contract before v29 published it with no receipt, so it needs none here.
    Above 1 the ceiling is a claim about a run: the unit must name a receipt,
    the receipt must be published, it must have EXECUTED this unit, and it
    must have served at least the declared world.  A receipt at a smaller
    world than the ceiling claims would let ``tessera_tp_world_attested``
    admit a degree nobody served.
    """
    receipt_id = unit.get("world_size_receipt")
    if receipt_id is None:
        raise TesseraContractError(
            f"{where}: {name} declares max_world_size {declared} but names no "
            "world_size_receipt; a ceiling above 1 is admitted only on a "
            "published served run, never on the table's word")
    if str(receipt_id) not in receipts:
        raise TesseraContractError(
            f"{where}: {name} cites world_size_receipt {receipt_id!r}, which "
            "tensor_parallel.world_size_receipts does not publish")
    world_size, executed = receipts[str(receipt_id)]
    if name not in executed:
        raise TesseraContractError(
            f"{where}: world_size_receipt {receipt_id!r} did not execute "
            f"{name} (executed_units {sorted(executed)}), so it attests no "
            "world size for it")
    if world_size < declared:
        raise TesseraContractError(
            f"{where}: {name} declares max_world_size {declared} but "
            f"world_size_receipt {receipt_id!r} served world size "
            f"{world_size}; the ceiling may not exceed the run that covers it")


def _parse_tensor_parallel_limits(payload: Mapping[str, Any], path: str) -> dict[str, int]:
    """Read the closed-world TP ceiling for both pin paths."""
    return _parse_tensor_parallel(payload, path)[0]


@lru_cache(maxsize=8)
def published_tensor_parallel_limits(path: str, sha: str) -> Mapping[str, int]:
    """Read packaged TP metadata, keyed by the attesting table identity.

    This accessor reports ceilings only; it does not grant TP admission.
    """
    return _published_tensor_parallel(path, sha)[0]


@lru_cache(maxsize=8)
def published_tensor_parallel_axes(
    path: str, sha: str,
) -> Mapping[str, Mapping[str, str]]:
    """``family -> axis -> status``, read from a contract file on disk.

    The same reader the parsed contract uses, for the two callers that hold a
    contract path rather than a loaded contract: the packaged-table route
    admission, and ``prismaquant.tessera_tp_audit``'s ``--contract``.  One
    reader means the vocabulary refusal is identical on every path.
    """
    return _published_tensor_parallel(path, sha)[1]


def _published_tensor_parallel(
    path: str, sha: str,
) -> tuple[Mapping[str, int], Mapping[str, Mapping[str, str]]]:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise TesseraContractError(
            f"{path}: tensor-parallel metadata digest differs from attesting table")
    return _parse_tensor_parallel(json.loads(raw), path)


def _parse(payload: Mapping[str, Any], *, commit: str, sha: str, path: str
           ) -> TesseraContract:
    schema = payload.get("schema")
    if schema != TESSERA_CONTRACT_SCHEMA:
        raise TesseraContractError(
            f"{path}: schema must be {TESSERA_CONTRACT_SCHEMA!r}, got "
            f"{schema!r}. An older contract is not a subset of this one, so it "
            "is refused rather than partially read."
        )
    reader_range: dict[str, tuple[int, int]] = {}
    attested: dict[str, frozenset[int]] = {}
    formats = _require(payload, "formats", path)
    if not isinstance(formats, Sequence) or isinstance(formats, (str, bytes)):
        raise TesseraContractError(f"{path}.formats must be a JSON array")
    for i, entry in enumerate(formats):
        if not isinstance(entry, Mapping):
            raise TesseraContractError(f"{path}.formats[{i}] must be an object")
        where = f"{path}.formats[{i}]"
        kind = str(_require(entry, "kind", where))
        if kind != "tessera_wire":
            raise TesseraContractError(
                f"{where}.kind is {kind!r}; this reader knows only "
                "'tessera_wire' and will not guess at another kind's rung "
                "vocabulary"
            )
        family = str(_require(entry, "family", where))
        lo, hi = (int(v) for v in _require(entry, "reader_rate_range_q256", where))
        reader_range[family] = (lo, hi)
        # ``attested_rungs_q256`` is the field's name since Tessera contract
        # v2; ``candidate_rungs_q256`` is the deprecated alias it kept so the
        # rename stayed additive, and Tessera's own reader refuses the two if
        # they disagree.  Read the current name first so this reader survives
        # the alias being dropped, and accept the alias alone so it can still
        # read a v1 table.  Reading the alias *preferentially* is how the gap
        # the rename closed would reopen: the alias never was the decodable
        # set.
        rungs = entry.get("attested_rungs_q256", entry.get("candidate_rungs_q256"))
        if rungs is None:
            raise TesseraContractError(
                f"{where} publishes no 'attested_rungs_q256' (nor its "
                "deprecated alias 'candidate_rungs_q256')"
            )
        attested[family] = frozenset(int(r) for r in rungs)

    # The extension table is read BEFORE the lane table because the lane
    # table's cells launch through it: its own refusals (no table, an empty
    # one, a match rule this reader cannot apply, a lane block it cannot
    # decide) come first and by their own names.
    extensions = _parse_native_extensions(
        _require(payload, "native_extensions", path),
        where=f"{path}.native_extensions",
    )

    lane = _require(payload, "lane_eligibility", path)
    if not isinstance(lane, Mapping):
        raise TesseraContractError(f"{path}.lane_eligibility must be an object")
    lane_schema = lane.get("schema")
    if lane_schema not in TESSERA_LANE_SCHEMAS:
        raise TesseraContractError(
            f"{path}.lane_eligibility.schema must be one of {sorted(TESSERA_LANE_SCHEMAS)!r}, "
            f"got {lane_schema!r}"
        )
    try:
        # The lane table is read beside the extension table it launches
        # through (contract v20): the reader binds every extension launch a
        # cell names to that extension's lane. The predicate itself is NOT
        # decided here -- loading a contract imports no encoder and no
        # serving code; admission (``native_cells``) does.
        table = _parse_table(lane, formats, "", commit, sha,
                             native_extensions=payload["native_extensions"])
    except LaneEligibilityError as exc:
        raise TesseraContractError(f"{path}: {exc}") from exc
    cells: list[TesseraRouteCell] = []
    for cell in table.cells:
        if cell.predicates:
            raise TesseraContractError(
                f"{path}.lane_eligibility.cells[{cell.id!r}].predicates "
                "cannot be evaluated by shape-free development admission; "
                "a constrained route must not enter an unconditional menu. "
                "Pass structural facts through admission before accepting "
                "predicated cells."
            )
        cells.append(TesseraRouteCell(
            cell_id=cell.id,
            platform=cell.platform,
            family=cell.family,
            structure=cell.structure,
            regime=cell.regime,
            rungs_q256=frozenset(cell.rungs_q256),
            activation_contract=cell.activation_contract,
            route_status=cell.route_status,
            qualification=cell.qualification,
            requires_plugin=cell.requires_plugin,
            requires_serve_flags=cell.requires_serve_flags,
            executes=cell.executes,
            residency_modes=cell.residency_modes,
            runtime_image=cell.runtime_image,
            execution_modes=cell.execution_modes,
            runtime_vllm=cell.runtime_vllm,
            runtime_torch=cell.runtime_torch,
            evidence=cell.evidence,
        ))

    world, loader_axes = _parse_tensor_parallel(payload, path)

    fused = _parse_fused_module(payload, path)

    versions = payload.get("versions", {})
    method = payload.get("quant_method", {})
    return TesseraContract(
        reader_rate_range=reader_range,
        attested_rungs=attested,
        cells=tuple(cells),
        native_extensions=extensions,
        max_world_size=world,
        loader_axes=loader_axes,
        fused_module=fused,
        quant_method=str(method.get("canonical", "")),
        contract_version=int(payload.get("contract_version", 0)),
        plugin_version=str(versions.get("tessera", "")),
        default_serve_image=str(versions.get("default_serve_image", "")),
        commit=commit,
        sha256=sha,
        path=path,
        lane_schema=table.schema,
        regimes=table.regimes,
        activation_quantizers=_parse_activation_quantizers(payload, path),
    )


@lru_cache(maxsize=8)
def _load_at(path: str, sha: str, commit: str) -> TesseraContract:
    """Parse one contract file.  Keyed on the SHA, so a changed file re-reads.

    ``route_admission``'s docstring explains why a cache over a runtime
    contract is a defect when its key cannot state the contract's identity.
    This key is exactly that identity, so the cache is safe: edit the file and
    the sha changes, which is a different key.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _parse(payload, commit=commit, sha=sha, path=path)


def load_tessera_contract() -> "TesseraContract | None":
    """The packaged Tessera contract under the dev pin, or ``None``.

    ``None`` means this development pin is not requested. Serving and export
    have separate exact pin checks. Every other failure --
    a contract whose *answer* is not the reviewed one, a missing or malformed
    file -- raises.  A mismatch never degrades to "unattested"; that would turn
    a stale pin into a silently empty menu.
    """
    requested = dev_pin_requested()
    if not requested:
        return None
    from importlib.resources import as_file

    with as_file(contract_path()) as path:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise TesseraContractError(
                f"cannot read the packaged Tessera contract at {path}: {exc}"
            ) from exc
        sha = hashlib.sha256(raw).hexdigest()
        contract = _load_at(str(path), sha, TESSERA_DEV_PIN_COMMIT)
        drift = _answer_drift(TESSERA_DEV_PIN_ANSWER, contract_answer(contract))
        if drift:
            raise TesseraContractError(
                "Tessera moved and its answer moved with it -- re-review the pin.\n"
                f"The pin in {__name__} was reviewed against Tessera "
                f"{TESSERA_DEV_PIN_COMMIT} (contract sha256 "
                f"{TESSERA_DEV_PIN_CONTRACT_SHA256}); {path} hashes to {sha} and "
                "publishes a different answer:\n" + "\n".join(drift) + "\n"
                "This is not a corruption warning. Read what moved, decide whether "
                "PrismaQuant should admit it, and update TESSERA_DEV_PIN_ANSWER "
                "(with the commit and sha) in the same commit -- that diff is the "
                "review."
            )
        # The middle link of the §7.4 chain, checked wherever both objects are in
        # hand.  It is deliberately AFTER the answer check: a moved answer is the
        # more informative refusal, and the pin cannot be judged against a
        # contract this producer has not accepted.
        require_pin_native_extensions_match_contract(contract)
        return contract
