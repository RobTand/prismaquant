"""Serving-lane eligibility, ATTESTED from the pinned runtime's own contract.

Principle 14: a claim about another runtime is *derived from a machine-readable
table the pinned runtime publishes, or refused*. This module is the consumption
half of that rule. It never encodes what a serving runtime does; it reads what
that runtime *says* it does, from the packaged ``runtime_contract.json`` its
installed distribution carries, and reports ``UNATTESTED`` when the pinned
release publishes no claim covering a unit.

The live publisher is Tessera's own vLLM plugin (``tessera.serving``). Until
2026-09-02 it was the Gridbook codebook plugin, and this module was named
``gridbook_lane_eligibility``; the Gridbook lane was retired that day (Rob:
"put Tessera in PrismaQuant and remove Gridbook") and the module was renamed
to the neutral name it should always have had. See
``archive/gridbook_lane_2026-09-02/README.md``.

Why this exists (the measured defect, on the retired lane). The shipped DSv4
87 GB codebook artifact carried 11 routed FP8-CB layers whose
``gate_proj``/``up_proj`` bound distinct learned codebooks. That runtime's
persistent-B prefill lane refused per-role split books, so those layers took
the announced expand+grouped-bridge route above the token threshold. Nothing in
the producer knew: no serving-profile lane declared a structured
``route_status``, so eligibility was not a gate input and a user discovered it
at serve time. Its twin on the vanilla-vLLM lane is
``units_on_fallback_route=0`` -- vacuous, because no spec declares route status
at all, so the counter is reachable only by never having looked. (That counter
is retired (#1377): ``route_status_counts`` is the provenance's one route
answer, and it counts ``no_declared_lane`` rather than reading it as clean.)

The shape of the fix is therefore as important as the values:

* Verdicts are NEVER literals in this repository. A serving-profile lane
  declares which eligibility key it consults; the verdict is resolved here from
  the pinned contract.
* Absence is LOUD and typed. When the pinned contract publishes no eligibility
  table every unit resolves to :data:`ROUTE_STATUS_UNATTESTED`, and the
  provenance payload omits the backed/fallback counters entirely rather than
  reporting them as zero. A vacuous zero must be unrepresentable.
* Route status alone never removes an honestly priced rung from the allocator's
  menu (principle 1). It gates EXPORT, per artifact, per principle 9.

Schemas v3/v4/v5, and why absence carries the whole weight
---------------------------------------------------
``tessera.lane-eligibility.v3`` is a **closed-world cell table**. It declares
``platforms``, ``regimes``, ``structures`` and a list of ``cells``, each cell
naming exactly one ``(platform, family, structure, regime)`` and the rung set it
covers -- ``rungs`` (codebook K) for a ``cb_product`` family, ``rungs_q256``
(body bits per 256 weights) for a RATE-addressed family. Which discriminator a
cell uses is NOT a key on the cell: it is decided by whether the cell's family
appears in ``formats[]`` with a rate-addressed ``kind``
(:data:`RATE_ADDRESSED_FORMAT_KINDS`), exactly as the publisher's own
validator decides it.

``tessera.lane-eligibility.v4`` adds a required non-empty ``executes`` set of
``{symbol, decoder}`` launches and makes residency an explicit resolution
axis. A caller must name a residency; two cells in the same scope may never
claim the same residency. The published serve flag selects that axis, and the
family's ``residency_modes`` bounds it. Legacy v3 tables retain their original
resolution semantics and never acquire fabricated launch claims.

``tessera.lane-eligibility.v5`` additionally requires every cell to name its
exact image digest and execution-mode scope. Missing target context is
unattested, never a request to use the global dense image or another cell.

``tessera.lane-eligibility.v6`` requires every cell to carry ``evidence``
(a derived grade, KL receipts, a greedy smoke's status) and the vLLM/torch
versions it was measured under. ``v7`` (Tessera #195) adds the smoke's
``control`` -- the reference it was compared against -- and an
``attribution`` derived from it. ``v8`` (Tessera #198) adds
``evidence.artifact``, the encoder scope of the KL: which commit wrote the
bytes it was measured on, and whether a later encoder reproduces them.
Each schema names the fields this reader consumes, and each is required at
its own schema; see :func:`parse_cell_evidence` and, for what the reader
DECIDES on, :func:`cell_evidence_admits`.

Additive fields (#1548). A field or block this reader does not know is
accepted and never read, so a Tessera release that adds one does not break
this reader. A producer that adds a field an old reader may not skip lists it
in the object's ``must_understand`` array, and this reader then refuses the
table. Both rules live in :mod:`prismaquant.record_fields`. The ``requires``
predicate is the exception: every key in it is a condition, so an unknown
requirement is still refused (:func:`parse_lane_claim`).

The lane predicate (contract v20, Tessera #264)
-----------------------------------------------
A cell names the LAUNCHES it executes (``executes[].decoder``); since
contract v20 the contract also publishes, per ``native_extensions[]`` row, a
``lane`` block naming the decoder that extension serves and -- for the
window-GEMV kernel -- ``requires``, the predicate a unit's WIRE must satisfy
for the kernel to read it (column rates, window bits, body, plane, no release
overrides, no diagonals, no rotation, no start state, a scalar grid).  The
loader refuses a unit that fails it, so a producer that selected such a unit
would ship bytes whose serve substitutes or refuses.  This reader parses the
predicate closed at Tessera's own vocabulary (:func:`parse_lane_claim`) and
:func:`cell_lane_admits` decides it for a cell -- by handing THIS producer's
planned wire (``tessera_render.planned_wire_facts``) to Tessera's own
decision core (``tessera.serving.scheme.decide_lane_requirements``).  The
rule has one home and it is not in this repository; what lives here is the
facts and the refusal.  A lane launch is made only at a rung the lane admits:
since contract v42 a cell can name a lane launch beside a lane-free one (the
fused routed pair beside the compact adapter), and where the lane refuses the
plan the cell admits on the launches left (:func:`cell_rung_launches`, PQ
#1274).

One parser, and why the vocabulary is wider than one publisher
---------------------------------------------------------------
``gridbook.lane-eligibility.v3`` was the same wire format from the retired
lane, and this parser served both. What remains of that is vocabulary, not a
second authority: the ``cb_product`` kind, its ``rungs`` discriminator and the
``tcq_trellis`` rate-addressed kind are still parsed, because they are the
closed-world grammar a v3 table is written in, and a parser that silently
dropped a kind would mis-read a table rather than refuse it. Only
:data:`LANE_ELIGIBILITY_SCHEMAS` decides whose tables are accepted, and since
2026-09-02 that set names Tessera alone.

The publisher's cell status vocabulary is ``backed | backed_with_serve_flag |
fallback``. There is deliberately **no ``unbacked`` cell**: a runtime does not
enumerate what it cannot serve, so the ONLY negative signal a v3 table carries
is *absence* -- no cell names this platform, this family, this rung. This
module therefore resolves an uncovered unit to :data:`ROUTE_STATUS_UNATTESTED`
rather than inventing ``unbacked`` from silence, and the export gate fails
closed on an unattested unit whose family the contract governs. A parser that
silently admitted an unlisted rate would turn the one negative signal the table
has into no signal at all.

Scope, derived rather than typed. A unit is *in scope* when its payload family
appears in the pinned contract's ``formats[]`` table -- that is the runtime
saying "I decode these bytes", so its eligibility table is the authority for
them. BF16, a SOURCE passthrough and a stock compressed-tensors rung derive no
family, land out of scope, and are counted and reported rather than refused.
The scope test comes from the published table, never from a list typed here.

Vocabulary note. Principle 9's lane enum is
``backed | backed_with_serve_flag | unbacked``; this module uses it verbatim,
plus ``unattested`` for the no-claim state and, at *regime* granularity only,
``fallback`` for a route that serves by an announced non-native path.
``allocator_candidates.ROUTE_STATUS_*`` is a DIFFERENT and older tri-state
(``backed | pending | blocked``) describing source-passthrough contracts; the
two are deliberately not unified here -- one describes a passthrough rung's
audit state, the other a lane's executed route under the pinned release.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import record_fields
from .digests import file_sha256hex


#: Schema of the eligibility table PrismaQuant consumes, published by Tessera's
#: own vLLM plugin
#: (``tessera.serving``, entry point ``tessera``, ``quant_method: "tessera"``).
#: v4 adds launches/residency; v5 adds exact runtime image/execution scope;
#: v10 turns each ``platforms`` entry from a bare key into an object that
#: states the platform's backend and what it EXECUTES per family (Tessera
#: #456, contract v23), so the table can say a family has no native route
#: on a device before any cell on that device exists.
#: The parser owns these grammars; plugin requirements remain optional only
#: for explicitly identified legacy v3 tables.
#:
#: Until 2026-09-02 this set also carried ``gridbook.lane-eligibility.v3``, the
#: same wire format published by the retired Gridbook codebook lane. That lane
#: was removed with Rob's decision to put Tessera in PrismaQuant and remove
#: Gridbook; see ``archive/gridbook_lane_2026-09-02/README.md``.
LANE_ELIGIBILITY_SCHEMA_TESSERA_V12 = "tessera.lane-eligibility.v12"
LANE_ELIGIBILITY_SCHEMA_TESSERA_V11 = "tessera.lane-eligibility.v11"
LANE_ELIGIBILITY_SCHEMA_TESSERA_V10 = "tessera.lane-eligibility.v10"
LANE_ELIGIBILITY_SCHEMA_TESSERA_V9 = "tessera.lane-eligibility.v9"
LANE_ELIGIBILITY_SCHEMA_TESSERA_V8 = "tessera.lane-eligibility.v8"
LANE_ELIGIBILITY_SCHEMA_TESSERA_V7 = "tessera.lane-eligibility.v7"
LANE_ELIGIBILITY_SCHEMA_TESSERA_V6 = "tessera.lane-eligibility.v6"
LANE_ELIGIBILITY_SCHEMA_TESSERA_V5 = "tessera.lane-eligibility.v5"
LANE_ELIGIBILITY_SCHEMA_TESSERA_V4 = "tessera.lane-eligibility.v4"
LANE_ELIGIBILITY_SCHEMA_TESSERA_LEGACY_V3 = "tessera.lane-eligibility.v3"

#: The legacy v10 schema alias. It is not the active pin or the accepted
#: grammar roster. Scope is a property the authoritative set answers, not a
#: single constant: see :data:`SCOPED_LANE_SCHEMAS`. Keeping an old alias
#: must never silently demote another supported grammar to unscoped.
LANE_ELIGIBILITY_SCHEMA_TESSERA = LANE_ELIGIBILITY_SCHEMA_TESSERA_V10

#: The schemas whose cells carry a per-cell runtime scope, so an explicit
#: serving context (image + execution mode) can be matched rather than
#: borrowed from a global field. v5 introduced the block; v6 widened it with
#: the vLLM and torch versions the cell was measured under; v7, v8 and v9
#: widened the EVIDENCE block (a smoke's control, an artifact's encoder scope,
#: a smoke's record) and left the runtime scope as v6 published it; v10
#: widened the PLATFORM entry and left every cell byte-identical, which is
#: why it belongs in this set and in each evidence set below.
SCOPED_LANE_SCHEMAS = frozenset({
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V12,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V11,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V5,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V6,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V7,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V8,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V9,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V10,
})

#: The schemas whose cells carry a required ``evidence`` block (v6 and every
#: grammar after it), the ones whose ``smoke`` names its control and derived
#: attribution (v7, Tessera #195) and the ones whose evidence names the
#: artifact and encoder its KL was measured on (v8, Tessera #198). Each is a
#: set and not an ``== V6`` so that the NEXT bump cannot silently demote the
#: grammar it succeeds to "publishes no evidence".
EVIDENCE_LANE_SCHEMAS = frozenset({
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V12,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V11,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V6,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V7,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V8,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V9,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V10,
})
ATTRIBUTED_SMOKE_LANE_SCHEMAS = frozenset({
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V12,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V11,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V7,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V8,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V9,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V10,
})
ENCODER_SCOPED_LANE_SCHEMAS = frozenset({
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V12,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V11,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V8,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V9,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V10,
})

#: The schemas whose ``smoke`` carries a ``record`` -- the rule a status was
#: derived by, the instrument that applied it, and the (prompt, form,
#: interface) rows it was applied to (v9, Tessera #327).  On these tables the
#: status and the attribution are RE-DERIVED through Tessera's own functions
#: rather than through a rule restated here; see :func:`_parse_smoke_record`.
#: v10 republishes the same ten cells byte for byte, so it carries the
#: record too -- a set v10 were missing from would read an attested cell as
#: one that publishes no evidence.
RECORDED_SMOKE_LANE_SCHEMAS = frozenset({
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V12,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V11,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V9,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V10,
})

#: The schemas whose ``platforms`` entries are OBJECTS rather than bare keys
#: (v10, Tessera #456). Under v9 and earlier the value was never read: the
#: only thing a platform key answered was whether a cell could name it, so the
#: single way to say anything about a device was to have served on it. v10
#: lets the document state what a platform EXECUTES per family before any cell
#: on it exists, and ``null`` there is a claim somebody looked -- which is the
#: measured platform fact principle 9's carve-out turns on, and the reason
#: this bump is not additive.
PLATFORM_AXIS_LANE_SCHEMAS = frozenset({
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V12,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V11,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V10,
})

#: The backends a platform entry may declare. TRANSCRIBED from the publisher's
#: own validator (``tessera.serving.contract.PLATFORM_BACKENDS``), closed the
#: same way :data:`CELL_ROUTE_STATUSES` is. The device TYPE cannot answer this
#: question: a ROCm torch reports ``device.type == "cuda"`` for an AMD device.
PLATFORM_BACKENDS = frozenset({"cuda", "hip"})

#: The key that names the hardware, one per backend, and exactly one is
#: present. A CUDA platform is a compute capability and a HIP platform is a
#: gcnArchName; an entry carrying both would be two devices under one key.
#: Transcribed from ``tessera.serving.contract.PLATFORM_ARCH_KEYS``.
PLATFORM_ARCH_KEYS = {"cuda": "compute_capability", "hip": "gcn_arch"}

#: Keys a platform entry may carry beyond the required ones. They describe the
#: machine rather than what it executes, and nothing here reads them; they are
#: named so a closed key check does not refuse the entries the runtime ships.
PLATFORM_OPTIONAL_KEYS = frozenset({"wavefront", "lds_bytes"})

#: What a platform entry says about one family, when the table declares the
#: platform at all. A platform key the table does not carry is a THIRD state
#: and not a synonym for ``None``: the document declined to answer, and a
#: reader that folded the two together would report an unread question as a
#: measured refusal.
PLATFORM_EXECUTES_UNSTATED = "unstated"

#: Every eligibility-table schema this parser accepts. The check is a set
#: membership, never a prefix match: an unrecognised vendor is a table this
#: repository was not handed, and an unlisted version is not treated as a
#: subset of either supported grammar (see ``_parse_table``).
LANE_ELIGIBILITY_SCHEMAS = frozenset({
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V12,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V11,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V10,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V9,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V8,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V7,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V6,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V5,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V4,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_LEGACY_V3,
})

#: The residency vocabulary in Tessera's v4 flag grammar. Each format row
#: publishes the subset its route supports; no cell can widen that subset.
TESSERA_RESIDENCY_MODES = frozenset({"resident", "streamed"})
TESSERA_EXECUTION_MODES = frozenset({"eager", "compiled"})
_LAUNCH_SCHEMAS = frozenset({
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V4,
} | SCOPED_LANE_SCHEMAS)
#: The schemas whose cells derive rule coverage from an ``allowable_rungs``
#: window rule (v11) and keep that coverage beside the census rungs. v12
#: inherits the rule and adds per-launch census scopes beside it.
_RULE_COVERAGE_SCHEMAS = frozenset({
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V12,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V11,
})
_DIGEST_IMAGE = re.compile(
    r"[a-z0-9][a-z0-9._/-]*[a-z0-9]@sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

#: Schema of the provenance payload this module produces. It was
#: ``prismaquant.cb_route_attestation.v2`` until 2026-09-02, when the Gridbook
#: codebook lane was retired: the name and its ``gridbook_serving_*`` fields
#: both named a runtime that no longer has a lane here, and the only reader of
#: those fields (``cb_route_status_gate``) went into the archive with it, so
#: the rename costs no shipped artifact a reader.
ROUTE_ATTESTATION_SCHEMA = "prismaquant.lane_route_attestation.v3"

# --- Principle 9's lane vocabulary, verbatim. -------------------------------
ROUTE_STATUS_BACKED = "backed"
ROUTE_STATUS_BACKED_WITH_SERVE_FLAG = "backed_with_serve_flag"
ROUTE_STATUS_UNBACKED = "unbacked"
#: Not one of principle 9's three: the honest state when the pinned runtime
#: publishes no claim covering this unit -- because it packages no eligibility
#: table at all, or because no cell in the table it does package names this
#: platform/family/rung. It is a REFUSAL TO CLAIM, not a verdict.
ROUTE_STATUS_UNATTESTED = "unattested"
#: Regime granularity only. The route serves, by an announced non-native path.
#: Rolls up into a unit-level ``backed`` plus a recorded fallback regime.
ROUTE_STATUS_FALLBACK = "fallback"

LANE_ROUTE_STATUSES = frozenset({
    ROUTE_STATUS_BACKED,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
    ROUTE_STATUS_UNBACKED,
})
REGIME_ROUTE_STATUSES = LANE_ROUTE_STATUSES | {ROUTE_STATUS_FALLBACK}

#: The CLOSED set a packaged cell may declare, mirroring the publisher's
#: ``_LANE_ROUTE_STATUSES`` exactly. ``unbacked`` is absent on purpose: the
#: runtime never enumerates what it refuses, so a cell claiming ``unbacked`` is
#: a table this repository must not have been handed. Accepting one would make
#: this parser laxer than the publisher's own validator.
CELL_ROUTE_STATUSES = frozenset({
    ROUTE_STATUS_BACKED,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
    ROUTE_STATUS_FALLBACK,
})

#: How far a cell's claim was taken. ``compile_only`` means the kernels
#: cross-compile for that compute capability and nothing more; only
#: ``device_qualified`` means a real serve on that device loaded, dispatched
#: and generated. Both are recorded; neither is silently upgraded.
QUALIFICATION_COMPILE_ONLY = "compile_only"
QUALIFICATION_DEVICE_QUALIFIED = "device_qualified"
CELL_QUALIFICATIONS = frozenset({
    QUALIFICATION_COMPILE_ONLY,
    QUALIFICATION_DEVICE_QUALIFIED,
})

# --- Schema v6: what EVIDENCE a cell rests on ------------------------------
#: Transcribed from the publisher's own validator
#: (``tessera.serving.contract.EVIDENCE_KL_KINDS`` /
#: ``EVIDENCE_SMOKE_STATUSES`` / ``EVIDENCE_GRADES``), closed the same way
#: :data:`CELL_ROUTE_STATUSES` is. A grade or status this reader does not know
#: is a table it was not handed, and guessing at one is how an unmeasured cell
#: gets read as a measured one.
EVIDENCE_KL_KIND_TOPK_LOWER_BOUND = "topk_intersection_lower_bound"
EVIDENCE_KL_KIND_FULL_VOCAB = "full_vocab"
EVIDENCE_KL_KINDS = frozenset({
    EVIDENCE_KL_KIND_TOPK_LOWER_BOUND,
    EVIDENCE_KL_KIND_FULL_VOCAB,
})

#: A greedy smoke's outcome, in the receipt's own words. ``repetitive`` is not
#: a softer ``recorded``: it says a smoke WAS run on this cell's route and the
#: model degenerated. Principle 9 requires a route to generate correctly, so
#: that is a measured serving defect published in a field a gate can read.
EVIDENCE_SMOKE_RECORDED = "recorded"
EVIDENCE_SMOKE_REPETITIVE = "repetitive"
EVIDENCE_SMOKE_NOT_RECORDED = "not_recorded"
EVIDENCE_SMOKE_STATUSES = frozenset({
    EVIDENCE_SMOKE_RECORDED,
    EVIDENCE_SMOKE_REPETITIVE,
    EVIDENCE_SMOKE_NOT_RECORDED,
})

#: The smoke outcomes that REFUSE a cell. One member today, and a set rather
#: than an ``== "repetitive"`` so a new failing outcome is a data change here
#: instead of a new branch at every call site.
EVIDENCE_SMOKE_REFUSALS = frozenset({EVIDENCE_SMOKE_REPETITIVE})

# --- Schema v7 (Tessera #195): the CONTROL a greedy smoke was compared to ---
#: Transcribed from ``tessera.serving.contract.EVIDENCE_CONTROL_REFERENCES`` /
#: ``EVIDENCE_CONTROL_OUTCOMES`` / ``EVIDENCE_SMOKE_ATTRIBUTIONS`` /
#: ``EVIDENCE_CONTROL_KEYS``. A smoke's ``control`` is either ``null`` (nobody
#: ran the reference) or ``{reference, outcome, receipt}``: the same prompt,
#: byte for byte, against the unquantised source the route is a quantisation
#: of, under the smoke's own runtime and execution mode. ``outcome`` says
#: whether the reference returned the SAME completion -- and nothing about
#: whether the reference was healthy, which no string comparison decides.
EVIDENCE_CONTROL_BF16_SOURCE = "bf16_source"
EVIDENCE_CONTROL_REFERENCES = frozenset({EVIDENCE_CONTROL_BF16_SOURCE})
EVIDENCE_OUTCOME_IDENTICAL = "identical_completion"
EVIDENCE_OUTCOME_DIFFERENT = "different_completion"
EVIDENCE_CONTROL_OUTCOMES = frozenset({
    EVIDENCE_OUTCOME_IDENTICAL,
    EVIDENCE_OUTCOME_DIFFERENT,
})
EVIDENCE_CONTROL_KEYS = frozenset({"reference", "outcome", "receipt"})

#: What the control DERIVES about where a symptom lives. Read off the control
#: and checked, exactly as the grade is read off the KL entries: no control is
#: ``unattributed`` (the status is an observation, not an attribution); an
#: identical completion is ``shared_with_reference`` (the model and the prompt
#: produce it, not this route); a different one is
#: ``not_shared_with_reference`` -- weaker than "the route is at fault", and
#: deliberately not spelled that way.
EVIDENCE_ATTRIBUTION_UNATTRIBUTED = "unattributed"
EVIDENCE_ATTRIBUTION_SHARED = "shared_with_reference"
EVIDENCE_ATTRIBUTION_NOT_SHARED = "not_shared_with_reference"
EVIDENCE_SMOKE_ATTRIBUTIONS = frozenset({
    EVIDENCE_ATTRIBUTION_UNATTRIBUTED,
    EVIDENCE_ATTRIBUTION_SHARED,
    EVIDENCE_ATTRIBUTION_NOT_SHARED,
})

# --- Schema v8 (Tessera #198): the ENCODER the evidence is scoped to --------
#: Transcribed from ``tessera.serving.contract.EVIDENCE_PAYLOAD_RELATIONS`` /
#: ``EVIDENCE_WEIGHT_ERROR_RELATIONS``. ``evidence.artifact`` is ``null`` when
#: no encoder-reproduction comparison was recorded; otherwise it names the
#: historical artifact a cell's KL was measured on, the encoder commit that
#: wrote it, and a SINGLE-UNIT re-encode screen at a later commit: whether the
#: payload bytes came out identical and how the weight SSE moved. It is a
#: weight-space screen and never served KL; it never changes the grade.
EVIDENCE_PAYLOAD_IDENTICAL = "identical"
EVIDENCE_PAYLOAD_DIFFERENT = "different"
EVIDENCE_PAYLOAD_RELATIONS = frozenset({
    EVIDENCE_PAYLOAD_IDENTICAL,
    EVIDENCE_PAYLOAD_DIFFERENT,
})
EVIDENCE_WEIGHT_ERROR_LOWER = "lower"
EVIDENCE_WEIGHT_ERROR_EQUAL = "equal"
EVIDENCE_WEIGHT_ERROR_HIGHER = "higher"
EVIDENCE_WEIGHT_ERROR_RELATIONS = frozenset({
    EVIDENCE_WEIGHT_ERROR_LOWER,
    EVIDENCE_WEIGHT_ERROR_EQUAL,
    EVIDENCE_WEIGHT_ERROR_HIGHER,
})
#: The only metric the screen may name. It is a constant and not a set so the
#: reader cannot be widened to "any metric" by a data edit: a served-KL number
#: written here would be read as weight-space evidence.
EVIDENCE_ARTIFACT_METRIC = "weight_sse"
_FULL_GIT_SHA1 = re.compile(r"\A[0-9a-f]{40}\Z")

EVIDENCE_GRADE_ROUTE_ONLY = "route_only"
EVIDENCE_GRADE_KL_LOWER_BOUND = "kl_lower_bound"
EVIDENCE_GRADE_KL_FULL_VOCAB = "kl_full_vocab"
EVIDENCE_GRADES = frozenset({
    EVIDENCE_GRADE_ROUTE_ONLY,
    EVIDENCE_GRADE_KL_LOWER_BOUND,
    EVIDENCE_GRADE_KL_FULL_VOCAB,
})

#: Every receipt path is repository-relative under this root. A wheel ships no
#: docs, so this reader checks the GRAMMAR and never the file.
EVIDENCE_RECEIPT_ROOT = "docs/measurements/"

#: ``native_extensions[].lane`` (contract v20, Tessera #264): the decoder an
#: extension serves and, optionally, ``requires`` -- the predicate a unit's
#: wire must satisfy for that kernel to read it. The vocabulary below is
#: transcribed from Tessera's own contract validator
#: (``tessera.serving.contract.LANE_FIELDS`` / ``LANE_REQUIREMENT_FIELDS``)
#: and CLOSED here: a requirement outside it is refused by name, never
#: skipped, because a gate that skipped a published condition would call a
#: unit selectable that the loader refuses. This is vocabulary, not the rule:
#: the DECISION is Tessera's ``scheme.decide_lane_requirements`` and
#: :func:`cell_lane_admits` calls it rather than restating it.
LANE_FIELDS = frozenset({"decoder", "requires"})
LANE_REQUIREMENT_FIELDS = (
    "column_rates", "window_bits", "body", "plane", "release_overrides",
    "diagonals", "rotation", "start_state", "grid_arities",
    # Tessera v45: the rates the ROUTED-EXPERT launch reaches, a subset of
    # ``column_rates`` (the gate/up launch's two tables do not fit the
    # target's shared memory at the higher rates).  STRUCTURE-scoped: decided
    # only for a ``routed_moe`` unit, over the unit's stated structure.
    "column_rates_routed_moe",
)
#: Non-empty ascending unique positive integer lists.
LANE_REQUIREMENT_LISTS = frozenset({
    "column_rates", "window_bits", "grid_arities", "column_rates_routed_moe"})
#: JSON booleans: whether the lane reads units that CARRY the thing.
LANE_REQUIREMENT_CARRIES = frozenset({"release_overrides", "diagonals", "start_state"})
#: Checkpoint-dialect spellings, exactly as ``formats[].attested_wire`` and
#: the lane predicate publish them; Tessera's ``route_wire_spelling`` maps
#: them onto its manifest names at decision time.
LANE_ROTATION_STATES = frozenset({"none", "r_in_only"})
LANE_BODIES = frozenset({"tcq", "window"})
LANE_PLANES = frozenset({"s6b", "lut16", "channel"})

#: Structural classes a unit can belong to. The two take different runtime
#: dispatch paths and therefore different eligibility cells.
STRUCTURE_DENSE = "dense"
STRUCTURE_ROUTED_MOE = "routed_moe"
STRUCTURES = frozenset({STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE})

#: The ``formats[].kind`` discriminator. It lives on the FORMAT row, never on a
#: lane cell -- a cell's rung vocabulary follows from its family's kind.
FORMAT_KIND_CB_PRODUCT = "cb_product"
FORMAT_KIND_TCQ_TRELLIS = "tcq_trellis"
#: Tessera's discriminator for the same idea: a family addressed by a RATE
#: (body bits per 256 weights), not by a codebook size. ``tcq_trellis`` is
#: the retired lane's spelling of it and ``tessera_wire`` is Tessera's; both resolve to
#: the ``rungs_q256`` rung vocabulary and to ``EligibilityCell.is_trellis``.
FORMAT_KIND_TESSERA_WIRE = "tessera_wire"
FORMAT_KINDS = frozenset({
    FORMAT_KIND_CB_PRODUCT,
    FORMAT_KIND_TCQ_TRELLIS,
    FORMAT_KIND_TESSERA_WIRE,
})

#: The kinds whose rung axis is a RATE. ``EligibilityCell.is_trellis`` means
#: exactly "rate-addressed" -- the name is historical, from the era when
#: ``tcq_trellis`` was the only such kind -- and every dispatch on the rung
#: vocabulary tests membership here, never one kind constant. There are two
#: such dispatch sites (``_published_families`` and ``resolve_payload_rung``)
#: and they must agree, or a name resolves to a family with no rate and every
#: downstream cell match fails closed for the wrong reason.
RATE_ADDRESSED_FORMAT_KINDS = frozenset({
    FORMAT_KIND_TCQ_TRELLIS,
    FORMAT_KIND_TESSERA_WIRE,
})

class LaneEligibilityError(ValueError):
    """The materialized contract or its eligibility table is malformed."""


@dataclass(frozen=True)
class ServingContext:
    """The explicit target of a cell lookup; no field has a runtime default."""

    platform: str
    structure: str
    residency: str
    runtime_image: str
    execution_mode: str
    kernel_build: str | None = None

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise LaneEligibilityError(f"serving_context.{name} must be a non-empty string")
        for name, allowed in (("structure", STRUCTURES),
                              ("residency", TESSERA_RESIDENCY_MODES),
                              ("execution_mode", TESSERA_EXECUTION_MODES)):
            if getattr(self, name) not in allowed:
                raise LaneEligibilityError(
                    f"serving_context.{name} must be one of {sorted(allowed)}")
        if not _DIGEST_IMAGE.fullmatch(self.runtime_image):
            raise LaneEligibilityError(
                "serving_context.runtime_image must be an exact repository@sha256:<64 lowercase hex> reference")

    def as_dict(self) -> dict[str, str]:
        values = asdict(self)
        if self.kernel_build is None:
            values.pop("kernel_build")
        return values

    def key(self) -> tuple[str, ...]:
        values = self.as_dict()
        if self.kernel_build is not None:
            values.pop("runtime_image")
        return tuple(values.values())


#: The default of every serving-code check: read the tracked pin's digest
#: (:func:`tessera_serving_runtime_pin.pinned_serving_source_sha256`). It is a
#: sentinel rather than ``None`` because ``None`` means "skip", and a caller
#: that omits the keyword must be checked against the pin, not skipped.
PINNED_SERVING_SOURCE = object()


def resolve_serving_source_sha256(value: Any = PINNED_SERVING_SOURCE) -> str | None:
    """The digest a code check compares against: the argument, or the tracked pin's.

    ``None`` is the v2 answer (no code check); a string is a v3 pin's digest.
    """
    if value is PINNED_SERVING_SOURCE:
        from .tessera_serving_runtime_pin import pinned_serving_source_sha256

        return pinned_serving_source_sha256()
    if value is not None and (not isinstance(value, str) or not _SHA256.fullmatch(value)):
        raise LaneEligibilityError(
            f"serving_source_sha256 must be None or 64 lowercase hex digits, got {value!r}")
    return value


def cell_serving_code_admits(
    cell: Any, serving_source_sha256: Any = PINNED_SERVING_SOURCE,
) -> tuple[bool, str]:
    """Whether a cell's evidence was taken on the code the pin serves (#1561).

    ``serving_source_sha256`` is the pinned serving code digest. ``None`` (a
    v2 pin) skips the check. A string (a v3 pin) requires the cell's
    ``runtime.serving_source_sha256`` to equal it; a cell that names no code
    was measured on code nobody recorded, so it does not match either. The
    refusal names the cell and both digests, so a unit left unattested by it
    says which code the evidence was taken on and which code serves.
    """
    pinned = resolve_serving_source_sha256(serving_source_sha256)
    if pinned is None:
        return True, ""
    stamped = getattr(cell, "runtime_serving_source_sha256")
    cell_id = getattr(cell, "id", None) or getattr(cell, "cell_id", "")
    if not stamped:
        return False, (
            f"cell {cell_id!r} names no serving code (runtime.serving_source_sha256 "
            f"is absent), and the pinned serving code is {pinned}; a cell measured "
            "on unrecorded code does not attest the pinned code")
    if stamped != pinned:
        return False, (
            f"cell {cell_id!r} was measured on serving code {stamped} "
            f"(tessera commit {getattr(cell, 'runtime_tessera_commit', '')}), "
            f"and the pinned serving code is {pinned}; evidence taken on other "
            "code does not attest the pinned code")
    return True, ""


def cell_matches_serving_context(
    cell: Any,
    context: ServingContext,
    *,
    serving_source_sha256: Any = PINNED_SERVING_SOURCE,
) -> bool:
    """Match a parsed v5 cell's whole scope, shared by every admission path.

    Since #1561 the scope includes the serving code: see
    :func:`cell_serving_code_admits`. The default reads the tracked pin, so a
    v2 pin (no digest) matches exactly as before and a v3 pin cannot be
    skipped by a caller that passes nothing.
    """
    return (
        cell.platform == context.platform
        and cell.structure == context.structure
        and context.residency in cell.residency_modes
        and ((cell_kernel_build(cell) == context.kernel_build)
             if context.kernel_build is not None else cell.runtime_image == context.runtime_image)
        and context.execution_mode in cell.execution_modes
        and cell_serving_code_admits(cell, serving_source_sha256)[0]
    )


def legacy_runtime_scope_refusal(schema: str) -> str:
    """The one refusal for a scoped query a legacy table cannot attest."""
    return (
        f"lane schema {schema!r} carries no per-cell runtime scope; an explicit "
        f"serving context (runtime-image, kernel-build, or execution query) requires one of "
        f"{sorted(SCOPED_LANE_SCHEMAS)!r}. "
        "Global runtime identity is not a scoped admission."
    )


# ---------------------------------------------------------------------------
# Structural facts of one selected unit
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UnitStructuralFacts:
    """What the EXPORT knows about a unit, in the runtime's own vocabulary.

    Every field is a structural fact of the bytes the exporter is about to
    write, not a producer opinion: the payload family and rung name the codec,
    ``n_sub`` the sub-table split, ``rate_q256`` a trellis unit's body bits per
    256 weights, ``role_split`` whether an expert stack binds more than one
    codebook across its projections, and the two shape fields the load gates.
    An eligibility cell may predicate on any of them.

    ``k`` and ``rate_q256`` are the two rung vocabularies and they are mutually
    exclusive by construction: a ``cb_product`` family carries a codebook ``k``,
    a RATE-addressed family (``tcq_trellis`` or ``tessera_wire``) carries a
    rate. Neither is ever a rounded bpw.
    Both stay ``None`` when the pinned release publishes no such rung, so every
    rung predicate and every cell match fails closed rather than passing on a
    rate the runtime never listed.

    ``role_split`` is the fact the DSv4 defect turned on and the one no
    producer-side structure carried: it is knowable ONLY at export, after the
    per-``(qname, format)`` codebook cells resolve, which is why the gate lives
    at export rather than at allocation.
    """

    qname: str
    format_name: str
    payload_family: str
    k: int | None
    n_sub: int | None
    structure: str
    role_split: bool
    in_features: int
    out_features: int
    #: Trellis body bits per 256 weights. ``None`` for every CB / passthrough /
    #: stock unit, and ``None`` for a trellis name whose rate falls outside the
    #: family's published ``reader_rate_range_q256``.
    rate_q256: int | None = None

    def __post_init__(self) -> None:
        if self.structure not in STRUCTURES:
            raise LaneEligibilityError(
                f"{self.qname}: structure must be one of "
                f"{sorted(STRUCTURES)}, got {self.structure!r}")
        if self.k is not None and self.rate_q256 is not None:
            raise LaneEligibilityError(
                f"{self.qname}: a unit carries a codebook rung OR a trellis "
                f"rate, never both; got k={self.k} rate_q256={self.rate_q256}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "qname": self.qname,
            "format": self.format_name,
            "payload_family": self.payload_family,
            "k": self.k,
            "n_sub": self.n_sub,
            "rate_q256": self.rate_q256,
            "structure": self.structure,
            "role_split": self.role_split,
            "in_features": self.in_features,
            "out_features": self.out_features,
        }

    def fact(self, name: str) -> Any:
        if name not in _PREDICABLE_FACTS:
            raise LaneEligibilityError(
                f"eligibility cell predicates on unknown fact {name!r}; "
                f"the attestable facts are {sorted(_PREDICABLE_FACTS)}")
        return getattr(self, _PREDICABLE_FACTS[name])


#: The closed set of facts a packaged eligibility cell may predicate on. A cell
#: naming anything else is a malformed contract, not a silently ignored cell --
#: an unknown predicate that no-ops would let a newer runtime's narrower cell
#: read as unconditionally eligible.
_PREDICABLE_FACTS: dict[str, str] = {
    "payload_family": "payload_family",
    "k": "k",
    "n_sub": "n_sub",
    "rate_q256": "rate_q256",
    "role_split": "role_split",
    "in_features": "in_features",
    "out_features": "out_features",
}


# ---------------------------------------------------------------------------
# The packaged table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CellKlEvidence:
    """One KL receipt a v6 cell rests on, in the publisher's own vocabulary.

    There is no NUMBER here, and that is the publisher's design: the receipt
    holds the value with its bounds and caveats, and a bare float beside a
    grade is exactly the prose-shaped field principle 14 refuses. What this
    reader keeps is what a gate can decide on -- which KIND of measurement it
    was, how wide a top-K it covered, which regime it was scored in, and under
    which execution modes.
    """

    kind: str
    #: Positive for a top-K intersection bound; ``None`` for a full-vocab KL.
    top_k: int | None
    regime: str
    execution_modes: tuple[str, ...]
    receipt: str
    #: The rung the receipt was scored at, published since Tessera contract
    #: v32 (#560's D2b): a cell whose family covers several rungs must say
    #: WHICH one each KL measured, or a receipt taken at one rate stands in
    #: for every rate the cell attests.  ``None`` on every entry published
    #: before that grammar, which is every entry the previous pin carried.
    q256: int | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "kind": self.kind,
            "top_k": self.top_k,
            "regime": self.regime,
            "execution_modes": list(self.execution_modes),
            "receipt": self.receipt,
        }
        if self.q256 is not None:
            out["q256"] = self.q256
        return out


@dataclass(frozen=True)
class SmokeControl:
    """The reference a v7 greedy smoke was compared against (Tessera #195).

    ``outcome`` establishes exactly one thing: whether the reference returned
    the SAME completion for the same prompt under the smoke's own runtime and
    execution mode. It does not establish that the reference was healthy.
    """

    reference: str
    outcome: str
    receipt: str

    def as_dict(self) -> dict[str, Any]:
        return {"reference": self.reference, "outcome": self.outcome,
                "receipt": self.receipt}


@dataclass(frozen=True)
class EvidenceArtifact:
    """The encoder scope of a v8 cell's evidence (Tessera #198).

    The KL a cell publishes was measured on bytes SOME encoder wrote. This
    names that artifact and commit, and one later commit's re-encode of a
    single unit from the same source: did the payload come out identical, and
    which way did the weight SSE move. It describes the named commit and unit
    only -- never every unit, never a future encoder -- and it is a
    weight-space screen, never served KL.
    """

    id: str
    encoder_commit: str
    reencode_encoder_commit: str
    reencode_unit: str
    reencode_payload: str
    reencode_weight_error: str
    reencode_receipt: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "encoder_commit": self.encoder_commit,
            "reencode": {
                "encoder_commit": self.reencode_encoder_commit,
                "unit": self.reencode_unit,
                "payload": self.reencode_payload,
                "metric": EVIDENCE_ARTIFACT_METRIC,
                "weight_error": self.reencode_weight_error,
                "receipt": self.reencode_receipt,
            },
        }

    def answer(self) -> list[Any]:
        return [self.id, self.encoder_commit, self.reencode_encoder_commit,
                self.reencode_unit, self.reencode_payload,
                self.reencode_weight_error]


@dataclass(frozen=True)
class SmokeRecordRow:
    """One observation behind a smoke status (v9, Tessera #327).

    ``prompt`` names WHICH prompt was run, never the completion it produced:
    the contract records the shape of the observation and points at a receipt
    for the text. ``status``/``reference_status`` are the same vocabulary the
    cell's own status uses, for the route and for the reference arm. The
    reference status is null exactly when the record names no reference arm.
    """

    prompt: str
    form: str
    interface: str
    status: str
    reference_status: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "form": self.form,
                "interface": self.interface, "status": self.status,
                "reference_status": self.reference_status}


@dataclass(frozen=True)
class SmokeRecord:
    """v9's ``smoke.record``: the rule a status was derived by, and its rows.

    It exists because of what Tessera #327 found in contract v21: both
    ``routed_moe`` cells published ``status: "recorded"`` on an aggregation
    rule that lived only in a dated measurements file, was derived and checked
    by nothing, and was satisfiable by an empty completion. Putting the rule
    and its observations in the contract makes the status a DERIVATION a
    consumer can re-run instead of an assertion it must take on trust.

    ``None`` where no record was published -- on a pre-v9 table, and on a v9
    cell nobody re-ran (``record: null``), which is not the same thing as a
    record with no rows and is refused from being spelled that way.
    """

    instrument: str
    rule: str
    reference: str | None
    rows: tuple[SmokeRecordRow, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"instrument": self.instrument, "rule": self.rule,
                "reference": self.reference,
                "rows": [row.as_dict() for row in self.rows]}

    def answer(self) -> list[Any]:
        """The projection a re-review must see move (see ``contract_answer``).

        The rule is in it because a status derived by a DIFFERENT rule is a
        different claim wearing the same word, which is the failure #327
        reports; the rows are in it because dropping one changes what the
        status rests on without changing the status.
        """
        return [self.instrument, self.rule, self.reference,
                [list(row.as_dict().values()) for row in self.rows]]


@dataclass(frozen=True)
class CellEvidence:
    """A cell's ``evidence`` block: what its route claim actually rests on.

    Schema v6 made this required on every cell, and it is the first field in
    the lane table that says something about QUALITY rather than dispatch. The
    grade is derived from the KL entries' kinds and re-derived here, never
    trusted as written: "the grade is read off the entries, never asserted
    beside them" is the publisher's own rule and a consumer that took the
    written grade would be trusting an assertion where a derivation exists.

    v7 added the smoke's CONTROL and the attribution derived from it; v8 added
    the ARTIFACT the evidence is scoped to; v9 added the smoke's RECORD, the
    rule and rows its status was derived from. A table older than the field
    leaves it at its "never published" value -- ``""``/``None`` -- which is
    distinct from v7's ``unattributed`` and v8's/v9's ``null`` on purpose: a v6
    table did not say "nobody ran the reference", it said nothing.
    """

    grade: str
    kl: tuple[CellKlEvidence, ...]
    smoke_status: str
    #: The recorded smoke's receipt, or "" when no smoke was recorded.
    smoke_receipt: str = ""
    #: v7: one of :data:`EVIDENCE_SMOKE_ATTRIBUTIONS`, or "" on a pre-v7 table.
    smoke_attribution: str = ""
    #: v7: the control the attribution was derived from; ``None`` when nobody
    #: ran one AND on a pre-v7 table (``smoke_attribution`` tells them apart).
    smoke_control: SmokeControl | None = None
    #: v8: the encoder scope, or ``None`` when no comparison was recorded AND
    #: on a pre-v8 table.
    artifact: EvidenceArtifact | None = None
    #: v9: the rule and rows the status was derived from; ``None`` when no
    #: record was published AND on a pre-v9 table.
    smoke_record: SmokeRecord | None = None
    #: v9: whether ``smoke_status`` was DERIVED from that record or merely
    #: asserted -- Tessera's ``smoke_status_is_derived``, asked rather than
    #: inferred here. False on every pre-v9 table, where nothing could be
    #: derived because there was no record to derive from.
    smoke_status_is_derived: bool = False

    def as_dict(self) -> dict[str, Any]:
        smoke: dict[str, Any] = {"status": self.smoke_status,
                                 "receipt": self.smoke_receipt or None}
        if self.smoke_attribution:
            smoke["attribution"] = self.smoke_attribution
            smoke["control"] = (self.smoke_control.as_dict()
                                if self.smoke_control else None)
        if self.smoke_record is not None:
            smoke["record"] = self.smoke_record.as_dict()
            # Provenance says which cells' words were derived, so a shipcard
            # can tell an attested status from an asserted one (principle 12).
            smoke["status_is_derived"] = self.smoke_status_is_derived
        return {
            "grade": self.grade,
            "kl": [entry.as_dict() for entry in self.kl],
            "smoke": smoke,
            "artifact": self.artifact.as_dict() if self.artifact else None,
        }

    def answer(self) -> list[Any]:
        """The projection a re-review must see move (see ``contract_answer``).

        The attribution and the control's outcome are here because the
        refusal text names them and Tessera's own consumer rule decides on
        them; the artifact is here because a shipcard carries it. A control
        that flips from identical to different, or an encoder scope that
        appears or vanishes, is a re-review, not a silent bump.
        """
        return [self.grade, self.smoke_status,
                sorted(entry.as_dict()["kind"] + f"@{entry.top_k}"
                       + (f"@q{entry.q256}" if entry.q256 is not None else "")
                       for entry in self.kl),
                self.smoke_attribution,
                self.smoke_control.outcome if self.smoke_control else None,
                self.artifact.answer() if self.artifact else None,
                self.smoke_record.answer() if self.smoke_record else None]


def derive_evidence_grade(entries: Sequence[CellKlEvidence]) -> str:
    """The grade a cell's KL entries derive, from their kinds alone.

    Mirrors ``tessera.serving.contract.derive_evidence_grade``. No entry means
    ``route_only``: the census attests DISPATCH and nothing attests quality.
    """
    kinds = {entry.kind for entry in entries}
    if EVIDENCE_KL_KIND_FULL_VOCAB in kinds:
        return EVIDENCE_GRADE_KL_FULL_VOCAB
    if EVIDENCE_KL_KIND_TOPK_LOWER_BOUND in kinds:
        return EVIDENCE_GRADE_KL_LOWER_BOUND
    return EVIDENCE_GRADE_ROUTE_ONLY


def derive_smoke_attribution(control: SmokeControl | None) -> str:
    """What a smoke's control DERIVES about where the symptom lives.

    Mirrors ``tessera.serving.contract.derive_smoke_attribution``: no control
    is ``unattributed``; an identical completion is ``shared_with_reference``;
    anything else the control could say is ``not_shared_with_reference``.
    """
    if control is None:
        return EVIDENCE_ATTRIBUTION_UNATTRIBUTED
    if control.outcome == EVIDENCE_OUTCOME_IDENTICAL:
        return EVIDENCE_ATTRIBUTION_SHARED
    return EVIDENCE_ATTRIBUTION_NOT_SHARED


def _tessera_contract_module():
    """Tessera's own contract module -- the home of the v9 smoke vocabulary.

    A v9 lane table is, by construction, the packaged contract of an installed
    Tessera, so this import cannot be the thing that fails on a box that has a
    v9 table to read. It is a hard import for the same reason
    ``decide_lane_requirements`` is: a fallback that answers when Tessera
    cannot is a second home for Tessera's rule.
    """
    from tessera.serving import contract as _contract

    return _contract


def _tessera_published(name: str, where: str) -> Any:
    """One of Tessera's v9 names, or a refusal that says which one is missing.

    A v9 table beside a runtime too old to publish the rule it is written
    against is a mis-pin, and it has to say so by name rather than surface as
    an ``AttributeError`` from inside a parser.
    """
    module = _tessera_contract_module()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise LaneEligibilityError(
            f"{where} needs Tessera's {name} to decide, and the installed "
            "tessera.serving.contract does not publish it: this table calls "
            f"itself {LANE_ELIGIBILITY_SCHEMA_TESSERA_V9} but the runtime "
            "beside it is older. Re-pin, never restate the rule here -- a "
            "second copy is how the two halves of one contract drift."
        ) from exc


def _parse_smoke_record(payload: Any, where: str, *,
                        control: Any = None) -> SmokeRecord | None:
    """Read v9 records through the pinned producer's owning pure validator.

    Validating only vocabularies before deriving a status is insufficient:
    observation identity, portable paths and nullable reference arms are also
    producer grammar. The immutable pin binds this private metadata helper's
    API; a runtime lacking it refuses by name through ``_tessera_published``.
    No serving execution or kernel module is imported here.
    """
    validate = _tessera_published("_evidence_smoke_record", where)
    try:
        parsed = validate(payload, where, control, where.removesuffix(".record"))
    except ValueError as exc:
        raise LaneEligibilityError(str(exc)) from exc
    if parsed is None:
        return None
    return SmokeRecord(
        instrument=parsed["instrument"], rule=parsed["rule"],
        reference=parsed["reference"],
        rows=tuple(SmokeRecordRow(**row) for row in parsed["rows"]))


def _require_receipt(value: Any, where: str) -> str:
    if (not isinstance(value, str) or not value.startswith(EVIDENCE_RECEIPT_ROOT)
            or len(value) <= len(EVIDENCE_RECEIPT_ROOT)):
        raise LaneEligibilityError(
            f"{where}.receipt must be a repository path under "
            f"{EVIDENCE_RECEIPT_ROOT!r} (the receipt that holds the number and "
            f"its caveats), got {value!r}")
    return value


def _parse_smoke_control(payload: Any, where: str) -> SmokeControl | None:
    """v7's ``smoke.control``: ``null`` or the closed ``{reference, outcome, receipt}``."""
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise LaneEligibilityError(f"{where} must be null or a JSON object")
    _require_keys(payload, where, required=set(EVIDENCE_CONTROL_KEYS), optional=set())
    reference = payload["reference"]
    if reference not in EVIDENCE_CONTROL_REFERENCES:
        raise LaneEligibilityError(
            f"{where}.reference must be one of {sorted(EVIDENCE_CONTROL_REFERENCES)}, "
            f"got {reference!r}; a reference this reader cannot name is prose, "
            "and the whole point of the control is that a gate reads it")
    outcome = payload["outcome"]
    if outcome not in EVIDENCE_CONTROL_OUTCOMES:
        raise LaneEligibilityError(
            f"{where}.outcome must be one of {sorted(EVIDENCE_CONTROL_OUTCOMES)}, "
            f"got {outcome!r}")
    return SmokeControl(reference=str(reference), outcome=str(outcome),
                        receipt=_require_receipt(payload["receipt"], where))


def _require_full_sha1(value: Any, where: str) -> str:
    if not isinstance(value, str) or _FULL_GIT_SHA1.match(value) is None:
        raise LaneEligibilityError(
            f"{where}.encoder_commit must be a full lowercase Git SHA-1; a short "
            f"or floating ref does not name the encoder that wrote the bytes, "
            f"got {value!r}")
    return value


def _parse_evidence_artifact(payload: Any, where: str) -> EvidenceArtifact | None:
    """v8's ``evidence.artifact``: ``null`` or the closed encoder-scope record.

    Mirrors ``tessera.serving.contract._evidence_artifact`` rule for rule. An
    ``identical`` payload with a weight error other than ``equal`` is refused
    because the two cannot both be true of the same bytes; a metric other than
    ``weight_sse`` is refused because this screen is not served KL and a
    reader must not be widened into taking one for the other.
    """
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise LaneEligibilityError(f"{where} must be null or a JSON object")
    _require_keys(payload, where, required={"id", "encoder_commit", "reencode"},
                  optional=set())
    artifact_id = payload["id"]
    if (not isinstance(artifact_id, str) or not artifact_id
            or "\\" in artifact_id or any(c.isspace() for c in artifact_id)
            or any(part in ("", ".", "..") for part in artifact_id.split("/"))):
        raise LaneEligibilityError(
            f"{where}.id must be a portable relative artifact identifier, "
            f"got {artifact_id!r}")
    encoder_commit = _require_full_sha1(payload["encoder_commit"], where)
    reencode = payload["reencode"]
    spot = f"{where}.reencode"
    if not isinstance(reencode, Mapping):
        raise LaneEligibilityError(f"{spot} must be a JSON object")
    _require_keys(reencode, spot,
                  required={"encoder_commit", "unit", "payload", "metric",
                            "weight_error", "receipt"},
                  optional=set())
    reencode_commit = _require_full_sha1(reencode["encoder_commit"], spot)
    unit = reencode["unit"]
    if not isinstance(unit, str) or not unit.strip():
        raise LaneEligibilityError(
            f"{spot}.unit must name the single unit compared, got {unit!r}")
    relation = reencode["payload"]
    if relation not in EVIDENCE_PAYLOAD_RELATIONS:
        raise LaneEligibilityError(
            f"{spot}.payload must be one of {sorted(EVIDENCE_PAYLOAD_RELATIONS)}, "
            f"got {relation!r}")
    if reencode["metric"] != EVIDENCE_ARTIFACT_METRIC:
        raise LaneEligibilityError(
            f"{spot}.metric must be {EVIDENCE_ARTIFACT_METRIC!r}; this screen is "
            f"not served KL, got {reencode['metric']!r}")
    weight_error = reencode["weight_error"]
    if weight_error not in EVIDENCE_WEIGHT_ERROR_RELATIONS:
        raise LaneEligibilityError(
            f"{spot}.weight_error must be one of "
            f"{sorted(EVIDENCE_WEIGHT_ERROR_RELATIONS)}, got {weight_error!r}")
    if (relation == EVIDENCE_PAYLOAD_IDENTICAL
            and weight_error != EVIDENCE_WEIGHT_ERROR_EQUAL):
        raise LaneEligibilityError(
            f"{spot}.weight_error must be 'equal' for an identical payload; the "
            f"same bytes cannot carry a {weight_error!r} weight error")
    return EvidenceArtifact(
        id=artifact_id, encoder_commit=encoder_commit,
        reencode_encoder_commit=reencode_commit, reencode_unit=unit,
        reencode_payload=str(relation), reencode_weight_error=str(weight_error),
        reencode_receipt=_require_receipt(reencode["receipt"], spot))


def parse_cell_evidence(payload: Any, where: str, *, cell_regime: str,
                        execution_modes: Sequence[str] = (),
                        cell_rungs: Sequence[int] | None = None,
                        schema: str = LANE_ELIGIBILITY_SCHEMA_TESSERA) -> CellEvidence:
    """The ``evidence`` grammar at the table's schema.

    Every structural rule the publisher's validator enforces is re-checked
    here rather than assumed, because the two that matter most are exactly the
    ones a stale or hand-edited table would break: an entry's regime must be
    the CELL's regime (a prefill bound written into a decode cell is the
    confusion this field exists to refuse), and the written grade must equal
    the derived one.  Since the v32 grammar the same is true of an entry's
    ``q256``: a bound scored at a rate the cell does not cover is another
    cell's evidence, and ``cell_rungs`` is the covering list to check it
    against (``None`` when the caller has not resolved the rung axis, which
    leaves the check off rather than guessing a vocabulary).

    ``schema`` selects the member set: v6 is ``{grade, kl, smoke{status,
    receipt}}``; v7 adds ``smoke.attribution`` and ``smoke.control``; v8 adds
    ``artifact``; v9 adds ``smoke.record``. The schema decides what this
    reader reads: a field from a later grammar on an older table is accepted
    and never read, like any other additive field (#1548), so a v6 table that
    carries an attribution still reads as a v6 table with no attribution.
    """
    if not isinstance(payload, Mapping):
        raise LaneEligibilityError(f"{where} must be a JSON object")
    attributed = schema in ATTRIBUTED_SMOKE_LANE_SCHEMAS
    encoder_scoped = schema in ENCODER_SCOPED_LANE_SCHEMAS
    recorded_smoke = schema in RECORDED_SMOKE_LANE_SCHEMAS
    required = {"grade", "kl", "smoke"}
    if encoder_scoped:
        required.add("artifact")
    _require_keys(payload, where, required=required, optional=set())
    grade = payload["grade"]
    if grade not in EVIDENCE_GRADES:
        raise LaneEligibilityError(
            f"{where}.grade must be one of {sorted(EVIDENCE_GRADES)}, got {grade!r}")
    raw = payload["kl"]
    if not isinstance(raw, list) or any(not isinstance(e, Mapping) for e in raw):
        raise LaneEligibilityError(
            f"{where}.kl must be a JSON array of "
            "{kind, top_k, regime, execution_modes, receipt} objects")
    entries: list[CellKlEvidence] = []
    for i, entry in enumerate(raw):
        spot = f"{where}.kl[{i}]"
        _require_keys(entry, spot,
                      required={"kind", "top_k", "regime", "execution_modes",
                                "receipt"},
                      optional={"q256"})
        kind = entry["kind"]
        if kind not in EVIDENCE_KL_KINDS:
            raise LaneEligibilityError(
                f"{spot}.kind must be one of {sorted(EVIDENCE_KL_KINDS)}, got {kind!r}")
        top_k = entry["top_k"]
        if kind == EVIDENCE_KL_KIND_TOPK_LOWER_BOUND:
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
                raise LaneEligibilityError(
                    f"{spot}.top_k must be a positive integer for a top-K "
                    f"intersection bound, got {top_k!r}")
        elif top_k is not None:
            raise LaneEligibilityError(
                f"{spot}.top_k must be null for a full-vocabulary KL, got {top_k!r}")
        regime = entry["regime"]
        if cell_regime and regime != cell_regime:
            raise LaneEligibilityError(
                f"{spot}.regime {regime!r} is not the cell's regime "
                f"{cell_regime!r}: a bound scored in another regime is another "
                "cell's evidence, and reading it here is how a prefill number "
                "came to stand in for decode quality")
        modes = entry["execution_modes"]
        if (not isinstance(modes, list) or not modes
                or any(not isinstance(m, str) or m not in TESSERA_EXECUTION_MODES
                       for m in modes)
                or len(set(modes)) != len(modes)):
            raise LaneEligibilityError(
                f"{spot}.execution_modes must be a non-empty list of distinct "
                f"values from {sorted(TESSERA_EXECUTION_MODES)}, got {modes!r}")
        outside = sorted(set(modes) - set(execution_modes)) if execution_modes else []
        if outside:
            raise LaneEligibilityError(
                f"{spot} claims execution_modes {outside} the cell does not "
                f"cover ({sorted(execution_modes)}); a KL under a mode the "
                "census never joined attests a runtime this cell does not scope")
        # v32 grammar: the rung the receipt was scored at.  Positive integer
        # when published, absent before -- and a rung this cell does not
        # COVER is another cell's evidence exactly as a mismatched regime is.
        q256 = entry.get("q256")
        if q256 is not None:
            if (not isinstance(q256, int) or isinstance(q256, bool)
                    or q256 <= 0):
                raise LaneEligibilityError(
                    f"{spot}.q256 must be a positive integer rung, got "
                    f"{q256!r}")
            covered = tuple(cell_rungs) if cell_rungs is not None else ()
            if covered and q256 not in covered:
                raise LaneEligibilityError(
                    f"{spot}.q256 {q256} is not a rung this cell covers "
                    f"({list(covered)}); a bound scored at another rate is "
                    "another cell's evidence")
        entries.append(CellKlEvidence(
            kind=kind, top_k=top_k, regime=str(regime),
            execution_modes=tuple(modes),
            receipt=_require_receipt(entry["receipt"], spot), q256=q256))
    keys = [(e.kind, e.top_k, e.regime, e.execution_modes, e.receipt, e.q256)
            for e in entries]
    if len(set(keys)) != len(keys):
        raise LaneEligibilityError(
            f"{where}.kl repeats an entry; the field is a set of receipts")

    smoke = payload["smoke"]
    if not isinstance(smoke, Mapping):
        raise LaneEligibilityError(f"{where}.smoke must be a JSON object")
    smoke_keys = {"status", "receipt"}
    if attributed:
        smoke_keys |= {"attribution", "control"}
    if recorded_smoke:
        smoke_keys.add("record")
    _require_keys(smoke, f"{where}.smoke", required=smoke_keys, optional=set())
    status = smoke["status"]
    if status not in EVIDENCE_SMOKE_STATUSES:
        raise LaneEligibilityError(
            f"{where}.smoke.status must be one of "
            f"{sorted(EVIDENCE_SMOKE_STATUSES)}, got {status!r}")
    control: SmokeControl | None = None
    attribution = ""
    record: SmokeRecord | None = None
    status_is_derived = False
    if attributed:
        control = _parse_smoke_control(smoke["control"], f"{where}.smoke.control")
    if recorded_smoke:
        record = _parse_smoke_record(
            smoke["record"], f"{where}.smoke.record", control=smoke["control"])
        # "Is this cell's word derived, or asserted?" is a state Tessera
        # NAMES, so it is asked rather than inferred from the key: a reader
        # that spelled it `record is not None` would be restating the rule
        # one level up from the one it already refuses to restate.
        status_is_derived = bool(_tessera_published(
            "smoke_status_is_derived", f"{where}.smoke")(dict(smoke)))
        if status_is_derived:
            derived_status = _tessera_published(
                "derive_smoke_status", f"{where}.smoke")(dict(smoke))
            if status != derived_status:
                raise LaneEligibilityError(
                    f"{where}.smoke.status is {status!r} but Tessera's own "
                    f"derive_smoke_status derives {derived_status!r} from the "
                    "record beside it; the status is read off the record, "
                    "never asserted beside it. This repository does not "
                    "re-implement the rule -- restating it here is the second "
                    "home RobTand/tessera#327 was filed about -- so a "
                    "disagreement is a contract defect and is refused rather "
                    "than resolved.")
    if status == EVIDENCE_SMOKE_NOT_RECORDED:
        if smoke["receipt"] is not None:
            raise LaneEligibilityError(
                f"{where}.smoke: status not_recorded names a receipt "
                f"{smoke['receipt']!r}; a receipt is where a recorded smoke "
                "lives, so one here says the status is wrong")
        if control is not None:
            raise LaneEligibilityError(
                f"{where}.smoke: status not_recorded names a control "
                f"{control.as_dict()!r}; no completion came back, so there is "
                "nothing for a reference to have matched")
        receipt = ""
    else:
        if smoke["receipt"] is None:
            raise LaneEligibilityError(
                f"{where}.smoke: status {status!r} names no receipt; a smoke "
                "nobody recorded is not_recorded")
        receipt = _require_receipt(smoke["receipt"], f"{where}.smoke")
    if attributed:
        attribution = smoke["attribution"]
        if attribution not in EVIDENCE_SMOKE_ATTRIBUTIONS:
            raise LaneEligibilityError(
                f"{where}.smoke.attribution must be one of "
                f"{sorted(EVIDENCE_SMOKE_ATTRIBUTIONS)}, got {attribution!r}")
        if recorded_smoke:
            # v9 derives the attribution from the RECORD, which is a
            # projection of rows this repository does not restate.
            derived_attribution = _tessera_published(
                "derive_smoke_attribution", f"{where}.smoke")(dict(smoke))
            if attribution != derived_attribution:
                raise LaneEligibilityError(
                    f"{where}.smoke.attribution is {attribution!r} but "
                    "Tessera's own derive_smoke_attribution derives "
                    f"{derived_attribution!r} from the record beside it; the "
                    "attribution is read off the record, never asserted "
                    "beside it")
        else:
            derived_attribution = derive_smoke_attribution(control)
            if attribution != derived_attribution:
                raise LaneEligibilityError(
                    f"{where}.smoke.attribution is {attribution!r} but its "
                    f"control derives {derived_attribution!r}; the attribution "
                    "is read off the control, never asserted beside it")
    artifact: EvidenceArtifact | None = None
    if encoder_scoped:
        artifact = _parse_evidence_artifact(payload["artifact"], f"{where}.artifact")

    derived = derive_evidence_grade(entries)
    if grade != derived:
        raise LaneEligibilityError(
            f"{where}.grade is {grade!r} but its kl entries derive {derived!r}; "
            "the grade is read off the entries, never asserted beside them")
    return CellEvidence(grade=grade, kl=tuple(entries), smoke_status=str(status),
                        smoke_receipt=receipt, smoke_attribution=str(attribution),
                        smoke_control=control, artifact=artifact,
                        smoke_record=record,
                        smoke_status_is_derived=status_is_derived)


def cell_evidence_admits(cell: Any) -> tuple[bool, str]:
    """Whether a cell's own published evidence lets this producer use it.

    ONE predicate, read by both admission legs -- the development menu
    (``tessera_render.tessera_attesting_cells``) and the per-artifact export
    gate (``resolve_unit_route``) -- because a rung the menu offers and the
    export refuses, or the reverse, is the split-brain principle 8 exists to
    stop.

    What it refuses is a MEASURED serving defect, not a structure: a cell
    whose greedy smoke the runtime recorded as degenerate
    (:data:`EVIDENCE_SMOKE_REFUSALS`) fails principle 9's "generates correctly"
    leg, and it fails it in a structured field rather than in prose. Nothing
    here mentions ``routed_moe``: a hardcoded structure ban would be principle
    1's vetoed band-aid, and a cell this refuses is refused for what was
    measured on it, not for what it is. The answer therefore tracks whatever
    status the PINNED table publishes and nothing else -- the two routed-MoE
    cells were refused from contract v17 through v20 on ``repetitive`` and are
    not refused at v21, which publishes ``recorded``, with no edit here either
    time. Whether that ``recorded`` is CHECKABLE is a different question and
    not this predicate's: RobTand/tessera#327 found that v21's rule lived only
    in a dated measurements file and was satisfiable by an empty completion,
    which lane schema v9 answers by putting the rule and its rows in
    ``smoke.record`` -- re-derived at parse through Tessera's own
    ``derive_smoke_status`` (:func:`_parse_smoke_record`), so a status this
    predicate reads is one a reader could check. Whether routed-MoE Tessera is
    PROMOTED past the menu remains a human's call (prismaquant #198).

    What it deliberately does NOT refuse is a low GRADE. Every cell in the
    installed table is ``route_only`` or ``kl_lower_bound`` -- the publisher's
    changelog records that every served KL in that repository is a top-K
    intersection lower bound -- so a grade gate would refuse the whole lane,
    including rungs this producer already ships. Raising the grade bar is a
    promotion-ladder move and belongs to a human; the grade travels into
    provenance so a shipcard says which grade attested each unit.

    What it deliberately does NOT decide on is the v7 ATTRIBUTION.
    Tessera's contract v18 changelog states the consumer rule it expects --
    "a gate that refused on status alone now refuses on status 'repetitive'
    AND attribution other than 'shared_with_reference'" -- and on the v20
    table that rule admitted both routed-MoE cells, whose control showed the
    BF16 source returning the same degenerate completion. This reader refuses
    on the status: ``shared_with_reference`` removes the evidence AGAINST the
    route without adding any FOR it, and admitting a structure this producer
    has never shipped on that basis is a promotion, which is a human's call.
    The refusal names the control so the reviewer sees what was read and not
    decided on; prismaquant #198 holds the decision. v21 retired the control
    from those cells, so the branch is exercised on the v20 shape in
    ``tests/test_tessera_lane_v8.py`` rather than on an installed cell.

    A pre-v6 cell carries no evidence block and is admitted unchanged: the
    grammar that never published the field cannot be read as publishing a
    failure.
    """
    evidence = getattr(cell, "evidence", None)
    if evidence is None:
        return True, ""
    if evidence.smoke_status in EVIDENCE_SMOKE_REFUSALS:
        control = evidence.smoke_control
        if control is not None:
            attributed = (
                f" Its control (reference {control.reference!r}, outcome "
                f"{control.outcome!r}, receipt {control.receipt!r}) derives "
                f"attribution {evidence.smoke_attribution!r}; this producer "
                "reads that and still refuses on the status alone -- an "
                "attribution that the reference shares the symptom is not a "
                "record of this route generating correctly. Admitting on it is "
                "a promotion held for a human in prismaquant #198.")
        elif evidence.smoke_attribution:
            attributed = (
                f" Its attribution is {evidence.smoke_attribution!r}: nobody "
                "ran the reference, so the status is an observation and not "
                "an attribution.")
        else:
            attributed = ""
        return False, (
            f"cell {getattr(cell, 'id', '?')!r} publishes "
            f"evidence.smoke.status={evidence.smoke_status!r} "
            f"(receipt {evidence.smoke_receipt!r}): the runtime recorded a "
            "greedy smoke on this route and the generation degenerated. "
            "Principle 9 requires a route to generate correctly, so this cell "
            "attests a measured serving defect and is not admitted."
            + attributed
        )
    return True, ""


# ---------------------------------------------------------------------------
# The lane predicate (contract v20)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LaneClaim:
    """One ``native_extensions[].lane`` block: what a kernel reads.

    ``requires`` is ``None`` when the lane publishes no predicate of its own
    -- its eligibility is then the route's, already published in ``formats``
    -- and otherwise a closed mapping in the contract's own vocabulary
    (:data:`LANE_REQUIREMENT_FIELDS`), lists kept as tuples. The distinction
    between ``None`` and an empty block is the publisher's: an empty
    ``requires`` is refused at parse, exactly as Tessera's validator refuses
    it.
    """

    #: The ``module_name_prefix`` of the row this lane belongs to.
    extension: str
    #: The decoder name the cell's ``executes[].decoder`` spells when it
    #: launches through this extension.
    decoder: str
    requires: Mapping[str, Any] | None = None

    def answer(self) -> dict[str, Any]:
        """The gate-read projection, JSON-shaped, for the reviewed answer."""
        requires = None
        if self.requires is not None:
            requires = {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in self.requires.items()
            }
        return {"decoder": self.decoder, "requires": requires}


def parse_lane_claim(payload: Any, where: str, *, extension: str) -> LaneClaim:
    """Read one ``lane`` block at Tessera's vocabulary, or refuse by name.

    Mirrors the publisher's own validator (``tessera.serving.contract``,
    ``_validate_lane``): required ``decoder``, optional non-empty
    ``requires`` whose every key is a requirement this reader has learned;
    the three list requirements non-empty, ascending, unique, positive; the
    three carry requirements JSON booleans; ``rotation`` a non-empty unique
    list of known states; ``body`` and ``plane`` known spellings. A block this
    reader cannot read is a refusal of the whole table -- a lane whose
    predicate is unreadable is a lane no gate can decide, and absent evidence
    is not a pass.

    The block's own fields are read tolerantly (#1548): an added field beside
    ``decoder`` and ``requires`` is accepted unless the producer marks it
    ``must_understand``. The keys of ``requires`` are not fields. Each one is
    a condition a unit's wire must satisfy, so an unknown requirement stays a
    refusal: skipping it would admit a unit the loader refuses.
    """
    if not isinstance(payload, Mapping):
        raise LaneEligibilityError(
            f"{where} ({extension}): lane must be a JSON object naming the "
            "decoder the extension serves")
    _require_keys(payload, f"{where} ({extension})", required={"decoder"},
                  optional=LANE_FIELDS - {"decoder"})
    decoder = payload["decoder"]
    if not isinstance(decoder, str) or not decoder:
        raise LaneEligibilityError(
            f"{where}.decoder ({extension}) must be a non-empty string")
    if "requires" not in payload:
        return LaneClaim(extension=extension, decoder=decoder)
    requires = payload["requires"]
    if not isinstance(requires, Mapping):
        raise LaneEligibilityError(
            f"{where}.requires ({extension}) must be a JSON object keyed by "
            f"requirement name; the known names are {list(LANE_REQUIREMENT_FIELDS)}")
    if not requires:
        raise LaneEligibilityError(
            f"{where}.requires ({extension}) is empty. A lane with no predicate "
            "omits the block; an empty one is a claim this reader cannot tell "
            "from 'reads everything' and is refused")
    unknown = sorted(set(requires) - set(LANE_REQUIREMENT_FIELDS))
    if unknown:
        raise LaneEligibilityError(
            f"{where}.requires ({extension}) publishes requirement(s) {unknown} "
            f"this reader cannot decide (it reads {list(LANE_REQUIREMENT_FIELDS)}). "
            "A checker that skipped a published condition would select a unit "
            "the loader refuses, so the table is refused instead: teach "
            "lane_eligibility.LANE_REQUIREMENT_FIELDS the name once Tessera's "
            "scheme.decide_lane_requirements decides it.")
    parsed: dict[str, Any] = {}
    for name in LANE_REQUIREMENT_FIELDS:
        if name not in requires:
            continue
        at = f"{where}.requires.{name} ({extension})"
        value = requires[name]
        if name in LANE_REQUIREMENT_LISTS:
            parsed[name] = _parse_lane_int_list(value, at)
        elif name in LANE_REQUIREMENT_CARRIES:
            if not isinstance(value, bool):
                raise LaneEligibilityError(
                    f"{at} must be a JSON boolean (does the lane read units that "
                    f"carry this?), got {value!r}")
            parsed[name] = value
        elif name == "rotation":
            if (not isinstance(value, list) or not value
                    or len(set(value)) != len(value)):
                raise LaneEligibilityError(
                    f"{at} must be a non-empty list of distinct rotation states "
                    f"{sorted(LANE_ROTATION_STATES)}, got {value!r}")
            bad = sorted(str(v) for v in value if v not in LANE_ROTATION_STATES)
            if bad:
                raise LaneEligibilityError(
                    f"{at} names rotation state(s) {bad} this reader does not "
                    f"know; the known states are {sorted(LANE_ROTATION_STATES)}")
            parsed[name] = tuple(str(v) for v in value)
        elif name == "body":
            if value not in LANE_BODIES:
                raise LaneEligibilityError(
                    f"{at} must be one of {sorted(LANE_BODIES)}, got {value!r}")
            parsed[name] = str(value)
        else:  # plane
            if value not in LANE_PLANES:
                raise LaneEligibilityError(
                    f"{at} must be one of {sorted(LANE_PLANES)}, got {value!r}")
            parsed[name] = str(value)
    routed = parsed.get("column_rates_routed_moe")
    if routed is not None:
        # Tessera's contract validator holds the routed set to a subset of
        # ``column_rates``: a routed launch cannot reach a rate the lane does
        # not read at all.  Mirror it so a table the loader refuses is refused
        # here at parse time, not decided against.
        base = parsed.get("column_rates")
        if base is None or not set(routed) <= set(base):
            raise LaneEligibilityError(
                f"{where}.requires ({extension}) publishes column_rates_routed_moe "
                f"{list(routed)} that is not a subset of its column_rates "
                f"{None if base is None else list(base)}; a routed-expert launch "
                "cannot reach a rate the lane does not read")
    return LaneClaim(extension=extension, decoder=decoder, requires=parsed)


def _parse_lane_int_list(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise LaneEligibilityError(
            f"{where} must be a non-empty list of positive integers, got {value!r}")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in value):
        raise LaneEligibilityError(
            f"{where} must name positive integers only, got {value!r}")
    if list(value) != sorted(set(value)):
        raise LaneEligibilityError(
            f"{where} must be ascending and unique, got {value!r}")
    return tuple(int(v) for v in value)


def parse_lane_claims(native_extensions: Any, where: str) -> tuple[LaneClaim, ...]:
    """Every ``native_extensions[].lane`` block, in table order.

    Only the two fields a lane gate reads are taken from each row -- the
    prefix that names the row and its ``lane`` -- so the rest of the row's
    grammar stays with ``tessera_runtime_contract._parse_native_extensions``,
    which refuses the table on the fingerprint's behalf. A row without a
    ``lane`` is refused here: since contract v20 every row publishes one,
    and a launch whose lane is unstated cannot be decided.
    """
    if (not isinstance(native_extensions, Sequence)
            or isinstance(native_extensions, (str, bytes))):
        raise LaneEligibilityError(f"{where} must be a JSON array")
    claims: list[LaneClaim] = []
    seen: set[str] = set()
    for i, row in enumerate(native_extensions):
        at = f"{where}[{i}]"
        if not isinstance(row, Mapping):
            raise LaneEligibilityError(f"{at} must be a JSON object")
        prefix = row.get("module_name_prefix")
        if not isinstance(prefix, str) or not prefix:
            raise LaneEligibilityError(
                f"{at}.module_name_prefix must be a non-empty string")
        if prefix in seen:
            raise LaneEligibilityError(f"{at}.module_name_prefix {prefix!r} is declared twice")
        seen.add(prefix)
        if "lane" not in row:
            raise LaneEligibilityError(
                f"{at} ({prefix}) publishes no 'lane': which decoder this "
                "extension serves, and what a unit's wire must be for it to "
                "read it, is unstated, so no launch through it can be decided")
        claims.append(parse_lane_claim(row["lane"], f"{at}.lane", extension=prefix))
    return tuple(claims)


def lane_claim_for_cell(cell: Any, lanes: Sequence[LaneClaim]) -> LaneClaim | None:
    """The first lane-bearing claim among this cell's launches, or ``None``.

    This answers only whether a cell is lane-gated at all. Since Tessera
    contract v42 a cell can name several launches, some through a lane and
    some beside it, so the first claim is not the decision: whether a rung is
    admitted, and through which launches, is :func:`cell_rung_launches`
    (PQ #1274). A cell is lane-gated exactly when one of the decoders it
    EXECUTES is the decoder a lane serves and that lane publishes a predicate. A decoder no
    lane names (``torch_window``, ``torch_materialize_stock``) is the route's
    own path, gated by the cell's route status and evidence alone -- the
    contract publishes no wire predicate for it, and inventing one here would
    be the second copy this module exists to refuse.
    """
    decoders = {decoder for _symbol, decoder in getattr(cell, "executes", ())}
    for claim in lanes:
        if claim.requires is not None and claim.decoder in decoders:
            return claim
    return None


def lane_claims_for_cell(cell: Any, lanes: Sequence[LaneClaim]
                         ) -> tuple[LaneClaim, ...]:
    """Every lane whose predicate governs one of this cell's launches.

    :func:`lane_claim_for_cell` answers whether a cell is lane-gated at all;
    this answers through WHICH lanes, because since Tessera contract v42 a
    cell can launch through a lane and beside it (PQ #1274).
    """
    decoders = {decoder for _symbol, decoder in getattr(cell, "executes", ())}
    return tuple(claim for claim in lanes
                 if claim.requires is not None and claim.decoder in decoders)


class _ExecutesView:
    """A cell narrowed to the launches one rung keeps, for lane claims."""

    __slots__ = ("_cell", "executes")

    def __init__(self, cell: Any, executes: tuple[tuple[str, str], ...]) -> None:
        self._cell = cell
        self.executes = executes

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cell, name)


def _launch_covers(cell: Any, index: int, rate_q256: int) -> bool:
    """Whether the launch at ``executes[index]`` covers ``rate_q256``."""
    covers = getattr(cell, "launch_covers_rate", None)
    if callable(covers):
        return bool(covers(index, rate_q256))
    scopes = tuple(getattr(cell, "launch_rungs_q256", ()) or ())
    scope = scopes[index] if index < len(scopes) else None
    if scope is None:
        covers_rate = getattr(cell, "covers_rate", None)
        return bool(covers_rate(rate_q256)) if callable(covers_rate) else True
    return rate_q256 in scope


def cell_rung_launches(cell: Any, rate_q256: int | None, lanes: Sequence[LaneClaim]
                       ) -> tuple[bool, str, tuple[tuple[str, str], ...]]:
    """The launches a cell makes for THIS producer's plan at one rung.

    Returns ``(admits, reason, launches)``.  A cell's ``executes`` is the
    UNION of the launches its runtime makes over every rung it lists: Tessera
    derives it that way (``contract._validate_cell_executes`` narrows
    ``scheme.route_launches`` by the lanes each rung reaches), and a launch
    through a lane happens only at a rung that lane's published predicate
    admits. Under lane schema v12 a launch with its own ``rungs_q256`` scope
    joins only at a rung that scope covers; a launch without one keeps the
    scope of its cell. So at one rung the cell launches through every
    in-scope lane-free decoder it names and through each in-scope lane whose
    predicate this producer's planned wire satisfies; a launch through a lane
    that refuses the plan is not made there. The cell admits the unit when
    at least one launch is left, and ``launches`` is that set -- what the
    serve runs for these bytes, which a route record stamps instead of the
    union.

    Until contract v41 every lane-gated cell launched ONLY through its lane,
    so a refusing lane left nothing and the cell refused; that case still
    refuses, with the same reason.  Contract v42 is the first to name a lane
    launch beside a lane-free one in one cell: the four window routed cells
    carry the fused routed pair beside the compact adapter, whose dispatch
    keeps the compact pair for every stack the fused lane refuses (through
    v44, the mixed-rate q256=896 rung).  Refusing those cells whole shrank
    routed E4M3 admission to q256 1024 (PQ #1274).  Since v45 the fused lane
    reads every rung those cells list; the compact-only case on the pinned
    table is a routed plan outside ``column_rates_routed_moe``, which no
    listed rung makes.

    The rule is not here. Tessera publishes the predicate
    (``native_extensions[].lane.requires``) and owns the decision
    (``tessera.serving.scheme.decide_lane_requirements``, the one home its
    loader, its byte-side report and its plan-time gate all call); this
    function supplies the FACTS -- ``tessera_render.planned_wire_facts``, the
    wire this producer will encode for the cell's family at this rung, read
    off the same recipe and decoration the render encodes with -- and turns
    the refusals into a reason that names the cell, the lane and the launch.
    Both imports are lazy: ``planned_wire_facts`` needs the encoder, and
    ``scheme`` pulls in ``tessera.serving.contract`` for its wire spellings
    (the module, not its validator's dispatch tables; neither imports torch or
    vLLM). A contract is LOADED without either -- this runs at admission.

    The unit's STRUCTURE (``dense`` | ``routed_moe``) is one of those facts,
    read off the cell (a serving-profile / decision-unit fact, never inferred
    from bytes) and stated to ``planned_wire_facts``. The structure-scoped
    ``column_rates_routed_moe`` requirement is decided by Tessera's core over
    it: a ``routed_moe`` unit outside the set is refused, naming the compact
    adapter the unit keeps; a dense unit ignores the field; a unit with no
    structure fact is refused by name, since absent evidence is not a pass.

    Fail-closed at every edge: a requirement the decision core has not
    learned RAISES (its own rule, re-raised with the lane named) rather than
    being skipped; a family this producer cannot plan, or a cell asked
    without a rung, is refused with the reason, never passed.
    """
    executes = tuple(getattr(cell, "executes", ()))
    scopes = tuple(getattr(cell, "launch_rungs_q256", ()) or ())
    if scopes and len(scopes) != len(executes):
        raise LaneEligibilityError(
            f"cell {getattr(cell, 'id', getattr(cell, 'cell_id', '?'))!r} names "
            f"{len(executes)} launches and {len(scopes)} launch scopes; the "
            "two move together or the scope is unreadable")
    if rate_q256 is not None and scopes:
        executes = tuple(
            pair for index, pair in enumerate(executes)
            if _launch_covers(cell, index, int(rate_q256)))
        if not executes:
            scoped_id = getattr(cell, "id", getattr(cell, "cell_id", "?"))
            return False, (
                f"cell {scoped_id!r} names no launch covering rung {rate_q256}; "
                "its launches each cover only the census rungs their "
                "rungs_q256 scope names"), ()
    claims = lane_claims_for_cell(_ExecutesView(cell, executes), lanes)
    if not claims:
        return True, "", executes
    cell_id = getattr(cell, "id", getattr(cell, "cell_id", "?"))
    family = str(getattr(cell, "family", ""))

    def head(claim: LaneClaim) -> str:
        launches = sorted(symbol for symbol, decoder in executes
                          if decoder == claim.decoder)
        return (
            f"cell {cell_id!r} launches {launches} through the "
            f"{claim.extension!r} lane (decoder {claim.decoder!r}), whose "
            "published predicate this producer's planned wire")

    if rate_q256 is None:
        return False, (
            f"{head(claims[0])} cannot be decided against: the unit's rung was "
            "not read, and the lane reads a rate set that depends on it"), ()
    from . import tessera_render
    from .tessera_formats import TesseraFormatError

    try:
        facts = tessera_render.planned_wire_facts(
            family, int(rate_q256), structure=getattr(cell, "structure", None))
    except TesseraFormatError as exc:
        return False, (
            f"{head(claims[0])} cannot be decided against: this producer "
            f"cannot plan family {family!r} at rung {rate_q256} ({exc}), and a "
            "plan that does not exist is not a plan the lane reads"), ()
    from tessera.serving.scheme import decide_lane_requirements

    refused: list[str] = []
    refused_decoders: set[str] = set()
    for claim in claims:
        try:
            refusals = decide_lane_requirements(
                claim.extension, dict(claim.requires), facts)
        except ValueError as exc:
            raise LaneEligibilityError(
                f"{head(claim)} cannot be decided against: the lane publishes a "
                f"requirement Tessera's own decision core does not decide -- {exc}"
            ) from exc
        if refusals:
            refused_decoders.add(claim.decoder)
            refused.append(f"{head(claim)} for {family} R{rate_q256} fails: "
                           + "; ".join(refusals))
    launches = tuple(pair for pair in executes if pair[1] not in refused_decoders)
    if launches:
        return True, "", launches
    return False, (
        ". ".join(refused)
        + ". No launch the cell names is left at this rung, so the kernel "
        "would refuse these bytes at load and the route is not admitted; the "
        "predicate is Tessera's, read from the contract, and the plan is this "
        "producer's -- change the plan or re-pin, never this gate."
    ), ()


def cell_lane_admits(cell: Any, rate_q256: int | None, lanes: Sequence[LaneClaim]
                     ) -> tuple[bool, str]:
    """Whether a lane a cell launches through leaves it a launch for this plan.

    ONE predicate for every admission leg (the menu's
    ``tessera_render.tessera_attesting_cells``, the development contract's
    ``TesseraContract.native_cells``, the export gate's
    :func:`resolve_unit_route`), beside :func:`cell_evidence_admits` and for
    the same reason: a rung the menu offers and the export refuses is the
    split-brain principle 8 exists to stop.  :func:`cell_rung_launches` holds
    the rule and the reason; this is its verdict.
    """
    admits, why, _launches = cell_rung_launches(cell, rate_q256, lanes)
    return admits, why


def _allowable_rung_tables(entry: Mapping[str, Any], where: str) -> dict[int, tuple[int, ...]]:
    """Evaluate the publisher's v11 window_rate_set; never infer a new rule."""
    rule = entry.get("allowable_rungs")
    if rule is None:
        return {}
    at = f"{where}.allowable_rungs"
    keys = {"rule", "code_arity", "range_q256", "step_q256", "run_tables",
            "excluded_run_tables", "excluded_q256", "wire", "evidence"}
    _require_keys(rule, at, required=keys, optional=set())
    if rule["rule"] != "window_rate_set":
        raise LaneEligibilityError(f"{at}.rule: unknown allowable rung rule")
    # This is public producer grid metadata, not a serving import or vendor copy.
    from tessera.alphabet import SERIALISABLE_GRIDS
    grid = next((g for g in SERIALISABLE_GRIDS.values()
                 if g.name == str(entry["grid"])), None)
    if grid is None:
        raise LaneEligibilityError(f"{at}: unknown serialisable grid")
    arity = int(grid.arity)
    cap = int(entry["native_terminal_q256"]) * arity // 256
    if type(rule["code_arity"]) is not int or rule["code_arity"] != arity:
        raise LaneEligibilityError(f"{at}.code_arity differs from the producer grid")
    rng, step = rule["range_q256"], rule["step_q256"]
    if (not isinstance(rng, list) or len(rng) != 2
            or any(type(q) is not int for q in rng)
            or type(step) is not int or step < 1):
        raise LaneEligibilityError(f"{at}: invalid range_q256 or step_q256")
    low, high = entry["reader_rate_range_q256"]
    reader_step = entry["reader_rate_step_q256"]
    if (not low <= rng[0] <= rng[1] <= high or step % reader_step
            or any((q-low) % reader_step for q in rng)):
        raise LaneEligibilityError(f"{at}: rule is outside the reader grid")

    def tables(value: Any, label: str) -> tuple[tuple[int, ...], ...]:
        if not isinstance(value, list):
            raise LaneEligibilityError(f"{at}.{label} must be an array")
        out = []
        for table in value:
            if (not isinstance(table, list) or len(table) not in (1, 2)
                    or any(type(r) is not int for r in table)
                    or not 1 <= table[0] <= table[-1] <= cap
                    or (len(table) == 2 and table[1] != table[0]+1)):
                raise LaneEligibilityError(f"{at}.{label}: invalid run table")
            out.append(tuple(table))
        if out != sorted(set(out)):
            raise LaneEligibilityError(f"{at}.{label} must be ascending and distinct")
        return tuple(out)

    allowed = tables(rule["run_tables"], "run_tables")
    excluded = tables(rule["excluded_run_tables"], "excluded_run_tables")
    if set(allowed) & set(excluded):
        raise LaneEligibilityError(f"{at}: a run table is both allowed and excluded")
    ex_q = rule["excluded_q256"]
    if (not isinstance(ex_q, list) or any(type(q) is not int for q in ex_q)
            or ex_q != sorted(set(ex_q))
            or any(not rng[0] <= q <= rng[1] or (q-rng[0]) % step for q in ex_q)):
        raise LaneEligibilityError(f"{at}.excluded_q256 is outside the rule grid")
    reached = {}
    for q in range(rng[0], rng[1]+1, step):
        root, remainder = divmod(q*arity, 256)
        rates = (root, root+1) if remainder else (root,)
        if not 1 <= rates[0] <= rates[-1] <= cap:
            raise LaneEligibilityError(f"{at}: rung {q} exceeds the grid's rate cap")
        reached[q] = rates
    if set(allowed) - set(reached.values()):
        raise LaneEligibilityError(f"{at}: run table unreachable from the rule grid")
    wire_keys = {"body", "span", "plane", "window_bits", "seed", "sigma", "channel_sigma"}
    wire = rule["wire"]
    _require_keys(wire, f"{at}.wire", required=wire_keys, optional=set())
    expected_wire = ("tcq", 2, "lut16") if arity == 2 else ("window", 1, "channel")
    if ((wire["body"], wire["span"], wire["plane"]) != expected_wire
            or not isinstance(wire["window_bits"], int)
            or not isinstance(wire["seed"], int)
            or any(v is not None and not isinstance(v, (int, float))
                   for v in (wire["sigma"], wire["channel_sigma"]))):
        raise LaneEligibilityError(f"{at}.wire: invalid window wire stamp")
    if not isinstance(rule["evidence"], list) or not rule["evidence"]:
        raise LaneEligibilityError(f"{at}.evidence must name repository receipts")
    for path in rule["evidence"]:
        if (not isinstance(path, str) or not path or path.startswith("/")
                or ".." in path.split("/") or "\\" in path):
            raise LaneEligibilityError(f"{at}.evidence: invalid repository path")
    accepted = {q: rates for q, rates in reached.items()
                if q not in ex_q and rates in allowed and rates not in excluded}
    for stamp in entry["attested_wire"]:
        if stamp["q256"] in accepted and any(stamp[k] != wire[k] for k in wire_keys):
            raise LaneEligibilityError(f"{at}.wire differs from a covered census stamp")
    return accepted


def _cell_rule_coverage(payload: Mapping[str, Any],
                        allowable: Mapping[int, tuple[int, ...]], where: str
                        ) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    if not isinstance(payload, Mapping):
        raise LaneEligibilityError(f"{where} must be a JSON object")
    census = _parse_rungs(payload.get("rungs_q256"), f"{where}.rungs_q256")
    want = sorted({allowable[q] for q in census if q in allowable})
    got = payload.get("run_tables", [])
    if got != [list(t) for t in want]:
        raise LaneEligibilityError(f"{where}.run_tables differs from its allowable census rungs")
    covered = tuple(q for q, rates in allowable.items() if rates in want)
    return tuple(want), covered


def _launch_rule_coverage(
    scopes: tuple[tuple[int, ...] | None, ...],
    allowable: Mapping[int, tuple[int, ...]] | None,
    covered: tuple[int, ...],
) -> tuple[tuple[int, ...] | None, ...]:
    """Per-launch derived rule coverage, in ``executes`` order.

    Each scoped launch covers the allowable run-table rungs its census rungs
    derive: a rung the cell's rule covers whose run table is one of the
    launch's census tables. An unscoped launch carries ``None`` and keeps
    the cell's own coverage. Without a rule (no ``allowable`` map) every
    scoped launch carries ``()``: census rungs alone decide.
    """
    if allowable is None:
        return tuple(None if scope is None else () for scope in scopes)
    covered_tables = {q: allowable[q] for q in covered if q in allowable}
    out: list[tuple[int, ...] | None] = []
    for scope in scopes:
        if scope is None:
            out.append(None)
            continue
        want = {allowable[q] for q in scope if q in allowable}
        out.append(tuple(q for q in covered
                         if covered_tables.get(q) in want and q not in scope))
    return tuple(out)


@dataclass(frozen=True)
class EligibilityCell:
    """One packaged cell: bytes, platform, regime, residency, runtime and launch.

    A cell is scoped to exactly one ``(platform, family, structure, regime)``
    and covers an explicit, non-empty rung list. It carries no prose: a
    validator refuses ``detail``/``rationale`` keys on a cell, because a gate
    cannot read prose (principle 14). Legacy v3 alone permits an absent plugin
    key. v4 requires Tessera, launch declarations and residency flags; v5
    additionally scopes every cell to an exact image and execution-mode set.
    """

    id: str
    platform: str
    family: str
    structure: str
    regime: str
    route_status: str
    qualification: str
    #: CB codebook rungs. Empty for a rate-addressed cell.
    rungs: tuple[int, ...] = ()
    #: Body bits per 256 weights. Empty for a CB cell.
    rungs_q256: tuple[int, ...] = ()
    #: The activation contract this route executes. Rate-addressed cells only;
    #: a CB cell publishes none and this stays "".
    activation_contract: str = ""
    requires_serve_flags: tuple[str, ...] = ()
    #: The vLLM plugin whose installation this route requires, or "" when the
    #: route is reachable in the pinned runtime as shipped. It is a
    #: machine-readable CELL field rather than prose because an export gate
    #: has to be able to refuse an artifact whose serve command would not
    #: install the plugin -- stock vLLM has no reader for Tessera bytes, so
    #: those routes are plugin-gated, not merely flag-gated. Retired-lane cells
    #: publish none and this stays "".
    requires_plugin: str = ""
    predicates: tuple[tuple[str, str, Any], ...] = ()
    #: "This cell's family is addressed by a RATE, not by a codebook size."
    #: The name is historical -- ``tcq_trellis`` was the only such kind when it
    #: was chosen -- and ``tessera_wire`` families set it too.
    is_trellis: bool = False
    #: v4's published launches, retained as pairs rather than inferred from IDs.
    executes: tuple[tuple[str, str], ...] = ()
    #: v12's per-launch census scope, in ``executes`` order. Each entry is the
    #: launch's ``rungs_q256`` tuple, or ``None`` when the launch names none
    #: and keeps the scope of its cell. An empty tuple never appears: the
    #: parser refuses an empty list, so ``()`` cannot mean "no scope".
    launch_rungs_q256: tuple[tuple[int, ...] | None, ...] = ()
    #: v12's per-launch derived rule coverage, in ``executes`` order. Each
    #: entry holds the allowable run-table rungs the launch's census rungs
    #: derive, or ``None`` when the launch names no scope. Evaluated once at
    #: parse beside the cell's own coverage, from the same rule.
    launch_covered_rungs_q256: tuple[tuple[int, ...] | None, ...] = ()
    residency_modes: tuple[str, ...] = ()
    runtime_image: str = ""
    execution_modes: tuple[str, ...] = ()
    #: v6's per-cell runtime versions. The image alone stopped identifying the
    #: build the day a cell was measured on a dev wheel inside a pinned image,
    #: which is why ``versions.attested_on`` -- one global claim for every
    #: cell -- was withdrawn from the contract. Empty for pre-v6 grammars.
    runtime_vllm: str = ""
    runtime_torch: str = ""
    #: v6's required ``evidence`` block. ``None`` for pre-v6 grammars, which
    #: published no such field; see :func:`cell_evidence_admits`.
    evidence: CellEvidence | None = None
    #: The Tessera code the cell's evidence was taken on (Tessera contract
    #: v41, optional, both or neither): the 40-hex commit for a person, and
    #: the ``tessera.package_source.v1`` digest a v3 pin compares
    #: (:func:`cell_serving_code_admits`). Empty when the cell names no code.
    runtime_tessera_commit: str = ""
    runtime_serving_source_sha256: str = ""
    #: V11 preserves census rungs separately from derived rule coverage.
    covered_rungs_q256: tuple[int, ...] = ()
    run_tables: tuple[tuple[int, ...], ...] | None = None
    runtime_kernel_build: str = ""

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        where: str,
        *,
        trellis_families: frozenset[str],
        schema: str = LANE_ELIGIBILITY_SCHEMA_TESSERA_LEGACY_V3,
        residency_modes: Sequence[str] = (),
        covered_rungs_q256: tuple[int, ...] = (),
        run_tables: tuple[tuple[int, ...], ...] | None = None,
        allowable: Mapping[int, tuple[int, ...]] | None = None,
    ) -> "EligibilityCell":
        if not isinstance(payload, Mapping):
            raise LaneEligibilityError(f"{where} must be a JSON object")
        if schema not in LANE_ELIGIBILITY_SCHEMAS:
            raise LaneEligibilityError(f"{where}: unsupported lane schema {schema!r}")
        family = str(payload.get("family", ""))
        if not family:
            raise LaneEligibilityError(
                f"{where}: cell must name a payload family")
        # The rung vocabulary follows the FAMILY's kind, exactly as the publisher's
        # own validator dispatches it. A cell carries no ``kind`` key.
        is_trellis = family in trellis_families
        rung_key = "rungs_q256" if is_trellis else "rungs"
        required = {
            "id", "platform", "family", "structure", "regime", rung_key,
            "route_status", "qualification", "requires_serve_flags",
            "predicates",
        }
        if is_trellis:
            required.add("activation_contract")
        is_v4 = schema in _LAUNCH_SCHEMAS
        is_scoped = schema in SCOPED_LANE_SCHEMAS
        has_evidence = schema in EVIDENCE_LANE_SCHEMAS
        if is_v4:
            required.update({"requires_plugin", "executes"})
        if is_scoped:
            required.add("runtime")
        if has_evidence:
            required.add("evidence")
        _require_keys(payload, where, required=required,
                      optional=({"run_tables"} if schema in _RULE_COVERAGE_SCHEMAS
                                else set()) if is_v4 else {"requires_plugin"})

        status = str(payload["route_status"])
        if status not in CELL_ROUTE_STATUSES:
            raise LaneEligibilityError(
                f"{where}.route_status must be one of "
                f"{sorted(CELL_ROUTE_STATUSES)}, got {status!r}. The runtime "
                "does not enumerate what it refuses; absence, not an "
                f"{ROUTE_STATUS_UNBACKED!r} cell, is how a lane table says no.")
        qualification = str(payload["qualification"])
        if qualification not in CELL_QUALIFICATIONS:
            raise LaneEligibilityError(
                f"{where}.qualification must be one of "
                f"{sorted(CELL_QUALIFICATIONS)}, got {qualification!r}")
        structure = str(payload["structure"])
        if structure not in STRUCTURES:
            raise LaneEligibilityError(
                f"{where}.structure must be one of {sorted(STRUCTURES)}, "
                f"got {structure!r}")

        rungs = _parse_rungs(payload[rung_key], f"{where}.{rung_key}")

        activation_contract = ""
        if is_trellis:
            activation_contract = str(payload["activation_contract"])
            if not activation_contract:
                raise LaneEligibilityError(
                    f"{where}.activation_contract must name the contract this "
                    "route executes; an empty one attests nothing")

        requires_plugin = str(payload.get("requires_plugin", ""))
        if is_v4 and payload["requires_plugin"] != "tessera":
            raise LaneEligibilityError(
                f"{where}.requires_plugin must be 'tessera'; stock vLLM "
                "has no reader for these bytes")
        if requires_plugin and status not in LANE_ROUTE_STATUSES:
            # Mirrors the ``requires_serve_flags`` rule below. A plugin
            # requirement is an instruction for reaching a route that EXISTS;
            # naming one on a cell whose route is an announced fallback says
            # nothing an operator can act on, and would let a reader believe a
            # plugin install turns a fallback into a native route.
            raise LaneEligibilityError(
                f"{where}: requires_plugin is {requires_plugin!r} but "
                f"route_status is {status!r}; a plugin requirement is only "
                f"meaningful on a cell whose route is one of "
                f"{sorted(LANE_ROUTE_STATUSES)}")
        executes: tuple[tuple[str, str], ...] = ()
        launch_scopes: tuple[tuple[int, ...] | None, ...] = ()
        cell_modes: tuple[str, ...] = ()
        if is_v4:
            executes, launch_scopes, cell_modes = parse_v4_cell_contract(
                payload, where, residency_modes=residency_modes, schema=schema,
                cell_rungs=rungs)
        runtime_image = ""
        execution_modes: tuple[str, ...] = ()
        runtime_vllm = ""
        runtime_torch = ""
        runtime_commit = runtime_digest = ""
        if is_scoped:
            runtime_image, execution_modes, runtime_vllm, runtime_torch = (
                parse_runtime_scope(payload["runtime"], where + ".runtime",
                                    require_versions=has_evidence))
            runtime_commit, runtime_digest = parse_runtime_code(
                payload["runtime"], where + ".runtime")
        evidence: CellEvidence | None = None
        if has_evidence:
            evidence = parse_cell_evidence(
                payload["evidence"], where + ".evidence",
                cell_regime=str(payload["regime"]),
                execution_modes=execution_modes,
                cell_rungs=rungs, schema=schema)
        flags = tuple(str(v) for v in payload["requires_serve_flags"])
        if flags and status != ROUTE_STATUS_BACKED_WITH_SERVE_FLAG:
            raise LaneEligibilityError(
                f"{where}: requires_serve_flags is non-empty but route_status "
                f"is {status!r}; a flag-gated route is "
                f"{ROUTE_STATUS_BACKED_WITH_SERVE_FLAG!r} by definition")
        if status == ROUTE_STATUS_BACKED_WITH_SERVE_FLAG and not flags:
            raise LaneEligibilityError(
                f"{where}.requires_serve_flags: route_status is "
                f"{ROUTE_STATUS_BACKED_WITH_SERVE_FLAG!r} but no serve flag is "
                "named; an operator cannot reach an unnamed flag")

        return cls(
            id=str(payload["id"]),
            platform=str(payload["platform"]),
            family=family,
            structure=structure,
            regime=str(payload["regime"]),
            route_status=status,
            qualification=qualification,
            rungs=() if is_trellis else rungs,
            rungs_q256=rungs if is_trellis else (),
            activation_contract=activation_contract,
            requires_serve_flags=flags,
            requires_plugin=requires_plugin,
            predicates=_parse_predicates(payload["predicates"], where),
            is_trellis=is_trellis,
            executes=executes,
            launch_rungs_q256=launch_scopes,
            launch_covered_rungs_q256=_launch_rule_coverage(
                launch_scopes, allowable, covered_rungs_q256),
            residency_modes=cell_modes,
            runtime_image=runtime_image,
            execution_modes=execution_modes,
            runtime_vllm=runtime_vllm,
            runtime_torch=runtime_torch,
            evidence=evidence,
            runtime_tessera_commit=runtime_commit,
            runtime_serving_source_sha256=runtime_digest,
            covered_rungs_q256=covered_rungs_q256,
            run_tables=run_tables,
            runtime_kernel_build=parse_kernel_build(payload.get("runtime", {}), where),
        )

    def launch_covers_rate(self, index: int, rate_q256: int) -> bool:
        """Whether the launch at ``executes[index]`` covers ``rate_q256``.

        A launch without a v12 ``rungs_q256`` scope keeps the scope of its
        cell. A scoped launch covers its census rungs plus the allowable
        run-table rungs those rungs derive, evaluated once at parse beside
        the cell's own coverage, so both halves narrow together.
        """
        scopes = tuple(getattr(self, "launch_rungs_q256", ()) or ())
        scope = scopes[index] if index < len(scopes) else None
        if scope is None:
            return self.covers_rate(rate_q256)
        if rate_q256 in scope:
            return True
        covered = tuple(getattr(self, "launch_covered_rungs_q256", ()) or ())
        scope_covered = covered[index] if index < len(covered) else None
        return scope_covered is not None and rate_q256 in scope_covered

    def covers_rate(self, rate_q256: int) -> bool:
        return rate_q256 in self.rungs_q256 or rate_q256 in self.covered_rungs_q256

    @property
    def covered_rates(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.rungs_q256) | set(self.covered_rungs_q256)))

    def covers_rung(self, facts: UnitStructuralFacts) -> bool:
        """Whether this cell's published rung list names the unit's rung.

        A unit whose rung is ``None`` -- an unpublished CB K, a trellis rate
        outside the family's reader range -- is covered by nothing. That is the
        point of the list: absence means unattested.
        """
        if self.is_trellis:
            return (facts.rate_q256 is not None
                    and self.covers_rate(facts.rate_q256))
        return facts.k is not None and facts.k in self.rungs

    def matches(self, facts: UnitStructuralFacts) -> bool:
        return all(
            _predicate_holds(facts.fact(name), op, value)
            for name, op, value in self.predicates
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "platform": self.platform,
            "family": self.family,
            "structure": self.structure,
            "regime": self.regime,
            "route_status": self.route_status,
            "qualification": self.qualification,
        }
        if self.is_trellis:
            payload["rungs_q256"] = list(self.rungs_q256)
            if self.run_tables is not None:
                payload["run_tables"] = [list(t) for t in self.run_tables]
            payload["activation_contract"] = self.activation_contract
        else:
            payload["rungs"] = list(self.rungs)
        payload["requires_serve_flags"] = list(self.requires_serve_flags)
        if self.requires_plugin:
            # Emitted only when non-empty, so a keyless cell's serialization
            # is byte-identical to what it was before this key existed.
            payload["requires_plugin"] = self.requires_plugin
        if self.executes:
            scopes = tuple(getattr(self, "launch_rungs_q256", ()) or ())
            payload["executes"] = [
                ({"symbol": symbol, "decoder": decoder}
                 if index >= len(scopes) or scopes[index] is None
                 else {"symbol": symbol, "decoder": decoder,
                       "rungs_q256": list(scopes[index])})
                for index, (symbol, decoder) in enumerate(self.executes)
            ]
        if self.runtime_image:
            payload["runtime"] = {
                "image": self.runtime_image, "execution_modes": list(self.execution_modes),
            }
            if self.runtime_kernel_build:
                payload["runtime"]["kernel_build"] = self.runtime_kernel_build
            if self.runtime_serving_source_sha256:
                # Emitted only when the cell names its code, so a cell that
                # names none serializes exactly as it did before v41.
                payload["runtime"]["tessera_commit"] = self.runtime_tessera_commit
                payload["runtime"]["serving_source_sha256"] = (
                    self.runtime_serving_source_sha256)
        return payload


@dataclass(frozen=True)
class PlatformEntry:
    """One ``lane_eligibility.platforms`` entry, under a platform-axis schema.

    ``executes`` is the whole point: a map over every family the contract
    publishes, whose value is that family's OWN route contract when the
    pinned runtime dispatches those bytes natively on this device, and
    ``None`` when it does not. ``None`` is a claim, not an omission -- the
    publisher's validator refuses a family left out of the map precisely so a
    consumer can tell "unbacked here" from "this document did not say".

    A producer reads this to price a target it has never served on. It is not
    an attestation that anything WAS served: that is a cell, and a platform
    with no cell keeps resolving ``unattested`` at the seam export gates on
    (``serving_profiles.ServingLaneSpec.route_status_for``).
    """

    key: str
    backend: str
    arch_key: str
    arch: Any
    serve_image: "str | None"
    executes: Mapping[str, "str | None"]

    def backs(self, family: str) -> bool:
        """Does the pinned runtime execute ``family`` natively on this device?"""
        return self.executes.get(family) is not None

    def answer(self) -> dict[str, Any]:
        """The projection a reviewer reads, in a stable order."""
        return {
            "backend": self.backend,
            self.arch_key: self.arch,
            "serve_image": self.serve_image,
            "executes": {k: self.executes[k] for k in sorted(self.executes)},
        }


def _parse_platform_entries(
    platforms_block: Mapping[str, Any],
    contracts_by_family: Mapping[str, "str | None"],
    where: str,
) -> dict[str, PlatformEntry]:
    """The v10 platform grammar, transcribed from the publisher's validator.

    Transcribed and not re-derived: the closed sets above mirror
    ``tessera.serving.contract``'s, and the one rule this reader adds nothing
    to is the last -- a non-null ``executes`` value must equal the
    ``activation_contract`` the family's own ``formats[]`` row publishes. That
    is what stops a platform entry from naming a contract the dispatch does
    not run, and it is checked here rather than trusted because a value a gate
    reads is either derived or refused (principle 14).
    """
    entries: dict[str, PlatformEntry] = {}
    for key, entry in platforms_block.items():
        at = f"{where}.platforms[{str(key)!r}]"
        if not isinstance(entry, Mapping):
            raise LaneEligibilityError(
                f"{at} must be a JSON object. Before schema v10 a platform was "
                "a bare KEY, and a reader that goes on treating it as one "
                "cannot see that a family is published unbacked here -- which "
                "is the whole point of the axis, and why v10 is not additive")
        backend = entry.get("backend")
        if backend not in PLATFORM_BACKENDS:
            raise LaneEligibilityError(
                f"{at}.backend must be one of {sorted(PLATFORM_BACKENDS)}, got "
                f"{backend!r}; the device type cannot answer it, because a "
                'ROCm torch reports device.type "cuda" for an AMD device')
        arch_key = PLATFORM_ARCH_KEYS[str(backend)]
        if arch_key not in entry:
            raise LaneEligibilityError(
                f"{at} declares backend {backend!r} and must carry {arch_key!r}")
        others = sorted(
            (set(PLATFORM_ARCH_KEYS.values()) - {arch_key}) & set(entry))
        if others:
            raise LaneEligibilityError(
                f"{at} declares backend {backend!r} and also carries {others}; "
                "a platform key names one device, and two architecture "
                "spellings under one key is two devices sharing an identity")
        _require_keys(
            entry, at,
            required={"backend", arch_key, "serve_image", "executes"},
            optional=set(PLATFORM_OPTIONAL_KEYS),
        )
        serve_image = entry["serve_image"]
        if serve_image is not None and (
                not isinstance(serve_image, str)
                or _DIGEST_IMAGE.fullmatch(serve_image) is None):
            raise LaneEligibilityError(
                f"{at}.serve_image must be a digest-pinned image or null, got "
                f"{serve_image!r}")
        executes = entry["executes"]
        if not isinstance(executes, Mapping) or set(executes) != set(
                contracts_by_family):
            raise LaneEligibilityError(
                f"{at}.executes must name every family in formats[] "
                f"({sorted(contracts_by_family)}), got "
                f"{sorted(executes) if isinstance(executes, Mapping) else executes!r}. "
                "A family left out is not 'unbacked' -- it is a question this "
                "document declined to answer about a platform it declares, and "
                "a consumer cannot tell the two apart; null is how a table says "
                "unbacked")
        for family, value in executes.items():
            if value is None:
                continue
            expected = contracts_by_family[family]
            if value != expected:
                raise LaneEligibilityError(
                    f"{at}.executes[{family!r}] is {value!r}, but that family's "
                    f"formats[] row publishes activation_contract {expected!r}. "
                    "A platform entry does not get to name a contract the "
                    "dispatch does not run: the value is DERIVED from the route "
                    "or it is a claim about a runtime nobody read")
        entries[str(key)] = PlatformEntry(
            key=str(key),
            backend=str(backend),
            arch_key=arch_key,
            arch=entry[arch_key],
            serve_image=serve_image,
            executes={str(f): (None if v is None else str(v))
                      for f, v in executes.items()},
        )
    return entries


@dataclass(frozen=True)
class EligibilityTable:
    """The packaged ``lane_eligibility`` block, or the ABSENT sentinel.

    ``present is False`` is not an error and not a zero. It is the state in
    which this repository declines to make a route claim at all.

    ``families`` is the set of payload families the pinned contract's
    ``formats[]`` table publishes. It is the SCOPE of this table's authority:
    inside it, silence is a refusal; outside it, the runtime has said nothing
    about these bytes one way or the other and the gate reports rather than
    refuses.
    """

    present: bool
    runtime_version: str
    runtime_commit: str
    contract_sha256: str
    schema: str = ""
    platforms: tuple[str, ...] = ()
    #: The v10 platform axis, keyed by platform id. Empty under every earlier
    #: grammar, where the entry's VALUE was never read -- so a caller must
    #: distinguish "no entry" from "``executes`` says null" and
    #: :meth:`platform_executes` does that with a third state rather than
    #: letting an older table read as a measured refusal.
    platform_entries: Mapping[str, PlatformEntry] = field(
        default_factory=dict)
    regimes: tuple[str, ...] = ()
    structures: tuple[str, ...] = ()
    cells: tuple[EligibilityCell, ...] = ()
    families: frozenset[str] = frozenset()
    trellis_families: frozenset[str] = frozenset()
    absent_reason: str = ""
    #: The ``native_extensions[].lane`` claims of the same contract, in table
    #: order: which decoder each extension serves and the predicate (if any)
    #: its bytes must satisfy. Read by :func:`cell_lane_admits` at every
    #: admission leg; ``()`` for a v3 table, whose only launches are torch's.
    lanes: tuple[LaneClaim, ...] = ()

    def governs(self, family: str) -> bool:
        """Whether the pinned contract publishes a codec for this family."""
        return family in self.families

    def platform_executes(self, family: str, platform: str) -> "str | None":
        """The contract this platform executes ``family`` by, or the third state.

        Returns the family's route contract when the platform backs it,
        ``None`` when the entry publishes ``null`` -- the pinned runtime has no
        native route for those bytes on that device, which is a measured
        platform fact -- and :data:`PLATFORM_EXECUTES_UNSTATED` when this table
        makes no statement at all: an earlier grammar, an absent table, or a
        platform key the document does not carry. The three are kept apart on
        purpose; folding ``unstated`` into ``None`` would report an unread
        question as a measured refusal, which is the mistake principle 14's
        corollary is about.
        """
        entry = self.platform_entries.get(platform)
        if entry is None:
            return PLATFORM_EXECUTES_UNSTATED
        return entry.executes.get(family, PLATFORM_EXECUTES_UNSTATED)

    def platform_backs(self, family: str, platform: str) -> bool:
        """Fail-closed: only a published non-null contract is backing."""
        executed = self.platform_executes(family, platform)
        return executed is not None and executed != PLATFORM_EXECUTES_UNSTATED

    def provenance(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "present" if self.present else "absent",
            "serving_runtime_version": self.runtime_version,
            "serving_runtime_commit": self.runtime_commit,
            "contract_sha256": self.contract_sha256,
            "lane_eligibility_schema": self.schema or None,
        }
        if not self.present:
            payload["reason"] = self.absent_reason
        else:
            payload["platforms"] = list(self.platforms)
            if self.platform_entries:
                payload["platform_executes"] = {
                    key: dict(entry.executes)
                    for key, entry in sorted(self.platform_entries.items())
                }
            payload["regimes"] = list(self.regimes)
            payload["structures"] = list(self.structures)
            payload["published_families"] = sorted(self.families)
            payload["trellis_families"] = sorted(self.trellis_families)
            payload["cell_ids"] = [cell.id for cell in self.cells]
            required_plugins = sorted({
                cell.requires_plugin for cell in self.cells
                if cell.requires_plugin
            })
            if required_plugins:
                # Only when non-empty: a table without the key keeps a payload
                # is unchanged by this widening.
                payload["required_plugins"] = required_plugins
            if self.lanes:
                # The predicate the gate decided against, verbatim from the
                # contract: a receipt that names the rule it was read under.
                payload["lanes"] = [
                    claim.answer() | {"extension": claim.extension}
                    for claim in self.lanes
                ]
        return payload


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeRoute:
    regime: str
    route_status: str
    cell_id: str | None
    requires_serve_flags: tuple[str, ...] = ()
    #: The vLLM plugins the matched cell requires, aggregated exactly as
    #: ``requires_serve_flags`` is. A tuple rather than a scalar because it
    #: rolls up the same way at unit granularity, and one shape at both levels
    #: is what stops a consumer having to special-case the regime view.
    requires_plugins: tuple[str, ...] = ()
    qualification: str = ""
    activation_contract: str = ""
    detail: str = ""
    executes: tuple[tuple[str, str], ...] = ()
    residency: str = ""
    runtime_image: str = ""
    execution_mode: str = ""
    kernel_build: str = ""
    #: v6 evidence, carried so a shipcard says WHICH grade attested this
    #: regime (principle 12). Recorded, never gated on: see
    #: :func:`cell_evidence_admits`.
    evidence_grade: str = ""
    evidence_smoke: str = ""
    #: v7's derived attribution ("" on an older table) and v8's encoder scope
    #: (``None`` when none was recorded, or on an older table), carried for
    #: the same reason: a shipcard has to say which encoder wrote the bytes
    #: this unit's KL was measured on, because the current one may not
    #: reproduce them.
    evidence_attribution: str = ""
    evidence_artifact: EvidenceArtifact | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "regime": self.regime,
            "route_status": self.route_status,
            "cell_id": self.cell_id,
            "requires_serve_flags": list(self.requires_serve_flags),
            "qualification": self.qualification or None,
            "activation_contract": self.activation_contract or None,
            "detail": self.detail,
        }
        if self.requires_plugins:
            payload["requires_plugins"] = list(self.requires_plugins)
        if self.executes:
            payload["executes"] = [
                {"symbol": symbol, "decoder": decoder}
                for symbol, decoder in self.executes
            ]
        if self.residency:
            payload["residency"] = self.residency
        if self.runtime_image:
            payload["runtime_image"] = self.runtime_image
            payload["execution_mode"] = self.execution_mode
        if self.kernel_build:
            payload["kernel_build"] = self.kernel_build
        if self.evidence_grade:
            payload["evidence_grade"] = self.evidence_grade
            payload["evidence_smoke"] = self.evidence_smoke
            payload["evidence_attribution"] = self.evidence_attribution
            payload["evidence_artifact"] = (
                self.evidence_artifact.as_dict() if self.evidence_artifact else None)
        return payload


@dataclass(frozen=True)
class UnitRoute:
    """One unit's resolved route status across every declared regime."""

    facts: UnitStructuralFacts
    route_status: str
    regimes: tuple[RegimeRoute, ...] = ()
    requires_serve_flags: tuple[str, ...] = ()
    #: The vLLM plugins every backed regime of this unit requires, aggregated
    #: over the same regimes ``requires_serve_flags`` is aggregated over. An
    #: artifact whose units carry a non-empty set is servable ONLY where those
    #: plugins are installed, which is a fact its serve command and its
    #: shipcard have to carry.
    requires_plugins: tuple[str, ...] = ()
    #: True when the pinned contract publishes this unit's payload family, i.e.
    #: when the eligibility table is the authority for these bytes; False when
    #: it does not. ``None`` means the question was never asked, which is the
    #: only honest value when no table was consulted at all (an absent index,
    #: an unreadable contract). It is NOT a synonym for True: a default that
    #: rides into provenance unevaluated is the same defect class as a zero
    #: that reads as a verdict, and ``as_dict`` omits the key rather than
    #: publish one.
    in_scope: bool | None = None
    #: Why no claim was made. Empty unless the status is ``unattested``.
    unattested_reason: str = ""

    @property
    def attested(self) -> bool:
        return self.route_status != ROUTE_STATUS_UNATTESTED

    @property
    def fallback_regimes(self) -> tuple[str, ...]:
        return tuple(
            r.regime for r in self.regimes
            if r.route_status == ROUTE_STATUS_FALLBACK
        )

    @property
    def unattested_regimes(self) -> tuple[str, ...]:
        return tuple(
            r.regime for r in self.regimes
            if r.route_status == ROUTE_STATUS_UNATTESTED
        )

    @property
    def qualifications(self) -> tuple[str, ...]:
        return tuple(sorted({
            r.qualification for r in self.regimes if r.qualification
        }))

    @property
    def activation_contracts(self) -> tuple[str, ...]:
        return tuple(sorted({
            r.activation_contract for r in self.regimes
            if r.activation_contract
        }))

    def as_dict(self) -> dict[str, Any]:
        payload = {
            **self.facts.as_dict(),
            "route_status": self.route_status,
            "requires_serve_flags": list(self.requires_serve_flags),
            "regime_routes": [r.as_dict() for r in self.regimes],
        }
        if self.requires_plugins:
            payload["requires_plugins"] = list(self.requires_plugins)
        if self.in_scope is not None:
            payload["in_scope"] = self.in_scope
        if self.attested:
            payload["announced_fallback_regimes"] = list(self.fallback_regimes)
            payload["qualifications"] = list(self.qualifications)
            payload["activation_contracts"] = list(self.activation_contracts)
        else:
            payload["unattested_reason"] = self.unattested_reason
            payload["unattested_regimes"] = list(self.unattested_regimes)
        return payload


def resolve_unit_route(
    facts: UnitStructuralFacts,
    table: EligibilityTable,
    *,
    platform: str | None = None,
    residency: str | None = None,
    runtime_image: str | None = None,
    execution_mode: str | None = None,
    serving_source_sha256: Any = PINNED_SERVING_SOURCE,
    kernel_build: str | None = None,
) -> UnitRoute:
    """Resolve one unit's route status against the pinned eligibility table.

    Absent table -> ``unattested`` with no regime detail. Never a zero, never a
    guess, and never principle 9's ``backed`` by default.

    ``platform`` is the exact runtime platform id the artifact targets (the
    serving profile's ``target_platform``, e.g. ``sm_121``). Lane cells are
    platform-scoped, so resolving without one cannot name a route: a missing or
    unpublished platform yields ``unattested``, never a match-any.

    v4 and v5 additionally require ``residency``, the explicit serve mode for the
    artifact. It filters cells before route selection; omitting it cannot
    choose whichever same-scope cell happened to be listed first. v3 keeps
    its original behavior and makes no claim about executed launches.
    V5 also requires ``runtime_image`` and ``execution_mode``. Every regime
    must resolve on that same complete target; cells from different runtime
    scopes cannot jointly attest one artifact.

    ``serving_source_sha256`` is the pinned serving code digest (#1561); the
    default reads the tracked pin. A scoped cell that names this unit but was
    measured on other code, or on none, is kept as a refusal beside its
    regime, naming the cell and both digests, exactly as a cell refused by
    its own evidence is. ``None`` (a v2 pin) skips the check.
    """
    if not table.present:
        return UnitRoute(
            facts=facts,
            route_status=ROUTE_STATUS_UNATTESTED,
            unattested_reason=table.absent_reason,
        )

    if not table.governs(facts.payload_family):
        return UnitRoute(
            facts=facts,
            route_status=ROUTE_STATUS_UNATTESTED,
            in_scope=False,
            unattested_reason=(
                f"payload family {facts.payload_family!r} is not published in "
                f"the pinned release's formats table "
                f"({sorted(table.families)}); the lane eligibility table is "
                "not the authority for these bytes and makes no claim about "
                "them either way"
            ),
        )

    if not platform:
        return UnitRoute(
            facts=facts,
            route_status=ROUTE_STATUS_UNATTESTED,
            in_scope=True,
            unattested_reason=(
                "no declared target platform; lane cells are platform-scoped, so "
                "no route can be named without one. Declare "
                "'target_platform' on the serving profile this artifact "
                f"targets; the pinned release publishes {list(table.platforms)}"
            ),
        )
    if platform not in table.platforms:
        return UnitRoute(
            facts=facts,
            route_status=ROUTE_STATUS_UNATTESTED,
            in_scope=True,
            unattested_reason=(
                f"the pinned release publishes no lane cells for platform "
                f"{platform!r} (it publishes {list(table.platforms)}); an "
                "unpublished platform attests nothing"
            ),
        )

    is_v4 = table.schema in _LAUNCH_SCHEMAS
    is_scoped = table.schema in SCOPED_LANE_SCHEMAS
    if not is_scoped and (
            runtime_image is not None or execution_mode is not None or kernel_build is not None):
        return UnitRoute(
            facts=facts, route_status=ROUTE_STATUS_UNATTESTED, in_scope=True,
            unattested_reason=legacy_runtime_scope_refusal(table.schema))
    if is_v4 and residency not in TESSERA_RESIDENCY_MODES:
        return UnitRoute(
            facts=facts,
            route_status=ROUTE_STATUS_UNATTESTED,
            in_scope=True,
            unattested_reason=(
                f"no declared supported residency (got {residency!r}); v4 "
                "cells require an explicit residency to identify their "
                f"launches: {sorted(TESSERA_RESIDENCY_MODES)}"
            ),
        )

    serving_context = None
    if is_scoped:
        try:
            serving_context = ServingContext(
                platform=platform, structure=facts.structure, residency=residency,
                runtime_image=runtime_image, execution_mode=execution_mode,
                kernel_build=kernel_build)
        except LaneEligibilityError as exc:
            return UnitRoute(facts=facts, route_status=ROUTE_STATUS_UNATTESTED,
                             in_scope=True, unattested_reason=str(exc))

    matched = [
        cell for cell in table.cells
        if cell.platform == platform
        and cell.family == facts.payload_family
        and cell.structure == facts.structure
        and (not is_v4 or residency in cell.residency_modes)
        and (not is_scoped or cell_matches_serving_context(
            cell, serving_context, serving_source_sha256=None))
        and cell.covers_rung(facts)
        and cell.matches(facts)
    ]
    # Every table, scoped or not: a legacy grammar has no runtime block, so
    # its cells name no code and a v3 pin admits none of them.
    pinned_code = resolve_serving_source_sha256(serving_source_sha256)
    # A cell whose own published evidence refuses it is NOT dropped silently
    # into "no cell names this unit": the two are different facts and the
    # shipcard has to be able to tell them apart. Keep the refusal beside its
    # regime so the regime route can name the cell and the reason. The lane
    # predicate is the same kind of fact -- a cell names this unit, and the
    # kernel it launches through would refuse the bytes -- and lands in the
    # same slot.
    candidates: list[EligibilityCell] = []
    refusals: dict[str, tuple[str, str]] = {}
    #: The launches each admitted cell makes at THIS rung (PQ #1274): a cell
    #: that launches through a lane and beside it runs the lane only where
    #: its predicate admits the plan, so the route records what runs here,
    #: not the cell's union over its rungs.
    rung_launches: dict[str, tuple[tuple[str, str], ...]] = {}
    for cell in matched:
        # The code scope first: evidence taken on other code says nothing
        # about the pinned code, whatever the evidence itself says.
        admits, why = cell_serving_code_admits(cell, pinned_code)
        if admits:
            admits, why = cell_evidence_admits(cell)
        if admits:
            admits, why, launches = cell_rung_launches(
                cell, facts.rate_q256, table.lanes)
        if admits:
            candidates.append(cell)
            rung_launches[cell.id] = launches
        elif cell.regime not in refusals:
            refusals[cell.regime] = (cell.id, why)

    regimes: list[RegimeRoute] = []
    for regime in table.regimes:
        best: EligibilityCell | None = None
        for cell in candidates:
            if cell.regime != regime:
                continue
            if best is None or _CELL_RANK[cell.route_status] > _CELL_RANK[
                    best.route_status]:
                best = cell
        if best is None:
            refused = refusals.get(regime)
            if refused is not None:
                # A cell DOES name this unit here; its own evidence, or the
                # predicate of the lane it launches through, refuses it.
                # Recording the cell id and the published reason is what
                # makes this reviewable -- "no cell" and "a cell that failed
                # its smoke" are opposite facts about the runtime.
                regimes.append(RegimeRoute(
                    regime=regime,
                    route_status=ROUTE_STATUS_UNATTESTED,
                    cell_id=refused[0],
                    detail=refused[1],
                ))
                continue
            # No packaged cell names this unit in this regime. Under a
            # closed-world table that is the ONLY negative signal there is,
            # and it must not be laundered into a verdict: the honest state is
            # "the runtime made no claim", and the export gate fails closed on
            # it for any family the contract governs.
            regimes.append(RegimeRoute(
                regime=regime,
                route_status=ROUTE_STATUS_UNATTESTED,
                cell_id=None,
                detail=(
                    "no packaged lane cell names this platform, family, "
                    "structure and rung in this regime; a rung the table does "
                    "not list is unattested, never admitted"
                ),
            ))
            continue
        regimes.append(RegimeRoute(
            regime=regime,
            route_status=best.route_status,
            cell_id=best.id,
            requires_serve_flags=best.requires_serve_flags,
            requires_plugins=(
                (best.requires_plugin,) if best.requires_plugin else ()),
            qualification=best.qualification,
            activation_contract=best.activation_contract,
            executes=rung_launches.get(best.id, best.executes),
            residency=str(residency) if is_v4 else "",
            runtime_image=str(runtime_image) if is_scoped else "",
            execution_mode=str(execution_mode) if is_scoped else "",
            kernel_build=kernel_build or "",
            evidence_grade=best.evidence.grade if best.evidence else "",
            evidence_smoke=best.evidence.smoke_status if best.evidence else "",
            evidence_attribution=(
                best.evidence.smoke_attribution if best.evidence else ""),
            evidence_artifact=best.evidence.artifact if best.evidence else None,
        ))

    unclaimed = [
        r for r in regimes if r.route_status == ROUTE_STATUS_UNATTESTED
    ]
    backed = [
        r for r in regimes
        if r.route_status in (ROUTE_STATUS_BACKED,
                              ROUTE_STATUS_BACKED_WITH_SERVE_FLAG)
    ]
    flags = tuple(sorted({
        flag for r in backed for flag in r.requires_serve_flags
    }))
    plugins = tuple(sorted({
        plugin for r in backed for plugin in r.requires_plugins
    }))

    if unclaimed:
        # Partial coverage is not coverage. One unclaimed regime means the
        # runtime has not said this unit serves everywhere it will be asked to.
        status = ROUTE_STATUS_UNATTESTED
        reason = (
            f"no lane cell covers regime(s) "
            f"{[r.regime for r in unclaimed]} for {facts.payload_family} "
            f"rung {facts.k if facts.k is not None else facts.rate_q256!r} on "
            f"{platform}"
            + (f" at residency {residency!r}" if is_v4 else "")
            + (f", runtime_image {runtime_image!r}, execution_mode {execution_mode!r}"
               if is_scoped else "")
        )
        return UnitRoute(
            facts=facts,
            route_status=status,
            regimes=tuple(regimes),
            in_scope=True,
            unattested_reason=reason,
        )

    if not backed:
        # Every regime serves, but none natively. Principle 9: a unit with no
        # backed route for its declared target is UNBACKED. This IS attested --
        # the runtime published a fallback for each regime and nothing better.
        status = ROUTE_STATUS_UNBACKED
    elif flags:
        status = ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    else:
        status = ROUTE_STATUS_BACKED

    return UnitRoute(
        facts=facts,
        route_status=status,
        regimes=tuple(regimes),
        in_scope=True,
        requires_serve_flags=flags,
        requires_plugins=plugins,
    )


#: Ranking among the cells that DO match, so the best published route wins a
#: regime. ``unattested`` is deliberately absent: it is produced by absence, is
#: never a cell status, and therefore never competes here.
_CELL_RANK = {
    ROUTE_STATUS_FALLBACK: 1,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG: 2,
    ROUTE_STATUS_BACKED: 3,
}


# ---------------------------------------------------------------------------
# Loading the pinned contract
# ---------------------------------------------------------------------------
def load_eligibility_table(
    version: str | None = None,
    *,
    contract_path: Path | None = None,
) -> EligibilityTable:
    """Load the eligibility table the pinned SERVING runtime packages.

    ``contract_path`` names the packaged ``runtime_contract.json`` of the
    runtime whose routes are being attested; the caller resolves it from that
    runtime's own installed package, never from a copy in this repository
    (``tessera_render.tessera_serving_contract_path`` is the one live caller).

    Until 2026-09-02 this function had a second mode: with no
    ``contract_path`` it read Gridbook's SERVING pin and the byte-verbatim
    contract copy indexed under ``prismaquant/gridbook_runtime/``. The
    Gridbook lane is retired (``archive/gridbook_lane_2026-09-02/``), the
    materialized copies went with it, and there is no default table any more.
    Calling with neither argument is therefore not an error but an honest
    ABSENCE: it returns a table with ``present=False``, so every unit resolves
    to ``UNATTESTED`` and the export gate fails closed, which is exactly what
    "no pinned runtime claims this route" should mean.
    """
    version = str(version or "")
    commit = ""
    path: Path | None = Path(contract_path) if contract_path is not None else None

    if path is None or not path.exists():
        return EligibilityTable(
            present=False,
            runtime_version=version,
            runtime_commit=commit,
            contract_sha256="",
            absent_reason=(
                "no packaged runtime contract was supplied, so no serving "
                "lane can be attested. Pass the pinned runtime's own "
                "runtime_contract.json (contract_path=). Route status stays "
                "UNATTESTED until then."
            ),
        )

    sha = _sha256(path)

    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaneEligibilityError(
            f"cannot read {path}: {exc}") from exc

    block = contract.get("lane_eligibility")
    if block is None:
        return EligibilityTable(
            present=False,
            runtime_version=version or "",
            runtime_commit=commit,
            contract_sha256=sha,
            absent_reason=(
                f"{path} packages no 'lane_eligibility' table, so no "
                "serving-lane route can be attested for this pin. This is a "
                "REFUSAL TO CLAIM, not a clean bill: the runtime's lane "
                "predicates exist but are not published."
            ),
        )

    return _parse_table(
        block, contract.get("formats", ()), version or "", commit, sha,
        native_extensions=contract.get("native_extensions"))


def load_published_formats(
    version: str | None = None,
    *,
    contract_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """The pinned release's PUBLISHED format table, keyed by family.

    ``formats[]`` carries ``family``, ``name_pattern`` and, since contract v12,
    a ``kind`` discriminator: a ``cb_product`` row carries ``grid``/``mode``/
    ``n_sub``/``rungs``, while a RATE-addressed row (``tcq_trellis`` in a
    retired Gridbook contract, ``tessera_wire`` in Tessera's) carries
    ``attested_rungs_q256`` (named ``candidate_rungs_q256`` before contract v2,
    which Tessera keeps as a deprecated alias and says it drops at schema v2)
    /``reader_rate_range_q256``/``native_terminal_q256``. A unit's payload family, sub-table split, rung
    legality and body rate are therefore genuinely DERIVED here rather than
    read out of a local table -- which is the point of principle 14.
    """
    path = Path(contract_path) if contract_path is not None else None
    if path is None or not path.exists():
        return {}
    contract = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(entry["family"]): dict(entry)
        for entry in contract.get("formats", ())
        if isinstance(entry, Mapping) and entry.get("family")
    }


def _name_prefix(entry: Mapping[str, Any]) -> str:
    """The literal head of a format's ``name_pattern``, e.g. ``TCQ_E2M1_R``.

    Keying on the pattern rather than on the family is what lets one resolver
    serve every kind: a CB family IS its name prefix (``FP8_CB_K``), a
    rate-addressed family is not (``TCQ_E2M1_R256`` and ``TESSERA_E2M1_K2_R896``
    name rates around a 256-weight block, and are never a rung of themselves).
    """
    pattern = str(entry.get("name_pattern", ""))
    head, sep, _ = pattern.partition("{k}")
    if not sep:
        return ""
    return head.upper()


def resolve_payload_rung(
    format_name: str,
    published_formats: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, int | None, int | None]:
    """``(payload_family, k, rate_q256)`` for a format name, DERIVED.

    The one place a format name is turned into the runtime's own vocabulary,
    so the export gate and the serving-lane resolver cannot disagree about
    what ``FP8_CB_K44`` or ``TCQ_E2M1_R512`` is. Both rung fields are ``None``
    when the pinned release publishes no such rung, which is what makes every
    downstream match fail closed instead of admitting an unlisted rate.

    Returns the raw upper-cased name as the family when the contract publishes
    no codec for it (BF16, a SOURCE passthrough, a stock CT rung), which is
    also the signal that the lane table is not the authority for those bytes.
    """
    if published_formats is None:
        published_formats = load_published_formats()

    upper = str(format_name).upper()
    candidates = sorted(
        ((_name_prefix(entry), str(fam), entry)
         for fam, entry in published_formats.items()),
        key=lambda item: -len(item[0]),
    )
    for prefix, fam, entry in candidates:
        if not prefix or not upper.startswith(prefix):
            continue
        suffix = upper[len(prefix):]
        if not suffix.isdigit():
            continue
        value = int(suffix)
        if str(entry.get("kind", FORMAT_KIND_CB_PRODUCT)) in (
                RATE_ADDRESSED_FORMAT_KINDS):
            lo, hi = (int(v) for v in entry["reader_rate_range_q256"])
            # Outside the published reader range the rate stays None, so every
            # cell's rung list fails to cover it and the unit is unattested.
            return fam, None, (value if lo <= value <= hi else None)
        if value in {int(r) for r in entry.get("rungs", ())}:
            return fam, value, None
        # The pinned release does not instantiate this rung: leaving k None
        # makes every k-predicate and every cell match fail closed rather than
        # pass silently.
        return fam, None, None
    return upper, None, None


def unit_structural_facts(
    qname: str,
    format_name: str,
    *,
    is_routed_moe: bool,
    role_split: bool,
    in_features: int,
    out_features: int,
    published_formats: Mapping[str, Mapping[str, Any]] | None = None,
) -> UnitStructuralFacts:
    """Build one unit's facts, with family/n_sub/k/rate DERIVED from the contract.

    ``role_split`` is the fact the DSv4 defect turned on. It is True when the
    unit's expert stack binds more than one codebook across its projections --
    knowable only after the per-``(qname, format)`` codebook cells resolve, and
    therefore only at export.
    """
    if published_formats is None:
        published_formats = load_published_formats()

    family, k, rate_q256 = resolve_payload_rung(format_name, published_formats)
    n_sub: int | None = None
    if k is not None:
        entry = published_formats.get(family, {})
        n_sub = int(entry["n_sub"])

    return UnitStructuralFacts(
        qname=str(qname),
        format_name=str(format_name),
        payload_family=family,
        k=k,
        n_sub=n_sub,
        structure=STRUCTURE_ROUTED_MOE if is_routed_moe else STRUCTURE_DENSE,
        role_split=bool(role_split),
        in_features=int(in_features),
        out_features=int(out_features),
        rate_q256=rate_q256,
    )


def _published_families(formats: Any) -> tuple[frozenset[str], frozenset[str]]:
    """``(all families, rate-addressed families)`` from the formats table.

    The second set is what ``EligibilityCell.is_trellis`` is built from, and
    it holds every family whose ``kind`` is in
    :data:`RATE_ADDRESSED_FORMAT_KINDS` -- the retired lane's ``tcq_trellis`` and
    Tessera's ``tessera_wire`` alike. Such a family's cells carry
    ``rungs_q256``; a ``cb_product`` family's carry ``rungs``.
    """
    if not isinstance(formats, Sequence) or isinstance(formats, (str, bytes)):
        raise LaneEligibilityError(
            "runtime_contract.formats must be a JSON array; the lane table's "
            "rung vocabulary is decided by each family's published kind")
    families: set[str] = set()
    trellis: set[str] = set()
    for i, entry in enumerate(formats):
        if not isinstance(entry, Mapping):
            raise LaneEligibilityError(
                f"runtime_contract.formats[{i}] must be a JSON object")
        family = str(entry.get("family", ""))
        if not family:
            raise LaneEligibilityError(
                f"runtime_contract.formats[{i}] publishes no family")
        kind = str(entry.get("kind", FORMAT_KIND_CB_PRODUCT))
        if kind not in FORMAT_KINDS:
            raise LaneEligibilityError(
                f"runtime_contract.formats[{i}].kind {kind!r} is not one of "
                f"{sorted(FORMAT_KINDS)}")
        families.add(family)
        if kind in RATE_ADDRESSED_FORMAT_KINDS:
            trellis.add(family)
    return frozenset(families), frozenset(trellis)


def _parse_table(block: Any, formats: Any, version: str, commit: str, sha: str,
                 *, native_extensions: Any) -> EligibilityTable:
    """Read a ``lane_eligibility`` block beside the ``native_extensions`` it launches through.

    ``native_extensions`` is the contract's own table, or ``None`` when the
    contract publishes none. It is a REQUIRED argument rather than a default
    because every cell since v4 names the launches it executes, and a launch
    through an extension is subject to that extension's published predicate:
    a caller that forgets the table would build a table whose lane gate
    passes everything, and this reader would rather not compile than do that.
    A v3 table publishes no launches and reads ``()`` lanes.
    """
    where = "runtime_contract.lane_eligibility"
    if not isinstance(block, Mapping):
        raise LaneEligibilityError(f"{where} must be a JSON object")
    # The schema string is checked BEFORE the key set, deliberately. An older
    # table fails both, and "missing field(s) ['cells', 'platforms']" would
    # send its reader off to add keys to a v2 block rather than to
    # re-materialize the contract from a release with a supported schema.
    if block.get("schema") not in LANE_ELIGIBILITY_SCHEMAS:
        raise LaneEligibilityError(
            f"{where}.schema must be one of "
            f"{sorted(LANE_ELIGIBILITY_SCHEMAS)}, got {block.get('schema')!r}. "
            "An older lane table is not a subset of these -- v1/v2 cells are "
            "not platform-scoped and carry no rung list, so reading one here "
            "would admit every rung on every platform. An unrecognised VENDOR "
            "prefix is a table this repository was not handed at all. Either "
            "way, re-materialize the contract from a release that publishes a "
            "schema named above rather than editing the table.")
    _require_keys(
        block, where,
        required={"schema", "platforms", "regimes", "structures", "cells"},
        optional=set(),
    )

    platforms_block = block["platforms"]
    if not isinstance(platforms_block, Mapping) or not platforms_block:
        raise LaneEligibilityError(
            f"{where}.platforms must be a non-empty JSON object keyed by "
            "platform id")
    platforms = tuple(str(p) for p in platforms_block)
    platform_entries: dict[str, PlatformEntry] = {}
    if str(block["schema"]) in PLATFORM_AXIS_LANE_SCHEMAS:
        # The family -> executed-contract map the entries are checked against,
        # read off the same ``formats[]`` rows the rest of this parse uses.
        contracts_by_family = {
            str(entry["family"]): (
                None if entry.get("activation_contract") is None
                else str(entry["activation_contract"]))
            for entry in formats
            if isinstance(entry, Mapping) and entry.get("family")
        }
        platform_entries = _parse_platform_entries(
            platforms_block, contracts_by_family, where)

    regimes = tuple(str(r) for r in block["regimes"])
    if not regimes or len(set(regimes)) != len(regimes):
        raise LaneEligibilityError(
            f"{where}.regimes must be a non-empty list of unique ids")

    structures = tuple(str(s) for s in block["structures"])
    if not structures or len(set(structures)) != len(structures):
        raise LaneEligibilityError(
            f"{where}.structures must be a non-empty list of unique ids")
    unknown = sorted(set(structures) - STRUCTURES)
    if unknown:
        raise LaneEligibilityError(
            f"{where}.structures names {unknown}, which this repository has no "
            f"dispatch path for; the known set is {sorted(STRUCTURES)}")

    families, trellis_families = _published_families(formats)
    schema = str(block["schema"])
    is_v4 = schema in _LAUNCH_SCHEMAS
    is_scoped = schema in SCOPED_LANE_SCHEMAS
    family_modes: dict[str, tuple[str, ...]] = {}
    if is_v4:
        for i, entry in enumerate(formats):
            family = str(entry["family"])
            modes = entry.get("residency_modes")
            if (not isinstance(modes, list) or not modes
                    or any(not isinstance(mode, str)
                           or mode not in TESSERA_RESIDENCY_MODES for mode in modes)
                    or len(set(modes)) != len(modes)):
                raise LaneEligibilityError(
                    f"runtime_contract.formats[{i}].residency_modes must "
                    "publish a non-empty list of distinct supported "
                    f"residencies {sorted(TESSERA_RESIDENCY_MODES)}")
            family_modes[family] = tuple(modes)

    format_by_family = {str(e["family"]): e for e in formats}
    allowable = ({family: _allowable_rung_tables(entry, f"formats[{family}]")
                  for family, entry in format_by_family.items()}
                 if schema in _RULE_COVERAGE_SCHEMAS else {})

    cells_block = block["cells"]
    if not isinstance(cells_block, Sequence) or isinstance(
            cells_block, (str, bytes)):
        raise LaneEligibilityError(f"{where}.cells must be a JSON array")
    cells = []
    for i, cell in enumerate(cells_block):
        at = f"{where}.cells[{i}]"
        family = str(cell.get("family", "")) if isinstance(cell, Mapping) else ""
        tables, covered = (None, ())
        if schema in _RULE_COVERAGE_SCHEMAS:
            tables, covered = _cell_rule_coverage(
                cell, allowable.get(family, {}), at)
        cells.append(EligibilityCell.from_dict(
            cell, at, trellis_families=trellis_families, schema=schema,
            residency_modes=family_modes.get(family, ()),
            covered_rungs_q256=covered, run_tables=tables,
            allowable=allowable.get(family)))
    cells = tuple(cells)

    for cell in cells:
        if cell.regime not in regimes:
            raise LaneEligibilityError(
                f"{where}.cells[{cell.id!r}].regime {cell.regime!r} is not a "
                f"declared regime {list(regimes)}")
        if cell.platform not in platforms:
            raise LaneEligibilityError(
                f"{where}.cells[{cell.id!r}].platform {cell.platform!r} is not "
                f"a declared platform {list(platforms)}")
        if cell.structure not in structures:
            raise LaneEligibilityError(
                f"{where}.cells[{cell.id!r}].structure {cell.structure!r} is "
                f"not a declared structure {list(structures)}")
        if cell.family not in families:
            raise LaneEligibilityError(
                f"{where}.cells[{cell.id!r}].family {cell.family!r} is not "
                f"published in runtime_contract.formats "
                f"({sorted(families)}); a lane cell for a codec the runtime "
                "does not publish attests a route to nothing")
    ids = [cell.id for cell in cells]
    if len(set(ids)) != len(ids):
        raise LaneEligibilityError(f"{where}.cells ids must be unique")

    lanes: tuple[LaneClaim, ...] = ()
    if is_v4:
        extension_launches = sorted({
            symbol for cell in cells for symbol, _decoder in cell.executes
            if "::" in symbol})
        if native_extensions is None and extension_launches:
            # Refused whenever a cell names a qualified launch and no
            # extension table is published. With a table, the decoders it
            # serves say whether such a launch rides an extension (below);
            # with no table there is nothing to read that against, so a
            # launch that does ride one would pass ungated. A table whose
            # cells launch only through torch/vLLM paths has no qualified
            # launch, no lane to decide, and reads () lanes, exactly as a v3
            # table does.
            raise LaneEligibilityError(
                f"runtime_contract publishes a {schema} lane table whose cells "
                f"launch through an extension ({extension_launches}), but no "
                "'native_extensions' table: the lane predicates those "
                "launches are subject to cannot be read, so no launch through "
                "an extension can be decided. Re-materialize the contract "
                "from a release that publishes both, never one of them.")
        if native_extensions is not None:
            lanes = parse_lane_claims(
                native_extensions, "runtime_contract.native_extensions")
        # Bind every extension launch a cell names to the lane that serves
        # it. The lane gate keys on the DECODER (the name Tessera's census
        # stamps and its launch table derives cells from), so a cell that
        # launched through an extension's symbol under another decoder name
        # would slip past the gate; that is a contract inconsistency and it
        # is refused here, once, where the two tables meet.
        lane_decoders = {claim.extension: claim.decoder for claim in lanes}
        served_by_a_lane = {
            decoder: extension for extension, decoder in lane_decoders.items()}
        for cell in cells:
            for symbol, decoder in cell.executes:
                prefix, sep, _rest = symbol.partition("::")
                if not sep:
                    continue        # a torch/vLLM launch: the route's own path
                if prefix not in lane_decoders:
                    # A qualified symbol whose prefix is not an extension's
                    # module_name_prefix. It is NOT automatically an
                    # extension launch: Tessera's contract v34 mints four
                    # dense cells on `tessera::window_gemm_dense` and states
                    # in the same entry that the launch "carries lane null
                    # because it is a launch, not an extension lane", and the
                    # two tables agree -- no native_extensions row declares
                    # it, and its decoder is no lane's decoder. Such a launch
                    # stands exactly where `torch._scaled_mm` stands: the
                    # route's own path, gated by the cell's route status and
                    # evidence, with no wire predicate to read
                    # (`lane_claim_for_cell` already returns None for it).
                    #
                    # What the prefix rule really guarded is still guarded,
                    # one field over and on the field the gate actually keys
                    # on: a launch that takes a decoder some lane SERVES,
                    # while naming an extension no row declares, would escape
                    # that lane's predicate, so it is refused here.
                    if decoder in served_by_a_lane:
                        raise LaneEligibilityError(
                            f"{where}.cells[{cell.id!r}].executes launches "
                            f"{symbol!r} under decoder {decoder!r}, which "
                            f"native_extensions[{served_by_a_lane[decoder]!r}]"
                            ".lane serves, but no native_extensions row "
                            f"declares {prefix!r}; a launch read by that "
                            "lane must name the extension that publishes "
                            "the predicate, or it escapes it")
                    continue
                if decoder != lane_decoders[prefix]:
                    raise LaneEligibilityError(
                        f"{where}.cells[{cell.id!r}].executes launches {symbol!r} "
                        f"under decoder {decoder!r}, but native_extensions "
                        f"[{prefix}].lane.decoder is {lane_decoders[prefix]!r}; a "
                        "launch through an extension is read by that "
                        "extension's lane, and a cell that names another "
                        "decoder for it would escape the lane's predicate")
        scopes: dict[tuple, str] = {}
        for cell in cells:
            for mode in cell.residency_modes:
                for execution in cell.execution_modes if is_scoped else ("",):
                    base = (cell.platform, cell.family, cell.structure, cell.regime, mode, execution)
                    keys = (("image", cell.runtime_image, base),
                            ("build", cell_kernel_build(cell), base)) if is_scoped else (("unscoped", base),)
                    for scope in keys:
                        previous = scopes.get(scope)
                        if previous is not None:
                            raise LaneEligibilityError(
                                f"{where}.cells {previous!r} and {cell.id!r} both cover "
                                f"{scope}; overlapping serving scopes make route "
                                "resolution depend on cell order")
                        scopes[scope] = cell.id

    return EligibilityTable(
        present=True,
        runtime_version=version,
        runtime_commit=commit,
        contract_sha256=sha,
        schema=schema,
        platforms=platforms,
        platform_entries=platform_entries,
        regimes=regimes,
        structures=structures,
        cells=cells,
        families=families,
        trellis_families=trellis_families,
        lanes=lanes,
    )


def parse_kernel_build(payload: Mapping[str, Any], where: str) -> str:
    """Read the optional portable build name without a new qualification claim."""
    if "kernel_build" not in payload:
        return ""
    value = payload["kernel_build"]
    if not isinstance(value, str) or not value or value != value.strip():
        raise LaneEligibilityError(f"{where}.kernel_build must be a complete non-empty name")
    return value


def _cell_identifier(cell: Any) -> str:
    return getattr(cell, "id", None) or cell.cell_id


def cell_kernel_build(cell: Any) -> str:
    """Use the same effective build name for lookup and compatibility keys."""
    return getattr(cell, "runtime_kernel_build", "") or "legacy:" + _cell_identifier(cell)


def cell_key_compatibility(cells) -> dict[str, tuple]:
    """Map historical receipt names to build and module-kind keys."""
    return {_cell_identifier(cell): (cell_kernel_build(cell), cell.structure,
            cell.platform, cell.family, cell.regime, cell.residency_modes,
            cell.execution_modes) for cell in cells}


def parse_runtime_scope(payload: Any, where: str, *, require_versions: bool = False
                        ) -> tuple[str, tuple[str, ...], str, str]:
    """The per-cell ``runtime`` grammar, shared by both contract readers.

    v5 published ``{image, execution_modes}``; v6 requires ``{image,
    execution_modes, vllm, torch}`` and withdrew the global
    ``versions.attested_on``. The two are parsed by ONE function with a flag
    rather than by two, because the only difference is which keys are required
    and a second copy is how the image check and the mode check drift apart.

    Returned as ``(image, execution_modes, vllm, torch)``; the two version
    strings are ``""`` under the v5 grammar, which published neither.
    """
    if not isinstance(payload, Mapping):
        raise LaneEligibilityError(f"{where} must be a JSON object")
    required = {"image", "execution_modes"}
    if require_versions:
        required |= {"vllm", "torch"}
    _require_keys(payload, where, required=required, optional={"kernel_build"})
    parse_kernel_build(payload, where)
    image = payload["image"]
    if not isinstance(image, str) or not _DIGEST_IMAGE.fullmatch(image):
        raise LaneEligibilityError(
            f"{where}.image must be an exact repository@sha256:<64 lowercase hex> reference")
    modes = payload["execution_modes"]
    if (not isinstance(modes, list) or not modes
            or any(not isinstance(mode, str) or mode not in TESSERA_EXECUTION_MODES for mode in modes)
            or len(set(modes)) != len(modes)):
        raise LaneEligibilityError(
            f"{where}.execution_modes must be a non-empty list of distinct values from "
            f"{sorted(TESSERA_EXECUTION_MODES)}")
    vllm = torch_version = ""
    if require_versions:
        vllm = payload["vllm"]
        torch_version = payload["torch"]
        for name, value in (("vllm", vllm), ("torch", torch_version)):
            if not isinstance(value, str) or not value.strip():
                raise LaneEligibilityError(
                    f"{where}.{name} must be the non-empty version string this "
                    f"cell was measured under, got {value!r}")
    return image, tuple(modes), str(vllm), str(torch_version)


#: The code half of a cell's ``runtime`` block (Tessera contract v41), spelled
#: as Tessera's ``contract.RUNTIME_CODE_KEYS``: optional, both or neither.
RUNTIME_CODE_KEYS = ("tessera_commit", "serving_source_sha256")


def parse_runtime_code(payload: Any, where: str) -> tuple[str, str]:
    """``(tessera_commit, serving_source_sha256)`` a cell names, or ``("", "")``.

    Both or neither, exactly as the publisher's ``cell_runtime_code`` reads
    them: a commit with no digest gives a program nothing to compare, and a
    digest with no commit gives a person no tree to find. The shapes are
    checked because a malformed digest can never equal a pinned one, and a
    table that carries one is not the table the publisher's validator admits.
    """
    if not isinstance(payload, Mapping):
        raise LaneEligibilityError(f"{where} must be a JSON object")
    present = [key for key in RUNTIME_CODE_KEYS if key in payload]
    if not present:
        return "", ""
    if len(present) != len(RUNTIME_CODE_KEYS):
        missing = [key for key in RUNTIME_CODE_KEYS if key not in present]
        raise LaneEligibilityError(
            f"{where} names {present} without {missing}; a cell names the "
            "Tessera code it was measured on with both fields or neither")
    commit, digest = payload["tessera_commit"], payload["serving_source_sha256"]
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise LaneEligibilityError(
            f"{where}.tessera_commit must be a 40-hex lowercase commit, got {commit!r}")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise LaneEligibilityError(
            f"{where}.serving_source_sha256 must be 64 lowercase hex digits, got {digest!r}")
    return commit, digest


def parse_v5_runtime(payload: Any, where: str) -> tuple[str, tuple[str, ...]]:
    """The v5 spelling, kept for callers that want only the two v5 fields."""
    image, modes, _, _ = parse_runtime_scope(payload, where)
    return image, modes


def parse_v4_cell_contract(
    payload: Mapping[str, Any],
    where: str,
    *,
    residency_modes: Sequence[str],
    schema: str = LANE_ELIGIBILITY_SCHEMA_TESSERA_LEGACY_V3,
    cell_rungs: tuple[int, ...] = (),
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[int, ...] | None, ...], tuple[str, ...]]:
    """Parse the v4 launch set and residency selector, without runtime imports.

    The runtime owns whether a launch is correct for its route. This consumer
    verifies the published grammar and preserves that claim; it never derives
    a launch from a family name or a cell ID.

    Under lane schema v12 a launch may name ``rungs_q256``: the sorted census
    rungs it covers. A launch without the key keeps the scope of its cell.
    Under every earlier schema the key is refused, because a reader that
    accepted it and ignored its meaning would read a scoped launch at every
    rung of its cell.
    """
    raw_executes = payload.get("executes")
    if not isinstance(raw_executes, list) or not raw_executes:
        raise LaneEligibilityError(
            f"{where}.executes must be a non-empty JSON array of "
            "{symbol, decoder} objects")
    scoped_schema = schema == LANE_ELIGIBILITY_SCHEMA_TESSERA_V12
    launches: list[tuple[str, str]] = []
    scopes: list[tuple[int, ...] | None] = []
    for i, launch in enumerate(raw_executes):
        spot = f"{where}.executes[{i}]"
        if not isinstance(launch, Mapping):
            raise LaneEligibilityError(f"{spot} must be a JSON object")
        _require_keys(launch, spot, required={"symbol", "decoder"},
                      optional={"rungs_q256"} if scoped_schema else set())
        if any(not isinstance(launch[key], str) or not launch[key].strip()
               for key in ("symbol", "decoder")):
            raise LaneEligibilityError(
                f"{spot}.symbol and decoder must be non-empty strings")
        launches.append((launch["symbol"], launch["decoder"]))
        scopes.append(_parse_launch_rungs(launch, spot, cell_rungs, scoped_schema))
    if len(set(launches)) != len(launches):
        raise LaneEligibilityError(
            f"{where}.executes must not repeat a (symbol, decoder) pair")

    head = "TESSERA_SERVE_MODE="
    flags = payload.get("requires_serve_flags")
    if (not isinstance(flags, list)
            or any(not isinstance(flag, str) or not flag for flag in flags)):
        raise LaneEligibilityError(
            f"{where}.requires_serve_flags must be a JSON array of non-empty strings")
    named = [flag for flag in flags if flag.startswith(head)]
    if len(named) != 1:
        raise LaneEligibilityError(
            f"{where}.requires_serve_flags must name exactly one "
            "TESSERA_SERVE_MODE residency flag")
    modes = tuple(named[0][len(head):].split("|"))
    if (len(set(modes)) != len(modes)
            or any(mode not in TESSERA_RESIDENCY_MODES for mode in modes)):
        raise LaneEligibilityError(
            f"{where}.requires_serve_flags names invalid or repeated "
            f"residency values {list(modes)}")
    if not set(modes).issubset(residency_modes):
        raise LaneEligibilityError(
            f"{where}.requires_serve_flags residency {list(modes)} exceeds "
            f"the family's published residency_modes {list(residency_modes)}")
    return tuple(launches), tuple(scopes), modes

def _parse_launch_rungs(launch: Mapping[str, Any], spot: str,
                        cell_rungs: tuple[int, ...], scoped_schema: bool,
                        ) -> tuple[int, ...] | None:
    """One launch's ``rungs_q256`` scope, or ``None`` for the cell's scope.

    Under lane schema v12 the key is optional: absent means the launch covers
    the full scope of its cell. Present means a sorted, non-empty list of
    unique integer census rungs inside the cell's own rung list. The checks
    are strict by design: an empty, unsorted, duplicated, non-integer or
    out-of-cell list would silently narrow or widen the launch, and a reader
    that normalized it instead of refusing would read a different scope than
    the publisher attested. Under every earlier schema any key is refused.
    """
    if "rungs_q256" not in launch:
        return None
    if not scoped_schema:
        raise LaneEligibilityError(
            f"{spot}.rungs_q256 names a per-launch scope, but this table's "
            "lane schema publishes no per-launch scope; a scoped launch read "
            "under an earlier grammar would run at every rung of its cell")
    raw = launch["rungs_q256"]
    if not isinstance(raw, list) or isinstance(raw, (str, bytes)):
        raise LaneEligibilityError(f"{spot}.rungs_q256 must be a JSON array")
    if not raw:
        raise LaneEligibilityError(
            f"{spot}.rungs_q256 must name at least one rung; an empty launch "
            "scope covers nothing and would silently drop its launch")
    out: list[int] = []
    for i, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int):
            raise LaneEligibilityError(
                f"{spot}.rungs_q256[{i}] must be an integer rung, got {item!r}")
        out.append(int(item))
    if len(set(out)) != len(out):
        raise LaneEligibilityError(f"{spot}.rungs_q256 must not repeat a rung")
    if out != sorted(out):
        raise LaneEligibilityError(
            f"{spot}.rungs_q256 must list its rungs in ascending order")
    outside = sorted(set(out) - set(cell_rungs))
    if outside:
        raise LaneEligibilityError(
            f"{spot}.rungs_q256 names {outside}, which this cell's own "
            f"rungs_q256 {sorted(cell_rungs)} does not publish; a launch "
            "covers rungs of its cell, never rungs beside them")
    return tuple(out)

def _parse_rungs(payload: Any, where: str) -> tuple[int, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise LaneEligibilityError(f"{where} must be a JSON array")
    if not payload:
        raise LaneEligibilityError(
            f"{where} must name at least one rung; an empty rung list covers "
            "nothing and would silently make its cell unreachable")
    out: list[int] = []
    for i, item in enumerate(payload):
        if isinstance(item, bool) or not isinstance(item, int):
            raise LaneEligibilityError(
                f"{where}[{i}] must be an integer rung, got {item!r}")
        out.append(int(item))
    if len(set(out)) != len(out):
        raise LaneEligibilityError(f"{where} must not repeat a rung")
    return tuple(out)


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------
_PREDICATE_OPS = frozenset({
    "equals", "in", "multiple_of", "at_least", "at_most",
})


def _parse_predicates(payload: Any, where: str
                      ) -> tuple[tuple[str, str, Any], ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise LaneEligibilityError(
            f"{where}.predicates must be a JSON array")
    out: list[tuple[str, str, Any]] = []
    for i, item in enumerate(payload):
        spot = f"{where}.predicates[{i}]"
        if not isinstance(item, Mapping):
            raise LaneEligibilityError(f"{spot} must be a JSON object")
        _require_keys(item, spot, required={"fact", "op", "value"}, optional=set())
        fact = str(item["fact"])
        if fact not in _PREDICABLE_FACTS:
            raise LaneEligibilityError(
                f"{spot}.fact {fact!r} is not an attestable structural fact; "
                f"the closed set is {sorted(_PREDICABLE_FACTS)}. An unknown "
                "predicate is a malformed contract, never a no-op rule.")
        op = str(item["op"])
        if op not in _PREDICATE_OPS:
            raise LaneEligibilityError(
                f"{spot}.op {op!r} is not one of {sorted(_PREDICATE_OPS)}")
        out.append((fact, op, item["value"]))
    return tuple(out)


def _predicate_holds(actual: Any, op: str, value: Any) -> bool:
    if actual is None:
        # A cell that predicates on a fact the unit does not have cannot claim
        # it. Fail-closed, not "unconstrained".
        return False
    if op == "equals":
        return actual == value
    if op == "in":
        return actual in list(value)
    if op == "multiple_of":
        return int(value) != 0 and int(actual) % int(value) == 0
    if op == "at_least":
        return int(actual) >= int(value)
    if op == "at_most":
        return int(actual) <= int(value)
    raise LaneEligibilityError(f"unknown predicate op {op!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_keys(payload: Mapping[str, Any], where: str, *,
                  required: set[str], optional: set[str]) -> None:
    """Admit one published object through the shared tolerant rule (#1548).

    Every field this reader consumes must be present; a field it does not know
    is accepted and never read, unless the producer lists it in the object's
    ``must_understand`` array. See :mod:`prismaquant.record_fields`.
    """
    record_fields.admit_fields(payload, where, required=required, optional=optional,
                               error=LaneEligibilityError)


_sha256 = file_sha256hex


__all__ = [
    "ATTRIBUTED_SMOKE_LANE_SCHEMAS",
    "CellEvidence",
    "CellKlEvidence",
    "ENCODER_SCOPED_LANE_SCHEMAS",
    "RECORDED_SMOKE_LANE_SCHEMAS",
    "EVIDENCE_ARTIFACT_METRIC",
    "EVIDENCE_CONTROL_OUTCOMES",
    "EVIDENCE_CONTROL_REFERENCES",
    "EVIDENCE_GRADES",
    "EVIDENCE_KL_KINDS",
    "EVIDENCE_LANE_SCHEMAS",
    "EVIDENCE_PAYLOAD_RELATIONS",
    "EVIDENCE_SMOKE_ATTRIBUTIONS",
    "EVIDENCE_SMOKE_REFUSALS",
    "EVIDENCE_SMOKE_STATUSES",
    "EVIDENCE_WEIGHT_ERROR_RELATIONS",
    "EvidenceArtifact",
    "LANE_ELIGIBILITY_SCHEMA_TESSERA",
    "LANE_ELIGIBILITY_SCHEMA_TESSERA_V10",
    "LANE_ELIGIBILITY_SCHEMA_TESSERA_LEGACY_V3",
    "LANE_ELIGIBILITY_SCHEMA_TESSERA_V5",
    "LANE_ELIGIBILITY_SCHEMA_TESSERA_V6",
    "LANE_ELIGIBILITY_SCHEMA_TESSERA_V7",
    "LANE_ELIGIBILITY_SCHEMA_TESSERA_V8",
    "LANE_ELIGIBILITY_SCHEMA_TESSERA_V9",
    "LANE_ELIGIBILITY_SCHEMA_TESSERA_V11",
    "LANE_ELIGIBILITY_SCHEMA_TESSERA_V12",
    "LANE_ELIGIBILITY_SCHEMAS",
    "LANE_BODIES",
    "LANE_FIELDS",
    "LANE_PLANES",
    "LANE_REQUIREMENT_CARRIES",
    "LANE_REQUIREMENT_FIELDS",
    "LANE_REQUIREMENT_LISTS",
    "LANE_ROTATION_STATES",
    "LaneClaim",
    "PLATFORM_AXIS_LANE_SCHEMAS",
    "PLATFORM_BACKENDS",
    "PLATFORM_ARCH_KEYS",
    "PLATFORM_EXECUTES_UNSTATED",
    "PlatformEntry",
    "SCOPED_LANE_SCHEMAS",
    "SmokeControl",
    "SmokeRecord",
    "SmokeRecordRow",
    "cell_evidence_admits",
    "cell_lane_admits",
    "cell_rung_launches",
    "cell_serving_code_admits",
    "lane_claim_for_cell",
    "lane_claims_for_cell",
    "parse_lane_claim",
    "parse_lane_claims",
    "derive_evidence_grade",
    "derive_smoke_attribution",
    "parse_cell_evidence",
    "parse_runtime_code",
    "parse_runtime_scope",
    "resolve_serving_source_sha256",
    "parse_v4_cell_contract",
    "ROUTE_ATTESTATION_SCHEMA",
    "ROUTE_STATUS_BACKED",
    "ROUTE_STATUS_BACKED_WITH_SERVE_FLAG",
    "ROUTE_STATUS_UNBACKED",
    "ROUTE_STATUS_UNATTESTED",
    "ROUTE_STATUS_FALLBACK",
    "LANE_ROUTE_STATUSES",
    "REGIME_ROUTE_STATUSES",
    "CELL_ROUTE_STATUSES",
    "CELL_QUALIFICATIONS",
    "QUALIFICATION_COMPILE_ONLY",
    "QUALIFICATION_DEVICE_QUALIFIED",
    "FORMAT_KIND_CB_PRODUCT",
    "FORMAT_KIND_TCQ_TRELLIS",
    "FORMAT_KIND_TESSERA_WIRE",
    "FORMAT_KINDS",
    "RATE_ADDRESSED_FORMAT_KINDS",
    "STRUCTURE_DENSE",
    "STRUCTURE_ROUTED_MOE",
    "LaneEligibilityError",
    "UnitStructuralFacts",
    "EligibilityCell",
    "EligibilityTable",
    "RegimeRoute",
    "UnitRoute",
    "resolve_unit_route",
    "load_eligibility_table",
    "load_published_formats",
    "resolve_payload_rung",
    "unit_structural_facts",
]
