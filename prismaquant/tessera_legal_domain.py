"""The complete legal Tessera rate domain for GLM, and its support ledger.

Work package A of the quality-prefill experiment
(``docs/design/tessera_quality_prefill_experiment.md`` section 2.1).  It answers
one question and refuses to answer any other: **which ``(family, rate)`` pairs
exist at the pins this process actually reads, what do their bytes weigh, and
what does each of five independent support facts say about them.**

Three properties are the whole point.

**The domain is derived, never asserted.**  Every rate comes from
``tessera_formats.family_q256_bounds`` walked against
``tessera_menu.tessera_shape_legal`` at the real GLM Linear shapes.  The
audited counts -- 1,793 for ``TESSERA_E4M3_K1`` and 3,841 for
``TESSERA_BF16_K1`` -- are stated here as an *oracle to compare against*
(:data:`AUDITED_RATE_COUNTS`), never as the value returned.  If the derivation
disagrees, :func:`audit_oracle_report` reports the disagreement; nothing in
this module clamps a domain to reach a number.

**The five support facts are five sets, not a ladder.**  ``producer_legal``,
``reader_supported``, ``implemented_route``, ``native_qualification`` and
``export_and_served_validation`` are five separate fields on
:class:`SupportFacts`, each with its own ``source`` string naming the table
that answered.  Membership in one implies nothing about the next.  At the
current pin that asymmetry is extreme and it is the ledger's main content:
5,634 producer-legal candidates, and exactly three ``(family, rate,
structure)`` cells with native qualification.  A ledger that let singleton
attestation shrink the research domain would be wrong, so
:func:`build_inventory` computes the domain without consulting the attestation
at all and attaches it afterwards.

**A pin the code reads is not a pin the inventory was frozen under.**
:func:`live_pins` re-reads every pin at call time; :data:`FROZEN_PINS` records
what they were when the audited counts were taken; :func:`pin_drift` reports
the difference.  ``__main__`` prints the pin block and the drift verdict
*before* the counts, so a stale pin can never arrive as a silent number.

What this module deliberately does not do: it defines no manifest schema, no
driver CLI, no population selection and no allocation.  Its return values are
plain frozen dataclasses and dicts; the manifest package is expected to wrap
them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from .quality_prefill_population import RateDomain as _RateDomain
from .tessera_formats import (
    TesseraFamily,
    TesseraFormatError,
    family_q256_bounds,
    get_tessera_family,
    scale_plane_name,
    tessera_wire_recipe,
)


SCHEMA = "prismaquant.tessera_legal_domain.v1"

#: The two families section 2.1 requires.  Tessera-8 and Tessera-16.
PRIMARY_FAMILIES = ("TESSERA_E4M3_K1", "TESSERA_BF16_K1")

#: Retained as an existing, separately licensed control (section 2.1).  It is
#: NOT part of the primary roster and never enters a primary count.  It is
#: carried because it is the only family at this pin where ``producer_legal``
#: and ``reader_supported`` disagree, which is the property the ledger exists
#: to be able to express.
CONTROL_FAMILIES = ("TESSERA_E2M1_K2",)

#: The counts the frozen source audit derived, to compare a fresh derivation
#: against.  ``tessera-legal-domain-source-audit-01.md``, SHA-256
#: ``b2ec2c6be76c6c0c557ffa642a9290e26f960b9781fb0e843c9d101fbc2c88b2``.
#: **This is an oracle, not an answer.**  Nothing reads it to produce a domain.
AUDITED_RATE_COUNTS = {"TESSERA_E4M3_K1": 1793, "TESSERA_BF16_K1": 3841}

#: The audit's stated table-width transitions, likewise an oracle.  The widths
#: this module reports come from ``tessera_wire_recipe(family, rate)``.
AUDITED_BF16_TABLE_TRANSITIONS = ((256, 14), (3585, 15), (3841, 16))

#: The model this inventory is built for.  ``export_and_served_validation`` is
#: scope-bound to it: a cell's evidence artifact counts only when the artifact
#: is this model.  Principle 14's corollary -- a recorded capability claim
#: inherits the scope of the artifact it was measured on -- so an artifact from
#: another model is recorded in the fact's ``detail`` and never in its value.
ARTIFACT_SCOPE = "GLM-5.3-Flash"


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

#: Where the real GLM-5.3-Flash config lives when it is mounted.  The shapes
#: below are derived from it by :func:`glm_linear_shapes`; the test re-derives
#: them from this file rather than trusting the constant.
GLM53_CONFIG_PATH = "/mnt/shared/models/GLM-5.3-Flash-BF16/config.json"


def glm_linear_shapes(config: Mapping[str, object]) -> tuple[tuple[int, int], ...]:
    """The distinct ``(rows, columns)`` a GLM text-tower Linear takes.

    ``rows`` is out-features and ``columns`` is in-features, which is the
    orientation every Tessera accountant in the tree uses: the Bresenham rate
    schedule walks the *column* axis (``TesseraFamily.column_schedule``) and a
    CHANNEL scale plane charges one fp16 per *row*.

    Three widths generate all five shapes -- ``hidden_size`` (attention and
    every MLP input), ``intermediate_size`` (the dense MLP) and
    ``moe_intermediate_size`` (a shared or routed expert) -- so the roster is
    derived from the config's own three integers rather than transcribed.
    """
    text = config.get("text_config", config)
    if not isinstance(text, Mapping):
        raise TesseraFormatError("config.text_config must be an object")
    try:
        hidden = int(text["hidden_size"])
        dense = int(text["intermediate_size"])
        expert = int(text["moe_intermediate_size"])
    except KeyError as exc:
        raise TesseraFormatError(
            f"GLM config publishes no {exc.args[0]!r}; the Linear shapes "
            "cannot be derived and must not be guessed"
        ) from exc
    shapes = {
        (hidden, hidden),   # attention projections that square the residual
        (dense, hidden),    # dense MLP gate/up
        (hidden, dense),    # dense MLP down
        (expert, hidden),   # shared / routed expert gate/up
        (hidden, expert),   # shared / routed expert down
    }
    return tuple(sorted(shapes))


#: The five shapes :func:`glm_linear_shapes` derives from the GLM-5.3-Flash
#: config (hidden 4096, dense 12288, expert 2048).  Frozen so the domain walk
#: needs no mounted model; ``tests/test_tessera_legal_domain.py`` re-derives
#: them from the real config when it is mounted and refuses on a mismatch.
GLM53_LINEAR_SHAPES = (
    (2048, 4096), (4096, 2048), (4096, 4096), (4096, 12288), (12288, 4096),
)


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DomainPins:
    """Every pin that decides what this module answers.

    ``producer_installed_contract_sha256`` is the only field that describes the
    ``tessera`` package actually importable in this process; the rest are
    literals PrismaQuant tracks.  They are separate on purpose: a producer that
    is not the pinned one is exactly the drift :func:`pin_drift` exists to
    surface, and collapsing the two would hide it.
    """

    reader_dev_pin_commit: str
    reader_dev_pin_contract_sha256: str
    serving_runtime_pinned_commit: str
    serving_runtime_pinned_version: str
    serving_runtime_pinned_contract_sha256: str
    producer_installed_contract_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "reader_dev_pin_commit": self.reader_dev_pin_commit,
            "reader_dev_pin_contract_sha256": self.reader_dev_pin_contract_sha256,
            "serving_runtime_pinned_commit": self.serving_runtime_pinned_commit,
            "serving_runtime_pinned_version": self.serving_runtime_pinned_version,
            "serving_runtime_pinned_contract_sha256":
                self.serving_runtime_pinned_contract_sha256,
            "producer_installed_contract_sha256":
                self.producer_installed_contract_sha256,
        }


#: The producer commit the *specification* names as the frozen study producer.
#: Documentary only, and deliberately NOT a member of :class:`DomainPins`: it
#: appears nowhere in PrismaQuant's source, it is 25 commits after the reader
#: pin, and putting it in the comparable block would make the drift report fire
#: on every run against a value no code reads.  The frozen source audit records
#: that its pertinent recipe and contract behaviour match the reader pin; this
#: module derives from the reader pin's own packaged contract and says so.
SPEC_NAMED_STUDY_PRODUCER = "d403cc5a3199a348cc7ee6262f4adbdab8138745"


#: sha256 of the *source files* this module's numbers are derived through, at
#: each Tessera state that exists on this box.  The point is that ``import
#: tessera`` resolves to whatever is installed -- on this box an editable
#: install of a working checkout at ``a9eb572e``, which is neither pin and is
#: not a descendant of either.  A number derived through that state is derived
#: through none of the pins, and would match the audit only by coincidence.
#: :func:`tessera_source_state` hashes the bytes actually imported and names
#: which state they are, so every derived number carries its origin.
#:
#: ``grammar.py`` is byte-identical at all three states: the rate-grammar
#: refusals that set the domain endpoints have not moved at all.  ``export.py``
#: separates them, and it is where ``_window_bits_for`` and ``wire_recipe``
#: live.
TESSERA_SOURCE_STATES = {
    "reader-pin-387eda36": {
        "commit": "387eda36fd410d6b2a4fb86b22285eab2a5e072c",
        "export.py":
            "9f7604bcba619673a6fe6d3de97737c39ba4d372749ef7f946bf53cd9dd92e87",
    },
    "study-producer-d403cc5a": {
        "commit": "d403cc5a3199a348cc7ee6262f4adbdab8138745",
        "export.py":
            "b1b04f269edc137b7d4b2195b332a647501ace067ecb8eb7304d05eab5b8950d",
    },
    "unpinned-working-checkout-a9eb572e": {
        "commit": "a9eb572e1b90b17f716562192910681e65430fba",
        "export.py":
            "79e8b8f4301870c943b79bde4c411e59958a74cd6c2ce0a821efe4c1ea9c1c3b",
    },
}

#: ``grammar.py``'s digest, which all three states share.  The rate bounds and
#: the whole-unit-quota refusal -- the two things that decide where the legal
#: domain starts and stops -- are these bytes at every state, so the roster is
#: not a function of which one is imported.  That is a derived fact, not an
#: assumption: it is asserted against the importable tree by the tests.
TESSERA_GRAMMAR_DIGEST = (
    "f2545274c8e03534d040c64fb4fd1a02085de7b5eb106f80e8fb31b05454fac8"
)

#: The states whose ``export.py`` bytes produce the same wire for the two
#: primary families.  ``_window_bits_for``, ``wire_recipe``, the WINDOW raw-cap
#: expression and the ``*_WINDOW_BITS`` constants are byte-identical between the
#: reader pin and the frozen study producer; the two files differ only by the
#: additive ``ScalePlaneKind.MX`` plane (a third plane kind, plus its grid
#: refusal, its pack branch and its materialiser), which no ``TESSERA_E4M3_K1``
#: or ``TESSERA_BF16_K1`` rung reaches -- both are WINDOW bodies over CHANNEL.
#: So a number derived through either is derived through both, and this module
#: says so from the bytes rather than repeating the audit's prose.
TESSERA_EQUIVALENT_SOURCE_STATES = (
    "reader-pin-387eda36", "study-producer-d403cc5a",
)


def _file_digest(path: str) -> str:
    """sha256 of a file's bytes."""
    import hashlib

    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def tessera_source_state() -> dict[str, object]:
    """Which Tessera source state the importable package actually is.

    Reads the bytes of the ``tessera.export`` and ``tessera.grammar`` modules
    that ``import`` resolves to and matches them against
    :data:`TESSERA_SOURCE_STATES`.  An unrecognised state is reported as such
    rather than guessed at -- the caller then knows its numbers came from a
    state this module has never compared against the pins.
    """
    import tessera.export as _export
    import tessera.grammar as _grammar

    export_digest = _file_digest(_export.__file__)
    grammar_digest = _file_digest(_grammar.__file__)
    named = [
        name for name, state in TESSERA_SOURCE_STATES.items()
        if state["export.py"] == export_digest
    ]
    state = named[0] if len(named) == 1 else None
    return {
        "schema": SCHEMA,
        "state": state,
        "is_a_pin": state in TESSERA_EQUIVALENT_SOURCE_STATES,
        "commit": (
            TESSERA_SOURCE_STATES[state]["commit"] if state else None
        ),
        "export_path": _export.__file__,
        "export_sha256": export_digest,
        "grammar_sha256": grammar_digest,
        "grammar_matches_every_state": grammar_digest == TESSERA_GRAMMAR_DIGEST,
        "verdict": (
            f"deriving through {state} "
            + ("(a pin, and the two pins' wire is byte-identical for these "
               "families)" if state in TESSERA_EQUIVALENT_SOURCE_STATES
               else "-- NOT a pin: these numbers are derived through no pinned "
                    "state")
            if state else
            "UNRECOGNISED tessera source state: these numbers are derived "
            "through bytes this module has not compared against either pin"
        ),
    }


def live_pins() -> DomainPins:
    """Re-read every pin from the code and the installed package, right now.

    Nothing here is a literal typed in this file.  The two tracked pins are
    read from the modules that own them, and the producer digest is hashed off
    the ``runtime_contract.json`` inside the importable ``tessera``.
    """
    from .tessera_runtime_contract import (
        TESSERA_DEV_PIN_COMMIT,
        TESSERA_DEV_PIN_CONTRACT_SHA256,
    )
    from .tessera_serving_runtime_pin import (
        TESSERA_SERVING_RUNTIME_PINNED_COMMIT,
        TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256,
        TESSERA_SERVING_RUNTIME_PINNED_VERSION,
        installed_tessera_contract_sha256,
    )

    return DomainPins(
        reader_dev_pin_commit=TESSERA_DEV_PIN_COMMIT,
        reader_dev_pin_contract_sha256=TESSERA_DEV_PIN_CONTRACT_SHA256,
        serving_runtime_pinned_commit=TESSERA_SERVING_RUNTIME_PINNED_COMMIT,
        serving_runtime_pinned_version=TESSERA_SERVING_RUNTIME_PINNED_VERSION,
        serving_runtime_pinned_contract_sha256=(
            TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256
        ),
        producer_installed_contract_sha256=installed_tessera_contract_sha256(),
    )


#: What :func:`live_pins` returned when :data:`AUDITED_RATE_COUNTS` was taken.
#: Every value was read from the code and the installed package at that moment,
#: not transcribed from prose; they are literals *here* only so that a later
#: run has something to diff against.
#:
#: **Re-taken 2026-09-12 for the v23 pin**, and the re-take is the reason the
#: audited counts did not have to be re-measured with it.  The pin moved
#: ``387eda36`` -> ``1c827abc`` (contract v22 -> v23, lane schema v9 -> v10).
#: The counts in :data:`AUDITED_RATE_COUNTS` are decided by two files, and both
#: were hashed at the new pin rather than assumed:
#:
#: * ``src/tessera/grammar.py`` is ``f2545274…`` at ``1c827abc`` -- the same
#:   bytes as at every state in :data:`TESSERA_SOURCE_STATES`, so the rate-range
#:   and whole-unit-quota refusals that set the domain endpoints have not moved.
#: * ``src/tessera/export.py`` is ``b1b04f26…`` at ``1c827abc``, which is
#:   BYTE-IDENTICAL to the frozen study producer ``d403cc5a`` already named in
#:   :data:`TESSERA_EQUIVALENT_SOURCE_STATES`.  So the new reader pin's wire
#:   bytes are a state this module had already audited as one answer for the
#:   primary families, and :func:`tessera_source_state` names it by that digest.
#:
#: What v23 changed is the ``platforms`` entry in the packaged contract's lane
#: table (#527), which no number here reads.  A pin move whose ``export.py`` or
#: ``grammar.py`` digest were NEW would land here as a re-measurement, not as a
#: re-transcription; this one is a re-transcription because the bytes say so.
FROZEN_PINS = DomainPins(
    reader_dev_pin_commit="1c827abc4affdd9bed9c6b25af0705480381bf3a",
    reader_dev_pin_contract_sha256=(
        "bafe8a4e9eff8551b34bbd2d7be9c29bf2cfa7bd836724ac9a9ab2f4e0bb922a"
    ),
    serving_runtime_pinned_commit="1c827abc4affdd9bed9c6b25af0705480381bf3a",
    serving_runtime_pinned_version="0.1.0",
    serving_runtime_pinned_contract_sha256=(
        "bafe8a4e9eff8551b34bbd2d7be9c29bf2cfa7bd836724ac9a9ab2f4e0bb922a"
    ),
    producer_installed_contract_sha256=(
        "bafe8a4e9eff8551b34bbd2d7be9c29bf2cfa7bd836724ac9a9ab2f4e0bb922a"
    ),
)


def pin_drift(
    frozen: DomainPins = FROZEN_PINS, live: "DomainPins | None" = None,
) -> dict[str, object]:
    """Field-by-field difference between the frozen and the live pins.

    Returns a report, never a refusal: the caller decides whether a drifted pin
    is a reason to stop.  What the report guarantees is that a drifted run
    cannot be mistaken for the frozen one -- ``matches`` is False and
    ``differences`` names every field with both values.
    """
    observed = live_pins() if live is None else live
    frozen_fields = frozen.as_dict()
    live_fields = observed.as_dict()
    differences = {
        name: {"frozen": frozen_fields[name], "live": live_fields[name]}
        for name in frozen_fields
        if frozen_fields[name] != live_fields[name]
    }
    return {
        "schema": SCHEMA,
        "matches": not differences,
        "frozen": frozen_fields,
        "live": live_fields,
        "differences": differences,
        "spec_named_study_producer": SPEC_NAMED_STUDY_PRODUCER,
        "verdict": (
            "frozen pins match the pins the code reads"
            if not differences else
            "PIN DRIFT: the inventory was frozen under different pins; "
            f"{len(differences)} field(s) differ -- "
            + ", ".join(sorted(differences))
        ),
    }


# ---------------------------------------------------------------------------
# The five support facts
# ---------------------------------------------------------------------------

FACT_PRODUCER_LEGAL = "producer_legal"
FACT_READER_SUPPORTED = "reader_supported"
FACT_IMPLEMENTED_ROUTE = "implemented_route"
FACT_NATIVE_QUALIFICATION = "native_qualification"
FACT_EXPORT_AND_SERVED_VALIDATION = "export_and_served_validation"

#: The five names, in the order section 2.1 lists them.  Exported so a consumer
#: can assert there are five and that it handles all of them.
SUPPORT_FACT_NAMES = (
    FACT_PRODUCER_LEGAL,
    FACT_READER_SUPPORTED,
    FACT_IMPLEMENTED_ROUTE,
    FACT_NATIVE_QUALIFICATION,
    FACT_EXPORT_AND_SERVED_VALIDATION,
)


@dataclass(frozen=True, slots=True)
class SupportFact:
    """One of the five facts: a value, the table that answered, and why.

    ``source`` names the table, never this module.  ``detail`` explains and is
    never read as a value (principle 14).
    """

    name: str
    value: bool
    source: str
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "value": self.value,
            "source": self.source, "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class SupportFacts:
    """Five independent facts about one candidate.  **Never one Boolean.**

    There is deliberately no ``supported`` property and no ``all()`` helper.
    Collapsing these is the failure the spec names: a candidate can be
    producer-legal and reader-unsupported, or reader-supported and natively
    unqualified, and the ledger has to be able to say so.
    """

    producer_legal: SupportFact
    reader_supported: SupportFact
    implemented_route: SupportFact
    native_qualification: SupportFact
    export_and_served_validation: SupportFact

    def facts(self) -> tuple[SupportFact, ...]:
        return (
            self.producer_legal, self.reader_supported,
            self.implemented_route, self.native_qualification,
            self.export_and_served_validation,
        )

    def disagree(self) -> bool:
        """Do the five facts not all carry the same value?"""
        values = {fact.value for fact in self.facts()}
        return len(values) > 1

    def as_dict(self) -> dict[str, object]:
        return {fact.name: fact.as_dict() for fact in self.facts()}


# ---------------------------------------------------------------------------
# Serving scopes, derived from the contract's own cells
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AttestedCell:
    """One native-qualification cell, exactly as the pinned contract wrote it.

    Its ``serving_context`` is built from the cell's own platform, structure,
    runtime image, execution mode and the residency its ``requires_serve_flags``
    names -- never from a default this module chose.  The routed-MoE cells run
    on a *different* runtime image than the dense ones, and asking with the
    contract's ``default_serve_image`` reports them unattested; deriving the
    scope from the cell is what keeps that from reading as an absence.
    """

    cell_id: str
    family: str
    rate_q256: int
    structure: str
    platform: str
    regime: str
    residency: str
    runtime_image: str
    execution_mode: str
    route_status: str
    requires_serve_flags: tuple[str, ...]
    activation_contract: str
    evidence_grade: str
    evidence_artifact_id: "str | None"
    smoke_status: str

    def serving_context(self):
        from .lane_eligibility import ServingContext

        return ServingContext(
            platform=self.platform, structure=self.structure,
            residency=self.residency, runtime_image=self.runtime_image,
            execution_mode=self.execution_mode,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id, "family": self.family,
            "rate_q256": self.rate_q256, "structure": self.structure,
            "platform": self.platform, "regime": self.regime,
            "residency": self.residency, "runtime_image": self.runtime_image,
            "execution_mode": self.execution_mode,
            "route_status": self.route_status,
            "requires_serve_flags": list(self.requires_serve_flags),
            "activation_contract": self.activation_contract,
            "evidence_grade": self.evidence_grade,
            "evidence_artifact_id": self.evidence_artifact_id,
            "smoke_status": self.smoke_status,
        }


_RESIDENCY_FLAG = "TESSERA_SERVE_MODE="


def _residencies_from_flags(flags: Sequence[str]) -> tuple[str, ...]:
    """The residency modes a cell's ``requires_serve_flags`` names.

    ``TESSERA_SERVE_MODE=resident|streamed`` is the contract's own spelling for
    a cell that serves under either, so the ``|`` split is the table's grammar
    and not a convenience here.
    """
    for flag in flags:
        if flag.startswith(_RESIDENCY_FLAG):
            value = flag[len(_RESIDENCY_FLAG):]
            return tuple(part for part in value.split("|") if part)
    return ()


def packaged_contract_payload() -> dict:
    """The installed Tessera's ``runtime_contract.json``, parsed.

    Read through ``tessera_runtime_contract.contract_path`` -- the one resolver
    both producer readers share -- so these bytes are the bytes every other
    gate on this side reads.
    """
    from importlib.resources import as_file

    from .tessera_runtime_contract import contract_path

    with as_file(contract_path()) as path:
        return json.loads(path.read_text(encoding="utf-8"))


def attested_cells(payload: "Mapping[str, object] | None" = None
                   ) -> tuple[AttestedCell, ...]:
    """Every native-qualification cell the pinned contract publishes.

    One row per ``(cell, rate, residency)`` the cell names, because a cell that
    serves under two residency modes is two distinct serving scopes and
    flattening them would lose the serve flag one of them needs.
    """
    data = packaged_contract_payload() if payload is None else payload
    lanes = data.get("lane_eligibility") or {}
    rows: list[AttestedCell] = []
    for cell in lanes.get("cells", ()):
        runtime = cell.get("runtime") or {}
        image = runtime.get("image")
        if not image:
            continue
        flags = tuple(cell.get("requires_serve_flags") or ())
        evidence = cell.get("evidence") or {}
        artifact = evidence.get("artifact")
        smoke = evidence.get("smoke") or {}
        for rate in cell.get("rungs_q256") or ():
            for residency in _residencies_from_flags(flags):
                for mode in runtime.get("execution_modes") or ():
                    rows.append(AttestedCell(
                        cell_id=str(cell.get("id")),
                        family=str(cell.get("family")),
                        rate_q256=int(rate),
                        structure=str(cell.get("structure")),
                        platform=str(cell.get("platform")),
                        regime=str(cell.get("regime")),
                        residency=str(residency),
                        runtime_image=str(image),
                        execution_mode=str(mode),
                        route_status=str(cell.get("route_status")),
                        requires_serve_flags=flags,
                        activation_contract=str(cell.get("activation_contract")),
                        evidence_grade=str(evidence.get("grade")),
                        evidence_artifact_id=(
                            None if not isinstance(artifact, Mapping)
                            else str(artifact.get("id"))
                        ),
                        smoke_status=str(smoke.get("status")),
                    ))
    return tuple(rows)


def native_qualification_set(payload: "Mapping[str, object] | None" = None
                             ) -> frozenset[tuple[str, int, str]]:
    """The ``(family, rate_q256, structure)`` triples a runtime qualifies.

    Confirmed against the seam rather than read off the JSON alone:
    ``tessera_menu.route_admission`` is asked with each cell's own derived
    serving context, and a triple enters only when that seam calls it attested.
    A cell whose scope the seam refuses is therefore absent here, which is the
    fail-closed direction.
    """
    from .tessera_menu import route_admission

    triples: set[tuple[str, int, str]] = set()
    for cell in attested_cells(payload):
        name = f"{cell.family}_R{cell.rate_q256}"
        try:
            admission = route_admission(
                name, serving_context=cell.serving_context())
        except Exception:
            continue
        if admission.attested:
            triples.add((cell.family, cell.rate_q256, cell.structure))
    return frozenset(triples)


# ---------------------------------------------------------------------------
# Bytes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ByteAccount:
    """Exact serialized bytes of one candidate at one shape, by component.

    Every field comes from ``tessera_footprint.tessera_tensor_payload_breakdown``
    -- the accountant the exporter's own bytes are priced through -- so a
    number here is a number the wire writes.  ``R/256`` is **not** the
    bits-per-parameter: at ``TESSERA_BF16_K1_R1792`` on a ``(2048, 4096)``
    Linear the body alone is 7.0 bits/weight and the unit still pays a
    32,768-byte window table and a 4,096-byte channel plane on top.

    The four component fields sum to :attr:`total_bytes` exactly; the test
    recomputes that sum from the components rather than reading the total.
    """

    family: str
    rate_q256: int
    rows: int
    columns: int
    body_kind: str
    scale_plane: str
    table_width_bits: int
    #: The schedule's body bits, already in bytes: the payload planes the
    #: Bresenham schedule writes for every weight.
    body_bytes: int
    #: The WINDOW body's inline table (``2**L`` grid codes), or the TCQ body's
    #: ALPHABET plane.
    table_bytes: int
    #: A TCQ body's DESCENDANT plane.  Zero on a WINDOW body, which has no
    #: forest.
    descendant_bytes: int
    #: The scale plane: one fp16 per output row on CHANNEL, a block plane
    #: otherwise.
    scale_bytes: int
    #: Padding and alignment the tight layout still charges, and any sidecar
    #: header.  Zero at ``layout="tight"`` with no sidecar; carried as its own
    #: field so it can never be silently folded into another component.
    alignment_and_header_bytes: int
    total_bytes: int
    exact_bits_per_weight: Fraction

    def components(self) -> dict[str, int]:
        return {
            "body_bytes": self.body_bytes,
            "table_bytes": self.table_bytes,
            "descendant_bytes": self.descendant_bytes,
            "scale_bytes": self.scale_bytes,
            "alignment_and_header_bytes": self.alignment_and_header_bytes,
        }

    def as_dict(self) -> dict[str, object]:
        payload = {
            "family": self.family, "rate_q256": self.rate_q256,
            "rows": self.rows, "columns": self.columns,
            "body_kind": self.body_kind, "scale_plane": self.scale_plane,
            "table_width_bits": self.table_width_bits,
            "total_bytes": self.total_bytes,
            "exact_bits_per_weight": str(self.exact_bits_per_weight),
        }
        payload.update(self.components())
        return payload


def byte_account(
    family: "str | TesseraFamily", rate_q256: int, shape: Sequence[int],
) -> ByteAccount:
    """Exact serialized bytes for one candidate at one shape, by component.

    Delegates to ``tessera_footprint.tessera_tensor_payload_breakdown`` and
    re-labels its planes; it computes no size of its own.  A second accountant
    is the defect PrismaQuant #126 was, so this one only reads.
    """
    from .tessera_footprint import tessera_tensor_payload_breakdown

    spec = get_tessera_family(family)
    rows, columns = int(shape[0]), int(shape[1])
    payload = tessera_tensor_payload_breakdown(
        (rows, columns), family=spec.name, body_rate_q256=int(rate_q256),
    )
    table = int(payload["alphabet_bytes"])
    descendant = int(payload["descendant_bytes"])
    header = int(payload["sidecar_header_bytes"])
    total = int(payload["total_bytes"])
    # The scale plane is the only remaining per-unit charge, and the body is
    # what is left.  Deriving the pair by subtraction rather than by a second
    # formula is deliberate: the total is the exporter's, and any component
    # this module invented could disagree with it.
    scale = _scale_plane_bytes(spec, payload, rows, columns)
    body = total - table - descendant - header - scale
    if body < 0:
        raise TesseraFormatError(
            f"{spec.name}_R{rate_q256} at {(rows, columns)}: component sum "
            f"{table + descendant + header + scale} exceeds the exporter's "
            f"total {total}; the plane labelling is wrong, not the total"
        )
    return ByteAccount(
        family=spec.name, rate_q256=int(rate_q256), rows=rows, columns=columns,
        body_kind=str(payload["body_kind"]),
        scale_plane=str(payload["scale_contract"]),
        table_width_bits=int(payload["window_bits"]),
        body_bytes=body, table_bytes=table, descendant_bytes=descendant,
        scale_bytes=scale, alignment_and_header_bytes=header,
        total_bytes=total,
        exact_bits_per_weight=Fraction(*payload["exact_bpw_rational"]),
    )


def _scale_plane_bytes(
    spec: TesseraFamily, payload: Mapping[str, object], rows: int, columns: int,
) -> int:
    """Bytes the resolved scale plane charges this unit.

    CHANNEL is one fp16 per output row (``tessera.layout._counts_for``).  The
    block planes charge per position and are already inside the body figure the
    payload reports, so they contribute nothing separable here and are zero
    rather than double-counted.
    """
    plane = str(payload["scale_contract"])
    if plane == "channel":
        return 2 * rows
    return 0


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

#: Every BF16 rung decodes to a plain BF16 tile, and none of them is a copy of
#: the source bytes.  The flag travels on the candidate so no consumer has to
#: infer it from ``terminal_format``.
FLAG_NOT_SOURCE_BF16_PASSTHROUGH = "NOT_SOURCE_BF16_PASSTHROUGH"
#: The rung sits at the top of its family's domain.  Carried separately from
#: the passthrough flag because the two say different things.
FLAG_DOMAIN_ENDPOINT = "DOMAIN_ENDPOINT"
#: The rung's table is wider than the family's 14-bit default.
FLAG_WIDENED_TABLE = "WIDENED_TABLE"


@dataclass(frozen=True, slots=True)
class RateCandidate:
    """One ``(family, rate)`` of the legal domain, with its support ledger."""

    family: str
    rate_q256: int
    format_name: str
    #: The body and scale plane the exporter writes at THIS rung, resolved per
    #: rung because the recipe may legally vary with it.
    body_kind: str
    scale_plane: str
    #: ``window_bits`` -- the inline table's width.  14 by default, widened by
    #: the producer to the schedule's rate where that is larger.
    table_width_bits: int
    #: What a decoded tile materialises as.  ``BF16`` here means "decodes to a
    #: plain BF16 tile", which is a statement about the *output* of the window
    #: body and is NOT source-byte passthrough.
    terminal_format: "str | None"
    #: The wire's own description: a WINDOW body over a CHANNEL scale plane
    #: with an inline ALPHABET table.
    representation: str
    #: Always False, and always present so nothing has to infer its absence.
    source_bf16_passthrough: bool
    #: Why it is False, in one sentence, for the report to quote.
    passthrough_note: str
    activation_contract: str
    flags: tuple[str, ...]
    support: SupportFacts
    #: The structure this ledger row was resolved for.  ``native_qualification``
    #: is a joint fact of the rate AND the structure, so a row without one
    #: would be answering a question nobody asked.
    structure: str

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family, "rate_q256": self.rate_q256,
            "format_name": self.format_name, "structure": self.structure,
            "body_kind": self.body_kind, "scale_plane": self.scale_plane,
            "table_width_bits": self.table_width_bits,
            "terminal_format": self.terminal_format,
            "representation": self.representation,
            "source_bf16_passthrough": self.source_bf16_passthrough,
            "passthrough_note": self.passthrough_note,
            "activation_contract": self.activation_contract,
            "flags": list(self.flags),
            "support": self.support.as_dict(),
        }


def table_width_bits(family: "str | TesseraFamily", rate_q256: int) -> int:
    """The inline table's width at this rung, from the producer's own recipe.

    No lookup table of transition points: ``tessera.export.wire_recipe`` is
    asked per rung and reports the width it will hand the encoder.  The BF16
    transitions at 3585 and 3841 are therefore *observed* here, not restated --
    which is what makes the boundary-witness tests confirm a derivation.
    """
    spec = get_tessera_family(family)
    return int(tessera_wire_recipe(spec, int(rate_q256)).window_bits)


def table_width_transitions(family: "str | TesseraFamily") -> tuple[tuple[int, int], ...]:
    """Every rate at which :func:`table_width_bits` changes, and its new width.

    The first entry is always the domain's lower endpoint with its width, so
    the sequence describes the whole domain rather than only its steps.
    """
    spec = get_tessera_family(family)
    lo, hi = family_q256_bounds(spec)
    transitions: list[tuple[int, int]] = []
    previous: "int | None" = None
    for rate in range(lo, hi + 1):
        width = table_width_bits(spec, rate)
        if width != previous:
            transitions.append((rate, width))
            previous = width
    return tuple(transitions)


def _body_kind_name(recipe) -> str:
    """``"window"`` or ``"tcq"``, through Tessera's own enum.

    ``WireRecipe.body`` is not guaranteed to arrive as an enum member -- the
    export seam builds recipes from raw config values -- so it is normalised
    the way every other reader in the tree does it, through ``BodyKind(...)``.
    Formatting the attribute with ``str()`` reads ``2`` on a plain int and
    produced a "TCQ body over a 2 scale plane" label for a WINDOW rung.
    """
    from tessera.manifest import BodyKind

    return BodyKind(recipe.body).name.lower()


def schedule_signature(
    family: "str | TesseraFamily",
    rate_q256: int,
    shapes: Sequence[Sequence[int]] = GLM53_LINEAR_SHAPES,
) -> tuple:
    """Everything the canonical resolver decides about one rung.

    The wire recipe -- body, span, scale plane, table width -- **and** the set
    of distinct per-column rates Tessera's own Bresenham scheduler lays down at
    each shape.  The second half is what makes a 256-multiple a transition: at
    ``R = 256k`` over a column count divisible by 256 the schedule is uniform
    at rate ``k``, and at ``R = 256k + 1`` it becomes the mixed ``{k, k+1}``
    walk.  That is a different schedule, not a rounding of the same one, and a
    screen that treated the two as one regime would be interpolating across it.
    """
    spec = get_tessera_family(family)
    rate = int(rate_q256)
    recipe = tessera_wire_recipe(spec, rate)
    per_shape = []
    for shape in shapes:
        columns = int(shape[1])
        per_shape.append((
            columns,
            tuple(sorted(set(spec.column_schedule(rate, columns, recipe=recipe)))),
        ))
    return (
        _body_kind_name(recipe), int(recipe.span),
        scale_plane_name(recipe.scale_plane),
        int(recipe.window_bits), tuple(per_shape),
    )


def resolver_transitions(
    family: "str | TesseraFamily",
    rates: "Sequence[int] | None" = None,
    shapes: Sequence[Sequence[int]] = GLM53_LINEAR_SHAPES,
) -> tuple[int, ...]:
    """Every legal rate at which the resolver's answer changes.

    **A transition names the FIRST rate of the NEW regime.**  ``ceil(3585/256)``
    is 15 and ``3584/256`` is exactly 14, so the width changes at 3585 and 3584
    still belongs to the old regime; the same reading makes ``256k`` -- where
    the schedule becomes uniform -- the transition rather than ``256k - 1``.

    The domain's first rate is not a transition: nothing precedes it.  Section
    5.3's mandatory set gets it from the domain ends instead.

    At the GLM widths this returns every 256-multiple above the lower endpoint
    and every ``256k + 1``, and the two table-width transitions are a subset of
    the second group (3585 = 14*256+1, 3841 = 15*256+1).
    """
    spec = get_tessera_family(family)
    walk = tuple(rates) if rates is not None else legal_rates(spec, shapes)[0]
    transitions: list[int] = []
    previous = None
    for rate in walk:
        signature = schedule_signature(spec, rate, shapes)
        if previous is not None and signature != previous:
            transitions.append(rate)
        previous = signature
    return tuple(transitions)


#: The narrow object work package A hands work package C — one class, defined
#: in C (:class:`prismaquant.quality_prefill_population.RateDomain`) and
#: imported here. A owns the derivation below; C owns the type, because C has
#: no Tessera dependency and this module has one at import time, so a shared
#: definition can only live on C's side. Its refusals — empty, unsorted or
#: duplicated rates, a transition outside the domain, a non-tuple field — are
#: C's and raise ``PopulationSelectionError``; this module previously raised
#: :class:`~prismaquant.tessera_formats.TesseraFormatError` for the first
#: three, and nothing in the tree guarded those constructions with it.
RateDomain = _RateDomain


def legal_rate_domain(
    family: str,
    shapes: Sequence[Sequence[int]] = GLM53_LINEAR_SHAPES,
) -> RateDomain:
    """The provider work package C consumes: family name in, domain out.

    ``Callable[[str], RateDomain]``.  Bind ``shapes`` with ``functools.partial``
    to ask about a different model's widths; the default is GLM-5.3-Flash.
    """
    spec = get_tessera_family(family)
    rates, holes = legal_rates(spec, shapes)
    if holes:
        raise TesseraFormatError(
            f"{spec.name}: {len(holes)} rate(s) inside the producer's own "
            f"bounds are refused at these shapes ({sorted(holes)[:5]}); the "
            "domain is reported with the holes, never silently clamped"
        )
    return RateDomain(
        family=spec.name, rates=rates,
        transition_rates=resolver_transitions(spec, rates, shapes),
    )


def rate_domain_payload(
    family: str,
    shapes: Sequence[Sequence[int]] = GLM53_LINEAR_SHAPES,
) -> dict[str, object]:
    """:func:`legal_rate_domain` as plain data, for ``RateDomain(**payload)``."""
    domain = legal_rate_domain(family, shapes)
    return {
        "family": domain.family, "rates": domain.rates,
        "transition_rates": domain.transition_rates,
    }


def legal_rates(
    family: "str | TesseraFamily",
    shapes: Sequence[Sequence[int]] = GLM53_LINEAR_SHAPES,
) -> tuple[tuple[int, ...], dict[int, tuple[str, ...]]]:
    """``(rates legal at EVERY shape, {rate: refusals})``.

    The domain is the producer's ``family_q256_bounds`` walked against
    ``tessera_menu.tessera_shape_legal`` at each shape.  A rate that any shape
    refuses is excluded from the first value and its reasons are recorded in
    the second -- a *hole*, which the spec asks to be reported rather than
    smoothed.  At the GLM widths there are none, because every column count is
    divisible by 256 and the arity is one, so the exact-quota grammar closes on
    every integer q256.
    """
    from .tessera_menu import tessera_shape_legal

    spec = get_tessera_family(family)
    lo, hi = family_q256_bounds(spec)
    legal: list[int] = []
    holes: dict[int, tuple[str, ...]] = {}
    for rate in range(lo, hi + 1):
        refusals = []
        for shape in shapes:
            ok, reason = tessera_shape_legal(spec, rate, tuple(shape))
            if not ok:
                refusals.append(f"{tuple(shape)}: {reason}")
        if refusals:
            holes[rate] = tuple(refusals)
        else:
            legal.append(rate)
    return tuple(legal), holes


def _reader_range(family: str) -> "tuple[int, int] | None":
    """The pinned contract's declared reader range for a family, or None."""
    payload = packaged_contract_payload()
    for row in payload.get("formats") or ():
        if row.get("family") == family:
            span = row.get("reader_rate_range_q256")
            if span:
                return (int(span[0]), int(span[1]))
    return None


def support_facts(
    family: "str | TesseraFamily",
    rate_q256: int,
    structure: str,
    *,
    shapes: Sequence[Sequence[int]] = GLM53_LINEAR_SHAPES,
    native_set: "frozenset[tuple[str, int, str]] | None" = None,
    cells: "Sequence[AttestedCell] | None" = None,
    artifact_scope: str = ARTIFACT_SCOPE,
) -> SupportFacts:
    """The five facts for one candidate, each from the table that answers it.

    Five reads of four different tables, never one read reused.  In particular
    ``reader_supported`` is the contract's ``reader_rate_range_q256`` and says
    nothing about attestation; ``native_qualification`` is exact cell
    membership and says nothing about the reader.
    """
    from .tessera_formats import tessera_serving_route
    from .tessera_menu import tessera_shape_legal

    spec = get_tessera_family(family)
    rate = int(rate_q256)
    name = f"{spec.name}_R{rate}"
    pins = live_pins()

    # 1. producer legal --------------------------------------------------
    lo, hi = family_q256_bounds(spec)
    in_bounds = lo <= rate <= hi
    refusals = []
    if in_bounds:
        for shape in shapes:
            ok, reason = tessera_shape_legal(spec, rate, tuple(shape))
            if not ok:
                refusals.append(f"{tuple(shape)}: {reason}")
    producer = SupportFact(
        name=FACT_PRODUCER_LEGAL,
        value=bool(in_bounds and not refusals),
        source=(
            "tessera.grammar via tessera_formats.family_q256_bounds + "
            f"tessera_menu.tessera_shape_legal @ contract "
            f"{pins.producer_installed_contract_sha256[:12]}"
        ),
        detail=(
            f"domain [{lo}, {hi}]" if in_bounds and not refusals
            else f"outside [{lo}, {hi}]" if not in_bounds
            else "; ".join(refusals)
        ),
    )

    # 2. reader supported ------------------------------------------------
    span = _reader_range(spec.name)
    reader = SupportFact(
        name=FACT_READER_SUPPORTED,
        value=bool(span is not None and span[0] <= rate <= span[1]),
        source=(
            "tessera runtime_contract.json formats[].reader_rate_range_q256 @ "
            f"{pins.serving_runtime_pinned_contract_sha256[:12]}"
        ),
        detail=(
            "the pinned contract publishes no reader range for this family"
            if span is None else f"reader range [{span[0]}, {span[1]}]"
        ),
    )

    # 3. implemented route ------------------------------------------------
    try:
        route = tessera_serving_route(spec, rung=rate)
        has_route = route is not None and route.materialises
        route_detail = (
            f"{route.contract} -> {route.terminal_format}"
            if has_route else
            f"{route.contract} does not materialise into a stock format; "
            "no implemented execution route"
        )
    except TesseraFormatError as exc:
        has_route, route_detail = False, str(exc)
    implemented = SupportFact(
        name=FACT_IMPLEMENTED_ROUTE,
        value=bool(has_route),
        source="tessera_formats.tessera_serving_route",
        detail=route_detail,
    )

    # 4. native qualification ---------------------------------------------
    triples = native_qualification_set() if native_set is None else native_set
    qualified = (spec.name, rate, structure) in triples
    native = SupportFact(
        name=FACT_NATIVE_QUALIFICATION,
        value=bool(qualified),
        source=(
            "tessera_menu.route_admission against runtime_contract.json "
            f"lane_eligibility.cells @ {pins.reader_dev_pin_commit[:12]}"
        ),
        detail=(
            f"a device-qualified cell covers {spec.name} R{rate} on {structure}"
            if qualified else
            f"no device-qualified cell covers {spec.name} R{rate} on "
            f"{structure} at this pin; attestation is exact membership and "
            "does not extrapolate to a neighbouring rate or structure"
        ),
    )

    # 5. export and served validation --------------------------------------
    rows = attested_cells() if cells is None else cells
    matching = [
        cell for cell in rows
        if cell.family == spec.name and cell.rate_q256 == rate
        and cell.structure == structure
    ]
    with_artifact = [cell for cell in matching if cell.evidence_artifact_id]
    in_scope = [
        cell for cell in with_artifact
        if artifact_scope.lower() in str(cell.evidence_artifact_id).lower()
    ]
    out_of_scope = sorted({
        str(cell.evidence_artifact_id) for cell in with_artifact
    } - {str(cell.evidence_artifact_id) for cell in in_scope})
    if in_scope:
        detail = (
            "artifacts " + ", ".join(sorted({
                str(c.evidence_artifact_id) for c in in_scope
            })) + "; grades " + ", ".join(sorted({
                c.evidence_grade for c in in_scope
            }))
        )
    elif out_of_scope:
        detail = (
            f"the pinned contract records exported artifact(s) "
            + ", ".join(out_of_scope)
            + f" for this cell, but none of them is a {artifact_scope} "
            "artifact; a capability claim inherits the scope of the artifact "
            "it was measured on, so this is recorded and not counted"
        )
    else:
        detail = (
            "no cell at this pin records an exported artifact for this "
            "candidate, so no complete assignment has passed export plus "
            "served validation"
        )
    validated = SupportFact(
        name=FACT_EXPORT_AND_SERVED_VALIDATION,
        value=bool(in_scope),
        source=(
            "tessera runtime_contract.json lane_eligibility.cells[].evidence @ "
            f"{pins.serving_runtime_pinned_contract_sha256[:12]} "
            f"scoped to {artifact_scope}"
        ),
        detail=detail,
    )

    return SupportFacts(
        producer_legal=producer, reader_supported=reader,
        implemented_route=implemented, native_qualification=native,
        export_and_served_validation=validated,
    )


def _candidate(
    spec: TesseraFamily, rate: int, structure: str, facts: SupportFacts,
) -> RateCandidate:
    from .tessera_formats import tessera_serving_route

    recipe = tessera_wire_recipe(spec, rate)
    route = tessera_serving_route(spec, rung=rate)
    lo, hi = family_q256_bounds(spec)
    width = int(recipe.window_bits)
    body = _body_kind_name(recipe)
    plane = scale_plane_name(recipe.scale_plane)
    terminal = route.terminal_format if route is not None else None

    flags: list[str] = []
    if terminal == "BF16":
        flags.append(FLAG_NOT_SOURCE_BF16_PASSTHROUGH)
    if rate == hi:
        flags.append(FLAG_DOMAIN_ENDPOINT)
    if width > 14:
        flags.append(FLAG_WIDENED_TABLE)

    return RateCandidate(
        family=spec.name, rate_q256=rate,
        format_name=f"{spec.name}_R{rate}", structure=structure,
        body_kind=body, scale_plane=plane, table_width_bits=width,
        terminal_format=terminal,
        representation=f"{body.upper()} body over a {plane.upper()} scale plane",
        source_bf16_passthrough=False,
        passthrough_note=(
            "a WINDOW body over a CHANNEL scale plane with an inline "
            f"{1 << width}-entry ALPHABET table; it decodes TO a "
            f"{terminal} tile and is not a copy of the source bytes. Its "
            "error, its artifact bytes including that table, and its "
            "throughput against source BF16 are all unmeasured."
        ),
        activation_contract=(route.contract if route is not None else ""),
        flags=tuple(flags), support=facts,
    )


# ---------------------------------------------------------------------------
# The inventory
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FamilyInventory:
    """The complete legal rate domain of one family, plus its ledger."""

    family: str
    rate_lo: int
    rate_hi: int
    rate_count: int
    rates: tuple[int, ...]
    holes: dict[int, tuple[str, ...]]
    table_width_transitions: tuple[tuple[int, int], ...]
    #: Every legal rate at which the canonical resolver's answer changes --
    #: the value work package C reads as ``RateDomain.transition_rates``.
    resolver_transitions: tuple[int, ...]
    candidates_by_structure: dict[str, tuple[RateCandidate, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "rate_lo": self.rate_lo, "rate_hi": self.rate_hi,
            "rate_count": self.rate_count,
            "holes": {k: list(v) for k, v in self.holes.items()},
            "table_width_transitions": [
                list(t) for t in self.table_width_transitions
            ],
            "resolver_transitions": list(self.resolver_transitions),
            "structures": sorted(self.candidates_by_structure),
        }


#: The unit structures the contract itself names.  Read from the table rather
#: than typed, so a new structure joins the ledger by joining the contract.
def contract_structures() -> tuple[str, ...]:
    payload = packaged_contract_payload()
    lanes = payload.get("lane_eligibility") or {}
    return tuple(sorted(lanes.get("structures") or ()))


def build_inventory(
    families: Sequence[str] = PRIMARY_FAMILIES,
    *,
    shapes: Sequence[Sequence[int]] = GLM53_LINEAR_SHAPES,
    structures: "Sequence[str] | None" = None,
    ledger_rates: "Sequence[int] | None" = None,
    artifact_scope: str = ARTIFACT_SCOPE,
) -> dict[str, object]:
    """The complete legal domain and its five-part support ledger.

    The domain is computed first and without consulting the attestation, so a
    singleton native set cannot shrink it.  The ledger is then attached; by
    default only the boundary witnesses and the natively qualified rates carry
    full :class:`RateCandidate` rows, because the five facts cost a contract
    read each and 5,634 of them is a campaign, not an inventory.  Pass
    ``ledger_rates`` to widen it.

    The return value is a plain dict of frozen dataclasses.  **No manifest
    schema is defined here** -- section 12 gives that to another package, which
    is expected to wrap this.
    """
    pins = live_pins()
    drift = pin_drift(live=pins)
    structures = tuple(structures or contract_structures())
    native_set = native_qualification_set()
    cells = attested_cells()

    inventories: dict[str, FamilyInventory] = {}
    for family in families:
        spec = get_tessera_family(family)
        rates, holes = legal_rates(spec, shapes)
        transitions = table_width_transitions(spec)
        steps = resolver_transitions(spec, rates, shapes)
        witnesses = boundary_witnesses(spec, rates, transitions)
        wanted = sorted(set(witnesses) | {
            rate for (fam, rate, _s) in native_set if fam == spec.name
        } | set(ledger_rates or ()))
        by_structure: dict[str, tuple[RateCandidate, ...]] = {}
        for structure in structures:
            rows = []
            for rate in wanted:
                if rate not in set(rates):
                    continue
                facts = support_facts(
                    spec, rate, structure, shapes=shapes,
                    native_set=native_set, cells=cells,
                    artifact_scope=artifact_scope,
                )
                rows.append(_candidate(spec, rate, structure, facts))
            by_structure[structure] = tuple(rows)
        lo, hi = family_q256_bounds(spec)
        inventories[spec.name] = FamilyInventory(
            family=spec.name, rate_lo=lo, rate_hi=hi,
            rate_count=len(rates), rates=rates, holes=holes,
            table_width_transitions=transitions,
            resolver_transitions=steps,
            candidates_by_structure=by_structure,
        )

    return {
        "schema": SCHEMA,
        "pins": pins.as_dict(),
        "pin_drift": drift,
        "tessera_source_state": tessera_source_state(),
        "shapes": [list(s) for s in shapes],
        "artifact_scope": artifact_scope,
        "structures": list(structures),
        "families": inventories,
        "native_qualification_set": sorted(native_set),
        "attested_cells": [cell.as_dict() for cell in cells],
        "audit_oracle": audit_oracle_report(inventories),
    }


def boundary_witnesses(
    family: "str | TesseraFamily",
    rates: "Sequence[int] | None" = None,
    transitions: "Sequence[tuple[int, int]] | None" = None,
) -> tuple[int, ...]:
    """Domain ends, every table-width transition, and its immediate neighbour.

    Section 5.3's mandatory set for this axis: the two endpoints, each rate at
    which the recipe changes, and the last legal rate below it, so a test can
    witness a transition from both sides.
    """
    spec = get_tessera_family(family)
    lo, hi = family_q256_bounds(spec)
    ordered = tuple(rates) if rates is not None else tuple(range(lo, hi + 1))
    legal = set(ordered)
    steps = [rate for rate, _width in (
        transitions if transitions is not None
        else table_width_transitions(spec)
    )]
    witnesses = {ordered[0], ordered[-1]}
    index = {rate: position for position, rate in enumerate(ordered)}
    for rate in steps:
        if rate not in legal:
            continue
        witnesses.add(rate)
        position = index[rate]
        if position:
            # The previous LEGAL rate, not ``rate - 1``: a rate one integer
            # away need not be legal, and package C reads the neighbour the
            # same way.
            witnesses.add(ordered[position - 1])
    return tuple(sorted(witnesses))


def audit_oracle_report(
    inventories: Mapping[str, FamilyInventory],
) -> dict[str, object]:
    """Compare the derived counts against the frozen audit's, and say so.

    This never adjusts a count.  A disagreement is reported with both numbers
    and the pin that produced the derived one, which is the finding the spec
    asks for.
    """
    rows = {}
    for family, expected in AUDITED_RATE_COUNTS.items():
        inventory = inventories.get(family)
        if inventory is None:
            continue
        rows[family] = {
            "derived": inventory.rate_count,
            "audited": expected,
            "agrees": inventory.rate_count == expected,
            "holes": len(inventory.holes),
        }
    return {
        "source": "tessera-legal-domain-source-audit-01.md",
        "source_sha256": (
            "b2ec2c6be76c6c0c557ffa642a9290e26f960b9781fb0e843c9d101fbc2c88b2"
        ),
        "families": rows,
        "agrees": all(row["agrees"] for row in rows.values()) if rows else False,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(inventory: Mapping[str, object]) -> str:
    """The human-readable inventory: pins and drift FIRST, then counts."""
    lines: list[str] = []
    drift = inventory["pin_drift"]
    lines.append("Tessera legal rate domain -- GLM Linear shapes")
    lines.append("")
    lines.append("Pins the code reads at this execution:")
    for name, value in inventory["pins"].items():
        lines.append(f"  {name:42s} {value}")
    lines.append(f"  {'spec_named_study_producer (documentary)':42s} "
                 f"{SPEC_NAMED_STUDY_PRODUCER}")
    lines.append("")
    lines.append(f"Pin drift: {drift['verdict']}")
    for name, pair in sorted(drift["differences"].items()):
        lines.append(f"    {name}: frozen={pair['frozen']} live={pair['live']}")
    lines.append("")
    state = inventory["tessera_source_state"]
    lines.append(f"Tessera source state: {state['verdict']}")
    lines.append(f"  {'export.py':42s} {state['export_sha256']}")
    lines.append(f"  {'export.py path':42s} {state['export_path']}")
    lines.append(
        f"  {'grammar.py shared by all three states':42s} "
        f"{state['grammar_matches_every_state']}"
    )
    lines.append("")
    lines.append(f"Shapes (rows, columns): {inventory['shapes']}")
    lines.append("")
    lines.append("Legal rate domain:")
    for family, entry in inventory["families"].items():
        lines.append(
            f"  {family:20s} R{entry.rate_lo}-R{entry.rate_hi}  "
            f"count {entry.rate_count}  holes {len(entry.holes)}"
        )
        widths = ", ".join(f"R{r}->L{w}" for r, w in entry.table_width_transitions)
        lines.append(f"  {'':20s} table width: {widths}")
        steps = entry.resolver_transitions
        lines.append(
            f"  {'':20s} resolver transitions: {len(steps)} "
            f"({sum(1 for r in steps if r % 256 == 0)} at a 256-multiple, "
            f"{sum(1 for r in steps if r % 256 == 1)} at 256k+1)"
        )
        for rate, reasons in sorted(entry.holes.items()):
            lines.append(f"  {'':20s} HOLE R{rate}: {reasons[0]}")
    lines.append("")
    oracle = inventory["audit_oracle"]
    lines.append(f"Audit oracle ({oracle['source']}):")
    for family, row in sorted(oracle["families"].items()):
        verdict = "agrees" if row["agrees"] else "DISAGREES"
        lines.append(
            f"  {family:20s} derived {row['derived']} vs audited "
            f"{row['audited']} -- {verdict}"
        )
    lines.append("")
    lines.append("Native qualification (exact cell membership, not a range):")
    for family, rate, structure in inventory["native_qualification_set"]:
        lines.append(f"  {family}_R{rate} on {structure}")
    lines.append(
        f"  total {len(inventory['native_qualification_set'])} "
        "(family, rate, structure) triples"
    )
    lines.append("")
    lines.append("Support ledger at the boundary witnesses:")
    header = (
        f"  {'candidate':30s} {'struct':11s} {'L':>3s}  "
        + "  ".join(f"{n[:12]:12s}" for n in SUPPORT_FACT_NAMES)
    )
    lines.append(header)
    for family, entry in inventory["families"].items():
        for structure, rows in sorted(entry.candidates_by_structure.items()):
            for candidate in rows:
                marks = "  ".join(
                    f"{('yes' if f.value else 'no'):12s}"
                    for f in candidate.support.facts()
                )
                flags = ("  " + ",".join(candidate.flags)) if candidate.flags else ""
                lines.append(
                    f"  {candidate.format_name:30s} {structure:11s} "
                    f"{candidate.table_width_bits:3d}  {marks}{flags}"
                )
    return "\n".join(lines)


def main(argv: "Sequence[str] | None" = None) -> int:
    inventory = build_inventory()
    print(format_report(inventory))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
