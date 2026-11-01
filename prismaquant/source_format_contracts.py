"""Native source storage, footprint, wire and measured serving contracts."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from . import format_registry as fr
from .footprint import plain_source_dtype_tensor_payload_breakdown


ROUTE_DELEGATED_NATIVE = "delegated_native"
# The name of the route that USED to execute a raw-resident block-128 source
# plane: Gridbook's ``Fp8SourceW8A16LinearMethod``, which consumed BF16
# activations unchanged.  The lane was retired on 2026-09-02
# (archive/gridbook_lane_2026-09-02/) and this constant is kept BECAUSE it is
# retired: it is the ``serving_route`` on the FP8_BLOCK_UE8M0_SOURCE row below,
# whose ``route_status`` is now BLOCKED, and that row's whole job is to say
# which route died. Renaming it to something lane-neutral would erase the scope
# of the measurement the row carries (principle 14's corollary: a recorded
# capability claim inherits the scope of the artifact it was measured on).
#
# ``ROUTE_GRIDBOOK_MXFP8_DENSE = "gridbook_mxfp8_dense"`` sat here until
# 2026-09-02.  It named ``Mxfp8DenseLinearMethod`` and had no reader anywhere
# in the tree -- no contract row, no test, no consumer of the string -- so it
# was a dangling name rather than a record of anything, and it is deleted
# rather than archived.
ROUTE_GRIDBOOK_FP8_SOURCE_W8A16 = "gridbook_fp8_source_w8a16"

# What a MEASUREMENT says about serving a passthrough's bytes. These are
# verdicts from a real serve attempt on real hardware, not design intent —
# which matters, because on DSv4-Flash/GB10 the measured answers came out the
# OPPOSITE way round from the obvious guess.
ROUTE_STATUS_BACKED = "backed"          # measured serving, possibly with a requirement
ROUTE_STATUS_PENDING = "pending"        # no verdict yet; unaudited
ROUTE_STATUS_BLOCKED = "blocked"        # measured, every known route dead


@dataclass(frozen=True)
class SourcePassthroughContract:
    """One native source format the producer can ship back UNCHANGED.

    Keeping a native source format always preserves its stored weight plane;
    it is an allocator option only when the serving activation contract is
    also priced honestly. This table records the source/storage contract, one
    entry per
    (source format, unit contract) the census finds, so adding a newly
    encountered native format is a data change rather than a new code path.

    Fields:
      ``source_kind``  the token ``_scan_source_dtype_manifest`` stamps on a
        unit whose stored bytes ARE this format. It is the whole legality
        gate: the format is legal exactly where the source already is it, in
        both directions — BF16 is masked on mxfp4 experts and MXFP4_SOURCE is
        masked on the bf16 embedding by the identical rule.
      ``zero_cost_by_construction`` whether the allocator SYNTHESIZES the
        candidate rather than requiring a cost-table column. True for formats
        no cost run will ever have a column for.
      ``serving_route``  the P5b lane route id.
      ``wire_format_id``  the closed-enum spelling the artifact declares and
        the serving side reads (quant_config.json ``source_passthrough``).
        Distinct from ``format_name``: the producer's registry name is ours to
        rename, the wire id is a cross-repo contract.
      ``route_status``  what a MEASUREMENT says about serving these bytes, per
        ``ROUTE_STATUS_*``. Route status alone never removes an honestly
        priced rung from the menu — an allocator that wants an unservable
        passthrough is reporting a serving gap — but anything other than
        ``backed`` makes export fail closed without an explicit override.
        This is orthogonal to cost admission: a terminal whose activation-side
        loss has no currency must stay out of that allocator campaign even if
        the serving route itself exists.
      ``route_requirement``  the serving-side condition that makes a BACKED
        route actually fire (e.g. a non-default vLLM MoE backend). Belongs in
        the artifact's serving notes; a backed route with an unmet
        requirement serves no better than a blocked one.
      ``route_evidence``  what was measured, and on what hardware. A route
        verdict without its evidence is a rumour.
      ``detail``  why this entry exists, for the mask/provenance records.
    """

    format_name: str
    source_kind: str
    zero_cost_by_construction: bool
    serving_route: str
    route_status: str
    detail: str
    wire_format_id: str | None = None
    route_requirement: str | None = None
    route_evidence: str | None = None

    @property
    def route_backed(self) -> bool:
        """Whether a serve route for these bytes is known to exist."""
        return self.route_status == ROUTE_STATUS_BACKED


SOURCE_PASSTHROUGH_CONTRACTS: dict[str, SourcePassthroughContract] = {
    contract.format_name: contract
    for contract in (
        # Legacy DeepSeek-V3 / MiniMax block-FP8: FP32 ``weight_scale_inv``.
        # Serves through stock compressed-tensors block-fp8 today.
        SourcePassthroughContract(
            format_name="FP8_SOURCE",
            source_kind="fp8",
            zero_cost_by_construction=False,
            serving_route=f"{ROUTE_DELEGATED_NATIVE}_fp8_block",
            route_status=ROUTE_STATUS_BACKED,
            detail=(
                "native block-FP8 with an FP32 weight_scale_inv plane; the "
                "cost pipeline emits real rows for it, so its candidate is "
                "not synthesized"
            ),
            route_evidence=(
                "stock compressed-tensors block-fp8; served on the "
                "checkpoints this format was written for (MiniMax M2, "
                "DeepSeek-V3)"
            ),
        ),
        # DeepSeek-V3.1/V4 block-FP8: one-byte UE8M0 block exponents. Same
        # element grid as FP8_SOURCE, different scale plane and different
        # byte count — see the FormatSpec comment.
        SourcePassthroughContract(
            format_name="FP8_BLOCK_UE8M0_SOURCE",
            source_kind="fp8_ue8m0",
            wire_format_id="fp8_e4m3_ue8m0_block128",
            zero_cost_by_construction=True,
            serving_route=ROUTE_GRIDBOOK_FP8_SOURCE_W8A16,
            # BLOCKED since 2026-09-02, and by a measurement rather than by an
            # opinion: the only route that ever executed these bytes was the
            # Gridbook plugin's Fp8SourceW8A16LinearMethod, and that lane was
            # retired (archive/gridbook_lane_2026-09-02/). No sanctioned
            # runtime -- vanilla vLLM, llama.cpp/vLLM-GGUF, the Tessera plugin
            # -- reads a block-128 UE8M0 source plane. The rung stays PRICED
            # (principle 1: an allocator that wants it is reporting a serving
            # gap, and that signal is the point) and the exporter now refuses
            # a selection containing it without an explicit override, via
            # ROUTE_PENDING_PASSTHROUGH_FORMATS. The evidence below is kept
            # verbatim because it is a real measurement whose SCOPE is the
            # retired runtime -- it says what was true, not what is.
            route_status=ROUTE_STATUS_BLOCKED,
            route_requirement=(
                "a Gridbook release attesting "
                "abi_features.source_fp8_block128_w8a16=1; first released in "
                "0.8.5 commit e992e5980c96333a48149f96392d6cff56ae9e3f under "
                "runtime-contract v3, carried unchanged by 0.8.11 under v4, "
                "and carried unchanged again by the currently pinned 0.9.1 "
                "commit 227420f9821bab7089632ee914f0ba050f82b817 under v12 "
                "(the two packaged contracts' abi_features maps are equal; "
                "v12 changes formats and adds lane_eligibility, not this)"
            ),
            detail=(
                "block-FP8 with UE8M0 block exponents, consumed by "
                "Gridbook's dedicated Fp8SourceW8A16LinearMethod. Both "
                "source planes stay resident and byte-verbatim and BF16 "
                "activations are unchanged. The numerical terminal is exact "
                "by construction. Full-artifact served parity remains a "
                "separate ship gate, not an exporter route override."
            ),
            route_evidence=(
                "measured on Gridbook 0.8.5 commit "
                "e992e5980c96333a48149f96392d6cff56ae9e3f; "
                "installed-wheel GPU gate on GB10/sm121: 91 passed, 0 skipped. "
                "Fp8SourceW8A16LinearMethod keeps raw E4M3/UE8M0 planes "
                "resident, dispatches decode to native GEMV and prefill to "
                "the owned grouped BF16 CUTLASS bridge, and attests the exact "
                "JIT extension identity/capability. Carried forward to the "
                "pinned 0.9.1 on the packaged contract's unchanged "
                "source_fp8_block128_w8a16 attestation; the GPU gate has not "
                "been re-run on 0.8.11 or on 0.9.1, so the measurement's scope "
                "is still 0.8.5 and this row says so rather than inheriting a "
                "claim it did not make."
            ),
        ),
        # DeepSeek-V4 routed experts: nibble-packed E2M1 + E8M0 group scales.
        SourcePassthroughContract(
            format_name="MXFP4_SOURCE",
            source_kind="mxfp4",
            wire_format_id="mxfp4_e2m1_ue8m0_g32",
            zero_cost_by_construction=True,
            serving_route=f"{ROUTE_DELEGATED_NATIVE}_mxfp4",
            route_status=ROUTE_STATUS_BACKED,
            route_requirement="vllm --moe-backend marlin",
            detail=(
                "native packed MXFP4 routed experts, served by the model's "
                "own path rather than a codebook decoder. BACKED, but only "
                "off the default: the auto-selected backend is broken on our "
                "target, so the requirement below is part of the contract, "
                "not a tuning hint."
            ),
            route_evidence=(
                "sm121 measured 2026-08-03: native-confirmed via vLLM Marlin "
                "MoE (--moe-backend marlin). The AUTO default DeepGEMM_MXFP4 "
                "asserts on the SF transformation; FlashInfer is gated to "
                "capability family 100; the OAI Triton path is hard-excluded "
                "on SM12x (0/15 kernels)."
            ),
        ),
        # Unquantized passthrough. Not a "source format" in the census sense,
        # but it obeys the same rule and predates this table.
        SourcePassthroughContract(
            format_name="BF16",
            source_kind="bf16",
            zero_cost_by_construction=False,
            serving_route=f"{ROUTE_DELEGATED_NATIVE}_bf16",
            route_status=ROUTE_STATUS_BACKED,
            detail="unquantized passthrough on a bf16 source",
            route_evidence="plain container floats; no kernel required",
        ),
    )
}

# Kept as the flat {format: required_kind} view every existing consumer
# (allocator promotion legality, export defensive checks, the visual
# passthrough contract) already imports. DERIVED from the table above so a new
# census format cannot be legal in one place and unknown in the other.
PASSTHROUGH_SOURCE_REQUIREMENTS: dict[str, str] = {
    name: contract.source_kind
    for name, contract in SOURCE_PASSTHROUGH_CONTRACTS.items()
}

# An allocator candidate may never carry more serialized tensor payload than
# the representation already stored for that unit.  Source kinds are resolved
# through the passthrough contracts rather than through a numeric bpp table:
# the registered source FormatSpec owns the exact weight/scale footprint.
SOURCE_BPP_EXCEEDED_REASON = "candidate_exceeds_source_bpp"
SOURCE_BPP_UNKNOWN_REASON = "source_bpp_unknown"
SOURCE_BPP_POLICY_SCHEMA = "prismaquant.source_bpp_legality.v1"

_source_kinds = [
    contract.source_kind
    for contract in SOURCE_PASSTHROUGH_CONTRACTS.values()
]
if len(set(_source_kinds)) != len(_source_kinds):
    duplicates = sorted(
        kind for kind, count in Counter(_source_kinds).items() if count > 1
    )
    raise RuntimeError(
        "SOURCE_PASSTHROUGH_CONTRACTS has ambiguous source footprint "
        f"owners for source kinds {duplicates}"
    )
SOURCE_FORMAT_BY_KIND: dict[str, str] = {
    contract.source_kind: name
    for name, contract in SOURCE_PASSTHROUGH_CONTRACTS.items()
}


@dataclass(frozen=True)
class SourceFootprintOwner:
    """Exact source-payload authority selected by a census kind."""

    source_kind: str
    format_name: str | None = None
    safetensors_dtype: str | None = None


def source_footprint_owner_for_kind(
    source_kind: str,
) -> SourceFootprintOwner | None:
    """Resolve special scaled formats or an ordinary safetensors dtype.

    Scale-bearing native formats are owned by their registered producer
    ``FormatSpec``.  Every other supported dtype is derived through
    ``footprint``'s safetensors storage-width authority.  Sentinels such as
    ``unknown``/``heterogeneous`` have neither owner and therefore fail closed.
    """
    kind = str(source_kind)
    format_name = SOURCE_FORMAT_BY_KIND.get(kind)
    if format_name is not None:
        return SourceFootprintOwner(kind, format_name=format_name)
    try:
        item = plain_source_dtype_tensor_payload_breakdown(kind, (1,))
    except ValueError:
        return None
    return SourceFootprintOwner(
        kind,
        safetensors_dtype=str(item["source_dtype"]),
    )


def source_format_for_kind(source_kind: str) -> fr.FormatSpec | None:
    """Resolve a census source kind to its registered footprint owner.

    ``SOURCE_FORMAT_BY_KIND`` is derived from
    ``SOURCE_PASSTHROUGH_CONTRACTS`` and validated one-to-one at import time.
    Adding a future source representation to that contract table automatically
    subjects every candidate to its exact registered footprint; there is no
    parallel numeric source-bpp lookup to update or drift.
    """
    owner = source_footprint_owner_for_kind(source_kind)
    if owner is None or owner.format_name is None:
        return None
    return fr.get_format(owner.format_name)

# Passthrough formats whose Δloss is EXACT BY CONSTRUCTION, so the allocator
# synthesizes their candidate instead of requiring a cost-table column.
#
# Every cost in this pipeline is measured against the DEQUANTIZED SOURCE. A
# format that ships the source bytes UNCHANGED is therefore the identity
# transform on the reference: its Δloss is 0.0 as a matter of arithmetic, not
# of measurement, and no cost run can ever produce a more accurate number for
# it. FP8_SOURCE and BF16 are deliberately NOT members: the cost pipeline
# already emits real rows for them on the checkpoints where they are legal,
# and synthesizing over a table that has an entry would hide a disagreement.
# A byte-copy contract belongs here only when BOTH its W and A paths are the
# identity. ``FP8_BLOCK_UE8M0_SOURCE`` qualifies because its contract declares
# exactly that (resident E4M3 + UE8M0 planes, BF16 activations unchanged);
# whether any runtime serves it is a separate fact, and since the Gridbook
# lane's retirement (2026-09-02) none does -- its row below is
# ``ROUTE_STATUS_BLOCKED`` and stays fail-closed independently of membership.
#
# Membership is not self-certifying: ``cost_entry_is_source_passthrough``
# additionally requires the format to be a registered passthrough format whose
# activation path is the identity, so a stray ``cost_source`` string in a
# hand-written table cannot claim exactness for an activation-quantizing rung.
SOURCE_PASSTHROUGH_FORMATS: frozenset[str] = frozenset(
    name for name, contract in SOURCE_PASSTHROUGH_CONTRACTS.items()
    if contract.zero_cost_by_construction
)

# Passthrough formats with no DEMONSTRATED serve route on the target — either
# unaudited (pending) or measured dead (blocked). An otherwise honestly priced
# candidate remains useful serving-gap evidence, but the exporter refuses to
# ship a selection containing one without an explicit override.  This set does
# not override independent cost-currency admission for activation-changing
# re-quantization routes such as direct G32 MXFP8 W8A8; the raw block-128
# source route is W8A16 and preserves A.
ROUTE_PENDING_PASSTHROUGH_FORMATS: frozenset[str] = frozenset(
    name for name, contract in SOURCE_PASSTHROUGH_CONTRACTS.items()
    if not contract.route_backed
)

# The closed wire enum the artifact declares and the serving side reads
# (quant_config.json "source_passthrough"). Registry names are ours to rename;
# these are a cross-repo contract, so they are declared once here and the
# exporter maps through this table rather than spelling them again.
PASSTHROUGH_WIRE_FORMAT_IDS: dict[str, str] = {
    name: contract.wire_format_id
    for name, contract in SOURCE_PASSTHROUGH_CONTRACTS.items()
    if contract.wire_format_id is not None
}

# The same cross-repo wire enum, for formats this producer RE-ENCODES rather
# than copies. Kept as its own table beside the passthrough ids rather than
# folded into SOURCE_PASSTHROUGH_CONTRACTS, because that table means something
# specific and load-bearing: every member copies the source WEIGHT payload
# exactly and is source-kind gated. End-to-end Δloss is zero only for the
# derived ``SOURCE_PASSTHROUGH_FORMATS`` subset whose activation path is also
# identity. A re-quantization rung has a real weight encoder, so declaring it
# here would still falsely claim byte-copy provenance
# (tests/test_source_passthrough_family.py pins that invariant).
#
# What the two tables DO share is that the id is a contract with the consumer:
# the registry name is ours to rename, the wire id is not.
#
# NOTE FOR ORCHESTRATOR RECONCILIATION: ``mxfp8_e4m3_e8m0_g32`` is the
# spelling the Gridbook consumer side proposed before the lane's retirement
# (2026-09-02; the wire id is kept as the persisted spelling). It diverges from this repo's
# own convention, which spells the unsigned-E8M0 scale plane ``ue8m0``
# (``mxfp4_e2m1_ue8m0_g32``, ``fp8_e4m3_ue8m0_block128``). The consumer's
# spelling is used verbatim here rather than guessed at; if the two repos
# settle on ``mxfp8_e4m3_ue8m0_g32`` instead, this one string is the only
# producer-side edit.
REQUANT_WIRE_FORMAT_IDS: dict[str, str] = {
    "MXFP8_UE8M0_G32": "mxfp8_e4m3_e8m0_g32",
}

# Every wire id this producer can declare, from either table. Ids must be
# globally unique: the consumer dispatches on the id alone and cannot tell
# which producer-side table it came from.
WIRE_FORMAT_IDS: dict[str, str] = {
    **PASSTHROUGH_WIRE_FORMAT_IDS,
    **REQUANT_WIRE_FORMAT_IDS,
}


def passthrough_serving_notes() -> dict[str, dict[str, str]]:
    """Per-format serving requirements/evidence for the artifact's notes.

    A BACKED route with an unmet requirement serves no better than a blocked
    one, so the requirement travels with the artifact rather than living in a
    run book.
    """
    return {
        name: {
            key: value
            for key, value in (
                ("route_status", contract.route_status),
                ("route", contract.serving_route),
                ("requirement", contract.route_requirement),
                ("evidence", contract.route_evidence),
            )
            if value is not None
        }
        for name, contract in sorted(SOURCE_PASSTHROUGH_CONTRACTS.items())
    }
