"""Exact serialized bytes for one Tessera Linear, from Tessera's own layout.

This replaces ``trellis_footprint`` (archived 2026-09-02 under
``archive/trellis_wire_2026-09-02/``, #118) in the allocator's byte path.  The
two are not ports of each other and must not be: ``trellis_footprint`` prices the
``gridbook.trellis.wire.v1`` layout -- an 88-byte binary header, its own row
alignment and block-offset rules -- and Tessera has a different wire
(``prismaquant.tessera.v1``) with a different plane set.  Re-deriving Tessera's
byte count here from Gridbook's model would produce a number that no exporter
would ever write.

So nothing here counts bytes. ``tessera.layout`` does, through the shared
plane/terminal extent arithmetic used by the artifact writer. Older producers
expose that arithmetic through ``build_planes`` / ``build_terminal``; newer
ones also expose payload-free extent builders. This module arranges the call
and reports the result. That is deliberate: the
allocator's byte budget, the accountant's figures and the exported artifact
have to be one number, and the only way to guarantee that is to have one
implementation of it.

The allocator reads four fields -- ``exact_bpw``, ``format``, ``shape`` and
``total_bytes``.  The rest is provenance, carried so a priced candidate can be
audited back to the planes it was priced from.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from fractions import Fraction
from functools import lru_cache

from .tessera_menu import menu_scaled_cache
from .tessera_formats import (
    SUPERBLOCK_WEIGHTS,
    TesseraFamily,
    TesseraFormatError,
    _schedule_rates,
    family_rate_cap,
    get_tessera_family,
    tessera_serving_route,
    recipe_from_wire_names,
    scale_plane_name,
    tessera_served_wire_recipe,
    validate_body_rate_q256,
)
from .digests import DIRECT_ASCII_LAX_DEFAULT_STR
from .name_projection import strip_weight_leaf

__all__ = [
    "TESSERA_TENSOR_PAYLOAD_SCHEMA",
    "TesseraShapeRate",
    "tessera_exact_bits_for_shape",
    "tessera_member_name_bytes",
    "tessera_tensor_payload_breakdown",
    "validate_tessera_tensor_payload_breakdown",
]

# v2 (#1609): the breakdown prices the container side bytes (manifest,
# container header, fused framing, member name) beside the plane region.
TESSERA_TENSOR_PAYLOAD_SCHEMA = "prismaquant.tessera_tensor_payload.v2"

#: Tessera's group/half geometry: the block scale planes are laid out per 32
#: weights (the S6b E8M0 base byte) and per 16 (the S6b nibble refinement, or
#: the LUT plane's nibble index).  The CHANNEL plane uses neither -- it is one
#: fp16 per output row -- so these size a plane only when the recipe asks for
#: one.  Carried on the wire rather than assumed, which is why they are named
#: here and not inlined.
GROUP_WEIGHTS = 32
HALF_WEIGHTS = 16

try:  # pragma: no cover
    from tessera import layout as _layout
    from tessera.grammar import forest_plane_bytes
    from tessera.layout import TerminalSpec, build_planes, build_terminal
    from tessera.manifest import BodyKind, Geometry
    # The side bytes a unit carries beside its plane region (#1609): the
    # container header, the canonical manifest and the TSRFUSE1 framing.  All
    # of it is sized by Tessera's own writers, called and never restated.
    from tessera.canonical import Writer as _CanonicalWriter, fits_uint
    from tessera.container import HEADER_BYTES as _CONTAINER_HEADER_BYTES
    # Underscore-private in Tessera, but they are the published TSRFUSE1 wire
    # constants (`<8sBB` once per blob, `<HIQ` per member); tessera.fused
    # publishes no size function to call instead.
    from tessera.fused import _HEADER as _FUSED_HEADER, _MEMBER as _FUSED_MEMBER
    from tessera.grammar import bresenham_rate_schedule, root_from_q256
    from tessera.layout import ZERO_DIGEST
    from tessera.manifest import (
        ArrangementMode,
        BranchIdentity,
        ContainerClass,
        Manifest,
        ReachParams,
        RotationState,
        ScalePlane,
        ScalePlaneKind,
        TerminalRecord,
    )
    from tessera.planes import PlaneDescriptor
except ImportError as exc:  # pragma: no cover
    raise TesseraFormatError(
        "prismaquant.tessera_footprint requires the `tessera` package, which "
        "owns the plane layout these byte counts come from."
    ) from exc


# The existing reviewed producer pin remains supported. Newer producer inputs
# can price directly from extents without allocating or hashing payload bytes.
_build_plane_extents = getattr(_layout, "build_plane_extents", None)
_build_terminal_extent = getattr(_layout, "build_terminal_extent", None)


#: What the recipe digest covers.  Named because a content address is only
#: meaningful next to its scope: this addresses the *pre-render recipe* -- the
#: family, rung, geometry and plane counts -- and deliberately not the rendered
#: weights, which do not exist yet when a candidate is priced.
_IDENTITY_SCOPE = "family+rung+geometry+planes"


def _recipe_identity(breakdown: Mapping[str, object]) -> str:
    """SHA-256 over canonical JSON of everything but the digest itself.

    A content address, not an authorization signature.  Recomputing it detects
    a report that has been edited or has drifted from the layout that produced
    it -- which is exactly what a downstream price must not be built on.
    """
    body = {k: v for k, v in breakdown.items()
            if k != "pre_render_recipe_identity_sha256"}
    return DIRECT_ASCII_LAX_DEFAULT_STR.sha256(body)


def _alphabet_bytes(
    spec: TesseraFamily, alphabets: "Mapping[int, Sequence[int]] | None",
    cap: "int | None" = None,
) -> int:
    """Bytes for the anchor tables, one entry per code in the grid's width.

    The width is ``PayloadGrid.code_bytes`` -- Tessera's own answer, one byte
    up to 256 codes and two up to 65536, refusing anything wider as a schema
    change.  It used to be a local ``1 or 4``, which is a second spelling of a
    wire fact and disagreed with the grid above 256 codes; no shipping figure
    moves (every family that reaches this path is a byte-coded one), and the
    disagreement is gone.  The tables are per *rate*, because a rate's anchor
    set is ``2^(R+1)`` codes and a unit that mixes rates carries a table for
    each rate it uses.
    """
    if not alphabets:
        return 0
    cap = spec.rate_cap if cap is None else int(cap)
    width = spec.code_bytes
    total = 0
    for rate, codes in alphabets.items():
        if not 1 <= rate <= cap:
            raise TesseraFormatError(
                f"{spec.name}: alphabet given for rate {rate}, outside 1..{cap}"
            )
        expected = 1 << (rate + 1)
        if len(codes) != expected:
            raise TesseraFormatError(
                f"{spec.name} rate {rate}: alphabet has {len(codes)} anchors, "
                f"|A_R| = 2^(R+1) = {expected}"
            )
        total += len(codes) * width
    return total


@lru_cache(maxsize=1)
def _max_ratio_bytes() -> int:
    """The widest ``ratio`` the canonical codec can write, from its own writer.

    A ratio is ``sint(numerator) + uint(denominator)``; the widest is the
    largest numerator and denominator ``fits_uint`` admits.  Asked of
    ``tessera.canonical.Writer`` rather than written down, so the bound moves
    when the codec does.
    """
    top = 1
    while fits_uint(top * 2 + 1):
        top = top * 2 + 1
    return len(_CanonicalWriter().ratio(Fraction((top - 1) // 2, top)).bytes)


def _fused_member_bytes(member_name: str) -> int:
    """TSRFUSE1 bytes one member costs beside its blob: row plus UTF-8 name."""
    return _FUSED_MEMBER.size + len(member_name.encode("utf-8"))


def _container_side_bytes(
    *, spec, wire, rung, rows, columns, geometry, rates, planes, record,
    plane, body, span, window_bits, unit_id,
) -> "tuple[int, int]":
    """Bytes one unit carries beside its plane region, as an upper bound.

    Built by encoding a placeholder ``tessera.manifest.Manifest`` of the unit's
    own shape and rung with Tessera's writer, then adding the container header
    and the TSRFUSE1 framing that is not the member name.  Every field that
    depends on weights is a fixed-width digest, so the manifest length is
    shape-determined except for the ratios (a scale plane's global scale and a
    reach sigma), which are priced at the codec's widest ratio.  Returns
    ``(side_bytes_without_member_name, ratio_slack_bytes)``.
    """
    def _as(cls, obj, **extra):
        kw = {f.name: getattr(obj, f.name) for f in fields(cls)
              if hasattr(obj, f.name)}
        kw.update(extra)
        return cls(**kw)

    descs = tuple(
        p if isinstance(p, PlaneDescriptor)
        else _as(PlaneDescriptor, p, content_digest=ZERO_DIGEST)
        for p in planes
    )
    terminal = (
        record if isinstance(record, TerminalRecord)
        else _as(TerminalRecord, record, payload_digest=ZERO_DIGEST)
    )
    ratio = _max_ratio_bytes()
    placeholder_ratio = len(_CanonicalWriter().ratio(Fraction(1)).bytes)
    slack = 0
    if plane == "s6b":
        scale_plane = ScalePlane.s6b()
    elif plane == "lut16":
        scale_plane = ScalePlane.lut(bytes(range(0x08, 0x18)), 1.0)
        slack += ratio - placeholder_ratio
    elif plane == "channel":
        scale_plane = ScalePlane.channel(1.0)
        slack += ratio - placeholder_ratio
    else:  # pragma: no cover - the plane names are a closed set upstream
        raise TesseraFormatError(f"{spec.name}: unknown scale plane {plane!r}")
    root_q256 = rung * spec.arity
    arrangement = (
        ArrangementMode.BRESENHAM
        if tuple(rates) == tuple(bresenham_rate_schedule(
            root_from_q256(root_q256), columns, cap=None))
        else ArrangementMode.STORED
    )
    # Reach is stored only when a byte-moving spelling is set; the flag is one
    # byte either way, and each stored sigma is one presence byte plus a ratio.
    reach = None
    reach_slack = 0
    if body is BodyKind.WINDOW and (
            getattr(wire, "window_seed", 0)
            or getattr(wire, "window_sigma", None) is not None):
        reach = ReachParams(window_seed=int(getattr(wire, "window_seed", 0)),
                            window_sigma=getattr(wire, "window_sigma", None))
    if plane == "channel" and getattr(wire, "channel_sigma", None) is not None:
        reach = ReachParams(
            window_seed=reach.window_seed if reach else 0,
            window_sigma=reach.window_sigma if reach else None,
            channel_sigma=wire.channel_sigma,
        )
    if reach is not None:
        stored = sum(v is not None
                     for v in (reach.window_sigma, reach.channel_sigma))
        reach_slack = stored * ratio
    manifest = Manifest(
        encoder_profile_id=ZERO_DIGEST,
        branch=BranchIdentity(
            unit_id=unit_id, root_q256=root_q256,
            rotation=RotationState.NONE, container=ContainerClass.GRIDBOOK),
        geometry=geometry,
        arrangement=arrangement,
        rates=tuple(rates),
        planes=descs,
        terminals=(terminal,),
        payload_digest=ZERO_DIGEST,
        span=span,
        scale_plane=scale_plane,
        body=body,
        window_bits=window_bits,
        reach=reach,
        # The exporter stamps a 32-byte fixture id on every real artifact.
        encoder_fixture_id=ZERO_DIGEST,
    )
    # Only sigmas that were priced at 1.0-width placeholders need slack; a
    # reach sigma is priced by its real value, so its slack is the gap to max.
    slack += max(0, reach_slack - _reach_ratio_bytes(reach))
    side = (len(manifest.encode()) + _CONTAINER_HEADER_BYTES
            + _FUSED_HEADER.size + _FUSED_MEMBER.size)
    return side, slack


def _reach_ratio_bytes(reach) -> int:
    """Bytes the placeholder's own reach sigmas spent on their ratios."""
    if reach is None:
        return 0
    total = 0
    for value in (reach.window_sigma, reach.channel_sigma):
        if value is not None:
            total += len(_CanonicalWriter().ratio(Fraction(float(value))).bytes)
    return total


def tessera_tensor_payload_breakdown(
    shape: Sequence[int],
    *,
    family: "str | TesseraFamily",
    body_rate_q256: int,
    layout: str = "tight",
    schedule: "Sequence[int] | None" = None,
    alphabets: "Mapping[int, Sequence[int]] | None" = None,
    alphabet_bytes: "int | None" = None,
    sidecar_header_bytes: int = 0,
    completion: "int | None" = 0,
    with_diagonals: bool = False,
    span: "int | None" = None,
    scale_plane: "str | None" = None,
    recipe=None,
    structure: "str | None" = None,
    member_name: "str | None" = None,
) -> dict[str, object]:
    """Exact serialized bytes for one 2-D Linear weight at one Tessera rung.

    The total is the whole fused wire blob, not only the plane region (#1609):
    plane bytes, the canonical manifest, the container header and the TSRFUSE1
    framing, each sized by Tessera's own writer (``_container_side_bytes``).
    That side figure is an upper bound with a few bytes of slack -- the
    manifest is shape-determined but for the ratio fields, which are priced at
    the codec's widest.  ``member_name`` is the tensor's name inside the fused
    blob; it is not shape-determined, so the caller supplies it and it is priced
    exactly.  Left unset, the name is not priced (``member_name_priced`` is
    False) and the caller adds ``member_name_bytes`` for its own name.

    ``recipe`` -- a ``tessera.export.WireRecipe`` -- is the wire being priced,
    and an explicit recipe remains authoritative. Otherwise ``structure``
    selects ``tessera_served_wire_recipe``: a routed stack is priced on its
    existing served wire, while ``None`` preserves the research recipe.
    Unattested explicit serving structures are refused by that resolver.
    ``span``/``scale_plane`` remain as
    the two-scalar spelling for callers that predate the recipe; naming both is
    refused.  The resolved body, span, plane and window width are all recorded
    in the breakdown and re-derived by
    ``validate_tessera_tensor_payload_breakdown``, so a WINDOW or CHANNEL
    footprint revalidates as itself rather than as the default wire.

    ``schedule`` may be omitted, in which case the canonical Bresenham schedule
    for the rung is used -- the same one the encoder would build.  Passing one
    is how an importance-placed arrangement gets priced; it is checked against
    the rung rather than trusted, because a schedule that does not realise its
    root would price a rung the artifact does not contain.
    """
    spec = get_tessera_family(family)
    if recipe is not None and (span is not None or scale_plane is not None):
        raise TesseraFormatError(
            "name a recipe or the span/scale_plane scalars, not both"
        )
    if recipe is None:
        wire = tessera_served_wire_recipe(
            spec, body_rate_q256, structure=structure,
        )
        if span is not None or scale_plane is not None:
            wire = recipe_from_wire_names(
                int(wire.span if span is None else span),
                scale_plane_name(wire.scale_plane if scale_plane is None
                                 else scale_plane),
            )
    else:
        wire = recipe
    body = BodyKind(wire.body)
    span = int(wire.span)
    plane = scale_plane_name(wire.scale_plane)
    window_bits = int(wire.window_bits)
    # The rate ceiling is the body's, exactly as ``tessera.export.plan_for``
    # and ``unit_artifact`` dispatch it: the TCQ trellis spends a payload bit
    # on its code, the WINDOW body spends none.
    cap = family_rate_cap(spec, wire)
    rung = validate_body_rate_q256(spec, body_rate_q256, recipe=wire)
    if span < 1:
        raise TesseraFormatError(f"span must be positive, got {span}")
    if body is BodyKind.WINDOW:
        if span != 1:
            raise TesseraFormatError("a window body is span 1")
        if completion not in (None, 0):
            raise TesseraFormatError("a window body has no completion axis")
        completion = 0

    dims = tuple(shape)
    if len(dims) != 2 or any(type(v) is not int or v <= 0 for v in dims):
        raise TesseraFormatError(
            f"Tessera tensor shape must be two positive integers, got {dims}"
        )
    rows, columns = dims
    # A column count that is not a whole number of superblocks is priced, not
    # refused (#1849).  The wire holds a trailing partial superblock:
    # ``tessera.grammar.superblock_count`` ceilings the partition,
    # ``superblock_quota_ok`` constrains only complete superblocks, and
    # ``layout.build_planes`` (called below through ``Geometry``) gives the
    # partial block its granule.  Whether a rung *exists* over these columns is
    # the schedule's question, asked just below by ``column_schedule`` and the
    # quota check -- the same two legs ``tessera_menu.tessera_shape_legal``
    # asks, so the menu and this accountant agree about which rungs exist.
    if type(sidecar_header_bytes) is not int or sidecar_header_bytes < 0:
        raise TesseraFormatError("sidecar_header_bytes must be nonnegative")

    # ``_schedule_rates`` is ``column_schedule`` with Tessera's refusal of a
    # rung whose quota does not close over these columns re-raised as a
    # ``TesseraFormatError`` -- the one refusal type every caller guards with.
    # The superblock refusal above used to fire first on every column count
    # where a rung can fail to close, so the raw ``GrammarError`` never leaked.
    rates = (
        _schedule_rates(spec, rung, columns, wire)
        if schedule is None
        else tuple(int(r) for r in schedule)
    )
    if len(rates) != columns:
        raise TesseraFormatError(
            f"schedule covers {len(rates)} columns for shape {dims}"
        )
    # A schedule is only this rung's schedule if its quota matches.  Checking
    # here is what stops a mispriced candidate reaching the DP.
    quota = sum(rates) * SUPERBLOCK_WEIGHTS
    expected = rung * spec.arity * columns
    if quota * 256 != expected * SUPERBLOCK_WEIGHTS:
        raise TesseraFormatError(
            f"{spec.name}: schedule sums to {sum(rates)} bits over {columns} "
            f"columns, which is not rung {rung} (q256)"
        )
    if any(not 1 <= r <= cap for r in rates):
        raise TesseraFormatError(
            f"{spec.name}: schedule has a rate outside 1..{cap}"
        )
    if body is BodyKind.WINDOW and window_bits < max(rates):
        raise TesseraFormatError(
            f"{spec.name}: window_bits {window_bits} cannot hold a rate-"
            f"{max(rates)} position's bits"
        )

    # --- arity: the code grid is not the weight grid -----------------------
    # A tuple code covers `arity` **consecutive rows** (tessera's `tuple_grid`:
    # codes map onto k consecutive rows, because the trellis runs down columns
    # and a tuple must be contiguous along the trellis axis).  So the BODY and
    # COMPLETION planes have `rows // arity` code-rows, not `rows`.
    #
    # This is declared the way the *wire* declares it: a weight-space
    # `Geometry` plus an explicit `arity` handed to `build_planes` /
    # `build_terminal`, which is what `tessera.unit_artifact` and
    # `tessera.calculator.terminal_rate` both do.  It used to be declared the
    # other way -- code rows with the group geometry divided by the same factor
    # -- which cancels exactly for the per-code and per-block planes and so
    # priced every TCQ artifact identically.  It does **not** cancel for a
    # per-*row* plane: DIAG_SV holds one fp16 per output channel
    # (`layout._counts_for`), and shrunk rows under-declared the CHANNEL scale
    # plane by exactly `arity`.  Verified against `terminal_rate` at three
    # shapes, both arities, both spans: identical everywhere except CHANNEL at
    # arity 2, where this convention is the wire's and the other one was wrong.
    if rows % spec.arity:
        raise TesseraFormatError(
            f"{spec.name}: {rows} rows is not a multiple of arity {spec.arity}; "
            "a tuple code spans that many consecutive rows and cannot straddle "
            "the end of the tensor"
        )
    code_rows = rows // spec.arity
    if code_rows % span:
        raise TesseraFormatError(
            f"{spec.name}: {code_rows} code rows is not a whole number of "
            f"span-{span} super-symbols; this shape cannot carry the span"
        )

    # The **forest**: a TCQ body's two anchor planes, and the term this
    # accountant charged at zero until 2026-09-03 (RobTand/prismaquant#126).
    # ALPHABET holds `2^(R+1)` anchor codes and DESCENDANT `2^(cap+1)` bytes,
    # once per *distinct* rate in the schedule, both written inline in the unit
    # (`tessera.unit_artifact._forest_planes`).  Sized by tessera's own
    # `forest_plane_bytes` -- called, not restated -- because a second
    # implementation of one accountant is exactly the defect being repaired.
    # A window body has no forest: its ALPHABET plane is its table and its
    # DESCENDANT plane is empty.
    forest_alphabet, descendant_bytes = (
        (0, 0) if body is BodyKind.WINDOW else forest_plane_bytes(rates, cap)
    )
    # `alphabet_bytes` is the already-counted figure, which is what a recorded
    # footprint carries; `alphabets` is the table itself.  Revalidating a report
    # must use the recorded count, or it re-prices a different unit.
    if alphabet_bytes is None:
        alphabet_bytes = (
            _alphabet_bytes(spec, alphabets, cap) if alphabets else forest_alphabet
        )
    elif type(alphabet_bytes) is not int or alphabet_bytes < 0:
        raise TesseraFormatError("alphabet_bytes must be a nonnegative integer")
    if body is BodyKind.TCQ and alphabet_bytes != forest_alphabet:
        # The forest is not a caller's choice.  The exporter writes one anchor
        # table per distinct rate whatever a caller supplies, so a count that
        # disagrees is a report of a unit the artifact does not hold -- which
        # includes every footprint recorded before this term existed, and those
        # SHOULD be refused: their `total_bytes` is light by exactly this.
        raise TesseraFormatError(
            f"{spec.name}: a TCQ body's ALPHABET plane is its forest -- "
            f"{forest_alphabet} bytes over the {len(set(rates))} distinct "
            f"rate(s) rung {rung} schedules -- and the wire writes it whatever "
            f"the caller supplies; got alphabet_bytes={alphabet_bytes}"
        )
    if body is BodyKind.WINDOW:
        # The ALPHABET plane *is* the window table: `2^window_bits` grid codes
        # of `PayloadGrid.code_bytes` each, written inline in the unit
        # (`tessera.unit_artifact`).  It is charged here because it is charged
        # on the wire, and an accountant that left it out would disagree with
        # the artifact by exactly the bytes that distinguish a wide window from
        # a narrow one -- or, with the width hardcoded to one byte, by exactly
        # the bytes that distinguish the 16-bit route from the 8-bit one.
        table_bytes = spec.code_bytes << window_bits
        if alphabets or alphabet_bytes not in (0, table_bytes):
            raise TesseraFormatError(
                f"{spec.name}: a window body's ALPHABET plane is its own "
                f"{table_bytes}-byte table; an anchor alphabet cannot be "
                "supplied alongside it"
            )
        alphabet_bytes = table_bytes

    geometry = Geometry(
        rows=rows,
        columns=columns,
        superblock_columns=SUPERBLOCK_WEIGHTS,
        group_weights=GROUP_WEIGHTS,
        half_weights=HALF_WEIGHTS,
        quantizable_params=rows * columns,
    )
    terminal_spec = TerminalSpec(
        # The exporter's terminal slot (`unit_artifact.py`, "t-nvfp4").  The slot
        # name is written into the manifest's terminal record, so its length is
        # side bytes; Tessera exports no constant for it.
        slot_id="t-nvfp4",
        # ``completion`` is the second rate axis, and the default is the
        # exporter's: ``encode_linear(completion=0)``.  It briefly defaulted to
        # the cap instead, because ``unit_artifact`` was writing the COMPLETION
        # plane at full width whatever depth the encoder spent -- so the cap
        # was, for a few hours, what the wire really did.  Both defaults have
        # been wrong in opposite directions and by the same amount, and the
        # only defence is that this spec now sizes the *planes* as well as the
        # terminal, so the two cannot describe different artifacts.
        completion_bits=tuple(
            (cap - r) if completion is None
            else min(completion, cap - r)
            for r in rates
        ),
        released_positions=0,
        # A LUT plane has no base plane; its table lives in the manifest
        # (side bytes, priced by ``_container_side_bytes``, not here).  A
        # CHANNEL plane has no block plane at all: the scale is one fp16 per
        # output row on DIAG_SV (schema minor 3, `tessera.scale_channel`).
        with_scale_base=plane == "s6b",
        with_scale_refine=plane in ("s6b", "lut16"),
        with_diagonals=with_diagonals,
        with_row_scale=plane == "channel",
    )
    if plane == "channel" and with_diagonals:
        raise TesseraFormatError(
            f"{spec.name}: a CHANNEL plane *is* the DIAG_SV field; segment 2a "
            "cannot be fitted under it (tessera.encode.encode_unit)"
        )
    layout_kwargs = dict(
        with_diagonals=with_diagonals,
        cap=cap,
        arity=spec.arity,
        spec=terminal_spec,
        span=span,
        with_row_scale=plane == "channel",
    )
    if _build_plane_extents is not None and _build_terminal_extent is not None:
        planes = _build_plane_extents(
            geometry, rates, alphabet_bytes, descendant_bytes, **layout_kwargs)
        terminal_builder = _build_terminal_extent
    else:
        # Compatibility with the pinned producer: its writer remains the
        # byte authority, including blob extents and plane alignment.
        planes = build_planes(
            geometry, rates, bytes(alphabet_bytes), bytes(descendant_bytes),
            **layout_kwargs)
        terminal_builder = build_terminal
    record = terminal_builder(geometry, rates, terminal_spec, planes,
                              alphabet_bytes, descendant_bytes,
                              cap=cap, arity=spec.arity, span=span)

    route = tessera_serving_route(spec, wire, rung)
    unit_id = spec.format_name(rung, recipe=wire)
    frame_bytes, ratio_slack_bytes = _container_side_bytes(
        spec=spec, wire=wire, rung=rung, rows=rows, columns=columns,
        geometry=geometry, rates=rates, planes=planes, record=record,
        plane=plane, body=body, span=span, window_bits=window_bits,
        unit_id=unit_id,
    )
    member_name_bytes = (
        len(member_name.encode("utf-8")) if member_name is not None else 0)
    container_side_bytes = frame_bytes + ratio_slack_bytes + member_name_bytes
    total_bytes = (record.exact_bytes + sidecar_header_bytes
                   + container_side_bytes)
    exact_bpw = Fraction(total_bytes * 8, rows * columns)
    breakdown = {
        "schema": TESSERA_TENSOR_PAYLOAD_SCHEMA,
        "wire_schema": "prismaquant.tessera.v1",
        "format": spec.format_name(rung, recipe=wire),
        "family": spec.name,
        "grid": spec.base,
        "arity": spec.arity,
        "lane": spec.lane,
        "shape": [rows, columns],
        "layout": layout,
        "body_rate_q256": rung,
        "scale_contract": plane,
        "trellis_span": span,
        "body_kind": body.name.lower(),
        "window_bits": window_bits,
        "rate_cap": cap,
        "superblock_weights": SUPERBLOCK_WEIGHTS,
        "schedule_bits_per_code_row": sum(rates),
        "code_rows": code_rows,
        "distinct_rates": sorted(set(rates)),
        "alphabet_bytes": alphabet_bytes,
        "descendant_bytes": descendant_bytes,
        "sidecar_header_bytes": sidecar_header_bytes,
        "plane_elements": list(record.plane_elements),
        "payload_bytes": record.exact_bytes,
        # Bytes beside the plane region (#1609): manifest + container header +
        # TSRFUSE1 framing, from Tessera's own writer, with ratio widths at the
        # codec's maximum (an upper bound; the slack is ``ratio_slack_bytes``).
        "container_frame_bytes": frame_bytes,
        "ratio_slack_bytes": ratio_slack_bytes,
        "member_name": member_name,
        "member_name_bytes": member_name_bytes,
        "member_name_priced": member_name is not None,
        "container_side_bytes": container_side_bytes,
        "total_bytes": total_bytes,
        "exact_bpp_payload": str(record.exact_bpp),
        "exact_bpw": float(exact_bpw),
        "exact_bpw_rational": [exact_bpw.numerator, exact_bpw.denominator],
        # The route the decoded tile executes on: a joint property of the base
        # grid and the scale plane, not of the grid alone.  Reported because a
        # byte win that disappears at load is not a byte win, and a byte win on
        # a route no kernel takes is not a byte win either (principle 12).
        # `lane` above is the grid's -- can these values be a hardware format
        # at all -- while these three are the *recipe's*: an E4M3 tile over a
        # per-16 block plane is stock-lane and materialises into nothing,
        # because no kernel reads FP8 weights at that scale granularity.
        "terminal_format": route.terminal_format,
        "materialises": route.materialises,
        "activation_contract": route.contract,
        "min_capability_sm": route.min_capability_sm,
        "pre_render_recipe_identity_scope": _IDENTITY_SCOPE,
    }
    breakdown["pre_render_recipe_identity_sha256"] = _recipe_identity(breakdown)
    return breakdown


@menu_scaled_cache
def _exact_bits_for_shape(
    family_name: str,
    body_rate_q256: int,
    rows: int,
    columns: int,
    recipe,
) -> Fraction:
    """One rung's exact bits at one shape, memoised at the menu cache bound.

    This is the expensive half of a menu -- ``build_planes`` and the Bresenham
    schedule run per rung -- and it is the memo tessera#46 caught undersized:
    at ``maxsize=4096`` against a 6916-rung menu a pass over one shape evicted
    its own entries, so the second pass hit nothing.  The bound is
    ``tessera_menu.menu_cache_bound()``, which is the widest menu one shape can
    produce times the shapes a pass keeps live; the first factor is what makes
    self-eviction impossible, and it is asked of the family roster rather than
    written down here.
    """
    payload = tessera_tensor_payload_breakdown(
        (rows, columns),
        family=family_name,
        body_rate_q256=body_rate_q256,
        recipe=recipe,
    )
    return Fraction(int(payload["total_bytes"]) * 8)


def tessera_exact_bits_for_shape(
    family: "str | TesseraFamily",
    body_rate_q256: int,
    shape: Sequence[int],
    *,
    recipe=None,
    structure: "str | None" = None,
) -> Fraction:
    """Exact serialized bits for one Tessera tensor, planes included.

    The size question asked without the report: same family, same rung, same
    recipe (or declared serving structure), same arithmetic as
    :func:`tessera_tensor_payload_breakdown` --
    literally the same call -- so a rung cannot be priced one way for a
    ``FormatSpec`` and another way for an allocator candidate.  It is the
    ``layout="tight"``, canonical-schedule figure, which is what a
    format-level price is.  It carries the container side bytes every unit
    pays whatever it is called -- the manifest at its shape-determined
    maximum, the 24-byte container header and the fused framing -- and not
    the fused member name, which belongs to a unit, not to a format:
    :func:`tessera_member_name_bytes` prices that from the tensor's name.

    This is the answer for a wire the shape-free accountant cannot state -- a
    CHANNEL plane charges one fp16 per output row, a WINDOW body charges a
    ``2**L``-byte table per unit -- and it is exact for every other wire too.

    A packed ``(experts, out, in)`` stack is priced as ``experts`` units of
    ``(out, in)``, each paying its own window table and its own row field,
    because that is what the wire does rather than a convention chosen here:
    ``tessera.export.export_checkpoint_streaming`` encodes one unit per source
    tensor name and every source ships ``experts.{i}.*`` as separate 2-D
    tensors, the trellis runs down rows within a column so a fused
    ``(experts*out, in)`` unit would carry the path across expert boundaries
    and no single expert could be decoded alone, and the kernel lane decodes a
    unit against its own table (``_pack_window_unit``).  It is also the rule
    ``FormatSpec.scale_count_for_shape`` already applies to a stacked tensor:
    outer count times the per-matrix figure.  Anything that is not 2-D or 3-D
    is **refused** (as a ``ValueError``, which every caller of
    ``memory_bytes_for_shape`` in the tree already treats as "this format
    cannot take this tensor") rather than flattened into a rate.
    """

    spec = get_tessera_family(family)
    dims = tuple(int(d) for d in shape)
    if len(dims) not in (2, 3):
        raise TesseraFormatError(
            f"{spec.name}: an exact Tessera size needs a 2-D Linear weight or "
            f"a 3-D (experts, out, in) stack of them, got shape {dims}"
        )
    experts = dims[0] if len(dims) == 3 else 1
    if experts <= 0:
        raise TesseraFormatError(
            f"{spec.name}: a packed stack needs at least one expert, got "
            f"shape {dims}"
        )
    rows, columns = dims[-2], dims[-1]
    wire = (tessera_served_wire_recipe(spec, body_rate_q256, structure=structure)
            if recipe is None else recipe)
    return experts * _exact_bits_for_shape(
        spec.name, int(body_rate_q256), rows, columns, wire,
    )


@dataclass(frozen=True, slots=True)
class TesseraShapeRate:
    """The shape-aware size of one Tessera rung, as a callable.

    Handed to ``FormatSpec.bits_for_shape_fn``.  A frozen dataclass rather than
    a closure so two specs synthesized for the same rung compare equal and
    pickle -- ``get_format`` builds a fresh ``FormatSpec`` on every call, and a
    lambda would make those specs unequal and unsendable.
    """

    family: str
    body_rate_q256: int
    recipe: object

    def __call__(self, shape: Sequence[int]) -> Fraction:
        return tessera_exact_bits_for_shape(
            self.family, self.body_rate_q256, shape, recipe=self.recipe,
        )

    def unit_name_bytes(self, tensor_name: str, shape: Sequence[int]) -> int:
        """The fused member-name bytes the format price leaves to the unit."""
        return tessera_member_name_bytes(tensor_name, shape)


def tessera_member_name_bytes(tensor_name: str, shape: Sequence[int]) -> int:
    """Bytes of the TSRFUSE1 member names one tensor's wire carries.

    The exporter frames each unit as ``pack_fused([(projection, rows, blob)])``
    (``tessera.export_serving``), so a Linear's member name is its projection
    role -- the last component of the recipe name, without the parameter
    leaf -- and a packed ``(experts, out, in)`` stack ships one framed unit
    per expert, each naming the same role.  Priced exactly from the name the
    caller holds; nothing is guessed.
    """
    dims = tuple(int(d) for d in shape)
    experts = dims[0] if len(dims) == 3 else 1
    role = strip_weight_leaf(str(tensor_name)).rsplit(".", 1)[-1]
    return experts * len(role.encode("utf-8"))


def validate_tessera_tensor_payload_breakdown(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Re-check a footprint at an API boundary, and recompute what it claims.

    The arithmetic is re-derived rather than trusted: a report that has been
    edited or has drifted from the layout that produced it is exactly the
    thing a downstream price should not be built on.
    """
    if not isinstance(payload, Mapping):
        raise TesseraFormatError("Tessera footprint must be a mapping")
    copied = dict(payload)
    if copied.get("schema") != TESSERA_TENSOR_PAYLOAD_SCHEMA:
        raise TesseraFormatError(
            f"footprint schema must be {TESSERA_TENSOR_PAYLOAD_SCHEMA}, "
            f"got {copied.get('schema')!r}"
        )
    shape = copied.get("shape")
    if not isinstance(shape, Sequence) or len(shape) != 2:
        raise TesseraFormatError("footprint shape must be a two-element sequence")
    rows, columns = int(shape[0]), int(shape[1])
    recomputed = tessera_tensor_payload_breakdown(
        (rows, columns),
        family=str(copied["family"]),
        body_rate_q256=int(copied["body_rate_q256"]),
        layout=str(copied.get("layout", "tight")),
        alphabet_bytes=int(copied.get("alphabet_bytes", 0)),
        sidecar_header_bytes=int(copied.get("sidecar_header_bytes", 0)),
        member_name=copied.get("member_name"),
        # A report written before minor 1 carries none of these fields and
        # means the wire of its day; one written after names what it priced.
        # The recipe is rebuilt from the report rather than looked up, so a
        # footprint keeps revalidating as *itself* after the exporter's default
        # recipe moves -- which is the whole reason the fields are recorded.
        recipe=recipe_from_wire_names(
            span=int(copied.get("trellis_span", 1)),
            scale_plane=str(copied.get("scale_contract", "s6b")),
            body=str(copied.get("body_kind", "tcq")),
            window_bits=int(copied.get("window_bits", 0)),
        ),
    )
    claimed = copied.get("pre_render_recipe_identity_sha256")
    if claimed != _recipe_identity(copied):
        raise TesseraFormatError(
            "footprint recipe identity does not address its own contents; the "
            "report has been edited or has drifted from its layout"
        )
    for field in ("total_bytes", "payload_bytes", "exact_bpw", "format",
                  "container_side_bytes", "member_name_bytes"):
        if copied.get(field) != recomputed[field]:
            raise TesseraFormatError(
                f"footprint {field} is {copied.get(field)!r}, but the layout "
                f"gives {recomputed[field]!r}"
            )
    return copied
