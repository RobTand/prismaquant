"""Price every plane on the explicit wire recipe, including each TCQ forest.

Tessera's control accountant uses the served recipe at v68. The research
encoder retains its research default. Compare each price against the same
recipe. Large encoder checks require PRISMAQUANT_TESSERA_SLOW_ENCODE=1.
"""

import os
from fractions import Fraction

import pytest
from tessera.errors import GrammarError
import torch

from prismaquant.tessera_footprint import tessera_tensor_payload_breakdown
from prismaquant.tessera_formats import (
    TesseraFormatError,
    artifact_bpp,
    family_rate_cap,
    get_tessera_family,
    tessera_wire_recipe,
)

from tessera.control import grid_for_name, unit_wire_bits
from tessera.export import encode_linear, served_recipe
from tessera.grammar import forest_plane_bytes
from tessera.manifest import BodyKind

#: The six shapes the issue tabulates: four MoE-expert-sized units where the
#: fixed 512 B is 0.05-0.13 bpp, and two dense ones where it rounds away.
ISSUE_SHAPES = ((96, 768), (64, 512), (96, 320), (64, 640), (512, 2048), (1024, 3072))
SMALL_SHAPES = ISSUE_SHAPES[:4]
LARGE_SHAPES = ISSUE_SHAPES[4:]

#: ``(family, tessera grid name, rungs)``.  R895/R896 straddle the E2M1x2
#: recipe change -- a window body with a 4096-byte table, then the coset
#: trellis with a 512-byte forest -- and R511/R512 are the arity-1 pair where
#: the two-forest mechanism lives.
FAMILIES = (
    ("TESSERA_E2M1_K2", "E2M1x2", (895, 896)),
    ("TESSERA_E2M1_K1", "E2M1", (511, 512)),
)

_SLOW = os.environ.get("PRISMAQUANT_TESSERA_SLOW_ENCODE") == "1"


def _cases(shapes):
    """Every (family, grid, rung, shape) the *grammar* admits at that shape.

    A rung is not universally legal: the Bresenham quota has to close over the
    unit's columns, so R895 is unrealisable over 320 and R511 over 320 or 640.
    Filtering here rather than hardcoding a list is what keeps this table a
    statement about bytes instead of a statement about which rungs exist.
    """
    out = []
    for family, grid, rungs in FAMILIES:
        spec = get_tessera_family(family)
        for shape in shapes:
            for rung in rungs:
                wire = tessera_wire_recipe(family, rung)
                try:
                    spec.column_schedule(rung, shape[1], recipe=wire)
                except Exception:
                    continue
                out.append(pytest.param(family, grid, rung, shape,
                                        id=f"{grid}-R{rung}-{shape[0]}x{shape[1]}"))
    return out


def _priced_bytes(family, rung, shape, *, recipe=None):
    rows, columns = shape
    bpp = artifact_bpp(family, rung, shape=shape, recipe=recipe)
    priced = bpp * rows * columns
    assert priced.denominator == 1, (family, rung, shape, bpp)
    assert priced % 8 == 0, "a plane region is a whole number of bytes"
    return priced // 8


@pytest.mark.parametrize("family,grid,rung,shape", _cases(ISSUE_SHAPES))
def test_the_accountant_and_the_wire_are_one_number(family, grid, rung, shape):
    """``artifact_bpp`` must equal ``unit_wire_bits`` exactly, at every shape.

    No tolerance: both are exact ``Fraction`` arithmetic over integer plane
    extents, and the byte budget is spent in this currency.  This is the leg
    that covers all six shapes, because it costs nothing to evaluate.
    """
    rows, columns = shape
    recipe = served_recipe(grid_for_name(grid), rung)
    assert _priced_bytes(family, rung, shape, recipe=recipe) == unit_wire_bits(
        grid, rung, rows, columns
    ) / 8


@pytest.mark.parametrize("family,grid,rung,shape", _cases(SMALL_SHAPES))
def test_the_accountant_prices_what_the_exporter_writes(family, grid, rung, shape):
    """...and both must equal the bytes ``encode_linear`` serializes.

    ``tessera_tensor_payload_breakdown`` is checked on the same line, because
    it is the *other* PrismaQuant accountant -- the one the allocator's byte
    path reads -- and it was handing ``build_planes`` an empty descendant blob,
    so it charged zero descendant bytes on every rung.  It is asserted at every
    shape, including the two whose columns are not whole 256-column
    superblocks: it used to refuse those, and the wire holds a trailing
    partial superblock (#1849).
    """
    rows, columns = shape
    priced = _priced_bytes(family, rung, shape)
    torch.manual_seed(11)
    exported = encode_linear(
        torch.randn(rows, columns), grid=grid_for_name(grid), q256=rung,
        scale_refit=0,
    ).exact_bytes
    assert priced == exported, (family, rung, shape)
    breakdown = tessera_tensor_payload_breakdown(
        shape, family=family, body_rate_q256=rung
    )
    # The accountant also prices the container/fused frame (#1609), which
    # ``exact_bytes`` (the plane region) excludes by definition.
    assert breakdown["payload_bytes"] == exported, (family, rung, shape)
    assert (breakdown["total_bytes"] - int(breakdown["container_side_bytes"])
            == exported), (family, rung, shape)


@pytest.mark.slow
@pytest.mark.skipif(not _SLOW, reason="PRISMAQUANT_TESSERA_SLOW_ENCODE!=1")
@pytest.mark.parametrize("family,grid,rung,shape", _cases(LARGE_SHAPES))
def test_the_exporter_agrees_at_the_large_shapes_too(family, grid, rung, shape):
    """The same identity where the 512 B rounds to 0.001 bpp.

    Recheck the byte contract at large shapes, where dropping this per-unit
    term makes a smaller difference to reported bpp.
    """
    rows, columns = shape
    torch.manual_seed(11)
    exported = encode_linear(
        torch.randn(rows, columns), grid=grid_for_name(grid), q256=rung
    ).exact_bytes
    assert _priced_bytes(family, rung, shape) == exported


def test_the_hole_the_issue_measured_is_the_forest_and_nothing_else():
    """512 B at the E2M1x2 cap, 24 B at R512, 44 B at R511 -- from the grammar.

    The under-report the issue tabulated is exactly
    ``forest_plane_bytes(schedule, cap)`` summed, so this pins the *size* of
    the correction against Tessera's own statement of it rather than against a
    number copied out of the issue.  R511's schedule spans rates 1 and 2 and
    therefore carries two forests, which is why it is the larger of the two
    arity-1 corrections and why it outweighs the uniform rung above it.
    """
    spec = get_tessera_family("TESSERA_E2M1_K2")
    wire = tessera_wire_recipe("TESSERA_E2M1_K2", 896)
    assert BodyKind(wire.body) is BodyKind.TCQ
    # A k=2 code has 8 payload bits, so its TCQ cap is 7 and the coset rung
    # schedules every column at 7: one 256-code alphabet, one 256-byte
    # descendant table.  The cap is the *recipe's*, never a subtraction.
    cap = family_rate_cap(spec, wire)
    rates = spec.column_schedule(896, 768, recipe=wire)
    assert (cap, set(rates)) == (7, {7})
    assert sum(forest_plane_bytes(rates, cap)) == 512

    k1 = get_tessera_family("TESSERA_E2M1_K1")
    assert family_rate_cap(k1) == 3
    schedule_511 = k1.column_schedule(511, 512, recipe=tessera_wire_recipe(
        "TESSERA_E2M1_K1", 511))
    schedule_512 = k1.column_schedule(512, 512, recipe=tessera_wire_recipe(
        "TESSERA_E2M1_K1", 512))
    assert set(schedule_511) == {1, 2} and set(schedule_512) == {2}
    assert sum(forest_plane_bytes(schedule_511, 3)) == 44
    assert sum(forest_plane_bytes(schedule_512, 3)) == 24
    # ...and the two-forest rung really does outweigh the uniform one above it.
    assert _priced_bytes("TESSERA_E2M1_K1", 511, (64, 512)) > _priced_bytes(
        "TESSERA_E2M1_K1", 512, (64, 512))


def test_no_tessera_rung_has_a_rate_without_a_shape():
    """The design consequence, stated so it cannot be quietly undone.

    Every wire Tessera writes now charges something per *unit*: a TCQ body its
    forest, a WINDOW body its table, a CHANNEL plane its row field.  So there
    is no shape-free bits-per-parameter for any Tessera rung, and asking for
    one raises instead of returning a floor -- a ``Fraction`` cannot carry
    "this is a lower bound", and the only consumer of a shape-free rate,
    ``FormatSpec.exact_bits_per_param``, is multiplied by a parameter count as
    an exact rate.
    """
    from prismaquant.tessera_formats import enumerate_grid_space, wire_overhead_q256

    checked = 0
    for spec in enumerate_grid_space():
        lo, hi = spec.mathematical_q256_bounds
        for rung in (lo, hi):
            with pytest.raises(TesseraFormatError, match="per unit"):
                wire_overhead_q256(spec, recipe=tessera_wire_recipe(spec, rung),
                                   rung=rung)
            with pytest.raises(TesseraFormatError, match="per unit"):
                artifact_bpp(spec, rung)
            checked += 1
    assert checked >= 8, checked


def test_the_synthesized_format_spec_prices_by_shape_and_never_by_a_scalar():
    """``exact_bits_per_param`` is None on every Tessera rung; the fn is set.

    ``FormatSpec`` holds exactly one of the two, so no consumer can pick the
    cheaper one -- and with the forest charged, the scalar is the one that
    cannot exist.  ``memory_bytes_for_shape`` is what the byte gate reads, so
    it is asserted here against the same three numbers above.
    """
    from prismaquant.tessera_render import synthesize_tessera_spec

    # ``bits_for_shape_fn`` routes through ``tessera_tensor_payload_breakdown``,
    # which also prices trailing partial superblocks (#1849); these cases
    # use whole superblocks.
    for name, shape in (("TESSERA_E2M1_K2_R896", (64, 512)),
                        ("TESSERA_E2M1_K1_R512", (96, 768)),
                        ("TESSERA_E4M3_K1_R1024", (96, 512))):
        spec = synthesize_tessera_spec(name)
        assert spec.exact_bits_per_param is None, name
        assert spec.bits_for_shape_fn is not None, name
        family, rung = name.rsplit("_R", 1)
        # ``memory_bytes_for_shape`` is the whole-unit price: plane region plus
        # the container/fused frame the breakdown carries (#1609).
        side = int(tessera_tensor_payload_breakdown(
            shape, family=family, body_rate_q256=int(rung)
        )["container_side_bytes"])
        assert spec.memory_bytes_for_shape(shape) == Fraction(
            _priced_bytes(family, int(rung), shape) + side
        ), (name, shape)


# ---------------------------------------------------------------------------
# The guard that does not need the next plane to be named
# ---------------------------------------------------------------------------

#: Three shapes for the exhaustive sweep: an MoE expert column count where a
#: per-unit term is 0.1 bpp, a wider expert, and a dense Linear where the same
#: term rounds to 0.002 bpp and an accountant that dropped it would look right.
SWEEP_SHAPES = ((96, 320), (96, 768), (2048, 1024))


def _grid_name(spec) -> str:
    """Tessera's own name for a family's payload grid."""
    return spec.base if spec.arity == 1 else f"{spec.base}x{spec.arity}"


def test_every_realisable_rung_of_every_family_prices_what_tessera_prices():
    """Both accountants price the served recipe at every realizable rung.

    The encoder tests separately check the research recipe. No comparison
    grants a serving cell or a tensor-parallel receipt.
    """
    from prismaquant.tessera_menu import menu_families

    checked = 0
    refused = 0
    total_rungs = 0
    families = 0
    for spec in menu_families():
        families += 1
        grid = _grid_name(spec)
        lo, hi = spec.mathematical_q256_bounds
        total_rungs += hi - lo + 1
        for rows, columns in SWEEP_SHAPES:
            for rung in range(lo, hi + 1):
                try:
                    recipe = served_recipe(grid_for_name(grid), rung)
                    priced = _priced_bytes(spec.name, rung, (rows, columns), recipe=recipe)
                except TesseraFormatError:
                    # The quota does not close over these columns.  Then
                    # Tessera must refuse the same pair: the two accountants
                    # agree about which rungs EXIST, not only what they cost.
                    with pytest.raises(GrammarError):
                        unit_wire_bits(grid, rung, rows, columns)
                    refused += 1
                    continue
                assert priced == unit_wire_bits(grid, rung, rows, columns) / 8, (
                    spec.name, rung, (rows, columns)
                )
                checked += 1
    # Not vacuous, and the bound is derived rather than typed: 256 divides
    # 768 and 1024, so every rung of every family is realisable at those two
    # widths and the sweep must have priced all of them there; 320 is not a
    # multiple of 256, so some rungs are refused there and the refusal path
    # above must have been exercised.
    assert families >= 4, families
    assert checked >= 2 * total_rungs, (checked, total_rungs)
    assert refused > 0, refused


@pytest.mark.parametrize("family,grid,rung", (
    ("TESSERA_E4M3_K1", "E4M3", 256),
    ("TESSERA_BF16_K1", "BF16", 256),
))
def test_the_window_families_agree_with_the_exporter_too(family, grid, rung):
    """The exporter leg on the two WINDOW/CHANNEL families, not just E2M1.

    The forest is a TCQ term, so these two families were never wrong -- which
    is exactly why they belong here.  The identity above is between two
    accountants; if a plane were added that *both* of them missed, only an
    encode would say so, and an encode leg that covers one body kind is not a
    statement about the wire.  One rung each, at the family's floor, on a
    64x512 unit.  Scale refits improve reconstruction quality, which this
    byte-extent identity does not inspect; the real encoder still selects
    codes, serializes every plane and verifies the round trip.
    """
    torch.manual_seed(11)
    exported = encode_linear(
        torch.randn(64, 512), grid=grid_for_name(grid), q256=rung,
        scale_refit=0,
    ).exact_bytes
    assert _priced_bytes(family, rung, (64, 512)) == exported


def test_an_unrealisable_rung_refuses_in_this_modules_own_error_type():
    """A grammar refusal must not cross the seam as a ``tessera`` exception.

    Charging the forest is what first put a Bresenham walk underneath
    ``artifact_bpp``: before it, a TCQ rung over a block plane had a shape-free
    price and never asked whether the quota closed.  ``bresenham_rate_schedule``
    raises ``tessera.errors.GrammarError``, which descends from
    ``TesseraError`` and **not** from ``ValueError``, so every caller in this
    tree that guards a price with ``except TesseraFormatError`` --
    ``menu_families``, ``enumerate_grid_space``, ``expand_tessera_menu`` --
    would have seen it propagate straight out of the byte gate.  Re-raised in
    this module's type, exactly as ``TesseraFamily.__post_init__`` does with
    ``tuple_grid``'s refusal.
    """
    from tessera.errors import GrammarError

    assert not issubclass(GrammarError, ValueError)
    # R257 needs 5/4 columns at rate 2 over 320 columns: the quota cannot close.
    with pytest.raises(TesseraFormatError, match="not realisable over 320"):
        artifact_bpp("TESSERA_E2M1_K1", 257, shape=(96, 320))
    # ...and it is still a rung the grammar admits at a column count that works.
    assert _priced_bytes("TESSERA_E2M1_K1", 257, (96, 768)) > 0
