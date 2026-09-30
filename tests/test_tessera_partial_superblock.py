"""A unit whose columns are not whole 256-column superblocks is priced (#1849).

GLM-5.3's KDA gate up-projections, ``self_attn.f_b_proj`` and
``self_attn.g_b_proj``, are ``[8192, 128]``: 34 layers of each, 68 units the
attention census (#1842) has to price.  ``tessera_menu.tessera_shape_legal``
admitted them, because the Bresenham schedule closes over 128 columns, and
``tessera_footprint.tessera_tensor_payload_breakdown`` refused any column count
that is not a multiple of 256.  So ``expand_tessera_menu`` raised straight out
of the census before its first forward pass (PB d3f57e6c).

The wire holds a trailing partial superblock: ``tessera.grammar.
superblock_count`` rounds the partition up, ``superblock_quota_ok`` constrains
only complete superblocks, and ``layout.build_planes`` gives the partial block
a granule.  The fix prices the shape instead of refusing it.  These tests check
three things:

* The census's own call, the readable menu at ``[8192, 128]``, returns a
  priced menu instead of raising.  This is the reproduction.
* The price is Tessera's price.  The breakdown equals ``tessera.control.
  unit_wire_bits`` at the real shape for every realisable rung of every menu
  family.  It also equals the bytes ``encode_linear`` writes at a small
  128-column shape, for the TCQ family (both arities) and both window
  families.
* The menu and the accountant agree about which rungs exist:
  ``tessera_shape_legal`` admits a rung if and only if the breakdown prices it
  and Tessera's accountant prices it.

CPU only.  ``exact_bytes`` is a property of the layout, not of the weights.
"""

import pytest
import torch
from tessera.control import grid_for_name, unit_wire_bits
from tessera.errors import GrammarError
from tessera.export import encode_linear

from prismaquant.tessera_footprint import tessera_tensor_payload_breakdown
from prismaquant.tessera_formats import (
    TesseraFormatError,
    get_tessera_family,
    tessera_wire_recipe,
)
from prismaquant.tessera_menu import (
    MENU_READABLE,
    expand_tessera_menu,
    menu_families,
    tessera_shape_legal,
)

#: ``self_attn.f_b_proj`` / ``self_attn.g_b_proj`` in GLM-5.3-Flash, read from
#: the source safetensors headers.
KDA_GATE = (8192, 128)

#: A 128-column unit small enough that an encode is seconds on CPU.
SMALL_128 = (64, 128)


def _grid_name(spec) -> str:
    return spec.base if spec.arity == 1 else f"{spec.base}x{spec.arity}"


def test_the_census_menu_prices_the_kda_gate_shape():
    """The reproduction: PB d3f57e6c's call, which raised before the fix."""
    menu = expand_tessera_menu(KDA_GATE, mode=MENU_READABLE)
    assert menu, "a 128-column unit must keep Tessera rungs, not an empty menu"
    for rung in menu:
        # Exact bits over the unit's own parameters, planes included.
        assert rung.bits_per_param > 0
        assert rung.memory_bytes > 0


@pytest.mark.parametrize("family, grid, rung", (
    ("TESSERA_E2M1_K1", "E2M1", 512),
    ("TESSERA_E2M1_K2", "E2M1x2", 896),
    ("TESSERA_E4M3_K1", "E4M3", 256),
    ("TESSERA_E4M3_K1", "E4M3", 1024),
    ("TESSERA_BF16_K1", "BF16", 256),
))
def test_the_breakdown_prices_what_the_exporter_writes_at_128_columns(
        family, grid, rung):
    """The encode leg: the bytes that ship, at a partial superblock.

    Both body kinds, because the forest is a TCQ term and the table a WINDOW
    term, and a plane both accountants forgot would show up only here.
    """
    spec = get_tessera_family(family)
    rows, columns = SMALL_128
    spec.column_schedule(rung, columns, recipe=tessera_wire_recipe(family, rung))
    torch.manual_seed(11)
    exported = encode_linear(
        torch.randn(rows, columns), grid=grid_for_name(grid), q256=rung
    ).exact_bytes
    breakdown = tessera_tensor_payload_breakdown(
        SMALL_128, family=family, body_rate_q256=rung
    )
    assert breakdown["payload_bytes"] == exported, (family, rung)
    assert (breakdown["total_bytes"] - int(breakdown["container_side_bytes"])
            == exported), (family, rung)
    assert unit_wire_bits(grid, rung, rows, columns) == 8 * exported


def test_every_rung_at_the_kda_gate_shape_is_tesseras_price_or_tesseras_refusal():
    """The sweep at the real shape: price, existence and legality are one fact.

    For every rung of every menu family: either all three of the breakdown,
    ``unit_wire_bits`` and ``tessera_shape_legal`` refuse it (its quota does
    not close over 128 columns), or all three admit it and the two byte
    figures are equal.  Nothing is refused by only one of them, which is the
    disagreement that crashed the census.
    """
    rows, columns = KDA_GATE
    priced = refused = 0
    for spec in menu_families():
        grid = _grid_name(spec)
        lo, hi = spec.mathematical_q256_bounds
        for rung in range(lo, hi + 1):
            legal, why = tessera_shape_legal(spec, rung, KDA_GATE)
            try:
                breakdown = tessera_tensor_payload_breakdown(
                    KDA_GATE, family=spec.name, body_rate_q256=rung
                )
            except TesseraFormatError:
                assert not legal, (spec.name, rung, "menu admits, accountant refuses")
                with pytest.raises(GrammarError):
                    unit_wire_bits(grid, rung, rows, columns)
                refused += 1
                continue
            assert legal, (spec.name, rung, why)
            assert 8 * breakdown["payload_bytes"] == unit_wire_bits(
                grid, rung, rows, columns), (spec.name, rung)
            priced += 1
    # Not vacuous: the reproduction needs a priced menu at this shape.
    assert priced > 0, (priced, refused)


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="an [8192, 128] window encode is minutes on CPU; the census runs it on CUDA",
)


@CUDA
@pytest.mark.parametrize("fmt", (
    "TESSERA_E4M3_K1_R1024",
    "TESSERA_BF16_K1_R1024",
    "TESSERA_E2M1_K2_R896",
))
def test_a_kda_gate_unit_encodes_decodes_and_prices_on_cuda(fmt):
    """The census's own device, at the real shape: encode, decode, price.

    The CPU legs above prove the identity at a small 128-column unit; this one
    runs the unit the census will actually encode, on the device it will
    encode it on, and checks the decoded artifact comes back at its shape.
    """
    from tessera.unit_artifact import read_unit_artifact

    from prismaquant.tessera_formats import parse_tessera_format_name
    from prismaquant.tessera_render import _grid_for

    family, rung = parse_tessera_format_name(fmt)
    torch.manual_seed(0)
    weight = torch.randn(*KDA_GATE, device="cuda", dtype=torch.bfloat16)
    unit = encode_linear(
        weight, grid=_grid_for(family), q256=int(rung), name=fmt, verify=True
    )
    breakdown = tessera_tensor_payload_breakdown(
        KDA_GATE, family=family, body_rate_q256=int(rung)
    )
    assert unit.exact_bytes == breakdown["payload_bytes"], fmt
    decoded = read_unit_artifact(unit.blob, device="cuda")
    assert tuple(decoded.shape) == KDA_GATE, fmt
