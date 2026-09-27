"""The recipe a campaign encodes is the served wire the pinned contract attests (#1502).

``tessera.export.wire_recipe`` is the research default: below the E2M1x2 cap it
is the WINDOW body, span 1, ``window_bits`` 12. The pinned contract's
``formats[].attested_wire`` stamps the served routed wire, TCQ span 2, at every
E2M1_K2 rung from 128 to 896, and the routed cells attest those rungs. A
routed unit priced on the research wire is priced on bytes the routed decoder
cannot read; G2 spent w02r row-0045 on exactly that.
"""
import pytest

pytest.importorskip("tessera")

from prismaquant.joint_catalog_extension import added_format_recipe
from prismaquant.tessera_formats import (
    TesseraRouteRefused, parse_tessera_format_name, tessera_served_wire_recipe,
    tessera_wire_recipe,
)


def _served(fmt, structure):
    family, rung = parse_tessera_format_name(fmt)
    return tessera_served_wire_recipe(family, rung, structure=structure)


def _attested(fmt):
    recipe = dict(added_format_recipe(fmt))
    recipe.pop("grid")
    recipe.pop("q256")
    return recipe


@pytest.mark.parametrize("fmt", ["TESSERA_E2M1_K2_R640", "TESSERA_E2M1_K2_R768"])
def test_a_routed_subcap_e2m1_unit_encodes_the_attested_wire(fmt):
    assert _served(fmt, "routed_moe").to_config() == _attested(fmt)


@pytest.mark.parametrize("fmt", [
    "TESSERA_E2M1_K2_R896",
    *(f"TESSERA_E4M3_K1_R{q}" for q in (768, 896, 1152, 1408, 1536, 2048)),
    *(f"TESSERA_BF16_K1_R{q}" for q in (960, 1088, 1152, 1408, 1536, 2048)),
])
@pytest.mark.parametrize("structure", ["dense", "routed_moe"])
def test_where_the_research_wire_is_attested_the_encode_is_unchanged(fmt, structure):
    family, rung = parse_tessera_format_name(fmt)
    assert _served(fmt, structure) == tessera_wire_recipe(family, rung)
    assert _served(fmt, structure).to_config() == _attested(fmt)


@pytest.mark.parametrize("fmt", ["TESSERA_E2M1_K2_R640", "TESSERA_E2M1_K2_R768"])
def test_a_dense_subcap_e2m1_unit_is_refused_as_a_route_fact(fmt):
    # The attested wire differs from the research wire at this rung, and no
    # dense cell attests the rung (the E2M1 dense cells publish [896]): the
    # dense route has no attested reading of either spelling.
    with pytest.raises(TesseraRouteRefused, match="dense"):
        _served(fmt, "dense")
