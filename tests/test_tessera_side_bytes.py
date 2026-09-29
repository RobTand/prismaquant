"""The byte price covers what a Linear's wire frame carries besides its planes (#1609).

``tessera.layout`` sizes the plane region only.  A unit on the wire is
``container(24 B header + canonical manifest + plane region)`` and the
exporter frames it in a ``TSRFUSE1`` blob (10 B header, 14 B + name per
member).  Pricing the plane region alone under-counted every Linear by the
manifest, the two headers and the member name -- 748 to 812 B on a GLM-5.3
expert -- so an allocator "fit the card" answer sat below the artifact's real
size (-28.7 MB over 36,288 routed Linears).

These tests build a real unit with Tessera's own writer
(``encode_linear_planes`` -> ``pack_fused``) and require the price to be an
upper bound on the bytes that writer produced, with only a few bytes of
slack.  Nothing here restates a width: the wire is measured, not derived.
"""
import pytest

from prismaquant.tessera_footprint import tessera_tensor_payload_breakdown
from prismaquant.tessera_formats import get_tessera_family, tessera_wire_recipe

# Preserve CI's per-test deadline in PB too; these are CPU wire-contract tests.
pytestmark = pytest.mark.timeout(300)

# (family, rung, shape).  Covers the CHANNEL plane on the window body and the
# LUT16 plane on both bodies, the three shapes of manifest the writer emits.
# R1023 needs 256 columns for its exact mixed {3, 4} rate schedule, not a
# production-sized row batch.  Eight rows retain independent channel scales;
# the other cases retain large geometry and both LUT16 manifest forms (#1746).
CASES = [
    ("TESSERA_E4M3_K1", 1024, (256, 256)),
    ("TESSERA_E4M3_K1", 1023, (8, 256)),
    ("TESSERA_E2M1_K1", 768, (256, 256)),
    ("TESSERA_E2M1_K2", 896, (256, 256)),
]
IDS = [f"{family}-R{rung}" for family, rung, _ in CASES]

# A few bytes: the ratio fields are priced at their widest varint and the
# fused header is charged per Linear rather than per blob.  A regression that
# prices the frame twice, or by a heuristic pad, blows through this.
SLACK_BOUND = 64


def _encode(family, rung, shape):
    import torch
    from tessera.export import encode_linear_planes

    spec = get_tessera_family(family)
    wire = tessera_wire_recipe(spec, rung)
    generator = torch.Generator().manual_seed(1609)
    weight = torch.randn(shape, generator=generator, dtype=torch.float32)
    exported, unit, _forests = encode_linear_planes(
        weight, grid=spec.payload_grid(), q256=rung,
        name=spec.format_name(rung, recipe=wire), verify=False,
    )
    if family == "TESSERA_E4M3_K1" and rung == 1023:
        from tessera.container import parse
        from tessera.manifest import BodyKind, ScalePlaneKind
        from tessera.planes import PlaneKind

        parsed = parse(exported.blob)
        manifest = parsed.manifest
        assert manifest.body is BodyKind.WINDOW
        assert manifest.scale_plane.kind is ScalePlaneKind.CHANNEL
        assert (manifest.geometry.rows, manifest.geometry.columns) == shape
        assert manifest.rates == unit.rates
        assert set(manifest.rates) == {3, 4}
        assert sum(manifest.rates) * 256 == rung * shape[1]
        assert manifest.window_bits > max(manifest.rates)
        assert unit.window_codes.numel() == 2 ** manifest.window_bits
        assert unit.scale_rows.numel() == shape[0]
        planes = {plane.kind: plane for plane in manifest.planes}
        assert planes[PlaneKind.DIAG_SV].counts[-1] == shape[0]
        assert planes[PlaneKind.ALPHABET].counts[-1] == unit.window_codes.numel()
        assert planes[PlaneKind.BODY].counts[-1] > 0
        assert parsed.side_bytes > 24
        assert len(parsed.plane_region) == exported.exact_bytes
    return spec, exported


@pytest.mark.parametrize("family,rung,shape", CASES, ids=IDS)
def test_the_price_covers_the_container_the_writer_produced(family, rung, shape):
    spec, exported = _encode(family, rung, shape)
    price = tessera_tensor_payload_breakdown(
        shape, family=spec, body_rate_q256=rung
    )["total_bytes"]
    assert price >= len(exported.blob), (
        f"priced {price} B, the container is {len(exported.blob)} B "
        f"(plane region {exported.exact_bytes} B)"
    )


@pytest.mark.parametrize("family,rung,shape", CASES, ids=IDS)
def test_the_price_covers_the_fused_wire_and_is_tight(family, rung, shape):
    from tessera.fused import pack_fused

    spec, exported = _encode(family, rung, shape)
    member = "down_proj"
    wire = pack_fused([(member, shape[0], exported.blob)])
    # Without a caller-supplied name the price carries the frame but not the
    # name; add the name back to compare with the wire.
    price = tessera_tensor_payload_breakdown(
        shape, family=spec, body_rate_q256=rung
    )["total_bytes"]
    slack = price + len(member) - len(wire)
    assert 0 <= slack <= SLACK_BOUND, (
        f"priced {price} B (+{len(member)} B name), wire is {len(wire)} B, "
        f"slack {slack} B"
    )


@pytest.mark.parametrize("family,rung,shape", CASES[:1], ids=IDS[:1])
def test_the_member_name_is_priced_exactly_from_the_caller(family, rung, shape):
    spec = get_tessera_family(family)
    bare = tessera_tensor_payload_breakdown(shape, family=spec, body_rate_q256=rung)
    named = tessera_tensor_payload_breakdown(
        shape, family=spec, body_rate_q256=rung, member_name="down_proj"
    )
    assert named["total_bytes"] - bare["total_bytes"] == len("down_proj")
    assert named["member_name_bytes"] == len("down_proj")


def test_a_packed_expert_stack_pays_one_member_name_per_expert():
    from prismaquant.tessera_footprint import tessera_member_name_bytes

    assert tessera_member_name_bytes(
        "model.layers.3.mlp.experts.down_proj.weight", (256, 2048, 4096)
    ) == 256 * len("down_proj")
    assert tessera_member_name_bytes(
        "model.layers.3.mlp.shared_experts.gate_proj.weight", (2048, 4096)
    ) == len("gate_proj")
