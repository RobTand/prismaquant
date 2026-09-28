"""Lane format families: core resolves a lane's formats through lane data (#1551).

Decoupling step 6, part 1. A lane declares its format families in its lane
spec (``format_families``) and names its code plugin (``plugin``). Core asks
``format_registry.format_family_of`` whose a name is -- a prefix read off the
spec, with no plugin import -- and reaches synthesis, context admission and
the production render through ``lane_spec.family_hook``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from prismaquant import lane_spec
from prismaquant.lane_spec import LaneFormatFamily, LaneSpec

ROOT = Path(__file__).resolve().parents[1]


def _minimal_lane(**extra) -> dict:
    return {
        "schema": lane_spec.SCHEMA,
        "id": "fourth",
        "export_container": "fourth",
        "runtime": "vllm",
        "wired_architectures": ["*"],
        "endpoint": {"kind": "none"},
        "kl_evaluator": {"kind": "validate_assignments_kl", "entrypoint": "x"},
        **extra,
    }


def test_tessera_lane_declares_its_family_and_plugin():
    spec = lane_spec.load_lane_spec("tessera")
    assert spec.plugin == "prismaquant.tessera_lane"
    (family,) = spec.format_families
    assert (family.id, family.lane, family.name_prefix) == ("tessera", "tessera", "TESSERA_")
    assert family.requires_production_render and family.rate_axis
    assert spec.layer_config_meta_prefixes == ("tessera_",)


def test_stock_lanes_declare_no_family():
    for lane in ("compressed_tensors", "gguf"):
        assert lane_spec.load_lane_spec(lane).format_families == ()


def test_family_claims_by_prefix_after_canonical_case():
    family = LaneFormatFamily(id="f", lane="l", name_prefix="FOURTH_", label="F")
    assert family.claims("FOURTH_A_R1") and family.claims(" fourth_a ")
    assert not family.claims("NVFP4") and not family.claims(None)


def test_family_without_plugin_is_refused():
    with pytest.raises(ValueError, match="no `plugin`"):
        LaneSpec.from_dict(_minimal_lane(format_families=[
            {"id": "f", "name_prefix": "FOURTH_"}]))


def test_family_prefix_must_be_upper_case():
    with pytest.raises(ValueError, match="upper-case"):
        LaneSpec.from_dict(_minimal_lane(plugin="x", format_families=[
            {"id": "f", "name_prefix": "fourth_"}]))


def test_overlapping_family_prefixes_are_refused(monkeypatch):
    specs = (
        LaneSpec.from_dict(_minimal_lane(plugin="x", format_families=[
            {"id": "a", "name_prefix": "FOURTH_"}])),
        LaneSpec.from_dict({**_minimal_lane(plugin="y", format_families=[
            {"id": "b", "name_prefix": "FOURTH_X_"}]), "id": "fifth"}),
    )
    monkeypatch.setattr(lane_spec, "_declared_lane_specs", lambda: specs)
    lane_spec.format_families.cache_clear()
    try:
        with pytest.raises(ValueError, match="one name must have one owner"):
            lane_spec.format_families()
    finally:
        lane_spec.format_families.cache_clear()


def test_a_family_whose_plugin_lacks_a_hook_refuses():
    family = lane_spec.format_family("tessera")
    with pytest.raises(LookupError, match="no 'no_such_hook' hook"):
        lane_spec.family_hook(family, "no_such_hook")


def test_every_family_hook_core_calls_exists_on_the_tessera_plugin():
    family = lane_spec.format_family("tessera")
    for hook in ("synthesize_format", "format_admitted_in_contexts",
                 "render_production", "resolved_serving_lane",
                 "require_canonical_subfamily", "format_subfamily"):
        assert callable(lane_spec.family_hook(family, hook))
    plugin = lane_spec.single_lane_plugin("serving_runtime_pin_path")
    assert plugin is lane_spec.lane_plugin("tessera")
    assert issubclass(plugin.ServingRuntimePinError, ValueError)


def test_family_lookup_imports_neither_the_plugin_nor_tessera():
    """The stock hot path asks "whose format is this?" for free."""
    code = textwrap.dedent("""
        import sys
        from prismaquant import format_registry as fr
        assert fr.format_family_of("TESSERA_E4M3_K1_R1024").id == "tessera"
        assert fr.format_family_of("NVFP4") is None
        assert fr.requires_production_render("TESSERA_E4M3_K1_R1024")
        assert not fr.requires_production_render("NVFP4")
        assert fr.rate_axis_format("TESSERA_BF16_K1_R832")
        assert not fr.rate_axis_format("FP8_DYNAMIC")
        assert fr.format_owner_label("TESSERA_X") == "Tessera"
        loaded = sorted(m for m in sys.modules
                        if m == "tessera" or m.startswith(("tessera.", "prismaquant.tessera_")))
        assert not loaded, loaded
    """)
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_synthesized_spec_carries_its_family_capabilities():
    pytest.importorskip("tessera")
    from prismaquant import format_registry as fr

    spec = fr.get_format("TESSERA_E4M3_K1_R1024")
    family = fr.format_family_of(spec.name)
    assert spec.render_owner == family.id
    assert spec.requires_production_render is family.requires_production_render
    assert fr.get_format("NVFP4").render_owner is None
    assert fr.get_format("NVFP4").requires_production_render is False


def test_unknown_name_outside_every_family_is_the_registry_keyerror():
    from prismaquant import format_registry as fr

    with pytest.raises(KeyError, match="Unknown format"):
        fr.get_format("NOT_A_FORMAT_ANYWHERE")


def test_layer_config_meta_prefixes_come_from_the_lanes():
    assert "tessera_" in lane_spec.layer_config_meta_prefixes()


def test_lane_specs_parse_through_the_declared_registry():
    raw = {p.stem: json.loads(p.read_text()) for p in
           (ROOT / "prismaquant" / "lane_specs").glob("*.json")}
    ids = {spec.id for spec in lane_spec._declared_lane_specs()}
    assert ids == {payload["id"] for payload in raw.values()}
