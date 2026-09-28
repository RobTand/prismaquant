"""The allocator's DP receives lane facts through hooks (#1552).

Decoupling step 6, part 2a. ``allocator_solver``, ``allocator_candidates``,
``cost_currency``, ``prepriced_cost`` and ``unit_topology_restamp`` import no
lane module. A format name's grammar (promotion class, whole-group options,
the fused-module signature, the rung parse) is answered by the owning family's
plugin through ``format_registry``; the fused-module licence and its field
vocabulary by the licensing lane through ``allocator_solver``; the campaign
currency by the family's lane-spec declaration. The DP's exact reductions
(``prune_dominated``, ``collapse_to_dp_bins``) live in ``allocator_solver``.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from prismaquant import allocator_solver as solver
from prismaquant import format_registry as fr
from prismaquant import lane_spec

ROOT = Path(__file__).resolve().parents[1]

# Registry formats, rungs, group options, a family class, malformed and
# non-string names: the hooks must answer each exactly as the lane's own
# grammar does.
_NAMES = (
    "NVFP4", "FP8_DYNAMIC", "BF16",
    "TESSERA_E2M1_K2_R896", "TESSERA_E4M3_K1_R1024",
    "TESSERA_E2M1_K2_G0", "TESSERA_E2M1_K2",
    "tessera_e2m1_k2_r896", " TESSERA_E2M1_K2_R896", "TESSERA_NOPE",
    None, 3,
)


@pytest.mark.parametrize("name", _NAMES)
def test_registry_grammar_hooks_agree_with_the_lane(name):
    from prismaquant import tessera_formats as tf

    assert fr.format_promotion_class(name) == tf.format_promotion_class(name)
    assert fr.is_group_option(name) == tf.is_tessera_group_option(name)
    shared = ("family", "grid", "body", "plane", "q256")
    try:
        expected_sig = tf.fused_shared_signature(name, shared)
    except Exception as exc:  # noqa: BLE001 -- the hook must raise alike
        with pytest.raises(type(exc)):
            fr.fused_shared_signature(name, shared)
    else:
        assert fr.fused_shared_signature(name, shared) == expected_sig
    try:
        expected = tf.parse_tessera_format_name(name)
    except ValueError as exc:
        with pytest.raises(type(exc)):
            fr.parse_family_rung(name)
    else:
        assert fr.parse_family_rung(name) == expected


def test_an_illegal_rung_of_a_family_shaped_name_still_raises():
    from prismaquant.tessera_formats import TesseraFormatError

    with pytest.raises(TesseraFormatError):
        fr.parse_family_rung("TESSERA_E2M1_K2_R1")


def test_group_option_names_come_from_the_family():
    from prismaquant.tessera_formats import tessera_group_option_name

    assert (fr.group_option_name("TESSERA_E2M1_K2", 4)
            == tessera_group_option_name("TESSERA_E2M1_K2", 4))
    with pytest.raises(ValueError, match="registry format"):
        fr.group_option_name("NVFP4", 0)


def test_the_dp_reductions_live_in_the_solver():
    from prismaquant import tessera_menu

    assert tessera_menu.prune_dominated is solver.prune_dominated
    assert tessera_menu.collapse_to_dp_bins is solver.collapse_to_dp_bins


def test_the_licence_follows_the_lane_read(monkeypatch):
    """Substituting the lane's one read substitutes what the DP receives."""
    from prismaquant import tessera_menu

    marker = object()
    monkeypatch.setattr(tessera_menu, "fused_module_licence", lambda: marker)
    assert solver.lane_fused_module_licence() is marker


def test_the_licence_vocabulary_is_the_lanes():
    from prismaquant import tessera_formats as tf

    assert solver.lane_fused_module_fields() == (
        tf.FUSED_MODULE_RUNG_FIELDS, tf.FUSED_MODULE_SHAPE_FIELDS,
        tf.FUSED_MODULE_RATE_FIELD)


def test_a_licence_without_a_vocabulary_refuses(monkeypatch):
    real = lane_spec.single_lane_hook
    monkeypatch.setattr(
        lane_spec, "single_lane_hook",
        lambda name: None if name == "fused_module_fields" else real(name))
    with pytest.raises(LookupError, match="fused_module_fields"):
        solver.lane_fused_module_fields()


def test_every_dp_hook_core_calls_exists_on_the_tessera_plugin():
    family = lane_spec.format_family_by_id("tessera")
    for hook in ("promotion_class", "is_group_option", "group_option_name",
                 "fused_shared_signature", "parse_format_name",
                 "route_admission", "menu_mode",
                 "tensor_parallel_applicability"):
        assert callable(lane_spec.family_hook(family, hook))
    for hook in ("fused_module_licence", "fused_module_fields",
                 "hessian_identity", "restamp_unit_topology"):
        assert callable(lane_spec.single_lane_hook(hook))


def test_registry_names_need_no_lane_import():
    """A stock menu's promotion and pruning questions import no lane code."""
    code = textwrap.dedent("""
        import sys
        from prismaquant import format_registry as fr
        from prismaquant import allocator_solver as solver
        assert fr.format_promotion_class("NVFP4") == "NVFP4"
        assert not fr.is_group_option("FP8_DYNAMIC")
        assert fr.fused_shared_signature("BF16", ("family",)) is None
        assert fr.parse_family_rung("NVFP4") is None
        rows = [(10, 1.0, "a"), (12, 2.0, "b"), (8, 3.0, "c")]
        assert [r[2] for r in solver.prune_dominated(rows)] == ["c", "a"]
        loaded = sorted(m for m in sys.modules
                        if m == "tessera" or m.startswith(("tessera.", "prismaquant.tessera_")))
        assert not loaded, loaded
    """)
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
