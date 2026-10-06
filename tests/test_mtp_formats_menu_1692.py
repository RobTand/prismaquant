"""``--mtp-formats`` declares the GLM MTP layer's menu (PQ #1692).

The body menu is ``--formats``; the MTP selector read only the runtime's
attestation, so a release that declares one family (all Tessera-8) could not
say so for layer 45. The declaration intersects the attested menu, BF16
passthrough included; the record names it and what it removed, and the groups
stay selections, never ``fixed``. An intersection that leaves a unit or a group
without a rung refuses (exit 2 from the allocator). Unset is unchanged.

Rows are the synthetic joint-AURA fixture of ``test_glm_mtp_selection``; these
tests make no serving-quality claim.
"""
from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("torch")

from tests.test_glm_mtp_selection import (  # noqa: E402
    CONSTANTS, E4M3_SHARED, R832, R1024, ROUTED, SHARED, _bytes, _payload, _row,
    _allocator_attestation_fixture, _write_payload)
from test_rung_allowability import allowability_cli_args, publication  # noqa: E402

BF16_WIRE = "TESSERA_BF16_K1_R1024"
R896 = "TESSERA_E4M3_K1_R896"
NEW_KEYS = {"mtp_formats", "menu_restricted_rungs", "mtp_formats_unoffered"}


def _with_bf16_wire(payload):
    """Add a BF16_K1 routed rung that is CHEAPER in E than R1024, so the open
    menu picks it: the family declaration must move the choice."""
    for name in ROUTED:
        payload["costs"][name][BF16_WIRE] = _row(
            name, BF16_WIRE, [0.005, 0.006, 0.004, 0.005],
            payload["costs"][name][R1024]["probe_identity"])
        payload["wire_bytes"][name][BF16_WIRE] = payload["wire_bytes"][name][R1024]
    return payload


def test_unset_leaves_the_selection_record_unchanged():
    from prismaquant.glm_mtp_selection import select_mtp_rungs

    payload = _with_bf16_wire(_payload())
    budget = _bytes(R1024, "BF16")
    open_menu = select_mtp_rungs(payload, byte_budget=budget, constants=CONSTANTS)
    explicit_none = select_mtp_rungs(payload, byte_budget=budget, constants=CONSTANTS,
                                     formats=None)
    assert json.dumps(open_menu, sort_keys=True) == json.dumps(explicit_none, sort_keys=True)
    assert not NEW_KEYS & set(open_menu)
    assert open_menu["rung_by_group"]["routed"] == BF16_WIRE


def test_the_declared_family_is_selected_and_recorded_not_fixed():
    from prismaquant.glm_mtp_selection import select_mtp_rungs

    payload = _with_bf16_wire(_payload())
    budget = _bytes(R1024, "BF16")
    got = select_mtp_rungs(payload, byte_budget=budget, constants=CONSTANTS,
                           formats=[R1024, R832, "BF16", R896])
    assert got["rung"] == f"routed={R1024}|shared=BF16"
    assert "fixed_formats" not in got
    assert got["mtp_formats"] == sorted(["BF16", R832, R1024, R896])
    assert got["menu_restricted_rungs"] == {BF16_WIRE: len(ROUTED)}
    assert got["mtp_formats_unoffered"] == [R896]
    assert all(BF16_WIRE not in row["name"] for row in got["selection"]["menu"])


def test_the_declaration_composes_with_a_shared_only_pin():
    from prismaquant.glm_mtp_selection import select_mtp_rungs

    payload = _with_bf16_wire(_payload())
    got = select_mtp_rungs(payload, byte_budget=10**12, constants=CONSTANTS,
                           formats=[R1024, R832, "BF16"], fixed_formats={"shared": "BF16"})
    assert got["fixed_formats"] == {"shared": "BF16"}
    assert got["rung_by_group"] == {"routed": "BF16", "shared": "BF16"}
    tight = select_mtp_rungs(payload, byte_budget=_bytes(R1024, "BF16"), constants=CONSTANTS,
                             formats=[R1024, R832, "BF16"], fixed_formats={"shared": "BF16"})
    assert tight["rung_by_group"] == {"routed": R1024, "shared": "BF16"}
    assert "routed" not in tight["fixed_formats"]


@pytest.mark.parametrize("formats, match", [
    ([R832], "empty menu"),                       # shared units priced only at R1024 / BF16
    ([], "at least one format"),
    (["", R1024], "at least one format"),
    (["NOT_A_FORMAT_XYZ"], "unknown format"),
])
def test_an_empty_or_unknown_declaration_refuses(formats, match):
    from prismaquant.glm_mtp_selection import MtpMenuRefused, select_mtp_rungs

    with pytest.raises(MtpMenuRefused, match=match):
        select_mtp_rungs(_payload(), byte_budget=10**12, constants=CONSTANTS, formats=formats)


def test_a_group_left_without_a_complete_rung_refuses():
    from prismaquant.glm_mtp_selection import MtpMenuRefused, select_mtp_rungs

    # Every routed unit is priced at R896. ROUTED[0] lacks R832 and ROUTED[1]
    # lacks R1024, so the open menu selects routed=R896, while a declaration
    # without R896 leaves each unit a rung and the routed group none.
    payload = _payload(drop=(ROUTED[0], R832))
    del payload["costs"][ROUTED[1]][R1024]
    del payload["wire_bytes"][ROUTED[1]][R1024]
    probe = payload["costs"][SHARED[0]][E4M3_SHARED]["probe_identity"]
    for name in ROUTED:
        payload["costs"][name][R896] = _row(name, R896, [0.02, 0.02, 0.02, 0.02], probe)
        payload["wire_bytes"][name][R896] = 7 * 64 * 128 // 16 + 16
    payload["source_dtype"] = {name: "float8_e4m3fn" for name in payload["source_dtype"]}
    opened = select_mtp_rungs(payload, byte_budget=10**12, constants=CONSTANTS)
    assert opened["rung_by_group"]["routed"] == R896
    with pytest.raises(MtpMenuRefused, match="no complete rung"):
        select_mtp_rungs(payload, byte_budget=10**12, constants=CONSTANTS,
                         formats=[R832, R1024])


def _allocator_argv(tmp_path, monkeypatch, publication, *extra):
    from tests.test_allocator_output_pin_1304 import _stock_inputs

    from prismaquant import format_registry

    _allocator_attestation_fixture(monkeypatch)
    argv = [*_stock_inputs(tmp_path), *allowability_cli_args(publication)]
    payload, constants = _write_payload(tmp_path, _payload())
    out = tmp_path / "with-mtp.json"
    argv = [*argv[:argv.index("--layer-config")], "--layer-config", str(out),
            *argv[argv.index("--layer-config") + 2:],
            "--mtp-joint-cost", str(payload), "--mtp-byte-budget", str(10**12),
            "--mtp-serve-constants", str(constants), *extra]
    monkeypatch.setattr(sys, "argv", ["allocator", *argv])
    return out


def test_allocator_without_the_flag_stamps_the_prior_record_keys(tmp_path, monkeypatch, publication):
    from prismaquant import allocator

    out = _allocator_argv(tmp_path, monkeypatch, publication)
    allocator.main()
    record = json.loads(out.read_text())[allocator.LAYER_CONFIG_META_KEY]["mtp_selection"]
    assert not NEW_KEYS & set(record)
    # The fixture's budget admits BF16 passthrough everywhere; the open menu takes it.
    assert record["rung"] == "routed=BF16|shared=BF16"


def test_allocator_flag_restricts_the_menu(tmp_path, monkeypatch, publication):
    from prismaquant import allocator
    from prismaquant import format_registry as fr

    out = _allocator_argv(tmp_path, monkeypatch, publication, "--mtp-formats", f" {R1024} ")
    allocator.main()
    got = json.loads(out.read_text())
    record = got[allocator.LAYER_CONFIG_META_KEY]["mtp_selection"]
    assert record["rung"] == f"routed={R1024}|shared={E4M3_SHARED}"
    assert record["mtp_formats"] == [R1024]
    assert record["menu_restricted_rungs"] == {"BF16": len(ROUTED) + len(SHARED)}
    assert "fixed_formats" not in record
    for name in SHARED:
        assert got[name] == fr.get_format(E4M3_SHARED).autoround_config()


def test_allocator_refuses_an_empty_intersection_with_exit_2(tmp_path, monkeypatch, capsys, publication):
    from prismaquant import allocator

    # R832 is unattested in this fixture: the declared menu reaches no unit.
    out = _allocator_argv(tmp_path, monkeypatch, publication, "--mtp-formats", R832)
    with pytest.raises(SystemExit) as exc:
        allocator.main()
    assert exc.value.code == 2
    # The selector's refusal, not argparse's exit 2 for an unknown flag.
    err = capsys.readouterr().err
    assert "MTP menu (--mtp-formats)" in err and "empty menu" in err
    assert not out.exists()
