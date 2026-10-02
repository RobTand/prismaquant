"""Malformed unit storage numbers must not be coerced into draft budgets."""
import pytest

from prismaquant.glm_mtp_selection import _unit_rows, select_mtp_rungs
from prismaquant.mtp_rung_selection import group_product_menu
from test_glm_mtp_selection import CONSTANTS, ROUTED, R1024, _payload


@pytest.mark.parametrize("bad", [True, 1.5, "4096", -1, 0])
def test_mtp_payload_refuses_invalid_wire_bytes_before_eligibility(bad):
    payload = _payload()
    payload["wire_bytes"][ROUTED[0]][R1024] = bad
    with pytest.raises(ValueError, match="wire_bytes.*integer"):
        _unit_rows(payload, eligible=lambda _name, _fmt: False)


@pytest.mark.parametrize("bad", [True, 1.5, "8192", -1, 0])
def test_mtp_payload_refuses_invalid_parameter_counts(bad):
    payload = _payload()
    payload["params"][ROUTED[0]] = bad
    with pytest.raises(ValueError, match="params.*integer"):
        _unit_rows(payload)


def test_negative_member_bytes_cannot_hide_inside_a_positive_group_total():
    rows = {"a": {"X": (0.1, -1)}, "b": {"X": (0.2, 100)}}
    with pytest.raises(ValueError, match="resident_bytes.*integer"):
        group_product_menu({"g": ["a", "b"]}, rows, params={"a": 4, "b": 4})


@pytest.mark.parametrize("bad", [True, 1.5, "100"])
def test_generic_mtp_menu_refuses_coerced_member_bytes(bad):
    rows = {"a": {"X": (0.1, bad)}}
    with pytest.raises(ValueError, match="resident_bytes.*integer"):
        group_product_menu({"g": ["a"]}, rows, params={"a": 4})


@pytest.mark.parametrize("bad", [True, 1.5, "8", -1, 0])
def test_generic_mtp_menu_requires_each_member_parameter_count(bad):
    rows = {"a": {"X": (0.1, 100)}, "b": {"X": (0.2, 100)}}
    with pytest.raises(ValueError, match="params.*integer"):
        group_product_menu({"g": ["a", "b"]}, rows, params={"a": bad, "b": 100})


@pytest.mark.parametrize("bad", [True, 1.5, "1000000000000", -1])
def test_mtp_selection_refuses_a_coerced_byte_budget(bad):
    with pytest.raises(ValueError, match="byte_budget.*integer"):
        select_mtp_rungs(_payload(), byte_budget=bad, constants=CONSTANTS)
