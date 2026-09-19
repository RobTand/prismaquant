"""The model-wide full-domain ledger: measured vs predicted vs missing (#581).

CPU-only: the legal roster is derived through `tessera_legal_domain`, and
every price class comes from the cost rows' own `cost_source` spelling.
No extrapolation is ever admitted, and no pipeline default moves.
"""
import pytest

from prismaquant.tessera_full_domain_ledger import (
    FullDomainLedgerError,
    build_full_domain_ledger,
    missing_acquisition_work,
    require_full_domain,
)

FAMILY = "TESSERA_E4M3_K1"
CURRENCY = "output_mse_under_route_activation_contract"


def _row(unit, rate, source="tessera_campaign_measured", currency=CURRENCY, **extra):
    return {"unit": unit, "tessera_family": FAMILY, "tessera_body_rate_q256": rate,
            "cost_source": source, "currency": currency, **extra}


def _ledger(rows, **kwargs):
    return build_full_domain_ledger(rows, families=[FAMILY], **kwargs)


def test_measured_rates_are_covered_and_everything_else_is_missing():
    ledger = _ledger([_row("u", 832), _row("u", 960), _row("u", 1088)])
    entry = ledger["entries"]["u|" + FAMILY]
    assert entry["measured_q256"] == [832, 960, 1088]
    assert entry["measured_envelope_q256"] == [832, 1088]
    assert entry["predicted_q256"] == {}
    assert entry["legal_rate_count"] == 1793
    assert entry["missing_rate_count"] == 1793 - 3
    assert entry["complete"] is False


def test_an_interior_interpolation_is_predicted_not_measured():
    ledger = _ledger([_row("u", 832), _row("u", 960), _row("u", 1088),
                      _row("u", 896, source="tessera_campaign_interpolated")])
    entry = ledger["entries"]["u|" + FAMILY]
    assert entry["predicted_q256"] == {"896": "interpolated"}
    assert "896" not in entry["missing_q256"]
    assert 896 not in entry["measured_q256"]


def test_a_transfer_law_row_carries_its_subkind():
    ledger = _ledger([_row("u", 832), _row("u", 960), _row("u", 1088),
                      _row("u", 1024, source="tessera_campaign_interpolated",
                           transfer_law={"slopes": {}})])
    entry = ledger["entries"]["u|" + FAMILY]
    assert entry["predicted_q256"] == {"1024": "transfer_law"}


def test_a_reprice_of_one_rate_refuses():
    with pytest.raises(FullDomainLedgerError, match="reprices"):
        _ledger([_row("u", 960), _row("u", 960, source="tessera_campaign_interpolated")])


def test_a_prediction_outside_the_envelope_is_missing_not_a_price():
    ledger = _ledger([_row("u", 832), _row("u", 960),
                      _row("u", 1088, source="tessera_campaign_interpolated")])
    entry = ledger["entries"]["u|" + FAMILY]
    assert entry["predicted_q256"] == {}
    assert entry["missing_q256"]["1088"] == "predicted_outside_measured_envelope"


def test_a_prediction_with_no_measured_support_is_missing():
    ledger = _ledger([_row("u", 960, source="tessera_campaign_interpolated")])
    entry = ledger["entries"]["u|" + FAMILY]
    assert entry["measured_q256"] == []
    assert entry["predicted_q256"] == {}
    assert entry["missing_q256"]["960"] == "predicted_without_measured_support"


def test_a_complete_unit_passes_require_and_an_incomplete_one_names_itself():
    full = _ledger([_row("u", 832), _row("u", 960)])
    full["entries"]["u|" + FAMILY]["missing_q256"] = {}
    full["entries"]["u|" + FAMILY]["missing_rate_count"] = 0
    full["entries"]["u|" + FAMILY]["complete"] = True
    require_full_domain(full)
    partial = _ledger([_row("u", 832), _row("u", 960)])
    with pytest.raises(FullDomainLedgerError, match="u\\|TESSERA_E4M3_K1"):
        require_full_domain(partial)


def test_require_scopes_to_units_and_families():
    ledger = _ledger([_row("u1", 832), _row("u2", 960)])
    with pytest.raises(FullDomainLedgerError, match="u1\\|"):
        require_full_domain(ledger, units=["u1"])
    with pytest.raises(FullDomainLedgerError, match="u2\\|"):
        require_full_domain(ledger, units=["u2"])


def test_missing_work_names_endpoints_boundaries_and_interiors():
    ledger = _ledger([_row("u", 832), _row("u", 960), _row("u", 1088)])
    work = {(item["rate_q256"], item["reason"]) for item in
            missing_acquisition_work(ledger)}
    entry = ledger["entries"]["u|" + FAMILY]
    legal = entry["legal_q256"]
    assert (legal[0], "missing_domain_endpoint") in work
    assert (legal[-1], "missing_domain_endpoint") in work
    assert any(reason == "missing_recipe_boundary" for _, reason in work)
    assert any(reason == "missing_interior" for _, reason in work)
    assert all(item["unit"] == "u" and item["family"] == FAMILY for item in
               missing_acquisition_work(ledger))


def test_a_foreign_currency_row_refuses():
    with pytest.raises(FullDomainLedgerError, match="currency"):
        _ledger([_row("u", 960, currency="some_other_objective")])


def test_an_unknown_cost_source_refuses():
    with pytest.raises(FullDomainLedgerError, match="unknown cost_source"):
        _ledger([_row("u", 960, source="tessera_campaign_guessed")])


def test_a_price_outside_the_legal_domain_refuses():
    with pytest.raises(FullDomainLedgerError, match="outside the legal domain"):
        _ledger([_row("u", 999999)])


def test_families_are_an_explicit_roster():
    with pytest.raises(FullDomainLedgerError, match="families"):
        build_full_domain_ledger([], families=[])
    ledger = build_full_domain_ledger(
        [_row("u", 960),
         {"unit": "u", "tessera_family": "TESSERA_BF16_K1",
          "tessera_body_rate_q256": 1024,
          "cost_source": "tessera_campaign_measured", "currency": CURRENCY}],
        families=[FAMILY])
    assert sorted(ledger["entries"]) == ["u|" + FAMILY]


def test_stack_sample_measurements_count_as_measured():
    ledger = _ledger([_row("u", 960, source="tessera_campaign_measured_stack_sample")])
    entry = ledger["entries"]["u|" + FAMILY]
    assert entry["measured_q256"] == [960]
