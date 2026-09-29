"""Split-pin provenance must not silently disappear from legal-domain drift."""
from __future__ import annotations

from dataclasses import replace

import pytest

from prismaquant import tessera_legal_domain as domain
from prismaquant import tessera_runtime_contract as producer_pin
from prismaquant import tessera_serving_runtime_pin as serving_pin


LEGACY_KEYS = {
    "reader_dev_pin_commit",
    "reader_dev_pin_contract_sha256",
    "serving_runtime_pinned_commit",
    "serving_runtime_pinned_version",
    "serving_runtime_pinned_contract_sha256",
    "producer_installed_contract_sha256",
}
PRODUCER_FIELD = "serving_runtime_pinned_producer_commit"
SOURCE_FIELD = "serving_runtime_pinned_serving_source_sha256"


def _split_pins():
    return replace(
        domain.FROZEN_PINS,
        serving_runtime_pinned_producer_commit="1" * 40,
        serving_runtime_pinned_serving_source_sha256="2" * 64,
    )


def test_current_v2_frozen_transcription_keeps_exactly_the_legacy_fields():
    assert set(domain.FROZEN_PINS.as_dict()) == LEGACY_KEYS
    report = domain.pin_drift(domain.FROZEN_PINS, domain.FROZEN_PINS)
    assert report["matches"] is True
    assert report["differences"] == {}
    assert report["frozen"] == report["live"] == domain.FROZEN_PINS.as_dict()


def test_legacy_drift_preserves_the_existing_serialization_order():
    frozen = domain.FROZEN_PINS
    live = replace(
        frozen,
        reader_dev_pin_commit="b" * 40,
        serving_runtime_pinned_commit="c" * 40,
        producer_installed_contract_sha256="d" * 64,
    )
    report = domain.pin_drift(frozen, live)
    assert list(report["frozen"]) == list(report["live"]) == [
        "reader_dev_pin_commit",
        "reader_dev_pin_contract_sha256",
        "serving_runtime_pinned_commit",
        "serving_runtime_pinned_version",
        "serving_runtime_pinned_contract_sha256",
        "producer_installed_contract_sha256",
    ]
    assert list(report["differences"]) == [
        "reader_dev_pin_commit",
        "serving_runtime_pinned_commit",
        "producer_installed_contract_sha256",
    ]


def test_split_transcription_retains_both_independently_reviewed_values():
    pins = _split_pins()
    fields = pins.as_dict()
    assert set(fields) == LEGACY_KEYS | {PRODUCER_FIELD, SOURCE_FIELD}
    assert fields[PRODUCER_FIELD] == "1" * 40
    assert fields[SOURCE_FIELD] == "2" * 64
    report = domain.pin_drift(pins, pins)
    assert report["matches"] is True
    assert report["differences"] == {}


@pytest.mark.parametrize(
    "field, value",
    [(PRODUCER_FIELD, "3" * 40), (SOURCE_FIELD, "4" * 64)],
)
def test_each_split_field_is_drift_even_when_all_legacy_values_match(field, value):
    frozen = _split_pins()
    live = replace(frozen, **{field: value})
    report = domain.pin_drift(frozen, live)
    assert report["matches"] is False
    assert report["differences"] == {
        field: {"frozen": frozen.as_dict()[field], "live": value},
    }
    assert field in report["verdict"]


@pytest.mark.parametrize("activate", [True, False])
def test_v2_v3_transition_is_symmetric_explicit_drift_not_a_key_error(activate):
    legacy = domain.FROZEN_PINS
    split = _split_pins()
    frozen, live = (legacy, split) if activate else (split, legacy)
    report = domain.pin_drift(frozen, live)
    assert report["matches"] is False
    assert set(report["differences"]) == {PRODUCER_FIELD, SOURCE_FIELD}
    for field in (PRODUCER_FIELD, SOURCE_FIELD):
        assert report["differences"][field] == {
            "frozen": frozen.as_dict().get(field),
            "live": live.as_dict().get(field),
        }


@pytest.mark.parametrize("field", [PRODUCER_FIELD, SOURCE_FIELD])
def test_a_missing_split_member_is_reported_independently(field):
    frozen = _split_pins()
    live = replace(frozen, **{field: None})
    report = domain.pin_drift(frozen, live)
    assert report["matches"] is False
    assert report["differences"] == {
        field: {"frozen": frozen.as_dict()[field], "live": None},
    }


def test_live_v2_pins_do_not_add_split_keys_or_change_existing_values(monkeypatch):
    monkeypatch.setattr(
        serving_pin, "TESSERA_SERVING_RUNTIME_PINNED_SERVING_SOURCE_SHA256", None,
    )
    monkeypatch.setattr(
        serving_pin, "installed_tessera_contract_sha256", lambda: "5" * 64,
    )
    fields = domain.live_pins().as_dict()
    assert fields == {
        "reader_dev_pin_commit": producer_pin.TESSERA_DEV_PIN_COMMIT,
        "reader_dev_pin_contract_sha256": producer_pin.TESSERA_DEV_PIN_CONTRACT_SHA256,
        "serving_runtime_pinned_commit": serving_pin.TESSERA_SERVING_RUNTIME_PINNED_COMMIT,
        "serving_runtime_pinned_version": serving_pin.TESSERA_SERVING_RUNTIME_PINNED_VERSION,
        "serving_runtime_pinned_contract_sha256": serving_pin.TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256,
        "producer_installed_contract_sha256": "5" * 64,
    }


def test_live_split_values_are_read_from_their_owners_not_aliased(monkeypatch):
    monkeypatch.setattr(
        producer_pin, "TESSERA_DEV_PIN_COMMIT", "6" * 40,
    )
    monkeypatch.setattr(
        serving_pin, "TESSERA_SERVING_RUNTIME_PINNED_COMMIT", "7" * 40,
    )
    monkeypatch.setattr(
        serving_pin, "TESSERA_SERVING_RUNTIME_PINNED_PRODUCER_COMMIT", "8" * 40,
    )
    monkeypatch.setattr(
        serving_pin, "TESSERA_SERVING_RUNTIME_PINNED_SERVING_SOURCE_SHA256", "9" * 64,
    )
    monkeypatch.setattr(
        serving_pin, "installed_tessera_contract_sha256", lambda: "a" * 64,
    )
    fields = domain.live_pins().as_dict()
    assert fields["reader_dev_pin_commit"] == "6" * 40
    assert fields["serving_runtime_pinned_commit"] == "7" * 40
    assert fields[PRODUCER_FIELD] == "8" * 40
    assert fields[SOURCE_FIELD] == "9" * 64
    assert fields["producer_installed_contract_sha256"] == "a" * 64
    assert domain.FROZEN_PINS.as_dict().keys() == LEGACY_KEYS
