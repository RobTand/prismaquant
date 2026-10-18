"""One captured document supplies identity, lane facts and format facts."""
import hashlib
import json

import pytest

from prismaquant import tessera_runtime_contract as runtime
from prismaquant.lane_eligibility import load_eligibility_table, load_published_formats
from prismaquant.tessera_serving_runtime_pin import TesseraServingRuntimePinError


def test_snapshot_facts_keep_the_captured_document_after_replacement(tmp_path):
    original = runtime.installed_runtime_snapshot()
    path = tmp_path / "runtime_contract.json"
    path.write_bytes(original.raw)
    captured = runtime.read_runtime_snapshot(path)
    changed = json.loads(original.raw)
    changed["lane_eligibility"] = None
    changed["formats"] = []
    replacement = json.dumps(changed).encode()
    path.write_bytes(replacement)

    assert captured.eligibility.cells == original.eligibility.cells
    assert captured.published_formats == original.published_formats
    assert captured.eligibility.contract_sha256 == hashlib.sha256(original.raw).hexdigest()
    current = runtime.read_runtime_snapshot(path)
    assert current.eligibility.present is False
    assert current.published_formats == {}
    assert current.eligibility.contract_sha256 == hashlib.sha256(replacement).hexdigest()


def test_file_adapters_preserve_explicit_version_labels_and_same_facts(tmp_path):
    snapshot = runtime.installed_runtime_snapshot()
    path = tmp_path / "runtime_contract.json"
    path.write_bytes(snapshot.raw)
    table = load_eligibility_table("fixture-version", contract_path=path)
    assert table.runtime_version == "fixture-version"
    assert table.runtime_commit == ""
    assert table.cells == snapshot.eligibility.cells
    assert table.contract_sha256 == snapshot.sha256
    assert load_published_formats(contract_path=path) == snapshot.published_formats


def test_exact_pin_checks_the_snapshot_even_after_the_file_is_restored(tmp_path):
    original = runtime.installed_runtime_snapshot()
    changed = json.loads(original.raw)
    changed["contract_version"] += 1
    path = tmp_path / "runtime_contract.json"
    path.write_text(json.dumps(changed))
    unreviewed = runtime.read_runtime_snapshot(path)
    path.write_bytes(original.raw)

    original.require_exact_pin()
    with pytest.raises(TesseraServingRuntimePinError):
        unreviewed.require_exact_pin()


def test_format_adapter_mutation_does_not_change_the_snapshot(tmp_path):
    original = runtime.installed_runtime_snapshot()
    path = tmp_path / "runtime_contract.json"
    path.write_bytes(original.raw)
    family = next(iter(original.published_formats))
    formats = load_published_formats(contract_path=path)
    formats[family]["family"] = "different"
    formats[family]["reader_rate_range_q256"][0] = 0
    assert runtime.read_runtime_snapshot(path).published_formats[family]["family"] == family
    assert runtime.read_runtime_snapshot(path).published_formats[family][
        "reader_rate_range_q256"] == original.published_formats[family]["reader_rate_range_q256"]


def test_snapshot_format_views_are_owned_by_the_query(tmp_path):
    original = runtime.installed_runtime_snapshot()
    path = tmp_path / "runtime_contract.json"
    path.write_bytes(original.raw)
    captured = runtime.read_runtime_snapshot(path)
    family = next(iter(captured.published_formats))
    view = captured.published_formats
    view[family]["reader_rate_range_q256"][0] = 0

    assert captured.published_formats[family]["reader_rate_range_q256"] == (
        original.published_formats[family]["reader_rate_range_q256"])


def test_missing_file_and_missing_table_remain_distinct(tmp_path):
    path = tmp_path / "runtime_contract.json"
    missing = load_eligibility_table(contract_path=path)
    assert missing.present is False
    assert missing.contract_sha256 == ""
    assert load_published_formats(contract_path=path) == {}
    path.write_text('{"formats": []}')
    absent = load_eligibility_table(contract_path=path)
    assert absent.present is False
    assert absent.contract_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "packages no 'lane_eligibility'" in absent.absent_reason
