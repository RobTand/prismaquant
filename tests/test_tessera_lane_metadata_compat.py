"""Lane parsing uses Tessera's public metadata API, not serving internals."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from tessera import metadata

from prismaquant import lane_eligibility as lane
from prismaquant.tessera_runtime_contract import contract_path


RECORD = {
    "instrument": "tools/smoke.py",
    "rule": "observed scoring rule",
    "reference": None,
    "rows": [{"prompt": "p", "form": "pure_greedy",
              "interface": "raw_completion", "status": "recorded",
              "reference_status": None}],
}


@pytest.fixture
def published():
    return json.loads(contract_path().read_text(encoding="utf-8"))


def _table(document):
    return lane._parse_table(
        document["lane_eligibility"], document["formats"], "", "", "test",
        native_extensions=document["native_extensions"])


def test_public_record_parser_uses_keyword_arguments(monkeypatch):
    parse = metadata.parse_smoke_record
    calls = []

    def observed(payload, *, control, where):
        calls.append((payload, control, where))
        return parse(payload, control=control, where=where)

    monkeypatch.setattr(metadata, "parse_smoke_record", observed)
    record = copy.deepcopy(RECORD)
    parsed = lane._parse_smoke_record(record, "smoke.record", control=None)
    assert parsed.as_dict() == record
    assert calls == [(record, None, "smoke.record")]
    assert lane._parse_smoke_record(None, "smoke.record", control=None) is None


@pytest.mark.parametrize("updates", [
    {"rows": []},
    {"instrument": "../smoke.py"},
    {"rows": RECORD["rows"] * 2},
    {"rows": [{**RECORD["rows"][0], "status": "unknown"}]},
])
def test_public_record_validation_still_refuses_malformed_records(updates):
    record = {**copy.deepcopy(RECORD), **copy.deepcopy(updates)}
    with pytest.raises(ValueError) as producer:
        metadata.parse_smoke_record(record, control=None, where="smoke.record")
    with pytest.raises(lane.LaneEligibilityError) as consumer:
        lane._parse_smoke_record(record, "smoke.record", control=None)
    assert str(consumer.value) == str(producer.value)
    assert isinstance(consumer.value.__cause__, ValueError)


@pytest.mark.parametrize("version", [None, "unknown"])
def test_incompatible_public_version_refuses(monkeypatch, published, version):
    monkeypatch.setattr(metadata, "METADATA_API_VERSION", version)
    with pytest.raises(lane.LaneEligibilityError, match="incompatible Tessera metadata API"):
        _table(published)


@pytest.mark.parametrize("capability", [
    "parse_smoke_record", "smoke_status_is_derived", "derive_smoke_status",
    "derive_smoke_attribution", "decide_lane_requirements",
])
def test_missing_public_capability_refuses(monkeypatch, published, capability):
    monkeypatch.setattr(metadata, capability, None)
    with pytest.raises(lane.LaneEligibilityError, match="callable.*" + capability):
        _table(published)


def test_missing_public_module_refuses_without_private_fallback(monkeypatch, published):
    monkeypatch.setitem(sys.modules, "tessera.metadata", None)
    with pytest.raises(lane.LaneEligibilityError, match="public metadata API"):
        _table(published)


@pytest.mark.parametrize("derivation,field,alternative", [
    ("derive_smoke_status", "status", "repetitive"),
    ("derive_smoke_attribution", "attribution", "fixture-disagreement"),
])
def test_public_derivation_disagreement_still_refuses(
        monkeypatch, published, derivation, field, alternative):
    smoke = published["lane_eligibility"]["cells"][0]["evidence"]["smoke"]
    assert smoke[field] != alternative
    if field == "status":
        monkeypatch.setattr(metadata, "smoke_status_is_derived", lambda smoke: True)
    monkeypatch.setattr(metadata, derivation, lambda smoke: alternative)
    with pytest.raises(lane.LaneEligibilityError, match=derivation):
        _table(published)


def test_published_table_parses_without_serving_imports(published, tmp_path):
    child = '''
import json
import sys
class ForbiddenRuntime:
    def find_spec(self, name, path=None, target=None):
        if name == 'tessera.serving' or name.startswith('tessera.serving.'):
            raise AssertionError('forbidden runtime import: ' + name)
sys.meta_path.insert(0, ForbiddenRuntime())
from prismaquant import lane_eligibility as lane
document = json.load(sys.stdin)
table = lane._parse_table(document['lane_eligibility'], document['formats'], '', '', 'test',
                          native_extensions=document['native_extensions'])
assert table.present and table.cells
assert not any(name == 'tessera.serving' or name.startswith('tessera.serving.') for name in sys.modules)
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([
        str(Path(metadata.__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parents[1]),
    ])
    result = subprocess.run(
        [sys.executable, "-c", child], input=json.dumps(published), cwd=tmp_path,
        env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
