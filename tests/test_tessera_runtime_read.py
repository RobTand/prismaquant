"""Contract facts and provenance must describe the same bytes."""
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import tessera_runtime_contract as runtime


def _contract_file(tmp_path):
    raw = runtime.contract_path().read_bytes()
    path = tmp_path / "runtime_contract.json"
    path.write_bytes(raw)
    return path, raw, json.loads(raw)


def test_development_reader_parses_the_bytes_it_hashed(monkeypatch, tmp_path):
    path, raw, payload = _contract_file(tmp_path)
    changed = {**payload, "contract_version": payload["contract_version"] + 1}
    changed_raw = json.dumps(changed).encode()
    original_read = Path.read_bytes

    def replace_after_read(self):
        captured = original_read(self)
        if self == path:
            self.write_bytes(changed_raw)
        return captured

    monkeypatch.setattr(runtime, "contract_path", lambda: path)
    monkeypatch.setenv(runtime.TESSERA_DEV_PIN_ENV, "identity-regression")
    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    parsed = runtime.load_tessera_contract()

    assert original_read(path) == changed_raw
    assert parsed.contract_version == payload["contract_version"]
    assert parsed.sha256 == hashlib.sha256(raw).hexdigest()


def test_development_reader_keeps_answer_equivalent_byte_drift(monkeypatch, tmp_path):
    path, raw, payload = _contract_file(tmp_path)
    changed = {**payload, "contract_version": payload["contract_version"] + 1}
    changed_raw = json.dumps(changed).encode()
    path.write_bytes(changed_raw)
    monkeypatch.setattr(runtime, "contract_path", lambda: path)
    monkeypatch.setenv(runtime.TESSERA_DEV_PIN_ENV, "prose-only-drift")

    parsed = runtime.load_tessera_contract()

    assert parsed.contract_version == changed["contract_version"]
    assert parsed.sha256 == hashlib.sha256(changed_raw).hexdigest()
    assert parsed.sha256 != hashlib.sha256(raw).hexdigest()
    assert runtime.contract_answer(parsed) == runtime.TESSERA_DEV_PIN_ANSWER
    assert parsed.identity()["bytes_are_the_reviewed_bytes"] is False


def test_declared_reader_refuses_replaced_bytes_after_a_cached_read(tmp_path):
    path, raw, payload = _contract_file(tmp_path)
    digest = hashlib.sha256(raw).hexdigest()
    first = runtime.read_tessera_contract(str(path), digest, "fixture")
    changed = {**payload, "contract_version": payload["contract_version"] + 1}
    changed_raw = json.dumps(changed).encode()
    path.write_bytes(changed_raw)

    with pytest.raises(runtime.TesseraContractError, match="declared digest"):
        runtime.read_tessera_contract(str(path), digest, "fixture")

    current = runtime.read_tessera_contract(
        str(path), hashlib.sha256(changed_raw).hexdigest(), "fixture")
    assert first.contract_version == payload["contract_version"]
    assert current.contract_version == changed["contract_version"]
    assert current.sha256 == hashlib.sha256(changed_raw).hexdigest()
