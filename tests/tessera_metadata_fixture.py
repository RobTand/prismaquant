"""Owned packaged metadata and a parsed matching pin for policy witnesses."""
from copy import deepcopy
import json

from prismaquant import tessera_runtime_contract as runtime
from prismaquant import tessera_serving_runtime_pin as pins


def install_metadata(monkeypatch, payload, *, pinned=True):
    snapshot = runtime.RuntimeContractSnapshot(
        json.dumps(payload).encode(), "owned-policy-metadata")
    monkeypatch.setattr(runtime, "installed_runtime_snapshot", lambda: snapshot)
    monkeypatch.delenv(runtime.TESSERA_DEV_PIN_ENV, raising=False)
    if pinned:
        manifest = json.loads(pins.tessera_serving_runtime_pin_path().read_text())
        manifest["contract_sha256"] = snapshot.sha256
        parsed = pins.parse_tessera_serving_runtime_pin(manifest, where="owned-policy-pin")
        monkeypatch.setattr(pins, "load_tessera_serving_runtime_pin", lambda: deepcopy(parsed))
        monkeypatch.setattr(pins, "TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256", snapshot.sha256)
    return snapshot
