"""PACT's shape-time scope is verified and read from one captured contract."""
import json

import pytest

from prismaquant import tessera_lane as lane
from prismaquant import tessera_runtime_contract as runtime
from prismaquant.tessera_serving_runtime_pin import TesseraServingRuntimePinError
from prismaquant.tessera_serving_scope import ServingTarget
from tests.tessera_metadata_fixture import install_metadata

TARGET = ServingTarget(platform="sm_121", runtime_image="fixture@sha256:" + "a" * 64,
                       execution_mode="eager", residency="resident")


@pytest.fixture
def payload(monkeypatch):
    monkeypatch.delenv(runtime.TESSERA_DEV_PIN_ENV, raising=False)
    return json.loads(runtime.installed_runtime_snapshot().raw)


def _counted(monkeypatch):
    captured = runtime.installed_runtime_snapshot
    reads = []

    def counted():
        reads.append(1)
        return captured()

    monkeypatch.setattr(runtime, "installed_runtime_snapshot", counted)
    return reads


def test_the_pin_and_the_priced_facts_come_from_one_capture(payload, monkeypatch):
    snapshot = install_metadata(monkeypatch, payload)
    reads = _counted(monkeypatch)

    scope, eligibility, published = lane.allocation_shape_price_scope(
        TARGET, tensor_parallel=1)

    assert len(reads) == 1
    assert scope.contract_sha256 == snapshot.sha256
    assert eligibility.cells == snapshot.eligibility.cells
    assert published == snapshot.published_formats


def test_unpinned_bytes_admit_no_shape_table(payload, monkeypatch):
    install_metadata(monkeypatch, {**payload, "contract_version": 0}, pinned=False)
    with pytest.raises(TesseraServingRuntimePinError):
        lane.allocation_shape_price_scope(TARGET, tensor_parallel=1)


def test_a_missing_contract_is_a_lookup_refusal(monkeypatch):
    def missing():
        raise ModuleNotFoundError("No module named 'tessera'")

    monkeypatch.setattr(runtime, "installed_runtime_snapshot", missing)
    with pytest.raises(LookupError, match="no installed Tessera runtime contract"):
        lane.allocation_shape_price_scope(TARGET, tensor_parallel=1)
