"""CPU metadata adapter regression; no CUDA preparation or new kernel evidence.

The authentic producer artifact is unchanged. Historical source/capture/wire
acquisition is bounded fixture plumbing. The real acquisition prepare function
must deliver its parsed table and image to the real activation oracle gate.
The native GPU preparation seam stops there; this is no numerical qualification.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import glm_native_dense_acquire as acquire
from prismaquant import native_operator_panel as panel
from prismaquant import tessera_runtime_contract as contract

ARTIFACT = Path(__file__).with_name("glm-dense-authentic-activation-quantizers.json")
ARTIFACT_SHA = "a015090e0110f0654ecc0a7f3860d9e71800f9c5bb58776838f049066c2f5838"
CONTRACT = "e2m1_group16_ue4m3_static"
ACTIVATION = {"quantizes_input": True, "static_contract": {"execution": CONTRACT}}


class OracleReached(Exception):
    def __init__(self, stamp):
        self.stamp = stamp


class AcquisitionTensor:
    """A pre-gate fixture placeholder; never passed into native GPU preparation."""
    def to(self, *_args, **_kwargs):
        return self


def _artifact():
    raw = ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ARTIFACT_SHA
    block = json.loads(raw)
    assert block["platforms"]["sm_121"]["generated"]["image"] == acquire.IMAGE
    return block


def test_authentic_standalone_table_passes_existing_canonical_owner():
    block = _artifact()
    image, rows = contract._parse_activation_platform_entry(
        block["platforms"]["sm_121"], platform="sm_121", where="authentic GLM producer")
    stamp = panel.require_attested_activation_oracle(
        ACTIVATION, platform="sm_121", table={"sm_121": {image: rows}},
        executing_image=acquire.IMAGE)
    assert stamp["generated"]["image"] == acquire.IMAGE
    assert stamp["vectors"] == 11 and stamp["elements"] == 176


def test_acquisition_prepare_preserves_authentic_table_image(monkeypatch, tmp_path):
    import safetensors
    from tessera import cached_unit
    from prismaquant import tessera_calibration_cache as capture_owner
    from prismaquant import tessera_formats, tessera_hessian
    from prismaquant.production_weight_cache import ProductionWeightCache

    _artifact()
    unit = acquire.UNIT
    format_name = "TESSERA_E2M1_K1_R896"
    source = AcquisitionTensor()
    acts = AcquisitionTensor()
    encoding = {"encoder_source_sha256": "a" * 64, "fixture_coordinate": "metadata-only"}
    row = tmp_path / "row"
    wire_root = row / "cache" / "wire"
    wire_root.mkdir(parents=True)
    (wire_root / "bounded-wire.bin").write_bytes(b"metadata-boundary-only")
    record = {"file": "bounded-wire.bin", "identity": dict(encoding)}
    payload = pickle.dumps({"wire_records": {format_name: record}})
    checkpoint = row / "cost.anchors.json.parts" / "units" / (hashlib.sha256(unit.encode()).hexdigest() + ".pkl")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(pickle.dumps({"payload": payload,
        "payload_sha256": hashlib.sha256(payload).hexdigest()}))
    cache_file = tmp_path / "cache.pkl"
    cache = ProductionWeightCache(weights={(unit, format_name): "unused-render.pt"},
        levers=[], activation_max_abs={unit: 1.0})
    cache_file.write_bytes(pickle.dumps(cache))
    census_file = tmp_path / "census.json"
    census_file.write_text(json.dumps({"model": str(tmp_path), "expert_projection": {
        "producer": {"source": {"tensors": {unit + ".weight": "not-opened.safetensors"}}}}}))
    capture = {"identity": {"census_sha256": acquire.sha(census_file),
        "calibration": {"fixture": "metadata-only"}}}
    prepared_file = tmp_path / "prepared.json"
    prepared_file.write_text(json.dumps({"formats_by_qname": {unit: [format_name]},
        "production_cache": {"path": str(cache_file), "sha256": acquire.sha(cache_file)},
        "calibration_input": {"schema": "prismaquant.calibration_input.v1",
            "calibration_sha256": "b" * 64}}))
    monkeypatch.setattr(acquire, "PREPARED", prepared_file)
    monkeypatch.setattr(acquire, "PREPARED_SHA", acquire.sha(prepared_file))
    monkeypatch.setattr(acquire, "CAPTURE", tmp_path / "capture.json")
    monkeypatch.setattr(acquire, "CENSUS", census_file)
    monkeypatch.setattr(acquire, "ROW", row)

    class SourceHandle:
        def __enter__(self):
            return self
        def __exit__(self, *_exc):
            return False
        def get_tensor(self, name):
            assert name == unit + ".weight"
            return source

    monkeypatch.setattr(safetensors, "safe_open", lambda *_args, **_kwargs: SourceHandle())
    monkeypatch.setattr(capture_owner, "require_capture_contract", lambda *_a, **_k: capture)
    monkeypatch.setattr(capture_owner, "prefetch_capture", lambda *_a, **_k:
        (({unit: acts}, {}, {}, {}), None))
    monkeypatch.setattr(tessera_hessian, "activation_source", lambda *_a, **_k: None)
    monkeypatch.setattr(cached_unit, "encoding_input_identity", lambda *_a, **_k: dict(encoding))
    monkeypatch.setattr(tessera_formats, "parse_tessera_format_name", lambda _name:
        (SimpleNamespace(payload_grid=lambda: "metadata-only"), 896))

    def admit_metadata(*_args, activation_quantizers=None, runtime_image="", **_kwargs):
        stamp = panel.require_attested_activation_oracle(
            ACTIVATION, platform="sm_121", table=activation_quantizers,
            executing_image=runtime_image)
        assert stamp["generated"]["image"] == acquire.IMAGE
        raise OracleReached(stamp)

    monkeypatch.setattr(panel, "prepare_native_inputs", admit_metadata)
    args = SimpleNamespace(format=format_name, activation_quantizers=ARTIFACT,
        out=tmp_path / "unpublished")
    with pytest.raises(OracleReached) as result:
        acquire.prepare(args)
    assert result.value.stamp["vectors"] == 11
    assert not args.out.exists()
