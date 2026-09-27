"""Original cached cohorts join without changing either child's receipt."""

import hashlib
import json

import pytest

from tessera.cached_unit import CACHE_SCHEMA, COMPOSED_CACHE_SCHEMA
from tools.compose_tessera_cached_units import compose
from prismaquant import tessera_export_lane as export


def _child(tmp_path, name, seal):
    root = tmp_path / name
    root.mkdir()
    path = root / "cached_units.json"
    source = {"config_sha256": "c" * 64, "auxiliary_sha256": {},
              "files": {"model.safetensors": "f" * 64},
              "tensors": {name + ".weight": "model.safetensors"}}
    record = {"file": name + ".tessera", "blob_sha256": "b" * 64,
              "blob_bytes": 8, "identity": {"unit": name,
              "encoder_source_sha256": seal}}
    path.write_text(json.dumps({"schema": CACHE_SCHEMA, "source": source,
                                "units": {name: record}}))
    return path, source, record


def _descriptor(path, seal):
    return {"manifest": {"path": str(path),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            "producer_package": {"path": str(path.parent), "sha256": seal}}


def test_composer_keeps_child_receipts_and_original_producer_seals(tmp_path):
    body, source, original_body = _child(tmp_path, "body", "a" * 64)
    mtp, _, original_mtp = _child(tmp_path, "mtp", "d" * 64)
    # The two closed children must agree on the whole source, even though
    # their selected unit rosters are disjoint.
    mtp_payload = json.loads(mtp.read_text())
    mtp_payload["source"] = source
    mtp.write_text(json.dumps(mtp_payload))
    parent, bundle = compose([_descriptor(body, "a" * 64),
                              _descriptor(mtp, "d" * 64)],
                             output=tmp_path / "combined.json")
    assert parent["schema"] == COMPOSED_CACHE_SCHEMA
    assert bundle.units == {"body": original_body, "mtp": original_mtp}
    assert set(bundle.producer_packages) == {"a" * 64, "d" * 64}
    assert json.loads(body.read_text())["units"]["body"] == original_body
    assert json.loads(mtp.read_text())["units"]["mtp"] == original_mtp

    output = tmp_path / "combined.json"
    output.write_text(json.dumps(parent))
    scope = {"by_unit": {"body": {}, "mtp": {}},
             "expert_projection": {"source": source, "units": {"body": original_body}},
             "mtp_expert_projection": {"source": source}}
    metadata = {"mtp_selection": {"mtp_expert_wires": {"mtp": original_mtp}}}
    accepted = export.require_composed_cached_units(output, scope=scope, metadata=metadata)
    assert accepted["units"] == 2
    wrong = json.loads(json.dumps(metadata))
    wrong["mtp_selection"]["mtp_expert_wires"]["mtp"]["blob_sha256"] = "0" * 64
    with pytest.raises(export.TesseraExportLaneError, match="MTP receipt differs"):
        export.require_composed_cached_units(output, scope=scope, metadata=wrong)

    mtp.write_text(mtp.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        compose(parent["children"], output=tmp_path / "other.json")


def test_composer_refuses_duplicate_child_and_missing_original_package(tmp_path):
    body, _, _ = _child(tmp_path, "body", "a" * 64)
    descriptor = _descriptor(body, "a" * 64)
    with pytest.raises(ValueError, match="overlap"):
        compose([descriptor, descriptor], output=tmp_path / "combined.json")
    mtp, source, _ = _child(tmp_path, "mtp", "d" * 64)
    mtp_payload = json.loads(mtp.read_text())
    mtp_payload["source"] = json.loads(body.read_text())["source"]
    mtp.write_text(json.dumps(mtp_payload))
    missing = _descriptor(mtp, "d" * 64)
    missing["producer_package"] = None
    with pytest.raises(ValueError, match="historical producer package"):
        compose([descriptor, missing], output=tmp_path / "combined.json")


def test_preflight_refuses_mtp_selection_without_composed_cache(tmp_path):
    assignment = tmp_path / "layer_config.json"
    assignment.write_text(json.dumps({"__prismaquant__": {"mtp_selection": {
        "mtp_expert_wires": {"mtp": {}}}}}))
    with pytest.raises(export.TesseraExportLaneError, match="no encode fallback"):
        export.preflight(tmp_path, assignment_path=assignment)
