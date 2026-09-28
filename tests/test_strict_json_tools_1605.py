"""Tools strict-JSON loaders and their refusal vocabularies (issue #1605).

Only the host-side ``dsv4_wikitext_inputs`` loader delegates to
``prismaquant/schemas.py::strict_json_loads``. The serving-container tools
(``serve_fingerprint``, ``prismaquant_runtime_snapshot``) and
``container_runtime_identity`` keep local hooks: they run with no installed
package (see ``test_stdlib_tools_no_package_1605.py``). These tmp_path
tests pin the refusal vocabulary per site -- the error type and the message
fragment a caller matches on -- plus one acceptance each, so a refusal
cannot widen or narrow unnoticed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import container_runtime_identity as cri  # noqa: E402
import dsv4_wikitext_inputs as wiki  # noqa: E402
import prismaquant_runtime_snapshot as snap  # noqa: E402
import serve_fingerprint as sf  # noqa: E402

MODEL = "test-model"
CARD = {
    "object": "list",
    "data": [{
        "id": MODEL,
        "object": "model",
        "created": 123,
        "root": "/r",
        "owned_by": "o",
    }],
}


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_pin_loader_accepts_and_refuses(tmp_path: Path) -> None:
    good = _write(tmp_path / "pin.json", '{"a": 1}')
    assert sf._read_tessera_serving_pin_payload(good) == {"a": 1}
    with pytest.raises(ValueError, match="serving pin repeats JSON key"):
        sf._read_tessera_serving_pin_payload(
            _write(tmp_path / "dup.json", '{"a": 1, "a": 2}'))
    with pytest.raises(ValueError, match="not valid JSON"):
        sf._read_tessera_serving_pin_payload(
            _write(tmp_path / "bad.json", '{"a":'))
    with pytest.raises(ValueError, match="must be a JSON object"):
        sf._read_tessera_serving_pin_payload(
            _write(tmp_path / "list.json", '[1, 2]'))
    # No constant= here: NaN still parses as today.
    assert math.isnan(sf._read_tessera_serving_pin_payload(
        _write(tmp_path / "nan.json", '{"a": NaN}'))["a"])


def test_models_endpoint_binding_accepts_and_refuses() -> None:
    bound = sf.models_endpoint_binding_from_bytes(
        json.dumps(CARD).encode(), request_url="http://x/v1",
        expected_served_model=MODEL)
    assert bound["model_count"] == 1
    with pytest.raises(ValueError, match="repeats JSON key"):
        sf.models_endpoint_binding_from_bytes(
            b'{"object": "list", "object": "list", "data": []}',
            request_url="http://x/v1", expected_served_model=MODEL)
    with pytest.raises(ValueError, match="non-finite number"):
        bad = dict(CARD["data"][0], created=float("nan"))
        sf.models_endpoint_binding_from_bytes(
            json.dumps({"object": "list", "data": [bad]}).encode(),
            request_url="http://x/v1", expected_served_model=MODEL)
    with pytest.raises(ValueError, match="did not return valid UTF-8 JSON"):
        sf.models_endpoint_binding_from_bytes(
            b'{"object":', request_url="http://x/v1",
            expected_served_model=MODEL)


def test_snapshot_manifest_accepts_and_refuses(tmp_path: Path) -> None:
    good = _write(tmp_path / "m.json", '{"a": 1}')
    assert snap._load_manifest(good) == {"a": 1}
    with pytest.raises(snap.SnapshotError, match="duplicate manifest member"):
        snap._load_manifest(_write(tmp_path / "d.json", '{"a": 1, "a": 2}'))
    with pytest.raises(snap.SnapshotError, match="non-finite JSON value"):
        snap._load_manifest(_write(tmp_path / "n.json", '{"a": NaN}'))
    with pytest.raises(snap.SnapshotError, match="must be a JSON object"):
        snap._load_manifest(_write(tmp_path / "l.json", '[1]'))


def test_container_identity_accepts_and_refuses(tmp_path: Path) -> None:
    good = _write(tmp_path / "o.json", '{"a": 1}')
    assert cri._load_json_object(good, where="probe") == {"a": 1}
    with pytest.raises(cri.RuntimeIdentityError, match="duplicate JSON member"):
        cri._load_json_object(
            _write(tmp_path / "d.json", '{"a": 1, "a": 2}'), where="probe")
    # No constant= here either: NaN still parses as today.
    assert math.isnan(cri._load_json_object(
        _write(tmp_path / "n.json", '{"a": NaN}'), where="probe")["a"])


def test_wikitext_inputs_accept_and_refuse(tmp_path: Path) -> None:
    good = _write(tmp_path / "w.json", '{"a": 1}')
    assert wiki._strict_json_load(good) == {"a": 1}
    with pytest.raises(
            wiki.DSv4WikiTextInputsError, match="duplicate object member"):
        wiki._strict_json_load(_write(tmp_path / "d.json", '{"a": 1, "a": 2}'))
    with pytest.raises(
            wiki.DSv4WikiTextInputsError, match="non-JSON constant"):
        wiki._strict_json_load(_write(tmp_path / "n.json", '{"a": NaN}'))
