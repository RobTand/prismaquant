import hashlib
import json

from prismaquant import digests, select_validated_frontier as selector
from prismaquant.footprint import assignment_serialization_sha256


def test_resolved_assignment_payload_uses_direct_utf8_strict(tmp_path, monkeypatch):
    source = str(tmp_path / "café.json")
    assignment = {"model.layers.0.self_attn.q_proj": "NVFP4"}
    identity = assignment_serialization_sha256(assignment)
    payload = {
        "schema": "prismaquant.validated_resolved_assignment.v1",
        "source_path": source,
        "assignment": assignment,
        "assignment_sha256": identity,
    }
    expected = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({"results": [{
        "label": "candidate",
        "path": source,
        "bpp": 2.3,
        "last_token_kl": 0.01,
        "assignment_sha256": identity,
        "resolved_assignment_payload": payload,
        "resolved_assignment_payload_sha256": hashlib.sha256(expected).hexdigest(),
    }]}))

    owner_outputs = []
    hash_inputs = []
    real_text = digests.JsonProfile.text
    real_hash = selector.text_sha256hex

    def record_text(self, value):
        text = real_text(self, value)
        if self is digests.DIRECT_UTF8_STRICT and value == payload:
            owner_outputs.append(text)
        return text

    def record_hash(text):
        hash_inputs.append(text)
        return real_hash(text)

    monkeypatch.setattr(digests.JsonProfile, "text", record_text)
    monkeypatch.setattr(selector, "text_sha256hex", record_hash)
    selector.main([
        "--validation-json", str(validation),
        "--mode", "best-kl",
        "--output-layer-config", str(tmp_path / "layer_config.json"),
        "--output-assignment", str(tmp_path / "assignment.json"),
        "--output-summary", str(tmp_path / "summary.json"),
    ])

    assert [text.encode("utf-8") for text in hash_inputs] == [expected]
    assert any(text is hash_inputs[0] for text in owner_outputs), (
        "The selector must hash the text from DIRECT_UTF8_STRICT."
    )
