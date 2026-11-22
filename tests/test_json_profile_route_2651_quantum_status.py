from prismaquant import digests, joint_cost_quantum


def test_status_record_uses_direct_ascii_spaced_strict(tmp_path, monkeypatch):
    calls = []
    original = digests.JsonProfile.text

    def recording_text(self, value):
        if self is digests.DIRECT_ASCII_SPACED_STRICT:
            calls.append(value)
        return original(self, value)

    monkeypatch.setattr(digests.JsonProfile, "text", recording_text)
    monkeypatch.setattr(joint_cost_quantum.time, "time", lambda: 123.0)
    record = {
        "quantum_id": "layer-\u00e9",
        "identity_sha256": "a" * 64,
        "output_space": {
            "root": tmp_path,
            "counters": tmp_path / "counters.json",
            "results": tmp_path / "results.json",
        },
    }
    status = joint_cost_quantum.publish_quantum_outputs(
        record, payload=None, result={}, counters={})

    assert (tmp_path / "status.json").read_bytes() == (
        b'{"identity_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        b'"quantum_id": "layer-\\u00e9", "schema": "prismaquant.joint_layer_quantum.status.v1", '
        b'"status": "gapped", "units": [0, 0], "unix": 123.0}\n')
    assert len(calls) == 1
    assert calls[0] is status
