"""PQ #1747: route unchanged compatibility byte inputs to their digest owner.

Synthetic producer evidence tests CPU control flow only; it does not qualify a
real capture, native graph, image or release input.
"""
import hashlib
import json
from types import SimpleNamespace

import pytest

from prismaquant import glm_capture_compatibility as compatibility
from prismaquant.digests import bytes_sha256hex


@pytest.mark.parametrize("raw", [b"", b"\x00\xffGLM\r\n\xe9\x9b\xaa\n"])
@pytest.mark.parametrize("site", ["_bytes", "_verify_cas_receipt", "_producer", "_issuance_static_inputs"])
@pytest.mark.parametrize("mismatch", [False, True])
def test_each_byte_hash_routes_and_retains_its_refusal(tmp_path, monkeypatch, raw, site, mismatch):
    calls = []

    def owned(value):
        calls.append(value)
        assert bytes_sha256hex(value) == hashlib.sha256(value).hexdigest()
        return bytes_sha256hex(value)

    monkeypatch.setattr(compatibility, "bytes_sha256hex", owned)
    digest = hashlib.sha256(raw).hexdigest()
    wrong = "0" * 64
    source = tmp_path / "modeling.py"
    source.write_bytes(raw)
    binding = dict(path=str(source), sha256=digest)
    snapshot = dict(input="fixture-snapshot", parent=compatibility.CAPTURE_SOURCE)
    sdk = SimpleNamespace(POOL_OUTCOME_SCHEMA_V1="fixture-terminal",
                          cas_receipt_self_check=lambda receipt: None)
    monkeypatch.setattr(compatibility, "client_sdk", lambda: sdk)
    capture = dict(path="fixture-capture", sha256="c" * 64)
    output = ("[campaign] complete streamed calibration capture: " + repr(capture) + "\n" +
              json.dumps(dict(schema="prismaquant.tessera_campaign_container.v1",
                              image_content_sha256=compatibility.ORIGINAL_IMAGE_CONTENT_SHA256,
                              image_id="fixture-image")) + "\n").encode()
    output_path = tmp_path / "output.log"
    output_path.write_bytes(output)
    evidence = {name: dict(path=name, sha256="a" * 64)
                for name in ("request", "terminal", "receipt", "image_inspection")}
    evidence["request"]["sha256"] = compatibility.CAPTURE_REQUEST_SHA256
    evidence["modeling_source"] = binding
    evidence["output"] = dict(path=str(output_path), sha256=hashlib.sha256(output).hexdigest())
    config = tmp_path / "config.json"
    config_binding = dict(path=str(config), sha256="d" * 64)
    records = dict(
        request=dict(action_key=compatibility.CAPTURE_ACTION, params=dict(
            checkout_snapshot=snapshot, command=["python3", "-m", "tools.tessera_campaign_container",
                "--spec", json.dumps(dict(container=dict(
                    content_sha256=compatibility.ORIGINAL_IMAGE_CONTENT_SHA256)))])),
        terminal=dict(action_key=compatibility.CAPTURE_ACTION, schema=sdk.POOL_OUTCOME_SCHEMA_V1,
                      status="executed", detail=dict(returncode=0),
                      resource_scope_cleanup=dict(complete=True), checkout_snapshot=snapshot),
        receipt=dict(action_key=compatibility.CAPTURE_ACTION, producer=dict(inputs=[snapshot["input"]]),
                     result=dict(sha256=hashlib.sha256(output).hexdigest(), bytes=len(output))),
        image_inspection=dict(image=dict(Id="fixture-image")),
        graph=dict(source_final=dict(authenticated=[dict(path=str(config), actual_sha256="d" * 64,
                                                        expected_sha256="d" * 64)])),
    )
    monkeypatch.setattr(compatibility, "bound_json", lambda bound, label: records.get(bound["path"], {}))
    from tools import container_runtime_identity
    monkeypatch.setattr(container_runtime_identity, "image_content_sha256",
                        lambda image: compatibility.ORIGINAL_IMAGE_CONTENT_SHA256)
    monkeypatch.setattr(compatibility, "ORIGINAL_MODELING_SHA256", wrong if mismatch else digest)
    if site == "_bytes":
        binding["sha256"] = wrong if mismatch else digest
        run = lambda: compatibility._bytes(binding, "fixture")
        expected, message = [raw], "fixture bytes changed"
    elif site == "_verify_cas_receipt":
        receipt = dict(producer=dict(inputs=[snapshot["input"]]),
                       result=dict(sha256=wrong if mismatch else digest, bytes=len(raw)))
        run = lambda: compatibility._verify_cas_receipt(receipt, snapshot, raw)
        expected, message = [raw], "original CAS output is not bound by its receipt"
    elif site == "_producer":
        run = lambda: compatibility._producer(evidence, capture)
        expected, message = [output, output, raw, raw], "original modeling source differs"
    else:
        plan = dict(producer=evidence, model_config=config_binding,
                    forward_equivalence=dict(corrected_graph=dict(path="graph")))
        run = lambda: compatibility._issuance_static_inputs(plan)
        expected, message = [raw, raw], "original modeling source differs"
    if mismatch:
        with pytest.raises(ValueError, match=message) as error:
            run()
        assert str(error.value) == "GLM source derivative: " + message
    else:
        result = run()
        if site == "_bytes":
            assert result == raw
        elif site == "_producer":
            assert isinstance(result, dict)
            assert result["modeling_sha256"] == digest and result["source_snapshot"] == snapshot
    assert calls == expected
