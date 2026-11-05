"""A declared native treatment changes only its exact extension residency."""
import copy
import sys

import pytest

from prismaquant import boundary_control as bc
from tools import serve_fingerprint as sf
from test_boundary_control import _binding, _contract, _response


IMAGE = "fixture@sha256:" + "d" * 64


def _declared_pair():
    bindings = {"bf16_control": _binding(), "candidate": _binding("quant")}
    extensions = {"bf16_control": ["sampling.so"],
                  "candidate": ["sampling.so", "trtllm_utils.so"]}
    contract = {"schema": "prismaquant.boundary_stack_contract/1", "image": IMAGE,
                "roles": {role: {"artifact_id": bound["artifact_id"],
                                  "resident_extensions": extensions[role]}
                          for role, bound in bindings.items()}}
    for role, bound in bindings.items():
        manifest = {
            "image": IMAGE, "gpu_name": "GB10", "gpu_uuid": "gpu-1", "gpu_count": 1,
            "driver_version": "595.84", "compute_capability": "12.1",
            "gpu_compute_capabilities": ["12.1"], "package_versions": {"vllm": "pinned"},
            "vllm_compilation_provenance": {"source": "pinned"},
            "resident_extensions": extensions[role], "residency_readable": True,
            "normalized_performance_argv": ["vllm", "serve", "<arm>", "--enforce-eager"],
            "server_process_environment": {"values": {"OMP_NUM_THREADS": "2"}},
            "listener_binding": {"base_url": "http://server", "launch_host": "127.0.0.1",
                                 "launch_port": "8187"},
            "serve_session_id": bound["serve_session_id"],
            "host_identity": {"boot_id": bound["host_boot_id"]},
        }
        manifest["performance_stack_fingerprint"] = sf.performance_stack_fingerprint(manifest)
        bound["serve_fingerprint"] = manifest["performance_stack_fingerprint"]
        bound["paired_stack"] = {"schema": "prismaquant.boundary_stack_binding/1",
                                 "role": role, "contract": copy.deepcopy(contract),
                                 "manifest": manifest}
    return bindings


def _control(monkeypatch, bindings):
    monkeypatch.setattr(bc.vqm, "_post_json", lambda _url, body: _response(body["max_tokens"]))
    return bc.measure_control("http://server", "control", _contract(), bindings["bf16_control"])


def _rehash(binding):
    manifest = binding["paired_stack"]["manifest"]
    manifest["performance_stack_fingerprint"] = sf.performance_stack_fingerprint(manifest)
    binding["serve_fingerprint"] = manifest["performance_stack_fingerprint"]


def test_declared_native_treatment_pairs_with_raw_fingerprints_retained(monkeypatch):
    bindings = _declared_pair()
    assert bindings["bf16_control"]["serve_fingerprint"] != bindings["candidate"]["serve_fingerprint"]
    control = _control(monkeypatch, bindings)
    candidate = bc.measure_candidate("http://server", "candidate", control, bindings["candidate"])
    decision = bc.decide_no_new_failures(control, candidate)
    assert decision["verdict"] == "accepted"
    assert bc.replay_no_new_failures(decision, control, candidate) == decision
    assert control["schema"] == candidate["schema"] == "prismaquant.boundary_control/2"
    assert control["binding"]["serve_fingerprint"] != candidate["binding"]["serve_fingerprint"]


@pytest.mark.parametrize("field,value", [
    ("image", "other@sha256:" + "e" * 64), ("gpu_name", "other"),
    ("gpu_uuid", "gpu-2"), ("gpu_count", 2), ("driver_version", "other"),
    ("compute_capability", "9.0"), ("gpu_compute_capabilities", ["9.0"]),
    ("package_versions", {"vllm": "other"}),
    ("vllm_compilation_provenance", {"source": "other"}),
    ("normalized_performance_argv", ["vllm", "serve", "<arm>"]),
    ("server_process_environment", {"values": {"OMP_NUM_THREADS": "4"}}),
    ("listener_binding", {"base_url": "http://other", "launch_host": "127.0.0.1", "launch_port": "8187"}),
])
def test_declared_residency_never_waives_other_stack_fields(monkeypatch, field, value):
    bindings = _declared_pair()
    control = _control(monkeypatch, bindings)
    candidate = bindings["candidate"]
    candidate["paired_stack"]["manifest"][field] = value
    _rehash(candidate)
    monkeypatch.setattr(bc.vqm, "_post_json", lambda *_args: pytest.fail("unpaired stack reached server"))
    with pytest.raises(ValueError):
        bc.measure_candidate("http://server", "candidate", control, candidate)


@pytest.mark.parametrize("extensions", [["sampling.so"], ["sampling.so", "trtllm_utils.so", "extra.so"]])
def test_exact_role_residency_refuses_missing_and_extra_extensions(monkeypatch, extensions):
    bindings = _declared_pair()
    control = _control(monkeypatch, bindings)
    candidate = bindings["candidate"]
    candidate["paired_stack"]["manifest"]["resident_extensions"] = extensions
    _rehash(candidate)
    with pytest.raises(ValueError, match="resident_extensions"):
        bc.measure_candidate("http://server", "candidate", control, candidate)


def test_raw_fingerprint_is_recomputed_not_taken_from_binding(monkeypatch):
    bindings = _declared_pair()
    control = _control(monkeypatch, bindings)
    candidate = bindings["candidate"]
    candidate["serve_fingerprint"] = control["binding"]["serve_fingerprint"]
    with pytest.raises(ValueError, match="raw.*fingerprint"):
        bc.measure_candidate("http://server", "candidate", control, candidate)


@pytest.mark.parametrize("mutation", ["role", "artifact", "declaration", "missing", "unreadable", "campaign"])
def test_stack_proof_cannot_be_swapped_or_dropped(monkeypatch, mutation):
    bindings = _declared_pair()
    control = _control(monkeypatch, bindings)
    candidate = bindings["candidate"]
    proof = candidate["paired_stack"]
    if mutation == "role":
        proof["role"] = "bf16_control"
    elif mutation == "artifact":
        proof["contract"]["roles"]["candidate"]["artifact_id"] = "a" * 64
    elif mutation == "declaration":
        proof["contract"]["roles"]["bf16_control"]["resident_extensions"] = []
    elif mutation == "missing":
        del candidate["paired_stack"]
    elif mutation == "campaign":
        candidate["campaign_id"] = "another-campaign"
    else:
        proof["manifest"]["residency_readable"] = False
        _rehash(candidate)
    with pytest.raises(ValueError):
        bc.measure_candidate("http://server", "candidate", control, candidate)


def test_declared_receipt_cannot_be_downgraded_to_legacy_schema(monkeypatch):
    control = _control(monkeypatch, _declared_pair())
    control["schema"] = "prismaquant.boundary_control/1"
    with pytest.raises(ValueError, match="schema"):
        bc.replay_control(control)


def test_declared_control_cannot_erase_proof_under_new_schema(monkeypatch):
    control = _control(monkeypatch, _declared_pair())
    del control["binding"]["paired_stack"]
    with pytest.raises(ValueError, match="schema"):
        bc.replay_control(control)


def test_projected_stack_comparison_preserves_json_types(monkeypatch):
    bindings = _declared_pair()
    control = _control(monkeypatch, bindings)
    candidate = bindings["candidate"]
    candidate["paired_stack"]["manifest"]["gpu_count"] = True
    _rehash(candidate)
    with pytest.raises(ValueError, match="gpu_count"):
        bc.measure_candidate("http://server", "candidate", control, candidate)


ROLES = ("bf16_control", "candidate")
CONTENT_REFUSAL = "artifact_content requires model_sha and exact weight content manifest"
ARTIFACT_REFUSAL = "paired stack contract requires exact artifact_id"


class _Text(str):
    """An ordinary str subclass with no overridden behavior."""


ACCEPTED_SHA = [pytest.param("a" * 64, id="lowercase"),
                pytest.param("0123456789abcdef" * 4, id="all-hex-digits"),
                pytest.param(_Text("b" * 64), id="str-subclass")]
REFUSED_SHA = [pytest.param("A" * 64, id="uppercase"),
               pytest.param("a" * 63 + "A", id="one-uppercase"),
               pytest.param("a" * 63, id="short"),
               pytest.param("a" * 65, id="long"),
               pytest.param("", id="empty"),
               pytest.param("a" * 64 + "\n", id="newline-suffix"),
               pytest.param("\n" + "a" * 64, id="newline-prefix"),
               pytest.param("g" * 64, id="non-hex"),
               pytest.param(" " + "a" * 63, id="space"),
               pytest.param(_Text("A" * 64), id="str-subclass-uppercase"),
               pytest.param(b"a" * 64, id="bytes"),
               pytest.param(None, id="none"),
               pytest.param(0, id="zero"),
               pytest.param(int("1" * 64), id="integer")]


def _content(model_sha):
    content = copy.deepcopy(_binding("quant")["artifact_content"])
    content["model_sha"] = model_sha
    return content


def _stack_contract():
    return copy.deepcopy(_declared_pair()["candidate"]["paired_stack"]["contract"])


def _refusal(call, *args):
    with pytest.raises(Exception) as caught:
        call(*args)
    return type(caught.value), str(caught.value)


def test_boundary_declarations_use_shared_sha256_grammar(monkeypatch):
    consumer = sys.modules["prismaquant.boundary_control"]
    owner = sys.modules["prismaquant.digests"]
    real = owner.is_sha256hex
    content = copy.deepcopy(_binding("quant")["artifact_content"])
    contract = copy.deepcopy(_declared_pair()["candidate"]["paired_stack"]["contract"])
    expected_id = consumer.digest(content)
    seen = []

    def recording(value):
        seen.append(value)
        return real(value)

    monkeypatch.setattr(owner, "is_sha256hex", recording, raising=False)
    monkeypatch.setattr(consumer, "is_sha256hex", recording, raising=False)
    assert consumer.artifact_content_id(content) == expected_id
    assert consumer.validate_stack_contract(contract) is contract
    expected = [content["model_sha"]] + [contract["roles"][role]["artifact_id"]
                                         for role in ("bf16_control", "candidate")]
    assert seen == expected, "wrapper did not record the owner calls"


@pytest.mark.parametrize("model_sha", ACCEPTED_SHA)
def test_artifact_content_id_accepts_lowercase_sha256_text(model_sha):
    content = _content(model_sha)
    plain = _content(str(model_sha))
    assert bc.artifact_content_id(content) == bc.digest(plain)
    assert content["model_sha"] is model_sha


@pytest.mark.parametrize("model_sha", REFUSED_SHA)
def test_artifact_content_id_refuses_other_model_sha(model_sha):
    assert _refusal(bc.artifact_content_id, _content(model_sha)) == (ValueError, CONTENT_REFUSAL)


def test_artifact_content_id_checks_model_sha_before_manifest():
    content = _content("A" * 64)
    content["weight_content_manifest"] = {}
    assert _refusal(bc.artifact_content_id, content) == (ValueError, CONTENT_REFUSAL)
    content["model_sha"] = "a" * 64
    assert _refusal(bc.artifact_content_id, content) == (
        ValueError, "invalid closed weight content manifest in boundary artifact_content")


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("artifact_id", ACCEPTED_SHA)
def test_stack_contract_accepts_lowercase_artifact_id_per_role(role, artifact_id):
    contract = _stack_contract()
    contract["roles"][role]["artifact_id"] = artifact_id
    assert bc.validate_stack_contract(contract) is contract
    assert contract["roles"][role]["artifact_id"] is artifact_id


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("artifact_id", REFUSED_SHA)
def test_stack_contract_refuses_other_artifact_id_per_role(role, artifact_id):
    contract = _stack_contract()
    contract["roles"][role]["artifact_id"] = artifact_id
    assert _refusal(bc.validate_stack_contract, contract) == (ValueError, ARTIFACT_REFUSAL)


def test_stack_contract_keeps_field_precedence_for_artifact_id():
    contract = _stack_contract()
    for role in ROLES:
        contract["roles"][role]["artifact_id"] = "A" * 64
    contract["image"] = "fixture@sha256:" + "D" * 64
    assert _refusal(bc.validate_stack_contract, contract) == (
        ValueError, "invalid paired stack contract")
    contract = _stack_contract()
    contract["roles"]["candidate"]["artifact_id"] = None
    contract["roles"]["candidate"]["resident_extensions"] = ["bad name"]
    assert _refusal(bc.validate_stack_contract, contract) == (ValueError, ARTIFACT_REFUSAL)
    contract = _stack_contract()
    contract["roles"]["bf16_control"]["resident_extensions"] = ["bad name"]
    contract["roles"]["candidate"]["artifact_id"] = b"a" * 64
    assert _refusal(bc.validate_stack_contract, contract) == (
        ValueError, "paired resident_extensions must be exact sorted unique basenames")
    contract = _stack_contract()
    contract["roles"]["bf16_control"]["artifact_id"] = "a" * 64
    contract["roles"]["candidate"]["artifact_id"] = _Text("a" * 64)
    assert _refusal(bc.validate_stack_contract, contract) == (
        ValueError, "paired stack contract requires distinct treatment artifacts")
