"""CPU metadata contracts; no model forward or source-byte authentication."""

import copy

import pytest

from prismaquant.digests import DIRECT_ASCII_STRICT
from prismaquant.streaming_initialization import validate_streaming_selected_initialization_witness


def selected_fixture(*, derived=False):
    state = {
        "embed_tokens.weight": {"shape": [2, 2], "dtype": "torch.float32", "kind": "checkpoint"},
        "layers.0.weight": {"shape": [2, 2], "dtype": "torch.float32", "kind": "checkpoint"},
    }
    if derived:
        state["layers.0.weight"] = {"shape": [2], "dtype": "torch.float32",
                                    "kind": "derived_buffer", "sha256": "a" * 64}
    return {
        "schema": "prismaquant.streaming_selected_initialization.v1",
        "scope": "streamed_text_source_selected", "status": "completed",
        "transformers_version": "fixture", "model_class": "fixture.Model",
        "dtype": "torch.float32", "layers_prefix": "layers.", "total_model_layers": 2,
        "observed_layers": [0], "head_state_names": ["embed_tokens.weight"], "state": state,
        "persistent_tensors": 1 if derived else 2, "derived_buffers": 1 if derived else 0,
        "state_sha256": DIRECT_ASCII_STRICT.sha256(state), "source_map_sha256": "b" * 64,
    }


@pytest.mark.parametrize("field,bad,derived", [
    ("persistent_tensors", True, True),
    ("persistent_tensors", 1.0, True),
    ("persistent_tensors", 2.0, False),
    ("derived_buffers", False, False),
    ("derived_buffers", 0.0, False),
    ("derived_buffers", True, True),
    ("derived_buffers", 1.0, True),
    ("persistent_tensors", "2", False),
    ("persistent_tensors", None, False),
    ("persistent_tensors", -1, False),
    ("derived_buffers", "0", False),
    ("derived_buffers", None, False),
    ("derived_buffers", -1, False),
])
def test_selected_counts_require_exact_integers(field, bad, derived):
    witness = selected_fixture(derived=derived)
    witness[field] = bad
    with pytest.raises(ValueError, match="digest/counts differ"):
        validate_streaming_selected_initialization_witness(witness)


@pytest.mark.parametrize("derived", [False, True])
def test_selected_integer_counts_preserve_existing_output(derived):
    witness = selected_fixture(derived=derived)
    before = copy.deepcopy(witness)
    assert validate_streaming_selected_initialization_witness(witness) == before
    assert witness == before


def test_selected_witness_refuses_body_tensor_as_head_coverage():
    witness = selected_fixture()
    del witness["state"]["embed_tokens.weight"]
    witness["head_state_names"] = ["layers.0.weight"]
    witness["persistent_tensors"] = 1
    witness["state_sha256"] = DIRECT_ASCII_STRICT.sha256(witness["state"])
    before = copy.deepcopy(witness)
    with pytest.raises(ValueError, match=r"^Body state cannot stand in for head coverage$"):
        validate_streaming_selected_initialization_witness(witness)
    assert witness == before


def test_existing_full_prefix_selected_imports_and_outputs():
    from prismaquant import streaming_initialization as policy
    from prismaquant import streaming_model as runtime

    assert runtime._initialization_digest is policy._initialization_digest
    assert runtime.DIRECT_ASCII_STRICT is DIRECT_ASCII_STRICT
    selected = selected_fixture()
    prefix = {**selected, "schema": "prismaquant.streaming_prefix_initialization.v1",
              "scope": "streamed_text_source_prefix"}
    full = {key: selected[key] for key in (
        "status", "transformers_version", "model_class", "dtype", "layers_prefix",
        "persistent_tensors", "derived_buffers", "state_sha256", "source_map_sha256")}
    full.update(schema="prismaquant.streaming_initialization.v1",
                scope="streamed_text_source_forward", num_layers=2)
    for name, fixture in (
        ("validate_streaming_initialization_contract", full),
        ("validate_streaming_prefix_initialization_contract", prefix),
        ("validate_streaming_selected_initialization_witness", selected),
    ):
        old_import, shared = getattr(runtime, name), getattr(policy, name)
        assert old_import is shared
        before = copy.deepcopy(fixture)
        assert shared(fixture) == before
        assert fixture == before
        pending = {**fixture, "status": "pending"}
        with pytest.raises(ValueError):
            shared(pending)
        with pytest.raises(ValueError):
            shared({**fixture, "extra": "not in the existing schema"})


def test_initialization_contracts_use_shared_sha256_grammar(monkeypatch):
    import json
    import sys

    policy = sys.modules["prismaquant.streaming_initialization"]
    owner = sys.modules["prismaquant.digests"]
    real = owner.is_sha256hex
    calls = []

    def record(value):
        calls.append(value)
        return real(value)

    monkeypatch.setattr(owner, "is_sha256hex", record, raising=False)
    monkeypatch.setattr(policy, "is_sha256hex", record, raising=False)

    class DigestText(str):
        pass

    selected = selected_fixture(derived=True)
    prefix = {**selected, "schema": "prismaquant.streaming_prefix_initialization.v1",
              "scope": "streamed_text_source_prefix"}
    full = {key: selected[key] for key in (
        "status", "transformers_version", "model_class", "dtype", "layers_prefix",
        "persistent_tensors", "derived_buffers", "state_sha256", "source_map_sha256")}
    full.update(schema="prismaquant.streaming_initialization.v1",
                scope="streamed_text_source_forward", num_layers=2)
    state = {"mtp.weight": {"shape": [1], "dtype": "torch.float32", "sha256": "a" * 64}}
    mtp = {
        "schema": "prismaquant.mtp_layer_initialization.v1",
        "scope": "mtp_layer_checkpoint_load", "status": "completed",
        "transformers_version": "fixture", "model_class": "fixture.Model",
        "dtype": "torch.float32", "layer_prefix": "mtp.", "experts_implementation": "fixture",
        "checkpoint_tensors": 1, "state": state,
        "state_sha256": DIRECT_ASCII_STRICT.sha256(state), "source_map_sha256": "b" * 64,
        "input": {"schema": "fixture", "path": "fixture.json", "sha256": "c" * 64},
    }
    incomplete = "Missing or incomplete streaming initialization contract"
    derived = "Invalid derived-buffer digest"
    cases = (
        (policy.validate_streaming_initialization_contract, full, (
            (("source_map_sha256",), incomplete, False),
            (("state_sha256",), incomplete, False))),
        (policy.validate_streaming_prefix_initialization_contract, prefix, (
            (("source_map_sha256",), "Streaming prefix state digest/counts differ", True),
            (("state", "layers.0.weight", "sha256"), derived, True),
            (("state_sha256",), "Streaming prefix state digest/counts differ", False))),
        (policy.validate_streaming_selected_initialization_witness, selected, (
            (("source_map_sha256",), "Streaming selected-layer state digest/counts differ", True),
            (("state", "layers.0.weight", "sha256"), derived, True),
            (("state_sha256",), "Streaming selected-layer state digest/counts differ", False))),
        (policy.validate_mtp_layer_initialization_contract, mtp, (
            (("source_map_sha256",), "Missing or incomplete MTP layer initialization contract", False),
            (("state", "mtp.weight", "sha256"), "Invalid MTP layer tensor witness", False),
            (("state_sha256",), "MTP layer state digest differs from its tensor witnesses", False),
            (("input", "sha256"), "MTP layer initialization contract has no hash-bound input", False))),
    )
    values = (None, 17, 1.5, b"a" * 64, "a" * 63, "a" * 65,
              "A" * 64, "g" * 64, "a" * 64 + "\n", DigestText("a" * 64),
              int("1" * 64), "a" * 64)
    routed = []
    for validate, fixture, sites in cases:
        start = len(calls)
        before = copy.deepcopy(fixture)
        result = validate(fixture)
        assert result == before
        assert result is not fixture
        assert fixture == before
        if validate is policy.validate_mtp_layer_initialization_contract:
            assert result["state"] is not fixture["state"]
        elif "state" in fixture:
            assert result["state"] is fixture["state"]
        for path, message, coerces in sites:
            for value in values:
                candidate = copy.deepcopy(fixture)
                target = candidate
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                accepts = isinstance(value, str) and value == "a" * 64
                accepts |= coerces and type(value) is int and value == int("1" * 64)
                if path[0] == "state" and accepts:
                    candidate["state_sha256"] = DIRECT_ASCII_STRICT.sha256(candidate["state"])
                if path == ("state_sha256",) and validate is not policy.validate_streaming_initialization_contract:
                    accepts = False
                before = copy.deepcopy(candidate)
                if accepts:
                    expected = json.loads(json.dumps(before)) if fixture is mtp else before
                    assert validate(candidate) == expected
                else:
                    with pytest.raises(ValueError) as caught:
                        validate(candidate)
                    assert type(caught.value) is ValueError
                    assert str(caught.value) == message
                assert candidate == before
        routed.append((validate.__name__, len(calls) - start))
    for name, count in routed:
        assert count > 0, f"The shared SHA-256 predicate received no calls from {name}."
