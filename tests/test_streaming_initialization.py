"""CPU metadata contracts; no model forward or source-byte authentication."""

import copy
import sys

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


# Shared SHA-256 grammar (PQ #3318, part of #3061). The consumer module and
# the digest owner come from sys.modules, loaded by the imports above.
_POLICY = "prismaquant.streaming_initialization"
_FULL = "validate_streaming_initialization_contract"
_PREFIX = "validate_streaming_prefix_initialization_contract"
_SELECTED = "validate_streaming_selected_initialization_witness"
_MTP = "validate_mtp_layer_initialization_contract"


def _full_fixture(selected):
    full = {key: selected[key] for key in (
        "status", "transformers_version", "model_class", "dtype", "layers_prefix",
        "persistent_tensors", "derived_buffers", "state_sha256", "source_map_sha256")}
    full.update(schema="prismaquant.streaming_initialization.v1",
                scope="streamed_text_source_forward", num_layers=2)
    return full


def _prefix_fixture(selected):
    return {**selected, "schema": "prismaquant.streaming_prefix_initialization.v1",
            "scope": "streamed_text_source_prefix"}


def mtp_fixture():
    """A dictionary-only MTP layer witness; no model is instantiated."""
    state = {
        "mtp.0.weight": {"shape": [2, 2], "dtype": "torch.float32", "sha256": "c" * 64},
        "mtp.0.bias": {"shape": [2], "dtype": "torch.float32", "sha256": "d" * 64},
    }
    return {
        "schema": "prismaquant.mtp_layer_initialization.v1",
        "scope": "mtp_layer_checkpoint_load", "status": "completed",
        "transformers_version": "fixture", "model_class": "fixture.Model",
        "dtype": "torch.float32", "layer_prefix": "mtp.0.",
        "experts_implementation": "eager", "checkpoint_tensors": 2, "state": state,
        "state_sha256": DIRECT_ASCII_STRICT.sha256(state), "source_map_sha256": "f" * 64,
        "input": {"schema": "fixture.hidden_states.v1", "path": "hidden.safetensors",
                  "sha256": "e" * 64},
    }


def _fixtures():
    selected = selected_fixture(derived=True)
    return {
        _FULL: _full_fixture(selected),
        _PREFIX: _prefix_fixture(selected),
        _SELECTED: selected,
        _MTP: mtp_fixture(),
    }


def test_initialization_contracts_use_shared_sha256_grammar(monkeypatch):
    policy = sys.modules["prismaquant.streaming_initialization"]
    digests = sys.modules["prismaquant.digests"]
    real = digests.is_sha256hex
    calls = []

    def recording(value):
        calls.append(value)
        return real(value)

    monkeypatch.setattr(digests, "is_sha256hex", recording, raising=False)
    monkeypatch.setattr(policy, "is_sha256hex", recording, raising=False)
    selected = selected_fixture(derived=True)
    prefix = {**selected, "schema": "prismaquant.streaming_prefix_initialization.v1",
              "scope": "streamed_text_source_prefix"}
    full = {key: selected[key] for key in (
        "status", "transformers_version", "model_class", "dtype", "layers_prefix",
        "persistent_tensors", "derived_buffers", "state_sha256", "source_map_sha256")}
    full.update(schema="prismaquant.streaming_initialization.v1",
                scope="streamed_text_source_forward", num_layers=2)
    mtp_state = {
        "mtp.0.weight": {"shape": [2, 2], "dtype": "torch.float32", "sha256": "c" * 64},
        "mtp.0.bias": {"shape": [2], "dtype": "torch.float32", "sha256": "d" * 64},
    }
    mtp = {
        "schema": "prismaquant.mtp_layer_initialization.v1",
        "scope": "mtp_layer_checkpoint_load", "status": "completed",
        "transformers_version": "fixture", "model_class": "fixture.Model",
        "dtype": "torch.float32", "layer_prefix": "mtp.0.",
        "experts_implementation": "eager", "checkpoint_tensors": 2, "state": mtp_state,
        "state_sha256": DIRECT_ASCII_STRICT.sha256(mtp_state), "source_map_sha256": "f" * 64,
        "input": {"schema": "fixture.hidden_states.v1", "path": "hidden.safetensors",
                  "sha256": "e" * 64},
    }
    fixtures = {
        "validate_streaming_initialization_contract": full,
        "validate_streaming_prefix_initialization_contract": prefix,
        "validate_streaming_selected_initialization_witness": selected,
        "validate_mtp_layer_initialization_contract": mtp,
    }
    recorded = {}
    for name, fixture in fixtures.items():
        before = copy.deepcopy(fixture)
        calls.clear()
        assert getattr(policy, name)(fixture) == before
        assert fixture == before
        recorded[name] = list(calls)
    expected = {
        "validate_streaming_initialization_contract": [full["state_sha256"], "b" * 64],
        "validate_streaming_prefix_initialization_contract": ["a" * 64, "b" * 64],
        "validate_streaming_selected_initialization_witness": ["a" * 64, "b" * 64],
        "validate_mtp_layer_initialization_contract": ["f" * 64, "c" * 64, "d" * 64, "e" * 64],
    }
    assert recorded == expected, (
        "streaming initialization validators did not call digests.is_sha256hex "
        "at each SHA-256 grammar site")


class _Hex(str):
    """A str subclass; the grammar accepts it as the existing regex did."""


_FULL_REFUSAL = "Missing or incomplete streaming initialization contract"
_PREFIX_REFUSAL = "Streaming prefix state digest/counts differ"
_SELECTED_REFUSAL = "Streaming selected-layer state digest/counts differ"
_DERIVED_REFUSAL = "Invalid derived-buffer digest"
_MTP_REFUSAL = "Missing or incomplete MTP layer initialization contract"
_MTP_STATE_REFUSAL = "Invalid MTP layer tensor witness"
_MTP_INPUT_REFUSAL = "MTP layer initialization contract has no hash-bound input"
_DERIVED = ("state", "layers.0.weight", "sha256")

# Each touched field, with the refusal its malformed digest raises today.
_SITES = [
    (_FULL, ("state_sha256",), _FULL_REFUSAL, False),
    (_FULL, ("source_map_sha256",), _FULL_REFUSAL, False),
    (_PREFIX, _DERIVED, _DERIVED_REFUSAL, True),
    (_PREFIX, ("source_map_sha256",), _PREFIX_REFUSAL, True),
    (_SELECTED, _DERIVED, _DERIVED_REFUSAL, True),
    (_SELECTED, ("source_map_sha256",), _SELECTED_REFUSAL, True),
    (_MTP, ("source_map_sha256",), _MTP_REFUSAL, False),
    (_MTP, ("state", "mtp.0.weight", "sha256"), _MTP_STATE_REFUSAL, False),
    (_MTP, ("state", "mtp.0.bias", "sha256"), _MTP_STATE_REFUSAL, False),
    (_MTP, ("input", "sha256"), _MTP_INPUT_REFUSAL, False),
]

# (label, value, accepted only where the site coerces with str(...))
_VALUES = [
    ("str-subclass", _Hex("9" * 64), None),
    ("int-coerced-to-64-digits", 10 ** 63, "coerced"),
    ("uppercase", "A" * 64, False),
    ("nonhex", "g" * 64, False),
    ("newline-suffix", "a" * 64 + "\n", False),
    ("short", "a" * 63, False),
    ("long", "a" * 65, False),
    ("empty", "", False),
    ("none", None, False),
    ("bytes", b"a" * 64, False),
    ("list", ["a" * 64], False),
    ("small-int", 7, False),
]


def _mutated(name, path, bad):
    value = _fixtures()[name]
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad
    if path[0] == "state":
        try:
            value["state_sha256"] = DIRECT_ASCII_STRICT.sha256(value["state"])
        except (TypeError, ValueError):
            pass
    return value


@pytest.mark.parametrize("name,path,refusal,coerces", _SITES,
                         ids=[f"{n.split('_', 1)[1]}-{'.'.join(p)}" for n, p, _, _ in _SITES])
@pytest.mark.parametrize("label,bad,accepted", _VALUES, ids=[v[0] for v in _VALUES])
def test_sha256_grammar_sites_keep_outputs_and_refusals(
        name, path, refusal, coerces, label, bad, accepted):
    validate = getattr(sys.modules[_POLICY], name)
    value = _mutated(name, path, bad)
    before = copy.deepcopy(value)
    admits = accepted is None or (accepted == "coerced" and coerces)
    if admits:
        result = validate(value)
        assert result == before and result is not value
    else:
        with pytest.raises(ValueError) as caught:
            validate(value)
        assert str(caught.value) == refusal
    assert value == before


def test_sha256_grammar_valid_outputs_are_copies():
    policy = sys.modules[_POLICY]
    for name, fixture in _fixtures().items():
        before = copy.deepcopy(fixture)
        result = getattr(policy, name)(fixture)
        assert result == before and result is not fixture, name
        assert fixture == before, name
    mtp = mtp_fixture()
    result = policy.validate_mtp_layer_initialization_contract(mtp)
    assert result["input"] is not mtp["input"] and result["state"] is not mtp["state"]


@pytest.mark.parametrize("name,refusal", [
    (_PREFIX, _PREFIX_REFUSAL), (_SELECTED, _SELECTED_REFUSAL)])
def test_state_digest_equality_and_counts_are_kept(name, refusal):
    validate = getattr(sys.modules[_POLICY], name)
    fixture = _fixtures()[name]
    good = fixture["state_sha256"]
    assert validate({**fixture, "state_sha256": _Hex(good)}) == fixture
    for bad in ("0" * 64, good.upper(), good + "\n", None):
        with pytest.raises(ValueError) as caught:
            validate({**fixture, "state_sha256": bad})
        assert str(caught.value) == refusal
    for field, bad in (("persistent_tensors", True), ("derived_buffers", 1.0),
                       ("persistent_tensors", 2), ("derived_buffers", 0)):
        with pytest.raises(ValueError) as caught:
            validate({**fixture, field: bad})
        assert str(caught.value) == refusal


def test_full_and_mtp_type_refusals_are_kept():
    policy = sys.modules[_POLICY]
    full = _fixtures()[_FULL]
    for field, bad in (("num_layers", True), ("persistent_tensors", 0), ("derived_buffers", -1)):
        with pytest.raises(ValueError) as caught:
            policy.validate_streaming_initialization_contract({**full, field: bad})
        assert str(caught.value) == _FULL_REFUSAL
    mtp = mtp_fixture()
    with pytest.raises(ValueError) as caught:
        policy.validate_mtp_layer_initialization_contract({**mtp, "checkpoint_tensors": True})
    assert str(caught.value) == _MTP_REFUSAL
    with pytest.raises(ValueError) as caught:
        policy.validate_mtp_layer_initialization_contract({**mtp, "state_sha256": "0" * 64})
    assert str(caught.value) == "MTP layer state digest differs from its tensor witnesses"


def test_mtp_input_binding_is_kept():
    validate = sys.modules[_POLICY].validate_mtp_layer_initialization_contract
    mtp = mtp_fixture()
    for source in (
        {**mtp["input"], "extra": "x"},
        {key: mtp["input"][key] for key in ("schema", "sha256")},
        {**mtp["input"], "path": ""},
        {**mtp["input"], "schema": 1},
        "e" * 64,
        None,
    ):
        with pytest.raises(ValueError) as caught:
            validate({**mtp, "input": source})
        assert str(caught.value) == _MTP_INPUT_REFUSAL


def test_prefix_selected_and_complete_witnesses_stay_distinct():
    policy = sys.modules[_POLICY]
    fixtures = _fixtures()
    prefix, selected, full = fixtures[_PREFIX], fixtures[_SELECTED], fixtures[_FULL]
    with pytest.raises(ValueError, match="invalid streaming prefix"):
        policy.validate_streaming_prefix_initialization_contract(selected)
    with pytest.raises(ValueError, match="invalid streaming selected-layer witness"):
        policy.validate_streaming_selected_initialization_witness(prefix)
    for witness in (prefix, selected):
        with pytest.raises(ValueError, match="incomplete streaming initialization contract"):
            policy.validate_streaming_initialization_contract(witness)
    with pytest.raises(ValueError, match="incomplete streaming initialization contract"):
        policy.validate_streaming_initialization_contract(
            {**full, "scope": "streamed_text_source_prefix"})
    # A later layer alone is a valid selection but not a prefix.
    state = {"embed_tokens.weight": selected["state"]["embed_tokens.weight"],
             "layers.1.weight": selected["state"]["layers.0.weight"]}
    later = {**selected, "observed_layers": [1], "state": state,
             "state_sha256": DIRECT_ASCII_STRICT.sha256(state)}
    assert policy.validate_streaming_selected_initialization_witness(later) == later
    with pytest.raises(ValueError, match="remain a proper prefix"):
        policy.validate_streaming_prefix_initialization_contract(
            {**later, "schema": prefix["schema"], "scope": prefix["scope"]})
    # Every layer is a valid selection but not a proper prefix.
    state = {**selected["state"],
             "layers.1.weight": {"shape": [1], "dtype": "torch.float32", "kind": "checkpoint"}}
    every = {**selected, "observed_layers": [0, 1], "state": state, "persistent_tensors": 2,
             "state_sha256": DIRECT_ASCII_STRICT.sha256(state)}
    assert policy.validate_streaming_selected_initialization_witness(every) == every
    with pytest.raises(ValueError, match="remain a proper prefix"):
        policy.validate_streaming_prefix_initialization_contract(
            {**every, "schema": prefix["schema"], "scope": prefix["scope"]})
