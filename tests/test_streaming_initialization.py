"""CPU metadata contracts; no model forward or source-byte authentication."""

import copy

import pytest

from prismaquant.digests import DIRECT_ASCII_STRICT
from prismaquant.streaming_model import validate_streaming_selected_initialization_witness


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
