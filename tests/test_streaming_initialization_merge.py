"""Real CPU audit parity for metadata-only merging, without any forward."""

import copy
import itertools
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prismaquant.digests import DIRECT_ASCII_STRICT
from prismaquant.streaming_model import _StreamingInitializationAudit


class TinyState(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.register_buffer("phase", torch.arange(2, dtype=torch.float32), persistent=False)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = TinyState()
        self.norm = TinyState()
        self.lm_head = TinyState()
        self.layers = nn.ModuleList([TinyState() for _ in range(4)])


def audit_fixture():
    model = TinyModel()
    checkpoint = {name: name for name, _ in model.named_parameters()}
    context = SimpleNamespace(
        model=model, layers_prefix="layers.", num_layers=4, dtype=torch.float32,
        weight_ckpt=checkpoint, weight_shard={name: "toy.safetensors" for name in checkpoint},
        install_resolvers=[{f"layers.{i}.weight": None} for i in range(4)],
    )

    def observed(layers):
        audit = _StreamingInitializationAudit(context)
        for layer in layers:
            audit.observe_layer(layer, context.install_resolvers[layer])
        return audit

    expected = observed(range(4)).complete()
    witnesses = [observed(layers).complete_selected(layers) for layers in ([0], [1, 2], [3])]
    return expected, witnesses, observed


def merge(witnesses, expected):
    # Kept inside the call so RED proves the genuinely missing helper, not
    # an unrelated fixture or dependency collection failure.
    from prismaquant.streaming_initialization import merge_streaming_selected_initialization_witnesses

    return merge_streaming_selected_initialization_witnesses(witnesses, expected_contract=expected)


def reseal(witness):
    state = witness["state"]
    witness["persistent_tensors"] = sum(r["kind"] == "checkpoint" for r in state.values())
    witness["derived_buffers"] = sum(r["kind"] == "derived_buffer" for r in state.values())
    witness["state_sha256"] = DIRECT_ASCII_STRICT.sha256(state)


def test_merge_equals_real_audit_complete_for_every_input_order():
    expected, witnesses, _ = audit_fixture()
    before = copy.deepcopy(witnesses)
    for order in itertools.permutations(witnesses):
        assert merge(order, expected) == expected
    assert witnesses == before


def test_merge_single_complete_selection_and_generator():
    expected, _, observed = audit_fixture()
    witness = observed(range(4)).complete_selected(range(4))
    assert merge(iter([witness]), expected) == expected


@pytest.mark.parametrize("case", ["empty", "gap", "overlap", "noncontiguous", "incomplete"])
def test_merge_refuses_bad_coverage(case):
    expected, witnesses, observed = audit_fixture()
    if case == "empty":
        witnesses = []
    elif case == "gap":
        witnesses.pop(1)
    elif case == "overlap":
        witnesses.append(observed([2, 3]).complete_selected([2, 3]))
    elif case == "noncontiguous":
        witnesses = [observed([0, 2]).complete_selected([0, 2]),
                     observed([1, 3]).complete_selected([1, 3])]
    else:
        witnesses[1]["status"] = "pending"
    with pytest.raises(ValueError):
        merge(witnesses, expected)


@pytest.mark.parametrize("field", ["transformers_version", "model_class", "dtype",
                                   "layers_prefix", "source_map_sha256", "total_model_layers"])
def test_merge_refuses_identity_difference(field):
    expected, witnesses, _ = audit_fixture()
    if field == "total_model_layers":
        witnesses[1][field] += 1
    elif field == "source_map_sha256":
        witnesses[1][field] = "f" * 64
    elif field == "layers_prefix":
        old = witnesses[1][field]
        witnesses[1][field] = "other.layers."
        witnesses[1]["state"] = {
            name.replace(old, "other.layers.") if name.startswith(old) else name: record
            for name, record in witnesses[1]["state"].items()
        }
        reseal(witnesses[1])
    else:
        witnesses[1][field] += "-other"
    with pytest.raises(ValueError):
        merge(witnesses, expected)


@pytest.mark.parametrize("case", ["roster", "shape", "dtype", "kind", "derived_digest"])
def test_merge_refuses_every_head_record_difference(case):
    expected, witnesses, _ = audit_fixture()
    witness = witnesses[1]
    # Change a head other than the first to check the entire roster/record set.
    name = "norm.weight"
    record = witness["state"][name]
    if case == "roster":
        witness["head_state_names"].remove(name)
        del witness["state"][name]
    elif case == "shape":
        record["shape"] = [4]
    elif case == "dtype":
        record["dtype"] = "torch.bfloat16"
    elif case == "kind":
        record.update(kind="derived_buffer", sha256="f" * 64)
    else:
        witness["state"]["norm.phase"]["sha256"] = "f" * 64
    reseal(witness)
    with pytest.raises(ValueError):
        merge(witnesses, expected)


@pytest.mark.parametrize("case", ["missing_body", "state_digest", "body_record", "count_type"])
def test_merge_refuses_state_or_census_difference(case):
    expected, witnesses, _ = audit_fixture()
    witness = witnesses[1]
    if case == "missing_body":
        for name in [name for name in witness["state"] if name.startswith("layers.1.")]:
            del witness["state"][name]
        reseal(witness)
    elif case == "state_digest":
        witness["state_sha256"] = "f" * 64
    elif case == "body_record":
        witness["state"]["layers.1.phase"]["sha256"] = "f" * 64
        reseal(witness)
    else:
        witness["persistent_tensors"] = float(witness["persistent_tensors"])
    with pytest.raises(ValueError):
        merge(witnesses, expected)


@pytest.mark.parametrize("field", ["state_sha256", "persistent_tensors", "derived_buffers",
                                   "source_map_sha256", "num_layers", "status", "schema"])
def test_merge_requires_exact_validated_expected_census(field):
    expected, witnesses, _ = audit_fixture()
    if field in {"persistent_tensors", "derived_buffers", "num_layers"}:
        expected[field] += 1
    elif field in {"state_sha256", "source_map_sha256"}:
        expected[field] = "f" * 64
    else:
        expected[field] = "wrong"
    with pytest.raises(ValueError):
        merge(witnesses, expected)


def test_merge_has_no_unchecked_completion_path():
    from prismaquant.streaming_initialization import merge_streaming_selected_initialization_witnesses

    _, witnesses, _ = audit_fixture()
    with pytest.raises(TypeError):
        merge_streaming_selected_initialization_witnesses(witnesses)
    for expected in (None, [], {}, True):
        with pytest.raises(ValueError):
            merge(witnesses, expected)
