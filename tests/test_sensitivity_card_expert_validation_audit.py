"""Packed sensitivity-card arrays must be admitted before AQUA pricing."""
from dataclasses import replace

import numpy as np
import pytest

from prismaquant.sensitivity_card import (
    CardProvenance,
    SensitivityCard,
    SensitivityUnit,
    UnitTopology,
)


def _packed_unit():
    return SensitivityUnit(
        topology=UnitTopology(name="model.layers.0.mlp.experts.down_proj"),
        out_features=3, in_features=4, n_params=24, n_tokens=8,
        h_trace_raw=1.0, h_w2_sum_raw=1.0, w_norm_sq=1.0, w_max_abs=1.0,
        expert_g_sq_sum=np.array([[1.0] * 3, [0.0] * 3], dtype=np.float32),
        expert_act_sq_sum=np.array([[1.0] * 4, [0.0] * 4], dtype=np.float32),
        expert_act_absmax=np.array([[1.0] * 4, [0.0] * 4], dtype=np.float32),
        expert_tokens=np.array([4.0, 0.0], dtype=np.float32),
    )


@pytest.mark.parametrize("with_absmax", [False, True])
def test_valid_packed_card_roundtrip_keeps_unrouted_expert(tmp_path, with_absmax):
    unit = _packed_unit()
    if not with_absmax:
        unit = replace(unit, expert_act_absmax=None)
    provenance = CardProvenance(
        model_id="audit/packed", calib_hash="a" * 64, n_calib_samples=1,
        seq_len=8, probe_commit="b" * 40,
    )
    card = SensitivityCard(provenance, [unit])
    card.validate()
    path = tmp_path / "card.npz"
    card.to_npz(str(path))
    restored = SensitivityCard.from_npz(str(path))
    restored.validate()
    assert restored[unit.topology.name].n_experts == 2
    np.testing.assert_array_equal(
        restored[unit.topology.name].expert_tokens, [4.0, 0.0],
    )


@pytest.mark.parametrize("field", [
    "expert_g_sq_sum", "expert_act_sq_sum", "expert_act_absmax", "expert_tokens",
])
@pytest.mark.parametrize("bad_value", [-1.0, float("nan"), float("inf")])
def test_packed_card_refuses_invalid_array_values(field, bad_value):
    unit = _packed_unit()
    values = getattr(unit, field).copy()
    values.flat[0] = bad_value
    malformed = replace(unit, **{field: values})
    with pytest.raises(ValueError, match=field):
        malformed.validate()


@pytest.mark.parametrize("field,shape", [
    ("expert_g_sq_sum", (2, 4)),
    ("expert_act_sq_sum", (2, 3)),
    ("expert_act_absmax", (3, 4)),
    ("expert_tokens", (3,)),
])
def test_packed_card_refuses_inconsistent_array_geometry(field, shape):
    malformed = replace(_packed_unit(), **{field: np.ones(shape)})
    with pytest.raises(ValueError, match=field):
        malformed.validate()


@pytest.mark.parametrize("field", [
    "expert_g_sq_sum", "expert_act_sq_sum", "expert_tokens",
])
def test_packed_card_refuses_incomplete_required_arrays(field):
    malformed = replace(_packed_unit(), **{field: None})
    with pytest.raises(ValueError, match="expert"):
        malformed.validate()


def test_dense_card_needs_no_expert_arrays():
    dense = replace(
        _packed_unit(), n_params=12, expert_g_sq_sum=None,
        expert_act_sq_sum=None, expert_act_absmax=None, expert_tokens=None,
    )
    dense.validate()


def test_legacy_packed_card_needs_no_expert_arrays():
    legacy = replace(
        _packed_unit(), expert_g_sq_sum=None, expert_act_sq_sum=None,
        expert_act_absmax=None, expert_tokens=None,
    )
    legacy.validate()


@pytest.mark.parametrize("n_params", [0, 23, 36])
def test_expert_arrays_must_match_packed_parameter_count(n_params):
    with pytest.raises(ValueError, match="expert"):
        replace(_packed_unit(), n_params=n_params).validate()


def test_expert_absmax_alone_does_not_admit_partial_statistics():
    partial = replace(
        _packed_unit(), expert_g_sq_sum=None,
        expert_act_sq_sum=None, expert_tokens=None,
    )
    with pytest.raises(ValueError, match="expert"):
        partial.validate()
