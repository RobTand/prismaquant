"""Synthetic numerical contracts for the sparse-rate research predictors."""
from __future__ import annotations

import numpy as np

from experiments.sparse_rate_models import (
    CURRENCY, ONE_MODELS, RANCHOR, RHI, RLO, SCHEMA, TWO_MODELS,
    one_predict, two_predict, validate_choice, validate_dataset,
)


RATES = np.asarray([RLO, 896, RANCHOR, 1024, RHI], dtype=np.int64)
LO, MID, HI = 0, 2, 4


def _complete_dataset_fixture():
    """Small complete manifest/data pair accepted by the provenance gate."""
    rates = np.asarray([RLO, RANCHOR, RHI], dtype=np.int64)
    data = {
        "values": np.asarray([[.5, .25, .125], [.4, .2, .1]], dtype=np.float64),
        "rates": rates,
        "qnames": np.asarray(["model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"]),
        "families": np.asarray(["TESSERA_E4M3_K1", "TESSERA_E4M3_K1"]),
        "activation_contracts": np.asarray(["fp8", "fp8"]),
        "layers": np.asarray([0, 1], dtype=np.int64),
        "roles": np.asarray(["dense", "dense"]),
        "structures": np.asarray(["down_proj", "down_proj"]),
        "rows": np.asarray([8, 8], dtype=np.int64),
        "cols": np.asarray([4, 4], dtype=np.int64),
        "counts": np.asarray([2, 2], dtype=np.int64),
        "wire_bytes": np.asarray([[10, 11, 12], [10, 11, 12]], dtype=np.int64),
        "encode_seconds": np.asarray([[1., 2., 3.], [1., 2., 3.]], dtype=np.float64),
    }
    manifest = {
        "schema": "prismaquant.sparse_rate_dataset.v1", "currency": CURRENCY,
        "research_only": True, "not_joint_aura": True, "not_serving_admission": True,
        "arrays": {name: {"dtype": str(array.dtype), "shape": list(array.shape)}
                   for name, array in data.items()},
    }
    identity = {"dataset_npz_sha256": "a" * 64, "dataset_manifest_sha256": "b" * 64,
                "script_sha256": "c" * 64}
    return data, manifest, identity


def _endpoint_population(n=16):
    """Positive curves with enough independent training observations for fits."""
    values = np.empty((n, len(RATES)), dtype=np.float64)
    for unit in range(n):
        left, right = 2.0 ** (-8 - unit / 20), 2.0 ** (-11 - unit / 25)
        for index, rate in enumerate(RATES):
            t = (rate - RLO) / (RHI - RLO)
            # The shared quadratic log correction is zero at exact anchors.
            values[unit, index] = 2.0 ** ((1 - t) * np.log2(left) + t * np.log2(right)
                                         + .17 * t * (1 - t))
    return values


def test_two_anchor_models_preserve_observed_endpoints_exactly():
    values = _endpoint_population()
    train = np.arange(len(values)) < 12
    target = ~train

    for model in ("log_chord", "value_chord", "log_constant", "log_ridge2", "log_surface2", "exp4"):
        prediction = two_predict(model, values, RATES, train, target)
        np.testing.assert_array_equal(prediction[target, LO], values[target, LO])
        np.testing.assert_array_equal(prediction[target, HI], values[target, HI])


def test_one_anchor_models_preserve_the_observed_middle_anchor_exactly():
    values = _endpoint_population()
    train = np.arange(len(values)) < 12
    target = ~train
    counts = np.arange(1, len(values) + 1, dtype=np.int64)

    for model in ("one_constant", "one_ridge1", "one_ridge2"):
        prediction = one_predict(model, values, RATES, train, target, counts)
        np.testing.assert_array_equal(prediction[target, MID], values[target, MID])


def test_affine_exponential_with_floor_matches_endpoint_conditioned_exp_template():
    beta, floor, amplitude = 3, 0.125, 7.0
    values = np.empty((13, len(RATES)), dtype=np.float64)
    for unit in range(len(values)):
        # This underlying function is independent of the endpoint-conditioned
        # interpolation algebra.  It has a nonzero activation-like floor.
        unit_floor, unit_amplitude = floor * (unit + 1), amplitude * (unit + 2)
        for index, rate in enumerate(RATES):
            t = (rate - RLO) / (RHI - RLO)
            values[unit, index] = unit_floor + unit_amplitude * 2.0 ** (-beta * t)
    train = np.arange(len(values)) < 12
    target = ~train

    prediction = two_predict("exp3", values, RATES, train, target)

    np.testing.assert_allclose(prediction[target], values[target], rtol=5e-15, atol=0)


def test_endpoint_conditioned_surface_transfers_to_heldout_unit_at_arbitrary_interior_rate():
    rates = np.asarray([RLO, 864, 901, RANCHOR, 1037, 1056, RHI], dtype=np.int64)
    values = np.empty((16, len(rates)), dtype=np.float64)
    for unit in range(len(values)):
        left, right = 2.0 ** (-7 - unit / 10), 2.0 ** (-10 - unit / 12)
        logleft, slope = np.log2(left), np.log2(right) - np.log2(left)
        for index, rate in enumerate(rates):
            t = (rate - RLO) / (RHI - RLO)
            # Exactly the surface feature basis used by log_surface1.
            correction = .07 + .011 * logleft - .019 * slope + .031 * t
            values[unit, index] = 2.0 ** (logleft + t * slope + t * (1 - t) * correction)
    train = np.arange(len(values)) < 15
    target = ~train
    # The queried rates are audit-only for the held-out unit.  The surface fit
    # gets t=.125/.5/.875 from other units, so t dependence is identifiable
    # without reading the query columns.
    values[train, 2] = np.nan
    values[train, 4] = np.nan

    prediction = two_predict("log_surface1", values, rates, train, target)

    np.testing.assert_allclose(prediction[target, 2], values[target, 2], rtol=2e-4, atol=0)
    np.testing.assert_allclose(prediction[target, 4], values[target, 4], rtol=2e-4, atol=0)


def test_withheld_layer_truth_does_not_enter_the_two_anchor_fit():
    values = _endpoint_population()
    train = np.arange(len(values)) < 12
    target = ~train
    baseline = two_predict("log_ridge2", values, RATES, train, target)
    mutated = values.copy()
    # These are the held-out layer's audit values, and changing them must not
    # change its prediction.  Its endpoints remain the allowed anchors.
    mutated[target, 1:4] *= np.asarray([1e9, 1e-9, 1e7])
    replay = two_predict("log_ridge2", mutated, RATES, train, target)

    np.testing.assert_array_equal(replay, baseline)


def test_missing_required_anchor_produces_no_prediction_for_that_target_unit():
    values = _endpoint_population(14)
    train = np.arange(len(values)) < 12
    target = ~train
    values[12, LO] = np.nan
    values[13, MID] = np.nan
    counts = np.ones(len(values), dtype=np.int64)

    two = two_predict("log_chord", values, RATES, train, target)
    one = one_predict("one_constant", values, RATES, train, target, counts)

    assert np.isnan(two[12]).all()
    assert np.isfinite(two[13, LO]) and np.isfinite(two[13, HI])
    assert np.isnan(one[13]).all()
    assert np.isfinite(one[12, MID])


def test_validate_choice_rejects_a_stale_script_identity_and_unknown_model():
    _, _, identity = _complete_dataset_fixture()
    choice = {"schema": SCHEMA, "identity": dict(identity),
              "two_anchor": TWO_MODELS[0], "one_anchor": ONE_MODELS[0]}
    stale = {**choice, "identity": {**identity, "script_sha256": "d" * 64}}
    with np.testing.assert_raises_regex(ValueError, "frozen choice"):
        validate_choice(stale, identity)
    unknown = {**choice, "two_anchor": "made_up_model"}
    with np.testing.assert_raises_regex(ValueError, "frozen choice"):
        validate_choice(unknown, identity)


def test_validate_dataset_rejects_wrong_currency_and_malformed_array_shape():
    data, manifest, _ = _complete_dataset_fixture()
    validate_dataset(data, manifest)

    wrong_currency = {**manifest, "currency": "different_currency"}
    with np.testing.assert_raises_regex(ValueError, "dataset contract"):
        validate_dataset(data, wrong_currency)
    malformed = dict(data)
    malformed["encode_seconds"] = data["encode_seconds"][:, :2]
    with np.testing.assert_raises_regex(ValueError, "dimensions|shape"):
        validate_dataset(malformed, manifest)
