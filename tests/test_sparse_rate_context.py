"""Synthetic contracts for contextual sparse-rate surfaces."""
from __future__ import annotations

import numpy as np

from experiments.sparse_rate_context import (
    CONTEXT_MODELS, RHI, RLO, predict_context_surface,
)


RATES = np.asarray([RLO, 896, 960, 1037, RHI], dtype=np.int64)


def _data(layers=12, experts=2):
    rows = []
    for layer in range(layers):
        for expert in range(experts):
            stem = f"model.layers.{layer}.mlp.experts.{expert}"
            support = 1 + layer * 7 + expert
            # The modulo component makes sibling context informative beyond a
            # smooth layer trend, the down endpoints, and token support.
            gate_level = (-6.0 - .03 * layer - .02 * expert
                          + .12 * ((layer + 3 * expert) % 5))
            gate_slope = -2.1 + .01 * layer
            for structure, level, slope in (
                ("gate_proj", gate_level, gate_slope),
                ("up_proj", gate_level - .1, gate_slope + .04),
                ("down_proj", -7.0 - .02 * layer, -2.7 + .015 * layer),
            ):
                values = []
                sibling_signal = gate_level + .7 * gate_slope
                for rate in RATES:
                    t = (rate - RLO) / (RHI - RLO)
                    correction = (.05 + .004 * level - .009 * slope
                                  + .006 * np.log1p(support) + .003 * layer
                                  + .012 * sibling_signal + .02 * t)
                    values.append(np.exp2(level + (t - .5) * slope
                                          + t * (1 - t) * correction))
                rows.append((f"{stem}.{structure}", layer, structure, support, values))
    return {
        "values": np.asarray([row[4] for row in rows]),
        "qnames": np.asarray([row[0] for row in rows]),
        "families": np.asarray(["F"] * len(rows)),
        "activation_contracts": np.asarray(["A"] * len(rows)),
        "roles": np.asarray(["expert"] * len(rows)),
        "structures": np.asarray([row[2] for row in rows]),
        "rows": np.asarray([4096] * len(rows)),
        "cols": np.asarray([2048] * len(rows)),
        "layers": np.asarray([row[1] for row in rows]),
        "counts": np.asarray([row[3] for row in rows]),
    }


def _masks(data):
    down = data["structures"] == "down_proj"
    train = data["layers"] < 10
    target = data["layers"] >= 10
    return train, target, down


def test_context_models_preserve_endpoints_and_predict_arbitrary_interior_rates():
    data = _data()
    train, target, down = _masks(data)
    for model in CONTEXT_MODELS:
        predicted = predict_context_surface(model, data, RATES, train, target, down)
        selected = target & down
        np.testing.assert_array_equal(predicted[selected, 0], data["values"][selected, 0])
        np.testing.assert_array_equal(predicted[selected, -1], data["values"][selected, -1])
        np.testing.assert_allclose(predicted[selected, 1:-1], data["values"][selected, 1:-1],
                                   rtol=2e-3, atol=0)
        assert np.isnan(predicted[~selected]).all()


def test_target_interior_truth_and_sibling_interiors_do_not_enter_prediction():
    data = _data()
    train, target, down = _masks(data)
    baseline = predict_context_surface(
        "context_siblings", data, RATES, train, target, down,
    )
    changed = {key: value.copy() for key, value in data.items()}
    # Change every target interior, including the gate/up siblings.  Endpoint
    # values remain the only permitted target-fold observations.
    changed["values"][target, 1:-1] *= np.asarray([1e8, 1e-8, 1e6])
    replay = predict_context_surface(
        "context_siblings", changed, RATES, train, target, down,
    )
    np.testing.assert_array_equal(replay, baseline)


def test_nonpilot_training_layer_interiors_do_not_expand_the_fit():
    data = _data()
    _train, target, down = _masks(data)
    # Simulate a bounded shared panel: only layers 0..7 are selected pilots;
    # layers 8..9 are development data available to the driver but absent
    # from both the fitting and target masks for this call.
    pilots = data["layers"] < 8
    baseline = predict_context_surface(
        "context_siblings", data, RATES, pilots, target, down,
    )
    changed = {key: value.copy() for key, value in data.items()}
    nonpilot = (data["layers"] >= 8) & (data["layers"] < 10)
    changed["values"][nonpilot, 1:-1] *= np.asarray([1e7, 1e-7, 1e5])
    replay = predict_context_surface(
        "context_siblings", changed, RATES, pilots, target, down,
    )
    np.testing.assert_array_equal(replay, baseline)


def test_sibling_model_reads_endpoint_context_that_self_model_does_not():
    data = _data()
    train, target, down = _masks(data)
    self_before = predict_context_surface("context_self", data, RATES, train, target, down)
    sibling_before = predict_context_surface("context_siblings", data, RATES, train, target, down)
    changed = {key: value.copy() for key, value in data.items()}
    sibling_target = target & ~down
    changed["values"][sibling_target, 0] *= 4.0
    changed["values"][sibling_target, -1] *= .25
    self_after = predict_context_surface("context_self", changed, RATES, train, target, down)
    sibling_after = predict_context_surface("context_siblings", changed, RATES, train, target, down)
    np.testing.assert_array_equal(self_after, self_before)
    assert not np.array_equal(sibling_after, sibling_before)


def test_context_fit_refuses_cross_segment_training():
    data = _data()
    train, target, down = _masks(data)
    mixed = down.copy()
    mixed[np.flatnonzero(data["structures"] == "gate_proj")[0]] = True
    with np.testing.assert_raises_regex(ValueError, "crosses dataset field 'structures'"):
        predict_context_surface("context_self", data, RATES, train, target, mixed)
