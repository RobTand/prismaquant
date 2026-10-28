"""The regret study draws and estimates through the campaign table (C-MIG-01).

``experiments/glm_probe_reduction_regret.py`` reuses the campaign's own
stack-sample draw and Horvitz-Thompson estimator. Both moved to
``prismaquant.tessera_campaign_table``; these tiny inputs exercise the study's
public ``draw`` and ``ht_total`` against that owner. No study data is read.
"""
import importlib.util
import math
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import glm_probe_reduction_regret as regret
from prismaquant import tessera_campaign_table as campaign_table


def test_the_study_samples_through_the_campaign_table():
    assert regret.SAMPLING_SOURCE == "prismaquant.tessera_campaign_table"
    assert regret.TC is campaign_table


def test_draw_and_total_are_the_owner_draw_and_estimator(monkeypatch):
    weights = {str(expert): float(1 + expert) for expert in range(8)}
    draw, used = regret.draw(weights, 4, seed=7, stack="layer3")
    assert used == 4
    assert draw == campaign_table.draw_stack_sample(weights, 4, seed=7, stack="layer3")
    sampled = [int(unit) for unit in draw["units"]]
    pi = {int(k): float(v) for k, v in draw["inclusion_probability"].items() if float(v) > 0}
    y = {expert: 0.5 * (expert + 1) ** 2 for expert in sampled}

    estimator = campaign_table._horvitz_thompson_stack
    seen = []

    def spy(sample, member_dloss):
        seen.append((sample.sampled_experts, dict(sample.inclusion_prob), dict(member_dloss)))
        return estimator(sample, member_dloss)

    monkeypatch.setattr(campaign_table, "_horvitz_thompson_stack", spy)
    total = regret.ht_total(sampled, pi, y)
    assert seen == [(tuple(sampled), pi, y)]
    assert total == pytest.approx(math.fsum(y[e] / pi[e] for e in sampled), rel=1e-12)


def test_a_single_random_draw_is_widened_by_one():
    """One certainty unit and n=2 leaves one random draw, which the owner refuses."""
    weights = {"0": 100.0, "1": 1.0, "2": 1.0, "3": 1.0}
    with pytest.raises(RuntimeError, match="exactly one"):
        campaign_table.draw_stack_sample(weights, 2, seed=0, stack="layer0")
    draw, used = regret.draw(weights, 2, seed=0, stack="layer0")
    assert used == 3
    assert draw == campaign_table.draw_stack_sample(weights, 3, seed=0, stack="layer0")


@pytest.mark.parametrize("entrypoint", ["script", "module"])
def test_population_dotted_layer_owner_preserves_parsed_results(tmp_path, monkeypatch, entrypoint):
    """#2779 / #1303: tiny measured populations, including the existing refusal."""
    from prismaquant import qnames

    owner = qnames.DOTTED_LAYER_QNAME
    searched = []

    def search(name):
        searched.append(name)
        return owner.search(name)

    monkeypatch.setattr(qnames, "DOTTED_LAYER_QNAME", SimpleNamespace(search=search))
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "glm_probe_reduction_regret.py"
    spec = importlib.util.spec_from_file_location("_regret_qname_2779", script)
    consumer = importlib.util.module_from_spec(spec)
    # Dataclasses with postponed annotations need a registered module.
    monkeypatch.setitem(sys.modules, spec.name, consumer)
    spec.loader.exec_module(consumer)
    searched.clear()

    cases = [
        ("model.layers.7.mlp.down_proj", 7, False, "down_proj"),
        ("model.layers.7.mlp.experts.2.gate_proj", 7, True, "gate_proj"),
        ("layers.7.mlp.down_proj", None, False, "down_proj"),
        ("model.layers.7", None, False, "7"),
        ("model.layers.3.layers.19.mlp.down_proj", 3, False, "down_proj"),
        ("model.layers.bad.layers.19.mlp.down_proj", 19, False, "down_proj"),
        ("model.layers.٧.mlp.experts.2.up_proj", 7, True, "up_proj"),
        ("model.embed_tokens", None, False, "embed_tokens"),
        # Unlike the snapshot, the regret loader requires a numeric expert ID.
        ("model.layers.7.mlp.experts.shared.down_proj", 7, False, "down_proj"),
    ]
    family = consumer.EXPERT_FAMILY
    for index, (qname, layer, is_expert, proj) in enumerate(cases):
        cells = {
            f"F_R{rate}": {
                "output_mse_measured": True,
                "cost_source": "tessera_campaign_measured",
                "tessera_provenance": "measured",
                "tessera_family": family,
                "tessera_body_rate_q256": rate,
                "output_mse": 0.5 + k,
                "wire_bytes": 100 + k,
                "encode_seconds": 1.0 + k,
            }
            for k, rate in enumerate(consumer.RATES)
        }
        payload = tmp_path / f"row-{index}.pkl"
        payload.write_bytes(pickle.dumps({"currency": consumer.CURRENCY, "costs": {qname: cells}}))
        census = {"counts": {qname: 11}, "unit_shapes": {qname: [16, 8]}}
        if layer is None:
            # This refactor must not turn the loader's AttributeError into a new policy.
            with pytest.raises(AttributeError, match="group"):
                consumer.build_population([str(payload)], census, lambda message: None)
            continue

        stacks, dense, hashes, ledger = consumer.build_population(
            [str(payload)], census, lambda message: None,
        )
        assert len(hashes) == 1
        cls = "expert" if is_expert else "dense"
        assert ledger == {
            f"{cls}|{family}|{rate}": [1, 1.0 + k]
            for k, rate in enumerate(consumer.RATES)
        }
        if is_expert:
            assert dense == {}
            assert list(stacks) == [layer]
            stack = stacks[layer]
            assert stack.layer == layer
            assert stack.experts == [2]
            assert stack.projections == (proj,)
            np.testing.assert_array_equal(stack.mse, [[[0.5, 1.5, 2.5]]])
            np.testing.assert_array_equal(stack.bytes, [[[100, 101, 102]]])
            np.testing.assert_array_equal(stack.counts, [11])
            assert stack.extra_cells == stack.extra_enc_s == 0
        else:
            assert stacks == {}
            assert list(dense) == [qname]
            unit = dense[qname]
            assert (unit.qname, unit.layer, unit.proj, unit.prefix) == (
                qname, layer, proj, qname.rsplit(".", 1)[0],
            )
            assert unit.shape == (16, 8)
            assert unit.counts == 11
            assert unit.curves == {family: {
                rate: (0.5 + k, 100 + k, 1.0 + k)
                for k, rate in enumerate(consumer.RATES)
            }}

    args = [str(script)] if entrypoint == "script" else ["-m", "experiments.glm_probe_reduction_regret"]
    help_result = subprocess.run(
        [sys.executable, "-E", *args, "--help"], cwd=root,
        capture_output=True, text=True, timeout=30,
    )
    assert help_result.returncode == 0, help_result.stdout + help_result.stderr
    assert "--out" in help_result.stdout
    assert "--rows-limit" in help_result.stdout
    assert searched == [case[0] for case in cases], "population loader bypassed the dotted-layer owner"
