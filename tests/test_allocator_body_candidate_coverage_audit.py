"""Mutable body units cannot disappear between probe, admission and emission."""
from __future__ import annotations

import json
import pickle
import re

import pytest

from prismaquant import allocator
from prismaquant.layer_config import load_assignment
from test_allocator_main_enforcement import _write_fixture


def _inputs(tmp_path):
    probe_path, cost_path = _write_fixture(tmp_path)
    output = tmp_path / "layer_config.json"
    argv = [
        "--probe", str(probe_path), "--costs", str(cost_path),
        "--formats", "NVFP4,FP8_E4M3", "--target-bits", "8.0",
        "--pareto-targets", "5.0,8.0", "--layer-config", str(output),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
        "--allow-default-profile",
    ]
    return probe_path, cost_path, output, argv


@pytest.mark.parametrize("gap", ["missing_cost", "all_error", "all_masked"])
def test_real_main_refuses_an_omitted_mutable_body_unit(tmp_path, gap):
    probe_path, cost_path, output, argv = _inputs(tmp_path)
    probe = pickle.loads(probe_path.read_bytes())
    payload = pickle.loads(cost_path.read_bytes())
    name = sorted(probe["stats"])[-1]
    if gap == "missing_cost":
        del payload["costs"][name]
    elif gap == "all_error":
        payload["costs"][name] = {
            fmt: {"error": "measurement failed"}
            for fmt in payload["costs"][name]
        }
    else:
        # Only NVFP4 has a cost, and its group size cannot serve this width.
        payload["costs"][name].pop("FP8_E4M3")
        probe["stats"][name].update(
            n_params=32 * 33, in_features=33, out_features=32, shape=[32, 33])
        probe_path.write_bytes(pickle.dumps(probe))
    cost_path.write_bytes(pickle.dumps(payload))

    with pytest.raises((SystemExit, ValueError),
                       match="(?s)candidate coverage.*" + re.escape(name)):
        allocator.main(argv)
    assert not output.exists()


def test_real_main_refuses_a_body_unit_dropped_by_profile_filter(tmp_path):
    probe_path, cost_path, output, argv = _inputs(tmp_path)
    probe = pickle.loads(probe_path.read_bytes())
    payload = pickle.loads(cost_path.read_bytes())
    name = sorted(probe["stats"])[-1]
    payload["costs"][name] = {
        "FP8_E5M2": payload["costs"][name]["FP8_E4M3"],
    }
    cost_path.write_bytes(pickle.dumps(payload))
    argv[argv.index("--formats") + 1] += ",FP8_E5M2"
    argv += ["--target-profile", "vllm_packed_moe"]

    # Legacy rows carry no in/out dimensions, so serving-profile filtering
    # occurs after candidate construction and aggregation.
    with pytest.raises((SystemExit, ValueError),
                       match="(?s)candidate coverage.*" + re.escape(name)):
        allocator.main(argv)
    assert not output.exists()


@pytest.mark.parametrize("scope", [
    "complete_body", "pinned_head", "uniform_visual", "fixed_mtp", "unpriced_mtp",
])
def test_real_main_preserves_explicit_fixed_and_exempt_scopes(tmp_path, scope):
    probe_path, cost_path, output, argv = _inputs(tmp_path)
    probe = pickle.loads(probe_path.read_bytes())
    payload = pickle.loads(cost_path.read_bytes())
    body_names = set(probe["stats"])
    body_params = sum(row["n_params"] for row in probe["stats"].values())
    extra_name = {
        "pinned_head": "lm_head",
        "uniform_visual": "model.visual.blocks.0.attn.qkv",
        "fixed_mtp": "mtp.layers.0.proj",
        "unpriced_mtp": "mtp.layers.0.proj",
    }.get(scope)
    if extra_name is not None:
        probe["stats"][extra_name] = dict(probe["stats"][sorted(body_names)[0]])
        if scope == "fixed_mtp":
            payload["costs"][extra_name] = {"BF16": {"predicted_dloss": 0.0}}
        probe_path.write_bytes(pickle.dumps(probe))
        cost_path.write_bytes(pickle.dumps(payload))

    allocator.main(argv)
    assignment = load_assignment(output)
    assert body_names <= assignment.keys()
    metadata = json.loads(output.read_text())["__prismaquant__"]
    assert metadata["body_assignment_quantizable_params"] == body_params
    if scope in {"uniform_visual", "fixed_mtp"}:
        assert assignment[extra_name] == "BF16"
    elif extra_name is not None:
        assert extra_name not in assignment
