"""Real resident intake must preserve each configured expert projection.

The source is a tiny BF16 Qwen3.5-MoE checkpoint; no reader, packer, install path,
memory admission or initialization witness is replaced.
"""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant import streaming_model as owner
from tests.test_resident_source_decode_audit import (
    _check_values, _cpu_guard, _load, _source,
)


@pytest.mark.parametrize("offset_rows", [False, True],
                         ids=["configured-gate-up", "compensating-gate-up-rows"])
def test_resident_refuses_compensating_expert_projection_rows(
    tmp_path, monkeypatch, offset_rows,
):
    root, expected = _source(tmp_path, native=False)
    parent = next(name for name in sorted(expected)
                  if name.endswith(".experts.gate_up_proj"))
    prefix = parent.rsplit(".", 1)[0]
    gate_key = f"{prefix}.0.gate_proj.weight"
    up_key = f"{prefix}.0.up_proj.weight"
    configured_rows = expected[parent].shape[1] // 2
    assert configured_rows > 1
    if offset_rows:
        tensors = load_file(root / "model.safetensors")
        gate, up = tensors[gate_key], tensors[up_key]
        assert gate.shape == up.shape == (configured_rows, expected[parent].shape[2])
        assert not torch.equal(gate[0], up[0])
        tensors[gate_key] = torch.cat((gate, gate[:1]), dim=0)
        tensors[up_key] = up[1:].clone()
        assert tensors[gate_key].shape[0] + tensors[up_key].shape[0] == 2 * configured_rows
        replacement = root / "model.replacement.safetensors"
        save_file(tensors, replacement)
        replacement.replace(root / "model.safetensors")
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in tensors},
        }))
        del tensors, gate, up

    payload_calls = []
    materialize = owner._materialize

    def observe_payload(*args, **kwargs):
        payload_calls.append(True)
        return materialize(*args, **kwargs)

    monkeypatch.setattr(owner, "_materialize", observe_payload)
    guard = _cpu_guard(tmp_path, monkeypatch)
    if offset_rows:
        with pytest.raises(ValueError, match="shape|rows"):
            _load(root, tmp_path, guard)
        assert not payload_calls, "configured source geometry must refuse before payload allocation"
        assert guard.last is None
    else:
        model = _load(root, tmp_path, guard)
        _check_values(model, expected, torch.bfloat16, False)
        assert payload_calls
        assert guard.last["label"] == "resident_source:complete"


@pytest.mark.parametrize("rows", [(4, 4, 4), (5, 4, 3), (5, 4, 4)],
                         ids=["three-equal-roles", "compensating-three-roles", "indivisible-live-rows"])
def test_packer_checks_each_declared_role_before_consuming_sources(rows):
    from prismaquant.layer_streaming import _pack_per_expert_into_packed

    projections = ("alpha", "beta", "gamma")
    source = {f"experts.0.{role}.weight": torch.full((height, 7), float(index))
              for index, (role, height) in enumerate(zip(projections, rows))}
    original = dict(source)
    arguments = dict(
        is_per_expert=lambda name: name.startswith("experts.0."),
        parent_for_projection=lambda role: "packed" if role in projections else None,
        projection_names_for=lambda parent: projections,
        live_param_shape={"experts.packed": (1, sum(rows), 7)}.get,
    )
    if len(set(rows)) > 1:
        with pytest.raises(ValueError, match="shape|rows"):
            _pack_per_expert_into_packed(source, **arguments)
        assert source.keys() == original.keys()
        assert all(source[key] is value for key, value in original.items())
    else:
        assert _pack_per_expert_into_packed(source, **arguments) == 1
        assert set(source) == {"experts.packed"}
        assert torch.equal(source["experts.packed"][0], torch.cat(list(original.values())))
