"""One default for the joint prepare's render-file pool width (#1382).

A plan that omits ``file_hash_workers`` made the joint prepare load render
files serially while every other ``prepare_cache`` caller got four workers:
two defaults for one knob. The joint prepare must resolve the omitted key to
the same derivation ``prepare_cache`` itself applies, bounded by the
PB-assigned affinity, and an explicit plan value must still win.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

import prismaquant.tessera_joint_aura as bridge
from prismaquant.tessera_joint_aura import (
    default_file_load_workers,
    prepare_cache,
    resolve_file_hash_workers,
)


def _affinity() -> int:
    return len(os.sched_getaffinity(0))


def test_joint_prepare_default_equals_prepare_cache_default_for_the_environment():
    # prepare_cache owns the width: its signature default IS the derivation
    # (None -> default_file_load_workers()). The joint plan's omitted key
    # resolves to that same width in this same process and environment.
    signature_default = inspect.signature(prepare_cache).parameters[
        "file_load_workers"].default
    prepared_default = (default_file_load_workers() if signature_default is None
                        else min(signature_default, _affinity()))
    assert resolve_file_hash_workers({}) == prepared_default
    # The derived default never exceeds the PB-assigned affinity, so a plan
    # without the key passes the same gate an explicit value must pass.
    assert resolve_file_hash_workers({}) <= _affinity()


@pytest.mark.parametrize("explicit", [1, 2, 3])
def test_explicit_plan_value_still_wins(explicit):
    assert resolve_file_hash_workers({"file_hash_workers": explicit}) == explicit


def test_explicit_value_still_respects_the_affinity_gate():
    with pytest.raises(ValueError, match="exceeds PB-assigned CPU affinity"):
        resolve_file_hash_workers({"file_hash_workers": _affinity() + 1})


@pytest.mark.parametrize("bad", [0, -1, "4", 2.0, True, None])
def test_non_positive_or_non_integer_explicit_value_refuses(bad):
    with pytest.raises(ValueError, match="positive file_hash_workers required"):
        resolve_file_hash_workers({"file_hash_workers": bad})


def test_regenerate_tool_routes_the_plan_key_through_the_resolver(monkeypatch):
    # tools/regenerate_joint_quanta.py feeds its one head intake from the same
    # resolution: an omitted key takes prepare_cache's derivation, an explicit
    # plan value wins.
    import sys
    from types import SimpleNamespace

    import prismaquant.aura_cost as aura
    import prismaquant.joint_stage_b_head as head_module
    root = Path(__file__).resolve().parents[1]
    if str(root / "tools") not in sys.path:
        sys.path.insert(0, str(root / "tools"))
    import regenerate_joint_quanta as tool

    seen = {}

    def intake(_inputs, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(formats_by_qname={}, cells={})

    # The tool imports load_declared_reader from prismaquant.tessera_reader and
    # these plans declare no reader, so load_declared_reader(None) returns None
    # untouched; the intake itself is intercepted on the module the tool
    # imports it from at call time.
    monkeypatch.setattr(bridge, "load_measured_anchor_input", intake)
    monkeypatch.setattr(aura, "_aura_source_sha256", lambda: "0" * 64)
    monkeypatch.setattr(head_module, "build_head_slices", lambda **_kwargs: {})

    def build(plan):
        return tool._build_head_slices(
            plan, plan_sha256="0" * 64, prepared={}, prepared_binding={},
            production_cache=None, layers=[], output_root="unused",
            metadata_root="unused")

    assert build({"inputs": {}}) == {}
    assert seen["file_hash_workers"] == resolve_file_hash_workers({"inputs": {}})
    seen.clear()
    assert build({"inputs": {}, "file_hash_workers": 2}) == {}
    assert seen["file_hash_workers"] == 2
