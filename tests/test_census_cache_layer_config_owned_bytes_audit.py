"""A checked external assignment must describe the census wires published."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant.tessera_census_cache import census_layer_config
from test_tessera_census_cache import DENSE, _build_tool, _control, _tool_argv


@pytest.fixture
def census_cli(tmp_path):
    names, _, _, cost, roster, shapes, assignment, _ = _control(tmp_path)
    config = census_layer_config(
        cost, assignment, target_profile="glm_packed_research_sm121",
        entry_for=lambda fmt: {"data_type": "tessera", "tessera_format": fmt})
    out = tmp_path / "out"
    argv = _tool_argv(tmp_path, config, cost, roster, shapes)
    argv += ["--out-dir", str(out)]
    path = tmp_path / "layer_config.json"
    return _build_tool(), argv, out, path, config, names


def _replace_after_bound(tool, monkeypatch, path, config):
    real_bound = tool._bound
    observed = []
    changed = copy.deepcopy(config)
    changed[DENSE] = {"data_type": "float", "bits": 16}

    def bound_then_replace(named, digest, label):
        raw = real_bound(named, digest, label)
        if label == "layer config":
            assert Path(named) == path
            assert hashlib.sha256(raw).hexdigest() == digest
            observed.append(raw)
            replacement = path.with_suffix(".replacement.json")
            replacement.write_text(json.dumps(changed) + "\n")
            replacement.replace(path)
        return raw

    monkeypatch.setattr(tool, "_bound", bound_then_replace)
    return observed


def test_census_cache_uses_the_bound_external_assignment(census_cli, monkeypatch):
    tool, argv, out, path, config, names = census_cli
    checked_raw = path.read_bytes()
    observed = _replace_after_bound(tool, monkeypatch, path, config)
    try:
        result = tool.main(argv)
    except (SystemExit, ValueError) as refusal:
        assert observed == [checked_raw]
        if isinstance(refusal, SystemExit):
            assert refusal.code not in (0, None)
        assert "layer config" in str(refusal)
        assert not (out / "manifest.json").exists()
        assert not (out / "summary.json").exists()
        return

    assert result == 0
    assert observed == [checked_raw]
    assert path.read_bytes() != checked_raw
    summary = json.loads((out / "summary.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert summary["sha256"]["layer_config"] == hashlib.sha256(checked_raw).hexdigest()
    assert set(manifest["units"]) == set(names), (
        "the cache publisher attributed a reread BF16 selection to the "
        "digest of the checked all-Tessera assignment")


def test_unchanged_external_assignment_publishes_its_selected_wires(census_cli):
    tool, argv, out, path, _, names = census_cli
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert tool.main(argv) == 0
    summary = json.loads((out / "summary.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert summary["sha256"]["layer_config"] == expected_sha
    assert set(manifest["units"]) == set(names)
    assert summary["export_qualified"] is False
    assert summary["serving_qualified"] is False


def test_optional_plan_projection_keeps_its_changed_config_refusal(
        census_cli, monkeypatch, tmp_path):
    tool, argv, out, path, config, _ = census_cli
    checked_raw = path.read_bytes()
    observed = _replace_after_bound(tool, monkeypatch, path, config)
    projection = tmp_path / "plan" / "layer_config.json"
    with pytest.raises(SystemExit, match="layer config .* changed after it was bound"):
        tool.main([*argv, "--plan-layer-config-out", str(projection)])
    assert observed == [checked_raw]
    assert not projection.exists()
    assert not (out / "manifest.json").exists()
    assert not (out / "summary.json").exists()


def test_assignment_and_metadata_share_the_bound_document(census_cli, monkeypatch):
    from prismaquant.layer_config import LAYER_CONFIG_META_KEY

    tool, argv, out, path, config, names = census_cli
    checked_raw = path.read_bytes()
    real_bound = tool._bound

    def bound_then_replace(named, digest, label):
        raw = real_bound(named, digest, label)
        if label == "layer config":
            changed = copy.deepcopy(config)
            changed[LAYER_CONFIG_META_KEY] = {"unbound_metadata": True}
            path.write_text(json.dumps(changed))
        return raw

    monkeypatch.setattr(tool, "_bound", bound_then_replace)
    assert tool.main(argv) == 0
    summary = json.loads((out / "summary.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert summary["sha256"]["layer_config"] == hashlib.sha256(checked_raw).hexdigest()
    assert summary["layer_config_meta_keys"] == sorted(config[LAYER_CONFIG_META_KEY])
    assert summary["stack_formats"] == config[LAYER_CONFIG_META_KEY]["tessera_expert_stack_formats"]
    assert set(manifest["units"]) == set(names)
