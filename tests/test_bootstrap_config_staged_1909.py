"""The first text-only config input must use declared metadata delivery."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant import sensitivity_probe, staged_whole_file
from prismaquant.residency_map import residency_report
from prismaquant.staged_tier_policy import TierPolicyRefused
from prismaquant.digests import file_sha256hex
from test_streamed_metadata_staged_reads import (
    checkpoint, _activate, _deny_pool_opens,
)
from test_strict_reader_tier_enforcement import _forget_state  # noqa: F401

pytestmark = pytest.mark.own_process


def _config(checkpoint, *, rewrite):
    root, config, _index, manifest = checkpoint
    if rewrite:
        config = dict(config, num_local_experts=2)
        (root / "config.json").write_text(json.dumps(config))
        for entry in manifest["entries"]:
            if entry["path"] == str(root / "config.json"):
                entry["bytes"] = (root / "config.json").stat().st_size
                entry["sha256"] = file_sha256hex(root / "config.json")
    return root, config, manifest


@pytest.mark.parametrize("rewrite", [False, True])
def test_first_config_read_uses_actual_staged_material(checkpoint, tmp_path, monkeypatch, rewrite):
    root, config, manifest = _config(checkpoint, rewrite=rewrite)
    _activate(tmp_path, monkeypatch, manifest)
    _deny_pool_opens(monkeypatch, root)
    result = Path(sensitivity_probe._stage_text_only_impl(
        str(root), staging_root=tmp_path / "work"))
    if rewrite:
        assert result.parent == tmp_path / "work"
        assert result != root
        expected = dict(config, num_experts=2)
        assert json.loads((result / "config.json").read_text()) == expected
    else:
        assert result == root
    report = residency_report()
    assert report is not None and report["fallbacks"] == []


def test_missing_config_binding_refuses_before_bootstrap(checkpoint, tmp_path, monkeypatch):
    root, _config, _index, manifest = checkpoint
    _activate(tmp_path, monkeypatch, manifest, skip={str(root / "config.json")})
    _deny_pool_opens(monkeypatch, root)
    with pytest.raises(TierPolicyRefused):
        sensitivity_probe._stage_text_only_impl(str(root), staging_root=tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_damaged_staged_config_refuses_before_bootstrap(checkpoint, tmp_path, monkeypatch):
    root, _config, _index, manifest = checkpoint
    _activate(tmp_path, monkeypatch, manifest)
    original = staged_whole_file.read_staged_entry

    def damaged(*args, **kwargs):
        return original(*args, **kwargs) + b" "

    monkeypatch.setattr(staged_whole_file, "read_staged_entry", damaged)
    _deny_pool_opens(monkeypatch, root)
    with pytest.raises(TierPolicyRefused, match="digest"):
        sensitivity_probe._stage_text_only_impl(str(root), staging_root=tmp_path / "work")
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("rewrite", [False, True])
def test_inactive_input_keeps_legacy_transforms_and_source_config(checkpoint, tmp_path, rewrite):
    root, config, _manifest = _config(checkpoint, rewrite=rewrite)
    before = (root / "config.json").read_bytes()
    result = Path(sensitivity_probe._stage_text_only_impl(
        str(root), staging_root=tmp_path / "work"))
    if rewrite:
        assert json.loads((result / "config.json").read_text()) == dict(config, num_experts=2)
    else:
        assert result == root
    assert (root / "config.json").read_bytes() == before
