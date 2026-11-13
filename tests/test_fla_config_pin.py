"""The pinned fla kernel configs load in fla's strict format (#3446)."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from prismaquant.fla_config_pin import fla_config_identity

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "prismaquant" / "fla_configs" / "gb10"


def test_every_entry_is_keyed_as_fla_strict_mode_looks_it_up():
    files = sorted(CONFIGS.glob("*.json"))
    assert len(files) == 13
    for path in files:
        data = json.loads(path.read_text())
        assert data["kernel_name"] == path.stem
        for key_hash, entry in data["autotune_entries"].items():
            key = json.dumps(entry["autotune_key"], separators=(",", ":"), sort_keys=True)
            assert key_hash == hashlib.md5(key.encode()).hexdigest()  # AutotuneKey.key_hash
            assert {"kwargs", "num_warps", "num_stages", "num_ctas"} <= entry["config"].keys()


def test_run_pipeline_exports_strict_mode_and_the_repo_config_dir():
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    assert 'export FLA_CACHE_MODE="${FLA_CACHE_MODE:-strict}"' in script
    assert 'export FLA_CONFIG_DIR="${FLA_CONFIG_DIR:-$PIPELINE_SCRIPT_DIR/fla_configs/gb10}"' in script


def test_identity_digest_follows_the_file_contents(tmp_path, monkeypatch):
    copy = tmp_path / "gb10"
    shutil.copytree(CONFIGS, copy)
    monkeypatch.setenv("FLA_CACHE_MODE", "strict")
    monkeypatch.setenv("FLA_CONFIG_DIR", str(copy))
    before = fla_config_identity()
    assert before["mode"] == "strict" and before["files"] == 13 and len(before["sha256"]) == 64
    (copy / "l2norm_fwd_kernel.json").write_text("{}\n")
    assert fla_config_identity()["sha256"] != before["sha256"]
    monkeypatch.delenv("FLA_CONFIG_DIR")
    assert fla_config_identity() == {"mode": "strict", "dir": None, "files": 0, "sha256": None,
                                     "autotuned": []}


def test_plain_autotuned_kernels_have_pins_and_need_strict_mode(monkeypatch):
    from prismaquant import fla_config_pin
    for name in fla_config_pin.PLAIN_AUTOTUNED:
        keys = [e["autotune_key"][:2] for e in fla_config_pin._pinned_entries(CONFIGS, name)]
        assert sorted(keys) == [[128, 1], [128, 24]]
    monkeypatch.delenv("FLA_CACHE_MODE", raising=False)
    monkeypatch.setenv("FLA_CONFIG_DIR", str(CONFIGS))
    fla_config_pin.pin_fla_configs()  # no strict mode: imports nothing, changes nothing
