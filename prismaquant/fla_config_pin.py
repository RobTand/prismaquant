"""Describe the pinned fla kernel configs that this process uses (#3446).

run-pipeline.sh sets FLA_CACHE_MODE=strict and FLA_CONFIG_DIR to
prismaquant/fla_configs/gb10/. For each autotune key in those files, fla loads
the recorded config instead of timing the candidates, so two runs choose the
same kernel config and agree bitwise. A key that is not in the files still
autotunes. The probe and the AURA cost stage record this identity as data.
"""
from __future__ import annotations

import os
from pathlib import Path

from .digests import canonical_json_sha256, file_sha256hex


def fla_config_identity() -> dict:
    """The fla cache mode, config directory and a SHA-256 over its files."""
    root = os.environ.get("FLA_CONFIG_DIR")
    files = sorted(Path(root).glob("*.json")) if root and Path(root).is_dir() else []
    return {
        "mode": os.environ.get("FLA_CACHE_MODE", "disabled"),
        "dir": root,
        "files": len(files),
        "sha256": canonical_json_sha256(
            {path.name: file_sha256hex(path) for path in files},
            where="fla config set") if files else None,
    }
