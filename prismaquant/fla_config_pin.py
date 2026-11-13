"""Pin and describe the fla kernel configs that this process uses (#3446).

run-pipeline.sh sets FLA_CACHE_MODE=strict and FLA_CONFIG_DIR to
prismaquant/fla_configs/gb10/. For each autotune key in those files, fla loads
the recorded config instead of timing the candidates, so two runs choose the
same kernel config and agree bitwise. fla's strict mode serves only the
kernels decorated with fla_cache_autotune; pin_fla_configs() fills the cache
of the plain triton.autotune kernels from the same files. A key that is not in
the files still autotunes, and fla_config_identity() records it as data.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

from .digests import canonical_json_sha256, file_sha256hex

#: fla kernels with a plain @triton.autotune, which strict mode does not read.
PLAIN_AUTOTUNED = {
    "layer_norm_gated_fwd_kernel": "fla.modules.fused_norm_gate",
    "layer_norm_gated_bwd_kernel": "fla.modules.fused_norm_gate",
}


def _pinned_entries(root: Path, name: str) -> list[dict]:
    return list(json.loads((root / f"{name}.json").read_text())["autotune_entries"].values())


def pin_fla_configs() -> None:
    """Fill the plain fla autotuners' caches from FLA_CONFIG_DIR (strict mode only)."""
    root = os.environ.get("FLA_CONFIG_DIR")
    if os.environ.get("FLA_CACHE_MODE") != "strict" or not root:
        return
    import triton
    for name, module in PLAIN_AUTOTUNED.items():
        tuner = getattr(importlib.import_module(module), name)
        while not hasattr(tuner, "cache"):  # @triton.heuristics wraps the Autotuner
            tuner = tuner.fn
        for entry in _pinned_entries(Path(root), name):
            c = entry["config"]
            tuner.cache[tuple(entry["autotune_key"])] = triton.Config(
                c["kwargs"], num_warps=c["num_warps"], num_stages=c["num_stages"],
                num_ctas=c["num_ctas"], maxnreg=c.get("maxnreg"))


def _autotuned_keys(files: list[Path]) -> list[dict]:
    """Every Triton autotuner cache entry in this process that is not a pinned key."""
    autotuner = getattr(sys.modules.get("triton.runtime.autotuner"), "Autotuner", None)
    if autotuner is None:
        return []
    pinned = {path.stem: {json.dumps(e["autotune_key"]) for e in _pinned_entries(path.parent, path.stem)}
              for path in files}
    found, seen = [], set()
    for module in list(sys.modules.values()):
        for obj in list(getattr(module, "__dict__", {}).values()):
            for _ in range(3):  # Heuristics -> Autotuner; never chase an unrelated chain
                if obj is None or isinstance(obj, autotuner):
                    break
                obj = obj.fn if type(obj).__name__ == "Heuristics" else None
            if not isinstance(obj, autotuner) or id(obj) in seen:
                continue
            seen.add(id(obj))
            name = obj.base_fn.__name__
            found += [{"kernel": name, "key": list(key)} for key in obj.cache
                      if json.dumps(list(key)) not in pinned.get(name, ())]
    return sorted(found, key=lambda row: (row["kernel"], json.dumps(row["key"])))


def fla_config_identity() -> dict:
    """The fla cache mode, the config directory, a SHA-256 over its files and
    every kernel key that autotuned in this process instead of using a pin."""
    root = os.environ.get("FLA_CONFIG_DIR")
    files = sorted(Path(root).glob("*.json")) if root and Path(root).is_dir() else []
    return {
        "mode": os.environ.get("FLA_CACHE_MODE", "disabled"),
        "dir": root,
        "files": len(files),
        "sha256": canonical_json_sha256(
            {path.name: file_sha256hex(path) for path in files},
            where="fla config set") if files else None,
        "autotuned": _autotuned_keys(files),
    }
