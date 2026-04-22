"""Profile auto-detection + manual registration.

Usage:

    from prismaquant.model_profiles import detect_profile
    profile = detect_profile("/path/to/Qwen3.6-35B-A3B")
    # profile is a Qwen3_5Profile instance.

External architectures can register their own profile at runtime:

    from prismaquant.model_profiles import register_profile, ModelProfile

    class MyArchProfile(ModelProfile):
        ...

    register_profile(MyArchProfile)

Registered profiles are consulted in registration order; the first one
whose `.matches()` returns True wins. `DefaultProfile` is the terminal
fallback when nothing matches.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import ModelProfile
from .default import DefaultProfile
from .gemma4 import Gemma4Profile
from .qwen3_5 import Qwen3_5Profile
from .qwen3_5_dense import Qwen3_5DenseProfile


_REGISTERED: list[type[ModelProfile]] = [
    Qwen3_5DenseProfile,  # must precede Qwen3_5Profile (dense is a subset)
    Qwen3_5Profile,
    Gemma4Profile,
]


def register_profile(cls: type[ModelProfile]) -> None:
    """Register a new ModelProfile subclass for auto-detection.

    Profiles are consulted in registration order. Register earlier than
    built-in profiles to override them."""
    if cls not in _REGISTERED:
        _REGISTERED.insert(0, cls)


def detect_profile(model_path: str) -> ModelProfile:
    """Pick the right ModelProfile for a checkpoint directory.

    Reads `config.json`, walks registered profiles, returns the first
    whose `.matches()` returns True. Falls back to `DefaultProfile` if
    nothing matches."""
    cfg_path = Path(model_path) / "config.json"
    model_type = ""
    archs: list[str] = []
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type") or ""
            archs = list(cfg.get("architectures") or [])
        except (json.JSONDecodeError, OSError):
            pass
    for cls in _REGISTERED:
        try:
            if cls.matches(model_type, archs):
                return cls()
        except Exception:
            continue
    return DefaultProfile(architectures=archs)
