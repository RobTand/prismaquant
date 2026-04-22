"""PrismaQuant model profiles — architecture-specific adapters.

Exports:
  - ModelProfile: abstract base class
  - DefaultProfile: generic fallback
  - Qwen3_5Profile: covers Qwen3.5 and Qwen3.6 MoE (w/ MTP)
  - Gemma4Profile: covers Gemma 4 dense + MoE multimodal
  - detect_profile(model_path): auto-detect profile from HF config
  - register_profile(cls): register a custom profile at runtime
"""
from .base import ModelProfile
from .default import DefaultProfile
from .gemma4 import Gemma4Profile
from .qwen3_5 import Qwen3_5Profile
from .qwen3_5_dense import Qwen3_5DenseProfile
from .registry import detect_profile, register_profile

__all__ = [
    "ModelProfile",
    "DefaultProfile",
    "Qwen3_5Profile",
    "Qwen3_5DenseProfile",
    "Gemma4Profile",
    "detect_profile",
    "register_profile",
]
