"""Calibration-time observer classes for prismaquant.

Lightweight forward-hook-based measurement utilities that plug into the
probe pipeline to collect signals beyond weight sensitivity. Unlike the
main `sensitivity_probe` Fisher accumulator, observers here do not
modify backward passes and never touch weight gradients — they are
purely forward-pass measurements.

Current observers:
  - `ExpertSaliencyTracker` — REAP-style router-weighted expert
    activation pruning saliency. Feeds per-expert DROP candidates into
    the allocator.
"""

from .expert_saliency import ExpertSaliencyTracker, saliency_from_moe_structure

__all__ = ["ExpertSaliencyTracker", "saliency_from_moe_structure"]
