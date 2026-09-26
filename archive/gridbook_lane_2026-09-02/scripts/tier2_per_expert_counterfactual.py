#!/usr/bin/env python3
"""CLI entry point for the tier-2 per-expert counterfactual."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from prismaquant.tier2_per_expert_counterfactual import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
