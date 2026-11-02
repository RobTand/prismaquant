#!/usr/bin/env python3
"""Compatibility command for the unified graph-impact check."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ci.graph_impact_pq import main

if __name__ == '__main__':
    sys.exit(main())
