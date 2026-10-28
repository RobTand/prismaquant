"""Make the imported harness available to its isolated CPU tests."""
import os
from pathlib import Path
import sys

HARNESS = Path(__file__).resolve().parents[2] / "tools" / "g3job"
sys.path.insert(0, str(HARNESS))
os.environ.setdefault("G3_PQ_ROOT", str(HARNESS.parents[1]))
