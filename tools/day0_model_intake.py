#!/usr/bin/env python3
"""Read source metadata and write a draft; launch a runtime check only on request."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prismaquant_source_bootstrap import activate_prismaquant_source

activate_prismaquant_source()
from prismaquant.day0_model_intake import main

if __name__ == "__main__":
    raise SystemExit(main())
