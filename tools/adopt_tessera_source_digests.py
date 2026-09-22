#!/usr/bin/env python3
"""Validate original source proof and publish truthful adopted digest metadata."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prismaquant.tessera_source_digest_adoption import main
if __name__=='__main__':raise SystemExit(main())
