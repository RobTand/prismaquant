#!/usr/bin/env python3
"""Write selected priced scales from an immutable allocation; never recalibrate."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prismaquant.tessera_selected_scales import main
if __name__ == '__main__':
    raise SystemExit(main())
