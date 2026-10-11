#!/usr/bin/env python3
"""Reuse the original G3v2 D30 launcher unchanged; only select this workspace."""
import importlib.util
from pathlib import Path
import runpy
import sys
OWNER = Path('/mnt/shared/tessera-measurements/g3-v2-rebaseline-20261005/source')
sys.path.insert(0, str(OWNER))
if len(sys.argv) > 1 and sys.argv[1] == '--container':
    runpy.run_path(str(OWNER/'v2_launch.py'), run_name='__main__')
else:
    spec = importlib.util.spec_from_file_location('stage1_original_v2_launch', OWNER/'v2_launch.py')
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    launch.HERE = Path(__file__).resolve().parent
    raise SystemExit(launch.main())
