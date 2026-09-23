"""A plain sample for ``tests/test_own_process_isolation.py``.

Importing it plants a module that stands in for a second ``prismabuild``:
an ``own_process`` test that shares this process would see it.
"""
import os
from pathlib import Path
import sys
import types

sys.modules.setdefault("own_process_sample_shadow",
                       types.ModuleType("own_process_sample_shadow"))

OUT = Path(os.environ["OWN_PROCESS_SAMPLE_OUT"])


def test_shared():
    (OUT / "shared.pid").write_text(str(os.getpid()))
