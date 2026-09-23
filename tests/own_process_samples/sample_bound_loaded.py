"""An ``own_process`` sample that reports the per-test bound it runs under.

Not collected by the suite (no ``test_`` prefix): the isolation test names it
on a child pytest's command line (PQ #1055). Its one test writes, as JSON,
whether PrismaBuild's per-test bound plugin is registered, the bound it
armed, and whether the ``prismabuild`` package itself was imported. The
Stage A harness refuses a process whose ``prismabuild`` is not its pinned
candidate, so loading the bound must never import the package.
"""
import json
import os
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.own_process

OUT = Path(os.environ["OWN_PROCESS_SAMPLE_OUT"])


def test_reports_its_bound(request):
    config = request.config
    (OUT / "bound.json").write_text(json.dumps({
        "pid": os.getpid(),
        "registered": config.pluginmanager.has_plugin(
            "prismabuild.pytest_test_bound"),
        "bound_s": getattr(config, "_prismabuild_test_bound", None),
        "prismabuild_imported": "prismabuild" in sys.modules,
    }))
