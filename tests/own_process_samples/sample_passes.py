"""An ``own_process`` sample whose tests all pass (4f).

Not collected by the suite (no ``test_`` prefix): the isolation test names it on a child pytest's command line. Each test writes its
process id where the outer test can read it, so a pass is checked by where it ran as well as by its report.
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.own_process

OUT = Path(os.environ["OWN_PROCESS_SAMPLE_OUT"])


def test_first():
    (OUT / "pass-first.pid").write_text(str(os.getpid()))


def test_second():
    (OUT / "pass-second.pid").write_text(str(os.getpid()))
