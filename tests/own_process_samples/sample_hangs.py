"""An ``own_process`` sample with one test that hangs (PQ #1027).

Not collected by the suite (no ``test_`` prefix): the isolation test names it
on a child pytest's command line, under a per-test bound of a few seconds.
The hang alone outlasts one bound, so a test after it can pass only if each
test is bounded on its own rather than the module as a whole. Each test
writes its process id where the outer test can read it.
"""
import os
from pathlib import Path
import time

import pytest

pytestmark = pytest.mark.own_process

OUT = Path(os.environ["OWN_PROCESS_SAMPLE_OUT"])


def _record(name):
    (OUT / f"{name}.pid").write_text(str(os.getpid()))


def test_before():
    _record("before")


def test_hangs():
    _record("hangs")
    time.sleep(3600)


def test_after():
    _record("after")
