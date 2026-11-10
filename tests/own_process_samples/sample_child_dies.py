"""An ``own_process`` sample whose child process dies mid-module (4f).

Not collected by the suite (no ``test_`` prefix): the isolation test names it on a child pytest's command line. The second
test kills its own process, so the child reports a first test and then nothing; the parent must fail every test the child did
not report, by name, and never pass or skip it.
"""
import os
import signal

import pytest

pytestmark = pytest.mark.own_process


def test_reports_before_dying():
    pass


def test_kills_its_process():
    os.kill(os.getpid(), signal.SIGKILL)


def test_never_runs():
    pass
