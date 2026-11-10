"""An ``own_process`` sample whose collection fails (4f).

Not collected by the suite (no ``test_`` prefix): the isolation test names it on a child pytest's command line.
A collection error in an own-process module is a failure of every test the module declares, never a skip.
"""
import pytest

pytestmark = pytest.mark.own_process

import own_process_sample_module_that_does_not_exist  # noqa: F401,E402


def test_never_collected():
    pass
