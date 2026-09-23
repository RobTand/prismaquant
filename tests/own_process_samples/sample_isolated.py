"""An ``own_process`` sample for ``tests/test_own_process_isolation.py``.

Not collected by the suite (no ``test_`` prefix): the isolation test names
it on a child pytest's command line. Each test writes its process id where
the outer test can read it.
"""
import os
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.own_process

OUT = Path(os.environ["OWN_PROCESS_SAMPLE_OUT"])
if os.environ.get("PQ_OWN_PROCESS_REPORT"):
    with open(OUT / "child-imports", "a") as record:
        record.write(f"{os.getpid()}\n")


def _record(name):
    (OUT / f"{name}.pid").write_text(str(os.getpid()))


def test_passes():
    _record("passes")
    # sample_shared imports this in the shared session's own process.
    assert "own_process_sample_shadow" not in sys.modules


def test_fails():
    assert 1 == 2, "sample failure text 7f3a"


def test_skips():
    pytest.skip("sample skip reason 91c2")


@pytest.mark.xfail(reason="sample xfail reason 4be0", strict=True)
def test_xfails():
    assert False


@pytest.mark.parametrize("value", [1, 2])
def test_param(value):
    _record(f"param-{value}")


def test_deselected():
    _record("deselected")
