"""A module-level skip on an own-process proxy retains its source location."""
import pytest

pytestmark = [pytest.mark.own_process,
              pytest.mark.skipif(True, reason="sample module skip 2113")]


def test_marked_skip():
    raise AssertionError("the skipped sample body must not run")
