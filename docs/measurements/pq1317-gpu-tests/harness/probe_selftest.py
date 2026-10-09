"""Synthetic tests for ``native_probe``. The CPU preflight runs them under xdist.

They need no GPU and no Tessera. ``container_entry.py`` checks that the probe
records the first assertion (it holds a float) and skips the second (an integer).
"""


def test_a_float_assertion_is_recorded():
    tolerance = 0.25
    measured = abs(0.5 - 0.375)
    assert measured < tolerance


def test_an_integer_assertion_is_not_recorded():
    assert 3 == 3
