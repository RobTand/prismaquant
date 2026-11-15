"""Check the optional positive-integer evidence policy."""
import pytest

from prismaquant import shipcard_slots


class IntegerSubclass(int):
    pass


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "1", None, IntegerSubclass(1)])
def test_optional_policy_accepts_only_positive_integers(value):
    optional = shipcard_slots._positive_int(value)
    if type(value) is IntegerSubclass:
        assert optional is value
    else:
        assert optional is None


@pytest.mark.parametrize("value", [1, 37])
def test_optional_policy_preserves_positive_builtin(value):
    assert shipcard_slots._positive_int(value) is value
