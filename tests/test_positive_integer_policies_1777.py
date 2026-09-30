"""Keep exact builtin-required and optional int-subclass policies different."""
import pytest

from prismaquant import shipcard
from prismaquant import tessera_reduced_schedule as schedule
from prismaquant.tessera_formats import TesseraFormatError


class IntegerSubclass(int):
    pass


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "1", None, IntegerSubclass(1)])
def test_required_builtin_policy_refuses_without_widening(value):
    with pytest.raises(TesseraFormatError, match="^fixture: expected a positive integer$"):
        schedule._require_positive_builtin_int(value, "fixture")
    optional = shipcard._positive_int(value)
    if type(value) is IntegerSubclass:
        assert optional is value
    else:
        assert optional is None


@pytest.mark.parametrize("value", [1, 37])
def test_both_policies_preserve_positive_builtin(value):
    assert schedule._require_positive_builtin_int(value, "fixture") is value
    assert shipcard._positive_int(value) is value
