# Optional positive-integer evidence policy

`shipcard_slots._positive_int` accepts positive `int` subclasses and rejects bool.
Invalid values return `None`.
`tests/test_positive_integer_policies_1777.py` checks these inputs.
