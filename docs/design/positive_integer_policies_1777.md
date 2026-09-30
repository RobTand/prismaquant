# Different positive-integer policies — #1777 / #1301

These are not duplicate predicates to unify:

- `shipcard._positive_int(value: Any) -> int | None` is an optional evidence
  parser: positive `int` subclasses are accepted, bool is rejected, and invalid
  values return `None`.
- `tessera_reduced_schedule._require_positive_builtin_int(value: object,
  where: str) -> int` requires the exact builtin `int` type and positive value;
  invalid input raises `TesseraFormatError` with the caller's existing label.

The strict helper formerly shared the `_positive_int` name. #1767 introduced
that overloaded name without the inherited same-name ratchet reflecting it.
Renaming the strict helper and its five local calls restores the143-group
baseline without merging different policies or grandfathering inventory growth.
Predicates, return values, labels, errors and schedule controls do not change.
`tests/test_positive_integer_policies_1777.py` records bool, zero, negative,
float, string, None, int subclass and positive builtin inputs; the existing
reduced-schedule suite preserves public controls. Verification uses PrismaBuild.
No production numerical, GPU, wire or performance claim accompanies this fix.
