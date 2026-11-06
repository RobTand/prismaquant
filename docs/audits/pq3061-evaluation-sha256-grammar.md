# Evaluation SHA-256 grammar: #3317

## Owner and sites

`prismaquant.digests.is_sha256hex` owns the bare digest grammar.
It accepts 64 lowercase hexadecimal characters in a `str`.
It accepts `str` subclasses.
Its `fullmatch` refuses a trailing newline.
The owner stays unchanged.

- `validate_panel_descriptor` delegates `eval_ids_sha256` to the owner.
  Local callers are `select_panel` and `evaluation_execution`.
  `tessera_joint_allocation` and `tessera_joint_aura` also call it.
  Selection and shape checks still precede the digest check.
  The binding seal still follows it.
- `_draw_sha256` delegates to the owner.
  `validate_eval_draw_descriptor` calls it for the old binding first.
  The binding seal precedes the token identity check.
  The closed input check precedes the artifact pin check.
  Shape and path checks follow the artifact pin check.
  `load_eval_draw` and `evaluation_execution` call the draw validator.

## Compatibility

The recording test checks all four calls with the real owner as delegate.
It uses tiny CPU data through the real validators.
Frozen controls accept lowercase strings and string subclasses.
They refuse uppercase, nonhex text, newlines, wrong lengths, bytes, None and integers.
They check exact errors, descriptor identity, precedence, paths and seal modes.
Existing tests retain token replay and draw loading coverage.

## Residual work

`calibration_data.py` and `shipcard_slots.py` retain bare grammar sites.
Composite image and commit grammars stay unchanged.
Repository-wide inventory integration and scanner gaps stay with parent #3061.
This note is not a complete inventory.
The duplication baseline stays unchanged.
This change migrates no hash constructor or JSON encoding owned by #1301.
This change makes no speed or serving-quality claim.
