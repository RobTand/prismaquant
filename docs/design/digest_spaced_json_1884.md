# Default-spaced JSON metadata identity profile (PQ #1884)

Refs PQ #1301. This bounded consolidation does not close the umbrella inventory.

## Encoding contract

`DIRECT_ASCII_SPACED_LAX` owns the inherited direct encoding from
`json.dumps(value, sort_keys=True)`: sorted keys, separators `(', ', ': ')`,
ASCII escaping, permissive `NaN`/`Infinity`/`-Infinity`, no default serializer,
no JSON reload/round-trip normalization and no trailing LF. Encoding to bytes
remains strict UTF8. Integer mapping keys are sorted as their original keys
before stdlib stringification; tuples remain arrays. JSON-escaped lone
surrogates stay escaped. Unsupported objects, invalid keys, incomparable key
types and cycles retain stdlib refusals.

The existing frozen `JsonProfile` owns these options. An optional separators
field follows the existing default-serializer field; its default remains
`(',', ':')`. The four prior compact constants and four-position constructor
use keep their encodings. The new configuration/name is additive. It does not
redefine compact profiles or merge normalized strict JSON with direct lax JSON.

## Current direct-profile inventory (PQ #1988, Refs #1301)

The shared owner now declares seven direct JSON profiles: `DIRECT_UTF8_STRICT`,
`DIRECT_UTF8_LAX`, `DIRECT_ASCII_STRICT`, `DIRECT_ASCII_LAX`,
`DIRECT_ASCII_LAX_DEFAULT_STR`, `DIRECT_ASCII_SPACED_LAX` and
`DIRECT_ASCII_INDENT2_LAX`. `DIRECT_UTF8_LAX` adds compact sorted Unicode JSON,
permissive nonfinite tokens, no fallback serializer and strict UTF-8 encoding.
It is direct encoding, not canonical JSON round-trip normalization.

`footprint.assignment_serialization_sha256` delegates only its final encoding
to this profile; registry aliases, caller-order coercion/key collisions and
native refusals stay with the caller, and `bytes_sha256hex` still owns the full
64-hex digest. The strict UTF-8 profile is not interchangeable: the inherited
recipe permits nonfinite values, including values returned by registry seams.
The independent literals and refusal/routing fixtures are in
`tests/test_assignment_json_profile_1988.py`; existing #1816 cases are unchanged.
The primitive ratchet removes only this caller's sorted-JSON entry (509 to 508,
sorted 257 to 256, raw 252 unchanged). No numerical, format, wire, pin, default,
serving, GPU or performance claim is made.

## Per-site inventory

### `CardProvenance.fingerprint`

This load-bearing sensitivity provenance identity retains its five-field
mapping: model_id, calib_hash, n_calib_samples, seq_len and render_basis.value.
Notes and probe_commit remain excluded. Default-spaced ASCII/lax text reaches
the existing text SHA owner and returns full64hex. There is no numeric/bool
coercion or fallback serializer. Bad basis attributes and native JSON failures
are unchanged. NPZ, AQUA, summary, compatibility and numerical paths are untouched.

### `build_preparation_template`

This load-bearing generic write-only produced-template identity retains root
normalization and its exact positive integer maximum guard before encoding.
The complete body, str(tier) spelling, zero working window, payload/checkpoint/
temp ceilings, slots and write_only flag are unchanged. Default-spaced ASCII/
lax text reaches the existing text SHA owner, retaining its sixteen-hex suffix
and prefix. Acquisition, publication, lease and SDK policies do not change.

Neither site migrates an identity or published representation. Both retain
their already-shared `text_sha256hex` calls; this slice moves only serializer
options to the named owner. Do not substitute a compact or round-trip profile
merely because parsed values compare equal.

## Evidence

OLD source characterization runs the new sixteen cases plus existing site
goldens: literal card bytes, ASCII controls/surrogate/nonfinite/-0/bool spelling,
template normalization/validation/width and legacy constructor compatibility.
Only the two missing new-profile routing aliases are intended to fail before
production changes; terminal counts and action keys are recorded in the PR.

GREEN adds named-profile literal byte goldens, buffered/streamed SHA equivalence,
compact/spaced distinction and native refusal checks. The old compact owner and
both consumer suites remain required. The ratchet removes only the two
consumers' sorted-JSON entries; all raw scopes and unrelated entries stay.
Actual counts, terminal/CAS receipts, compile and static review belong in the PR.

All tests and compile checks run through admitted PrismaBuild. This is not
local, GPU, performance or numerical qualification. Existing source type
findings and controller-only pytest environment mismatch are not repaired by
casts, stubs, suppressions or global configuration changes. No runtime pin,
default, stage graph, wire bytes or pinned input changes are part of this slice.
