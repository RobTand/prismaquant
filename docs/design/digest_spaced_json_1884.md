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

The shared owner now declares eight direct JSON profiles: `DIRECT_UTF8_STRICT`,
`DIRECT_UTF8_LAX`, `DIRECT_ASCII_STRICT`, `DIRECT_ASCII_LAX`,
`DIRECT_ASCII_LAX_DEFAULT_STR`, `DIRECT_ASCII_SPACED_LAX`,
`DIRECT_ASCII_INDENT2_LAX` and `DIRECT_UTF8_INDENT2_STRICT`. `DIRECT_UTF8_LAX` adds compact sorted Unicode JSON,
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


## Preparation publication encodings (Refs #1301, 2026-10-02)

The journal writer additionally names `DIRECT_UTF8_INDENT2_STRICT`: sorted
keys, two-space indentation, literal Unicode, strict nonfinite refusal, no
fallback serializer and no final LF. It preserves the former direct writer
rather than normalizing the object again. Its immutable manifest wire is
load-bearing on journal resume and in upstream manifest SHA bindings.

The selected-cache writer has three distinct output encodings: the manifest
uses `indent2_json_file_bytes` (ASCII escapes, strict, final LF), read-path
publication uses `DIRECT_ASCII_INDENT2_LAX` plus LF, and stdout uses
`DIRECT_ASCII_SPACED_LAX` plus print's LF. The manifest SHA and acquired input
SHA both use `bytes_sha256hex` on the same already-owned bytes. Input bindings,
manifest publication and hashes are load-bearing. The stdout report mirrors
those bindings; it is preserved as an externally consumed interface. The
read-path document's exact bytes can become a submitted/stored input identity.

Migration dedup keys use `DIRECT_ASCII_SPACED_LAX.text` for both old/new pins.
The first source-name-ordered record still wins; nonfinite tokens remain
accepted, mixed key sorting and unsupported values raise their old errors,
and no input is reconstructed or canonicalized differently. The dedup outcome
is load-bearing in the merged checkpoint provenance.

`tests/test_digest_prepare_io_1301.py` freezes 25 actual old-production outcomes
from PB RED `2f65986b6b36be629cb1512a1eb5f0cd15c961257fb408bb08303fbbe9fd40fa`
(base `8811ad5f1e9586ce0de9f5db7f64d1c8589f70d2`; source snapshot
`4147891c92511ae6e1902efe7b4a8cd4f0c80132`). Nineteen byte/refusal tests passed;
four intended owner-routing seams failed. Temporary path text is normalized
by the existing GoldenTable. The selected-cache control pickle includes that
temporary path, so its derived handoff digest is verified independently then
masked as `<handoff-sha256>` in stdout goldens; manifest bytes and SHA stay
unmasked. The old tape's same field receives the same mask. No golden is
recorded from replaced production code. The ratchet removes exactly five
primitive scopes, 508 to 503; no new scope is admitted.

## Tools spaced-JSON bundle (Refs #1301, 2026-10-05)

The original selection contained 58 serialization calls in 45 scopes across
36 files. At the CEO-authorized proof cutoff, 40 proved calls remain on the
existing profile in 32 scopes across 24 files; 18 individually unproved calls
are restored to their exact `99a578e0` expressions as explicit #1301 residuals.
Proved neighboring calls in mixed scopes remain routed. Callers still own
argument construction, text/bytes, hashing, line feeds and publication.

The original routing assertions and generic recipe/profile comparisons did
not prove each consumer. They have been removed. Their pre-change failures
were style/source failures, not behavioral regressions. The historical
`b2cb7017` run failed nine declared-dependency cases for missing `xxhash` and
remains failed; agreement with an equally broken baseline did not make it green.

`tests/test_spaced_lax_tools_routing_1301.py` now exercises actual state
append/refusal, calculated benchmark summaries, spec construction, checkpoint
parsing, container submissions, analysis publication and worker file bytes.
Its frozen outcomes were recorded from original commit `99a578e0`, not from
the replacement profile. Actual lightweight launch tests run outside the
repository without `PYTHONPATH` and without site packages. The image driver,
chain spec, wire reader, metadata audit and render analysis retain their
standard-library dependency boundary. Band handoff, checkpoint parsing and
Stage B head help/argument handling still precede scientific imports; the
other tools retain their original package-dependent qualified contexts.

The immutable version-three cutover packet reconciles all 58 original calls
as 40 retained proved routes plus 18 explicitly restored residuals. It records
retained-input and original CPU-fixture provenance, native output/framing and
omitted computation; a source location alone is not proof. Earlier packets,
all red actions and late native diagnostics remain preserved. The native
deadline wrapper did not qualify either GPU route for retention. This is
boundary/fixture evidence, not a producer, campaign, wire or serving rerun.
No model roll, capture, CUDA benchmark or image rebuild was performed.
Mixed compact, indented and strict recipes remain outside this migration.
The separately staged `tools/tessera_fleet/model_worker.py` keeps its original
standard-library serialization; its actual publication bytes are tested.
The existing duplication ratchet retains syntactic policy. The three package
`json.dump` file writers remain a different bundle; #1301 stays open.
