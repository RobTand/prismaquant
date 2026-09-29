# Incremental source-record framing — #1752

`prismaquant.digests.LengthFramedSourceSha256` owns this exact SHA-256 byte
profile, starting from an empty SHA-256 state. Each `update(name, payload)`
feeds four fields in order:

1. Four-byte big-endian length of `name.encode("utf-8")`.
2. Those strict UTF-8 name bytes.
3. Eight-byte big-endian length of the payload.
4. The payload bytes, with no separator, normalization or final trailer.

`hexdigest()` observes the current state without finishing it. The object
retains the digest state, not a collection of source files or their payloads.
It does not discover files, sort names, deduplicate entries or reconstruct
source. Those semantics remain with each caller.

## First three callers

The `source_proof` routines in `joint_aura_source_transition`,
`joint_aura_run_transition` and `joint_aura_retained_budget_transition` keep
separate current and reconstructed states. Their traversal, bytecode
exclusions, transition-file exclusions, exact reversed rewrites, seen-file
requirements and contract comparisons are unchanged. The run transition's
dev-mode branch still records the executing tree without reconstruction or
contract comparison. Standalone loading uses the existing stdlib-only digest
module dependency; no new runtime dependency is introduced.

These are load-bearing source identities; dev-mode observations use the same
profile. This is not an identity migration, a new seal, or a relaxation of any
existing refusal. It makes no performance, serving, calibration, KL or bpp
claim and changes no format menu, stage graph or ship gate.

## Package-source identity consumer — #1764

`production_weight_cache._production_cache_source_sha256` uses the same
profile for its durable package-input tree. The caller still discovers and
sorts paths, excludes interpreter bytecode, includes packaged JSON/lattice
and other non-code data, selects the default installed root, follows the
same file symlinks, and wraps read errors with the same cause. Only framing
moves to the existing owner. Rendered weights, tensors, residency, cache
allocation and prefetch are not changed.

`tests/test_pwc_source_framing_1764.py` independently spells the legacy record
with `struct.pack`, verifies exact owner inputs, and exercises default/explicit
roots, Unicode/binary data, followed-file symlinks, bytecode-only/missing roots,
read-error causes and strict UTF-8 name refusal.

## Verification

`tests/test_source_framing_1752.py` contains a frozen GoldenTable generated from
an independent literal legacy recipe through PrismaBuild. It covers empty
streams/fields, ambiguous unframed pairs, caller order, duplicate names,
non-ASCII names and binary/CRLF payloads. Caller spies prove exact current and
reconstructed inputs route to the owner. Separate tests preserve missing and
duplicate hunk, incomplete proof, wrong contract, UTF-8 and dev-mode behavior.
The existing transition tests cover their standalone loaders and contracts.

The parent #1386 remains open for the remaining actual tree callers. Its
newline family was already delivered in #1447. The projection kernel's
source/torch/CUDA/flag recipe is different and is not moved to this profile.
