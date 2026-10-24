# Retained issue #2949 evidence (inert fixtures)

These are historical source/packet snapshots, not freshly executed checks and
not a comparability acceptance. No application code or discovered tests live
here. Nothing here asserts current-main development policy or main adoption.

`identities.json` binds the exact base 5918b4c5, head 8967a796, companion
7daab5d3, and accepted-owner b5221b25 full commits and Git root/package trees.
It includes original/bridge durable-package inventories (package-relative
SHA-256 and bytes), their historical producer hashes, all 18 changed source
paths, the accepted catalog owner digest, and every other retained file.
Its own digest is deliberately not self-referential. `fixture_origins` records
where each raw snapshot came from; absolute historical paths are provenance,
not application defaults or prerequisites for consuming this fixture.

Git objects were read from the task-supplied bare repository. Package hashes
use the historical `_production_cache_source_sha256` algorithm: sorted durable
package files excluding __pycache__ and .pyc/.pyo; hash 4-byte big-endian UTF-8
name length, name, 8-byte big-endian payload length, and raw payload. The
head and companion packages match; 326 package files exist on both sides,
with only three changed. The accepted b522 catalog owner matches head bytes.
The source-closure packet inventories match the independently read Git blobs.
These are byte-identity statements, not numerical/full-price qualification.

`source-inventories.json` is a compact complete tracked-blob map for both
source trees, including modes and Git blob SHA-1s as well as SHA-256/bytes.
Reconstruct base with shared + original and head with shared + bridge; the
per-side maps contain changed/added files only. Omission represents absence.
The Git root-tree SHA-1s additionally bind directory topology and modes.
Only inventories are retained, not the full source trees or large payloads.

`source.diff.txt`, when listed in identities, is the exact raw stdout of
`git diff --binary --full-index --no-renames BASE HEAD` (full commits in the
manifest), with no external-diff, context, or rename overrides. The optional
raw diff is kept only if at most 200,000 bytes; its binding is always retained.
The eight `test_*.py.txt` files are the actual final-head regression sources
among those 18 paths, not synthetic cases or base revisions. Their `.txt`
suffix prevents pytest discovery. They are snapshots, not runnable fixtures.
`historical-verifier.py.txt` is the unchanged final-head historical tool for
adaptation; do not execute it. It includes obsolete fleet-specific literals
and pickle/control loading intentionally preserved as historical evidence.
Portable application behavior must use explicit configured inputs, not those
historical paths. The tool did not independently establish acceptance.

The four JSON packets are unmodified bytes copied from the companion-b399
measurement directory. Their embedded external paths/digests are retained
claims, not a recursive archive or proof that external artifacts remain
available. In particular, source closure is companion 7daa, while the package
attestation and exact-review handoff identify test-only successor head 8967.
The readiness packet reports 37 passed/16 failed before and 161 passed/0 failed
after in six targeted files, not whole-suite, serving, or CUDA coverage.
The handoff reports configured real-proof population 28 passed/0 skipped,
versus unconfigured 10 passed/18 skipped (no real retained integration).
These are DIFFERENT configured/unconfigured proof populations, not equivalent
green coverage, and none of these reported counts was rerun for this fixture.

Bounds: no tests, historical verifier, GPU work, source adoption, row restamp,
price admission, publication, launch, or signal/control actions were performed.
The packets retain accepted_as_comparable=false and incomplete/unpublished
readiness; local fixture identity validation does not change those states.
Large cost/control artifacts (including referenced cost.pkl and wires), CAS
receipts/results and qualification packets referenced by these four packets
are intentionally not copied. No current-main policy or default assertion is
inferred from historical sources or packets.
