# Quantum record TOCTOU at `verify_quantum_identity`

Scope: PQ #2067 (P1). Base `8811ad5f1e9586ce0de9f5db7f64d1c8589f70d2`.
Branch `sol/pq-quantum-control-read-20261002`. This is a CPU gate fix and its
causal regression; no GPU, model, format, serving, performance or release
qualification follows from it.

## Defect

`prismaquant/joint_cost_quantum.py:214-218` (pre-fix) hashed `quantum_path`
through `_digest_of`/`Path.read_bytes`, then parsed a separate
`Path.read_text`. A file replaced between the two reads could return a
different, internally valid record whose raw wire bytes were never hashed; the
second record's canonical self-identity binds only its own bytes.

## Fix

`verify_quantum_identity` now reads the record through the shared
`stage_inputs.read_bound({"path", "sha256"})` owner and parses those exact
authenticated bytes, so the digest check and the parse consume one owned read.
`QuantumIdentityRefused` and the digest-mismatch diagnostic are preserved, and
the schema, identity, campaign, adjoint, chunk and window checks are unchanged.
Plan and prepared inputs keep their existing `_digest_of` checks; the adjoint
loader already authenticates its own read.

## Causal regression

`tests/test_joint_cost_quantum_runtime.py::test_quantum_record_is_authenticated_from_one_owned_read`
runs in certified mode (`PRISMAQUANT_DEV_MODE=0`) on the real `identity_files`
fixture. It writes the original record, then patches `Path.read_bytes` for the
record path to return the original bytes and atomically replace the live file
with a second, internally valid same-campaign record (an advisory
`window.names` field this verifier ignores). The gate must return the original
record and consume the record bytes once; the now-changed file must refuse the
old digest.

- RED (source fix stashed, test present): `5e379184aefc` first exposed a test
  authoring bug; after fixing it, `832a8ab4`-style run returned the substituted
  record, identity `832a8ab4...` instead of the original `926d2a72...`.
- GREEN: `6638efdf3626` -- 38 passed.
- Regression set `7e8547e429a9` (launch contract), `db0140909449` (boundary
  readset), `ee82e1deab18` (executable readset), `c727aac91c28` (catalog
  extension), `9a85f2835f10` (load-plan readset), `1a1386d3cb21`
  (redeclared-plan coverage): 139 passed. Combined 177 passed, 0 skipped,
  0 failed.
- Compile: `17e48c209325` `py_compile` on the two touched modules and the test
  module, rc 0.

Environment: qualified CPU interpreter
`/home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python`
(prismabuild `95a59051`, tessera-quant `b40c93cb`), PB runtime
`5b6b97c0f717`, one CPU / 4 GiB per file, native threads 1, priority -10,
180-240 s bounds. Receipts in
`/home/rob/tmp/astra-resume-20261002/pq_spill/toctou-*.log` and `toctou-*.json`.

Not measured: any GPU path, served artifact, residency or performance claim.
Downstream gates may still refuse some substituted records; this finding is the
raw-byte/parsed-record relationship at this one gate.
