# Shared joint probe identity verification

This is the first J2 unit, following J1's ownership and dispatcher fixes.
`joint_aura.make_joint_probe_identity` owns the common v2 probe fields.
The whole streamed AURA path, Stage B quantum and MTP adapter now call it.
Their arithmetic extensions and source-transition identities remain explicit.
MTP still uses its own objective and `T - 1` token geometry. This unit does
not claim completion of J2's shared journal and numerical layer step.

The builder hashes the existing logical token bytes and records their shape
and dtype. The version, normalization, distribution, probe coordinates,
temperature, source execution and arithmetic keep the same values for the
same explicit inputs. Existing validated identity snapshots and the unit
envelope codec are unchanged. No cache, format, default, serving gate,
kernel, or accumulation order changes. No performance improvement is claimed.

## PrismaBuild evidence

The baseline passed all 121 tests in two CPU shards. The final run passed all
167 tests, adding fixed calibration-byte and noncontiguous-input tests,
source-transition checks and the architecture check. Neither run skipped a
test. The existing single-run versus quantum test still compares signed
components, x2 samples, and actual checkpoint measurement envelopes bitwise.
The MTP tests compare the real lease's rows against independent autograd
projections and verify its distinct objective and one shared probe identity.

Both runs used the qualified interpreter
`/home/rob/venvs/pq-pin-fca4c6ce0-pb027103d9/bin/python`, with verified PB
`027103d9a8417e06c7f13356e58779a313cd7088` and Tessera
`fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb` packages. Each shard reserved
one CPU and 6 GiB, one pytest worker and one native thread, priority `-10`,
portable placement, an action ceiling of 1,200 seconds and a per-test bound
of 180 seconds. All endings were `executed`, exit 0, with CAS receipts and
actual pytest completion records. The admitted compile check also exited 0
and published a CAS receipt. It used `--anywhere`, one CPU and 2 GiB.

| Run | Actions |
|---|---|
| Baseline | `ae1eaa05b9da2f5a126b80f9a6c891ae02f95a737feafacf5b692f63a2f7b66f`, `d3e7c85817cd227c8d97b942d40fb6481f74be6af0f9cad22c0b5b011a2f64ed` |
| Refactor | `a35df2c18376ce3bd4f6b7cb31c53907cc5a66d058bd459af5b9fa00b93526b6`, `5f9fe6bbf4fec30039d7d75bd47ee747c9b75c9e19bea17de05d1b7a05e3e529` |
| Compile | `e2a4ac49d5b3a7a3136cace8eced9609ad61229189f5a63fadb5b80706bd154e` |

Commands, logs, per-test completion and terminal/CAS inspection are retained
under `/home/rob/tmp/pq-refactor-20261014/reports/` as
`joint-j2-identity-baseline.log`, `joint-j2-identity-baseline.json`,
`joint-j2-identity-after.log`, `joint-j2-identity-after.json`,
`joint-j2-identity-compile.log` and `joint-j2-identity-receipts.json`.
An initial discovery command named the absent `tests/test_joint_aura.py`.
PB refused before publication. The baseline file selection was corrected
before execution. No missing dependency or tool was hidden by a skip.

No tests were removed in this small unit. The new tests exercise the actual
constructor with fixed bytes and logical token order. They do not inspect
source text or monkeypatch successful admission. GPU qualification was not
needed for this record-construction refactor, and no GPU result is claimed.
