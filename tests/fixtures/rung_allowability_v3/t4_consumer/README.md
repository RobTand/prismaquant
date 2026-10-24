# Offline T4 consumer test inputs — SYNTHETIC mechanics only

- `producer.py` is the exact, unmodified Tessera semantic owner at commit
  `8f881a7248f86321beee1c40f15ea3cc617591ff`; check its SHA-256 against
  `provenance.json` before loading. It is frozen test input, not application policy.
- `synthetic_t4_scope.json` is a v3 input analogous to the producer test helper
  `approved_t4_table`, NOT a copy of the private approved measurement table.
  The producer menu supplies build ID, activation, all cell IDs, M values, shape
  dimensions/modes, routing, epilogues, recipe, execution paths and rung menu.
- Native geometry comes from AST-extracted `fixture` and `fixture_v2` constructors
  at that same commit, called as `fixture_v2('tcq', 'native_tcq')` during input
  creation only. No test module or test function was run. No plane-size or decoder
  formula was invented. As in `approved_t4_table`, the base o_proj geometry is
  reused for every cell, NOT asserted to be real per-shape compiler/plane evidence.
  This includes the base fixture's tile-bit rational; it is not a byte accountant.
- Each 10 us kernel time and both 10 us pass times are copied from the base fixture:
  fixtures, NOT observed measurements. `measured`, `observed_rungs`, build IDs and
  other schema vocabulary exercise branches only. The fixed timestamp and resource
  fields are fixture inputs, not a new observation. There are no retained samples,
  authentic private measurements, measured KL, or scientific quality claims.
  Every row's quality is exactly `{"anomaly_flags": []}`; sampled-weight quality
  fields were removed. Geometry classes use the canonical public constructor.
- Offline consumers can load the local snapshot with standard-library importlib
  (or compile/exec in a private namespace) and read the JSON. No installed Tessera,
  private PB API, network, private table path, source checkout or runtime git call
  is needed. The upstream absolute table path inside the verbatim producer is
  provenance text only; do not dereference it. Acceptance tests are separate work.
- `serving_qualified` is deliberately not asserted true; synthetic data cannot
  qualify a producer, native path, numerical result, or serving deployment. Any
  mechanical `allow` result is not evidence of real admission or qualification.
  The legacy missing-`allowable_rungs` gate is a separate acceptance concern.

**test-only metadata snapshot; no runtime/kernel pin changes**

No application/source/main-test files were edited and no tests were run to create
these inputs. Keep the snapshot immutable; a future refresh needs new provenance.
