# Standalone-gold digest contract slice

Research source: `6d8288adc94b3c4b624c143ed372fb6513f32784`.
Inspected integration base: `8bc0b5af8c69af352b559dccda99bd4e6fa8b418`.

## Disposition and integration

`tests/fixtures/digest_1301/standalone-gold/ledger.json` accounts for
**46 primary calls and 12 supplemental JSON calls in 13 files**. These are
individual calls, not ratchet scopes. The local check independently enumerates
live calls with the existing scanner traversal and checks exact locations and
Git blob identities. It does not modify scanner behavior or exclusions.

This is a **retention-only** slice: no production source bytes changed and
`completed_calls` is empty. Retained rows carry research-base provenance,
current source identities, recipes, classification, contract witnesses and
executable operation tests. The shared census and baseline are deliberately not
edited; their owner can reconcile this shard serially. This is not a parent
closure verdict, nor evidence that every eligible call elsewhere is resolved.

The retention grounds are deliberately different:

* The eight gold companion/runner modules plus `serve_fingerprint.py` appear in
  the current fingerprint common/tool-specific source rosters. Current producer
  identity records sizes and hashes of these exact files. This is an actual
  whole-file source binding, not a deduction from their recipes. Some paths
  already import the package, including fingerprint's artifact branch; we do
  **not** call those paths package-free. Their file remains source-bound.
* Fingerprint additionally has an existing isolated deployment with only its
  public file and runtime pin. Runtime snapshot similarly has an existing
  no-package public-file layout. Neither may gain an unavailable owner import.
* `dispatch_model.stage_checkout` copies the model worker to `worker.py` and
  invokes it separately. Its deployment does not stage the package owner.
* The sampler explicitly declares a stdlib-only separate-process/arming
  contract. The archived collector is not evidence of a current qualified run.
  Tests exercise append/error publication and unsafe/unarmed refusals.
* The scanner, including its BLAKE2b shingle and JSON publication calls, is
  explicitly byte-for-byte protected by Agreement S. Its operation test writes
  a baseline only in a temporary miniature repository, never the real baseline.

Current source binding is **not current qualification**. No pin, producer
roster, historical qualified fixture or protected file is repaired or changed.
A historical qualification record with different source blobs can demonstrate
binding policy, but cannot qualify HEAD. The gold source roster is not an
exhaustive package-dependency closure; the ledger records that boundary rather
than expanding the roster. Remote engines, HTTP observations, generated/staged
worker execution, validators, incremental operations, dynamic aliases and shell
boundaries are separately described. Enumeration is not proof that arbitrary
supplemental/runtime populations are exhaustive.

## Compatibility evidence

Tests execute actual consumer functions or AST-selected original implementation
functions where importing a serving stack would require unavailable vLLM.
Expectations are literal wire bytes or independent stdlib/struct references,
not values computed by replacement owners. For retained operations these inline
fixtures are the ledger's fixture references; the frozen call registry supplies
provenance only. Existing operation tests are referenced rather than duplicated.

The new tests cover strict versus lax/default-str JSON, Unicode, NaN/infinity,
mixed-key and circular refusals, raw-file versus joined-text versus tensor
identities, publication bytes/order, argv round trips, exact producer records,
worker output tampering, snapshot committed-byte closure/materialization, and
sampler append/error semantics. Isolated subjects run staged under `python -I
-S` without a PrismaQuant package. CPU adapters are not real serving engines.

A separate read-only review re-enumerated all 58 call locations and inspected
retentions/classifications and fixture independence. It found no blocking defect
within this slice. The current-roster test derives membership from production
rosters: it tests source binding and is not a frozen independent roster oracle.

## Observed checks

1. `python -m pytest tests/test_digest_1301/standalone-gold tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py tests/test_duplication_baseline.py tests/test_tessera_fleet_drivers.py tests/test_spaced_lax_tools_routing_1301.py tests/test_prismaquant_runtime_snapshot.py -q --basetemp="$TMPDIR/pytest"`

   **329 passed, 2 skipped**, 14 Torch deprecation warnings. The existing tiny
   registered-operator bootstrap cases require CUDA and are skipped on CPU;
   this is no GPU evidence.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"074e85c6b6322b5b9925d840f5fd2ff967179e877a372b8c959c6ab82ad69451","size":1154}},"operation":"33d4b5f63237f52a5620ca1ec8faf964367f818dfcfe0d0aae49e533cdd4d7ae","receipt":"0d355b385710db959151da9e338f497ac7fa75ca17f88bdd818560737a03b1ba"}]

2. Execute the union of every ledger `tests` node:

   ```sh
   python -c 'import json, os, pytest; from pathlib import Path; d=json.loads(Path("tests/fixtures/digest_1301/standalone-gold/ledger.json").read_text()); nodes=sorted({n for r in d["calls"]+d["supplemental_calls"] for n in r["tests"]}); assert nodes; raise SystemExit(pytest.main(["-q", "--basetemp="+os.path.join(os.environ["TMPDIR"], "consumer-pytest"), *nodes]))'
   ```

   **47 passed**, no skips, 14 Torch deprecation warnings.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"74ad6293077f25497f45c6258289e2d8824398461d410f50f3d8fcaf85f1c1ca","size":757}},"operation":"db2b39727589a9782d1de701dacd33f46e65eb2932cc002b5a5d3536a6a24b5a","receipt":"477e00efa63cdf5c94dd8a1341be72292c8bd74880b07570d090a61392f13f71"}]

No live fleet, vLLM, GPU, whole-suite or current qualified-runtime run was
performed. Parent acceptance/census closure is left to the integrator after the
other assigned slices; ratchet success alone does not demonstrate completion.
