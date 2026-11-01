# PrismaSnap durable publication consolidation

Partial delivery for PRI-226 and #1295. Inspected base:
`05b62b4742793a2188abd4ed041fc4d46fd6c2c0`.
Tested source: `d2d9065137e70f2cda4180cccf2530789b701fb1`.
The later audit-document commit changes documentation only.

## Contract and owner

At the base, `prismaquant/prismasnap_checkpoint.py:326` `_atomic_json`
encodes `DIRECT_UTF8_INDENT2_STRICT`, then repeats the publication sequence
owned by `_atomic_bytes` at line 1029 in the same module. Both use a fixed
`.<name>.tmp` sibling, reject any existing temporary with `lexists`, open
exclusively, write and flush, fsync the file, replace the destination, then
fsync its directory. Their failure artifacts and stale-file diagnostics agree.

`_atomic_json` now encodes and calls the existing `_atomic_bytes`. It retains
its serialization-level call and monkeypatch boundary. Encoding still fails
before filesystem inspection. No new publisher, loader, cache, pool, or
residency mechanism is introduced. Generalizing the IO engine would widen
the change without removing additional proven duplication.

The complete direct production callers at the recorded base are:

| Callee | Enclosing function | Lines in prismasnap_checkpoint.py |
| --- | --- | --- |
| `_atomic_json` | `bind_legacy_text_probe` | 1198, 1203 |
| `_atomic_json` | `scan_tensor_metadata_manifest` | 1326 |
| `_atomic_json` | `plan_dense_checkpoint` | 2133 |
| `_atomic_json` | `merge_plans` | 3404, 3410, 3438 |
| `_atomic_json` | `realize_bf16_plan` | 3763, 3766, 3960 |
| `_atomic_json` | `_materialize_selected_shards` | 4235, 4251, 4460 |
| `_atomic_json` | `_copy_checkpoint_metadata` | 4606 |
| `_atomic_json` | `materialize_checkpoint` | 4918 |
| `_atomic_json` | `materialize_checkpoint_part` | 5022 |
| `_atomic_json` | `merge_checkpoint_parts` | 5493, 5498, 5574, 5616, 5650 |
| `_atomic_bytes` | `bind_legacy_text_probe` | 1202 |

Existing tests save or replace `_atomic_json` in
`tests/test_prismasnap_checkpoint.py:1267,1276,1300,1547,1559,1585`.
These callers remain unchanged. The census used source reading, symbol
searches, and an AST walk of enclosing functions. It makes no reflective or
external-caller completeness claim.

## Compatibility evidence

`tests/test_prismasnap_atomic_publication.py` retains literal expected bytes
and filesystem observations established on the old source. Nine old-source
cases pass. They cover Unicode, indentation, sorted keys, escaped newlines,
negative zero, no final newline, existing-destination replacement, file and
directory sync order, failures at each publication step, stale regular files,
directories and dangling symlinks, and encoding failure before stale-file
refusal. A tenth case injects failure at the byte-publication owner; it fails
on the old JSON writer and passes after delegation.

| PrismaBuild action | Outcome |
| --- | --- |
| `52904abf0081e6a22c5afab35de79f8e5fde2eeffa76ed02f6d3ea9a2841ea97` | Old publication path: 9 passed, 1 expected shared-owner regression failed |
| `2fddb59773a06ea1985a3215654d653fee2f4a97d248efb56108884cad76978a` | Old distribution checker: both missing-dispatch-asset regressions failed |
| `4fbb75fae3d8de9a9a83a75ccbc4fe58924c0a492ccfd5caf757a1224b58ada9` | 17 passed on sparky |
| `2fcf395381aa03b78e19098be7e436ad28d96514aef97aba0b05ea2f266eadfb` | 46 passed on sparky |
| `8fb1cb5d4e287efdb7fa4f38be46ce2ecc83f195c278d5f5766518d50accdcdf` | 12 passed on sparklina |
| `ab7fb64002cc739177f29e5258839d2083fa6b2373cf9f19780ebcfe9c85a8a9` | 11 passed on sparky |
| `c35e99e4c1fc1689119f2d9e41e6a8bda97a4039032bf93a2a7c84d0a380396a` | Six changed Python files compile on dl380g10 |

The green cohort has 86 passes, no failures or skips, complete collection,
outcomes and teardown. It includes the new tests plus
`test_prismasnap_checkpoint`, `test_prismasnap_contract`,
`test_prismasnap_validation`, `test_prismasnap_routing_1713`,
`test_duplication_baseline`, `test_io_site_freeze`, `test_docs_staleness`, and
`test_architecture_doc`. All jobs are CPU-only. Test interpreters verify the
installed PB and Tessera development pins before pytest.

The green checkout snapshot is
`6cf019200cfcb3f148f382c5a4e6da2ea6d3e501`, bundle SHA-256
`55e9a80f88137ac1d60b858de37fd201adb33360ebf1b7cfc29c7409122500e7`.
Every tracked blob at the tested source matches that snapshot. Its only
additional file is the PB checkout-closure record. The four test result
claims and compile claim pass all nine own-byte CAS checks, including payload
hashes. Worker-attestation verification is outside those checks.

Commands and raw evidence are retained under
`/mnt/shared/paperclip-runs/PRI-226/`: `green-launch.json`, `green.json`,
`verified-green.json`, `source-binding.json`, the per-shard action/claim/log
files, `compile-action.json`, and `compile-claim.json`.
`verify_evidence.py` reruns the test-report, terminal, log and claim checks.
Transient units are `pb-pri226-atomic-red`, `pb-pri226-dist-red2`, and
`pb-pri226-green`. One earlier distribution submission was refused during
snapshot because the checkout changed. It executed no tests. Two compile
submission attempts were refused before the accepted action above.

## Limits and incidental fixes

The equality claim is publication bytes for the same supplied payload.
`_producer_identity` still binds the actual Git commit and implementation
files. Changing this source changes producer identity; historical plans are
not migrated or made compatible with a different runtime. Existing producer
drift refusals remain intact.

This is metadata publication. Tensor rendering, numerical formulas, source
acquisition, resident prefetch, calibration, format bytes, runtime pins and
serving gates are unchanged. No GPU, served-quality, KL, bpp, performance or
residency qualification is inferred. No performance claim is made.

Two incidental findings are fixed in separate commits:

- The release archive checker omitted the serving-dispatch table asset family
  already declared in `pyproject.toml`. Independent incomplete wheel and
  sdist fixtures now prove refusal. This does not change packaged assets.
- Stale prose described MXFP8 as a menu choice, GGUF as supported, and
  `serve_fingerprint` as the cross-arm comparison key. The corrections follow
  existing repository policy and `tools/kl_ab.py`. Gold-tool executable ASTs
  are unchanged after removing docstrings.

The existing duplication baseline and scanner are unchanged. Same-file
publication copies are outside its cross-file pair detector. PRI-226 still
owes complete semantic coverage, an appropriate extension of the existing
inventory where warranted, residual migrations, and independent current-main
reconciliation. #1301 retains digest ownership, #1303 retains domain numerics,
and #1295/#2768 retain streaming and integration acceptance. This slice closes
none of those programs. All 621 tracked archive or GGUF-named paths at the
base retain their blob identities.
