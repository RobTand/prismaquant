# Compiled re-allocation offline evidence

This is the U4 engineering slice of [PrismaQuant #2777](https://github.com/RobTand/prismaquant/issues/2777).
It follows approved PRI-11 spec revision `43adca93-5c6a-4e7b-bf5f-fe57569ecd43`.
It proves CPU metadata interoperability and refusal behavior. It does not
establish real compiled allocation parity or complete the release criterion.

## Finding and change

The pinned publisher contract has eager populations and no complete compiled
population. The existing allocator correctly refuses compiled allocation.
Compiled batch cells with eager-only decode also remain insufficient.

No production defect was found in this path. The missing engineering coverage
was a cross-mode allocation comparison and the batch-only compiled case.
The change adds five regression cases and exposes the existing scope fixture's
payload for reuse. It changes no allocator, admission rule, pin, export gate,
route-receipt verifier, cache, default, or serving behavior.

The retained runtime pin is Tessera
`aaa417003f0ee690e8ace138acd18df7f9531369`, with contract SHA-256
`ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e`.
The pinned refusal test authenticates those actual installed bytes through
`read_tessera_contract` and checks all declared cell/rung/residency contexts.
It also requires an eager positive witness, avoiding a vacuous empty-table pass.

## Fixture boundaries

The explicit interoperability inputs use the existing legacy v5 scope fixture.
They are synthetic populations, including hypothetical compiled decode cells.
They do not publish cells, implement future measured-scope grammar, or qualify
any compiled route. The tests patch the development menu's explicit table input.
They do not patch installed-runtime enforcement or either runtime pin.

The tests exercise:

- Digest-bound metadata reading and refusal after replacing publisher bytes.
- Complete-regime refusal at both `native_rung_verdict` and `allocator.main`,
  with no allocation file written for absent or batch-only compiled scope.
- Two real allocator CLI runs over identical probe and cost bytes, geometry,
  budget, candidate policy, profile, residency and explicit metadata.
- A nonuniform three-unit assignment. Dense and shared units select E4M3/K1
  R1024, and the routed expert selects E4M3/K1 R768 in both modes.
- Per-unit family, rung and real `Candidate.memory_bytes` equality. Selected
  byte prices must sum to the emitted body payload bits.
- Actual eager/compiled mode in the emitted target, each unit's serving scope,
  and each unit's lane provenance. Mode-specific provenance is not required
  to be byte-identical.

These fixtures load no model weights and need no GPU, vLLM, network download,
or model artifact. Their cost values are synthetic, not measured KL. Production
cache behavior and serving qualification are outside their evidence.

## Fleet verification

The x86 cohort ran on dl380g10 with ordinary CPU admission, without a
host-draining measurement. All 111 collected cases ran and passed, with no skips.

| Cohort | Result | PrismaBuild action key |
|---|---|---|
| New compiled allocation regression | 5 passed | `30b741e6b344e21dcd6fcaaee5ad8d964c7d8384d2d5172a03e9dd97dc0339d7` |
| Existing scope endpoints | 21 passed | `b934c4099b5fe24bb6c1737faca0fbb1185e5d830dfa84c42df342a5fdb23ad9` |
| Existing scoped route receipt | 85 passed | `9274c646274e7a085a535c5de8dbdd642e541c06e5360e885456ca40fd895813` |
| Compile both touched test files | Exit 0 | `7fac900d580ffd5c9c7eb9923c9b8d9d66da515aa4d41440f336b6db223ed23d` |

The test interpreter was
`/home/rob/venvs/pq-pin-aaa41700-pb20261016-git/bin/python`.
PrismaBuild's dependency preflight verified the exact Tessera commit above and
PrismaBuild SDK `027103d9a8417e06c7f13356e58779a313cd7088`.
The retained environment records Python 3.14.4, CPU PyTorch 2.11.0 and
Transformers 5.17.0.

The tested source commit is `c88c7b9957d09ae0e3ab67c8cfbbca198638eb91`,
based on `05b62b4742793a2188abd4ed041fc4d46fd6c2c0`.
The test snapshot is `e701135da7bef810225cbd995f6da129266c8c4a`.
The compile snapshot is `9a81b636bf37c2a6236709ec68b3f1c94d631e9f`.
Both CAS bundle digests and the relevant source bytes were checked against
the committed files. The subsequent documentation commit changes this report only.

Terminal stdout/stderr hashes and sizes were verified against the native attempt
records. Each successful action's `pb_verify_claim(hash_payload=True)` returned
`checks_passed: true`. The evidence bundle retains the native CAS receipts,
claims, reports and source verification. This check does not claim an independent
worker-attestation audit.

The retained commands are reproducible from the task worktree:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/fleet/pbtest.py \
  --checkout /home/rob/paperclip-worktrees/prismaquant/PRI-14 \
  --python /home/rob/venvs/pq-pin-aaa41700-pb20261016-git/bin/python \
  --tag x86 --workers-per-shard 1 --threads-per-shard 1 --mem-gb 6 \
  --timeout-s 240 --wait-s 600 \
  --json /mnt/shared/paperclip-runs/PRI-14/x86-v1.json \
  tests/test_tessera_compiled_allocation.py \
  tests/test_tessera_scope_endpoints.py tests/test_tessera_scoped_route_receipt.py
```

Long-lived clients use a transient user unit, here `pb-pri14-x86-v1`.
The compile action used the same interpreter with `-m compileall -q` and
the two touched test paths, through `pbrun.py --tag x86 --cpus 1
--demand mem_gb=1 --timeout-s 60`.

## Negative control and development outcomes

An isolated checkout removed only `native_rung_cells`' full-regime set check.
The new batch-only regression then failed with `AssertionError: assert not True`
because a single compiled batch cell was admitted. The no-compiled-cell control
still passed. Action
`ff2a856b54e9018bd146c4a76e1b04ea1f62c10dc9582becf636e588ab1a955d`
ran on dl380g10 and ended with the expected exit 1, one failed, one passed and
three deselected cases. Its failed action has no successful CAS receipt.
Its exact mutation and hash-verified terminal logs are retained.
This is a deliberately injected defect, not a defect claimed on unchanged main.

The first development cohort passed the pinned refusal and synthetic parity,
but two new tests expected `ValueError` where the CLI correctly used `SystemExit`.
The assertions were corrected. The next fleet cohort passed all 111 cases,
followed by the x86 run above. No production change was needed to make them pass.

Evidence lives under `/mnt/shared/paperclip-runs/PRI-14/`, with the final bundle
attached to PRI-14 for review. No performance, residency, quality or served
qualification result is claimed.

## Remaining release criteria

Keep #2777 open. Real positive completion requires immutable publisher bytes
with genuinely admitted compiled cells for every required regime, plus accepted
campaign inputs and actual allocation/route evidence. Eager and compiled runs
must use identical artifact geometry, sensitivities, budget, candidate policy,
topology, residency, caches and deterministic settings. Every selected family,
rung and priced byte count must match. Differences or refusal remain unresolved
unless Rob records a change to the acceptance criterion.

Batch-only qualification cannot satisfy that population. Existing compiled
dense/ambiguous census refusal remains in force. Missing decode scope, route
evidence or a compatible measured-scope contract is not repaired by these tests.
This slice neither changes the runtime pin nor authorizes upstream qualification,
a new exact-tuple GPU run, or closure of the related release issues.
