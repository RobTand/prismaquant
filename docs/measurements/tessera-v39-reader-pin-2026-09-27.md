# Tessera v39 reader pin retake (PQ #1456 / #1464)

This is CPU reader/admission evidence, not a served-quality, GPU, or kernel
performance measurement. The pin is `4c4ff1c2eb68d4ffc3f8e0d3fcf9e019db5ab253`,
contract v39 / lane v10, digest
`f2f909486841c6e21ef6825fdc67ea5f57cf8ff8ffc241ccd89a521ea781c0bb`.
Tessera #646 changed the cached reader, not its runtime contract; the v39
scope changes came from #641 and were explicitly approved for this pin retake.

## Why fixtures moved

The first full PR CI (run `36292916935`, head `d25008c443a`) reported
55 failed, 13,444 passed, 725 skipped, one xfailed and 88 passed subtests.
Population: GitHub CPU runner, Python 3.12.14, torch 2.14.0+cpu,
transformers 5.16.1, four xdist workers / worksteal, one native thread each.
These failures named withdrawn E2M1 cell IDs/default-image scopes or asserted
v38 rung rosters. Representative failures:

- `the installed contract publishes no cell 'tessera_e2m1_k2_dense_sm121_decode'`
- allocator refusal: `reader-only/unsupported entries: ['TESSERA_E2M1_K2_R896']`
- `assert 39 == 38`

Fixtures now use the current cells' explicit image/scope. The v5 allocator
fixture selects the one image carrying A4 instead of the default image, which
now carries only dense A8. It still takes one image, never flattens a union.
Legacy KL grammar is tested with an explicitly synthetic transplanted receipt;
no test asserts that v39's route-only cells measured KL. Closed-world refusal,
source-pin, byte/shape, lane-predicate and activation-agreement gates are not
weakened. Admission scopes are pinned independently in
`tests/test_tessera_pin_v38_scope.py`.

## PB validation

Manual test execution used `pbtest.py --checkout <worktree> --python
/home/rob/venvs/pq-pb461728e4-tessera-4c4ff1c2-tf516/bin/python --tag gb10
--priority -10 --timeout-s 1800 --workers-per-shard 2 --threads-per-shard 1
--cpus-per-shard 2 --mem-gb 6 --pytest-args '["--durations=10"]' <files>`.
Final small retakes used timeout 900. PB hid CUDA; both Sparks ran Python
3.12.3 / torch 2.11.0+cu130 / transformers 5.16.1. PB checked the exact Git
Tessera pin before pytest. Each final action exited zero and reconciled its
assigned files. These are targeted file checks, not a full fleet suite.

The fixture retake plus direct callers covers 18 files: **451 passed,
3 skipped**. The remaining skips are:

1. `explicit immutable v5 publisher contract was not supplied`
2. Two real producer bridge cases: `producer projection tool unavailable:
   TESSERA_REPO is unset, so experiments/plan_from_layer_config.py cannot be
   located. This repository NAMES Tessera's tools instead of vendoring them;
   point TESSERA_REPO at the checkout of the pinned release.`

Those bridge cases need the new full source checkout at
`/mnt/shared/tessera-pins/4c4ff1c2eb68d4ffc3f8e0d3fcf9e019db5ab253`;
a non-editable wheel alone does not contain the producer experiments.
No skipped bridge is claimed as covered. The old-pin baseline's bridge cases
ran because its source checkout was present.

Only files that failed were checked against pristine `8fd6d931acc` with the
old `09d6559d` interpreter: 14 files, 342 passed / one v5-contract skip;
the separately checked legal-domain file passed 63/0/0. No whole baseline
suite was computed. The first new-pin legal-domain run failed two old roster
assertions at lines 305/328 (`a2bf47350d76…`), before their retake; wire-domain
counts remained 1,793 E4 rates plus 3,841 BF16 rates.

Final residual-failure retake receipts:

| File | PB key | Result |
|---|---|---|
| allocator golden | `7f1b9e81928679b72613a21b6a722a9957c7e7692e7ca7cc49b245f4c2d25d9a` | 2 passed |
| lane admission | `22f750c128c33843549a9545854503a4514c6bf61c10b619174952b8147ab20b` | 27 passed |
| menu | `0ca5bda314a890a899d5acd10a71fef42b46ac93c82ff88c89da2c684e3f0893` | 74 passed |
| stack population caller | `1538f76d38add0b89a024bdf6e2e00d13d7521d5882be7c5d639819c053b277c` | 7 passed |
| expert projection caller | `72512b72b36d3180460d3ad000fc3eeaeb4fad5052aed4f1f164b9e184fcf465` | 20 passed |

Full per-file keys, reconciled populations and verbatim skip records are in
`/home/rob/tmp/claude-campaign-20260926/pi/ts-644/pq-v39-fixtures.json`,
`pq-v39-fixtures-final.json`, `pq-fixture-callers.json` and
`pq-ci-baseline-files.json` in that directory. PB's terminal records, logs and
CAS receipts are the evidence, not the submission acknowledgements.

## Allocator golden: price did not move

A temporary PB-only capture ran the same golden fixture before and after the
pin/fixture retake, emitted normalised outputs, and was then removed:

- before: `e6167bc0d679d14b8661b78343d9a7a242a7227f2950d53540abb0ec4813406a`
- after: `9d2e1de243ebdf9db98cf76a88ee50a72b8e729babae0d56f107c76f24a58d26`
  (expected red against the old digest)

Exactly eight fields changed, all under `layer.json.__prismaquant__`:
`contract_version`, `reviewed_contract_sha256`, and each of the three units'
`route.detail` (cell IDs) and `route.requires_serve_flags` (resident-only).
Assignments, applicability, Pareto CSV and knee outputs were identical after
the existing temporary-path/wall-time normalisation. No allocator arithmetic
changed. The layer golden moved from `8850afe1…` to
`079a8ccada658ad5fc30c48b0cbcac0c0419933692b4bc31bc6332563c96df46`.
The full field-level delta and both normalised documents are retained as
`pq-golden-delta.json` and `pq-golden-{before,after}-normalized.json` in the
receipt directory above.

Full CI for the final PR head is recorded on PR #1464, independently of these
PB checks. The PR remains unmerged until Claude coordinates all workers'
new `--python` paths; old interpreters were not modified.

## Producer halves integrated

After #1440 and #1441 merged, this branch integrated main
`fc0f2ebe158c9bb7e4515b64cd5905a79cc1d988`; the only merge conflict was the
architecture provenance stamp, which retains both entries. The actual
proof-less #1441 manifest now reaches the new reader through the producer
helper in `tests/test_tessera_selected_cache.py`: dev accepts it with per-unit
warnings and matched served scales; certified mode refuses the same document.

The expanded bridge test was red on pre-producer-main `6d490b228a8` with its
old pin: PB `54b2e14f1d843e83767bdcbc6ac5ebca3004d4c89557372790b2f8996a86ffe9`,
18 passed / one failed, `producer package differs from accepted migration
proof`. The integrated branch then passed 19 selected-cache cases under PB
`27ee11a8caffad05c22b62ca0c82fd72c118e357f1a8e5567f3546390e87a8d3`, plus
15 served-activation policy cases under
`4b5763f26c1ab0554f4518ddf2700d37bedb87de24f3d20754f3ccea8f6017e6`.
Both are CPU-only Spark/Python3.12.3/torch2.11.0+cu130 (CUDA hidden),
transformers5.16.1, xdist2/native1, zero skips, rc0 and present CAS receipts
with matching log bytes. Timeout900, otherwise the same pbtest invocation
above. Receipts: `pq-producer-bridge-{before,after}.json` in the receipt
directory. The separate helper and CLI pre-fix regressions also failed before
the reader-mode wiring; the end-to-end bridge does not replace those checks.
