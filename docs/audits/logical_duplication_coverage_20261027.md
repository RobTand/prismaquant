# Tracked duplication coverage

PRI-226 continues the repository-wide program under open #1295. This slice
extends `tools/duplication_inventory.py`; it does not consolidate another
runtime family or close the program.

## Report contract

```bash
python3 tools/duplication_inventory.py --coverage-report /absolute/path/coverage.json
```

The report accounts for every Git-tracked path. Existing excluded components
remain visible with their names. Non-Python files receive a `not_python` row;
this does not mean their behavior has been read. The remaining Python files
use the existing scanner's structural, same-name and primitive-site analysis,
with same-file pairs enabled. Audit function IDs include line numbers so
repeated qualified names retain distinct locations.

Git metadata, readable Python files and valid syntax are required. A tracked
Python symlink refuses rather than reading its target. A failure produces no
new report. An old output file can remain after a refused invocation; check
the process exit and the report's source identity before using it.

`source_commit` names the analyzed checkout's HEAD, and `tracked_dirty` records
tracked changes. A dirty report is not a content seal. Use an immutable source
snapshot for retained evidence. The report does not bind mutable files to a
commit merely by printing HEAD.

The default `scan()` and `digest_sites()` APIs retain their file population,
exclusions, identifiers and output. The default gate remains cross-file Python
under `prismaquant/` and `tools/`. `--write-baseline` and `--coverage-report`
are mutually exclusive. This slice does not edit the baseline or add an
exclusion. Widening the default gate was rejected because it would require
baseline growth before semantic adjudication.

Matches are reading leads. MinHash proposes candidates and exact Jaccard checks
them. The report can miss behaviorally equivalent functions with different
structures, small functions, non-Python logic and dynamically bound calls.
Its digest classification retains the documented static alias limits. It is
not a semantic census or proof that a matched pair should share an owner.
Archive and deprecated GGUF preservation still applies, including to GGUF
paths that appear in read-only candidate evidence.

## Source and fleet evidence

The analysis refactor is `8b49699aec`; report behavior is `7f99406b04`.
The later `6ded233c6c` changes two test expectations to canonical lexical pair
order. Both migrations use the same existing analysis; no parallel scanner
was introduced. The branch was rebased once before review, with no base change.

Evidence lives under `/mnt/shared/paperclip-runs/PRI-226/`. The rerunnable
`scanner_jobs.py` submits validation through the published fleet client;
`verify_scanner.py` checks native endings, terminal log hashes, source snapshot
blobs and local-result claims with payload hashes. The claim checker does not
verify the worker attestation against the full manifest.

| Check | Action | Result |
| --- | --- | --- |
| Missing API/CLI regressions on old source | `0a91a7299f26a316625f5b2fa54bcd57566e5b15f3bfea18af4d7c7b2f8d9edf` | Four expected failures |
| Initial coverage, existing baseline and lexical-scope tests | `4c7caf46690c18c8b9693a1a8bc2dc6ba2005906a7817c8a0cc6b09a2d01b61a` | 36 passed |
| Line-qualified locator fixtures | `448b5573991641814f7cbece347ece8a7b1c9f2615eca4857834c86181546b07` | Two fixture-order failures, 36 passed |
| Final regressions and compile checks | `0ea55055ba289a7614f9dd3454397294d1c9248d1b5ecec8ead749d1ccbecda3` | 38 passed, zero failures/skips; both files compile |
| Actual tracked-tree report and compile checks | `50bf2bbf0cf4fdc1e93763ba7b666663c59ba98223c647bc4a84efb3cc13d85e` | Exit 0 |

The actual report is `scanner-coverage-7f99406b04.json`, SHA-256
`17a7d33aa834365465e6a0f5dd2f89458acbd17b20976c930a3d66f23c833bf7`,
5,395,450 bytes. It names snapshot `f0db7e563313e1ace36906d54e67c9d6f953c79e`
with parent `7f99406b043193b86a9b5121ed8b419924e185a9`.
The two subsequent test-assertion edits are outside that report's source;
the scanner implementation is identical. Final validation binds source `6ded233c6c` separately; its snapshot is
`ada1127f7eafe72bb8b02b303adcc597ef8dcaf6`. All tracked blobs match their respective
source commits. Both native endings, four terminal logs and the two CAS
claims with result-payload hashes were checked successfully.

The raw report contains 3,909 paths: 3,908 repository paths and one
PrismaBuild snapshot closure file. It analyzes 2,079 Python paths, reports
1,163 non-Python paths, and records 667 existing excluded paths. Of the analyzed
Python paths, 505 belong to the default gate. It records 32,413 function
locations, 269 candidate pairs including 109 within one file, 744 same-name
groups, and 1,648 primitive digest sites outside the existing owners. These
counts describe this report population, not new defects or a larger baseline.

## Remaining program work

The package, tools and support census documents on PRI-226 retain their
bounded semantic findings and uninspected areas. This report makes the static
coverage gap visible; it does not erase those gaps. Non-Python behavior,
profile and lifetime differences, callers, authoritative owners and existing
#1301/#1303 migrations still need semantic reconciliation. No raw site here
supersedes the digest owner's current manifest.

The first publication migration remains separately approved in PR #3065;
merge proof is still required before calling it current-main delivery. Final
program closure requires an independent residual census on current main.

This scanner-only work changes no production hot path, cache residency,
calibration, numerical recipe, artifact bytes, default menu, plugin pin or
serving gate. It makes no speed, cost, KL, bpp, GPU or served-quality claim.
