# Exact body source selector qualification — 2026-10-02

PQ #2032, Refs #1962. This extends the existing snapshot-only selection owner;
no original weights were read and no GPU or full-source qualification follows
from the selector API alone.

The isolated L05 replay needs declared non-Linear KDA/mHC state and router bias,
besides selected Linear weights. `configure_selected_source_tensors(keys,
layers=...)` accepts those exact declared body keys under an explicit layer
allowlist. The existing `configure_selected_snapshot(names, profile)` still
uses the unchanged selected-weight dependency resolver and delegates its
transition to the same owner. Both source maps commit only after header
authentication and size estimation succeed. Empty, duplicate, undeclared,
nonbody and foreign-layer selections refuse before headers; active/already
configured contexts refuse another transition. Forward and complete
initialization guards are unchanged.

Published PrismaBuild CPU receipts, submitted from dl380g10, native threads 1,
CPU 2 and aggregate memory 8 GiB:

- `5a6dc7cdc55774c4e464ee9fe3458ac003b05e2e2e1c0ae5fdb02d40f7b2af7f`,
  parent `c08af10544c0504124eac5361e202078f28d81cb`: **37 passed, 2 failed**,
  no skips. The new selector/refusal/rollback cases and architecture/staleness
  tests passed. Two existing GLM snapshot fixtures omitted the explicit source
  derivative declaration required by the corrected image and refused correctly;
  this containing action is not a suite pass. Image PB content `b5cfa805...`,
  actual image content `d0256efb...`. CPU/GPU-unattached scope, cleanup complete.
- `fb6f280aab5a9d39dd1f89f3d3f1eb63179e2a444957842b705cd2a55f1a9488`,
  parent `669b04e3c33e0e45d1651bf14393595e4318ab79`: **2 passed, 18 deselected**,
  no skips, exit 0 and cleanup complete. Only those two unchanged fixtures ran
  in stock Transformers image PB content `83db39fd...`, actual `eb8592ab...`.
  CAS receipt `772697307f44d6e075aae0199378af23a1b1a032dc4fd39ff14aa627eb875753`,
  result `7039bb2b8719a3f09581aec789af3d4d532ab70387b9e8f725a0df565442f05c`.

Commands were `python3 -m pytest -q tests/test_selected_snapshot_scope.py
 tests/test_architecture_doc.py tests/test_docs_staleness.py` for the first
receipt, and `python3 -m pytest -q -o cache_dir=/tmp/pq1962-stock-pytest-cache
 tests/test_selected_snapshot_scope.py -k glm_snapshot_preserves_source_bytes`
for the second. Both ran in the existing admitted container frontend. This is
37 corrected-image cases plus two stock-image fixtures, not blanket credit
from an all-green corrected-image suite.

Raw sealed requests, source bundles and logs were read and hash-verified.
`/mnt/shared/tessera-measurements/pq1962-sol-20261002/pq2032-selector-source-bindings.json`
records exact snapshots, request/source/CAS hashes, byte-exact test module
comparison and AST equality of all three selector methods against the review
code. Main-base integration changes no qualified selector/test owner.

The separate experimental whole CPU source/cache/isolation/coordinate replay
passed `86aebc0893c05f431f7fc4190acb685196e76ceba1e7c869bfbb4a7e939af848`,
parent `17f832853f7f9eedb10303a1959a2a8056eb4de5`. All 17 input opens used RAM
leases (1,372,510 bytes), with no pool reads, misses or fallbacks. The synthetic
38-key KDA/mHC/MoE layer and 15 MLP units preserve resident-reference coordinates
and row terms; fixed-g/operator discrepancy is 4.8733e-7 of reference RMS.
The entrypoint and GPU lifetime/backend controls remain separate experimental
work under #1962; this PR carries only the shared selector, tests and contract.
