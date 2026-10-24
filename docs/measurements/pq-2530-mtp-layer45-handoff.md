# #2530 / #2832 retained-evidence handoff (incomplete)

## Status and ownership

This is an additive documentary delivery, **not an authenticated completed owner
handoff or release qualification**. No capture or pricing rerun was performed.
No GPU, checkpoint, download, application implementation, or pricing change was
performed. The existing #2530 capture/pricing owner retains responsibility for
original-payload inspection and selectors. The existing #2980 W1 owner must
supply the immutable selectable-population/artifact handoff and authorize its
reuse. That artifact was not located in the current or served-main checkout;
shared-workspace research did not identify an authoritative W1 record either.
This is not proof that no such record exists elsewhere.

`pq-2530-mtp-layer45/handoff.json` is explicitly incomplete. Its W1 path, digest,
owner and membership selectors are null. Original Stage A/B, render and producer
references and render hash/size are also null: these are unresolved joins, not
opaque invented selectors. **O1–O3 and O5 remain unmet.** The retained observations
and O4 accounting are useful independently; they must not be mistaken for W1
membership or cell-payload authentication. No parallel population manifest was
created, and no authority or approval is attributed to this delivery.

## Implementation boundary

Existing `prismaquant/glm_mtp_capture.py` owns capture/final-hidden production.
Existing `prismaquant/tessera_joint_aura.py` owns joint pricing.
`prismaquant/glm_mtp_selection.py::enrich_mtp_cost_wires` owns offline M3/M4
joining: it binds merged provenance to M4 parts and M3 anchors, checks rung/wire
rosters and projection identity, and checks producer expert-wire receipts before
export intake hashes selected blobs. These mechanisms were not reimplemented,
rerun or weakened here. The separate #2771 protocol
(`docs/design/glm_mtp_option_b_protocol_2771.md`) is unchanged. This delivery does
not redo prismaquant-965653881d43-wp1 or reopen source-count/receipt joining.
#2983–#2985 serving, quality and performance packages are outside scope.

## Retained originals

All paths below are relative to `docs/measurements/pq-2530-mtp-layer45/`.

| File | SHA-256 |
| --- | --- |
| `audit.json` | `95d0c34279ccbdd2c11be4373f13101f51632f7f5c256e113ad5bba478f7974a` |
| `units.csv` | `64ce7611b5d7fbba2c39bd8b68f2a285fd15d1255eab05fb655a34a621807416` |
| `cells.csv` | `c8d30e8f73b2788af8b5813d7f496c35c18e0e79d080720064bbe0dca3c9c6b7` |
| `final-hidden-manifest.json` | `9819766aae872b78e4af54d9d93885e275789a73eda5315564a1f3c3540438d1` |

The first three are byte-identical Git blobs recovered from
`b10937aa316fcc2b20ae2d9f077d4cd80df77b3e`. This historical object is an
**evidence locator**, not evidence of merged implementation or qualification.
The final-hidden manifest was copied byte-for-byte from the original input
identified by the audit's `final-hidden` role; its digest was checked against
that anchor and the acceptance contract. Embedded fleet paths are historical
strings preserved in the originals, not portable runtime dependencies. Checks
read committed metadata only: no historical Git commands, fleet reads, tensors,
checkpoints or downloads are required to consume the retained package.

## Portable draft contract and semantics

The intended complete contract is documented in
`tests/test_mtp_source_handoff_2832.py`. This draft deliberately does **not**
satisfy its required non-null joins. Consumers must reject
`status: incomplete_not_owner_authenticated` as a completed handoff.

* `source_dispositions` contains exactly 867 original units: 864 routed and three
  shared. Capture digests and observed integer counts/maxima come directly from
  `units.csv`. All rows are held pending authentic W1 membership. A zero is an
  observed numerical value, never a fallback for missing evidence. Missing
  selectors are null. `source_bf16_passthrough: false` describes these historical
  quantized-cell sources, not authorization of any current W1 selection.
* `cells` contains all 3,468 original `(unit, rung)` pairs, preserving original
  inventory prices, wire hash, exact serialized byte counts, receipt
  classifications and match flags. CSV integer/float fields become JSON numbers;
  the original wire payloads are **not** delivered or independently rehashed by
  this package. Historical `producer_verified` and match flags remain historical
  audit observations, not a new verification result.
* Stage A stores `stage_a_output_mse` / `output_mse`. It is **not** self-KL.
  Stage B's objective is `mtp_head_self_kl`; its original price is
  `stage_b_predicted_dloss`. No currency or provenance was relabelled.
* `TESSERA_BF16_*` is a quantized grid, not source BF16 passthrough. All 12 shared
  quantized cells have explicit hold dispositions. Their shared-selection
  receipt limitation is not repaired by the existence of prices, bytes or a
  checkpoint anchor. They need inspection of the original shared receipt chain
  and selection eligibility before any qualified use.
* `dev_stamps` retains all four original Stage B prepare/run stamps for R896 and
  R1024, including `dev_uncertified: true` and their producer-source digests.
  Every cell remains `dev_uncertified: true`, `release_qualified: false`.
* `calibration` records 512 sequences of length 512, totaling 262,144 tokens;
  artifact and token-byte hashes are copied from `audit.json`. `final_hidden`
  contains all 512 original sequence observations. Sequence is joined by
  `records[*].metadata.identity.coordinates.batch` to `head_check[*].sequence`;
  tensor hash, `[1,512,4096]` shape, top1 and nll are copied without rounding.
  This verifies metadata accounting, not tensor bytes or fresh model execution.
* `later_reprobes` is empty: none were performed or promoted into original
  observations in this delivery. Historical probes already embedded in the
  byte-identical audit remain historical probes, not new evidence or permission.

## Unresolved joins and owner action

1. **#2980 existing W1 owner:** deliver the authentic immutable W1 artifact,
   digest and attributable owner record. Inspect all 867 membership selectors
   against that artifact. Inventory equality is not membership authentication.
   No W1 identity can be inferred from #2619 inventory totals.
2. **#2530 existing capture/pricing owner, with reviewer:** inspect the original
   inputs identified by audit roles `m3-{r1024,r896}-cost`,
   `m4-{r1024,r896}-price`, `m4-{r1024,r896}-production-cache`, and the producer
   `m3-{r1024,r896}-{receipts,checkpoint}`. Resolve all 3,468 unit/rung selectors,
   render digests/sizes and original producer joins, including the 12 shared
   cells. These roles locate historical candidates; this delivery does not
   assert that uninspected pickle keys identify the claimed payloads. Underlying
   workspace locations were found during research, but no full original-payload
   inspection or authenticated crosswalk was established. Missing inspection is
   not evidence of absent payloads.

No later re-probe, regenerated population, invented JSON pointer, guessed pickle
key or substitute measurement can discharge these dependencies. O5 requires
substantive owner/reviewer inspection even if CPU schema tests eventually pass.

## Certification refusals and qualification holds

The retained audit records refusal of all four body-admission payloads with
`joint AURA requires matching aura/joint provenance`. It also records the
current-owner refusal `MTP M4 part 0 wire roster differs from merged cost`.
The audit's relabelled-provenance admission probe is not authorization to change
currency or certification. All original refusals and dev stamps are preserved
byte-for-byte, and no release hold is lifted here.

Historical source-count CPU regressions and PR #2619 inventories are not
original-model, serialized-byte, numerical-quality or serving qualification.
These checks cannot authenticate owner authority, W1 selector meaning, original
producer/render contents, valid certificates, end-to-end quality or serving.

## Verification

Command: `python -m pytest tests/test_mtp_source_handoff_2832.py -q --basetemp="$TMPDIR/pytest"`.
Observed result: **3 failed, 1 passed in 2.58s** (exit 1). A final run after
JSON whitespace compaction returned **3 failed, 1 passed in 0.69s**, with the
same missing joins. Both retained stdout artifacts were read back. Retained stdout was
read back using `read_test_artifact` and verified. O4's complete original
calibration/final-hidden accounting passed. O1 fails at the null W1 artifact
path; O2 fails at a null W1 membership selector; O3 fails at the null render
digest. These are incomplete deliverables, not acceptance-test defects. All
four retained-original digest checks passed before O1 reached the missing W1
path. No GPU qualification is claimed.

Final rerun:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"8dd3eda395c1e26a485e62025365650ca9eb62196b1f58f6327873881769962f","size":5738}},"operation":"288ddc1c38d7630635cf020df2a1e6032eef85e62ddc3ee3fea4ca3a954bad48","receipt":"700dee07b66ce5fdc9b7d96743ced4745395c9a9e4d173426533d3473d4b397d"}]

Initial run:

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3c95648fa7d78e98b79f74de712b2fb3bdeb23e7b6e2ee7d76feb54fc2965d9e","size":5677}},"operation":"24604fb0bb6d3d8b2bb687646c22e3b7250da72c7f01a8384f4e293708eaf56d","receipt":"f7d11bb1e2c3a1c7f2edb138331717254917357cfebf5235f6b347f526fc7a59"}]
