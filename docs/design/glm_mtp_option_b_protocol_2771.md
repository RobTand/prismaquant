# GLM MTP Option B — bounded adoption and approval status (#2771)

Status: **qualification held; no approved content-bound comparison protocol
established from the inspected evidence.** This is the single local handoff
record, not a new execution plan, submission, or qualification result. Only the
cited historical bindings and supplied scope constraints are settled here;
missing campaign decisions are not frozen or approved by this document.

Evidence baseline: PrismaQuant main
`914bf79807120fa60f433714e95cffd44b917d92` (M below). Historical objects
`b10937aa316fcc2b20ae2d9f077d4cd80df77b3e` (H) and
`086618fb1ff351af5ba3992b6b07db8b980ac16c` (E) are not ancestors of M.
A reference of the form H:path means that exact Git object and path, not the
current file or evidence that the child is complete today. No artifacts are
copied into this record.

## Immutable inputs and panel

Reconciliation precedes adoption: see the child ledger below. M's
[`mtp_rung_selection.md`](mtp_rung_selection.md) is a normative selector policy
(attributed there to Robert, 2026-07-20), not an approved, content-bound GLM
Option B comparison protocol. Its throughput formula and calibration requirement
do not settle whole-artifact body tradeoffs. M's `glm5_next.py:718–755`,
`allocator.py:912–953`, and `docs/ARCHITECTURE.md:9985–10021` establish dedicated
MTP source scope and independent sub-budget selection before body allocation,
not completed Option B. E's enumeration implementation is likewise not a
comparison approval. No adequate approved owner protocol was found to adopt;
if one is supplied, adopt its exact immutable reference here after adequacy and
authority review, rather than author a competing campaign.

Historical bindings available for reconciliation (not new execution inputs):

| Input | Immutable evidence / disposition |
| --- | --- |
| Source | H:`docs/measurements/pq-2530-mtp-layer45-manifest.json`: GLM-5.3-Flash-BF16 source content SHA-256 `c45d2d35b4520d729e1026b926dc1eb0de9d8fc905d46733c7edd322af2009a0`, source identity file `4db808a769eb4fa4fa3df3c30c9857bd197200555355671e1c0c84ee8faefdfd`. Prospective source adoption unresolved. |
| Calibration draw | Same manifest: calibration SHA-256 `dd5be8c887a48465fcd7df567b8aba876026339785da03f6642c7179413668e1`, artifact `9cd1fa129f249abd80d22efaeb8bc7e8b2d3b4252f173a8c6f2b2e496a4f8329`; probe `eab33bfbfc744aa46d7bbb5072ded168804492331fb46c2214cb10ff63703b9e`, objective `mtp_head_self_kl`. These do not bind a new evaluation sample set. |
| Teacher and tokenizer | Historical BF16 source and probe do not independently authenticate the comparison teacher, tokenizer files/revision, special tokens or prompt rendering. Exact content identities and owner attestation unresolved; source hash alone is insufficient. |
| Rival | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` revision `a5fee929cf4888b1824323e33e8a19b60129e025`; M:`docs/measurements/glm-mtp-bf16-fixed-2026-09-26.md` (R below) records inventories, not a complete newly authenticated per-file inventory. Preserve this revision; inventory digest/authentication for new comparisons unresolved. |
| Configured Tessera/runtime | H records producer commit `07bfcc0e9b7da13276938cb722bc7dcd893e6c63`, encoder source `da5805bc707c041efaeb27c5083fb7809fd8cdc51d501f41103b791270b569cb`, container content `d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026`. This producer tuple is not a prospective serving tuple. Exact configured Tessera commit, contract hash, runtime/plugin/image identities, rival runtime and configuration digests remain unresolved. A current default pin cannot substitute for campaign approval. |
| Workload and tensor parallel scope | R's intended eager one-step TP2 census did not run. No executed TP2 qualification exists in R. Prospective TP scope, hardware/memory scope, workload content hashes, prompts/token indices and decoding configuration are unresolved. No TP1/TP2 or eager/compiled evidence substitution is authorized. |

Bounded panel disposition: the requested panel is **two admitted quantized MTP
choices**, one **authenticated BF16-MTP control**, and the named rival; no
approved bounded-panel exception was found. Admission and identities of the two
choices are unresolved, not selected here. R's compact `TESSERA_BF16` grid R1024
and earlier E4M3-grid R1024 are historical quantized leads, not two newly admitted
choices. Compact BF16-grid wires are not the unquantized BF16-MTP control.
The control needs authenticated source BF16 tensors and a content-bound body
assignment under the same authorized accounting. None is established here.

Coverage cannot shrink: H accounts for **867 units = 864 routed + three shared**
layer-45 Linears, with 3,468 priced cells and 12 shared wire cells. All 864 routed
units and all 3 shared units require candidate-specific coverage and receipt
joins. Unsupported cells remain explicit holds, not exclusions: H records
uncertified execution, certified refusal of routed `(4096,2048)` and
`(2048,4096)` shapes, and no dense shared-unit selection-receipt path despite
verified shared wires. It also records body admission dependent on payload
provenance (relabelled provenance was admitted). Historical price presence is
not runtime admission. The existing child owners must provide matching full
coverage, eligibility and provenance evidence; Rob as campaign decision authority
must approve input adoption. Missing identities or unsupported/shared evidence
block affected candidate admission and the full-panel comparison.

## Byte accounting

**Global cap: unresolved; no exact authorized integer cap for this prospective
comparison was found.** Authority is Rob (campaign policy), with accounting
supplied by the existing children. Neither the agent nor a selector may turn a
historical inventory into a new authorization.

R at M retains the exact rival revision
`a5fee929cf4888b1824323e33e8a19b60129e025` and these distinct quantities:

| Historical scope | Rival | Historical export |
| --- | ---: | ---: |
| 120 safetensors shard bytes | 175,642,157,752 B | 175,576,305,954 B |
| All-file bytes (every regular file) | 175,790,111,275 B | 175,643,087,583 B |

The historical strict **all-file** ceiling was derived from the rival's
**shard** inventory: 175,642,157,752 B. The actual export exceeded it by
**929,831 B**. Rob's historical decision waived that ceiling criterion in PR
#1416 and left the artifact as exported; the criterion was **not met**.
Being smaller than the rival's larger all-file inventory does not erase this
miss or authorize 175,790,111,275 B as a replacement cap. R's export action and
read-only gate both exited 1 without CAS receipts; a finished exporter is not
a successful qualification gate.

Historical estimate only: selected payload 175,542,306,328 B, prior observed
body overhead 88,611,981 B, plus 4 MiB metadata allowance, gave reserve
92,806,285 B and upper estimate 175,635,112,613 B. Actual all-file measurement,
not this reserve estimate, exposed the miss. These are not new reserve rules.

Required authority disposition remains on hold:

| Unresolved accounting item | Evidence/decision needed | Qualification affected |
| --- | --- | --- |
| Exact integer global cap and scope | Rob's attributable content-bound decision reconciling shard versus all-file scope and the waived miss, plus full rival file inventory | Every byte-matched comparison |
| Fixed regions | Immutable inventory of body, MTP, passthrough/shared tensors, embeddings/head and other fixed regions, with sharded/replicated and counted-once rules | Candidate and control body allocation |
| Reserves | Exact integer container/metadata/index/tokenizer/config allowances and treatment of any other emitted regular files; actual post-export inventory reconciliation | Budget compliance |
| Byte-match tolerance | Exact integer tolerance and comparison rule authorized by Rob; neither zero nor 929,831 B is inferred as the new tolerance | Matched-byte qualification |
| BF16 control body accounting | Exact BF16 draft bytes, all other fixed bytes and reserves under the authorized scope; content-bound control body allocation and no double counting | Control validity and quality comparison |

Accounting invariant, not a selected budget: within an approved scope, body
room equals global cap minus fixed non-body/non-draft regions, charged MTP
draft bytes and reserves (with disjoint counted-once sets). For a BF16-MTP
control, the full BF16 draft charge reduces body allocation; it cannot inherit
a quantized-draft body's allocation for free. R's historical source MTP charge
14,545,846,272 B and replacement 3,707,898,528 B explain that export's arithmetic,
not an authenticated new control's byte charge. Disk budget and resident-memory
fit are separate constraints. All new numeric decisions above remain blocked
pending authority, not delegated to this document.

## Prospective evaluation

No new results are collected or authorized here. Existing results remain
retrospective. The following supplied correctness and comparison constraints
are retained; a complete numerical protocol is **not frozen** while the table
below is unresolved. Approval must precede any new candidate results eligible
for prospective qualification.

- Greedy speculative-on output must match **each target's own speculative-off**
  output with **zero token mismatches** on identical bound prompts, token limits
  and decoding settings. The rival is not the greedy token oracle for a changed
  target body. This does not replace stochastic distributional correctness.
- Body quality is measured **without speculation**, against the authenticated
  teacher on the bound evaluation draw. MTP-head self-KL is not body KL and is
  not measured served acceptance.
- Keep three separate Pareto axes: **minimize body KL**, **maximize measured
  acceptance**, **maximize decode speed**. No agent-selected scalar exchange
  weight, modeled throughput or historical self-KL proxy replaces these axes.
  Rob must approve any trade among nondominated candidates by reference to the
  exact candidate identities, uncertainties and protocol; no automatic winner
  or scalar tie-break is authorized.

| Unresolved evaluation rule | Required existing-owner evidence / authority decision | Hold |
| --- | --- | --- |
| Workloads and sample counts | Content-addressed correctness, quality and performance draws, exact sample/token counts, strata, seeds, lengths, batching/concurrency and stop rules; child owners supply, Rob approves | All prospective comparisons |
| Stochastic tests | Sampling parameters, seed coupling policy, distributional test/statistic, exact sample size, significance level, power/effect-size and multiplicity rule | Stochastic correctness |
| Numerical/quality bounds | Exact dtype/reference, body KL aggregation, PPL/task metrics if required, finite-value checks, numeric tolerances and allowed quality degradation with reference denominator | Quality admission |
| Warmup and repetitions | Exact excluded warmup steps, measured repetitions, pairing/order, cache/graph/residency conditions and timing boundaries; no post-result tuning | Decode speed and acceptance comparisons |
| Counter attribution/reset rules | Exact accepted/drafted-token counters and denominators, per-position versus aggregate definitions, target/draft attribution, rank aggregation, reset/snapshot boundaries and failure/restart treatment | Measured acceptance and timing validity |
| Uncertainty | Exact confidence interval/estimator, dependence unit, outlier/missing-run policy, simultaneous comparisons and inconclusive/tie rule | Pareto dominance and trade approval |

For every row, the existing measurement child owners must supply the immutable
rule record and Rob must approve its numerical policy or cite an already
approved adequate policy. Current owner identities/approvals are not established
here. Until then, affected qualification is withheld; no sample count,
numerical bound, repetition count or confidence level is invented. In-process
profiles and aligned two-host power/system telemetry required by repository
measurement policy are not supplied by this CPU documentary check. The R TP2
census did not run and contributes no counter, correctness or speed evidence.

## Child handoff and historical reuse

This ledger binds the work to **#1271's existing campaign**, not a parallel
execution campaign. Search of available main documentation and the supplied H/E
historical leads established the following limited reconciliation, not live
issue closure or authenticated contact with owners:

| Existing child | Available immutable deliverable | Adoption/reuse disposition |
| --- | --- | --- |
| #2530 | H:`docs/measurements/pq-2530-mtp-layer45-qualification.md`, `pq-2530-mtp-layer45-manifest.json` and `pq-2530-mtp-layer45/{audit.json,units.csv,cells.csv}` | Reuse only historical capture/price/coverage evidence within its limits. The manifest binds 45 inputs via audit.json SHA-256 `95d0c34279ccbdd2c11be4373f13101f51632f7f5c256e113ad5bba478f7974a`; it makes no performance claims. Not prospective approval. |
| #2532 | E:`prismaquant/allocator.py`, `prismaquant/glm_mtp_selection.py`, `tests/test_mtp_option_b_2532.py`, `docs/ARCHITECTURE.md` | Historical opt-in enumeration work, emitting `prismaquant.mtp_option_b_candidates.v1` with `winner_selected: false`. Not evidence of a measured candidate panel, current-main integration or approval. Do not duplicate enumeration implementation. |
| #2534 | No attributable protocol/deliverable located in inspected main docs or H/E docs | Current owner, immutable deliverable and approval unresolved; no completion inferred. Existing owner must identify authoritative record and demonstrated scope. |
| #2536 | No attributable protocol/deliverable located in the same evidence | Same unresolved handoff; no replacement workstream or invented role assignment. |
| #2539 | No attributable protocol/deliverable located in the same evidence | Same unresolved handoff; no result or approval inferred. |

H records 181 audit checks passed, 13/14 accepted actions in dev mode and all
four Stage B results `dev_uncertified`. A CPU audit does not certify those
runs, repair shape qualification, or create the missing shared receipt join.
Its original merged prices also lack current `wire_rungs` provenance and fail
current receipt joining. Reuse needs the existing owners' matching identities
and compatible joins, not blanket requalification or silent provenance edits.

H's audit action is
`9d713876e24dee84e2f8b9754e75a65c1f266b1a6a4b189b5f62aea6b4a8b934`,
recorded receipt SHA-256
`02a4f63516b3763d13dc7a9d8a63290731ee8151baeda02eefd82509a7409af7`.
This record inspected the historical manifest, not a fresh fleet receipt
verification. Its hashes locate evidence; they do not authenticate new approval.

R's fixed selection, export inventory, original wire receipts and preflights
are historical only. Reuse is limited to the exact source, assignment,
producer/runtime, contract and declared scope each actually demonstrates;
changed identities require owner disposition. R's path-expansion correction
must be used to locate relative `m6/...` and `integration/...` receipts:
`/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/ws-mtp-20260925/`,
not the campaign root without `ws-mtp-20260925`. These are historical locators,
not deployment assumptions or required local paths for CPU checks. No previously
observed outcome becomes prospective by attaching this record.

**#2772 owns reconciliation of #1271 parent acceptance** from the existing
children and current qualified receipts; this record neither closes children
nor decides parent acceptance. **DFlash #2674, #2675 and #2676** own a separate
drafter arm and are excluded from this bounded MTP quantization panel. No source
mapping, MTP-head self-KL, allocation, receipt or export implementation is reopened.

## Ownership and approval

Authority: Rob is the campaign policy/approval decision authority identified by
the supplied #2771 scope; each existing child owner retains its deliverable.
The supplied issue has no approval comments establishing a complete protocol.
No current completion state or authenticated owner approval for all five children
was established. Historical Git authorship, normative selector status, a receipt
hash and a historical waiver are not approval of this document's content.

Approval status: **unresolved / not approved for qualification**. This is a
bounded status record, not a proposal to execute its gaps. No physical action,
credential or login is requested. Missing decisions are qualification holds,
not permission to choose policy on the authority's behalf.

To discharge a hold, the existing campaign authority's attributable decision
must reference the exact protocol content (commit plus path or content digest),
the adopted owner protocol if any, immutable input manifests, candidate/control
assignments, authorized integer accounting and numerical evaluation rules.
Approval must identify the approver, decision reference and scope, with evidence
that prospective rules were fixed before eligible new outcomes were observed.
A mutable issue title, abbreviated runtime pin or unbound 'approved' label is
insufficient. Changes to bound content require explicit authority disposition;
no existing approval is presumed to float to changed inputs.

The unresolved input/panel, accounting, evaluation and child handoff entries
above **block affected qualification**. Rob's authorization and the existing
owners' immutable evidence are needed to lift them; no new cap, exception,
scalar trade weight or unsupported-cell exemption is granted here. Reviewer
inspection under O5 must assess approval authenticity, temporal prospectivity,
full accounting and semantic adequacy of any adoption. CPU regression tests
only check that this record exposes the decisions and historical discrepancy;
they cannot establish approval, performance, correctness or qualification.

No application code, runtime pin, admission gate, format, default or budget is
changed. No GPU execution, submission, publication, served claim or new campaign
is authorized or performed by this documentary deliverable.
