# Full original-roster GLM anchor preparation — 2026-09-08

These are reviewable **drafts**, not a submitted or qualified GPU campaign.
They preserve all 132 whole anchor groups and 36,423 original units. Root owns
acceptance of the complete original capture, final producer and PrismaQuant
source freeze, full-engine serving integration, native qualification and launch.
The capture SHA placeholder deliberately prevents these draft rows executing.

## Concrete artifacts and route

- `anchor-spec.draft.json`: original calibration contract, entire readable
  envelope (`--max-artifact-bpp 0`), selected capture reuse, bounded source
  residency, scalar anchor calls, exact producer source mount and container.
- `invocation.draft.json`: actual dispatcher plan/submit command arguments,
  operational timeout, qualification demands and remaining gates.
- `qualification-manifest.draft.json`: two concrete review-only rows derived
  through the existing dispatcher `_row`, retaining their original row paths.
- `preflight.py`: PB CPU inspection of source identity, menu, all original input
  file sizes, original token equality and whole-group resource admission.
- `draft_rows.py`: PB CPU draft-row authoring helper. It neither creates a fake
  complete capture nor bypasses the actual dispatcher's capture validation.

After the capture and sources are accepted, copy the exact original census
bytes into the new workspace, freeze the spec and run the existing dispatcher
`plan` through PB CPU. Require all 132 admissible rows and all 36,423 distinct
original units, with one whole group per row. Instrument master rows `row-0076`
and `row-0087` once, following the draft. Copy those same row objects into the
two-item qualification manifest. PB owns placement and retries. Retain the
identical instrumented rows in the full master manifest so successful actions
reuse their receipts and interrupted work resumes the same journals. Submit
through the existing dispatcher; merge only complete, validated full rows.

The old `8` bpp draft ceiling was removed with root authorization because it
would leave legal readable BF16-family rungs unpriced. Six downstream allocations
choose their budgets from the resulting measured surface. This does not change
quantizable-only bpp accounting or make research-readable formats a ship default.
There is no sampling, rate band, old-source wire seed, selected forward,
recapture, campaign deadline or finite refinement-round cap. Existing limits
remain anchors 3, budget 12, LOO gate 0.25, anchor batch size 1, 512 original
samples, sequence length 512, seed 0 and retained activation rows 512.

## Measured CPU preparation

PB action `336f6b64a65670398d0ed46114fc4f6de0584f35e02dd5178b1edf9a58fd6bbc`
completed on `dl380g10`, exit 0, two CPUs and 4 GiB reserved. It read metadata,
reconstructed tokens and inspected source; it did not read model weights or
activation/Hessian payloads, execute a model forward, or use a GPU.

| Observation | Result |
|---|---|
| Original census | 132 groups; 36,423 units; 62,409,468 bytes |
| Original per-unit files | All 36,423 regular, nonempty files inspected |
| Largest serialized input | 629,147,557 bytes |
| Largest Hessian | 603,979,776 bytes (576 MiB), input width 12,288 |
| Largest retained activation prefix | 25,165,824 bytes (24 MiB) |
| Existing selected-source memory plan | All 132 rows fit 104 GiB; maximum 98.041520916 GiB, rounded reservation 99 GiB |
| Original calibration equality | Reconstructed int64 IDs exactly equal the sealed original token artifact; fit/text hashes match census |
| Four actual Linear shapes | `[2048,4096]`, `[4096,2048]`, `[12288,4096]`, `[4096,12288]` |
| Readable menu per shape | 5,635 rungs: 1 E2M1 K2, 1,793 E4M3 K1 and 3,841 BF16 K1 |

The current campaign reconstructs the original deterministic draw through its
existing tokenizer/dataset helper; it does not directly consume the token
artifact. The CPU comparison proves equality for these inspected dependencies.
Each real row still checks the canonical census token/text contract. Source or
dataset changes require another comparison rather than assuming equivalence.

The Hessian-reference policy uses the actual largest input-file and Hessian
sizes above. The 128 MiB metadata cap is the existing producer hard limit and
remains provisional until final canonical, census and reference-descriptor sizes
are checked. The producer reader performs bounded file access; the selected
row's existing resident Hessian path remains the encoding input.

## Qualification and measurement limits

| Original master row | Scope | Base memory | Observer allowance | PB CPU / total memory |
|---|---|---:|---:|---:|
| `row-0076` | Whole layer 4 expert group, 864 units | 99 GiB | 4 GiB | 8 / 103 GiB |
| `row-0087` | Whole layer 0 down projection `[4096,12288]`, 1 unit | 31 GiB | 4 GiB | 8 / 35 GiB |

Both are `measurement: true`, `host_class: gb10`, without a host mapping. All
other master rows retain their ordinary six-CPU, derived-memory demand. The
observer extends the existing `experiments.glm_full_capture_profile` monitor;
it wraps the actual scalar `_measure_anchor` call without retaining input
weights/activations or changing the call arguments, output or scheduling.
It records CPU/CUDA windows for call indices 0 and 31 when reached, owned-Python
RSS/I/O each second and Netdata from both Sparks every five seconds. The dense
row can finish before index 31; a valid index-0 CUDA window suffices, with the
unreached index explicitly reported. A journal-only retry says
`native_anchor_profiled: false`; it does not manufacture fresh CUDA evidence.

Per-window trace acceptance is capped at 512 MiB after export. This is **not**
a hard live profiler-memory or disk-growth bound. The 4 GiB observer allowance
is provisional; actual native resource peaks must fit the reservation. Python
sampler output is capped at 512 MiB and Netdata output at 3 GiB per attempt.
Each retry gets an attributable attempt directory. A timeout, missing CUDA
events, oversized trace, insufficient telemetry or incomplete outputs fails
qualification; none justifies shrinking the row or relabeling partial success.

The 14,400-second PB timeout comes from the existing dispatcher default. It is
an operational bound, not a measured completion estimate. Current menu/budget
settings allow 7 initial distinct pairs per unit and at most 25 successful
unit/family/rung pairs: 6,048 initial and 21,600 maximum for the expert row;
7 initial and 25 maximum for the dense row. Four hours for the expert maximum
would require an average at most 0.667 seconds per successful pair including
overhead. That rate is unmeasured; previous tiny native fixtures cannot justify
it. Root must assess real progress and complete journaled rows before accepting
qualification. No full-model runtime or GPU memory success is claimed here.

## Source and evidence boundaries

The prospective raw producer archive is commit
`9d2314819f027e53ba169039a10cd27594d36670`, archive SHA256
`30bf315c472062bf3aca1d47b76d21d3a92f157711baae68cf0f73a1739d875b`.
All 1,176 regular source files were checked against the archive. The CPU-observed
raw encoder seal was
`8fcfafdbcce1fc56e30e38b1fe848e1b7602ca3360e40bc6b4497d74d10cb5e4`;
the runtime-contract SHA was
`a688f8de244f936ec3a63a782e20af7985733e7a6fb0b4b981b5fe4c44112212`.
These bind the raw source tree, not an installed wheel. The nested read-only
source mount and `PYTHONPATH` bind pricing/export to the same tree. A serving
package edit can change the conservative encoder seal: freeze the final source
before pricing, re-audit changes and never relabel existing wires.

The content-qualified producer container is
`prismaquant-glm-producer:content-qualified-20260908`, content SHA256
`eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`.
This is selected original-capture anchor reuse; any later joint-backward run
uses its separately reviewed environment. `PRISMAQUANT_TESSERA_DEV_PIN` stays
unset. The preflight's package SHA and route-status observations predate root's
pin/source integration; they are historical observations, not the final pin
or a claim of full-model/TP2 shipping qualification.

Shared evidence root:
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/full-anchor-preparation-01/`.
`preflight.json`, `producer-source-staging-9d2314819.json` and
`verified-preparation-actions.json` hold attributable preparation results.
Observer PB tests: `1388f214f7ff` (26 passed, one native CUDA capture skip);
compile: `c4df16f54ef0` (exit 0). Their terminal records, stdout, CAS payloads and
exact tested source snapshots were independently checked; CAS checks do not
assert signer-attestation verification. The initial `1ac38487365b` preflight
failed JSON serialization after its calculations and remains failed evidence.
The observer has not yet been measured on either real qualification row.

Preparation exposed and fixed an adaptive no-progress bug in separate commit
`85f714dfd5`; see `../tessera_adaptive_progress_20260908.md` for the before/after
regression and 50 passing PB CPU tests. Successful member/rung prices survive
failed siblings, no-progress rounds refuse after journaling, and resume reaches
missing interior work even when endpoint anchors were already journaled.

Draft-row authoring PB action `562025acf49cbd142bf2b4626346b291c0c63199623d656cc5a0fbb57f3c76f1`
completed CPU-only, exit 0. Both resulting rows were inspected for original
paths, scalar observer routing, 103/35 GiB demands and the unfrozen SHA. Stdout
and CAS payload checks passed; generator, dispatcher and spec bytes match the
sealed executed snapshot. See `verified-draft-rows.json` and its action receipts.
