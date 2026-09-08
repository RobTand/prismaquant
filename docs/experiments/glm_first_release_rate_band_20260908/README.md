# First GLM artifact: narrow-band CPU derivation

The candidate first-proof pricing configuration is body `--rate-band 832,1088`
(3.25–4.25), with an explicit routed-family restriction to `TESSERA_E4M3_K1`
and the existing readable dense families. The current campaign does not yet
expose that restriction; this report is a CPU plan, not a launch authorization
or a serving qualification. Every one of the original 36,423 Linears remains
in the empirical roster, in the same 132 whole-group PrismaBuild quanta.

The comparison budget is **155,668,854,528 tensor-payload bytes** over exactly
305,915,756,544 common quantizable parameters, or **4.070894713933706 bpp**.
It is the root's complete 120-shard EXL3 header audit
`exl3-first-artifact-01/root-common-surface-header-audit.json`, SHA256
`245bc10daacd7f0f2626b5d857d9ba1341ec89c9bab17cdb6fccf8927a611472`.
It includes the EXL3 BF16 charge for 135 eligible dense MLPs, excludes pinned
regions and excludes extra MTP tensors absent from the common capture.

All paths below are under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/` unless absolute.

## Exact byte witnesses

The existing `expand_tessera_menu` and `tessera_exact_bits_for_shape`
accountants supplied intrinsic wire bytes, including every plane. Their
uniform points establish budget feasibility only; the artifact must still
allocate from per-Linear empirical measurements, obey packed stack constraints,
and audit its actual exported bytes, including outer wire/container envelopes.
No KL or runtime quality is inferred from these numbers.

| Dense choice in the witness | Uniform routed E4M3 q256 | Common intrinsic bytes | Common bpp |
| --- | ---: | ---: | ---: |
| E4M3 at the same body rate | 1036 | 155,546,148,864 | 4.067685839297466 |
| Tessera BF16 family at the same body rate | 1036 | 155,548,360,704 | 4.067743681103981 |
| E2M1 K2 at its only readable rung, 896 | 1036 | 155,534,298,624 | 4.067375943785477 |
| Retain the baseline's plain BF16 dense payload charge | 1021 | 155,569,618,944 | 4.068299605133268 |

The last row is a byte witness that retains the EXL3 dense payload charge,
not a proposed Tessera BF16 wire. The producer's `serving/scheme.py:206`
publishes only `TESSERA_FP8` in `MOE_BUILDERS`, so routed Tessera BF16 and
NVFP4 wires are not physical routed candidates. The unrestricted uniform
BF16/E2M1 calculations in `derivation.json` are abstract accountant results;
`physical-family-analysis.json` supplies the structural interpretation.
Neither file promotes a pending runtime release or a missing serving cell.

The alternative 768–1024 band tops out at 4.020810839297466 common bpp for
all-E4M3, leaving 1,915,180,800 bytes unused. The 896–1152 band brackets the
budget too, but extends its upper endpoint to about 4.52 bpp on routed shapes.
The proposed 832–1088 band surrounds the affordable rate without that extra
upper extension. This is a scoped starting band, not a measured quality optimum.

All three unrestricted bands produce 182,115 initial per-Linear anchors:
two BF16-family, one E2M1 K2 and two E4M3 anchors per unit. Restricting only
routed units to E4M3 gives **73,251 initial anchors**: 72,576 routed and 675
dense. This count is the initial endpoint schedule; adaptive interior anchors
and any declared audits are additional. It is not a GPU-time prediction.

## Compatible-batch admission

The baseline plan's per-phase terms were recomputed for every row. For each
width, the widest row was independently checked by the existing
`_streamed_resource_plan(..., selected_source=True)`. The only changed terms
are the existing compatible-weight batch and encoder memo widths.

| Batch width | Widest rounded GiB | All 132 fit 104 GiB | Fits with prior extra 4 GiB observer allowance |
| ---: | ---: | :---: | :---: |
| 1 | 99 | Yes | Yes |
| 2 | 99 | Yes | Yes |
| 4 | 99 | Yes | Yes |
| 8 | 100 | Yes | Yes |
| 16 | 101 | Yes | No |
| 24 | 103 | Yes | No |
| 32 | 104 | Yes | No |

Width **8** is the largest checked width retaining that observer allowance.
This is a planner bound, not native fit or a speed measurement. The previous
oversized native trace remains a failed observer result; a bounded before/after
profile and both boxes' telemetry are still needed to qualify batch throughput.
Batching remains inside each admitted whole-group action and changes no PB
placement or sharding. Dense anchors remain scalar under the existing adapter.

## Reuse and implementation boundary

The two completed journal entries are BF16-family R256: the layer-4 expert-0
down projection and layer-0 dense down projection. Their recorded encoding
times are respectively 17.139 and 45.689 seconds. These are two different
completed encodes, not representative timing estimates for the new band.
Their envelope payload hashes were checked; this CPU analysis did not re-read
or re-verify the large wire blobs. Both remain outside the proposed band, and
the routed family has no physical stack builder. Retain them as bounded
historical evidence; do not seed them into the first-proof campaign.

`rate_band` is already part of journal identity, so changing it refuses an
ordinary resume. Existing seed adoption checks wire identity but does not
restrict admitted old anchors to the new band. Unfiltered seeds can therefore
widen later adaptive grids. The new opt-in restriction must bind its canonical
policy and authoritative unit structures, separate menu-cache entries, and
refuse incompatible family or out-of-band seed candidates before linking them.
It should extend `expand_tessera_menu(families=...)` through
`expand_menus_for_targets`, retaining all existing shape, reader and route gates.

The frozen pricing checkout was untouched. Both derivations ran from an
isolated checkout based on `21ae7d28b7d313654beda6b4e622d732d4276a34`, using
the existing producer source `07ad344c3275`. The physical follow-up recomputed
and matched encoder SHA256
`42783e5214ba1a56d020925155641f9348139ce3f1980ec6df016f71583ee4c3`
and PrismaQuant package SHA256
`5a507741716b7a7a7f50dd8f650852e2f5b910d245e9325244b9ccf736c64a53`.
The complete capture remained
`f4bcbf408d3aa81b04c1fabd1d2d7457176a95dcccdd5ed368800de5c37e277c`.

## CPU execution evidence

Both actions used PrismaBuild, x86 because the scoped interpreter is installed
there, `/home/rob/venvs/pq-cpu312/bin/python`, one CPU, no GPU, and native thread
limits of one. The first reserved 6 GiB and the physical follow-up 4 GiB.
The commands run `derive.py` and `derive_eligible.py` respectively, with
`PYTHONPATH=.:tests:<full-anchor-preparation-03/producer-source-07ad344c3/src>`
and `TESSERA_REPO` pointing to that existing producer tree.

| Action | Outcome | Output SHA256 |
| --- | --- | --- |
| `69481049912c4dff79ed15bd09e9c44dbb31d9c46ecff9f8ef678c6f8b975ceb` | exit 0; menu, seven planner checks, two journal records | `bc2edd132b4646063f10bb5f507a260a227081ec2443cac4da6618fbbd5cfed6` |
| `8841ade4799dae41d8b9ed5ef66e66778b4480bb2f4e2a852f41571988180163` | exit 0; structural counts, mixed-family byte witnesses, frozen source seals | `fd6267169f0b3f789336f2248bc146616f01814e7b9da2fa41f85b037a58d3d2` |

Actual stdout hashes, sealed CAS source bundles, helper bytes, output hashes,
receipt and result payload were independently checked. Read-only CAS
verification reports `checks_passed=true`; it does not independently verify
the worker's attestation. The adjacent PB action/claim JSON and source audit
retain the receipt pointers. The first physical follow-up attempt
`5c6f3d540dab57f57a9e3f68452ac8397b9a5f18877baae12abec9835d78dbd3`
failed before analysis because the helper imported a source-hash function from
the caller module instead of its owning module. Correcting the import produced
the second successful action; no failing attempt is counted as evidence.

## Implemented restriction and full-roster check

Implementation commit `f9432892b9` adds the opt-in restriction through the
existing campaign and fanout mechanisms described above. The pre-fix PB action
`a11dd5c1c650099564b6714200ed1132cf1cb22e78e9d469e5a1481a29347676`
reproduced two failures on the absent restriction and seed gate; its sealed
campaign source was independently compared with frozen `21ae7d28`.

The core CPU suite `10baa5e75d279dfb81b5d18d6071698a723d44c5e41dbcb9403221f03a8b4a4c`
passed 134 tests and skipped the existing CUDA-only encoder CLI test at
`tests/test_tessera_campaign_batch.py:65`. The broader suite
`cfde31ccffb7f33737b416da8b63a65c336276de807777bbd2019b3ddd9e51e8`
passed 143 tests and skipped five CUDA encode tests at
`tests/test_tessera_campaign.py:194`, `:222`, `:248`, `:742`, and `:1350`.
The suites overlap on the focused restriction tests; their counts are not
added into a unique-test claim. Both reserved eight CPUs/16 GiB with native
threads bounded to one. The broader source snapshot matches all four final
implementation files exactly. Adjacent `family-cpu-source-audit.json` records
stdout, source bundles and verified result receipts; no skip is a GPU pass.

The actual new restricted menu was then expanded over the complete accepted
census by `check_restricted_plan.py`, using the existing producer projection
binder and profile-aware structure validator. Every unit agreed with the
independent EXL3 common-surface roster. The result was exactly **135 dense +
36,288 routed units**, **132 unchanged groups**, **73,251 initial anchors**
and **six distinct menu-cache entries**. The all-E4M3 R1036 intrinsic witness
still costs 155,546,148,864 bytes under the exact 155,668,854,528-byte budget.
The report carries the reviewed band 832–1088 and batch width eight arguments.

That check ran as CPU PB action
`88376e4e88e4283ef56c8619d85653139d722dd10ca102c17c10e7028651e15e`,
exit 0, with no weights, activation payloads, Hessian payloads or GPU access.
`restricted-plan-check.json` SHA256 is
`27b5f1f9c1c2ea7e9826bbcef983deafe17a4f49780a62df92000659d7c6c047`.
Its actual source bundle, helper/source bytes, stdout, output, CAS receipt and
result payload were independently checked; the adjacent source audit records
their bindings. A new pricing source seal must follow integration with the
separately reviewed batch observer. This check did not submit GPU rows.
