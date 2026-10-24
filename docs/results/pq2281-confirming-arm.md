# #2776: retained failed A8S L41–L44 R832 confirming arm (parent #2281)

## Conclusion and scope

**Failed hypothesis confirmation.** The October 5, 2026 arm-minus-A8S paired KL delta is positive but below the preregistered **delta >= 0.0015** criterion for both teachers. Teacher2/all25 remains the primary teacher/population; criterion_met=false. Successful execution is not hypothesis confirmation. This result neither establishes a production-quality gate nor definitively explains the original regression.

This standalone #2776 CPU-only note adopts the existing publication at `4accbe7cbde18c15e0d6b26c08e87e07c32d3f8a:docs/results/pq2281-confirming-arm.md` on `tl/prismaquant-ca44e1548449-wp1`, preserving its evidence catalog rather than starting a second report. #2533 is stopped, not this note's live owner; parent #2281 remains unresolved. No authoritative current child-ledger mapping or parent acceptance was recovered; a branch name proves neither. No scheduling or bookkeeping transition was performed. The supplied open-issue list contains no matching replacement: #2972 concerns #2535's separate numeric evidence, and source qualification #2531/#2282 is not this result note.

This records the existing `campaign-t8-late-cut-confirmation`, not a new campaign. Rob's binding correction at **2026-10-09T21:16:20Z**, with decisions `dec-1009-165401-0740` and `dec-1009-211419-a139`, supersedes the earlier dispatch request: the October 5 arm already ran and failed; the later no-run/publication authority cancels the proposed 240 GPU minutes. The rounded **+0.00083 delta / 0.00016 SE** is binding CEO-attributed evidence, separately corroborated below by CPU reconstruction of recovered bytes. The primary decision transcripts remain unverified. No GPU submission, new measurement, weight rewrite, encoder work, teacher/instrument substitution, source mutation, or serving/performance qualification was performed for this publication. There is no W1/W2 dependency.

Against the original positive regression **+0.001982807772259931**, the primary arm-minus-A8S delta gives a descriptive ratio of **41.978664157%, approximately 42%**. A positive KL delta is worse than A8S, **not an improvement over A8S**. This ratio does not establish additivity, identify all remaining responsible layers, or qualify shipping prices. No uncertainty is assigned to subtraction from the original regression: that would require paired covariance, not independent-error propagation.

## Paired measurement

| Teacher | Population | n windows | Candidate KL | A8S KL | Signed paired delta | Across-window paired SE | >= 0.0015 |
|---|---|---:|---:|---:|---:|---:|---|
| teacher2 (primary) | all25 (primary) | 25 | 0.028342628003488222 | 0.02751027178789452 | +0.0008323562155937031 | 0.0001602368391371253 | false |
| teacher04 | all25 | 25 | 0.02828275958062752 | 0.027426096704948534 | +0.0008566628756789866 | 0.0001537487041147601 | false |

Independently reconstructed on CPU from the four recovered per-position arrays (row means over 2,047 positions, then paired across 25 windows): teacher2/all25 delta **+0.000832356215593703**, SE **0.00016023683913712466**; supplementary teacher04 delta **+0.0008566628756789864**, SE **0.00015374870411476006**. The table and JSON retain the adopted summary's numerically equivalent last-bit values.

Recomputed from matched retained per-window samples: d_i = candidate_i − A8S_i; delta = mean(d_i), SE = sample SD(d_i)/sqrt(25), with sample SD using denominator 24. This is descriptive across-window uncertainty, not repeat-run variability or uncertainty from 51,175 independent positions. Last-bit decimal differences from floating-point summation do not affect the criterion. No clean17 estimate is substituted for all25; its frozen subset remains recorded in the paired artifact.

Candidate and control have identical ordered 25 unique window IDs, identical per-window teacher-logit data hashes for each of both teachers, identical source initialization contract and source model identity, and equal producer_identity objects. Context length is 2048, with 2,047 unmasked causal positions 0..2046 per window (51,175 total). The metric is KL(teacher||candidate), full 154,880 vocabulary, FP64 normalization and reduction from raw FP32 logits. This does not attest teacher2's generating arithmetic dtype merely from a BF16 model name. The paired result preserves the reductions; the two full result files preserve ordered windows, per-window teacher-logit hashes and producer/source identity objects, not merely aggregate scores. Equal candidate/control producer objects and matching logit hashes do not independently authenticate teacher2's producer: the explicit equality flags in those results refer to teacher04. The retained teacher manifest hashes and panel identity bind the historical contract, but these portable fixtures do not contain the full teacher manifests or token arrays. The historical A8S scorer bytes are not independently authenticated here; “unchanged offline G3” means numerical-method compatibility, not universal historical byte equality.

## Selection and frozen input audit

The exact handoff, CPU qualification, complete units/tensors and change table describe **36,423 units, 38,760 tensors, 3,456 changed units, 32,967 unchanged units**. Complete-record inspection finds exactly routed L41–L44, experts 0–287, down_proj/gate_proj/up_proj: 288 changes per layer/projection. All change from `TESSERA_E4M3_K1_R1024` to `TESSERA_E4M3_K1_R832`; shapes and `fp8_per_token_dynamic` contracts remain unchanged. Changed rows equal the complete change table, and unchanged rows preserve control metadata. Selected tensor roots count 35,304 a8 plus 3,456 t8r. No early upgrade or other routed cut is included.

Existing a8 members are from `/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported`; existing t8r members are from `/mnt/shared/tessera-measurements/t8-v1-allocation-20261005T022641Z/release-all-priced-T8-BF16/exported-progress`. These are historical identities, not requests to regenerate or rewrite weights. This is an immutable manifest overlay, not a newly materialized/shipped checkpoint. Summed priced-member-byte reduction is 2,717,951,616 bytes of metadata accounting, not measured runtime memory or checkpoint savings. Original file-content/audited tensor identities and immutable selected identities are listed below separately from raw record hashes.

The actual frozen runner is under `source-ready/` in the retained measurement root, not today's repository G3/v2. Its scorer `g3_offline_decoded_kl.py` has SHA-256 `f52bc336ae4c040fde4dffece06c4b72ff5871813de72e07aa5589a642f84455`. The older freeze snapshot `dbe0320a7e57812a30b0c7241afb1d60386f3ab9` records scorer `dea951c9562cf86b92e63fb8bd0c4cea4737d134438b88a6e7e8a90e480e152f`: **these are not byte-equal**. Inspected changes replace the identity-cache consumer with prepared-source binding and proof arguments, with launcher preparation/preflight plumbing; no decoder or numerical-arithmetic edits appear in that scorer diff. Paired compatibility is supported by the actual input/scoring records, not by asserting equality to that older freeze. Source-ready hashes are enumerated below. Additional launch bindings are `g3_launch.py` SHA-256 `3710ec8f85768371cebe863cd8ff729ab7f771a823665e6f9c88c82fec346e26` and `latecut_launch.py` SHA-256 `3facdf08907062b14e73d8c78dcf402c8e6381cc77b9461a6a4861c3e7b3b9af`.

Source initialization contract SHA-256 is `7763ca90592e2ac011c39704bf1f0676e2a04f9e7aed625ad29bc0716d6ebdca`. The equal producer objects in both results retain experiment file identities and gold-source commit `7882eda3a87fd6a45a1142de140aa17c49a8021e`, gold source SHA-256 `5a507741716b7a7a7f50dd8f650852e2f5b910d245e9325244b9ccf736c64a53`, and gold tools source-files SHA-256 `d1e7cc5358662b81441ba2f10f81c252f93333eb9eedab899d241bdfa3df3a02`.

## Preparation, failures and preflight

Three retained CPU stages succeeded: source preparation, complete selection, and actual consumer/parser qualification; action and receipt identities follow below. Qualification was CPU-only x86 with CUDA invisible, zero skips/uncollected modules, 1,594 fixed source tensors and 140 authenticated original files. Preparation consumed upstream-bound stat-validated caches: 120 cached shards, zero newly hashed shards, zero source-body bytes read, zero legacy fingerprint rewrites. Do not confuse cache authentication with a fresh payload rehash.

`ready-handoff.json` (measurement root), SHA-256 `a42113ec8a0638975d4ce5dd671dfcf8d8b4c0943b681dceebfdaab41286e0c0`, records CPU successes and historical readiness, explicitly saying no quality result yet at that stage. Published checkout is `e06d7e495f2626626649c9c3b34dbbe03d8f12d2`, qualified predecessor `5dcf4d328a5117bb3c405ed9ec6d74057c2ad3ed`.

Preserve both failed CPU actions, listed below: the first had action return code 2 (5.479233 s), the second 1 (2.511420 s), each one attempt and failed terminal status. Their CAS receipts were absent. The preparation history records pre-extension rejection of `--a8-export`; the corrected wrapper uses only selector `--arms`. Separately, `--progress-cycle` was rejected without publication because no eligible worker offered `progress-cycle-v1`; the published action uses startup/run phases. These are not extra successful scoring attempts.

The intermediate `metadata/handoff-proof-ready.json` SHA-256 `cdaf53a76d7a2e7c9ffd1de61a715c45b4b690d2673458f118a15566fc928f63` rounded nanosecond integers through JavaScript. The original `metadata/handoff-proof.json` SHA-256 `5e2e8d95d774f86b21e2564b59b633b4b34b749a7fd212742748b3592021bf87` was used as raw text to repair precision. The exact handoff's files structure equals the original's; the rounded intermediate's does not (1,038 differing integer leaves). This was not adoption of new physical fingerprints. Use **handoff-proof-ready-exact.json**, bound below. Candidate-inode, manifest-byte and source-proof-byte negative drift refusals were retained.

The actual `invocation-preflight.json` binds manifest, scorer, preparation proof and NVIDIA GB10, Torch 2.13.0+cu130 / CUDA 13.0; its plan is 36,312 decoded, 111 source-kept, zero carried. Twelve representative decode checks include the three L41 expert-0 R832 roles, not exhaustive checks of all changed GPU units. CPU preparation is distinct from the launch-time metadata/authentication preflight; a separately measured GPU-preflight interval was **not found**, so no sub-ten-minute claim is made. The record's representative decode checks must not be relabeled metadata-only work merely to satisfy the old request.

## Authorization and execution chronology

The retained action was published **2026-10-05 13:18:02.323229 UTC** (queue `published_unix=1791206282.3232286`), before the October 9 decision `dec-1009-165401-0740`. Both October 9 decisions postdate the run; they cannot be prior authorization for October 5. They are later recognition/publication authority, not evidence of the actual earlier authorization timestamp, which remains unverified. No timestamp is inferred from decision-ID digits. The exact handoff declares: “CEO explicit release received by Main; one priority-0 class-gb10 any-Spark arm is authorized only after exact PB x86 authentication passes. No encode or extra arm; do not withdraw for Window4.” Its parent gate is NEXT D30 terminal plus independent physical cleanup/release, or explicit CEO release. This is a recorded declaration, **not a separately authenticated primary authorization transcript**. October 9's later no-run decision authorizes this publication, not retrospective fabrication of Window4 or dated kernels evidence for October 5. Primary Window4/kernels/campaign records outside the promised readable mounts were not independently inspected; the original October 9 submit-condition claim cannot be verified here.

Window4/republish conditions are audited only as historical compliance: no new submission or republish is ordered, and the retained declaration alone cannot establish compliance with the earlier condition.

Inherited workflow history is separate from experiment attempts: supplied event **seq 1010055**, workflow attempt **3**, records **“activity interrupted without proof of worker loss”**. It is not an additional scoring attempt, and this note does not reset inherited history or independent-review counters. The later seeded workflow event does not establish a scientific review count.

The request declares its checkout snapshot input (SHA-256 `62853508af289863b5dbe1aa51ac70f42f123d8a9f66fe35a8919f0f9a014870`) and no child dependency list. The handoff is bound by path/hash in command arguments. There is no submitted dependency on k1 or another child; CPU qualification and parent release are historical prerequisites, not scheduler-enforced child edges. Resource admission and authorization holds are distinct from those dependencies, and child processes are not child actions.

The historical request used 12 CPUs, 96 GiB total memory, one exclusive GB10 GPU and an 88 GiB GPU-memory budget. Admission at `1791206296.931573` (13:18:16.931573 UTC), attempt claim at `1791206297.1067252`, terminal finish at `1791210521.5754535` (14:28:41.575454 UTC). It executed on **sparky**, return code **0**, exactly **one attempt**, `max_attempts=1`, `retry_safe=false`, elapsed **4223.7810299396515 s**. Resource-profile wall time is 4223.847310 s; candidate scorer elapsed is 4195.220685 s, different boundaries. Exclusive admission is not an idle-host guarantee: baseline says unmeasured / holders_present.

Request, complete receipt, terminal summary and attempt citations follow. Manifest, producer attestation and receipt digests are canonical identities, not raw hashes of their outer JSON files. The CAS result digest binds **114,469 bytes of captured text**, not directly the scored JSON. The additional candidate/control/paired raw-file digests below bind the numerical result files separately; do not conflate these chains.

Named attempt stdout/stderr sidecars were absent on inspection; only the attempt outcome JSON was present. Terminal embedded stdout is truncated, but the CAS textual result is readable. The named stdout was 117,775 bytes / `489a0dce8f4de0065a8c73d4ff3e7bcce88bdb12fb478ef650919f0a448a80d7`, stderr 43 bytes / `6ae22e28d8e7d3c9541ecc30e0edf8b3931838f003d6d63a9b15a9f79f0f4b0f`; neither is the CAS-text blob. A separately established independent result-review count was not found. Progress acceptance/replay counters are not reviews; unknown is not zero.

## Observer alignment and limits

Both in-process traces and retained telemetry are digest-bound below. First-layer trace has 382,023 events over about 10.804 s starting 13:18:42.124744 UTC; L5-install trace has 197,030 events over about 10.809 s starting 13:22:09.406235 UTC. CPU operations, CUDA runtime, kernels and copies are present. These are selected first-layer/L5-install observations, **not complete L41–L44 or whole-run kernel coverage**. Profiler no-warmup warning limits interpretation.

Telemetry wrapper spans `1791206310.9772573`–`1791210517.6282334`. Netdata has 835 collection rounds, 1,670 host rows: sparky `1791206311.033952`–`1791210516.3671453`, sparklina `1791206311.0531657`–`1791210516.3838038`. Python sampler has 4,196 samples at `1791206310.997452`–`1791210517.4056053`. These align with the scorer and encompass the two trace windows, but start roughly 13.87 s after claim and stop 3.95 s before terminal completion. Trace alignment uses baseTimeNanoseconds `1790857026000000000` plus event ts in microseconds, not the displayTimeUnit label.

Netdata retains power/GPU, CPU/frequency, memory/PSI/swap, I/O/NFS and PCIe series. Configured periods are 5 s Netdata and 1 s Python; observed maximum gaps are about 5.47 s and 1.843 s. Chart last_updated must be considered alongside row time. Sparklina is contextual telemetry, not a second execution host. Mount-latency/probe charts were required-charts-relaxed. Python is main-thread-only: 3,550 samples end in threading.wait, which does not imply idle GPU/workers. Sampling supports coarse correlation, not a definitive causal explanation.

Telemetry result reports complete with no errors, but **collections=[] and profile_layers=[]**. Configured forward_windows_zero_based `[[0,1],[31,32]]` is not evidence of collecting those windows in a 25-window panel. The nonempty retained traces do not fill those coverage gaps. Whole-box resource statistics and declared fallback power references are not action-exclusive performance measurements. No serving, production-quality, or performance claim follows.

## Evidence catalog

Paths below are historical citations, not runtime dependencies for checks. SHA-256 values marked raw are byte-file hashes. The fixture README documents byte-for-byte copies of the three result files; its evidence-index is a **research projection, not a new producer receipt**. The report retains original external paths rather than copying bulky artifacts into the repository.

### Scored results (raw)

| Record | Historical path | SHA-256 |
|---|---|---|
| paired-result.json | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/paired-result.json` | `1133209b27b114c739fff1d803280d6a33dae1cf949daa7cca0f693eaf11d675` |
| candidate-result.json | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/gpu-score/latecut41_44_wa-latecut41-44/result.json` | `168707a152c5f71202da55c8c13823a08a24b811829ca5576d6def3647595d13` |
| a8s-result.json | `/mnt/shared/tessera-measurements/surrogate-diag-20260929/g3/runs/a8s_wa-gate2/result.json` | `7b18d2a579df67324c5252c3d1078b7eb3a82de7920217c47bab44c354dd6df5` |

### Recovered per-position arrays (raw)

Portable byte-for-byte copies and provenance are under `tests/fixture-data/pq2281-confirming-arm/`, adopted from prior fixture work `a7d20474c0bf21a543514294474a1ccca9bdcc9b` plus the recovered arrays. Checks need neither fleet access nor historical Git objects. All arrays are NPY v1 little-endian float64, C-order, shape (25, 2047), 409,528 bytes; rows follow each result's ordered `windows` list.

| Fixture | SHA-256 |
|---|---|
| candidate-teacher2.npy | `43c0dfa831c538d8efa574ab961fe34d9f60e677dbb3a912a7d98346b5b70f69` |
| candidate-teacher04.npy | `25083251ab141744dbee33961bd7a3b19da5c851a42cfebd69de11d643b9ea4e` |
| a8s-teacher2.npy | `5e13b59a9ce27df01357c137ace36c725dd1f72dccaab65ab89e3cae94507a62` |
| a8s-teacher04.npy | `5debf6cff9bf04f0bd2dc06f81c840b087eced3f82ab6f6ce3f4616937235973` |

Candidate historical originals: `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/gpu-score/latecut41_44_wa-latecut41-44/per_position_kl.{teacher2,teacher04}.npy`.
Control historical originals: `/mnt/shared/tessera-measurements/surrogate-diag-20260929/g3/runs/a8s_wa-gate2/per_position_kl.{teacher2,teacher04}.npy`.
Recovery and CPU arithmetic corroborate the attributed numbers, not an uninterrupted custody chain. The producer's CAS text digest does not directly bind these arrays or result JSONs.

### Additional source bindings from the full results

| Identity | SHA-256 |
|---|---|
| A8S control manifest | `504df10d2bffa1ab99dd88e3d6fae606a4c45034587c29c22c51dcc70f507bcf` |
| root-bf16-upstream-source-binding.json | `23370e25f6b42e316f72d6cbc8f3f5747a65705b388c910da93545f487edc0fa` |
| original-execution.json source derivative | `38e0b9de817f645c4bec37c0d4a3e58baecccb040f5718dc069a72c7385a0bed` |
| common container content | `eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd` |

### Input records (raw)

| Record | Historical path | SHA-256 |
|---|---|---|
| exact_handoff | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/metadata/handoff-proof-ready-exact.json` | `b314803643a9fcca8c12e8f11d52f82980342611f1c158d70a8f2e3fc9a50d89` |
| source_preparation | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/source-prepared/source-preparation.json` | `c80d72c4386bc187f2deca3c4736576452ed26813ef5319c31d8fd98c3e56c42` |
| cpu_qualification | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/metadata/cpu-qualification.json` | `834fa8ec2b1e0baa2ba592eaaca77f1d9672f5cad8e1a585adede29dd61fdb94` |
| changes | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/metadata/unit-change-table.json` | `b31109cd3858adf6c1d588724e1e6d0b9e690725e54d29d4b715f3a515749077` |
| manifest | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/metadata/unit_manifest_latecut.json` | `a626901c7bfc594fed8a99bf74bed8f46bf393d8913f2f2ac4f500b4eadb819e` |
| tensors | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/metadata/complete-selected-tensors.json` | `fcee6d755b1348eef038c892048e894a4ddb131bf6808d009dfe159c3fca42f9` |
| units | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/metadata/complete-selected-units.json` | `a06b5fe6557263fe0bdda010d1cb451ddf03557f4043418b7d804ed8964c510d` |
| gpu_preflight | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/gpu-score/invocation-preflight.json` | `04e46594f37a08e827a6497d6acc238f9069d6d4fe0cef2a14b3b8f6985e6254` |

### Profiles and telemetry (raw)

| Record | Historical path | SHA-256 |
|---|---|---|
| first_layer | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/gpu-score/multi-latecut41-44/first-layer.trace.json` | `1ce7c71eeebd4bac7aedceda4f2476720e67dfeef8cc464e95bfa7f613ac0a1b` |
| install_L5 | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/gpu-score/multi-latecut41-44/install-L5-latecut41_44_wa.trace.json` | `1597fd3395a15d8807f758b8a104980b70d56409210b19d4a8c480959f266e66` |
| after.json | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/gpu-score/multi-latecut41-44.telemetry/after.json` | `7aa57179d2a7726054f3e8484ab096df305b077a74548a4d102407a1da5880fb` |
| before.json | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/gpu-score/multi-latecut41-44.telemetry/before.json` | `cca4f23afc1e273ff1b618b2ca82dbdb58688cc447baa6efb9e0ec9d842a49de` |
| netdata.jsonl | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/gpu-score/multi-latecut41-44.telemetry/netdata.jsonl` | `5ee63d443d380b5502b7b0018386b201118807527b293d7a751df93d2e173051` |
| python_sampler.jsonl | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/gpu-score/multi-latecut41-44.telemetry/python_sampler.jsonl` | `333925f11f1db3ed2953204dba3d5deacea04ef5f610ca959b5a97211634f892` |
| result.json | `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/gpu-score/multi-latecut41-44.telemetry/result.json` | `388ee9e9702f5bd948536cd619a463053704b3c778ec961d2429c1b6b5f64df9` |

### Logical input identities

| Identity | Digest |
|---|---|
| original_a8s_file_content_sha256 | `fff0b36f013644dc1b3a6cdedece02f0d293baf42327c0947dc67bfc65cb36e8` |
| original_r832_file_content_sha256 | `4e7ce0dddb060bde76e690e17254708bc583880ff2da247d1b425fa4a8135865` |
| original_r832_audited_tensor_identity | `8fa757ba161c556cab3be513e8608dde329661f108a2860c873eb1aaf6a2e925` |
| selected_tensor_identity_sha256 | `d666f372639fa18cff21e7ab93044429cc58931d491f33d6e7fdf4ad0919c24a` |
| selected_unit_identity_sha256 | `c703a00bf817923ccac961ac05806e1b97015cae1b73a909865c168f3c3feb26` |
| source_model_identity_sha256 | `8bdd62ee422d82c895506a075031e3a44fb5c0023e2f00662ccb41410f8d5cde` |
| panel_sha256 | `35f0c5c973be614f29db757e9bd4bce407ea218b974a8407ec7e64c571aad72b` |
| teacher2_manifest_sha256 | `da595505a3e9a2bcced69bde1f156f71b62f43571b1797cede21d12f4b9c423b` |
| teacher04_manifest_sha256 | `1cc798a32a3457f996e859f778fe61fd987561b91490fe2953b698457ea747ae` |
| scorer_sha256 | `f52bc336ae4c040fde4dffece06c4b72ff5871813de72e07aa5589a642f84455` |

### Actual source-ready file bindings

Relative to `/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/source-ready/`. These identify the retained files, not a claim of byte equality to every older freeze.

| File | SHA-256 |
|---|---|
| `exl3_torch.py` | `a8d841710fa21f6998b95a6a929aedc5fe74da09a6c4ad81a7eb1be32f66ab2c` |
| `g3_lib.py` | `6aa379a999e9852ddb2eb29981be6772b704a0e90b993d31599ae390a2353494` |
| `g3_manifest.py` | `be59178ff0a2e735d1d697e2117c33af1d6bb0a5d974bfa560573632ab2044ba` |
| `g3_offline_decoded_kl.py` | `f52bc336ae4c040fde4dffece06c4b72ff5871813de72e07aa5589a642f84455` |
| `g3_prepared_source.py` | `69fe06af9a184175bbb8199d8281a30594d65217caef563860a9860710f3d68c` |
| `g3_progress_launch.py` | `f5f4bb2c4b444f50a1a24df0626d3d74467360c309436b01c8b394d08b472a34` |
| `g3_v1_preflight.py` | `f57163177800c55bb150f3e6046d50732bba7f4012f509460412103ac43515f5` |

### Historical GPU execution chain

- Request: `/mnt/shared/prismabuild-fleet/cas/requests/1c/1cb0d64a4d0a745288b23cae33b351aeb2c25d89192f105e2e80f578801cf56f.json`.
- Complete receipt: `/mnt/shared/prismabuild-fleet/cas/actions/v3/1c/1cb0d64a4d0a745288b23cae33b351aeb2c25d89192f105e2e80f578801cf56f.json`.
- Terminal outcome: `/mnt/shared/prismabuild-fleet/pb-queue/done/1cb0d64a4d0a745288b23cae33b351aeb2c25d89192f105e2e80f578801cf56f.json`.
- Attempt 1: `/mnt/shared/prismabuild-fleet/pb-queue/attempts/1cb0d64a4d0a745288b23cae33b351aeb2c25d89192f105e2e80f578801cf56f/7eb62dd08bfad280f4c3c36802cfe49ed22ba89d901e06291d7371f968f1a4c0/00000001.json`.
- CAS text: `/mnt/shared/prismabuild-fleet/cas/blobs/38/389eb7a289ec7faaa6c0827b84bf88dd9d5e854bd7d065b1c00c498597d4aba2`.
- Action key: `1cb0d64a4d0a745288b23cae33b351aeb2c25d89192f105e2e80f578801cf56f`.
- Canonical action manifest: `0eca1ebe2cfd2b084d57703a038202e15f7b53ae9304f05820a0124072320e24`.
- Canonical complete receipt: `84033f45f1bf8dd11c86b46d10f6179738a5e05ea6c371cf24e115f51715be78`.
- Producer attestation: `c3c92603daa3b022e85e6b34a193cd3b3ac5c6114daf4f740009d0b4ee20d3da`.
- CAS text raw SHA-256: `389eb7a289ec7faaa6c0827b84bf88dd9d5e854bd7d065b1c00c498597d4aba2`.

### Successful CPU actions and complete receipts

- **CPU_preparation**: action `1b3179b70290aac486c686d92fde6ffea62d1f25d40a7b93a83f033b76e8ca57`, return code 0, executed. Receipt `/mnt/shared/prismabuild-fleet/cas/actions/v3/1b/1b3179b70290aac486c686d92fde6ffea62d1f25d40a7b93a83f033b76e8ca57.json`, canonical receipt digest `bb823671f11e3207e97229efb2c74360ccc0c37c331b81b03541ef9350bbfa6a`.
- **CPU_complete_selection**: action `e8d5bd6bd25d18c97fa6e08e9c3d4dddb2de6382ade2faef1dd7b839e1ac5bb2`, return code 0, executed. Receipt `/mnt/shared/prismabuild-fleet/cas/actions/v3/e8/e8d5bd6bd25d18c97fa6e08e9c3d4dddb2de6382ade2faef1dd7b839e1ac5bb2.json`, canonical receipt digest `ae1ef6fb6a96107280936bee7543b63737a2c7b594fd89b46cc21592301d930a`.
- **CPU_actual_consumer_and_parser**: action `7a528b272d9e2bedb32d9ce374a51ac67d7158146353695de47293af112f4dac`, return code 0, executed. Receipt `/mnt/shared/prismabuild-fleet/cas/actions/v3/7a/7a528b272d9e2bedb32d9ce374a51ac67d7158146353695de47293af112f4dac.json`, canonical receipt digest `77555d6b7d8db8d7d24c4167844a7113968b765080114bbf07bdf5bdd5674f1a`.

### Preserved failed CPU actions

- `ea30295354d2c46405163d4bea16689d44c14bebee6818c9062a4730e6eaed6f`: failed, action return code 2, one attempt; referenced CAS receipt `/mnt/shared/prismabuild-fleet/cas/actions/v3/ea/ea30295354d2c46405163d4bea16689d44c14bebee6818c9062a4730e6eaed6f.json` absent. Terminal record: `/mnt/shared/prismabuild-fleet/pb-queue/failed/ea30295354d2c46405163d4bea16689d44c14bebee6818c9062a4730e6eaed6f.json`.
- `19674e6c3f342eda7618053c69120e3e0f46b085446dcb34f289be6a8f811289`: failed, action return code 1, one attempt; referenced CAS receipt `/mnt/shared/prismabuild-fleet/cas/actions/v3/19/19674e6c3f342eda7618053c69120e3e0f46b085446dcb34f289be6a8f811289.json` absent. Terminal record: `/mnt/shared/prismabuild-fleet/pb-queue/failed/19674e6c3f342eda7618053c69120e3e0f46b085446dcb34f289be6a8f811289.json`.

## Machine-readable numerical summary

```json
{
  "schema": "pq2281.confirming_arm.summary.v1",
  "candidate_action": "1cb0d64a4d0a745288b23cae33b351aeb2c25d89192f105e2e80f578801cf56f",
  "primary_teacher": "teacher2",
  "primary_population": "all25",
  "delta_definition": "arm-minus-A8S",
  "pre_registered_delta_threshold": 0.0015,
  "criterion_met": false,
  "results": {
    "teacher2": {
      "all25": {
        "n_windows": 25,
        "candidate_mean": 0.028342628003488222,
        "a8s_mean": 0.02751027178789452,
        "paired_delta_mean": 0.0008323562155937032,
        "paired_delta_se": 0.0001602368391371253
      }
    },
    "teacher04": {
      "all25": {
        "n_windows": 25,
        "candidate_mean": 0.02828275958062752,
        "a8s_mean": 0.027426096704948534,
        "paired_delta_mean": 0.0008566628756789865,
        "paired_delta_se": 0.0001537487041147601
      }
    }
  }
}
```

## Publication verification

Current verification: `python -m pytest tests/test_issue2776_cpu_note.py -q -s --basetemp="$TMPDIR/pytest"` — **4 passed in 1.94 s**, exit code 0; retained stdout read back. The CPU reconstruction produced the values and 41.978664157% ratio reported above. An initial run had 3 passes and one citation-wording failure (singular “layer” did not satisfy the required plural topic); corrected without changing the result or tests.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"01049c698d2e7ab3e5684fa4b64c0278c9b576e3bae22eeab2d015939fe4dc07","size":3072}},"operation":"6c92e983006a5f3a935735b0791926ad20ffd5954b09c8ab7f842b30862721b3","receipt":"373fe4beec565fa69938aefedd221e9866004edeeb1a6ae07899b36082d36f48"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"8f961d784e97ea183317d65db03cc56a9996f543aca52b5676162a8ca572cfb2","size":709}},"operation":"d587a6ad4a84c0d71553bdc212fa4220c7ebb1356817f23c48b87fbd79ffd822","receipt":"88f32c28295623a20fbd81d4124b0a1318d216f149b0bd20fa80f6b7b9c8f8ea"}]

The CPU acceptance check verifies numerical publication and required citation presence against retained fixtures; it does not authenticate every historical claim. O5 requires evidence review of the audit and its explicitly unavailable records.

Historical verification preserved from the adopted report (not rerun here): `python -m pytest tests/test_pq2281_confirming_arm_report.py -q --basetemp="$TMPDIR/pytest"` — **3 passed in 0.59 s**, exit code 0; retained stdout read back. No GPU check was run or needed. The unavailable historical checks are identified above, not counted as passes.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"0a4669cdfa71de5e492f11a0bea2fb342e280c1d5fd396a3eab388fbb30a1336","size":268}},"operation":"cef456e606d021e067b1c33ff53a244f8ed0446b0aeb0e506b02934bfa47a2fa","receipt":"eaba879b57de3d5a1d997bbd2bcab12bcbf28dfa1cab057569c0a86aa80114f1"}]
