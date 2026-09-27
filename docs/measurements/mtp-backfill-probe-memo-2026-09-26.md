# GLM MTP metadata backfill: shared probe validation

Date: 2026-09-26. Scope: offline CPU metadata join for PrismaQuant #1413;
neither run remeasured costs, allocated rungs, encoded wires, or used a GPU.
Both runs used PrismaBuild's x86 `dl380g10` worker, 2 reserved CPUs, 16 GiB
memory, priority -10, and the same pinned caller environment at
`/home/rob/venvs/pq-pb461728e4-tessera-af7a86d4` (Python 3.14.4). This
environment name is historical; these runs used PrismaQuant source snapshots
and did not exercise Tessera's newer Hessian collection API.

| Run | PB action key | Worker elapsed | Peak RSS | New config SHA-256 |
| --- | --- | ---: | ---: | --- |
| Before, row-by-row source-model validation | `4d4916ac7fa7cffa21744fc4b4c932dc1f67e6ca9c0074ca9c51e9ab4b249ee6` | 571 s | 1.3 GiB | `6809762acd6d5a0e0c211f00f49fb843190a355e4500c54bcb347f75d96d47aa` |
| After, existing `prepare_joint_aura_identities` / `release_joint_aura_identities` | `3963ca40f0660bbccaed8f9d883309f0b4473de8465ed09a12dfd06fc7c642dc` | 24 s (including py-spy) | 1.3 GiB | `6809762acd6d5a0e0c211f00f49fb843190a355e4500c54bcb347f75d96d47aa` |

The **same complete backfill workload** wrote separate new outputs from the
unchanged 237 MiB Step1 config and M6/M4/M3 source files. The measured worker
elapsed reduction is 547 s (23.8x); the after run includes profiler overhead.
The exact output digest and 864 selected receipt count agree. A separate PB
action (`e9abff0d6562b4bdfd904d40f8a0e7e1e7313e09ff5f7aea13f5e35ee9093b29`)
checked every non-metadata config entry and every pre-existing metadata field
against the historical Step1 file: all were equal. The original files were
never overwritten.

After this comparison, the backfill dropped a second full read/hash of the
merged M6 file: its published digest already comes from the exact bytes loaded
at the start. This small follow-up was covered by targeted tests; no further
full-file timing claim is made for it.

The before py-spy dump of the admitted child showed
`json.encoder.iterencode → digests.canonical_json →
validate_streamed_model_identity → validate_joint_aura_entry → _mtp_probe`.
That is attribution at the observed instant, not a time percentage. PB action
`638f103addc8024a79aa3da5fe0ceec49bcc63b7f20f869fddd0efcb5b4b39a8`
counted 3,468 rows referring to just **two shared probe objects** in the real
M6 pickle. The repository's immutable validated-probe wrapper authenticates
each distinct source identity once, retains per-row/operator checks, and is
released to ordinary dicts on success or refusal. The after
speedscope profile at
`/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/ws-mtp-20260925/m6/step1/mtp-backfill-after-speedscope.json`
contains 1,035 samples and 90 sampling errors; its largest leaf was the
M3/M4 receipt join (~11.16 weighted seconds), with JSON parse/encode behind it.

Netdata `system.cpu` over the before observation on dl380 reported ~2.7–3.2%
host-wide user and ~0.05% I/O wait; Sparky was ~1.5–2.6% user with ~0.4%
I/O wait. Around the after run dl380 reported ~1.4–2.6% user with ~0.05%
I/O wait; Sparky reported ~1.3–3.5% user with ~0.3% I/O wait. PB showed the
before child consuming one full CPU core at ~1.1 GiB resident during the
slow phase. This was repeated serialization inside the process, not disk or
host saturation. No GPU throughput or work-per-joule claim is made for this
offline metadata step.
