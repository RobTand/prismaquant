# EXL3 repeated-window diagnostic — 2026-09-09

Issue: RobTand/prismaquant#468. The first sealed window scored mean KL
0.04757906332418833 in qualification09 and 0.028483384216691037 in the
subsequent full-panel run. Candidate, teacher, panel and scorer bytes match;
the observed runtime binding differs only in allocated KV block counts.
The cause is unresolved. Neither the whole-panel mean nor close means from
two earlier boots establish repeatability.

The existing experimental vLLM scorer gains an explicit
`--diagnostic-repeat-first-window 4 --qualify-hook` mode. It authenticates the
unchanged complete panel and teacher, then scores final-0000 four times with
monotonic capture request indices in one loaded engine. The resident teacher,
FP64 full-vocabulary KL, native prompt alignment and TP ownership checks are
the existing mechanisms. The checkpoint, serving recipe and allocation remain
unchanged. This is a correctness experiment, with no performance claim.

A read-only capture subclass hashes each rank-zero raw logits call, retains
its first sixteen rows at 512 deterministic vocabulary columns, and releases
the bounded CPU staging buffer after the call. It observes outputs without
returning a replacement tensor. Raw byte differences can include softmax-
invariant offsets; interpret them alongside per-position KL and raw samples.
Worker observations retain kernel configuration, selected non-secret EXL3,
GLM53, FlashInfer, CUDA/NCCL/Torch environment controls and Torch numerical
settings. A progress artifact survives after each completed repeat.

The result uses `prismaquant.glm_tr3_repeatability_diagnostic/1` and
`completed`, never the qualification schema or its `passed` field. It cannot
qualify a normal whole-panel replay. No repeated final-panel values feed the
allocator. Uniformity within one engine narrows the investigation to differences
between starts; variation within one engine motivates native route/state
diagnosis. No numerical tolerance is asserted in advance as acceptable noise.

Validation covers unchanged logits, repeat ordering, independently computed
raw SHA256, a softmax-invariant raw-byte change, and bounded mode refusal.
The full existing capture contract suite is also run. Native output and exact
launch seals are recorded separately under the canonical campaign evidence
root. vLLM execution is exempt from PrismaBuild by Rob's standing instruction;
the CPU contract suite is submitted through PrismaBuild.

## First native result

Source `812fcfa412dde96a525a25f4526ab0d199b53d8d`, four requests in one engine,
TP2 across both Sparks. The 84 CPU capture checks passed without skips via PB
`5c44682ec9f68e59613b447c9f8958e9d20a467986b8ae9d8836a4856cb912b4`;
CAS payload and executed source were checked (only the closure file and this
subsequently added document differ from the final source).

Native result `tr3-exl3-repeatability-01/repeatability.json` under the canonical
campaign root has SHA256
`40704ded216ab089808422bb4b582ae721d4dedb65fb540a4b1caa6e69da7af6`,
2,137,484 bytes. The head exited 0 at 09:15:17 UTC. Per-request mean KL:
0.04676430184428184, 0.026519084908803215, 0.047837852412688134,
0.026992125752628725. Every raw-logit chunk hash differs across all four
requests. Native target-logprob alignment error stays below 2.86e-6.
The first scored position differs too.

A restart or change in KV capacity is therefore not necessary to trigger the
variation. These four requests do not establish independent random draws, a
statistical distribution, or a particular state-machine cause. The whole-panel
EXL3 score is unusable for comparison until the request dependence is resolved.

Read-only container observations record identical image layers/repository digest
on both hosts and identical loaded exllamav3, sampling and sparse-MLA extension
hashes. The image's `flashinfer_autotune` function returns before invoking its
tuner, and both previous runs logged the same skip. The configuration bit alone
does not establish that autotuning occurred.

## Localization follow-up

`--diagnostic-layers` extends the same four-repeat experiment with read-only
input/output hooks on decoder, attention and MLP modules. It hashes all raw
bytes of the first token at each boundary and retains at most 128 scalar samples
per tensor. Both TP ranks report their own boundaries. Attention observations
also retain bounded request metadata, including initial-state flags, state
indices and slot mappings when exposed by that module's native metadata.

This is intended to locate the earliest divergence, not to estimate a noise
distribution through more repetitions. The observations copy only one token
per tensor and retain no GPU references. The copies introduce synchronization;
if the variation disappears under observation, that is evidence of sensitivity
to instrumentation, not proof of correctness or a fix. No model output is
replaced. The result remains diagnostic and cannot qualify a full-panel replay.
