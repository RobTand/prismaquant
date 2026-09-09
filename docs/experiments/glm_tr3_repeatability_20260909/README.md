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
