# Portable campaign image identity — 2026-09-08

The qualified producer image has different local IDs on the two Sparks after
Docker save/load. Sparky reports OCI index
`sha256:cf3f7f83e6820fa75aae249393e8fa4840af4562192203a1aed3f2082f3ea2f9`;
Sparklina reports
`sha256:9f9b9f05b17531399ba66dc6415b054cf5d68c82270626d0e9150e75c808435f`.
Their complete ordered RootFS layers and runtime Config match. A local ID
alone therefore cannot name both installations for portable PB placement.

The existing container identity helper now derives an optional content seal
from OS, architecture, variant, RootFS and complete runtime Config. The campaign
adapter checks it before launch and executes the resolved immutable ID. A
changed tag, layer, platform or runtime setting cannot silently substitute for
the reviewed content. External source/data mounts retain their own identities.
The common alias is `prismaquant-glm-producer:content-qualified-20260908` and
its content SHA256 is
`eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`.
No runtime serving cell, pin, menu or scheduler policy changes (issue #371).

## Validation

Fifteen regression cases failed before implementation (`079006c010b7`): no
shared content identity existed and the adapter refused the new declaration.
After source `d6652ed91`, 40 CPU cases passed in 7.42 seconds on DL380
(`815d8742956c`), including the existing container/runtime identity and
architecture tests. Four workers used four reserved CPUs, 8 GiB total memory
and one native thread each. No cases skipped.

A portable `gb10` PB action selected Sparklina and launched the common alias
with the declared seal. Its recorded observed seal matched before execution;
22 adapter/content cases passed inside the actual producer image in 8.87
seconds (`1df8bafab834`, no skips). The action reserved two CPUs, 12 GiB total
physical memory and an 8 GiB GPU subset. It proves runtime preflight and
execution, not GPU numerical fit or speed. The prior second-worker capture
qualification covered native CUDA/tiny-GLM parity separately (`24e642c1bb11`,
34 passed; see the capture allocator report).

Evidence root: `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/`.
`producer-second-box-image-inspection.json` records both complete inspections;
`producer-portable-image-content-01.json` binds their common seal.
`producer-portable-image-native-invocation-01.json` records the exact launch.
`portable-image-cpu-root-audit-01.json` and
`portable-image-native-root-audit-01.json` verify canonical CAS receipts,
payloads, zero exit status, cleanup, and full checkout snapshots against the
implementation commit; only PB closure files differ. Read-only pytest cache
warnings were retained. No repeated benchmark or performance claim is involved.
