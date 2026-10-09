# Exact-candidate GPU qualification for Tessera #610 and #611

Status: complete. GPU runs qualify candidate `fca4c6ce0`.

## Candidate

Tessera `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`, contract v60.
Image X is `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`.
Target device is sm_121 on NVIDIA GB10.
Facts live in `candidate.json`.

## Corrections

Tessera #610 closed by PR #642. Tessera #611 closed by PR #643.
Both changed tests only. Both merges precede the candidate.
Old receipts predate the candidate and prove nothing new.
New runs qualify the candidate. Facts live in `corrections.json`.

## Roster

Collection on image X lists 136 nodes.
MoE suite holds 71 nodes. Dense suite holds 65 nodes.
Facts live in `roster.json`. Outcomes live in `results/`.

## Verdict

136 passed, 0 failed, 0 skipped on sm_121.
MoE action `fe1ae1f2402b865d7dc84a6f86e56745f4767db4b8aee0bddeec73d964d03462` passed 71 on sparky.
Dense action `370fead21f36ad81db81b7f6b536e85c09ccc97baae95f297105510dd2931fac` passed 65 on sparklina.
CPU collection and old receipts give no qualification.
Commands live in `gpu-commands.md`. Logs live in `results/` and on the shared mount.
