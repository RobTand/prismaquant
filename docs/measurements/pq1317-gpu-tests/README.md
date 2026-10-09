# Exact-candidate GPU qualification for Tessera #610 and #611

Status: incomplete. No GPU action ran for candidate `fca4c6ce0`.

## Candidate

Tessera `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`, contract v60.
Image X names `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0`.
Target device is sm_121. Facts live in `candidate.json`.

## Corrections

Tessera #610 closed by PR #642. Tessera #611 closed by PR #643.
Both changed tests only. Both merges are ancestors of the candidate.
Their GPU receipts predate the candidate. Facts live in `corrections.json`.

## Roster

Five test files cover both issues. The candidate adds fused dense
launches and new parametrization. Old node counts (71 and 59) do not
match the candidate estimate (136). The owner roster is missing.
Facts live in `roster.json`.

## Verdict

Closure does not qualify the candidate. Old receipts do not qualify
the candidate. CPU skips do not qualify any GPU node. Every required
node needs a passing sm_121 result on image X with native extensions.
Exact commands live in `gpu-commands.md`. GPU work waits for CEO
decision `dec-1009-211441-d65c` and the owner roster.
