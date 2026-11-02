# Graph-impact replay

Shared core: `75aff74ae1351a279eff79dd45947623d956faa4`.
The old rules use core `72af52b554ae61a6c5947759cf19a999de528883`.

| Repository | Old failures | Reach-mode failures |
|---|---:|---:|
| PrismaQuant | 16/30 | 8/30 |
| Tessera | 16/30 | 3/30 |

Each policy uses the same frozen base/head graph for each merged PR.
Reach mode uses that range's base package maxima and test-edge count, with
at most five added test edges. Changed test bodies use content hashes.
Gain recording excludes baseline-maintenance noise. The prior strict revised
rules failed 13/30 in each repo; reach mode permits inherited sibling reach.

Reach-mode failures:

- PrismaQuant: #2504, #2610, #2689, #2696, #3199, #3198, #3192, #3130.
- Tessera: #1393, #1558, #1573.

PrismaQuant uses generator evidence and the hosted gate population; revisions
before that selector use all test files. Tessera uses its native #1590 evidence
at `23909243b94bb308dc19c80c9f91567412aaa855`, including its source resolver
and conftest closure. Historical application code is parsed, never executed.
TL3's replay belongs to the coordinator and was not duplicated.

Full per-PR reasons, added edges, base/head SHAs, cached graphs, drivers and
hash manifest are retained outside the repository under
`/home/rob/tmp/bazel-wave2/pq/notes/graphimpact/replay/`:
`reach-report.md`, `*-reach-results.json`, `*-reach-failure.json`,
`*-ranges.json`, `*-snapshots/`, `replay_reach.py`, `report_reach.py`,
and `reach-manifest.json`.
The previous strict proof tarball is retained in the parent notes directory,
`graph-impact-replay-fb655551.tar.gz`; no binary proof bundle lives in docs.
