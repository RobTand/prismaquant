# G3 native JSON audit (read-only source)

## Identity and boundary

Research commit: `6d8288adc94b3c4b624c143ed372fb6513f32784`.
Inspected checkout HEAD: `8bc0b5af8c69af352b559dccda99bd4e6fa8b418`.
These commits are **not identical**, but all eight assigned files have identical
research/HEAD/worktree Git blobs. `ledger.json` records each blob, SHA-256,
exact call source, lexical scope and AST coordinates (line 1-based, column
0-based UTF-8 byte offset, exclusive end). Worktree `git diff RESEARCH -- <eight
files>` was empty; direct `git show` byte comparisons also matched all eight.
Existing unrelated worktree changes were not modified.

Read project `AGENTS.md` and `CLAUDE.md`; inspected the shared owner, checkout
policy import bridge, submission wrapper and image-seal test as context.
Only this local audit fragment was written. No tool source/shared owner changes,
new owner, new recipe, retention authority, publication or workflow transition.

The exhaustive claim is **native stdlib JSON call sites in the eight assigned
Python files**, not all serializers reachable from their dependencies.
Existing profile construction/text and delegated record writes are separate
populations. Static sites are not execution counts.

Generated-code boundary: literal inspection found no embedded `json.*` code in
these eight sources. Preflight executes a parser reconstructed from scorer AST;
imported/generated code is not recursively inventoried. The context image-seal
test generates a fake Docker executable, also outside the assigned population.
Shell/deployment boundary: argv serializers are inventoried, but shell wrappers,
PB helpers, pinned runpy adapter and imported internals are not an exhaustive
shell/generated JSON census.

## Counts and options

| Assigned file (`tools/g3job/g3_*.py`) | dumps | dump | loads | load | Total |
|---|---:|---:|---:|---:|---:|
| launch | 5 | 1 | 0 | 0 | 6 |
| offline_decoded_kl | 1 | 0 | 2 | 0 | 3 |
| one_arm_smoke | 1 | 0 | 0 | 0 | 1 |
| progress_launch | 1 | 0 | 2 | 0 | 3 |
| readset | 2 | 0 | 5 | 0 | 7 |
| real_arm_smoke | 5 | 0 | 5 | 0 | 10 |
| residency_smoke | 4 | 1 | 1 | 1 | 7 |
| v1_preflight | 1 | 1 | 1 | 0 | 3 |
| **Total** | **20** | **3** | **16** | **1** | **40** |

All **23 native encoders** effectively use `sort_keys=False`, `ensure_ascii=True`,
`allow_nan=True`, `default=None`, `skipkeys=False`, `check_circular=True`,
`cls=None`. Their whitespace options are:

- 17: `indent=None`, separators `(', ', ': ')` (stdlib defaults).
- 3: `indent=None`, separators `(',', ':')` (explicit compact).
- 1: `indent=1`, separators `(',', ': ')` (effective indented default).
- 2: `indent=2`, separators `(',', ': ')` (effective indented default).

All **17 decoders** have no explicit options: preserve stdlib duplicate-key
last-wins, normal numeric parsing and acceptance of non-finite constants.
They are not encoding-recipe gaps or automatic strict-parser migration targets.

Separately: **2 existing owner sites** (`JsonProfile` construction at scorer
73:10 and `G3_JSON.text` at 173:19), and **9 delegated record-write calls**.
Those already use `g3-result`: sorted keys, ASCII, lax non-finite handling,
`default=str`, indent 1, separators `(',', ': ')`; temp-write plus atomic replace.
They are not unsorted stdlib sites and are not counted again as native encoders.

## Gaps and consumer significance

**23 individually recorded unresolved recipe gaps**: one for every native
`dump`/`dumps` call, each with its own ID, exact location/options and consumer.
`JsonProfile._encoder` (`prismaquant/digests.py:277-287`) always sorts keys;
there is no unsorted recipe. A similarly named spaced/indented owner is not
byte-equivalent. This audit assigns neither a new owner nor scientific retention.

Consumers differ: host-mount/read-plan environment, launch provenance, adapter
and preflight argv, synthetic model index, staged/union manifest gzip bytes,
CLI range specs, semantic-progress JSONL, comparison events and stdout reports.
The preflight result file is immediately SHA-256 hashed. Progress event evidence
retains original JSON lines; event JSONL is flushed/fsynced before commit.
Header/index/teacher/panel decoders determine byte extents and execution inputs.
These distinctions are in each ledger entry, not a blanket scientific exemption.
A sorted-vs-unsorted witness demonstrates a recipe mismatch, **not** that every
actual consumer demands byte equality; consumer-specific qualification is open.

## Native execution/import constraints

- Submission (`g3_submit.py:25-37`) uses PB staged residency, declared resource
  admission and semantic phases, then executes the progress wrapper. The wrapper
  launches the host launcher and uses the configured PB commit helper.
- Launcher has module-scope argparse, output writes and container execution.
  It loads the **pinned** producer adapter, not checkout's adapter. The existing
  `fleet_data` image-seal test explicitly documents why launcher cannot simply
  be imported. The launch spec mounts checkout `/workspace`, numerical `/pq`
  and Tessera source separately; these are observed existing assumptions, not
  newly introduced configuration or proof that deployment is available here.
- Runner/readset/real smoke use sibling path setup, eager Torch/NumPy imports,
  deferred imports to avoid runner/readset cycles, and a pinned numerical tree.
  `g3_pq_policy/__init__.py:1-4` points policy imports at checkout `prismaquant`
  **without overlaying the pinned numerical package**. A casual bare
  `prismaquant` import can bind the wrong tree or add unnecessary dependencies.
- Readset runs in CPU preparation and container execution; residency smoke also
  shares CPU prepare/CUDA consume paths. New imports affect both modes.
- Preflight performs module-scope argv parsing, AST execution, CUDA decoding and
  exclusive output writes. It is not an import-safe serialization helper.

Per-file context and exact import/execution source anchors are in the ledger.
No GPU, container, pinned fleet adapter or end-to-end quality job was run.

## Validation evidence

PASS: independent AST census, exact slices/scopes/options, all 40 native calls,
23 individual gaps, 2 owner sites and 9 delegated writes; snapshot blob hashes
match recorded identities. All 23 option witnesses preserve ASCII/non-finite
behavior and reject arbitrary objects; stream `dump` matches `dumps`, while
sorting changes witness bytes. No assigned executable was imported.

Initial combined validation **failed**: test snapshot Git history could not
resolve the research path. Research/HEAD comparisons succeeded separately in
the source worktree; revised snapshot validation independently recomputed Git
blob hashes rather than assuming its Git history matched the source worktree.

[Test evidence {"artifacts":{"stderr":{"digest":"48d70e6296e6013b428175ffc532800b0a478cf3d9e8a4f247b406b8fe80dac1","size":902},"stdout":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0}},"operation":"174d8fe7164eb3a780b5a0173b03577c51aa89df15e68f8d3bbb228541a51521","receipt":"dd9d893d127e3e7f44eee671a09a0debb4ceaeed92ad4c1b5ed2f422ba5c951d"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"cb132dd48777a59271f5f1e6f7376b5845010f8aced1b76ed2183dbc8eaeb61d","size":344}},"operation":"4f36b2401cd61fca802f1fd62518b5b7034e228a4660a1488f37e5c33e016f37","receipt":"f49ea2e7fa81f6f46e6a95a61e649c9b3614a031815d6747d36c85cff78f4d78"}]
