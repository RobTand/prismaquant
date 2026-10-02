# Census builder byte digests

Scoped completion of #2087, parent #1301. The research census-cache builder
uses the existing `digests.bytes_sha256hex` profile for three load-bearing
sites. Every site hashes the same already-owned bytes as before.

| Site | Compared or published identity | Preserved caller contract |
|---|---|---|
| `_bound` | Input file bytes compared with the supplied SHA-256 | One read; original open errors and labeled mismatch `SystemExit` |
| `_write` | Exact atomically published bytes reported in manifest/summary references | First-writer publication completes before hashing; no readback; atomic refusal unchanged |
| `main` projection fence | Current layer-config bytes compared with their previously bound digest | Refuse a changed config before writing its planner projection |

All JSON encodings remain at their existing call sites: unsorted strict
indented layer configs, sorted strict indented manifests, sorted lax indented
summaries and existing stdout spellings. Output timing, summary-dict mutation,
the final LF and public helper signatures also remain unchanged. No identity
migration, new profile, source acquisition or wire/cache implementation occurs.

The actual unchanged builder ran through PrismaBuild action
`3e56c42340a47f634662b8a70baf327090e7c9b4e54d770f00431d1545e5267e`, snapshot
`db965dcddc92eafb9347deb9a1051c73b8bdcd5d` rooted at main `8811ad5f1e9`.
It passed eleven byte/error tests and failed exactly the three intended
owner-routing seams, with no skips. Nineteen immutable old outcome rows
cover empty/binary/Unicode bytes, mismatched input digests, missing/directory
inputs, existing-file and dangling-symlink publication refusals, and unchanged
versus changed post-binding config behavior. The old log supplied the fixture;
replacement production never records these expected outcomes. Routing tests
also enforce publication-before-digest ordering without a file readback.

This independent branch removes exactly three primitive scopes from 508 to
505. No duplicate allowance grows. Other pending #1301 slices remove distinct
scopes; the actual combined baseline must be checked during integration. No
CPU/GPU speedup, original-model, export or serving qualification is claimed.
