# Newline roster identities (#1301, #1446)

## Scope and source inventory

This slice extends `prismaquant/digests.py`, which already owns the JSON,
file, bytes and text recipes from #1327 and #1372. It does not replace that
module or claim to finish #1301. References below describe the pre-change
source at `66635f81eb0bb371bade769bd4f1038cc42e3852`.

| Site | Class | Responsibility and persistence evidence | Encoding |
|---|---|---|---|
| `prismaquant/joint_layer_quanta.py:173-180`, `roster_digest` | Load-bearing | Campaign unit roster; the joiner consumes the same helper (module contract, lines 44-48). | Sort first; refuse non-exact-str, empty or duplicate entries; join with LF; UTF-8; no final LF. |
| `prismaquant/joint_head_walk_quanta.py:47-55`, `roster_digest` | Load-bearing | Descriptor `roster_sha256` (line 98); checker compares it at line 119; journal collector compares the persisted identity at lines 251-252. | Sort; join with LF; UTF-8; no final LF. No extra validation. |
| `prismaquant/joint_head_walk_quanta.py:99`, `head_walk_quanta` | Load-bearing | Writes each descriptor's `slice_sha256`. | Join the already ordered slice with LF; UTF-8; no final LF. |
| `prismaquant/joint_head_walk_quanta.py:126-127`, `check_quantum_for_roster` | Load-bearing | Compares the descriptor's stored `slice_sha256` before accepting it. | Same ordered-slice recipe. |

The profiles are `newline_utf8_bytes`, `newline_utf8_sha256` and
`sorted_newline_utf8_sha256`. Ordered and sorted inputs are not interchangeable.
The layer caller retains its existing validation and its order of checks.
Embedded newlines are not escaped; `[]` and `[""]` encode identically in the
unchecked profile. These historical properties are preserved, not repaired.

## Nearby recipes that are not migrated

These are inventory leads for #1386, not permission to merge identities:

- `joint_replay_frontier.roster_sha256`, lines 138-144: order-sensitive,
  domain-prefixed JSON strings with eight-byte length framing. It is **not**
  the sorted newline recipe described in that issue's lead list.
- `joint_stage_b_head.roster_sha256`, lines 103-107: canonical JSON of a
  mapping from names to format lists. It is **not** a name-only roster.
- `tessera_joint_aura.load_measured_anchor_input`, line 1098: the same ordered
  newline construction, written into a head-walk journal identity. This large
  loader is outside this first two-module slice.
- `production_weight_cache._qname_set_sha256`, lines 3979-3988: coerces each
  element through `str` before sorting. Its input policy is distinct, and the
  cache/IO workstream also owns changes to this module.

All four are load-bearing; none may collapse onto another encoding without
an explicit migration decision. PACT, MTP export, kernel qualification pins,
and the source-tree framing families remain outside this slice.

## Verification contract

`tests/test_digest_rosters_1446.py` records the pre-change sites through PB,
then checks the migrated sites against that frozen table. It captures both
the bytes passed into SHA-256 and the exact returned value or exception
(type, text and chained cause). Inputs cover ordering, Unicode, embedded
newlines, empty names, duplicates, tuples, iterators, floats, Paths, nested
containers, `None` and lone surrogates. Descriptor production and checking
cover multiple slice widths and a corrupted stored digest.

Pre-change recording: PB action
`bc697679cb55de563d147cb2b268d63210920034c9518240b062ed20f9393bde`,
35 tests passed, 45 frozen outcomes. Published result blob:
`bccfdcdf1edad5e6a7fef020eef0b29f2a9db42ea8dbb341d56251316a898df8`.

The real stored-value test reads this exact record without modifying it:
`/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/joint-panel/layer-quanta/layer-000.json`.
Its `campaign.prepared_path` names the prepared input. The test compares both
roster helpers over those 36,423 names with the stored `unit_roster_sha256`,
`4da42b8df5cbf1720406ef7f4c19ac5b7f11bdc5e0f51d0fef6378970a1ee6ea`.
CI without that fleet artifact explicitly skips this one comparison; the
portable golden table still runs. Full-suite and post-change receipts belong
in the PR and campaign result after their terminal statuses are checked.

No arithmetic, dtype, device, cache/residency behavior, format, pipeline
default, serving lane or ship gate changes. This is not a performance claim;
there is no KL/bpp/runtime comparison because no rendering or metric changes.
