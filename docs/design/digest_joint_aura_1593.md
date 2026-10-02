# Digest consolidation — `tessera_joint_aura` (PQ #1593)

Round-3 subsystem slice after the digest-site ratchet (#1508), continuing the
#1512 (`tessera_calibration_cache`) shape on the next-largest gated file:
`prismaquant/tessera_joint_aura.py` carried **12 gated primitive digest
sites** (10 `hashlib`, 2 sorted-JSON; baseline 688 at origin/main
`145ddd28`). This slice removes 11 of them onto `prismaquant/digests.py`;
the ratchet baseline shrinks **688 → 677**. Every delegation is
byte-identical; the golden fixture
`tests/fixtures/digest_joint_aura_1593.json` records the old outputs (old
spellings verbatim on the pre-change tree) and
`tests/test_digest_joint_aura_1593.py` asserts the new code reproduces them,
so wire receipts, marker fences, head-walk rosters, pretty bank files and
qualification seals recorded before the change still verify.

## Site-by-site map (old line numbers at `145ddd28`)

| Old site | Recipe (old bytes) | Now |
|---|---|---|
| `_read_verified_wire_blob` (L249) | `hashlib.sha256(blob).hexdigest()` | `bytes_sha256hex` |
| `_read_wire_bytes` (L328) | `hashlib.sha256(blob).hexdigest()` | `bytes_sha256hex` |
| `_json` (L336) | `json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"` → `.encode()` | `indent2_json_file_bytes` (the #1512 pretty-file owner) |
| `load_measured_anchor_input.walk_one` (L707) | `hashlib.sha256(blob).hexdigest()` | `bytes_sha256hex` |
| `load_measured_anchor_input` roster (L1114) | `hashlib.sha256("\n".join(roster).encode("utf-8")).hexdigest()` | `newline_utf8_sha256` (the #1446 roster owner; caller order kept — the head-walk roster is never sorted here) |
| `_banked_unit_still_binds` marker (L1168) | `hashlib.sha256(marker.read_bytes()).hexdigest()` | `bytes_sha256hex(marker.read_bytes())` |
| marker fence capture (L1306) | same marker recipe | `bytes_sha256hex(marker.read_bytes())` |
| `_synthesize_render_from_wire` (L1520) | `hashlib.sha256(blob).hexdigest()` | `bytes_sha256hex` |
| `_qualification_file_sha` (L1637) | `hashlib.file_digest(handle, 'sha256').hexdigest()` over the stat-fenced descriptor | `file_digest_sha256hex` (new owner function; `file_sha256hex` cannot take an already-pinned handle) |
| `_qualification_cells_sha256` row (L1752–1760) | `json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')` | `DIRECT_UTF8_STRICT.encoded` |
| `execute` sampling session (L2744) | `hashlib.sha256(session_bytes).hexdigest()` | `bytes_sha256hex` |

## Original remaining site (historical #1593 scope)

- **The length-framed qualification seal `_qualification_cells_sha256`
  (L1749).** `hashlib.sha256(QUALIFICATION_CELLS_SCHEMA.encode() + b"\n")`
  with per-row 8-byte big-endian length framing *is* the seal's identity, in
  the same family as the replay-roster length framing
  (`joint_replay_frontier.py`) that #1386's correction kept distinct. Only
  the row encoder delegated (above); the framing stays in this module and the
  golden test recomputes the whole seal end-to-end from the old spelling.

## Ratchet

`tools/duplication_inventory.py`: gated sites 688 → 677 (11 removed, none
added; the only new owner code lives in `prismaquant/digests.py`, which the
ratchet already owns). No new near-duplicate or same-name groups.


## Length-frame owner completion (Refs #1301, #2059, 2026-10-02)

The remaining qualification constructor now delegates to the explicit
`digests.length_framed_bytes_sha256` profile: exact caller-supplied prefix,
caller-ordered raw frames, unsigned eight-byte big-endian length per frame,
with no trailer, normalization or sorting. The qualification caller retains
`QUALIFICATION_CELLS_SCHEMA + LF`, sorted cells and `DIRECT_UTF8_STRICT` row
bytes. Its generator keeps a single row live and no second full-roster copy.

The existing source-tree v2 owner uses the same byte-frame primitive with
its distinct `SOURCE_TREE_V2 + NUL` prefix and alternating byte-sorted name
and content frames. Domain tags and schemas do not change; source v1 retains
its exact legacy recipe. This is shared framing code, not a collapse of the
two load-bearing identities or an artifact/pin migration.

The existing #1593 immutable row/full-seal goldens and #1762 independent
source v1/v2 framing oracles are reused without rerecording. New routing and
lazy-consumption/error tests pin both callers to the owner. This slice removes
one raw scope from the branch's 508-site primitive baseline (508 to 507); the
separate pending preparation-JSON slice removes five other scopes. The parent
census remains open. No performance or GPU qualification is claimed.
