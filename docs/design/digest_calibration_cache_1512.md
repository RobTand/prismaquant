# Digest consolidation — `tessera_calibration_cache` (PQ #1512)

Round-3 subsystem slice after the digest-site ratchet (#1508): the largest
non-owner file, `prismaquant/tessera_calibration_cache.py`, carried **16 gated
primitive digest sites** (11 `hashlib`, 5 sorted-JSON). This slice removes 14
of them onto `prismaquant/digests.py`; the ratchet baseline shrinks
**697 → 683**. Every delegation is byte-identical; the golden fixture
`tests/fixtures/digest_calibration_cache_1512.json` records the old outputs on
the pre-change tree (PrismaBuild run on `bddf455bdba`, the ratchet head) and
`tests/test_digest_calibration_cache_1512.py` asserts the new code reproduces
them, so existing receipts, seals and identity chains still verify.

## Site-by-site map (old line numbers at `bddf455bdba`)

| Old site | Recipe (old bytes) | Now |
|---|---|---|
| `_json` (L79) | `json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'` | `indent2_json_file_bytes` (new owner function) |
| `capture_identity` (L98) | `hashlib.sha256(census_raw).hexdigest()` | `bytes_sha256hex` |
| `CaptureSourceAuthentication.adopt_streamed_identity_cache` (L339) | `hashlib.sha256(raw).hexdigest()` | `bytes_sha256hex` |
| `CaptureSourceAuthentication.admit_derived_census` (L356) | `hashlib.sha256(raw).hexdigest()` | `bytes_sha256hex` |
| `_load_execution` identity derivation (L596–599) | streaming `json.JSONEncoder(sort_keys=True, separators=(',', ':')).iterencode` into `hashlib.sha256` | `DIRECT_ASCII_LAX.sha256_streamed` (new `JsonProfile` method) |
| `_load_execution` chain seed (L607) | `hashlib.sha256(b'').hexdigest()` | `bytes_sha256hex(b'')` |
| `merge_load_execution` (L623–624) | `hashlib.sha256((left + right).encode()).hexdigest()` | `hex_chain_sha256hex` (new owner function) |
| `fold_load_receipt` (L694–695) | same chain recipe | `hex_chain_sha256hex` |
| `require_capture_contract` (L901) | `hashlib.sha256(raw).hexdigest()` | `bytes_sha256hex` |
| `CaptureMetadataOwner.__init__` manifest digest (L947) | `hashlib.sha256(raw).hexdigest()` | `bytes_sha256hex` |
| `CaptureMetadataOwner.__init__` identity compare (L954–957) | `json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False)` | `DIRECT_ASCII_STRICT.text` |
| `CaptureMetadataOwner._assert_unchanged` (L979) | `hashlib.sha256(bytes).hexdigest()` | `bytes_sha256hex` |
| `CaptureMetadataOwner.load_execution` policy JSON (L992) | compact strict dumps | `DIRECT_ASCII_STRICT.text` |
| `CaptureMetadataOwner.load_execution` cached digest (L1000) | `hashlib.sha256(raw).hexdigest()` | `bytes_sha256hex` |

## Deliberately left in place (2 sites, still ratcheted)

- **The guarded source hasher `sha256()` (L39–82).** Incremental
  `hashlib.sha256()` + `hashlib.file_digest` *is* the production-cache guard:
  descriptor-alias opens, stat fences against mutation/replacement, page
  release via `posix_fadvise`, and admission resource checks interleaved with
  the block loop. It is not a plain byte profile; folding it into the digest
  owner would move production cache machinery and deserves its own review.
- **`CaptureSourceAuthentication.__init__` identity JSON (L186).** Default
  separators (`', '`, `': '`), `sort_keys=True`, `allow_nan=False`. The string
  is only ever re-parsed (`json.loads` at L368/L425), never persisted or
  compared byte-wise against stored artifacts, and no existing profile spells
  default-separator JSON. Left rather than silently re-encoded; a future
  profile may absorb it.

## New owner capabilities

- `JsonProfile.sha256_streamed(value)` — hashes the profile's own encoding by
  streaming `iterencode` chunks in UTF-8; byte-identical to `sha256()` for
  accepted values and to the old hand-rolled lax encoder loop (the loop never
  set `allow_nan=False`, so `DIRECT_ASCII_LAX` is the exact match).
- `indent2_json_file_bytes(value)` — the pretty capture-file form
  (`sort_keys=True, indent=2, allow_nan=False`, default separators, trailing
  LF, UTF-8).
- `hex_chain_sha256hex(left, right)` — the ordered hex-identity chain;
  delegates to `text_sha256hex(left + right)`.

## Behavior notes

- The `sha256_streamed`/`DIRECT_ASCII_STRICT.text` split preserves an
  existing asymmetry on purpose: the streaming derivation is lax about NaN
  (as the old encoder was) while the owner-side identity comparison stays
  strict, exactly as before the change.
- `merge_load_execution`/`fold_load_receipt` receipts recorded before this
  change still verify: the chain recipe and every digest above are unchanged
  byte-for-byte, pinned by the golden fixture.
