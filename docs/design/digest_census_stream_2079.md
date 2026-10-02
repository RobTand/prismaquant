# Census checkpoint JSON streaming

Scoped completion of #2079, parent #1301.

The census seal uses `digests.checkpoint_json_sha256`: the existing
`DIRECT_UTF8_STRICT` codec, partitioned at depth two through an encoder
adapter, and the existing `_stream_sha256` loop. The actual hash loop retains
its implementation. The stream interface is explicit and imports only the
standard library. The reader supplies `CensusCacheError`; the owner imports
no reader or tensor package.

| Site | Exact inherited recipe | Owner |
|---|---|---|
| `tessera_census_cache._canonical_chunks` | Sorted compact strict UTF-8 JSON; first two dict/list levels streamed separately; deeper leaves use the direct encoder | `_CheckpointJsonEncoder` using `DIRECT_UTF8_STRICT` |
| `canonical_json_sha256_of_loaded` | SHA-256 of those UTF-8 string chunks in iteration order | `checkpoint_json_sha256` and existing `_stream_sha256` |

This is load-bearing: `seal_roster` compares the result against the stored
checkpoint identity SHA before loading its unit journals. The public reader
function retains its signature. No graph normalization, new source read,
cache or stored-identity migration occurs.

The inherited depth and error order are deliberate. A shallow non-string key
raises the reader's own error, while deeper numeric-key leaves retain their
direct-encoder acceptance. Mixed-key sorting, unsupported values, cycles,
nonfinite numbers and UTF-8 surrogate errors retain their original exception
type, text and cause. An earlier nonfinite leaf still fails before a later bad
key; a whole-graph precheck would change that behavior. The chunk partition
also preserves the Unicode error's leaf-relative offset and avoids one large
whole-roster JSON allocation.

Actual old production outcomes were captured through PB action
`e7a5fcbf70f9f15704a2c56cdc909c6081afc9598f00df67ed2fe83375869726`, source
snapshot `48b696a72ce991c702a5a20d3040a789b924897c`, before replacement source
was sealed. Eighteen byte/error cases passed; one intended owner-routing seam
failed, no skips. The immutable old log supplies all 18 fixture rows; no
replacement-code output was recorded into the goldens. Existing corpus/source
identity and selected-cache tests provide integration coverage.

This branch removes exactly two primitive scopes, 508 to 506, with no new
allowance. Other pending #1301 slices remove different scopes; each actual
combined baseline must be checked after integration. No CPU/GPU speedup,
memory reduction, original-model or serving qualification is claimed.
