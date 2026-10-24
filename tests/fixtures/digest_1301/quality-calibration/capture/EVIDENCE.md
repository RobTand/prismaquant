# Capture owner routing evidence

Scope: only `tools/capture_glm_routed_layers.py`,
`tools/capture_glm_routing_replay.py`, `tools/exl3_grouped_down_replay.py`, and
`tools/glm_mtp_capture.py`, plus this capture fixture/ledger fragment and
`tests/test_digest_1301/quality-calibration/test_capture.py`. No owner, shared
census, or baseline modifications. Existing unrelated assignment changes are
not part of this fragment.

## Routes and gaps

The ledger records the original commit
`6d8288adc94b3c4b624c143ed372fb6513f32784`, individual source blob hashes,
scanner family/scope IDs plus original line/column per-call disambiguators,
fixtures, and executable test nodes without parameter suffixes.

Seven scanner calls migrated: four `hashlib.sha256(...).hexdigest()` calls to
`prismaquant.digests.bytes_sha256hex`, and three sorted spaced serializers to
existing `DIRECT_ASCII_SPACED_STRICT.text` (routed receipt) or
`DIRECT_ASCII_SPACED_LAX.text` (replay receipt and status). Newlines and byte
encoding stay at their existing call sites. Tensor interpretation remains in
`_tensor_bytes`, unchanged; no tensor semantics moved into the digest owner.

Five explicit gaps remain, not owner retention claims:

* `result_json`, projection publication, and MTP summary use sorted indent2
  strict ASCII. No exact named owner profile exists. The existing
  `DIRECT_UTF8_INDENT2_STRICT` produces literal non-ASCII rather than ASCII
  escapes. Tests witness `héllo` as `h\\u00e9llo`, strict NaN rejection,
  formatting, and trailing-newline behavior.
* Routed capture progress and completion use insertion-ordered spaced lax
  JSON, with no exact existing named profile. These two calls are outside
  the scanner's sorted-json family, explicitly marked as such in the ledger.
  Tests witness the real consumer key order, not sorted order.

No new profiles were introduced. Reads, scientific/reference checks, and
publication order were not edited. CPU differential tests execute the actual
consumer source blocks/functions against verbatim original fixtures. Capture
intake, clock, and publication dependencies are controlled; torch serialization,
local tensor interpretation, JSON recipes, returned hashes, append/report
order, and refusal/error behavior execute in the consumer. These are not full
model capture or GPU qualification tests.

## Observed checks

Final command:

```
python -m pytest -q -rs tests/test_digest_1301/quality-calibration/test_capture.py tests/test_exl3_grouped_down_replay.py tests/test_glm_mtp_capture.py tests/test_glm_routing_replay.py --basetemp="$TMPDIR/pytest"
```

Result: **70 passed, 6 skipped, 14 warnings** on Python 3.14.4 / torch 2.11.0+cpu.
All seven new capture tests pass. One existing check skipped for an unavailable
external scorer; five existing checks skipped for an unavailable declared
projection-producer interpreter. Warnings are existing torch JIT deprecations.
No GPU qualification is claimed.

Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9517b424979e33b896acc937ba389c86598d5f145c7eb163c051d5e779dba652","size":2045}},"operation":"20a02ed15f7c230b0ddbe0081bcad84b22a0fde5da7b96cf8047620afd4dca70","receipt":"f026a721c03f0e3fe43e03e7e256d96f64b36356d92efa754181cb842a56460e"}

Earlier runs failed while developing the harness: seven failures from initially
using nonexistent SORTED_* names (corrected to the actual DIRECT_* owners),
then two failures from a misaligned replay fixture slice and missing projection
publication dependencies. These were corrected before the passing runs.

An additional historical-source provenance command through run_tests could not
complete: the isolated test snapshot does not retain the original commit's
source paths in git (`git show <commit>:tools/capture_glm_routed_layers.py`
returned exit 128). Original source and blob hashes were inspected in the
working repository when producing the verbatim fixtures; the historical git
check is not claimed as retained passing test evidence. Consumer differential
tests are self-contained and do not depend on historical git objects.

Test evidence {"artifacts":{"stderr":{"digest":"228e085dde38b9e519420ec1347d65961223680fdc983308c7a9f9b78c1d2c10","size":922},"stdout":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0}},"operation":"da863029f539bc35f0634aa84ad5253f1335b30538a91f8012a203d240fcd789","receipt":"04ceaf03ad8fcac3fe59f934aab2460a2b5ea88516f0b2eb34fbafca0acf8c6e"}
