# Served task consumer reconciliation (#2894)

Reuses PR #2470's module/API boundaries from approved head
`80f95d02717d1e4c2a88dbe13683199041e340e4`, selectively ported onto HF-only
main. This does not create another adapter, delivery lane or deployment project.
Adjacent producer issues #2995/#3004 are not duplicated. No dependency pin changes.

## Portable configuration and independent facts

All deployment inputs are supplied by task configuration, for example:

```json
{
  "backend": {
    "name": "served",
    "serving_runtime": {
      "witness": "evidence/witness.json",
      "expected": "evidence/independent-expected.json",
      "endpoint": "http://localhost:8000",
      "served_alias": "configured-model",
      "attempt_id": "host/boot/pid/start_ticks",
      "ranks": [0, 1],
      "verifier": "producer/tools/verify_endpoint_witness.py",
      "served_dir": "artifacts/selected",
      "tokenizer_dir": "artifacts/tokenizer"
    }
  }
}
```

Values are illustrative, not defaults. Normal task/sampling configuration is also
required. Paths must be consumer-accessible. `tokenizer_dir` is optional when
files share `served_dir`. The combined independent expectation document supplies
endpoint, served_alias, attempt_id and complete ordered ranks matching config,
plus `artifacts` (filename to SHA-256/byte count) and `tokenizer` (`files` mapping
filename to SHA-256/byte count, loaded `backend`, `vocab` mapping and
`special_ids`). Never generate expectations from the receipt. Alias-only and
size-only inputs are insufficient.

Deployment requires W1's admitted standalone public verifier. The admitted
Tessera aaa41700 package lacks this public surface; application code does not
import or default to the test fixture. The configured CLI receives selected
artifact/tokenizer directories, all deployment expectations, `--offline`, and
separate temporary artifact/tokenizer expectation files outside the checkout.

## Authority and proof boundary

`bind_served_task` itself requires a successful bounded public verifier process;
preflight consumes that result once. No local-only authoritative pass exists.
Require exit zero and one strict JSON object with schema
`tessera.endpoint_witness_verdict.v1`, mode `offline`, verdict `valid`, reason
null, proof_scope `recorded_runtime_byte_binding`, current_endpoint_verified
exactly false. Process failure, timeout, malformed/ambiguous JSON, unknown schema,
refusal and live claims refuse. The binding records receipt fingerprint,
endpoint/alias/attempt/ranks, verified artifact/tokenizer facts and public verdict.
Offline success is **recorded launch evidence**, not current availability.

Producer source inspected at Tessera
`17b3fdb396410698c9a142615345f046d2f94a5d` establishes the public contract.
Producer-supported source-to-loaded normalization ignores only pre-tokenizer
ByteLevel trim_offsets and decoder ByteLevel add_prefix_space/trim_offsets/
use_regex, including nested sequences. Exact independent loaded-fact expectations
remain exact, and ID-affecting changes refuse. Delegation removes the old roughly
600-line duplicate semantic implementation rather than maintaining drift patches.

Served paths import no serving runtime, start no ranks, seal no identity, invoke
no HF inference and never fall back to HF. Measurement/replay bind then explicitly
refuse pending #2430 before HF execution. Current HF validation, initialization,
reader, preflight and replay behavior are preserved, including model runtime
initialization and source-checkpoint imports excluded from the old PR port.

## Evidence limits

The producer's committed live-measurement note reports a real one-rank tiny
Qwen3 witness. Research did not obtain/replay its raw receipt; these checks do not
change that. The unchanged attributed test-only producer snapshot is exercised
through the actual public CLI with tiny synthetic constructors, without downloads,
historical git objects or dependency installs. Valid two-rank joins, corruption,
normalization and failure paths are **synthetic CPU regressions only**, not W1
launch acceptance, D50 qualification, quality or current availability. Actual
served measurement and replay remain #2430.

## Implementation verification

Command (repository root; explicit marker selects the opt-in suite directly):

```sh
python -m pytest tests/test_served_task_backend.py tests/test_served_task_public_verifier.py tests/test_served_task_tessera_crosscheck.py tests/test_served_task_wiring.py tests/test_task_suite.py tests/test_task_reader_capability.py tests/test_task_quality_replay.py -m task_suite -q --basetemp="$TMPDIR/pytest"
```

Observed: **169 passed**, exit 0, CPU. No skipped/unavailable checks in this gate.
O5 documentation accuracy remains review-only. Historical red acceptance receipts
in `tests/SERVED_TASK_RECONCILIATION_EVIDENCE.md` are unchanged.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"23be31de4746c2cbf4d346ea6a950441943228f69823fdaa5bb2117e5bc4ca40","size":430}},"operation":"9abe9092ee2616ceaf9d6aa7c75f6769be15c466181cbfbd3934db8a8f95ab44","receipt":"3f926edb88b07089f675789ec1c8ca1f0deb4464a36a25672a83c8a52370fb17"}]
