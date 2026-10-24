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
The mutable witness source is read once into a private byte-exact temporary copy
outside the checkout. Strict parsing, public verification and the fingerprint
record all use that owned copy, which is removed on success or refusal. Replacing
the source pathname cannot substitute the receipt being verified. Witness,
expectation and verdict parsing reuse `schemas.strict_json_loads`; temporary
expectation JSON does not require sorted encoding.
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

Initial implementation command (repository root; explicit marker selects the
opt-in suite directly). This predates the receipt-ownership repair below:

```sh
python -m pytest tests/test_served_task_backend.py tests/test_served_task_public_verifier.py tests/test_served_task_tessera_crosscheck.py tests/test_served_task_wiring.py tests/test_task_suite.py tests/test_task_reader_capability.py tests/test_task_quality_replay.py -m task_suite -q --basetemp="$TMPDIR/pytest"
```

Observed: **169 passed**, exit 0, CPU. No skipped/unavailable checks in this gate.
O5 documentation accuracy remains review-only. Historical red acceptance receipts
in `tests/SERVED_TASK_RECONCILIATION_EVIDENCE.md` are unchanged.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"23be31de4746c2cbf4d346ea6a950441943228f69823fdaa5bb2117e5bc4ca40","size":430}},"operation":"9abe9092ee2616ceaf9d6aa7c75f6769be15c466181cbfbd3934db8a8f95ab44","receipt":"3f926edb88b07089f675789ec1c8ca1f0deb4464a36a25672a83c8a52370fb17"}]

### Receipt ownership and repository guard repair

Added two synthetic CPU regressions in `test_served_task_receipt_identity.py`,
using the actual archived producer CLI. They replace the original source just
before the subprocess reads its input: invalid-to-valid must refuse, and
valid-to-valid must verify and record the original owned bytes. Both assert
byte-exact input and temporary-file cleanup. The acceptance files remain unchanged.

Before repair, the following command reproduced all reported causes: **4 failed,
5 passed** (two receipt-ownership cases, duplicate strict-JSON helper and new
sorted-JSON sites). No baseline was expanded or weakened.

```sh
python -m pytest tests/test_served_task_receipt_identity.py tests/test_duplication_baseline.py -m 'task_suite or not task_suite' -q --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"30f1d1dd81edd66babab5fe2908f12fae12a507201a3425d33d94e49110148d7","size":8394}},"operation":"8652936c8daccc809b58182c7e2c6ea3df09ce8a5292c4decd1c7d23fe50e044","receipt":"abca6df697ccdb38b05a68eef3c9d353f718ae4afa26d130871285558a34e0f8"}]

After repair, acceptance, HF and receipt-identity regressions: **171 passed**,
exit 0. No skips or unavailable checks.

```sh
python -m pytest tests/test_served_task_receipt_identity.py tests/test_served_task_backend.py tests/test_served_task_public_verifier.py tests/test_served_task_tessera_crosscheck.py tests/test_served_task_wiring.py tests/test_task_suite.py tests/test_task_reader_capability.py tests/test_task_quality_replay.py -m task_suite -q --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"d676cada7c58a4599cebaf2d8d7824502a18e8b75f02a459a879667388071879","size":430}},"operation":"35e72e82cfb7e0fa3e4123aa3efeab7b9237b81a942674b38810c0c013626e28","receipt":"14f707be23f6f8db784f6b23c52ab2d45874dbb55788df1c374a364855ab7f2a"}]

Repository duplication/architecture/docs/ship gates: **97 passed**, exit 0,
14 torch JIT deprecation warnings. The explicit marker expression includes both
ordinary and opt-in ship-gate tests, so none were skipped.

```sh
python -m pytest tests/test_duplication_baseline.py tests/test_duplication_alias_scope_audit.py tests/test_architecture_doc.py tests/test_docs_entrypoints.py tests/test_docs_staleness.py tests/test_doc_claims_driver_truth.py tests/test_ship_gates.py -m 'task_suite or not task_suite' -q --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b6f489081ac0b895229dd28012baf0c2506e0595fab63150743851d5adb8fbde","size":840}},"operation":"de4c846fe7a7fb6a3d70102d0293bc5a2135b5febdb9e68c16f3228e3ca97350","receipt":"5234eb4023e964c774548e8426b16a0ac752a96f28562592c4714f8a2c469957"}]
