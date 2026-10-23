# Self-contained SDK5 release smoke (#2826)

The existing `test_held_producer_context_authenticates_and_joins` selector and
fixture-call signature are unchanged. Its evidence now comes from
`tests/native_carrier_fixture.py`, entirely under the test's `tmp_path`:

- A sealed standard-capture request, CAS payload/receipt, private helper root,
  synthetic claim, immutable selected attempt, stdout/stderr evidence and
  scope attestation are constructed with the installed SDK's writers.
- Installed runtime source bytes are copied into the temporary helper root.
  Runtime writer identities are temporarily rebound with `monkeypatch` and
  restored at teardown. The fixture models a claim directly rather than
  requiring live scheduler measurement-census admission. No worker, broker RPC,
  native execution or GPU operation is performed.
- Selectors, identity mirrors and receipt/runtime/attestation digests come from
  the fixture evidence. The real installed SDK5 public reader authenticates it
  with `require_native_producer_context=True`; the real capture binder and
  consumer join are not replaced.
- Matching certified identity reaches the intentional placeholder helper-tree
  refusal. Certified foreign nonce/scope controls still refuse. D32 dev-mode
  controls join and stamp rather than refusing, without computing a helper-tree
  digest. SDK5 pins and application/admission policy are unchanged.

This CPU fixture proves **contract integration**, not real fleet/native-GPU
qualification or recovery of the historical producer's missing retained stdout.
It needs no served checkout, historical Git object, live artifact, download or
install at test time. The original smoke output assertions remain in force.

## Verification

```sh
python -m pytest tests/test_held_producer_release_gate_2826.py tests/test_held_producer_sdk5_smoke_2152.py tests/test_original_source_admission.py tests/test_pb_sdk5_pin_2152.py tests/test_fleet_sdk_skip.py -q --basetemp="$TMPDIR/pytest"
```

Observed on the supplied CPU environment: **157 passed**, no skips, 14 existing
Torch Python-3.14 deprecation warnings. Retained stdout was read back. No GPU
qualification was attempted or required. Early fixture-construction checks
failed on differences between served PrismaBuild main and the installed SDK5
pin; the fixture now accommodates the runtime/record owner differences and
uses the installed broker socket identity as evidence only, without connecting.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"76fc012ebabd7cb3a8d6f7aaa49c09ca30b01c33d215fbec30169d8a8490fd95","size":993}},"operation":"f46bfa5657a6b9efebee6f5ccb3894b50641cb459b36d5e045099bfc08a6adba","receipt":"b42b3efe115f577c8d4fb254d991fb078d681b8155cbdbdb7e1dd8c9599abb77"}]
