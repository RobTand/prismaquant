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

## Current-main repair, 2026-10-27

The preserved implementation is reused from PR #2963 on branch
`prismaquant-2826`, based on unchanged main
`05b62b4742793a2188abd4ed041fc4d46fd6c2c0`.
The tested fixture/smoke commit is
`8e7dbf876be6a60916866872016f5400e79281d7`; acceptance-only commit
`ec41a41f5ae` precedes the fix.

Current pins require
`/home/rob/venvs/pq-pin-aaa41700-pb20261016-git/bin/python`.
The first attempts with the October 9 layered interpreter were refused before
pytest for an older Tessera pin. They provide no test coverage. The replacement
pbtest runs verified installed PrismaBuild `027103d9a8417e06c7f13356e58779a313cd7088`
and Tessera `aaa417003f0ee690e8ace138acd18df7f9531369` before pytest.

| Check | Fleet action | Outcome |
| --- | --- | --- |
| Unchanged-main held selector, current pins | `bbc6e5aef7f2d841687e4dacf4747d64e1b2e0c60451f2710ef4912219178c40` | Missing retained stdout; 1 failed |
| Preserved acceptance before fixture fix | `ba432c8f791f98b4071e565c0d45fa3e0c45931bc1a85389b96675b747065679` | Fixture-owned queue assertion; 1 failed |
| Repaired acceptance | `8a5ab21afcd0a5978382450319b601caa201dc43ecd06289d67081fab10a08fd` | 1 passed, no skips |
| Repaired original selector | `67bcb2005afc8d28946c1ca8b3d334aef81a9ff23be1007e8dd80c11df8ab248` | 1 passed, no skips |
| Touched-module compilation | `b9d4ea6e5d7c6e3c5669647c436147b1473194a9efd9f77afe069d53ee2b4399` | Executed, exit 0 |

The acceptance CAS receipt is
`97ab22d09b24b5c75212809f253ea14ae7b56067acd560c5d523138f894207ef`;
the original selector receipt is
`455571aa061fe69470d8fd2fab2cf181d42006163d642b90b489fca77cba5b7b`.
Terminal evidence and log digests were read back with `pbwait --endings`.

The 11-file adjacent run collected 241 tests: 240 passed, 1 failed, no skips.
Original-source admission, SDK5 pin, SDK availability, seal, architecture,
documentation and duplication checks passed. The failure is the unchanged
`test_source_consumers_restore_the_public_plugin_import_graph`, action
`f525cfa19a1f40f57c214a68ad838e9d5f83cb9ebc2bbb5375e417022202555d`.
Its nested pilot fails at `tests/test_pilot_verified_result_1293.py:135`
with `KeyError: 'published_unix'`, owned separately by
[prismaquant#2996](https://github.com/RobTand/prismaquant/issues/2996).
A standalone pilot run on unchanged main, action
`ffcc05495fd8a3d051dd168619afc8c8b6442b4f265c2f64c6b45b01b34f40cb`,
reported 23 failures at that same fixture field on sparky. The candidate's
standalone pilot reported 23 passes on dl380g10; this variation does not
establish a repair of #2996.

All local evidence is under `/mnt/shared/paperclip-runs/PRI-100/`:
`main-current.json`, `red-current.json`, `green.json`,
`pilot-main-current.json`, their verified ending and log files, and
`collect.py`, which re-collects terminal records and checks log digests.
The exact commands and transient-unit names remain in the Paperclip task.
The full composed `pb-tests` check, independent review, and merge remain required.
No native GPU, served, performance, residency, or quality claim follows.
