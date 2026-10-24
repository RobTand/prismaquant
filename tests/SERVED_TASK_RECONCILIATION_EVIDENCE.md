# #2894 research / red acceptance evidence

## Design decision made before test construction

Verdict: **as asked**. Reconcile PR #2470, not a parallel adapter.

Observed base: `6d8288adc94b3c4b624c143ed372fb6513f32784`.
Observed existing approved source: `80f95d02717d1e4c2a88dbe13683199041e340e4`.
`git merge-base --is-ancestor` returned 1. Main lacks both served modules and
`task_suite.validate_config` accepts only HF. Approval does not establish delivery.
At the old head, `preflight_served` permits `public_verifier='not_configured'`;
`run_public_verifier` ignores `completed.returncode` and parses only stdout's
last line. Its backend duplicates producer semantics and requires raw tokenizer
backend equality, unlike the current producer's supported ByteLevel loading.

Simplest root-cause fix: reuse those module/API boundaries and wiring, but make
one mandatory successful public offline process the authority for every binding.
Keep strict config/JSON intake and label consistency, remove the duplicate
producer-semantic checker rather than evolve a second verifier. Record fingerprint,
verified byte/tokenizer facts and endpoint/alias/attempt/ranks. No live/quality
claim follows. Keep HF initialization and reader behavior from current main;
the old PR's unrelated HF import changes must not be copied blindly.

Rejected: unchanged cherry-pick (keeps bypass, exit-status bug and drift); new
adapter (duplicates #2469/#2470); merely changing the old tokenizer equality
(symptom-only and retains competing producer semantics); dependency/fleet upgrade
(unnecessary to this configured-standalone-verifier boundary, and overlaps
producer/deployment work). Supplied #2995 and #3004 are adjacent producer-contract
work, not evidence of a delivered consumer. Their supplied titles alone do not
establish exact duplicates. Parent #2430 still owns actual served measurement.
No follow-up design flaw or additional delivery lane is proposed.

The archived public oracle fixture and its README document producer provenance,
synthetic expectations, W1 evidence limits, and the installed-pin limitation.
Application code must never default to or import that test-only oracle.

## Final acceptance run before implementation

Command from repository root:

```sh
python -m pytest tests/test_served_task_backend.py tests/test_served_task_public_verifier.py tests/test_served_task_tessera_crosscheck.py tests/test_served_task_wiring.py -m task_suite -q --tb=short --basetemp="$TMPDIR/pytest"
```

Observed: **126 failed, 5 passed; exit 1**, no collection errors or skips.
Failures are unmet served recognition/binding or descriptive assertions that
PR #2470's modules are absent, not missing third-party imports. Six negative
cases were tightened to require served recognition before exercising damage,
avoiding accidental passes on main's generic HF-only tokenizer error.
The passing guards validate three synthetic producer fixtures (plain and
flat/nested ByteLevel), unchanged oracle provenance, and existing HF routing.
They do not claim the consumer is resolved. Every objective acceptance command
also selects the red consumer cases.

Use `-m task_suite`: existing conftest otherwise skips this opt-in suite.
This marker selects tests directly, not a private test layer. No private layer,
submission wrapper, download, git-history lookup, checkpoint, GPU or service is
required by the checks.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"950921ccf4d77e775514f6d9e30eff092cf8b3b7c632bced619d56ce93db34ed","size":96351}},"operation":"dca37f28aab27ccef79ac0d574bdddaf365d8ec495162078292c9dedaafbe471","receipt":"7eae399beca1dfff73dd761ec8ccbc9be5972d2d7ca0d7ee5360b2687f0741c3"}]

## Existing HF task regression baseline

```sh
python -m pytest tests/test_task_suite.py tests/test_task_reader_capability.py tests/test_task_quality_replay.py -m task_suite -q --tb=short --basetemp="$TMPDIR/pytest"
```

Observed: **38 passed; exit 0**. This is a non-regression baseline, not delivery
of served support. Retained stdout was read back for both final runs.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"47864b4f57b07e0b5c95038f2ec4d4d58592293dc9645a488087717d45f1eef6","size":269}},"operation":"aec1713e56e3903f74bdb05c62d7d090529704a0af6f29a5577b6c40ce609073","receipt":"ee21af378983f36237aa05da26a792309ef0e5edc7452e7f5beef59a8dc0f65f"}]

Research changed only tests and their fixtures/evidence. No implementation,
publication, merge, current endpoint test, or served measurement was performed.
