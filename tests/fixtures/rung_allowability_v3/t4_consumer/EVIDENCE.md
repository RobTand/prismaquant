# Issue #2421 — failing-first CPU consumer evidence

Base inspected: `6d8288adc94b3c4b624c143ed372fb6513f32784`.
PR2406's final approved head `e8cb42e65c89013fdceb305d434c8221b9698376`
is already an ancestor of this base. No Campaign/application source, runtime pin,
active index, kernel or serving qualification was changed during research.

## Design before acceptance implementation

The consumer delegates admission/classes/timing to the canonical producer, but
`load_rung_allowability` rejects all families without a legacy v11
`allowable_rungs` rule. The real E2M1 registry entry publishes its reader range
and step, not that WINDOW rule. Thus even a v3 native-TCQ table accepted by its
canonical producer cannot reach the existing allocator/PACT path.

The proposed local repair is to use the canonical registry's reader constraints
for this v3 native family while continuing to require canonical per-cell
admission and all independent gates. Do not invent a v11 rule, copy the q896
roster or class formula into application code, or confuse representability with
performance admission. Keep existing v11/T8/T16 handling unchanged.

## Inputs and limits

See README.md and provenance.json: the frozen producer is verbatim at Tessera
`8f881a7248f86321beee1c40f15ea3cc617591ff`. The table is synthetic, explicitly
NOT the authentic private table and NOT numerical/native/serving evidence.
The approved q896-only semantics belong to that canonical owner. Scalar R768
capacity is not admission. Tests derive rates and class identities from the
owners; they make no private API, history, network or GPU calls. The installed
runtime's older T4 denial remains separate and must not be bypassed or repinned
by this package. First-ship deployment qualification is not established here.

Existing related work: #2364 and the retained consumer adoption associated with
PR2647 (local candidate head `f2d79d7b908f00d4e17f26d81d5c28d1941f2c19`).
That head is not an ancestor of this base. It contains private-table-dependent
checks; these new checks are offline and expose the public loader gate rather
than constructing a RungAllowability object around it. No duplicate publication
or workflow transition was performed. Campaign remains the exact-head reviewer.

## Final unchanged-source acceptance run

Command, inside the admitted CPU PrismaBuild action (zero GPUs):

```sh
python -m pytest tests/test_t4_menu_followup.py tests/test_rung_allowability.py tests/test_shape_runtime_prices.py tests/test_pact_hull.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **19 failed, 129 passed, 14 warnings in 11.32s**, exit 1.
All 19 new cases reach the canonical validator and fail at the real loader gate,
not test discovery, imports, missing services, API signature or fixture setup:

```text
prismaquant/rung_allowability.py:714: RungAllowabilityError
TESSERA_E2M1_K2: existing v11 allowable_rungs rule required
```

The 129 passing tests are pre-existing regression coverage, not proof of T4
completion. The retained stdout contains the full failure traces and runner
fingerprint. Its first failure was inspected using read_test_artifact.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"dc1f0cb259090ff3ca3e009630293338596df5a0e639168de1ba323b9ee4a23a","size":65694}},"operation":"8169a71ebaa6e36f977e36e2c6bf9a23aeaf9fa861cfe1a28ccb5886baa1963a","receipt":"8e2f2be4114cb9d27d02295a966ec6d8e179f896bc590ad3e5bd9aa8ba6e563a"}]

An earlier research run used an unsupported index_path argument; that test-author
error was corrected before the final run and is not acceptance evidence.
Temporary test-only diagnostics bypassed just the legacy rule gate to inspect
later assertions; they were deleted and are not implementation or completion
proof. No test-only bypass is part of the delivered acceptance checks.
