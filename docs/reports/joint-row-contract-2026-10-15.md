# Cold Joint AURA row contracts, 2026-10-15

Provenance: 2026-10-14 PrismaQuant architecture refactor, base 138069b7993.

The independent review at `062cce9920` found one missed dependency in the
campaign table: AQUA coverage imported `allocator_candidates`, then
`priced_cell`, `format_registry` and Torch, even for an ordinary nonjoint row.
The prior cold test proved payload construction and acquisition scope only.
PB `56c16aa609e0` reproduced the Torch import in a fresh process.

Moving the predicate to `priced_cell` alone would retain that dependency.
The implemented ownership move separates actual scalar contracts from tensor
execution. `joint_aura_rows` owns row publication and validation, immutable probe
identity custody, the identity digest, coordinate binding and attribution
reconstruction. `joint_projection_contract` owns the existing packaged backend
qualification and serialized arithmetic validator. Tensor/model execution stays
in `joint_aura` and `joint_projection_backend`. Canonical claims still pass
through `complete_scientific_price`; no early ordinary-row shortcut replaces
their validation.

The public row operations include:

```python
validate_joint_aura_entry(entry: Mapping) -> bool
cost_entry_is_joint_aura(cost_entry: dict) -> bool
joint_row_binds_cell(cost_entry: dict, name: str, fmt: str, *, where: str) -> bool
joint_identity_sha256(value) -> str
require_probe_alignment(left: Mapping, right: Mapping, *, where: str, message: str) -> None
```

All active callers migrate directly, including the prefix test that previously
imported the digest through `native_moe_panel`. Old definitions and API paths
are removed. The table retains its existing `ScopeRefused` translation for
malformed and rekeyed rows. Internal immutable identities still serialize as
ordinary dictionaries, so serialized validation trust does not survive reload.
Schema strings, source-execution schemas, scalar arithmetic, operation order,
encodings and independent scientific refusal policies retain their meaning.
Package provenance naturally records the new producer source files.

`tools/refactor_joint_row_contract.py` reruns the extraction and caller move.
It follows actual imported export paths and respects lexical shadowing during
the digest rename. Its verification mode checks 155 original operation bodies
after normalizing module qualification/import seams and intentional API names,
and checks that no old API reaches remain. This source comparison supplements
the behavior tests; it does not substitute for them.

The digest golden table was not regenerated. Its test loads the actual new
callable but retains the old pytest case IDs used by the frozen oracle. Thus
the old expected digest/error bytes remain the independent comparison. Two
redundant assertions were removed: function-object identity in the digest
round-trip test and an inheritance assertion already exercised by actual
coordinate-error catches. Byte/refusal and real row safety tests remain.
The duplication baseline shrinks for the old digest owner and the disappeared
`_qualification` name pair; no new helper duplicate is admitted.

## Verification

Every test, compile and verification execution ran through the published
PrismaBuild tools. Each action used the strict interpreter
`/home/rob/venvs/pq-pin-d2e98b3c-pb20261015-git/bin/python`, native threads 1,
PB-assigned affinity and an immutable frozen checkout. Suite shards reserved
CPU 1 and memory 6 GiB each; compile/proof actions reserved CPU 1 and memory
2 GiB. No GPU work or performance claim is part of this metadata unit.

The new cold regression uses a row produced by the existing tiny streamed
Joint AURA fixture. A fresh child process forbids Torch, NumPy, Transformers,
Triton, vLLM, tools, serving code and the numerical Joint AURA/allocator owners.
It checks ordinary rows, genuine measured rows, corrupted scalar values and
rekeyed rows. Before extraction all four cases failed at the numerical import
(PB `16a2ef757c6a`, exit 1). All four now pass with their real refusal behavior.

Final retakes:

| Action | Actual result | Result SHA-256 | Bytes |
|---|---|---|---:|
| `a2bc44ae022a` | 202 passed | `f67c139fc7135fdf6968be0da6aaec9363bb6d6b7917be01534cf2fe4c677925` | 121250 |
| `431bcc39d6b8` | 104 passed, 32 skipped | `770116679c05732864711bad60c6abce9d390dfae07bbaef3103cb37d3fa3586` | 80927 |
| `a6363f790a96` | architecture/docs and currency: 39 passed | `6df6a1a79c171a3b8d9b62f4036352a430eeaa95958f11597971745e81a31fde` | 22096 |
| `aea7993b13a4` | 155 preserved operation bodies, no legacy APIs; compile 84 files | `4c4d717109e011ffa948af1c59c16555891d5786c6e9b68f94477b6793d8419b` | 8966 |
| `6eac569466fe` | migration rerun from a clean prior checkout, then 174 passed | `19bc38f0cd6f3830b614dd82d09aa068e9af841ca6f3f5829783c6518f73964f` | 10907 |

The retakes cover cold coverage, quantum identity reuse, metadata publication,
Stage B completion, the native prefix caller and duplication ratchets. Earlier
expanded checks cover canonical quantities, pricing, packed/dense measurements,
sequence attribution, assignment diagnostics, acquisition, native panels and
receipt tables. Consolidating the latest observed outcome per node across the
staged checks yields 1,467 passed and 32 skipped in 51 files, with no remaining
failed nodes. This is layered verification, not one final full-suite execution.

The 32 skips remain explicit incomplete qualification: 31 require
administrator-provisioned private direct-I/O scratch in
`PQ_STAGE_B_SPILL_TEST_ROOT`; one requires CUDA's pinned-host allocator.
No substitute scratch was used and no device/spill qualification is inferred.

Failed candidate receipts remain in the audit trail. They exposed an incorrect
expected outer exception in the new malformed-row test, a historical golden
case-ID mismatch, required baseline shrinkage, a missed imported export caller
and a lexical parameter rename. The implementation/tool corrections were
verified by the final retakes and by rerunning the tool on a clean checkout.
The broader run also encountered the already owned PQ237 `CandidateBuild`
caller defect. Exact root fix `d668b0939c` is a composed dependency, not a new
fix in this unit. Root guard fix `c5301f9796` is also a dependency.

All listed actions have actual terminal exit 0 and complete resource cleanup.
Their CAS result bytes were read and size/SHA-256 verified. Failed actions have
no CAS result where PB did not publish one; their canonical stdout/stderr bytes
were read and verified instead. Detailed evidence, including the original REDs,
all candidate failures and exact skipped node reasons, is retained at:

- `/home/rob/tmp/pq-refactor-20261014/reports/cold-row-receipts.json`
- `/home/rob/tmp/pq-refactor-20261014/reports/cold-row-latest-case-evidence.json`
- `/home/rob/tmp/pq-refactor-20261014/reports/cold-row-skips.json`
- `/home/rob/tmp/pq-refactor-20261014/reports/inspect_cold_row_receipts.py`

For a fresh pre-migration checkout, rerun the lever with
`python tools/refactor_joint_row_contract.py --root CHECKOUT`. Submit
`python tools/refactor_joint_row_contract.py --verify-base BASE` and the relevant
tests/compile commands through PrismaBuild. The successful tool-rerun action
used a clean `bbe6de2ff6` checkout, applied the actual lever, verified preserved
operations and ran the cold table, validation-reuse, prefix-intake, frozen
digest and Stage B completion behavior tests.
