# MTP storage admission: CPU regression

Refs #1271. During the current MTP selection audit, per-unit wire/parameter
counts and the byte budget were found to pass through `int()` coercions. The
generic group menu also validated only aggregate counts. A fractional/string
count could be silently changed, and a negative member's bytes could hide
inside a positive group total before the draft budget decision.

Storage metadata is first separated from rung eligibility (refactor
`6b17d5983ee181d5db30ed837053299b50de41b5`). The fix reuses the existing
`schemas.Contract.integer` at both input boundaries: GLM payload wire bytes
and parameter counts are exact positive integers before eligibility filtering;
the GLM byte budget is an exact nonnegative integer before objective/selection
work; the generic group menu requires each member's positive integer parameter
count and nonnegative integer resident bytes. Generic zero residency remains
legal. Current producer output already explicitly writes Python integer counts
(`glm_mtp_quantum.py`), so valid inputs retain their choices and arithmetic.

This is a fix made in the context of the #1271 audit, not a deferred ticket.
It changes malformed-metadata refusal only; it supplies no new MTP prices,
acceptance measurements, objective exchange rate, serving format or kernel.

## Evidence

Baseline RED PB action
`e0ffe88995e9497a6ca193ca5c1fc36cbf8111aa19f41711f83b5a1ba1e246e6`
ran 23 new cases on main `8811ad5f...`: all 23 failed, zero skipped, exit 1.
The affected field/group cases failed to raise; invalid small/negative budgets
received a later no-rung-fits refusal instead of rejecting the metadata, while
a string `1000000000000` budget was accepted. RED has no success receipt.

GREEN `1bd1cc454291e86431b593f7a5b275efb6ae80123b224c4e02d2a26cf4a696b5`
passed the same 23 cases with zero skips, exit 0. CAS result
`1ac8ed8f8a0af4766baf0db6b93589e18c423450d9ebad465c957f2bffb70791`,
7277 bytes; receipt
`7403bc0f1176b0590f450feaa9ba40fda278067651735cd7371ad75bb0890350`.

The related PB fanout ran 11 shards, 121 collected/executed/passed, zero skips
and no missing collection. Those 121 include the same 23, counted once. It
covers generic selection, GLM objective/menu selection, bound priced-wire
intake, declared menus, card rebasing, cost SHA binding, allocator output,
architecture/staleness and duplication checks. Compile action
`f2f0648496b60adb7daa9b509df95686cb85b65db370f924b370fa221ef0deaa`
compiled both touched modules and the new regression file. Its CAS result is
`6d7c04add77eefcc7ab7e77a26fc32fe5bb6c44eed9aee84c2d7cb8f68db25b7`,
712 bytes; receipt
`874f4a07fbfef67cd001dc6ed4f49f4911f6c86ca9729741124ac58b0c8f0dbf`.

Every action ran CPU-only on dl380g10 at priority -10, one CPU, native
threads one and preserved affinity. Test actions reserved 4 GiB; targeted
checks were bounded to 180 seconds and file shards to 300 seconds. Compilation
reserved 1 GiB and hard 60 seconds. The named current interpreter was
`/home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python`: Python 3.14.4,
Torch 2.11.0+cpu, Transformers 5.16.1 and pytest 9.1.1. Published pin preflight
verified PB `95a59051d48cda82eea7927f31870c6c862d7174` and Tessera
`b40c93cb73745097e57a1ba4cf5b9eee166c759a`, including RECORD/import ownership.

| Executable/contract input | SHA-256 |
| --- | --- |
| `prismaquant/glm_mtp_selection.py` | `192b6a93450114e09ce1474c63defb068f9897261825f4f6b9564030f639948a` |
| `prismaquant/mtp_rung_selection.py` | `5dba0bafe0bb0588398fbfca07fe80300b7587ba5aa62010177ccbf26b8effe3` |
| `tests/test_mtp_storage_numbers_1271.py` | `ffe4b4907cf0ba657b704b23662e8bb809f75da94fcc446b2c2f6239a8a31a7c` |
| Same-commit `docs/ARCHITECTURE.md` | `4ec5071fcdd992986c76a4fc1925ad4264cbfa4c37d490854311ce00bec0311a` |

The compile source bundle is
`80cec3432374f8aac7e2929f01b9d0dcb68e3dde256f2326be43c4c29300dd03`,
snapshot `85987e99c5d24985b5a4ba5a2ca676673b6a3447` over the refactor.
Independent bundle readback reproduced all four digests above. Reconciled
populations, raw logs, JUnit and compiled files remain under
`/mnt/shared/tessera-measurements/pq-serving-instrumentation-20261002/`,
with the fanout report `mtp-storage-regressions.json`.

An initial read-only receipt query timed out on two rows; a bounded follow-up
read recovered both actual terminal records and CAS receipts. No action was
rerun or declared passed from that incomplete query. All 12 fanout/compile
receipts are retained in the lane's `mtp-receipts.json` packet.

The #1271 original/current price qualification and matched TP2 acceptance
against BF16 MTP and the rival remain open. This CPU result changes none of
the source, encoded bytes, numerical kernels, runtime pins, format eligibility,
serving gates or production defaults.
