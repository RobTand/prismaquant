# PQ1317 qualification: historical candidate versus live pins

## Immutable evidence, not a qualification of later code

PR #2584's attribution of the retained candidate to main's runtime pin is now
stale. Observed at PrismaQuant revision
`6d8288adc94b3c4b624c143ed372fb6513f32784`, both the development pin (owned by
`prismaquant/tessera_runtime_contract.py`, read by
`tools/resolve_tessera_dev_pin.py`) and serving pin
(`prismaquant/tessera_runtime/tessera_serving_runtime_pin.json`) name Tessera
`aaa417003f0ee690e8ace138acd18df7f9531369`. This is a revision-scoped observation,
not a timeless assertion about main. That later candidate is **NOT qualified by this evidence**.

The historical package is retained in PrismaQuant commit
`4e5380f573dff3319fd687a26b82678eaead7dcd`, under
`docs/measurements/pq1317-gpu-tests/`. In that immutable tree:

- [candidate.json](https://github.com/RobTand/prismaquant/blob/4e5380f573dff3319fd687a26b82678eaead7dcd/docs/measurements/pq1317-gpu-tests/candidate.json)
  binds Tessera `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`, Git tree
  `3b304c6cab442c530cb864e532be2d147571c792`, and contract v60 SHA-256
  `ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e`.
- [results/requalification.json](https://github.com/RobTand/prismaquant/blob/4e5380f573dff3319fd687a26b82678eaead7dcd/docs/measurements/pq1317-gpu-tests/results/requalification.json)
  binds the same candidate and records the four accepted native-v2 launches:
  `run-moe.4yrogwtx`, `run-dense.8iph6q0c`, `run-moe.lze3_4vo`, and
  `run-dense.qg186gcx`. Each of two rounds passed 71 MoE and 65 dense nodes.
  These were four container launches in one operation, not four separate actions.

These records were inspected afresh read-only, not accepted merely on salvage
branch authority. The package is absent from the inspected main tree; this note
neither restores nor edits it. Parent #2547 / PR #2584 and the separate
qualification-package-integration piece retain package ownership; this
correction introduces no dependency on that work.

## Same contract does not mean same measured code

PrismaQuant pin transitions were:

| PrismaQuant commit | Tessera pin introduced |
| --- | --- |
| `45019bc61986b706d7cadb82a6b68c505326bd69` | `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb` |
| `60a38553f12396733f1ba0e1ea4ab2413c9fbd55` | `d2e98b3cb2fcb6fdc27a552442ea021ee02da8ff` |
| `cbb0467821623a5739f8de09c987408d53a94eed` | `aaa417003f0ee690e8ace138acd18df7f9531369` |

The endpoint pin JSON differs only in its commit field. Read-only Tessera object
comparison found identical `src/tessera/serving/runtime_contract.json` bytes
(blob `1ef67e7e5794c2a408af65ed696ec1d253e5b6ab`), but the later source tree is
`cedcc5a47f0442a26115a2f9aae55f3be8c1aa56`, not the historical tree above.
Eight intervening commits change 28 paths. For example,
`src/tessera/serving/dense_ownership.py` moves fused-owner naming and role lookup
to the public `src/tessera/metadata.py`; `src/tessera/projection.py` and
`src/tessera/serving/projection_routes.py` also change. New
`tests/test_producer_ownership_metadata.py` checks ordinary QKV members/roles,
GLM group order, and config precedence over source-name fallback; new
`tests/test_producer_route_metadata.py` and changes to
`tests/test_serving_projection_routes.py` further distinguish the test trees.
This establishes different measured source, not numerical incorrectness of the
later pin. Identical contract metadata/digest cannot transfer qualification.

## Three different operations

1. **Offline checking of retained historical records:** CPU regression/replay can
   check retained bindings, hashes of available records, and internal consistency.
   It neither executes GPU tests nor independently rehashes unavailable external
   source/library bytes. Recorded inventories and external raw-log hashes are not
   a new observation of those bytes.
2. **Possible future v3 execution of the frozen candidate:** running a v3 harness
   against the same `fca4c6ce` candidate would produce separate execution evidence.
   **v3 has CPU regression/replay evidence only**; CPU replay does not establish
   that future GPU execution and the accepted native-v2 runs are not v3 runs.
3. **Qualification after a candidate change:** explicitly versioned candidate/roster
   bindings must identify the new source, contract, harness and case population.
   Analyze affected cases and perform appropriate fresh execution for the claims
   sought. Do not relabel historical receipts or silently retarget frozen records.

At the retained commit, `harness/run_suite.py` reads package-local `candidate.json`
and `roster.json`, checks their candidate agreement, and fetches
`candidate['tessera_commit']`. Its `--source-url` changes origin only, not the
candidate. An alternate origin is not a later-candidate qualification switch.

All evidence remains **component-only**: no whole-model serving qualification,
real distributed TP2, performance claim, or unexercised routed fused-lane claim.
In particular, native-v2 MoE inventories contain no built/mapped native shared
libraries; their accepted empty-inventory gate is not a native-library invocation.
This note documents boundaries; it does not initiate or require fresh GPU work,
change runtime pins, or alter application gates or historical manifests.

## Compare the live checkout without the package

Run this single block from the repository root. It uses only Python standard
library modules and Git's current revision lookup: no application/numerical
imports, historical Git objects, package directory, deployment paths, installs,
network, Docker or GPU. The full historical SHA is an explicit Python argument.
A mismatch in either pin prints exactly `NOT qualified by this evidence`.
Even matching pins alone do not establish new qualification.

```sh
python - fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb <<'PY'
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

historical = sys.argv[1]
if re.fullmatch(r"[0-9a-f]{40}", historical) is None:
    raise SystemExit("historical candidate must be a full lowercase Git SHA")
development = runpy.run_path('tools/resolve_tessera_dev_pin.py')['resolve_tessera_dev_pin']()
serving = json.loads(Path('prismaquant/tessera_runtime/tessera_serving_runtime_pin.json').read_text(encoding='utf-8'))['commit']
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
print('PrismaQuant revision:', revision)
print('Development pin:', development)
print('Serving pin:', serving)
print('Historical candidate:', historical)
if development != historical or serving != historical:
    print('NOT qualified by this evidence')
else:
    print('Pins match historical candidate; this comparison is not new qualification')
PY
```

## CPU validation receipts

Implementer validation extracted and executed the exact fenced command above,
then ran `python -m pytest tests/test_pq1317_qualification_pin_scope.py tests/test_docs_entrypoints.py tests/test_docs_staleness.py tests/test_doc_claims_driver_truth.py -q --basetemp="$TMPDIR/pytest"`:
13 passed, including relocated minimal-input pin mutations. Both live pins printed
`aaa417003f0ee690e8ace138acd18df7f9531369` and the mismatch verdict. The printed
PrismaQuant revision `5a1edaf3c20f5e171982216b214c34134f53bccc` identifies this
runner's sealed snapshot, not a GPU candidate. Retained stdout was verified.
No GPU execution was attempted.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f6d0045789041fe4667ad50977793b433553753e0b6679128e7d0f1d68e46b49","size":538}},"operation":"eb731e74a7a7aa4c57a877c898760f7eca69511955682cf316cd9def8c2e1be2","receipt":"7e84eb92e38008611f3673d23e6b2de8e91358f472a86576f8be8613428e41ea"}]

Research validation ran the portable recipe and the three existing docs files
(`tests/test_docs_entrypoints.py`, `tests/test_docs_staleness.py`, and
`tests/test_doc_claims_driver_truth.py`): 11 tests passed. Its reported PrismaQuant
revision `f697477e65e5795087fae3fbe0b59bbce1ff3d08` is the runner's sealed test
snapshot, not the inspected main revision or retained GPU candidate. Retained
stdout was verified with `read_test_artifact`.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"4a5a2e131f58bba3c7ba1cb67ca07b4777316c58eb56da6c6f5f5d2d1bb452b0","size":545}},"operation":"da96b70aa7e4b08ac1f71c27ff3cfdd48c2a2af668df1d0126f16892f2185634","receipt":"6fc878240127115520e37f819a70bba14567878815c38fba2ae22982091efcf9"}]
