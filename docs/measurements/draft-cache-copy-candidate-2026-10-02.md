# Draft cache copy: bounded CPU candidate

Refs #1514. The serving image and general `CacheConfig` implementation are
unchanged. This records a candidate for the existing draft dtype copy, with no
serving or artifact qualification.

The original `eagle/utils.py` was read from a stopped container of
`localhost/prismaquant/spark-vllm-nccl230@sha256:c2e75e03cfc52c15489b40fe58e65acb7347f6fa3ddf2e81afda86760698147b`.
Its SHA-256 is `65d882b8fb476eb0dd6161247346cd7b0392b22832abf36fbb11e795979f9e28`.
The patch refuses any other bytes. It validates the new dtype through the
existing `replace`, then restores the target's `user_specified_block_size`
before nested `VllmConfig` validation. It retains automatic resolved sizes
16/64/256 and explicit constraints without changing backend support tables.

CPU regressions reproduce the pinned Pydantic field/validator shape. RED
replaces only the copy operation with the original `replace(cache_config,
cache_dtype=...)`, leaving the current assertions intact. This is a controlled
field-shape regression, not an import or serve of the original vLLM runtime.

| PB action | Result |
| --- | --- |
| `7e265cd554b223475f46fd8e4f4e34af80fe4448e4c795b3b4b7c32149ddf582` | RED: four expected flag failures, four passes, zero skips, exit 1. |
| `6ccc882e46195ed6d35046d8bfde1bba6169c00e2740d16a8d86bfbd203d3658` | GREEN: eight passes, zero skips, two touched modules compiled, exit 0. |
| `fa5951f82b77fd9a6e6511ab35acffe6babfb511a0a334c64f737c985aa62f64` | Candidate applied to the exact original image file and compiled; exit 0. |

The regression actions used dl380g10, Python 3.14.4, Pydantic 2.13.5,
pytest 9.1.1, CPU-only mode, one CPU/4 GiB per action, native threads one,
PB priority -10 and a 180-second hard bound. Original-image application used
one CPU/1 GiB and a 60-second hard bound. PB preserved assigned affinity.
The GREEN CAS result is `80aa6ee8b97397763f807a1e7798339b0d4be444fab2149719e37936b037c97e`,
4191 bytes; receipt `1865462af186aed322b4a7147f618cf4e8cd7dcdf7f7c038e1cefcfcbcdd17fe`.
The application CAS result is
`116a226383b35dc7beebd5d9a34ac26cea12b4554ea1ec769d9affa35bf6e62b`, 434 bytes;
receipt `2e7711061bb7553979a5baf6cd1c43ce239bde3cdd4fa6b4e4df8a1b19212e5f`.
The actual transformed file hashes to
`7bd432ae331d66cc95ca1ef3687e6b526d99aebb6a0a2dc4f8f9763a70a04e65`.

Commands, source digests, runtime inventories, JUnit, application output and
compiled files remain under
`/mnt/shared/tessera-measurements/pq-serving-instrumentation-20261002/`.
The admitted CPU regression driver is `draft_cpu_driver.py`, SHA-256
`be9ce602809b1ca8f004c2f6b650df73d7e183e022a1277b884bddee8fddda1c`.
The patch file SHA-256 is
`95bc14ee78d2471d3d478af55b1e13839d142ed1dcf86ace236d7399c8628462`;
the test file is `da0d1d6a4b666fd39725bd4bcd34a7e78690fbcb6b09c66350cc698737fb56eb`.

Two earlier actions, `f340d9ff...` and `2cef8f36...`, failed before collection
because the selected minimal interpreter lacked Pydantic. They remain failures
and supply no behavioral RED. A source-application submission was refused
before publication because an external `.py` input was outside the code closure;
the successful action sealed both the application driver and original file in
the owned validation snapshot. Those temporary files are retained as evidence,
not vendored into the delivered repository.

No derived image was built, no runtime/pin/default/gate moved, and no MTP or
EAGLE load, GPU backend selection, acceptance, performance or energy was
measured. The parent issue stays open for actual-image qualification or an
upstream fix.
