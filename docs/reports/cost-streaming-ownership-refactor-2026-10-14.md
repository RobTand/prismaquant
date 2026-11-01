# Streamed cost ownership

The review proposed three owners for `cost_streaming.py`. The implementation
keeps the forward API in that module and moves the persistence and identity
definitions into `streamed_boundary_artifacts.py` and `streamed_model_identity.py`.
The alternative kept persistence in the old module and moved forward calls.
Keeping the forward owner matches the original module's purpose and leaves
model execution callers on their existing entry point.

Boundary window helpers belong with the artifact lifecycle because they borrow,
prefetch and release its references. The identity owner uses the standard library
and shared digest and provenance owners; it has no Torch dependency.

`tools/refactor_cost_streaming.py --base 138069b7993 --write` migrates imports,
module attribute reads, monkeypatch targets and the existing policy/golden
inventories. Without `--write`, it compares every moved definition's AST against
that source commit. It preserves golden values and site counts while moving
their owner names. It leaves no compatibility re-export for a moved operation.

Run `tools/verify_cost_streaming_refactor.py --base 138069b7993` through
PrismaBuild with four CPUs and 16 GiB host memory. It checks the definition
comparison, compiles the project and runs the streamed behavior suites.
`--baseline` runs those same suites on the original source. The caller sets
native math threads to one and retains PrismaBuild's assigned CPU affinity.

Baseline PB `b1204cf45dfd1e59f66c6191ffdd9cfed34f0b136f76aa1ca7adee8a20fda3df`
passed 177 cases with one skip. The first post-move run,
`3e09a73d45fb360053b04b8ca9405047d1c61ed8b855b8b8f26d6ecd2aa4de3c`,
passed the definition and compile checks but failed 16 cases because a test
fixture patched identity functions through the old module. The fixture and
codemod now target the identity owner. A second run,
`bcf02161f50679bfbb2b02f1a0e5ec2b598a556f6b7d52f6b5009a76cdc90d56`,
caught a missing forward import after the window helpers moved. The forward
owner and codemod now derive that dependency from the source definitions.

Final behavior PB
`573919e36fb4cbf1940897f21a890dd72d129a4cbbc4d595cc4cbc22aaff77a3`
passed 177 cases with the same one skip, after proving all 67 moved definitions
unchanged and compiling the project. The skipped CUDA token-device regression
has no CUDA device in this admitted CPU action. The exit was zero, the terminal
record reports complete cleanup, and its published CAS result is
`c0658c4b78f084a3077e5e29cc7cbaae8495ae8dd74b5a95cc0165c73758411b`.

Final structural, metadata and migration checks in PB
`da078b958e27d3b1d8f999b1c9185c5896ae044911bebcfdb155a4d3c7875081`
passed all 189 cases with no skips. This includes the existing dependency,
duplication, seal, I/O and architecture guards, plus regressions for Unicode
source offsets and repeatable policy-inventory migration. Its exit was zero;
the result CAS digest is
`d3cf121c75bc51f182a17e41d4a2b6a8e9911f81bad681a009fb9e221de816f2`.

The tests retain exact tensor equality, resume behavior, partition coverage,
source authentication, stat-reuse refusal and immutable source proofs. Static
thread, seal and duplication inventories remain safety checks. Their owner
paths change without increasing their permitted sites. No numerical method,
format menu, serving admission, calibration draw or performance claim changes.
