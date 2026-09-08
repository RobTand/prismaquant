# Dispatcher partition integration, 2026-09-08

An over-wide row no longer withholds fitting campaign rows. The dispatcher
records every row and its unchanged derived demand, submits the fitting
partition, and retains its refusal to merge a plan with missing rows. PrismaBuild
still owns admission and placement. This integrates PR #407 / issue #391.

The coordinator also found and fixed a failed-replan bug in a separate commit:
`cmd_plan` used to replace selection files before checking fit. Rebundling an
existing plan and then failing the fit check left the old manifest referring
to changed member bytes. A regression on integration `dbbea2bb1c` failed exactly
on those changed selection files. Commit `1d2474beb1` derives all selections and
resource demands before the fit check and publishes selections only after it
passes. This preserves the prior published plan on fit refusal; concurrent
planning and crash-atomic publication are outside this result.

Five independent PB CPU shards passed **70 tests, with one skip** after the fix:
39 fanout, 7 container, 13 architecture, 6 staleness, and 5 stack-driver tests.
The stack-driver module retained its producer-dependent skip; no additional
native qualification is inferred from it. Each shard reserved one CPU/four GiB
with native threads bounded to one on dl380g10, Python 3.12/Torch 2.10 CPU.
The two changed dispatcher/test modules also passed a one-CPU/one-GiB compile.

The failed-fit regression action is
`663258efe046746f5ab57976b1afabe320d994cdac0c1215f48e83c7f54d756d`:
one failed, 38 deselected, exit 1, complete cleanup. Its source snapshot changes
only the regression test above the integration merge. All five successful
CPU outputs and the compile output were independently verified against their
CAS receipts and source bundles. Runtime/test files equal the fixed commit;
the subsequent dispatcher-demand documentation correction changes prose only.
Exact action keys, output tails and source diffs are in the root audit files.

The author demonstrated the original partition defect on x86 and Sparklina CPU:
both negative actions had four failures and 34 passes. Nine successful author
receipts were independently checked, including the stack-driver caller. Every
successful snapshot is closure-only above reviewed author `b29950184e`.
The negative records retain source hashes, actual failure output and cleanup.

Separate incidental prose corrections clarify that resident-model rows price
the full checkpoint, streamed selected rows use their source/capture/encoder
resource plan, and page-advice test assertions establish calls and file
identity rather than physical eviction. No speedup or native fit is claimed.
