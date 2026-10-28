# Static tautology guard (D-32)

Decision D-32 forbids tautological tests, including historical tests. The
standard-library-only scanner is vendored unchanged as
`prismaquant/tautology.py` from `RobTand/prismagraph`:

- Work: `PG-UPLIFT-001-SB156`.
- Source branch: `pg/PG-UPLIFT-001-SB156/PG-UPLIFT-001-SB156.a1`.
- Source revision: `b905a80d838bb3d743be29889f54e8567bbc12cf`.
- Source path: `prismagraph/tautology.py`.
- Git blob: `3f2c21d0ad75fe7f8c67bf375a857d5b2a552f76`.

The source issue names
`docs/prismagraph/bootstrap/contracts/SB156-static-tautology-guard.md`.
That contract was not available at the source revision or on upstream main
when retrieved on 2026-10-28; it is not reconstructed here. The scanner bytes
and source revision above, rather than an independently rewritten detector,
are the vendoring provenance.

`tests/test_no_tautological_tests.py` scans the entire repository, including
archived test trees, using the upstream filename and test-function discovery
rules. It has no allowlist or suppression. Every finding is reported as
`path:line: test: rule`. Run it with:

```sh
python -m pytest -q tests/test_no_tautological_tests.py tests/test_tautology.py
```

The scanner can also run without importing PrismaQuant or third-party packages:

```sh
python -I -S prismaquant/tautology.py .
```

The three mechanical rules are `self_compare`, `constant`, and `swallowed`.
They are not a semantic proof of test quality: repeated calls may intentionally
observe changing behavior, and the unchanged scanner does not inspect embedded
source strings, non-test functions, or files that fail its parse/read checks.
Passing this guard does not replace independent review or the separate semantic
repair work referenced by the source issue.

The initial repository-wide scan found one historical self-comparison in
`archive/reap_2026-05-15/tests/test_allocator_prune_candidates.py`. Its dense
candidate check now compares the post-call object against the object saved
before the call, so replacing that candidate can fail the assertion. The
existing expert-pruning archive skip and disabled runtime policy are unchanged;
this repair does not reactivate that lane.

Local checks are development evidence only. Integration still requires the
independent review gate and GitHub checks on the integrated SHA in the Mergify
queue; PrismaBuild is not CI.
